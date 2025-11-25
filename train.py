import os
import json
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    TrainerCallback,
)
from safetensors.torch import save_file
import wandb
from qwen3_model import Qwen3ForNAS
from fla.models.rwkv7 import RWKV7Config
from fla.models.rwkv7.modeling_rwkv7 import RWKV7Block
from dataset_setup import (
    get_tokenized_dataset,
    get_data_collator,
    DATASET_URL,
    TOTAL_TOKENS,
    TOKENS_PER_DATAPOINT,
)


class LinearAttentionModel(nn.Module):
    def __init__(self, config, layer_idx=0, **kwargs):
        super().__init__()
        self.decode_block = RWKV7Block(config=config, layer_idx=layer_idx)
        self.layer_idx = layer_idx
        self.config = config
    
    def forward(self, hidden_states, attention_mask=None):
        # For layer_idx != 0, RWKV7 expects v_first from layer 0
        # Since we're training independently, create a dummy v_first with the right shape
        v_first = None
        if self.layer_idx != 0:
            # v_first shape: [batch_size, seq_len, value_dim]
            # value_dim defaults to hidden_size if not specified
            batch_size, seq_len, _ = hidden_states.shape
            value_dim = self.config.value_dim[self.layer_idx] if isinstance(self.config.value_dim, list) else (self.config.value_dim or self.config.hidden_size)
            v_first = torch.zeros(batch_size, seq_len, value_dim, dtype=hidden_states.dtype, device=hidden_states.device)
        
        # RWKV7Block returns (hidden_states, attentions, past_key_values, v_first)
        output, _, _, _ = self.decode_block(
            hidden_states, 
            attention_mask=attention_mask,
            v_first=v_first
        )
        return output


class AttentionTeacherForcing(PreTrainedModel):
    def __init__(self, teacher_model, linear_attention, config):
        super().__init__(config)
        self.teacher_model = teacher_model
        self.student_model = linear_attention
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Teacher (Qwen) handles its own causal mask internally; we don't need to supply one.
        with torch.no_grad():
            prev_hidden, target_hidden = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )

        # FlashAttention only supports fp16/bf16 – use bfloat16 for better numerical stability
        # Convert hidden states to bfloat16 for FlashAttention compatibility
        prev_hidden_bf16 = prev_hidden.to(torch.bfloat16)
        target_hidden_bf16 = target_hidden.to(torch.bfloat16)

        # Student (RWKV7Block) gets the same 2D padding mask.
        # Causality (no look-ahead) is enforced inside rwkv7 attention; the mask here just handles padding.
        student_output = self.student_model(prev_hidden_bf16, attention_mask=attention_mask)
        # Convert back to float32 for loss computation to maintain precision
        loss = nn.functional.mse_loss(student_output.to(torch.float32), target_hidden_bf16.to(torch.float32))
        return {"loss": loss, "logits": student_output, "hidden_states": student_output}


class CosineAnnealingWithMinLRCallback(TrainerCallback):
    """Callback to replace the default cosine scheduler with one that has a minimum learning rate."""
    def __init__(self, min_lr=1e-5):
        self.min_lr = min_lr
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        trainer = kwargs.get('trainer')
        if trainer is None:
            return
        
        # Get the optimizer and current scheduler
        optimizer = trainer.optimizer
        num_warmup_steps = args.warmup_steps
        num_training_steps = args.max_steps
        initial_lr = args.learning_rate
        
        # Create a custom lambda function for cosine annealing with min LR
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            if progress >= 1.0:
                return self.min_lr / initial_lr
            
            # Cosine annealing from 1.0 to min_lr/initial_lr
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_lr_ratio = self.min_lr / initial_lr
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor
        
        # Replace the scheduler
        trainer.lr_scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)


class LossTrackerCallback(TrainerCallback):
    """Callback to track the last N losses for computing rolling average."""
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.recent_losses = []
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs is not None and 'loss' in logs:
            self.recent_losses.append(logs['loss'])
            # Keep only the last window_size losses
            if len(self.recent_losses) > self.window_size:
                self.recent_losses.pop(0)
    
    def get_average_last_n_losses(self):
        """Get the average of the last N losses."""
        if not self.recent_losses:
            return None
        return sum(self.recent_losses) / len(self.recent_losses)


class HiddenStateDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
        }


# Setup
model_path = "Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_path)

llm_config = AutoConfig.from_pretrained(model_path)

with open(f"{model_path}/config.json", "r") as f:
    llm_config_dict = json.load(f)
    # Check for both possible key names
    if "attention_hook_idx" in llm_config_dict:
        llm_config.attention_hook_idx = llm_config_dict["attention_hook_idx"]
    elif "current_attention_hook_idx" in llm_config_dict:
        llm_config.attention_hook_idx = llm_config_dict["current_attention_hook_idx"]
    else:
        llm_config.attention_hook_idx = 2
    if not hasattr(llm_config, "layer_types") or getattr(llm_config, "layer_types", None) is None:
        use_sliding_window = llm_config_dict.get("use_sliding_window", False)
        if use_sliding_window:
            llm_config.layer_types = ["full_attention"] * llm_config.num_hidden_layers
        else:
            llm_config.layer_types = ["full_attention"] * llm_config.num_hidden_layers
teacher_model = Qwen3ForNAS.from_pretrained(
    model_path, 
    config=llm_config, 
    torch_dtype=torch.float32
)

with open("linear_attn/rwkv7_config.json", "r") as f:
    rwkv7_config_dict = json.load(f)

train_dataset, tokenizer = get_tokenized_dataset(
    dataset_url=DATASET_URL,
    tokenizer=tokenizer,
    max_length=1024,
    streaming=True,
    seed=42,
)

data_collator = get_data_collator(tokenizer, mlm=False)

per_device_batch_size = 32
tokens_per_step = per_device_batch_size * TOKENS_PER_DATAPOINT
max_steps = TOTAL_TOKENS // tokens_per_step

results_path = "layer_results.csv"
if not os.path.exists(results_path):
    with open(results_path, "w") as f:
        f.write("layer_idx,train_loss\n")

num_layers = llm_config.num_hidden_layers

for layer_idx in range(1, num_layers + 1):
    print(f"\n===== Training layer {layer_idx}/{num_layers} =====")
    
    # Initialize wandb for each layer separately to get separate plots
    wandb.init(
        project="compute-aware-arch-search",
        name=f"qwen3_linear_attention_student_layer_{layer_idx}",
        reinit=True,  # Allow reinitialization for each layer
    )

    llm_config.attention_hook_idx = layer_idx
    if hasattr(teacher_model, "model") and hasattr(teacher_model.model, "current_attention_hook_idx"):
        teacher_model.model.current_attention_hook_idx = layer_idx
    elif hasattr(teacher_model, "current_attention_hook_idx"):
        teacher_model.current_attention_hook_idx = layer_idx

    linear_attention_config = RWKV7Config(**rwkv7_config_dict)
    student_model = LinearAttentionModel(config=linear_attention_config, layer_idx=layer_idx)
    student_model = student_model.to(torch.bfloat16)

    wrapper_model = AttentionTeacherForcing(
        teacher_model=teacher_model,
        linear_attention=student_model,
        config=llm_config,
    )

    layer_output_dir = os.path.join("linear_attention_checkpoints", f"layer_{layer_idx}")
    training_args = TrainingArguments(
        output_dir=layer_output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=1e-3,
        lr_scheduler_type="cosine",  # Will be replaced by callback
        warmup_steps=int(max_steps * 0.1),
        max_grad_norm=1.0,  # Enable g sradient clipping at norm 1.0
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        remove_unused_columns=False,
        report_to=["wandb"],
        run_name=f"qwen3_linear_attention_student_layer_{layer_idx}",
    )

    # Create loss tracker callback to track last 100 losses
    loss_tracker = LossTrackerCallback(window_size=100)
    
    trainer = Trainer(
        model=wrapper_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[
            CosineAnnealingWithMinLRCallback(min_lr=1e-5),  # Cosine annealing from 1e-3 to 1e-5
            loss_tracker,  # Track losses for rolling average
        ],
    )

    train_result = trainer.train()
    
    # Get average loss over last 100 steps
    avg_last_100_loss = loss_tracker.get_average_last_n_losses()
    if avg_last_100_loss is None:
        # Fallback to overall average if we don't have enough steps
        avg_last_100_loss = float(getattr(train_result, "training_loss", None) or train_result.metrics.get("train_loss", 0.0))
    
    final_loss = avg_last_100_loss

    safetensors_dir = os.path.join("linear_attention_checkpoints", "safetensors")
    os.makedirs(safetensors_dir, exist_ok=True)
    final_weights_path = os.path.join(safetensors_dir, f"student_layer_{layer_idx}.safetensors")
    state_dict_fp32 = {
        k: v.to(torch.float32) if v.dtype == torch.bfloat16 else v
        for k, v in student_model.state_dict().items()
    }
    save_file(state_dict_fp32, final_weights_path)
    print(f"Saved final safetensors checkpoint for layer {layer_idx} to {final_weights_path}")

    # Append final loss for this layer to results file
    with open(results_path, "a") as f:
        f.write(f"{layer_idx},{final_loss}\n")
    
    # Finish wandb run for this layer
    wandb.finish()
