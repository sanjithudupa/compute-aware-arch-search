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

# OPTION 1: RWKV7
# from fla.models.rwkv7 import RWKV7Config
# from fla.models.rwkv7.modeling_rwkv7 import RWKV7Block

# OPTION 2: GLA (Gated Linear Attention) - ACTIVE
from fla.models.gla import GLAConfig
from fla.models.gla.modeling_gla import GLABlock

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
        # OPTION 1: RWKV7
        # self.decode_block = RWKV7Block(config=config, layer_idx=layer_idx)
        
        # OPTION 2: GLA - ACTIVE
        self.decode_block = GLABlock(config=config, layer_idx=layer_idx)
        
        self.layer_idx = layer_idx
        self.config = config
    
    def forward(self, hidden_states, attention_mask=None):
        # GLA doesn't need v_first - simpler forward pass than RWKV7
        # GLABlock returns (hidden_states, attentions, past_key_values)
        output, _, _ = self.decode_block(
            hidden_states, 
            attention_mask=attention_mask
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

        prev_hidden_bf16 = prev_hidden.to(torch.bfloat16)
        target_hidden_bf16 = target_hidden.to(torch.bfloat16)
        student_output = self.student_model(prev_hidden_bf16, attention_mask=attention_mask)
        student_output_fp32 = student_output.to(torch.float32)
        target_hidden_fp32 = target_hidden_bf16.to(torch.float32)
        prev_hidden_fp32 = prev_hidden_bf16.to(torch.float32)
        
        # MSE loss for backpropagation (training)
        loss = nn.functional.mse_loss(student_output_fp32, target_hidden_fp32)
        
        # Normalized loss: relative to teacher's update magnitude
        # Teacher's update: target_hidden - prev_hidden
        # Normalized by: MSE(prev_hidden, target_hidden) = baseline error
        # This gives relative error: 0.0 = perfect, 1.0 = as bad as input, <1.0 = improvement
        teacher_update_mse = nn.functional.mse_loss(prev_hidden_fp32, target_hidden_fp32)
        # Avoid division by zero
        if teacher_update_mse > 1e-8:
            normalized_loss = loss / teacher_update_mse
        else:
            normalized_loss = loss  # Fallback if teacher update is too small
        
        return {
            "loss": loss,  # Use this for backprop
            "normalized_loss": normalized_loss,  # Use this for reporting/comparison
            "logits": student_output,
            "hidden_states": student_output
        }


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


class CustomTrainer(Trainer):
    """Custom Trainer that ensures normalized_loss gets logged to wandb."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_normalized_loss = None
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to extract normalized_loss and store it for logging.
        """
        outputs = model(**inputs)
        loss = outputs.get("loss")
        normalized_loss = outputs.get("normalized_loss")
        
        # Store normalized_loss as instance variable for logging
        if normalized_loss is not None:
            if torch.is_tensor(normalized_loss):
                normalized_loss = normalized_loss.item()
            self._current_normalized_loss = float(normalized_loss)
        else:
            self._current_normalized_loss = None
        
        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs, start_time=None):
        """
        Override log to add normalized_loss before logging to wandb.
        """
        # Add normalized_loss to logs if available
        if hasattr(self, '_current_normalized_loss') and self._current_normalized_loss is not None:
            logs['normalized_loss'] = self._current_normalized_loss
        
        # Call parent log method (this will trigger callbacks and send to wandb)
        super().log(logs, start_time)


class LossTrackerCallback(TrainerCallback):
    """Callback to track the last N losses for computing rolling average."""
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.recent_losses = []
        self.recent_normalized_losses = []
    
    def on_log(self, args, state, control, model=None, logs=None, trainer=None, **kwargs):
        if logs is not None:
            # Extract normalized_loss from trainer if available and add to logs
            if trainer is not None and hasattr(trainer, '_current_normalized_loss'):
                normalized_loss = trainer._current_normalized_loss
                logs['normalized_loss'] = normalized_loss
            
            # Track losses
            if 'loss' in logs:
                self.recent_losses.append(logs['loss'])
                # Keep only the last window_size losses
                if len(self.recent_losses) > self.window_size:
                    self.recent_losses.pop(0)
            
            # Track normalized losses (either from trainer or already in logs)
            if 'normalized_loss' in logs:
                self.recent_normalized_losses.append(logs['normalized_loss'])
                # Keep only the last window_size normalized losses
                if len(self.recent_normalized_losses) > self.window_size:
                    self.recent_normalized_losses.pop(0)
    
    def get_average_last_n_losses(self):
        """Get the average of the last N losses."""
        if not self.recent_losses:
            return None
        return sum(self.recent_losses) / len(self.recent_losses)
    
    def get_average_last_n_normalized_losses(self):
        """Get the average of the last N normalized losses."""
        if not self.recent_normalized_losses:
            return None
        return sum(self.recent_normalized_losses) / len(self.recent_normalized_losses)


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

# OPTION 1: RWKV7
# with open("linear_attn/rwkv7_config.json", "r") as f:
#     rwkv7_config_dict = json.load(f)

# OPTION 2: GLA - ACTIVE
with open("linear_attn/gla_config.json", "r") as f:
    gla_config_dict = json.load(f)

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

# Resume from layer 6 (layers 1-5 already completed)
for layer_idx in range(6, num_layers + 1):
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

    # OPTION 1: RWKV7
    # linear_attention_config = RWKV7Config(**rwkv7_config_dict)
    
    # OPTION 2: GLA - ACTIVE
    linear_attention_config = GLAConfig(**gla_config_dict)
    
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
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        learning_rate=1e-3,
        lr_scheduler_type="cosine",  # Will be replaced by callback
        warmup_steps=int(max_steps * 0.1),
        max_grad_norm=1.0,  # Enable gradient clipping at norm 1.0
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        remove_unused_columns=False,
        report_to=["wandb"],
        run_name=f"qwen3_linear_attention_student_layer_{layer_idx}",
    )

    # Create loss tracker callback to track last 100 losses
    loss_tracker = LossTrackerCallback(window_size=100)
    
    trainer = CustomTrainer(
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
    
    # Get average normalized loss over last 100 steps (for reporting/comparison across layers)
    avg_last_100_normalized_loss = loss_tracker.get_average_last_n_normalized_losses()
    if avg_last_100_normalized_loss is None:
        # Fallback: if we don't have enough logged steps, compute from final metrics
        # This shouldn't happen if training completed, but handle edge case
        avg_last_100_normalized_loss = 0.0
    
    # Also get regular MSE loss for reference
    avg_last_100_loss = loss_tracker.get_average_last_n_losses()
    if avg_last_100_loss is None:
        # Fallback to overall average if we don't have enough steps
        avg_last_100_loss = float(getattr(train_result, "training_loss", None) or train_result.metrics.get("train_loss", 0.0))
    
    # Use normalized loss for reporting (comparable across layers)
    # Normalized loss = MSE(student, target) / MSE(prev, target)
    # This gives relative error: 0.0 = perfect match, 1.0 = as bad as input, <1.0 = improvement
    final_loss = avg_last_100_normalized_loss if avg_last_100_normalized_loss is not None else avg_last_100_loss

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
