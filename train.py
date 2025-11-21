import os
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
    PreTrainedModel,
)
from safetensors.torch import save_file
import wandb
from qwen3_model import Qwen3ForNAS
from fla.models.deltaformer import DeltaFormerConfig
from fla.models.deltaformer.modeling_deltaformer import DeltaFormerBlock
from dataset_setup import (
    get_tokenized_dataset,
    get_data_collator,
    DATASET_URL,
    TOTAL_TOKENS,
    TOKENS_PER_DATAPOINT,
)


class LinearAttentionModel(nn.Module):
    # temporary fix until we start on runpod
    def __init__(self, config, **kwargs):
        super().__init__()
        # DeltaFormerBlock requires layer_idx, use 0 for single block
        self.decode_block = DeltaFormerBlock(config=config, layer_idx=0)

    def forward(self, hidden_states, attention_mask=None):
        output, _, _ = self.decode_block(hidden_states, attention_mask=attention_mask)
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

        # Student (DeltaFormerBlock) gets the same 2D padding mask.
        # Causality (no look-ahead) is enforced inside deltaformer_attn; the mask here just handles padding.
        student_output = self.student_model(prev_hidden_bf16, attention_mask=attention_mask)
        # Convert back to float32 for loss computation to maintain precision
        loss = nn.functional.mse_loss(student_output.to(torch.float32), target_hidden_bf16.to(torch.float32))
        return {"loss": loss, "logits": student_output, "hidden_states": student_output}


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

wandb.init(
    project="compute-aware-arch-search",
    name="qwen3_linear_attention_student",
)

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

with open("linear_attn/deltaformer_config.json", "r") as f:
    deltaformer_config_dict = json.load(f)

train_dataset, tokenizer = get_tokenized_dataset(
    dataset_url=DATASET_URL,
    tokenizer=tokenizer,
    max_length=2048,
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

    llm_config.attention_hook_idx = layer_idx
    if hasattr(teacher_model, "model") and hasattr(teacher_model.model, "current_attention_hook_idx"):
        teacher_model.model.current_attention_hook_idx = layer_idx
    elif hasattr(teacher_model, "current_attention_hook_idx"):
        teacher_model.current_attention_hook_idx = layer_idx

    linear_attention_config = DeltaFormerConfig(**deltaformer_config_dict)
    student_model = LinearAttentionModel(config=linear_attention_config)
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
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=int(max_steps * 0.1),
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        remove_unused_columns=False,
        report_to=["wandb"],
        run_name=f"qwen3_linear_attention_student_layer_{layer_idx}",
    )

    trainer = Trainer(
        model=wrapper_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    train_result = trainer.train()
    final_loss = float(getattr(train_result, "training_loss", None) or train_result.metrics.get("train_loss", 0.0))

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
