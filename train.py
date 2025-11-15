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
import wandb
from qwen3_model import Qwen3ForNAS
# from fla.layers import MultiScaleRetention
import json
from dataset_setup import get_tokenized_dataset, get_data_collator, DATASET_URL, TOTAL_TOKENS, TOKENS_PER_DATAPOINT

class LinearAttentionModel(nn.Module):
    #temprorary fix until we start on runpod
    def __init__(self, hidden_size, num_heads=8, **kwargs):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, hidden_states):
        # hidden_states shape: [batch, seq_len, hidden_size]
        # nn.MultiheadAttention expects (batch, seq, embed_dim) if batch_first=True
        attn_output, _ = self.self_attn(hidden_states, hidden_states, hidden_states)
        attn_output = self.norm(attn_output + hidden_states)
        return attn_output


class AttentionTeacherForcing(PreTrainedModel):
    def __init__(self, teacher_model, linear_attention, config):
        super().__init__(config)
        self.teacher_model = teacher_model
        self.student_model = linear_attention
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        with torch.no_grad():
            prev_hidden, target_hidden = self.teacher_model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                **kwargs
            )
        student_output = self.student_model(prev_hidden)
        loss = nn.functional.mse_loss(student_output, target_hidden)
        return {"loss": loss, 'logits': student_output, 'hidden_states': student_output}


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

# Initialize Weights & Biases logging
wandb.init(
    project="compute-aware-arch-search",
    name="qwen3_linear_attention_student",
)

# Load config and ensure all required attributes are set
config = AutoConfig.from_pretrained(model_path)

with open(f"{model_path}/config.json", "r") as f:
    config_dict = json.load(f)
    if "attention_hook_idx" in config_dict:
        config.attention_hook_idx = config_dict["attention_hook_idx"]
    else:
        config.attention_hook_idx = 2
    if not hasattr(config, "layer_types") or getattr(config, "layer_types", None) is None:
        use_sliding_window = config_dict.get("use_sliding_window", False)
        if use_sliding_window:

            config.layer_types = ["full_attention"] * config.num_hidden_layers
        else:
            config.layer_types = ["full_attention"] * config.num_hidden_layers

# Load teacher model
teacher_model = Qwen3ForNAS.from_pretrained(
    model_path, 
    config=config, 
    torch_dtype=torch.float32
)

# Create student model (use config.hidden_size, not hardcoded 1024)
student_model = LinearAttentionModel(
    hidden_size=config.hidden_size, 
    num_heads=config.num_attention_heads
)

# Create wrapper model
wrapper_model = AttentionTeacherForcing(
    teacher_model=teacher_model, 
    linear_attention=student_model,
    config=config
)

train_dataset, tokenizer = get_tokenized_dataset(
    dataset_url=DATASET_URL,
    tokenizer=tokenizer,
    max_length=2048,
    streaming=True,
    seed=42,
)

data_collator = get_data_collator(tokenizer, mlm=False)

per_device_batch_size = 2
tokens_per_step = per_device_batch_size * TOKENS_PER_DATAPOINT
max_steps = TOTAL_TOKENS // tokens_per_step

training_args = TrainingArguments(
    output_dir="./linear_attention_checkpoints",
    max_steps=max_steps,
    per_device_train_batch_size=per_device_batch_size,
    per_device_eval_batch_size=2,
    learning_rate=1e-4,
    logging_steps=10,
    save_steps=100,
    eval_strategy="no",
    save_total_limit=2,
    remove_unused_columns=False,
    report_to=["wandb"],
    run_name="qwen3_linear_attention_student",
)

# Create trainer
trainer = Trainer(
    model=wrapper_model,
    args=training_args, 
    train_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()