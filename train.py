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
from qwen3_model import Qwen3ForNAS
from fla.layers import MultiScaleRetention
import json
from dataset_setup import get_tokenized_dataset, get_data_collator

class LinearAttentionModel(nn.Module):
    def __init__(self, hidden_size, num_heads, **kwargs):
        super().__init__()
        self.linear_attn = MultiScaleRetention(hidden_size=hidden_size, num_heads=num_heads)
    
    def forward(self, hidden_states):
        return self.linear_attn(hidden_states)


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
config = AutoConfig.from_pretrained(model_path)

# Load attention_hook_idx from config.json
with open(f"{model_path}/config.json", "r") as f:
    config_dict = json.load(f)
    config.attention_hook_idx = config_dict.get("attention_hook_idx", 2)

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

# Training arguments
training_args = TrainingArguments(
    output_dir="./linear_attention_checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=1e-4,
    logging_steps=10,
    save_steps=100,
    evaluation_strategy="no",
    save_total_limit=2,
    remove_unused_columns=False,
)

# Create trainer
trainer = Trainer(
    model=wrapper_model, 
    args=training_args, 
    train_dataset=data,
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()