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
from fla.models.deltaformer import DeltaFormerConfig
from fla.models.deltaformer.modeling_deltaformer import DeltaFormerBlock
import json
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
        output, _ = self.decode_block(hidden_states, attention_mask=attention_mask)
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

        # Student (DeltaFormerBlock) gets the same 2D padding mask.
        # Causality (no look-ahead) is enforced inside deltaformer_attn; the mask here just handles padding.
        student_output = self.student_model(prev_hidden, attention_mask=attention_mask)
        loss = nn.functional.mse_loss(student_output, target_hidden)
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

# Initialize Weights & Biases logging
wandb.init(
    project="compute-aware-arch-search",
    name="qwen3_linear_attention_student",
)

# Load config and ensure all required attributes are set
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

# Load teacher model
teacher_model = Qwen3ForNAS.from_pretrained(
    model_path, 
    config=llm_config, 
    torch_dtype=torch.float32
)

# Load DeltaFormer config for student model
with open("linear_attn/deltaformer_config.json", "r") as f:
    deltaformer_config_dict = json.load(f)
linear_attention_config = DeltaFormerConfig(**deltaformer_config_dict)

# Create student model
student_model = LinearAttentionModel(config=linear_attention_config)

# Create wrapper model
wrapper_model = AttentionTeacherForcing(
    teacher_model=teacher_model, 
    linear_attention=student_model,
    config=llm_config
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