import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from qwen3_model import Qwen3WithLinearAttention
from dataset_setup import get_tokenized_dataset, get_data_collator, DATASET_URL
import wandb
from datetime import datetime

class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, alpha=0.5, temperature=4.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
        
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        
        kl_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss
        
        return (loss, student_outputs) if return_outputs else loss

if __name__ == "__main__":
    
    CONFIG_NAME = "top10"
    
    config_path = f"hybrid_model_configs/{CONFIG_NAME}.json"
    teacher_path = "Qwen3-1.7B"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    student_model = Qwen3WithLinearAttention.from_config_json(config_path=config_path)
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_path, torch_dtype=torch.float32)
    teacher_model = teacher_model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(teacher_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset, _ = get_tokenized_dataset(
        dataset_url=DATASET_URL,
        tokenizer=tokenizer,
        max_length=1024,
        streaming=True,
        seed=42,
    )
    data_collator = get_data_collator(tokenizer, mlm=False)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"distill_{CONFIG_NAME}_{timestamp}"
    output_dir = f"distilled_checkpoints/{CONFIG_NAME}/{run_name}"
    
    wandb.init(
        project="compute-aware-arch-search",
        name=run_name,
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        num_train_epochs=1,
        max_steps=1000,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        warmup_steps=100,
        max_grad_norm=1.0,
        fp16=True,
        report_to=["wandb"],
        run_name=run_name,
    )
    
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        alpha=0.5,
        temperature=4.0,
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    
    final_save_dir = os.path.join(output_dir, "final_model")
    student_model.save_pretrained(final_save_dir)
    tokenizer.save_pretrained(final_save_dir)
    print(f"Saved final model to {final_save_dir}")
    
    wandb.finish()

