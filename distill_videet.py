import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from qwen3_model import Qwen3WithLinearAttention
from dataset_setup import get_tokenized_dataset, get_data_collator, DATASET_URL
import wandb
from datetime import datetime

class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, teacher_device, student_device, alpha=0.5, temperature=4.0, *args, **kwargs):
        self.teacher_model = teacher_model.to(teacher_device)
        self.teacher_device = teacher_device
        self.student_device = student_device
        self.alpha = alpha
        self.temperature = temperature

        # Freeze teacher
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad_(False)

        super().__init__(*args, **kwargs)

    def _move_model_to_device(self, model, device):
        """
        Ensure the student model always lives on the requested student_device,
        regardless of what Transformers picks as the default device.
        """
        return model.to(self.student_device)

    def _wrap_model(self, model, training=True, dataloader=None):
        """
        Avoid DataParallel wrapping; just return the bare model on the student device.
        """
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        return model.to(self.student_device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Standard causal LM labels
        labels = inputs["labels"]

        # Student forward pass on student device (Trainer already moved model & inputs)
        student_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        student_outputs = model(**student_inputs)
        # Cast logits to fp32 for numerically stable loss computation
        student_logits = student_outputs.logits.float()  # [B, T, V]

        # Teacher forward pass on teacher device, no gradients
        with torch.no_grad():
            teacher_inputs = {
                k: (v.to(self.teacher_device) if isinstance(v, torch.Tensor) else v)
                for k, v in student_inputs.items()
            }
            teacher_outputs = self.teacher_model(**teacher_inputs)
            teacher_logits = teacher_outputs.logits.to(student_logits.device).float()

        # Cross-entropy loss with ground-truth labels
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        # KL divergence distillation loss between student and teacher logits
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
        ) * (self.temperature ** 2)

        loss = self.alpha * ce_loss + (1.0 - self.alpha) * kl_loss

        return (loss, student_outputs) if return_outputs else loss

if __name__ == "__main__":
    
    CONFIG_NAME = "top10"
    
    config_path = f"hybrid_model_configs/{CONFIG_NAME}.json"
    teacher_path = "Qwen3-1.7B"
    
    # Set up two-GPU configuration: teacher on GPU 0, student on GPU 1
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script. At least 2 GPUs are needed.")
    
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        raise RuntimeError(f"This script requires 2 GPUs, but only {num_gpus} GPU(s) available.")
    
    teacher_device = torch.device("cuda:0")
    student_device = torch.device("cuda:1")
    
    print(f"Teacher model will run on: {teacher_device}")
    print(f"Student model will run on: {student_device}")
    
    # Load student model (will be moved to student_device by the Trainer)
    student_model = Qwen3WithLinearAttention.from_config_json(config_path=config_path)
    # Keep weights in fp16 for memory/speed; losses are computed in fp32 above
    student_model = student_model.to(torch.float16)
    
    # Load teacher model on teacher device in fp16
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_path,
        torch_dtype=torch.float16,
    ).to(teacher_device)
    
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
        per_device_train_batch_size=1,  # Reduced to 1 to save memory
        gradient_accumulation_steps=16,  # Increased to maintain effective batch size
        learning_rate=5e-5,
        num_train_epochs=1,
        max_steps=1000,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        warmup_steps=100,
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False,  # Disable unused parameter finding for DDP
        dataloader_pin_memory=False,  # Disable pin_memory to save GPU memory
        dataloader_num_workers=0,  # Disable multiprocessing to save memory
        fp16=True,
        report_to=["wandb"],
        run_name=run_name,
    )
    
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        teacher_device=teacher_device,
        student_device=student_device,
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

