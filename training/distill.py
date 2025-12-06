import os
import sys

# Add project root to Python path to allow imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from models.qwen3_model import Qwen3WithLinearAttention
from utils.dataset_setup import get_tokenized_dataset, get_data_collator, DATASET_URL
import wandb
from datetime import datetime

class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, teacher_device, student_device, alpha=0.5, temperature=4.0, *args, **kwargs):
        # Set device attributes BEFORE calling super().__init__() because
        # super().__init__() calls _move_model_to_device which needs student_device
        self.teacher_model = teacher_model
        self.teacher_device = teacher_device
        self.student_device = student_device
        self.alpha = alpha
        self.temperature = temperature
        
        # Now call super().__init__() which will call _move_model_to_device
        super().__init__(*args, **kwargs)
        
        # Set teacher model to eval mode after initialization
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def _move_model_to_device(self, model, device):
        # Override to ensure model goes to student device, not default device
        # student_device should be set before this is called (in __init__)
        if hasattr(self, 'student_device'):
            model = model.to(self.student_device)
        else:
            # Fallback to provided device if student_device not set yet
            model = model.to(device)
        return model
    
    def _wrap_model(self, model, training=True, dataloader=None):
        # Prevent DataParallel wrapping which causes Triton autotuner issues
        # The model should stay on the student device without DataParallel
        if isinstance(model, torch.nn.DataParallel):
            # If somehow wrapped, unwrap it
            model = model.module
        # Ensure model is on student device - move all parameters explicitly
        model = model.to(self.student_device)
        # Double-check all parameters are on the right device
        for param in model.parameters():
            if param.device != self.student_device:
                param.data = param.data.to(self.student_device)
        return model
    
    def _prepare_inputs(self, inputs):
        # Override to ensure inputs are on student device for student model
        # Don't call super() first as it might move inputs to wrong device
        # Instead, manually prepare inputs and move to student device
        if isinstance(inputs, dict):
            prepared = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    prepared[k] = v.to(self.student_device)
                else:
                    prepared[k] = v
        elif isinstance(inputs, (list, tuple)):
            prepared = type(inputs)(v.to(self.student_device) if isinstance(v, torch.Tensor) else v 
                                   for v in inputs)
        else:
            prepared = inputs.to(self.student_device) if isinstance(inputs, torch.Tensor) else inputs
        return prepared
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        
        # Get the device the model is actually on (from Trainer's perspective)
        # This is important because Trainer checks that loss is on the same device as model
        # Don't move the model here - Trainer tracks the original device
        model_device = next(model.parameters()).device
        
        # Create copies of inputs for teacher and student to avoid modifying the original
        # Move inputs to model's device (which should be student_device) and ensure correct dtypes
        # input_ids must stay as Long (int64), attention_mask can be Long or Bool, only float tensors go to fp32
        student_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                # Move to model's device (should be student_device)
                v = v.to(model_device)
                # Only convert float tensors to fp32, keep integer tensors as-is
                if k == "input_ids" or k.endswith("_ids") or (v.dtype in [torch.long, torch.int64, torch.int32, torch.bool]):
                    # Keep integer/bool tensors as-is (input_ids, position_ids, etc.)
                    student_inputs[k] = v
                elif v.dtype.is_floating_point:
                    # Convert floating point tensors to fp32
                    student_inputs[k] = v.to(torch.float32)
                else:
                    # Keep other types as-is
                    student_inputs[k] = v
            else:
                student_inputs[k] = v
        labels_student = labels.to(model_device).to(torch.long) if isinstance(labels, torch.Tensor) else labels
        
        # Student forward pass on student device in fp32
        # Disable autocast to ensure everything stays in fp32
        with torch.amp.autocast('cuda', enabled=False):
            student_outputs = model(**student_inputs)
        student_logits = student_outputs.logits
        
        # Move inputs to teacher device for teacher forward pass
        # Use original inputs (before student device move) to avoid unnecessary copies
        teacher_inputs = {k: v.to(self.teacher_device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
        
        # Teacher forward pass on teacher device
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**teacher_inputs)
            teacher_logits = teacher_outputs.logits
        
        # Immediately move teacher logits to model device and free teacher GPU memory
        # This is critical to avoid OOM - move logits before clearing cache
        teacher_logits_fp32 = teacher_logits.to(model_device).to(torch.float32)
        
        # Clear teacher model's CUDA cache to free memory on teacher GPU
        # This helps prevent OOM when teacher model is large
        del teacher_outputs, teacher_logits
        torch.cuda.empty_cache()  # Clear cache on all devices
        # Specifically clear teacher device cache
        with torch.cuda.device(self.teacher_device):
            torch.cuda.empty_cache()
        student_logits_fp32 = student_logits.to(torch.float32)  # Already fp32, but ensure it
        labels_fp32 = labels_student.to(torch.long)  # Labels should be long, not float
        
        # Compute losses in fp32 (matching train.py's approach of computing losses in fp32)
        ce_loss = F.cross_entropy(
            student_logits_fp32.view(-1, student_logits_fp32.size(-1)),
            labels_fp32.view(-1),
            ignore_index=-100
        )
        
        kl_loss = F.kl_div(
            F.log_softmax(student_logits_fp32 / self.temperature, dim=-1),
            F.softmax(teacher_logits_fp32 / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss
        
        # CRITICAL: Ensure loss is on the same device as the model
        # Trainer checks that loss.device == model.device, so we must match it
        if loss.device != model_device:
            loss = loss.to(model_device)
        
        return (loss, student_outputs) if return_outputs else loss

if __name__ == "__main__":
    
    CONFIG_NAME = "top10"
    
    config_path = f"configs/hybrid_model_configs/{CONFIG_NAME}.json"
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
    
    # Load student model and move to student device
    student_model = Qwen3WithLinearAttention.from_config_json(config_path=config_path)
    # Explicitly move all parts of the model to student device and ensure fp32
    student_model = student_model.to(student_device).to(torch.float32)
    # Verify all parameters are on the correct device and dtype
    for name, param in student_model.named_parameters():
        if param.device != student_device:
            print(f"Warning: {name} is on {param.device}, moving to {student_device}")
            param.data = param.data.to(student_device)
        if param.dtype != torch.float32:
            print(f"Warning: {name} is {param.dtype}, converting to float32")
            param.data = param.data.to(torch.float32)
    # Also move buffers (like running stats in BatchNorm, etc.)
    for name, buffer in student_model.named_buffers():
        if buffer.device != student_device:
            print(f"Warning: buffer {name} is on {buffer.device}, moving to {student_device}")
            buffer.data = buffer.data.to(student_device)
        if buffer.dtype != torch.float32:
            print(f"Warning: buffer {name} is {buffer.dtype}, converting to float32")
            buffer.data = buffer.data.to(torch.float32)
    
    # Load teacher model and move to teacher device
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_path, torch_dtype=torch.float32)
    teacher_model = teacher_model.to(teacher_device)
    
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
    
    # Disable DataParallel by using ddp_find_unused_parameters=False and ensuring
    # we're not using multi-GPU DataParallel. The model is already on the correct device.
    # Use fp32 for everything to avoid dtype mismatches with RWKV7's v_first
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
        # No fp16/bf16 - use fp32 for everything to avoid dtype mismatches
        ddp_find_unused_parameters=False,  # Disable unused parameter finding for DDP
        dataloader_pin_memory=False,  # Disable pin_memory to save GPU memory
        dataloader_num_workers=0,  # Disable multiprocessing to save memory
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
    
    # After Trainer initialization, ensure model is still on student device and in fp32
    # Accelerate might have moved it or changed dtype, so we need to fix it
    if hasattr(trainer, 'model'):
        trainer.model = trainer.model.to(student_device).to(torch.float32)
        # Verify all parameters are on student device and in fp32
        for name, param in trainer.model.named_parameters():
            if param.device != student_device:
                print(f"Fixing device for {name}: {param.device} -> {student_device}")
                param.data = param.data.to(student_device)
            if param.dtype != torch.float32:
                print(f"Fixing dtype for {name}: {param.dtype} -> float32")
                param.data = param.data.to(torch.float32)
        for name, buffer in trainer.model.named_buffers():
            if buffer.device != student_device:
                print(f"Fixing device for buffer {name}: {buffer.device} -> {student_device}")
                buffer.data = buffer.data.to(student_device)
            if buffer.dtype != torch.float32:
                print(f"Fixing dtype for buffer {name}: {buffer.dtype} -> float32")
                buffer.data = buffer.data.to(torch.float32)
    
    trainer.train()
    
    final_save_dir = os.path.join(output_dir, "final_model")
    student_model.save_pretrained(final_save_dir)
    tokenizer.save_pretrained(final_save_dir)
    print(f"Saved final model to {final_save_dir}")
    
    wandb.finish()

