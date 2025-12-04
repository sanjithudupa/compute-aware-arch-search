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
        # Don't move teacher model if it's already on the correct device (avoids duplication)
        # If using device_map, model is already on device
        if next(teacher_model.parameters()).device != teacher_device:
            self.teacher_model = teacher_model.to(teacher_device)
        else:
            self.teacher_model = teacher_model
        self.teacher_device = teacher_device
        self.student_device = student_device
        self.alpha = alpha
        self.temperature = temperature

        # CRITICAL: Ensure teacher has NO gradient tracking, checkpointing, or wasteful features
        self.teacher_model.eval()  # Eval mode
        for p in self.teacher_model.parameters():
            p.requires_grad_(False)  # No gradients
        
        # Disable gradient checkpointing on teacher (wasteful for eval-only)
        if hasattr(self.teacher_model, 'gradient_checkpointing_disable'):
            self.teacher_model.gradient_checkpointing_disable()
        if hasattr(self.teacher_model, 'model') and hasattr(self.teacher_model.model, 'gradient_checkpointing_disable'):
            self.teacher_model.model.gradient_checkpointing_disable()

        super().__init__(*args, **kwargs)
        
        # CRITICAL: After Trainer init, force student model to student_device
        # The Trainer might have moved it during initialization
        if hasattr(self, 'model') and self.model is not None:
            self.model = self.model.to(self.student_device)
            print(f"After Trainer init: Student model on {next(self.model.parameters()).device}, expected {self.student_device}")

    def _move_model_to_device(self, model, device):
        """
        Ensure the student model always lives on the requested student_device,
        regardless of what Transformers picks as the default device.
        """
        # Force model to student device (ignore the device argument from Trainer)
        model = model.to(self.student_device)
        return model

    def _wrap_model(self, model, training=True, dataloader=None):
        """
        Avoid DataParallel wrapping; just return the bare model on the student device.
        """
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        # Force model to student device
        model = model.to(self.student_device)
        return model
    
    def _prepare_inputs(self, inputs):
        """
        Override to ensure inputs are moved to student_device, not the default device.
        """
        # Move all tensors to student device
        prepared = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                prepared[k] = v.to(self.student_device)
            else:
                prepared[k] = v
        return prepared

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Standard causal LM labels
        labels = inputs["labels"]

        # Student forward pass on student device (Trainer already moved model & inputs)
        # Accelerate's autocast will handle fp16 conversion automatically when fp16=True
        student_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        
        # DEBUG: Check memory before student forward
        if hasattr(self, '_step_count'):
            self._step_count += 1
        else:
            self._step_count = 1
        
        if self._step_count % 10 == 1:  # Print every 10 steps to avoid spam
            print(f"\n[Step {self._step_count}] ========== MEMORY DEBUG ==========")
            print(f"[Step {self._step_count}] Model device: {next(model.parameters()).device}")
            print(f"[Step {self._step_count}] Input device: {next(iter(student_inputs.values())).device if student_inputs else 'N/A'}")
            print(f"[Step {self._step_count}] Teacher model device: {next(self.teacher_model.parameters()).device}")
            print(f"[Step {self._step_count}] GPU 0 memory: {torch.cuda.memory_allocated(self.teacher_device) / 1e9:.2f} GB (reserved: {torch.cuda.memory_reserved(self.teacher_device) / 1e9:.2f} GB)")
            print(f"[Step {self._step_count}] GPU 1 memory: {torch.cuda.memory_allocated(self.student_device) / 1e9:.2f} GB (reserved: {torch.cuda.memory_reserved(self.student_device) / 1e9:.2f} GB)")
        
        student_outputs = model(**student_inputs)
        # Keep logits in float16 to save memory - we'll convert to float32 only when needed for loss
        # For very long sequences (900+ tokens) with large vocab (151k), float32 logits can be 1+ GB
        student_logits = student_outputs.logits  # [B, T, V] - keep in original dtype (float16 with autocast)
        
        # Clear student_outputs immediately to free memory
        del student_outputs
        
        if self._step_count % 10 == 1:
            print(f"[Step {self._step_count}] Student logits device: {student_logits.device}, shape: {student_logits.shape}, dtype: {student_logits.dtype}")
            print(f"[Step {self._step_count}] Student model device: {next(model.parameters()).device}")

        # Teacher forward pass on teacher device, no gradients
        # Use torch.inference_mode() for maximum efficiency (faster than no_grad)
        # CRITICAL: Ensure teacher model is in eval mode and no cache is used
        self.teacher_model.eval()
        
        if self._step_count % 10 == 1:
            print(f"[Step {self._step_count}] GPU 0 memory before teacher forward: {torch.cuda.memory_allocated(self.teacher_device) / 1e9:.2f} GB")
        
        with torch.inference_mode():
            # Move inputs to teacher device
            teacher_inputs = {
                k: (v.to(self.teacher_device) if isinstance(v, torch.Tensor) else v)
                for k, v in student_inputs.items()
            }
            
            if self._step_count % 10 == 1:
                print(f"[Step {self._step_count}] GPU 0 memory after moving inputs: {torch.cuda.memory_allocated(self.teacher_device) / 1e9:.2f} GB")
            
            # Forward pass with use_cache=False to avoid keeping KV cache
            teacher_outputs = self.teacher_model(
                **teacher_inputs,
                use_cache=False,  # CRITICAL: Don't cache KV to save memory
                output_attentions=False,  # Don't output attentions
                output_hidden_states=False,  # Don't output hidden states
            )
            
            if self._step_count % 10 == 1:
                print(f"[Step {self._step_count}] GPU 0 memory after teacher forward: {torch.cuda.memory_allocated(self.teacher_device) / 1e9:.2f} GB")
                print(f"[Step {self._step_count}] Teacher logits shape: {teacher_outputs.logits.shape}, dtype: {teacher_outputs.logits.dtype}")
            
            # Extract logits and move to student device IMMEDIATELY
            # Keep in float16 to match student_logits and save memory
            if self._step_count % 10 == 1:
                print(f"[Step {self._step_count}] Teacher logits device before move: {teacher_outputs.logits.device}")
                print(f"[Step {self._step_count}] Target device (student_logits.device): {student_logits.device}")
            
            # Move to student device and keep in float16 (teacher is already in float16)
            teacher_logits = teacher_outputs.logits.to(student_logits.device)
            
            if self._step_count % 10 == 1:
                print(f"[Step {self._step_count}] GPU 0 memory after moving logits to GPU 1: {torch.cuda.memory_allocated(self.teacher_device) / 1e9:.2f} GB")
                print(f"[Step {self._step_count}] Teacher logits device after move: {teacher_logits.device}, shape: {teacher_logits.shape}")
            
            # Aggressively free teacher GPU memory IMMEDIATELY
            del teacher_outputs
            del teacher_inputs
            
            if self._step_count % 10 == 1:
                print(f"[Step {self._step_count}] GPU 0 memory after del: {torch.cuda.memory_allocated(self.teacher_device) / 1e9:.2f} GB")
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear cache on teacher device specifically
            with torch.cuda.device(self.teacher_device):
                torch.cuda.empty_cache()
            # Also clear general cache
            torch.cuda.empty_cache()
            
            if self._step_count % 10 == 1:
                print(f"[Step {self._step_count}] GPU 0 memory after cache clear: {torch.cuda.memory_allocated(self.teacher_device) / 1e9:.2f} GB (reserved: {torch.cuda.memory_reserved(self.teacher_device) / 1e9:.2f} GB)")

        # Cross-entropy loss with ground-truth labels
        # Convert to float32 only for loss computation to save memory
        ce_loss = F.cross_entropy(
            student_logits.float().view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        # KL divergence distillation loss between student and teacher logits
        # Compute in chunks to save memory for very long sequences
        # For sequences > 500 tokens, chunking is critical to avoid OOM
        if self._step_count % 10 == 1:
            print(f"[Step {self._step_count}] Before log_softmax - GPU 0: {torch.cuda.memory_allocated(self.teacher_device) / 1e9:.2f} GB, GPU 1: {torch.cuda.memory_allocated(self.student_device) / 1e9:.2f} GB")
            print(f"[Step {self._step_count}] student_logits device: {student_logits.device}, teacher_logits device: {teacher_logits.device}")
            print(f"[Step {self._step_count}] Sequence length: {student_logits.shape[1]}, Batch size: {student_logits.shape[0]}")
        
        # Chunk KL divergence computation to avoid OOM with long sequences
        # Process in chunks of 256 tokens at a time
        chunk_size = 256
        seq_len = student_logits.shape[1]
        kl_losses = []
        
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            
            # Extract chunk and convert to float32 only for this computation
            student_chunk = student_logits[:, chunk_start:chunk_end, :].float()
            teacher_chunk = teacher_logits[:, chunk_start:chunk_end, :].float()
            
            # Compute log_softmax and softmax for this chunk
            student_log_probs_chunk = F.log_softmax(student_chunk / self.temperature, dim=-1)
            teacher_probs_chunk = F.softmax(teacher_chunk / self.temperature, dim=-1)
            
            # Compute KL divergence for this chunk
            kl_chunk = F.kl_div(
                student_log_probs_chunk,
                teacher_probs_chunk,
                reduction="none",  # Get per-token loss
            ) * (self.temperature ** 2)
            
            # Sum over vocab dimension, keep batch and sequence dimensions
            kl_chunk = kl_chunk.sum(dim=-1)  # [B, T_chunk]
            kl_losses.append(kl_chunk)
            
            # Clear chunk tensors immediately
            del student_chunk, teacher_chunk, student_log_probs_chunk, teacher_probs_chunk, kl_chunk
        
        # Concatenate all chunks and compute mean
        kl_loss = torch.cat(kl_losses, dim=1).mean()  # Mean over batch and sequence
        
        # Clear intermediate tensors immediately
        del kl_losses
        
        # Clear logits if memory is getting tight (they're no longer needed after loss computation)
        # We can recompute them if needed, but typically we don't need them after loss
        if self._step_count % 10 == 1:
            print(f"[Step {self._step_count}] After KL computation - GPU 1: {torch.cuda.memory_allocated(self.student_device) / 1e9:.2f} GB")
        
        # Optionally clear logits to free memory (they're not needed after loss computation)
        # Uncomment if still running out of memory:
        # del student_logits, teacher_logits
        # torch.cuda.empty_cache()

        loss = self.alpha * ce_loss + (1.0 - self.alpha) * kl_loss

        # CRITICAL: Trainer checks that loss.device == args.device
        # Even though the model is on student_device (cuda:1), Trainer's args.device is cuda:0
        # We need to move the loss to args.device to pass the Trainer's validation check
        # The Trainer will handle moving it back to the correct device for backward pass
        if loss.device != self.args.device:
            loss = loss.to(self.args.device)
        
        if self._step_count % 10 == 1:
            print(f"[Step {self._step_count}] Loss device: {loss.device}, args.device: {self.args.device}, model device: {next(model.parameters()).device}")

        return (loss, student_outputs) if return_outputs else loss

if __name__ == "__main__":
    
    CONFIG_NAME = "top10"
    
    config_path = f"hybrid_model_configs/{CONFIG_NAME}.json"
    teacher_path = "Qwen3-8B"
    
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
    
    # Load student model and move to student_device BEFORE passing to Trainer
    # This ensures the Trainer doesn't move it to the default GPU 0
    student_model = Qwen3WithLinearAttention.from_config_json(config_path=config_path)
    # Keep model in fp32 - Accelerate will handle mixed precision via autocast when fp16=True
    # Don't manually convert to fp16 as it breaks Accelerate's gradient scaler
    student_model = student_model.to(torch.float32)
    # CRITICAL: Move to student_device BEFORE Trainer initialization
    student_model = student_model.to(student_device)
    print(f"Student model moved to {student_device}")
    print(f"Student model device check: {next(student_model.parameters()).device}")
    
    # Don't enable gradient checkpointing here - let TrainingArguments handle it
    # This ensures it's set up correctly with the Trainer's training mode
    # The gradient_checkpointing=True in TrainingArguments will handle it properly
    
    # Load teacher model on teacher device in fp16
    # Teacher should fit comfortably in fp16 - no quantization needed
    print(f"Loading teacher model on {teacher_device}...")
    print(f"GPU 0 memory before loading: {torch.cuda.memory_allocated(teacher_device) / 1e9:.2f} GB")
    
    # Load to CPU first, then move to GPU to avoid duplication issues with device_map
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_path,
        torch_dtype=torch.float16,  # fp16 to save memory
    )
    print(f"GPU 0 memory after from_pretrained (before .to()): {torch.cuda.memory_allocated(teacher_device) / 1e9:.2f} GB")
    
    # Move to device explicitly (don't use device_map as it can cause memory issues)
    teacher_model = teacher_model.to(teacher_device)
    print(f"GPU 0 memory after .to(device): {torch.cuda.memory_allocated(teacher_device) / 1e9:.2f} GB")
    
    # CRITICAL: Disable ALL gradient tracking and checkpointing on teacher
    teacher_model.eval()  # Eval mode - no batch norm updates, no dropout
    for param in teacher_model.parameters():
        param.requires_grad = False  # No gradients needed
    
    # Disable gradient checkpointing on teacher (it's wasteful for eval-only model)
    if hasattr(teacher_model, 'gradient_checkpointing_disable'):
        teacher_model.gradient_checkpointing_disable()
    if hasattr(teacher_model, 'model') and hasattr(teacher_model.model, 'gradient_checkpointing_disable'):
        teacher_model.model.gradient_checkpointing_disable()
    
    # Ensure no autograd tracking
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    # Calculate actual model size
    teacher_param_size = sum(p.numel() * p.element_size() for p in teacher_model.parameters())
    teacher_buffer_size = sum(b.numel() * b.element_size() for b in teacher_model.buffers())
    print(f"Teacher model parameter size: {teacher_param_size / 1e9:.2f} GB")
    print(f"Teacher model buffer size: {teacher_buffer_size / 1e9:.2f} GB")
    print(f"Teacher model total size: {(teacher_param_size + teacher_buffer_size) / 1e9:.2f} GB")
    
    # Clear any cached memory after loading
    torch.cuda.empty_cache()
    print(f"GPU 0 memory after loading (allocated): {torch.cuda.memory_allocated(teacher_device) / 1e9:.2f} GB")
    print(f"GPU 0 memory after loading (reserved): {torch.cuda.memory_reserved(teacher_device) / 1e9:.2f} GB")
    
    # Check for parameter duplication across devices
    teacher_params_on_gpu0 = sum(1 for p in teacher_model.parameters() if p.device == teacher_device)
    teacher_params_on_gpu1 = sum(1 for p in teacher_model.parameters() if p.device == student_device)
    teacher_params_on_cpu = sum(1 for p in teacher_model.parameters() if p.device.type == 'cpu')
    print(f"Teacher parameters: {teacher_params_on_gpu0} on GPU 0, {teacher_params_on_gpu1} on GPU 1, {teacher_params_on_cpu} on CPU")
    
    if teacher_params_on_gpu1 > 0 or teacher_params_on_cpu > 0:
        print("WARNING: Teacher model parameters are on multiple devices! This could cause memory issues.")
    
    tokenizer = AutoTokenizer.from_pretrained(teacher_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset, _ = get_tokenized_dataset(
        dataset_url=DATASET_URL,
        tokenizer=tokenizer,
        max_length=1024,  # Back to 1024 - teacher should fit comfortably in fp16
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
        max_steps=10000,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        warmup_steps=100,
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False,  # Disable unused parameter finding for DDP
        dataloader_pin_memory=False,  # Disable pin_memory to save GPU memory
        dataloader_num_workers=0,  # Disable multiprocessing to save memory
        fp16=True,  # Use mixed precision - Accelerate handles autocast automatically
        # PRECISION BREAKDOWN with fp16=True:
        # - Model parameters: fp32 (stored in fp32, master weights)
        # - Activations during forward: fp16 (via autocast, saves ~50% memory)
        # - Gradients during backward: fp16 (computed in fp16, saves ~50% memory)
        # - Gradient accumulation: fp32 (accumulated in fp32 for stability)
        # - Optimizer states: fp32 (Adam momentum/variance in fp32)
        # - Loss computation: fp32 (converted via .float() for numerical stability)
        # MEMORY: With gradient checkpointing + fp16 activations, should fit in ~20-25GB
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory (~30-50% reduction)
        report_to=["wandb"],
        run_name=run_name,
    )
    
    # CRITICAL: Set current CUDA device to student_device before Trainer init
    # This ensures Accelerate and Trainer use GPU 1, not GPU 0
    torch.cuda.set_device(student_device)
    print(f"Set CUDA current device to: {torch.cuda.current_device()} (should be {student_device.index})")
    
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
    
    # Final verification
    print(f"Final check - Student model device: {next(trainer.model.parameters()).device}")
    print(f"Final check - Trainer args.device: {trainer.args.device} (read-only property)")
    print(f"Final check - CUDA current device: {torch.cuda.current_device()}")
    
    trainer.train()
    
    final_save_dir = os.path.join(output_dir, "final_model")
    student_model.save_pretrained(final_save_dir)
    tokenizer.save_pretrained(final_save_dir)
    print(f"Saved final model to {final_save_dir}")
    
    wandb.finish()

