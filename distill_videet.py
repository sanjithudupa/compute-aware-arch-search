import os
import json
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
        # Track component losses for logging
        self._current_ce_loss = None
        self._current_kl_loss = None
        # Track step count for debugging
        self._step_count = 0
        # Track skipped batches due to NaN
        self._skipped_batches = 0

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
        
        # Force student model to student_device after Trainer init
        if hasattr(self, 'model') and self.model is not None:
            self.model = self.model.to(self.student_device)
            self.model.train()
            for p in self.model.parameters():
                p.requires_grad_(True)

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
        self._step_count += 1
        labels = inputs["labels"]
        student_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        
        # Student forward
        student_outputs = model(**student_inputs)
        student_logits = student_outputs.logits
        
        # Check for NaN in student logits - skip this batch if NaN detected
        if torch.isnan(student_logits).any():
            self._skipped_batches += 1
            print(f"WARNING: NaN detected in student logits at step {self._step_count}. Skipping batch (total skipped: {self._skipped_batches})")
            
            # Return zero loss to skip this batch (produces zero gradients)
            zero_loss = torch.tensor(0.0, device=self.student_device, requires_grad=True)
            if zero_loss.device != self.args.device:
                zero_loss = zero_loss.to(self.args.device)
            
            # Store component losses as zero for logging
            self._current_ce_loss = 0.0
            self._current_kl_loss = 0.0
            
            if return_outputs:
                return zero_loss, student_outputs
            return zero_loss
        
        # Teacher forward
        with torch.inference_mode():
            teacher_inputs = {
                k: v.to(self.teacher_device) if isinstance(v, torch.Tensor) else v
                for k, v in student_inputs.items()
            }
            teacher_outputs = self.teacher_model(**teacher_inputs, use_cache=False)
            teacher_logits = teacher_outputs.logits.to(student_logits.device)
            del teacher_outputs, teacher_inputs
            torch.cuda.empty_cache()
        
        # Cross-entropy loss
        labels = labels.to(student_logits.device)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        
        mask = (labels != -100).unsqueeze(-1)  # [B, T, 1]
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none",
        ).sum(dim=-1)  # [B, T]
        
        kl_loss = (kl_loss * mask.squeeze(-1)).sum() / mask.sum()
        kl_loss = kl_loss * (self.temperature ** 2)
        
        # Store component losses for logging
        self._current_ce_loss = ce_loss.item() if torch.is_tensor(ce_loss) else float(ce_loss)
        self._current_kl_loss = kl_loss.item() if torch.is_tensor(kl_loss) else float(kl_loss)
        
        loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss
        
        # Trainer validates that loss.device == args.device
        if loss.device != self.args.device:
            loss = loss.to(self.args.device)
        
        if return_outputs:
            return loss, student_outputs
        return loss
    
    def log(self, logs, start_time=None):
        """
        Override log to add component losses (ce_loss, kl_loss) to logs.
        This ensures we can track individual loss components in wandb.
        """
        # Add component losses to logs if available
        if hasattr(self, '_current_ce_loss') and self._current_ce_loss is not None:
            logs['ce_loss'] = self._current_ce_loss
        if hasattr(self, '_current_kl_loss') and self._current_kl_loss is not None:
            logs['kl_loss'] = self._current_kl_loss
        
        # Add skipped batch count to logs
        if hasattr(self, '_skipped_batches'):
            logs['skipped_batches'] = self._skipped_batches
        
        # Ensure total loss is in logs (should already be there, but double-check)
        if 'loss' not in logs:
            # This shouldn't happen, but add a warning if it does
            print("WARNING: 'loss' not found in logs!")
        
        # Call parent log method (this will trigger callbacks and send to wandb)
        super().log(logs, start_time)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train student model via distillation")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from (e.g., /path/to/checkpoint-3000 or /path/to/output_dir)"
    )
    args = parser.parse_args()
    
    CONFIG_NAME = "top10_gla"
    
    config_path = f"hybrid_model_configs/{CONFIG_NAME}.json"
    teacher_path = "Qwen3-8B"
    
    # Determine if we're resuming from a checkpoint
    resume_from_checkpoint = None
    output_dir = None
    optimizer_corrupted = False
    
    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        print(f"\n{'='*70}")
        print(f"RESUMING TRAINING FROM CHECKPOINT: {checkpoint_path}")
        print(f"{'='*70}\n")
        
        # Check if the path is a checkpoint directory (checkpoint-XXXX) or output directory
        if os.path.basename(checkpoint_path).startswith("checkpoint-"):
            # It's a checkpoint directory, use its parent as output_dir
            output_dir = os.path.dirname(checkpoint_path)
            resume_from_checkpoint = checkpoint_path
        elif os.path.isdir(checkpoint_path):
            # It's an output directory, find the latest checkpoint
            checkpoint_dirs = [d for d in os.listdir(checkpoint_path) 
                             if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_path, d))]
            if checkpoint_dirs:
                latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[1]))[-1]
                resume_from_checkpoint = os.path.join(checkpoint_path, latest_checkpoint)
                output_dir = checkpoint_path
                print(f"Found latest checkpoint: {resume_from_checkpoint}")
            else:
                raise ValueError(f"No checkpoint directories found in {checkpoint_path}")
        else:
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
        
        # Verify checkpoint exists and has required files
        if not os.path.exists(resume_from_checkpoint):
            raise ValueError(f"Checkpoint directory does not exist: {resume_from_checkpoint}")
        
        checkpoint_files = os.listdir(resume_from_checkpoint)
        has_model = any(f.startswith("model") and f.endswith(".safetensors") for f in checkpoint_files)
        optimizer_path = os.path.join(resume_from_checkpoint, "optimizer.pt")
        has_optimizer = os.path.exists(optimizer_path)
        
        print(f"Checkpoint contents:")
        print(f"  - Model weights: {'✓' if has_model else '✗'}")
        print(f"  - Optimizer state: {'✓' if has_optimizer else '✗'}")
        
        if not has_model:
            raise ValueError(f"Checkpoint missing model weights: {resume_from_checkpoint}")
        
        # Try to validate optimizer state if it exists
        if has_optimizer:
            try:
                # Try to load optimizer state to check if it's corrupted
                optimizer_state = torch.load(optimizer_path, map_location="cpu", weights_only=False)
                if not isinstance(optimizer_state, dict):
                    optimizer_corrupted = True
                    print(f"  ⚠️  Optimizer state file exists but appears corrupted (not a dict)")
                elif "state" not in optimizer_state:
                    optimizer_corrupted = True
                    print(f"  ⚠️  Optimizer state file exists but missing 'state' key")
                else:
                    print(f"  ✓ Optimizer state appears valid")
            except Exception as e:
                optimizer_corrupted = True
                print(f"  ⚠️  Failed to load optimizer state: {e}")
                print(f"  ⚠️  Optimizer will be initialized from scratch")
        
        if not has_optimizer or optimizer_corrupted:
            optimizer_corrupted = True
            print(f"\n{'='*70}")
            print(f"WARNING: Optimizer state is missing or corrupted!")
            print(f"  → Training will resume with model weights from checkpoint")
            print(f"  → Optimizer will be initialized from scratch (momentum/Adam states reset)")
            print(f"  → Learning rate schedule will restart from the current step")
            print(f"{'='*70}\n")
    
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
    
    # Load config to get layer_attention_types
    print(f"\nLoading config from {config_path}...")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    layer_attention_types = config_dict.get('layer_attention_types')
    if layer_attention_types is None:
        raise ValueError("layer_attention_types not found in config file")
    
    print(f"Layer attention types: {layer_attention_types[:5]}... (showing first 5)")
    
    # Load student model and move to student_device BEFORE passing to Trainer
    # This ensures the Trainer doesn't move it to the default GPU 0
    torch.cuda.set_device(student_device)

    if resume_from_checkpoint:
        # When resuming, Trainer will load the model from checkpoint automatically
        # But we still need to initialize the model structure for the Trainer
        # Load config from checkpoint to get layer_attention_types
        checkpoint_config_path = os.path.join(resume_from_checkpoint, "config.json")
        if os.path.exists(checkpoint_config_path):
            with open(checkpoint_config_path, 'r') as f:
                checkpoint_config = json.load(f)
            # Use layer_attention_types from checkpoint config if available
            if 'layer_attention_types' in checkpoint_config:
                print(f"Using layer_attention_types from checkpoint config")
        # Initialize model structure (Trainer will load weights from checkpoint)
        print(f"\nInitializing student model structure (weights will be loaded from checkpoint)...")
        student_model = Qwen3WithLinearAttention.from_config_json(config_path=config_path)
    else:
        # Initialize new model
        student_model = Qwen3WithLinearAttention.from_config_json(config_path=config_path)
    
    # Move model to device and ensure all buffers/parameters are on GPU
    student_model = student_model.to(torch.float32).to(student_device)
    for name, buffer in student_model.named_buffers():
        if buffer.device != student_device:
            buffer.data = buffer.data.to(student_device)
    print(student_model)
    student_model.train()
    for p in student_model.parameters():
        p.requires_grad_(True)
    
    # Load teacher model on teacher device
    print(f"Loading teacher model on {teacher_device}...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_path,
        torch_dtype=torch.float16,
    )
    teacher_model = teacher_model.to(teacher_device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    torch.cuda.empty_cache()
    
    tokenizer = AutoTokenizer.from_pretrained(teacher_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset, _ = get_tokenized_dataset(
        dataset_url=DATASET_URL,
        tokenizer=tokenizer,
        max_length=512,  # Back to 1024 - teacher should fit comfortably in fp16
        streaming=True,
        seed=42,
    )
    data_collator = get_data_collator(tokenizer, mlm=False)
    
    # Set output directory and run name
    if output_dir is None:
        # New training run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"distill_{CONFIG_NAME}_{timestamp}"
        output_dir = f"distilled_checkpoints/{CONFIG_NAME}/{run_name}"
    else:
        # Resuming - extract run name from output_dir
        run_name = os.path.basename(output_dir)
        if not run_name.startswith("distill_"):
            # Fallback if basename doesn't match expected format
            run_name = f"distill_{CONFIG_NAME}_resumed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize wandb (resume if resuming from checkpoint)
    wandb.init(
        project="compute-aware-arch-search",
        name=run_name,
        resume="allow" if resume_from_checkpoint else None,
        id=None,  # Let wandb auto-detect or create new run
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,  # Reduced to 1 to save memory
        gradient_accumulation_steps=8,  # Increased to maintain effective batch size
        learning_rate=5e-5,
        num_train_epochs=1,
        max_steps=10000,
        logging_steps=1,
        save_steps=3000,
        save_total_limit=6,
        warmup_steps=200,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",  # Use cosine annealing LR schedule
        ddp_find_unused_parameters=False,  # Disable unused parameter finding for DDP
        dataloader_pin_memory=False,  # Disable pin_memory to save GPU memory
        dataloader_num_workers=0,  # Disable multiprocessing to save memory
        # fp16=False,  # Use mixed precision - Accelerate handles autocast automatically
        # PRECISION BREAKDOWN with fp16=True:
        bf16=True,
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
    
    try:
        # Resume from checkpoint if specified
        # Note: If optimizer state is corrupted, Trainer will automatically create a new optimizer
        # The model weights and training step will still be loaded from the checkpoint
        if resume_from_checkpoint and optimizer_corrupted:
            print(f"\nResuming training (optimizer will be reinitialized due to corruption)...")
        elif resume_from_checkpoint:
            print(f"\nResuming training from checkpoint...")
        
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        final_save_dir = os.path.join(output_dir, "final_model")
        student_model.save_pretrained(final_save_dir)
        tokenizer.save_pretrained(final_save_dir)
        print(f"Saved final model to {final_save_dir}")
        
    except (ValueError, RuntimeError, Exception) as e:
        print(f"\n{'='*70}")
        print(f"ERROR occurred during training: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"{'='*70}\n")
        
        # Check if any checkpoints were saved before the error
        print("Checking for existing checkpoints...")
        if os.path.exists(output_dir):
            checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
            if checkpoint_dirs:
                latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[1]))[-1]
                latest_checkpoint_path = os.path.join(output_dir, latest_checkpoint)
                print(f"Found checkpoint directory: {latest_checkpoint_path}")
                
                # Check what files exist in the latest checkpoint
                if os.path.exists(latest_checkpoint_path):
                    checkpoint_files = os.listdir(latest_checkpoint_path)
                    model_files = [f for f in checkpoint_files if f.startswith("model") and f.endswith(".safetensors")]
                    optimizer_exists = "optimizer.pt" in checkpoint_files
                    training_state_exists = "training_state.bin" in checkpoint_files
                    
                    print(f"  - Model weight files: {len(model_files)} found")
                    print(f"  - Optimizer state: {'EXISTS' if optimizer_exists else 'MISSING'}")
                    print(f"  - Training state: {'EXISTS' if training_state_exists else 'MISSING'}")
                    
                    if model_files and not optimizer_exists:
                        print(f"\n  ⚠️  WARNING: Model weights saved but optimizer state missing/corrupted.")
                        print(f"  ⚠️  You can resume from this checkpoint, but optimizer state will be reset.")
        
        print("\nAttempting to save recovery checkpoint...")
        recovery_save_dir = os.path.join(output_dir, "recovery_checkpoint")
        os.makedirs(recovery_save_dir, exist_ok=True)
        
        try:
            student_model.save_pretrained(recovery_save_dir)
            tokenizer.save_pretrained(recovery_save_dir)
            print(f"✓ Model saved to recovery checkpoint: {recovery_save_dir}")
        except Exception as save_error:
            print(f"✗ ERROR: Failed to save recovery checkpoint: {save_error}")
            if "No space left on device" in str(save_error):
                print(f"  → Disk is full. Check existing checkpoints in: {output_dir}")
        
        # Re-raise the original error
        raise
    
    finally:
        wandb.finish()

