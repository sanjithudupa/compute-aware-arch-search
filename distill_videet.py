import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from qwen3_model import Qwen3WithLinearAttention
from dataset_setup import get_tokenized_dataset, get_data_collator, DATASET_URL
import wandb
from datetime import datetime


def check_and_fix_gla_config(gla_config_path):
    """
    Check GLA config for issues that cause NaN.
    Returns True if config needs fixing.
    """
    if not os.path.exists(gla_config_path):
        print(f"Warning: GLA config file not found: {gla_config_path}")
        return False, {}
    
    with open(gla_config_path, 'r') as f:
        config = json.load(f)
    
    issues = []
    fixes = {}
    
    # Check clamp_min
    if config.get('clamp_min') is None:
        issues.append("clamp_min is None - gates can become too negative, causing exp() underflow")
        fixes['clamp_min'] = -5.0
    
    # Check gate_logit_normalizer
    if 'gate_logit_normalizer' not in config:
        issues.append("gate_logit_normalizer not set - defaulting to 16")
        fixes['gate_logit_normalizer'] = 16
    elif config.get('gate_logit_normalizer', 16) < 8:
        issues.append(f"gate_logit_normalizer={config['gate_logit_normalizer']} is low")
    
    if issues:
        print("\n" + "!"*70)
        print("GLA CONFIG ISSUES FOUND:")
        print("!"*70)
        for issue in issues:
            print(f"  - {issue}")
        
        print("\nSuggested fixes:")
        for k, v in fixes.items():
            print(f"  {k}: {v}")
        
        # Offer to fix
        print("\nTo fix, update your gla_config.json with:")
        fixed_config = {**config, **fixes}
        print(json.dumps(fixed_config, indent=2))
        print("!"*70 + "\n")
        
        return True, fixes
    
    print("GLA config looks OK")
    return False, {}


def debug_gla_forward(model, device, layer_attention_types):
    """
    Debug GLA layers to find NaN source.
    Call this before training starts.
    """
    print("\n" + "="*70)
    print("GLA NaN DIAGNOSTIC")
    print("="*70)
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randint(100, 1000, (1, 128), device=device)
    dummy_mask = torch.ones_like(dummy_input)
    
    with torch.no_grad():
        # Step 1: Check embeddings
        embeds = model.model.embed_tokens(dummy_input)
        print(f"\n[1] Embeddings: shape={embeds.shape}, "
              f"min={embeds.min():.4f}, max={embeds.max():.4f}, "
              f"has_nan={torch.isnan(embeds).any()}")
        
        if torch.isnan(embeds).any():
            print("ERROR: NaN in embeddings!")
            return
        
        # Step 2: Process through layers
        hidden_states = embeds
        
        for layer_idx, layer in enumerate(model.model.layers):
            attn_type = layer_attention_types[layer_idx]
            
            # Apply input layernorm
            normed = layer.input_layernorm(hidden_states)
            
            print(f"\n[Layer {layer_idx}] Type: {attn_type}")
            print(f"  Input: min={hidden_states.min():.4f}, max={hidden_states.max():.4f}")
            print(f"  After LayerNorm: min={normed.min():.4f}, max={normed.max():.4f}")
            
            if attn_type == "gla":
                print(f"\n  === GLA DETAILED DEBUG ===")
                
                # Get the GLA attention module
                gla_block = layer.attention
                gla_attn = gla_block.attn
                
                # Apply block's norm
                normed_for_gla = gla_block.attn_norm(hidden_states)
                print(f"  After GLA attn_norm: min={normed_for_gla.min():.4f}, max={normed_for_gla.max():.4f}")
                
                # Check projections
                q = gla_attn.q_proj(normed_for_gla)
                k = gla_attn.k_proj(normed_for_gla)
                v = gla_attn.v_proj(normed_for_gla)
                
                print(f"  Q: min={q.min():.4f}, max={q.max():.4f}, has_nan={torch.isnan(q).any()}")
                print(f"  K: min={k.min():.4f}, max={k.max():.4f}, has_nan={torch.isnan(k).any()}")
                print(f"  V: min={v.min():.4f}, max={v.max():.4f}, has_nan={torch.isnan(v).any()}")
                
                # CRITICAL: Check gate projection
                gk_raw = gla_attn.gk_proj(normed_for_gla)
                print(f"\n  gk_proj OUTPUT (raw): min={gk_raw.min():.4f}, max={gk_raw.max():.4f}")
                
                if torch.isnan(gk_raw).any():
                    print("  ERROR: NaN in gk_proj output!")
                    # Check the two linear layers in gk_proj
                    for i, m in enumerate(gla_attn.gk_proj):
                        if hasattr(m, 'weight'):
                            w = m.weight
                            print(f"    gk_proj[{i}] weight: min={w.min():.4f}, max={w.max():.4f}, has_nan={torch.isnan(w).any()}")
                            if hasattr(m, 'bias') and m.bias is not None:
                                b = m.bias
                                print(f"    gk_proj[{i}] bias: min={b.min():.4f}, max={b.max():.4f}, has_nan={torch.isnan(b).any()}")
                
                # Check logsigmoid
                gk_logsig = F.logsigmoid(gk_raw)
                print(f"  After logsigmoid: min={gk_logsig.min():.4f}, max={gk_logsig.max():.4f}")
                
                # Check normalization
                normalizer = gla_attn.gate_logit_normalizer
                gk_norm = gk_logsig / normalizer
                print(f"  After /normalizer({normalizer}): min={gk_norm.min():.4f}, max={gk_norm.max():.4f}")
                
                # Check clamp
                clamp_min = gla_attn.clamp_min
                print(f"  clamp_min setting: {clamp_min}")
                if clamp_min is not None:
                    gk_clamped = torch.clamp_min(gk_norm, clamp_min)
                    print(f"  After clamp: min={gk_clamped.min():.4f}, max={gk_clamped.max():.4f}")
                else:
                    gk_clamped = gk_norm
                    print(f"  WARNING: clamp_min is None - values can be arbitrarily negative!")
                
                # Check exp (this is what kernels compute)
                gk_exp = torch.exp(gk_clamped)
                has_inf = torch.isinf(gk_exp).any()
                has_nan = torch.isnan(gk_exp).any()
                print(f"  exp(gk): min={gk_exp.min():.6e}, max={gk_exp.max():.6e}, has_inf={has_inf}, has_nan={has_nan}")
                
                if has_inf or has_nan:
                    print(f"\n  !!! PROBLEM FOUND !!!")
                    print(f"  Gate values too extreme. exp() overflows.")
                    print(f"  Fix: Set clamp_min=-5.0 in GLA config")
                
                # Try running actual forward
                print(f"\n  Running actual GLA forward...")
                try:
                    out, _, _ = gla_attn(
                        normed_for_gla,
                        attention_mask=dummy_mask,
                    )
                    print(f"  GLA output: min={out.min():.4f}, max={out.max():.4f}, has_nan={torch.isnan(out).any()}")
                    
                    if torch.isnan(out).any():
                        print("  ERROR: NaN in GLA attention output!")
                except Exception as e:
                    print(f"  ERROR in GLA forward: {e}")
                    import traceback
                    traceback.print_exc()
                
                print(f"  === END GLA DEBUG ===")
                
                # Don't continue - GLA debugging is what we need
                break
            
            else:
                # For non-GLA layers, just do a simple forward
                # We need position embeddings for full attention
                print(f"  (Skipping full attention layer forward)")
        
    model.train()
    print("\n" + "="*70)
    print("END DIAGNOSTIC")
    print("="*70 + "\n")


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
        labels = inputs["labels"]
        student_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        
        # Student forward
        student_outputs = model(**student_inputs)
        student_logits = student_outputs.logits
        
        # Check for NaN in student logits
        if torch.isnan(student_logits).any():
            print(f"ERROR: NaN detected in student logits!")
            raise ValueError("NaN in student logits")
        
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
        
        # KL divergence with proper masking
        mask = (labels != -100).unsqueeze(-1)  # [B, T, 1]
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none",
        ).sum(dim=-1)  # [B, T]
        
        # Apply mask and compute mean
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
        
        # Ensure total loss is in logs (should already be there, but double-check)
        if 'loss' not in logs:
            # This shouldn't happen, but add a warning if it does
            print("WARNING: 'loss' not found in logs!")
        
        # Call parent log method (this will trigger callbacks and send to wandb)
        super().log(logs, start_time)

if __name__ == "__main__":
    
    CONFIG_NAME = "top10_gla"
    
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
    
    # Load config to get layer_attention_types and check GLA config
    print(f"\nLoading config from {config_path}...")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    layer_attention_types = config_dict.get('layer_attention_types')
    if layer_attention_types is None:
        raise ValueError("layer_attention_types not found in config file")
    
    print(f"Layer attention types: {layer_attention_types[:5]}... (showing first 5)")
    
    # Check GLA config if GLA layers are present
    if "gla" in layer_attention_types:
        gla_config_path = config_dict.get('gla_config_path', 'linear_attn/gla_config.json')
        print(f"\nChecking GLA config at {gla_config_path}...")
        config_needs_fix, fixes = check_and_fix_gla_config(gla_config_path)
        
        if config_needs_fix:
            print("WARNING: GLA config has issues that may cause NaN!")
            print("Consider updating the GLA config file before proceeding.")
    
    # Load student model and move to student_device BEFORE passing to Trainer
    # This ensures the Trainer doesn't move it to the default GPU 0
    torch.cuda.set_device(student_device)

    student_model = Qwen3WithLinearAttention.from_config_json(config_path=config_path)
    
    # Fix GLA config if clamp_min is None (prevents NaN)
    if "gla" in layer_attention_types:
        for layer_idx, layer in enumerate(student_model.model.layers):
            if layer_attention_types[layer_idx] == "gla":
                gla_block = layer.attention
                if hasattr(gla_block, 'attn') and hasattr(gla_block.attn, 'clamp_min'):
                    if gla_block.attn.clamp_min is None:
                        print(f"Setting clamp_min=-5.0 for GLA layer {layer_idx}")
                        gla_block.attn.clamp_min = -5.0
    
    # Move model to device and ensure all buffers/parameters are on GPU
    student_model = student_model.to(torch.float32).to(student_device)
    for name, buffer in student_model.named_buffers():
        if buffer.device != student_device:
            buffer.data = buffer.data.to(student_device)
    
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
        gradient_accumulation_steps=8,  # Increased to maintain effective batch size
        learning_rate=5e-5,
        num_train_epochs=1,
        max_steps=10000,
        logging_steps=1,
        save_steps=2000,
        save_total_limit=2,
        warmup_steps=100,
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False,  # Disable unused parameter finding for DDP
        dataloader_pin_memory=False,  # Disable pin_memory to save GPU memory
        dataloader_num_workers=0,  # Disable multiprocessing to save memory
        # fp16=False,  # Use mixed precision - Accelerate handles autocast automatically
        # PRECISION BREAKDOWN with fp16=True:
        bf16=True,
        # - Model parameters: fp32 (stored in fp32, master weights)
        # - Activations during forward: fp16 (via autocast, saves ~50% memory)
        # - Gradients during backward: fp16 (computed in fp16, saves ~50% memory)
        # - Gradient accumulation: fp32 (accumulated in fp32 for stability)
        # - Optimizer states: fp32 (Adam momentum/variance in fp32)
        # - Loss computation: fp32 (converted via .float() for numerical stability)
        # MEMORY: With gradient checkpointing + fp16 activations, should fit in ~20-25GB
        gradient_checkpointing=False,  # Enable gradient checkpointing to save memory (~30-50% reduction)
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
    
    # Run GLA diagnostic before training (optional, can be disabled)
    if "gla" in layer_attention_types:
        debug_gla_forward(student_model, student_device, layer_attention_types)
    
    trainer.train()
    
    final_save_dir = os.path.join(output_dir, "final_model")
    student_model.save_pretrained(final_save_dir)
    tokenizer.save_pretrained(final_save_dir)
    print(f"Saved final model to {final_save_dir}")
    
    wandb.finish()

