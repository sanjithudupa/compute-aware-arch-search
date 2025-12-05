import os
import gc
import json
import traceback
from typing import Optional, Union, Dict

import torch
import torch.nn as nn

from ellm.model import AutoCausalModel
from ellm.eval.loss_eval import LossEvaluator
from ellm.eval.lm_harness import LMHarnessEvaluator
from ellm.eval.ruler import RULEREvaluator

from ellm.utils import (
    is_master,
    build_dist_model,
    compiled_sd_to_normal,
    seed_all,
    dist_barrier,
    DistLogger,
    setup_logger,
    setup_wandb,
    get_dist_rank,
    get_dist_size,
    ParallelMesh
)

import logging
logger = logging.getLogger(__name__)
dist_logger = DistLogger(logger)


__all__ = ["MetaEvaluator"]


class MetaEvaluator():
    def __init__(self, model_config: dict, work_dir: str, wandb_cfg: dict, 
                 eval_cfg: dict, dist_config: Optional[dict] = None, 
                 parallel_mesh: Optional[ParallelMesh] = None, resume: bool = False, seed: int = 42):
        self.model_config = model_config
        self.dist_config = dist_config
        self.parallel_mesh = parallel_mesh
        self.evaluators: Dict[str, Union[LossEvaluator, LMHarnessEvaluator, RULEREvaluator]] = {}
        self.work_dir = os.path.realpath(os.path.expanduser(work_dir))
        self.wandb_panel_name = eval_cfg["evaluator"]["wandb_panel_name"]
        self.local_dir_name = eval_cfg["evaluator"]["local_dir_name"]
        self.work_dir = os.path.join(self.work_dir, self.local_dir_name)
        self.wandb_dir = os.path.join(self.work_dir, "wandb")
        self.resume = resume
        self.seed = seed
        self.wandb_global_step_shift = wandb_cfg.get("global_step_shift", 0) # for continual training
        self.wandb_commit_strategy = wandb_cfg.get("commit_strategy", "default")
        self._wandb_cache = {}
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.wandb_dir, exist_ok=True)
        self.device = torch.cuda.current_device()
        setup_logger(
            path=os.path.join(self.work_dir, "log/"),
            name="ellm",
            rank=get_dist_rank(),
            dist_size=get_dist_size(),
            propagate=False
        )
        dist_logger.info(f"### Experiment Begin ###")
        self._setup_evaluate(wandb_cfg, eval_cfg)
    
    def get_model(self, load_path: str, ckpt_type: Optional[str] = None) -> nn.Module:
        model = AutoCausalModel.from_config(self.model_config)
        if load_path != "hf_skip_loading":
            sd = torch.load(load_path, map_location="cpu", weights_only=True)
            if model.model_type == "hf" and ckpt_type == "hf":
                model.base_model.load_state_dict(sd)
            else:
                model.load_state_dict(compiled_sd_to_normal(sd))
        # count model parameters
        dist_logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        # sync_model_gpu(model, self.device)
        if self.dist_config is not None:
            model = build_dist_model(model, self.model_config, self.dist_config, self.parallel_mesh)
        return model
    
    def _setup_evaluate(self, wandb_cfg: dict, eval_cfg: dict):
        self.eval_cfg = eval_cfg
        wandb_cfg["name"] = wandb_cfg["name"] + f"_{self.local_dir_name}"
        self.wandb = setup_wandb(wandb_cfg, self.work_dir, resume=self.resume) if is_master() else None
        dist_logger.info(f"Eval config:\n{json.dumps(eval_cfg, indent=2)}")
        self.ckpt_with_steps = eval_cfg.get("ckpt_with_steps", None)
        self.ckpt_path = eval_cfg.get("ckpt_path", None)
        self.ckpt_type = eval_cfg.get("ckpt_type", None)
        self.ckpt_step = eval_cfg.get("ckpt_step", None)
        if self.ckpt_step is not None and self.ckpt_with_steps is not None: 
            # HACK: enable single ckpt evaluation in multiple ckpt mode
            self.ckpt_path = os.path.join(self.ckpt_path, self.ckpt_with_steps["name_expr"].format(self.ckpt_step))
            self.ckpt_with_steps = None
        self.dummy_steps = eval_cfg.get("dummy_steps", [])

    def add_evaluator(self, evaluator: LossEvaluator | LMHarnessEvaluator, name: str) -> None:
        self.evaluators[name] = evaluator

    def wandb_write_log(self, log_dict: dict, step: Optional[int] = None, commit_strategy: str = "default") -> None:
        if self.wandb is None:
            return
    
        step = step + self.wandb_global_step_shift
        if commit_strategy == "default":
            self.wandb.log(log_dict, step=step)
        elif commit_strategy == "cache":
            self._wandb_cache[step] = {**log_dict, **self._wandb_cache.get(step, {})}
        elif commit_strategy.startswith("offload"):
            self._wandb_cache[step] = {**log_dict, **self._wandb_cache.get(step, {})}
            log_interval = int(commit_strategy.split(":")[-1])
            if len(self._wandb_cache) >= log_interval:
                self.wandb_save_cache()
                self._wandb_cache = {}
        elif commit_strategy == "commit":
            self.wandb.log(log_dict, step=step, commit=True)
        else:
            raise ValueError(f"Invalid commit_strategy: {commit_strategy}")

    def wandb_push_cache(self) -> None:
        if self.wandb is not None and len(self._wandb_cache) > 0:
            for step, log_dict in self._wandb_cache.items():
                self.wandb.log(log_dict, step=step, commit=True)
            self._wandb_cache = {}

    def wandb_save_cache(self) -> None:
        if self.wandb is not None:
            with open(os.path.join(self.work_dir, "wandb_id.txt"), "r") as fout:
                wandb_id = fout.readline().strip()

            with open(os.path.join(self.wandb_dir, f"log_{wandb_id}.jsonl"), "a") as fout:
                wandb_logs = sorted(self._wandb_cache.items(), key=lambda x: x[0])
                for step, log_dict in wandb_logs:
                    fout.write(json.dumps({"step": step, "log": log_dict}) + "\n")

    def load_ckpt(self, path):
        self.model.load_state_dict(torch.load(path), map_location="cpu")
    
    def local_write_log(self, results, step):
        if self.ckpt_with_steps is not None:
            output_dir = os.path.dirname(os.path.join(self.work_dir, self.ckpt_with_steps["name_expr"].format(step)))
        else:
            output_dir = os.path.join(self.work_dir, str(step))
            os.makedirs(output_dir, exist_ok=True)
            
        with open(os.path.join(self.work_dir, "latest.txt"), "w") as f:
            f.write(str(step))
            
        os.makedirs(output_dir, exist_ok=True)
            
        with open(os.path.join(output_dir, f"eval_results.json"), "w") as f:
            json.dump(results, f, indent=4)
        
        with open(os.path.join(self.work_dir, "log.txt"), "a") as f:
            if self.ckpt_with_steps is not None:
                f.write(f"CKPT: {self.ckpt_with_steps['name_expr'].format(step)}\n")
            else:
                f.write(f"CKPT: {self.ckpt_path}, STEP: {step}\n")
            f.write(json.dumps(results, indent=4) + "\n\n")
       
    def _get_latest_step(self):
        if os.path.exists(os.path.join(self.work_dir, "latest.txt")):
            with open(os.path.join(self.work_dir, "latest.txt"), "r") as f:
                step = int(f.read().strip())
            return step
        return None
        
    def evaluate(self):
        assert self.ckpt_path is not None
        if self.ckpt_with_steps is None:
            ckpt_step = 0 if self.ckpt_step is None else int(self.ckpt_step)
            dist_logger.info(f"Evaluating {self.ckpt_path}")
            if self.resume and self._get_latest_step() == ckpt_step:
                dist_logger.info(f"There are already evaluation results in {self.work_dir}. Skip")
            else:
                self.model = self.get_model(self.ckpt_path, self.ckpt_type)
                full_results = {}
                for name in self.evaluators:
                    evaluator = self.evaluators[name]
                    # Allow custom output directory from eval_kwargs, otherwise use default
                    default_save_dir = os.path.join(os.path.join(self.work_dir, os.path.basename(os.path.dirname(self.ckpt_path))))
                    custom_output_dir = self.eval_cfg.get("eval_kwargs", {}).get("output_dir", None)
                    if custom_output_dir:
                        # If output_dir is specified, use it (can be absolute or relative)
                        full_save_dir = os.path.realpath(os.path.expanduser(custom_output_dir))
                        dist_logger.info(f"Using custom output directory: {full_save_dir}")
                    else:
                        full_save_dir = default_save_dir
                    if self.seed is not None:
                        seed_all(self.seed)  # for reproducibility
                        dist_logger.info("Setting seed to {} for reproducibility".format(self.seed))
                    eval_kwargs = {"full_save_dir": full_save_dir, "enable_tqdm": True}
                    # Delegate to diffusion evaluation if configured
                    if isinstance(evaluator, RULEREvaluator) and self.eval_cfg.get("evaluator", {}).get("use_diffusion", False):
                        eval_kwargs.update({
                            "use_diffusion": True,
                            "diffusion_steps": int(self.eval_cfg["evaluator"].get("diffusion_steps", 32)),
                            "logits_temp": float(self.eval_cfg["evaluator"].get("logits_temp", 0.9)),
                            "default_tokens_to_generate": int(self.eval_cfg["evaluator"].get("default_tokens_to_generate", 2048)),
                        })
                    results = evaluator.evaluate(self.model, **eval_kwargs)
                    for key in results:
                        full_results[f"{name}_{key}"] = results[key]
                if is_master():
                    dist_logger.info(f"Evaluation results for {self.ckpt_path}: {json.dumps(full_results, indent=2)}")
                    self.local_write_log(full_results, step=ckpt_step)
                prefix = f"{self.wandb_panel_name}/" if self.wandb_panel_name is not None else ""
                self.wandb_write_log({f"{prefix}{k}": v for k, v in full_results.items()}, step=ckpt_step, commit_strategy=self.wandb_commit_strategy)
        else:
            dist_logger.info(f"Start evaluating from {self.ckpt_with_steps['start']} to {self.ckpt_with_steps['end']} with interval {self.ckpt_with_steps['interval']}")
            start_step = self.ckpt_with_steps["start"]
            if self.resume:
                latest_step = self._get_latest_step()
                if latest_step is not None:
                    start_step = latest_step + self.ckpt_with_steps["interval"]
                    dist_logger.info(f"Resume from {latest_step} to {self.ckpt_with_steps['end']}")
            
            for s in range(start_step, 
                           self.ckpt_with_steps["end"] + 1, 
                           self.ckpt_with_steps["interval"]):
                ckpt_path = os.path.join(self.ckpt_path, self.ckpt_with_steps["name_expr"].format(s))
                dist_logger.info(f"Evaluating: {ckpt_path}")
                try:
                    self.model = self.get_model(ckpt_path, self.ckpt_type)
                except Exception as e:
                    dist_logger.error(traceback.format_exc())
                    dist_logger.error(f"Failed to load ckpt {ckpt_path} because of {e}")
                    continue
                full_results = {}
                for name in self.evaluators:
                    evaluator = self.evaluators[name]
                    # Allow custom output directory from eval_kwargs, otherwise use default
                    default_save_dir = os.path.dirname(os.path.join(self.work_dir, self.ckpt_with_steps["name_expr"].format(s)))
                    custom_output_dir = self.eval_cfg.get("eval_kwargs", {}).get("output_dir", None)
                    if custom_output_dir:
                        # If output_dir is specified, use it (can be absolute or relative)
                        # For multi-step evaluation, append step number if output_dir doesn't contain {step}
                        if "{step}" in custom_output_dir:
                            full_save_dir = os.path.realpath(os.path.expanduser(custom_output_dir.format(step=s)))
                        else:
                            full_save_dir = os.path.realpath(os.path.expanduser(os.path.join(custom_output_dir, str(s))))
                        dist_logger.info(f"Using custom output directory: {full_save_dir}")
                    else:
                        full_save_dir = default_save_dir
                    if self.seed is not None:
                        seed_all(self.seed)  # for reproducibility
                        dist_logger.info("Setting seed to {} for reproducibility".format(self.seed))
                    eval_kwargs = {"full_save_dir": full_save_dir, "enable_tqdm": True, "use_dummy": (s in self.dummy_steps)}
                    if isinstance(evaluator, RULEREvaluator) and self.eval_cfg.get("evaluator", {}).get("use_diffusion", False):
                        eval_kwargs.update({
                            "use_diffusion": True,
                            "diffusion_steps": int(self.eval_cfg["evaluator"].get("diffusion_steps", 32)),
                            "logits_temp": float(self.eval_cfg["evaluator"].get("logits_temp", 0.9)),
                            "default_tokens_to_generate": int(self.eval_cfg["evaluator"].get("default_tokens_to_generate", 2048)),
                        })
                    results = evaluator.evaluate(self.model, **eval_kwargs)
                    for key in results:
                        full_results[f"{name}_{key}"] = results[key]
                
                dist_logger.info(f"Evaluation results for {self.ckpt_with_steps['name_expr'].format(s)}:\n{json.dumps(full_results, indent=2)}")
                if is_master():
                    self.local_write_log(full_results, step=s)
                
                prefix = f"{self.wandb_panel_name}/" if self.wandb_panel_name is not None else ""
                self.wandb_write_log({f"{prefix}{k}": v for k, v in full_results.items()}, step=s, commit_strategy=self.wandb_commit_strategy)
                
                del self.model
                gc.collect()
                torch.cuda.empty_cache()
        
        self.finalize()

    def finalize(self) -> None:
        if is_master():
            self.wandb_save_cache()
            self.wandb_push_cache()
            self.wandb.finish()
        dist_barrier()