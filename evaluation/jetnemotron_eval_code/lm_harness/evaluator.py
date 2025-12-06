
import os
import json
import inspect
import lm_eval
import pandas as pd

import torch
import torch.nn as nn
import torch.distributed as dist

from time import time
from typing import Optional, Any

from .wrapper import LMEvalWrapper

from ellm.utils import (
    is_master,
    val2tuple,
    DistLogger,
)
from ellm.tokenizer import Tokenizer

from ellm.eval.base import BaseEvaluator


import logging
logger = logging.getLogger(__name__)
dist_logger = DistLogger(logger)


__all__ = ["LMHarnessEvaluator", "build_lm_harness_evaluator"]

pd.set_option("future.no_silent_downcasting", True)


def prepare_for_json(obj):
    if isinstance(obj, dict):
        if any(not isinstance(key, str) for key in obj.keys()):
            return {str(k): prepare_for_json(v) for k, v in obj.items()}
        return {k: prepare_for_json(v) for k, v in obj.items()}
    
    if isinstance(obj, (list, tuple, set)):
        return [prepare_for_json(item) for item in obj]
    
    return obj


def save_to_local(results: dict, save_dir: str, keys_to_remove: list[str] = None) -> None:
    if is_master():
        os.makedirs(save_dir, exist_ok=True)
    dist_logger.info(f"Saving full results to {save_dir}")
    
    keys_to_remove = keys_to_remove or []
    
    if "samples" in results:
        samples = results.pop("samples")
        with open(os.path.join(save_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4, default=str)
        sample_save_dir = os.path.join(save_dir, "samples")
        os.makedirs(sample_save_dir, exist_ok=True)
        for key, value in samples.items():
            with open(os.path.join(sample_save_dir, f"{key}.json"), "w") as f:
                value = prepare_for_json(value)
                for inst in value:
                    for k in keys_to_remove:
                        if k in inst:
                            del inst[k]
                json.dump(value, f, indent=4, default=str)


class LMHarnessEvaluator(BaseEvaluator):
    def __init__(self, 
        wrapper: LMEvalWrapper, 
        tasks: list[tuple[str, str]], 
        subtasks: list[tuple[str, str]] = None,
        cache_requests: bool = True,
        max_test_num: Optional[int] = None,
        apply_chat_template: bool = False,
        system_instruction: Optional[str] = None,
        keys_to_remove_in_save: Optional[str] = None
    ) -> None:

        self.wrapper = wrapper
        self.tasks = tasks
        self.subtasks = subtasks
        self.runtime_cache = {}
        self.cache_requests = cache_requests
        self.max_test_num = max_test_num
        self.apply_chat_template = apply_chat_template
        self.system_instruction = system_instruction
        self.keys_to_remove_in_save = keys_to_remove_in_save
        dist_logger.info(f"Tasks:\n{json.dumps(self.tasks, indent=2)}")
        if self.subtasks is not None:
            dist_logger.info(f"Sub tasks:\n{json.dumps(self.subtasks, indent=2)}")

    def _get_task_metric_score(self, task_name: str, metric_name: str, results: dict) -> float:
        try:
            eval_dict = results[task_name]
        except KeyError:
            raise ValueError(f"Task {task_name} not found in evaluation results, eval_dict keys: {results.keys()}")
        if metric_name not in eval_dict:
            raise ValueError(f"Metric {metric_name} not found in evaluation results for task {task_name}, eval_dict: {eval_dict}")
        return eval_dict[metric_name]

    def _set_dummy_results(self):
        verbose_dict = {}
        subtask_dict = {}
        all_metrics = []

        for task in self.tasks:
            dummy_scores = task.get("dummy_scores", {})
            task_name = task["name"]
            display_name = task.get("display_name", task_name)
            verbose_dict[display_name] = dummy_scores.get("main", 0.0)
            all_metrics.append(verbose_dict[display_name])
            secondary_metric_names = task.get("secondary_metrics", None)
            if secondary_metric_names is not None:
                for secondary_metric_name in val2tuple(secondary_metric_names):
                    secondary_score = dummy_scores.get(secondary_metric_name, 0.0)
                    verbose_dict[f"{display_name}@{secondary_metric_name}"] = secondary_score
        
        if self.subtasks is not None:
            for task in self.subtasks:
                task_name = task["name"]
                display_name = task.get("display_name", task_name)
                subtask_dict[display_name] = dummy_scores.get("main", 0.0)
                secondary_metric_names = task.get("secondary_metrics", None)
                if secondary_metric_names is not None:
                    for secondary_metric_name in val2tuple(secondary_metric_names):
                        secondary_score = dummy_scores.get(secondary_metric_name, 0.0)
                        subtask_dict[f"{display_name}@{secondary_metric_name}"] = secondary_score

        return verbose_dict, subtask_dict, all_metrics

    def _get_invalid_ratio(self, samples: list[dict], invalid_key: str = "[invalid]") -> int:        
        invalid_count = 0
        for sample in samples:
            if sample["filtered_resps"][0] == invalid_key:
                invalid_count += 1
        return invalid_count / len(samples)

    def evaluate(
        self, model: Optional[nn.Module], 
        forward_kwargs: dict = None, 
        full_save_dir: Optional[str] = None, 
        verbose: bool = True, 
        use_dummy: bool = False,
        **kwargs
    ) -> dict[str, Any]:
        if model is not None:
            self.wrapper.model = model
        
        if forward_kwargs is not None:
            self.wrapper.forward_kwargs = forward_kwargs
        
        is_training = self.wrapper.model.training
        self.wrapper.model.eval()

        if use_dummy:
            dist_logger.info("Using dummy results")
            verbose_dict, subtask_dict, all_metrics = self._set_dummy_results()
            return {
                "avg": sum(all_metrics) / len(all_metrics),
                "verbose": verbose_dict,
                "subtasks": subtask_dict
            }

        verbose_dict = {}
        subtask_dict = {}
        all_metrics = []

        with torch.no_grad():
            st = time()
            torch.cuda.synchronize()
            # Build kwargs for simple_evaluate, conditionally including return_runtime_cache
            eval_kwargs = {
                "model": self.wrapper,
                "tasks": [t["name"] for t in self.tasks],
                "verbosity": "ERROR",
                "bootstrap_iters": 0,
                "cache_requests": self.cache_requests,
                "limit": self.max_test_num,
                "log_samples": verbose,
                "confirm_run_unsafe_code": True,
                "apply_chat_template": self.apply_chat_template,
                "system_instruction": self.system_instruction,
            }
            # Only add return_runtime_cache if the wrapper supports it and lm_eval version supports it
            if self.wrapper.use_runtime_cache:
                # Check if lm_eval version supports return_runtime_cache
                if "return_runtime_cache" in inspect.signature(lm_eval.simple_evaluate).parameters:
                    eval_kwargs["return_runtime_cache"] = True
                eval_kwargs.update(self.runtime_cache)
            else:
                eval_kwargs.update(self.runtime_cache)
            
            out = lm_eval.simple_evaluate(**eval_kwargs)
            torch.cuda.synchronize()
            et = time()
            dist_logger.debug(f"lm-eval took {et - st:.2f}s")
            if self.wrapper.use_runtime_cache and isinstance(out, tuple):
                results, self.runtime_cache = out
            else:
                results = out

        if is_master():
            full_results = results
            results = results["results"]
            for task in self.tasks:
                task_name = task["name"]
                metric_name = task["metric"]
                display_name = task.get("display_name", task_name)
                verbose_dict[display_name] = self._get_task_metric_score(task_name, metric_name, results)
                all_metrics.append(verbose_dict[display_name])
                secondary_metric_names = task.get("secondary_metrics", None)
                if secondary_metric_names is not None:
                    for secondary_metric_name in val2tuple(secondary_metric_names):
                        if secondary_metric_name == "invalid_ratio":
                            invalid_ratio = 1.0
                            if "samples" in full_results:
                                invalid_ratio = self._get_invalid_ratio(full_results["samples"][task_name], invalid_key=task.get("invalid_key", "[invalid]"))
                            verbose_dict[f"{display_name}@{secondary_metric_name}"] = invalid_ratio
                        else:
                            secondary_score = self._get_task_metric_score(task_name, secondary_metric_name, results)
                            verbose_dict[f"{display_name}@{secondary_metric_name}"] = secondary_score
            
            if self.subtasks is not None:
                for task in self.subtasks:
                    task_name = task["name"]
                    metric_name = task["metric"]
                    display_name = task.get("display_name", task_name)
                    subtask_dict[display_name] = self._get_task_metric_score(task_name, metric_name, results)
                    secondary_metric_names = task.get("secondary_metrics", None)
                    if secondary_metric_names is not None:
                        for secondary_metric_name in val2tuple(secondary_metric_names):
                            secondary_score = self._get_task_metric_score(task_name, secondary_metric_name, results)
                            subtask_dict[f"{display_name}@{secondary_metric_name}"] = secondary_score

            if verbose:
                if full_save_dir is not None:
                    save_to_local(full_results, full_save_dir, keys_to_remove=self.keys_to_remove_in_save)
            
            out = {
                "avg": sum(all_metrics) / len(all_metrics),
                "verbose": verbose_dict,
                "subtasks": subtask_dict
            }
        else:
            out = None

        out = [out]
        dist.broadcast_object_list(out, src=0)
        out = out[0]

        self.wrapper.model.train(is_training)

        return out


def build_lm_harness_evaluator(
    tokenizer: Tokenizer,
    evaluator_cfg: dict, 
    max_seq_len: int,
    max_new_tokens: int,
    eval_batch_size: int,
    amp_dtype: torch.dtype,
    device: torch.device,
    use_runtime_cache: bool = True,
    cache_requests: bool = True,
    max_test_num: Optional[int] = None,
    additional_until_tokens: Optional[int] = None,
    add_prefix: bool = False,
    generation_num_chunks: int = 1,
    prefill_chunk_size: Optional[int] = None,) -> LMHarnessEvaluator:
    """
    Build the LM Harness evaluator based on the provided configuration and tokenizer.
    
    Args:
        config (dict): Configuration dictionary for the evaluator.
        tokenizer (Tokenizer): Tokenizer instance to be used by the evaluator.
    
    Returns:
        LMHarnessEvaluator: An instance of the LM Harness evaluator.
    """
    eval_wrapper = LMEvalWrapper(None, tokenizer, max_seq_len=max_seq_len,
                                    max_new_tokens=max_new_tokens,
                                    batch_size=eval_batch_size,
                                    additional_until_tokens=additional_until_tokens,
                                    add_prefix=add_prefix,
                                    generation_num_chunks=generation_num_chunks,
                                    prefill_chunk_size=prefill_chunk_size,
                                    amp_dtype=amp_dtype,
                                    device=device,
                                    use_runtime_cache=use_runtime_cache,
                                    eval_kwargs=evaluator_cfg.get("eval_kwargs", {}))

    evaluator = LMHarnessEvaluator(eval_wrapper, 
                                    evaluator_cfg["tasks"], 
                                    evaluator_cfg.get("subtasks", None),
                                    cache_requests=cache_requests,
                                    max_test_num=max_test_num,
                                    apply_chat_template=evaluator_cfg.get("apply_chat_template", False),
                                    system_instruction=evaluator_cfg.get("system_instruction", None),
                                    keys_to_remove_in_save=evaluator_cfg.get("keys_to_remove_in_save", None))

    return evaluator