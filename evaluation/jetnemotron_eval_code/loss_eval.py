import os
import math
from tqdm import tqdm
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ellm.data.data_loader import build_eval_dataloader
from ellm.utils.dist import is_master
from ellm.utils.misc import AverageMeter

from ellm.eval.base import BaseEvaluator


__all__ = ["build_loss_evaluator", "LossEvaluator"]


def build_loss_evaluator(
    data_cfg: dict, eval_data_config: dict, max_seq_len: int, eval_batch_size: int,
) -> "LossEvaluator":
        
    all_evaluator = []
    for name, cfg in eval_data_config.items():
        if cfg is None:
            continue
        eval_dataloader = build_eval_dataloader(data_cfg, cfg, max_seq_len, eval_batch_size)
        all_evaluator.append((name, _MicroLossEvaluator(eval_dataloader)))
    return LossEvaluator(all_evaluator)


class _MicroLossEvaluator:
    def __init__(self, eval_dataloader: DataLoader) -> None:
        self.eval_dataloader = eval_dataloader
        self.loss_fn = nn.CrossEntropyLoss()

    def evaluate(
        self, model: nn.Module, enable_tqdm: bool = False, desc="Eval", 
        forward_kwargs: dict = None, amp_dtype: torch.dtype = torch.bfloat16
    ) -> dict[str, Any]:
        is_training = model.training
        forward_kwargs = forward_kwargs or {}
        model.eval()

        loss_meter = AverageMeter()
        with torch.no_grad():
            with tqdm(total=len(self.eval_dataloader), desc=desc, disable=not (enable_tqdm and is_master())) as t:
                for feed_dict in self.eval_dataloader:
                    input_ids = feed_dict["input_ids"].cuda(non_blocking=True)
                    position_ids = torch.arange(input_ids.shape[1], dtype=torch.long, device=input_ids.device)
                    position_ids = position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)
                    labels = feed_dict.get("labels", input_ids.clone())[..., 1:].cuda(non_blocking=True)
                    if "attention_mask" in feed_dict:
                        attention_mask = feed_dict["attention_mask"].cuda(non_blocking=True)
                    else:
                        attention_mask = None

                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                        logits = model(input_ids, position_ids, attention_mask=attention_mask, return_loss=False, return_logits=True, **forward_kwargs)["logits"]
                    logits_for_loss = logits[..., :-1, :]
                    logits_for_loss = logits_for_loss.contiguous().view(-1, logits_for_loss.size(-1))
                    labels = labels.contiguous().view(-1)
                    loss = self.loss_fn(logits_for_loss, labels).item()

                    input_shape = list(input_ids.size())
                    loss_meter.update(loss, input_shape[0])
                    t.set_postfix(
                        {
                            "input_shape": input_shape,
                            "count": int(loss_meter.get_count()),
                            "loss": float(loss_meter.get_avg()),
                            "ppl": math.exp(min(float(loss_meter.get_avg()), 14)),
                        }
                    )
                    t.update()

        model.train(is_training)
        return {
            "count": int(loss_meter.get_count()),
            "loss": float(loss_meter.get_avg()),
            "ppl": math.exp(min(float(loss_meter.get_avg()), 14)),
        }


class LossEvaluator(BaseEvaluator):
    def __init__(self, all_evaluator: list[tuple[str, _MicroLossEvaluator]]) -> None:
        self.all_evaluator = {name: evaluator for name, evaluator in all_evaluator}

    def evaluate(
        self, model: nn.Module, 
        forward_kwargs: dict = None, 
        enable_tqdm: bool = True, 
        desc: str = "Loss Evaluation",
        amp_dtype: torch.dtype = torch.bfloat16,
        **kwargs
    ) -> dict[str, Any]:
        all_results = {}
        loss_meter = AverageMeter(dist=False)
        for name, evaluator in self.all_evaluator.items():
            all_results[name] = evaluator.evaluate(model, enable_tqdm, desc=f"{desc} {name}",
                                                   forward_kwargs=forward_kwargs, amp_dtype=amp_dtype)
            loss_meter.update(all_results[name]["loss"], all_results[name]["count"])
        return {
            "count": int(loss_meter.get_count()),
            "loss": float(loss_meter.get_avg()),
            "ppl": math.exp(min(float(loss_meter.get_avg()), 14)),
            "verbose": all_results,
        }
