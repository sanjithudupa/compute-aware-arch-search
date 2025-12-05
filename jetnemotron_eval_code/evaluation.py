import os
import argparse

import torch

from ellm.tokenizer import Tokenizer

from ellm.eval.meta_evaluator import MetaEvaluator
from ellm.eval.loss_eval import build_loss_evaluator
from ellm.eval.lm_harness import build_lm_harness_evaluator
from ellm.eval.ruler import build_ruler_evaluator
from ellm.eval.prompt.evaluator import build_prompt_evaluator

from ellm.utils.parallel_mesh import ParallelMesh
from ellm.utils.dist import dist_init, get_dist_size, dist_close
from ellm.utils.misc import seed_all, get_amp_dtype
from ellm.utils.config import (
    build_config,
    dump_config,
    parse_unknown_args,
    partial_update_config,
    resolve_and_load_config
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", metavar="FILE", help="config file", default="config/debug.yaml")
    parser.add_argument("--load_config", action="store_true")
    
    # enable code eval
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    
    # parse args
    args, opt = parser.parse_known_args()
    opt = parse_unknown_args(opt)

    ### build config ###
    config = resolve_and_load_config(args.config)
    ### update config via args ###
    config = partial_update_config(config, opt)
    config = build_config(config, recursive=True)

    # setup dist env
    dist_init(gpu=config["gpu"], cudnn_benchmark=config["cudnn"])
    
    device = torch.cuda.current_device()

    parallel_mesh = ParallelMesh(
        dp=config["dist_training"]["dp_degree"],
        sp=config["dist_training"]["sp_degree"],
        world_size=get_dist_size(),
    )
    device_mesh = parallel_mesh.build_device_mesh(device_type="cuda")

    ### setup random seed ###
    seed_all(config["seed"])

    evaluator_cfg = build_config(config["offline_eval"]["evaluator"])
    config["offline_eval"]["evaluator"] = evaluator_cfg
    
    meta_evaluator = MetaEvaluator(
        config["model"], config["work_dir"], config["wandb"], config["offline_eval"],
        dist_config=config["dist_training"], parallel_mesh=device_mesh, resume=config["resume"], seed=config["seed"])
    
    if evaluator_cfg["evaluator"] == "loss":
        evaluator = build_loss_evaluator(
            data_cfg=config["data"],
            eval_data_config=evaluator_cfg["data"],
            max_seq_len=config["model"]["max_seq_len"],
            eval_batch_size=config["eval_batch_size"],  
        )
        meta_evaluator.add_evaluator(evaluator, evaluator_cfg["evaluator"])
    elif evaluator_cfg["evaluator"] == "lm_harness":
        tokenizer = Tokenizer.from_pretrained(config["tokenizer"]["path"], trust_remote_code=True)
        evaluator = build_lm_harness_evaluator(
            tokenizer=tokenizer,
            evaluator_cfg=evaluator_cfg,
            max_seq_len=config["model"]["max_seq_len"],
            max_new_tokens=config["offline_eval"].get("max_new_tokens", 256),
            eval_batch_size=config["eval_batch_size"],
            amp_dtype=get_amp_dtype(config["amp"]),
            device=device,
            additional_until_tokens=config["offline_eval"].get("additional_until_tokens", None),
            add_prefix=config["tokenizer"].get("add_prefix", False),
            generation_num_chunks=config["offline_eval"].get("generation_num_chunks", 1),
            prefill_chunk_size=config["offline_eval"].get("prefill_chunk_size", None),
            cache_requests=config["offline_eval"].get("cache_requests", True),
            max_test_num=config["offline_eval"].get("max_test_num", None)
        )
        meta_evaluator.add_evaluator(evaluator, evaluator_cfg["evaluator"])
    elif evaluator_cfg["evaluator"] == "ruler":
        tokenizer = Tokenizer.from_pretrained(config["tokenizer"]["path"], trust_remote_code=True)
        evaluator = build_ruler_evaluator(
            evaluator_cfg=evaluator_cfg,
            tokenizer=tokenizer,
            eval_batch_size=config["eval_batch_size"],
            amp_dtype=get_amp_dtype(config["amp"]),
            verbose=evaluator_cfg.get("verbose", False),
            prefill_chunk_size=config["offline_eval"].get("prefill_chunk_size", None),
        )
        meta_evaluator.add_evaluator(evaluator, evaluator_cfg["evaluator"])
    elif evaluator_cfg["evaluator"] == "prompt":
        tokenizer = Tokenizer.from_pretrained(config["tokenizer"]["path"], trust_remote_code=True)
        evaluator = build_prompt_evaluator(
            tokenizer=tokenizer,
            evaluator_cfg=evaluator_cfg,
            eval_batch_size=config["eval_batch_size"],
            amp_dtype=get_amp_dtype(config["amp"]),
        )
        meta_evaluator.add_evaluator(evaluator, evaluator_cfg["evaluator"])
    else:
        raise ValueError(f"Unknown evaluator: {evaluator_cfg['evaluator']}")
    
    dump_config(config, meta_evaluator.work_dir, config_name="config.yaml")

    meta_evaluator.evaluate()
    dist_close()


if __name__ == "__main__":
    main()
