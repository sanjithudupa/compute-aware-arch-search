from copy import deepcopy
from typing import Any

import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import pandas as pd
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.models.utils import handle_stop_sequences
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window
from tqdm import tqdm

from ellm.tokenizer import Tokenizer
from ellm.utils import (
    chunk_list_by_size, 
    get_device, 
    get_dist_rank,
    get_dist_size,
    DistLogger,
    is_master,
)
import torch.distributed as dist

try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

import logging
logger = logging.getLogger(__name__)
dist_logger = DistLogger(logger)

__all__ = ["LMEvalWrapper"]

pd.set_option("future.no_silent_downcasting", True)


class LMEvalWrapper(LM):
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        max_seq_len: int,
        batch_size: int = 1,
        max_new_tokens: int = 256,
        add_prefix: bool = False,
        additional_until_tokens: str = None,
        generation_num_chunks: int = 1, # chunk_generation
        prefill_chunk_size: int = None, # chunk_prefill
        amp_dtype: torch.dtype=torch.bfloat16,
        device: torch.device = "cuda",
        disable_tqdm: bool = False,
        use_runtime_cache: bool = False,
        forward_kwargs: dict = None,
        eval_kwargs: dict = None,
    ) -> None:
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_new_tokens = max_new_tokens  # Keep original for validation
        self.generation_num_chunks = generation_num_chunks
        self.batch_size = batch_size
        self.prefill_chunk_size = prefill_chunk_size

        self._world_size = get_dist_size()
        self._rank = get_dist_rank()
        self.device = device

        # Use actual Accelerator from HuggingFace accelerate if available
        # Otherwise create a simple fallback
        if ACCELERATE_AVAILABLE:
            self.accelerator = Accelerator()
        else:
            # Fallback: Create a simple accelerator-like object for lm_eval compatibility
            class SimpleAccelerator:
                def __init__(self, world_size, rank, device):
                    self.world_size = world_size
                    self.rank = rank
                    self.device = device
                
                def gather(self, data):
                    """Gather data from all processes. Handles both tensors and lists."""
                    if not dist.is_initialized() or self.world_size == 1:
                        return data
                    
                    # If it's a tensor, gather it
                    if isinstance(data, torch.Tensor):
                        gathered = [torch.zeros_like(data) for _ in range(self.world_size)]
                        dist.all_gather(gathered, data)
                        return torch.cat(gathered, dim=0)
                    # If it's a list, convert to tensor, gather, then convert back
                    elif isinstance(data, list):
                        # Try to convert to tensor if possible
                        try:
                            if len(data) > 0 and isinstance(data[0], (int, float)):
                                tensor_data = torch.tensor(data, device=self.device)
                                gathered = [torch.zeros_like(tensor_data) for _ in range(self.world_size)]
                                dist.all_gather(gathered, tensor_data)
                                return torch.cat(gathered, dim=0)
                            else:
                                # For complex lists, just return as-is (single process)
                                return data
                        except Exception:
                            # If conversion fails, return as-is
                            return data
                    else:
                        # For other types, return as-is
                        return data
            
            self.accelerator = SimpleAccelerator(self._world_size, self._rank, device)

        self.add_prefix = add_prefix
        if additional_until_tokens is not None:
            additional_until_tokens = additional_until_tokens.split("|")
            additional_until_tokens_list = []
            for tokens in additional_until_tokens:
                tokens = tokens.split(",")
                tokens = [int(token) for token in tokens]
                additional_until_tokens_list.append(tokens)
            self.additional_until_tokens = additional_until_tokens_list
        else:
            self.additional_until_tokens = None

        self.disable_tqdm = disable_tqdm
        self.amp_dtype = amp_dtype

        self.batch_size = batch_size
        self.use_runtime_cache = use_runtime_cache
        self.loglikelihood_cache = None
        self.generation_cache = None
        self.forward_kwargs = forward_kwargs if forward_kwargs is not None else {}
        self.eval_kwargs = eval_kwargs if eval_kwargs is not None else {}

        if use_runtime_cache:
            dist_logger.info("Using runtime cache for LM evaluation")

        # Optional diffusion generation mode (for diffusion LLMs)
        # Configure via evaluator cfg -> eval_kwargs
        self.use_diffusion = bool(self.eval_kwargs.get("use_diffusion", False))
        self.diffusion_steps = int(self.eval_kwargs.get("diffusion_steps", 32))
        self.diffusion_logits_temp = float(self.eval_kwargs.get("logits_temp", 0.9))
        # Allow overriding max_new_tokens specifically for diffusion generation
        # This only applies during diffusion generation, not for validation
        self.diffusion_max_new_tokens = self.eval_kwargs.get("diffusion_tokens") or self.eval_kwargs.get("max_new_tokens") or None
        # If not provided, we'll fallback to tokenizer.mask_token_id or 151665 later
        self.diffusion_mask_token_id = self.eval_kwargs.get("mask_token_id", None)
        # Top-k and top-p filtering for diffusion generation
        self.diffusion_top_k = self.eval_kwargs.get("top_k", None)
        self.diffusion_top_p = self.eval_kwargs.get("top_p", None)
        # Repetition penalty for diffusion generation
        self.diffusion_repetition_penalty = self.eval_kwargs.get("repetition_penalty", None)
        # V4-specific parameters
        self.diffusion_alg = self.eval_kwargs.get("alg", "maskgit_plus")
        self.diffusion_eps = float(self.eval_kwargs.get("eps", 1e-3))
        self.diffusion_alg_temp = self.eval_kwargs.get("alg_temp", None)
        # Batch size for diffusion generation (can be smaller than regular batch_size due to memory)
        self.diffusion_batch_size = self.eval_kwargs.get("diffusion_batch_size", None)
        if self.diffusion_batch_size is None:
            # Default to smaller batch size for diffusion (more memory intensive)
            # Diffusion processes full sequence at once, so needs more memory
            self.diffusion_batch_size = max(1, min(self.batch_size, 2))
        # Optional debug toggle for verbose generation logging
        self.debug_generate = bool(self.eval_kwargs.get("debug_generate", False))

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def _loglikelihood_tokens(
        self, requests: list[tuple[tuple[str, str], list[int], list[int]]]
    ) -> list[tuple[float, bool]]:
        total_requests_num = len(requests)
        sorted_indices, requests = zip(*sorted(enumerate(requests), key=lambda x: len(x[1][1] + x[1][2]), reverse=True))

        batched_requests = chunk_list_by_size(requests, self.batch_size)

        device = get_device(self.model)
        response1 = []
        response2 = []
        for batch in tqdm(
            batched_requests,
            total=len(batched_requests),
            disable=self.disable_tqdm or (self.rank != 0),
            desc=f"Running requests",
        ):
            _, batched_context_enc, batched_continuation_enc = zip(*batch)
            batched_all_enc = [
                context_enc + continuation_enc[:-1]
                for context_enc, continuation_enc in zip(batched_context_enc, batched_continuation_enc)
            ]
            assert all(len(enc) <= self.max_seq_len for enc in batched_all_enc)
            max_len = max(len(enc) for enc in batched_all_enc)
            batched_all_enc = [enc + [self.tokenizer.pad_token_id] * (max_len - len(enc)) for enc in batched_all_enc]
            batched_inp = torch.tensor(batched_all_enc, dtype=torch.long, device=device)
                        
            with torch.autocast(device_type=str(device), dtype=self.amp_dtype, enabled=True):
                logits = self.model(batched_inp, attention_mask=None, 
                                    return_loss=False, return_logits=True, **self.forward_kwargs)["logits"]
                batched_logprob = F.log_softmax(logits, dim=-1)          
            for logprob, context_enc, continuation_enc in zip(
                batched_logprob, batched_context_enc, batched_continuation_enc
            ):
                logprob = logprob[len(context_enc) - 1 : len(context_enc) + len(continuation_enc) - 1]
                greedy_tokens = logprob.argmax(dim=-1)
                max_equal = greedy_tokens.tolist() == continuation_enc
                logprob = [logprob[i][continuation_enc[i]] for i in range(len(continuation_enc))]
                
                if self.eval_kwargs.get("average_logprobs", False):
                    response1.append(float(sum(logprob)) / (len(logprob) + 1e-8))
                else:
                    response1.append(float(sum(logprob)))

                response2.append(bool(max_equal))

        # pack
        response = list(zip(response1, response2))[:total_requests_num]
        _response = sorted(zip(sorted_indices, response), key=lambda x: x[0])
        response = [x[1] for x in _response]
        return response

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """Compute log-likelihood of generating a continuation from a context.
        Downstream tasks should attempt to use loglikelihood instead of other
        LM calls whenever possible.

        :param requests: list[Instance]
            A list of Instance objects, with property `args` which returns a tuple (context, continuation).
            `context: str`
                Context string. Implementations of LM must be able to handle an
                empty context string.
            `continuation: str`
                The continuation over which log likelihood will be calculated. If
                there is a word boundary, the space should be in the continuation.
                For example, context="hello" continuation=" world" is correct.

        :return: list[tuple[float, bool]]
            A list of pairs (logprob, isgreedy)
            `logprob: float`
                The log probability of `continuation`.
            `isgreedy`:
                Whether `continuation` would be generated by greedy sampling from `context`.
        """
        new_reqs = []
        if self.use_runtime_cache and self.loglikelihood_cache is not None:
            new_reqs = self.loglikelihood_cache
        else:
            for context, continuation in [req.args for req in requests]:
                continuation = str(continuation)
                                
                if context == "":
                    context_enc = [self.tokenizer.prefix_token_id]
                    continuation_enc = self.tokenizer.encode(continuation, add_special_tokens=False)
                else:
                    rstrip_context = context.rstrip()
                    n_spaces = len(context) - len(rstrip_context)
                    if n_spaces > 0:
                        continuation = context[-n_spaces:] + continuation
                        context = context[:-n_spaces]
                        assert context == rstrip_context

                    context_enc = [self.tokenizer.prefix_token_id] if self.add_prefix else []
                    context_enc += self.tokenizer.encode(context, add_special_tokens=False)
                    full_enc = [self.tokenizer.prefix_token_id] if self.add_prefix else []
                    full_enc += self.tokenizer.encode(context + continuation, add_special_tokens=False)
                    continuation_enc = full_enc[len(context_enc) :]

                new_reqs.append(((context, continuation), context_enc, continuation_enc))

        if self.use_runtime_cache and self.loglikelihood_cache is None:
            self.loglikelihood_cache = new_reqs

        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        """Compute full log-likelihood of a string, with no truncation, for perplexity computation
        - We will use the full max context length of the model.
        - For inputs that exceed the max context length, we divide the tokenized string into chunks of up to
        the max context length.
        - IMPORTANT: Each document's loglikelihood/perplexity is computed *separately*, unlike other implementations
          which may simply concatenate multiple documents together.
        - IMPORTANT: We maximize the amount of context for each prediction. Specifically, for inputs that we break into
          multiple chunks, the last input will still a full-sized context.
          Example:
            Input tokens: [ 0 1 2 3 4 5 6 7 8 9 ]
            Prefix: BOS/EOS
            Max context length: 4
            Resulting input/prediction pairs:

                INPUT:  BOS   0   1   2
                PRED:     0   1   2   3

                INPUT:    3   4   5   6
                PRED:     4   5   6   7

                INPUT:    5   6   7   8
                PRED:             8   9

          Observe that:
            1. Each token is predicted exactly once
            2. For the last pair, we provide the full context, but only score the last two tokens

        :param requests: list[Instance]
            A list of Instance objects with property `args` which returns a tuple (context,).
            string: str
                String for which we are computing overall loglikelihood
        :return: list[tuple[float]]
            A list of tuples (logprob,)
            logprob: float
                The log probability of `context` conditioned on the BOS/EOS token.
                Can also be overridden for custom cases by `prefix_token_id`.
        """
        loglikelihoods = []

        for (string,) in [req.args for req in requests]:
            rolling_token_windows = list(
                map(
                    make_disjoint_window,
                    get_rolling_token_windows(
                        token_list=self.tokenizer.encode(string, add_special_tokens=False),
                        prefix_token=self.tokenizer.prefix_token_id,
                        max_seq_len=self.max_seq_len,
                        context_len=1,
                    ),
                )
            )
            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            string_nll = self._loglikelihood_tokens(rolling_token_windows)
            string_nll = [x[0] for x in string_nll]
            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

        return loglikelihoods

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None, add_prefix=None) -> list[int]:
        add_prefix = add_prefix if add_prefix is not None else self.add_prefix
        # default for None - empty dict, use predefined tokenizer param
        # used for all models except for CausalLM or predefined value
        special_tokens_kwargs = {}

        # by default for CausalLM - false or self.add_prefix is set
        if add_special_tokens is None:
            special_tokens_kwargs = {"add_special_tokens": False}
        else:
            special_tokens_kwargs = {"add_special_tokens": add_special_tokens}

        encoding = self.tokenizer.encode(string, **special_tokens_kwargs)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            if add_prefix:
                encoding = encoding[-left_truncate_len+1:]
            else:
                encoding = encoding[-left_truncate_len:]
        
        if add_prefix:
            encoding = [self.tokenizer.prefix_token_id] + encoding

        if left_truncate_len:
            assert len(encoding) <= left_truncate_len, (len(encoding), left_truncate_len)
        return encoding

    def tok_batch_encode(
        self,
        strings: list[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # encode a batch of strings. converts to tensors and pads automatically, unlike tok_encode.

        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            padding_side=padding_side,
            add_special_tokens=False
        )
        if left_truncate_len:
            if self.add_prefix:
                left_truncate_len += 1
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][:, -left_truncate_len:]

        if self.add_prefix:
            encoding["input_ids"] = torch.cat(
                [torch.full((encoding["input_ids"].shape[0], 1), self.tokenizer.prefix_token_id, dtype=torch.long), encoding["input_ids"]],
                dim=1,
            )
            encoding["attention_mask"] = torch.cat(
                [torch.ones((encoding["attention_mask"].shape[0], 1), dtype=torch.long), encoding["attention_mask"]],
                dim=1,
            )
        
        return encoding["input_ids"], encoding["attention_mask"]

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _get_encoded_requests_for_generation(self, requests: list[Instance]) -> list[int, tuple[list[int], dict]]:
        context_encs = [self.tok_encode(req.args[0], left_truncate_len=self.max_seq_len-self.max_new_tokens) for req in requests]
        prompt_lens = [len(enc) for enc in context_encs]
        kwargs = [req.args[1] for req in requests]
        for kwarg in kwargs:
            if "max_gen_toks" in kwarg:
                kwarg["max_new_tokens"] = kwarg.pop("max_gen_toks")
                
        ids = list(range(len(requests)))
        return list(zip(ids, prompt_lens, context_encs, kwargs))

    def _prepare_batch_for_generation(self, batch: list[tuple[int, int, list[int], dict]]) -> dict[str, Any]:
        cids, prompt_lens, context_encs, kwargs_origin = zip(*batch)

        pad_to_len = max(map(len, context_encs))
        attn_masks = [[0] * (pad_to_len - len(enc)) +  [1] * len(enc) for enc in context_encs]
        attn_masks = torch.tensor(attn_masks, dtype=torch.long)
        context_encs = [[self.tokenizer.pad_token_id] * (pad_to_len - len(enc)) + enc for enc in context_encs]
        context_encs = torch.tensor(context_encs, dtype=torch.long)

        # we assume all gen kwargs in the batch are the same
        # this is safe to assume because the `grouper` object ensures it.
        kwargs_origin = kwargs_origin[0]
        # unpack our keyword arguments.
        kwargs = deepcopy(kwargs_origin)  # edge case for repeats > 1
        # add EOS token to stop sequences
        if "max_new_tokens" not in kwargs.keys():
            kwargs["max_new_tokens"] = self.max_new_tokens
        else:
            assert kwargs["max_new_tokens"] <= self.max_new_tokens, (kwargs["max_new_tokens"], self.max_new_tokens)
            assert kwargs["max_new_tokens"] <= self.max_seq_len, (kwargs["max_new_tokens"], self.max_new_tokens)

        max_ctx_len = self.max_seq_len - kwargs["max_new_tokens"]
        assert context_encs.shape[1] <= max_ctx_len, (context_encs.shape[1], max_ctx_len)
        assert all(l <= max_ctx_len for l in prompt_lens), (max_ctx_len, prompt_lens)
        # context_encs = context_encs[:, -max_ctx_len:]
        # attn_masks = attn_masks[:, -max_ctx_len:]
        # prompt_lens = [min(l, max_ctx_len) for l in prompt_lens]
        
        eos = self.tok_decode(self.tokenizer.eos_token_id, skip_special_tokens=False)
        until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
        until_tokens = [self.tok_encode(x, add_prefix=False) for x in until]
        if self.additional_until_tokens is not None:
            until_tokens = until_tokens + self.additional_until_tokens
        
        device = get_device(self.model)
        context_encs = context_encs.to(device)
        attn_masks = attn_masks.to(device)

        return {
            "cids": cids,
            "prompt_lens": prompt_lens,
            "context_encs": context_encs,
            "attn_masks": attn_masks,
            "until": until,
            "until_tokens": until_tokens,
            "kwargs": kwargs,
        }

    def _make_batched_requests(self, requests: tuple[list[int], dict], batch_size: int) -> list[dict[str, Any]]:
        groups = defaultdict(list)
        
        for cid, prompt_len, context_enc, kwargs in requests:
            key = json.dumps(kwargs, sort_keys=True)
            groups[key].append((cid, prompt_len, context_enc, kwargs))
        
        batched_requests = []
        for group in groups.values():
            batched_requests.extend(chunk_list_by_size(group, batch_size))
        
        batched_requests = [self._prepare_batch_for_generation(batch) for batch in batched_requests]

        return batched_requests

    def generate_until(self, requests: list[Instance], disable_tqdm: bool = False) -> list[str]:
        original_len = len(requests)
        requests = self._get_encoded_requests_for_generation(requests)
        requests = sorted(requests, key=lambda x: x[1], reverse=True)
        
        if self.use_runtime_cache and self.generation_cache is not None:
            batched_requests = self.generation_cache
        else:
            batched_requests = self._make_batched_requests(requests, self.batch_size)
            if self.use_runtime_cache and self.generation_cache is None:
                self.generation_cache = batched_requests
        
        # Branch 1: Diffusion-based generation (for diffusion LLMs)
        if self.use_diffusion:
            device = get_device(self.model)
            pbar = tqdm(
                total=len(requests),
                disable=(disable_tqdm or (self.rank != 0)),
                desc="Running generate_until requests (diffusion)",
            )

            res = []
            # In diffusion mode we generate in a single pass per batch (no chunked decoding).
            # Use smaller batch size for diffusion due to memory constraints
            diffusion_batched_requests = []
            for batch in batched_requests:
                # Split large batches into smaller chunks for diffusion
                batch_size = batch["context_encs"].shape[0]
                if batch_size > self.diffusion_batch_size:
                    # Split the batch into smaller chunks
                    for i in range(0, batch_size, self.diffusion_batch_size):
                        end_idx = min(i + self.diffusion_batch_size, batch_size)
                        chunk_batch = {
                            "cids": batch["cids"][i:end_idx],
                            "prompt_lens": batch["prompt_lens"][i:end_idx],
                            "context_encs": batch["context_encs"][i:end_idx],
                            "attn_masks": batch["attn_masks"][i:end_idx],
                            "until": batch["until"],
                            "until_tokens": batch["until_tokens"],
                            "kwargs": batch["kwargs"],
                        }
                        diffusion_batched_requests.append(chunk_batch)
                else:
                    diffusion_batched_requests.append(batch)
            
            for batch in diffusion_batched_requests:
                # Use diffusion-specific max_new_tokens if set, otherwise use the batch's requested value
                requested_gen_len = batch["kwargs"].get("max_new_tokens", self.max_new_tokens)
                if self.diffusion_max_new_tokens is not None:
                    # Cap at diffusion_max_new_tokens if set
                    gen_len = min(int(requested_gen_len), int(self.diffusion_max_new_tokens))
                else:
                    gen_len = int(requested_gen_len)
                # Build src_mask and x following the RULER diffusion evaluator
                bsz, pref_len = batch["context_encs"].shape
                src_mask = torch.ones((bsz, pref_len + gen_len), dtype=torch.bool, device=device)
                src_mask[:, pref_len:] = False

                pad_id = getattr(self.tokenizer, 'pad_token_id', 0)
                fallback_mask_id = getattr(self.tokenizer, 'mask_token_id', None)
                mask_or_pad_id = self.diffusion_mask_token_id if self.diffusion_mask_token_id is not None else (fallback_mask_id if fallback_mask_id is not None else 151665)

                x = torch.full((bsz, pref_len + gen_len), pad_id, dtype=batch["context_encs"].dtype, device=device)
                x[:, :pref_len] = batch["context_encs"].to(device)

                if self.debug_generate:
                    try:
                        maskable = int((~src_mask).sum().item())
                    except Exception:
                        maskable = -1
                    dist_logger.debug(
                        f"[LMHarness.diff] bsz={bsz} pref_len={pref_len} gen_len={gen_len} maskable={maskable} temp={self.diffusion_logits_temp} steps={self.diffusion_steps}"
                    )

                t0 = time.perf_counter()
                with torch.autocast(device_type=str(device), dtype=self.amp_dtype, enabled=True):
                    gen_tokens = self.model.generate_samples_v4(
                        input_ids=x,
                        src_mask=src_mask,
                        diffusion_steps=self.diffusion_steps,
                        logits_temp=self.diffusion_logits_temp,
                        top_k=self.diffusion_top_k,
                        top_p=self.diffusion_top_p,
                        shift=True,
                        mask_token_id=mask_or_pad_id,
                        eval_mode=True,
                        tokenizer=None,  # Don't log intermediate steps
                        log_steps_path=None,  # Don't log intermediate steps
                        repetition_penalty=self.diffusion_repetition_penalty,
                        alg=self.diffusion_alg,
                        eps=self.diffusion_eps,
                        alg_temp=self.diffusion_alg_temp,
                    )
                dt = time.perf_counter() - t0

                # Take only the continuation window
                if gen_tokens.shape[1] >= gen_len:
                    cont_ids = gen_tokens[:, -gen_len:]
                else:
                    pad_col = torch.full((bsz, gen_len - gen_tokens.shape[1]), pad_id, dtype=gen_tokens.dtype, device=gen_tokens.device)
                    cont_ids = torch.cat([pad_col, gen_tokens], dim=1)

                if self.debug_generate:
                    try:
                        sample_preview = cont_ids[0, : min(16, cont_ids.shape[1])].tolist()
                    except Exception:
                        sample_preview = []
                    dist_logger.debug(
                        f"[LMHarness.diff] gen_tokens.shape={tuple(gen_tokens.shape)} cont_ids.shape={tuple(cont_ids.shape)} dt={dt*1000:.2f}ms preview_ids={sample_preview}"
                    )

                # Decode and apply stop sequences
                generated_texts = [self.tok_decode(t.tolist()) for t in cont_ids]
                num_finished = 0
                for cid, prompt_len, text in zip(batch["cids"], batch["prompt_lens"], generated_texts):
                    s = text
                    for term in batch["until"]:
                        if len(term) > 0:
                            s = s.split(term)[0]
                    res.append((cid, s))
                    num_finished += 1

                pbar.set_postfix({"size": batch["context_encs"].size()})
                pbar.update(num_finished)

            # reorder results back to original unsorted form
            res = sorted(res, key=lambda x: x[0])
            res = [x[1] for x in res]

            assert len(res) == len(requests), (len(res), len(requests))
            original_res = res
            assert len(original_res) == original_len
            pbar.close()
            return original_res

        # Branch 2: Standard autoregressive generation
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )

        res = []
        
        assert self.max_new_tokens % self.generation_num_chunks == 0
        max_new_tokens_chunk = self.max_new_tokens // self.generation_num_chunks
        device = get_device(self.model)
        for chunk_id in range(self.generation_num_chunks):
            remained_batch = []
            dist_logger.debug(f"Generate for chunk {chunk_id}/{self.generation_num_chunks} num batched_requests: {len(batched_requests)}")
            for batch in batched_requests:
                kwargs_chunk = deepcopy(batch["kwargs"])
                kwargs_chunk["max_new_tokens"] = min(max_new_tokens_chunk, batch["kwargs"]["max_new_tokens"])
                if self.debug_generate:
                    bsz, pref_len = batch["context_encs"].shape
                    dist_logger.debug(
                        f"[LMHarness.ar] bsz={bsz} pref_len={pref_len} gen_len={kwargs_chunk['max_new_tokens']}"
                    )
                t0 = time.perf_counter()
                with torch.autocast(device_type=str(device), dtype=self.amp_dtype, enabled=True):
                    out = self.model.generate(
                        input_ids=batch["context_encs"],
                        attention_mask=batch["attn_masks"],
                        stop_token_list=batch["until_tokens"],
                        pad_token_id=self.tokenizer.pad_token_id,
                        return_dict=True,
                        return_unfinished=True,
                        prefill_chunk_size=self.prefill_chunk_size,
                        **self.forward_kwargs,
                        **kwargs_chunk,
                    )
                dt = time.perf_counter() - t0
                batch_full_tokens = out["sequences"]
                unfinished = out["unfinished_sequences"]
                
                starts = torch.argmax((batch_full_tokens != self.tokenizer.pad_token_id).long(), dim=-1)
                full_tokens_list = batch_full_tokens.tolist()
                num_finished = 0
                if self.debug_generate:
                    try:
                        preview = batch_full_tokens[0, -min(16, batch_full_tokens.shape[1]):].tolist()
                    except Exception:
                        preview = []
                    dist_logger.debug(
                        f"[LMHarness.ar] sequences.shape={tuple(batch_full_tokens.shape)} dt={dt*1000:.2f}ms preview_tail_ids={preview}"
                    )
                                
                for cid, prompt_len, full_tokens, start, ufi in zip(
                    batch["cids"], batch["prompt_lens"], full_tokens_list, starts, unfinished):
                    # discard context + left-padding toks if using causal decoder-only LM
                    num_new_tokens = len(full_tokens)-batch["context_encs"].shape[1]
                    if (ufi == 0) or (num_new_tokens >= batch["kwargs"]["max_new_tokens"]):
                        cont_toks = full_tokens[start+prompt_len:]
                        s = self.tok_decode(cont_toks)

                        # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
                        for term in batch["until"]:
                            if len(term) > 0:
                                s = s.split(term)[0]

                        res.append((cid, s))
                        num_finished += 1
                    else:
                        kwargs_next_chunk = deepcopy(batch["kwargs"])
                        kwargs_next_chunk["max_new_tokens"] = batch["kwargs"]["max_new_tokens"] - max_new_tokens_chunk
                        kwargs_next_chunk["until"] = batch["until"]
                        remained_batch.append((cid, prompt_len, full_tokens[start:], kwargs_next_chunk))
                
                pbar.set_postfix({"size": batch["context_encs"].size()})
                pbar.update(num_finished)
                            
            batched_requests = self._make_batched_requests(remained_batch, self.batch_size)
        
        # reorder this group of results back to original unsorted form
        res = sorted(res, key=lambda x: x[0])
        res = [x[1] for x in res]
        
        assert len(res) == len(requests), (len(res), len(requests))

        original_res = res        
        assert len(original_res) == original_len        
        
        pbar.close()

        return original_res
    
    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

        return chat_templated
