"""End-to-end DeepSeek-V2-Lite benchmark (mini-deepseek).

Runs the real checkpoint through the engine. On a 12 GiB card use
``--expert-offload`` (optionally with ``--expert-quantization fp8``); BF16
offload is the exact-precision path, FP8 trades accuracy for host memory.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from minisgl.core import SamplingParams
from minisgl.llm import LLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/home/yzd/models/DeepSeek-V2-Lite")
    parser.add_argument("--attention-backend", default="fi")
    parser.add_argument("--expert-offload", action="store_true", default=True)
    parser.add_argument("--expert-cache-size", type=int, default=8)
    parser.add_argument("--expert-quantization", choices=("none", "fp8", "int8"), default="none")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--memory-ratio", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine_kwargs: dict = {
        "attention_backend": args.attention_backend,
        "max_seq_len_override": args.max_seq_len,
        "max_extend_tokens": args.max_seq_len,
        "max_running_req": 2,
        "cuda_graph_bs": [],
        "memory_ratio": args.memory_ratio,
    }
    if args.expert_offload:
        engine_kwargs.update(
            moe_expert_offload=True,
            moe_expert_cache_size=args.expert_cache_size,
        )
    if args.expert_quantization != "none":
        engine_kwargs["moe_expert_quantization"] = args.expert_quantization

    started = time.perf_counter()
    llm = LLM(args.model, **engine_kwargs)
    load_seconds = time.perf_counter() - started

    prompt = [1000 + index % 37 for index in range(args.prompt_tokens)]
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)

    # First run is cold (experts still on host); the second is the steady state.
    cold_started = time.perf_counter()
    llm.generate([prompt], sampling)
    cold_seconds = time.perf_counter() - cold_started
    torch.cuda.synchronize()
    warm_started = time.perf_counter()
    outputs = llm.generate([prompt], sampling)
    torch.cuda.synchronize()
    warm_seconds = time.perf_counter() - warm_started

    status = (
        llm.cache_manager.hicache_status() if hasattr(llm.cache_manager, "hicache_status") else {}
    )
    report = {
        "model": args.model,
        "expert_offload": args.expert_offload,
        "expert_quantization": args.expert_quantization,
        "expert_cache_size": args.expert_cache_size,
        "kv_pool": type(llm.engine.kv_cache).__name__,
        "attn_backend": type(llm.engine.attn_backend).__name__,
        "num_pages": llm.engine.num_pages,
        "load_seconds": round(load_seconds, 2),
        "cold_seconds": round(cold_seconds, 3),
        "warm_seconds": round(warm_seconds, 3),
        "warm_tokens_per_second": round(args.max_tokens / warm_seconds, 2),
        "output_tokens": outputs[0]["token_ids"],
        "status": status.get("enabled", None),
    }
    print("DEEPSEEK_BENCHMARK_RESULT=" + json.dumps(report, sort_keys=True))
    llm.shutdown()


if __name__ == "__main__":
    main()
