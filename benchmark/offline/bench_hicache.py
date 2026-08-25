from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Dict, List

import torch
from minisgl.core import SamplingParams
from minisgl.llm import LLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare forced recomputation, L2 restore, or L3 restore latency."
    )
    parser.add_argument("--model", required=True, help="Local model path or Hugging Face ID")
    parser.add_argument("--attention-backend", "--attn", default="fi")
    parser.add_argument("--num-pages", type=int, default=192)
    parser.add_argument("--prefix-tokens", type=int, default=130)
    parser.add_argument("--tier", choices=["none", "l2", "l3"], default="l3")
    parser.add_argument("--host-ratio", type=float)
    parser.add_argument("--storage-ratio", type=float, default=2.0)
    parser.add_argument("--storage-path")
    parser.add_argument("--staging-pages", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup-alternations", type=int, default=2)
    parser.add_argument("--policy", choices=["always", "cost"], default="always")
    parser.add_argument("--transfer-backend", choices=["auto", "triton", "torch"], default="auto")
    return parser.parse_args()


def make_prompt(start: int, length: int, period: int) -> List[int]:
    return [start + index % period for index in range(length)]


def main() -> None:
    args = parse_args()
    if args.prefix_tokens + 1 > args.num_pages:
        raise ValueError("A single request must fit in the configured L1 page count")
    if args.prefix_tokens * 2 <= args.num_pages:
        raise ValueError("Two prefixes must exceed L1 capacity to force eviction")
    if args.iterations < 1 or args.warmup_alternations < 0:
        raise ValueError("Iterations must be positive and warmup must be non-negative")

    engine_kwargs = {}
    if args.tier == "l2":
        engine_kwargs.update(
            enable_hicache=True,
            hicache_ratio=args.host_ratio if args.host_ratio is not None else 2.0,
            hicache_policy=args.policy,
            hicache_transfer_backend=args.transfer_backend,
        )
    elif args.tier == "l3":
        engine_kwargs.update(
            enable_hicache=True,
            hicache_ratio=args.host_ratio if args.host_ratio is not None else 1.0,
            hicache_storage_ratio=args.storage_ratio,
            hicache_storage_path=args.storage_path,
            hicache_staging_pages=args.staging_pages,
            hicache_policy=args.policy,
            hicache_transfer_backend=args.transfer_backend,
        )

    llm = LLM(
        args.model,
        attention_backend=args.attention_backend,
        max_seq_len_override=args.num_pages,
        max_extend_tokens=args.num_pages,
        max_running_req=4,
        cuda_graph_bs=[1],
        num_page_override=args.num_pages,
        **engine_kwargs,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    prompt_a = make_prompt(1000, args.prefix_tokens, 37)
    prompt_b = make_prompt(3000, args.prefix_tokens, 41)

    try:
        expected: Dict[str, List[int]] = {}

        def run_one(prompt: List[int]) -> tuple[float, List[int]]:
            start = time.perf_counter()
            result = llm.generate([prompt], sampling)[0]
            torch.cuda.synchronize()
            token_ids = result["token_ids"]
            assert isinstance(token_ids, list)
            return time.perf_counter() - start, token_ids

        _, expected["a"] = run_one(prompt_a)
        _, expected["b"] = run_one(prompt_b)
        for _ in range(args.warmup_alternations):
            run_one(prompt_a)
            run_one(prompt_b)

        samples: List[float] = []
        outputs_equal = True
        for index in range(args.iterations):
            name, prompt = ("a", prompt_a) if index % 2 == 0 else ("b", prompt_b)
            duration, output = run_one(prompt)
            samples.append(duration)
            outputs_equal &= output == expected[name]

        llm.cache_manager.check_integrity()
        report = {
            "tier": args.tier,
            "outputs_equal": outputs_equal,
            "output_tokens": expected,
            "samples_ms": [round(value * 1000, 3) for value in samples],
            "median_ms": round(statistics.median(samples) * 1000, 3),
            "mean_ms": round(statistics.mean(samples) * 1000, 3),
            "hicache": llm.cache_manager.hicache_status(),
        }
        print(json.dumps(report, indent=2))
        if not report["outputs_equal"]:
            raise RuntimeError("Restored output differs from the original computed output")
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
