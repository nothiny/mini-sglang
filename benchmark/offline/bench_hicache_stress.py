from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from typing import Dict, List, Sequence

import torch
from minisgl.core import SamplingParams
from minisgl.llm import LLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stress the radix cache with a working set larger than L1 and compare "
            "recomputation, L2, and L3 behavior."
        )
    )
    parser.add_argument("--model", required=True, help="Local model path or Hugging Face ID")
    parser.add_argument("--attention-backend", "--attn", default="fi")
    parser.add_argument("--tier", choices=["none", "l2", "l3"], default="l3")
    parser.add_argument("--num-pages", type=int, default=512)
    parser.add_argument("--prefix-tokens", type=int, default=350)
    parser.add_argument("--num-prefixes", type=int, default=8)
    parser.add_argument("--requests", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--host-ratio", type=float, default=1.0)
    parser.add_argument("--storage-ratio", type=float, default=8.0)
    parser.add_argument("--storage-path")
    parser.add_argument("--io-workers", type=int, default=2)
    parser.add_argument("--staging-pages", type=int, default=8)
    parser.add_argument("--policy", choices=["always", "cost"], default="cost")
    parser.add_argument("--transfer-backend", choices=["auto", "triton", "torch"], default="auto")
    parser.add_argument(
        "--check-every",
        type=int,
        default=0,
        help="Run an untimed cache integrity check every N batches; zero checks only at the end.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "num-pages": args.num_pages,
        "prefix-tokens": args.prefix_tokens,
        "num-prefixes": args.num_prefixes,
        "requests": args.requests,
        "batch-size": args.batch_size,
        "max-tokens": args.max_tokens,
        "host-ratio": args.host_ratio,
        "io-workers": args.io_workers,
        "staging-pages": args.staging_pages,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"These options must be positive: {', '.join(invalid)}")
    if args.warmup_rounds < 0 or args.check_every < 0 or args.storage_ratio < 0:
        raise ValueError("Warmup rounds, check interval, and storage ratio must be non-negative")
    if args.num_prefixes < args.batch_size:
        raise ValueError("num-prefixes must be at least batch-size")
    if args.batch_size * (args.prefix_tokens + args.max_tokens) > args.num_pages:
        raise ValueError("The largest concurrent batch must fit in the configured L1 page count")
    if args.num_prefixes * args.prefix_tokens <= args.num_pages:
        raise ValueError("The prefix working set must exceed L1 capacity to create cache pressure")
    if args.tier == "l3" and args.storage_ratio <= 0:
        raise ValueError("The L3 stress mode requires a positive storage ratio")


def make_prompt(index: int, length: int) -> List[int]:
    """Build distinct, model-independent token patterns in a conservative ID range."""
    start = 1000 + index * 97
    period = 37 + index * 2
    return [start + position % period for position in range(length)]


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def metric_delta(
    before: Dict[str, int | float], after: Dict[str, int | float]
) -> Dict[str, int | float]:
    result: Dict[str, int | float] = {}
    for name, value in after.items():
        if name.endswith("_gib_s"):
            continue
        previous = before.get(name, 0)
        if isinstance(value, int) and isinstance(previous, int):
            result[name] = value - previous
        elif isinstance(value, (int, float)) and isinstance(previous, (int, float)):
            result[name] = round(float(value) - float(previous), 6)

    for direction in ("d2h", "h2d", "storage_write", "storage_read"):
        num_bytes = float(result[f"{direction}_bytes"])
        seconds = float(result[f"{direction}_seconds"])
        result[f"{direction}_gib_s"] = round(num_bytes / seconds / (1 << 30), 3) if seconds else 0.0
    return result


def memory_snapshot() -> Dict[str, float]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "device_total_gib": round(total_bytes / (1 << 30), 3),
        "device_free_gib": round(free_bytes / (1 << 30), 3),
        "torch_allocated_gib": round(torch.cuda.memory_allocated() / (1 << 30), 3),
        "torch_reserved_gib": round(torch.cuda.memory_reserved() / (1 << 30), 3),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)

    engine_kwargs: Dict[str, object] = {}
    if args.tier != "none":
        engine_kwargs.update(
            enable_hicache=True,
            hicache_ratio=args.host_ratio,
            hicache_policy=args.policy,
            hicache_transfer_backend=args.transfer_backend,
        )
        if args.tier == "l3":
            engine_kwargs.update(
                hicache_storage_ratio=args.storage_ratio,
                hicache_storage_path=args.storage_path,
                hicache_io_workers=args.io_workers,
                hicache_staging_pages=args.staging_pages,
            )

    graph_batch_sizes = sorted({1, args.batch_size})
    llm = LLM(
        args.model,
        attention_backend=args.attention_backend,
        max_seq_len_override=args.prefix_tokens + args.max_tokens,
        max_extend_tokens=args.num_pages,
        max_running_req=max(4, args.batch_size),
        cuda_graph_bs=graph_batch_sizes,
        num_page_override=args.num_pages,
        **engine_kwargs,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)
    prompts = [make_prompt(index, args.prefix_tokens) for index in range(args.num_prefixes)]

    try:
        expected: Dict[int, List[int]] = {}
        for index, prompt in enumerate(prompts):
            result = llm.generate([prompt], sampling)[0]["token_ids"]
            if not isinstance(result, list):
                raise TypeError("LLM returned non-list token_ids")
            expected[index] = result

        llm.cache_manager.check_integrity()

        def run_batch(indices: Sequence[int]) -> tuple[float, int]:
            torch.cuda.synchronize()
            started_at = time.perf_counter()
            outputs = llm.generate([prompts[index] for index in indices], sampling)
            torch.cuda.synchronize()
            duration = time.perf_counter() - started_at
            mismatches = 0
            for index, output in zip(indices, outputs):
                token_ids = output["token_ids"]
                if not isinstance(token_ids, list) or token_ids != expected[index]:
                    mismatches += 1
            return duration, mismatches

        warmup_indices = list(range(args.num_prefixes))
        for _ in range(args.warmup_rounds):
            for offset in range(0, len(warmup_indices), args.batch_size):
                _, mismatches = run_batch(warmup_indices[offset : offset + args.batch_size])
                if mismatches:
                    raise RuntimeError("Warmup output differs from the primed reference")
        llm.cache_manager.check_integrity()

        status_before = llm.cache_manager.hicache_status()
        metrics_before = status_before["metrics"]
        if not isinstance(metrics_before, dict):
            raise TypeError("HiCache status returned invalid metrics")
        memory_before = memory_snapshot()

        access_order = [index % args.num_prefixes for index in range(args.requests)]
        batch_samples: List[float] = []
        mismatched_requests = 0
        integrity_check_seconds = 0.0
        benchmark_started_at = time.perf_counter()
        for batch_index, offset in enumerate(range(0, args.requests, args.batch_size), start=1):
            duration, mismatches = run_batch(access_order[offset : offset + args.batch_size])
            batch_samples.append(duration)
            mismatched_requests += mismatches
            if args.check_every and batch_index % args.check_every == 0:
                integrity_started_at = time.perf_counter()
                llm.cache_manager.check_integrity()
                integrity_check_seconds += time.perf_counter() - integrity_started_at
        benchmark_wall_seconds = time.perf_counter() - benchmark_started_at
        service_seconds = sum(batch_samples)

        llm.cache_manager.check_integrity()
        status_after = llm.cache_manager.hicache_status()
        metrics_after = status_after["metrics"]
        if not isinstance(metrics_after, dict):
            raise TypeError("HiCache status returned invalid metrics")

        report = {
            "configuration": {
                "model": args.model,
                "tier": args.tier,
                "attention_backend": args.attention_backend,
                "num_pages": args.num_pages,
                "prefix_tokens": args.prefix_tokens,
                "num_prefixes": args.num_prefixes,
                "working_set_tokens": args.num_prefixes * args.prefix_tokens,
                "requests": args.requests,
                "batch_size": args.batch_size,
                "max_tokens": args.max_tokens,
                "warmup_rounds": args.warmup_rounds,
                "host_ratio": args.host_ratio if args.tier != "none" else 0.0,
                "storage_ratio": args.storage_ratio if args.tier == "l3" else 0.0,
                "policy": args.policy if args.tier != "none" else None,
                "transfer_backend": args.transfer_backend if args.tier != "none" else None,
            },
            "correctness": {
                "requests_checked": args.requests,
                "mismatched_requests": mismatched_requests,
                "integrity_check": "passed",
            },
            "performance": {
                "service_seconds": round(service_seconds, 6),
                "benchmark_wall_seconds": round(benchmark_wall_seconds, 6),
                "periodic_integrity_check_seconds": round(integrity_check_seconds, 6),
                "request_throughput_per_second": round(args.requests / service_seconds, 3),
                "effective_ms_per_request": round(service_seconds * 1000 / args.requests, 3),
                "batch_latency_ms": {
                    "mean": round(statistics.mean(batch_samples) * 1000, 3),
                    "p50": round(percentile(batch_samples, 0.50) * 1000, 3),
                    "p95": round(percentile(batch_samples, 0.95) * 1000, 3),
                    "p99": round(percentile(batch_samples, 0.99) * 1000, 3),
                    "max": round(max(batch_samples) * 1000, 3),
                },
                "batch_samples_ms": [round(value * 1000, 3) for value in batch_samples],
            },
            "memory_before_stress": memory_before,
            "memory_after_stress": memory_snapshot(),
            "hicache_metric_delta": metric_delta(metrics_before, metrics_after),
            "hicache_final": status_after,
        }
        print(json.dumps(report, indent=2))
        if mismatched_requests:
            raise RuntimeError(f"{mismatched_requests} restored outputs differed from references")
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
