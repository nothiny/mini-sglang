"""Synthetic MoE layer benchmark for decode and prefill regimes."""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Dict, Iterable, List

import torch
from minisgl.moe import MoeBackendConfig
from minisgl.moe.fused import FusedMoe
from minisgl.moe.weights import quantize_expert_weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=768)
    parser.add_argument("--token-counts", default="1,4,8,16,64,256")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--inner-repeats", type=int, default=5)
    parser.add_argument(
        "--small-m-threshold",
        "--moe-small-m-threshold",
        type=int,
        default=MoeBackendConfig.small_m_threshold,
    )
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--include-autotune", action="store_true")
    parser.add_argument("--include-int8", action="store_true")
    parser.add_argument("--include-offload", action="store_true")
    parser.add_argument("--offload-cache-size", type=int, default=16)
    args = parser.parse_args()
    args.token_counts = tuple(int(value) for value in args.token_counts.split(","))
    if any(value < 1 for value in args.token_counts):
        parser.error("--token-counts must contain positive integers")
    if args.offload_cache_size < 1:
        parser.error("--offload-cache-size must be positive")
    if args.inner_repeats < 1:
        parser.error("--inner-repeats must be positive")
    return args


def benchmark_call(
    call,
    warmup: int,
    iterations: int,
    inner_repeats: int,
) -> Dict[str, float]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        started.record()
        for _ in range(inner_repeats):
            call()
        finished.record()
        finished.synchronize()
        samples.append(started.elapsed_time(finished) * 1_000 / inner_repeats)
    return {
        "mean_us": statistics.mean(samples),
        "median_us": statistics.median(samples),
        "p95_us": sorted(samples)[int(0.95 * (len(samples) - 1))],
    }


def make_backend(
    *,
    workspace_cache: bool,
    small_m_threshold: int,
    autotune: bool = False,
    expert_offload: bool = False,
    expert_cache_size: int = 0,
) -> FusedMoe:
    return FusedMoe(
        MoeBackendConfig(
            enable_workspace_cache=workspace_cache,
            small_m_threshold=small_m_threshold,
            autotune=autotune,
            expert_offload=expert_offload,
            expert_cache_size=expert_cache_size,
        )
    )


def run_case(
    *,
    name: str,
    backend: FusedMoe,
    token_counts: Iterable[int],
    hidden_size: int,
    num_experts: int,
    top_k: int,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
    inner_repeats: int,
) -> List[Dict[str, float | int | str]]:
    results: List[Dict[str, float | int | str]] = []
    for num_tokens in token_counts:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(1_000 + num_tokens)
        source = torch.randn(
            num_tokens,
            hidden_size,
            dtype=dtype,
            device="cuda",
            generator=generator,
        )
        input_buffer = torch.empty_like(source)
        router_logits = torch.randn(
            num_tokens,
            num_experts,
            dtype=dtype,
            device="cuda",
            generator=generator,
        )

        def call() -> None:
            input_buffer.copy_(source)
            backend.forward(
                input_buffer,
                w1,
                w2,
                router_logits,
                top_k,
                True,
                "silu",
                False,
                w1_scale,
                w2_scale,
            )

        timing = benchmark_call(call, warmup, iterations, inner_repeats)
        results.append(
            {
                "case": name,
                "num_tokens": num_tokens,
                **timing,
                "workspace_allocations": backend.workspace_cache.allocation_count,
                "expert_cache_hits": sum(cache.hits for cache in backend._resident_caches.values()),
                "expert_cache_misses": sum(
                    cache.misses for cache in backend._resident_caches.values()
                ),
                "host_to_device_bytes": sum(
                    cache.bytes_transferred for cache in backend._resident_caches.values()
                ),
                "autotuned_direct_shapes": sum(backend._kernel_choices.values()),
                "autotuned_shapes": len(backend._kernel_choices),
            }
        )
    return results


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)
    device = torch.device("cuda")
    w1 = (
        torch.randn(
            args.num_experts,
            2 * args.intermediate_size,
            args.hidden_size,
            dtype=dtype,
            device=device,
        )
        / args.hidden_size**0.5
    )
    w2 = (
        torch.randn(
            args.num_experts,
            args.hidden_size,
            args.intermediate_size,
            dtype=dtype,
            device=device,
        )
        / args.intermediate_size**0.5
    )

    results = run_case(
        name="grouped-no-workspace-cache",
        backend=make_backend(workspace_cache=False, small_m_threshold=0),
        token_counts=args.token_counts,
        hidden_size=args.hidden_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
        w1=w1,
        w2=w2,
        w1_scale=None,
        w2_scale=None,
        dtype=dtype,
        warmup=args.warmup,
        iterations=args.iterations,
        inner_repeats=args.inner_repeats,
    )
    results += run_case(
        name="grouped-workspace-cache",
        backend=make_backend(workspace_cache=True, small_m_threshold=0),
        token_counts=args.token_counts,
        hidden_size=args.hidden_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
        w1=w1,
        w2=w2,
        w1_scale=None,
        w2_scale=None,
        dtype=dtype,
        warmup=args.warmup,
        iterations=args.iterations,
        inner_repeats=args.inner_repeats,
    )
    if args.include_autotune:
        results += run_case(
            name="autotuned",
            backend=make_backend(
                workspace_cache=True,
                small_m_threshold=args.small_m_threshold,
                autotune=True,
            ),
            token_counts=args.token_counts,
            hidden_size=args.hidden_size,
            num_experts=args.num_experts,
            top_k=args.top_k,
            w1=w1,
            w2=w2,
            w1_scale=None,
            w2_scale=None,
            dtype=dtype,
            warmup=args.warmup,
            iterations=args.iterations,
            inner_repeats=args.inner_repeats,
        )
    if args.include_offload:
        w1_host = w1.cpu().pin_memory()
        w2_host = w2.cpu().pin_memory()
        results += run_case(
            name=f"offload-cache-{args.offload_cache_size}",
            backend=make_backend(
                workspace_cache=True,
                small_m_threshold=args.small_m_threshold,
                expert_offload=True,
                expert_cache_size=args.offload_cache_size,
            ),
            token_counts=args.token_counts,
            hidden_size=args.hidden_size,
            num_experts=args.num_experts,
            top_k=args.top_k,
            w1=w1_host,
            w2=w2_host,
            w1_scale=None,
            w2_scale=None,
            dtype=dtype,
            warmup=args.warmup,
            iterations=args.iterations,
            inner_repeats=args.inner_repeats,
        )
    results += run_case(
        name="optimized",
        backend=make_backend(
            workspace_cache=True,
            small_m_threshold=args.small_m_threshold,
        ),
        token_counts=args.token_counts,
        hidden_size=args.hidden_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
        w1=w1,
        w2=w2,
        w1_scale=None,
        w2_scale=None,
        dtype=dtype,
        warmup=args.warmup,
        iterations=args.iterations,
        inner_repeats=args.inner_repeats,
    )
    if args.include_int8:
        w1_int8, w1_scale = quantize_expert_weight(w1)
        w2_int8, w2_scale = quantize_expert_weight(w2)
        del w1, w2
        torch.cuda.empty_cache()
        results += run_case(
            name="optimized-int8",
            backend=make_backend(
                workspace_cache=True,
                small_m_threshold=args.small_m_threshold,
            ),
            token_counts=args.token_counts,
            hidden_size=args.hidden_size,
            num_experts=args.num_experts,
            top_k=args.top_k,
            w1=w1_int8,
            w2=w2_int8,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            dtype=dtype,
            warmup=args.warmup,
            iterations=args.iterations,
            inner_repeats=args.inner_repeats,
        )

    print(
        "MOE_BENCHMARK_RESULT="
        + json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "dtype": args.dtype,
                "num_experts": args.num_experts,
                "top_k": args.top_k,
                "hidden_size": args.hidden_size,
                "intermediate_size": args.intermediate_size,
                "results": results,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
