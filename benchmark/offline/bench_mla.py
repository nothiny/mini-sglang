"""MLA memory and decode-latency benchmark.

Reports the KV cache cost of MLA versus the MHA-equivalent for a DeepSeek-V2
shaped model, plus paged absorbed-MLA decode latency as the context grows.

This is a kernel/memory measurement, not end-to-end serving: the full
DeepSeek-V2-Lite path needs the MoE from the mini-deepseek branch.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch
from flashinfer.mla import BatchMLAPagedAttentionWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--qk-nope-head-dim", type=int, default=128)
    parser.add_argument("--qk-rope-head-dim", type=int, default=64)
    parser.add_argument("--v-head-dim", type=int, default=128)
    parser.add_argument("--kv-lora-rank", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=27)
    parser.add_argument("--contexts", default="512,2048,8192,32768")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--vram-gib", default="12,24,48,80")
    return parser.parse_args()


def kv_memory_report(args: argparse.Namespace, itemsize: int) -> dict:
    mla_per_token = (args.kv_lora_rank + args.qk_rope_head_dim) * itemsize * args.num_layers
    mha_per_token = (
        args.num_heads
        * ((args.qk_nope_head_dim + args.qk_rope_head_dim) + args.v_head_dim)
        * itemsize
        * args.num_layers
    )
    report = {
        "mla_bytes_per_token": mla_per_token,
        "mha_bytes_per_token": mha_per_token,
        "mla_over_mha": mha_per_token / mla_per_token,
        "max_context": {},
    }
    for vram in (float(v) for v in args.vram_gib.split(",")):
        budget = int(vram * (1 << 30))
        report["max_context"][f"{vram:g}GiB"] = {
            "mla_tokens": budget // mla_per_token,
            "mha_tokens": budget // mha_per_token,
        }
    return report


def decode_latency(args: argparse.Namespace, dtype: torch.dtype) -> list[dict]:
    device = "cuda"
    rank = args.kv_lora_rank
    kpe = args.qk_rope_head_dim
    sm_scale = (args.qk_nope_head_dim + kpe) ** -0.5
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device), backend="fa2"
    )

    results: list[dict] = []
    for context in (int(c) for c in args.contexts.split(",")):
        ckv = torch.randn(context, 1, rank, dtype=dtype, device=device)
        kpe_cache = torch.randn(context, 1, kpe, dtype=dtype, device=device)
        q_abs = torch.randn(1, args.num_heads, rank, dtype=dtype, device=device)
        q_pe = torch.randn(1, args.num_heads, kpe, dtype=dtype, device=device)
        qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
        kv_indptr = torch.tensor([0, context], dtype=torch.int32, device=device)
        kv_indices = torch.arange(context, dtype=torch.int32, device=device)
        kv_len = torch.tensor([context], dtype=torch.int32, device=device)
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_len,
            args.num_heads,
            rank,
            kpe,
            1,
            True,
            sm_scale,
            dtype,
            dtype,
        )

        def call() -> None:
            wrapper.run(q_abs, q_pe, ckv, kpe_cache)

        for _ in range(args.warmup):
            call()
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            call()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000)
        results.append(
            {
                "context": context,
                "median_us": statistics.median(samples),
                "p95_us": sorted(samples)[int(0.95 * (len(samples) - 1))],
            }
        )
    return results


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    try:
        report = {
            "gpu": torch.cuda.get_device_name(),
            "dtype": args.dtype,
            "kv_memory": kv_memory_report(args, dtype.itemsize),
            "decode": decode_latency(args, dtype),
        }
    except RuntimeError as exc:  # pragma: no cover - requires CUDA + FlashInfer
        report = {"error": str(exc)}
    print("MLA_BENCHMARK_RESULT=" + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
