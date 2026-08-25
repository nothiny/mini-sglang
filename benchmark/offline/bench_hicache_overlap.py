from __future__ import annotations

import argparse
import json
import time
from typing import List

from minisgl.core import SamplingParams
from minisgl.llm.llm import LLM, RequestAllFinished
from minisgl.scheduler.utils import PendingReq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that an L3 prefix restore overlaps an active decode batch."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--attention-backend", "--attn", default="fi")
    parser.add_argument("--num-pages", type=int, default=512)
    parser.add_argument("--prefix-tokens", type=int, default=350)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--host-ratio", type=float, default=1.0)
    parser.add_argument("--storage-ratio", type=float, default=2.0)
    parser.add_argument("--staging-pages", type=int, default=8)
    parser.add_argument("--transfer-backend", choices=["auto", "triton", "torch"], default="auto")
    return parser.parse_args()


def make_prompt(start: int, length: int, period: int) -> List[int]:
    return [start + index % period for index in range(length)]


def main() -> None:
    args = parse_args()
    if args.prefix_tokens + 1 > args.num_pages:
        raise ValueError("A prefix must fit in L1")
    if args.prefix_tokens * 2 <= args.num_pages:
        raise ValueError("Two prefixes must exceed L1 capacity")

    llm = LLM(
        args.model,
        attention_backend=args.attention_backend,
        max_seq_len_override=args.num_pages,
        max_extend_tokens=args.num_pages,
        max_running_req=4,
        cuda_graph_bs=[1],
        num_page_override=args.num_pages,
        enable_hicache=True,
        hicache_ratio=args.host_ratio,
        hicache_storage_ratio=args.storage_ratio,
        hicache_staging_pages=args.staging_pages,
        hicache_policy="always",
        hicache_prefetch=True,
        hicache_transfer_backend=args.transfer_backend,
    )
    one_token = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    long_decode = SamplingParams(temperature=0.0, max_tokens=args.decode_tokens, ignore_eos=True)
    prompt_a = make_prompt(1000, args.prefix_tokens, 37)
    prompt_b = make_prompt(3000, args.prefix_tokens, 41)
    decode_prompt = make_prompt(5000, 32, 29)

    try:
        expected = llm.generate([prompt_a], one_token)[0]["token_ids"]
        llm.generate([prompt_b], one_token)
        llm.cache_manager.check_integrity()

        probe = PendingReq(0, llm._tokenize_one(prompt_a), one_token)
        match = llm.cache_manager.match_req(probe)
        residency_before = {
            "l1_tokens": match.cuda_handle.cached_len,
            "l2_tokens": (match.host_handle.cached_len if match.host_handle is not None else 0),
            "l3_tokens": (
                match.storage_handle.cached_len if match.storage_handle is not None else 0
            ),
        }
        if residency_before["l3_tokens"] <= max(
            residency_before["l1_tokens"], residency_before["l2_tokens"]
        ):
            raise RuntimeError(
                "Setup did not leave a lower-tier-only L3 prefix; reduce L1/L2 capacity"
            )

        llm.pending_requests = [(decode_prompt, long_decode)]
        llm.status_map = {}
        llm.counter = 0
        data = llm.overlap_loop(None)
        if not llm.decode_manager.runnable:
            raise RuntimeError("The decode request did not become runnable")

        before = llm.cache_manager.metrics.snapshot()
        llm.pending_requests.append((prompt_a, one_token))
        started_at = time.perf_counter()
        try:
            while True:
                data = llm.overlap_loop(data)
        except RequestAllFinished:
            pass
        elapsed = time.perf_counter() - started_at

        after = llm.cache_manager.metrics.snapshot()
        actual = llm.status_map[1].output_ids
        report = {
            "outputs_equal": actual == expected,
            "residency_before": residency_before,
            "decode_output_tokens": len(llm.status_map[0].output_ids),
            "restored_output_tokens": actual,
            "elapsed_ms": round(elapsed * 1000, 3),
            "overlapped_restore_requests": int(
                after["overlapped_restore_requests"] - before["overlapped_restore_requests"]
            ),
            "overlapped_restore_ms": round(
                (
                    float(after["restore_overlapped_seconds"])
                    - float(before["restore_overlapped_seconds"])
                )
                * 1000,
                3,
            ),
            "hicache": llm.cache_manager.hicache_status(),
        }
        print(json.dumps(report, indent=2))
        if not report["outputs_equal"]:
            raise RuntimeError("The overlapped L3 restore changed the generated token")
        if report["overlapped_restore_requests"] < 1:
            raise RuntimeError("No L3 restore overlapped decode")
        llm.cache_manager.check_integrity()
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
