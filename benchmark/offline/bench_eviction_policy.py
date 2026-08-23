"""End-to-end benchmark for Radix prefix-cache eviction policies.

The workload keeps three page-aligned prefixes resident, makes prefix A old but
frequent, then inserts a fourth prefix.  LRU should prefer recency and evict A,
while frequency-aware policies should normally retain it.  A final probe of A
or C turns that policy decision into a measurable Qwen prefill latency gap.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from typing import Any, Dict, List

import torch
from minisgl.core import SamplingParams
from minisgl.kvcache import (
    AdaptivePolicy,
    EvictionPolicyConfig,
    create_prefix_cache,
)
from minisgl.llm import LLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--policy",
        choices=("lru", "lfu", "lru-k", "frequency-decay", "cost-aware", "adaptive"),
        default="lru",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--num-pages", type=int, default=25)
    parser.add_argument("--prefix-tokens", type=int, default=128)
    parser.add_argument("--probe-prefix", choices=("A", "C"), default="A")
    parser.add_argument("--eviction-k", type=int, default=2)
    parser.add_argument("--eviction-half-life", type=float, default=60.0)
    parser.add_argument("--adaptive-learning-rate", type=float, default=0.05)
    parser.add_argument("--adaptive-seed", type=int, default=0)
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.prefix_tokens < 1 or args.prefix_tokens % args.page_size:
        parser.error("--prefix-tokens must be a positive multiple of --page-size")
    resident_prefixes = args.num_pages * args.page_size // args.prefix_tokens
    if resident_prefixes != 3:
        parser.error("the benchmark requires capacity for exactly three prefixes")
    if args.num_pages * args.page_size == 3 * args.prefix_tokens:
        parser.error("one extra page is required for the uncached query token")
    return args


def make_prefix(prefix_id: int, length: int) -> List[int]:
    """Build unrelated, valid Qwen token-id prefixes."""

    first_token = 1_000 + prefix_id
    return [first_token] + [2_000 + prefix_id * 257 + (i * 17) % 251 for i in range(length - 1)]


def make_policy_config(args: argparse.Namespace) -> EvictionPolicyConfig:
    return EvictionPolicyConfig(
        policy=args.policy,
        lru_k=args.eviction_k,
        frequency_half_life=args.eviction_half_life,
        adaptive_learning_rate=args.adaptive_learning_rate,
        adaptive_seed=args.adaptive_seed,
    )


def reset_cache(llm: LLM, args: argparse.Namespace) -> None:
    """Reset an idle offline cache without reloading model weights."""

    manager = llm.cache_manager
    torch.cuda.synchronize(llm.device)
    manager.free_slots = (
        torch.arange(manager.num_pages, dtype=torch.int32, device=manager.device)
        * manager.page_size
    )
    manager.prefix_cache = create_prefix_cache(
        device=manager.device,
        type="radix",
        eviction_policy_config=make_policy_config(args),
        total_tokens=manager.num_pages * manager.page_size,
        kv_bytes_per_token=llm.engine.kv_bytes_per_token,
    )
    manager.check_integrity()


def resident_tokens(cache: Any, prefix: List[int]) -> int:
    """Inspect exact residency without updating replacement metadata."""

    input_ids = torch.tensor(prefix, dtype=torch.int32)
    _, matched = cache._tree_walk(  # noqa: SLF001 - benchmark-only observability
        input_ids,
        now_ns=time.monotonic_ns(),
        record_access=False,
    )
    return matched


def instrument_cache(cache: Any) -> tuple[Dict[str, Any], List[int], List[int]]:
    active: Dict[str, Any] = {}
    selection_ns: List[int] = []
    candidate_counts: List[int] = []

    original_match = cache.match_prefix

    def measured_match(input_ids: torch.Tensor):
        result = original_match(input_ids)
        active["match_calls"] = active.get("match_calls", 0) + 1
        active["matched_tokens"] = active.get("matched_tokens", 0) + int(
            result.cuda_handle.cached_len
        )
        return result

    cache.match_prefix = measured_match

    policy = cache.eviction_policy
    original_select = policy.select_victim

    def measured_select(candidates, context):
        started_ns = time.perf_counter_ns()
        victim = original_select(candidates, context)
        selection_ns.append(time.perf_counter_ns() - started_ns)
        candidate_counts.append(len(candidates))
        return victim

    policy.select_victim = measured_select
    return active, selection_ns, candidate_counts


def run_request(
    llm: LLM,
    sampling_params: SamplingParams,
    active: Dict[str, Any],
    label: str,
    prompt: List[int],
) -> Dict[str, Any]:
    active.clear()
    torch.cuda.synchronize(llm.device)
    started = time.perf_counter()
    llm.generate([prompt], sampling_params)
    torch.cuda.synchronize(llm.device)
    elapsed_s = time.perf_counter() - started
    matched_tokens = int(active.get("matched_tokens", 0))
    return {
        "label": label,
        "latency_ms": elapsed_s * 1_000,
        "input_tokens": len(prompt),
        "matched_tokens": matched_tokens,
        "prefill_tokens": len(prompt) - matched_tokens,
        "match_calls": int(active.get("match_calls", 0)),
    }


def summarize(repetitions: List[Dict[str, Any]], probe_label: str) -> Dict[str, Any]:
    operation_values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    all_operations: List[Dict[str, Any]] = []
    all_selection_ns: List[int] = []
    all_candidate_counts: List[int] = []

    for repetition in repetitions:
        all_selection_ns.extend(repetition["selection_ns"])
        all_candidate_counts.extend(repetition["candidate_counts"])
        for operation in repetition["operations"]:
            all_operations.append(operation)
            for key in ("latency_ms", "matched_tokens", "prefill_tokens"):
                operation_values[operation["label"]][key].append(operation[key])

    total_time_s = sum(op["latency_ms"] for op in all_operations) / 1_000
    logical_input_tokens = sum(op["input_tokens"] for op in all_operations)
    matched_tokens = sum(op["matched_tokens"] for op in all_operations)
    cacheable_tokens = len(all_operations) * repetitions[0]["prefix_tokens"]
    per_operation = {
        label: {key: statistics.median(values) for key, values in metrics.items()}
        for label, metrics in operation_values.items()
    }

    return {
        "total_time_s": total_time_s,
        "requests_per_s": len(all_operations) / total_time_s,
        "logical_input_tokens_per_s": logical_input_tokens / total_time_s,
        "prefix_token_hit_rate": matched_tokens / cacheable_tokens,
        "matched_tokens": matched_tokens,
        "prefill_tokens": sum(op["prefill_tokens"] for op in all_operations),
        "probe_latency_ms_median": per_operation[probe_label]["latency_ms"],
        "probe_matched_tokens_median": per_operation[probe_label]["matched_tokens"],
        "selection_calls": len(all_selection_ns),
        "selection_us_mean": (
            statistics.mean(all_selection_ns) / 1_000 if all_selection_ns else 0.0
        ),
        "selection_us_median": (
            statistics.median(all_selection_ns) / 1_000 if all_selection_ns else 0.0
        ),
        "candidate_count_mean": (
            statistics.mean(all_candidate_counts) if all_candidate_counts else 0.0
        ),
        "per_operation_median": per_operation,
    }


def main() -> None:
    args = parse_args()
    max_seq_len = max(256, args.prefix_tokens + 2)

    llm = LLM(
        args.model,
        attention_backend="fi",
        max_seq_len_override=max_seq_len,
        max_extend_tokens=max(512, max_seq_len),
        max_running_req=8,
        cuda_graph_max_bs=0,
        page_size=args.page_size,
        num_page_override=args.num_pages,
        cache_eviction_policy=args.policy,
        eviction_k=args.eviction_k,
        eviction_half_life=args.eviction_half_life,
        adaptive_learning_rate=args.adaptive_learning_rate,
        adaptive_seed=args.adaptive_seed,
    )
    sampling_params = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=1)

    prefixes = {name: make_prefix(i, args.prefix_tokens) for i, name in enumerate("ABCD")}
    query_token = 149_000

    # Compile both full-prefill and one-token-extend paths before measurement.
    warmup_prefix = make_prefix(99, args.prefix_tokens)
    warmup_prompt = warmup_prefix + [query_token]
    llm.generate([warmup_prompt], sampling_params)
    llm.generate([warmup_prompt], sampling_params)
    torch.cuda.synchronize(llm.device)

    repetitions: List[Dict[str, Any]] = []
    try:
        for repeat in range(args.repeats):
            reset_cache(llm, args)
            cache = llm.cache_manager.prefix_cache
            active, selection_ns, candidate_counts = instrument_cache(cache)

            sequence = (
                ("A_fill", "A"),
                ("A_hot_1", "A"),
                ("A_hot_2", "A"),
                ("B_fill", "B"),
                ("C_fill", "C"),
                ("B_recent", "B"),
                ("D_pressure", "D"),
            )
            operations = [
                run_request(
                    llm,
                    sampling_params,
                    active,
                    label,
                    prefixes[prefix_name] + [query_token],
                )
                for label, prefix_name in sequence
            ]
            residency_after_pressure = {
                name: resident_tokens(cache, prefix) for name, prefix in prefixes.items()
            }
            operations.append(
                run_request(
                    llm,
                    sampling_params,
                    active,
                    f"{args.probe_prefix}_probe",
                    prefixes[args.probe_prefix] + [query_token],
                )
            )
            llm.cache_manager.check_integrity()

            policy = cache.eviction_policy
            adaptive_weights = None
            if isinstance(policy, AdaptivePolicy):
                adaptive_weights = dict(policy.expert_weights)
            repetitions.append(
                {
                    "repeat": repeat,
                    "prefix_tokens": args.prefix_tokens,
                    "operations": operations,
                    "residency_after_pressure": residency_after_pressure,
                    "selection_ns": selection_ns,
                    "candidate_counts": candidate_counts,
                    "adaptive_weights": adaptive_weights,
                }
            )

        result = {
            "model": args.model,
            "policy": args.policy,
            "repeats": args.repeats,
            "page_size": args.page_size,
            "num_pages": args.num_pages,
            "cache_tokens": args.page_size * args.num_pages,
            "prefix_tokens": args.prefix_tokens,
            "probe_prefix": args.probe_prefix,
            "kv_bytes_per_token": llm.engine.kv_bytes_per_token,
            "kv_cache_mib": (
                args.page_size * args.num_pages * llm.engine.kv_bytes_per_token / (1024 * 1024)
            ),
            "summary": summarize(repetitions, f"{args.probe_prefix}_probe"),
            "repetitions": repetitions,
        }
        print("EVICTION_BENCHMARK_RESULT=" + json.dumps(result, sort_keys=True))
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
