# Radix Cache Eviction Policies

Mini-SGLang separates Radix tree correctness from victim selection. The Radix cache owns
reference counting, protected-node filtering, tree mutation, and KV page release. An eviction
policy only receives immutable metadata for eligible, unreferenced leaf nodes and chooses one
`node_id`.

## Available policies

| Name | Selection rule |
| --- | --- |
| `lru` | Least recently accessed node. This is the default and preserves the original behavior. |
| `lfu` | Lowest access count, with recency as the tie-breaker. |
| `lru-k` | Oldest K-th access; nodes with fewer than K observations are evicted first. |
| `frequency-decay` | Lowest exponentially decayed frequency. |
| `cost-aware` | Lowest estimated reuse probability times recompute cost per KV byte. |
| `adaptive` | Online regret learner over a configurable set of policy experts. |

The cost-aware estimate is:

```text
recompute_cost = segment_tokens * (prefix_depth + segment_tokens / 2)
retention_value = reuse_probability * recompute_cost / kv_bytes
```

`adaptive` asks every expert for a proposal, samples an expert using learned weights, and records
the resulting eviction in a bounded ghost history. If the evicted prefix is requested again, the
chosen expert is penalized in proportion to the estimated recomputation regret. This preserves all
expert histories while adapting online.

## Command-line configuration

```bash
python -m minisgl --model Qwen/Qwen3-0.6B \
  --cache radix \
  --cache-eviction-policy lru
```

```bash
python -m minisgl --model Qwen/Qwen3-0.6B \
  --cache radix \
  --cache-eviction-policy lru-k \
  --eviction-k 2
```

```bash
python -m minisgl --model Qwen/Qwen3-0.6B \
  --cache radix \
  --cache-eviction-policy frequency-decay \
  --eviction-half-life 60
```

```bash
python -m minisgl --model Qwen/Qwen3-0.6B \
  --cache radix \
  --cache-eviction-policy adaptive \
  --adaptive-experts lru,lfu,cost-aware \
  --adaptive-learning-rate 0.05 \
  --eviction-ghost-capacity 4096
```

Use `--adaptive-seed` to make expert sampling reproducible. `--eviction-half-life` is expressed in
seconds and is shared by `frequency-decay` and the recency component of `cost-aware`.

## Python and class injection

Policies can be selected through `SchedulerConfig`/`LLM` keyword arguments or injected directly
for experiments:

```python
import torch

from minisgl.core import Context, set_global_ctx
from minisgl.kvcache.eviction import CostAwarePolicy
from minisgl.kvcache.radix_cache import RadixPrefixCache

set_global_ctx(Context(page_size=1))
cache = RadixPrefixCache(
    device=torch.device("cuda"),
    eviction_policy=CostAwarePolicy(half_life_seconds=60),
    total_tokens=131072,
    kv_bytes_per_token=262144,
)
```

A custom strategy implements `BaseEvictionPolicy.select_victim()`. Event callbacks provide access,
insert, split, eviction, and ghost-hit feedback. Policies must never mutate Radix nodes, reference
counts, or page allocations. The cache validates the returned ID against the current set of
unreferenced leaves before changing the tree.

Set `uses_ghost_feedback = True` on a custom policy that consumes `on_ghost_hit()` events. Policies
that do not request ghost feedback avoid prefix fingerprint scans on the matching hot path.

Node splits preserve the original node ID for the suffix so outstanding cache handles remain valid.
Consequently, `on_split(old_node_id, prefix_node_id, suffix_node_id)` may receive the same value for
`old_node_id` and `suffix_node_id`; policies should copy the old history to both resulting segments.

## Reproducible benchmark

`benchmark/offline/bench_eviction_policy.py` loads a real model, warms up both full-prefill and
prefix-hit paths, and then forces one of four 128-token prefixes out of a cache that can hold only
three. Prefix A is frequent but old; prefix C is recent but infrequent. Use `--probe-prefix A` for
a frequency-friendly trace and `--probe-prefix C` for a recency-friendly trace:

```bash
python benchmark/offline/bench_eviction_policy.py \
  --model Qwen/Qwen3-4B \
  --policy adaptive \
  --probe-prefix A \
  --repeats 5
```

The JSON result reports per-operation latency and matched/prefill token counts, total hit rate and
throughput, victim-selection time, post-pressure residency, and adaptive expert weights. Model
loading and the initial kernel warmup are excluded from measurements.
