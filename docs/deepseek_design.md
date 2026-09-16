# Mini-DeepSeek Design

Status: runs DeepSeek-V2-Lite end to end on a single 12 GiB GPU.

Mini-DeepSeek is the composition of two branches:

- **mini-mla** (`docs/mla_design.md`): the MLA attention path, latent KV cache,
  paged FlashInfer backend, and the `DeepseekV2ForCausalLM` model.
- **mini-moe** (`docs/moe_optimizations.md`): the fused expert kernel, FP8/INT8
  expert quantization, Expert Parallel, and CPU expert offload.

This document only covers the glue: what DeepSeek needs beyond the two halves,
and the measured single-GPU run.

## What DeepSeek-V2-Lite adds

Config (27 layers, hidden 2048, 16 heads, `kv_lora_rank=512`,
`qk_rope_head_dim=64`, `v_head_dim=128`):

```text
n_routed_experts       = 64
n_shared_experts       = 2
num_experts_per_tok    = 6
moe_intermediate_size  = 1408
first_k_dense_replace  = 1     # layer 0 is a dense FFN, layers 1..26 are MoE
moe_layer_freq         = 1
scoring_func           = softmax
topk_method            = greedy
norm_topk_prob         = False
routed_scaling_factor  = 1.0
```

Two pieces that Mini-SGLang's Qwen3 MoE does not have:

1. **Shared experts**: a dense gated MLP applied to every token, added to the
   routed-expert output. Implemented by `DeepseekV2MoE` in
   `models/deepseek_v2.py`, which reuses `MoELayer` for the routed part and a
   plain `GatedMLP` for `n_shared_experts * moe_intermediate_size`.
2. **Layer selection**: `first_k_dense_replace` / `moe_layer_freq`. Layer 0 uses
   a dense `GatedMLP`, the rest use `DeepseekV2MoE`.

`ModelConfig` now reads `n_routed_experts` (DeepSeek's name) and defines
`is_moe` as `num_experts > 0`, so DeepSeek selects the fused MoE backend
automatically.

## Weight loading

`models/weight.py` needed two adjustments:

- MLA has a standalone `self_attn.q_proj`, not a fused q/k/v projection, so it
  must not be held in the q/k/v merge buffer forever: `_get_merge_info` skips the
  `.q_proj` merge when the model is MLA.
- `self_attn.kv_b_proj` is column-parallel over heads, so `.kv_b_proj` was added
  to `_SPLIT_DIM_0` (a no-op at TP=1, correct for TP>1).

The rest already lines up: `kv_a_proj_with_mqa` / `kv_a_layernorm` are
replicated, `o_proj` / `down_proj` are row-parallel, routed experts are packed
into `[E, ...]` by the existing expert stacker, and `shared_experts.gate_proj` /
`up_proj` merge into `shared_experts.gate_up_proj`.

## Running on 12 GiB

The 15.7B-parameter checkpoint is ~31 GiB in BF16, so the experts stay on host
and the GPU holds the dense weights, a per-layer expert LRU, and the latent KV
cache. MLA helps here: the KV cache is ~31 KB/token instead of ~276 KB for the
MHA-equivalent, which is what leaves room for the expert cache at all.

```bash
python benchmark/offline/bench_deepseek.py \
  --model /home/yzd/models/DeepSeek-V2-Lite \
  --expert-offload --expert-cache-size 8 --expert-quantization none
```

Measured on an RTX 5070 (12 GiB), BF16 experts + CPU offload:

```text
pool: MLAKVCache   backend: MLAttentionBackend   num_pages: 92437
CUDA graphs disabled (MLA)
load_seconds = 106
greedy output:
  "The capital of France is" -> " a city of culture, history, and art. ..."
  "1 + 1 ="                  -> " 2 ..."
```

The first request (71 s) is dominated by cold expert loading over PCIe; later
requests are much faster. `bench_deepseek.py` reports cold and warm latency and
the steady-state tokens/s. Use `--expert-quantization fp8` to halve host memory
and speed up the H2D copies at some accuracy cost.

## Remaining work

- MLA CUDA graph capture (currently disabled) and TP sharding of the latent.
- Bit-exact end-to-end comparison against `transformers` (attention is already
  exact; the fused MoE is validated separately against a torch reference).
- FP8 KV cache and the DeepSeek-V3 `noaux_tc` router.
