# Mini-MLA Design

Status: M0-M2 implemented. See section 10 for per-milestone status.

Mini-MLA adds Multi-head Latent Attention (MLA) to Mini-SGLang so it can run
DeepSeek-V2-style models. It is the attention half of a future **Mini-DeepSeek**
stack: Mini-MLA (attention) + the `mini-moe` branch (experts, FP8, CPU offload) =
a DeepSeek-V2-Lite serving path on a single consumer GPU.

Target model: **DeepSeek-V2-Lite** (27 layers, 64 routed + 2 shared experts, MLA).
Existing dense and Qwen3-MoE paths must stay unchanged; MLA is selected by model
architecture, not by a global flag.

## 1. Goals and scope

In scope:

- MLA math and KV layout, naive (up-projected) and absorbed (latent) forms.
- A latent KV pool, paged attention, radix-cache compatibility.
- DeepSeek-V2-Lite weight loading and a `DeepseekV2ForCausalLM` model class.
- Tensor-parallel sharding and CUDA-graph decode.
- Correctness against `transformers` and against a hand-written reference.

Out of scope (explicitly deferred):

- DeepSeek-V3 `noaux_tc` routing and the `group_limited_greedy` variant.
- FP8/FP4 attention, sparse MLA, MTP (multi-token prediction).
- Training.
- HiCache over the MLA layout (follow-up; see section 7).

The combined Mini-DeepSeek milestone (shared experts, MoE, FP8, offload) is
section 10, milestone M5.

## 2. Why MLA

Decode is memory-bandwidth bound, and the KV cache is the dominant decode memory.
Standard MHA/GQA caches full K and V per head. MLA keeps a **low-rank latent** per
token plus a **decoupled RoPE key** shared across heads.

For DeepSeek-V2-Lite (`kv_lora_rank=512`, `qk_rope_head_dim=64`, 16 heads,
`qk_nope_head_dim=128`, `v_head_dim=128`, 27 layers):

| Cache | Elements / token / layer | BF16 bytes / token |
|---|---:|---:|
| MLA (`c_KV` + `k_pe`) | `512 + 64 = 576` | 31,104 (27 layers) |
| MHA-equivalent (16 heads × (192 K + 128 V)) | `5120` | 276,480 |

That is a **8.9x** reduction, which is what makes a 160K-context DeepSeek model
servable at all. The cost moves to compute: attention now runs over an effective
head dimension of `kv_lora_rank + qk_rope_head_dim` (576), which is why the
"absorbed" form matters.

## 3. The computation

Config fields (added to `ModelConfig`, see section 6):

```text
q_lora_rank        # None for V2-Lite; 1536 for V2
kv_lora_rank       # 512
qk_nope_head_dim   # 128
qk_rope_head_dim   # 64
v_head_dim         # 128
```

Projections (per layer):

```text
q_a  = W_DQ @ h                     # only if q_lora_rank is set
q_a  = RMSNorm(q_a)                 # q_a_layernorm
q    = W_UQ @ q_a  (or W_Q @ h)
q_nope, q_pe = split(q, [heads*qk_nope_head_dim, heads*qk_rope_head_dim])
q_pe = RoPE(q_pe)

c_KV = W_DKV @ h                    # kv_lora_rank
c_KV = RMSNorm(c_KV)                # kv_a_layernorm
k_pe = W_KR @ h                     # qk_rope_head_dim, one vector per token
k_pe = RoPE(k_pe)
cache(c_KV, k_pe)                   # <- everything that is cached
```

`W_DKV` and `W_KR` are fused in the checkpoint as `kv_a_proj_with_mqa`
(output `kv_lora_rank + qk_rope_head_dim`); `W_UK` and `W_UV` are fused as
`kv_b_proj` (output `heads * (qk_nope_head_dim + v_head_dim)`).

Two equivalent evaluation strategies:

- **Naive**: up-project `k_nope = W_UK @ c_KV`, `v = W_UV @ c_KV`, concatenate
  `k_nope` with the broadcast `k_pe`, then run standard attention over
  `qk_nope + qk_rope` query/key dims and `v_head_dim` values.
- **Absorbed**: fold `W_UK` into the query and `W_UV` into the output projection,
  then attend directly over the latent:

  ```text
  q_absorbed = W_UK^T @ q_nope            # [heads, kv_lora_rank]
  scores     = q_absorbed @ c_KV^T + q_pe @ k_pe^T
  z          = softmax(scores) @ c_KV     # [heads, kv_lora_rank]
  o          = W_O_absorbed @ z           # W_UV folded into W_O
  ```

Prefill is compute-bound and the naive form has better FLOPS shape; decode is
memory-bound and the absorbed form avoids materializing K/V. The design supports
both and picks per phase (section 5).

## 4. Data layout

Current MHA pool (`kvcache/mha_pool.py`):

```text
[2, layers, pages, page_size, local_kv_heads, head_dim]
```

`MLAKVCache` keeps two buffers, because FlashInfer's
`BatchMLAPagedAttentionWrapper` takes the latent and the rope key separately:

```text
ckv: [layers, tokens, kv_lora_rank]
kpe: [layers, tokens, qk_rope_head_dim]
```

Page size 1 initially, matching the rest of the engine; later page sizes are a
pure indexing change. `store_mla(c_KV, k_pe, out_loc, layer_id)` writes both.

Per page (page_size = 1, BF16): `576 * 2 = 1152` bytes. The engine's KV sizing
(`engine.py::_get_kv_bytes_per_token`) becomes layout-aware:

```text
mha_bytes_per_token = 2 * head_dim * kv_heads_per_rank * itemsize * layers
mla_bytes_per_token = (kv_lora_rank + qk_rope_head_dim) * itemsize * layers
```

## 5. Attention backend

FlashInfer 0.6.17 already ships what we need:

- `flashinfer.mla.BatchMLAPagedAttentionWrapper` for the **absorbed** paged path
  (both decode and prefill; the former "naive prefill" path is not needed). Its
  `q_nope` input is the already-absorbed query `q_abs` `[tokens, heads,
  kv_lora_rank]`, it reads the two caches above, and it returns the latent
  output `z` `[tokens, heads, kv_lora_rank]`. The layer then applies `W_UV` and
  `W_O`. The caller still uses `sm_scale = (qk_nope + qk_rope) ** -0.5`, i.e. the
  head dimension before absorption.
- `flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper` (naive,
  `head_dim_qk = 192`, `head_dim_vo = 128`) is kept only as a possible future
  prefill path; it is not wired in.

A `MLAttentionBackend` wraps these and mirrors the existing `BaseAttnBackend`
metadata/capture lifecycle (`attention/base.py`, `attention/fi.py`). Because the
current `forward(q, k, v, layer_id, batch)` interface is K/V-centric, MLA needs a
small interface extension rather than a full rewrite. Two options:

1. Add an `MLAttentionBackend` with its own `forward_mla(...)` and let
   `MLAttentionLayer` call it directly.
2. Generalize `BaseAttnBackend.forward` to take the projection outputs.

Option 1 keeps the existing backends untouched. A pure-PyTorch/torch fallback is
implemented first anyway, for correctness tests and CPU debugging, so the design
does not depend on FlashInfer's MLA kernel at M0/M1.

RoPE detail: only `q_pe` and `k_pe` are rotated (rotary dim = `qk_rope_head_dim`, 64);
`q_nope` and `c_KV` are not. Two concrete constraints, both verified against
the installed dependencies:

- DeepSeek's reference applies RoPE with **adjacent pairing**
  (`view_as_complex(x.reshape(*x.shape[:-1], -1, 2))`, the interleaved / GPT-J
  layout). FlashInfer's `apply_rope_with_cos_sin_cache_inplace` defaults to
  `is_neox=True`, which is the split-half / Neox layout; the two do **not**
  match. MLA must pass `is_neox=False` (or ship its own rotated-pair kernel).
- `RotaryEmbedding` asserts `rotary_dim == head_size` and
  `head_size in [64, 128, 256, 512]`, so RoPE must be applied to the 64-dim
  `q_pe` / `k_pe` slices, never to the concatenated 192-dim query.

DeepSeek-V2-Lite also uses **YaRN**:

```text
rope_scaling = {type: yarn, factor: 40, original_max_position_embeddings: 4096,
                beta_fast: 32, beta_slow: 1, mscale: 0.707, mscale_all_dim: 0.707}
```

The existing `_get_rope` yarn branch implements the frequency correction. HF's
generic YaRN path additionally derives an `attention_factor` from
`mscale`/`mscale_all_dim` and folds it into the cos/sin cache. For both V2 and
V2-Lite `mscale == mscale_all_dim`, so
`attention_factor = get_mscale(40, 0.707) / get_mscale(40, 0.707) = 1.0` and drops
out. MLA should still implement it (or assert it is `1.0`) so other YaRN configs
are not silently wrong.

## 6. Model integration

- `models/config.py`: add `q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`,
  `qk_rope_head_dim`, `v_head_dim`, plus `is_mla` and the DeepSeek MoE fields
  (`n_routed_experts`, `n_shared_experts`, `first_k_dense_replace`,
  `moe_layer_freq`, `topk_method`). Populate from `PretrainedConfig` in
  `from_hf`.
- `layers/attention.py`: add `MLAttentionLayer` (query split, `kv_a_layernorm`,
  RoPE, cache store) alongside the existing `AttentionLayer`.
- `models/utils.py`: add `MLAttention` (projections + `MLAttentionLayer`),
  reusable by the DeepSeek model.
- `models/deepseek_v2.py`: `DeepseekV2Model` / `DeepseekV2ForCausalLM`.
  - Dense FFN for the first `first_k_dense_replace` layers, MoE after.
  - Shared experts are part of M5; until then the model can be validated at the
    attention level (section 9).
- `models/register.py`: register `DeepseekV2ForCausalLM`.
- `kvcache/mla_pool.py`: `MLAKVCache` implementing `BaseKVCachePool` with a
  `store_kv` that writes `(c_KV, k_pe)` from `batch.out_loc`.
- `kvcache/__init__.py`: route pool creation on `config.is_mla`.
- `models/weight.py`: the DeepSeek merges. `kv_a_proj_with_mqa` is a single
  checkpoint tensor; `kv_b_proj` must be split into `W_UK` and `W_UV` during
  loading; `q_a_proj`/`q_b_proj` chain when `q_lora_rank` is set.

The radix cache and page table are token-index based and layout-agnostic, so they
should work unchanged once the pool is registered.

## 7. Interactions

- **Radix cache**: no change expected; the pool hides the layout.
- **HiCache**: `CacheTransferManager` hard-codes the MHA transpose
  (`[2, layer, page, ...] <-> [page, 2, layer, ...]`). MLA needs a new
  page-major path. Deferred: tracked as `mini-hicache-mla`, ideally after M2.
- **CUDA graph**: the absorbed decode wrapper supports graph capture; mirror
  `FICaptureData`/`prepare_for_capture`/`prepare_for_replay`.
- **TP/EP**: `num_qo_heads` shards by TP as usual. `kv_lora_rank` and the
  `k_pe` head are replicated (or sharded) consistently across ranks; this needs a
  decision in M3.
- **Chunked prefill / overlap scheduling**: unchanged; MLA only changes the
  attention op.

## 8. Weight loading and sharding

DeepSeek-V2-Lite projection names:

```text
q_proj / q_a_proj / q_b_proj
kv_a_proj_with_mqa
kv_b_proj
o_proj
q_a_layernorm / kv_a_layernorm
```

Sharding rules:

- `q_proj` / `q_b_proj`: column-parallel over heads (reuse `LinearColParallelMerged`).
- `kv_a_proj_with_mqa`: output is `kv_lora_rank + qk_rope_head_dim`; both parts are
  replicated across TP ranks (the latent is not per-head).
- `kv_b_proj`: output is `heads * (qk_nope + v)` and shards over heads.
- `o_proj`: row-parallel, with `W_UV` absorption handled in M2/M3.
- `Embedding` / `LM head`: `ParallelLMHead` + `VocabParallelEmbedding` as today.

## 9. Validation

- **M0**: naive attention vs a hand-written `torch` reference on random tensors;
  absorbed output must equal naive (within tolerance). Both on CPU.
- **M1**: a tiny synthetic DeepSeek-shaped model (2 layers, small dims) end to end,
  compared token-for-token against a reference implemented with the same weights.
- **M2**: paged MLA vs unpaged reference at several sequence lengths and batch
  sizes, including a partial-cache prefix hit; `check_integrity()` passes.
- **M3**: load real DeepSeek-V2-Lite weights; compare the attention output at each
  layer against `transformers.models.deepseek_v2` on the same hidden states, then
  compare greedy tokens on a few prompts (full-model match requires shared experts,
  so the token-level check may land in M5).
- **M5**: DeepSeek-V2-Lite end to end vs `transformers`, plus the KV-memory and
  throughput numbers.

## 10. Milestones

| # | Deliverable | Acceptance | Status |
|---|---|---|---|
| M0 | MLA math + torch reference (unpaged) | absorbed == naive; reference matches HF attention math | done |
| M1 | `MLAttention`, `MLAKVCache`, synthetic model | tiny model forward matches reference | done (`MLAttention` == HF attention) |
| M2 | Paged MLA via FlashInfer, radix integration | token match; page layout tests; `check_integrity()` | done (paged == reference; engine E2E) |
| M3 | DeepSeek-V2-Lite attention weights, TP, CUDA graph | per-layer attention matches `transformers`; graph replay works | not started (CUDA graph deferred) |
| M4 | Benchmarks + docs | KV bytes/token, max context, decode tokens/s tables | not started |
| M5 | Mini-DeepSeek: shared experts + MoE + FP8/offload | full V2-Lite token match vs `transformers`; single-GPU run | `mini-deepseek` branch |

M0-M2 are the `mini-mla` branch. M5 is the `mini-deepseek` branch that merges
`mini-mla` and `mini-moe`.

## 11. Metrics

- KV bytes/token and pages: MLA vs the MHA-equivalent (expected ~8.9x).
- Maximum context length and concurrent requests at a fixed VRAM budget.
- Decode latency / tokens-per-second at several batch sizes and context lengths.
- Prefill latency, naive vs absorbed.
- Correctness: exact token match against `transformers`; attention-output max error.

## 12. Risks

| Risk | Mitigation |
|---|---|
| RoPE convention mismatch (interleaved vs split-half) | M0 unit test; MLA passes `is_neox=False` |
| YaRN `attention_factor` missing from `_get_rope` | implement it (no-op for V2/V2-Lite) or assert it is `1.0` |
| FlashInfer MLA API/version drift | keep the torch fallback as the correctness oracle |
| `kv_lora_rank` TP sharding ambiguity | decide in M3, document, add a test |
| Weight-name/merge complexity | reuse `_get_merge_info`; add loader tests first |
| HiCache layout coupling | keep MLA off the HiCache path until M2 lands |
| Naive vs absorbed numerical drift | compare both in M0/M2 with a tolerance |

## 13. Reading list

- DeepSeek-V2 paper, section on MLA (`arXiv:2405.04434`).
- DeepSeek-V3 technical report (MLA + auxiliary-loss-free routing).
- FlashInfer MLA blog and `BatchMLAPagedAttentionWrapper` docs.
- SGLang and vLLM MLA implementations (absorbed vs naive placement).
- `transformers/models/deepseek_v2` as the correctness reference.
