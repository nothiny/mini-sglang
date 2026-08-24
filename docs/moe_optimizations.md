# MoE Optimizations

Mini-SGLang's fused Qwen3 MoE backend includes three independent optimization paths. All options
are disabled or conservative by default where they change memory placement or numerical behavior.

## 1. Single-GPU execution

Temporary routing, alignment, and expert activation tensors are kept in a grow-only workspace and
reused by sequential MoE layers. A direct route-by-route Triton kernel avoids expert sorting and
block padding for small token counts. It can be enabled with `--moe-small-m-threshold`; optional
first-use autotuning measures direct versus grouped execution for small M and only selects direct
when it wins by at least 5%. For prefill it compares the 16-, 32-, and 64-row grouped Triton
configurations. Choices are remembered per shape and power-of-two M bucket.

```bash
python -m minisgl --model Qwen/Qwen3-30B-A3B \
  --moe-small-m-threshold 1 \
  --moe-autotune
```

Use `--disable-moe-workspace-cache` for an allocation ablation. Autotuning is skipped during CUDA
graph capture and expert offload, and its one-time measurements add startup latency.

## 2. Expert Parallel

Expert Parallel (EP) stores a disjoint contiguous expert range on each rank. Because the existing
TP stack presents the same token rows to every rank, token `i` is assigned to origin rank
`i % EP-size`; every token/expert route is therefore sent and evaluated exactly once. Variable-size
All-to-All dispatch is reversed after expert computation, followed by an All-Reduce that restores
the replicated hidden states expected by the attention layers.

```bash
python -m minisgl --model Qwen/Qwen3-30B-A3B \
  --tp 4 \
  --expert-parallel-size 4
```

Large dispatch collectives are launched asynchronously. Routes whose destination is the current
rank are copied before launch and evaluated while remote transfers are in flight. Disable this
overlap for an ablation with `--disable-moe-communication-overlap`.

Current constraints:

- EP size is either `1` or equal to TP size, and the expert count must be divisible by it.
- EP uses PyTorch NCCL All-to-All, so PyNCCL and CUDA graphs are disabled in this mode.
- Attention, embeddings, and the LM head remain tensor parallel; only MoE expert weights use EP.

## 3. INT8 experts and CPU/GPU residency

`--moe-expert-quantization int8` stages checkpoint experts on CPU, then applies symmetric,
per-output-channel weight-only quantization before moving them to GPU. The engine therefore does
not require enough free GPU memory for a transient BF16 copy of every expert. Scales remain FP32
and activations retain the configured model dtype.

CPU offload keeps checkpoint expert tensors in pinned host memory from load time onward. Each MoE
layer owns a fixed-size GPU LRU. Missing experts are copied on a dedicated CUDA stream; per-slot
events prevent eviction from overwriting weights still used by queued kernels. If a prefill batch
routes to more experts than fit in the cache, routes are processed in expert-sized waves and summed
back into their token rows.

```bash
python -m minisgl --model Qwen/Qwen3-30B-A3B \
  --moe-expert-quantization int8 \
  --moe-expert-offload \
  --moe-expert-cache-size 16
```

The engine reserves the persistent GPU-cache bytes before sizing the KV cache. Dynamic residency is
not CUDA-graph compatible. Offload is primarily a capacity optimization: cache misses cross PCIe
and can substantially increase latency, so the cache size should be measured against the workload's
active-expert set. Quantized model quality should be evaluated on the target workload.

## Microbenchmark

The synthetic benchmark defaults to the expert dimensions of Qwen3-30B-A3B (`128` experts,
top-`8`, hidden size `2048`, MoE intermediate size `768`) and covers decode and prefill token counts:

```bash
python benchmark/offline/bench_moe.py \
  --token-counts 1,4,8,16,64,256 \
  --moe-small-m-threshold 1 \
  --include-autotune \
  --include-int8
```

Add `--include-offload --offload-cache-size 16` to report cache hits, misses, and host-to-device
bytes. Results are emitted as a single `MOE_BENCHMARK_RESULT=<json>` record for experiment scripts.

In one RTX 5070 run (50 warmups, 30 samples, 10 inner repeats), workspace reuse reduced temporary
allocations from `2100` to `6` across the six token sizes, while median grouped-kernel latency stayed
within a few percent. The fixed direct `M=1` choice varied substantially on this consumer GPU and
was slower in that run (`316 us` versus `188 us` grouped), which is why its default threshold is
zero and the autotuner uses a 5% margin. INT8 was most useful beyond single-token decode: at `M=64`
it measured `1.14 ms` versus `1.99 ms` BF16, and at `M=256`, `1.25 ms` versus `2.08 ms`. These are
kernel-level measurements, not end-to-end model throughput; rerun the script on the deployment GPU
before choosing thresholds or quantization.
