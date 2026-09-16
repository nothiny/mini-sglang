# MoE Optimizations

Mini-SGLang's fused Qwen3 MoE backend supports reusable workspaces, route-calibrated kernel
selection, native FP8 Tensor Core execution, pipelined CPU expert residency, and two Expert
Parallel dispatch modes. Options that change numerical behavior or placement remain explicit.

## 1. Single-GPU execution

Temporary routing, alignment, and expert activation tensors are kept in a grow-only workspace and
reused by sequential MoE layers. A direct route-by-route Triton kernel avoids expert sorting and
block padding for small token counts. It can be enabled with `--moe-small-m-threshold`; optional
first-use autotuning measures direct versus grouped execution for small M and only selects direct
when it wins by at least 5%. For prefill it compares 16-, 32-, and 64-row grouped configurations
across several N/K/group shapes. Trials use the real routed expert IDs, so the selected kernel is
calibrated against observed route skew and padding. Choices are remembered per tensor shape and
power-of-two M bucket.

```bash
python -m minisgl --model Qwen/Qwen3-30B-A3B \
  --moe-small-m-threshold 1 \
  --moe-autotune
```

Use `--disable-moe-workspace-cache` for an allocation ablation. Autotuning is skipped during CUDA
Graph capture and expert offload, and its one-time measurements add startup latency.

### Native FP8 Tensor Cores

`--moe-expert-quantization fp8` stores E4M3 expert weights with per-output-channel FP32 scales.
For grouped execution, each activation matrix is dynamically quantized per row by a Triton kernel;
GEMMs then run FP8 x FP8 with FP32 accumulation and apply both scales after the reduction. Unlike
the INT8 path, this does not dequantize weights to BF16 before `tl.dot`. At M=1, the backend instead
uses the decode-oriented direct kernel over FP8 weights, avoiding activation quantization and poor
Tensor Core occupancy.

```bash
python -m minisgl --model Qwen/Qwen3-30B-A3B \
  --moe-expert-quantization fp8
```

On an RTX 5070 synthetic Qwen3-30B-A3B layer (30 samples), median hybrid FP8 latency was `202 us`,
`727 us`, `1.16 ms`, and `1.30 ms` at M=`1,16,64,256`, versus BF16 medians of `201 us`, `1.23 ms`,
`2.02 ms`, and `2.14 ms`. M=1 was effectively neutral and M=`16-256` was about `1.65-1.74x`
faster. Re-run the benchmark because clocks and route distributions materially affect these
numbers. Quantized model quality must also be evaluated on the target workload.

## 2. Expert Parallel

Expert Parallel (EP) stores a disjoint expert range on each rank. Because the existing TP stack
presents the same token rows to every rank, token `i` is assigned to origin rank `i % EP-size`;
every token/expert route is therefore sent and evaluated exactly once. Dispatch is reversed after
expert computation, followed by an All-Reduce that restores the replicated hidden states expected
by attention layers.

```bash
python -m minisgl --model Qwen/Qwen3-30B-A3B \
  --tp 4 \
  --expert-parallel-size 4
```

Large collectives are asynchronous. Routes whose destination is the current rank are copied before
launch and evaluated while remote transfers are in flight. Expert IDs and router weights share one
FP32 metadata collective, reducing dynamic dispatch from three payload All-to-All operations to
two. Disable overlap for an ablation with `--disable-moe-communication-overlap`.

Dynamic mode sends compact variable-sized messages but reads the small split vector on the host.
Static mode packs routes into equal-sized destination buckets entirely on device. It sends padding
for unused capacity, but removes that host synchronization and keeps shapes stable for CUDA Graphs:

```bash
python -m minisgl --model Qwen/Qwen3-30B-A3B \
  --tp 4 \
  --expert-parallel-size 4 \
  --moe-expert-parallel-dispatch static
```

Contiguous placement is the default. `--moe-expert-placement round-robin` spreads neighboring
expert IDs across ranks, useful when router hotspots are clustered. A profiled hot set can also be
replicated on every rank; replicated routes stay on their token's origin rank and avoid dispatch
while distributing load:

```bash
python -m minisgl --model Qwen/Qwen3-30B-A3B \
  --tp 4 --expert-parallel-size 4 \
  --moe-expert-placement round-robin \
  --moe-replicated-experts 3,17,42
```

Replicas consume one additional expert's weights per listed ID on every EP rank. Select IDs from a
representative route trace; blindly replicating experts can reduce usable KV-cache memory.

Current constraints:

- EP size is either `1` or equal to TP size, and the expert count must be divisible by it.
- EP uses PyTorch NCCL All-to-All, so PyNCCL is disabled. Dynamic dispatch disables CUDA Graphs;
  static dispatch permits capture but still requires graph-capable NCCL/PyTorch versions.
- Attention, embeddings, and the LM head remain tensor parallel; only MoE expert weights use EP.

## 3. INT8 experts and CPU/GPU residency

`--moe-expert-quantization int8` applies symmetric per-output-channel weight-only quantization.
INT8 and FP8 experts are quantized layer-by-layer while the checkpoint is assembled, and quantized
tensors and scales are loaded directly into the model. This avoids retaining a full BF16 copy of
all experts in host memory and avoids accidentally converting quantized state back to BF16.

CPU offload keeps the full checkpoint in ordinary host memory and allocates bounded pinned staging
slots matching each layer's GPU-cache capacity; it does not pin every expert. Each layer owns a
fixed-size GPU LRU. Missing experts move through staging on a dedicated CUDA stream; per-slot events
prevent staging reuse or GPU eviction while a prior copy/kernel is live. Resident routes can compute
immediately. Missing experts are consumed in grouped micro-waves as their ready events arrive,
allowing later H2D copies to overlap without paying one kernel launch per expert. If a prefill batch
routes to more experts than fit, waves are accumulated back into their original token rows.

```bash
python -m minisgl --model Qwen/Qwen3-30B-A3B \
  --moe-expert-quantization fp8 \
  --moe-expert-offload \
  --moe-expert-cache-size 8 \
  --moe-expert-offload-wave-size 8
```

Use `--disable-moe-expert-offload-overlap` for a serialized baseline. PCIe overlap is workload and
GPU dependent: too-small micro-waves add launch/padding overhead, while too-large waves expose more
copy time. The default is `8` because smaller groups usually regressed cold top-8 routing on the
tested RTX 5070; benchmark the wave size on the target token and cache-hit regime.

The engine reserves persistent GPU-cache bytes before sizing the KV cache. Dynamic residency is not
CUDA-Graph compatible. Offload is primarily a capacity optimization: cache misses cross PCIe and
can substantially increase latency, so measure cache size against the workload's active-expert set.

## Microbenchmark

The synthetic benchmark defaults to Qwen3-30B-A3B dimensions (`128` experts, top-`8`, hidden size
`2048`, MoE intermediate size `768`) and covers decode and prefill token counts:

```bash
python benchmark/offline/bench_moe.py \
  --token-counts 1,4,8,16,64,256 \
  --moe-small-m-threshold 1 \
  --include-autotune \
  --include-int8 \
  --include-fp8
```

Add `--include-offload --offload-cache-size 8 --offload-wave-size 8` to compare overlapped and
serialized residency and report hits, misses, and host-to-device bytes. Each row also reports active
experts and peak routes per expert, which helps identify route skew and select replica candidates.
Results are emitted as one `MOE_BENCHMARK_RESULT=<json>` record for experiment scripts.

In an earlier RTX 5070 run, workspace reuse reduced temporary allocations from `2100` to `6` across
six token sizes. The fixed direct `M=1` choice varied substantially and was slower in that run,
which is why its default threshold is zero and the autotuner requires a 5% win. These are
kernel-level measurements rather than end-to-end throughput; rerun all ablations on the deployment
GPU before selecting defaults.

## Download and run Qwen3 MoE

The repository's reused virtual environment already contains `huggingface_hub`; this command uses
its module entry point and does not require another CLI installation:

```bash
.venv/bin/python -m huggingface_hub.commands.huggingface_cli download \
  Qwen/Qwen3-30B-A3B \
  --local-dir /home/yzd/models/Qwen3-30B-A3B
```

For a single 12 GB RTX 5070, start with FP8 expert offload and an eight-expert cache:

```bash
.venv/bin/python -m minisgl \
  --model /home/yzd/models/Qwen3-30B-A3B \
  --dtype bfloat16 \
  --moe-expert-quantization fp8 \
  --moe-expert-offload \
  --moe-expert-cache-size 8 \
  --memory-ratio 0.8
```

Qwen3-4B is a dense model and cannot exercise these MoE paths. Qwen3-30B-A3B has 128 experts and
activates eight per token, matching the benchmark defaults. Its download is much larger than its
3.3B activated-parameter count; ensure sufficient disk and roughly 32 GB of available host memory
for the streamed FP8/offload configuration.
