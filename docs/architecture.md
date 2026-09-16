# Architecture

Design under the **pure Rust** principle. Status: policies fixed; GPU backend
implemented and verified on both HIP/ROCm and Vulkan — quantized GEMM, GDN
AR/chunked, grouped MoE, element-wise kernels, and a GPU-resident decode
frame for qwen4exp (see [decisions.md](decisions.md) ADR-0009/0011/0017).

## Data Flow

```mermaid
flowchart LR
    A[GGUF loader\nsplit, metadata, quant blocks] --> B[Model graph\nqwen35/qwen4exp]
    B --> C[Scheduler\nubatch, KV/GDN state, offload]
    C --> D[matmul dispatch\nAccelerator trait]
    D --> E1[CPU\npure Rust reference]
    D --> E2[GPU\nrawhip (HIP) / rawvk (Vulkan)]
    C --> F[Sampler + MTP draft]
    F --> G[CLI/server]
    P[profiler] -.->|debug instrumentation| C
```

Current implementation: the engine core (`crates/core`) runs both model
graphs with a runtime-injectable `Accelerator`. `llm170 infer --backend gpu`
offloads every weight projection (GDN qkv/gate/beta/alpha/out, attention
q/k/v/wo, FFN gate/up/down, output head) to quantized GEMM kernels in
`crates/backend-gpu`; weights are uploaded once and stay resident on the
device. The element-wise kernel set (`backend-gpu/src/ew.rs` — norms with
f64 sequential accumulation, activations, RoPE, top-k routing, GDN
conv/beta/softplus), the GDN AR decode kernel and the single-launch chunked
GDN prefill kernel, grouped MoE GEMM (prefill and batched down-projection),
and attention score/softmax/mix now also run on GPU. cubecl was removed in
favor of two independent backends: `rawhip` (HIP C++ via hipRTC + offline
code objects, full GPU-resident pipeline) and `rawvk` (GLSL/SPIR-V via ash,
matmul accelerator). See [backend-architecture.md](backend-architecture.md).
Runtime selection: `--backend gpu --gpu-runtime hip` or
`LLM170_GPU_RUNTIME=vulkan`.

qwen4exp is structured as stage modules (`core/src/qwen4exp/stages/`):
`hc`, `gdn`, `qsa`, `moe`, `ple` are free functions over
`Ctx { model, acc }` (+ `&mut SeqState4` for stateful stages), with dispatch
variants for same-input projection groups and per-expert paired rows.
`Engine4` retains forward/prefill chunking/decode timing only. GPU memory is
owned exclusively by the buffer arena (`backend-gpu/src/rawhip/mod.rs`,
ADR-0014): weights live in `WeightStore` behind a `WRef` enum that makes
host-fallback misuse unrepresentable, and `ScratchPool` retains every
transient upload — nothing is ever freed (VRAM bounded by accounting, not
by frees). `llm170 check` runs the three-stage verification path
(tensor scan, GPU-vs-CPU GEMM at t in {1, 64, 1024}, long-chunk smoke)
in debug or release builds.

Decode residency: qwen4exp decodes through a GPU-resident **frame**
(`core/src/qwen4exp/frame.rs`, default on — ADR-0017): activations live in
device buffers for the whole step, kernels chain by handle (hc → GDN → MoE →
head), per-sequence handle sets support parallel decode, and the PLE hash
(host) plus QSA attention (value bridge) are the only per-step crossings
(~14 syncs/step, down from ~600). If residency cannot be established (small
device regions), the engine permanently falls back to the per-op value path.

The HTTP server (`llm170 serve`) runs a llama.cpp-style slot scheduler:
decode-first budgeting, prefill in 1,024-token chunks in the remaining
budget, slots returned on completion, and client disconnects (SSE flush
failure) cancel the job.

## Device Measurement (replaces the mode system, 2026-09-13)

There is no hardware-profile flag. ADR-0002's `--mode` set two environment
knobs, of which only the prefill chunk had a reader, and the decode frame
already capped that; the flag was measured to be inert and was removed
(plans/64).

Instead the runtime measures the device once at accelerator init and routes
on the result:

- `hipDeviceGetName` / `hipMemGetInfo` — identity, free and total device
  memory.
- Host↔device transfer bandwidth (three 64 MiB round trips, pageable) — the
  UMA vs discrete signal: unified memory reaches memory-bandwidth rates,
  PCIe cards do not, so the same residency logic adapts without a flag.
- `wmma_probe` — the attention WMMA tile path is only taken when the probe
  reproduces the CPU mirror (otherwise the scalar `wk8` path runs).

Reported as one line at startup, e.g.

```
# device: Radeon 8060S | mem free=..GiB total=..GiB | h2d=..GB/s d2h=..GB/s | wmma=ok
```

Everything else already routes on measurement or capability: the
device-resident decode frame falls back to the value path when its buffers
cannot be allocated, the weight store derives its budget from the measured
total, and kernel variants are chosen per shape (GEMV vs GEMM by token
count, tile vs accumulator by `n_in`).

## Backend Strategy

1. **CPU backend (pure Rust)** — reference implementation and the portability
   baseline. Ground truth for all golden tests; the full stack verifies
   without a GPU.
2. **GPU backends (post-cubecl)** — `rawhip`: HIP C++ kernel strings
   JIT-compiled via hipRTC (`rawhip/kernels.rs::SRC`) plus optional offline
   code objects; `rawvk`: GLSL compute shaders precompiled to SPIR-V
   (`rawvk/spv/*.comp` via `scripts/build_spv.py`). Quant bytes travel as
   u32 words and kernels unpack with shifts/masks (llama.cpp CUDA
   convention). Both backends mirror the CPU W4A8 integer arithmetic;
   see backend-architecture.md for the kernel contract and verification
   gates.
3. **Profiler** — span/event macros, collected in debug builds, zero-cost in
   release (`profile` feature re-enables).

CUDA-specific code remains prohibited until the CMP 170HX arrives (cannot be
verified on this machine); the rawhip path carries over directly.

## FP Semantics and the cmp-stock Rules

Rust is strict FP by default (no implicit FMA contraction). Therefore:
- **No `f32::mul_add` in hot paths** (explicit FMA = up to 32x penalty under
  cmp-stock throttling).
- No fast-math compiler flags.
- These rules keep one kernel set shareable across hardware; a full-rate
  variant (mul_add allowed) is added after unlock measurements exist.
- Verification duty: confirm generated code contains no FMA (SASS/SPIR-V dump)
  — to be folded into the profiler/CI procedure. GPU accumulation order
  matches the CPU reference (block-sequential, element-sequential per row), so
  CPU and GPU greedy streams agree token-for-token.

## Required Kernel Surface (both models)
`gated_delta_net` (fused AR + chunked), `ssm_conv`, `solve_tri`/`cumsum`/`tri`
(chunked GDN), `l2_norm`, `top_k` (QSA/MoE), `argsort_top_k` (MoE routing),
`mul_mat_id` (expert GEMM), IMROPE (sections [11,11,10,0]), K-quant dequant
GEMM (Q4_K/Q5_K/Q6_K/Q3_K/Q8_0/IQ4_XS/IQ4_NL/IQ3_S — implemented in
`crates/backend-gpu`: decode-shaped k-lane/token-tile variants plus
token-block batched prefill), `get_rows` (PLE 20M-row table, offloaded).
PLE hashing (host u64) ports directly to CPU Rust.
The rest are standard element-wise ops (mul/silu/rms_norm/softmax/...).
Details live in local research notes (`source/research/`, untracked).

## Memory / Scheduler Design

- Hybrid cache: per-layer KV (full-attn layers only) vs GDN S state, plus
  rollback snapshot rows.
- Offload planner: per-tensor CPU/NVMe placement (BAR1 64 MiB, PCIe 0.85 GB/s
  assumed). The PLE table is the archetype.
- MTP draft (27B) — qwen4exp ships no MTP layers in GGUF.

## Staged Plan

1. ~~Workspace scaffold + GGUF v3 parser + `gguf-dump` parity~~ done.
2. ~~CPU backend + full qwen35 inference (incl. quantized dequant) — token
   parity with llama.cpp~~ done (verification matrix in `scripts/verify.py`).
3. ~~Profiler v1~~ done.
4. ~~qwen4exp (HC/QSA/PLE/MoE) CPU path — PLE NVMe offload mandatory~~ done
   (long-prompt matrix token-exact, incl. parallel sequences).
5. GPU backend: ~~quantized GEMM resident (HIP + Vulkan)~~ done → ~~GDN
   AR/chunked + attention + MoE + element-wise kernels~~ done → full
   GPU-resident forward (qwen4exp decode frame done and default-on; qwen35
   frame + PLE/QSA frame bridges remain) → prefill throughput (batched GEMM
   landed 2026-09-02; see [benchmarks.md](benchmarks.md)).
6. On CMP arrival: unlock/throttle measurements → `cmp-unlocked` kernel
   variants.
