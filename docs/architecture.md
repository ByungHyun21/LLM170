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
    D --> E2[GPU\ncubecl kernels, HIP or Vulkan runtime]
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
and attention score/softmax/mix now also run on GPU. `--gpu-runtime vulkan`
selects a wgpu/Vulkan client; the same kernel source compiles for both
runtimes. Remaining per-step host crossings in the frame: the PLE gather
(host-side hash) and QSA attention (value bridge — cache upload + readback).

qwen4exp is structured as stage modules (`core/src/qwen4exp/stages/`):
`hc`, `gdn`, `qsa`, `moe`, `ple` are free functions over
`Ctx { model, acc }` (+ `&mut SeqState4` for stateful stages), with dispatch
variants for same-input projection groups and per-expert paired rows.
`Engine4` retains forward/prefill chunking/decode timing only. GPU memory is
owned exclusively by the buffer arena (`backend-gpu/src/buffers.rs`,
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

## Mode System

`universal` / `cmp-stock` / `cmp-unlocked` — see [overview.md](overview.md).
- Mode = runtime flag + **kernel variant selection** (decomposed mul+add vs
  full-rate FMA) + memory budget profile.
- The engine core (loader, graph, scheduler, sampler) is mode-agnostic.
  Mode-dependent code is confined to kernel selection and the memory planner.
- Implemented (2026-09-01): `core/src/mode.rs` + `infer|serve --mode` sets
  the weight-residency and prefill-chunk defaults (`LLM170_W_CAP_GB`,
  `LLM170_Q4_CHUNK`); explicit env values win. Update (2026-09-02): when the
  qwen4exp decode frame is active (its default), the mode's W_CAP preset is
  skipped so the weight store derives its budget from the measured device
  region (the frame needs the full expert stacks resident). Kernel variants
  are the remaining half — `Mode` is the branch key when the cmp-stock
  kernel set lands.

## Backend Strategy

1. **CPU backend (pure Rust)** — reference implementation and `universal`
   default. Ground truth for all golden tests; the full stack verifies without
   a GPU.
2. **GPU backend (cubecl)** — kernels written in Rust via cubecl macros.
   `hanzo-cubecl-hip` JIT-compiles to gfx1151 (dev machine, ROCm); the same
   kernels compile through cubecl-wgpu/WGSL to Vulkan, and to CUDA/PTX (sm_80)
   for the CMP 170HX. Runtime selection via `--backend cpu|gpu` and
   `--gpu-runtime hip|vulkan` — build features are not used (single binary).
   Constraint discovered in practice: WGSL has no u8 buffer element, so quant
   bytes travel as u32 words and kernels unpack bytes with shifts/masks (the
   same convention as llama.cpp CUDA kernels). The HIP dialect also miscompiles
   `as i8` casts and if-expressions on the RHS of binary operators, so kernels
   use pure u32 arithmetic for all byte manipulation.
3. **Profiler** — span/event macros, collected in debug builds, zero-cost in
   release (`profile` feature re-enables).

CUDA-specific code remains prohibited until the CMP 170HX arrives (cannot be
verified on this machine); the cubecl path is expected to carry over directly.

## FP Semantics and the cmp-stock Rules

Rust is strict FP by default (no implicit FMA contraction). Therefore:
- **No `f32::mul_add` in hot paths** (explicit FMA = up to 32x penalty under
  cmp-stock throttling).
- No fast-math compiler flags.
- These rules alone make universal/cmp-stock kernel sharing work.
  `cmp-unlocked` adds a full-rate variant (mul_add allowed) after unlock
  measurements.
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
