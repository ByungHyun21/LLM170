# LLM170 Project Overview

A **pure-Rust** LLM inference engine targeting the NVIDIA CMP 170HX as its final
target. Built from scratch — no llama.cpp, no ggml.

## Goals & Priorities

1. **Priority 1 — CMP 170HX** (GA100, sm_80): maximum performance, assuming the
   40/64 GB unlock.
2. **Priority 2 — portability**: runs on arbitrary hardware (any CPU, any GPU).
   Solo development, so the universal mode doubles as the development and
   verification baseline.
3. **What happens on this machine now**: **universal-mode development**. The CMP
   170HX is not yet accessible.

## Mode System

| Mode | Target | Assumption | Kernel strategy |
|---|---|---|---|
| `universal` | Any device (CPU / generic GPU) | none | Portability first. This PC is the reference environment |
| `cmp-stock` | CMP 170HX stock (8 GB, eFUSE throttle) | no FFMA, no tensor cores | half2/INT32, decomposed FMA |
| `cmp-unlocked` | CMP 170HX unlocked (64 GB, compute unlock) | full rate allowed | Performance first (to be finalized after unlock measurements) |

- Modes are implemented as **runtime flags + kernel-variant selection**. The
  engine core (model, scheduler, loader) is mode-agnostic.
- Memory budgets per mode: stock ~7 GiB / unlocked 64 GiB — see
  [hardware/CMP 170HX](hardware/cmp170hx.md).

## Reference Models (GGUF)

| Model | Architecture | Composition | Size | Spec |
|---|---|---|---|---|
| Qwen3.8-27B | `qwen35` dense hybrid | 64 layers = 12×(3×GDN + 1×Gated Attn) + 1 MTP layer, ctx 262144, VL | Q4 16.5 / Q6 24.1 / Q8 31.4 GB | [models/qwen35.md](models/qwen35.md) |
| Qwen3.8-Flash-Next | `qwen4exp` MoE hybrid | 48 layers = 12×(3×GDN + 1×QSA), hc=4 residual, MoE 512 experts (A6B), PLE 51 B, ctx 262144 | UD-Q4 111.3 GB (4-split) | [models/qwen4exp.md](models/qwen4exp.md) |

Both models are GDN (Gated DeltaNet) linear-attention hybrids — qwen35 first
(simpler reference), qwen4exp as the extension.

## Development Principles

- **Pure Rust** — no C/C++ kernel sources or toolchains
  ([decisions.md](decisions.md) ADR-0001).
- **Debug build = detailed profiling**: the built-in lightweight profiler is the
  instrumentation vehicle (Nsight does not support this card).
- **Release build = the shippable artifact.**
- Evidence-first culture: performance numbers always quote their conditions
  (context / batch / quantization / driver) with reproduction steps.

## Documentation Layout

- `docs/` — project documentation (tracked, English)
- `source/` — external reference material, **local and untracked** (research
  reports, engine clones used read-only during development)
- `plans/` — work plans (gitignored)
- Operational issues live at the bottom of each model/topic document
  (date header + symptom/cause/verification)

## Current Stage (2026-09-12)

Both reference models run end-to-end and are verified against llama.cpp greedy
streams — including long-prompt and parallel-sequence cases, under the
near-tie-aware standard (ADR-0012 in [decisions.md](decisions.md)). The latest
full gate run is 17/19 PASS + 2 INFO (the two INFO cases are the accepted
long-context f16-prefill class, where kernel accuracy is guaranteed separately
by `attn-check`).

On the GPU the engine has two independent backends (see
backend-architecture.md): rawhip (HIP/ROCm — full pipeline: all quantized GEMM
projections in 8 types with batched prefill tiles, flash attention, the GDN
scan, and a bit-exact element-wise set) and rawvk (Vulkan — subgroup GEMV
decode kernels for all 8 quant types, cooperative-matrix prefill tiles, fused
residual/RMS/GDN/attention kernels and a GPU-side argmax, reaching 0.91×
(pp512) and 0.95× (tg32) of llama.cpp Vulkan on the reference APU).

**HIP standing against llama-bench ROCm, CLI-to-CLI, same GGUF:** pp512 364 t/s
(1.03×), pp3314 339 t/s (1.01×), tg512 11.6 t/s (1.01×), tg3314 11.3 t/s
(0.98×); MTP speculative decode 22.1 t/s single-stream (1.92×) and 30.6 t/s
aggregate at np4 (1.97×); vision encoding 1.1 s vs 1.60 s (1.45×). The
attention path was redesigned to get there: an fp16 WMMA tile kernel for
prefill (Q in registers, its shared buffer reused as the K/V tile) and a
decode kernel that puts one key per lane and computes each 256-dim dot with
`v_dot2_f32_f16` — no cross-lane reduction — over an f16 KV mirror maintained
at the KV write sites. qwen4exp decodes through a GPU-resident frame by
default (ADR-0017) — kernels chained by handle, ~600 per-step host syncs down
to ~14. The HTTP server schedules continuous batching across slots; qwen35 has
MTP speculative decoding (`--spec k`), whose verify runs as a batched GPU
forward.

Remaining before the CMP arrives: the qwen35 decode frame, PLE/QSA frame
bridges for qwen4exp, KV quantization (a capacity lever, and the last
bandwidth lever for decode attention), and the cmp-stock kernel variants. The
staged plan lives in [architecture.md](architecture.md).
