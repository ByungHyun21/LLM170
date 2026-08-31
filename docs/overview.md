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
| `cmp-unlocked` | CMP 170HX unlocked (40–64 GB, compute unlock) | full rate allowed | Performance first (to be finalized after unlock measurements) |

- Modes are implemented as **runtime flags + kernel-variant selection**. The
  engine core (model, scheduler, loader) is mode-agnostic.
- Memory budgets per mode: stock ~7 GiB / unlocked 40–64 GiB — see
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

- `docs/` — project documentation (tracked)
- `source/` — external reference material: research reports, excerpts (tracked,
  not project output)
- `plans/` — work plans (gitignored)
- Operational issues live at the bottom of each model/topic document or in
  separate ISSUES-style files (date header + symptom/cause/verification)

## Current Stage (2026-08-31)

Both reference models run end-to-end and are verified against llama.cpp greedy
streams. GPU backend (cubecl) executes all weight projections on
HIP/ROCm and Vulkan with token-exact parity to the CPU reference. Remaining
before the CMP arrives: GPU-resident GDN/QSA compute (decode latency), long
prefill throughput, and the cmp-stock kernel variants. See
[architecture.md](architecture.md) §Staged Plan.
