# LLM170 Project Overview

A **pure-Rust** LLM inference engine targeting the NVIDIA CMP 170HX as its final
target. Built from scratch — no llama.cpp, no ggml.

## Goals & Priorities

1. **Priority 1 — CMP 170HX** (GA100, sm_80): maximum performance, assuming the
   40/64 GB unlock.
2. **Priority 2 — portability**: runs on arbitrary hardware (any CPU, any GPU).
   Solo development, so this machine (Ryzen AI Max+ 395 / gfx1151) doubles as
   the development and verification baseline.
3. **What happens on this machine now**: development on the integrated GPU. The
   CMP 170HX is not yet accessible.

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

## Current Stage (2026-09-18)

Both reference models run end-to-end and are verified against llama.cpp greedy
streams — including long-prompt and parallel-sequence cases, under the
near-tie-aware standard (ADR-0012 in [decisions.md](decisions.md)).

Benchmarking is CLI-to-CLI (`llm170 bench` vs `llama-bench`, one full-shape
warm-up per measurement; see the README tables). Standing on the dev APU,
same-session matched runs: 27B pp512 **367-370 t/s** (llama 340), tg128
11.5 (llama 11.65/11.21 @4k/@16k); Flash-Next leads every cell — pp4096
**275-278 vs 210**, pp16384 **244-249 vs 200**, tg@16k **18.5 vs 13.9**;
27B np4 aggregate 32.1 t/s (engine batched decode). Vulkan covers both
models (27B long-context prefill still aborts with `ERROR_DEVICE_LOST`
above pp4096 — a 27B-specific radv limit).

Structure (after the plans/75/78/79 refactor campaigns): every major file
sits under a 2k-line ceiling (`q4acc/`, `decode/`, `probes/`, `frame/`,
`decoder/` module trees), the `Accelerator` surface is split into
capability traits, clippy is at zero warnings, and the env surface is
catalogued and cached. Remaining engineering items live in plans/77: the
Flash-Next chunk-size invariance defect (`Q4_CHUNK=63/64` prefill
divergence — blocks multi-sequence batched prefill adoption), two Vulkan
server issues (27B long-context DEVICE_LOST, `--spec` + Vulkan boot hang),
and the np4/FN decode scaling gap vs llama.

