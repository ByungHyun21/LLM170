# LLM170

A **pure-Rust** LLM inference engine built from the ground up — originally
for the NVIDIA CMP 170HX, currently developed and benchmarked on AMD APUs
(Radeon 8060S / gfx1151, ROCm + Vulkan) with portability as a standing goal.

No llama.cpp. No ggml. No C/C++ toolchain. Every layer of the stack — GGUF parsing, quantization, kernels, scheduling, profiling — is implemented in Rust from scratch.

> **Why the CMP 170HX?** It's a GA100 (A100-class) die sold as a mining card at a fraction of the price: 8 GB HBM2e at ~1.5 TB/s, sm_80, and standard NVIDIA drivers. The catch: eFUSE throttling caps FP32 FFMA at 1/32 rate and tensor cores at ~12% — while leaving **FP16 half2 (~42 TFLOPS) and INT32 at full speed**. Stock inference stacks are crippled by this; an engine designed around the constraint is not. With the 2026 community unlock (64 GB), the card becomes a serious decode machine. Details in [docs/hardware/cmp170hx.md](docs/hardware/cmp170hx.md).

## Benchmarks

All rows are solo, greedy; `llm170` numbers are ROCm 10 userspace with the
rocBLAS Tensile path pinned, llama.cpp is the master build (d222767c7) unless
noted. Full conditions and history: [docs/benchmarks.md](docs/benchmarks.md).

### CMP 170HX (GA100) — benchmark in preparation

| Mode | pp512 | tg128 |
|---|---|---|
| `cmp-stock` (8 GB) | — | — |
| `cmp-unlocked` (64 GB) | — | — |

### Strix Halo (Ryzen AI Max+ 395 / Radeon 8060S, gfx1151)

Greedy, single-tenant, same GGUF, t/s; `—` = not measured.

#### Qwen3.8-27B (Q4_K_XL 16.3 GiB)

Matched scorecard, 2026-09-17, same host and session.

| backend | pp512 | pp4096 | pp16384 | tg128@4k | tg128@16k |
|---|---|---|---|---|---|
| LLM170 hip | **356.8**-359.5 | **335.5**-336.9 | 293.0-294.3 | 11.5-11.6 | 11.6 |
| LLM170 vulkan | — | 150 | — | 9.2 | — |
| llama.cpp (ROCm 10) | 344.0 | 333.6 | **296.4** | **11.67** | **11.21** |

np4 prefill aggregate (4 slots prefilling concurrently, prompt tokens/s).

| backend | np4 pp aggregate |
|---|---|
| LLM170 hip | **374** |
| LLM170 vulkan | — |
| llama.cpp (ROCm 10) | — |

Decode modes: aggregate t/s over 4 parallel slots; MTP = `--spec 3`
(`LLM170_SLOTS=4` / `llama-server -np 4` are required for the np4 rows).

| mode | tg agg | tg agg | tg agg |
|---|---|---|---|
| | **LLM170 hip** | **LLM170 vulkan** | **llama.cpp (ROCm 10)** |
| tg single | 11.5-11.6 (4k) / 11.6 (16k) | 9.2 | 11.67 / 11.21 |
| MTP single | **15.4** (k=2) / 9.3 (k=3) | 9.2 (no MTP) | ~12 (MTP, old build) |
| np4 aggregate | **20.85** | — | **26.04** |
| MTP + np4 | **20.4** | — | 15.5 *(old build)* |

#### Qwen3.8-Flash-Next (177B-A3B, Q4_K_XL 103.7 GiB)

Matched scorecard, 2026-09-17; llama ran the `qwen4exp build-ab` build with
the model's required `-ot per_layer_token_embd=CPU --load-mode mmap -fit off`.

| backend | pp512 | pp4096 | pp16384 | tg128@4k | tg128@16k |
|---|---|---|---|---|---|
| LLM170 hip | **221.4**-252.3 | **268.9**-274.9 | **239.5**-246.1 | 18.42-18.56 | 18.37 |
| LLM170 vulkan | — | 271 | 240 | 17.1 | 16.7 |
| llama.cpp (ROCm 10) | 245.2 | 259.6 | ~229 | **20.23** | 17.79 (@4k) |

np4 prefill aggregate (same convention as the 27B table above).

| backend | np4 pp aggregate |
|---|---|
| LLM170 hip | — |
| LLM170 vulkan | — |
| llama.cpp (ROCm 10) | — |

Decode modes: aggregate t/s over 4 parallel slots; the model has no nextn/MTP
head, so the MTP rows are structurally inapplicable.

| mode | tg agg | tg agg | tg agg |
|---|---|---|---|
| | **LLM170 hip** | **LLM170 vulkan** | **llama.cpp (ROCm 10)** |
| tg single | **18.56** (ctx 4k) / **18.35** (ctx 16k) | 17.1 | **20.23** / 17.79 |
| MTP single | — | — | — |
| np4 aggregate | **24.94** | — | **41.07** |
| MTP + np4 | — | — | — |

### Vision (mmproj), 27B — 2026-09-17

Same image and question, greedy; model loads excluded.

| phase | LLM170 | llama.cpp |
|---|---|---|
| vision encode + LLM prefill | 1.1 s (vit) + 1.2 s (300 tok) | 1.62 s (362 tok, clip folded in) |
| decode 48 tok | ~11.6 t/s | 10.19 t/s |
| total | ~6.4 s | ~6.3 s |

### Recent improvements (2026-09-16/17)

- **MTP is no longer slower than plain decode.** The speculative verify batch's
  carry-forward was capped at 16 rows, so a partial acceptance grew the verify
  batch to 17 rows and every cycle re-executed O(carry) rows — `--spec 3`
  measured 8.02 t/s against 11.6 t/s plain. Capping the carry at `1 + k + 4`
  gives **spec2 = 15.4 t/s (+33% over plain)** with the spec stream
  token-identical to greedy (k=2) and both gates bit-exact.
- **Batched decode steps now match or beat llama.cpp's**: Flash-Next live t=4
  step 91-93 ms vs its 97 ms; 27B 141 ms vs its 153.6 ms (this is why the
  remaining np4 gap is the per-request prefill weight pass, not the batch
  kernel — see docs/benchmarks.md).
- **Structural cleanup**: `Accelerator`'s 51-method surface split into five
  capability traits (GraphCapture / MatmulHost / EwOps / QsaOps / FrameHost)
  with zero call-site changes; `rawhip/mod.rs` split into `graph.rs` +
  `ktrace.rs`; 26 dead items and 7 unreferenced shaders removed; `cargo build`
  is warning-free and `cargo test` is 16/16.
- Matched-protocol scorecard redone (same host, same client, same prompts,
  `LLM170_SLOTS=4` vs `-np 4` for np): **prefill wins on both models at every
  measured length** and tg is at/near parity.

## Safety: pre-load resource guard

Before any subcommand loads a model, the engine compares the model's size
(all split parts) against free VRAM + 0.85 x available RAM and refuses to
start when it would not fit — instead of driving the host into an OOM freeze
(`LLM170_NO_RSRC_GUARD=1` overrides). This exists because running two
inference servers on one 96 GiB APU with a 104 GiB model froze the machine
once (2026-09-16).

## Build & run

Requires Rust 1.95+ (stable, edition 2024). No GPU or CUDA toolkit needed to build.

```bash
cargo build --release

# Inspect a model — structure, hyperparameters, quantization mix
cargo run --release -- gguf-dump --meta-only <model.gguf>

# Greedy CPU inference (token-id input)
cargo run --release -- infer \
    --model <model.gguf> \
    --prompt-tokens 760,6511,314,9338,369 \
    --n-predict 16

# Same inference with the GPU-resident decoder.
# --gpu-runtime hip (ROCm, default) or vulkan; weights upload once and stay resident.
cargo run --release -- infer \
    --model <model.gguf> \
    --prompt-tokens 760,6511,314,9338,369 \
    --n-predict 16 \
    --gpu-runtime vulkan

# HTTP server: /health, /v1/models, /tokenize, /v1/completions, /v1/chat/completions (SSE),
# /v1/messages (Anthropic). Continuous batching across slots; client disconnects cancel the job.
cargo run --release -- serve --model <model.gguf> --port 8080 --backend gpu

# llama-bench-style PP/TG measurement (t/s). --spec k: effective t/s with MTP speculative decode.
cargo run --release -- bench --model <model.gguf> --pp 512 --tg 128 --gpu-runtime hip

# Runtime mode (memory-budget profile today; kernel variants when cmp-stock lands)
cargo run --release -- infer --model <model.gguf> --prompt-tokens 760,6511 --n-predict 16
```

The HIP attention path is a recent redesign worth naming: the prefill kernel is
an fp16 WMMA tile implementation (Q consumed into registers and its shared
buffer reused as the K/V tile, 32 KB of dynamic shared) that replaced a
shuffle-bound scalar kernel, and the decode kernel follows llama.cpp's tile
structure — one key per lane with the whole 256-dim dot computed in a single
thread via `v_dot2_f32_f16`, so the QK phase has no cross-lane reduction at
all — reading an f16 KV mirror maintained at the KV write sites. Both were
validated against CPU references (`wmma-attn-check`, `gqa-bench`) before
becoming defaults.

GPU kernel self-check subcommands (cross-validated against the CPU reference):
`vk-check` (device/coopmat smoke) · `gdn-check` (GDN/attention kernel suite) ·
`vk-gemv-check` / `vk-gemv8-check` (per-type GEMV) · `rawhip-check` (HIP GEMV
bit-parity) · `subsum-check` (subgroup reductions) · `qk-check` ·
`wmma-attn-check` (WMMA attention vs CPU reference) · `gqa-bench` (decode
attention variants, timing + differential) ·
`iq3s-probe` · `check` (tensor scan + cross-validation + chunk smoke) ·
`w4a8-check`. Run `llm170 help` for the full list.

Debug builds are fully instrumented — every stage is timed by the built-in profiler, by design. Release builds carry zero instrumentation.

GPU kernels are HIP C++ (JIT-compiled at runtime via hipRTC from per-family
source assets) and GLSL compute shaders (precompiled to SPIR-V), with Rust
owning all orchestration — one kernel source per backend, arithmetic mirrored
from the CPU reference for bit-level verification. ROCm/HIP drives AMD GPUs;
the Vulkan backend needs only a conformant driver. See
[docs/backend-architecture.md](docs/backend-architecture.md).

## Repository layout

```
crates/gguf        GGUF v3 parser
crates/core        dequantization, matmul, Gated DeltaNet, qwen35 + qwen4exp engines
crates/backend-gpu raw HIP (hipRTC) + raw Vulkan (SPIR-V) GPU backends
                   (rawhip/: ctx·launch·buffers, graph.rs, ktrace.rs, decode.rs)
crates/profiler    debug-gated lightweight profiler
crates/server      llm170 CLI + HTTP server
docs/              specs & decisions (hardware, models, architecture, ADRs, benchmarks)
scripts/           verification harness (synthetic-model generators, llama.cpp cross-checks)
```

## Verification philosophy

- **Cross-checked against llama.cpp** greedy token streams, with a near-tie-aware standard for flat-top distributions (top-6 membership + logprob-gap threshold).
- Internal consistency where it's cheap and strong: the chunked and autoregressive GDN paths are cross-validated on arbitrary inputs, and every GPU kernel is checked against the CPU reference before entering the engine.
- Performance numbers are only ever quoted with their conditions (context / batch / quantization / backend) — see [docs/benchmarks.md](docs/benchmarks.md).

## Documentation

- [Project overview](docs/overview.md) — goals, modes, reference models
- [Architecture](docs/architecture.md) — backend strategy, FP discipline, kernel surface
- [Benchmarks](docs/benchmarks.md) — pp/tg comparison with conditions
- [qwen35 spec](docs/models/qwen35.md) · [qwen4exp spec](docs/models/qwen4exp.md) — tensor-level implementation specs
- [CMP 170HX hardware spec](docs/hardware/cmp170hx.md) — throttle matrix, unlock, memory budgets
- [Decision records](docs/decisions.md) — ADRs
