# LLM170

A **pure-Rust** LLM inference engine built from the ground up — originally
for the NVIDIA CMP 170HX, currently developed and benchmarked on AMD APUs
(Radeon 8060S / gfx1151, ROCm + Vulkan) with portability as a standing goal.

No llama.cpp. No ggml. No C/C++ toolchain. Every layer of the stack — GGUF parsing, quantization, kernels, scheduling, profiling — is implemented in Rust from scratch.

> **Why the CMP 170HX?** It's a GA100 (A100-class) die sold as a mining card at a fraction of the price: 8 GB HBM2e at ~1.5 TB/s, sm_80, and standard NVIDIA drivers. The catch: eFUSE throttling caps FP32 FFMA at 1/32 rate and tensor cores at ~12% — while leaving **FP16 half2 (~42 TFLOPS) and INT32 at full speed**. Stock inference stacks are crippled by this; an engine designed around the constraint is not. With the 2026 community unlock (64 GB), the card becomes a serious decode machine. Details in [docs/hardware/cmp170hx.md](docs/hardware/cmp170hx.md).

## Benchmarks

All rows are solo, greedy, **CLI-to-CLI**: `llm170 bench` vs `llama-bench`,
same host and session (2026-09-18), one full-shape warm-up run before each
measurement (matching llama-bench's built-in warm-up). `llm170` numbers are
ROCm 10 userspace with the rocBLAS Tensile path pinned. llama.cpp is the
master build (d222767c7) unless noted; Flash-Next rows use a qwen4exp-capable
local build (2cc83c6f4) since master's llama-bench rejects the `qwen4exp`
architecture. Full conditions and history: [docs/benchmarks.md](docs/benchmarks.md).

### CMP 170HX (GA100) — benchmark in preparation

| Mode | pp512 | tg128 |
|---|---|---|
| `cmp-stock` (8 GB) | — | — |
| `cmp-unlocked` (64 GB) | — | — |

### Strix Halo (Ryzen AI Max+ 395 / Radeon 8060S, gfx1151)

Greedy, single-tenant, same GGUF, t/s; `—` = not measured.

#### Qwen3.8-27B (Q4_K_XL 16.3 GiB)

| backend | pp512 | pp4096 | pp16384 | tg128@4k | tg128@16k |
|---|---|---|---|---|---|
| LLM170 hip | **367-370** | 326-337 | 290-296 | 11.52 | **11.53** |
| LLM170 vulkan | 318 | 145 | n/a | 11.26 | 11.30 |
| llama.cpp (ROCm 10) | 340 | **335** | **297** | **11.65** | 11.21 |

Vulkan's 27B pp16384 is `n/a`: the run aborts with `ERROR_DEVICE_LOST`
(radv: "The CS has been cancelled because the context is lost"). The Flash-Next
16k prefill on Vulkan completes normally (243), so this is a 27B long-context
Vulkan limit, not a general backend failure. `MTP + np4` on Vulkan is likewise
`n/a`: `serve --spec k` with `--gpu-runtime vulkan` loads the model and then
stops responding before its slots come up (the HIP path is fine).

np4 aggregate — `llm170 bench --np 4` (engine slot loop, disjoint prompts,
prefill excluded from tg timing; 4 slots, ctx 8192).

| backend | pp512 agg | pp2048 agg | tg128 agg |
|---|---|---|---|
| LLM170 hip | **352.9** | **331.8** | **32.1** |
| LLM170 vulkan | 305.4 | 200.4 | 10.6 |
| llama.cpp (ROCm 10) | 188.5 | 226.5 | 25.2 |

† llama's np4 rows are the prior HTTP measurements (`llama-server -np 4`,
`scripts/bench_np.py`): llama-bench has no multi-slot mode, so no CLI
equivalent exists. All LLM170 np4 rows above are engine-API bench rows.

Decode modes (27B; aggregate t/s over 4 slots):

| mode | LLM170 hip | LLM170 vulkan | llama.cpp (ROCm 10) |
|---|---|---|---|
| tg single | 11.5 | 11.26-11.30 | 11.65 / 11.21 (@4k/@16k) |
| MTP single | 14.4 (k=2) / 8.3 (k=3) | — | ~12 *(MTP, old build)* |
| np4 aggregate | **32.1** | 10.6 | **25.2** *(HTTP†)* |
| MTP + np4 | 6.5-6.8 | n/a | 15.5 *(old build, HTTP†)* |

(MTP rows via `bench --spec k`; MTP+np4 is the engine merged spec×np path —
`spec_step_multi`. The prior 20.4 HTTP figure for MTP+np4 came from a
different protocol/session; the CLI engine path measures 6.5-6.8, and the
pre-refactor main build measures the same, so it is the current engine truth.
k=3 drafts are largely rejected in this workload — see docs/benchmarks.md.)

#### Qwen3.8-Flash-Next (177B-A3B, Q4_K_XL 103.7 GiB)

llama-bench runs with the model's required
`-ot per_layer_token_embd=CPU --load-mode mmap` on the qwen4exp-capable build
(2cc83c6f4); master cannot load this architecture.

| backend | pp512 | pp4096 | pp16384 | tg128@4k | tg128@16k |
|---|---|---|---|---|---|
| LLM170 hip | **253-275** | **275-278** | **244-249** | **18.60** | **18.54** |
| LLM170 vulkan | 237 | 277 | 243 | 18.49 | — |
| llama.cpp (ROCm 10) | 222 | 210 | 200 | 17.43 | 13.88 |

np4 aggregate — `llm170 bench --np 4` (same convention as the 27B table).

| backend | pp512 agg | pp2048 agg | tg128 agg |
|---|---|---|---|
| LLM170 hip | **248-250** | **287-288** | **45.9** |
| LLM170 vulkan | 248.1 | 285.1 | 18.6 |
| llama.cpp (ROCm 10) | 20.6 | 229.8 | 41.1-41.7 *(HTTP†)* |

Decode modes (FN; the model has no nextn/MTP head, so MTP rows are
structurally inapplicable):

| mode | LLM170 hip | LLM170 vulkan | llama.cpp (ROCm 10) |
|---|---|---|---|
| tg single | 18.5-18.6 | 18.49 | 17.43 / 13.88 (@4k/@16k) |
| np4 aggregate | **45.9** | 18.6 | 41.1-41.7 *(HTTP†)* |

### Vision (mmproj), 27B — 2026-09-17

Same image and question, greedy; model loads excluded.

| phase | LLM170 | llama.cpp |
|---|---|---|
| vision encode + LLM prefill | 1.1 s (vit) + 1.2 s (300 tok) | 1.62 s (362 tok, clip folded in) |
| decode 48 tok | ~11.6 t/s | 10.19 t/s |
| total | ~6.4 s | ~6.3 s |

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
