# LLM170

A **pure-Rust** LLM inference engine built from the ground up — originally
for the NVIDIA CMP 170HX, currently developed and benchmarked on AMD APUs
(Radeon 8060S / gfx1151, ROCm + Vulkan) with portability as a standing goal.

No llama.cpp. No ggml. No C/C++ toolchain. Every layer of the stack — GGUF parsing, quantization, kernels, scheduling, profiling — is implemented in Rust from scratch.

> **Why the CMP 170HX?** It's a GA100 (A100-class) die sold as a mining card at a fraction of the price: 8 GB HBM2e at ~1.5 TB/s, sm_80, and standard NVIDIA drivers. The catch: eFUSE throttling caps FP32 FFMA at 1/32 rate and tensor cores at ~12% — while leaving **FP16 half2 (~42 TFLOPS) and INT32 at full speed**. Stock inference stacks are crippled by this; an engine designed around the constraint is not. With the 2026 community unlock (64 GB), the card becomes a serious decode machine. Details in [docs/hardware/cmp170hx.md](docs/hardware/cmp170hx.md).

## Benchmarks

Qwen3.8-27B (hybrid GDN + full attention), greedy, single-tenant `llm170 bench`.
Numbers are only ever quoted with their conditions — full tables and history in
[docs/benchmarks.md](docs/benchmarks.md).

### CMP 170HX (GA100) — benchmark in preparation

| Mode | pp512 | tg128 |
|---|---|---|
| `cmp-stock` (8 GB) | — | — |
| `cmp-unlocked` (64 GB) | — | — |

### Strix Halo (Ryzen AI Max+ 395 / Radeon 8060S, gfx1151)

Greedy, single-tenant solo runs, same GGUF, t/s; unmeasured cells are blank.
Session gains (2026-09-15, hip): Flash-Next tg 13.40 → 17.5 (+31%, short ctx)
and 11.3 → 16.8 (+49%, 16k) — QSA indexer selection and PLE moved onto the GPU,
five warp-per-row GEMV kernels, bit-identical bitonic top-k; all gate streams
unchanged. The hip backend carries the current optimization set; vulkan reflects
the pre-port path.

#### Qwen3.8-27B (Q4_K_XL 16.3 GiB)

| backend | pp4096 | pp16384 | tg128@4k | tg128@16k |
|---|---|---|---|---|
| LLM170 hip | 324 | 253 | 11.1 | 10.5 |
| LLM170 vulkan | VK27PP4 | VK27PP16 | VK27TG4 | VK27TG16 |
| llama.cpp (ROCm 10) | **342** | **317** | **11.6** | **11.9** |

#### Qwen3.8-Flash-Next (177B-A3B, Q4_K_XL 103.7 GiB)

| backend | pp4096 | pp16384 | tg128@4k | tg128@16k |
|---|---|---|---|---|
| LLM170 hip | **270** | **243** | 17.1 | 16.8 |
| LLM170 vulkan | VKNPP4 | VKNPP16 | VKNTG4 | VKNTG16 |
| llama.cpp (ROCm 10) | 237 | 229 | **20.2** | **20.0** |

#### Speculative decode (27B, MTP `--spec 3`)

| Config | t/s | vs llama MTP |
|---|---|---|
| single-stream | **23.5** | 2.04× |
| np4 aggregate | **31.0** | 2.00× |

Full analysis: [docs/benchmarks.md](docs/benchmarks.md).

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
