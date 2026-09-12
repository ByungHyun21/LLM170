# LLM170

A **pure-Rust** LLM inference engine built from the ground up — originally
for the NVIDIA CMP 170HX, currently developed and benchmarked on AMD APUs
(Radeon 8060S / gfx1151, ROCm + Vulkan) with portability as a standing goal.

No llama.cpp. No ggml. No C/C++ toolchain. Every layer of the stack — GGUF parsing, quantization, kernels, scheduling, profiling — is implemented in Rust from scratch.

> **Why the CMP 170HX?** It's a GA100 (A100-class) die sold as a mining card at a fraction of the price: 8 GB HBM2e at ~1.5 TB/s, sm_80, and standard NVIDIA drivers. The catch: eFUSE throttling caps FP32 FFMA at 1/32 rate and tensor cores at ~12% — while leaving **FP16 half2 (~42 TFLOPS) and INT32 at full speed**. Stock inference stacks are crippled by this; an engine designed around the constraint is not. With the 2026 community unlock (40–64 GB), the card becomes a serious decode machine. Details in [docs/hardware/cmp170hx.md](docs/hardware/cmp170hx.md).

## Benchmarks

Qwen3.8-27B (hybrid GDN + full attention) on the dev machine (Radeon 8060S,
gfx1151), greedy, single-tenant `llm170 bench`. Numbers are only ever quoted
with their conditions — full tables and history in
[docs/benchmarks.md](docs/benchmarks.md).

| Backend | pp512 prefill | decode (tg32) | note |
|---|---|---|---|
| ROCm/HIP (`rawhip`) | **364 t/s** | **11.6 t/s** | llama-bench ROCm, same GGUF, CLI-to-CLI: 354.7 / 11.56 → **1.03× / 1.01×** |
| Vulkan (`rawvk`) | 318 t/s | 10.9 t/s | llama.cpp Vulkan: 350.8 / 11.48 → 0.91× / 0.95× |
| CPU (W4A8) | 181 t/s (pp64) | 11.7 t/s (tg24) | bit-exact reference engine |

At longer contexts the HIP backend holds its lead on prompt processing
(pp3314 339 t/s vs llama-bench 335, 1.01×) and closes to 0.98× on decode
(11.3 vs 11.5). The decode attention is bandwidth-bound at that point: it
moves its f16 KV at ~221 GB/s effective.

The Vulkan backend reached these numbers with three decode/prefill kernel
families of its own — subgroup GEMV (decode, faithful llama dmmv ports),
cooperative-matrix tiles (prefill) and fused elementwise kernels — all
arithmetic-mirrored from the CPU reference. A year of measured experiments
behind the current numbers is logged in [docs/benchmarks.md](docs/benchmarks.md).

Speculative decode (HIP, MTP) runs its verify as a batched GPU forward by
default: **22.1 t/s** single-stream at `--spec 3` and **30.6 t/s aggregate** at
np4, against llama.cpp's MTP references of 11.5 and 15.5 — **1.92× and 1.97×** —
with the accepted token stream identical to non-spec greedy on both short and
2.3k-token prompts.

Reference models: Qwen3.8-27B (`qwen35` — Gated DeltaNet + Gated Attention)
and Qwen3.8-Flash-Next (`qwen4exp` — sparse attention, MoE, PLE).

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
cargo run --release -- infer --model <model.gguf> --prompt-tokens 760,6511 --n-predict 16 --mode universal
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
