# LLM170

A **pure-Rust** LLM inference engine built from the ground up for the NVIDIA CMP 170HX — with real portability as a second goal.

No llama.cpp. No ggml. No C/C++ toolchain. Every layer of the stack — GGUF parsing, quantization, kernels, scheduling, profiling — is implemented in Rust from scratch.

> **Why the CMP 170HX?** It's a GA100 (A100-class) die sold as a mining card at a fraction of the price: 8 GB HBM2e at ~1.5 TB/s, sm_80, and standard NVIDIA drivers. The catch: eFUSE throttling caps FP32 FFMA at 1/32 rate and tensor cores at ~12% — while leaving **FP16 half2 (~42 TFLOPS) and INT32 at full speed**. Stock inference stacks are crippled by this; an engine designed around the constraint is not. With the 2026 community unlock (40–64 GB), the card becomes a serious decode machine. Details in [docs/hardware/cmp170hx.md](docs/hardware/cmp170hx.md).

## Status

Both reference models run end-to-end, CPU and GPU, with greedy-token parity against llama.cpp:

| Component | State |
|---|---|
| GGUF v3 parser (metadata · tensors · splits) | ✅ verified against real models |
| `gguf-dump` CLI (quant-mix analysis) | ✅ |
| Lightweight profiler (debug-instrumented, zero-cost in release) | ✅ |
| Dequantization — 12 types: f32/f16/bf16, q4_K / q5_K / q6_K / q8_0 / q3_K / q5_1 / iq4_xs / iq4_nl / iq3_s | ✅ cross-checked against an independent reference implementation |
| Gated DeltaNet — chunked & autoregressive | ✅ two paths cross-validated; llama.cpp token parity |
| qwen35 inference engine (CPU + GPU) | ✅ MTP speculative decoding (`--spec k`) |
| qwen4exp engine (CPU + GPU) | ✅ hyper-connections, QSA indexer, MoE (512 experts, token-expert grouped batching), PLE offload — incl. long-prompt and parallel-sequence cases |
| GPU kernel set (cubecl) | ✅ quantized GEMM (8 types + batched prefill variants), GDN AR + chunked prefill, grouped MoE GEMM, element-wise set — GPU greedy stream == CPU stream token-for-token |
| GPU-resident decode frame (qwen4exp) | ✅ default on; bit-exact against the per-op path on synthetic e2e |
| HTTP server | ✅ OpenAI/Anthropic-compatible endpoints, SSE streaming, continuous batching slot scheduler, client-disconnect cancellation |

**Modes.** The engine runs in three runtime profiles: `universal` (any device — the development baseline), `cmp-stock` (8 GB, eFUSE-throttled; FMA-free, tensor-core-free kernels), and `cmp-unlocked` (40–64 GB unlocked silicon). The engine core is mode-agnostic; modes select kernel variants and memory budgets.

**Reference models.** Qwen3.8-27B (`qwen35` hybrid — Gated DeltaNet + Gated Attention) and Qwen3.8-Flash-Next (`qwen4exp` — sparse attention, MoE, n-gram PLE embedding).

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

# Same inference with all weight projections on the GPU (cubecl).
# --gpu-runtime hip (default) or vulkan; weights upload once and stay resident.
cargo run --release -- infer \
    --model <model.gguf> \
    --prompt-tokens 760,6511,314,9338,369 \
    --n-predict 16 \
    --backend gpu --gpu-runtime vulkan

# HTTP server: /health, /v1/models, /tokenize, /v1/completions, /v1/chat/completions (SSE),
# /v1/messages (Anthropic). Continuous batching across slots; client disconnects cancel the job.
cargo run --release -- serve --model <model.gguf> --port 8080 --backend gpu

# llama-bench-style PP/TG measurement (t/s). --spec k: effective t/s with MTP speculative decode.
cargo run --release -- bench --model <model.gguf> --pp 512 --tg 128 --backend gpu

# Runtime mode (memory-budget profile today; kernel variants when cmp-stock lands)
cargo run --release -- infer --model <model.gguf> --prompt-tokens 760,6511 --n-predict 16 --mode universal
```

GPU kernel self-check subcommands (cross-validated against the CPU reference):
`gpu-ew-check` · `gdn-ar-check` · `gdn-chunk-check` · `moe-down-check` · `w4a8-check` · `w4a8-gpu`.

Debug builds are fully instrumented — every stage is timed by the built-in profiler, by design. Release builds carry zero instrumentation.

GPU kernels are written in Rust via [CubeCL](https://github.com/tracel-ai/cubecl) macros and JIT-compiled for the target architecture at runtime — AMD GPUs via HIP/ROCm or Vulkan today, CUDA/PTX (sm_80) for the CMP 170HX. The same Rust kernel source compiles for every backend.

## Repository layout

```
crates/gguf      GGUF v3 parser
crates/core      dequantization, matmul, Gated DeltaNet, qwen35 + qwen4exp engines
crates/profiler  debug-gated lightweight profiler
crates/server    llm170 CLI + HTTP server
docs/            specs & decisions (hardware, models, architecture, ADRs, benchmarks)
scripts/         verification harness (synthetic-model generators, llama.cpp cross-checks)
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
