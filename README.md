# LLM170

A **pure-Rust** LLM inference engine built from the ground up for the NVIDIA CMP 170HX — with real portability as a second goal.

No llama.cpp. No ggml. No C/C++ toolchain. Every layer of the stack — GGUF parsing, quantization, kernels, scheduling, profiling — is implemented in Rust from scratch.

> **Why the CMP 170HX?** It's a GA100 (A100-class) die sold as a mining card at a fraction of the price: 8 GB HBM2e at ~1.5 TB/s, sm_80, and standard NVIDIA drivers. The catch: eFUSE throttling caps FP32 FFMA at 1/32 rate and tensor cores at ~12% — while leaving **FP16 half2 (~42 TFLOPS) and INT32 at full speed**. Stock inference stacks are crippled by this; an engine designed around the constraint is not. With the 2026 community unlock (40–64 GB), the card becomes a serious decode machine. Details in [docs/hardware/cmp170hx.md](docs/hardware/cmp170hx.md).

## Status — early development

| Component | State |
|---|---|
| GGUF v3 parser (metadata · tensors · splits) | ✅ verified against real models — Qwen3.8-27B (866 tensors, 0 B slack) & Flash-Next 4-way split |
| `gguf-dump` CLI (quant-mix analysis) | ✅ |
| Lightweight profiler (debug-instrumented, zero-cost in release) | ✅ |
| Dequantization — q4_K / q5_K / q6_K / q8_0 / q3_K / q5_1 / iq4_xs / iq4_nl / iq3_s | ✅ cross-checked against an independent reference implementation |
| Gated DeltaNet — chunked & autoregressive (CPU) | ✅ two paths agree to 1e-7; cross-validated against llama.cpp |
| qwen35 inference engine (CPU + GPU matmul) | ✅ greedy tokens match llama.cpp — 11/11 matrix (6 exact + 5 near-tie) |
| qwen4exp engine (CPU + GPU matmul) | ✅ hyper-connections, QSA indexer, MoE (512 experts, token-expert grouped batching), PLE NVMe offload — llama.cpp parity on the reduced matrix |
| qwen4exp (QSA / hyper-connections / PLE / MoE) | ⏳ planned |
| GPU backend — cubecl, quantized GEMM (HIP/ROCm + Vulkan) | ✅ all 8 quant types; GPU greedy stream == CPU stream token-for-token |

**Modes.** The engine runs in three profiles: `universal` (any device — the development baseline), `cmp-stock` (8 GB, eFUSE-throttled; FMA-free, tensor-core-free kernels), and `cmp-unlocked` (40–64 GB unlocked silicon). Engine core is mode-agnostic; modes select kernel variants and memory budgets.

**Reference models.** Qwen3.8-27B (`qwen35` hybrid — Gated DeltaNet + Gated Attention) first; Qwen3.8-Flash-Next (`qwen4exp` — sparse attention, MoE, n-gram PLE embedding) next.

## Build & run

Requires Rust 1.95+ (stable, edition 2024). No GPU or CUDA toolkit needed to build.

```bash
cargo build --release

# Inspect a model — structure, hyperparameters, quantization mix
cargo run --release -- gguf-dump --meta-only <model.gguf>

# Greedy CPU inference (token-id input; tokenizer in progress)
cargo run --release -- infer \
    --model <model.gguf> \
    --prompt-tokens 760,6511,314,9338,369 \
    --n-predict 16

# Same inference with all weight projections on the GPU (cubecl).
# --gpu-runtime hip (default, ROCm) or vulkan (wgpu); weights upload once and stay resident.
cargo run --release -- infer \
    --model <model.gguf> \
    --prompt-tokens 760,6511,314,9338,369 \
    --n-predict 16 \
    --backend gpu --gpu-runtime vulkan
```

Debug builds are fully instrumented — every stage is timed by the built-in profiler, by design. Nsight doesn't support this card; the engine is its own profiler.

GPU kernels are written in Rust via [CubeCL](https://github.com/tracel-ai/cubecl) macros and JIT-compiled for the target architecture (gfx1151 on the dev machine, sm_80 for the CMP 170HX). The same Rust kernel source compiles to both backends.

## Repository layout

```
crates/gguf      GGUF v3 parser
crates/core      dequantization, matmul, Gated DeltaNet, qwen35 CPU reference engine
crates/profiler  debug-gated lightweight profiler
crates/server    llm170 CLI
docs/            specs & decisions (hardware, model, architecture, ADRs)
source/          external reference material (research reports; engine clones are gitignored)
scripts/         verification harness (synthetic-model generators, llama.cpp cross-checks)
```

## Verification philosophy

- **Everything is cross-checked against llama.cpp** on this machine. Synthetic qwen35 models — random weights, real and reduced dimensions — are generated as GGUF and run through both engines, compared down to greedy token streams.
- Internal consistency is enforced where it's cheap and strong: e.g. the chunked and autoregressive GDN paths must agree to 1e-7 on arbitrary inputs.
- Performance numbers are only ever quoted with their conditions (context / batch / quantization). Server-based streaming benchmarks are the standard; synthetic microbenchmarks alone are not.

## Documentation

- [Project overview](docs/overview.md) — goals, modes, reference models
- [Benchmarks vs llama.cpp](docs/benchmarks.md) — pp/tg comparison with conditions
- [Architecture](docs/architecture.md) — backend strategy, FP discipline, kernel surface
- [CMP 170HX hardware spec](docs/hardware/cmp170hx.md) — throttle matrix, unlock, memory budgets
- [Dev machine (ROCm)](docs/hardware/dev-8060s.md)
- [qwen35 spec](docs/models/qwen35.md) · [qwen4exp spec](docs/models/qwen4exp.md) — tensor-level implementation specs
- [Decision records](docs/decisions.md) — ADRs
- [AGENTS.md](AGENTS.md) — contribution & engineering rules

Historical ADRs and internal engineering notes are Korean; all docs under `docs/` are English going forward.
