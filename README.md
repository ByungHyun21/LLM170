# LLM170

A pure-Rust LLM inference engine. No llama.cpp. No ggml. No C/C++ toolchain.

Currently benchmarked on AMD APUs (Radeon 8060S / gfx1151, ROCm + Vulkan), with NVIDIA CMP 170HX as the original target hardware.

## Benchmarks

Solo, greedy, `llm170 bench` vs `llama-bench`, same host (2026-09-18). Full conditions: [docs/benchmarks.md](docs/benchmarks.md).

### Qwen3.8-27B (Q4_K_XL 16.3 GiB)

| backend | pp512 | pp4096 | pp16384 | tg128@4k |
|---|---|---|---|---|
| LLM170 hip | **367-370** | 326-337 | 290-296 | 11.52 |
| LLM170 vulkan | 318 | 145 | n/a | 11.26 |
| llama.cpp hip | 340 | **335** | **297** | 11.65 |
| llama.cpp vulkan | 343 | 318 | — | **12.05** |

| mode | LLM170 hip | LLM170 vulkan | llama hip | llama vulkan |
|---|---|---|---|---|
| tg single | 11.5 | 11.26 | 11.65 | 12.05 |
| MTP single (k=2) | **14.4** | — | ~12 | — |
| MTP + np4 | 7.2 | n/a | 15.5 *(HTTP†)* | — |

### Qwen3.8-Flash-Next (177B-A3B, Q4_K_XL 103.7 GiB)
| backend | pp512 | pp4096 | pp16384 | tg128@4k |
|---|---|---|---|---|
| LLM170 hip | **253-275** | **275-278** | **244-249** | **18.60** |
| LLM170 vulkan | 237 | 277 | 243 | 18.49 |
| llama.cpp hip | 222 | 210 | 200 | 17.43 |
| llama.cpp vulkan | 230 | — | — | **22.98** |

| mode | LLM170 hip | LLM170 vulkan | llama hip | llama vulkan |
|---|---|---|---|---|
| tg single | **18.60** | 18.49 | 17.43 | 22.98 |
| np4 aggregate | **45.9** | 18.6 | 41.1 *(HTTP†)* | — |

## Build & run

Requires Rust 1.95+ (edition 2024). No GPU toolchain needed to build.

```bash
cargo build --release

# Inference
cargo run --release -- infer --model <model.gguf> --prompt-tokens 760,6511 --n-predict 16 --gpu-runtime hip

# HTTP server (OpenAI-compatible)
cargo run --release -- serve --model <model.gguf> --port 8080 --backend gpu

# Benchmark
cargo run --release -- bench --model <model.gguf> --pp 512 --tg 128 --np 4 --gpu-runtime hip

# Model inspection
cargo run --release -- gguf-dump --meta-only <model.gguf>
```

## Repository layout

```
crates/gguf        GGUF v3 parser
crates/core        dequantization, matmul, Gated DeltaNet, qwen35 + qwen4exp engines
crates/diag        shared diagnostics (fingerprint differ, NaN guard, trace)
crates/backend-gpu raw HIP (hipRTC) + raw Vulkan (SPIR-V) GPU backends
crates/server      llm170 CLI + HTTP server
docs/              specs & decisions
```

## Documentation

- [Architecture](docs/architecture.md)
- [Benchmarks](docs/benchmarks.md)
- [Decision records](docs/decisions.md)
- [CMP 170HX hardware spec](docs/hardware/cmp170hx.md)
