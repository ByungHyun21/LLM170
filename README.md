# LLM170

A pure-Rust LLM inference engine. No llama.cpp. No ggml. No C/C++ toolchain.

Currently benchmarked on AMD APUs (Radeon 8060S / gfx1151, ROCm + Vulkan), with NVIDIA CMP 170HX as the original target hardware.

## Benchmarks

<!-- 측정 프로토콜: hip(ROCm) 수치는 반드시 ROCm 10으로 측정한다
     (LD_LIBRARY_PATH=/opt/rocm-10.0.0/install/lib). Vulkan 수치는 무관. -->

### Qwen3.8-27B (Q4_K_XL 16.3 GiB)

| backend | pp512 | pp4096 | pp8192 | pp16384 | tg128@4k |
|---|---|---|---|---|---|
| LLM170 hip | 350.6 | 323 | 312 | 291 | **11.55** |
| LLM170 vulkan | 336 | 318 | 308 | 284 | 10.02 |
| llama.cpp hip | **350** | **335** | **327** | **313** | 9.56 |
| llama.cpp vulkan | 331 | 308 | 291 | 262 | 11.54 |

| mode | LLM170 hip | LLM170 vulkan | llama hip | llama vulkan |
|---|---|---|---|---|
| tg single | **11.55** | 10.02 | 9.56 | 11.33 |
| np4 greedy | 33.18 | **33.62** | 29.22 | — |
| MTP k=2 | **12.64** | 12.51 | 17.40 | — |
| MTP k=3 | 10.21 | 10.11 | 17.22 | — |
| MTP + np4 k=2 | 14.43 | n/a | **35.97** | — |
| MTP + np4 k=3 | 12.71 | n/a | 33.95 | — |

### Qwen3.8-Flash-Next (177B-A3B, Q4_K_XL 103.7 GiB)

| backend | pp512 | pp4096 | pp16384 | tg128@8k |
|---|---|---|---|---|
| LLM170 hip | 299.1 | 291.4 | 201.9 | 14.11 |
| LLM170 vulkan (frame) | 401.9 | — | — | — |
| llama.cpp hip | **483** | **482** | **447** | **19.88** |
| llama.cpp vulkan | 474 | 502 | 448.8 | 23.68 |

| mode | LLM170 hip | llama hip |
|---|---|---|
| tg single | 14.11 | **19.88** |
| np4 greedy | 33.19 | **49.03** |
| MTP k=2 | 7.5 | n/s |
| MTP k=3 | 7.0 | n/s |
| MTP + np4 k=2 | 8.41 | n/s |
| MTP + np4 k=3 | 7.27 | n/s |

<!-- 금지: 이 영역에 게이트 통과·기능 나열·세션 노트 등 잡다한 산출물을
     적지 않는다 — 표 안의 측정값만 갱신한다 (repo 규칙). -->

## Build & run

Requires Rust 1.95+ (edition 2024). No GPU toolchain needed to build.

```bash
cargo build --release

# Inference
cargo run --release -- infer --model <model.gguf> --prompt-tokens 760,6511 --n-predict 16 --backend hip

# HTTP server (OpenAI-compatible)
cargo run --release -- serve --model <model.gguf> --port 8080 --slots 4 --backend hip   # --slots N: continuous batching (default 1)

# Benchmark
cargo run --release -- bench --model <model.gguf> --pp 512 --tg 128 --np 4 --backend hip

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
