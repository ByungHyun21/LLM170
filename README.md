# LLM170

A pure-Rust LLM inference engine. No llama.cpp. No ggml. No C/C++ toolchain.

Currently benchmarked on AMD APUs (Radeon 8060S / gfx1151, ROCm + Vulkan), with NVIDIA CMP 170HX as the original target hardware.

## Benchmarks

<!-- 이 섹션에는 표 외 텍스트 기입 금지 (repo rule) — 표 안에는 값만 기입.
     프로토콜: hip 수치는 ROCm 10(LD_LIBRARY_PATH=/opt/rocm-10.0.0/install/lib),
     bench --reps 3 중앙값. vk는 독립. -->

### Qwen3.8-27B (EXL3 SC_4.00bpw, 16.35 GB)

| backend | pp512 | pp4096 | pp16384 | tg128@4k |
|---|---|---|---|---|
| LLM170 vulkan | 137.71 | — | — | 3.31 |
| LLM170 hip | 133.12 | — | — | 6.72 |

| mode | LLM170 vulkan | LLM170 hip |
|---|---|---|
| tg single | 3.31 | 6.72 |
| pp batch512 | 137.71 | 128.72 |
| serve | 3.9 | 5.62 |
| MTP k=2 | — | — |
| np4 greedy | — | — |

### Qwen3.8-Flash-Next (EXL3 5.05bpw, 123.1 GB)

| backend | pp512 | pp4096 | pp16384 | tg128@8k |
|---|---|---|---|---|
| LLM170 vulkan | — | — | — | — |

| mode | LLM170 vulkan |
|---|---|
| tg single | 2.55 (direct trellis) |
| np4 greedy | — |
| MTP k=2 | — |
| MTP k=3 | — |
| MTP + np4 k=2 | — |
| MTP + np4 k=3 | — |

### DeepSeek-V4-Flash-Vision-Exp (EXL3 3.04bpw, 118.4 GB, vision + MTP3)

| backend | pp512 | pp4096 | tg128 | VL |
|---|---|---|---|---|
| LLM170 vulkan | — | — | — | — |

| mode | LLM170 vulkan |
|---|---|
| tg single | 2.55 (direct trellis) |
| np4 greedy | — |
| MTP k=2 | — |
| MTP k=3 | — |
| MTP + np4 k=2 | — |
| MTP + np4 k=3 | — |

### Q4_K (GGUF)

#### Qwen3.8-27B (Q4_K_XL 16.3 GiB)

| backend | pp512 | pp4096 | pp8192 | pp16384 | tg128@4k |
|---|---|---|---|---|---|
| LLM170 hip | 360.0 | 328.6 | 316.6 | 291.5 | 11.57 |
| LLM170 vulkan | 338.6 | 318 | 308 | 284 | 11.47 |
| llama.cpp hip | 362.68 | **340.10** | **331.13** | **316.94** | 11.89 |
| llama.cpp vulkan | **384.85** | 347.56 | 323.94 | 296.85 | **12.17** |

| mode | LLM170 hip | LLM170 vulkan | llama hip | llama vulkan |
|---|---|---|---|---|
| tg single | 11.55 | 10.02 | 11.89 | **12.17** |
| np4 greedy | 33.40 | **33.62** | 28.35 | 23.00 |
| MTP k=2 | 18.93 | 12.51 | **19.88** | 18.41 |
| MTP k=3 | 16.91 | 10.11 | **21.37** | 18.94 |
| MTP + np4 k=2 | 23.99 | n/a | **40.79** | 23.86 |
| MTP + np4 k=3 | 14.68 | n/a | **43.39** | 23.94 |

#### Qwen3.8-Flash-Next (177B-A3B, Q4_K_XL 103.7 GiB)

| backend | pp512 | pp4096 | pp16384 | tg128@8k |
|---|---|---|---|---|
| LLM170 hip | 319.1 | 332.0 | 308.3 | 14.57 |
| LLM170 vulkan (frame) | 401.9 | — | — | — |
| llama.cpp hip | 417.67 | **511.95** | **466.97** | 21.81 |
| llama.cpp vulkan | **546.01** | 497.66 | 478.40 | **25.63** |

| mode | LLM170 hip | llama hip |
|---|---|---|
| tg single | 14.57 | **22.07** |
| np4 greedy | 31.90 | **41.01** |
| MTP k=2 | 3.02 | n/s |
| MTP k=3 | 4.05 | n/s |
| MTP + np4 k=2 | 3.28 | n/s |
| MTP + np4 k=3 | 5.07 | n/s |

<!-- 표 외 텍스트 금지 — 표 안에는 값만 (plan 참조 등 부가 설명 기입 금지, repo rule). -->

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
```
