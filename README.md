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
| LLM170 hip | 128.72 | — | — | 5.78 |

| mode | LLM170 vulkan | LLM170 hip |
|---|---|---|
| tg single | 3.31 | 5.78 |
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

### Q4_K (GGUF) — deprecated (EXL3 transition, 2026-10)

#### Qwen3.8-27B (Q4_K_XL 16.3 GiB)

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

#### Qwen3.8-Flash-Next (177B-A3B, Q4_K_XL 103.7 GiB)

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
