# LLM170

A pure-Rust LLM inference engine. No llama.cpp. No ggml. No C/C++ toolchain.

Currently benchmarked on AMD APUs (Radeon 8060S / gfx1151, ROCm + Vulkan), with NVIDIA CMP 170HX as the original target hardware.

## Benchmarks

Solo, greedy, `llm170 bench` vs `llama-bench`, same host (2026-09-23, plans/92;
single-rep numbers carry a ±5-10% thermal / page-cache spread on this APU —
ranges where observed). Full conditions: [docs/benchmarks.md](docs/benchmarks.md).

<!-- 측정 프로토콜: hip(ROCm) 수치는 반드시 ROCm 10으로 측정한다
     (LD_LIBRARY_PATH=/opt/rocm-10.0.0/install/lib). Vulkan 수치는 무관. -->

### Qwen3.8-27B (Q4_K_XL 16.3 GiB)

| backend | pp512 | pp4096 | pp8192 | pp16384 | tg128@4k |
|---|---|---|---|---|---|
| LLM170 hip | **371** | **327** | **314** | 290 | 11.46 |
| LLM170 vulkan | 351.8 | 299.7 | 272.9 | 231.1 | **11.81** |
| llama.cpp hip | 328 | 321 | 314 | **301** | 9.42 |
| llama.cpp vulkan | 331 | 308 | 291 | 262 | 11.54 |

| mode | LLM170 hip | LLM170 vulkan | llama hip | llama vulkan |
|---|---|---|---|---|
| tg single | **11.66** | 10.77 | 11.34 | 11.33 |
| np4 greedy (GPU argmax) | 33.31 | **34.65** | — | — |
| np4 full-logits | 10.86 | 10.48 | **30.09** *(HTTP†)* | — |
| MTP single (k=2) | 6.17 | 0.90 *(spec2 구현·수용률 미조정)* | **~11.3** | — |
| MTP + np4 | 13.12 | n/a | **30.09** *(HTTP†)* | — |

### Qwen3.8-Flash-Next (177B-A3B, Q4_K_XL 103.7 GiB)

<!-- 금지: 이 표 위·아래에 엔진 구현 설명, 세션 노트, 백엔드 변동사 등
     어떤 프로즈도 적지 않는다. 표 안의 측정값만 갱신한다. -->
| backend | pp512 | pp4096 | pp16384 | tg128@8k |
|---|---|---|---|---|
| LLM170 hip | 246.2 | 278.8 | 249.8 | 5.86 |
| LLM170 vulkan (frame) | 465.8 | 395.9 | 296.4 | 14.82 |
| llama.cpp hip | 451 | 427 | 391 | 20.61 |
| llama.cpp vulkan | **474** | **502** | **448.8** | **23.68** |

| mode | LLM170 hip | llama hip |
|---|---|---|
| tg single | 5.86 | **20.04** |
| np4 aggregate | 28.6 *(serve --slots 4, HTTP)* | **49.03** *(llama HTTP†)* |

- Greedy gates: 27B hip / 27B vulkan / FN hip / FN vulkan all PASS
  (`scripts/gate-27b.sh`, `scripts/gate-flash-next.sh`).
- Tokenizer: 100% llama.cpp token-for-token match — 86 corpus files ×
  special on/off, 398,177 tokens, both models (`scripts/verify_tok.py`).
- Sampling: seed-reproducible, temp→0 = argmax; temperature / top_k /
  top_p / min_p / repeat_penalty / seed on all completion endpoints.
- Generation: Korean Q&A over HTTP answers correctly (e.g. "서울") with
  coherent `<think>` reasoning; chunk invariance fenced by
  `llm170 diag chunk-check`.

## Build & run

Requires Rust 1.95+ (edition 2024). No GPU toolchain needed to build.

```bash
cargo build --release

# Inference
cargo run --release -- infer --model <model.gguf> --prompt-tokens 760,6511 --n-predict 16 --gpu-runtime hip

# HTTP server (OpenAI-compatible)
cargo run --release -- serve --model <model.gguf> --port 8080 --slots 4 --backend gpu   # --slots N: continuous batching (default 1)

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
