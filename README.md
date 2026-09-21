# LLM170

A pure-Rust LLM inference engine. No llama.cpp. No ggml. No C/C++ toolchain.

Currently benchmarked on AMD APUs (Radeon 8060S / gfx1151, ROCm + Vulkan), with NVIDIA CMP 170HX as the original target hardware.

## Benchmarks

Solo, greedy, `llm170 bench` vs `llama-bench`, same host (2026-09-19, plans/83 close). Full conditions: [docs/benchmarks.md](docs/benchmarks.md).

### Qwen3.8-27B (Q4_K_XL 16.3 GiB)

| backend | pp512 | pp4096 | pp8192 | pp16384 | tg128@4k |
|---|---|---|---|---|---|
| LLM170 hip | **363** | 319 | 315 | 292 | 11.53 |
| LLM170 vulkan | 341 | 231 | 174.5 | 116.8 | 11.13 |
| llama.cpp hip | 340 | 318-335 | 317 | 293-297 | 11.65 |
| llama.cpp vulkan | 343 | 318 | 301 | 273 | **12.05** |

| mode | LLM170 hip | LLM170 vulkan | llama hip | llama vulkan |
|---|---|---|---|---|
| tg single | 11.5 | 11.13 | 11.65 | 12.05 |
| MTP single (k=2) | **14.4** | — | ~12 | — |
| MTP + np4 | 7.2 | n/a | 15.5 *(HTTP†)* | — |

### Qwen3.8-Flash-Next (177B-A3B, Q4_K_XL 103.7 GiB)

Vulkan qwen4exp is back (plans/86): a device-resident frame pipeline
(prefill + decode, QSA selection chain on device, pread-staged weight
uploads) now runs by default — kill switch `LLM170_VK_FRAME=0`. The VK
row (2026-09-21) is 2.3-5.2× the VK value path it replaced; absolute
parity with HIP prefill is future work (grouped tile GEMM).

| backend | pp512 | pp4096 | pp16384 | tg128@4k |
|---|---|---|---|---|
| LLM170 hip | **231-275** | 276 | 246 | **18.10-18.43** |
| LLM170 vulkan (frame) | 10.7 | 10.2 | — | 2.25 |
| llama.cpp hip | 222 | 210 | 200 | 17.43 |
| llama.cpp vulkan (coopmat) | 234 | **347** | **332** | **23.22** |

| mode | LLM170 hip | llama hip |
|---|---|---|
| tg single | **18.10-18.43** | 17.43 |
| np4 aggregate | **45.9** | 41.1 *(HTTP†)* |

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
