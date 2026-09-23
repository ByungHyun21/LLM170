# LLM170

A pure-Rust LLM inference engine. No llama.cpp. No ggml. No C/C++ toolchain.

Currently benchmarked on AMD APUs (Radeon 8060S / gfx1151, ROCm + Vulkan), with NVIDIA CMP 170HX as the original target hardware.

## Benchmarks

Solo, greedy, `llm170 bench` vs `llama-bench`, same host (2026-09-23, plans/92;
single-rep numbers carry a ±5-10% thermal / page-cache spread on this APU —
ranges where observed). Full conditions: [docs/benchmarks.md](docs/benchmarks.md).

### Qwen3.8-27B (Q4_K_XL 16.3 GiB)

| backend | pp512 | pp4096 | pp8192 | pp16384 | tg128@4k |
|---|---|---|---|---|---|
| LLM170 hip | **363** | 319 | 315 | 292 | 11.53 |
| LLM170 vulkan | **360-365** | 302 | 273 | **231** | 11.2 |
| llama.cpp hip | 356-359 | 337 | — | **312** | 11.7 |
| llama.cpp vulkan | 333 | 319 | — | 278 | 12.0 |

| mode | LLM170 hip | LLM170 vulkan | llama hip | llama vulkan |
|---|---|---|---|---|
| tg single | 11.5 | 11.2 | 11.7 | 12.0 |
| np4 greedy (GPU argmax) | 33.2 | **33.5** | — | — |
| np4 full-logits | **30.6** | 27.4 | 15.5 *(HTTP†)* | — |
| MTP single (k=2) | **14.4** | 2.5 *(spec2 구현·수용률 미조정)* | ~12 | — |
| MTP + np4 | 7.2 | n/a | 15.5 *(HTTP†)* | — |

### Qwen3.8-Flash-Next (177B-A3B, Q4_K_XL 103.7 GiB)

Vulkan qwen4exp runs a device-resident frame pipeline (plans/86-89):
prefill + decode on device, llama-dmmv decode GEMV family, coopmat dense
prefill tiles, device MoE tiles (q5_1 down coopmat sg1 default, q4_K scalar
tile — the coopmat variant is race-blocked, see below), PLE math on device
(bit-identical to host), step-level batching, pread-staged weight uploads.
Kill switch `LLM170_VK_FRAME=0`; MoE coopmat tiles opt-in
`LLM170_VK_MOECM=1` (RADV subgroup-scheduling race, q4_K 1-sg included —
docs/decisions.md (32b)/(32c), repro: `scripts/moecm-repro.sh`).


| backend | pp512 | pp4096 | pp16384 | tg128@8k |
|---|---|---|---|---|
| LLM170 hip | **231-275** | 276 | 246 | **18.4** |
| LLM170 vulkan (frame) | 192-223 | 198-213 | 166 | 14.7 |
| llama.cpp hip | 490-520 | 478 | 415 | 21.2 |
| llama.cpp vulkan | 506 | 488 | **433** | **23.6** |

Session 2026-09-22 (plans/89): OpSDot q4_K MoE tile (4.85x), rms coalescing
(8x/dispatch), dense tile routing completion — FN pp512 131.8 -> ~180.
Session 2026-09-23 (plans/92): register-resident prefill flash
(qsa_flash_reg — LDS/barrier-free; +59% on 27B pp16384), multi-row rms
wide plate, tile128 single dispatch — FN vk pp512 ~180 -> 192-223.

llama.cpp refreshed 2026-09-23 (b1ff4ca23; was 2026-08 builds): upstream
qwen4exp prefill matured — FN llama pp jumped 222-234 -> 490-520
(flash-attn auto + MoE batching; today's tip lands IQ4_XS MMQ/MMV on
Vulkan), 27B pp16384 +5%. Our FN frame pipeline is now the chase side
(~2.3x behind llama pp); MoE-cm race and decode dmmv remain the blockers.

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
