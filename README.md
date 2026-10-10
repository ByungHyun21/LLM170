# LLM170

A pure-Rust LLM inference engine. Single track: **CUDA + W4A16** — int4 sym
g128 group quantization with f16 activations — built for the NVIDIA CMP 170HX
64GB (GA100-class HBM2e, headless, sm_80).

W4A16 fits the 170HX well: no NVIDIA GPU before Blackwell has int4×fp16
tensor cores, so every 4-bit format runs dequant → fp16 mma on the tensor
cores anyway. The split packed/scale layout (4.125 bits/weight) keeps HBM2e
pressure low and rides the mature GPTQ-style kernel line.

Current state: `serve`/`infer` run the **CUDA chain** end-to-end on W4A16
directories (compressed-tensors / AutoRound auto_gptq packing —
`.weight_packed` · `.weight_scale` · `.weight_shape`): weights are
VRAM-resident, the layer chain runs device-resident (activations stay on
GPU — the only per-token round trips are the embedding upload and the logits
read), and the chain is captured as a CUDA graph (one launch per token;
`LLM170_GRAPH=0` falls back to direct launches). Dense models (27B) run
g128 split weights — t=1 GEMV decode + mma prefill (register dequant,
cp.async staging); 35B-A3B MoE runs g32 split experts behind a device
router top-k, with a plain bf16 path for attention/shared layers and
batched expert dispatch. KV cache is int8 (single path). The token stream
matches the golden set; module gates are `w4a16-gemv`/`w4a16-gemm` (bit
judgment vs `dot_row_w4a16_lane`) and `w4a16-eval ppl|agree` tracks
quality drift.

Bit contract: CUDA kernel outputs must match the CPU reference
(`crates/core/src/quant/lane.rs` — `dot_row_w4a16_lane`).

## Benchmarks

<!-- 표 외 텍스트 금지 (repo rule) — 값은 표 안에만. -->
Dev-machine ledger (RTX 4090): [benchmark/4090.md](benchmark/4090.md)

### Qwen3.8-27B-W4A16 (g128)

| backend | pp512 | pp4096 | pp16384 | tg128@4k |
|---|---|---|---|---|
| CUDA (170HX, sm_80) | — | — | — | — |

| mode | CUDA (170HX) |
|---|---|
| tg single | — |
| np4 greedy | — |
| serve | — |

## Build & run

Requires Rust 1.99+ (edition 2024). No GPU toolchain needed to build.

```bash
cargo build --release

# Reference runner (CPU oracle — token judgment / debugging)
cargo run --release -- w4a16-ref <w4a16_dir> --prompt-tokens 148678,65233,202419 --n-predict 16

# HTTP server (OpenAI/Anthropic-compatible; CUDA chain)
cargo run --release -- serve --model <w4a16_dir> --port 8080 --slots 4   # --slots N: continuous batching (default 1)

# GPU chain run (probe) / CPU oracle
cargo run --release -- w4a16-gpu <w4a16_dir> --prompt-tokens 148678,65233,202419 --n-predict 8
cargo run --release -- w4a16-ref <w4a16_dir> --prompt-tokens 148678,65233,202419 --n-predict 8

# Loader completeness check / tokenizer
cargo run --release -- w4a16-load <w4a16_dir>
cargo run --release -- tokenize --model <w4a16_dir> --text "hello"
```

## Repository layout

```
crates/core        W4A16 loader, CPU reference kernels (quant lane), qwen35 CPU engine,
                   safetensors/JSON readers
crates/diag        shared diagnostics (env flags, dumps, fallback ledger)
crates/backend-gpu rawcuda — hand-rolled CUDA driver bindings + kernel assets (.cu/fatbin)
crates/server      llm170 CLI + OpenAI/Anthropic HTTP server (continuous batching)
benchmark/         measurement ledgers (dev machine: 4090.md)
```

Kernel asset rebuild: `scripts/build_cuda_kernels.sh` (nvcc, sm_80+sm_89).
Pre-commit gate: `scripts/preflight.sh`.
