# LLM170

A pure-Rust LLM inference engine. Single track: **CUDA + W4A16** — int4 sym
g128 group quantization with f16 activations — built for the NVIDIA CMP 170HX
64GB (GA100-class HBM2e, headless, sm_80); also runs on RTX 4090 (sm_89).

W4A16 fits the 170HX: no NVIDIA GPU before Blackwell has int4×fp16 tensor
cores, so every 4-bit format runs dequant → fp16 mma anyway; the split
packed/scale layout (4.125 bits/weight) keeps HBM pressure low.

`serve`/`infer` run the CUDA chain end-to-end on W4A16 directories
(compressed-tensors / AutoRound): the chain stays device-resident (per-token
round trips are only the embedding upload and the logits read) and is captured
as a CUDA graph (`LLM170_GRAPH=0` disables). 27B dense runs g128 (t=1 GEMV
decode, mma prefill); 35B-A3B MoE runs g32 experts behind a device router top-k
(experts can stream from host RAM). KV cache is int8 (single path). Every
numeric change passes CPU-oracle goldens, a frozen PPL fingerprint, and a
ΔPPL/needle gate (`w4a16-eval`) — then replaces the single path or is deleted;
runtime flags never select arithmetic.

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

Requires Rust 1.99+. No GPU toolchain needed to build.

```bash
cargo build --release

cargo run --release -- serve --model <w4a16_dir> --port 8080 --ctx auto
cargo run --release -- bench --model <w4a16_dir> --pp 512,1024,4096 --tg 128 --np 4
cargo run --release -- w4a16-gpu <w4a16_dir> --prompt-tokens 148678,65233,202419 --n-predict 8
cargo run --release -- w4a16-ref <w4a16_dir> --prompt-tokens 148678,65233,202419 --n-predict 8
cargo run --release -- w4a16-load <w4a16_dir>
```

## Repository layout

```
crates/core        W4A16 loader, CPU reference (quant lane), qwen35 CPU engine
crates/diag        diagnostics (env flags, dumps)
crates/backend-gpu rawcuda — CUDA driver bindings + kernel assets (.cu/fatbin)
crates/server      CLI + HTTP server (continuous batching)
benchmark/         measurement ledgers (dev machine: 4090.md)
```

Kernel assets: `scripts/build_cuda_kernels.sh` (nvcc, sm_80+sm_89).
Pre-commit gate: `scripts/preflight.sh`.
