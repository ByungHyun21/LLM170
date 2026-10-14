# LLM170

Pure-Rust LLM inference engine — single track **CUDA + W4A16** (int4 sym g128,
f16 activations) for the NVIDIA CMP 170HX 64GB (sm_80); also runs on RTX 4090
(sm_89). `serve`/`infer` run W4A16 directories (compressed-tensors / AutoRound)
end-to-end on GPU with an int8 KV cache and CUDA-graph decode — 27B dense
(g128) and 35B-A3B MoE (g32).

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

Kernel assets: `scripts/build_cuda_kernels.sh` (nvcc, sm_80+sm_89); kernel
outputs must match the CPU reference (`crates/core/src/quant/lane.rs`).
Pre-commit gate: `scripts/preflight.sh` · numeric gate: `w4a16-eval`
(goldens · ΔPPL band · needle).
