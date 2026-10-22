# LLM170

Pure-Rust LLM inference engine — single track **CUDA + W4A16** (int4 sym g128,
f16 activations) for the NVIDIA CMP 170HX 64GB (sm_80).
`serve`/`infer` run W4A16 directories (compressed-tensors / AutoRound)
end-to-end on GPU with an int8 KV cache and CUDA-graph decode — 27B dense
(g128) and 35B-A3B MoE (g32).

## Benchmarks



### Environment (CMP 170HX)

| item | value |
|---|---|
| GPU | 4x NVIDIA CMP 170HX (GA100, sm_80, 64 GB each) |
| driver | 610.57.04 (unlock patch, pinned) |
| PCIe | Gen2 x16 all 4 |
| power limit | 180 W (fixed) |
| SM / L2 | 70 SM; 32 MiB L2; persisting max 20 MiB |
| host | AMD EPYC 7452 32C/64T, 107 GB RAM |

### Throughput (CMP 170HX, 180 W)

| model | cards | pp512 | pp1024 | pp4096 | tg128@4k | np4 |
|---|---|---|---|---|---|---|
| Qwen3.8-27B-W4A16 | 1 | 1409.30 | 1429.20 | 1368.90 | 43.74 | 153.44 |
| Qwen3.6-35B-A3B-W4A16 | 1 | 4635.51 | 4572.49 | 4292.66 | 118.94 | 287.31 |
| Gemma-4-26B-A4B-it-QAT-W4A16 | 1 | 4591.30 | 4319.32 | 3654.96 | 74.90 | — |
| Qwen3.8-Flash-Next-W4A16 | 2 | 223 | — | 258 | — | — |

### Decode mode (512 tok prompt · 128 gen)

| model | greedy t/s | MTP t/s | n-gram t/s |
|---|---|---|---|
| Qwen3.8-27B-W4A16 | 37.4 | 41.6 | 52.6 |
| Qwen3.6-35B-A3B-W4A16 | 114.2 | — | 114.1 |

### Serve (27B)

| model | TTFT cold ms | TTFT warm ms | prefix reuse ratio |
|---|---|---|---|
| Qwen3.8-27B-W4A16 | 1054 | 305 | 0.582 |

## Build & run

Requires Rust 1.99+. No GPU toolchain needed to build.

```bash
cargo build --release

cargo run --release -- serve --model <w4a16_dir> --kv-mem 24G --port 8080
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
```

Kernel assets: `scripts/build_cuda_kernels.sh` (nvcc, sm_80); kernel
outputs must match the CPU reference (`crates/core/src/quant/lane.rs`).
Pre-commit gate: `scripts/preflight.sh` · numeric gate: `w4a16-eval`
(goldens · ΔPPL band · needle).
