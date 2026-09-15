# LLM170

A **pure-Rust** LLM inference engine built from the ground up — originally
for the NVIDIA CMP 170HX, currently developed and benchmarked on AMD APUs
(Radeon 8060S / gfx1151, ROCm + Vulkan) with portability as a standing goal.

No llama.cpp. No ggml. No C/C++ toolchain. Every layer of the stack — GGUF parsing, quantization, kernels, scheduling, profiling — is implemented in Rust from scratch.

> **Why the CMP 170HX?** It's a GA100 (A100-class) die sold as a mining card at a fraction of the price: 8 GB HBM2e at ~1.5 TB/s, sm_80, and standard NVIDIA drivers. The catch: eFUSE throttling caps FP32 FFMA at 1/32 rate and tensor cores at ~12% — while leaving **FP16 half2 (~42 TFLOPS) and INT32 at full speed**. Stock inference stacks are crippled by this; an engine designed around the constraint is not. With the 2026 community unlock (64 GB), the card becomes a serious decode machine. Details in [docs/hardware/cmp170hx.md](docs/hardware/cmp170hx.md).

## Benchmarks

Qwen3.8-27B (hybrid GDN + full attention), greedy, single-tenant `llm170 bench`.
Numbers are only ever quoted with their conditions — full tables and history in
[docs/benchmarks.md](docs/benchmarks.md).

### CMP 170HX (GA100) — benchmark in preparation

| Mode | pp512 | tg128 |
|---|---|---|
| `cmp-stock` (8 GB) | — | — |
| `cmp-unlocked` (64 GB) | — | — |

### Strix Halo (Ryzen AI Max+ 395 / Radeon 8060S, gfx1151)

Greedy, single-tenant solo runs, same GGUF, t/s; unmeasured cells are blank.
Session gains (2026-09-15, hip): Flash-Next tg 13.40 → 17.5 (+31%, short ctx)
and 11.3 → 16.8 (+49%, 16k) — QSA indexer selection and PLE moved onto the GPU,
five warp-per-row GEMV kernels, bit-identical bitonic top-k; all gate streams
unchanged. The vulkan rows reflect the pre-port path for 27B (the current
optimization set is hip-only) and the Vulkan-era Flash-Next work; "—" cells
failed with ERROR_DEVICE_LOST under the Vulkan driver at that shape.

#### Qwen3.8-27B (Q4_K_XL 16.3 GiB)

| backend | pp512 | pp4096 | pp16384 | tg128@4k | tg128@16k |
|---|---|---|---|---|---|
| LLM170 hip | **374** | 324 | 253 | 11.1 | 11.5 |
| LLM170 vulkan | — | 150 | — | 9.2 | — |
| llama.cpp (ROCm 10) | 347 | **342** | **317** | **11.6** | **11.9** |

(Measured 2026-09-16 on the current hip build, greedy, natural-text prompt, pp512
prompt / ctx 4096 (tg@4k) and 16384 (tg@16k).)

Decode modes (aggregate t/s over 4 parallel slots where noted; MTP =
`--spec 3`). MTP does not change prefill — the np4 pp aggregate applies
unchanged under MTP+np4:

| mode | pp agg | tg agg | pp agg | tg agg | pp agg | tg agg |
|---|---|---|---|---|---|---|
| | **LLM170 hip** | | **LLM170 vulkan** | | **llama.cpp (ROCm 10)** | |
| tg single | — | 11.1 (4k) / 11.5 (16k) | — | 9.2 | — | 11.6 / 11.9 |
| MTP single | — | **14.3** | — | 9.2 (no MTP) | — | ~12 (MTP) |
| np4 aggregate | 374 (pp512) | **25.1** | T27NPP4V | T27NP4V | L27NPP4 | L27NP4 |
| MTP + np4 | (np4) | **20.4** | — | — | (np4) | 15.5 |

Conditions for the filled 2026-09-16 cells: HIP, ROCm 10, greedy, natural-text
prompt, pp512 / ctx 8192 / tg128, all np slots prefilled with the same prompt
(`LLM170_BENCH_NP`). llama's np4+MTP 15.5 t/s is the recorded server reference
(ROCm 7.2.2 era, 11.75k-token slots) — the comparison is cross-condition, noted
as such. MTP = `--spec 3`; acceptance is perfect (4 tokens/cycle) with the
gguf's own nextn head, and spec output is token-identical to greedy
(`scripts/verify.py` spec cases).

#### Qwen3.8-Flash-Next (177B-A3B, Q4_K_XL 103.7 GiB)

| backend | pp4096 | pp16384 | tg128@4k | tg128@16k |
|---|---|---|---|---|
| LLM170 hip | **270** | **243** | 18.0 | 17.9 |
| LLM170 vulkan | 271 | 240 | 17.1 | 16.7 |
| llama.cpp (ROCm 10) | 237 | 229 | **20.2** | **20.0** |

(Decode cells re-measured 2026-09-16: 16-lane/row GEMV for small shapes,
17.2 -> 18.0 t/s at the standard point, gate stream identical.)

Decode modes (aggregate t/s over 4 parallel slots; the model has no
nextn/MTP head — MTP rows are structurally inapplicable):

| mode | pp agg | tg agg | pp agg | tg agg | pp agg | tg agg |
|---|---|---|---|---|---|---|
| | **LLM170 hip** | | **LLM170 vulkan** | | **llama.cpp (ROCm 10)** | |
| tg single | — | **18.0** (ctx 4k) / **17.9** (ctx 16k) | — | 17.1 | — | 19.8 / 20.0 |
| MTP single | — | — | — | — | — | — |
| np4 aggregate | — | **21.0-22.5** | TFNPP4V | TFNP4V | — | **39.4** |
| MTP + np4 | — | — | — | — | — | — |

(Flash-Next np4, measured 2026-09-16 same-host/same-prompt HTTP 4-way:
**LLM170 21.0-22.5 t/s aggregate** vs **llama-server 39.4 t/s** (0.55x). The
frame now batches np decode (2026-09-16): weight-streaming GEMMs run once for
all rows — a new multi-token q8_0 GEMV (`gemm_q8_0_mt`, one weight-row read,
per-token accumulation, bit-identical arithmetic) removed the per-row weight
re-read — while per-sequence state (GDN conv ring, AR, QSA rope/selection/KV,
PLE) runs per row at t=1 through row views. np4 output is token-identical to
sequential decoding (52/52 verified; np2 shows one documented near-tie flip,
logit gap 0.13). The remaining gap is structural: rows pick different experts
(diverse prompts share almost none), so the MoE weight traffic is irreducibly
per-row — dedup would only pay under identical-prompt routing, i.e. the
benchmark condition itself, which is not optimized for on principle.)

Full analysis: [docs/benchmarks.md](docs/benchmarks.md).

## Safety: pre-load resource guard

Before any subcommand loads a model, the engine compares the model's size
(all split parts) against free VRAM + 0.85 x available RAM and refuses to
start when it would not fit — instead of driving the host into an OOM freeze
(`LLM170_NO_RSRC_GUARD=1` overrides). This exists because running two
inference servers on one 96 GiB APU with a 104 GiB model froze the machine
once (2026-09-16).

## Build & run

Requires Rust 1.95+ (stable, edition 2024). No GPU or CUDA toolkit needed to build.

```bash
cargo build --release

# Inspect a model — structure, hyperparameters, quantization mix
cargo run --release -- gguf-dump --meta-only <model.gguf>

# Greedy CPU inference (token-id input)
cargo run --release -- infer \
    --model <model.gguf> \
    --prompt-tokens 760,6511,314,9338,369 \
    --n-predict 16

# Same inference with the GPU-resident decoder.
# --gpu-runtime hip (ROCm, default) or vulkan; weights upload once and stay resident.
cargo run --release -- infer \
    --model <model.gguf> \
    --prompt-tokens 760,6511,314,9338,369 \
    --n-predict 16 \
    --gpu-runtime vulkan

# HTTP server: /health, /v1/models, /tokenize, /v1/completions, /v1/chat/completions (SSE),
# /v1/messages (Anthropic). Continuous batching across slots; client disconnects cancel the job.
cargo run --release -- serve --model <model.gguf> --port 8080 --backend gpu

# llama-bench-style PP/TG measurement (t/s). --spec k: effective t/s with MTP speculative decode.
cargo run --release -- bench --model <model.gguf> --pp 512 --tg 128 --gpu-runtime hip

# Runtime mode (memory-budget profile today; kernel variants when cmp-stock lands)
cargo run --release -- infer --model <model.gguf> --prompt-tokens 760,6511 --n-predict 16
```

The HIP attention path is a recent redesign worth naming: the prefill kernel is
an fp16 WMMA tile implementation (Q consumed into registers and its shared
buffer reused as the K/V tile, 32 KB of dynamic shared) that replaced a
shuffle-bound scalar kernel, and the decode kernel follows llama.cpp's tile
structure — one key per lane with the whole 256-dim dot computed in a single
thread via `v_dot2_f32_f16`, so the QK phase has no cross-lane reduction at
all — reading an f16 KV mirror maintained at the KV write sites. Both were
validated against CPU references (`wmma-attn-check`, `gqa-bench`) before
becoming defaults.

GPU kernel self-check subcommands (cross-validated against the CPU reference):
`vk-check` (device/coopmat smoke) · `gdn-check` (GDN/attention kernel suite) ·
`vk-gemv-check` / `vk-gemv8-check` (per-type GEMV) · `rawhip-check` (HIP GEMV
bit-parity) · `subsum-check` (subgroup reductions) · `qk-check` ·
`wmma-attn-check` (WMMA attention vs CPU reference) · `gqa-bench` (decode
attention variants, timing + differential) ·
`iq3s-probe` · `check` (tensor scan + cross-validation + chunk smoke) ·
`w4a8-check`. Run `llm170 help` for the full list.

Debug builds are fully instrumented — every stage is timed by the built-in profiler, by design. Release builds carry zero instrumentation.

GPU kernels are HIP C++ (JIT-compiled at runtime via hipRTC from per-family
source assets) and GLSL compute shaders (precompiled to SPIR-V), with Rust
owning all orchestration — one kernel source per backend, arithmetic mirrored
from the CPU reference for bit-level verification. ROCm/HIP drives AMD GPUs;
the Vulkan backend needs only a conformant driver. See
[docs/backend-architecture.md](docs/backend-architecture.md).

## Repository layout

```
crates/gguf        GGUF v3 parser
crates/core        dequantization, matmul, Gated DeltaNet, qwen35 + qwen4exp engines
crates/backend-gpu raw HIP (hipRTC) + raw Vulkan (SPIR-V) GPU backends
crates/profiler    debug-gated lightweight profiler
crates/server      llm170 CLI + HTTP server
docs/              specs & decisions (hardware, models, architecture, ADRs, benchmarks)
scripts/           verification harness (synthetic-model generators, llama.cpp cross-checks)
```

## Verification philosophy

- **Cross-checked against llama.cpp** greedy token streams, with a near-tie-aware standard for flat-top distributions (top-6 membership + logprob-gap threshold).
- Internal consistency where it's cheap and strong: the chunked and autoregressive GDN paths are cross-validated on arbitrary inputs, and every GPU kernel is checked against the CPU reference before entering the engine.
- Performance numbers are only ever quoted with their conditions (context / batch / quantization / backend) — see [docs/benchmarks.md](docs/benchmarks.md).

## Documentation

- [Project overview](docs/overview.md) — goals, modes, reference models
- [Architecture](docs/architecture.md) — backend strategy, FP discipline, kernel surface
- [Benchmarks](docs/benchmarks.md) — pp/tg comparison with conditions
- [qwen35 spec](docs/models/qwen35.md) · [qwen4exp spec](docs/models/qwen4exp.md) — tensor-level implementation specs
- [CMP 170HX hardware spec](docs/hardware/cmp170hx.md) — throttle matrix, unlock, memory budgets
- [Decision records](docs/decisions.md) — ADRs
