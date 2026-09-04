# Backend Architecture

LLM170 runs Qwen3.8-27B (hybrid GDN + full-attention) on AMD APUs through
two independent GPU backends plus a CPU reference path. All three produce
**identical greedy streams** on the standard verification prompts.

## Paths

| Runtime | Entry | Scope |
|---|---|---|
| CPU (W4A8) | `--backend cpu` (no GPU env) | Reference engine; every kernel has a bit-matching mirror here |
| ROCm/HIP | `--backend gpu --gpu-runtime hip` | Full GPU pipeline (`rawhip`): all matmuls, flash attention, GDN scan, EW ops |
| Vulkan | `--gpu-runtime vulkan` (with `--backend cpu`) | Matmul accelerator (`rawvk`): 8-quant GEMV + coopmat tile, GPU quantize/rms/silu, FFN resident chain |

## Kernel contract

- Quantized matmul kernels mirror `dot_row_w4a8_*` in `crates/core/src/quant.rs`
  (integer `isum` per 32-element block; scales applied per block). Integer
  arithmetic is exact, so default-path outputs match the CPU reference to
  ≤2.4e-7 relative (float reduction order only).
- The optional WMMA/coopmat tiles stage operands as **f16** (maxrel ~4e-4,
  llama.cpp MMA class). They are env-gated because they trade a small
  numerical tolerance for speed.
- `exp` is computed via an f64 Horner polynomial (`exp_cr`) reproduced
  bit-identically in HIP C++, GLSL, and Rust — device `expf` differs by 1 ulp
  and breaks the contract.

## Verification gates

- `llm170 vk-check` — Vulkan device capabilities, cooperative matrix probe,
  smoke compute, axpy.
- `llm170 vk-gemv-check <model> <tensor> [t]` — per-type GEMV vs CPU mirror.
- `llm170 mm-bench2 <model> <tensor>` — HIP per-kernel timing + mismatch check.
- `scripts/logit-diff.sh` — fast-vs-exact logit divergence gate.
- Streams: GPU greedy output must equal the CPU reference token-for-token
  (the primary quality gate used for every change).

## Backend notes

### HIP (`rawhip`)
Kernels are HIP C++ strings JIT-compiled via hipRTC (`kernels.rs::SRC`), plus
optional offline code objects (`LLM170_CO_PATH` family) built by
`scripts/build_co.py` for wave32-compiled variants. DecodeState keeps
activations, KV cache, and GDN states resident on device across the whole
forward pass.

### Vulkan (`rawvk`)
GLSL compute shaders precompiled to SPIR-V by `scripts/build_spv.py` and
embedded. `VkAcc` implements the `Accelerator` trait (lazy pipelines, resident
weight cache chunked to RADV's 128MB `maxStorageBufferRange`). Shader
porting constraints discovered on RADV/gfx1151:

- cooperative-matrix ops require uniform execution across the subgroup
  (no early returns around them);
- loop-carried coopmat indexing silently drops results — fully unroll with
  independent fragment variables;
- weights >128MB must be split across multiple SSBO bindings.

## Performance (Qwen3.8-27B UD-Q4_K_XL, see benchmarks.md for full table)

| | HIP | Vulkan | CPU |
|---|---|---|---|
| decode tg24 | 10.4 t/s | 10.4 t/s | 9.9 t/s |
| prefill pp64 | 163-169 t/s | ~128 t/s | ~128 t/s |

Vulkan prefill is bounded by the CPU-side attention/GDN layers (the matmul
offload itself saturates); porting those is the next backend milestone.


## Vulkan performance path (2026-09-05)

Status: correctness-complete (stream ★ 41/41 vs CPU reference; MTP, np functional).
Throughput: tg 2.06 t/s after command-buffer batching (+11%); pp32 4.2 t/s.

Cost model per decode token (~490 ms): ~173 ms GPU submit/fence (was ~290 ms before
batching; the FFN resident chain and matmul groups now single-submit), remainder is
host-side: the VkAcc design runs GDN (49 recurrent layers: conv, AR, norm-gated) and
full attention softmax on the CPU, plus per-op activation upload/readback.

The GDN/attention GPU residency is what the HIP backend already implements
(rawhip decode.rs kernels); porting that kernel family to SPIR-V is the known path to
parity — tracked as plans/19 phase 2. On a healthy host the CPU-side share shrinks
several-fold; today's host ran ~7 h of continuous compute and its CPU-side engine
measured 10-180x slower than at session start (cores parked at 2.0 GHz).


## Vulkan GDN kernel suite (2026-09-05)

Five GDN kernels ported to SPIR-V and verified against CPU mirrors on the 27B host
(`llm170 gdn-check`): split3 (exact), gdn_conv_t (4.5e-8), gdn_beta_g (1.4e-7),
gdn_ar (1.5e-8 — subgroupAdd reduction over 32 lanes x 4 kdim, grid (dt_rank, d_state)).
A subgroup-reduction probe (`llm170 subsum-check`) validates xor-tree/add/broadcast
(496 expected, all match).

Porting RCA: GLSL must use gl_WorkGroupID (not gl_GlobalInvocationID) for block
dimensions — the latter folds blockIdx * localSize + localID, conflating the pair
index with the lane index and scrambling per-pair state. Deterministic probes
(beta=0 / g=1 / s=i+1 patterns) isolated the failure to indexing, not subgroup ops.
