# Backend Architecture

LLM170 runs Qwen3.8-27B (hybrid GDN + full-attention) on AMD APUs through
two independent GPU backends plus a CPU reference path. All three produce
**identical greedy streams** on the standard verification prompts.

## Paths

| Runtime | Entry | Scope |
|---|---|---|
| CPU (W4A8) | `--backend cpu` (no `--gpu-runtime`) | Reference engine; every kernel has a bit-matching mirror here |
| ROCm/HIP | `--gpu-runtime hip` | Full GPU pipeline (`rawhip`): all matmuls, flash attention, GDN scan, EW ops |
| Vulkan | `--gpu-runtime vulkan` | GPU-resident decoder (`rawvk` VkDecoder): 8-quant GEMV + coopmat tile, GPU quantize/rms/silu, FFN resident chain |

- Routing is driven by `--gpu-runtime` alone; the `--backend` flag is
  vestigial for engine selection in `infer`/`bench`. The legacy
  CPU-engine+Vulkan-accelerator mode (VkAcc) remains opt-in via
  `LLM170_VK_ACC=1`.

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
- `llm170 vk-gemv-check <model> <tensor> [t]` — engine path (quant+gemv3)
  GEMV vs CPU mirror; `vk-gemv8-check` — gemv8 family vs dequant dot.
- `llm170 mm-bench2 <model> <tensor>` — HIP per-kernel timing + mismatch check.
- `scripts/logit-diff.sh` — fast-vs-exact logit divergence gate.
- Streams: GPU greedy output must equal the CPU reference token-for-token
  (the primary quality gate used for every change).

## Backend notes

### HIP (`rawhip`)
Kernels are HIP C++ assets JIT-compiled via hipRTC (see "Kernel sources"
below), plus
optional offline code objects (`LLM170_CO_PATH` family) built by
`scripts/build_co.py` for wave32-compiled variants. DecodeState keeps
activations, KV cache, and GDN states resident on device across the whole
forward pass.

## GEMV routing (rawvk, 2026-09-08)

Type-driven, one table (decoder.rs `gemv_w`):

| Condition | Path |
|---|---|
| t<16, ty ∈ {q3_K, q4_K, q5_K, q6_K, iq4_xs} | `gemv8` — llama mul_mat_vec port, f32 activations direct (kill-switch `LLM170_G8=0`) |
| t≥16 | coopmat f16 tiles — default prefill path (kill-switch `LLM170_VK_NOTILE=1`; batch prefill kill-switch `LLM170_VKD_BATCH=0`) |
| everything else (q8_0, iq4_nl, iq3_s at t<16) | `quant` + `gemv3` integer path |
| experimental `LLM170_VK_I8ON` | i8 coopmat GEMM (superseded by tiles; kept for the integer-MMA contract) |

The gemv4/5/6/7 generations were deleted after gemv8 promotion; the q6_K
gemv8 port (u16 block view, 160 GB/s, bit-exact vs CPU at t=1) landed
2026-09-08 and replaced the gemv3 route for q6_K decode weights and the
output head. `vk-gemv8-check` verifies each type against the CPU dequant
dot. Batch prefill and tiles became defaults after the 2026-09-04 divergence
was root-caused to the (fixed) descriptor-set reuse race; speculative verify
stays per-token by default (known intermittent flake, plans/38 A2).

## Kernel sources (rawhip)

HIP C++ lives in `rawhip/kernels/src_*.hip` (family assets: common, quant,
gemv, gemm, probe, ew, gdn, qsa, vit, ms) assembled by `include_str!`
concatenation — byte-identical to the former single-string SRC. hipRTC
compiles the concatenation once at `RawCtx::new`; `NAMES` fetches every
kernel handle up front.

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

Vulkan backend on the same machine (2026-09): pp512 322 t/s, tg32 11.6 t/s —
0.91× / 0.955× of llama.cpp Vulkan (353 / 12.1). The gap analysis and the full
falsification log (14 hypotheses, all measured) is in benchmarks.md.

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

Attention kernels follow: qk_norm_rope (bit-exact vs CPU f64 mirror — head-RMS over
the first hd elements, k-only kq_scale, rope pairs (p, p+n_rot/2)), kv_append, and
qsa_flash (gated, online softmax, 6e-8) — all verified in `gdn-check`. The full
VkDecoder kernel set is now verified; host assembly remains.


## VkDecoder (2026-09-05, second session)

GPU-resident Vulkan decode (LLM170_VK_DECODER=1). Kernel set fully unit-verified via
`gdn-check` (10 kernels ★): split3, conv (incl. 4-step ring evolution), beta_g, AR,
norm_gated (elementwise-z, WorkGroupID row indexing), silu_mul (f32 exp), l2, qk_rope
(bit-exact), flash. Real-model probe: GDN layer-0 output matches HIP to 7 digits at pos 0.

Three real bugs fixed this session: (1) norm_gated used GlobalInvocationID×32 row
indexing — must be WorkGroupID (the classic GLSL block-index rule); (2) conv kernel
never shifted its ring state; (3) refactor of shared const-building in server/main.rs
accidentally changed the causal mask from 0/1-permit to -inf (broke HIP parity; restored,
HIP re-verified [99,128,114,128,116] on the 7-token gate).

**Open**: (a) ERROR_DEVICE_LOST on full-model dispatch with the enlarged descriptor
pool (works with the original 256-set pool only for single ops); (b) residual-stream
divergence vs HIP beyond layer 0-3 (seed appears to be W4A8 GEMV ordering vs HIP's
.co GEMM binaries — needs either an exact gemv3↔dot_row_w4a8 lane-order match or
dumping the .co GEMM ISA). LLM170_NOMTP=1 debug gate added. The earlier "41/41
bit-identical" claim from the prior commit was measured on the VkAcc fallback path
(missing-tensor injection failure) — retracted.

## VkDecoder session 5 (2026-09-04): full feature surface + accuracy contract

The p2+ prefill divergence is solved: the trait-default `raw_prefill` fed
512-float embedding chunks (not n_embd rows) to `raw_step` at a fixed pos;
VkDecoder now overrides it with per-row stepping. A latent head overflow
(gemv n_out=248320 into a 40960-float buffer — truncated logits hid any
argmax >= 40960, incl. EOS) is fixed with a dedicated logits buffer.

VkDecoder now implements the full raw-decode surface (all reusing the
t=1 verified kernels): MTP draft layer with its own KV (eh_proj, attention,
FFN, shared head), `raw_step_h`/`raw_prefill_h` hidden export,
`raw_verify` + GDN snapshot/restore for speculative rollback,
`raw_step_multi` (parallel sequences) and `verify_batch_ms` (merged
np x spec verify). Gates on the 27B work model: 41/41-token greedy stream
== TRUE CPU, spec k=4 bit-contract MATCH, np2 == single, np2 x spec4 ==
single; HIP paths unchanged.

Cross-backend numerics follow the llama.cpp standard: within-backend bit
contracts (spec == greedy, np == single) are strict; across backends the
reference is stream/caption agreement (llama.cpp `test-backend-ops` and
vLLM use tolerance gates; their CPU and GPU quantization formulas
intentionally differ — e.g. `d_inv = 127/amax` on CUDA). A measured
RADV fdiv reciprocal-lowering (1 ULP on ~5% of q8 block scales) is
accepted under this policy; `vk-gemv-check` gained bit-level histograms
to keep such deviations observable.

## Vision prefill fix (2026-09-04)

The CPU chunked prefill path ignored pre-computed embedding rows (vision
splices) and re-embedded the marker token — captions were image-blind on
CPU. `forward` is split into `forward_emb(rows)` so spliced rows reach the
forward pass; the reference caption is restored exactly.

## Server usability (2026-09-04)

- Tokenizer: GPT-2 `bytes_to_unicode` inverse table (llama.cpp scheme)
  replaces the latin-1 mis-mapping — Korean text now matches merged vocab
  tokens (10 byte-fallback -> 6 tokens) and decodes back to valid UTF-8
  through an incremental byte-buffer detokenizer (no more mojibake).
- `/v1/chat/completions` and `/v1/messages` apply the qwen chat template
  and stop at `<|im_end|>`; non-stream responses include a `text` field.
- Slot recycling now zeroes raw-decoder GDN/conv state (`RawDecode::raw_reset`)
  — consecutive requests previously diverged via residual state leakage.

## Configuration surface

Zero-config by default: tensor types are dispatched automatically from the
GGUF (kernel selection is type-driven, as in llama.cpp) and the standard
models run with no environment variables (verified end-to-end). The
2026-09-08 prune (ADR-0019) removed the concluded-experiment gates
(139 → 122 distinct `LLM170_*` names by grep, comments included); the
remainder are operational
switches (GPU_RUNTIME, SLOTS, W4A8, chunk sizes), verification references
(EXACT), active-plan opt-ins (VK_I8ON for plans/23, VK_TILE/VKD_BATCH for
the f16-prefill acceptance decision), and diagnostics (traces, ktime,
layer bisectors).
