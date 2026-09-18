# Chunked-prefill invariance

A prefill that processes a prompt in chunks must produce the same result as
processing it in one pass. Everything before the last chunk is causally
closed — an early row cannot depend on later tokens — so any difference
between chunk sizes is a defect, not a tolerance.

This note records how we verify that property in this codebase, what the
verification cleared, what remains open, and the current state of the
investigation (2026-09-18, plans/80).

## Rule

- Never infer the row count of a frame op from a buffer's capacity; pass it
  explicitly. Frame buffers are allocated for `t_max` rows and a call may use
  fewer.
- Any op that carries sequence state (GDN conv ring, GDN AR, attention KV,
  PLE n-gram history and conv ring, QSA rope/selection) must be written so
  that chaining N calls of `t` rows equals one call of `N*t` rows.
- Kernel selection keyed on the row count is a correctness boundary. If two
  families must exist (tile vs GEMV, WMMA vs DP4A), their *results* must be
  compared, not just their speed — and the comparison needs a tokenizer-level
  prompt, not synthetic ids: random-looking prompts sit in a chaotic regime
  where any arithmetic difference amplifies and every chunk size looks
  "different".

## Verification: three layers

1. **Token stream, CLI level.** Run the same prompt at several chunk sizes
   and compare generated tokens. Cheap, but only sensitive when the model is
   in a contractive regime — use a real prompt.
2. **Frame checkpoints.** `LLM170_NP_CHECKSUM=1` prints, per layer and stage,
   `sum/v0/mid0/last0` samples of the resident frame buffers (`frame_ck`).
   `LLM170_NP_ROWS=<tags>` additionally dumps per-row bit samples, and
   `LLM170_NP_ROW0FULL=1` dumps the first 8 rows of each tagged buffer in
   full (hex bits). Because `last0` is always the last row of the current
   call, two chunkings can be compared at a common token boundary.
3. **Kernel micro-probes.** `rawhip::probes::gdn_ar_invariance` /
   `gdn_conv_invariance` (synthetic sequence, no model), `llm170 moe-row-check`
   and `llm170 mm-row-check` (real weights; run the same input rows through
   two row counts and bit-compare the shared rows).

## What the probes cleared (2026-09-17/18)

- GDN AR / GDN conv kernels: state and output bit-identical across `t`/`per`
  combinations (sequential per-row recursion; association order fixed).
- Dense projections `frame_mm`/`frame_mm_group` (Q8_0, F32 tested): every
  element of the shared rows bit-identical between t=16 and t=64 launches
  (`mm-row-check`).
- Expert stack GEMMs `frame_moe_gemm` (Q4K gate/up, Q5_1 down): bit-identical
  shared rows at rows 160 vs 640, including collision-heavy ids
  (`moe-row-check`).
- MoeTop10 / gather / scatter / quant_q8: row-local, deterministic kernels.

## Current state of the defect (2026-09-18)

Two distinct phenomena remain; the catastrophic collapse is **not** a race.

### 1. Catastrophic collapse — deterministic, multi-chunk only

`LLM170_Q4_CHUNK=63/64` on a 208-token prompt collapses into repetition.
Facts established on a verified-neutral build:

- A **single** t=64 call is clean: on a 64-token prompt, chunk=64 and
  chunk=512 (both one call, t=64) produce bit-identical checksums and the
  normal stream. The chunk *value* does not leak into the frame path by
  itself.
- The collapse requires the **multi-call structure** (64,64,64,16).
- `LLM170_FRAME_SYNC=1` (a device drain after every frame stage) does **not**
  fix the collapse — it is not an async-ordering race.
- The GDN/AR state carry between calls was re-verified bit-identical by the
  kernel probes.

### 2. Micro-divergence at equal inputs — timing/geometry sensitive

On a 64-token prompt, comparing a t=16 first call against a t=64 single call
(same first 16 tokens): every input of layer 2's MoE is **bit-identical**
(mixf/mxsel full rows, ids, weights), yet the gate-GEMM output row 0 differs
in all 640 elements (~1e-3 relative) — while the same GEMM in isolation is
provably row-count invariant. Whether the divergence appears depends on

- whether checkpoint reads (`LLM170_NP_CHECKSUM`) are enabled, and
- whether `HIP_LAUNCH_BLOCKING=1` is set (both together: fully identical),

which is characteristic of a memory-level defect (out-of-bounds read/write
into a neighboring frame buffer) whose visibility depends on buffer geometry
(`t_max` = chunk when `LLM170_Q4_CHUNK` is set) and pipeline timing, not of
kernel arithmetic.

Next step: matched-call state comparison — run the 208-token prompt at
chunk=64 and compare the device state after call k against the state after
the equivalent single call on the same token prefix; the first call whose
end state diverges brackets the defect.

## Reproduction

```sh
# collapse (multi-chunk): 208-token Korean prompt, chunk 63/64
LLM170_Q4_CHUNK=64  llm170 infer --model <fn.gguf> --prompt-tokens <ids> --n-predict 16 ...
LLM170_Q4_CHUNK=512 llm170 infer ...   # normal stream

# single-call t=64 is clean: 64-token prompt, chunk 64 vs 512 → identical
# kernel-level invariance (seconds):
cargo test --release -p llm170-backend-gpu --lib -- --nocapture gdn_ar_t_invariance
llm170 moe-row-check <model> blk.0.ffn_gate_exps.weight 16 64
llm170 mm-row-check   <model> blk.0.attn_qkv.weight 16 64
```
