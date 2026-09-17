# Chunked-prefill invariance

A prefill that processes a prompt in chunks must produce the same result as
processing it in one pass. Everything before the last chunk is causally
closed — an early row cannot depend on later tokens — so any difference
between chunk sizes is a defect, not a tolerance.

This note records how we verify that property in this codebase, what the
verification cleared, and what remains open. It exists because a
user-visible failure (Flash-Next output degrading into repetition at
`LLM170_Q4_CHUNK=64`, independent of prompt content) traced back to a
violation of this contract rather than to a broken kernel.

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
   `sum/v0/mid0/last0` samples of the resident frame buffers
   (`frame_ck`). Because `last0` is always the last row of the current call,
   two chunkings can be compared at a common token boundary.
3. **Kernel micro-probes.** `rawhip::probes::gdn_ar_invariance` and
   `gdn_conv_invariance` feed a synthetic sequence to the GDN AR / conv ops
   as one `t`-row call and as `T/per` row-band calls, then compare both the
   carried state and the output. No model, no weights, seconds to run. These
   are the decisive tests: they isolate a kernel from the frame's
   orchestration.

## What the probes cleared (2026-09-17)

- GDN AR: state and output bit-identical across `t`/`per` combinations up to
  `t=208` — 0 differing elements.
- GDN conv (`gdn_conv_t2` + `gdn_conv_state` two-launch form): ring and
  output bit-identical across the same combinations.
- Dense projection `frame_mm_group` with real weights (qkv/z/beta/alpha of
  layer 0): bit-identical for `1×64` vs `4×16`.

So the row-count dependence observed in the frame path is **not** in those
kernels' arithmetic.

## Open leads (as of 2026-09-17)

- The frame path shows cross-call sensitivity: adding unrelated verification
  blocks to a harness changes later runs' results. That points at cached,
  geometry-keyed machinery in the frame rather than at arithmetic, e.g. the
  lazily created row-view tables (`ensure_np_views`, `ensure_pre_views`) and
  the size-keyed scratch pool in `rawctx`.
- `moe_frame_np` restores the ambient row count to the *sequence* count, not
  to the token count. It is currently masked by the caller restoring `t`
  right after, but it is the kind of implicit state this document warns
  about; prefer passing the row count explicitly.

## Reproduction

```sh
# same prompt, two chunk sizes, Flash-Next — 64 collapses into repetition
LLM170_Q4_CHUNK=64  llm170 infer --model <fn.gguf> --prompt-tokens <ids> --n-predict 16 ...
LLM170_Q4_CHUNK=512 llm170 infer --model <fn.gguf> --prompt-tokens <ids> --n-predict 16 ...

# kernel-level invariance (seconds, no model)
cargo test --release -p llm170-backend-gpu --lib -- --nocapture gdn_ar_t_invariance
```
