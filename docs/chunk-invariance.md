# Chunked-prefill invariance

A prefill that processes a prompt in chunks must produce the same result as
processing it in one pass. Everything before the last chunk is causally
closed — an early row cannot depend on later tokens — so any difference
between chunk sizes is a defect, not a tolerance.

This note records the contract, how we verify it, and the defect that was
found and fixed on 2026-09-18 (plans/80 §A).

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
- **A host-side bridge that keeps sequence history must serve boundary
  lookbacks from the call-start snapshot, never from state it has already
  mutated within the call.** This is the bug class that produced the defect
  below; llama.cpp avoids the surface structurally by computing the n-gram
  in-graph for the whole batch.

## The defect (2026-09-18): PLE n-gram lookback read live history

**Symptom**: `LLM170_Q4_CHUNK=63/64` Flash-Next prefill collapsed into
repetition on real prompts; other chunk sizes produced mutually different
streams; the value path (pure CPU, no GPU at all) was chunk-dependent too,
which ruled out every backend/kernel theory.

**Root cause** (`stages/ple.rs::ple_hash`): the boundary lookback for the
trigram of the first tokens of a chunk indexed the *live* history vector,
into which the current call had already pushed its own tokens. For chunked
calls the trigram of chunk token 1 hashed the just-pushed chunk token 0
instead of the true predecessor from the previous chunk. A single unchunked
call never takes that branch, so single-call results were correct and every
chunking silently disagreed with them — at chunk boundaries only, amplified
chaotically through 48 layers.

**Fix**: read boundary lookbacks from the call-start snapshot `hist0`
(kept immutable), while the live vector continues to accumulate the state
to carry out. One indexing change plus one `clone()`.

**Verification ladder used to find it** (kept as permanent assets):

1. Kernel-level row-count invariance probes (`llm170 moe-row-check`,
   `llm170 mm-row-check` with real weights) — proved all GEMM families
   bit-invariant across t; cleared the entire GPU layer early.
2. Same-config runs are bit-deterministic; `HIP_LAUNCH_BLOCKING` and
   per-stage drains change nothing — ruled out races.
3. The pure-CPU value path (`--backend cpu`) reproduced the divergence —
   moved the defect out of every GPU/backend surface.
4. Stage-skip bisect on the CPU path (`LLM170_STAGE_SKIP=ple` → chunk16 ≡
   chunk512) — isolated PLE.
5. Per-sub-stage hashes + row dumps of `ple_hash` outputs pinned the wrong
   trigram at chunk-boundary tokens.

**After the fix**: pure CPU chunk16 ≡ chunk512 exactly; GPU chunk63 ≡
chunk64 exactly; chunk 61/62/128/2048 reproduce the gate baseline stream;
no chunk size collapses. A residual, much smaller axis remains: different
chunk sizes can flip near-tie tokens (~1 token in 10 on adversarial
prompts) because batched GPU GEMM dispatch differs by row count — the
project's near-tie adjudication standard applies there, not the
bit-identity standard.

## Verification: three layers

1. **Token stream, CLI level.** Run the same prompt at several chunk sizes
   and compare generated tokens. Cheap, but only sensitive when the model is
   in a contractive regime — use a real prompt.
2. **Frame checkpoints.** `LLM170_NP_CHECKSUM=1` prints per layer/stage
   samples (`frame_ck`). `LLM170_NP_ROWS=<tags>` adds per-row bit samples;
   `LLM170_NP_ROW0FULL=1` dumps the first rows of tagged buffers in full
   (hex bits). `LLM170_NP_BUFHASH=1` FNV-hashes the valid region of every
   frame buffer at each layer boundary.
3. **Kernel micro-probes.** `rawhip::probes::gdn_ar_invariance`,
   `gdn_conv_invariance` (synthetic, no model); `llm170 moe-row-check` /
   `mm-row-check` (real weights, two row counts, shared rows bit-compared).

## Reproduction

```sh
# collapse (before fix) / clean stream (after fix): 208-token Korean prompt
LLM170_Q4_CHUNK=64  llm170 infer --model <fn.gguf> --prompt-tokens <ids> --n-predict 16 ...
LLM170_Q4_CHUNK=512 llm170 infer ...   # reference

# pure-CPU chunk invariance (fastest end-to-end check, no GPU):
LLM170_Q4_CHUNK=16  llm170 infer --backend cpu ...
LLM170_Q4_CHUNK=512 llm170 infer --backend cpu ...   # streams must match

# kernel-level invariance (seconds):
cargo test --release -p llm170-backend-gpu --lib -- --nocapture gdn_ar_t_invariance
llm170 moe-row-check <model> blk.0.ffn_gate_exps.weight 16 64
llm170 mm-row-check   <model> blk.0.attn_qkv.weight 16 64
```
