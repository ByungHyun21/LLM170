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
2. **Frame checkpoints.** `LLM170_DUMP` keys (single frontend):
   `checksum` prints per layer/stage samples (`frame_ck`), `rows:<tags>`
   (tags separated by `;`) adds per-row bit samples, `row0full` dumps the
   first rows of tagged buffers in full (hex bits), `bufhash` FNV-hashes
   the valid region of every frame buffer at each layer boundary, `moe`
   hashes MoE grouped-GEMM inputs. Example:
   `LLM170_DUMP="checksum,rows:L1.ple_hash;L2.moe_sc,bufhash"`.
3. **Kernel micro-probes.** `rawhip::probes::gdn_ar_invariance`,
   `gdn_conv_invariance` (synthetic, no model); `llm170 moe-row-check` /
   `mm-row-check` (real weights, two row counts, shared rows bit-compared).

## Automated checker (2026-09-18, plans/83 C)

```sh
# single command, both architectures, prompt = token ids or text (BPE):
llm170 diag chunk-check <model.gguf> "some real prompt text" 16 63 64 512
llm170 diag chunk-check <model.gguf> 386,18,15,15 16 64 --backend cpu
```

Verdicts per size: `bits-identical` (exact), `near-tie` (argmax preserved,
max|Δ| < 1e-3 — the row-count residual axis), `FAIL` otherwise. Reference
is a single un-chunked prefill.

### Defects found by the checker — status (2026-09-20, plans/84 A)

1. **qwen35 GPU state leak across resets — FIXED.** Two independent
   defects:
   - `Engine::reset_states` replaced only the CPU `SeqState`s; the
     GPU-resident GDN S-state / conv ring live in the raw decoder and
     were left dirty, so the next prefill on that slot started from the
     previous conversation's state (second identical prefill diverged
     max|Δ| ≈ 14). `reset_states` now zeroes the raw decoder state for
     every slot (mirroring `reset_seq`) and invalidates `frame_clean`.
   - **Chunked prefill was not bit-invariant under any split.** After the
     leak fix the checker exposed that *every* multi-call prefill on the
     GPU diverged (max|Δ| 0.25–1.2, argmax flips) while the CPU path was
     invariant. Root cause: kernel-family selection keyed on the row
     count `t` (g4 for t=2–4, tile `_mm` < 32, MMQ/tile `_wm` ≥ 32,
     j128 tile > 64, q8_0 GEMV ≤ 64, flash single-pass np ≤ 128 vs split
     wk8i, serial vs side-stream gate, mt-variant GEMV for t=2–8, and
     t=1 prefill calls routed through the decode path). The families are
     individually deterministic and row-invariant, but they disagree with
     *each other* by a few ulps, and the chunked-vs-single difference
     amplifies chaotically through the layers. Fix: `step_batch` (the
     prefill entry) now sets a **prefill family pin** that forces the
     large-t family for every dispatch decision above, for all `t` —
     decode (`raw_step`), np and spec paths keep their own dispatch.
     Single-token prefill calls also route through the batch path.
   Verification: `llm170 diag chunk-check <27B> <208tok> 4 8 16 63 128
   512` and `512 512 512` (repeat) all **bits-identical**; 9-token prompt
   at sizes 1–7 bits-identical; fresh-slot variant (`LLM170_CC_SEQ=1`)
   clean; `gate-27b.sh` stream unchanged. New kernel-level fences:
   `llm170 mmq-row-check <model> <tensor> <t1> <t2>` and
   `llm170 tile-row-check ...` compare shared rows bit-exactly across two
   batch sizes (MMQ q4/q5/q6/iq4_xs, j128 tile, v4 tile all invariant).
2. **Flash-Next GPU chunk16 collapse** — open (qwen4exp frame path).

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
