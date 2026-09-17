# `crates/backend-gpu/src/rawhip/q4acc/qsa.rs` — measurement record

## 2026-09-05 session 3 — flash attention rewrite (+9% pp512)
- Replaced the split flash-attention kernel with a warp-per-query design
  (no __syncthreads or shared-memory round trips inside the key loop,
  32 queries per block, generic head-dim via per-lane dim slicing).
  Kernel verified bit-exact against the previous kernel on synthetic
  inputs (maxabs = 0) and within 1.2e-6 on real activations.
- Small GEMV latency fix: quantized tile path restricted to n_out >= 128
  (48-output projections were running on a single CU).
- Negative results recorded: 4-column AR blocking (-3%), side-stream
  GEMV overlap (-2%), q8 flash multiplexing (neutral).
- pp512 250 -> 272-274 t/s (llama.cpp ROCm same model: 347-366,
  ratio 0.75x). tg8 unchanged 10.3-10.4 (0.95x).
- Session 3 addendum: cross-checked both engines under rocprof in the same
  measurement window. GEMM tile totals are at parity (~600ms vs 663ms for the
  same p128 workload); the AR recurrent kernel is at parity in isolated
  harnesses (577us ours vs 560us for a faithful re-implementation of the
  llama kernel body). The remaining prefill gap concentrates in launch gaps,
  the first-chunk flash path (kept on the legacy kernel for bit-stability),
  and small fused kernels. Final: pp512 265-272 t/s (0.75x), tg8 10.3-10.4
  (0.95x).
- Session 3 final win: recovered the true (double-buffer) sources of the three
  embedded code objects — an earlier -5.5% verdict against a kernel patch was
  actually source drift (the /tmp sources had been left in a slower
  single-buffer experimental state). Re-applied the token-quadrant z-grid to
  the true sources (neutral at gz=1, interleaved A/B), made 512-token prefill
  chunks the default: pp512 272 -> 277.5 (+2%), bit-identical output streams
  verified with both 19- and 600-token gates. Canonical sources preserved in
  plans/i8_arc/co_src.
- Addendum: llama.cpp mul_mat_q kernels (q4_K, q5_K) integrated via offline
  code objects with an f32->block_q8_1_mmq prequant path (+2.1% prefill,
  interleaved A/B). Token-quadrant z-grid prefill chunks of 512 made default
  after recovering canonical kernel sources (an earlier regression verdict was
  a source-drift artifact). Attention rewritten warp-per-query. Cumulative
  session: pp512 250 -> ~287 (llama.cpp ratio 0.69x -> 0.81x), tg8 0.95x
  (DRAM-bound).
- Final session addendum: the empty-stub root cause (-DRDNA3 vs -DRDNA3_5
  config table selection for gfx1151) unlocked iq4_xs MMQ as well — total
  MMQ coverage q4_K/q5_K/iq4_xs. Session close: pp512 250 -> 294-296
  (+18%, llama.cpp ratio 0.83x), tg8 10.4 (0.95x, DRAM-bound). q6_K remains
  the one excluded type (J-independent corruption; Q6_K-specific SRAM layout
  suspected).
- q6_K closure: llama.cpp's own dispatcher caps MMQ for q6_K at batch<=256
  on RDNA3.5 (prefill uses dequant + hipBLAS MFMA instead), so our exclusion
  of MMQ-q6 costs nothing relative to their path. The actual q6 lever is a
  dequant->fp16 MFMA GEMM path (~3-4% potential).
- Closing win: MMQ extended to the side-stream GEMMs (gate/up/gz) with a
  dedicated y buffer per stream (the shared buffer was a cross-stream race).
  Interleaved A/B: 300-303 vs 274-281 on/off. Session final: pp512
  250 -> ~301 (+20%, llama.cpp ratio 0.85x), tg8 0.95x.
- Opt-in profile (LLM170_F32SILU): __expf replaces the f64-accurate exp
  in 4 elementwise kernels (silu_mul, norm_gated_silu, gdn_conv_t2,
  gdn_beta_g) — pp512 318-320 (+5% over default) at 1e-7-level numeric
  drift (flips argmax on one sensitive prompt). Default keeps the exact
  CPU-mirror bit contract.


## Long-context decode: GQA-sharing attention kernel (2026-09-12)

Root cause of the long-context tg deficit: `qsa_flash_split4q4` runs one WG per
(query head, segment), so each of the 24 query heads re-reads its KV head's K and V -
with 6 query heads per KV head that is a 6x amplification of the KV traffic (at 3k
context: 24 heads x 3k keys x 2 KB = 147 MB per layer per token). Measured slope before
the fix: 0.31 us per key per layer, versus llama.cpp's 0.053.

New kernel `qsa_flash_gqa`: one WG per (KV head, segment) handling all `n_head/n_kv`
query heads of that KV head from a single K/V load (mask read is shared too). The four-row
structure of split4q4 is mapped onto the head axis 1:1 (same 4-key batching, same
warp-tree + LDS two-stage reduction, same softmax update order), so per-head results are
**bit-identical** - verified: base stream == the pre-fix reference, and spec == non-spec.

Gated by context (`LLM170_GQA_TH`, default 768) because the GQA kernel has 6x fewer WGs
and loses slightly when the KV is small:

| context | old | GQA | ratio to llama |
|---|---|---|---|
| 512 | 10.82 | 10.69 | (old kept) |
| 1024 | 10.49 | 10.66 | - |
| 2048 | 9.91 | 10.38 | - |
| 3072 | 9.54 | **10.38** | llama 10.70 -> 0.97x (was 0.89x) |
| 6337 | - | **9.69** | llama 10.69 -> 0.91x (was ~0.85x) |

`LLM170_NO_GQA=1` restores the old path.


## GQA + 32-key segments as decode-attention defaults (2026-09-12)

The GQA kernel's 6x smaller grid hurt at short context, so the segment size was re-tuned:
smaller segments restore the WG count while keeping the shared K/V traffic. Sweep
(tg16, natural text):

| ctx | old (per-head, sg=128) | GQA sg=128 | GQA sg=64 | GQA sg=32 | GQA sg=16 |
|---|---|---|---|---|---|
| 512 | 10.82 | 10.68 | - | 11.02 | **11.08** |
| 3072 | 9.60 | 10.38 | 10.35 | **10.44** | 10.37 |

sg=32 is the best compromise (11.02 / 10.44), so GQA is now unconditional (no threshold)
with `LLM170_T1SG=32` as the default; `LLM170_NO_GQA=1` restores the old path.

Versus llama at the same contexts: 128 -> 10.99 vs 11.48 (0.957x), 3072 -> 10.45 vs 10.70
(0.977x, was 0.89x), 6337 -> 9.80 vs 10.69 (0.917x, was ~0.85x).

Gates after the change: judge **16/19 PASS** (all 10 spec cases, same 3 long-context
near-ties as before), VL gate **4/5 PASS** (one semantic WARN: our 24-token answer starts
in a `<think>` block), and `llm170 check` full pass (866 tensors, GPU<->CPU GEMM
cross-validation intact - the attention's segment split was never part of that contract).


## Why the decode attention cannot be restructured (2026-09-12)

Attempt: `qsa_flash_gqa_w` - one warp per query head, no LDS and no `__syncthreads` in the
key loop, 5 shuffle steps per key instead of ~30 (the launch probe says the current kernel
spends ~1.5 us per key on those shuffles, 1.46 ms/token across 16 layers).

Result: **rejected**. With it the engine's spec stream no longer equals the non-spec
stream (`LLM170_SPEC_GPU=1` vs plain), i.e. the verify batch and the decode step produce
different numbers for the same row. That equality is a product invariant - the batched
verify path (`qsa_flash_split4q4`/`wk` with tl=k+1) must reproduce the decode path's row
results bit for bit, which is how the 10 spec cases in the judge pass. Any change to the
per-(head,key) reduction order or to the online-softmax update order on one side must be
mirrored on the other, so the decode attention's structure is effectively frozen by the
verify contract until both sides are rewritten together.

That is the structural reason behind the long-context attention plateau reported over the
previous sessions, and it is the last identified blocker for the remaining ~2% of
base-mode tg: the per-key shuffle cost (1.46 ms/token) is only accessible by a paired
rewrite of the decode *and* verify attention kernels.

GQA (adopted earlier today) was compatible precisely because it mapped split4q4's four-row
structure onto the head axis 1:1, keeping every arithmetic operation in place.


## Paired attention rewrite: attempted, reverted (2026-09-12)

Following up on the shuffle-bound diagnosis, a single kernel (`qsa_flash_wh`: one warp per
(row, head) work item, lane-serial dot over hd/32 dims + a 5-step warp tree, no LDS/syncs)
was wired into *both* sides of the contract - the t=1 decode and the t<=8 verify/small-batch
branch of `step_batch` - with a unified 32-key segment so the per-row arithmetic is
identical by construction.

Result: the two paths agree for the first few tokens (760, 6511, 198 in both) and then
diverge, i.e. the *exact* spec contract (which requires bit-identical rows, not ties) is
violated somewhere the first tokens do not exercise - with `LLM170_NO_WH=1` restoring the
session baseline exactly. Reverted rather than debugged blind: localising it needs the
verify-path logits compared row by row (the `LLM170_MS_LOGITS` dump path exists for this),
which is a session of its own.

Both attention restructures have now been tried and rejected on contract grounds:
per-head GQA sharing was accepted because it maps split4q4's four-row structure 1:1 onto
the head axis (every arithmetic operation preserved), whereas any change to the reduction
depth changes row results and must be proven bit-identical against the verify path first.
The decode attention's 1.46 ms/token therefore stays, and remains the largest single
identified item in the base-mode tg gap.


## Verify/decode attention unified + where the last 1.4% actually lives (2026-09-12)

Two results.

**(1) The attention contract is now structural.** The verify (t<=8) was running `qsa_flash_wk`
while the decode runs `qsa_flash_split4q4` - different kernels, hence different reduction
orders, hence near-ties that any attention-numerics change re-rolls on one side only. `wk` is
now gated to t>8 (it is the *prefill* kernel: forcing split4q4 there costs 15% pp), so the
verify uses the decode's kernel and the contract is guaranteed by construction rather than
empirically. Judge: 17/19, unchanged; pp512 347 and tg32 11.33 unchanged, so the alignment is
free. Any future attention change (the row x head rewrite included) is now safe to land.

**(2) The base tg gap is dispatch overhead, not GPU throughput.** A decode token reads the
15.67GB of weights and takes 88ms of wall (11.3 t/s) - ~178GB/s of the APU's ~256GB/s peak,
bandwidth-bound like llama.cpp's 11.48 t/s. The GPU-side mark sum per token is ~34.5ms, so
the rest is not arithmetic. The per-token dispatch count is ~600 at 2-5us of submit+barrier
each, i.e. **~1.2-3ms of the 88ms token** - which brackets the entire 1.4% gap (1.4ms). The
documented next lever is therefore dispatch *count*: fusing the same-input qkv/gate/up GEMVs
into one row-ranged dispatch (~192 fewer dispatches per forward) is numerics-neutral (each
output row keeps its own accumulation order) and is now the highest-value remaining item for
base/mmproj tg.


## Why the decode attention is slow, precisely: a serial latency chain per block (2026-09-12)

`qsa_flash_gqa` takes 107us per call at 512 tokens. Its grid is only (1, n_kv=8, nseg=4) = **32
blocks**, each doing ~2us of arithmetic (128 keys x a load/mul plus 5-level shuffle trees for
gq=3 heads) - i.e. it runs at ~2% of its instruction throughput. The reason is the structure of
the inner loop: per 4-key group there is a strict chain of load (DRAM latency) -> multiply ->
5-level shuffle tree -> shared store -> __syncthreads -> warp-0 combine -> __syncthreads -> exp
-> V load -> FMA, which is ~3us of *latency*, and 32 groups per segment gives the measured ~107us.
There is no independent work to overlap because the grid is tiny.

This also explains why `LLM170_QSA_SEG` is exactly neutral (32/64/128/256 all measure 11.33/11.24):
more segments multiply the blocks (better latency hiding for the flash kernel) but grow the merge
kernel proportionally (the merge is itself a 24-block latency-bound kernel), so the two effects
cancel. The segmentation knob cannot win; the kernels need software pipelining (issue the next
group's K/V loads before the current group's reductions) or a fundamentally different decomposition.

Both routes change either the reduction tree or the online-softmax rescale points, and the latter is
exactly what the spec contract (decode argmax == verify argmax) is sensitive to - which is why this
remains the last item behind the batch/single kernel-arithmetic unification.


## Prefill attention: four structural hypotheses tested and excluded (2026-09-12)

All measured at pp3314 (the length-dependent deficit lives here), each reverted after measurement:

| hypothesis | experiment | result |
|---|---|---|
| instruction throughput (shuffle count) | `qsa_flash_wk8`: 8 lanes/row x 16 dims instead of 32 lanes x 4 dims, 4 rows in parallel - 56 -> 22 ops per lane per key | **neutral** (302.6 vs 303.5 t/s) |
| load latency | key loop unrolled 4x so the next key's K/V loads issue during the current key's butterfly/softmax | **neutral** (303.2 vs 303.5) |
| parallelism | `LLM170_QSA_SEG` 64/128/256/512 segments | neutral (298.7-303.8, larger slightly better) |
| gq-head KV reuse | (not implemented - blocked: the block already owns one head and sharing needs 3x the state) | - |

So the prefill attention is not bound by instructions, load latency, or grid parallelism. It also
cannot be the K/V bandwidth (4.2GB of KV reads for 1.29s = 3.2GB/s against a ~190GB/s wall). Its
33.7 GFLOP in 1.29s is 26 GFLOP/s = ~3% of even the *scalar* MAC roof measured on this device (7.0
TIOPS), which points at something structural in the per-key scalar pipeline (butterfly + online
softmax + mask handling) rather than any single knob. The WMMA route (plans/47) sidesteps the whole
structure - the roof probe measures 23.6-48.4 TFLOPS for rocwmma 16x16x16 on this part - and is the
only remaining lever with the ~1000x headroom the arithmetic implies.


## The pp3314 gap is 100% attention: the matmuls are already at the roof (2026-09-12)

Sanity estimate from the pp3314 kernel budget: the MMQ family (7.70s of 11.06s) processes the
model's 15.67GB of weights across 3314 tokens, i.e. ~9.3e13 MAC = 1.85e14 FLOP, which is
**24.1 TFLOP/s** - 102% of the L1-fed rocwmma roof measured on this part (23.56 TFLOPS) and half
the register-resident roof. In other words the prefill matmuls have no headroom left; the WMMA
conversion already happened where it pays (the MMQ kernels).

So the entire length-dependent deficit sits in `qsa_flash_wk`: 1.29s for 33.7 GFLOP = 26 GFLOP/s
(~0.2% of the L1-fed WMMA roof), and it does not respond to instruction count, load prefetching,
unrolling, lane mapping or segmentation (all measured neutral, above). Closing pp3314 from 0.89x to
~1.00x therefore reduces to replacing that one kernel with a tile-based WMMA flash attention -
plans/47-attention-wmma.md (plan since removed: the kernel shipped as the default) - with the reference and the infrastructure both already in the tree.

Nothing else in the prefill budget is actionable: `gdn_ar_w_swap` 0.66s (the recurrence, t>=512
chunks), `gemm_q8_j128` 0.20s, `silu_mul` 0.19s, `mmq_quant_y` 0.16s - all small and near their own
bounds.


## First valid prefill-attention win: 16 lanes/row (2026-09-12)

`qsa_flash_wk16` (registered in kernels/mod.rs this time): 16 lanes per row with 2 rows in flight
per warp (grid t/16), instead of 32 lanes per row with 4 rows processed sequentially. The
per-(row,key) cost drops from hd MACs + 32*2*5 shuffle lane-ops to hd MACs + 16*2*4, i.e. ~1.5x
fewer lane-ops. Paired 3-rep means at pp3314: **303.7 vs 299.9 t/s (+1.3%, run noise +-1.5%)**;
single runs ranged +1.1% (pp512) to +2.0%. Judge: **17/19, unchanged** - the prefill is
contract-free but the correctness gate still passes. Now the default for hd=256
(`LLM170_NO_WK16=1` restores the old kernel).

One real bug surfaced and was fixed: the first version returned early per *lane* when its row was
out of range, which deadlocks `__shfl_xor_sync` when the tail block has <16 rows (the judge's
~20-token cases caught it, rc=1). Lanes whose row is out of range now keep participating and only
suppress their stores.

The win is real but small: the attention is shuffle-bound in a way that only shows ~1.3-2%, so a
large pp gain still needs the WMMA tile path (plans/47) - but that path must be re-attempted with
the kernel registered, since the earlier "neutral" reading was an artifact.


## The attention bottleneck is shuffle *throughput*, and a 2.3% kernel that fails the gate (2026-09-12)

Timing-only experiment: wrap `qsa_flash_wk16`'s butterfly in a skip (wrong results, valid timing) and
pp3314 jumps **316.3 -> 370.1 t/s (+17%)**. Remaining at 1.10x llama if the butterfly cost nothing,
so the shuffle work - not the FMAs, not the loads - is what the prefill attention spends its time on.

Two attempts to attack it:

| attempt | result |
|---|---|
| interleaving two keys' butterflies in source (bit-identical by construction) | **neutral** (316.7 vs 315.8) - the shuffles are throughput-limited, not latency-limited |
| `qsa_flash_wk8` for hd=256: 8 lanes/row x 32 dims, warp = 4 rows, butterfly 4 -> 3 levels (2.7x fewer shuffles) | **+2.3% paired** (317.4 vs 310.4) **but 16/19 on the judge, with 3 real divergences** (long_np2_seq1, long_np4_seq1, long_np4_seq2; top-3 gaps 2.2-6.0, i.e. not ties) - **reverted** |

The divergence appeared only on the judge's long prompts. A 200-token single-chunk prompt produced
*bit-identical* tokens, so the bug lives in the chunked-prefill path (nseg > 4, n_past well beyond the
chunk). Next step for this line: re-add the `part` dump in the launcher, run a >512-token prompt with
wk8 and with the shipped kernel, and diff (row, head, segment) - the same method that localised the
shared-memory bug in the WMMA attempt. The +2.3% (and the 17% ceiling measured above) makes it worth
resuming.


## Strategic finding: the attention's remaining speedup is blocked by numerics tolerance, not design (2026-09-12)

`qsa_flash_wk8` (8 lanes/row, butterfly 4 -> 3 levels) was +2.3% but 16/19 on the judge, and the
part-diff explains exactly why:

- **layer 0 of the first chunk matches wk16 to 9.2e-5** - i.e. the kernel itself is correct and the
  shallower reduction tree costs only rounding.
- **every later layer diverges, up to 9.5** - the 1e-5-level rounding difference is *amplified*
  through the model's recurrent (GDN) layers until the argmax moves.
- The judge's failure signature confirms amplification rather than a kernel bug: it flipped tokens
  where the reference has 5.99 and 2.20 logit margins, which no rounding-level difference can do by
  itself; it needs the recurrence to carry it there.

Open question, stated honestly rather than papered over:

- a 200-token single-chunk prompt with wk8 produced identical *tokens* to wk16 - but tokens are
  argmaxes and survive rounding-level differences, so this says only that the divergence needs a
  longer trajectory, not that wk8's internals were identical;
- yet on multi-chunk prompts wk8 diverges, and its layer-0 part diff (9.2e-5) is ~100x larger than
  tree reordering alone would explain (relative error of a cancellation-prone sum, so not impossible,
  but larger than expected).

So the cause is either (a) length-dependent amplification through the recurrence, or (b) a residual
bug in wk8 that only manifests with the chunked path (nseg > 4). The discriminator is a harness that
feeds *identical* Q/K/V to both kernels for a late layer and compares the part buffers; if wk8 is
then bit-identical, the divergence is amplification, otherwise it is a bug. Until that is settled,
wk8 stays out.

Either way the ceiling is measured: **17% of the prefill** (316 -> 370 t/s when the butterfly is
skipped), and the shipped `wk16` remains the fastest variant that passes the gate.

Same reasoning applies to the WMMA tile path and to the decode attention: their value is real
(+17% ceiling measured by skipping the butterfly) but they trade numerical identity for it.


## Adopted: wk8 prefill attention + reference-gate reset (2026-09-12, user decision)

Per the user's choice (reset), the judge's long-prompt cases now report divergences as **INFO**
(detailed diagnostics retained: first divergence position, top-k gap, both token prefixes) instead
of failing, for both the plain long cases (`long_prompt`, `long_np2_*`, `long_np4_*`) and the
long-context spec cases (`spec_long`, `spec_long_np4`). Short and medium cases keep the original
strict criterion (exact match, or tie within top-6 and a <1.5 nat gap), so real bugs still gate.

Justification recorded in `scripts/verify.py`: a changed reduction tree differs by <=1e-5, the model's
recurrence amplifies it, and the kernel's own correctness is guaranteed by `llm170 attn-check`
(identical inputs, max|delta| 1.1e-5, 0 outliers in 50.3M accumulators) rather than by trajectory
matching. The judge's own log confirms the class is unstable: `spec_long_np4_seq2` failed in one run
and passed in the next with nothing but the harness change between them.

With that, `qsa_flash_wk8` (8 lanes/row, 3-level butterfly) is the default prefill attention for
hd=256: paired pp3314 317.4 vs 310.4 t/s (+2.3%), `LLM170_NO_WK8=1` restores wk16. Judge: 16 PASS,
3 INFO, 0 FAIL.


## The scalar attention line is closed: shuffles are the floor, MMA is the only way past (2026-09-12)

Two independent checks say the remaining 17% (measured by skipping the butterfly: pp3314 316 -> 370
t/s) cannot be recovered on the scalar path:

- **Bandwidth**: SIMD shuffle throughput is ~128 float/cycle/SM on this part versus 32 float/cycle/SM
  for shared memory - 4x in the shuffle's favour. Moving the dot's reduction to shared memory (the
  obvious alternative) costs 512 KB of shared traffic per 16x16 tile, i.e. ~62 cycles per (row, key)
  against ~17 for the shuffle version. That prediction matches the earlier measurement where shared
  staging came out 6.8% *worse*.
- **Structure**: with 8 lanes/row and a 3-level butterfly the shuffle+add overhead is 48 of 560
  lane-ops per (row,key) on paper, yet removing it measures 17% of the *whole prefill* - the SIMD
  shuffle unit is the actual limiter, not the instruction count. wk8 (adopted, +2.3%) already banks
  the part of that which is reachable by shaving levels.

So the prefill attention's remaining headroom requires the reduction to happen *in hardware*: a
tensor-core (WMMA) tile kernel, which is what llama.cpp uses. That is the open item in
plans/47-attention-wmma.md (plan since removed: the kernel shipped as the default); the ad-hoc attempt got as far as compiles-and-runs with the fragment
layouts verified by `wmma-check`, but still produces NaN on multi-chunk prompts. The measured prize
is pp3314 ~370 t/s = 1.10x llama, which would close the pp cell.


## `wmma-check-ldm`: the attention kernel's actual stride pattern is correct (2026-09-12)

The first WMMA verification only exercised 16x16 tiles with ldm=16, while the attention kernel loads
fragments from 16x256 tiles with **ldm=256**. New diagnostic (`llm170 wmma-check-ldm`,
`wmma_probe_ldm`) tests exactly that: **max|delta| = 0.0000, NaN 0/256** - the fragment loaders are
right at the real stride. So neither the loaders nor the layouts nor the shared-memory size (the
53248 B bug, fixed) explain the multi-chunk NaN; the remaining suspects inside the tile kernel are
the softmax bookkeeping around `mrun`/`srun` per fragment slot and the P hand-off through shared
memory. Everything else about the kernel has now been verified in isolation.


## The attention redesign, specified from llama.cpp's own kernel (2026-09-12)

Our kernel's shape was assumed to be wrong; it is not. llama.cpp's fp16 MMA flash attention
(`fattn-mma-f16.cuh`, the path gfx1151 actually takes via `AMD_WMMA_AVAILABLE`, and the one that
produced the 335.06 t/s reference) is *also* "block owns a query tile, loops over the whole KV" with
stream-K off at pp3314 lengths. The differences that matter are four, all of them implementable:

| | ours (`qsa_flash_wmma`) | llama.cpp RDNA3 hd=256 f16 (config table at `fattn-mma-f16.cuh:128-177`) |
|---|---|---|
| threads | 256 | 256 (8 wave32) |
| **KV rows per iteration** | **16** | **64** (nbatch_fa) |
| **K/V smem** | 16 KB, K and V simultaneously | nbatch_K2 = nbatch_V2 = 128 half2 = the whole head dim in one load, K then V through the *same* buffer |
| **Q smem** | 32 KB, kept live | **Q_in_reg = true**: the Q is consumed into registers once per block and its 33,792 B buffer is then **reused as the K/V tile** |
| dynamic smem total | 61,440 B | **38,400 B** = max(Q, KV+mask, combine), not the sum |
| occupancy target | 1 (implied) | **2** |
| softmax row reduction | 3-stage `__shfl_xor` (8/4/2/1) | **one `__shfl_xor(16)` per column per KV chunk** |

The two changes that unlock occupancy 2 are coupled: `tile_K = tile_Q` only works *because* the Q
lives in registers, and the 64-row KV step only fits *because* that 32 KB was freed. Together they
take the budget from 61 KB to ~38 KB, which is what the occupancy-2 target needs.

The one-shuffle reduction is a *layout* difference, not an algorithmic one: llama's mirrored RDNA3
mma layout (`mma_tile_sizes`, `fattn-mma-f16.cuh:1085-1130`) puts a softmax row across 2 lane groups
rather than 16 lanes, so one xor suffices where we need three. Closing that needs the same mma
instruction/layout we currently get from rocWMMA - a deeper change than the buffer reshuffle, and
worth deferring until the occupancy work has been measured.

Also confirmed from the scout: the 335.06 t/s reference is the ROCm/HIP build (build 8b4b3558f), i.e.
*this* kernel - not the Vulkan shader. So the pp3314 target is a specific, reachable fp16-mma
implementation, and the redesign above is the difference between it and ours.


## WMMA attention becomes the default: pp512 ahead of llama (2026-09-12)

Applying llama.cpp's two structural tricks (Q_in_reg + its 33,792 B buffer reused for the K/V tile)
took the kernel from parity-past to ahead of the scalar at *both* lengths:

| config | scalar (wk8) | WMMA, Q_in_reg + smem reuse | llama-bench reference |
|---|---|---|---|
| pp512 | 360.2 | **361.3 (+0.3%)** | 354.66 -> **1.019x** |
| pp3314 | 324.7 | **329.2 (+1.4%)** | 335.06 -> **0.983x** |

The shared budget went 61,440 -> 32,768 B (K at 0, V at 8192, score exchange at 16384, P at 24576 -
26,624 B used, so the Q's buffer holds everything and the occupancy should now be two blocks per CU).
A first attempt serialized the K-then-V staging through one buffer and *regressed* pp512 to 343.6
(+1 sync per key-tile and lower memory-level parallelism); restoring the parallel staging while
keeping Q_in_reg is what produced the numbers above. That A/B is the useful datum: the win comes from
freeing the Q's buffer, not from touching the staging.

Verification: `wmma-attn-check` reports 0 mismatches (max|delta| 6e-4, f16 accumulation), the
engine-level `attn-check` reports max|delta| 1.1e-5 with 0 of 50.3 M elements above 1e-4, and a
600-token multi-chunk prompt produces token-identical output to the scalar path. The default is the
WMMA path for hd=256 prefill (`LLM170_NO_WK_WMMA=1` restores wk8) - the same user decision that
accepted the non-bit-exact wk8 applies, and here the arithmetic difference is bounded by 1.1e-5.

What is left on this axis: llama's 64-row KV step (ours is 16, so 4x more syncs and staging rounds),
which needs the shared budget above 32 KB and so requires re-checking the occupancy-2 target, and the
single-`__shfl_xor(16)` softmax reduction, which needs llama's mirrored RDNA3 mma layout rather than
the generic rocWMMA one.

Median-based confirmation of the WMMA default (same day): pp512 four runs 356.9/360.0/359.3/359.9
(median 359.6) against wk8's 361.5/360.5, i.e. parity at 512 and not the 361.3-vs-360.2 single-run
claim above; pp3314 two runs 329.2/333.0 (median 331.1) against wk8's 324.7/324.1, i.e. **+2.0%**.
One pp512 run measured 341.5 in between - a single-run outlier, so any future A/B here should take a
median of three or four, not one. Opposite the llama-bench reference the medians give **pp512 1.014x
and pp3314 0.988x** (the latter was 0.89x before this change).


## Decode attention v2: reduction restructured, tg3314 +2% (2026-09-12)

The fix scoped in the entry above is implemented and shipped as `qsa_flash_gqa2` (default;
`LLM170_NO_GQA2=1` restores the old kernel). The change is the decomposition: each warp now owns
four keys and finishes their dot products *inside the warp* (lane = 8 dims, 5-stage butterfly),
where the old kernel split hd across all 256 threads and therefore needed a 5-stage shuffle *plus* a
cross-warp shared round-trip *plus* two block syncs for every four keys. The softmax tile work is one
lane per head, so nothing in the shared score array is written by a lane while another reads it.

`llm170 gqa-bench` (new probe: runs both kernels on identical inputs and times them):

| n_past | v1 | v2 | ratio | max rel. diff | mismatches |
|---|---|---|---|---|---|
| 512 | 79.8 us | 33.9 us | 2.36x | 5.1e-7 | 0 |
| 1024 | 99.9 | 68.1 | 1.47x | 5.1e-7 | 0 |
| 2048 | 191.0 | 114.3 | 1.67x | 5.1e-7 | 0 |
| 3314 | 292.9 | **171.5** | **1.71x** | 5.1e-7 | 0 |

The first version of the kernel was wrong (max rel. diff 1.55) because the warp-0 softmax had two
races - it summed the score array while other lanes were still overwriting it with exponentials, and
`pmx[r]` crossed lanes without a barrier. One lane per head fixed both.

Engine effect (same-run A/B, `--tg 32`, greedy): tg512 **11.58 vs 11.48** (+0.9%), tg3314 **11.02 vs
10.80** (+2.0%). Against llama-bench that is tg512 **1.002x** and tg3314 **0.955x** (from 0.93x). The
kernel now moves its 27 MB of unique K/V per layer at ~157 GB/s, about 83% of what the weight-stream
sustains - so it is close to bandwidth-bound and the next gain here needs *less traffic* (an f16 KV
cache would halve it), not more restructuring.

Verification: `gqa-bench` 0 mismatches at 5.1e-7; a 300-token prompt generates token-identical output
to the old kernel (`LLM170_NO_GQA2=1`); spec==nonspec is identical at both 100 and 1250 prompt tokens.
The old kernel was written to be bit-identical to `split4q4`, which the short-context spec contract
rested on; v2's reduction order cannot reproduce that bit-for-bit, so the contract is now
verified-by-test rather than by construction - the tests above are the ones to re-run if this kernel
changes again.


## f16 KV measured for the decode attention: real but too small to ship (2026-09-12)

The standing hypothesis after the decode rewrite was that an f16 KV cache would halve the attention's
traffic and buy back the remaining tg3314 gap. Measured directly instead of assumed: `kv_f16` (a
vectorised f32->f16 pass) plus `qsa_flash_gqa2h` (gqa2 reading the f16 mirror; the K load drops from
two float4 to one 16-byte load) are both in-tree and wired into `gqa-bench` as a third variant.

| n_past | v2 (f32 KV) | v2h (f16 KV) | gain | max rel. diff | over 1e-3 |
|---|---|---|---|---|---|
| 512 | 33.9 us | 31.2 us | 1.09x | 4.8e-4 | 0 |
| 1024 | 66.9 | 61.4 | 1.09x | 4.8e-4 | 0 |
| 2048 | 116.2 | 109.3 | 1.06x | 4.8e-4 | 0 |
| 3314 | 178.4 | **157.0** | **1.14x** | 4.8e-4 | 0 |

So the decode attention gains 6-14%, i.e. ~21 us per launch at 3314, and a token runs 16 launches
(8 layers x attention+merge): **~0.34 ms/token, or +0.4% tg**. The numeric cost of f16 KV is 4.8e-4
relative (the same class as the f16 prefill the user already accepted). Not shipped: a change that
touches the KV writers, every attention kernel and the MTP KV, for +0.4% on one cell, is not worth
its verification surface - the earlier expectation that the bytes were the limiter was wrong, because
the kernel is latency-bound, not bandwidth-bound (0.7% of FP32 peak, 59 GB/s in situ).

The measurement is kept as a probe (`gqa-bench` variant v2h) so the decision can be re-taken if the
economics change - e.g. if KV capacity rather than speed becomes the constraint (RAM/SSD offloading),
where halving the KV footprint is worth more than +0.4%.

Also recorded here because it cost time: the first two runs of this probe reported "inf" differences
because the conversion block had been inserted *before* the h2d uploads, so it converted zeros. A
probe that silently reads uninitialised device memory is indistinguishable from a broken kernel -
the launch error was absent (it was a legal launch over zeros).


## The decode attention's real design, extracted from llama.cpp (2026-09-12)

Scout extraction from the vendored source. On gfx1151 with f16 KV, hd=256 and GQA 24/4, llama.cpp's
decode does **not** run `fattn-vec` and does **not** run the WMMA path: `ggml_cuda_get_best_fattn_kernel`
(`fattn.cu:673-687`) excludes WMMA because `Q->ne[1] * gqa_ratio_eff = 1*2 = 2` is not > 16, and
`gqa_opt_applies` excludes the vec kernel for non-quantized KV. It runs **`flash_attn_tile<256,256,1,2>`**
(RDNA config row `fattn-tile.cuh:293`):

| | llama `flash_attn_tile` | ours (`qsa_flash_gqa2`) |
|---|---|---|
| threads | **64 (2 warps)** | 256 (8 warps) |
| work per lane | **one key: the whole 256-dim dot, serial** | 8 dims of a key, partial |
| QK reduction | **none - 128 `v_dot2_f32_f16` in one thread** | **5 shuffle stages per (key, head)** |
| keys per block | 32, one per lane | 32, four per warp |
| shuffles per 32-key tile | **one 5-stage max butterfly** | 5 stages x 6 heads x 32 keys |
| KV staging | shared, 128-bit copies, nbatch_K=64 halves per pass | direct global float4 x2 per lane |
| probabilities | shared KQ buffer hand-off | shared score tile |
| KV axis | split across ~26 parallel blocks + combine kernel | split into 104 segments + `qsa_flash_merge` |
| occupancy | 8 | ~2 |

The headline is the QK: llama computes each key's dot product **inside a single thread** with the RDNA
dot-product instruction `v_dot2_f32_f16` (half2 x half2 -> f32 accumulate, `common.cuh:763-770`), so
the cross-lane reduction disappears entirely. Ours pays 5 shuffle stages for *every* (key, head) pair -
that is the latency chain the ktrace attributed 3.65 ms/token to, and it is also why the kernel sits at
0.7% of FP32 peak.

This also re-frames the f16-KV measurement above: `v_dot2_f32_f16` *requires* half inputs, so f16 KV is
the enabling condition for the fast path rather than a bandwidth optimization. The probe measured only
1.06-1.14x because it kept the scalar f32-FMA dot; with a lane-per-key + v_dot2 kernel the shuffle work
(a few hundred instructions per tile) collapses to 128 dot instructions per lane.

Implementation spec for the next session: f16 KV (writers or mirror), a decode kernel with lane=key and
`v_dot2_f32_f16` over the 256-dim head (K in f16, Q converted once per block), the softmax as a per-warp
max/sum butterfly over the tile, the P handed to the thread-per-dim PV stage through shared, and the
existing `part`+merge split (llama's combine is the same scheme). Expected to remove most of the 3.65
ms/token the attention costs at 3314.


## v_dot2 decode attention shipped: tg3314 11.02 -> 11.15 (2026-09-12)

The decode kernel now follows llama's tile design: `qsa_flash_gqa2d` puts **one key per lane** and
computes that key's whole 256-dim dot with 128 `v_dot2_f32_f16` in a single thread - the QK has *no
cross-lane reduction at all* (our previous shape needed 5 shuffle stages per (key, head)). The K tile
is staged in shared with a 258-half row stride (2 halves of padding make the stride an odd number of
words, so the per-lane row reads are bank-conflict-free), and the softmax is a 5-stage max/sum
butterfly per warp with the probabilities handed to the thread-per-dim PV stage through shared. The
running max/sum live in per-warp registers - an earlier draft kept them in shared and the lane-to-lane
race collapsed `e_m` to 1, which `gqa-bench` caught immediately.

`gqa-bench` now measures four variants on identical inputs:

| n_past | v1 (original) | v2 (f32, warp-per-key) | v2h (f32 kernel on f16 KV) | v2d (f16 + v_dot2) | v2d/v2 |
|---|---|---|---|---|---|
| 512 | 79.8 us | 33.9 | 31.5 | **22.5** | 1.51x |
| 1024 | 100.0 | 69.2 | 61.0 | **42.6** | 1.62x |
| 2048 | 190.5 | 113.7 | 108.9 | **66.0** | 1.72x |
| 3314 | 291.0 | 175.9 | 157.4 | **106.9** | 1.65x |

Correctness: max relative difference 5.05e-4 against the f32 kernel (0 elements over 1e-3), i.e. the
f16 input quantization, same class as the accepted f16 prefill.

Engine: an f16 mirror of the KV is maintained **at the write sites** (`kv_to_f16` after each
`kv_append_t` in the prefill and after the decode's single-row copy) rather than by a watermark in the
attention launcher - the watermark variant would silently go stale whenever the spec path truncates
the KV, which is exactly the kind of bug that costs a session. End-to-end with the mirror:

| metric | before | after |
|---|---|---|
| tg512 | 11.58 | 11.60 |
| tg3314 | 11.02 | **11.15** (+1.2%, 0.966x of llama-bench) |
| MTP spec3 (steady) | 22.04 | **22.21** |
| tokens | - | identical on a 300-token prompt; spec==nonspec identical at 21 and 2302 tokens |

`LLM170_NO_GQA2D=1` restores the f32-KV kernel. The f16 mirror also halves the KV footprint for one
sequence, which is the configuration RAM/SSD offloading will care about.


## qsa_flash_merge measured: 0.61ms, not worth chasing (2026-09-12)

The todolist carried this as "0.64ms to merge 2.6MB (50x off bandwidth), +0.45ms of the context
penalty". Two corrections from measuring instead of trusting that framing:

1. **0.64 ms is the total across the 16 launches of a decode step**, i.e. ~40 us per launch, not per
   launch as implied. Against a 93.8 ms step at 3314 the whole merge is **0.65%** - the maximum any
   merge work can return, context penalty included.
2. **The obvious suspect was wrong.** The m_i scan reads one float per segment at a 1032 B stride, so
   every read is a different cache line, and each thread looped all nseg=104 of them serially - it
   looked like pure exposed latency. Parallelising that scan across a warp (lanes stride segments,
   shuffle max) is bit-identical and moved the total only **0.64 -> 0.61 ms** (-5%). So the scan was
   not the limiter; what is left is 24 blocks (one per head, ~12% occupancy) each taking ~40 us, and
   splitting that further would need either dim-split blocks or a tree merge.

Kept anyway because it is bit-identical and free. Recorded so the next session does not spend another
hour on a 0.6 ms item: the decode attention (2.59 ms in situ) and the prefill attention are where the
remaining base gap lives, not here.


## v_dot2 decode attention: 101 -> 61 us at 3314 (2026-09-12)

Two changes on top of the shipped v_dot2 kernel, both bit-identical:

1. **Dropped `volatile` from the `v_dot2_f32_f16` asm** (106.9 -> 101.2 us). `volatile` forbids
   reordering, so the LDS loads feeding each dot could not be hoisted; without it the compiler
   pipelines them. llama's own helper keeps the volatile - here it cost 5%.
2. **Vectored the K/V staging to 16-byte loads** (101.2 -> **61.2 us**). The staging loop was reading
   the f16 KV with 2-byte scalar loads (32 per thread); `uint4` reads make it 4 per thread. That was
   the real bottleneck: the kernel had looked bandwidth-limited at 160 GB/s, but the scalar loads, not
   the bytes, were setting the pace. This is the largest single decode-attention gain of the session.

`gqa-bench` with the both changes:

| n_past | v2 (f32, warp-per-key) | v2d (f16 + v_dot2) | v2d/v2 |
|---|---|---|---|
| 2048 | 112.2 us | **40.5** | 2.77x |
| 3314 | 182.4 | **61.2** | 2.98x |

(against the original v1: 4.6x). Correctness unchanged: 5.05e-4 max relative difference, 0 elements
over 1e-3. Engine: tg512 11.60 -> 11.65, **tg3314 11.16 -> 11.32** (+1.4%, now **0.981x** of
llama-bench, from 0.966x), tokens identical on a 300-token prompt.

Lesson worth keeping: the kernel was misdiagnosed as bandwidth-bound from its GB/s figure alone; the
actual limiter was load width in a loop that looked trivial.


## Regression found and fixed: the f32-KV removal missed three launcher paths (2026-09-12)

The f32-KV removal above shipped with a silent breakage on three paths the initial verification did
not cover, found when the Vulkan agent's acceptance test compared a short prompt across backends:

- **Short prompts** (np <= 128) take the single-kernel `qsa_flash` prefill path, not the split path.
  That kernel was left f32 while the launcher was switched to the f16 mirror -> garbage output
  (`? ? ?` from a 5-token prompt; the 300-token test used the split path and was bit-identical).
- **The MTP draft and the per-sequence (spec/np) paths** pass the KV through `self.kv_k` / a
  per-row pointer table (`ms_kvk_ptr_to`), all of which still pointed at the now-NULL f32 buffers:
  `rawhip: d2h-sync: 700` (illegal address) on np4 and on any `--spec` run. The t=1 fused flash in
  the spec path had the same problem.
- `LLM170_NO_GQA2D=1` crashed for the same reason (the f32 allocation is conditional on the legacy
  envs, and that env was missing from the list).

Fixes: `qsa_flash` converted to f16 like the other prefill kernels; an f16 mirror added for the MTP
KV with conversions at its append sites; the per-seq append loops and the pointer tables switched to
the mirrors (plus the mirror conversion they were missing entirely); `NO_GQA2D` added to
`legacy_f32()`; the debug d2h guarded.

Verification after the fix: 5-token coherent, **300-token bit-identical**, np4 no-spec 68 tokens OK,
**np4 x spec3 31.03 t/s (2.00x llama)** and spec3 single-stream steady **23.5 t/s (2.04x)** - both
above the previous records - plus `attn-check` 0 outliers and `wmma-attn-check` 0 mismatches.

Lesson for the next kernel-input-type change: the launcher passes `*mut c_void`, so a type mismatch
between a kernel and its buffers is **silent**. Every launcher of the changed kernel must be grepped
and every distinct execution path (short prompt, split prefill, spec, np, MTP) token-tested - the
300-token prompt alone exercises only the split path.


## Flash-Next prefill split: a quarter of it is the QSA bridge (2026-09-14)

With LLM170_FRAME_TIME the pp2048 frame reports 8,831 ms for the 2048-token chunk
(223-230 t/s, matching the bench wall of 8,917 ms). The frame's own timer accounts for
76% of that; the remainder, ~2,119 ms, is the QSA bridge - the read/drain/write round
trip around each QSA stage. Twelve QSA layers therefore cost ~175 ms each in transfers
alone, which is a larger and more targeted lever than anything inside the frame stages.
The Q4_TRACE stage prints only fire on the NaN guard path, so the older per-stage sums
(21 ms per layer) do not account for the chunked frame's real cost and should not be used
for the prefill; the frame-total and bridge timers are the trustworthy pair.



## Why the QSA bridge costs 24% of the prefill (2026-09-14)

Direct measurement with LLM170_Q4_TIME corrects the earlier frame-timer reading. For the
2048-token chunk, across 24 QSA calls (12 layers x 2 chunks), the bridge is (note: this
5.0 s total spans two chunks, so the stage is ~2.5 s per 8.9 s chunk, not the 54% the text
below originally claimed - see the corrected per-t table at the end of this file):

| part | total | share |
|---|---|---|
| read (d2h + drain) | 0.02 s | 0% |
| stage (host QSA) | 4.74 s | 99% |
| write (h2d) | 0.05 s | 1% |

So the transfers are irrelevant and the bridge is not really a bridge problem: the host
QSA stage itself is 4.74 s of CPU work, 81 ms per 1024-token layer call and 54% of the
8,831 ms frame total - the largest single component of the Flash-Next prefill, and the
reason the frame timer's 24% figure was wrong (that timer nests inside the frame). The
per-call heap churn (2048 per-row Vecs then out.concat(), ~98k allocations per forward)
is real but secondary.

The fix is to move the QSA value path onto the device rather than trimming the round trip
around it: that removes the 4.74 s of CPU work, the full-synchronisation drain that idles
the device through it, and the allocation churn together. This is the Flash-Next prefill's
main remaining lever, and it is an architectural change rather than a tuning one.



## The host QSA stage's 4.96 s is two CPU kernels that already exist on the device (2026-09-14)

LLM170_Q4_TIME splits the host stage (2048-token chunk, 24 layer calls) as:

| stage | total (2 chunks) | share of the 5.0 s | per chunk |
|---|---|---|---|
| attn (device call + copies + mm_batch) | 1.87 s | 37% | 0.94 s |
| mm_group | 1.92 s | 38% | 0.96 s |
| sel_build (host selection list) | 0.71 s | 14% | 0.36 s |
| sel+proj | 0.52 s | 10% | 0.26 s |
| wlookup, passB | 0.00 s | 0% | - |

Follow-up instrumentation splits the attention lap and corrects part of the picture. The
device path is in fact taken - no fallback message fires - and dev_weight caches device
copies by host pointer, so neither mm_group nor attn is a re-upload. mm_group and attn
are accelerator work already, run at prefill batch shapes through GEMV-class kernels
rather than MMQ-class ones. sel_build is the one clearly host-side item at 0.71 s, but it is not the allocation
churn: rewriting it to reuse a thread-local buffer and fill by push (no zero-initialised
Vec) measured 0.70 s, i.e. neutral, and was reverted. The cost is the materialisation
itself - a token selecting top_k blocks expands to top_k*r positions and the whole list is
written out per call, tens of megabytes at memory-write speed. Removing it means having the
attention kernel walk the block list directly instead of a flattened index array, which is
a kernel-interface change rather than a host cleanup. Routing the prefill's QSA through the device kernels therefore removes
~4.45 s of the 8,831 ms chunk, i.e. about half of the Flash-Next prefill, without writing
new math. The work is structural - the frame accelerator's interface has to expose the QSA
op so the QSA layers can run device-resident - and it is the single largest remaining win
for this model.



## Long-context decode: the QSA stage is 58% of the step (2026-09-14)

pp8192 then 4 decode steps, frame timers plus KTRACE (the bench's tg without a long prompt
never exercises this - n_past is the prompt length, so ctx alone does not reach it):

| item | total | per call |
|---|---|---|
| frame-total t=1 (decode step) | 684.7 ms / 5 | 137 ms/step |
| stage_mm_group | 3,654 ms / 120 | 30 ms |
| stage_sel+proj | 2,147 ms / 120 | 18 ms |
| stage_sel_build | 1,617 ms / 120 | 13 ms |
| stage_attn (host side only) | 144 ms / 60 | 2.4 ms |
| q4_qsa_attn_sel (KTRACE, device) | 17.1 ms/step | 1.425 ms |

Two things stand out. First, the wall at long context is 195.7 ms/step against 83.9 ms of
kernels, so over half the step is outside the kernel list even though the host submit cost
is small elsewhere - that gap is unexplained and is the next thing to attribute. Second,
within the QSA stage the cost is not the attention kernel but mm_group / sel+proj /
sel_build, ~62 ms of the 137 ms step; these are host-visible and sync-bound because
run_prepared ends in a d2h. mm_group at t=1 costing 30 ms is not explained by its transfer
volume (25 KB), so launch/sync count or prepare_x is the suspect and needs measuring.

The stage timers do not include device time for asynchronous launches: stage_attn reports
2.4 ms while the same kernel measures 1.425 ms per call in KTRACE, so frame stage timings
are host-side unless the stage ends in a d2h.


## Corrected QSA stage shares, split by t (2026-09-14, supersedes the 54%/57%/58% figures)

The stage timers mix prefill and decode calls unless t is filtered, and summing them
without that filter produced three wrong shares earlier in this file. With t as a label:

| stage | t=1 per call | x12 layers = per step | t=2048 per call |
|---|---|---|---|
| attn | 2.55 ms | 30.6 ms | 100.97 ms |
| sel_build | 1.39 ms | 16.7 ms | 32.66 ms |
| sel+proj | 0.69 ms | 8.3 ms | 41.89 ms |
| mm_group | 0.40 ms | 4.8 ms | 72.76 ms |
| total | 5.03 ms | 60.4 ms | 248.3 ms |

So the QSA stage is 42% of the long-context decode step (144.2 ms frame, 83.9 ms of
kernels), not 58%, and 2.98 s per 8.9 s prefill chunk (33%), not 54-57% - the 5.0 s
figure was a two-chunk total. In both regimes the largest single stage is attn, and
KTRACE confirms the attention kernel is the device-side maximum at long context
(1.425 ms/call, 17.1 ms/step). The stage prints now carry a t label so this cannot
recur.



## QSA decode attention: position splitting (flash-style) lands -12.7% at long context (2026-09-14)

The t=1 attention kernel was latency-bound: `_sel4` puts 4 heads in a warp (register
limit) and walks the whole selection list serially per warp, so at n_past 8192 a warp
runs 8192 positions x ~35 cycles = ~200 us - and the t=1 grid is only (1, n_head/8) = 3
blocks, i.e. almost no parallelism. Measured 1.425 ms/call = 17.1 ms/step.

New: `q4_qsa_attn_sel4s` splits the selection list across (split, head-group) blocks,
each warp keeping its own online softmax over its chunk, with
`q4_qsa_attn_sel4s_merge` combining the partials in flash order. Splits adapt to the
list length (list/32, capped at 64) so short contexts pay nothing.

| measurement | before | after |
|---|---|---|
| pp8192 tg, per step | 142.5 ms | 124.4 ms (**-12.7%**) |
| pp24 tg, per step | 566-576 ms / 8 | 574.6 ms / 8 (no regression) |

The garbage first attempt is worth recording: the partial buffer was laid out
[head][split][m,l,acc0..7], but acc is per-lane, so 32 lanes overwrote the same ten
floats and the model diverged after the first token. The layout is now
[head][split][lane][m,l,acc0..7].

Correctness: the merge changes the summation order, so this is not bit-identical; the
diverse stream and a 200-token prompt reproduce token-for-token, which is the repo's
contract for such kernels (same as the q5_1 MMQ tile). LLM170_QSA_SPLIT=0 restores the
bit-exact path. The 27B is untouched (no QSA) and its timings are unchanged.



## QSA attention: 6 heads per warp instead of 4 (2026-09-14)

`_sel4` keeps the gate in registers, so q + gate + acc = 96 floats per lane and 4 heads is
the ceiling - which means 24 heads split into 6 warps that each read the same K/V rows,
a 6x re-read. The gate is only needed at the end, so reading it from global there frees
those registers and lets the same 96-float budget hold 6 heads, cutting the re-read to 4x.
That matters because the prefill attention is bandwidth-bound: at t=2048 each call moves
~34 GB against 236 GB/s ~= the measured 101 ms.

`q4_qsa_attn_sel6` (prefill path, chosen when n_head % 12 == 0) and the split kernel (both
head groups now 6) preserve the arithmetic order exactly, so this is bit-identical - the
diverse stream and the 200-token prompt reproduce unchanged, with no numerical contract
relaxed.

| measurement | before | after |
|---|---|---|
| pp2048 | 9,012-9,118 ms | 8,748-8,908 ms (-1.6 to -2.9%) |
| pp8192 tg8 (with the split kernel) | 1,140.3 ms | 1,019.9 ms (-10.6% cumulative) |

LLM170_QSA_H6=0 restores the 4-head kernel.



## Removing a redundant KV clone in the QSA stage: -27.6% on long-context decode (2026-09-14)

frame.rs' QSA bridge copied the KV prefix into fresh Vecs on every layer call:

    let ck = seq.kv_k[full_idx][..kn].to_vec();   // kn = n_past_max * n_kv * hd

At n_past 8192 that is 33.6 MB per layer, 403 MB per decode step, and it showed up as the
sel_build stage timer (1.39 ms/layer = 24 GB/s, exactly the measured memcpy rate). The
accelerator takes &[f32], so borrowing the slice is sufficient - NLL accepts it because the
fallback later reborrows seq immutably.

| measurement | before | after |
|---|---|---|
| pp8192 tg8 | 1,019.9 ms | **738.4 ms (-27.6%)**, 92.3 ms/step |
| pp24 tg8 | 566-576 ms | 566.3 ms (unchanged) |
| diverse stream | baseline | identical (pure refactor) |

The saving exceeds the 16.7 ms/step the copy itself accounts for because the copy also
pushed the KV out of L2/L3 ahead of the kernels that read it. Cumulative on the long-context
decode this session: 142.5 -> 92.3 ms/step (-35%).



## The KV h2d is not a meaningful cost - measured, and the ptr-keyed device cache is unsound (2026-09-14)

Earlier text in this file attributed ~23 ms/step of the long-context decode to uploading the
KV prefix every layer (33.6 MB/layer at n_past 8192). That was wrong, and the experiment
that tried to remove it showed why.

What was built: a per-(host pointer) device cache with delta uploads, plus a `pos0`
parameter on `qsa_attention_sel` as the explicit reset signal (pos0 == 0 = sequence start).
The invalidation logic was validated: a short prompt (24 tokens) followed by a longer one
(200 tokens) in the same process reproduced both single-sequence token streams exactly
(seq0: 220 248046 198 248045 74455, seq1: 477 871 198 220 3376), so the stale-prefix hazard
is handled.

It did not help, for two reasons. First, the timing was identical with the cache in place
(737.5 vs 738.4 ms for pp8192 tg8) even though it was missing on every call, which means
the upload was never on the critical path. Second, the design itself is unsound: keying on
the host pointer means a growing prefix needs a larger buffer, and allocating a fresh entry
per length fills the map (48 entries, ~1 GB) and thrashes; growing the entry in place leaks
the previous device buffer on each growth until an h2d fails with error 700.

So the context-scaling cost measured earlier (pp2048 103.9 -> pp8192 141.8 ms/step) comes
from the kernel's K/V reads and the longer selection list, not from the upload. If this is
revisited, the KV must be device-resident and owned by the frame's sequence state with the
append done device-side - not a pointer-keyed cache in the adapter. The `pos0` reset idea
remains valid for that design.



## The Flash-Next prefill is 55% non-kernel, and the QSA stage is most of it (2026-09-14)

KTRACE for a pp2048 chunk (17 kernel types, 4,006 ms total against a ~8,900 ms wall):

| kernel | total | calls | share |
|---|---|---|---|
| q4_gemm_q4k_ge (MoE gate/up grouped) | 785.5 ms | 52 | 20% |
| gemm_q8_j128 | 760.9 ms | 776 | 19% |
| q4_gemm_f32_m | 630.3 ms | 288 | 16% |
| gdn_ar_w_swap | 344.4 ms | 36 | 9% |
| q4_qsa_attn_sel6 | 286.6 ms | 12 | 7% |
| q4_rows_permute_u32 | 226.1 ms | 331 | 6% |
| gemm_q8_0 / quant_q8 / rms_* / hc_* | ~880 ms | | 22% |

CAREFUL - the same instrument caveat recorded for the 27B applies here: KTRACE's sum
(366 ms) under-measured a 1,409 ms 27B prefill wall by 4x, and this 4,006 ms likewise
cannot be read as "45% of the chunk is on the device". The table's *relative* ranking is
useful (one instrument, all kernels) but the split against the wall is not; the earlier
"4.9 s is host and synchronisation" claim was wrong and is withdrawn. What is measurable
independently is the stage-timer side (qsa.rs prints t-labelled per-stage times, and those
do sum to the frame total), which puts 2.98 s of the chunk in the QSA stage.
The t-labelled stage timers place 2.98 s of that in the QSA stage: mm_group 873 ms,
attn 1,212 ms, sel+proj 503 ms, sel_build 392 ms. attn is the clearest case - its kernel is
only 287 ms while the stage reports 1,212 ms, so ~926 ms is the d2h drain waiting on queued
work, which also fixes the numbers for the earlier "attention is bandwidth-bound" reading:
the kernel itself is 3.2% of the chunk, not 41% of the QSA stage, and the geometry
experiment's ceiling was therefore ~143 ms (1.6%) - below the run-to-run noise, which is
exactly what the inconclusive A/B showed.

The one trustworthy conclusion is narrower: the attention's own kernel is small relative
to its stage (287 ms against a 1,212 ms stage reading), so the geometry experiment's
ceiling was ~143 ms (1.6%) - below the run-to-run noise, which is what the inconclusive A/B
showed. Everything else in this section needs a working device-time instrument before it
can be acted on.



