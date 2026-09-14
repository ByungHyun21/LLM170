# `crates/core/src/qwen4exp/stages/qsa.rs + stages/moe.rs` — 측정 기록

> `docs/benchmarks.md`에서 **이 파일에 해당하는 항목만** 옮긴 것(제목 기준).
> 표·수치는 원문 그대로. 요약 지표는 benchmarks.md에 남는다.

## qwen4exp decode: the MoE grouping round trip and the device-side attempt (2026-09-14)

The per-layer MoE grouping is the decode's largest single cost. Skipping the
moe.top10 stage (which also keeps the grouping cache valid) runs tg8 in 497.5 ms
versus 770.4 ms normally, i.e. the round trip - a synchronous d2h of the routing
ids, the host table build and three h2d uploads, 48 times per step - costs
34 ms/step. Removing it alone would put the decode at 62 ms/step, level with
llama.cpp's 61 on this model.

A device-side grouping kernel (q4_moe_group_t1) builds the same tables on the GPU
and is bit-identical (the diverse prompt output is unchanged), and it also has a
block-parallel form, but the path measures 1009-1011 ms against the host path's
770, so it is opt-in behind LLM170_MOE_GROUP_DEV. Its additions were isolated in
the micro test (group kernel 30.95 us/call, d2h_issue+d2h_wait 15.98 us/call, about
3.8 ms/step together) and a forced-bound experiment put the enlarged buffers at only
2.4 ms/step, so every added operation is excluded and the remaining candidate is the
changed stream ordering: the host path drains the stream at its ids d2h where the
device path does not. Decisive next experiment: drop d2h_issue from the device path
and re-run the A/B to separate ordering from added work.


## Confirmed with the reps-3 protocol: the grouping round trip is 35% of the decode (2026-09-14)

Warm reps (--reps 3, reps 1-2):
| run | warm ms (tg8) | per step |
|---|---|---|
| baseline | 735.4 / 728.6 = 732.0 | 91.5 ms |
| moe.top10 skipped (keeps the grouping cache valid) | 473.1 / 485.5 = 479.3 | **59.9 ms** |
| device-side grouping (LLM170_MOE_GROUP_DEV=1) | 921.1 / 943.2 = 932.2 | 116.5 ms |

So the per-layer grouping round trip - a synchronous d2h of the routing ids, the host
table build and three h2d uploads, 48 times per step - is 31.6 ms/step, 35% of the
decode, and removing it alone reaches 59.9 ms/step, below llama.cpp's 61 on this model.
The device-side grouping is bit-identical but 27 ms/step slower, reproducibly and
warm, and every component it adds has been measured cheap in isolation (group kernel
31 us, async d2h 16 us, bound-sized buffers 2.4 ms/step) while the ordering and drain
variants make no difference. That unexplained 27 ms is what gates reaching llama
parity, and it needs device-side timestamps or a counter-based method to resolve.


## MoE GEMM ceiling (qwen4exp, measured 2026-09-14)

The grouped MoE GEMM (q4_K, per-expert 16-row-aligned padded layout) cannot use a
single dense WMMA launch: an MMA fragment shares the A operand across the token
axis, but the grouped layout makes A (the expert weights) a function of the token
(the expert varies per token). Folding the expert into the grid z axis reads out
of range; the only valid decomposition is one launch per expert.

Per-expert dense `gemm_q4k_j128` launches were implemented and measured: numerics
valid (diverse-prompt output identical; an 11,750-token random prompt diverges at
one newline token, 271 vs 198 - a rounding tie-break, the expected signature of an
alternative accumulation order). Speed: 110.3 s vs 112.1 s total = -1.6%, within
the ~10 s run-to-run load variance. No gain.

Reason: at t=2048 there are 16,384 grouped rows over ~512 active experts, i.e.
~32 rows per expert, so a 128-token WMMA tile is 75% wasted, plus ~512 launches
per chunk. The scalar grouped kernel (4.4 TFLOPS) is effectively optimal for this
shape. The -11% seen with the (incorrect) dense-tiling kernel is therefore not
reachable in the grouped layout.


## qwen4exp decode: the per-layer MoE grouping round trip is 44% of the step (2026-09-14)

Stage-skip deltas after the L2Rows fix (tg8, two runs each, noise +-4%):
none 790/887/837 ms, moe.shared 775/771 (about -8 ms/step), **moe.top10 491/478
(-353 ms over 8 steps = -44 ms/step = 44% of the 100 ms step)**.

The top10 op itself is not the cost: moving its selection arrays from (dynamically
indexed, hence local-memory) `int sel[64]; float sp[64]` to shared memory changed
nothing measurable (826.8 ms, bit-identical output). What the skip actually removes
is the `moe_gen` bump, which invalidates the MoE grouping cache - so the 44 ms is
the per-layer host round trip: a synchronous d2h of the routing ids, the host builds
of the perm/inv/rowexp/padded tables, and three h2d uploads, 48 times per step. Each
d2h flushes the pipeline, which is why the GPU sits near 20% utilisation even though
the kernels themselves are microseconds.

Fix direction: build those tables on the device (they are pure functions of the ids),
which removes the round trip and keeps the results bit-identical; the prefill can keep
the host path. This also unblocks the graph capture facility (segments collapse).


## P1 (deq-f16 weights) does not transfer to this engine (measured 2026-09-14)

The reference stack's #05 (bf16/f16 dequant cache + tensor-core GEMM, ablation 1.42x
on a 16384-token prefill) was ported as far as this engine already allowed: the
dequant_q4k_f16 kernel and the gemm_f16_v4 tensor-core path existed, so a q4_K sibling
of gemm_f16_q6 was added and wired into both the single-weight and the grouped frame
GEMMs, gated to t>=128. Measured on pp2048 with --reps 3: 9125.1/8176.9 ms against
9079.4/8192.9 baseline, i.e. neutral. Reverted.

The reason is visible in the numbers: our prefill runs at 2.8 TFLOPS (5.5% of the f32
peak) and streams the weights far below DRAM bandwidth, so it is bounded neither by
compute nor by bandwidth - the deq-f16 route addresses both of those and therefore
changes nothing. The prefill needs the same treatment the decode got: decompose it by
stage, find the inefficiency, and fix that. This also lowers P1's priority for the 27B.



## Direct-ids MoE GEMM for t=1: -8% decode, bit-identical (2026-09-14)

The t=1 MoE no longer builds a grouping at all. Two new kernels
(q4_gemm_q4k_ge_ids for the gate/up, q4_gemm_q5_1_gm_ids for the down - the down is
Q5_1 in this quantization, which is why making only the gate/up direct recovered
nothing: the grouping still fired at the down) read ids[row] directly, so the row
order is the ids order and the gather, the scatter and the whole host round trip
(a synchronous ids d2h, the table build, three h2d uploads) disappear. The per-row
dot products are unchanged, so the diverse-prompt output is bit-identical and the 27B
is unaffected.

Warm tg8 with --reps 3: 716.7/715.2 before, 673.2/669.2 after = 671.2 ms, i.e. the
decode goes from 91.3 to 83.9 ms/step (-8%), and the session's total from 131.9 to
83.9 (-36%). The direct path is the default; LLM170_MOE_GROUPED=1 restores the
grouped path.

Note on an earlier number: the moe.top10 stage-skip showed -33 ms/step, but that
measurement leaves stale ids in place so the model degenerates and its GEMMs speed up.
With the direct path in place the same skip still shows -27 ms/step, which is that
artefact plus the top-k op itself; the real recoverable grouping cost was the 5.3
ms/step this change delivers.



## Parallel top-k selection in the MoE router: -12 ms/step, bit-identical (2026-09-14)

After the direct-ids change the decode step's kernel list showed q4_moe_top10_m as the
single largest entry: 13.3 ms/step at 0.278 ms per call for 48 calls, against a
theoretical ~10 us of work. The cause was the selection: a single lane ran a 512
iteration stable insertion sort. It is now a k-round warp selection - each of the 32
lanes owns 16 experts, picks its own best unused one, and a shuffle butterfly
reduces to the global best, whose lane then consumes that slot; the zeros and the
weighted sum follow the same order as before, so the diverse-prompt output is
bit-identical.

Warm tg8 (--reps 3): 673.2/669.2 -> 579.7/568.9 = 574.3 ms, i.e. 83.9 -> 71.8 ms/step.
Across the session the decode went from 131.9 to 71.8 ms/step (-46%), and against
llama.cpp's 61 ms/step the gap is now 1.18x, down from 2.16x.



## The MoE gate/up GEMM is already near-optimal; the down projection is the gap (2026-09-14)

Arithmetic correction. The t=1 grouped GEMM for the gate/up launches 94 times per
step at 4.76 ms total (51 us per call), and each call reads ten experts' slices of
0.92 MB, i.e. 9.2 MB - so the effective pull is 180 GB/s against the 236 GB/s the
probe measures. The kernel is at ~76% of DRAM and there is little to win there, which
is exactly why the K-split (4x the blocks) bought 1.4%, the GEMV-style grid (16x the
blocks, per-output-row tree reduction) was neutral, and the access-pattern probes
never matched a 10x deficit: the deficit was arithmetic, not architectural. Both
experiments were reverted.

The real gap is the down projection: q4_gemm_q5_1_gm_ids takes 5.97 ms for 0.43 GB,
i.e. 72 GB/s against the gate/up's 180 - a 2.5x deficit worth about 4 ms/step. That
kernel keeps a shared-memory tile load from the grouped form (it stages a 16-row
weight tile cooperatively) and its per-row expert lookup defeats that staging; giving
it the gate/up treatment is the next step.



## Why the MoE down projection cannot use the grouped kernel's staging (2026-09-14)

The down projection (Q5_1, direct ids) runs at 72 GB/s. The cause is the weight access
pattern: each thread owns one (output, row) pair and walks its own weight row, so a warp
covers 16 consecutive output rows 480 bytes apart - sixteen 32-byte sectors per 64 useful
bytes, an 8x sector amplification that puts the real DRAM traffic at the limit while the
useful rate reads as 72 GB/s. `q5_1_gm` avoids this by staging a 16-row weight tile
cooperatively (its (oo, ww) load has consecutive ww, hence coalesced).

Porting that staging to the direct-ids kernel was attempted and fails structurally: the
staging invariant is one expert per tile (tile_exp), while the unsorted direct form has a
different expert per row. An output row's weights must come from sixteen different
experts (one per row in the tile), so no single shared buffer represents it. The kernel
was reverted; the diverse baseline is intact.

A warp-per-output-row kernel was designed as the remaining candidate and also fails on
inspection: q5_1 stores six words per 32-value super-block, so a lane holding whole
super-blocks walks the row in 24-byte strides and still touches 15 sectors per 80 useful
bytes, a threefold gain at best rather than the eightfold the pattern suggests. Reaching
full coalescing requires splitting a lane's work by word role (scale, high bits, four
quants) and shuffling them back, at which point the kernel's complexity outweighs the
~2.5 ms/step. Routing the down projection through the sorted grouped path instead costs a
per-layer permute plus the grouping round trip, which the earlier measurement puts at
about the same 2.4 ms the coalescing would recover - i.e. neutral. Conclusion: the down
projection's 6 ms is inherent to the Q5_1 layout under the unsorted direct-ids scheme, and
the gate/up path's 180 GB/s is not a reachable target for it. The Flash-Next decode
therefore sits near its practical floor at 70.8 ms/step, with the prefill (MoE down plus
QSA) as the remaining lever.



## The Q4_K MMQ-class tile is correct but not faster at prefill shapes (2026-09-14)

q4_gemm_q4k_y (row-batched tile, dequant hoisted out of the row loop) was default-off with
a comment citing a measured wrong result at ffn_gate_exps t=20. Re-verified:

| check | default | LLM170_Q4K_MMQ=1 | +LLM170_Q4K_Y=1 |
|---|---|---|---|
| q4k-micro (16x256, t=16) vs CPU ref | 3.66e-4 | **0.0 (bit-identical)** | 4.88e-4 |
| Flash-Next diverse 24-token prefill + 8 decode | baseline | **baseline (bit-identical)** | **baseline (bit-identical)** |
| pp2048 | 9,012-9,118 ms | - | 8,899-9,155 ms |

So the numeric objection no longer holds - the kernel agrees with dot_q4k_q8 exactly on the
probe and the model stream is unchanged with it enabled. It is still default-off, but now
for a speed reason: pp2048 is neutral within noise, and q4k-bench at t=20, 2560x640 shows
0.498 vs 0.434 ms per call, i.e. the tiled form loses at these shapes. That closes the
tiled-Q4_K route as a lever for the QSA stage's mm_group: the prefill's Q4_K projections
are not slow because the wrong kernel is selected.



## The fused f16 dequant gate is not the lever either - it needs a per-graph cache (2026-09-14)

The f16 fused-dequant GEMM was gated to q8_0 only (ty0 == 8) even though dequant_q4k_f16
and dequant_q6k_f16 were already wired into its dispatcher. Extending the gate to
matches!(ty0, 8 | 12 | 14) and re-measuring:

- the path is taken: LLM170_F16_DBG=1 shows 48 calls at a 200-token prefill, including the
  QSA projections ([6144x2560] x12 = wq, [2560x512] x24 = wk/wv/indexer) and the shared
  expert's FFN ([2560x12288] x12);
- it is correct: a 200-token prompt (t >= 32, so the gate is actually exercised, unlike the
  24-token diverse which never reaches it) produces token-for-token the same greedy stream
  as the default path;
- it is neutral: pp2048 9,087.8 ms against 9,043.9 ms baseline.

So plans/66 P1's actual content is the part that was skipped: "dequant once per graph and
cache". Here the dequant runs per call, so every call pays a full dequant pass over the
weight (8.8 MB Q4_K -> 25 MB f16 written and re-read) and the saving on the integer-ALU
side is cancelled. The gate extension was reverted as neutral; the recorded next step is a
graph-scoped dequant cache, which is what would make the f16 WMMA route pay.



## The Flash-Next prefill GEMMs are flat at ~2.5 TFLOPS across batch sizes (2026-09-14)

q4k-bench on the QSA wq shape (2560x6144, Q4_K, MMQ tile path by default):

| t | ms/call | TFLOPS | weight rate |
|---|---|---|---|
| 128 | 1.474 | 2.73 | 6.0 GB/s |
| 512 | 6.401 | 2.52 | 1.4 GB/s |
| 1024 | 12.581 | 2.56 | 0.7 GB/s |
| 2048 | 27.203 | 2.37 | 0.3 GB/s |

Flat efficiency means this is not an occupancy or batch-shape problem: the kernel costs
the same per FLOP at t=128 as at t=2048, and the weight rate is nowhere near DRAM either,
so it is per-element cost in the Q4_K dequant/dot path for this shape. That is also why
the 27B reaches 19.5 TFLOPS with the same kernel family - its n_in/n_out (5120x17408) are
large enough to amortise the row work. Closing the Flash-Next prefill gap therefore means
a better tile for short-K wide-N shapes, not a scheduling change.



## The prefill's MoE grouping is 25 ms per call = ~2.4 s per chunk (27%) (2026-09-14)

Measured with the existing LLM170_MOE_TIME (moe-phase buckets, 604 calls over a cold+warm
pair):

| phase | total | calls | per call |
|---|---|---|---|
| weight | 21.77 s | 144 | 151 ms (one-off cold uploads: 48 layers x 3 tensors = 144) |
| group | 4.855 s | 192 | 25.3 ms |
| gemms | 0.005 s | 14 | 0.33 ms |
| gather | 0.000 s | 4 | 0.12 ms |

The weight bucket is the first-use weight upload (exactly 144 = 48x3, so it is a cold cost,
not steady state). The group bucket is steady state and is the real item: at 192 calls over
~2 chunks it is ~2.4 s per chunk, about 27% of the prefill.

Where it comes from: frame_moe_gemm branches on t_cur() == 1. Decode takes the device
path (d2h_issue, async, deliberately hidden behind the gate/up GEMMs), while the prefill
takes the host path - a synchronous d2h of the routing ids (rows = t*k_sel = 20,480 u32 =
80 KB), then a host-side sort into perm/inv/rowexp/tilexp tables, then three h2d's. That
host path was chosen because the device path pads each expert to 16 rows (bound =
rows*16+16), which at 20,480 rows is 16x the gather/scatter work - so the device variant is
t=1 only, with an A/B recorded in the code (host 755.4 ms vs device 1006.9 ms at tg8).

The lever is therefore a device grouping with an exact (unpadded) bound for the prefill, or
at minimum removing the synchronous 80 KB d2h from the host path. Either way it is the
largest identified prefill item now that the gap map is trustworthy.

Attempts so far (all reverted). q4_moe_group_t1 computes off_pad as
`accp += (r + 15) / 16 * 16` per expert, i.e. exactly 16-row alignment, so
Sigma_e ceil(r_e/16)*16 <= rows + 16*ne is a valid bound and the path's rows*16+16 is a 16x
overestimate. Enabling the device path for the prefill with the tighter bound (and dropping
the t_cur()==1 gate) still fails immediately with h2d error 700, so the bound is not what
breaks: the fault is elsewhere in the device path when it runs at prefill scale (rows
20,480 against the decode's 10). That was checked: the fault is a second copy of the bound.
q4_moe_group_t1 also zero-fills perm_pad from max(rp, bound & ~15) up to `int bound =
rows*16+16` computed *inside the kernel*, so a host that uses a smaller bound writes past
the allocation. Passing the bound in as an argument (one signature change plus one launcher
arg) fixed that inconsistency but the run still fails with h2d 700 at the first call, so at
least one more place still assumes the 16x bound - the padded GEMM/gather path is the
obvious next suspect. Until all of them agree, the prefill stays on the host path and the
25.3 ms/call stands.

