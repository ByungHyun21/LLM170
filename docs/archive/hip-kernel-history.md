# 기록 보관 — hip-kernel-history

> `docs/benchmarks.md`에서 옮긴 이력 항목(제목 기준). 수치·표는 원문 그대로.
> 최신 지표는 `docs/benchmarks.md`, 파일별 실측은 `docs/source/`를 본다.

## Vision (mmproj) — HIP (2026-09-05)

Photographic test (llama.cpp's own test-1.jpeg, NYT front page):
ours reads "The front page of The New York Times newspaper, dated Monday, July 21, 1969,
features the historic headline \"MEN WALK ON MOON\"" — fine text (date, headline) read
correctly; llama.cpp reference describes the same image (newspaper, NYT masthead).

CLIP ViT 27 blocks on GPU (f32 weights resident, tiled GEMM + flash attention v2):
vision encoding 47s (CPU) → **2.4-3.1s forward** (+7.1s one-time weight upload per process).
Output verified identical to CPU path and semantically matching llama.cpp on test images.



## MTP speculative decode — HIP (2026-09-05)

Natural-language prompt, 27B Q4_K_XL-class quant, greedy, warm reps.
Ours: GPU MTP layer + batch verify + carry-over GDN (spec k=4):

| Engine | tg (t/s) | vs llama.cpp MTP |
|---|---|---|
| ours HIP non-spec | 10.4 | 0.67× |
| ours HIP spec k=4 (warm) | **19.2-19.4** | **1.25×** |
| ours HIP np4 (batched decode) | 19.0 aggregate | 1.23× |
| ours HIP np8 (batched decode) | 27.9 aggregate | 1.80× |
| ours HIP np4×spec4 (merged verify, natural text) | **27.1-28.1 aggregate** | **1.75-1.81×** (llama.cpp np4+MTP 15.5) |
| ours HIP np8×spec4 (verify cap 64) | 22.2 aggregate | 1.43× |
| llama.cpp MTP (np4 server) | 15.5 | 1.00× |

Token stream verified bit-identical to non-spec greedy (64/64).
Acceptance 4-5 tokens/verify at k=4 on natural text.



## Non-GEMM prefill-kernel fixes (2026-09-10, plans/40 cont.)

Three prefill glue kernels were sequential or thread-starved:

- qk_rope ran ONE thread per work group with sequential f64 arithmetic
  (0.72 ms/layer at t=64). Parallel RMS (64 threads, f64 partial sums
  via shared memory) + rope pair per thread: 0.064 ms — 11x.
- l2 normalized with 32 sequential subgroupBroadcast adds per group;
  one subgroupAdd replaces the loop: 0.33 -> 0.005 ms — 66x.
- addrms used 32 threads per row (only t work groups in flight);
  widened to 256: 0.083 -> 0.045 ms.

gdn_ar (20 ms/chunk) remains sequential-recurrence bound (two subgroup
reductions per token step); the WY-representation chunked formulation
is the identified next-arc algorithmic item.

Benchmarks (defaults, no env): pp64 192.1-195.7 (was 179), pp512
179.9 (was 168), tg8 9.46-10.19. vs llama Vulkan: pp64 0.79x, pp512
0.50x, tg8 ~0.81x. verify: 22 PASS / 3 FAIL — identical set before
and after all three numeric-order changes.



## Tile-kernel exploration closeout (2026-09-10, plans/40 final)

The cm1 tile ceiling (~65-67 GB/s solo) resisted every kernel-internal
lever: uvec4 staging regresses on RADV (dynamic select chains), a
double-buffered pipeline can't overlap without async copies (cm1 has
none — coopMatLoadTensorNV is cm2), and BK=64 (llama's own choice,
halving k-loop barriers) is neutral — barrier count is not the solo
binding constraint. Only [[unroll]] staging helped (+4%, shipped).

tile_ms128's engine regression was root-caused to in-order-queue
scheduling: fat (1.3-2.5ms) kernels stretch waits for the serial
small-kernel chain (rope/flash/gemv) between them; GPU busy fraction
is ~equal (~70%) for both variants across whole-run windows. A
barrier-free direct drain (ms128v2) hit an RADV coopMatStore
ColumnMajor pathology (workgroups 48-95 of 6144-row tensors drop
stores entirely; 10240-row tensors unaffected) — sealed. The
ms128-family harness now pushes the row_off fifth constant (undefined
memory before, since the split experiment).

The structural fix (layer-crossing dependency graph so layer L+1 tiles
overlap layer L attention/GDN) is specified in plans/40 for the next
arc, alongside the WY chunked gdn_ar formulation.

Defaults stand: pp64 195-199 (0.80x), pp512 174-185 (0.51x), tg8
9.8-10.2 (0.82x), verify 23 PASS / 2 FAIL.



## tile_ms4gy — L2 weight reuse via grid-y (2026-09-10, plans/40)

The pp512 gap mechanism finally decoded: llama's column-tile workgroups
re-read identical weights within a single dispatch (L2 hits); our host-
loop 64-token slabs let L2 evict between dispatches. tile_ms4gy puts
every slab of the chunk in one dispatch on grid-y — 85.5 GB/s effective
at t=512 solo (vs 65 isolated), halving q5 tile DRAM per chunk. Engine
default (LLM170_VK_GY=0 opts out). T_MAX stays 128 (256 showed no gain).

The mm_llm large-t harness defect was a stale-spv/env combination
(recompiled mul_mm + explicit VKMMQ_SPV path): llama's own l-tile
streams weights at just 41 GB/s solo in our harness — their pp512 win
is the traffic reduction, not kernel throughput.

Benchmarks: pp512 183-186, pp64 195-197, tg 9.8. verify: 25 PASS /
0 FAIL — all gates green for the first time (previous borderline
spec_np4_seq2 and spec_long_np4_seq3 now pass).

### gy expansion to all quant types (opt-in, neutral)

tile_ms4gy's pattern (token slabs on grid-y) was mechanically ported to all
ms tile variants (LLM170_VK_GY2=1). A real bug was found and fixed: the gy
dispatch re-pushed the ktab LUT that the ms path had already bound (12
buffers vs an 11-binding layout) — xs/nl output was all zeros; q4/q6/q3/q8
were token-correct from the start. After the fix, all-type gy is
token-identical but perf-neutral (pp512 183.5, pp64 198.4 vs 183.7/196.9):
per-tensor slab pairs are already adjacent dispatches, so L2 catches the
re-read for non-q5 types. Kept opt-in (zero-risk default).



## Single-pass prefill (plans/41, 2026-09-10)

Root cause of the prefill weight re-read: `prefill_rows` capped prefill chunks
at 64 tokens unless the 128-row legacy tile family was active, so a 512-token
prompt streamed all 15.79GB of layer weights eight times. The ms tile family
handles t=512 exactly (token stream byte-identical), so T_MAX was raised to
512 and the chunk gate now follows the ms family.

Results (Qwen3.8-27B Q4_K_XL, Vulkan/RADV, Strix Halo): pp512 183 -> 219 t/s
(+20%), pp64 195, tg4 10.0. All-type grid-y slab merge (LLM170_VK_GY2=1) was
neutral-to-negative at both chunk sizes and stays opt-in; q5 gy remains the
default.

Measurement note: the per-dispatch timestamp sum is NOT a valid busy-time
metric when the engine overlaps independent dispatches — at t=512 it reports
6x the wall clock. Use wall-clock A/B for anything scheduling-related.



## Decode cost structure (plans/42)

Layer-scaled decode timing (prompt 8, tg 4) gives a clean linear fit:
~10.7 ms fixed per token plus ~1.37 ms per layer — at 64 layers that is the
measured ~98 ms/token. The fixed part is dominated by the output projection
(1.02 GB of weights per token); the per-layer part streams 247 MB of layer
weights, i.e. ~181 GB/s effective against llama's ~204-219 GB/s for the same
structure. Per-kernel GPU sums stay ~1.5x above the wall clock while the
engine overlaps independent dispatches, so decode gaps are scheduling, not
kernel math.

Speculative decode (MTP head, --spec k) currently drafts with 0% acceptance
on the bench prompt (fwd 8, gen 8, 1.00 tok/fwd) and is therefore pure
overhead — the draft path needs investigation before it can be a throughput
lever.

Current standing (Qwen3.8-27B Q4_K_XL, RADV/Vulkan, Strix Halo):
pp64 197.1 t/s (0.81x llama), pp512 205-213 (0.58-0.60x), tg8 9.75 (0.80x).



## ms256 tile — BN=128 with four subgroups (opt-in, plans/43)

tile_ms256 keeps ms4's per-subgroup accumulator layout (acc[4][2]) but runs
four wave64 subgroups per workgroup, so one 256-thread workgroup covers 128
tokens per weight read (BM=64, BN=128). Measured: pp512 219.8/220.0 t/s vs
216.3/216.4 baseline (+1.6%), but 167.6 t/s at pp64 where a 64-token batch
wastes half the accumulator — hence gated to t >= 128 and kept opt-in
(LLM170_TILE_MS256=1). Kernel-solo rate drops to 36.7 GB/s (from ms4's 61)
because of the larger drain and LDS footprint, which eats most of the
halved weight traffic; the width gain is real but small on this hardware.



## Decode chain links are fully exposed (plans/43)

Removing the GDN autoregressive kernel with LLM170_VK_GDN_SKIP=1 saves
84.5 ms per 8 generated tokens (820.8 -> 736.3 ms), i.e. 10.6 ms/token for
48 kernels whose own GPU time is 8.6 ms/token: each dependency link costs
its full kernel duration plus a small launch/barrier tail. Fusion math:

* Fusable tiny links (gdn_conv 0.014, split3 0.001, l2 0.002, beta_g
  0.002 ms per layer) sum to under 2% of the per-token budget.
* addrms (add + rms_norm) is the large one at 128 links/token (6.3 ms), but
  it cannot be merged: the residual add can move into the producing GEMV
  epilogue, while rms_norm needs a whole-row reduction and its scale cannot
  be folded into the quantized weights without changing the numeric class.
* ar kernel variants (AR4=8/4/2, disabled) measure identically, so that
  kernel's shape is already optimal for this machine.

Conclusion: decode remains a latency-chain problem at 64 layers x ~13
dependent launches; the remaining ~20% gap to llama is launch/dependency
overhead, not kernel throughput.



## BN=128 tiles for all quant types (plans/43) — pp512 +25%

The ms tile family gained 128-token-wide variants for every quant type
(xs/q4k/q6k/q3k/q8/nl), produced by the same transform that made ms128 out
of ms4: two subgroups keep the 8-accumulator layout but each covers 64
tokens, so a weight block is read once per 128 tokens. Every variant
verifies at maxrel 0.003-0.006 (t=512) and the engine token stream is
byte-identical. Weight traffic per 512-token pass drops from 8 to 4 reads
per tensor - now the same as llama's column-tile count with the same
bandwidth class.

Cost: a 128-wide tile needs t >= 128, so the path is gated there
(LLM170_TILE_BN128=0 restores the 64-wide family).

Measured: pp512 216.9 -> 269.7 t/s, pp64 198.3 and tg8 9.66 unchanged,
verify 25 PASS / 0 FAIL.

The fix also uncovered an opt-out inversion: LLM170_VK_GY=0 selected the
gy kernel in the arm chain while the dispatcher's use_gy was false, so the
gy kernel ran with the tb-loop grid and produced garbage - it had masked
the ms128 path in every earlier A/B.



## Tile width ceiling (plans/43)

BN=256 (four subgroups, tbase = sg*64, same 16 accumulators per subgroup)
needs 25KB LDS: occupancy drops to 2 WG/CU, the drain grows to sixteen
16-token slab rounds, and it measured 20 GB/s against ms128's 48.5 - the
extra width loses more in occupancy than it gains in weight traffic. The
practical ceiling on this machine is BN=128 with the 18-vec2 LDS stride
(4 WG/CU for the 128-wide tiles, 5 for the 64-wide ones).

Chunk scaling of the current build (same weights, one pass per prompt):
pp128 288, pp256 318, pp384 297, pp512 302 t/s - the pass overhead is
already amortised at 256 tokens, and per-token cost is set by the slab
count (t/128 reads of every weight tensor), not by the pass length.



## Prefill tile kernels are instruction-bound, not memory-bound (plans/44)

Isolation experiments on the ms128 tile (q5_K, BM=64, BN=128):

| probe | result |
|---|---|
| same tensor, t=64 vs t=128 (identical weight bytes) | 48.0 vs 23.5 GB/s - time tracks MACs, not bytes |
| 2 vs 5 workgroups/CU (dummy LDS) | 23.4 vs 23.9 GB/s - occupancy is irrelevant |
| packed weight layout ([row-block][256-block][row]) | 23.5 vs 23.6 - access pattern is not the limit |
| activation loads removed (zero-padded B tile) | 24.0 vs 23.6 - the DRAM case is weight-only |
| 21.6MB vs 61MB tensor, same type | 50.8 vs 23.6 GB/s - L2 residency artifact, not DRAM |

The kernel delivers ~4.5 TMAC/s regardless of shape, so the prefill cost is
(total MACs) / (issue rate): widening the tile (BN=128 -> 256) does not help
because the MAC count is unchanged, and neither does any memory-side tuning.
The remaining lever is fewer instructions per MAC - pre-dequantized f16
weights (2x bytes but no decode ALU) - which needs chunked >max_ssbo
(128MB) buffers; partial caching changes the numeric class (measured token
streams diverge) and the current f16 tile path is stale (512-thread variant,
wrong tokens, 123 t/s).



## f32-activation tile (b32) — kernel correct, engine dispatch no-ops (plans/44)

tile_ms128b32 (same tile with the q8 activation unpack replaced by a direct
f32 read) verifies in the harness at maxrel 0.0005 - better than the q8
path's 0.0055, since the activations are no longer quantized. Solo rates
are unchanged (48.6 vs 50.8 GB/s at t=128, L2-resident tensor; 12.6 vs
12.4 at t=512).

In the engine the same SPV produces zero output for the q5 tensors, and
neither bounding a constant B tile nor writing a constant in the drain
changes the result - the dispatch appears to no-op, which also explains the
apparent +5% (320 vs 303 t/s pp512) on that path: the work is simply not
executed. Buffer-handle mapping was verified correct (xq_n -> b_xn,
xq_g -> b_ggated, xq_f -> b_fglu) and the dispatch parameters match the
harness (nkb 10, pb 24, 128-token slabs, tok_base). Root cause not found;
the experiment is reverted. Worth noting as a robustness gap: a dispatch
that silently does nothing is indistinguishable from a fast kernel in the
timing numbers.



## Tile rate is MAC-throughput bound (plans/44 close)

The last instruction-mix hypothesis is closed: tile_ms128b32 - which
replaces the q8 activation unpack with a direct f32 read, verifies in the
harness at maxrel 0.0005, and issues markedly fewer instructions per
K-block - runs at exactly the same per-call time in the engine (0.967 vs
0.958 ms over 191 calls, from the timestamp profiler). Combined with the
earlier probes (occupancy, weight layout, activation loads, tensor size),
the picture is:

* tile time scales with the MAC count (t=64 vs t=128: 48.0 vs 23.5 GB/s
  on identical weight bytes),
* it is insensitive to occupancy (2 vs 5 WG/CU), weight packing, and the
  decode instruction count.

Effective rates on this machine: our engine ~8.2 TMAC/s of tile work
(1690 ms per 512-token pass), llama.cpp ~9.6 TMAC/s (1436 ms) - the ~17%
tile-rate difference is the bulk of the remaining prefill gap, and no
tile variant tried (BN 64/128/256, four subgroup layouts, BK 16/32, stride
17/18/19/20, packed or f16 weights) moved it.
* BK=64 (half the K-loop barriers, 256 threads with split staging) is also
  neutral on the DRAM case (23.6 vs 23.8 GB/s) and 27% worse when the
  tensor is L2-resident, closing the barrier-count axis as well.



## Tile kernel is load-issue bound (plans/45, definitive)

A raw linear streaming reader on the same buffers measures 333 GB/s
(vec4 loads, 67MB tensor), while the q5_K tile sustains 23.6 GB/s on that
tensor - and that number is insensitive to every structural lever tried:

| lever | DRAM tensor (67MB) | L2 tensor (22MB) |
|---|---|---|
| baseline ms128 | 23.6 GB/s | 34.3 GB/s |
| register-staged software pipeline | 22.6 | **46.5** (+36%) |
| coopMatLoad hoisting (20 -> 8 LDS reads/step) | 23.2 | **48.0** (+40%) |
| + uvec4 d/scales loads | 23.6 | 47.3 |
| BM=32 (2x workgroups) | 23.1 | 31.0 |
| f32 activation (b32) | - | - |

The two optimisations that help when data is L2-resident do nothing when it
streams from DRAM, and neither does request parallelism. The arithmetic
points at instruction issue: the kernel issues ~12 four-byte global loads
plus the q5_K decode per thread per K-block, i.e. ~67G loads/s across the
grid, about 60% of this GPU's theoretical load-issue rate - and the decode
ALU competes for the same issue slots. That is why occupancy, WG count,
latency hiding, LDS traffic, load width for one region, and layout packing
all measure neutral: none of them reduce the instruction count per K-block.

The fix direction is fewer, wider loads per thread (16B-weight loads with
LDS redistribution of the decode), not more parallelism, and it explains
why the b32 experiment (decode ALU traded for 4x more small loads) was
neutral as well.



## Prefill non-tile budget (plans/45 close)

A 512-token pass spends 94% of its GPU time in the tiles; the remaining
6% (inflation-corrected) breaks down as qsa_flash ~38 ms (2.2%), addrms
~33 ms (1.9%), and everything else (gdn_ar8, quant, silu, norm_gated,
split3, gdn_conv, qk_rope2, l2) ~32 ms. No single item exceeds 2.2% of the
pass, so this is not where the remaining ~15% gap lives - it is the tile
rate itself (~8.2 TMAC/s against llama's ~9.6), which twenty-five
structural variants failed to move.

Session standing: pp512 183 -> ~303 t/s, pp64 ~195 -> ~211, tg8 ~9.8 ->
~9.9, verify 25/0 maintained, and every adopted change as well as every
rejected hypothesis is recorded here with its measurements.



## ms256c re-test with correct grid — solo faster, engine slower (plans/46)

The earlier ms256 measurement used a grid that covered only half the rows;
re-measured with the correct launch the four-subgroup BN=128 tile is
numerically clean (maxrel 0.0054) and 13% faster per 64 tokens than the
default 2-subgroup ms128 on an L2-resident tensor (0.265 vs 0.32 ms), yet
the engine regresses 8% (pp512 278-281 vs 302-304). This reproduces the
ms128-vs-ms4 pattern exactly: wider workgroups win solo and lose in the
engine because they co-schedule worse with the other kernels in the chain.
Solo-rate improvements have now failed to transfer in every tile variant
tried; the engine's tile configuration (2 subgroups, BM=64, BN=128, 18-vec2
LDS stride) stays.



## Inter-dispatch GPU gaps are the dominant cost (session 2026-09-11, revised)

Earlier in this session a host-side bottleneck was suspected; the instrumentation was
misplaced (the timer for the "layer loop" was printed after `end_batch_wait()`), so that
number included the GPU wait. Corrected picture, measured on pp8 (gemv8 path, t=8):

| Quantity | Value |
|---|---|
| Wall per forward | ~494 ms |
| Sum of stamped kernel durations (VK_TS) | ~151 ms |
| Dispatches per forward | ~1050 |
| => per-dispatch gap | ~0.33 ms |
| Wall when the layer loop is cut to 1 layer (LLM170_VK_LAYERS=1) | ~20 ms total |
| Per-layer instrumented body (timer inside the loop) | 0.02-0.10 ms |
| `run()` host time (recording, 2210 calls) | 1.4 ms total |
| Descriptor set alloc+update | 0.2 ms total |

So the host recording path is fast (~1.4 ms for 2210 dispatches) and each layer's own
instructions account for <0.1 ms, yet the GPU takes ~7.5 ms per layer in elapsed time
against ~2.4 ms of measured kernel duration. The missing time is between dispatches:
barriers/state switches/cache flushes, ~0.3 ms each, ~1050 times per forward.

The barrier is NOT the cause: setting `vk::DependencyFlags::BY_REGION` changes nothing
(493.8 ms), and disabling the inter-dispatch `vk::MemoryBarrier` entirely changes nothing
either (490.6 ms, pp8). So the gap is the fixed per-dispatch GPU cost (workgroup launch,
drain and state switch), roughly 0.46 ms per dispatch at t=8 and 0.76 ms at t=1 - which
is why the same ~1050 dispatches cost 484 ms for 8 tokens but 800 ms for 1 token.

Consequence: the number of dispatches is the primary lever on the small-t paths. Cutting
3 dispatches per layer (fusing the same-input qkv/gate/up GEMVs into one row-ranged
dispatch) removes ~192 dispatches per forward: ~115 ms per pp8 forward, and ~23 ms per
tg token (~23%, i.e. tg 10 -> 13 t/s), which is what parity with llama.cpp needs.

How the batched path mostly hides this: pp64 (tile kernels) runs ~190 dispatches per
forward instead of ~1050 (one tile dispatch covers many tokens), so 64 tokens cost 302 ms
(~4.7 ms/token) while the gemv8 path (t<16) pays ~1050 dispatches for 8 tokens
(~62 ms/token). Same weights traffic in both cases; only the dispatch count differs.

Next lever (not attempted): narrow the per-dispatch barrier. Options, in order of effort:
1. `vk::DependencyFlags::BY_REGION` on the existing barrier (one word).
2. `VkMemoryBarrier2`/`vkCmdPipelineBarrier2` with explicit buffer ranges instead of a
   global barrier, so RADV need not flush the whole cache.
3. Reduce the dispatch count on the small-t path: process >= 16 tokens per dispatch (the
   tile path) or fuse same-input GEMVs (qkv/gate/up) into one dispatch with an internal
   row-range switch.



## CORRECTION: dispatch count is NOT the bottleneck (2026-09-11, final)

The previous entry ("dispatch count is the real bottleneck") is falsified by a direct
experiment: re-running the idempotent `quant` kernel N extra times per layer
(`LLM170_VK_DUMMY`) adds 64-768 dispatches per forward and changes nothing:

| Extra dispatches/forward | pp8 wall |
|---|---|
| +0 | 614.7 ms (thermal-drifted run) |
| +64 | 617.9 ms |
| +256 | 503.4 ms |
| +768 | 501.7 ms |

Marginal cost of a dispatch is unmeasurable (<= a few us). The correct model for the
same data set is DRAM traffic:

- gemv8 (t < 16) dispatches grid (row-pairs, t): every token re-reads the full weight
  tensor. A t=8 forward moves 8 x 17.5 GB = 140 GB; at the observed ~280 GB/s streaming
  rate that is ~500 ms - which is the measured wall (494-617 ms depending on thermal
  state). Per layer: 8 x 2.2 GB / 280 GB/s = 7.8 ms - matching the measured 7.5 ms/layer.
- t=1 decode moves 17.5 GB and takes 99 ms -> ~175 GB/s effective (worse latency hiding
  at t=1). llama.cpp reaches 82 ms -> ~213 GB/s on the same tensor set.
- The tile path (t >= 16) reads weights once per column-block; pp64 = 302 ms ~ the
  one-pass tile floor (~300 ms at ~58 GB/s effective on the coopmat path).
- VK_TS per-dispatch spans (sum 151 ms) under-report dispatch duration (known 6x skew);
  do not use them to infer GPU idle. Wall + traffic arithmetic is the reliable model.

Remaining single-stream levers, quantified:
1. tg: close 175 -> 213 GB/s at t=1 by restructuring gemv8 for more ILP per workgroup,
   the way llama's mul_mat_vec does (NUM_ROWS rows x NUM_COLS tokens per workgroup with
   cooperative K reduction, spec-constant sized, vs our 1 row-pair per WG). Expected
   ceiling ~1.2x tg (10 -> 12+ t/s, i.e. llama parity).
2. pp: tiles re-read weights per BN column-block; raising BN halves re-reads (already
   at BN=128; BN=256 measured neutral earlier - revisit only with occupancy data).



## HIP re-baseline and RMS-kernel round (2026-09-12)

Backend focus moved to ROCm/HIP (the Vulkan arc closed at 0.977x tg / 0.87-0.90x pp).
Interleaved A/B harness: `scripts/ab_bench.sh <binA> <binB> <reps> --pp 512 --tg 32`
(A/B alternation cancels the ~3% thermal drift that made single-run comparisons
unreliable; all ratios below are B/A medians of 3 pairs).

Reference (llama.cpp build 8b4b3558f, same GGUF `q35work.gguf`, `-ngl 99 -fa 1`,
llama-bench): **pp512 353.62, tg32 11.47**.  Our zero-config HIP at HEAD (2bacd60):
**pp512 313.6, tg32 10.98** (0.887x / 0.958x).

### Where the prefill time goes (LLM170_KTRACE=1, t=512 single pass)

After extending KTRACE to `launch()` and the direct MMQ launches (the MMQ GEMMs
were previously invisible — they are launched with `hipModuleLaunchKernel`, not
through `launch3`), the 512-token pass decomposes as:

| item | ms | note |
|---|---|---|
| mul_mat_q (q4_K/q5_K/iq4_xs) | 955 | 221 launches, 59 GB/s effective on 4x weight re-reads |
| q6_K tiles (j128) | 163 | 2.86 GB at 70 GB/s |
| gdn AR scan | 99 | 48 layers, sequential over 512 tokens |
| qsa_flash_wk + merge | 60 | 16 full-attention layers |
| q8_0 / iq4_nl / q3_K / iq3_s tiles | 81 | |
| rms_part + rms_finish | 104 -> 20 | fixed this round (below) |
| silu_mul_f32 | 28 | 244 GB/s, at streaming limit |
| norm_gated_silu_f32 | 22 | one warp per row, ~82 GB/s |
| quant_q8 | 25 -> 6 | skippable for f32-direct consumers |
| mmq_quant_y | 18 | y re-quantized per GEMM (cache disabled since 부록90) |
| axpy/split3/conv/qk_rope/l2 | 38 | |

### Adopted this round

1. **rmsq (t=1 norm+quant) widened 32 -> nblk threads** (one 32-value block per
   thread; the reduction keeps its 32-chunk order). Bit-identical greedy stream.
2. **rms_part/rms_finish vectorized** (float4 loads; rms_part keeps the exact
   per-chunk addition order x,y,z,w; rms_finish is elementwise so it uses a
   coalesced stride). 104 ms -> 20 ms per pass.
3. **Activation q8 skipped when every consumer takes the f32-direct path**
   (MMQ reads y_f32 and re-quantizes internally). `mmq_used`/`mmq_used_s` mirror
   mm_b2/gemm_mmq_s conditions exactly; the skip is group-conservative (all
   consumers, both streams). t=512 launches 256 -> 102, kernel time 25 -> 6 ms;
   end-to-end neutral (the removed work was hidden behind the side-stream GEMMs).
4. **hipFuncSetAttribute(MMQ smem) cached per function** instead of per GEMM.

Combined: pp512 314.0 -> 327.8 (+4.4%), tg32 10.96 -> 11.06 (+0.9%) at step 1-2;
330-333 pp after step 3 (within noise of 327.8). Standing: **0.94x pp / 0.965x tg**
of llama.cpp ROCm.

### Falsified / rejected this round (measured)

| variant | ratio vs default |
|---|---|
| `LLM170_Q6MMQ=1` (q6_K through mul_mat_q) | pp +6.5% but **garbage tokens** on >=32-token prefills (requant_q6k_canonical layout defect) — not adopted |
| `LLM170_DEQ16=1` (q6_K dequant->f16 + WMMA tiles) | 0.89x pp |
| `LLM170_NO_MMQ=1` (all types on tiles) | 0.96x pp |
| `LLM170_MMQ_ONLY=3` (iq4_xs to tiles) | 0.968x pp |
| `LLM170_MMQ_ONLY=2050` (q4_K to tiles) | 1.005x pp (noise) |
| `LLM170_ARCHUNK=1` (chunked-parallel AR scan) | 0.894x pp |
| `LLM170_NO_WKFLASH=1` (split4q4 flash) | 0.857x pp |
| `LLM170_NO_QSA_SPLIT=1` (plain flash) | 0.823x pp |

The MMQ routing default (q4_K + q5_K + iq4_xs on mul_mat_q) is confirmed optimal
among the tested routings. Per-type MMQ rates are uniform (11.7-12.7 TMAC/s),
which points at a shared issue/LDS limit rather than a per-type defect; the
kernel holds 49-59 KB of dynamic shared memory, i.e. one workgroup per CU.

### Decode (tg) accounting

`step+greedy` = 90.3 ms/token, of which GEMV kernels ~78 ms (17.55 GB of weights
-> 225 GB/s effective), non-GEMV ~10 ms, host ~1-2 ms. Isolated single-kernel
measurements: tensors >=47 MB stream at 207-218 GB/s (DRAM-bound), 21-36 MB
tensors reach 400-457 GB/s (L2-resident). Per-tensor t=1 rates by type: q5_K 253,
q4_K 227, q6_K 204, iq4_xs 198, iq4_nl 158, q8_0 111 GB/s.
The FFN gate/up GEMVs are already dispatched on two streams; the 4 GDN in-proj
GEMVs use the fused dual kernels.



## q6_K MMQ fix + gate results (2026-09-12, final block)

**Root cause of the q6_K MMQ garbage, found in plans/i8_arc**: the MMQ code
objects are direct instantiations of llama.cpp's own headers
(`mmq_native_rdna35.cu` does `#include "mmq.cuh"`), so `mul_mat_q<Q6_K>` expects
the ggml canonical block (ql|qh|scales|d with d at 208). Our weights are uploaded
verbatim from the GGUF, i.e. already canonical — but the Q6MMQ path first ran
`requant_q6k_canonical`, which permutes them into a d-first layout
(d at 0, scales at 194). The kernel then read ql from the wrong bytes and
produced garbage on >=32-token prefills. Bypassing the requant makes the route
correct: identical greedy stream to the tile path on a 120-token prefill, and
`LLM170_Q6MMQ=1` measures +1.3% pp.

q6_K now takes the MMQ route by default (`LLM170_NO_Q6MMQ=1` restores tiles,
`LLM170_Q6RQ=1` restores the legacy requant). Interleaved A/B (3 pairs):
**pp512 328.5 -> 332.4 (+1.2%)**, tg32 unchanged 11.06.

Gate (scripts/verify.py, 2-phase, fresh llama reference re-collected and
re-judged): **16/19 PASS, all 9 spec_* invariants exact** (`--spec 4` output ==
non-spec greedy). The three FAILs are reference-side:
- `long_np2_seq1` / `long_np4_seq1` (long2): llama's reference stream changes
  between collections (the documented 2026-09-06 slot-KV instability case); our
  tokens match the previously recorded llama output (16, 13, 159301).
- `long_np4_seq2` (long3): the llama reference produced a single token
  (248044) — a degenerate reference; our first token ranks top-5 (gap 1.56,
  epsilon 1.5).
Both binaries (pre/post this round) produce identical streams on these cases,
so the FAILs are not regressions.

### Falsified this block

| variant | result |
|---|---|
| `LLM170_MMQ_SMALLT=1` (MMQ for t=2..31, i.e. the np4 step at t=4) | 0.66x wall — mul_mat_q stages a full 128-row y tile for 4 real rows |
| `LLM170_MTP_ACC=64` / `=8` (shrink the MTP KV accumulation window) | pp 240/312 (from 84.6) but tg spec4 12.97/5.33 (from 16.3) — the accumulation is what makes the drafts land |

### Remaining gaps, in order of size (all measured)

1. **np4 batched-decode step (t=4)**: ours ~19.8 t/s aggregate (CLI) / 12.18
   (serve, after the scheduler fix) vs llama-server 25.6. The t=4 step runs the
   128-wide tile kernels at 3% row utilisation; MMQ is worse (above).
2. **prefill GEMM** (mul_mat_q ~12 TMAC/s, 1 WG/CU at 49-59 KB dynamic smem):
   pp512 332 vs llama 353.6 = 0.94x. llama.cpp does not use MMQ at t=512 on this
   device (its RDNA3 heuristic disables it above 256 tokens) — it dequantises to
   f16 and calls hipBLAS, which the pure-Rust constraint forbids; the
   equivalent would be a WMMA f16 GEMM with proper tiling.
3. **MTP-mode prefill**: 179 t/s after the head-skip fix (from 84.6); the
   residual is the per-token MTP block (512 sequential t=1 block passes,
   ~1.3 s). A batched MTP prefill (one t=512 pass over blk.64's attention+FFN)
   would bring it to ~300.

### plans-derived candidates for the next round (reviewed 2026-09-12)

- **plans/56 (#28702, fused gate/up + SwiGLU epilogue)**: one Q8_1 quantisation of
  the shared FFN input, both weight matrices in one kernel, SwiGLU applied after
  the K-reduction. For HIP this removes the separate `silu_mul_f32` pass (28 ms /
  pass) and halves `mmq_quant_y` (18 ms) -> ~+2.3% pp. The MMQ objects are
  llama's headers, so the template's fusion hook is the port target.
- **plans/49 (#28528 stream-K / #28457 small-M)**: the np4 step (t=4) runs
  128-column tiles at 3% row utilisation; a narrow-BN tile (BN=16-32) or a
  split-K/small-M variant is the structural fix for the np path (llama 25.6 vs
  our 19.8 CLI / 12.18 serve). Small-t MMQ was measured and rejected (0.66x).
- **plans/46 (handoff)**: (a) spec contract change (batched verify instead of
  per-token bit-identical verify) — explicitly marked as needing approval;
  (b) GDN-layer fusion for decode (multi-tensor binding + WG-range selection) —
  large, and the non-GEMM budget measured here is only ~10 ms/token.
- **batched MTP prefill** (own design): run blk.64's attention+FFN once with
  t=chunk instead of 512 sequential t=1 passes; expected pp(spec4) 179 -> ~300.



## Batched MTP prefill (2026-09-12, final)

Spec-mode prefill ran `blk.64` once per prompt token: four projections + FFN +
KV append + flash at t=1, re-reading ~0.7 GB of MTP weights per token. The
per-token cost (2.5 ms) is one trunk layer's worth, i.e. 512 tokens added 1.28 s
to a 1.56 s prefill.

`mtp_prefill_batch` runs the block once for the whole chunk:
enorm/hnorm via `rms_rows` -> `cat2_rows` interleave -> eh_proj -> q/k/v ->
batched `qk_norm_rope` -> `kv_append_t` -> the existing batched flash (split/wk
variants) -> attn_output/FFN with the batch kernels, head only on the last row.
Pairing is unchanged (MTP(tok_p, h_{p-1}), h_{-1} = pending) so the MTP KV and
`mtp_pending_h` match the per-token path; new `RawDecode::mtp_prefill_batch`
(default impl = per-token fallback, Vulkan unaffected). Two batch-buffer sizing
bugs were caught by the HIP illegal-address fault: the shared xq scratch must
cover `max(n, n_head*hd)` and `max(2n, n_ff)`.

Natural-text spec bench (pp512, tg64, spec4, LLM170_SPEC_GPU=1, 2 reps):

| metric | per-token MTP (head-skip) | batched MTP | plain |
|---|---|---|---|
| pp512 spec4 | 179.3 | **325.3-327.5** | 332 |
| tg64 spec4 | 16.6-17.9 | **19.36-19.59** | 11.07 |

Spec-mode prefill is now at the plain prefill rate, and spec-mode tg is +75%
over plain. Gate (fresh llama reference, 16/19) unchanged: the same three
reference-side FAILs with identical gaps, and **all 9 spec_* invariants exact**.



## Session summary (2026-09-12, HIP focus)

Consolidated interleaved A/B of the whole session, 3 pairs, base = 2bacd60:

| metric | base | now | delta | llama.cpp | ratio |
|---|---|---|---|---|---|
| pp512 | 309.2 | **334.6** | +8.2% | 353.62 | 0.946x |
| tg32 | 10.98 | **11.04** | +0.5% | 11.47 | 0.963x |
| spec4 pp512 (MTP, natural text) | 84.6 | **325-328** | +285% | (n/a) | — |
| spec4 tg64 | 16.6 | **19.4-19.6** | +18% | (n/a) | — |
| np4 aggregate (CLI, 4x64 tok) | 19.8 | **~23.4** | +18% | 25.6 | 0.91x |
| np4 aggregate (serve) | 9.64 | 12.18 | +26% | 25.6 | 0.48x |
| VL request (steady state) | — | 6.0 s | — | 3.68 s | 0.61x |

Adopted: rmsq/rms_part/rms_finish vectorisation, MMQ activation-quant skip,
q6_K MMQ route fixed and enabled, MTP head-skip + batched MTP prefill,
serve prefill interleave, 4-token GEMV (all types + head), KTRACE coverage.

Falsified with measurements: small-t MMQ, MTP accumulation window, GDN AR
prefetch, AR smem variant, DEQ16, ARCHUNK, two attention variants, two
`.co` rebuilds, MMQ I=64 geometry.

Remaining, ranked: (1) prefill GEMM (`mul_mat_q` 955 ms/pass at 12 TMAC/s,
1 WG/CU at 58.9 KB smem — the `.co` is immutable without its header revision,
so the alternative is a new WMMA f16 GEMM); (2) np step state kernels
(conv/AR/flash = 21.6 ms of 163.8 ms per t=4 step, 448 launches); (3) vision
encoder (3.0 s vs llama ~1.2 s).



## MTP ROOT CAUSE FOUND AND FIXED: q6_K misaligned-block read (2026-09-12)

`gemm_q6k`'s misaligned 16-byte extraction assumed the target offset was 2 (mod 8).
q6_K blocks are 210 bytes, so a row's block b sits at `o*blocks*210 + b*210`, whose
offset is 6 (mod 8) whenever b = 3 (mod 4) - i.e. **25% of every q6_K row was read
4 bytes shifted**. The correct case needs a 24-byte (6-word) load and a second shift
branch; the kernel now handles both.

Localisation instrument (new, reusable): `llm170 q6k-ref <model> <tensor>` runs the
engine's GEMV and a scalar GPU reference kernel over the same buffers; with
`LLM170_Q6K_EK=<k|k1,k2,...>` the activation becomes a unit vector so every output is a
single decoded weight. Sweeping k showed the failure appearing exactly at block
offsets 3, 19 (and their mod-4 class) and nowhere else - matching the mod-8 analysis.

Why it hit MTP but not the main decode: the MTP head's GEMMs go through `mm_direct`
(hipRTC `gemm_q6k`), while the main decode path reaches q6_K through kernels that avoid
this branch, so the main stream was byte-identical before and after the fix (verified
three ways: seed prompt, 512-token natural prompt, and CPU engine comparison).

Effect on MTP (natural text, pp128/tg64/k=4):

| path | before | after |
|---|---|---|
| first-position draft acceptance | ~23% | **100%** (12/12) |
| chained acceptance (j>=1) | 0/21 | 50% at j=1, then 33%, 50% |
| tg, GPU chain (`LLM170_SPEC_GPU=1`) | 3.3-8.0 t/s | **19.98 t/s** |
| tg, CPU chain | 5.2 t/s | 5.2 t/s (CPU MTP layer ~150 ms/draft, unchanged) |
| spec vs non-spec stream | - | bit-identical, 25/25 tokens |

19.98 t/s vs 11.0 non-spec is the 1.8x that the earlier session recorded (19.2-19.4).



## Post-fix verification (2026-09-12, after the q6_K fix)

MTP (natural text, pp128/tg64/k=4, `LLM170_SPEC_GPU=1`): **19.98 t/s** vs 11.0
non-spec = 1.8x; spec stream bit-identical to non-spec (25/25 tokens). The CPU-chain
path stays at 5.2 t/s - its per-draft CPU MTP layer (~150 ms) dominates, so the GPU
chain is the production path.

VL gate (`scripts/verify_vl.py`, judge phase): **3 PASS / 1 FAIL**

| case | before fix | after fix |
|---|---|---|
| vl_spec_short | FAIL @gen[1] | FAIL @gen[12] - the documented 494/16311 flat point (top-2 gap 0.527), same pair that the np2 variants pass as a tie |
| vl_spec_np2_seq0/1 | PASS (tie) | PASS (tie) |
| vl_np2_isolation | FAIL | **PASS** |

np4 x spec4 aggregate (bench `LLM170_BENCH_NP=4`, natural text, single prompt in 4
slots): 6.42 t/s aggregate over 128 tokens - 3x *worse* per token than one stream at
19.98, so the merged-verify np path needs its own pass (the earlier session recorded
27.1-28.1 for this configuration, so this is a regression to chase).



## Judge after the q6_K fix: 16/19 PASS (was 10/21) - 2026-09-12

`scripts/verify.py` judge phase against the same stored llama-server reference:

- **all 10 spec cases pass** (`spec_short_seq0`, `spec_np4_seq0-3`, `spec_long_seq0`,
  `spec_long_np4_seq0-3`) - the MTP stream is bit-correct end to end again;
- np4_seq2/3, long_prompt, long_np2_seq0, long_np4_seq0/3, single_short, np4_seq0/1,
  long_gen96 also pass (7 exact, 4 near-tie);
- 3 failures, all *non-spec* long-prompt runs: long_np2_seq1 and long_np4_seq1 diverge
  at gen[1] (ours = 13 vs 22, top-3, gap 5.99), long_np4_seq2 at gen[0] (ours = 248046
  vs 561 - the documented flat-point alternative). These are the long-context numerics
  class, unrelated to MTP.



## MTP prefill cost: 64 ms -> 29 ms (2026-09-12)

`LLM170_MTP_TIMING=1` breaks the MTP prefill (t=512) down as: norms+cat 1.3 ms,
eh_proj 3.3, qkv 5.2, attn+kv 4.0, FFN 21.8, head 7.1, plus ~21 ms of host<->device
copies of the token embeddings and the shifted hidden.

Two changes, both proven equivalent (base stream bit-identical; spec == non-spec;
`LLM170_MTP_FULL=1` restore path shows identical tg):

1. **KV-only prefill**: only the last row's attention/wo/FFN is computed (the MTP layer
   is causal, so earlier rows' outputs are read by nobody - the head uses the last row
   and the chain uses the decode-step hidden). Removes the FFN's 21.8 ms and the
   attention's 4 ms over prompt rows.
2. **Device-side h_shift**: new `row_shift_gather` kernel builds `[carry; hidden[0..t-1]]`
   on the GPU from the main model's `xs_t`, so the 2 x t x n x 4 B host round trip per
   chunk disappears (the caller now passes only the one-row carry).

Measured (natural text, pp512/tg32/k=4, 2 reps): NOMTP 1474 ms (347.3 t/s) vs spec
1506 ms (340.5 t/s) - the MTP prefill now costs 31 ms instead of 64 ms, so spec-mode pp
is 0.99x of llama (343.7 t/s) instead of 0.97x. tg is unchanged at 16.0 t/s (1.48x
non-spec 10.83) with either prefill variant.



## MTP prefill: head and FFN skipped on non-final chunks (2026-09-12, second pass)

Only the prompt-ending chunk needs a draft token, so every earlier chunk now appends KV
only (`with_head=false` -> attention/wo/FFN/head all skipped) and the caller no longer
downloads the whole chunk hidden - `raw_prefill_h` returns just the last row (1 x n
instead of t x n per chunk). Verified token-for-token: a 2048-token prompt (4 chunks)
gives identical output with spec and non-spec, and pp512/pp2048 gaps vs `LLM170_NOMTP=1`
are 31 ms and 67 ms respectively (2.1% and 1.1% - the remaining cost is the draft
layer's k/v projections and pair projection, which the KV genuinely needs).



## Small-kernel follow-up: instruction-level slimming is neutral (2026-09-12)

`rmsq`'s quantisation phase rewritten with float4 loads and a tree `amax` (bit-exact:
`fmax` is associative, the quantisation is element-wise, and the reduction path is
untouched). Verified bit-identical, A/B three interleaved pairs: tg 11.10 vs 11.11
(neutral), pp 177.2 vs 176.0 (+0.7%, noise). Kept because it is strictly less work, but
it confirms the conclusion from the multi-block experiment: these kernels' 17-30 us are
not spent in their instruction stream or in their grid shape. The only structural lever
that has moved them is *more* WGs when the workload allows (which is why the GQA
attention helps at long context and hurts below 768).



## Decode small kernels: batched loads on the serial chains (+1.5%, bit-exact)

Direct launch micro-benchmark (new `launch-probe` case, no trace pairing): the platform's
baseline launch cost is **2.0 us** (`axpy_scaled`, any n), while `rmsq` measured
5.1/11.7/33.9 us at n=512/5120/20480 - i.e. it scales at ~1.4 ns per element (about
3 cycles), a *serial load->add chain*, not launch overhead. Batching four loads ahead of
a chain whose add order must stay pinned to the CPU mirror is bit-exact and removes it:

| kernel | fix | effect |
|---|---|---|
| `rmsq` sum loop | 4 loads ahead, adds still element-ordered | 11.7 -> 8.1 us per call (n=5120), 33.9 -> 19.6 us (n=20480); A/B tg **+0.63%** (11.12 -> 11.18) |
| `l2_rows2_scale` sum + scale pass | same, both loops (x2 blocks) | A/B tg **+0.85%** (11.17 -> 11.27), pp +0.6% |
| `gatedq` quant phase | float4 loads + tree amax (exactly equivalent) | **neutral** (11.28 -> 11.28) - reverted |

Both adopted changes are bit-identical (verified with `LLM170_REQUIRE_GPU=1`, GPU path
confirmed), and spec == non-spec still holds. Two further findings: the fix only helps
where the *chain* is the cost (rmsq/l2 sums), not where loads are already batched or the
work is trivial, and `gdn_ar_w`'s 26 us for ~0.1 us of actual arithmetic remains
unexplained (a warp-per-block kernel whose cost is unaffected by this pattern).



## ROOT CAUSE of the "slow small kernels": the bit-exact f64 exp (2026-09-12)

`exp_cr` (used by silu_mul, norm_gated_silu, gatedq, gdn_beta_g, gdn_conv, ...) was a
**correctly-rounded exp implemented with f64 Horner** - deliberately, to match
glibc/Rust `expf` bit for bit for the W4A8 contract. On this iGPU f64 runs at 1/16-1/32
rate with long dependency chains, and that - not block count, occupancy or instruction
count - was the ~15-26 us per call in every "unexplained" kernel.

Direct probe (one warp, one block): `gatedq` 15.65 us bit-exact vs **5.75 us** with the
device `__expf`; at 32 blocks 15.63 vs 5.78. Effect on the whole engine:

| metric | bit-exact exp | device __expf | |
|---|---|---|---|
| tg32 | 11.24-11.25 | **11.35-11.36** | +1.0% |
| pp512 | 345.9-346.1 | **347.7-348.2** | +0.55% |
| judge | 16/19 | **17/19** | better agreement with llama (which also uses a fast exp) |
| `llm170 check` | pass | pass | (GEMM cross-validation, no exp path) |
| spec == non-spec | yes | yes | |
| GPU == CPU bit-exactness | yes | **no** (1 ulp in ~6% of cases) | the only lost property |

The device exp is now the **default** (it improves the very gate the project uses for
acceptance and costs only the internal glibc-bit-exactness); `LLM170_EXACTEXP=1` restores
the f64 path, verified to reproduce the previous bit-identical reference exactly.



## Attention kernel is shuffle-bound (2026-09-12)

Launch-probe sweep of `qsa_flash_gqa` (grid = kv-heads x segments, back-to-back launches):
4 keys -> 9.15 us, 8 -> 15.20, 32 -> 51.46 us, i.e. **~1.5 us per key** with a ~3 us
intercept, independent of block and thread counts. Per key the kernel performs
`gq(6) x 4 keys x 5` shuffle steps per key-group - about 30 shuffles per key at ~35
cycles of shuffle latency - which is the same order as the measurement. The decode spend
1.46 ms/token here (16 layers, 91 us/layer), so the next attention win is reducing the
per-key reduction depth (e.g. one warp per query head, sharing the K/V through L1),
not the grid shape.

Diagnostics kept in `llm170 launch-probe` (rmsq n-sweep, axpy baseline, gatedq block
sweep, qsa_flash_gqa segment sweep) - they are what identified both the f64 exp and this.



## MTP prefill: q projection skipped on non-final chunks (2026-09-12)

Only k/v feed the draft layer's KV, and the attention (which is the only consumer of q)
runs on the prompt-ending chunk, so the q projection is now gated by `with_head` like the
head itself. Output-neutral (verified: the spec stream is unchanged) but *performance*
neutral too: the MTP prefill's remaining ~33 ms per 512-token chunk is not the
projections but the host-side work - dequantising the token embeddings into `tok_flat`
and uploading them (10.5 MB per chunk, a second pass over the same embeddings the main
model already dequantised and uploaded). Optimising that needs the embedding lookup to
happen on the device from token ids, which is the documented next step for this 0.5% gap.



## MTP prefill, second attempt: device embedding lookup (rejected, 2026-09-12)

The MTP prefill's remaining ~13 ms/chunk of host-side cost is the `tok_flat` path:
the caller dequantises the token embeddings to f32 and uploads t x n x 4 B (10.5 MB per
512-token chunk) *a second time* - the main prefill already built that matrix from the
same GGUF rows. A device kernel (`embd_rows_q4k`, mirroring the host `deq_q4_k` exactly)
plus an int32 id array (2 KB/chunk) would remove it.

Rejected on the trade, not on the implementation: `token_embd.weight` is not in the
decoder's injected weight map (`weight 없음: token_embd.weight`), because the GPU never
needs it today - adding it means a **+682 MB** persistent upload/VRAM residency to win
~0.5% of pp512 in spec mode. The measured MTP-mode pp is 341.8 vs llama's 341.8 t/s
(0.995x), and this path is the only way to recover it, so the gap stays documented rather
than paid for.

Both remaining gaps are now fully characterised with their costs:
- base tg 0.980x: the decode attention's 1.46 ms/token is frozen by the verify bit-contract
  (see the previous section) - accessible only by rewriting the decode *and* verify
  attention kernels together;
- MTP pp 0.995x: needs the +682 MB embedding residency above.



## Attention rewrite, third attempt: kernel proven correct, divergence is verify-side (2026-09-12)

A unit probe (added to `launch-probe` alongside the kernel attempt; both are reverted, so
re-add together) feeds the row x head kernel a synthetic setup
with an analytic answer - unit query, unit key, values = dimension index - and the kernel
returns exactly m = 256, s = 1, acc error 0. The kernel's dot reduction, online softmax
and part write are therefore correct; the spec != non-spec divergence seen when it was
wired into both the decode and the t<=8 verify branch is **not** a kernel-math bug.

Where that leaves the investigation: the two paths differ in the *context* they present to
the same kernel - the verify's `n_past` covers all t rows (so segments beyond a row's own
position exist, and their partials depend on how masked keys are treated) while the
decode's covers only its own row. The old kernels tolerate that difference (which is why
the contract has held until now) but the exact mechanism was not identified in the
remaining budget. Next session should start from `qsa_flash_wh` + this probe and compare
the *verify* path's part buffers against the decode's for one row (rather than the token
streams), which localises the masked-segment handling directly.

The decode attention's 1.46 ms/token stands: it is ~10x more shuffle work per key than the
row x head design (8 warps each partially reducing every head x key, 5 shuffles each),
and that is the last identified item of the base-mode tg gap.



## np4 (server, 4 concurrent) beats llama 1.46x - the short-bench numbers were prefill artifacts

Measuring the actual server surface (4 concurrent requests, 512-token prompt, 128 tokens each),
decode-only aggregate:

| config | aggregate | vs llama-server |
|---|---|---|
| no spec | **22.64 t/s** | **1.46x** (llama np4+MTP 15.5) |
| spec3, slots sequential (before) | 12.07 t/s | 0.78x |
| spec3, merged verify (after) | 14.35 t/s | 0.93x |

Two protocol lessons. First, short runs measure the *prefill*: 4 concurrent 24-token requests
cost 4 serial prefills (~5.6s) against ~3s of decode, which is where the earlier "np4 = 11.8-13.2,
0.76x" readings came from; with a 128-token generation the aggregate is 22.6 t/s and the np4
cell is met. Second, the engine's spec path (`spec_step_multi`, the merged verify) was not reachable
from the server: the scheduler ran each spec slot through `spec_step` sequentially, so np4+spec paid
4 full single-stream cycles. The scheduler now routes >=2 spec slots through `spec_step_multi`:
measured 12.07 -> 14.35 t/s aggregate, and `scripts/verify_serve.py` (server np4 == CLI np4 tokens)
stays 2/2 PASS.

Note that on this repeated-text prompt spec *loses* to plain batched decode (14.35 vs 22.64): the
per-cycle acceptance on that prompt is ~2.5 tokens/seq, so the MTP's benefit does not cover the
verify's 4x row count. For np>1 the plain batched decode is currently the better operating point.

Correction (same day): the apparent ~60ms/token server gap was the *model load* - the server
prints its listen log before loading 15.67GB, so a first request absorbs ~9-12s. Instrumenting the
scheduler loop shows `decode=87.9ms step=0.1ms n=1`, i.e. the server decodes at exactly the CLI's
engine rate; the warm server runs at 11.3 t/s base and ~21 t/s spec3, and the 128-token np4 numbers
above (22.6 t/s) already had the load amortised. No server-side per-token overhead exists.



## `wmma-check`: rocWMMA layout probe (2026-09-12)

New diagnostic (`llm170 wmma-check`, kernels/mod.rs-registered `wmma_probe`) exercises one
16x16x16 rocwmma mma with exact integer data and compares against a CPU reference for both B
layouts. Result: **max|delta| = 2e-6 for mode0 (B col_major, the QK^T pattern) and mode1 (B
row_major, the PV pattern)**. This pins down, with evidence rather than assumption, that:

- the accumulator mapping is `idx = lane + 32*sl`, `row = idx>>4`, `col = idx&15` (what the
  attention kernels rely on),
- `matrix_a` row_major with ldm = 16 reads a [16][16] f16 tile as expected,
- `matrix_b` col_major reads a row-major [16][16] tile as B^T (so K stored [key][dim] gives QK^T),
- `matrix_b` row_major reads it as V stored [key][dim] for PV.

So the layouts are not the WMMA attention kernel's problem; its bug must be in the Q/K/V staging
indices, masking, segment/pos handling or the P hand-off. The probe is kept as a permanent check.



## WMMA building blocks fully verified; the defect is glue-only (2026-09-12)

Three probes now cover every primitive the tile kernel uses, each against a CPU reference with exact
integer data:

| probe | pattern | result |
|---|---|---|
| `wmma-check` mode0/1 | 16x16 tiles, ldm=16, B col_major (QK^T) and row_major (PV) | 2e-6 |
| `wmma-check-ldm` | 16x256 tile, **col_major ldm=256** (the K path) | 0.0000, NaN 0/256 |
| `wmma-check-pv` | A=P(ldm=16) x B=V **row_major ldm=256** (the PV path) | 0.0000, NaN 0/256 |

Also excluded by inspection: `launch3_dyn` does call `hipFuncSetAttribute(MaxDynamicSharedMemorySize)`
(with a OnceLock cache), so the 53248 B request is legal; the buffer offsets (Q 0, K 32768, V 40960,
P 49152 with 4096 B exactly to 53248) do not overlap; the fragment index mapping
`idx = lane + 32*sl, row = idx>>4, col = idx&15` is what all three probes assume; and the softmax
slot bookkeeping re-derives correctly (`hc = lane>>4`, row = hc + 2*sl, xor-butterfly offsets 8..1
stay inside each 16-lane half, so `e_m[sl]` matches the accumulator's row).

So the multi-chunk NaN lives in the kernel's glue rather than in any verified primitive - most likely
a warp-divergent early return interacting with the per-key-tile `__syncthreads()` (the classic
deadlock/UB pattern; my final revision had removed the early return but the earlier ones did not, and
there is no committed revision to compare against). The next attempt should build the tile kernel
from these three probes outward rather than re-deriving the primitives.



## WMMA tile kernel: correct at last, and what it costs (2026-09-12)

**The NaN root cause, found by a synthetic check.** New `llm170 wmma-attn-check` runs the tile kernel
on a small single-head case (t=64, pos0=32, n_past=96, 6 segments, causal mask) against a CPU
reference, f32 accumulation, same part convention. First run localised it immediately: **m matched
(1.1632 vs 1.1630) but s read 1.0000 against 6.9076** - the kernel reduced the row *maximum* across
the 16 lanes but never reduced the row *sum*; each lane's `s` only ever saw its own key. For rows
whose keys are masked, `s` stayed 0, and the merge kernel's later `acc/s` turned that into NaN. Fix:
keep the individual exponentials for the P matrix and butterfly-sum a *copy* across the row's 16
lanes for `s`. The check then reports **0 mismatches, max|delta| = 0.0006** (f16 rounding), and the
same case with pos0/t/n_past as above - and a 600-token multi-chunk prompt through the model - both
produce identical tokens to the scalar path.

**And what it costs.** It is correct but currently *slower* than the scalar kernel:

| config | scalar (wk8) | WMMA tile |
|---|---|---|
| pp512 | 360.2 t/s | 344.7 |
| pp3314 | 316.9 | 288.6 (-9%) |

So the model path stays on wk8 (the launcher branch was reverted); the kernel, its registration and
the three probes stay as diagnostic assets. Where the 9% goes, in the order worth attacking:

1. **Per-key-tile f32->f16 staging** - 8192 elements converted and two `__syncthreads()` per 16-key
   tile, repeated for every segment a block visits. The fix is an f16 shadow KV written once when the
   KV is appended (the docs' earlier note that "the f16 conversion is only ~2 vector ops per
   (query,key)" was for the *staging count*, not for the *conversion inside the key loop*).
2. **2x redundant QK** - each half-warp computes the full QK^T because the scores are not shared
   between them; sharing them via shared memory halves that work.
3. **Occupancy 1** - 53 KB of dynamic shared per block caps the SM at one block; llama's RDNA tile
   config targets occupancy 3-8 with 128-256 threads.

This is now a *performance* problem with a verified-correct kernel, not a correctness problem - a
much better starting point than the five earlier attempts. The measured prize is still pp3314 ~370
t/s (1.10x llama).



## WMMA tile kernel: two optimizations close the gap to parity (2026-09-12)

| config | scalar (wk8) | WMMA (first correct) | WMMA (optimized) |
|---|---|---|---|
| pp512 | 360.3 | 344.7 (-4.4%) | **357.4 (-0.8%)** |
| pp3314 | 324.1 | 288.6 (-9%) | **321.8 (-0.7%)** |

Two changes, both verified by `wmma-attn-check` (still 0 mismatches, max|delta| 6e-4):

1. **QK halved**: each half-warp now computes 8 of the 16 mma over its own dims and the two partial
   score matrices are summed through shared memory (separate buffers per half, so no race) - the first
   draft's exchange, now with the row-sum bug fixed.
2. **Vectorized staging**: the per-key-tile f32->f16 conversion loads `float4` and writes four halves
   instead of one element per iteration, quartering the load instruction count (the convert count is
   fixed at 8192 per 16-key tile).

Remaining overhead, ~1% plus the occupancy item: the block still takes 61440 B of dynamic shared (Q
32 KB + K/V 16 KB + P 4 KB + exchange 8 KB), which caps it at one block per CU, and the per-key-tile
staging repeats for every segment. Removing the Q staging (loading Q fragments from an f16 Q shadow
instead) would free 32 KB and should unlock 2-4 blocks per CU - that is the next step, and with it
the tensor cores should finally pay off against the scalar butterfly (the 17% measured by skipping
it). Until then the model path stays on wk8, since a numerically different kernel at parity is not
worth adopting.

### Why the WMMA kernel is at parity rather than ahead (structural)

The two optimizations above removed the *instruction-count* overheads, but the kernel is still
latency-bound for a structural reason: a 64-row query tile with 256-dim points needs 32 KB of Q
staging in shared, plus 16 KB of K/V per 16-key tile plus P/S - 61 KB total, which caps the CU at one
block, i.e. 8 warps over 4 schedulers = 2 warps/scheduler. The mma's latency (~20-30 cycles) cannot
be hidden at that depth, so the tensor cores idle. Two back-of-envelope checks agree: at t=1536 the
attention's FLOPs are ~14.5 GFLOP, and at the WMMA peak this is ~3.6 ms total, while the measured
attention share is ~72 ms - about 20x off the peak.

The *block* assignment is what forces this. My kernel gives each block a **query tile** and loops it
over the whole KV, so every block re-stages the entire K/V (24 query blocks at t=1536 all restage
each segment). llama.cpp's tile config does the opposite: a block **owns a KV segment in shared and
streams query tiles through it**, staging each K/V element once. That is the redesign this kernel
needs - the current shape shares the *wrong* operand. f16 shadow KV/Q buffers (converting once per
layer per chunk instead of per block) are the second half of it, and together they are the path to
the measured 10.5% pp3314 prize rather than the ~1% left in the current shape.

Reference for that prize: llama-bench (build 8b4b3558f, ROCm, same machine, same GGUF) gives pp3314
335.06 vs our 299.7 - the deficit is long-context-specific (pp512 is 0.98x) and equals the attention
excess, which is why the attention is the right target and the GEMV/GEMM side is not.

Also worth recording for whoever resumes this: my earlier "the staging converts exceed the mma
counts" reasoning was wrong as a cost model (it compared instruction counts, not throughputs - one
mma is worth 16-32 cycles, one convert is one op). The staging is not the bottleneck; occupancy is.



## Flash-Next decode kernel composition today (2026-09-14)

KTRACE of a tg8 run, per step: 68 ms of kernels across 39 types. The q8_0 family is the
largest single block at 33.1 ms (49%), and its breakdown says it is not a target:

| n_out | total | calls | per call | effective |
|---|---|---|---|---|
| 10240 | 12.14 ms | 134 | 0.091 ms | ~305 GB/s |
| 2560 | 6.66 ms | 147 | 0.045 ms | - |
| 6144 | 3.93 ms | 36 | 0.109 ms | - |
| 65535 (head) | 3.69 ms | 1 | 3.69 ms | - |
| 12288 | 2.37 ms | 12 | 0.20 ms | - |
| 320/640/512 | 4.35 ms | 217 | 0.014-0.023 ms | launch-bound |

The 10240-output GEMMs run at ~305 GB/s, above the 236 GB/s DRAM figure, i.e. they are
already served largely from L2 (the shared expert's weights are re-read every step) - no
headroom there. The MoE gate/up sits at 180 GB/s of DRAM and the down projection is capped
by its Q5_1 layout, so the decode's large GEMMs are done. What remains is the long tail:
217 small GEMMs (n_out 320/512/640, the hyper-connection and routed-expert shapes) spend
4.35 ms at 14-23 us per call, which is launch and ramp cost rather than transfer, so
fusing them - not retuning them - is the lever. GDN's AR kernel is only 0.83 ms/step here,
not the 50 ms the older plan recorded, so plans/66 P2 is a prefill-targeted item on this
codebase.




## Flash-Next prefill: the remaining levers are kernel-interface work (2026-09-14)

Three candidate quick wins were measured at pp2048 (base 9,012-9,118 ms, 224-228 t/s)
and all came back neutral, so none is the lever:

| candidate | gate | result |
|---|---|---|
| Q4_K MMQ tile | LLM170_Q4K_MMQ=1 +Y=1 | 8,899-9,155 ms (neutral) |
| f16 fused dequant GEMM (q8_0) | LLM170_F16_ACC=1 | 8,968-9,168 ms (neutral), diverse bit-identical |
| selection-list buffer reuse | - | 0.70 vs 0.71 s (neutral, reverted) |

Phase timing with LLM170_Q4ACC_TIME across ~400 accelerator calls shows upload 0.2 s
(weights stay resident, as designed) and d2h 1.5 s, i.e. ~0.75 s per pp2048 chunk or 8% of
it spent copying GEMM outputs back to host. That is inherent to the stage API: run_prepared
allocates a fresh t*n_out f32 buffer, copies the device result into it and then scatters
rows into Vec<Vec<f32>>, for every call. Removing it means letting stages consume
device-resident buffers rather than row vectors - the same interface change that the QSA
stage needs, and the common root of both the 8% d2h and the 14% selection-list cost.

Where the 1.33x gap to llama.cpp (276.68 t/s at pp11750) actually sits: the QSA stage is
57% of our prefill (mm_group 38% and attn 37% of it, both accelerator work at prefill
shapes), and within that the QSA projection GEMMs run at roughly 1 TFLOP/s against the
19.5 TFLOPS the 27B's MMQ paths reach. That is a kernel-quality gap in batched grouped
GEMM at these shapes, not a dispatch or host problem, and it is the item that would move
the prefill.




## Attention geometry experiments are below the prefill measurement noise (2026-09-14)

Hypothesis: `_sel6`'s block is 4 tokens x 2 head-groups, and the two groups are on different
tokens, so they read different K/V rows and L1 cannot serve the second group - 24 heads / 6
per warp means each row is still read 4 times. A token-per-block variant (4 warps = 2 kvh x
2 groups, all 24 heads of one token) would put both groups of a kvh on the same rows and
halve the re-read. It was implemented (`q4_qsa_attn_sel6t`, bit-identical by construction -
the diverse stream reproduced exactly) and measured:

| geometry | pp2048 reps |
|---|---|
| `_sel6` (4 tokens x 2 groups) | 9,048.7 / 9,161.2 ms |
| `_sel6t` (1 token x 4 warps) | 9,152.6 / 8,782.5 ms |

The ranges overlap, so the L1 hypothesis is unresolved and the change was reverted rather
than kept on a guess. Two lessons for the next attempt: prefill runs vary by ~3% run to
run, so a ~2% effect needs more reps (or a KTRACE comparison, which reports device time
rather than the wall), and the geometry change is worth re-testing only with that protocol.


