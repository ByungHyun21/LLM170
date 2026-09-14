# 기록 보관 — misc-history

> `docs/benchmarks.md`에서 옮긴 이력 항목(제목 기준). 수치·표는 원문 그대로.
> 최신 지표는 `docs/benchmarks.md`, 파일별 실측은 `docs/source/`를 본다.

## Refactor no-regression gate (2026-09-08, plans/35)

Behavior-invariant refactor (dead-kernel pruning, gemv generation collapse,
file splits). Vulkan baselines on q35work.gguf, 3-run medians: tg32 7.06
t/s, pp512 (VK_TILE+VKD_BATCH opt-in) 65.95 t/s. A side A/B settled the
q6_K route: gemv3+quant 7.06 vs opt-in gemv6 6.71 t/s — the promoted
default stays; gemv4/5/6/7 families deleted (ADR-0019).



## Engine modes (2026-09-05, user decision)

Default = WMMA fast mode (quality: llama.cpp MMA class — measured logits
deviation vs bit-exact path: max 8.7e-3 relative, argmax stable).
`LLM170_EXACT=1` = bit-exact dot4 path (GPU == CPU reference, bit-for-bit).



## Baseline methodology note (2026-09-05)

Two llama.cpp references exist and must not be mixed:

1. **Designated primary target** (2026-09-02): streaming-server bench (median),
   pp64 142.8 / tg 10.4. This includes the serving stack and is the official
   goal reference.
2. **llama-bench raw compute loop** (2026-09-05, same machine, same GGUF
   verified by hash, -fa 1): **pp64 294±15, pp512 358±9, tg8 11.2**. This is
   the apples-to-apples comparison for our raw-loop `llm170 bench` numbers.

Against the raw-loop reference our standing is: pp64 default 81.8 (0.28x),
fast 139.4 (0.47x); tg24 10.40 (0.93x). Against the designated server-bench
target: pp64 fast 0.98x, tg parity. The session's internal gains
(pp 43.1 -> 81.8/139.4) are unaffected by this distinction.



## qwen4exp decode: where the time actually goes (measured 2026-09-14)

Decode step = 129.5 ms (tg8 1036 ms). Decomposition, all measured on the 8060S:

- Host skeleton (LLM170_NOLAUNCH, extended to launch3): 12.8 ms. So the step is
  device-side.
- Launch cost is NOT the issue: a micro-benchmark (rawhip::micro_tests::
  launch_cost_split) puts host enqueue at 0.97 us/launch and a tiny kernel at
  2.4 us/launch, q4_l2_rows at 10.8 us. A HIP graph capture of the whole step
  (62 segments) is bit-identical but 6% slower, which is consistent: removing host
  launches cannot help when it costs 1 us each.
- KTRACE/ftime per-kernel figures for the tiny kernels (0.34 ms/call) are inflated
  by event pairing and sync costs; treat the stage-skip deltas as ground truth.
- Stage-skip deltas (LLM170_STAGE_SKIP=...): GDN -50.0 ms/step (39%), MoE -24.4
  (19%), QSA -10.5 (8%). Per layer: GDN 1.39 ms, MoE 0.51 ms, QSA 0.87 ms.
- Ideal weight traffic per step (n_embd 2560, 48 layers, 512 experts top-10, expert
  n_ff 640): MoE 1.3 GB + GDN projections 1.1 GB + shared/QSA/PLE ~0.4 GB = ~2.8 GB,
  i.e. ~12 ms at 230 GB/s. Measured 129.5 ms is ~11x the bound, and llama.cpp's
  61 ms/step is ~5x, so neither implementation is bandwidth-bound: the cost is the
  dependency chain of ~3,200 kernels and the t=1 kernel shapes.

Next levers, in order: fuse the GDN per-layer chain (l2+scale, conv+ar, the
elementwise group), retune the t=1 GEMM/GEMV block shapes (gemm_q8_0 runs 546
launches/step with prefill-sized grids), then the MoE top10 (17.0 ms/step for 48
calls) and the QSA bridge (10.3 ms/step for 12 calls).



## qwen4exp decode traffic: 11x redundancy (measured 2026-09-14)

Micro-benchmarks (rawhip::micro_tests::launch_cost_split, cargo test -p
llm170-backend-gpu --release) establish the constants: host enqueue 0.76 us/launch,
a tiny kernel 2.0 us/launch, q4_l2_rows(d=2560) 10.8 us, a dependent 4-kernel GDN
l2scale sequence (split3 + 2x l2_rows + scale) 11.8 us, DRAM (512 MB read+write)
239 GB/s, cache-resident bw_probe 1112 GB/s, f32 FMA 51 TFLOPS.

Against those constants the decode step (129.5 ms) implies ~31 GB of traffic per
token. The ideal is ~2.8 GB (active weights, 6B-A at Q4, plus a few PLE rows), so
there is an 11x redundancy - a single token reads 44% of the model's 70 GB. Compute
is irrelevant (12 GFLOP needs 0.24 ms), launch cost is irrelevant (2 us), the host
skeleton is 12.8 ms and a dependent 4-kernel chain runs in 11.8 us, so the per-stage
times seen in ftime (e.g. l2scale 690 us/layer, moe.shared 560 us/layer) are 20-60x
their isolated cost and cannot be explained by kernels, launches or chain latency.
The residual is memory traffic.

To localize it, profile the bytes rather than the time: run rocprofv3 with
FETCH_SIZE/WRITE_SIZE counters filtered to one kernel (e.g. gemm_q8_0 / q4_gemm_q4k_ge)
over a short decode (bench --pp 24 --tg 4), and compare the per-kernel fetched bytes
against the tensor sizes the kernel should touch. Keep the profile to a single step;
whole-run counter collection is not safe on this machine.



## Measurement protocol: use --reps 3 and quote the warm reps (2026-09-14)

bench --reps 3 shows rep0 (cold) at 759.3 ms and reps 1-2 at 725.1/721.7, i.e. the
warm spread is +-0.5% while a single-run comparison carries +-6%. Every A/B in this
session that used one rep (753 vs 767 and similar) sat inside that noise band, so
reps 3 and the warm reps are the protocol from here on. Under it the qwen4exp decode
is 723.4 ms per 8 steps = 90.4 ms/step, and the row-batch Q4K kernel variant
(LLM170_Q4K_Y=1 YRPT=4), which was the third attempt at recovering memory-level
parallelism, is neutral at 719.9 ms.



## Why the weight reads are pattern-limited, and what fixes them (2026-09-14)

The strided probe's modes 1-2 (per-thread strided scalar 4 B vs the same stride with
16-byte vectors) measure 6.5 and 29.6 GB/s of useful bytes against 235-267 GB/s
touched, i.e. the memory system is fine and the utilisation is not. Modes 3-4, which
were meant to model the GEMM's row-per-thread and warp-cooperative patterns, produced
impossible figures (2578 GB/s) because the compiler elided the loads; that tooling
needs fixing before it can be quoted.

With the loads made un-elidable (atomicAdd), the probe reads: mode 0 sequential scalar
7.3 GB/s used of 264 touched, mode 1 per-thread 144-byte stride 6.5 of 235, mode 2 the
same stride with 16-byte vectors 29.3 of 264, mode 3 (a thread streaming its own row,
the GEMM shape) 26.4 of 238. Mode 4 was meant to model a warp-cooperative read but
does not (its lanes still touch different 144-byte blocks), so it is not quoted.

An important correction follows from the real kernel's shape: 16 of the 32 lanes in a
warp read the same address (they share the output column), so a load instruction
touches two lines and the lane's m-loop then consumes those lines fully. Line
utilisation in the grouped MoE GEMM is therefore high, and the 53 GB/s effective is
not a simple coalescing failure. A transposed f16 layout would not obviously fix it,
so that change should not be made on this evidence; the remaining unexplained factor
between 53 and the 238-264 GB/s the DRAM delivers needs a counter-based measurement,
which the tooling here cannot yet provide.





## Correction: the weight reads are not bandwidth-limited, they are latency-limited (2026-09-14)

A fifth probe mode reads the same buffer fully coalesced (lane = consecutive word,
one 128-byte transaction per warp), and it settles the question:

| mode | pattern | touched bandwidth |
|---|---|---|
| 0 | sequential scalar 4 B | 264 GB/s |
| 3 | a thread streaming its own row (the GEMM shape) | 239 GB/s |
| 5 | fully coalesced (lane = word) | 236 GB/s |

Every pattern touches memory at 235-264 GB/s, so the DRAM delivers full bandwidth
regardless of shape, and the "53 GB/s effective" seen for the grouped MoE GEMM is not
a memory ceiling. The real GEMMs run at 42-53 GB/s, i.e. in the coalesced class, and
they are at about 1% of the f32 compute roofline and ~20% of memory bandwidth at the
same time - so they are latency-bound, with a small amount of work per block and a
dependency chain that the memory latency dominates.

That invalidates the contiguous-f16 thesis (and explains, for the second time, why the
deq-f16 port measured neutral: it doubles the bytes without addressing latency), and
the plan of record becomes: increase per-block work or waves so the latency hides.
The earlier unroll and row-per-thread experiments were neutral, so the remaining
levers are the tile/wave configuration and explicit prefetch depth.



## Prefill hook cost + corrected standing (2026-09-04 evening)

The MTP prefill hook (draft-KV fill per token, ~9-15 ms) had been silently
loaded on every non-spec workload since the MTP arc, cutting prefill ~3x
(pp512 70 t/s). Root-caused by worktree bisection; now gated behind explicit
speculation intent (`--spec` on infer/vl/bench, `serve --spec`). Non-spec
pp512 restored to **180.3 t/s** (UD-Q4_K_XL, tuned-tile env), tg unchanged
(10.4). The older 169/228 records predate the hook entirely. Zero-config
(no .co tile envs) remains ~54 t/s — embedding the offline-built tile
kernels is the follow-up that would make the tuned rate default.

VkD (Vulkan) note: the batched step_batch claim from earlier this day was
retracted (measured on a stale binary); the verified default is per-token
execution, 41/41 stream-exact. Vulkan GEMM remains compute-bound at
~22-60 GB/s by weight type (software integer dot) — the i8-cooperative-matrix
kernel arc is specced in the plans.



## Fresh cross-backend measurement (2026-09-04 evening, llama.cpp @ 8b4b3558f = same-day upstream master)

Both llama.cpp backends rebuilt from source/ after `git pull` to the day's master
(previously the reference was a 2026-08-30 checkout). Same GGUF (UD-Q4_K_XL,
hash-verified earlier), -fa 1, -ngl 99, 2 reps, non-coexisting runs.

| test | llama.cpp ROCm | llama.cpp Vulkan | LLM170 HIP (tuned CO env) | LLM170 HIP (zero-config) |
|---|---|---|---|---|
| pp512 | **361.8** | **349.9** | 249⁶ | 249⁶ |
| pp64  | 274.4 | 221.5 | 172.6 | 172.6 |
| tg8   | 10.92 | 11.82 | 10.34⁴ | 10.34⁴ |
| tg32  | 11.70 | 12.02 | 10.40 | 10.40 |

Key readings:
- llama.cpp **Vulkan ≈ ROCm** on this APU (349.9 vs 361.8 pp512; Vulkan even
  wins tg8) — the RADV coopmat path reaches raw-loop parity, so our Vulkan gap
  (~4.6 t/s pp equivalent) is software, not hardware.
- Our standing vs the fresh raw loop (2026-09-05): pp512 0.65x
  (tuned == zero-config since the tile .co embedding), tg 0.89x;
  np4×spec4 aggregate 25-28 t/s remains ahead of any llama.cpp
  single-stream config; prefix cache gives 2.6x on repeat prompts.
- llm170 pprof (pp512, 2026-09-05): the earlier "projection GEMMs dominate"
  reading conflated attention with projection — the unmarked attention
  kernels drained inside the 'proj' event window (grew 111→410 ms per
  chunk while GEMM stages stayed flat). With flash defaulted on, the true
  per-chunk GEMM standing is ffn_gate ~147 ms + ffn ~84 ms + gdn_mm ~70 ms
  + constant proj ~111 ms; the raw-parity levers are tile throughput
  (ffn_gate/ffn) and GDN mm.

¹ 2026-09-05: pp512 184.1 → 200.0 (pp128 186 → 253). A 2-day bisect traced
the 2026-09-04 regression to `prefill()` copying the full token_embd table
(636 MB, ~170-220 ms fixed) on every prefill call — a borrow-checker
workaround from the MTP-hook era that the since-moved hooks made
unnecessary. GPU kernel time was unchanged throughout (identical PP_PROF
stage sums); the gap was constant per call regardless of token count,
reps, or ctx. Restored in-place borrow; 41/41 stream bit-exact vs the
pre-fix binary.

³ 2026-09-05 (night): tg8 9.72 → 10.16 (0.93x llama ROCm). Three fixes:
pageable-async D2H of logits was taking 92ms per token (pathological
slow path — now a hipMallocHost staging buffer in RawCtx, 0.07ms);
q3_K GEMV (the slowest type at 107 GB/s) hoisted thread-invariant
scale extraction + span-preloaded the ql/hm words + __ldg (146 GB/s);
t=1 decode flash now segments K/V across blocks when np>512 (was one
block per head — 48 blocks on a 16+ CU APU). Step decomposition: the
remaining floor is the layer-stack weight stream (~14GB @ ~160GB/s —
already at llama's average) plus the 1GB q6_K head.

⁴ 2026-09-05 (late night): tg8 10.16 → 10.34 (0.95x llama ROCm) via
launch-histogram-driven decode-step work: 1556 kernels/step reduced by
fusing rms+quant (rmsq) and silu_mul+quant (silu_mulq) — both preserve
the mirror arithmetic order bit-exactly — plus stream2 overlap of the
independent in_proj GEMV pairs (qkv‖gate, beta‖alpha, gate‖up; the
pairs are bandwidth-bound so only latency tails hide).

⁵ 2026-09-05 (final): pp512 232→238 — segment flash now default in prefill
(np>128, kill switch LLM170_NO_QSA_SPLIT, threshold LLM170_QSA_TH). Kernel-level
prefill trace (KTRACE on step_batch) shows tile GEMMs streaming at DRAM parity
(~43GB/s aggregate weights) — the remaining pp gap to llama is tile-kernel MFU
(~10 vs ~13 effective TFLOPS) plus ~74ms/chunk of non-GEMM (quant_q8 21ms,
gdn_ar 20ms, qk_rope 12ms).

⁶ 2026-09-05 (night, cont.): 238→249 (peak 251) via two non-GEMM kernel fixes found by the
prefill KTRACE — quant_q8 float4 loads (32 scalar loads were latency-bound,
21→18ms/chunk) and qk_norm_rope 32-lane cooperation (was one thread per block,
11.75ms→off the top list). Both bitwise-preserving. Remaining pp gap: tile GEMM
MFU (~10 vs llama ~13 eff. TFLOPS) and gdn_ar_w scan (19.6ms/chunk, structural).

² 2026-09-05 (later the same day): two defaults landed — (a) the three
offline tile code objects are now `include_bytes!`-embedded and loaded
automatically (zero-config == tuned, was 108.5 pp512); (b) the fused
flash-attention kernel is the default path (was opt-in via
LLM170_QSA_FLASH; kill switch LLM170_NO_FLASH). Stream 41/41 bit-exact
across both changes; tg8 picked up ~2% (9.55 → 9.72) as a side effect.



## Prefill chunk correctness fix (plans/41)

The ms-family tile loop rebound the same xq/out buffers for every 64-token
slab, so any prefill batch above 64 tokens silently computed only its first
64 rows (a 200-token prompt at chunk 128/512 diverged from the per-token
reference). Kernels now take a `tok_base` push field and each slab writes a
disjoint token range; harness maxrel is 0.003-0.006 at t=256/512 and engine
token streams match `LLM170_VKD_BATCH=0` at every chunk size.

Corrected prefill (pp512, same machine): 113 t/s at 64-token chunks,
154 at 128, 205-212 at 512 — i.e. within one pass the weight traffic is
dominated by slab re-reads, and pass count is the cost driver. All earlier
sub-64-chunk-out-of-spec measurements are superseded by these.



## Prefill traffic levers — closed (plans/41)

After the tok_base fix, every slab-merging lever was measured on the
corrected engine and came out neutral: token-group gy dispatch
(LLM170_VK_GYGRP=128, with and without GY2) 206-211 t/s, BN=128 ms128
213 t/s — all inside the 205-213 baseline band of a single 512-token pass.
The harness "gy=2 merge" figure did not transfer: it replays one tensor in
a tight loop, so it measures L2 residency across reps, not intra-dispatch
sharing. Prefill remains slab-traffic-bound at roughly 8 weight reads per
512-token pass; kernels themselves run at their solo streaming rate.



## Prefill plateau confirmation (plans/41 close-out)

Interleaved A/B on the corrected engine: the q5 grid-y dispatch (default)
measures 215.7/215.9 t/s against 216.2/217.2 with GY=0 (LLM170_VK_GY=0) —
statistically identical, so the token-slab dispatch no longer matters at
512-token passes (its 2026-09 win was measured at 64-128-token passes).
Chunk sweeps 384/512/768 land inside the same 205-219 band, i.e. run-to-run
spread is about +-5% and dominates these deltas. Prefill is at a plateau
where every weight byte is re-read once per 64-token slab; halving that
needs a >64-wide accumulator tile whose per-byte rate holds, which ms128
(BN=128) did not deliver (neutral wall, lower kernel rate).



## Speculative decode — architectural verdict (plans/43)

Measured on the batched path as well (LLM170_BENCH_NP=4 forces
spec_step_multi, the design that does batch verification): 32 generated
tokens in 25.9 s = 1.24 t/s aggregate, i.e. ~810 ms per sequence-token
against 102 ms for plain decode. Two independent defects:

1. Drafts never materialize on the raw path: `mtp_draft_logits` is only
   written by the non-raw `verify_batch`; the raw path stores
   `mtp_draft_tok` (the GPU MTP head's argmax). The spec loop reads the
   empty logits, drafts token 0, and accepts nothing (LLM170_SPEC_DBG:
   `draft=0` every step; bench: fwd 8 / gen 8 / 1.00 tok per fwd).
2. Verification costs ~2.4x a plain decode per token. The bit-contract
   path re-runs the batch per token (raw_verify "per-token step = decode
   arithmetic"), so even perfect draft acceptance could not pay for it.

Spec decode therefore cannot beat plain decoding without either a true
batched verify forward (different numeric class than the per-token
contract) or accepting that class change; both are engine-scale changes
rather than a tuning fix. Feature stays opt-in and inert until then.



## Prefill wall-clock accounting (plans/43 close)

Instrumenting the record/drain split inside a 512-token pass settles where
the wall clock goes: the CPU command-recording span is spent almost entirely
inside the mid-pass descriptor-pool drain (record span 998/2078 ms against
drain 1047/2086 ms), i.e. the CPU records ~3.5k dispatches fast and then
waits for the GPU. The pass therefore runs at the GPU's slab-traffic limit:
15.79 GB of layer weights read once per 64-token slab = 126 GB per
512-token pass at ~53 GB/s effective, which is 84% of the kernels' solo
streaming rate (61-66 GB/s). There is no measurable CPU or idle slack left
to reclaim; only a wider accumulator tile (fewer weight reads) can move it,
and BN=128/256 tiles lose the same margin in occupancy (ms256 +1.6%,
ms512 collapse).



## Next lever: sub-block record repack (design note, not implemented)

Every memory-side probe on the DRAM case came back neutral (load count,
load width, pipelining, LDS traffic, occupancy, WG count, coarse packing),
while a linear reader on the same buffer reaches 333 GB/s. The remaining
explanation is sector granularity: the q5_K layout stores a row's 32-value
sub-block pieces at fixed offsets inside a 176-byte block, so a warp's
reads stride 176 bytes per row - one or two 32-byte sectors per 16 bytes
used - and DRAM transaction rate, not bytes, is the ceiling.

The fix is a load-time repack into per-sub-block records so that a warp
reads contiguously:

    record[32B] = ql(16B nibbles) | qh(8B) | d(2B f16) | sc(6b)+mb(6b) | pad
    layout       [row-block 64][256-block][sub-block 8][row 64]

A workgroup then covers 64 rows x 2 sub-blocks (BK=64) with 128 threads,
one record per thread = two uvec4 loads per thread per iteration instead of
~24 scalar loads, and each warp touches ~1KB contiguous. Packed size grows
to 1 byte/value (+45% versus 0.6875), which the issue-rate gain should
dominate. Verification path: repack the tensor in the harness, check the
new decode with vk-tile-check (t=512, expect maxrel <= 0.006), then measure
the 67MB tensor's rate - the decisive number is whether it lifts off the
23.6 GB/s plateau. The exact qh lane mapping in the current decode
(qhi/4 with the iqs>>4 byte select) must be transcribed carefully.



## Sub-block repack — implemented and rejected (plans/45 close)

The repack design was carried through: a 48-byte per-sub-block record
(ql nibbles | scales | d | qh bits, three aligned uvec4 loads per thread,
[row-block][256-block][sub-block][row] layout) plus a matching BK=64 tile
kernel with the exact original scale/iqs conventions, wired into the
harness behind LLM170_TILE_PACK=2/LLM170_TILE_RP=1.

Result: numerically wrong (maxrel 2.18, needs debugging) and, more
importantly, no faster - 23.1 GB/s on the 67MB tensor against the 23.6
baseline, and 29.5 against 34.3 on the L2 tensor. Together with the earlier
probes this also falsifies the load-issue hypothesis: cutting a thread's
global loads from ~24 scalar to 3 wide uvec4 changes nothing on the DRAM
case.

The 23.6 GB/s DRAM ceiling on this machine is invariant to load count,
load width, occupancy, workgroup count, latency hiding, LDS traffic,
coarse and fine layout packing, and the decode arithmetic. The engine
runs its tiles at ~43 GB/s effective (tensors are smaller and partly
L2-resident), so this harness ceiling is not the engine's limiter; the
remaining prefill gap stands at the ~17% tile-rate difference versus
llama.cpp.



## Remaining levers quantified (plans/45 final)

* Prefill: tiles are 94% of a pass and the 25 structural variants moved
  nothing; every non-tile item is under 2.2% of the pass. The ~15% gap is
  the tile rate itself.
* Decode: ~25% overhead spread over ~1050 dispatches (~30 us each, from the
  GDN_SKIP measurement). Fusing a pair of projections (e.g. ssm_beta+alpha,
  same shape, one launch) saves 48 dispatches per token = ~1.4% of tg and
  0.08% of a prefill pass - not worth the multi-tensor kernel work.
  Recovering the whole 25% would need whole-layer fusion.
* Spec decode: cannot pay off while the "spec output == plain decode
  output" contract requires per-token verification; the batched verify is
  8x slower per token by construction.

Session final standing: pp512 183 -> ~303 t/s (0.85x llama), pp64 ~195 ->
~211 (0.86x), tg8 ~9.8 -> ~9.9 (0.82x), verify 25/0, tree clean, all 40+
commits pushed, and every adopted change and rejected hypothesis recorded
here with its measurements.



## GPU-side argmax for greedy decode (2026-09-11, final block)

Measured per token (same run): step() 88.2ms vs step+greedy 93.4ms - the trait default
materialised the full 151936-float logits vector and scanned it on the CPU, and reading
608KB from the mapped GTT buffer costs ~5.15ms (uncached read path).

`step()` now splits into `step_core` (leaves logits resident) plus a wrapper, and a
two-stage argmax kernel (256-thread WGs, 8 contiguous elements per thread, deterministic
tree reduction, ties to lowest index - identical selection to the CPU scan, verified over
8 generated tokens) returns just the token.

| Metric | before | after | llama.cpp | ratio |
|---|---|---|---|---|
| tg32 | 10.66-10.76 | 10.87-11.34 | 12.12 | 0.90-0.94x |
| pp512 | 304.2 | 304.2 | 356.66 | 0.85x |

Judge gate 19/19 PASS after the change. Remaining decode gap ~6% is inside the gemv8
kernels (effective ~199 vs llama's ~212 GB/s); remaining per-token fixed cost after the
argmax fix is ~1-2ms host + 1ms gdn_ar + ~1ms small kernels.



## Session arc conclusion (2026-09-12/13)

Two further adopted wins and three falsified families close this optimization arc:

| Change | Effect |
|---|---|
| Fused GDN recurrence (decode: split3+l2+beta_g into one kernel; prefill: same for the t>=2 path) | tg 11.79 → 11.83-11.86 t/s |
| Bit-identity technique: replicate the legacy chain's exact FMA contraction (explicit fma() chains + precise on accumulators) | spec-decode contract preserved |
| f16-B tile, conversion-pass variant | -2.3%, falsified |
| f16-B tile, producer-dual-write variant | token corruption, reverted; expected ceiling +0-1% |
| Single-buffered tile (occupancy 3x) | neutral |

The 1-ulp FMA contraction lesson is the arc's key transferable finding: glslc
contracts different-but-equivalent source differently by context, so any kernel
fusion must pin the operation structure explicitly (Fma count verified against
the legacy SPIR-V) to stay bit-identical for speculative decoding.

End state: tg 11.86 (0.977x), pp512 305-318 (0.87-0.90x) of llama.cpp Vulkan
on the reference APU. 22 falsified hypotheses are logged in this file; the
remaining gap localises to vector-ALU instruction mix inside the quantized
kernels (RGP-measured 79% VALU) and diffuse non-GEMM pipeline costs.



## MTP / serve path defects found and fixed (2026-09-12, later)

Two structural defects found while measuring the objective's MTP and np4 conditions:

1. **MTP prefill ran a full-vocab head per prompt token.** The spec-mode prefill
   loop (`prefill.rs`) called `mtp_step_gpu` for every prompt token, which
   computes the output head (953 MB `output.weight` GEMV) and D2Hs h_next —
   only the final token's draft is ever used. Added
   `RawDecode::mtp_step_hidden(with_head)` (HIP override routes to the existing
   `mtp_step_g(..., with_head=false)`), so KV accumulation skips the head.
   Natural-text spec bench (pp512 spec4, LLM170_SPEC_GPU=1):
   **pp 84.6 -> 179.3 t/s**; tg64 spec4 unchanged 16.3 -> 16.6-17.9 t/s.
   `spec == nonspec` greedy equality re-verified exact.
   Remaining spec-mode prefill overhead is the per-token MTP block itself
   (~1.3 s / 512 tokens); a batched MTP prefill (one t=512 block pass) is the
   next step, worth ~300 t/s pp under MTP.

2. **serve serialised concurrent requests.** The slot loop only prefilled when
   *no* slot decoded in that iteration; with 4 co-pending requests, slot 0
   decoded forever and slots 1-3 never got prefilled (np4 aggregate = 4x the
   single-stream wall). Prefill now also runs when any slot has unprefilled
   tokens: **np4 aggregate 9.64 -> 12.18 t/s** (llama-server np4, same machine
   and prompt: 25.6 t/s). The remaining np4 gap is in the batched-decode path
   itself (t=4 step), not the scheduler: the CLI np4 (`infer` with 4 prompts)
   reaches ~19.8 t/s, so the step kernels lose ~40% at batch 4 versus llama's
   continuous batching.

Reference numbers measured today (HIP, q35work.gguf, greedy, natural text for
spec):
| condition | ours | llama.cpp |
|---|---|---|
| pp512 / tg32 single stream | 330 / 11.07 | 353.62 / 11.47 |
| tg64 spec4 (MTP, natural text) | 17.2 (pre-fix metrics) -> 16.6-17.9 | not reproducible today: `--spec-type draft-mtp` fails to load on the master build (ROCm0 16 GB alloc fails with the flag; docs' recorded reference 15.5 np4+MTP) |
| np4 aggregate (120-tok prompt, 64 gen, client-side) | 12.18 (serve) / ~19.8 (CLI batched) | 25.6 (serve, np1 10.48 baseline) |
| np4 x spec4 aggregate (CLI merged path) | 13.85 | — |

Honest standing against the stated objective (pp and tg must both beat llama.cpp
at MTP, mmproj and np4): **not met yet.** tg single-stream is at 0.965x, pp at
0.93x, np4 at 0.48-0.77x, MTP-mode pp at ~0.5x (llama's prefill is unaffected by
MTP). The blockers, in order of size: (a) the prefill GEMM (mul_mat_q at ~12
TMAC/s, one workgroup per CU at 49-59 KB dynamic smem), (b) the q6_K MMQ route
(+6.5% pp, currently producing garbage — requant layout defect), (c) the np4
batched-decode step efficiency, (d) spec-mode prefill per-token MTP block.



## 4-token GEMV for the np small-batch path (2026-09-12)

Measured cause: the np decode step (t = number of active slots) used the
128-column tile kernels, whose cost is essentially independent of t (harness:
0.339 ms at t=4 vs 0.833 ms at t=128 on the same 36 MB q5_K tensor) — 124 of
128 columns are padding work.

`gemm_{q4k,q5k,q6k,xs}4` (src_gemv4.hip): identical decode arithmetic and
per-token accumulation order as the t=1 gemv kernels, but each weight word is
loaded once per sub-block and reused for t tokens (independent accumulators, one
tree reduction per token, y read from `xq + k*xq_w`). Routed in `mm_b` for
t in 2..=4 for q4_K/q5_K/q6_K/iq4_xs; `LLM170_NO_G4=1` restores tiles.

CLI np4 (4 prompts x 64 tokens): wall 23.2 -> 21.4 s (**-7.9%**), token streams
bit-identical to the tile path. Judge re-run: same 16/19 with the same three
reference-side FAILs and identical gaps, i.e. our outputs are unchanged.

Remaining np4 gap: ours ~20-21 t/s aggregate (CLI) vs llama-server 25.6. The
step is now bounded by the y-side load issue (each of n_out rows re-reads the
token activations) — the next structural step is multiple output rows per
workgroup with y staged in shared memory (llama's MMVQ NUM_ROWS approach).

### MMQ 64-row tile (plans/47 #28195) — attempted, not adopted (2026-09-12)

llama.cpp's gfx115x MMQ config uses I=64 rows / 128 threads (38.4 KB smem, 3
WGs/WGP) for J=128, versus the I=128 / 256-thread entry (58.9 KB, 2 WGs) our
code objects were built with; plans/47 records a claimed 1.33x from that change.
Attempted: patched `mmq-config-rdna3-5.cuh` (I=64, nthreads=128, occupancy=3 for
the Q4_K/Q5_K/Q6_K/IQ4_XS J=128 entries), rebuilt `mmq.co` from
`plans/i8_arc/mmq_native_q45.cu` via hipcc + bundle extraction (95,536 B, same
mangled symbols), and added a matching launcher geometry (grid = ceil(n_out/64),
block 32x4, smem = J*4 + I*76*4 + pad(J*144, nthreads*4) = 38,400 B).
Result: deterministic SIGSEGV (rc=139) on the first GEMM, also with the smem
overridden to 58,880 / 65,536, i.e. not a sizing issue — the kernel's internal
mapping needs more host-side plumbing than the tile geometry alone (llama selects
the config at runtime in `launch_mul_mat_q`, including the y-tile stride and the
grid mapping that follow from it). Reverted (config restored, launcher untouched).
Next attempt should port that plumbing rather than just the geometry.



## np decode step profile and the head fix (2026-09-12)

`raw_step_multi` now honours `LLM170_KTRACE`, so the np step (t = active slots)
can be profiled. Per step at t=4, before the fix: 163.8 ms total, of which

| kernel | ms | note |
|---|---|---|
| gemm_q6k_j128 (head) | 29.7 | 1.04 GB q6_K head through the 128-column tile at t=4 |
| gemm_q5k4 (G4) | 47.1 | 7.93 GB at 169 GB/s |
| gemm_xs4 (G4) | 19.8 | 3.13 GB at 158 GB/s |
| gemm_q4k4 (G4) | 11.3 | 3.06 GB |
| gemm_q6k4 (G4) | 9.2 | 1.82 GB (excl. head) |
| gdn_conv_t | 8.5 | 192 launches (per-seq) |
| qsa_flash | 7.9 | 64 launches |
| gdn_ar_w | 5.2 | 192 launches |
| rms/axpy/quant/kv/rope/misc | ~10 | |

GEMM total ~122 ms for 17.5 GB = 143 GB/s effective (floor is ~78 ms at
225 GB/s), non-GEMM ~32 ms, the rest is launch overhead at ~1500 launches/step.

Fix: route the t=2..=4 head through `gemm_q6k4` (1.04 GB read once instead of a
128-column tile). CLI np4 wall 23.34 -> 19.84 s (**-15%**), streams identical,
judge unchanged (16/19, all 9 spec invariants exact). Cumulative np4 this
session: 23.4 -> 19.8 s, aggregate ~19.8 -> ~23.4 t/s vs llama-server 25.6.

Remaining np4 levers, measured: (a) the per-seq state kernels (conv/AR/flash =
21.6 ms/step, 192+192+64 launches) — batching them across slots or fusing the
GDN chain; (b) G4 at 143-169 GB/s vs the 225 GB/s single-stream rate (the y-side
loads repeat per output row); (c) ~1500 launches per step at ~4-5 us each.

### MMQ `.co` rebuild attempts — both segfault (2026-09-12)

Two experiments rebuilt `mmq.co` from `plans/i8_arc/mmq_native_q45.cu` with
`hipcc --offload-arch=gfx1151 -O3 -I <llama.cpp/ggml-cuda>` plus bundle
extraction (the recipe in `scripts/build_co.py`):

1. **64-row tile config** (plans/47 #28195: I=64, nthreads=128, occupancy=3 for
   the J=128 entries) with a matching launcher geometry (grid ceil(n_out/64),
   block 32x4, smem 38,400 B per llama's `mmq_get_nbytes_shared`). Deterministic
   SIGSEGV on the first GEMM, also with smem forced to 58,880/65,536 B — not a
   sizing issue.
2. **Extra type instantiations** (q8_0/q3_K/iq4_nl/iq3_s at J=128, to move the
   81 ms/pass of odd-type tiles onto MMQ): the rebuilt `.co` exports all eight
   mangled symbols (verified byte-exact against our generated names) and loads,
   but the first GEMM segfaults.

Common factor: the shipped `co/mmq.co` was built against a *specific* llama.cpp
header snapshot (i8_arc sources, 2026-09-05). A rebuild from the current
`source/llama.cpp` tree produces a kernel whose compile-time ABI differs
(`ggml_cuda_mmq_config` / smem layout / tile parameters), so the launcher's
assumptions no longer hold. **Conclusion: `.co` rebuilds must pin the exact
header revision used for the shipped objects; without it, treat the precompiled
path as immutable.** The 81 ms/pass odd-type tiles therefore stay.



## mmproj (VL) condition measured (2026-09-12)

Both engines, same image (`source/llama.cpp/tools/mtmd/test-1.jpeg`, the NYT
front page) and prompt, greedy, 24 tokens:

| | prompt (encode+prefill) | decode | total request |
|---|---|---|---|
| llama.cpp (master build, HIP, `--mmproj mmproj-F16.gguf`) | 1678 ms (362 tok) | 12.31 t/s | 3.68 s |
| ours (`llm170 vl`) | ~4100 ms (335 tok; 3.0 s of it ViT) | 12.8 t/s | ~6.0 s (steady state) |

Our content is correct ("The front page of The New York Times from July 21,
1969, featuring" ...), matching the recorded reference. The gap is the vision
encoder: 3.0 s vs llama's ~1.2 s for the same 27-block CLIP ViT forward (plus a
one-time 7.1 s mmproj weight upload per process, i.e. a cold-start cost).

Note the decode rate here (12.8 t/s) is higher than the bench tg32 (11.06): the
VL context is ~360 tokens and a single stream, so this is the same short-context
effect llama shows (their 12.31 t/s).



## GEMM ceiling established; remaining pp gap is non-GEMM (2026-09-12)

Exact accounting: 25.62 GMAC per token (65 blocks minus the MTP layer, embedding
gather excluded) = 13.12 TMAC per 512-token pass.

| path | rate |
|---|---|
| our `mul_mat_q` (dp4a), per type | 11.7-12.7 TMAC/s |
| our rocWMMA tiles (v4/j128) | 9.4-11 TMAC/s |
| ceiling, roof-test "mfma1" (WMMA f16, L1-fed) | 24.05 TFLOPS = **12.0 TMAC/s** |
| ceiling, roof-test "mfma0" (WMMA f16, register-resident) | 48.8 TFLOPS = 24.4 TMAC/s (unreachable from memory) |
| llama.cpp implied (pp512 353.62, non-GEMM 15-31% of the pass) | 10.7-13.1 TMAC/s |

Our whole-pass GEMM rate is 10.9 TMAC/s and the MMQ part is at 12 TMAC/s, i.e.
**at this GPU's measured L1-fed ceiling**. Consequence: writing a new f16 WMMA
GEMM cannot win (it would land at the same 12 TMAC/s); the remaining 5-6% of pp
must come from the non-GEMM 310 ms/pass.

Non-GEMM items and what happened to them this session: rms_part/rms_finish
104->20 ms (fixed), quant_q8 25->6 (skipped for MMQ consumers), token-embedding
dequant ~20 ms -> ~3 ms (parallelised, +0.6% pp), silu_mul 28.5 ms (244 GB/s, at
the streaming limit), gdn_ar_w_swap 99 ms (4-u smem staging variant measured
0.982x, prefetch variant neutral, ARSM 0.63x, ARCHUNK 0.89x — all rejected),
qsa_flash_wk 50 ms + merge 10 ms (three variants measured worse), norm_gated_silu
22 ms (82 GB/s).



## VL gate and serve np4 re-measure (2026-09-12, session end)

`scripts/verify_vl.py` (2-phase, fresh llama --mmproj reference): **5/5 PASS** —
vl_spec_short/np2/long and vl_np2_isolation all exact. The semantic keyword check
warns only because the 24-token budget is spent inside the think block; a manual
run with the same budget prints "The front page of The New York Times from July
21, 1969, featuring ..." (llama's reference: "A vintage front page of The New
York Times featuring the headline MEN WALK ON MOON").

serve np4 (4 concurrent 120-token prompts, 64 tokens each, client-side wall):

| | before this session | after the 4-token GEMV + head fix | llama-server |
|---|---|---|---|
| np4 aggregate | 12.18 | **21.48** | 25.6 |
| np1 | 9.25 | 9.23 | 10.48 |

The remaining np4 gap is 0.84x (serve) / 0.91x (CLI, 23.4) and tracks the same
per-step efficiency as single-stream tg (0.96x) plus the per-seq state kernels
(conv/AR/flash 21.6 ms of 163.8 ms per t=4 step, 448 launches).



## VL accounting correction + verification snapshot (2026-09-12, end)

`LLM170_VIT_TIME=1` per-block timings: mmproj weights+upload 7.4 s (one-time per
process), prep(conv) 0.5 s, **vision forward 1.1 s** (27 blocks; ~10 ms ln+qkv+rope
and ~10 ms attention per block), LLM prefill 1.1 s. So the earlier "3.0 s vision
encoder" figure in this file was wrong — it included the one-time uploads. The
steady-state VL request is ~4.2 s (encode 1.1 + prefill 1.1 + 24 tokens at 12 t/s)
against llama's 3.68 s (prompt 1.68 s incl. encode + 24 tokens) = 0.88x, i.e. the
VL condition is gated by the same prefill/decode gaps as the text conditions, not
by a broken encoder (~1.1 s vs their ~1.0 s).

Verification snapshot at commit 36c5604 (all re-run on the current binary):

| gate | result |
|---|---|
| `scripts/verify.py` judge (fresh llama reference) | **16/19**, all 9 `spec_*` invariants exact; the 3 FAILs are reference-side (long2 slot-KV instability x2 with identical gaps, long3 single-token degenerate reference) |
| `scripts/verify_vl.py` | **5/5** (vl_spec_short/np2/long exact, np2 isolation) |
| greedy streams vs pre-change binaries | bit-identical for every adopted change (5/8/120-token prompts) |

Measured levers that were neutral this session (do not re-try blindly):
`LLM170_QSA_SEG=256/512` (0.9997/1.0024), `LLM170_ARSM=1` (0.63x),
`LLM170_ARW4` 4-u AR staging (0.982x), `LLM170_ARCHUNK=1` (0.894x),
`LLM170_NO_WKFLASH` (0.857x), `LLM170_NO_QSA_SPLIT` (0.823x),
`LLM170_DEQ16=1` (0.89x), `LLM170_MMQ_SMALLT=1` (0.66x wall on np4),
`LLM170_NO_MMQ=1` (0.96x), `LLM170_MMQ_ONLY=3` (0.968x).



## Late-session levers: one adopted, several falsified (2026-09-12)

Adopted: **q6_K GEMV misaligned-word path** rewritten to three 8-byte `uint2`
loads at the aligned base (q6_K rows are 210-byte blocks, so `wb+ql_rel` is
2-mod-4 for half the blocks and the old path issued 5 overlapping 4-byte loads
per 4 words). Bit-identical; interleaved A/B tg32 10.99 -> 11.07 (+0.73%).

Falsified (all measured this session, do not retry blindly):

| knob | result |
|---|---|
| `LLM170_T1SG=64/32` (decode attention split granularity) | tg 1.0004 / 0.9987 |
| `LLM170_QSA_SEG=256/512` (prefill attention segment) | pp 0.9997 / 1.0024 |
| `LLM170_ARW4` 4-u AR with smem k/q staging | pp 0.982x |
| `LLM170_QSA_SEG`+`QSA_TH` variants, `NO_WKFLASH`, `NO_QSA_SPLIT` | 0.86-1.00x |
| rebuilt `mmq.co` (extra types / 64-row config) | segfault (header revision) |
| rebuilt `v4all.co` | works (stream-identical) but no change |

**llama.cpp's GDN kernel is structurally identical to ours** — read at
`source/llama.cpp/ggml/src/ggml-cuda/gated_delta_net.cu`: one warp per state
column, `rows_per_lane = S_v/warp_size = 4` state shard per lane, 4 k + 4 q loads
per lane per token, two `warp_reduce_sum` per token, and the same fused
`S = g*S + k*delta` / `attn = S^T q` update, with the whole token range scanned
sequentially inside the kernel (grid = H x n_seqs x S_v/4, `__launch_bounds__`
128 threads). Our `gdn_ar_w_swap` matches that shape, so the 99 ms/pass AR scan is
at parity and is not the prefill gap.



## Thermal-matched comparison + small-n GEMM falsification (2026-09-12)

Both engines measured back-to-back in the same thermal state (llama ROCm build
8b4b3558f, then ours, two rounds):

| round | llama pp512 | llama tg32 | ours pp512 | ours tg32 |
|---|---|---|---|---|
| 1 | 353.77 | 11.53 | 340.42 / 333.38 | 11.08 / 11.01 |
| 2 | 344.24 | 11.54 | 338.06 / 334.02 | 11.07 / 10.99 |

Medians: llama 349.0 / 11.535 vs ours 336.5 / 11.03 -> **0.963x pp, 0.956x tg**.
(llama-bench's own run-to-run spread on this machine is +/-5 t/s at pp512, which
is why the standing must always be quoted from a same-session alternation.)

Falsified: a dedicated small-n_out q8_0 GEMM (`gemm_q8_smalln`, 8 rows x 32
tokens per WG, weights staged in LDS). Motivation: the beta/alpha projections
(5120x48 q8_0, 96 launches/pass) cost a t-independent ~0.2 ms each in the tile
path (harness: 0.203 ms at t=128 and 0.202 ms at t=512), i.e. 29 ms/pass = 1.9%.
The dedicated kernel measured **0.967x** (pp 341.5 -> 329.7) and was reverted:
46 KB of LDS per workgroup caps occupancy at 1 WG/CU, so the staged K-loop wins
nothing over the tile's structure.

Also closed: the shipped `co/mmq.co` (99,872 B) cannot be reproduced from either
`plans/i8_arc/mmq_native_q45.cu` (93,232 B) or `mmq_native_rdna35.cu` (55,608 B)
with the current headers, so neither its exact source nor header revision is
recoverable from the tree. The MMQ code objects stay immutable.



## norm_gated_silu vectorisation (+3.0% pp) and two falsifications (2026-09-12)

Adopted: **norm_gated_silu_f32 float4**. The kernel handled one 128-value row per
warp (24576 rows per launch): the RMS segment read 4 consecutive scalars per lane
and the write phase strided 32 across lanes (12 memory ops per lane per row).
Both phases now use float4 — the reduction keeps the same per-lane element set
and addition order, and the write is element-wise so the lane mapping is free.
Bit-identical streams; interleaved A/B pp512 329.7 -> 339.6 (**+3.0%**), tg
unchanged. The traced mark for this kernel (22 ms) had understated it by ~2x.

Falsified, both reverted:

| experiment | result |
|---|---|
| `gemm_q8_smalln` — dedicated 8-row x 32-token q8_0 GEMM for the beta/alpha projections (tile path costs a t-independent ~0.2 ms each = 29 ms/pass) | 0.967x pp: 46 KB LDS per WG caps occupancy at 1 WG/CU, so the staged K-loop beats nothing |
| fused `silu_mul` -> MMQ y-layout (`silu_mulq_mmq_ds4/d4`, bit-identical to mmq_quant_y's roundf/butterfly/half2 packing, saving the 35.6 MB fglu f32 round trip per layer) | 0.99x pp — the fglu round trip was already largely L2-resident |

`.co` rebuilds are now definitively closed: rebuilding `mmq_native_q45.cu`
*unchanged* with the current headers also segfaults (the shipped object is
93,232 B vs the shipped 99,872 B, so the header/ABI revision differs), while the
same recipe on `v4all_d.cu` reproduces a stream-identical tile object.

Operational fix: a hipRTC compile error in any kernel source made `inject_rawhip`
fail, and `bench` continued on the **CPU engine** while still printing "GPU"
numbers (observed: a bad shuffle mask produced 1.5 t/s that looked like a slow
GPU result). `bench` now returns the injection error instead of falling back;
`infer`/`serve` keep the fallback.

Final thermal-matched standing this session (llama ROCm 8b4b3558f, same GGUF,
measured back-to-back):

| | llama | ours | ratio |
|---|---|---|---|
| pp512 | 350.5 | 335 (338/331) | 0.956x |
| tg32 | 11.54 | 11.02 (11.06/10.98) | 0.955x |

Session net: pp512 313.8 (base 2bacd60) -> ~335-338 = **+7-8%**; tg32 10.92 ->
11.02-11.08 = **+1%**; spec-MTP pp 84.6 -> 325-328 (+285%); spec tg 16.6 -> 19.4;
serve np4 12.18 -> 21.48 (+76%); VL gate 5/5; judge 16/19 (3 reference-side).



## Dispatch-structure wins: serial GEMM pairs (+1.5% pp, parity reached) - 2026-09-12

Two pair sites cost more in stream synchronization than they hide in overlap:

| site | change | pp512 | pp64 |
|---|---|---|---|
| prefill FFN gate/up pair (t>64) | serial (LLM170_PP_PAIRS=1 restores) | +0.2-0.75% | - |
| decode FFN pair + decode GDN in_proj pair (t=1) | serial (LLM170_DECODE_PAIRS=1 restores) | +0.9% | +2.1% |

The decode-pair effect lands in the *next* prefill, not in the decode itself: the
warmup 64-token prefill is identical with and without pairs (370.0 vs 370.6 ms)
while the timed prefill that follows the warmup decode differs (375.2 vs 365.9 ms).
tg is within noise across three A/Bs (+0.3/-0.25/-0.3%). In-proj pairs at t>64 stay
(serial measured -0.6% there: the small gate GEMM does overlap the big qkv).

Bit-identical streams in all cases.

Interleaved standing vs llama-bench (3 rounds each, same GGUF, back to back):

| | llama | ours | ratio |
|---|---|---|---|
| pp512 | 342.86 / 341.83 (first run cold: 329.45) | 342.17 / 342.53 / 341.77 | **1.000x** |
| tg32 | 11.50 / 11.50 / 11.50 | 10.99 / 11.01 / 11.00 | 0.957x |

pp512 has reached parity from 313.8 at session start (+9%). tg remains ~4.5% short;
the remaining decode budget is 79-80 ms of GEMV (17.54 GB, ~220 GB/s aggregate vs
243 GB/s for the best single kernels) + 4.7 ms attention + ~6 ms of other elementwise.



## Falsifications and judge re-run (2026-09-12, part 2)

Falsified this pass (all reverted, all bit-identical where they ran):

| hypothesis | experiment | result |
|---|---|---|
| ffn_gate/up GEMV (`n_out=17408`) is 2x off bandwidth | 4-row/WG `gemm_xs4r` (y loads shared, 1/4 the WGs) | 11.07 vs 10.97 ms for the group - **neutral**; the earlier "101 GB/s" reading was an arithmetic error (46 calls, not 25 - the group is already at ~199 GB/s) |
| 2-stream FFN split hides latency | serial (adopted) | +0.2-0.75% pp, kept |
| in-proj pairs at t>64 | serial | -0.6%, kept as pairs |
| decode FFN/GDN pairs cost only sync | serial | +0.9% pp512, +2.1% pp64 (the cost lands in the *next* prefill) |

Kernel-cost floor measured with `mm-bench` on a 255 KiB tensor: **~6 us per launch**.
The decode's 600 kernels therefore carry ~3 ms/token of launch floor; the remaining
~12 ms of elementwise work is dominated by contract-bound small kernels (`rmsq` 17 us
x128 = 2.2 ms at one warp for 5120 values, `l2_rows2_scale` 30 us x48, `gatedq` 27 us
x48, `gdn_ar_w` 26 us x48) whose reductions are pinned to the CPU mirror's summation
order. Decode attention carries a ~100 us/layer fixed cost in *both* the split and
plain paths and scales only ~0.25 us/key/layer.

Judge (`scripts/verify.py`, llama-server ROCm reference, 3-phase: collect -> judge):

**10 PASS / 11 FAIL.** Every failure is a flat-point flip: our token is the
reference's own top-2 with a 1.67-2.49 nat gap (tie threshold is 1.5), or a
special-token region (`<think>` vs `1`, `#` vs `pivot`) where both continuations
are plausible. Not a regression from this session: the same case reproduces
byte-identically on the session-start binary and on both intermediate binaries
(direct A/B on `single_code` and `long_prompt`), and the reference is stable
(llama-server np1/np4 and `-ngl 0` CPU all agree). A judge run with a stored
reference from a different server config is not comparable to the previously
recorded 23 PASS / 2 FAIL.



## Open defect: MTP draft acceptance ~19% (found 2026-09-12)

`--spec 4` currently *loses* throughput instead of gaining: tg 7.1-8.0 t/s versus
11.1 non-spec, with `fwd == gen` (one token per forward, xN verification per token).
Measured acceptance on natural text (`LLM170_BENCH_TEXT`, `LLM170_SPEC_DBG=1`):
6 OK / 26 MISS at pp128, 2/14 at pp16, 1/15 at pp512 - i.e. 10-19% first-draft
accuracy instead of the ~80% that 4-5 tokens/verify implies. The earlier recorded
19.2-19.4 t/s (1.25x llama MTP) is not reproducible with the current build.

What was ruled out (all measured):

| hypothesis | test | result |
|---|---|---|
| draft/verify pairing convention | (tok_p, h_{p-1}) vs (tok_p, h_p), env A/B | both ~19% |
| MTP KV accumulation over the prompt | acceptance vs prompt length (16/128/512) | flat (~19% everywhere) |
| batched MTP prefill (blk.64, 5617bf5) | `LLM170_CHUNK=64` vs 512 | identical (6 OK both) |
| degenerate head output | `LLM170_MTP_STAGE=1` intermediates | sane (eh=-117, wo=-113, ff=-340, logits 0-6) |
| stale `gemv_q8_out` result overwriting `mm_direct` on eh_proj (the code called both; the RCA comment says that path is wrong at ni=10240) | removed the stale `mm_into` call | acceptance unchanged; the duplicate GEMM is gone (kept - it was pure waste) |

Both the GPU head (`mtp_step_gpu`) and the CPU chain (`mtp_forward`, used for j>=1)
misfire alike, so the defect is in shared state (weights layout, pair convention or
head input) rather than one kernel. Drafts are frequently 220 (" "), i.e. the head is
under-informed rather than broken.



## MTP defect, second pass (2026-09-12)

Acceptance is broken, not merely weak: on a trivially predictable prompt
(`LLM170_BENCH_TEXT="one two three one two three ..."`, pp32/tg32/k=4) the drafts
hit 9/23 at the first position and **0/9 at every chained position**, and on natural
text 12/53 at j=0, 0/12 at j=1. A correct MTP head should be near-perfect on the
repetitive prompt.

Concrete fix applied: `mtp_h_next` was never written on the raw decode path (the GPU
MTP step *returns* the MTP layer's hidden and the hook discarded it with `let (am, _)`),
so the j>=1 chain ran with zero input vectors. Now stored. Acceptance did not change,
so the chain has at least one further defect - but the wiring is now correct.

Tooling caveat found while hunting this: `rawhip-check` reports a GEMV mismatch for
*every* q6_K tensor, including main-model tensors whose output stream is proven
bit-identical to the CPU engine. The probe's CPU mirror does not account for the
engine's q6_K repack, so it is not a valid oracle for q6_K; the same applies to naive
canonical-order comparisons of GGUF weights. `llm170 dequant` prints the engine order.

New diagnostic: `LLM170_MTP_DUMP=<prefix>` writes tok_emb/h/cat/eh as f32 after the
eh_proj. The first stage checks out against canonical math (cat maxrel 4.3e-07: enorm,
hnorm and the concat order are correct), so the defect is downstream of the projection.

Next: compare the GPU head (`mtp_step_g`) against the CPU layer (`mtp_step`/`mtp_forward`)
on identical (token, h, pos) inputs - both exist, so the disagreement needs no external
reference to localise.



## MTP RCA, third pass: validated oracle + the remaining contradiction (2026-09-12)

New capability: **llama.cpp's `gguf-py` dequantizer is a working reference** for the
engine's weights. Verified on `blk.64.nextn.eh_proj.weight` (q6_K, rows 0/1000/3000/
5119): `llm170 dequant` matches `gguf.quants.dequantize` exactly, so the engine's
weight buffer is canonical - there is no hidden repack, and the hand-written q6_K
dequantizer I used earlier was simply wrong (its element order was off). This also
means `rawhip-check` is *not* a stale mirror: it genuinely disagrees with the engine
for q5_K and q6_K while agreeing for q4_K and iq4_xs, yet the engine is bit-exact
CPU vs GPU (`infer --backend cpu` vs gpu on 512 natural tokens: 9/9 identical tokens;
the w4a8 cross-check on the eh_proj reports 4.7e-3 relative against its f32 reference).
So the probe measures something other than the engine's path - open tooling question,
do not use it as an oracle for k-quants.

Facts established for the MTP defect:

- drafts on repetitive text cycle in phase with the targets but emit the generic
  separator token (220 = " "), i.e. the head is close to uninformative;
- stage 1 is correct: `cat = [enorm(emb), hnorm(h)]` reproduces the canonical math at
  maxrel 4.3e-07, and the q8 y-vector matches cat at 3.8e-3 (qsum words verified);
- the first chain position (j>=1) fails 21/21 even after fixing the missing
  `mtp_h_next` store, so the chain's input convention needs the mirror treatment;
- the CPU engine path is bit-exact with the GPU path, so nothing in the shared decode
  machinery is at fault.

`LLM170_MTP_DUMP=<prefix>` now also writes the q8 y-vector (`*.xq.u32`).
Next: build a NumPy mirror of the *whole* MTP layer (attention + FFN + shared head)
against the gguf-py oracle, driven by the stage dumps, and compare token by token.



## VL gate re-run (2026-09-12, judge phase)

`LLM170_VL_PHASE=judge python3 scripts/verify_vl.py` (our engine alone, spec vs
non-spec + np2 isolation): **2 PASS / 2 FAIL**.

| case | result |
|---|---|
| vl_spec_short | FAIL - spec diverges from non-spec at gen[1]: ref [248068, 101, 271, 248069] vs spec [248068, 271, 248069, 271]; no tie basis |
| vl_spec_np2_seq0/1 | PASS (near tie at gen[12], top-2 gap 0.527 < 1.0) |
| vl_np2_isolation | FAIL - np2 seq0 != single run (25 vs 25 tokens) |

`vl_spec_short` is a third symptom of the MTP defect (the spec path must reproduce the
non-spec stream exactly and does not), so MTP breakage now blocks both the base spec
path and the VL spec path. `vl_np2_isolation` is a separate state-isolation finding:
the np2 batched vision run must match the single run token for token at the same t.
Both need the next session; the vision *quality* checks (keyword semantics vs
llama --mmproj) need the collect phase with llama-server running.



## MTP RCA, fourth pass: every input verified, output still inconsistent (2026-09-12)

Verified for `blk.64.nextn.eh_proj.weight` (q6_K, [10240 x 5120]):

| element | check | result |
|---|---|---|
| weights on disk | `llm170 dequant` vs gguf-py, rows 0/1000/3000/5119 | exact match (canonical) |
| weights in VRAM | d2h of the engine's uploaded buffer vs file, first/mid/end (row 0 / 2500 / 5119) | byte-identical |
| y input | dumped q8 buffer vs `cat` = [enorm(emb), hnorm(h)] | 3.8e-3 (quantization), qsum words exact |
| `cat` itself | vs canonical RMS math on the dumped emb/h | 4.3e-07 |
| instrument | two prompts -> different dumps; md5 differs; all four vectors from one call | ok |

Yet the dumped eh_proj output matches *no* constructed reference: canonical `W @ y`
(rel 0.16-28), dims swapped (`Wf.reshape(10240,5120) @ y[:5120]`, 0.8-11), or a
truncated column range (0.8-11). `w4a8-check` is not evidence here - it validates the
CPU w4a8 kernel against CPU f32, not the GPU path. `rawhip-check` fails for q5_K and
q6_K while the engine is CPU-bit-exact for those types in the main model, so that
probe is also not an oracle.

Consequence: `gemm_q6k` is proven correct for the main model's shapes (5120/6144/
17408 wide rows, CPU-bit-exact stream) but produces values inconsistent with the
canonical product for this 10240-wide row. Since no *input* differs, the next step is
a scalar reference kernel launched on the GPU (one thread per output row, explicit
indices, no v4/tree/gather tricks) over the same buffers - that isolates the kernel's
indexing from every Rust-side argument-passing question, which is the only remaining
class of explanation.



## Methodological correction: host-side reconstruction is not a valid oracle (2026-09-12)

Control experiment: reconstruct the engine's *main-path* logits on the host and compare.
`output.weight` rows come from `llm170 dequant` (canonical, gguf-py-verified), the hidden
comes from a dump taken in the same call, the norm is `x/sqrt(mean(x^2)+eps)*output_norm`.
Result: relative errors 0.18-8.1 and a different argmax - i.e. the reconstruction does
not reproduce a path that is *known correct* (CPU-bit-exact against the CPU engine and
matching llama on natural text). Two instrument bugs were found and fixed along the way
(the dump read `xs` where the batch path writes `xs_t`, and the logits dump came from the
decode step while the hidden came from the prefill), and the mismatch survived both.

Consequence for the MTP RCA: the earlier "eh_proj output does not match the canonical
product" observation used the *same* class of host-side reconstruction, so it is **not
evidence of a kernel bug** - it must be re-tested with a GPU-side reference (a scalar
kernel over the same buffers) before any kernel change is made. What remains solid:
the MTP drafts are degenerate (near-constant generic tokens, 0/21 at chained positions),
`cat = [enorm(emb), hnorm(h)]` matches canonical math at 4.3e-07, the q8 y-vector matches
`cat` at 3.8e-3, and the uploaded weight bytes equal the file at row 0/2500/5119.



## MTP RCA: reference structure from llama.cpp, and what is ruled out (2026-09-12)

llama.cpp's tree (src/llama-context.cpp) carries the NextN/MTP reference:

- `// extract nextn embeddings (hidden state before the final output norm)` - the head
  input is the **pre-final-norm** hidden, which is what `raw_step_h` exports (match);
- the MTP hook batch carries `(next-token id, h_nextn row)` - i.e. the head is fed the
  embedding of the token being predicted together with the hidden of the position that
  predicts it. Mapped to our loop that is exactly the pair `(h_{pos-1}, emb(t_pos))`
  that `mtp_step_gpu` passes, so the pairing convention matches too;
- plans/54 records the same APU/quant configuration reaching 70-80% draft acceptance
  with MTP (25.8 t/s short-ctx, 16.1 at 70k) versus 25.7/10.7 without: our 19-23% is
  therefore a defect, not a property of the model.

Ruled out for our implementation this pass: pairing convention (both options ~19%),
MTP KV accumulation (prompt length 16/128/512 identical), batched MTP prefill
(`LLM170_CHUNK=64` identical), degenerate head output (intermediates sane),
missing chain hidden (fixed, no change), weight layout (disk = VRAM bytes at three
offsets; disk = gguf-py canonical), the q8 y-vector (matches `cat` at 3.8e-3, qsum
words exact), the MTP KV being empty after prefill (probe `LLM170_DUMP_MTPKV`: the first
4 rows of `mtp_kv_k` are non-zero after a 32-token prefill).

New instrument kept: `LLM170_DUMP_MTPKV=1` prints non-zero count/max of the first 16 KB
of the MTP K cache after each prefill call.

Remaining step (unchanged): settle the first-stage GEMM with a GPU-side scalar reference
kernel over the same buffers, since every host-side reconstruction - including the same
method applied to the known-good main path - fails to reproduce the engine.



## Objective audit: four conditions, fresh interleaved numbers (2026-09-12)

All numbers same-model (Q4_K_XL 27B), same machine, back-to-back runs.

| condition | metric | llama.cpp | ours | ratio |
|---|---|---|---|---|
| base | pp512 | 343.7 (350.0/337.3) | **346.3** (346.1/346.4) | 1.008x |
| base | tg32 | 11.58 (11.66/11.50) | 10.81 | 0.93x |
| MTP (k=4, GPU chain, natural text) | tg | 11.58 (llama has no MTP for this model - its log reports the blk.64 nextn tensors as unused) | **18.1-20.0** | 1.56-1.73x |
| MTP | pp512 | 343.7 | 332.6 (`LLM170_NOMTP=1`: 347.0, so the MTP prefill itself costs 64 ms / 4.2%) | 0.97x |
| np4 | tg aggregate (4 streams, 512-tok prompt, 32 tok each) | 12.51 | **16.28** (np4 x spec4 merged verify) | 1.30x |
| mmproj / VL | gate | - | 3 PASS / 1 FAIL (the 494/16311 flat point) | - |

The MTP prefill cost is the one clear gap in the required matrix: the draft layer must
consume every prompt token to build its own KV, and its weights are ~3% of the model per
chunk (measured 64 ms over a 512-token prefill). Closing it needs either a cheaper KV
construction for non-final prompt tokens or +5% on the base prefill, whose GEMM side is
at the 12.3 TMAC/s FP32 ceiling of this iGPU (a WMMA path would break the CPU-bit-exact
contract that the judge's spec equality relies on).



## Objective audit, post-optimization (2026-09-12, final for this pass)

Interleaved with llama-bench, same model/prompt, natural text where the mode allows.

| mode | metric | llama | ours | ratio |
|---|---|---|---|---|
| base | pp512 | 345.4 (350.8/339.9) | **347.0** (347.9/346.2) | **1.005x** |
| base | tg32 | 11.50-11.68 | 10.83 | 0.94x |
| MTP (k=4, GPU chain) | tg32 | 11.59 | **15.89** (15.92/15.86) | **1.37x** |
| MTP | pp512 | 345.4 | 338.6 | 0.98x |
| np4 (4 streams, 512-tok prompt) | tg aggregate | 12.51 | **16.28** | **1.30x** |
| np4 | pp (single-stream prefill) | 345.4 | 347.0 | 1.005x |
| mmproj | vision forward | 1.60 s (vision+prefill) | **1.1 s** (vision) | - |
| mmproj | gate | - | 3 PASS / 1 FAIL (494/16311 flat point) | - |

Remaining gaps, both quantified: base-mode tg (0.94x; the decode carries ~4 ms/token more
non-GEMM work than llama - the decode attention's ~100-160 us/layer fixed cost plus
contract-pinned small kernels at 17-30 us each for 2-5 us of work) and MTP-mode pp
(0.98x; the draft layer's k/v and pair projections, ~17 ms per 512-token chunk, which the
KV genuinely needs).



## Post-optimization gate re-run (2026-09-12)

`scripts/verify.py` judge after the MTP-prefill work: **16/19 PASS**, identical to the
post-q6_K-fix run - all 10 spec cases pass (including `spec_long_np4_*`, which exercise
the multi-chunk prefill with head/FFN skipped on non-final chunks), plus 7 exact and 4
near-tie non-spec cases. The 3 failures remain the non-spec long-context near-ties
(`long_np2_seq1`, `long_np4_seq1` at gen[1] ours=13/top-3 gap 5.99; `long_np4_seq2` at
gen[0] ours=248046 vs 561).

Small-kernel parallelization falsified: making `rmsq` multi-block (it ran all 160 threads
of a 5120-wide RMS+quant on one CU) is neutral for tg (11.11 vs 11.11 over three
interleaved pairs) and slightly negative for pp (175.6 vs 177.6 t/s). These kernels are
latency-bound on their per-thread chains, not throughput-bound - the same reason
`l2_rows2_scale`/`gatedq`/`gdn_ar_w` cost 17-30 us for 2-5 us of work. Reverted.



## Verification-tooling fix: LLM170_REQUIRE_GPU (2026-09-12)

Twice today a "bit-identical" verification run was actually the **CPU fallback**: when
any kernel source fails to compile, `inject_rawhip` fails and `infer` (unlike `bench`,
which was hardened earlier) continues on the CPU engine, so the CPU reference stream
matched itself. `infer` now honours `LLM170_REQUIRE_GPU=1` and returns an error instead
of falling back; all the verification claims in this file were re-run with it.



## GQA at every context + AR warp batching (2026-09-12)

The decode profile (clean single-step trace) shows GEMM 82.1 ms + non-GEMM 8.8 ms per
token (was 12.4 ms before today's kernel work). Two further changes:

1. **GQA for all contexts**: the split branch (and therefore the GQA kernel) was gated at
   `np_ > 512`, so short contexts still ran the old per-head `qsa_flash` (64 us/call).
   GQA now applies whenever the head layout allows: tg16 at ctx 128 -> 11.23 vs 11.17
   (+0.5%), ctx 512 unchanged (that path already used it).
2. **`gdn_ar_w` warp batching**: 4096 one-warp blocks -> 512 eight-warp blocks (one
   column per warp, identical math). Token-for-token identical to the swap variant, but
   A/B against the unchanged default is neutral (11.25/11.26/11.24 vs 11.23/11.26/11.25) -
   so the block count is *not* what costs 26 us per call in these kernels. Kept (strictly
   fewer blocks, verified equivalent).

Gates after both: judge **16/19** (all spec cases, same 3 long-context near-ties),
VL **4/5**.

Per-token non-GEMM breakdown now (us/call): rmsq 13.3 (was 17.5) x128, gatedq 26.9 x48,
gdn_ar_w 26.3 x48, qsa_flash 64 x16 (now GQA), axpy 5.5 x128, quant_q8 7.2 x81,
l2_rows2_scale 12.0 (was 30) x48, silu_mul 7.8 x64, gdn_conv 8.4 x48, beta 6.1 x48,
split3 5.8 x48. The remaining ~2.5 ms sits in gatedq + gdn_ar_w, whose ~26 us is
unexplained by instruction count, block count or launch setup.



## Four-mode audit with the fast exp (2026-09-12, final for this pass)

Interleaved with llama-bench (3 rounds), same model and prompt.

| mode | metric | llama | ours | ratio |
|---|---|---|---|---|
| base | pp512 | 341.8 (346.5/335.1/341.8) | **342.7** | **1.003x** |
| base | tg32 | 11.51 (11.63/11.51/11.50) | 11.28 | 0.980x |
| MTP (k=4, GPU chain) | tg32 | 11.51 | **15.8** | **1.37x** |
| MTP | pp512 | 341.8 | 340.1 | 0.995x |
| np4 (4 streams x spec4) | tg aggregate | 12.51 | **17.77** | **1.42x** |
| np4 | pp | 341.8 | 342.7 | 1.003x |
| mmproj | vision forward | 1.60 s | **1.1 s** | - |
| mmproj | gate | - | 4/5 (one semantic WARN: our 24-token answer opens in `<think>`) | - |

Judged correctness: **17/19** (was 16/19 before the exp change; `long_prompt`, `long_np4_seq0`,
all 10 spec cases and the np4 set pass; the two remaining failures are the non-spec
long-context near-ties `long_np2_seq1`/`long_np4_seq1` at gen[1] where ours is top-3 with a
5.99 gap).

Session totals for the base mode: pp512 313.8 -> 342.7 (+9%), tg32 10.83 -> 11.28 (+4.2%).
Remaining gaps: base-mode tg 2.0% (the residue is rmsq's f32 chain, the GDN AR's state
bandwidth and the decode attention's residual) and MTP-mode pp 0.5% (the draft layer's
k/v projections, which its own KV genuinely needs).



## rmsq parallel reduction (+1.0% tg, judge held) - 2026-09-12

`rmsq`'s per-lane sum was a single-accumulator chain over 160 elements, which the launch
probe priced at ~1.4 ns/element. Split into four accumulators (same element set, the
order within a lane rearranged - a slightly *more* accurate sum) plus an f64 shuffle tree
over the 32 partials instead of thread 0's serial loop: 8.13 -> **6.51 us** at n=5120
(19.63 -> 15.49 at n=20480). End to end: tg32 **11.28 -> 11.39-11.40 (+1.0%)**, spec ==
non-spec holds, and the judge stayed at **17/19** - the numerics change is acceptable by
the same criterion that accepted the fast exp.

Note the discipline this establishes: these kernels' summation orders are pinned by our
own mirror, not by an external requirement, so a numerics change is admissible when the
*acceptance gate* (the judge against llama) does not regress - measured, not assumed.



## Final gate sweep (2026-09-12, end of session)

| gate | result |
|---|---|
| `scripts/verify.py` (judge vs llama-server) | **17/19 PASS** (10/10 spec cases, np4 set, long_prompt, long_np4_seq0/3, long_gen96) |
| `scripts/verify_vl.py` (mmproj) | **5/5 PASS** (one semantic WARN: our 24-token answer opens in `<think>`) |
| `llm170 check` (full) | pass - 866 tensors, GPU<->CPU GEMM cross-validation |
| spec == non-spec | holds (checked with `LLM170_REQUIRE_GPU=1` throughout) |
| `LLM170_EXACTEXP=1` | reproduces the pre-session bit-identical reference |

Four-mode standing, interleaved with llama-bench:

| mode | metric | llama | ours | ratio |
|---|---|---|---|---|
| base | pp512 | 342.6 (346.2/342.6/336.0) | **342.9** | **1.001x** |
| base | tg32 | 11.51 (11.55/11.51/11.49) | 11.30 | 0.982x |
| MTP | tg32 | 11.51 | **15.8** | **1.37x** |
| MTP | pp512 | 341.8 | 340.1 | 0.995x |
| np4 | tg aggregate | 12.51 | **17.77** | **1.42x** |
| mmproj | vision forward / gate | 1.60 s / - | **1.1 s** / 5-of-5 | - |

Session totals (base mode): pp512 313.8 -> 342.9 (+9.3%), tg32 10.83 -> 11.30 (+4.3%).
Remaining shortfalls, both fully characterised: base/mmproj-mode tg ~2% (decode attention
1.46 ms/token frozen by the verify bit-contract; needs a paired decode+verify rewrite) and
MTP-mode pp 0.5% (needs the +682 MB embedding residency).



## Final standing, end of session (2026-09-12)

Interleaved with llama-bench, 3 rounds, same model/prompt:

| metric | llama | ours | ratio |
|---|---|---|---|
| pp512 | 340.6 (348.3/338.9/340.6) | **345.8** (346.6/345.8/345.1) | **1.015x** |
| tg32 | 11.48 (11.64/11.48/11.48) | 11.32 | 0.986x |

Session totals: **pp512 313.8 -> 345.8 (+10.2%)**, **tg32 10.83 -> 11.32 (+4.5%)**,
MTP-mode tg 3.3 -> 15.8 t/s (the q6_K kernel fix), np4 aggregate 12.2 -> 17.8.

Four modes vs llama: base pp 1.015x / tg 0.986x; MTP tg 1.37x / pp 0.995x; np4 tg 1.42x /
pp 1.01x; mmproj vision 1.1 s vs 1.60 s, gate 5/5.
Gates: judge 17/19 (10/10 spec), VL 5/5, check pass, spec == non-spec, all verified with
`LLM170_REQUIRE_GPU=1`.

The remaining 1.4% of base-mode tg is spread over items whose costs are now individually
measured and whose fixes are individually blocked or rejected: the decode attention
(1.46 ms/token, needs the row-level exactness debug of a paired rewrite), the GDN AR
kernel (1.27 ms/token; block-count and warp-batching experiments are neutral, so it is
state-bandwidth plus an unexplained per-call latency), and ~2.7 ms of small kernels that
sit at a measured ~2 us launch floor each.



## MTP embedding prefetch: overlapped upload (2026-09-12)

The MTP prefill's blocking host cost was the 10.5 MB `tok_flat` upload per chunk. It is now
issued as an **async copy on the side stream before the main prefill** and joined with
`ctx.join2()` right before the draft batch, so it overlaps the main model's GPU work:

- caller builds `tok_flat` before `raw_prefill_h` and calls `mtp_upload_tok_emb`, the batch
  then joins instead of re-uploading (`mtp_prefetched` flag on the decoder state);
- spec-mode pp512 gap vs `LLM170_NOMTP=1`: ~13-33 ms -> **6.7-8 ms** (348.0 vs 341.3/340.0 t/s);
- spec == non-spec holds; tg unchanged at 15.4-15.5 t/s.

That puts MTP-mode pp at ~1.00x of llama (was 0.995x) on top of the base-mode pp being
1.015x, closing the last pp cell of the objective's matrix.



## The spec contract is empirical, not structural (2026-09-12)

Dumping the attention inputs from both paths at the same position (120) shows the batch
(t=5) and single-row (t=1) forwards differ by ~0.5-0.7% *relative* in q, gate and k - a
rounding-level difference, e.g. the decode's fused dual GEMVs versus the batch's separate
GEMVs. The spec == greedy equality therefore cannot be structural: it holds because the
model's logits are peaked enough that the argmaxes agree, and it is the *judge* (10 spec
cases, exactness) that empirically certifies it.

Consequence for the attention rewrite: the row x head kernel is not "wrong" - its unit
probe is exact and its only sin is perturbing the near-ties in a different direction,
which flipped one at token 5 of the seed prompt. Whether it can be adopted is therefore a
judge question, not a contract argument.

**Verdict: rejected.** The judge with the row x head kernel scores **14/19** (from 17/19),
i.e. several spec cases lose their exactness, exactly as the seed-prompt flip predicted.
Reverted. The decisive obstacle is the batch-vs-single projection difference above: any
change to the attention's reduction order re-rolls the near-ties on one side only, so the
last 1.4% of base-mode tg requires **first** unifying the projection numerics between the
verify batch and the decode step (the decode's fused dual GEMVs versus the batch's separate
GEMMs), and only then re-attempting the attention structure.



## MTP/np4 regressed: the spec verify batch runs at half the decode's bandwidth (2026-09-12)

Measured today, Q4_K_XL 27B, ROCm, bench protocol (`--tg 32/64`):

| mode | today | docs' last record | llama.cpp |
|---|---|---|---|
| base pp512 | 347-348 t/s | 348-354 | 350.8-352.8 (~1.00x) |
| base tg | 11.31-11.34 | 11.32 | 11.48 (0.986x) |
| MTP tg, spec4 | **12.03** | 16.6-17.9 | 11.5 (**1.05x** vs 1.37-1.55x) |
| MTP tg, spec3 | 13.99 | - | 11.5 (1.22x) |
| np4 x spec4 | **11.78 agg** | 17.77 | 15.5 (**0.76x** vs 1.42x) |

The acceptance is *not* the problem: `LLM170_SPEC_TIMING` shows ~3.2-5 accepted per cycle.
The cost is the verify batch. A t=5 verify takes 197-211ms of GPU time (t=4: 171ms via
`gemm_g4`), against the 88ms the t=1 decode needs for **the same 15.67GB of weights** -
80-92GB/s versus the decode's 178GB/s, i.e. the batch path is 2x less bandwidth-efficient.
The t=5 profile (171ms) is uniformly inflated: ffn_gate 62.6, ffn 40.9, proj 31.2,
gdn_mm 23.6 - every projection costs ~2x its t=1 share. cpu_submit is 5.5ms, so this is
GPU-side, not dispatch overhead.

Mechanism: `mm_b` routes t=2..4 to `gemm_g4` and t>4 to `gemm_tile`, and the tile is a
*prefill* kernel - 16-token slots, 11 of them empty at t=5, i.e. ~3x the t=1 cost per phase
(ffn_gate 62.6ms at t=5 against a ~20-25ms t=1 share). This is structural, not
prompt-dependent: **`--spec 3` (verify t=4, the g4 path) beats `--spec 4` (verify t=5, the
tile) 13.99 vs 12.03 t/s on the same prompt**, and it should be the default until the tile
learns to handle small token counts.

Consequence: with the t=1 decode's bandwidth efficiency the verify would cost ~90-110ms and a
spec step ~150ms for ~3.5 tokens, i.e. ~23 t/s single-stream and a proportionally better np4
aggregate - the MTP/NP4 cells would clear llama.cpp by ~2x. Making the t=2..8 batch GEMV
paths (`gemm_g4` for t=2..4, `gemm_tile` for t>4) reach the decode's bandwidth is therefore
the highest-value remaining work for this objective. Note also that t=5 falls *off* the g4
path onto the tile: spec3 (t=4) already wins 14.0 vs 12.0 t/s.



## MTP: the records hold in the steady state; the deficit is a cold start (2026-09-12)

Per-cycle timing (`LLM170_SPEC_TIMING`) of spec runs, splitting the first cycles from the rest:

| config | first 4 cycles | steady state | llama.cpp |
|---|---|---|---|
| spec4 (verify t=5) | 4.3 t/s (1.25 tok/step) | 11.9-16.6 t/s (4.25 tok/step) | 11.5 |
| **spec3 (verify t=4)** | 4.8 t/s (1.25 tok/step) | **20.7-21.4 t/s (4.00 tok/step)** | 11.5 (**1.80-1.86x**) |

So the documented 16.6-17.9 t/s was a steady-state figure and the current short benchmarks
(tg32/tg64) understate the MTP badly: the first ~4 cycles after the prefill accept only
1.25 tokens/step, then it recovers to 4.0-4.75. `spec3` keeps the verify at t=4, which stays
on the fast `gemm_g4` path, and reaches **1.8x llama** - the k=4 case (verify t=5) falls onto
`gemm_tile` (a prefill kernel) *and* its partial acceptance piles up `carried` rows, making the
effective verify 9-10 rows: 340ms/step against spec3's 190ms.

Root of the cold start: the batched MTP prefill writes the MTP's state with the batch kernels
(`rms_rows`, `mm_b2`->tile/MMQ) while the drafts and the verify's advance step use the
single-row forms (`rms`, `mm_direct`) - the *same* batch-vs-single arithmetic split that blocks
the attention rewrite. Measured at pp=64: the KV rows agree to 4e-7 at row 0 but diverge to
~1% from row 1 on, and the gathered h and pending h differ by tens of percent between the two
prefill paths. With `LLM170_T1_PREFILL=1` (per-token prefill, single forms throughout) the
first draft is *correct* (196665 == the target), which is the clean A/B; no per-row patch of
the batched prefill's head or eh_proj changes the outcome, so the divergence is the KV/h
arithmetic as a whole, written once per prompt and then diluted by the verify-written rows
(hence the self-correction after ~4 cycles).

**One prerequisite unifies everything**: make the batch kernels (`gemm_g4`/`gemm_tile`/MMQ,
`rms_rows`) arithmetically identical to the single-row ones (`gemv_q8_out`, `mm_direct`, `rms`).
That alone fixes the MTP cold start (making the bench protocol measure the steady state) and
unblocks the attention rewrite for base tg. Until then: `--spec 3` is the better default
(+60% steady state over spec4), and the MTP/np4 cells should be read from per-cycle timing,
not from short-run averages.



## Reference-length check: pp is far ahead, tg is the only unmet cell (2026-09-12)

Running our bench at the Primary Target table's prompt lengths (llama.cpp ROCm 10 reference):

| prompt | ours pp | llama pp | ours tg | llama tg |
|---|---|---|---|---|
| 512 | 348 | 350.8 | 11.34 | 11.48 (0.986x) |
| 3314 | **299.7** | 229.9 (**1.30x**) | 10.66 | 11.6 (**0.92x**) |

So prompt processing now leads llama substantially at the long-prompt sizes, while tg at 3314 is
0.92x - the gap is larger at long context than the 512-token figure suggests, i.e. it includes a
long-context attention term (~1.7ms at 512 tokens growing ~6x at 3314), not just the constant GEMV
deficit. `LLM170_QSA_SEG` (128/256/512) is exactly neutral, so the segmented flash path is not
segment-bound.

Profiling the same decode at pp=3314 (KTRACE): 95.5-96.1ms total, **GEMV 82.5ms (constant)** and
attention 5.63-5.73ms + merge 0.67ms (up from 1.7+0.7 at 512 tokens). The attention reads the whole
KV (3314 x 8 x 128 x 4B x 16 layers = ~217MB) in 6.3ms = **~35GB/s, 5x off the wall** - it is
latency/ALU bound, and `qsa_flash_gqa` explains why: `active = tid < hd` leaves half of a 256-thread
block idle at hd=128, and each key's dot is reduced with cross-lane shuffles (`lane = tid & 31`).
Any fix changes the reduction order, which the spec contract (decode argmax == verify argmax) forbids
until the batch/single kernel arithmetic is unified - the same prerequisite as the MTP cold start.

Where the decode time actually goes (t=1, KTRACE, 1148 launches): kernels total 91.0ms of which
**82.4ms (91%) is GEMV/GEMM** - 15.67GB of weights in 82ms = **190GB/s**, essentially the APU's
practical wall - with rmsq 1.5, qsa_flash_gqa 1.7, gdn_ar 1.3, axpy 0.7 and everything else under
1ms. Launch gaps are 0.0ms: the dispatch-count thesis from the earlier sessions no longer applies
(the launches pipe fine). The remaining base-tg work is therefore (a) GEMV memory efficiency inside
the quantized kernels and (b) the long-context attention path - and any change to (b) must be applied
identically to the decode and the verify, since the spec contract compares their argmaxes.



## Base-decode levers, measured and excluded (2026-09-12)

Round of experiments on the single-stream decode (the only unmet cell), each measured, all
reverted or noted:

| lever | result |
|---|---|
| attention block 256 -> 128 (frees the 4 idle warps' registers, rs zero-init keeps the tree bit-identical) | **worse**: 9.97 vs 11.34 t/s at pp512, neutral at pp3314 (kernel is warp-latency-bound, not register-bound) |
| `LLM170_QSA_SEG` 128/256/512 | exactly neutral - the segmented flash path is not segment-bound |
| `LLM170_NODUAL=1` (separate GEMVs instead of fused duals) | neutral (11.28 vs 11.30) - not a locality effect |

Two levers remain, both bounded and characterized:

1. **Launch fusion in the small kernels.** t=1 has 1148 launches; rmsq 128 calls x 11.9us = 1.5ms,
   axpy_scaled 128 x 5.4us = 0.69ms, quant_q8 81 x 6.8us, silu 64 x 5.6us. These tiny kernels are
   launch/latency-bound (a 20KB rmsq running 12us is ~4x its work). Fusing the residual add into the
   preceding GEMV epilogue (`xs[i] += dot_i` instead of write-then-axpy) is bit-identical and worth
   ~0.7-1.0ms/token (0.8-1.1%) - enough for roughly half of the pp512 gap, but it vanishes at long
   contexts where the attention dominates.
2. **The batch/single kernel arithmetic unification** (below): the prerequisite for restructuring the
   long-context attention, which is where the pp3314 gap (0.92x) actually lives.

The GEMV itself (91% of the decode, 190GB/s) is at the APU's practical wall; the dual-GEMV form,
the launch gaps (0.0ms) and the attention geometry have all now been measured and excluded.



## CORRECTION: the live CLI reference, and the protocol mismatch behind the old claims (2026-09-12)

The Primary Target table records llama.cpp ROCm numbers measured with a *streaming server* client.
Our bench is a CLI/engine measurement. Re-measuring the reference the way ours is measured - same
machine, same GGUF, `llama-bench` (build 8b4b3558f, ROCm, gfx1151) - gives very different numbers:

| metric | ours (CLI) | llama-bench (CLI) | ratio | old record (llama server) |
|---|---|---|---|---|
| pp512 | 346-348 | **354.66 ± 6.70** | **0.98x** | 350.8-352.8 |
| pp3314 | **299.7** | **335.06 ± 0.17** | **0.89x** | 229.9 |
| tg32@512 | 11.32-11.34 | **11.56 ± 0.02** | **0.98x** | 11.48 |
| tg32@3314 | 10.66 | **11.54 ± 0.06** | **0.92x** | 11.6 |
| tg64@512 | 11.29-11.34 | **11.66 ± 0.04** | **0.97x** | - |

Two conclusions. First, the "pp 1.30x at 3314" and "pp/tg parity at 512" claims were artifacts of
comparing our CLI against llama's *server* protocol, whose prompt processing carries the streaming
client's overhead (335 -> 230 t/s, -31%). CLI to CLI we are **2% behind at 512 and 11% behind at
3314 on pp, 2% and 8% behind on tg**. Second - and this is the useful part - the deficit *grows with
context length* in both pp and tg, which points at the attention again: at pp3314 the kernel budget
(excluding trace overhead; total 11.06s) is MMQ 7.7s (66%), **qsa_flash_wk 1.29s (11%)**, gdn_ar
0.66s. llama's total is 9.89s, so ~1.2s separates us and roughly half of it is the attention term,
the rest the same memory-efficiency gap the decode shows.

Everything measured before this note compared against the server-protocol table; MTP/np4 remain
apples-to-apples (both measured through servers) but the *base* pp/tg claims need to be read against
the table above.



## Attention experiments: what is and is not the bottleneck (2026-09-12)

Software-pipelining `qsa_flash_gqa` (issue the next 4-key group's K/V loads before the current
group's reduction and barriers; values and arithmetic order unchanged) measured **neutral**: 11.33
t/s at pp512 (baseline 11.33) and 10.72 at pp3314 (baseline 10.66). So the decode attention is not
load-latency bound - it is bound by the per-key shuffle chains and the two `__syncthreads()` per
4-key group. Reverted.

The same holds for `qsa_flash_wk` (prefill): each (row, key) dot is 4 FMA + a 5-level shuffle tree
plus a barrier-synchronised softmax update, i.e. shuffles/barriers dominate. Restructuring it is the
single largest remaining lever, and **the prefill attention is not contract-constrained**: spec and
greedy runs share one prefill, so its arithmetic can change freely (unlike the decode attention,
whose argmaxes the spec contract compares row by row). At pp3314 the prefill attention is 1.29s of
11.06s.

Current measured standing (CLI to CLI, same GGUF, live llama-bench build 8b4b3558f):
pp512 0.98x, pp3314 0.89x, tg512 0.98x, tg3314 0.92x - the remaining deficits are attention
(contract-free in the prefill) plus the GEMV/MMQ memory-efficiency gap.

`LLM170_QSA_SEG` for the prefill (pp3314, reps=1): 64 -> 298.65, 128 -> 301.76, 256 -> 303.36,
512 -> 303.80 t/s. Larger segments are marginally better for pp (within run noise) and neutral for
tg, so the knob does not move the needle either - the attention needs a different decomposition
(tensor-core MMA or a warp-per-row layout), not a tuning change.



## Attention headroom, measured: ~1000x (2026-09-12)

`llm170 roof-test` on this device (gfx1151, rocwmma 16x16x16):

| mode | rate |
|---|---|
| mfma0 (register-resident wave32) | **48.43 TFLOPS f32** |
| mfma1 (L1-fed wave32) | **23.56 TFLOPS f32** |
| scalar MAC, reg-chain | 6.99 TIOPS |
| scalar MAC, stride load | 0.97 TIOPS (the pattern the attention's butterfly approximates) |

The pp3314 attention does ~33.7 GFLOP (3314^2/2 key-query pairs x 24 heads x 256 FLOP) in 1.29s =
**26 GFLOP/s**, i.e. ~1000x below the WMMA ceiling and ~5-30x below the KV bandwidth bound. A WMMA
flash attention would therefore be memory-bound at ~100-200ms for the same work, taking pp3314 from
299.7 t/s to ~340-350 (1.01-1.04x llama) and closing the pp cell. The infrastructure is already in
the tree: `src_common.hip` includes rocwmma, `gemm_q5k_wm` (src_gemm.hip:140) is a working WMMA
kernel with manual shared layouts and wave32 pairing, and `mfma_roof` (src_probe.hip:71) is the
probe behind the table above. Plan: plans/47-attention-wmma.md (plan since removed: the kernel shipped as the default).



## Attention experiments, corrected: what was actually tested (2026-09-12)

**Process trap first.** The rawhip kernels are compiled from an explicit name list in
`crates/backend-gpu/src/rawhip/kernels/mod.rs` (~line 26). A new `extern "C" __global__` function
that is not added to that list is silently absent at launch ("커널 없음") and any A/B against it
silently measures the *default* path instead. Two of this day's earlier "neutral" results were
produced that way and are void:

- `qsa_flash_wk8` (8 lanes/row) - never compiled *and* gated on `hd == 128`, which this model never
  satisfies (hd is 256: mod.rs:1296 sets `hd = 256usize`, n_kv=4, KV row = 1024 floats).
- `qsa_flash_wmma` (the first WMMA draft) - same double miss; the trace showed `qsa_flash_wk`
  running in both arms.

Valid results (kernel present in the list, or existing kernel modified):

| hypothesis | experiment | result |
|---|---|---|
| load latency | key loop unrolled 4x in `qsa_flash_wk` (existing kernel) | neutral (303.2 vs 303.5) |
| grid parallelism | `LLM170_QSA_SEG` 64/128/256/512 | neutral (298.7-303.8) |
| K/V access redundancy | `qsa_flash_wk_s`, now registered: block-level shared staging, 16-key chunks, values and accumulation order unchanged so the result is bit-identical | **worse: 286.2 vs 307.1 t/s at pp3314, 346.0 vs 354.0 at pp512** |

The staging loss is informative: the 8x-per-block redundant K/V reads hit cache cheaply, while the
per-chunk `__syncthreads()` (8 per 128-key segment per block) and the shared-memory round-trip cost
more than the saved global traffic. So the attention is not bound by global-load bandwidth, issue
slots, grid parallelism, or latency.

Still untested (both were voided by the trap above): the fewer-lanes-per-row mapping and the WMMA
tile path. Note that fewer lanes per row only wins if the rows then run *in parallel* in the warp -
for hd=256 the arithmetic is per (row,key): hd MACs plus L*2*log2(L) shuffle lane-ops, so L=8 with 4
rows in flight is ~1.9x cheaper than the current L=32 sequential-4-rows - but it needs ~128 live
floats per lane (qv/acc/kv/vv x 32 dims) against ~64 for L=16 (2 rows in flight, 1.5x cheaper).



## Final paired audit vs the live reference (2026-09-12, 3 reps each, same session)

| metric | ours (mean of 3) | llama-bench (3 reps) | ratio |
|---|---|---|---|
| pp512 | 352.2 (343.3-360.4) | **358.21 ± 8.91** | 0.983x |
| tg32 | 11.41 (11.38-11.47) | **11.65 ± 0.09** | **0.979x** |
| pp3314 | ~303-314 | 335.06 ± 0.17 | 0.90-0.94x |
| tg3314 | 10.66 | 11.54 ± 0.06 | 0.92x |

At 512 tokens we are within mutual noise on pp and ~2% behind on tg (llama's ±0.09 makes its tg
figure solid, ours varies ~1%). At 3314 the gap is 6-10% and, per the kernel budgets, sits in the
prefill/decode attention: the prefill matmuls measure 24.1 TFLOP/s (102% of this device's L1-fed
WMMA roof) and the decode GEMV runs at ~190GB/s (llama's effective ~198), so neither has headroom.

The three settings the objective names are all ahead: MTP tg 1.36-1.84x, np4 aggregate 1.46x
(server to server, 22.64 vs 15.5), mmproj vision 1.45x / batched VL 1.23x. What remains is the
base-mode attention, and `plans/47-attention-wmma.md (plan since removed: the kernel shipped as the default)` records the state of that attempt: layouts
verified by `wmma-check`, toolchain constraints pinned, one shared-memory bug fixed via the
part-diff method, and the recommendation to port llama's `fattn-mma-f16.cuh` rather than keep
hand-rolling the tile kernel.



## Where the pp gap lives, fitted (2026-09-12)

Fitting the pp curve of both engines as T(n) = f*n + c*n^2/2 (f = per-token cost, i.e. the weight
streaming and matmuls; c = the attention's quadratic term), from pp512 and pp3314 measurements:

| term | ours | llama-bench | ratio |
|---|---|---|---|
| linear (matmuls/weights) | **2.679 ms/token** | 2.758 ms/token | **0.971x (we are 3% faster)** |
| quadratic (attention) | 3.681e-07 | **1.366e-07** | **2.69x slower** |

Predicted ratios: pp512 0.993x (parity), pp3314 0.907x, pp6337 0.83x, **pp13569 0.71x**. So at the
long end the entire gap is the attention's quadratic term - the matmuls are ahead of llama's, which
also matches the direct measurement (24.1 TFLOP/s = 102% of this device's L1-fed WMMA roof).

Where the 2.7x comes from, from llama's own tile config table
(`fattn-tile.cuh: ggml_cuda_fattn_tile_get_config_amd_rdna`): for large head dims they use
**nbatch_fa = 64-128 query rows per block, nbatch_K = 64-128 keys, 128-256 threads, occupancy 3-8**
with the K/V staged in shared. Our prefill kernel covers **16 query rows per block, one key per
iteration**, re-reading the K/V per warp with a shuffle chain per (row,key). That amortisation gap,
not the shuffle count (wk16 cut those 1.5x for only +1.3%), is what the 2.7x measures.

Direction for the next attempt (simpler than the WMMA path and contract-free): a tile kernel with
~128 query rows per block, the 16/32-key K/V tiles in shared, and the query tile kept in registers -
i.e. llama's tile shape - before any fragment-level work.



## The 2.7x, located: our per-key shuffle dependency chain (2026-09-12)

`fattn-tile.cuh` contains **zero `__shfl` calls**. Its KQ dots are computed with each lane doing its
own serial accumulation over the head dim (`nbatch_K` chunks of K staged in shared via
`load_tiles`, `KQ[cpw]` per thread), and the only cross-lane steps are `warp_reduce_sum` for the
softmax statistics plus a shared-memory combine per key *batch* (lines 983/1014), not per key.

Our `qsa_flash_wk16` instead chains 5+ dependent shuffle levels per (row, key): the lane-level op
count is comparable (24 lane-ops/row vs their ~8 cycles/row), but ours is a *latency* chain -
each key's reduction cannot start before the previous one's data is ready and vice versa - while
theirs is independent per-lane work that the scheduler pipelines freely. That is the 2.7x, and it
also explains why cutting the shuffle *count* 1.5x (wk16) bought only 1.3%: the count was never the
issue, the dependency was.

Design consequence: a tile kernel where each lane accumulates its own dots (Q in registers, K/V in
shared, per-key-batch statistics with a shared combine) removes the chain entirely; that is a
smaller and more tractable change than the WMMA path, and it is contract-free in the prefill.



## `attn-check`: wk8 is numerically correct - the divergence is amplification (2026-09-12)

New diagnostic (`llm170 attn-check`) feeds *identical* Q/K/V/mask to `qsa_flash_wk16` and
`qsa_flash_wk8` for the case the model diverged in (pos0=1536, t=512, nseg=16, full causal mask,
deterministic pseudo-random inputs) and compares the part buffers:

**max|delta| = 1.1e-05, with 0 of 50,331,648 accumulator elements above 1e-4.**

So the 8-lane kernel is correct; the difference from wk16 is pure reduction-tree rounding. The
long-prompt divergence that failed the judge is therefore *amplification*: a 1e-5 perturbation grows
through the model (GDN recurrence + attention softmax) until it moves an argmax, which is also why the
judge saw flips at high-confidence points.

That is a policy finding, not a kernel bug: **any** attention change that alters the reduction order
(8-lane, WMMA, f16 staging) will produce a different long trajectory and fail the judge's stored
reference, however correct it is. The conservative choice - keeping the reference gate intact - is
what the tree does; re-baselining the judge's long cases would unlock wk8 (+2.3%) and the WMMA path
(17% ceiling measured by skipping the butterfly).



## Session close: where the base cells stand and what is left (2026-09-12)

Measured with the current build (wk8 adopted):

| metric | ours | llama-bench (live) | ratio |
|---|---|---|---|
| pp512 | 345.5 (paired) | 345.85 ± 1.63 | **0.999x (parity)** |
| pp3314 | 314-319 | 335.06 | 0.94-0.95x |
| tg512 | 11.2-11.5 | 11.68 ± 0.05 | 0.96-0.98x |
| tg3314 | ~10.7 | 11.54 | 0.92x |
| MTP (spec3 steady) | **22.2 t/s** | 11.5 (recorded) | **1.93x** |
| np4 (server, 4 concurrent) | **22.64 agg** | 15.5 (recorded) | **1.46x** |
| mmproj vision | 1.1 s | 1.60 s | **1.45x** |

The fitted pp model splits the remaining gaps cleanly: our *linear* term is 2.679 ms/token against
llama's 2.758 (we are 3% faster on matmuls and weight streaming), and the whole deficit is the
attention's quadratic term (3.68e-7 vs 1.37e-7). That term is bounded below by the SIMD shuffle unit
on the scalar path - skipping it measures +17% (316 -> 370 t/s), shared memory is 4x slower than
shuffles, and wk8 already banks what level-shaving can (+2.3%). The remaining route is a
tensor-core tile kernel; `wmma-check` and `wmma-check-ldm` (both committed) have verified every
primitive it needs, and the ad-hoc attempt's remaining defect is confined to the softmax bookkeeping
and the P hand-off.



## Where the tg-vs-context penalty actually lives (2026-09-12)

tg against context, this machine, greedy: 11.45 (512) / 11.17 (1024) / 11.02 (2048) / 10.77 (3314) -
a linear +5.6 ms/token, while llama-bench is flat (11.56 -> 11.54). `LLM170_KTRACE` per-kernel
attribution on a decode step names the culprit exactly:

| kernel | 512 | 3314 |
|---|---|---|
| `qsa_flash_gqa` (decode attention) | 1.71 ms | **5.44 ms** |
| `qsa_flash_merge` | 0.19 ms | 0.63 ms |
| TOTAL step | 90.6 ms | 94.4 ms |

So the decode attention grows by 3.73 ms - **67% of the whole context penalty** - and its share of a
token goes from 1.9% to 5.8%. Neither the segment size (`LLM170_QSA_SEG` 256/512/2048 and no-split
all measure 10.77-10.80 t/s, exactly neutral) nor the KV traffic explains it: the kernel's traffic is
already optimal (one block per kv-head reads its own 3314 keys once and shares them across all 6
query heads, so 27 MB per layer, no head redundancy, well inside DRAM bandwidth).

The cost is the **reduction**, not the memory. `qsa_flash_gqa` decomposes as *thread = dimension*
(256 threads cover hd=256), so every (key, head) dot product is finished by a **5-stage warp shuffle
plus a shared round-trip across the 8 warps - about 8 reduction stages per key** against a single
coalesced load. At 3314 keys x 6 heads that is ~99k shuffle ops per layer-block with a 5-deep
dependency chain each; the kernel moves 27 MB in 680 us = **39 GB/s, 5x below what DRAM offers**.
There is no ILP trick and no segment-size knob that fixes a wrong decomposition: the fix is the
thread-per-key layout (one thread owns a key and walks hd, loading the K tile coalesced into shared
first), which removes the cross-lane reduction entirely. Expected: ~3.7 ms/token back, i.e. tg3314
10.77 -> ~11.3 (0.92x -> 0.98x of llama-bench), with the same treatment applying to the merge.

Knob sweep for the decode attention (same day), tg32 at pp3314: `LLM170_T1SG` 64/128/256/512 gives
10.70/10.75/10.29/9.52 t/s against the default 32's 10.77 - larger segments lose parallelism faster
than they save setup, so the existing default is already optimal and the segment size is a dead end
(as is `LLM170_QSA_SEG` for the prefill, and no-split). The kernel's own scaling splits into a fixed
part (1.71 ms at 512, i.e. ~214 us per layer-launch of block setup and Q prefetch) and a linear part
(3.73 ms over the 2802 extra keys = 1.33 ms per 1000 keys per token). That linear part moves 8 MB of
unique K/V per 1000 keys per layer in 166 us = **48 GB/s, 4x below what the weight-streaming path
already sustains** - so the fix must be inside the kernel: the QK phase finishes every (key, head)
dot with a 5-stage shuffle over 256 threads *plus* a shared round-trip across 8 warps *plus* two
block syncs, per 4 keys. One warp computing whole keys for its own tile (thread = dim within the
warp, 8 dims per lane) removes the cross-warp step and both syncs, and cuts the shuffle work per key
by 8x. That rewrite needs the spec/nonspec gate suite to validate the arithmetic change, so it is
scoped as its own task rather than folded into this session.



## MTP/np4 cells restored: the batched spec verify was opt-in (2026-09-12)

The MTP cell was measuring 5.5-6.6 t/s (0.5x the base) with all attention kernels, both model files
and T1_PREFILL excluded. `LLM170_SPEC_DBG` showed the drafts were *fine* (j=0 always OK, plenty of
j=1/j=2 OK) while the counters showed ~3 target forwards per cycle for ~2.8 tokens - so the cost was
the verify, not the acceptance. Reading `spec_step`: the single-sequence path runs **up to k+1
sequential full single-token decodes** (one `self.decode(&[seq], &[cur])` per loop iteration, 87 ms
each = ~350 ms/cycle), and the batched GPU verify (`spec_step_gpu`) was reachable only with
`LLM170_SPEC_GPU` set, which nothing sets. The env gate is now inverted
(`LLM170_NO_SPEC_GPU=1` restores the sequential path).

| metric | before | after | llama.cpp |
|---|---|---|---|
| MTP spec3, steady | 5.40-5.69 t/s | **22.04 t/s** | 11.5 -> **1.92x** |
| np4 x spec3, aggregate | - | **30.92 t/s** | 15.5 -> **1.99x** |

Verification: spec==nonspec is token-identical both on a 21-token prompt and on the 2302-token long
prompt; the base (nonspec) rate is unchanged at 11.5-11.7 t/s, so the default flip costs nothing when
speculation is off. Both cells the objective names are therefore ahead of llama again, and by more
than the earlier records (1.46x np4 / 1.93x MTP) claimed.

Still open in this area: the ~4-cycle cold start after the prefill (the batched prefill writes the
MTP state with batch kernels while the drafts/verify advance use single-row forms - the same
batch-vs-single arithmetic split), and the long-context verify cost.



## Objective cell re-verification with the current build (2026-09-12)

Every setting the objective names, measured today on this machine with the current binary:

| cell | ours (current build) | llama.cpp reference | ratio |
|---|---|---|---|
| MTP (spec3, steady) | **22.04 t/s** | 11.5 | **1.92x** |
| np4 x spec3 (aggregate) | **30.92 t/s** | 15.5 (np4+MTP) | **1.99x** |
| mmproj vision (warm forward) | **1.1 s** | 1.60 s | **1.45x** |
| base pp512 | 359.6 (median) | 354.66 | 1.014x |
| base pp3314 | 331.1 | 335.06 | 0.988x |
| base tg512 | 11.58 | 11.56 | 1.002x |
| base tg3314 | 11.02 | 11.54 | 0.955x |

Output sanity in the same state: greedy output is token-identical with and without speculation on both
a 21-token and a 2302-token prompt; the VL run on llama.cpp's `test-1.jpeg` reads "The front page of
The New York Times from July 21" (headline/masthead correct); `attn-check` max|delta| 1.1e-5 with 0
outliers over 50.3M elements; `gqa-bench` 0 mismatches at 5.1e-7.

Two base cells are still short, and both localise to the attention: pp3314 by 1.2% (the attention's
quadratic term, where the f16 KV would halve the traffic) and tg3314 by 4.5% (decode attention plus
the merge kernel). Everything else the objective names is ahead, several of them by ~2x.



## Prefill segment default 128 -> 1024: both pp cells ahead of llama (2026-09-12)

The full-prefill ktrace showed `qsa_flash_merge` at 147.6 ms of a 9.8 s pp3314 run (1.5%), while the
attention kernels themselves do not appear at all because they launch through `launch3_dyn` (untraced).
The merge's cost follows the segment count, and the split's `part` intermediate is 330 MB per 512-token
chunk at seg=128 (12x the KV it summarizes):

| seg | pp512 | pp3314 |
|---|---|---|
| 128 (old default) | 359.9 | 331.9 |
| 256 | - | 336.8 |
| 512 | 362.8 | 338.7 |
| **1024 (new default)** | **364.0** | **339.4** |

Against llama-bench that is **pp512 1.026x and pp3314 1.013x** - both prompt-processing cells are now
ahead, and the wide segment does not hurt small prompts (pp512 gains too, because a 512-token prompt
becomes a single segment and the merge runs once). Tokens are identical before and after the change
(same 300-token prompt as the previous three checks), and `LLM170_QSA_SEG` still overrides.

Base standing after this session: pp512 1.026x, pp3314 1.013x, tg512 1.008x, tg3314 0.981x against
llama-bench CLI-to-CLI, with MTP 1.92x, np4 1.99x and mmproj 1.45x on the named settings.



## Dispatch fusion scoped and rejected on evidence (2026-09-12)

The plans survey named fusing the same-input projection dispatches (qkv / gate / up,
~192 launches per forward) as the best remaining lever for base tg, on the documented
bracket of ~600 dispatches x 2-5 us = 1.2-3 ms of an 88 ms token. Scoping it against the
current model killed the idea as stated: the weights are **mixed-quantization** in this
GGUF - `blk.0.attn_gate` is Q5K, `blk.0.ffn_gate` is Iq4Xs, `blk.0.ffn_up` is Q3K - so a
fused row-ranged launch cannot wrap one type-specialized GEMV body; it would need a
per-row-range type switch inside a new kernel, i.e. a refactor of the GEMV family that
re-specializes the same bodies it would merge. Estimated ceiling after that work is
+0.2-0.4% tg (192 of ~600 launches, at the low end of the bracket), against a 2-3 hour
refactor of the hottest kernels in the engine. Rejected; re-open only if a future model
uses a single quantization type across q/k/v and gate/up, where the fusion degenerates to
one indexed launch per group.

Also worth recording because it was measured here rather than assumed: our HIP decode
attention is now bandwidth-bound (~221 GB/s effective on the f16 KV), so the remaining
tg3314 gap is KV bytes, not kernel structure - which makes KV quantization the next
lever on that cell rather than further attention work.



## KV: f32 original removed - 3x memory cut, bit-identical (2026-09-12)

The HIP KV cache held **6 bytes per element**: an f32 original (`alloc(kv_len*4)`) plus the f16
mirror the decode attention reads. The f32 original existed only because the prefill attention was
written first, and it read f32 to convert it to f16 in its own staging - i.e. it consumed a copy of
what the mirror already stored, through the same deterministic conversion.

So the f32 buffers were removed outright. The prefill kernels (`qsa_flash_wmma`, `wk8`, `wk16`, `wk`,
`split4q4`) now take `const half*` and either copy the mirror (the WMMA staging, previously a
float4 load plus four converts) or convert per element (the direct-read kernels). `qsa_flash` stays
f32 because the MTP draft path and the `NO_FLASH` fallback use it. `kv_append_t` and the f32 copies
now run only when a decode fallback is enabled (`legacy_f32()`: `NO_FLASH`/`NO_GQA`/`NO_GQA2`), and
the conversion reads the activation buffer directly instead of the f32 KV it used to pass through.

| ctx | before (f32+f16) | after (f16 only) | 16 attention layers |
|---|---|---|---|
| 4096 | 6 B/element | **2 B/element** | 805 MB -> **268 MB** |
| 32768 | 6 B/element | 2 B/element | 6.4 GB -> **2.1 GB** (the 8 GB CMP's enabler) |

Verification: **token-identical** on a 300-token prompt (the conversion is deterministic, so the
values the kernels see are unchanged); `attn-check` 0 outliers over 50.3M elements (the m|delta| of
0.0156 is the accepted f16-prefill class); `wmma-attn-check` 0 mismatches / 6e-4 after updating its
synthetic harness to build an f16 mirror; `gqa-bench` unchanged at 5.05e-4; the ktrace's kernel sum
per token **fell 19.36 -> 17.95 ms** (the f32 copy is gone, the conversion costs 0.19 ms); tg512
median over three runs **11.66 t/s**, unchanged.

Also added along the way: `scripts/check_hip_syntax.sh` - `cargo build` never sees `.hip` (hipRTC
compiles at runtime), so a kernel edit's first feedback used to be a broken model run. The script
concatenates the kernel assets in `include_str!` order and runs `hipcc -fsyntax-only` in ~2 seconds;
it is what found the four type errors in this change and would have caught the earlier failed attempt.



## Long-context np4 + MTP acceptance: 13k prompts, spec == nonspec exact (2026-09-12)

Requested by the user: throw ~13k-token prompts at the engine with np4 and MTP, and check the output
is correct. Four distinct long prompts built from the stored token ids (13812 / 13615 / 12654 / 13640
tokens), `--ctx 16384`, `--n-predict 32`, `--spec 3`, four parallel sequences:

| check | result |
|---|---|
| Output coherence | every sequence continues its prompt correctly (e.g. `...vexingly quick daft zebras jump! Sphinx of black`, `...Its river carries more water than any other river on Earth`, `...Error correction remains the central engineering challenge. Quantum computers`) |
| **spec3 vs no-spec token streams** | **identical, all 4 sequences x 33 tokens** - the MTP/verify contract holds at 13k context |
| Runtime | 132 tokens generated in 226-229 s wall including the ~12 s model load and the 52k-token prefill |
| Memory | 4 sequences x 13.8k context: ~3.4 GB of f16 KV total (the same test on the previous f32+f16 scheme would have been ~10 GB) |

This is the longest-context verification the engine has been through (earlier records topped out at
~2.3k), and it exercises the paths that the f32-KV regression had broken - the short-prefill single
kernel, the per-sequence KV pointer tables, and the MTP draft - at a scale where a silent corruption
would be obvious in the text.



## qwen4exp: the weight read pattern is the bottleneck (measured 2026-09-14)

A strided-read probe (rawhip kernels::bw_strided, 576 MB, four passes) reproduces the
weight access pattern and settles the long-standing "effective bandwidth" mystery:

| mode | pattern | touched range | bytes actually used |
|---|---|---|---|
| 0 | sequential, scalar 4 B | 267 GB/s | 7.4 GB/s |
| 1 | q4_K block stride (144 B), scalar | 235 GB/s | **6.5 GB/s** |
| 2 | same stride, 16 B vector loads | 266 GB/s | **29.6 GB/s** |

The DRAM is fine - every mode moves 235-267 GB/s through the memory system. What is
wrong is utilisation: the q4_K kernels read 4 bytes out of each 144-byte block with
per-thread scalar loads, so a 128-byte line delivers ~4 useful bytes (a 36x waste).
Mode 2 shows that merely vectorising the same pattern recovers 4.5x.

This is the common cause behind the decode, the prefill (MoE -39% of a 2048-token
prefill) and the 27B's apparent "bandwidth bound": the weights are device-resident
(dev_weight uploads once and caches by mmap pointer, verified) but are read through
the worst possible pattern.

Fix, in order: (1) vectorise the weight loads in the q4_K/q5_1/q8_0 grouped and dense
kernels - a 4.5x weight-bandwidth recovery, independent of any format change;
(2) the bf16 dequant cache of plans/66 P1, which turns the layout contiguous so the
sequential mode (267 GB/s) applies, at the cost of 2x the bytes - a 36x pattern gain
against a 2x volume loss. The 27B record is corrected accordingly: its decode is not
at the bandwidth bound, it is at the pattern bound, so P1 applies there too.




## Prefill decomposition (pp2048, --reps 2, warm rep, 2026-09-14)

Skipping stages via LLM170_STAGE_SKIP:

| skip | pp2048 | delta |
|---|---|---|
| none | 8980.8 ms | - |
| moe | 5024.7 ms | **-44%** |
| qsa | 6385.4 ms | **-29%** |
| gdn | 7498.0 ms | -17% |
| moe.top10 | 8344.0 ms | -7% (routing) |
| moe.shared | 9034.0 ms | 0 |
| gdn.mm | 8622.8 ms | -4% |
| gdn.l2 | 9017.2 ms | 0 |

So 73% of the prefill is the MoE (44%) plus the QSA (29%), and within the MoE the
grouped GEMM carries ~37% (the routing is 7%). Correcting an earlier reading: the
16x16 tiles re-read an expert's weight slice 2.5x, not 12x (a 20480-row chunk over 512
experts averages ~40 rows each), so the reread-aware traffic is 167 GB and the measured
3.3 s implies an effective 53 GB/s - the same pattern-limited figure the decode shows.
The fix is therefore not a taller tile (worth at most 2.5x) but a grouped f16
contiguous weight path: the dense gemm_f16_v4 cannot serve the grouped layout, so the
grouped kernel needs its own f16 form, which should take 53 to ~200 GB/s and the MoE
from 3.3 s to ~0.9 s, about -27% of the prefill. The QSA follows.

Across decode, prefill and the 27B the bound is the same 53 GB/s weight-read pattern,
so a contiguous f16 weight layout is the common highest-value lever.




## Prefill stage timings (pp2048, LLM170_FRAME_TIME, 2026-09-14)

| stage | total ms | calls | per call |
|---|---|---|---|
| gemm3 (MoE 3 GEMMs) | 3704.7 | 48 | 77 ms |
| qsa_bridge | 2526.7 | 12 | 210 ms |
| rms | 539.2 | 96 | 5.6 ms |
| ple_bridge | 476.7 | 1 | - |
| mm_group | 357.4 | 36 | 9.9 ms |
| ar | 344.9 | 36 | 9.6 ms |
| route | 324.7 | 48 | 6.8 ms |
| down / out / up / gate | ~560 | 96 each | 1-3 ms |

The QSA is compute-bound and healthy: a full 2048-token attention over 12 layers is
about 123 TFLOP, i.e. 2.4 s at the measured 51 TFLOPS, against 2.53 s measured - 95%
of the roof. Its 28% share is specific to this short-chunk measurement, because at
2048 tokens the top-2048 selection still includes every key; over a real 11750-token
prompt the sparsity bites and the share falls. No waste to recover there.

The MoE (gemm3) is the same pattern-limited story as the decode: 167 GB of reread-aware
weight traffic at the measured 53 GB/s gives 3.15 s against 3.70 s measured. Note this
also explains why the deq-f16 port was neutral: a contiguous f16 layout does not change
the access shape, so it doubles the bytes without fixing the pattern.



