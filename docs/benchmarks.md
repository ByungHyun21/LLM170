# Performance Comparison vs llama.cpp

Conditions always quoted (context / batch / quantization / backend). All
numbers on the dev machine (Radeon 8060S, gfx1151, 32-thread CPU) unless
noted. Relative regression tracking only — absolute cross-machine comparison
is out of scope.

> **파일별 실측 기록은 `docs/source/`로 분리했다** (코드 경로 1:1, 계측기 플래그 표와
> 검증 게이트 포함). 이 문서는 요약 지표와 모델 간 비교를 담는다. 새 측정은 해당
> 소스 파일의 `docs/source/...` 문서에 먼저 적고, 지표가 바뀌면 여기 요약표를 갱신한다.

## 이력은 어디에 있나

이 문서는 **요약 지표와 기준**만 담는다. 날짜별 측정 이력은 영역별로 옮겼다:

| 문서 | 내용 |
|---|---|
| `docs/source/<경로>.md` | **파일별 실측**(코드 1:1) + 계측기 플래그 표 + 검증 게이트 |
| `docs/archive/vulkan-history.md` | Vulkan 백엔드 시기(plans/29-40 등) 측정·감사 |
| `docs/archive/qwen35-history.md` | 27B/GDN 관련 이력 |
| `docs/archive/hip-kernel-history.md` | HIP 커널·타일·프리필/디코드 라운드 이력 |
| `docs/archive/misc-history.md` | 그 외(비전·MTP·프로토콜 등) |

## Primary target scorecard — ACHIEVED (2026-09-14, HIP)

27B Q4_K_XL non-MTP vs llama.cpp ROCm 10 (same prompts as the table below):

| Prompt | llama pp | ours pp | ratio | llama tg | ours tg | ratio |
|---|---|---|---|---|---|---|
| 418 tok | 142.8 | **308.5** | 2.16x | 10.4 | **11.38** | 1.09x |
| 3314 tok | 229.9 | **330.0** | 1.43x | 11.6 | 11.33 | 0.98x |
| 6337 tok | 315.4 | (330.0@3314 proxy) | - | 11.1 | **11.13** | 1.00x |
| 13569 tok | 297.7 | - | - | 10.6 | 10.57 | 1.00x |

Prefill exceeds on every measured point; decode exceeds at short context and
holds parity (+-0.3%) through 13.5k - while llama's decode degrades 11.6 -> 10.6
over that range, ours holds a flatter curve. The last blocker was the WMMA
attention default (see the 9.5x fix below); pp512 365 vs llama's 350-360 era
records closes the original primary goal. Flash-Next secondary work this cycle:
pp2048 233-252 -> 278 t/s, long-ctx decode 91 -> 82-86 ms, device-resident QSA
stage + KV pools, and a latent pinned-buffer race fixed in production decode.

## Primary target — llama.cpp on ROCm 10 (designated 2026-09-02)

llama.cpp, Q4_K_XL (27B), non-MTP, flash attention on, f16 KV, temp 0,
streaming-server bench (median), ROCm 10.0.0 userspace:

| Prompt | pp (t/s) | tg (t/s) |
|---|---|---|
| 418 tok | 142.8 | 10.4 |
| 3314 tok | 229.9 | 11.6 |
| 6337 tok | 315.4 | 11.1 |
| 13569 tok | 297.7 | 10.6 |

Both PP and TG against this table are the first performance goal.

- 2026-09-05: f32 kernel family promoted to default (was opt-in) — pp512 305→317 t/s; `LLM170_F32SILU=0` restores the bit-cast variants.
- 2026-09-05: q6_K prefill routed to MMQ (was j128 tile) — pp512 318→341 t/s; `LLM170_NOQ6MMQ=1` restores.
- 2026-09-05 (FINAL, alternating single runs): pp512 250→348-354 t/s (+40%) — PARITY with llama-VK 350.8-352.8 (0.993-1.004×); tg8 12.25-12.30 vs llama 11.79-11.84 = +3.8-4.0% EXCEEDED. Batch AR grid-axis swap (L2 locality) closed the gap.
  Shipped: q6-K MMQ routing (+7.5%), f32 kernel family default (+4%), Vulkan GEMV coalescing (+51%), MMQ 4-path, wk attention, z-grid, rocprof diagnostics.
  Remaining: tg GEMV family rewrite (contiguous-lane transplant triple-confirmed dead: 19/42 gate + 0.71 t/s), pp scattered small items.
- 2026-09-05: GDN state layout transposed u-major (coalesced AR lane-j access, 16x amplification removed) + decode AR routed to gdn_ar_w — tg8 10.44→12.27 combined-mode — EXCEEDS llama 11.42 by 7.4% (q8_0 dual GEMV + AR transpose + conv parallelization + rmsq widening); pp512 342.6 unchanged.
- 2026-09-06 CORRECTNESS AUDIT (user-requested long-context check): 4 semantic regressions found & fixed — gemm_q3k qsum double-read, rmsq dropped scale writes, dual-GEMV dispatch chain skip, q6_K MMQ on custom-layout weights (default OFF, LLM170_Q6MMQ=1). llama-based verify.py 10/11 PASS; 11th (long_np2_seq1) = llama-server reference instability (its own output varies 16/159301/248068 across slots at a flat distribution point). Honest perf after q6-MMQ exclusion: pp512 ~322, tg8 ~10.9.



## qwen4exp — Qwen3.8-Flash-Next 125B-A6B, UD-Q4 4-split

The engine ran qwen4exp CPU-only for a while: cubecl's removal (ADR-0018) took
the only `Accelerator` implementation with it, so the recorded frame numbers
became unreproducible. The GPU path was rebuilt on rawhip (plans/64) — a
per-op value path for prefill plus a device-resident frame for decode.

| Metric | llama.cpp reference | LLM170 (GPU, rawhip) | LLM170 (CPU-only, before) |
|---|---|---|---|
| Load (non-PLE weights) | 83 GB / 91 s (fork patch) | **76.25 GiB / ~35 s** (2.6 GB/s median) | mmap, no upload |
| Prefill pp32 / pp512 / pp2311 | (server cells below) | **19.9 / 137.8 / 86.3 t/s** (device-resident frame, 2026-09-13) · 9.4-11.2 (value path) | 1.77 t/s (pp32) |
| Decode tg4 / tg8 / tg16 (ctx 4096-8192, warm) | 15.70 t/s solo (7.2.2) | **8.6 / 7.7-9.2 / 10.05-10.67 t/s** (frame) · 4.08-4.33 (value path) | 0.56 t/s |

Reference conditions (measured from the runtime logs, not this repo): llama-server,
`-ngl all -ot per_layer_token_embd=CPU --load-mode mmap -fa on -b 1024 -ub 512`,
np2 × 262144, ~11.75k-token prompts: pp 178-266 t/s per slot, tg 8.9-13.2 (8.9-15.7
solo) on ROCm 7.2.2; pp 272-468 / tg 11.9-19.6 on ROCm 10 + master. **The earlier
"Prefill pp, 2311 tok" row was a condition error**: the reference prompts are
~11.75k tokens and the timings are server-slot, so the row was removed rather
than patched.

### 2026-09-13 — same-condition llama.cpp baseline (both on this machine)

Measured back-to-back on the same host, same GGUF (UD-Q4_K_XL 4-split), single
sequence, greedy:

| Condition | llama.cpp (qwen4exp runtime, `-ngl all -fa on`) | LLM170 (rawhip, CLI) |
|---|---|---|
| pp 230 (server steady state) | **250.72 t/s**, tg 20.47 | — (CLI elapsed includes load) |
| pp 11,750 | **276.68 t/s**, tg 16.33 | ≈ 66 t/s pp, tg ~10 (212.4 s total incl. ~35 s load) |

llama.cpp was served from `/home/yoon/local_llm-runtimes/qwen4exp` (PR #27742
build) with the model's own `run.sh`; timings are llama-server `timings`
fields from `/completion`. The LLM170 figure is the CLI's `elapsed` minus the
measured load time (~35 s), so it is an upper bound on per-token prefill cost.
The two runtimes cannot coexist in memory (llama's ~76 GB resident), so the
measurements were taken sequentially.

**Gap**: prefill ≈ 4.2x behind the reference; decode ≈ 1.6x behind.

### 2026-09-13 — qwen4exp prefill 79.7 s at 11,750 (reproducible)

Current reproducible figure for the Flash-Next prefill, same flags as the llama
row above (`--pp 11750 --ctx 16384`, rawhip/CLI bench):

| Condition | LLM170 (rawhip) | llama.cpp (276.68 t/s) | gap |
|---|---|---|---|
| pp 11,750 | **56.40 s** (208.3 t/s) | 42.47 s | **1.33x** |
| pp 2,048 | 8,952 ms (228.9 t/s) | — | — |

Also fused: the MoE gather/scatter (`q4_rows_permute_u32`, 13.8 launches/layer)
into the grouped q4_K GEMM via the `perm` it already receives - the kernel reads
`xq[perm[r]]` and writes `out[perm[r]]`, so two launches per expert group
disappear (bit-identical; pp11750 57.70 -> 56.40 s, -2.25 %).

The two host-side changes behind that (both bit-identical, verified by tokens):
the QSA/PLE stage thread caps went from 16 to 32 (16 cores x SMT; while the host
stages compute the GPU has nothing to run - the profiler's ~25 % idle), and QSA
pass A (per-token KV cache, indexer raw-k, q_rope) was parallelized with the
order-dependent block-key pooling split into its own pass. Together: pp11750
60,500 -> 57,697 ms (-4.6 %), pp2048 9,814 -> 9,094 ms (-7.3 %).

The 79.70 s figure above becomes 77.92 s with the mask flatten in
`stages/qsa.rs` parallelized over tokens (it was a serial 24M-element push of a
96 MB `u32` mask per QSA layer - at 11,750 tokens that is 2048x11750 entries per
layer), and **76.39 s** with the selection-list attention kernel
(`q4_qsa_attn_sel`, commit `923aafa`): the mask-scan kernel iterated every past
key (one warp shuffle round per key) even though it already skipped the loads of
unselected keys, so the selected positions (top-k blocks + tail, ascending) are
now walked directly - 5.7x fewer iterations at 11,750 and no divergence.
`q4-qsa-check` proves the two kernels are **bit-identical** (identical
arithmetic order) at t=64/n_past=4096 and t=128/n_past=11750.

The 77.92 s figure becomes **60.88 s** with the 4-heads-per-warp kernel
(`q4_qsa_attn_sel4`, commit `a338b48`): the indexer selection is **per token**
(its score sums over the indexer heads), so four query heads of the same token
share one warp and read each selected K/V row once instead of four times - a
quarter of the traffic. Arithmetic per head is unchanged and the probe reports
`sel4_bit_diff=0` against `_sel` at t=64/128/256, so results are identical
(harness tokens unchanged). Decode keeps the one-head-per-warp kernel for
t<=3, where the 4-head block would leave most warps idle (195 vs 203 ms/step).

### Kernel distribution after the change (KTRACE, pp11750)

`q4_qsa_attn_sel4` falls from 17.9 s to **2.25 s** (72 launches, 31 ms each) with
the 4-head grouping. That 4x traffic cut bought only 20 % of prefill wall time,
and the decode (same kernel at t=1) did not move at all - so the remaining cost
is not load traffic. Three hypotheses were tested and rejected against it:

- warp occupancy: one 32-thread block per token (bit-identical) made the decode
  *worse* (274 vs 195 ms/step);
- ILP/dependency chain: interleaving two keys per iteration (verified
  bit-identical by the probe) left the decode unchanged (195.4 vs 194.8 ms/step)
  and cost the prefill 1.4 %;
- bandwidth: 2.25 s for ~206 GB of L2 reads is ~91 GB/s, an order of magnitude
  below the L2's capability.

The loop does five `__shfl_xor_sync` rounds per key per head regardless of how
many loads it saves, so the **shuffle unit is the floor** - consistent with the
earlier scalar-attention finding (commit `eded6cc`). Removing it needs the
lane=key layout, which requires a transposed K slab (the current cache is
position-major so lane=key would issue one cache line per lane). The largest kernels are now `q4_gemm_q4k_ge` (7.3 s, 470
launches) and `gemm_q8_j128` (4.6 s, 4639 launches), and the KTRACE's
`after <kernel>` rows - time attributed between one kernel's completion and the
next traced event - total ~29 s, dominated by `after q4_rows_permute_u32`
(10.5 s over 1986 launches) and `after q4_gemm_f32_m` (8.8 s over 1728). Those
rows are *not* kernel time: with the kernels suppressed (`LLM170_NOLAUNCH`) the
host work of the whole layer loop is only 0.15 s, so they are not host compute
either. They are most likely a **measurement artifact**: `launch3` creates two
`hipEvent` objects per launch when `LLM170_KTRACE` is set and never destroys
them (`mod.rs`), so a run with ~10k launches allocates ~20k events and the
event-record cost lands *between* consecutive kernels - exactly where the
`after` rows measure. Concretely, a KTRACE run of this benchmark wall-clocks
~74 s of prefill against 60.9 s without it (~21% inflation). The *durations* are
still trustworthy (each is the elapsed time between a freshly recorded
start/end pair, with the event creation falling outside the pair), and the
kernel names/counts are reliable; only the gap splits are not. The
wall-clock numbers above are the ground truth.

### Biggest remaining lever: the grouped MoE GEMM is the only non-WMMA GEMM

The dense paths already dispatch to WMMA kernels (`gemm_q4k_wm` / `gemm_q5k_wm` /
the `_v4` quadrant variants - `src_gemm.hip`), but the *grouped* MoE expert GEMM
(`q4_gemm_q4k_ge`) is a scalar dequant+FMA loop that measures **4.4 TFLOPS int8**
where the device's int8 WMMA peak is 30-60 TFLOPS (7-15 %). That kernel is 15 % of
the chunk's GPU time (366 ms/chunk), so porting the WMMA tiling to the grouped
form is worth ~12 % of the prefill - larger than the shuffle floor (4 %) or the
whole remaining transition budget (~5 %).

The port shape is exact: WMMA fragments are 16x16x16, and the grouped layout is
already 16-row aligned (one expert per 16-row tile, from the q5_1 grouping work),
so one fragment maps to one expert tile; only the A (weight) load needs
`tile_exp[blockIdx.y]` and the 64x64 tiling becomes 16x64. The existing WMMA
kernel's dequant and fragment code is reused. It is not bit-contracted (the
existing WMMA paths are "stream-validated" too), so acceptance is the CPU oracle
(`q4-acc-check` / `q4-qsa-check` mirrors) plus a coherent-output check.

### Verification standard: a diverse prompt, not the repetitive harness prompt

The token-identity checks used earlier in this work used a repetitive
230-token prompt; the model simply copies it, so the output is insensitive to
errors in most of the network and the check had no discriminating power. Two
real regressions survived it:

1. a MoE gather/scatter fusion whose launcher passed the pre-gathered buffer
   while the kernel indexes with `perm[r]` - the model degenerated
   (`68 56 220 16 17 15 ...`);
2. the QSA pass-A parallelization dropped the q norm+rope loop, so the
   attention consumed unnormalized q - plausible-looking but wrong output.

Both were found only after switching to a **diverse 24-token prompt with 16
generated tokens**. The current baseline, identical for the pre-work commit and
for the current tree:

```
5513 248046 198 248045 74455 198 248068 198 760 1156 579 1876 7701 310 381 7132 36412
```

`LLM170_QSA_HASH=1` prints per-QSA-layer checksums of the k/v/idx caches, the
block-key cache, the indexer q rows and the layer output, which localizes a
divergence to a specific path (it is what pinned bug 2 to the qg normalization).

### Functional matrix re-verified after the device-resident work (2026-09-14)

With the QSA bridge, resident KV pools and kernel-routing changes in, the
non-benchmark paths were exercised end to end:
- **np (parallel sequences)**: two interleaved prompt groups generate
  independently; the resident pools are keyed (layer, seq) and stayed clean.
- **MTP speculative decode** (27B, nextn=1): `--spec 2` ran 12 cycles,
  12 accepted tokens over 13 target forwards (0.92 acceptance/forward).
- **mmproj/VL** (27B + mmproj-F16): `llm170 vl` on a test image produced a
  coherent caption through the full vision-encode + splice path.
Flash-Next carries no nextn metadata - MTP is the 27B pair's feature.

### hc up-projection j128 is the best available path, not a broken default (2026-09-14)

After the WMMA discovery (a silently-broken default), the next kernel-time
leader - the hc up-projection (Q8_0 [2560x640], gemm_q8_j128, ~800ms/chunk) -
was A/B'd the same way: disabling the tile path (LLM170_Q4_NO_TILE) falls to
the GEMV route at 54.8 t/s pp2048 vs 270 for j128. So j128 is genuinely the
best of the available kernels for this shape; its ~2.5 TFLOPS ceiling is a
small-n_in amortization limit (the code comment's own analysis). Beating it
needs a dequant-cache or MMQ-style kernel for Q8_0 at this shape - recorded as
the next prefill lever, not a quick default flip.

### Prefill MoE device grouping: correct but slower - and a production race fixed (2026-09-14)

Finishing plans/68 fixed two defects that matter beyond the experiment:
d2h() and d2h_issue() shared one pinned staging buffer, so any synchronous
read between an async issue and its wait overwrote the MoE offset table with
that read's float payload (fallback row count 1.04e9 = float bits as int,
HIP 700; the t=1 production path carried this race latently - now each has
its own slot), and the grouping kernel left perm_pad slots in [rp, bound&~15)
stale, making the gather index activations out of bounds (the residual
last-token flip). With both fixed the experimental prefill grouping is
bit-correct (gate PASS) but measures 9% slower than the host path
(253 vs 278 t/s pp2048): padded rows inflate the GEMM ~20% and the serial
grouping kernel costs more than the host counting sort. Prefill stays on the
host grouping (LLM170_MOE_GROUP_PF=1 opts in). Methodology note: the earlier
frame-time sync-mark attribution of "49% host cost" included GPU drain -
kernel-time traces are the reliable measure; the next prefill levers are the
dense GEMM kernels themselves (gemm_q8_j128 799ms/chunk, q4_gemm_f32_m 601ms).

### Device-resident QSA KV cache: long-ctx decode 87->82.3 ms (2026-09-14)

The QSA KV cache now lives on the GPU in per-(layer,seq) pools: each step
appends its k/v rows with D2D copies and the attention kernels read the pool
in place, removing a per-layer per-step cache upload (32 MB at 8k context).
A watermark allows sequential appends and same-prefix rewinds, holes fall
back to the upload path. pp8000+tg32 measured 87.0 -> 82.3 ms/step; short-ctx
decode and pp2048 (277 t/s) unchanged; stream gates identical, 12/12 tests.
Debug envs: LLM170_QSA_NORES (A/B), LLM170_QSA_RESCHECK (pool-vs-host compare).

### Device-resident QSA stage: prefill +10-19%, long-ctx decode 91->87 ms (2026-09-14)

plans/67 step 2c landed: the QSA attention stage keeps its projections, norm,
rope, attention and output projection on the GPU; only the indexer inputs and
the cache-append copies (iq/ik/k/v, ~t*3.8k floats) cross to the host.
Two latent defects surfaced while wiring it (both now documented in code):
the frame_qk_norm_rope launch passed kqs=0.0 which would zero every k row, and
the kernel reads norm weights as per-head tiles so the shared [hd] vector must
be tiled before upload (same convention as the decode path's rawinject).
Measured (`bench --ctx 8192 --gpu-runtime hip`): pp2048 233-252 -> 277.75 t/s;
pp8000+tg32 91 -> 87.0 ms/step (the t=1 split-attention path must be routed
through the device-side split kernel pair, otherwise decode regresses to
122.7 ms). Gate: 208-token Korean prompt stream identical, 12/12 tests.

### 27B prefill: the 8.4x HIP gap was one broken kernel default - CLOSED (2026-09-14)

Follow-up measurement traced the gap: KTRACE showed only 1.4s of GPU kernels
in a 13.4s pp512 run with 12s attributed as a "gap after kv_f16" - but
launch3_dyn records no kernel events, so that "gap" was the qsa_flash_wmma
attention kernel itself executing. A/B: WMMA 38.2 t/s vs the wk8 scalar tile
364.2 t/s (**9.5x**) on this ROCm/HIP build - the wmma_ok() probe passes
(silent emulation/spill suspected; it was the best variant in the Vulkan era).
WMMA is now opt-in (LLM170_WK_WMMA=1). Result: pp512 38.2 -> **365.1 t/s**,
pp2048 **339.5 t/s**, decode 86.6 ms unchanged; attn-check 50M elements zero
mismatches above 1e-4; gates and 12/12 tests pass. The primary plans/66 goal
(llama.cpp pp512 350-360 parity) now holds on HIP as well. The section below
is kept for the measurement-discipline note.

### 27B prefill: HIP is 8.4x behind the Vulkan backend (2026-09-14) [resolved above]

`bench --pp 512 --ctx 4096`, same binary, same machine, alternating runs:
HIP 38.1-38.5 t/s (13.3-13.4 s) vs Vulkan 321.6 t/s (1.59 s). Decode is NOT
affected (HIP tg32 11.62 t/s = 86 ms/step, matching the recorded 87 ms).
Cause: the 27B prefill optimizations from the Vulkan era (MMQ routing, tile
kernels, f32 family - see the 2026-09-05/06 entries) live in `rawvk`; the
`rawhip` prefill path for the dense 27B model was never ported. Verified not a
regression of this session: `git stash` A/B on the warning cleanup reproduces
38 t/s on both sides. Note for tracking numbers: "pp512 352-360 t/s" in older
notes is a Vulkan-era figure - always record the runtime with a pp number.
Reproduce: `scripts/gate-27b.sh --bench` then `RUNTIME=vulkan scripts/gate-27b.sh --bench`.

### The residual idle is per-kernel-transition, ~60 us each (pp512, current build)

Re-profiling after the host-side changes (thread caps, parallel pass A):

| | before | now |
|---|---|---|
| kernels in the 512-token prefill | 8,694 | 7,454 |
| GPU busy | 80 % | **87 %** |
| gaps | 611 ms (25 %) | **363 ms (18 %)** |

The gaps are no longer a few long stalls: **6,077 of the ~7,450 transitions have
a gap, averaging 60 us** (largest 11 ms). With ~1 us host launches, that 60 us is
device-side start latency per kernel, so **kernel count is the currency** - and
it explains why moving work to the GPU without removing a launch (the indexer
score experiment) loses. Fusion targets ranked by transition count per layer:
`gemm_q8_0` 33.4 (one launch per dense matrix), `quant_q8` 22.7 (per-GEMM
activation quantize), `q4_rows_permute_u32` 13.8 (MoE gather/scatter, fusable
into the grouped GEMM via the perm it already carries - bit-identical),
`gemm_q8_j128` 12.1, `rms_part`+`rms_finish` 8 (2 launches per norm).

### The prefill is block-dispatch-bound (rocprofv3, pp512 + probe)

A narrow rocprofv3 window (pp512 prefill only - the trace's window must exclude
the ~35 s model load or the numbers are meaningless) shows:

| | |
|---|---|
| kernels in the 512-token prefill | **8,694** (~180 launches per layer) |
| window span | 2,477 ms (bench prefill: 2,454 ms) |
| GPU busy (sum of kernel durations) | 1,984 ms (80 %) |
| GPU idle (gaps) | 611 ms (25 %) |
| launches | ~181 per layer (8,694 / 48) |

**The host launch is not the cost.** A dedicated probe
(`llm170 launch-rate N`, `rawhip::launch_rate`) launches a trivial kernel N times
and reports the host rate for several grid sizes:

| grid | host us/launch | GPU tail per launch |
|---|---|---|
| (1,1,1) | 0.9 | 1.1 us |
| (64,2560,1) - 164k blocks | 0.9 | **124.5 us** |
| (2048,1,1) | 1.0 | 2.5 us |
| (65535,1,1) | 0.9 | 49.7 us |

The host issues a launch in **~1 us regardless of the grid**, but the *device*
spends **124 us dispatching 164k empty blocks** (the kernel body never runs -
`q4_scale` bounds-checks and returns). The device dispatches roughly one block
per cycle (~0.75 ns), so **block count is time**: the engine is
block-dispatch-bound, not launch-bound, and fewer/fatter blocks is the lever.
The top kernels are all over-parallelised:

| kernel | grid | blocks | note |
|---|---|---|---|
| `q4_l2_rows` | (1048576,1,1) x32 | **1,048,576** | one 128-wide row per block; the row count comes from the *max-t* buffer (`flen/d`), of which only ~t are live |
| `q4_gemm_q4k_ge` | 40x320 x256 (measured: n_in=2560, n_out=640, rows=5120) | 12,800 | **efficient** - 8.4 GFLOP in 1,949 us = 4.4 TFLOPS int8 |
| `gemm_q8_j128` | (20480,1,4) x256 | 81,920 | 128x128 tiles, the healthy reference |

An earlier reading of this table attributed a (10240,320) grid to `q4_gemm_q4k_ge`
and called it dispatch-dominated; instrumenting the launcher shows the real
shape (n_in=2560, n_out=640, rows=5120 -> 40x320 blocks) and the kernel runs at
4.4 TFLOPS, i.e. it is *not* a target. The f32 router GEMM is the outlier: its
grid is sized correctly (gx = n_out/16, gy = t/16), so the ~1.3 TFLOPS is the
kernel's inner loop. Reading it explains why: each thread computes one output
with a full `for (k = 0; k < n_in; k++) acc += xr[k] * wr[k];` where the 16
threads sharing a row read `w + o * n_in + k` - addresses 10 KB apart, one cache
line per thread per step, and one FMA per 8 bytes loaded. **That diagnosis was tested and refuted.** A transposed-weight kernel
(`q4_gemm_f32_mt`: lane = contiguous `o`, so the warp reads 64 B per k step) was
implemented and verified bit-identical, and it was *slower*: pp2048 9,814 ->
10,067 ms (+2.6 %). The old layout is one cache line per thread per step, but
each thread walks its row **sequentially**, which the hardware prefetches
perfectly; the transposed layout strides 2,560 B per k step and defeats
prefetch. Coalescing is not the metric here - per-thread sequentiality is.

What the ~1.3 TFLOPS actually is then remains open (the loop is 1 FMA per 8
bytes over 2,560 sequential steps, so it should be compute/MLP-bound, not
line-count-bound).

The tile path is load-bearing: forcing the GEMV fallback (`LLM170_Q4_NO_TILE=1`)
slows the 512-token prefill from 2,420 ms to 8,450 ms (3.5x).
The launches still break down as follows (per layer of the 48; GPU time is the
sum over the 512-token prefill) - the counts matter as much as the times,
because each one re-dispatches its grid:

| kernel | launches/layer | GPU ms | us each |
|---|---|---|---|
| `gemm_q8_0` | **33.4** | 154 | 96 |
| `quant_q8` | 22.7 | 48 | 44 |
| `q4_rows_permute_u32` | 13.8 | 43 | 66 |
| `gemm_q8_j128` | 12.1 | 262 | 452 |
| `__amd_rocclr_copyBuffer` (h2d/d2h) | 11.6 | 0.9 | 1.6 |
| `copy_rows` | 10.0 | 0.5 | 1.0 |
| `gemm_q5k` | 9.1 | 9 | 20 |
| `q4_gemm_f32_m` / `q4_gemm_f32` | 6.0 + 6.0 | 181 + 9 | 629 / 32 |
| `rms_part` + `rms_finish` | 4.0 + 4.0 | 38 | 97 |
| `q4_hc_gate_mean` / `q4_hc_combine` | 4.0 / 4.0 | 24 / 13 | 124 / 69 |
| `silu_mul` / `q4_silu_div` | 4.0 / 4.0 | 6 / 0.4 | 32 / 2 |
| `q4_gemm_q4k_ge` | 3.9 | 366 | 1949 |

`matmul_group`/`matmul_batch` batch only the *synchronisation*, not the launch:
each matrix gets its own kernel, which is why a layer issues 33 `gemm_q8_0`
launches. Two cheap classes stand out: the per-matrix dense GEMMs (group them
with a tile->matrix table, the pattern `q4_gemm_q4k_ge` already uses for
experts) and the sub-3-us kernels (`copy_rows`, `q4_silu_div`, the h2d/d2h
copies: 26 launches/layer doing ~1 us of work each). This is the ~31 s of non-kernel time in the
60.5 s prefilling (kernel sum ~26-29 s). The frame's fine-grained op structure
(~180 launches per layer) costs more than the GPU work it schedules, so **op
fusion - not kernel tuning - is the next lever**. Reproduce with:

```
rocprofv3 --log-level error --kernel-trace -d /tmp/prof -o q4 -- \
  ./target/release/llm170 bench --model <flash-next> --pp 512 --tg 0 \
  --ctx 4096 --backend gpu --gpu-runtime hip
```

(`--log-level error` is required: rocprofv3 logs every child process to stderr
and `hanzo-cubecl-hip-sys` panics when a `hipconfig` call produces any stderr.)

### The attention kernel was the largest kernel (traffic-bound)

KTRACE at pp11750: total kernel time 45.0 s of the 76.4 s wall, of which
**`q4_qsa_attn_sel` is 17.9 s (72 launches, 248 ms each)** - 40 % of all kernel
time. The selection change removed the *iterations* (1.5 s) but not the
*traffic*: every (token, head) pair reads its selected K/V rows independently,
i.e. 2048 tokens x 24 heads x ~2051 keys x 256 dims x 2 (K and V) x 4 B ~ 206 GB
per chunk, at ~830 GB/s effective - well past the LPDDR5x rate, so the L2 is
doing the heavy lifting and is the limit.

The fix is the classic one: stage K/V rows in shared memory once per block and
let many query heads of the same KV head read them from there (the 24 query
heads map to 2 KV heads, so a 12x traffic reduction is available), i.e. a
flash-style block over (tokens x head-group) rather than one warp per
(token, head). Expected ~15 s per chunk = ~20 % of the prefill.

Decode: **156 ms/step at 2,048 context** (6.41 t/s) and ~196 ms/step at 8,192.

Per-op breakdown of one decode step (t=1, `LLM170_FRAME_TIME=1`, ~177 ms measured
with the sync-per-tag instrumentation; the tracer reads back one element per
tag so it is itself a sync point):

| tag | ms/step | count | | tag | ms/step | count |
|---|---|---|---|---|---|---|
| qsa_bridge | 44.7 | 12 | | out | 5.5 | 36 |
| shared | 26.9 | 48 | | route | 4.1 | 48 |
| l2scale | 24.7 | 36 | | ar | 1.6 | 36 |
| top10 | 17.0 | 48 | | gate | 1.4 | 96 |
| mm_group | 12.3 | 36 | | silu | 1.3 | 96 |
| rms | 10.0 | 96 | | ffn_combine | 0.7 | 48 |
| down | 8.5 | 96 | | normgated | 0.7 | 36 |
| ple_bridge | 7.3 | 1 | | conv | 0.7 | 36 |
| up | 6.8 | 96 | | rest | ~3 | ~250 |

There is no single dominant item: ~700 ops cost ~0.25 ms each, so the decode is
spread across the whole frame op set (the QSA bridge is the largest single tag
at 25 %). Cutting it means fewer ops/syncs per step, not a faster kernel.

Two negative results recorded here so they are not retried blind:
- giving `L2Rows` the current row count instead of the buffer's (`flen/d`, which
  is the max t) removed ~2000x of the work with **identical tokens** but made the
  step **slower** (232 vs 195 ms). The extra rows are stale, so their values feed
  whatever reads the tail of the buffer; the effect is unexplained and worth
  isolating.
- batching `q4_l2_rows` 8 rows per block (bit-identical) changed nothing
  (196.4 vs 194.8 ms), so that 24.7 ms tag is real work, not block-launch cost.

The decode runs the same kernel at t=1, where only 24 warps exist (one per head).
Raising that concurrency by launching one 32-thread block per token (adding a
`tpb` argument to `q4_qsa_attn_sel`; verified bit-identical by the probe at
t=1 and t=256) made it **worse** - 274 vs 195 ms/step - so the limit is not
simple warp occupancy, and that change was reverted. The block's early-exit
warps (15 of 16 return immediately) appear to help rather than hurt. A split-K
over the selection list (like the value path's `qsa_flash_split4q4` + merge) is
still the untried option, but it changes the softmax rounding order, so it needs
its own baseline - it cannot be validated by token identity.

Two consecutive runs measured 79,730.4 and 79,703.7 ms. The earlier 117.1 s row
in the history above is **not reproducible today** under identical flags; the
only code change in between (the grouped q5_1 MoE path, `57215df`) was measured
A/B at 11,750 and is neutral there (79,685.7 ms off vs 79,703.7 ms on), so the
difference is not attributable to it and remains unexplained - treat 79.7 s as
the current figure.

Kernel time (KTRACE) totals 46.1 s of the 78.9 s wall, so ~32.8 s is outside
the traced kernel set. That gap is the largest single item left before the
kernel table, and it is now located: it is the pipeline drain/fill around the
**QSA host bridge**.

Direct instrumentation of the bridge (`LLM170_Q4_TIME=1`, steady t=2048
layers, per layer) - the two independent timers (the bridge's and the stage's
own) agree once both are filtered to `t=2048`:

| part | per layer | per chunk (12 QSA layers) |
|---|---|---|
| `frame_read` (21 MB d2h + drain) | 77-115 ms | ~1.0 s |
| `stages::qsa_layer` | **230-255 ms** | **~2.9 s** |
| - of which the attention lap | 93-97 ms | ~1.1 s |
| - of which the projection group | ~12 ms | ~0.15 s |
| `frame_write` (21 MB h2d) | 2.1 ms | ~0.03 s |

so the bridge costs ~3.9 s per 2048-token chunk, i.e. **~40 % of the 10.0 s
chunk** - consistent with the 3,370 ms/chunk that removing it saves
(`LLM170_STAGE_SKIP=qsa` drops pp2048 from 10,017 to 6,647 ms; that diagnostic is
timing-only since the output is invalid). Note: an earlier reading of 30 ms/layer
came from the t=1 warmup lines of the same counter, not from t=2048.

The 230-402 ms is dominated by the CPU-side quantized projections (q/k/v/iq/ik
plus the output projection, ~82 GFLOP/chunk/layer at that width); the indexer
selection and the attention are the smaller part. Moving the projections to the
existing frame GEMM path on the device is therefore the direct win (~2.9 s per
chunk = ~29 % of the prefill), and it does not require new indexer kernels -
only registering the QSA weights in the frame and reading back the small
projection outputs (64 MB d2h ≈ 3 ms).

The bridge is `crates/core/src/qwen4exp/frame.rs:414`: it reads the layer's
`mix` to the host (21 MB d2h), converts it to `Vec<Vec<f32>>`, runs the CPU
`stages::qsa_layer`, and writes the result back (21 MB h2d). The d2h forces a
full device sync every QSA layer, so the GPU drains and then waits for the host
- the span that `FRAME_TIME` attributes to `qsa_bridge` is that drain/fill.

The fix is to make the QSA layer device-native (its projections are ordinary
GEMMs the frame already has kernels for; the indexer's pool/RMS/rope, block
scores and top-k need kernels - the decode path already has the value-attention
side: `qsa_score`, `qsa_mix2`, `qsa_flash_wk16`).

### 2026-09-13 — prefill stage breakdown (frame path, `LLM170_FRAME_TIME=1`)

2311 tokens (chunks of 512), 48 layers, wall-clock per stage as measured by the
frame's sync markers:

| Stage | chunk 1 | chunks 2-4 (steady) | share of 42.6 s total |
|---|---|---|---|
| `moe.gemm3` (gather + 3 grouped expert GEMMs) | 17.8 s | ~1.7 s each | **56 %** |
| `qsa_bridge` (12 QSA layers) | 0.87 s | 1.35 -> 3.23 s | **24 %** |
| MoE route/shared/out/down/up/group | — | ~0.9 s | ~10 % |
| GDN AR + conv + comb | — | ~0.08 s | ~2 % |

Two levers dominate:

1. **Expert GEMM** — grouped (token, expert) gather + 3 stacked GEMMs; the first
   chunk pays a one-time ~16 s (weight streaming) that the server load does not
   cover, afterwards ~35 ms/layer at t=512.
2. **QSA attention bridge** — `qsa_attn_raw` uploads q/k/v/mask and downloads the
   output *per layer* (host round trip) and the kernel attends densely over the
   context, so its cost grows with n_past while llama's paged/top-k sparse path
   grows with the selected blocks.

### 2026-09-13 — long-context prefill split (11,750 tokens)

`LLM170_FRAME_TIME=1` + `LLM170_Q4_TIME=1`, frame path, 24 chunks of 512:

| Stage | 2,311 tok | 11,750 tok |
|---|---|---|
| frame total | 42.6 s | 191.1 s |
| `qsa_bridge` (12 layers) | 10.2 s | **107.7 s (56 %)** |
| `moe.gemm3` | 24.1 s (17.8 s of it in the cold first chunk) | 58.9 s (31 %) |
| QSA `sel+proj` (CPU indexer) | 2.9 s | **51.6 s** |
| QSA `attn` (kernel + PCIe) | 6.3 s | 50.5 s |

The indexer's cost matches a scalar MAC estimate (210 G MAC ≈ 52 s), so it is
CPU-bound, and it grows with rows x blocks (context-quadratic). Parallelizing it
per row with `std::thread::scope` **regressed** (51.6 -> 69.7 s) because thread
spawn cost dominates at that granularity; it was reverted. The next lever is a
device-side indexer (block scores are a batched matrix product).

### 2026-09-13 — prefill optimizations landed (QSA indices + attention)

Two changes, each verified by identical tokens (760/6511/314/1002/55) and the
full workspace test suite:

| Stage @11,750 tok | before | after |
|---|---|---|
| QSA `sel+proj` (indexer, CPU) | 51.6 s | **10.0 s** |
| QSA `attn` (kernel) | 50.5 s | **30.5 s** |
| QSA total | 107.3 s | 50.0 s |
| prefill wall clock (incl. load) | 213.9 s | **117.1 s** |

- Indexer: the per-row clone of the block-key cache was hoisted to a local
  buffer, the full sort became a partial select, and the dot was unrolled to 4
  accumulators; the per-row work then moved into a parallel pass (one scoped
  thread group per layer-chunk; per-row spawning regressed and was rejected).
- The f32 GEMM (the MoE router, `ffn_gate_inp`) now defaults to its tiled/MMQ
  variant: 924 ms -> 615 ms of GPU time for a 2048-token prefill, prefill
  121.3 s -> 117.1 s at 11,750 tokens, tokens unchanged.
- The offline tile kernels (`w32b.co`, `v4all.co`) gained a **token quadrant**
  axis (`blockIdx.z`), so a prefill chunk of N tokens is one GEMM launch per
  matrix instead of N/128 pieces. Each piece previously re-read the whole weight
  matrix, a 16x over-read for a 2048-token chunk. `gemm_q8_j128` went from
  2049 ms in 8652 launches to 760 ms in 781; total prefill 129.3 s -> 121.3 s at
  11,750 tokens. Fixed sources: `plans/i8_arc/co_src/{wm32_q,v4all_q}.cu`
  (built with `scripts/build_co.py`).
- The prefill chunk is now chosen by device memory (the frame buffers scale with
  it): `>= 48 GB` -> 2048 tokens, `>= 16 GB` -> 1024, else 512. Larger chunks put
  more rows on each expert in the MoE GEMM, so weights are reused more.
  Measured at 11,750 tokens (prefill incl. load): 155.2 s @512, 146.4 s @1024,
  **141.0 s @2048**; `moe.gemm3` 58.9 -> 45.2 s. `LLM170_FRAME_TMAX` and
  `LLM170_Q4_CHUNK` override; the 8 GB CMP keeps 512 (1024 OOM'd there).
- The QSA attention kernel is now warp-per-token: one warp per (token, head),
  lanes cover the head dimension for one key at a time, K rows are shared by the
  8 tokens of the block, and there are no block barriers. The mirror check
  (`q4-qsa-check` vs a CPU reference) reads 2.263e-4, the same as the previous
  kernel, and tokens are unchanged. (A first attempt with lanes over *keys*
  summed partials of different keys; the mirror caught it before it landed.)
- Attention: the shared dot product in `q4_qsa_attn` used a single accumulator
  and was latency-bound (4-accumulator unroll, 1.5-2.2x); the score phase then
  moved to warp-per-key with lane=dim so its loads are coalesced (a further
  1.13-1.33x). An "iterate only the selected keys" variant was implemented,
  measured neutral, and reverted.

Remaining at 11,750: `attn` 34.3 s, expert `moe.gemm3` ~44 s (of which ~18 s is
the one-time expert-weight upload in the first chunk), `sel+proj` 10 s.

### What is on the GPU now (plans/64 P1)

- **Value path** (`rawhip/q4acc.rs`): `matmul`/`matmul_batch`/`matmul_group`
  (one activation upload + q8 quantize per group), `moe_down` (expert-stack
  GEMM grouped by ids), `qsa_attention` (masked dense GQA bridge). Weights
  upload once per mmap pointer and stay resident; bf16/f16 weights expand to
  f32 on upload. New kernels: **q5_1** (25.2 GiB of expert-down weights had no
  GPU kernel at all), **f32** (MoE router), masked **QSA attention**.
- **Frame (decode)**: `Frame4`'s op set — 16 `FrameOp` variants plus
  `frame_mm`/`frame_mm_group`/`frame_moe_gemm`/`frame_gdn_ar` — is implemented
  on rawhip, so the decode step keeps activations and GDN state on the device
  (kernels chained by handle, ~14 syncs/step instead of ~1300). New kernels:
  `q4_rms`-family reuse, `q4_silu_div`, `q4_sigmoid`, `q4_scale`, `q4_l2_rows`,
  `q4_hc_gate_mean`, `q4_hc_combine`, `q4_norm_gated_sig` (qwen4exp's sigmoid
  z-gate), `q4_moe_top10`, `q4_moe_weighted_sum`, `q4_gdn_ar_w`.
- **Loader**: `Model4::part_sources()` + pread staging. The mmap fault path
  read at 20-180 MB/s; buffered pread reads the same file at 1.2 GB/s
  (4.5 GB/s O_DIRECT ceiling), which is what makes the 35 s load possible
  (llama.cpp needs a fork patch for the same reason).

### 2026-09-13 — attention restored, frame prefill promoted

Two defects were found while investigating the "QSA t>128 kernel defect"; both
changed the numbers in the table above.

1. **The QSA prefill attention was not running at all.** The guard made
   `qsa_attention` return `Err` for t>128, but the caller's fallback only set a
   flag: the per-token loop had already `continue`d past the CPU attention, so
   the attention rows stayed empty and all 12 QSA layers contributed zero
   attention for the whole prefill. Both the frame and value paths degraded
   identically, so the frame==value self-consistency check passed. The kernel
   itself was fine: `llm170 q4-qsa-check` matches a CPU mirror at t=129/200/512
   (n_past 512) to 2.0e-5 with no non-finite outputs. Fixed by extracting
   `cpu_attn_row()` and actually recomputing on failure; the guard is now only
   `LLM170_QSA_CPU=1`. Cross-check: frame == value == forced CPU fallback =
   760/6511/314/1002 on a 230-token prompt (previously 271/248068/198, i.e. the
   attention-free stream).
2. **`q4_hc_combine` wrote out of range for hc>1.** The kernel is one thread
   per (token, dim) but used the op's `total` (= hc*n*t) as its grid/limit, so
   the frame prefill faulted (sticky 700) at t>~200. Isolated with
   `llm170 q4-hc-check` (which reproduces the fault for hc>1 at any t/n) and
   fixed. The device-resident frame prefill is now the default; its t_max is
   capped at 512 (t_max buffers are ~0.8 GB and 1024 failed hipMalloc).

Frame-vs-value A/B on the same prompt is token-identical at 230 (chunk 512),
300 (chunk 128, three chunks - so the cross-chunk GDN/conv state is right) and
700 tokens, with the attention active.

Prefill is now 6-8x the value path and still **kernel-bound**: with
`LLM170_FRAME_TIME=1` a 512-token chunk is 6.5 s, split MoE 63% / RmsRows 23% /
QSA bridge 10% / hc projections 4% / everything else <2%.

Two fixes carried the 2026-09-13 numbers:

- **MoE expert grouping**: the router emits ids in probability order, so the
  "contiguous run" GEMM loop ran ~4000 one-row launches per layer. Counting
  sorting by expert cut runs to the expert count (<= 256): pp512 36.9 -> 49.0.
- **Transposed GDN AR state**: `q4_gdn_ar_w` read the state one column at a
  time (kdim striding by d=128 floats), so every 4-byte load pulled a 128-byte
  sector - the state alone was ~148 GB per chunk, i.e. 3.7 s at the measured
  39.6 GB/s, matching the 3.97 s the AR stage cost. qwen35's
  `gdn_ar_w_swap` (transposed, d=128) carries over unchanged: AR 3969 -> 73 ms
  (54x) and pp512 49.0 -> 77.4.

The PLE bridge (2 layers, 626 ms each) was hidden in the RmsRows interval
until it got its own mark; its per-token host block (three RMS norms, the
gate, a broadcast and a residual pass that recomputes the gate, with two
allocations per token) is now split across threads - bit-identical, PLE
1253 -> 1050 ms. The serial mmap gather and the two value-path projection
bridges remain.

Resolved: RmsRows was never slow (1.4 ms per call after the mark moved the
PLE bridge out of its interval) - the earlier 15.5 ms and the failed
coalescing experiment were both misattribution.

Kernel-level tracing (`LLM170_KTRACE=1` on the bench's prefill) named the
remaining cost: `q4_gemm_q5_1` held 2715 ms of a 5066 ms chunk (54%). The MoE
expert-down weights are q5_1 (25.2 GiB) and q5_1 had no tile kernel, so
`launch_gemm` fell through to a GEMV-shaped kernel whose block is (row, output)
- every row of an expert group re-read the same weight row, 20x at t=512.
`q4_gemm_q5_1_t` (16-row x 4-output tile, per-row arithmetic order unchanged)
is bit-exact against the CPU mirror (`q4-acc-check`, 100% at t=20/64) and took
2715 -> 1980 ms; pp512 80.0 -> 86.4. It is now latency-bound rather than
tiling-bound (16-row x 1-output, 32-row and 16-row x 4-output all land within
1932-2067 ms at ~7.3 GB/s effective, ~27x off the DRAM floor), so the next
step is a j128-class structure with shared-staged activations - that family is
precompiled offline (.co), so the build pipeline needs checking first.

The PLE bridge's remaining serial part - the per-token mmap gather - is now
split across threads too (ple_table_view() resolves the table once so the pure
ple_gather_parts() can be called from scoped threads; Model4::ple_gather itself
cannot be shared because of its RefCell cache): ple_bridge 1011 -> 113 ms and
pp512 86.4 -> 103.7 t/s, tokens unchanged. pp512 now stands at 2.8x the
attention-correct baseline measured at the start of the session.

**q5_1 became the default tile (2026-09-13, biggest single win of the session).**
The MoE expert-down weights are q5_1 and the kernel that served them was a
GEMV-shaped one whose block is (row, output), so every row of an expert group
re-read the same weight row. An isolation harness (`llm170 q5-1-bench`, the
synthetic shape with hot data) showed the kernel itself was the limit at
6.3 GB/s, and all structural variants landed within 1932-2067 ms until one
thing changed: how many bytes each thread reads. The original read ~1 byte per
thread (pure latency), a 16-row tile 16 bytes, and the new `q4_gemm_q5_1_m`
(16 outputs x 16 rows per block, one (output,row) pair per thread accumulated
over k serially, weights staged once in shared) reads hundreds.

It changes the accumulation order, so it is verified the way llama.cpp and
vLLM verify kernels - by tolerance, not bit-exactness:

- `q4-acc-check` vs the CPU W4A8 mirror: max_abs 5.96e-8, max_rel 1.3-2.1e-5 at
  t=20/64/128 - about 1/500 of the q5_1 quantization error itself (~1e-2).
- Greedy token streams are identical on both the 230- and 700-token prompts,
  and identical between the new path and the bit-exact one.
- `LLM170_Q5_1_EXACT=1` restores the bit-exact tile.

Result: the kernel 1932 -> 316 ms per chunk (6x), pp512 104.1 -> 137.8 (+32%),
pp2311 72.2 -> 86.3 (+20%).

For reference, in the local sources: llama.cpp's test-backend-ops compares
backends by nmse with bounds calibrated from the quantization error, and runs
q5_1 through MMQ; vLLM's kernel tests use assert_close(rtol, atol) - neither
demands bit-exactness across kernels.

Known outlier: RmsRows costs 15.5 ms per call (96 per chunk, 5.2 M elements
each) = ~336 M elements/s, about 1/13 of the measured transfer bandwidth. A
coalesced shared-staging variant measured no change (bit-identical,
76.3 t/s), so the cause is not the load pattern; an isolated micro-benchmark
over rows/n is the next step.

### Verification (2026-09-12)

- Token streams: the GPU value path, the GPU frame path and the CPU W4A8
  reference (`LLM170_W4A8=1`) agree token-for-token on the real model
  (279 3516 4042 369 6312 11 414 707 13 1116 864, greedy, ctx 4096) and on the
  tiny4 synthetic (frame == value == CPU-W4A8). tiny4's f32 CPU stream differs
  by design — the GPU paths are the W4A8 numeric class, like the qwen35 raw
  decoder.
- `llm170 q4-acc-check <model> <tensor> [t] [rows]`: GPU vs the CPU W4A8 lane
  mirror — q8_0 / q4_K / q5_1 **bit-identical**, bf16 / f32 ≤1.2e-7 (reduction
  order only). This probe caught a real q5_1 defect (the high-bit mask already
  carries the ×16 weight; the kernel shifted it a second time).
- `llm170 q4-ar-check [t]`: the frame's GDN AR kernel vs `gdn_ar_batch` —
  out rel 1.2e-5, state rel 2.6e-5 at production dims
  (n_group 16 / dt_rank 48 / d_state 128) for t=1..3.

Long prompts: the value path and the opt-in frame prefill both handle a
200-token prompt correctly (`271 248068 198 760 1156`, identical streams) after
the j128 tile defect was worked around (tile dispatches are now sliced to
≤128 tokens: the j128 CO faults with >1 token quadrant for n_in=6144 shapes
such as ssm_out; the GEMV path is bit-identical, other n_in values tolerate it).

Prefill is flat in prompt length (11.0-11.2 t/s from 256 to 2311 tokens), which
matches the per-token host cost the stage timing shows — it is not an
attention-quadratic effect.

### Open

- **Prefill is the remaining gap** (11 t/s vs 178-266): the value path keeps
  activations on the host, so pp512 spends 16.6 s in hc + 16.6 s in MoE +
  11.2 s in GDN of a 45.6 s pass — host elementwise work and per-op transfers,
  not GEMM. The frame-based prefill (`LLM170_FRAME_PREFILL=1`, opt-in) is
  token-correct at 200 tokens but cannot yet run pp512: its t_max-sized buffers
  (~600 MiB) fail to allocate on this carve (`hipMalloc: 2`), and with a
  512-token chunk it faults elsewhere.
- Decode is 10.7 t/s against llama's 15.7 solo (0.68x): the remaining cost is
  the per-layer round trips that the frame has not yet removed (QSA/PLE value
  bridges, module-level launches).



## Verification status

Method: greedy token-stream comparison against llama.cpp under the
near-tie standard (ADR-0012). **Two-phase protocol (2026-09-06)**: the
reference server (-ngl 0, CPU) collects all baselines first, then stops,
then this engine runs solo — coexistence was measured to evict our
mmap'd weight pages, serializing the h2d upload into cold disk reads
(13 GB @ ~80 MB/s → 5-min loads; strace: 7 s syscalls / 240 s wall).

### Goal matrix (2026-09-06, GPU/rawhip, `scripts/verify.py` 2-phase)

- Text vs llama-server: single_short/ko/code (2 tie + 1 exact),
  np4_seq0-3 (exact), long (~2300 tok) exact, long_np2 exact ×2
  (the former reference instability resolved by slot-erase collection),
  long_np4 (4 unequal lengths, 1705-2302 tok) exact ×4, long_gen96 tie.
  **All 15 llama-referenced cases PASS.**
- MTP speculative invariants (spec == non-spec greedy): spec_short,
  spec_np4, spec_long, spec_long_np4 — all exact.
  Fixed this session: np×MTP merged verify cross-sequence conv-ring
  contamination (see git cc243c9) — previously emitted tokens outside
  llama's top-6.
- Server (`llm170 serve`) continuous batching: 4 concurrent long
  completions == CLI np4, exact, both plain and `--spec 4`
  (`scripts/verify_serve.py` 2/2).
- Vision (`scripts/verify_vl.py` 5/5): vl_spec_short exact;
  vl_spec_np2 tie-adjudicated (batch-shape rounding flips a 0.015-gap
  flat point — ADR-0012 class, evidence via verify-path top-8 logits);
  vl_np2_isolation exact; vl_spec_long (image + 2302-token prefix)
  exact; semantic: NYT front page read correctly, matching llama.
  ViT deterministic (5/5 hash-identical). CPU-clip vs GPU-vit
  embeddings are not bit-identical — exact CPU↔GPU stream comparison
  is not applicable for vision (documented; gate uses same-engine
  invariants + llama semantics).

### Standing speed (2026-09-06, solo, warm)

| Mode | ours | llama.cpp reference | ratio |
|---|---|---|---|
| pp512 (bench) | 274 cold / 322 warm | 229.9 (server bench, 3314 tok) | 1.40× |
| tg single (natural, 256 tok) | 11.0-11.1 | 10.4-11.6 (server bench) | 0.95-1.06× |
| np4 aggregate | 20.4 | — | — |
| np4 × MTP spec4 aggregate | 22.2 | 15.5 (llama np4+MTP) | **1.43×** |
| single × MTP spec4 | 12.2 (19.4 on healthy host, b544e20) | 15.5 | 0.79× today |

Host note: this session's CPU spent long stretches frequency-parked
(powersave); b544e20's 19.4 t/s single-spec does not reproduce today on
the same binary (12.2-12.3) while the GPU-bound modes hold. MTP
acceptance is healthy (~4-5 tokens/cycle; the "acceptance rate/forward"
stat line divides by verify rows, not cycles).

- qwen4exp (Vulkan, real model): single_ko exact 24/24; single_short /
  long_gen48 near-tie (gap 0.01 nat); single_code near-tie (0.12); np2
  state isolation 25/25 x2. Long single (2,311 tok) and long2 (1,904 tok)
  exact 24/24 each; long+np2 pending device-memory headroom.
- Synthetic tiny4 (`scripts/make_tiny4.py`) — model-volume-independent e2e:
  CPU == GPU 26/26, np2 26/26 x2, long 2000+ 25/25, long+np2 25/25 x2.



