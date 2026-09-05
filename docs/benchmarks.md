# Performance Comparison vs llama.cpp

Conditions always quoted (context / batch / quantization / backend). All
numbers on the dev machine (Radeon 8060S, gfx1151, 32-thread CPU) unless
noted. Relative regression tracking only — absolute cross-machine comparison
is out of scope.

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
- 2026-09-05 (session final): pp512 250→343 t/s (+37%, 0.96× of llama 358 same-moment); tg8 10.44 (0.914× of 11.42).
  Shipped: q6-K MMQ routing (+7.5%), f32 kernel family default (+4%), Vulkan GEMV coalescing (+51%), MMQ 4-path, wk attention, z-grid, rocprof diagnostics.
  Remaining: tg GEMV family rewrite (contiguous-lane transplant triple-confirmed dead: 19/42 gate + 0.71 t/s), pp scattered small items.

## qwen35 — Qwen3.8-27B, UD-Q4_K_XL

Current standing (raw-HIP backend — pure Rust executor, cubecl removed,
single-tenant `llm170 bench`, 2026-09-03). All paths bit-exact against the
CPU W4A8 reference engine (greedy streams identical, incl. 64-token batch
cross-verification against per-token):

| Metric | llama.cpp target | LLM170 | Ratio |
|---|---|---|---|
| Decode tg24, t=1 | 10.4 t/s | **10.4-10.5 t/s** (GPU argmax, logits resident) | 1.00-1.01x |
| Prefill pp64 (chunk 64) | 142.8 t/s | **169 t/s** (88.4 exact mode) | **1.18x** (0.62x) |
| Prefill pp128 (… + full drain swap) | — | **254.6 t/s** (fresh-ref verified) | 0.87x vs llama-bench 294 |
| Prefill pp512 / pp2048 (… + split4q4) | 229.9 t/s (3314 tok) | **228.5 / 184.9 t/s** | ~1.0x / 0.62x |
| Prefill pp512 | ~230 t/s (server-bench) | **~68 t/s** | ~0.30x |

Numerical-quality chain (2026-09-03): f32 full-precision path, W4A8
quantized path, and raw-HIP GPU path produce **identical 16-token greedy
streams** — zero quantization-induced divergence on this benchmark.

Key techniques (kernels are HIP C++ strings JIT-compiled via hipRTC,
arithmetic mirrors `dot_row_w4a8_*_lane` in `crates/core/src/quant.rs`):

- GDN kernels: causal conv fully parallel over (channel, token) — state
  updated by a separate tail kernel; AR recurrence keeps its state slice
  resident in shared memory across the sequential scan, with the state
  update and output passes fused (bit-identical element order).
- Optional WMMA fast mode (LLM170_WMMA=1): all four quant types (q5_K, q4_K, q6_K, iq4_xs) use
  fp16 tensor-core MMA with scales folded into the f16 operands.
  Per-tensor deviation ~4e-4 relative (same numeric class as llama.cpp's
  MMA path); diverges from the bit-exact stream, so it is opt-in — the
  default engine remains bit-exact. WMMA engages only for chunks of 32+ tokens.
- MMQ tile kernels (llama.cpp mul_mat_q structure): 64-row x 16-token
  blocks with cooperatively staged unpacked weights and activations in
  shared memory, thread fragments owning sb%4 sub-blocks — bit-exact via
  paired CPU mirrors. Ownership layout follows llama.cpp's MMQ vec_dot:
  each thread exclusively owns one row x 8 tokens, accumulates in f32
  registers across all k-chunks (no shared partial-sum round-trip), and
  hoists its weight words/scales to registers per sub-block, reusing
  them across every owned token. Covers up to 64 tokens per launch.
- GPU-resident logits with deterministic on-device argmax
  (lowest-index tie-break, identical semantics to the CPU greedy): 8-byte
  readback per token instead of a full vocabulary transfer.
- f32 lane accumulation: consumer-RDNA f64 runs at 1/16 rate and was
  ~80% of GEMV issue bandwidth; all lane partials accumulate in f32
  (mirrors redefined in lockstep — the bit contract is kernel==mirror),
  combined through an f64 tree reduction.
- `__ockl_sdot4` integer dot (i8x4 lanes) for all K-quant GEMV
  (llama.cpp MMVQ analog). Lane-wise u32 subtraction is forbidden
  (borrow crosses lanes) — decompose into separate dot chains instead.
- In-kernel tree reduction (shared half-exchange + 32-wide shuffle tree,
  HIP shuffles cannot cross lane 32) eliminates the partials roundtrip;
  CPU mirror sums lanes in the identical tree order.
- Bit-exact transcendentals: `exp_cr`/`ln_cr` f64-fma polynomials shared
  verbatim between Rust and HIP (glibc `expf` is 0.5-ulp — device `expf`
  differs on ~6% of inputs); `-ffp-contract=off` blocks FMA contraction.
- Batch prefill: gy-dimension kernels for quant/rms/elementwise,
  sequential-state kernels with internal t-loops for conv/AR, tiled GEMM
  (one block per output row, TT=16 token registers, weights read once)
  for the four dominant quant types.

Session progression (same binary lineage, gfx1151):

- First cut: pp64 **2.12 t/s**, tg24 **1.32 t/s**.
- GDN chunk `kkt` precompute kernel (2-stage solve): GDN-chunk-on-GPU
  measures identical to GDN-chunk-on-CPU at pp64 — chunked GDN is **not**
  the prefill bottleneck.
- `gpu-mm` microbench on `ffn_gate` (t=64): ~4.8 GFLOPS vs ~1400 GFLOPS
  llama estimate; GFLOPS is flat ~4 across t in {1,16,64,256} — the k-lane
  kernel re-reads weights per token with zero batch amortization.
  **Prefill priority moved to quantized-GEMM batch throughput.**
- `gemm_q7` (16-token blocks, unrolled register accumulators):
  pp64 2.25 -> **6.49 t/s (2.9x)**.
- `de4` (block-invariant hoisting: one dequant of d/dmin/scales per
  4-element batch instead of per element): pp64 -> **13.04 t/s (5.8x
  cumulative)**, tg24 1.33 -> **2.63 t/s (2.0x)**.
- Per-type GEMV bandwidth (dedicated harness, t=1, 2026-09-02): q3_K
  **18 GB/s**, iq4_xs **67**, q5_K **97**, q8_0 **94 — vs ~161 GB/s
  effective for llama.cpp on the same APU. The earlier "141 GB/s,
  llama-level" rocprof-derived figure was measurement error (queue-wait
  inclusion); the per-type kernel bandwidth above reconciles exactly with
  the 281 ms/token wall (GEMV ~207 ms by type mix + bridges + glue).
  **Decode priority: GEMV kernel bandwidth per quant type** — q3_K
  (`ffn_up`, 35% of wall) first. Concurrent-stream GEMV was measured and
  rejected: aggregate saturates at the single-stream rate (1.00-1.13x).
- `q3_K de4` (block-invariant scale hoisting for q3_K, previously only
  K-quants/iq4_xs — element cost ~16 loads -> ~4.25, value-identical):
  `ffn_up` 18 -> **50 GB/s** isolated; decode tg24 **3.56 -> 3.86 t/s**
  (2026-09-02, minimal-prefill measurement; a preceding 22-min pp512 run
  measurably throttles the APU and masks gains — measure tg with --pp 8).
- W4A8 full rollout (2026-09-02, `LLM170_W4A8=1`): all 7 quant types on the
  integer path (iq4_xs/q3_K/q4_K/q5_K/q6_K/q8_0/iq4_nl), the attention
  CPU bridge removed via an f64-intermediate rms+rope kernel (FMA-contraction
  immune — the technique that unlocked what P1 deemed impossible), and
  prefill GEMMs on weight-amortized batch kernels. Decode **tg24 3.86 ->
  6.96 t/s**; every step verified GPU==CPU greedy-stream identical.
  Prefill W4A8 = 10.26 t/s vs the f32 path's 13.8 — the f32 path is
  faster but CPU/GPU numerically inconsistent (near-tie divergence), so
  W4A8 is kept for correctness; structural prefill work (host round-trips,
  GDN chunk) is the remaining path to 143.
- W4A8 integer-MAC prototype (`gemm_q8i`, 2026-09-02): activations
  quantized to q8 (per-32-block scales), integer accumulation, per-block
  float contributions accumulated in **f64 lane partials** — grouping-
  independent bit-exactness vs a CPU mirror of the same op sequence.
  iq4_xs `ffn_gate`: **146 GB/s, bit-exact on all rows** (2.18x the f32
  path's 67, 91% of llama.cpp's effective rate). Known open issue: the
  `ffn_down` shape (n_in=17408) regresses to 40-61 GB/s — under
  investigation. Engine wiring (on-GPU q8 quantize kernel, q3_K variant,
  frame integration) is the follow-up.

Measurement caution: a co-resident run (llama-server holding VRAM) measured
tg 0.58 t/s — invalid per the non-coexistence rule (see Verification below),
quoted only as a warning.

## qwen4exp — Qwen3.8-Flash-Next 125B-A6B, UD-Q4 4-split

| Metric | llama.cpp (PR #27742 runtime) | LLM170 (GPU) |
|---|---|---|
| Prefill pp, 2311 tok (single) | 178–237 t/s per slot (HIP 7.2.2; 2026-08-27) | **~3.3 t/s** (~699 s incl. load+decode; Vulkan, 2026-09-01 — token-exact 24/24) |
| Decode tg16, real model | 11.6–15.1 t/s per slot (HIP 7.2.2) | value path **0.43 t/s** -> frame **2.87 t/s** (2026-09-02, 6.7x) |

Decode-frame follow-up (2026-09-02): with the GPU-resident frame verified
bit-exact on the synthetic e2e (frame == non-frame == CPU), the real model
went tg16 0.43 -> 2.87 t/s and pp32 5.23 -> 6.01. The frame is now
**default on** (`LLM170_FRAME=0` disables): the default path, no env vars,
re-measured tg16 2.85 t/s / pp32 6.37. Same kernels as the value path,
chained by handle — the host-glue-elimination thesis confirmed empirically.
Remaining qwen4exp gap: the PLE gather and QSA dense attention still cross
to values each step (not yet framed).

Per-step decode breakdown before the frame (value path, `LLM170_Q4_TIME`,
2026-09-01, after same-input projection grouping): MoE ~190 ms (was ~690 —
expert gate/up grouped into one call, down as a paired batch), GDN ~126 ms
(CPU recurrence; the GPU AR kernel landed after), HC ~55 ms, QSA ~18 ms.
Host round-trips dropped from ~1,680/step to ~600/step; the frame takes
that to ~14.

Runtime attribution note (corrected 2026-09-01): the qwen4exp infer path
previously hardcoded the HIP runtime, so earlier "Vulkan" attributions were
wrong; measurements after the fix are runtime-correct. The intermittent
`Memory page N doesn't exist` fault was root-caused to a cubecl-runtime
memory-sweep defect (page reindexing invalidating live handles) and patched
locally — [decisions.md](decisions.md) ADR-0016.

## Verification status

Method: greedy token-stream comparison against llama.cpp under the
near-tie standard (ADR-0012), run **non-coexisting** — the reference server
collects the stream first, then stops, then this engine runs alone (both
servers resident would contend for the same VRAM on this UMA machine).

- qwen35: 11/11 matrix (6 exact + 5 near-tie) on the CPU reference engine
  (2026-09-02). GPU server <-> CLI self-consistency 7/7 exact (2026-08-31
  binary). Two-phase GPU-backend verification (baseline collect, then
  offline judge) is queued; the qwen35 GPU numbers above are `llm170 bench`
  measurements, not verify.py.
- qwen4exp (Vulkan, real model): single_ko exact 24/24; single_short /
  long_gen48 near-tie (gap 0.01 nat); single_code near-tie (0.12); np2
  state isolation 25/25 x2. Long single (2,311 tok) and long2 (1,904 tok)
  exact 24/24 each (immediately preceding binary); long+np2 pending
  device-memory headroom (Vulkan places weights in host GTT on this iGPU).
- Synthetic tiny4 (`scripts/make_tiny4.py`) — model-volume-independent e2e:
  CPU == GPU 26/26, np2 26/26 x2, long 2000+ 25/25, long+np2 25/25 x2
  (every combination; the frame path is hash-identical to the value path).

## Vulkan — FIXED (2026-09-05)

Root cause of the full-model failures was never a driver leak: the sysfs GTT counters are
in **bytes** (the "18.6 GB stranded" was 18.6 MB). The real issue was memory-type
selection — VkCtx picked the first HOST_VISIBLE type (heap 0, GTT-aperture-limited), and
16.3 GB of weights cannot pin in a 15.5 GiB GTT aperture, so the first submit failed with
amdgpu vm_validate. Fix: prefer DEVICE_LOCAL|HOST_VISIBLE (RADV STRIX_HALO heap 1,
74 GiB carveout) for weights, cached GTT type for scratch, and unmap weight buffers
after upload. Full-model Vulkan now runs and the token stream matches the CPU reference
exactly (41/41 ★).

Feature coverage on Vulkan after the fix: np multi-sequence streams match the
np reference (s1 exact; s0 differs only at the known 18/19 near-tie), and MTP
speculative verify lines show live acceptance (OK/MISS) — MTP, np, and the plain
path all function on Vulkan.

Per note: after ~7 h of continuous compute on this 1.5-day-uptime host, the CPU-side
engine degraded ~180x (identical bench command: 9.33 t/s earlier in the session vs
0.05 t/s now; all 24 cores parked at 2.0 GHz, governor=powersave, 43°C — cause unclear,
no external load). Even with the host partially recovered (CPU probe 112s -> 8.4s), Vulkan holds at
1.85 t/s tg / 4 t/s pp32 — so the dominant cost is architectural, not host state:
the VkAcc design submits+waits a fence per matmul (~0.9 ms × ~325 matmuls/token
≈ 290 ms/token of sync overhead; single GEMV itself is healthy at 48.7 GB/s ★).
Command-buffer batching (plans/19) landed with **correct semantics**: per-op fresh
descriptor sets (recorded commands must not share a set — all would observe the last
binding) and write→read memory barriers between dispatches (submit+wait provided
implicit sync that a single command buffer does not). Stream matches the CPU reference
exactly (41/41 ★) and single-GEMV checks stay ★.

Performance is neutral (tg 1.88 t/s vs 1.85 unbatched): an earlier +11% reading was
taken on corrupted output (the first batch implementation lost the descriptor-set
hazard) and is retracted. On this host the barrier cost matches the fence savings;
the dominant costs remain host-side GDN/attention and per-op upload/readback, so the
path to Vulkan throughput is unchanged — GPU residency of the GDN family
(plans/19 phase 2). HIP remains the performance path at 28 t/s np4×spec4.

## Vulkan status (2026-09-05) — superseded

rawvk smoke suite passes (coopmat probe, axpy bit-exact, 25.5 GB/s) — the Vulkan compute
path is healthy. Full-model Vulkan runs are currently blocked by a kernel TTM leak:
18.6 GB of GTT stranded by crashed processes (zero holder PIDs), leaving 11 GB available
against the 16.3 GB weight pin requirement, so the first queue submit fails with
ErrorDeviceLost. The reference llama.cpp Vulkan build fails identically. Kernel log confirms:
`amdgpu_cs_ioctl: Not enough memory for command submission` — GTT total 16.6 GB
with 18.6 GB stranded (counter over total, zero holder processes; lost-device
buffer releases never ran). Full-model Vulkan ran fine earlier in this same
boot (tg 10.4, pp 128), so the leak is the sole blocker. A reboot clears the
stranded GTT; re-verification and the Vulkan MTP port follow.

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
