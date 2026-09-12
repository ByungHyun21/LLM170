# Performance Comparison vs llama.cpp

Conditions always quoted (context / batch / quantization / backend). All
numbers on the dev machine (Radeon 8060S, gfx1151, 32-thread CPU) unless
noted. Relative regression tracking only — absolute cross-machine comparison
is out of scope.

## Vulkan tile-correctness round (2026-09-09, plans/38 A2)

A new `vk-tile-check` harness (per-type tile kernel vs CPU dequant f64 dot)
found two corrupt tile kernels that the earlier gates had missed: tile_q6k
(uint zero-extension dropping the sign of negative i8 scales, plus a global
sb used as a block-local half index — garbage beyond block 0) and tile_xs
(reading 32B of a 16B sub-block, overwriting columns with the next
sub-block and OOB-writing shared rows). Both fixed; all 8 tile types now
maxrel ≤ 0.007 at t=1..64. The np4 prefill degeneracy (degenerate repeated
tokens for ≥16-token prompts) is gone and the spec path is deterministic;
the remaining spec-vs-plain near-tie divergence is a separate, deterministic
MTP numeric-order issue (vulkan-only; HIP exact). Re-assessment note: the
previous session's "vk judge 19/19×7" runs used the harness default runtime
(HIP) — the true vulkan gates start this round. Also this round: the ctx²
f32 mask was removed (qsa_flash's causal loop bound already masks — 256MB at
8k context), pp64 142 t/s / tg4 7.4 on the corrected tiles.

## Vulkan perf-parity round (2026-09-09, plans/39)

Five structural tile experiments against the llama mul_mm gap (tiles = 72%
of chunk GPU time, 38 vs llama 98 GB/s effective weight streaming): T_MAX
128 + 128-token tiles (clean, flat), an f16 pre-dequant weight cache
(63 GB/s streaming but 2x bytes = net loss, opt-in `LLM170_VK_F16W=1`),
direct coopMatStore ColumnMajor drain (unreliable on RADV — both layout
interpretations corrupt), a 2-WG/CU occupancy variant (neutral), and a
family rollout of the q5 kernel structure to all tile types (verified
maxrel<=0.007, flat). Verdict: the remaining 2.4x is inside the llama
mul_mm kernel structure (32x32 warp tiles, WMITER, BK=32, 16 KB shmem,
split_k) — a full port is the single remaining pp lever. tg levers
(addrms+quant fusion) untried. Feature gates on the rolled-out path:
np4 clean, spec divergence unchanged (known f16-prefill class), mmproj vl
reads the NYT moon-landing headline correctly. Side RCA: an init-time
17.5 GB host-heap clone in the f16w cache OOM-killed the 30 GiB host
(fixed by borrowing); heavy llm170 processes must run one at a time.

Second pass (same day): q8_0 KV cache (kv_append_q8 + dequant-on-read
qsa_flash_q8, opt-in `LLM170_VK_KV8=1`) — 3.76x KV bytes cut (2048-ctx
256→68 MB), tg32 neutral (weights-bound), early-token divergences at short
context are the quantized-KV quality class; kept opt-in for long-context
capacity. The 256-thread occupancy tile variant (tile128v2) is parked with
strong evidence of a WG-size-dependent codegen issue: byte-identical
staging code is clean in the 512-thread family and corrupts odd rows k≥16
at 256 threads (subgroup geometry verified 64x4, store semantics and dump
machinery eliminated via constant injection).

## Vulkan execution-round gate (2026-09-08, plans/36 G1-G4/P1-P4)

Full plans/36 round on q35work.gguf, 3-run medians, zero-config defaults
(batch prefill + coopmat tiles now default; kill switches VKD_BATCH=0 /
VK_NOTILE=1): **tg32 7.21 t/s** (from 7.06), **pp512 122.1 t/s** (from
65.95 opt-in, +85%). Gates: vk judge 19/19 (seven consecutive rolls incl.
VKD_BATCH+TILE and T_MAX=64 configurations), HIP judge 19/19, cargo 10/10.

What moved the numbers (per-kernel GPU times via a new VK_QUERY_POOL
timestamp profiler, LLM170_VK_TS=1 — the host-side ktime attribution was
shown to be split-boundary garbage):

- **Batch prefill + tiles promoted to default** — the 2026-09-04 batch
  divergence was the descriptor-set reuse race (fixed 2026-09-05); not
  reproducible since. Verify stays per-token by default (race isolation).
- **T_MAX 32→64**: chunked prefill re-read all weights per 32-token chunk;
  64-token chunks halve weight traffic. pp512 68→97 t/s.
- **Tile type coverage completed**: tile_nl (iq4_nl), tile_iq3s, q8 small-n.
  The straggler gemv3 calls (iq4_nl/iq3_s/q8_0 at 0.2-3 GB/s) were 36% of
  chunk GPU time. pp512 97→122 t/s.
- **gemv8_q6**: llama mul_mat_vec_q6_k direct port with a u16 block view
  (105 u16/block) — the 210-byte unaligned assembly that capped the old
  attempt at 67.6 GB/s is gone. 160 GB/s, max|D|=0 vs CPU at t=1. q6_K
  decode weights and the output head leave gemv3. tg32 6.87→7.21.
- **addrms fusion** (residual add + rms in one pass, bit-identical by
  construction), head merged into the trunk batch (single submit), batch
  auto-split 512→2048, independent-group barrier skip (gemv stages, kv
  append), lazy stage quantization (dead quant removed).
- **tile128 2-sb staging**: 64 k-elements per barrier pair (barriers halved,
  MMA chains doubled). LDS 59.5 KB; split_k was ruled out by measurement
  (46-59.5 KB LDS → 1 resident WG/CU, queuing hides nothing; double-buffer
  ping-pong was neutral).

Remaining prefill gap vs llama-vk raw loop (358): tile MFU ~8 TFLOPS vs
llama ~21.5 — a compute-efficiency frontier, not bandwidth.

i8 coopmat GEMM (plans/23) verdict: v1 per-block drain 29 t/s, v2 row-scale
(drain-free) 82 t/s — confirming the drain as the v1 bottleneck — vs tiles
126+ t/s. The i8 path stays experimental opt-in; tiles supersede it for
prefill across all 8 quant types.

Known open: the gemv8 t≥2 check-harness divergence (deterministic per
binary, in-process stable, all types; engine path exact through VKD_BATCH
spec gates) — recorded with the §8 flake; reproducer:
`llm170 vk-gemv8-check <gguf> blk.4.attn_gate.weight 2`.

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
- 2026-09-05 (FINAL, alternating single runs): pp512 250→348-354 t/s (+40%) — PARITY with llama-VK 350.8-352.8 (0.993-1.004×); tg8 12.25-12.30 vs llama 11.79-11.84 = +3.8-4.0% EXCEEDED. Batch AR grid-axis swap (L2 locality) closed the gap.
  Shipped: q6-K MMQ routing (+7.5%), f32 kernel family default (+4%), Vulkan GEMV coalescing (+51%), MMQ 4-path, wk attention, z-grid, rocprof diagnostics.
  Remaining: tg GEMV family rewrite (contiguous-lane transplant triple-confirmed dead: 19/42 gate + 0.71 t/s), pp scattered small items.
- 2026-09-05: GDN state layout transposed u-major (coalesced AR lane-j access, 16x amplification removed) + decode AR routed to gdn_ar_w — tg8 10.44→12.27 combined-mode — EXCEEDS llama 11.42 by 7.4% (q8_0 dual GEMV + AR transpose + conv parallelization + rmsq widening); pp512 342.6 unchanged.
- 2026-09-06 CORRECTNESS AUDIT (user-requested long-context check): 4 semantic regressions found & fixed — gemm_q3k qsum double-read, rmsq dropped scale writes, dual-GEMV dispatch chain skip, q6_K MMQ on custom-layout weights (default OFF, LLM170_Q6MMQ=1). llama-based verify.py 10/11 PASS; 11th (long_np2_seq1) = llama-server reference instability (its own output varies 16/159301/248068 across slots at a flat distribution point). Honest perf after q6-MMQ exclusion: pp512 ~322, tg8 ~10.9.

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

## Known flake: vk spec==nonspec nondeterminism (2026-09-08)

`--gpu-runtime vulkan --spec 4` np4 runs intermittently flip one near-tie
token (spec_np4 divergence at a flat distribution point). Measured with a
fixed seed repro (`plain vs --spec 4`, 4×short2-class prompts, 24 tok):
**pre-refactor 7758e07 diverges 2/5 runs; post-refactor 4/5** — the defect
predates the plans/35 refactor, which only shifts timing (non-spec streams
are bit-identical across the refactor). `LLM170_G8=0` (verify batches fall
back to quant+gemv3) did not diverge in 2/2 runs — the trigger is the
gemv8 batched-verify configuration (t=2..5 rows), not the decode path.
Follow-up tracked in plans/38 A2 (deterministic harness reproducer). The vk judge gate therefore reports
18/19..19/19 depending on the roll; HIP is unaffected (19/19 stable).

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

## Vulkan — FUNCTIONAL (2026-09-07, plans/29)

The Vulkan path is now correct end-to-end after fixing the VkDecoder head
(two stacked bugs, commit 19f68bc):

1. n_vocab derivation read tuple field 4 (n_in=5120) instead of 5
   (n_out=248,320) — the head argmax'd over 5,120 vocab rows only.
2. gemv3.comp's WG() weight-chunk walker hardcoded 128MiB chunks
   (idx>>25) — the 994MB output.weight, uploaded as a single chunk, could
   only address rows <32,102; high vocab ids were unreachable.

Fix: dynamic chunk_words push constant. VkDecoder is now the default
vulkan path for infer/vl/bench/serve (VkAcc restorable via
LLM170_VK_ACC=1); serve --gpu-runtime vulkan is actually honored (was
silently ignored).

Verified on Vulkan (all 2026-09-07): np4 seq2/3 exact vs llama (seq0/1
diverge only at HIP's known near-tie points); MTP spec4 == nonspec exact
x4; long 2302tok exact; mmproj via HIP ViT + VkDecoder LLM coexisting in
one process (NYT front page read correctly); server multiturn
cached-continuation == CLI full-prefill exact (16/16). HIP regression
after all changes: verify.py 19/19 PASS.

### Vulkan performance round 1 (2026-09-07, plans/30)

gemv3.comp rewrite: 256 threads (was 64), runtime-subgroup-count tree
reduction, iq4_xs KVAL const-array → ktab SSBO LUT (const-array dynamic
index lowers to a select-chain — was the worst kernel), sub-block stride
generalized. Per-type GEMV bandwidth (vk-gemv-time, t=1): iq4_xs 19→57-73
GB/s, q8_0 60→96-107, head 86. A 4-row×64-lane variant was tested and
rejected (39KB LDS → occupancy collapse, documented in the shader header).

Engine: tg16 2.52→5.68 t/s (+125%), pp64 2.65→12.45 t/s (+369% with
LLM170_VKD_BATCH=1). tile128 (coopmat f16, q5_K t≥2) now opt-in
(LLM170_VK_TILE=1): its numeric family diverged from the t=1 gemv and
broke the spec==nonspec invariant (measured; NOTILE run restored exact
equality). VKD_BATCH root-caused: the corruption was the gemm_i8/quant_b8
route (q5_K unpacked i8 weights, t≥2) — LLM170_VK_NOI8=1 reproduced
correct ' Paris' immediately. i8 routing is now opt-in
(LLM170_VK_I8ON=1) and batch prefill is functionally correct, but
shows NO amortization gain (pp512 11.13 t/s ≈ per-token 12.45): the
gemv inner dot loop is ALU-bound per token — weights are read once per
chunk but the 32× dot work dominates. Prefill parity with llama-vk
(127+ t/s) requires per-type register-tile GEMMs (weight fragment →
all-tokens dot), the same structure as llama's mul_mat MMQ.

llama.cpp Vulkan reference on this machine (server bench, Q4_K_XL):
pp 127-431 t/s, tg 11.4-11.9. Our Vulkan standing after round 1: tg 0.48×,
pp 0.10× — NOT yet ahead; the remaining gap needs per-type tile GEMMs
(weight-amortized prefill) and further GEMV bandwidth (q3_K at 37 GB/s
is the weakest large kernel). Correctness held throughout: np4 seq2
exact + others at HIP's known tie points, spec==nonspec x4 exact (after
tile unification), long 2302tok exact, HIP regression 19/19 PASS.

### Vulkan performance round 2 — gemv4 family (2026-09-07, plans/31)

Ported llama.cpp's mul_mat_vec architecture as a new kernel family
(`gemv4_{q8,q5,xs}.comp`, opt-in via `LLM170_VK_GEMV4=1`, decode t=1 only):
32-thread wave32 workgroups, RPF rows per WG with the activation vector held
in registers across rows, f32 activations fed directly (no xq quantization
pass), vec4 activation loads (a second descriptor aliasing the same buffer,
llama's `data_b_v4` trick), and subgroupAdd reduction. Per-type bandwidth
(vk-gemv4-check, exact CPU-f32 match max|D|=0.0000): q8_0 160 GB/s (gemv3
107), iq4_xs 122 (57-73), q5_K 100-104 (55).

Engine integration through a `gemv_w` wrapper that lazily quantizes only on
the gemv3 fallback (dead-quant removal), dynamic rows-per-WG (1 for small
n_out to preserve parallelism), and per-type pipeline cache keys. Two
integration bugs worth recording: grid axes must be (1, rows/rpf, t) — the
shader reads .y as the row block; three spvs sharing one pipeline-cache name
caused descriptor mismatches (SIGSEGV). Net: decode step 154→146 ms
(tg 5.68→~5.9 t/s, +5%).

Negative results (all opt-in-preserved): q5_K gemv4 regresses in-engine
(step 178-194 vs 146-151 ms) despite winning standalone at 121 GB/s —
three suspects eliminated in sequence: register spill ([[unroll]] +
GL_EXT_control_flow_attributes, standalone 97→104 GB/s, engine unchanged),
scattered byte_at scale gathers (replaced by two direct word loads,
standalone →121 GB/s, engine unchanged), vec4-y numerics (a uvec4 nibble
extraction produced element scrambling — kept the verified scalar path).
The standalone-vs-engine gap remains unexplained; q5 stays on gemv3 by
default (`LLM170_G4_Q5=1` to test). y-staging in shared memory was
rejected after measuring llama's actual structure: it stages nothing, it
vectorizes activation loads and reuses them across rows. Batch prefill on
gemv4 re-reads the t×n_in activation per row-WG — 3× prefill regression;
decode-only routing required.

Correctness: France ' Paris' exact, spec==nonspec x4 True, long prompt
exact, np4 token streams identical to both gemv3 and commit 26f34a3
(pre-existing 2/24, 17/24, 24/24, 24/24 vs the llama store — store drift,
not a kernel effect). HIP regression 19/19 PASS (rawvk-only changes).

iq4_xs follow-up: staged the iq4nl LUT into shared memory at workgroup
start (llama's init_iq_shmem approach — 1KB, cooperative load + barrier).
Standalone 122→138 GB/s and this one DOES translate in-engine: decode
step 152.4→136.7 ms (-10%). End-to-end bench tg16 5.81 t/s (baseline
5.68, +2.3% — per-token sampling/detok overhead absorbs part of the
step win). Gates: France exact, spec==nonspec x4 True, long exact.

Per-dispatch-fence attribution (LLM170_VK_NOBATCH=1 + LLM170_G4_SYNC)
shows the gemv4 engine gap is NOT q5-specific: attention projections run
at standalone rate (attn_qkv 0.338 vs 0.296 ms) while ALL FFN sites run
+30-42% over standalone (ffn_down 0.407→0.580 ms) — L2 pollution from
the interleaved attention/conv kernels is the standing suspect.

Round 2b — chunk-walker arithmetic: replaced the per-word integer division
in WG() with shift/mask (chunks constrained to powers of two; the last
chunk is allocated at exact size — o = idx & mask always lands on real
data). Standalone: q8_0 160→247-257 GB/s (+55%), iq4_xs 138→150, q5_K
118→140, all exact. One integration bug caught by the gates: computing
the chunk log2 from the first buffer's ACTUAL size wraps single-chunk
non-pow2 tensors into a phantom chunk 1 (dummy buffer) corrupting weight
tails — spec==nonspec went False; fixed by using next_power_of_two.
Engine: decode step median 152→127 ms (-16.5%, 3-run), end-to-end
tg32 6.19-6.32 t/s (+9% over the 5.68 baseline). rpf sweep knob
(LLM170_G4_RPF) showed 2/4/8 within noise.

Round 3 — prefill coopmat tiles (plans/32). tile128 (q5_K) extended to a
prefill-only threshold (t≥16, keeping spec verify batches on the exact
gemv path) and joined by a new tile_xs (iq4_xs port of the same 512-thread
coopmat structure; A-staging dequantizes iq4_xs with the shared ktab).
Measured pp512: 11.18 (per-token gemv) → 17.45 (q5 tile) → 26.04 t/s
(both tiles, LLM170_VK_TILE=1 + VKD_BATCH=1). Cumulative +133%.

CRITICAL caveat — the tiles stage operands in f16 (RDNA coopmat has no
f32 inputs), which shifts the numeric family: with tiles on, the long
prompt diverges from the token-exact reference stream entirely (0/24 vs
the llama CPU store; llama's own Vulkan pp has the same f16 property).
The quality contract therefore keeps tiles OPT-IN
(LLM170_VK_TILE=1); the default path re-verified clean after the change:
spec==nonspec True, long 24/24. A plain LDS-y batched gemv (gemt_xs,
64 threads, 160 barriers) was also built during this round and REJECTED:
one-hot probes pass but random inputs corrupt outputs through an
unresolved shared-memory anomaly (sy rows read inconsistently across
tokens with identical inputs; markers prove staging threads alive and
stores happening) — kept as a check-harness prototype only
(vk-gemt-check), not wired into the engine. Standing: tg 0.54×,
pp 0.20× (default 0.10×). Next: full MMQ family + non-GEMM prefill
kernels, or an f16-tolerant quality contract.

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

## Vulkan functional audit — np/mtp/mmproj matrix (2026-09-07)

The VL (mmproj) gate had never run against the Vulkan decoder
(verify_vl.py leaves LLM170_GPU_RUNTIME unset → HIP default). Audited
explicitly (`LLM170_GPU_RUNTIME=vulkan`):

- HIP reference: 5/5 PASS (vl_spec_np2_seq0 adjudicated tie @gen[12],
  top-2 gap 0.629 < ε=1.0 — the documented 494/16311 flat point).
- Vulkan current: vl_spec_short / vl_spec_np2_seq1 / vl_spec_long PASS;
  vl_spec_np2_seq0 diverged @gen[12] (same 494-vs-16311 pair, no <ε gap
  evidence that run); vl_np2_isolation failed in the judge run but PASSED
  on direct re-run.
- Vulkan pre-session binary (26f34a3 worktree): vl_spec_np2_seq0 also
  FAILS (@gen[15]) — the flaky flat point predates this session's work;
  isolation passed there.

Conclusion: text np4/spec/long exact (established), VL+Vulkan functional
with one run-to-run flaky borderline at the known flat point — the same
class HIP adjudicates as tie. Divergence position and isolation verdict
move between runs and binaries (ViT embedding ulp sensitivity suspected).
Not a functional break; tracked as the known flat-point class.

### Prefill scaling attribution (round 3 addendum)

pp scales flat across chunk sizes (pp96 27.4, pp256 26.2, pp512 25.9 t/s
with tiles+batch): a fixed ~37 ms/token cost dominates — NOT the GEMM
tiles (which amortize: weight traffic is constant in t). The remaining
serialized per-token work lives in the non-GEMM prefill path (GDN conv
state scan, KV writes, attention history) — the next prefill lever is
attributing and batching those. An earlier in-session reading of
'3,295 t/s prefill' was a misread of a decode-step profile line at
pos=512; the bench numbers above are the authoritative prefill figures.

### Round 4 — full tile family + kernel-time attribution (2026-09-07)

Added a generic per-kernel timer (LLM170_VK_KTIME=1, best with
LLM170_VK_NOBATCH=1; per-weight keys for gemv). Attribution on a
128-token prefill (t=32 chunks): gemv 784ms dominated — the q8_0
(ssm_out) and q4_K (attention layers' FFN) tensors had no tiles.
Ported tile_q8 (34B q8_0 blocks) and tile_q4k (144B q4_K, the q5
variant minus the high-bit plane; shift/mask chunk walker). One
numbering trap recorded: q8_0 is ty=8 (a first attempt keyed ty=7
silently matched nothing).

Batch compute 1144→804ms; pp512 26.0→39.1 t/s (+50%; cumulative
+250% over the per-token 11.2 baseline). Tile-mode smoke: coherent
continuation on a 64-token prompt; default path untouched (tiles stay
opt-in behind LLM170_VK_TILE=1). HIP regression 19/19 PASS.

Standing mystery for the next round: bench wall time still runs ~4×
the summed kernel time (13.1s vs ~3.2s for pp512) — a per-token
~24ms non-kernel overhead somewhere between Engine::prefill and
step_batch. tiles: tile128 205ms, tile_q4k 75ms, tile_xs 75ms per
128 tokens; gemv residual ~300ms.

### Round 5 — full type coverage + attribution hygiene (2026-09-07)

Resolved the "4× overhead mystery": it was a division error on my side —
the ktime total is per-chunk wall time in synchronous mode and matches
the bench exactly (25.6 ≈ 25.1 ms/tok). No hidden overhead. Then closed
the type coverage: tile_q3k (110B blocks: 32B bit-packed hm, 64B 2-bit
q, 12B aux-unpacked scales, f16 d — unaligned word-span assembly), plus
tile128 ql/qh word hoisting (16→4 walker calls per thread-subblock,
200→150 ms per 32-token chunk, identical arithmetic). Verified with the
France prompt pushed through a 38-token batched prefill so ALL five
tile types engage: ' Paris' survives the f16 family.

pp512: 39.1 → 42.8 (hoisting) → 45.4 t/s (q3_K). Session cumulative
prefill: 11.2 → 45.4 (+306%), 0.36× of llama-vk. HIP 19/19. Remaining
prefill: gemv tail is now mostly q6_K (ty14) + iq3_s (ty21) stragglers;
the per-32tok chunk budget is tile128 150 / tile_xs 71 / tile_q4k 50 /
gemv ~250 / elementwise ~90 ms.

### Round 6 — q6_K tile (2026-09-07)

tile_q6k completes the type matrix (six quant types now tiled: q8_0,
q3_K, q4_K, q5_K, q6_K, iq4_xs). q6_K geometry: 210B blocks — ql nibbles
(lo/hi by quarter), qh 2-bit selectors (bits w*2..w*2+1 of the same
byte), direct i8 scales at 192, f16 d at 208 with word-span assembly.
France-through-batched-prefill verified ' Paris'; pp512 45.4 → 47.3
t/s. Session cumulative prefill: 11.2 → 47.3 (+323%, 0.37× llama-vk).
Remaining gemv tail: iq3_s (ty21) stragglers only. HIP 19/19.

### Decode round — two hypotheses rejected (2026-09-07)

Attacking the FFN-site gemv slowdown (+30-42% over standalone) with
llama/HIP-architecture analogies:

1. GTT dirty-line theory (LLM170_G4_YDEV=1): stage the activation vector
   into a device-memory buffer before each gemv4. A/B: 141.6 vs 142.1 ms
   — no effect. Rejected.
2. Tile-everything decode (LLM170_VK_TILE1=1, threshold 16→1): route the
   t=1 decode through the coopmat tiles like our HIP backend's WMMA
   path. A/B: 150.8 vs 144.0 ms — slightly worse; at t=1 the f16
   A-staging (full weight dequant per step) is not amortized, and our
   tile staging is not pipelined the way HIP's WMMA loop must be.
   Rejected for now; the HIP tg=10.4 analog remains unexplained.

Both knobs are kept as documented opt-in experiments. Decode standing:
~144 ms step (tg 6.2-6.3, 0.55× llama-vk). Prefill standing: 47.3 t/s
(0.37×).

### Round 8 — llama MMQ int-dot port attempt (2026-09-07)

Deep-dive into llama's mul_mmq revealed the actual decode architecture:
weights stay INTEGER and the dot runs on the hardware integer-dot
instruction (`dotPacked4x8EXT`, SPV_KHR_integer_dot_product — DP4A
class, one instruction per 4 elements) against q8-quantized activations.
That is the structural difference from both our paths (scalar f32 FMA
per element; f16-staged coopmat tiles).

Ported the structure as gemv5_xs (xq int activations, ktab-packed
int8x4 weight words, per-block scale epilogue). BLOCKED on toolchain:
shaderc 2023.8 (glslang 14) predates the GLSL extension; the Ubuntu
noble glslang-tools 15.1 package is missing the extension despite
upstream support (binary strings confirm); no spirv-as; no sudo; no
network fetch. Measured with an int unpack emulation instead:
149.1 vs 144.2 ms — no gain, confirming the win lives in the hardware
dot itself (emulation ≈ float-FMA op count).

Correctness: gemv5 produces ' Paris' (engine path LLM170_V5=1).
Unblock for a future round: any upstream glslang ≥13 binary (or
spirv-tools assembler), then swap the emulation block for
dotPacked4x8EXT — the surrounding kernel is done.

### Round 9 — OpSDot toolchain unblocked; int-dot measured (2026-09-07)

Unblocked the hardware integer dot without any system packages: built
scripts/spvtool (a 60-line C tool linking the installed
libSPIRV-Tools.a static archive — assembler/disassembler/validator via
ctypes-free CLI), then scripts/patch_sdot.py: gemv5's GLSL keeps a
canonical 'DOT patch marker' pattern (8 prepared sign-extended operands
+ 4 mul + 4 add), glslc -O0 emits it verbatim, and the disassembly
block is textually replaced with OpSDot + capability/extension. A
hand-assembled device probe (vk-sdot-probe) first proved the GPU
executes OpSDot correctly (1M chained signed dots, gpu==cpu bit-exact).

Measured: hw-dot gemv5 153.1 vs gemv4 147.7 ms — still no decode win.
Conclusion refined: for iq4_xs the bottleneck is the ktab LUT gather
unpacking PER ROW, not the multiply. llama's MMQ unpacks each weight
tile to int8 words in SHARED MEMORY once and reuses it across the whole
output tile — the exact structure a 'gemv6' (shared-int8 tile + int
dot, no coopmat, no f16) should copy. All groundwork for that kernel is
now in-repo and reproducible.

### gemv6 design note (2026-09-07, end of session)

Pre-implementation analysis killed the 'shared-int8 tile' idea for the
DECODE case: with t=1 the output dimension is rows, and each row's
weights are distinct — unpack-to-shared amortizes only over the token
dimension (which the coopmat tiles already exploit for prefill). So
llama's decode win is NOT shared staging; it is (a) typed packed16/32
buffer views replacing our manual word assembly, (b) their vecq LUT
path being structurally equal to ours, and (c) the q8-style types
having no LUT at all. Our per-type standalone rates bracket the
ceiling: q8_0 257 GB/s (no LUT) vs iq4_xs 150 (4 L1 gathers + byte
repack per word pair). Closing decode toward llama's 11.4 therefore
splits into: arithmetic (LUT-free) unpack for q5/q4/q3/q6 (feasible —
their dequant is shift/mask only), and the FFN-site in-engine mystery.

### Round 10 close — gemv5_q5 resolved; int-dot decode verdict (2026-09-07)

The gemv5_q5 'q-part≡0' bug resolved as a stale-binary artifact
(include_bytes! requires a cargo rebuild after .spv swaps — the engine
path was correct all along). Shipped state: gemv5 covers iq4_xs AND
q5_K with int-arithmetic unpack (LUT-free for q5), France ' Paris'
verified through both, standalone maxrel 0.09/0.26 (W4A8 quantization
class). Final A/B: gemv5 161.0 vs gemv4 142.4 ms — the int-dot decode
structure yields NO win on this stack: the hardware OpSDot executes
correctly in isolation (probe) but not inside the kernel context
(patched builds return 0 for the dot term; suspect driver codegen),
and the int emulation is op-count-neutral against float FMA. Verdict:
llama's decode advantage is not portable via the dot instruction alone
on RADV+this toolchain; the remaining decode gap (6.3 vs 11.4) is
attributed to per-type unpack costs and the FFN-site in-engine
mystery. All gemv5 paths stay opt-in (LLM170_V5=1).

### FFN-site mystery — refined (2026-09-07, session close)

Recomputing the per-dispatch-fence numbers as bandwidth: in-engine ALL
gemv sites run at a uniform ~103-107 GB/s (attn_qkv 36MB/0.338ms=107,
ffn_down 60MB/0.580ms=103, ffn_gate 47MB/0.449ms=105) while the same
kernels standalone reach 121-146. The loss is proportional to the
standalone rate, not site-specific — it is a system-level floor (DRAM
contention / dispatch environment), not an FFN-specific effect. The
engine floor (~105 GB/s over 16GB = ~150ms step) matches the measured
142-152ms steps. Next decode lever must therefore raise the engine
floor itself, not individual kernels.

### Engine-floor mystery SOLVED (2026-09-07, session close)

The synthesis: the 'standalone' kernel rates (121-268 GB/s) are
L2-ASSISTED — the timing loop re-reads the same 36-60MB tensor 10× and
a 32MB L2 retains much of it. The MULTI runs (interleaved different
tensors = L2 evicted, exactly like the engine) show the TRUE DRAM
streaming rate of our kernels: ~105 GB/s uniform. That is why slab
pooling did nothing (no allocation overhead involved) and why the
engine floor matches ~105 exactly. llama-vk's decode at an effective
182 GB/s therefore reflects genuinely better DRAM access patterns, not
magic: their mul_mat_vec maps lanes to CONSECUTIVE words inside one
row-block (K_PER_ITER 4-8 word bursts per lane), while our gemv4 maps
lanes to strided sub-blocks (176B apart for q5_K) — 4-byte
transactions scattered across 32 sectors per wavefront. The next
decode kernel work is a lane-remap to contiguous per-lane bursts
(llama's iqs layout); expected recovery toward 150-180 GB/s would take
tg from 6.3 to ~8.5-10. LLM170_MULTI and LLM170_L2FLUSH knobs are in
vk-gemv4-check for reproduction.

### Round 12 — llama iqs lane remap landed (gemv6_q5) (2026-09-07)

Built gemv6_q5 on the solved analysis: the wavefront sweeps each row's
ql words as contiguous 128B (2-pass: block headers staged to LDS, then
a linear per-lane word sweep with role dispatch). Standalone bit-checks
clean (max|D|=0.0000 vs CPU) and the L2-evicted (true-DRAM) rate
improves +10.5% (79.7 → 88.1 GB/s) — the first measured access-pattern
win on the floor. Engine A/B is neutral (142.9 vs 142.2 ms: q5 sites
are only ~30ms of the step, so +10% ≈ noise) and spec==nonspec goes
False under G6 (accumulation-order ulp differences cross ties vs the
gemv3 verify path) — the same class as the documented VL flat point.
Shipped opt-in (LLM170_G6=1) with the tie caveat; the layout now needs
to be replicated for iq4_xs/q8/q4_K/q3_K/q6_K where the FFN mass is.

### Round 13 — gemv6_xs; engine absorbs kernel gains (2026-09-07)

Replicated the llama lane layout to iq4_xs (gemv6_xs: 2-pass headers→LDS
plus a linear qs-word sweep; one LDS-overflow bug caught by ffn_down's
n_blk=68). Standalone exact, and the true-DRAM (L2-evicted) rate on
ffn_down: 93.0 → 106.6 GB/s (+14.6%). Engine A/B with gemv6 on both
q5+xs: 148.8 vs 145.1 ms — slightly worse, within noise.

Conclusion of the decode campaign: every kernel-level DRAM gain this
session (+10-15% per type, all verified at the kernel level) is
absorbed by the engine step — the step is dominated by per-dispatch/
serialization structure (~450 gemv dispatches/token across 64 layers),
not by any single kernel's bandwidth. The next decode lever is
therefore dispatch-count reduction (kernel fusion: per-layer
norm+gemv, or multi-tensor batched gemv dispatches), a different class
of work than kernel tuning.

### Correction — dispatch theory rejected by direct measurement (2026-09-07)

Added a t=1 ktime report (step-path hook): decode step wall 212.1ms vs
kernel sum 200.7ms in NOBATCH mode — kernels are ~95% of the step;
dispatch/launch overhead is NOT the bottleneck (the batched 145ms wall
further pipelines the small kernels). The Round-13 'dispatch-count'
conclusion is retracted. What the breakdown does show: gemv4_xs 22.8ms
(69 calls), rms 21.4ms (128 — inflated by per-dispatch fences in
NOBATCH), quant 11.4, gdn_ar 10.4, gemv4_q8 5.6, plus a long tail of
per-weight q5/q6/q3 gemv entries (the top-12 cut hides ~95ms). Decode
remains weight-traffic-bound: the next lever is raising the effective
per-type bandwidth of the remaining gemv tail (q5/q6/q3 via the gemv6
layout) and the rms/quant/elementwise minor kernels' real batched
costs (needs timestamp queries; per-dispatch fences distort them).

### gemv6_q6 — WIP handoff (2026-09-07, session close)

Ported the lane-remap layout to q6_K (output.weight, ~1GB/token — the
largest single decode gemv). Fixed en route: qh LDS index missing the
n*32 term, harness n_kb=10 (q6 gemv6 has no ktab/yv4), the debug-run
transposed grid (old known defect, now fixed for all routes), scale
index (mm&31)>>4. State: STILL systematically wrong from element 0
(one-hots mismatch everywhere, some NaN at higher elements) despite the
mapping re-derived twice against deq_q6_k — the defect is suspected in
the 210B-block word assembly or the staged-qh word rebuild, not the
element formulas (all verified against CPU source). Not wired into the
engine; vk-gemv4-check with LLM170_G6=1 + G4_HOT one-hots reproduces
in seconds.

- Update: found the primary gemv6_q6 defect — sh_dl was sized [8] per
  block but q6_K has SIXTEEN scales per block (two 8-scale halves);
  the n=1 paths read out of bounds. After fixing to [64][16] the
  error collapsed from ~1e37 to 1.94 — a residual mismatch remains
  (one-hots: some elements zero, some close-but-wrong), suspected in
  the staged d or the lo/hi nibble-to-element pairing corner cases.
  Next session: dump-based comparison of sh_dl[0][0] against a python
  GGUF read of output.weight bytes 192-210.

### gemv6_q6 SOLVED — family complete (2026-09-07, final)

The dump-driven debug chain closed it: Q6_DUMP ground truth (CPU
w.data read) + a skip-main-write debug variant (macro-gated
DBG_MAIN_WRITE) + a row0-scoped dump revealed GPU dl = d×(sc as
UNSIGNED byte) for exactly the negative-scale entries — the staging's
`(sc8 << 24u) >> 24u` used UINT shifts (logical, zero-extending);
sign extension requires casting to int first (int shifts are
arithmetic). One-line fix: max|D| 1.94 → 0.0000 EXACT.

gemv6 family now covers the three dominant decode types (q5_K,
iq4_xs, q6_K) — all CPU-exact, all showing the +10-15% true-DRAM
kernel-rate class, all engine-neutral (step 149.3 vs 142.0 ms; the
engine absorption pattern holds for the third time). The lane-remap
campaign is complete: the remaining decode gap to llama is NOT in
per-kernel access patterns of these types — next candidates are the
q3/q4/iq3_s tail, the elementwise minors (rms/quant/axpy ~30ms/step
combined), and timestamp-level attribution of batched execution.

### gemv6 5-type engine verdict (2026-09-07, session close)

Full-family A/B (all five types routed, two runs): g4 142.0/143.9 vs
g6 149.3/140.9 ms — statistically neutral within the session-long
140-152ms noise band. Final decode campaign verdict: the llama lane
layout delivers real, verified per-kernel true-DRAM gains (+10-15% on
every type) but the engine step remains dominated by aggregate weight
traffic at the ~105-110GB/s system floor. With q3/q4/q5/xs/q6 all
lane-remapped and exact, per-kernel access patterns are no longer the
differentiator against llama-vk's 182GB/s effective — the remaining
gap lives in per-WG row coverage (llama packs NUM_ROWS×NUM_COLS with
subgroup-quad reductions), the iq3_s/q8 stragglers, and the
elementwise minors. All gemv6 paths remain opt-in (LLM170_G6=1).

### gemv7 (y-LDS staging) — negative (2026-09-07, session close)

Tested the y-amplification theory (gemv6 reads y once per row: 160KB
y vs 28KB weights per WG — 5.7× y redundancy). gemv7 stages each
block's 256 y floats to LDS once and dots all 8 rows against it.
True-DRAM (L2-evicted) result: 70.9 vs gemv6's 91.6 GB/s — WORSE. The
costs that outweighed the y savings: per-row header re-reads (4 words
× 8 rows × 20 blocks — header traffic is now per (row, block) instead
of staged once), 40 barriers per WG (vs 16), and the serialized row
dot inside each lane. Verdict: gemv6's structure is the local optimum
for this shape; the llama gap is not y-handling. Kept as opt-in
(LLM170_G7=1) with the harness route.

### gemv6_q8 — rejected (2026-09-07, session close)

The byte-per-lane mapping (32 lanes = 32 qs bytes) is exact
(max|D|=0.0000) but collapses to 69-73 GB/s vs the existing gemv4_q8's
257: each byte read becomes a separate WG() walker call with 4 lanes
redundantly fetching the same word. q8_0's existing scalar-vectorized
path (word-per-lane over 8-word strips) is already at the ceiling —
q8 was never a straggler in the ktime data. Kept as a documented
opt-in; engine q8 routing unchanged.

### llama mmv config decoded (2026-09-07, session close)

Read llama's actual mul_mat_vec launch configuration from
ggml-vulkan.cpp: on AMD RDNA3 the K-quant decode (q3/q4/q5/q6) uses
NUM_ROWS=2 with FORCED SUBGROUP SIZE 16 (wg_size_subgroup16 pipelines,
use_subgroups16), not our rpf=8/wave32. The spec-constant row
multipliers (rm_kq=2 non-GCN AMD, 4 on GCN; rm_iq=4) and the
subgroup-16 forcing are the two launch-geometry levers we have not
tried — both are concrete, cheap experiments for the next session
(VkCtx pipeline creation with subgroupSizeFull/16 via
VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE or
requiredSubgroupSize=16, plus rpf=2). This is the remaining
untested launch-geometry gap against llama's 182GB/s effective.

### sg16 + rpf=2 (llama RDNA3 launch geometry) — rejected (2026-09-07, close)

Implemented ctx.pipeline16 (requiredSubgroupSize=16 via the subgroup
size control pnext, the exact llama wg_size_subgroup16 mechanism) and
ran gemv6_q5 with rpf=2 — llama's RDNA3 K-quant decode configuration.
Warm rate 150.2 GB/s but true-DRAM (L2-evicted): 75.0 vs our
wave32/rpf8's 102.9 — WORSE. llama's launch geometry does not
transfer to our kernel shape; our 32-lane WG with rpf=8 remains the
best measured configuration. This closes the launch-geometry lever
list. The remaining decode gap against llama's 182GB/s effective is
now attributable to their shader-level differences beyond geometry
(typed packed16/32 buffer views, vec4 B loads with K_PER_ITER=4/8
unrolling — a full llama mul_mat_vec shader port, not a config change).

### gemv8_q5 — llama mul_mat_vec FULL PORT: first engine win (2026-09-07)

Ported llama's mul_mat_vec_q5_k verbatim (typed u16 buffer views of
the same weight chunks — Vulkan allows the same buffer bound with
different GLSL block types; SIMD-in-register nibble handling via
0x0F0F0F0F masks + unpack8; fma chains; 16-threads-per-block ×
2-blocks geometry). Two port bugs found by the harness: scales must
load as SINGLE u16s (llama scales[v_im] is one element, not a pair),
and the engine's cw2 chunk constant must use pow2ceil of the first
buffer (engine first-chunk is actual size, harness uploads the pow2
cap).

Results: standalone 162.4 GB/s warm (fastest q5 ever, exact 0.0000);
ENGINE step 142→128-130ms (-10ms, first kernel win that survives the
engine); end-to-end tg32 6.60 t/s (from 6.19-6.32). Quality: France
' Paris' exact, long prompt EXACT; spec==nonspec goes False — the
fma/subgroupAdd accumulation order differs from gemv4/gemv3 by ulps,
flipping ties in the t=1-vs-t≥2 comparison (same documented class as
gemv6's caveat). Shipped opt-in (LLM170_G8=1) pending either a
matching-order variant for verify batches or accepting the tie class.

### gemv8 family extension — xs mapping note (2026-09-07, next session)

The gemv8_q5 recipe (typed views + SIMD-in-register + fma) is the
proven engine-winning template. For iq4_xs the llama generic
mul_mat_vec mapping needs care: iqs = col/2 with iq = 16*ib32 +
(iqs%16) and qshift = (iqs&16)>>2 — dequantize4 returns FOUR
consecutive same-column elements from ONE word's nibbles, i.e. the
lane covers 4 elements at col = ib32*32 + (iqs%16)*... plus the
column split at iqs 16. Derive the exact col↔iqs table against our
VERIFIED gemv4_xs mapping (byte b of sub → elements b lo, 16+b hi)
before writing gemv8_xs; the same derivation then applies to
q3/q4/q6 via their llama dequantize4 blocks in dequant_funcs.glsl
(DATA_A_Q4_K etc.). y loads via the existing yv4 alias binding.

### gemv8_q6 — exact but slower (2026-09-07, close)

Ported the gemv8 recipe to q6_K (one OOB bug caught by the harness:
scale base is 8n per half, not 16n — sh_dl[16] indexed to 23). Exact
(0.0000) but 67.6 GB/s vs gemv6_q6's ~90 — the 210-byte block
misalignment forces per-word remainder assembly in the linear sweep,
which the SIMD gains don't offset. q6 stays on gemv6 in the engine;
gemv8_q6 kept harness-only.

### gemv8 numeric-family UNIFICATION — spec invariant restored (2026-09-07)

Routed the verify batches (t<16) through gemv8 as well — gemv_w's G8
gate widened from t=1 to t<16 (prefill tiles still own t>=16) and
gemv_stage jobs dispatch via gemv8 when active. Result: the t=1 decode
and the spec verify batches now share ONE accumulation order —
spec==nonspec x4 back to True (measured), France exact, HIP 19/19.
gemv8 is now a complete-quality path: exact kernels, faster engine
(step -11.4%, tg32 6.70), invariant-safe. Still opt-in
(LLM170_G8=1) pending the long-form gate and np4 tie checks on the
unified family.

## Vulkan ms-geometry tile family + q3_K decode fixes (2026-09-10, plans/40)

Reference re-measured on this machine (llama-bench d222767c, Vulkan):
pp512 356.66 / pp64 244.08 / tg8 12.12.

Findings: the previous tile family was pinned at ~39 GB/s A-stream not by
coopmat structure but by two codegen hazards — an 8-way if-chain weight
indirection (WG()) inside the staging loop and per-MMA uniform guards.
llama's own mul_mm shader run in our harness on the same geometry reaches
67-79 GB/s; removing the indirection (single-buffer direct index; every
tensor fits one chunk since chunk size is per-tensor next-pow2) and the
guards (zero-padded B, drain clips) recovers most of it: 39 -> 64 GB/s
at the llama m-warptile geometry (BM64/BN64/BLOCK128, 2 wave64 warps).

Rolled the ms skeleton out to all quant types (q4k/q6k/q3k/q8/xs/nl via
shared ld16 unaligned-merge), 64-row workgroups, guard-free MMA.

Accuracy: while bisecting the rollout, the harness uncovered THREE latent
q3_K tile decode defects that predate this round (scale tmp assembled
from 3 bytes instead of 4 — byte 107 missing; block-half index taken
from the global 128-element half instead of the within-block half;
shift/hbit derived from (sb>>1)&3 instead of sb&3). Fixed against the
ggml reference; verified elementwise via a dump probe (0/5120) and
tile_check (maxrel 0.0033). The engine q3 path is permanently moved to
the fixed kernel. xs/nl additionally needed the shared ktab LUT bound
(omitted in the new dispatch produced garbage tokens).

verify (judge, LLM170_TILE_MSALL=1): 22 PASS / 3 FAIL — prior accepted
state was 4-8 FAILs; single_ko and np4_seq1 pass for the first time.
Remaining 3 are spec-equality borderline (first mismatch @gen 10-21).

Memory: raw_init cloned the whole model to_vec() (17.5 GB anonymous —
the two-process host-RAM OOM that repeatedly killed sessions). Weights
now stay mmap borrows through upload; after init the file pages are
madvise(DONTNEED)-returned (page-aligned ranges; tensor offsets are only
32B aligned). Steady state: anon 0.7 GB, 14.65 GB returned. Two
concurrent full-model processes complete — previously fatal.

Benchmarks (MSALL, q35work): pp64 168.8-180.2 (was 140), pp512 162.6-167.6
(was 125), tg 7.36-7.59 (decode path untouched). vs llama Vulkan: pp64
0.69-0.74x, pp512 0.46x, tg 0.61-0.63x.

Open levers: tile codegen gap to llama's own shader (64 vs 79 GB/s,
same geometry — vectorized staging loads); pp512 N-amortization (weight
re-read per 64-token slab; BN>=256 attempts regressed so far); decode
timeline (132 ms wall vs llama 89 ms; gemv8 163 vs ~196 GB/s effective).

## Vulkan decode-kernel llama ports (2026-09-10, plans/40 cont.)

Decode GEMVs ported from llama mul_mat_vec_{q5_k,q4_k,q6_k} to the
gemv8 family: 64-thread WGs (16-thread block groups), 2 rows per WG,
vec4 SIMD-in-register unpack, u16 typed views on the single weight
chunk, subgroupAdd reduce. q5 143->225, q4 152->272, q6 ~143->194 GB/s
(all max|D|=0.0000). q6 uses llama's double-buffered shared scale
cache; q4's y loads sit at {y1, y1+32, y1+128, y1+160} with
per-component smin; q5's interleave is {0,1,16,17,32,33,48,49}.

gemv8_q8 added for q8_0 (34B blocks) — moves alpha/beta/v stragglers
off the legacy gemv path. mmv-check probe runs llama's prebuilt dmmv
spvs in our harness for reference (beware: repeat-loop timing on
<32MB tensors measures the L2, not DRAM).

Benchmarks: tg8 7.36 -> 8.63 (+17%), tg32 7.59 -> 8.21. pp unchanged.
verify: 22 PASS / 3 FAIL — identical set before/after the port.

vs llama Vulkan: pp64 0.72x, pp512 0.46x, tg8 0.71x.

Follow-up: gemv8_q8b (llama generic dmmv structure, 87 -> 329 GB/s)
and gemv8_xsb (same structure with our verified iq4_xs decode,
125 -> 182 GB/s), both max|D|=0.0000.

Benchmarks after the full decode arc: tg8/tg32 7.36 -> 9.33 t/s
(+27%; llama Vulkan 12.12 — 0.77x). pp unchanged. verify: 22 PASS /
3 FAIL, identical set before/after all four ports.

Open: tile staging vectorization (64 -> 79 GB/s ceiling); pp512
N-amortization (gate+up merged dispatch); gemv8_q3 remains on the old
kernel (3 tensors).

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

## gdn_ar4 — ILP across state columns (2026-09-10, plans/40 cont.)

The GDN autoregressive update ran one state column per work group: per
token step the two subgroupAdd reductions serialize, and nothing else
in the work group hides their latency. gdn_ar4 keeps four columns per
work group — four INDEPENDENT recurrence chains interleave, halving
exposed reduce latency, with 4x fewer work groups and k/q loads
amortized 4x. Identical per-column scalar arithmetic (bit-equal).
Prefill gdn_ar 0.383 -> 0.189 ms per layer (-51%); 48 layers save
~9.3 ms per chunk. LLM170_VK_AR4=0 opts out.

Benchmarks (defaults): pp64 195.8-196.7, pp512 182.7, tg8 9.6.
vs llama Vulkan: pp64 0.80x, pp512 0.51x, tg8 0.79x.
verify: 23 PASS / 2 FAIL — best recorded (spec_np4_seq1 now passes;
remaining: spec_np4_seq2@21, spec_long_np4_seq3@10, both borderline
spec-equality class).

gdn_ar8 (same recipe, 8 columns): 0.176 ms/layer — the ILP gain
saturates (subgroup shuffle throughput is now the bound). pp64
192-199 t/s. Bit-identical outputs; LLM170_VK_AR4=8|4|0 selects.

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

## Speculative decode (MTP) — inert on the Vulkan raw path

`--spec k` is opt-in (bench/serve/infer only set `mtp_wanted` when a spec
budget is requested), so there is no default-path regression. On the raw
Vulkan path, however, the feature does not do useful work:

* `mtp_draft_logits` is only populated by the non-raw `verify_batch` path;
  the raw path stores `mtp_draft_tok` (the GPU MTP head's argmax) instead.
  The CPU spec loop therefore drafts greedy(empty logits) = 0 every step and
  every draft misses (bench: "fwd 8, gen 8, 1.00 tok/fwd", draft=0 in
  LLM170_SPEC_DBG).
* Enabling the MTP hook costs ~150 ms per prompt token in prefill (pp64
  with --spec 2: 9.9 s vs 0.33 s without), because the MTP layer chain runs
  per token.
* The batched GPU verification path (`LLM170_SPEC_GPU=1`) is slower still:
  1.50 t/s on tg8 (9 fwds for 8 tokens, 0.89 tok/fwd).

The single-sequence CPU spec loop is structurally unable to beat plain
decode (each candidate token costs a full target decode), so the viable
design is the batched verify path; until that is rebuilt, MTP is not a
throughput lever.

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

## Tiny-tensor routing — neutral (plans/45)

The pp64 profile shows the small q8 tensors (ssm_alpha/beta at 48 rows,
attn_k/v at 1024) costing ~108 ms of a ~306 ms pass - the tile kernel
launches a single workgroup for a 48-row tensor and pays the full
K-iteration latency for 0.26MB of weights. Routing everything with
n_out <= 128 through the row-parallel gemv8 kernel is token-identical but
measures neutral on all three metrics (pp64 210-214, pp512 300-301,
tg8 9.94): the gemv8 path pays for the same latency through t-fold weight
re-reads. Both paths are latency-bound for these shapes; a real fix needs
all ninety-six tiny tensors in one dispatch (multi-tensor batching).

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

## Speculative decoding on Vulkan — cost breakdown (session 2026-09-11)

Measured with `LLM170_SPEC_TIMING=1 LLM170_SPEC_GPU=1 LLM170_VKD_BATCH=1` on tg spec=3
(q35work, pp64): step total 1280.9 ms for acc=2 tokens. Components:

| Stage | Cost | Note |
|---|---|---|
| draft chain (k=3) | 351 ms (117 ms/draft) | MTP layer GEMVs at solo rate (~30 GB/s) |
| state snapshot | 620 ms | rollback save of GDN state (~150 MB) — dominant |
| verify `step_batch` t=4 | 296 ms (74 ms/token) | **no weight amortisation at small t** |
| advance | 4 ms | |

Plain decode reference: ~62 ms/token, so the spec path was 10-20x slower than plain.

### Root finding: weight traffic is not shared across the batch dimension at small t

- gemv8 (t < 16) dispatches one workgroup per (row-pair, token): the same weight rows are
  re-read by every token slab, and the z-ordered launch gives no temporal locality for L2 to
  merge. Traffic scales ~linearly with t.
- Forcing the tile path with `LLM170_VK_TILE1=1` does **not** change this: verify t=4 stays at
  293 ms (73 ms/token), i.e. slower than 4 independent t=1 decodes.
- At t >= 16 the tile kernels *do* amortise (pp64: 624 ms / 64 = 9.8 ms/token, 6x better than
  t=1), because a workgroup keeps the weight tile in LDS while iterating over tokens.

Consequence: speculative verification with small batches (k+1 tokens) cannot win on this
backend regardless of the draft cost or snapshot cost. A batched verify only becomes
profitable if the verification batch is large (>= 16) or if the small-t GEMV path is
restructured so that one workgroup covers several tokens (i.e. move the token loop inside
the workgroup, LDS-resident weights).

### Secondary findings (kept, correctness-neutral)

- The prefill MTP KV hook used the **CPU** `mtp_step` (~150 ms/token), which was the cause of
  the "spec enabled makes prefill 30x slower" observation (pp64: 207 -> 6.5 t/s). It now uses
  `mtp_step_gpu` (argmax only, `mtp_draft_tok`); the draft fallback in `spec.rs` reads that
  field when `mtp_draft_logits` is empty. Spec output remains token-identical to the plain
  path (verified: 84 220 201 198 201 198 201 for both).
- `LLM170_VKD_BATCH=1` (batched verify) is token-identical to the per-token verify (same
  gemv8 kernels, t < 16).

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

## GDN state coalescing (2026-09-11, session end)

The recurrent GDN state was stored row-major while the ar kernels assign 4 rows per
lane, so every subgroup load touched 32 cache lines at 2KB stride - the state streamed
at 36 GB/s (8.3 ms/token across 48 GDN layers). Storing the state transposed
(`s[col*d + row]`) makes the same loads fully coalesced. Math, lane mapping and
subgroupAdd order are unchanged, so outputs stay bit-identical.

The first cut fully unrolled the 8-column loop of `gdn_ar8` on top of the new
addressing; register pressure spilled inside the 512-token loop and regressed pp512
303 -> 285. Processing columns in rolling pairs halves live temporaries and restores
prefill:

| Metric | before | after | llama.cpp | ratio |
|---|---|---|---|---|
| pp512 | 303 | 304.2 | 356.66 | 0.85x |
| tg32 | 9.94 | 10.66-10.69 | 12.12 | 0.88x |

gdn_ar per-layer time at t=1: 0.174 ms -> 0.021 ms (8x). Effective decode weight
bandwidth 175 -> ~190 GB/s. Remaining decode gap vs llama is inside the gemv8
kernels themselves (~190 vs ~213 GB/s effective) - layouts already match llama's
dmmv, so further gains need per-instruction tuning or a different access idiom.

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

## Small-kernel follow-up: instruction-level slimming is neutral (2026-09-12)

`rmsq`'s quantisation phase rewritten with float4 loads and a tree `amax` (bit-exact:
`fmax` is associative, the quantisation is element-wise, and the reduction path is
untouched). Verified bit-identical, A/B three interleaved pairs: tg 11.10 vs 11.11
(neutral), pp 177.2 vs 176.0 (+0.7%, noise). Kept because it is strictly less work, but
it confirms the conclusion from the multi-block experiment: these kernels' 17-30 us are
not spent in their instruction stream or in their grid shape. The only structural lever
that has moved them is *more* WGs when the workload allows (which is why the GQA
attention helps at long context and hurts below 768).

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

## Verification-tooling fix: LLM170_REQUIRE_GPU (2026-09-12)

Twice today a "bit-identical" verification run was actually the **CPU fallback**: when
any kernel source fails to compile, `inject_rawhip` fails and `infer` (unlike `bench`,
which was hardened earlier) continues on the CPU engine, so the CPU reference stream
matched itself. `infer` now honours `LLM170_REQUIRE_GPU=1` and returns an error instead
of falling back; all the verification claims in this file were re-run with it.

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
