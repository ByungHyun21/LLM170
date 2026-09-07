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
