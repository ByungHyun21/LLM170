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

## qwen35 — Qwen3.8-27B, UD-Q4_K_XL

Current standing (raw-HIP backend — pure Rust executor, cubecl removed,
single-tenant `llm170 bench`, 2026-09-03). All paths bit-exact against the
CPU W4A8 reference engine (greedy streams identical, incl. 64-token batch
cross-verification against per-token):

| Metric | llama.cpp target | LLM170 | Ratio |
|---|---|---|---|
| Decode tg24, t=1 | 10.4 t/s | **10.0-10.5 t/s** (GPU argmax, logits resident) | 0.97-1.01x |
| Prefill pp64 | 142.8 t/s | **97.4 t/s** (81.8 exact mode, LLM170_EXACT=1) | 0.68x (0.57x) |
| Prefill pp512 | ~230 t/s (server-bench) | **60.9 t/s** | 0.26x |

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
