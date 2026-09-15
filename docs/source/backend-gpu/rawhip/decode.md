# `crates/backend-gpu/src/rawhip/decode.rs` — measurement record

> Items from `docs/benchmarks.md` that correspond to this file (by section title).
> Tables and numbers are verbatim. Summary metrics remain in benchmarks.md.

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


## 27B (qwen35) decode kernel breakdown: it is already near the DRAM limit (2026-09-14)

KTRACE of a 27B decode step (88.6 ms of kernels, 40 kernel types):

| kernel | total ms | calls | effective |
|---|---|---|---|
| gemm_q5k (n_out=17408) | 14.21 | 51 | ~179 GB/s |
| gemm_q5k (n_out=5120) | 13.92 | 75 | ~85 GB/s |
| gemm_xs (n_out=17408) | 10.87 | 46 | ~180 GB/s |
| gemm_q6k (65535 / 5120) | 10.88 | 27 | ~180 GB/s |
| gemm_q4k (17408 / 10240) | 7.83 | 42 | ~150 GB/s |

Across the step the model's ~15 GB of weights move at ~169 GB/s, 71% of the 236 GB/s
the probe measures, and the large FFN GEMMs individually sit at ~179 GB/s. So the 27B's
decode is close to the practical DRAM limit and there is only ~10-20% of headroom
there; for this model the earlier "pattern-limited" correction does not apply. Its real
headroom is the prefill: 363 t/s is 34% of the f32 peak while the weights stream at
only 10.6 GB/s, i.e. compute/tile limited, which is what the plans/66 P1-style work
(bf16 tensor-core GEMM, measured 1.42x in the reference stack) targets.



## WB prefill kernel map (2026-09-14, accurate after KTRACE fix)

Re-measuring pp512 (warm) with the fixed KTRACE: **kernel 1,386.1ms + gap
24.7ms = 1,410.8ms**, explaining 93% of the 1,518ms wall clock. The gap is
only **1.7%** — 27B prefill is **kernel-bound**, the opposite of Flash-Next
(gap 43%). The optimization lever is inside the kernels.

| Kernel | Time | Calls | Share |
|---|---|---|---|
| mmq_q5k | 506.1 ms | 161 | 37% |
| mmq_xs | 237.2 ms | 67 | 17% |
| mmq_q4k | 203.0 ms | 55 | 15% |
| mmq_q6k | 118.5 ms | 48 | 9% |
| gdn_ar_w_swap | 95.6 ms | 48 | 7% |
| gemm_q8_j128 | 30.5 ms | 108 | 2% |
| silu_mul_f32 | 28.5 ms | 64 | 2% |

`mmq_*` accounts for 77% of the total. Processing 2 x 512 x 27e9 ≈ 27.6 TFLOP
in 1.386s gives **19.9 TFLOPS = 34% of the wmma peak (59)** — headroom exists
but the lever is tile/WMMA-family kernel rewrites (plans/66 P1).
Side note: before the `LLM170_KTRACE` fix, this prefill was under-reported
4x at 366ms (see the pairing entry in qsa.md).

## plans/73 — 27B attention round (2026-09-15)

**Adaptive decode segment** — sg was fixed 32 (short-ctx latency hiding) but
at 16k context nseg=512 forced `qsa_flash_merge` through 500+ partials per
head serially (KTRACE 16k: gqa2d 5.59 + merge 4.17ms/step). sg now scales:
`((pos+1)/64).clamp(32,256)` keeps nseg ≤ 64; short-context numerics
unchanged (the gate prompt stays in the sg=32 regime — bit-identical).
tg128@16k 10.3 → 10.54.

**qsa_flash_wk8i (prefill, default)** — 4-key register ILP on the attention
fmaf chains (strict-FP forbids reassociation, so the 32-fma chain per key
was the critical path). Per-key j-order, shuffle tree and softmax update
order preserved = bit-identical (gate PASS at t=208 prefill). pp4096
315 → 323.6 (+2.7%), pp16384 tie (253 vs 254). A shared-tile variant was
tried first and rejected: the own*32 lane stride makes 4-way LDS bank
conflicts and 32KB shared halves occupancy — pp16k 257 → 209 (reverted).

**qsa_flash_wk8d (v_dot2 QK, opt-in LLM170_WK8D=1)** — gqa2d's structure
extended to t>1 (warp=query, lane=key, 128 v_dot2 per lane, shared K/V
tile, thread=dim PV). Two structural defects found via differential
attention-output dumps (new LLM170_ATTN_DUMP debug env, layer 0 after
merge): (1) PV accumulation AND the final part write must cover ALL 8
queries per thread (gqa2d convention — the first cut left query w's row
filled only at dims 32w..32w+31); (2) fully-masked causal tiles (which
never occur at t=1) need e=0 explicitly or exp(-MAX-(-MAX))=1 leaks future
keys with weight 1. After both fixes the kernel is numerically sane
(matches wk8i's token at t=129) but the scalar-f32 PV phase dominates:
pp16384 234 vs wk8i 253 — kept as an asset; the identified next step is a
WMMA/tensor-core PV. Prefill seg sweep at 16k: 1024 optimal (2048→249.7,
4096→246.0).
