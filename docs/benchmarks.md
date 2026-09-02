# Performance Comparison vs llama.cpp

Conditions always quoted (context / batch / quantization / backend). All
numbers on the dev machine (Radeon 8060S, gfx1151, 32-thread CPU) unless
noted. Relative regression tracking only — absolute cross-machine comparison
is out of scope.

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

Current standing (GPU backend, single-tenant `llm170 bench`, 2026-09-02):

| Metric | llama.cpp target | LLM170 | Ratio |
|---|---|---|---|
| Decode tg24, t=1 | 10.4 t/s | **3.86 t/s** | 0.37x |
| Prefill pp64 | 142.8 t/s (418-tok case) | **13.04 t/s** | 0.09x |

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
