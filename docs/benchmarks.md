# Performance Comparison vs llama.cpp

Conditions always quoted (context / batch / quantization / backend). All
numbers on the dev machine (Radeon 8060S, ROCm 7.2.2, 32-thread CPU) unless
noted. Relative regression tracking only — absolute cross-machine comparison
is out of scope.

## qwen35 (Qwen3.8-27B, UD-Q4_K_XL)

| Metric | llama.cpp | LLM170 (GPU backend) | Ratio |
|---|---|---|---|
| Decode tg, t=1 | 10.7–11.8 t/s (server bench, 2026-08-18) | **~1.3 t/s** (0.77 s/step; 2026-08-31) | 0.11× |
| Prefill pp, ~2400 tok | 310–345 t/s | **~5.3 t/s** (437 s wall incl. load; q2 kernel + CPU GDN/attn) | 0.017× |

llama.cpp conditions: `llama-server -ngl all`, f32 KV, GPU kernels for every
op. LLM170 conditions: matmul projections on GPU (all quant types), GDN
recurrence + attention + norms still on CPU per layer, host round-trip per
matmul group.

2026-09-02 update (`llm170 bench`, single-tenant, ROCm 7.2.2-linked
binary): pp64 **2.12 t/s**, tg24 **1.32 t/s**. Co-resident measurement
(llama-server holding VRAM) gave tg 0.58 t/s — invalid per the
non-coexistence rule, quoted only as a caution. PP bottleneck: GDN chunk
path (sequential per-head forward substitution at real dims — first
real-dim run took >12 min for pp512 before being killed). TG bottleneck:
value-path host roundtrips. First performance target: the ROCm 10 llama
table in AGENTS.md (pp 142.8–315.4 t/s, tg 10.4–11.6 t/s).

Follow-up (same day, kkt kernel landed): GDN chunk GPU vs CPU
(`LLM170_GDN_CPU=1`) measure identically on pp64 — the chunk is NOT
the prefill bottleneck. `gpu-mm` on ffn_gate (t=64) measures ~4.8
GFLOPS on that shape vs llama's ~1400 GFLOPS estimate (~300x on the
microbenchmark, ~15x engine-wide at ~97 GFLOPS average). **Prefill
optimization priority moves to quantized-GEMM batch throughput**
(tiling/split-K/vectorization for t=64 shapes).

## qwen4exp (Qwen3.8-Flash-Next 125B-A6B, UD-Q4 4-split)

| Metric | llama.cpp (PR #27742 runtime) | LLM170 (GPU backend) |
|---|---|---|
| Prefill pp, 2311 tok (single) | 178–237 t/s per slot (HIP 7.2.2; 2026-08-27) | **~3.3 t/s** (~699 s incl. load+decode; Vulkan, 2026-09-01 — token-exact 24/24) |
optimization priority moves to quantized-GEMM batch throughput**.
Microbench matrix (ffn_gate, 512 rows): GFLOPS is FLAT ~4 across t in
{1,16,64,256} — per-op time scales linearly with t, i.e. the k-lane
(q2) kernel re-reads weights per token with zero batch amortization.
Fix: token-block accumulation with weights held in registers/L1
(extend the q3 token-tile design to prefill).

Landed same day: gemm_q7 (16-token blocks, unrolled register
accumulators) + de4 (block-invariant hoisting: one dequant of
d/dmin/scales per 4-element batch instead of per element). Real-model
results: **pp64 2.25 -> 13.04 t/s (5.8x), tg24 1.33 -> 2.63 t/s
(2.0x)** vs llama 143/10.4 targets. Engine-trace (rocprof) analysis:
decode GEMV already streams at 141 GB/s (llama-level); the remaining
tg gap is ~280 ms/token of host glue (per-op readback + buffer
acquire + launch marshalling) — next lever is a device-resident
decode frame for qwen35 (the qwen4exp frame.rs pattern).

Decode step breakdown (`LLM170_Q4_TIME`, 2026-09-01, after same-input
projection grouping): MoE ~190 ms (was ~690 — expert gate·up grouped to one
call, down as paired batch), GDN ~126 ms (CPU recurrence — GPU kernel is the
next item), HC ~55 ms, QSA ~18 ms. Round-trip inventory dropped from
~1,680/step to ~600/step; eliminating per-call sync entirely (persistent
device activations + single end-of-forward sync) is the structural next step.

Known remaining CPU-serial sections in the GPU path: GDN recurrence (chunked
prefill + AR decode), QSA indexer scores (O(p²) over context), HC residual
math. Note (corrected 2026-09-01): the qwen4exp infer path previously
hardcoded the HIP runtime, so earlier "Vulkan" attributions were wrong; the
intermittent "Memory page 0 doesn't exist" fault is runtime-attribution
pending, with true-Vulkan long runs queued after the fix.

## Verification status (same binaries as the numbers above)

- qwen4exp matrix (Vulkan): single_ko exact 24/24; single_short/code near-tie
  (gap 0.01/0.12 nat); long 2311-tok and long2 1904-tok **exact 24/24 each**.
- qwen35: GPU server↔CLI self-consistent matrix 7/7 exact (2026-08-31 binary);
  full matrix re-run pending local filesystem recovery (transient ENOENT
  windows on the model volume, unrelated to the engine).
