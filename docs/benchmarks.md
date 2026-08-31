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

Gap analysis (measured, `LLM170_GPU_TIME`):
- Sync round trips: eliminated for grouped projections (up/launch ≈ 0);
  remain per group (≈5 groups × 64 layers).
- Kernel throughput: element-wise dequant ALU is the residual in-kernel
  bottleneck (~10× from the bandwidth floor); token-amortized decode kernel
  (q3) closed 1.5× already.
- CPU-side GDN/attention per layer dominates decode step time today.

## qwen4exp (Qwen3.8-Flash-Next 125B-A6B, UD-Q4 4-split)

| Metric | llama.cpp (PR #27742 runtime) | LLM170 (GPU backend) |
|---|---|---|
| Prefill pp, np2×262K ctx | 178–237 t/s per slot (HIP 7.2.2; 2026-08-27) | long-prompt case pending (see below) |
| Decode tg, np2 | 11.6–15.1 t/s | ~1.4 ms/forward compute (2.2 ms stages + overhead, hot cache) — MoE grouped batching landed 2026-08-31 |
| PLE table | NVMe mmap + `-ot ...=CPU` (identical pattern) | NVMe mmap + `MADV_RANDOM` (same) |

Known remaining CPU-serial sections in the LLM170 GPU path: QSA masked dense
GQA (O(p²) over context), GDN chunked recurrence, HC residual math — these
dominate long-prefill wall time today (single core during prefill).

## Method

- llama.cpp numbers: `llama-server` slot timings (`print_timing`) and
  `bench-*.txt` artifacts, quoted with np/ctx/date.
- LLM170 numbers: end-to-end `llm170 infer` wall clock (incl. model load
  unless noted), or `LLM170_Q4_TIME`/`LLM170_GPU_TIME` stage instrumentation
  for compute-only figures. Server-based streaming measurement is the
  standard once a server mode exists.
