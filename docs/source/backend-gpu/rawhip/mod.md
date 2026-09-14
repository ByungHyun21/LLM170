# `crates/backend-gpu/src/rawhip/mod.rs` — 측정 기록

> `docs/benchmarks.md`에서 **이 파일에 해당하는 항목만** 옮긴 것(제목 기준).
> 표·수치는 원문 그대로. 요약 지표는 benchmarks.md에 남는다.

## Prefill measurement instrument caveats and the 27B op breakdown (2026-09-14)

`LLM170_PP_PROF` records hipEvents around each raw-path section and is valid only for
the dense/raw path (the 27B). On the Flash-Next frame path it emits nothing; use the
stage timers there (and note pp2048 measures 224.5 t/s today, not the earlier figure).

For the 27B at pp512 the sections sum to the full wall, 1,409 ms:

| section | ms | | section | ms |
|---|---|---|---|---|
| ffn_gate | 520.6 | | ffn_silu | 28.5 |
| ffn | 279.2 | | gdn | 18.6 |
| gdn_mm | 212.8 | | split+l2 | 18.4 |
| proj | 204.4 | | norm+quant | 21.3 |
| trace | 95.8 | | head | 5.8 |

The host is not a factor on this path: the same run reports cpu_submit=11.0 ms for the
whole warm step_batch, so every remaining millisecond is device-side. The four GEMM
sections (ffn_gate, ffn, gdn_mm, proj) are 1,217 ms or 86%; at 2*512*27e9 FLOP that is
~19.5 TFLOPS, 34% of the f32 peak, so the lever is the batched GEMM tiles rather than
anything host-side. Two instruments mislead on this path and should not be trusted for
it: LLM170_KTRACE sums only 366 ms against a 1,409 ms wall, and LLM170_NOLAUNCH reports
a 1,014 ms "host skeleton" that contradicts the 11 ms cpu_submit - both are unreliable
for batched launches. The trace section's 95.8 ms is the PP_PROF instrumentation's own
hipEventCreate cost, not work.



## KTRACE fixed: the prefill is 49% kernels and 43% launch gaps, and the gaps are the lever (2026-09-14)

The KTRACE sum was wrong because ktrace_dump paired events with a heuristic ("consecutive
events with the same name and gy are a start/end pair"), which breaks whenever the same
kernel launches repeatedly - exactly what batched shapes do. It now uses the fixed stride-2
pairing (each launch records exactly two events), the same assumption the gap code already
used. Validation: the 27B pp512 now reports TOTAL 1,380 ms against a 1,516 ms wall (it was
366 ms, a 4x under-measurement), and the Flash-Next pp2048 chunk reports 5,088 ms of
kernels plus 3,839 ms of gaps against an ~8,900 ms wall - fully accounted.

| kernel | total | calls | share |
|---|---|---|---|
| gemm_q8_j128 | 1,085.4 ms | 776 | 25% |
| q4_gemm_q4k_ge (MoE gate/up) | 787.0 ms | 52 | 18% |
| q4_gemm_f32_m | 609.2 ms | 288 | 14% |
| gdn_ar_w_swap | 344.6 ms | 36 | 8% |
| q4_qsa_attn_sel6 | 285.8 ms | 12 | 7% |
| q4_rows_permute_u32 | 226.3 ms | 331 | 5% |
| gemm_q8_0 / quant_q8 / rms_* / hc_* | ~950 ms | | 22% |

Launch gaps, by predecessor (3,838.6 ms total):

| after | gap | calls | per call |
|---|---|---|---|
| q4_rows_permute_u32 | 1,728.7 ms | 331 | 5.2 ms |
| gemm_q8_j128 | 722.1 ms | 776 | 0.93 ms |
| q4_gemm_f32_m | 654.4 ms | 288 | 2.3 ms |
| q4_qsa_attn_sel6 | 367.1 ms | 12 | 30.6 ms |
| q4_hc_combine + hc_gate_mean | 325.4 ms | 193 | ~1.7 ms |

The single largest item is therefore the host work that follows rows_permute_u32: 5.2 ms per
call, 331 calls, 1.73 s per chunk (19% of the prefill). It is host, not device - the gap is
the device idling while the host prepares the next submission.

One candidate was tested and ruled out: the MoE routing's top-k selection sorts all 512
expert logits per token (`idx.sort_by` in stages/moe.rs). Replacing it with
`select_nth_unstable_by` plus a k-element sort - which preserves the selection and is
bit-identical on the diverse stream - measured neutral at pp2048 (8,940-8,959 vs
8,878-9,034 ms) and -1.2% at the decode (567.5 vs 574.3 ms per 8 steps), so it was reverted;
the routing sort is not what the 5.2 ms consists of. The remaining suspects are the
per-expert host loop in frame_moe_gemm (one GEMM launch per non-empty expert, plans/57's
mul_mat_id target) and the offsets d2h that precedes it.

A second candidate was also tested and ruled out. The routing loop normalises all 512
expert logits per token (a third pass of f32 divisions), which at 2048 tokens is ~1M
divisions ~= 7 ms per layer - the right order of magnitude for the 5.2 ms gap. Narrowing
that pass to the selected k (bit-identical: the same v/zs per selected expert, summed in
the same order) left the gap unchanged at 1,716.7 ms against 1,728.7 ms, so it was reverted.
The arithmetic that made it look promising also shows why it cannot be the whole story: the
routing runs once per layer (48 times) while the gap is counted 331 times, so most of those
gaps are the host work for whatever runs next, not the routing. The other gaps are the same shape at smaller scale: host submit
work after each small launch.


