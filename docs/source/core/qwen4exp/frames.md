
## Per-layer QSA norm uploads: single-slot cache → (ptr,len) map (2026-09-16)

`LLM170_Q4_TIME` split the QSA host path: `mm+rope = 3.4ms/layer` with every
sub-phase (indexer extract, sel_dev, kv_dev, attn_sel) at 0.00-0.05ms. The time
sat in `frame_qk_norm_rope`, whose three constant uploads used a **single-slot**
cache keyed by content hash. QSA layers each carry a different tiled q/k norm
(24KB + 2KB), so every layer missed and re-uploaded through a pageable
`hipMemcpyAsync` that drains the stream. Measured: `qn MISS` on every layer,
every step.

Fix: the frame builds the head-tiled norms once into `Frame4::qsa_qn_t` /
`qsa_kn_t` (stable pointers), and the accelerator caches uploads in a
`(ptr,len) -> GBuf` map (`upload_map`). Steady state: zero uploads, zero hashing.

Effect: KTRACE decode launch gaps 40.0ms -> 10.2ms (`after qk_norm_rope`
31.9ms x12 -> 0.4ms x12). Wall time at tg128 was unchanged (17.2 t/s) because
the GPU stayed busy on the queued work — the queues were deep enough to hide
the stalls; the fix matters because it removes ~40ms/step of host blocking
(and makes KTRACE gaps meaningful again).

## q8_0 GEMV lane efficiency experiments — both variants lose (2026-09-16)

The GDN `mm_group` (qkv [2560->10240] + gate [2560->6144] per layer, q8_0)
runs at ~106-112 GB/s against a 236 GB/s probe, and the 64-lane/row kernel
wastes lanes whenever `n_sub = n_in/32` is not a multiple of 64 (62.5% at
n_sub=80, 31% for `gemm_q8_0_w` at n_sub=10). Two replacements were built and
measured on Flash-Next tg128 (baseline 17.23 t/s):

| variant | lanes/row | arithmetic | t/s | delta |
|---|---|---|---|---|
| `gemm_q8_0` (incumbent) | 64, sb=l+64m | — | **17.23** | — |
| `gemm_q8_0_w4` | 16 x 4 quadrants (no lane waste) | bit-identical (rebuilds the 64-lane f64 tree) | 15.64 | -9% |
| `gemm_q8_0_w16` | 16 consecutive + 2-way ILP | reordered | 15.97 | -7% |

Both pass the stream gate (w4 by construction, w16 empirically) — the loss is
memory-side: the quadrant variant scatters each lane's addresses into four
544B chunks, and even the consecutive variant's 4-rows-per-warp layout loses to
the incumbent's contiguous 32-lane sweep. Conclusion: at these shapes the
limiter is coalescing/ILP, not lane occupancy. Both kernels stay registered
behind `LLM170_Q8W4=1` / `LLM170_Q8W16=1` as documented negatives.

## np batched frame decode (2026-09-16, session 4 tail)

`frame_forward_np` + `Engine4::decode_batch`: weight-streaming ops (mm_group,
hc frames, MoE, head) run once for all t rows; sequence-owned state (GDN conv
ring + AR, QSA rope/selection/KV append/attention, PLE) runs per row at t=1
through **row views** (`Accelerator::frame_slice` registers base+off handles,
one row-view set cached in `Frame4::np_views`, always built for 8 rows — a
first-batch-of-2 then 4 would otherwise index out of bounds).

Two bugs found and fixed on the way, both instructive:
1. Per-row sections must run under `t_cur = 1` — several state kernels
   (gdn AR scan, `frame_moe_gemm` row count) derive their row count from the
   frame-global `t_cur`; left at t they read/write past the row view into the
   neighboring row.
2. Conversely the elementwise Split3/L2/Scale section must run at `t_cur = t`:
   left at 1 it only wrote row 0, and row 1 consumed stale gq/gk/gv (probes:
   conv rows identical, AR row 1 diverging, ar_in_q differing — pinpointed it).

`gemm_q8_0_mt` (multi-token q8_0 GEMV): the old grid=(t, n_out) batch re-read
the weight row per token (hc up measured exactly 4.1x at t=4). The mt kernel
reads the weight row once and keeps a per-token f32 chain + per-token f64 tree
reduction — same lane→sub-block mapping, so **bit-identical** to the t=1
kernel (verified: np4 52/52 tokens exact, 27B spec==greedy PASS).

Measured (4-slot HTTP, same prompt, 4x64 tokens): sequential-per-slot 16.2 ->
batched **21.0-22.5 t/s** (+30-39%). Step composition at t=4 (FRAME_TIME):
MoE ~51ms (per-row t=1 default — expert picks differ per row, weights not
shareable; batched gather variant saves only ~10ms and reorders the weighted
sum, opt-in `LLM170_NP_MOE_BATCH=1`), gdn 26.4, hc up 16.0, rms 18.5,
qsa 14.1, down 8.5. llama-server reference on the same host: 39.4 — the
remaining gap is expert-id dedup/grouped MoE GEMMs plus small-kernel work.
