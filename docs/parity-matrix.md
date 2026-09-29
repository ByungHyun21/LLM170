# vk/hip operation parity matrix

Same-concept kernels and paths across the two GPU backends, as of the
refactor-107 campaign (2026-09-28). "✔" = implemented and gate-verified on
that backend; "—" = absent (either never ported or intentionally backend-
specific); "⚠" = present but gated/limited (see notes).

| Concept | hip (`rawhip`) | vk (`rawvk`) | Notes |
|---|---|---|---|
| Dense decode GEMV (q5_1/q8_0/q4k/q6k…) | ✔ `gemv_tile` family | ✔ `gemv8`/dmmv family | bit-matched to W4A8 mirror |
| Dense prefill tile (f32) | ✔ j128/v4 CO | ✔ `fn_tile_f32`/f32s | |
| MoE routing (top-k) | ✔ `moe_top10` | ✔ `moe_top10` frame op | |
| MoE grouped GEMM | ✔ expert-loop + mmq | ✔ ids2 dmmv + mmq tiles (llmmq promoted) | |
| MoE weighted sum + scatter | ✔ | ✔ | |
| GDN conv ring | ✔ | ✔ | |
| GDN AR (sequential/chunked) | ✔ ar/chunk kernels | ✔ `gdn_ar` family | chunk invariance gate-verified |
| QSA attention | ✔ qsa_flash family | ✔ flash family | |
| RMS / silu / sigmoid / scale EW | ✔ | ✔ | |
| Speculative (MTP) chain + verify | ✔ `mtp_step_g` D2H-light | ✔ GPU argmax verify (spec2 fixed, 2.93 t/s) | vk verify no longer transfers vocab logits |
| GDN snapshot/restore (rollback) | ✔ device copy | ✔ device copy (`copy_dev`) | |
| Prefill chunk pipelining | ✔ | partial | host-side chunk loop; overlap = plans/107 W1.5-3/4 |
| qwen4exp decode frame | ✔ q4acc | ✔ VkAcc frame | charhash-protected |
| res_hc f16 bus (RESF16) | — (q4 hip uses f32 res) | ✔ promoted default (ledger 88) | |
| Vulkan-only experiments (HCF16/MOEH16/…) | — | removed | ledgers 86, 90 |

Known asymmetries (open):

1. **qwen35 vk decode nondeterminism** — gated to loud hip fallback
   (ledger 87/90/92). Root cause suspected at driver/scheduler level;
   per-dispatch submission is deterministic.
2. **FN hip decode 5.5 t/s** vs vk 18 t/s history — hip default path runs
   W4A8 per-step quantization where vk's dmmv skips quant; porting the dmmv
   family to hip is an arithmetic-class change (drift gate + user approval
   required, plans/107 W1.5-7).
3. Prefill upload overlap (W1.5-3) and chunk scheduler tuning (W1.5-6) are
   unexplored on both backends.
