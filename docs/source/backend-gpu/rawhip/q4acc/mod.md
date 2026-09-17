# `crates/backend-gpu/src/rawhip/q4acc.rs` — measurement record

## Index (2026-09-15 regroup — details in the dated sections below)

**Adopted (in production paths)**: device-side QSA selection (`q4_idx_*`,
pools, bitonic topk) · warp GEMV family (`q4_gemm_f32_w`,
`q4_gemm_q5_1_w_ids`, `gemm_q8_0_ids`, `gemm_q8_0_w`) with original-class
reductions · PLE device path (`q4_ple_gate/conv/residual`, `exp_cr_exact`)
· const-upload caching · Q5_1/Q8_0 direct-ids MoE routing.

**Rejected (kept as opt-in assets, env names in the sections)**: lane-0-only
`__shfl_sync` reduction (divergent-warp HW exception) · f32 warp-tree
reduction for q5_1 (gate token flip) · shared-tile attention (bank
conflicts + occupancy) · rms_small fusion · gemm_q5k_v2 · q8_0 MMQ pair ·
per-warp query-only PV write structure (must cover all 8 queries).



> **Note**: The detailed measurement prose below was written in Korean (the
> project's working language during development). Section titles and key
> conclusions are in English; full translation of the prose is planned as
> the document stabilizes. Numbers, tables, and code are language-neutral.

> Items from `docs/benchmarks.md` specific to this file (by section title).
> Tables and numbers verbatim. Summary metrics remain in benchmarks.md.


> **Split (2026-09-17, plans/78 R1)**: the source file
> `crates/backend-gpu/src/rawhip/q4acc.rs` was decomposed into the `q4acc/`
> module directory (mod/value/moe/frame/qsa/checks). This measurement record
> moved with it, partitioned by topic — attention/QSA → `qsa.md`, MoE
> grouping → `moe.md`, staging/upload/launch machinery + plans/73 kernel
> rounds → `value.md`. All sections verbatim; this index covers them all.

