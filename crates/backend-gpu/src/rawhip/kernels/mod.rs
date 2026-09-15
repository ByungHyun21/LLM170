//! 원시 HIP 커널 소스 — core 미러(dot_row_w4a8_*_lane)와 동일 연산열.
//! 스트라이드 레인(레인 l: 서브블록 l, l+64, …), f64 부분합, 레인 순서
//! 합 후 1회 f32 캐스트. 그룹핑 무관 비트 일치 설계 (2026-09-02 k2).
//! plans/35 P3: 단일 문자열 → 패밀리별 텍스트 자산 분할, include_str! 조립
//! (SRC 바이트 불변 — 분할 해시 대조 검증).

pub const SRC: &str = concat!(
    include_str!("src_common.hip"),
    include_str!("src_quant.hip"),
    include_str!("src_gemv.hip"),
    include_str!("src_gemv4.hip"),
    include_str!("src_gemm.hip"),
    include_str!("src_probe.hip"),
    include_str!("src_ew.hip"),
    include_str!("src_gdn.hip"),
    include_str!("src_qsa.hip"),
    include_str!("src_vit.hip"),
    include_str!("src_ms.hip"),
    include_str!("src_q4.hip"),
);

pub const NAMES: &[&str] = &[
    "quant_q8", "q4_gemm_q5_1", "q4_gemm_f32", "q4_silu_div", "q4_sigmoid", "q4_scale", "q4_l2_rows", "q4_hc_gate_mean", "q4_hc_combine", "q4_norm_gated_sig", "q4_moe_top10", "q4_moe_group_t1", "bw_strided", "q4_moe_top10_m", "q4_moe_weighted_sum", "q4_moe_gather", "q4_moe_scatter", "q4_rows_permute_u32", "q4_gemm_q5_1_t", "q4_gemm_q5_1_m", "q4_gemm_q5_1_gm", "q4_gemm_q5_1_gm_ids", "q4_gemm_f32_m", "q4_gemm_q4k_m", "q4_gemm_q4k_ge", "q4_gemm_q4k_ge_ids", "q4_gemm_q4k_ids_reduce", "q4_qsa_attn_wt", "q4_qsa_attn_sel", "q4_qsa_attn_sel4", "q4_qsa_attn_sel6", "q4_qsa_attn_sel4s", "q4_qsa_attn_sel4s_merge", "q4_gemm_q4k_g", "q4_gemm_q4k_x", "q4_gemm_q4k_y", "q4_axpy_scaled_t", "q4_gdn_ar_w", "silu_mul_f32", "dequant_q6k_f16", "dequant_q4k_f16", "dequant_q8_0_f16", "requant_q6k_canonical", "rmsq", "gemm_q5k2", "silu_mulq", "gatedq",
    "gemm_xs", "row_shift_gather", "q6k_ref_scalar", "gemm_q5k", "gemm_q8_0", "gemm_q4k", "gemm_q6k", "gemm_nl", "gemm_q3k",
    "silu_mul", "axpy_scaled", "copy_rows", "bcast_rows", "rms_part", "rms_finish", "qk_norm_rope",
    "gdn_conv", "gdn_beta_g", "gdn_beta_g_f32", "norm_gated_silu", "norm_gated_silu_f32", "l2_rows2_scale", "split3",
    "qsa_score", "qsa_mix2", "qsa_flash", "qsa_flash_split4q4", "qsa_flash_gqa", "qsa_flash_gqa2", "qsa_flash_gqa2h", "kv_f16", "qsa_flash_gqa2d", "qsa_flash_wk", "qsa_flash_wk16", "qsa_flash_wk8", "wmma_probe", "wmma_probe_ldm", "wmma_probe_pv", "qsa_flash_wmma", "qsa_flash_merge", "mfma_roof", "gemm_iq3s", "dp4a_probe", "bw_probe", "gdn_conv_t", "gdn_conv_t2", "gdn_conv_t2_f32", "gdn_conv_state", "gdn_ar_w_swap", "gemm_q5k_v2", "gemm_q8_0_dual", "gemm_q5k4", "gemm_xs4", "gemm_q4k4", "gemm_q6k4", "cat2_rows", "gdn_ar_sm", "gdn_ar_t", "gdn_ar_w", "gdn_ar_chunk_a", "gdn_ar_chunk_b", "gdn_ar_chunk_c2", "l2_rows2_scale_w", "kv_append_t", "dot_roof", "gemm_q5k_mm", "gemm_q5k_wm", "dequant_f16_q5k", "gemm_q5k_wc", "dequant_f16_xs", "gemm_xs_wc", "dequant_f16_xs_dbg", "q4_shexp_gu", "q4_shexp_da", "gemm_q8_0_w256", "gemm_q4k_wm", "gemm_q6k_wm", "gemm_xs_wm", "gemm_q4k_mm", "gemm_q6k_mm", "gemm_xs_mm", "argmax64", "gdn_conv_t2_ms", "gdn_conv_t2_ms_f32", "gdn_conv_state_ms", "add_f32", "pack_strided", "gemm_f32t", "layernorm_t", "vit_rope", "flash_vit", "gelu_t",
    "q4_idx_q_rope", "q4_idx_bk_update", "q4_idx_score", "q4_idx_rank", "q4_idx_expand", "q4_gemm_f32_w", "q4_idx_topk", "gemm_q8_0_w", "q4_gemm_q5_1_w_ids", "gemm_q8_0_ids",
];
// ─── 원시 HIP ew 계열 (큐브cl ew.rs 산술 이식, 다음 검증 대상) ───


