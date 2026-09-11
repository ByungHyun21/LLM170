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
);

pub const NAMES: &[&str] = &[
    "quant_q8", "silu_mul_f32", "dequant_q6k_f16", "requant_q6k_canonical", "rmsq", "gemm_q5k2", "silu_mulq", "gatedq",
    "gemm_xs", "row_shift_gather", "q6k_ref_scalar", "gemm_q5k", "gemm_q8_0", "gemm_q4k", "gemm_q6k", "gemm_nl", "gemm_q3k",
    "silu_mul", "axpy_scaled", "copy_rows", "rms_part", "rms_finish", "qk_norm_rope",
    "gdn_conv", "gdn_beta_g", "gdn_beta_g_f32", "norm_gated_silu", "norm_gated_silu_f32", "l2_rows2_scale", "split3",
    "qsa_score", "qsa_mix2", "qsa_flash", "qsa_flash_split4q4", "qsa_flash_wk", "qsa_flash_merge", "mfma_roof", "gemm_iq3s", "dp4a_probe", "bw_probe", "gdn_conv_t", "gdn_conv_t2", "gdn_conv_t2_f32", "gdn_conv_state", "gdn_ar_w_swap", "gemm_q5k_v2", "gemm_q8_0_dual", "gemm_q5k4", "gemm_xs4", "gemm_q4k4", "gemm_q6k4", "cat2_rows", "gdn_ar_sm", "gdn_ar_t", "gdn_ar_w", "gdn_ar_chunk_a", "gdn_ar_chunk_b", "gdn_ar_chunk_c2", "l2_rows2_scale_w", "kv_append_t", "dot_roof", "gemm_q5k_mm", "gemm_q5k_wm", "gemm_q4k_wm", "gemm_q6k_wm", "gemm_xs_wm", "gemm_q4k_mm", "gemm_q6k_mm", "gemm_xs_mm", "argmax64", "gdn_conv_t2_ms", "gdn_conv_t2_ms_f32", "gdn_conv_state_ms", "add_f32", "pack_strided", "gemm_f32t", "layernorm_t", "vit_rope", "flash_vit", "gelu_t",
];
// ─── 원시 HIP ew 계열 (큐브cl ew.rs 산술 이식, 다음 검증 대상) ───


