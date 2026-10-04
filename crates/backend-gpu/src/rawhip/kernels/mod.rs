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
    include_str!("src_exl3.hip"),
    include_str!("src_ew.hip"),
    include_str!("src_gdn.hip"),
    include_str!("src_qsa.hip"),
    include_str!("src_vit.hip"),
    include_str!("src_ms.hip"),
    include_str!("src_q4.hip"),
);

pub const NAMES: &[&str] = &[
    "exl3_had_in",
    "exl3_gemv",
    "exl3_had_out",
    "exl3_norm_resid",
    "exl3_norm_resid_p",
    "exl3_gemm2",
    "exl3_gemm2_kseg",
    "exl3_gemm2_mma",
    "exl3_pos_bump",
    "exl3_gdn_conv",
    "exl3_gdn_l2perm",
    "exl3_gdn_scan",
    "exl3_attn_prep",
    "exl3_attn_fwd3",
    "exl3_attn_fwd3s",
    "exl3_ew",
    "exl3_argmax",
    "exl3_gdn_gate",
    "quant_q8",
    "q4_gemm_q5_1",
    "q4_gemm_f32",
    "q4_silu_div",
    "q4_sigmoid",
    "q4_scale",
    "q4_l2_rows",
    "q4_hc_gate_mean",
    "q4_hc_gate",
    "q4_hc_combine",
    "q4_norm_gated_sig",
    "q4_moe_group_t1",
    "bw_strided",
    "q4_moe_top10_m",
    "q4_moe_weighted_sum",
    "q4_moe_gather",
    "q4_moe_scatter",
    "q4_rows_permute_u32",
    "q4_gemm_q5_1_t",
    "q4_gemm_q5_1_m",
    "q4_gemm_q5_1_gm",
    "q4_gemm_q5_1_gm_ids",
    "q4_gemm_f32_m",
    "q4_gemm_q4k_m",
    "q4_gemm_q4k_ge",
    "q4_gemm_q4k_ge_ids",
    "q4_gemm_q4k_ids_reduce",
    "q4_qsa_attn_wt",
    "q4_qsa_attn_sel",
    "q4_qsa_attn_sel4",
    "q4_qsa_attn_sel6",
    "q4_qsa_attn_sel4s",
    "q4_qsa_attn_sel4s_merge",
    "q4_gemm_q4k_g",
    "q4_axpy_scaled_t",
    "silu_mul_f32",
    "dequant_q6k_f16",
    "dequant_q4k_f16",
    "dequant_q8_0_f16",
    "rmsq",
    "gemm_q5k2",
    "silu_mulq",
    "gatedq",
    "gemm_xs",
    "row_shift_gather",
    "q6k_ref_scalar",
    "gemm_q5k",
    "gemm_q8_0",
    "gemm_q4k",
    "gemm_q6k",
    "gemm_nl",
    "gemm_q3k",
    "silu_mul",
    "axpy_scaled",
    "copy_rows",
    "bcast_rows",
    "rms_part",
    "rms_finish",
    "qk_norm_rope",
    "gdn_conv",
    "gdn_beta_g",
    "gdn_beta_g_f32",
    "norm_gated_silu_f32",
    "l2_rows2_scale",
    "split3",
    "qsa_score",
    "qsa_mix2",
    "qsa_flash",
    "qsa_flash_split4q4",
    "qsa_flash_gqa",
    "qsa_flash_gqa2",
    "qsa_flash_gqa2h",
    "kv_f16",
    "qsa_flash_gqa2d",
    "qsa_flash_wk",
    "qsa_flash_wk16",
    "qsa_flash_wk8",
    "qsa_flash_merge",
    "gemm_iq3s",
    "dp4a_probe",
    "bw_probe",
    "q4_gemm_f32_w2",
    "gemm_mix_dual",
    "gdn_conv_t",
    "gdn_conv_t2",
    "gdn_conv_t2_f32",
    "gdn_conv_state",
    "gdn_ar_w_swap",
    "gemm_q5k_v2",
    "gemm_q8_0_dual",
    "gemm_q5k4",
    "gemm_xs4",
    "gemm_q4k4",
    "gemm_q6k4",
    "cat2_rows",
    "gdn_ar_t",
    "gdn_ar_w",
    "gdn_ar_chunk_a",
    "gdn_ar_chunk_b",
    "gdn_ar_chunk_c2",
    "l2_rows2_scale_w",
    "kv_append_t",
    "gemm_q5k_mm",
    "gemm_q5k_wm",
    "dequant_f16_q5k",
    "q4_idx_q_rope",
    "q4_idx_bk_update",
    "q4_idx_score",
    "q4_idx_rank",
    "q4_idx_expand",
    "q4_gemm_f32_w",
    "q4_idx_topk",
    "gemm_q8_0_w",
    "gemm_q8_0_mt",
    "gemm_q8_0_mt_w",
    "gemm_q8_0_w4",
    "gemm_q8_0_w16",
    "gemm_q8_0_dmmv",
    "q4_gemm_q4k_dmmv_ids",
    "q5_1_gemm_dmmv_ids",
    "q4_gemm_q5_1_w_ids",
    "gemm_q8_0_ids",
    "qsa_flash_wk8i",
    "rms_small",
    "q4_ple_gate",
    "q4_ple_conv",
    "q4_ple_residual",
    "q4_emb_q8g",
    "q4_emb_q8g_f16", // A4(plans/129) 정합 테스트 포착 — value.rs f16 경로가 런치하나 미등록이었다(런타임 실패)
    "q4_ple_gather",
    "argmax_rows_s1",
    "argmax_rows_s2",
    "gdn_ar_w_np",
    "gdn_conv_np",
    "gemm_q8_0_mt16",
    "gemm_q5k4_w2",
    "gemm_q4k4_w2",
    "qsa_flash_wmma2v2",
    "q4_gemm_f32_mt",
    "q4_shexp_gu",
    "q4_shexp_da",
    "q4_identity_sel",
    "q4_gemm_q5k_gm",
    "q4_gemm_q8_gm",
    "gdn_split_l2_scale",
    "q4_logits_topk_cand",
    "q4_shexp_gu_t",
    "q4_shexp_da_t",
    "add_f32",
    "argmax64",
    "dequant_f16_xs",
    "dequant_f16_xs_dbg",
    "flash_vit",
    "gdn_conv_state_ms",
    "gdn_conv_t2_ms_f32",
    "gelu_t",
    "gemm_f32t",
    "gemm_q4k_mm",
    "gemm_q4k_wm",
    "gemm_q5k_wc",
    "gemm_q6k_mm",
    "gemm_q6k_wm",
    "gemm_xs_mm",
    "gemm_xs_wc",
    "gemm_xs_wm",
    "layernorm_t",
    "pack_strided",
    "vit_rope",
];
// ─── 원시 HIP ew 계열 (큐브cl ew.rs 산술 이식, 다음 검증 대상) ───
// 마커 hip1
// 마커 hip2

#[cfg(test)]
mod tests {
    //! A4(plans/129): 커널 이중 등록 계약의 정적 테스트 — NAMES 중복 0,
    //! NAMES↔SRC extern 정의 쌍방 정합. 무GPU(컴파일 타임 문자열 자산).

    use super::{NAMES, SRC};

    /// SRC에서 extern "C" __global__ 커널 정의명 추출.
    fn extern_defs() -> Vec<&'static str> {
        let mut out = Vec::new();
        let mut rest = SRC;
        while let Some(i) = rest.find("__global__ void ") {
            rest = &rest[i + "__global__ void ".len()..];
            // __launch_bounds__(N) 수식자 스킵(정의명이 아님).
            if let Some(rb) = rest.strip_prefix("__launch_bounds__") {
                rest = &rb[rb.find(')').map(|p| p + 1).unwrap_or(0)..];
                rest = rest.trim_start();
            }
            // exl3_tile_word 같은 __device__ inline은 제외됨(전역 아님).
            if let Some(name) = rest.split(['(', ' ', '\n']).next()
                && !name.is_empty()
                && name
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
            {
                out.push(name);
            }
            rest = &rest[1.min(rest.len())..];
        }
        out
    }

    #[test]
    fn names_no_duplicates() {
        let mut s = NAMES.to_vec();
        s.sort_unstable();
        let dups: Vec<&str> = s
            .windows(2)
            .filter(|w| w[0] == w[1])
            .map(|w| w[0])
            .collect();
        assert!(dups.is_empty(), "NAMES 중복: {dups:?}");
    }

    #[test]
    fn names_all_defined_in_src() {
        let defs = extern_defs();
        for n in NAMES {
            assert!(defs.contains(n), "NAMES에 등록됐으나 SRC에 정의 없음: {n}");
        }
    }

    #[test]
    fn src_defs_all_registered() {
        let defs = extern_defs();
        let unreg: Vec<&str> = defs
            .iter()
            .copied()
            .filter(|d| !NAMES.contains(d))
            .collect();
        assert!(
            unreg.is_empty(),
            "SRC 정의 중 NAMES 미등록(AGENTS: 누락 시 삭제 대상): {unreg:?}"
        );
    }
}
