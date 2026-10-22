
pub struct Asset {
    pub name: &'static str,
    pub env: &'static str,
    pub paths: &'static [&'static str],
    pub syms: &'static [&'static str],
    pub boot: bool,
}

pub const ASSETS: &[Asset] = &[
    Asset {
        name: "gptq4",
        env: "LLM170_CUDA_GPTQ4_FATBIN_PATH",
        paths: &[
            "crates/backend-gpu/src/rawcuda/assets/gptq4.fatbin",
            "src/rawcuda/assets/gptq4.fatbin",
        ],
        syms: &[
            "w4a16_copy",
            "w4a16_gemv_experts_glu",
            "w4a16_gemv_experts_glu_gelu",
            "w4a16_gemm_g128",
            "w4a16_gemv_g128",
            "w4a16_gemv_g128_multi",
            "w4a16_gemv_g128_multi_t",
            "w4a16_gemv_g128_t",
            "w4a16_gemm_g32_bf16",
            "w4a16_gemv_g32_bf16",
            "w4a16_cast_x32",
            "w4a16_gemv_experts_g32_bf16",
            "w4a16_gemm_g32_mma_grp",
            "w4a16_gemm_g32_mma_grp2",
            "w4a16_gemv_g32_bf16_tall",
            "w4a16_gemv_g32_bf16_multi_t",
            "w4a16_moe_align",
            "w4a16_gemm_g128_mma",
            "w4a16_gemv_g32_bf16_t",
        ],
        boot: true,
    },
    Asset {
        name: "gemma4",
        env: "LLM170_CUDA_GEMMA4_FATBIN_PATH",
        paths: &[
            "crates/backend-gpu/src/rawcuda/assets/gemma4.fatbin",
            "src/rawcuda/assets/gemma4.fatbin",
        ],
        syms: &["w4a16_gemv_g32_bf16_tall2", "w4a16_gemv_g32_bf16_tall4"],
        boot: true,
    },
    Asset {
        name: "decode",
        env: "LLM170_CUDA_DECODE_FATBIN_PATH",
        paths: &[
            "crates/backend-gpu/src/rawcuda/assets/decode.fatbin",
            "src/rawcuda/assets/decode.fatbin",
        ],
        syms: &[
            "w4a16_gemv_sk_marlin",
            "w4a16_gemv_sk_reduce",
            "w4a16_gemv_sk1_marlin",
            "w4a16_gemv_sk1_reduce",
        ],
        boot: true,
    },
    Asset {
        name: "moe",
        env: "LLM170_CUDA_MOE_FATBIN_PATH",
        paths: &[
            "crates/backend-gpu/src/rawcuda/assets/moe.fatbin",
            "src/rawcuda/assets/moe.fatbin",
        ],
        syms: &[
            "w4a16_gemv_bf16",
            "w4a16_gemv_multi",
            "w4a16_gemv_bf16_t",
            "w4a16_gemv_bf16_multi_t",
            "w4a16_gemv_bf16_multi_t_v4",
            "w4a16_cast_bf16",
            "w4a16_cast_hilo",
            "w4a16_gemm_bf16_mma",
            "w4a16_gemm_bf16_mma16",
            "w4a16_gemm_bf16",
            "w4a16_gemm_bf16_t",
            "w4a16_axpy",
            "w4a16_moe_accum",
            "w4a16_moe_pes_scale",
            "w4a16_moe_topk_t",
            "w4a16_moe_topk256",
            "w4a16_moe_align2",
            "w4a16_h2f",
            "w4a16_moe_align_marlin",
            "w4a16_moe_act_gelu_f16",
            "w4a16_moe_act_silu_f16",
            "w4a16_moe_topk_sum_f16",
            "w4a16_shared_add",
            "w4a16_moe_accum_sg",
            "w4a16_gemm_g32_mma_grp_d64",
            "w4a16_gemm_g128_marlin",
            "w4a16_gemm_g32_marlin",
            "w4a16_gemm_g128_marlin_m16",
            "w4a16_scale",
        ],
        boot: true,
    },
    Asset {
        name: "moepp",
        env: "LLM170_CUDA_MOEPP_FATBIN_PATH",
        paths: &[
            "crates/backend-gpu/src/rawcuda/assets/moepp.fatbin",
            "src/rawcuda/assets/moepp.fatbin",
        ],
        syms: &[
            "w4a16_cast_f16",
            "w4a16_gemm_bf16_mma2",
            "w4a16_gemm_bf16_mma2_16",
            "w4a16_gemm_bf16_mma3",
            "w4a16_gemm_g32_mma_grp2_f16",
            "w4a16_gemm_g32_mma_grp_d64_f16",
            "w4a16_gemm_g32_mma_grp2_sf16",
            "w4a16_gemm_g32_mma_grp_d64_sf16",
        ],
        boot: true,
    },
    Asset {
        name: "fn",
        env: "LLM170_CUDA_FN_FATBIN_PATH",
        paths: &[
            "crates/backend-gpu/src/rawcuda/assets/fn.fatbin",
            "src/rawcuda/assets/fn.fatbin",
        ],
        syms: &[
            "fn_hc_init",
            "fn_hc_norm",
            "fn_hc_silu",
            "fn_hc_mix",
            "fn_hc_write",
            "fn_ple_gate",
            "fn_ple_conv",
            "fn_ple_conv_wb",
            "gdn_gate_sig",
            "w4a16_gemv_experts_g128_f16",
            "w4a16_gemv_experts_g32_f16",
            "attn_qsa_part",
        ],
        boot: true,
    },
    Asset {
        name: "norm",
        env: "LLM170_CUDA_NORM_FATBIN_PATH",
        paths: &[
            "crates/backend-gpu/src/rawcuda/assets/norm.fatbin",
            "src/rawcuda/assets/norm.fatbin",
        ],
        syms: &["norm_resid", "norm_resid_g", "rms_rows"],
        boot: true,
    },
    Asset {
        name: "gdn",
        env: "LLM170_CUDA_GDN_FATBIN_PATH",
        paths: &[
            "crates/backend-gpu/src/rawcuda/assets/gdn.fatbin",
            "src/rawcuda/assets/gdn.fatbin",
        ],
        syms: &[
            "gdn_conv",
            "gdn_l2perm",
            "gdn_l2perm8",
            "gdn_wu",
            "gdn_h",
            "gdn_o",
            "gdn_spec_scan",
            "gdn_scan_akq",
            "gdn_wy_solve",
            "gdn_wy_reconstruct",
            "gdn1_part",
            "gdn1_comb",
            "gdn1_upd",
            "gdn_rescale",
            "gdn_gate",
        ],
        boot: true,
    },
    Asset {
        name: "attn",
        env: "LLM170_CUDA_ATTN_FATBIN_PATH",
        paths: &[
            "crates/backend-gpu/src/rawcuda/assets/attn.fatbin",
            "src/rawcuda/assets/attn.fatbin",
        ],
        syms: &[
            "attn_fwd3s_part_q",
            "attn_fwd3s_part_qg",
            "attn_prep_q",
            "attn_fwd3s_merge",
            "attn_fa2_qg",
            "attn_prep_g4",
            "attn_fwd3s_part_g4",
            "attn_fa2_g4_512",
            "attn_fwd3s_merge_g4",
        ],
        boot: true,
    },
    Asset {
        name: "marlin_moe",
        env: "LLM170_CUDA_MARLIN_MOE_FATBIN_PATH",
        paths: &[
            "crates/backend-gpu/src/rawcuda/assets/marlin_moe.fatbin",
            "src/rawcuda/assets/marlin_moe.fatbin",
        ],
        syms: &[
            "_ZN16marlin_moe_wna166MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi1ELi4ELi8ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_PKiS8_S8_S6_ibiiiPibbb",
            "_ZN16marlin_moe_wna166MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi1ELi4ELi8ELb1ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_PKiS8_S8_S6_ibiiiPibbb",
            "_ZN16marlin_moe_wna166MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi1ELi8ELi4ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_PKiS8_S8_S6_ibiiiPibbb",
            "_ZN16marlin_moe_wna166MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi1ELi8ELi4ELb1ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_PKiS8_S8_S6_ibiiiPibbb",
            "_ZN16marlin_moe_wna166MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi2ELi4ELi8ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_PKiS8_S8_S6_ibiiiPibbb",
            "_ZN16marlin_moe_wna166MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi2ELi8ELi4ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_PKiS8_S8_S6_ibiiiPibbb",
            "_ZN16marlin_moe_wna166MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi3ELi4ELi8ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_PKiS8_S8_S6_ibiiiPibbb",
            "_ZN16marlin_moe_wna166MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi3ELi8ELi4ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_PKiS8_S8_S6_ibiiiPibbb",
            "_ZN16marlin_moe_wna166MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi4ELi4ELi8ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_PKiS8_S8_S6_ibiiiPibbb",
            "_ZN16marlin_moe_wna166MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi4ELi8ELi4ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_PKiS8_S8_S6_ibiiiPibbb",
            "_ZN16marlin_moe_wna166MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi256ELi1ELi8ELi8ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_PKiS8_S8_S6_ibiiiPibbb",
            "_ZN16marlin_moe_wna166MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi256ELi1ELi8ELi8ELb1ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_PKiS8_S8_S6_ibiiiPibbb",
            "_ZN16marlin_moe_wna166MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi256ELi2ELi16ELi4ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_PKiS8_S8_S6_ibiiiPibbb",
            "_ZN16marlin_moe_wna166MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi256ELi3ELi16ELi4ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_PKiS8_S8_S6_ibiiiPibbb",
            "_ZN16marlin_moe_wna166MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi256ELi4ELi16ELi4ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_PKiS8_S8_S6_ibiiiPibbb",
        ],
        boot: true,
    },
    Asset {
        name: "marlin",
        env: "LLM170_CUDA_MARLIN_FATBIN_PATH",
        paths: &[
            "crates/backend-gpu/src/rawcuda/assets/marlin.fatbin",
            "src/rawcuda/assets/marlin.fatbin",
        ],
        syms: &[
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi1ELi4ELi8ELb0ELi4ELi8ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi1ELi8ELi4ELb0ELi4ELi8ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi2ELi4ELi8ELb0ELi4ELi8ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi2ELi8ELi4ELb0ELi4ELi8ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi3ELi4ELi8ELb0ELi4ELi8ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi3ELi8ELi4ELb0ELi4ELi8ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi4ELi4ELi8ELb0ELi4ELi8ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi4ELi8ELi4ELb0ELi4ELi8ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi256ELi1ELi16ELi4ELb0ELi4ELi8ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi256ELi1ELi8ELi8ELb0ELi4ELi8ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi256ELi2ELi16ELi4ELb0ELi4ELi8ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi256ELi3ELi16ELi4ELb0ELi4ELi8ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi256ELi4ELi16ELi4ELb0ELi4ELi8ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
        ],
        boot: true,
    },
    Asset {
        name: "marlin_g32",
        env: "LLM170_CUDA_MARLIN_G32_FATBIN_PATH",
        paths: &[
            "crates/backend-gpu/src/rawcuda/assets/marlin_g32.fatbin",
            "src/rawcuda/assets/marlin_g32.fatbin",
        ],
        syms: &[
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi1ELi4ELi8ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi1ELi8ELi4ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi2ELi4ELi8ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi2ELi8ELi4ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi3ELi4ELi8ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi3ELi8ELi4ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi4ELi4ELi8ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi128ELi4ELi8ELi4ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi256ELi1ELi16ELi4ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi256ELi1ELi8ELi8ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi256ELi2ELi16ELi4ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi256ELi3ELi16ELi4ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
            "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi256ELi4ELi16ELi4ELb0ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
        ],
        boot: true,
    },
    Asset {
        name: "head",
        env: "LLM170_CUDA_HEAD_FATBIN_PATH",
        paths: &[
            "crates/backend-gpu/src/rawcuda/assets/head.fatbin",
            "src/rawcuda/assets/head.fatbin",
        ],
        syms: &[
            "w4a16_argmax_min",
            "w4a16_argmax_min_t",
            "w4a16_gemv_i8",
            "w4a16_gemv_i8_t",
        ],
        boot: true,
    },
    Asset {
        name: "ew",
        env: "LLM170_CUDA_EW_FATBIN_PATH",
        paths: &[
            "crates/backend-gpu/src/rawcuda/assets/ew.fatbin",
            "src/rawcuda/assets/ew.fatbin",
        ],
        syms: &["ew", "ew_gelu"],
        boot: true,
    },
    Asset {
        name: "embed",
        env: "LLM170_CUDA_EMBED_FATBIN_PATH",
        paths: &[
            "crates/backend-gpu/src/rawcuda/assets/embed.fatbin",
            "src/rawcuda/assets/embed.fatbin",
        ],
        syms: &[
            "w4a16_embed_gather",
            "w4a16_mtp_chain_step",
            "w4a16_kstep_ctrl",
        ],
        boot: true,
    },
    Asset {
        name: "smoke",
        env: "LLM170_CUDA_SMOKE_FATBIN_PATH",
        paths: &[
            "crates/backend-gpu/src/rawcuda/assets/smoke.fatbin",
            "src/rawcuda/assets/smoke.fatbin",
        ],
        syms: &["llm170_mma_smoke"],
        boot: false,
    },
];

pub fn asset(name: &str) -> &'static Asset {
    ASSETS
        .iter()
        .find(|a| a.name == name)
        .unwrap_or_else(|| panic!("자산 매니페스트에 없음: {name}"))
}

pub fn asset_bytes(a: &Asset) -> Result<Vec<u8>, String> {
    if let Some(p) = llm170_diag::flag::val(a.env) {
        return std::fs::read(p).map_err(|e| format!("{}({p}) 읽기 실패: {e}", a.env));
    }
    for r in a.paths {
        if let Ok(b) = std::fs::read(r) {
            return Ok(b);
        }
    }
    Err(format!("자산 부재 — {:?} 또는 {}", a.paths, a.env))
}

pub fn cu_source(name: &str) -> &'static str {
    match name {
        "gptq4" => include_str!("assets/gptq4.cu"),
        "gemma4" => include_str!("assets/gemma4.cu"),
        "moe" => include_str!("assets/moe.cu"),
        "moepp" => include_str!("assets/moepp.cu"),
        "decode" => include_str!("assets/decode.cu"),
        "norm" => include_str!("assets/norm.cu"),
        "gdn" => include_str!("assets/gdn.cu"),
        "attn" => include_str!("assets/attn.cu"),
        "head" => include_str!("assets/head.cu"),
        "ew" => include_str!("assets/ew.cu"),
        "embed" => include_str!("assets/embed.cu"),
        "fn" => include_str!("assets/fn.cu"),
        "smoke" => include_str!("assets/smoke.cu"),
        "marlin" => include_str!("assets/marlin.cu"),
        "marlin_moe" => include_str!("assets/marlin_moe.cu"),
        _ => panic!("cu_source: 미지 자산 {name}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cu_symbols(src: &str) -> Vec<String> {
        let mut out = Vec::new();
        for line in src.lines() {
            let Some(rest) = line.trim().strip_prefix("extern \"C\" __global__ void ") else {
                continue;
            };
            let rest = if let Some(r2) = rest.strip_prefix("__launch_bounds__") {
                match r2.find(')') {
                    Some(close) => r2[close + 1..].trim(),
                    None => continue,
                }
            } else {
                rest
            };
            if let Some(sym) = rest.split('(').next() {
                out.push(sym.trim().to_string());
            }
        }
        out
    }

    #[test]
    fn asset_manifest_matches_cu_symbols() {
        for a in ASSETS {
            if a.name == "marlin" {
                assert_eq!(a.syms.len(), 13, "marlin 인스턴스 13종 계약");
                for s in a.syms {
                    assert!(
                        s.starts_with("_ZN6marlin6MarlinILl"),
                        "marlin 심볼 형태 위반: {s}"
                    );
                }
                continue;
            }
            if a.name == "marlin_g32" {
                assert_eq!(a.syms.len(), 13, "marlin_g32 인스턴스 13종 계약");
                for s in a.syms {
                    assert!(
                        s.starts_with("_ZN6marlin6MarlinILl"),
                        "marlin_g32 심볼 형태 위반: {s}"
                    );
                }
                continue;
            }
            if a.name == "marlin_moe" {
                assert_eq!(a.syms.len(), 15, "marlin_moe 인스턴스 15종 계약");
                for s in a.syms {
                    assert!(
                        s.starts_with("_ZN16marlin_moe_wna166MarlinILl"),
                        "marlin_moe 심볼 형태 위반: {s}"
                    );
                }
                continue;
            }
            let syms = cu_symbols(cu_source(a.name));
            for s in a.syms {
                assert!(
                    syms.iter().any(|x| x == s),
                    "{}: 매니페스트 심볼 {s}가 .cu에 없음",
                    a.name
                );
            }
            for s in &syms {
                assert!(
                    a.syms.contains(&s.as_str()),
                    "{}: .cu 심볼 {s}가 매니페스트에 없음(로드 누락)",
                    a.name
                );
            }
        }
    }

    #[test]
    fn asset_dir_prefix_consistent() {
        for a in ASSETS {
            for p in a.paths {
                assert!(
                    p.starts_with("crates/backend-gpu/src/rawcuda/assets/")
                        || p.starts_with("src/"),
                    "{}: 경로 표기 규약 위반 {p}",
                    a.name
                );
            }
        }
    }
}
