//! fatbin 자산 매니페스트 — [R9 2026-10-10] 모듈명·env 키·후보 경로·심볼
//! 목록의 **단일 출처**. 종전에는 w4a16_dec(6)·probe(smoke)·gptq4(자체 해석)
//! 3곳에 목록이 흩어져 커널 추가 시 동시 수정이 필요했다(int4 커널 실측).
//!
//! [R24] `asset_manifest_matches_cu_symbols` 테스트가 각 .cu의
//! `extern "C" __global__` 심볼과 이 표를 **양방향 자동 대조**한다 —
//! 심볼 누락/오타가 런타임 load 실패 대신 테스트에서 잡힌다.

/// fatbin 1개 — 로드 단위.
pub struct Asset {
    /// 모듈명(cuModuleGetFunction 대상 모듈).
    pub name: &'static str,
    /// env 경로 오버라이드 키(LLM170_CUDA_*_FATBIN_PATH).
    pub env: &'static str,
    /// 후보 경로(레포 루트 기준, env 미설정 시 순서대로 시도).
    pub paths: &'static [&'static str],
    /// 로드할 커널 심볼 전량(.cu 정의와 일치해야 함 — R24 테스트).
    pub syms: &'static [&'static str],
    /// 부팅(W4a16Dec::new) 로드 여부 — smoke는 프로브 지연 로드.
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
            // [P9 2026-10-17] 레거시 tall — 디스패치 미사용(tall2가 단일 경로).
            // gptq4.cu 바이트 동결(코드젠 고정, gemma4.cu 헤더 참조)로 커널이
            // fatbin에 잔존 — 심볼 등록만 유지(부팅 바인딩·발사 없음).
            "w4a16_gemv_g32_bf16_tall",
            // [⑮② 2026-10-18] 다중세그먼트(q/k/v·gate/up) — gemma4 청크 융합.
            "w4a16_gemv_g32_bf16_multi_t",
            // [2026-10-18] 레거시 — 디스패치 미사용(moe.cu align2가 단일 경로).
            // gptq4.cu 드로우 동결로 커널만 잔존(심볼 등록·발사 없음).
            "w4a16_moe_align",
            "w4a16_gemm_g128_mma",
            "w4a16_gemv_g32_bf16_t",
        ],
        boot: true,
    },
    Asset {
        // [P9 2026-10-17] gptq4.cu 분리 — gemma4 tall GEMV 계열(코드젠 고정,
        // gemma4.cu 헤더 주석 참조). -fmad=false 계약은 gptq4와 동일.
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
        // [split-k marlin 디코드 2026-10-17] t=1 GEMV 대체 — moe.cu m16 마린
        // 기반 k분할 커널(독립 .cu — 코드젠 고정, decode.cu 헤더 참조).
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
        // [R5 2026-10-10] gptq4.cu 분할 — MoE·플레인 bf16 계열.
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
            // [2026-10-18] 전문가-우선 순열 v2(gptq4.cu 원본의 직렬 스캔 대체).
            "w4a16_moe_align2",
            // [2026-10-18] 마린 f16 C → f32 스트림 변환.
            "w4a16_h2f",
            // [2026-10-19] 마린 MoE 정렬 변환(gslot→sorted_ids·expert_ids·num_post).
            "w4a16_moe_align_marlin",
            // [2026-10-19] 마린 MoE 활성(gelu_tanh·silu f16)·topk 합산.
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
        // [⑥ 2026-10-18] MoE 프리필 grp 커널의 A=f16 단일패스 변형 —
        // gptq4.cu/moe.cu 드로우 동결 유지용 독립 파일(.cu 헤더 참조).
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
        ],
        boot: true,
    },
    Asset {
        // [FN M1 2026-10-20] Qwen3.8-Flash-Next 전용 — hyper-connection
        // (Gated Residual)·PLE(게이트/숏컨브)·GDN sigmoid 게이트판·전문가
        // g128/f16 GEMV(gptq4.cu 바이트동결으로 자체 포함).
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
        // [2026-10-19] vLLM marlin_moe_wna16 벤더(Apache-2.0 — assets/marlin_moe/
        // 사본, 업스트림 대조 정본). 35B/gemma4 프리필 전문가 GEMM 후보.
        // 채리픽 = g32·f16 세트 15 인스턴스 — 맹글명(툴체인 계약).
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
        // [2026-10-18] vLLM 마린 벤더(Apache-2.0 — assets/marlin/ 사본, 업스트림
        // 대조 정본·변경 금지). 27B dense t≥16 W4A16 후보. 심볼 = 템플릿 인스턴스
        // 맹글명(툴체인·시그니처 계약 고정 — nvcc/템플릿 변경 시 재생성).
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
        // [2026-10-19] 마린 g32 밀집 — gemma4 dense(g32·bf16 스케일) t>16 전용.
        // assets/marlin/ 벤더 템플릿의 group_blocks=2 인스턴스 13종(g128 marlin과
        // 동일 채리픽 — 독립 .cu: marlin.cu 코드젠 재추첨 방지).
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
        // [S3-1 2026-10-11] 임베딩 행 gather — 프리필 행 조립 대체.
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

/// 이름으로 자산 조회 — 로드부가 env·경로를 재계산하지 않게 한다.
pub fn asset(name: &str) -> &'static Asset {
    ASSETS
        .iter()
        .find(|a| a.name == name)
        .unwrap_or_else(|| panic!("자산 매니페스트에 없음: {name}"))
}

/// 자산 바이트 해석 — env 오버라이드 우선, 후보 경로 순차 시도.
/// (종전 w4a16_dec::asset_bytes·gptq4::fatbin_bytes 중복 통합.)
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

/// [R24] 자산별 .cu 소스 — 심볼 대조 테스트용(컴파일 타임 포함).
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
        "smoke" => include_str!("assets/smoke.cu"),
        "marlin" => include_str!("assets/marlin.cu"),
        "marlin_moe" => include_str!("assets/marlin_moe.cu"),
        _ => panic!("cu_source: 미지 자산 {name}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// .cu의 `extern "C" __global__ [__launch_bounds__(..)]` 심볼 스캔.
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

    /// [R24] 매니페스트 심볼 ↔ .cu 정의 양방향 일치 — 누락/오타/미로드 검출.
    #[test]
    fn asset_manifest_matches_cu_symbols() {
        for a in ASSETS {
            // [2026-10-18] 마린 = C++ 템플릿 인스턴스(맹글명) — extern "C" 대조
            // 대상 아님. 형태(맹글 접두·개수)만 확인한다.
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

    /// 경로 표기 규약 — 레포 루트 상대 또는 src/ 상대.
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
