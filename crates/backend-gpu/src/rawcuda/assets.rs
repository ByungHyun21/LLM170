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
            "w4a16_gemv_multi",
            "w4a16_copy",
            "w4a16_moe_topk256",
            "w4a16_gemv_experts_glu",
            "w4a16_gemm_g128",
            "w4a16_gemv_g128",
            "w4a16_gemv_g128_t",
            "w4a16_gemm_g32_bf16",
            "w4a16_gemv_g32_bf16",
            "w4a16_gemv_bf16_t",
            "w4a16_cast_x32",
            "w4a16_cast_bf16",
            "w4a16_axpy",
            "w4a16_shared_add",
            "w4a16_gemv_experts_g32_bf16",
            "w4a16_gemm_g32_mma_grp",
            "w4a16_moe_align",
            "w4a16_moe_accum",
            "w4a16_moe_topk_t",
            "w4a16_gemv_bf16",
            "w4a16_gemm_bf16",
            "w4a16_gemm_bf16_t",
            "w4a16_gemm_bf16_mma",
            "w4a16_gemm_g128_mma",
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
        syms: &["norm_resid"],
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
            "gdn_scan",
            "gdn_spec_scan",
            "gdn_scan_akq",
            "gdn1_part",
            "gdn1_comb",
            "gdn1_upd",
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
        syms: &["attn_fwd3s_part_q", "attn_prep_q", "attn_fwd3s_merge"],
        boot: true,
    },
    Asset {
        name: "head",
        env: "LLM170_CUDA_HEAD_FATBIN_PATH",
        paths: &[
            "crates/backend-gpu/src/rawcuda/assets/head.fatbin",
            "src/rawcuda/assets/head.fatbin",
        ],
        syms: &["w4a16_argmax_min", "w4a16_argmax_min_t"],
        boot: true,
    },
    Asset {
        name: "ew",
        env: "LLM170_CUDA_EW_FATBIN_PATH",
        paths: &[
            "crates/backend-gpu/src/rawcuda/assets/ew.fatbin",
            "src/rawcuda/assets/ew.fatbin",
        ],
        syms: &["ew"],
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
        "norm" => include_str!("assets/norm.cu"),
        "gdn" => include_str!("assets/gdn.cu"),
        "attn" => include_str!("assets/attn.cu"),
        "head" => include_str!("assets/head.cu"),
        "ew" => include_str!("assets/ew.cu"),
        "smoke" => include_str!("assets/smoke.cu"),
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
