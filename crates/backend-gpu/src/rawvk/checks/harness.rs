//! checks 공통 하니스(plans/107 W4) — 체커마다 반복되던
//! `VkAcc::new()` + 모델/텐서 참조 로딩(arch 판별 qwen4exp/qwen35) 보일러플레이트 헬퍼.

use crate::rawvk::vkacc::VkAcc;

/// arch 판별 후 단일 로드한 참조 모델 — 진단 전용 값 semantic(박싱 없이 값 소유).
#[allow(clippy::large_enum_variant)]
pub enum AnyModel {
    Q35(llm170_core::qwen35::Model),
    Q4(llm170_core::qwen4exp::Model4),
}

impl AnyModel {
    /// 텐서 참조 해석 — q35는 `w()`(없으면 "텐서 없음"), q4는 `w4()` 에러 전파.
    pub fn w(&self, tname: &str) -> Result<llm170_core::matmul::Weight<'_>, String> {
        match self {
            AnyModel::Q35(m) => m.w(tname).ok_or_else(|| "텐서 없음".into()),
            AnyModel::Q4(m) => m.w4(tname).map_err(|e| e.to_string()),
        }
    }

    /// qwen4exp 전용 체커용 — 다른 arch면 원래 로드 실패와 동일하게 거부.
    pub fn q4(&self) -> Result<&llm170_core::qwen4exp::Model4, String> {
        match self {
            AnyModel::Q4(m) => Ok(m),
            AnyModel::Q35(_) => Err("qwen4exp 모델이 아님".into()),
        }
    }
}

/// 경로 arch 판별(qwen4exp 멀티파트 폴백) 후 단일 로드.
pub fn load_ref(path: &str) -> Result<AnyModel, String> {
    let is_q4 = llm170_gguf::GgufFile::open(std::path::Path::new(path))
        .ok()
        .and_then(|g| g.arch().map(|a| a == "qwen4exp"))
        .unwrap_or(false);
    if is_q4 {
        Ok(AnyModel::Q4(
            llm170_core::qwen4exp::Model4::load(std::path::Path::new(path))
                .map_err(|e| e.to_string())?,
        ))
    } else {
        Ok(AnyModel::Q35(
            llm170_core::qwen35::Model::load(std::path::Path::new(path))
                .map_err(|e| e.to_string())?,
        ))
    }
}

/// 진단 체커 공통 하니스: `VkAcc` 기동(+ 참조 모델 로딩) 반복 패턴 헬퍼.
pub struct CheckHarness {
    pub acc: VkAcc,
}

impl CheckHarness {
    /// GPU 가속기 기동만 필요한 체커용.
    pub fn new() -> Result<Self, String> {
        Ok(Self { acc: VkAcc::new()? })
    }

    /// GPU 기동 + 참조 모델 로딩 — CPU 대조 체커 표준 진입.
    pub fn with_ref(path: &str) -> Result<(Self, AnyModel), String> {
        Ok((Self::new()?, load_ref(path)?))
    }
}
