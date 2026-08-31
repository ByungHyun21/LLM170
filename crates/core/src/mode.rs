//! 실행 모드 (ADR-0002) — 런타임 플래그. 엔진 코어는 모드 무관,
//! 모드는 메모리 예산과 커널 변형 선택만 담당한다.
//!
//! `--mode` 플래그는 server가 파싱해 환경변수 기본값(LLM170_W_CAP_GB·
//! LLM170_Q4_CHUNK)으로 반영한다 — 백엔드·엔진은 기존 env 관례를 그대로
//! 읽으므로 단일 소스 (두 번째 관례 병립 금지).
//!
//! | 모드 | 타깃 | 가중치 상주 기본 | 프리필 청크 |
//! |---|---|---|---|
//! | universal | 임의 CPU/GPU (개발기 8060S 96GiB) | 72GiB | 1024 |
//! | cmp-stock | CMP 170HX 스톡 (40GiB HBM2, 스로틀) | 36GiB | 512 |
//! | cmp-unlocked | CMP 170HX 언락 (40→64GiB) | 36GiB | 1024 |
//!
//! 커널 변형(cmp-stock: half2/BF16 벡터·INT32 경로, tensor core·FMA 금지)은
//! cubecl 커널 세트가 갖춰질 때 이 enum을 분기 키로 사용한다.

/// 실행 모드 — 빌드 feature 아님 (단일 바이너리, 런타임 플래그).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// 범용 (기본) — 임의 GPU/CPU.
    Universal,
    /// CMP 170HX 스톡 BIOS — 스로틀 매트릭스 하위 최적화 변형.
    CmpStock,
    /// CMP 170HX 언락 — 메모리 예산만 상이.
    CmpUnlocked,
}

impl Mode {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "universal" => Some(Mode::Universal),
            "cmp-stock" => Some(Mode::CmpStock),
            "cmp-unlocked" => Some(Mode::CmpUnlocked),
            _ => None,
        }
    }

    /// 가중치 상주 예산 기본 (GiB) — LLM170_W_CAP_GB 미설정 시.
    pub fn w_cap_gb(self) -> usize {
        match self {
            Mode::Universal => 72,
            // CMP HBM2 40GiB — 스톡·언락 공통 (스크래치 여유 확보)
            Mode::CmpStock | Mode::CmpUnlocked => 36,
        }
    }

    /// 프리필 청크 토큰 기본 — LLM170_Q4_CHUNK 미설정 시.
    pub fn prefill_chunk(self) -> usize {
        match self {
            Mode::Universal | Mode::CmpUnlocked => 1024,
            // 스톡 대역폭 스로틀에서 VRAM 성장 여유 확보
            Mode::CmpStock => 512,
        }
    }
}
