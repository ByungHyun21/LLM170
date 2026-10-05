//! 폴백 카운터 qwen4exp 진입점 (plans/107 W8) — 원장 구현은 plans/129 A5로
//! diag 공유층(`llm170_diag::fb`)으로 이동했다: EXL3(vk)·vl 등 core 밖
//! 폴백과 같은 이름 공간을 써야 `diag fb`/main 종료 [fb] 출력이 단일 원장을
//! 보이기 때문. 이 파일은 기존 호출부(forward.rs·layers.rs 등 17지점)의
//! 타입화 API(Id → 이름 문자열)만 유지한다.

use llm170_diag::fb as dfb;

#[repr(usize)]
pub enum Id {
    EmbQ8g,
    PleGgpu,
    QsaDevsel,
    QsaDevselMt,
    QsaIdxpool,
    QsaAttn,
    FrameCreate,
    FrameCreateNp,
    MtpDraft,
    MtpSpec,
}

impl Id {
    fn name(self) -> &'static str {
        match self {
            Id::EmbQ8g => "emb-q8g",
            Id::PleGgpu => "ple-ggpu",
            Id::QsaDevsel => "qsa-devsel",
            Id::QsaDevselMt => "qsa-devsel-mt",
            Id::QsaIdxpool => "qsa-idxpool",
            Id::QsaAttn => "qsa-attn",
            Id::FrameCreate => "frame-create",
            Id::FrameCreateNp => "frame-create-np",
            Id::MtpDraft => "mtp-draft",
            Id::MtpSpec => "mtp-spec",
        }
    }
}

pub fn incr(id: Id) {
    dfb::incr(id.name());
}

/// 전체 원장 보고 — diag 공유 원장(EXL3·vl 카운터 포함), 0이 아닌 것만.
pub fn report() -> String {
    dfb::report()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incr_bumps_named_counter() {
        let before = dfb::count("qsa-attn");
        incr(Id::QsaAttn);
        incr(Id::QsaAttn);
        assert_eq!(dfb::count("qsa-attn"), before + 2);
        assert!(report().contains("qsa-attn"));
    }
}
