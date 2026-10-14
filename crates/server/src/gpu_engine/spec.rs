//! gpu_engine.rs에서 분해(순수 이동 — 코드 무변).
use super::QwenEngine;

impl QwenEngine {
    /// [§7.1 2026-10-11] 스펙 서빙 — WY 표현 버퍼 확보(모델당 1회).
    /// [dsave 대체 2026-10-14] 정밀 상태 스냅샷(dsave)은 프로브 전용 지연
    /// 할당으로 강등 — serve 경로 상주분(27B/4슬롯 315MiB, dring0 5.6MiB
    /// 대체)은 예약·컨텍스트로 환원됐다(RESERVE 2.25GiB→1995MiB).
    /// [2026-10-12] 스펙 활성 여부 — 비활성(스트리밍·VRAM 부족)이면 스케줄러가
    /// 라운드를 시도하지 않는다.
    pub fn spec_on(&self) -> bool {
        self.dec.spec_on()
    }

    pub fn enable_spec(&mut self) -> Result<(), String> {
        self.dec.enable_spec()
    }
    pub fn spec_verify(&mut self, slot: usize, rows: &[f32], t: usize) -> Result<Vec<u32>, String> {
        self.dec.spec_verify(slot, rows, t)
    }
    /// [§7 2단계] 검증 체인 로짓([t][vocab]) — 비greedy 거절샘플링용.
    pub fn spec_verify_logits(
        &mut self,
        slot: usize,
        rows: &[f32],
        t: usize,
    ) -> Result<Vec<f32>, String> {
        let mut lg = self.dec.spec_verify_logits(slot, rows, t)?;
        self.softcap_row(&mut lg);
        Ok(lg)
    }

    pub fn spec_rollback(&mut self, slot: usize, keep: usize) -> Result<(), String> {
        self.dec.spec_rollback(slot, keep)
    }
    pub fn spec_rewind_pos(&mut self, slot: usize, n: u32) -> Result<(), String> {
        self.dec.spec_rewind_pos(slot, n)
    }
}
