//! qsa — 상주 KV/인덱서 풀 워터마크 프로토콜(hip ↔ vk 공용).
//!
//! 풀은 (full_idx, seq) 키로 위치 p의 k/v/ik를 누적 적립한다. 워터마크 w는
//! "이 키에 적립된 다음 위치"로, 유일한 갱신 규칙:
//!
//! - `pos0 == w`: 순차 적립(정상 경로).
//! - `pos0 <  w`: **접두어 되감기** — 위치 p의 값은 (토큰 접두어, p)의 결정
//!   함수라 접두어가 동일한 재적립(벤치 워밍업 재시작, 스펙 부분수용 롤백,
//!   슬롯 재프리필)에서 기존 값과 새 값이 같다. 재구축도 순서대로 진행해
//!   과거 청크가 이번 재구축분을 덮는다.
//! - `pos0 >  w`: **구멍**(값 경로 청크 등이 풀을 우회) — 읽을 수 없으므로
//!   호출부가 업로드/재구축 경로로 폴백해야 한다.
//!
//! 이 규칙은 hip `qsa_kv_dev_impl`/`qsa_idx_append`(rawhip/q4acc/qsa.rs)과
//! vk `qsa_kv_dev`/`qsa_idx_append`(rawvk/vkacc/qsa.rs)이 공유한다.

/// 워터마크 갱신 — 구멍이면 Err(호출부 폴백), 아니면 w = pos0 + t.
pub fn wm_advance(w: &mut usize, pos0: usize, t: usize) -> Result<(), String> {
    if pos0 > *w {
        return Err(format!("qsa 풀 워터마크 구멍 w={} pos0={pos0} — 업로드 경로로 폴백", w));
    }
    *w = pos0 + t;
    Ok(())
}
