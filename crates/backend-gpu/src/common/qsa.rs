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
        return Err(format!(
            "qsa 풀 워터마크 구멍 w={} pos0={pos0} — 업로드 경로로 폴백",
            w
        ));
    }
    *w = pos0 + t;
    Ok(())
}

/// 선택 목록 산술 — (n_sel, list_len). `n_past = pos0 + t`, `n_blocks = n_past / r`.
///
/// 양쪽 비교(5줄 바이트 동일 — `qsa_sel_dev` 디코드 선택):
/// - hip(rawhip/q4acc/qsa.rs):
///   `let tail_start = n_blocks * r; let tail_cnt = n_past - tail_start;`
///   `let width = n_past.min(idx_top_k + r - 1);`
///   `let n_sel = ((width - tail_cnt) / r).min(n_blocks);`
///   `let list_len = n_sel * r + tail_cnt;`
/// - vk(rawvk/vkacc/qsa.rs): 위와 동일 코드(주석 "n_sel 산술은
///   stages::qsa_select 패스 B와 동일" 까지 대응).
///
/// vk `qsa_sel_dev_mt`(프리필 다중 토큰)의 토큰별 루프도 같은 식을
/// per-token 전개해 list_len 총합을 낸다(hip엔 t>1 디바이스 선택이 없음).
/// 정수 산술이라 무동기 — 호출부는 스크래치 크기·커널 인자로만 소비.
pub fn sel_counts(n_past: usize, n_blocks: usize, r: usize, idx_top_k: usize) -> (usize, usize) {
    let tail_start = n_blocks * r;
    let tail_cnt = n_past - tail_start;
    let width = n_past.min(idx_top_k + r - 1);
    let n_sel = ((width - tail_cnt) / r).min(n_blocks);
    (n_sel, n_sel * r + tail_cnt)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 기저 사례 + 항등 선택 단축 불변식(n_past ≤ idx_top_k+r-1 → 전체 선택).
    #[test]
    fn sel_counts_cases() {
        // n_past=100, r=16: nb=6, tail=4, width=min(100, 47)=47,
        // n_sel=((47-4)/16).min(6)=2, list_len=2·16+4=36.
        assert_eq!(sel_counts(100, 100 / 16, 16, 32), (2, 36));
        // 항등 선택: 상한 이하에선 목록이 n_past 전체(코어 패스 B 단축 근거).
        for n_past in [1usize, 15, 16, 17, 46, 47] {
            let (ns, ll) = sel_counts(n_past, n_past / 16, 16, 32);
            assert_eq!(ll, n_past, "n_past={n_past}");
            assert_eq!(ns, n_past / 16);
        }
        // 상한 넘으면 목록이 n_past 미만 — top-k 만큼만(48은 tail=0: 2블록=32).
        let (ns, ll) = sel_counts(48, 3, 16, 32);
        assert_eq!(ns, 2);
        assert_eq!(ll, 32);
        assert!(ll < 48);
    }
}
