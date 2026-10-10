//! [A-1] 스페큘러티브 초안 — n-gram 단순형(llama.cpp ngram-simple 미러).
//!
//! 히스토리(프롬프트+생성)에서 마지막 n토큰 패턴의 **직전 출현**을 역방향
//! 선형 스캔으로 찾아, 그 뒤 m토큰을 초안으로 복사한다. copy_max가 n 미만
//! 이면 포기(무상태·추가 메모리 0 — common/ngram-map.cpp:49-112 동형).
//! 초안은 검증(배치 argmax 비교)으로만 수용되므로 정확성 위험은 없다.

/// n-gram 초안 — 반환 길이 ≤ max_draft(호출부가 배치 상한으로 절단).
pub fn ngram_draft(hist: &[u32], n: usize, m: usize, max_draft: usize) -> Vec<u32> {
    if n == 0 || hist.len() < n + 1 || max_draft == 0 {
        return Vec::new();
    }
    let pat = &hist[hist.len() - n..];
    // 역방향 선형 스캔 — 첫 매치에서 중단(최근 출현 우선).
    let mut match_pos = None;
    let mut i = hist.len() - n - 1;
    loop {
        if hist[i..i + n] == *pat {
            match_pos = Some(i);
            break;
        }
        if i == 0 {
            break;
        }
        i -= 1;
    }
    let Some(mp) = match_pos else {
        return Vec::new();
    };
    let after = mp + n;
    let copy_max = (hist.len() - after).min(m);
    if copy_max < n {
        return Vec::new();
    }
    hist[after..after + copy_max.min(max_draft)].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draft_copies_after_last_match() {
        // "a b c d a b" → 마지막 "a b"의 직전 출현 뒤 = "c d a b"…
        let h = [1u32, 2, 3, 4, 1, 2];
        let d = ngram_draft(&h, 2, 8, 8);
        assert_eq!(d, vec![3, 4, 1, 2]);
    }

    #[test]
    fn draft_rejects_short_tail() {
        // 매치 뒤 남은 토큰(copy_max) < n이면 포기 — [1,2] 매치가 꼬리에 붙은 경우.
        let h = [1u32, 2, 3, 1, 2];
        // 패턴 [1,2]의 직전 출현(0) 뒤 = 3토큰 ≥ n=2 → 초안 [3,1,2].
        assert_eq!(ngram_draft(&h, 2, 8, 8), vec![3, 1, 2]);
        // n=3: 패턴 [3,1,2]의 직전 출현 없음 → 빈 초안.
        assert!(ngram_draft(&h, 3, 8, 8).is_empty());
    }

    #[test]
    fn draft_no_match() {
        let h = [1u32, 2, 3, 4, 5, 6];
        assert!(ngram_draft(&h, 3, 8, 8).is_empty());
    }

    #[test]
    fn draft_respects_max() {
        let h: Vec<u32> = (0..40).chain(0..40).collect();
        // 패턴 [36,37,38,39]의 직전 출현 = 36 → 뒤는 40.. = [0,1,2,…].
        let d = ngram_draft(&h, 4, 48, 3);
        assert_eq!(d, vec![0, 1, 2]);
    }
}
