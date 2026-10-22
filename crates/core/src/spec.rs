
pub fn ngram_draft(hist: &[u32], n: usize, m: usize, max_draft: usize) -> Vec<u32> {
    if n == 0 || hist.len() < n + 1 || max_draft == 0 {
        return Vec::new();
    }
    let pat = &hist[hist.len() - n..];
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

pub fn ngram_draft_multi(hist: &[u32], m: usize, max_draft: usize) -> Vec<u32> {
    let depths: &[usize] = if hist.len() >= 16 {
        &[8, 6, 4, 3]
    } else if hist.len() >= 8 {
        &[6, 4, 3]
    } else if hist.len() >= 4 {
        &[3]
    } else {
        return Vec::new();
    };
    for &n in depths {
        let d = ngram_draft(hist, n, m, max_draft);
        if !d.is_empty() {
            return d;
        }
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draft_copies_after_last_match() {
        let h = [1u32, 2, 3, 4, 1, 2];
        let d = ngram_draft(&h, 2, 8, 8);
        assert_eq!(d, vec![3, 4, 1, 2]);
    }

    #[test]
    fn draft_rejects_short_tail() {
        let h = [1u32, 2, 3, 1, 2];
        assert_eq!(ngram_draft(&h, 2, 8, 8), vec![3, 1, 2]);
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
        let d = ngram_draft(&h, 4, 48, 3);
        assert_eq!(d, vec![0, 1, 2]);
    }
}
