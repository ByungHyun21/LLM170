
#[derive(Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

#[derive(Clone, Debug)]
pub struct SamplerParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
}

impl Default for SamplerParams {
    fn default() -> Self {
        SamplerParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            seed: 0,
        }
    }
}

impl SamplerParams {
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
            && self.top_k == 0
            && self.top_p >= 1.0
            && self.min_p <= 0.0
            && (self.repeat_penalty == 1.0 || self.repeat_last_n == 0)
    }
}

pub struct Sampler {
    pub params: SamplerParams,
    rng: Rng,
    recent: std::collections::VecDeque<u32>,
}

impl Sampler {
    pub fn new(params: SamplerParams) -> Self {
        let rng = Rng::new(params.seed);
        Sampler {
            params,
            rng,
            recent: Default::default(),
        }
    }

    pub fn is_greedy(&self) -> bool {
        self.params.is_greedy()
    }

    pub fn push_tokens(&mut self, toks: impl IntoIterator<Item = u32>) {
        let cap = self.params.repeat_last_n;
        if cap == 0 {
            self.recent.clear();
            return;
        }
        for t in toks {
            self.recent.push_back(t);
            while self.recent.len() > cap {
                self.recent.pop_front();
            }
        }
    }

    pub fn apply_penalties(&self, logits: &mut [f32]) {
        if self.params.repeat_penalty != 1.0 && !self.recent.is_empty() {
            let p = self.params.repeat_penalty;
            let mut uniq: Vec<u32> = Vec::with_capacity(self.recent.len());
            for &t in &self.recent {
                if !uniq.contains(&t) {
                    uniq.push(t);
                }
            }
            for t in uniq {
                if let Some(v) = logits.get_mut(t as usize) {
                    if *v > 0.0 {
                        *v /= p;
                    } else {
                        *v *= p;
                    }
                }
            }
        }
    }

    pub fn apply_penalties_ext(&self, logits: &mut [f32], extra: &[u32]) {
        if self.params.repeat_penalty == 1.0 || (self.recent.is_empty() && extra.is_empty()) {
            return;
        }
        let p = self.params.repeat_penalty;
        let mut uniq: Vec<u32> = Vec::with_capacity(self.recent.len() + extra.len());
        for &t in self.recent.iter().chain(extra.iter()) {
            if !uniq.contains(&t) {
                uniq.push(t);
            }
        }
        for t in uniq {
            if let Some(v) = logits.get_mut(t as usize) {
                if *v > 0.0 {
                    *v /= p;
                } else {
                    *v *= p;
                }
            }
        }
    }

    pub fn next_coin(&mut self) -> f64 {
        self.rng.next_f64()
    }

    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        if self.is_greedy() {
            return crate::matmul::greedy_from(logits);
        }
        self.apply_penalties(logits);
        if self.params.temperature <= 0.0 {
            return crate::matmul::greedy_from(logits);
        }
        if let Some(tok) = self.sample_core(logits, true) {
            return tok;
        }
        self.sample_core(logits, false).unwrap_or(0)
    }

    pub fn prob_of(&self, logits: &[f32], x: u32) -> f64 {
        const CUT_D: f32 = 20.0;
        const CAND_N: usize = 4096;
        let cmp_desc = |a: &(u32, f32), b: &(u32, f32)| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        };
        let mut n_below = 0usize;
        let mut mx = f32::NEG_INFINITY;
        for &v in logits {
            if v > mx {
                mx = v;
            }
        }
        let cutv = mx - CUT_D;
        let mut cand: Vec<(u32, f32)> = Vec::with_capacity(1024);
        for (i, &v) in logits.iter().enumerate() {
            if !v.is_finite() {
                continue;
            }
            if v > cutv {
                cand.push((i as u32, v));
            } else {
                n_below += 1;
            }
        }
        if cand.is_empty() {
            return 0.0;
        }
        if self.params.top_k > 0 && cand.len() > self.params.top_k {
            cand.select_nth_unstable_by(self.params.top_k - 1, cmp_desc);
            cand.truncate(self.params.top_k);
        }
        if cand.len() > CAND_N {
            cand.select_nth_unstable_by(CAND_N - 1, cmp_desc);
            cand.truncate(CAND_N);
        }
        cand.sort_unstable_by(cmp_desc);
        let t = self.params.temperature;
        let sc = |v: f32| if t > 0.0 && t != 1.0 { v / t } else { v };
        let max = sc(cand[0].1);
        let mut probs: Vec<f64> = cand
            .iter()
            .map(|&(_, v)| ((sc(v) - max) as f64).exp())
            .collect();
        let mut sum: f64 = probs.iter().sum();
        if n_below > 0 {
            sum += (n_below as f64) * (-(CUT_D as f64)).exp();
        }
        if sum <= 0.0 || !sum.is_finite() {
            return 0.0;
        }
        for p in &mut probs {
            *p /= sum;
        }
        if self.params.top_p < 1.0 {
            let p_thr = self.params.top_p as f64;
            let mut cum = 0.0;
            let mut keep = probs.len();
            for (i, &p) in probs.iter().enumerate() {
                cum += p;
                if cum >= p_thr {
                    keep = i + 1;
                    break;
                }
            }
            probs.truncate(keep);
            cand.truncate(keep);
            let s2: f64 = probs.iter().sum();
            if s2 <= 0.0 {
                return 0.0;
            }
            for p in &mut probs {
                *p /= s2;
            }
        }
        if self.params.min_p > 0.0 {
            let mxp = probs.first().copied().unwrap_or(0.0);
            let thr = mxp as f32 * self.params.min_p;
            let mut keep = probs.len();
            for (i, &p) in probs.iter().enumerate() {
                if (p as f32) < thr {
                    keep = i;
                    break;
                }
            }
            if keep == 0 {
                keep = 1;
            }
            probs.truncate(keep);
            cand.truncate(keep);
            let s3: f64 = probs.iter().sum();
            if s3 <= 0.0 {
                return 0.0;
            }
            for p in &mut probs {
                *p /= s3;
            }
        }
        for (i, &(tok, _)) in cand.iter().enumerate() {
            if tok == x {
                return probs[i];
            }
        }
        0.0
    }

    fn sample_core(&mut self, l: &[f32], cut: bool) -> Option<u32> {
        const CUT_D: f32 = 20.0;
        let cmp_desc = |a: &(u32, f32), b: &(u32, f32)| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        };
        let mut n_below = 0usize;
        let mut cand: Vec<(u32, f32)> = if cut {
            let mut mx = f32::NEG_INFINITY;
            for &v in l {
                if v > mx {
                    mx = v;
                }
            }
            let cutv = mx - CUT_D;
            let mut c = Vec::with_capacity(1024);
            for (i, &v) in l.iter().enumerate() {
                if !v.is_finite() {
                    continue;
                }
                if v > cutv {
                    c.push((i as u32, v));
                } else {
                    n_below += 1;
                }
            }
            c
        } else {
            l.iter()
                .enumerate()
                .filter(|(_, v)| v.is_finite())
                .map(|(i, &v)| (i as u32, v))
                .collect()
        };
        if cand.is_empty() {
            return None;
        }
        if self.params.top_k > 0 && cand.len() > self.params.top_k {
            cand.select_nth_unstable_by(self.params.top_k - 1, cmp_desc);
            cand.truncate(self.params.top_k);
        }
        const CAND_N: usize = 4096;
        let tail_from = if cand.len() > CAND_N {
            cand.select_nth_unstable_by(CAND_N - 1, cmp_desc);
            CAND_N
        } else {
            cand.len()
        };
        cand[..tail_from].sort_unstable_by(cmp_desc);
        let t = self.params.temperature;
        let sc = |v: f32| if t > 0.0 && t != 1.0 { v / t } else { v };
        let max = sc(cand[0].1);
        let mut probs: Vec<f64> = cand[..tail_from]
            .iter()
            .map(|&(_, v)| ((sc(v) - max) as f64).exp())
            .collect();
        let mut sum: f64 = probs.iter().sum();
        for &(_, v) in &cand[tail_from..] {
            sum += ((sc(v) - max) as f64).exp();
        }
        if n_below > 0 {
            sum += (n_below as f64) * (-(CUT_D as f64)).exp();
        }
        if sum <= 0.0 || !sum.is_finite() {
            return Some(cand[0].0);
        }
        for p in &mut probs {
            *p /= sum;
        }
        if self.params.top_p < 1.0 {
            let p_thr = self.params.top_p as f64;
            let mut cum = 0.0;
            let mut keep = probs.len();
            let mut closed = false;
            for (i, &p) in probs.iter().enumerate() {
                cum += p;
                if cum >= p_thr {
                    keep = i + 1;
                    closed = true;
                    break;
                }
            }
            if !closed {
                if cut {
                    return None;
                }
                if tail_from < cand.len() {
                    cand[tail_from..].sort_unstable_by(cmp_desc);
                    for &(_, v) in &cand[tail_from..] {
                        probs.push(((sc(v) - max) as f64).exp() / sum);
                    }
                    let mut ext = None;
                    for (off, &pp) in probs[keep..].iter().enumerate() {
                        cum += pp;
                        if cum >= p_thr {
                            ext = Some(keep + off + 1);
                            break;
                        }
                    }
                    if let Some(k) = ext {
                        keep = k;
                    }
                }
            }
            probs.truncate(keep);
            cand.truncate(keep);
            let s: f64 = probs.iter().sum();
            for p in &mut probs {
                *p /= s;
            }
        }
        if self.params.min_p > 0.0 {
            let mx = probs.first().copied().unwrap_or(0.0);
            let thr = mx as f32 * self.params.min_p;
            let mut keep = probs.len();
            for (i, &p) in probs.iter().enumerate() {
                if (p as f32) < thr {
                    keep = i;
                    break;
                }
            }
            if keep == 0 {
                keep = 1;
            }
            probs.truncate(keep);
            cand.truncate(keep);
            let s: f64 = probs.iter().sum();
            for p in &mut probs {
                *p /= s;
            }
        }
        let r = self.rng.next_f64();
        let mut acc = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            acc += p;
            if r < acc {
                return Some(cand[i].0);
            }
        }
        Some(cand.last().map(|&(i, _)| i).unwrap_or(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits(v: &[f32]) -> Vec<f32> {
        let mut l = vec![-10f32; 100];
        for (i, &x) in v.iter().enumerate() {
            l[i] = x;
        }
        l
    }

    #[test]
    fn greedy_default_is_argmax() {
        let mut s = Sampler::new(SamplerParams::default());
        assert!(s.is_greedy());
        assert_eq!(s.sample(&mut logits(&[1.0, 5.0, 3.0])), 1);
        assert_eq!(s.sample(&mut logits(&[7.0, 5.0, 7.0])), 0);
    }

    #[test]
    fn seed_reproducibility() {
        let p = SamplerParams {
            temperature: 1.5,
            top_p: 0.9,
            seed: 42,
            ..Default::default()
        };
        let mut l = logits(&[1.0, 2.0, 0.5, 1.7, 0.9]);
        let mut a = Sampler::new(p.clone());
        let mut b = Sampler::new(p);
        let sa: Vec<u32> = (0..32).map(|_| a.sample(&mut l)).collect();
        let sb: Vec<u32> = (0..32).map(|_| b.sample(&mut l)).collect();
        assert_eq!(sa, sb);
        let mut c = Sampler::new(SamplerParams {
            temperature: 1.5,
            seed: 7,
            ..Default::default()
        });
        let sc: Vec<u32> = (0..32).map(|_| c.sample(&mut l)).collect();
        assert_ne!(sa, sc);
    }

    #[test]
    fn temp_zero_converges_argmax() {
        let mut s = Sampler::new(SamplerParams {
            temperature: 0.0,
            seed: 1,
            ..Default::default()
        });
        assert_eq!(s.sample(&mut logits(&[0.1, 9.0, 2.0])), 1);
    }

    fn sample_reference_full(p: &SamplerParams, rng: &mut Rng, logits: &[f32]) -> u32 {
        let l = logits.to_vec();
        if p.temperature <= 0.0 {
            return crate::matmul::greedy_from(&l);
        }
        let mut cand: Vec<(u32, f32)> = l
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .map(|(i, &v)| (i as u32, v))
            .collect();
        if p.top_k > 0 && cand.len() > p.top_k {
            cand.select_nth_unstable_by(p.top_k - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            cand.truncate(p.top_k);
        }
        cand.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if p.temperature != 1.0 {
            let t = p.temperature;
            for (_, v) in &mut cand {
                *v /= t;
            }
        }
        let max = cand.first().map(|&(_, v)| v).unwrap_or(0.0);
        let mut probs: Vec<f64> = cand
            .iter()
            .map(|&(_, v)| ((v - max) as f64).exp())
            .collect();
        let sum: f64 = probs.iter().sum();
        if sum <= 0.0 || !sum.is_finite() {
            return cand.first().map(|&(i, _)| i).unwrap_or(0);
        }
        for q in &mut probs {
            *q /= sum;
        }
        if p.top_p < 1.0 {
            let mut cum = 0.0;
            let mut keep = probs.len();
            for (i, &q) in probs.iter().enumerate() {
                cum += q;
                if cum >= p.top_p as f64 {
                    keep = i + 1;
                    break;
                }
            }
            probs.truncate(keep);
            cand.truncate(keep);
            let s: f64 = probs.iter().sum();
            for q in &mut probs {
                *q /= s;
            }
        }
        if p.min_p > 0.0 {
            let mx = probs.first().copied().unwrap_or(0.0);
            let thr = mx as f32 * p.min_p;
            let mut keep = probs.len();
            for (i, &q) in probs.iter().enumerate() {
                if (q as f32) < thr {
                    keep = i;
                    break;
                }
            }
            if keep == 0 {
                keep = 1;
            }
            probs.truncate(keep);
            cand.truncate(keep);
            let s: f64 = probs.iter().sum();
            for q in &mut probs {
                *q /= s;
            }
        }
        let r = rng.next_f64();
        let mut acc = 0.0;
        for (i, &q) in probs.iter().enumerate() {
            acc += q;
            if r < acc {
                return cand[i].0;
            }
        }
        cand.last().map(|&(i, _)| i).unwrap_or(0)
    }

    #[test]
    fn partial_select_matches_full_sort() {
        let mut l: Vec<f32> = (0..8192)
            .map(|i| 9.0 - (i as f32) * 0.0008 - ((i % 11) as f32) * 0.00003)
            .collect();
        l[3] = f32::NAN;
        l[4000] = f32::NEG_INFINITY;
        l[500] = l[501];
        for &(tp, tk, mp) in &[
            (0.9f32, 0usize, 0.0f32),
            (0.99, 0, 0.0),
            (1.0, 40, 0.0),
            (0.95, 0, 0.02),
        ] {
            let p = SamplerParams {
                temperature: 0.8,
                top_p: tp,
                top_k: tk,
                min_p: mp,
                seed: 0x5EED,
                ..Default::default()
            };
            let mut a = Sampler::new(p.clone());
            let mut rng = Rng::new(p.seed);
            for _ in 0..16 {
                let x = a.sample(&mut l);
                let y = sample_reference_full(&p, &mut rng, &l);
                assert_eq!(x, y, "tp={tp} tk={tk} mp={mp}");
            }
        }
    }

    #[test]
    fn top_k_one_is_argmax() {
        let mut s = Sampler::new(SamplerParams {
            temperature: 1.0,
            top_k: 1,
            seed: 3,
            ..Default::default()
        });
        assert_eq!(s.sample(&mut logits(&[0.1, 9.0, 2.0])), 1);
    }

    #[test]
    fn repeat_penalty_suppresses_recent() {
        let p = SamplerParams {
            repeat_penalty: 1000.0,
            temperature: 0.0,
            ..Default::default()
        };
        let mut s = Sampler::new(p);
        s.push_tokens([1u32]);
        let mut l = logits(&[0.0, 9.0, 8.0]);
        assert_eq!(s.sample(&mut l), 2);
    }

    #[test]
    fn repeat_penalty_unique_once() {
        let p = SamplerParams {
            repeat_penalty: 2.0,
            temperature: 0.0,
            ..Default::default()
        };
        let mut s = Sampler::new(p);
        s.push_tokens([1u32, 1u32, 1u32]);
        let mut l = logits(&[0.0, 9.0, 4.4]);
        assert_eq!(s.sample(&mut l), 1);
    }

    #[test]
    fn repeat_last_n_zero_disables() {
        let p = SamplerParams {
            repeat_penalty: 1000.0,
            repeat_last_n: 0,
            temperature: 0.0,
            ..Default::default()
        };
        assert!(p.is_greedy());
        let mut s = Sampler::new(p);
        s.push_tokens([1u32]);
        assert_eq!(s.sample(&mut logits(&[0.0, 9.0, 8.0])), 1);
    }

    #[test]
    fn top_p_truncates_tail() {
        let mut s = Sampler::new(SamplerParams {
            temperature: 1.0,
            top_p: 0.5,
            seed: 5,
            ..Default::default()
        });
        let mut l = logits(&[0.0, 20.0, 0.0]);
        assert_eq!(s.sample(&mut l), 1);
    }

    #[test]
    fn min_p_filters_tail_deterministically() {
        for seed in [0u64, 1, 7] {
            let mut s = Sampler::new(SamplerParams {
                temperature: 1.0,
                min_p: 0.5,
                seed,
                ..Default::default()
            });
            assert!(!s.is_greedy());
            for _ in 0..16 {
                assert_eq!(s.sample(&mut logits(&[10.0, 0.0])), 0);
            }
        }
        let mut s = Sampler::new(SamplerParams {
            temperature: 1.0,
            min_p: 0.95,
            seed: 3,
            ..Default::default()
        });
        for _ in 0..16 {
            assert_eq!(s.sample(&mut logits(&[1.0, 0.9])), 0);
        }
    }

    #[test]
    fn temperature_distribution_contract() {
        let base = [2.0f32, 1.2, 0.4, -0.3, -1.1, -1.9, -2.7, -3.5];
        for t in [0.8f32, 1.25] {
            let mut s = Sampler::new(SamplerParams {
                temperature: t,
                seed: 0x5EED_C0DE,
                ..Default::default()
            });
            let n = 200_000usize;
            let mut counts = [0u64; 8];
            for _ in 0..n {
                let mut lg = base;
                counts[s.sample(&mut lg) as usize] += 1;
            }
            let mx = base.iter().cloned().fold(f32::MIN, f32::max);
            let zsum: f64 = base.iter().map(|&v| f64::from(((v - mx) / t).exp())).sum();
            let mut chi2 = 0.0f64;
            for (i, &v) in base.iter().enumerate() {
                let e = f64::from(((v - mx) / t).exp()) / zsum * n as f64;
                chi2 += (counts[i] as f64 - e).powi(2) / e;
            }
            let df = base.len() - 1;
            let z = (chi2 / df as f64 - 1.0) * (df as f64 / 2.0).sqrt();
            assert!(
                z.abs() < 4.0,
                "t={t}: chi2={chi2:.1} df={df} z={z:+.2} counts={counts:?}"
            );
        }
    }

    #[test]
    fn prob_of_matches_sampling_frequency() {
        let l = logits(&[3.0, 2.0, 1.0, 0.5]);
        let mut s = Sampler::new(SamplerParams {
            temperature: 1.0,
            seed: 7,
            ..Default::default()
        });
        let p0 = s.prob_of(&l, 0);
        let p1 = s.prob_of(&l, 1);
        let (mut c0, mut c1, n) = (0usize, 0usize, 4000usize);
        for _ in 0..n {
            let mut ll = l.clone();
            match s.sample(&mut ll) {
                0 => c0 += 1,
                1 => c1 += 1,
                _ => {}
            }
        }
        let (f0, f1) = (c0 as f64 / n as f64, c1 as f64 / n as f64);
        assert!((f0 - p0).abs() < 0.03, "p0 {p0} vs {f0}");
        assert!((f1 - p1).abs() < 0.03, "p1 {p1} vs {f1}");
    }

    #[test]
    fn prob_of_topk_exclusion_is_zero() {
        let l = logits(&[3.0, 2.0, 1.0]);
        let s = Sampler::new(SamplerParams {
            top_k: 1,
            ..Default::default()
        });
        assert!(s.prob_of(&l, 0) > 0.99);
        assert_eq!(s.prob_of(&l, 1), 0.0);
    }
}
