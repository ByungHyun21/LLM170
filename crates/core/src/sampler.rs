//! 샘플러 — temperature/top_k/top_p/min_p/repeat_penalty/seed (plans/83 B).
//!
//! 기본값은 전부 off → `is_greedy()`가 참이면 `sample()`은 CPU greedy와
//! 동일 의미(argmax, 동률 최저 인덱스)로 폴백한다 — 기존 게이트 무변화.
//! RNG는 자작 splitmix64 (외부 rand 크레이트 금지 규칙).
//!
//! 적용 순서 (llama.cpp chain과 동일 계열):
//!   repeat_penalty → temperature → top_k → top_p → min_p → softmax → 난수 추출

/// 난수 상태 — seed 고정시 생성 스트림 완전 재현.
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

    /// [0, 1) 균등 — 상위 53비트 사용 (f64 가수 전체).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// 요청 단위 샘플링 파라미터. 기본값 = 완전 greedy.
#[derive(Clone, Debug)]
pub struct SamplerParams {
    pub temperature: f32,
    /// 0 = 비활성
    pub top_k: usize,
    /// 1.0 = 비활성
    pub top_p: f32,
    /// 0 = 비활성
    pub min_p: f32,
    /// 1.0 = 비활성
    pub repeat_penalty: f32,
    /// repeat_penalty 적용 대상 직전 토큰 수
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
    /// 전 옵션 비활성 판정 — 참이면 greedy 경로 (GPU argmax 유지).
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
            && self.top_k == 0
            && self.top_p >= 1.0
            && self.min_p <= 0.0
            && self.repeat_penalty == 1.0
    }
}

/// 슬롯별 샘플러 상태 — RNG + 반복 패널티 히스토리.
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

    /// 생성 토큰 기록 — 패널티 히스토리 갱신 (프롬프트 토큰도 마지막 n개 반영).
    pub fn push_tokens(&mut self, toks: impl IntoIterator<Item = u32>) {
        let cap = self.params.repeat_last_n.max(1);
        for t in toks {
            self.recent.push_back(t);
            while self.recent.len() > cap {
                self.recent.pop_front();
            }
        }
    }

    /// logits → 토큰. greedy시 argmax (동률 최저 인덱스 — `greedy_from` 동일 의미).
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        if self.is_greedy() {
            return crate::matmul::greedy_from(logits);
        }
        let mut l = logits.to_vec();
        // ① repeat penalty — 히스토리 토큰에 양수 나눗셈/음수 곱셈 (llama.cpp 계열)
        if self.params.repeat_penalty != 1.0 && !self.recent.is_empty() {
            let p = self.params.repeat_penalty;
            for &t in &self.recent {
                if let Some(v) = l.get_mut(t as usize) {
                    if *v > 0.0 {
                        *v /= p;
                    } else {
                        *v *= p;
                    }
                }
            }
        }
        // ② temperature
        if self.params.temperature > 0.0 {
            let t = self.params.temperature;
            for v in &mut l {
                *v /= t;
            }
        }
        // 후보 = (인덱스, 로짓) — 정렬용
        let mut cand: Vec<(u32, f32)> = l
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .map(|(i, &v)| (i as u32, v))
            .collect();
        // ③ top_k — 상위 k개
        if self.params.top_k > 0 && cand.len() > self.params.top_k {
            cand.select_nth_unstable_by(self.params.top_k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            cand.truncate(self.params.top_k);
        }
        // 안정 softmax 순서를 위해 내림차순 정렬 (top_p/min_p 누적에 필요)
        cand.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let max = cand.first().map(|&(_, v)| v).unwrap_or(0.0);
        let mut probs: Vec<f64> = cand.iter().map(|&(_, v)| ((v - max) as f64).exp()).collect();
        let sum: f64 = probs.iter().sum();
        if sum <= 0.0 || !sum.is_finite() {
            return cand.first().map(|&(i, _)| i).unwrap_or(0);
        }
        for p in &mut probs {
            *p /= sum;
        }
        // ④ top_p (nucleus) — 누적 ≥ p인 최소 접두
        if self.params.top_p < 1.0 {
            let mut cum = 0.0;
            let mut keep = probs.len();
            for (i, &p) in probs.iter().enumerate() {
                cum += p;
                if cum >= self.params.top_p as f64 {
                    keep = i + 1;
                    break;
                }
            }
            probs.truncate(keep);
            cand.truncate(keep);
            let s: f64 = probs.iter().sum();
            for p in &mut probs {
                *p /= s;
            }
        }
        // ⑤ min_p — 최대확률 대비 비율 미달 제거
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
        // ⑥ 추출
        let r = self.rng.next_f64();
        let mut acc = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            acc += p;
            if r < acc {
                return cand[i].0;
            }
        }
        cand.last().map(|&(i, _)| i).unwrap_or(0)
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
        assert_eq!(s.sample(&logits(&[1.0, 5.0, 3.0])), 1);
        // 동률 최저 인덱스
        assert_eq!(s.sample(&logits(&[7.0, 5.0, 7.0])), 0);
    }

    #[test]
    fn seed_reproducibility() {
        let p = SamplerParams { temperature: 1.5, top_p: 0.9, seed: 42, ..Default::default() };
        let l = logits(&[1.0, 2.0, 0.5, 1.7, 0.9]);
        let mut a = Sampler::new(p.clone());
        let mut b = Sampler::new(p);
        let sa: Vec<u32> = (0..32).map(|_| a.sample(&l)).collect();
        let sb: Vec<u32> = (0..32).map(|_| b.sample(&l)).collect();
        assert_eq!(sa, sb);
        // 다른 시드는 (실질적으로 확률 1로) 다른 스트림
        let mut c = Sampler::new(SamplerParams { temperature: 1.5, seed: 7, ..Default::default() });
        let sc: Vec<u32> = (0..32).map(|_| c.sample(&l)).collect();
        assert_ne!(sa, sc);
    }

    #[test]
    fn temp_zero_converges_argmax() {
        let mut s = Sampler::new(SamplerParams { temperature: 0.0, seed: 1, ..Default::default() });
        // 온도 0 = greedy — top_p 등 나머지 기본 off
        assert_eq!(s.sample(&logits(&[0.1, 9.0, 2.0])), 1);
    }

    #[test]
    fn top_k_one_is_argmax() {
        let mut s = Sampler::new(SamplerParams { temperature: 1.0, top_k: 1, seed: 3, ..Default::default() });
        assert_eq!(s.sample(&logits(&[0.1, 9.0, 2.0])), 1);
    }

    #[test]
    fn repeat_penalty_suppresses_recent() {
        let p = SamplerParams { repeat_penalty: 1000.0, temperature: 0.0, ..Default::default() };
        // 온도 0이라 greedy지만 패널티가 1번을 억누르면 2번이 승
        let mut s = Sampler::new(p);
        s.push_tokens([1u32]);
        let l = logits(&[0.0, 9.0, 8.0]);
        // 9.0 / 1000 = 0.009 < 8.0
        assert_eq!(s.sample(&l), 2);
    }

    #[test]
    fn top_p_truncates_tail() {
        // 압도적 1등 — top_p 0.5면 사실상 1번만 생존
        let mut s = Sampler::new(SamplerParams { temperature: 1.0, top_p: 0.5, seed: 5, ..Default::default() });
        let l = logits(&[0.0, 20.0, 0.0]);
        assert_eq!(s.sample(&l), 1);
    }
}
