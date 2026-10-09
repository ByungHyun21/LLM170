//! CPU matmul — 스레드 수 · 단행/배치 내적 + W4A16 split arm(분리 버퍼 디양자화).

pub fn n_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(32)
}
use super::weight::Weight;
use llm170_diag::profile_span;

pub fn matmul(x: &[f32], w: &Weight, out: &mut [f32]) {
    profile_span!("cpu::matmul1");
    // W4A16 split(§3.5 A안 직접 로드) — 분리 버퍼 디양자화 + f32 내적.
    // f32 레퍼런스(비트 격리 계약은 quant lane 소관).
    if w.ty == crate::wtype::WType::W4a16Split {
        let scale = w
            .aux
            .expect("w4a16 split: aux(scale) 필수 계약 — Model::w 보장");
        let n_in = w.n_in as usize;
        let nt = n_threads().max(1).min(out.len());
        let rows_per = out.len().div_ceil(nt);
        let mut chunks: Vec<&mut [f32]> = out.chunks_mut(rows_per).collect();
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (lo, ch) in chunks.iter_mut().enumerate() {
                let row0 = lo * rows_per;
                handles.push(scope.spawn(move || {
                    let mut scratch = vec![0.0f32; n_in];
                    for (r, o) in ch.iter_mut().enumerate() {
                        dequant_row_w4a16_split(
                            w.data,
                            scale,
                            row0 + r,
                            n_in,
                            w.group,
                            w.scale_bf16,
                            &mut scratch,
                        );
                        let mut acc = 0.0f32;
                        for i in 0..n_in {
                            acc += x[i] * scratch[i];
                        }
                        *o = acc;
                    }
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
        });
        return;
    }
    let n_in = w.n_in as usize;
    let nt = n_threads().max(1).min(out.len());
    let rows_per = out.len().div_ceil(nt);
    let mut chunks: Vec<&mut [f32]> = out.chunks_mut(rows_per).collect();
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for (lo, ch) in chunks.iter_mut().enumerate() {
            let row0 = lo * rows_per;
            handles.push(scope.spawn(move || {
                let mut scratch = vec![0.0f32; n_in];
                for (r, o) in ch.iter_mut().enumerate() {
                    crate::quant::dequant_row(
                        w.ty,
                        w.data,
                        (row0 + r) as u64,
                        w.n_in,
                        &mut scratch,
                    );
                    let mut acc = 0.0f32;
                    for i in 0..n_in {
                        acc += x[i] * scratch[i];
                    }
                    *o = acc;
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    });
}

/// 배치: outs[t][o] = Σ_i xs[t][i]·W[o,i].
/// 행(o)별로 한 번 디양자화해 B 토큰과 내적 — prefill에서 디양자화 비용 상각.
/// 스레드별 로컬 결과 [T][rows_per] → 조인 후 스캐터 (행 슬라이스 교차 차입 회피).
pub fn matmul_batch(xs: &[Vec<f32>], w: &Weight, outs: &mut [Vec<f32>]) {
    // W4A16 split(§3.5 A안) — 행별 1회 디양자화 후 T토큰 내적(일반 경로 미러).
    if w.ty == crate::wtype::WType::W4a16Split {
        let scale = w
            .aux
            .expect("w4a16 split: aux(scale) 필수 계약 — Model::w 보장");
        profile_span!("cpu::matmulB");
        let n_in = w.n_in as usize;
        let n_out = w.n_out as usize;
        let t = xs.len();
        assert_eq!(outs.len(), t);
        let nt = n_threads().max(1).min(n_out);
        let rows_per = n_out.div_ceil(nt);
        let mut locals: Vec<Vec<f32>> = vec![vec![0.0f32; t * rows_per]; nt];
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (g, local) in locals.iter_mut().enumerate() {
                let row0 = g * rows_per;
                handles.push(scope.spawn(move || {
                    let mut scratch = vec![0.0f32; n_in];
                    let rows = n_out.saturating_sub(row0).min(rows_per);
                    for r in 0..rows {
                        dequant_row_w4a16_split(
                            w.data,
                            scale,
                            row0 + r,
                            n_in,
                            w.group,
                            w.scale_bf16,
                            &mut scratch,
                        );
                        for (ti, x) in xs.iter().enumerate() {
                            let mut acc = 0.0f32;
                            for i in 0..n_in {
                                acc += x[i] * scratch[i];
                            }
                            local[ti * rows_per + r] = acc;
                        }
                    }
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
        });
        for (g, local) in locals.iter().enumerate() {
            let row0 = g * rows_per;
            let rows = n_out.saturating_sub(row0).min(rows_per);
            for ti in 0..t {
                for r in 0..rows {
                    outs[ti][row0 + r] = local[ti * rows_per + r];
                }
            }
        }
        return;
    }
    profile_span!("cpu::matmulB");
    let n_in = w.n_in as usize;
    let n_out = w.n_out as usize;
    let t = xs.len();
    assert_eq!(outs.len(), t);
    let nt = n_threads().max(1).min(n_out);
    let rows_per = n_out.div_ceil(nt);

    let mut locals: Vec<Vec<f32>> = vec![vec![0.0f32; t * rows_per]; nt];
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for (g, local) in locals.iter_mut().enumerate() {
            let row0 = g * rows_per;
            handles.push(scope.spawn(move || {
                let mut scratch = vec![0.0f32; n_in];
                let rows = n_out.saturating_sub(row0).min(rows_per);
                for r in 0..rows {
                    crate::quant::dequant_row(
                        w.ty,
                        w.data,
                        (row0 + r) as u64,
                        w.n_in,
                        &mut scratch,
                    );
                    for (ti, x) in xs.iter().enumerate() {
                        let mut acc = 0.0f32;
                        for i in 0..n_in {
                            acc += x[i] * scratch[i];
                        }
                        local[ti * rows_per + r] = acc;
                    }
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    });
    for (g, local) in locals.iter().enumerate() {
        let row0 = g * rows_per;
        let rows = n_out.saturating_sub(row0).min(rows_per);
        for ti in 0..t {
            for r in 0..rows {
                outs[ti][row0 + r] = local[ti * rows_per + r];
            }
        }
    }
}

/// W4A16 split 행 디양자화 — data=packed[n][k/8 u32], scale=[n][k/group],
/// zp=8(sym 상수). 그룹·스케일 dtype 일반화(W4-1: g128/f16 27B · g32/bf16 35B).
/// cpu matmul 전용(레퍼런스 f32 — 비트 계약은 lane 소관).
fn dequant_row_w4a16_split(
    q: &[u8],
    s: &[u8],
    row: usize,
    k: usize,
    group: usize,
    scale_bf16: bool,
    out: &mut [f32],
) {
    let nb = k / group;
    let qrow = &q[row * (k / 2)..];
    let srow = &s[row * (k * 2 / group)..];
    for b in 0..nb {
        let bits = u16::from_le_bytes([srow[2 * b], srow[2 * b + 1]]);
        let sc = if scale_bf16 {
            crate::quant::deq::bf16_to_f32(bits)
        } else {
            crate::quant::half_to_f32(bits)
        };
        for i in 0..group {
            let woff = 4 * (b * (group / 8) + i / 8);
            let w =
                u32::from_le_bytes([qrow[woff], qrow[woff + 1], qrow[woff + 2], qrow[woff + 3]]);
            let nib = ((w >> (4 * (i % 8))) & 0xF) as i32;
            out[b * group + i] = (nib - 8) as f32 * sc;
        }
    }
}

/// logits → argmax (greedy와 동일 의미, 트레이트 기본구현용).
pub fn greedy_from(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    let mut snd = (usize::MAX, f32::NEG_INFINITY);
    let top2 = llm170_diag::dump::opts().top2;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            snd = (best, bv);
            bv = v;
            best = i;
        } else if v > snd.1 {
            snd = (i, v);
        }
    }
    if top2 {
        // 107 W1: 근접타이 마진 계측 — 두 변형의 아그맥스 뒤집힘이 합법
        // 타이인지(마진 < 엡실론) 판정하는 1회 측정용 계측.
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        eprintln!(
            "[top2] #{n} best={best}({bv:.4}) 2nd={}({:.4}) margin={:.4}",
            snd.0,
            snd.1,
            bv - snd.1
        );
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::greedy_from;

    /// greedy 아그맥스 계약(2026-10-09): 엄격 비교 → 동률은 최저 인덱스,
    /// NaN은 순위 제외(비교 false), 전부 NaN이면 0 — 실수 경로는 finiteness
    /// 필터가 선행한다는 전제(서버 GPU argmax도 같은 의미론을 미러).
    #[test]
    fn greedy_first_max_nan_contract() {
        assert_eq!(greedy_from(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(greedy_from(&[3.0, 3.0, 1.0]), 0);
        assert_eq!(greedy_from(&[1.0, 3.0, 3.0]), 1);
        assert_eq!(greedy_from(&[f32::NAN, 2.0, 1.0]), 1);
        assert_eq!(greedy_from(&[f32::NEG_INFINITY, 0.0, f32::NAN]), 1);
        assert_eq!(greedy_from(&[f32::NAN, f32::NAN]), 0);
    }
}
