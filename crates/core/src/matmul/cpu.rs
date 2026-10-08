/// W4A8 정수 GEMV 경로 활성 (LLM170_W4A8=1) — iq4_xs·q3_K 디코드
/// matmul을 레인 f64 미러 정수 내적으로 전환. GPU frame/value 경로와
/// 동일 비트 (그룹핑 무관 설계). 프리필(t>1)은 무관.
/// out[o] = Σ_i x[i]·W[o,i] (단일 토큰). 스레드별 행 슬라이스 소유.
/// 원시 HIP 디코드 (LLM170_RAWHIP=1) — 백엔드가 상주 DecodeState로
/// 토큰 1스텝 전체를 수행. 엔진은 임베딩 dequant·pos만 제공.
pub fn n_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(32)
}
use super::weight::Weight;
use llm170_diag::profile_span;
pub fn w4a8_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| llm170_diag::flag::on("LLM170_W4A8"))
}

/// W4A8 대상 타입 (정수 커널·미러 구현 완료분).
pub fn w4a8_ty(ty: crate::wtype::WType) -> bool {
    matches!(ty, |crate::wtype::WType::Iq4Xs| crate::wtype::WType::Iq3S
        | crate::wtype::WType::Q3K
        | crate::wtype::WType::Q4K
        | crate::wtype::WType::Q5K
        | crate::wtype::WType::Q8_0
        | crate::wtype::WType::Q5_1
        | crate::wtype::WType::Iq4Nl
        | crate::wtype::WType::Q6K)
}

pub fn matmul(x: &[f32], w: &Weight, out: &mut [f32]) {
    profile_span!("cpu::matmul1");
    // W4A16 split(§3.5 A안 직접 로드) — 분리 버퍼 디양자화 + f32 내적.
    // f32 레퍼런스(비트 격리 계약은 quant lane 소관).
    if w.ty == crate::wtype::WType::W4a16G128Split {
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
                        dequant_row_w4a16_split(w.data, scale, row0 + r, n_in, &mut scratch);
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
    // W4A8 디코드 전환 — 활성 시 전 경로 동일 비트
    if w4a8_enabled() && w4a8_ty(w.ty) && x.len() == w.n_in as usize {
        let y = crate::quant::quantize_row_q8_ref(x);
        let blck = w.ty.blck_size() as usize;
        let bsize = w.ty.type_size() as usize;
        let row_bytes = (w.n_in as usize / blck) * bsize;
        for (o, out_o) in out.iter_mut().enumerate() {
            let row = &w.data[o * row_bytes..];
            *out_o = match w.ty {
                crate::wtype::WType::Q3K => crate::quant::dot_row_w4a8_q3k_lane(row, w.n_in, &y),
                crate::wtype::WType::Iq3S => crate::quant::dot_row_w4a8_iq3s_lane(row, w.n_in, &y),
                crate::wtype::WType::Q4K => crate::quant::dot_row_w4a8_q4k_lane(row, w.n_in, &y),
                crate::wtype::WType::Q5K => crate::quant::dot_row_w4a8_q5k_lane(row, w.n_in, &y),
                crate::wtype::WType::Q8_0 => crate::quant::dot_row_w4a8_q8_0_lane(row, w.n_in, &y),
                crate::wtype::WType::Iq4Nl => {
                    crate::quant::dot_row_w4a8_iq4nl_lane(row, w.n_in, &y)
                }
                crate::wtype::WType::Q6K => crate::quant::dot_row_w4a8_q6k_lane(row, w.n_in, &y),
                crate::wtype::WType::Q5_1 => crate::quant::dot_row_w4a8_q5_1_lane(row, w.n_in, &y),
                crate::wtype::WType::Iq4Xs => {
                    crate::quant::dot_row_w4a8_iq4xs_lane(row, w.n_in, &y)
                }
                // w4a8_ty 진입 게이트가 9타입 전부 위 팔로 커버 — 신규 타입
                // 추가 시 여기서 즉시 패닉(무결 오염 방지 계약).
                _ => unreachable!("w4a8_ty에 포함됐으나 lane 미구현: {:?}", w.ty),
            };
        }
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
    if w.ty == crate::wtype::WType::W4a16G128Split {
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
                        dequant_row_w4a16_split(w.data, scale, row0 + r, n_in, &mut scratch);
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
    // W4A8 (지원 타입) — 행별 레인 미러 정수 내적 (GPU 배치 경로와 동일 비트)
    if w4a8_enabled() && w4a8_ty(w.ty) {
        let y_all: Vec<_> = xs
            .iter()
            .map(|r| crate::quant::quantize_row_q8_ref(r))
            .collect();
        let blck = w.ty.blck_size() as usize;
        let bsize = w.ty.type_size() as usize;
        let row_bytes = (w.n_in as usize / blck) * bsize;
        for (ti, out) in outs.iter_mut().enumerate() {
            let y = &y_all[ti];
            for (o, out_o) in out.iter_mut().enumerate() {
                let row = &w.data[o * row_bytes..];
                *out_o = match w.ty {
                    crate::wtype::WType::Q3K => crate::quant::dot_row_w4a8_q3k_lane(row, w.n_in, y),
                    crate::wtype::WType::Iq3S => {
                        crate::quant::dot_row_w4a8_iq3s_lane(row, w.n_in, y)
                    }
                    crate::wtype::WType::Q4K => crate::quant::dot_row_w4a8_q4k_lane(row, w.n_in, y),
                    crate::wtype::WType::Q5K => crate::quant::dot_row_w4a8_q5k_lane(row, w.n_in, y),
                    crate::wtype::WType::Q8_0 => {
                        crate::quant::dot_row_w4a8_q8_0_lane(row, w.n_in, y)
                    }
                    crate::wtype::WType::Iq4Nl => {
                        crate::quant::dot_row_w4a8_iq4nl_lane(row, w.n_in, y)
                    }
                    crate::wtype::WType::Q6K => crate::quant::dot_row_w4a8_q6k_lane(row, w.n_in, y),
                    crate::wtype::WType::Q5_1 => {
                        crate::quant::dot_row_w4a8_q5_1_lane(row, w.n_in, y)
                    }
                    crate::wtype::WType::Iq4Xs => {
                        crate::quant::dot_row_w4a8_iq4xs_lane(row, w.n_in, y)
                    }
                    _ => unreachable!("w4a8_ty에 포함됐으나 lane 미구현: {:?}", w.ty),
                };
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

/// W4A16 split 행 디양자화 — data=packed[n][k/8 u32], scale=[n][k/128 u16],
/// zp=8(sym 상수). cpu matmul 전용(레퍼런스 f32).
fn dequant_row_w4a16_split(q: &[u8], s: &[u8], row: usize, k: usize, out: &mut [f32]) {
    let nb = k / 128;
    let qrow = &q[row * (k / 2)..];
    let srow = &s[row * (k / 64)..];
    for b in 0..nb {
        let sc = crate::quant::half_to_f32(u16::from_le_bytes([srow[2 * b], srow[2 * b + 1]]));
        for i in 0..128usize {
            let woff = 4 * (b * 16 + i / 8);
            let w =
                u32::from_le_bytes([qrow[woff], qrow[woff + 1], qrow[woff + 2], qrow[woff + 3]]);
            let nib = ((w >> (4 * (i % 8))) & 0xF) as i32;
            out[b * 128 + i] = (nib - 8) as f32 * sc;
        }
    }
}

/// W4A8 변형 단일 벡터 matmul — x를 q8로 양자화해 타입별 정수 내적.
/// 성능 경로: 기준(f32) 대비 활성 양자화 오차 허용 전제.
pub fn matmul_w4a8(x: &[f32], w: &Weight, out: &mut [f32]) {
    profile_span!("cpu::matmul_w4a8");
    use crate::quant::{dot_row_w4a8, quantize_row_q8_ref};
    let n_in = w.n_in as usize;
    let y = quantize_row_q8_ref(x);
    let nt = n_threads().max(1).min(out.len());
    let rows_per = out.len().div_ceil(nt);
    let mut chunks: Vec<&mut [f32]> = out.chunks_mut(rows_per).collect();
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for (lo, ch) in chunks.iter_mut().enumerate() {
            let row0 = lo * rows_per;
            let y = &y;
            handles.push(scope.spawn(move || {
                for (r, o) in ch.iter_mut().enumerate() {
                    let row = row0 + r;
                    let base = row * (n_in / w.ty.blck_size() as usize) * w.ty.type_size() as usize;
                    *o = dot_row_w4a8(w.ty, &w.data[base..], w.n_in, y);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    });
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
        // 107 W1: 근접타이 마진 원장 — 두 변형의 아그맥스 뒤집힘이 합법
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
