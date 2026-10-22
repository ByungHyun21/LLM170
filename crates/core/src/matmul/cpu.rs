
pub fn n_threads() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(32)
    })
}
use super::weight::Weight;
use llm170_diag::profile_span;

fn run_rows(n: usize, job: impl Fn(usize) -> Box<dyn FnOnce() + Send + 'static>) {
    crate::gdn::ar_pool::run_par(n, job);
}

pub fn matmul(x: &[f32], w: &Weight, out: &mut [f32]) {
    profile_span!("cpu::matmul1");
    let n_in = w.n_in as usize;
    let nt = n_threads().max(1).min(out.len().max(1));
    let rows_per = out.len().div_ceil(nt).max(1);
    let nch = out.len().div_ceil(rows_per);
    if w.ty == crate::wtype::WType::W4a16Split {
        let scale = w
            .aux
            .expect("w4a16 split: aux(scale) 필수 계약 — Model::w 보장");
        if crate::quant::exact::hi_prec() {
            let (group, sbf) = (w.group, w.scale_bf16);
            let qb = n_in / 2;
            let sb = n_in * 2 / group;
            for r in 0..out.len() {
                out[r] = crate::quant::exact::dot_row_w4a16_exact_bytes(
                    &w.data[r * qb..],
                    &scale[r * sb..],
                    x,
                    n_in,
                    group,
                    sbf,
                ) as f32;
            }
            return;
        }
        let (dp, dl, sp, sl) = (
            w.data.as_ptr() as usize,
            w.data.len(),
            scale.as_ptr() as usize,
            scale.len(),
        );
        let (xp, xl) = (x.as_ptr() as usize, x.len());
        let (op, olen) = (out.as_mut_ptr() as usize, out.len());
        let (group, sbf) = (w.group, w.scale_bf16);
        run_rows(nch, move |g| {
            Box::new(move || {
                let data = unsafe { std::slice::from_raw_parts(dp as *const u8, dl) };
                let scale = unsafe { std::slice::from_raw_parts(sp as *const u8, sl) };
                let x = unsafe { std::slice::from_raw_parts(xp as *const f32, xl) };
                let out = unsafe { std::slice::from_raw_parts_mut(op as *mut f32, olen) };
                let row0 = g * rows_per;
                let rows = olen.saturating_sub(row0).min(rows_per);
                let mut scratch = vec![0.0f32; n_in];
                for r in 0..rows {
                    dequant_row_w4a16_split(data, scale, row0 + r, n_in, group, sbf, &mut scratch);
                    let mut acc = 0.0f32;
                    for i in 0..n_in {
                        acc += x[i] * scratch[i];
                    }
                    out[row0 + r] = acc;
                }
            })
        });
        return;
    }
    let (dp, dl) = (w.data.as_ptr() as usize, w.data.len());
    let (xp, xl) = (x.as_ptr() as usize, x.len());
    let (op, olen) = (out.as_mut_ptr() as usize, out.len());
    let (ty, n_in_w) = (w.ty, w.n_in);
    run_rows(nch, move |g| {
        Box::new(move || {
            let data = unsafe { std::slice::from_raw_parts(dp as *const u8, dl) };
            let x = unsafe { std::slice::from_raw_parts(xp as *const f32, xl) };
            let out = unsafe { std::slice::from_raw_parts_mut(op as *mut f32, olen) };
            let row0 = g * rows_per;
            let rows = olen.saturating_sub(row0).min(rows_per);
            let mut scratch = vec![0.0f32; n_in];
            for r in 0..rows {
                crate::quant::dequant_row(ty, data, (row0 + r) as u64, n_in_w, &mut scratch);
                let mut acc = 0.0f32;
                for i in 0..n_in {
                    acc += x[i] * scratch[i];
                }
                out[row0 + r] = acc;
            }
        })
    });
}

pub fn matmul_batch(xs: &[Vec<f32>], w: &Weight, outs: &mut [Vec<f32>]) {
    let refs: Vec<&[f32]> = xs.iter().map(|v| v.as_slice()).collect();
    matmul_batch_ref(&refs, w, outs);
}

pub fn matmul_batch_ref(xs: &[&[f32]], w: &Weight, outs: &mut [Vec<f32>]) {
    profile_span!("cpu::matmulB");
    let n_in = w.n_in as usize;
    let n_out = w.n_out as usize;
    let t = xs.len();
    assert_eq!(outs.len(), t);
    let nt = n_threads().max(1).min(n_out.max(1));
    let rows_per = n_out.div_ceil(nt).max(1);
    let nch = n_out.div_ceil(rows_per);
    let lstride = t * rows_per;
    let mut locals: Vec<f32> = vec![0.0f32; nch * lstride];
    let locals_base = locals.as_mut_ptr() as usize;
    let (dp, dl) = (w.data.as_ptr() as usize, w.data.len());
    let (ty, n_in_w) = (w.ty, w.n_in);
    let xp: Vec<usize> = xs.iter().map(|v| v.as_ptr() as usize).collect();
    let xl = n_in;
    let split = w.ty == crate::wtype::WType::W4a16Split;
    let (sp, sl) = match w.aux {
        Some(s) if split => (s.as_ptr() as usize, s.len()),
        _ => (0usize, 0usize),
    };
    let (group, sbf) = (w.group, w.scale_bf16);
    run_rows(nch, move |g| {
        let xp = xp.clone();
        Box::new(move || {
            let data = unsafe { std::slice::from_raw_parts(dp as *const u8, dl) };
            let local = unsafe {
                std::slice::from_raw_parts_mut((locals_base as *mut f32).add(g * lstride), lstride)
            };
            let xsl: Vec<&[f32]> = xp
                .iter()
                .map(|&p| unsafe { std::slice::from_raw_parts(p as *const f32, xl) })
                .collect();
            let row0 = g * rows_per;
            let rows = n_out.saturating_sub(row0).min(rows_per);
            let mut scratch = vec![0.0f32; n_in];
            for r in 0..rows {
                if split {
                    let scale = unsafe { std::slice::from_raw_parts(sp as *const u8, sl) };
                    dequant_row_w4a16_split(data, scale, row0 + r, n_in, group, sbf, &mut scratch);
                } else {
                    crate::quant::dequant_row(ty, data, (row0 + r) as u64, n_in_w, &mut scratch);
                }
                for (ti, x) in xsl.iter().enumerate() {
                    let mut acc = 0.0f32;
                    for i in 0..n_in {
                        acc += x[i] * scratch[i];
                    }
                    local[ti * rows_per + r] = acc;
                }
            }
        })
    });
    for g in 0..nch {
        let row0 = g * rows_per;
        let rows = n_out.saturating_sub(row0).min(rows_per);
        for ti in 0..t {
            for r in 0..rows {
                outs[ti][row0 + r] = locals[g * lstride + ti * rows_per + r];
            }
        }
    }
}

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
    use super::{dequant_row_w4a16_split, greedy_from};

    #[test]
    fn split_dequant_row_g128_f16() {
        let (k, group) = (256usize, 128usize);
        let mut q = vec![0u8; k / 2];
        for w in 0..k / 8 {
            let mut word: u32 = 0;
            for j in 0..8u32 {
                word |= (j & 0xF) << (4 * j);
            }
            q[w * 4..w * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        let mut s = Vec::new();
        s.extend_from_slice(&0x3C00u16.to_le_bytes());
        s.extend_from_slice(&0x3800u16.to_le_bytes());
        let mut out = vec![0f32; k];
        dequant_row_w4a16_split(&q, &s, 0, k, group, false, &mut out);
        for (i, v) in out.iter().enumerate() {
            let nib = (i % 8) as i32;
            let sc = if i < 128 { 1.0 } else { 0.5 };
            assert_eq!(*v, (nib - 8) as f32 * sc, "i={i}");
        }
    }

    #[test]
    fn split_dequant_row_g32_bf16() {
        let (k, group) = (64usize, 32usize);
        let mut q = vec![0u8; k / 2];
        for w in 0..k / 8 {
            let mut word: u32 = 0;
            for j in 0..8u32 {
                word |= ((7 - j) & 0xF) << (4 * j);
            }
            q[w * 4..w * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        let mut s = Vec::new();
        for _ in 0..2 {
            s.extend_from_slice(&0x3F80u16.to_le_bytes());
        }
        let mut out = vec![0f32; k];
        dequant_row_w4a16_split(&q, &s, 0, k, group, true, &mut out);
        for (i, v) in out.iter().enumerate() {
            let nib = (7 - (i % 8)) as i32;
            assert_eq!(*v, (nib - 8) as f32, "i={i}");
        }
    }

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
