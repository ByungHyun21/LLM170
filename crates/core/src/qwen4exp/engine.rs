
use super::bind::Qwen4ExpCfg;
use crate::st::{StArchive, StDtype, StEntry};
use std::path::Path;

#[derive(Clone, Copy)]
pub(crate) struct SendPtr(usize);

impl SendPtr {
    pub(crate) fn new<T>(p: *mut T) -> Self {
        Self(p as usize)
    }
    pub(crate) unsafe fn as_ptr<T>(&self) -> *mut T {
        self.0 as *mut T
    }
}
pub(crate) fn par_ranges<F: Fn(usize, usize) + Send + Sync>(n: usize, f: F) {
    use std::cell::Cell;
    std::thread_local! {
        static IN_PAR: Cell<bool> = const { Cell::new(false) };
    }
    if IN_PAR.with(|x| x.get()) {
        f(0, n);
        return;
    }
    let t = std::thread::available_parallelism()
        .map_or(1, |x| x.get())
        .min(n.div_ceil(64))
        .max(1);
    if t <= 1 {
        f(0, n);
        return;
    }
    let step = n.div_ceil(t);
    let f_ref = &f;
    IN_PAR.with(|x| x.set(true));
    std::thread::scope(|s| {
        for c in 0..t {
            let (b0, b1) = (c * step, ((c + 1) * step).min(n));
            if b0 >= b1 {
                continue;
            }
            if c == t - 1 {
                f(b0, b1);
            } else {
                s.spawn(move || f_ref(b0, b1));
            }
        }
    });
    IN_PAR.with(|x| x.set(false));
}

const GPTQ_GROUP: usize = 128;

pub struct Model {
    pub cfg: Qwen4ExpCfg,
    ar: StArchive,
    mmaps: Vec<memmap2::Mmap>,
    pub eps: f32,
}

impl Model {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let cfg = Qwen4ExpCfg::load(dir)?;
        let ar = StArchive::open(dir).map_err(|e| format!("safetensors: {e}"))?;
        let mut mmaps = Vec::new();
        for p in ar.shard_paths() {
            let f = std::fs::File::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
            mmaps.push(unsafe { memmap2::Mmap::map(&f).map_err(|e| e.to_string())? });
        }
        Ok(Model {
            cfg,
            ar,
            mmaps,
            eps: 1e-6,
        })
    }

    pub fn ent(&self, name: &str) -> Result<&StEntry, String> {
        self.ar
            .entry(name)
            .ok_or_else(|| format!("텐서 부재: {name}"))
    }

    pub(crate) fn slice(&self, name: &str) -> Result<&[u8], String> {
        let e = self.ent(name)?;
        let base = self
            .ar
            .shard_data_base(e.shard)
            .ok_or_else(|| format!("샤드 좌표 부재: {name}"))?;
        let m = self
            .mmaps
            .get(e.shard)
            .ok_or_else(|| format!("샤드 mmap 부재: {name}"))?;
        Ok(&m[(base + e.begin) as usize..(base + e.end) as usize])
    }

    pub(crate) fn f32_vec(&self, name: &str) -> Result<Vec<f32>, String> {
        let e = self.ent(name)?;
        let d = self.slice(name)?;
        let n = e.nbytes() as usize / e.dtype.nbytes() as usize;
        let mut v = vec![0.0f32; n];
        match e.dtype {
            StDtype::Bf16 => {
                for i in 0..n {
                    v[i] = f32::from_bits(
                        u32::from(u16::from_le_bytes(d[2 * i..2 * i + 2].try_into().unwrap()))
                            << 16,
                    );
                }
            }
            StDtype::F16 => {
                for i in 0..n {
                    v[i] = half_f32(u16::from_le_bytes(d[2 * i..2 * i + 2].try_into().unwrap()));
                }
            }
            other => return Err(format!("{name}: 1-D f32 변환 불가 dtype {other:?}")),
        }
        Ok(v)
    }

    pub fn norm_vec(&self, name: &str) -> Result<Vec<f32>, String> {
        let mut v = self.f32_vec(name)?;
        plus1(&mut v);
        Ok(v)
    }

    pub fn matvec_bf16(&self, name: &str, x: &[f32]) -> Result<Vec<f32>, String> {
        let e = self.ent(name)?;
        if e.dtype != StDtype::Bf16 || e.shape.len() != 2 {
            return Err(format!(
                "{name}: [n,k] BF16 계약 위반 {:?} {:?}",
                e.dtype, e.shape
            ));
        }
        let (n, k) = (e.shape[0] as usize, e.shape[1] as usize);
        if k != x.len() {
            return Err(format!("{name}: k={k} ≠ 입력 {}]", x.len()));
        }
        let d = self.slice(name)?;
        let mut y = vec![0.0f32; n];
        if n as u64 * k as u64 >= 4_000_000 {
            let y_ptr = SendPtr::new(y.as_mut_ptr());
            par_ranges(n, |b0, b1| {
                let yp = unsafe { y_ptr.as_ptr::<f32>() };
                for r in b0..b1 {
                    let row = &d[2 * r * k..2 * (r + 1) * k];
                    let mut acc = 0.0f64;
                    for (j, &xv) in x.iter().enumerate() {
                        let b = u16::from_le_bytes([row[2 * j], row[2 * j + 1]]);
                        acc += (f32::from_bits((b as u32) << 16)) as f64 * xv as f64;
                    }
                    unsafe { *yp.add(r) = acc as f32 };
                }
            });
        } else {
            for (r, yr) in y.iter_mut().enumerate() {
                let row = &d[2 * r * k..2 * (r + 1) * k];
                let mut acc = 0.0f64;
                for (j, &xv) in x.iter().enumerate() {
                    let b = u16::from_le_bytes([row[2 * j], row[2 * j + 1]]);
                    acc += (f32::from_bits((b as u32) << 16)) as f64 * xv as f64;
                }
                *yr = acc as f32;
            }
        }
        Ok(y)
    }

    pub(crate) fn embed_row(&self, name: &str, r: usize) -> Result<Vec<f32>, String> {
        let e = self.ent(name)?;
        let k = *e.shape.last().ok_or("임베딩 shape 부재")? as usize;
        let d = self.slice(name)?;
        let row = &d[2 * r * k..2 * (r + 1) * k];
        Ok((0..k)
            .map(|j| {
                f32::from_bits((u16::from_le_bytes([row[2 * j], row[2 * j + 1]]) as u32) << 16)
            })
            .collect())
    }

    pub(crate) fn expert_row(&self, base: &str, r: usize) -> Result<Vec<f32>, String> {
        let (qw, qz, sc, n, k) = self.expert_triple(base)?;
        let g = GPTQ_GROUP;
        let mut row = vec![0.0f32; k];
        for (j, v) in row.iter_mut().enumerate() {
            let word = u32::from_le_bytes(
                qw[4 * ((j / 8) * n + r)..4 * ((j / 8) * n + r) + 4]
                    .try_into()
                    .unwrap(),
            );
            let nib = ((word >> (4 * (j % 8))) & 0xF) as i32;
            let gi = j / g;
            let zw = u32::from_le_bytes(
                qz[4 * (gi * (n / 8) + r / 8)..4 * (gi * (n / 8) + r / 8) + 4]
                    .try_into()
                    .unwrap(),
            );
            let zero = (((zw >> (4 * (r % 8))) & 0xF) as i32) + 1;
            let sb = 2 * (gi * n + r);
            let s = half_f32(u16::from_le_bytes([sc[sb], sc[sb + 1]]));
            *v = (nib - zero) as f32 * s;
        }
        Ok(row)
    }

    fn expert_triple(&self, base: &str) -> Result<(&[u8], &[u8], &[u8], usize, usize), String> {
        let eq = self.ent(&format!("{base}.qweight"))?;
        let ez = self.ent(&format!("{base}.qzeros"))?;
        let es = self.ent(&format!("{base}.scales"))?;
        if eq.dtype != StDtype::I32 || ez.dtype != StDtype::I32 || es.dtype != StDtype::F16 {
            return Err(format!("{base}: gptq 트리플 dtype 위반"));
        }
        let (n, k) = (eq.shape[1] as usize, eq.shape[0] as usize * 8);
        if ez.shape != [k.div_ceil(GPTQ_GROUP) as u64, (n / 8) as u64]
            || es.shape != [k.div_ceil(GPTQ_GROUP) as u64, n as u64]
        {
            return Err(format!(
                "{base}: gptq 트리플 형상 위반 qw={:?} qz={:?} sc={:?} (n={n},k={k})",
                eq.shape, ez.shape, es.shape
            ));
        }
        Ok((
            self.slice(&format!("{base}.qweight"))?,
            self.slice(&format!("{base}.qzeros"))?,
            self.slice(&format!("{base}.scales"))?,
            n,
            k,
        ))
    }

    pub fn expert_mat(&self, base: &str) -> Result<(Vec<Vec<f32>>, usize, usize), String> {
        let (_, _, _, n, k) = self.expert_triple(base)?;
        let mut rows = Vec::with_capacity(n);
        for r in 0..n {
            rows.push(self.expert_row(base, r)?);
        }
        Ok((rows, n, k))
    }

    pub fn is_linear(&self, il: usize) -> bool {
        (il % self.cfg.full_interval.max(1)) != self.cfg.full_interval.max(1) - 1
    }

    pub fn embed(&self, tokens: &[u32]) -> Result<Vec<Vec<f32>>, String> {
        tokens
            .iter()
            .map(|&t| {
                if t as usize >= self.cfg.vocab {
                    Err(format!("토큰 {t} ≥ vocab {}", self.cfg.vocab))
                } else {
                    self.embed_row("model.language_model.embed_tokens.weight", t as usize)
                }
            })
            .collect()
    }

    pub fn head(&self, h: &[f32]) -> Result<Vec<f32>, String> {
        self.matvec_bf16("lm_head.weight", h)
    }
}

pub(crate) fn plus1(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x += 1.0;
    }
}

pub(crate) fn half_f32(b: u16) -> f32 {
    let s: f32 = if b >> 15 == 1 { -1.0 } else { 1.0 };
    let e = ((b >> 10) & 0x1F) as i32;
    let m = (b & 0x3FF) as f32;
    match e {
        0 => s * m * 2.0f32.powi(-24),
        31 => f32::NAN * s,
        _ => s * (1.0 + m / 1024.0) * 2.0f32.powi(e - 15),
    }
}


pub fn probe(dir: &Path) -> Result<String, String> {
    let m = Model::load(dir)?;
    let cfg = &m.cfg;
    let mut n_lin = 0;
    let mut n_ful = 0;
    let mut ple_layers = Vec::new();
    for il in 0..cfg.num_hidden_layers {
        let has_lin = m
            .ent(&format!(
                "model.language_model.layers.{il}.linear_attn.A_log"
            ))
            .is_ok();
        let has_ful = m
            .ent(&format!(
                "model.language_model.layers.{il}.self_attn.q_proj.weight"
            ))
            .is_ok();
        if has_lin == has_ful {
            return Err(format!("층 {il}: linear/full 어텐션 텐서 이중/부재"));
        }
        n_lin += has_lin as usize;
        n_ful += has_ful as usize;
        if m.ent(&format!(
            "model.language_model.layers.{il}.ple.conv1d.weight"
        ))
        .is_ok()
        {
            ple_layers.push(il);
        }
    }
    if (n_lin, n_ful) != (cfg.n_linear, cfg.n_full) {
        return Err(format!(
            "층 분포 실측 ({n_lin}, {n_ful}) ≠ config ({}, {})",
            cfg.n_linear, cfg.n_full
        ));
    }
    let n_exp_real = (0..cfg.n_experts)
        .filter(|&e| {
            m.ent(&format!(
                "model.language_model.layers.0.mlp.experts.{e}.gate_proj.qweight"
            ))
            .is_ok()
        })
        .count();
    if n_exp_real != cfg.n_experts {
        return Err(format!("전문가 실측 {n_exp_real}/{}", cfg.n_experts));
    }

    let tokens: Vec<u32> = [9707, 11, 728, 264].to_vec();
    let mut xs = m.embed(&tokens)?;
    let mut lines = vec![format!(
        "fn-probe {} (CPU 오라클 골격)\n  구조: linear {} / full {} (interval {}) · PLE 층 {:?} · 전문가 {n_exp_real} top-{} · shared {}",
        dir.display(),
        cfg.n_linear,
        cfg.n_full,
        cfg.full_interval,
        ple_layers,
        cfg.top_k,
        cfg.shared_ffn
    )];
    for il in [0usize, 3usize] {
        let attn_out = if m.is_linear(il) {
            super::stages::linear::forward(&m, il, &xs)?
        } else {
            super::stages::full::forward(&m, il, &xs)?
        };
        for t in 0..xs.len() {
            for i in 0..cfg.hidden_size {
                xs[t][i] += attn_out[t][i];
            }
        }
        let ffn = super::stages::moe::forward(&m, il, &xs)?;
        for t in 0..xs.len() {
            for i in 0..cfg.hidden_size {
                xs[t][i] += ffn[t][i];
            }
        }
        let sum: f64 = xs[0].iter().map(|&v| v as f64).sum();
        let mx = xs[0].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b.abs()));
        if !xs[0].iter().all(|v| v.is_finite()) {
            return Err(format!("층 {il} forward 비유한값"));
        }
        lines.push(format!(
            "  층 {il} ({}) forward 통과 — |x|max={mx:.4} sum={sum:.4}",
            if m.is_linear(il) {
                "linear/GDN"
            } else {
                "full-attn"
            }
        ));
    }
    let logits = m.head(&xs[xs.len() - 1])?;
    let argmax = crate::matmul::greedy_from(&logits);
    let lmax = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    lines.push(format!(
        "  head 통과 — vocab {} 로짓 max={lmax:.4} top1={argmax}",
        logits.len()
    ));
    if let Some(&pil) = ple_layers.first() {
        let v = super::stages::ple::verify(&m, pil)?;
        let mut rp: Vec<Vec<f32>> = xs.iter().map(|e| e.repeat(m.cfg.hc_count)).collect();
        let mut pst = super::stages::ple::PleState::new(m.cfg.hc_count * cfg.hidden_size);
        super::stages::ple::forward(&m, pil, &tokens, 0, &mut rp, &mut pst)
            .map_err(|e| format!("PLE forward 실패: {e}"))?;
        let pm = rp.iter().flatten().fold(0.0f32, |a, &b| a.max(b.abs()));
        if !rp.iter().all(|t| t.iter().all(|v| v.is_finite())) {
            return Err("PLE forward 비유한값".into());
        }
        lines.push(format!("  PLE 층 {pil} forward 통과(|R|max={pm:.4}) — {v}"));
    }
    {
        let lp = "model.language_model.layers.3.attn_hyper_connection";
        let rh: Vec<Vec<f32>> = xs.iter().map(|e| e.repeat(m.cfg.hc_count)).collect();
        let (blk, inj) = super::stages::hc::mix(&m, lp, &rh)?;
        let mut rc = rh;
        for t in 0..rc.len() {
            super::stages::hc::combine_tok(
                &mut rc[t],
                &blk[t],
                &inj[t * m.cfg.hc_count..(t + 1) * m.cfg.hc_count],
                m.cfg.hc_count,
                cfg.hidden_size,
            );
        }
        let hm = rc.iter().flatten().fold(0.0f32, |a, &b| a.max(b.abs()));
        lines.push(format!(
            "  hyper-connection 층 3 mix+combine 통과(|R|max={hm:.4})"
        ));
    }
    {
        let mut kv = super::stages::qsa::KvCache::new();
        let mut ix = super::stages::qsa::QsaCache::new();
        let oa = super::stages::qsa::full_forward(&m, 3, &xs, 0, &mut kv, &mut ix)?;
        let qm = oa.iter().flatten().fold(0.0f32, |a, &b| a.max(b.abs()));
        if !oa.iter().all(|t| t.iter().all(|v| v.is_finite())) {
            return Err("QSA forward 비유한값".into());
        }
        lines.push(format!(
            "  indexer+QSA 층 3 forward 통과(|out|max={qm:.4}) — raw {}토큰·압축 {}블록",
            ix.raw_k.len(),
            ix.comp_k.len()
        ));
    }
    lines.push(
        "  판정: FN D2 통과 — 상태ful 오라클(oracle)·w4a16-ref greedy·골든 사용 가능".to_string(),
    );
    Ok(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_decode() {
        assert_eq!(half_f32(0x3C00), 1.0);
        assert_eq!(half_f32(0xBC00), -1.0);
        assert_eq!(half_f32(0x4000), 2.0);
        assert_eq!(half_f32(0x0000), 0.0);
        assert!((half_f32(0x3555) - 0.33325195).abs() < 1e-6);
    }

    #[test]
    fn fn_probe_real_model() {
        let Some(d) = llm170_diag::flag::val("LLM170_FN_MODEL") else {
            return;
        };
        let rep = probe(std::path::Path::new(&d)).expect("fn-probe 실패");
        eprintln!("{rep}");
        assert!(rep.contains("D2 통과"), "리포트: {rep}");
    }

    #[test]
    fn norm_plus1() {
        let mut v = vec![0.0f32, -0.25, 0.5, -1.0];
        plus1(&mut v);
        assert_eq!(v, vec![1.0, 0.75, 1.5, 0.0]);
    }

    #[test]
    fn fn_norm_contract_real_model() {
        let Some(d) = llm170_diag::flag::val("LLM170_FN_MODEL") else {
            return;
        };
        let m = Model::load(std::path::Path::new(&d)).expect("로드 실패");
        let gemma = "model.language_model.layers.3.self_attn.q_norm.weight";
        let raw = m.f32_vec(gemma).expect("q_norm 부재");
        let syn = m.norm_vec(gemma).expect("norm_vec 실패");
        assert_eq!(raw.len(), syn.len());
        for (r, s) in raw.iter().zip(syn.iter()) {
            assert_eq!(*s, r + 1.0, "Gemma 노름은 정확히 +1");
        }
        let gdn = m
            .f32_vec("model.language_model.layers.0.linear_attn.norm.weight")
            .expect("linear_attn.norm 부재");
        let mean = gdn.iter().sum::<f32>() / gdn.len() as f32;
        assert!(
            (0.8..1.2).contains(&mean),
            "linear_attn.norm 평균 {mean} — 평범 가중치 계약 위반"
        );
    }

    #[test]
    fn fn_qzeros_sym_scan_real_model() {
        let Some(d) = llm170_diag::flag::val("LLM170_FN_MODEL") else {
            return;
        };
        let m = Model::load(std::path::Path::new(&d)).expect("로드 실패");
        let mut hist = [0u64; 16];
        let mut tensors = 0u64;
        let mut const_tensors = 0u64;
        let mut const_value_hist = [0u64; 16];
        let mut layer_hist: std::collections::BTreeMap<u64, [u64; 16]> =
            std::collections::BTreeMap::new();
        let mut worst: Vec<(u64, u64, u64)> = Vec::new();
        let mut names: Vec<&String> =
            m.ar.entries()
                .keys()
                .filter(|n| n.contains(".mlp.experts.") && n.ends_with(".qzeros"))
                .collect();
        names.sort_unstable();
        assert_eq!(names.len(), 48 * 512 * 3, "전문가 qzeros 텐서 수 위반");
        for name in &names {
            let ent = m.ent(name).expect("엔트리");
            assert_eq!(ent.dtype, StDtype::I32, "{name}: dtype");
            let d = m.slice(name).expect("슬라이스");
            let mut th = [0u64; 16];
            for c in d.as_chunks::<4>().0 {
                let w = u32::from_le_bytes(*c);
                for i in 0..8 {
                    th[((w >> (4 * i)) & 0xF) as usize] += 1;
                }
            }
            let used: u64 = th.iter().filter(|&&c| c > 0).count() as u64;
            let (mode_i, &mode_c) = th.iter().enumerate().max_by_key(|(_, c)| *c).unwrap();
            let il: u64 = name
                .split('.')
                .nth(3)
                .and_then(|s| s.parse().ok())
                .expect("층 파싱");
            tensors += 1;
            if used == 1 {
                const_tensors += 1;
                const_value_hist[mode_i] += 1;
            } else {
                let ex: u64 = name
                    .split('.')
                    .nth(5)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                worst.push((il, ex, th.iter().sum::<u64>() - mode_c));
                worst.sort_by_key(|&(_, _, dv)| std::cmp::Reverse(dv));
                worst.truncate(8);
            }
            for i in 0..16 {
                hist[i] += th[i];
                layer_hist.entry(il).or_insert([0; 16])[i] += th[i];
            }
        }
        let total: u64 = hist.iter().sum();
        let mode = hist.iter().enumerate().max_by_key(|(_, c)| *c).unwrap().0;
        let off_mode: u64 = total - hist[mode];
        eprintln!(
            "qzeros 스캔: 텐서 {tensors} · 니블 {total} · 최빠값 {mode}({}) · 상이 {off_mode} ({:.6}%)",
            hist[mode],
            100.0 * off_mode as f64 / total as f64
        );
        eprintln!("전역 히스토그램: {hist:?}");
        eprintln!("상수 텐서 {const_tensors}/{tensors} — 상숫값 분포 {const_value_hist:?}");
        eprintln!("비상수 최악 8(il,e,상이수): {worst:?}");
        for &il in &[0u64, 1, 2, 23, 47] {
            if let Some(h) = layer_hist.get(&il) {
                let t: u64 = h.iter().sum();
                let md = h.iter().enumerate().max_by_key(|(_, c)| *c).unwrap().0;
                eprintln!("층 {il}: 니블 {t} · 최빠 {md} · 상이 {}", t - h[md]);
            }
        }
        assert!(
            off_mode == 0,
            "비상수 qzeros 발견 — GPTQ-asym 스토어 경로 필요"
        );

        for base in [
            "model.language_model.layers.0.mlp.experts.0.gate_proj",
            "model.language_model.layers.1.mlp.experts.7.down_proj",
            "model.language_model.layers.2.mlp.experts.13.up_proj",
            "model.language_model.layers.23.mlp.experts.5.gate_proj",
            "model.language_model.layers.47.mlp.experts.511.down_proj",
        ] {
            let d = m.slice(&format!("{base}.qweight")).expect("qweight");
            let mut qh = [0u64; 16];
            for c in d.as_chunks::<4>().0 {
                let w = u32::from_le_bytes(*c);
                for i in 0..8 {
                    qh[((w >> (4 * i)) & 0xF) as usize] += 1;
                }
            }
            eprintln!("qweight {base}: {qh:?}");
            assert!(qh.iter().all(|&c| c > 0), "니블 전 범위 미사용 — 팩 의심");
        }

        let base = "model.language_model.layers.23.mlp.experts.5.gate_proj";
        let r = 7usize;
        let row = m.expert_row(base, r).expect("expert_row");
        let (qw, qz, sc, n, k) = m.expert_triple(base).expect("트리플");
        const ZERO: i32 = 8;
        for j in [0usize, 1, 63, 127, 128, k / 2, k - 1] {
            let word = u32::from_le_bytes(qw[4 * ((j / 8) * n + r)..][..4].try_into().unwrap());
            let nib = ((word >> (4 * (j % 8))) & 0xF) as i32;
            let sb = 2 * ((j / GPTQ_GROUP) * n + r);
            let s = half_f32(u16::from_le_bytes([sc[sb], sc[sb + 1]]));
            assert_eq!(row[j], (nib - ZERO) as f32 * s, "j={j} 디큐트 불일치");
        }
        let zw = u32::from_le_bytes(qz[4 * (r / 8)..][..4].try_into().unwrap());
        assert_eq!(((zw >> (4 * (r % 8))) & 0xF) as usize, mode);
        eprintln!("교차검증 통과: zero=8(스캔 니블 7 + vLLM zero_offset 1) · expert_row 비트일치");
    }
}
