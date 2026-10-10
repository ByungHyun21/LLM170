//! [R2] 모듈 게이트 — w4a16-gemv/gemm(커널 vs core lane 비트 판정).

use super::super::{arg, arg_str};

/// 모듈 게이트 — 커널 vs core `dot_row_w4a16_lane` 비트 판정.
/// 형상은 스토어에서 자동 열거(distinct (n,k) → 첫 base), x는 결정적 생성.
///   w4a16-gemv <dir> [--rows N] [--seed X]          (t=1)
///   w4a16-gemm <dir> [--rows N] [--t N≤8] [--seed X]
pub(super) fn gemm_gate(args: &[String], default_t: usize) -> Result<String, String> {
    let dir = arg_str(args, 0, "");
    if dir.is_empty() {
        return Err(
            "w4a16-gemv|w4a16-gemm <dir> [--rows N] [--t N] [--seed X] — 사용법: llm170 w4a16-gemv ../models/Qwen3.8-27B-W4A16-AutoRound".into(),
        );
    }
    let mut rows_limit = 4usize;
    let mut t = default_t;
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut lin: Option<String> = None;
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--rows" => {
                rows_limit = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--rows requires a number")?;
            }
            "--t" => {
                t = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--t requires a number")?;
            }
            "--seed" => {
                seed = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--seed requires a number")?;
            }
            "--lin" => {
                lin = Some(arg(&mut it, "--lin requires a name")?.to_string());
            }
            other => return Err(format!("unknown flag: {other}")),
        }
    }
    if t == 0 || t > 8 {
        return Err(format!("--t {t}: 1..=8"));
    }
    let g4 = llm170_backend_gpu::Gptq4::new()?;
    // 형상 열거 — --lin이면 model.w()(순열 사본 포함) 단일, 아니면 store 전수.
    let mut shapes: std::collections::BTreeMap<(usize, usize), String> =
        std::collections::BTreeMap::new();
    let mut lin_data: Option<(Vec<u8>, Vec<u8>, usize, bool)> = None;
    let store = llm170_core::w4a16::W4a16Model::open(std::path::Path::new(&dir))
        .map_err(|e| e.to_string())?;
    if let Some(name) = &lin {
        let model = llm170_core::qwen35::Model::load(std::path::Path::new(&dir))
            .map_err(|e| e.to_string())?;
        let w = model
            .w(name)
            .ok_or_else(|| format!("--lin {name}: 무게 없음"))?;
        let s = w.aux.ok_or_else(|| format!("--lin {name}: split 아님"))?;
        shapes.insert((w.n_out as usize, w.n_in as usize), name.clone());
        lin_data = Some((w.data.to_vec(), s.to_vec(), w.group, w.scale_bf16));
    } else {
        for (base, n, k) in store.lin_shapes() {
            shapes.entry((n, k)).or_insert(base);
        }
    }
    let mut lines = Vec::new();
    let mut all_ok = true;
    let n_shape = shapes.len();
    for ((n, k), base) in &shapes {
        let (qb, sb) = if let Some((q, s, _, _)) = &lin_data {
            (q.as_slice(), s.as_slice())
        } else {
            (
                store
                    .tensor_slice(&format!("{base}.weight_packed"))
                    .ok_or_else(|| format!("{base}: weight_packed 슬라이스 부재"))?,
                store
                    .tensor_slice(&format!("{base}.weight_scale"))
                    .ok_or_else(|| format!("{base}: weight_scale 슬라이스 부재"))?,
            )
        };
        // 그룹·스케일 dtype — --lin이면 Weight 실측, 아니면 스토어 실측.
        let (group, bf16) = match &lin_data {
            Some((_, _, g, b)) => (*g, *b),
            None => (store.group(), store.scale_is_bf16(base)),
        };
        let r = rows_limit.min(*n);
        let q: Vec<u32> = qb[..r * (k / 8) * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        let s: Vec<u16> = sb[..r * (k / group) * 2]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c))
            .collect();
        // 참조용 f32 스케일 — f16/bf16 디코드(둘 다 정확).
        let sf32: Vec<f32> = s
            .iter()
            .map(|&h| {
                if bf16 {
                    llm170_core::quant::deq::bf16_to_f32(h)
                } else {
                    llm170_core::quant::half_to_f32(h)
                }
            })
            .collect();
        // x — 결정적(splitmix64) f16 비트. 스케일이 작은 모델이라 ±1 균일.
        let mut rnd = SplitMix64::new(seed ^ ((*n as u64) << 32) ^ *k as u64);
        let x: Vec<u16> = (0..t * k)
            .map(|_| llm170_backend_gpu::f32_to_f16(rnd.next_pm1()))
            .collect();
        let got = g4.gemm(&x, t, &q, &s, r, *k, group, bf16)?;
        let z8 = vec![8u32; k / group];
        let mut mism = 0usize;
        let mut maxd = 0f64;
        for ti in 0..t {
            for o in 0..r {
                let qrow = &q[o * (k / 8)..(o + 1) * (k / 8)];
                let srow = &sf32[o * (k / group)..(o + 1) * (k / group)];
                let want = llm170_core::quant::dot_row_w4a16_lane_group(
                    qrow,
                    &z8,
                    srow,
                    &x[ti * k..(ti + 1) * k],
                    group,
                );
                let g = got[ti * r + o];
                if g.to_bits() != want.to_bits() {
                    mism += 1;
                    maxd = maxd.max((g as f64 - want as f64).abs());
                }
            }
        }
        let ok = mism == 0;
        all_ok &= ok;
        lines.push(format!(
            "  n={n:<6} k={k:<6} g{group}{} rows={r} t={t}  {}",
            if bf16 { "/bf16" } else { "/f16" },
            if ok {
                "PASS(비트일치)".to_string()
            } else {
                format!("FAIL mism={mism} maxdiff={maxd:.3e}")
            }
        ));
    }
    let name = if default_t == 1 {
        "w4a16-gemv"
    } else {
        "w4a16-gemm"
    };
    let head = format!(
        "{name} {dir}\n  형상 {n_shape}종 × rows≤{rows_limit} × t={t} — 커널 vs dot_row_w4a16_lane(비트 판정)"
    );
    let body = lines.join("\n");
    if all_ok {
        Ok(format!("{head}\n{body}\n  판정: 전 형상 비트일치"))
    } else {
        Err(format!("{head}\n{body}\n  판정: 불일치"))
    }
}

/// splitmix64 — 결정적 테스트 입력(rand 크레이트 금지 계약).
struct SplitMix64(u64);
impl SplitMix64 {
    fn new(s: u64) -> Self {
        Self(s)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_pm1(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}
