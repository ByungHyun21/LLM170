//! W4A16 로더 프로브 — `w4a16-load` / `w4a16-xcheck` (plans/cuda-models.md §3.5).
//!
//! - `w4a16-load <dir>`: 로더 완전성 검증 — 트리플 구조·weight_shape 값·
//!   커버리지(기대 텐서 전수) 판정. 무게는 헤더 + 행 단위 pread만.
//! - `w4a16-xcheck <w4a16_dir> <gguf> [il] [row]`: 니블 순서 확정 —
//!   동일 기저(Qwen3.8-27B) GGUF 행 디양자화와의 상관으로 lsb-first vs
//!   msb-first 판별(quant/lane.rs §3.6 "로더 프로브로 확정" 계약의 이행).
//!   판정: corr(lsb) > corr(msb)+0.1 이고 corr(lsb) > 0.5.
//!
//! 팔 형상: gate_proj ↔ blk.{il}.ffn_gate · full층 q_proj ↔ blk.{il}.attn_q ·
//! GDN층 in_proj_qkv ↔ blk.{il}.attn_qkv (§3.5 실측 대응).

use super::{arg_num, arg_str, finish};
use std::process::ExitCode;

pub fn try_run(cmd: &str, args: &[String]) -> Option<ExitCode> {
    match cmd {
        "w4a16-load" => Some(finish(load(args))),
        "w4a16-xcheck" => Some(finish(xcheck(args))),
        "w4a16-to-gguf" => Some(finish(to_gguf(args))),
        _ => None,
    }
}

/// W4A16 → llm170 dialect GGUF(W4A16G128) — qwen35 엔진 직행 변환.
fn to_gguf(args: &[String]) -> Result<String, String> {
    let dir = arg_str(args, 0, "");
    let out = arg_str(args, 1, "");
    if dir.is_empty() || out.is_empty() {
        return Err("w4a16-to-gguf <w4a16_dir> <out.gguf>".into());
    }
    let m = llm170_core::w4a16::W4a16Model::open(std::path::Path::new(&dir))
        .map_err(|e| e.to_string())?;
    let st = m
        .to_gguf(std::path::Path::new(&out))
        .map_err(|e| e.to_string())?;
    Ok(format!(
        "w4a16-to-gguf: {}개 텐서 {:.1} GB — {:.0}초 → {out} (MTP 15종 미기입)",
        st.tensors,
        st.bytes as f64 / 1e9,
        st.elapsed_s
    ))
}

fn load(args: &[String]) -> Result<String, String> {
    let dir = arg_str(args, 0, "");
    if dir.is_empty() {
        return Err(
            "w4a16-load <dir> — 사용법: llm170 w4a16-load ../models/Qwen3.8-27B-W4A16-AutoRound"
                .into(),
        );
    }
    let m = llm170_core::w4a16::W4a16Model::open(std::path::Path::new(&dir))
        .map_err(|e| e.to_string())?;
    let rep = m.validate().map_err(|e| e.to_string())?;
    let c = &m.cfg;
    let head = format!(
        "w4a16-load {dir}\n  hidden={} layers={} heads={}/{} head_dim={} ffn={} vocab={} interval={}\n  양자화=compressed-tensors pack-quantized int4 sym g{} (zp=8 상수) · 선형 {}개",
        c.hidden,
        c.layers,
        c.heads,
        c.kv_heads,
        c.head_dim,
        c.ffn,
        c.vocab,
        c.full_interval,
        c.group_size,
        m.n_lins()
    );
    let body = rep.summary();
    if rep.ok() {
        Ok(format!("{head}\n{body}  판정: 완전성 검증 통과"))
    } else {
        Err(format!("{head}\n{body}  판정: 검증 실패"))
    }
}

fn xcheck(args: &[String]) -> Result<String, String> {
    let dir = arg_str(args, 0, "");
    let gguf = arg_str(args, 1, "");
    let il = arg_num(args, 2, 0usize);
    let row = arg_num(args, 3, 0u64);
    if dir.is_empty() || gguf.is_empty() {
        return Err("w4a16-xcheck <w4a16_dir> <gguf> [il=0] [row=0]".into());
    }
    let w4 = llm170_core::w4a16::W4a16Model::open(std::path::Path::new(&dir))
        .map_err(|e| e.to_string())?;
    let g = llm170_core::qwen35::Model::load(std::path::Path::new(&gguf))
        .map_err(|e| format!("GGUF 로드: {e}"))?;

    let full = (il + 1).is_multiple_of(w4.cfg.full_interval);
    let mut pairs: Vec<(String, String)> = vec![(
        format!("model.language_model.layers.{il}.mlp.gate_proj"),
        format!("blk.{il}.ffn_gate.weight"),
    )];
    if full {
        pairs.push((
            format!("model.language_model.layers.{il}.self_attn.q_proj"),
            format!("blk.{il}.attn_q.weight"),
        ));
    } else {
        pairs.push((
            format!("model.language_model.layers.{il}.linear_attn.in_proj_qkv"),
            format!("blk.{il}.attn_qkv.weight"),
        ));
    }

    let mut out = format!("w4a16-xcheck {dir} ↔ {gguf} (il={il} row={row})\n");
    let mut ok = true;
    for (base, gname) in pairs {
        let (_, k) = w4
            .lin_shape(&base)
            .ok_or_else(|| format!("{base} 부재(W4A16)"))?;
        let q = w4
            .packed_rows_u32(&base, row, row + 1)
            .map_err(|e| e.to_string())?;
        let s = w4
            .scale_rows_u16(&base, row, row + 1)
            .map_err(|e| e.to_string())?;
        let lsb = deq_row(&q, &s, k, false);
        let msb = deq_row(&q, &s, k, true);
        let w = g.w(&gname).ok_or_else(|| format!("GGUF {gname} 부재"))?;
        let mut grow = vec![0f32; w.n_in as usize];
        llm170_core::quant::dequant_row(w.ty, w.data, row, w.n_in, &mut grow);
        if grow.len() != k {
            return Err(format!(
                "{base}: k={k} vs GGUF {gname} k={} 불일치 — 대응 텐서 재확인",
                grow.len()
            ));
        }
        let cl = pearson(&lsb, &grow);
        let cm = pearson(&msb, &grow);
        let verdict = if cl > cm + 0.1 && cl > 0.5 {
            "lsb"
        } else if cm > cl + 0.1 && cm > 0.5 {
            "msb"
        } else {
            "불확정"
        };
        if verdict != "lsb" {
            ok = false;
        }
        out.push_str(&format!(
            "  [{base} ↔ {gname} ({:?})] corr(lsb)={cl:.4} corr(msb)={cm:.4} → {verdict}\n",
            w.ty
        ));
    }
    if ok {
        out.push_str("판정: lsb-first 확정 (워드 j의 니블 n = 원소 8·(j/8)+n) — §3.6 종결");
        Ok(out)
    } else {
        Err(format!(
            "{out}판정: FAIL — lsb-first 아님/불확정 (lane §3.6 계약 위반 — 보고 필요)"
        ))
    }
}

/// f16 비트 → f32 (프로브 전용 소형 — w4a16 모듈 것과 동일 규약).
#[inline]
fn f16b(v: u16) -> f32 {
    let sign = ((v >> 15) as u32) << 31;
    let e = ((v >> 10) & 0x1F) as u32;
    let m = (v & 0x3FF) as u32;
    if e == 0 {
        return ((m as f32) * (2.0f64).powi(-24) as f32).copysign(f32::from_bits(sign));
    }
    if e == 31 {
        return f32::NAN;
    }
    f32::from_bits(sign | ((e - 15 + 127) << 23) | (m << 13))
}

/// 행 1개 디양자화(프로브 전용) — zp=8(sym) 고정, 니블 순서 선택 가능.
fn deq_row(q: &[u32], s: &[u16], k: usize, msb: bool) -> Vec<f32> {
    let mut out = vec![0f32; k];
    for (i, o) in out.iter_mut().enumerate() {
        let word = q[i / 8];
        let sh = if msb { 4 * (7 - (i % 8)) } else { 4 * (i % 8) };
        let nib = ((word >> sh) & 0xF) as i32;
        let sc = f16b(s[i / 128]);
        *o = (nib - 8) as f32 * sc;
    }
    out
}

/// Pearson 상관(프로브 판정용, f64 누산).
fn pearson(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().map(|&v| v as f64).sum::<f64>() / n;
    let mb = b.iter().map(|&v| v as f64).sum::<f64>() / n;
    let mut num = 0.0f64;
    let mut da = 0.0f64;
    let mut db = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        let (dx, dy) = (x as f64 - ma, y as f64 - mb);
        num += dx * dy;
        da += dx * dx;
        db += dy * dy;
    }
    if da == 0.0 || db == 0.0 {
        return f64::NAN;
    }
    num / (da.sqrt() * db.sqrt())
}
