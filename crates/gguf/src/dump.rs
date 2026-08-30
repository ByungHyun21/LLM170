//! gguf-dump 텍스트 리포트 (llama.cpp gguf_dump 대응).

use std::collections::BTreeMap;
use std::io::Write;

use crate::{GgufFile, TensorInfo};

pub fn fmt_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut unit = 0;
    while v >= 1024.0 && unit < UNITS.len() - 1 {
        v /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{n} B")
    } else {
        format!("{v:.2} {}", UNITS[unit])
    }
}

fn fmt_dims(t: &TensorInfo) -> String {
    let d: Vec<String> = t.ne[..t.n_dims as usize].iter().map(|v| v.to_string()).collect();
    format!("[{}]", d.join(", "))
}

/// 전체 덤프를 `out` 에 기록. `tensor_limit`=Some(n) 이면 텐서 표 상위 n개만.
/// `meta_only` 이면 텐서 표 생략(타입별 집계는 유지).
pub fn write_dump(f: &GgufFile, tensor_limit: Option<usize>, meta_only: bool, out: &mut impl Write) -> std::io::Result<()> {
    writeln!(out, "# GGUF dump: {}", f.path.display())?;
    writeln!(out, "file_size       : {}", fmt_bytes(f.file_size))?;
    writeln!(out, "version         : {}", f.version)?;
    writeln!(out, "alignment       : {}", f.alignment)?;
    writeln!(out, "data_offset     : {} ({})", f.data_offset, fmt_bytes(f.data_offset))?;
    writeln!(out, "kv_count        : {}", f.kv.len())?;
    writeln!(out, "tensor_count    : {}", f.tensors.len())?;
    if let Some(a) = f.arch() {
        writeln!(out, "architecture    : {a}")?;
    }
    if let Some(s) = f.split() {
        writeln!(out, "split           : part {} of {} (tensors in part: {})", s.no, s.count, s.tensors_count)?;
    }
    if let Some(total) = f.tensor_bytes_total() {
        writeln!(out, "tensor_bytes    : {} (data section {} → {}, {} slack)",
            fmt_bytes(total), fmt_bytes(f.data_offset),
            fmt_bytes(f.data_offset + total),
            fmt_bytes(f.file_size.saturating_sub(f.data_offset + total)))?;
    }

    writeln!(out, "\n## metadata")?;
    for (k, v) in &f.kv {
        writeln!(out, "  {k} = {v}")?;
    }

    // 타입별 집계 — UD 혼합 양자화 구성 확인이 주 목적
    let mut by_type: BTreeMap<&str, (u64, u64)> = BTreeMap::new(); // name -> (bytes, count)
    let mut broken = 0usize;
    for t in &f.tensors {
        match t.nbytes() {
            Some(n) => by_type.entry(t.ty.name()).and_modify(|e| { e.0 += n; e.1 += 1; }).or_insert((n, 1)),
            None => { broken += 1; continue; }
        };
    }
    let total: u64 = by_type.values().map(|(n, _)| n).sum();
    writeln!(out, "\n## tensor bytes by type")?;
    writeln!(out, "  {:<10} {:>12} {:>8} {:>8} {:>7}", "type", "bytes", "share", "tensors", "bpw")?;
    for (name, (bytes, count)) in by_type.iter() {
        let share = if total > 0 { *bytes as f64 / total as f64 * 100.0 } else { 0.0 };
        let bpw = f.tensors.iter().find(|t| t.ty.name() == *name).map(|t| t.ty.bits_per_weight()).unwrap_or(0.0);
        writeln!(out, "  {name:<10} {:>12} {:>7.2}% {:>8} {:>7.3}", fmt_bytes(*bytes), share, count, bpw)?;
    }
    if broken > 0 {
        writeln!(out, "  (경고: nbytes 계산 불가 텐서 {broken}개 — ne[0] % blck != 0)")?;
    }

    if !meta_only {
        writeln!(out, "\n## tensors")?;
        let shown = tensor_limit.unwrap_or(f.tensors.len());
        for (i, t) in f.tensors.iter().enumerate().take(shown) {
            let size = t.nbytes().map(fmt_bytes).unwrap_or_else(|| "??".into());
            let range = t.file_range(f.data_offset)
                .map(|(a, b)| format!("{a}..{b}"))
                .unwrap_or_else(|| "??".into());
            writeln!(out, "  {:>5} {:<44} {:<26} {:<8} off={:<12} {:>10} @{}",
                i, t.name, fmt_dims(t), t.ty.name(), t.offset, size, range)?;
        }
        if shown < f.tensors.len() {
            writeln!(out, "  … {} more (use --limit to show)", f.tensors.len() - shown)?;
        }
    }
    Ok(())
}
