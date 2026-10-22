
use super::engine;
use crate::st::{StArchive, StDtype, StEntry};
use crate::w4a16::W4a16Model;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

const G: usize = 128;
const QZ_WORD: u32 = 0x7777_7777;
const SHAPES: [(usize, usize); 2] = [(640, 2560), (2560, 640)];

fn expert_base(name: &str) -> Option<&str> {
    name.strip_suffix(".qweight")
        .filter(|b| b.contains(".mlp.experts."))
}

fn name_safe(name: &str) -> bool {
    name.chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'))
}

fn is_model_shard(file: &str) -> bool {
    let Some(stem) = file.strip_suffix(".safetensors") else {
        return false;
    };
    let Some(rest) = stem.strip_prefix("model-") else {
        return false;
    };
    let mut parts = rest.splitn(2, "-of-");
    let a = parts.next().unwrap_or("");
    let b = parts.next().unwrap_or("");
    !a.is_empty()
        && !b.is_empty()
        && a.bytes().all(|c| c.is_ascii_digit())
        && b.bytes().all(|c| c.is_ascii_digit())
}

enum Out {
    Repack {
        base: String,
        n: usize,
        kw: usize,
        kg: usize,
    },
    Copy { name: String, nbytes: u64 },
}

impl Out {
    fn nbytes(&self) -> u64 {
        match self {
            Out::Repack { n, kw, kg, .. } => {
                (*n as u64) * (*kw as u64) * 4 + (*n as u64) * (*kg as u64) * 4 * 2 + 16
            }
            Out::Copy { nbytes, .. } => *nbytes,
        }
    }
    fn dtypes(&self) -> [(&'static str, Vec<u64>, u64); 3] {
        match self {
            Out::Repack { n, kw, kg, .. } => [
                ("I32", vec![*n as u64, *kw as u64], 4),
                ("F16", vec![*n as u64, (*kg as u64) * 4], 2),
                ("I64", vec![2], 8),
            ],
            Out::Copy { .. } => unreachable!("Copy는 names() 경로만 사용"),
        }
    }
    fn names(&self) -> Vec<(String, &'static str, Vec<u64>, u64)> {
        match self {
            Out::Repack { base, .. } => {
                let dt = self.dtypes();
                [
                    format!("{base}.weight_packed"),
                    format!("{base}.weight_scale"),
                    format!("{base}.weight_shape"),
                ]
                .into_iter()
                .zip(dt)
                .map(|(nm, (d, sh, eb))| {
                    let nb = sh.iter().product::<u64>() * eb;
                    (nm, d, sh, nb)
                })
                .collect()
            }
            Out::Copy { name, nbytes } => {
                vec![(name.clone(), "", Vec::new(), *nbytes)]
            }
        }
    }
}

fn check_triple(ar: &StArchive, base: &str) -> Result<(usize, usize, usize), String> {
    let eq: &StEntry = ar
        .entry(&format!("{base}.qweight"))
        .ok_or_else(|| format!("{base}: qweight 부재"))?;
    let ez = ar
        .entry(&format!("{base}.qzeros"))
        .ok_or_else(|| format!("{base}: qzeros 부재"))?;
    let es = ar
        .entry(&format!("{base}.scales"))
        .ok_or_else(|| format!("{base}: scales 부재"))?;
    if eq.dtype != StDtype::I32 || ez.dtype != StDtype::I32 || es.dtype != StDtype::F16 {
        return Err(format!("{base}: gptq 트리플 dtype 위반"));
    }
    if eq.shape.len() != 2 {
        return Err(format!("{base}: qweight 랭크 {:?} != 2", eq.shape));
    }
    let (kw, n) = (eq.shape[0] as usize, eq.shape[1] as usize);
    let k = kw * 8;
    if !SHAPES.contains(&(n, k)) {
        return Err(format!("{base}: 형상 (n={n},k={k}) — 실측 쌍 아님"));
    }
    let kg = k / G;
    if ez.shape != [kg as u64, (n / 8) as u64] || es.shape != [kg as u64, n as u64] {
        return Err(format!(
            "{base}: qzeros {:?}/scales {:?} 형상 위반 (kg={kg},n={n})",
            ez.shape, es.shape
        ));
    }
    Ok((n, kw, kg))
}

fn mmap_all(ar: &StArchive) -> Result<Vec<memmap2::Mmap>, String> {
    let mut v = Vec::new();
    for p in ar.shard_paths() {
        let f = std::fs::File::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
        v.push(unsafe { memmap2::Mmap::map(&f).map_err(|e| e.to_string())? });
    }
    Ok(v)
}

fn tbytes<'a>(ar: &StArchive, mmaps: &'a [memmap2::Mmap], name: &str) -> Result<&'a [u8], String> {
    let e = ar.entry(name).ok_or_else(|| format!("텐서 부재: {name}"))?;
    let base = ar
        .shard_data_base(e.shard)
        .ok_or_else(|| format!("샤드 좌표 부재: {name}"))?;
    let m = mmaps
        .get(e.shard)
        .ok_or_else(|| format!("샤드 mmap 부재: {name}"))?;
    m.get((base + e.begin) as usize..(base + e.end) as usize)
        .ok_or_else(|| format!("슬라이스 범위 위반: {name}"))
}

fn repack_triple<W: Write>(
    ar: &StArchive,
    mmaps: &[memmap2::Mmap],
    base: &str,
    n: usize,
    kw: usize,
    kg: usize,
    out: &mut W,
) -> Result<(), String> {
    let qw = tbytes(ar, mmaps, &format!("{base}.qweight"))?;
    let qz = tbytes(ar, mmaps, &format!("{base}.qzeros"))?;
    let sc = tbytes(ar, mmaps, &format!("{base}.scales"))?;
    for (i, c) in qz.as_chunks::<4>().0.iter().enumerate() {
        let w = u32::from_le_bytes(*c);
        if w != QZ_WORD {
            return Err(format!(
                "{base}.qzeros[{i}] = {w:#010x} — 비상수 zero-point(D1 계약 위반)"
            ));
        }
    }
    let mut pk = vec![0u8; n * kw * 4];
    for jw in 0..kw {
        let row = &qw[jw * n * 4..(jw + 1) * n * 4];
        for (r, w4) in row.as_chunks::<4>().0.iter().enumerate() {
            let d = (r * kw + jw) * 4;
            pk[d..d + 4].copy_from_slice(w4);
        }
    }
    out.write_all(&pk)
        .map_err(|e| format!("{base}.packed: {e}"))?;
    let kg32 = kg * (G / 32);
    let mut s = vec![0u8; n * kg32 * 2];
    for gi in 0..kg {
        let row = &sc[gi * n * 2..(gi + 1) * n * 2];
        for (r, s2) in row.as_chunks::<2>().0.iter().enumerate() {
            for rep in 0..(G / 32) {
                let d = (r * kg32 + gi * (G / 32) + rep) * 2;
                s[d..d + 2].copy_from_slice(s2);
            }
        }
    }
    out.write_all(&s)
        .map_err(|e| format!("{base}.scale: {e}"))?;
    let k = kw * 8;
    out.write_all(&(n as i64).to_le_bytes())
        .and_then(|_| out.write_all(&(k as i64).to_le_bytes()))
        .map_err(|e| format!("{base}.shape: {e}"))?;
    Ok(())
}

#[derive(Debug)]
pub struct RepackStats {
    pub experts: usize,
    pub tensors: usize,
    pub bytes: u64,
    pub ple_files: usize,
    pub elapsed: Duration,
}

impl std::fmt::Display for RepackStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        write!(
            f,
            "재팩 완료: 전문가 투영 {} · 출력 텐서 {} · 데이터 {:.2} GiB · PLE 파일 복사 {} · {:.1}s",
            self.experts,
            self.tensors,
            gib(self.bytes),
            self.ple_files,
            self.elapsed.as_secs_f64()
        )
    }
}

pub fn repack(src_dir: &Path, dst_dir: &Path) -> Result<RepackStats, String> {
    repack_inner(src_dir, dst_dir, 48 * 512 * 3)
}

fn repack_inner(
    src_dir: &Path,
    dst_dir: &Path,
    expect_experts: usize,
) -> Result<RepackStats, String> {
    let t0 = Instant::now();
    let ar = StArchive::open(src_dir).map_err(|e| format!("소스: {e}"))?;
    let mmaps = Arc::new(mmap_all(&ar)?);

    let mut rebuild: Vec<(usize, String)> = Vec::new();
    let mut filecopy: Vec<(usize, String)> = Vec::new();
    for (i, p) in ar.shard_paths().iter().enumerate() {
        let fname = p
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or("샤드 파일명 UTF-8 아님")?
            .to_string();
        if is_model_shard(&fname) {
            rebuild.push((i, fname));
        } else {
            filecopy.push((i, fname));
        }
    }
    if rebuild.is_empty() {
        return Err("model-* 샤드 부재 — FN 배포 구조 아님".into());
    }

    let mut plans = Vec::new();
    let mut experts = 0usize;
    for (out_i, (shard, _src_name)) in rebuild.iter().enumerate() {
        let mut names: Vec<&String> = ar
            .entries()
            .iter()
            .filter(|(_, e)| e.shard == *shard)
            .map(|(n, _)| n)
            .collect();
        names.sort_unstable();
        let mut outs = Vec::new();
        for name in names {
            if !name_safe(name) {
                return Err(format!("텐서 이름 JSON 불안전: {name}"));
            }
            if let Some(base) = expert_base(name) {
                let (n, kw, kg) = check_triple(&ar, base)?;
                outs.push(Out::Repack {
                    base: base.to_string(),
                    n,
                    kw,
                    kg,
                });
                experts += 1;
            } else if name.contains(".mlp.experts.")
                && (name.ends_with(".qzeros") || name.ends_with(".scales"))
            {
            } else {
                outs.push(Out::Copy {
                    name: name.clone(),
                    nbytes: ar.entry(name).expect("순회 중 엔트리").nbytes(),
                });
            }
        }
        plans.push((
            format!("model-{:05}-of-{:05}.safetensors", out_i + 1, rebuild.len()),
            *shard,
            outs,
        ));
    }
    if experts != expect_experts {
        return Err(format!(
            "전문가 투영 {experts} ≠ {expect_experts}(48×512×3)"
        ));
    }

    std::fs::create_dir_all(dst_dir).map_err(|e| format!("출력 디렉터리: {e}"))?;
    for (out_name, _, _) in &plans {
        let p = dst_dir.join(out_name);
        if p.exists() {
            return Err(format!("{} 이미 존재 — 삭제 후 재시도", p.display()));
        }
    }
    for (_, fname) in &filecopy {
        if dst_dir.join(fname).exists() {
            return Err(format!("{} 이미 존재 — 삭제 후 재시도", fname));
        }
    }

    let copy_bytes: u64 = filecopy
        .iter()
        .filter_map(|(i, _)| ar.shard_paths().get(*i).map(|p| p.metadata().ok()))
        .flatten()
        .map(|m| m.len())
        .sum();
    let total_out: u64 = plans
        .iter()
        .map(|(_, _, outs)| outs.iter().map(Out::nbytes).sum::<u64>())
        .sum::<u64>()
        + copy_bytes;

    eprintln!(
        "[fn-repack] 샤드 {} 재구성 + 파일 복사 {} · 전문가 투영 {} · 출력 예정 {:.2} GiB",
        plans.len(),
        filecopy.len(),
        experts,
        total_out as f64 / (1u64 << 30) as f64
    );

    let _ = ();
    let mut wm = String::from("{\"metadata\":{\"total_size\":");
    wm.push_str(&total_out.to_string());
    wm.push_str("},\"weight_map\":{");
    let mut first = true;
    let mut index_count = 0usize;
    for (out_name, _, outs) in &plans {
        for o in outs {
            for (nm, _dt, _sh, _nb) in o.names() {
                if !first {
                    wm.push(',');
                }
                first = false;
                wm.push_str(&format!("\"{nm}\":\"{out_name}\""));
                index_count += 1;
            }
        }
    }
    let mut copy_names: Vec<(String, String)> = Vec::new();
    for (shard, fname) in &filecopy {
        for (name, e) in ar.entries() {
            if e.shard == *shard {
                copy_names.push((name.clone(), fname.clone()));
            }
        }
    }
    copy_names.sort_unstable();
    for (nm, fname) in &copy_names {
        if !first {
            wm.push(',');
        }
        first = false;
        wm.push_str(&format!("\"{nm}\":\"{fname}\""));
        index_count += 1;
    }
    wm.push_str("}}");
    let expect_total = ar.entries().len();
    if index_count != expect_total {
        return Err(format!(
            "출력 인덱스 {index_count} ≠ 입력 텐서 {expect_total} — 트리플 3→3 불변 위반"
        ));
    }

    let n_workers = plans.len() + filecopy.len();
    let ple_files = filecopy.len();
    let bytes_done = Arc::new(AtomicU64::new(0));
    let ar = Arc::new(ar);
    let (tx, rx) = std::sync::mpsc::channel::<(String, Result<u64, String>)>();
    let mut done = 0usize;
    let mut failed: Vec<String> = Vec::new();
    std::thread::scope(|s| {
        for (out_name, _shard, outs) in plans {
            let tx = tx.clone();
            let ar = Arc::clone(&ar);
            let mmaps = Arc::clone(&mmaps);
            let bytes_done = Arc::clone(&bytes_done);
            let dst = dst_dir.to_path_buf();
            s.spawn(move || {
                let r = write_shard(&ar, &mmaps, &dst, &out_name, &outs, &bytes_done);
                let _ = tx.send((out_name.clone(), r));
            });
        }
        for (shard, fname) in filecopy {
            let tx = tx.clone();
            let src_p = ar.shard_paths()[shard].to_path_buf();
            let dst = dst_dir.join(&fname);
            let bytes_done = Arc::clone(&bytes_done);
            s.spawn(move || {
                let r = std::fs::copy(&src_p, &dst).map_err(|e| format!("{fname}: {e}"));
                if let Ok(n) = r {
                    bytes_done.fetch_add(n, Ordering::Relaxed);
                }
                let _ = tx.send((fname.clone(), r));
            });
        }
        drop(tx);
        let t_start = Instant::now();
        loop {
            match rx.recv_timeout(Duration::from_secs(3)) {
                Ok((name, Ok(_nb))) => {
                    done += 1;
                    let el = t_start.elapsed().as_secs_f64();
                    let cur = bytes_done.load(Ordering::Relaxed);
                    eprintln!(
                        "[fn-repack] {done}/{n_workers} {name} 완료 — 누적 {:.2}/{:.2} GiB ({:.2} GiB/s)",
                        cur as f64 / (1u64 << 30) as f64,
                        total_out as f64 / (1u64 << 30) as f64,
                        cur as f64 / (1u64 << 30) as f64 / el.max(1e-9)
                    );
                }
                Ok((name, Err(e))) => {
                    done += 1;
                    failed.push(format!("{name}: {e}"));
                    eprintln!("[fn-repack] {name} 실패: {e}");
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    let cur = bytes_done.load(Ordering::Relaxed);
                    eprintln!(
                        "[fn-repack] 진행 {done}/{n_workers} · {:.2}/{:.2} GiB",
                        cur as f64 / (1u64 << 30) as f64,
                        total_out as f64 / (1u64 << 30) as f64
                    );
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        if done != n_workers {
            eprintln!("[fn-repack] 워커 종료 {done}/{n_workers}");
        }
    });
    if !failed.is_empty() {
        return Err(format!(
            "워커 실패 {}건 — 첫째: {}",
            failed.len(),
            failed[0]
        ));
    }
    std::fs::write(dst_dir.join("model.safetensors.index.json"), &wm)
        .map_err(|e| format!("index.json: {e}"))?;

    write_config(src_dir, dst_dir)?;
    copy_aux(src_dir, dst_dir)?;

    let out_ar = StArchive::open(dst_dir).map_err(|e| format!("자체검증: {e}"))?;
    if out_ar.entries().len() != expect_total {
        return Err(format!(
            "자체검증: 출력 엔트리 {} ≠ {}",
            out_ar.entries().len(),
            expect_total
        ));
    }
    let store = W4a16Model::open(dst_dir).map_err(|e| format!("자체검증(W4a16 open): {e}"))?;
    if store.n_lins() != expect_experts || store.group() != 32 {
        return Err(format!(
            "자체검증: lins {} (기대 {expect_experts}) · g{}",
            store.n_lins(),
            store.group()
        ));
    }
    drop(store);
    drop(out_ar);

    Ok(RepackStats {
        experts,
        tensors: index_count,
        bytes: total_out,
        ple_files,
        elapsed: t0.elapsed(),
    })
}

fn write_shard(
    ar: &StArchive,
    mmaps: &[memmap2::Mmap],
    dst_dir: &Path,
    out_name: &str,
    outs: &[Out],
    bytes_done: &AtomicU64,
) -> Result<u64, String> {
    let t0 = Instant::now();
    let mut hdr = String::from("{\"__metadata__\":{\"format\":\"pt\"}");
    let mut off = 0u64;
    for o in outs {
        match o {
            Out::Repack { .. } => {
                for (nm, dt, sh, nb) in o.names() {
                    let shape: Vec<String> = sh.iter().map(|v| v.to_string()).collect();
                    hdr.push_str(&format!(
                        ",\"{nm}\":{{\"dtype\":\"{dt}\",\"shape\":[{}],\"data_offsets\":[{off},{}]}}",
                        shape.join(","),
                        off + nb
                    ));
                    off += nb;
                }
            }
            Out::Copy { name, .. } => {
                let e = ar.entry(name).ok_or_else(|| format!("{name}: 부재"))?;
                let shape: Vec<String> = e.shape.iter().map(|v| v.to_string()).collect();
                hdr.push_str(&format!(
                    ",\"{name}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{off},{}]}}",
                    dtype_name(e.dtype),
                    shape.join(","),
                    off + e.nbytes()
                ));
                off += e.nbytes();
            }
        }
    }
    hdr.push('}');

    let path = dst_dir.join(out_name);
    let f = std::fs::File::create(&path).map_err(|e| format!("{path:?}: {e}"))?;
    let mut w = BufWriter::with_capacity(8 << 20, f);
    w.write_all(&(hdr.len() as u64).to_le_bytes())
        .map_err(|e| format!("{out_name} 헤더길이: {e}"))?;
    let hb = hdr.as_bytes();
    w.write_all(hb)
        .map_err(|e| format!("{out_name} 헤더: {e}"))?;
    let mut written = 0u64;
    for o in outs {
        match o {
            Out::Repack { base, n, kw, kg } => {
                repack_triple(ar, mmaps, base, *n, *kw, *kg, &mut w)?;
                written += o.nbytes();
            }
            Out::Copy { name, .. } => {
                let b = tbytes(ar, mmaps, name)?;
                w.write_all(b).map_err(|e| format!("{name}: {e}"))?;
                written += b.len() as u64;
            }
        }
        bytes_done.store(written + (hb.len() as u64 + 8), Ordering::Relaxed);
    }
    bytes_done.fetch_add(hdr.len() as u64 + 8, Ordering::Relaxed);
    let f = w
        .into_inner()
        .map_err(|e| format!("{out_name} flush: {e}"))?;
    f.sync_all().map_err(|e| format!("{out_name} fsync: {e}"))?;
    eprintln!(
        "[fn-repack] {out_name} 기록 {:.2} GiB — {:.1}s",
        written as f64 / (1u64 << 30) as f64,
        t0.elapsed().as_secs_f64()
    );
    Ok(written)
}

fn dtype_name(d: StDtype) -> &'static str {
    match d {
        StDtype::F16 => "F16",
        StDtype::Bf16 => "BF16",
        StDtype::F32 => "F32",
        StDtype::I16 => "I16",
        StDtype::I32 => "I32",
        StDtype::I64 => "I64",
        StDtype::U8 => "U8",
        StDtype::F8E4M3 => "F8_E4M3",
    }
}

fn write_config(src_dir: &Path, dst_dir: &Path) -> Result<(), String> {
    let cfg = std::fs::read_to_string(src_dir.join("config.json"))
        .map_err(|e| format!("config.json: {e}"))?;
    const KEY: &str = "\"quantization_config\"";
    let i = cfg
        .find(KEY)
        .ok_or_else(|| "config.json: quantization_config 부재".to_string())?;
    let rest = &cfg[i + KEY.len()..];
    let vstart = rest
        .find('{')
        .ok_or_else(|| "quantization_config 값이 객체 아님".to_string())?;
    let b = rest.as_bytes();
    let mut depth = 0usize;
    let mut in_str = false;
    let mut esc = false;
    let mut vend = 0usize;
    for (j, &c) in b.iter().enumerate().skip(vstart) {
        if in_str {
            if esc {
                esc = false;
            } else if c == b'\\' {
                esc = true;
            } else if c == b'"' {
                in_str = false;
            }
        } else if c == b'"' {
            in_str = true;
        } else if c == b'{' {
            depth += 1;
        } else if c == b'}' {
            depth -= 1;
            if depth == 0 {
                vend = j + 1;
                break;
            }
        }
    }
    if vend == 0 {
        return Err("quantization_config 객체 미종결".into());
    }
    let abs_start = i + KEY.len() + vstart;
    let abs_end = i + KEY.len() + vend;
    let mut out = String::with_capacity(cfg.len() + 2048);
    out.push_str(&cfg[..abs_start]);
    out.push_str(CT_QUANT_CONFIG);
    out.push_str(&cfg[abs_end..]);
    crate::json::Json::parse(&out).map_err(|e| format!("치환 후 config 파싱: {e}"))?;
    std::fs::write(dst_dir.join("config.json"), &out)
        .map_err(|e| format!("config.json 쓰기: {e}"))?;
    Ok(())
}

const CT_QUANT_CONFIG: &str = r#"{
    "config_groups": {
      "group_0": {
        "format": "pack-quantized",
        "input_activations": null,
        "output_activations": null,
        "targets": ["Linear"],
        "weights": {
          "actorder": null,
          "block_structure": null,
          "dynamic": false,
          "group_size": 32,
          "num_bits": 4,
          "observer": "mse",
          "observer_kwargs": {},
          "scale_dtype": null,
          "strategy": "group",
          "symmetric": true,
          "type": "int",
          "zp_dtype": null
        }
      }
    },
    "format": "pack-quantized",
    "global_compression_ratio": null,
    "ignore": [
      "lm_head",
      ".*embed_tokens.*",
      ".*linear_attn.*",
      ".*self_attn.*",
      ".*indexer.*",
      ".*ple.*",
      ".*hyper_connection.*",
      ".*mlp\\.gate.*",
      ".*shared_expert.*",
      "model.visual.*"
    ],
    "quant_method": "compressed-tensors"
  }"#;

fn copy_aux(src_dir: &Path, dst_dir: &Path) -> Result<(), String> {
    const FILES: &[&str] = &[
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "hybrid_sources.json",
        "SOURCE.txt",
        "LICENSE",
        "README.md",
    ];
    for f in FILES {
        let p = src_dir.join(f);
        if p.is_file() {
            std::fs::copy(&p, dst_dir.join(f)).map_err(|e| format!("{f}: {e}"))?;
        }
    }
    for dir in ["docs", "runtime"] {
        let p = src_dir.join(dir);
        if !p.is_dir() {
            continue;
        }
        copy_tree(&p, &dst_dir.join(dir), &["mtp-int4-g32"])?;
    }
    Ok(())
}

fn copy_tree(src: &Path, dst: &Path, skip_top: &[&str]) -> Result<(), String> {
    std::fs::create_dir_all(dst).map_err(|e| format!("{dst:?}: {e}"))?;
    for e in std::fs::read_dir(src).map_err(|e| format!("{src:?}: {e}"))? {
        let e = e.map_err(|e| e.to_string())?;
        let name = e.file_name();
        let name = name.to_string_lossy();
        if skip_top.contains(&name.as_ref()) {
            continue;
        }
        let sp = e.path();
        let dp = dst.join(name.as_ref());
        if sp.is_dir() {
            copy_tree(&sp, &dp, &[])?;
        } else {
            std::fs::copy(&sp, &dp).map_err(|e| format!("{sp:?}: {e}"))?;
        }
    }
    Ok(())
}

pub fn verify_sample(
    src_dir: &Path,
    dst_dir: &Path,
    layers: &[u64],
    experts: &[u64],
) -> Result<String, String> {
    let gq = engine::Model::load(src_dir).map_err(|e| format!("GPTQ 원본: {e}"))?;
    let ct = W4a16Model::open(dst_dir).map_err(|e| format!("CT 출력: {e}"))?;
    let g = ct.group();
    if g != 32 {
        return Err(format!("CT 그룹 {g} ≠ 32(g32 복제 계약)"));
    }
    let mut rows = 0u64;
    let mut cells = 0u64;
    let mut words = 0u64;
    for &il in layers {
        for &e in experts {
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                let base = format!("model.language_model.layers.{il}.mlp.experts.{e}.{proj}");
                let (pk, sc, n, k) = ct
                    .expert_at(&base)
                    .ok_or_else(|| format!("{base}: CT 트리플 부재"))?;
                let qw = gq
                    .slice(&format!("{base}.qweight"))
                    .map_err(|e| format!("{base}: {e}"))?;
                let gs = gq
                    .slice(&format!("{base}.scales"))
                    .map_err(|e| format!("{base}: {e}"))?;
                let (kw, kg) = (k / 8, k / g);
                for r in 0..n {
                    let want = gq
                        .expert_row(&base, r)
                        .map_err(|er| format!("{base}: {er}"))?;
                    for j in 0..k {
                        let w = u32::from_le_bytes(
                            pk[4 * (r * kw + j / 8)..][..4].try_into().expect("4B"),
                        );
                        let q = ((w >> (4 * (j % 8))) & 0xF) as i32;
                        let sb = 2 * (r * kg + j / g);
                        let s = engine::half_f32(u16::from_le_bytes([sc[sb], sc[sb + 1]]));
                        let v = (q - 8) as f32 * s;
                        if v.to_bits() != want[j].to_bits() {
                            return Err(format!("{base} r={r} j={j}: CT {v:E} ≠ GPTQ {}", want[j]));
                        }
                        cells += 1;
                    }
                    rows += 1;
                }
                for r in 0..n {
                    for jw in 0..kw {
                        let a = u32::from_le_bytes(
                            pk[4 * (r * kw + jw)..][..4].try_into().expect("4B"),
                        );
                        let bq =
                            u32::from_le_bytes(qw[4 * (jw * n + r)..][..4].try_into().expect("4B"));
                        if a != bq {
                            return Err(format!(
                                "{base} packed[{r},{jw}]={a:#010x} ≠ qweight[{jw},{r}]={bq:#010x}"
                            ));
                        }
                        words += 1;
                    }
                    let rep_n = G / 32;
                    for gi128 in 0..(kg / rep_n) {
                        let bq = [gs[2 * (gi128 * n + r)], gs[2 * (gi128 * n + r) + 1]];
                        for rep in 0..rep_n {
                            let o32 = 2 * (r * kg + gi128 * rep_n + rep);
                            let a = [sc[o32], sc[o32 + 1]];
                            if a != bq {
                                return Err(format!(
                                    "{base} scale[{r},{}](g32 복제) 비트 불일치",
                                    gi128 * rep_n + rep
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(format!(
        "비트일치 통과: {rows}행 · 원소 {cells} · packed 워드 {words} · 대상 {}투영(층 {} × 전문가 {})",
        layers.len() * experts.len() * 3,
        layers.len(),
        experts.len()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_classify() {
        assert!(is_model_shard("model-00001-of-00017.safetensors"));
        assert!(is_model_shard("model-00002-of-00002.safetensors"));
        assert!(!is_model_shard("model-plefp8-00000.safetensors"));
        assert!(!is_model_shard("mtp.safetensors"));
        assert!(!is_model_shard("model.safetensors"));
    }

    #[test]
    fn name_safety() {
        assert!(name_safe(
            "model.language_model.layers.0.mlp.experts.1.gate_proj.weight_packed"
        ));
        assert!(!name_safe("bad\"name"));
        assert!(!name_safe("bad\\name"));
    }

    #[test]
    fn repack_synthetic_roundtrip() {
        let tmp = std::env::temp_dir().join(format!("fn-repack-test-{}", std::process::id()));
        let src = tmp.join("src");
        let dst = tmp.join("dst");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();

        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let write_st = |dir: &Path, file: &str, ents: Vec<(String, &str, Vec<u64>, Vec<u8>)>| {
            let mut hdr = String::from("{");
            let mut off = 0u64;
            for (nm, dt, sh, data) in &ents {
                let shape: Vec<String> = sh.iter().map(|v| v.to_string()).collect();
                hdr.push_str(&format!(
                    ",\"{nm}\":{{\"dtype\":\"{dt}\",\"shape\":[{}],\"data_offsets\":[{off},{}]}}",
                    shape.join(","),
                    off + data.len() as u64
                ));
                off += data.len() as u64;
            }
            let hdr = hdr.replacen("{,", "{", 1) + "}";
            let mut buf = Vec::new();
            buf.extend_from_slice(&(hdr.len() as u64).to_le_bytes());
            buf.extend_from_slice(hdr.as_bytes());
            for (_, _, _, data) in &ents {
                buf.extend_from_slice(data);
            }
            std::fs::write(dir.join(file), &buf).unwrap();
        };

        let triples = [
            (
                "model.language_model.layers.0.mlp.experts.0.gate_proj",
                640,
                2560,
            ),
            (
                "model.language_model.layers.0.mlp.experts.0.down_proj",
                2560,
                640,
            ),
        ];
        let mut model_ents: Vec<(String, &str, Vec<u64>, Vec<u8>)> = Vec::new();
        let mut want_words: Vec<(String, Vec<u32>, usize, usize)> = Vec::new();
        let mut want_scales: Vec<(String, Vec<u16>, usize, usize)> = Vec::new();
        for (base, n, k) in triples {
            let (kw, kg) = (k / 8, k / 128);
            let qw: Vec<u8> = (0..kw * n)
                .flat_map(|_| (rnd() as u32).to_le_bytes())
                .collect();
            let qz: Vec<u8> = vec![0x77; kg * (n / 8) * 4];
            let sc: Vec<u8> = (0..kg * n)
                .flat_map(|_| (((rnd() >> 16) as u16) | 0x3800).to_le_bytes())
                .collect();
            model_ents.push((
                format!("{base}.qweight"),
                "I32",
                vec![kw as u64, n as u64],
                qw.clone(),
            ));
            model_ents.push((
                format!("{base}.qzeros"),
                "I32",
                vec![kg as u64, (n / 8) as u64],
                qz,
            ));
            model_ents.push((
                format!("{base}.scales"),
                "F16",
                vec![kg as u64, n as u64],
                sc.clone(),
            ));
            let wv: Vec<u32> = qw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c))
                .collect();
            want_words.push((base.to_string(), wv, n, kw));
            let sv: Vec<u16> = sc
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_le_bytes(*c))
                .collect();
            want_scales.push((base.to_string(), sv, n, kg));
        }
        let dense = (0..1024).flat_map(|i| (i as u16).to_le_bytes()).collect();
        model_ents.push((
            "model.language_model.layers.0.mlp.gate.weight".to_string(),
            "BF16",
            vec![1024],
            dense,
        ));
        write_st(&src, "model-00001-of-00002.safetensors", model_ents);
        write_st(&src, "model-00002-of-00002.safetensors", vec![]);
        let ple: Vec<u8> = (0..64).flat_map(|i| (i as f32).to_le_bytes()).collect();
        write_st(
            &src,
            "model-plefp8-00000.safetensors",
            vec![(
                "model.language_model.layers.1.ple.ple_embedding.weight_scale".to_string(),
                "F32",
                vec![64],
                ple.clone(),
            )],
        );
        let mut wm = String::from("{\"weight_map\":{");
        let names0 = [
            "model.language_model.layers.0.mlp.experts.0.gate_proj",
            "model.language_model.layers.0.mlp.experts.0.down_proj",
            "model.language_model.layers.0.mlp.gate.weight",
        ];
        let mut items = Vec::new();
        for b in names0 {
            for s in ["qweight", "qzeros", "scales"] {
                if b.ends_with("mlp.gate.weight") {
                    continue;
                }
                items.push(format!("\"{b}.{s}\":\"model-00001-of-00002.safetensors\""));
            }
        }
        items.push("\"model.language_model.layers.0.mlp.gate.weight\":\"model-00001-of-00002.safetensors\"".into());
        items.push("\"model.language_model.layers.1.ple.ple_embedding.weight_scale\":\"model-plefp8-00000.safetensors\"".into());
        wm.push_str(&items.join(","));
        wm.push_str("}}");
        std::fs::write(src.join("model.safetensors.index.json"), &wm).unwrap();
        std::fs::write(
            src.join("config.json"),
            "{\"architectures\":[\"Qwen4ExpForConditionalGeneration\"],\"quantization_config\":{\"quant_method\":\"auto_round:auto_gptq\",\"bits\":4}}",
        )
        .unwrap();

        let stats = repack_inner(&src, &dst, 2).expect("재팩 실패");
        assert_eq!(stats.experts, 2);
        assert_eq!(stats.tensors, 8);
        assert_eq!(stats.ple_files, 1);

        let out = StArchive::open(&dst).expect("출력 개방 실패");
        for (base, wv, n, kw) in &want_words {
            let e = out.entry(&format!("{base}.weight_packed")).expect("packed");
            assert_eq!(e.shape, vec![*n as u64, *kw as u64]);
            let raw = out.read(&format!("{base}.weight_packed")).unwrap();
            for r in 0..*n {
                for jw in 0..*kw {
                    let got = u32::from_le_bytes(raw[4 * (r * kw + jw)..][..4].try_into().unwrap());
                    assert_eq!(got, wv[jw * n + r], "{base} [{r},{jw}]");
                }
            }
            let (_, sv, _, kg) = want_scales.iter().find(|(b, ..)| b == base).unwrap();
            let sraw = out.read(&format!("{base}.weight_scale")).unwrap();
            let kg32 = kg * 4;
            for r in 0..*n {
                for gi in 0..*kg {
                    for rep in 0..4 {
                        let got = u16::from_le_bytes(
                            sraw[2 * (r * kg32 + gi * 4 + rep)..][..2]
                                .try_into()
                                .unwrap(),
                        );
                        assert_eq!(got, sv[gi * n + r], "{base} scale [{r},{}]", gi * 4 + rep);
                    }
                }
            }
            let shp = out.read(&format!("{base}.weight_shape")).unwrap();
            let nk = (
                i64::from_le_bytes(shp[0..8].try_into().unwrap()),
                i64::from_le_bytes(shp[8..16].try_into().unwrap()),
            );
            let k = kw * 8;
            assert_eq!(nk, (*n as i64, k as i64));
        }
        assert_eq!(
            out.read("model.language_model.layers.0.mlp.gate.weight")
                .unwrap()
                .len(),
            2048
        );
        assert_eq!(
            out.read("model.language_model.layers.1.ple.ple_embedding.weight_scale")
                .unwrap(),
            ple
        );
        let store = W4a16Model::open(&dst).expect("W4a16 open 실패");
        assert_eq!(store.n_lins(), 2);
        assert_eq!(store.group(), 32);
        let cfg = std::fs::read_to_string(dst.join("config.json")).unwrap();
        assert!(cfg.contains("compressed-tensors"));
        assert!(cfg.contains("Qwen4ExpForConditionalGeneration"));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn repack_rejects_nonconstant_qzeros() {
        let tmp = std::env::temp_dir().join(format!("fn-repack-neg-{}", std::process::id()));
        let src = tmp.join("src");
        let dst = tmp.join("dst");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();
        let (n, k) = (640usize, 2560usize);
        let (kw, kg) = (k / 8, k / 128);
        let qw = vec![0u8; kw * n * 4];
        let mut qz = vec![0x77u8; kg * (n / 8) * 4];
        qz[0] = 0x76;
        let sc = vec![0u8; kg * n * 2];
        let ents = vec![
            (
                "model.language_model.layers.0.mlp.experts.0.gate_proj.qweight",
                "I32",
                vec![kw as u64, n as u64],
                qw,
            ),
            (
                "model.language_model.layers.0.mlp.experts.0.gate_proj.qzeros",
                "I32",
                vec![kg as u64, (n / 8) as u64],
                qz,
            ),
            (
                "model.language_model.layers.0.mlp.experts.0.gate_proj.scales",
                "F16",
                vec![kg as u64, n as u64],
                sc,
            ),
        ];
        let mut hdr = String::from("{");
        let mut off = 0u64;
        for (nm, dt, sh, data) in &ents {
            let shape: Vec<String> = sh.iter().map(|v| v.to_string()).collect();
            hdr.push_str(&format!(
                ",\"{nm}\":{{\"dtype\":\"{dt}\",\"shape\":[{}],\"data_offsets\":[{off},{}]}}",
                shape.join(","),
                off + data.len() as u64
            ));
            off += data.len() as u64;
        }
        let hdr = hdr.replacen("{,", "{", 1) + "}";
        let mut buf = (hdr.len() as u64).to_le_bytes().to_vec();
        buf.extend_from_slice(hdr.as_bytes());
        for (_, _, _, data) in &ents {
            buf.extend_from_slice(data);
        }
        std::fs::write(src.join("model-00001-of-00001.safetensors"), &buf).unwrap();
        std::fs::write(
            src.join("model.safetensors.index.json"),
            "{\"weight_map\":{\"model.language_model.layers.0.mlp.experts.0.gate_proj.qweight\":\"model-00001-of-00001.safetensors\",\"model.language_model.layers.0.mlp.experts.0.gate_proj.qzeros\":\"model-00001-of-00001.safetensors\",\"model.language_language.layers.0.mlp.experts.0.gate_proj.scales\":\"model-00001-of-00001.safetensors\"}}",
        )
        .unwrap();
        std::fs::write(
            src.join("model.safetensors.index.json"),
            "{\"weight_map\":{\"model.language_model.layers.0.mlp.experts.0.gate_proj.qweight\":\"model-00001-of-00001.safetensors\",\"model.language_model.layers.0.mlp.experts.0.gate_proj.qzeros\":\"model-00001-of-00001.safetensors\",\"model.language_model.layers.0.mlp.experts.0.gate_proj.scales\":\"model-00001-of-00001.safetensors\"}}",
        )
        .unwrap();
        std::fs::write(src.join("config.json"), "{\"quantization_config\":{}}").unwrap();
        let err = repack_inner(&src, &dst, 1).expect_err("비상수 qzeros인데 성공");
        assert!(err.contains("비상수"), "오류 메시지: {err}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn fn_repack_equiv_real_model() {
        let (Some(src), Some(dst)) = (
            llm170_diag::flag::val("LLM170_FN_MODEL"),
            llm170_diag::flag::val("LLM170_FN_MODEL_CT"),
        ) else {
            return;
        };
        let rep = verify_sample(
            std::path::Path::new(&src),
            std::path::Path::new(&dst),
            &[0, 23, 47],
            &[0, 7, 511],
        )
        .expect("동치 검증 실패");
        eprintln!("{rep}");
        assert!(rep.contains("비트일치 통과"));
    }
}
