//! [FN D1→재팩 2026-10-20] GPTQ(auto_round) → CT(compressed-tensors) 무손실 재패커.
//!
//! D1 판정(커밋 39787db0 — engine.rs fn_qzeros_sym_scan 전수 스캔): 전문가
//! qzeros 전 니블 = 7 상수 → zero-point = 7+1(vLLM zero_offset) = **8** =
//! CT sym zp=8와 동일. 따라서 GPTQ→CT 재팩은 **무손실 순수 니블 전치**다.
//!
//! [비트계약 — engine.rs expert_triple 실측]
//! - GPTQ qweight I32[kw, n] (kw=k/8): 워드 (jw, r)이 입력 j=jw·8..jw·8+7의
//!   니블 8개(LSB-first)를 행 r로 담는다.
//! - CT weight_packed I32[n, kw] (lane.rs §3.6 lsb-first): 워드 (r, jw)이 같은
//!   니블 8개를 같은 위치로 담는다 → **워드 값 자체가 불변**, 순수 u32 전치.
//!   packed[r·kw + jw] = qweight[jw·n + r].
//! - scales F16[kg, n] → weight_scale F16[n, kg] — **비트 그대로** 전치.
//! - qzeros I32[kg, n/8]은 폐기(재팩 시 전수 상수 0x77777777 검증 후 버림).
//! - weight_shape I64[2] = (n, k) 신규 생성.
//!
//! [대상] 48층 × 512전문가 × {gate,up,down} = 73,728 투영만 변환. 형상은
//! 실측 쌍 (n,k) ∈ {(640,2560),(2560,640)}로 고정(예외 = 즉시 거부).
//! gate/up k=2560·n=640 / down k=640·n=2560, g=128.
//!
//! [출력 구조] 소스 model-* 샤드 경계를 그대로 미러(샤드 재구성 — 전문가
//! 트리플은 소스에서 샤드 내 완결 실측)하되 연속 재번호(00001-of-000NN).
//! PLE 샤드(model-plefp8-*)는 파일 단위 복사. 나머지 텐서(BF16 민감층·
//! embed·head 등)는 원시 바이트 그대로 재기록. config.json의
//! quantization_config만 CT(pack-quantized int4 sym g128)로 치환.
//!
//! [성능] mmap 읽기 + 스트리밍 쓰기(샤드당 워커 1 — 쓰기 순차 유지),
//! 진행 로그. GPU 무사용.

use super::engine;
use crate::st::{StArchive, StDtype, StEntry};
use crate::w4a16::W4a16Model;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// gptq 그룹 크기 — engine.rs GPTQ_GROUP과 동일 계약(단일 상산 출처는
/// 엔진 쪽이지만 재패커는 core 내부라 여기 재정의 없이 상수 참조).
const G: usize = 128;
/// qzeros 정상 워드 — 전 니블 7(zero=8). D1 판정값.
const QZ_WORD: u32 = 0x7777_7777;
/// 전문가 투영 허용 형상 (n, k) 실측 쌍.
const SHAPES: [(usize, usize); 2] = [(640, 2560), (2560, 640)];

/// 전문가 투영 이름(.qweight 등 접미사 전) 판별.
fn expert_base(name: &str) -> Option<&str> {
    name.strip_suffix(".qweight")
        .filter(|b| b.contains(".mlp.experts."))
}

/// 텐서 이름 JSON 안전성 — safetensors 이름은 [A-Za-z0-9_.-]만(헤더 직렬화
/// 무이스케이프 계약). 백슬래시·따옴표 등이 오면 즉시 거부.
fn name_safe(name: &str) -> bool {
    name.chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'))
}

/// 재구성 대상 model-* 샤드 파일명 판별(^model-\d+-of-\d+\.safetensors$).
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

/// 샤드 하나의 출력 계획 — 항목은 이름 정렬순(결정론적 출력).
enum Out {
    /// gptq 트리플 → CT 트리플(weight_packed · weight_scale · weight_shape).
    Repack {
        base: String,
        n: usize,
        kw: usize,
        kg: usize,
    },
    /// 원시 바이트 그대로 재기록.
    Copy { name: String, nbytes: u64 },
}

impl Out {
    fn nbytes(&self) -> u64 {
        match self {
            // packed + scale(g32 복제 ×4) + shape
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
                // dtype/shape은 헤더 기록 시점 엔트리에서 조립 — 여기(인덱스
                // 용)이름만 유효. 빈 목록을 돌려주면 index.json에서 복사
                // 텐서가 누락된다(합성 테스트 실측).
                vec![(name.clone(), "", Vec::new(), *nbytes)]
            }
        }
    }
}

/// gptq 트리플 형상 검증(엔진 expert_triple 미러 + 실측 형상 고정) → (n, kw, kg).
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

/// 아카이브 전체 샤드 mmap(엔진 Model::load 동일 계약).
fn mmap_all(ar: &StArchive) -> Result<Vec<memmap2::Mmap>, String> {
    let mut v = Vec::new();
    for p in ar.shard_paths() {
        let f = std::fs::File::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
        // SAFETY: 읽기 전용 매핑 — 수정하지 않는다(엔진 mmap 동일 계약).
        v.push(unsafe { memmap2::Mmap::map(&f).map_err(|e| e.to_string())? });
    }
    Ok(v)
}

/// mmap 상 텐서 원시 슬라이스.
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

/// 투영 1개 변환: qzeros 전수 검증 → packed 워드 전치 → scale 비트 전치 → shape.
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
    // qzeros 전수 상수 검증 — 무손실(sym zp=8) 계약의 관문. 하나라도 어긋나면
    // 이 재패커의 전제가 무너진다(명시 거부 — GPTQ-asym 경로 필요).
    for (i, c) in qz.as_chunks::<4>().0.iter().enumerate() {
        let w = u32::from_le_bytes(*c);
        if w != QZ_WORD {
            return Err(format!(
                "{base}.qzeros[{i}] = {w:#010x} — 비상수 zero-point(D1 계약 위반)"
            ));
        }
    }
    // packed[n, kw] u32 — qweight[kw, n]의 순수 워드 전치(니블 위치 불변).
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
    // weight_scale[n, kg32] F16 — scales[kg, n] 전치 + **g128→g32 스케일 복제**.
    // [FN M1 2026-10-20] FN 전문가(g128)를 기존 g32 마린 MoE 경로에 태우기
    // 위해 각 g128 스케일을 4개 g32 그룹에 복제 — 니블 불변·스케일 동값으로
    // 디큐트 값 비트동일(상위 그룹 스케일의 자식 그룹 재사용). kg32 = k/32.
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
    // weight_shape I64[2] = (n, k).
    let k = kw * 8;
    out.write_all(&(n as i64).to_le_bytes())
        .and_then(|_| out.write_all(&(k as i64).to_le_bytes()))
        .map_err(|e| format!("{base}.shape: {e}"))?;
    Ok(())
}

/// 재패커 실행 보고.
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

/// 전체 재팩 — src(FN gptq 원본) → dst(CT 트리플 새 디렉터리).
pub fn repack(src_dir: &Path, dst_dir: &Path) -> Result<RepackStats, String> {
    repack_inner(src_dir, dst_dir, 48 * 512 * 3)
}

/// 본체 — expect_experts = 이 모델 패밀리의 전문가 투영 수 계약(FN 실측
/// 73,728. 합성 테스트는 축소판을 넣는다).
fn repack_inner(
    src_dir: &Path,
    dst_dir: &Path,
    expect_experts: usize,
) -> Result<RepackStats, String> {
    let t0 = Instant::now();
    let ar = StArchive::open(src_dir).map_err(|e| format!("소스: {e}"))?;
    let mmaps = Arc::new(mmap_all(&ar)?);

    // 샤드 분류: model-* 재구성 / 나머지(plefp8 등) 파일 복사.
    let mut rebuild: Vec<(usize, String)> = Vec::new(); // (src shard, src 파일명)
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

    // 샤드별 출력 계획(이름 정렬순 — 결정론).
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
                // 트리플 동반 — qweight에서 이미 처리.
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

    // 출력 디렉터리 준비 — 이미 결과 조각이 있으면 거부(재시도 안전).
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

    // PLE 등 파일 복사분 바이트(진행률 분모).
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

    // index.json 본문·파생 수치는 스코프 전 계획 기준으로 선행 완성 —
    // plans/filecopy는 스코프 클로저로 이동하기 때문.
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
    // 파일 복사 샤드의 엔트리는 원본 이름·원본 샤드 파일명 그대로.
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

    // 워커: 재구성 샤드당 1 + 파일 복사당 1 — 순차 쓰기 유지 채널 수집.
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
        // 진행 로그 — 3초 간격 수신 타임아웃 활용.
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

    // config.json — quantization_config 값만 CT로 치환(나머지 원문 보존).
    write_config(src_dir, dst_dir)?;
    // 부속 파일(tokenizer·template·runtime — mtp 가중치 제외).
    copy_aux(src_dir, dst_dir)?;

    // 출력 구조 자체검증 — 헤더 재파싱 + W4a16Model 완전 개방(트리플·양자 계약).
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

/// 샤드 1개 기록: 헤더(크기 선행 계산) → 데이터 스트리밍 → fsync.
fn write_shard(
    ar: &StArchive,
    mmaps: &[memmap2::Mmap],
    dst_dir: &Path,
    out_name: &str,
    outs: &[Out],
    bytes_done: &AtomicU64,
) -> Result<u64, String> {
    let t0 = Instant::now();
    // 헤더 직렬화 — data_offsets는 크기 합으로 선행 확정.
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

/// dtype → safetensors 헤더 문자열(st.rs parse의 역매핑).
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

/// config.json 복사 + quantization_config 값 치환(brace 매칭 — 문자열 내부
/// 중괄호 무시, w4a16.rs extract_embedded_quant 동일 알고리즘 위치 반환판).
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

/// CT quantization_config 본문 — w4a16.rs check_quant 계약(compressed-tensors
/// ·pack-quantized·int4·sym·g128) 충족. ignore = FN 원본 extra_config의
/// BF16/FP8 제외 패밀리(전문가 라우팅 선형만 양자화 대상).
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

/// 부속 파일 복사 — tokenizer·템플릿·메타·runtime(mtp 가중치 제외: 본 과제
/// 대상 = 라우팅 전문가 73,728 텐서 한정, mtp-int4-g32는 별도 g32 GPTQ 팩).
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

/// 디렉터리 재귀 복사 — skip_top: 최상위 자식 이름 제외 목록.
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

/// 동치 검증 — 표본 전문가 전 행 CT 디큐트(w4a16 스토어 경로·lane.rs 계약)
/// vs GPTQ 직접 디큐트(engine.rs expert_row) **비트일치** 판정.
/// 함께: packed 워드 ↔ qweight 워드 전수 대조 + scale 비트 대조(니블 전치
/// 무손실의 직접 증명 — (q−8)·s 곱은 q=8/s=0에서 변형을 숨길 수 있어).
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
                    // CT 행 디큐트 — lane.rs dot_row_w4a16_lane_parts와 동일
                    // 니블 계약(lsb-first, zp=8)·engine.rs half_f32 공유.
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
                // 워드·스케일 비트 직접 대조(전수).
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
                    // g32 복제 검증 — 원본 g128 스케일이 4자식 그룹에 동값.
                    // (여기 kg = CT 스토어 그룹 개수 = k/32 — 원본은 kg/4개.)
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

    /// 샤드 판별 — model-* 숫자 샤드만 재구성, plefp8 등은 파일 복사.
    #[test]
    fn shard_classify() {
        assert!(is_model_shard("model-00001-of-00017.safetensors"));
        assert!(is_model_shard("model-00002-of-00002.safetensors"));
        assert!(!is_model_shard("model-plefp8-00000.safetensors"));
        assert!(!is_model_shard("mtp.safetensors"));
        assert!(!is_model_shard("model.safetensors"));
    }

    /// 이름 안전성 — 이스케이프 필요 문자 거부.
    #[test]
    fn name_safety() {
        assert!(name_safe(
            "model.language_model.layers.0.mlp.experts.1.gate_proj.weight_packed"
        ));
        assert!(!name_safe("bad\"name"));
        assert!(!name_safe("bad\\name"));
    }

    /// gptq → CT 트리플 합성 왕복 — 실측 형상(n=640,k=2560)으로 워커 직접
    /// 호출 없이 repack 전체 경로(헤더·index·config splice 포함) 검증.
    #[test]
    fn repack_synthetic_roundtrip() {
        let tmp = std::env::temp_dir().join(format!("fn-repack-test-{}", std::process::id()));
        let src = tmp.join("src");
        let dst = tmp.join("dst");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();

        // ── 합성 소스: 1층 × 2전문가 × 3투영 + 밀집 1 + ple풍 샤드 1.
        // LCG — 결정론적 유사랜덤 워드(니블 전 범위).
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
            // 첫 항목 앞 콤마 정리
            let hdr = hdr.replacen("{,", "{", 1) + "}";
            let mut buf = Vec::new();
            buf.extend_from_slice(&(hdr.len() as u64).to_le_bytes());
            buf.extend_from_slice(hdr.as_bytes());
            for (_, _, _, data) in &ents {
                buf.extend_from_slice(data);
            }
            std::fs::write(dir.join(file), &buf).unwrap();
        };

        // gate_proj(n=640,k=2560) 1개 + down_proj(n=2560,k=640) 1개 — 두 형상 모두.
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
        // 밀집 복사 대상 1개.
        let dense = (0..1024).flat_map(|i| (i as u16).to_le_bytes()).collect();
        model_ents.push((
            "model.language_model.layers.0.mlp.gate.weight".to_string(),
            "BF16",
            vec![1024],
            dense,
        ));
        write_st(&src, "model-00001-of-00002.safetensors", model_ents);
        // 두 번째 model 샤드(빈 재구성 경로) + ple풍 샤드(F32 1텐서).
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
        // index + config.
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

        // ── 실행.
        let stats = repack_inner(&src, &dst, 2).expect("재팩 실패");
        assert_eq!(stats.experts, 2);
        assert_eq!(stats.tensors, 8); // 2트리플×3 + 밀집 1 + ple 1
        assert_eq!(stats.ple_files, 1);

        // ── 출력 검증: 아카이브 개방 + 트리플 직접 대조.
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
        // 밀집·ple 원시 보존.
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
        // W4a16Model 완전 개방(quant splice · 트리플 계약).
        let store = W4a16Model::open(&dst).expect("W4a16 open 실패");
        assert_eq!(store.n_lins(), 2);
        assert_eq!(store.group(), 32);
        let cfg = std::fs::read_to_string(dst.join("config.json")).unwrap();
        assert!(cfg.contains("compressed-tensors"));
        assert!(cfg.contains("Qwen4ExpForConditionalGeneration"));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// qzeros 비상수 — 즉시 거부(D1 게이트).
    #[test]
    fn repack_rejects_nonconstant_qzeros() {
        let tmp = std::env::temp_dir().join(format!("fn-repack-neg-{}", std::process::id()));
        let src = tmp.join("src");
        let dst = tmp.join("dst");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();
        // 최소 트리플 1개 — qzeros 1워드 오염. 형상은 실측 쌍 유지(작게 못
        // 줄인다 — check_triple 고정 계약).
        let (n, k) = (640usize, 2560usize);
        let (kw, kg) = (k / 8, k / 128);
        let qw = vec![0u8; kw * n * 4];
        let mut qz = vec![0x77u8; kg * (n / 8) * 4];
        qz[0] = 0x76; // 니블 하나 어긋남
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
        // index의 scales 이름 오타 → StArchive가 먼저 거부하는 경로와 무관하게
        // 여기선 qzeros 검증 도달이 목적 — 정확한 이름으로 재작성.
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

    /// 실모델 동치 검증 — LLM170_FN_MODEL(GPTQ 원본) + LLM170_FN_MODEL_CT
    /// (재팩 출력) 지정 시에만 실행(CI·회귀 무관 — fn_probe 게이트 패턴).
    /// 표본: 층 {0,23,47} × 전문가 {0,7,511} × 3투영 — **전 행** 비트일치.
    #[test]
    fn fn_repack_equiv_real_model() {
        let (Ok(src), Ok(dst)) = (
            std::env::var("LLM170_FN_MODEL"),
            std::env::var("LLM170_FN_MODEL_CT"),
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
