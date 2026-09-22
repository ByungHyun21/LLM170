//! frame 진단 자산 — 체크섬·스텝 타이밍·스테이지 스킵 (plans/78 R9 선분리).
//! 청크 불변성 결함 조사(plans/77 §1)의 프로브가 여기에 모인다 —
//! forward 본체(frame/mod.rs)와 분리해 조사 중 편집 충돌을 줄인다.

use super::*;

// 스테이지 동기 마커 (LLM170_FRAME_SYNC=1) — 스티키 폴트의 발생 지점을
// 즉시 드러낸다(폴트는 다음 API 호출에서야 보고된다).
thread_local! {
    /// 스테이지 누적 시간 — (마지막 경계 시각, [(접미사, us, 호출수)]).
    static FT: std::cell::RefCell<(std::time::Instant, Vec<(String, u64, u64)>)> =
        std::cell::RefCell::new((std::time::Instant::now(), Vec::new()));
}

/// 프레임 스테이지 시간 계측 (LLM170_FRAME_TIME=1). sync_mark가 만드는 경계
/// 에서만 측정한다 — 프레임 op는 비동기라 호출 시간만으로는 GPU 시간이 안 나온다.
pub(super) fn ftime_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("LLM170_FRAME_TIME").is_some())
}

/// 진단용 스테이지 스킵(LLM170_STAGE_SKIP="qsa,gdn,moe") — 비용 분해 전용.
pub fn stage_skipped(name: &str) -> bool {
    std::env::var("LLM170_STAGE_SKIP")
        .map(|v| v.split(',').any(|x| x.trim() == name))
        .unwrap_or(false)
}

/// 진단용 프레임 체크섬 — `LLM170_DUMP=...,checksum`. np·단일·배치 경로 공용.
/// 버퍼 앞 t·n개를 전부 읽어 합과 행 표본(첫·중간·마지막 행의 첫 원소)을
/// 보고한다. 청크 크기가 다른 두 실행에서 "같은 층·같은 단계·같은 토큰 수"를
/// 맞대어 첫 발산 지점을 찾는 용도 — 합만으로는 상쇄로 가려질 수 있어 행
/// 표본을 함께 낸다. 기본 꺼짐(1회 판독).
pub(super) fn frame_ck(acc: &dyn Accelerator, h: u64, n: usize, t: usize, tag: &str) {
    let o = llm170_diag::dump::opts();
    if !o.checksum && !o.row_on(tag) {
        return;
    }
    let rows_on = o.row_on(tag);
    let mut v = vec![0.0f32; n * t];
    if acc.frame_read(h, &mut v).is_ok() {
        if o.checksum {
            let s: f64 = v.iter().map(|&x| x as f64).sum();
            let mid = v[(t / 2) * n];
            let last = v[(t - 1) * n];
            eprintln!(
                "[npck] {tag} t={t} sum={s:.6} v0={:.6} mid0={mid:.6} last0={last:.6}",
                v[0]
            );
        }
        if rows_on {
            let cap = t.min(720);
            // 행당 4표본(첫·둘째·100번째·끝 원소) — 요소 축 커버리지.
            let e1 = 1.min(n.saturating_sub(1));
            let e2 = 100.min(n.saturating_sub(1));
            let e3 = n.saturating_sub(1);
            let vals: Vec<String> = (0..cap)
                .map(|r| {
                    let b = r * n;
                    format!(
                        "{:08x},{:08x},{:08x},{:08x}",
                        v[b].to_bits(),
                        v[b + e1].to_bits(),
                        v[b + e2].to_bits(),
                        v[b + e3].to_bits()
                    )
                })
                .collect();
            eprintln!("[nprd] {tag} t={t} n={n} {}", vals.join(" "));
        }
        if rows_on && o.row0full {
            let full = n.min(2560);
            let rows_n = t.min(16);
            for r in 0..rows_n {
                let vals: Vec<String> = v[r * n..r * n + full]
                    .iter()
                    .map(|x| format!("{:08x}", x.to_bits()))
                    .collect();
                eprintln!("[npr0] {tag} r={r} n={n} {}", vals.join(" "));
            }
        }
    }
}

/// 진단용 버퍼 FNV 해시 — `LLM170_DUMP=...,bufhash`. 앞 len 원소의 비트를
/// 해시해 한 줄로 찍는다. 층 경계마다 전 버퍼를 찍어 첫 오염 버퍼를
/// 찾는 용도(plans/80 §A — 메모리 결함 추적).
pub(super) fn buf_hash(acc: &dyn Accelerator, h: u64, len: usize, tag: &str) {
    if !llm170_diag::dump::opts().bufhash || len == 0 {
        return;
    }
    let mut v = vec![0.0f32; len];
    acc.frame_sync();   // plans/84 E.2: 커스텀 스트림 미완결 쓰기 경합 제거
    if acc.frame_read(h, &mut v).is_ok() {
        if std::env::var_os("LLM170_DUMP_E2VALS").is_some() && v.len() >= 8 {
            eprintln!("[npbv] {tag} first8={:x?}", v[..8].iter().map(|f| f.to_bits()).collect::<Vec<_>>());
        }
        let mut x = 0xcbf29ce484222325u64;
        for f in &v {
            x ^= f.to_bits() as u64;
            x = x.wrapping_mul(0x100000001b3);
        }
        eprintln!("[npbh] {tag} len={len} h={x:016x}");
    }
}

pub(super) fn sync_mark(acc: &dyn Accelerator, tag: &str, h: u64) -> Result<(), Q4Error> {
    match (ftime_on(), std::env::var_os("LLM170_FRAME_SYNC").is_some()) {
        (false, false) => return Ok(()),
        (ft, sync) => {
            // 1원소 판독 = 동기 + 폴트 보고 (barrier는 오류를 삼킨다).
            let mut v = [0.0f32; 1];
            acc.frame_read(h, &mut v)
                .map_err(|e| Q4Error::Io(format!("fsync {tag}: {e}")))?;
            if ft {
                FT.with(|s| {
                    let mut s = s.borrow_mut();
                    let dt = s.0.elapsed().as_micros() as u64;
                    s.0 = std::time::Instant::now();
                    let key = tag.rsplit('.').next().unwrap_or(tag).to_string();
                    match s.1.iter_mut().find(|e| e.0 == key) {
                        Some(e) => {
                            e.1 += dt;
                            e.2 += 1;
                        }
                        None => s.1.push((key, dt, 1)),
                    }
                });
            }
            if sync {
                eprintln!("# fsync {tag}");
            }
        }
    }
    Ok(())
}

/// 청크/스텝 단위 리포트 — 누적이 있으면 한 줄 출력 후 초기화.
/// (디코드 t=1도 찍는다: 프리필과 달리 동기 지점이 많아 누적-지연 왜곡이 없다.)
pub(super) fn ftime_report(_t: usize) {
    if !ftime_on() {
        return;
    }
    FT.with(|s| {
        let mut s = s.borrow_mut();
        if !s.1.is_empty() {
            s.1.sort_by_key(|b| std::cmp::Reverse(b.1));
            let mut line = String::from("# frame-time(t) ");
            for (k, us, n) in s.1.iter() {
                line.push_str(&format!("{k}={:.1}ms×{n} ", *us as f64 / 1e3));
            }
            eprintln!("{line}");
            s.1.clear();
        }
        s.0 = std::time::Instant::now();
    });
}

/// 단계 덤프 (LLM170_Q4_DBG=1) — 값 경로와 같은 양을 찍어 대조한다.
pub(super) fn dbg(tag: &str, acc: &dyn Accelerator, h: u64, n: usize) {
    if std::env::var_os("LLM170_Q4_DBG").is_none() {
        return;
    }
    let mut v = vec![0.0f32; n];
    if acc.frame_read(h, &mut v).is_err() {
        return;
    }
    let s: f64 = v.iter().map(|&x| x as f64).sum();
    let mx = v.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    eprintln!("# fdbg {tag}: sum={s:.6} max={mx:.6} v0..3={:?}", &v[..3.min(n)]);
}

/// LLM170_PLE_CHECK 그림자(plans/90 B4 이동) — PLE 직전 res_hc에서 호스트
/// 재계산해 디바이스 key/value/gate/출력과 대조. 조사 프로브(플랜 74 잔재).
#[allow(clippy::too_many_arguments)]
pub(super) fn ple_check_shadow(
    acc: &dyn Accelerator,
    f: &super::Frame4,
    ctx: &stages::Ctx,
    model: &Model4,
    seq_st: &mut SeqState4,
    il: usize,
    emb: &[f32],
    ple_rows: &[u32],
    pre_capture: &[f32],
    hc: usize,
    n: usize,
    t: usize,
) -> Result<(), Q4Error> {
    let hp = &model.hp;
    let mut rows2: Vec<Vec<f32>> = vec![pre_capture.to_vec()];
    stages::ple_block(ctx, seq_st, il, &mut rows2, ple_rows, Some(vec![emb.to_vec()]))?;
    let host: Vec<f32> = rows2.concat();
    let mut dkey = vec![0.0f32; hc * n];
    let mut dval = vec![0.0f32; n];
    acc.frame_read(f.ple_key, &mut dkey).map_err(Q4Error::Io)?;
    acc.frame_read(f.ple_value, &mut dval).map_err(Q4Error::Io)?;
    let nk = model.f32_vec4(&format!("blk.{il}.ple_norm_key.weight"))?;
    let nq = model.f32_vec4(&format!("blk.{il}.ple_norm_query.weight"))?;
    let nc = model.f32_vec4(&format!("blk.{il}.ple_norm_conv.weight"))?;
    let mut hkey = vec![vec![0.0f32; hc * n]; 1];
    let w_key2 = model.w4(&format!("blk.{il}.ple_key.weight"))?;
    let w_value2 = model.w4(&format!("blk.{il}.ple_value.weight"))?;
    ctx.mm_batch(&[emb.to_vec()], &w_key2, &mut hkey)?;
    let mut hval = vec![vec![0.0f32; n]; 1];
    ctx.mm_batch(&[emb.to_vec()], &w_value2, &mut hval)?;
    let mk = dkey.iter().zip(hkey[0].iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let mv = dval.iter().zip(hval[0].iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let mut dgate = vec![0.0f32; hc];
    let mut dgated = vec![0.0f32; hc * n];
    acc.frame_read(f.ple_gate, &mut dgate).map_err(Q4Error::Io)?;
    acc.frame_read(f.ple_gated, &mut dgated).map_err(Q4Error::Io)?;
    // 호스트 게이트 재계산(ple_block 잔차부와 동일식)
    let mut hgate = vec![0.0f32; hc];
    for s in 0..hc {
        let kk = &hkey[0][s * n..(s + 1) * n];
        let kn = crate::ops::rms_norm(kk, &nk[s * n..(s + 1) * n], hp.eps);
        let qq = &pre_capture[s * n..(s + 1) * n];
        let qn = crate::ops::rms_norm(qq, &nq[s * n..(s + 1) * n], hp.eps);
        let mut dot = 0.0f32;
        for i in 0..n { dot += kn[i] * qn[i]; }
        dot /= (n as f32).sqrt();
        let mag = dot.abs().max(1e-6).sqrt();
        hgate[s] = crate::ops::sigmoid(if dot >= 0.0 { mag } else { -mag });
    }
    eprintln!("# ple-check lens nk={} nq={} nc={} pre.len={} key.len={}", nk.len(), nq.len(), nc.len(), pre_capture.len(), hkey[0].len());
    let mg = dgate.iter().zip(hgate.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("# ple-check gate dev={:?} host={:?} max={mg:.3e}", dgate.iter().map(|x| (x*1e4).round()/1e4).collect::<Vec<_>>(), hgate.iter().map(|x| (x*1e4).round()/1e4).collect::<Vec<_>>());
    eprintln!("# ple-check key max|d-h|={mk:.3e} value max|d-h|={mv:.3e}");
    let mut r2 = vec![0.0f32; hc * n];
    acc.frame_read(f.res_hc, &mut r2).map_err(Q4Error::Io)?;
    let mut md = 0.0f32;
    let mut at = 0usize;
    for (i, (a, b)) in r2.iter().zip(host.iter()).enumerate() {
        let d = (a - b).abs();
        if d > md { md = d; at = i; }
    }
    eprintln!(
        "# ple-check pos={} max|dev-host|={md:.3e} at={at} (dev={:.4} host={:.4})",
        seq_st.pos, r2[at.min(r2.len() - 1)], host[at.min(host.len() - 1)]
    );
    seq_st.qsa_host_stale = false;
    let _ = (dgated, t);
    Ok(())
}
