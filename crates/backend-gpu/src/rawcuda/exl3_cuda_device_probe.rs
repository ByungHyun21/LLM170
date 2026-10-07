//! S10 디바이스 상주 forward 검증 — 호스트 스테이징 경로와의 종단 대조.
//!
//! [왜 값으로 증명하는가] 두 경로의 산술은 같은 커널·같은 순서지만 **값이
//! 이동하는 경로가 다르다**(디바이스 체인은 d2d·상주 버퍼만, S5는
//! d2h→h2d 왕복). "같은 코드"라는 보장이 아니라 토큰열로 증명해야 한다.
//!
//! [왜 종단 토큰열인가] 1스텝 로짓 대조는 **상태 오염을 놓친다**. 실제
//! 결함(dab_dev 미초기화)이 첫 실행에는 우연히 맞고 두 번째 forward부터
//! 나타나 1스텝 검사로는 절대 잡히지 않았다(원장 S10). 그래서 여러 스텝의
//! 그리디 토큰열을 비교한다 — GDN 스캔·KV 누적·pos 진행이 한꺼번에 드러난다.
//!
//! [진단 도구分层] 아래 함수들은 결함 국소화용이다. 순서대로 좁혀 쓴다:
//! embed_resid → first_norm → layer_trace → attn_input → mid.
//! 어느 층·어느 버퍼에서 벌어지는지 확정하기 위한 계기이며, 판정
//! 자체는 `stream_compare`가 한다.

use crate::rawcuda::exl3_cuda::Exl3CudaDecoder;
use crate::rawcuda::ffi::CUdeviceptr;

/// 디바이스 버퍼 n개를 호스트로 읽는다.
fn take(d: &Exl3CudaDecoder, ptr: CUdeviceptr, n: usize) -> Result<Vec<f32>, String> {
    let _g = d.cc.guard()?;
    let mut buf = vec![0u8; n * 4];
    d.cc.d2h(&mut buf, ptr)?;
    d.cc.sync()?;
    // SAFETY: d2h 동기 완료; buf는 n개의 f32 LE 값이다.
    Ok(unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const f32, n) }.to_vec())
}

fn md_of(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// argmax — CUDA 커널의 타이 규칙('>' strictly, 후발 승)을 흉내낸다.
fn argmax(v: &[f32]) -> usize {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    bi
}

/// S10 게이트: 두 경로의 종단 토큰열 비교 결과.
pub struct StreamReport {
    pub first_bad_step: Option<usize>,
    pub host: Vec<usize>,
    pub dev: Vec<usize>,
}

/// 두 경로로 동일 토큰열을 그리디 디코드해 비교한다.
pub fn stream_compare(
    host: &mut Exl3CudaDecoder,
    dev: &mut Exl3CudaDecoder,
    toks: &[u32],
    n_gen: usize,
) -> Result<StreamReport, String> {
    if toks.len() < 2 {
        return Err("S10 스트림: 최소 2토큰 프롬프트 필요".into());
    }
    // 프롬프트의 앞 n-1토큰으로 두 경로를 동일하게 프리필.
    for &t in &toks[..toks.len() - 1] {
        let row_h = host.embed_row_host(t);
        let row_d = dev.embed_row_host(t);
        if row_h.len() != row_d.len() {
            return Err("임베딩 행 길이 불일치".into());
        }
        let _ = host.forward(0, &row_h)?;
        let _ = dev.forward_device(0, &row_d)?;
    }
    // 마지막 프롬프트 토큰의 로짓 = 첫 예측.
    let row_h = host.embed_row_host(toks[toks.len() - 1]);
    let row_d = dev.embed_row_host(toks[toks.len() - 1]);
    let h_lg = host.forward(0, &row_h)?.0;
    let d_lg = dev.forward_device(0, &row_d)?.0;
    let mut h_stream = vec![argmax(&h_lg)];
    let mut d_stream = vec![argmax(&d_lg)];
    for _ in 0..n_gen.saturating_sub(1) {
        let t = *h_stream.last().unwrap() as u32;
        let row_h = host.embed_row_host(t);
        let row_d = dev.embed_row_host(t);
        let h_lg = host.forward(0, &row_h)?.0;
        let d_lg = dev.forward_device(0, &row_d)?.0;
        h_stream.push(argmax(&h_lg));
        d_stream.push(argmax(&d_lg));
    }
    let first_bad_step = h_stream
        .iter()
        .zip(d_stream.iter())
        .position(|(a, b)| a != b);
    Ok(StreamReport {
        first_bad_step,
        host: h_stream,
        dev: d_stream,
    })
}

/// 종단 검증 + 사람이 읽는 보고. 이것이 S10의 판정이다.
pub fn cuda_s10_stream_check(
    host: &mut Exl3CudaDecoder,
    dev: &mut Exl3CudaDecoder,
    toks: &[u32],
    n_gen: usize,
) -> Result<String, String> {
    let dev_name = dev.device_name().to_string();
    let r = stream_compare(host, dev, toks, n_gen)?;
    let same = r.first_bad_step.is_none();
    Ok(format!(
        "device: {dev_name} | exl3-cuda-s10 stream ({}+{n_gen} tok): host{:?} dev{:?} | \
         first_diverged={:?} | {}",
        toks.len(),
        r.host,
        r.dev,
        r.first_bad_step,
        if same {
            "IDENTICAL | PASS"
        } else {
            "DIVERGED | FAIL"
        }
    ))
}

// ── 진단 계기(판정 아님 — 결함 규위 좁히기용) ──

/// 임베딩 행이 디바이스에 올바르게 올라갔는지(잔차 dres writeback).
pub fn embed_resid_probe(
    host: &mut Exl3CudaDecoder,
    dev: &mut Exl3CudaDecoder,
    tok: u32,
) -> Result<String, String> {
    let rh = host.embed_row_host(tok);
    let rd = dev.embed_row_host(tok);
    let md_rows = md_of(&rh, &rd);
    let _g = dev.cc.guard()?;
    dev.ensure_chain_probe_bufs_pub()?;
    // SAFETY: f32 [hidden] 바이트 뷰 — 업로드까지 살아 있다.
    let hb = unsafe { std::slice::from_raw_parts(rh.as_ptr() as *const u8, rh.len() * 4) };
    dev.cc.h2d(dev.dres, hb)?;
    let back = take(dev, dev.dres, dev.hidden)?;
    Ok(format!(
        "S10 embed: row(host vs dev) maxdiff={md_rows:.3e} | dres writeback maxdiff={:.3e}",
        md_of(&rh, &back)
    ))
}

/// 첫 노름(w=0, ab=0) 직후의 xn·잔차·가중 대조.
pub fn first_norm_probe(
    host: &mut Exl3CudaDecoder,
    dev: &mut Exl3CudaDecoder,
    tok: u32,
) -> Result<String, String> {
    let rh = host.embed_row_host(tok);
    let rd = dev.embed_row_host(tok);
    let zb_h = vec![0u8; host.hidden * 4];
    let zb_d = vec![0u8; dev.hidden * 4];
    {
        let _g = host.cc.guard()?;
        // SAFETY: f32 슬라이스 → 바이트 뷰.
        let hb = unsafe { std::slice::from_raw_parts(rh.as_ptr() as *const u8, rh.len() * 4) };
        host.cc.h2d(host.dx, hb)?;
        host.cc.h2d(host.dab, &zb_h)?;
        host.norm_resid(0, host.dab, 1)?;
    }
    {
        let _g = dev.cc.guard()?;
        dev.ensure_chain_probe_bufs_pub()?;
        // SAFETY: 동일.
        let hb = unsafe { std::slice::from_raw_parts(rd.as_ptr() as *const u8, rd.len() * 4) };
        dev.cc.h2d(dev.dres, hb)?;
        dev.cc.h2d(dev.dab, &zb_d)?;
        dev.norm_resid_dev(0, dev.dres, dev.dab, 1)?;
    }
    let xn = md_of(
        &take(host, host.dxn, host.hidden)?,
        &take(dev, dev.dxn, dev.hidden)?,
    );
    let res = md_of(
        &take(host, host.dx, host.hidden)?,
        &take(dev, dev.dres, dev.hidden)?,
    );
    let nw = md_of(
        &take(host, host.dnw, host.hidden)?,
        &take(dev, dev.dnw, dev.hidden)?,
    );
    Ok(format!(
        "S10 first-norm: xn maxdiff={xn:.3e} resid maxdiff={res:.3e} nw[0] maxdiff={nw:.3e}"
    ))
}

/// 어텐션 입력(qg/kin/vin)·가중·GEMV 입력(xn) 대조.
/// qh가 다른데 입력이 다르면 그보다 앞선 단계가 원인이다.
pub fn attn_input_probe(
    host: &mut Exl3CudaDecoder,
    dev: &mut Exl3CudaDecoder,
) -> Result<String, String> {
    let dm = dev.attn_dims()?;
    let rh = host.embed_row_host(100);
    let rd = dev.embed_row_host(100);
    let _ = host.forward(0, &rh)?;
    let _ = dev.forward_device(0, &rd)?;
    let qg = md_of(
        &take(host, host.dqg_a, dm.qg_dim())?,
        &take(dev, dev.dqg_a, dm.qg_dim())?,
    );
    let kk = md_of(
        &take(host, host.dkin_a, dm.kv_dim())?,
        &take(dev, dev.dkin_a, dm.kv_dim())?,
    );
    let vv = md_of(
        &take(host, host.dvin_a, dm.kv_dim())?,
        &take(dev, dev.dvin_a, dm.kv_dim())?,
    );
    let nw = md_of(
        &take(host, host.dqnw_a, dm.n_attn * dm.d)?,
        &take(dev, dev.dqnw_a, dm.n_attn * dm.d)?,
    );
    let xn = md_of(
        &take(host, host.dxn, dev.hidden)?,
        &take(dev, dev.dxn, dev.hidden)?,
    );
    Ok(format!(
        "S10 attn-input: qg={qg:.3e} kin={kk:.3e} vin={vv:.3e} | q_norm_w={nw:.3e} | \
         xn(input)={xn:.3e}"
    ))
}

/// 어텐션 산출(qh)·GDN 산출(o_lc) 대조 — 어느 체인이 어긋났는지.
pub fn mid_layer_probe(
    host: &mut Exl3CudaDecoder,
    dev: &mut Exl3CudaDecoder,
    tok: u32,
) -> Result<String, String> {
    let dm = dev.attn_dims()?;
    let gd = dev.gdn_dims()?;
    let rh = host.embed_row_host(tok);
    let rd = dev.embed_row_host(tok);
    let _ = host.forward(0, &rh)?;
    let _ = dev.forward_device(0, &rd)?;
    let qh = md_of(
        &take(host, host.dqh_a, dm.q_dim())?,
        &take(dev, dev.dqh_a, dm.q_dim())?,
    );
    let o = md_of(
        &take(host, host.dgo, gd.v_len())?,
        &take(dev, dev.dgo, gd.v_len())?,
    );
    Ok(format!(
        "S10 mid: attn qh maxdiff={qh:.3e} | gdn o_lc maxdiff={o:.3e}"
    ))
}

/// 1스텝 상태·로짓 대조.
pub fn one_step_state_probe(
    host: &mut Exl3CudaDecoder,
    dev: &mut Exl3CudaDecoder,
    tok: u32,
) -> Result<String, String> {
    let rh = host.embed_row_host(tok);
    let rd = dev.embed_row_host(tok);
    let (lh, hh) = host.forward(0, &rh)?;
    let (ld, hd) = dev.forward_device(0, &rd)?;
    Ok(format!(
        "S10 1step: logits maxdiff={:.3e} hidden(pre-final-norm) maxdiff={:.3e} \
         pos host={} dev={} argmax host={} dev={}",
        md_of(&lh, &ld),
        md_of(&hh, &hd),
        host.slot_pos[0],
        dev.slot_pos[0],
        argmax(&lh),
        argmax(&ld)
    ))
}

/// 층 단위 추적 — 어느 층의 GEMV 산출부터 벌어지는지.
pub fn layer_trace_probe(
    host: &mut Exl3CudaDecoder,
    dev: &mut Exl3CudaDecoder,
    tok: u32,
    n_layers: usize,
) -> Result<String, String> {
    let gd = dev.gdn_dims()?;
    let h = dev.hidden;
    let rh = host.embed_row_host(tok);
    let rd = dev.embed_row_host(tok);
    // 두 디코더를 임베딩 → 첫 norm까지만 전진.
    let zb_h = vec![0u8; h * 4];
    let zb_d = vec![0u8; h * 4];
    {
        let _g = host.cc.guard()?;
        // SAFETY: f32 슬라이스 → 바이트 뷰.
        let hb = unsafe { std::slice::from_raw_parts(rh.as_ptr() as *const u8, rh.len() * 4) };
        host.cc.h2d(host.dx, hb)?;
        host.cc.h2d(host.dab, &zb_h)?;
        host.norm_resid(0, host.dab, 1)?;
    }
    {
        let _g = dev.cc.guard()?;
        dev.ensure_chain_probe_bufs_pub()?;
        // SAFETY: 동일.
        let hb = unsafe { std::slice::from_raw_parts(rd.as_ptr() as *const u8, rd.len() * 4) };
        dev.cc.h2d(dev.dres, hb)?;
        dev.cc.h2d(dev.dab, &zb_d)?;
        dev.norm_resid_dev(0, dev.dres, dev.dab, 1)?;
    }
    let mut first_bad: Option<String> = None;
    let mut gi = 0usize;
    for il in 0..n_layers {
        let lp = format!("model.language_model.layers.{il}");
        let key = if il % 4 == 3 {
            format!("{lp}.self_attn.q_proj")
        } else {
            format!("{lp}.linear_attn.in_proj_qkv")
        };
        let xn_h = take(host, host.dxn, h)?;
        let qh = host.gemv_host(&key, &xn_h)?;
        let host_val = if il % 4 == 3 {
            let kk = host.gemv_host(&format!("{lp}.self_attn.k_proj"), &xn_h)?;
            let vv = host.gemv_host(&format!("{lp}.self_attn.v_proj"), &xn_h)?;
            let att = host.attn_chain_host(0, il / 4, 1, &qh, &kk, &vv, host.slot_pos[0])?;
            att.outv
        } else {
            host.gdn_chain_host(0, gi, 1, &xn_h, &qh, &vec![0.0; gd.v_len()], None, None)?
        };
        let qp = dev.gemv_dev(&key, dev.dxn)?;
        let n_q = dev.lin_shape(&key).map(|x| x.1).unwrap_or(0);
        let qh_d = take(dev, qp, n_q)?;
        let md_q = md_of(&qh, &qh_d);
        let dev_val = if il % 4 == 3 {
            take(dev, dev.doutv_a, dev.attn_dims()?.q_dim())?
        } else {
            take(dev, dev.dgate, gd.v_len())?
        };
        let md_b = md_of(&host_val, &dev_val);
        if (md_q > 1e-2 || md_b > 1e-2) && first_bad.is_none() {
            first_bad = Some(format!("L{il} {key} gemv={md_q:.3e} branch={md_b:.3e}"));
        }
        if il % 4 != 3 {
            gi += 1;
        }
    }
    Ok(format!(
        "S10 layer-trace: {}",
        first_bad.unwrap_or_else(|| "no divergence within N layers".into())
    ))
}
