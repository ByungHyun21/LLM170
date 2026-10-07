//! S11 배치 프리필 검증 — T>1 배치 경로가 T=1 순차와 **같은 값**을 내는지.
//!
//! [왜 이것이 전제인가] 배치 프리필은 프롬프트 N토큰을 T=N으로 한 번에
//! 넘긴다. GEMM2·norm은 이미 T축을 지원하지만 GDN scan의 청크 알고리즘
//! (CS=32)과 어텐션 fwd3s(T≤8)는 T>1에서 상태 누적 규칙이 달라질 수 있다.
//! "모듈 프로브가 T=32로 PASS"만으로는 순차 경로와의 **동치성**이 증명되지
//! 않는다 — 배치 경로가 자기 규칙으로 맞게 계산될 뿐, 순차와 같은 결과인지
//! 는 별도 질문이다.
//!
//! [판정] 동일 임베딩 T개를 (a) T=1 순차로 N회, (b) T=N 배치로 1회 돌려
//! **마지막 행**을 비교한다. 배치가 틀리면 프리필을 배치화해서는 안 된다.

use crate::rawcuda::exl3_cuda::Exl3CudaDecoder;
use crate::rawcuda::exl3_cuda_probe::gen_unif;
use crate::rawcuda::ffi::CUdeviceptr;
use crate::rawcuda::gdn_cuda::GdnDims;

/// 배치 GEMM(bsdage) vs 순차 GEMM — 첫 층 선형 하나로 좁힌다.
/// GDN 층의 in_proj_qkv(T행) 결과를 순차 T회 gemv_host와 비교한다.
pub fn batch_gemv_probe(
    seq: &mut Exl3CudaDecoder,
    bat: &mut Exl3CudaDecoder,
    key: &str,
    x_rows: &[f32],
) -> Result<(f32, usize), String> {
    let k = seq.lin_shape(key).map(|x| x.0).ok_or("선형 없음")?;
    let n = seq.lin_shape(key).map(|x| x.1).ok_or("선형 없음")?;
    let t = x_rows.len() / k;
    // 순차: T회 gemv_host, 마지막 행.
    let mut last = Vec::new();
    for r in 0..t {
        last = seq.gemv_host(key, &x_rows[r * k..(r + 1) * k])?;
    }
    // 배치: 1회 gemm2_host(T행).
    let all = bat.gemm2_host(key, x_rows)?;
    let md = last
        .iter()
        .zip(all[(t - 1) * n..].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    Ok((md, n))
}

/// 배치 forward의 argmax 일치 판정 — 실사용 기준의 최종 게이트.
/// [왜 maxdiff가 아니라 argmax인가] GEMM2는 mma f16 누적이라 GEMV와 환원
/// 순서가 구조적으로 다르다(T>1에서 0이 될 수 없다). 그래서 "같은 토큰을
/// 고르는가"가 유일하게 의미 있는 판정이다 — 게이트 스크립트가 하는 것과
/// 같은 기준이다.
pub fn forward_batch_argmax(
    seq: &mut Exl3CudaDecoder,
    bat: &mut Exl3CudaDecoder,
    embed_rows: &[f32],
) -> Result<(bool, usize, usize), String> {
    let h = seq.hidden;
    let t = embed_rows.len() / h;
    let mut last = Vec::new();
    for r in 0..t {
        last = seq.forward_device(0, &embed_rows[r * h..(r + 1) * h])?.0;
    }
    let (lb, _) = bat.forward_batch_device(0, embed_rows)?;
    Ok((am(&last) == am(&lb), am(&last), am(&lb)))
}

fn am(v: &[f32]) -> usize {
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

/// 배치 forward 중간값 대조 — 어느 **층의 어느 버퍼**에서 벌어지는지.
/// 로짓 maxdiff만으로는 규위가 넓다(64층 중 어디서인지 모른다).
/// S10에서 이 계기가 결함을 3개씩 좁혔다(원장 S10).
pub fn batch_mid_probe(
    seq: &mut Exl3CudaDecoder,
    bat: &mut Exl3CudaDecoder,
    embed_rows: &[f32],
) -> Result<String, String> {
    let h = seq.hidden;
    let t = embed_rows.len() / h;
    // 두 디코더를 임베딩 → 첫 norm까지만 전진(T행).
    {
        let _g = seq.cc.guard()?;
        seq.ensure_chain_probe_bufs_pub()?;
        // SAFETY: f32 슬라이스 → 바이트 뷰.
        let eb = unsafe {
            std::slice::from_raw_parts(embed_rows.as_ptr() as *const u8, embed_rows.len() * 4)
        };
        seq.cc.h2d(seq.dres, eb)?;
        seq.norm_resid_dev(0, seq.dres, seq.dab_dev, t)?;
    }
    {
        let _g = bat.cc.guard()?;
        bat.ensure_chain_probe_bufs_pub()?;
        // SAFETY: 동일.
        let eb = unsafe {
            std::slice::from_raw_parts(embed_rows.as_ptr() as *const u8, embed_rows.len() * 4)
        };
        bat.cc.h2d(bat.dres, eb)?;
        bat.norm_resid_dev(0, bat.dres, bat.dab_dev, t)?;
    }
    // 첫 norm 산출 xn 대조 — 여기서 다르면 임베딩/norm 문제.
    let xn = md_of(&take(seq, seq.dxn, t * h)?, &take(bat, bat.dxn, t * h)?);
    Ok(format!("S11 batch first-norm: xn maxdiff={xn:.3e} (T={t})"))
}

fn md_of(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn take(d: &Exl3CudaDecoder, ptr: CUdeviceptr, n: usize) -> Result<Vec<f32>, String> {
    let _g = d.cc.guard()?;
    let mut buf = vec![0u8; n * 4];
    d.cc.d2h(&mut buf, ptr)?;
    d.cc.sync()?;
    // SAFETY: d2h 동기 완료; buf는 n개의 f32 LE 값.
    Ok(unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const f32, n) }.to_vec())
}

/// 배치 forward vs 순차(T=1) forward 종단 비교.
/// [판정] 같은 T개 임베딩을 순차 T회와 배치 1회로 돌려 마지막 로짓을
/// 비교한다. S11 동치성 게이트의 최종 단계 — 모듈 단위(GDN/어텐션)로는
/// 맞아도 forward 배선이 틀리면 여기서 잡힌다.
pub fn forward_batch_equivalence(
    seq: &mut Exl3CudaDecoder,
    bat: &mut Exl3CudaDecoder,
    embed_rows: &[f32],
) -> Result<(f32, usize), String> {
    let h = seq.hidden;
    let t = embed_rows.len() / h;
    if h == 0 || t == 0 {
        return Err("forward_batch_equivalence: 빈 입력".into());
    }
    // 순차: T회 T=1 (S10 디바이스 경로 = 서버 실제 경로).
    let mut last = Vec::new();
    for r in 0..t {
        last = seq.forward_device(0, &embed_rows[r * h..(r + 1) * h])?.0;
    }
    // 배치: 1회 T=t.
    let (lb, _) = bat.forward_batch_device(0, embed_rows)?;
    let md = last
        .iter()
        .zip(lb.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    Ok((md, lb.len()))
}

/// GDN 상수 묶음 — set_gdn 재호출용(디코더가 상수를 들지 않으므로 호출부가
/// 그대로 재제공한다. 동치성 검사에서 상태를 리셋하려면 상수를 다시 넣어야
/// 링·스캔이 0으로 돌아온다).
pub struct GdnConsts {
    pub cw: Vec<f32>,
    pub ab: Vec<f32>,
    pub alog: Vec<f32>,
    pub dtb: Vec<f32>,
    pub nw: Vec<f32>,
}

/// GDN 체인 동치성 — 같은 N개 행을 T=1 N회 vs T=N 1회로 돌려 마지막 행 비교.
/// 반환 (마지막 행 maxdiff, 비교 원소 수).
pub fn gdn_t_equivalence(
    dec: &mut Exl3CudaDecoder,
    dims: GdnDims,
    consts: &GdnConsts,
    layer: usize,
    n_rows: usize,
    seed: u64,
) -> Result<(f32, usize), String> {
    let (cch, vl, hd) = (dims.conv_ch(), dims.v_len(), dims.hidden);
    let xn = gen_unif(n_rows * hd, seed ^ 1, 0.3);
    let qkv = gen_unif(n_rows * cch, seed ^ 2, 0.5);
    let z = gen_unif(n_rows * vl, seed ^ 3, 0.4);
    let ring0 = gen_unif(3 * cch, seed ^ 4, 0.2);
    let s0 = gen_unif(dims.h_v * 128 * 128, seed ^ 5, 0.1);

    // (a) T=1 순차 — 첫 행에만 시드, 이후는 상주 상태를 이어 쓴다.
    let mut seq_last = Vec::new();
    for r in 0..n_rows {
        let out = dec.gdn_chain_host(
            0,
            layer,
            1,
            &xn[r * hd..(r + 1) * hd],
            &qkv[r * cch..(r + 1) * cch],
            &z[r * vl..(r + 1) * vl],
            if r == 0 { Some(s0.as_slice()) } else { None },
            if r == 0 { Some(ring0.as_slice()) } else { None },
        )?;
        seq_last = out;
    }
    // (b) T=N 배치 — 상태·링을 0으로 되돌리고(상수 재주입) 한 번에.
    dec.set_gdn(
        dims,
        &consts.cw,
        &consts.ab,
        &consts.alog,
        &consts.dtb,
        &consts.nw,
    )?;
    let out_b = dec.gdn_chain_host(0, layer, n_rows, &xn, &qkv, &z, Some(&s0), Some(&ring0))?;
    let tail = &out_b[(n_rows - 1) * vl..];
    let md = seq_last
        .iter()
        .zip(tail.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    Ok((md, vl))
}

/// 어텐션 동치성 — 같은 pos0에서 N행 어텐션이 순차 N회와 같은지.
/// [주의] 어텐션은 KV를 누적하므로 순차와 배치는 **원리적으로 다르다**.
/// 배치가 맞는 기준은 hip의 forward_batch_toks(배치 어텐션)이며, 여기서는
/// "T=1 N회가 T=N 1회와 같은가"만 확인한다 — prefill에서 T>1을 쓸 수 있는
/// 전제.
pub fn attn_t_equivalence(
    dec: &mut Exl3CudaDecoder,
    layer: usize,
    pos0: u32,
    n_rows: usize,
    seed: u64,
) -> Result<(f32, usize), String> {
    let dm = dec.attn_dims()?;
    let qg = gen_unif(n_rows * dm.qg_dim(), seed ^ 11, 0.3);
    let kin = gen_unif(n_rows * dm.kv_dim(), seed ^ 12, 0.3);
    let vin = gen_unif(n_rows * dm.kv_dim(), seed ^ 13, 0.3);
    let t = n_rows;
    if t > crate::rawcuda::attn_cuda::ATTN_F3S_TMAX {
        return Err(format!(
            "attn 동치성: T={t} > fwd3s 상한 {}",
            crate::rawcuda::attn_cuda::ATTN_F3S_TMAX
        ));
    }
    // 배치 1회.
    let out_b = dec.attn_chain_host(0, layer, t, &qg, &kin, &vin, pos0)?;
    // 순차 T회 — pos는 1씩 증가해야 한다(KV 누적).
    let mut seq_last = Vec::new();
    for r in 0..n_rows {
        let o = dec.attn_chain_host(
            0,
            layer,
            1,
            &qg[r * dm.qg_dim()..(r + 1) * dm.qg_dim()],
            &kin[r * dm.kv_dim()..(r + 1) * dm.kv_dim()],
            &vin[r * dm.kv_dim()..(r + 1) * dm.kv_dim()],
            pos0 + r as u32,
        )?;
        seq_last = o.outv;
    }
    let n = dm.q_dim();
    let md = seq_last
        .iter()
        .zip(out_b.outv[(n_rows - 1) * n..].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    Ok((md, n))
}
