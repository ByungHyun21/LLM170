//! [하네스 저작 원칙 — plans/129-cuda-only C1(원본 129 A10) 체크리스트]
//! ① 선행 단계 공유 버퍼 오염 점검: GdnFixture는 상수·입력을 매 케이스 독립 생성하고 S0≠0·링≠0 초기 상태를 의무 주입한다(합성 S0=0은 상태 버그를 가린다 — plans/124 §3.3).
//! ② 형상은 전 선형 메타에서 자동 열거(t=1..tmax·혼합 krate·S0≠0):
//!    실모델 형상 27B(hv=48)·35B-A3B(hv=32) 전수 — S0≠0 상태 경로·lc 순열 방향이 프로브 계약(도메인 열거는 실측 config 기준).
//! ③ 캡처-재생 3방향: 합성 입력·실층 캡처·쌍 디코더 A/B.
//! ④ 종단 상태가 유일 불변량: 판정은 종단 값 maxdiff(중간 단계는 진단).
//!
//! [검증층 원장 요약 — sm_89 실측 2026-10-04, 전체 원장은 exl3_cuda_probe.rs
//! 및 아래 GDN 체인 오라클 배너]
//! (i) 27B lay=47 T=32 S0≠0: 전 단계 0.000e0 · rel>5% 0/196608 ·
//! (ii) 35B lay=29: 0.000e0 · rel 0/131072 · 음성대조 (a) l2perm gather
//! 2.706e-1 · (b) S0=0 1.150e0 → NEG-DETECTED. 임계 2e-4(GDN_THRESH).
//!
//! [속도] 측정 대기 sm_80 — CMP 170HX 미도착(plans/124 §0).
//!
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 의존 금지.

use crate::rawcuda::exl3_cuda::Exl3CudaDecoder;
use crate::rawcuda::exl3_cuda_probe::{
    Rng, gdn_expf, gdn_logf, h16f, maxdiff_nan, red128_tree, st_to_f32,
};
use crate::rawcuda::gdn_cuda::GdnDims;

// ── GDN 체인 오라클(G5) — 커널 산술의 f16-정밀 미러 ──
// 근거 소스(전부 워크트리 줄번호, 2026-10-04):
// - rawhip/kernels/src_exl3.hip exl3_gdn_conv L330-365 · exl3_gdn_gate
//   L368-393 · exl3_gdn_l2perm L397-472 · exl3_gdn_scan L484-608
//   (산술 원본 — assets/exl3_gdn.cu가 1:1 직이식; 폭 인자화만 차이).
// - rawvk/checks/exl3_probes.rs scan_ref L237-328 · h16 L234-236
//   (청크 전개·f16 저장 지점의 독립 구조 미러 — vk↔코어 비트검증 계급).
// - rawhip/exl3_hip_probe.rs hip_gdn_check L728-986(체인 미러 원형 —
//   본 오라클은 이 흐름을 f16 저장 지점까지 정밀화).
// - crates/core/src/qwen4exp/stages/gdn.rs conv 링 회전 L69-96 ·
//   crates/core/src/ops.rs silu L128 / sigmoid L133 / softplus L138
//   (CPU 참조 계급 — 커널 __expf/__logf(수 ulp 근사 내장)는 호스트
//   정밀 f32 exp/ln으로 재현; 차이는 값당 ~수 ulp = 임계 2e-4의
//   1/100 미만 계급. scan의 expf/rsqrtf도 동일 취급).
//
// f16 저장 지점 계약(커널과 동일): sk/sv/A/KQ/KS/QS는 __float2half_rn
// 저장→__half2float 판독. 미러의 f32_to_f16/f16_to_f32는 결함 20호
// 수정판(RTNE 경계 0x1000) — G4 원장: 전 행 커널과 비트일치 실측.
// KS/QS는 8패스 16타일 증분 f16 누산(커널 패스 구조 그대로), dc/o는
// f32 전진 소거(먼저 d[i]=β·(v−e^g·KS), 이후 j<i 감산 — 순서 계약).

/// 오라클 단계 산출(단계별 판정용).
pub(crate) struct GdnRefStages {
    pub conv_q: Vec<f32>,
    pub conv_k: Vec<f32>,
    pub conv_v: Vec<f32>,
    pub ring_post: Vec<f32>,
    pub q2: Vec<f32>,
    pub k2: Vec<f32>,
    pub v2: Vec<f32>,
    pub bg: Vec<f32>,
    pub o_lc: Vec<f32>,
    pub st_post: Vec<f32>,
    pub gated: Vec<f32>,
}

/// GDN 체인 f16-정밀 오라클 — conv → l2perm(scatter) → scan(FLA 청크,
/// CS=32·TILE=16·8패스) → gate. 인자의 층 슬라이스(cw_l 등)는 호출측이
/// 전층 배열에서 잘라 전달(결정론 — 시드 기반 합성과 무관하게 동일).
#[allow(clippy::too_many_arguments)]
// [rustfmt 병리 실측 2026-10-04] 이 함수(230행 밀착 미러 — 다중 중첩
// 루프·인덱스 산식)를 포맷하면 rustfmt가 초선형 폭주(스킵 없이
// 100s+ 미완 — 좀비 프로세스 1362s CPU 실측, 스킵 시 전체 파일 61s
// --check 통과). PERM_INV와 동일 계열의 skip 계약: 본체는 수동
// rustfmt 스타일로 유지한다(미러 구조라 재포맷 불필요).
#[rustfmt::skip]
fn gdn_reference_chain(
    dm: &GdnDims,
    cw_l: &[f32],
    ab_l: &[f32],
    alog_l: &[f32],
    dtb_l: &[f32],
    nw_l: &[f32],
    xn: &[f32],
    qkv: &[f32],
    z: &[f32],
    ring0: &[f32],
    s0: &[f32],
    t_len: usize,
) -> GdnRefStages {
    let (hd, kl, vl, cch, hk, hv) = (dm.hidden, dm.k_len(), dm.v_len(), dm.conv_ch(), dm.h_k, dm.h_v);
    let g = hv / hk; // lc 순열 전치 폭(27B 3 · 35B 2)
    let p_inv = |h: usize| (h % g) * hk + h / g;

    // ── conv: 채널별 3탭 링 순차 회전(hip L330-365 · core gdn.rs L69-96) ──
    let mut conv_q = vec![0f32; t_len * kl];
    let mut conv_k = vec![0f32; t_len * kl];
    let mut conv_v = vec![0f32; t_len * vl];
    let mut ring = ring0.to_vec();
    for c in 0..cch {
        let (w0, w1, w2, w3) = (cw_l[c * 4], cw_l[c * 4 + 1], cw_l[c * 4 + 2], cw_l[c * 4 + 3]);
        let (mut h0, mut h1, mut h2) = (
            ring[c],
            ring[cch + c],
            ring[2 * cch + c],
        );
        for t in 0..t_len {
            let x = qkv[t * cch + c];
            let o = w3 * x + w0 * h0 + w1 * h1 + w2 * h2;
            let o = o / (1.0 + gdn_expf(-o));
            if c < kl {
                conv_q[t * kl + c] = o;
            } else if c < 2 * kl {
                conv_k[t * kl + (c - kl)] = o;
            } else {
                conv_v[t * vl + (c - 2 * kl)] = o;
            }
            h0 = h1;
            h1 = h2;
            h2 = x;
        }
        ring[c] = h0;
        ring[cch + c] = h1;
        ring[2 * cch + c] = h2;
    }

    // ── l2perm: a/b 도트(레인 분할 환원) + q/k L2 + v·beta|g scatter ──
    let mut q2 = vec![0f32; t_len * kl];
    let mut k2 = vec![0f32; t_len * kl];
    let mut v2 = vec![0f32; t_len * vl];
    let mut bg = vec![0f32; t_len * 2 * hv];
    let dot_lane = |xrow: &[f32], wrow: &[f32]| -> f32 {
        let mut red = [0f32; 128];
        for lane in 0..128 {
            let mut p = 0f32;
            let mut i = lane;
            while i < hd {
                p += xrow[i] * wrow[i];
                i += 128;
            }
            red[lane] = p;
        }
        red128_tree(&mut red);
        red[0]
    };
    for t in 0..t_len {
        let xrow = &xn[t * hd..(t + 1) * hd];
        for h in 0..hv {
            let a_v = dot_lane(xrow, &ab_l[h * hd..(h + 1) * hd]);
            let b_v = dot_lane(xrow, &ab_l[(hv + h) * hd..(hv + h + 1) * hd]);
            let pi = p_inv(h);
            // tid==0 블록 — softplus>20 직선 규약 포함(hip L445-452).
            let ssm_a = -gdn_expf(alog_l[h]);
            let adt = a_v + dtb_l[h];
            let sp = if adt > 20.0 { adt } else { gdn_logf(1.0 + gdn_expf(adt)) };
            bg[t * 2 * hv + pi] = 1.0 / (1.0 + gdn_expf(-b_v));
            bg[t * 2 * hv + hv + pi] = sp * ssm_a;
        }
        for kh in 0..hk {
            let mut red = [0f32; 128];
            for i in 0..128 {
                red[i] = conv_q[t * kl + kh * 128 + i] * conv_q[t * kl + kh * 128 + i];
            }
            red128_tree(&mut red);
            let qi = 1.0 / (red[0] + 1e-6).sqrt();
            for i in 0..128 {
                q2[t * kl + kh * 128 + i] = conv_q[t * kl + kh * 128 + i] * qi;
            }
            let mut red = [0f32; 128];
            for i in 0..128 {
                red[i] = conv_k[t * kl + kh * 128 + i] * conv_k[t * kl + kh * 128 + i];
            }
            red128_tree(&mut red);
            let ki = 1.0 / (red[0] + 1e-6).sqrt();
            for i in 0..128 {
                k2[t * kl + kh * 128 + i] = conv_k[t * kl + kh * 128 + i] * ki;
            }
        }
        for h in 0..hv {
            let pi = p_inv(h);
            for i in 0..128 {
                v2[t * vl + pi * 128 + i] = conv_v[t * vl + h * 128 + i];
            }
        }
    }

    // ── scan: FLA 청크(CS=32) — f16 저장 지점·소거 순서 정밀 미러 ──
    let cs = 32usize;
    let qs = 1.0f32 / (128.0f32).sqrt();
    let mut st = s0.to_vec();
    let mut o_lc = vec![0f32; t_len * vl];
    let n_chunks = t_len.div_ceil(cs);
    for c0 in 0..n_chunks {
        let t0 = c0 * cs;
        let n = (t_len - t0).min(cs);
        for h in 0..hv {
            let kh = h % hk;
            // sk/sv f16 저장(h16f 왕복 — 커널 __float2half_rn 지점).
            let mut sk = [[0f32; 128]; 32];
            let mut sv = [[0f32; 128]; 32];
            for i in 0..cs {
                if i < n {
                    for s2 in 0..128 {
                        sk[i][s2] = h16f(k2[(t0 + i) * kl + kh * 128 + s2]);
                        sv[i][s2] = h16f(v2[(t0 + i) * vl + h * 128 + s2]);
                    }
                }
            }
            let mut bp = [0f32; 32];
            let mut gcs = [0f32; 33];
            let mut acc = 0f32;
            for t in 0..cs {
                acc += if t < n { bg[(t0 + t) * 2 * hv + hv + h] } else { 0.0 };
                gcs[t] = acc;
            }
            gcs[cs] = acc;
            for i in 0..cs {
                bp[i] = if i < n { bg[(t0 + i) * 2 * hv + h] } else { 0.0 };
            }
            // A/KQ — j<=i 쌍만(나머지는 0 기록, 판독측 0-가드와 동일).
            let mut a_m = [[0f32; 32]; 32];
            let mut kq = [[0f32; 32]; 32];
            for i in 0..n {
                for j in 0..=i {
                    let mut dk = 0f32;
                    let mut dq = 0f32;
                    for s2 in 0..128 {
                        dk += sk[i][s2] * sk[j][s2];
                        dq += q2[(t0 + i) * kl + kh * 128 + s2] * sk[j][s2];
                    }
                    if j < i {
                        a_m[i][j] = h16f(dk * bp[i] * gdn_expf(gcs[i] - gcs[j]));
                    }
                    kq[i][j] = h16f(dq * qs * gdn_expf(gcs[i] - gcs[j]));
                }
            }
            // KS/QS — 8패스 16타일 증분 f16 누산(커널 패스 구조).
            let mut ks = [[0f32; 128]; 32];
            let mut qsm = [[0f32; 128]; 32];
            for pass_ in 0..8 {
                let s2b = pass_ * 16;
                for i in 0..cs {
                    for col in 0..128 {
                        let mut ak = 0f32;
                        let mut aq = 0f32;
                        for s2p in 0..16 {
                            let s_el = st[h * 128 * 128 + (s2b + s2p) * 128 + col];
                            ak += sk[i][s2b + s2p] * s_el;
                            // plans/cuda-port.md S5: 마지막 청크의 비활성
                            // q행은 GPU와 동일하게 0으로 마스킹한다.
                            let qv = if i < n {
                                q2[(t0 + i) * kl + kh * 128 + s2b + s2p]
                            } else {
                                0.0
                            };
                            aq += qv * s_el;
                        }
                        ks[i][col] = h16f(ks[i][col] + ak);
                        qsm[i][col] = h16f(qsm[i][col] + aq * qs);
                    }
                }
            }
            // 전진 소거(순서 계약): d[i] 먼저 β괄호, 이후 j<i 감산 —
            // o는 각 i 소거 직후(필요 dc[p≤i] 전부 확정).
            let mut dc = [[0f32; 128]; 32];
            for i in 0..n {
                for col in 0..128 {
                    let mut rhs = bp[i] * (sv[i][col] - gdn_expf(gcs[i]) * ks[i][col]);
                    for j in 0..i {
                        let aij = a_m[i][j];
                        if aij != 0.0 {
                            rhs -= aij * dc[j][col];
                        }
                    }
                    dc[i][col] = rhs;
                    let mut oi = gdn_expf(gcs[i]) * qsm[i][col];
                    for p2 in 0..=i {
                        let w = kq[i][p2];
                        if w != 0.0 {
                            oi += w * dc[p2][col];
                        }
                    }
                    o_lc[(t0 + i) * vl + h * 128 + col] = oi;
                }
            }
            // 상태 P6 갱신 — wsm(사망 행 0)·j 순차 f32.
            let gtot = gcs[cs];
            let mut wsm = [0f32; 32];
            for j in 0..cs {
                wsm[j] = if j < n { gdn_expf(gtot - gcs[j]) } else { 0.0 };
            }
            for s2 in 0..128 {
                for col in 0..128 {
                    let base = h * 128 * 128 + s2 * 128 + col;
                    let mut a2 = st[base] * gdn_expf(gtot);
                    for j in 0..n {
                        a2 += sk[j][s2] * wsm[j] * dc[j][col];
                    }
                    st[base] = a2;
                }
            }
        }
    }

    // ── gate: rms(o_lc)·nw·silu(z) → gated(HF, 역순열) ──
    let mut gated = vec![0f32; t_len * vl];
    for t in 0..t_len {
        for h in 0..hv {
            let pi = p_inv(h);
            let mut red = [0f32; 128];
            for i in 0..128 {
                let ov = o_lc[t * vl + pi * 128 + i];
                red[i] = ov * ov;
            }
            red128_tree(&mut red);
            let inv = 1.0 / (red[0] / 128.0 + 1e-6).sqrt();
            for i in 0..128 {
                let ov = o_lc[t * vl + pi * 128 + i];
                let zv = z[t * vl + h * 128 + i];
                gated[t * vl + h * 128 + i] = ov * inv * nw_l[i] * (zv / (1.0 + gdn_expf(-zv)));
            }
        }
    }
    GdnRefStages {
        conv_q,
        conv_k,
        conv_v,
        ring_post: ring,
        q2,
        k2,
        v2,
        bg,
        o_lc,
        st_post: st,
        gated,
    }
}

// ── GDN 픽스처(실가중 상수 + 결정론 시드 입력 — hip 프로브 방법론) ──

/// GDN 체인 값 maxdiff 임계(plans/124 §1 — hip 원장 1.724e-4 기준).
const GDN_THRESH: f32 = 2e-4;

/// 픽스처 — 상수는 실아카이브(conv1d·in_proj_a/b·A_log·dt_bias·
/// norm.weight — 변환 없는 원값, vk gdn_frame_init과 동일 경로), 입력은
/// 결정론 시드(hip 프로브 계급 스케일 xn ±0.3·qkv ±0.5·z ±0.4), 초기
/// 상태 S0≠0·링≠0(상태 경로 의무 — S0=0 합성이 상태 버그를 가린
/// 원장, §3.3). 대상 층 = n_gdn-1(말단 — 전층 스트라이드 경계).
struct GdnFixture {
    dims: GdnDims,
    layer: usize,
    xn: Vec<f32>,
    qkv: Vec<f32>,
    z: Vec<f32>,
    ring0: Vec<f32>,
    s0: Vec<f32>,
    cw: Vec<f32>,
    ab: Vec<f32>,
    alog: Vec<f32>,
    dtb: Vec<f32>,
    nw: Vec<f32>,
}

impl GdnFixture {
    /// 상수는 dir 아카이브에서 실측값을 읽고(대상 층만 — 잔여 0,
    /// set_gdn 전층 계약 유지), 입력은 seed로 생성. seed는 형상 구분.
    fn generate(dims: GdnDims, dir: &str, seed: u64) -> Result<Self, String> {
        use crate::rawcuda::exl3_cuda::StArchive;
        let (n, cch, hv, hd) = (dims.n_gdn, dims.conv_ch(), dims.h_v, dims.hidden);
        let layer = n - 1;
        // g-th GDN 층 ↔ 모델 층 il(4층 블록에 GDN 3개 — il%4!=3).
        let il = (layer / 3) * 4 + layer % 3;
        let lp = format!("model.language_model.layers.{il}.linear_attn");
        let ar = StArchive::open(std::path::Path::new(dir))?;
        let readf = |name: &str| -> Result<Vec<f32>, String> {
            let dt = ar.dtype_of(name).ok_or_else(|| format!("{name} 없음"))?;
            let raw = ar.read(name)?;
            st_to_f32(&raw, dt)
        };
        let cw_l = readf(&format!("{lp}.conv1d.weight"))?;
        let a_l = readf(&format!("{lp}.in_proj_a.weight"))?;
        let b_l = readf(&format!("{lp}.in_proj_b.weight"))?;
        let alog_l = readf(&format!("{lp}.A_log"))?;
        let dtb_l = readf(&format!("{lp}.dt_bias"))?;
        let nw_l = readf(&format!("{lp}.norm.weight"))?;
        if cw_l.len() != cch * 4
            || a_l.len() != hv * hd
            || b_l.len() != hv * hd
            || alog_l.len() != hv
            || dtb_l.len() != hv
            || nw_l.len() != 128
        {
            return Err(format!(
                "GDN 상수 형상 불일치: cw={} a={} b={} alog={} dtb={} nw={} (기대 cch={cch} hv={hv} hd={hd})",
                cw_l.len(),
                a_l.len(),
                b_l.len(),
                alog_l.len(),
                dtb_l.len(),
                nw_l.len()
            ));
        }
        let mut rng = Rng::new(seed);
        let unif = |rng: &mut Rng, amp: f64| ((rng.next_f64() * 2.0 - 1.0) * amp) as f32;
        let xn = (0..32 * hd).map(|_| unif(&mut rng, 0.3)).collect();
        let qkv = (0..32 * cch).map(|_| unif(&mut rng, 0.5)).collect();
        let z = (0..32 * dims.v_len())
            .map(|_| unif(&mut rng, 0.4))
            .collect();
        let ring0 = (0..3 * cch).map(|_| unif(&mut rng, 0.2)).collect();
        let s0 = (0..hv * 128 * 128).map(|_| unif(&mut rng, 0.1)).collect();
        // 전층 배열 + 대상 층 실측값 이식(ab는 a 다음 b — hip abuf 계약).
        let mut cw = vec![0f32; n * cch * 4];
        cw[layer * cch * 4..(layer + 1) * cch * 4].copy_from_slice(&cw_l);
        let mut ab = vec![0f32; n * 2 * hv * hd];
        ab[layer * 2 * hv * hd..layer * 2 * hv * hd + hv * hd].copy_from_slice(&a_l);
        ab[layer * 2 * hv * hd + hv * hd..(layer + 1) * 2 * hv * hd].copy_from_slice(&b_l);
        let mut alog = vec![0f32; n * hv];
        alog[layer * hv..(layer + 1) * hv].copy_from_slice(&alog_l);
        let mut dtb = vec![0f32; n * hv];
        dtb[layer * hv..(layer + 1) * hv].copy_from_slice(&dtb_l);
        let mut nw = vec![0f32; n * 128];
        nw[layer * 128..(layer + 1) * 128].copy_from_slice(&nw_l);
        Ok(Self {
            dims,
            layer,
            xn,
            qkv,
            z,
            ring0,
            s0,
            cw,
            ab,
            alog,
            dtb,
            nw,
        })
    }

    /// 대상 층 상수 슬라이스(오라클 입력).
    fn layer_consts(&self) -> (&[f32], &[f32], &[f32], &[f32], &[f32]) {
        let (cch, hv, hd) = (self.dims.conv_ch(), self.dims.h_v, self.dims.hidden);
        let l = self.layer;
        (
            &self.cw[l * cch * 4..(l + 1) * cch * 4],
            &self.ab[l * 2 * hv * hd..(l + 1) * 2 * hv * hd],
            &self.alog[l * hv..(l + 1) * hv],
            &self.dtb[l * hv..(l + 1) * hv],
            &self.nw[l * 128..(l + 1) * 128],
        )
    }

    fn reference(&self) -> GdnRefStages {
        let (cw_l, ab_l, alog_l, dtb_l, nw_l) = self.layer_consts();
        gdn_reference_chain(
            &self.dims,
            cw_l,
            ab_l,
            alog_l,
            dtb_l,
            nw_l,
            &self.xn,
            &self.qkv,
            &self.z,
            &self.ring0,
            &self.s0,
            32,
        )
    }
}

/// rel 잠행 집계(hip 프로브 판정 — d>1e-3 && d/max|want|>5%).
fn gdn_rel_bad(got: &[f32], want: &[f32]) -> usize {
    let mut bad = 0usize;
    for (g, w) in got.iter().zip(want) {
        let d = (g - w).abs();
        if d > 1e-3 && d / w.abs().max(1e-3) > 0.05 {
            bad += 1;
        }
    }
    bad
}

/// config.json 읽기 + GdnDims 유도(실측 차원 계약).
fn gdn_dims_from_model(dir: &str) -> Result<GdnDims, String> {
    let cfg = std::fs::read_to_string(format!("{dir}/config.json"))
        .map_err(|e| format!("{dir}/config.json: {e}"))?;
    GdnDims::from_config(&cfg)
}

/// exl3-cuda-gdn — plans/124 G5 GDN 체인(conv→l2perm→scan→gate) 값
/// maxdiff 판정(종단 임계 2e-4 · rel 0/N). (i) 27B 형상 T=32 S0≠0 ·
/// (ii) 35B-A3B 형상 T=32 S0≠0(실측 config.json 차원 — linear 헤드수
/// 상이: hv 48→32) · (iii) 단계별 국소화(conv 링 회전·l2perm scatter
/// 방향·scan 소거 순서 vs 오라클 단계). 형상 인자는 모델 디렉터리
/// (기본 D:/models 실측 인벤토리 — plans/124 §5). 하나라도 FAIL이면
/// Err(→ CLI 비영).
pub fn cuda_gdn_check(dir27: &str, dir35: &str) -> Result<String, String> {
    // plans/cuda-port.md S8: 슬롯 격리를 프로브로 증명한다. 슬롯 1로
    // 할당해 슬롯 0/1에 서로 다른 층 시드를 넣고 각자 CPU 미러와
    // 0.000e0로 맞는지 본다 — 슬롯 오프셋이 없으면 층 간 상태가 섞여
    // 반드시 어긋난다(가중치 인덱스 오염은 별도로 qnw 재사용으로 방어).
    let mut dec = Exl3CudaDecoder::empty()?;
    dec.n_slots = 2;
    dec.slot_pos = vec![0; 2];
    let dev = dec.device_name().to_string();
    let t_len = 32usize;
    let mut fails: Vec<String> = Vec::new();
    let mut report = String::new();

    for (tag, dir, seed) in [
        ("i", dir27, 0x170C_0DA0_0000_000Au64),
        ("ii", dir35, 0x170C_0DA0_0000_000Bu64),
    ] {
        let dims = gdn_dims_from_model(dir)?;
        let fx = GdnFixture::generate(dims, dir, seed)?;
        let lay = fx.layer;
        dec.set_gdn(dims, &fx.cw, &fx.ab, &fx.alog, &fx.dtb, &fx.nw)?;
        let want = fx.reference();
        let got = dec.gdn_chain_host(
            0,
            lay,
            t_len,
            &fx.xn,
            &fx.qkv,
            &fx.z,
            Some(&fx.s0),
            Some(&fx.ring0),
        )?;
        let (md, nan) = maxdiff_nan(&got, &want.gated);
        let rel = gdn_rel_bad(&got, &want.gated);
        let mname = if tag == "i" { "27B" } else { "35B-A3B" };
        let pass = md <= GDN_THRESH && nan == 0 && rel == 0;
        println!(
            "device: {dev} | exl3-cuda-gdn ({tag}) {mname} n_gdn={} hidden={} hk={} hv={} conv_ch={} lay={lay} T={t_len} S0!=0: end-to-end maxdiff={md:.3e} nan={nan} rel>5%={rel}/{} | {}",
            dims.n_gdn,
            dims.hidden,
            dims.h_k,
            dims.h_v,
            dims.conv_ch(),
            t_len * dims.v_len(),
            if pass { "PASS" } else { "FAIL" }
        );
        report.push_str(&format!(
            "({tag}) maxdiff={md:.3e} rel={rel}/{}",
            t_len * dims.v_len()
        ));
        if !pass {
            fails.push(format!(
                "({tag}) end-to-end maxdiff={md:.3e} nan={nan} rel={rel}"
            ));
        }

        // (iii) 단계별 국소화 — 직전 체인의 잔류 버퍼 판독(링 회전·
        // scatter 방향·소거 순서가 각 단계 값으로 잡히는지).
        let mids = dec.gdn_mids_host(0, lay, t_len)?;
        let stages: [(&str, &[f32], &[f32]); 10] = [
            ("conv q", &mids.conv_q, &want.conv_q),
            ("conv k", &mids.conv_k, &want.conv_k),
            ("conv v", &mids.conv_v, &want.conv_v),
            ("ring post-T", &mids.ring_post, &want.ring_post),
            ("l2 q2(L2)", &mids.q2, &want.q2),
            ("l2 k2(L2)", &mids.k2, &want.k2),
            ("l2 v2(lc scatter)", &mids.v2, &want.v2),
            ("l2 bg(beta|g)", &mids.bg, &want.bg),
            ("scan o_lc(elim)", &mids.o_lc, &want.o_lc),
            ("state post-T", &mids.st_post, &want.st_post),
        ];
        let mut worst = 0f32;
        for (name, g, w) in stages {
            let (m, n2) = maxdiff_nan(g, w);
            worst = worst.max(m);
            if m > GDN_THRESH || n2 > 0 {
                fails.push(format!("({tag} stage) {name} maxdiff={m:.3e} nan={n2}"));
                println!(
                    "device: {dev} | exl3-cuda-gdn ({tag}) stage {name}: maxdiff={m:.3e} nan={n2} | FAIL"
                );
            }
        }
        println!(
            "device: {dev} | exl3-cuda-gdn ({tag}) stages worst maxdiff={worst:.3e} (conv ring/l2 scatter/scan elim/ring·state post) | {}",
            if worst <= GDN_THRESH { "PASS" } else { "FAIL" }
        );
        report.push_str(&format!(" · ({tag}) stages worst={worst:.3e}"));
        // plans/cuda-port.md S5: 순차 디코드는 T=1, 첫 층 GI=0부터
        // 시작한다. T=32·마지막 층 프로브만으로는 메모리 경계가 검증되지 않는다.
        dec.set_gdn(dims, &fx.cw, &fx.ab, &fx.alog, &fx.dtb, &fx.nw)?;
        let s0 = vec![0.0f32; dims.h_v * dims.d * dims.d];
        let ring0 = vec![0.0f32; 3 * dims.conv_ch()];
        for layer in [0, lay] {
            let (cch, hv, hd) = (dims.conv_ch(), dims.h_v, dims.hidden);
            let want1 = gdn_reference_chain(
                &dims,
                &fx.cw[layer * cch * 4..(layer + 1) * cch * 4],
                &fx.ab[layer * 2 * hv * hd..(layer + 1) * 2 * hv * hd],
                &fx.alog[layer * hv..(layer + 1) * hv],
                &fx.dtb[layer * hv..(layer + 1) * hv],
                &fx.nw[layer * 128..(layer + 1) * 128],
                &fx.xn[..hd],
                &fx.qkv[..cch],
                &fx.z[..dims.v_len()],
                &ring0,
                &s0,
                1,
            );
            let got1 = dec.gdn_chain_host(
                0,
                layer,
                1,
                &fx.xn[..hd],
                &fx.qkv[..cch],
                &fx.z[..dims.v_len()],
                None,
                None,
            )?;
            let (md1, nan1) = maxdiff_nan(&got1, &want1.gated);
            let rel1 = gdn_rel_bad(&got1, &want1.gated);
            let pass1 = md1 <= GDN_THRESH && nan1 == 0 && rel1 == 0;
            println!(
                "device: {dev} | exl3-cuda-gdn ({tag}) GI={layer} T=1 zero-state maxdiff={md1:.3e} nan={nan1} rel={rel1} | {}",
                if pass1 { "PASS" } else { "FAIL" }
            );
            if !pass1 {
                fails.push(format!(
                    "({tag}) GI={layer} T=1 maxdiff={md1:.3e} nan={nan1} rel={rel1}"
                ));
            }
        }
        // S8 슬롯 격리: 슬롯 0(층 0)과 슬롯 1(층 lay)에 각각 시드를 넣고
        // 서로 다른 층·상태로 두 슬롯을 교차 실행한다. 슬롯 오프셋이 없으면
        // 슬롯 1이 슬롯 0의 링/상태를 덮어써 둘 중 하나가 반드시 어긋난다.
        let (cch, hv, hd) = (dims.conv_ch(), dims.h_v, dims.hidden);
        let seed_of = |l: usize| -> (Vec<f32>, Vec<f32>) {
            let base = 0x5177_0000u64.wrapping_mul(l as u64 + 1);
            let mut r = Rng::new(base);
            let mut st = vec![0f32; hv * 128 * 128];
            for v in st.iter_mut() {
                *v = ((r.next_f64() * 2.0 - 1.0) * 0.05) as f32;
            }
            let mut ring = vec![0f32; 3 * cch];
            for v in ring.iter_mut() {
                *v = ((r.next_f64() * 2.0 - 1.0) * 0.1) as f32;
            }
            (st, ring)
        };
        let (st0, ring0a) = seed_of(0);
        let (st1, ring0b) = seed_of(lay);
        for &(slot, layer, st, ring) in &[
            (0usize, 0usize, &st0, &ring0a),
            (1usize, lay, &st1, &ring0b),
        ] {
            let want = gdn_reference_chain(
                &dims,
                &fx.cw[layer * cch * 4..(layer + 1) * cch * 4],
                &fx.ab[layer * 2 * hv * hd..(layer + 1) * 2 * hv * hd],
                &fx.alog[layer * hv..(layer + 1) * hv],
                &fx.dtb[layer * hv..(layer + 1) * hv],
                &fx.nw[layer * 128..(layer + 1) * 128],
                &fx.xn[..hd],
                &fx.qkv[..cch],
                &fx.z[..dims.v_len()],
                ring,
                st,
                1,
            );
            let got = dec.gdn_chain_host(
                slot,
                layer,
                1,
                &fx.xn[..hd],
                &fx.qkv[..cch],
                &fx.z[..dims.v_len()],
                Some(st),
                Some(ring),
            )?;
            let (md, nan) = maxdiff_nan(&got, &want.gated);
            let rel = gdn_rel_bad(&got, &want.gated);
            let pass = md <= GDN_THRESH && nan == 0 && rel == 0;
            println!(
                "device: {dev} | exl3-cuda-gdn ({tag}) S8 slot={slot} GI={layer} T=1 S0!=0 maxdiff={md:.3e} nan={nan} rel={rel} | {}",
                if pass { "PASS" } else { "FAIL" }
            );
            if !pass {
                fails.push(format!(
                    "({tag}) slot{slot} GI={layer} maxdiff={md:.3e} nan={nan} rel={rel}"
                ));
            }
        }
    }

    if fails.is_empty() {
        Ok(format!("device: {dev} | {report} | ALL PASS"))
    } else {
        Err(format!(
            "exl3-cuda-gdn 실패 — {} (device: {dev})",
            fails.join(", ")
        ))
    }
}

/// exl3-cuda-gdn-neg — 음성대조 2종(원장 17호: 계기도 스스로 검증).
/// (a) l2perm gather 방향(결함류: 방향 — CPU 미러 쪽, §3.3 사고 재현):
///     정상(scatter) 오라클 대비 종단 maxdiff가 임계 초과.
/// (b) S0=0 입력: 동일 픽스처에서 상태를 0으로 주면 참출력(오라클은
///     S0≠0)과 이격 — 상태 경로 무시 구현이 탐지됨을 증명(§3.3 의무).
/// 양쪽 모두 초과 시 NEG-DETECTED 마커와 함께 Err(→ CLI 비영 exit).
pub fn cuda_gdn_negative_check(dir27: &str) -> Result<String, String> {
    let mut dec = Exl3CudaDecoder::empty()?;
    let dev = dec.device_name().to_string();
    let dims = gdn_dims_from_model(dir27)?;
    let fx = GdnFixture::generate(dims, dir27, 0x170C_0DA0_0000_000Au64)?;
    let lay = fx.layer;
    dec.set_gdn(dims, &fx.cw, &fx.ab, &fx.alog, &fx.dtb, &fx.nw)?;
    let want = fx.reference();

    // (a) gather l2perm(검증 전용 모듈 진입) vs scatter 오라클.
    let got_a = dec.gdn_chain_host_gather_l2perm(
        0,
        lay,
        32,
        &fx.xn,
        &fx.qkv,
        &fx.z,
        Some(&fx.s0),
        Some(&fx.ring0),
    )?;
    let (md_a, nan_a) = maxdiff_nan(&got_a, &want.gated);
    println!(
        "device: {dev} | exl3-cuda-gdn (iva) negative control l2perm gather vs scatter oracle: maxdiff={md_a:.3e} nan={nan_a} | FAIL(expected)"
    );
    let det_a = md_a > GDN_THRESH;

    // (b) S0=0 상태 입력 vs S0≠0 참오라클(상태 경로 판별력 증명).
    let zero_s0 = vec![0f32; dims.h_v * 128 * 128];
    let got_b = dec.gdn_chain_host(
        0,
        lay,
        32,
        &fx.xn,
        &fx.qkv,
        &fx.z,
        Some(&zero_s0),
        Some(&fx.ring0),
    )?;
    let (md_b, nan_b) = maxdiff_nan(&got_b, &want.gated);
    println!(
        "device: {dev} | exl3-cuda-gdn (ivb) negative control S0=0 input vs S0!=0 oracle: maxdiff={md_b:.3e} nan={nan_b} | FAIL(expected)"
    );
    let det_b = md_b > GDN_THRESH;

    if det_a && det_b {
        Err(format!(
            "NEG-DETECTED (a) gather maxdiff={md_a:.3e} (b) S0=0 maxdiff={md_b:.3e} > {GDN_THRESH:.0e} — 검증계기 정상(방향·상태 경로 결함 감지)"
        ))
    } else {
        Err(format!(
            "NEG-MISSED (a)={md_a:.3e} (b)={md_b:.3e} <= {GDN_THRESH:.0e} — 검증계기 결함: 음성이 탐지되지 않음"
        ))
    }
}
