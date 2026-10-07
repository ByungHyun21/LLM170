//! [하네스 저작 원칙 — plans/129-cuda-only C1(원본 129 A10) 체크리스트]
//! ① 선행 단계 공유 버퍼 오염 점검: AttnFixture는 케이스마다 KV 캐시를 시드 주입(seed_kv)해 초기화한다 — fwd3s가 이전 케이스 행을 읽지 않는다.
//! ② 형상은 전 선형 메타에서 자동 열거(t=1..tmax·혼합 krate·S0≠0):
//!    실모델 형상 27B(q24/kv4)·35B-A3B(q16/kv2) 전수, T∈{1,8,9}(9=도메인 거부) — t=1..tmax 경계 포함.
//! ③ 캡처-재생 3방향: 합성 입력·실층 캡처·쌍 디코더 A/B.
//! ④ 종단 상태가 유일 불변량: 판정은 종단 값 maxdiff(중간 단계는 진단).
//!
//! [검증층 원장 요약 — sm_89 실측 2026-10-04, 전체 원장은 아래 어텐션 배너]
//! (i) 27B T=1: 0.000e0 · (ib) T=8: 0.000e0 · (ii) 35B T=8: 0.000e0
//! (비트동일) · (iii) T=9 도메인 Err · 음성대조 호스트 pos 사본 1.498e-1
//! → NEG-DETECTED(결함 4호). 임계 2e-7(ATTN_THRESH — 전 모듈 최tight).
//!
//! [속도] 측정 대기 sm_80 — CMP 170HX 미도착(plans/124 §0).
//!
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 의존 금지.

use crate::rawcuda::attn_cuda::AttnDims;
use crate::rawcuda::exl3_cuda::Exl3CudaDecoder;
use crate::rawcuda::exl3_cuda_probe::{
    Rng, gdn_exp_d, gdn_expf, maxdiff_nan, red128_tree, st_to_f32,
};
use std::ffi::c_void;

// ── 어텐션 프로브(G6) — plans/124 §1 "어텐션 (prep/fwd3s)", 임계 2e-7 ──
// 근거 소스(전부 워크트리 줄번호, 2026-10-04):
// - rawhip/kernels/src_exl3.hip exl3_attn_prep L607-687 · exl3_attn_fwd3s
//   L799-853 · exl3_pos_bump L1152-1155(산술 원본 — assets/exl3_attn.cu가
//   1:1 직이식; 폭 인자화·pos0_ 파라미터 제거가 유일한 차이).
// - rawhip/exl3_hip.rs 어텐션 발사(그리드 (T,28)/(T,24)·블록 128/256·
//   ai=il/4 색인 — L946-1031 배치 · L1629-1677 디코드).
// - rawvk/checks/exl3_resident.rs attn_norms_dump L781-801(q/k_norm 저장소
//   w−1 → +1 규약, §3.4 constant_bias=1.0).
// 트랜센던트 트윈 계약(G5 노선 계승 — 어텐션은 계약 최tight 2e-7이라
// 필수): exp는 gdn_exp_d와 동일 DAG(.cu attn_exp_d가 gdn_exp_d의 베낀
// 사본), theta/sincos는 아래 attn_theta/attn_sincos_d(.cu 쌍둥이와
// 리터럴까지 동일, -fmad=false 빌드). Rust f64 산술은 IEEE 정확 반올림·
// 수축 없음 → 장치와 비트동일(sqrt/div도 양측 IEEE).
//
// [실측 원장 2026-10-04, RTX 4070 SUPER(sm_89) — 검증 호스트]
//   (i) 27B q24/kv4 lay=15 pos0=33 T=1: qh·KC·VC·outv 전 단계 maxdiff
//   0.000e0(오라클과 비트동일) · (ib) 27B T=8: 0.000e0 · (ii) 35B-A3B
//   q16/kv2 lay=9 T=8: 0.000e0 — 미러 계약 실증(임계 2e-7 대비 무한대
//   여유, f32 ulp 6e-8의 절반 이하 계급 = 잔차 0).
//   (iii) T=9: 모듈 진입 Err(도메인 거부) + 커널 원시 발사 센티널 불변
//   (부분 기록 없음) — 도메인 이중 강제의 값 검증.
//   음성대조: pos_bump 장치 판독 32→33 실측 · 호스트 사본(32) 경로 종단
//   1.498e-1 > 2e-7 → NEG-DETECTED(결함 4호: 디바이스 pp[0] 판독 계약의
//   값 검증력 입증).
//   sm_80 자원 증거(cuobjdump --dump-resource-usage, 커밋 fatbin):
//   exl3_attn_prep(+hostpos 쌍둥이) REG:26 SHARED:1536 → GA100 블록 128
//   스레드=4와프, 와프 48 상한 기준 12블록/SM(레지스터 26×128=3.3K는
//   여유) · exl3_attn_fwd3s REG:34 SHARED:6144 → 256스레드=8와프, 6블록/SM
//   (와프 상한) — T=8 그리드 192블록(27B)은 소형 도메인 설계 그대로
//   (T=1 디코드 24블록 저점유는 정확성 우선 피벗, 처리량 경로는 t-블록
//   fwd3 후속 목표). 개발기 4070(sm_89)은 정합 검증 전용 — CMP 170HX
//   실측은 도착 후(plans/124 §0).

/// 어텐션 값 maxdiff 임계(plans/124 §1 — 전 모듈 통틀어 최tight).
const ATTN_THRESH: f32 = 2e-7;

/// rope theta 트윈 — .cu attn_theta와 동일 f64 DAG: e=−2·tid/64(정확),
/// exp(ln(1e7)·e) — ln(1e7) 리터럴 동일 문자열(비트동일 계약).
fn attn_theta(tid: usize) -> f32 {
    let e = -(2.0 * tid as f64) / 64.0;
    gdn_exp_d(16.11809565095832 * e) as f32
}

/// sincos f64 트윈 — .cu attn_sincos_d와 동일 DAG: Cody-Waite 2분할
/// 환원(k·pio2_1은 53비트 내 정확) + z⁶ Horner 테일러 + 사분면 n=k&3.
/// 도메인 0 ≤ a ≤ 2^20(rope ang = pos·theta ≤ cap).
// [rustfmt skip — G6 병리 계승(2026-10-04, 시간제한 계약)] Horner 9중 괄호.
#[rustfmt::skip]
fn attn_sincos_d(a: f64) -> (f64, f64) {
    let invpio2 = 6.36619772367581342433e-01f64;
    let pio2_1 = 1.57079632673412561417e+00f64;
    let pio2_1t = 6.07710050650619224932e-11f64;
    let kd = (a * invpio2 + 0.5).floor();
    let k = kd as i64;
    let mut r = a - k as f64 * pio2_1;
    r -= k as f64 * pio2_1t;
    let z = r * r;
    let st = r
        * (1.0 - z
            * (0.16666666666666666f64
                - z * (0.008333333333333333f64
                    - z * (1.9841269841269841e-04f64
                        - z * (2.7557319223985893e-06f64
                            - z * (2.5052108385441720e-08f64
                                - z * 1.6059043836821613e-10f64))))));
    let ct = 1.0
        - z * (0.5f64
            - z * (0.041666666666666664f64
                - z * (0.0013888888888888889f64
                    - z * (2.4801587301587302e-05f64
                        - z * (2.7557319223985893e-07f64
                            - z * 2.0876756987868100e-09f64)))));
    // 사분면 매핑 — fdlibm 규약(n=1 → (−st, ct) · n=3 → (st, −ct)).
    // [결함 21호 수정 2026-10-05] 기존 표는 1/3 교차(커널과 자기일치해 미검출,
    // FND 교차측정으로 발견). 바른 표로 교정 — libm 스위프 ≤2 f32 ulp 검증.
    match (k & 3) as i32 {
        0 => (ct, st),
        1 => (-st, ct),
        2 => (-ct, -st),
        _ => (st, -ct),
    }
}

fn attn_cosf(a: f32) -> f32 {
    attn_sincos_d(a as f64).0 as f32
}

fn attn_sinf(a: f32) -> f32 {
    attn_sincos_d(a as f64).1 as f32
}

/// prep 1헤드분 rms+rope 미러(.cu exl3_attn_prep j블록 — ss 2항·red[128]
/// 트리·inv=IEEE 1/sqrt·rope 불32쌍 회전까지 동일 순서).
fn attn_prep_head(x256: &[f32], nw: &[f32], pos: i32) -> [f32; 256] {
    let mut hd = [0f32; 256];
    hd.copy_from_slice(x256);
    let mut red = [0f32; 128];
    for tid in 0..128 {
        red[tid] = hd[tid] * hd[tid] + hd[tid + 128] * hd[tid + 128];
    }
    red128_tree(&mut red);
    let inv = 1.0f32 / (red[0] / 256.0 + 1e-6).sqrt();
    for tid in 0..128 {
        hd[tid] = hd[tid] * inv * nw[tid];
        hd[tid + 128] = hd[tid + 128] * inv * nw[128 + tid];
    }
    for tid in 0..32 {
        let theta = attn_theta(tid);
        let ang = pos as f32 * theta;
        let c = attn_cosf(ang);
        let s = attn_sinf(ang);
        let (x0, x1) = (hd[tid], hd[tid + 32]);
        hd[tid] = x0 * c - x1 * s;
        hd[tid + 32] = x0 * s + x1 * c;
    }
    hd
}

/// 어텐션 오라클 단계 산출(단계별 판정용).
pub(crate) struct AttnRefStages {
    pub qh: Vec<f32>,
    pub kc_rows: Vec<f32>,
    pub vc_rows: Vec<f32>,
    pub outv: Vec<f32>,
}

/// 어텐션 체인 f32-정밀 오라클: prep(q rms+rope·KV rms+rope→캐시·v 복사)
/// → fwd3s(스코어 스레드분할 순차 d-누산·fmax 트리·지수·ls 스레드 보폭
/// 순차+red[256] 트리 합·AV 순차·게이트 sigmoid — 커널과 동일 순서).
/// kc/vc는 대상 층 전체 캐시([cap*kv_dim], 히스토리 포함)를 받아 신규
/// T행을 그 위에 기록한다(디바이스 attn_seed_kv + prep과 동일 상태).
// [rustfmt 병리 실측 2026-10-04 — G5 원장 계승] 이 함수(밀착 다중
// 중첩 미러 — fwd3s 스레드 보폭·red[256] 트리 순서)를 포함한 트리
// 전체 --check가 초선형 폭주(10분+ 미완, 좀비 rustfmt — gdn_reference_chain
// 1362s 사고와 동일 계열). #[rustfmt::skip]로 본체는 수동 rustfmt
// 스타일 유지(G5 PERM_INV·gdn_reference_chain 계열 skip 계약).
#[rustfmt::skip]
fn attn_reference_chain(
    dm: &AttnDims,
    qnw_l: &[f32],
    knw_l: &[f32],
    kc: &mut [f32],
    vc: &mut [f32],
    qg: &[f32],
    kin: &[f32],
    vin: &[f32],
    pos0: u32,
    t_len: usize,
) -> AttnRefStages {
    let (kv_dim, q_dim, qg_dim) = (dm.kv_dim(), dm.q_dim(), dm.qg_dim());
    let mut qh = vec![0f32; t_len * q_dim];
    for t in 0..t_len {
        let pos = pos0 as i32 + t as i32;
        for j in 0..dm.q_heads {
            let head = attn_prep_head(&qg[t * qg_dim + j * 512..t * qg_dim + j * 512 + 256], qnw_l, pos);
            qh[t * q_dim + j * 256..t * q_dim + j * 256 + 256].copy_from_slice(&head);
        }
        for m in 0..dm.kv_heads {
            let head = attn_prep_head(&kin[t * kv_dim + m * 256..t * kv_dim + m * 256 + 256], knw_l, pos);
            let dst = (pos0 as usize + t) * kv_dim + m * 256;
            kc[dst..dst + 256].copy_from_slice(&head);
            let vsrc = t * kv_dim + m * 256;
            vc[dst..dst + 256].copy_from_slice(&vin[vsrc..vsrc + 256]);
        }
    }
    let kc_rows: Vec<f32> = (0..t_len)
        .flat_map(|t| {
            let b = (pos0 as usize + t) * kv_dim;
            kc[b..b + kv_dim].to_vec()
        })
        .collect();
    let vc_rows: Vec<f32> = (0..t_len)
        .flat_map(|t| {
            let b = (pos0 as usize + t) * kv_dim;
            vc[b..b + kv_dim].to_vec()
        })
        .collect();

    let gq = dm.gq();
    let mut outv = vec![0f32; t_len * q_dim];
    for t in 0..t_len {
        for h in 0..dm.q_heads {
            let kh = h / gq;
            let kbase = kh * 256;
            let qbase = t * q_dim + h * 256;
            let lim = pos0 as i32 + t as i32 + 1;
            let mut sarr = vec![0f32; lim as usize];
            {
                let qs = &qh[qbase..qbase + 256];
                for row in 0..lim as usize {
                    let krow = &kc[row * kv_dim + kbase..row * kv_dim + kbase + 256];
                    let mut p = 0f32;
                    for d in 0..256 {
                        p += qs[d] * krow[d];
                    }
                    sarr[row] = p * 0.0625f32;
                }
            }
            // max — fmaxf 스트라이드+트리와 동일 값(유한·순서 무관).
            let gmax = sarr.iter().fold(-1e30f32, |a, &b| a.max(b));
            // ls — 스레드 보폭 순차 누산 + red[256] 트리(순서 미러 계약).
            let mut reds = [0f32; 256];
            for tid in 0..256 {
                let mut ls = 0f32;
                let mut i = tid;
                while i < lim as usize {
                    let e = gdn_expf(sarr[i] - gmax);
                    sarr[i] = e;
                    ls += e;
                    i += 256;
                }
                reds[tid] = ls;
            }
            let mut st = 128usize;
            while st > 0 {
                for tid in 0..st {
                    reds[tid] += reds[tid + st];
                }
                st >>= 1;
            }
            let wsum = reds[0];
            for tid in 0..256 {
                let mut acc = 0f32;
                for row in 0..lim as usize {
                    acc += sarr[row] * vc[row * kv_dim + kbase + tid];
                }
                let g = qg[t * qg_dim + h * 512 + 256 + tid];
                outv[t * q_dim + h * 256 + tid] = (acc / wsum) * (1.0 / (1.0 + gdn_expf(-g)));
            }
        }
    }
    AttnRefStages {
        qh,
        kc_rows,
        vc_rows,
        outv,
    }
}

// ── 어텐션 픽스처(실가중 q/k_norm + 결정론 시드 — G5 방법론 계승) ──

/// config.json 읽기 + AttnDims 유도(실측 차원 계약).
fn attn_dims_from_model(dir: &str) -> Result<AttnDims, String> {
    let cfg = std::fs::read_to_string(format!("{dir}/config.json"))
        .map_err(|e| format!("{dir}/config.json: {e}"))?;
    AttnDims::from_config(&cfg)
}

struct AttnFixture {
    dims: AttnDims,
    /// 대상 어텐션 층(말단 — ai = n_attn-1, il = ai*4+3).
    layer: usize,
    pos0: u32,
    qg: Vec<f32>,
    kin: Vec<f32>,
    vin: Vec<f32>,
    /// 히스토리만 담은 층 캐시 [cap*kv_dim](시딩용 — 신규행 미기록).
    kc_hist: Vec<f32>,
    vc_hist: Vec<f32>,
    qnw: Vec<f32>,
    knw: Vec<f32>,
    want: AttnRefStages,
}

impl AttnFixture {
    /// 상수는 dir 아카이브에서 실측(q/k_norm — bf16 w−1 저장 → +1.0,
    /// §3.4 규약), 잔여 층은 합성. 히스토리 pos0행은 합성 kin/vin을 오라클
    /// prep(kv 경로)로 rms+rope해 생성(실제 캐시 내용의 계급). 현 체인
    /// 입력 qg/kin/vin은 결정론 시드(hip 프로브 계급 스케일).
    // [rustfmt skip — G6 병리 계승(2026-10-04, 시간제한 계약)] 중첩 루프·클로저 밀착.
    #[rustfmt::skip]
    fn generate(
        dims: AttnDims,
        dir: &str,
        t_len: usize,
        pos0: u32,
        seed: u64,
    ) -> Result<Self, String> {
        use crate::rawcuda::exl3_cuda::StArchive;
        let layer = dims.n_attn - 1;
        let il = layer * 4 + 3;
        let lp = format!("model.language_model.layers.{il}.self_attn");
        let ar = StArchive::open(std::path::Path::new(dir))?;
        let readn = |name: &str| -> Result<Vec<f32>, String> {
            let dt = ar.dtype_of(name).ok_or_else(|| format!("{name} 없음"))?;
            let raw = ar.read(name)?;
            let mut v = st_to_f32(&raw, dt)?;
            if v.len() != 256 {
                return Err(format!("{name}: {} != 256", v.len()));
            }
            for f in v.iter_mut() {
                *f += 1.0; // 저장소 w−1 규약(§3.4) — 등록값은 +1
            }
            Ok(v)
        };
        let q_l = readn(&format!("{lp}.q_norm.weight"))?;
        let k_l = readn(&format!("{lp}.k_norm.weight"))?;
        let mut rng = Rng::new(seed);
        let unif = |rng: &mut Rng, amp: f64| ((rng.next_f64() * 2.0 - 1.0) * amp) as f32;
        // 전층 배열(잔여 층 합성) + 대상 층 실측 이식.
        let mut qnw = vec![0f32; dims.n_attn * 256];
        let mut knw = vec![0f32; dims.n_attn * 256];
        for a in 0..dims.n_attn {
            for i in 0..256 {
                qnw[a * 256 + i] = (0.5 + rng.next_f64()) as f32;
                knw[a * 256 + i] = (0.5 + rng.next_f64()) as f32;
            }
        }
        qnw[layer * 256..(layer + 1) * 256].copy_from_slice(&q_l);
        knw[layer * 256..(layer + 1) * 256].copy_from_slice(&k_l);
        let qnw_l: Vec<f32> = qnw[layer * 256..(layer + 1) * 256].to_vec();
        let knw_l: Vec<f32> = knw[layer * 256..(layer + 1) * 256].to_vec();

        // 히스토리 pos0행 — 오라클 kv prep로 생성.
        let kv_dim = dims.kv_dim();
        let mut kc_hist = vec![0f32; dims.cap * kv_dim];
        let mut vc_hist = vec![0f32; dims.cap * kv_dim];
        for p in 0..pos0 as usize {
            let krow: Vec<f32> = (0..kv_dim).map(|_| unif(&mut rng, 0.5)).collect();
            let vrow: Vec<f32> = (0..kv_dim).map(|_| unif(&mut rng, 0.5)).collect();
            for m in 0..dims.kv_heads {
                let head = attn_prep_head(&krow[m * 256..(m + 1) * 256], &knw_l, p as i32);
                kc_hist[p * kv_dim + m * 256..p * kv_dim + (m + 1) * 256].copy_from_slice(&head);
            }
            vc_hist[p * kv_dim..(p + 1) * kv_dim].copy_from_slice(&vrow);
        }

        // 현 체인 입력(T행).
        let qg: Vec<f32> = (0..t_len * dims.qg_dim()).map(|_| unif(&mut rng, 0.3)).collect();
        let kin: Vec<f32> = (0..t_len * kv_dim).map(|_| unif(&mut rng, 0.5)).collect();
        let vin: Vec<f32> = (0..t_len * kv_dim).map(|_| unif(&mut rng, 0.5)).collect();
        let mut kcf = kc_hist.clone();
        let mut vcf = vc_hist.clone();
        let want = attn_reference_chain(
            &dims, &qnw_l, &knw_l, &mut kcf, &mut vcf, &qg, &kin, &vin, pos0, t_len,
        );
        Ok(Self {
            dims,
            layer,
            pos0,
            qg,
            kin,
            vin,
            kc_hist,
            vc_hist,
            qnw,
            knw,
            want,
        })
    }
}

/// 케이스 1개 실행 + 4단계(qh·KC·VC·outv) 값 판정.
fn attn_run_case(
    dec: &mut Exl3CudaDecoder,
    tag: &str,
    mname: &str,
    dir: &str,
    t_len: usize,
    pos0: u32,
    seed: u64,
    dev: &str,
    fails: &mut Vec<String>,
    report: &mut String,
) -> Result<(), String> {
    let dims = attn_dims_from_model(dir)?;
    let fx = AttnFixture::generate(dims, dir, t_len, pos0, seed)?;
    let lay = fx.layer;
    dec.set_attn(dims, &fx.qnw, &fx.knw)?;
    dec.attn_seed_kv(lay, &fx.kc_hist, &fx.vc_hist)?;
    let got = dec.attn_chain_host(lay, t_len, &fx.qg, &fx.kin, &fx.vin, pos0)?;
    let (md_qh, n1) = maxdiff_nan(&got.qh, &fx.want.qh);
    let (md_kc, n2) = maxdiff_nan(&got.kc_rows, &fx.want.kc_rows);
    let (md_vc, n3) = maxdiff_nan(&got.vc_rows, &fx.want.vc_rows);
    let (md_ov, n4) = maxdiff_nan(&got.outv, &fx.want.outv);
    let nan = n1 + n2 + n3 + n4;
    let md = md_qh.max(md_kc).max(md_vc).max(md_ov);
    let pass = md <= ATTN_THRESH && nan == 0;
    println!(
        "device: {dev} | exl3-cuda-attn ({tag}) {mname} q_heads={} kv_heads={} d={} n_attn={} lay={lay} pos0={pos0} T={t_len}: qh maxdiff={md_qh:.3e} kc={md_kc:.3e} vc={md_vc:.3e} outv={md_ov:.3e} nan={nan} | {}",
        dims.q_heads,
        dims.kv_heads,
        dims.d,
        dims.n_attn,
        if pass { "PASS" } else { "FAIL" }
    );
    report.push_str(&format!("({tag}) worst={md:.3e}",));
    if !pass {
        fails.push(format!(
            "({tag}) qh={md_qh:.3e} kc={md_kc:.3e} vc={md_vc:.3e} outv={md_ov:.3e} nan={nan}"
        ));
    }
    Ok(())
}

/// exl3-cuda-attn — plans/124 G6 어텐션(prep+fwd3s) 값 maxdiff 판정
/// (임계 2e-7 — 계약 최tight). (i) 27B T=1(순차 디코드) · (ib) 27B T=8
/// (스펙 상한) · (iii) T>8 도메인 거부(모듈 Err + 커널 원시 발사
/// 센티널 불변 — 이중 강제 값 검증) · (ii) 35B-A3B T=8(실측 config.json
/// 차원 — q/kv 헤드 상이). 히스토리 33행 + pos0=33(디바이스 pp[0] 판독
/// 경로 값 검증). 하나라도 FAIL이면 Err(→ CLI 비영).
// [rustfmt skip — G6 병리 계승(2026-10-04, 시간제한 계약)] 원시 발사 인자 배열·중첩 판정 블록.
#[rustfmt::skip]
pub fn cuda_attn_check(dir27: &str, dir35: &str) -> Result<String, String> {
    let mut dec = Exl3CudaDecoder::empty()?;
    let dev = dec.device_name().to_string();
    let pos0: u32 = 33;
    let mut fails: Vec<String> = Vec::new();
    let mut report = String::new();

    attn_run_case(
        &mut dec, "i", "Qwen3.8-27B", dir27, 1, pos0, 0x170C_0DA0_0000_00A1, &dev, &mut fails,
        &mut report,
    )?;
    attn_run_case(
        &mut dec, "ib", "Qwen3.8-27B", dir27, 8, pos0, 0x170C_0DA0_0000_00A2, &dev, &mut fails,
        &mut report,
    )?;

    // (iii) 도메인 강제 — 모듈 진입(T=9)은 Err(커널 미발사), 커널 원시
    // 발사(t_len=9)는 조기복귀로 outv 미기록(센티널 불변). 양측 모두
    // 성립해야 깨끗한 거부다(부분 기록 오염 없음).
    {
        let dims = attn_dims_from_model(dir27)?;
        let qd = dims.q_dim();
        let t9 = 9usize;
        let (qg9, k9, v9) = (
            vec![0f32; t9 * dims.qg_dim()],
            vec![0f32; t9 * dims.kv_dim()],
            vec![0f32; t9 * dims.kv_dim()],
        );
        let rej = dec.attn_chain_host(0, t9, &qg9, &k9, &v9, pos0);
        let mod_rejected = matches!(&rej, Err(e) if e.contains("도메인"));
        println!(
            "device: {dev} | exl3-cuda-attn (iii) T=9 module entry: {} (Err 마커: 도메인)",
            if mod_rejected { "REJECTED" } else { "NOT-REJECTED" }
        );
        // 커널 직접 발사 — 직전 (ib) 체인의 버퍼(27B, t_cap=8) 재사용.
        let sentinel = -12345.0e0f32;
        // SAFETY: 센티널 f32 배열의 바이트 뷰(길이·정렬 일치).
        let sb: Vec<u8> = unsafe {
            std::slice::from_raw_parts(sentinel.to_le_bytes().as_ptr(), 4).repeat(8 * qd)
        };
        dec.cc.h2d(dec.doutv_a, &sb)?;
        let f = dec.cc.function("exl3_attn_fwd3s")?;
        let (mut tl, mut lay) = (t9 as i32, 0i32);
        let (mut qh, mut kvh, mut cp) = (dims.q_heads as i32, dims.kv_heads as i32, dims.cap as i32);
        let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5) =
            (dec.dqh_a, dec.dkc, dec.dvc, dec.dqg_a, dec.doutv_a, dec.dpp);
        let mut args: [*mut c_void; 11] = [
            (&mut a0) as *mut _ as *mut c_void,
            (&mut a1) as *mut _ as *mut c_void,
            (&mut a2) as *mut _ as *mut c_void,
            (&mut a3) as *mut _ as *mut c_void,
            (&mut a4) as *mut _ as *mut c_void,
            (&mut a5) as *mut _ as *mut c_void,
            (&mut tl) as *mut _ as *mut c_void,
            (&mut lay) as *mut _ as *mut c_void,
            (&mut qh) as *mut _ as *mut c_void,
            (&mut kvh) as *mut _ as *mut c_void,
            (&mut cp) as *mut _ as *mut c_void,
        ];
        dec.cc
            .launch(f, t9 as u32, dims.q_heads as u32, 256, &mut args)?;
        let mut rb = vec![0u8; 8 * qd * 4];
        dec.cc.d2h(&mut rb, dec.doutv_a)?;
        dec.cc.sync()?;
        // SAFETY: d2h 완료 후 재해석(길이·정렬 일치).
        let wrote = unsafe { std::slice::from_raw_parts(rb.as_ptr() as *const f32, 8 * qd) }
            .iter()
            .any(|&v| v.to_bits() != sentinel.to_bits());
        let guard_ok = mod_rejected && !wrote;
        println!(
            "device: {dev} | exl3-cuda-attn (iii) T=9 raw kernel launch: sentinel {} | {}",
            if wrote { "OVERWRITTEN" } else { "INTACT" },
            if guard_ok { "PASS" } else { "FAIL" }
        );
        report.push_str(&format!(" · (iii) rejected={guard_ok}"));
        if !guard_ok {
            fails.push(format!(
                "(iii) module_rejected={mod_rejected} kernel_wrote={wrote}"
            ));
        }
    }

    attn_run_case(
        &mut dec, "ii", "Qwen3.6-35B-A3B", dir35, 8, pos0, 0x170C_0DA0_0000_00A3, &dev, &mut fails,
        &mut report,
    )?;

    // ── (iv) 결함 21호 영구 회귀 검사(2026-10-05): 트윈 sincos vs f64 libm.
    // 커널·오라클이 같은 표를 공유하는 자기일관 구조의 맹점을 막는 교차검증
    // — pos 0..=4095 x rope theta 64 = 262,144 각도(전 사분면), 게이트 ≤ 2 f32 ulp.
    {
        let invpio2 = 6.36619772367581342433e-01f64;
        let mut max_ulp = 0u32;
        let mut quad = [0u64; 4];
        for pos in 0..=4095u32 {
            for tid in 0..64usize {
                let e = -(2.0 * tid as f64) / 64.0;
                let theta = (1e7f64.ln() * e).exp() as f32;
                let a = (pos as f32 * theta) as f64;
                let k = (a * invpio2 + 0.5).floor() as i64;
                quad[(k & 3) as usize] += 1;
                let (c, s) = attn_sincos_d(a);
                let ulp = |x: f32, y: f32| -> u32 {
                    (x.to_bits() as i64 - y.to_bits() as i64).unsigned_abs() as u32
                };
                max_ulp = max_ulp
                    .max(ulp(c as f32, a.cos() as f32))
                    .max(ulp(s as f32, a.sin() as f32));
            }
        }
        let cover = quad.iter().all(|&q| q > 0);
        println!(
            "device: {dev} | exl3-cuda-attn (iv) sincos-vs-libm: max_ulp={max_ulp} quad={quad:?} | {}",
            if max_ulp <= 2 && cover { "PASS" } else { "FAIL" }
        );
        report.push_str(&format!(" · (iv) sincos ulp={max_ulp}"));
        if max_ulp > 2 || !cover {
            fails.push(format!("(iv) sincos_vs_libm max_ulp={max_ulp} quad_cover={cover}"));
        }
    }

    if fails.is_empty() {
        Ok(format!("device: {dev} | {report} | ALL PASS"))
    } else {
        Err(format!(
            "exl3-cuda-attn 실패 — {} (device: {dev})",
            fails.join(", ")
        ))
    }
}

/// exl3-cuda-attn-neg — 음성대조(원장 17호: 계기도 스스로 검증).
/// 시나리오(결함 4호 — hip 배치 루프의 실제 사고 계급): pp[0]=32 기록 →
/// exl3_attn_pos_bump로 장치값 33 전진(그래프 내 전진 메커니즘) → 호스트
/// 사본은 32로 낡음. (a) 실경로(디바이스 33 판독)는 오라클과 정합(계기
/// 전제). (b) 쌍둥이 prep(호스트 사본 32) + 실 fwd3s(디바이스 33)는 KV
/// 기록 위치 1행 어긋남 → 종단 maxdiff가 임계 초과 — 디바이스 판독
/// 계약이 "값으로 검증 가능"함의 증명. 초과 시 NEG-DETECTED 마커와 함께
/// Err(→ CLI 비영 exit).
pub fn cuda_attn_negative_check(dir27: &str) -> Result<String, String> {
    let mut dec = Exl3CudaDecoder::empty()?;
    let dev = dec.device_name().to_string();
    let dims = attn_dims_from_model(dir27)?;
    let t_len = 4usize;
    let pos0_dev = 33u32;
    let pos0_host = 32u32;
    let fx = AttnFixture::generate(dims, dir27, t_len, pos0_dev, 0x170C_0DA0_0000_00A4)?;
    let lay = fx.layer;
    dec.set_attn(dims, &fx.qnw, &fx.knw)?;
    dec.attn_seed_kv(lay, &fx.kc_hist, &fx.vc_hist)?;

    // 장치 pp: 32 기록 → bump 1회 → 33(bump 실측 — d2h 판정).
    dec.attn_set_pos(pos0_host)?;
    let mut pb = [0u8; 4];
    dec.cc.d2h(&mut pb, dec.dpp)?;
    let v0 = u32::from_le_bytes(pb);
    dec.attn_pos_bump()?;
    dec.cc.d2h(&mut pb, dec.dpp)?;
    dec.cc.sync()?;
    let v1 = u32::from_le_bytes(pb);
    let bump_ok = v0 == pos0_host && v1 == pos0_dev;
    println!(
        "device: {dev} | exl3-cuda-attn (prep) pos_bump device readback: {v0} -> {v1} | {}",
        if bump_ok { "PASS" } else { "FAIL" }
    );

    // (a) 실경로 — 디바이스 판독(33) + 동일 입력 → 오라클 정합(전제).
    let got_a = dec.attn_chain_host(lay, t_len, &fx.qg, &fx.kin, &fx.vin, pos0_dev)?;
    let (md_a, nan_a) = maxdiff_nan(&got_a.outv, &fx.want.outv);
    println!(
        "device: {dev} | exl3-cuda-attn (iva) real device-read path pp[0]={pos0_dev}: outv maxdiff={md_a:.3e} nan={nan_a} | {}",
        if md_a <= ATTN_THRESH && nan_a == 0 {
            "PASS(sanity)"
        } else {
            "FAIL"
        }
    );

    // (b) 쌍둥이 — 호스트 사본 32로 KV 기록 + 디바이스 33 판독 fwd3s.
    // 캐시 재시딩((a)이 신규행을 기록했으므로 히스토리 상태로 복원).
    dec.attn_seed_kv(lay, &fx.kc_hist, &fx.vc_hist)?;
    let got_b = dec.attn_chain_host_hostpos(lay, t_len, &fx.qg, &fx.kin, &fx.vin, pos0_host)?;
    let (md_b, nan_b) = maxdiff_nan(&got_b, &fx.want.outv);
    println!(
        "device: {dev} | exl3-cuda-attn (ivb) negative control host-pos copy ({pos0_host}) vs device pp[0] ({pos0_dev}): outv maxdiff={md_b:.3e} nan={nan_b} | FAIL(expected)"
    );
    let det = md_b > ATTN_THRESH && bump_ok;

    if det {
        Err(format!(
            "NEG-DETECTED host-pos maxdiff={md_b:.3e} > {ATTN_THRESH:.0e} — 검증계기 정상(결함 4호: 디바이스 pp[0] 판독 계약의 값 검증력 입증, bump {v0}->{v1})"
        ))
    } else {
        Err(format!(
            "NEG-MISSED host-pos maxdiff={md_b:.3e} <= {ATTN_THRESH:.0e} (bump_ok={bump_ok}) — 검증계기 결함: 호스트 사본 이격이 탐지되지 않음"
        ))
    }
}
