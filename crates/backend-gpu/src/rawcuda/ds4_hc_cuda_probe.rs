//! [하네스 저작 원칙 — plans/129-cuda-only C1(원본 129 A10) 체크리스트]
//! ① 선행 단계 공유 버퍼 오염 점검: ds4-hc 케이스마다 잔류·가중치를
//!    재업로드한다(모듈 작업 버퍼는 현 t 행 전체를 매 체인마다 덮어쓴다).
//! ② 형상은 픽스처 config.json에서 자동 확정(DeepSeek-V4 Vision-Exp 실측
//!    hc_mult=4 · hidden_size=4096 · rms_norm_eps=1e-20 · hc_eps=1e-6 ·
//!    hc_sinkhorn_iters=20)+ T=1..3 토큰축.
//! ③ 캡처-재생: 실측 픽스처 hc 가중치(EXL3 샤드 safetensors — 헤더
//!    판독 후 data_offsets 오프셋 직독, 전량 적재 금지) + 합성 잔류로
//!    구동. hc_{attn,ffn}_{fn,base,scale}·hc_head_{fn,base,scale} 은
//!    전부 F32 plain(실측 2026-10-05: fn[24,16384]·base[24]·scale[3],
//!    헤드 fn[4,16384]·base[4]·scale[1]).
//! ④ 종단 값이 유일 불변량: 판정은 hc_pre(y·post·comb)·hc_post(X')·
//!    hc_head(y) 산출의 값 maxdiff+비트 불일치 수(중간 단계는 진단).
//!    Sinkhorn 이중확률 근사(행·열합 ≈1)·반복순서 민감성도 긍정 케이스
//!    에서 명시 검증(근사 잔차 실측 ~7.5e-2 — 20iter 근사 계약).
//!
//! [오라클 — core deepseek4 참조 직이식(값 maxdiff 판정의 유일 기준,
//! plans/124 §6). 인용은 전부 워크트리 기준 줄번호, 2026-10-05]
//! - crates/core/src/deepseek4/stages/hc.rs — hc_split_sinkhorn L31-112
//!   (pre L52-54 · post L55-58 · raw L59-64 · softmax_rows L67-78 ·
//!   +eps→col L81-95 · 19×{row;col} L96-112) · hc_pre L115-152 ·
//!   hc_post L155-181 · hc_head L185-214(frame.rs L173-184 사용).
//! - crates/core/src/deepseek4/ops.rs — rms_scale L48-55(**순수 f32 순차
//!   제곱합** — qwen4exp sq_sum f64 세그먼트와 다른 deepseek4 계약) ·
//!   bf16_round L17-22(RNE 비트 경로).
//! - crates/core/src/ops.rs — exp_cr L52-119(f64 fma 호너 13단 — 커널
//!   ds4_exp_cr 과 리터럴 동일) · sigmoid L133-133(1/(1+exp_cr(-x))).
//!
//! [정합 원장 요약 — 전 케이스 비트동일 목표(판정 기준)]
//! (i) attn-hc 실측 가중치 T=3·T=1 · (ii) ffn-hc 실측 가중치 T=2
//! (hc_pre+hc_post 체인) · (iii) hc_head 실측 가중치 T=2 전부
//! bitdiff=0 · nan=0 기대. 1 ulp 이내 잔차가 측정되면 원장 갱신 커밋으로
//! 문서화한다(<=1 ulp 허용 문서화 계약 — 정합 목표 비트동일).
//!
//! [음성대조 — 원장 17호(계기 자체 검증)] 3종 모두 NEG-DETECTED 필
//! (실측 효과 크기, f64 추정 2026-10-05): (a) Sinkhorn 반복순서 교환
//!     (19×{col;row} — comb maxdiff ~7.4e-2), (b) comb 전치(hc_post 에
//!     comb^T 적용 — ~1.0e-1), (c) eps 제거(pre +eps·Sinkhorn eps 전부
//!     0 — comb ~1.1e-3 + pre 1e-6 편차).
//!
//! [속도] 측정 대기 sm_80 — CMP 170HX 미도착(plans/124 §0). 개발기
//! (RTX 4070 SUPER) 타이밍 금지.
//!
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 금지.

use crate::rawcuda::ds4_hc_cuda::{DS4_HC_HEAD_IL, Ds4HcCuda, Ds4HcDims};
use crate::rawcuda::exl3_cuda::{JParser, JVal};
use crate::rawcuda::exl3_cuda_probe::{gen_unif, maxdiff_nan};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// 기본 픽스처 경로(과제 지정 — 프로브 인자로 대체 가능).
pub const DS4_HC_EXL3_DIR: &str = "D:/models/DeepSeek-V4-Flash-Vision-Exp-exl3-3.04bpw";

/// 음성대조 탐지 임계(fn-hc HC_THRESH 계급). 실측 효과 최소 ~1.1e-3
/// (eps 제거) ≫ 임계 — 탐지 여유 300배 이상.
const DS4_HC_NEG_THRESH: f32 = 3e-6;

/// Sinkhorn 20iter 이중확률 **근사** 잔차 상한(행·열합 |Σ−1|) — 실측
/// ~7.5e-2(2026-10-05, 실측 가중치·합성 잔류 f64 추정). 근사 계약
/// (hc.rs L96-112 주석 "이중확률 근사")의 산업 판정 한계치.
const DS4_HC_DS_TOL: f32 = 0.25;

// ── core 미러 오라클 — 인용 줄번호는 헤드 [오라클] 항 ──

/// ops.rs exp_cr(L52-119) 직이식 — f64 fma 호너 13단 + 2^k 비트 재구성.
/// 커널 ds4_exp_cr(assets/ds4_hc.cu)과 리터럴까지 1:1(mul_add ≡
/// __fma_rn — 올림-정확 단일 반올림이라 양측 비트동일).
fn core_exp_cr(x: f32) -> f32 {
    let xd = x as f64;
    if xd > 88.72 {
        return f32::INFINITY;
    }
    if xd < -103.97 {
        return 0.0;
    }
    const LN2_HI: f64 = 6.931_471_803_691_238e-1;
    const LN2_LO: f64 = 1.908_214_929_270_587_7e-10;
    const INV_LN2: f64 = std::f64::consts::LOG2_E;
    let kd = (xd * INV_LN2).round_ties_even();
    let k = kd as i64;
    let mut r = (-kd).mul_add(LN2_HI, xd);
    r = (-kd).mul_add(LN2_LO, r);
    let mut p = 1.0f64 / 1307674368000.0;
    p = p.mul_add(r, 1.0 / 479001600.0);
    p = p.mul_add(r, 1.0 / 39916800.0);
    p = p.mul_add(r, 1.0 / 3628800.0);
    p = p.mul_add(r, 1.0 / 362880.0);
    p = p.mul_add(r, 1.0 / 40320.0);
    p = p.mul_add(r, 1.0 / 5040.0);
    p = p.mul_add(r, 1.0 / 720.0);
    p = p.mul_add(r, 1.0 / 120.0);
    p = p.mul_add(r, 1.0 / 24.0);
    p = p.mul_add(r, 1.0 / 6.0);
    p = p.mul_add(r, 0.5);
    p = p.mul_add(r, 1.0);
    p = p.mul_add(r, 1.0);
    if k > 127 {
        return f32::INFINITY;
    }
    let scale = f64::from_bits(((k + 1023) as u64) << 52);
    (p * scale) as f32
}

/// ops.rs sigmoid(L133) 직이식 — 1/(1+exp_cr(-x)).
fn core_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + core_exp_cr(-x))
}

/// deepseek4/ops.rs bf16_round(L17-22) 직이식 — RNE 1회 비트 경로.
fn core_bf16_round(x: f32) -> f32 {
    let b = x.to_bits();
    let hi = ((b >> 16) as u16) as u32 & 1;
    f32::from_bits(((b + 0x7FFF + hi) >> 16) << 16)
}

/// deepseek4/ops.rs rms_scale(L48-55) 직이식 — **순수 f32 순차 제곱합**
/// (세그먼트 분할·f64 결합 없음 — deepseek4 계약, 커널 1스레드 미러).
fn core_rms_scale_f32(x: &[f32], eps: f32) -> f32 {
    let mut sum = 0.0f32;
    for &v in x {
        sum += v * v;
    }
    1.0 / (sum / x.len() as f32 + eps).sqrt()
}

/// hc.rs hc_split_sinkhorn(L31-112) 직이식. `order_swapped`=true 면
/// 19× 반복을 {col;row} 순서로(음성대조 (a)), eps 는 호출부가 0.0 을
/// 넘기면 pre·Sinkhorn 전체에서 제거(음성대조 (c)).
fn ref_split_sinkhorn(
    mixes: &[f32],
    scale: &[f32; 3],
    base: &[f32],
    hc: usize,
    iters: usize,
    eps: f32,
    order_swapped: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    debug_assert_eq!(mixes.len(), (2 + hc) * hc);
    let mut pre = vec![0.0f32; hc];
    let mut post = vec![0.0f32; hc];
    let mut comb = vec![0.0f32; hc * hc];
    for j in 0..hc {
        pre[j] = core_sigmoid(mixes[j] * scale[0] + base[j]) + eps;
    }
    for j in 0..hc {
        post[j] = 2.0 * core_sigmoid(mixes[hc + j] * scale[1] + base[hc + j]);
    }
    for j in 0..hc {
        for k in 0..hc {
            comb[j * hc + k] = mixes[2 * hc + j * hc + k] * scale[2] + base[2 * hc + j * hc + k];
        }
    }
    // 1) 행 softmax — 행최대 분산(L67-78).
    for j in 0..hc {
        let row = &mut comb[j * hc..(j + 1) * hc];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = core_exp_cr(*v - m);
            sum += *v;
        }
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
    // 2) +eps → /(colsum+eps)(L81-95).
    for v in comb.iter_mut() {
        *v += eps;
    }
    let mut col = vec![0.0f32; hc];
    for j in 0..hc {
        for k in 0..hc {
            col[k] += comb[j * hc + k];
        }
    }
    for j in 0..hc {
        for k in 0..hc {
            comb[j * hc + k] /= col[k] + eps;
        }
    }
    // 3) (iters-1)× { /(rowsum+eps); /(colsum+eps) }(L96-112) — 교환 변형은
    // { col; row } 순서로.
    let mut row = vec![0.0f32; hc];
    for _ in 1..iters {
        let row_pass = |comb: &mut [f32], row: &mut [f32], hc: usize| {
            row.fill(0.0);
            for j in 0..hc {
                for k in 0..hc {
                    row[j] += comb[j * hc + k];
                }
            }
            for j in 0..hc {
                for k in 0..hc {
                    comb[j * hc + k] /= row[j] + eps;
                }
            }
        };
        let col_pass = |comb: &mut [f32], col: &mut [f32], hc: usize| {
            col.fill(0.0);
            for j in 0..hc {
                for k in 0..hc {
                    col[k] += comb[j * hc + k];
                }
            }
            for j in 0..hc {
                for k in 0..hc {
                    comb[j * hc + k] /= col[k] + eps;
                }
            }
        };
        if order_swapped {
            col_pass(&mut comb, &mut col, hc);
            row_pass(&mut comb, &mut row, hc);
        } else {
            row_pass(&mut comb, &mut row, hc);
            col_pass(&mut comb, &mut col, hc);
        }
    }
    (pre, post, comb)
}

/// hc.rs hc_pre(L115-152) 직이식 — (y, post, comb) 반환. 변형 플래그는
/// 음성대조용(order_swapped·eps 드롭).
fn hc_pre_ref(
    x: &[f32],
    d: usize,
    hc: usize,
    fns: &[f32],
    base: &[f32],
    scale: &[f32],
    norm_eps: f32,
    hc_eps: f32,
    iters: usize,
    order_swapped: bool,
    drop_eps: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mix_hc = (2 + hc) * hc;
    let rsqrt = core_rms_scale_f32(x, norm_eps);
    let mut mixes = vec![0.0f32; mix_hc];
    for (i, m) in mixes.iter_mut().enumerate() {
        let fr = &fns[i * (hc * d)..(i + 1) * (hc * d)];
        let mut acc = 0.0f32;
        for (a, b) in x.iter().zip(fr.iter()) {
            acc += a * b;
        }
        *m = acc * rsqrt;
    }
    let scale3 = [scale[0], scale[1], scale[2]];
    let eps = if drop_eps { 0.0 } else { hc_eps };
    let (pre, post, comb) =
        ref_split_sinkhorn(&mixes, &scale3, base, hc, iters, eps, order_swapped);
    let mut y = vec![0.0f32; d];
    for j in 0..hc {
        let xj = &x[j * d..(j + 1) * d];
        let pj = pre[j];
        for (yi, &xv) in y.iter_mut().zip(xj.iter()) {
            *yi += pj * xv;
        }
    }
    for yi in y.iter_mut() {
        *yi = core_bf16_round(*yi);
    }
    (y, post, comb)
}

/// hc.rs hc_post(L155-181) 직이식 — `transpose`=true 면 comb^T 적용
/// (음성대조 (b)).
fn hc_post_ref(
    f: &[f32],
    residual: &[f32],
    post: &[f32],
    comb: &[f32],
    d: usize,
    hc: usize,
    transpose: bool,
) -> Vec<f32> {
    let mut y = vec![0.0f32; hc * d];
    for j in 0..hc {
        let pj = post[j];
        for i in 0..d {
            y[j * d + i] = pj * f[i];
        }
        for k in 0..hc {
            let c = if transpose {
                comb[k * hc + j]
            } else {
                comb[j * hc + k]
            };
            let rk = &residual[k * d..(k + 1) * d];
            for i in 0..d {
                y[j * d + i] += c * rk[i];
            }
        }
        for i in 0..d {
            y[j * d + i] = core_bf16_round(y[j * d + i]);
        }
    }
    y
}

/// hc.rs hc_head(L185-214) 직이식 — fn[4·d]·base[4]·scale[1] 헤드 변형.
fn hc_head_ref(
    x: &[f32],
    d: usize,
    hc: usize,
    fns: &[f32],
    base: &[f32],
    scale: &[f32],
    norm_eps: f32,
    hc_eps: f32,
) -> Vec<f32> {
    let rsqrt = core_rms_scale_f32(x, norm_eps);
    let mut pre = vec![0.0f32; hc];
    for j in 0..hc {
        let fr = &fns[j * (hc * d)..(j + 1) * (hc * d)];
        let mut acc = 0.0f32;
        for (a, b) in x.iter().zip(fr.iter()) {
            acc += a * b;
        }
        pre[j] = core_sigmoid(acc * rsqrt * scale[0] + base[j]) + hc_eps;
    }
    let mut y = vec![0.0f32; d];
    for j in 0..hc {
        let xj = &x[j * d..(j + 1) * d];
        for (yi, &xv) in y.iter_mut().zip(xj.iter()) {
            *yi += pre[j] * xv;
        }
    }
    for yi in y.iter_mut() {
        *yi = core_bf16_round(*yi);
    }
    y
}

// ── 합성 자료(결정론 시드 — splitmix64 미러 재사용) ──

/// 잔류 행 [t][hc_dim] — ±amp(임베드 방송 계급 ±0.5).
fn gen_res_hc(t: usize, hcd: usize, seed: u64, amp: f64) -> Vec<Vec<f32>> {
    (0..t)
        .map(|ti| gen_unif(hcd, seed + ti as u64, amp))
        .collect()
}

/// 비트 불일치 수(to_bits 동일 판정 — NaN 은 maxdiff_nan 이 별도 집계).
fn bitdiff(got: &[f32], want: &[f32]) -> usize {
    got.iter()
        .zip(want)
        .filter(|(g, w)| g.to_bits() != w.to_bits())
        .count()
}

/// 2차원 산출 판독 — 행 접합 후 (maxdiff, bitdiff, nan).
fn flat_judge(got: &[Vec<f32>], want: &[Vec<f32>]) -> (f32, usize, usize) {
    let g: Vec<f32> = got.iter().flatten().copied().collect();
    let w: Vec<f32> = want.iter().flatten().copied().collect();
    let (md, nan) = maxdiff_nan(&g, &w);
    (md, bitdiff(&g, &w), nan)
}

// ── 실측 픽스처: EXL3 샤드 safetensors 오프셋 직독 ──

/// 요청 텐서만 샤드 헤더에서 찾아 오프셋 직독(F32 고정 — hc 파라미터
/// 전부 F32, 픽스처 변경 가드). 전량 적재 금지 계약 — 헤더 판독 후
/// data_offsets 위치의 정확 바이트만(헤더 ~0.6-1.0MB/샤드, 16샤드).
fn ds4_hc_fixture(
    dir: &Path,
    names: &[&str],
    shapes: &[(&[usize], u64)], // (기대 shape, 기대 원소수) — name 과 동순
) -> Result<HashMap<String, Vec<f32>>, String> {
    let mut shards: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("model-") && n.ends_with(".safetensors"))
                .unwrap_or(false)
        })
        .collect();
    shards.sort();
    if shards.is_empty() {
        return Err(format!("ds4-hc: {dir:?} 에 model-*.safetensors 샤드 없음"));
    }
    let mut want: HashMap<&str, usize> = names.iter().enumerate().map(|(i, n)| (*n, i)).collect();
    let mut out: HashMap<String, Vec<f32>> = HashMap::new();
    for sp in shards.iter() {
        if want.is_empty() {
            break;
        }
        let mut f = std::fs::File::open(sp).map_err(|e| format!("{}: {e}", sp.display()))?;
        let mut lenb = [0u8; 8];
        f.read_exact(&mut lenb).map_err(|e| e.to_string())?;
        let hlen = u64::from_le_bytes(lenb);
        if hlen == 0 || hlen > (1 << 24) {
            return Err(format!("ds4-hc: 헤더 길이 {hlen} — 가드(실측 ~1MB)"));
        }
        let mut hb = vec![0u8; hlen as usize];
        f.read_exact(&mut hb).map_err(|e| e.to_string())?;
        let data_base = 8 + hlen;
        let v = JParser { b: &hb, p: 0 }.parse()?;
        let obj = v.as_obj().ok_or("ds4-hc: 헤더가 객체 아님")?;
        for (name, tv) in obj {
            let Some(&wi) = want.get(name.as_str()) else {
                continue;
            };
            let dt = tv
                .get("dtype")
                .and_then(JVal::as_str)
                .ok_or("ds4-hc: dtype 없음")?;
            if dt != "F32" {
                return Err(format!(
                    "ds4-hc: {name} dtype {dt} — F32 고정(픽스처 변경 가드)"
                ));
            }
            let shape = tv
                .get("shape")
                .and_then(JVal::as_arr)
                .ok_or("ds4-hc: shape 없음")?;
            let sh: Vec<usize> = shape
                .iter()
                .filter_map(JVal::as_f64)
                .map(|v| v as usize)
                .collect();
            let offs = tv
                .get("data_offsets")
                .and_then(JVal::as_arr)
                .ok_or("ds4-hc: data_offsets 없음")?;
            let g64 = |i: usize| {
                offs.get(i)
                    .and_then(JVal::as_f64)
                    .map(|v| v as u64)
                    .ok_or_else(|| format!("ds4-hc: data_offsets[{i}] 없음"))
            };
            let (b, e) = (g64(0)?, g64(1)?);
            let (want_shape, want_len) = shapes[wi];
            if sh != want_shape || (e - b) != want_len * 4 {
                return Err(format!(
                    "ds4-hc: {name} shape {sh:?} offsets {b}..{e} — 기대 {want_shape:?}·{want_len}원소(dims 불일치 가드)"
                ));
            }
            f.seek(SeekFrom::Start(data_base + b))
                .map_err(|e| e.to_string())?;
            let mut buf = vec![0u8; (e - b) as usize];
            f.read_exact(&mut buf).map_err(|e| e.to_string())?;
            let vals: Vec<f32> = buf
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            if vals.iter().any(|v| !v.is_finite()) {
                return Err(format!("ds4-hc: {name} 비유한 값 — 픽스처 가드"));
            }
            out.insert(name.clone(), vals);
            want.remove(name.as_str());
        }
    }
    let missing: Vec<&str> = want.keys().copied().collect();
    if !missing.is_empty() {
        return Err(format!("ds4-hc: 픽스처 텐서 없음: {missing:?}"));
    }
    Ok(out)
}

// ── 메인 프로브 ──

/// ds4-hc — DeepSeek-V4 mHC 스테이지 값 판정(비트동일 기준). 케이스:
/// (i) 실측 attn 가중치 T=3·T=1 · (ii) 실측 ffn 가중치 T=2 hc_pre+
/// hc_post 체인 · (iii) 실측 hc_head 가중치 T=2 · (iv) Sinkhorn 이중확률
/// 근사 + 반복순서 민감성. 전 케이스 bitdiff=0·nan=0 요구.
pub fn cuda_ds4_hc_check(dir: &str) -> Result<String, String> {
    let cfg_path = Path::new(dir).join("config.json");
    let cfg = std::fs::read_to_string(&cfg_path).map_err(|e| {
        format!(
            "{}: {e}(픽스처 계약 — config.json 필수)",
            cfg_path.display()
        )
    })?;
    let dims = Ds4HcDims::from_config(&cfg)?;
    let (hc, n, hcd, mix_rows) = (dims.hc, dims.n_embd, dims.hc_dim(), dims.mix_rows());
    // 픽스처: L0 attn(해시 SWA층 — hc 구조는 전층 동일)·L2 ffn(CSA층)·
    // 헤드(최상위). fn[24,16384]·base[24]·scale[3] / 헤드 fn[4,16384]·
    // base[4]·scale[1] — 실측 형상 가드.
    let names = [
        "layers.0.hc_attn_fn",
        "layers.0.hc_attn_base",
        "layers.0.hc_attn_scale",
        "layers.2.hc_ffn_fn",
        "layers.2.hc_ffn_base",
        "layers.2.hc_ffn_scale",
        "hc_head_fn",
        "hc_head_base",
        "hc_head_scale",
    ];
    let shapes: [(&[usize], u64); 9] = [
        (&[mix_rows, hcd], (mix_rows * hcd) as u64),
        (&[mix_rows], mix_rows as u64),
        (&[3], 3),
        (&[mix_rows, hcd], (mix_rows * hcd) as u64),
        (&[mix_rows], mix_rows as u64),
        (&[3], 3),
        (&[hc, hcd], (hc * hcd) as u64),
        (&[hc], hc as u64),
        (&[1], 1),
    ];
    let fx = ds4_hc_fixture(Path::new(dir), &names, &shapes)?;
    let g = |k: &str| fx.get(k).map(|v| v.as_slice()).unwrap_or(&[]);
    let (a_fn, a_base, a_scale) = (
        g("layers.0.hc_attn_fn"),
        g("layers.0.hc_attn_base"),
        g("layers.0.hc_attn_scale"),
    );
    let (f_fn, f_base, f_scale) = (
        g("layers.2.hc_ffn_fn"),
        g("layers.2.hc_ffn_base"),
        g("layers.2.hc_ffn_scale"),
    );
    let (h_fn, h_base, h_scale) = (g("hc_head_fn"), g("hc_head_base"), g("hc_head_scale"));

    let mut modl = Ds4HcCuda::new(dims)?;
    let dev = modl.device_name().to_string();
    let mut fails: Vec<String> = Vec::new();
    let mut report = String::new();

    // (i) attn-hc 실측 가중치 — T=3 + T=1(토큰축 배치 가드).
    modl.register(0, "attn", a_fn, a_base, a_scale)?;
    let mut first_comb: Option<Vec<Vec<f32>>> = None;
    for (t, tag) in [(3usize, "T=3"), (1, "T=1")] {
        let res = gen_res_hc(t, hcd, 0x5EED_F00D_0000_1100, 0.5);
        let mut wy = Vec::new();
        let mut wpost = Vec::new();
        let mut wcomb = Vec::new();
        for r in &res {
            let (y, post, comb) = hc_pre_ref(
                r,
                n,
                hc,
                a_fn,
                a_base,
                a_scale,
                dims.norm_eps,
                dims.hc_eps,
                dims.iters,
                false,
                false,
            );
            wy.push(y);
            wpost.push(post);
            wcomb.push(comb);
        }
        let (gy, gpost, gcomb) = modl.hc_pre(0, "attn", &res)?;
        let (mdy, bdy, ny) = flat_judge(&gy, &wy);
        let (mdp, bdp, np_) = flat_judge(&gpost, &wpost);
        let (mdc, bdc, nc) = flat_judge(&gcomb, &wcomb);
        let pass = bdy == 0 && ny == 0 && bdp == 0 && np_ == 0 && bdc == 0 && nc == 0;
        println!(
            "device: {dev} | ds4-hc (i) attn L0 {tag}: y maxdiff={mdy:.3e} bitdiff={bdy}/{} nan={ny} | post bitdiff={bdp}/{} | comb bitdiff={bdc}/{} nan={nc} | {}",
            t * n,
            t * hc,
            t * hc * hc,
            if pass { "PASS" } else { "FAIL" }
        );
        report.push_str(&format!("(i {tag}) y={mdy:.3e}/{bdy}b c={mdc:.3e}/{bdc}b"));
        if !pass {
            fails.push(format!(
                "(i {tag}) y {mdy:.3e}/{bdy}b nan={ny} post {mdp:.3e}/{bdp}b comb {mdc:.3e}/{bdc}b nan={nc}"
            ));
        }
        if t == 3 {
            first_comb = Some(gcomb);
        }
    }

    // (iv) Sinkhorn 근사 성질 — 모듈 comb(T=3, 토큰 0): 양수 + 이중확률
    // 근사(행·열합, 허용 DS_TOL) + 반복순서 민감성(교환 변형 comb 와
    // maxdiff > 임계 — 순서가 판정에 고정되어 있음을 증명).
    if let Some(comb3) = &first_comb {
        let c0 = &comb3[0];

        let mut rs_max = 0.0f32;
        let mut cs_max = 0.0f32;
        for j in 0..hc {
            let rs: f32 = c0[j * hc..(j + 1) * hc].iter().sum();
            rs_max = rs_max.max((rs - 1.0).abs());
        }
        for k in 0..hc {
            let cs: f32 = (0..hc).map(|j| c0[j * hc + k]).sum();
            cs_max = cs_max.max((cs - 1.0).abs());
        }
        let pos_ok = c0.iter().all(|&v| v > 0.0);
        // 교환 변형 comb — 동일 토큰 mixes 로 재계산.
        let res0 = gen_res_hc(1, hcd, 0x5EED_F00D_0000_1100, 0.5);
        let x0 = &res0[0];
        let rsqrt = core_rms_scale_f32(x0, dims.norm_eps);
        let mut mixes = vec![0.0f32; mix_rows];
        for (i, m) in mixes.iter_mut().enumerate() {
            let fr = &a_fn[i * hcd..(i + 1) * hcd];
            let mut acc = 0.0f32;
            for (a, b) in x0.iter().zip(fr.iter()) {
                acc += a * b;
            }
            *m = acc * rsqrt;
        }
        let s3 = [a_scale[0], a_scale[1], a_scale[2]];
        let (_, _, swapped) =
            ref_split_sinkhorn(&mixes, &s3, a_base, hc, dims.iters, dims.hc_eps, true);
        let got: Vec<f32> = comb3.iter().flatten().copied().collect();
        let (mdo, _) = maxdiff_nan(&got, &swapped);
        let pass =
            pos_ok && rs_max < DS4_HC_DS_TOL && cs_max < DS4_HC_DS_TOL && mdo > DS4_HC_NEG_THRESH;
        println!(
            "device: {dev} | ds4-hc (iv) sinkhorn: rowsum|Δ|={rs_max:.3e} colsum|Δ|={cs_max:.3e} (tol {DS4_HC_DS_TOL:.0e}, 20iter 근사) pos={} | order-swap comb maxdiff={mdo:.3e} > {DS4_HC_NEG_THRESH:.0e} | {}",
            pos_ok,
            if pass { "PASS" } else { "FAIL" }
        );
        report.push_str(&format!(
            " · (iv sink) ds={rs_max:.1e}/{cs_max:.1e} swap={mdo:.1e}"
        ));
        if !pass {
            fails.push(format!(
                "(iv sink) rowsum={rs_max:.3e} colsum={cs_max:.3e} pos={pos_ok} swap={mdo:.3e}"
            ));
        }
    }

    // (ii) ffn-hc 실측 가중치 T=2 — hc_pre → F → hc_post 체인(X' 종단).
    modl.register(2, "ffn", f_fn, f_base, f_scale)?;
    let res2 = gen_res_hc(2, hcd, 0x5EED_F00D_0000_2100, 0.5);
    let mut wy = Vec::new();
    let mut wpost = Vec::new();
    let mut wcomb = Vec::new();
    for r in &res2 {
        let (y, post, comb) = hc_pre_ref(
            r,
            n,
            hc,
            f_fn,
            f_base,
            f_scale,
            dims.norm_eps,
            dims.hc_eps,
            dims.iters,
            false,
            false,
        );
        wy.push(y);
        wpost.push(post);
        wcomb.push(comb);
    }
    let (gy, gpost, gcomb) = modl.hc_pre(2, "ffn", &res2)?;
    let (mdy, bdy, ny) = flat_judge(&gy, &wy);
    let (mdp, bdp, _) = flat_judge(&gpost, &wpost);
    let (mdc, bdc, _) = flat_judge(&gcomb, &wcomb);
    let f_syn: Vec<Vec<f32>> = (0..2)
        .map(|ti| gen_unif(n, 0x5EED_F00D_0000_2110 + ti, 0.5))
        .collect();
    let mut want_post = Vec::new();
    for ti in 0..2 {
        want_post.push(hc_post_ref(
            &f_syn[ti], &res2[ti], &wpost[ti], &wcomb[ti], n, hc, false,
        ));
    }
    let got_post = modl.hc_post(&f_syn, &res2, &gpost, &gcomb)?;
    let (mdo, bdo, no) = flat_judge(&got_post, &want_post);
    let pass = bdy == 0 && ny == 0 && bdp == 0 && bdc == 0 && bdo == 0 && no == 0;
    println!(
        "device: {dev} | ds4-hc (ii) ffn L2 T=2: pre y maxdiff={mdy:.3e} bitdiff={bdy}/{} post/comb bitdiff={bdp}/{bdc} | hc_post X' maxdiff={mdo:.3e} bitdiff={bdo}/{} nan={no} | {}",
        2 * n,
        2 * hcd,
        if pass { "PASS" } else { "FAIL" }
    );
    report.push_str(&format!(
        " · (ii ffn) y={mdy:.3e}/{bdy}b X'={mdo:.3e}/{bdo}b"
    ));
    if !pass {
        fails.push(format!(
            "(ii ffn) y {mdy:.3e}/{bdy}b nan={ny} post {mdp:.3e}/{bdp}b comb {mdc:.3e}/{bdc}b X' {mdo:.3e}/{bdo}b nan={no}"
        ));
    }

    // (iii) hc_head 실측 가중치 T=2 — 헤드 변형(fn[4·d]·scale[1]).
    modl.register(DS4_HC_HEAD_IL, "head", h_fn, h_base, h_scale)?;
    let resh = gen_res_hc(2, hcd, 0x5EED_F00D_0000_3100, 0.4);
    let want_h: Vec<Vec<f32>> = resh
        .iter()
        .map(|r| hc_head_ref(r, n, hc, h_fn, h_base, h_scale, dims.norm_eps, dims.hc_eps))
        .collect();
    let got_h = modl.hc_head(&resh)?;
    let (mdh, bdh, nh) = flat_judge(&got_h, &want_h);
    let pass = bdh == 0 && nh == 0;
    println!(
        "device: {dev} | ds4-hc (iii) hc_head T=2: maxdiff={mdh:.3e} bitdiff={bdh}/{} nan={nh} | {}",
        2 * n,
        if pass { "PASS" } else { "FAIL" }
    );
    report.push_str(&format!(" · (iii head) {mdh:.3e}/{bdh}b"));
    if !pass {
        fails.push(format!("(iii head) {mdh:.3e}/{bdh}b nan={nh}"));
    }

    if fails.is_empty() {
        Ok(format!(
            "device: {dev} | dims hc={hc} n={n} eps(norm/hc)={}/{} iters={} | {report} | ALL PASS (비트동일 기준)",
            dims.norm_eps, dims.hc_eps, dims.iters
        ))
    } else {
        Err(format!(
            "ds4-hc 실패 — {} (device: {dev}, dims hc={hc} n={n})",
            fails.join(", ")
        ))
    }
}

/// ds4-hc-neg — 음성대조 3종(원장 17호: 검증 계기도 스스로 검증).
/// (a) Sinkhorn 반복순서 교환(19×{col;row}) — comb 에서 탐지.
/// (b) comb 전치(hc_post 에 comb^T) — X' 에서 탐지.
/// (c) eps 제거(pre +eps·Sinkhorn eps 전부 0) — y·X' 에서 탐지.
/// 전부 maxdiff > DS4_HC_NEG_THRESH 여야 NEG-DETECTED.
pub fn cuda_ds4_hc_negative_check(dir: &str) -> Result<String, String> {
    let cfg_path = Path::new(dir).join("config.json");
    let cfg = std::fs::read_to_string(&cfg_path).map_err(|e| {
        format!(
            "{}: {e}(픽스처 계약 — config.json 필수)",
            cfg_path.display()
        )
    })?;
    let dims = Ds4HcDims::from_config(&cfg)?;
    let (hc, n, hcd, mix_rows) = (dims.hc, dims.n_embd, dims.hc_dim(), dims.mix_rows());
    let names = [
        "layers.0.hc_attn_fn",
        "layers.0.hc_attn_base",
        "layers.0.hc_attn_scale",
    ];
    let shapes: [(&[usize], u64); 3] = [
        (&[mix_rows, hcd], (mix_rows * hcd) as u64),
        (&[mix_rows], mix_rows as u64),
        (&[3], 3),
    ];
    let fx = ds4_hc_fixture(Path::new(dir), &names, &shapes)?;
    let g = |k: &str| fx.get(k).map(|v| v.as_slice()).unwrap_or(&[]);
    let (a_fn, a_base, a_scale) = (
        g("layers.0.hc_attn_fn"),
        g("layers.0.hc_attn_base"),
        g("layers.0.hc_attn_scale"),
    );

    let mut modl = Ds4HcCuda::new(dims)?;
    let dev = modl.device_name().to_string();
    modl.register(0, "attn", a_fn, a_base, a_scale)?;
    let res = gen_res_hc(3, hcd, 0x5EED_F00D_0000_1100, 0.5);
    let (gy, gpost, gcomb) = modl.hc_pre(0, "attn", &res)?;
    let f_syn: Vec<Vec<f32>> = (0..3)
        .map(|ti| gen_unif(n, 0x5EED_F00D_0000_1110 + ti, 0.5))
        .collect();
    let gx_post = modl.hc_post(&f_syn, &res, &gpost, &gcomb)?;

    // (a) 교환 변형 오라클 vs 모듈 — comb 판정(y 는 pre 만으로 결정되어
    // 순서에 둔감 — comb 가 이 대조의 감지면).
    let mut wcomb = Vec::new();
    for r in &res {
        let (_, _, comb) = hc_pre_ref(
            r,
            n,
            hc,
            a_fn,
            a_base,
            a_scale,
            dims.norm_eps,
            dims.hc_eps,
            dims.iters,
            true,
            false,
        );
        wcomb.push(comb);
    }
    let (mda, bda, _) = flat_judge(&gcomb, &wcomb);
    println!(
        "device: {dev} | ds4-hc-neg (a) sinkhorn order swapped oracle (19x{{col;row}}) vs kernel: comb maxdiff={mda:.3e} bitdiff={bda}/{} | FAIL(expected)",
        3 * hc * hc
    );
    let det_a = mda > DS4_HC_NEG_THRESH;

    // (b) comb 전치 오라클(hc_post 에 comb^T 적용) vs 모듈 X'.
    let mut want_t = Vec::new();
    for ti in 0..3 {
        let (_, post, comb) = hc_pre_ref(
            &res[ti],
            n,
            hc,
            a_fn,
            a_base,
            a_scale,
            dims.norm_eps,
            dims.hc_eps,
            dims.iters,
            false,
            false,
        );
        want_t.push(hc_post_ref(&f_syn[ti], &res[ti], &post, &comb, n, hc, true));
    }
    let (mdb, bdb, _) = flat_judge(&gx_post, &want_t);
    println!(
        "device: {dev} | ds4-hc-neg (b) comb^T oracle in hc_post vs kernel: X' maxdiff={mdb:.3e} bitdiff={bdb}/{} | FAIL(expected)",
        3 * hcd
    );
    let det_b = mdb > DS4_HC_NEG_THRESH;

    // (c) eps 제거 오라클(pre+0·Sinkhorn eps=0) vs 모듈 — y·X' 판정.
    let mut wy = Vec::new();
    let mut wpost = Vec::new();
    let mut wcomb = Vec::new();
    for r in &res {
        let (y, post, comb) = hc_pre_ref(
            r,
            n,
            hc,
            a_fn,
            a_base,
            a_scale,
            dims.norm_eps,
            dims.hc_eps,
            dims.iters,
            false,
            true,
        );
        wy.push(y);
        wpost.push(post);
        wcomb.push(comb);
    }
    let (mdc, bdc, _) = flat_judge(&gy, &wy);
    let mut want_c = Vec::new();
    for ti in 0..3 {
        want_c.push(hc_post_ref(
            &f_syn[ti], &res[ti], &wpost[ti], &wcomb[ti], n, hc, false,
        ));
    }
    let (mdcp, bdcp, _) = flat_judge(&gx_post, &want_c);
    println!(
        "device: {dev} | ds4-hc-neg (c) eps-dropped oracle (pre/sinkhorn eps=0) vs kernel: y maxdiff={mdc:.3e} bitdiff={bdc}/{} X' maxdiff={mdcp:.3e} bitdiff={bdcp}/{} | FAIL(expected)",
        3 * n,
        3 * hcd
    );
    let det_c = mdc > DS4_HC_NEG_THRESH || mdcp > DS4_HC_NEG_THRESH;

    if det_a && det_b && det_c {
        Err(format!(
            "NEG-DETECTED (a) sinkhorn-order comb maxdiff={mda:.3e} (b) comb^T X' maxdiff={mdb:.3e} (c) eps-dropped y/X' maxdiff={mdc:.3e}/{mdcp:.3e} > {DS4_HC_NEG_THRESH:.0e} — 검증계기 정상(반복순서·전치·eps 편차 감지)"
        ))
    } else {
        Err(format!(
            "NEG-MISSED (a)={mda:.3e} (b)={mdb:.3e} (c)={mdc:.3e}/{mdcp:.3e} <= {DS4_HC_NEG_THRESH:.0e} — 검증계기 결함: 음성이 탐지되지 않음"
        ))
    }
}
