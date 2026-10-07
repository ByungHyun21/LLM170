//! [하네스 저작 원칙 — plans/129-cuda-only C1(원본 129 A10) 체크리스트]
//! ① 선행 단계 공유 버퍼 오염 점검: hc 케이스마다 잔류·가중치를 재업로드
//!    한다(모듈 작업 버퍼는 현 t 행 전체를 매 체인마다 덮어쓴다).
//! ② 형상은 픽스처 config.json에서 자동 확정(Flash-Next 실측
//!    hc=4·n_embd=2560·low_rank=320·eps=1e-6)+ T=1..3 토큰축.
//! ③ 캡처-재생 3방향: 결정론 합성 + 실측 픽스처 재생
//!    (mtp_hyper_connection_mixer_patch.safetensors F16 3텐서 —
//!    오프셋 직독, 합성 잔류로 구동).
//! ④ 종단 값이 유일 불변량: 판정은 hc_mix 산출 mixed·inject 의 값
//!    maxdiff+비트 불일치 수(중간 단계는 진단).
//!
//! [오라클 — core qwen4exp 참조 직이식(값 maxdiff 판정의 유일 기준,
//! plans/124 §6). 인용은 전부 워크트리 기준 줄번호, 2026-10-05]
//! - crates/core/src/qwen4exp/stages/hc.rs — grouped_rms L14-21 ·
//!   hc_mix_ex L25-86(grouped rms → down(+inject) → silu(lo/hc) L58-61 →
//!   up → xn·sigmoid(gate) L66-71 → 스트림 평균 /=hc L72-79).
//! - crates/core/src/ops.rs — sq_sum L11-31(32세그먼트 f32 순차 누산 →
//!   f64 순차 결합) · rms_norm L33-37(f64 평균+eps → sqrt → f32 역수) ·
//!   exp_cr L63-119(f64 fma 호너 13단 — 커널 hc_exp_cr 과 리터럴 동일) ·
//!   silu L127-130 · sigmoid L131-133.
//! - crates/core/src/matmul/cpu.rs — matmul L64-76(행별 f32 순차 누산,
//!   mul+add 무-FMA — ADR-0005).
//!
//! [정합 원장 요약 — 전 케이스 비트동일 목표(FNC 판정 기준)]
//! 합성 attn/ffn(T=3·T=1)·head·nextn_head·실측 패치 픽스처 전부
//! bitdiff=0 · nan=0 기대 — 측정값은 골든 원장(scripts/.verify-cuda-golden.tsv)
//! 갱신 커밋으로 기록한다.
//!
//! [음성대조 — 원장 17호(계기 자체 검증)] 2종 모두 NEG-DETECTED 필:
//! (a) 잘못된 연결 수: 스트림 평균·silu 스케일이 hc=3 로 오설정된
//!     구현 산출(오라클 변형)과의 maxdiff — 임계 초과 검증.
//! (b) 잘못된 스트림 순서: norm 감마의 스트림 0↔1 치환 등록(모듈 실경로)
//!     대비 정답 오라클 — 임계 초과 검증.
//!
//! [속도] 측정 대기 sm_80 — CMP 170HX 미도착(plans/124 §0).
//!
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 금지.

use crate::rawcuda::exl3_cuda::{JParser, JVal};
use crate::rawcuda::exl3_cuda_probe::{Rng, f16_to_f32, gen_unif, maxdiff_nan};
use crate::rawcuda::hc_cuda::{HC_HEAD_IL, HcCuda, HcDims};
use std::path::Path;

/// 기본 픽스처 경로(과제 지정 — 프로브 인자로 대체 가능).
pub const FN_HC_EXL3_DIR: &str = "D:/models/Qwen3.8-Flash-Next-exl3-5.05bpw";

/// 값 maxdiff 폴백 임계(norm 계급 — plans/124 §1). 본 모듈 판정 기준은
/// 비트동일(bitdiff=0)이며, 이 임계는 음성대조 "탐지됨" 판정 경계.
const HC_THRESH: f32 = 3e-6;

// ── core 미러 오라클 — 인용 줄번호는 헤드 [오라클] 항 ──

/// ops.rs sq_sum(L11-31) 직이식 — 32세그먼트 f32 순차 누산 → f64 결합.
fn core_sq_sum(x: &[f32]) -> f64 {
    const SEG: usize = 32;
    let n = x.len();
    let chunk = n.div_ceil(SEG);
    let mut sum = 0.0f64;
    for u in 0..SEG {
        let lo = u * chunk;
        if lo >= n {
            break;
        }
        let hi = (lo + chunk).min(n);
        let mut part = 0.0f32;
        for &v in &x[lo..hi] {
            part += v * v;
        }
        sum += part as f64;
    }
    sum
}

/// ops.rs rms_norm(L33-37) 직이식 — f64 평균+eps → sqrt → f32 역수 →
/// v·scale·g(연산별 f32 반올림).
fn core_rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let sum = core_sq_sum(x);
    let scale = 1.0 / ((sum / x.len() as f64 + eps as f64).sqrt() as f32);
    x.iter().zip(w).map(|(&v, &g)| v * scale * g).collect()
}

/// ops.rs exp_cr(L63-119) 직이식 — f64 fma 호너 13단 + 2^k 비트 재구성.
/// 커널 hc_exp_cr(assets/exl3_fn_hc.cu)과 리터럴까지 동일(mul_add ≡
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

/// ops.rs silu(L127-130) 직이식.
fn core_silu(x: f32) -> f32 {
    x / (1.0 + core_exp_cr(-x))
}

/// ops.rs sigmoid(L131-133) 직이식.
fn core_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + core_exp_cr(-x))
}

/// cpu.rs matmul(L64-76) 행내적 미러 — f32 순차 누산(mul+add 무-FMA).
fn dot_row(x: &[f32], w: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..x.len() {
        acc += x[i] * w[i];
    }
    acc
}

/// stages/hc.rs grouped_rms(L14-21) 직이식 — [hc·n] 행을 스트림별 절단,
/// 각각 core_rms_norm.
fn hc_grouped_rms_ref(x: &[f32], w: &[f32], hc: usize, n: usize, eps: f32) -> Vec<f32> {
    let mut xn = vec![0.0f32; hc * n];
    for s in 0..hc {
        xn[s * n..(s + 1) * n].copy_from_slice(&core_rms_norm(
            &x[s * n..(s + 1) * n],
            &w[s * n..(s + 1) * n],
            eps,
        ));
    }
    xn
}

/// stages/hc.rs hc_mix_ex(L25-86) 직이식 — (mixed, inject) 반환.
/// `hc_mix_hc`: silu 스케일·스트림 평균에 쓰는 연결 수(정답=dims.hc —
/// 음성대조 (a)가 3 주입). linear 는 down [lr][hcn]·up [hcn][lr]·
/// inject [hc][hcn] 행우선.
fn hc_mix_ex_ref(
    dims: &HcDims,
    w_norm: &[f32],
    w_down: &[f32],
    w_up: &[f32],
    w_inject: Option<&[f32]>,
    res_hc: &[Vec<f32>],
    hc_mix_hc: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let (hc, n, lr) = (dims.hc, dims.n_embd, dims.low_rank);
    let hcn = hc * n;
    let t = res_hc.len();
    // 1) grouped RMSNorm — 전 토큰(L40-45).
    let xn_all: Vec<Vec<f32>> = res_hc
        .iter()
        .map(|x| hc_grouped_rms_ref(x, w_norm, hc, n, dims.eps))
        .collect();
    // 2) down → (inject 있으면 동일 입력 xn_all — D4 그룹 계약).
    let mut lo_all = vec![vec![0.0f32; lr]; t];
    for (ti, lo) in lo_all.iter_mut().enumerate() {
        for o in 0..lr {
            lo[o] = dot_row(&xn_all[ti], &w_down[o * hcn..(o + 1) * hcn]);
        }
    }
    let inject_all: Vec<Vec<f32>> = match w_inject {
        Some(wi) => (0..t)
            .map(|ti| {
                (0..hc)
                    .map(|o| dot_row(&xn_all[ti], &wi[o * hcn..(o + 1) * hcn]))
                    .collect()
            })
            .collect(),
        None => vec![Vec::new(); t],
    };
    // silu(lo/hc)(L58-61) — hc_mix_hc 오설정 시 여기서 어긋난다.
    for lo in lo_all.iter_mut() {
        for v in lo.iter_mut() {
            *v = core_silu(*v / hc_mix_hc as f32);
        }
    }
    // up.
    let mut gate_all = vec![vec![0.0f32; hcn]; t];
    for (ti, gate) in gate_all.iter_mut().enumerate() {
        for o in 0..hcn {
            gate[o] = dot_row(&lo_all[ti], &w_up[o * lr..(o + 1) * lr]);
        }
    }
    // 3) 게이트 적용 + 스트림 평균(L66-79).
    let mut mixed = Vec::with_capacity(t);
    for (gate, xn) in gate_all.iter().zip(xn_all.iter()) {
        let mut m = vec![0.0f32; n];
        for s in 0..hc_mix_hc {
            for i in 0..n {
                m[i] += xn[s * n + i] * core_sigmoid(gate[s * n + i]);
            }
        }
        for v in m.iter_mut() {
            *v /= hc_mix_hc as f32;
        }
        mixed.push(m);
    }
    (mixed, inject_all)
}

// ── 합성 자료(결정론 시드 — splitmix64 미러 재사용) ──

/// lo..hi 균일 행렬 [rows][cols] 행우선(노름 감마 0.8..1.2 계급).
fn gen_mat(rows: usize, cols: usize, seed: u64, lo: f64, hi: f64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..rows * cols)
        .map(|_| (lo + rng.next_f64() * (hi - lo)) as f32)
        .collect()
}

/// 잔류 행 [t][hc_dim] — ±amp(초기 잔류 스트림 계급).
fn gen_res_hc(t: usize, hcn: usize, seed: u64, amp: f64) -> Vec<Vec<f32>> {
    (0..t)
        .map(|ti| gen_unif(hcn, seed + ti as u64, amp))
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

// ── 실측 픽스처: mtp_hyper_connection_mixer_patch.safetensors ──

/// 패치 믹서 3텐서 오프셋 직독(F16 → f32). 전량 적재 금지 계약 —
/// 헤더(360B 실측) 판독 후 data_offsets 위치의 정확 바이트만.
fn patch_mixer(dir: &Path, dims: &HcDims) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    let path = dir.join("mtp_hyper_connection_mixer_patch.safetensors");
    let mut f = std::fs::File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut lenb = [0u8; 8];
    std::io::Read::read_exact(&mut f, &mut lenb).map_err(|e| e.to_string())?;
    let hlen = u64::from_le_bytes(lenb);
    if hlen == 0 || hlen > (1 << 22) {
        return Err(format!("hc 패치 헤더 길이 {hlen} — 가드(실측 360)"));
    }
    let mut hb = vec![0u8; hlen as usize];
    std::io::Read::read_exact(&mut f, &mut hb).map_err(|e| e.to_string())?;
    let data_base = 8 + hlen;
    use std::io::{Seek, SeekFrom};
    let v = JParser { b: &hb, p: 0 }.parse()?;
    let obj = v.as_obj().ok_or("hc 패치: 헤더가 객체 아님")?;
    let (hcn, lr) = (dims.hc_dim(), dims.low_rank);
    let mut norm = Vec::new();
    let mut down = Vec::new();
    let mut up = Vec::new();
    for (name, tv) in obj {
        if !name.starts_with("mtp.hyper_connection_mixer.") {
            continue;
        }
        let dt = tv
            .get("dtype")
            .and_then(JVal::as_str)
            .ok_or("hc 패치: dtype 없음")?;
        if dt != "F16" {
            return Err(format!("hc 패치: dtype {dt} — F16 고정(픽스처 변경 가드)"));
        }
        let shape = tv
            .get("shape")
            .and_then(JVal::as_arr)
            .ok_or("hc 패치: shape 없음")?;
        let sh: Vec<usize> = shape
            .iter()
            .filter_map(JVal::as_f64)
            .map(|v| v as usize)
            .collect();
        let offs = tv
            .get("data_offsets")
            .and_then(JVal::as_arr)
            .ok_or("hc 패치: data_offsets 없음")?;
        let g64 = |i: usize| {
            offs.get(i)
                .and_then(JVal::as_f64)
                .map(|v| v as u64)
                .ok_or_else(|| format!("hc 패치: data_offsets[{i}] 없음"))
        };
        let (b, e) = (g64(0)?, g64(1)?);
        let want: Vec<usize> = if name.ends_with("hc_norm.weight") {
            vec![hcn]
        } else if name.ends_with("input_mix_weight_down.weight") {
            vec![lr, hcn]
        } else if name.ends_with("input_mix_weight_up.weight") {
            vec![hcn, lr]
        } else {
            continue;
        };
        if sh != want {
            return Err(format!(
                "hc 패치: {name} shape {sh:?} != {want:?}(dims 불일치 가드)"
            ));
        }
        f.seek(SeekFrom::Start(data_base + b))
            .map_err(|e| e.to_string())?;
        let mut buf = vec![0u8; (e - b) as usize];
        std::io::Read::read_exact(&mut f, &mut buf).map_err(|e| e.to_string())?;
        let vals: Vec<f32> = buf
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        if name.ends_with("hc_norm.weight") {
            norm = vals;
        } else if name.ends_with("input_mix_weight_down.weight") {
            down = vals;
        } else {
            up = vals;
        }
    }
    if norm.is_empty() || down.is_empty() || up.is_empty() {
        return Err("hc 패치: 3텐서(norm/down/up) 인식 실패".into());
    }
    Ok((norm, down, up))
}

// ── 메인 프로브 ──

/// fn-hc — Flash-Next hc 스테이지 값 판정(비트동일 기준). 케이스:
/// (i) 합성 attn T=3 + T=1(inject 포함) · (ii) 합성 head·nextn_head
/// (inject 없음) · (iii) 실측 패치 픽스처 nextn_head T=2 ·
/// (iv) grouped_rms 단독 1행. 전 케이스 bitdiff=0·nan=0 요구.
pub fn cuda_hc_check(dir: &str) -> Result<String, String> {
    let cfg_path = Path::new(dir).join("config.json");
    let cfg = std::fs::read_to_string(&cfg_path).map_err(|e| {
        format!(
            "{}: {e}(픽스처 계약 — config.json 필수)",
            cfg_path.display()
        )
    })?;
    let dims = HcDims::from_config(&cfg)?;
    let (hc, n, lr, hcn) = (dims.hc, dims.n_embd, dims.low_rank, dims.hc_dim());
    let mut modl = HcCuda::new(dims)?;
    let dev = modl.device_name().to_string();
    let mut fails: Vec<String> = Vec::new();
    let mut report = String::new();

    // (i) 합성 attn 믹서(inject 포함) — initializer_range 0.02 계급.
    let norm = gen_mat(1, hcn, 0x5EED_F00D_0000_1000, 0.8, 1.2);
    let down = gen_mat(lr, hcn, 0x5EED_F00D_0000_1001, -0.02, 0.02);
    let up = gen_mat(hcn, lr, 0x5EED_F00D_0000_1002, -0.02, 0.02);
    let inject = gen_mat(hc, hcn, 0x5EED_F00D_0000_1003, -0.02, 0.02);
    modl.register(0, "attn", &norm, &down, &up, Some(&inject))?;
    for (t, tag) in [(3usize, "T=3"), (1, "T=1")] {
        let res = gen_res_hc(t, hcn, 0x5EED_F00D_0000_1100, 0.5);
        let (want_m, want_i) = hc_mix_ex_ref(&dims, &norm, &down, &up, Some(&inject), &res, hc);
        let (got_m, got_i) = modl.hc_mix(0, "attn", &res)?;
        let (mdm, bdm, nm) = flat_judge(&got_m, &want_m);
        let (mdi, bdi, ni) = flat_judge(&got_i, &want_i);
        let inj_span = want_i.iter().flatten().fold(0f32, |a, &v| a.max(v.abs()));
        let pass = bdm == 0 && nm == 0 && bdi == 0 && ni == 0 && inj_span > 0.0;
        println!(
            "device: {dev} | fn-hc (i) attn {tag}: mixed maxdiff={mdm:.3e} bitdiff={bdm}/{}/{} nan={nm} | inject maxdiff={mdi:.3e} bitdiff={bdi}/{}/{} nan={ni} span={inj_span:.3e} | {}",
            t * n,
            t * n,
            t * hc,
            t * hc,
            if pass { "PASS" } else { "FAIL" }
        );
        report.push_str(&format!("(i {tag}) m={mdm:.3e}/{bdm}b i={mdi:.3e}/{bdi}b"));
        if !pass {
            fails.push(format!(
                "(i {tag}) mixed {mdm:.3e}/{bdm}b nan={nm} inject {mdi:.3e}/{bdi}b nan={ni}"
            ));
        }
    }

    // (ii) head(output_hc_*)·nextn_head 합성 — inject 없음(빈 행 계약).
    let norm_h = gen_mat(1, hcn, 0x5EED_F00D_0000_2000, 0.8, 1.2);
    let down_h = gen_mat(lr, hcn, 0x5EED_F00D_0000_2001, -0.02, 0.02);
    let up_h = gen_mat(hcn, lr, 0x5EED_F00D_0000_2002, -0.02, 0.02);
    modl.register(HC_HEAD_IL, "head", &norm_h, &down_h, &up_h, None)?;
    let res2 = gen_res_hc(2, hcn, 0x5EED_F00D_0000_2100, 0.5);
    let (want_h, _) = hc_mix_ex_ref(&dims, &norm_h, &down_h, &up_h, None, &res2, hc);
    let got_h = modl.hc_mix_head(&res2)?;
    let (mdh, bdh, nh) = flat_judge(&got_h, &want_h);
    let pass = bdh == 0 && nh == 0;
    println!(
        "device: {dev} | fn-hc (ii) head T=2: maxdiff={mdh:.3e} bitdiff={bdh}/{} nan={nh} | {}",
        2 * n,
        if pass { "PASS" } else { "FAIL" }
    );
    report.push_str(&format!(" · (ii head) {mdh:.3e}/{bdh}b"));
    if !pass {
        fails.push(format!("(ii head) {mdh:.3e}/{bdh}b nan={nh}"));
    }
    let norm_n = gen_mat(1, hcn, 0x5EED_F00D_0000_2200, 0.8, 1.2);
    let down_n = gen_mat(lr, hcn, 0x5EED_F00D_0000_2201, -0.02, 0.02);
    let up_n = gen_mat(hcn, lr, 0x5EED_F00D_0000_2202, -0.02, 0.02);
    modl.register(7, "nextn_head", &norm_n, &down_n, &up_n, None)?;
    let res7 = gen_res_hc(2, hcn, 0x5EED_F00D_0000_2300, 0.4);
    let (want_n, _) = hc_mix_ex_ref(&dims, &norm_n, &down_n, &up_n, None, &res7, hc);
    let got_n = modl.hc_mix_nextn_head(7, &res7)?;
    let (mdn, bdn, nn) = flat_judge(&got_n, &want_n);
    let pass = bdn == 0 && nn == 0;
    println!(
        "device: {dev} | fn-hc (ii) nextn_head il=7 T=2: maxdiff={mdn:.3e} bitdiff={bdn}/{} nan={nn} | {}",
        2 * n,
        if pass { "PASS" } else { "FAIL" }
    );
    report.push_str(&format!(" · (ii nextn) {mdn:.3e}/{bdn}b"));
    if !pass {
        fails.push(format!("(ii nextn) {mdn:.3e}/{bdn}b nan={nn}"));
    }

    // (iii) 실측 패치 픽스처 — mtp.hyper_connection_mixer 3텐서(F16).
    let (pn, pd, pu) = patch_mixer(Path::new(dir), &dims)?;
    modl.register(0, "nextn_head", &pn, &pd, &pu, None)?;
    let res3 = gen_res_hc(2, hcn, 0x5EED_F00D_0000_3100, 0.3);
    let (want_p, _) = hc_mix_ex_ref(&dims, &pn, &pd, &pu, None, &res3, hc);
    let got_p = modl.hc_mix_nextn_head(0, &res3)?;
    let (mdp, bdp, np_) = flat_judge(&got_p, &want_p);
    let pass = bdp == 0 && np_ == 0;
    println!(
        "device: {dev} | fn-hc (iii) patch fixture nextn_head T=2 (F16→f32 {}: {}·{}·{}): maxdiff={mdp:.3e} bitdiff={bdp}/{} nan={np_} | {}",
        "정확변환",
        pn.len(),
        pd.len(),
        pu.len(),
        2 * n,
        if pass { "PASS" } else { "FAIL" }
    );
    report.push_str(&format!(" · (iii patch) {mdp:.3e}/{bdp}b"));
    if !pass {
        fails.push(format!("(iii patch) {mdp:.3e}/{bdp}b nan={np_}"));
    }

    // (iv) grouped_rms 단독 1행 — grouped_rms L14-21 직접 대조.
    let resr = gen_res_hc(1, hcn, 0x5EED_F00D_0000_4000, 0.6)[0].clone();
    let want_r = hc_grouped_rms_ref(&resr, &norm, hc, n, dims.eps);
    let got_r = modl.hc_grouped_rms(&resr, &norm)?;
    let (mdr, nr) = maxdiff_nan(&got_r, &want_r);
    let bdr = bitdiff(&got_r, &want_r);
    let pass = bdr == 0 && nr == 0;
    println!(
        "device: {dev} | fn-hc (iv) grouped_rms 1행: maxdiff={mdr:.3e} bitdiff={bdr}/{hcn} nan={nr} | {}",
        if pass { "PASS" } else { "FAIL" }
    );
    report.push_str(&format!(" · (iv rms) {mdr:.3e}/{bdr}b"));
    if !pass {
        fails.push(format!("(iv rms) {mdr:.3e}/{bdr}b nan={nr}"));
    }

    if fails.is_empty() {
        Ok(format!(
            "device: {dev} | dims hc={hc} n={n} lr={lr} eps={} | {report} | ALL PASS (비트동일 기준)",
            dims.eps
        ))
    } else {
        Err(format!(
            "fn-hc 실패 — {} (device: {dev}, dims hc={hc} n={n} lr={lr})",
            fails.join(", ")
        ))
    }
}

/// fn-hc-neg — 음성대조 2종(원장 17호: 검증 계기도 스스로 검증).
/// (a) 잘못된 연결 수: 평균·silu 스케일 hc=3 오라클 변형 vs 모듈 정답.
/// (b) 잘못된 스트림 순서: 모듈에 norm 스트림 0↔1 치환 등록(실경로)
///     vs 정답 오라클. 양쪽 maxdiff > HC_THRESH 여야 NEG-DETECTED.
pub fn cuda_hc_negative_check(dir: &str) -> Result<String, String> {
    let cfg_path = Path::new(dir).join("config.json");
    let cfg = std::fs::read_to_string(&cfg_path).map_err(|e| {
        format!(
            "{}: {e}(픽스처 계약 — config.json 필수)",
            cfg_path.display()
        )
    })?;
    let dims = HcDims::from_config(&cfg)?;
    let (hc, n, lr, hcn) = (dims.hc, dims.n_embd, dims.low_rank, dims.hc_dim());
    let mut modl = HcCuda::new(dims)?;
    let dev = modl.device_name().to_string();

    let norm = gen_mat(1, hcn, 0x5EED_F00D_0000_1000, 0.8, 1.2);
    let down = gen_mat(lr, hcn, 0x5EED_F00D_0000_1001, -0.02, 0.02);
    let up = gen_mat(hcn, lr, 0x5EED_F00D_0000_1002, -0.02, 0.02);
    let inject = gen_mat(hc, hcn, 0x5EED_F00D_0000_1003, -0.02, 0.02);
    modl.register(0, "attn", &norm, &down, &up, Some(&inject))?;
    let res = gen_res_hc(3, hcn, 0x5EED_F00D_0000_1100, 0.5);
    let (got_m, _) = modl.hc_mix(0, "attn", &res)?;

    // (a) hc=3 오설정 구현 산출(오라클 변형) vs 모듈 정답.
    let (want3, _) = hc_mix_ex_ref(&dims, &norm, &down, &up, Some(&inject), &res, 3);
    let (mda, bda, na) = flat_judge(&got_m, &want3);
    println!(
        "device: {dev} | fn-hc-neg (a) wrong connection count hc=3 oracle vs kernel hc={hc}: maxdiff={mda:.3e} bitdiff={bda}/{} nan={na} | FAIL(expected)",
        3 * n
    );
    let det_a = mda > HC_THRESH;

    // (b) norm 스트림 0↔1 치환 등록(모듈 실경로) vs 정답 오라클.
    let mut norm_sw = norm.clone();
    for i in 0..n {
        norm_sw.swap(i, n + i);
    }
    modl.register(5, "ffn", &norm_sw, &down, &up, Some(&inject))?;
    let (got_sw, _) = modl.hc_mix(5, "ffn", &res)?;
    let (want_b, _) = hc_mix_ex_ref(&dims, &norm, &down, &up, Some(&inject), &res, hc);
    let (mdb, bdb, nb) = flat_judge(&got_sw, &want_b);
    println!(
        "device: {dev} | fn-hc-neg (b) stream order norm[0↔1] registration vs correct oracle: maxdiff={mdb:.3e} bitdiff={bdb}/{} nan={nb} | FAIL(expected)",
        3 * n
    );
    let det_b = mdb > HC_THRESH;

    if det_a && det_b {
        Err(format!(
            "NEG-DETECTED (a) maxdiff={mda:.3e} (b) maxdiff={mdb:.3e} > {HC_THRESH:.0e} — 검증계기 정상(연결 수·스트림 순서 편차 감지)"
        ))
    } else {
        Err(format!(
            "NEG-MISSED (a)={mda:.3e} (b)={mdb:.3e} <= {HC_THRESH:.0e} — 검증계기 결함: 음성이 탐지되지 않음"
        ))
    }
}
