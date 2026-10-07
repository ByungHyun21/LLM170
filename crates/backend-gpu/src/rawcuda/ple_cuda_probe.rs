//! Flash-Next PLE(n-gram 해시 임베딩) 검증층 — plans/124 FNB, 2026-10-05.
//! 3층 분리 원칙(§5): 프로브만 함수 삽입 금지(모듈 파일 오염 사고 1호) —
//! 본 파일은 ple_cuda.rs(모듈층)의 검증 자산만 소유한다.
//!
//! [129 A10 체크리스트 정식화 — plans/129-cuda C1]
//! ① 선행 단계 공유 버퍼 오염 점검: 게더→블록 t 일치 가드(모듈 gather_t),
//!    게더 결정성 2회 판독 비트동일, 블록 mids는 매 호출 전체 재판독.
//! ② 형상은 실측 메타에서 자동 열거(추정 금지): FnGguf kv→FnDims 가드
//!    (ple.layers=[1]·ngram3·hpng8·conv4·hd160·hc4·n_embd2560·실측 텐서
//!    타입 IQ4_NL/F32×4/Q8_0×2 — 2026-10-05 실측과 대조), 해시 파라미터
//!    mult/offs/vs는 GGUF kv ↔ EXL3 ngram 헤더 교차 대조.
//! ③ 캡처-재생: 실해시 rows→실 IQ4_NL 테이블 행 pread 재생(합성 아닌 실
//!    데이터 경로) + 결정론 토큰(Rng) 3세트·EOS 절단 경로·hist 진화.
//! ④ 종단 상태가 유일 불변량: 청크 분할 8+5 vs 단일 13 — res_hc·conv 상태
//!    비트동일(2청크째 상태는 비영 — S0≠0 정신, plans/124 §3.3 계승).
//!
//! 오라클: 내장 core-f32-미러(비트의식 트윈) — 산술 원천 인용:
//! - 해시: stages/ple.rs ple_hash_rows L276-335(래핑 곱/xor/%vs+offs·EOS
//!   cut 전파·hist0 스냅샷 lookback·hist drain).
//! - 게더: mod.rs ple_gather_parts L471-505 + quant/deq.rs deq_iq4_nl
//!   L241-247·f16 L6-8→half_to_f32 L11-38·tables.rs KVALUES_IQ4NL L73-75.
//! - Q8_0 디양자(key/value 공유 입력): deq.rs deq_q8_0 — y[j] = i8·d
//!   (곱 순서 q·d 그대로).
//! - 노름: ops.rs sq_sum L11-31(32세그먼트 f32 부분합→f64 순차 결합)·
//!   rms_norm L33-37(f64 sqrt→f32 캐스트→f32 역수·(v·scale)·g)·
//!   stages/hc.rs grouped_rms L14-21.
//! - exp/sigmoid/silu: ops.rs exp_cr L52-91(f64 FMA 호너 15차+포화 가드 —
//!   .cu fn_ple_expf와 리터럴까지 동일)·L128-137.
//! - 블록: stages/ple.rs L119-235 — 토큰·스트림 산출을 단일 스레드로 재현
//!   (core 병렬 분할은 "토큰별 결과가 그대로" 주석 L77·L87과 같이 수치
//!   불변 — 트윈은 순서만 계약).
//!   key/value 투영값은 양측 공유 입력(실 Q8_0 가중 디양자 + 순차 f32
//!   내적 — 투영 자체는 REUSE 영역이라 판정 대상 아님).
//!
//! 프로브(plans/124 §5·§6 — 판정은 값 maxdiff/정수 불일치, argmax 금지):
//! (i)   ple_block: 실가중·실해시·실테이블 t=13 — emb/gates/gated/conv_out/
//!       res_hc/상태 전 단계 비트동일(to_bits) + 청크 8+5 불변. PLE 코싱
//!       산술은 f32 결정론 경로(트랜센던트도 트윈 비트동일) — 임계는
//!       bitdiff=0(정수 판정, maxdiff는 부수 보고).
//! (ii)  ple_hash: 결정론 토큰(t=64/17/1·EOS 절단 포함)·실측 mult/offs/vs로
//!       모듈 ple_hash_rows vs 오라클 u32 전량 일치 + 청크 분할 16+48=64
//!       rows/hist 동일 + 행 도메인 [offs[h], offs[h]+vs[h]) 검증.
//! (iii) 음성대조(원장 17호): (a) 해시 계수 vs[3] 오염(+7) → rows→emb 이격,
//!       (b) 게더 인덱스 오염(스테이징 행 시프트) → emb 이격 — 둘 다
//!       NEG-DETECTED + 비영 exit이어야 한다(계기 자체 검증).
//!
//! [커널→프루브 매핑(plans/129-cuda C5)] exl3_fn_ple.cu 5커널
//! (iq4nl_gather/gate/conv/conv_state/resid) ↔ 본 파일 `ple`(전 커널 값
//! 판정)·`ple-neg`(계수·인덱스 오염 탐지).
//!
//! 가중치 적재 계약: 필요 행만 파일 오프셋 직독(FnGguf.read_rows —
//! 37GiB 테이블 전량 적재 금지, plans/86 §6). quantization_config.json
//! (92MB)은 fn_support 스트림 파서로 상위 스칼라만(전량 적재 금지 계약).

use crate::rawcuda::exl3_cuda_probe::{Rng, maxdiff_nan};
use crate::rawcuda::fn_support::{
    FN_EXL3_DIR, FnDims, FnGguf, FnNgramHead, fn_quant_config_stream,
};
use crate::rawcuda::ple_cuda::{
    IQ4NL_BLCK, IQ4NL_BYTES, PLE_TABLE_TENSOR, PleCuda, PleMids, ple_hash_rows,
};
use std::path::Path;

/// 음성대조 "검출" 하한 — emb 값 계급 O(1)에 대해 오염 시 ≫ 이 값.
const PLE_NEG_THRESH: f32 = 1e-3;
/// 검증 토큰 수(청크 분할 16+48=64·t=17·t=1·EOS 절단).
const T_HASH: usize = 64;
const T_CHUNK1: usize = 8;
const T_BLOCK: usize = 13;

// ── f16→f32 비트 변환 — deq.rs half_to_f32 L11-38 직미러 ──
fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match exp {
        0 => {
            if frac == 0 {
                sign << 31
            } else {
                let mut e = 127 - 15 + 1;
                let mut f = frac;
                while f & 0x400 == 0 {
                    f <<= 1;
                    e -= 1;
                }
                f &= 0x3ff;
                (sign << 31) | (e << 23) | (f << 13)
            }
        }
        0x1f => (sign << 31) | (0xff << 23) | (frac << 13),
        _ => (sign << 31) | ((exp + 112) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

/// KVALUES_IQ4NL — tables.rs L73-75 직이식(리터럴 정리 금지).
const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// deq_iq4_nl — deq.rs L241-247 직미러(블록 18B → 32원소, 곱 순서 d·kv).
fn deq_iq4_nl(blk: &[u8], y: &mut [f32]) {
    let d = half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
    let qs = &blk[2..18];
    for j in 0..16 {
        y[j] = d * KVALUES_IQ4NL[(qs[j] & 0xF) as usize] as f32;
        y[16 + j] = d * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
    }
}

/// ple_gather_parts — mod.rs L471-505 직미러(행 단위 — hd=160은 게더 커널
/// 형식 계약과 동일 가드, 행 원시 바이트 90B).
fn oracle_gather_row(raw: &[u8], hd: usize, out: &mut [f32]) {
    let n_blocks = hd.div_ceil(IQ4NL_BLCK);
    let mut tmp = [0.0f32; 512];
    for b in 0..n_blocks {
        deq_iq4_nl(
            &raw[b * IQ4NL_BYTES..(b + 1) * IQ4NL_BYTES],
            &mut tmp[b * IQ4NL_BLCK..(b + 1) * IQ4NL_BLCK],
        );
    }
    out[..hd].copy_from_slice(&tmp[..hd]);
}

/// deq_q8_0 — deq.rs 직미러(블록 34B → 32원소, 곱 순서 i8·d).
fn deq_q8_0(blk: &[u8], y: &mut [f32]) {
    let d = half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
    for j in 0..32 {
        y[j] = blk[2 + j] as i8 as f32 * d;
    }
}

/// Q8_0 선형 원시 행들 → f32 [rows][k] 디양자.
fn dequant_q8_rows(raw: &[u8], rows: usize, k: usize) -> Vec<f32> {
    let blocks = k / 32;
    let mut out = vec![0.0f32; rows * k];
    for r in 0..rows {
        for b in 0..blocks {
            deq_q8_0(
                &raw[(r * blocks + b) * 34..][..34],
                &mut out[r * k + b * 32..][..32],
            );
        }
    }
    out
}

// ── 코어 트랜센던트 트윈 — ops.rs exp_cr L52-91 직미러(리터럴·FMA 순서까지
// 동일 — .cu fn_ple_expf와 비트동일 계약, 절대 재작성 금지: 원장 17호) ──
fn exp_cr(x: f32) -> f32 {
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

fn silu(x: f32) -> f32 {
    x / (1.0 + exp_cr(-x))
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp_cr(-x))
}

// ── 노름 트윈 — ops.rs sq_sum L11-31·rms_norm L33-37·hc.rs grouped_rms L14-21 ──
fn sq_sum(x: &[f32]) -> f64 {
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

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let sum = sq_sum(x);
    let scale = 1.0 / ((sum / x.len() as f64 + eps as f64).sqrt() as f32);
    x.iter().zip(w).map(|(&v, &g)| v * scale * g).collect()
}

fn grouped_rms(x: &[f32], w: &[f32], hc: usize, n: usize, eps: f32) -> Vec<f32> {
    let mut xn = vec![0.0f32; hc * n];
    for s in 0..hc {
        let head = x[s * n..(s + 1) * n].to_vec();
        xn[s * n..(s + 1) * n].copy_from_slice(&rms_norm(&head, &w[s * n..(s + 1) * n], eps));
    }
    xn
}

// ── 해시 오라클 — stages/ple.rs ple_hash_rows L276-335 직미러(모듈층 사본과
// 독립 — 편집 회귀 감지용 쌍) ──
#[allow(clippy::too_many_arguments)]
fn oracle_hash_rows(
    hist0: &[u32],
    hist_valid: bool,
    tokens: &[u32],
    ngram: usize,
    hpng: usize,
    mult: &[u64],
    offs: &[u64],
    vs: &[u64],
    eos: u32,
) -> (Vec<u32>, Vec<u32>) {
    let heads = hpng * 2;
    let mut hist: Vec<u32> = if hist_valid {
        hist0.to_vec()
    } else {
        vec![eos; ngram - 1]
    };
    let mut rows = Vec::with_capacity(tokens.len() * heads);
    for (i, &tok) in tokens.iter().enumerate() {
        let mut ctx = vec![tok as u64; ngram];
        let mut cut = false;
        for s in 1..ngram {
            let j = i as i64 - s as i64;
            let prev: u64 = if j >= 0 {
                tokens[j as usize] as u64
            } else {
                let back = s as i64 - i as i64;
                let k = hist0.len() as i64 - back;
                if k >= 0 && (k as usize) < hist0.len() {
                    hist0[k as usize] as u64
                } else {
                    eos as u64
                }
            };
            ctx[s] = if cut { eos as u64 } else { prev };
            if ctx[s] == eos as u64 {
                cut = true;
            }
        }
        for n in 2..=ngram {
            let mut mixed = ctx[0].wrapping_mul(mult[0]);
            for j in 1..n {
                mixed ^= ctx[j].wrapping_mul(mult[j]);
            }
            let base = (n - 2) * hpng;
            for g in 0..hpng {
                let h = base + g;
                rows.push((mixed % vs[h] + offs[h]) as u32);
            }
        }
        hist.push(tok);
        if hist.len() > ngram - 1 {
            let cutn = hist.len() - (ngram - 1);
            hist.drain(..cutn);
        }
    }
    (rows, hist)
}

// ── ple_block 오라클 — stages/ple.rs L119-235 단일 스레드 재현 ──
/// 입·출력: res_hc[t][hc_dim] 제자리 잔차 가산, st[hist·hc_dim] 상태 진화.
/// 반환: (gates[t][hc], gated_norm[t][hc_dim], conv_out[t][hc_dim]).
#[allow(clippy::too_many_arguments)]
fn oracle_ple_block(
    emb: &[Vec<f32>],
    key: &[Vec<f32>],
    value: &[Vec<f32>],
    n_key: &[f32],
    n_query: &[f32],
    n_conv: &[f32],
    conv_w: &[f32],
    res_hc: &mut [Vec<f32>],
    st: &mut [f32],
    hc: usize,
    n_embd: usize,
    kern: usize,
    dil: usize,
    eps: f32,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let _ = emb; // 게더 산출은 key/value 투영으로만 소비(입력 공유 확인용).
    let hc_dim = hc * n_embd;
    let t = res_hc.len();
    let hist = (kern - 1) * dil;
    let mut gated_hist: Vec<Vec<f32>> = vec![Vec::new(); t];
    let mut gates_hist: Vec<Vec<f32>> = vec![Vec::new(); t];
    for ti in 0..t {
        // grouped norm key/query → 게이트(ple.rs L119-150).
        let k_n = grouped_rms(&key[ti], n_key, hc, n_embd, eps);
        let q_n = grouped_rms(&res_hc[ti], n_query, hc, n_embd, eps);
        let mut gate = vec![0.0f32; hc];
        for s in 0..hc {
            let mut dot = 0.0f32;
            for i in 0..n_embd {
                dot += k_n[s * n_embd + i] * q_n[s * n_embd + i];
            }
            dot /= (n_embd as f32).sqrt();
            let mag = dot.abs().max(1e-6).sqrt();
            gate[s] = sigmoid(if dot >= 0.0 { mag } else { -mag });
        }
        // value 방송×게이트 → grouped norm(L155-161).
        let mut gated = vec![0.0f32; hc_dim];
        for s in 0..hc {
            for i in 0..n_embd {
                gated[s * n_embd + i] = value[ti][i] * gate[s];
            }
        }
        let normalized = grouped_rms(&gated, n_conv, hc, n_embd, eps);
        gates_hist[ti] = gate;
        gated_hist[ti] = normalized;
    }
    // dilated conv(kern·dil·hist) — 상태 이용(L169-184) + tail 갱신(L189-193).
    let mut padded: Vec<Vec<f32>> = Vec::with_capacity(hist + t);
    for j in 0..hist {
        padded.push(st[j * hc_dim..(j + 1) * hc_dim].to_vec());
    }
    for g in gated_hist.iter() {
        padded.push(g.clone());
    }
    let mut conv_out = vec![vec![0.0f32; hc_dim]; t];
    for ti in 0..t {
        for k in 0..kern {
            let start = hist + ti - (kern - 1 - k) * dil;
            let src = &padded[start];
            for c in 0..hc_dim {
                conv_out[ti][c] += conv_w[c * kern + k] * src[c];
            }
        }
        for c in 0..hc_dim {
            conv_out[ti][c] = silu(conv_out[ti][c]);
        }
    }
    for j in 0..hist {
        let src = &padded[t + j];
        st[j * hc_dim..(j + 1) * hc_dim].copy_from_slice(src);
    }
    // 잔차 2경로(L217-235) — 게이트 재사용(plans/90 B4 D6).
    for ti in 0..t {
        let gate = &gates_hist[ti];
        for s in 0..hc {
            let g = gate[s];
            for i in 0..n_embd {
                res_hc[ti][s * n_embd + i] += value[ti][i] * g + conv_out[ti][s * n_embd + i];
            }
        }
    }
    (gates_hist, gated_hist, conv_out)
}

// ── 비트 판정(정수) + 값 보고 ──
/// (bitdiff 수, maxdiff) — 비트동일 계약: bitdiff==0이 판정 기준.
fn bitdiff(got: &[f32], want: &[f32]) -> (usize, f32) {
    let (md, _) = maxdiff_nan(got, want);
    let bd = got
        .iter()
        .zip(want)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    (bd, md)
}

// ── 픽스처 ──

/// PLE 실측 픽스처 — GGUF(형상·해시 파라미터·가중) + EXL3(ngram 헤더 교차
/// 대조·quant 스트림 게이트). 형상·타입 가드는 129 A10 ②.
struct PleFixture {
    g: FnGguf,
    dims: FnDims,
    mult: Vec<u64>,
    offs: Vec<u64>,
    vs: Vec<u64>,
    eos: u32,
    n_key: Vec<f32>,
    n_query: Vec<f32>,
    n_conv: Vec<f32>,
    conv_w: Vec<f32>,
    /// 디양자 Q8_0 key[hc_dim][emb_w]·value[n_embd][emb_w](행우선 평탄).
    key_w: Vec<f32>,
    value_w: Vec<f32>,
}

fn load_fixture(gguf_main: &str) -> Result<PleFixture, String> {
    let g = FnGguf::open(Path::new(gguf_main))?;
    let dims = FnDims::from_gguf(&g)?;
    // 형상 실측 대조(2026-10-05 — core mod.rs tests loader_split_contract와
    // 동일 값 + PLE 5종 형상).
    let got = (
        dims.n_layer,
        dims.n_embd,
        dims.hc,
        dims.ple_layers.clone(),
        dims.ple_ngram,
        dims.ple_heads_per_ngram,
        dims.ple_conv_k,
        dims.ple_head_dim,
    );
    let want = (48usize, 2560usize, 4usize, vec![1usize], 3, 8, 4, 160);
    if got != want {
        return Err(format!(
            "ple: 형상 {got:?} ≠ 실측 {want:?} — 픽스처 변경 가드"
        ));
    }
    let mult = g
        .kv_arr_u64("qwen4exp.ple.layer_multipliers")
        .ok_or("gguf kv: ple.layer_multipliers 없음")?
        .to_vec();
    let offs = g
        .kv_arr_u64("qwen4exp.ple.head_offsets")
        .ok_or("gguf kv: ple.head_offsets 없음")?
        .to_vec();
    let vs = g
        .kv_arr_u64("qwen4exp.ple.head_vocab_sizes")
        .ok_or("gguf kv: ple.head_vocab_sizes 없음")?
        .to_vec();
    let eos = g
        .kv_u64("qwen4exp.ple.eos_token_id")
        .ok_or("gguf kv: ple.eos_token_id 없음")? as u32;
    if mult.len() != dims.ple_ngram || offs.len() != 16 || vs.len() != 16 {
        return Err(format!(
            "ple: 해시 파라미터 길이 {}/{}/{} ≠ {}/16/16",
            mult.len(),
            offs.len(),
            vs.len(),
            dims.ple_ngram
        ));
    }
    // 헤드 파티션 인접 가드: offs[h]+vs[h] == offs[h+1](실측 계약).
    for h in 0..15 {
        if offs[h] + vs[h] != offs[h + 1] {
            return Err(format!("ple: 헤드 파티션 끊김 h={h} — 실측 레이아웃 가드"));
        }
    }
    // EXL3 ngram 헤더 교차 대조(FNA 실측: 두 원천 동일) + quant 스트림 게이트.
    let exl3 = Path::new(FN_EXL3_DIR);
    let ng = FnNgramHead::open(exl3)?;
    if ng.head_offsets != offs || ng.head_vocab_sizes != vs || ng.layer_multipliers != mult {
        return Err("ple: GGUF kv ↔ EXL3 ngram 헤더 파라미터 불일치 — 교차 대조 가드".into());
    }
    let q = fn_quant_config_stream(&exl3.join("quantization_config.json"))?;
    // 텐서 타입·형상 실측 대조(2026-10-05) + 필요 행만 직독.
    let want_tensor = |name: &str, ty: u32, dims: &[u64]| -> Result<(), String> {
        let t = g
            .tensor(name)
            .ok_or_else(|| format!("gguf 텐서 없음: {name}"))?;
        if t.ty != ty || t.dims != dims {
            return Err(format!(
                "ple: {name} ty{} {:?} ≠ 실측 ty{ty} {dims:?}",
                t.ty, t.dims
            ));
        }
        Ok(())
    };
    want_tensor(PLE_TABLE_TENSOR, 20, &[160, 320_001_536])?;
    want_tensor("blk.1.ple_norm_key.weight", 0, &[10240])?;
    want_tensor("blk.1.ple_norm_query.weight", 0, &[10240])?;
    want_tensor("blk.1.ple_norm_conv.weight", 0, &[10240])?;
    want_tensor("blk.1.ple_conv1d.weight", 0, &[4, 10240])?;
    want_tensor("blk.1.ple_key.weight", 8, &[2560, 10240])?;
    want_tensor("blk.1.ple_value.weight", 8, &[2560, 2560])?;
    let f32_1d = |name: &str| -> Result<Vec<f32>, String> {
        let b = g.read_rows(name, 0, 1)?;
        Ok(b.as_chunks::<4>().0.iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    };
    let n_key = f32_1d("blk.1.ple_norm_key.weight")?;
    let n_query = f32_1d("blk.1.ple_norm_query.weight")?;
    let n_conv = f32_1d("blk.1.ple_norm_conv.weight")?;
    // conv1d [4,10240] 행우선 전체 — flat c·kern+k(core f32_vec4 파일 순서).
    let cb = g.read_rows("blk.1.ple_conv1d.weight", 0, 10240)?;
    let conv_w: Vec<f32> = cb
        .as_chunks::<4>().0.iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let kb = g.read_rows("blk.1.ple_key.weight", 0, 10240)?;
    let key_w = dequant_q8_rows(&kb, 10240, 2560);
    let vb = g.read_rows("blk.1.ple_value.weight", 0, 2560)?;
    let value_w = dequant_q8_rows(&vb, 2560, 2560);
    if n_key.len() != 10240 || conv_w.len() != 40960 || key_w.len() != 10240 * 2560 {
        return Err("ple: 가중 직독 길이 가드 위반".into());
    }
    eprintln!(
        "[ple] fixture: {gguf_main} | EXL3 ngram 헤더 일치(mult {}/offs {}/vs {}) · quant stream {}B bits={} method={}",
        mult.len(),
        offs.len(),
        vs.len(),
        q.consumed_bytes,
        q.bits,
        q.quant_method,
    );
    Ok(PleFixture {
        g,
        dims,
        mult,
        offs,
        vs,
        eos,
        n_key,
        n_query,
        n_conv,
        conv_w,
        key_w,
        value_w,
    })
}

/// 결정론 토큰 스트림 — Rng [0, 262144) 유니폼(어휘 248320 이내의 실계급.
/// 해시 파라미터는 실측 — ③ 캡처-재생).
fn det_tokens(n: usize, seed: u64) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| (rng.next_u64() % 262_144) as u32).collect()
}

/// t개 토큰의 해시 rows→테이블 원시 행 바이트 pread 팩(행 순서 rows[ti·h]).
fn pack_table_rows(g: &FnGguf, rows: &[u32]) -> Result<Vec<u8>, String> {
    let mut raw = Vec::with_capacity(rows.len() * 90);
    for &r in rows {
        raw.extend_from_slice(&g.read_rows(PLE_TABLE_TENSOR, r as u64, 1)?);
    }
    Ok(raw)
}

/// emb[t][emb_w] → key[t][hc_dim]·value[t][n_embd] — 실 Q8_0 디양자 가중
/// 순차 f32 내적(공유 입력 — 투영은 REUSE 영역, 판정 대상 아님).
fn project(
    emb: &[Vec<f32>],
    key_w: &[f32],
    value_w: &[f32],
    k: usize,
    n_key_out: usize,
    n_val_out: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let t = emb.len();
    let mut key = vec![vec![0.0f32; n_key_out]; t];
    let mut value = vec![vec![0.0f32; n_val_out]; t];
    for ti in 0..t {
        for o in 0..n_key_out {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += emb[ti][i] * key_w[o * k + i];
            }
            key[ti][o] = acc;
        }
        for o in 0..n_val_out {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += emb[ti][i] * value_w[o * k + i];
            }
            value[ti][o] = acc;
        }
    }
    (key, value)
}

/// res_hc 초기화 — Rng ±0.2(잔류 스트림 계급 — 초기층 잔차 스케일급,
/// exl3_cuda_probe gen_unif 노선). 평탄 [t·hc_dim] 반환.
fn det_res_hc(t: usize, hc_dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..t * hc_dim)
        .map(|_| ((rng.next_f64() * 2.0 - 1.0) * 0.2) as f32)
        .collect()
}

/// 2차원 → 평탄화.
fn flat2(v: &[Vec<f32>]) -> Vec<f32> {
    v.iter().flatten().copied().collect()
}

/// ple — (i)~(ii) 통합 검증 프루브. 판정 실패는 Err(비영 exit).
pub fn cuda_ple_check(gguf_main: &str) -> Result<String, String> {
    let fx = load_fixture(gguf_main)?;
    let dims = &fx.dims;
    let (hc, n_embd, hc_dim, kern, dil) = (
        dims.hc,
        dims.n_embd,
        dims.hc * dims.n_embd,
        dims.ple_conv_k,
        dims.ple_ngram,
    );
    let heads = dims.ple_heads_per_ngram * 2;
    let emb_w = heads * dims.ple_head_dim;
    let hist = (kern - 1) * dil;
    let hpng = dims.ple_heads_per_ngram;

    // ── (ii) ple_hash: 모듈 vs 오라클 — 결정론 3세트 + EOS 절단 + 청크 ──
    let mut hash_rows_cmp = 0usize;
    let mut hash_bad = 0usize;
    let mut hash_sets = 0usize;
    for (n, seed) in [(T_HASH, 0xF00Du64), (17, 0xBEEF), (1, 0x1)] {
        let toks = det_tokens(n, seed);
        let (mrows, mhist) = ple_hash_rows(
            &[],
            false,
            &toks,
            dims.ple_ngram,
            hpng,
            &fx.mult,
            &fx.offs,
            &fx.vs,
            fx.eos,
        );
        let (orows, ohist) = oracle_hash_rows(
            &[],
            false,
            &toks,
            dims.ple_ngram,
            hpng,
            &fx.mult,
            &fx.offs,
            &fx.vs,
            fx.eos,
        );
        hash_rows_cmp += mrows.len();
        hash_bad += mrows.iter().zip(&orows).filter(|(a, b)| a != b).count();
        hash_bad += usize::from(mhist != ohist);
        hash_sets += 1;
        // 행 도메인 — 실 테이블 파티션 [offs[h], offs[h]+vs[h]).
        for (i, &r) in mrows.iter().enumerate() {
            let h = i % heads;
            if (r as u64) < fx.offs[h] || (r as u64) >= fx.offs[h] + fx.vs[h] {
                hash_bad += 1;
            }
        }
    }
    // EOS 절단 경로(cut 전파): 스트림 내 eos — 이후 prev 전부 eos.
    let mut toks_eos = det_tokens(24, 0xE05);
    toks_eos[5] = fx.eos;
    toks_eos[20] = fx.eos;
    {
        let (mrows, mhist) = ple_hash_rows(
            &[],
            false,
            &toks_eos,
            dims.ple_ngram,
            hpng,
            &fx.mult,
            &fx.offs,
            &fx.vs,
            fx.eos,
        );
        let (orows, ohist) = oracle_hash_rows(
            &[],
            false,
            &toks_eos,
            dims.ple_ngram,
            hpng,
            &fx.mult,
            &fx.offs,
            &fx.vs,
            fx.eos,
        );
        hash_rows_cmp += mrows.len();
        hash_bad += mrows.iter().zip(&orows).filter(|(a, b)| a != b).count();
        hash_bad += usize::from(mhist != ohist);
        hash_sets += 1;
    }
    // 청크 분할 16+48=64 — hist 진화 연쇄가 단일 호출과 동일(청크 불변성,
    // ple.rs L296 스냅샷 계약 + plans/109 P7).
    let toks64 = det_tokens(T_HASH, 0xF00D);
    let (rows_a, hist_a) = ple_hash_rows(
        &[],
        false,
        &toks64[..16],
        dims.ple_ngram,
        hpng,
        &fx.mult,
        &fx.offs,
        &fx.vs,
        fx.eos,
    );
    let (rows_b, hist_b) = ple_hash_rows(
        &hist_a,
        true,
        &toks64[16..],
        dims.ple_ngram,
        hpng,
        &fx.mult,
        &fx.offs,
        &fx.vs,
        fx.eos,
    );
    let (rows_full, hist_full) = ple_hash_rows(
        &[],
        false,
        &toks64,
        dims.ple_ngram,
        hpng,
        &fx.mult,
        &fx.offs,
        &fx.vs,
        fx.eos,
    );
    let chunk_ok = rows_a.len() + rows_b.len() == rows_full.len()
        && rows_a
            .iter()
            .chain(rows_b.iter())
            .zip(rows_full.iter())
            .all(|(a, b)| a == b)
        && hist_b == hist_full;
    if !chunk_ok {
        hash_bad += 1;
    }
    if hash_bad != 0 {
        return Err(format!(
            "ple: 해시 미러 불일치 {hash_bad}건(비교 {hash_rows_cmp}행/{hash_sets}세트)"
        ));
    }
    println!(
        "[ple] hash: sets={hash_sets} rows={hash_rows_cmp} mismatch=0 | chunk 16+48=64 identical | domain [offs,offs+vs) OK | eos-cut OK"
    );

    // ── 실데이터 체인: 모듈 생성·가중 상주 ──
    let mut m = PleCuda::new(dims.clone())?;
    let dev = m.device_name().to_string();
    let sh = m.shapes();
    if sh != (heads, emb_w, hc_dim, kern, hist, 90) {
        return Err(format!("ple: 모듈 유도 형상 {sh:?} 가드 위반"));
    }
    m.ple_load_block_weights(&fx.n_key, &fx.n_query, &fx.n_conv, &fx.conv_w)?;

    // ── (ii-b) 실해시 rows → 게더(실 IQ4_NL 행 pread) 비트 판정 ──
    let toks8 = det_tokens(8, 0x8BAD);
    let (rows8, _) = ple_hash_rows(
        &[],
        false,
        &toks8,
        dims.ple_ngram,
        hpng,
        &fx.mult,
        &fx.offs,
        &fx.vs,
        fx.eos,
    );
    let raw8 = pack_table_rows(&fx.g, &rows8)?;
    m.ple_gather_stage(&raw8, 8)?;
    let emb8_dev = m.ple_emb_download(8)?;
    // 게더 결정성(① 공유 버퍼 오염 점검): 재스테이징 후 동일 비트.
    m.ple_gather_stage(&raw8, 8)?;
    let emb8_dev2 = m.ple_emb_download(8)?;
    let (bd_emb_det, _) = bitdiff(&emb8_dev, &emb8_dev2);
    let mut emb8_or = vec![0.0f32; 8 * emb_w];
    for (hi, chunk) in emb8_or.chunks_mut(160).enumerate() {
        oracle_gather_row(&raw8[hi * 90..(hi + 1) * 90], 160, chunk);
    }
    let (bd_emb, md_emb) = bitdiff(&emb8_dev, &emb8_or);
    if bd_emb != 0 || bd_emb_det != 0 {
        return Err(format!(
            "ple: 게더 비트 불일치 — 오라클 대비 bitdiff={bd_emb} (maxdiff={md_emb:.3e}), 재스테이징 bitdiff={bd_emb_det}"
        ));
    }
    println!(
        "[ple] gather: t=8 rows=128 real IQ4_NL pread | bitdiff=0 maxdiff={md_emb:.3e} | restage deterministic"
    );

    // ── (i) ple_block t=13 — 실해시→실테이블→실가중 전체 체인 ──
    let toks13 = det_tokens(T_BLOCK, 0xD00D);
    let (rows13, _) = ple_hash_rows(
        &[],
        false,
        &toks13,
        dims.ple_ngram,
        hpng,
        &fx.mult,
        &fx.offs,
        &fx.vs,
        fx.eos,
    );
    let raw13 = pack_table_rows(&fx.g, &rows13)?;
    // 오라클 emb — 평탄 [t·emb_w]을 160원소 행 단위로 디양자(모듈·오라클
    // 동일 행 순서 rows[ti·heads+h]).
    let mut emb13_flat = vec![0.0f32; T_BLOCK * emb_w];
    for (hi, chunk) in emb13_flat.chunks_mut(160).enumerate() {
        oracle_gather_row(&raw13[hi * 90..(hi + 1) * 90], 160, chunk);
    }
    let emb13_or: Vec<Vec<f32>> = emb13_flat.chunks(emb_w).map(|c| c.to_vec()).collect();
    let (key13, value13) = project(&emb13_or, &fx.key_w, &fx.value_w, emb_w, hc_dim, n_embd);
    let mut mids = PleMids::default();
    // 모듈 경로 — 상태 0 시작(fresh, ① 오염 점검 후 명시 재주입).
    let mut res_dev = det_res_hc(T_BLOCK, hc_dim, 0x5EED);
    let st_zero = vec![0.0f32; hist * hc_dim];
    m.ple_set_conv_state(&st_zero)?;
    m.ple_gather_stage(&raw13, T_BLOCK)?;
    let emb13_dev = m.ple_emb_download(T_BLOCK)?;
    let (bd_e13, md_e13) = bitdiff(&emb13_dev, &flat2(&emb13_or));
    m.ple_block(
        T_BLOCK,
        &flat2(&key13),
        &flat2(&value13),
        &mut res_dev,
        &mut mids,
    )?;
    let st_dev = m.ple_conv_state()?;
    // 오라클 경로(동일 seed 초기값·공유 key/value).
    let res_seed = det_res_hc(T_BLOCK, hc_dim, 0x5EED);
    let mut res_or: Vec<Vec<f32>> = res_seed.chunks(hc_dim).map(|c| c.to_vec()).collect();
    let mut st_or = st_zero.clone();
    let (gates_or, gated_or, conv_or) = oracle_ple_block(
        &emb13_or,
        &key13,
        &value13,
        &fx.n_key,
        &fx.n_query,
        &fx.n_conv,
        &fx.conv_w,
        &mut res_or,
        &mut st_or,
        hc,
        n_embd,
        kern,
        dil,
        dims.eps,
    );
    let (bd_g, md_g) = bitdiff(&mids.gates, &flat2(&gates_or));
    let (bd_gt, md_gt) = bitdiff(&mids.gated, &flat2(&gated_or));
    let (bd_c, md_c) = bitdiff(&mids.conv_out, &flat2(&conv_or));
    let (bd_r, md_r) = bitdiff(&res_dev, &flat2(&res_or));
    let (bd_s, md_s) = bitdiff(&st_dev, &st_or);
    if bd_e13 != 0 || bd_g != 0 || bd_gt != 0 || bd_c != 0 || bd_r != 0 || bd_s != 0 {
        return Err(format!(
            "ple: 블록 비트 불일치 — emb={bd_e13} gates={bd_g} gated={bd_gt} conv={bd_c} res={bd_r} state={bd_s} (maxdiff {md_e13:.3e}/{md_g:.3e}/{md_gt:.3e}/{md_c:.3e}/{md_r:.3e}/{md_s:.3e})"
        ));
    }
    println!(
        "[ple] block: t=13 real weights+hash+table | bitdiff emb/gates/gated/conv/res/state = {bd_e13}/{bd_g}/{bd_gt}/{bd_c}/{bd_r}/{bd_s} | maxdiff res={md_r:.3e}"
    );

    // ── 청크 분할 8+5 vs 단일 13 — 종단 상태·res 비트동일(④) ──
    // 2청크째 진입 상태는 비영(1청크 gated norm 8열) — S0≠0 정신.
    let mut res_ck = det_res_hc(T_BLOCK, hc_dim, 0x5EED);
    m.ple_set_conv_state(&st_zero)?;
    let n_b = T_BLOCK - T_CHUNK1;
    m.ple_gather_stage(&raw13[..T_CHUNK1 * heads * 90], T_CHUNK1)?;
    m.ple_block(
        T_CHUNK1,
        &flat2(&key13[..T_CHUNK1]),
        &flat2(&value13[..T_CHUNK1]),
        &mut res_ck[..T_CHUNK1 * hc_dim],
        &mut mids,
    )?;
    let st_mid = m.ple_conv_state()?;
    if st_mid.iter().all(|&v| v == 0.0) {
        return Err("ple: 1청크 후 상태 전부 0 — 상태 경로 가드(S0≠0 정신 위반)".into());
    }
    m.ple_gather_stage(&raw13[T_CHUNK1 * heads * 90..], n_b)?;
    m.ple_block(
        n_b,
        &flat2(&key13[T_CHUNK1..]),
        &flat2(&value13[T_CHUNK1..]),
        &mut res_ck[T_CHUNK1 * hc_dim..],
        &mut mids,
    )?;
    let st_ck = m.ple_conv_state()?;
    let (bd_cr, md_cr) = bitdiff(&res_ck, &res_dev);
    let (bd_cs, _) = bitdiff(&st_ck, &st_dev);
    if bd_cr != 0 || bd_cs != 0 {
        return Err(format!(
            "ple: 청크 8+5 불변 위반 — res bitdiff={bd_cr}(maxdiff {md_cr:.3e}) state bitdiff={bd_cs}"
        ));
    }
    println!("[ple] chunk: 8+5 vs 13 — res_hc/state bit-identical | 2nd-chunk state nonzero");

    Ok(format!(
        "device: {dev} | ple-cuda PASS — hash {hash_rows_cmp} rows 0 bad, gather bit-exact, block t=13 bit-exact (6 stages), chunk 8+5 invariant"
    ))
}

/// ple-neg — (iii) 음성대조(원장 17호): (a) 해시 계수 vs[3] 오염(+7)이
/// rows→게더→emb에서, (b) 게더 인덱스 오염(스테이징 행 시프트)이 emb에서
/// 각각 값으로 탐지되어야 한다. 정상 동작은 NEG-DETECTED + Err(비영 exit).
pub fn cuda_ple_negative_check(gguf_main: &str) -> Result<String, String> {
    let fx = load_fixture(gguf_main)?;
    let dims = &fx.dims;
    let heads = dims.ple_heads_per_ngram * 2;
    let emb_w = heads * 160;

    // 기준: 실해시 rows 8토큰.
    let toks8 = det_tokens(8, 0x8BAD);
    let (rows8, _) = ple_hash_rows(
        &[],
        false,
        &toks8,
        dims.ple_ngram,
        dims.ple_heads_per_ngram,
        &fx.mult,
        &fx.offs,
        &fx.vs,
        fx.eos,
    );
    let raw8 = pack_table_rows(&fx.g, &rows8)?;
    let mut emb8_or = vec![0.0f32; 8 * emb_w];
    for (hi, chunk) in emb8_or.chunks_mut(160).enumerate() {
        oracle_gather_row(&raw8[hi * 90..(hi + 1) * 90], 160, chunk);
    }

    let mut m = PleCuda::new(dims.clone())?;
    let dev = m.device_name().to_string();

    // (a) 해시 계수 오염 — vs[3] += 7(유한 유지·계통 변화): 헤드 3 행이
    // 다른 테이블 행을 가리킨다 → emb 전체 재판독으로 탐지.
    let mut vs_bad = fx.vs.clone();
    vs_bad[3] += 7;
    let (rows_bad, _) = ple_hash_rows(
        &[],
        false,
        &toks8,
        dims.ple_ngram,
        dims.ple_heads_per_ngram,
        &fx.mult,
        &fx.offs,
        &vs_bad,
        fx.eos,
    );
    let raw_bad = pack_table_rows(&fx.g, &rows_bad)?;
    m.ple_gather_stage(&raw_bad, 8)?;
    let emb_bad = m.ple_emb_download(8)?;
    let (md_a, _) = maxdiff_nan(&emb_bad, &emb8_or);

    // (b) 게더 인덱스 오염 — 정상 rows의 스테이징 0번 행에 1번 행 원시
    // 바이트(off-by-one 게더 인덱스 계급).
    let mut raw_shift = raw8.clone();
    let row1: Vec<u8> = raw8[90..180].to_vec();
    raw_shift[0..90].copy_from_slice(&row1);
    m.ple_gather_stage(&raw_shift, 8)?;
    let emb_shift = m.ple_emb_download(8)?;
    let (md_b, _) = maxdiff_nan(&emb_shift, &emb8_or);

    println!(
        "device: {dev} | ple-neg (iii) corrupted: (a) hash modulus vs[3]+7 maxdiff={md_a:.3e} (b) gather index shift maxdiff={md_b:.3e} | FAIL(expected)"
    );
    if md_a > PLE_NEG_THRESH && md_b > PLE_NEG_THRESH {
        Err(format!(
            "NEG-DETECTED modulus={md_a:.3e} gather-idx={md_b:.3e} > {PLE_NEG_THRESH:.0e} — 검증계기 정상(계수·인덱스 오염 모두 감지)"
        ))
    } else {
        Err(format!(
            "NEG-MISSED modulus={md_a:.3e} gather-idx={md_b:.3e} <= {PLE_NEG_THRESH:.0e} — 검증계기 결함: 오염이 탐지되지 않음"
        ))
    }
}
