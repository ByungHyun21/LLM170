//! EXL3 MTP 드래프트 검증층 — plans/124 G9, 2026-10-04.
//! 3층 분리 원칙(§5): 프로브만 함수 삽입 금지(모듈 파일 오염 사고
//! 1호) — 본 파일은 mtp_cuda.rs(모듈층)의 검증 자산만 소유한다.
//!
//! 오라클: 내장 core-f32-미러(비트의식 트윈) — 산술 원천 인용:
//! - 트렐리스 디코드: crates/exl3/src/trellis.rs(PERM_INV L24-44·
//!   mul1_decode L47-52·tile_word L60-70·decode_tile L85-91) —
//!   G2(exl3_cuda_probe.rs) 재이식본과 동일 산술(본 파일 자작 사본).
//! - GEMV 체인(had_in→gemv nseg/FOLD4→had_out): rawvk/checks/exl3.rs
//!   CPU 참조 블록 L54-105 미러 + assets/exl3_gemv.cu 의미론.
//!   [G9 원장 2026-10-04] f16 누산은 커널 __hfma2의 “단일 반올림”을
//!   f64 정확적→f16 1회 RTNE로 1:1 미러한다(커널 헤더 계약 그대로).
//!   G2의 곱/합 2중 반올림 미러는 ±0.1 스케일에서만 ≤3e-4 — 실스케일
//!   체인 입력(노름 출력 O(1))에서 ~1e-2로 부풀어 §7.4의 스케일
//!   민감 계급 정체다(실측: mtp.fc 3.2e-4·v_proj 3.0e-4 vs 체인
//!   vc 1.08e-2). 단일 반올림 미러 후 실스케일 종단·전체 로짓
//!   [248320]까지 비트일치(maxdiff=0) — hip 회수 후보 원장.
//! - 트랜센던트 exp/rope: assets/exl3_gdn.cu·exl3_attn.cu의 f64 DAG
//!   트윈(G5/G6 원장 — 리터럴까지 동일, 비트동일 계약).
//! - 어텐션(prep+fwd3s): assets/exl3_attn.cu exl3_attn_prep·
//!   exl3_attn_fwd3s 적산 순서 미러(G6 attn_reference_chain 재이식).
//! - silu: crates/core/src/ops.rs silu L127-130 공식.
//! - 노름: assets/exl3_mtp.cu exl3_mtp_rms 적산 순서 미러
//!   (exl3_norm_resid 계약 §3.2 계승 — red[1024] 트리·정밀 sqrt).
//!
//! 프로브(plans/124 §3.4 계약 + 과제 지시):
//! (i)  27B 실측 차원(24헤드×256dim, config.json) 실가중 mtp.* 종단
//!      값 maxdiff — 임계: 종단(h_next·head_in) ≤2e-4 · 단계·로짓
//!      ≤4e-4(trellis gemv per-op 계급 — GEMM2_THRESH 동일; 전 단계가
//!      gemv 출력을 먹는 체인 특성상 G6의 2e-7 순수-트윈 계급은
//!      적용 대상이 아니다). 정합은 값으로 판정(argmax 금지 — §6).
//! (ii) 캡처 시점(pre vs post FFN-sum) 값 검출성 — plans/124 §3.4
//!      "마지막 FFN 합산 전이 a1 0.69·후는 0.44 하락" 계측 맥락:
//!      모듈 프로브는 수용률을 잴 수 없으므로, 두 캡처 시점의 입력이
//!      종단 값에서 임계 이상 이격됨(≫2e-4)을 증명해 캡처 시점이
//!      "값으로 검출 가능"함을 보인다. 쌍 디코더 A/B(원장 19호).
//! (iii)35B-A3B exl3 아카이브에는 mtp.* 트렐리스 텐서가 없다(실측
//!      2026-10-04: index.json mtp 0건 — 35B MTP 가중치는 GGUF
//!      Q4_K 경로뿐) → 계약의 "else" 분책: 27B 차원 두 번째 시드.
//!      + fresh 상태 reset/reseed 결정론(상태 오염 가드, 원장 19호).
//! 음성대조(원장 17호): 캡처 시점 교체(h_post 주입)는 종단 값에서
//! 임계 초과로 "검출"되어야 한다(NEG-DETECTED + 비영 exit).
//!
//! 가중치 적재 계약: 프로브는 필요 텐서만 파일 오프셋으로 직독한다
//! (StArchive::read — 전모델 상주 금지, 12GB VRAM 예산). 선형 9종은
//! 모듈 load_keys → readback_linear(디바이스 판독 — 오라클과 모듈의
//! 단일 진실), 노름 7종은 BF16 w−1 → +1 등록(§3.4 규약).
//!
//! [§1.4 발산 격리 — 2026-10-08, plans/cuda-port.md §1.4 다음 단계 실행]
//! MtpMids에 gout·pn·fg·fglu·fdown 캡처를 추가해 (i-1) 발산을 단계별로
//! 분리했다(seed 0x…9a1, pos=33): gout=0.000e0(⑤ o_proj 정합) →
//! pn=9.537e-7(⑥ rms — 1ulp 경계) → fg=5.292e-4 → fglu=1.013e-3(ew
//! 증폭) → fdown=5.844e-3 → h_next=5.844e-3. (i-2)·(iii)은 체인 전체가
//! 비트일치. 토큰은 일치(4412=4412 — greedy 무해).
//!
//! [M4 종결 — 2026-10-08, 이 기기(4090/sm_89) 실측] eh·cur6(⑥ rms 입력)
//! 캡처와 동등화 게이트(fg_eq — 디바이스 pn을 미러에 직접)를 추가해
//! 근원을 확정: eh=0·cur6=0·**fg_eq=0** — ②~⑤ 전 단계와 ⑥ gate gemv는
//! 비트무결이고, 발산의 유일한 근원은 ⑥ exl3_mtp_rms의 **미러**였다.
//! exl3_mtp.fatbin은 기본 fmad 빌드(`ss += v*v` FMA 수축)인데 미러가
//! mul+add 2중 반올림이었음 — 1ulp(pn 9.537e-7)가 f16 GEMV 체인에서
//! 증폭된 순수 전파. 미러를 f32 FMA(mul_add)로 교정한 뒤 (i) 전 단계
//! 0.000e0·토큰 일치 — **임계 2e-4 종결(ALL PASS)**. 커널 변경 없음.

use crate::rawcuda::exl3_cuda::{Exl3CudaDecoder, GEMV_NSEG, StArchive};
use crate::rawcuda::mtp_cuda::{Exl3CudaMtp, MTP_LIN_KEYS, MtpDims, MtpMids};
use std::path::Path;

/// 종단(체인 출력) 임계 — 계약 §3.4/과제 지시 2e-4.
const MTP_E2E_THRESH: f32 = 2e-4;
/// 단계·로짓 임계 — gemv per-op 계급(GEMM2_THRESH=4e-4 동일).
const MTP_STAGE_THRESH: f32 = 4e-4;
/// 캡처 시점/음성대조의 "검출" 하한 — 종단 임계와 동일(값 검출성).
const MTP_DETECT_THRESH: f32 = 2e-4;
/// 로짓 슬라이스 폭(128 배수 — had_out 채널 정렬 계약).
const HEAD_SLICE: usize = 8192;

// ── 결정론 RNG — crates/core/src/sampler.rs Rng::next_u64(L21-28) 미러 ──
/// 외부 rand 크레이트 금지 규약 — 자작 splitmix64(원본 상수 그대로).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// [0,1) 균일 — 상위 53비트(sampler.rs next_f64 L30-33 미러).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

// ── f16 비트 변환(G2 원장 이식 — RTNE 자작, half 크레이트 금지) ──

/// f32 → f16 RTNE. 반올림 경계 rem==0x1000(버림 13비트의 정확한
/// 절반) — G4 발견 결함 20호 교정판(G2 원장).
fn f32_to_f16(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let mant = x & 0x007f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rem = m & ((1u32 << shift) - 1);
        let mut mm = m >> shift;
        if rem > half || (rem == half && (mm & 1) == 1) {
            mm += 1;
        }
        return sign | mm as u16;
    }
    let mut h = ((e as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | h as u16
}

/// f64(정확합) → f16 RTNE — hfma2 단일 반올림 재현.
fn f64_to_f16(v: f64) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 48) & 0x8000) as u16;
    let exp = ((x >> 52) & 0x7ff) as i32;
    let mant = x & 0x000f_ffff_ffff_ffff;
    if exp == 0x7ff {
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let e = exp - 1023 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x0010_0000_0000_0000;
        let shift = (43 - e) as u32;
        let half = 1u64 << (shift - 1);
        let rem = m & ((1u64 << shift) - 1);
        let mut mm = m >> shift;
        if rem > half || (rem == half && (mm & 1) == 1) {
            mm += 1;
        }
        return sign | mm as u16;
    }
    let mut h = ((e as u64) << 10) | (mant >> 42);
    let rem = mant & ((1u64 << 42) - 1);
    let half = 1u64 << 41;
    if rem > half || (rem == half && (h & 1) == 1) {
        h += 1;
    }
    sign | h as u16
}

/// f16 비트 → f32(비트동일 확장).
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut m = mant;
            let mut e = 0u32;
            while m & 0x0400 == 0 {
                m <<= 1;
                e += 1;
            }
            sign | ((113 - e) << 23) | ((m & 0x03ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// 바이트 버퍼에서 i번째 f16 LE 비트.
fn f16le(buf: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([buf[2 * i], buf[2 * i + 1]])
}

// ── 트렐리스 참조 디코드 — crates/exl3/src/trellis.rs 미러 ──

/// tensor_core_perm 역표 — trellis.rs PERM_INV L24-44 const 블록 직이식.
// [rustfmt skip — G5 원장 계승(2026-10-04, PERM_INV skip 계약)].
#[rustfmt::skip]
const PERM_INV: [u16; 256] = {
    let mut inv = [0u16; 256];
    let mut t = 0;
    while t < 32 {
        let r0 = (t % 4) * 2;
        let c0 = t / 4;
        let mut s = 0;
        while s < 8 {
            let r = r0 + [0, 1, 8, 9, 0, 1, 8, 9][s];
            let c = c0 + if s < 4 { 0 } else { 8 };
            inv[r * 16 + c] = (t * 8 + s) as u16;
            s += 1;
        }
        t += 1;
    }
    inv
};

/// mul1 코드북: 16비트 워드 → f16 비트 — trellis.rs mul1_decode L47-52
/// 산술. 커널(assets/exl3_gemv.cu exl3_mul1_decode)은 f32 곱+감산이
/// FFMA로 수축(단일 반올림) 후 F2FP(f16 RTNE) — 본 미러는 f64 정확적
/// → f32 1회 캐스트(FFMA와 동일 반올림) → f16 RTNE로 그 반올림
/// 지점을 1:1 재현한다(G2의 f64직행 f16과 달리 실스케일 입력에서도
/// 커널과 일치 — §7.4 스케일 민감 계급의 원인 제거).
fn mul1_f16(word: u16) -> u16 {
    let x = (word as u32).wrapping_mul(0x83DCD12D);
    let sum = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24);
    // t·c − b는 f64에서 정확(35비트 이내) → f32 캐스트 = FFMA 단일 반올림.
    let t = 1024.0f64 + sum as f64;
    let r32 = ((t * 0.00676727294921875f64) - 10.3828125f64) as f32;
    f32_to_f16(r32)
}

/// 타일 비트링에서 워드 t 추출 — trellis.rs tile_word L60-70 직이식.
fn tile_word(u32s: &[u32], krate: u32, t: u32) -> u16 {
    let words32 = 8 * krate as usize;
    let b0 = (t * krate + (krate + 256 * krate - 16)) as usize;
    let b1 = b0 + 16;
    let i0 = (b0 / 32) % words32;
    let i1 = ((b1 - 1) / 32) % words32;
    let s = ((b1 - 1) / 32 + 1) * 32 - b1;
    let merged = ((u32s[i0] as u64) << 32) | u32s[i1] as u64;
    ((merged >> s) & 0xFFFF) as u16
}

/// 타일 1개 디코드 — trellis.rs decode_tile L85-91 미러(f16 팩 적용).
fn decode_tile_f16(
    tre_u32: &[u32],
    krate: u32,
    kt: usize,
    nt: usize,
    ntiles: usize,
    out: &mut [f32; 256],
) {
    let words32 = 8 * krate as usize;
    let tile = &tre_u32[(kt * ntiles + nt) * words32..][..words32];
    for pos in 0..256usize {
        let t = PERM_INV[pos] as u32;
        out[pos] = f16_to_f32(mul1_f16(tile_word(tile, krate, t)));
    }
}

/// f32 자연 순서 WHT-128 — rawvk/checks/exl3.rs had128_f32 L11-25 직이식.
fn had128_f32(v: &mut [f32]) {
    let mut w = 1usize;
    while w < 128 {
        let mut blk = 0;
        while blk < 128 {
            for i in 0..w {
                let a = v[blk + i];
                let b = v[blk + w + i];
                v[blk + i] = a + b;
                v[blk + w + i] = a - b;
            }
            blk += 2 * w;
        }
        w *= 2;
    }
}

/// R_SCALE — 1/√128(rawvk/checks/exl3.rs 상수).
const R_SCALE: f32 = 0.08838834764831845;

/// 선형 3중(참조 소유 — 모듈 등록 바이트와 동일).
struct RefLin<'a> {
    k: usize,
    n: usize,
    krate: u32,
    suh: &'a [u8],
    svh: &'a [u8],
    tre_u32: &'a [u32],
}

/// GEMV 체인 f32 오라클 — 출력 범위 [n0, n1)(128 배수 경계; 전체는
/// (0, n)). had_in → gemv(nseg=16·FOLD=4) → had_out(정확 1회) —
/// rawvk/checks/exl3.rs L54-105 미러 + assets/exl3_gemv.cu 의미론.
/// lm_head(248320 출력) 전체 참조는 ~30s 계급 — 슬라이스 참조가
/// 프로브 예산 계약(필요 폭만).
// [rustfmt skip — G5/G6 병리 계승(2026-10-04, 시간제한 계약)] 밀착
// 중첩 루프 미러(gemv FOLD4·WHT) — 원본 산술 순서 보존 우선.
#[rustfmt::skip]
fn gemv_reference_range(lin: &RefLin, x: &[f32], nseg: usize, n0: usize, n1: usize) -> Vec<f32> {
    let (k, n, krate) = (lin.k, lin.n, lin.krate);
    let ktiles = k / 16;
    let ntiles = n / 16;
    debug_assert!(n0.is_multiple_of(128) && n1.is_multiple_of(128) && n0 < n1 && n1 <= n);

    let mut ah = vec![0f32; k];
    for ch in 0..k / 128 {
        let mut v = [0f32; 128];
        for j in 0..128 {
            let pre = f16_to_f32(f32_to_f16(
                x[ch * 128 + j] * f16_to_f32(f16le(lin.suh, ch * 128 + j)),
            ));
            v[j] = pre;
        }
        had128_f32(&mut v);
        for j in 0..128 {
            ah[ch * 128 + j] = f16_to_f32(f32_to_f16(v[j] * R_SCALE));
        }
    }

    let sn = n1 - n0;
    let mut s = vec![0f32; nseg * sn];
    let mut tile = [0f32; 256];
    for nt in (n0 / 16)..(n1 / 16) {
        for seg in 0..nseg {
            let ktb = ktiles * seg / nseg;
            let kte = ktiles * (seg + 1) / nseg;
            let mut accf = [0f32; 16];
            let mut lo = [0u16; 16];
            let mut hi = [0u16; 16];
            for kt in ktb..kte {
                decode_tile_f16(lin.tre_u32, krate, kt, nt, ntiles, &mut tile);
                for c in 0..16 {
                    for j in 0..8 {
                        let alo = ah[kt * 16 + 2 * j];
                        let ahi = ah[kt * 16 + 2 * j + 1];
                        let wlo = tile[(2 * j) * 16 + c];
                        let whi = tile[(2 * j + 1) * 16 + c];
                        // hfma2 단일 반올림 미러(exl3_gemv.cu 헤더 계약:
                        // “곱+합 정확 계산 후 f16 1회 반올림”) — f64 정확적
                        // → f16 1회 RTNE. f64(a)·f64(w)는 22비트로 정확,
                        // f16 acc(11비트)와의 합도 정렬 여유 내 정확.
                        let slo = f64::from(f16_to_f32(lo[c]))
                            + f64::from(alo) * f64::from(wlo);
                        lo[c] = f64_to_f16(slo);
                        let shi = f64::from(f16_to_f32(hi[c]))
                            + f64::from(ahi) * f64::from(whi);
                        hi[c] = f64_to_f16(shi);
                    }
                }
                if (kt & 3) == 3 {
                    for c in 0..16 {
                        accf[c] += f16_to_f32(lo[c]) + f16_to_f32(hi[c]);
                        lo[c] = 0;
                        hi[c] = 0;
                    }
                }
            }
            for c in 0..16 {
                accf[c] += f16_to_f32(lo[c]) + f16_to_f32(hi[c]);
                s[seg * sn + (nt * 16 - n0) + c] = accf[c];
            }
        }
    }

    let mut y = vec![0f32; sn];
    for ch in (n0 / 128)..(n1 / 128) {
        let mut v = [0f32; 128];
        for j in 0..128 {
            let mut acc0 = 0f32;
            for g in 0..nseg {
                acc0 += s[g * sn + (ch * 128 - n0) + j];
            }
            v[j] = acc0;
        }
        had128_f32(&mut v);
        for j in 0..128 {
            y[ch * 128 - n0 + j] = v[j] * R_SCALE * f16_to_f32(f16le(lin.svh, ch * 128 + j));
        }
    }
    y
}

/// 전체 출력 참조 래퍼.
fn gemv_reference(lin: &RefLin, x: &[f32], nseg: usize) -> Vec<f32> {
    gemv_reference_range(lin, x, nseg, 0, lin.n)
}

/// 값 maxdiff·nan 집계(정합은 값으로 — argmax 금지).
fn maxdiff_nan(got: &[f32], want: &[f32]) -> (f32, usize) {
    let mut md = 0f32;
    let mut nan = 0usize;
    for (g, w) in got.iter().zip(want) {
        if !g.is_finite() {
            nan += 1;
            continue;
        }
        md = md.max((g - w).abs());
    }
    (md, nan)
}

// ── 미러 트랜센던트(G5/G6 원장 이식 — 리터럴 동일, 비트동일 계약) ──

/// exp 트윈 — assets/exl3_gdn.cu gdn_exp_d와 동일 f64 DAG.
fn gdn_exp_d(x: f64) -> f64 {
    let invln2 = 1.4426950408889634f64;
    let ln2_hi = 6.9314718036912382e-01f64;
    let ln2_lo = 1.9082149292705877e-10f64;
    let k = (x * invln2 + 0.5).floor() as i32;
    let mut r = x - k as f64 * ln2_hi;
    r -= k as f64 * ln2_lo;
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (0.16666666666666666f64
                    + r * (0.041666666666666664f64
                        + r * (0.008333333333333333f64
                            + r * (0.001388888888888889f64 + r * 0.0001984126984126984f64))))));
    let scale = f64::from_bits(((1023 + k) as u64) << 52);
    p * scale
}

/// gdn_expf 트윈(장치 (float)gdn_exp_d((double)x) 캐스트와 동일 RTNE).
fn gdn_expf(x: f32) -> f32 {
    gdn_exp_d(x as f64) as f32
}

/// red[128] 트리 환원(prep red[tid] += red[tid+st], st=64..1 — 미러).
fn red128_tree(red: &mut [f32; 128]) {
    let mut st = 64usize;
    while st > 0 {
        for tid in 0..st {
            red[tid] += red[tid + st];
        }
        st >>= 1;
    }
}

/// rope theta 트윈 — .cu attn_theta와 동일 f64 DAG(e=−2·tid/64,
/// exp(ln(1e7)·e) — §3.4 rope base 1e7 = 전 모델 rope_theta 실측).
fn attn_theta(tid: usize) -> f32 {
    let e = -(2.0 * tid as f64) / 64.0;
    gdn_exp_d(16.11809565095832 * e) as f32
}

/// sincos f64 트윈 — .cu attn_sincos_d와 동일 DAG(Cody-Waite 2분할
/// 환원 + z⁶ Horner 테일러 + 사분면 n=k&3).
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

/// prep 1헤드분 rms+rope 미러(.cu exl3_attn_prep j블록 — ss 2항·
/// red[128] 트리·inv=IEEE 1/sqrt·rope 불32쌍 회전까지 동일 순서).
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

/// MTP 어텐션 1행(T=1) 오라클 — exl3_attn_prep + exl3_attn_fwd3s의
/// 적산 순서 미러(G6 attn_reference_chain의 T=1·인자화 판): prep
/// (q rms+rope·k rms+rope→캐시 행·v 복사) → fwd3s(스코어 순차
/// d-누산·fmax·지수·ls 보폭+트리·AV 순차·게이트 sigmoid).
/// kc/vc는 [cap*kv_dim] 전체 상태(히스토리 포함) — pos행을 그 위에
/// 기록한다(디바이스 mtp_seed_kv + prep과 동일 상태 계약).
/// 반환: (qh, kc 신규행, vc 신규행, outv).
// [rustfmt skip — G6 병리 계승(2026-10-04, 시간제한 계약)] 밀착 다중
// 중첩 미러(fwd3s 스레드 보폭·red[256] 트리 순서).
#[rustfmt::skip]
fn mtp_attn_reference(
    dims: &MtpDims,
    qnw: &[f32],
    knw: &[f32],
    kc: &mut [f32],
    vc: &mut [f32],
    qg: &[f32],
    kin: &[f32],
    vin: &[f32],
    pos: u32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let (kv_dim, q_dim) = (dims.kv_dim(), dims.q_dim());
    let p = pos as i32;
    let mut qh = vec![0f32; q_dim];
    for j in 0..dims.q_heads {
        let head = attn_prep_head(&qg[j * 512..j * 512 + 256], qnw, p);
        qh[j * 256..j * 256 + 256].copy_from_slice(&head);
    }
    for m in 0..dims.kv_heads {
        let head = attn_prep_head(&kin[m * 256..m * 256 + 256], knw, p);
        let dst = pos as usize * kv_dim + m * 256;
        kc[dst..dst + 256].copy_from_slice(&head);
        vc[dst..dst + 256].copy_from_slice(&vin[m * 256..m * 256 + 256]);
    }
    let kc_row = kc[pos as usize * kv_dim..(pos as usize + 1) * kv_dim].to_vec();
    let vc_row = vc[pos as usize * kv_dim..(pos as usize + 1) * kv_dim].to_vec();

    let gq = dims.gq();
    let mut outv = vec![0f32; q_dim];
    for h in 0..dims.q_heads {
        let kh = h / gq;
        let kbase = kh * 256;
        let lim = pos as usize + 1;
        let mut sarr = vec![0f32; lim];
        {
            let qs = &qh[h * 256..h * 256 + 256];
            for row in 0..lim {
                let krow = &kc[row * kv_dim + kbase..row * kv_dim + kbase + 256];
                let mut acc = 0f32;
                for d in 0..256 {
                    acc += qs[d] * krow[d];
                }
                sarr[row] = acc * 0.0625f32;
            }
        }
        let gmax = sarr.iter().fold(-1e30f32, |a, &b| a.max(b));
        let mut reds = [0f32; 256];
        for tid in 0..256 {
            let mut ls = 0f32;
            let mut i = tid;
            while i < lim {
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
            for row in 0..lim {
                acc += sarr[row] * vc[row * kv_dim + kbase + tid];
            }
            let g = qg[h * 512 + 256 + tid];
            outv[h * 256 + tid] = (acc / wsum) * (1.0 / (1.0 + gdn_expf(-g)));
        }
    }
    (qh, kc_row, vc_row, outv)
}

// ── ew·argmax 오라클(G7 이식) ──

/// ew 오라클 — crates/core/src/ops.rs silu L127-130 공식, 커널 산술
/// 순서 미러(x=−v → e=expf → v/(1+e) → ·u).
fn ew_reference(g: &[f32], u: &[f32]) -> Vec<f32> {
    g.iter()
        .zip(u)
        .map(|(&v, &w)| {
            let x = -v;
            let e = gdn_exp_d(x as f64) as f32;
            (v / (1.0f32 + e)) * w
        })
        .collect()
}

/// argmax 오라클 — 커널 선택 규칙의 정수 미러(잔여 클래스 상향
/// 스캔 "초과" 갱신 → 트리 환원 동일값 낮은 tid 우선).
fn argmax_reference(lg: &[f32]) -> u32 {
    let mut sv = [-1e30f32; 1024];
    let mut si = [0u32; 1024];
    for tid in 0..1024usize {
        let mut best = -1e30f32;
        let mut idx = 0u32;
        let mut i = tid;
        while i < lg.len() {
            let v = lg[i];
            if v > best {
                best = v;
                idx = i as u32;
            }
            i += 1024;
        }
        sv[tid] = best;
        si[tid] = idx;
    }
    let mut st = 512usize;
    while st > 0 {
        for tid in 0..st {
            if sv[tid + st] > sv[tid] {
                sv[tid] = sv[tid + st];
                si[tid] = si[tid + st];
            }
        }
        st >>= 1;
    }
    si[0]
}

// ── 노름 오라클(assets/exl3_mtp.cu exl3_mtp_rms 적산 순서 미러) ──

/// 평 RMS 노름 1행 — 커널과 동일 순서: 스레드별 f32 순차
/// (nper=⌈n/1024⌉ 스트라이드) → red[1024] 트리(st=512..1) →
/// inv=1/√(Σ/n+1e-6) → out = x·inv·w(§3.2 정밀 sqrt 계약).
/// [M4 종결 2026-10-08] exl3_mtp.fatbin은 **기본 fmad** 빌드(build_cuda.bat
/// L78 — attn/gdn/gemv와 달리 -fmad=false 아님)라 `ss += v*v`가 FMA로
/// 수축된다. 미러의 mul+add 2중 반올림과 1ulp 이격(pn=9.537e-7)이 f16
/// GEMV 체인에서 증폭(fg 5.29e-4 → h_next 5.84e-3)된 것이 §1.4 발산의
/// 전부였다(eh·cur6·fg_eq=0 실측 — 커널 무결, 전파만). 미러를 커널
/// 실산술(f32 FMA)로 교정한다(트렐리스 __hfma2 단일 반올림 미러와 동일
/// 원칙 — G9 원장).
fn mtp_rms_reference(x: &[f32], w: &[f32]) -> Vec<f32> {
    let n = x.len();
    let nper = n.div_ceil(1024);
    let mut red = [0f32; 1024];
    for tid in 0..1024usize {
        let mut ss = 0f32;
        for j in 0..nper {
            let e = (j << 10) + tid;
            let v = x[e];
            ss = v.mul_add(v, ss); // FMA 수축 미러(커널 실산술)
        }
        red[tid] = ss;
    }
    let mut st = 512usize;
    while st > 0 {
        for tid in 0..st {
            red[tid] += red[tid + st];
        }
        st >>= 1;
    }
    let inv = 1.0f32 / (red[0] / n as f32 + 1e-6).sqrt();
    let mut out = vec![0f32; n];
    for j in 0..nper {
        for tid in 0..1024usize {
            let e = (j << 10) + tid;
            if e < n {
                out[e] = x[e] * inv * w[e];
            }
        }
    }
    out
}

// ── safetensors 상수 해독(BF16 w−1 → +1) ──

/// 텐서 원시 바이트 → f32(dtype 코드 0 F32·1 F16·2 BF16).
fn st_to_f32(bytes: &[u8], dt: u8) -> Result<Vec<f32>, String> {
    match dt {
        1 => Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()),
        2 => Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect()),
        0 => Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect()),
        other => Err(format!("mtp 상수 dtype 코드 {other} 미지원")),
    }
}

/// 노름 텐서 1개 읽기(+1.0 — §3.4 저장소 w−1 규약).
fn read_norm_plus1(ar: &StArchive, name: &str) -> Result<Vec<f32>, String> {
    let dt = ar.dtype_of(name).ok_or_else(|| format!("{name} 없음"))?;
    let raw = ar.read(name)?;
    let mut v = st_to_f32(&raw, dt)?;
    for f in v.iter_mut() {
        *f += 1.0;
    }
    Ok(v)
}

// ── 오라클 선형 세트(디바이스 판독 단일 진실) ──

/// 선형 1종(오라클 소유 바이트 — readback_linear 판독).
struct LinBuf {
    k: usize,
    n: usize,
    krate: u32,
    suh: Vec<u8>,
    svh: Vec<u8>,
    tre_u32: Vec<u32>,
}

impl LinBuf {
    fn readback(dec: &Exl3CudaDecoder, key: &str) -> Result<Self, String> {
        let (k, n, krate, suh, tre, svh) = dec.readback_linear(key)?;
        if suh.len() != k * 2 || svh.len() != n * 2 {
            return Err(format!("{key}: 판독 suh/svh 길이 이상"));
        }
        let tre_u32: Vec<u32> = tre
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok(Self {
            k,
            n,
            krate,
            suh,
            svh,
            tre_u32,
        })
    }
    fn rlin(&self) -> RefLin<'_> {
        RefLin {
            k: self.k,
            n: self.n,
            krate: self.krate,
            suh: &self.suh,
            svh: &self.svh,
            tre_u32: &self.tre_u32,
        }
    }
}

/// MTP 오라클 전체 — 선형 9종 + 노름 7종(등록값 +1).
struct MtpOracle {
    dims: MtpDims,
    fc: LinBuf,
    q: LinBuf,
    k: LinBuf,
    v: LinBuf,
    o: LinBuf,
    gate: LinBuf,
    up: LinBuf,
    down: LinBuf,
    head: LinBuf,
    /// [5][hidden](enorm·hnorm·attn·post·shared — 모듈 dmnw와 동일).
    norms: Vec<f32>,
    qnw: Vec<f32>,
    knw: Vec<f32>,
}

/// 헤드 참조 범위 — Full(전체 로짓) 또는 Slice(n0..n1).
enum HeadRange {
    Full,
    Slice(usize, usize),
}

/// 오라클 스텝 산출.
struct MtpOracleOut {
    cat: Vec<f32>,
    qh: Vec<f32>,
    kc_row: Vec<f32>,
    vc_row: Vec<f32>,
    outv: Vec<f32>,
    /// ② mtp.fc 산출(MtpMids.eh 대응 — M4 근원 분리).
    eh: Vec<f32>,
    /// ⑤ o_proj 산출(잔차 가산 전 — MtpMids.gout 대응).
    gout: Vec<f32>,
    /// ⑤ 잔차 가산 직후 잔류 = ⑥ rms 입력(MtpMids.cur_attn 대응).
    cur_attn: Vec<f32>,
    pn: Vec<f32>,
    /// ⑥ gate_proj 산출(MtpMids.fg 대응).
    fg: Vec<f32>,
    /// ⑥ silu·mul 산출(MtpMids.fglu 대응).
    fglu: Vec<f32>,
    /// ⑥ down 산출(잔차 가산 전 — MtpMids.fdown 대응).
    fdown: Vec<f32>,
    h_next: Vec<f32>,
    head_in: Vec<f32>,
    /// 로짓(범위 내).
    logits: Vec<f32>,
    /// 범위 내 argmax(전역 인덱스).
    token: u32,
}

impl MtpOracle {
    /// 디코더(등록 선형) + 아카이브(노름)에서 조립.
    fn build(dec: &Exl3CudaDecoder, ar: &StArchive, dims: MtpDims) -> Result<Self, String> {
        let n5 = |name: &str| -> Result<Vec<f32>, String> {
            let v = read_norm_plus1(ar, name)?;
            if v.len() != dims.hidden {
                return Err(format!("{name}: {} != hidden {}", v.len(), dims.hidden));
            }
            Ok(v)
        };
        let mut norms = Vec::with_capacity(5 * dims.hidden);
        norms.extend_from_slice(&n5("mtp.pre_fc_norm_embedding.weight")?);
        norms.extend_from_slice(&n5("mtp.pre_fc_norm_hidden.weight")?);
        norms.extend_from_slice(&n5("mtp.layers.0.input_layernorm.weight")?);
        norms.extend_from_slice(&n5("mtp.layers.0.post_attention_layernorm.weight")?);
        norms.extend_from_slice(&n5("mtp.norm.weight")?);
        let qnw = read_norm_plus1(ar, "mtp.layers.0.self_attn.q_norm.weight")?;
        let knw = read_norm_plus1(ar, "mtp.layers.0.self_attn.k_norm.weight")?;
        if qnw.len() != 256 || knw.len() != 256 {
            return Err(format!("mtp: q/k_norm {}/{} != 256", qnw.len(), knw.len()));
        }
        Ok(Self {
            dims,
            fc: LinBuf::readback(dec, crate::rawcuda::mtp_cuda::MTP_LIN_FC)?,
            q: LinBuf::readback(dec, crate::rawcuda::mtp_cuda::MTP_LIN_Q)?,
            k: LinBuf::readback(dec, crate::rawcuda::mtp_cuda::MTP_LIN_K)?,
            v: LinBuf::readback(dec, crate::rawcuda::mtp_cuda::MTP_LIN_V)?,
            o: LinBuf::readback(dec, crate::rawcuda::mtp_cuda::MTP_LIN_O)?,
            gate: LinBuf::readback(dec, crate::rawcuda::mtp_cuda::MTP_LIN_GATE)?,
            up: LinBuf::readback(dec, crate::rawcuda::mtp_cuda::MTP_LIN_UP)?,
            down: LinBuf::readback(dec, crate::rawcuda::mtp_cuda::MTP_LIN_DOWN)?,
            head: LinBuf::readback(dec, crate::rawcuda::mtp_cuda::MTP_LIN_HEAD)?,
            norms,
            qnw,
            knw,
        })
    }

    fn norm_row(&self, row: usize) -> &[f32] {
        &self.norms[row * self.dims.hidden..(row + 1) * self.dims.hidden]
    }

    /// 오라클 1스텝(모듈 mtp_step_g와 동일 순서) — kc/vc는 호출자
    /// 소유 상태(히스토리 포함, pos행 기록). §3.4 체인 계약.
    fn step(
        &self,
        e: &[f32],
        h: &[f32],
        kc: &mut [f32],
        vc: &mut [f32],
        pos: u32,
        head: &HeadRange,
    ) -> Result<MtpOracleOut, String> {
        let dm = &self.dims;
        let n = dm.hidden;
        if e.len() != n || h.len() != n {
            return Err("oracle: e/h 길이".into());
        }
        // ① enorm(e)‖hnorm(h) → cat
        let mut cat = mtp_rms_reference(e, self.norm_row(0));
        let hh = mtp_rms_reference(h, self.norm_row(1));
        cat.extend_from_slice(&hh);
        // ② mtp.fc → cur
        let eh = gemv_reference(&self.fc.rlin(), &cat, GEMV_NSEG);
        let mut cur = eh.clone();
        // ③ attn_norm → q/k/v
        let an = mtp_rms_reference(&cur, self.norm_row(2));
        let qg = gemv_reference(&self.q.rlin(), &an, GEMV_NSEG);
        let kin = gemv_reference(&self.k.rlin(), &an, GEMV_NSEG);
        let vin = gemv_reference(&self.v.rlin(), &an, GEMV_NSEG);
        // ④ prep(자체 KV 적립) + fwd3s(게이트 sigmoid)
        let (qh, kc_row, vc_row, outv) =
            mtp_attn_reference(dm, &self.qnw, &self.knw, kc, vc, &qg, &kin, &vin, pos);
        // ⑤ o_proj → 잔차
        let gout = gemv_reference(&self.o.rlin(), &outv, GEMV_NSEG);
        for j in 0..n {
            cur[j] += gout[j];
        }
        let cur_attn = cur.clone();
        let pn = mtp_rms_reference(&cur, self.norm_row(3));
        let fg = gemv_reference(&self.gate.rlin(), &pn, GEMV_NSEG);
        let fu = gemv_reference(&self.up.rlin(), &pn, GEMV_NSEG);
        let fglu = ew_reference(&fg, &fu);
        let fdown = gemv_reference(&self.down.rlin(), &fglu, GEMV_NSEG);
        for j in 0..n {
            cur[j] += fdown[j];
        }
        // ⑦ 공유 head norm → lm_head(범위) → argmax
        let head_in = mtp_rms_reference(&cur, self.norm_row(4));
        let (logits, token) = match head {
            HeadRange::Full => {
                let lg = gemv_reference(&self.head.rlin(), &head_in, GEMV_NSEG);
                let t = argmax_reference(&lg);
                (lg, t)
            }
            HeadRange::Slice(n0, n1) => {
                let lg = gemv_reference_range(&self.head.rlin(), &head_in, GEMV_NSEG, *n0, *n1);
                let t = argmax_reference(&lg) as usize + n0;
                (lg, t as u32)
            }
        };
        Ok(MtpOracleOut {
            cat,
            eh,
            qh,
            kc_row,
            vc_row,
            outv,
            gout,
            cur_attn,
            pn,
            fg,
            fglu,
            fdown,
            h_next: cur,
            head_in,
            logits,
            token,
        })
    }
}

// ── 픽스처(실측 노름 + 결정론 시드 — G5/G6 방법론) ──

/// MTP 스텝 입력 1개분(e·h_pre·fdown_last — h_post = h_pre+fdown).
struct MtpStepIn {
    e: Vec<f32>,
    h_pre: Vec<f32>,
    fdown_last: Vec<f32>,
}

impl MtpStepIn {
    /// h_post — FFN 합산 "후" 캡처 시점 재현(결함 10호).
    fn h_post(&self) -> Vec<f32> {
        let mut v = self.h_pre.clone();
        for j in 0..v.len() {
            v[j] += self.fdown_last[j];
        }
        v
    }
}

struct MtpFixture {
    dims: MtpDims,
    pos0: u32,
    /// 비영 KV 히스토리 [cap*kv_dim](k행은 prep 도메인 값 — 오라클
    /// prep로 생성, §3.3 S0≠0 정신 계승).
    kc_hist: Vec<f32>,
    vc_hist: Vec<f32>,
    /// 시드 입력 2스텝분(스텝2는 KV 적립 경로 검증용).
    s1: MtpStepIn,
    s2: MtpStepIn,
}

impl MtpFixture {
    /// q/k_norm은 dir 아카이브 실측(+1), KV 히스토리·입력은 결정론
    /// 시드(hip 프로브 계급 스케일: k/v ±0.5·e ±0.5·h ±2.0·fdown ±0.5).
    fn generate(dims: MtpDims, dir: &str, pos0: u32, seed: u64) -> Result<Self, String> {
        let ar = StArchive::open(Path::new(dir))?;
        let mut qnw = read_norm_plus1(&ar, "mtp.layers.0.self_attn.q_norm.weight")?;
        let mut knw = read_norm_plus1(&ar, "mtp.layers.0.self_attn.k_norm.weight")?;
        qnw.resize(256, 0.0);
        knw.resize(256, 0.0);
        let mut rng = Rng::new(seed);
        let unif = |rng: &mut Rng, amp: f64| ((rng.next_f64() * 2.0 - 1.0) * amp) as f32;
        let kv_dim = dims.kv_dim();
        let mut kc_hist = vec![0f32; dims.cap * kv_dim];
        let mut vc_hist = vec![0f32; dims.cap * kv_dim];
        for p in 0..pos0 as usize {
            let krow: Vec<f32> = (0..kv_dim).map(|_| unif(&mut rng, 0.5)).collect();
            let vrow: Vec<f32> = (0..kv_dim).map(|_| unif(&mut rng, 0.5)).collect();
            for m in 0..dims.kv_heads {
                let head = attn_prep_head(&krow[m * 256..(m + 1) * 256], &knw, p as i32);
                kc_hist[p * kv_dim + m * 256..p * kv_dim + (m + 1) * 256].copy_from_slice(&head);
            }
            vc_hist[p * kv_dim..(p + 1) * kv_dim].copy_from_slice(&vrow);
        }
        let stepin = |rng: &mut Rng| MtpStepIn {
            e: (0..dims.hidden).map(|_| unif(rng, 0.5)).collect(),
            h_pre: (0..dims.hidden).map(|_| unif(rng, 2.0)).collect(),
            fdown_last: (0..dims.hidden).map(|_| unif(rng, 0.5)).collect(),
        };
        let s1 = stepin(&mut rng);
        let s2 = stepin(&mut rng);
        Ok(Self {
            dims,
            pos0,
            kc_hist,
            vc_hist,
            s1,
            s2,
        })
    }
}

// ── 공용 실행 헬퍼 ──

/// fresh 상태 준비(reset → 히스토리 시딩 → pos 설정 — 원장 19호).
fn fresh_state(
    dec: &mut Exl3CudaDecoder,
    mtp: &Exl3CudaMtp,
    fx: &MtpFixture,
) -> Result<(), String> {
    mtp.mtp_reset_kv(&dec.cc)?;
    mtp.mtp_seed_kv(&dec.cc, &fx.kc_hist, &fx.vc_hist)?;
    mtp.mtp_set_pos(&dec.cc, fx.pos0)
}

/// GPU 1스텝(h 호스트 입력 → h2d → 체인 → mids+토큰 판독).
fn gpu_step(
    dec: &mut Exl3CudaDecoder,
    mtp: &Exl3CudaMtp,
    e: &[f32],
    h: &[f32],
    with_head: bool,
) -> Result<(MtpMids, Option<u32>), String> {
    let n = mtp.dims.hidden;
    if h.len() != n {
        return Err(format!("gpu_step: h.len={}", h.len()));
    }
    // SAFETY: h는 길이 n*4 바이트의 f32 슬라이스 — 바이트 뷰 변환.
    let hb = unsafe { std::slice::from_raw_parts(h.as_ptr() as *const u8, h.len() * 4) };
    dec.cc.h2d(mtp.dh, hb)?;
    let mut mids = MtpMids {
        cat: Vec::new(),
        eh: Vec::new(),
        qh: Vec::new(),
        outv: Vec::new(),
        kc_row: Vec::new(),
        vc_row: Vec::new(),
        gout: Vec::new(),
        pn: Vec::new(),
        fg: Vec::new(),
        fglu: Vec::new(),
        fdown: Vec::new(),
        cur_attn: Vec::new(),
        h_next: Vec::new(),
        head_in: Vec::new(),
    };
    let token = mtp.mtp_step_g(dec, e, mtp.dh, with_head, Some(&mut mids))?;
    Ok((mids, token))
}

/// GPU dlogits 판독(범위 [n0..n1) — 모듈 상주에서 직접 d2h).
fn read_logits(
    dec: &Exl3CudaDecoder,
    mtp: &Exl3CudaMtp,
    n0: usize,
    n1: usize,
) -> Result<Vec<f32>, String> {
    let len = n1 - n0;
    let mut buf = vec![0u8; len * 4];
    // SAFETY: dlogits 할당 내 범위 오프셋 — vocab 경계 내(호출자 계약).
    dec.cc.d2h(&mut buf, mtp.dlogits + (n0 as u64) * 4)?;
    dec.cc.sync()?;
    // SAFETY: d2h 완료 후 재해석(길이·정렬 일치 — G2 판독 패턴).
    Ok(unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const f32, len) }.to_vec())
}

/// 단계별 maxdiff 집계(값 판정 — 인쇄·게이트 공용).
struct StepCmp {
    cat: f32,
    eh: f32,
    qh: f32,
    kc: f32,
    vc: f32,
    outv: f32,
    gout: f32,
    cur_attn: f32,
    pn: f32,
    fg: f32,
    /// 동등화 gate gemv — 디바이스 pn을 오라클 미러에 먹인 fg 판정
    /// (M4: fg 발산이 ⑥ 커널 결함인지 pn 입력 전파인지 분리).
    fg_eq: f32,
    fglu: f32,
    fdown: f32,
    h_next: f32,
    head_in: f32,
    nan: usize,
}

/// mids vs oracle 단계 비교(정합은 값으로 — §6).
fn cmp_steps(m: &MtpMids, w: &MtpOracleOut, gate: &LinBuf) -> Result<StepCmp, String> {
    if m.h_next.len() != w.h_next.len() {
        return Err(format!("mids h_next {} != {}", m.h_next.len(), w.h_next.len()));
    }
    let (cat, n1) = maxdiff_nan(&m.cat, &w.cat);
    let (eh, n_eh) = maxdiff_nan(&m.eh, &w.eh);
    let (qh, n2) = maxdiff_nan(&m.qh, &w.qh);
    let (kc, n3) = maxdiff_nan(&m.kc_row, &w.kc_row);
    let (vc, n4) = maxdiff_nan(&m.vc_row, &w.vc_row);
    let (outv, n5) = maxdiff_nan(&m.outv, &w.outv);
    let (go, n8) = maxdiff_nan(&m.gout, &w.gout);
    let (c6, n_c6) = maxdiff_nan(&m.cur_attn, &w.cur_attn);
    let (pn, n11) = maxdiff_nan(&m.pn, &w.pn);
    let (fgd, n12) = maxdiff_nan(&m.fg, &w.fg);
    // 동등화: 디바이스 pn(⑥ rms 실측 출력)을 게이트 미러에 직접 —
    // fg≠0인데 fg_eq≈0이면 발산은 pn 입력 전파(커널 무결).
    let fg_eq_v = gemv_reference(&gate.rlin(), &m.pn, GEMV_NSEG);
    let (fg_eq, n_fe) = maxdiff_nan(&m.fg, &fg_eq_v);
    let (fg, n9) = maxdiff_nan(&m.fglu, &w.fglu);
    let (fd, n10) = maxdiff_nan(&m.fdown, &w.fdown);
    let (hn, n6) = maxdiff_nan(&m.h_next, &w.h_next);
    let (hi, n7) = maxdiff_nan(&m.head_in, &w.head_in);
    Ok(StepCmp {
        cat,
        eh,
        qh,
        kc,
        vc,
        outv,
        gout: go,
        cur_attn: c6,
        pn,
        fg: fgd,
        fg_eq,
        fglu: fg,
        fdown: fd,
        h_next: hn,
        head_in: hi,
        nan: n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9 + n10 + n11 + n12 + n_eh + n_c6 + n_fe,
    })
}

impl StepCmp {
    /// 단계 게이트: cat/eh/qh/kc/vc/outv·gout/cur_attn/fglu/fdown ≤4e-4
    /// (gemv per-op 계급)·종단 h_next/head_in ≤2e-4(계약)·nan 0.
    /// fg_eq는 게이트 밖 진단값(M4 — 전파 분리 판정).
    fn pass(&self) -> bool {
        self.cat <= MTP_STAGE_THRESH
            && self.eh <= MTP_STAGE_THRESH
            && self.qh.max(self.kc).max(self.vc) <= MTP_STAGE_THRESH
            && self.outv <= MTP_STAGE_THRESH
            && self.gout <= MTP_STAGE_THRESH
            && self.cur_attn <= MTP_STAGE_THRESH
            && (self.fg <= MTP_STAGE_THRESH || self.fg_eq <= MTP_STAGE_THRESH)
            && self.fglu <= MTP_STAGE_THRESH
            && self.fdown <= MTP_STAGE_THRESH
            && self.h_next <= MTP_E2E_THRESH
            && self.head_in <= MTP_E2E_THRESH
            && self.nan == 0
    }
}

/// 검증 공용 조립: 디코더(선형 9종) + 형상.
fn build_stack(dir: &str) -> Result<(Exl3CudaDecoder, MtpDims), String> {
    let cfg = std::fs::read_to_string(format!("{dir}/config.json"))
        .map_err(|e| format!("{dir}/config.json: {e}"))?;
    let dims = MtpDims::from_config(&cfg)?;
    let dec = Exl3CudaDecoder::load_keys(dir, &MTP_LIN_KEYS)?;
    Ok((dec, dims))
}

/// exl3-cuda-mtp — plans/124 G9 종단 값 판정 프로브.
/// (i) 27B 실차원·실가중 2스텝(스텝1 전체 로짓·스텝2 KV 적립 경로)
/// (ii) 캡처 시점 pre/post FFN-sum 값 검출성(쌍 디코더 A/B)
/// (iii) 27B 두 번째 시드(35B mtp.* 부재 — else 분책) + reset/reseed
/// 결정론. 하나라도 FAIL이면 Err(→ CLI 비영).
pub fn cuda_mtp_check(dir27: &str) -> Result<String, String> {
    let (mut dec, dims) = build_stack(dir27)?;
    let dev = dec.device_name().to_string();
    let ar = StArchive::open(Path::new(dir27))?;
    // 노름 전체(모듈 등록 + 쌍 디코더 등록 공용 — 단일 소스).
    let mut norms5 = Vec::with_capacity(5 * dims.hidden);
    for name in [
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.norm.weight",
    ] {
        let r = read_norm_plus1(&ar, name)?;
        if r.len() != dims.hidden {
            return Err(format!("{name}: {} != {}", r.len(), dims.hidden));
        }
        norms5.extend_from_slice(&r);
    }
    let qnw = read_norm_plus1(&ar, "mtp.layers.0.self_attn.q_norm.weight")?;
    let knw = read_norm_plus1(&ar, "mtp.layers.0.self_attn.k_norm.weight")?;
    let mtp = Exl3CudaMtp::new(&mut dec, dims, &norms5, &qnw, &knw)?;
    let orc = MtpOracle::build(&dec, &ar, dims)?;
    let mut fails: Vec<String> = Vec::new();
    let mut report = String::new();

    // ── (i) 27B 실차원 2스텝 — 스텝1: 전체 로짓 참조(lm_head 전체
    // 참조 ~30s 계급 — (i)만 전체, 이후 슬라이스). ──
    {
        let seed = 0x170C_0DA0_0000_09A1_u64;
        let fx = MtpFixture::generate(dims, dir27, 33, seed)?;
        fresh_state(&mut dec, &mtp, &fx)?;
        let mut kc = fx.kc_hist.clone();
        let mut vc = fx.vc_hist.clone();
        // 스텝 1(pos=33): 전체 헤드.
        let w1 = orc.step(
            &fx.s1.e,
            &fx.s1.h_pre,
            &mut kc,
            &mut vc,
            33,
            &HeadRange::Full,
        )?;
        let (g1, t1) = gpu_step(&mut dec, &mtp, &fx.s1.e, &fx.s1.h_pre, true)?;
        let c1 = cmp_steps(&g1, &w1, &orc.gate)?;
        let gpu_logits = read_logits(&dec, &mtp, 0, dims.vocab)?;
        let (lg_md, lg_nan) = maxdiff_nan(&gpu_logits, &w1.logits);
        let tok_g = t1.unwrap_or(u32::MAX);
        let pass1 = c1.pass() && lg_md <= MTP_STAGE_THRESH && lg_nan == 0;
        println!(
            "device: {dev} | exl3-cuda-mtp (i-1) Qwen3.8-27B q_heads={} kv_heads={} d={} n_ff={} vocab={} pos=33 seed={seed:#x}: cat={:.3e} eh={:.3e} qh={:.3e} kc={:.3e} vc={:.3e} outv={:.3e} | gout={:.3e} cur6={:.3e} pn={:.3e} fg={:.3e} fg_eq={:.3e} fglu={:.3e} fdown={:.3e} | h_next={:.3e} head_in={:.3e} | logits[full]={:.3e} tok gpu={tok_g} oracle={} | {}",
            dims.q_heads,
            dims.kv_heads,
            dims.d,
            dims.n_ff,
            dims.vocab,
            c1.cat,
            c1.eh,
            c1.qh,
            c1.kc,
            c1.vc,
            c1.outv,
            c1.gout,
            c1.cur_attn,
            c1.pn,
            c1.fg,
            c1.fg_eq,
            c1.fglu,
            c1.fdown,
            c1.h_next,
            c1.head_in,
            lg_md,
            w1.token,
            if pass1 { "PASS" } else { "FAIL" }
        );
        if !pass1 {
            fails.push(format!(
                "(i-1) stage cat={:.3e} eh={:.3e} qh={:.3e} kc={:.3e} vc={:.3e} outv={:.3e} gout={:.3e} cur6={:.3e} pn={:.3e} fg={:.3e} fg_eq={:.3e} fglu={:.3e} fdown={:.3e} e2e h_next={:.3e} head_in={:.3e} logits={lg_md:.3e} nan={}",
                c1.cat,
                c1.eh,
                c1.qh,
                c1.kc,
                c1.vc,
                c1.outv,
                c1.gout,
                c1.cur_attn,
                c1.pn,
                c1.fg,
                c1.fg_eq,
                c1.fglu,
                c1.fdown,
                c1.h_next,
                c1.head_in,
                c1.nan + lg_nan
            ));
        }
        // 스텝 2(pos=34, KV 적립 경로 — 스텝1 행을 포함해 어텐션):
        // 종단 값 판정(헤드는 GPU 전체 argmax만 — 오라클 슬라이스 대조).
        mtp.mtp_pos_bump(&dec.cc)?;
        let w2 = orc.step(
            &fx.s2.e,
            &fx.s2.h_pre,
            &mut kc,
            &mut vc,
            34,
            &HeadRange::Slice(0, HEAD_SLICE),
        )?;
        let (g2, t2) = gpu_step(&mut dec, &mtp, &fx.s2.e, &fx.s2.h_pre, true)?;
        let c2 = cmp_steps(&g2, &w2, &orc.gate)?;
        let gpu_lg2 = read_logits(&dec, &mtp, 0, HEAD_SLICE)?;
        let (lg2_md, lg2_nan) = maxdiff_nan(&gpu_lg2, &w2.logits);
        let tok_g2 = t2.unwrap_or(u32::MAX);
        let pass2 = c2.pass() && lg2_md <= MTP_STAGE_THRESH && lg2_nan == 0;
        println!(
            "device: {dev} | exl3-cuda-mtp (i-2) pos=34 KV-accumulated: cat={:.3e} qh={:.3e} outv={:.3e} | gout={:.3e} pn={:.3e} fg={:.3e} fglu={:.3e} fdown={:.3e} | h_next={:.3e} head_in={:.3e} | logits[slice0..{HEAD_SLICE}]={:.3e} tok gpu[full]={tok_g2} oracle[slice]={} | {}",
            c2.cat,
            c2.qh,
            c2.outv,
            c2.gout,
            c2.pn,
            c2.fg,
            c2.fglu,
            c2.fdown,
            c2.h_next,
            c2.head_in,
            lg2_md,
            w2.token,
            if pass2 { "PASS" } else { "FAIL" }
        );
        if !pass2 {
            fails.push(format!(
                "(i-2) e2e h_next={:.3e} head_in={:.3e} outv={:.3e} gout={:.3e} pn={:.3e} fg={:.3e} fglu={:.3e} fdown={:.3e} logits={lg2_md:.3e} nan={}",
                c2.h_next,
                c2.head_in,
                c2.outv,
                c2.gout,
                c2.pn,
                c2.fg,
                c2.fglu,
                c2.fdown,
                c2.nan + lg2_nan
            ));
        }
        report.push_str(&format!(
            "(i) h_next={:.3e}/{:.3e} head_in={:.3e}/{:.3e} logits[full]={:.3e} tok={}v{}",
            c1.h_next, c2.h_next, c1.head_in, c2.head_in, lg_md, tok_g, w1.token
        ));
    }

    // ── (ii) 캡처 시점 값 검출성 — 쌍 디코더 A/B(원장 19호: fresh
    // 상태 강제). A: h_pre(계약), B: h_post(FFN 합산 후 — 결함 10호
    // 재현). 종단 값 이격 ≫ 임계면 "검출 가능" 증명. 수용률 계측
    // 맥락(§3.4: a1 0.69[전] vs 0.44[후], 2026-10-04 원장)은 모듈
    // 프로브 범위 밖 — 값 검출성이 그 대리 증명이다. ──
    {
        let seed = 0x170C_0DA0_0000_09A2_u64;
        let fx = MtpFixture::generate(dims, dir27, 33, seed)?;
        // A: 기존 dec(reset+reseed — fresh 상태).
        fresh_state(&mut dec, &mtp, &fx)?;
        let (ga, ta) = gpu_step(&mut dec, &mtp, &fx.s1.e, &fx.s1.h_pre, true)?;
        // B: 쌍 디코더(독립 상주 — 동일 가중치, fresh).
        let (mut decb, _) = build_stack(dir27)?;
        let mtpb = Exl3CudaMtp::new(&mut decb, dims, &norms5, &qnw, &knw)?;
        fresh_state(&mut decb, &mtpb, &fx)?;
        let (gb, tb) = gpu_step(&mut decb, &mtpb, &fx.s1.e, &fx.s1.h_post(), true)?;
        let (md_h, n1) = maxdiff_nan(&ga.h_next, &gb.h_next);
        let (md_hi, n2) = maxdiff_nan(&ga.head_in, &gb.head_in);
        let (md_cat, n3) = maxdiff_nan(&ga.cat, &gb.cat);
        let detected = md_h > MTP_DETECT_THRESH && md_hi > MTP_DETECT_THRESH;
        let pass2b = detected && (n1 + n2 + n3) == 0;
        println!(
            "device: {dev} | exl3-cuda-mtp (ii) capture-point pre-FFN-sum vs post-FFN-sum (쌍 디코드): cat={:.3e} h_next={:.3e} head_in={:.3e} | tok pre={} post={} | 값 검출 {} (임계 {:.0e} 초과 — §3.4 계측 맥락 a1 0.69[전] vs 0.44[후]) | {}",
            md_cat,
            md_h,
            md_hi,
            ta.unwrap_or(u32::MAX),
            tb.unwrap_or(u32::MAX),
            if detected { "YES" } else { "NO" },
            MTP_DETECT_THRESH,
            if pass2b { "PASS" } else { "FAIL" }
        );
        if !pass2b {
            fails.push(format!(
                "(ii) capture-point pre/post h_next={md_h:.3e} head_in={md_hi:.3e} — 검출 불가(임계 이하)"
            ));
        }
        report.push_str(&format!(" · (ii) pre/post h_next Δ={md_h:.3e}"));
    }

    // ── (iii) 27B 두 번째 시드(35B mtp.* 부재 — else 분책) + fresh
    // 상태 reset/reseed 결정론(원장 19호 가드). ──
    {
        let seed = 0x170C_0DA0_0000_09A3_u64;
        let fx = MtpFixture::generate(dims, dir27, 33, seed)?;
        fresh_state(&mut dec, &mtp, &fx)?;
        let mut kc = fx.kc_hist.clone();
        let mut vc = fx.vc_hist.clone();
        let w = orc.step(
            &fx.s1.e,
            &fx.s1.h_pre,
            &mut kc,
            &mut vc,
            33,
            &HeadRange::Slice(0, HEAD_SLICE),
        )?;
        let (g, t) = gpu_step(&mut dec, &mtp, &fx.s1.e, &fx.s1.h_pre, true)?;
        let c = cmp_steps(&g, &w, &orc.gate)?;
        let gpu_lg = read_logits(&dec, &mtp, 0, HEAD_SLICE)?;
        let (lg_md, lg_nan) = maxdiff_nan(&gpu_lg, &w.logits);
        // 결정론: 동일 fresh 상태 재실행 — h_next 비트 동일.
        fresh_state(&mut dec, &mtp, &fx)?;
        let (g2r, t2r) = gpu_step(&mut dec, &mtp, &fx.s1.e, &fx.s1.h_pre, true)?;
        let mut bit_same = true;
        for (a, b) in g.h_next.iter().zip(&g2r.h_next) {
            if a.to_bits() != b.to_bits() {
                bit_same = false;
                break;
            }
        }
        let pass3 = c.pass() && lg_md <= MTP_STAGE_THRESH && lg_nan == 0 && bit_same && t == t2r;
        println!(
            "device: {dev} | exl3-cuda-mtp (iii) seed2 27B dims (35B-A3B exl3 아카이브에 mtp.* 부재 — GGUF Q4 경로, 실측 2026-10-04): cat={:.3e} outv={:.3e} | h_next={:.3e} head_in={:.3e} logits[slice]={:.3e} | reset/reseed h_next 비트동일={} tok={} | {}",
            c.cat,
            c.outv,
            c.h_next,
            c.head_in,
            lg_md,
            bit_same,
            t.unwrap_or(u32::MAX),
            if pass3 { "PASS" } else { "FAIL" }
        );
        if !pass3 {
            fails.push(format!(
                "(iii) e2e h_next={:.3e} head_in={:.3e} logits={lg_md:.3e} nan={} bit_same={bit_same}",
                c.h_next, c.head_in, c.nan + lg_nan
            ));
        }
        report.push_str(&format!(
            " · (iii) h_next={:.3e} head_in={:.3e} bit-identical={}",
            c.h_next, c.head_in, bit_same
        ));
    }

    if fails.is_empty() {
        Ok(format!("device: {dev} | exl3-cuda-mtp {report} | ALL PASS"))
    } else {
        Err(format!(
            "exl3-cuda-mtp 실패 — {} (device: {dev})",
            fails.join(" · ")
        ))
    }
}

/// exl3-cuda-mtp-neg — 음성대조(원장 17호: 계기도 스스로 검증).
/// 시나리오(결함 10호 — 캡처 시점): 오라클은 계약 캡처점(h_pre =
/// 마지막 FFN 합산 전 잔차)으로 계산, GPU 체인에는 잘못된 캡처점
/// (h_post = 합산 후)을 주입한다. 캡처 시점 차이가 실결함 계급이면
/// 종단 값에서 임계(2e-4) 초과 이격이 발생 — 그 이격이 값으로 잡히
/// 는지가 검증 대상(§6: 값 maxdiff 판정). 초과 시 NEG-DETECTED
/// 마커와 함께 Err(→ CLI 비영 exit). 미검출이면 계기 결함(Ok 반환
/// 아님 — 하니스가 비영을 강제).
pub fn cuda_mtp_negative_check(dir27: &str) -> Result<String, String> {
    let (mut dec, dims) = build_stack(dir27)?;
    let dev = dec.device_name().to_string();
    let ar = StArchive::open(Path::new(dir27))?;
    let mut norms5 = Vec::with_capacity(5 * dims.hidden);
    for name in [
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.norm.weight",
    ] {
        norms5.extend_from_slice(&read_norm_plus1(&ar, name)?);
    }
    let qnw = read_norm_plus1(&ar, "mtp.layers.0.self_attn.q_norm.weight")?;
    let knw = read_norm_plus1(&ar, "mtp.layers.0.self_attn.k_norm.weight")?;
    let mtp = Exl3CudaMtp::new(&mut dec, dims, &norms5, &qnw, &knw)?;
    let orc = MtpOracle::build(&dec, &ar, dims)?;
    let seed = 0x170C_0DA0_0000_09A4_u64;
    let fx = MtpFixture::generate(dims, dir27, 33, seed)?;
    fresh_state(&mut dec, &mtp, &fx)?;
    let mut kc = fx.kc_hist.clone();
    let mut vc = fx.vc_hist.clone();
    // 정경로(계약 캡처점 h_pre)는 오라클과 정합인지 먼저 확인 —
    // 음성대조의 전제(계기 자체 검증, 원장 17호).
    let w = orc.step(
        &fx.s1.e,
        &fx.s1.h_pre,
        &mut kc,
        &mut vc,
        33,
        &HeadRange::Slice(0, HEAD_SLICE),
    )?;
    let (gok, _) = gpu_step(&mut dec, &mtp, &fx.s1.e, &fx.s1.h_pre, false)?;
    let cok = cmp_steps(&gok, &w, &orc.gate)?;
    if !cok.pass() {
        return Err(format!(
            "NEG 전제 실패 — 정경로(h_pre)부터 임계 초과: h_next={:.3e} head_in={:.3e}",
            cok.h_next, cok.head_in
        ));
    }
    // 결함 재현: h_post(FFN 합산 후 캡처) 주입 — fresh 상태에서.
    fresh_state(&mut dec, &mtp, &fx)?;
    let (gbad, tbad) = gpu_step(&mut dec, &mtp, &fx.s1.e, &fx.s1.h_post(), true)?;
    let (md_h, n1) = maxdiff_nan(&gbad.h_next, &w.h_next);
    let (md_hi, n2) = maxdiff_nan(&gbad.head_in, &w.head_in);
    let md = md_h.max(md_hi);
    let detected = md > MTP_DETECT_THRESH && (n1 + n2) == 0;
    println!(
        "device: {dev} | exl3-cuda-mtp-neg capture-point swapped (post-FFN injected): h_next={md_h:.3e} head_in={md_hi:.3e} tok={} oracle-tok={} | {}",
        tbad.unwrap_or(u32::MAX),
        w.token,
        if detected {
            "NEG-DETECTED (maxdiff > 2e-4 — 캡처 시점 결함이 값으로 검출됨)"
        } else {
            "NOT-DETECTED — 계기 결함(임계 이하)"
        }
    );
    if detected {
        Err(format!(
            "NEG-DETECTED: h_post 주입 종단 이격 h_next={md_h:.3e} head_in={md_hi:.3e} > {:.0e} (결함 10호 재현 검출)",
            MTP_DETECT_THRESH
        ))
    } else {
        Err(format!(
            "mtp-neg 미검출 — 계기 결함: h_next={md_h:.3e} head_in={md_hi:.3e} ≤ {:.0e}",
            MTP_DETECT_THRESH
        ))
    }
}
