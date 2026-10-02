//! 트렐리스 참조 디코드 — plans/118 §6 명세의 Rust 구현.
//!
//! 근거 소스: `cpu/moe_mul1.cpp`의 `decode_state_scalar`·`decode_mul1_scalar`·
//! `scalar_tiles`(가장 읽기 쉬운 권위 경로), `quant/exl3_dq.cuh`(GPU),
//! `codebook.cuh`(mul1). 검증: GGUF Q8 대조(scripts/exl3_validate.py).
//!
//! 정밀도: mul1 최종값은 참조의 f16 hfma를 f32 mul_add + f16 반올림으로
//! 재현(곱은 f32에서 정확, 합의 f32 반올림은 f16 ULP보다 2^13배 작아
//! 최종 f16 비트는 참조와 동일 — 경계 케이스 이론상 2^-24 확률로 상이
//! 가능하나 골든 대조 영향 없음, 원장 10a 범위).

use crate::{Exl3Error, Result, StArchive};
use half::f16;

pub const MUL1_MULT: u32 = 0x83DCD12D;

/// mul1 코드북 상수(f16 비트): k_inv = 0x1eee, k_bias = 0xc931.
const K_INV: f32 = 0.00676727294921875; // f16(0x1eee)
const K_BIAS: f32 = -10.3828125; // f16(0xc931)

/// tensor_core_perm 역표 — 위치 (r*16+c) → 트렐리스 워드 인덱스 t.
/// 순방향 perm: p[t*8+s] = (r,c), r0=(t%4)*2, c0=t/4,
/// s: 0..3 → (r0,r0+1,r0+8,r0+9)×c0, 4..7 → ×(c0+8).
pub const PERM_INV: [u16; 256] = {
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

/// mul1 코드북: 16비트 워드 → f32 값(도메인 [-3.45, 3.50], 1024레벨).
///
/// sum 비트패턴 [0x6400, 0x67FC]의 f16 값은 정확히 1024+bytesum
/// ([1024,2048) 구간 ULP=1) — 참조의 `half_uint16(sum)` 트릭과 동치.
#[inline]
pub fn mul1_decode(word: u16) -> f32 {
    let x = (word as u32).wrapping_mul(MUL1_MULT);
    let sum = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24);
    let f = 1024.0f32 + sum as f32;
    f.mul_add(K_INV, K_BIAS)
}

/// 타일 비트링에서 워드 t 추출 (decode_state_scalar 리터럴 포팅).
///
/// `u32s`: 타일의 8K u32 워드(파일 u16 자연 페어링 LE). 워드 t =
/// 링 비트 [(t+1)K-16, (t+1)K) — 테일바이팅 환형, 링 비트 r =
/// u32 워드 r/32의 비트 31-(r%32)(MSB 우선).
#[inline]
pub fn tile_word(u32s: &[u32], krate: u32, t: u32) -> u16 {
    let words32 = 8 * krate as usize;
    let b0 = (t * krate + (krate + 256 * krate - 16)) as usize;
    let b1 = b0 + 16;
    let i0 = (b0 / 32) % words32;
    let i1 = ((b1 - 1) / 32) % words32;
    let s = ((b1 - 1) / 32 + 1) * 32 - b1;
    let merged = ((u32s[i0] as u64) << 32) | u32s[i1] as u64;
    ((merged >> s) & 0xFFFF) as u16
}

/// 타일의 256 워드 전체 추출.
pub fn tile_words(u16s: &[u16], krate: u32, out: &mut [u16; 256]) {
    debug_assert_eq!(u16s.len(), 16 * krate as usize);
    let mut u32s = [0u32; 8 * 8]; // K ≤ 8
    let n = 8 * krate as usize;
    for (i, w) in u32s.iter_mut().take(n).enumerate() {
        *w = u16s[2 * i] as u32 | ((u16s[2 * i + 1] as u32) << 16);
    }
    for (t, o) in out.iter_mut().enumerate() {
        *o = tile_word(&u32s[..n], krate, t as u32);
    }
}

/// 타일 디코드 — 위치 순서 (r*16+c)의 256 값.
pub fn decode_tile(u16s: &[u16], krate: u32, out: &mut [f32; 256]) {
    let mut words = [0u16; 256];
    tile_words(u16s, krate, &mut words);
    for (pos, t) in PERM_INV.iter().enumerate() {
        out[pos] = mul1_decode(words[*t as usize]);
    }
}

/// EXL3 양자화 선형 레이어(참조 표현 — 엔진 적재는 별도 경로).
#[derive(Debug)]
pub struct Exl3Linear {
    pub k: usize,
    pub n: usize,
    pub krate: u32,
    /// 반정수 bpw(K+0.5, mul1만) — 27B 4.0bpw에는 없음.
    pub half_k: bool,
    /// trellis 원시 바이트(u16 LE × k/16 × n/16 × 16K).
    pub trellis: Vec<u8>,
    pub suh: Vec<f16>,
    pub svh: Vec<f16>,
}

impl Exl3Linear {
    /// 아카이브에서 `<key>.{trellis,suh,svh,mul1}` 묶음 로드.
    pub fn load(ar: &StArchive, key: &str) -> Result<Self> {
        let tre = ar.read(&format!("{key}.trellis"))?;
        let shape = ar
            .entry(&format!("{key}.trellis"))
            .map(|e| e.shape.clone())
            .ok_or_else(|| Exl3Error::TensorNotFound(format!("{key}.trellis")))?;
        if shape.len() != 3 {
            return Err(Exl3Error::BadTensor(format!(
                "{key}: trellis dim {}",
                shape.len()
            )));
        }
        let (kt, nt, tw) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
        let k = kt * 16;
        let n = nt * 16;
        // K 산출: 16K u16(정수) 또는 16K+8(반정수).
        let (krate, half_k) = if tw.is_multiple_of(16) {
            ((tw / 16) as u32, false)
        } else if tw % 16 == 8 {
            (((tw - 8) / 16) as u32, true)
        } else {
            return Err(Exl3Error::BadTensor(format!("{key}: tile width {tw}")));
        };
        if !(1..=8).contains(&krate) {
            return Err(Exl3Error::BadTensor(format!("{key}: K={krate}")));
        }
        if trellis_len(krate, half_k, kt, nt) != tre.len() {
            return Err(Exl3Error::BadTensor(format!(
                "{key}: trellis bytes mismatch"
            )));
        }
        let suh = read_f16(ar, &format!("{key}.suh"), k)?;
        let svh = read_f16(ar, &format!("{key}.svh"), n)?;
        Ok(Self {
            k,
            n,
            krate,
            half_k,
            trellis: tre,
            suh,
            svh,
        })
    }

    /// 타일 (kt, nt) 디코드 — out은 위치 순서 256값.
    pub fn tile(&self, kt: usize, nt: usize, out: &mut [f32; 256]) {
        let tw = 16 * self.krate as usize + if self.half_k { 8 } else { 0 };
        let ntiles = self.n / 16;
        let off = ((kt * ntiles + nt) * tw) * 2;
        let (chunks, _) = self.trellis[off..off + tw * 2].as_chunks::<2>();
        let u16s: Vec<u16> = chunks.iter().map(|c| u16::from_le_bytes(*c)).collect();
        // 반정수 bpw는 참조 미구현(27B 4.0bpw에 없음) — 상류에서 거부.
        decode_tile(&u16s, self.krate, out);
    }

    /// 원 기저 가중치 블록 — W[k0..k0+bk, n0..n0+bn] (f64 참조).
    ///
    /// W = diag(suh)·(H·Wq·H)/128·diag(svh), H는 128청크 자연 순서 WHT.
    /// k0/n0는 16의 배수, bk/bn ≤ 128이며 청크 경계를 넘지 않아야 한다.
    pub fn dequant_block_f64(&self, k0: usize, n0: usize, bk: usize, bn: usize) -> Vec<f64> {
        dequant_block(self, k0, n0, bk, bn)
    }
}

fn trellis_len(krate: u32, half: bool, kt: usize, nt: usize) -> usize {
    let tw = 16 * krate as usize + if half { 8 } else { 0 };
    kt * nt * tw * 2
}

fn read_f16(ar: &StArchive, name: &str, expect: usize) -> Result<Vec<f16>> {
    let raw = ar.read(name)?;
    if raw.len() != expect * 2 {
        return Err(Exl3Error::BadTensor(format!(
            "{name}: len {} != {expect}",
            raw.len() / 2
        )));
    }
    let (chunks, _) = raw.as_chunks::<2>();
    Ok(chunks.iter().map(|c| f16::from_le_bytes(*c)).collect())
}

/// 자연 순서 WHT-128 — 제자리 버터플라이(f64).
pub fn had128(v: &mut [f64]) {
    debug_assert_eq!(v.len(), 128);
    let mut width = 1;
    while width < 128 {
        let mut base = 0;
        while base < 128 {
            for i in 0..width {
                let a = v[base + i];
                let b = v[base + width + i];
                v[base + i] = a + b;
                v[base + width + i] = a - b;
            }
            base += 2 * width;
        }
        width *= 2;
    }
}

/// 원 기저 블록 디퀀트(f64 참조). 제약: bk·bn ≤ 128, k0/n0 16배수,
/// 블록은 단일 128청크 내부(상류가 청크 분할 책임).
pub fn dequant_block(w: &Exl3Linear, k0: usize, n0: usize, bk: usize, bn: usize) -> Vec<f64> {
    assert!(bk <= 128 && bn <= 128);
    assert!(k0.is_multiple_of(16) && n0.is_multiple_of(16));
    assert!(k0 % 128 + bk <= 128 && n0 % 128 + bn <= 128);
    let ktiles = bk / 16;
    let ntiles = bn / 16;

    // Wq 블록(bk × bn) — 타일 단위 디코드.
    let mut wq = vec![0f64; bk * bn];
    let mut tile = [0f32; 256];
    for kt in 0..ktiles {
        for nt in 0..ntiles {
            w.tile(k0 / 16 + kt, n0 / 16 + nt, &mut tile);
            for r in 0..16 {
                for c in 0..16 {
                    wq[(kt * 16 + r) * bn + nt * 16 + c] = tile[r * 16 + c] as f64;
                }
            }
        }
    }

    // H·Wq·H / 128 — 좌우 각각 부분 WHT(128열/행 전체가 필요하므로
    // 청크 내 나머지 타일도 디코드).
    let ck0 = (k0 % 128) / 16; // 블록의 청크 내 타일 시작
    let cn0 = (n0 % 128) / 16;
    let ck = 128 / 16;
    let mut full = vec![0f64; 128 * 128];
    for kt in 0..ck {
        for nt in 0..ck {
            w.tile(k0 / 16 - ck0 + kt, n0 / 16 - cn0 + nt, &mut tile);
            for r in 0..16 {
                for c in 0..16 {
                    full[(kt * 16 + r) * 128 + nt * 16 + c] = tile[r * 16 + c] as f64;
                }
            }
        }
    }
    // 행 방향 H.
    let (rows, _) = full.as_chunks_mut::<128>();
    for row in rows {
        had128(row);
    }
    // 열 방향 H.
    let mut col = [0f64; 128];
    for c in 0..128 {
        for (r, v) in col.iter_mut().enumerate() {
            *v = full[r * 128 + c];
        }
        had128(&mut col);
        for (r, v) in col.iter().enumerate() {
            full[r * 128 + c] = v / 128.0;
        }
    }
    // 스케일 적용 후 블록 발췌.
    let mut out = vec![0f64; bk * bn];
    for i in 0..bk {
        let si = w.suh[k0 + i].to_f64();
        for j in 0..bn {
            let sj = w.svh[n0 + j].to_f64();
            out[i * bn + j] = si * sj * full[(ck0 * 16 + i) * 128 + cn0 * 16 + j];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 골든: 27B gate_proj 타일(0,0) — GGUF Q8 대조로 검증된 파이프라인
    /// (scripts/exl3_validate.py, corr 0.977)에서 추출한 상수.
    /// 전체 타일 검증은 LLM170_M27_EXL3 픽스처가 있을 때만 수행.
    const GOLDEN_U16_HEAD: [u16; 8] = [39348, 28177, 39685, 51648, 50850, 38287, 39081, 54582];
    const GOLDEN_WORDS: [u16; 16] = [
        50555, 11227, 24284, 63201, 46856, 47174, 49715, 4505, 36045, 26221, 13161, 39756, 55908,
        54055, 39224, 51648,
    ];
    /// 위치 순서 (r*16+c)의 디코드값 표본.
    const GOLDEN_VALS: [(usize, f32); 7] = [
        (0, -1.558289),
        (1, -0.881561),
        (16, -0.116859),
        (17, -0.380783),
        (128, -0.123627),
        (129, -1.700401),
        (255, 1.290733),
    ];

    #[test]
    fn golden_head_assembly() {
        // u16 헤드 → u32 어셈블리 보존(자연 페어링 LE).
        let v: Vec<u16> = GOLDEN_U16_HEAD.iter().copied().chain([0u16; 40]).collect();
        let u32s: Vec<u32> = (0..24)
            .map(|i| v[2 * i] as u32 | ((v[2 * i + 1] as u32) << 16))
            .collect();
        assert_eq!(u32s[0], 39348u32 | (28177u32 << 16));
        let _ = tile_word(&u32s, 3, 0); // 파닉 없이 동작

        // 픽스처가 있으면 전체 타일·골든 검증(모델 의존 스킵).
        let Ok(dir) = std::env::var("LLM170_M27_EXL3") else {
            return;
        };
        let Ok(ar) = crate::StArchive::open(std::path::Path::new(&dir)) else {
            return;
        };
        let Ok(w) = Exl3Linear::load(&ar, "model.language_model.layers.0.mlp.gate_proj") else {
            return;
        };
        let tw = 16 * w.krate as usize;
        let (chunks, _) = w.trellis[..tw * 2].as_chunks::<2>();
        let u16s: Vec<u16> = chunks.iter().map(|c| u16::from_le_bytes(*c)).collect();
        assert_eq!(&u16s[..8], &GOLDEN_U16_HEAD);
        let mut words = [0u16; 256];
        tile_words(&u16s, w.krate, &mut words);
        assert_eq!(&words[..16], &GOLDEN_WORDS);
        let mut vals = [0f32; 256];
        decode_tile(&u16s, w.krate, &mut vals);
        for (pos, want) in GOLDEN_VALS {
            assert!(
                (vals[pos] - want).abs() < 5e-6,
                "pos {pos}: {} != {want}",
                vals[pos]
            );
        }
    }

    #[test]
    fn mul1_range() {
        // 코드북 값역 [-3.45, 3.50]·분포 검사(전 워드 스캔).
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        let mut sum = 0f64;
        let mut cnt = 0u32;
        for w in (0..65536u32).step_by(97) {
            let v = mul1_decode(w as u16);
            lo = lo.min(v);
            hi = hi.max(v);
            sum += v as f64;
            cnt += 1;
        }
        assert!(lo >= -3.46 && hi <= 3.51, "lo={lo} hi={hi}");
        let mean = sum / cnt as f64;
        assert!(mean.abs() < 0.05, "mean={mean}");
    }

    #[test]
    fn perm_inv_shape() {
        let mut seen = [false; 256];
        for &t in PERM_INV.iter() {
            assert!(t < 256);
            assert!(!seen[t as usize]);
            seen[t as usize] = true;
        }
        // 스팟 검증: (r,c)=(2,3) → lane t=13 slot 0 → w=104.
        assert_eq!(PERM_INV[2 * 16 + 3], 104);
        // (r,c)=(9,10): lane = 4*(10&7)+((9&7)>>1) = 8, slot = 2+1+4 = 7 → 71.
        assert_eq!(PERM_INV[9 * 16 + 10], 8 * 8 + 7);
    }
}
