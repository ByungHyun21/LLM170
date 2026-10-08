//! 디양자화 — 플레인(F32·F16·Bf16) 행 전개. W4A16 split은 비경유(matmul arm).

use crate::wtype::WType;

/// IEEE 754 binary16 → f32
pub fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match exp {
        0 => {
            if frac == 0 {
                sign << 31
            } else {
                // subnormal
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

pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// 한 행(k 원소)을 f32 로 펼친다. `data` 는 해당 텐서의 데이터 시작 바이트.
pub fn dequant_row(ty: WType, data: &[u8], row: u64, k: u64, out: &mut [f32]) {
    let (blck, bsize) = ty.block_info();
    let blocks = (k / blck) as usize;
    let bsize = bsize as usize;
    debug_assert_eq!(out.len(), k as usize);
    let base = row as usize * blocks * bsize;
    match ty {
        WType::F32 => {
            for j in 0..k as usize {
                let o = base + j * 4;
                out[j] = f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
            }
        }
        WType::F16 | WType::Bf16 => {
            for j in 0..k as usize {
                let h = u16::from_le_bytes([data[base + j * 2], data[base + j * 2 + 1]]);
                out[j] = if ty == WType::F16 {
                    half_to_f32(h)
                } else {
                    bf16_to_f32(h)
                };
            }
        }
        // W4A16 split은 분리 버퍼(packed+scale) — dequant_row 비경유 계약.
        // 여기 도달하면 로더/이름맵 결함이다(조용한 오염 방지 — 즉시 패닉).
        other => unimplemented!("dequant_row: 지원 밖 타입 {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plains_dequant_row() {
        // F32
        let mut data = Vec::new();
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&(-2.5f32).to_le_bytes());
        let mut out = [0.0f32; 2];
        dequant_row(WType::F32, &data, 0, 2, &mut out);
        assert_eq!(out, [1.0, -2.5]);

        // F16(1.0=0x3C00)·Bf16(1.0=0x3F80)
        let mut out1 = [0.0f32; 1];
        dequant_row(WType::F16, &[0x00, 0x3C], 0, 1, &mut out1);
        assert_eq!(out1[0], 1.0);
        dequant_row(WType::Bf16, &[0x80, 0x3F], 0, 1, &mut out1);
        assert_eq!(out1[0], 1.0);
    }

    #[test]
    #[should_panic(expected = "지원 밖 타입")]
    fn split_type_is_not_dequant_row() {
        let mut out = [0.0f32; 128];
        dequant_row(WType::W4a16G128Split, &[0u8; 66], 0, 128, &mut out);
    }
}
