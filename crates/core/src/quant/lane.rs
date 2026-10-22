
pub fn tree64(v: &[f64; 64]) -> f64 {
    let mut a = *v;
    for i in 0..32 {
        a[i] += a[i + 32];
    }
    for &off in &[16usize, 8, 4, 2, 1] {
        for i in 0..off {
            a[i] += a[i + off];
        }
    }
    a[0]
}

#[inline]
fn f16b(v: u16) -> f32 {
    super::deq::half_to_f32(v)
}

pub fn dot_row_w4a16_lane(qrow: &[u32], zrow: &[u32], srow: &[u16], x: &[u16]) -> f32 {
    tree64(&dot_row_w4a16_lane_parts(qrow, zrow, srow, x)) as f32
}

pub fn dot_row_w4a16_lane_parts(qrow: &[u32], zrow: &[u32], srow: &[u16], x: &[u16]) -> [f64; 64] {
    let k = x.len();
    assert!(k.is_multiple_of(128), "w4a16: k는 g128 그룹정렬 필요");
    assert_eq!(qrow.len(), k / 8, "w4a16: qrow 길이");
    assert_eq!(srow.len(), k / 128, "w4a16: srow 길이");
    lane_parts_scale(qrow, zrow, x, 128, |g| f16b(srow[g]))
}

pub fn dot_row_w4a16_lane_group(
    qrow: &[u32],
    zrow: &[u32],
    srow_f32: &[f32],
    x: &[u16],
    group: usize,
) -> f32 {
    tree64(&lane_parts_scale(qrow, zrow, x, group, |g| srow_f32[g])) as f32
}

fn lane_parts_scale<F: Fn(usize) -> f32>(
    qrow: &[u32],
    zrow: &[u32],
    x: &[u16],
    group: usize,
    scale: F,
) -> [f64; 64] {
    let k = x.len();
    assert!(
        group >= 32 && group.is_multiple_of(32) && k.is_multiple_of(group),
        "w4a16: k는 group(32배수) 정렬 필요 — group={group} k={k}"
    );
    assert_eq!(qrow.len(), k / 8, "w4a16: qrow 길이");
    assert_eq!(zrow.len(), k / group, "w4a16: zrow 길이");
    let mut lane = [0.0f64; 64];
    for l in 0..64usize {
        let mut acc = 0.0f32;
        let mut i = l;
        while i < k {
            let g = i / group;
            let q = ((qrow[i >> 3] >> (4 * (i & 7))) & 0xF) as i32;
            let w = (q - zrow[g] as i32) as f32 * scale(g);
            acc += w * f16b(x[i]);
            i += 64;
        }
        lane[l] = acc as f64;
    }
    lane
}

#[cfg(test)]
mod w4a16_tests {
    use super::*;

    fn to_f16(v: f32) -> u16 {
        let b = v.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xFF) as i32 - 127 + 15;
        let man = b & 0x7F_FFFF;
        if exp <= 0 {
            return sign;
        }
        if exp >= 31 {
            return sign | 0x7C00;
        }
        let half_man = man >> 13;
        let round = (man >> 12) & 1;
        let m = half_man + round;
        let (m, e) = if m & 0x400 != 0 {
            (m & 0x3FF, exp + 1)
        } else {
            (m, exp)
        };
        sign | ((e as u16) << 10) | (m as u16)
    }

    #[test]
    fn w4a16_lane_matches_scalar() {
        let k = 256usize;
        let mut seed = 0x9e37_79b9u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut qrow = vec![0u32; k / 8];
        for v in qrow.iter_mut() {
            *v = rnd() as u32;
        }
        let mut zrow = vec![0u32; k / 128];
        for v in zrow.iter_mut() {
            *v = (rnd() % 16) as u32;
        }
        let mut srow = vec![0u16; k / 128];
        for v in srow.iter_mut() {
            *v = to_f16(0.25f32 + (rnd() % 100) as f32 / 400.0);
        }
        let mut x = vec![0u16; k];
        for v in x.iter_mut() {
            *v = to_f16((rnd() % 2000) as f32 / 1000.0 - 1.0);
        }

        let got = dot_row_w4a16_lane(&qrow, &zrow, &srow, &x);
        let mut want = 0.0f64;
        for i in 0..k {
            let g = i / 128;
            let q = ((qrow[i / 8] >> (4 * (i % 8))) & 0xF) as i32;
            let w = (q - zrow[g] as i32) as f32 * f16b(srow[g]);
            want += (w * f16b(x[i])) as f64;
        }
        let rel = (got as f64 - want).abs() / want.abs().max(1e-9);
        assert!(
            rel < 1e-6,
            "w4a16 미러 불일치: got={got} want={want} rel={rel}"
        );
    }

    #[test]
    fn f16b_is_ieee_half() {
        assert_eq!(f16b(0x3C00), 1.0);
        assert_eq!(f16b(0x3800), 0.5);
        assert_eq!(f16b(0xC000), -2.0);
    }

    #[test]
    fn w4a16_lane_g32_bf16_scale() {
        let k = 256usize;
        let mut seed = 0x1234_5678u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let qrow: Vec<u32> = (0..k / 8).map(|_| rnd() as u32).collect();
        let zrow: Vec<u32> = (0..k / 32).map(|_| (rnd() % 16) as u32).collect();
        let srow: Vec<f32> = (0..k / 32)
            .map(|_| {
                let v = 0.25f32 + (rnd() % 100) as f32 / 400.0;
                crate::quant::deq::bf16_to_f32((v.to_bits() >> 16) as u16)
            })
            .collect();
        let x: Vec<u16> = (0..k)
            .map(|_| to_f16((rnd() % 2000) as f32 / 1000.0 - 1.0))
            .collect();
        let got = dot_row_w4a16_lane_group(&qrow, &zrow, &srow, &x, 32);
        let mut lanes = [0.0f64; 64];
        for l in 0..64usize {
            let mut acc = 0.0f32;
            let mut i = l;
            while i < k {
                let g = i / 32;
                let q = ((qrow[i >> 3] >> (4 * (i & 7))) & 0xF) as i32;
                let w = (q - zrow[g] as i32) as f32 * srow[g];
                acc += w * f16b(x[i]);
                i += 64;
            }
            lanes[l] = acc as f64;
        }
        assert_eq!(got.to_bits(), (tree64(&lanes) as f32).to_bits());
    }

    #[test]
    fn w4a16_sym_zp8_center() {
        let k = 128usize;
        let qrow = vec![0x8888_8888u32; k / 8];
        let zrow = vec![8u32; 1];
        let srow = vec![to_f16(0.5); 1];
        let x = vec![to_f16(1.0); k];
        assert_eq!(dot_row_w4a16_lane(&qrow, &zrow, &srow, &x), 0.0);
    }
}
