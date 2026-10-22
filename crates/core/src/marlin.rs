
pub fn marlin_repack_q(q: &[u32], n: usize, k: usize) -> Result<Vec<u32>, String> {
    if !k.is_multiple_of(16) || !n.is_multiple_of(64) {
        return Err(format!("marlin repack: k={k}%16·n={n}%64 계약 위반"));
    }
    let k8 = k / 8;
    if q.len() != n * k8 {
        return Err(format!(
            "marlin repack: q len {} != n*k/8 {}",
            q.len(),
            n * k8
        ));
    }
    let (kt_n, nt_n) = (k / 16, n / 64);
    let mut out = vec![0u32; kt_n * nt_n * 128];
    let nib = |kk: usize, nn: usize| -> u32 { (q[nn * k8 + kk / 8] >> ((kk % 8) * 4)) & 0xF };
    const OFF: [usize; 4] = [0, 1, 8, 9];
    const PACK: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
    for kt in 0..kt_n {
        for nt in 0..nt_n {
            let tile = &mut out[(kt * nt_n + nt) * 128..(kt * nt_n + nt) * 128 + 128];
            for tid in 0..32 {
                let tc_col = tid / 4;
                let tc_row = (tid % 4) * 2;
                for warp in 0..4 {
                    let cur_n = nt * 64 + warp * 16 + tc_col;
                    let mut vals = [0u32; 8];
                    for i in 0..4 {
                        let kk = kt * 16 + tc_row + OFF[i];
                        vals[i] = nib(kk, cur_n);
                        vals[4 + i] = nib(kk, cur_n + 8);
                    }
                    let mut res = 0u32;
                    for i in 0..8 {
                        res |= vals[PACK[i]] << (i * 4);
                    }
                    tile[tid * 4 + warp] = res;
                }
            }
        }
    }
    Ok(out)
}

pub fn marlin_permute_scales(s: &[f32], n: usize, k: usize, gs: usize) -> Result<Vec<f32>, String> {
    if !n.is_multiple_of(64) || !k.is_multiple_of(gs) {
        return Err(format!("marlin scales: n={n}%64·k={k}%gs={gs} 계약 위반"));
    }
    let kg = k / gs;
    if s.len() != n * kg {
        return Err(format!(
            "marlin scales: len {} != n*k/gs {}",
            s.len(),
            n * kg
        ));
    }
    let mut out = vec![0f32; kg * n];
    for g in 0..kg {
        for chunk in 0..(n / 64) {
            for p in 0..64 {
                let src = p / 8 + 8 * (p % 8);
                out[g * n + chunk * 64 + p] = s[(chunk * 64 + src) * kg + g];
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unrepack(out: &[u32], n: usize, k: usize) -> Vec<u32> {
        let k8 = k / 8;
        let mut q = vec![0u32; n * k8];
        let mut put = |kk: usize, nn: usize, v: u32| {
            q[nn * k8 + kk / 8] |= v << ((kk % 8) * 4);
        };
        const OFF: [usize; 4] = [0, 1, 8, 9];
        const PACK: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
        for kt in 0..(k / 16) {
            for nt in 0..(n / 64) {
                let tile = &out[(kt * (n / 64) + nt) * 128..(kt * (n / 64) + nt) * 128 + 128];
                for tid in 0..32 {
                    let tc_col = tid / 4;
                    let tc_row = (tid % 4) * 2;
                    for warp in 0..4 {
                        let cur_n = nt * 64 + warp * 16 + tc_col;
                        let res = tile[tid * 4 + warp];
                        let mut vals = [0u32; 8];
                        for i in 0..8 {
                            vals[PACK[i]] = (res >> (i * 4)) & 0xF;
                        }
                        for i in 0..4 {
                            let kk = kt * 16 + tc_row + OFF[i];
                            put(kk, cur_n, vals[i]);
                            put(kk, cur_n + 8, vals[4 + i]);
                        }
                    }
                }
            }
        }
        q
    }

    #[test]
    fn repack_roundtrip() {
        let (n, k) = (128usize, 64usize);
        let mut seed: u64 = 0x1234_5678_9abc_def0;
        let mut q = vec![0u32; n * (k / 8)];
        for w in q.iter_mut() {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *w = (seed >> 32) as u32;
        }
        let packed = marlin_repack_q(&q, n, k).unwrap();
        assert_eq!(packed.len(), (k / 16) * (n / 64) * 128);
        let back = unrepack(&packed, n, k);
        assert_eq!(back, q, "라운드트립 불일치");
    }

    #[test]
    fn scale_perm_roundtrip() {
        let (n, k, gs) = (64usize, 128usize, 128usize);
        let s: Vec<f32> = (0..n * (k / gs)).map(|i| i as f32).collect();
        let out = marlin_permute_scales(&s, n, k, gs).unwrap();
        for p in 0..64 {
            let src = p / 8 + 8 * (p % 8);
            assert_eq!(out[p], s[src]);
        }
    }
}
