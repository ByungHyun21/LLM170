use super::*;

impl W4a16Dec {
    pub fn bench_gemv_t(&mut self, name: &str, t: usize, reps: usize) -> Result<f64, String> {
        let (_, _, n, k) = self.lin_spec(name)?;
        let x = self.cc.alloc(t * k * 4)?;
        let y = self.cc.alloc(t * n * 4)?;
        self.cc.sync()?;
        let r = (|| -> Result<f64, String> {
            self.gemv_t_launch(name, x, y, t)?;
            self.cc.sync()?;
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                self.gemv_t_launch(name, x, y, t)?;
            }
            self.cc.sync()?;
            Ok(t0.elapsed().as_secs_f64() * 1e3 / reps as f64)
        })();
        let _ = self.cc.free(x);
        let _ = self.cc.free(y);
        r
    }

    pub fn mma_smoke(&mut self) -> Result<String, String> {
        let _g = self.cc.guard()?;
        let a = crate::rawcuda::assets::asset("smoke");
        self.cc
            .load_fatbin(a.name, &crate::rawcuda::assets::asset_bytes(a)?, a.syms)?;
        let mk = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 53) as f32 / 1024.0 - 1.0
                })
                .collect()
        };
        let (a, b) = (mk(16 * 16), mk(16 * 8));
        let bf = |x: f32| -> f32 {
            let u = x.to_bits();
            f32::from_bits((u.wrapping_add(0x7FFF + ((u >> 16) & 1))) & 0xFFFF_0000)
        };
        let ra: Vec<f32> = a.iter().map(|&x| bf(x)).collect();
        let rb: Vec<f32> = b.iter().map(|&x| bf(x)).collect();
        let mut cref = vec![0f32; 16 * 8];
        for m in 0..16 {
            for n2 in 0..8 {
                let mut acc = 0f32;
                for k in 0..16 {
                    acc += ra[m * 16 + k] * rb[k * 8 + n2];
                }
                cref[m * 8 + n2] = acc;
            }
        }
        let da = self.cc.alloc(a.len() * 4)?;
        let db = self.cc.alloc(b.len() * 4)?;
        let dc = self.cc.alloc(cref.len() * 4)?;
        let r = (|| -> Result<Vec<f32>, String> {
            let ab = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len() * 4) };
            let bb = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u8, b.len() * 4) };
            self.cc.h2d(da, ab)?;
            self.cc.h2d(db, bb)?;
            let f = self.cc.function("llm170_mma_smoke")?;
            let (mut pa, mut pb, mut pc) = (da, db, dc);
            self.cc.launch(
                f,
                1,
                1,
                32,
                &mut crate::rawcuda::args::l3(&mut pa, &mut pb, &mut pc),
            )?;
            self.cc.sync()?;
            let mut ob = vec![0u8; cref.len() * 4];
            self.cc.d2h(&mut ob, dc)?;
            Ok(ob
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect())
        })();
        let _ = self.cc.free(da);
        let _ = self.cc.free(db);
        let _ = self.cc.free(dc);
        let out = r?;
        let mut maxd = 0f32;
        for i in 0..cref.len() {
            maxd = maxd.max((out[i] - cref[i]).abs());
        }
        if !maxd.is_finite() || maxd > 1e-4 {
            return Err(format!(
                "mma_smoke: 최대 오차 {maxd:.3e} > 1e-4 — 프래그먼트/누적 불일치"
            ));
        }
        Ok(format!(
            "mma_smoke OK — bf16 m16n8k16 f32누적(16×16 × 16×8) 최대오차 {maxd:.2e}"
        ))
    }

    pub fn plain_gemm_selfcheck(&mut self) -> Result<String, String> {
        let _g = self.cc.guard()?;
        let (name, n, k) = self
            .plains
            .iter()
            .find(|(nm, (_, n, _))| nm.contains("attn_qkv") && *n <= 8192)
            .map(|(nm, &(_, n, k))| (nm.clone(), n, k))
            .ok_or("plain_gemm_selfcheck: 플레인 qkv 가중 없음")?;
        let t = 8usize;
        let x: Vec<f32> = (0..t * k)
            .map(|i| ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as f32 / 2048.0 - 0.5)
            .collect();
        let xb = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) };
        let dx = self.cc.alloc(t * k * 4)?;
        let da = self.cc.alloc(t * n * 4)?;
        let db = self.cc.alloc(t * n * 4)?;
        let r = (|| -> Result<(Vec<f32>, Vec<f32>), String> {
            self.cc.h2d(dx, xb)?;
            self.plain_gemm_launch(&name, dx, da, t)?;
            let mut bl = Vec::with_capacity(t * n);
            for ti in 0..t {
                self.plain_gemv_launch(&name, dx + (ti * k * 4) as u64, db)?;
                self.cc.sync()?;
                let mut vb = vec![0u8; n * 4];
                self.cc.d2h(&mut vb, db)?;
                bl.extend(vb.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)));
            }
            self.cc.sync()?;
            let mut ob = vec![0u8; t * n * 4];
            self.cc.d2h(&mut ob, da)?;
            let a: Vec<f32> = ob
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            Ok((a, bl))
        })();
        let _ = self.cc.free(dx);
        let _ = self.cc.free(da);
        let _ = self.cc.free(db);
        let (a, b) = r?;
        let mism = a
            .iter()
            .zip(b.iter())
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        let nan_a = a.iter().filter(|v| v.is_nan()).count();
        let nan_b = b.iter().filter(|v| v.is_nan()).count();
        Ok(format!(
            "plain-gemm-selfcheck {name}: n={n} k={k} t={t} — 불일치 {mism}/{} nan A={nan_a} B={nan_b}",
            a.len()
        ))
    }

    pub fn moe_selfcheck(&mut self) -> Result<String, String> {
        if !self.moe.moe_resident || self.moe.n_experts == 0 {
            return Err("moe_selfcheck: 상주 MoE 구성 필요".into());
        }
        let _g = self.cc.guard()?;
        self.ensure_moe_bufs()?;
        let k = self.hidden;
        let n = self.moe.moe_ffn;
        if self.moe.moe_tab.is_empty() {
            return Err("moe_selfcheck: 테이블 부재".into());
        }
        let x: Vec<f32> = (0..k)
            .map(|i| ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as f32 / 2048.0 - 0.5)
            .collect();
        let xb = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, k * 4) };
        let dx = self.cc.alloc(k * 4)?;
        let da = self.cc.alloc(n * 4)?;
        let db = self.cc.alloc(n * 4)?;
        let r = (|| -> Result<(Vec<f32>, Vec<f32>), String> {
            self.cc.h2d(dx, xb)?;
            let e0 = self.moe.moe_tab[0];
            self.gemv_launch_raw(e0.0, e0.2, n, k, dx, da)?;
            let idx = [0u32];
            let ib = unsafe { std::slice::from_raw_parts(idx.as_ptr() as *const u8, 4) };
            self.cc.h2d(self.moe.moe_idx, ib)?;
            self.gemv_experts_launch(0, 1, dx, 0, 1, db, n, k)?;
            self.cc.sync()?;
            let mut a = vec![0u8; n * 4];
            let mut b = vec![0u8; n * 4];
            self.cc.d2h(&mut a, da)?;
            self.cc.d2h(&mut b, db)?;
            let fa: Vec<f32> = a
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            let fb: Vec<f32> = b
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            Ok((fa, fb))
        })();
        let _ = self.cc.free(dx);
        let _ = self.cc.free(da);
        let _ = self.cc.free(db);
        let (fa, fb) = r?;
        let mism = fa
            .iter()
            .zip(fb.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let maxd = fa
            .iter()
            .zip(fb.iter())
            .map(|(a, b)| (a - b).abs() as f64)
            .fold(0.0f64, f64::max);
        Ok(format!(
            "moe-selfcheck: 직접 vs 배치 n={n} k={k} — 불일치 {mism}/{n} maxdiff={maxd:.3e} (head A={:?} B={:?})",
            &fa[..3],
            &fb[..3]
        ))
    }

    pub fn gemv_walk_bench(&mut self, reps: usize) -> Result<(f64, u64), String> {
        let _g = self.cc.guard()?;
        let mut mk = 0usize;
        let mut mn = 0usize;
        let mut wb = 0u64;
        for &(_, _, n, k) in self.lins.values() {
            mk = mk.max(k);
            mn = mn.max(n);
            wb += (n * k / 2) as u64;
        }
        let xf: Vec<f32> = vec![1.0f32; mk];
        let xb = unsafe { std::slice::from_raw_parts(xf.as_ptr() as *const u8, xf.len() * 4) };
        let dx = self.cc.alloc(xf.len() * 4)?;
        self.cc.h2d(dx, xb)?;
        let dy = self.cc.alloc(mn * 4)?;
        let names: Vec<String> = self.lins.keys().cloned().collect();
        let walk = |me: &mut Self| -> Result<(), String> {
            for name in &names {
                me.gemv_launch(name, dx, dy)?;
            }
            Ok(())
        };
        walk(self)?;
        self.cc.sync()?;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            walk(self)?;
        }
        self.cc.sync()?;
        let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
        self.cc.free(dx)?;
        self.cc.free(dy)?;
        Ok((ms, wb))
    }

    pub fn mma_diff_check(&mut self, name: &str, t: usize) -> Result<String, String> {
        let _g = self.cc.guard()?;
        if !self.lins.contains_key(name) {
            return self.mma_diff_plain(name, t);
        }
        let (dq, ds, n, k) = self.lin_spec(name)?;
        let xf: Vec<f32> = (0..t * k)
            .map(|i| {
                let v =
                    ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as f32 / 4096.0 - 0.5;
                crate::rawcuda::gptq4::h2f(crate::f32_to_f16(v))
            })
            .collect();
        let xh: Vec<u16> = xf.iter().map(|&v| crate::f32_to_f16(v)).collect();
        let xb = unsafe { std::slice::from_raw_parts(xf.as_ptr() as *const u8, xf.len() * 4) };
        let hb = unsafe { std::slice::from_raw_parts(xh.as_ptr() as *const u8, xh.len() * 2) };
        let dx = self.cc.alloc(t * k * 4)?;
        let dx16 = self.cc.alloc(t * k * 2)?;
        self.cc.h2d(dx, xb)?;
        self.cc.h2d(dx16, hb)?;
        let da = self.cc.alloc(t * n * 4)?;
        let db = self.cc.alloc(t * n * 4)?;
        let r = (|| -> Result<(Vec<f32>, Vec<f32>), String> {
            let f = self.cc.function("w4a16_gemm_g128")?;
            let (mut p_q, mut p_s, mut p_x, mut p_y) = (dq, ds, dx, da);
            let (mut p_n, mut p_k, mut p_t) = (n as i32, k as i32, t as i32);
            self.cc.launch(
                f,
                n.div_ceil(8) as u32,
                1,
                512,
                &mut crate::rawcuda::args::l7(
                    &mut p_q, &mut p_s, &mut p_x, &mut p_y, &mut p_n, &mut p_k, &mut p_t,
                ),
            )?;
            let fm = self.cc.function("w4a16_gemm_g128_mma")?;
            let (mut m_q, mut m_s, mut m_x, mut m_y) = (dq, ds, dx16, db);
            let (mut m_n, mut m_k, mut m_t) = (n as i32, k as i32, t as i32);
            self.cc.launch(
                fm,
                t.div_ceil(GEMM_MMA_M) as u32,
                n.div_ceil(GEMM_MMA_N) as u32,
                256,
                &mut crate::rawcuda::args::l7(
                    &mut m_q, &mut m_s, &mut m_x, &mut m_y, &mut m_n, &mut m_k, &mut m_t,
                ),
            )?;
            self.cc.sync()?;
            let mut oa = vec![0u8; t * n * 4];
            let mut ob = vec![0u8; t * n * 4];
            self.cc.d2h(&mut oa, da)?;
            self.cc.d2h(&mut ob, db)?;
            let fa: Vec<f32> = oa
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            let fb: Vec<f32> = ob
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            Ok((fa, fb))
        })();
        let _ = self.cc.free(dx16);
        let _ = self.cc.free(dx);
        let _ = self.cc.free(da);
        let _ = self.cc.free(db);
        let (fa, fb) = r?;
        let mut maxd = 0f32;
        let mut maxr = 0f32;
        let mut bad = 0usize;
        for i in 0..fa.len() {
            let d = (fa[i] - fb[i]).abs();
            if d > maxd {
                maxd = d;
            }
            let rl = d / fa[i].abs().max(1e-6);
            if rl > maxr {
                maxr = rl;
            }
            if !fb[i].is_finite() {
                bad += 1;
            }
        }
        let mut firstbad: Vec<(usize, usize)> = Vec::new();
        let mut badrows = std::collections::BTreeSet::new();
        let mut badcols = std::collections::BTreeSet::new();
        for r in 0..t {
            for c in 0..n {
                let v = fb[r * n + c];
                if !v.is_finite() || (v - fa[r * n + c]).abs() > 1e-2 {
                    if firstbad.len() < 5 {
                        firstbad.push((r, c));
                    }
                    badrows.insert(r);
                    badcols.insert(c);
                }
            }
        }
        Ok(format!(
            "mma-diff {name} n={n} k={k} t={t}: maxabs={maxd:.3e} 비유한={bad}/{} 첫불량={:?} 불량행={} 불량열={} A[0..3]={:?} B[0..3]={:?}",
            fa.len(),
            firstbad,
            badrows.len(),
            badcols.len(),
            &fa[..3],
            &fb[..3]
        ))
    }

    pub fn marlin_diff_check(
        &mut self,
        name: &str,
        t: usize,
        qm: &[u32],
    ) -> Result<String, String> {
        let _g = self.cc.guard()?;
        let (dq, ds, n, k) = self.lin_spec(name)?;
        if k % 128 != 0 || n % 64 != 0 {
            return Err(format!(
                "marlin-diff {name}: k%128·n%64 계약 위반 k={k} n={n}"
            ));
        }
        let want = (k / 16) * (n / 64) * 128;
        if qm.len() != want {
            return Err(format!(
                "marlin-diff {name}: qm {} != (k/16)(n/64)128 = {want}",
                qm.len()
            ));
        }
        let x16: Vec<u16> = (0..t * k)
            .map(|i| {
                let v =
                    ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as f32 / 4096.0 - 0.5;
                crate::f32_to_f16(v)
            })
            .collect();
        let xb = unsafe { std::slice::from_raw_parts(x16.as_ptr() as *const u8, x16.len() * 2) };
        let dx = self.cc.alloc(t * k * 2)?;
        self.cc.h2d(dx, xb)?;
        let dqm = self.cc.alloc(qm.len() * 4)?;
        let qb = unsafe { std::slice::from_raw_parts(qm.as_ptr() as *const u8, qm.len() * 4) };
        self.cc.h2d(dqm, qb)?;
        let da = self.cc.alloc(t * n * 4)?;
        let db = self.cc.alloc(t * n * 4)?;
        let r = (|| -> Result<(Vec<f32>, Vec<f32>), String> {
            let f = self.cc.function("w4a16_gemm_g128_mma")?;
            let (mut p_q, mut p_s, mut p_x, mut p_y) = (dq, ds, dx, da);
            let (mut p_n, mut p_k, mut p_t) = (n as i32, k as i32, t as i32);
            self.cc.launch(
                f,
                t.div_ceil(GEMM_MMA_M) as u32,
                n.div_ceil(GEMM_MMA_N) as u32,
                256,
                &mut crate::rawcuda::args::l7(
                    &mut p_q, &mut p_s, &mut p_x, &mut p_y, &mut p_n, &mut p_k, &mut p_t,
                ),
            )?;
            let fm = self.cc.function("w4a16_gemm_g128_marlin")?;
            let (mut m_q, mut m_s, mut m_x, mut m_y) = (dqm, ds, dx, db);
            let (mut m_n, mut m_k, mut m_t) = (n as i32, k as i32, t as i32);
            self.cc.launch(
                fm,
                t.div_ceil(GEMM_MMA_M) as u32,
                n.div_ceil(GEMM_MMA_N) as u32,
                256,
                &mut crate::rawcuda::args::l7(
                    &mut m_q, &mut m_s, &mut m_x, &mut m_y, &mut m_n, &mut m_k, &mut m_t,
                ),
            )?;
            self.cc.sync()?;
            let mut oa = vec![0u8; t * n * 4];
            let mut ob = vec![0u8; t * n * 4];
            self.cc.d2h(&mut oa, da)?;
            self.cc.d2h(&mut ob, db)?;
            let fa: Vec<f32> = oa
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            let fb: Vec<f32> = ob
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            Ok((fa, fb))
        })();
        let _ = self.cc.free(dx);
        let _ = self.cc.free(dqm);
        let _ = self.cc.free(da);
        let _ = self.cc.free(db);
        let (fa, fb) = r?;
        let mut mism = 0usize;
        let mut maxd = 0f32;
        let mut maxr = 0f32;
        let mut firstbad: Vec<(usize, usize)> = Vec::new();
        for i in 0..fa.len() {
            let d = (fa[i] - fb[i]).abs();
            if fa[i].to_bits() != fb[i].to_bits() {
                mism += 1;
                if firstbad.len() < 5 {
                    firstbad.push((i / n, i % n));
                }
            }
            if d > maxd {
                maxd = d;
            }
            let rl = d / fa[i].abs().max(1e-6);
            if rl > maxr {
                maxr = rl;
            }
        }
        Ok(format!(
            "marlin-diff {name} n={n} k={k} t={t}: 비트불일치={mism}/{} maxabs={maxd:.3e} maxrel={maxr:.3e} 첫불량={:?}",
            fa.len(),
            firstbad
        ))
    }

    pub fn marlin_selfcheck(
        &mut self,
        sym: &str,
        bm: usize,
        bn: usize,
        thr: u32,
        qm: &[u32],
        s: &[u16],
        x16: &[u16],
        expect: &[f32],
        t: usize,
        n: usize,
        k: usize,
    ) -> Result<String, String> {
        let _g = self.cc.guard()?;
        if qm.len() != (k / 16) * (n / 64) * 128
            || (s.len() != n * (k / 128) && s.len() != n * (k / 32))
            || x16.len() != t * k
            || expect.len() != t * n
        {
            return Err("marlin-selfcheck: 계약 위반(qm/s/x/expect 크기)".into());
        }
        let qb = unsafe {
            std::slice::from_raw_parts(qm.as_ptr() as *const u8, std::mem::size_of_val(qm))
        };
        let sb = unsafe {
            std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s))
        };
        let xb = unsafe {
            std::slice::from_raw_parts(x16.as_ptr() as *const u8, std::mem::size_of_val(x16))
        };
        let dqm = self.cc.alloc(qm.len() * 4)?;
        let ds = self.cc.alloc(s.len() * 2)?;
        let dx = self.cc.alloc(x16.len() * 2)?;
        let dout = self.cc.alloc(t * n * 4)?;
        self.cc.h2d(dqm, qb)?;
        self.cc.h2d(ds, sb)?;
        self.cc.h2d(dx, xb)?;
        let r = (|| -> Result<(), String> {
            let fm = self.cc.function(sym)?;
            let (mut m_q, mut m_s, mut m_x, mut m_y) = (dqm, ds, dx, dout);
            let (mut m_n, mut m_k, mut m_t) = (n as i32, k as i32, t as i32);
            self.cc.launch(
                fm,
                t.div_ceil(bm) as u32,
                n.div_ceil(bn) as u32,
                thr,
                &mut crate::rawcuda::args::l7(
                    &mut m_q, &mut m_s, &mut m_x, &mut m_y, &mut m_n, &mut m_k, &mut m_t,
                ),
            )?;
            self.cc.sync()
        })();
        let mut ob = vec![0u8; t * n * 4];
        let d2h = self.cc.d2h(&mut ob, dout);
        let _ = self.cc.free(dqm);
        let _ = self.cc.free(ds);
        let _ = self.cc.free(dx);
        let _ = self.cc.free(dout);
        r?;
        d2h?;
        let got: Vec<f32> = ob
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let mut mism = 0usize;
        let mut first: Vec<(usize, usize, f32, f32)> = Vec::new();
        let mut maxd = 0f32;
        for i in 0..got.len() {
            if got[i].to_bits() != expect[i].to_bits() {
                mism += 1;
                if first.len() < 8 {
                    first.push((i / n, i % n, expect[i], got[i]));
                }
                maxd = maxd.max((got[i] - expect[i]).abs());
            }
        }
        Ok(format!(
            "marlin-selfcheck t={t} n={n} k={k}: 불일치={mism}/{} maxabs={maxd:.3e} (기대,실제)={:?}",
            got.len(),
            first
        ))
    }

    pub fn marlin_bench(
        &mut self,
        name: &str,
        t: usize,
        reps: usize,
        qm: &[u32],
        msym: &str,
        mbm: usize,
        mbn: usize,
        mthr: u32,
        vs_gemv: bool,
        cold: bool,
    ) -> Result<String, String> {
        let _g = self.cc.guard()?;
        let (dq, ds, n, k) = self.lin_spec(name)?;
        let want = (k / 16) * (n / 64) * 128;
        if qm.len() != want {
            return Err(format!(
                "marlin-bench {name}: qm {} != (k/16)(n/64)128 = {want}",
                qm.len()
            ));
        }
        let x16: Vec<u16> = (0..t * k)
            .map(|i| {
                let v =
                    ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as f32 / 4096.0 - 0.5;
                crate::f32_to_f16(v)
            })
            .collect();
        let xb = unsafe { std::slice::from_raw_parts(x16.as_ptr() as *const u8, x16.len() * 2) };
        let dx = self.cc.alloc(t * k * 2)?;
        self.cc.h2d(dx, xb)?;
        let dqm = self.cc.alloc(qm.len() * 4)?;
        let qb = unsafe { std::slice::from_raw_parts(qm.as_ptr() as *const u8, qm.len() * 4) };
        self.cc.h2d(dqm, qb)?;
        let da = self.cc.alloc(t * n * 4)?;
        let db = self.cc.alloc(t * n * 4)?;
        let dx32 = if vs_gemv {
            let x32: Vec<f32> = (0..t * k)
                .map(|i| {
                    ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as f32 / 4096.0 - 0.5
                })
                .collect();
            let xb32 =
                unsafe { std::slice::from_raw_parts(x32.as_ptr() as *const u8, x32.len() * 4) };
            let d = self.cc.alloc(t * k * 4)?;
            self.cc.h2d(d, xb32)?;
            d
        } else {
            0
        };
        let run = |me: &Self,
                   sym: &str,
                   pq: CUdeviceptr,
                   py: CUdeviceptr,
                   bm: usize,
                   bn: usize,
                   thr: u32|
         -> Result<(), String> {
            let f = me.cc.function(sym)?;
            let (mut p_q, mut p_s, mut p_x, mut p_y) = (pq, ds, dx, py);
            let (mut p_n, mut p_k, mut p_t) = (n as i32, k as i32, t as i32);
            me.cc.launch(
                f,
                t.div_ceil(bm) as u32,
                n.div_ceil(bn) as u32,
                thr,
                &mut crate::rawcuda::args::l7(
                    &mut p_q, &mut p_s, &mut p_x, &mut p_y, &mut p_n, &mut p_k, &mut p_t,
                ),
            )
        };
        let run_cmp = |me: &Self, py: CUdeviceptr| -> Result<(), String> {
            if vs_gemv {
                if t == 1 {
                    let f = me.cc.function("w4a16_gemv_g128")?;
                    let (mut p_q, mut p_s, mut p_x, mut p_y) = (dq, ds, dx32, py);
                    let (mut p_n, mut p_k) = (n as i32, k as i32);
                    me.cc.launch(
                        f,
                        n as u32,
                        1,
                        64,
                        &mut crate::rawcuda::args::l6(
                            &mut p_q, &mut p_s, &mut p_x, &mut p_y, &mut p_n, &mut p_k,
                        ),
                    )
                } else {
                    let f = me.cc.function("w4a16_gemv_g128_t")?;
                    let (mut p_q, mut p_s, mut p_x, mut p_y) = (dq, ds, dx32, py);
                    let (mut p_n, mut p_k, mut p_t) = (n as i32, k as i32, t as i32);
                    me.cc.launch(
                        f,
                        n.div_ceil(GEMV_TR) as u32,
                        1,
                        256,
                        &mut crate::rawcuda::args::l7(
                            &mut p_q, &mut p_s, &mut p_x, &mut p_y, &mut p_n, &mut p_k, &mut p_t,
                        ),
                    )
                }
            } else {
                run(
                    me,
                    "w4a16_gemm_g128_mma",
                    dq,
                    py,
                    GEMM_MMA_M,
                    GEMM_MMA_N,
                    256,
                )
            }
        };
        let r = if cold {
            let bytes = 256usize << 20;
            let f0 = self.cc.alloc(bytes)?;
            let f1 = self.cc.alloc(bytes)?;
            let flush = |me: &Self| me.cc.d2d(f0, f1, bytes);
            let r = (|| -> Result<(f64, f64), String> {
                for _ in 0..3 {
                    flush(self)?;
                    run_cmp(self, da)?;
                    flush(self)?;
                    run(self, msym, dqm, db, mbm, mbn, mthr)?;
                }
                self.cc.sync()?;
                let rounds = 4usize;
                let per = (reps / rounds).max(1);
                let (mut base, mut ta, mut tb) = (0f64, 0f64, 0f64);
                for _ in 0..rounds {
                    let t0 = std::time::Instant::now();
                    for _ in 0..per {
                        flush(self)?;
                        self.cc.sync()?;
                    }
                    base += t0.elapsed().as_secs_f64();
                    let t0 = std::time::Instant::now();
                    for _ in 0..per {
                        flush(self)?;
                        self.cc.sync()?;
                        run_cmp(self, da)?;
                        self.cc.sync()?;
                    }
                    ta += t0.elapsed().as_secs_f64();
                    let t0 = std::time::Instant::now();
                    for _ in 0..per {
                        flush(self)?;
                        self.cc.sync()?;
                        run(self, msym, dqm, db, mbm, mbn, mthr)?;
                        self.cc.sync()?;
                    }
                    tb += t0.elapsed().as_secs_f64();
                }
                let n = (per * rounds) as f64;
                Ok(((ta - base) * 1e3 / n, (tb - base) * 1e3 / n))
            })();
            let _ = self.cc.free(f0);
            let _ = self.cc.free(f1);
            r
        } else {
            (|| -> Result<(f64, f64), String> {
                for _ in 0..3 {
                    run_cmp(self, da)?;
                    run(self, msym, dqm, db, mbm, mbn, mthr)?;
                }
                self.cc.sync()?;
                let rounds = 4usize;
                let per = (reps / rounds).max(1);
                let (mut ta, mut tb) = (0f64, 0f64);
                for _ in 0..rounds {
                    let t0 = std::time::Instant::now();
                    for _ in 0..per {
                        run_cmp(self, da)?;
                    }
                    self.cc.sync()?;
                    ta += t0.elapsed().as_secs_f64();
                    let t0 = std::time::Instant::now();
                    for _ in 0..per {
                        run(self, msym, dqm, db, mbm, mbn, mthr)?;
                    }
                    self.cc.sync()?;
                    tb += t0.elapsed().as_secs_f64();
                }
                let n = (per * rounds) as f64;
                Ok((ta * 1e3 / n, tb * 1e3 / n))
            })()
        };
        let _ = self.cc.free(dx);
        if vs_gemv {
            let _ = self.cc.free(dx32);
        }
        let _ = self.cc.free(dqm);
        let _ = self.cc.free(da);
        let _ = self.cc.free(db);
        let (ms_a, ms_b) = r?;
        let flop = 2.0 * n as f64 * k as f64 * t as f64;
        let cmp = if vs_gemv { "gemv" } else { "mma" };
        let mode = if cold { "cold" } else { "warm" };
        Ok(format!(
            "marlin-bench {name} n={n} k={k} t={t} reps={reps} [{mode}]: \
             {cmp} {ms_a:.3}ms({:.1}TF) · marlin {ms_b:.3}ms({:.1}TF) · 비율 {:.3}×",
            flop * 1e-9 / (ms_a * 1e-3),
            flop * 1e-9 / (ms_b * 1e-3),
            ms_a / ms_b
        ))
    }

    pub fn attn_fa2_bench(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        reps: usize,
        pos0: u32,
    ) -> Result<String, String> {
        let _g = self.cc.guard()?;
        let dm = self.attn.attn.ok_or("attn: 형상 미등록")?;
        let (qgd, kvd) = (dm.qg_dim(), dm.kv_dim());
        let mut seed = 0x243F_6A88_85A3_08D3u64 ^ ((t_len as u64) << 32);
        let mut rnd = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 40) as f32 / 2048.0) - 0.5
        };
        let qg: Vec<f32> = (0..t_len * qgd).map(|_| rnd()).collect();
        let kin: Vec<f32> = (0..t_len * kvd).map(|_| rnd()).collect();
        let vin: Vec<f32> = (0..t_len * kvd).map(|_| rnd()).collect();
        self.ensure_attn_bufs(t_len)?;
        let b =
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        self.cc.h2d(self.attn.dqg_a, b(&qg))?;
        self.cc.h2d(self.attn.dkin_a, b(&kin))?;
        self.cc.h2d(self.attn.dvin_a, b(&vin))?;
        self.attn_set_pos(slot, pos0)?;
        let (aq, ak, av) = (self.attn.dqg_a, self.attn.dkin_a, self.attn.dvin_a);
        let r = (|| -> Result<(f64, f64), String> {
            for _ in 0..2 {
                self.attn_prep_launch(slot, layer, t_len, aq, ak, av)?;
                self.attn_fwd3s_launch(slot, layer, t_len, aq)?;
                self.attn_prep_launch(slot, layer, t_len, aq, ak, av)?;
                self.attn_fa2_launch(slot, layer, t_len, aq)?;
            }
            self.cc.sync()?;
            let rounds = 4usize;
            let per = (reps / rounds).max(1);
            let (mut ta, mut tb) = (0f64, 0f64);
            for _ in 0..rounds {
                let t0 = std::time::Instant::now();
                for _ in 0..per {
                    self.attn_prep_launch(slot, layer, t_len, aq, ak, av)?;
                    self.attn_fwd3s_launch(slot, layer, t_len, aq)?;
                }
                self.cc.sync()?;
                ta += t0.elapsed().as_secs_f64();
                let t0 = std::time::Instant::now();
                for _ in 0..per {
                    self.attn_prep_launch(slot, layer, t_len, aq, ak, av)?;
                    self.attn_fa2_launch(slot, layer, t_len, aq)?;
                }
                self.cc.sync()?;
                tb += t0.elapsed().as_secs_f64();
            }
            let n = (per * rounds) as f64;
            Ok((ta * 1e3 / n, tb * 1e3 / n))
        })();
        let (old_ms, new_ms) = r?;
        Ok(format!(
            "attn-fa2-bench t={t_len} pos0={pos0} layer={layer} reps={reps}: 기존 {old_ms:.3}ms · FA2 {new_ms:.3}ms · 비율 {:.3}× (prep 포함)",
            old_ms / new_ms
        ))
    }

    pub fn attn_fa2_diff(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        pos0: u32,
    ) -> Result<String, String> {
        let _g = self.cc.guard()?;
        let dm = self.attn.attn.ok_or("attn: 형상 미등록")?;
        let (qgd, kvd, qdd) = (dm.qg_dim(), dm.kv_dim(), dm.q_dim());
        let mut seed = 0x9E37_79B9_7F4A_7C15u64 ^ ((t_len as u64) << 32) ^ (layer as u64);
        let mut rnd = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 40) as f32 / 2048.0) - 0.5
        };
        let qg: Vec<f32> = (0..t_len * qgd).map(|_| rnd()).collect();
        let kin: Vec<f32> = (0..t_len * kvd).map(|_| rnd()).collect();
        let vin: Vec<f32> = (0..t_len * kvd).map(|_| rnd()).collect();
        self.ensure_attn_bufs(t_len)?;
        let b =
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        self.cc.h2d(self.attn.dqg_a, b(&qg))?;
        self.cc.h2d(self.attn.dkin_a, b(&kin))?;
        self.cc.h2d(self.attn.dvin_a, b(&vin))?;
        self.attn_set_pos(slot, pos0)?;
        let (aq, ak, av) = (self.attn.dqg_a, self.attn.dkin_a, self.attn.dvin_a);
        self.attn_prep_launch(slot, layer, t_len, aq, ak, av)?;
        self.attn_fwd3s_legacy_launch(slot, layer, t_len, aq)?;
        let mut obuf = vec![0u8; t_len * qdd * 4];
        self.cc.d2h(&mut obuf, self.attn.doutv_a)?;
        self.cc.sync()?;
        let old: Vec<f32> = obuf
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        self.attn_prep_launch(slot, layer, t_len, aq, ak, av)?;
        self.attn_fa2_launch(slot, layer, t_len, self.attn.dqg_a)?;
        let mut buf = vec![0u8; t_len * qdd * 4];
        self.cc.d2h(&mut buf, self.attn.doutv_a)?;
        self.cc.sync()?;
        let new: Vec<f32> = buf
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let mut maxd = 0f32;
        let mut maxr = 0f32;
        let mut bad = 0usize;
        let mut first: Vec<(usize, f32, f32)> = Vec::new();
        for i in 0..old.len() {
            let d = (old[i] - new[i]).abs();
            if d > maxd {
                maxd = d;
            }
            let rl = d / old[i].abs().max(1e-3);
            if rl > maxr {
                maxr = rl;
            }
            if rl > 5e-3 {
                bad += 1;
                if first.len() < 5 {
                    first.push((i, old[i], new[i]));
                }
            }
        }
        Ok(format!(
            "attn-fa2-diff t={t_len} pos0={pos0} layer={layer}: maxabs={maxd:.3e} maxrel={maxr:.3e} rel>5e-3={bad}/{} 예={:?}",
            old.len(),
            first
        ))
    }

    pub fn bench_ew(&mut self) -> Result<String, String> {
        let _g = self.cc.guard()?;
        let n = 245_760usize;
        let dg = self.cc.alloc(n * 4)?;
        let du = self.cc.alloc(n * 4)?;
        let dy = self.cc.alloc(n * 4)?;
        let v: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.1 - 0.3).collect();
        let b = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n * 4) };
        self.cc.h2d(dg, b)?;
        self.cc.h2d(du, b)?;
        let f = self.cc.function("ew")?;
        let (mut a0, mut a1, mut a2, mut a3, mut a4) = (dg, du, dy, 0u64, 0u64);
        let mut nn = n as i32;
        for _ in 0..10 {
            self.cc.launch(
                f,
                n.div_ceil(128) as u32,
                1,
                128,
                &mut crate::rawcuda::args::l6(&mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut nn),
            )?;
        }
        self.cc.sync()?;
        let t0 = std::time::Instant::now();
        for _ in 0..100 {
            self.cc.launch(
                f,
                n.div_ceil(128) as u32,
                1,
                128,
                &mut crate::rawcuda::args::l6(&mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut nn),
            )?;
        }
        self.cc.sync()?;
        let el = t0.elapsed().as_secs_f64() / 100.0;
        let _ = self.cc.free(dg);
        let _ = self.cc.free(du);
        let _ = self.cc.free(dy);
        Ok(format!(
            "ew bench n={n}: {:.3}ms/launch · {:.1} GB/s (12B/elem)",
            el * 1000.0,
            (n as f64 * 12.0) / el / 1e9
        ))
    }

    pub fn moe_topk_check(&mut self, t: usize, n: usize, k: usize) -> Result<String, String> {
        let _g = self.cc.guard()?;
        let lg: Vec<f32> = (0..t * n)
            .map(|i| ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as f32 / 2048.0 - 0.5)
            .collect();
        let lb = unsafe { std::slice::from_raw_parts(lg.as_ptr() as *const u8, lg.len() * 4) };
        let dl = self.cc.alloc(t * n * 4)?;
        let di = self.cc.alloc(t * k * 4)?;
        let dw = self.cc.alloc(t * k * 4)?;
        self.cc.h2d(dl, lb)?;
        let f = self.cc.function("w4a16_moe_topk_t")?;
        let (mut p_lg, mut p_ix, mut p_wt) = (dl, di, dw);
        let (mut p_t, mut p_n, mut p_k) = (t as i32, n as i32, k as i32);
        self.cc.launch(
            f,
            t.div_ceil(8) as u32,
            1,
            256,
            &mut crate::rawcuda::args::l6(
                &mut p_lg, &mut p_ix, &mut p_wt, &mut p_t, &mut p_n, &mut p_k,
            ),
        )?;
        self.cc.sync()?;
        let mut ib = vec![0u8; t * k * 4];
        let mut wb = vec![0u8; t * k * 4];
        self.cc.d2h(&mut ib, di)?;
        self.cc.d2h(&mut wb, dw)?;
        let gi: Vec<u32> = ib
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        let gw: Vec<f32> = wb
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let _ = self.cc.free(dl);
        let _ = self.cc.free(di);
        let _ = self.cc.free(dw);
        let mut idx_bad = 0usize;
        let mut wmax = 0f32;
        for ti in 0..t {
            let host = moe_topk(&lg[ti * n..(ti + 1) * n], k);
            for (r, (e, w)) in host.iter().enumerate() {
                if gi[ti * k + r] as usize != *e {
                    idx_bad += 1;
                }
                let d = (gw[ti * k + r] - w).abs();
                if d > wmax {
                    wmax = d;
                }
            }
        }
        Ok(format!(
            "moe-topk-check t={t} n={n} k={k}: idx 불일치 {idx_bad}/{} · |Δw|max={wmax:.3e} · dev[0..4]={:?} host[0..4]={:?}",
            t * k,
            &gi[..4],
            &moe_topk(&lg[..n], k)
                .iter()
                .map(|&(e, _)| e as u32)
                .collect::<Vec<_>>()[..4]
        ))
    }

    pub(super) fn mma_diff_plain(&mut self, name: &str, t: usize) -> Result<String, String> {
        let (w, n, k) = self.plain_spec(name)?;
        let xf: Vec<f32> = (0..t * k)
            .map(|i| ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as f32 / 4096.0 - 0.5)
            .collect();
        let xb = unsafe { std::slice::from_raw_parts(xf.as_ptr() as *const u8, xf.len() * 4) };
        let dx = self.cc.alloc(t * k * 4)?;
        self.cc.h2d(dx, xb)?;
        let da = self.cc.alloc(t * n * 4)?;
        let db = self.cc.alloc(t * n * 4)?;
        let r = (|| -> Result<(Vec<f32>, Vec<f32>), String> {
            let f = self.cc.function("w4a16_gemm_bf16_t")?;
            let (mut p_w, mut p_x, mut p_o) = (w, dx, da);
            let (mut p_n, mut p_k, mut p_t) = (n as i32, k as i32, t as i32);
            self.cc.launch(
                f,
                n.div_ceil(8) as u32,
                1,
                512,
                &mut crate::rawcuda::args::l6(
                    &mut p_w, &mut p_x, &mut p_o, &mut p_n, &mut p_k, &mut p_t,
                ),
            )?;
            let fm = self.cc.function("w4a16_gemm_bf16_mma2")?;
            let (mut m_w, mut m_x, mut m_o) = (w, dx, db);
            let (mut m_n, mut m_k, mut m_t) = (n as i32, k as i32, t as i32);
            let mut m_b = 0i32;
            self.cc.launch(
                fm,
                t.div_ceil(GEMM_BMMA_M) as u32,
                n.div_ceil(GEMM_BMMA_N) as u32,
                256,
                &mut crate::rawcuda::args::l7(
                    &mut m_w, &mut m_x, &mut m_o, &mut m_n, &mut m_k, &mut m_t, &mut m_b,
                ),
            )?;
            self.cc.sync()?;
            let mut oa = vec![0u8; t * n * 4];
            let mut ob = vec![0u8; t * n * 4];
            self.cc.d2h(&mut oa, da)?;
            self.cc.d2h(&mut ob, db)?;
            let fa: Vec<f32> = oa
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            let fb: Vec<f32> = ob
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            Ok((fa, fb))
        })();
        let _ = self.cc.free(dx);
        let _ = self.cc.free(da);
        let _ = self.cc.free(db);
        let (fa, fb) = r?;
        let mut maxd = 0f32;
        let mut bad = 0usize;
        for i in 0..fa.len() {
            let d = (fa[i] - fb[i]).abs();
            if d > maxd {
                maxd = d;
            }
            if !fb[i].is_finite() {
                bad += 1;
            }
        }
        Ok(format!(
            "mma-diff(plain) {name} n={n} k={k} t={t}: maxabs={maxd:.3e} 비유한={bad}/{} A[0..3]={:?} B[0..3]={:?}",
            fa.len(),
            &fa[..3],
            &fb[..3]
        ))
    }

    pub fn plain_bench_launch(
        &mut self,
        name: &str,
        x: CUdeviceptr,
        y: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        self.plain_gemm_launch(name, x, y, t)
    }

    pub fn gemv_bench_launch(
        &mut self,
        name: &str,
        x: CUdeviceptr,
        y: CUdeviceptr,
    ) -> Result<(), String> {
        self.gemv_launch(name, x, y)
    }

    pub fn decode_bench(&mut self, name: &str, reps: usize, cold: bool) -> Result<String, String> {
        let _g = self.cc.guard()?;
        let (dq, ds, n, k) = self.lin_spec(name)?;
        let Some((dqm, _, _)) = self.lin_mk(name) else {
            return Err(format!(
                "decode-bench {name}: marlin 재패킹 상주 없음(업로드 확인)"
            ));
        };
        if !k.is_multiple_of(128) {
            return Err(format!("decode-bench {name}: k={k}%128 계약"));
        }
        let x16: Vec<u16> = (0..k)
            .map(|i| {
                let v =
                    ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as f32 / 4096.0 - 0.5;
                crate::f32_to_f16(v)
            })
            .collect();
        let x32: Vec<f32> = (0..k)
            .map(|i| ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as f32 / 4096.0 - 0.5)
            .collect();
        let hb = unsafe { std::slice::from_raw_parts(x16.as_ptr() as *const u8, x16.len() * 2) };
        let fb = unsafe { std::slice::from_raw_parts(x32.as_ptr() as *const u8, x32.len() * 4) };
        let dxh = self.cc.alloc(k * 2)?;
        let dxf = self.cc.alloc(k * 4)?;
        self.cc.h2d(dxh, hb)?;
        self.cc.h2d(dxf, fb)?;
        let blocks = n.div_ceil(256);
        let part = self.cc.alloc(blocks * 16 * 256 * 4)?;
        let ya = self.cc.alloc(n * 4)?;
        let yb = self.cc.alloc(n * 4)?;
        let ksps: Vec<usize> = [1usize, 2, 4, 8, 16]
            .into_iter()
            .filter(|&p| (k / 128) % p == 0)
            .collect();
        let run_gemv = |me: &Self, y: CUdeviceptr| -> Result<(), String> {
            let f = me.cc.function("w4a16_gemv_g128")?;
            let (mut p_q, mut p_s, mut p_x, mut p_y) = (dq, ds, dxf, y);
            let (mut p_n, mut p_k) = (n as i32, k as i32);
            me.cc.launch(
                f,
                n as u32,
                1,
                64,
                &mut crate::rawcuda::args::l6(
                    &mut p_q, &mut p_s, &mut p_x, &mut p_y, &mut p_n, &mut p_k,
                ),
            )
        };
        let run_sk = |me: &Self, y: CUdeviceptr, ksp: usize| -> Result<(), String> {
            let f = me.cc.function("w4a16_gemv_sk1_marlin")?;
            let (mut p_q, mut p_s, mut p_x, mut p_o, mut p_p) = (dqm, ds, dxh, y, part);
            let (mut p_n, mut p_k, mut p_sp) = (n as i32, k as i32, ksp as i32);
            me.cc.launch(
                f,
                blocks as u32,
                ksp as u32,
                128,
                &mut crate::rawcuda::args::l8(
                    &mut p_q, &mut p_s, &mut p_x, &mut p_o, &mut p_p, &mut p_n, &mut p_k, &mut p_sp,
                ),
            )?;
            if ksp > 1 {
                let f = me.cc.function("w4a16_gemv_sk1_reduce")?;
                me.cc.launch(
                    f,
                    blocks as u32,
                    1,
                    256,
                    &mut crate::rawcuda::args::l4(&mut p_p, &mut p_o, &mut p_n, &mut p_sp),
                )
            } else {
                Ok(())
            }
        };
        let flush_bytes = 256usize << 20;
        let (f0, f1) = if cold {
            (self.cc.alloc(flush_bytes)?, self.cc.alloc(flush_bytes)?)
        } else {
            (0, 0)
        };
        let r = (|| -> Result<Vec<String>, String> {
            run_gemv(self, ya)?;
            run_sk(self, yb, ksps[0])?;
            self.cc.sync()?;
            let mut oa = vec![0u8; n * 4];
            let mut ob = vec![0u8; n * 4];
            self.cc.d2h(&mut oa, ya)?;
            self.cc.d2h(&mut ob, yb)?;
            self.cc.sync()?;
            let (mut maxd, mut maxr) = (0f64, 0f64);
            for i in 0..n {
                let a = f32::from_le_bytes(oa[i * 4..i * 4 + 4].try_into().unwrap());
                let b = f32::from_le_bytes(ob[i * 4..i * 4 + 4].try_into().unwrap());
                maxd = maxd.max((a - b).abs() as f64);
                maxr = maxr.max(((a - b).abs() / a.abs().max(1e-9)) as f64);
            }
            let mut acc = vec![0f64; ksps.len() + 1];
            let rounds = 4usize;
            let per = (reps / rounds).max(1);
            for _ in 0..rounds {
                let mut t0 = std::time::Instant::now();
                for _ in 0..per {
                    if cold {
                        self.cc.d2d(f0, f1, flush_bytes)?;
                        self.cc.sync()?;
                        t0 = std::time::Instant::now();
                    }
                    run_gemv(self, ya)?;
                    if cold {
                        self.cc.sync()?;
                        acc[0] += t0.elapsed().as_secs_f64();
                    }
                }
                if !cold {
                    self.cc.sync()?;
                    acc[0] += t0.elapsed().as_secs_f64();
                }
                for (vi, &ksp) in ksps.iter().enumerate() {
                    let mut t0 = std::time::Instant::now();
                    for _ in 0..per {
                        if cold {
                            self.cc.d2d(f0, f1, flush_bytes)?;
                            self.cc.sync()?;
                            t0 = std::time::Instant::now();
                        }
                        run_sk(self, yb, ksp)?;
                        if cold {
                            self.cc.sync()?;
                            acc[vi + 1] += t0.elapsed().as_secs_f64();
                        }
                    }
                    if !cold {
                        self.cc.sync()?;
                        acc[vi + 1] += t0.elapsed().as_secs_f64();
                    }
                }
            }
            let mut out = Vec::new();
            out.push(format!(
                "decode-bench {name} n={n} k={k} blocks={blocks} reps={reps} [{}]: \
                 gemv {ms:.3}ms · 값대조 maxabs={maxd:.3e} maxrel={maxr:.3e}",
                if cold { "cold" } else { "warm" },
                ms = acc[0] * 1e3 / (per * rounds) as f64,
            ));
            let wb = (n * k) as f64 / 2.0 + (n * k / 128) as f64 * 2.0;
            for (vi, &ksp) in ksps.iter().enumerate() {
                let ms = acc[vi + 1] * 1e3 / (per * rounds) as f64;
                out.push(format!(
                    "  ksp={ksp}: {ms:.3}ms · {:.0}GB/s · gemv 대비 {:.3}×",
                    wb * 1e-9 / (ms * 1e-3),
                    acc[0] / acc[vi + 1],
                ));
            }
            Ok(out)
        })();
        let _ = self.cc.free(dxh);
        let _ = self.cc.free(dxf);
        let _ = self.cc.free(part);
        let _ = self.cc.free(ya);
        let _ = self.cc.free(yb);
        if cold {
            let _ = self.cc.free(f0);
            let _ = self.cc.free(f1);
        }
        Ok(r?.join("\n"))
    }

    pub fn decode_selfcheck(&mut self) -> Result<String, String> {
        let _g = self.cc.guard()?;
        let (n, k) = (256usize, 512usize);
        let q: Vec<u32> = (0..n * k / 8)
            .map(|i| (i as u32).wrapping_mul(0x9E37_79B9) ^ 0x1234_5678u32)
            .collect();
        let s: Vec<u16> = vec![0x3C00; n * k / 128];
        let x16: Vec<u16> = vec![0x3C00; k];
        let qm = llm170_core::marlin::marlin_repack_q(&q, n, k)?;
        let mut want = vec![0f32; n];
        for c in 0..n {
            let mut acc = 0i64;
            for j in 0..k {
                let nib = ((q[c * k / 8 + j / 8] >> ((j % 8) * 4)) & 0xF) as i64 - 8;
                acc += nib;
            }
            want[c] = acc as f32;
        }
        let dqm = self.cc.alloc(qm.len() * 4)?;
        let qb = unsafe { std::slice::from_raw_parts(qm.as_ptr() as *const u8, qm.len() * 4) };
        let sb = unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 2) };
        let xb = unsafe { std::slice::from_raw_parts(x16.as_ptr() as *const u8, x16.len() * 2) };
        self.cc.h2d(dqm, qb)?;
        let ds = self.cc.alloc(sb.len())?;
        self.cc.h2d(ds, sb)?;
        let dx = self.cc.alloc(xb.len())?;
        self.cc.h2d(dx, xb)?;
        let blocks = n.div_ceil(256);
        let part = self.cc.alloc(blocks * 4 * 256 * 4)?;
        let y = self.cc.alloc(n * 4)?;
        let mut out = Vec::new();
        let mut bad = 0usize;
        let mut check = |got: &[u8], tag: &str| {
            let (mut mism, mut first) = (0usize, None);
            for c in 0..n {
                let v = f32::from_le_bytes(got[c * 4..c * 4 + 4].try_into().unwrap());
                if v.to_bits() != want[c].to_bits() {
                    mism += 1;
                    if first.is_none() {
                        first = Some((c, want[c], v));
                    }
                }
            }
            if mism > 0 {
                bad += 1;
            }
            out.push(format!(
                "  {tag}: 불일치={mism}/{n}{}",
                first
                    .map(|(c, w, v)| format!(" 첫불량=(col{c}, 기대 {w}, 실제 {v})"))
                    .unwrap_or_default(),
            ));
        };
        for ksp in [1usize, 2, 4] {
            let zero = vec![0u8; n * 4];
            self.cc.h2d(y, &zero)?;
            let f = self.cc.function("w4a16_gemv_sk1_marlin")?;
            let (mut p_q, mut p_s, mut p_x, mut p_o, mut p_p) = (dqm, ds, dx, y, part);
            let (mut p_n, mut p_k, mut p_sp) = (n as i32, k as i32, ksp as i32);
            self.cc.launch(
                f,
                blocks as u32,
                ksp as u32,
                128,
                &mut crate::rawcuda::args::l8(
                    &mut p_q, &mut p_s, &mut p_x, &mut p_o, &mut p_p, &mut p_n, &mut p_k, &mut p_sp,
                ),
            )?;
            if ksp > 1 {
                let f = self.cc.function("w4a16_gemv_sk1_reduce")?;
                self.cc.launch(
                    f,
                    blocks as u32,
                    1,
                    256,
                    &mut crate::rawcuda::args::l4(&mut p_p, &mut p_o, &mut p_n, &mut p_sp),
                )?;
            }
            let mut got = vec![0u8; n * 4];
            self.cc.d2h(&mut got, y)?;
            self.cc.sync()?;
            check(&got, &format!("sk1 ksp={ksp}"));
        }
        {
            let (y2, part2) = (self.cc.alloc(n * 4)?, self.cc.alloc(2 * 2 * 256 * 4)?);
            let tab: Vec<u64> = vec![
                dqm, ds, y, n as u64, k as u64, dqm, ds, y2, n as u64, k as u64,
            ];
            let tb =
                unsafe { std::slice::from_raw_parts(tab.as_ptr() as *const u8, tab.len() * 8) };
            let dt = self.cc.alloc(tb.len())?;
            self.cc.h2d(dt, tb)?;
            let (mut p_t, mut p_x, mut p_p) = (dt, dx, part2);
            let (mut p_c, mut p_s) = (2i32, 2i32);
            let f = self.cc.function("w4a16_gemv_sk_marlin")?;
            self.cc.launch(
                f,
                2,
                2,
                128,
                &mut crate::rawcuda::args::l5(&mut p_t, &mut p_x, &mut p_p, &mut p_c, &mut p_s),
            )?;
            let f = self.cc.function("w4a16_gemv_sk_reduce")?;
            self.cc.launch(
                f,
                2,
                1,
                256,
                &mut crate::rawcuda::args::l4(&mut p_t, &mut p_p, &mut p_c, &mut p_s),
            )?;
            let mut got = vec![0u8; n * 4];
            let mut got2 = vec![0u8; n * 4];
            self.cc.d2h(&mut got, y)?;
            self.cc.d2h(&mut got2, y2)?;
            self.cc.sync()?;
            check(&got, "multi seg0 ksp=2");
            check(&got2, "multi seg1 ksp=2");
            let _ = self.cc.free(y2);
            let _ = self.cc.free(part2);
            let _ = self.cc.free(dt);
        }
        let _ = self.cc.free(dqm);
        let _ = self.cc.free(ds);
        let _ = self.cc.free(dx);
        let _ = self.cc.free(part);
        let _ = self.cc.free(y);
        let summary = if bad == 0 {
            "전체 통과".to_string()
        } else {
            format!("불일치 케이스 {bad}")
        };
        Ok(format!(
            "decode-selfcheck n={n} k={k} (기대=Σ(nib−8) 비트정확):\n{}\n{summary}",
            out.join("\n")
        ))
    }

    pub fn gemm_bench_launch(
        &mut self,
        name: &str,
        xh: CUdeviceptr,
        y: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        self.gemm_launch(name, xh, y, t)
    }
}
