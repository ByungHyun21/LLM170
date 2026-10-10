use super::*;

impl W4a16Dec {
    /// [A9 진단] t행 GEMV 벤치(ms/회) — 지정 선형 반복.
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

    /// 플레인 GEMM 자가 점검 — 배치 GEMM(t=8) vs 토큰별 GEMV 비트 비교
    /// (판정 계약: 플레인 경로는 토큰 수준이나 같은 레인/환원 순서라 동일해야
    /// 한다 — 다르면 t 처리 결함).
    /// [2026-10-09] 텐서코어 도구·수치 스모크 — bf16 mma.m16n8k16 → f32 누적을
    /// CPU 참조(bf16 RN 반올림 입력 + f32 k순 합)와 대조. 차이는 누적 순서뿐
    /// (허용오차 1e-4). 1b(플레인 mma GEMM) 착륙 전 도구·프래그먼트 검증.
    pub fn mma_smoke(&mut self) -> Result<String, String> {
        let _g = self.cc.guard()?;
        let a = crate::rawcuda::assets::asset("smoke");
        self.cc
            .load_fatbin(a.name, &crate::rawcuda::assets::asset_bytes(a)?, a.syms)?;
        // 결정적 준난수 ∈ [-1, 1) — 곱·합 ≤ 16이라 f32 누적순서 오차 ~1e-6.
        let mk = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 53) as f32 / 1024.0 - 1.0
                })
                .collect()
        };
        let (a, b) = (mk(16 * 16), mk(16 * 8));
        // bf16 RN(짝수) — 커널 __floats2bfloat162_rn과 동일 규약.
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
            // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
            let ab = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len() * 4) };
            // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
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
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
        let xb = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) };
        let dx = self.cc.alloc(t * k * 4)?;
        let da = self.cc.alloc(t * n * 4)?;
        let db = self.cc.alloc(t * n * 4)?;
        let r = (|| -> Result<(Vec<f32>, Vec<f32>), String> {
            self.cc.h2d(dx, xb)?;
            // A: 배치 GEMM(v1, t=8)
            self.plain_gemm_launch(&name, dx, da, t)?;
            // B: 토큰별 GEMV(t=1) ×8 → 이어붙임
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

    /// MoE 자가 점검 — 직접 GEMV vs 배치(간접) GEMV 비트 비교(층0·전문가0·
    /// gate_proj). 상주 기기 브링업·회귀 판정용.
    pub fn moe_selfcheck(&mut self) -> Result<String, String> {
        if !self.moe_resident || self.n_experts == 0 {
            return Err("moe_selfcheck: 상주 MoE 구성 필요".into());
        }
        let _g = self.cc.guard()?;
        self.ensure_moe_bufs()?;
        let k = self.hidden;
        let n = self.moe_ffn;
        if self.moe_tab.is_empty() {
            return Err("moe_selfcheck: 테이블 부재".into());
        }
        // x — 결정적(splitmix64 계열 상수) ±0.5.
        let x: Vec<f32> = (0..k)
            .map(|i| ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as f32 / 2048.0 - 0.5)
            .collect();
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
        let xb = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, k * 4) };
        let dx = self.cc.alloc(k * 4)?;
        let da = self.cc.alloc(n * 4)?;
        let db = self.cc.alloc(n * 4)?;
        let r = (|| -> Result<(Vec<f32>, Vec<f32>), String> {
            self.cc.h2d(dx, xb)?;
            // 직접 — moe_tab[0] = (층0, 전문가0, gate_proj).
            let e0 = self.moe_tab[0];
            self.gemv_launch_raw(e0.0, e0.2, n, k, dx, da)?;
            // 배치 — idx=[0], base=0, nslots=1.
            let idx = [0u32];
            // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
            let ib = unsafe { std::slice::from_raw_parts(idx.as_ptr() as *const u8, 4) };
            self.cc.h2d(self.moe_idx, ib)?;
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

    /// GEMV 전 선형 1회 순회(벤치) — 실사용과 동일한 DRAM 스트림(13.9GB ≫ L2).
    /// 반환: (ms/회, 가중치 바이트 합). x는 k별 1.0 f32(수치 무의미).
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
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
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

    /// mma GEMM 수치 게이트(허용오차) — split 원본(w4a16_gemm_g128) vs
    /// w4a16_gemm_g128_mma 출력을 같은 x로 비교. (T1 검증 — 계약 완화 후
    /// 비트 대신 허용오차 판정.)
    pub fn mma_diff_check(&mut self, name: &str, t: usize) -> Result<String, String> {
        let _g = self.cc.guard()?;
        // 플레인(bf16)이면 T2 경로로.
        if !self.lins.contains_key(name) {
            return self.mma_diff_plain(name, t);
        }
        let (dq, ds, n, k) = self.lin_spec(name)?;
        let xf: Vec<f32> = (0..t * k)
            .map(|i| ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as f32 / 4096.0 - 0.5)
            .collect();
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
        let xb = unsafe { std::slice::from_raw_parts(xf.as_ptr() as *const u8, xf.len() * 4) };
        let dx = self.cc.alloc(t * k * 4)?;
        self.cc.h2d(dx, xb)?;
        let da = self.cc.alloc(t * n * 4)?;
        let db = self.cc.alloc(t * n * 4)?;
        let r = (|| -> Result<(Vec<f32>, Vec<f32>), String> {
            // 원본(8행/블록, 512스레드).
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
            // mma(32×64, 256스레드).
            let fm = self.cc.function("w4a16_gemm_g128_mma")?;
            let (mut m_q, mut m_s, mut m_x, mut m_y) = (dq, ds, dx, db);
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
        // 불량 위치 패턴(첫 5개 + 행·열 분포) — 디버그.
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

    /// [진단] ew 커널 독립 벤치 — 순수 커널 처리량(파이프라인 무관).
    pub fn bench_ew(&mut self) -> Result<String, String> {
        let _g = self.cc.guard()?;
        let n = 245_760usize;
        let dg = self.cc.alloc(n * 4)?;
        let du = self.cc.alloc(n * 4)?;
        let dy = self.cc.alloc(n * 4)?;
        let v: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.1 - 0.3).collect();
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
        let b = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n * 4) };
        self.cc.h2d(dg, b)?;
        self.cc.h2d(du, b)?;
        let f = self.cc.function("ew")?;
        let (mut a0, mut a1, mut a2) = (dg, du, dy);
        let mut nn = n as i32;
        for _ in 0..10 {
            self.cc.launch(
                f,
                n.div_ceil(128) as u32,
                1,
                128,
                &mut crate::rawcuda::args::l4(&mut a0, &mut a1, &mut a2, &mut nn),
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
                &mut crate::rawcuda::args::l4(&mut a0, &mut a1, &mut a2, &mut nn),
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

    /// MoE top-k 게이트 — 호스트 moe_topk vs 디바이스 w4a16_moe_topk_t.
    pub fn moe_topk_check(&mut self, t: usize, n: usize, k: usize) -> Result<String, String> {
        let _g = self.cc.guard()?;
        let lg: Vec<f32> = (0..t * n)
            .map(|i| ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as f32 / 2048.0 - 0.5)
            .collect();
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
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

    /// 플레인 mma 수치 게이트 — v3(원본) vs bf16 mma.
    pub(super) fn mma_diff_plain(&mut self, name: &str, t: usize) -> Result<String, String> {
        let (w, n, k) = self.plain_spec(name)?;
        let xf: Vec<f32> = (0..t * k)
            .map(|i| ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as f32 / 4096.0 - 0.5)
            .collect();
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
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
            let fm = self.cc.function("w4a16_gemm_bf16_mma")?;
            let (mut m_w, mut m_x, mut m_o) = (w, dx, db);
            let (mut m_n, mut m_k, mut m_t) = (n as i32, k as i32, t as i32);
            let mut m_b = 0i32; // [FLA-10] 게이트는 f32 경로(xbf=0).
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

    /// 플레인 GEMM 단발 발사(벤치·진단 — T2 mma 비교용).
    pub fn plain_bench_launch(
        &mut self,
        name: &str,
        x: CUdeviceptr,
        y: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        self.plain_gemm_launch(name, x, y, t)
    }

    /// GEMV 단발 발사(벤치·진단 — import 없이 이름·x·y만).
    pub fn gemv_bench_launch(
        &mut self,
        name: &str,
        x: CUdeviceptr,
        y: CUdeviceptr,
    ) -> Result<(), String> {
        self.gemv_launch(name, x, y)
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
