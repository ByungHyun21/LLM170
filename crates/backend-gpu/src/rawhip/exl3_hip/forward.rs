//! EXL3 hip 순방향 오케스트라(forward) — 테일 [마커] 주석은 원본 단계 원장(보존).
use super::{Exl3HipDecoder, HipLin};

impl Exl3HipDecoder {
    /// 배치 GEMM(dah16 → out [T][n]) — gemm2 커널.
    /// 1토큰 forward → 로짓. ew(silu·mul)는 호스트(정확성 우선 — 추후 커널화).
    pub fn forward(&mut self, embed_row: &[f32]) -> Result<(Vec<f32>, Vec<f32>), String> {
        // KV 상한 가드(plans/128 P0) — 초과 시 폴트 대신 Err로 우아한 거절.
        if self.pos as usize + 1 > self.kvcap as usize {
            return Err(format!(
                "context overflow: pos={} + 1 > kvcap={} (--ctx 상향 필요)",
                self.pos, self.kvcap
            ));
        }
        let n_layers = self.loaded_layers.min(self.n_layers);
        let f32b =
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        self.hc.h2d(self.dx, f32b(embed_row))?;
        let mut ab = self.dzero;
        for il in 0..n_layers {
            let lp = format!("model.language_model.layers.{il}");
            let gdn_il = (0..il).filter(|i| i % 4 != 3).count();
            self.norm(2 * il, ab)?;
            if llm170_diag::dump::opts().key("htrace") {
                let mut hb = vec![0u8; self.hidden * 4];
                let _ = self.hc.d2h(&mut hb, self.dx);
                let _ = self.hc.sync();
                // SAFETY: d2h 완료 후 재해석.
                let row = unsafe {
                    std::slice::from_raw_parts(hb.as_ptr() as *const f32, self.hidden).to_vec()
                };
                self.htrace.push(vec![row]); // 플랫: 호출당 64항 — [호출×64+il][1행]
            }
            if il == 1 {
                let mut xb2 = vec![0u8; self.hidden * 4];
                self.hc.d2h(&mut xb2, self.dx)?;
                // SAFETY: d2h 완료 후 재해석.
                let xf2: &[f32] =
                    unsafe { std::slice::from_raw_parts(xb2.as_ptr() as *const f32, self.hidden) };
                let r2 = (xf2.iter().map(|v| v * v).sum::<f32>() / self.hidden as f32).sqrt();
                eprintln!("  [hipl] L1 진입 x rms={r2:.5} x0={:.6}", xf2[0]);
            }
            if il <= 1 {
                let mut xnb = vec![0u8; self.hidden * 4];
                self.hc.d2h(&mut xnb, self.dxn)?;
                // SAFETY: d2h 완료 후 재해석.
                let xnf: &[f32] =
                    unsafe { std::slice::from_raw_parts(xnb.as_ptr() as *const f32, self.hidden) };
                let r = (xnf.iter().map(|v| v * v).sum::<f32>() / self.hidden as f32).sqrt();
                eprintln!("  [hipl] L{il} xn rms={r:.5} xn0={:.6}", xnf[0]);
            }
            if il % 4 == 3 {
                let mut ai = (il / 4) as i32;
                let mut kv = self.kvcap;
                let lq_key = format!("{lp}.self_attn.q_proj");
                let lk_key = format!("{lp}.self_attn.k_proj");
                let lv_key = format!("{lp}.self_attn.v_proj");
                let lo_key = format!("{lp}.self_attn.o_proj");
                let (dxn, dqh, dgq, dgv, dou, dab) =
                    (self.dxn, self.dqh, self.dgq, self.dgv, self.dou, self.dab);
                let lq = HipLin {
                    k: self.lin[&lq_key].k,
                    n: self.lin[&lq_key].n,
                    krate: self.lin[&lq_key].krate,
                    suh: self.lin[&lq_key].suh,
                    tre: self.lin[&lq_key].tre,
                    svh: self.lin[&lq_key].svh,
                };
                let lk = HipLin {
                    k: self.lin[&lk_key].k,
                    n: self.lin[&lk_key].n,
                    krate: self.lin[&lk_key].krate,
                    suh: self.lin[&lk_key].suh,
                    tre: self.lin[&lk_key].tre,
                    svh: self.lin[&lk_key].svh,
                };
                let lv = HipLin {
                    k: self.lin[&lv_key].k,
                    n: self.lin[&lv_key].n,
                    krate: self.lin[&lv_key].krate,
                    suh: self.lin[&lv_key].suh,
                    tre: self.lin[&lv_key].tre,
                    svh: self.lin[&lv_key].svh,
                };
                let lo = HipLin {
                    k: self.lin[&lo_key].k,
                    n: self.lin[&lo_key].n,
                    krate: self.lin[&lo_key].krate,
                    suh: self.lin[&lo_key].suh,
                    tre: self.lin[&lo_key].tre,
                    svh: self.lin[&lo_key].svh,
                };
                self.gemv_chain(&lq, dxn, dqh)?;
                self.gemv_chain(&lk, dxn, dgq)?;
                self.gemv_chain(&lv, dxn, dgv)?;
                let mut tl2 = 1i32;
                let mut p0v = self.pos as i32;
                let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7, mut a8) = (
                    dqh, dgq, dgv, self.dqnw, self.dknw, self.dq2, self.dkc, self.dvc, self.dpp,
                );
                self.hc.launch3(
                    "exl3_attn_prep",
                    1,
                    28,
                    1,
                    128,
                    &mut [
                        &mut a0 as *mut *mut u8 as *mut _,
                        &mut a1 as *mut *mut u8 as *mut _,
                        &mut a2 as *mut *mut u8 as *mut _,
                        &mut a3 as *mut *mut u8 as *mut _,
                        &mut a4 as *mut *mut u8 as *mut _,
                        &mut a5 as *mut *mut u8 as *mut _,
                        &mut a6 as *mut *mut u8 as *mut _,
                        &mut a7 as *mut *mut u8 as *mut _,
                        &mut a8 as *mut *mut u8 as *mut _,
                        &mut tl2 as *mut i32 as *mut _,
                        &mut p0v as *mut i32 as *mut _,
                        &mut ai as *mut i32 as *mut _,
                        &mut kv as *mut i32 as *mut _,
                    ],
                )?;
                let (mut f0, mut f1, mut f2, mut f3, mut f4, mut f5) =
                    (self.dq2, self.dkc, self.dvc, dqh, dou, self.dpp);
                self.hc.launch3(
                    "exl3_attn_fwd3s",
                    1,
                    24,
                    1,
                    256,
                    &mut [
                        &mut f0 as *mut *mut u8 as *mut _,
                        &mut f1 as *mut *mut u8 as *mut _,
                        &mut f2 as *mut *mut u8 as *mut _,
                        &mut f3 as *mut *mut u8 as *mut _,
                        &mut f4 as *mut *mut u8 as *mut _,
                        &mut f5 as *mut *mut u8 as *mut _,
                        &mut tl2 as *mut i32 as *mut _,
                        &mut p0v as *mut i32 as *mut _,
                        &mut ai as *mut i32 as *mut _,
                        &mut kv as *mut i32 as *mut _,
                    ],
                )?;
                if llm170_diag::dump::opts().key("atrace") && il == 3 {
                    let mut hb = vec![0u8; 6144 * 4];
                    let _ = self.hc.d2h(&mut hb, dou);
                    let _ = self.hc.sync();
                    // SAFETY: d2h 완료 후 재해석.
                    self.atrace_dou.push(unsafe {
                        std::slice::from_raw_parts(hb.as_ptr() as *const f32, 6144).to_vec()
                    });
                    // fwd3s 입력 4종 — 이번 콜의 pos(self.pos) 행이 k/v 기록 위치.
                    let mut qb = vec![0u8; 6144 * 4];
                    let _ = self.hc.d2h(&mut qb, self.dq2);
                    let mut kb = vec![0u8; 1024 * 4];
                    // SAFETY: dkc ai(=0) 슬라이스 내 현재 pos 행.
                    let _ = self.hc.d2h(&mut kb, unsafe {
                        self.dkc.add(self.pos as usize * 1024 * 4)
                    });
                    let mut vb = vec![0u8; 1024 * 4];
                    let _ = self.hc.d2h(&mut vb, unsafe {
                        self.dvc.add(self.pos as usize * 1024 * 4)
                    });
                    let mut gb = vec![0u8; 12288 * 4];
                    let _ = self.hc.d2h(&mut gb, dqh);
                    let mut xnb = vec![0u8; 5120 * 4];
                    let _ = self.hc.d2h(&mut xnb, self.dxn);
                    let _ = self.hc.sync();
                    // SAFETY: d2h 완료 후 재해석.
                    unsafe {
                        self.atrace_qh.push(
                            std::slice::from_raw_parts(qb.as_ptr() as *const f32, 6144).to_vec(),
                        );
                        self.atrace_k.push(
                            std::slice::from_raw_parts(kb.as_ptr() as *const f32, 1024).to_vec(),
                        );
                        self.atrace_v.push(
                            std::slice::from_raw_parts(vb.as_ptr() as *const f32, 1024).to_vec(),
                        );
                        self.atrace_g.push(
                            std::slice::from_raw_parts(gb.as_ptr() as *const f32, 12288).to_vec(),
                        );
                        self.atrace_xn.push(
                            std::slice::from_raw_parts(xnb.as_ptr() as *const f32, 5120).to_vec(),
                        );
                    }
                }
                self.gemv_chain(&lo, dou, dab)?;
                if llm170_diag::dump::opts().key("atrace") && il == 3 {
                    let mut hb = vec![0u8; self.hidden * 4];
                    let _ = self.hc.d2h(&mut hb, self.dab);
                    let _ = self.hc.sync();
                    // SAFETY: d2h 완료 후 재해석.
                    self.atrace_dab.push(unsafe {
                        std::slice::from_raw_parts(hb.as_ptr() as *const f32, self.hidden).to_vec()
                    });
                }
            } else {
                let lq_key = format!("{lp}.linear_attn.in_proj_qkv");
                let lz_key = format!("{lp}.linear_attn.in_proj_z");
                let lo_key = format!("{lp}.linear_attn.out_proj");
                let lq = HipLin {
                    k: self.lin[&lq_key].k,
                    n: self.lin[&lq_key].n,
                    krate: self.lin[&lq_key].krate,
                    suh: self.lin[&lq_key].suh,
                    tre: self.lin[&lq_key].tre,
                    svh: self.lin[&lq_key].svh,
                };
                let lz = HipLin {
                    k: self.lin[&lz_key].k,
                    n: self.lin[&lz_key].n,
                    krate: self.lin[&lz_key].krate,
                    suh: self.lin[&lz_key].suh,
                    tre: self.lin[&lz_key].tre,
                    svh: self.lin[&lz_key].svh,
                };
                let lo = HipLin {
                    k: self.lin[&lo_key].k,
                    n: self.lin[&lo_key].n,
                    krate: self.lin[&lo_key].krate,
                    suh: self.lin[&lo_key].suh,
                    tre: self.lin[&lo_key].tre,
                    svh: self.lin[&lo_key].svh,
                };
                let (dxn, dqkv, dzv) = (self.dxn, self.dqkv, self.dzv);
                self.gemv_chain(&lq, dxn, dqkv)?;
                self.gemv_chain(&lz, dxn, dzv)?;
                let mut tl = 1i32;
                let mut lay = gdn_il as i32;
                let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5) = (
                    self.dqkv, self.dcw, self.dring, self.dgq, self.dgk, self.dgv,
                );
                self.hc.launch(
                    "exl3_gdn_conv",
                    80,
                    1,
                    128,
                    &mut [
                        &mut a0 as *mut *mut u8 as *mut _,
                        &mut a1 as *mut *mut u8 as *mut _,
                        &mut a2 as *mut *mut u8 as *mut _,
                        &mut a3 as *mut *mut u8 as *mut _,
                        &mut a4 as *mut *mut u8 as *mut _,
                        &mut a5 as *mut *mut u8 as *mut _,
                        &mut tl as *mut i32 as *mut _,
                        &mut lay as *mut i32 as *mut _,
                    ],
                )?;

                let (mut hk16, mut hv48, mut dd128) = (16i32, 48i32, 128i32);
                let (
                    mut b0,
                    mut b1,
                    mut b2,
                    mut b3,
                    mut b4,
                    mut b5,
                    mut b6,
                    mut b7,
                    mut b8,
                    mut b9,
                    mut bb,
                ) = (
                    self.dgq, self.dgk, self.dgv, self.dxn, self.dab_c, self.dal, self.ddt,
                    self.dq2, self.dk2, self.dv2, self.dbg,
                );
                eprintln!(
                    "  [l2dbg] L{il} q={:p} k={:p} v={:p} xn={:p} ab={:p} al={:p} dt={:p} qo={:p} ko={:p} vo={:p}",
                    self.dgq,
                    self.dgk,
                    self.dgv,
                    self.dxn,
                    self.dab_c,
                    self.dal,
                    self.ddt,
                    self.dq2,
                    self.dk2,
                    self.dv2
                );
                self.hc.launch3(
                    "exl3_gdn_l2perm",
                    48,
                    1,
                    1,
                    128,
                    &mut [
                        &mut b0 as *mut *mut u8 as *mut _,
                        &mut b1 as *mut *mut u8 as *mut _,
                        &mut b2 as *mut *mut u8 as *mut _,
                        &mut b3 as *mut *mut u8 as *mut _,
                        &mut b4 as *mut *mut u8 as *mut _,
                        &mut b5 as *mut *mut u8 as *mut _,
                        &mut b6 as *mut *mut u8 as *mut _,
                        &mut b7 as *mut *mut u8 as *mut _,
                        &mut b8 as *mut *mut u8 as *mut _,
                        &mut b9 as *mut *mut u8 as *mut _,
                        &mut bb as *mut *mut u8 as *mut _,
                        &mut tl as *mut i32 as *mut _,
                        &mut lay as *mut i32 as *mut _,
                    ],
                )?;
                let (mut c0, mut c1, mut c2, mut c3, mut c4, mut c5) =
                    (self.dq2, self.dk2, self.dv2, self.dbg, self.dgst, self.dgo);
                self.hc.launch3(
                    "exl3_gdn_scan",
                    48,
                    1,
                    1,
                    128,
                    &mut [
                        &mut c0 as *mut *mut u8 as *mut _,
                        &mut c1 as *mut *mut u8 as *mut _,
                        &mut c2 as *mut *mut u8 as *mut _,
                        &mut c3 as *mut *mut u8 as *mut _,
                        &mut c4 as *mut *mut u8 as *mut _,
                        &mut c5 as *mut *mut u8 as *mut _,
                        &mut tl as *mut i32 as *mut _,
                        &mut hk16 as *mut i32 as *mut _,
                        &mut hv48 as *mut i32 as *mut _,
                        &mut dd128 as *mut i32 as *mut _,
                        &mut lay as *mut i32 as *mut _,
                    ],
                )?;
                let (mut e0, mut e1, mut e2, mut e3) = (self.dgo, self.dzv, self.dnw_g, self.dgate);
                self.hc.launch3(
                    "exl3_gdn_gate",
                    48,
                    1,
                    1,
                    128,
                    &mut [
                        &mut e0 as *mut *mut u8 as *mut _,
                        &mut e1 as *mut *mut u8 as *mut _,
                        &mut e2 as *mut *mut u8 as *mut _,
                        &mut e3 as *mut *mut u8 as *mut _,
                        &mut tl as *mut i32 as *mut _,
                        &mut lay as *mut i32 as *mut _,
                    ],
                )?;
                let dgate = self.dgate;
                self.gemv_chain(&lo, dgate, self.dab)?;
            }
            self.norm(2 * il + 1, self.dab)?;
            {
                let mut xb = vec![0u8; self.hidden * 4];
                self.hc.d2h(&mut xb, self.dx)?;
                // SAFETY: d2h 완료 후 재해석.
                let xf: &[f32] =
                    unsafe { std::slice::from_raw_parts(xb.as_ptr() as *const f32, self.hidden) };
                let r = (xf.iter().map(|v| v * v).sum::<f32>() / self.hidden as f32).sqrt();
                eprintln!("  [hipl] L{il} post-attn rms={r:.5}");
                if il == 1 {
                    let mut gb2 = vec![0u8; 8];
                    self.hc.d2h(&mut gb2, self.dgate)?;
                    // SAFETY: d2h 완료 후 재해석.
                    let gg: &[f32] =
                        unsafe { std::slice::from_raw_parts(gb2.as_ptr() as *const f32, 2) };
                    eprintln!("  [hipl] L1 gated[0..2]={gg:?}");
                }
            }
            let lg_key = format!("{lp}.mlp.gate_proj");
            let lu_key = format!("{lp}.mlp.up_proj");
            let ld_key = format!("{lp}.mlp.down_proj");
            let lg = HipLin {
                k: self.lin[&lg_key].k,
                n: self.lin[&lg_key].n,
                krate: self.lin[&lg_key].krate,
                suh: self.lin[&lg_key].suh,
                tre: self.lin[&lg_key].tre,
                svh: self.lin[&lg_key].svh,
            };
            let lu = HipLin {
                k: self.lin[&lu_key].k,
                n: self.lin[&lu_key].n,
                krate: self.lin[&lu_key].krate,
                suh: self.lin[&lu_key].suh,
                tre: self.lin[&lu_key].tre,
                svh: self.lin[&lu_key].svh,
            };
            let ld = HipLin {
                k: self.lin[&ld_key].k,
                n: self.lin[&ld_key].n,
                krate: self.lin[&ld_key].krate,
                suh: self.lin[&ld_key].suh,
                tre: self.lin[&ld_key].tre,
                svh: self.lin[&ld_key].svh,
            };
            let (dxn, dqh, dgo, dew, dab) = (self.dxn, self.dqh, self.dgo, self.dew, self.dab);
            self.gemv_chain(&lg, dxn, dqh)?;
            self.gemv_chain(&lu, dxn, dgo)?;
            let mut ewn = lg.n as i32;
            let (mut w0, mut w1, mut w2) = (dqh, dgo, dew);
            self.hc.launch3(
                "exl3_ew",
                (lg.n.div_ceil(128)) as u32,
                1,
                1,
                128,
                &mut [
                    &mut w0 as *mut *mut u8 as *mut _,
                    &mut w1 as *mut *mut u8 as *mut _,
                    &mut w2 as *mut *mut u8 as *mut _,
                    &mut ewn as *mut i32 as *mut _,
                ],
            )?;
            self.gemv_chain(&ld, dew, dab)?;
            ab = dab;
            if self.dbg_hcurve && [0usize, 1, 8, 32, 63].contains(&il) {
                let mut cb = vec![0u8; self.hidden * 4];
                let _ = self.hc.d2h(&mut cb, self.dx);
                let _ = self.hc.sync();
                // SAFETY: d2h 완료 후 재해석.
                let cv = unsafe {
                    std::slice::from_raw_parts(cb.as_ptr() as *const f32, self.hidden).to_vec()
                };
                self.hcurve.push((il, cv));
            }

            {
                let mut xb = vec![0u8; self.hidden * 4];
                self.hc.d2h(&mut xb, self.dx)?;
                // SAFETY: d2h 완료 후 재해석.
                let xf: &[f32] =
                    unsafe { std::slice::from_raw_parts(xb.as_ptr() as *const f32, self.hidden) };
                let r = (xf.iter().map(|v| v * v).sum::<f32>() / self.hidden as f32).sqrt();
                eprintln!("  [hipl] L{il} post-ffn rms={r:.5}");
                if il == 0 {
                    let mut fb = vec![0u8; 8];
                    self.hc.d2h(&mut fb, self.dab)?;
                    // SAFETY: d2h 완료 후 재해석.
                    let ff: &[f32] =
                        unsafe { std::slice::from_raw_parts(fb.as_ptr() as *const f32, 2) };
                    eprintln!("  [hipl] L0 down[0..2]={ff:?}");
                }
            }
        }
        let mut hb = vec![0u8; self.hidden * 4];
        self.hc.d2h(&mut hb, self.dx)?;
        self.norm(128, self.dab)?;
        self.pos += 1;
        self.hc.h2d(self.dpp, &self.pos.to_le_bytes())?;
        let lh_key = "lm_head".to_string();
        let llh = HipLin {
            k: self.lin[&lh_key].k,
            n: self.lin[&lh_key].n,
            krate: self.lin[&lh_key].krate,
            suh: self.lin[&lh_key].suh,
            tre: self.lin[&lh_key].tre,
            svh: self.lin[&lh_key].svh,
        };
        self.gemv_chain(&llh, self.dxn, self.dyb)?;
        let mut lb = vec![0u8; llh.n * 4];
        self.hc.d2h(&mut lb, self.dyb)?;
        // SAFETY: d2h 완료 후 재해석.
        let logits =
            unsafe { std::slice::from_raw_parts(lb.as_ptr() as *const f32, llh.n).to_vec() };
        let hidden =
            unsafe { std::slice::from_raw_parts(hb.as_ptr() as *const f32, self.hidden).to_vec() };
        Ok((logits, hidden))
    }
    // 마커 mod1
    // 마커 ll
    // 마커 l2d
    // 마커 chk
    // 마커 hid
    // 마커 rms1
    // 마커 dcmp
    // 마커 l1g
    // 마커 xn1
    // 마커 xn2
    // 마커 lx
    // 마커 rw2
    // 마커 abfix
    // 마커 posr
    // 마커 final
    // 마커 ew1
    // 마커 sy1
    // 마커 sr1
    // 마커 fx649
    // 마커 dropfx
    // 마커 syn2
    // 마커 am1
    // 마커 mtp1
    // 마커 mtp2
    // 마커 mtp3
    // 마커 mtp4
    // 마커 mtph
    // 마커 mtpi
    // 마커 cl3
    // 마커 fb1
    // 마커 fb2
    // 마커 fb3
    // 마커 fb5
    // 마커 fb6
    // 마커 fbfx
    // 마커 ho1
    // 마커 ho2
    // 마커 dpp
    // 마커 bld
    // 마커 blf
    // 마커 bd2
    // 마커 gy2
    // 마커 dcl
    // 마커 mr2
    // 마커 mr3
    // 마커 dg2
    // 마커 dg3
    // 마커 dg4
    // 마커 npf
    // 마커 amf
    // 마커 nwp
    // 마커 d3p
    // 마커 d5p
    // 마커 npz
    // 마커 kvh
    // 마커 lhf
    // 마커 ks2
    // 마커 ho3
    // 마커 gcap
    // 마커 gcr
    // 마커 gcr2
    // 마커 gm2
    // 마커 pin1
    // 마커 fx25
    // 마커 gdb
    // 마커 gdb2
    // 마커 dpf
    // 마커 ks16
    // 마커 rb1
    // 마커 hpt
    // 마커 hpf
    // 마커 pga
    // 마커 hcv
    // 마커 hcf
    // 마커 rs1
}
