//! EXL3 hip 배치 경로 — dump8/norm_ptr/gemv_host/forward_batch/batch_core/캡처·재생/llh_n/스펙 배치(MTP 검증 포함).
use super::{Exl3HipDecoder, HipLin};

impl Exl3HipDecoder {
    /// 검증 덤프: 버퍼 선두 8 f32(동기 포함 — 디버그 전용).
    pub(super) fn dump8_dev(&mut self, ptr: *mut u8) -> Vec<f32> {
        let mut rb = vec![0u8; 32];
        let _ = self.hc.d2h(&mut rb, ptr);
        let _ = self.hc.sync();
        // SAFETY: d2h 완료 후 재해석.
        unsafe { std::slice::from_raw_parts(rb.as_ptr() as *const f32, 8).to_vec() }
    }

    /// 순수+잔차 노름(norm_resid_p 직접 포인터) — x += ab, xn = norm(x)·nw. 커널 인자 순서.
    pub(super) fn norm_ptr(
        &mut self,
        x: *mut u8,
        nw: *mut u8,
        ab: *mut u8,
        xn: *mut u8,
        t_len: usize,
    ) -> Result<(), String> {
        let mut tl = t_len as i32;
        let (mut a0, mut a1, mut a2, mut a3) = (x, nw, ab, xn);
        self.hc.launch(
            "exl3_norm_resid_p",
            t_len as u32,
            1,
            1024,
            &mut [
                &mut a0 as *mut *mut u8 as *mut _,
                &mut a1 as *mut *mut u8 as *mut _,
                &mut a2 as *mut *mut u8 as *mut _,
                &mut a3 as *mut *mut u8 as *mut _,
                &mut tl as *mut i32 as *mut _,
            ],
        )?;
        Ok(())
    }

    /// 호스트 벡터 → gemv 체인 1회 → 호스트 결과(드래프트 소형 전용 — 본체는 상주 경로).
    pub(super) fn gemv_host(&mut self, key: &str, x: &[f32]) -> Result<Vec<f32>, String> {
        let l = self.lin.get(key).ok_or(format!("lin {key}"))?;
        let l = HipLin {
            k: l.k,
            n: l.n,
            krate: l.krate,
            suh: l.suh,
            tre: l.tre,
            svh: l.svh,
        };
        self.hc.h2d(self.dmtpin, unsafe {
            std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4)
        })?;
        self.hc.sync()?;
        let out = self.dyb;
        self.gemv_chain(&l, self.dmtpin, out)?;
        let mut ob = vec![0u8; l.n * 4];
        self.hc.d2h(&mut ob, out)?;
        self.hc.sync()?;
        // SAFETY: d2h 완료 후 재해석.
        let y: Vec<f32> =
            unsafe { std::slice::from_raw_parts(ob.as_ptr() as *const f32, l.n).to_vec() };
        Ok(y)
    }

    /// 배치 forward(프리필·MTP 검증 공용) — rows: [T][hidden] 임베딩 행.
    /// GDN·FFN·선형은 gemm2 배치, 어텐션은 행별 prep/fwd3s 루프(소형-T 전용).
    /// 상태: dring/dgst/dkc/dvc는 pos..pos+T-1 순차 기록(디코드와 동일 규약).
    pub fn forward_batch(
        &mut self,
        rows: &[Vec<f32>],
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>), String> {
        let t = rows.len();
        if t == 0 || t > 64 {
            return Err(format!("forward_batch: T={t} 범위 외(1..64)"));
        }
        // KV 상한 가드(plans/128 P0) — 초과 시 폴트 대신 Err로 우아한 거절.
        if self.pos as usize + t > self.kvcap as usize {
            return Err(format!(
                "context overflow: pos={} + T={} > kvcap={} (--ctx 상향 필요)",
                self.pos, t, self.kvcap
            ));
        }
        let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        // SAFETY: pstage 64×hidden 상한 내 — 핀 쓰기.
        unsafe {
            std::ptr::copy_nonoverlapping(flat.as_ptr() as *const u8, self.pstage, flat.len() * 4);
        }
        self.batch_core(t)?;
        self.hc.sync()?; // 판독 배리어 — 캡처 코어는 비동기, g_out 확정 대기
        self.pos += t as u32;
        // SAFETY: g_out 재해석.
        let n = self.llh_n();
        let out: Vec<Vec<f32>> = (0..t)
            .map(|r| {
                // SAFETY: pgout 행 오프셋.
                let b = unsafe {
                    std::slice::from_raw_parts(self.pgout.add(r * n * 4) as *const f32, n)
                };
                b.to_vec()
            })
            .collect();
        // SAFETY: pgh 재해석.
        let last_h =
            unsafe { std::slice::from_raw_parts(self.pgh as *const f32, self.hidden).to_vec() };
        Ok((out, last_h))
    }

    /// 토큰 id 직행 배치 프리필(plans/130 A3) — 임베딩 행 d2h 판독+h2d 재업로드
    /// 대신 디바이스 gather. 반환 계약은 forward_batch와 동일.
    pub fn forward_batch_toks(
        &mut self,
        toks: &[u32],
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>), String> {
        let t = toks.len();
        if t == 0 || t > 64 {
            return Err(format!("forward_batch_toks: T={t} 범위 외(1..64)"));
        }
        if self.pos as usize + t > self.kvcap as usize {
            return Err(format!(
                "context overflow: pos={} + T={} > kvcap={} (--ctx 상향 필요)",
                self.pos, t, self.kvcap
            ));
        }
        self.embed_gather(toks, self.dbx)?;
        self.batch_core_opt(t, false)?;
        self.hc.sync()?; // 판독 배리어 — 캡처 코어는 비동기, g_out 확정 대기
        self.pos += t as u32;
        // SAFETY: g_out 재해석.
        let n = self.llh_n();
        let out: Vec<Vec<f32>> = (0..t)
            .map(|r| {
                // SAFETY: pgout 행 오프셋.
                let b = unsafe {
                    std::slice::from_raw_parts(self.pgout.add(r * n * 4) as *const f32, n)
                };
                b.to_vec()
            })
            .collect();
        // SAFETY: pgh 재해석.
        let last_h =
            unsafe { std::slice::from_raw_parts(self.pgh as *const f32, self.hidden).to_vec() };
        Ok((out, last_h))
    }

    /// 그래프 코어 — 모든 입출력이 고정 포인터(stage_rows/g_out/g_h/dpos).
    /// 캡처·재생·비캡처 공용(캡처 호환: 내부 sync/h2d-from-stack 없음).
    pub(super) fn batch_core(&mut self, t: usize) -> Result<(), String> {
        self.batch_core_opt(t, true)
    }

    /// staged=false: dbx가 임베딩 gather 등으로 이미 기록된 경로(plans/130 A3).
    pub(super) fn batch_core_opt(&mut self, t: usize, staged: bool) -> Result<(), String> {
        let n_layers = self.loaded_layers.min(self.n_layers);
        // 캡처 호환 업로드 — 원시 핀 h2d(h2d는 내부 sync 포함 — 캡처 무효화).
        if staged {
            self.hc
                .h2d_nosync(self.dbx, self.pstage, t * self.hidden * 4)?;
        }
        self.hc.d2d(self.dpp, self.dpos, 4)?;
        let mut ab = self.dbzero;
        for il in 0..n_layers {
            let lp = format!("model.language_model.layers.{il}");
            let gdn_il = (0..il).filter(|i| i % 4 != 3).count();
            self.norm_p(2 * il, ab, t)?;
            if llm170_diag::dump::opts().key("htrace") {
                // 프로브 전용 덤프(d2h+sync — 그래프 캡처와 양립 불가, plans/128 P1 선행).
                let mut rows_v = Vec::with_capacity(t);
                for r in 0..t {
                    let mut hb = vec![0u8; self.hidden * 4];
                    // SAFETY: dbx 내 행 오프셋 — t≤64 경계 내.
                    let p = unsafe { self.dbx.add(r * self.hidden * 4) };
                    let _ = self.hc.d2h(&mut hb, p);
                    let _ = self.hc.sync();
                    // SAFETY: d2h 완료 후 재해석.
                    rows_v.push(unsafe {
                        std::slice::from_raw_parts(hb.as_ptr() as *const f32, self.hidden).to_vec()
                    });
                }
                self.htrace.push(rows_v); // 플랫: [il][row] — 배치 호출 1회 가정
            }
            // [C 계기 2026-10-04] hstage — 순차 경로의 [hipl] L{il} xn 덤프와
            // 배치 경로의 값을 직접 대조해 첫 발산 층 경계를 확정한다.
            if llm170_diag::dump::opts().key("hstage") && il <= 1 {
                let mut xnb = vec![0u8; self.hidden * 4];
                let _ = self.hc.d2h(&mut xnb, self.dbxn);
                let _ = self.hc.sync();
                // SAFETY: d2h 완료 후 재해석 — 행0(5120원소).
                let xnf: &[f32] =
                    unsafe { std::slice::from_raw_parts(xnb.as_ptr() as *const f32, self.hidden) };
                let r = (xnf.iter().map(|v| v * v).sum::<f32>() / self.hidden as f32).sqrt();
                let mut xb2 = vec![0u8; self.hidden * 4];
                let _ = self.hc.d2h(&mut xb2, self.dbx);
                let _ = self.hc.sync();
                // SAFETY: d2h 완료 후 재해석 — 행0 잔차.
                let xf2: &[f32] =
                    unsafe { std::slice::from_raw_parts(xb2.as_ptr() as *const f32, self.hidden) };
                let r2 = (xf2.iter().map(|v| v * v).sum::<f32>() / self.hidden as f32).sqrt();
                eprintln!(
                    "  [hstb] L{il} 진입 x rms={r2:.5} x0={:.6} · xn rms={r:.5} xn0={:.6}",
                    xf2[0], xnf[0]
                );
            }
            if il % 4 == 3 {
                // 어텐션층 — q/k/v gemm2 후 행별 prep+fwd3s
                let lq = self.lin[&format!("{lp}.self_attn.q_proj")].clone_shallow();
                let lk = self.lin[&format!("{lp}.self_attn.k_proj")].clone_shallow();
                let lv = self.lin[&format!("{lp}.self_attn.v_proj")].clone_shallow();
                let lo = self.lin[&format!("{lp}.self_attn.o_proj")].clone_shallow();
                // [결함 판정 2026-10-05 P1] gemm2 계약상 입력은 had16의 dah16인데
                // 이 분기의 had16 호출이 누락돼 q/k/v가 직전 GDN층의 잔여 dah16
                // (전혀 다른 벡터)로 계산됐다 — L3(첫 어텐션층)부터 계통 발산
                // (htrace L4 진입 9.06 · fwd3s 입력 4종 전부 11~14 차이·플립 2/9,
                // hip-batch·atrace로 국소화). suh는 선형별(입력채널 스케일)이라
                // 세 had16 각각 자기 suh로 호출 — GDN 분기·MTP 경로와 동일 패턴.
                self.had16_batch(self.dbxn, lq.k, t, lq.suh)?;
                self.gemm2_batch(&lq, t, self.dsb)?;
                self.had16_batch(self.dbxn, lk.k, t, lk.suh)?;
                self.gemm2_batch(&lk, t, self.dsb2)?;
                self.had16_batch(self.dbxn, lv.k, t, lv.suh)?;
                self.gemm2_batch(&lv, t, self.dsb3)?;
                let mut ai = (il / 4) as i32;
                let mut kv = self.kvcap;
                // 층 진입 시 pp를 청크 기준 pos로 리셋 후 **단일 t-런치**(2026-10-05
                // P1 배치화 — hip-batch·atrace 국소화의 종결). 커널은 이미 전부
                // t(blockIdx.x) 인덱싱·pos=pp[0]+t·lim=pp[0]+t+1 산출이라 행별
                // (1,·) 루프는 하네스 중복이었고, 그 행별 구조가 순차 대비
                // 계통 편차(fwd3s 출력 maxdiff ~1.5)의 담체였다 — 검증된 attn
                // 프로브의 일괄 런치 형태와 동일 구조로 교체. 행별 pos_bump도
                // 불필요(층마다 리셋) — 이전 층별 pp 누적 결함의 잔재도 함께 제거.
                self.hc.d2d(self.dpp, self.dpos, 4)?;
                let mut tl2 = t as i32;
                let mut p0v = self.pos as i32;
                let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7, mut a8) = (
                    self.dsb, self.dsb2, self.dsb3, self.dqnw, self.dknw, self.dq2, self.dkc,
                    self.dvc, self.dpp,
                );
                self.hc.launch3(
                    "exl3_attn_prep",
                    t as u32,
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
                    (self.dq2, self.dkc, self.dvc, self.dsb, self.dou, self.dpp);
                self.hc.launch3(
                    "exl3_attn_fwd3s",
                    t as u32,
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
                    for r in 0..t {
                        let mut hb = vec![0u8; 6144 * 4];
                        // SAFETY: dou 내 행 오프셋.
                        let p = unsafe { self.dou.add(r * 6144 * 4) };
                        let _ = self.hc.d2h(&mut hb, p);
                        let _ = self.hc.sync();
                        // SAFETY: d2h 완료 후 재해석.
                        self.atrace_dou.push(unsafe {
                            std::slice::from_raw_parts(hb.as_ptr() as *const f32, 6144).to_vec()
                        });
                        // fwd3s 입력 4종(plans/128 P1 — 발산 입력 판별):
                        // qh(dq2 행)·k/vc(슬라이스 ai행)·게이트(dsb 행)
                        let mut xb = vec![0u8; 5120 * 4];
                        // SAFETY: dbxn 행 r — 노름 출력(점근 입력).
                        let _ = self.hc.d2h(&mut xb, unsafe { self.dbxn.add(r * 5120 * 4) });
                        let mut qb = vec![0u8; 6144 * 4];
                        let _ = self.hc.d2h(&mut qb, unsafe { self.dq2.add(r * 6144 * 4) });
                        let mut kb = vec![0u8; 1024 * 4];
                        // SAFETY: dkc 내 ai(=il/4=0) 슬라이스 행 r.
                        let _ = self.hc.d2h(&mut kb, unsafe { self.dkc.add(r * 1024 * 4) });
                        let mut vb = vec![0u8; 1024 * 4];
                        let _ = self.hc.d2h(&mut vb, unsafe { self.dvc.add(r * 1024 * 4) });
                        let mut gb = vec![0u8; 12288 * 4];
                        // SAFETY: dsb 내 행 오프셋.
                        let _ = self.hc.d2h(&mut gb, unsafe { self.dsb.add(r * 12288 * 4) });
                        let _ = self.hc.sync();
                        // SAFETY: d2h 완료 후 재해석.
                        unsafe {
                            self.atrace_qh.push(
                                std::slice::from_raw_parts(qb.as_ptr() as *const f32, 6144)
                                    .to_vec(),
                            );
                            self.atrace_k.push(
                                std::slice::from_raw_parts(kb.as_ptr() as *const f32, 1024)
                                    .to_vec(),
                            );
                            self.atrace_v.push(
                                std::slice::from_raw_parts(vb.as_ptr() as *const f32, 1024)
                                    .to_vec(),
                            );
                            self.atrace_g.push(
                                std::slice::from_raw_parts(gb.as_ptr() as *const f32, 12288)
                                    .to_vec(),
                            );
                            self.atrace_xn.push(
                                std::slice::from_raw_parts(xb.as_ptr() as *const f32, 5120)
                                    .to_vec(),
                            );
                        }
                    }
                }
                // o_proj gemm2 — 입력 dou [T][6144]
                self.had16_batch(self.dou, 6144, t, lo.suh)?;
                self.gemm2_batch(&lo, t, self.dbab)?;
                if llm170_diag::dump::opts().key("atrace") && il == 3 {
                    for r in 0..t {
                        let mut hb = vec![0u8; self.hidden * 4];
                        // SAFETY: dbab 내 행 오프셋.
                        let p = unsafe { self.dbab.add(r * self.hidden * 4) };
                        let _ = self.hc.d2h(&mut hb, p);
                        let _ = self.hc.sync();
                        // SAFETY: d2h 완료 후 재해석.
                        self.atrace_dab.push(unsafe {
                            std::slice::from_raw_parts(hb.as_ptr() as *const f32, self.hidden)
                                .to_vec()
                        });
                    }
                }
            } else {
                let lq = self.lin[&format!("{lp}.linear_attn.in_proj_qkv")].clone_shallow();
                let lz = self.lin[&format!("{lp}.linear_attn.in_proj_z")].clone_shallow();
                let lo = self.lin[&format!("{lp}.linear_attn.out_proj")].clone_shallow();
                self.had16_batch(self.dbxn, lq.k, t, lq.suh)?;
                self.gemm2_batch(&lq, t, self.dsb)?;
                self.had16_batch(self.dbxn, lz.k, t, lz.suh)?;
                self.gemm2_batch(&lz, t, self.dsb2)?;

                let mut tl = t as i32;
                let mut lay = gdn_il as i32;
                let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5) =
                    (self.dsb, self.dcw, self.dring, self.dgq, self.dgk, self.dgv);
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
                    self.dgq, self.dgk, self.dgv, self.dbxn, self.dab_c, self.dal, self.ddt,
                    self.dq2, self.dk2, self.dv2, self.dbg,
                );
                self.hc.launch3(
                    "exl3_gdn_l2perm",
                    48,
                    t as u32,
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

                let (mut hk16, mut hv48, mut dd128) = (16i32, 48i32, 128i32);
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
                let (mut e0, mut e1, mut e2, mut e3) =
                    (self.dgo, self.dsb2, self.dnw_g, self.dgate);
                self.hc.launch3(
                    "exl3_gdn_gate",
                    48,
                    t as u32,
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

                self.had16_batch(self.dgate, lo.k, t, lo.suh)?;
                self.gemm2_batch(&lo, t, self.dbab)?;
            }
            self.norm_p(2 * il + 1, self.dbab, t)?;
            let lg = self.lin[&format!("{lp}.mlp.gate_proj")].clone_shallow();
            let lu = self.lin[&format!("{lp}.mlp.up_proj")].clone_shallow();
            let ld = self.lin[&format!("{lp}.mlp.down_proj")].clone_shallow();
            self.had16_batch(self.dbxn, lg.k, t, lg.suh)?;
            self.gemm2_batch(&lg, t, self.dsb)?;
            self.had16_batch(self.dbxn, lu.k, t, lu.suh)?;
            self.gemm2_batch(&lu, t, self.dsb2)?;
            let mut ewn = (t * lg.n) as i32;
            let (mut w0, mut w1, mut w2) = (self.dsb, self.dsb2, self.dew);
            self.hc.launch3(
                "exl3_ew",
                (t * lg.n).div_ceil(128) as u32,
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
            self.had16_batch(self.dew, ld.k, t, ld.suh)?;
            self.gemm2_batch(&ld, t, self.dbab)?;
            ab = self.dbab;
            if self.dbg_hcurve && [0usize, 1, 8, 32, 63].contains(&il) {
                let mut cb = vec![0u8; self.hidden * 4];
                let _ = self.hc.d2h(&mut cb, self.dbx);
                let _ = self.hc.sync();
                // SAFETY: d2h 완료 후 재해석.
                let cv = unsafe {
                    std::slice::from_raw_parts(cb.as_ptr() as *const f32, self.hidden).to_vec()
                };
                self.hcurve.push((il, cv));
            }
        }
        // last_h 캡처 — 최종 노름(마지막 FFN 합산) 전 잔차(vk 12/12·a1 0.625가
        // 측정된 규약 = h_seq와 동일 시점. 결함 11호 과교정 정정: '전'이되 dbx).
        {
            // SAFETY: 전 행 pre-norm dbx → pgall(캡처 호환, 핀).
            for r in 0..t {
                let prow = unsafe { self.dbx.add(r * self.hidden * 4) };
                let dst = unsafe { self.pgall.add(r * self.hidden * 4) };
                self.hc.d2h_pin_async(dst, prow, self.hidden * 4)?;
            }
            // SAFETY: 마지막 행 → g_h(호환 유지).
            let plast = unsafe { self.dbx.add((t - 1) * self.hidden * 4) };
            self.hc.d2h_pin_async(self.pgh, plast, self.hidden * 4)?;
        }
        // 최종 노름 + lm_head 행별 로짓
        self.norm_p(128, self.dbab, t)?;
        let llh = self.lin["lm_head"].clone_shallow();
        self.had16_batch(self.dbxn, llh.k, t, llh.suh)?;
        self.gemm2_batch(&llh, t, self.dsb)?;
        // dpos += t — 디바이스 pos 전진(그래프 재생 시 다음 라운드 위치).
        for _ in 0..t {
            let mut pb1 = self.dpos;
            self.hc.launch3(
                "exl3_pos_bump",
                1,
                1,
                1,
                32,
                &mut [&mut pb1 as *mut *mut u8 as *mut _],
            )?;
        }
        // 출력 d2h — 고정 호스트 버퍼(캡처 노드).
        for r in 0..t {
            // SAFETY: dsb 내 lm_head 행 오프셋.
            let rowp = unsafe { self.dsb.add(r * llh.n * 4) };
            // SAFETY: pgout 행 오프셋 — 원시 핀 d2h(캡처 호환).
            let dstp = unsafe { self.pgout.add(r * llh.n * 4) };
            self.hc.d2h_pin_async(dstp, rowp, llh.n * 4)?;
        }
        Ok(())
    }

    /// 배치 코어를 hipGraph로 캡처(런치 오버헤드 제거 — plans/121 hip 스케줄링).
    /// 워밍 1회 후 캡처: 모든 입출력이 고정 포인터라 재생은 현재 내용을 읽는다.
    pub fn capture_batch(&mut self, t: usize) -> Result<(), String> {
        use crate::rawhip::ctx::hipgraph as hg;
        // 워밍(커널 자원 초기화 완료 후 캡처)
        self.batch_core(t)?;
        self.hc.sync()?;
        unsafe {
            let st = hg::hipStreamBeginCapture(self.hc.stream as *mut _, 2);
            eprintln!("  [gcap] BeginCapture={st}");
            if st != 0 {
                return Err(format!("BeginCapture {st}"));
            }
            let core = self.batch_core(t);
            let mut graph: hg::Graph = std::ptr::null_mut();
            let en = hg::hipStreamEndCapture(self.hc.stream as *mut _, &mut graph);
            eprintln!("  [gcap] EndCapture={en} graph={graph:?} core={core:?}");
            core?;
            if en != 0 {
                return Err(format!("EndCapture {en}"));
            }
            let mut exec: hg::GraphExec = std::ptr::null_mut();
            let ie = hg::hipGraphInstantiate(&mut exec, graph, 0);
            hg::hipGraphDestroy(graph);
            if ie != 0 {
                return Err(format!("Instantiate {ie}"));
            }
            self.gexec = Some((t, exec));
        }
        Ok(())
    }

    /// 그래프 재생 — stage_rows를 채우고 launch. 반환 = (로짓 행들, last_h).
    pub fn replay_batch(&mut self, rows: &[Vec<f32>]) -> Result<(Vec<Vec<f32>>, Vec<f32>), String> {
        let (t, exec) = self.gexec.ok_or("그래프 미캡처 — capture_batch 먼저")?;
        if rows.len() != t {
            return Err(format!("캡처 T={t}와 불일치 rows={}", rows.len()));
        }
        let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        // SAFETY: pstage 핀 쓰기.
        unsafe {
            std::ptr::copy_nonoverlapping(flat.as_ptr() as *const u8, self.pstage, flat.len() * 4);
        }
        use crate::rawhip::ctx::hipgraph as hg;
        unsafe {
            let le = hg::hipGraphLaunch(exec, self.hc.stream as *mut _);
            if le != 0 {
                return Err(format!("GraphLaunch {le}"));
            }
        }
        self.hc.sync()?;
        self.pos += t as u32;
        let n = self.llh_n();
        // SAFETY: 재생 완료 후 g_out/g_h 재해석.
        let out: Vec<Vec<f32>> = (0..t)
            .map(|r| {
                // SAFETY: pgout 행 오프셋.
                let b = unsafe {
                    std::slice::from_raw_parts(self.pgout.add(r * n * 4) as *const f32, n)
                };
                b.to_vec()
            })
            .collect();
        let last_h =
            unsafe { std::slice::from_raw_parts(self.pgh as *const f32, self.hidden).to_vec() };
        Ok((out, last_h))
    }

    /// lm_head 열수(래퍼 디코딩용).
    pub(super) fn llh_n(&self) -> usize {
        self.lin["lm_head"].n
    }

    /// forward_batch + MTP KV 적립 훅(vk 패턴): 각 행의 타깃 hidden으로
    /// mtp 어텐션 KV[pos]를 채운다 — 이후 mtp_draft_gpu는 전체 문맥을 본다.
    pub fn forward_batch_with_mtp(
        &mut self,
        rows: &[Vec<f32>],
        toks: &[u32],
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>), String> {
        let t = rows.len();
        let out = self.forward_batch(rows)?;
        // 종료 시점: dbab에 최종 잔차(전 노름 전) t행 존재 — 행별 적립.
        let fc = self.lin["mtp.fc"].clone_shallow();
        let lq = self.lin["mtp.layers.0.self_attn.q_proj"].clone_shallow();
        let lk = self.lin["mtp.layers.0.self_attn.k_proj"].clone_shallow();
        let lv = self.lin["mtp.layers.0.self_attn.v_proj"].clone_shallow();
        // SAFETY: dmtpnw 내 행.
        let dmtpnw = self.dmtpnw;
        let nrow = move |i: usize| unsafe { dmtpnw.add(i * 5120 * 4) };
        for r in 0..t {
            let pos = (self.pos as usize - t + r) as u32;
            // h 행 — g_h 규약('마지막 FFN 합산 전')과 동일 클래스를 쓰려면
            // norm_p(128) 전 dbx가 필요하지만 훅은 최종 노름 후 실행 — 마지막 행만
            // g_h(전)에서, 나머지 행은 dbx(후)에서 읽는다(전 행은 마지막 라운드 토큰만
            // 드래프트 입력이 됨). r<t-1 행의 h는 다음 라운드 검증에서만 사용.
            let mut hb = vec![0u8; self.hidden * 4];
            if r + 1 == t {
                // SAFETY: pgh(핀 last_h) 판독 — 상위 forward_batch 완료 동기 후.
                let ghs =
                    unsafe { std::slice::from_raw_parts(self.pgh as *const u8, self.hidden * 4) };
                hb.copy_from_slice(ghs);
            } else {
                // SAFETY: dbx 행(후 시점 — 검증 전용).
                let hp = unsafe { self.dbx.add(r * self.hidden * 4) };
                self.hc.d2h(&mut hb, hp)?;
                self.hc.sync()?;
            }
            // SAFETY: d2h 완료 후 재해석.
            let h: &[f32] =
                unsafe { std::slice::from_raw_parts(hb.as_ptr() as *const f32, self.hidden) };
            let e = self.embed_row_host(toks[r]);
            let eps = 1e-6f32;
            let rms = |x: &[f32], w: &[f32]| -> Vec<f32> {
                let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
                let sc = 1.0 / (ms + eps).sqrt();
                x.iter().zip(w).map(|(v, wv)| v * sc * wv).collect()
            };
            let e_n = rms(&e, &self.mtp_norms[0]);
            let h_n = rms(h, &self.mtp_norms[1]);
            let mut cat = Vec::with_capacity(2 * self.hidden);
            cat.extend_from_slice(&e_n);
            cat.extend_from_slice(&h_n);
            let f32b = |v: &[f32]| unsafe {
                std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4)
            };
            self.hc.h2d(self.dbx, f32b(&cat))?;
            // fc → attn_norm → q/k/v → prep(KV 적립만)
            self.had16_batch(self.dbx, fc.k, 1, fc.suh)?;
            self.gemm2_batch(&fc, 1, self.dbab)?;
            self.norm_ptr(self.dbab, nrow(2), self.dbzero, self.dbxn, 1)?;
            self.had16_batch(self.dbxn, lq.k, 1, lq.suh)?;
            self.gemm2_batch(&lq, 1, self.dsb)?;
            self.had16_batch(self.dbxn, lk.k, 1, lk.suh)?;
            self.gemm2_batch(&lk, 1, self.dsb2)?;
            self.had16_batch(self.dbxn, lv.k, 1, lv.suh)?;
            self.gemm2_batch(&lv, 1, self.dsb3)?;
            self.hc.h2d(self.dmtpp, &pos.to_le_bytes())?;
            let mut tl2 = 1i32;
            let mut p0v = pos as i32;
            let mut ai = 0i32;
            let mut kv = self.kvcap;
            let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7, mut a8) = (
                self.dsb,
                self.dsb2,
                self.dsb3,
                nrow(5),
                nrow(6),
                self.dq2,
                self.dmtpk,
                self.dmtpv,
                self.dmtpp,
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
        }
        Ok(out)
    }
}
