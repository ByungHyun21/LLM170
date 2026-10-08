//! S11 배치 forward — 프롬프트 N토큰을 T=N으로 한 번에 처리한다.
//!
//! [왜 배치인가] S10 이후에도 프리필은 **토큰당 T=1 순차 1회**라 장문
//! 프롬프트가 느리다. T>1로 넘기면 GEMM2·norm이 행 병렬로 처리하고
//! lm_head도 T행을 한 번에 돈다.
//!
//! [전제 — S11 동치성 게이트] 배치가 T=1 순차와 **같은 값**을 내야 한다.
//! `cuda_probe s11`이 GDN(T=2,4)·어텐션(T=2,4)에서 검증한다: 어텐션
//! 0.000e0, GDN ≤7.5e-4(f16 청크 누산, 임계 1e-3 이내). 이 게이트가 실패하면
//! 배치는 "자기 규칙으로 맞는" 계산일 뿐 정답이 아니므로 열지 않는다.
//!
//! [도메인] T는 1..=ATTN_F3S_TMAX(8)로 제한한다. 어텐션 fwd3s가 그 상한을
//! 강제하고, S9의 위치축 한계(1024)와 별개로 GDN scan의 CS=32 청크도 T≤8
//! 에서는 청크 경계를 넘지 않아 순차와 정확히 같다.
//!
//! [산술 동일성] 각 단계는 S10과 **같은 커널**을 쓴다. T=1일 때 S10 디바이스
//! 경로와 토큰열이 동일한지 S10 게이트가 보장하므로, T>1은 그 계산의 행 병렬
//! 확장에 해당한다.

use crate::rawcuda::exl3_cuda::Exl3CudaDecoder;
use crate::rawcuda::ffi::CUdeviceptr;

impl Exl3CudaDecoder {
    /// T행 임베딩 → (마지막 행 로짓, 최종 norm 이전 잔차 마지막 행).
    /// 슬롯 잔차·KV·pos가 T만큼 전진한다.
    pub fn forward_batch_device(
        &mut self,
        slot: usize,
        embed_rows: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let (h, n_slots) = (self.hidden, self.n_slots.max(1));
        if h == 0 || embed_rows.is_empty() || !embed_rows.len().is_multiple_of(h) {
            return Err(format!(
                "exl3-cuda: 배치 임베딩 {} — [t][hidden] 계약 위반",
                embed_rows.len()
            ));
        }
        let t = embed_rows.len() / h;
        if slot >= n_slots {
            return Err(format!("exl3-cuda: slot={slot} >= n_slots={n_slots}"));
        }
        let tmax = crate::rawcuda::attn_cuda::ATTN_F3S_TMAX;
        if t > tmax {
            return Err(format!(
                "exl3-cuda: T={t} > fwd3s 상한 {tmax} — 프리필을 {tmax}토큰 \
                 단위로 나눠 호출할 것 (S11)"
            ));
        }
        let pos = self.slot_pos[slot];
        // [S12] 위치축은 cap(=--ctx)만 제한한다 — fwd3s가 청크 온라인
        // 소프트맥스로 바뀌어 공유메모리 상한(1024)이 사라졌다.
        let cap = self.attn_dims()?.cap;
        if pos as usize + t > cap {
            return Err(format!(
                "context overflow: slot{slot} pos={pos}+T={t} > kvcap={cap} (--ctx 상향 필요)"
            ));
        }
        if self.norm_w_rows < 2 * self.n_layers + 1 || self.gdn.is_none() {
            return Err("exl3-cuda: 디코더 상수 미등록 — load_slots 필요".into());
        }
        let _g = self.cc.guard()?;
        self.ensure_chain_probe_bufs_pub()?;
        self.prewarm_chain_bufs()?;
        self.ensure_batch_bufs()?;
        // T행 임베딩을 잔차 버퍼로 올린다(프리필당 유일한 대량 h2d).
        // SAFETY: f32 [t][hidden] 바이트 뷰 — 업로드까지 살아 있다.
        let eb = unsafe {
            std::slice::from_raw_parts(embed_rows.as_ptr() as *const u8, embed_rows.len() * 4)
        };
        self.cc
            .h2d(self.dres, eb)
            .map_err(|e| format!("임베딩 {}행 업로드: {e}", t))?;

        let mut ab = self.dab_dev;
        let mut gi = 0usize;
        for il in 0..self.loaded_layers.min(self.n_layers) {
            let lp = format!("model.language_model.layers.{il}");
            // 노름은 T축 지원(그리드 t_len) — S10과 같은 커널.
            let xn = self.norm_resid_dev(2 * il, self.dres, ab, t)?;
            let branch = if il % 4 == 3 {
                // 슬롯 4는 어텐션 q_dim(6144) 행을 담는다 — 최대 폭으로
                // 미리 잡아 두고 읽기 전에 존재하게 한다.
                self.ensure_stage(4, self.attn_dims()?.q_dim(), t)?;
                let att = format!("{lp}.self_attn");
                self.bstage(&format!("{att}.q_proj"), xn, 0, t)?;
                self.bstage(&format!("{att}.k_proj"), xn, 1, t)?;
                self.bstage(&format!("{att}.v_proj"), xn, 2, t)?;
                let c0 = self.bchain(0)?;
                let c1 = self.bchain(1)?;
                let c2 = self.bchain(2)?;
                // attn_chain_dev_run은 dqg_a/dkin_a/dvin_a(작업 버퍼)와
                // doutv_a(산출)에 복사한다. ensure_attn_bufs가 T=1 용량으로
                // 잡힐 수 있으므로 T를 명시해 재확보한다.
                self.ensure_attn_bufs_pub(t)?;
                let out = self.attn_chain_dev_run(slot, il / 4, t, c0, c1, c2)?;
                // 산출은 [T][q_dim]이므로 배치 스테이징으로 옮긴다.
                let odst = self.bchain(4)?;
                self.cc.d2d(odst, out, t * self.attn_dims()?.q_dim() * 4)?;
                odst
            } else {
                let att = format!("{lp}.linear_attn");
                // 슬롯 4는 GDN v_len(6144) 행을 담는다.
                self.ensure_stage(4, self.gdn_dims()?.v_len(), t)?;
                self.bstage(&format!("{att}.in_proj_qkv"), xn, 0, t)?;
                self.bstage(&format!("{att}.in_proj_z"), xn, 1, t)?;
                let c0 = self.bchain(0)?;
                let c1 = self.bchain(1)?;
                // GDN 산출도 [T][v_len]이므로 배치 스테이징으로 옮긴다.
                self.ensure_gdn_bufs_pub(t)?;
                let g = self
                    .gdn_chain_dev_run(slot, gi, t, xn, c0, c1)
                    .map_err(|e| format!("L{il} gdn(T={t}): {e}"))?;
                let gd = self.bchain(4)?;
                self.cc.d2d(gd, g, t * self.gdn_dims()?.v_len() * 4)?;
                gi += 1;
                gd
            };
            let lo = if il % 4 == 3 {
                format!("{lp}.self_attn.o_proj")
            } else {
                format!("{lp}.linear_attn.out_proj")
            };
            // out_proj/o_proj은 어텐션·GDN 산출 [T][q_dim]/[T][v_len]을 읽는다.
            // branch의 행 폭을 층 종류별로 정확히 넘긴다 — 틀린 stride는
            // 두 번째 행부터 엉뚱한 값을 읽는다(원장 S11).
            let bstride = if il % 4 == 3 {
                self.attn_dims()?.q_dim()
            } else {
                self.gdn_dims()?.v_len()
            };
            let out = self.bgemv(&lo, branch, bstride, t)?;
            let xn2 = self.norm_resid_dev(2 * il + 1, self.dres, out, t)?;
            let mlp = format!("{lp}.mlp");
            self.bstage(&format!("{mlp}.gate_proj"), xn2, 0, t)?;
            self.bstage(&format!("{mlp}.up_proj"), xn2, 1, t)?;
            let (c0, c1) = (self.bchain(0)?, self.bchain(1)?);
            self.ew_batch(c0, c1, t)?;
            let act = self.bchain(3)?;
            ab = self.bgemv_down(&format!("{mlp}.down_proj"), act, t)?;
        }
        let xn_final = self.norm_resid_dev(2 * self.n_layers, self.dres, ab, t)?;
        let n_head = self.lin_copy("lm_head").map(|l| l.n)?;
        // lm_head는 [T][hidden] → [T][n_head]. 마지막 행만 d2h한다.
        let hd_src = self.bchain(3)?;
        self.cc.d2d(hd_src, xn_final, t * self.hidden * 4)?;
        let logits_ptr = self.gemm2_dev("lm_head", hd_src, t)?;
        let mut lb = vec![0u8; n_head * 4];
        self.cc
            .d2h(&mut lb, logits_ptr + ((t - 1) * n_head) as u64 * 4)?;
        self.cc.sync()?;
        self.slot_pos[slot] = pos + t as u32;
        self.attn_set_pos(slot, pos + t as u32)?;
        // SAFETY: d2h 동기 완료; lb는 n_head개의 f32 LE 값.
        let logits =
            unsafe { std::slice::from_raw_parts(lb.as_ptr() as *const f32, n_head) }.to_vec();
        // 최종 norm 이전 잔차의 마지막 행.
        let mut hb = vec![0u8; self.hidden * 4];
        self.cc
            .d2h(&mut hb, self.dres + ((t - 1) * self.hidden) as u64 * 4)?;
        self.cc.sync()?;
        let hidden =
            unsafe { std::slice::from_raw_parts(hb.as_ptr() as *const f32, self.hidden) }.to_vec();
        Ok((logits, hidden))
    }

    /// P0-3(plans/cuda-models.md §3.2): 스펙 검증 배치 — T행 forward +
    /// 행별 argmax + MTP KV 적립(행 r의 MTP 입력 h = 행 r-1의 pre-final-norm
    /// 잔차; h0는 배치 직전 커밋 잔사, None이면 행 0을 건너뛴다 — 첫 프리필
    /// 청크의 "위치 0의 h_{-1}은 없음" 계약, qwen4exp P15④와 동일).
    /// 반환: (행별 argmax[t], 마지막 행 잔차 h_new, 마지막 행 로짓).
    /// 타깃 경로 산술은 forward_batch_device와 동일(MTP는 자체 KV에만 쓴다).
    pub fn verify_batch_with_mtp(
        &mut self,
        slot: usize,
        embed_rows: &[f32],
        mtp: &crate::rawcuda::mtp_cuda::Exl3CudaMtp,
        h0: Option<&[f32]>,
    ) -> Result<(Vec<u32>, Vec<f32>, Vec<f32>), String> {
        let (h, n_slots) = (self.hidden, self.n_slots.max(1));
        if h == 0 || embed_rows.is_empty() || !embed_rows.len().is_multiple_of(h) {
            return Err(format!(
                "exl3-cuda: 검증 임베딩 {} — [t][hidden] 계약 위반",
                embed_rows.len()
            ));
        }
        let t = embed_rows.len() / h;
        if slot >= n_slots {
            return Err(format!("exl3-cuda: slot={slot} >= n_slots={n_slots}"));
        }
        let tmax = crate::rawcuda::attn_cuda::ATTN_F3S_TMAX;
        if t > tmax {
            return Err(format!("exl3-cuda: 검증 T={t} > fwd3s 상한 {tmax}"));
        }
        let pos = self.slot_pos[slot];
        let cap = self.attn_dims()?.cap;
        if pos as usize + t > cap {
            return Err(format!(
                "context overflow: slot{slot} pos={pos}+T={t} > kvcap={cap}"
            ));
        }
        if self.norm_w_rows < 2 * self.n_layers + 1 || self.gdn.is_none() {
            return Err("exl3-cuda: 디코더 상수 미등록 — load_slots 필요".into());
        }
        let _g = self.cc.guard()?;
        self.ensure_chain_probe_bufs_pub()?;
        self.prewarm_chain_bufs()?;
        self.ensure_batch_bufs()?;
        // SAFETY: f32 [t][hidden] 바이트 뷰.
        let eb = unsafe {
            std::slice::from_raw_parts(embed_rows.as_ptr() as *const u8, embed_rows.len() * 4)
        };
        self.cc.h2d(self.dres, eb)?;

        let mut ab = self.dab_dev;
        let mut gi = 0usize;
        for il in 0..self.loaded_layers.min(self.n_layers) {
            let lp = format!("model.language_model.layers.{il}");
            let xn = self.norm_resid_dev(2 * il, self.dres, ab, t)?;
            let branch = if il % 4 == 3 {
                self.ensure_stage(4, self.attn_dims()?.q_dim(), t)?;
                let att = format!("{lp}.self_attn");
                self.bstage(&format!("{att}.q_proj"), xn, 0, t)?;
                self.bstage(&format!("{att}.k_proj"), xn, 1, t)?;
                self.bstage(&format!("{att}.v_proj"), xn, 2, t)?;
                let c0 = self.bchain(0)?;
                let c1 = self.bchain(1)?;
                let c2 = self.bchain(2)?;
                self.ensure_attn_bufs_pub(t)?;
                let out = self.attn_chain_dev_run(slot, il / 4, t, c0, c1, c2)?;
                let odst = self.bchain(4)?;
                self.cc.d2d(odst, out, t * self.attn_dims()?.q_dim() * 4)?;
                odst
            } else {
                let att = format!("{lp}.linear_attn");
                self.ensure_stage(4, self.gdn_dims()?.v_len(), t)?;
                self.bstage(&format!("{att}.in_proj_qkv"), xn, 0, t)?;
                self.bstage(&format!("{att}.in_proj_z"), xn, 1, t)?;
                let c0 = self.bchain(0)?;
                let c1 = self.bchain(1)?;
                self.ensure_gdn_bufs_pub(t)?;
                let g = self
                    .gdn_chain_dev_run(slot, gi, t, xn, c0, c1)
                    .map_err(|e| format!("L{il} gdn(T={t}): {e}"))?;
                let gd = self.bchain(4)?;
                self.cc.d2d(gd, g, t * self.gdn_dims()?.v_len() * 4)?;
                gi += 1;
                gd
            };
            let lo = if il % 4 == 3 {
                format!("{lp}.self_attn.o_proj")
            } else {
                format!("{lp}.linear_attn.out_proj")
            };
            let bstride = if il % 4 == 3 {
                self.attn_dims()?.q_dim()
            } else {
                self.gdn_dims()?.v_len()
            };
            let out = self.bgemv(&lo, branch, bstride, t)?;
            let xn2 = self.norm_resid_dev(2 * il + 1, self.dres, out, t)?;
            let mlp = format!("{lp}.mlp");
            self.bstage(&format!("{mlp}.gate_proj"), xn2, 0, t)?;
            self.bstage(&format!("{mlp}.up_proj"), xn2, 1, t)?;
            let (c0, c1) = (self.bchain(0)?, self.bchain(1)?);
            self.ew_batch(c0, c1, t)?;
            let act = self.bchain(3)?;
            ab = self.bgemv_down(&format!("{mlp}.down_proj"), act, t)?;
        }

        // ── MTP KV 적립(레이어 루프 후, 최종 노름 전 — dres가 전 행의
        // pre-final-norm 잔차다). 행 r의 h 입력 = dres[r-1](r=0은 h0).
        // with_head=false — 드래프트 헤드 없이 KV·상태만 전진.
        for r in 0..t {
            let h_dev = if r == 0 {
                match h0 {
                    Some(hv) => {
                        if hv.len() != h {
                            return Err(format!("mtp h0 {} != hidden {h}", hv.len()));
                        }
                        // SAFETY: f32 슬라이스 바이트 뷰(호출 내 유효).
                        let hb = unsafe {
                            std::slice::from_raw_parts(hv.as_ptr() as *const u8, hv.len() * 4)
                        };
                        self.cc.h2d(mtp.dh_buf(), hb)?;
                        mtp.dh_buf()
                    }
                    None => continue, // 청크 행 0 — h_{-1} 부재(첫 프리필)
                }
            } else {
                self.dres + ((r - 1) * h) as u64 * 4
            };
            let e_row = &embed_rows[r * h..(r + 1) * h];
            mtp.mtp_step_g(self, e_row, h_dev, false, None)?;
            mtp.mtp_pos_bump(&self.cc)?;
        }

        let xn_final = self.norm_resid_dev(2 * self.n_layers, self.dres, ab, t)?;
        let n_head = self.lin_copy("lm_head").map(|l| l.n)?;
        let hd_src = self.bchain(3)?;
        self.cc.d2d(hd_src, xn_final, t * self.hidden * 4)?;
        let logits_ptr = self.gemm2_dev("lm_head", hd_src, t)?;
        let mut ams = Vec::with_capacity(t);
        for r in 0..t {
            let tok = self.argmax_dev(logits_ptr + (r * n_head) as u64 * 4, n_head)?;
            ams.push(tok);
        }
        let mut lb = vec![0u8; n_head * 4];
        self.cc
            .d2h(&mut lb, logits_ptr + ((t - 1) * n_head) as u64 * 4)?;
        self.cc.sync()?;
        // SAFETY: d2h 동기 완료 — lb는 n_head개 f32 LE(마지막 행 로짓).
        let logits_last =
            unsafe { std::slice::from_raw_parts(lb.as_ptr() as *const f32, n_head) }.to_vec();
        self.slot_pos[slot] = pos + t as u32;
        self.attn_set_pos(slot, pos + t as u32)?;
        let mut hb = vec![0u8; self.hidden * 4];
        self.cc.d2h(&mut hb, self.dres + ((t - 1) * self.hidden) as u64 * 4)?;
        self.cc.sync()?;
        // SAFETY: d2h 동기 완료 — hb는 hidden개 f32 LE.
        let hnew =
            unsafe { std::slice::from_raw_parts(hb.as_ptr() as *const f32, self.hidden) }.to_vec();
        Ok((ams, hnew, logits_last))
    }

    /// 배치 GEMM → 스테이징 슬롯 s. [T][k] 입력을 받아 [T][n]을 쓴다.
    pub(crate) fn bstage(
        &mut self,
        key: &str,
        x_dev: CUdeviceptr,
        s: usize,
        t: usize,
    ) -> Result<(), String> {
        let n = self.lin_copy(key).map(|l| l.n)?;
        // ensure_stage가 할당하므로 bchain보다 먼저 부른다.
        self.ensure_stage(s, n, t)?;
        let dst = self.bchain(s)?;
        let p = self
            .gemm2_dev(key, x_dev, t)
            .map_err(|e| format!("bstage({key}, T={t}): {e}"))?;
        self.cc
            .d2d(dst, p, t * n * 4)
            .map_err(|e| format!("bstage({key}) 복사 {t}x{n}: {e}"))
    }

    /// 어텐션 o_proj · GDN out_proj — 산출이 이미 [T][n]이므로 T=1 GEMV를
    /// 행별로 T회 돌린다(GEMV 커널에 T 인자가 없다). **x_stride는 입력
    /// 산출의 행 폭**(어텐션 q_dim / GDN v_len)이고 출력 n과 다르다.
    pub(crate) fn bgemv(
        &mut self,
        key: &str,
        x_dev: CUdeviceptr,
        x_stride: usize,
        t: usize,
    ) -> Result<CUdeviceptr, String> {
        let n = self.lin_copy(key).map(|l| l.n)?;
        self.ensure_stage(4, n, t)?;
        let dst = self.bchain(4)?;
        for i in 0..t {
            let src = x_dev + (i * x_stride) as u64 * 4;
            // 행별로 GEMV가 dx를 입력으로 삼으므로 한 칸 스테이징을 거친다.
            let tmp = self.dchain[0];
            self.cc.d2d(tmp, src, x_stride * 4)?;
            let p = self.gemv_dev(key, tmp)?;
            self.cc.d2d(dst + (i * n) as u64 * 4, p, n * 4)?;
        }
        Ok(dst)
    }

    /// down_proj — [T][ffn] → [T][hidden]. GEMV에 T 인자가 없어 행별 T회다.
    pub(crate) fn bgemv_down(
        &mut self,
        key: &str,
        x_dev: CUdeviceptr,
        t: usize,
    ) -> Result<CUdeviceptr, String> {
        let n = self.lin_copy(key).map(|l| l.n)?;
        self.ensure_stage(4, n, t)?;
        let dst = self.bchain(4)?;
        let k = self.lin_copy(key).map(|l| l.k)?;
        for i in 0..t {
            let tmp = self.dchain[0];
            self.cc.d2d(tmp, x_dev + (i * k) as u64 * 4, k * 4)?;
            let p = self.gemv_dev(key, tmp)?;
            self.cc.d2d(dst + (i * n) as u64 * 4, p, n * 4)?;
        }
        Ok(dst)
    }

    /// silu(g)·u 배치 — ew 커널은 T 인자가 없어 행별로 T회 발사한다.
    pub(crate) fn ew_batch(
        &mut self,
        g: CUdeviceptr,
        u: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        let n = self.stg_w2;
        self.ensure_stage(3, n, t)?;
        let dst = self.bchain(3)?;
        for i in 0..t {
            self.ew_dev(
                g + (i * n) as u64 * 4,
                u + (i * n) as u64 * 4,
                dst + (i * n) as u64 * 4,
                n,
            )?;
        }
        Ok(())
    }

    pub(crate) fn bchain(&self, s: usize) -> Result<CUdeviceptr, String> {
        self.bchain
            .get(s)
            .copied()
            .filter(|p| *p != 0)
            .ok_or_else(|| {
                format!(
                    "exl3-cuda: 배치 스테이징 {s} 미할당 (cap {:?})",
                    self.bchain_cap
                )
            })
    }

    pub(crate) fn ensure_stage(&mut self, s: usize, n: usize, t: usize) -> Result<(), String> {
        let want = n * t;
        if self.bchain_cap[s] < want {
            if self.bchain[s] != 0 {
                self.cc.free(self.bchain[s])?;
            }
            // 모든 슬롯을 최대 폭(FFN 17408) × TMAX으로 잡는다. 층마다 n이
            // 달라지므로(6144·10240·12288·17408) 슬롯별 최소 할당은
            // 연속 호출에서 재할당을 유발하고, 그 사이 커널이 이전
            // 포인터를 읽는 창이 생긴다(원장 S11).
            let widest = self.stg_w2.max(self.stg_w0).max(n);
            let cap = widest * crate::rawcuda::attn_cuda::ATTN_F3S_TMAX;
            self.bchain[s] = self.cc.alloc(cap * 4)?;
            self.bchain_cap[s] = cap;
        }
        Ok(())
    }

    /// bstage의 gemv_t 판(plans/cuda-port.md §1 착수조건 1) — bstage와
    /// 동일 계약([T][k] 입력 → 슬롯 s에 [T][n] 사본)이나 gemm2(mma f32
    /// 누산) 대신 행병렬 gemv_t를 써서 행별 T=1 GEMV와 **비트동일**을
    /// 보장한다. 슬롯 간 배치 디코드 전용 — S11 프리필은 기존
    /// bstage(gemm2) 기본선을 그대로 유지한다(게이트 무변화).
    pub(crate) fn bstage_gt(
        &mut self,
        key: &str,
        x_dev: CUdeviceptr,
        s: usize,
        t: usize,
    ) -> Result<(), String> {
        let n = self.lin_copy(key).map(|l| l.n)?;
        // ensure_stage가 할당하므로 bchain보다 먼저 부른다.
        self.ensure_stage(s, n, t)?;
        let dst = self.bchain(s)?;
        let p = self
            .gemv_t_dev(key, x_dev, t)
            .map_err(|e| format!("bstage_gt({key}, T={t}): {e}"))?;
        self.cc
            .d2d(dst, p, t * n * 4)
            .map_err(|e| format!("bstage_gt({key}) 복사 {t}x{n}: {e}"))
    }

    /// bgemv/bgemv_down의 gemv_t 판 — 입력이 [T][k] 연속 행이므로 행별
    /// 스테이징 T회(gemm2 체인의 d2d 왕복) 없이 한 번에 들어간다.
    /// 어텐션 산출 [T][q_dim](o_proj.k=q_dim)·GDN 산출 [T][v_len]
    /// (out_proj.k=v_len)·ew 산출 [T][ffn](down_proj.k=ffn) 모두 행 폭이
    /// 곧 해당 선형의 k라 이 계약이 성립한다. 출력은 bchain(4) 스테이징
    /// 사본(dyt는 다음 gemv_t가 덮는다 — 즉시 소비 규약).
    pub(crate) fn bgemv_gt(
        &mut self,
        key: &str,
        x_dev: CUdeviceptr,
        t: usize,
    ) -> Result<CUdeviceptr, String> {
        let n = self.lin_copy(key).map(|l| l.n)?;
        self.ensure_stage(4, n, t)?;
        let dst = self.bchain(4)?;
        let p = self
            .gemv_t_dev(key, x_dev, t)
            .map_err(|e| format!("bgemv_gt({key}, T={t}): {e}"))?;
        self.cc
            .d2d(dst, p, t * n * 4)
            .map_err(|e| format!("bgemv_gt({key}) 복사 {t}x{n}: {e}"))?;
        Ok(dst)
    }

    fn ensure_batch_bufs(&mut self) -> Result<(), String> {
        Ok(())
    }
}
