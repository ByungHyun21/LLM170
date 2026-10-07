//! EXL3 CUDA 디바이스 상주 forward — 호스트 왕복 제거(S10 성능 캠페인).
//! S5의 호스트 스테이징 경로는 GEMV마다 d2h→h2d를 반복해 층당 ~8회
//! 전송한다(64층 × 토큰마다). 여기서는 잔차·분기·중간값을 전부 디바이스에
//! 남기고 **토큰당 임베딩 업로드 1회 + 최종 잔차/로짓 d2h 2회**만 왕복한다.
//!
//! [산술 동일성] 각 단계는 S5와 **같은 커널·같은 순서**를 쓴다:
//! GEMV(had_in→gemv→had_out) · norm_resid · gdn/attn 체인 · ew.
//! 바뀌는 것은 값이 이동하는 경로뿐이라 CPU 오라클 비트일치가 성립해야
//! 하고, S10 검증에서 실측한다.
//!
//! [버퍼 소유] 커널 인자로 넘기는 포인터는 **디코더 필드 상주**다.
//! 스택 지역에 두면 진행 중 재할당으로 dangling이 된다. 각 모듈의
//! `ensure_*`는 용량 부족 시에만 재할당하며 체인 진행 중에는 재할당되지
//! 않는다(t_len 고정).

use crate::rawcuda::exl3_cuda::Exl3CudaDecoder;
use crate::rawcuda::ffi::CUdeviceptr;

impl Exl3CudaDecoder {
    /// 슬롯 임베딩 행 1개 → (로짓, 최종 노름 이전 잔차).
    pub fn forward_device(
        &mut self,
        slot: usize,
        embed_row: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let (h, n_slots) = (self.hidden, self.n_slots.max(1));
        if h == 0 || embed_row.len() != h {
            return Err(format!(
                "exl3-cuda: 임베딩 폭 {} != hidden {}",
                embed_row.len(),
                h
            ));
        }
        if slot >= n_slots {
            return Err(format!("exl3-cuda: slot={slot} >= n_slots={n_slots}"));
        }
        let pos = self.slot_pos[slot];
        let s9 = crate::rawcuda::attn_cuda::ATTN_SCORE_SCAP;
        if pos as usize >= self.attn_dims()?.cap {
            return Err(format!(
                "context overflow: slot{slot} pos={pos}+1 > kvcap (--ctx 상향 필요)"
            ));
        }
        if pos as usize + 1 > s9 {
            return Err(format!(
                "context overflow: slot{slot} pos={pos}+1 > {s9} \
                 (CUDA fwd3s 위치축 한계 — 위치 청크 미구현, plans/cuda-port.md S9)"
            ));
        }
        if self.norm_w_rows < 2 * self.n_layers + 1 || self.gdn.is_none() {
            return Err("exl3-cuda: 디코더 상수 미등록 — load_slots 필요".into());
        }
        let _g = self.cc.guard()?;
        self.ensure_chain_bufs()?;
        // 진행 중 dyb/dah/dsb 재할당을 원천 차단한다(위 prewarm 주석).
        self.prewarm_chain_bufs()?;
        // SAFETY: f32 [hidden]을 바이트 뷰로 복사; 업로드까지 살아 있다.
        let row = unsafe { std::slice::from_raw_parts(embed_row.as_ptr() as *const u8, h * 4) };
        self.cc.h2d(self.dres, row)?;

        let [s0, s1, s1b, s2, s3] = self.dchain;
        let (w0, w1, w2) = (self.stg_w0, self.stg_w1, self.stg_w2);
        // 분기 버퍼는 norm 모듈의 dab가 아니라 S10 전용 필드를 쓴다 —
        // 필드를 공유하면 ensure_norm_bufs가 재할당하며 체인 진행 중 커널
        // 인자가 dangling이 된다.
        let mut ab = self.dab_dev;
        let mut gi = 0usize;
        for il in 0..self.loaded_layers.min(self.n_layers) {
            let lp = format!("model.language_model.layers.{il}");
            let xn = self
                .norm_resid_dev(2 * il, self.dres, ab, 1)
                .map_err(|e| format!("L{il} input norm: {e}"))?;
            let branch = if il % 4 == 3 {
                let att = format!("{lp}.self_attn");
                // q/k/v를 스테이징으로 옮기고 어텐션 체인을 발사한다.
                // k와 v는 kv_dim 폭이라 같은 버퍼(s1)를 순차 재사용한다.
                self.gemv_stage(&format!("{att}.q_proj"), xn, s0, w0)
                    .map_err(|e| format!("L{il} q_proj: {e}"))?;
                self.gemv_stage(&format!("{att}.k_proj"), xn, s1, w1)
                    .map_err(|e| format!("L{il} k_proj: {e}"))?;
                self.gemv_stage(&format!("{att}.v_proj"), xn, s1b, w1)
                    .map_err(|e| format!("L{il} v_proj: {e}"))?;
                self.attn_chain_dev_run(slot, il / 4, 1, s0, s1, s1b)?
            } else {
                let att = format!("{lp}.linear_attn");
                self.gemv_stage(&format!("{att}.in_proj_qkv"), xn, s0, w0)
                    .map_err(|e| format!("L{il} in_proj_qkv: {e}"))?;
                self.gemv_stage(&format!("{att}.in_proj_z"), xn, s1, w1)
                    .map_err(|e| format!("L{il} in_proj_z: {e}"))?;
                let gated = self.gdn_chain_dev_run(slot, gi, 1, xn, s0, s1)?;
                gi += 1;
                gated
            };
            // 어텐션/GDN 산출이 dyb에 있다. o_proj·out_proj가 그것을 읽는다.
            let lo = if il % 4 == 3 {
                format!("{lp}.self_attn.o_proj")
            } else {
                format!("{lp}.linear_attn.out_proj")
            };
            let out = self
                .gemv_dev(&lo, branch)
                .map_err(|e| format!("L{il} {lo}: {e}"))?;
            let xn2 = self
                .norm_resid_dev(2 * il + 1, self.dres, out, 1)
                .map_err(|e| format!("L{il} post norm: {e}"))?;
            let mlp = format!("{lp}.mlp");
            self.gemv_stage(&format!("{mlp}.gate_proj"), xn2, s0, w0)
                .map_err(|e| format!("L{il} gate_proj: {e}"))?;
            self.gemv_stage(&format!("{mlp}.up_proj"), xn2, s1, w1)
                .map_err(|e| format!("L{il} up_proj: {e}"))?;
            self.ew_dev(s0, s1, s2, w2)?;
            let down = self
                .gemv_dev(&format!("{mlp}.down_proj"), s2)
                .map_err(|e| format!("L{il} down_proj: {e}"))?;
            // down 결과(dyb)를 다음 층 분기 버퍼로 복사 — dyb는 다음 GEMV가
            // 덮으므로 상주 사본이 필요하다.
            self.cc.d2d(s3, down, self.hidden * 4)?;
            ab = s3;
        }
        // 최종 norm 직전 잔차 = dres — S5의 두 번째 반환값과 같은 의미.
        let mut hidden_bytes = vec![0u8; h * 4];
        self.cc.d2h(&mut hidden_bytes, self.dres)?;
        self.cc.sync()?;
        let xn_final = self.norm_resid_dev(2 * self.n_layers, self.dres, ab, 1)?;
        let n_head = self.lin_copy("lm_head").map(|l| l.n)?;
        let logits_ptr = self.gemv_dev("lm_head", xn_final)?;
        let mut lb = vec![0u8; n_head * 4];
        self.cc.d2h(&mut lb, logits_ptr)?;
        self.cc.sync()?;
        self.slot_pos[slot] = pos + 1;
        self.attn_set_pos(slot, pos + 1)?;
        // SAFETY: d2h 동기 완료; 각 버퍼는 hidden·n_head개의 f32 LE 값.
        let logits =
            unsafe { std::slice::from_raw_parts(lb.as_ptr() as *const f32, n_head) }.to_vec();
        let hidden =
            unsafe { std::slice::from_raw_parts(hidden_bytes.as_ptr() as *const f32, h) }.to_vec();
        Ok((logits, hidden))
    }

    /// GEMV 1회 → 상주 스테이징으로 복사(단일 대여 — 중첩 빌림 회피).
    /// 복사량은 **그 선형의 실제 출력폭 n**이다. 스테이징 버퍼 폭 w는
    /// 여러 선형이 공유하므로 w까지 복하면 dyb(정확히 n×4B)를 넘겨
    /// cuMemcpyDtoD가 INVALID_VALUE로 거절한다.
    fn gemv_stage(
        &mut self,
        key: &str,
        x_dev: CUdeviceptr,
        dst: CUdeviceptr,
        w: usize,
    ) -> Result<(), String> {
        let n = self.lin_copy(key).map(|l| l.n)?;
        if n > w {
            return Err(format!(
                "gemv_stage({key}): 출력폭 {n} > 스테이징 폭 {w} — ensure_chain_bufs 형상 계산 오류"
            ));
        }
        let p = self.gemv_dev(key, x_dev)?;
        self.cc.d2d(dst, p, n * 4)
    }

    /// 체인 버퍼 보장 — 배치 경로(S11)·검증층에서 함께 쓴다.
    pub(crate) fn ensure_chain_probe_bufs_pub(&mut self) -> Result<(), String> {
        self.ensure_chain_bufs()
    }

    /// 체인 작업 버퍼 보장(상주). 폭은 모듈 형상에서 산출한다 — hidden을
    /// 가정하면 어텐션 qg(27B 12288)·GDN qkv(10240)·ew(17408)가 넘친다.
    ///
    /// [S11] dres·dab_dev는 **T×hidden** 용량이어야 한다(배치 forward가
    /// T행 임베딩을 한 번에 올린다). T=1(S10)만 보면 1행분이면 충분해
    /// 보이지만, 배치 경로는 h2d가 INVALID_VALUE로 죽는다.
    fn ensure_chain_bufs(&mut self) -> Result<(), String> {
        if self.chain_bufs_ok {
            return Ok(());
        }
        let h = self.hidden;
        let ad = self.attn_dims()?;
        let gd = self.gdn_dims()?;
        let ff = self
            .lin
            .get("model.language_model.layers.0.mlp.gate_proj")
            .map(|l| l.n)
            .ok_or("gate_proj 미등록 — FFN 폭 산출 불가")?;
        // 슬롯 0/1은 여러 선형이 공유하므로 각 슬롯에 들어갈 수 있는 최대
        // 출력폭을 쓴다: 슬롯 0 = qg(어텐션) · qkv(GDN) · gate/up(FFN),
        // 슬롯 1 = kin/vin(어텐션) · z(GDN) · up(FFN). ew가 gate·up을
        // **동시에** 읽으므로 둘 다 FFN 폭까지 확보돼야 한다.
        let ff_w = self
            .lin
            .get("model.language_model.layers.0.mlp.up_proj")
            .map(|l| l.n)
            .unwrap_or(ff);
        let w0 = ad.qg_dim().max(gd.conv_ch()).max(ff);
        let w1 = ad.kv_dim().max(gd.v_len()).max(ff_w);
        for p in [
            self.dres,
            self.dab_dev,
            self.dchain[0],
            self.dchain[1],
            self.dchain[2],
            self.dchain[3],
            self.dchain[4],
        ] {
            if p != 0 {
                self.cc.free(p)?;
            }
        }
        // 잔차·분기는 TMAX행까지 확보한다(S11 배치 forward가 T행으로 한 번에 쓴다).
        let tmax = crate::rawcuda::attn_cuda::ATTN_F3S_TMAX;
        self.dres = self.cc.alloc(h * tmax * 4)?;
        self.dab_dev = self.cc.alloc(h * tmax * 4)?;
        // 첫 층의 분기(ab)는 0이어야 한다 — 잔차 스트림이 복구(reset)되지
        // 않으면 직전 실행의 down_proj 결과가 첫 층에 더해진다(원장 S10:
        // 초기화 누락은 첫 norm까지는 0.000e0로看似 정상이나 두 번째
        // forward부터 hidden이 벌어진다).
        self.cc.h2d(self.dab_dev, &vec![0u8; h * tmax * 4])?;
        let b0 = self.cc.alloc(w0 * 4)?;
        let b1 = self.cc.alloc(w1 * 4)?;
        let b1b = self.cc.alloc(w1 * 4)?;
        let b2 = self.cc.alloc(ff * 4)?;
        let b3 = self.cc.alloc(h * tmax * 4)?;
        self.dchain = [b0, b1, b1b, b2, b3];
        self.stg_w0 = w0;
        self.stg_w1 = w1;
        self.stg_w2 = ff;
        self.chain_bufs_ok = true;
        Ok(())
    }
}
