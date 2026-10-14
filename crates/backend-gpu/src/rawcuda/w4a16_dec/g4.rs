//! [gemma4 P2-3 2026-10-14] gemma4 v0 체인 조각 — 호스트 스테이징.
//!
//! qwen35 체인(`chains.rs`)과 달리 gemma4의 층 산술(노름·잔차·layer_scalar)은
//! **호스트 코어 계약**(`llm170_core::ops`)으로 조립한다 — v0 목적은 GPU 커널
//! (GEMV·어텐션·전문가·head)의 토큰 수준 검증(P2 게이트)이다. 디코더가
//! 제공하는 것은 내부 기계가 필요한 두 조각뿐:
//!
//! - `g4_moe_host`: 라우터 입력과 전문가 입력이 **다른** gemma4 MoE 계약
//!   (라우터 = 가중 RMS 행 = moe_scale·h^-0.5, 전문가 = pre_ff_norm2) + pes
//!   (per-expert μ_e) + shared 없음(gelu 전문가).
//! - `head_logits_host`: 최종 노름 hidden → GPU head(행별 int8 W8A16) 로짓.
//!
//! 계약 출처: `crates/core/src/gemma4/{mod.rs,stages/ffn.rs}`(CPU 오라클) —
//! 시퀀스: 라우터 GEMV → 디바이스 top-k(전문가 전체 softmax → top-k →
//! 재정규화) → pes → 전문가 gate/up(GLU_gelu)+down → 가중 누적.

use super::*;

impl W4a16Dec {
    /// gemma4 MoE 스텝 — `hn` = 라우터 입력(가중 RMS 행 적용 후), `xn` = 전문가
    /// 입력(pre_ff_norm2), `pes` = 층의 per-expert 스케일(μ_e). 반환 = 전문가
    /// 출력(hidden f32, 호스트). shared 없음(`set_moe(shared_ffn=0)` 전제).
    pub fn g4_moe_host(
        &mut self,
        il: usize,
        hn: &[f32],
        xn: &[f32],
        pes: &[f32],
    ) -> Result<Vec<f32>, String> {
        let h = self.hidden;
        let n_exp = self.moe.n_experts;
        if hn.len() != h || xn.len() != h {
            return Err(format!(
                "g4_moe: 입력 계약 위반 hn={} xn={} != hidden={h}",
                hn.len(),
                xn.len()
            ));
        }
        if n_exp == 0 || pes.len() != n_exp {
            return Err(format!(
                "g4_moe: pes={} != n_experts={n_exp} (set_moe 선행 확인)",
                pes.len()
            ));
        }
        let _g = self.cc.guard()?;
        self.ensure_moe_bufs()?;
        let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let dh = self.cc.alloc(h * 4)?;
        let dx = self.cc.alloc(h * 4)?;
        let dp = self.cc.alloc(n_exp * 4)?;
        let r = (|| -> Result<Vec<f32>, String> {
            self.cc.h2d(dh, &f32b(hn))?;
            self.cc.h2d(dx, &f32b(xn))?;
            self.cc.h2d(dp, &f32b(pes))?;
            // ① 라우터 — 업로드 방식(GEMV 상주/플레인)에 따라 심볼 분기.
            let rn = format!("blk.{il}.moe_gate.weight");
            if self.lins.contains_key(&rn) {
                self.gemv_launch(&rn, dh, self.moe.drt.ptr)?;
            } else {
                self.plain_gemv_launch(&rn, dh, self.moe.drt.ptr)?;
            }
            // ② top-k(디바이스) — 전체 softmax → k선택(내림차순·동률 낮은 idx)
            //    → 재정규화. moe_idx/moe_wt 기록(CPU route 미러).
            self.moe_topk_dev()?;
            // ③ pes — 재정규화 후 가중에 μ_e(인플레이스 — HF Gemma4TextRouter 순서).
            {
                let f = self.cc.function("w4a16_moe_pes_scale")?;
                let (mut p_w, mut p_i, mut p_p) = (self.moe.moe_wt, self.moe.moe_idx, dp);
                let mut p_n = self.moe.top_k as i32;
                self.cc.launch(
                    f,
                    (self.moe.top_k as u32).div_ceil(256).max(1),
                    1,
                    256,
                    &mut crate::rawcuda::args::l4(&mut p_w, &mut p_i, &mut p_p, &mut p_n),
                )?;
            }
            // ④ 전문가 — 상주: 전 슬롯 배치 + accum / 스트리밍: 자체 누적(axpy).
            if self.moe.moe_resident {
                self.moe_experts_batch(il, dx, self.moe.top_k)?;
                self.moe_accum_dev(
                    self.moe.moe_wt,
                    self.moe.exp.dn,
                    self.moe.dmo.ptr,
                    0,
                    self.moe.top_k,
                    h,
                )?;
            } else {
                let sel = self.moe_read_selection()?;
                Self::zero_dev(&self.cc, self.moe.dmo.ptr, h * 4)?;
                self.moe_experts_streaming(il, dx, &sel)?;
            }
            self.cc.sync()?;
            let mut ob = vec![0u8; h * 4];
            self.cc
                .d2h_async(ob.as_mut_ptr(), self.moe.dmo.ptr, h * 4)?;
            self.cc.sync()?;
            Ok(ob
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect())
        })();
        let _ = self.cc.free(dh);
        let _ = self.cc.free(dx);
        let _ = self.cc.free(dp);
        r
    }

    /// [gemma4 P3 2026-10-14] t행 GEMM 호스트 래퍼 — [t][k] f32 in → [t][n] out.
    /// `gemm_launch`(그룹 심볼 — g32 FFMA) + 사설 스크래치(1회 할당).
    pub fn g4_gemm_host(
        &mut self,
        name: &str,
        x: &[f32],
        t: usize,
        k: usize,
    ) -> Result<Vec<f32>, String> {
        let _g = self.cc.guard()?;
        let (_, _, n, kk) = self.lin_spec(name)?;
        if t == 0 || k != kk || x.len() != t * k {
            return Err(format!(
                "g4_gemm: {name} 계약 위반 t={t} k={k}(선형 {kk}) x={}",
                x.len()
            ));
        }
        if self.g4_a.cap < t * k {
            if self.g4_a.ptr != 0 {
                self.cc.free(self.g4_a.ptr)?;
            }
            self.g4_a.ptr = self.cc.alloc(t * k * 4)?;
            self.g4_a.cap = t * k;
        }
        if self.g4_y.cap < t * n {
            if self.g4_y.ptr != 0 {
                self.cc.free(self.g4_y.ptr)?;
            }
            self.g4_y.ptr = self.cc.alloc(t * n * 4)?;
            self.g4_y.cap = t * n;
        }
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
        let xb = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) };
        self.cc.h2d(self.g4_a.ptr, xb)?;
        self.gemm_launch(name, self.g4_a.ptr, self.g4_y.ptr, t)?;
        self.cc.sync()?;
        let mut ob = vec![0u8; t * n * 4];
        self.cc
            .d2h_async(ob.as_mut_ptr(), self.g4_y.ptr, t * n * 4)?;
        self.cc.sync()?;
        Ok(ob
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    /// [gemma4 P3 2026-10-14] t행(≤8) MoE 스텝 호스트 래퍼 — 라우터(hn)·
    /// 전문가(xn) 입력 분리 + pes + 무 shared. 반환 = [t][h] (토큰별 누적).
    /// 전문가 = 상주 GEMV 융합(gate+up+GLU_gelu) + down + accum(sp=top_k) —
    /// t=1 경로와 동일 커널 계열. 스트리밍(t행)은 미지원 → 호출부가 t=1 폴백.
    pub fn g4_moe_host_t(
        &mut self,
        il: usize,
        hn: &[f32],
        xn: &[f32],
        pes: &[f32],
        t: usize,
        pes_dev: CUdeviceptr,
    ) -> Result<Vec<f32>, String> {
        let h = self.hidden;
        let n_exp = self.moe.n_experts;
        let tk = self.moe.top_k.max(1);
        let n_ff = self.moe.moe_ffn;
        if t == 0 || t > 8 || hn.len() != t * h || xn.len() != t * h || pes.len() != n_exp {
            return Err(format!(
                "g4_moe_t: 계약 위반 t={t} hn={} xn={} pes={} (h={h} n_exp={n_exp})",
                hn.len(),
                xn.len(),
                pes.len()
            ));
        }
        if !self.moe.moe_resident {
            return Err("g4_moe_t: 스트리밍 t행 미지원(호출부 t=1 폴백)".into());
        }
        let _g = self.cc.guard()?;
        self.ensure_moe_bufs()?;
        // t행 입력 스크래치(g4_a/g4_y 재사용 — gemm 래퍼와 상호배타 순차).
        let need = t * h;
        if self.g4_a.cap < need {
            if self.g4_a.ptr != 0 {
                self.cc.free(self.g4_a.ptr)?;
            }
            self.g4_a.ptr = self.cc.alloc(need * 4)?;
            self.g4_a.cap = need;
        }
        if self.g4_y.cap < need {
            if self.g4_y.ptr != 0 {
                self.cc.free(self.g4_y.ptr)?;
            }
            self.g4_y.ptr = self.cc.alloc(need * 4)?;
            self.g4_y.cap = need;
        }
        let (dhn, dxn) = (self.g4_a.ptr, self.g4_y.ptr);
        let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        self.cc.h2d(dhn, &f32b(hn))?;
        self.cc.h2d(dxn, &f32b(xn))?;
        let _ = pes; // pes는 디바이스에 이미 있다(pes_dev — 층별 1회 업로드)
        // ① 라우터 → drt [t][n_exp].
        let rn = format!("blk.{il}.moe_gate.weight");
        if self.lins.contains_key(&rn) {
            self.gemm_launch(&rn, dhn, self.moe.drt.ptr, t)?;
        } else {
            self.plain_gemm_launch(&rn, dhn, self.moe.drt.ptr, t)?;
        }
        // ② top-k(디바이스, t행) — softmax 전체 → top-k → 재정규화.
        {
            let f = self.cc.function("w4a16_moe_topk_t")?;
            let (mut p_lg, mut p_ix, mut p_wt) =
                (self.moe.drt.ptr, self.moe.moe_idx, self.moe.moe_wt);
            let (mut p_t, mut p_n, mut p_k) = (t as i32, n_exp as i32, tk as i32);
            self.cc.launch(
                f,
                t.div_ceil(8) as u32,
                1,
                256,
                &mut crate::rawcuda::args::l6(
                    &mut p_lg, &mut p_ix, &mut p_wt, &mut p_t, &mut p_n, &mut p_k,
                ),
            )?;
        }
        // ③ pes(μ_e) — 인플레이스(가중 [t][top_k] 전량).
        {
            let ns = t * tk;
            let f = self.cc.function("w4a16_moe_pes_scale")?;
            let (mut p_w, mut p_i, mut p_p) = (self.moe.moe_wt, self.moe.moe_idx, pes_dev);
            let mut p_n = ns as i32;
            self.cc.launch(
                f,
                (ns as u32).div_ceil(256).max(1),
                1,
                256,
                &mut crate::rawcuda::args::l4(&mut p_w, &mut p_i, &mut p_p, &mut p_n),
            )?;
        }
        // ④ 전문가 — 상주 GEMV 융합(gate+up+GLU) + down + 누적(sp=tk).
        let ns = t * tk;
        let base = il * n_exp * 3;
        self.gemv_experts_glu_launch(base, ns, dxn, h, tk, self.moe.exp.act, n_ff, h)?;
        self.gemv_experts_launch(
            base + 2,
            ns,
            self.moe.exp.act,
            n_ff,
            1,
            self.moe.exp.dn,
            h,
            n_ff,
        )?;
        self.moe_accum_dev(
            self.moe.moe_wt,
            self.moe.exp.dn,
            self.moe.dmo.ptr,
            tk,
            ns,
            h,
        )?;
        self.cc.sync()?;
        let mut ob = vec![0u8; t * h * 4];
        self.cc
            .d2h_async(ob.as_mut_ptr(), self.moe.dmo.ptr, t * h * 4)?;
        self.cc.sync()?;
        Ok(ob
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    /// [gemma4 P3 2026-10-14] t행(≤8) head 로짓 — 가중 1판독 t행 공유(TR 커널).
    /// [t][head_n] 평탄 반환.
    pub fn g4_head_host_t(&mut self, xn: &[f32], t: usize) -> Result<Vec<f32>, String> {
        if self.head_w == 0 {
            return Err("g4_head_t: head 미등록".into());
        }
        let (h, n) = (self.hidden, self.head_n);
        if t == 0 || t > 8 || xn.len() != t * h {
            return Err(format!("g4_head_t: 계약 위반 t={t} xn={}", xn.len()));
        }
        let _g = self.cc.guard()?;
        if self.g4_a.cap < t * h {
            if self.g4_a.ptr != 0 {
                self.cc.free(self.g4_a.ptr)?;
            }
            self.g4_a.ptr = self.cc.alloc(t * h * 4)?;
            self.g4_a.cap = t * h;
        }
        // t행 출력 전용 버퍼(head_out은 1행 크기 — t행 d2h 범위 밖 결함 수정).
        if self.g4_y.cap < t * n {
            if self.g4_y.ptr != 0 {
                self.cc.free(self.g4_y.ptr)?;
            }
            self.g4_y.ptr = self.cc.alloc(t * n * 4)?;
            self.g4_y.cap = t * n;
        }
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
        let xb = unsafe { std::slice::from_raw_parts(xn.as_ptr() as *const u8, xn.len() * 4) };
        self.cc.h2d(self.g4_a.ptr, xb)?;
        self.head_gemv_t_launch(self.g4_a.ptr, self.g4_y.ptr, t)?;
        self.cc.sync()?;
        let mut ob = vec![0u8; t * n * 4];
        self.cc
            .d2h_async(ob.as_mut_ptr(), self.g4_y.ptr, t * n * 4)?;
        self.cc.sync()?;
        Ok(ob
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    /// [gemma4 P3 2026-10-14] 청크 디바이스 체인 — 스크래치 슬롯 확보.
    pub fn g4_scratch_slot(&mut self, idx: usize, bytes: usize) -> Result<CUdeviceptr, String> {
        if self.g4_pool.len() <= idx {
            self.g4_pool.resize(idx + 1, Buf { ptr: 0, cap: 0 });
        }
        let b = &mut self.g4_pool[idx];
        if b.cap < bytes {
            if b.ptr != 0 {
                self.cc.free(b.ptr)?;
            }
            b.ptr = self.cc.alloc(bytes)?;
            b.cap = bytes;
        }
        Ok(b.ptr)
    }

    /// 슬롯 f32 행렬 업로드 — [t][k] f32 → 슬롯(반환 = 디바이스 포인터).
    pub fn g4_dev_in(&mut self, slot: usize, x: &[f32]) -> Result<CUdeviceptr, String> {
        let _g = self.cc.guard()?;
        let p = self.g4_scratch_slot(slot, x.len() * 4)?;
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
        let xb = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) };
        self.cc.h2d(p, xb)?;
        Ok(p)
    }

    /// t행 GEMM → 지정 슬롯(디바이스 in/out — 중간 왕복 0). 반환 = 슬롯 포인터.
    pub fn g4_dev_gemm(
        &mut self,
        name: &str,
        x_dev: CUdeviceptr,
        t: usize,
        k: usize,
        y_slot: usize,
    ) -> Result<CUdeviceptr, String> {
        let _g = self.cc.guard()?;
        let (_, _, n, kk) = self.lin_spec(name)?;
        if t == 0 || k != kk {
            return Err(format!("g4_dev_gemm: {name} k={k} != {kk}"));
        }
        let y = self.g4_scratch_slot(y_slot, t * n * 4)?;
        self.gemm_launch(name, x_dev, y, t)?;
        Ok(y)
    }

    /// 슬롯 → 호스트 f32 회수(동기 1회).
    pub fn g4_dev_out(&mut self, slot: usize, elems: usize) -> Result<Vec<f32>, String> {
        let _g = self.cc.guard()?;
        let p = self.g4_scratch_slot(slot, elems * 4)?;
        self.cc.sync()?;
        let mut ob = vec![0u8; elems * 4];
        self.cc.d2h_async(ob.as_mut_ptr(), p, elems * 4)?;
        self.cc.sync()?;
        Ok(ob
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    /// [gemma4 P3 2026-10-14] ew_gelu 디바이스 — act = gelu(g)·u → y(슬롯).
    /// t=1 경로(`ew_gelu_host`)와 동일 커널 심볼(수치 정합).
    pub fn g4_dev_gelu(
        &mut self,
        g_dev: CUdeviceptr,
        u_dev: CUdeviceptr,
        y_slot: usize,
        n: usize,
    ) -> Result<CUdeviceptr, String> {
        if n == 0 {
            return Err("g4_dev_gelu: n=0".into());
        }
        let _g = self.cc.guard()?;
        let y = self.g4_scratch_slot(y_slot, n * 4)?;
        let f = self.cc.function("ew_gelu")?;
        let mut nn = n as i32;
        let (mut a0, mut a1, mut a2, mut a3, mut a4) = (g_dev, u_dev, y, 0u64, 0u64);
        self.cc.launch(
            f,
            n.div_ceil(128) as u32,
            1,
            128,
            &mut crate::rawcuda::args::l6(&mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut nn),
        )?;
        Ok(y)
    }

    /// 슬롯 포인터(할당 보장 없음 — g4_scratch_slot 선행 계약).
    pub fn g4_dev_ptr(&self, slot: usize) -> CUdeviceptr {
        self.g4_pool.get(slot).map(|b| b.ptr).unwrap_or(0)
    }

    /// [gemma4 P3 2026-10-14] g4 어텐션 **디바이스·행별 테이블** 런처 — q/k/v·
    /// 출력 모두 디바이스(중간 d2h 0). 행 i = 슬롯 slots[i]·위치 pos[i](연속일
    /// 필요 없음 — 배치 디코드 지원). KV는 각 슬롯 아레나 기록. 반환 = outv 슬롯.
    /// qnw/knw = set_attn d_max 패딩 배열의 이 층 행(커널 행 포인터 계약 정합).
    #[allow(clippy::too_many_arguments)]
    pub fn attn_g4_dev(
        &mut self,
        slots: &[usize],
        pos: &[u32],
        layer: usize,
        q_dev: CUdeviceptr,
        k_dev: CUdeviceptr,
        v_dev: CUdeviceptr,
        q_heads: usize,
        kv_heads: usize,
        d: usize,
        sw: usize,
        kq_scale: f32,
        splits: usize,
        qh_slot: usize,
        part_slot: usize,
        out_slot: usize,
    ) -> Result<CUdeviceptr, String> {
        let t = slots.len();
        if t == 0 || pos.len() != t {
            return Err(format!("attn_g4_dev: t={t} pos={}", pos.len()));
        }
        let _g = self.cc.guard()?;
        let dm = self.attn.attn.ok_or("attn: 형상 미등록")?;
        if self.attn.g4_rp == 0 || layer >= dm.n_attn {
            return Err(format!("attn_g4_dev: layer={layer} n_attn={}", dm.n_attn));
        }
        let qn = t * q_heads * d;
        let qh = self.g4_scratch_slot(qh_slot, qn * 4)?;
        let part = self.g4_scratch_slot(part_slot, t * q_heads * splits * (d + 2) * 4)?;
        let outv = self.g4_scratch_slot(out_slot, qn * 4)?;
        // 행별 (pos, 블록 테이블) 포인터 테이블 — 커널 row_blk/row_pp 계약.
        // 행별 (pos, 블록 테이블) 포인터 테이블 — 커널 row_blk/row_pp 계약.
        // **층 무관**(커널이 층 오프셋 가산) — (slots,pos) 동일하면 업로드 생략.
        // 층당 동기 h2d 4회(pageable = 디바이스 전체 동기화)가 파이프라인을
        // 층마다 드레인했다 — 청크당 1회로 축소. [주의] 테이블 슬롯(10/11/12)은
        // 청크 데이터 슬롯(0..9)과 **분리 유지** — 재사용하면 청크 중간값이
        // 테이블을 덮어 불법 접근(전면 상주 실험에서 실측).
        let mb = dm.cap.div_ceil(512);
        let pos_dev = self.g4_scratch_slot(10, t * 4)?;
        let rpp_dev = self.g4_scratch_slot(11, t * 8)?;
        let rblk_dev = self.g4_scratch_slot(12, t * 8)?;
        if self.g4_atab.0 != slots || self.g4_atab.1 != pos {
            let mut pb: Vec<u8> = Vec::with_capacity(t * 4);
            for &pv in pos {
                pb.extend_from_slice(&pv.to_le_bytes());
            }
            self.cc.h2d(pos_dev, &pb)?;
            let mut rppb: Vec<u8> = Vec::with_capacity(t * 8);
            let mut rblkb: Vec<u8> = Vec::with_capacity(t * 8);
            for (i, &sl) in slots.iter().enumerate() {
                rppb.extend_from_slice(&(pos_dev + (i * 4) as u64).to_le_bytes());
                let base = self.attn.dblk_tab + (sl * dm.n_attn * mb * 4 * 8) as u64;
                rblkb.extend_from_slice(&base.to_le_bytes());
            }
            self.cc.h2d(rpp_dev, &rppb)?;
            self.cc.h2d(rblk_dev, &rblkb)?;
            self.g4_atab = (slots.to_vec(), pos.to_vec());
        }
        let dpp = pos_dev; // null 경로 비사용(호환 인자)
        let d_max = self.attn.layers.iter().map(|l| l.d).max().unwrap_or(d);
        let qnw = self.attn.dqnw_a + (layer * d_max * 4) as u64;
        let knw = self.attn.dknw_a + (layer * d_max * 4) as u64;
        let dblk = self.attn.dblk_tab; // 비사용(행 테이블 우선) — 호환 인자
        let drp = self.attn.g4_rp;
        // ① prep_g4 — grid (t, q_heads+kv_heads) · 128스레드.
        {
            let f = self.cc.function("attn_prep_g4")?;
            let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7, mut a8) =
                (q_dev, k_dev, v_dev, qnw, knw, qh, dblk, dpp, drp);
            let (mut rblk, mut rpp) = (rblk_dev, rpp_dev); // 행별 테이블
            let (mut tl, mut lay, mut qh_, mut kvh, mut cp) = (
                t as i32,
                layer as i32,
                q_heads as i32,
                kv_heads as i32,
                dm.cap as i32,
            );
            self.cc.launch(
                f,
                t as u32,
                (q_heads + kv_heads) as u32,
                128,
                &mut crate::rawcuda::args::l16(
                    &mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut a5, &mut a6, &mut a7,
                    &mut rblk, &mut rpp, &mut a8, &mut tl, &mut lay, &mut qh_, &mut kvh, &mut cp,
                ),
            )?;
        }
        // ② f3s part_g4 — grid (t, q_heads×splits) · d스레드.
        {
            let f = self.cc.function("attn_fwd3s_part_g4")?;
            let (mut a0, mut a1, mut a2, mut a3) = (qh, dblk, part, dpp);
            let (mut rblk, mut rpp) = (rblk_dev, rpp_dev); // 행별 테이블
            let (mut tl, mut lay, mut qh_, mut kvh, mut cp) = (
                t as i32,
                layer as i32,
                q_heads as i32,
                kv_heads as i32,
                dm.cap as i32,
            );
            let (mut sp, mut dd, mut swv) = (splits as i32, d as i32, sw as i32);
            let mut scl = kq_scale;
            self.cc.launch(
                f,
                t as u32,
                (q_heads * splits) as u32,
                d as u32,
                &mut crate::rawcuda::args::l15(
                    &mut a0, &mut a1, &mut a2, &mut a3, &mut rblk, &mut rpp, &mut tl, &mut lay,
                    &mut qh_, &mut kvh, &mut cp, &mut sp, &mut dd, &mut swv, &mut scl,
                ),
            )?;
        }
        // ③ merge_g4 — grid (t, q_heads) · d스레드.
        {
            let f = self.cc.function("attn_fwd3s_merge_g4")?;
            let (mut a0, mut a1, mut a2, mut a3) = (part, outv, 0u64, 0u64);
            let (mut tl, mut qh_, mut sp, mut dd) =
                (t as i32, q_heads as i32, splits as i32, d as i32);
            self.cc.launch(
                f,
                t as u32,
                q_heads as u32,
                d as u32,
                &mut crate::rawcuda::args::l8(
                    &mut a0, &mut a1, &mut a2, &mut a3, &mut tl, &mut qh_, &mut sp, &mut dd,
                ),
            )?;
        }
        Ok(outv)
    }

    /// [gemma4 P3 2026-10-14] 층별 pes(μ_e) 디바이스 상주 — 1회 업로드 캐시.
    pub fn set_g4_pes(&mut self, il: usize, pes: &[f32]) -> Result<(), String> {
        let _g = self.cc.guard()?;
        if self.g4_pes.len() <= il {
            self.g4_pes.resize(il + 1, 0);
        }
        if self.g4_pes[il] == 0 {
            let bytes: Vec<u8> = pes.iter().flat_map(|x| x.to_le_bytes()).collect();
            let p = self.cc.alloc(bytes.len())?;
            self.cc.h2d(p, &bytes)?;
            self.g4_pes[il] = p;
        }
        Ok(())
    }

    /// 층 pes 디바이스 포인터(0 = 미등록).
    pub fn g4_pes_dev(&self, il: usize) -> CUdeviceptr {
        self.g4_pes.get(il).copied().unwrap_or(0)
    }

    /// 최종 노름 hidden → GPU head 로짓(호스트 회수). `upload_head` 선행 계약
    /// (행별 int8 W8A16 — W4A16 단일 head 계약).
    pub fn head_logits_host(&mut self, xn: &[f32]) -> Result<Vec<f32>, String> {
        if self.head_w == 0 {
            return Err("head_logits_host: head 미등록 — upload_head 선행".into());
        }
        if xn.len() != self.hidden {
            return Err(format!(
                "head_logits_host: xn={} != hidden={}",
                xn.len(),
                self.hidden
            ));
        }
        let _g = self.cc.guard()?;
        let dx = self.cc.alloc(self.hidden * 4)?;
        let r = (|| -> Result<Vec<f32>, String> {
            let b: Vec<u8> = xn.iter().flat_map(|x| x.to_le_bytes()).collect();
            self.cc.h2d(dx, &b)?;
            self.head_gemv_launch(dx, self.head_out)?;
            let mut ob = vec![0u8; self.head_n * 4];
            self.cc
                .d2h_async(ob.as_mut_ptr(), self.head_out, self.head_n * 4)?;
            self.cc.sync()?;
            Ok(ob
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect())
        })();
        let _ = self.cc.free(dx);
        r
    }
}
