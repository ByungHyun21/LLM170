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
