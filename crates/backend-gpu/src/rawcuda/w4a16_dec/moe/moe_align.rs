use super::super::*;

impl W4a16Dec {
    /// [P11] 전문가-우선 슬롯 순열 발사(프리필 전용) — w4a16_moe_align.
    /// n_exp ≤ 1024 계약(초과 시 호출부가 종전 경로로 폴백).
    pub(in crate::rawcuda::w4a16_dec) fn moe_align_launch(
        &self,
        nslots: usize,
    ) -> Result<(), String> {
        let f = self.cc.function("w4a16_moe_align")?;
        let (mut p_i, mut p_ns) = (self.moe.moe_idx, nslots as i32);
        let (mut p_g, mut p_c, mut p_o) =
            (self.moe.moe_gslot, self.moe.moe_gcnt, self.moe.moe_goff);
        let mut p_ne = self.moe.n_experts as i32;
        self.cc.launch(
            f,
            1,
            1,
            256,
            &mut crate::rawcuda::args::l6(
                &mut p_i, &mut p_ns, &mut p_g, &mut p_c, &mut p_o, &mut p_ne,
            ),
        )
    }

    /// 라우터 top-k 디바이스(P1) — 게이트 GEMV는 [B-4] 다중 세그먼트 런치가
    /// 수행(moe_multi_gate). 여기는 drt 로짓 → idx/wt 선택만.
    pub(in crate::rawcuda::w4a16_dec) fn moe_topk_dev(&mut self) -> Result<(), String> {
        // [B-3 2026-10-10] t=1도 **워프 병렬 topk_t** — 종전 단일 스레드
        // 커널(R3에서 제거 — 구 w4a16_moe_topk)은 128전문가 softmax+8라운드를
        // 한 스레드가 순차 처리(층당 수µs × 40층). 시맨틱은 동일 미러
        // (gptq4.cu 주석) — 골든으로 판정.
        // [B-4 후속 2026-10-10] n=128은 레지스터 전용 커널(비트동일) —
        // 종전 topk_t t=1 실측 10.2µs(2,474명령·로컬메모리 왕복).
        if self.moe.n_experts == 256 {
            let f = self.cc.function("w4a16_moe_topk256")?;
            let (mut p_lg, mut p_ix, mut p_wt) =
                (self.moe.drt.ptr, self.moe.moe_idx, self.moe.moe_wt);
            let mut p_k = self.moe.top_k as i32;
            return self.cc.launch(
                f,
                1,
                1,
                32,
                &mut crate::rawcuda::args::l4(&mut p_lg, &mut p_ix, &mut p_wt, &mut p_k),
            );
        }
        let f = self.cc.function("w4a16_moe_topk_t")?;
        let (mut p_lg, mut p_ix, mut p_wt) = (self.moe.drt.ptr, self.moe.moe_idx, self.moe.moe_wt);
        let (mut p_t, mut p_n, mut p_k) = (1i32, self.moe.n_experts as i32, self.moe.top_k as i32);
        self.cc.launch(
            f,
            1,
            1,
            32,
            &mut crate::rawcuda::args::l6(
                &mut p_lg, &mut p_ix, &mut p_wt, &mut p_t, &mut p_n, &mut p_k,
            ),
        )
    }

    /// [B6/B7 2026-10-12] 디바이스 top-k(id/wt, moe_idx/moe_wt) → 호스트 목록.
    /// 스트리밍 스테이징은 id 선택만 필요 — 종전 라우터 로짓 128 d2h+호스트
    /// 정렬(moe_route)을 대체한다(시맨틱은 topk_t가 미러 — 골든 판정).
    pub(super) fn moe_read_selection(&mut self) -> Result<Vec<(usize, f32)>, String> {
        let tk = self.moe.top_k.max(1);
        let mut ib = vec![0u8; tk * 4];
        let mut wb = vec![0u8; tk * 4];
        self.cc
            .d2h_async(ib.as_mut_ptr(), self.moe.moe_idx, tk * 4)?;
        self.cc
            .d2h_async(wb.as_mut_ptr(), self.moe.moe_wt, tk * 4)?;
        self.cc.sync()?;
        let mut out: Vec<(usize, f32)> = Vec::with_capacity(tk);
        for k in 0..tk {
            let e = u32::from_le_bytes(ib[k * 4..k * 4 + 4].try_into().unwrap()) as usize;
            let w = f32::from_le_bytes(wb[k * 4..k * 4 + 4].try_into().unwrap());
            out.push((e, w));
        }
        Ok(out)
    }
}
