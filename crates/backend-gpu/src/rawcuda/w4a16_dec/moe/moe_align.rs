use super::super::*;

impl W4a16Dec {
    pub(in crate::rawcuda::w4a16_dec) fn moe_align_launch(
        &self,
        nslots: usize,
    ) -> Result<(), String> {
        let f = self.cc.function("w4a16_moe_align2")?;
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

    pub(in crate::rawcuda::w4a16_dec) fn moe_topk_dev(&mut self) -> Result<(), String> {
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

    pub(in crate::rawcuda::w4a16_dec) fn moe_read_selection(
        &mut self,
    ) -> Result<Vec<(usize, f32)>, String> {
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
