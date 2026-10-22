use super::super::*;

pub(in crate::rawcuda::w4a16_dec) fn sk_ksp(blocks: usize, k: usize) -> Option<u8> {
    if !k.is_multiple_of(128) {
        return None;
    }
    let groups = k / 128;
    let cap = (groups / 5).clamp(1, SK_KSP_CAP);
    let mut best = 1usize;
    for ksp in [1usize, 2, 4, 8] {
        if ksp <= cap && groups.is_multiple_of(ksp) {
            best = ksp;
        }
    }
    (blocks * best >= SK_SM_TARGET).then_some(best as u8)
}

pub(in crate::rawcuda::w4a16_dec) const SK_SM_TARGET: usize = 70;

pub(in crate::rawcuda::w4a16_dec) const SK_KSP_CAP: usize = 8;

impl W4a16Dec {
    pub(in crate::rawcuda::w4a16_dec) fn sk_multi_dev(
        &mut self,
        il: usize,
        group: usize,
        x16: CUdeviceptr,
    ) -> Result<bool, String> {
        let Some(m) = self.dsk_meta.get(il).and_then(|m| m[group]) else {
            return Ok(false);
        };
        if m.cnt == 0 || self.dsk == 0 {
            return Ok(false);
        }
        let (blocks, ksp) = (m.blocks as usize, m.ksp as usize);
        let f = self.cc.function("w4a16_gemv_sk_marlin")?;
        let (mut p_t, mut p_x, mut p_p) = (self.dsk + m.off as u64 * 8, x16, self.dsk_part.ptr);
        let (mut p_c, mut p_s) = (m.cnt as i32, ksp as i32);
        self.cc.launch(
            f,
            blocks as u32,
            ksp as u32,
            128,
            &mut crate::rawcuda::args::l5(&mut p_t, &mut p_x, &mut p_p, &mut p_c, &mut p_s),
        )?;
        if ksp > 1 {
            let f = self.cc.function("w4a16_gemv_sk_reduce")?;
            self.cc.launch(
                f,
                blocks as u32,
                1,
                256,
                &mut crate::rawcuda::args::l4(&mut p_t, &mut p_p, &mut p_c, &mut p_s),
            )?;
        }
        Ok(true)
    }

    pub(in crate::rawcuda::w4a16_dec) fn sk_single_dev(
        &mut self,
        name: &str,
        x16: CUdeviceptr,
        out: CUdeviceptr,
    ) -> Result<bool, String> {
        let (_, ds, n, k) = self.lin_spec(name)?;
        let Some((dqm, _, _)) = self.lin_mk(name) else {
            return Ok(false);
        };
        let blocks = n.div_ceil(256);
        let Some(ksp) = sk_ksp(blocks, k) else {
            return Ok(false);
        };
        let f = self.cc.function("w4a16_gemv_sk1_marlin")?;
        let (mut p_q, mut p_s, mut p_x) = (dqm, ds, x16);
        let (mut p_o, mut p_p) = (out, self.dsk_part.ptr);
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
        Ok(true)
    }
}
