use super::*;

pub(in crate::rawcuda::w4a16_dec) mod decode;
mod gemm;
mod gemv;
mod marlin;
mod marlin_moe;
mod plain;

impl W4a16Dec {
    pub(in crate::rawcuda::w4a16_dec) fn norm_resid_at(
        &mut self,
        w: usize,
        x_dev: CUdeviceptr,
        ab_dev: CUdeviceptr,
        t_len: usize,
        xn32: CUdeviceptr,
    ) -> Result<CUdeviceptr, String> {
        if self.dnw == 0 {
            return Err("norm: 노름 가중 미등록".into());
        }
        if w >= self.norm_w_rows {
            return Err(format!("norm: w={w} >= rows={}", self.norm_w_rows));
        }
        let xn16 = if xn32 != 0 {
            let need = t_len * self.hidden;
            if need > self.dx16.cap {
                self.dx16 = self.realloc_buf(self.dx16, need * 2, need, "dx16_cap")?;
            }
            self.dx16.ptr
        } else {
            0
        };
        let f = self.cc.function(if self.hidden.is_multiple_of(1024) {
            "norm_resid"
        } else {
            "norm_resid_g"
        })?;
        let mut tl = t_len as i32;
        let mut wo = (w * self.hidden) as i32;
        let mut hd = self.hidden as i32;
        let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5) =
            (x_dev, self.dnw, ab_dev, self.norm.dxn, xn32, xn16);
        self.cc.launch(
            f,
            t_len as u32,
            1,
            1024,
            &mut crate::rawcuda::args::l9(
                &mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut a5, &mut tl, &mut wo, &mut hd,
            ),
        )?;
        Ok(self.norm.dxn)
    }

    pub(in crate::rawcuda::w4a16_dec) fn norm_resid_staged(
        &mut self,
        w: usize,
        x_dev: CUdeviceptr,
        ab: &[f32],
    ) -> Result<Vec<f32>, String> {
        if self.hidden == 0 || ab.len() != self.hidden || x_dev == 0 {
            return Err("norm: 순차 잔차/분기 폭 계약 위반".into());
        }
        self.ensure_norm_bufs(1)?;
        let abb = unsafe { std::slice::from_raw_parts(ab.as_ptr() as *const u8, ab.len() * 4) };
        self.cc.h2d(self.norm.dab, abb)?;
        let xn = self.norm_resid_at(w, x_dev, self.norm.dab, 1, 0)?;
        let mut bytes = vec![0u8; self.hidden * 4];
        self.cc.d2h(&mut bytes, xn)?;
        self.cc.sync()?;
        Ok(
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, self.hidden) }
                .to_vec(),
        )
    }

    pub(in crate::rawcuda::w4a16_dec) fn ew_gelu_dev(
        &mut self,
        g_dev: CUdeviceptr,
        u_dev: CUdeviceptr,
        y_dev: CUdeviceptr,
        n: usize,
    ) -> Result<(), String> {
        if n == 0 {
            return Err("ew_gelu: n=0".into());
        }
        let f = self.cc.function("ew_gelu")?;
        let mut nn = n as i32;
        let (mut a0, mut a1, mut a2, mut a3, mut a4) = (g_dev, u_dev, y_dev, 0u64, 0u64);
        self.cc.launch(
            f,
            n.div_ceil(128) as u32,
            1,
            128,
            &mut crate::rawcuda::args::l6(&mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut nn),
        )
    }

    pub(in crate::rawcuda::w4a16_dec) fn cast_f16_dev(
        &mut self,
        x_dev: CUdeviceptr,
        n: usize,
    ) -> Result<CUdeviceptr, String> {
        if n == 0 {
            return Err("cast_f16: n=0".into());
        }
        let p = self.ensure_dxp(n)?;
        let f = self.cc.function("w4a16_cast_f16")?;
        let (mut a0, mut a1, mut nn) = (x_dev, p, n as i32);
        self.cc.launch(
            f,
            n.div_ceil(1024) as u32,
            1,
            256,
            &mut crate::rawcuda::args::l3(&mut a0, &mut a1, &mut nn),
        )?;
        Ok(p)
    }

    pub(in crate::rawcuda::w4a16_dec) fn lin_forward(
        &mut self,
        plain: bool,
        path: LinPath,
        name: &str,
        x: CUdeviceptr,
        dst: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        match (plain, path) {
            (false, LinPath::Gemv1 { w }) => self.gemv_stage_x32(name, x, dst, w),
            (false, LinPath::Gemm) => self.gemm_launch(name, x, dst, t),
            (false, LinPath::GemvT) => {
                if self.has_orig_q(name) {
                    self.gemv_t_launch(name, x, dst, t)
                } else {
                    self.gemm_launch(name, x, dst, t)
                }
            }
            (true, LinPath::Gemv1 { w }) => self.plain_stage_x32(name, x, dst, w),
            (true, LinPath::Gemm) => self.plain_gemm_launch(name, x, dst, t),
            (true, LinPath::GemvT) => self.plain_gemv_t_launch(name, x, dst, t),
        }
    }

    pub(in crate::rawcuda::w4a16_dec) fn axpy_dev(
        &mut self,
        w: f32,
        x: CUdeviceptr,
        y: CUdeviceptr,
        n: usize,
    ) -> Result<(), String> {
        let f = self.cc.function("w4a16_axpy")?;
        let mut ww = w;
        let (mut p_x, mut p_y) = (x, y);
        let mut nn = n as i32;
        self.cc.launch(
            f,
            n.div_ceil(256) as u32,
            1,
            256,
            &mut crate::rawcuda::args::l4(&mut ww, &mut p_x, &mut p_y, &mut nn),
        )
    }

    pub(in crate::rawcuda::w4a16_dec) fn moe_accum_sg_dev(
        &mut self,
        w: CUdeviceptr,
        d: CUdeviceptr,
        sg: CUdeviceptr,
        sx: CUdeviceptr,
        y: CUdeviceptr,
        nslots: usize,
        n: usize,
    ) -> Result<(), String> {
        let f = self.cc.function("w4a16_moe_accum_sg")?;
        let (mut p_w, mut p_d, mut p_sg, mut p_sx, mut p_y) = (w, d, sg, sx, y);
        let (mut p_sp, mut p_ns, mut p_n) = (0i32, nslots as i32, n as i32);
        self.cc.launch(
            f,
            n.div_ceil(256) as u32,
            1,
            256,
            &mut crate::rawcuda::args::l8(
                &mut p_w, &mut p_d, &mut p_sg, &mut p_sx, &mut p_y, &mut p_sp, &mut p_ns, &mut p_n,
            ),
        )
    }

    pub(in crate::rawcuda::w4a16_dec) fn shared_add_dev(
        &mut self,
        sg: CUdeviceptr,
        x: CUdeviceptr,
        y: CUdeviceptr,
        n: usize,
        t: usize,
    ) -> Result<(), String> {
        let f = self.cc.function("w4a16_shared_add")?;
        let (mut p_sg, mut p_x, mut p_y) = (sg, x, y);
        let mut nn = n as i32;
        self.cc.launch(
            f,
            t.max(1) as u32,
            n.div_ceil(256) as u32,
            256,
            &mut crate::rawcuda::args::l4(&mut p_sg, &mut p_x, &mut p_y, &mut nn),
        )
    }

    pub(in crate::rawcuda::w4a16_dec) fn rms_rows_dev(
        &mut self,
        w: usize,
        x_dev: CUdeviceptr,
        y_dev: CUdeviceptr,
        t_len: usize,
        eps: f32,
    ) -> Result<(), String> {
        if self.dnw == 0 {
            return Err("rms_rows: 노름 가중 미등록".into());
        }
        if w >= self.norm_w_rows {
            return Err(format!("rms_rows: w={w} >= rows={}", self.norm_w_rows));
        }
        if t_len == 0 || x_dev == 0 || y_dev == 0 {
            return Err("rms_rows: t=0/포인터 0 계약 위반".into());
        }
        let f = self.cc.function("rms_rows")?;
        let smem = ((self.hidden + 33) * 4) as u32;
        let mut tl = t_len as i32;
        let mut wo = (w * self.hidden) as i32;
        let mut hd = self.hidden as i32;
        let mut ep = eps;
        let (mut a0, mut a1, mut a2) = (x_dev, self.dnw, y_dev);
        self.cc.launch_shared(
            f,
            t_len as u32,
            1,
            256,
            smem,
            &mut crate::rawcuda::args::l7(
                &mut a0, &mut a1, &mut a2, &mut tl, &mut wo, &mut hd, &mut ep,
            ),
        )
    }

    pub(in crate::rawcuda::w4a16_dec) fn norm_resid_dev(
        &mut self,
        w: usize,
        x_dev: CUdeviceptr,
        ab_dev: CUdeviceptr,
        t_len: usize,
        xn32: CUdeviceptr,
    ) -> Result<CUdeviceptr, String> {
        self.ensure_norm_bufs(t_len)?;
        self.norm_resid_at(w, x_dev, ab_dev, t_len, xn32)
    }

    pub(in crate::rawcuda::w4a16_dec) fn ew_dev(
        &mut self,
        g_dev: CUdeviceptr,
        u_dev: CUdeviceptr,
        y_dev: CUdeviceptr,
        n: usize,
    ) -> Result<(), String> {
        self.ew_dev_on(self.cc.stream, g_dev, u_dev, y_dev, n)
    }

    pub(in crate::rawcuda::w4a16_dec) fn ew_dev_on(
        &mut self,
        s: ffi::CUstream,
        g_dev: CUdeviceptr,
        u_dev: CUdeviceptr,
        y_dev: CUdeviceptr,
        n: usize,
    ) -> Result<(), String> {
        if n == 0 {
            return Err("ew: n=0".into());
        }
        let f = self.cc.function("ew")?;
        let mut nn = n as i32;
        let (mut a0, mut a1, mut a2, mut a3, mut a4) = (g_dev, u_dev, y_dev, 0u64, 0u64);
        self.cc.launch_on(
            s,
            f,
            n.div_ceil(128) as u32,
            1,
            128,
            &mut crate::rawcuda::args::l6(&mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut nn),
        )
    }

    pub(in crate::rawcuda::w4a16_dec) fn ew_dev_fused(
        &mut self,
        g_dev: CUdeviceptr,
        u_dev: CUdeviceptr,
        y_dev: CUdeviceptr,
        n: usize,
    ) -> Result<(), String> {
        if n == 0 {
            return Err("ew: n=0".into());
        }
        self.ensure_dx32(n)?;
        self.ensure_dx16(n)?;
        let f = self.cc.function("ew")?;
        let mut nn = n as i32;
        let (mut a0, mut a1, mut a2, mut a3, mut a4) =
            (g_dev, u_dev, y_dev, self.dx32.ptr, self.dx16.ptr);
        self.cc.launch(
            f,
            n.div_ceil(128) as u32,
            1,
            128,
            &mut crate::rawcuda::args::l6(&mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut nn),
        )
    }
}
