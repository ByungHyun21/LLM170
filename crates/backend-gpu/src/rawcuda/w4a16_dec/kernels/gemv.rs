use super::super::*;

impl W4a16Dec {
    pub(in crate::rawcuda::w4a16_dec) fn gemv_launch(
        &mut self,
        name: &str,
        x32_dev: CUdeviceptr,
        y_out: CUdeviceptr,
    ) -> Result<(), String> {
        let (dq, ds, n, k) = self.lin_spec(name)?;
        let (g, sb) = self.lin_grp.get(name).copied().unwrap_or((128, false));
        if dq == 0 {
            let x16 = self.cast_f16_dev(x32_dev, k)?;
            if self.marlin_gemv1_launch(name, x16, y_out)? {
                return Ok(());
            }
            return Err(format!("gemv {name}: 마린 경로 실패 — 원본 q 폐기됨"));
        }
        if k > g * 256 {
            return Err(format!(
                "gemv {name}: k={k} > {} (t=1 smem sc 계약, g{g})",
                g * 256
            ));
        }
        let sym = crate::rawcuda::gptq4::kernel_sym(false, g, sb)?;
        let f = self.cc.function(sym)?;
        let (mut p_q, mut p_s, mut p_x, mut p_y) = (dq, ds, x32_dev, y_out);
        let (mut p_n, mut p_k) = (n as i32, k as i32);
        self.cc.launch(
            f,
            n as u32,
            1,
            64,
            &mut crate::rawcuda::args::l6(
                &mut p_q, &mut p_s, &mut p_x, &mut p_y, &mut p_n, &mut p_k,
            ),
        )
    }

    pub(in crate::rawcuda::w4a16_dec) fn gemv_dev_x32(
        &mut self,
        name: &str,
        x32_dev: CUdeviceptr,
    ) -> Result<CUdeviceptr, String> {
        let (_, _, n, _) = self.lin_spec(name)?;
        let dy = self.ensure_dy(n)?;
        self.gemv_launch(name, x32_dev, dy)?;
        Ok(dy)
    }

    pub(in crate::rawcuda::w4a16_dec) fn gemv_stage_x32(
        &mut self,
        name: &str,
        x32_dev: CUdeviceptr,
        dst: CUdeviceptr,
        w: usize,
    ) -> Result<(), String> {
        let (_, _, n, _) = self.lin_spec(name)?;
        if n > w {
            return Err(format!("gemv_stage({name}): n={n} > 스테이징 {w}"));
        }
        self.gemv_launch(name, x32_dev, dst)
    }

    pub(in crate::rawcuda::w4a16_dec) fn head_gemv_launch(
        &self,
        x: CUdeviceptr,
        out: CUdeviceptr,
    ) -> Result<(), String> {
        self.head_gemv_i8_launch(x, out)
    }

    pub(in crate::rawcuda::w4a16_dec) fn head_gemv_t_launch(
        &self,
        x: CUdeviceptr,
        out: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        let f = self.cc.function("w4a16_gemv_i8_t")?;
        let (mut p_w, mut p_sc, mut p_x, mut p_o) = (self.head_w, self.head_sc, x, out);
        let (mut p_n, mut p_k, mut p_t) = (self.head_n as i32, self.head_k as i32, t as i32);
        self.cc.launch(
            f,
            self.head_n as u32,
            1,
            64,
            &mut crate::rawcuda::args::l7(
                &mut p_w, &mut p_sc, &mut p_x, &mut p_o, &mut p_n, &mut p_k, &mut p_t,
            ),
        )
    }

    pub(in crate::rawcuda::w4a16_dec) fn head_gemv_i8_launch(
        &self,
        x: CUdeviceptr,
        out: CUdeviceptr,
    ) -> Result<(), String> {
        let f = self.cc.function("w4a16_gemv_i8")?;
        let (mut p_w, mut p_sc, mut p_x, mut p_o) = (self.head_w, self.head_sc, x, out);
        let (mut p_n, mut p_k) = (self.head_n as i32, self.head_k as i32);
        self.cc.launch(
            f,
            self.head_n as u32,
            1,
            64,
            &mut crate::rawcuda::args::l6(
                &mut p_w, &mut p_sc, &mut p_x, &mut p_o, &mut p_n, &mut p_k,
            ),
        )
    }

    pub(in crate::rawcuda::w4a16_dec) fn gemv_t_launch(
        &mut self,
        name: &str,
        x_dev: CUdeviceptr,
        y_dev: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        let (dq, ds, n, k) = self.lin_spec(name)?;
        if dq == 0 {
            return self.gemm_launch(name, x_dev, y_dev, t);
        }
        let (g, sb) = self.lin_grp.get(name).copied().unwrap_or((128, false));
        let sym = match (g, sb) {
            (128, false) => "w4a16_gemv_g128_t",
            (32, true) => "w4a16_gemv_g32_bf16_t",
            _ => {
                return Err(format!(
                    "gemv_t {name}: g{g}/{} 커널 미비",
                    if sb { "bf16" } else { "f16" }
                ));
            }
        };
        if t > 16 && g == 32 && sb {
            let x16 = self.cast_f16_dev(x_dev, t * k)?;
            if self.marlin_gemm_try(name, t, n, k, x16, y_dev, g)? {
                return Ok(());
            }
        }
        if t > 16 && g == 32 && sb {
            let t16 = t & !15;
            let ft = self.cc.function("w4a16_gemv_g32_bf16_tall4")?;
            let (mut aq, mut as_, mut ax, mut ay) = (dq, ds, x_dev, y_dev);
            let (mut an, mut ak, mut at) = (n as i32, k as i32, t16 as i32);
            self.cc.launch(
                ft,
                n.div_ceil(32) as u32,
                t16.div_ceil(128) as u32,
                256,
                &mut crate::rawcuda::args::l7(
                    &mut aq, &mut as_, &mut ax, &mut ay, &mut an, &mut ak, &mut at,
                ),
            )?;
            if t == t16 {
                return Ok(());
            }
            let f = self.cc.function(sym)?;
            let (mut p_q, mut p_s, mut p_x, mut p_y) = (
                dq,
                ds,
                x_dev + (t16 * k * 4) as u64,
                y_dev + (t16 * n * 4) as u64,
            );
            let (mut p_n, mut p_k, mut p_t) = (n as i32, k as i32, (t - t16) as i32);
            return self.cc.launch(
                f,
                n.div_ceil(GEMV_TR) as u32,
                1,
                (32 * GEMV_TR) as u32,
                &mut crate::rawcuda::args::l7(
                    &mut p_q, &mut p_s, &mut p_x, &mut p_y, &mut p_n, &mut p_k, &mut p_t,
                ),
            );
        }
        let f = self.cc.function(sym)?;
        let (mut p_q, mut p_s, mut p_x, mut p_y) = (dq, ds, x_dev, y_dev);
        let (mut p_n, mut p_k, mut p_t) = (n as i32, k as i32, t as i32);
        self.cc.launch(
            f,
            n.div_ceil(GEMV_TR) as u32,
            1,
            (32 * GEMV_TR) as u32,
            &mut crate::rawcuda::args::l7(
                &mut p_q, &mut p_s, &mut p_x, &mut p_y, &mut p_n, &mut p_k, &mut p_t,
            ),
        )
    }

    pub(in crate::rawcuda::w4a16_dec) fn gemv_multi_launch(
        &self,
        plain: bool,
        table: CUdeviceptr,
        x_dev: CUdeviceptr,
        total: usize,
        count: usize,
    ) -> Result<(), String> {
        let name = if plain {
            "w4a16_gemv_multi"
        } else {
            "w4a16_gemv_g128_multi"
        };
        let f = self.cc.function(name)?;
        let (mut p_t, mut p_x) = (table, x_dev);
        let mut p_c = count as i32;
        self.cc.launch(
            f,
            total as u32,
            1,
            64,
            &mut crate::rawcuda::args::l3(&mut p_t, &mut p_x, &mut p_c),
        )
    }

    pub(in crate::rawcuda::w4a16_dec) fn gemv_multi_dev(
        &mut self,
        plain: bool,
        il: usize,
        group: usize,
        x_dev: CUdeviceptr,
    ) -> Result<(), String> {
        if plain {
            return self.gemv_multi_t_dev(true, il, group, x_dev, 1);
        }
        let m = *self
            .dmulti_meta
            .get(il)
            .ok_or_else(|| format!("gemv_multi: 메타 부족 il={il}"))?;
        let (off, cnt, total) = if group == 0 {
            (m[0], m[1], m[2])
        } else {
            (m[3], m[4], m[5])
        };
        if cnt == 0 {
            return Err(format!("gemv_multi: 세그먼트 없음 il={il} g={group}"));
        }
        self.gemv_multi_launch(
            plain,
            self.dmulti + off as u64 * 8,
            x_dev,
            total as usize,
            cnt as usize,
        )
    }

    pub(in crate::rawcuda::w4a16_dec) fn gemv_multi_t_launch(
        &self,
        plain: bool,
        table: CUdeviceptr,
        x_dev: CUdeviceptr,
        total: usize,
        count: usize,
        t: usize,
    ) -> Result<(), String> {
        let name = if plain {
            "w4a16_gemv_bf16_multi_t_v4"
        } else {
            "w4a16_gemv_g128_multi_t"
        };
        let f = self.cc.function(name)?;
        let (mut p_t, mut p_x) = (table, x_dev);
        let (mut p_c, mut p_tt) = (count as i32, t as i32);
        let blk = 32 * GEMV_TR as u32;
        self.cc.launch(
            f,
            total as u32,
            1,
            blk,
            &mut crate::rawcuda::args::l4(&mut p_t, &mut p_x, &mut p_c, &mut p_tt),
        )
    }

    pub(in crate::rawcuda::w4a16_dec) fn gemv_multi_t_dev(
        &mut self,
        plain: bool,
        il: usize,
        group: usize,
        x_dev: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        let m = *self
            .dmulti_meta
            .get(il)
            .ok_or_else(|| format!("gemv_multi_t: 메타 부족 il={il}"))?;
        let (off, cnt, grid) = if group == 0 {
            (m[0], m[1], m[6])
        } else {
            (m[3], m[4], m[7])
        };
        if cnt == 0 {
            return Err(format!("gemv_multi_t: 세그먼트 없음 il={il} g={group}"));
        }
        self.gemv_multi_t_launch(
            plain,
            self.dmulti + off as u64 * 8,
            x_dev,
            grid as usize,
            cnt as usize,
            t,
        )
    }

    pub(in crate::rawcuda::w4a16_dec) fn gemv_multi_g32_t_dev(
        &mut self,
        il: usize,
        group: usize,
        x_dev: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        let m = *self
            .dmulti_meta
            .get(il)
            .ok_or_else(|| format!("gemv_multi_g32_t: 메타 부족 il={il}"))?;
        let (off, cnt, tgrid) = if group == 0 {
            (m[0], m[1], m[6])
        } else {
            (m[3], m[4], m[7])
        };
        if cnt == 0 {
            return Err(format!("gemv_multi_g32_t: 세그먼트 없음 il={il} g={group}"));
        }
        let f = self.cc.function("w4a16_gemv_g32_bf16_multi_t")?;
        let (mut p_t, mut p_x) = (self.dmulti + off as u64 * 8, x_dev);
        let (mut p_c, mut p_tt) = (cnt as i32, t as i32);
        let blk = 32 * GEMV_TR as u32;
        self.cc.launch(
            f,
            tgrid,
            1,
            blk,
            &mut crate::rawcuda::args::l4(&mut p_t, &mut p_x, &mut p_c, &mut p_tt),
        )
    }

    pub(in crate::rawcuda::w4a16_dec) fn gemv_launch_raw(
        &self,
        dq: CUdeviceptr,
        ds: CUdeviceptr,
        n: usize,
        k: usize,
        x_dev: CUdeviceptr,
        y_out: CUdeviceptr,
    ) -> Result<(), String> {
        let sym =
            crate::rawcuda::gptq4::kernel_sym(false, self.moe.moe_group, self.moe.moe_scale_bf16)?;
        let f = self.cc.function(sym)?;
        let (mut p_q, mut p_s, mut p_x, mut p_y) = (dq, ds, x_dev, y_out);
        let (mut p_n, mut p_k) = (n as i32, k as i32);
        self.cc.launch(
            f,
            n as u32,
            1,
            64,
            &mut crate::rawcuda::args::l6(
                &mut p_q, &mut p_s, &mut p_x, &mut p_y, &mut p_n, &mut p_k,
            ),
        )
    }
}
