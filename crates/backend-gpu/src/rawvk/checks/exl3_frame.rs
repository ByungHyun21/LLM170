//! [프레임 모듈 — vk 측정 원장 2026-10-04]
//! 원-서브밋 프레임: pp512 123 t/s · norm_resid exact · fframe 스냅샷/롤백
use super::exl3_resident::{BATCH_TMAX, FFrame, TrellisResident, n_gdn_bytes};

impl TrellisResident {
    pub fn fframe_init(&mut self) -> Result<(), String> {
        if self.batch.as_ref().is_some_and(|b| b.fframe.is_some()) {
            return Ok(());
        }
        self.ensure_batch()?;
        let pnr = self
            .ctx
            .pipeline_pipes(include_bytes!("../spv/e3_norm_resid.spv"), 4, 8)?;
        let pnrh = self
            .ctx
            .pipeline_pipes(include_bytes!("../spv/e3_norm_resid_had.spv"), 8, 8)?;
        let xbuf = self.ctx.alloc_host_cached(BATCH_TMAX * 5120 * 4)?;
        let zeros = self.ctx.alloc_host_cached(BATCH_TMAX * 5120 * 4)?;
        let nw128 = self.ctx.alloc_host_cached(129 * 5120 * 4)?;
        unsafe {
            std::ptr::write_bytes(zeros.ptr, 0, BATCH_TMAX * 5120 * 4);
            std::ptr::write_bytes(xbuf.ptr, 0, BATCH_TMAX * 5120 * 4);
            for il in 0..self.n_layers {
                let lp = format!("model.language_model.layers.{il}");
                let wi = self
                    .norm(&format!("{lp}.input_layernorm.weight"))
                    .ok_or("input_ln")?;
                let wp = self
                    .norm(&format!("{lp}.post_attention_layernorm.weight"))
                    .ok_or("post_ln")?;
                std::ptr::copy_nonoverlapping(
                    wi.as_ptr(),
                    nw128.ptr.add((2 * il) * 5120 * 4) as *mut f32,
                    5120,
                );
                std::ptr::copy_nonoverlapping(
                    wp.as_ptr(),
                    nw128.ptr.add((2 * il + 1) * 5120 * 4) as *mut f32,
                    5120,
                );
            }
            let wo = self
                .norm("model.language_model.norm.weight")
                .ok_or("output_norm")?;
            // 행 128 = output_norm(행 127은 L63의 post_ln — 과거 덮어씀 버그)
            std::ptr::copy_nonoverlapping(
                wo.as_ptr(),
                nw128.ptr.add(128 * 5120 * 4) as *mut f32,
                5120,
            );
        }
        self.ctx.flush_buf(&nw128);
        self.ctx.flush_buf(&zeros);
        if let Some(b) = self.batch.as_mut() {
            b.fframe = Some(FFrame {
                xbuf,
                gsnap: None,
                zeros,
                nw128,
                pnr,

                pnrh,
            });
        }
        Ok(())
    }

    /// GDN 상태 전체 스냅샷(스펙 검증 전 — 롤백 보험, plans/121).
    pub fn gdn_state_snapshot(&mut self) -> Result<(), String> {
        // 지연 할당(첫 스냅샷 시 151MB — 일반 프리필은 미할당).
        let need_alloc = self
            .batch
            .as_ref()
            .and_then(|b| b.fframe.as_ref())
            .is_some_and(|f| f.gsnap.is_none());
        if need_alloc
            && let Some(gsnap) = self.ctx.alloc_host_cached(n_gdn_bytes()).ok()
            && let Some(b) = self.batch.as_mut()
            && let Some(f) = b.fframe.as_mut()
        {
            f.gsnap = Some(gsnap);
        }
        let dst = {
            let b = self.batch.as_ref().ok_or("batch")?;
            let f = b.fframe.as_ref().ok_or("fframe")?;
            f.gsnap.as_ref().ok_or("gsnap")?.buf
        };
        let (gs, gr, gn) = {
            let b = self.batch.as_ref().ok_or("batch")?;
            let g = b.gframe.as_ref().ok_or("gframe")?;
            (g.gstate.buf, g.gring.buf, g.gstate.bytes)
        };
        let rn = 48 * 3 * 10240 * 4;
        self.ctx.copy_dev(&[
            (gs, 0, dst, 0, gn as u64),
            (gr, 0, dst, gn as u64, rn as u64),
        ])
    }

    /// 스냅샷 복원(발산 라운드 — kvc는 재실행이 정확히 덮으므로 미복원).
    pub fn gdn_state_restore(&mut self) -> Result<(), String> {
        let dst = {
            let b = self.batch.as_ref().ok_or("batch")?;
            let f = b.fframe.as_ref().ok_or("fframe")?;
            f.gsnap.as_ref().ok_or("gsnap")?.buf
        };
        let (gs, gr, gn) = {
            let b = self.batch.as_ref().ok_or("batch")?;
            let g = b.gframe.as_ref().ok_or("gframe")?;
            (g.gstate.buf, g.gring.buf, g.gstate.bytes)
        };
        let rn = 48 * 3 * 10240 * 4;
        // 영역 소스: gstate←gsnap+0, gring←gsnap+gn(4703aa3 정리가 첫 소스를
        // gn 오프셋으로 잘못 바꿔 gstate에 gring 복사+OOB → DEVICE_LOST였음).
        self.ctx.copy_dev(&[
            (dst, 0, gs, 0, gn as u64),
            (dst, gn as u64, gr, 0, rn as u64),
        ])
    }

    /// 잔차 버퍼 포인터 — 호출자가 임베딩 행을 직접 기록한다.
    pub fn frame_x_ptr(&mut self) -> Result<*mut f32, String> {
        self.fframe_init()?;
        Ok(self
            .batch
            .as_ref()
            .and_then(|b| b.fframe.as_ref())
            .ok_or("fframe")?
            .xbuf
            .ptr as *mut f32)
    }

    pub fn frame_x_flush(&mut self, t_rows: usize) -> Result<(), String> {
        let ff = self
            .batch
            .as_ref()
            .and_then(|b| b.fframe.as_ref())
            .ok_or("fframe")?;
        self.ctx.flush_range(&ff.xbuf, t_rows * 5120 * 4);
        Ok(())
    }

    /// norm_resid 디스패치: xn=norm(x+ab)·w[row] → xtb, x+=ab → xbuf 제자리.
    /// ab에 zeros를 주면 사전 전용(잔차 0).
    pub fn frame_norm_resid(
        &mut self,
        w_row: usize,
        t_rows: usize,
        ab: ash::vk::Buffer,
    ) -> Result<(), String> {
        let ff = self
            .batch
            .as_ref()
            .and_then(|b| b.fframe.as_ref())
            .ok_or("fframe")?;
        let xtb = self.batch.as_ref().ok_or("batch")?.xtb.buf;
        let ds = self.ctx.fresh_ds_for(&ff.pnr, 4)?;
        self.ctx
            .bind_bufs(ds, &[ff.xbuf.buf, ff.nw128.buf, ab, xtb]);
        let push: Vec<u8> = [t_rows as u32, (w_row * 5120) as u32]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        crate::rawvk::context::site::set_tag("e3_norm_resid");
        self.ctx.run_rw(
            ff.pnr.pl,
            ds,
            ff.pnr.pipe,
            &push,
            t_rows as u32,
            1,
            1,
            &[ff.nw128.buf, ab],
            &[xtb, ff.xbuf.buf],
        )?;
        Ok(())
    }
}
