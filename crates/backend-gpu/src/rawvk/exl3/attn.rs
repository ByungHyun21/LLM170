//! [어텐션 모듈 — vk 측정 원장 2026-10-03]
//! fwd3(WG256·t블록4 k/v공유·LDS트리): 산술 5.3e-7·8.2ms/층 — pp512 +13% 기여
use super::resident::{AttnFrame, BATCH_TMAX, TrellisResident};

impl TrellisResident {
    pub fn attn_frame_init(&mut self) -> Result<(), String> {
        if self.batch.as_ref().is_some_and(|b| b.aframe.is_some()) {
            return Ok(());
        }
        self.ensure_batch()?;
        let pa = self
            .ctx
            .pipeline_pipes(include_bytes!("../spv/exl3_attn_prep.spv"), 9, 8)?;
        let pf3 = self
            .ctx
            .pipeline_pipes(include_bytes!("../spv/exl3_attn_fwd3.spv"), 6, 8)?;
        let pbuf = self.ctx.alloc_host_cached(16)?;
        unsafe {
            std::ptr::write_bytes(pbuf.ptr, 0, 16);
        }
        self.ctx.flush_buf(&pbuf);
        let kkc = self.ctx.alloc_host_cached(16 * 1024 * 1024 * 4)?;
        let vkc = self.ctx.alloc_host_cached(16 * 1024 * 1024 * 4)?;
        let qh = self.ctx.alloc_host_cached(BATCH_TMAX * 6144 * 4)?;
        let qnws = self.ctx.alloc_host_cached(16 * 256 * 4)?;
        let knws = self.ctx.alloc_host_cached(16 * 256 * 4)?;
        // 노름 업로드 — 어텐션 층(il%4==3)의 q_norm/k_norm.
        unsafe {
            std::ptr::write_bytes(kkc.ptr, 0, 16 * 1024 * 1024 * 4);
            std::ptr::write_bytes(vkc.ptr, 0, 16 * 1024 * 1024 * 4);
            let mut ai = 0usize;
            for il in 0..self.n_layers {
                if il % 4 != 3 {
                    continue;
                }
                let lp = format!("model.language_model.layers.{il}.self_attn");
                let qw = self.norm(&format!("{lp}.q_norm.weight")).ok_or("q_norm")?;
                let kw = self.norm(&format!("{lp}.k_norm.weight")).ok_or("k_norm")?;
                std::ptr::copy_nonoverlapping(
                    qw.as_ptr(),
                    qnws.ptr.add(ai * 256 * 4) as *mut f32,
                    256,
                );
                std::ptr::copy_nonoverlapping(
                    kw.as_ptr(),
                    knws.ptr.add(ai * 256 * 4) as *mut f32,
                    256,
                );
                ai += 1;
            }
        }
        self.ctx.flush_buf(&qnws);
        self.ctx.flush_buf(&knws);
        self.ctx.flush_buf(&kkc);
        self.ctx.flush_buf(&vkc);
        if let Some(b) = self.batch.as_mut() {
            b.aframe = Some(AttnFrame {
                pbuf,
                kkc,
                vkc,
                qh,
                qnws,
                knws,
                pa,
                pf3,
            });
        }
        Ok(())
    }

    /// 디버그: gstate 선두 8값 판독(재생 NaN 국소화 — plans/121 tg).
    pub fn debug_gstate_head(&mut self) -> Result<[f32; 8], String> {
        let g = self
            .batch
            .as_ref()
            .and_then(|b| b.gframe.as_ref())
            .ok_or("gframe")?;
        self.ctx.invalidate_range(&g.gstate, 32);
        let mut out = [0f32; 8];
        unsafe {
            std::ptr::copy_nonoverlapping(g.gstate.ptr as *const f32, out.as_mut_ptr(), 8);
        }
        Ok(out)
    }

    /// pos0 매개변수 버퍼 기록(재생 경로 — 호스트가 라운드마다 갱신).
    pub fn attn_set_pos(&mut self, pos0: u32) -> Result<(), String> {
        let af = self
            .batch
            .as_ref()
            .and_then(|b| b.aframe.as_ref())
            .ok_or("aframe")?;
        unsafe {
            *(af.pbuf.ptr as *mut u32) = pos0;
        }
        self.ctx.flush_range(&af.pbuf, 4);
        Ok(())
    }

    /// 어텐션 층 GPU 경로(plans/121 F2b): prep(q/k norm+rope+KV 적립) →
    /// fwd3(인과 어텐션+게이트) → xtb 직접 기록. yb0/1/2 = q‖gate/k/v GEMM 출력.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_layer_gpu(
        &mut self,
        attn_il: usize,
        t_rows: usize,
        pos0: u32,
        yb0: ash::vk::Buffer,
        yb1: ash::vk::Buffer,
        yb2: ash::vk::Buffer,
    ) -> Result<(), String> {
        self.attn_frame_init()?;
        let af = self
            .batch
            .as_ref()
            .and_then(|b| b.aframe.as_ref())
            .ok_or("aframe")?;
        let xtb = self.batch.as_ref().ok_or("batch")?.xtb.buf;
        let (kkc, vkc, qh, qnws, knws, pbuf) = (
            af.kkc.buf,
            af.vkc.buf,
            af.qh.buf,
            af.qnws.buf,
            af.knws.buf,
            af.pbuf.buf,
        );
        // pos0 → 매개변수 버퍼(재생 지원 — 커널은 pp[0] 판독, 푸시는 불변).
        unsafe {
            *(af.pbuf.ptr as *mut u32) = pos0;
        }
        self.ctx.flush_range(&af.pbuf, 4);
        self.ctx.begin_batch()?;
        {
            let ds = self.ctx.fresh_ds_for(&af.pa, 9)?;
            self.ctx
                .bind_bufs(ds, &[yb0, yb1, yb2, qnws, knws, qh, kkc, vkc, pbuf]);
            let push: Vec<u8> = [t_rows as u32, pos0, attn_il as u32]
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect();
            crate::rawvk::context::site::set_tag("e3_attn_prep");
            self.ctx.run_rw(
                af.pa.pl,
                ds,
                af.pa.pipe,
                &push,
                t_rows as u32,
                28,
                1,
                &[yb0, yb1, yb2, qnws, knws],
                &[qh, kkc, vkc],
            )?;
        }
        {
            let ds = self.ctx.fresh_ds_for(&af.pf3, 6)?;
            self.ctx.bind_bufs(ds, &[qh, kkc, vkc, yb0, xtb, pbuf]);
            let push: Vec<u8> = [t_rows as u32, pos0, attn_il as u32]
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect();
            let nq = t_rows.div_ceil(4);
            crate::rawvk::context::site::set_tag("e3_attn_fwd3");
            self.ctx.run_rw(
                af.pf3.pl,
                ds,
                af.pf3.pipe,
                &push,
                nq as u32,
                24,
                1,
                &[qh, kkc, vkc, yb0],
                &[xtb],
            )?;
        }
        self.ctx.end_batch_wait()?;
        self.ctx.wait_pending()?;
        Ok(())
    }

    /// 프리필 종료 시 KV 캐시 GPU→CPU 벌크 동기(차기 디코드 정합).
    pub fn attn_kv_sync(&mut self, attn_il: usize, kv_len: usize) -> Result<[Vec<f32>; 2], String> {
        let b = self.batch.as_ref().ok_or("batch")?;
        let a = b.aframe.as_ref().ok_or("aframe")?;
        let n = kv_len * 1024;
        self.ctx
            .invalidate_range_at(&a.kkc, attn_il * 1024 * 1024 * 4, n * 4);
        self.ctx
            .invalidate_range_at(&a.vkc, attn_il * 1024 * 1024 * 4, n * 4);
        let k = unsafe {
            std::slice::from_raw_parts(a.kkc.ptr.add(attn_il * 1024 * 1024 * 4) as *const f32, n)
                .to_vec()
        };
        let v = unsafe {
            std::slice::from_raw_parts(a.vkc.ptr.add(attn_il * 1024 * 1024 * 4) as *const f32, n)
                .to_vec()
        };
        Ok([k, v])
    }
}

impl TrellisResident {}
