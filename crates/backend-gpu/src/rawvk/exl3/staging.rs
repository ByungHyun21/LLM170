//! [스테이징·GEMM API — vk 측정 원장 2026-10-03]
//! gemm2 coopmat(BN128): 8.4TF · 메가융합 3호(dual입력) — T-적응 스케줄
use super::resident::TrellisResident;

impl TrellisResident {
    pub fn frame_norm_resid_had(
        &mut self,
        w_row: usize,
        t_rows: usize,
        ab: ash::vk::Buffer,
        suh1: ash::vk::Buffer,
        suh2: ash::vk::Buffer,
    ) -> Result<(), String> {
        let ff = self
            .batch
            .as_ref()
            .and_then(|b| b.fframe.as_ref())
            .ok_or("fframe")?;
        let b = self.batch.as_ref().ok_or("batch")?;
        let (xtb, ah0, ah1) = (b.xtb.buf, b.ah[0].buf, b.ah[1].buf);
        let ds = self.ctx.fresh_ds_for(&ff.pnrh, 8)?;
        self.ctx.bind_bufs(
            ds,
            &[ff.xbuf.buf, ff.nw128.buf, ab, xtb, suh1, suh2, ah0, ah1],
        );
        let push: Vec<u8> = [t_rows as u32, (w_row * 5120) as u32]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        crate::rawvk::context::site::set_tag("e3_norm_resid_had");
        self.ctx.run_rw(
            ff.pnrh.pl,
            ds,
            ff.pnrh.pipe,
            &push,
            t_rows as u32,
            1,
            1,
            &[ff.nw128.buf, ab, suh1, suh2],
            &[xtb, ff.xbuf.buf, ah0, ah1],
        )?;
        Ok(())
    }

    /// 듀얼 GEMM(메가융합 3호): ah0/ah1(선행 norm_had)을 단일 gemm2d로
    /// 병합 환원 — GEMM 커널 수 절반. 결과 yb[0]/yb[1].
    pub fn linear_pair_dual(
        &mut self,
        keys: [&str; 2],
        t_rows: usize,
    ) -> Result<[(ash::vk::Buffer, usize); 2], String> {
        let i1 = self.find_linear(keys[0])?;
        let i2 = self.find_linear(keys[1])?;
        let (k1, n1, kr1, kt1) = {
            let l = &self.linears[i1].1;
            (l.k, l.n, l.krate, l.n / 16)
        };
        let (k2, n2, kt2) = {
            let l = &self.linears[i2].1;
            (l.k, l.n, l.n / 16)
        };
        if k1 != k2 || kt1 % 4 != 0 || kt2 % 4 != 0 {
            return Err(format!(
                "linear_pair_dual: {}/{} k 불일치 또는 n 비64배수",
                keys[0], keys[1]
            ));
        }
        let (kr2v, n2b) = {
            let l = &self.linears[i2].1;
            (l.krate, l.n)
        };
        let _ = n2b;
        // 혼합정밀 아카이브(8/48층 qkv r=4 vs z r=5/3 — gemmd L5 재현 8.6e0):
        // krate 불일치 쌍은 듀얼 단일-K 디코드 불가 → preah 2체인 폴백.
        // A5(plans/129): 폴백 원장 등재 — 아카이브 구조상 확정 경로라 매 forward
        // 적립된다(정상 동작이지만 원장으로 프로덕션 혼입 여부가 판별된다).
        if kr1 != kr2v {
            llm170_diag::fb::incr("exl3-krate-preah");
            static ONCE_KRATE: std::sync::Once = std::sync::Once::new();
            ONCE_KRATE.call_once(|| {
                eprintln!(
                    "[fb] exl3-krate-preah: {}/{} krate({kr1}/{kr2v}) 불일치 — preah 2체인",
                    keys[0], keys[1]
                );
            });
            self.ensure_batch()?;
            self.ctx.begin_batch()?;
            for (slot, li) in [(0usize, i1), (1usize, i2)] {
                let (ah, yb) = {
                    let b = self.batch.as_ref().ok_or("batch")?;
                    (b.ah[slot].buf, b.yb[slot].buf)
                };
                self.chain_gemmonly(li, t_rows as u32, ah, yb)?;
            }
            self.ctx.end_batch_wait()?;
            self.ctx.wait_pending()?;
            let b = self.batch.as_ref().ok_or("batch")?;
            return Ok([(b.yb[0].buf, n1), (b.yb[1].buf, n2)]);
        }
        self.ensure_batch()?;
        self.ctx.begin_batch()?;
        {
            let b = self.batch.as_ref().ok_or("batch")?;
            let (ah0, ah1, tre1, tre2, sb) = (
                b.ah[0].buf,
                b.ah[1].buf,
                self.linears[i1].1.tre.buf,
                self.linears[i2].1.tre.buf,
                b.sb.buf,
            );
            let kk = self.linears[i1].1.krate;
            let ds = self.ctx.fresh_ds_for(&b.p2d, 5)?;
            self.ctx.bind_bufs(ds, &[ah0, tre1, ah1, tre2, sb]);
            let push: Vec<u8> = [(k1 / 16) as u32, kt1 as u32, kt2 as u32, kk, t_rows as u32]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            crate::rawvk::context::site::set_tag("e3_gemm2d");
            self.ctx.run_rw(
                b.p2d.pl,
                ds,
                b.p2d.pipe,
                &push,
                ((n1 + n2) / 64) as u32,
                t_rows.div_ceil(64) as u32,
                1,
                &[ah0, tre1, ah1, tre2],
                &[sb],
            )?;
        }
        // had_out_td 2회 — n_off로 슬래브 분리.
        for (slot, li, n_off, ntiles) in [(0usize, i1, 0usize, kt1), (1usize, i2, n1, kt2)] {
            let b = self.batch.as_ref().ok_or("batch")?;
            let (sb, svh, yb) = (b.sb.buf, self.linears[li].1.svh.buf, b.yb[slot].buf);
            let ds = self.ctx.fresh_ds_for(&b.p3d, 3)?;
            self.ctx.bind_bufs(ds, &[sb, svh, yb]);
            let n_this = ntiles * 16;
            let push: Vec<u8> = [
                (n_this / 128) as u32,
                1u32,
                ((kt1 + kt2) * 16) as u32,
                n_off as u32,
                n_this as u32,
            ]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
            crate::rawvk::context::site::set_tag("e3_had_out_td");
            self.ctx.run_rw(
                b.p3d.pl,
                ds,
                b.p3d.pipe,
                &push,
                (n_this / 128) as u32,
                t_rows as u32,
                1,
                &[sb, svh],
                &[yb],
            )?;
        }
        self.ctx.end_batch_wait()?;
        self.ctx.wait_pending()?;
        let b = self.batch.as_ref().ok_or("batch")?;
        Ok([(b.yb[0].buf, n1), (b.yb[1].buf, n2)])
    }

    /// ah 사전 기록 전제 2선형 배치(gemm2+had_out만) — norm_resid_had 소비용.
    pub fn linear_pair_preah(
        &mut self,
        keys: &[&str],
        t_rows: usize,
    ) -> Result<Vec<(ash::vk::Buffer, usize)>, String> {
        if keys.len() != 2 {
            return Err(format!("linear_pair_preah: keys {}개 (2 고정)", keys.len()));
        }
        let mut idxs = Vec::with_capacity(2);
        for key in keys {
            idxs.push(self.find_linear(key)?);
        }
        self.ensure_batch()?;
        self.ctx.begin_batch()?;
        for (slot, &li) in idxs.iter().enumerate() {
            let (ah, yb) = {
                let b = self.batch.as_ref().ok_or("batch")?;
                (b.ah[slot].buf, b.yb[slot].buf)
            };
            self.chain_gemmonly(li, t_rows as u32, ah, yb)?;
        }
        self.ctx.end_batch_wait()?;
        self.ctx.wait_pending()?;
        let b = self.batch.as_ref().ok_or("batch")?;
        Ok(idxs
            .iter()
            .enumerate()
            .map(|(slot, &li)| (b.yb[slot].buf, self.linears[li].1.n))
            .collect())
    }

    pub fn frame_zeros_buf(&mut self) -> Result<ash::vk::Buffer, String> {
        Ok(self
            .batch
            .as_ref()
            .and_then(|b| b.fframe.as_ref())
            .ok_or("fframe")?
            .zeros
            .buf)
    }

    /// 마지막 행 판독(로그릿용) — end_outer 후 호출.
    pub fn frame_read_xtb_row(&mut self, t_rows: usize) -> Result<Vec<f32>, String> {
        let b = self.batch.as_ref().ok_or("batch")?;
        let base = (t_rows - 1) * 5120;
        self.ctx.invalidate_range_at(&b.xtb, base * 4, 5120 * 4);
        let p = unsafe {
            std::slice::from_raw_parts(b.xtb.ptr.add(base * 4) as *const f32, 5120).to_vec()
        };
        Ok(p)
    }

    /// 디버그: xbuf 선두 행 판독(잔차 검증).
    pub fn debug_xbuf_row(&mut self) -> Result<Vec<f32>, String> {
        let ff = self
            .batch
            .as_ref()
            .and_then(|b| b.fframe.as_ref())
            .ok_or("fframe")?;
        self.ctx.invalidate_range_at(&ff.xbuf, 0, 5120 * 4);
        Ok(unsafe { std::slice::from_raw_parts(ff.xbuf.ptr as *const f32, 5120).to_vec() })
    }

    /// 디버그: xtb 선두 행 판독(FFN 입력 xn 검증).
    pub fn debug_xtb_row(&mut self) -> Result<Vec<f32>, String> {
        let b = self.batch.as_ref().ok_or("batch")?;
        self.ctx.invalidate_range_at(&b.xtb, 0, 5120 * 4);
        Ok(unsafe { std::slice::from_raw_parts(b.xtb.ptr as *const f32, 5120).to_vec() })
    }

    /// 디버그: yb[2] 선두 행 판독(FFN out row0 검증).
    pub fn debug_yb2_row(&mut self) -> Result<Vec<f32>, String> {
        let b = self.batch.as_ref().ok_or("batch")?;
        self.ctx.invalidate_range_at(&b.yb[2], 0, 5120 * 4);
        Ok(unsafe { std::slice::from_raw_parts(b.yb[2].ptr as *const f32, 5120).to_vec() })
    }

    /// 디버그: yb[0] 선두 행 판독(L0 out row0 검증).
    pub fn debug_yb0_row(&mut self, t_rows: usize) -> Result<Vec<f32>, String> {
        let b = self.batch.as_ref().ok_or("batch")?;
        let base = (t_rows - 1) * 5120;
        self.ctx.invalidate_range_at(&b.yb[0], base * 4, 5120 * 4);
        Ok(unsafe {
            std::slice::from_raw_parts(b.yb[0].ptr.add(base * 4) as *const f32, 5120).to_vec()
        })
    }

    /// 열린 외부 배치 내부용 선형 체인(begin/end 없음 — 프레임 내 lm_head 등).
    pub fn linear_chain_inside(
        &mut self,
        key: &str,
        t_rows: usize,
        slot: usize,
    ) -> Result<(ash::vk::Buffer, usize), String> {
        let li = self.find_linear(key)?;
        let n = self.linears[li].1.n;
        self.ensure_batch()?;
        let (xtb, ah0, yb0) = {
            let b = self.batch.as_ref().ok_or("batch scratch")?;
            (b.xtb.buf, b.ah[slot].buf, b.yb[slot].buf)
        };
        self.chain_batch_one(xtb, li, t_rows as u32, ah0, yb0, true)?;
        Ok((yb0, n))
    }

    /// yb 슬롯에서 t_rows×n 판독(스펙 행별 로짓).
    pub fn read_yb_rows(
        &mut self,
        slot: usize,
        t_rows: usize,
        n: usize,
    ) -> Result<Vec<f32>, String> {
        let b = self.batch.as_ref().ok_or("batch")?;
        self.ctx.invalidate_range(&b.yb[slot], t_rows * n * 4);
        let mut y = vec![0f32; t_rows * n];
        // SAFETY: end_outer 후 매핑 판독.
        unsafe {
            std::ptr::copy_nonoverlapping(b.yb[slot].ptr as *const f32, y.as_mut_ptr(), t_rows * n);
        }
        Ok(y)
    }

    /// xtb 선두 t_rows행 판독(스펙 행별 노름 입력).
    pub fn read_xtb_rows(&mut self, t_rows: usize) -> Result<Vec<f32>, String> {
        let b = self.batch.as_ref().ok_or("batch")?;
        self.ctx.invalidate_range(&b.xtb, t_rows * 5120 * 4);
        let mut y = vec![0f32; t_rows * 5120];
        // SAFETY: end_outer 후 매핑 판독.
        unsafe {
            std::ptr::copy_nonoverlapping(b.xtb.ptr as *const f32, y.as_mut_ptr(), t_rows * 5120);
        }
        Ok(y)
    }

    /// xbuf 마지막 행 판독(MTP h 스냅샷) — GPU 갱신 반영(invalidate).
    pub fn frame_read_x_last(&mut self, t_rows: usize) -> Result<Vec<f32>, String> {
        let ff = self
            .batch
            .as_ref()
            .and_then(|b| b.fframe.as_ref())
            .ok_or("fframe")?;
        let base = (t_rows - 1) * 5120;
        self.ctx.invalidate_range_at(&ff.xbuf, base * 4, 5120 * 4);
        Ok(unsafe {
            std::slice::from_raw_parts(ff.xbuf.ptr.add(base * 4) as *const f32, 5120).to_vec()
        })
    }

    /// 단일 선형 체인(판독 없음·flush 없음 — xtb가 GPU 기록 전제, plans/121 프레임).
    /// 결과는 yb[0]에 잔류: (버퍼, n) 반환.
    pub fn linear_chain(
        &mut self,
        key: &str,
        t_rows: usize,
        slot: usize,
    ) -> Result<(ash::vk::Buffer, usize), String> {
        let li = self.find_linear(key)?;
        let n = self.linears[li].1.n;
        self.ensure_batch()?;
        let (xtb, ah0, yb0) = {
            let b = self.batch.as_ref().ok_or("batch scratch")?;
            (b.xtb.buf, b.ah[slot].buf, b.yb[slot].buf)
        };
        self.ctx.begin_batch()?;
        self.chain_batch_one(xtb, li, t_rows as u32, ah0, yb0, true)?;
        self.ctx.end_batch_wait()?;
        self.ctx.wait_pending()?;
        Ok((yb0, n))
    }

    /// FFN 트리오 체인(판독 없음) — 결과 yb[0] 잔류(주의: gate/up이 ah/yb 슬롯
    /// 을 재사용하므로 down은 yb[2]가 아니라 ffn_trio의 슬롯 배정을 그대로
    /// 둔다 — 여기선 down 결과를 yb[0]으로 재배치한다).
    pub fn ffn_trio_chain(
        &mut self,
        key_g: &str,
        key_u: &str,
        key_d: &str,
        t_rows: usize,
    ) -> Result<(ash::vk::Buffer, usize), String> {
        let id = self.find_linear(key_d)?;
        let nd = self.linears[id].1.n;
        self.ffn_trio_impl(key_g, key_u, key_d, t_rows, true, false)?;
        let b = self.batch.as_ref().ok_or("batch")?;
        Ok((b.yb[2].buf, nd))
    }

    /// FFN 트리오 preah 변형(메가융합 2호) — ah0/ah1은 선행 norm_resid_had가
    /// gate/up suh로 기록. down 체인(f16 leg)은 불변.
    pub fn ffn_trio_preah(
        &mut self,
        key_g: &str,
        key_u: &str,
        key_d: &str,
        t_rows: usize,
    ) -> Result<(ash::vk::Buffer, usize), String> {
        let id = self.find_linear(key_d)?;
        let nd = self.linears[id].1.n;
        self.ffn_trio_impl(key_g, key_u, key_d, t_rows, true, true)?;
        let b = self.batch.as_ref().ok_or("batch")?;
        Ok((b.yb[2].buf, nd))
    }
}
