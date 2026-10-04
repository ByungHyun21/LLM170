//! [GDN 모듈 — vk 측정 원장 2026-10-03]
//! scan v3(전면 LDS): 53.4→~9ms·corr 0.999998(T512 0.9545→재작성 후)
//! conv/l2perm/gate: 정합 1.4e-4급 종단 · 역순열 scatter 방향(원장)
use super::resident::TrellisResident;
use super::resident::{BATCH_TMAX, BatchScratch, GdnFrame};

impl TrellisResident {
    pub fn gdn_chain_run(
        &mut self,
        t_rows: usize,
        xn: &[f32],
        qkv: &[f32],
        z: &[f32],
    ) -> Result<Vec<f32>, String> {
        self.gdn_frame_init()?;
        let (yb0_b, yb1_b) = {
            let b = self.batch.as_ref().ok_or("batch")?;
            (b.yb[0].buf, b.yb[1].buf)
        };
        {
            let (xptr, qptr, zptr) = {
                let b = self.batch.as_ref().ok_or("batch")?;
                (
                    b.xtb.ptr as *mut f32,
                    b.yb[0].ptr as *mut f32,
                    b.yb[1].ptr as *mut f32,
                )
            };
            unsafe {
                std::ptr::copy_nonoverlapping(xn.as_ptr(), xptr, t_rows * 5120);
                std::ptr::copy_nonoverlapping(qkv.as_ptr(), qptr, t_rows * 10240);
                std::ptr::copy_nonoverlapping(z.as_ptr(), zptr, t_rows * 6144);
            }
        }
        // 상태/ring 제로(layer0 슬라이스)
        {
            let gf = self
                .batch
                .as_ref()
                .and_then(|b| b.gframe.as_ref())
                .ok_or("gframe")?;
            unsafe {
                std::ptr::write_bytes(gf.gstate.ptr, 0, 48 * 16384 * 4);
                std::ptr::write_bytes(gf.gring.ptr, 0, 3 * 10240 * 4);
            }
            self.ctx.flush_range(&gf.gstate, 48 * 16384 * 4);
            self.ctx.flush_range(&gf.gring, 3 * 10240 * 4);
        }
        // 입력 flush
        {
            let b = self.batch.as_ref().ok_or("batch")?;
            self.ctx.flush_range(&b.xtb, t_rows * 5120 * 4);
            self.ctx.flush_range(&b.yb[0], t_rows * 10240 * 4);
            self.ctx.flush_range(&b.yb[1], t_rows * 6144 * 4);
        }
        self.gdn_layer_gpu(0, t_rows, std::ptr::null_mut(), yb0_b, yb1_b)?;
        // gated 판독
        let b = self.batch.as_ref().ok_or("batch")?;
        self.ctx.invalidate_range(&b.xtb, t_rows * 6144 * 4);
        let out =
            unsafe { std::slice::from_raw_parts(b.xtb.ptr as *const f32, t_rows * 6144).to_vec() };
        // 중간 산출(l2perm gbg/conv gqr 선두) 판독 — 단계 격리 진단.
        {
            let gf = self
                .batch
                .as_ref()
                .and_then(|b| b.gframe.as_ref())
                .ok_or("gframe")?;
            self.ctx.invalidate_range(&gf.gbg, t_rows * 96 * 4);
            self.ctx.invalidate_range(&gf.gqr, t_rows * 2048 * 4);
        }
        Ok(out)
    }

    /// 체인 중간 산출 판독(gbg·gqr·gq[L2 q]·go[scan]) — 프로브 진단.
    pub fn gdn_chain_mids(
        &mut self,
        t_rows: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), String> {
        let gf = self
            .batch
            .as_ref()
            .and_then(|b| b.gframe.as_ref())
            .ok_or("gframe")?;
        let bg =
            unsafe { std::slice::from_raw_parts(gf.gbg.ptr as *const f32, t_rows * 96).to_vec() };
        let gqr =
            unsafe { std::slice::from_raw_parts(gf.gqr.ptr as *const f32, t_rows * 2048).to_vec() };
        let _ = &gqr;
        self.ctx.invalidate_range(&gf.gq, t_rows * 2048 * 4);
        self.ctx.invalidate_range(&gf.gk, t_rows * 2048 * 4);
        self.ctx.invalidate_range(&gf.gv, t_rows * 6144 * 4);
        let gq =
            unsafe { std::slice::from_raw_parts(gf.gq.ptr as *const f32, t_rows * 2048).to_vec() };
        let gk =
            unsafe { std::slice::from_raw_parts(gf.gk.ptr as *const f32, t_rows * 2048).to_vec() };
        let gv =
            unsafe { std::slice::from_raw_parts(gf.gv.ptr as *const f32, t_rows * 6144).to_vec() };
        self.ctx.invalidate_range(&gf.go, t_rows * 6144 * 4);
        let go =
            unsafe { std::slice::from_raw_parts(gf.go.ptr as *const f32, t_rows * 6144).to_vec() };
        Ok((bg, gq, gk, gv, go))
    }

    /// layer0 체인 상수 판독(cw/ab/alog/dtb/nw) — 프로브 미러용.
    pub fn gdn_chain_consts(
        &mut self,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), String> {
        self.gdn_frame_init()?;
        let gf = self
            .batch
            .as_ref()
            .and_then(|b| b.gframe.as_ref())
            .ok_or("gframe")?;
        // 호스트 기록 상수는 그대로 판독(배치 후 GPU 미기입).
        unsafe {
            let cw = std::slice::from_raw_parts(gf.cw.ptr as *const f32, 48 * 10240 * 4).to_vec();
            let ab =
                std::slice::from_raw_parts(gf.ab.ptr as *const f32, 48 * 2 * 48 * 5120).to_vec();
            let alog = std::slice::from_raw_parts(gf.alog.ptr as *const f32, 48 * 48).to_vec();
            let dtb = std::slice::from_raw_parts(gf.dtb.ptr as *const f32, 48 * 48).to_vec();
            let nw = std::slice::from_raw_parts(gf.nw.ptr as *const f32, 48 * 128).to_vec();
            Ok((cw, ab, alog, dtb, nw))
        }
    }

    /// GPU GDN 상태 유효 플래그 전체 무효화 — fresh 시퀀스(슬롯 교체·기준
    /// 재생) 프리필 전 호출. GPU 상태가 다른 시퀀스의 잔류일 수 있어 강제 재업로드.
    pub fn gdn_st_invalidate_all(&mut self) {
        for v in self.gdn_st_valid.iter_mut() {
            *v = false;
        }
    }

    /// GPU 프레임(GDN/어텐션) 활성 여부 — 벌크 sync 가드(T≤8 CPU 경로 보호).
    pub fn gpu_frames_active(&self) -> bool {
        self.batch
            .as_ref()
            .is_some_and(|b| b.gframe.is_some() || b.aframe.is_some())
    }

    /// 배치 스크래치 지연 초기화.
    pub(crate) fn ensure_batch(&mut self) -> Result<(), String> {
        if self.batch.is_some() {
            return Ok(());
        }
        if llm170_diag::dump::opts().key("exl3_lindbg") {
            self.ctx.dump_mem_types();
        }
        let p1 = self
            .ctx
            .pipeline_pipes(include_bytes!("../spv/exl3_had_in_t.spv"), 3, 8)?;
        let p1f = self
            .ctx
            .pipeline_pipes(include_bytes!("../spv/exl3_had_in_tf32.spv"), 3, 8)?;
        let p2 = self
            .ctx
            .pipeline_pipes(include_bytes!("../spv/exl3_gemm2.spv"), 3, 16)?;
        let p2d = self
            .ctx
            .pipeline_pipes(include_bytes!("../spv/exl3_gemm2d.spv"), 5, 20)?;
        let p3d = self
            .ctx
            .pipeline_pipes(include_bytes!("../spv/exl3_had_out_td.spv"), 3, 20)?;
        let p3 = self
            .ctx
            .pipeline_pipes(include_bytes!("../spv/exl3_had_out_t.spv"), 3, 12)?;
        let p4t = self
            .ctx
            .pipeline_pipes(include_bytes!("../spv/exl3_ffn_ew_t.spv"), 3, 4)?;
        // 모델 전체 max — 부분 적재(load_keep) 축소 시에도 풀 적재와 동일 크기.
        // 2026-10-04 ffn 크래시: 필터 max_n 17408에서 248320급 기록(yb/sb/x2t
        // 경계)이 35MB 버퍼를 넘어 힙 파탄 — 풀 적재 508MB가 가리던 잠재 OOB.
        let max_k = self.max_k_g;
        let max_n = self.max_n_g;
        // CPU가 직접 읽/쓰는 버퍼(xtb 스테이징·yb 판독)는 호스트 RAM(캐시됨) —
        // APU 커브아웃 매핑 판독은 무캐시로 T×n MB급 판독이 ~300MB/s에
        // 걸려 프리필 병목이었다(2026-10-03 계측: lin_gu 34ms/층 중 대부분).
        // ah/sb/x2t는 GPU 전용 — 커브아웃 유지.
        let xtb = self.ctx.alloc_host_cached(BATCH_TMAX * max_k * 4)?;
        let ah1 = self.ctx.alloc(BATCH_TMAX * max_k * 2)?;
        let ah2 = self.ctx.alloc(BATCH_TMAX * max_k * 2)?;
        let ah3 = self.ctx.alloc(BATCH_TMAX * max_k * 2)?;
        let y1 = self.ctx.alloc_host_cached(BATCH_TMAX * max_n * 4)?;
        let y2 = self.ctx.alloc_host_cached(BATCH_TMAX * max_n * 4)?;
        let y3 = self.ctx.alloc_host_cached(BATCH_TMAX * max_n * 4)?;
        let sb = self.ctx.alloc(BATCH_TMAX * max_n * 4)?;
        let x2t = self.ctx.alloc(BATCH_TMAX * max_n * 2)?;
        self.batch = Some(BatchScratch {
            aframe: None,
            fframe: None,
            xtb,
            ah: [ah1, ah2, ah3],
            yb: [y1, y2, y3],
            sb,
            x2t,
            p1,
            p1f,
            p2,
            p2d,
            p3d,
            p3,
            p4t,
            gframe: None,
        });
        Ok(())
    }

    /// 배치 선형 1개 체인(had_in→gemm→had_out_t) — begin_batch 내부 전용.
    /// f32_in=true: x_src를 f32 [T][k]로 직독(had_in_tf32) — 스테이징 경로.
    /// false: f16 쌍팩(ew_t 출력 → down 레그).
    /// had_in 생략 체인(gemm2+had_out만) — norm_resid_had가 ah를 이미 기록한
    /// 소비용(plans/121 메가융합 1호). ah는 호출자가 지정(ah[slot]).
    pub(crate) fn chain_gemmonly(
        &mut self,
        li: usize,
        t_rows: u32,
        ah: ash::vk::Buffer,
        yb: ash::vk::Buffer,
    ) -> Result<(), String> {
        let l = &self.linears[li].1;
        let (k, n, krate) = (l.k, l.n, l.krate);
        let (tre_b, svh_b) = (l.tre.buf, l.svh.buf);
        let sb_b = self.batch.as_ref().ok_or("batch scratch 미초기화")?.sb.buf;
        {
            let b = self.batch.as_ref().ok_or("batch scratch")?;
            let ktiles = (k / 16) as u32;
            let ntiles = (n / 16) as u32;
            let ds = self.ctx.fresh_ds_for(&b.p2, 3)?;
            self.ctx.bind_bufs(ds, &[ah, tre_b, sb_b]);
            let push: Vec<u8> = [ktiles, ntiles, krate, t_rows]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            crate::rawvk::context::site::set_tag("e3_gemm2");
            self.ctx.run_rw(
                b.p2.pl,
                ds,
                b.p2.pipe,
                &push,
                ((n / 64) as u32).max(1),
                t_rows.div_ceil(128),
                1,
                &[ah, tre_b],
                &[sb_b],
            )?;
        }
        {
            let b = self.batch.as_ref().ok_or("batch scratch")?;
            let ds = self.ctx.fresh_ds_for(&b.p3, 3)?;
            self.ctx.bind_bufs(ds, &[sb_b, svh_b, yb]);
            let push: Vec<u8> = [(n / 128) as u32, 1u32, n as u32]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            crate::rawvk::context::site::set_tag("e3_had_out_t");
            self.ctx.run_rw(
                b.p3.pl,
                ds,
                b.p3.pipe,
                &push,
                (n / 128) as u32,
                t_rows,
                1,
                &[sb_b, svh_b],
                &[yb],
            )?;
        }
        Ok(())
    }

    pub(crate) fn chain_batch_one(
        &mut self,
        x_src: ash::vk::Buffer,
        li: usize,
        t_rows: u32,
        ah: ash::vk::Buffer,
        yb: ash::vk::Buffer,
        f32_in: bool,
    ) -> Result<(), String> {
        let l = &self.linears[li].1;
        let (k, n, krate) = (l.k, l.n, l.krate);
        let (suh_b, tre_b, svh_b) = (l.suh.buf, l.tre.buf, l.svh.buf);
        let b = self.batch.as_ref().ok_or("batch scratch 미초기화")?;
        let sb_b = b.sb.buf;

        // had_in: x_src × suh → ah (행별 WHT, grid=(k/128, T))
        {
            let (pl, pipe, tag) = if f32_in {
                (&b.p1f.pl, b.p1f.pipe, "e3_had_in_tf32")
            } else {
                (&b.p1.pl, b.p1.pipe, "e3_had_in_t")
            };
            let ds = self
                .ctx
                .fresh_ds_for(if f32_in { &b.p1f } else { &b.p1 }, 3)?;

            self.ctx.bind_bufs(ds, &[x_src, suh_b, ah]);
            let push: Vec<u8> = [(k / 128) as u32, k as u32]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            crate::rawvk::context::site::set_tag(tag);
            self.ctx.run_rw(
                *pl,
                ds,
                pipe,
                &push,
                (k / 128) as u32,
                t_rows,
                1,
                &[x_src, suh_b],
                &[ah],
            )?;
        }

        // gemm2(coopmat): ah × tre → sb ([T][n], nseg=1 — k-분할 없음,
        // grid=(n/64, ceil(T/128))). II-3(reconstruct+hgemm)은 측정 부정
        // (2026-10-03: 56.2 vs 71.2 t/s — 재구성 트래픽+디스패치가 이득
        // 상쇄, BN=128 gemm2는 이미 디코드를 128토큰에 상각) — 원장 참조。
        {
            let ktiles = (k / 16) as u32;
            let ntiles = (n / 16) as u32;
            let ds = self.ctx.fresh_ds_for(&b.p2, 3)?;
            self.ctx.bind_bufs(ds, &[ah, tre_b, sb_b]);
            let push: Vec<u8> = [ktiles, ntiles, krate, t_rows]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            crate::rawvk::context::site::set_tag("e3_gemm2");
            self.ctx.run_rw(
                b.p2.pl,
                ds,
                b.p2.pipe,
                &push,
                ((n / 64) as u32).max(1),
                t_rows.div_ceil(128),
                1,
                &[ah, tre_b],
                &[sb_b],
            )?;
        }

        // had_out_t: sb 환원 × svh → yb (grid=(n/128, T), nseg=1 — gemm2)
        {
            let ds = self.ctx.fresh_ds_for(&b.p3, 3)?;
            self.ctx.bind_bufs(ds, &[sb_b, svh_b, yb]);
            let push: Vec<u8> = [(n / 128) as u32, 1u32, n as u32]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            crate::rawvk::context::site::set_tag("e3_had_out_t");
            self.ctx.run_rw(
                b.p3.pl,
                ds,
                b.p3.pipe,
                &push,
                (n / 128) as u32,
                t_rows,
                1,
                &[sb_b, svh_b],
                &[yb],
            )?;
        }
        Ok(())
    }

    /// T-배치 공유 입력 다중 선형(스테이징): y_i[t] = x[t] @ W_i^T (i ≤ 3).
    /// 입력은 호출자가 stage_f32() 버퍼에 [T][k]로 미리 기록했어야 한다
    /// (생산 par_rows가 직접 기록 — CPU f16 변환·업로드 복사 제거, 원장 #3).
    /// 체인 직렬·슬롯 분리·동기 1회 — 디코드 linear_triple의 배치판.
    pub fn linear_batch_multi_staged(
        &mut self,
        keys: &[&str],
        t_rows: usize,
    ) -> Result<Vec<Vec<f32>>, String> {
        if keys.is_empty() || keys.len() > 3 {
            return Err(format!(
                "linear_batch_multi_staged: keys {}개 (1..=3)",
                keys.len()
            ));
        }
        if t_rows == 0 || t_rows > BATCH_TMAX {
            return Err(format!("linear_batch: T={t_rows} 상한 {BATCH_TMAX} 위반"));
        }
        let mut idxs = Vec::with_capacity(keys.len());
        for key in keys {
            idxs.push(self.find_linear(key)?);
        }
        let k0 = self.linears[idxs[0]].1.k;
        for &li in &idxs[1..] {
            if self.linears[li].1.k != k0 {
                return Err("linear_batch_multi_staged: 입력 차원 불일치".to_string());
            }
        }
        self.ensure_batch()?;

        // 진단 분해(원장 89 키: exl3_lindbg) — flush/디스패치/대기/판독.
        static LINDBG_N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let lindbg = llm170_diag::dump::opts().key("exl3_lindbg")
            && LINDBG_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 24;
        let lt0 = std::time::Instant::now();

        // 캐시(비결합) xtb 쓰기 → GPU 가시화 flush — 실사용 구간만(T×k).
        {
            let k0 = self.linears[idxs[0]].1.k;
            let b = self.batch.as_ref().ok_or("batch scratch")?;
            self.ctx.flush_range(&b.xtb, t_rows * k0 * 4);
        }
        let lt1 = std::time::Instant::now();

        let xtb = self.batch.as_ref().ok_or("batch scratch")?.xtb.buf;
        self.ctx.begin_batch()?;
        for (slot, &li) in idxs.iter().enumerate() {
            let b = self.batch.as_ref().ok_or("batch scratch")?;
            self.chain_batch_one(xtb, li, t_rows as u32, b.ah[slot].buf, b.yb[slot].buf, true)?;
        }
        let lt2 = std::time::Instant::now();
        self.ctx.end_batch_wait()?;
        self.ctx.wait_pending()?; // DBUF 비동기 잔여 배출
        let lt3 = std::time::Instant::now();

        // 캐시(비결합) yb — GPU 쓰기 판독 전 인밸리데이트(실사용 구간 T×n).
        let b = self.batch.as_ref().ok_or("batch scratch")?;
        for (slot, &li) in idxs.iter().enumerate() {
            let n = self.linears[li].1.n;
            self.ctx.invalidate_range(&b.yb[slot], t_rows * n * 4);
        }

        let mut outs = Vec::with_capacity(idxs.len());
        for (slot, &li) in idxs.iter().enumerate() {
            let n = self.linears[li].1.n;
            let b = self.batch.as_ref().ok_or("batch scratch")?;
            let mut y = vec![0f32; t_rows * n];
            // SAFETY: end_batch_wait 후 매핑 판독 — t_rows*n ≤ TMAX*max_n.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    b.yb[slot].ptr as *const f32,
                    y.as_mut_ptr(),
                    t_rows * n,
                );
            }
            outs.push(y);
        }
        if lindbg {
            let lt4 = std::time::Instant::now();
            eprintln!(
                "[lindbg] {:52} fl {:6.2} disp {:6.2} wait {:6.2} rd {:6.2} ms",
                keys[0],
                (lt1 - lt0).as_secs_f64() * 1e3,
                (lt2 - lt1).as_secs_f64() * 1e3,
                (lt3 - lt2).as_secs_f64() * 1e3,
                (lt4 - lt3).as_secs_f64() * 1e3,
            );
        }
        Ok(outs)
    }

    /// T-배치 단일 선형(스테이징) — stage_f32 버퍼의 [T][k]를 소비.
    pub fn linear_batch_staged(&mut self, key: &str, t_rows: usize) -> Result<Vec<f32>, String> {
        self.linear_batch_multi_staged(&[key], t_rows)?
            .into_iter()
            .next()
            .ok_or_else(|| "linear_batch_staged: 결과 없음".to_string())
    }

    /// 스테이징 버퍼 포인터 — 호출자의 병렬 생산자가 [T][k] f32 행을 직접
    /// 기록한다(행 스트라이드 = 해당 선형의 k). 기록 후 staged 호출로 소비.
    pub fn stage_f32(&mut self) -> Result<*mut f32, String> {
        self.ensure_batch()?;
        Ok(self.batch.as_ref().ok_or("batch scratch")?.xtb.ptr as *mut f32)
    }

    pub fn ffn_trio_batch(
        &mut self,
        key_g: &str,
        key_u: &str,
        key_d: &str,
        t_rows: usize,
    ) -> Result<Vec<f32>, String> {
        self.ffn_trio_impl(key_g, key_u, key_d, t_rows, false, false)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ffn_trio_impl(
        &mut self,
        key_g: &str,
        key_u: &str,
        key_d: &str,
        t_rows: usize,
        gpu_input: bool,
        preah: bool,
    ) -> Result<Vec<f32>, String> {
        if t_rows == 0 || t_rows > BATCH_TMAX {
            return Err(format!("ffn_trio_batch: T={t_rows} 상한 {BATCH_TMAX} 위반"));
        }
        let ig = self.find_linear(key_g)?;
        let iu = self.find_linear(key_u)?;
        let id = self.find_linear(key_d)?;
        let (kg, ng, nd) = (
            self.linears[ig].1.k,
            self.linears[ig].1.n,
            self.linears[id].1.n,
        );
        if self.linears[iu].1.k != kg || self.linears[id].1.k != ng {
            return Err(format!("{key_g}/{key_u}/{key_d}: FFN 차원 불일치"));
        }
        self.ensure_batch()?;
        if !gpu_input {
            let b = self.batch.as_ref().ok_or("batch scratch")?;
            self.ctx.flush_range(&b.xtb, t_rows * kg * 4);
        }
        let b0 = self.batch.as_ref().ok_or("batch scratch")?;
        let (xtb, ah0, yb0, ah1, yb1, ah2, yb2, x2t_b, p4t_pl, p4t_pipe) = (
            b0.xtb.buf,
            b0.ah[0].buf,
            b0.yb[0].buf,
            b0.ah[1].buf,
            b0.yb[1].buf,
            b0.ah[2].buf,
            b0.yb[2].buf,
            b0.x2t.buf,
            b0.p4t.pl,
            b0.p4t.pipe,
        );
        // ew_t 디스크립터는 체인 전에 선점 확보(borrow 분리 — 복사본 사용).
        let ds4 = self.ctx.fresh_ds_for(&b0.p4t, 3)?;
        self.ctx.begin_batch()?;
        // preah: 선행 norm_resid_had가 ah0(gate)/ah1(up) 기록 — had_in 생략.
        if preah {
            self.chain_gemmonly(ig, t_rows as u32, ah0, yb0)?;
            self.chain_gemmonly(iu, t_rows as u32, ah1, yb1)?;
        } else {
            self.chain_batch_one(xtb, ig, t_rows as u32, ah0, yb0, true)?;
            self.chain_batch_one(xtb, iu, t_rows as u32, ah1, yb1, true)?;
        }
        // ew_t: yb1(g) × yb2(u) → x2t(f16 쌍팩 [T][ng/2])
        self.ctx.bind_bufs(ds4, &[yb0, yb1, x2t_b]);
        let push4 = (ng as u32).to_le_bytes().to_vec();
        crate::rawvk::context::site::set_tag("e3_ffn_ew_t");
        self.ctx.run_rw(
            p4t_pl,
            ds4,
            p4t_pipe,
            &push4,
            ng.div_ceil(512) as u32,
            t_rows as u32,
            1,
            &[yb0, yb1],
            &[x2t_b],
        )?;
        // down: x2t(f16) 직독 — had_in_t(f16) 레그.
        self.chain_batch_one(x2t_b, id, t_rows as u32, ah2, yb2, false)?;
        self.ctx.end_batch_wait()?;
        self.ctx.wait_pending()?;
        if gpu_input {
            return Ok(Vec::new());
        }
        {
            let b = self.batch.as_ref().ok_or("batch scratch")?;
            self.ctx.invalidate_range(&b.yb[2], t_rows * nd * 4);
        }
        let mut y = vec![0f32; t_rows * nd];
        {
            let b = self.batch.as_ref().ok_or("batch scratch")?;
            // SAFETY: end_batch_wait 후 매핑 판독 — t_rows*nd ≤ TMAX*max_n.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    b.yb[2].ptr as *const f32,
                    y.as_mut_ptr(),
                    t_rows * nd,
                );
            }
        }
        Ok(y)
    }
    /// T-배치 다중 선형(GPU 상주 변형) — 결과를 yb 슬롯에 남긴다(판독 없음).
    /// plans/121 F1: GDN 프레임이 yb0/yb1을 직접 소비.
    /// 반환: 슬롯별 (buffer handle, n) — 소비자 커널이 바인딩에 사용.
    pub fn linear_batch_multi_gpu(
        &mut self,
        keys: &[&str],
        t_rows: usize,
    ) -> Result<Vec<(ash::vk::Buffer, usize)>, String> {
        if keys.is_empty() || keys.len() > 3 {
            return Err(format!(
                "linear_batch_multi_gpu: keys {}개 (1..=3)",
                keys.len()
            ));
        }
        if t_rows == 0 || t_rows > BATCH_TMAX {
            return Err(format!("linear_batch_gpu: T={t_rows} 상한 위반"));
        }
        let mut idxs = Vec::with_capacity(keys.len());
        for key in keys {
            idxs.push(self.find_linear(key)?);
        }
        self.ensure_batch()?;
        {
            let k0 = self.linears[idxs[0]].1.k;
            let b = self.batch.as_ref().ok_or("batch scratch")?;
            self.ctx.flush_range(&b.xtb, t_rows * k0 * 4);
        }
        let xtb = self.batch.as_ref().ok_or("batch scratch")?.xtb.buf;
        self.ctx.begin_batch()?;
        for (slot, &li) in idxs.iter().enumerate() {
            let b = self.batch.as_ref().ok_or("batch scratch")?;
            self.chain_batch_one(xtb, li, t_rows as u32, b.ah[slot].buf, b.yb[slot].buf, true)?;
        }
        self.ctx.end_batch_wait()?;
        self.ctx.wait_pending()?;
        // 판독 없음 — yb에 잔류. 소비자가 invalidate 후 판독하거나 GPU 직독.
        let b = self.batch.as_ref().ok_or("batch scratch")?;
        Ok(idxs
            .iter()
            .enumerate()
            .map(|(slot, &li)| (b.yb[slot].buf, self.linears[li].1.n))
            .collect())
    }

    /// GDN 프레임 초기화(plans/121 F1) — 상수 업로드 1회 + 스크래치/상태 할당.
    /// GDN 층 수는 64층 중 il%4!=3 → 48층.
    pub fn gdn_frame_init(&mut self) -> Result<(), String> {
        if self.batch.as_ref().is_some_and(|b| b.gframe.is_some()) {
            return Ok(());
        }
        self.ensure_batch()?;
        let n_gdn = self.n_layers - self.n_layers / 4; // 48

        // 파이프라인 4종
        let pgc = self
            .ctx
            .pipeline_pipes(include_bytes!("../spv/exl3_gdn_conv.spv"), 6, 8)?;
        let pgl = self
            .ctx
            .pipeline_pipes(include_bytes!("../spv/exl3_gdn_l2perm.spv"), 11, 20)?;
        let pgs = self
            .ctx
            .pipeline_pipes(include_bytes!("../spv/exl3_gdn_scan.spv"), 6, 20)?;
        let pgg = self
            .ctx
            .pipeline_pipes(include_bytes!("../spv/exl3_gdn_gate.spv"), 4, 20)?;

        // 스크래치(TMAX 기준)
        let gq = self.ctx.alloc_host_cached(BATCH_TMAX * 2048 * 4)?;
        let gk = self.ctx.alloc_host_cached(BATCH_TMAX * 2048 * 4)?;
        let gv = self.ctx.alloc_host_cached(BATCH_TMAX * 6144 * 4)?;
        let gbg = self.ctx.alloc_host_cached(BATCH_TMAX * 96 * 4)?;
        let go = self.ctx.alloc_host_cached(BATCH_TMAX * 6144 * 4)?;
        let gqr = self.ctx.alloc_host_cached(BATCH_TMAX * 2048 * 4)?;
        let gkr = self.ctx.alloc_host_cached(BATCH_TMAX * 2048 * 4)?;
        let gvr = self.ctx.alloc_host_cached(BATCH_TMAX * 6144 * 4)?;

        // 상수(48층 분) — 트레이리던트 norms에서 직접 복사.
        let ab = self.ctx.alloc_host_cached(n_gdn * 2 * 48 * 5120 * 4)?;
        let cw = self.ctx.alloc_host_cached(n_gdn * 10240 * 4 * 4)?;
        let alog = self.ctx.alloc_host_cached(n_gdn * 48 * 4)?;
        let dtb = self.ctx.alloc_host_cached(n_gdn * 48 * 4)?;
        let nw = self.ctx.alloc_host_cached(n_gdn * 128 * 4)?;

        // 상태(48층 분, GPU)
        let gring = self.ctx.alloc_host_cached(n_gdn * 3 * 10240 * 4)?;
        let gstate = self.ctx.alloc_host_cached(n_gdn * 48 * 16384 * 4)?;

        // 상수 업로드 — 각 GDN 층(il%4!=3)의 norms를 GPU 버퍼에.
        unsafe {
            let abp = ab.ptr as *mut f32;
            let cwp = cw.ptr as *mut f32;
            let alp = alog.ptr as *mut f32;
            let dtp = dtb.ptr as *mut f32;
            let nwp = nw.ptr as *mut f32;
            let mut g = 0usize;
            for il in 0..self.n_layers {
                if il % 4 == 3 {
                    continue;
                } // full attention
                let lp = format!("model.language_model.layers.{il}.linear_attn");
                let a_p = self
                    .norm(&format!("{lp}.in_proj_a.weight"))
                    .ok_or("a_proj")?;
                let b_p = self
                    .norm(&format!("{lp}.in_proj_b.weight"))
                    .ok_or("b_proj")?;
                let c_w = self.norm(&format!("{lp}.conv1d.weight")).ok_or("conv_w")?;
                let a_l = self.norm(&format!("{lp}.A_log")).ok_or("A_log")?;
                let d_b = self.norm(&format!("{lp}.dt_bias")).ok_or("dt_bias")?;
                let n_w = self.norm(&format!("{lp}.norm.weight")).ok_or("norm_w")?;
                std::ptr::copy_nonoverlapping(a_p.as_ptr(), abp.add(g * 2 * 48 * 5120), 48 * 5120);
                std::ptr::copy_nonoverlapping(
                    b_p.as_ptr(),
                    abp.add(g * 2 * 48 * 5120 + 48 * 5120),
                    48 * 5120,
                );
                std::ptr::copy_nonoverlapping(c_w.as_ptr(), cwp.add(g * 10240 * 4), 10240 * 4);
                std::ptr::copy_nonoverlapping(a_l.as_ptr(), alp.add(g * 48), 48);
                std::ptr::copy_nonoverlapping(d_b.as_ptr(), dtp.add(g * 48), 48);
                std::ptr::copy_nonoverlapping(n_w.as_ptr(), nwp.add(g * 128), 128);
                g += 1;
            }
        }
        self.ctx.flush_buf(&ab);
        self.ctx.flush_buf(&cw);
        self.ctx.flush_buf(&alog);
        self.ctx.flush_buf(&dtb);
        self.ctx.flush_buf(&nw);

        if let Some(b) = self.batch.as_mut() {
            b.gframe = Some(GdnFrame {
                gq,
                gk,
                gv,
                gbg,
                go,
                gqr,
                gkr,
                gvr,
                ab,
                cw,
                alog,
                dtb,
                nw,
                gring,
                gstate,
                pgc,
                pgl,
                pgs,
                pgg,
            });
        }
        Ok(())
    }

    /// GDN 층 전체 GPU 상주 처리(plans/121 F1) — qkv/yb0, z/yb1에서 xtb(gated)까지.
    /// 호출 전 linear_batch_multi_gpu(qkv+z)가 yb 슬롯에 결과를 남겼어야 한다.
    /// gdn_il은 GDN 층 인덱스(0..48, il%4!=3 순서).
    pub fn gdn_layer_gpu(
        &mut self,
        gdn_il: usize,
        t_rows: usize,
        _xn_ptr: *mut f32,
        yb0: ash::vk::Buffer,
        yb1: ash::vk::Buffer,
    ) -> Result<(), String> {
        let gf = match self.batch.as_ref().and_then(|b| b.gframe.as_ref()) {
            Some(g) => g,
            None => return Err("gdn_frame 미초기화".into()),
        };

        self.ctx.begin_batch()?;

        // ① conv: yb0(qkv) → gqr/gkr/gvr + ring 갱신
        {
            let ds = self.ctx.fresh_ds_for(&gf.pgc, 6)?;
            self.ctx.bind_bufs(
                ds,
                &[
                    yb0,
                    gf.cw.buf,
                    gf.gring.buf,
                    gf.gqr.buf,
                    gf.gkr.buf,
                    gf.gvr.buf,
                ],
            );
            let push: Vec<u8> = [t_rows as u32, gdn_il as u32]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            crate::rawvk::context::site::set_tag("e3_gdn_conv");
            self.ctx.run_rw(
                gf.pgc.pl,
                ds,
                gf.pgc.pipe,
                &push,
                80,
                1,
                1,
                &[yb0],
                &[gf.gqr.buf, gf.gkr.buf, gf.gvr.buf, gf.gring.buf],
            )?;
        }

        // ② l2perm: gqr/gkr/gvr + xn(xtb) + 상수 → gq/gk/gv/gbg
        {
            let ds = self.ctx.fresh_ds_for(&gf.pgl, 11)?;
            // xn은 xtb 버퍼 — 호출자가 stage_f32()에 기록했음.
            let xtb = self.batch.as_ref().ok_or("batch")?.xtb.buf;
            // 상수는 gdn_il 슬라이스… 마찬가지로 전체 버퍼 바인딩 + 커널에 층 오프셋 필요.
            // TODO: l2perm 커널에 gdn_il push 추가(상수 오프셋).
            self.ctx.bind_bufs(
                ds,
                &[
                    gf.gqr.buf,
                    gf.gkr.buf,
                    gf.gvr.buf,
                    xtb,
                    gf.ab.buf,
                    gf.alog.buf,
                    gf.dtb.buf,
                    gf.gq.buf,
                    gf.gk.buf,
                    gf.gv.buf,
                    gf.gbg.buf,
                ],
            );
            let push: Vec<u8> = [t_rows as u32, 16u32, 48u32, 128u32, gdn_il as u32]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            crate::rawvk::context::site::set_tag("e3_gdn_l2perm");
            self.ctx.run_rw(
                gf.pgl.pl,
                ds,
                gf.pgl.pipe,
                &push,
                48,
                t_rows as u32,
                1,
                &[
                    gf.gqr.buf,
                    gf.gkr.buf,
                    gf.gvr.buf,
                    xtb,
                    gf.ab.buf,
                    gf.alog.buf,
                    gf.dtb.buf,
                ],
                &[gf.gq.buf, gf.gk.buf, gf.gv.buf, gf.gbg.buf],
            )?;
        }

        // 실입력 캡처(모듈 격리 디버그 — plans/121 F2): l2perm 직후 gq/gk/gv/gbg.
        if gdn_il == 0
            && let Some(path) =
                llm170_diag::flag::val("LLM170_EXL3_SCAN_CAP").map(|s| s.to_string())
        {
            let g2 = self.batch.as_ref().and_then(|b| b.gframe.as_ref());
            if let Some(g) = g2 {
                let n_q = t_rows * 2048;
                let n_bg = t_rows * 96;
                self.ctx.invalidate_range(&g.gq, n_q * 4);
                self.ctx.invalidate_range(&g.gk, n_q * 4);
                self.ctx.invalidate_range(&g.gv, t_rows * 6144 * 4);
                self.ctx.invalidate_range(&g.gbg, n_bg * 4);
                // SAFETY: 호스트 매핑 버퍼 직독 — invalidate 직후 유효.
                let qs = unsafe { std::slice::from_raw_parts(g.gq.ptr as *const f32, n_q) };
                let ks = unsafe { std::slice::from_raw_parts(g.gk.ptr as *const f32, n_q) };
                let vs =
                    unsafe { std::slice::from_raw_parts(g.gv.ptr as *const f32, t_rows * 6144) };
                let bgs = unsafe { std::slice::from_raw_parts(g.gbg.ptr as *const f32, n_bg) };
                let mut buf = Vec::with_capacity((n_q * 2 + t_rows * 6144 + n_bg) * 4);
                // SAFETY: 위 직독 슬라이스의 원시 바이트 재해석 — 같은 라이프타임.
                unsafe {
                    for s in [qs, ks, vs, bgs] {
                        buf.extend_from_slice(std::slice::from_raw_parts(
                            s.as_ptr() as *const u8,
                            s.len() * 4,
                        ));
                    }
                }
                std::fs::write(&path, &buf).map_err(|e| format!("scan cap: {e}"))?;
                eprintln!("  [scancap] L0 t={t_rows} → {path}");
            }
        }
        // ③ scan: gq/gk/gv/gbg + gstate → go
        {
            let ds = self.ctx.fresh_ds_for(&gf.pgs, 6)?;
            self.ctx.bind_bufs(
                ds,
                &[
                    gf.gq.buf,
                    gf.gk.buf,
                    gf.gv.buf,
                    gf.gbg.buf,
                    gf.gstate.buf,
                    gf.go.buf,
                ],
            );
            let push: Vec<u8> = [t_rows as u32, 16u32, 48u32, 128u32, gdn_il as u32]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            crate::rawvk::context::site::set_tag("e3_gdn_scan");
            self.ctx.run_rw(
                gf.pgs.pl,
                ds,
                gf.pgs.pipe,
                &push,
                48,
                1,
                1,
                &[gf.gq.buf, gf.gk.buf, gf.gv.buf, gf.gbg.buf, gf.gstate.buf],
                &[gf.gstate.buf, gf.go.buf],
            )?;
        }

        // ④ gate: go + yb1(z) + nw → xtb(gated)
        {
            let ds = self.ctx.fresh_ds_for(&gf.pgg, 4)?;
            let xtb = self.batch.as_ref().ok_or("batch")?.xtb.buf;
            self.ctx.bind_bufs(ds, &[gf.go.buf, yb1, gf.nw.buf, xtb]);
            let push: Vec<u8> = [t_rows as u32, 16u32, 48u32, 128u32, gdn_il as u32]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            crate::rawvk::context::site::set_tag("e3_gdn_gate");
            self.ctx.run_rw(
                gf.pgg.pl,
                ds,
                gf.pgg.pipe,
                &push,
                48,
                t_rows as u32,
                1,
                &[gf.go.buf, yb1, gf.nw.buf],
                &[xtb],
            )?;
        }

        self.ctx.end_batch_wait()?;
        self.ctx.wait_pending()?;
        // 체인 입출력 캡처(plans/121 워크플로 — GDN 비선형 전체 프로브용):
        #[allow(clippy::collapsible_if, clippy::needless_borrows_for_generic_args)]
        // 입력 yb0(qkv)/yb1(z) + 게이트 출력 xtb를 gdn_il==0에서 파일로.
        if gdn_il == 0 {
            if let Some(path) = llm170_diag::flag::val("LLM170_EXL3_CHAINCAP") {
                let gf2 = self.batch.as_ref().and_then(|b| b.gframe.as_ref());
                if gf2.is_some() {
                    let b2 = self.batch.as_ref().ok_or("batch")?;
                    let nq = t_rows * 10240;
                    let nz = t_rows * 6144;
                    let nx = t_rows * 6144;
                    self.ctx.invalidate_range(&b2.yb[0], nq * 4);
                    self.ctx.invalidate_range(&b2.yb[1], nz * 4);
                    self.ctx.invalidate_range(&b2.xtb, nx * 4);
                    // SAFETY: end_batch_wait 후 매핑 판독.
                    let (q, z, x) = unsafe {
                        (
                            std::slice::from_raw_parts(b2.yb[0].ptr as *const f32, nq),
                            std::slice::from_raw_parts(b2.yb[1].ptr as *const f32, nz),
                            std::slice::from_raw_parts(b2.xtb.ptr as *const f32, nx),
                        )
                    };
                    let mut buf = Vec::with_capacity((nq + nz + nx) * 4);
                    // SAFETY: 위 직독 슬라이스의 바이트 재해석 — 동일 수명.
                    unsafe {
                        for sl in [q, z, x] {
                            buf.extend_from_slice(std::slice::from_raw_parts(
                                sl.as_ptr() as *const u8,
                                sl.len() * 4,
                            ));
                        }
                    }
                    std::fs::write(&path, buf).map_err(|e| format!("chain cap: {e}"))?;
                    eprintln!("  [chaincap] L0 t={t_rows} → {path}");
                }
            }
        }
        Ok(())
    }

    /// GDN 상태 GPU→CPU 동기화(plans/121 F1) — GPU 배치 후 차기 디코드 정합.
    /// states는 [48*16384] f32, conv는 [3*10240] f32 다운로드.
    /// 다운로드 후 CPU 사본이 권위 — 이후 CPU 디코드가 상태를 진화시킬 수
    /// 있으므로 GPU 유효 플래그는 해제한다(plans/121 F2 스케줄).
    pub fn gdn_state_sync(
        &mut self,
        gdn_il: usize,
        states: &mut [f32],
        conv: &mut [f32],
    ) -> Result<(), String> {
        let gf = match self.batch.as_ref().and_then(|b| b.gframe.as_ref()) {
            Some(g) => g,
            None => return Err("gdn_frame ì±ì ìí ëê¸°í ì¤í¨".into()),
        };
        let st_bytes = 48 * 16384 * 4;
        let ring_bytes = 3 * 10240 * 4;
        self.ctx
            .invalidate_range_at(&gf.gstate, gdn_il * st_bytes, st_bytes);
        self.ctx
            .invalidate_range_at(&gf.gring, gdn_il * ring_bytes, ring_bytes);
        let st_off = gdn_il * 48 * 16384;
        let ring_off = gdn_il * 3 * 10240;
        unsafe {
            std::ptr::copy_nonoverlapping(
                gf.gstate.ptr.add(st_off * 4) as *const f32,
                states.as_mut_ptr(),
                48 * 16384,
            );
            std::ptr::copy_nonoverlapping(
                gf.gring.ptr.add(ring_off * 4) as *const f32,
                conv.as_mut_ptr(),
                3 * 10240,
            );
        }
        if gdn_il < self.gdn_st_valid.len() {
            self.gdn_st_valid[gdn_il] = false;
        }
        Ok(())
    }

    /// xtb 범위 인밸리데이트(GPU gate 출력 → 호스트 가시화, plans/121 F1).
    pub fn invalidate_xtb(&mut self, bytes: usize) {
        if let Some(b) = self.batch.as_ref() {
            self.ctx.invalidate_range(&b.xtb, bytes);
        }
    }

    /// GDN 상태 CPU→GPU 업로드(plans/121 F1) — 초기화/리셋용.
    pub fn gdn_state_upload(
        &mut self,
        gdn_il: usize,
        states: &[f32],
        conv: &[f32],
    ) -> Result<(), String> {
        // 프리필 스케줄(plans/121 F2): GPU 상태가 이미 유효하면 업로드 스킵.
        if *self.gdn_st_valid.get(gdn_il).unwrap_or(&false) {
            return Ok(());
        }
        let gf = match self.batch.as_ref().and_then(|b| b.gframe.as_ref()) {
            Some(g) => g,
            None => return Err("gdn_frame 미초기화".into()),
        };
        let st_off = gdn_il * 48 * 16384;
        let ring_off = gdn_il * 3 * 10240;
        unsafe {
            std::ptr::copy_nonoverlapping(
                states.as_ptr(),
                gf.gstate.ptr.add(st_off * 4) as *mut f32,
                48 * 16384,
            );
            std::ptr::copy_nonoverlapping(
                conv.as_ptr(),
                gf.gring.ptr.add(ring_off * 4) as *mut f32,
                3 * 10240,
            );
        }
        let st_bytes = 48 * 16384 * 4;
        let ring_bytes = 3 * 10240 * 4;
        self.ctx
            .flush_range_at(&gf.gstate, gdn_il * st_bytes, st_bytes);
        self.ctx
            .flush_range_at(&gf.gring, gdn_il * ring_bytes, ring_bytes);
        if gdn_il < self.gdn_st_valid.len() {
            self.gdn_st_valid[gdn_il] = true;
        }
        Ok(())
    }
}
// 마커 gdnb
// 마커 gdnp
