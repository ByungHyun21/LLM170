//! vkacc::qsa — QSA 인덱서·선택 어텐션 계열. (plans/90 B1b: gemv.rs 순수 이동)

use super::*;

impl llm170_core::matmul::GraphCapture for VkAcc {}
impl llm170_core::matmul::QsaOps for VkAcc {
    /// 상주 KV 풀 — 워터마크 규약은 hip과 동일(순차 적립/접두어 되감기 허용).
    /// 적립은 디바이스 간 복사(copy_rows 판) — 프레임 k/v 버퍼에서 풀로.
    fn qsa_kv_dev(
        &self,
        full_idx: usize,
        seq: usize,
        k: u64,
        v: u64,
        t: usize,
        pos0: usize,
        n_kv: usize,
        hd: usize,
    ) -> Result<(u64, u64), String> {
        let ctx_len = self.qsa_ctx.load(std::sync::atomic::Ordering::Relaxed);
        if ctx_len == 0 {
            return Err("vk qsa_kv_dev: ctx_len 미주입".into());
        }
        let bytes = ctx_len * n_kv * hd * 4;
        let need_grow;
        {
            let mut m = self.qsa_pools.lock();
            let e = m.entry((full_idx, seq)).or_insert_with(|| {
                (
                    VkBuf { buf: vk::Buffer::null(), ptr: std::ptr::null_mut(), bytes: 0, mem: vk::DeviceMemory::null() },
                    VkBuf { buf: vk::Buffer::null(), ptr: std::ptr::null_mut(), bytes: 0, mem: vk::DeviceMemory::null() },
                    VkBuf { buf: vk::Buffer::null(), ptr: std::ptr::null_mut(), bytes: 0, mem: vk::DeviceMemory::null() },
                    VkBuf { buf: vk::Buffer::null(), ptr: std::ptr::null_mut(), bytes: 0, mem: vk::DeviceMemory::null() },
                    0,
                )
            });
            crate::common::qsa::wm_advance(&mut e.4, pos0, t)
                .map_err(|e2| format!("vk qsa_kv_dev: {e2}"))?;
            need_grow = e.0.bytes < bytes;
        }
        let mut ctx = self.ctx.lock();
        if need_grow {
            let mut m = self.qsa_pools.lock();
            let e = m.get_mut(&(full_idx, seq)).unwrap();
            // plans/84 B: 호스트 가시 — qsa_host_rebuild가 직접 판독한다.
            e.0 = ctx.alloc_host(bytes)?;
            e.1 = ctx.alloc_host(bytes)?;
        }
        let (kb, vb) = {
            let m = self.qsa_pools.lock();
            let e = m.get(&(full_idx, seq)).unwrap();
            (e.0.buf, e.1.buf)
        };
        let ksrc = self.fbuf(k)?;
        let vsrc = self.fbuf(v)?;
        let p = self.pipeline(&mut ctx, Slot::CopyRows)?;
        let n_f = t * n_kv * hd;
        let soff = (pos0 * n_kv * hd) as u32;
        let ds_k = ctx.bind_ds(&p, &[ksrc, kb])?;
        ctx.run(p.pl, ds_k, p.pipe, &push_u32s(&[n_f as u32, 0u32, soff]), (n_f as u32).div_ceil(256), 1, 1)?;
        let ds_v = ctx.bind_ds(&p, &[vsrc, vb])?;
        ctx.run(p.pl, ds_v, p.pipe, &push_u32s(&[n_f as u32, 0u32, soff]), (n_f as u32).div_ceil(256), 1, 1)?;
        Ok((kb.as_raw(), vb.as_raw()))
    }

    /// 상주 인덱서 풀 — ik 적립 + 완성 블록의 블록키(norm+rope) 갱신.
    fn qsa_idx_append_dev(
        &self,
        full_idx: usize,
        seq: usize,
        ik: u64,
        t: usize,
        pos0: usize,
        idx_dim: usize,
        r: usize,
        ikw: &[f32],
        cs_idx: &[f32],
        eps: f32,
    ) -> Result<(), String> {
        if r == 0 || idx_dim != 128 {
            return Err(format!("vk qsa_idx_append: 미지원 형상 r={r} idx_dim={idx_dim}"));
        }
        let ctx_len = self.qsa_ctx.load(std::sync::atomic::Ordering::Relaxed);
        if ctx_len == 0 {
            return Err("vk qsa_idx_append: ctx_len 미주입".into());
        }
        let nb_max = ctx_len / r + 1;
        {
            let mut m = self.qsa_pools.lock();
            // plans/85 §2: entry()로 생성+실제 갱신 — 종전 `&mut get_mut().map()
            // .unwrap_or(0)`는 임시값에 써서 워터마크가 영구 반영되지 않았고
            // (kv_dev가 대신 갱신해 온 것), 엔트리 없으면 아래 get_mut().unwrap()
            // 이 패닉했다(np 디코드: sel_dev가 kv_dev보다 먼저 append 호출).
            let e = m
                .entry((full_idx, seq))
                .or_insert_with(|| (vkbuf_null(), vkbuf_null(), vkbuf_null(), vkbuf_null(), 0));
            crate::common::qsa::wm_advance(&mut e.4, pos0, t)
                .map_err(|e2| format!("vk qsa_idx_append: {e2}"))?;
        }
        // idx_k 풀은 kv 풀과 별도 용량 — 필요시 성장(간단 재할당).
        let mut ctx = self.ctx.lock();
        let need = (ctx_len * idx_dim * 4, nb_max * idx_dim * 4);
        {
            let mut m = self.qsa_pools.lock();
            let e = m.get_mut(&(full_idx, seq)).unwrap();
            if e.2.bytes < need.0 {
                e.2 = crate::rawvk::context::site::scope("qsa_pool", || ctx.alloc_host(need.0))?;
            }
            if e.3.bytes < need.1 {
                e.3 = crate::rawvk::context::site::scope("qsa_pool", || ctx.alloc_host(need.1))?;
            }
        }
        let (ikb, bkb) = {
            let m = self.qsa_pools.lock();
            let e = m.get(&(full_idx, seq)).unwrap();
            (e.2.buf, e.3.buf)
        };
        let iksrc = self.fbuf(ik)?;
        let p = self.pipeline(&mut ctx, Slot::CopyRows)?;
        let n_f = t * idx_dim;
        let soff = (pos0 * idx_dim) as u32;
        let ds = ctx.bind_ds(&p, &[iksrc, ikb])?;
        ctx.run(p.pl, ds, p.pipe, &push_u32s(&[n_f as u32, 0u32, soff]), (n_f as u32).div_ceil(256), 1, 1)?;
        let b0 = pos0 / r;
        let b1 = (pos0 + t) / r;
        if b1 > b0 {
            // plans/86 §4 — 상수 업로드 캐시: cs 전체 표는 (ptr,len) 키로 1회
            // (셰이더 cs 인덱싱이 절대 pos 기반이라 접두 복사 불필요), ikw 는
            // 호출부가 매번 새 Vec을 만들어 고정 스크래치에 복사(512B).
            // 종전 매호출 alloc_host 두 개는 블록 완성마다 GTT에 누출했다.
            let cs_b = self.qk_const(&mut ctx, cs_idx)?;
            let ikw_b = {
                let mut g = self.qsa_ikw.lock();
                if !g.as_ref().is_some_and(|b| b.bytes >= ikw.len() * 4) {
                    *g = Some(crate::rawvk::context::site::scope("qsa_const", || ctx.alloc_host((ikw.len() * 4).max(4096)))?);
                }
                g.as_ref().unwrap().clone()
            };
            unsafe { std::ptr::copy_nonoverlapping(ikw.as_ptr(), ikw_b.ptr as *mut f32, ikw.len()) };
            let p2 = self.pipeline(&mut ctx, Slot::FnIdxBk)?;
            let ds2 = ctx.bind_ds(&p2, &[ikb, bkb, ikw_b.buf, cs_b.buf])?;
            // plans/85 §2: 셰이더 PC는 선언순 {eps, b0, r, idx_dim} — 종전
            // [b0,r,idx_dim,eps]는 멤버가 전부 어긋나 idx_dim≠128 조기복귀로
            // 블록키가 한 번도 갱신되지 않았다(항등 선택이라 프리필은 무영향).
            let mut push = eps.to_le_bytes().to_vec();
            push.extend_from_slice(&push_u32s(&[b0 as u32, r as u32, idx_dim as u32]));
            ctx.run(p2.pl, ds2, p2.pipe, &push, (b1 - b0) as u32, 1, 1)?;
        }
        Ok(())
    }

    /// 디바이스 풀 → 호스트 캐시 재구축 — 프리필이 디코딩을 건너뛴 뒤 값
    /// 경로 호스트 선택이 필요할 때 1회. 풀 내용을 그대로 내린다.
    fn qsa_host_rebuild(
        &self,
        full_idx: usize,
        seq: usize,
        pos: usize,
        kv_row: usize,
        kv_k: &mut [f32],
        kv_v: &mut [f32],
        idx_k: &mut [f32],
        bk: &mut [f32],
        r: usize,
        idx_dim: usize,
    ) -> Result<(), String> {
        let _ = pos;
        let m = self.qsa_pools.lock();
        let Some(e) = m.get(&(full_idx, seq)) else {
            return Err("vk qsa_host_rebuild: 풀 없음".into());
        };
        if e.0.bytes < kv_k.len() * 4 || e.2.bytes < idx_k.len() * 4 {
            return Err("vk qsa_host_rebuild: 풀 용량 부족".into());
        }
        // 풀은 호스트 가시(alloc_host) — 프레임 동기 후 직접 판독.
        {
            let mut c = self.ctx.lock();
            if c.batching.load(std::sync::atomic::Ordering::Relaxed) {
                let _ = c.end_batch_wait();
            }
        }
        let kv_floats = kv_k.len();
        unsafe {
            std::ptr::copy_nonoverlapping(e.0.ptr as *const f32, kv_k.as_mut_ptr(), kv_floats);
            std::ptr::copy_nonoverlapping(e.1.ptr as *const f32, kv_v.as_mut_ptr(), kv_v.len());
            std::ptr::copy_nonoverlapping(e.2.ptr as *const f32, idx_k.as_mut_ptr(), idx_k.len());
            let nb = (pos + r - 1) / r;
            std::ptr::copy_nonoverlapping(e.3.ptr as *const f32, bk.as_mut_ptr(), (nb * idx_dim).min(bk.len()));
        }
        let _ = kv_row;
        Ok(())
    }

    /// plans/85 §2 — 디코드(t=1) 선택의 디바이스판: ik 적립+블록키 갱신(기존
    /// qsa_idx_append_dev) → iq norm+rope → 블록 점수 → 순위 → 목록 전개.
    /// 산술은 hip qsa_sel_dev와 동일열(f64 순차 rms, f64 회전, 4누산 도트,
    /// 정수 순위) — 선택 목록이 호스트 top-k와 비트 일치.
    #[allow(clippy::too_many_arguments)]
    fn qsa_sel_dev(
        &self,
        full_idx: usize,
        seq: usize,
        iq: u64,
        ik: u64,
        t: usize,
        pos0: usize,
        idx_heads: usize,
        idx_dim: usize,
        r: usize,
        idx_top_k: usize,
        iqw: &[f32],
        ikw: &[f32],
        cs_idx: &[f32],
        eps: f32,
    ) -> Result<(u64, u64, usize), String> {
        if t != 1 {
            return Err(format!("vk qsa_sel_dev: t={t} (디코드 전용)"));
        }
        if r == 0 {
            return Err("vk qsa_sel_dev: r=0".into());
        }
        if idx_dim != 128 {
            return Err(format!("vk qsa_sel_dev: idx_dim={idx_dim} (128 전용)"));
        }
        let n_past = pos0 + t;
        let n_blocks = n_past / r;
        // (1) ik 적립 + 완성 블록 키 — 기존 구현(워터마크 규약 공유).
        self.qsa_idx_append_dev(full_idx, seq, ik, t, pos0, idx_dim, r, ikw, cs_idx, eps)?;
        // n_sel 산술은 stages::qsa_select 패스 B와 동일(정수 — 무동기).
        let tail_start = n_blocks * r;
        let tail_cnt = n_past - tail_start;
        let width = n_past.min(idx_top_k + r - 1);
        let n_sel = ((width - tail_cnt) / r).min(n_blocks);
        let list_len = n_sel * r + tail_cnt;
        let iqr_bytes = t * idx_heads * idx_dim * 4;
        let scr_bytes = n_blocks.max(1) * 4;
        let sd_bytes = list_len.max(1) * 4;
        let mut ctx = self.ctx.lock();
        // 스크래치 — 필요시 성장 재할당(매 호출 alloc_host 누출 회피).
        {
            let mut g = self.qsa_sel_bufs.lock();
            let need = match g.as_ref() {
                None => true,
                Some(b) => {
                    b.0.bytes < iqr_bytes
                        || b.1.bytes < scr_bytes
                        || b.2.bytes < scr_bytes
                        || b.3.bytes < iqw.len() * 4
                        || b.4.bytes < idx_dim * 4
                        || b.5.bytes < sd_bytes
                }
            };
            if need {
                *g = Some(crate::rawvk::context::site::scope("qsa_sel", || Ok::<_, String>((
                    ctx.alloc_host(iqr_bytes.max(1 << 16))?,
                    ctx.alloc_host(scr_bytes.max(4096))?,
                    ctx.alloc_host(scr_bytes.max(4096))?,
                    ctx.alloc_host((iqw.len() * 4).max(4096))?,
                    ctx.alloc_host((idx_dim * 4).max(1 << 16))?,
                    ctx.alloc_host(sd_bytes.max(1 << 16))?,
                    ctx.alloc_host(8)?,
                )))?);
            }
        }
        let (iqr, scr, flg, iqwb, csb, sdev, ofdev) = {
            let g = self.qsa_sel_bufs.lock();
            let b = g.as_ref().unwrap();
            (b.0.clone(), b.1.clone(), b.2.clone(), b.3.clone(), b.4.clone(), b.5.clone(), b.6.clone())
        };
        // 호스트 상수(매 호출 소량) — iqw 전체, cs는 pos0행 idx_dim.
        unsafe {
            std::ptr::copy_nonoverlapping(iqw.as_ptr(), iqwb.ptr as *mut f32, iqw.len());
            std::ptr::copy_nonoverlapping(
                cs_idx[pos0 * idx_dim..].as_ptr(),
                csb.ptr as *mut f32,
                idx_dim,
            );
        }
        // (2) iq norm+rope — 워크그룹 = (헤드, 토큰), 32스레드.
        let iqb = self.fbuf(iq)?;
        {
            // PC 선언순 {eps, idx_dim} — cs 인덱싱은 업로드 상대(행 y).
            let p = self.pipeline(&mut ctx, Slot::FnIdxQRope)?;
            let ds = ctx.bind_ds(&p, &[iqb, iqr.buf, iqwb.buf, csb.buf])?;
            let mut push = eps.to_le_bytes().to_vec();
            push.extend_from_slice(&push_u32s(&[idx_dim as u32]));
            ctx.run(p.pl, ds, p.pipe, &push, idx_heads as u32, t as u32, 1)?;
        }
        let bkb = {
            let m = self.qsa_pools.lock();
            m.get(&(full_idx, seq))
                .map(|e| e.3.buf)
                .ok_or("vk qsa_sel_dev: bk 풀 없음")?
        };
        if n_blocks > 0 {
            // (3) 블록 점수 — 스레드당 블록.
            let p = self.pipeline(&mut ctx, Slot::FnIdxScore)?;
            let ds = ctx.bind_ds(&p, &[iqr.buf, bkb, scr.buf])?;
            let push = push_u32s(&[n_blocks as u32, idx_heads as u32, idx_dim as u32]);
            ctx.run(p.pl, ds, p.pipe, &push, (n_blocks as u32).div_ceil(256), 1, 1)?;
            // (4) 순위 — 결정적 top-k(점수 내림, 인덱스 오름).
            let p = self.pipeline(&mut ctx, Slot::FnIdxRank)?;
            let ds = ctx.bind_ds(&p, &[scr.buf, flg.buf])?;
            let push = push_u32s(&[n_blocks as u32, n_sel as u32]);
            ctx.run(p.pl, ds, p.pipe, &push, (n_blocks as u32).div_ceil(256), 1, 1)?;
        }
        // (5) 목록 전개 — 단일 워크그룹(무공유메모리 판).
        {
            let p = self.pipeline(&mut ctx, Slot::FnIdxExpand)?;
            let ds = ctx.bind_ds(&p, &[flg.buf, sdev.buf, ofdev.buf])?;
            let push = push_u32s(&[n_blocks as u32, n_sel as u32, r as u32, n_past as u32]);
            ctx.run(p.pl, ds, p.pipe, &push, 1, 1, 1)?;
        }
        Ok((sdev.buf.as_raw(), ofdev.buf.as_raw(), list_len))
    }

    /// plans/73 SELCHECK 진단 — qsa_sel_dev가 만든 목록을 호스트로 내린다.
    fn qsa_sel_readback(
        &self,
        sel_idx: u64,
        sel_off: u64,
        list_len: usize,
    ) -> Result<(Vec<u32>, Vec<u32>), String> {
        {
            let mut c = self.ctx.lock();
            if c.batching.load(std::sync::atomic::Ordering::Relaxed) {
                let _ = c.end_batch_wait();
            }
        }
        let g = self.qsa_sel_bufs.lock();
        let b = g.as_ref().ok_or("vk qsa_sel_readback: 스크래치 없음")?;
        if b.5.buf.as_raw() != sel_idx as u64 || b.6.buf.as_raw() != sel_off as u64 {
            return Err("vk qsa_sel_readback: 핸들 불일치(스크래치 재성장)".into());
        }
        let mut si = vec![0u32; list_len];
        unsafe { std::ptr::copy_nonoverlapping(b.5.ptr as *const u32, si.as_mut_ptr(), list_len) };
        let mut so = vec![0u32; 2];
        unsafe { std::ptr::copy_nonoverlapping(b.6.ptr as *const u32, so.as_mut_ptr(), 2) };
        Ok((si, so))
    }

    /// plans/89 재개 — 프리필 다중 토큰 디바이스 선택: (적립+블록키) →
    /// q_rope(t행) → 점수(t×nb) → 토큰별 비토닉 top-k → 평탄 목록+sel_off.
    /// 호스트 d2h 4회(플러시)와 CPU 점수/정렬 소거. nb ≤ 4096.
    #[allow(clippy::too_many_arguments)]
    fn qsa_sel_dev_mt(
        &self,
        full_idx: usize,
        seq: usize,
        iq: u64,
        ik: u64,
        t: usize,
        pos0: usize,
        idx_heads: usize,
        idx_dim: usize,
        r: usize,
        idx_top_k: usize,
        iqw: &[f32],
        ikw: &[f32],
        cs_idx: &[f32],
        eps: f32,
    ) -> Result<(u64, u64, usize), String> {
        if r == 0 || idx_dim != 128 {
            return Err(format!("vk qsa_sel_dev_mt: r={r} idx_dim={idx_dim}"));
        }
        let nb_cap = (pos0 + t) / r;
        if nb_cap > 4096 {
            return Err(format!("vk qsa_sel_dev_mt: nb={nb_cap} > 4096 (호스트 폴백)"));
        }
        // 목록 총길이 — 산술(qsa_sel_list 동일식, 무동기).
        let mut list_len = 0usize;
        for tok in 0..t {
            let n_past = pos0 + tok + 1;
            let nb = n_past / r;
            let tail = n_past - nb * r;
            let width = n_past.min(idx_top_k + r - 1);
            let ns = ((width - tail) / r).min(nb);
            list_len += ns * r + tail;
        }
        self.qsa_idx_append_dev(full_idx, seq, ik, t, pos0, idx_dim, r, ikw, cs_idx, eps)?;
        let iqr_bytes = t * idx_heads * idx_dim * 4;
        let scr_bytes = t * nb_cap.max(1) * 4;
        let cs_bytes = t * (idx_dim / 2) * 2 * 4;
        let sd_bytes = list_len.max(1) * 4;
        let of_bytes = (t + 1) * 4;
        let mut ctx = self.ctx.lock();
        {
            let mut g = self.qsa_sel_bufs.lock();
            let need = match g.as_ref() {
                None => true,
                Some(b) => {
                    b.0.bytes < iqr_bytes
                        || b.1.bytes < scr_bytes
                        || b.4.bytes < cs_bytes
                        || b.5.bytes < sd_bytes
                        || b.6.bytes < of_bytes
                }
            };
            if need {
                *g = Some(crate::rawvk::context::site::scope("qsa_sel", || Ok::<_, String>((
                    ctx.alloc_host(iqr_bytes.max(1 << 16))?,
                    ctx.alloc_host(scr_bytes.max(4096))?,
                    ctx.alloc_host(4096)?,
                    ctx.alloc_host((iqw.len() * 4).max(4096))?,
                    ctx.alloc_host(cs_bytes.max(1 << 16))?,
                    ctx.alloc_host(sd_bytes.max(1 << 16))?,
                    ctx.alloc_host(of_bytes.max(4096))?,
                )))?);
            }
        }
        let (iqr, scr, _flg, iqwb, csb, sdev, ofdev) = {
            let g = self.qsa_sel_bufs.lock();
            let b = g.as_ref().unwrap();
            (b.0.clone(), b.1.clone(), b.2.clone(), b.3.clone(), b.4.clone(), b.5.clone(), b.6.clone())
        };
        unsafe {
            std::ptr::copy_nonoverlapping(iqw.as_ptr(), iqwb.ptr as *mut f32, iqw.len());
            std::ptr::copy_nonoverlapping(
                cs_idx[pos0 * idx_dim..].as_ptr(),
                csb.ptr as *mut f32,
                t * idx_dim,
            );
        }
        let iqb = self.fbuf(iq)?;
        {
            let p = self.pipeline(&mut ctx, Slot::FnIdxQRope)?;
            let ds = ctx.bind_ds(&p, &[iqb, iqr.buf, iqwb.buf, csb.buf])?;
            let mut push = eps.to_le_bytes().to_vec();
            push.extend_from_slice(&push_u32s(&[idx_dim as u32]));
            ctx.run(p.pl, ds, p.pipe, &push, idx_heads as u32, t as u32, 1)?;
        }
        let bkb = {
            let m = self.qsa_pools.lock();
            m.get(&(full_idx, seq))
                .map(|e| e.3.buf)
                .ok_or("vk qsa_sel_dev_mt: bk 풀 없음")?
        };
        if nb_cap > 0 {
            let p = self.pipeline(&mut ctx, Slot::FnIdxScoreMt)?;
            let ds = ctx.bind_ds(&p, &[iqr.buf, bkb, scr.buf])?;
            let push = push_u32s(&[
                pos0 as u32, r as u32, idx_heads as u32, idx_dim as u32, nb_cap as u32,
            ]);
            ctx.run(p.pl, ds, p.pipe, &push, nb_cap.div_ceil(256) as u32, t as u32, 1)?;
        }
        {
            let p = self.pipeline(&mut ctx, Slot::FnIdxTopkMt)?;
            let ds = ctx.bind_ds(&p, &[scr.buf, sdev.buf, ofdev.buf])?;
            let push = push_u32s(&[
                pos0 as u32, t as u32, r as u32, idx_top_k as u32, nb_cap as u32,
            ]);
            ctx.run(p.pl, ds, p.pipe, &push, 1, t as u32, 1)?;
        }
        Ok((sdev.buf.as_raw(), ofdev.buf.as_raw(), list_len))
    }

    /// plans/85 §2 — sel 목록이 디바이스에 있는 상주판 어텐션(업로드 없음,
    /// fn_qsa_attn_sel 그대로 — grid (t, n_head), t=1).
    #[allow(clippy::too_many_arguments)]
    fn qsa_attention_dev_sel(
        &self,
        q: u64,
        ck: u64,
        cv: u64,
        sel_idx: u64,
        sel_off: u64,
        _list_len: usize,
        kq_scale: f32,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
        out: u64,
    ) -> Result<(), String> {
        if hd != 256 {
            return Err(format!("vk qsa_attention_dev_sel: hd={hd} 미지원"));
        }
        let mut ctx = self.ctx.lock();
        let (qb, ob) = (self.fbuf(q)?, self.fbuf(out)?);
        let cb = vk::Buffer::from_raw(ck as u64);
        let vb = vk::Buffer::from_raw(cv as u64);
        let sib = vk::Buffer::from_raw(sel_idx as u64);
        let sob = vk::Buffer::from_raw(sel_off as u64);
        self.qsa_attn_sel_run(&mut ctx, qb, cb, vb, sib, sob, ob, kq_scale, n_head, n_kv, hd, t)
    }

    /// 선택 목록 어텐션 — fn_qsa_attn_sel 판(hd=256).
    #[allow(clippy::too_many_arguments)]
    fn qsa_attention_dev_res(
        &self,
        q: u64,
        ck: u64,
        cv: u64,
        sel_idx: &[u32],
        sel_off: &[u32],
        kq_scale: f32,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
        out: u64,
    ) -> Result<(), String> {
        if hd != 256 {
            return Err(format!("vk qsa_attention_dev_res: hd={hd} 미지원"));
        }
        let mut ctx = self.ctx.lock();
        let (qb, ob) = (self.fbuf(q)?, self.fbuf(out)?);
        let cb = vk::Buffer::from_raw(ck as u64);
        let vb = vk::Buffer::from_raw(cv as u64);
        // plans/86 §4 — sel 스크래치 캐시(종전 매호출 alloc_host 누출).
        let (si_b, so_b) = self.qsa_sel_scratch(&mut ctx, sel_idx.len(), sel_off.len())?;
        unsafe {
            std::ptr::copy_nonoverlapping(sel_idx.as_ptr(), si_b.ptr as *mut u32, sel_idx.len());
            std::ptr::copy_nonoverlapping(sel_off.as_ptr(), so_b.ptr as *mut u32, sel_off.len());
        }
        self.qsa_attn_sel_run(&mut ctx, qb, cb, vb, si_b.buf, so_b.buf, ob, kq_scale, n_head, n_kv, hd, t)
    }

    /// 업로드 판 어텐션 — 호스트 ck/cv 를 스크래치에 올려 동일 커널(plans/86 §3:
    /// §2 이후 프레임 폴백 꼬리가 이 경로를 요구한다 — 종전 미구현으로 CPU 폴백,
    /// 그 폴백의 mask_from_list(&[]) 가 패닉이었다).
    #[allow(clippy::too_many_arguments)]
    fn qsa_attention_dev(
        &self,
        q: u64,
        ck: &[f32],
        cv: &[f32],
        sel_idx: &[u32],
        sel_off: &[u32],
        kq_scale: f32,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
        out: u64,
    ) -> Result<(), String> {
        if hd != 256 {
            return Err(format!("vk qsa_attention_dev: hd={hd} 미지원"));
        }
        let mut ctx = self.ctx.lock();
        let (qb, ob) = (self.fbuf(q)?, self.fbuf(out)?);
        let (si_b, so_b) = self.qsa_sel_scratch(&mut ctx, sel_idx.len(), sel_off.len())?;
        let (ckb, cvb) = {
            let mut g = self.qsa_up_bufs.lock();
            let ok = g.as_ref().is_some_and(|b| b.0.bytes >= ck.len() * 4 && b.1.bytes >= cv.len() * 4);
            if !ok {
                let (kb, vb) = crate::rawvk::context::site::scope("qsa_sel", || Ok::<_, String>((
                    ctx.alloc_host((ck.len() * 4).max(1 << 16))?,
                    ctx.alloc_host((cv.len() * 4).max(1 << 16))?,
                )))?;
                let (_, _, old_si, old_so) = g.take().unwrap_or((vkbuf_null(), vkbuf_null(), vkbuf_null(), vkbuf_null()));
                *g = Some((kb, vb, old_si, old_so));
            }
            let b = g.as_ref().unwrap();
            (b.0.clone(), b.1.clone())
        };
        unsafe {
            std::ptr::copy_nonoverlapping(ck.as_ptr(), ckb.ptr as *mut f32, ck.len());
            std::ptr::copy_nonoverlapping(cv.as_ptr(), cvb.ptr as *mut f32, cv.len());
            std::ptr::copy_nonoverlapping(sel_idx.as_ptr(), si_b.ptr as *mut u32, sel_idx.len());
            std::ptr::copy_nonoverlapping(sel_off.as_ptr(), so_b.ptr as *mut u32, sel_off.len());
        }
        self.qsa_attn_sel_run(&mut ctx, qb, ckb.buf, cvb.buf, si_b.buf, so_b.buf, ob, kq_scale, n_head, n_kv, hd, t)
    }
}

impl VkAcc {
    /// plans/89 P0.4 — 선택 어텐션 발사: (n_head/n_kv)%4==0 이면 멀티헤드 판
    /// (grid (t, n_kv), 256스레드=4sg×헤드 — K/V 판독 12× 절감, 헤드별 산술
    /// 판과 동일). 킬스위치 LLM170_VK_QSAMH=0.
    fn qsa_attn_sel_run(
        &self,
        ctx: &mut VkCtx,
        qb: vk::Buffer,
        cb: vk::Buffer,
        vb: vk::Buffer,
        sib: vk::Buffer,
        sob: vk::Buffer,
        ob: vk::Buffer,
        kq_scale: f32,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
    ) -> Result<(), String> {
        let mh = n_kv >= 1
            && n_head % n_kv == 0
            && (n_head / n_kv) % 4 == 0
            && hd == 256
            && std::env::var("LLM170_VK_QSAMH").map(|v| v != "0").unwrap_or(true);
        let slot = if mh { Slot::FnQsaAttnSelMh } else { Slot::FnQsaAttnSel };
        let p = self.pipeline(ctx, slot)?;
        let ds2 = ctx.bind_ds(&p, &[qb, cb, vb, sib, sob, ob])?;
        let mut push = kq_scale.to_le_bytes().to_vec();
        push.extend_from_slice(&push_u32s(&[n_head as u32, n_kv as u32, hd as u32, t as u32]));
        let gy = if mh { n_kv as u32 } else { n_head as u32 };
        ctx.run(p.pl, ds2, p.pipe, &push, t as u32, gy, 1)
    }
}

impl VkAcc {
    /// plans/86 §2/§4 — (ptr,len) 키 상수 상주 업로드. 프레임이 테이블 Vec을
    /// 스텝 간 유지하므로 포인터가 곧 신원(호출부가 새 Vec을 만들면 미스).
    pub(super) fn qk_const(&self, ctx: &mut VkCtx, v: &[f32]) -> Result<VkBuf, String> {
        let key = (v.as_ptr() as usize, v.len());
        if let Some(b) = self.qk_consts.lock().get(&key) {
            return Ok(b.clone());
        }
        let b = crate::rawvk::context::site::scope("qsa_const", || ctx.alloc_host(v.len().max(1) * 4))?;
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), b.ptr as *mut f32, v.len()) };
        self.qk_consts.lock().insert(key, b.clone());
        Ok(b)
    }

    /// plans/86 §4 — sel_idx/sel_off 업로드 스크래치(성장 재할당, 매호출 alloc 회피).
    fn qsa_sel_scratch(&self, ctx: &mut VkCtx, si: usize, so: usize) -> Result<(VkBuf, VkBuf), String> {
        let mut g = self.qsa_up_bufs.lock();
        let ok = g.as_ref().is_some_and(|b| b.2.bytes >= si * 4 && b.3.bytes >= so * 4);
        if !ok {
            let (old_ck, old_cv, _, _) = g.take().unwrap_or((vkbuf_null(), vkbuf_null(), vkbuf_null(), vkbuf_null()));
            let (sib, sob) = crate::rawvk::context::site::scope("qsa_sel", || Ok::<_, String>((
                ctx.alloc_host((si * 4).max(1 << 16))?,
                ctx.alloc_host((so * 4).max(1 << 16))?,
            )))?;
            *g = Some((old_ck, old_cv, sib, sob));
        }
        let b = g.as_ref().unwrap();
        Ok((b.2.clone(), b.3.clone()))
    }

    /// 프레임 핸들 → 상주 버퍼 (없으면 Err).
    pub(super) fn fbuf(&self, h: u64) -> Result<vk::Buffer, String> {
        self.framebufs
            .lock()
            .get(&h)
            .map(|b| b.buf)
            .ok_or_else(|| format!("vk 프레임 핸들 없음: {h}"))
    }
}
