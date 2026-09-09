//! VkAcc — Vulkan matmul 가속기 (plans/12·13). HIP과 병립:
//! LLM170_GPU_RUNTIME=vulkan 시 주입, GDN/프레임은 CPU 폴백 (트레이트 Err).
//! 구조: 파이프라인·버퍼·가중치는 전부 지연 초기화 캐시, dispatch 헬퍼가
//! SSBO 바인딩+push+발사를 일원화 (M4b 확장 지점).

use crate::rawvk::context::{Pipes, VkBuf, VkCtx};
use ash::vk;
use llm170_core::matmul::{Accelerator, Weight};
use llm170_gguf::GgmlType;
use parking_lot::Mutex;
use std::collections::HashMap;

pub const GEMV_SPV: &[u8] = include_bytes!("spv/gemv3.spv");
const TILE128_SPV: &[u8] = include_bytes!("spv/tile128_q5k.spv");
pub const QUANT_SPV: &[u8] = include_bytes!("spv/quant_q8.spv");
pub const RMS_SPV: &[u8] = include_bytes!("spv/rms.spv");
pub const SILU_SPV: &[u8] = include_bytes!("spv/silu_mul.spv");

/// 파이프라인 세트 (vk 핸들은 복사 가능).
/// 지연 파이프라인 슬롯.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Slot {
    Gemv,
    Tile128,
    Quant,
    Rms,
    Silu,
}

pub struct VkAcc {
    ctx: Mutex<VkCtx>,
    pipes: Mutex<HashMap<Slot, Pipes>>,
    /// 가중치 캐시 (데이터 포인터 → 상주 청크들)
    wcache: Mutex<HashMap<usize, Vec<VkBuf>>>,
    tables: Mutex<Option<(VkBuf, VkBuf)>>,
    dummy: Mutex<Option<VkBuf>>,
    // 값-경로 버퍼 (필요시 성장)
    xfbuf: Mutex<Option<VkBuf>>,
    xbuf: Mutex<Option<VkBuf>>,
    obuf: Mutex<Option<VkBuf>>,
    sbufs: Mutex<Option<(VkBuf, VkBuf, VkBuf)>>,
    rbufs: Mutex<Option<(VkBuf, VkBuf, VkBuf)>>,
    // FFN 상주 체인 버퍼 (xf, xq0, fg, fu, glu, xq1, ob)
    ffnbufs: Mutex<Option<(VkBuf, VkBuf, VkBuf, VkBuf, VkBuf, VkBuf, VkBuf)>>,
    /// 그룹 배칭 가중별 출력 슬롯 (plans/19)
    gobufs: Mutex<Vec<Option<VkBuf>>>,
}

fn vk_ty(ty: GgmlType) -> Option<u32> {
    match ty {
        GgmlType::Q5K => Some(13),
        GgmlType::Q4K => Some(12),
        GgmlType::Q6K => Some(14),
        GgmlType::Iq4Xs => Some(23),
        GgmlType::Q8_0 => Some(8),
        GgmlType::Iq4Nl => Some(20),
        GgmlType::Q3K => Some(11),
        GgmlType::Iq3S => Some(21),
        _ => None,
    }
}

fn push_u32s(vals: &[u32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(vals.len() * 4);
    for x in vals {
        v.extend_from_slice(&x.to_le_bytes());
    }
    v
}

impl VkAcc {
    pub fn new() -> Result<Self, String> {
        let ctx = VkCtx::new()?;
        if !ctx.coop_matrix {
            eprintln!("rawvk: coop matrix 미지원 (타일 경로 M3에서 필요)");
        }
        Ok(Self {
            ctx: Mutex::new(ctx),
            pipes: Mutex::new(HashMap::new()),
            wcache: Mutex::new(HashMap::new()),
            tables: Mutex::new(None),
            dummy: Mutex::new(None),
            xfbuf: Mutex::new(None),
            xbuf: Mutex::new(None),
            obuf: Mutex::new(None),
            sbufs: Mutex::new(None),
            rbufs: Mutex::new(None),
            ffnbufs: Mutex::new(None),
            gobufs: Mutex::new(Vec::new()),
        })
    }

    // ─── 지연 초기화 공용 자원 ───

    fn pipeline(&self, ctx: &mut VkCtx, slot: Slot) -> Result<Pipes, String> {
        if let Some(&p) = self.pipes.lock().get(&slot) {
            return Ok(p);
        }
        let (spv, n_buf, pb) = match slot {
            Slot::Gemv => (GEMV_SPV, 12, 24u32),
            Slot::Tile128 => (TILE128_SPV, 10, 16),
            Slot::Quant => (QUANT_SPV, 2, 12),
            Slot::Rms => (RMS_SPV, 3, 12),
            Slot::Silu => (SILU_SPV, 3, 4),
        };
        let p = ctx.pipeline_pipes(spv, n_buf, pb)?;
        self.pipes.lock().insert(slot, p);
        Ok(p)
    }

    /// 배치 모드용 — p.ds 대신 fresh 세트에 바인딩해 반환 (세트 재사용 하저드:
    /// 녹화된 커맨드가 세트 객체를 참조 — 마지막 바인딩으로 전부 덮임).
    fn bind_ds(&self, ctx: &mut VkCtx, p: &Pipes, bufs: &[vk::Buffer]) -> Result<vk::DescriptorSet, String> {
        if ctx.batching.load(std::sync::atomic::Ordering::Relaxed) {
            ctx.batch_dsl.set(Some((p.dsl, p.pool)));
            let ds = ctx.fresh_ds(bufs.len() as u32)?;
            ctx.bind_bufs(ds, bufs);
            Ok(ds)
        } else {
            ctx.bind_bufs(p.ds, bufs);
            Ok(p.ds)
        }
    }

    /// ktab(iq4nl)·grid3s 테이블 + 더미 버퍼 — 최초 1회 업로드.
    fn ensure_shared(&self, ctx: &mut VkCtx) -> Result<(vk::Buffer, vk::Buffer, vk::Buffer), String> {
        if self.tables.lock().is_none() {
            let kv: Vec<u32> = llm170_core::ktab2_packed();
            let kb = ctx.alloc_host(1024)?;
            unsafe { std::ptr::copy_nonoverlapping(kv.as_ptr() as *const u8, kb.ptr, 1024) };
            let gb = ctx.alloc_host(2048)?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    llm170_core::IQ3S_GRID.as_ptr() as *const u8,
                    gb.ptr,
                    2048,
                );
            }
            *self.tables.lock() = Some((kb, gb));
        }
        if self.dummy.lock().is_none() {
            *self.dummy.lock() = Some(ctx.alloc_host(16)?);
        }
        let t = self.tables.lock();
        let (a, b) = t.as_ref().unwrap();
        Ok((a.buf, b.buf, self.dummy.lock().as_ref().unwrap().buf))
    }

    /// 가중치 상주 (ptr 키 — mmap 안정) — 128MB 청크 (RADV maxStorageBufferRange).
    fn weight_bufs(&self, ctx: &mut VkCtx, w: &Weight) -> Result<Vec<vk::Buffer>, String> {
        let key = w.data.as_ptr() as usize;
        {
            let mut wc = self.wcache.lock();
            if !wc.contains_key(&key) {
                let ch = ctx.max_ssbo; // plans/29: 균일 청크 — 크기는 push(chunk_words)로 전달
                let total = w.data.len();
                let mut bufs = Vec::new();
                let mut off = 0usize;
                while off < total {
                    let n = ch.min(total - off);
                    let mut b = ctx.alloc(n)?;
                    unsafe { std::ptr::copy_nonoverlapping(w.data.as_ptr().add(off), b.ptr, n) };
                    ctx.unmap(&mut b)?; // WC 매핑 즉시 해제 — op당 동기 비용 방지
                    bufs.push(b);
                    off += n;
                }
                wc.insert(key, bufs);
            }
        }
        let bufs: Vec<vk::Buffer> = {
            let wc = self.wcache.lock();
            wc.get(&key).unwrap().iter().map(|b| b.buf).collect()
        };
        if bufs.len() > 8 {
            return Err(format!("가중 청크 {}개 > 8 슬롯 (M2 한계)", bufs.len()));
        }
        Ok(bufs)
    }

    /// GEMV 1회 발사: 가중 청크(8) + xq + out + ktab + grid = 12 바인딩.
    #[allow(clippy::too_many_arguments)]
    fn gemv_run(
        &self,
        ctx: &mut VkCtx,
        wbufs: &[vk::Buffer],
        n_in: usize,
        n_out: usize,
        xq_w: usize,
        ty: u32,
        t: usize,
        xq_buf: vk::Buffer,
        out_buf: vk::Buffer,
    ) -> Result<(), String> {
        let (kb, gb, dbuf) = self.ensure_shared(ctx)?;
        let p = self.pipeline(ctx, Slot::Gemv)?;
        let mut binds: Vec<vk::Buffer> = wbufs.to_vec();
        while binds.len() < 8 {
            binds.push(dbuf);
        }
        binds.push(xq_buf);
        binds.push(out_buf);
        binds.push(kb);
        binds.push(gb);
        let ds2 = ctx.bind_ds(&p, &binds)?;
        // plans/29: 균일 청크 워드 수 (weight_bufs가 max_ssbo 단위로 분할).
        let chunk_words = (ctx.max_ssbo / 4) as u32;
        let push = push_u32s(&[n_in as u32, n_out as u32, xq_w as u32, ty, t as u32, chunk_words]);
        ctx.run(p.pl, ds2, p.pipe, &push, n_out as u32, 1, 1)
    }

    /// 128행 coopmat 타일 (q5_K, t≥2) — f16 스테이징, maxrel ~4.9e-4 (HIP v4급).
    fn tile128_run(
        &self,
        ctx: &mut VkCtx,
        wbufs: &[vk::Buffer],
        n_in: usize,
        n_out: usize,
        xq_w: usize,
        t: usize,
        xq_buf: vk::Buffer,
        out_buf: vk::Buffer,
    ) -> Result<(), String> {
        let (_, _, dbuf) = self.ensure_shared(ctx)?;
        let p = self.pipeline(ctx, Slot::Tile128)?;
        let mut binds: Vec<vk::Buffer> = wbufs.to_vec();
        while binds.len() < 8 {
            binds.push(dbuf);
        }
        binds.push(xq_buf);
        binds.push(out_buf);
        let ds2 = ctx.bind_ds(&p, &binds)?;
        let gx = (n_out + 127) as u32 / 128;
        for tb in (0..t).step_by(64) {
            let nt = (t - tb).min(64) as u32;
            let push = push_u32s(&[n_in as u32, n_out as u32, xq_w as u32, nt]);
            ctx.run(p.pl, ds2, p.pipe, &push, gx, 1, 1)?;
        }
        Ok(())
    }

    /// xs(f32) 업로드 → quant_q8 → xq 버퍼 (값 버퍼 자동 성장).
    fn quant_upload(
        &self,
        ctx: &mut VkCtx,
        xs: &[Vec<f32>],
        n_in: usize,
        xq_buf: vk::Buffer,
    ) -> Result<(), String> {
        let t = xs.len();
        {
            let mut xf = self.xfbuf.lock();
            let need = t * n_in * 4;
            if !xf.as_ref().map(|b| b.bytes >= need).unwrap_or(false) {
                *xf = Some(ctx.alloc_host(need.max(1 << 21))?);
            }
            let b = xf.as_ref().unwrap();
            for (ti, row) in xs.iter().enumerate() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        row.as_ptr(),
                        b.ptr.add(ti * n_in * 4) as *mut f32,
                        n_in,
                    );
                }
            }
        }
        let xfbuf = self.xfbuf.lock().as_ref().unwrap().buf;
        let p = self.pipeline(ctx, Slot::Quant)?;
        let ds2 = ctx.bind_ds(&p, &[xfbuf, xq_buf])?;
        let xq_w = n_in / 4 + n_in / 32 + n_in / 16;
        let push = push_u32s(&[n_in as u32, t as u32, xq_w as u32]);
        ctx.run(p.pl, ds2, p.pipe, &push, ((n_in / 32) + 63) as u32 / 64, t as u32, 1)
    }

    /// 값 버퍼 확보 (필요시 성장) → 핸들 반환.
    fn value_buf(&self, ctx: &mut VkCtx, slot: &Mutex<Option<VkBuf>>, need: usize) -> Result<vk::Buffer, String> {
        let mut g = slot.lock();
        if !g.as_ref().map(|b| b.bytes >= need).unwrap_or(false) {
            *g = Some(ctx.alloc_host(need.max(1 << 21))?);
        }
        Ok(g.as_ref().unwrap().buf)
    }

    /// out 버퍼에서 호스트 행 복사.
    fn download_out(&self, outs: &mut [Vec<f32>], n_out: usize, t: usize) {
        let ob = self.obuf.lock();
        let host = unsafe { std::slice::from_raw_parts(ob.as_ref().unwrap().ptr as *const f32, t * n_out) };
        for ti in 0..t {
            outs[ti].copy_from_slice(&host[ti * n_out..(ti + 1) * n_out]);
        }
    }
}

impl llm170_core::matmul::FrameState for VkAcc {}
unsafe impl Send for VkAcc {}
unsafe impl Sync for VkAcc {}

impl VkAcc {
    /// rms_norm 오프로드 — f32 세그먼트+f64 결합 (CPU sq_sum 미러와 동일 순서).
    pub fn rms_norm_gpu(
        &self,
        xs: &[Vec<f32>],
        w: &[f32],
        eps: f32,
        outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        let t = xs.len();
        let n = w.len();
        let mut ctx = self.ctx.lock();
        {
            let mut b = self.rbufs.lock();
            if b.is_none() {
                let xb = ctx.alloc_host((t * n * 4).max(1 << 21))?;
                let wb = ctx.alloc_host(n * 4)?;
                let ob = ctx.alloc_host((t * n * 4).max(1 << 21))?;
                *b = Some((xb, wb, ob));
            }
        }
        {
            let b = self.rbufs.lock();
            let (xv, wv, _) = b.as_ref().unwrap();
            for (ti, row) in xs.iter().enumerate() {
                unsafe { std::ptr::copy_nonoverlapping(row.as_ptr(), xv.ptr.add(ti * n * 4) as *mut f32, n) };
            }
            unsafe { std::ptr::copy_nonoverlapping(w.as_ptr(), wv.ptr as *mut f32, n) };
        }
        let (xb, wb, ob) = {
            let b = self.rbufs.lock();
            let r = b.as_ref().unwrap();
            (r.0.buf, r.1.buf, r.2.buf)
        };
        let p = self.pipeline(&mut ctx, Slot::Rms)?;
        let ds2 = ctx.bind_ds(&p, &[xb, wb, ob])?;
        let mut push = push_u32s(&[n as u32, t as u32]);
        push.extend_from_slice(&eps.to_le_bytes());
        ctx.run(p.pl, ds2, p.pipe, &push, t as u32, 1, 1)?;
        let host = {
            let b = self.rbufs.lock();
            unsafe { std::slice::from_raw_parts(b.as_ref().unwrap().2.ptr as *const f32, t * n) }
        };
        for ti in 0..t {
            outs[ti].copy_from_slice(&host[ti * n..(ti + 1) * n]);
        }
        Ok(())
    }

    /// silu_mul 오프로드 — exp_cr f64 호너 GLSL 비트 재현.
    pub fn silu_mul_gpu(
        &self,
        gs: &[Vec<f32>],
        us: &[Vec<f32>],
        outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        let t = gs.len();
        let n = gs[0].len();
        let total = t * n;
        let mut ctx = self.ctx.lock();
        {
            let mut b = self.sbufs.lock();
            if !b.as_ref().map(|(g, _, _)| g.bytes >= total * 4).unwrap_or(false) {
                let g = ctx.alloc_host((total * 4).max(1 << 21))?;
                let u = ctx.alloc_host((total * 4).max(1 << 21))?;
                let o = ctx.alloc_host((total * 4).max(1 << 21))?;
                *b = Some((g, u, o));
            }
        }
        {
            let b = self.sbufs.lock();
            let (gv, uv, _) = b.as_ref().unwrap();
            for (ti, row) in gs.iter().enumerate() {
                unsafe { std::ptr::copy_nonoverlapping(row.as_ptr(), gv.ptr.add(ti * n * 4) as *mut f32, n) };
            }
            for (ti, row) in us.iter().enumerate() {
                unsafe { std::ptr::copy_nonoverlapping(row.as_ptr(), uv.ptr.add(ti * n * 4) as *mut f32, n) };
            }
        }
        let (gb, ub, ob) = {
            let b = self.sbufs.lock();
            let r = b.as_ref().unwrap();
            (r.0.buf, r.1.buf, r.2.buf)
        };
        let p = self.pipeline(&mut ctx, Slot::Silu)?;
        let ds2 = ctx.bind_ds(&p, &[gb, ub, ob])?;
        let total_u = total as u32;
        ctx.run(p.pl, ds2, p.pipe, &total_u.to_le_bytes(), total_u.div_ceil(256), 1, 1)?;
        let host = {
            let b = self.sbufs.lock();
            unsafe { std::slice::from_raw_parts(b.as_ref().unwrap().2.ptr as *const f32, total) }
        };
        for ti in 0..t {
            outs[ti].copy_from_slice(&host[ti * n..(ti + 1) * n]);
        }
        Ok(())
    }

    /// FFN 상주 체인 — 업로드 1회(xs)·다운로드 1회(xs), gate/up/silu/glu/down 전부 GPU 상주.
    #[allow(clippy::too_many_arguments)]
    pub fn ffn_chain_gpu(
        &self,
        xs: &[Vec<f32>],
        gate_w: &Weight,
        up_w: &Weight,
        down_w: &Weight,
        xs_out: &mut [Vec<f32>],
    ) -> Result<(), String> {
        let t = xs.len();
        let n0 = gate_w.n_in as usize; // n_embd
        let n_ff = gate_w.n_out as usize;
        let xq0_w = n0 / 4 + n0 / 32 + n0 / 16;
        let xq1_w = n_ff / 4 + n_ff / 32 + n_ff / 16;
        let mut ctx = self.ctx.lock();
        // 체인 버퍼 (고정 용량 — 모델 최대 기준)
        let (xbf, bq0, bfg, bfu, bglu, bq1, bob, xf_ptr, ob_ptr) = {
            let mut b = self.ffnbufs.lock();
            if b.is_none() {
                let xf = ctx.alloc_host(1 << 23)?;
                let xq0 = ctx.alloc_host(1 << 22)?;
                let fg = ctx.alloc_host(1 << 24)?;
                let fu = ctx.alloc_host(1 << 24)?;
                let glu = ctx.alloc_host(1 << 24)?;
                let xq1 = ctx.alloc_host(1 << 24)?;
                let ob = ctx.alloc_host(1 << 23)?;
                *b = Some((xf, xq0, fg, fu, glu, xq1, ob));
            }
            let r = b.as_ref().unwrap();
            (r.0.buf, r.1.buf, r.2.buf, r.3.buf, r.4.buf, r.5.buf, r.6.buf, r.0.ptr, r.6.ptr)
        };
        // 배치 모드 — 6연산 단일 제출 (plans/19: sync ~0.9ms×5 절감)
        if std::env::var_os("LLM170_VK_NOBATCH").is_none() {
            ctx.begin_batch()?;
        }
        // 1) xs 업로드 → quant(n0)
        for (ti, row) in xs.iter().enumerate() {
            unsafe { std::ptr::copy_nonoverlapping(row.as_ptr(), xf_ptr.add(ti * n0 * 4) as *mut f32, n0) };
        }
        {
            let p = self.pipeline(&mut ctx, Slot::Quant)?;
            let ds2 = ctx.bind_ds(&p, &[xbf, bq0])?;
            let push = push_u32s(&[n0 as u32, t as u32, xq0_w as u32]);
            ctx.run(p.pl, ds2, p.pipe, &push, ((n0 / 32) + 63) as u32 / 64, t as u32, 1)?;
        }
        // 2) gate/up GEMV (같은 xq0) — 상주 출력
        for (w, obuf) in [(gate_w, bfg), (up_w, bfu)] {
            let ty = vk_ty(w.ty).ok_or("ffn 타입 미지원")?;
            let wbufs = self.weight_bufs(&mut ctx, w)?;
            self.gemv_run(&mut ctx, &wbufs, n0, w.n_out as usize, xq0_w, ty, t, bq0, obuf)?;
        }
        // 3) silu_mul 상주 (bfg, bfu → bglu)
        {
            let p = self.pipeline(&mut ctx, Slot::Silu)?;
            let ds2 = ctx.bind_ds(&p, &[bfg, bfu, bglu])?;
            let total = (t * n_ff) as u32;
            ctx.run(p.pl, ds2, p.pipe, &total.to_le_bytes(), total.div_ceil(256), 1, 1)?;
        }
        // 4) glu quant(n_ff)
        {
            // bglu는 f32가 아니라 f32→q8 변환 입력 — quant 셰이더에 직접.
            // (bglu는 silu 출력 f32 → quant가 읽는다)
            let p = self.pipeline(&mut ctx, Slot::Quant)?;
            let ds2 = ctx.bind_ds(&p, &[bglu, bq1])?;
            let push = push_u32s(&[n_ff as u32, t as u32, xq1_w as u32]);
            ctx.run(p.pl, ds2, p.pipe, &push, ((n_ff / 32) + 63) as u32 / 64, t as u32, 1)?;
        }
        // 5) down GEMV
        {
            let ty = vk_ty(down_w.ty).ok_or("ffn down 타입 미지원")?;
            let wbufs = self.weight_bufs(&mut ctx, down_w)?;
            self.gemv_run(&mut ctx, &wbufs, n_ff, down_w.n_out as usize, xq1_w, ty, t, bq1, bob)?;
        }
        // 6) 일괄 제출·대기 → 다운로드 1회
        if std::env::var_os("LLM170_VK_NOBATCH").is_none() {
            ctx.end_batch_wait()?;
        }
        let host = unsafe { std::slice::from_raw_parts(ob_ptr as *const f32, t * n0) };
        for ti in 0..t {
            xs_out[ti].copy_from_slice(&host[ti * n0..(ti + 1) * n0]);
        }
        Ok(())
    }
}

impl Accelerator for VkAcc {
    fn matmul_batch(
        &self,
        xs: &[Vec<f32>],
        w: &Weight,
        outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        let ty = match vk_ty(w.ty) {
            Some(t) => t,
            None => {
                llm170_core::matmul::matmul_batch(xs, w, outs);
                return Ok(());
            }
        };
        let n_in = w.n_in as usize;
        let n_out = w.n_out as usize;
        let t = xs.len();
        let xq_w = n_in / 4 + n_in / 32 + n_in / 16;
        let mut ctx = self.ctx.lock();
        let xq = self.value_buf(&mut ctx, &self.xbuf, t * xq_w * 4)?;
        let ob = self.value_buf(&mut ctx, &self.obuf, t * n_out * 4)?;
        self.quant_upload(&mut ctx, xs, n_in, xq)?;
        let wbufs = self.weight_bufs(&mut ctx, w)?;
        // 128행 타일 (q5_K, t≥2, env) — f16 fast 경로
        if ty == 13 && t >= 2 && std::env::var_os("LLM170_VK_TILE").is_some() {
            self.tile128_run(&mut ctx, &wbufs, n_in, n_out, xq_w, t, xq, ob)?;
            self.download_out(outs, n_out, t);
            return Ok(());
        }
        self.gemv_run(&mut ctx, &wbufs, n_in, n_out, xq_w, ty, t, xq, ob)?;
        self.download_out(outs, n_out, t);
        Ok(())
    }

    /// 같은 입력 그룹: 업로드+양자화 1회 → GEMV 각각 → 개별 다운로드.
    fn matmul_group(
        &self,
        xs: &[Vec<f32>],
        ws: &[Weight],
        outs: &mut [Vec<Vec<f32>>],
    ) -> Result<(), String> {
        if ws.iter().any(|w| vk_ty(w.ty).is_none())
            || ws.iter().any(|w| w.n_in != ws[0].n_in)
        {
            for (w, out) in ws.iter().zip(outs.iter_mut()) {
                self.matmul_batch(xs, w, out)?;
            }
            return Ok(());
        }
        let n_in = ws[0].n_in as usize;
        let t = xs.len();
        let xq_w = n_in / 4 + n_in / 32 + n_in / 16;
        let mut ctx = self.ctx.lock();
        let xq = self.value_buf(&mut ctx, &self.xbuf, t * xq_w * 4)?;
        self.quant_upload(&mut ctx, xs, n_in, xq)?;
        // 배치: 모든 가중 GEMV 녹화 → 단일 제출 → 일괄 다운로드 (plans/19)
        let do_batch = std::env::var_os("LLM170_VK_NOBATCH").is_none();
        if do_batch {
            ctx.begin_batch()?;
        }
        let mut hosts: Vec<(*mut u8, usize, usize)> = Vec::with_capacity(ws.len()); // (ptr, n_out, ti스트라이드)
        for (wi, w) in ws.iter().enumerate() {
            let ty = vk_ty(w.ty).unwrap();
            let n_out = w.n_out as usize;
            let wbufs = self.weight_bufs(&mut ctx, w)?;
            // 가중별 독립 출력 버퍼 (그룹 세션 슬롯)
            let ob = {
                let mut g = self.gobufs.lock();
                while g.len() <= wi {
                    g.push(None);
                }
                if g[wi].as_ref().map(|b| b.bytes >= t * n_out * 4).unwrap_or(false) {
                    g[wi].as_ref().unwrap().buf
                } else {
                    let b = ctx.alloc_host(t * n_out * 4)?;
                    let buf = b.buf;
                    g[wi] = Some(b);
                    buf
                }
            };
            self.gemv_run(&mut ctx, &wbufs, n_in, n_out, xq_w, ty, t, xq, ob)?;
            let ptr = self.gobufs.lock()[wi].as_ref().unwrap().ptr;
            hosts.push((ptr, n_out, wi));
        }
        if do_batch {
            ctx.end_batch_wait()?;
        }
        for (ptr, n_out, wi) in hosts {
            let host = unsafe { std::slice::from_raw_parts(ptr as *const f32, t * n_out) };
            for ti in 0..t {
                outs[wi][ti].copy_from_slice(&host[ti * n_out..(ti + 1) * n_out]);
            }
        }
        Ok(())
    }

    fn rms_norm(
        &self,
        xs: &[Vec<f32>],
        w: &[f32],
        eps: f32,
        outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        self.rms_norm_gpu(xs, w, eps, outs)
    }

    fn silu_mul(
        &self,
        gs: &[Vec<f32>],
        us: &[Vec<f32>],
        outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        self.silu_mul_gpu(gs, us, outs)
    }

    fn ffn_chain(
        &self,
        xs: &[Vec<f32>],
        gate_w: &Weight,
        up_w: &Weight,
        down_w: &Weight,
        xs_out: &mut [Vec<f32>],
    ) -> Result<(), String> {
        self.ffn_chain_gpu(xs, gate_w, up_w, down_w, xs_out)
    }

    fn matmul(&self, x: &[f32], w: &Weight, out: &mut [f32]) -> Result<(), String> {
        let xs = vec![x.to_vec()];
        let mut tmp = vec![vec![0.0f32; w.n_out as usize]];
        self.matmul_batch(&xs, w, &mut tmp)?;
        out.copy_from_slice(&tmp[0]);
        Ok(())
    }
}

/// vk-gemv-check — VkAcc matmul vs CPU W4A8 미러 단일 텐서 검증 + 타이밍.
pub fn gemv_check(path: &str, tname: &str, t: usize) -> Result<String, String> {
    let model = llm170_core::model::Model::load(std::path::Path::new(path))
        .map_err(|e| e.to_string())?;
    let w = model.w(tname).ok_or("텐서 없음")?;
    let wref = &w;
    let n_in = w.n_in as usize;
    let acc = VkAcc::new()?;
    // ── quant 비트 검증: GPU xq vs CPU quantize_row_q8_ref ──
    {
        let mut seed2 = 0x1234abcdu64;
        let mut lcg2 = || {
            seed2 = seed2.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed2 >> 33) as f32 / 2147483648.0 - 0.5
        };
        let xrow: Vec<f32> = (0..n_in).map(|_| lcg2()).collect();
        let mut ctxg = acc.ctx.lock();
        let xq_w = n_in / 4 + n_in / 32 + n_in / 16;
        let xqb = ctxg.alloc_host(xq_w * 4)?;
        acc.quant_upload(&mut ctxg, std::slice::from_ref(&xrow), n_in, xqb.buf)?;
        let gpu: &[u32] =
            unsafe { std::slice::from_raw_parts(xqb.ptr as *const u32, xq_w) };
        let yref = llm170_core::quant::quantize_row_q8_ref(&xrow);
        // CPU 재구성: qs 워드 + d 비트 + s0/s1
        let mut qdiff = 0usize;
        let mut ddiff = 0usize;
        let mut sdiff = 0usize;
        let nwords = n_in / 4;
        let nblk = n_in / 32;
        for b in 0..nblk {
            let d_cpu = yref[b].d.to_bits();
            let d_gpu = gpu[nwords + b];
            if d_cpu != d_gpu {
                ddiff += 1;
                if ddiff <= 3 {
                    let mut amax = 0.0f32;
                    for &v in &xrow[b * 32..b * 32 + 32] {
                        amax = amax.max(v.abs());
                    }
                    let (cpu_d, gpu_d, rust_d, f64d) = (d_cpu, d_gpu, (amax / 127.0f32).to_bits(), (amax as f64 / 127.0).to_bits() as u32);
                    eprintln!("dblk{b}: amax={amax:e} cpu_d={cpu_d:08x} gpu_d={gpu_d:08x} rust_d={rust_d:08x} f64lo={f64d:08x}");
                }
            }
            let mut s0 = 0u32;
            let mut s1 = 0u32;
            for wi in 0..8 {
                let mut word = 0u32;
                for k in 0..4 {
                    let qv = yref[b].qs[wi * 4 + k] as i8 as i32 as u32;
                    word |= (qv & 0xFF) << (8 * k);
                }
                if word != gpu[b * 8 + wi] {
                    qdiff += 1;
                }
                let sd: u32 = (0..4).map(|k| ((gpu[b * 8 + wi] >> (8 * k)) & 0xFF).count_ones().min(1) * 0).sum();
                let _ = sd;
                let bytes: i32 = (0..4)
                    .map(|k| ((gpu[b * 8 + wi] >> (8 * k)) & 0xFF) as i32)
                    .fold(0i32, |a, v| a + (((v << 24) as i32) >> 24));
                if wi < 4 {
                    s0 = s0.wrapping_add(bytes as u32);
                } else {
                    s1 = s1.wrapping_add(bytes as u32);
                }
            }
            let qsb = nwords + nblk;
            if s0 != gpu[qsb + b * 2] || s1 != gpu[qsb + b * 2 + 1] {
                sdiff += 1;
            }
        }
        eprintln!(
            "quant-bits: qs워드 {qdiff}/{nwords} d {ddiff}/{nblk} s {sdiff}/{nblk} 상이"
        );
    }
    let mut seed = 0x9e3779b9u64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as f32 / 2147483648.0 - 0.5
    };
    let xs: Vec<Vec<f32>> = (0..t).map(|_| (0..n_in).map(|_| lcg()).collect()).collect();
    let mut outs = vec![vec![0.0f32; w.n_out as usize]; t];
    acc.matmul_batch(&xs, wref, &mut outs)?;
    let t0 = std::time::Instant::now();
    for _ in 0..10 {
        acc.matmul_batch(&xs, wref, &mut outs)?;
    }
    let dt = t0.elapsed().as_secs_f64() / 10.0;
    eprintln!(
        "vk-gemv-time: {} {:.2}ms → {:.1}GB/s ({}B 가중)",
        tname,
        dt * 1e3,
        wref.data.len() as f64 / dt / 1e9,
        wref.data.len()
    );
    let mut ref_outs = vec![vec![0.0f32; w.n_out as usize]; t];
    llm170_core::matmul::matmul_batch(&xs, wref, &mut ref_outs);
    let mut mx = 0f64;
    let mut rel = 0f64;
    let mut ndiff = 0usize;
    let mut ulp_hist = std::collections::HashMap::<i64, usize>::new();
    for (a, b) in outs.iter().zip(ref_outs.iter()) {
        for (x, y) in a.iter().zip(b.iter()) {
            if x.to_bits() != y.to_bits() {
                ndiff += 1;
                let ulp = (x.to_bits() as i64 - y.to_bits() as i64).abs();
                *ulp_hist.entry(ulp).or_insert(0) += 1;
            }
            let d = (x - y).abs() as f64;
            if d > mx {
                mx = d;
            }
            let r = d / y.abs().max(1.0) as f64;
            if r > rel {
                rel = r;
            }
        }
    }
    let hist: Vec<String> = {
        let mut v: Vec<(i64, usize)> = ulp_hist.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v.iter().take(4).map(|(u, c)| format!("{c}x{u}ulp")).collect()
    };
    let ia = outs[0].iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i);
    let ib = ref_outs[0].iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i);
    Ok(format!(
        "vk-gemv {tname} t={t}: max|D|={mx:.3e} maxrel={rel:.2e} argmax {ia:?}=={ib:?} {} | bits {ndiff}/{} differ, top {hist:?}",
        if ia == ib { "★" } else { "MISMATCH" },
        outs.len() * w.n_out as usize
    ))
}

/// 부록87: llama matmul_q5_k_f16.spv 직접 로드 격리 측정 (t≥2 프리필 GEMM).
pub fn vk_mmq_check(path: &str, tname: &str, t: usize) -> Result<String, String> {
    let model = llm170_core::model::Model::load(std::path::Path::new(path))
        .map_err(|e| e.to_string())?;
    let w = model.w(tname).ok_or("텐서 없음")?;
    let n_in = w.n_in as usize;
    let n_out = w.n_out as usize;
    if std::env::var_os("VK_DUMP_W").is_some() {
        let _ = std::fs::write("/tmp/q5k_w.bin", w.data);
        eprintln!("W 더프: {}B n_in={} n_out={}", w.data.len(), n_in, n_out);
    }
    let spv_name = std::env::var("VKMMQ_SPV").unwrap_or_else(|_| "matmul_q5_k_f16".into());
    let b_is_f32 = spv_name.ends_with("_f32") || spv_name.contains("_f32_");
    let spv = std::fs::read(format!("/home/yoon/LLM170/source/llama.cpp/build-vk/ggml/src/ggml-vulkan/vulkan-shaders.spv/{}.spv", spv_name))
        .map_err(|e| e.to_string())?;
    let mut acc = VkAcc::new()?;
    let mut ctxg = acc.ctx.lock();
    // 버퍼: A=가중(호스트맵→h2d는 run 전 복사), B=f16 y, D=f32 out
    let mut ab = ctxg.alloc_host(w.data.len())?;
    unsafe { std::ptr::copy_nonoverlapping(w.data.as_ptr(), ab.ptr, w.data.len()); }
    ctxg.unmap(&mut ab)?;
    let mut ybuf: Vec<u16> = Vec::with_capacity(n_in * t);
    let mut seed = 0x1234u64;
    let mut lcg = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 33) as f32 / 2147483648.0 - 0.5 };
    // y [K][N] f16 (stride_b = K) — CPU 참조와 동일값 사용
    let mut yf: Vec<f32> = Vec::with_capacity(n_in * t);
    for _ in 0..n_in * t { let v = lcg(); yf.push(v); ybuf.push(hf(v)); }
    let b_bytes = if b_is_f32 { n_in * t * 4 } else { n_in * t * 2 };
    let mut bb = ctxg.alloc_host(b_bytes)?;
    if b_is_f32 {
        unsafe { std::ptr::copy_nonoverlapping(yf.as_ptr() as *const u8, bb.ptr, b_bytes); }
    } else {
        unsafe { std::ptr::copy_nonoverlapping(ybuf.as_ptr() as *const u8, bb.ptr, b_bytes); }
    }
    ctxg.unmap(&mut bb)?;
    eprintln!("bc: allocs ok");
    let mut db = ctxg.alloc_host(n_out * t * 4)?;
    // l 파이프라인 (non-cm, subgroup=64, gfx1151): ids 0..10 + ALIGNED=0
    let sp = std::env::var("VKMMQ_SPEC").unwrap_or_else(|_| "l".into());
    let is_cm1 = spv_name.contains("_cm1");
    let spec: Vec<u32> = match sp.as_str() {
        "m" => vec![128, 64, 64, 32, 64, 32, 2, 4, 2, 1, 64, 0],
        "s" => vec![64, 32, 32, 32, 32, 32, 2, 2, 2, 1, 64, 0],
        "c" => vec![128, 128, 32, 32, 64, 32, 2, 4, 4, 1, 64, 0],
        "m32" => vec![128, 64, 64, 32, 32, 32, 2, 4, 2, 1, 32, 0],
        "l32" => vec![128, 128, 128, 32, 64, 64, 2, 4, 4, 1, 32, 0],
        "cc" => vec![128, 64, 32, 32, 64, 32, 2, 4, 4, 1, 64, 0],
        "mini" => vec![32, 32, 16, 32, 32, 16, 1, 4, 4, 1, 32, 0],
        "def" => vec![64, 64, 64, 16, 32, 32, 2, 4, 2, 1, 32, 0],  // spv 기본값 (부록87 해독)
        "cm1" => vec![128, 128, 128, 16, 128, 64, 2, 16, 16, 16, 64, 0],
        _ => vec![128, 128, 128, 32, 128, 64, 2, 4, 4, 1, 64, 0],
    };
    eprintln!("bc: y/ab 채움");
    let (_dsl, pl, _dp, ds, pipe) = ctxg.pipeline_spec_fg(&spv, 3, 17 * 4, &spec, is_cm1)?;
    eprintln!("bc: 파이프라인 ok");
    let bufs = [ab.buf, bb.buf, db.buf];
    ctxg.bind_bufs(ds, &bufs);
    // push: M,N,K,stride_a=K,stride_b=K,stride_d=M,batch 0들 + k_split=1 등
    let mut pc: Vec<u32> = vec![
        n_out.div_ceil(8) as u32, t as u32, n_in as u32,      // M, N, K
        n_in as u32, n_in as u32, t as u32,       // stride_a=K, stride_b=K, stride_d=N
        0, 0, 0,                                  // batch strides
        0, 1, n_in as u32,                        // base_wg_z, num_batches, k_split=K (split_k=1 규약)
        1, 1, 1, 1,                               // ne02, ne12, broadcast2, broadcast3
        t as u32,                                 // padded_n (f16 B — 비양자화 경로)
    ];
    let pcb: Vec<u8> = pc.iter().flat_map(|v| v.to_le_bytes()).collect();
    // 그리드 분모 = 스펙의 BM/BN에 정합 (부록87 그리드-스펙 매칭)
    let (dx, dy) = match sp.as_str() {
        "m" | "m32" => (64u32, 64),
        "s" => (32, 32),
        "c" | "cc" => (64, 32),
        "mini" => (32, 16),
        "def" => (64, 64),
        _ => (128, 128),
    };
    let gx = (n_out as u32).div_ceil(dx);
    let gy = (t as u32).div_ceil(dy);
    ctxg.begin_batch()?;
    ctxg.run(pl, ds, pipe, &pcb, gx, gy, 1)?;
    ctxg.end_batch_wait()?;
    // CPU 참조 대조 (처음 8값) + 타이밍
    let out: &[f32] = unsafe { std::slice::from_raw_parts(db.ptr as *const f32, n_out * t) };
    // CPU 참조: 몇 개 (m, n) 지점 대조 — y는 f16 반올림값 사용
    let yh: Vec<f32> = ybuf.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
    let rb = w.data.len() / n_out;
    let mut dq = vec![0f32; n_in];
    let mut ok_nm = 0usize; let mut ok_mn = 0usize; let mut tot = 0usize;
    for &m in &[0usize, 100, 3000, 6143] {
        llm170_core::quant::dequant_row(w.ty, &w.data[m * rb..(m + 1) * rb], 0, n_in as u64, &mut dq);
        for n in 0..t {
            let yrow = &yh[n * n_in..(n + 1) * n_in];
            let mut acc = 0f32;
            for k in 0..n_in { acc += dq[k] * yrow[k]; }
            let g_nm = out[n * t + m];
            let g_mn = out[m * t + n];
            let r = |g: f32| (g - acc).abs() / acc.abs().max(1e-3);
            if r(g_nm) < 0.01 { ok_nm += 1; }
            if r(g_mn) < 0.01 { ok_mn += 1; }
            tot += 1;
        }
    }
    eprintln!("레이아웃 판별: [N][M]={}/{} · [M][N]={}/{}", ok_nm, tot, ok_mn, tot);
    {
        let m = 0usize;
        llm170_core::quant::dequant_row(w.ty, &w.data[m * rb..(m + 1) * rb], 0, n_in as u64, &mut dq);
        let mut goods = vec![];
        for n in 0..t {
            let yrow = &yh[n * n_in..(n + 1) * n_in];
            let mut acc = 0f32;
            for k in 0..n_in { acc += dq[k] * yrow[k]; }
            if ((out[n * t + m] - acc).abs() / acc.abs().max(1e-3)) < 0.01 { goods.push(n); }
        }
        eprintln!("m=0 정답 n ({}개): {:?}", goods.len(), &goods[..goods.len().min(20)]);
    }
    let mut maxrel = 0f32;
    for &(m, n) in &[(0, 0), (1, 0), (63, 0), (64, 0), (100, 0), (127, 0), (128, 0), (0, 1), (0, 63), (0, 64), (0, 100), (0, 127), (0, 128), (100, 7), (200, 100)] {
        if m >= n_out || n >= t { continue; }
        llm170_core::quant::dequant_row(w.ty, &w.data[m * rb..(m + 1) * rb], 0, n_in as u64, &mut dq);
        let yrow = &yh[n * n_in..(n + 1) * n_in];
        let mut acc = 0f64;
        for k in 0..n_in { acc += dq[k] as f64 * yrow[k] as f64; }
        let got = out[n * t + m];
        eprintln!("  ck m={m} n={n}: got={got:.5} ref={:.5}", acc);
        let rel = if acc.abs() > 1e-6 { ((got - acc as f32) / acc as f32).abs() } else { got.abs() };
        maxrel = maxrel.max(rel);
    }
    // 타이밍 20회
    ctxg.begin_batch()?;
    let t0 = std::time::Instant::now();
    for _ in 0..20 { ctxg.run(pl, ds, pipe, &pcb, gx, gy, 1)?; }
    ctxg.end_batch_wait()?;
    let dt = t0.elapsed().as_secs_f64() / 20.0;
    let _ = &mut pc;
    Ok(format!("vk-mmq({tname}) t={t}: {:.3}ms/회 · maxrel={maxrel:.4}", dt * 1e3))
}

fn hf(v: f32) -> u16 {
    // f32→f16 변환 (반올림)
    half::f16::from_f32(v).to_bits()
}


/// vk-sdot-probe — OpSDot(정수 dot) 장치 지원 검증+타이밍. plans/33.
pub fn sdot_probe() -> Result<String, String> {
    use std::time::Instant;
    let acc = VkAcc::new()?;
    let mut ctx = acc.ctx.lock();
    let buf = ctx.alloc_host(16)?;
    unsafe {
        let p = buf.ptr as *mut u32;
        *p.add(0) = 0x0182_0304;      // a (부호 혼합 i8x4)
        *p.add(1) = 0xF0FF_7F01;      // b
        *p.add(2) = 0;
        *p.add(3) = 0;
    }
    let spv = std::fs::read("crates/backend-gpu/src/rawvk/spv/sdot_probe.spv")
        .map_err(|e| e.to_string())?;
    let (dsl, pl, pool, ds, pipe) = ctx.pipeline(&spv, 1, 4)?;
    let _ = (dsl, pool);
    ctx.bind_bufs(ds, &[buf.buf]);
    let t0 = Instant::now();
    ctx.run(pl, ds, pipe, &1_000_000u32.to_le_bytes(), 1, 1, 1)?;
    let dt = t0.elapsed().as_secs_f32();
    let r = unsafe { *(buf.ptr as *const u32).add(2) };
    // CPU 기준: acc = a; 1M회 acc = sdot(acc, b) — i32 감쇠/순환값
    let mut cacc: i32 = 0x0182_0304u32 as i32;
    let b4: i32 = 0xF0FF_7F01u32 as i32;
    let bx = |v: i32, i: u32| -> i32 {
        let byte = (v >> (i * 8)) & 0xFF;
        if byte >= 128 { byte - 256 } else { byte }
    };
    for _ in 0..1_000_000 {
        let mut s = 0i32;
        for i in 0..4 { s += bx(cacc, i) * bx(b4, i); }
        cacc = s;
    }
    let expect = cacc as u32;
    Ok(format!(
        "sdot-probe: gpu={r:#010x} cpu={expect:#010x} {} · {dt:.1}ms (1M 의존 dot)",
        if r == expect { "일치" } else { "불일치" }
    ))
}


/// vk-gemv8-check — gemv8 패밀리(llama mul_mat_vec 포트, f32 직결) 검증+타이밍.
pub fn gemv8_check(path: &str, tname: &str, t: usize) -> Result<String, String> {
    use std::time::Instant;
    let model = llm170_core::model::Model::load(std::path::Path::new(path))
        .map_err(|e| e.to_string())?;
    let w = model.w(tname).ok_or("텐서 없음")?;
    let is_xs = w.ty == llm170_gguf::GgmlType::Iq4Xs;
    let is_q5 = w.ty == llm170_gguf::GgmlType::Q5K;
    let is_q6 = w.ty == llm170_gguf::GgmlType::Q6K;
    let is_q4 = w.ty == llm170_gguf::GgmlType::Q4K;
    let is_q3 = w.ty == llm170_gguf::GgmlType::Q3K;
    if !is_xs && !is_q5 && !is_q6 && !is_q4 && !is_q3 {
        return Err("gemv8 검증은 q3_K/q4_K/q5_K/q6_K/iq4_xs만".into());
    }
    let n_in = w.n_in as usize;
    let n_out = w.n_out as usize;
    let acc = VkAcc::new()?;
    let mut ctx = acc.ctx.lock();
    let mut seed = 0x1234u64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as f32 / 2147483648.0 - 0.5
    };
    let xs: Vec<Vec<f32>> = (0..t).map(|_| (0..n_in).map(|_| lcg()).collect()).collect();
    let xa = ctx.alloc_host(t * n_in * 4)?;
    for (j, x) in xs.iter().enumerate() {
        unsafe {
            // 행 스트라이드는 바이트 — n_in f32 = n_in*4바이트 (A2: 이 오타가
            // t≥2 하니스 오염의 전부였음 — 행1이 행0의 1/4 지점을 덮어씀)
            std::ptr::copy_nonoverlapping(
                x.as_ptr(), xa.ptr.add(j * n_in * 4) as *mut f32, n_in);
        }
    }
    let ob = ctx.alloc_host(t * n_out * 4)?;  // 매핑 유지 — 판독용
    // 가중 업로드 — gemv3와 동일한 균일 청크
    let ch = ctx.max_ssbo;
    let mut wbufs = Vec::new();
    let mut off = 0usize;
    let total = w.data.len();
    // 청크 크기 2의 거듭제곱 (WG 시프트 산술) — 마지막 청크는 실제 크기만 할당:
    // o = idx & mask 는 항상 청크 내 실데이터 오프셋만 생성하므로 패딩 불필요.
    let ch = total.next_power_of_two().min(1usize << (63 - ch.leading_zeros()));
    while off < total {
        let sz = ch.min(total - off);
        let mut b = ctx.alloc(sz)?;
        unsafe { std::ptr::copy_nonoverlapping(w.data.as_ptr().add(off), b.ptr, sz) };
        ctx.unmap(&mut b)?;
        wbufs.push(b.buf);
        off += sz;
    }
    let mut dummy = ctx.alloc_host(16)?;
    {
        let z = [0u8; 16];
        unsafe { std::ptr::copy_nonoverlapping(z.as_ptr(), dummy.ptr, 16) };
    }
    while wbufs.len() < 8 {
        wbufs.push(dummy.buf);
    }
    let chunk_words = (ch / 4) as u32;
    let spv_path = match w.ty {
        llm170_gguf::GgmlType::Q3K => "crates/backend-gpu/src/rawvk/spv/gemv8_q3.spv",
        llm170_gguf::GgmlType::Q4K => "crates/backend-gpu/src/rawvk/spv/gemv8_q4.spv",
        llm170_gguf::GgmlType::Q5K => "crates/backend-gpu/src/rawvk/spv/gemv8_q5.spv",
        llm170_gguf::GgmlType::Q6K => "crates/backend-gpu/src/rawvk/spv/gemv8_q6.spv",
        llm170_gguf::GgmlType::Iq4Xs => "crates/backend-gpu/src/rawvk/spv/gemv8_xs.spv",
        _ => return Err("gemv8: 미지원 타입".into()),
    };
    let spv = std::fs::read(spv_path).map_err(|e| e.to_string())?;
    let (kb, _gb, _db) = acc.ensure_shared(&mut ctx)?;
    let n_kb_h = if is_xs { 12 } else { 10 };
    let (dsl, pl, pool, ds, pipe) = ctx.pipeline(&spv, n_kb_h, 24)?;
    let _ = (dsl, pool);
    let mut binds: Vec<vk::Buffer> = wbufs.clone();
    binds.push(xa.buf);
    binds.push(ob.buf);
    if is_xs {
        binds.push(kb);
    }
    ctx.bind_bufs(ds, &binds);
    let rpf: u32 = if n_out < 4096 { 1 } else { 2 };   // llama NUM_ROWS=2
    let cw_log2 = 31u32 - chunk_words.leading_zeros();
    let cw_mask = (1u32 << cw_log2) - 1u32;
    // cw 단위: q5/q6(u16 typed 뷰)만 u16 단위, 나머지 u32
    let (cwpl, cwpm) = if is_q5 || is_q6 {
        (31u32 - (chunk_words * 2).leading_zeros(), (chunk_words * 2) - 1)
    } else { (cw_log2, cw_mask) };
    let push = push_u32s(&[n_in as u32, n_out as u32, t as u32, cwpl, cwpm, rpf]);
    ctx.run(pl, ds, pipe, &push, 1, n_out.div_ceil(rpf as usize) as u32, t as u32)?;
    let outs: Vec<f32> = unsafe {
        let mut v = vec![0f32; t * n_out];
        std::ptr::copy_nonoverlapping(ob.ptr as *const f32, v.as_mut_ptr(), t * n_out);
        v
    };
    // CPU 기준: 디양자화 내적
    let mut mx = 0f64;
    let mut ref_row = vec![0.0f32; n_in];
    for (j, x) in xs.iter().enumerate() {
        for r in 0..n_out.min(64) {
            llm170_core::quant::dequant_row(
                w.ty, w.data, r as u64, n_in as u64, &mut ref_row);
            let dot: f32 = ref_row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            mx = mx.max((dot - outs[j * n_out + r]).abs() as f64);
        }
    }
    let solo_t0 = Instant::now();
    for _ in 0..10 {
        ctx.run(pl, ds, pipe, &push, 1, n_out.div_ceil(rpf as usize) as u32, t as u32)?;
    }
    let solo_dt = solo_t0.elapsed().as_secs_f64() / 10.0;
    // L2 플러시 타이밍 — 반복 사이 자기 자신을 12회 연속 돌린 뒤
    // '매 반복 직전 타 텐서 1회' 교차 판독으로 캐시 몰아내기 (L2FLUSH=1).
    let flushed_dt: f64 = if std::env::var_os("LLM170_L2FLUSH").is_some() {
        // MULTI로 등록한 첫 extra 텐서를 플러시용으로 재사용: 그 weights로
        // 동일 커널 1회 (다른 ds/push 필요) — 여기선 간단히 xa를 8MB 재기록 후
        // 측정 대상 run 직전 xa 전체 재업로드 (호스트 memcpy가 L2 오염)
        let t4 = Instant::now();
        let xa2 = xs[0].clone();
        for _ in 0..10 {
            unsafe {
                std::ptr::copy_nonoverlapping(xa2.as_ptr(), xa.ptr as *mut f32, n_in);
            }
            ctx.run(pl, ds, pipe, &push, 1, n_out.div_ceil(rpf as usize) as u32, t as u32)?;
        }
        t4.elapsed().as_secs_f64() / 10.0
    } else { 0.0 };
    if flushed_dt > 0.0 {
        return Ok(format!(
            "gemv8-l2flush({tname}): {:.3}ms → {:.1}GB/s (웜 {})",
            flushed_dt * 1e3, w.data.len() as f64 / flushed_dt / 1e9,
            w.data.len() as f64 / solo_dt / 1e9
        ));
    }
    if let Ok(list) = std::env::var("LLM170_MULTI") {
        // TLB/할당수 가설: 추가 텐서들을 같은 컨텍스트에 로드(상주)시킨 뒤
        // 이 텐서의 타이밍 재측정 — 속도 붕괴 시 가설 확인.
        for extra in list.split(',').filter(|x| !x.is_empty()) {
            if extra == tname { continue; }
            let w2 = match model.w(extra) { Some(w) => w, None => continue };
            let mut off2 = 0usize;
            let tot2 = w2.data.len();
            while off2 < tot2 {
                let sz2 = ch.min(tot2 - off2);
                let mut b2 = ctx.alloc(sz2)?;
                unsafe { std::ptr::copy_nonoverlapping(w2.data.as_ptr().add(off2), b2.ptr, sz2) };
                ctx.unmap(&mut b2)?;
                off2 += sz2;
            }
        }
        let t2 = Instant::now();
        for _ in 0..10 {
            ctx.run(pl, ds, pipe, &push, 1, n_out.div_ceil(rpf as usize) as u32, t as u32)?;
        }
        let dt2 = t2.elapsed().as_secs_f64() / 10.0;
        let _ = &solo_dt;
        return Ok(format!(
            "gemv8-multi({tname}): {:.3}ms → {:.1}GB/s (단독 {:.1})",
            dt2 * 1e3, w.data.len() as f64 / dt2 / 1e9, w.data.len() as f64 / solo_dt / 1e9
        ));
    }
    Ok(format!(
        "gemv8({tname}) t={t}: {:.3}ms → {:.1}GB/s · max|D|={mx:.4}",
        solo_dt * 1e3,
        w.data.len() as f64 / solo_dt / 1e9
    ))
}

/// vk-tile-check — 타일(coopmat f16) 커널 vs CPU 디양자화 GEMM 검증 (plans/38 A2).
/// f16 스테이징 품질계약: maxrel 허용치 ~2e-2 (근접 아닌 구조 오류 검출 목적).
pub fn tile_check(path: &str, tname: &str, t: usize) -> Result<String, String> {
    use std::time::Instant;
    let model = llm170_core::model::Model::load(std::path::Path::new(path))
        .map_err(|e| e.to_string())?;
    let w = model.w(tname).ok_or("텐서 없음")?;
    let n_in = w.n_in as usize;
    let n_out = w.n_out as usize;
    if t < 1 || t > 64 {
        return Err("tile 검증 t는 1..=64".into());
    }
    let (spv_name, n_kb, extra) = match w.ty {
        llm170_gguf::GgmlType::Q5K if std::env::var("LLM170_TILE_V2").as_deref() == Ok("2") => ("tile128v2_dbg.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q5K if std::env::var_os("LLM170_TILE_V2").is_some() => ("tile128v2.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q5K => ("tile128_q5k.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q4K => ("tile_q4k.spv", 10, 0),
        llm170_gguf::GgmlType::Q6K => ("tile_q6k.spv", 10, 0),
        llm170_gguf::GgmlType::Q3K => ("tile_q3k.spv", 10, 0),
        llm170_gguf::GgmlType::Q8_0 => ("tile_q8.spv", 10, 0),
        llm170_gguf::GgmlType::Iq4Xs => ("tile_xs.spv", 11, 1),   // ktab
        llm170_gguf::GgmlType::Iq4Nl => ("tile_nl.spv", 11, 1),    // ktab
        llm170_gguf::GgmlType::Iq3S => ("tile_iq3s.spv", 11, 2),   // grid3s
        _ => return Err("tile 검증 불가 타입".into()),
    };
    let is_128 = w.ty == llm170_gguf::GgmlType::Q5K && std::env::var_os("LLM170_TILE_V2").is_none();
    let v2dbg = std::env::var("LLM170_TILE_V2").as_deref() == Ok("2");
    let acc = VkAcc::new()?;
    let mut ctx = acc.ctx.lock();
    let mut seed = 0x5deece66u64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as f32 / 2147483648.0 - 0.5
    };
    let xs: Vec<Vec<f32>> = (0..t).map(|_| (0..n_in).map(|_| lcg()).collect()).collect();
    // xq 양자화 (GPU quant — 비트 검증 완료 경로)
    let xq_w = n_in / 4 + n_in / 32 + n_in / 16;
    let xqb = ctx.alloc_host(t * xq_w * 4)?;
    acc.quant_upload(&mut ctx, &xs, n_in, xqb.buf)?;
    let ob = ctx.alloc_host(t * n_out * 4)?;
    // 가중 업로드
    let total = w.data.len();
    let ch = total.next_power_of_two().min(1usize << (63 - ctx.max_ssbo.leading_zeros()));
    let mut wbufs = Vec::new();
    let mut off = 0usize;
    while off < total {
        let sz = ch.min(total - off);
        let mut b = ctx.alloc(sz)?;
        unsafe { std::ptr::copy_nonoverlapping(w.data.as_ptr().add(off), b.ptr, sz) };
        ctx.unmap(&mut b)?;
        wbufs.push(b.buf);
        off += sz;
    }
    let (ktab, grid, dummy) = acc.ensure_shared(&mut ctx)?;
    while wbufs.len() < 8 {
        wbufs.push(dummy);
    }
    let spv = std::fs::read(format!("crates/backend-gpu/src/rawvk/spv/{spv_name}"))
        .map_err(|e| e.to_string())?;
    let pb: u32 = if is_128 { 16 } else { 24 };
    let (dsl, pl, pool, ds, pipe) = ctx.pipeline(&spv, n_kb, pb)?;
    let _ = (dsl, pool);
    let mut binds: Vec<vk::Buffer> = wbufs.clone();
    binds.push(xqb.buf);
    binds.push(ob.buf);
    if extra == 1 {
        binds.push(ktab);
    } else if extra == 2 {
        binds.push(grid);
    }
    ctx.bind_bufs(ds, &binds);
    let cw = (ch / 4) as u32;
    let cw = cw.next_power_of_two();
    let cw_log2 = 31u32 - cw.leading_zeros();
    let cw_mask = cw - 1;
    let gx = (n_out as u32 + 127) / 128;
    let t0 = Instant::now();
    if is_128 {
        let push = push_u32s(&[n_in as u32, n_out as u32, xq_w as u32, t as u32]);
        ctx.run(pl, ds, pipe, &push, gx, 1, 1)?;
    } else {
        let push = push_u32s(&[n_in as u32, n_out as u32, xq_w as u32, t as u32, cw_log2, cw_mask]);
        ctx.run(pl, ds, pipe, &push, gx, 1, 1)?;
    }
    let outs: Vec<f32> = unsafe {
        let mut v = vec![0f32; t * n_out];
        std::ptr::copy_nonoverlapping(ob.ptr as *const f32, v.as_mut_ptr(), t * n_out);
        v
    };
    let dt = t0.elapsed().as_secs_f64();
    if v2dbg {
        // B≡1 → out[t][row] = Σ 디양자화 행
        let mut ref_row2 = vec![0.0f32; n_in];
        let mut worst = (0usize, 0f64, 0f64);
        for r in 0..n_out.min(64) {
            llm170_core::quant::dequant_row(w.ty, w.data, r as u64, n_in as u64, &mut ref_row2);
            let s: f64 = ref_row2.iter().map(|&v| v as f64).sum();
            let g = outs[r] as f64;
            let rel = (g - s).abs() / s.abs().max(1.0);
            if rel > worst.1 { worst = (r, rel, s); }
        }
        return Ok(format!("tile-v2dbg({tname}) t={t}: rowsum maxrel={:.4} (r={} ref={:.4})", worst.1, worst.0, worst.2));
    }
    // CPU 기준: 디양자화 · f64 내적 — 행 0..64 × 전 토큰
    let mut ref_row = vec![0.0f32; n_in];
    let mut maxrel = 0f64;
    let mut worst = (0usize, 0usize, 0f64, 0f64);
    let mut bad_rows = 0usize;
    for (j, x) in xs.iter().enumerate() {
        let mut row_bad = false;
        for r in 0..n_out.min(64) {
            llm170_core::quant::dequant_row(w.ty, w.data, r as u64, n_in as u64, &mut ref_row);
            let dot: f64 = ref_row.iter().zip(x.iter()).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
            let g = outs[j * n_out + r] as f64;
            let rel = (g - dot).abs() / dot.abs().max(1.0);
            if rel > maxrel {
                maxrel = rel;
                worst = (j, r, dot, g);
            }
            if rel > 2e-2 {
                row_bad = true;
            }
        }
        if row_bad {
            bad_rows += 1;
        }
    }
    Ok(format!(
        "tile({tname}/{spv_name}) t={t}: {dt:.3}ms · maxrel={maxrel:.4} (worst j={} r={} ref={:.4} gpu={:.4}) · 2%초과 토큰 {bad_rows}/{t}",
        worst.0, worst.1, worst.2, worst.3
    ))
}

