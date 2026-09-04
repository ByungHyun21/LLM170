//! VkAcc — Vulkan matmul 가속기 (plans/12·13). HIP과 병립:
//! LLM170_GPU_RUNTIME=vulkan 시 주입, GDN/프레임은 CPU 폴백 (트레이트 Err).
//! 구조: 파이프라인·버퍼·가중치는 전부 지연 초기화 캐시, dispatch 헬퍼가
//! SSBO 바인딩+push+발사를 일원화 (M4b 확장 지점).

use crate::rawvk::context::{VkBuf, VkCtx};
use ash::vk;
use llm170_core::matmul::{Accelerator, Weight};
use llm170_gguf::GgmlType;
use parking_lot::Mutex;
use std::collections::HashMap;

const GEMV_SPV: &[u8] = include_bytes!("spv/gemv3.spv");
const TILE128_SPV: &[u8] = include_bytes!("spv/tile128_q5k.spv");
const QUANT_SPV: &[u8] = include_bytes!("spv/quant_q8.spv");
const RMS_SPV: &[u8] = include_bytes!("spv/rms.spv");
const SILU_SPV: &[u8] = include_bytes!("spv/silu_mul.spv");

/// 파이프라인 세트 (vk 핸들은 복사 가능).
#[derive(Clone, Copy)]
struct Pipes {
    pl: vk::PipelineLayout,
    ds: vk::DescriptorSet,
    pipe: vk::Pipeline,
    dsl: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
}

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
            Slot::Gemv => (GEMV_SPV, 12, 20u32),
            Slot::Tile128 => (TILE128_SPV, 10, 16),
            Slot::Quant => (QUANT_SPV, 2, 12),
            Slot::Rms => (RMS_SPV, 3, 12),
            Slot::Silu => (SILU_SPV, 3, 4),
        };
        let (dsl, pl, dp, ds, pipe) = ctx.pipeline(spv, n_buf, pb)?;
        let p = Pipes { pl, ds, pipe, dsl, pool: dp };
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
            let kv: Vec<u32> = (0..256u32)
                .map(|b| {
                    let lo = llm170_core::KVALUES_IQ4NL[(b & 0xF) as usize] as u8 as u32;
                    let hi = llm170_core::KVALUES_IQ4NL[(b >> 4) as usize] as u8 as u32;
                    lo | (hi << 8)
                })
                .collect();
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
                let ch = ctx.max_ssbo;
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
        let ds2 = self.bind_ds(ctx, &p, &binds)?;
        let push = push_u32s(&[n_in as u32, n_out as u32, xq_w as u32, ty, t as u32]);
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
        let ds2 = self.bind_ds(ctx, &p, &binds)?;
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
        let ds2 = self.bind_ds(ctx, &p, &[xfbuf, xq_buf])?;
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
        let ds2 = self.bind_ds(&mut ctx, &p, &[xb, wb, ob])?;
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
        let ds2 = self.bind_ds(&mut ctx, &p, &[gb, ub, ob])?;
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
            let ds2 = self.bind_ds(&mut ctx, &p, &[xbf, bq0])?;
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
            let ds2 = self.bind_ds(&mut ctx, &p, &[bfg, bfu, bglu])?;
            let total = (t * n_ff) as u32;
            ctx.run(p.pl, ds2, p.pipe, &total.to_le_bytes(), total.div_ceil(256), 1, 1)?;
        }
        // 4) glu quant(n_ff)
        {
            // bglu는 f32가 아니라 f32→q8 변환 입력 — quant 셰이더에 직접.
            // (bglu는 silu 출력 f32 → quant가 읽는다)
            let p = self.pipeline(&mut ctx, Slot::Quant)?;
            let ds2 = self.bind_ds(&mut ctx, &p, &[bglu, bq1])?;
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
    use llm170_core::matmul::Accelerator;
    let model = llm170_core::model::Model::load(std::path::Path::new(path))
        .map_err(|e| e.to_string())?;
    let w = model.w(tname).ok_or("텐서 없음")?;
    let wref = &w;
    let n_in = w.n_in as usize;
    let acc = VkAcc::new()?;
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
    for (a, b) in outs.iter().zip(ref_outs.iter()) {
        for (x, y) in a.iter().zip(b.iter()) {
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
    let ia = outs[0].iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i);
    let ib = ref_outs[0].iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i);
    Ok(format!(
        "vk-gemv {tname} t={t}: max|D|={mx:.3e} maxrel={rel:.2e} argmax {ia:?}=={ib:?} {}",
        if ia == ib { "★" } else { "MISMATCH" }
    ))
}
