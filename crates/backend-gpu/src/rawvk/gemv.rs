//! VkAcc — Vulkan matmul 가속기 (plans/12 M2). HIP과 병립:
//! LLM170_GPU_RUNTIME=vulkan 시 주입, GDN/프레임은 CPU 폴백 (트레이트 Err).

use crate::rawvk::context::{VkBuf, VkCtx};
use ash::vk;
use llm170_core::matmul::{Accelerator, Weight};
use llm170_gguf::GgmlType;
use std::collections::HashMap;
use parking_lot::Mutex;

const GEMV_SPV: &[u8] = include_bytes!("spv/gemv.spv");
const TILE_Q5K_SPV: &[u8] = include_bytes!("spv/tile_q5k.spv");
const QUANT_SPV: &[u8] = include_bytes!("spv/quant_q8.spv");
const RMS_SPV: &[u8] = include_bytes!("spv/rms.spv");

struct Pipes {
    dsl: vk::DescriptorSetLayout,
    pl: vk::PipelineLayout,
    dp: vk::DescriptorPool,
    ds: vk::DescriptorSet,
    pipe: vk::Pipeline,
}

struct TilePipes {
    dsl: vk::DescriptorSetLayout,
    pl: vk::PipelineLayout,
    dp: vk::DescriptorPool,
    ds: vk::DescriptorSet,
    pipe: vk::Pipeline,
}

struct RmsPipes {
    dsl: vk::DescriptorSetLayout,
    pl: vk::PipelineLayout,
    dp: vk::DescriptorPool,
    ds: vk::DescriptorSet,
    pipe: vk::Pipeline,
}

struct QuantPipes {
    dsl: vk::DescriptorSetLayout,
    pl: vk::PipelineLayout,
    dp: vk::DescriptorPool,
    ds: vk::DescriptorSet,
    pipe: vk::Pipeline,
}

pub struct VkAcc {
    ctx: Mutex<VkCtx>,
    tpipes: Mutex<Option<TilePipes>>,
    rpipe: Mutex<Option<RmsPipes>>,
    rbufs: Mutex<Option<(VkBuf, VkBuf, VkBuf)>>,
    qpipes: Mutex<Option<QuantPipes>>,
    xfbuf: Mutex<Option<VkBuf>>,
    tables: Mutex<Option<(VkBuf, VkBuf)>>,
    dummy: Mutex<Option<VkBuf>>,
    pipes: Mutex<Option<Pipes>>,
    /// 가중치 캐시 (데이터 포인터 → 상주 버퍼)
    wcache: Mutex<HashMap<usize, Vec<VkBuf>>>,
    xbuf: Mutex<Option<VkBuf>>,
    obuf: Mutex<Option<VkBuf>>,
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

impl VkAcc {
    pub fn new() -> Result<Self, String> {
        let ctx = VkCtx::new()?;
        if !ctx.coop_matrix {
            eprintln!("rawvk: coop matrix 미지원 (타일 경로 M3에서 필요)");
        }
        Ok(Self {
            ctx: Mutex::new(ctx),
            tpipes: Mutex::new(None),
            rpipe: Mutex::new(None),
            rbufs: Mutex::new(None),
            qpipes: Mutex::new(None),
            xfbuf: Mutex::new(None),
            tables: Mutex::new(None),
            dummy: Mutex::new(None),
            pipes: Mutex::new(None),
            wcache: Mutex::new(HashMap::new()),
            xbuf: Mutex::new(None),
            obuf: Mutex::new(None),
        })
    }

    fn qpipes(&self, ctx: &mut VkCtx) -> Result<QuantPipes, String> {
        let mut g = self.qpipes.lock();
        if g.is_none() {
            let (dsl, pl, dp, ds, pipe) = ctx.pipeline(QUANT_SPV, 2, 12)?;
            *g = Some(QuantPipes { dsl, pl, dp, ds, pipe });
        }
        let r = g.as_ref().unwrap();
        Ok(QuantPipes { dsl: r.dsl, pl: r.pl, dp: r.dp, ds: r.ds, pipe: r.pipe })
    }

    fn tpipes(&self, ctx: &mut VkCtx) -> Result<TilePipes, String> {
        let mut g = self.tpipes.lock();
        if g.is_none() {
            let (dsl, pl, dp, ds, pipe) = ctx.pipeline(TILE_Q5K_SPV, 10, 16)?;
            *g = Some(TilePipes { dsl, pl, dp, ds, pipe });
        }
        let r = g.as_ref().unwrap();
        Ok(TilePipes { dsl: r.dsl, pl: r.pl, dp: r.dp, ds: r.ds, pipe: r.pipe })
    }

    fn pipes(&self, ctx: &mut VkCtx) -> Result<Pipes, String> {
        let mut g = self.pipes.lock();
        if g.is_none() {
            let (dsl, pl, dp, ds, pipe) = ctx.pipeline(GEMV_SPV, 12, 16)?;
            *g = Some(Pipes { dsl, pl, dp, ds, pipe });
        }
        Ok(Pipes { ..g.as_ref().unwrap().clone() })
    }
}

// vk 핸들은 복사 가능 (NonNull 래퍼) — 파이프라인 세트 공유용 Clone.
impl Clone for Pipes {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for Pipes {}

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
            let need_x = t * n * 4;
            let need_w = n * 4;
            let ok = b.as_ref().map(|(x, _, _)| x.bytes >= need_x && x.bytes > 0).unwrap_or(false);
            if !ok {
                let xb = ctx.alloc(need_x.max(1 << 21))?;
                let wb = ctx.alloc(need_w)?;
                let ob = ctx.alloc(need_x.max(1 << 21))?;
                *b = Some((xb, wb, ob));
            }
        }
        let (xb, wb, ob) = {
            let b = self.rbufs.lock();
            let (a, bb, c) = b.as_ref().unwrap();
            (a.buf, bb.buf, c.buf)
        };
        {
            let b = self.rbufs.lock();
            let (xv, wv, _) = b.as_ref().unwrap();
            for (ti, row) in xs.iter().enumerate() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        row.as_ptr(),
                        xv.ptr.add(ti * n * 4) as *mut f32,
                        n,
                    );
                }
            }
            unsafe { std::ptr::copy_nonoverlapping(w.as_ptr(), wv.ptr as *mut f32, n); }
        }
        let p = {
            let mut g = self.rpipe.lock();
            if g.is_none() {
                let (dsl, pl, dp, ds, pipe) = ctx.pipeline(RMS_SPV, 3, 12)?;
                *g = Some(RmsPipes { dsl, pl, dp, ds, pipe });
            }
            let r = g.as_ref().unwrap();
            RmsPipes { dsl: r.dsl, pl: r.pl, dp: r.dp, ds: r.ds, pipe: r.pipe }
        };
        ctx.bind_bufs(p.ds, &[xb, wb, ob]);
        let mut push = Vec::with_capacity(12);
        push.extend_from_slice(&(n as u32).to_le_bytes());
        push.extend_from_slice(&(t as u32).to_le_bytes());
        push.extend_from_slice(&eps.to_le_bytes());
        ctx.run(p.pl, p.ds, p.pipe, &push, t as u32, 1, 1)?;
        let host = {
            let b = self.rbufs.lock();
            let (_, _, ov) = b.as_ref().unwrap();
            unsafe { std::slice::from_raw_parts(ov.ptr as *const f32, t * n) }
        };
        for ti in 0..t {
            outs[ti].copy_from_slice(&host[ti * n..(ti + 1) * n]);
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
                // 미지원 타입(q8_0/nl/q3k/iq3s) — CPU W4A8 경로
                llm170_core::matmul::matmul_batch(xs, w, outs);
                return Ok(());
            }
        };
        let n_in = w.n_in as usize;
        let n_out = w.n_out as usize;
        let t = xs.len();
        let xq_w = n_in / 4 + n_in / 32 + n_in / 16;
        let mut ctx = self.ctx.lock();
        // GPU 양자화: xs(f32) 업로드 → quant_q8 → xq (CPU 부담 제거)
        {
            let mut xf = self.xfbuf.lock();
            let need = t * n_in * 4;
            let ok = xf.as_ref().map(|b| b.bytes >= need).unwrap_or(false);
            if !ok {
                let b = ctx.alloc(need.max(1 << 21))?;
                *xf = Some(b);
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
        {
            let mut xb = self.xbuf.lock();
            let need = t * xq_w * 4;
            let ok = xb.as_ref().map(|b| b.bytes >= need).unwrap_or(false);
            if !ok {
                let b = ctx.alloc(need.max(1 << 21))?;
                *xb = Some(b);
            }
        }
        {
            let qp = self.qpipes(&mut ctx)?;
            let xfbuf = self.xfbuf.lock().as_ref().unwrap().buf;
            let xqbuf = self.xbuf.lock().as_ref().unwrap().buf;
            ctx.bind_bufs(qp.ds, &[xfbuf, xqbuf]);
            let mut push = Vec::with_capacity(12);
            push.extend_from_slice(&(n_in as u32).to_le_bytes());
            push.extend_from_slice(&(t as u32).to_le_bytes());
            push.extend_from_slice(&(xq_w as u32).to_le_bytes());
            ctx.run(qp.pl, qp.ds, qp.pipe, &push, ((n_in / 32) + 63) as u32 / 64, t as u32, 1)?;
        }
        // 가중 상주 (ptr 키 — mmap 안정) — 128MB 청크 분할 (maxStorageBufferRange)
        {
            let mut wc = self.wcache.lock();
            let key = w.data.as_ptr() as usize;
            if !wc.contains_key(&key) {
                let ch = ctx.max_ssbo;
                let total = w.data.len();
                let mut bufs = Vec::new();
                let mut off = 0usize;
                while off < total {
                    let n = ch.min(total - off);
                    let b = ctx.alloc(n)?;
                    unsafe {
                        std::ptr::copy_nonoverlapping(w.data.as_ptr().add(off), b.ptr, n);
                    }
                    bufs.push(b);
                    off += n;
                }
                wc.insert(key, bufs);
            }
        }
        let wbufs: Vec<_> = {
            let wc = self.wcache.lock();
            wc.get(&(w.data.as_ptr() as usize)).unwrap().iter().map(|b| b.buf).collect()
        };
        if wbufs.len() > 8 {
            return Err(format!("가중 청크 {}개 > 8 슬롯 (M2 한계)", wbufs.len()));
        }
        {
            let mut ob = self.obuf.lock();
            let need = t * n_out * 4;
            let ok = ob.as_ref().map(|b| b.bytes >= need).unwrap_or(false);
            if !ok {
                let b = ctx.alloc(need.max(1 << 20))?;
                *ob = Some(b);
            }
        }
        // coopmat 타일 경로 (q5_K, t≥2): 16토큰 블록 × 16행 그룹 — M3
        // 타일 경로 (f16 스테이징 → maxrel ~4e-4) — 스트림 게이트 민감해 env 게이트.
        // 기본 GEMV(비트근사 exact) — LLM170_VK_TILE=1로 타일 활성.
        if ty == 13 && t >= 2 && std::env::var_os("LLM170_VK_TILE").is_some() {
            // 더미
            {
                let mut d = self.dummy.lock();
                if d.is_none() {
                    let b = ctx.alloc(16)?;
                    *d = Some(b);
                }
            }
            let dbuf = self.dummy.lock().as_ref().unwrap().buf;
            let p = self.tpipes(&mut ctx)?;
            let mut binds: Vec<vk::Buffer> = wbufs.clone();
            while binds.len() < 8 {
                binds.push(dbuf);
            }
            let xb0 = self.xbuf.lock().as_ref().unwrap().buf;
            let ob0 = self.obuf.lock().as_ref().unwrap().buf;
            binds.push(xb0);
            binds.push(ob0);
            ctx.bind_bufs(p.ds, &binds);
            let gx = (n_out + 15) as u32 / 16;
            for tb in (0..t).step_by(16) {
                let nt = (t - tb).min(16) as u32;
                let mut push = Vec::with_capacity(16);
                push.extend_from_slice(&(n_in as u32).to_le_bytes());
                push.extend_from_slice(&(n_out as u32).to_le_bytes());
                push.extend_from_slice(&(xq_w as u32).to_le_bytes());
                push.extend_from_slice(&nt.to_le_bytes());
                ctx.run(p.pl, p.ds, p.pipe, &push, gx, 1, 1)?;
            }
            let host = unsafe {
                std::slice::from_raw_parts(
                    self.obuf.lock().as_ref().unwrap().ptr as *const f32,
                    t * n_out,
                )
            };
            for ti in 0..t {
                outs[ti].copy_from_slice(&host[ti * n_out..(ti + 1) * n_out]);
            }
            return Ok(());
        }
        let xb = self.xbuf.lock().as_ref().unwrap().buf;
        let ob = self.obuf.lock().as_ref().unwrap().buf;
        let p = self.pipes(&mut ctx)?;
        // ktab(iq4nl) + grid3s 업로드 (최초 1회)
        {
            let mut tb = self.tables.lock();
            if tb.is_none() {
                let kv: Vec<u32> = (0..256u32)
                    .map(|b| {
                        let lo = llm170_core::KVALUES_IQ4NL[(b & 0xF) as usize] as u8 as u32;
                        let hi = llm170_core::KVALUES_IQ4NL[(b >> 4) as usize] as u8 as u32;
                        lo | (hi << 8)
                    })
                    .collect();
                let kb = ctx.alloc(1024)?;
                unsafe { std::ptr::copy_nonoverlapping(kv.as_ptr() as *const u8, kb.ptr, 1024) };
                let gb = ctx.alloc(2048)?;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        llm170_core::IQ3S_GRID.as_ptr() as *const u8,
                        gb.ptr,
                        2048,
                    );
                }
                *tb = Some((kb, gb));
            }
        }
        // 미사용 청크 슬롯: 더미 버퍼 (null 바인딩 방지)
        {
            let mut d = self.dummy.lock();
            if d.is_none() {
                let b = ctx.alloc(16)?;
                *d = Some(b);
            }
        }
        let dbuf = self.dummy.lock().as_ref().unwrap().buf;
        let mut binds: Vec<vk::Buffer> = wbufs.clone();
        while binds.len() < 8 {
            binds.push(dbuf);
        }
        binds.push(xb);
        binds.push(ob);
        let (kb, gb) = {
            let tb = self.tables.lock();
            let (a, b) = tb.as_ref().unwrap();
            (a.buf, b.buf)
        };
        binds.push(kb);
        binds.push(gb);
        ctx.bind_bufs(p.ds, &binds);
        let mut push = Vec::with_capacity(16);
        push.extend_from_slice(&(n_in as u32).to_le_bytes());
        push.extend_from_slice(&(n_out as u32).to_le_bytes());
        push.extend_from_slice(&(xq_w as u32).to_le_bytes());
        push.extend_from_slice(&ty.to_le_bytes());
        ctx.run(p.pl, p.ds, p.pipe, &push, n_out as u32, t as u32, 1)?;
        let host = unsafe {
            std::slice::from_raw_parts(
                self.obuf.lock().as_ref().unwrap().ptr as *const f32,
                t * n_out,
            )
        };
        for ti in 0..t {
            outs[ti].copy_from_slice(&host[ti * n_out..(ti + 1) * n_out]);
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

    /// 같은 입력 그룹: 업로드+양자화 1회 → GEMV 각각 → 개별 다운로드.
    fn matmul_group(
        &self,
        xs: &[Vec<f32>],
        ws: &[Weight],
        outs: &mut [Vec<Vec<f32>>],
    ) -> Result<(), String> {
        // 지원 타입 혼합 그룹은 개별 폴백
        if ws.iter().any(|w| vk_ty(w.ty).is_none()) {
            for (w, out) in ws.iter().zip(outs.iter_mut()) {
                self.matmul_batch(xs, w, out)?;
            }
            return Ok(());
        }
        let n_in = ws[0].n_in as usize;
        if ws.iter().any(|w| w.n_in as usize != n_in) {
            for (w, out) in ws.iter().zip(outs.iter_mut()) {
                self.matmul_batch(xs, w, out)?;
            }
            return Ok(());
        }
        let t = xs.len();
        let xq_w = n_in / 4 + n_in / 32 + n_in / 16;
        let mut ctx = self.ctx.lock();
        // 업로드+양자화 1회 (matmul_batch와 동일 로직)
        {
            let mut xf = self.xfbuf.lock();
            let need = t * n_in * 4;
            let ok = xf.as_ref().map(|b| b.bytes >= need).unwrap_or(false);
            if !ok {
                let b = ctx.alloc(need.max(1 << 21))?;
                *xf = Some(b);
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
        {
            let mut xb = self.xbuf.lock();
            let need = t * xq_w * 4;
            let ok = xb.as_ref().map(|b| b.bytes >= need).unwrap_or(false);
            if !ok {
                let b = ctx.alloc(need.max(1 << 21))?;
                *xb = Some(b);
            }
        }
        {
            let qp = self.qpipes(&mut ctx)?;
            let xfbuf = self.xfbuf.lock().as_ref().unwrap().buf;
            let xqbuf = self.xbuf.lock().as_ref().unwrap().buf;
            ctx.bind_bufs(qp.ds, &[xfbuf, xqbuf]);
            let mut push = Vec::with_capacity(12);
            push.extend_from_slice(&(n_in as u32).to_le_bytes());
            push.extend_from_slice(&(t as u32).to_le_bytes());
            push.extend_from_slice(&(xq_w as u32).to_le_bytes());
            ctx.run(qp.pl, qp.ds, qp.pipe, &push, ((n_in / 32) + 63) as u32 / 64, t as u32, 1)?;
        }
        // 가중 캐시 (matmul_batch 공용 로직)
        let tables = {
            let mut tb = self.tables.lock();
            if tb.is_none() {
                let kv: Vec<u32> = (0..256u32)
                    .map(|b| {
                        let lo = llm170_core::KVALUES_IQ4NL[(b & 0xF) as usize] as u8 as u32;
                        let hi = llm170_core::KVALUES_IQ4NL[(b >> 4) as usize] as u8 as u32;
                        lo | (hi << 8)
                    })
                    .collect();
                let kb = ctx.alloc(1024)?;
                unsafe { std::ptr::copy_nonoverlapping(kv.as_ptr() as *const u8, kb.ptr, 1024) };
                let gb = ctx.alloc(2048)?;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        llm170_core::IQ3S_GRID.as_ptr() as *const u8,
                        gb.ptr,
                        2048,
                    );
                }
                *tb = Some((kb, gb));
            }
            let (a, b) = tb.as_ref().unwrap();
            (a.buf, b.buf)
        };
        let p = self.pipes(&mut ctx)?;
        let xqbuf = self.xbuf.lock().as_ref().unwrap().buf;
        for (wi, w) in ws.iter().enumerate() {
            let ty = vk_ty(w.ty).unwrap();
            let n_out = w.n_out as usize;
            let wbufs: Vec<_> = {
                let mut wc = self.wcache.lock();
                let key = w.data.as_ptr() as usize;
                if !wc.contains_key(&key) {
                    let ch = ctx.max_ssbo;
                    let total = w.data.len();
                    let mut bufs = Vec::new();
                    let mut off = 0usize;
                    while off < total {
                        let nb = ch.min(total - off);
                        let b = ctx.alloc(nb)?;
                        unsafe {
                            std::ptr::copy_nonoverlapping(w.data.as_ptr().add(off), b.ptr, nb);
                        }
                        bufs.push(b);
                        off += nb;
                    }
                    wc.insert(key, bufs);
                }
                wc.get(&key).unwrap().iter().map(|b| b.buf).collect()
            };
            if wbufs.len() > 8 {
                return Err("가중 청크 > 8 (M2 한계)".into());
            }
            {
                let mut ob = self.obuf.lock();
                let need = t * n_out * 4;
                let ok = ob.as_ref().map(|b| b.bytes >= need).unwrap_or(false);
                if !ok {
                    let b = ctx.alloc(need.max(1 << 21))?;
                    *ob = Some(b);
                }
            }
            let ob = self.obuf.lock().as_ref().unwrap().buf;
            {
                let mut d = self.dummy.lock();
                if d.is_none() {
                    let b = ctx.alloc(16)?;
                    *d = Some(b);
                }
            }
            let dbuf = self.dummy.lock().as_ref().unwrap().buf;
            let mut binds: Vec<vk::Buffer> = wbufs.clone();
            while binds.len() < 8 {
                binds.push(dbuf);
            }
            binds.push(xqbuf);
            binds.push(ob);
            binds.push(tables.0);
            binds.push(tables.1);
            ctx.bind_bufs(p.ds, &binds);
            let mut push = Vec::with_capacity(16);
            push.extend_from_slice(&(n_in as u32).to_le_bytes());
            push.extend_from_slice(&(n_out as u32).to_le_bytes());
            push.extend_from_slice(&(xq_w as u32).to_le_bytes());
            push.extend_from_slice(&ty.to_le_bytes());
            ctx.run(p.pl, p.ds, p.pipe, &push, n_out as u32, t as u32, 1)?;
            let host = unsafe {
                std::slice::from_raw_parts(self.obuf.lock().as_ref().unwrap().ptr as *const f32, t * n_out)
            };
            for ti in 0..t {
                outs[wi][ti].copy_from_slice(&host[ti * n_out..(ti + 1) * n_out]);
            }
        }
        Ok(())
    }

    fn matmul(&self, x: &[f32], w: &Weight, out: &mut [f32]) -> Result<(), String> {
        let xs = vec![x.to_vec()];
        let mut tmp = vec![vec![0.0f32; w.n_out as usize]];
        self.matmul_batch(&xs, w, &mut tmp)?;
        out.copy_from_slice(&tmp[0]);
        Ok(())
    }
}

/// vk-gemv-check — VkAcc matmul vs CPU W4A8 미러 단일 텐서 검증.
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
    // CPU 참조
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
    let mut dbg = String::new();
    for i in 0..0.min(outs[0].len()) {
        dbg += &format!(" [{:.4}|{:.4}]", outs[0][i], ref_outs[0][i]);
    }
    Ok(format!(
        "vk-gemv {tname} t={t}: max|D|={mx:.3e} maxrel={rel:.2e} argmax {ia:?}=={ib:?} {}{dbg}",
        if ia == ib { "★" } else { "MISMATCH" }
    ))
}
