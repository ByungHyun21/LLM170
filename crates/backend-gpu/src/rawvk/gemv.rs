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

pub struct VkAcc {
    ctx: Mutex<VkCtx>,
    tpipes: Mutex<Option<TilePipes>>,
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
            dummy: Mutex::new(None),
            pipes: Mutex::new(None),
            wcache: Mutex::new(HashMap::new()),
            xbuf: Mutex::new(None),
            obuf: Mutex::new(None),
        })
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
            let (dsl, pl, dp, ds, pipe) = ctx.pipeline(GEMV_SPV, 10, 16)?;
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
        let mut xq_h: Vec<u32> = Vec::with_capacity(t * xq_w);
        let q8s: Vec<_> = xs
            .iter()
            .map(|r| llm170_core::quant::quantize_row_q8_ref(r))
            .collect();
        for tok in &q8s {
            for blk in tok {
                for c in 0..8 {
                    let b = c * 4;
                    xq_h.push((blk.qs[b] as u32 & 0xFF)
                        | ((blk.qs[b + 1] as u32 & 0xFF) << 8)
                        | ((blk.qs[b + 2] as u32 & 0xFF) << 16)
                        | ((blk.qs[b + 3] as u32 & 0xFF) << 24));
                }
            }
            for blk in tok {
                xq_h.push(blk.d.to_bits());
            }
            for blk in tok {
                let s0: i32 = blk.qs[..16].iter().map(|&v| v as i32).sum();
                let s1: i32 = blk.qs[16..].iter().map(|&v| v as i32).sum();
                xq_h.push(s0 as u32);
                xq_h.push(s1 as u32);
            }
        }

        let mut ctx = self.ctx.lock();
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
            let mut xb = self.xbuf.lock();
            let need = t * xq_w * 4;
            let ok = xb.as_ref().map(|b| b.bytes >= need).unwrap_or(false);
            if !ok {
                let b = ctx.alloc(need.max(1 << 20))?;
                *xb = Some(b);
            }
            let b = xb.as_ref().unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(xq_h.as_ptr() as *const u8, b.ptr, need);
            }
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
