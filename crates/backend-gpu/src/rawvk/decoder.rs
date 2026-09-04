//! VkDecoder — GDN/어텐션 GPU 상주 디코드 (plans/19 2단계).
//! 커널 8종은 gdn-check ★ 검증 완료. 기존 gemv/quant/rms/silu SPIR-V 재사용.
//! rawhip DecodeState 대칭 — 배치 모드(단일 제출+배리어).

use crate::rawvk::context::{VkBuf, VkCtx};
use ash::vk;
use std::collections::HashMap;

const GDN_CONV_SPV: &[u8] = include_bytes!("spv/gdn_conv_t.spv");
const SPLIT3_SPV: &[u8] = include_bytes!("spv/split3.spv");
const L2_SPV: &[u8] = include_bytes!("spv/l2_rows2.spv");
const BETA_G_SPV: &[u8] = include_bytes!("spv/gdn_beta_g.spv");
const GDN_AR_SPV: &[u8] = include_bytes!("spv/gdn_ar.spv");
const NORM_GATED_SPV: &[u8] = include_bytes!("spv/norm_gated.spv");
const QK_ROPE_SPV: &[u8] = include_bytes!("spv/qk_rope.spv");
const KV_APPEND_SPV: &[u8] = include_bytes!("spv/kv_append.spv");
const QSA_FLASH_SPV: &[u8] = include_bytes!("spv/qsa_flash.spv");

pub struct VkDecoder {
    pub st: std::sync::Mutex<Option<DecoderState>>,
}

struct Pipes {
    pl: vk::PipelineLayout,
    ds: vk::DescriptorSet,
    pipe: vk::Pipeline,
    dsl: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
}

pub struct DecoderState {
    ctx: VkCtx,
    w: HashMap<String, (Vec<VkBuf>, u32, usize, usize)>,
    consts: HashMap<String, VkBuf>,
    is_recr: Vec<bool>,
    // 치수
    n_layer: usize,
    n_embd: usize,
    n_ff: usize,
    dt_rank: usize,
    d_state: usize,
    d_inner: usize,
    n_group: usize,
    n_head: usize,
    n_kv: usize,
    hd: usize,
    n_rot: usize,
    conv_k: usize,
    conv_ch: usize,
    k_len: usize,
    v_len: usize,
    ctx_len: usize,
    eps: f32,
    kq_scale: f32,
    max_ssbo: usize,
    // 상태 [full|recr][seq]
    kv_k: Vec<Vec<VkBuf>>,
    kv_v: Vec<Vec<VkBuf>>,
    st_gdn: Vec<Vec<VkBuf>>,
    st_conv: Vec<Vec<VkBuf>>,
    // 공유 테이블 (gemv용)
    ktab: VkBuf,
    grid3s: VkBuf,
    dummy: VkBuf,
    // 스크래치 (t_max)
    b_xs: VkBuf,
    b_xn: VkBuf,
    b_xq_n: VkBuf,
    b_xq_f: VkBuf,
    b_xq_g: VkBuf,
    b_gqkv: VkBuf,
    b_gconv: VkBuf,
    b_gq: VkBuf,
    b_gk: VkBuf,
    b_gv: VkBuf,
    b_gb: VkBuf,
    b_ga: VkBuf,
    b_gbg: VkBuf,
    b_gz: VkBuf,
    b_go: VkBuf,
    b_ggated: VkBuf,
    b_aq: VkBuf,
    b_ak: VkBuf,
    b_av: VkBuf,
    b_aout: VkBuf,
    b_gout: VkBuf,
    b_fgate: VkBuf,
    b_fup: VkBuf,
    b_fglu: VkBuf,
    b_fdown: VkBuf,
    b_out: VkBuf, // [t][n_embd] 결과 다운로드
    b_am: VkBuf,  // argmax 8바이트
    pipes: HashMap<&'static str, Pipes>,
    split_ctr: usize,
}

unsafe impl Send for DecoderState {}
unsafe impl Sync for DecoderState {}

impl llm170_core::matmul::RawDecode for VkDecoder {
    fn raw_init(
        &self,
        hp: &llm170_core::model::hparams::Hparams,
        weights: &[(String, llm170_core::matmul::Weight<'_>)],
        consts: &[(String, Vec<f32>)],
        n_seqs: usize,
        ctx_len: usize,
        is_recr: Vec<bool>,
    ) -> Result<(), String> {
        let ctx = VkCtx::new()?;
        let wv: Vec<(String, Vec<u8>, u32, usize, usize)> = weights
            .iter()
            .map(|(k, w)| (k.clone(), w.data.to_vec(), w.ty as u32, w.n_in as usize, w.n_out as usize))
            .collect();
        let cv: Vec<(String, Vec<f32>)> = consts.to_vec();
        let ds = DecoderState::new(ctx, wv, cv, hp, is_recr, n_seqs, ctx_len)?;
        *self.st.lock().map_err(|e| e.to_string())? = Some(ds);
        Ok(())
    }

    fn raw_step(&self, seq: usize, pos: usize, emb: &[f32]) -> Result<Vec<f32>, String> {
        let mut guard = self.st.lock().map_err(|e| e.to_string())?;
        guard
            .as_mut()
            .ok_or("vkdecoder: 미초기화")?
            .step(seq, pos, emb)
    }
}

impl VkDecoder {
    pub fn new() -> Self {
        Self {
            st: std::sync::Mutex::new(None),
        }
    }
}

const T_MAX: usize = 8;

impl DecoderState {
    /// 초기화 — 가중치(carveout)+상수(GTT) 업로드, 상태 0.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mut ctx: VkCtx,
        weights: Vec<(String, Vec<u8>, u32, usize, usize)>,
        consts: Vec<(String, Vec<f32>)>,
        hp: &llm170_core::model::hparams::Hparams,
        is_recr: Vec<bool>,
        n_seqs: usize,
        ctx_len: usize,
    ) -> Result<Self, String> {
        let n = hp.n_embd;
        let (n_head, n_kv, hd, n_rot) = (hp.n_head, hp.n_kv, hp.head_dim, hp.n_rot);
        let conv_ch = hp.conv_ch();
        let conv_k = 4;
        let k_len = n_group_len(hp);
        let v_len = hp.dt_rank * hp.d_state;
        let kv_len = ctx_len * n_kv * hd;
        let gdn_len = hp.dt_rank * hp.d_state * hp.d_state;
        let conv_len = (conv_k - 1) * conv_ch;

        // 가중치 — 캐브아웃 (carveout) 업로드 후 언맵 (매핑 잔류 방지)
        let mut w = HashMap::new();
        for (name, data, ty, ni, no) in weights {
            let mut bufs = Vec::new();
            let mut off = 0usize;
            while off < data.len() {
                let sz = ctx.max_ssbo.min(data.len() - off);
                let mut b = ctx.alloc(sz)?;
                unsafe { std::ptr::copy_nonoverlapping(data.as_ptr().add(off), b.ptr, sz) };
                ctx.unmap(&mut b)?;
                bufs.push(b);
                off += sz;
            }
            w.insert(name, (bufs, ty, ni, no));
        }
        // 상수 — GTT (읽기 전용). "one"은 axpy 계수 1.0.
        let mut consts_in = consts;
        consts_in.push(("one".to_string(), vec![1.0f32]));
        let consts = consts_in;
        // 상수 — GTT (읽기 전용, 매핑 유지 무방하나 언맵)
        let mut cmap = HashMap::new();
        for (name, vals) in consts {
            let mut b = ctx.alloc_host(vals.len() * 4)?;
            unsafe { std::ptr::copy_nonoverlapping(vals.as_ptr(), b.ptr as *mut f32, vals.len()) };
            cmap.insert(name, b);
        }
        // gemv 공유 테이블
        let kv: Vec<u32> = (0..256u32)
            .map(|b| {
                let lo = llm170_core::KVALUES_IQ4NL[(b & 0xF) as usize] as u8 as u32;
                let hi = llm170_core::KVALUES_IQ4NL[(b >> 4) as usize] as u8 as u32;
                lo | (hi << 8)
            })
            .collect();
        let mut ktab = ctx.alloc(1024)?;
        unsafe { std::ptr::copy_nonoverlapping(kv.as_ptr(), ktab.ptr as *mut u32, 256) };
        ctx.unmap(&mut ktab)?;
        let mut grid3s = ctx.alloc(2048)?;
        unsafe { std::ptr::copy_nonoverlapping(llm170_core::IQ3S_GRID.as_ptr() as *const u8, grid3s.ptr, 2048) }; // iq3s 512워드 진테이블 (VkAcc ensure_shared 대칭)
        ctx.unmap(&mut grid3s)?;
        let mut dummy = ctx.alloc(16)?;
        let z16 = [0u8; 16];
        unsafe { std::ptr::copy_nonoverlapping(z16.as_ptr(), dummy.ptr, 16) };

        let n_full = is_recr.iter().filter(|&&r| !r).count();
        let n_recr = is_recr.len() - n_full;
        let zeros_kv = vec![0u8; kv_len * 4];
        let mut kv_k = Vec::with_capacity(n_full);
        let mut kv_v = Vec::with_capacity(n_full);
        for _ in 0..n_full {
            let mut ck = Vec::with_capacity(n_seqs);
            let mut cv = Vec::with_capacity(n_seqs);
            for _ in 0..n_seqs {
                let k = ctx.alloc(kv_len * 4)?;
                unsafe { std::ptr::copy_nonoverlapping(zeros_kv.as_ptr(), k.ptr, zeros_kv.len()) };
                let v = ctx.alloc(kv_len * 4)?;
                unsafe { std::ptr::copy_nonoverlapping(zeros_kv.as_ptr(), v.ptr, zeros_kv.len()) };
                ck.push(k);
                cv.push(v);
            }
            kv_k.push(ck);
            kv_v.push(cv);
        }
        let zeros_g = vec![0u8; gdn_len * 4];
        let zeros_c = vec![0u8; conv_len * 4];
        let mut st_gdn = Vec::with_capacity(n_recr);
        let mut st_conv = Vec::with_capacity(n_recr);
        for _ in 0..n_recr {
            let mut gd = Vec::with_capacity(n_seqs);
            let mut cv = Vec::with_capacity(n_seqs);
            for _ in 0..n_seqs {
                let g = ctx.alloc(gdn_len * 4)?;
                unsafe { std::ptr::copy_nonoverlapping(zeros_g.as_ptr(), g.ptr, zeros_g.len()) };
                let c = ctx.alloc(conv_len * 4)?;
                unsafe { std::ptr::copy_nonoverlapping(zeros_c.as_ptr(), c.ptr, zeros_c.len()) };
                gd.push(g);
                cv.push(c);
            }
            st_gdn.push(gd);
            st_conv.push(cv);
        }

        let xq_sn = n / 4 + n / 32 + n / 16;
        let xq_sf = hp.n_ff / 4 + hp.n_ff / 32 + hp.n_ff / 16;
        let xq_sg = hp.d_inner / 4 + hp.d_inner / 32 + hp.d_inner / 16;
        let max_ssbo0 = ctx.max_ssbo;
        let (b_xs, b_xn, b_xq_n, b_xq_f, b_xq_g, b_gqkv, b_gconv, b_gq, b_gk, b_gv,
             b_gb, b_ga, b_gbg, b_gz, b_go, b_ggated, b_aq, b_ak, b_av, b_aout,
             b_gout, b_fgate, b_fup, b_fglu, b_fdown, b_out, b_am) = {
            let mut a = |sz: usize| -> Result<VkBuf, String> {
                ctx.alloc_host(sz.max(1) * 4).map_err(|e| e.to_string())
            };
            (
                a(T_MAX * n)?, a(T_MAX * n)?, a(T_MAX * xq_sn)?, a(T_MAX * xq_sf)?,
                a(T_MAX * xq_sg)?, a(T_MAX * conv_ch)?, a(T_MAX * conv_ch)?,
                a(T_MAX * k_len)?, a(T_MAX * k_len)?, a(T_MAX * v_len)?,
                a(T_MAX * hp.dt_rank)?, a(T_MAX * hp.dt_rank)?,
                a(T_MAX * hp.dt_rank * 2)?, a(T_MAX * hp.d_inner)?, a(T_MAX * v_len)?,
                a(T_MAX * hp.d_inner)?, a(T_MAX * n_head * 2 * hd)?,
                a(T_MAX * n_kv * hd)?, a(T_MAX * n_kv * hd)?, a(T_MAX * n_head * hd)?,
                a(T_MAX * n)?, a(T_MAX * hp.n_ff)?, a(T_MAX * hp.n_ff)?,
                a(T_MAX * hp.n_ff)?, a(T_MAX * n)?, a(T_MAX * n)?, a(8)?,
            )
        };
        Ok(Self {
            ctx,
            max_ssbo: max_ssbo0,
            w,
            consts: cmap,
            is_recr,
            n_layer: hp.n_layer,
            n_embd: n,
            n_ff: hp.n_ff,
            dt_rank: hp.dt_rank,
            d_state: hp.d_state,
            d_inner: hp.d_inner,
            n_group: hp.n_group,
            n_head,
            n_kv,
            hd,
            n_rot,
            conv_k,
            conv_ch,
            k_len,
            v_len,
            ctx_len,
            eps: hp.eps,
            kq_scale: 1.0 / (hd as f32).sqrt(),
            kv_k,
            kv_v,
            st_gdn,
            st_conv,
            ktab,
            grid3s,
            dummy,
            b_xs,
            b_xn,
            b_xq_n,
            b_xq_f,
            b_xq_g,
            b_gqkv,
            b_gconv,
            b_gq,
            b_gk,
            b_gv,
            b_gb,
            b_ga,
            b_gbg,
            b_gz,
            b_go,
            b_ggated,
            b_aq,
            b_ak,
            b_av,
            b_aout,
            b_gout,
            b_fgate,
            b_fup,
            b_fglu,
            b_fdown,
            b_out,
            b_am,
            pipes: HashMap::new(),
            split_ctr: 0,
        })
    }

    fn k_group_len(&self) -> usize {
        self.k_len
    }

    /// 파이프라인 지연 생성 캐시.
    fn pipe(&mut self, name: &'static str, spv: &[u8], n_buf: u32, pb: u32) -> Result<&Pipes, String> {
        if !self.pipes.contains_key(name) {
            let (dsl, pl, pool, ds, pipe) = self.ctx.pipeline(spv, n_buf, pb)?;
            self.pipes.insert(name, Pipes { pl, ds, pipe, dsl, pool });
        }
        Ok(self.pipes.get(name).unwrap())
    }

    /// 바인딩+런치 (배치 모드 자동 — fresh ds).
    fn run_pipe(&mut self, name: &'static str, spv: &[u8], n_buf: u32, pb: u32, bufs: &[vk::Buffer], push: &[u8], gx: u32, gy: u32, gz: u32) -> Result<(), String> {
        // 배치 자동 분할 — 디스패치 512마다 제출·대기·재시작 (세트/CMDBUF 누적 방지).
        if self.ctx.batching.load(std::sync::atomic::Ordering::Relaxed) {
            self.split_ctr += 1;
            if self.split_ctr >= 512 {
                self.split_ctr = 0;
                self.ctx.end_batch_wait()?;
                self.ctx.begin_batch()?;
            }
        }
        let (pl, ds_default, pipe, dsl, pool) = {
            let p = self.pipe(name, spv, n_buf, pb)?;
            (p.pl, p.ds, p.pipe, p.dsl, p.pool)
        };
        let ds = if self.ctx.batching.load(std::sync::atomic::Ordering::Relaxed) {
            self.ctx.batch_dsl.set(Some((dsl, pool)));
            let ds = self.ctx.fresh_ds(n_buf)?;
            self.ctx.bind_bufs(ds, bufs);
            ds
        } else {
            self.ctx.bind_bufs(ds_default, bufs);
            ds_default
        };
        self.ctx.run(pl, ds, pipe, push, gx, gy, gz)
    }

    fn push_u32s(vals: &[u32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// quant: [t][n] f32 → xq (q8 레이아웃).
    fn quant(&mut self, src: vk::Buffer, xq: vk::Buffer, n: usize, t: usize) -> Result<(), String> {
        let xq_w = n / 4 + n / 32 + n / 16;
        let push = Self::push_u32s(&[n as u32, t as u32, xq_w as u32]);
        self.run_pipe("quant", crate::rawvk::gemv::QUANT_SPV, 2, 12,
            &[src, xq], &push, (n / 32 + 63) as u32 / 64, t as u32, 1)
    }

    /// GEMV (12바인딩 gemv3): xq × 가중 → out.
    fn gemv(&mut self, xq: vk::Buffer, wkey: &str, out: vk::Buffer, t: usize) -> Result<(), String> {
        let (wbufs, ty, ni, no) = self.w.get(wkey).cloned().ok_or(format!("가중치 없음: {wkey}"))?;
        let xq_w = ni / 4 + ni / 32 + ni / 16;
        let mut binds: Vec<vk::Buffer> = wbufs.iter().map(|b| b.buf).collect();
        while binds.len() < 8 {
            binds.push(self.dummy.buf);
        }
        binds.push(xq);
        binds.push(out);
        binds.push(self.ktab.buf);
        binds.push(self.grid3s.buf);
        let push = Self::push_u32s(&[ni as u32, no as u32, xq_w as u32, ty, t as u32]);
        self.run_pipe("gemv", crate::rawvk::gemv::GEMV_SPV, 12, 20, &binds, &push, no as u32, 1, 1)
    }

    /// rms_norm (t행) — 상수 가중치 (consts).
    fn rms(&mut self, src: vk::Buffer, wkey: &str, out: vk::Buffer, n: usize, t: usize) -> Result<(), String> {
        let wbuf = self.consts.get(wkey).cloned().ok_or(format!("상수 없음: {wkey}"))?;
        let eps = self.eps;
        let mut push = Self::push_u32s(&[n as u32, t as u32]);
        push.extend_from_slice(&eps.to_le_bytes());
        self.run_pipe("rms", crate::rawvk::gemv::RMS_SPV, 3, 12,
            &[src, wbuf.buf, out], &push, t as u32, 1, 1)
    }

    /// t=1 단일 스텝 — 배치 모드로 전 층 단일 제출·다운로드 1회.
    pub fn step(&mut self, seq: usize, pos: usize, emb: &[f32]) -> Result<Vec<f32>, String> {
        let noba = std::env::var_os("LLM170_VK_NOBATCH").is_some();
        let n = self.n_embd;
        debug_assert_eq!(emb.len(), n);
        let (dt_rank, d_state, d_inner) = (self.dt_rank, self.d_state, self.d_inner);
        let (n_head, n_kv, hd, n_rot) = (self.n_head, self.n_kv, self.hd, self.n_rot);
        let conv_ch = self.conv_ch;
        let k_len = self.k_len;
        let v_len = self.v_len;
        unsafe { std::ptr::copy_nonoverlapping(emb.as_ptr(), self.b_xs.ptr as *mut f32, n) };
        if !noba { self.ctx.begin_batch()?; };
        let mut recr_idx = 0usize;
        let mut full_idx = 0usize;
        let layer_cut = std::env::var("LLM170_VK_LAYERS").ok().and_then(|v| v.parse::<usize>().ok());
        for il in 0..self.n_layer {
            if layer_cut.is_some_and(|c| il >= c) {
                break;
            }
            // ── attn_norm + quant
            {
                let (xs, xn) = (self.b_xs.clone(), self.b_xn.clone());
                self.rms(xs.buf, &format!("blk.{il}.attn_norm"), xn.buf, n, 1)?;
            }
            self.quant(self.b_xn.buf, self.b_xq_n.buf, n, 1)?;
            if self.is_recr[il] {
                // GDN 4 GEMV
                self.gemv(self.b_xq_n.buf, &format!("blk.{il}.attn_qkv.weight"), self.b_gqkv.buf, 1)?;
                self.gemv(self.b_xq_n.buf, &format!("blk.{il}.attn_gate.weight"), self.b_gz.buf, 1)?;
                self.gemv(self.b_xq_n.buf, &format!("blk.{il}.ssm_beta.weight"), self.b_gb.buf, 1)?;
                self.gemv(self.b_xq_n.buf, &format!("blk.{il}.ssm_alpha.weight"), self.b_ga.buf, 1)?;
                let gskip = std::env::var("LLM170_VK_GDN_SKIP").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(0);
                // conv (t=1 — ring)
                {
                    let cw = self.consts.get(&format!("blk.{il}.conv_w")).cloned().ok_or("conv_w")?;
                    // PC: {int ch; int k; int t} + local 64 — gdn-check 준거
                    let push = Self::push_u32s(&[conv_ch as u32, self.conv_k as u32, 1u32]);
                    self.run_pipe("gdn_conv", GDN_CONV_SPV, 4, 12,
                        &[self.b_gqkv.buf, cw.buf, self.st_conv[recr_idx][seq].buf, self.b_gconv.buf],
                        &push, conv_ch.div_ceil(64) as u32, 1, 1)?;
                }
                // split3
                {
                    let total = 2 * k_len + v_len;
                    let push = Self::push_u32s(&[k_len as u32, k_len as u32, v_len as u32]);
                    self.run_pipe("split3", SPLIT3_SPV, 4, 12,
                        &[self.b_gconv.buf, self.b_gq.buf, self.b_gk.buf, self.b_gv.buf],
                        &push, total.div_ceil(64) as u32, 1, 1)?;
                }
                // l2 — PC: {float eps; int d; int ng} (스케일은 AR이 적용 — rawhip l2_rows2_scale 준거)
                {
                    let mut push = self.eps.to_le_bytes().to_vec();
                    push.extend(Self::push_u32s(&[d_state as u32, self.n_group as u32]));
                    self.run_pipe("l2", L2_SPV, 2, 12,
                        &[self.b_gq.buf, self.b_gk.buf], &push, (2 * self.n_group) as u32, 1, 1)?;
                }
                // beta_g
                {
                    let dtb = self.consts.get(&format!("blk.{il}.dt_bias")).cloned().ok_or("dtb")?;
                    let ssa = self.consts.get(&format!("blk.{il}.ssm_a")).cloned().ok_or("ssa")?;
                    let push = Self::push_u32s(&[dt_rank as u32, dt_rank as u32]);
                    self.run_pipe("beta_g", BETA_G_SPV, 5, 8,
                        &[self.b_gb.buf, self.b_ga.buf, dtb.buf, ssa.buf, self.b_gbg.buf],
                        &push, dt_rank.div_ceil(64) as u32, 1, 1)?;
                }
                // AR (LLM170_VK_GDN_SKIP=1이면 스킵 — L3 크래시 분리용)
                if gskip & 1 == 0 {
                {
                    let scale = 1.0f32 / (d_state as f32).sqrt();
                    let mut push = Self::push_u32s(&[d_state as u32, k_len as u32, v_len as u32, dt_rank as u32, self.n_group as u32]);
                    push.extend_from_slice(&scale.to_le_bytes());
                    push.extend_from_slice(&1u32.to_le_bytes());
                    self.run_pipe("gdn_ar", GDN_AR_SPV, 6, 28,
                        &[self.st_gdn[recr_idx][seq].buf, self.b_gq.buf, self.b_gk.buf,
                          self.b_gv.buf, self.b_gbg.buf, self.b_go.buf],
                        &push, dt_rank as u32, d_state as u32, 1)?;
                }
                }
                // norm_gated (비트 2)
                if gskip & 2 == 0 {
                {
                    let sn = self.consts.get(&format!("blk.{il}.ssm_norm")).cloned().ok_or("sn")?;
                    // PC: {float eps; int d=d_state; int n_h=dt_rank} — rawhip norm_gated_silu 준거
                    let mut push = self.eps.to_le_bytes().to_vec();
                    push.extend(Self::push_u32s(&[d_state as u32, dt_rank as u32]));
                    self.run_pipe("norm_gated", NORM_GATED_SPV, 4, 12,
                        &[self.b_go.buf, self.b_gz.buf, sn.buf, self.b_ggated.buf],
                        &push, dt_rank as u32, 1, 1)?;
                }
                let xq_sg = d_inner / 4 + d_inner / 32 + d_inner / 16;
                let _ = xq_sg;
                }
                self.quant(self.b_ggated.buf, self.b_xq_g.buf, d_inner, 1)?;
                if gskip & 4 == 0 {
                self.gemv(self.b_xq_g.buf, &format!("blk.{il}.ssm_out.weight"), self.b_gout.buf, 1)?;
                }
                recr_idx += 1;
            } else {
                // 어텐션 (LLM170_VK_ATTN: 1=qkv gemv만, 2=+rope/kv, 3=+flash, 4=+wo)
                let attn_cut = std::env::var("LLM170_VK_ATTN").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(4);
                if std::env::var_os("LLM170_VKD_TRACE").is_some() && il == 3 {
                    let (bufs, tyq, niq, noq) = self.w.get(&format!("blk.{il}.attn_q.weight")).unwrap();
                    let total: usize = bufs.iter().map(|b| b.bytes).sum();
                    let (gbufs, _, _, gno) = self.w.get("blk.0.attn_qkv.weight").unwrap();
                    let gtotal: usize = gbufs.iter().map(|b| b.bytes).sum();
                    eprintln!("#  ATTN3 ty={} ni={} no={} chunks={} bytes={} max_ssbo={} | L0qkv no={} chunks={} bytes={}", tyq, niq, noq, bufs.len(), total, self.max_ssbo, gno, gbufs.len(), gtotal);
                }
                self.gemv(self.b_xq_n.buf, &format!("blk.{il}.attn_q.weight"), self.b_aq.buf, 1)?;
                self.gemv(self.b_xq_n.buf, &format!("blk.{il}.attn_k.weight"), self.b_ak.buf, 1)?;
                self.gemv(self.b_xq_n.buf, &format!("blk.{il}.attn_v.weight"), self.b_av.buf, 1)?;
                // qk_rope
                {
                    let qn = self.consts.get(&format!("blk.{il}.attn_q_norm")).cloned().ok_or("qn")?;
                    let kn = self.consts.get(&format!("blk.{il}.attn_k_norm")).cloned().ok_or("kn")?;
                    let cs = self.consts.get("cs").cloned().ok_or("cs")?;
                    // PC: {float eps; float kqs; int pos; int nh; int nk; int hd; int nr} — gdn-check 준거
                    let mut push = self.eps.to_le_bytes().to_vec();
                    push.extend_from_slice(&self.kq_scale.to_le_bytes());
                    push.extend(Self::push_u32s(&[pos as u32, n_head as u32, n_kv as u32, hd as u32, n_rot as u32]));
                    self.run_pipe("qk_rope", QK_ROPE_SPV, 5, 28,
                        &[self.b_aq.buf, self.b_ak.buf, qn.buf, kn.buf, cs.buf],
                        &push, (n_head + n_kv) as u32, 1, 1)?;
                }
                if attn_cut >= 2 {
                // kv append
                {
                    let push = Self::push_u32s(&[(n_kv * hd) as u32, pos as u32]);
                    self.run_pipe("kv_app", KV_APPEND_SPV, 2, 8,
                        &[self.b_ak.buf, self.kv_k[full_idx][seq].buf], &push,
                        (n_kv * hd).div_ceil(64) as u32, 1, 1)?;
                    self.run_pipe("kv_app", KV_APPEND_SPV, 2, 8,
                        &[self.b_av.buf, self.kv_v[full_idx][seq].buf], &push,
                        (n_kv * hd).div_ceil(64) as u32, 1, 1)?;
                }
                if attn_cut >= 3 {
                // flash
                {
                    let mask = self.consts.get("mask").cloned().ok_or("mask")?;
                    let push = Self::push_u32s(&[(pos + 1) as u32, n_head as u32, n_kv as u32, hd as u32, self.ctx_len as u32, pos as u32]);
                    self.run_pipe("qsa_flash", QSA_FLASH_SPV, 5, 24,
                        &[self.b_aq.buf, self.kv_k[full_idx][seq].buf, self.kv_v[full_idx][seq].buf, mask.buf, self.b_aout.buf],
                        &push, 1, n_head as u32, 1)?;
                }
                self.quant(self.b_aout.buf, self.b_xq_g.buf, n_head * hd, 1)?;
                }
                if attn_cut >= 4 {
                self.gemv(self.b_xq_g.buf, &format!("blk.{il}.attn_output.weight"), self.b_gout.buf, 1)?;
                }
                }
                full_idx += 1;
            }
            // 잔차
            self.axpy(self.b_xs.buf, self.b_gout.buf, n)?;
            // ── FFN
            {
                let (xs, xn) = (self.b_xs.clone(), self.b_xn.clone());
                self.rms(xs.buf, &format!("blk.{il}.post_norm"), xn.buf, n, 1)?;
            }
            self.quant(self.b_xn.buf, self.b_xq_n.buf, n, 1)?;
            self.gemv(self.b_xq_n.buf, &format!("blk.{il}.ffn_gate.weight"), self.b_fgate.buf, 1)?;
            self.gemv(self.b_xq_n.buf, &format!("blk.{il}.ffn_up.weight"), self.b_fup.buf, 1)?;
            self.silu_mul(self.b_fgate.buf, self.b_fup.buf, self.b_fglu.buf, self.n_ff)?;
            self.quant(self.b_fglu.buf, self.b_xq_f.buf, self.n_ff, 1)?;
            self.gemv(self.b_xq_f.buf, &format!("blk.{il}.ffn_down.weight"), self.b_fdown.buf, 1)?;
            self.axpy(self.b_xs.buf, self.b_fdown.buf, n)?;
            // 실험: L0 FFN 직후 attn_q gemv 강제 (층 위치 vs 가중치 분리)
            if std::env::var_os("LLM170_VK_FORCE_AQ").is_some() && il == 0 {
                self.gemv(self.b_xq_n.buf, "blk.3.attn_q.weight", self.b_aq.buf, 1)?;
            }
        }
        if !noba { self.ctx.end_batch_wait()?; } else { self.ctx.flush2()?; };
        // ── head: output_norm → quant → gemv(output) → 다운로드
        {
            let (xs, xn) = (self.b_xs.clone(), self.b_xn.clone());
            self.rms(xs.buf, "output_norm", xn.buf, n, 1)?;
        }
        self.quant(self.b_xn.buf, self.b_xq_n.buf, n, 1)?;
        if !noba { self.ctx.begin_batch()?; };
        self.gemv(self.b_xq_n.buf, "output.weight", self.b_gout.buf, 1)?;
        if !noba { self.ctx.end_batch_wait()?; } else { self.ctx.flush2()?; };
        let no_out = self.w.get("output.weight").map(|(_, _, _, no)| *no).unwrap_or(n);
        let mut logits = vec![0f32; no_out];
        unsafe { std::ptr::copy_nonoverlapping(self.b_gout.ptr as *const f32, logits.as_mut_ptr(), no_out.min(self.b_gout.bytes / 4)) };
        Ok(logits)
    }

    /// axpy: y += x·s[0] (s=one 버퍼).
    fn axpy(&mut self, y: vk::Buffer, x: vk::Buffer, n: usize) -> Result<(), String> {
        // one 버퍼 필요 — dummy는 0이므로 별도 1.0 버퍼 (init에서 만들었으면 재사용)
        let one = self.consts.get("one").ok_or("one 버퍼 없음")?;
        let push = (n as u32).to_le_bytes().to_vec();
        self.run_pipe("axpy", crate::rawvk::AXPY_SPV, 3, 4,
            &[y, x, one.buf], &push, n.div_ceil(256) as u32, 1, 1)
    }

    /// silu_mul g·u.
    fn silu_mul(&mut self, g: vk::Buffer, u: vk::Buffer, o: vk::Buffer, total: usize) -> Result<(), String> {
        let push = (total as u32).to_le_bytes().to_vec();
        self.run_pipe("silu", crate::rawvk::gemv::SILU_SPV, 3, 4,
            &[g, u, o], &push, total.div_ceil(256) as u32, 1, 1)
    }
}

fn n_group_len(hp: &llm170_core::model::hparams::Hparams) -> usize {
    hp.n_group * hp.d_state
}
