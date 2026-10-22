use crate::rawcuda::ctx::CudaCtx;
use crate::rawcuda::ffi::{self, CUdeviceptr};
use std::collections::HashMap;
use unified_cache::{CacheCategory, Tier, UnifiedCacheRegistry, base_util};

pub const SPEC_VERIFY_TMAX: usize = 9;

pub const CTX_OVERFLOW: &str = "context overflow";

pub const ATTN_F3S_TMAX: usize = 512;
pub const CHAIN_TMAX: usize = 512;
pub const GEMM_MMA_M: usize = 64;
pub const GEMM_MMA_N: usize = 128;
pub const GEMM_BMMA_M: usize = 32;
pub const GEMM_BMMA_N: usize = 64;
pub const GEMM_GRP_M: usize = 64;
pub const GEMM_GRP_N: usize = 32;
pub const MOE_PP_MZ: usize = 4;
pub const GDN_H_SMEM: u32 = 41_220;
pub const GDN_O_SMEM: u32 = 59_652;
pub const GDN_NGRP_O: usize = 8;
pub const GDN_WU_SMEM: u32 = 35_072;
pub const GDN_NSPLIT: usize = 4;
pub const GDN_NGRP: usize = 8;
pub const GDN_CS: usize = 32;
pub const BATCH_DEC_MAX: usize = 8;
pub const G4_CHUNK_MAX: usize = 512;
pub const G4_TROW_MAX: usize = 16;
pub const KSTEP_KMAX: usize = 8;
pub const GEMV_TR: usize = 8;
pub const GEMM_FFMA_TMAX: usize = 32;

#[derive(Clone, Copy, Default)]
pub struct Buf {
    pub ptr: CUdeviceptr,
    pub cap: usize,
}

#[derive(Clone, Copy, Default)]
pub struct NormBufs {
    pub dx: CUdeviceptr,
    pub dab: CUdeviceptr,
    pub dxn: CUdeviceptr,
    pub cap: usize,
}

#[derive(Clone, Copy, Default)]
pub struct EwBufs {
    pub dewg: CUdeviceptr,
    pub dewu: CUdeviceptr,
    pub dew: CUdeviceptr,
    pub cap: usize,
}

#[derive(Clone, Copy, Default)]
pub struct ExpBufs {
    pub gate: CUdeviceptr,
    pub up: CUdeviceptr,
    pub act: CUdeviceptr,
    pub dn: CUdeviceptr,
    pub cap: usize,
}

#[derive(Clone, Copy, Default)]
pub struct StgBufs {
    pub q: CUdeviceptr,
    pub s: CUdeviceptr,
    pub cap: (usize, usize),
}

#[derive(Clone, Copy, Default)]
pub(super) struct SkGroup {
    pub off: u32,
    pub cnt: u32,
    pub blocks: u32,
    pub ksp: u8,
}

#[derive(Clone, Copy)]
pub enum LinPath {
    Gemv1 { w: usize },
    Gemm,
    GemvT,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GdnDims {
    pub n_gdn: usize,
    pub hidden: usize,
    pub h_k: usize,
    pub h_v: usize,
    pub d: usize,
}

impl GdnDims {
    pub fn conv_ch(&self) -> usize {
        (2 * self.h_k + self.h_v) * self.d
    }
    pub fn k_len(&self) -> usize {
        self.h_k * self.d
    }
    pub fn v_len(&self) -> usize {
        self.h_v * self.d
    }
    pub fn bg_len(&self) -> usize {
        2 * self.h_v
    }
}

pub const ATTN_SPLITS: usize = 8;
pub const ATTN_SPLITS_D: usize = 64;
pub const ATTN_SPLITS_MAX: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttnLayer {
    pub kv_heads: usize,
    pub d: usize,
    pub sliding: bool,
    pub k_eq_v: bool,
    pub kq_scale: f32,
}

impl AttnLayer {
    pub fn scale(&self) -> f32 {
        if self.kq_scale > 0.0 {
            self.kq_scale
        } else {
            1.0 / (self.d as f32).sqrt()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttnDims {
    pub n_attn: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub d: usize,
    pub cap: usize,
    pub interval: usize,
    pub sliding_window: usize,
}

impl AttnDims {
    pub fn q_dim(&self) -> usize {
        self.q_heads * self.d
    }
    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.d
    }
    pub fn qg_dim(&self) -> usize {
        self.q_heads * 2 * self.d
    }
    pub fn kv_slot_elems(&self, slot: usize) -> usize {
        slot * self.n_attn * self.cap * self.kv_dim()
    }
    pub fn layer(&self, _il: usize) -> AttnLayer {
        AttnLayer {
            kv_heads: self.kv_heads,
            d: self.d,
            sliding: false,
            k_eq_v: false,
            kq_scale: 0.0,
        }
    }
}

pub fn moe_topk(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let n = logits.len();
    let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p = vec![0.0f32; n];
    let mut sum = 0.0f32;
    for (i, l) in logits.iter().enumerate() {
        p[i] = (l - mx).exp();
        sum += p[i];
    }
    for v in p.iter_mut() {
        *v /= sum;
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_unstable_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
    idx.truncate(k);
    let wsum: f32 = idx.iter().map(|&e| p[e]).sum();
    idx.iter().map(|&e| (e, p[e] / wsum)).collect()
}

pub fn f32_to_f16(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32;
    let man = b & 0x7F_FFFF;
    if exp == 0xFF {
        return sign | 0x7C00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7C00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let mut sub = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        if rem > half || (rem == half && (sub & 1) == 1) {
            sub += 1;
        }
        return sign | (sub as u16);
    }
    let mut h = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | (h as u16)
}

pub(super) struct GraphEntry {
    exec: ffi::CUgraphExec,
    handle: ffi::CUgraph,
    slot: usize,
    head: bool,
    argmax: bool,
}

pub(super) struct MoE {
    pub(super) n_experts: usize,
    pub(super) top_k: usize,
    pub(super) moe_ffn: usize,
    pub(super) shared_ffn: usize,
    pub(super) glu_gelu: bool,
    pub(super) moe_group: usize,
    pub(super) moe_scale_bf16: bool,
    pub(super) moe_tab: Vec<(u64, u64, u64, u64)>,
    pub(super) moe_resident: bool,
    pub(super) hot_per_layer: usize,
    pub(super) hot_tab: Vec<(u64, u64, u64, u64)>,
    pub(super) moe_dev_tab: CUdeviceptr,
    pub(super) moe_idx: CUdeviceptr,
    pub(super) moe_wt: CUdeviceptr,
    pub(super) moe_multi_tab: CUdeviceptr,
    pub(super) moe_multi_rows: Vec<u32>,
    pub(super) moe_multi_rows_t3: Vec<u32>,
    pub(super) moe_gslot: CUdeviceptr,
    pub(super) moe_gcnt: CUdeviceptr,
    pub(super) moe_goff: CUdeviceptr,
    pub(super) moe_gmax: usize,
    pub(super) pin_moe_idx: *mut std::ffi::c_void,
    pub(super) pin_moe_idx_cap: usize,
    pub(super) exp: ExpBufs,
    pub(super) stg: StgBufs,
    pub(super) host_stage: Vec<u8>,
    pub(super) host_stage_reg: usize,
    pub(super) drt: Buf,
    pub(super) dmo: Buf,
    pub(super) moe_bufs_ok: bool,
    pub(super) dsg: CUdeviceptr,
    pub(super) ev_fork: Vec<ffi::CUevent>,
    pub(super) dsh: [CUdeviceptr; 2],
    pub(super) dsh_cap: usize,
    pub(super) mcache_q: CUdeviceptr,
    pub(super) mcache_s: CUdeviceptr,
    pub(super) mcache_k: usize,
    pub(super) mcache_k_plan: usize,
    pub(super) mcache_ids: Vec<u16>,
    pub(super) mcache_age: Vec<u32>,
    pub(super) mcache_tick: u32,
    pub(super) mcache_hits: u64,
    pub(super) mcache_miss: u64,
}

pub(super) struct Mtp {
    pub(super) mtp_ok: bool,
    pub(super) mtp_nw: CUdeviceptr,
    pub(super) mtp_zero: CUdeviceptr,
    pub(super) mtp_buf: CUdeviceptr,
    pub(super) mtp_off: [usize; 17],
    pub(super) mtp_prev_off: usize,
    pub(super) mtp_layer: usize,
    pub(super) mtp_rows: usize,
    pub(super) mtp_ffn: usize,
    pub(super) mtp_draft: Vec<Vec<u32>>,
    pub(super) mtp_pos: Vec<u32>,
    pub(super) mtp_prev_pos: Vec<u32>,
    pub(super) mtp_off_slot: Vec<bool>,
    pub(super) mtp_graphs: Vec<Option<(ffi::CUgraphExec, ffi::CUgraph)>>,
    pub(super) mtp_graph_failed: bool,
    pub(super) mtp_pin_emb: *mut std::ffi::c_void,
    pub(super) mtp_pin_out: *mut std::ffi::c_void,
    pub(super) mtp_chain_ctr: CUdeviceptr,
    pub(super) mtp_chain_log: CUdeviceptr,
    pub(super) mtp_chain_graphs: Vec<Option<(ffi::CUgraphExec, ffi::CUgraph)>>,
}

pub(super) struct Attn {
    pub(super) layers: Vec<AttnLayer>,
    pub(super) attn: Option<AttnDims>,
    pub(super) dqnw_a: CUdeviceptr,
    pub(super) dknw_a: CUdeviceptr,
    pub(super) g4_rp: CUdeviceptr,
    pub(super) dkc: CUdeviceptr,
    pub(super) dvc: CUdeviceptr,
    pub(super) dksc: CUdeviceptr,
    pub(super) dvsc: CUdeviceptr,
    pub(super) dblk_tab: CUdeviceptr,
    pub(super) kv_pool: Option<blocktable::KvPool>,
    pub(super) kv_budget: Option<u64>,
    pub(super) kv_vram_cap: Option<u64>,
    pub(super) dkc_h: CUdeviceptr,
    pub(super) dvc_h: CUdeviceptr,
    pub(super) dksc_h: CUdeviceptr,
    pub(super) dvsc_h: CUdeviceptr,
    pub(super) hkc_h: *mut std::ffi::c_void,
    pub(super) hvc_h: *mut std::ffi::c_void,
    pub(super) hksc_h: *mut std::ffi::c_void,
    pub(super) hvsc_h: *mut std::ffi::c_void,
    pub(super) kv_vram_blocks: usize,
    pub(super) kv_host_bytes: u64,
    pub(super) kv_arena_bytes: u64,
    pub(super) dpp: CUdeviceptr,
    pub(super) dqg_a: CUdeviceptr,
    pub(super) dkin_a: CUdeviceptr,
    pub(super) dvin_a: CUdeviceptr,
    pub(super) dqh_a: CUdeviceptr,
    pub(super) dattn_part: CUdeviceptr,
    pub(super) doutv_a: CUdeviceptr,
    pub(super) attn_t_cap: usize,
}

pub(super) struct Gdn {
    pub(super) gdn: Option<GdnDims>,
    pub(super) dcw: CUdeviceptr,
    pub(super) dab_c: CUdeviceptr,
    pub(super) dalog: CUdeviceptr,
    pub(super) ddtb: CUdeviceptr,
    pub(super) dnwg: CUdeviceptr,
    pub(super) dring: CUdeviceptr,
    pub(super) dgst: CUdeviceptr,
    pub(super) dsc: CUdeviceptr,
    pub(super) dsc_ones: CUdeviceptr,
    pub(super) dqkv: CUdeviceptr,
    pub(super) dzv: CUdeviceptr,
    pub(super) dgxn: CUdeviceptr,
    pub(super) dgq: CUdeviceptr,
    pub(super) dgk: CUdeviceptr,
    pub(super) dgv: CUdeviceptr,
    pub(super) dq2: CUdeviceptr,
    pub(super) dk2: CUdeviceptr,
    pub(super) dv2: CUdeviceptr,
    pub(super) dbg: CUdeviceptr,
    pub(super) dgo: CUdeviceptr,
    pub(super) dgpart: CUdeviceptr,
    pub(super) dgdc: CUdeviceptr,
    pub(super) dakq: CUdeviceptr,
    pub(super) dhbuf: CUdeviceptr,
    pub(super) dwu: CUdeviceptr,
    pub(super) ddcbuf: CUdeviceptr,
    pub(super) dgate: CUdeviceptr,
    pub(super) gdn_t_cap: usize,
    pub(super) dwy_s0: CUdeviceptr,
    pub(super) dwy_dc: CUdeviceptr,
    pub(super) dwy_g: CUdeviceptr,
    pub(super) dwy_k: CUdeviceptr,
    pub(super) dsnap_ring: CUdeviceptr,
    pub(super) dring0: CUdeviceptr,
    pub(super) dsave: CUdeviceptr,
    pub(super) dchk: *mut std::ffi::c_void,
    pub(super) dchk_pos: Vec<[u32; 4]>,
    pub(super) dchk_saved: Vec<[u32; 4]>,
    pub(super) dchk_use: Vec<[u64; 4]>,
    pub(super) dchk_tick: u64,
    #[allow(dead_code)]
    pub(super) sig_gate: bool,
}

#[allow(dead_code)]
pub(super) struct Fn {
    pub(super) on: bool,
    pub(super) hc: usize,
    pub(super) lowrank: usize,
    pub(super) eps: f32,
    pub(super) d_hcn: CUdeviceptr,
    pub(super) ple_il: Option<usize>,
    pub(super) d_ple_w: CUdeviceptr,
    pub(super) d_ple_nk: CUdeviceptr,
    pub(super) d_ple_nq: CUdeviceptr,
    pub(super) d_ple_nc: CUdeviceptr,
    pub(super) d_ple_st: CUdeviceptr,
    pub(super) ple_init: Vec<bool>,
    pub(super) r: Buf,
    pub(super) xn: Buf,
    pub(super) g: Buf,
    pub(super) x: Buf,
    pub(super) lora: Buf,
    pub(super) y: Buf,
    pub(super) inj: Buf,
    pub(super) ple_e: Buf,
    pub(super) ple_k: Buf,
    pub(super) ple_v: Buf,
    pub(super) ple_gd: Buf,
    pub(super) ple_nm: Buf,
    pub(super) sel: Buf,
    pub(super) nsel: Buf,
    pub(super) idx_heads: usize,
    pub(super) wave: usize,
    pub(super) warm_t: usize,
}

impl Default for Fn {
    fn default() -> Self {
        Self {
            on: false,
            hc: 0,
            lowrank: 0,
            eps: 1e-6,
            d_hcn: 0,
            ple_il: None,
            d_ple_w: 0,
            d_ple_nk: 0,
            d_ple_nq: 0,
            d_ple_nc: 0,
            d_ple_st: 0,
            ple_init: Vec::new(),
            r: Buf::default(),
            xn: Buf::default(),
            g: Buf::default(),
            x: Buf::default(),
            lora: Buf::default(),
            y: Buf::default(),
            inj: Buf::default(),
            ple_e: Buf::default(),
            ple_k: Buf::default(),
            ple_v: Buf::default(),
            ple_gd: Buf::default(),
            ple_nm: Buf::default(),
            sel: Buf::default(),
            nsel: Buf::default(),
            idx_heads: 4,
            wave: 8,
            warm_t: 0,
        }
    }
}

pub(super) struct Kstep {
    pub(super) buf: CUdeviceptr,
    pub(super) graphs: Vec<Option<(u32, ffi::CUgraphExec, ffi::CUgraph)>>,
    pub(super) failed: bool,
}

pub(super) struct Graph {
    pub(super) graph_cache: Vec<GraphEntry>,
    pub(super) graph_failed: bool,
    pub(super) capture_pinned_src: bool,
    pub(super) pin_embed: *mut std::ffi::c_void,
    pub(super) pin_pos: *mut std::ffi::c_void,
    pub(super) pin_out: *mut std::ffi::c_void,
    pub(super) pin_out_len: usize,
    pub(super) pin_batch_tok: *mut std::ffi::c_void,
    pub(super) pin_row_blk: *mut std::ffi::c_void,
    pub(super) pin_row_pp: *mut std::ffi::c_void,
    pub(super) pin_row_ring: *mut std::ffi::c_void,
    pub(super) pin_row_st: *mut std::ffi::c_void,
    pub(super) pin_row_sc: *mut std::ffi::c_void,
    pub(super) pin_batch_lg: *mut std::ffi::c_void,
    pub(super) pin_g4_out: *mut std::ffi::c_void,
    pub(super) pin_g4_out_len: usize,
    pub(super) pin_batch_in: *mut std::ffi::c_void,
    pub(super) pin_batch_pos: *mut std::ffi::c_void,
    pub(super) batch_graphs: Vec<((usize, Vec<usize>), ffi::CUgraphExec, ffi::CUgraph)>,
    pub(super) spec_graphs: Vec<((usize, usize), ffi::CUgraphExec, ffi::CUgraph)>,
    pub(super) spec_graph_ok: bool,
    pub(super) capture_no_host: bool,
    pub(super) batch_capture_failed: bool,
    pub(super) g4_graphs: Vec<G4GraphEntry>,
    pub(super) g4_capture_failed: bool,
}

pub(super) struct G4GraphEntry {
    pub(super) t: usize,
    pub(super) slots: Vec<usize>,
    pub(super) head_last: bool,
    pub(super) exec: ffi::CUgraphExec,
    pub(super) handle: ffi::CUgraph,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerWindow {
    pub start: usize,
    pub end: usize,
}

impl LayerWindow {
    pub fn full(n_layers: usize) -> Self {
        Self {
            start: 0,
            end: n_layers,
        }
    }
    pub fn n(&self) -> usize {
        self.end - self.start
    }
    pub fn abs(&self, off: usize) -> usize {
        self.start + off
    }
    pub fn contains(&self, layer: usize) -> bool {
        layer >= self.start && layer < self.end
    }
    pub fn owns_final(&self, n_layers: usize) -> bool {
        self.end == n_layers
    }
    pub fn gdn_start(&self, interval: usize) -> usize {
        (0..self.start).filter(|&l| (l + 1) % interval != 0).count()
    }
    pub fn owns_attn(&self, l: usize, interval: usize, n_layers: usize) -> bool {
        let il = (l + 1) * interval - 1;
        if il < n_layers {
            self.contains(il)
        } else {
            self.owns_final(n_layers)
        }
    }
}
pub struct W4a16Dec {
    pub(super) moe: MoE,
    pub(super) tuning_lru_cap: Option<usize>,
    pub(super) tuning_lru_frac: Option<u64>,
    pub device_ord: i32,
    pub(super) mtp: Mtp,
    pub(super) kstep: Kstep,
    pub(super) attn: Attn,
    pub(super) gdn: Gdn,
    pub(super) fn_: Fn,
    pub(super) graph: Graph,
    pub cc: CudaCtx,
    ucache: UnifiedCacheRegistry,
    pub hidden: usize,
    pub n_layers: usize,
    pub window: LayerWindow,
    pub n_slots: usize,
    pub slot_pos: Vec<u32>,
    pub kv_assigned: Vec<usize>,
    pub kv_floor: Vec<u32>,
    lins: HashMap<String, (CUdeviceptr, CUdeviceptr, usize, usize)>,
    g4_a: Buf,
    g4_y: Buf,
    g4_pes: Vec<CUdeviceptr>,
    g4_ls: Vec<f32>,
    g4_pool: Vec<Buf>,
    g4_atab: (Vec<usize>, Vec<u32>),
    lin_grp: HashMap<String, (usize, bool)>,
    lins_mk: HashMap<String, (CUdeviceptr, usize, usize)>,
    lins_msc: HashMap<String, (CUdeviceptr, usize, usize)>,
    marlin_bytes: u64,
    marlin_dense: bool,
    mtp_attn_planned: bool,
    plains: HashMap<String, (CUdeviceptr, usize, usize)>,
    plain_weights: bool,
    weights_bytes: u64,
    experts_bytes: u64,
    xh: Buf,
    dy: Buf,
    dnw: CUdeviceptr,
    norm_w_rows: usize,
    norm: NormBufs,
    ew: EwBufs,
    dres: CUdeviceptr,
    dab_dev: CUdeviceptr,
    dchain: [CUdeviceptr; 5],
    stg_w0: usize,
    stg_w1: usize,
    stg_w2: usize,
    chain_bufs_ok: bool,
    dx32: Buf,
    dx16: Buf,
    dxp: Buf,
    dabf: Buf,
    dmc: Buf,
    dmlock: CUdeviceptr,
    moe_m13: CUdeviceptr,
    moe_m2: CUdeviceptr,
    moe_s13: CUdeviceptr,
    moe_s2: CUdeviceptr,
    moe_msorted: Buf,
    moe_mexp: Buf,
    moe_mnpost: Buf,
    moe_mc13: Buf,
    moe_mact: Buf,
    moe_mc2: Buf,
    moe_mctmp: Buf,
    dmulti: CUdeviceptr,
    dmulti_meta: Vec<[u32; 8]>,
    dsk: CUdeviceptr,
    dsk_meta: Vec<[Option<SkGroup>; 2]>,
    dsk_part: Buf,
    spec_on: bool,
    spec_scan_on: bool,
    dyt: Buf,
    head_w: CUdeviceptr,
    head_sc: CUdeviceptr,
    head_n: usize,
    head_k: usize,
    head_out: CUdeviceptr,
    argmax_out: CUdeviceptr,
    dbatch_lg: CUdeviceptr,
    dbatch_am: CUdeviceptr,
    drow_blk: CUdeviceptr,
    drow_pp: CUdeviceptr,
    drow_ring: CUdeviceptr,
    drow_st: CUdeviceptr,
    drow_sc: CUdeviceptr,
    pub batch_tmax_min: usize,
    emb_dev: CUdeviceptr,
    emb_n: usize,
    emb_k: usize,
    emb_bf16: bool,
    dtg: CUdeviceptr,
    pub debug_layers: bool,
    pub route_capture: bool,
    pub route_out: Vec<u32>,
}

unsafe impl Send for W4a16Dec {}

mod attn;
mod blocktable;
mod buffers;
mod chains;
mod fnchain;
mod g4;
mod gdn;
mod graphs;
mod kernels;
mod kstep;
mod mem;
mod moe;
mod mtp;
mod prefill;
mod probe;
mod shard;
pub mod unified_cache;
mod upload;
pub use shard::{forward_2shard, forward_2shard_argmax, forward_2shard_toks};

mod spec;

#[cfg(test)]
mod tests;

impl W4a16Dec {
    pub fn new_dev(
        n_slots: usize,
        hidden: usize,
        n_layers: usize,
        device: i32,
    ) -> Result<Self, String> {
        let mut cc = CudaCtx::new_dev(device)?;
        for a in crate::rawcuda::assets::ASSETS.iter().filter(|a| a.boot) {
            cc.load_fatbin(a.name, &crate::rawcuda::assets::asset_bytes(a)?, a.syms)?;
        }
        let mut ucache = UnifiedCacheRegistry::new();
        if let Some((free, _)) = crate::cuda_mem_free_dev(device) {
            ucache.set_budget(Tier::Hbm, free);
        }
        ucache.set_enforce(true);
        Ok(W4a16Dec {
            moe: MoE {
                n_experts: 0,
                top_k: 0,
                moe_ffn: 0,
                shared_ffn: 0,
                glu_gelu: false,
                moe_group: 32,
                moe_scale_bf16: true,
                moe_tab: Vec::new(),
                moe_resident: false,
                hot_per_layer: 0,
                hot_tab: Vec::new(),
                moe_dev_tab: 0,
                moe_idx: 0,
                moe_wt: 0,
                pin_moe_idx: std::ptr::null_mut(),
                pin_moe_idx_cap: 0,
                dsg: 0,
                ev_fork: Vec::new(),
                dsh: [0, 0],
                dsh_cap: 0,
                mcache_q: 0,
                mcache_s: 0,
                mcache_k: 0,
                mcache_k_plan: 0,
                mcache_ids: Vec::new(),
                mcache_age: Vec::new(),
                mcache_tick: 0,
                mcache_hits: 0,
                mcache_miss: 0,
                moe_multi_tab: 0,
                moe_multi_rows: Vec::new(),
                moe_multi_rows_t3: Vec::new(),
                moe_gslot: 0,
                moe_gcnt: 0,
                moe_goff: 0,
                moe_gmax: 0,
                exp: ExpBufs::default(),
                stg: StgBufs::default(),
                host_stage: Vec::new(),
                host_stage_reg: 0,
                drt: Buf::default(),
                dmo: Buf::default(),
                moe_bufs_ok: false,
            },
            tuning_lru_cap: None,
            tuning_lru_frac: None,
            device_ord: device,
            mtp: Mtp {
                mtp_ok: false,
                mtp_nw: 0,
                mtp_zero: 0,
                mtp_buf: 0,
                mtp_off: [0; 17],
                mtp_prev_off: 0,
                mtp_layer: 0,
                mtp_rows: 0,
                mtp_ffn: 0,
                mtp_draft: Vec::new(),
                mtp_pos: Vec::new(),
                mtp_prev_pos: Vec::new(),
                mtp_off_slot: Vec::new(),
                mtp_graphs: vec![None; 9],
                mtp_graph_failed: false,
                mtp_pin_emb: std::ptr::null_mut(),
                mtp_pin_out: std::ptr::null_mut(),
                mtp_chain_ctr: 0,
                mtp_chain_log: 0,
                mtp_chain_graphs: Vec::new(),
            },
            kstep: Kstep {
                buf: 0,
                graphs: Vec::new(),
                failed: false,
            },
            attn: Attn {
                layers: Vec::new(),
                attn: None,
                dqnw_a: 0,
                dknw_a: 0,
                g4_rp: 0,
                dkc: 0,
                dvc: 0,
                dksc: 0,
                dvsc: 0,
                dblk_tab: 0,
                kv_pool: None,
                kv_budget: None,
                kv_vram_cap: None,
                dkc_h: 0,
                dvc_h: 0,
                dksc_h: 0,
                dvsc_h: 0,
                hkc_h: std::ptr::null_mut(),
                hvc_h: std::ptr::null_mut(),
                hksc_h: std::ptr::null_mut(),
                hvsc_h: std::ptr::null_mut(),
                kv_vram_blocks: 0,
                kv_host_bytes: 0,
                kv_arena_bytes: 0,
                dpp: 0,
                dqg_a: 0,
                dkin_a: 0,
                dvin_a: 0,
                dqh_a: 0,
                doutv_a: 0,
                dattn_part: 0,
                attn_t_cap: 0,
            },
            fn_: Fn::default(),
            gdn: Gdn {
                gdn: None,
                sig_gate: false,
                dcw: 0,
                dab_c: 0,
                dalog: 0,
                ddtb: 0,
                dnwg: 0,
                dring: 0,
                dgst: 0,
                dsc: 0,
                dsc_ones: 0,
                dqkv: 0,
                dzv: 0,
                dgxn: 0,
                dgq: 0,
                dgk: 0,
                dgv: 0,
                dq2: 0,
                dk2: 0,
                dv2: 0,
                dbg: 0,
                dgo: 0,
                dgpart: 0,
                dgdc: 0,
                dakq: 0,
                dhbuf: 0,
                dwu: 0,
                ddcbuf: 0,
                dgate: 0,
                gdn_t_cap: 0,
                dwy_s0: 0,
                dwy_dc: 0,
                dwy_g: 0,
                dwy_k: 0,
                dsnap_ring: 0,
                dring0: 0,
                dsave: 0,
                dchk: std::ptr::null_mut(),
                dchk_pos: Vec::new(),
                dchk_saved: Vec::new(),
                dchk_use: Vec::new(),
                dchk_tick: 0,
            },
            graph: Graph {
                graph_cache: Vec::new(),
                graph_failed: false,
                capture_pinned_src: false,
                pin_embed: std::ptr::null_mut(),
                pin_pos: std::ptr::null_mut(),
                pin_out: std::ptr::null_mut(),
                pin_out_len: 0,
                pin_batch_tok: std::ptr::null_mut(),
                pin_row_blk: std::ptr::null_mut(),
                pin_row_pp: std::ptr::null_mut(),
                pin_row_ring: std::ptr::null_mut(),
                pin_row_st: std::ptr::null_mut(),
                pin_row_sc: std::ptr::null_mut(),
                pin_batch_lg: std::ptr::null_mut(),
                pin_g4_out: std::ptr::null_mut(),
                pin_g4_out_len: 0,
                pin_batch_in: std::ptr::null_mut(),
                pin_batch_pos: std::ptr::null_mut(),
                batch_graphs: Vec::new(),
                spec_graphs: Vec::new(),
                spec_graph_ok: true,
                capture_no_host: false,
                batch_capture_failed: false,
                g4_graphs: Vec::new(),
                g4_capture_failed: false,
            },
            cc,
            ucache,
            hidden,
            n_layers,
            window: LayerWindow::full(n_layers),
            n_slots: n_slots.max(1),
            slot_pos: vec![0; n_slots.max(1)],
            kv_assigned: vec![0; n_slots.max(1)],
            kv_floor: vec![0; n_slots.max(1)],
            lins: HashMap::new(),
            g4_a: Buf { ptr: 0, cap: 0 },
            g4_y: Buf { ptr: 0, cap: 0 },
            g4_pes: Vec::new(),
            g4_ls: Vec::new(),
            g4_pool: Vec::new(),
            g4_atab: (Vec::new(), Vec::new()),
            lin_grp: HashMap::new(),
            lins_mk: HashMap::new(),
            lins_msc: HashMap::new(),
            marlin_bytes: 0,
            marlin_dense: false,
            mtp_attn_planned: false,
            plains: HashMap::new(),
            plain_weights: false,
            weights_bytes: 0,
            experts_bytes: 0,
            xh: Buf::default(),
            dy: Buf::default(),
            dnw: 0,
            norm_w_rows: 0,
            norm: NormBufs::default(),
            ew: EwBufs::default(),
            dres: 0,
            dab_dev: 0,
            dchain: [0; 5],
            stg_w0: 0,
            stg_w1: 0,
            stg_w2: 0,
            chain_bufs_ok: false,
            dx32: Buf::default(),
            dx16: Buf::default(),
            dxp: Buf::default(),
            dabf: Buf::default(),
            dmc: Buf::default(),
            dmlock: 0,
            moe_m13: 0,
            moe_m2: 0,
            moe_s13: 0,
            moe_s2: 0,
            moe_msorted: Buf::default(),
            moe_mexp: Buf::default(),
            moe_mnpost: Buf::default(),
            moe_mc13: Buf::default(),
            moe_mact: Buf::default(),
            moe_mc2: Buf::default(),
            moe_mctmp: Buf::default(),
            dmulti: 0,
            dmulti_meta: Vec::new(),
            spec_on: false,
            spec_scan_on: true,
            dsk: 0,
            dsk_meta: Vec::new(),
            dsk_part: Buf::default(),
            dyt: Buf::default(),
            head_w: 0,
            head_sc: 0,
            head_n: 0,
            head_k: 0,
            head_out: 0,
            argmax_out: 0,
            dbatch_lg: 0,
            dbatch_am: 0,
            drow_blk: 0,
            drow_pp: 0,
            drow_ring: 0,
            drow_st: 0,
            drow_sc: 0,
            batch_tmax_min: 0,
            emb_dev: 0,
            emb_n: 0,
            emb_k: 0,
            emb_bf16: false,
            dtg: 0,
            debug_layers: false,

            route_capture: false,
            route_out: Vec::new(),
        })
    }

    pub fn layer_window(&self) -> LayerWindow {
        self.window
    }

    pub fn owns_final_layer(&self) -> bool {
        self.window.owns_final(self.n_layers)
    }

    pub(super) fn owns_attn_layer(&self, l: usize, dm: &AttnDims) -> bool {
        let real = self.n_layers / dm.interval.max(1);
        if l < real {
            self.window.owns_attn(l, dm.interval, self.n_layers)
        } else {
            self.mtp_attn_planned && l < dm.n_attn
        }
    }

    pub(super) fn n_attn_owned(&self, dm: &AttnDims) -> usize {
        (0..dm.n_attn)
            .filter(|&l| self.owns_attn_layer(l, dm))
            .count()
    }

    pub fn new(n_slots: usize, hidden: usize, n_layers: usize) -> Result<Self, String> {
        Self::new_dev(n_slots, hidden, n_layers, 0)
    }

    pub fn set_mtp_attn_planned(&mut self, on: bool) {
        self.mtp_attn_planned = on;
    }

    pub fn set_layer_window(&mut self, start: usize, end: usize) -> Result<(), String> {
        if start >= end || end > self.n_layers {
            return Err(format!(
                "set_layer_window: [{start},{end}) 위반 — n_layers={}",
                self.n_layers
            ));
        }
        self.window = LayerWindow { start, end };
        Ok(())
    }

    pub fn free_vram(&self) -> Option<(u64, u64)> {
        crate::cuda_mem_free_dev(self.device_ord)
    }

    pub fn device_name(&self) -> &str {
        &self.cc.device_name
    }

    pub fn free_all_device(&mut self) {
        let _ = self.free_kv_host_arenas();
        self.cc.free_all();
    }

    pub fn gemv_host(&mut self, name: &str, x: &[f32]) -> Result<Vec<f32>, String> {
        let _g = self.cc.guard()?;
        let Some(&(dq, ds, n, k)) = self.lins.get(name) else {
            return Err(format!("gemv: 상주 선형 없음: {name}"));
        };
        if x.len() != k {
            return Err(format!("gemv {name}: x={} != k={k}", x.len()));
        }
        if k > self.xh.cap {
            self.graph_invalidate();
            self.cc.sync()?;
            if self.xh.ptr != 0 {
                self.cc.free(self.xh.ptr)?;
            }
            self.xh.ptr = 0;
            self.xh.cap = 0;
            self.xh.ptr = self.cc.alloc(k * 2)?;
            self.xh.cap = k;
        }
        if n > self.dy.cap {
            self.graph_invalidate();
            self.cc.sync()?;
            if self.dy.ptr != 0 {
                self.cc.free(self.dy.ptr)?;
            }
            self.dy.ptr = 0;
            self.dy.cap = 0;
            self.dy.ptr = self.cc.alloc(n * 4)?;
            self.dy.cap = n;
        }
        if k > self.dx32.cap {
            self.graph_invalidate();
            self.cc.sync()?;
            if self.dx32.ptr != 0 {
                self.cc.free(self.dx32.ptr)?;
            }
            self.dx32.ptr = 0;
            self.dx32.cap = 0;
            self.dx32.ptr = self.cc.alloc(k * 4)?;
            self.dx32.cap = k;
        }
        let xf: Vec<f32> = x
            .iter()
            .map(|&v| crate::rawcuda::gptq4::h2f(f32_to_f16(v)))
            .collect();
        let xb = unsafe { std::slice::from_raw_parts(xf.as_ptr() as *const u8, xf.len() * 4) };
        self.cc.h2d(self.dx32.ptr, xb)?;
        if dq == 0 {
            let x16 = self.cast_f16_dev(self.dx32.ptr, k)?;
            if !self.marlin_gemv1_launch(name, x16, self.dy.ptr)? {
                return Err(format!("gemv {name}: 마린 경로 실패 — 원본 q 폐기됨"));
            }
            let mut ob = vec![0u8; n * 4];
            self.cc.d2h_async(ob.as_mut_ptr(), self.dy.ptr, n * 4)?;
            self.cc.sync()?;
            return Ok(ob
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect());
        }
        let (grp, sbf) = self.lin_grp.get(name).copied().unwrap_or((128, false));
        let sym = crate::rawcuda::gptq4::kernel_sym(true, grp, sbf)?;
        let f = self.cc.function(sym)?;
        let (mut p_q, mut p_s, mut p_x, mut p_y) = (dq, ds, self.dx32.ptr, self.dy.ptr);
        let (mut p_n, mut p_k, mut p_t) = (n as i32, k as i32, 1i32);
        self.cc.launch(
            f,
            n.div_ceil(8) as u32,
            1,
            512,
            &mut crate::rawcuda::args::l7(
                &mut p_q, &mut p_s, &mut p_x, &mut p_y, &mut p_n, &mut p_k, &mut p_t,
            ),
        )?;
        let mut ob = vec![0u8; n * 4];
        self.cc.d2h_async(ob.as_mut_ptr(), self.dy.ptr, n * 4)?;
        self.cc.sync()?;
        Ok(ob
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    pub fn gemv_f32_host(&mut self, name: &str, x: &[f32]) -> Result<Vec<f32>, String> {
        let _g = self.cc.guard()?;
        let Some(&(dq, ds, n, k)) = self.lins.get(name) else {
            return Err(format!("gemv_f32: 상주 선형 없음: {name}"));
        };
        if x.len() != k {
            return Err(format!("gemv_f32 {name}: x={} != k={k}", x.len()));
        }
        if n > self.dy.cap {
            self.graph_invalidate();
            self.cc.sync()?;
            if self.dy.ptr != 0 {
                self.cc.free(self.dy.ptr)?;
            }
            self.dy.ptr = self.cc.alloc(n * 4)?;
            self.dy.cap = n;
        }
        if k > self.dx32.cap {
            self.graph_invalidate();
            self.cc.sync()?;
            if self.dx32.ptr != 0 {
                self.cc.free(self.dx32.ptr)?;
            }
            self.dx32.ptr = self.cc.alloc(k * 4)?;
            self.dx32.cap = k;
        }
        let xb = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) };
        self.cc.h2d(self.dx32.ptr, xb)?;
        if dq == 0 {
            let x16 = self.cast_f16_dev(self.dx32.ptr, k)?;
            if !self.marlin_gemv1_launch(name, x16, self.dy.ptr)? {
                return Err(format!("gemv_f32 {name}: 마린 경로 실패 — 원본 q 폐기됨"));
            }
            let mut ob = vec![0u8; n * 4];
            self.cc.d2h_async(ob.as_mut_ptr(), self.dy.ptr, n * 4)?;
            self.cc.sync()?;
            return Ok(ob
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect());
        }
        let (grp, sbf) = self.lin_grp.get(name).copied().unwrap_or((128, false));
        if k > grp * 256 {
            return Err(format!(
                "gemv_f32 {name}: k={k} > {} (t=1 smem sc 계약, g{grp})",
                grp * 256
            ));
        }
        let f = self
            .cc
            .function(crate::rawcuda::gptq4::kernel_sym(false, grp, sbf)?)?;
        let (mut p_q, mut p_s, mut p_x, mut p_y) = (dq, ds, self.dx32.ptr, self.dy.ptr);
        let (mut p_n, mut p_k) = (n as i32, k as i32);
        self.cc.launch(
            f,
            n as u32,
            1,
            64,
            &mut crate::rawcuda::args::l6(
                &mut p_q, &mut p_s, &mut p_x, &mut p_y, &mut p_n, &mut p_k,
            ),
        )?;
        let mut ob = vec![0u8; n * 4];
        self.cc.d2h_async(ob.as_mut_ptr(), self.dy.ptr, n * 4)?;
        self.cc.sync()?;
        Ok(ob
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    pub fn ew_host(&mut self, g: &[f32], u: &[f32]) -> Result<Vec<f32>, String> {
        self.ew_act_host("ew", g, u)
    }

    pub fn ew_gelu_host(&mut self, g: &[f32], u: &[f32]) -> Result<Vec<f32>, String> {
        self.ew_act_host("ew_gelu", g, u)
    }

    fn ew_act_host(&mut self, kernel: &str, g: &[f32], u: &[f32]) -> Result<Vec<f32>, String> {
        if g.is_empty() || g.len() != u.len() {
            return Err(format!("{kernel}: g={} u={} 계약 위반", g.len(), u.len()));
        }
        let _g = self.cc.guard()?;
        self.ensure_ew_bufs(g.len())?;
        let gb = unsafe { std::slice::from_raw_parts(g.as_ptr() as *const u8, g.len() * 4) };
        let ub = unsafe { std::slice::from_raw_parts(u.as_ptr() as *const u8, u.len() * 4) };
        self.cc.h2d(self.ew.dewg, gb)?;
        self.cc.h2d(self.ew.dewu, ub)?;
        let f = self.cc.function(kernel)?;
        let mut nn = g.len() as i32;
        let (mut a0, mut a1, mut a2, mut a3, mut a4) =
            (self.ew.dewg, self.ew.dewu, self.ew.dew, 0u64, 0u64);
        self.cc.launch(
            f,
            g.len().div_ceil(128) as u32,
            1,
            128,
            &mut crate::rawcuda::args::l6(&mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut nn),
        )?;
        let mut yb = vec![0u8; g.len() * 4];
        self.cc.d2h(&mut yb, self.ew.dew)?;
        self.cc.sync()?;
        Ok(unsafe { std::slice::from_raw_parts(yb.as_ptr() as *const f32, g.len()) }.to_vec())
    }

    fn lin_spec(&self, name: &str) -> Result<(CUdeviceptr, CUdeviceptr, usize, usize), String> {
        self.lins
            .get(name)
            .copied()
            .ok_or_else(|| format!("상주 선형 없음: {name}"))
    }

    fn lin_mk(&self, name: &str) -> Option<(CUdeviceptr, usize, usize)> {
        self.lins_mk.get(name).copied()
    }

    pub(super) fn has_orig_q(&self, name: &str) -> bool {
        self.lins.get(name).is_some_and(|v| v.0 != 0)
    }

    pub fn dense_marlin_only(&self) -> bool {
        self.marlin_dense
    }

    pub fn moe_mode(&self) -> &'static str {
        if self.moe.n_experts == 0 {
            "none"
        } else if self.moe.moe_resident {
            "resident"
        } else {
            "streaming"
        }
    }

    pub fn mcache_stats(&self) -> (u64, u64, usize) {
        (
            self.moe.mcache_hits,
            self.moe.mcache_miss,
            self.moe.mcache_k,
        )
    }

    pub fn is_moe(&self) -> bool {
        self.moe.n_experts > 0
    }

    pub fn moe_experts_resident(&self) -> bool {
        self.moe.moe_resident
    }

    fn plain_spec(&self, name: &str) -> Result<(CUdeviceptr, usize, usize), String> {
        self.plains
            .get(name)
            .copied()
            .ok_or_else(|| format!("상주 플레인 없음: {name}"))
    }
}
