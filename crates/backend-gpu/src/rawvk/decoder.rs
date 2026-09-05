//! VkDecoder — GDN/어텐션 GPU 상주 디코드 (plans/19 2단계).
//! 커널 8종은 gdn-check ★ 검증 완료. 기존 gemv/quant/rms/silu SPIR-V 재사용.
//! rawhip DecodeState 대칭 — 배치 모드(단일 제출+배리어).

use crate::rawvk::context::{VkBuf, VkCtx};
use ash::vk;
use std::collections::HashMap;

const GDN_CONV_SPV: &[u8] = include_bytes!("spv/gdn_conv_t.spv");
const GDN_CONV_STATE_SPV: &[u8] = include_bytes!("spv/gdn_conv_state.spv");
const SPLIT3_SPV: &[u8] = include_bytes!("spv/split3.spv");
const L2_SPV: &[u8] = include_bytes!("spv/l2_rows2.spv");
const BETA_G_SPV: &[u8] = include_bytes!("spv/gdn_beta_g.spv");
const GDN_AR_SPV: &[u8] = include_bytes!("spv/gdn_ar.spv");
const NORM_GATED_SPV: &[u8] = include_bytes!("spv/norm_gated.spv");
const QK_ROPE_SPV: &[u8] = include_bytes!("spv/qk_rope.spv");
const KV_APPEND_SPV: &[u8] = include_bytes!("spv/kv_append.spv");
const QSA_FLASH_SPV: &[u8] = include_bytes!("spv/qsa_flash.spv");
const COPY_OFF_SPV: &[u8] = include_bytes!("spv/copy_off.spv");
const TILE128_SPV: &[u8] = include_bytes!("spv/tile128_q5k.spv");
const GEMM_I8_SPV: &[u8] = include_bytes!("spv/gemm_i8.spv");
const QUANT_B8_SPV: &[u8] = include_bytes!("spv/quant_b8.spv");
const QUANT_B8V2_SPV: &[u8] = include_bytes!("spv/quant_b8v2.spv");
const GEMM_I8V2_SPV: &[u8] = include_bytes!("spv/gemm_i8v2.spv");

/// q5_K 사전 언패분 — i8 가중 + 블록 스케일 (gemm_i8 전용).
struct I8W {
    w: VkBuf,
    wsp: VkBuf,
    wsm: VkBuf,
    n_out: usize,
    n_in: usize,
}

/// ishs/faccs 워크그룹 상한 — 전 i8w 텐서의 max(n_out/16).
fn i8_wg_max(map: &HashMap<String, I8W>) -> usize {
    map.values().map(|e| e.n_out.div_ceil(16)).max().unwrap_or(1)
}

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
    b_lg: VkBuf,  // head 로짓 [n_vocab] — b_gout 오버플로 수정 (T_MAX*n < vocab)
    b_lg_t: VkBuf, // head 로짓 [T_MAX][n_vocab] — verify 전 행 (plans/20)
    b_am: VkBuf,  // argmax 8바이트
    pipes: HashMap<&'static str, Pipes>,
    split_ctr: usize,
    // ── MTP (blk.64) — Phase A: 전부 t=1 검증 커널 재사용
    mtp_on: bool,
    n_vocab: usize,
    m_e: VkBuf,     // [n] 토큰 임베딩 / rms 임시
    m_cat: VkBuf,   // [2n] enorm‖hnorm
    m_xq2: VkBuf,   // [2n] q8
    m_cur: VkBuf,   // [n] MTP hidden
    m_xq: VkBuf,    // [n] q8
    m_h: VkBuf,     // [n] 호스트 h 업로드
    m_kv_k: Vec<VkBuf>,
    m_kv_v: Vec<VkBuf>,
    // GDN/conv 스냅샷 (spec 부분수용 롤백) — 매핑 ptr 직접 복사
    snap_gdn: Vec<Vec<f32>>,
    snap_conv: Vec<Vec<f32>>,
    // ── i8 coopmat GEMM (plans/23) — q5_K 사전 언패분
    i8w: HashMap<String, I8W>,
    wsr: HashMap<String, VkBuf>, // v2 행 스케일
    b8: VkBuf,   // [T_MAX][n_max] i8 활성 매트릭스
    ydb: VkBuf,  // [T_MAX][n_sub_max] f32
    qsb: VkBuf,  // [T_MAX][n_sub_max] i32
    ishs: VkBuf, // [640][256] i32 — coopMatStore SSBO (workgroup별)
    faccs: VkBuf, // [640][256] f32
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
        let ds = guard.as_mut().ok_or("vkdecoder: 미초기화")?;
        ds.step(seq, pos, emb)
    }

    /// 프리필 — 행[t][n_embd]별 t=1 스텝 (기본 구현의 512-float 청크 절단 결함 회피).
    /// t=1 스텝 산술은 p1 검증 경로와 동일 — 순차 상태 적립으로 수치 불변.
    fn raw_prefill(&self, seq: usize, pos0: usize, emb: &[f32]) -> Result<Vec<f32>, String> {
        let mut guard = self.st.lock().map_err(|e| e.to_string())?;
        let ds = guard.as_mut().ok_or("vkdecoder: 미초기화")?;
        let n = ds.n_embd;
        // 기본 per-token (검증 경로). LLM170_VKD_BATCH=1 옵트인 시에만
        // step_batch 청크 — 2026-09-04 계측: 배치 경로 산술 발산(idx5 동률
        // 플립, W4A8급) — 커널 원인 조사 전까지 비활성.
        if std::env::var_os("LLM170_VKD_BATCH").is_none() {
            let mut last = None;
            for (ti, ch) in emb.chunks(n).enumerate() {
                last = Some(ds.step(seq, pos0 + ti, ch)?);
            }
            return Ok(last.unwrap_or_default());
        }
        let mut last = None;
        for (off, ch) in emb.chunks(T_MAX * n).enumerate() {
            last = Some(ds.step_batch(seq, pos0 + off, ch, false)?);
        }
        Ok(last.unwrap_or_default())
    }

    /// raw_step + 최종 hidden 회수 (MTP 훅용).
    fn raw_step_h(
        &self,
        seq: usize,
        pos: usize,
        emb: &[f32],
        h_out: &mut Vec<f32>,
    ) -> Result<Vec<f32>, String> {
        let mut guard = self.st.lock().map_err(|e| e.to_string())?;
        let ds = guard.as_mut().ok_or("vkdecoder: 미초기화")?;
        let lg = ds.step(seq, pos, emb)?;
        h_out.clear();
        h_out.extend_from_slice(&ds.hidden_row());
        Ok(lg)
    }

    /// raw_prefill + 전 토큰 hidden 회수 (MTP KV 적립용).
    fn raw_prefill_h(
        &self,
        seq: usize,
        pos0: usize,
        emb: &[f32],
        h_all: &mut Vec<f32>,
    ) -> Result<Vec<f32>, String> {
        let mut guard = self.st.lock().map_err(|e| e.to_string())?;
        let ds = guard.as_mut().ok_or("vkdecoder: 미초기화")?;
        ds.verify_rows(seq, pos0, emb, &mut Vec::new(), h_all)
    }

    /// 배치 검증 (MTP spec) — per-token step = 디코드 산술과 동일 (비트계약).
    fn raw_verify(
        &self,
        seq: usize,
        pos0: usize,
        emb: &[f32],
        argmaxes: &mut Vec<u32>,
        h_all: &mut Vec<f32>,
    ) -> Result<(), String> {
        let mut guard = self.st.lock().map_err(|e| e.to_string())?;
        let ds = guard.as_mut().ok_or("vkdecoder: 미초기화")?;
        ds.verify_rows(seq, pos0, emb, argmaxes, h_all)?;
        Ok(())
    }

    /// np 배치 디코드 — seq별 순차 step (스트림 = 싱글 경로와 동일).
    fn raw_step_multi(
        &self,
        seqs: &[usize],
        poss: &[u32],
        emb: &[f32],
    ) -> Result<Vec<Vec<f32>>, String> {
        let mut guard = self.st.lock().map_err(|e| e.to_string())?;
        let ds = guard.as_mut().ok_or("vkdecoder: 미초기화")?;
        let n = ds.n_embd;
        let mut out = Vec::with_capacity(seqs.len());
        for (i, (&sq, &ps)) in seqs.iter().zip(poss.iter()).enumerate() {
            out.push(ds.step(sq, ps as usize, &emb[i * n..(i + 1) * n])?);
        }
        Ok(out)
    }

    /// np×spec 병합 검증 — 그룹(seq-major)별 per-token step.
    fn verify_batch_ms(
        &self,
        seqs: &[usize],
        poss: &[usize],
        group_starts: &[usize],
        emb: &[f32],
        argmaxes: &mut Vec<u32>,
        h_all: &mut Vec<f32>,
    ) -> Result<(), String> {
        let mut guard = self.st.lock().map_err(|e| e.to_string())?;
        let ds = guard.as_mut().ok_or("vkdecoder: 미초기화")?;
        let n = ds.n_embd;
        let total = emb.len() / n;
        argmaxes.clear();
        argmaxes.resize(total, 0);
        let mut am_i = Vec::with_capacity(total);
        let mut h_i = Vec::with_capacity(total * n);
        for (gi, (&sq, &p0)) in seqs.iter().zip(poss.iter()).enumerate() {
            let g0 = group_starts[gi];
            let g1 = group_starts.get(gi + 1).copied().unwrap_or(total);
            let mut am_g = Vec::new();
            let mut h_g = Vec::new();
            ds.verify_rows(sq, p0, &emb[g0 * n..g1 * n], &mut am_g, &mut h_g)?;
            am_i.extend(am_g);
            h_i.extend(h_g);
        }
        *argmaxes = am_i;
        *h_all = h_i;
        Ok(())
    }

    /// MTP 1스텝 (호스트 h, head, h_next 회수) — 프리필/디코드 훅용.
    fn mtp_step_gpu(
        &self,
        seq: usize,
        tok_emb: &[f32],
        h: &[f32],
        pos: usize,
    ) -> Result<(u32, Vec<f32>), String> {
        let mut guard = self.st.lock().map_err(|e| e.to_string())?;
        let ds = guard.as_mut().ok_or("vkdecoder: 미초기화")?;
        let am = ds
            .mtp_step_g(seq, tok_emb, false, h, pos, true)?
            .ok_or("mtp head")?;
        let mut h_next = vec![0f32; ds.n_embd];
        unsafe { std::ptr::copy_nonoverlapping(ds.m_cur.ptr as *const f32, h_next.as_mut_ptr(), ds.n_embd) };
        Ok((am, h_next))
    }

    /// MTP 체인 스텝 — h를 내부 mtp_cur에서 직접 (h2d 제거).
    fn mtp_step_chain(&self, seq: usize, tok_emb: &[f32], pos: usize) -> Result<u32, String> {
        let mut guard = self.st.lock().map_err(|e| e.to_string())?;
        let ds = guard.as_mut().ok_or("vkdecoder: 미초기화")?;
        ds.mtp_step_g(seq, tok_emb, true, &[], pos, true)?
            .ok_or("mtp head".into())
    }

    /// MTP 상태 진행 (호스트 trunk h, head 없음) — spec 수용 후 KV 동기.
    fn mtp_step_adv(
        &self,
        seq: usize,
        tok_emb: &[f32],
        h: &[f32],
        pos: usize,
    ) -> Result<(), String> {
        let mut guard = self.st.lock().map_err(|e| e.to_string())?;
        let ds = guard.as_mut().ok_or("vkdecoder: 미초기화")?;
        ds.mtp_step_g(seq, tok_emb, false, h, pos, false)?;
        Ok(())
    }

    /// 시퀀스 상태 제로화 (서버 슬롯 반환) — 매핑 ptr 직접 (GPU 유휴 보장).
    fn raw_reset(&self, seq: usize) -> Result<(), String> {
        let mut guard = self.st.lock().map_err(|e| e.to_string())?;
        let ds = guard.as_mut().ok_or("vkdecoder: 미초기화")?;
        let gl = ds.dt_rank * ds.d_state * ds.d_state;
        let cl = (ds.conv_k - 1) * ds.conv_ch;
        for r in 0..ds.st_gdn.len() {
            if seq < ds.st_gdn[r].len() {
                unsafe { std::ptr::write_bytes(ds.st_gdn[r][seq].ptr as *mut f32, 0, gl) };
                unsafe { std::ptr::write_bytes(ds.st_conv[r][seq].ptr as *mut f32, 0, cl) };
            }
        }
        Ok(())
    }

    /// GDN/conv 상태 스냅샷·복원 (spec 부분수용 롤백).
    fn gdn_snapshot(&self) -> Result<(), String> {
        let mut guard = self.st.lock().map_err(|e| e.to_string())?;
        guard.as_mut().ok_or("vkdecoder: 미초기화")?.snapshot_states()
    }

    fn gdn_restore(&self) -> Result<(), String> {
        let mut guard = self.st.lock().map_err(|e| e.to_string())?;
        guard.as_mut().ok_or("vkdecoder: 미초기화")?.restore_states()
    }

    /// 정규화 h → head GEMV → argmax (MTP draft).
    fn mtp_head_argmax(&self, h_normed: &[f32]) -> Result<u32, String> {
        let mut guard = self.st.lock().map_err(|e| e.to_string())?;
        let ds = guard.as_mut().ok_or("vkdecoder: 미초기화")?;
        let n = ds.n_embd;
        unsafe { std::ptr::copy_nonoverlapping(h_normed.as_ptr(), ds.m_e.ptr as *mut f32, n.min(h_normed.len())) };
        ds.head_argmax()
    }
}

impl VkDecoder {
    pub fn new() -> Self {
        Self {
            st: std::sync::Mutex::new(None),
        }
    }
}

const T_MAX: usize = 32;

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
        // MTP 탑재·vocab — weights 이동 전 산출.
        let mtp_on = weights.iter().any(|(k, ..)| k == "blk.64.nextn.eh_proj.weight");
        let n_vocab = weights
            .iter()
            .find(|(k, ..)| k == "output.weight")
            .map(|(_, _, _, no, _)| *no)
            .unwrap_or(n);
        // q5_K 원본 캡처 (i8 언패용 — 루프가 weights를 소비하기 전)
        let q5k_src: Vec<(String, Vec<u8>, usize, usize)> = weights
            .iter()
            .filter(|(_, _, ty, _, _)| *ty == 13)
            .map(|(k, d, _, ni, no)| (k.clone(), d.clone(), *ni, *no))
            .collect();
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
        // ── MTP (blk.64) 상주 상태 — has_mtp 시에만.
        let (mut mkk, mut mvv) = (Vec::new(), Vec::new());
        if mtp_on {
            let zeros = vec![0u8; kv_len * 4];
            for _ in 0..n_seqs {
                let k = ctx.alloc(kv_len * 4)?;
                unsafe { std::ptr::copy_nonoverlapping(zeros.as_ptr(), k.ptr, zeros.len()) };
                let v = ctx.alloc(kv_len * 4)?;
                unsafe { std::ptr::copy_nonoverlapping(zeros.as_ptr(), v.ptr, zeros.len()) };
                mkk.push(k);
                mvv.push(v);
            }
        }
        let xq_2n = 2 * n / 4 + 2 * n / 32 + 2 * n / 16;
        let mut ah = |sz: usize| -> Result<VkBuf, String> {
            ctx.alloc_host(sz.max(1) * 4).map_err(|e| e.to_string())
        };
        let m_e = ah(n)?;
        let m_cat = ah(2 * n)?;
        let m_xq2 = ah(xq_2n)?;
        let m_cur = ah(n)?;
        let m_xq = ah(xq_sn)?;
        let m_h = ah(n)?;
        let b_lg = ah(n_vocab)?;
        let b_lg_t = ah(T_MAX * n_vocab)?;
        // ── q5_K i8 언패 (plans/23, gemm_i8) — CPU 병렬, 업로드 1회.
        let mut i8w: HashMap<String, I8W> = HashMap::new();
        let mut wsr_map: HashMap<String, VkBuf> = HashMap::new();
        for (name, data, ni, no) in q5k_src {
            let n_sub = ni / 32;
            let nblk = ni / 256;
            let mut w8 = vec![0i8; no * ni];
            let mut wsp = vec![0f32; no * n_sub];
            let mut wsm = vec![0f32; no * n_sub];
            std::thread::scope(|s| {
                let rows_per = (no / 8).max(1);
                let mut hs = Vec::new();
                for r0 in (0..no).step_by(rows_per) {
                    let end = (r0 + rows_per).min(no);
                    let p8: *mut Vec<i8> = &mut w8;
                    let pp: *mut Vec<f32> = &mut wsp;
                    let pm: *mut Vec<f32> = &mut wsm;
                    let (w8s, wsps, wsms) = unsafe { (&mut *p8, &mut *pp, &mut *pm) };
                    let data = &data;
                    hs.push(s.spawn(move || {
                        for o in r0..end {
                            for bidx in 0..nblk {
                                let wb0 = (o * nblk + bidx) * 176; // 행 오프셋 — 블록은 행 우선
                                let wb = &data[wb0..wb0 + 176];
                                let d = llm170_core::quant::f16(wb, 0);
                                let dm = llm170_core::quant::f16(wb, 2);
                                for j in 0..8 {
                                    let (sc, m) = llm170_core::quant::scale_min_k4_local(wb, j);
                                    let it = j / 2;
                                    let half = j % 2;
                                    let u: u8 = if half == 0 { 1u8 << (2 * it) } else { 2u8 << (2 * it) };
                                    let sb = bidx * 8 + j;
                                    wsps[o * n_sub + sb] = d * sc as f32;
                                    wsms[o * n_sub + sb] = dm * m as f32;
                                    for e in 0..32 {
                                        let q = wb[48 + it * 32 + e];
                                        let nib = if half == 0 { q & 0xF } else { q >> 4 };
                                        let hi = if wb[16 + e] & u != 0 { 16i8 } else { 0i8 };
                                        w8s[o * ni + sb * 32 + e] = nib as i8 + hi;
                                    }
                                }
                            }
                        }
                    }));
                }
                for h in hs { let _ = h.join(); }
            });
            // carveout + 언맵 — alloc_host(대형)는 i8 coopmatLoad 경로에서
            // 데이터 붕괴 실측 (미니 재현: 소형 carveout ★, 대형 host ✗).
            let wspbuf = {
                let mut b = ctx.alloc(no * n_sub * 4)?;
                unsafe { std::ptr::copy_nonoverlapping(wsp.as_ptr(), b.ptr as *mut f32, no * n_sub) };
                ctx.unmap(&mut b)?;
                b
            };
            let wsmbuf = {
                let mut b = ctx.alloc(no * n_sub * 4)?;
                unsafe { std::ptr::copy_nonoverlapping(wsm.as_ptr(), b.ptr as *mut f32, no * n_sub) };
                ctx.unmap(&mut b)?;
                b
            };
            // v2 (mlx식): 정확 디양자화 후 행별 재양자 — w8 덮어씀 + wsr.
            // w_deq = d·sc·q − dm·m (q = w_int). v1 wsp/wsm은 이미 계산됨.
            {
                let mut wsr_v = vec![0f32; no];
                for o in 0..no {
                    let mut mx = 0f32;
                    for b in 0..n_sub {
                        let s = wsp[o * n_sub + b];
                        let mn = wsm[o * n_sub + b];
                        let mut isum_min = 0i64;
                        for e in 0..32 {
                            isum_min += w8[o * ni + b * 32 + e] as i64;
                        }
                        // 값 범위: max|d·sc·q − dm·m| 근사 — 실제 최댓값은 원소별 계산
                        let hi = (s * 47.0).abs() + mn.abs();
                        let lo = mn.abs();
                        mx = mx.max(hi.max(lo));
                    }
                    // 정확 최댓값: 원소별 (느려도 init 1회)
                    mx = 0f32;
                    for b in 0..n_sub {
                        let s = wsp[o * n_sub + b];
                        let mn = wsm[o * n_sub + b];
                        for e in 0..32 {
                            let v = s * w8[o * ni + b * 32 + e] as f32 - mn;
                            mx = mx.max(v.abs());
                        }
                    }
                    let d = mx / 127.0f32;
                    let id = if d > 0.0 { 1.0f32 / d } else { 0.0f32 };
                    wsr_v[o] = d;
                    for b in 0..n_sub {
                        let s = wsp[o * n_sub + b];
                        let mn = wsm[o * n_sub + b];
                        for e in 0..32 {
                            let v = s * w8[o * ni + b * 32 + e] as f32 - mn;
                            w8[o * ni + b * 32 + e] = (v * id).round().clamp(-127.0, 127.0) as i8;
                        }
                    }
                }
                let mut b = ctx.alloc(no * 4)?;
                unsafe { std::ptr::copy_nonoverlapping(wsr_v.as_ptr(), b.ptr as *mut f32, no) };
                ctx.unmap(&mut b)?;
                wsr_map.insert(name.clone(), b);
            }
            let wbuf = {
                let mut b = ctx.alloc(no * ni)?;
                unsafe { std::ptr::copy_nonoverlapping(w8.as_ptr() as *const u8, b.ptr, no * ni) };
                ctx.unmap(&mut b)?;
                b
            };
            i8w.insert(name, I8W { w: wbuf, wsp: wspbuf, wsm: wsmbuf, n_out: no, n_in: ni });
        }
        let n_max = hp.n_ff.max(n);
        let n_sub_max = n_max / 32;
        let b8 = ctx.alloc_host(T_MAX * n_max)?;
        let ydb = ctx.alloc_host(T_MAX * n_sub_max * 4)?;
        let qsb = ctx.alloc_host(T_MAX * n_sub_max * 4)?;
        let wg_max = i8_wg_max(&i8w).max(640);
        let ishs = ctx.alloc_host(wg_max * 256 * 4)?;
        let faccs = ctx.alloc_host(wg_max * 256 * 4)?;
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
            b_lg,
            b_lg_t,
            b_am,
            pipes: HashMap::new(),
            split_ctr: 0,
            mtp_on,
            n_vocab,
            m_e,
            m_cat,
            m_xq2,
            m_cur,
            m_xq,
            m_h,
            m_kv_k: mkk,
            m_kv_v: mvv,
            snap_gdn: vec![Vec::new(); n_recr * n_seqs],
            snap_conv: vec![Vec::new(); n_recr * n_seqs],
            i8w,
            wsr: wsr_map,
            b8,
            ydb,
            qsb,
            ishs,
            faccs,
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
    /// t≥2 + q5_K는 coopmat 128행 타일 (plans/20 — f16 스테이징 MMA,
    /// HIP 기본 WMMA와 동일 정확도 클래스 maxrel ~4.9e-4, argmax 안정).
    /// LLM170_VK_NOTILE=1이면 항상 gemv3 정밀 경로.
    fn gemv(&mut self, xq: vk::Buffer, wkey: &str, out: vk::Buffer, t: usize) -> Result<(), String> {
        let (wbufs, ty, ni, no) = self.w.get(wkey).cloned().ok_or(format!("가중치 없음: {wkey}"))?;
        if t >= 2 && ty == 13 && std::env::var_os("LLM170_VK_NOTILE").is_none() {
            let xq_w = ni / 4 + ni / 32 + ni / 16;
            let mut binds: Vec<vk::Buffer> = wbufs.iter().map(|b| b.buf).collect();
            while binds.len() < 8 {
                binds.push(self.dummy.buf);
            }
            binds.push(xq);
            binds.push(out);
            let gx = (no as u32 + 127) / 128;
            for tb in (0..t).step_by(64) {
                let nt = (t - tb).min(64) as u32;
                let push = Self::push_u32s(&[ni as u32, no as u32, xq_w as u32, nt]);
                self.run_pipe("tile128", TILE128_SPV, 10, 16, &binds, &push, gx, 1, 1)?;
            }
            return Ok(());
        }
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

    /// v2 (mlx식) — per-row 스케일: quant_b8v2 + gemm_i8v2. LLM170_VK_I8=2.
    fn gemm_i8v2(&mut self, wkey: &str, out: vk::Buffer, t: usize) -> Result<(), String> {
        let e = self.i8w.get(wkey).ok_or(format!("i8w 없음: {wkey}"))?;
        let (wbuf, ni, no) = (e.w.clone(), e.n_in, e.n_out);
        let wsr = self.wsr.get(wkey).cloned().ok_or("wsr 없음")?;
        // quant_b8v2: b_xn → b8 + ydr (ydb 첫 t 슬롯 재사용)
        let push = Self::push_u32s(&[ni as u32, t as u32]);
        self.run_pipe("quant_b8v2", QUANT_B8V2_SPV, 3, 8,
            &[self.b_xn.buf, self.b8.buf, self.ydb.buf], &push,
            1, t as u32, 1)?;
        let mut binds: Vec<vk::Buffer> = vec![wbuf.buf; 1];
        while binds.len() < 8 { binds.push(self.dummy.buf); }
        binds.push(self.b8.buf);
        binds.push(out);
        binds.push(wsr.buf);
        binds.push(self.ydb.buf);
        let push2 = Self::push_u32s(&[ni as u32, no as u32, t as u32]);
        self.run_pipe("gemm_i8v2", GEMM_I8V2_SPV, 12, 12, &binds, &push2,
            (no as u32 + 15) / 16, 1, 1)
    }

    /// gemm_i8 (plans/23) — q5_K 사전 언패분으로 t행 GEMM.
    fn gemm_i8(&mut self, wkey: &str, out: vk::Buffer, t: usize) -> Result<(), String> {
        let e = self.i8w.get(wkey).ok_or(format!("i8w 없음: {wkey}"))?;
        let (wbuf, wspbuf, wsmbuf, ni, no) = (e.w.clone(), e.wsp.clone(), e.wsm.clone(), e.n_in, e.n_out);
        let n_sub = ni / 32;
        let mut binds: Vec<vk::Buffer> = vec![wbuf.buf; 1];
        while binds.len() < 8 {
            binds.push(self.dummy.buf);
        }
        binds.push(self.b8.buf);
        binds.push(out);
        binds.push(wspbuf.buf);
        binds.push(wsmbuf.buf);
        binds.push(self.ydb.buf);
        binds.push(self.qsb.buf);
        binds.push(self.ishs.buf);
        binds.push(self.faccs.buf);
        let push = Self::push_u32s(&[ni as u32, no as u32, t as u32, n_sub as u32]);
        self.run_pipe("gemm_i8", GEMM_I8_SPV, 16, 16, &binds, &push,
            (no as u32 + 15) / 16, 1, 1)
    }

    /// 단계 공유 GEMV — q5_K·t≥2는 gemm_i8(LLM170_VK_NOI8 킬스위치),
    /// 나머지는 기존 xq+gemv3. xq 양자화는 호출부가 이미 수행.
    fn gemv_stage(&mut self, n: usize, t: usize, jobs: &[(String, vk::Buffer, vk::Buffer)]) -> Result<(), String> {
        let i8_on = t >= 2 && std::env::var_os("LLM170_VK_NOI8").is_none();
        let any_i8 = i8_on && jobs.iter().any(|(k, _, _)| self.i8w.contains_key(k));
        let _ = &any_i8;
        let v2 = std::env::var("LLM170_VK_I8").map(|v| v == "2").unwrap_or(false);
        for (k, xq, out) in jobs {
            if any_i8 && self.i8w.contains_key(k) {
                if v2 {
                    self.gemm_i8v2(k, *out, t)?;
                } else {
                    self.gemm_i8(k, *out, t)?;
                }
            } else {
                self.gemv(*xq, k, *out, t)?;
            }
        }
        Ok(())
    }

    /// quant_b8: src f32 [t][n] → b8/yd/qs (gemm_i8 입력).
    fn quant_b8(&mut self, src: vk::Buffer, n: usize, t: usize) -> Result<(), String> {
        let n_sub = n / 32;
        let push = Self::push_u32s(&[n as u32, t as u32, n_sub as u32]);
        self.run_pipe("quant_b8", QUANT_B8_SPV, 4, 12,
            &[src, self.b8.buf, self.ydb.buf, self.qsb.buf], &push,
            (n / 32 + 63) as u32 / 64, t as u32, 1)
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
        let vk_t0 = std::time::Instant::now();
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
                if std::env::var_os("LLM170_VKD_TRACE").is_some() && il < 2 {
                    self.ctx.end_batch_wait().ok();
        if std::env::var_os("LLM170_VK_PROF").is_some() { eprintln!("[vkprof] layers+head: {:.1}ms (pos {})", vk_t0.elapsed().as_secs_f32()*1e3, pos); }
                    self.ctx.begin_batch().ok();
                    let mut v = vec![0f32; n];
                    unsafe { std::ptr::copy_nonoverlapping(self.b_gout.ptr as *const f32, v.as_mut_ptr(), n) };
                    let sum: f64 = v.iter().map(|&x| x as f64).sum();
                    let s = |b: &VkBuf, len: usize| -> f64 {
                        let mut x = vec![0f32; len];
                        unsafe { std::ptr::copy_nonoverlapping(b.ptr as *const f32, x.as_mut_ptr(), len) };
                        x.iter().map(|&q| q as f64).sum()
                    };
                    let srow = |b: &VkBuf, r: usize, len: usize| -> f64 {
                        let mut x = vec![0f32; len];
                        unsafe { std::ptr::copy_nonoverlapping(b.ptr.add(r * len * 4) as *const f32, x.as_mut_ptr(), len) };
                        x.iter().map(|&q| q as f64).sum()
                    };
                    eprintln!("#  G0 il={il} gout={sum:.6} | xs0={:.4} xs1={:.4} xn0={:.4} xn1={:.4} gqkv0={:.4} gconv0={:.4} gq0={:.4} go0={:.4} ggated0={:.4}",
                        srow(&self.b_xs, 0, 64), srow(&self.b_xs, 1, 64),
                        srow(&self.b_xn, 0, 64), srow(&self.b_xn, 1, 64),
                        s(&self.b_gqkv, self.conv_ch.min(64)),
                        s(&self.b_gconv, self.conv_ch.min(64)), s(&self.b_gq, self.k_len.min(64)),
                        s(&self.b_go, self.v_len.min(64)), s(&self.b_ggated, self.d_inner.min(64)));
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
                    let push = Self::push_u32s(&[pos as u32, n_head as u32, n_kv as u32, hd as u32, self.ctx_len as u32]);
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
        if std::env::var_os("LLM170_VKD_TRACE").is_some() {
            let s = |b: &VkBuf, len: usize| -> f64 {
                let mut x = vec![0f32; len];
                unsafe { std::ptr::copy_nonoverlapping(b.ptr as *const f32, x.as_mut_ptr(), len) };
                x.iter().map(|&q| q as f64).sum()
            };
            eprintln!("#  ST state seq={seq} conv0={:.6} gdn0={:.6} kvk3r0={:.6} kvv3r0={:.6} xsL={:.6}",
                s(&self.st_conv[0][seq], 30720.min(self.conv_ch * 3)),
                s(&self.st_gdn[0][seq], 4096),
                s(&self.kv_k[0][seq], 1024), s(&self.kv_v[0][seq], 1024),
                s(&self.b_xs, 64));
        }
        // ── head: output_norm → quant → gemv(output) → 다운로드
        {
            let (xs, xn) = (self.b_xs.clone(), self.b_xn.clone());
            self.rms(xs.buf, "output_norm", xn.buf, n, 1)?;
        }
        self.quant(self.b_xn.buf, self.b_xq_n.buf, n, 1)?;
        if !noba { self.ctx.begin_batch()?; };
        self.gemv(self.b_xq_n.buf, "output.weight", self.b_lg.buf, 1)?;
        if !noba { self.ctx.end_batch_wait()?; } else { self.ctx.flush2()?; };
        let mut logits = vec![0f32; self.n_vocab];
        unsafe { std::ptr::copy_nonoverlapping(self.b_lg.ptr as *const f32, logits.as_mut_ptr(), self.n_vocab) };
        Ok(logits)
    }

    /// t행 배치 스텝 (plans/20) — 가중 1회 판독 분할 상각. 행별 산술은
    /// step()과 비트 동일(gemv3 행별 lane 축산·AR 내부 순차·conv 이력 판독).
    /// all_logits=true: 전 행 head 로짓 [t][n_vocab] (verify용 — b_lg_t).
    /// 아니면 마지막 행만 (b_lg). emb는 [t][n_embd].
    pub fn step_batch(&mut self, seq: usize, pos0: usize, emb: &[f32], all_logits: bool) -> Result<Vec<f32>, String> {
        let noba = std::env::var_os("LLM170_VK_NOBATCH").is_some();
        let n = self.n_embd;
        let t = emb.len() / n;
        if t == 0 || emb.len() != t * n || t > T_MAX {
            return Err(format!("step_batch t={t} (1..={T_MAX})"));
        }
        let (dt_rank, d_state, d_inner) = (self.dt_rank, self.d_state, self.d_inner);
        let (n_head, n_kv, hd, n_rot) = (self.n_head, self.n_kv, self.hd, self.n_rot);
        let conv_ch = self.conv_ch;
        let k_len = self.k_len;
        let v_len = self.v_len;
        unsafe {
            std::ptr::copy_nonoverlapping(emb.as_ptr(), self.b_xs.ptr as *mut f32, t * n);
        }
        if std::env::var_os("LLM170_VKD_TRACE").is_some() {
            let mut x = vec![0f32; 64];
            unsafe { std::ptr::copy_nonoverlapping(self.b_xs.ptr as *const f32, x.as_mut_ptr(), 64) };
            let s0: f64 = x.iter().map(|&v| v as f64).sum();
            eprintln!("#  SB upload t={t} xs0={s0:.4}");
        }
        if !noba { self.ctx.begin_batch()?; };
        let mut recr_idx = 0usize;
        let mut full_idx = 0usize;
        for il in 0..self.n_layer {
            {
                let (xs, xn) = (self.b_xs.clone(), self.b_xn.clone());
                self.rms(xs.buf, &format!("blk.{il}.attn_norm"), xn.buf, n, t)?;
            }
            self.quant(self.b_xn.buf, self.b_xq_n.buf, n, t)?;
            if std::env::var_os("LLM170_VKD_TRACE").is_some() && il == 0 {
                self.ctx.end_batch_wait().ok(); self.ctx.begin_batch().ok();
                let x = vec![0f32; 64];
                let s0: f64 = unsafe { std::slice::from_raw_parts(self.b_xs.ptr as *const f32, 64) }.iter().map(|&v| v as f64).sum();
                eprintln!("#  SB post-rms xs0={s0:.4}");
            }
            if self.is_recr[il] {
                if t >= 2 && std::env::var_os("LLM170_VK_NOI8").is_none()
                    && (self.i8w.contains_key(&format!("blk.{il}.attn_qkv.weight"))
                        || self.i8w.contains_key(&format!("blk.{il}.attn_gate.weight"))) {
                    self.quant_b8(self.b_xn.buf, n, t)?;
                }
                self.gemv_stage(n, t, &[
                    (format!("blk.{il}.attn_qkv.weight"), self.b_xq_n.buf, self.b_gqkv.buf),
                    (format!("blk.{il}.attn_gate.weight"), self.b_xq_n.buf, self.b_gz.buf),
                    (format!("blk.{il}.ssm_beta.weight"), self.b_xq_n.buf, self.b_gb.buf),
                    (format!("blk.{il}.ssm_alpha.weight"), self.b_xq_n.buf, self.b_ga.buf),
                ])?;
                if std::env::var_os("LLM170_VKD_TRACE").is_some() && il == 0 {
                    self.ctx.end_batch_wait().ok(); self.ctx.begin_batch().ok();
                    let s0: f64 = unsafe { std::slice::from_raw_parts(self.b_xs.ptr as *const f32, 64) }.iter().map(|&v| v as f64).sum();
                    let mut g = vec![0f32; 8];
                    unsafe { std::ptr::copy_nonoverlapping(self.b_gqkv.ptr as *const f32, g.as_mut_ptr(), 8) };
                    let gq: f64 = unsafe { std::slice::from_raw_parts(self.b_gqkv.ptr as *const f32, 128) }.iter().map(|&v| v as f64).sum();
                    let d0 = format!("{:?}", g);
                    let gz8: Vec<f32> = unsafe { std::slice::from_raw_parts(self.b_gz.ptr as *const f32, 8) }.to_vec();
                    let gb8: Vec<f32> = unsafe { std::slice::from_raw_parts(self.b_gb.ptr as *const f32, 4) }.to_vec();
                    eprintln!("#  stage2 gz={:?} gb={:?}", gz8, gb8);
                    let mut b8v = [0i8; 16];
                    unsafe { std::ptr::copy_nonoverlapping(self.b8.ptr as *const i8, b8v.as_mut_ptr(), 16) };
                    let b8r1: Vec<i8> = unsafe { std::slice::from_raw_parts(self.b8.ptr as *const i8, 32) }[16..].to_vec();
                    let mut ydv = [0f32; 4];
                    unsafe { std::ptr::copy_nonoverlapping(self.ydb.ptr as *const f32, ydv.as_mut_ptr(), 4) };
                    let mut qsv = [0i32; 4];
                    unsafe { std::ptr::copy_nonoverlapping(self.qsb.ptr as *const i32, qsv.as_mut_ptr(), 4) };
                    eprintln!("#  SB post-gemv4 xs0={s0:.4} gqkv0={gq:.4} first8={d0} b8={:?} b8tail={:?} yd={:?} qs={:?}", b8v.to_vec(), b8r1, ydv.to_vec(), qsv.to_vec());
                }
                // conv — gy=t (이력은 qkv에서 판독, t>1은 링을 conv_state가 갱신)
                {
                    let cw = self.consts.get(&format!("blk.{il}.conv_w")).cloned().ok_or("conv_w")?;
                    let push = Self::push_u32s(&[conv_ch as u32, self.conv_k as u32, t as u32]);
                    self.run_pipe("gdn_conv", GDN_CONV_SPV, 4, 12,
                        &[self.b_gqkv.buf, cw.buf, self.st_conv[recr_idx][seq].buf, self.b_gconv.buf],
                        &push, conv_ch.div_ceil(64) as u32, t as u32, 1)?;
                    if t > 1 {
                        let push = Self::push_u32s(&[conv_ch as u32, self.conv_k as u32, t as u32]);
                        self.run_pipe("gdn_conv_state", GDN_CONV_STATE_SPV, 2, 12,
                            &[self.b_gqkv.buf, self.st_conv[recr_idx][seq].buf],
                            &push, conv_ch.div_ceil(64) as u32, 1, 1)?;
                    }
                }
                if std::env::var_os("LLM170_VKD_TRACE").is_some() && il == 0 {
                    self.ctx.end_batch_wait().ok(); self.ctx.begin_batch().ok();
                    let s0: f64 = unsafe { std::slice::from_raw_parts(self.b_xs.ptr as *const f32, 64) }.iter().map(|&v| v as f64).sum();
                    eprintln!("#  SB post-conv xs0={s0:.4}");
                }
                // split3 — flat total*t
                {
                    let total = 2 * k_len + v_len;
                    let push = Self::push_u32s(&[k_len as u32, k_len as u32, v_len as u32]);
                    self.run_pipe("split3", SPLIT3_SPV, 4, 12,
                        &[self.b_gconv.buf, self.b_gq.buf, self.b_gk.buf, self.b_gv.buf],
                        &push, (total * t).div_ceil(64) as u32, 1, 1)?;
                }
                if std::env::var_os("LLM170_VKD_TRACE").is_some() && il == 0 {
                    self.ctx.end_batch_wait().ok(); self.ctx.begin_batch().ok();
                    let s0: f64 = unsafe { std::slice::from_raw_parts(self.b_xs.ptr as *const f32, 64) }.iter().map(|&v| v as f64).sum();
                    eprintln!("#  SB post-split3 xs0={s0:.4}");
                }
                // l2 — grid (2*ng, t)
                {
                    let mut push = self.eps.to_le_bytes().to_vec();
                    push.extend(Self::push_u32s(&[d_state as u32, self.n_group as u32]));
                    self.run_pipe("l2", L2_SPV, 2, 12,
                        &[self.b_gq.buf, self.b_gk.buf], &push, (2 * self.n_group) as u32, t as u32, 1)?;
                }
                if std::env::var_os("LLM170_VKD_TRACE").is_some() && il == 0 {
                    self.ctx.end_batch_wait().ok(); self.ctx.begin_batch().ok();
                    let s0: f64 = unsafe { std::slice::from_raw_parts(self.b_xs.ptr as *const f32, 64) }.iter().map(|&v| v as f64).sum();
                    eprintln!("#  SB post-l2 xs0={s0:.4}");
                }
                // beta_g — n_h = dt_rank*t
                {
                    let dtb = self.consts.get(&format!("blk.{il}.dt_bias")).cloned().ok_or("dtb")?;
                    let ssa = self.consts.get(&format!("blk.{il}.ssm_a")).cloned().ok_or("ssa")?;
                    let push = Self::push_u32s(&[(dt_rank * t) as u32, dt_rank as u32]);
                    self.run_pipe("beta_g", BETA_G_SPV, 5, 8,
                        &[self.b_gb.buf, self.b_ga.buf, dtb.buf, ssa.buf, self.b_gbg.buf],
                        &push, (dt_rank * t).div_ceil(64) as u32, 1, 1)?;
                }
                if std::env::var_os("LLM170_VKD_TRACE").is_some() && il == 0 {
                    self.ctx.end_batch_wait().ok(); self.ctx.begin_batch().ok();
                    let s0: f64 = unsafe { std::slice::from_raw_parts(self.b_xs.ptr as *const f32, 64) }.iter().map(|&v| v as f64).sum();
                    eprintln!("#  SB post-betag xs0={s0:.4}");
                }
                // AR — PC.t 내부 순차
                {
                    let scale = 1.0f32 / (d_state as f32).sqrt();
                    let mut push = Self::push_u32s(&[d_state as u32, k_len as u32, v_len as u32, dt_rank as u32, self.n_group as u32]);
                    push.extend_from_slice(&scale.to_le_bytes());
                    push.extend_from_slice(&(t as u32).to_le_bytes());
                    self.run_pipe("gdn_ar", GDN_AR_SPV, 6, 28,
                        &[self.st_gdn[recr_idx][seq].buf, self.b_gq.buf, self.b_gk.buf,
                          self.b_gv.buf, self.b_gbg.buf, self.b_go.buf],
                        &push, dt_rank as u32, d_state as u32, 1)?;
                }
                if std::env::var_os("LLM170_VKD_TRACE").is_some() && il == 0 {
                    self.ctx.end_batch_wait().ok(); self.ctx.begin_batch().ok();
                    let s0: f64 = unsafe { std::slice::from_raw_parts(self.b_xs.ptr as *const f32, 64) }.iter().map(|&v| v as f64).sum();
                    eprintln!("#  SB post-ar xs0={s0:.4}");
                }
                // norm_gated — grid (dt_rank, t)
                {
                    let sn = self.consts.get(&format!("blk.{il}.ssm_norm")).cloned().ok_or("sn")?;
                    let mut push = self.eps.to_le_bytes().to_vec();
                    push.extend(Self::push_u32s(&[d_state as u32, dt_rank as u32]));
                    self.run_pipe("norm_gated", NORM_GATED_SPV, 4, 12,
                        &[self.b_go.buf, self.b_gz.buf, sn.buf, self.b_ggated.buf],
                        &push, dt_rank as u32, t as u32, 1)?;
                }
                if std::env::var_os("LLM170_VKD_TRACE").is_some() && il == 0 {
                    self.ctx.end_batch_wait().ok(); self.ctx.begin_batch().ok();
                    let s0: f64 = unsafe { std::slice::from_raw_parts(self.b_xs.ptr as *const f32, 64) }.iter().map(|&v| v as f64).sum();
                    eprintln!("#  SB post-normgated xs0={s0:.4}");
                }
                self.quant(self.b_ggated.buf, self.b_xq_g.buf, d_inner, t)?;
                if std::env::var_os("LLM170_VKD_TRACE").is_some() && il == 0 {
                    self.ctx.end_batch_wait().ok(); self.ctx.begin_batch().ok();
                    let s0: f64 = unsafe { std::slice::from_raw_parts(self.b_xs.ptr as *const f32, 64) }.iter().map(|&v| v as f64).sum();
                    eprintln!("#  SB post-quant2 xs0={s0:.4}");
                }
                self.gemv(self.b_xq_g.buf, &format!("blk.{il}.ssm_out.weight"), self.b_gout.buf, t)?;
                recr_idx += 1;
            } else {
                if t >= 2 && std::env::var_os("LLM170_VK_NOI8").is_none()
                    && self.i8w.contains_key(&format!("blk.{il}.attn_q.weight")) {
                    self.quant_b8(self.b_xn.buf, n, t)?;
                }
                self.gemv_stage(n, t, &[
                    (format!("blk.{il}.attn_q.weight"), self.b_xq_n.buf, self.b_aq.buf),
                    (format!("blk.{il}.attn_k.weight"), self.b_xq_n.buf, self.b_ak.buf),
                    (format!("blk.{il}.attn_v.weight"), self.b_xq_n.buf, self.b_av.buf),
                ])?;
                // qk_rope — grid (nh+nk, t), pos = pos0+행
                {
                    let qn = self.consts.get(&format!("blk.{il}.attn_q_norm")).cloned().ok_or("qn")?;
                    let kn = self.consts.get(&format!("blk.{il}.attn_k_norm")).cloned().ok_or("kn")?;
                    let cs = self.consts.get("cs").cloned().ok_or("cs")?;
                    let mut push = self.eps.to_le_bytes().to_vec();
                    push.extend_from_slice(&self.kq_scale.to_le_bytes());
                    push.extend(Self::push_u32s(&[pos0 as u32, n_head as u32, n_kv as u32, hd as u32, n_rot as u32]));
                    self.run_pipe("qk_rope", QK_ROPE_SPV, 5, 28,
                        &[self.b_aq.buf, self.b_ak.buf, qn.buf, kn.buf, cs.buf],
                        &push, (n_head + n_kv) as u32, t as u32, 1)?;
                }
                // kv append — grid (n/64, t), pos0 기준 행별 위치
                {
                    let push = Self::push_u32s(&[(n_kv * hd) as u32, pos0 as u32]);
                    self.run_pipe("kv_app", KV_APPEND_SPV, 2, 8,
                        &[self.b_ak.buf, self.kv_k[full_idx][seq].buf], &push,
                        (n_kv * hd).div_ceil(64) as u32, t as u32, 1)?;
                    self.run_pipe("kv_app", KV_APPEND_SPV, 2, 8,
                        &[self.b_av.buf, self.kv_v[full_idx][seq].buf], &push,
                        (n_kv * hd).div_ceil(64) as u32, t as u32, 1)?;
                }
                // flash — grid (t, n_head), np = pos0+행+1
                {
                    let mask = self.consts.get("mask").cloned().ok_or("mask")?;
                    let push = Self::push_u32s(&[pos0 as u32, n_head as u32, n_kv as u32, hd as u32, self.ctx_len as u32]);
                    self.run_pipe("qsa_flash", QSA_FLASH_SPV, 5, 24,
                        &[self.b_aq.buf, self.kv_k[full_idx][seq].buf, self.kv_v[full_idx][seq].buf, mask.buf, self.b_aout.buf],
                        &push, t as u32, n_head as u32, 1)?;
                }
                self.quant(self.b_aout.buf, self.b_xq_g.buf, n_head * hd, t)?;
                self.gemv(self.b_xq_g.buf, &format!("blk.{il}.attn_output.weight"), self.b_gout.buf, t)?;
                full_idx += 1;
            }
            // 잔차 — flat n*t
            self.axpy(self.b_xs.buf, self.b_gout.buf, n * t)?;
            // FFN
            {
                let (xs, xn) = (self.b_xs.clone(), self.b_xn.clone());
                self.rms(xs.buf, &format!("blk.{il}.post_norm"), xn.buf, n, t)?;
            }
            self.quant(self.b_xn.buf, self.b_xq_n.buf, n, t)?;
            if t >= 2 && std::env::var_os("LLM170_VK_NOI8").is_none()
                && self.i8w.contains_key(&format!("blk.{il}.ffn_gate.weight")) {
                self.quant_b8(self.b_xn.buf, n, t)?;
            }
            self.gemv_stage(n, t, &[
                (format!("blk.{il}.ffn_gate.weight"), self.b_xq_n.buf, self.b_fgate.buf),
                (format!("blk.{il}.ffn_up.weight"), self.b_xq_n.buf, self.b_fup.buf),
            ])?;
            self.silu_mul(self.b_fgate.buf, self.b_fup.buf, self.b_fglu.buf, self.n_ff * t)?;
            self.quant(self.b_fglu.buf, self.b_xq_f.buf, self.n_ff, t)?;
            self.gemv(self.b_xq_f.buf, &format!("blk.{il}.ffn_down.weight"), self.b_fdown.buf, t)?;
            self.axpy(self.b_xs.buf, self.b_fdown.buf, n * t)?;
        }
        if !noba { self.ctx.end_batch_wait()?; } else { self.ctx.flush2()?; };
        if std::env::var_os("LLM170_VKD_TRACE").is_some() {
            let s = |b: &VkBuf, len: usize| -> f64 {
                let mut x = vec![0f32; len];
                unsafe { std::ptr::copy_nonoverlapping(b.ptr as *const f32, x.as_mut_ptr(), len) };
                x.iter().map(|&q| q as f64).sum()
            };
            eprintln!("#  SB state seq={seq} conv0={:.6} gdn0={:.6} kvk3r0={:.6} kvv3r0={:.6} xsL={:.6}",
                s(&self.st_conv[0][seq], 30720.min(self.conv_ch * 3)),
                s(&self.st_gdn[0][seq], 4096),
                s(&self.kv_k[0][seq], 1024), s(&self.kv_v[0][seq], 1024),
                s(&self.b_xs, 64));
        }
        // head
        if all_logits {
            self.rms(self.b_xs.buf, "output_norm", self.b_xn.buf, n, t)?;
            self.quant(self.b_xn.buf, self.b_xq_n.buf, n, t)?;
            if !noba { self.ctx.begin_batch()?; };
            self.gemv(self.b_xq_n.buf, "output.weight", self.b_lg_t.buf, t)?;
            if !noba { self.ctx.end_batch_wait()?; } else { self.ctx.flush2()?; };
            let mut out = vec![0f32; t * self.n_vocab];
            unsafe { std::ptr::copy_nonoverlapping(self.b_lg_t.ptr as *const f32, out.as_mut_ptr(), t * self.n_vocab) };
            Ok(out)
        } else {
            // 마지막 행 — GPU 유휴 상태 호스트 복사 후 t=1 head
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.b_xs.ptr.add((t - 1) * n * 4) as *const f32,
                    self.m_e.ptr as *mut f32, n);
            }
            self.rms(self.m_e.buf, "output_norm", self.b_xn.buf, n, 1)?;
            self.quant(self.b_xn.buf, self.b_xq_n.buf, n, 1)?;
            if !noba { self.ctx.begin_batch()?; };
            self.gemv(self.b_xq_n.buf, "output.weight", self.b_lg.buf, 1)?;
            if !noba { self.ctx.end_batch_wait()?; } else { self.ctx.flush2()?; };
            let mut logits = vec![0f32; self.n_vocab];
            unsafe { std::ptr::copy_nonoverlapping(self.b_lg.ptr as *const f32, logits.as_mut_ptr(), self.n_vocab) };
            Ok(logits)
        }
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

    // ══ Phase A: MTP·spec·np — 전부 t=1 검증 커널 재사용 (plans/19) ══

    /// b_xs 최종 hidden [n] 판독 (step 완료 후 — GPU 유휴 보장).
    fn hidden_row(&self) -> Vec<f32> {
        let n = self.n_embd;
        let mut v = vec![0f32; n];
        unsafe { std::ptr::copy_nonoverlapping(self.b_xs.ptr as *const f32, v.as_mut_ptr(), n) };
        v
    }

    /// copy_off: src[0..n) → dst[dst_off..).
    fn copy_off(&mut self, src: vk::Buffer, dst: vk::Buffer, n: usize, dst_off: usize) -> Result<(), String> {
        let push = Self::push_u32s(&[n as u32, dst_off as u32]);
        self.run_pipe("copy_off", COPY_OFF_SPV, 2, 8,
            &[src, dst], &push, n.div_ceil(256) as u32, 1, 1)
    }

    /// shared head — 정규화 입력(m_e) → 로짓 argmax (b_lg 매핑 판독 + CPU greedy).
    fn head_argmax(&mut self) -> Result<u32, String> {
        let n = self.n_embd;
        self.quant(self.m_e.buf, self.m_xq.buf, n, 1)?;
        self.ctx.begin_batch()?;
        self.gemv(self.m_xq.buf, "output.weight", self.b_lg.buf, 1)?;
        self.ctx.end_batch_wait()?;
        let lgr: &[f32] = unsafe { std::slice::from_raw_parts(self.b_lg.ptr as *const f32, self.n_vocab) };
        Ok(llm170_core::matmul::greedy_from(lgr))
    }

    /// MTP (blk.64) 1스텝 — rawhip mtp_step_g 산술 미러 (t=1 커널 재사용).
    /// h_from_cur=true: h 입력을 내부 m_cur에서 (체인). 반환: with_head면 argmax.
    fn mtp_step_g(
        &mut self,
        seq: usize,
        tok_emb: &[f32],
        h_from_cur: bool,
        h_host: &[f32],
        pos: usize,
        with_head: bool,
    ) -> Result<Option<u32>, String> {
        if !self.mtp_on {
            return Err("mtp_step_gpu: MTP 미로드".into());
        }
        let n = self.n_embd;
        let (n_head, n_kv, hd, n_rot) = (self.n_head, self.n_kv, self.hd, self.n_rot);
        debug_assert_eq!(tok_emb.len(), n);
        unsafe {
            std::ptr::copy_nonoverlapping(tok_emb.as_ptr(), self.m_e.ptr as *mut f32, n);
            if !h_from_cur {
                if h_host.len() >= n {
                    std::ptr::copy_nonoverlapping(h_host.as_ptr(), self.m_h.ptr as *mut f32, n);
                } else {
                    std::ptr::write_bytes(self.m_h.ptr as *mut f32, 0, n);
                }
            }
        }
        let h_buf = if h_from_cur { self.m_cur.buf } else { self.m_h.buf };
        let noba = std::env::var_os("LLM170_VK_NOBATCH").is_some();
        if !noba { self.ctx.begin_batch()?; }
        // enorm → cat[0..n] ‖ hnorm → cat[n..2n]
        let en = self.consts.get("blk.64.nextn.enorm").cloned().ok_or("enorm")?;
        let hn = self.consts.get("blk.64.nextn.hnorm").cloned().ok_or("hnorm")?;
        self.rms(self.m_e.buf.clone(), "blk.64.nextn.enorm", self.m_cat.buf, n, 1)?;
        self.rms(h_buf, "blk.64.nextn.hnorm", self.b_xn.buf, n, 1)?;
        self.copy_off(self.b_xn.buf, self.m_cat.buf, n, n)?;
        let _ = (en, hn);
        // eh_proj [2n → n]
        self.quant(self.m_cat.buf, self.m_xq2.buf, 2 * n, 1)?;
        self.gemv(self.m_xq2.buf, "blk.64.nextn.eh_proj.weight", self.m_cur.buf, 1)?;
        // attn_norm → q/k/v
        self.rms(self.m_cur.buf, "blk.64.attn_norm", self.m_e.buf, n, 1)?;
        self.quant(self.m_e.buf, self.m_xq.buf, n, 1)?;
        self.gemv(self.m_xq.buf, "blk.64.attn_q.weight", self.b_aq.buf, 1)?;
        self.gemv(self.m_xq.buf, "blk.64.attn_k.weight", self.b_ak.buf, 1)?;
        self.gemv(self.m_xq.buf, "blk.64.attn_v.weight", self.b_av.buf, 1)?;
        // q/k norm+rope (t=1 — step과 동일 디스패치)
        {
            let qn = self.consts.get("blk.64.attn_q_norm").cloned().ok_or("qn")?;
            let kn = self.consts.get("blk.64.attn_k_norm").cloned().ok_or("kn")?;
            let cs = self.consts.get("cs").cloned().ok_or("cs")?;
            let mut push = self.eps.to_le_bytes().to_vec();
            push.extend_from_slice(&self.kq_scale.to_le_bytes());
            push.extend(Self::push_u32s(&[pos as u32, n_head as u32, n_kv as u32, hd as u32, n_rot as u32]));
            self.run_pipe("qk_rope", QK_ROPE_SPV, 5, 28,
                &[self.b_aq.buf, self.b_ak.buf, qn.buf, kn.buf, cs.buf],
                &push, (n_head + n_kv) as u32, 1, 1)?;
        }
        // MTP 자체 KV 적립 + flash (np = pos+1)
        {
            let push = Self::push_u32s(&[(n_kv * hd) as u32, pos as u32]);
            self.run_pipe("kv_app", KV_APPEND_SPV, 2, 8,
                &[self.b_ak.buf, self.m_kv_k[seq].buf], &push,
                (n_kv * hd).div_ceil(64) as u32, 1, 1)?;
            self.run_pipe("kv_app", KV_APPEND_SPV, 2, 8,
                &[self.b_av.buf, self.m_kv_v[seq].buf], &push,
                (n_kv * hd).div_ceil(64) as u32, 1, 1)?;
        }
        {
            let mask = self.consts.get("mask").cloned().ok_or("mask")?;
            let push = Self::push_u32s(&[pos as u32, n_head as u32, n_kv as u32, hd as u32, self.ctx_len as u32]);
            self.run_pipe("qsa_flash", QSA_FLASH_SPV, 5, 24,
                &[self.b_aq.buf, self.m_kv_k[seq].buf, self.m_kv_v[seq].buf, mask.buf, self.b_aout.buf],
                &push, 1, n_head as u32, 1)?;
        }
        // wo + 잔차
        self.quant(self.b_aout.buf, self.b_xq_g.buf, n_head * hd, 1)?;
        self.gemv(self.b_xq_g.buf, "blk.64.attn_output.weight", self.b_gout.buf, 1)?;
        self.axpy(self.m_cur.buf, self.b_gout.buf, n)?;
        // FFN
        self.rms(self.m_cur.buf, "blk.64.post_attention_norm", self.m_e.buf, n, 1)?;
        self.quant(self.m_e.buf, self.m_xq.buf, n, 1)?;
        self.gemv(self.m_xq.buf, "blk.64.ffn_gate.weight", self.b_fgate.buf, 1)?;
        self.gemv(self.m_xq.buf, "blk.64.ffn_up.weight", self.b_fup.buf, 1)?;
        self.silu_mul(self.b_fgate.buf, self.b_fup.buf, self.b_fglu.buf, self.n_ff)?;
        self.quant(self.b_fglu.buf, self.b_xq_f.buf, self.n_ff, 1)?;
        self.gemv(self.b_xq_f.buf, "blk.64.ffn_down.weight", self.b_fdown.buf, 1)?;
        self.axpy(self.m_cur.buf, self.b_fdown.buf, n)?;
        if !noba { self.ctx.end_batch_wait()?; } else { self.ctx.flush2()?; }
        if !with_head {
            return Ok(None);
        }
        // shared head norm → head → argmax
        self.rms(self.m_cur.buf, "blk.64.nextn.shared_head_norm", self.m_e.buf, n, 1)?;
        Ok(Some(self.head_argmax()?))
    }

    /// per-token 검증 — 행별 step + argmax + hidden 회수.
    fn verify_rows(
        &mut self,
        seq: usize,
        pos0: usize,
        emb: &[f32],
        argmaxes: &mut Vec<u32>,
        h_all: &mut Vec<f32>,
    ) -> Result<Vec<f32>, String> {
        // 배치 경로 — LLM170_VKD_BATCH=1 옵트인 (발산 조사 전 비활성).
        if std::env::var_os("LLM170_VKD_BATCH").is_some() {
            let n = self.n_embd;
            let mut last = Vec::new();
            for (off, ch) in emb.chunks(T_MAX * n).enumerate() {
                let t = ch.len() / n;
                let rows = self.step_batch(seq, pos0 + off, ch, true)?;
                for r in 0..t {
                    argmaxes.push(llm170_core::matmul::greedy_from(
                        &rows[r * self.n_vocab..(r + 1) * self.n_vocab]));
                }
                let mut hv = vec![0f32; t * n];
                unsafe { std::ptr::copy_nonoverlapping(self.b_xs.ptr as *const f32, hv.as_mut_ptr(), t * n) };
                h_all.extend_from_slice(&hv);
                last = rows[(t - 1) * self.n_vocab..].to_vec();
            }
            return Ok(last);
        }
        let n = self.n_embd;
        let mut last = Vec::new();
        for (ti, ch) in emb.chunks(n).enumerate() {
            let lg = self.step(seq, pos0 + ti, ch)?;
            argmaxes.push(llm170_core::matmul::greedy_from(&lg));
            h_all.extend_from_slice(&self.hidden_row());
            last = lg;
        }
        Ok(last)
    }

    /// GDN/conv 상태 스냅샷 (매핑 ptr 직접 — GPU 유휴 시).
    fn snapshot_states(&mut self) -> Result<(), String> {
        if self.ctx.batching.load(std::sync::atomic::Ordering::Relaxed) {
            self.ctx.end_batch_wait()?;
        }
        let gl = self.dt_rank * self.d_state * self.d_state;
        let cl = (self.conv_k - 1) * self.conv_ch;
        for (r, rows) in self.st_gdn.iter().enumerate() {
            let stride = rows.len();
            for (s, b) in rows.iter().enumerate() {
                let src: &[f32] = unsafe { std::slice::from_raw_parts(b.ptr as *const f32, gl) };
                self.snap_gdn[r * stride + s] = src.to_vec();
                let cs: &[f32] = unsafe { std::slice::from_raw_parts(self.st_conv[r][s].ptr as *const f32, cl) };
                self.snap_conv[r * stride + s] = cs.to_vec();
            }
        }
        Ok(())
    }

    fn restore_states(&mut self) -> Result<(), String> {
        if self.ctx.batching.load(std::sync::atomic::Ordering::Relaxed) {
            self.ctx.end_batch_wait()?;
        }
        let gl = self.dt_rank * self.d_state * self.d_state;
        let cl = (self.conv_k - 1) * self.conv_ch;
        for (r, rows) in self.st_gdn.iter().enumerate() {
            let stride = rows.len();
            for (s, b) in rows.iter().enumerate() {
                let snap = self.snap_gdn[r * stride + s].clone();
                if snap.len() == gl {
                    unsafe { std::ptr::copy_nonoverlapping(snap.as_ptr(), b.ptr as *mut f32, gl) };
                }
                let snapc = self.snap_conv[r * stride + s].clone();
                if snapc.len() == cl {
                    unsafe { std::ptr::copy_nonoverlapping(snapc.as_ptr(), self.st_conv[r][s].ptr as *mut f32, cl) };
                }
            }
        }
        Ok(())
    }
}

fn n_group_len(hp: &llm170_core::model::hparams::Hparams) -> usize {
    hp.n_group * hp.d_state
}
