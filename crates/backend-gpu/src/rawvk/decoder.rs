//! VkDecoder — GDN/어텐션 GPU 상주 디코드 (plans/19 2단계).
//! 커널 8종은 gdn-check ★ 검증 완료. 기존 gemv/quant/rms/silu SPIR-V 재사용.
//! rawhip DecodeState 대칭 — 배치 모드(단일 제출+배리어).

use crate::rawvk::context::{Pipes, VkBuf, VkCtx};
use ash::vk;
use std::collections::HashMap;

const GDN_CONV_SPV: &[u8] = include_bytes!("spv/gdn_conv_t.spv");
const GEMV8_Q5_SPV: &[u8] = include_bytes!("spv/gemv8_q5.spv");
const GEMV8_Q4_SPV: &[u8] = include_bytes!("spv/gemv8_q4.spv");
const GEMV8_XS_SPV: &[u8] = include_bytes!("spv/gemv8_xs.spv");
const GEMV8_Q3_SPV: &[u8] = include_bytes!("spv/gemv8_q3.spv");
const GEMV8_Q6_SPV: &[u8] = include_bytes!("spv/gemv8_q6.spv");
const GEMV8_Q8_SPV: &[u8] = include_bytes!("spv/gemv8_q8.spv");
const GEMV8_Q5B_SPV: &[u8] = include_bytes!("spv/gemv8_q5b.spv");
const GEMV8_Q4B_SPV: &[u8] = include_bytes!("spv/gemv8_q4b.spv");
const GEMV8_Q6B_SPV: &[u8] = include_bytes!("spv/gemv8_q6b.spv");
const GEMV8_Q8B_SPV: &[u8] = include_bytes!("spv/gemv8_q8b.spv");
const GEMV8_XSB_SPV: &[u8] = include_bytes!("spv/gemv8_xsb.spv");
const TILE_Q6K_SPV: &[u8] = include_bytes!("spv/tile_q6k.spv");
const GDN_CONV_STATE_SPV: &[u8] = include_bytes!("spv/gdn_conv_state.spv");
const SPLIT3_SPV: &[u8] = include_bytes!("spv/split3.spv");
const L2_SPV: &[u8] = include_bytes!("spv/l2_rows2.spv");
const BETA_G_SPV: &[u8] = include_bytes!("spv/gdn_beta_g.spv");
const GDN_AR_SPV: &[u8] = include_bytes!("spv/gdn_ar.spv");
const NORM_GATED_SPV: &[u8] = include_bytes!("spv/norm_gated.spv");
const QK_ROPE_SPV: &[u8] = include_bytes!("spv/qk_rope.spv");
const QK_ROPE2_SPV: &[u8] = include_bytes!("spv/qk_rope2.spv");
const KV_APPEND_SPV: &[u8] = include_bytes!("spv/kv_append.spv");
const QSA_FLASH_SPV: &[u8] = include_bytes!("spv/qsa_flash.spv");
const KV_APPEND_Q8_SPV: &[u8] = include_bytes!("spv/kv_append_q8.spv");
const QSA_FLASH_Q8_SPV: &[u8] = include_bytes!("spv/qsa_flash_q8.spv");
const COPY_OFF_SPV: &[u8] = include_bytes!("spv/copy_off.spv");
const TILE128_SPV: &[u8] = include_bytes!("spv/tile128_q5k.spv");
const TILE_XS_SPV: &[u8] = include_bytes!("spv/tile_xs.spv");
const TILE_Q8_SPV: &[u8] = include_bytes!("spv/tile_q8.spv");
const TILE_Q4K_SPV: &[u8] = include_bytes!("spv/tile_q4k.spv");
const TILE_Q3K_SPV: &[u8] = include_bytes!("spv/tile_q3k.spv");
const GEMM_I8_SPV: &[u8] = include_bytes!("spv/gemm_i8.spv");
const QUANT_B8_SPV: &[u8] = include_bytes!("spv/quant_b8.spv");
const QUANT_B8V2_SPV: &[u8] = include_bytes!("spv/quant_b8v2.spv");
const GEMM_I8V2_SPV: &[u8] = include_bytes!("spv/gemm_i8v2.spv");
const ADDRMS_SPV: &[u8] = include_bytes!("spv/addrms.spv");
const TILE_NL_SPV: &[u8] = include_bytes!("spv/tile_nl.spv");
const TILE_IQ3S_SPV: &[u8] = include_bytes!("spv/tile_iq3s.spv");
const TILE_F16_SPV: &[u8] = include_bytes!("spv/tile_f16.spv");
const TILE128O_SPV: &[u8] = include_bytes!("spv/tile128o.spv");
const TILE128W_SPV: &[u8] = include_bytes!("spv/tile128w.spv");
const TILE_MS2_SPV: &[u8] = include_bytes!("spv/tile_ms2.spv");
const TILE_MS4_SPV: &[u8] = include_bytes!("spv/tile_ms4.spv");
const TILE_Q4KMS_SPV: &[u8] = include_bytes!("spv/tile_q4kms.spv");
const TILE_Q6KMS_SPV: &[u8] = include_bytes!("spv/tile_q6kms.spv");
const TILE_Q3KMS_SPV: &[u8] = include_bytes!("spv/tile_q3kms.spv");
const TILE_Q8MS_SPV: &[u8] = include_bytes!("spv/tile_q8ms.spv");
const TILE_XSMS_SPV: &[u8] = include_bytes!("spv/tile_xsms.spv");
const TILE_NLMS_SPV: &[u8] = include_bytes!("spv/tile_nlms.spv");
const TILE_MS128_SPV: &[u8] = include_bytes!("spv/tile_ms128.spv");

/// q5_K 사전 언패분 — i8 가중 + 블록 스케일 (gemm_i8 전용).
/// f32 → f16 비트 (반올림-최근접짝수). q8_0 헤더 인코딩용.
fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x007f_ffff;
    if exp == 255 {
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let e2 = exp - 127 + 15;
    if e2 >= 31 {
        return sign | 0x7c00; // 오버플로 → inf
    }
    if e2 <= 0 {
        if e2 < -10 {
            return sign;
        }
        // 비정규
        let m = (mant | 0x0080_0000) >> (1 - e2);
        let r = (m + 0x1000) >> 13; // RNE 근사
        return sign | r as u16;
    }
    let mut h = ((e2 as u32) << 10) | (mant >> 13);
    // RNE
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | h as u16
}

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
    #[allow(clippy::type_complexity)]
    ktimes: std::collections::HashMap<String, (f64, u64)>,
    ktime: bool,
    kkey: std::cell::RefCell<Option<String>>,
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
    // ── f16 사전 디양자화 가중 캐시 (plans/39) — 프리필 타일 전용
    f16w: HashMap<String, VkBuf>,
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
        // plans/40: 17.5GB to_vec() 클론 폐지 — mmap 뷰를 그대로 빌려 전달.
        // (클론이 익명 RAM 17.5GB를 상주시켜 2프로세스 OOM의 직접 원인.)
        let wv: Vec<(&str, &[u8], u32, usize, usize)> = weights
            .iter()
            .map(|(k, w)| (k.as_str(), w.data, w.ty as u32, w.n_in as usize, w.n_out as usize))
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
        // step_batch 청크가 기본 (가중 1회 판독 상각 — 2026-09-04 발산은
        // 2026-09-05 디스크립터 세트 재사용 경합으로 판명, 수리 후 재발 없음;
        // 2026-09-08 judge VKD_BATCH+TILE 19/19 — plans/36 P1 종결).
        // LLM170_VKD_BATCH=0 킬스위치.
        if std::env::var("LLM170_VKD_BATCH").map(|v| v == "0").unwrap_or(false) {
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

const T_MAX: usize = 128;

impl DecoderState {
    /// 초기화 — 가중치(carveout)+상수(GTT) 업로드, 상태 0.
    #[allow(clippy::too_many_arguments)]
    pub fn new<'a>(
        mut ctx: VkCtx,
        weights: Vec<(&'a str, &'a [u8], u32, usize, usize)>,
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
        // plans/30 q3q8 옵트인: 소유 사본 재팩을 먼저 수행하고 모든 소비자는
        // 최종 뷰(weights_final)를 본다 (기본 경로는 mmap 빌림 그대로 — 클론 0).
        let q3q8 = std::env::var("LLM170_VK_Q3Q8").map(|v| v == "1").unwrap_or(false);
        let mut weights_owned: Option<Vec<(String, Vec<u8>, u32, usize, usize)>> = if q3q8 {
            Some(weights.iter().map(|(k, d, ty, ni, no)| (k.to_string(), d.to_vec(), *ty, *ni, *no)).collect())
        } else { None };
        if let Some(wv) = weights_owned.as_mut() {
            for (_name, data, ty, _ni, _no) in wv.iter_mut() {
                if *ty != 11 {
                    continue;
                }
                let (rows, k) = (*_no, *_ni);
                let mut out = Vec::with_capacity(rows * (k / 32) * 34);
                let mut row = vec![0.0f32; k];
                for r in 0..rows {
                    llm170_core::quant::dequant_row(
                        llm170_gguf::GgmlType::Q3K,
                        data, r as u64, k as u64, &mut row);
                    for blk in row.chunks(32) {
                        let amax = blk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
                        let d = amax / 127.0;
                        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                        let h = f32_to_f16_bits(d);
                        out.extend_from_slice(&h.to_le_bytes());
                        for &v in blk {
                            out.push(((v * id).round()).clamp(-127.0, 127.0) as i8 as u8);
                        }
                    }
                }
                *data = out;
                *ty = 8;
            }
        }
        let weights_final: Vec<(&str, &[u8], u32, usize, usize)> = match &weights_owned {
            Some(v) => v.iter().map(|(k, d, t, a, b)| (k.as_str(), d.as_slice(), *t, *a, *b)).collect(),
            None => weights.clone(),
        };
        // MTP 탑재·vocab — weights 이동 전 산출.
        let mtp_on = weights_final.iter().any(|(k, ..)| *k == "blk.64.nextn.eh_proj.weight");
        let n_vocab = weights_final
            .iter()
            .find(|(k, ..)| *k == "output.weight")
            // plans/29: 튜플 (k, data, ty, n_in, n_out) — 5번째가 n_out.
            // 종래 4번째(n_in)를 읽어 헤드가 어휘 5120행만 봄 (발산 근원).
            .map(|(_, _, _, _, no)| *no)
            .unwrap_or(n);
        // q5_K 원본 캡처 (i8 언패용 — 루프가 weights를 소비하기 전)
        // plans/40: 빌림 유지 — 클론 제거 (구 d.clone()가 q5 전체 ~8GB 복제)
        let q5k_src: Vec<(&str, &[u8], usize, usize)> = weights_final
            .iter()
            .filter(|(_, _, ty, _, _)| *ty == 13)
            .map(|(k, d, _, ni, no)| (*k, *d, *ni, *no))
            .collect();

        // f16 사전 디양자화 캐시 (plans/39) — 데이터 복제 없음(대여만):
        // 디양자화를 가중 업로드 루프 앞에서 수행 (RCA: .cloned() 전체복제가
        // 30Gi 호스트 RAM을 초과해 OOM·세션 사망의 원인이었음).
        let f16w_on = std::env::var("LLM170_VK_F16W").map(|v| v == "1").unwrap_or(false);
        let f16w_max = std::env::var("LLM170_VK_F16W_MAX").ok().and_then(|v| v.parse::<usize>().ok());
        let mut f16w: HashMap<String, VkBuf> = HashMap::new();
        if f16w_on {
            let e0 = std::time::Instant::now();
            let mut cand: Vec<&(&str, &[u8], u32, usize, usize)> = weights_final
                .iter()
                .filter(|(_, _, ty, _, _)| matches!(*ty, 8 | 11 | 12 | 13 | 14 | 20 | 21 | 23))
                .collect();
            if let Some(mx) = f16w_max {
                cand.truncate(mx);
            }
            for grp in cand.chunks(8) {
                let outs: std::sync::Mutex<Vec<(String, Vec<u16>)>> = std::sync::Mutex::new(Vec::new());
                std::thread::scope(|sc| {
                    for (name, data, ty, ni, no) in grp {
                        let outs = &outs;
                        sc.spawn(move || {
                            let gty = llm170_gguf::GgmlType::from_u32(*ty).unwrap_or(llm170_gguf::GgmlType::Q5K);
                            let (ni, no) = (*ni, *no);
                            let mut buf16 = vec![0u16; ni * no];
                            let mut row = vec![0f32; ni];
                            for r in 0..no {
                                llm170_core::quant::dequant_row(gty, data, r as u64, ni as u64, &mut row);
                                for (k, &v) in row.iter().enumerate() {
                                    buf16[r * ni + k] = f32_to_f16_bits(v);
                                }
                            }
                            outs.lock().unwrap().push((name.to_string(), buf16));
                        });
                    }
                });
                for (name, buf16) in outs.into_inner().unwrap() {
                    let bytes = buf16.len() * 2;
                    let mut b = ctx.alloc(bytes)?;
                    unsafe { std::ptr::copy_nonoverlapping(buf16.as_ptr() as *const u8, b.ptr, bytes) };
                    ctx.unmap(&mut b)?;
                    f16w.insert(name, b);
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
            }
            eprintln!("[f16w] 디양자화+업로드 {} 텐서 {}s", f16w.len(), e0.elapsed().as_secs_f32());
        }
        let mut w = HashMap::new();
        for &(name, data, ty, ni, no) in &weights_final {
            let mut bufs = Vec::new();
            let mut off = 0usize;
            // gemv4 WG() 시프트 산술 — 청크 크기 2의 거듭제곱. 마지막 청크는 실제 크기만
            // 할당: o = idx & mask 는 항상 청크 내 실데이터 오프셋이라 패딩 불필요.
            let ch_eff = data.len().next_power_of_two().min(1usize << (63 - ctx.max_ssbo.leading_zeros()));
            while off < data.len() {
                let rem = data.len() - off;
                let sz = ch_eff.min(rem);
                let mut b = ctx.alloc(sz)?;
                unsafe { std::ptr::copy_nonoverlapping(data.as_ptr().add(off), b.ptr, sz) };
                ctx.unmap(&mut b)?;
                bufs.push(b);
                off += sz;
            }
            w.insert(name.to_string(), (bufs, ty, ni, no));
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
        let kv: Vec<u32> = llm170_core::ktab2_packed();
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
        let kv8 = std::env::var("LLM170_VK_KV8").map(|v| v == "1").unwrap_or(false);
        let kv_store: usize = if kv8 { kv_len / 32 * 34 } else { kv_len * 4 };
        let zeros_kv = vec![0u8; kv_store];
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
            let zeros = vec![0u8; kv_store];
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
        for &(name, data, ni, no) in &q5k_src {
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
                wsr_map.insert(name.to_string(), b);
            }
            let wbuf = {
                let mut b = ctx.alloc(no * ni)?;
                unsafe { std::ptr::copy_nonoverlapping(w8.as_ptr() as *const u8, b.ptr, no * ni) };
                ctx.unmap(&mut b)?;
                b
            };
            i8w.insert(name.to_string(), I8W { w: wbuf, wsp: wspbuf, wsm: wsmbuf, n_out: no, n_in: ni });
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
            ktimes: std::collections::HashMap::new(),
            ktime: std::env::var_os("LLM170_VK_KTIME").is_some(),
            kkey: std::cell::RefCell::new(None),
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
            f16w,
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
            let p = self.ctx.pipeline_pipes(spv, n_buf, pb)?;
            self.pipes.insert(name, p);
        }
        Ok(self.pipes.get(name).unwrap())
    }

    /// 바인딩+런치 (배치 모드 자동 — fresh ds). 배리어 포함.
    fn run_pipe(&mut self, name: &'static str, spv: &[u8], n_buf: u32, pb: u32, bufs: &[vk::Buffer], push: &[u8], gx: u32, gy: u32, gz: u32) -> Result<(), String> {
        self.run_pipe_b(name, spv, n_buf, pb, bufs, push, gx, gy, gz, true)
    }

    /// 바인딩+런치 — bar=false면 직후 배리어 생략 (독립 병렬 그룹 내부).
    /// 그룹 마지막 디스패치는 반드시 bar=true로 종결해야 소비자가 안전하다.
    fn run_pipe_b(&mut self, name: &'static str, spv: &[u8], n_buf: u32, pb: u32, bufs: &[vk::Buffer], push: &[u8], gx: u32, gy: u32, gz: u32, bar: bool) -> Result<(), String> {
        let t0k = std::time::Instant::now();
        // 배치 자동 분할 — 디스패치 2048마다 제출·대기·재시작 (세트/CMDBUF 누적 방지;
        // 풀 상한 4096세트 이내. 512→2048: 스텝당 중간 드레인 제거, plans/36 G3).
        if self.ctx.batching.load(std::sync::atomic::Ordering::Relaxed) {
            self.split_ctr += 1;
            if self.split_ctr >= 2048 {
                self.split_ctr = 0;
                self.ctx.end_batch_wait()?;
                self.ctx.begin_batch()?;
            }
        }
        self.ctx.nobar_next.set(!bar);
        if let Some(ts) = &self.ctx.ts {
            if ts.n.get() + 2 <= 8192 {
                let key = self.kkey.borrow_mut().take().unwrap_or_else(|| name.to_string());
                ts.labels.borrow_mut().push(key);
            }
        }
        let p = *self.pipe(name, spv, n_buf, pb)?;
        let ds = self.ctx.bind_ds(&p, bufs)?;
        let r = self.ctx.run(p.pl, ds, p.pipe, push, gx, gy, gz);
        if self.ktime {
            let e = t0k.elapsed().as_secs_f64() * 1e3;
            let key = self.kkey.borrow_mut().take().unwrap_or_else(|| name.to_string());
            let ent = self.ktimes.entry(key).or_insert((0.0f64, 0u64));
            ent.0 += e;
            ent.1 += 1;
        }
        r
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


    /// gemv8_q5 (plans/33) — llama mul_mat_vec_q5_k 완전 포트 (typed u16 로드,
    /// SIMD-in-register 니블, fma 체인). 웜 162GB/s (역대 최고). LLM170_G8=1.
    /// bar=false: 독립 병렬 그룹 내부 (직후 배리어 생략).
    fn gemv8_q5(&mut self, xn: vk::Buffer, wkey: &str, out: vk::Buffer, t: usize, bar: bool) -> Result<(), String> {
        let (wbufs, ty, ni, no) = self.w.get(wkey).cloned().ok_or(format!("가중치 없음: {wkey}"))?;
        if ty != 13 && ty != 12 && ty != 23 && ty != 11 && ty != 14 && ty != 8 {
            return Err("gemv8: q3_K/q4_K/q5_K/q6_K/q8_0/iq4_xs만".into());
        }

        let mut binds: Vec<vk::Buffer> = wbufs.iter().map(|b| b.buf).collect();
        while binds.len() < 8 {
            binds.push(self.dummy.buf);
        }
        binds.push(xn);
        binds.push(out);
        if ty == 23 {
            binds.push(self.ktab.buf);
        }
        // xsb (plans/40) — llama generic dmmv 구조 × 검증 xs 디코드: 125→182GB/s.
        if ty == 23 && std::env::var("LLM170_VK_XSB").map(|v| v == "0").unwrap_or(true) {
            let push = Self::push_u32s(&[ni as u32, no as u32, t as u32, 0, 0, 2]);
            return self.run_pipe_b("gemv8_xsb", GEMV8_XSB_SPV, 11, 24, &binds, &push,
                1, no.div_ceil(2) as u32, t as u32, bar);
        }
        let rpf: u32 = if no < 4096 { 1 } else { 2 };   // llama NUM_ROWS=2
        if ty == 8 {
            // q8b (plans/40) — llama generic dmmv 구조: 87→329GB/s. LLM170_VK_Q8B=0 옵트아웃.
            if std::env::var("LLM170_VK_Q8B").map(|v| v == "0").unwrap_or(true) {
                let push = Self::push_u32s(&[ni as u32, no as u32, t as u32, 0, 0, 2]);
                return self.run_pipe_b("gemv8_q8b", GEMV8_Q8B_SPV, 10, 24, &binds, &push,
                    1, no.div_ceil(2) as u32, t as u32, bar);
            }
            // q8_0 — 34B 블록 (plans/40: 소형 straggler 레이턴시 해소)
            let cw = wbufs.first().map(|b| b.bytes / 4).unwrap_or(1) as u32;
            let cw = cw.next_power_of_two();
            let cw_log2 = 31u32 - cw.leading_zeros();
            let cw_mask = cw - 1u32;
            let push = Self::push_u32s(&[ni as u32, no as u32, t as u32, cw_log2, cw_mask, rpf]);
            return self.run_pipe_b("gemv8_q8", GEMV8_Q8_SPV, 10, 24, &binds, &push,
                1, no.div_ceil(rpf as usize) as u32, t as u32, bar);
        }
        let (pname, spv8, n_kb8) = match ty {
            23 => ("gemv8_xs", GEMV8_XS_SPV, 11),
            11 => ("gemv8_q3", GEMV8_Q3_SPV, 10),
            _ => ("", &[][..], 0),
        };
        if ty == 23 || ty == 11 {
            let cw = wbufs.first().map(|b| b.bytes / 4).unwrap_or(1) as u32;
            let cw = cw.next_power_of_two();
            let cw_log2 = 31u32 - cw.leading_zeros();
            let cw_mask = cw - 1;
            let push = Self::push_u32s(&[ni as u32, no as u32, t as u32, cw_log2, cw_mask, rpf]);
            return self.run_pipe_b(pname, spv8, n_kb8, 24, &binds, &push,
                1, no.div_ceil(rpf as usize) as u32, t as u32, bar);
        }
        if ty == 12 {
            // q4b (plans/40) — llama mul_mat_vec_q4_k 이식: 152→272GB/s. LLM170_VK_Q4B=0 옵트아웃.
            if std::env::var("LLM170_VK_Q4B").map(|v| v == "0").unwrap_or(true) {
                let push = Self::push_u32s(&[ni as u32, no as u32, t as u32, 0, 0, 2]);
                return self.run_pipe_b("gemv8_q4b", GEMV8_Q4B_SPV, 10, 24, &binds, &push,
                    1, no.div_ceil(2) as u32, t as u32, bar);
            }
            // q4 — u32 워드 단위 (동일 WG 워커)
            let cw = wbufs.first().map(|b| b.bytes / 4).unwrap_or(1) as u32;
            let cw = cw.next_power_of_two();
            let cw_log2 = 31u32 - cw.leading_zeros();
            let cw_mask = cw - 1;
            let push = Self::push_u32s(&[ni as u32, no as u32, t as u32, cw_log2, cw_mask, rpf]);
            return self.run_pipe_b("gemv8_q4", GEMV8_Q4_SPV, 10, 24, &binds, &push,
                1, no.div_ceil(rpf as usize) as u32, t as u32, bar);
        }
        if ty == 14 {
            // q6b (plans/40) — llama mul_mat_vec_q6_k 충실 이식(sccache): +35%. LLM170_VK_Q6B=0 옵트아웃.
            if std::env::var("LLM170_VK_Q6B").map(|v| v == "0").unwrap_or(true) {
                let push = Self::push_u32s(&[ni as u32, no as u32, t as u32, 0, 0, 2]);
                return self.run_pipe_b("gemv8_q6b", GEMV8_Q6B_SPV, 10, 24, &binds, &push,
                    1, no.div_ceil(2) as u32, t as u32, bar);
            }
            // q6 — u16 뷰 (105 u16/블록), llama mul_mat_vec_q6_k 직역 (plans/36 G1)
            let cw2 = wbufs.first().map(|b| b.bytes / 2).unwrap_or(1) as u32;
            let cw2 = cw2.next_power_of_two();
            let cw2_log2 = 31u32 - cw2.leading_zeros();
            let cw2_mask = cw2 - 1;
            let push = Self::push_u32s(&[ni as u32, no as u32, t as u32, cw2_log2, cw2_mask, rpf]);
            return self.run_pipe_b("gemv8_q6", GEMV8_Q6_SPV, 10, 24, &binds, &push,
                1, no.div_ceil(rpf as usize) as u32, t as u32, bar);
        }
        // q5b (plans/40) — llama mul_mat_vec_q5_k 충실 이식 (64스레드·2행·vec4).
        // 단일 청크 typed 뷰 — 143→225GB/s. LLM170_VK_Q5B=0 옵트아웃.
        if std::env::var("LLM170_VK_Q5B").map(|v| v == "0").unwrap_or(true) {
            let push = Self::push_u32s(&[ni as u32, no as u32, t as u32, 0, 0, 2]);
            return self.run_pipe_b("gemv8_q5b", GEMV8_Q5B_SPV, 10, 24, &binds, &push,
                1, no.div_ceil(2) as u32, t as u32, bar);
        }
        // q5 — u16 단위 청크 상수 (typed 뷰), 첫 버퍼 실측 크기 → pow2ceil
        let cw2 = wbufs.first().map(|b| b.bytes / 2).unwrap_or(1) as u32;
        let cw2 = cw2.next_power_of_two();
        let cw2_log2 = 31u32 - cw2.leading_zeros();
        let cw2_mask = cw2 - 1;
        let push = Self::push_u32s(&[ni as u32, no as u32, t as u32, cw2_log2, cw2_mask, rpf]);
        self.run_pipe_b("gemv8_q5", GEMV8_Q5_SPV, 10, 24, &binds, &push,
            1, no.div_ceil(rpf as usize) as u32, t as u32, bar)
    }
    /// gemv 래우터 — t<16은 gemv8(f32 직결, llama 포트), 그 외·미지원 타입은
    /// quant+gemv3(범용 정수 경로). LLM170_G8=0 킬스위치.
    /// 2026-09-08 A/B: q6_K도 gemv3+quant가 gemv6_q6보다 우위(tg32 7.06 vs 6.71) —
    /// gemv4/5/6/7 세대 전원 삭제(plans/35 P2).
    fn gemv_w(&mut self, qsrc: vk::Buffer, xq: vk::Buffer, wkey: &str, out: vk::Buffer, t: usize, nq: usize) -> Result<(), String> {
        let g8_off = std::env::var("LLM170_G8").map(|v| v == "0").unwrap_or(false);
        if t < 16 && !g8_off {
            if self.gemv8_q5(qsrc, wkey, out, t, true).is_ok() {
                return Ok(());
            }
        }
        self.quant(qsrc, xq, nq, t)?;
        self.gemv(xq, wkey, out, t)
    }


    /// HIP 기본 WMMA와 동일 정확도 클래스 maxrel ~4.9e-4, argmax 안정).
    /// LLM170_VK_NOTILE=1이면 항상 gemv3 정밀 경로.
    fn gemv(&mut self, xq: vk::Buffer, wkey: &str, out: vk::Buffer, t: usize) -> Result<(), String> {
        self.gemv_bar(xq, wkey, out, t, true)
    }

    /// bar=false: 독립 그룹 내부 — 최종 디스패치 직후 배리어 생략.
    fn gemv_bar(&mut self, xq: vk::Buffer, wkey: &str, out: vk::Buffer, t: usize, bar: bool) -> Result<(), String> {
        let (_, ty, _, no) = self.w.get(wkey).cloned().ok_or(format!("가중치 없음: {wkey}"))?;
        let tile_min: usize = if std::env::var_os("LLM170_VK_TILE1").is_some() { 1 } else { 16 };
        // f16 캐시 경로 (plans/39) — 루프 내 디양자화 없는 통일 타일
        if t >= tile_min
            && std::env::var_os("LLM170_VK_NOTILE").is_none()
            && std::env::var_os("LLM170_VK_NOF16W").is_none()
            && self.f16w.contains_key(wkey)
        {
            let xq_w = no; // 자리표시 — 아래에서 ni 기반 재계산
            let _ = xq_w;
            let ni_f = self.w.get(wkey).map(|e| e.2).unwrap_or(0);
            let xq_wf = ni_f / 4 + ni_f / 32 + ni_f / 16;
            let fbuf = self.f16w.get(wkey).cloned().unwrap();
            let gx = (no as u32 + 127) / 128;
            for tb in (0..t).step_by(128) {
                let nt = (t - tb).min(128) as u32;
                let last = tb + 128 >= t && bar;
                let push = Self::push_u32s(&[ni_f as u32, no as u32, xq_wf as u32, nt]);
                self.run_pipe_b("tile_f16", TILE_F16_SPV, 3, 16,
                    &[fbuf.buf, xq, out], &push, gx, 1, 1, last)?;
            }
            return Ok(());
        }
        // 타일(coopmat f16) 기본 경로 (2026-09-08 judge TILE 19/19 수용 —
        // llama 자체 pp가 동일 f16-닷 품질계약). 킬스위치 LLM170_VK_NOTILE=1.
        if t >= tile_min && std::env::var_os("LLM170_VK_NOTILE").is_none()
            && (ty == 11 || ty == 12 || ty == 13 || ty == 14 || ty == 20 || ty == 21 || ty == 23 || ty == 8) {
            return self.gemv_tile(xq, wkey, out, t, bar);
        }
        self.gemv_xq(xq, wkey, out, t, bar)
    }

    /// 타일(coopmat) 경로 — 프리필 전용. plans/32.
    fn gemv_tile(&mut self, xq: vk::Buffer, wkey: &str, out: vk::Buffer, t: usize, bar: bool) -> Result<(), String> {
        let (wbufs, ty, ni, no) = self.w.get(wkey).cloned().ok_or(format!("가중치 없음: {wkey}"))?;
        if std::env::var_os("LLM170_VK_SHAPES").is_some() {
            eprintln!("[shape] {wkey} ty={ty} ni={ni} no={no} gx={}", no.div_ceil(128));
        }
        // plans/30→32: tile128(coopmat f16)은 t=1 gemv와 수치계열이 다르나
        // 프리필 전용(t≥TILE_MIN)이면 spec 검증 배치(t≤5)와 무관 — 불변식 유지.
        // 실측 pp512 11.18→17.45 t/s (+56%). 옵트인 LLM170_VK_TILE=1.
        {
            let xq_w = ni / 4 + ni / 32 + ni / 16;
            let mut binds: Vec<vk::Buffer> = wbufs.iter().map(|b| b.buf).collect();
            while binds.len() < 8 {
                binds.push(self.dummy.buf);
            }
            binds.push(xq);
            binds.push(out);
            let gx = (no as u32 + 127) / 128;
            // tile_llm (plans/39): llama mul_mm 구조 직역 — f16vec2 shmem 15.4KB → 4 WG/CU
            // tile_ms2 (plans/40): llama m-warptile 지오메트리 + 비트-병렬 q5_K 언팩
            if ty == 13 && std::env::var("LLM170_TILE_MS2").map(|v| v == "1").unwrap_or(false) {
                let gx_ms2 = (no as u32 + 63) / 64;
                for tb in (0..t).step_by(64) {
                    let nt = (t - tb).min(64) as u32;
                    let last = tb + 64 >= t && bar;
                    let push = Self::push_u32s(&[ni as u32, no as u32, xq_w as u32, nt]);
                    self.run_pipe_b("tile_ms2", TILE_MS2_SPV, 10, 16, &binds, &push, gx_ms2, 1, 1, last)?;
                }
                return Ok(());
            }
            // tile_msALL (plans/40): 전 타입 ms 골격 (iq3s 제외) — WG() 제거·가드 제거·64행 WG
            // plans/40: ms 패밀리 기본 경로 승격 — verify 22/3, pp64 140→177.
            // 옵트아웃: LLM170_TILE_MSALL=0 (구 패밀리 복귀).
            let msall = std::env::var("LLM170_TILE_MSALL").map(|v| v != "0").unwrap_or(true);
            // 타입별 옵트인 (바이섹트): LLM170_TILE_MS_TYPES="q5,q8,xs,..." — MSALL 대체
            let ms_types: Option<Vec<u32>> = std::env::var("LLM170_TILE_MS_TYPES").ok().map(|s| {
                s.split(',').filter_map(|t| match t.trim() {
                    "q5" => Some(13u32),
                    "q4" => Some(12),
                    "q6" => Some(14),
                    "q3" => Some(11),
                    "q8" => Some(8),
                    "nl" => Some(20),
                    "xs" => Some(0),   // 와일드카드 else-브랜치
                    _ => None,
                }).collect()
            });
            let ms_on = |t: u32, wildcard: bool| -> bool {
                if msall { return true; }
                match &ms_types {
                    Some(v) => v.contains(&t) || (wildcard && v.contains(&0)),
                    None => false,
                }
            };
            let ms_spv: Option<(&str, &[u8], u32)> = match ty {
                13 if ms_on(13, false) => {
                    if std::env::var("LLM170_TILE_MS128").map(|v| v == "1").unwrap_or(false) {
                        Some(("tile_ms128", TILE_MS128_SPV, 10))
                    } else {
                        Some(("tile_ms4", TILE_MS4_SPV, 10))
                    }
                }
                12 if ms_on(12, false) => Some(("tile_q4kms", TILE_Q4KMS_SPV, 10)),
                14 if ms_on(14, false) => Some(("tile_q6kms", TILE_Q6KMS_SPV, 10)),
                11 if ms_on(11, false) => Some(("tile_q3kms", TILE_Q3KMS_SPV, 10)),
                8 if ms_on(8, false) => Some(("tile_q8ms", TILE_Q8MS_SPV, 10)),
                20 if ms_on(20, false) => Some(("tile_nlms", TILE_NLMS_SPV, 11)),
                _ if ms_on(0, true) && ty != 21 => Some(("tile_xsms", TILE_XSMS_SPV, 11)),
                _ => None,
            };
            if let Some((nm, spv, nkb)) = ms_spv {
                if nkb == 11 {
                    binds.push(self.ktab.buf);   // xs/nl LUT (구경로와 동일)
                }
                let step: usize = if std::env::var("LLM170_TILE_MS128").map(|v| v == "1").unwrap_or(false) && ty == 13 { 128 } else { 64 };
                let gx_ms = (no as u32 + 63) / 64;
                for tb in (0..t).step_by(step) {
                    let nt = (t - tb).min(step) as u32;
                    let last = tb + step >= t && bar;
                    let push = Self::push_u32s(&[ni as u32, no as u32, xq_w as u32, nt]);
                    self.run_pipe_b(nm, spv, nkb, 16, &binds, &push, gx_ms, 1, 1, last)?;
                }
                return Ok(());
            }
            // tile_ms4 (plans/40): ms2 + WG() 제거 + MMA 가드 제거 — 단일 청크 직인덱스 63.7GB/s
            if ty == 13 && std::env::var("LLM170_TILE_MS4").map(|v| v == "1").unwrap_or(false) {
                let gx_ms4 = (no as u32 + 63) / 64;
                for tb in (0..t).step_by(64) {
                    let nt = (t - tb).min(64) as u32;
                    let last = tb + 64 >= t && bar;
                    let push = Self::push_u32s(&[ni as u32, no as u32, xq_w as u32, nt]);
                    self.run_pipe_b("tile_ms4", TILE_MS4_SPV, 10, 16, &binds, &push, gx_ms4, 1, 1, last)?;
                }
                return Ok(());
            }
            // tile128w (plans/39): 256스레드 WMITER=2 → 2 WG/CU 점유
            if ty == 13 && std::env::var("LLM170_TILE_W").map(|v| v == "1").unwrap_or(false) {
                for tb in (0..t).step_by(128) {
                    let nt = (t - tb).min(128) as u32;
                    let last = tb + 128 >= t && bar;
                    let push = Self::push_u32s(&[ni as u32, no as u32, xq_w as u32, nt]);
                    self.run_pipe_b("tile128w", TILE128W_SPV, 10, 16, &binds, &push, gx, 1, 1, last)?;
                }
                return Ok(());
            }
            // tile128o (점유 변형, plans/39): 64토큰/1-sb/LDS 29.7KB → 2 WG/CU
            if ty == 13 && std::env::var("LLM170_TILE_OCC").map(|v| v == "1").unwrap_or(false) {
                for tb in (0..t).step_by(64) {
                    let nt = (t - tb).min(64) as u32;
                    let last = tb + 64 >= t && bar;
                    let push = Self::push_u32s(&[ni as u32, no as u32, xq_w as u32, nt]);
                    self.run_pipe_b("tile128o", TILE128O_SPV, 10, 16, &binds, &push, gx, 1, 1, last)?;
                }
                return Ok(());
            }
            let step = if ty == 21 { 64 } else { 128 };   // 재생성 패밀리 128토큰, iq3s 구형 (plans/39)
            let n_tb = t.div_ceil(step);
            for (tbi, tb) in (0..t).step_by(step).enumerate() {
                let nt = (t - tb).min(step) as u32;
                let last = tbi + 1 == n_tb && bar;
                if ty == 13 {
                    let push = Self::push_u32s(&[ni as u32, no as u32, xq_w as u32, nt]);
                    self.run_pipe_b("tile128", TILE128_SPV, 10, 16, &binds, &push, gx, 1, 1, last)?;
                } else if ty == 12 {
                    // tile_q4k: ql@16 128B, qh 없음 — 시프트 청크 push
                    let cw = wbufs.first().map(|b| b.bytes.next_power_of_two() / 4).unwrap_or(1) as u32;
                    let cw_log2 = 31u32 - cw.leading_zeros();
                    let cw_mask = (1u32 << cw_log2) - 1u32;
                    let push = Self::push_u32s(&[ni as u32, no as u32, xq_w as u32, nt, cw_log2, cw_mask]);
                    self.run_pipe_b("tile_q4k", TILE_Q4K_SPV, 10, 24, &binds, &push, gx, 1, 1, last)?;
                } else if ty == 14 {
                    // tile_q6k: ql 니블 + qh 2비트 + i8 스케일 + d @208
                    let cw = wbufs.first().map(|b| b.bytes.next_power_of_two() / 4).unwrap_or(1) as u32;
                    let cw_log2 = 31u32 - cw.leading_zeros();
                    let cw_mask = (1u32 << cw_log2) - 1u32;
                    let push = Self::push_u32s(&[ni as u32, no as u32, xq_w as u32, nt, cw_log2, cw_mask]);
                    self.run_pipe_b("tile_q6k", TILE_Q6K_SPV, 10, 24, &binds, &push, gx, 1, 1, last)?;
                } else if ty == 11 {
                    // q3_K — 구 tile_q3k 디코드 결함(스케일 tmp 3바이트/하프 인덱스 — plans/40)
                    // → 검증된 tile_q3kms(ms 골격)로 영구 전환. maxrel 0.0033.
                    let gx_q3 = (no as u32 + 63) / 64;
                    let push = Self::push_u32s(&[ni as u32, no as u32, xq_w as u32, nt]);
                    self.run_pipe_b("tile_q3kms", TILE_Q3KMS_SPV, 10, 16, &binds, &push, gx_q3, 1, 1, last)?;
                } else if ty == 8 {
                    // tile_q8 (plans/32): q8_0 coopmat — 소형(beta/alpha)도 포함
                    // (gemv3 t≥16 소형은 0.2GB/s급 병목 — ts 프로파일 2026-09-08)
                    let cw = wbufs.first().map(|b| b.bytes.next_power_of_two() / 4).unwrap_or(1) as u32;
                    let cw_log2 = 31u32 - cw.leading_zeros();
                    let cw_mask = (1u32 << cw_log2) - 1u32;
                    let push = Self::push_u32s(&[ni as u32, no as u32, xq_w as u32, nt, cw_log2, cw_mask]);
                    self.run_pipe_b("tile_q8", TILE_Q8_SPV, 10, 24, &binds, &push, gx, 1, 1, last)?;
                } else if ty == 20 {
                    // tile_nl: iq4_nl 18B 블록 — ktab 니블 LUT (iq4_xs와 공유)
                    let cw = wbufs.first().map(|b| b.bytes.next_power_of_two() / 4).unwrap_or(1) as u32;
                    let cw_log2 = 31u32 - cw.leading_zeros();
                    let cw_mask = (1u32 << cw_log2) - 1u32;
                    binds.push(self.ktab.buf);
                    let push = Self::push_u32s(&[ni as u32, no as u32, xq_w as u32, nt, cw_log2, cw_mask]);
                    self.run_pipe_b("tile_nl", TILE_NL_SPV, 11, 24, &binds, &push, gx, 1, 1, last)?;
                    binds.pop();
                } else if ty == 21 {
                    // tile_iq3s: 110B 블록 + IQ3S_GRID 512워드 바인딩
                    let cw = wbufs.first().map(|b| b.bytes.next_power_of_two() / 4).unwrap_or(1) as u32;
                    let cw_log2 = 31u32 - cw.leading_zeros();
                    let cw_mask = (1u32 << cw_log2) - 1u32;
                    binds.push(self.grid3s.buf);
                    let push = Self::push_u32s(&[ni as u32, no as u32, xq_w as u32, nt, cw_log2, cw_mask]);
                    self.run_pipe_b("tile_iq3s", TILE_IQ3S_SPV, 11, 24, &binds, &push, gx, 1, 1, last)?;
                    binds.pop();
                } else {
                    // tile_xs (plans/32): iq4_xs coopmat — ktab 바인딩, 시프트 청크
                    let cw = wbufs.first().map(|b| b.bytes.next_power_of_two() / 4).unwrap_or(1) as u32;
                    let cw_log2 = 31u32 - cw.leading_zeros();
                    let cw_mask = (1u32 << cw_log2) - 1u32;
                    binds.push(self.ktab.buf);
                    let push = Self::push_u32s(&[ni as u32, no as u32, xq_w as u32, nt, cw_log2, cw_mask]);
                    self.run_pipe_b("tile_xs", TILE_XS_SPV, 11, 24, &binds, &push, gx, 1, 1, last)?;
                    binds.pop();
                }
            }
            return Ok(());
        }
        unreachable!("gemv_tile: 타입 미적용")
    }

    /// 비타일 gemv (원본 경로).
    fn gemv_xq(&mut self, xq: vk::Buffer, wkey: &str, out: vk::Buffer, t: usize, bar: bool) -> Result<(), String> {
        let (wbufs, ty, ni, no) = self.w.get(wkey).cloned().ok_or(format!("가중치 없음: {wkey}"))?;
        if self.ktime || self.ctx.ts.is_some() {
            *self.kkey.borrow_mut() = Some(format!("gemv:ty{ty}:{wkey}"));
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
        // plans/29: 균일 청크 워드 수 전달 (마지막 청크만 부분 — WG 호산소 정합).
        // 단일 청크면 전체/4 → c 항상 0.
        let chunk_words = wbufs.first().map(|b| b.bytes / 4).unwrap_or(1) as u32;
        let push = Self::push_u32s(&[ni as u32, no as u32, xq_w as u32, ty, t as u32, chunk_words]);
        self.run_pipe_b("gemv", crate::rawvk::gemv::GEMV_SPV, 12, 24, &binds, &push, no as u32, 1, 1, bar)
    }

    /// v2 (mlx식) — per-row 스케일: quant_b8v2 + gemm_i8v2. LLM170_VK_I8=2.
    fn gemm_i8v2(&mut self, wkey: &str, out: vk::Buffer, t: usize, bar: bool) -> Result<(), String> {
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
        self.run_pipe_b("gemm_i8v2", GEMM_I8V2_SPV, 12, 12, &binds, &push2,
            (no as u32 + 15) / 16, 1, 1, bar)
    }

    /// gemm_i8 (plans/23) — q5_K 사전 언패분으로 t행 GEMM.
    fn gemm_i8(&mut self, wkey: &str, out: vk::Buffer, t: usize, bar: bool) -> Result<(), String> {
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
        self.run_pipe_b("gemm_i8", GEMM_I8_SPV, 16, 16, &binds, &push,
            (no as u32 + 15) / 16, 1, 1, bar)
    }

    /// 단계 공유 GEMV 그룹 — 잡들은 상호 독립(동일 입력·상이 출력)이라
    /// 그룹 내부 배리어 생략, 마지막 잡이 배리어로 종결.
    /// xq 양자화는 실제 폴백 잡이 있을 때만 수행 (gemv8 직결 사이트의 dead
    /// quant 스킵 — plans/36 G2). i8 잡(t≥2·VK_I8ON)은 gemm_i8.
    fn gemv_stage(&mut self, n: usize, t: usize, jobs: &[(String, vk::Buffer, vk::Buffer)]) -> Result<(), String> {
        let g8 = t < 16 && std::env::var("LLM170_G8").map(|v| v != "0").unwrap_or(true);
        let i8_on = t >= 2 && std::env::var_os("LLM170_VK_I8ON").is_some();
        let v2 = std::env::var("LLM170_VK_I8").map(|v| v == "2").unwrap_or(false);
        // 사전 판정 (self 대여 분리 — 클로저로 두면 mut 대여와 충돌)
        let elig: Vec<bool> = jobs
            .iter()
            .map(|(k, _, _)| matches!(self.w.get(k).map(|e| e.1), Some(8 | 11 | 12 | 13 | 14 | 23)))
            .collect();
        let i8s: Vec<bool> = jobs.iter().map(|(k, _, _)| i8_on && self.i8w.contains_key(k)).collect();
        // xq 필요 조건: gemv8/타일 외 폴백 잡이 하나라도 있을 때
        let need_xq = (0..jobs.len()).any(|ji| !i8s[ji] && (!g8 || !elig[ji]));
        if need_xq {
            self.quant(self.b_xn.buf, self.b_xq_n.buf, n, t)?;
        }
        let last = jobs.len() - 1;
        for (ji, (k, xq, out)) in jobs.iter().enumerate() {
            let bar = ji == last;
            if i8s[ji] {
                if v2 {
                    self.gemm_i8v2(k, *out, t, bar)?;
                } else {
                    self.gemm_i8(k, *out, t, bar)?;
                }
            } else if g8 && elig[ji] {
                // spec 검증 배치(t≤5)도 gemv8 수치계열로 — 불변식 회복 (plans/33)
                self.gemv8_q5(self.b_xn.buf, k, *out, t, bar)?;
            } else {
                self.gemv_bar(*xq, k, *out, t, bar)?;
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

    /// 잔차 덧셈 + rms_norm 융합 (plans/36 G2) — y += x·1.0 결과를 그대로
    /// 세그먼트 축소하므로 axpy+rms 2디스패치와 비트동일, 런치만 절감.
    fn addrms(&mut self, y: vk::Buffer, x: vk::Buffer, wkey: &str, out: vk::Buffer, n: usize, t: usize) -> Result<(), String> {
        let wbuf = self.consts.get(wkey).cloned().ok_or(format!("상수 없음: {wkey}"))?;
        let eps = self.eps;
        let mut push = Self::push_u32s(&[n as u32, t as u32]);
        push.extend_from_slice(&eps.to_le_bytes());
        self.run_pipe("addrms", ADDRMS_SPV, 4, 12,
            &[y, x, wbuf.buf, out], &push, t as u32, 1, 1)
    }

    /// t=1 단일 스텝 — 배치 모드로 전 층 단일 제출·다운로드 1회.
    pub fn step(&mut self, seq: usize, pos: usize, emb: &[f32]) -> Result<Vec<f32>, String> {
        let noba = std::env::var_os("LLM170_VK_NOBATCH").is_some();
        let kv8 = std::env::var("LLM170_VK_KV8").map(|v| v == "1").unwrap_or(false);
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
            // ── attn_norm — 0층만 (이후 층 입력 규격화는 하단 fdown addrms에 융합)
            if il == 0 {
                let (xs, xn) = (self.b_xs.clone(), self.b_xn.clone());
                self.rms(xs.buf, "blk.0.attn_norm", xn.buf, n, 1)?;
            }
            if self.is_recr[il] {
                // GDN 4 GEMV — 독립 그룹 (gemv_stage: 내부 배리어 생략·dead quant 스킵)
                self.gemv_stage(n, 1, &[
                    (format!("blk.{il}.attn_qkv.weight"), self.b_xq_n.buf, self.b_gqkv.buf),
                    (format!("blk.{il}.attn_gate.weight"), self.b_xq_n.buf, self.b_gz.buf),
                    (format!("blk.{il}.ssm_beta.weight"), self.b_xq_n.buf, self.b_gb.buf),
                    (format!("blk.{il}.ssm_alpha.weight"), self.b_xq_n.buf, self.b_ga.buf),
                ])?;
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
                if gskip & 4 == 0 {
                    self.gemv_w(self.b_ggated.buf.clone(), self.b_xq_g.buf, &format!("blk.{il}.ssm_out.weight"), self.b_gout.buf, 1, d_inner)?;
                }
                if std::env::var_os("LLM170_VKD_TRACE").is_some() && il < 2 {
                    self.ctx.end_batch_wait().ok();
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
                self.gemv_stage(n, 1, &[
                    (format!("blk.{il}.attn_q.weight"), self.b_xq_n.buf, self.b_aq.buf),
                    (format!("blk.{il}.attn_k.weight"), self.b_xq_n.buf, self.b_ak.buf),
                    (format!("blk.{il}.attn_v.weight"), self.b_xq_n.buf, self.b_av.buf),
                ])?;
                // qk_rope
                {
                    let qn = self.consts.get(&format!("blk.{il}.attn_q_norm")).cloned().ok_or("qn")?;
                    let kn = self.consts.get(&format!("blk.{il}.attn_k_norm")).cloned().ok_or("kn")?;
                    let cs = self.consts.get("cs").cloned().ok_or("cs")?;
                    // PC: {float eps; float kqs; int pos; int nh; int nk; int hd; int nr} — gdn-check 준거
                    let mut push = self.eps.to_le_bytes().to_vec();
                    push.extend_from_slice(&self.kq_scale.to_le_bytes());
                    push.extend(Self::push_u32s(&[pos as u32, n_head as u32, n_kv as u32, hd as u32, n_rot as u32]));
                    self.run_pipe("qk_rope2", QK_ROPE2_SPV, 5, 28,
                        &[self.b_aq.buf, self.b_ak.buf, qn.buf, kn.buf, cs.buf],
                        &push, (n_head + n_kv) as u32, 1, 1)?;
                }
                if attn_cut >= 2 {
                // kv append
                {
                    let push = Self::push_u32s(&[(n_kv * hd) as u32, pos as u32]);
                    // k/v 어펜드는 상호 독립 — k 배리어 생략, v가 종결 (flash는 둘 다 판독)
                    if kv8 {
                        let gq = (n_kv * hd).div_ceil(32).div_ceil(64) as u32;
                        self.run_pipe_b("kv_app_q8", KV_APPEND_Q8_SPV, 2, 8,
                            &[self.b_ak.buf, self.kv_k[full_idx][seq].buf], &push, gq, 1, 1, false)?;
                        self.run_pipe("kv_app_q8", KV_APPEND_Q8_SPV, 2, 8,
                            &[self.b_av.buf, self.kv_v[full_idx][seq].buf], &push, gq, 1, 1)?;
                    } else {
                        self.run_pipe_b("kv_app", KV_APPEND_SPV, 2, 8,
                            &[self.b_ak.buf, self.kv_k[full_idx][seq].buf], &push,
                            (n_kv * hd).div_ceil(64) as u32, 1, 1, false)?;
                        self.run_pipe("kv_app", KV_APPEND_SPV, 2, 8,
                            &[self.b_av.buf, self.kv_v[full_idx][seq].buf], &push,
                            (n_kv * hd).div_ceil(64) as u32, 1, 1)?;
                    }
                }
                if attn_cut >= 3 {
                // flash
                {
                    let push = Self::push_u32s(&[pos as u32, n_head as u32, n_kv as u32, hd as u32]);
                    if kv8 {
                        self.run_pipe("qsa_flash_q8", QSA_FLASH_Q8_SPV, 4, 16,
                            &[self.b_aq.buf, self.kv_k[full_idx][seq].buf, self.kv_v[full_idx][seq].buf, self.b_aout.buf],
                            &push, 1, n_head as u32, 1)?;
                    } else {
                        self.run_pipe("qsa_flash", QSA_FLASH_SPV, 4, 16,
                            &[self.b_aq.buf, self.kv_k[full_idx][seq].buf, self.kv_v[full_idx][seq].buf, self.b_aout.buf],
                            &push, 1, n_head as u32, 1)?;
                    }
                }
                if attn_cut >= 4 {
                    self.gemv_w(self.b_aout.buf.clone(), self.b_xq_g.buf, &format!("blk.{il}.attn_output.weight"), self.b_gout.buf, 1, n_head * hd)?;
                }
                }
                }
                full_idx += 1;
            }
            // 잔차 + post_norm — addrms 융합 (plans/36 G2: axpy+rms 비트동일)
            self.addrms(self.b_xs.buf, self.b_gout.buf, &format!("blk.{il}.post_norm"), self.b_xn.buf, n, 1)?;
            // ── FFN
            self.gemv_stage(n, 1, &[
                (format!("blk.{il}.ffn_gate.weight"), self.b_xq_n.buf, self.b_fgate.buf),
                (format!("blk.{il}.ffn_up.weight"), self.b_xq_n.buf, self.b_fup.buf),
            ])?;
            self.silu_mul(self.b_fgate.buf, self.b_fup.buf, self.b_fglu.buf, self.n_ff)?;
            self.gemv_w(self.b_fglu.buf.clone(), self.b_xq_f.buf, &format!("blk.{il}.ffn_down.weight"), self.b_fdown.buf, 1, self.n_ff)?;
            // 잔차 + 다음층 attn_norm / head output_norm — addrms 융합
            let is_last = il + 1 >= self.n_layer || layer_cut.is_some_and(|c| il + 1 >= c);
            let nkey = if is_last {
                "output_norm".to_string()
            } else {
                format!("blk.{}.attn_norm", il + 1)
            };
            self.addrms(self.b_xs.buf, self.b_fdown.buf, &nkey, self.b_xn.buf, n, 1)?;
            // 실험: L0 FFN 직후 attn_q gemv 강제 (층 위치 vs 가중치 분리)
            if std::env::var_os("LLM170_VK_FORCE_AQ").is_some() && il == 0 {
                self.gemv_w(self.b_xn.buf.clone(), self.b_xq_n.buf, "blk.3.attn_q.weight", self.b_aq.buf, 1, n)?;
            }
        }
        // ── head: gemv(output) — output_norm은 마지막 addrms에 융합, quant는
        // gemv_w 폴백 시 내부 수행. 트렁크와 동일 배치로 단일 제출·대기 (G3).
        self.gemv_w(self.b_xn.buf.clone(), self.b_xq_n.buf, "output.weight", self.b_lg.buf, 1, n)?;
        if !noba { self.ctx.end_batch_wait()?; } else { self.ctx.flush2()?; };
        self.ctx.ts_report();
        if self.ktime {
            let mut v: Vec<_> = self.ktimes.iter().collect();
            v.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
            let tot: f64 = v.iter().map(|(_, (e, _))| *e).sum();
            eprintln!("[ktime1] wall {:.1}ms · 커널합 {tot:.1}ms", vk_t0.elapsed().as_secs_f32() * 1e3);
            for (k, (e, c)) in v.iter().take(12) {
                eprintln!("[ktime1] {:22} {:9.1}ms ({}회)", k, e, c);
            }
            self.ktimes.clear();
        }
        if std::env::var_os("LLM170_VK_PROF").is_some() { eprintln!("[vkprof] step: {:.1}ms (pos {})", vk_t0.elapsed().as_secs_f32()*1e3, pos); }
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
        let mut logits = vec![0f32; self.n_vocab];
        unsafe { std::ptr::copy_nonoverlapping(self.b_lg.ptr as *const f32, logits.as_mut_ptr(), self.n_vocab) };
        Ok(logits)
    }

    /// t행 배치 스텝 (plans/20) — 가중 1회 판독 분할 상각. 행별 산술은
    /// step()과 비트 동일(gemv3 행별 lane 축산·AR 내부 순차·conv 이력 판독).
    /// all_logits=true: 전 행 head 로짓 [t][n_vocab] (verify용 — b_lg_t).
    /// 아니면 마지막 행만 (b_lg). emb는 [t][n_embd].
    pub fn step_batch(&mut self, seq: usize, pos0: usize, emb: &[f32], all_logits: bool) -> Result<Vec<f32>, String> {
        let kv8 = std::env::var("LLM170_VK_KV8").map(|v| v == "1").unwrap_or(false);
        let noba = std::env::var_os("LLM170_VK_NOBATCH").is_some();
        let vk_t0b = std::time::Instant::now();
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
            // ── attn_norm — 0층만 (이후 fdown addrms 융합). xq는 gemv_stage 지연 양자화.
            if il == 0 {
                let (xs, xn) = (self.b_xs.clone(), self.b_xn.clone());
                self.rms(xs.buf, "blk.0.attn_norm", xn.buf, n, t)?;
            }
            if std::env::var_os("LLM170_VKD_TRACE").is_some() && il == 0 {
                self.ctx.end_batch_wait().ok(); self.ctx.begin_batch().ok();
                let x = vec![0f32; 64];
                let s0: f64 = unsafe { std::slice::from_raw_parts(self.b_xs.ptr as *const f32, 64) }.iter().map(|&v| v as f64).sum();
                eprintln!("#  SB post-rms xs0={s0:.4}");
            }
            if self.is_recr[il] {
                // plans/30: gemm_i8/quant_b8 경로는 배치 상태를 오염(실측 —
                // VK_NOI8=1로 재현 해소). LLM170_VK_I8ON=1 옵트인만 사용.
                if t >= 2 && std::env::var_os("LLM170_VK_I8ON").is_some() {
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
                self.gemv_w(self.b_ggated.buf.clone(), self.b_xq_g.buf, &format!("blk.{il}.ssm_out.weight"), self.b_gout.buf, t, d_inner)?;
                recr_idx += 1;
            } else {
                // i8 활성(소비 조건과 동일)일 때만 b8 양자화 — 기본 경로의 dead dispatch 제거
                if t >= 2 && std::env::var_os("LLM170_VK_I8ON").is_some()
                    && std::env::var_os("LLM170_VK_NOI8").is_none()
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
                    self.run_pipe("qk_rope2", QK_ROPE2_SPV, 5, 28,
                        &[self.b_aq.buf, self.b_ak.buf, qn.buf, kn.buf, cs.buf],
                        &push, (n_head + n_kv) as u32, t as u32, 1)?;
                }
                // kv append — grid (n/64, t). k/v 상호 독립 — k 배리어 생략, v가 종결
                {
                    let push = Self::push_u32s(&[(n_kv * hd) as u32, pos0 as u32]);
                    if kv8 {
                        let gq = (n_kv * hd).div_ceil(32).div_ceil(64) as u32;
                        self.run_pipe_b("kv_app_q8", KV_APPEND_Q8_SPV, 2, 8,
                            &[self.b_ak.buf, self.kv_k[full_idx][seq].buf], &push, gq, t as u32, 1, false)?;
                        self.run_pipe("kv_app_q8", KV_APPEND_Q8_SPV, 2, 8,
                            &[self.b_av.buf, self.kv_v[full_idx][seq].buf], &push, gq, t as u32, 1)?;
                    } else {
                        self.run_pipe_b("kv_app", KV_APPEND_SPV, 2, 8,
                            &[self.b_ak.buf, self.kv_k[full_idx][seq].buf], &push,
                            (n_kv * hd).div_ceil(64) as u32, t as u32, 1, false)?;
                        self.run_pipe("kv_app", KV_APPEND_SPV, 2, 8,
                            &[self.b_av.buf, self.kv_v[full_idx][seq].buf], &push,
                            (n_kv * hd).div_ceil(64) as u32, t as u32, 1)?;
                    }
                }
                // flash — grid (t, n_head), np = pos0+행+1
                {
                    let push = Self::push_u32s(&[pos0 as u32, n_head as u32, n_kv as u32, hd as u32]);
                    if kv8 {
                        self.run_pipe("qsa_flash_q8", QSA_FLASH_Q8_SPV, 4, 16,
                            &[self.b_aq.buf, self.kv_k[full_idx][seq].buf, self.kv_v[full_idx][seq].buf, self.b_aout.buf],
                            &push, t as u32, n_head as u32, 1)?;
                    } else {
                        self.run_pipe("qsa_flash", QSA_FLASH_SPV, 4, 16,
                            &[self.b_aq.buf, self.kv_k[full_idx][seq].buf, self.kv_v[full_idx][seq].buf, self.b_aout.buf],
                            &push, t as u32, n_head as u32, 1)?;
                    }
                }
                self.gemv_w(self.b_aout.buf.clone(), self.b_xq_g.buf, &format!("blk.{il}.attn_output.weight"), self.b_gout.buf, t, n_head * hd)?;
                full_idx += 1;
            }
            // 잔차 + post_norm — addrms 융합 (t행)
            self.addrms(self.b_xs.buf, self.b_gout.buf, &format!("blk.{il}.post_norm"), self.b_xn.buf, n, t)?;
            // FFN — xq는 gemv_stage 지연 양자화
            if t >= 2 && std::env::var_os("LLM170_VK_I8ON").is_some() {
                self.quant_b8(self.b_xn.buf, n, t)?;
            }
            self.gemv_stage(n, t, &[
                (format!("blk.{il}.ffn_gate.weight"), self.b_xq_n.buf, self.b_fgate.buf),
                (format!("blk.{il}.ffn_up.weight"), self.b_xq_n.buf, self.b_fup.buf),
            ])?;
            self.silu_mul(self.b_fgate.buf, self.b_fup.buf, self.b_fglu.buf, self.n_ff * t)?;
            self.quant(self.b_fglu.buf, self.b_xq_f.buf, self.n_ff, t)?;
            self.gemv_w(self.b_fglu.buf.clone(), self.b_xq_f.buf, &format!("blk.{il}.ffn_down.weight"), self.b_fdown.buf, t, self.n_ff)?;
            // 잔차 + 다음층 attn_norm / head output_norm — addrms 융합
            let is_last = il + 1 >= self.n_layer;
            let nkey = if is_last {
                "output_norm".to_string()
            } else {
                format!("blk.{}.attn_norm", il + 1)
            };
            self.addrms(self.b_xs.buf, self.b_fdown.buf, &nkey, self.b_xn.buf, n, t)?;
        }
        // ── head (all_logits) — output_norm은 마지막 addrms에 융합. 트렁크와
        /// 동일 배치로 단일 제출·대기 (G3). quant는 gemv_w 폴백 시 내부 수행.
        if all_logits {
            self.gemv_w(self.b_xn.buf.clone(), self.b_xq_n.buf, "output.weight", self.b_lg_t.buf, t, n)?;
        }
        if !noba { self.ctx.end_batch_wait()?; } else { self.ctx.flush2()?; };
        self.ctx.ts_report();
        if all_logits {
            let mut out = vec![0f32; t * self.n_vocab];
            unsafe { std::ptr::copy_nonoverlapping(self.b_lg_t.ptr as *const f32, out.as_mut_ptr(), t * self.n_vocab) };
            return Ok(out);
        }
        if self.ktime {
            let mut v: Vec<_> = self.ktimes.iter().collect();
            v.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
            let tot: f64 = v.iter().map(|(_, (e, _))| *e).sum();
            eprintln!("[ktime] t={t} 총 {tot:.0}ms");
            for (k, (e, c)) in v.iter().take(14) {
                eprintln!("[ktime] {:22} {:9.1}ms ({}회)", k, e, c);
            }
        }
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
        // 마지막 행 head — b_xn 마지막 행이 이미 output_norm 융합 결과
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.b_xn.ptr.add((t - 1) * n * 4) as *const f32,
                self.m_e.ptr as *mut f32, n);
        }
        if !noba { self.ctx.begin_batch()?; };
        self.gemv_w(self.m_e.buf.clone(), self.m_xq.buf, "output.weight", self.b_lg.buf, 1, n)?;
        if !noba { self.ctx.end_batch_wait()?; } else { self.ctx.flush2()?; };
        let mut logits = vec![0f32; self.n_vocab];
        unsafe { std::ptr::copy_nonoverlapping(self.b_lg.ptr as *const f32, logits.as_mut_ptr(), self.n_vocab) };
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
            self.run_pipe("qk_rope2", QK_ROPE2_SPV, 5, 28,
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
            let push = Self::push_u32s(&[pos as u32, n_head as u32, n_kv as u32, hd as u32]);
            self.run_pipe("qsa_flash", QSA_FLASH_SPV, 4, 16,
                &[self.b_aq.buf, self.m_kv_k[seq].buf, self.m_kv_v[seq].buf, self.b_aout.buf],
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
        self.gemv_w(self.b_fglu.buf.clone(), self.b_xq_f.buf, "blk.64.ffn_down.weight", self.b_fdown.buf, 1, self.n_ff)?;
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
        // 배치 검증 — LLM170_VKD_BATCH=1 명시 옵트인만 (기본 per-token:
        // gemv8 t≥2 검증 배치는 기지 간헐 레이스의 의심 트리거 — plans/36 §8).
        if std::env::var("LLM170_VKD_BATCH").map(|v| v == "1").unwrap_or(false) {
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

/// Engine에 VkDecoder 주입 — raw_names/raw_consts 기반 (server에서 이관, plans/35 P4).
pub fn inject(eng: &mut llm170_core::model::Engine) -> Result<(), String> {
    use llm170_core::matmul::RawDecode;
    let hp = eng.model.hp.clone();
    let is_recr: Vec<bool> = (0..hp.n_layer).map(|il| eng.model.is_recr(il)).collect();
    let (wnames, cnames) = llm170_core::model::rawinject::raw_names(eng);
    let mut weights: Vec<(String, llm170_core::matmul::Weight<'_>)> = Vec::new();
    for n in &wnames {
        let w = eng.model.wchk(n).map_err(|e| e.to_string())?;
        weights.push((n.clone(), w));
    }
    let mut consts = llm170_core::model::rawinject::raw_consts(eng, &cnames);
    // plans/38 A4: qsa_flash가 인과 루프 상한으로 자체 마스킹 — ctx² 마스크
    // 상수(8k=256MB) 업로드·상주 폐지.
    consts.retain(|(k, _)| k != "mask");
    let rd: std::sync::Arc<VkDecoder> = std::sync::Arc::new(VkDecoder::new());
    rd.raw_init(&hp, &weights, &consts, eng.seqs.len(), eng.ctx_len(), is_recr)
        .map_err(|e| format!("raw_init(vk): {e}"))?;
    eng.raw_decode = Some(rd);
    // plans/40: 가중은 이제 VRAM에만 상주 — mmap 클린 페이지를 커널에 반납해
    // 호스트 RSS를 emb/소형 상수 수준으로. LLM170_VK_KEEPW=1이면 유지(CPU 디버그).
    if std::env::var("LLM170_VK_KEEPW").map(|v| v == "1").unwrap_or(false) {
        eprintln!("[vk] 가중 mmap 페이지 유지 (LLM170_VK_KEEPW=1)");
    } else {
        let t0 = std::time::Instant::now();
        let freed = eng.model.discard_weight_pages(&["token_embd.weight", "output.weight", "output_norm.weight"]);
        eprintln!("[vk] 반납 {:.2}GB", freed as f64 / (1 << 30) as f64);
        eprintln!("[vk] 가중 mmap 페이지 반납 완료 ({:.1}s)", t0.elapsed().as_secs_f32());
    }
    Ok(())
}
