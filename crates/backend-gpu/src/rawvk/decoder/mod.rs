//! VkDecoder — GDN/어텐션 GPU 상주 디코드 (plans/19 2단계).
//! 커널 8종은 gdn-check ★ 검증 완료. 기존 gemv/quant/rms/silu SPIR-V 재사용.
//! rawhip DecodeState 대칭 — 배치 모드(단일 제출+배리어).

use crate::rawvk::context::{Pipes, VkBuf, VkCtx};

mod gemv;
mod spec;
mod step;
mod weights;
use ash::vk;
use std::collections::HashMap;

const GDN_CONV_SPV: &[u8] = include_bytes!("../spv/gdn_conv_t.spv");
const GEMV8_Q5_SPV: &[u8] = include_bytes!("../spv/gemv8_q5.spv");
const GEMV8_Q4_SPV: &[u8] = include_bytes!("../spv/gemv8_q4.spv");
const GEMV8_XS_SPV: &[u8] = include_bytes!("../spv/gemv8_xs.spv");
const GEMV8_Q3_SPV: &[u8] = include_bytes!("../spv/gemv8_q3.spv");
const GEMV8_Q3B_SPV: &[u8] = include_bytes!("../spv/gemv8_q3b.spv");
const GEMV8_Q6_SPV: &[u8] = include_bytes!("../spv/gemv8_q6.spv");
const GEMV8_Q8_SPV: &[u8] = include_bytes!("../spv/gemv8_q8.spv");
const GEMV8_Q5B_SPV: &[u8] = include_bytes!("../spv/gemv8_q5b.spv");
const GEMV8_Q5N_SPV: &[u8] = include_bytes!("../spv/gemv8_q5n.spv");
const GEMV8_NLB_SPV: &[u8] = include_bytes!("../spv/gemv8_nlb.spv");
const GEMV8_I3S_SPV: &[u8] = include_bytes!("../spv/gemv8_i3s.spv");
const GDN_ARF_SPV: &[u8] = include_bytes!("../spv/gdn_arf.spv");
const GDN_AR8F_SPV: &[u8] = include_bytes!("../spv/gdn_ar8f.spv");
const TILE_MS4GY_F16B_SPV: &[u8] = include_bytes!("../spv/tile_ms4gy_f16b.spv");
const QUANT_F16_SPV: &[u8] = include_bytes!("../spv/quant_f16.spv");
const GEMV8_Q4B_SPV: &[u8] = include_bytes!("../spv/gemv8_q4b.spv");
const GEMV8_Q6B_SPV: &[u8] = include_bytes!("../spv/gemv8_q6b.spv");
const GEMV8_Q8B_SPV: &[u8] = include_bytes!("../spv/gemv8_q8b.spv");
const GEMV8_XSB_SPV: &[u8] = include_bytes!("../spv/gemv8_xsb.spv");
const TILE_Q6K_SPV: &[u8] = include_bytes!("../spv/tile_q6k.spv");
const GDN_CONV_STATE_SPV: &[u8] = include_bytes!("../spv/gdn_conv_state.spv");
const SPLIT3_SPV: &[u8] = include_bytes!("../spv/split3.spv");
const L2_SPV: &[u8] = include_bytes!("../spv/l2_rows2.spv");
const BETA_G_SPV: &[u8] = include_bytes!("../spv/gdn_beta_g.spv");
const GDN_AR_SPV: &[u8] = include_bytes!("../spv/gdn_ar.spv");
const GDN_AR4_SPV: &[u8] = include_bytes!("../spv/gdn_ar4.spv");
const GDN_AR8_SPV: &[u8] = include_bytes!("../spv/gdn_ar8.spv");
const NORM_GATED_SPV: &[u8] = include_bytes!("../spv/norm_gated.spv");
const QK_ROPE2_SPV: &[u8] = include_bytes!("../spv/qk_rope2.spv");
const KV_APPEND_SPV: &[u8] = include_bytes!("../spv/kv_append.spv");
pub const QSA_FLASH_SPV: &[u8] = include_bytes!("../spv/qsa_flash.spv");
pub const QSA_FLASH_GQ_SPV: &[u8] = include_bytes!("../spv/qsa_flash_gq.spv");
const KV_APPEND_Q8_SPV: &[u8] = include_bytes!("../spv/kv_append_q8.spv");
const QSA_FLASH_Q8_SPV: &[u8] = include_bytes!("../spv/qsa_flash_q8.spv");
const COPY_OFF_SPV: &[u8] = include_bytes!("../spv/copy_off.spv");
const TILE128_SPV: &[u8] = include_bytes!("../spv/tile128_q5k.spv");
const TILE_XS_SPV: &[u8] = include_bytes!("../spv/tile_xs.spv");
const TILE_Q8_SPV: &[u8] = include_bytes!("../spv/tile_q8.spv");
const TILE_Q4K_SPV: &[u8] = include_bytes!("../spv/tile_q4k.spv");
const GEMM_I8_SPV: &[u8] = include_bytes!("../spv/gemm_i8.spv");
const QUANT_B8_SPV: &[u8] = include_bytes!("../spv/quant_b8.spv");
const QUANT_B8V2_SPV: &[u8] = include_bytes!("../spv/quant_b8v2.spv");
const GEMM_I8V2_SPV: &[u8] = include_bytes!("../spv/gemm_i8v2.spv");
const ADDRMS_SPV: &[u8] = include_bytes!("../spv/addrms.spv");
const TILE_NL_SPV: &[u8] = include_bytes!("../spv/tile_nl.spv");
const TILE_IQ3S_SPV: &[u8] = include_bytes!("../spv/tile_iq3s.spv");
const TILE_F16_SPV: &[u8] = include_bytes!("../spv/tile_f16.spv");
const TILE128O_SPV: &[u8] = include_bytes!("../spv/tile128o.spv");
const TILE_MS4GY_SPV: &[u8] = include_bytes!("../spv/tile_ms4gy.spv");
const TILE_XS128_SPV: &[u8] = include_bytes!("../spv/tile_xs128.spv");
const TILE_Q4K128_SPV: &[u8] = include_bytes!("../spv/tile_q4k128.spv");
const TILE_Q6K128_SPV: &[u8] = include_bytes!("../spv/tile_q6k128.spv");
const TILE_Q3K128_SPV: &[u8] = include_bytes!("../spv/tile_q3k128.spv");
const TILE_Q8128_SPV: &[u8] = include_bytes!("../spv/tile_q8128.spv");
const TILE_NL128_SPV: &[u8] = include_bytes!("../spv/tile_nl128.spv");
const TILE_Q4KMS_SPV: &[u8] = include_bytes!("../spv/tile_q4kms.spv");
const TILE_Q6KMS_SPV: &[u8] = include_bytes!("../spv/tile_q6kms.spv");
const TILE_Q3KMS_SPV: &[u8] = include_bytes!("../spv/tile_q3kms.spv");
const TILE_Q8MS_SPV: &[u8] = include_bytes!("../spv/tile_q8ms.spv");
const TILE_XSMS_SPV: &[u8] = include_bytes!("../spv/tile_xsms.spv");
const TILE_NLMS_SPV: &[u8] = include_bytes!("../spv/tile_nlms.spv");
const TILE_MS128_SPV: &[u8] = include_bytes!("../spv/tile_ms128.spv");
/// plans/91 P0 — np 배치 디코드: 행별 상태(링/AR/KV) 디바이스 주소 테이블 패밀리.
const GDN_CONV_NP_SPV: &[u8] = include_bytes!("../spv/gdn_conv_np.spv");
const GDN_ARF_NP_SPV: &[u8] = include_bytes!("../spv/gdn_arf_np.spv");
const QK_ROPE2_NP_SPV: &[u8] = include_bytes!("../spv/qk_rope2_np.spv");
const KV_APP_NP_SPV: &[u8] = include_bytes!("../spv/kv_app_np.spv");
const QSA_FLASH_NP_SPV: &[u8] = include_bytes!("../spv/qsa_flash_np.spv");
/// plans/91 P0 — np 배치 dmmv: WG가 t토큰 전체(가중 1회 판독). gemv8_*b 와
/// 행×토큰 비트 동일.
const GEMV8T_Q5_SPV: &[u8] = include_bytes!("../spv/gemv8t_q5.spv");
const GEMV8T_Q4_SPV: &[u8] = include_bytes!("../spv/gemv8t_q4.spv");
const GEMV8T_NL_SPV: &[u8] = include_bytes!("../spv/gemv8t_nl.spv");
const GEMV8T_Q6_SPV: &[u8] = include_bytes!("../spv/gemv8t_q6.spv");
const GEMV8T_Q3_SPV: &[u8] = include_bytes!("../spv/gemv8t_q3.spv");
const GEMV8T_Q8_SPV: &[u8] = include_bytes!("../spv/gemv8t_q8.spv");
const GEMV8T_XS_SPV: &[u8] = include_bytes!("../spv/gemv8t_xs.spv");
/// plans/91 P2 — MTP 프리필 배치 유틸(rawhip row_shift_gather/cat2_rows 미러).
const ROW_SHIFT_GATHER_SPV: &[u8] = include_bytes!("../spv/row_shift_gather.spv");
const CAT2_ROWS_SPV: &[u8] = include_bytes!("../spv/cat2_rows.spv");

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
    eps: f32,
    kq_scale: f32,
    max_ssbo: usize,
    #[allow(clippy::type_complexity)]
    ktimes: std::collections::HashMap<String, (f64, u64)>,
    ktime: bool,
    dbg_drain_ms: f64,
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
    b_lg: VkBuf,  // head 로짓 [n_vocab] — b_gout 오버플로 수정 (T_MAX*n < vocab)
    b_ams: VkBuf, // argmax 스테이지1 스크래치 [2*256] u32
    b_xf16: VkBuf, // f16-B 활성 [T_MAX*n] f16
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
    // ── plans/91 P2 — MTP 프리필 배치 버퍼 (blk.64 t행 1패스, T_MAX 상한).
    m_be: VkBuf,    // [T][n] 토큰 임베딩 선반입 / hnorm 임시
    m_bcur: VkBuf,  // [T][n] MTP hidden
    m_bhs: VkBuf,   // [T][n] h_shift (디바이스 조립)
    m_bcat: VkBuf,  // [T][2n] enorm‖hnorm
    m_bxq2: VkBuf,  // [T][xq(2n)]
    m_bxqn: VkBuf,  // [T][xq(n)]
    m_prefetched: std::sync::atomic::AtomicBool,
    // ── plans/91 P0 — np 배치: 상태 주소 테이블([그룹][슬롯] u64, 생성 후
    // 불변)·행별 pos/slot 맵(스텝당 호스트 기입)·greedy 행별 argmax 스크래치.
    np_conv_tbl: VkBuf,
    np_gdn_tbl: VkBuf,
    np_kvk_tbl: VkBuf,
    np_kv_v_tbl: VkBuf,
    np_pos: VkBuf,   // [n_seqs] u32 — 행 pos
    np_slot: VkBuf,  // [n_seqs] u32 — 행→슬롯
    b_amsc: VkBuf,   // [2*am_wg*T_MAX] u32 — 행별 argmax 스테이지1
    b_amr: VkBuf,    // [T_MAX] u32 — 행별 argmax 결과
}

unsafe impl Send for DecoderState {}
unsafe impl Sync for DecoderState {}

impl llm170_core::matmul::RawDecode for VkDecoder {
    fn raw_init(
        &self,
        hp: &llm170_core::qwen35::hparams::Hparams,
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

    /// greedy 스텝 — GPU argmax로 토큰만 회수 (608KB 로짓 전사·CPU 스캔 회피).
    fn raw_step_greedy(&self, seq: usize, pos: usize, emb: &[f32]) -> Result<u32, String> {
        let mut guard = self.st.lock().map_err(|e| e.to_string())?;
        let ds = guard.as_mut().ok_or("vkdecoder: 미초기화")?;
        ds.step_core(seq, pos, emb)?;
        ds.lg_argmax()
    }

    /// 프리필 — 행[t][n_embd]별 t=1 스텝 (기본 구현의 512-float 청크 절단 결함 회피).
    /// t=1 스텝 산술은 p1 검증 경로와 동일 — 순차 상태 적립으로 수치 불변.
    /// prefill 청크 게이트 (plans/41): ms 타일 패밀리는 t=512 단일 패스가
    /// 정확(tokens 불변 실측)하고 가중 판독이 1회로 줄어 빠름 — 512 승인.
    /// 구 패밀리 옵트아웃(MSALL=0) 시에는 64 유지.
    fn tile_big_chunk(&self) -> bool {
        std::env::var("LLM170_TILE_MSALL").map(|v| v != "0").unwrap_or(true)
    }

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
            let tw = std::time::Instant::now();
            last = Some(ds.step_batch(seq, pos0 + off, ch, false)?);
            if std::env::var_os("LLM170_DBG_WALL").is_some() {
                eprintln!("#  batch t={} wall={:.1}ms", ch.len() / n, tw.elapsed().as_secs_f64() * 1e3);
            }
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

    /// raw_prefill + 마지막 행 hidden (MTP carry).
    fn raw_prefill_h(
        &self,
        seq: usize,
        pos0: usize,
        emb: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let mut guard = self.st.lock().map_err(|e| e.to_string())?;
        let ds = guard.as_mut().ok_or("vkdecoder: 미초기화")?;
        let mut h_all: Vec<f32> = Vec::new();
        let lg = ds.verify_rows(seq, pos0, emb, &mut Vec::new(), &mut h_all)?;
        let n = ds.n_embd;
        let t = emb.len() / n;
        let h_last = if h_all.len() >= t * n { h_all[(t - 1) * n..].to_vec() } else { vec![0f32; n] };
        Ok((lg, h_last))
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

    /// np 배치 디코드 (plans/91 P0) — t행 단일 패스 (무게 1회 판독, 상태커널은
    /// 행별 상태 주소 테이블). 종전 seq별 순차 step 루프는 t배 비용이었다.
    /// 행별 산술은 순차 루프와 비트 동일 (step_batch_np_ex 문서 참조).
    fn raw_step_multi(
        &self,
        seqs: &[usize],
        poss: &[u32],
        emb: &[f32],
    ) -> Result<Vec<Vec<f32>>, String> {
        let mut guard = self.st.lock().map_err(|e| e.to_string())?;
        let ds = guard.as_mut().ok_or("vkdecoder: 미초기화")?;
        ds.step_batch_np_ex(seqs, poss, emb, false).map(|(l, _)| l)
    }

    /// np greedy 배치 — GPU 행별 argmax로 토큰만 회수 (로짓 전사 회피).
    fn raw_step_multi_greedy(
        &self,
        seqs: &[usize],
        poss: &[u32],
        emb: &[f32],
    ) -> Result<Vec<u32>, String> {
        let mut guard = self.st.lock().map_err(|e| e.to_string())?;
        let ds = guard.as_mut().ok_or("vkdecoder: 미초기화")?;
        ds.step_batch_np_ex(seqs, poss, emb, true).map(|(_, t)| t)
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

    /// MTP 임베딩 선반입 (plans/91 P2) — m_be 매핑 ptr 직접 기입.
    /// hip의 사이드 스트림 h2d 중첩은 없다(동기 기입) — 프리필 청크당
    /// 수 ms 수준, 원장에 기록.
    fn mtp_upload_tok_emb(&self, tok_flat: &[f32]) -> Result<(), String> {
        let mut guard = self.st.lock().map_err(|e| e.to_string())?;
        let ds = guard.as_mut().ok_or("vkdecoder: 미초기화")?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                tok_flat.as_ptr(),
                ds.m_be.ptr as *mut f32,
                tok_flat.len().min(T_MAX * ds.n_embd),
            );
        }
        ds.m_prefetched.store(true, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    /// MTP 프리필 배치 (plans/91 P2) — blk.64 t행 1패스(KV-only 최적화).
    fn mtp_prefill_batch(
        &self,
        seq: usize,
        tok_embs: &[f32],
        carry_h: &[f32],
        t: usize,
        pos0: usize,
        with_head: bool,
    ) -> Result<u32, String> {
        let mut guard = self.st.lock().map_err(|e| e.to_string())?;
        let ds = guard.as_mut().ok_or("vkdecoder: 미초기화")?;
        ds.mtp_prefill_batch_ex(seq, tok_embs, carry_h, t, pos0, with_head)
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

    /// per-seq GDN/conv 복원 (np×spec 부분수용).
    fn gdn_restore_seq(&self, seq: usize, _n_seqs: usize) -> Result<(), String> {
        let mut guard = self.st.lock().map_err(|e| e.to_string())?;
        guard.as_mut().ok_or("vkdecoder: 미초기화")?.restore_seq_states(seq)
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

impl Default for VkDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl VkDecoder {
    pub fn new() -> Self {
        Self {
            st: std::sync::Mutex::new(None),
        }
    }
}

const T_MAX: usize = 512;   // plans/41: 단일 패스 프리필 (가중 1회 판독)


fn n_group_len(hp: &llm170_core::qwen35::hparams::Hparams) -> usize {
    hp.n_group * hp.d_state
}

/// Engine에 VkDecoder 주입 — raw_names/raw_consts 기반 (server에서 이관, plans/35 P4).
pub fn inject(eng: &mut llm170_core::qwen35::Engine) -> Result<(), String> {
    use llm170_core::matmul::RawDecode;
    let hp = eng.model.hp.clone();
    let is_recr: Vec<bool> = (0..hp.n_layer).map(|il| eng.model.is_recr(il)).collect();
    let (wnames, cnames) = llm170_core::qwen35::rawinject::raw_names(eng);
    let mut weights: Vec<(String, llm170_core::matmul::Weight<'_>)> = Vec::new();
    for n in &wnames {
        let w = eng.model.wchk(n).map_err(|e| e.to_string())?;
        weights.push((n.clone(), w));
    }
    let mut consts = llm170_core::qwen35::rawinject::raw_consts(eng, &cnames);
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
