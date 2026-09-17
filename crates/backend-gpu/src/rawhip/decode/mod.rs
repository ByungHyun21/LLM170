//! 원시 HIP 디코드 실행기 — 1토큰 스텝을 원시 런치열로 구성 (2026-09-03).
//! frame(qwen35)의 op 순서를 그대로 옮기되 cubecl 프레임(op당 블로킹 제출)을
//! 대체: 영속 버퍼 + 비동기 런치 + 마지막 1회 동기. 수치는 커널 검증
//! 게이트(rawhip-check·미러)를 통과한 산술과 동일.
//!
//! 실측(2026-09-14, 27B Q4_K_XL):
//! - 디코드 스텝 88.6ms 커널 / 40종. 큰 FFN GEMM은 개별 ~179 GB/s, 스텝 전체
//!   ~169 GB/s = DRAM(236)의 71%로 사실상 한계다 — 이 모델의 디코드 여지는 ~10-20%뿐.
//! - 프리필 pp512 = 1,415ms이고 네 GEMM 섹션(ffn_gate/ffn/gdn_mm/proj)이 86%,
//!   ~19.5 TFLOPS = f32 피크의 34%. 호스트는 무죄다(cpu_submit=11.0ms).
//! - 계측기 주의: LLM170_KTRACE 합계(366ms)와 LLM170_NOLAUNCH(1,014ms "호스트
//!   스켈레톤")는 **배치 런치에서 신뢰 불가**하다(벽시계 1,409ms와 모순).
//!   LLM170_PP_PROF 마크는 이 raw 경로 전용이고(프레임 경로는 무출력),
//!   trace 섹션의 95.8ms는 계측기 자신의 hipEventCreate 비용이다.

use cubecl_hip_sys as hip;
use super::RawCtx;
use super::{CO_MMQ, CO_MMQ2, CO_MMQ3, probes};
use llm170_core::matmul::Weight;
use crate::rawhip::{env_on, env_eq};

/// 디코드 상주 상태 — 스텝마다 재사용, 해제 없음.
/// 원시 포인터는 단일 GPU 컨텍스트 소유 — Mutex 직렬화 하 Send 안전.
pub struct DecodeState {
    pub ctx: RawCtx,
    // 활성/중간 버퍼 (f32 바이트)
    pub xs: *mut u8,      // 잔차 스트림 [n_embd]
    pub xn: *mut u8,      // norm 출력 [n_embd]
    pub gqkv: *mut u8,    // in_proj 출력 [conv_ch]
    pub gconv: *mut u8,   // conv 출력 [conv_ch]
    pub gz: *mut u8,      // [d_inner]
    pub gb: *mut u8,      // [dt_rank]
    pub ga: *mut u8,      // [dt_rank]
    pub gbg: *mut u8,     // [dt_rank*2]
    pub gq: *mut u8,      // [k_len]
    pub gk: *mut u8,      // [k_len]
    pub gv: *mut u8,      // [v_len]
    pub go: *mut u8,      // [v_len]
    pub ggated: *mut u8,  // [d_inner]
    pub gout: *mut u8,    // [n_embd]
    pub fgate: *mut u8,   // [n_ff]
    pub fup: *mut u8,     // [n_ff]
    pub fglu: *mut u8,    // [n_ff]
    pub fdown: *mut u8,   // [n_embd]
    pub logits: *mut u8,  // [vocab]
    // q8 통합 버퍼 (워드+d비트)
    pub xq_n: *mut u8,    // (n_embd/4 + n_embd/32)*4
    pub xq_f: *mut u8,    // (n_ff/4 + n_ff/32)*4
    pub xq_g: *mut u8,    // (6144/4 + 6144/32 + 6144/16)*4
    // 어텐션
    pub aq: *mut u8,      // [n_head*2*hd]
    pub ak: *mut u8,      // [n_kv*hd]
    pub av: *mut u8,      // [n_kv*hd]
    pub aout: *mut u8,    // [n_head*hd]
    pub scores: *mut u8,  // [n_head * ctx_len]
    // rms 부분합
    pub p64: *mut u8,     // [rows*32*8] — 최대 행수로
    // 스케일 1.0 상수
    pub one: *mut u8,
    /// 프리필 배치 아레나 (t_max 고정) — 접미사 _t.
    pub b_t_max: usize,
    pub xs_t: *mut u8, xn_t: *mut u8, xq_n_t: *mut u8,
    pub gqkv_t: *mut u8, gz_t: *mut u8, gb_t: *mut u8, ga_t: *mut u8, gbg_t: *mut u8,
    pub gconv_t: *mut u8, gq_t: *mut u8, gk_t: *mut u8, gv_t: *mut u8, go_t: *mut u8,
    pub ggated_t: *mut u8, gout_t: *mut u8, xq_g_t: *mut u8,
    pub fgate_t: *mut u8, fup_t: *mut u8, fglu_t: *mut u8, fdown_t: *mut u8, xq_f_t: *mut u8,
    pub aq_t: *mut u8, ak_t: *mut u8, av_t: *mut u8, aout_t: *mut u8, scores_t: *mut u8,
    pub logits_all: *mut u8, // [t][vocab] — verify_batch head 출력
    pub gdn_snap: *mut u8,
    pub gdn_snap_bytes: usize,
    // MTP (blk.64) — spec draft용 GPU 상주
    pub mtp_on: bool,
    pub mtp_kv_k: Vec<*mut u8>,
    pub mtp_kv_v: Vec<*mut u8>,
    pub mtp_kv_k16: Vec<*mut u8>,
    pub mtp_kv_v16: Vec<*mut u8>,
    pub mtp_cat: *mut u8,   // [2n] enorm‖hnorm
    pub mtp_cur: *mut u8,   // [n] eh_proj 출력/레이어 hidden
    pub mtp_qkv: *mut u8,   // q(2hd×h)‖k‖v t=1
    pub mtp_ao: *mut u8,    // attn out
    // np×spec (plans/18) 행 메타·포인터 테이블
    pub ms_rowseq: *mut u8,
    pub ms_rowpos: *mut u8,
    pub ms_segstart: *mut u8,
    pub ms_segend: *mut u8,
    pub ms_rownp: *mut u8,
    pub ms_ptrbuf: *mut u8,
    pub ms_ptrbuf2: *mut u8, // K/V 테이블 분리 (flash 동시 참조)
    pub mtp_e: *mut u8,     // tok embd 임시
    pub mtp_h: *mut u8,     // h 입력 임시
    pub mtp_xq: *mut u8,    // n quant
    pub mtp_xq2: *mut u8,   // 2n quant (eh_proj)
    // 배치 MTP 프리필 (blk.64를 t행 한 번에) — mtp_prefill_batch 전용
    pub t_max_mtp: usize,    // 배치 MTP 버퍼 행 상한 (t_max와 동일)
    pub mtp_b_e: *mut u8,    // [t_max][n] enorm 출력
    pub mtp_b_hs: *mut u8,   // [t_max][n] hnorm 입력 (h_{p-1} 시프트)
    pub mtp_prefetched: std::sync::atomic::AtomicBool,  // 사이드 h2d 선반입됨
    pub mtp_b_cat: *mut u8,  // [t_max][2n] enorm‖hnorm
    pub mtp_b_cur: *mut u8,  // [t_max][n] hidden
    pub mtp_b_xqn: *mut u8,  // [t_max][xq(n)]
    pub mtp_b_xq2: *mut u8,  // [t_max][xq(2n)]
    // 상수 (norm 가중치·conv·cs 테이블·마스크)
    pub consts: std::collections::HashMap<String, *mut u8>,
    // 가중치 (dev 상주 — 업로드 1회)
    pub weights: std::collections::HashMap<String, (*mut u8, u32, usize, usize)>, // (ptr, ty, n_in, n_out)
    pub ktab2: *mut u8,
    // KV/GDN 상태 [seq][...]
    pub kv_k: Vec<Vec<*mut u8>>,  // [full층][seq]
    pub kv_k16: Vec<Vec<*mut u8>>, // f16 미러(디코드 v_dot2 경로용)
    pub kv_v16: Vec<Vec<*mut u8>>,
    pub kv_v: Vec<Vec<*mut u8>>,
    pub st_conv: Vec<Vec<*mut u8>>,  // [recr층][seq]
    pub st_gdn: Vec<Vec<*mut u8>>,
    // 하이퍼파라미터
    pub n_embd: usize,
    pub n_vocab: usize,
    pub n_vocab_set: bool,
    pub n_ff: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub hd: usize,
    pub n_rot: usize,
    pub eps: f32,
    pub d_inner: usize,
    pub n_group: usize,
    pub dt_rank: usize,
    pub d_state: usize,
    pub conv_k: usize,
    pub conv_ch: usize,
    pub k_len: usize,
    pub v_len: usize,
    pub ctx_len: usize,
    pub kq_scale: f32,
    pub is_recr: Vec<bool>,
}



/// Engine 주입용 RawDecode 구현 — DecodeState를 Mutex로 보관.
pub struct RawDecoder {
    st: std::sync::Mutex<Option<DecodeState>>,
}

impl Default for RawDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl RawDecoder {
    pub fn new() -> Self {
        RawDecoder { st: std::sync::Mutex::new(None) }
    }
}

/// f32 KV → f16 미러 변환 (디코드 어텐션의 v_dot2 경로 전제).
/// off·n 단위는 원소 수. 같은 스트림에 넣으므로 이후 커널이 순서대로 본다.
/// f32 KV 원본이 필요한 경로(덤프 진단)인지. 기본 경로는 f16 미러만 쓴다.
/// plans/78 R6: NO_FLASH/GQA/GQA2/GQA2D 커널 스위치는 폐기 — 이 함수는
/// KV 덤프 진단(weights.rs)을 위해서만 남고, NO_GQA2D=1이 f32 KV를 복원한다.
fn legacy_f32() -> bool {
    env_on("LLM170_NO_GQA2D")
}

fn kv_to_f16(ctx: &RawCtx, src: *mut u8, dst: *mut u8, src_off: usize, dst_off: usize, n: usize) -> Result<(), String> {
    let mut sp = unsafe { (src as *mut f32).add(src_off) } as *mut std::ffi::c_void;
    let mut dp = unsafe { (dst as *mut u16).add(dst_off) } as *mut std::ffi::c_void;
    let mut nn = n as i32;
    let mut args: Vec<*mut std::ffi::c_void> = vec![
        &mut sp as *mut _ as *mut std::ffi::c_void,
        &mut dp as *mut _ as *mut std::ffi::c_void,
        &mut nn as *mut _ as *mut std::ffi::c_void,
    ];
    ctx.launch3("kv_f16", n.div_ceil(1024) as u32, 1, 1, 256, &mut args)
}

impl llm170_core::matmul::RawDecode for RawDecoder {
    fn raw_init(
        &self,
        hp: &llm170_core::qwen35::hparams::Hparams,
        weights: &[(String, llm170_core::matmul::Weight<'_>)],
        consts: &[(String, Vec<f32>)],
        n_seqs: usize,
        ctx_len: usize,
        is_recr: Vec<bool>,
    ) -> Result<(), String> {
        let ctx = RawCtx::new()?;
        let ds = DecodeState::new(ctx, hp, weights, consts, n_seqs, ctx_len, is_recr)?;

        *self.st.lock().map_err(|e| e.to_string())? = Some(ds);
        Ok(())
    }

    fn raw_prefill(&self, seq: usize, pos0: usize, emb: &[f32]) -> Result<Vec<f32>, String> {
        let t0 = std::time::Instant::now();
        if env_on("LLM170_KTRACE") { crate::rawhip::ktrace_on(); }
        let guard = self.st.lock().map_err(|e| e.to_string())?;
        let ds = guard.as_ref().ok_or("raw_decode: 미초기화")?;
        ds.step_batch(seq, pos0, emb)?;
        let r = ds.read_logits();
        if env_on("LLM170_KTRACE") {
            eprintln!("{}", crate::rawhip::ktrace_dump());
        }
        if let (Some(path), Ok(v)) = (std::env::var_os("LLM170_DUMP_LOGITS"), r.as_ref()) {
            let _ = std::fs::write(&path, bytemuck::cast_slice(v));
        }
        if env_on("LLM170_RAWHIP_TIMING") {
            eprintln!("batch({} tok) wall={:.1}ms", emb.len() / ds.n_embd, t0.elapsed().as_secs_f64() * 1e3);
        }
                r
    }

    /// MTP 훅용 프리필: 로짓 + **마지막 행** hidden(체인 carry)만 반환한다
    /// (전행 d2h 10.5MB/chunk 제거 — MTP는 디바이스 xs_t를 직접 읽는다).
    /// MTP 임베딩 선반입: 사이드 스트림 async h2d (메인 프리필과 중첩).
    fn mtp_upload_tok_emb(&self, tok_flat: &[f32]) -> Result<(), String> {
        let guard = self.st.lock().map_err(|e| e.to_string())?;
        let ds = guard.as_ref().ok_or("raw_decode: 미초기화")?;
        ds.ctx.h2d_async_s(ds.mtp_b_e, bytemuck::cast_slice(tok_flat))?;
        ds.mtp_prefetched.store(true, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    fn raw_prefill_h(
        &self,
        seq: usize,
        pos0: usize,
        emb: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let t0 = std::time::Instant::now();
        if env_on("LLM170_KTRACE") { crate::rawhip::ktrace_on(); }
        let guard = self.st.lock().map_err(|e| e.to_string())?;
        let ds = guard.as_ref().ok_or("raw_decode: 미초기화")?;
        ds.step_batch(seq, pos0, emb)?;
        // 진단: 프리필 후 MTP KV가 채워졌는지 (비영 검사)
        if env_on("LLM170_DUMP_MTPKV") {
            let nw = 4096usize; // 앞 16KB
            let mut kv = vec![0f32; nw];
            if let Some(buf) = ds.mtp_kv_k.get(seq).copied() {
                let _ = ds.ctx.d2h(bytemuck::cast_slice_mut(&mut kv).as_mut(), buf);
            }
            let h = pos0 + emb.len() / ds.n_embd;
            eprintln!(
                "# mtpkv seq={seq} pos0={pos0} rows={h} first16KB: nonzero={} max={:.4}",
                kv.iter().filter(|v| **v != 0.0).count(),
                kv.iter().fold(0f32, |a, b| a.max(b.abs()))
            );
        }

        // 마지막 행 최종 hidden d2h (MTP carry 전용 — 전행 회수 제거)
        let t = emb.len() / ds.n_embd;
        let mut h_last = vec![0f32; ds.n_embd];
        let last_row = unsafe { ds.xs_t.add((t - 1) * ds.n_embd * 4) };
        ds.ctx.d2h(bytemuck::cast_slice_mut(&mut h_last).as_mut(), last_row)?;
        let r = ds.read_logits();
        if env_on("LLM170_KTRACE") {
            eprintln!("{}", crate::rawhip::ktrace_dump());
        }
        if env_on("LLM170_RAWHIP_TIMING") {
            eprintln!("batch_h({} tok) wall={:.1}ms", t, t0.elapsed().as_secs_f64() * 1e3);
        }
                Ok((r?, h_last))
    }

    fn raw_step_h(
        &self,
        seq: usize,
        pos: usize,
        emb: &[f32],
        h_out: &mut Vec<f32>,
    ) -> Result<Vec<f32>, String> {
        let guard = self.st.lock().map_err(|e| e.to_string())?;
        let ds = guard.as_ref().ok_or("raw_decode: 미초기화")?;
        ds.ctx.h2d(ds.xs, bytemuck::cast_slice(emb))?;
        ds.step(seq, pos)?;
        ds.read_hidden(h_out)?;
        ds.read_logits()
    }

    fn raw_verify(
        &self,
        seq: usize,
        pos0: usize,
        emb: &[f32],
        argmaxes: &mut Vec<u32>,
        h_all: &mut Vec<f32>,
    ) -> Result<(), String> {
        let guard = self.st.lock().map_err(|e| e.to_string())?;
        let ds = guard.as_ref().ok_or("raw_decode: 미초기화")?;
        let t_rv0 = std::time::Instant::now();
        ds.verify_batch(seq, pos0, emb, argmaxes)?;
        if env_on("LLM170_SPEC_TIMING") {
            eprintln!("[rv] verify_batch={:.1}ms", t_rv0.elapsed().as_secs_f64() * 1e3);
        }
        // 행별 최종 hidden export (MTP 상태 진행용) — xs_t에 step_batch 결과 잔존
        let t = emb.len() / ds.n_embd;
        h_all.clear();
        h_all.resize(t * ds.n_embd, 0.0);
        let t_h0 = std::time::Instant::now();
        ds.ctx
            .d2h(bytemuck::cast_slice_mut(h_all).as_mut(), ds.xs_t)?;
        if env_on("LLM170_SPEC_TIMING") {
            eprintln!("[rv] h_all d2h={:.1}ms", t_h0.elapsed().as_secs_f64() * 1e3);
        }
        Ok(())
    }

    fn verify_batch_ms(
        &self,
        seqs: &[usize],
        poss: &[usize],
        group_starts: &[usize],
        emb: &[f32],
        argmaxes: &mut Vec<u32>,
        h_all: &mut Vec<f32>,
    ) -> Result<(), String> {
        let guard = self.st.lock().map_err(|e| e.to_string())?;
        guard
            .as_ref()
            .ok_or("raw_decode: 미초기화")?
            .verify_batch_ms(seqs, poss, group_starts, emb, argmaxes, h_all)
    }

    fn raw_step_multi(
        &self,
        seqs: &[usize],
        poss: &[u32],
        emb: &[f32],
    ) -> Result<Vec<Vec<f32>>, String> {
        let guard = self.st.lock().map_err(|e| e.to_string())?;
        let ds = guard.as_ref().ok_or("raw_decode: 미초기화")?;
        if env_on("LLM170_KTRACE") { crate::rawhip::ktrace_on(); }
        let r = ds.step_batch_np(seqs, poss, emb);
        if env_on("LLM170_KTRACE") {
            eprintln!("{}", crate::rawhip::ktrace_dump());
        }
        r
    }

    fn mtp_step_chain(&self, seq: usize, tok_emb: &[f32], pos: usize) -> Result<u32, String> {
        let guard = self.st.lock().map_err(|e| e.to_string())?;
        guard
            .as_ref()
            .ok_or("raw_decode: 미초기화")?
            .mtp_step_chain(seq, tok_emb, pos)
    }

    fn mtp_step_adv(
        &self,
        seq: usize,
        tok_emb: &[f32],
        h: &[f32],
        pos: usize,
    ) -> Result<(), String> {
        let guard = self.st.lock().map_err(|e| e.to_string())?;
        guard
            .as_ref()
            .ok_or("raw_decode: 미초기화")?
            .mtp_step_adv(seq, tok_emb, h, pos)
    }

    /// np greedy — step_batch_np_greedy 위임 (logits 전사 회피).
    fn raw_step_multi_greedy(
        &self,
        seqs: &[usize],
        poss: &[u32],
        emb: &[f32],
    ) -> Result<Vec<u32>, String> {
        let guard = self.st.lock().map_err(|e| e.to_string())?;
        let ds = guard.as_ref().ok_or("raw_decode: 미초기화")?;
        if env_on("LLM170_KTRACE") {
            crate::rawhip::ktrace_on();
        }
        let r = ds.step_batch_np_greedy(seqs, poss, emb);
        if env_on("LLM170_KTRACE") {
            eprintln!("{}", crate::rawhip::ktrace_dump());
        }
        r
    }

    fn mtp_step_gpu(
        &self,
        seq: usize,
        tok_emb: &[f32],
        h: &[f32],
        pos: usize,
    ) -> Result<(u32, Vec<f32>), String> {
        let guard = self.st.lock().map_err(|e| e.to_string())?;
        guard
            .as_ref()
            .ok_or("raw_decode: 미초기화")?
            .mtp_step_gpu(seq, tok_emb, h, pos)
    }

    fn gdn_snapshot(&self) -> Result<(), String> {
        let guard = self.st.lock().map_err(|e| e.to_string())?;
        guard.as_ref().ok_or("raw_decode: 미초기화")?.gdn_snapshot()
    }

    fn raw_reset(&self, seq: usize) -> Result<(), String> {
        let guard = self.st.lock().map_err(|e| e.to_string())?;
        guard.as_ref().ok_or("raw_decode: 미초기화")?.reset_seq_state(seq)
    }

    fn gdn_restore(&self) -> Result<(), String> {
        let guard = self.st.lock().map_err(|e| e.to_string())?;
        guard.as_ref().ok_or("raw_decode: 미초기화")?.gdn_restore()
    }

    fn mtp_head_argmax(&self, h_normed: &[f32]) -> Result<u32, String> {
        let guard = self.st.lock().map_err(|e| e.to_string())?;
        let ds = guard.as_ref().ok_or("raw_decode: 미초기화")?;
        ds.mtp_head_argmax(h_normed)
    }

    /// MTP 프리필 배치 (HIP) — blk.64를 t행 한 번에.
    fn mtp_prefill_batch(
        &self,
        seq: usize,
        tok_embs: &[f32],
        carry_h: &[f32],
        t: usize,
        pos0: usize,
        with_head: bool,
    ) -> Result<u32, String> {
        let guard = self.st.lock().map_err(|e| e.to_string())?;
        let ds = guard.as_ref().ok_or("raw_decode: 미초기화")?;
        ds.mtp_prefill_batch(seq, tok_embs, carry_h, t, pos0, with_head)
    }

    /// MTP KV 적립 전용 — 헤드 생략 시 전체 vocab GEMV(953MB 읽기)를 건너뛴다.
    fn mtp_step_hidden(
        &self,
        seq: usize,
        tok_emb: &[f32],
        h: &[f32],
        pos: usize,
        with_head: bool,
    ) -> Result<Option<u32>, String> {
        let guard = self.st.lock().map_err(|e| e.to_string())?;
        let ds = guard.as_ref().ok_or("raw_decode: 미초기화")?;
        ds.ctx.h2d(ds.mtp_h, bytemuck::cast_slice(h))?;
        ds.mtp_step_g(seq, tok_emb, ds.mtp_h, pos, with_head)
    }

    fn tile_big_chunk(&self) -> bool {
        // 미초기화면 알 수 없다 — 원래 전역 비트 기준으로 초기화 후에만 호출된다.
        self.st
            .lock()
            .ok()
            .and_then(|g| g.as_ref().map(|ds| ds.ctx.co_loaded(super::CO_J128)))
            .unwrap_or(false)
    }

    fn raw_step(&self, seq: usize, pos: usize, emb: &[f32]) -> Result<Vec<f32>, String> {
        let t0 = std::time::Instant::now();
        if env_on("LLM170_KTRACE") { crate::rawhip::ktrace_on(); }
        let guard = self.st.lock().map_err(|e| e.to_string())?;
        let ds = guard.as_ref().ok_or("raw_decode: 미초기화")?;
        ds.ctx.h2d(ds.xs, bytemuck::cast_slice(emb))?;
        ds.step(seq, pos)?;
        let r = ds.read_logits();
        if env_on("LLM170_KTRACE") {
            eprintln!("{}", crate::rawhip::ktrace_dump());
        }
        if env_on("LLM170_RAWHIP_TIMING") {
            eprintln!("step cpu={:.2}ms", t0.elapsed().as_secs_f64() * 1e3);
        }
        r
    }
}

unsafe impl Send for DecodeState {}

impl DecodeState {
    /// 시퀀스 상태 제로화 (서버 슬롯 반환) — GDN/conv만 (KV는 위치 색인).
    pub fn reset_seq_state(&self, seq: usize) -> Result<(), String> {
        let gl = self.dt_rank * self.d_state * self.d_state;
        let cl = (self.conv_k - 1) * self.conv_ch;
        let zg = vec![0u8; gl * 4];
        let zc = vec![0u8; cl * 4];
        for r in 0..self.st_gdn.len() {
            if seq < self.st_gdn[r].len() {
                self.ctx.h2d(self.st_gdn[r][seq], &zg)?;
                self.ctx.h2d(self.st_conv[r][seq], &zc)?;
            }
        }
        Ok(())
    }

    pub fn gdn_snapshot(&self) -> Result<(), String> {
        let (gdn_len, conv_len) = (self.gdn_len(), self.conv_len());
        let mut off = 0usize;
        for v in &self.st_gdn {
            for b in v.iter() {
                self.copy(*b, self.gdn_snap, 0, off, gdn_len)?;
                off += gdn_len;
            }
        }
        for v in &self.st_conv {
            for b in v.iter() {
                self.copy(*b, self.gdn_snap, 0, off, conv_len)?;
                off += conv_len;
            }
        }
        Ok(())
    }

    /// 스냅샷 복원.
    pub fn gdn_restore(&self) -> Result<(), String> {
        let (gdn_len, conv_len) = (self.gdn_len(), self.conv_len());
        let mut off = 0usize;
        for v in &self.st_gdn {
            for b in v.iter() {
                self.copy(self.gdn_snap, *b, off, 0, gdn_len)?;
                off += gdn_len;
            }
        }
        for v in &self.st_conv {
            for b in v.iter() {
                self.copy(self.gdn_snap, *b, off, 0, conv_len)?;
                off += conv_len;
            }
        }
        Ok(())
    }

    /// 상태 길이 — copy_rows는 float 단위 (원소 수).
    fn gdn_len(&self) -> usize {
        self.dt_rank * self.d_state * self.d_state
    }
    fn conv_len(&self) -> usize {
        (self.conv_k - 1) * self.conv_ch
    }

    /// GPU argmax — 최저 인덱스 동률 (CPU greedy와 동일 의미). 토큰만 회수.
    pub fn argmax_token(&self) -> Result<u32, String> {
        let out = self.ctx.scratch(16)?;
        let mut xp = self.logits as *mut std::ffi::c_void;
        let mut n = self.n_vocab as i32;
        let mut op = out as *mut std::ffi::c_void;
        let mut args = vec![
            (&mut xp) as *mut _ as *mut std::ffi::c_void,
            (&mut n) as *mut _ as *mut std::ffi::c_void,
            (&mut op) as *mut _ as *mut std::ffi::c_void,
        ];
        self.ctx.launch("argmax64", 1, 1, 64, &mut args)?;
        self.ctx.sync()?;
        let mut r = [0u8; 8];
        self.ctx.d2h(&mut r, out)?;
        let idx = i32::from_le_bytes([r[4], r[5], r[6], r[7]]);
        Ok(idx as u32)
    }

    /// [t][row_f32] logits의 행별 GPU argmax — 토큰만 회수 (np greedy/MTP verify).
    /// 행당 1블록(64레인) 발사 후 8·t 바이트 단일 d2h(d2h가 스트림을 동기화).
    pub fn argmax_rows(&self, base: *mut u8, t: usize, row_f32: usize) -> Result<Vec<u32>, String> {
        // 병렬 2단계 (plans/74 N3) — argmax64 1블록 판은 vocab 248k 에서
        // ~1.2ms/행의 직렬 꼬리였다(npt4 KTRACE 4.8ms/step).
        let nblk = (row_f32 / 4096).clamp(1, 64) as u32;
        let part = self.ctx.scratch(t.max(1) * nblk as usize * 8)?;
        let outb = self.ctx.scratch(t.max(1) * 8 + 8 * nblk as usize * t.max(1))?;
        // out 을 part 와 다른 크기 슬롯에: scratch 는 크기 키 슬롯0 재사용.
        let out = unsafe { outb.add(t.max(1) * nblk as usize * 8) };
        {
            let mut xp = base as *mut std::ffi::c_void;
            let mut vb = row_f32 as i32;
            let mut pp = part as *mut std::ffi::c_void;
            let mut nb = nblk as i32;
            let mut args = vec![
                (&mut xp) as *mut _ as *mut std::ffi::c_void,
                (&mut vb) as *mut _ as *mut std::ffi::c_void,
                (&mut pp) as *mut _ as *mut std::ffi::c_void,
                (&mut nb) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx.launch3("argmax_rows_s1", nblk, t.max(1) as u32, 1, 256, &mut args)?;
        }
        {
            let mut pp = part as *mut std::ffi::c_void;
            let mut op = out as *mut std::ffi::c_void;
            let mut nb = nblk as i32;
            let mut args = vec![
                (&mut pp) as *mut _ as *mut std::ffi::c_void,
                (&mut op) as *mut _ as *mut std::ffi::c_void,
                (&mut nb) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx.launch3("argmax_rows_s2", t.max(1) as u32, 1, 1, 64, &mut args)?;
        }
        let mut r8 = vec![0u8; t * 8];
        self.ctx.d2h(&mut r8, out)?;
        Ok((0..t)
            .map(|s| {
                let b = &r8[s * 8..s * 8 + 8];
                u32::from_le_bytes([b[4], b[5], b[6], b[7]])
            })
            .collect())
    }

    /// 배치 rms — rows=t.
    fn rms_rows(&self, x: *mut u8, w: *mut u8, out: *mut u8, n: usize, t: usize) -> Result<(), String> {
        let mut xp = x as *mut std::ffi::c_void;
        let mut pp = self.p64 as *mut std::ffi::c_void;
        let mut na = n as i32;
        let mut a1 = vec![Self::p(&mut xp), Self::p(&mut pp), Self::p(&mut na)];
        self.ctx.launch("rms_part", t as u32, 1, 32, &mut a1)?;
        let mut wp = w as *mut std::ffi::c_void;
        let mut op = out as *mut std::ffi::c_void;
        let mut ep = self.eps;
        let mut wr = 1i32;
        let mut a2 = vec![Self::p(&mut xp), Self::p(&mut wp), Self::p(&mut pp), Self::p(&mut op), Self::p(&mut ep), Self::p(&mut na), Self::p(&mut wr)];
        self.ctx.launch("rms_finish", t as u32, 1, 256, &mut a2)
    }

    /// 배치 GEMV — xq [t][xq_w], out [t][n_out].
    #[allow(clippy::too_many_arguments)]
    /// mm_b의 f32 병행판 — q4_K/q5_K MMQ 경로 (하니스 검증 plans/27 부록5·14).
    fn mm_b2(&self, y_f32: *mut u8, xq: *mut u8, xq_w: usize, wp: *mut u8, ty: u32, n_in: usize, n_out: usize, out: *mut u8, t: usize) -> Result<(), String> {
        let only = { let _t = std::time::Instant::now(); std::env::var("LLM170_MMQ_ONLY").ok().and_then(|v| v.parse::<u32>().ok()) };
        if !env_on("LLM170_NO_MMQ") || only.is_some() {
            // plans/73 우선순위 수정: && 가 || 보다 먼저 결합해 좌변(K계열)이
            // t 게이트·CO 검사를 **우회**했다 — step_batch 의 verify(t=4~16)가
            // 전부 MMQ 로 돌아 296ms/4행 (3.9x, 무계약)을 낸 근원. 게이트가
            // 양쪽 분기 모두에 적용되도록 괄호 명시.
            if (((only.is_none() || only.is_some_and(|m| m & (1u32 << (ty - 12)) != 0)) && matches!(ty, 12 | 13 | 14 | 23))
                || ((ty == 8 && env_eq("LLM170_Q8MMQ", "1")) && (ty != 14 || !env_on("LLM170_NO_Q6MMQ"))))
                && (t >= 32 || (t == 1 && env_on("LLM170_Q1MMQ")))
                && self.ctx.co_loaded(super::CO_MMQ | super::CO_MMQ2 | super::CO_MMQ3) {
                        return self.ctx.gemm_mmq(ty, y_f32 as *const u8, wp as *const u8, n_in, n_out, t, out);
            }
            // q6_K: dequant→f16 v4 타일 (llama dequant+MFMA 경로 대응, 부록42)
            if ty == 14 && t >= 32 && self.ctx.co_loaded(super::CO_MMQ2)
                && env_on("LLM170_DEQ16") {
                return self.ctx.gemm_f16_q6(y_f32 as *const u8, wp as *const u8, n_in, n_out, t, out);
            }
        }
        self.mm_b(xq, xq_w, wp, ty, n_in, n_out, out, t)
    }

    /// plans/28 디버그: 버퍼의 행별 L1 노름 덤프 (지연 게이트) — 수치 오염 행 탐지.
    fn trace_rows(&self, label: &str, ptr: *const u8, row_f32: usize, t: usize) -> Result<(), String> {
        if !env_on("LLM170_MS_TRACE") {
            return Ok(());
        }
        let mut buf = vec![0f32; row_f32 * t];
        self.ctx.sync()?;
        self.ctx.d2h(bytemuck::cast_slice_mut(&mut buf).as_mut(), ptr)?;
        let norms: Vec<String> = (0..t)
            .map(|r| {
                let s: f64 = buf[r * row_f32..(r + 1) * row_f32]
                    .iter()
                    .map(|&v| v.abs() as f64)
                    .sum();
                format!("{s:.3}")
            })
            .collect();
        eprintln!("[mst] {label}: {}", norms.join(" "));
        Ok(())
    }

    fn mm_b(&self, xq: *mut u8, xq_w: usize, wp: *mut u8, ty: u32, n_in: usize, n_out: usize, out: *mut u8, t: usize) -> Result<(), String> {
        // np 소형 배치(t=2..4): 4-토큰 GEMV — 타일은 128열 고정이라 t=4에서
        // 124열을 낭비한다(t=4 0.34ms vs t=128 0.83ms, 동일 가중). 가중 1회
        // 독서로 토큰별 독립 누산. LLM170_NO_G4=1로 끔.
        if (2..=4).contains(&t)
            && matches!(ty, 12 | 13 | 14 | 23)
           
        {
            return self.ctx.gemm_g4(
                ty,
                xq as *const u8,
                wp as *const u8,
                self.ktab2 as *const u8,
                n_in,
                n_out,
                xq_w,
                t,
                out,
            );
        }
        // q5_K v2 (부록76): vdr=2 그리드-스트라이드 — 자체 스트림 (비트계약 아님)
        if ty == 13 && t == 1 && env_on("LLM170_Q5V2") {
            return self.ctx.gemv_q8_out_v2(xq as *const u8, wp as *const u8, ty, n_in, n_out, out, xq_w, t);
        }
        // 홀수 타입 타일 (plans/04): odd CO + t>=32에서만
        let odd_v4 = !env_on("LLM170_EXACT")
            && self.ctx.co_loaded(super::CO_ODD) && t >= 32
            && matches!(ty, 20 | 11 | 21);
        // q8_0 타일 (j128): 소형 GEMV 토큰당 재독 제거
        let q8t = ty == 8 && t > 64 && (n_out >= 128 || t >= 256) && !env_on("LLM170_EXACT")
            && self.ctx.co_loaded(super::CO_J128);
        if (matches!(ty, 12 | 13 | 14 | 23) && t > 1 || odd_v4 || q8t) {
            // 타일 경로 — 가중 1회 독서 (블록=1행, TT 토큰 레지스터)
            return self.ctx.gemm_tile(xq as *const u8, wp as *const u8, self.ktab2 as *const u8, ty, n_in, n_out, xq_w, t, out);
        }
        self.ctx.gemv_q8_out(xq as *const u8, wp as *const u8, self.ktab2 as *const u8, ty, n_in, n_out, out, xq_w, t)
    }
    /// mm_b 사이드 스트림판 — 타일형만 (비타일은 주 스트림 사용)
    #[allow(clippy::too_many_arguments)]
    /// mm_b_s의 MMQ판 — side stream에서 quant+mul_mat_q (부록48).
    fn mm_b2_s(&self, y_f32: *mut u8, xq: *mut u8, xq_w: usize, wp: *mut u8, ty: u32, n_in: usize, n_out: usize, out: *mut u8, t: usize) -> Result<(), String> {
        if matches!(ty, 12 | 13 | 23) && t >= 32
            && self.ctx.co_loaded(super::CO_MMQ | super::CO_MMQ2 | super::CO_MMQ3) {
            return self.ctx.gemm_mmq_s(ty, y_f32 as *const u8, wp as *const u8, n_in, n_out, t, out);
        }
        self.mm_b_s(xq, xq_w, wp, ty, n_in, n_out, out, t)
    }

    fn mm_b_s(&self, xq: *mut u8, xq_w: usize, wp: *mut u8, ty: u32, n_in: usize, n_out: usize, out: *mut u8, t: usize) -> Result<(), String> {
        self.ctx.gemm_tile_s(xq as *const u8, wp as *const u8, self.ktab2 as *const u8, ty, n_in, n_out, xq_w, t, out)
    }

}

/// Engine에 원시 HIP 디코더 주입 — 필요 가중치·상수 전체를 백엔드로.
/// (plans/28: 단계 타이밍 계측 추가 — 공존 지연 RCA용. server에서 이관 plans/35 P4)
pub fn inject(eng: &mut llm170_core::qwen35::Engine) -> Result<(), String> {
    let t0 = std::time::Instant::now();
    let hp = eng.model.hp.clone();
    let (wnames, cnames): (Vec<String>, Vec<String>) =
        llm170_core::qwen35::rawinject::raw_names(eng);
    let is_recr: Vec<bool> = (0..hp.n_layer).map(|il| eng.model.is_recr(il)).collect();
    let t1 = std::time::Instant::now();
    let weights: Vec<(String, llm170_core::matmul::Weight<'_>)> = wnames
        .iter()
        .filter_map(|k| eng.model.wchk(k).ok().map(|w| (k.clone(), w)))
        .collect();
    if weights.len() != wnames.len() {
        return Err(format!("rawhip: 가중치 누락 {}/{}", weights.len(), wnames.len()));
    }
    eprintln!(
        "# inject: names+weights {:.1?} ({} tensors, {:.2}GB)",
        t1.elapsed(),
        weights.len(),
        weights.iter().map(|(_, w)| w.data.len()).sum::<usize>() as f64 / (1 << 30) as f64
    );
    let consts = llm170_core::qwen35::rawinject::raw_consts(eng, &cnames);
    eprintln!("# inject: consts @+{:.1?}", t0.elapsed());
    let rd: std::sync::Arc<RawDecoder> = std::sync::Arc::new(RawDecoder::new());
    use llm170_core::matmul::RawDecode;
    let r = rd
        .raw_init(&hp, &weights, &consts, eng.seqs.len(), eng.ctx_len(), is_recr)
        .map_err(|e| format!("raw_init: {e}"));
    eprintln!("# inject: raw_init @+{:.1?}", t0.elapsed());
    r?;
    eng.raw_decode = Some(rd);
    Ok(())
}


mod np;
mod spec;
mod step;
mod weights;
