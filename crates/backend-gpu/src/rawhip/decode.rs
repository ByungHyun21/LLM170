//! 원시 HIP 디코드 실행기 — 1토큰 스텝을 원시 런치열로 구성 (2026-09-03).
//! frame35의 op 순서를 그대로 옮기되 cubecl 프레임(op당 블로킹 제출)을
//! 대체: 영속 버퍼 + 비동기 런치 + 마지막 1회 동기. 수치는 커널 검증
//! 게이트(rawhip-check·미러)를 통과한 산술과 동일.

use cubecl_hip_sys as hip;
use super::RawCtx;
use llm170_core::matmul::Weight;

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

impl DecodeState {
    fn a(ctx: &RawCtx, bytes: usize) -> Result<*mut u8, String> {
        ctx.alloc(bytes).map_err(|e| e.to_string())
    }

    /// 모델에서 상주 상태 구축 — 가중치 업로드 1회.
    pub fn new(
        ctx: RawCtx,
        hp: &llm170_core::model::hparams::Hparams,
        weights: &[(String, Weight<'_>)],
        consts: &[(String, Vec<f32>)],
        n_seqs: usize,
        ctx_len: usize,
        is_recr: Vec<bool>,
    ) -> Result<Self, String> {

        let (n, n_ff) = (hp.n_embd, hp.n_ff);
        let (d_inner, conv_ch) = (hp.d_inner, hp.conv_ch());
        let (k_len, v_len) = (hp.n_group * hp.d_state, hp.dt_rank * hp.d_state);
        let g6 = hp.n_head.max(hp.n_kv) * hp.head_dim; // ggated·aout 길이 상한
        let mut c = std::collections::HashMap::new();
        for (k, v) in consts {
            let d = ctx.alloc(v.len() * 4).map_err(|e| e.to_string())?;
            ctx.h2d(d, bytemuck::cast_slice(v))?;
            c.insert(k.clone(), d);
        }
        let mut wmap = std::collections::HashMap::new();
        for (k, w) in weights {
            let d = ctx.alloc(w.data.len()).map_err(|e| e.to_string())?;
            ctx.h2d(d, w.data)?;
            wmap.insert(k.clone(), (d, w.ty as u32, w.n_in as usize, w.n_out as usize));
        }
        let ktab2: Vec<u32> = llm170_core::ktab2_packed();
        let kt = ctx.alloc(1024).map_err(|e| e.to_string())?;
        ctx.h2d(kt, bytemuck::cast_slice(&ktab2))?;
        let one = ctx.alloc(4).map_err(|e| e.to_string())?;
        ctx.h2d(one, bytemuck::cast_slice(&[1.0f32]))?;
        // KV·GDN 상태
        let kv_len = ctx_len * hp.n_kv * hp.head_dim;
        let conv_len = (hp.conv_k - 1) * conv_ch;
        let gdn_len = hp.dt_rank * hp.d_state * hp.d_state;
        let n_recr = is_recr.iter().filter(|&&r| r).count();
        let n_full = is_recr.iter().filter(|&&r| !r).count();
        let mut st_conv = Vec::with_capacity(n_recr);
        let mut st_gdn = Vec::with_capacity(n_recr);
        let zero_k = vec![0f32; kv_len];
        let mut kv_k = Vec::with_capacity(n_full);
        let mut kv_v = Vec::with_capacity(n_full);
        for _ in 0..n_full {
            let mut ck = Vec::with_capacity(n_seqs);
            let mut cv2 = Vec::with_capacity(n_seqs);
            for s in 0..n_seqs {
                ck.push(ctx.alloc(kv_len * 4).map_err(|e| e.to_string())?);
                cv2.push(ctx.alloc(kv_len * 4).map_err(|e| e.to_string())?);
                ctx.h2d(ck[s], bytemuck::cast_slice(&zero_k))?;
                ctx.h2d(cv2[s], bytemuck::cast_slice(&zero_k))?;
            }
            kv_k.push(ck);
            kv_v.push(cv2);
        }
        let zero_conv = vec![0f32; conv_len];
        let zero_gdn = vec![0f32; gdn_len];
        for _ in 0..n_recr {
            let mut cv = Vec::with_capacity(n_seqs);
            let mut gd = Vec::with_capacity(n_seqs);
            for s in 0..n_seqs {
                cv.push(ctx.alloc(conv_len * 4).map_err(|e| e.to_string())?);
                gd.push(ctx.alloc(gdn_len * 4).map_err(|e| e.to_string())?);
                ctx.h2d(cv[s], bytemuck::cast_slice(&zero_conv))?;
                ctx.h2d(gd[s], bytemuck::cast_slice(&zero_gdn))?;
            }
            st_conv.push(cv);
            st_gdn.push(gd);
        }
        let max_rows = (n / 32).max(n_ff.max(g6)).max(hp.n_head + hp.n_kv).max(1);
        let bs = |nb: usize| Self::a(&ctx, nb).unwrap();
        // 배치 아레나 — 기본 청크 512 (z-그리드). LLM170_CHUNK≤128이면 128.
        let t_max = std::env::var("LLM170_CHUNK").ok().and_then(|v| v.parse::<usize>().ok())
            .map(|c| if c > 128 { 512 } else { 128 }).unwrap_or(512);
        let (n_kv, hd) = (hp.n_kv, hp.head_dim);
        let xq_sn = n / 4 + n / 32 + n / 16;
        let xq_sf = n_ff / 4 + n_ff / 32 + n_ff / 16;
        let xq_sg = d_inner / 4 + d_inner / 32 + d_inner / 16;
        let b_xs_t = bs(t_max * n * 4);
        let b_xn_t = bs(t_max * n * 4);
        let b_xq_n_t = bs(t_max * xq_sn * 4);
        let b_gqkv_t = bs(t_max * conv_ch * 4);
        let b_gz_t = bs(t_max * d_inner * 4);
        let b_gb_t = bs(t_max * hp.dt_rank * 4);
        let b_ga_t = bs(t_max * hp.dt_rank * 4);
        let b_gbg_t = bs(t_max * hp.dt_rank * 2 * 4);
        let b_gconv_t = bs(t_max * conv_ch * 4);
        let b_gq_t = bs(t_max * k_len * 4);
        let b_gk_t = bs(t_max * k_len * 4);
        let b_gv_t = bs(t_max * v_len * 4);
        let b_go_t = bs(t_max * v_len * 4);
        let b_ggated_t = bs(t_max * d_inner * 4);
        let b_gout_t = bs(t_max * n * 4);
        let b_xq_g_t = bs(t_max * xq_sg * 4);
        let b_fgate_t = bs(t_max * n_ff * 4);
        let b_fup_t = bs(t_max * n_ff * 4);
        let b_fglu_t = bs(t_max * n_ff * 4);
        let b_fdown_t = bs(t_max * n * 4);
        let b_xq_f_t = bs(t_max * xq_sf * 4);
        let b_aq_t = bs(t_max * hp.n_head * 2 * hp.head_dim * 4);
        let b_ak_t = bs(t_max * n_kv * hd * 4);
        let b_av_t = bs(t_max * n_kv * hd * 4);
        let b_aout_t = bs(t_max * hp.n_head * hp.head_dim * 4);
        let b_scores_t = bs(t_max * hp.n_head * ctx_len * 4);
        // 최대 64행 — carried 재실행 행 포함 (np8×k4=40)
        let b_logits_all = ctx.alloc(hp.vocab * 4 * 64).map_err(|e| e.to_string())?;
        // GDN 상태 스냅샷 (spec 부분수용 롤백용) — recr 전층 (gdn_s + conv) × seq
        let gdn_bytes: usize = (n_recr * n_seqs * (gdn_len + conv_len)) * 4;
        let b_gdn_snap = ctx.alloc(gdn_bytes.max(16)).map_err(|e| e.to_string())?;
        // MTP층 (가중치 존재 시)
        let mtp_on = weights
            .iter()
            .any(|(k, _)| k == "blk.64.nextn.eh_proj.weight");
        let n_ao = hp.n_head * hp.head_dim; // wo 입력 (6144 > n)
        let b_ms_meta = ctx.alloc(320 * 4).map_err(|e| e.to_string())?; // i32 5×64
        let b_ms_ptr = ctx.alloc(64 * 8 * 2).map_err(|e| e.to_string())?; // K/V 테이블 2×64행
        let b_mtp_xq_sz = (n_ao / 4 + n_ao / 32 + n_ao / 16) * 4;
        let b_mtp_xq2_sz = (2 * n / 4 + 2 * n / 32 + 2 * n / 16) * 4;
        let (mut v_mtp_k, mut v_mtp_v, b_mtp_cat, b_mtp_cur, b_mtp_qkv, b_mtp_ao, b_mtp_e, b_mtp_h, b_mtp_xq, b_mtp_xq2) = if mtp_on {
            let mut vk = Vec::with_capacity(n_seqs);
            let mut vv = Vec::with_capacity(n_seqs);
            for _ in 0..n_seqs {
                vk.push(ctx.alloc(kv_len * 4).map_err(|e| e.to_string())?);
                vv.push(ctx.alloc(kv_len * 4).map_err(|e| e.to_string())?);
            }
            (
                vk, vv,
                ctx.alloc(n * 2 * 4).map_err(|e| e.to_string())?,
                ctx.alloc(n * 4).map_err(|e| e.to_string())?,
                ctx.alloc((hp.n_head * 2 * hp.head_dim + hp.n_kv * hp.head_dim * 2) * 4)
                    .map_err(|e| e.to_string())?,
                ctx.alloc(hp.n_head * hp.head_dim * 4).map_err(|e| e.to_string())?,
                ctx.alloc(n * 4).map_err(|e| e.to_string())?,
                ctx.alloc(n * 4).map_err(|e| e.to_string())?,
                ctx.alloc(b_mtp_xq_sz).map_err(|e| e.to_string())?,
                ctx.alloc(b_mtp_xq2_sz).map_err(|e| e.to_string())?,
            )
        } else {
            (Vec::new(), Vec::new(), std::ptr::null_mut(),
             std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(),
             std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut())
        };
        // 배치 MTP용 q8 폭 — n(정규화 입력)과 n_head*hd(attn_output 입력) 중 큰 쪽.
        let b_xq_m = n.max(hp.n_head * hp.head_dim);
        let b_xq_n_sz = (b_xq_m / 4 + b_xq_m / 32 + b_xq_m / 16) * 4;
        let (b_mtp_be, b_mtp_bhs, b_mtp_bcat, b_mtp_bcur, b_mtp_bxqn, b_mtp_bxq2) = if mtp_on {
            (
                ctx.alloc(t_max * n * 4).map_err(|e| e.to_string())?,
                ctx.alloc(t_max * n * 4).map_err(|e| e.to_string())?,
                ctx.alloc(t_max * 2 * n * 4).map_err(|e| e.to_string())?,
                ctx.alloc(t_max * n * 4).map_err(|e| e.to_string())?,
                ctx.alloc(t_max * b_xq_n_sz).map_err(|e| e.to_string())?,
                ctx.alloc(t_max * {
                    // eh_proj(2n)와 ffn_down 입력(n_ff) 중 큰 쪽
                    let l = (2 * n).max(hp.n_ff);
                    (l / 4 + l / 32 + l / 16) * 4
                }).map_err(|e| e.to_string())?,
            )
        } else {
            (std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(),
             std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut())
        };
        if mtp_on {
            // KV 0 초기화 (h2d zeros)
            let zk = vec![0u8; kv_len * 4];
            for &p in v_mtp_k.iter() { ctx.h2d(p, &zk)?; }
            for &p in v_mtp_v.iter() { ctx.h2d(p, &zk)?; }
        }
        let (b_xs, b_xn, b_gqkv, b_gconv) = (bs(n * 4), bs(n * 4), bs(conv_ch * 4), bs(conv_ch * 4));
        let (b_gz, b_gb, b_ga, b_gbg) = (bs(d_inner * 4), bs(hp.dt_rank * 4), bs(hp.dt_rank * 4), bs(hp.dt_rank * 2 * 4));
        let (b_gq, b_gk, b_gv, b_go) = (bs(k_len * 4), bs(k_len * 4), bs(v_len * 4), bs(v_len * 4));
        let (b_ggated, b_gout) = (bs(d_inner * 4), bs(n * 4));
        let (b_fgate, b_fup, b_fglu, b_fdown) = (bs(n_ff * 4), bs(n_ff * 4), bs(n_ff * 4), bs(n * 4));
        let b_logits = bs(hp.vocab * 4);
        let (b_xqn, b_xqf, b_xqg) = (bs((n / 4 + n / 32 + n / 16) * 4), bs((n_ff / 4 + n_ff / 32 + n_ff / 16) * 4), bs((g6 / 4 + g6 / 32 + g6 / 16) * 4));
        let (b_aq, b_ak, b_av) = (bs(hp.n_head * 2 * hp.head_dim * 4), bs(hp.n_kv * hp.head_dim * 4), bs(hp.n_kv * hp.head_dim * 4));
        let (b_aout, b_scores, b_p64) = (bs(hp.n_head * hp.head_dim * 4), bs(hp.n_head * ctx_len * 4), bs(max_rows * 32 * 8));
        let ds = DecodeState {
            ctx,
            xs: b_xs, xn: b_xn, gqkv: b_gqkv, gconv: b_gconv, gz: b_gz, gb: b_gb,
            ga: b_ga, gbg: b_gbg, gq: b_gq, gk: b_gk,
            gv: b_gv, go: b_go, ggated: b_ggated, gout: b_gout,
            fgate: b_fgate, fup: b_fup, fglu: b_fglu, fdown: b_fdown,
            logits: b_logits, xq_n: b_xqn, xq_f: b_xqf, xq_g: b_xqg,
            aq: b_aq, ak: b_ak, av: b_av, aout: b_aout,
            scores: b_scores, p64: b_p64,
            one, consts: c, weights: wmap, ktab2: kt, n_vocab: hp.vocab as usize, n_vocab_set: true,
            b_t_max: t_max,
            xs_t: b_xs_t, xn_t: b_xn_t, xq_n_t: b_xq_n_t,
            gqkv_t: b_gqkv_t, gz_t: b_gz_t, gb_t: b_gb_t, ga_t: b_ga_t, gbg_t: b_gbg_t,
            gconv_t: b_gconv_t, gq_t: b_gq_t, gk_t: b_gk_t, gv_t: b_gv_t, go_t: b_go_t,
            ggated_t: b_ggated_t, gout_t: b_gout_t, xq_g_t: b_xq_g_t,
            fgate_t: b_fgate_t, fup_t: b_fup_t, fglu_t: b_fglu_t, fdown_t: b_fdown_t, xq_f_t: b_xq_f_t,
            aq_t: b_aq_t, ak_t: b_ak_t, av_t: b_av_t, aout_t: b_aout_t, scores_t: b_scores_t,
            logits_all: b_logits_all,
            gdn_snap: b_gdn_snap,
            gdn_snap_bytes: gdn_bytes,
            mtp_on,
            mtp_kv_k: std::mem::take(&mut v_mtp_k),
            mtp_kv_v: std::mem::take(&mut v_mtp_v),
            mtp_cat: b_mtp_cat,
            mtp_cur: b_mtp_cur,
            mtp_qkv: b_mtp_qkv,
            mtp_ao: b_mtp_ao,
            ms_rowseq: b_ms_meta,
            ms_rowpos: unsafe { b_ms_meta.add(64 * 4) },
            ms_segstart: unsafe { b_ms_meta.add(128 * 4) },
            ms_segend: unsafe { b_ms_meta.add(192 * 4) },
            ms_rownp: unsafe { b_ms_meta.add(256 * 4) },
            ms_ptrbuf: b_ms_ptr,
            ms_ptrbuf2: unsafe { b_ms_ptr.add(32 * 8) },
            mtp_e: b_mtp_e,
            mtp_h: b_mtp_h,
            mtp_xq: b_mtp_xq,
            mtp_xq2: b_mtp_xq2,
            t_max_mtp: t_max,
            mtp_b_e: b_mtp_be,
            mtp_b_hs: b_mtp_bhs,
            mtp_prefetched: std::sync::atomic::AtomicBool::new(false),
            mtp_b_cat: b_mtp_bcat,
            mtp_b_cur: b_mtp_bcur,
            mtp_b_xqn: b_mtp_bxqn,
            mtp_b_xq2: b_mtp_bxq2,
            kv_k, kv_v, st_conv, st_gdn,
            n_embd: n, n_ff, n_layer: hp.n_layer, n_head: hp.n_head, n_kv: hp.n_kv,
            hd: hp.head_dim, n_rot: hp.n_rot, eps: hp.eps, d_inner, n_group: hp.n_group,
            dt_rank: hp.dt_rank, d_state: hp.d_state, conv_k: hp.conv_k, conv_ch,
            k_len, v_len, ctx_len, kq_scale: hp.kq_scale(), is_recr,
        };
        Ok(ds)
    }
}

impl DecodeState {
    /// GEMV를 상주 out에 직접 기록 (gemv_q8의 내부 out을 복사 없이 쓰기 위해
    /// out 포인터를 받는 변형이 필요 — 현재는 gemv 후 d2h→h2d. 최적화 후술.)
    fn mm_into(&self, xq: *mut u8, wp: *mut u8, ty: u32, n_in: usize, n_out: usize, out: *mut u8) -> Result<(), String> {
        self.ctx.gemv_q8_out(xq as *const u8, wp as *const u8, self.ktab2 as *const u8, ty, n_in, n_out, out, n_in / 4 + n_in / 32 + n_in / 16, 1)
    }
    /// 듀얼 텐서 q5_K GEMV — 같은 xq·같은 포맷 독립 2 GEMV를 1런치로.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn mm_into2_q8(&self, xq: *mut u8, w1: *mut u8, no1: usize, out1: *mut u8, w2: *mut u8, no2: usize, out2: *mut u8, ni: usize) -> Result<(), String> {
        let gy = (no1 + no2).min(65535) as u32;
        let gz = (no1 + no2).div_ceil(65535) as u32;
        let mut xq_p = xq as *mut std::ffi::c_void;
        let mut w1p = w1 as *mut std::ffi::c_void;
        let mut w2p = w2 as *mut std::ffi::c_void;
        let mut o1 = out1 as *mut std::ffi::c_void;
        let mut o2 = out2 as *mut std::ffi::c_void;
        let mut ni_a = ni as i32;
        let mut no1a = no1 as i32;
        let mut no2a = no2 as i32;
        let mut xw = (ni / 4 + ni / 32 + ni / 16) as i32;
        let mut args = vec![
            Self::p(&mut xq_p), Self::p(&mut w1p), Self::p(&mut w2p), Self::p(&mut o1), Self::p(&mut o2),
            Self::p(&mut ni_a), Self::p(&mut no1a), Self::p(&mut no2a), Self::p(&mut xw),
        ];
        self.ctx.launch3("gemm_q8_0_dual", 1, gy, gz, 64, &mut args)
    }

    fn mm_into2_q5k(&self, xq: *mut u8, w1: *mut u8, no1: usize, out1: *mut u8,
                    w2: *mut u8, no2: usize, out2: *mut u8, n_in: usize) -> Result<(), String> {
        let mut xp = xq as *mut std::ffi::c_void;
        let mut w1p = w1 as *mut std::ffi::c_void;
        let mut w2p = w2 as *mut std::ffi::c_void;
        let mut o1p = out1 as *mut std::ffi::c_void;
        let mut o2p = out2 as *mut std::ffi::c_void;
        let mut ni = n_in as i32;
        let mut n1 = no1 as i32;
        let mut n2 = no2 as i32;
        let mut xw = (n_in / 4 + n_in / 32 + n_in / 16) as i32;
        let mut args = vec![
            &mut xp as *mut _ as *mut std::ffi::c_void,
            &mut w1p as *mut _ as *mut std::ffi::c_void,
            &mut w2p as *mut _ as *mut std::ffi::c_void,
            &mut o1p as *mut _ as *mut std::ffi::c_void,
            &mut o2p as *mut _ as *mut std::ffi::c_void,
            &mut ni as *mut _ as *mut std::ffi::c_void,
            &mut n1 as *mut _ as *mut std::ffi::c_void,
            &mut n2 as *mut _ as *mut std::ffi::c_void,
            &mut xw as *mut _ as *mut std::ffi::c_void,
        ];
        let tot = no1 + no2;
        let gy = tot.min(65535) as u32;
        let gz = tot.div_ceil(65535) as u32;
        self.ctx.launch3("gemm_q5k2", 1, gy, gz, 64, &mut args)
    }

    /// 사이드 스트림 GEMV — side_wait_main 선행 + join2 후속이 계약.
    fn mm_into_s(&self, xq: *mut u8, wp: *mut u8, ty: u32, n_in: usize, n_out: usize, out: *mut u8) -> Result<(), String> {
        self.ctx.gemv_q8_out_s(xq as *const u8, wp as *const u8, self.ktab2 as *const u8, ty, n_in, n_out, out, n_in / 4 + n_in / 32 + n_in / 16, 1)
    }
    /// gemv_q8_out과 동일 인자를 직접 launch — q6k/q4k/q5k/q8 단일행.
    fn mm_direct(&self, xq: *mut u8, wp: *mut u8, ty: u32, n_in: usize, n_out: usize, out: *mut u8) -> Result<(), String> {
        let kern: &'static str = match ty {
            13 => "gemm_q5k",
            8 => "gemm_q8_0",
            12 => "gemm_q4k",
            14 => "gemm_q6k",
            _ => return self.mm_into(xq, wp, ty, n_in, n_out, out),
        };
        let mut xp = xq as *mut std::ffi::c_void;
        let mut wp2 = wp as *mut std::ffi::c_void;
        let mut pp = self.ctx.scratch(n_out * 64 * 8)? as *mut std::ffi::c_void;
        let mut op = out as *mut std::ffi::c_void;
        let mut ni_a = n_in as i32;
        let mut no_a = n_out as i32;
        let mut xw_a = (n_in / 4 + n_in / 32 + n_in / 16) as i32;
        let mut args = vec![Self::p(&mut xp), Self::p(&mut wp2), Self::p(&mut pp),
            Self::p(&mut op), Self::p(&mut ni_a), Self::p(&mut no_a), Self::p(&mut xw_a)];
        let gy = n_out.min(65535) as u32;
        let gz = n_out.div_ceil(65535) as u32;
        self.ctx.launch3(kern, 1, gy, gz, 64, &mut args)
    }

    fn ew_l(&self, name: &str, n: usize, args: &mut [*mut std::ffi::c_void]) -> Result<(), String> {
        self.ctx.launch(name, n.div_ceil(64) as u32, 1, 64, args)
    }
    fn p<T>(v: &mut T) -> *mut std::ffi::c_void {
        v as *mut T as *mut std::ffi::c_void
    }
    fn rms(&self, x: *mut u8, w: *mut u8, out: *mut u8, n: usize) -> Result<(), String> {
        let mut xp = x as *mut std::ffi::c_void;
        let mut pp = self.p64 as *mut std::ffi::c_void;
        let mut na = n as i32;
        let mut a1 = vec![Self::p(&mut xp), Self::p(&mut pp), Self::p(&mut na)];
        self.ctx.launch("rms_part", 1, 1, 32, &mut a1)?;
        let mut wp = w as *mut std::ffi::c_void;
        let mut op = out as *mut std::ffi::c_void;
        let mut ep = self.eps;
        let mut wr = 1i32;
        let mut a2 = vec![
            Self::p(&mut xp), Self::p(&mut wp), Self::p(&mut pp),
            Self::p(&mut op), Self::p(&mut ep), Self::p(&mut na), Self::p(&mut wr),
        ];
        self.ctx.launch("rms_finish", 1, 1, 256, &mut a2)
    }
    fn quant(&self, x: *mut u8, xq: *mut u8, n: usize) -> Result<(), String> {
        self.ctx.quant_q8(x as *const u8, xq, n)
    }
    /// rms+quant 융합 (t=1, n%1024==0) — 3런치 1런치. 산술 미러 동일열.
    fn rms_quant(&self, x: *mut u8, w: *mut u8, xq: *mut u8, n: usize) -> Result<(), String> {
        if n % 1024 != 0 || std::env::var_os("LLM170_RMSQ_SPLIT").is_some() {
            self.rms(x, w, self.xn, n)?;
            return self.quant(self.xn, xq, n);
        }
        let mut xp = x as *mut std::ffi::c_void;
        let mut wp = w as *mut std::ffi::c_void;
        let mut qp = xq as *mut std::ffi::c_void;
        let mut ep = self.eps;
        let mut na = n as i32;
        let mut args = vec![Self::p(&mut xp), Self::p(&mut wp), Self::p(&mut qp), Self::p(&mut ep), Self::p(&mut na)];
        // 블록당 32-블록 1개 (감축은 스레드 0..31) — 32 미만이면 감축만.
        let blk = (n >> 5).clamp(32, 1024) as u32;
        self.ctx.launch("rmsq", 1, 1, blk, &mut args)
    }
    fn axpy(&self, y: *mut u8, x: *mut u8, n: usize) -> Result<(), String> {
        let mut yp = y as *mut std::ffi::c_void;
        let mut xp = x as *mut std::ffi::c_void;
        let mut op = self.one as *mut std::ffi::c_void;
        let mut na = n as i32;
        let mut args = vec![Self::p(&mut yp), Self::p(&mut xp), Self::p(&mut op), Self::p(&mut na)];
        self.ew_l("axpy_scaled", n, &mut args)
    }
    fn copy(&self, src: *mut u8, dst: *mut u8, src_off: usize, dst_off: usize, n: usize) -> Result<(), String> {
        let mut sp = src as *mut std::ffi::c_void;
        let mut dp = dst as *mut std::ffi::c_void;
        let mut so = src_off as i32;
        let mut doff = dst_off as i32;
        let mut na = n as i32;
        let mut args = vec![Self::p(&mut sp), Self::p(&mut dp), Self::p(&mut so), Self::p(&mut doff), Self::p(&mut na)];
        self.ew_l("copy_rows", n, &mut args)
    }
    fn w(&self, key: &str) -> Result<(*mut u8, u32, usize, usize), String> {
        self.weights.get(key).copied().ok_or_else(|| format!("weight 없음: {key}"))
    }

    /// mm_b2/mm_b2_s가 f32 y를 직접 소비하는 경로(MMQ/q6 DEQ16)를 택하는가.
    /// 라우팅 조건은 mm_b2와 동일해야 한다 — 어긋나면 활성 q8을 건너뛴 쪽이
    /// 미초기화 버퍼를 읽는다.
    fn mmq_used(&self, ty: u32, t: usize) -> bool {
        if !matches!(ty, 12 | 13 | 14 | 23) {
            return false;
        }
        let only = std::env::var("LLM170_MMQ_ONLY").ok().and_then(|v| v.parse::<u32>().ok());
        if std::env::var_os("LLM170_NO_MMQ").is_some() && only.is_none() {
            return false;
        }
        if let Some(m) = only {
            if m & (1u32 << (ty - 12)) == 0 {
                return false;
            }
        }
        if ty == 14 && std::env::var_os("LLM170_NO_Q6MMQ").is_some() {
            // q6_K 킬스위치: 타일 경로(활성 q8 소비). DEQ16만 f32 직소비.
            return std::env::var_os("LLM170_DEQ16").is_some()
                && t >= 32
                && super::co_loaded(super::CO_MMQ2);
        }
        (t >= 32 || (t == 1 && std::env::var_os("LLM170_Q1MMQ").is_some()))
            && super::co_loaded(super::CO_MMQ | super::CO_MMQ2 | super::CO_MMQ3)
    }

    /// 사이드 스트림(mm_b2_s → gemm_mmq_s)의 f32 직소비 여부 — 조건 미러.
    /// (gemm_mmq_s는 ty14를 다루지 않는다.)
    fn mmq_used_s(&self, ty: u32, t: usize) -> bool {
        matches!(ty, 12 | 13 | 23)
            && t >= 32
            && std::env::var_os("LLM170_NO_MMQ").is_none()
            && std::env::var_os("LLM170_NO_MMQ_S").is_none()
            && super::co_loaded(super::CO_MMQ | super::CO_MMQ2 | super::CO_MMQ3)
    }

    /// 지정 가중치들이 모두 f32 직소비 경로면 활성 quant를 생략할 수 있다.
    fn grp_mmq(&self, names: &[String], t: usize) -> bool {
        let r = names.iter().all(|n| {
            self.weights
                .get(n)
                .map_or(false, |&(_, ty, _, _)| self.mmq_used(ty, t) && self.mmq_used_s(ty, t))
        });
        if std::env::var_os("LLM170_QSKIP_DBG").is_some() {
            eprintln!("# qskip t={t} n={} -> {r}", names.len());
        }
        r
    }

    /// 디코드 1스텝 (t=1, 단일 시퀀스) — logits 반환. xs에 임베딩 h2d 완료 전제.
    #[allow(clippy::too_many_lines)]
    pub fn step(&self, seq: usize, pos: usize) -> Result<Vec<f32>, String> {
        let t0 = std::time::Instant::now();
        let _ = &t0;

        let n = self.n_embd;
        let (k_len, v_len, conv_ch) = (self.k_len, self.v_len, self.conv_ch);
        let (n_head, n_kv, hd, n_rot) = (self.n_head, self.n_kv, self.hd, self.n_rot);
        let mut full_idx = 0usize;
        let mut recr_idx = 0usize;
        for il in 0..self.n_layer {
            if std::env::var_os("LLM170_RAWHIP_TRACE").is_some() {
                eprintln!("# rawhip: layer {il} (recr={})", self.is_recr[il]);
            }
            // pre-norm + quant
            let wn = *self.consts.get(&format!("blk.{il}.attn_norm")).ok_or("attn_norm")?;
            self.rms_quant(self.xs, wn, self.xq_n, n)?;
            if self.is_recr[il] {
                // in_proj 4종
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_qkv.weight"))?;
                let (wg2, tg2, nig2, nog2) = self.w(&format!("blk.{il}.attn_gate.weight"))?;
                let (wb2, tb2, nib2, nob2) = self.w(&format!("blk.{il}.ssm_beta.weight"))?;
                let (wa2, ta2, nia2, noa2) = self.w(&format!("blk.{il}.ssm_alpha.weight"))?;
                // 독립 4 GEMV — 2스트림 페어 (산술 불변, 2026-09-05)
                // 회귀 픽스: 듀얼 분기 독립 체인 — 기존 if/else는 q5k듀얼시 beta/alpha,
                // q8듀얼시 qkv/gate를 건너뛰었다 (부록90).
                if ty == 13 && tg2 == 13 && ni == nig2 && std::env::var("LLM170_NODUAL").is_err() {
                    self.mm_into2_q5k(self.xq_n, wp, no, self.gqkv, wg2, nog2, self.gz, ni)?;
                } else if std::env::var_os("LLM170_DECODE_PAIRS").is_some() {
                    self.ctx.side_wait_main()?;
                    self.mm_into(self.xq_n, wp, ty, ni, no, self.gqkv)?;
                    self.mm_into_s(self.xq_n, wg2, tg2, nig2, nog2, self.gz)?;
                    self.ctx.join2()?;
                } else {
                    self.mm_into(self.xq_n, wp, ty, ni, no, self.gqkv)?;
                    self.mm_into(self.xq_n, wg2, tg2, nig2, nog2, self.gz)?;
                }
                if tb2 == 8 && ta2 == 8 && nib2 == nia2 {
                    self.mm_into2_q8(self.xq_n, wb2, nob2, self.gb, wa2, noa2, self.ga, nib2)?;
                } else if std::env::var_os("LLM170_DECODE_PAIRS").is_some() {
                    self.ctx.side_wait_main()?;
                    self.mm_into(self.xq_n, wb2, tb2, nib2, nob2, self.gb)?;
                    self.mm_into_s(self.xq_n, wa2, ta2, nia2, noa2, self.ga)?;
                    self.ctx.join2()?;
                } else {
                    self.mm_into(self.xq_n, wb2, tb2, nib2, nob2, self.gb)?;
                    self.mm_into(self.xq_n, wa2, ta2, nia2, noa2, self.ga)?;
                }
                // conv + ring
                let cw = *self.consts.get(&format!("blk.{il}.conv_w")).ok_or("conv_w")?;
                {
                    let mut qp = self.gqkv as *mut std::ffi::c_void;
                    let mut cp = cw as *mut std::ffi::c_void;
                    let mut sp = self.st_conv[recr_idx][seq] as *mut std::ffi::c_void;
                    let mut op = self.gconv as *mut std::ffi::c_void;
                    let mut ch = conv_ch as i32;
                    let mut kk = self.conv_k as i32;
                    let mut args = vec![Self::p(&mut qp), Self::p(&mut cp), Self::p(&mut sp), Self::p(&mut op), Self::p(&mut ch), Self::p(&mut kk)];
                    self.ctx.launch("gdn_conv", (conv_ch as u32).div_ceil(256), 1, 256, &mut args)?;
                }
                // split3 (q/k/v)
                {
                    let mut sp = self.gconv as *mut std::ffi::c_void;
                    let mut q0 = self.gq as *mut std::ffi::c_void;
                    let mut q1 = self.gk as *mut std::ffi::c_void;
                    let mut q2 = self.gv as *mut std::ffi::c_void;
                    let mut n0 = k_len as i32;
                    let mut n1 = k_len as i32;
                    let mut n2 = v_len as i32;
                    let total = 2 * k_len + v_len;
                    let mut args = vec![Self::p(&mut sp), Self::p(&mut q0), Self::p(&mut q1), Self::p(&mut q2), Self::p(&mut n0), Self::p(&mut n1), Self::p(&mut n2)];
                    self.ew_l("split3", total, &mut args)?;
                }
                // l2²+scale
                {
                    let scale = 1.0f32 / (self.d_state as f32).sqrt();
                    let mut qp = self.gq as *mut std::ffi::c_void;
                    let mut kp = self.gk as *mut std::ffi::c_void;
                    let mut ep = self.eps;
                    let mut sc = scale;
                    let mut d = self.d_state as i32;
                    let mut ng = self.n_group as i32;
                    let mut args = vec![Self::p(&mut qp), Self::p(&mut kp), Self::p(&mut ep), Self::p(&mut sc), Self::p(&mut d), Self::p(&mut ng)];
                    self.ctx.launch("l2_rows2_scale", (2 * self.n_group) as u32, 1, 32, &mut args)?;
                }
                // beta/e^g
                let dtb = *self.consts.get(&format!("blk.{il}.dt_bias")).ok_or("dtb")?;
                let ssa = *self.consts.get(&format!("blk.{il}.ssm_a")).ok_or("ssa")?;
                {
                    let mut bp = self.gb as *mut std::ffi::c_void;
                    let mut ap = self.ga as *mut std::ffi::c_void;
                    let mut dp = dtb as *mut std::ffi::c_void;
                    let mut sp2 = ssa as *mut std::ffi::c_void;
                    let mut bgp = self.gbg as *mut std::ffi::c_void;
                    let mut nh = self.dt_rank as i32;
                    let mut dr = self.dt_rank as i32;
                    let mut args = vec![Self::p(&mut bp), Self::p(&mut ap), Self::p(&mut dp), Self::p(&mut sp2), Self::p(&mut bgp), Self::p(&mut nh), Self::p(&mut dr)];
                    self.ew_l(if std::env::var("LLM170_F32SILU").as_deref() != Ok("0") { "gdn_beta_g_f32" } else { "gdn_beta_g" }, self.dt_rank, &mut args)?;
                }
                // AR
                {
                    let n_pairs = self.dt_rank;
                    let mut sp3 = self.st_gdn[recr_idx][seq] as *mut std::ffi::c_void;
                    let mut qp = self.gq as *mut std::ffi::c_void;
                    let mut kp = self.gk as *mut std::ffi::c_void;
                    let mut vp = self.gv as *mut std::ffi::c_void;
                    let mut bgp = self.gbg as *mut std::ffi::c_void;
                    let mut op = self.go as *mut std::ffi::c_void;
                    let mut d = self.d_state as i32;
                    let mut ks = k_len as i32;
                    let mut vs = v_len as i32;
                    let mut hv = self.dt_rank as i32;
                    let mut hk = self.n_group as i32;
                    let mut asc = 1.0f32 / (self.d_state as f32).sqrt();
                    let mut args = vec![Self::p(&mut sp3), Self::p(&mut qp), Self::p(&mut kp), Self::p(&mut vp), Self::p(&mut bgp), Self::p(&mut op), Self::p(&mut d), Self::p(&mut ks), Self::p(&mut vs), Self::p(&mut hv), Self::p(&mut hk), Self::p(&mut asc)];
                    // 전치 레이아웃(부록77)에서는 _w(레인=j)가 코얼레스 — t=1도 포함
                    let mut t1 = 1i32;
                    args.push(Self::p(&mut t1));
                    self.ctx.launch3("gdn_ar_w", n_pairs as u32, self.d_state as u32, 1, 32, &mut args)?;
                }
                if std::env::var_os("LLM170_RAWHIP_TRACE").is_some() && il == 0 {
                    self.ctx.sync()?;
                    let mut ho = vec![0f32; v_len];
                    self.ctx.d2h(bytemuck::cast_slice_mut(&mut ho).as_mut(), self.go)?;
                    let sumo: f64 = ho.iter().map(|&v| v as f64).sum();
                    let mut hq = vec![0f32; k_len];
                    self.ctx.d2h(bytemuck::cast_slice_mut(&mut hq).as_mut(), self.gq)?;
                    let sumq: f64 = hq.iter().map(|&v| v as f64).sum();
                    let mut xco: u64 = 0; let mut xcq: u64 = 0;
                    for &v in &ho { xco ^= (v.to_bits() as u64).wrapping_mul(0x9E3779B97F4A7C15); }
                    for &v in &hq { xcq ^= (v.to_bits() as u64).wrapping_mul(0x9E3779B97F4A7C15); }
                    eprintln!("#  G0dbg go sum={sumo:.6} xor={xco:016x} gq xor={xcq:016x}");
                    let mut hk2 = vec![0f32; k_len];
                    self.ctx.d2h(bytemuck::cast_slice_mut(&mut hk2).as_mut(), self.gk)?;
                    let mut xck: u64 = 0;
                    for &v in &hk2 { xck ^= (v.to_bits() as u64).wrapping_mul(0x9E3779B97F4A7C15); }
                    let mut hv2 = vec![0f32; v_len];
                    self.ctx.d2h(bytemuck::cast_slice_mut(&mut hv2).as_mut(), self.gv)?;
                    let mut xcv: u64 = 0;
                    for &v in &hv2 { xcv ^= (v.to_bits() as u64).wrapping_mul(0x9E3779B97F4A7C15); }
                    let mut hbg = vec![0f32; self.dt_rank * 2];
                    self.ctx.d2h(bytemuck::cast_slice_mut(&mut hbg).as_mut(), self.gbg)?;
                    let mut xcb: u64 = 0; let mut xcg: u64 = 0;
                    for (i, &v) in hbg.iter().enumerate() {
                        if i % 2 == 0 { xcb ^= (v.to_bits() as u64).wrapping_mul(0x9E3779B97F4A7C15); }
                        else { xcg ^= (v.to_bits() as u64).wrapping_mul(0x9E3779B97F4A7C15); }
                    }
                    eprintln!("#  G0dbg gk xor={xck:016x} gv xor={xcv:016x} beta xor={xcb:016x} eg xor={xcg:016x}");
                }
                // norm_gated silu + quant 융합 (행=d_state, 플랫 xq 인덱싱)
                let snorm = *self.consts.get(&format!("blk.{il}.ssm_norm")).ok_or("ssm_norm")?;
                if self.d_state % 32 == 0 && self.d_inner % 1024 == 0 {
                    let mut op = self.go as *mut std::ffi::c_void;
                    let mut zp = self.gz as *mut std::ffi::c_void;
                    let mut wp = snorm as *mut std::ffi::c_void;
                    let mut qp = self.xq_g as *mut std::ffi::c_void;
                    let mut ep = self.eps;
                    let mut d = self.d_state as i32;
                    let mut nh = self.dt_rank as i32;
                    let mut nt = self.d_inner as i32;
                    let mut args = vec![Self::p(&mut op), Self::p(&mut zp), Self::p(&mut wp), Self::p(&mut qp), Self::p(&mut ep), Self::p(&mut d), Self::p(&mut nh), Self::p(&mut nt)];
                    self.ctx.launch("gatedq", self.dt_rank as u32, 1, 32, &mut args)?;
                } else {
                    let mut op = self.go as *mut std::ffi::c_void;
                    let mut zp = self.gz as *mut std::ffi::c_void;
                    let mut wp = snorm as *mut std::ffi::c_void;
                    let mut outp = self.ggated as *mut std::ffi::c_void;
                    let mut ep = self.eps;
                    let mut d = self.d_state as i32;
                    let mut nh = self.dt_rank as i32;
                    let mut args = vec![Self::p(&mut op), Self::p(&mut zp), Self::p(&mut wp), Self::p(&mut outp), Self::p(&mut ep), Self::p(&mut d), Self::p(&mut nh)];
                    self.ctx.launch(if std::env::var("LLM170_F32SILU").as_deref() != Ok("0") { "norm_gated_silu_f32" } else { "norm_gated_silu" }, self.dt_rank as u32, 1, 32, &mut args)?;
                    self.quant(self.ggated, self.xq_g, self.d_inner)?;
                }
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.ssm_out.weight"))?;
                self.mm_into(self.xq_g, wp, ty, ni, no, self.gout)?;
                if std::env::var_os("LLM170_RAWHIP_TRACE").is_some() && il == 0 {
                    self.ctx.sync()?;
                    let mut ho = vec![0f32; n];
                    self.ctx.d2h(bytemuck::cast_slice_mut(&mut ho).as_mut(), self.gout)?;
                    let sumo: f64 = ho.iter().map(|&v| v as f64).sum();
                    eprintln!("#  G0dbg gout sum={sumo:.6} gout[0..4]={:?}", &ho[0..4]);
                }
                recr_idx += 1;
            } else {
                if std::env::var_os("LLM170_RAWHIP_TRACE").is_some() && il == 3 {
                    self.ctx.sync()?;
                    let mut hn = vec![0f32; n];
                    self.ctx.d2h(bytemuck::cast_slice_mut(&mut hn).as_mut(), self.xn)?;
                    eprintln!("#  A3dbg xn[0..6]={:?}", &hn[0..6]);
                    // 결정적 A/B: 이 xn으로 호스트 미러 av[0] 계산
                    if il == 3 && std::env::var_os("LLM170_RAWHIP_HOSTAB").is_some() {
                        let (wp, ty, _ni, _no) = self.w(&format!("blk.{il}.attn_v.weight"))?;
                        let mut wrow = vec![0u8; (5120 / 256) * 176];
                        self.ctx.d2h(&mut wrow, wp)?;
                        let y = llm170_core::quant::quantize_row_q8_ref(&hn);
                        let mv = match ty {
                            13 => llm170_core::quant::dot_row_w4a8_q5k_lane(&wrow, 5120, &y),
                            14 => llm170_core::quant::dot_row_w4a8_q6k_lane(&wrow, 5120, &y),
                            12 => llm170_core::quant::dot_row_w4a8_q4k_lane(&wrow, 5120, &y),
                            _ => f32::NAN,
                        };
                        eprintln!("#  A3dbg host-mirror av[0]={mv:e} d={:e} d_bits={:#x}", y[0].d, y[0].d.to_bits());
                    }
                    let mut hq8 = vec![0u8; (n / 4 + n / 32 + n / 16) * 4];
                    self.ctx.d2h(&mut hq8, self.xq_n)?;
                    let w0 = u32::from_le_bytes([hq8[0], hq8[1], hq8[2], hq8[3]]);
                    eprintln!("#  A3dbg xq_n word0={w0:#010x} d0={:e}", f32::from_bits(u32::from_le_bytes([hq8[n], hq8[n+1], hq8[n+2], hq8[n+3]])));
                }
                // q/k/v mm
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_q.weight"))?;
                self.mm_into(self.xq_n, wp, ty, ni, no, self.aq)?;
                if std::env::var_os("LLM170_RAWHIP_TRACE").is_some() { self.ctx.sync()?; eprintln!("#  aq ok"); }
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_k.weight"))?;
                self.mm_into(self.xq_n, wp, ty, ni, no, self.ak)?;
                if std::env::var_os("LLM170_RAWHIP_TRACE").is_some() { self.ctx.sync()?; eprintln!("#  ak ok"); }
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_v.weight"))?;
                self.mm_into(self.xq_n, wp, ty, ni, no, self.av)?;
                if std::env::var_os("LLM170_RAWHIP_TRACE").is_some() { self.ctx.sync()?; eprintln!("#  av ok"); }
                // q/k norm+rope (in-place)
                let qn = *self.consts.get(&format!("blk.{il}.attn_q_norm")).ok_or("qn")?;
                let kn = *self.consts.get(&format!("blk.{il}.attn_k_norm")).ok_or("kn")?;
                let cs = *self.consts.get("cs").ok_or("cs")?;
                {
                    let mut qp = self.aq as *mut std::ffi::c_void;
                    let mut kp = self.ak as *mut std::ffi::c_void;
                    let mut qwp = qn as *mut std::ffi::c_void;
                    let mut kwp = kn as *mut std::ffi::c_void;
                    let mut csp = cs as *mut std::ffi::c_void;
                    let mut ep = self.eps;
                    let mut kq = self.kq_scale;
                    let mut pp = pos as i32;
                    let mut nh = n_head as i32;
                    let mut nk = n_kv as i32;
                    let mut h = hd as i32;
                    let mut nr = n_rot as i32;
                    let rows = n_head + n_kv;
                    let mut args = vec![Self::p(&mut qp), Self::p(&mut kp), Self::p(&mut qwp), Self::p(&mut kwp), Self::p(&mut csp), Self::p(&mut ep), Self::p(&mut kq), Self::p(&mut pp), Self::p(&mut nh), Self::p(&mut nk), Self::p(&mut h), Self::p(&mut nr)];
                    self.ctx.launch("qk_norm_rope", rows as u32, 1, 32, &mut args)?;
                    if std::env::var_os("LLM170_RAWHIP_TRACE").is_some() { self.ctx.sync()?; eprintln!("#  qk_norm ok"); }
                }
                // KV append
                if std::env::var_os("LLM170_RAWHIP_TRACE").is_some() && il == 3 {
                    self.ctx.sync()?;
                    let mut hk = vec![0f32; n_kv * hd];
                    self.ctx.d2h(bytemuck::cast_slice_mut(&mut hk).as_mut(), self.ak)?;
                    eprintln!("#  A3dbg pos{pos} ak[0..4]={:?}", &hk[0..4]);
                    let mut hck = vec![0f32; (pos + 1) * n_kv * hd];
                    self.ctx.d2h(bytemuck::cast_slice_mut(&mut hck).as_mut(), self.kv_k[full_idx][seq])?;
                    let b0 = pos * n_kv * hd;
                    eprintln!("#  A3dbg pos{pos} cache_k[b0..4]={:?} cache_k[0..4]={:?}", &hck[b0..b0 + 4], &hck[0..4]);
                    let mut hv = vec![0f32; n_kv * hd];
                    self.ctx.d2h(bytemuck::cast_slice_mut(&mut hv).as_mut(), self.av)?;
                    eprintln!("#  A3dbg av[0..4]={:?}", &hv[0..4]);
                    let mut hq = vec![0f32; n_head * 2 * hd];
                    self.ctx.d2h(bytemuck::cast_slice_mut(&mut hq).as_mut(), self.aq)?;
                    eprintln!("#  A3dbg gate h0 [0..4]={:?}", &hq[hd..hd + 4]);
                }
                self.copy(self.ak, self.kv_k[full_idx][seq], 0, pos * n_kv * hd, n_kv * hd)?;
                self.copy(self.av, self.kv_v[full_idx][seq], 0, pos * n_kv * hd, n_kv * hd)?;
                // score
                let mask = *self.consts.get("mask").ok_or("mask")?;
                let flash1 = std::env::var_os("LLM170_NO_FLASH").is_none() && hd <= 256;
                if !flash1 {
                {
                    let n_past = pos + 1;
                    let mut qp = self.aq as *mut std::ffi::c_void;
                    let mut ckp = self.kv_k[full_idx][seq] as *mut std::ffi::c_void;
                    let mut mp = mask as *mut std::ffi::c_void;
                    let mut scp = self.scores as *mut std::ffi::c_void;
                    let mut np_ = n_past as i32;
                    let mut nh = n_head as i32;
                    let mut nk = n_kv as i32;
                    let mut h = hd as i32;
                    let mut tl = 1i32;
                    let mut ss = self.ctx_len as i32;
                    let mut p0 = pos as i32;
                    let gx = n_past.div_ceil(64) as u32;
                    let mut args = vec![Self::p(&mut qp), Self::p(&mut ckp), Self::p(&mut mp), Self::p(&mut scp), Self::p(&mut np_), Self::p(&mut nh), Self::p(&mut nk), Self::p(&mut h), Self::p(&mut tl), Self::p(&mut ss), Self::p(&mut p0)];
                    self.ctx.launch3("qsa_score", gx, n_head as u32, 1, 64, &mut args)?;
                    if std::env::var_os("LLM170_RAWHIP_TRACE").is_some() { self.ctx.sync()?; eprintln!("#  score ok"); }
                }
                // mix
                {
                    let n_past = pos + 1;
                    let mut qp = self.aq as *mut std::ffi::c_void;
                    let mut scp = self.scores as *mut std::ffi::c_void;
                    let mut cvp = self.kv_v[full_idx][seq] as *mut std::ffi::c_void;
                    let mut op = self.aout as *mut std::ffi::c_void;
                    let mut np_ = n_past as i32;
                    let mut nh = n_head as i32;
                    let mut nk = n_kv as i32;
                    let mut h = hd as i32;
                    let mut tl = 1i32;
                    let mut ss = self.ctx_len as i32;
                    let mut p0 = pos as i32;
                    let gx = hd.div_ceil(64) as u32;
                    let mut args = vec![Self::p(&mut qp), Self::p(&mut scp), Self::p(&mut cvp), Self::p(&mut op), Self::p(&mut np_), Self::p(&mut nh), Self::p(&mut nk), Self::p(&mut h), Self::p(&mut tl), Self::p(&mut ss), Self::p(&mut p0)];
                    self.ctx.launch3("qsa_mix2", gx, n_head as u32, 1, 64, &mut args)?;
                    if std::env::var_os("LLM170_RAWHIP_TRACE").is_some() { self.ctx.sync()?; eprintln!("#  mix ok"); }
                }
                }
                // t=1 fused flash (score/mix2 대체)
                if std::env::var_os("LLM170_NO_FLASH").is_none() && hd <= 256 {
                    let mut qp = self.aq as *mut std::ffi::c_void;
                    let mut ckp = self.kv_k[full_idx][seq] as *mut std::ffi::c_void;
                    let mut cvp = self.kv_v[full_idx][seq] as *mut std::ffi::c_void;
                    let mut mp = mask as *mut std::ffi::c_void;
                    let mut op = self.aout as *mut std::ffi::c_void;
                    let mut np_ = (pos + 1) as i32;
                    let mut nh = n_head as i32;
                    let mut nk = n_kv as i32;
                    let mut h = hd as i32;
                    let mut tl = 1i32;
                    let mut ss = self.ctx_len as i32;
                    let mut p0 = pos as i32;
                    // GQA 공유 커널은 세그먼트 수와 무관하게 우월 — 단문(ctx<512)에서도
                    // 평문 qsa_flash 대신 사용한다 (gq=1이면 종전과 동일 산술).
                    let gqa_ok = std::env::var_os("LLM170_NO_GQA").is_none() && hd <= 256 && n_head % n_kv == 0;
                    if gqa_ok || np_ > (std::env::var("LLM170_T1SEG").ok().and_then(|v| v.parse::<i32>().ok()).unwrap_or(512)) {
                        // 분할 flash — 헤드당 1블록(48블록)은 대역폭 저활용,
                        // 세그먼트 병렬화 (t=1도 nq 가드로 안전, 2026-09-05)
                        // 32키 세그먼트 + GQA 공유가 최적 (실측: ctx512 11.08/11.02 vs
                        // 종전 128+헤드별 10.82; ctx3072 10.44 vs 9.60). 세그먼트가
                        // 작을수록 WG가 많아 플랫폼 지연을 숨긴다.
                        let sg = std::env::var("LLM170_T1SG").ok().and_then(|v| v.parse().ok()).unwrap_or(32usize);
                        let nseg = ((pos + 1) + sg - 1) / sg;
                        let part = self.ctx.scratch(1 * n_head * nseg * (hd + 2) * 4)?;
                        let mut pp2 = part as *mut std::ffi::c_void;
                        let mut sg_a = sg as i32;
                        let mut args = vec![Self::p(&mut qp), Self::p(&mut ckp), Self::p(&mut cvp), Self::p(&mut mp), Self::p(&mut pp2), Self::p(&mut np_), Self::p(&mut nh), Self::p(&mut nk), Self::p(&mut h), Self::p(&mut tl), Self::p(&mut ss), Self::p(&mut p0), Self::p(&mut sg_a)];
                        // GQA 공유(t=1): kv-head당 WG 하나가 q-head 전부를 처리 —
                        // K/V 트래픽 1/(q-heads per kv-head). LLM170_NO_GQA=1이면 종전.
                        // 실측 교차점: ctx<768은 종전(더 많은 WG), 그 이상은 GQA 공유가
                        // 이김 (pp512 −1.2%, 1024 +1.6%, 2048 +4.7%, 3072 +8.8%).
                        let gqa = gqa_ok;
                        if gqa {
                            self.ctx.launch3("qsa_flash_gqa", 1, n_kv as u32, nseg as u32, 256, &mut args)?;
                        } else {
                            self.ctx.launch3("qsa_flash_split4q4", 1, n_head as u32, nseg as u32, 256, &mut args)?;
                        }
                        let mut margs = vec![Self::p(&mut qp), Self::p(&mut pp2), Self::p(&mut op), Self::p(&mut np_), Self::p(&mut nh), Self::p(&mut h), Self::p(&mut tl), Self::p(&mut sg_a)];
                        self.ctx.launch3("qsa_flash_merge", 1, n_head as u32, 1, 256, &mut margs)?;
                    } else {
                        let mut args = vec![Self::p(&mut qp), Self::p(&mut ckp), Self::p(&mut cvp), Self::p(&mut mp), Self::p(&mut op), Self::p(&mut np_), Self::p(&mut nh), Self::p(&mut nk), Self::p(&mut h), Self::p(&mut tl), Self::p(&mut ss), Self::p(&mut p0)];
                        self.ctx.launch3("qsa_flash", 1, n_head as u32, 1, 256, &mut args)?;
                    }
                }
                if std::env::var_os("LLM170_RAWHIP_TRACE").is_some() && il == 3 {
                    self.ctx.sync()?;
                    let mut ho = vec![0f32; n_head * hd];
                    self.ctx.d2h(bytemuck::cast_slice_mut(&mut ho).as_mut(), self.aout)?;
                    let mut hs = vec![0f32; n_head * (pos + 1)];
                    self.ctx.d2h(bytemuck::cast_slice_mut(&mut hs).as_mut(), self.scores)?;
                    let sumsc: f64 = hs.iter().map(|&v| v as f64).sum();
                    eprintln!("#  A3dbg h0 scores_sum={sumsc:.6} n_past={} aout[0..4]={:?}", pos + 1, &ho[0..4]);
                }
                // wo
                self.quant(self.aout, self.xq_g, n_head * hd)?;
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_output.weight"))?;
                self.mm_into(self.xq_g, wp, ty, ni, no, self.gout)?;
                full_idx += 1;
            }
            // 잔차
            self.axpy(self.xs, self.gout, n)?;
            if std::env::var_os("LLM170_RAWHIP_TRACE").is_some() {
                self.ctx.sync()?;
                let mut hv = vec![0f32; n];
                self.ctx.d2h(bytemuck::cast_slice_mut(&mut hv).as_mut(), self.xs)?;
                let sum: f64 = hv.iter().map(|&v| v as f64).sum();
                eprintln!("#  L{il} xs: sum={sum:.6} x0={:.5} x1={:.5}", hv[0], hv[1]);
            }
            // FFN
            let pw = *self.consts.get(&format!("blk.{il}.post_norm")).ok_or("post_norm")?;
            self.rms_quant(self.xs, pw, self.xq_n, n)?;
            let (wg, tg, nig, nog) = self.w(&format!("blk.{il}.ffn_gate.weight"))?;
            let (wu, tu, niu, nou) = self.w(&format!("blk.{il}.ffn_up.weight"))?;
            if std::env::var_os("LLM170_Q1MMQ").is_some() && matches!(tg | tu, 12 | 13 | 14 | 23)
                && super::co_loaded(super::CO_MMQ | super::CO_MMQ2 | super::CO_MMQ3) {
                // 실험(부록 74): 디코드 GEMV를 mmq 타일로 — f32 직행, 별도 quant 불요.
                self.rms(self.xs, pw, self.xn, n)?;
                self.mm_b2(self.xn as *mut u8, self.xq_n, n / 4 + n / 32 + n / 16, wg, tg, nig, nog, self.fgate, 1)?;
                self.mm_b2(self.xn as *mut u8, self.xq_n, n / 4 + n / 32 + n / 16, wu, tu, niu, nou, self.fup, 1)?;
            } else if std::env::var_os("LLM170_DECODE_PAIRS").is_some() {
                self.ctx.side_wait_main()?;
                self.mm_into(self.xq_n, wg, tg, nig, nog, self.fgate)?;
                self.mm_into_s(self.xq_n, wu, tu, niu, nou, self.fup)?;
                self.ctx.join2()?;
            } else {
                self.mm_into(self.xq_n, wg, tg, nig, nog, self.fgate)?;
                self.mm_into(self.xq_n, wu, tu, niu, nou, self.fup)?;
            }
            // silu_mul+quant 융합 (t=1, n_ff%2048==0) — 동일 산술열
            if self.n_ff % 2048 == 0 {
                let mut gp = self.fgate as *mut std::ffi::c_void;
                let mut up = self.fup as *mut std::ffi::c_void;
                let mut qp = self.xq_f as *mut std::ffi::c_void;
                let mut na = self.n_ff as i32;
                let mut args = vec![Self::p(&mut gp), Self::p(&mut up), Self::p(&mut qp), Self::p(&mut na)];
                let gx = (self.n_ff >> 5).div_ceil(64) as u32;
                self.ctx.launch3("silu_mulq", gx, 1, 1, 64, &mut args)?;
            } else {
                let mut gp = self.fgate as *mut std::ffi::c_void;
                let mut up = self.fup as *mut std::ffi::c_void;
                let mut op = self.fglu as *mut std::ffi::c_void;
                let mut na = self.n_ff as i32;
                let mut args = vec![Self::p(&mut gp), Self::p(&mut up), Self::p(&mut op), Self::p(&mut na)];
                self.ew_l("silu_mul", self.n_ff, &mut args)?;
                self.quant(self.fglu, self.xq_f, self.n_ff)?;
            }
            let (wd, td, nid, nod) = self.w(&format!("blk.{il}.ffn_down.weight"))?;
            if std::env::var_os("LLM170_DOWN_PROF").is_some() { self.ctx.sync()?; }
            let __td = std::time::Instant::now();
            self.mm_into(self.xq_f, wd, td, nid, nod, self.fdown)?;
            if std::env::var_os("LLM170_DOWN_PROF").is_some() {
                self.ctx.sync()?;
                eprintln!("# down ty={td} no={nod} {:.3}ms", __td.elapsed().as_secs_f64() * 1e3);
            }
            self.axpy(self.xs, self.fdown, n)?;
            if std::env::var_os("LLM170_RAWHIP_TRACE").is_some() {
                self.ctx.sync()?;
                let mut hv = vec![0f32; n];
                self.ctx.d2h(bytemuck::cast_slice_mut(&mut hv).as_mut(), self.xs)?;
                let sum: f64 = hv.iter().map(|&v| v as f64).sum();
                eprintln!("#  E{il} xs: sum={sum:.6} x0={:.5} x1={:.5}", hv[0], hv[1]);
            }
        }
        // head
        let wn = *self.consts.get("output_norm").ok_or("output_norm")?;
        self.rms(self.xs, wn, self.xn, n)?;
        self.quant(self.xn, self.xq_n, n)?;
        let (wh, th, nih, noh) = self.w("output.weight")?;
        self.mm_into(self.xq_n, wh, th, nih, noh, self.logits)?;
        if std::env::var_os("LLM170_RAWHIP_TIMING").is_some() {
            eprintln!("step gpu={:.2}ms", t0.elapsed().as_secs_f64() * 1e3);
        }
        Ok(Vec::new()) // logits 상주
    }
}


/// Engine 주입용 RawDecode 구현 — DecodeState를 Mutex로 보관.
pub struct RawDecoder {
    st: std::sync::Mutex<Option<DecodeState>>,
}

impl RawDecoder {
    pub fn new() -> Self {
        RawDecoder { st: std::sync::Mutex::new(None) }
    }
}

impl llm170_core::matmul::RawDecode for RawDecoder {
    fn raw_init(
        &self,
        hp: &llm170_core::model::hparams::Hparams,
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
        if std::env::var_os("LLM170_KTRACE").is_some() { crate::rawhip::ktrace_on(); }
        let guard = self.st.lock().map_err(|e| e.to_string())?;
        let ds = guard.as_ref().ok_or("raw_decode: 미초기화")?;
        ds.step_batch(seq, pos0, emb)?;
        let r = ds.read_logits();
        if std::env::var_os("LLM170_KTRACE").is_some() {
            eprintln!("{}", crate::rawhip::ktrace_dump());
        }
        if let (Some(path), Ok(v)) = (std::env::var_os("LLM170_DUMP_LOGITS"), r.as_ref()) {
            let _ = std::fs::write(&path, bytemuck::cast_slice(v));
        }
        if std::env::var_os("LLM170_RAWHIP_TIMING").is_some() {
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
        if std::env::var_os("LLM170_KTRACE").is_some() { crate::rawhip::ktrace_on(); }
        let guard = self.st.lock().map_err(|e| e.to_string())?;
        let ds = guard.as_ref().ok_or("raw_decode: 미초기화")?;
        ds.step_batch(seq, pos0, emb)?;
        // 진단: 프리필 후 MTP KV가 채워졌는지 (비영 검사)
        if std::env::var_os("LLM170_DUMP_MTPKV").is_some() {
            let nw = 4096usize; // 앞 16KB
            let mut kv = vec![0f32; nw];
            if let Some(buf) = ds.mtp_kv_k.get(seq).copied() {
                let _ = ds.ctx.d2h(bytemuck::cast_slice_mut(&mut kv).as_mut(), buf);
            }
            let h = (pos0 + emb.len() / ds.n_embd) as usize;
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
        if std::env::var_os("LLM170_KTRACE").is_some() {
            eprintln!("{}", crate::rawhip::ktrace_dump());
        }
        if std::env::var_os("LLM170_RAWHIP_TIMING").is_some() {
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
        if std::env::var_os("LLM170_SPEC_TIMING").is_some() {
            eprintln!("[rv] verify_batch={:.1}ms", t_rv0.elapsed().as_secs_f64() * 1e3);
        }
        // 행별 최종 hidden export (MTP 상태 진행용) — xs_t에 step_batch 결과 잔존
        let t = emb.len() / ds.n_embd;
        h_all.clear();
        h_all.resize(t * ds.n_embd, 0.0);
        let t_h0 = std::time::Instant::now();
        ds.ctx
            .d2h(bytemuck::cast_slice_mut(h_all).as_mut(), ds.xs_t)?;
        if std::env::var_os("LLM170_SPEC_TIMING").is_some() {
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
        if std::env::var_os("LLM170_KTRACE").is_some() { crate::rawhip::ktrace_on(); }
        let r = ds.step_batch_np(seqs, poss, emb);
        if std::env::var_os("LLM170_KTRACE").is_some() {
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
        super::co_loaded(super::CO_J128)
    }

    fn raw_step(&self, seq: usize, pos: usize, emb: &[f32]) -> Result<Vec<f32>, String> {
        let t0 = std::time::Instant::now();
        if std::env::var_os("LLM170_KTRACE").is_some() { crate::rawhip::ktrace_on(); }
        let guard = self.st.lock().map_err(|e| e.to_string())?;
        let ds = guard.as_ref().ok_or("raw_decode: 미초기화")?;
        ds.ctx.h2d(ds.xs, bytemuck::cast_slice(emb))?;
        ds.step(seq, pos)?;
        let r = ds.read_logits();
        if std::env::var_os("LLM170_KTRACE").is_some() {
            eprintln!("{}", crate::rawhip::ktrace_dump());
        }
        if std::env::var_os("LLM170_RAWHIP_TIMING").is_some() {
            eprintln!("step cpu={:.2}ms", t0.elapsed().as_secs_f64() * 1e3);
        }
        r
    }
}

unsafe impl Send for DecodeState {}


impl DecodeState {
    /// 프리필 배치 스텝 — t 토큰 (emb: [t][n_embd], pos0..pos0+t-1), 마지막 logits.
    /// mm/quant/rms/silu/l2/split3/beta_g/norm_gated/qk_norm_rope 배치,
    /// conv/AR/KV/qsa 순차·토큰 의존 — 토큰 루프. 산술은 step()과 토큰당 동일열.
    #[allow(clippy::too_many_lines)]
    pub fn step_batch(&self, seq: usize, pos0: usize, emb: &[f32]) -> Result<Vec<f32>, String> {
        let t = emb.len() / self.n_embd;
        debug_assert!(t >= 1 && t <= self.b_t_max);
        let n = self.n_embd;
        let prof = std::env::var_os("LLM170_PP_PROF").is_some();
        let t0w = std::time::Instant::now();
        let mut marks: Vec<(String, hip::hipEvent_t)> = Vec::new();
        let mut gmark = |lab: &str, marks: &mut Vec<(String, hip::hipEvent_t)>| {
            if prof {
                let mut ev: hip::hipEvent_t = std::ptr::null_mut();
                unsafe { hip::hipEventCreate(&mut ev); hip::hipEventRecord(ev, self.ctx.stream); }
                marks.push((lab.to_string(), ev));
            }
        };
        let (k_len, v_len, conv_ch) = (self.k_len, self.v_len, self.conv_ch);
        let (n_head, n_kv, hd, n_rot) = (self.n_head, self.n_kv, self.hd, self.n_rot);
        let xq_sn = n / 4 + n / 32 + n / 16;
        let xq_sf = self.n_ff / 4 + self.n_ff / 32 + self.n_ff / 16;
        let xq_sg = self.d_inner / 4 + self.d_inner / 32 + self.d_inner / 16;
        self.ctx.h2d(self.xs_t, bytemuck::cast_slice(emb))?;
        let cs = *self.consts.get("cs").ok_or("cs")?;
        let mask = *self.consts.get("mask").ok_or("mask")?;
        let mut recr_idx = 0usize;
        let mut full_idx = 0usize;
        for il in 0..self.n_layer {
            let wn = *self.consts.get(&format!("blk.{il}.attn_norm")).ok_or("attn_norm")?;
            self.rms_rows(self.xs_t, wn, self.xn_t, n, t)?;
            self.ctx.mmq_y_bump();  // 부록81: xn_t 재기 → quant_y 캐시 무효화
            // qkv/gate/beta/alpha(또는 q/k/v)가 모두 f32 직소비면 q8 활성은 사장 —
            // MMQ는 y_f32를 직접 읽고 내부에서 mmq 레이아웃으로 재양자화한다.
            let proj_names: Vec<String> = if self.is_recr[il] {
                ["attn_qkv", "attn_gate", "ssm_beta", "ssm_alpha"].iter()
                    .map(|k2| format!("blk.{il}.{k2}.weight")).collect()
            } else {
                ["attn_q", "attn_k", "attn_v"].iter()
                    .map(|k2| format!("blk.{il}.{k2}.weight")).collect()
            };
            if !self.grp_mmq(&proj_names, t) {
                self.ctx.quant_q8_b(self.xn_t, self.xq_n_t, n, xq_sn, t)?;
            }
gmark("norm", &mut marks);
            if self.is_recr[il] {
                // 2스트림: qkv+beta(주) ‖ gate+alpha(사이드) — 4독립 GEMM
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_qkv.weight"))?;
                let qkv_tile = matches!(ty, 12 | 13 | 14 | 23) && t > 64;
                let (wg2, tg2, nig2, nog2) = self.w(&format!("blk.{il}.attn_gate.weight"))?;
                let gate_tile = matches!(tg2, 12 | 13 | 14 | 23) && t > 64;
                let (wb2, tb2, nib2, nob2) = self.w(&format!("blk.{il}.ssm_beta.weight"))?;
                let (wa2, ta2, nia2, noa2) = self.w(&format!("blk.{il}.ssm_alpha.weight"))?;
                if qkv_tile && gate_tile {
                    // 사이드는 타일형만 (beta/alpha는 q8_0 gemv — 주 스트림)
                    // (사이드 이동 시도: L2 경합으로 -5 t/s 회귀 — 2026-09-05 측정)
                    self.ctx.side_wait_main()?;
                    self.mm_b2_s(self.xn_t, self.xq_n_t, xq_sn, wg2, tg2, nig2, nog2, self.gz_t, t)?;
                    self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wp, ty, ni, no, self.gqkv_t, t)?;
                    self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wb2, tb2, nib2, nob2, self.gb_t, t)?;
                    self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wa2, ta2, nia2, noa2, self.ga_t, t)?;
                    self.ctx.join2()?;
                } else {
                    self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wp, ty, ni, no, self.gqkv_t, t)?;
                    self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wg2, tg2, nig2, nog2, self.gz_t, t)?;
                    self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wb2, tb2, nib2, nob2, self.gb_t, t)?;
                    self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wa2, ta2, nia2, noa2, self.ga_t, t)?;
                }
                let cw = *self.consts.get(&format!("blk.{il}.conv_w")).ok_or("conv_w")?;
                let dtb = *self.consts.get(&format!("blk.{il}.dt_bias")).ok_or("dtb")?;
                let ssa = *self.consts.get(&format!("blk.{il}.ssm_a")).ok_or("ssa")?;
                let snorm = *self.consts.get(&format!("blk.{il}.ssm_norm")).ok_or("ssm_norm")?;
gmark("gdn_mm", &mut marks);
                // conv+ring 배치 (채널 블록 × t 내부 순차)
                {
                    let mut qp = self.gqkv_t as *mut std::ffi::c_void;
                    let mut cp = cw as *mut std::ffi::c_void;
                    let mut sp = self.st_conv[recr_idx][seq] as *mut std::ffi::c_void;
                    let mut op = self.gconv_t as *mut std::ffi::c_void;
                    let mut ch = conv_ch as i32;
                    let mut kk = self.conv_k as i32;
                    let mut tt = t as i32;
                    let mut args = vec![Self::p(&mut qp), Self::p(&mut cp), Self::p(&mut sp), Self::p(&mut op), Self::p(&mut ch), Self::p(&mut kk), Self::p(&mut tt)];
                    if t >= self.conv_k - 1 {
                        self.ctx.launch3(if std::env::var("LLM170_F32SILU").as_deref() != Ok("0") { "gdn_conv_t2_f32" } else { "gdn_conv_t2" }, conv_ch.div_ceil(64) as u32, t as u32, 1, 64, &mut args)?;
                        let mut qp2 = self.gqkv_t as *mut std::ffi::c_void;
                        let mut sp2 = self.st_conv[recr_idx][seq] as *mut std::ffi::c_void;
                        let mut ch2 = conv_ch as i32;
                        let mut kk2 = self.conv_k as i32;
                        let mut tt2 = t as i32;
                        let mut args2 = vec![Self::p(&mut qp2), Self::p(&mut sp2), Self::p(&mut ch2), Self::p(&mut kk2), Self::p(&mut tt2)];
                        self.ctx.launch3("gdn_conv_state", (self.conv_k - 1) as u32, conv_ch.div_ceil(64) as u32, 1, 64, &mut args2)?;
                    } else {
                        self.ctx.launch3("gdn_conv_t", conv_ch as u32, 1, 1, 32, &mut args)?;
                    }
                }
                if il == 0 {
                    self.trace_rows("tr_gconv", self.gconv_t, conv_ch, t)?;
                }
                // split3 전체 배치 (요소별)
                {
                    let mut sp = self.gconv_t as *mut std::ffi::c_void;
                    let mut q0 = self.gq_t as *mut std::ffi::c_void;
                    let mut q1 = self.gk_t as *mut std::ffi::c_void;
                    let mut q2 = self.gv_t as *mut std::ffi::c_void;
                    let mut n0 = k_len as i32;
                    let mut n1 = k_len as i32;
                    let mut n2 = v_len as i32;
                    let total = (2 * k_len + v_len) * t;
                    let mut args = vec![Self::p(&mut sp), Self::p(&mut q0), Self::p(&mut q1), Self::p(&mut q2), Self::p(&mut n0), Self::p(&mut n1), Self::p(&mut n2)];
                    self.ew_l("split3", total, &mut args)?;
                }
                // l2 전체 배치 (gy=t)
                {
                    let scale = 1.0f32 / (self.d_state as f32).sqrt();
                    let mut qp = self.gq_t as *mut std::ffi::c_void;
                    let mut kp = self.gk_t as *mut std::ffi::c_void;
                    let mut ep = self.eps;
                    let mut sc = scale;
                    let mut d = self.d_state as i32;
                    let mut ng = self.n_group as i32;
                    let mut args = vec![Self::p(&mut qp), Self::p(&mut kp), Self::p(&mut ep), Self::p(&mut sc), Self::p(&mut d), Self::p(&mut ng)];
                    let l2k = if std::env::var_os("LLM170_EXACT").is_some() { "l2_rows2_scale" } else { "l2_rows2_scale_w" };
                    self.ctx.launch3(l2k, (2 * self.n_group) as u32, t as u32, 1, 32, &mut args)?;
                }
gmark("split+l2", &mut marks);
                // beta/e^g 전체 배치 (요소별)
                {
                    let mut bp = self.gb_t as *mut std::ffi::c_void;
                    let mut ap = self.ga_t as *mut std::ffi::c_void;
                    let mut dp = dtb as *mut std::ffi::c_void;
                    let mut sp2 = ssa as *mut std::ffi::c_void;
                    let mut bgp = self.gbg_t as *mut std::ffi::c_void;
                    let mut nh = (self.dt_rank * t) as i32;
                    let mut dr = self.dt_rank as i32;
                    let mut args = vec![Self::p(&mut bp), Self::p(&mut ap), Self::p(&mut dp), Self::p(&mut sp2), Self::p(&mut bgp), Self::p(&mut nh), Self::p(&mut dr)];
                    self.ew_l(if std::env::var("LLM170_F32SILU").as_deref() != Ok("0") { "gdn_beta_g_f32" } else { "gdn_beta_g" }, self.dt_rank * t, &mut args)?;
                }
gmark("betag", &mut marks);
                // AR 배치 (pair 블록 × t 내부 순차)
                {
                    let mut sp3 = self.st_gdn[recr_idx][seq] as *mut std::ffi::c_void;
                    let mut qp = self.gq_t as *mut std::ffi::c_void;
                    let mut kp = self.gk_t as *mut std::ffi::c_void;
                    let mut vp = self.gv_t as *mut std::ffi::c_void;
                    let mut bgp = self.gbg_t as *mut std::ffi::c_void;
                    let mut op = self.go_t as *mut std::ffi::c_void;
                    let mut d = self.d_state as i32;
                    let mut ks = k_len as i32;
                    let mut vs = v_len as i32;
                    let mut hv = self.dt_rank as i32;
                    let mut hk = self.n_group as i32;
                    let mut asc = 1.0f32 / (self.d_state as f32).sqrt();
                    let mut tt = t as i32;
                    let mut args = vec![Self::p(&mut sp3), Self::p(&mut qp), Self::p(&mut kp), Self::p(&mut vp), Self::p(&mut bgp), Self::p(&mut op), Self::p(&mut d), Self::p(&mut ks), Self::p(&mut vs), Self::p(&mut hv), Self::p(&mut hk), Self::p(&mut asc), Self::p(&mut tt)];
                    if std::env::var_os("LLM170_EXACT").is_some() || std::env::var_os("LLM170_AR_T").is_some() {
                        self.ctx.launch3("gdn_ar_t", self.dt_rank as u32, (self.d_state / 64) as u32, 1, 64, &mut args)?;
                    } else if t >= 128 && self.d_state == 128 && std::env::var_os("LLM170_ARCHUNK").is_some() {
                        // 청크 스캔 (부록 72): A(로컬)→B(캐리)→C(보정)
                        const CH: usize = 64;
                        let npair = self.dt_rank;
                        let mut d = self.d_state as i32;
                        let nc = t.div_ceil(CH);
                        let nc_max = self.b_t_max.div_ceil(CH);
                        // ffn 버퍼 별칭 (AR은 층 내 ffn 이전 — 라이프타임 무충돌)
                        let mut sstart = self.fgate_t as *mut u8;
                        let mut pbuf = self.fup_t as *mut u8;
                        let mut pgb = self.fglu_t as *mut u8;
                        let mut lend = (self.fglu_t as *mut u8).wrapping_add(262144);
                        let _ = nc_max;
                        let mut chi = CH as i32;
                        let mut ca: Vec<*mut std::ffi::c_void> = args.clone();
                        ca.extend([Self::p(&mut chi), Self::p(&mut lend), Self::p(&mut pgb), Self::p(&mut pbuf)].iter().copied());
                        self.ctx.launch3("gdn_ar_chunk_a", npair as u32, d as u32, nc as u32, 32, &mut ca)?;
                        let (mut bnp, mut bnc) = (npair as i32, nc as i32);
                        let mut cb: Vec<*mut std::ffi::c_void> = vec![
                            Self::p(&mut sp3), Self::p(&mut lend), Self::p(&mut pgb),
                            Self::p(&mut sstart), Self::p(&mut d), Self::p(&mut tt),
                            Self::p(&mut chi), Self::p(&mut bnp), Self::p(&mut bnc),
                        ];
                        self.ctx.launch3("gdn_ar_chunk_b", npair as u32, 1, 1, d as u32, &mut cb)?;
                        let mut cc: Vec<*mut std::ffi::c_void> = vec![
                            Self::p(&mut op), Self::p(&mut qp), Self::p(&mut pbuf),
                            Self::p(&mut sstart), Self::p(&mut d), Self::p(&mut ks),
                            Self::p(&mut vs), Self::p(&mut hv), Self::p(&mut hk),
                            Self::p(&mut asc), Self::p(&mut tt), Self::p(&mut chi),
                            Self::p(&mut bnp),
                        ];
                        self.ctx.launch3("gdn_ar_chunk_c2", npair as u32, nc as u32, 1, d as u32, &mut cc)?;
                    } else if std::env::var_os("LLM170_ARSM").is_some() && self.d_state == 128 {
                        // 부록82: smem 스테이징 AR (k/q 128중 재독 제거)
                        self.ctx.launch3("gdn_ar_sm", self.dt_rank as u32, 2, 1, 64, &mut args)?;
                    } else {
                        // 부록88 기본: 축스왑(u블록 인접) — k/q L2 국소성 +1.1% (350-354)
                        if std::env::var_os("LLM170_NO_ARSWAP").is_none() {
                            self.ctx.launch3("gdn_ar_w_swap", self.d_state as u32, self.dt_rank as u32, 1, 32, &mut args)?;
                        } else {
                            // 8워프/블록 (워프당 1열) — 블록 수 1/8
                            let uw = 8u32;
                            self.ctx.launch3("gdn_ar_w", self.dt_rank as u32, (self.d_state as u32).div_ceil(uw), 1, 32 * uw, &mut args)?;
                        }
                    }
                }
                if il == 0 {
                    self.trace_rows("tr_go", self.go_t, v_len, t)?;
                }
                if std::env::var_os("LLM170_RAWHIP_TRACE").is_some() && il == 0 {
                    self.ctx.sync()?;
                    let mut hq = vec![0f32; k_len * t];
                    self.ctx.d2h(bytemuck::cast_slice_mut(&mut hq).as_mut(), self.gq_t)?;
                    let mut xq_: u64 = 0;
                    for &v in &hq { xq_ ^= (v.to_bits() as u64).wrapping_mul(0x9E3779B97F4A7C15); }
                    let mut ho = vec![0f32; v_len * t];
                    self.ctx.d2h(bytemuck::cast_slice_mut(&mut ho).as_mut(), self.go_t)?;
                    let mut xo_: u64 = 0;
                    for &v in &ho { xo_ ^= (v.to_bits() as u64).wrapping_mul(0x9E3779B97F4A7C15); }
                    eprintln!("#  G0dbg batch gq xor={xq_:016x} go xor={xo_:016x}");
                }
gmark("trace", &mut marks);
                // norm_gated 전체 배치 (gy=t)
                {
                    let mut op = self.go_t as *mut std::ffi::c_void;
                    let mut zp = self.gz_t as *mut std::ffi::c_void;
                    let mut wp = snorm as *mut std::ffi::c_void;
                    let mut outp = self.ggated_t as *mut std::ffi::c_void;
                    let mut ep = self.eps;
                    let mut d = self.d_state as i32;
                    let mut nh = self.dt_rank as i32;
                    let mut args = vec![Self::p(&mut op), Self::p(&mut zp), Self::p(&mut wp), Self::p(&mut outp), Self::p(&mut ep), Self::p(&mut d), Self::p(&mut nh)];
                    self.ctx.launch3(if std::env::var("LLM170_F32SILU").as_deref() != Ok("0") { "norm_gated_silu_f32" } else { "norm_gated_silu" }, self.dt_rank as u32, t as u32, 1, 32, &mut args)?;
                }
gmark("gdn", &mut marks);
gmark("normg", &mut marks);
                // out proj 배치
                if !self.grp_mmq(&[format!("blk.{il}.ssm_out.weight")], t) {
                    self.ctx.quant_q8_b(self.ggated_t, self.xq_g_t, self.d_inner, xq_sg, t)?;
                }
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.ssm_out.weight"))?;
gmark("outproj", &mut marks);
                self.mm_b2(self.ggated_t, self.xq_g_t, xq_sg, wp, ty, ni, no, self.gout_t, t)?;
                recr_idx += 1;
} else {
gmark("attn", &mut marks);
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_q.weight"))?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wp, ty, ni, no, self.aq_t, t)?;
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_k.weight"))?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wp, ty, ni, no, self.ak_t, t)?;
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_v.weight"))?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wp, ty, ni, no, self.av_t, t)?;
                let qn = *self.consts.get(&format!("blk.{il}.attn_q_norm")).ok_or("qn")?;
                let kn = *self.consts.get(&format!("blk.{il}.attn_k_norm")).ok_or("kn")?;
                // q/k norm+rope 전체 배치 (gy=t — 커널 pos+y)
                {
                    let mut qp = self.aq_t as *mut std::ffi::c_void;
                    let mut kp = self.ak_t as *mut std::ffi::c_void;
                    let mut qwp = qn as *mut std::ffi::c_void;
                    let mut kwp = kn as *mut std::ffi::c_void;
                    let mut csp = cs as *mut std::ffi::c_void;
                    let mut ep = self.eps;
                    let mut kq = self.kq_scale;
                    let mut pp = pos0 as i32;
                    let mut nh = n_head as i32;
                    let mut nk = n_kv as i32;
                    let mut h = hd as i32;
                    let mut nr = n_rot as i32;
                    let rows = n_head + n_kv;
                    let mut args = vec![Self::p(&mut qp), Self::p(&mut kp), Self::p(&mut qwp), Self::p(&mut kwp), Self::p(&mut csp), Self::p(&mut ep), Self::p(&mut kq), Self::p(&mut pp), Self::p(&mut nh), Self::p(&mut nk), Self::p(&mut h), Self::p(&mut nr)];
                    self.ctx.launch3("qk_norm_rope", rows as u32, t as u32, 1, 32, &mut args)?;
                }
                // KV append 배치 (gy=t)
                {
                    let mut sp = self.ak_t as *mut std::ffi::c_void;
                    let mut dp = self.kv_k[full_idx][seq] as *mut std::ffi::c_void;
                    let mut na = (n_kv * hd) as i32;
                    let mut p0 = pos0 as i32;
                    let mut args = vec![Self::p(&mut sp), Self::p(&mut dp), Self::p(&mut na), Self::p(&mut p0)];
                    self.ctx.launch3("kv_append_t", (n_kv * hd).div_ceil(64) as u32, t as u32, 1, 64, &mut args)?;
                }
                {
                    let mut sp = self.av_t as *mut std::ffi::c_void;
                    let mut dp = self.kv_v[full_idx][seq] as *mut std::ffi::c_void;
                    let mut na = (n_kv * hd) as i32;
                    let mut p0 = pos0 as i32;
                    let mut args = vec![Self::p(&mut sp), Self::p(&mut dp), Self::p(&mut na), Self::p(&mut p0)];
                    self.ctx.launch3("kv_append_t", (n_kv * hd).div_ceil(64) as u32, t as u32, 1, 64, &mut args)?;
                }
                // qsa 배치 (단일 런치 — sstride=ctx, n_past=pos0+t 최대;
                // 초과 p는 마스크→-3e38→w=0 기여로 원소 산술열 불변)
                {
                    // flash 시 기존 score/mix2 스킵 (이중실행 방지)
                    let flash_only = std::env::var_os("LLM170_NO_FLASH").is_none() && hd <= 256;
                    if !flash_only {
                    let np_max = (pos0 + t) as i32;
                    let sstr = self.ctx_len as i32;
                    {
                        let mut qp = self.aq_t as *mut std::ffi::c_void;
                        let mut ckp = self.kv_k[full_idx][seq] as *mut std::ffi::c_void;
                        let mut mp = mask as *mut std::ffi::c_void;
                        let mut scp = self.scores_t as *mut std::ffi::c_void;
                        let mut np_ = np_max;
                        let mut nh = n_head as i32;
                        let mut nk = n_kv as i32;
                        let mut h = hd as i32;
                        let mut tl = t as i32;
                        let mut ss = sstr;
                        let mut p0 = pos0 as i32;
                        let gx = np_max.unsigned_abs().div_ceil(64);
                        let mut args = vec![Self::p(&mut qp), Self::p(&mut ckp), Self::p(&mut mp), Self::p(&mut scp), Self::p(&mut np_), Self::p(&mut nh), Self::p(&mut nk), Self::p(&mut h), Self::p(&mut tl), Self::p(&mut ss), Self::p(&mut p0)];
                        self.ctx.launch3("qsa_score", gx, n_head as u32, t as u32, 64, &mut args)?;
                    }
                    {
                        let mut qp = self.aq_t as *mut std::ffi::c_void;
                        let mut scp = self.scores_t as *mut std::ffi::c_void;
                        let mut cvp = self.kv_v[full_idx][seq] as *mut std::ffi::c_void;
                        let mut op = self.aout_t as *mut std::ffi::c_void;
                        let mut np_ = np_max;
                        let mut nh = n_head as i32;
                        let mut nk = n_kv as i32;
                        let mut h = hd as i32;
                        let mut tl = t as i32;
                        let mut ss = sstr;
                        let mut p0 = pos0 as i32;
                        let gx = hd.div_ceil(64) as u32;
                        let mut args = vec![Self::p(&mut qp), Self::p(&mut scp), Self::p(&mut cvp), Self::p(&mut op), Self::p(&mut np_), Self::p(&mut nh), Self::p(&mut nk), Self::p(&mut h), Self::p(&mut tl), Self::p(&mut ss), Self::p(&mut p0)];
                        self.ctx.launch3("qsa_mix2", gx, n_head as u32, t as u32, 64, &mut args)?;
                    }
                    }
                }
                // qsa fused flash: score+softmax+mix 단일 패스 (maxrel 1e-6 검증, 스트림★)
                if std::env::var_os("LLM170_NO_FLASH").is_none() && hd <= 256 {
                    let mut qp = self.aq_t as *mut std::ffi::c_void;
                    let mut ckp = self.kv_k[full_idx][seq] as *mut std::ffi::c_void;
                    let mut cvp = self.kv_v[full_idx][seq] as *mut std::ffi::c_void;
                    let mut mp = mask as *mut std::ffi::c_void;
                    let mut op = self.aout_t as *mut std::ffi::c_void;
                    let mut np_ = (pos0 + t) as i32;
                    let mut nh = n_head as i32;
                    let mut nk = n_kv as i32;
                    let mut h = hd as i32;
                    let mut tl = t as i32;
                    let mut ss = self.ctx_len as i32;
                    let mut p0 = pos0 as i32;
                    // 분할 flash 기본 ON (2026-09-05: pp512 +5 — 청크 2-4의 np 성장
                    // 구간 병렬화; LLM170_NO_QSA_SPLIT으로 원경로)
                    if std::env::var_os("LLM170_NO_QSA_SPLIT").is_none() && np_ > std::env::var("LLM170_QSA_TH").ok().and_then(|v| v.parse::<i32>().ok()).unwrap_or(128) {
                        let sg = std::env::var("LLM170_QSA_SEG").ok().and_then(|v| v.parse().ok()).unwrap_or(128usize).max(64);
                        let nseg = (pos0 + t + sg - 1) / sg;
                        // part: [t][n_head][nseg][hd+2]
                        let part = self.ctx.scratch(t * n_head * nseg * (hd + 2) * 4)?;
                        let mut pp2 = part as *mut std::ffi::c_void;
                        let mut sg_a = sg as i32;
                        let mut args = vec![Self::p(&mut qp), Self::p(&mut ckp), Self::p(&mut cvp), Self::p(&mut mp), Self::p(&mut pp2), Self::p(&mut np_), Self::p(&mut nh), Self::p(&mut nk), Self::p(&mut h), Self::p(&mut tl), Self::p(&mut ss), Self::p(&mut p0), Self::p(&mut sg_a)];
                        // q4 다중화 기본: ck/cv 1회 로드로 t 4행 공유 (레지스터 여유 내 최대 배율)
                        // wk는 t>8(프리필) 전용 — 소형 배치(검증 t<=8)는 디코드와 같은
                        // split4q4를 써서 spec/greedy 계약을 구조적으로 만든다.
                        let wk = t > 8 && std::env::var_os("LLM170_NO_WKFLASH").is_none();
                        // hd=256 프리필은 16레인/행 판이 기본 (판정기 17/19 유지).
                        // 3회 평균 pp3314 303.7 vs 종전 299.9 (+1.3%, 구동 잡음 ±1.5%).
                        // LLM170_NO_WK16=1 이면 종전 32레인 판으로 복귀.
                        let wk16 = wk && hd == 256 && std::env::var_os("LLM170_NO_WK16").is_none();
                        if wk16 {
                            if wk && hd == 256 && std::env::var_os("LLM170_NO_WK_WMMA").is_none() {
                                // WMMA 타일 판(기본): Q_in_reg + Q 버퍼를 K/V 로 재사용. 공유 32768B.
                                // pp512 361.3 / pp3314 329.2 vs 스칼라 360.2 / 324.7 (2026-09-12).
                                // 산술이 f16 누적이라 스칼라와 다른데, 토큰 동일성은 600토큰
                                // 다중 청크에서 확인했고 커널 정확성은 wmma-attn-check 가 보증한다.
                                // LLM170_NO_WK_WMMA=1 이면 wk8 로 복귀.
                                self.ctx.launch3_dyn("qsa_flash_wmma", ((t + 63) / 64) as u32, n_head as u32, nseg as u32, 256, 32768, &mut args)?;
                            } else {
                            // hd=256 프리필은 8레인/행 판(셔플 3단)이 기본 — wk16 대비 페어 +2.3%.
                            // 산술(트리 깊이)이 달라 장문 궤적이 갈리지만 커널 정확성은
                            // `llm170 attn-check` 로 보증된다(사용자 결정 2026-09-12).
                            // LLM170_NO_WK8=1 이면 wk16(4단)으로 복귀.
                            if std::env::var_os("LLM170_NO_WK8").is_none() {
                                self.ctx.launch3("qsa_flash_wk8", ((t + 31) / 32) as u32, n_head as u32, nseg as u32, 256, &mut args)?;
                            } else {
                                self.ctx.launch3("qsa_flash_wk16", ((t + 15) / 16) as u32, n_head as u32, nseg as u32, 256, &mut args)?;
                            }
                            }
                        } else {
                            let (kn, gx) = if wk { ("qsa_flash_wk", ((t + 31) / 32) as u32) } else { ("qsa_flash_split4q4", ((t + 3) / 4) as u32) };
                            self.ctx.launch3(kn, gx, n_head as u32, nseg as u32, 256, &mut args)?;
                        }
                        let mut margs = vec![Self::p(&mut qp), Self::p(&mut pp2), Self::p(&mut op), Self::p(&mut np_), Self::p(&mut nh), Self::p(&mut h), Self::p(&mut tl), Self::p(&mut sg_a)];
                        self.ctx.launch3("qsa_flash_merge", t as u32, n_head as u32, 1, 256, &mut margs)?;
                    } else {
                        let mut args = vec![Self::p(&mut qp), Self::p(&mut ckp), Self::p(&mut cvp), Self::p(&mut mp), Self::p(&mut op), Self::p(&mut np_), Self::p(&mut nh), Self::p(&mut nk), Self::p(&mut h), Self::p(&mut tl), Self::p(&mut ss), Self::p(&mut p0)];
                        self.ctx.launch3("qsa_flash", t as u32, n_head as u32, 1, 256, &mut args)?;
                    }
                }
                // wo 배치
                if !self.grp_mmq(&[format!("blk.{il}.attn_output.weight")], t) {
                    self.ctx.quant_q8_b(self.aout_t, self.xq_g_t, n_head * hd, xq_sg, t)?;
                }
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_output.weight"))?;
                self.mm_b2(self.aout_t, self.xq_g_t, xq_sg, wp, ty, ni, no, self.gout_t, t)?;
                full_idx += 1;
            }
self.axpy(self.xs_t, self.gout_t, n * t)?;
            gmark("proj", &mut marks);
            // FFN 배치
            let pw = *self.consts.get(&format!("blk.{il}.post_norm")).ok_or("post_norm")?;
            self.rms_rows(self.xs_t, pw, self.xn_t, n, t)?;
            self.ctx.mmq_y_bump();  // 부록81: xn_t 재기 → quant_y 캐시 무효화
            if !self.grp_mmq(
                &[format!("blk.{il}.ffn_gate.weight"), format!("blk.{il}.ffn_up.weight")],
                t,
            ) {
                self.ctx.quant_q8_b(self.xn_t, self.xq_n_t, n, xq_sn, t)?;
            }
gmark("ffn_quant", &mut marks);
            // 2스트림: gate(사이드) ‖ up(주) — 독립 GEMM, 출력버퍼 분리
            let (wg, tg, nig, nog) = self.w(&format!("blk.{il}.ffn_gate.weight"))?;
            let (wu, tu, niu, nou) = self.w(&format!("blk.{il}.ffn_up.weight"))?;
            let gate_tile = matches!(tg, 12 | 13 | 14 | 23) && t > 64;
            let up_tile = matches!(tu, 12 | 13 | 14 | 23) && t > 64;
            if std::env::var_os("LLM170_PP_PAIRS").is_none() {
                // 기본: 직렬 — 2스트림 페어는 join2(이벤트) 오버헤드가 이득을 넘는다
                // (2026-09-12 A/B: 직렬 +0.75%, LLM170_PP_PAIRS=1로 페어 복원).
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wg, tg, nig, nog, self.fgate_t, t)?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wu, tu, niu, nou, self.fup_t, t)?;
            } else if gate_tile && !up_tile {
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wu, tu, niu, nou, self.fup_t, t)?;
                self.ctx.side_wait_main()?;
                self.mm_b2_s(self.xn_t, self.xq_n_t, xq_sn, wg, tg, nig, nog, self.fgate_t, t)?;
                self.ctx.join2()?;
            } else if !gate_tile && up_tile {
                self.ctx.side_wait_main()?;
                self.mm_b2_s(self.xn_t, self.xq_n_t, xq_sn, wu, tu, niu, nou, self.fup_t, t)?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wg, tg, nig, nog, self.fgate_t, t)?;
                self.ctx.join2()?;
            } else if gate_tile && up_tile {
                // 둘 다 타일 — 하나 사이드
                self.ctx.side_wait_main()?;
                self.mm_b2_s(self.xn_t, self.xq_n_t, xq_sn, wu, tu, niu, nou, self.fup_t, t)?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wg, tg, nig, nog, self.fgate_t, t)?;
                self.ctx.join2()?;
            } else {
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wg, tg, nig, nog, self.fgate_t, t)?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wu, tu, niu, nou, self.fup_t, t)?;
            }
gmark("ffn_gate", &mut marks);
gmark("ffn_up", &mut marks);
            {
                let mut gp = self.fgate_t as *mut std::ffi::c_void;
                let mut up = self.fup_t as *mut std::ffi::c_void;
                let mut op = self.fglu_t as *mut std::ffi::c_void;
                let mut na = (self.n_ff * t) as i32;
                let mut args = vec![Self::p(&mut gp), Self::p(&mut up), Self::p(&mut op), Self::p(&mut na)];
                self.ew_l(if std::env::var("LLM170_F32SILU").as_deref() != Ok("0") { "silu_mul_f32" } else { "silu_mul" }, self.n_ff * t, &mut args)?;
            }
gmark("ffn_silu", &mut marks);
            if std::env::var_os("LLM170_DUMP_XQN").is_some() && il == 0 {
                self.ctx.sync()?;
                let mut bytes = vec![0u8; xq_sn * 4 * t];
                self.ctx.d2h(bytes.as_mut_slice(), self.xq_n_t)?;
                let _ = std::fs::write(std::env::var_os("LLM170_DUMP_XQN").unwrap(), &bytes);
                eprintln!("#  xq_n_t dumped: {} words", xq_sn * t);
            }
            if std::env::var_os("LLM170_RAWHIP_TRACE").is_some() && il == 0 {
                self.ctx.sync()?;
                let mut hf = vec![0f32; self.n_ff * t];
                self.ctx.d2h(bytemuck::cast_slice_mut(&mut hf).as_mut(), self.fgate_t)?;
                let bg = hf.iter().filter(|v| v.is_nan() || v.is_infinite()).count();
                let mut hl = vec![0f32; self.n_ff * t];
                self.ctx.d2h(bytemuck::cast_slice_mut(&mut hl).as_mut(), self.fglu_t)?;
                let bl = hl.iter().filter(|v| v.is_nan() || v.is_infinite()).count();
                eprintln!("#  F0 fgate nan/inf {bg} | fglu nan/inf {bl}");
            }
            if !self.grp_mmq(&[format!("blk.{il}.ffn_down.weight")], t) {
                self.ctx.quant_q8_b(self.fglu_t, self.xq_f_t, self.n_ff, xq_sf, t)?;
            }
gmark("ffn_quant2", &mut marks);
            let (wd, td, nid, nod) = self.w(&format!("blk.{il}.ffn_down.weight"))?;
            self.mm_b2(self.fglu_t, self.xq_f_t, xq_sf, wd, td, nid, nod, self.fdown_t, t)?;
self.axpy(self.xs_t, self.fdown_t, n * t)?;
            gmark("ffn", &mut marks);
            if std::env::var_os("LLM170_RAWHIP_TRACE").is_some() {
                self.ctx.sync()?;
                let mut hv = vec![0f32; n * t];
                self.ctx.d2h(bytemuck::cast_slice_mut(&mut hv).as_mut(), self.xs_t)?;
                let sum0: f64 = hv[..n].iter().map(|&v| v as f64).sum();
                let suml: f64 = hv[n * (t - 1)..].iter().map(|&v| v as f64).sum();
                eprintln!("#  E{il} t={t} xs0={sum0:.6} xs_last={suml:.6}");
            }
        }
        // head — 마지막 토큰만
        let wn = *self.consts.get("output_norm").ok_or("output_norm")?;
        let last = unsafe { self.xs_t.offset(((t - 1) * n * 4) as isize) } as *mut u8;
        {
            let mut xp = last as *mut std::ffi::c_void;
            let mut pp = self.p64 as *mut std::ffi::c_void;
            let mut na = n as i32;
            let mut a1 = vec![Self::p(&mut xp), Self::p(&mut pp), Self::p(&mut na)];
            self.ctx.launch("rms_part", 1, 1, 32, &mut a1)?;
            let mut wp = wn as *mut std::ffi::c_void;
            let mut op = self.xn as *mut std::ffi::c_void;
            let mut ep = self.eps;
            let mut wr = 1i32;
            let mut a2 = vec![Self::p(&mut xp), Self::p(&mut wp), Self::p(&mut pp), Self::p(&mut op), Self::p(&mut ep), Self::p(&mut na), Self::p(&mut wr)];
            self.ctx.launch("rms_finish", 1, 1, 256, &mut a2)?;
        }
        self.ctx.quant_q8(self.xn, self.xq_n, n)?;
        let (wh, th, nih, noh) = self.w("output.weight")?;
        self.mm_into(self.xq_n, wh, th, nih, noh, self.logits)?;
        gmark("head", &mut marks);
        if prof {
            let wall = t0w.elapsed().as_secs_f64() * 1e3;
            eprintln!("cpu_submit={:.1}ms (CPU returned from launches)", wall);
            if let Some((_, last)) = marks.last() {
                unsafe { hip::hipEventSynchronize(*last); }
            }
            let mut acc: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
            for w in marks.windows(2) {
                let mut d = 0f32;
                unsafe { hip::hipEventElapsedTime(&mut d, w[0].1, w[1].1); }
                *acc.entry(w[1].0.clone()).or_insert(0.0) += d;
            }
            let mut v: Vec<_> = acc.into_iter().collect();
            v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let tot: f32 = v.iter().map(|x| x.1).sum();
            eprintln!("pprof t={t} total_marks={tot:.1}ms");
            for (k, ms) in v { eprintln!("  pprof[{k}] {ms:.1}ms"); }
        }
        Ok(Vec::new()) // logits 상주 — d2h는 호출부 선택
    }

    /// logits 전체 d2h (비-greedy 샘플링용).
    pub fn read_logits(&self) -> Result<Vec<f32>, String> {
        let noh = self.n_vocab;
        let mut out = vec![0f32; noh];
        self.ctx.d2h(bytemuck::cast_slice_mut(&mut out).as_mut(), self.logits)?;
        Ok(out)
    }

    /// 마지막 스텝의 최종 hidden (output_norm 전) d2h — MTP 훅용.
    pub fn read_hidden(&self, out: &mut Vec<f32>) -> Result<(), String> {
        out.resize(self.n_embd, 0.0);
        self.ctx.d2h(bytemuck::cast_slice_mut(out).as_mut(), self.xs)
    }

    /// 정규화된 h(공유 head norm 적용됨) → GPU head GEMV + argmax — MTP draft 토큰.
    pub fn mtp_head_argmax(&self, h_normed: &[f32]) -> Result<u32, String> {
        let n = self.n_embd;
        self.ctx.h2d(self.xn, bytemuck::cast_slice(h_normed))?;
        self.ctx.quant_q8(self.xn, self.xq_n, n)?;
        let (wh, th, nih, noh) = self.w("output.weight")?;
        self.mm_into(self.xq_n, wh, th, nih, noh, self.logits)?;
        self.argmax_token()
    }

    /// 배치 검증: 토큰 [pos0..pos0+t) 처리 + 전 토큰 head argmax 반환 (MTP spec).
    /// logits_all 버퍼 [t][vocab] — 행별 argmax는 per-row 발사.
    pub fn verify_batch(
        &self,
        seq: usize,
        pos0: usize,
        emb: &[f32],
        argmaxes: &mut Vec<u32>,
    ) -> Result<(), String> {
        let t = emb.len() / self.n_embd;
        if t > 32 {
            return Err(format!("verify_batch t={t} > 64 (logits_all 상한)"));
        }
        let n = self.n_embd;
        let t_b0 = std::time::Instant::now();
        self.step_batch(seq, pos0, emb)?;
        if std::env::var_os("LLM170_SPEC_TIMING").is_some() {
            eprintln!("[vb] trunk t={t}: {:.1}ms", t_b0.elapsed().as_secs_f64() * 1e3);
        }
        if std::env::var_os("LLM170_SPEC_DBG").is_some() { eprintln!("[vb] step_batch ok"); }
        // head: 전 행 rms → quant → output 타일 → 행별 argmax
        let wn = *self.consts.get("output_norm").ok_or("output_norm")?;
        self.rms_rows(self.xs_t, wn, self.xn_t, n, t)?;
            self.ctx.mmq_y_bump();  // 부록81: xn_t 재기 → quant_y 캐시 무효화
        let xq_sn = n / 4 + n / 32 + n / 16;
        self.ctx.quant_q8_b(self.xn_t, self.xq_n_t, n, xq_sn, t)?;
        let (wh, th, nih, noh) = self.w("output.weight")?;
        if std::env::var_os("LLM170_SPEC_DBG").is_some() { eprintln!("[vb] head tile t={t} ty={th} no={noh}"); }
        if std::env::var_os("LLM170_SPEC_TIMING").is_some() {
            self.ctx.sync()?;
        }
        let t_h0 = std::time::Instant::now();
        self.ctx.gemm_tile_head(
            self.xq_n_t as *const u8,
            wh as *const u8,
            self.ktab2 as *const u8,
            th,
            nih,
            noh,
            xq_sn,
            t,
            self.logits_all,
        )?;
        if std::env::var_os("LLM170_SPEC_TIMING").is_some() {
            self.ctx.sync()?;
            eprintln!("[vb] head mm t={t}: {:.1}ms", t_h0.elapsed().as_secs_f64() * 1e3);
        }
        self.ctx.sync()?;
        if std::env::var_os("LLM170_SPEC_DBG").is_some() { eprintln!("[vb] head ok"); }
        self.ctx.sync()?;
        let t_a0 = std::time::Instant::now();
        argmaxes.clear();
        argmaxes.resize(t, 0);
        // 단일 블록 d2h — 행별 동기 왕복이 사이클당 수백 ms였음 (2026-09-04).
        let mut all_buf = vec![0f32; t * noh];
        let t_d0 = std::time::Instant::now();
        self.ctx.d2h(bytemuck::cast_slice_mut(&mut all_buf).as_mut(), self.logits_all)?;
        if std::env::var_os("LLM170_SPEC_TIMING").is_some() {
            eprintln!("[vb] d2h={:.1}ms ({}MB)", t_d0.elapsed().as_secs_f64() * 1e3, t * noh * 4 / 1048576);
        }
        let t_am0 = std::time::Instant::now();
        for ti in 0..t {
            let mut best = 0usize; let mut bv = f32::NEG_INFINITY;
            for (i, &v) in all_buf[ti * noh..(ti + 1) * noh].iter().enumerate() {
                if v > bv { bv = v; best = i; }
            }
            argmaxes[ti] = best as u32;
        }
        if std::env::var_os("LLM170_SPEC_TIMING").is_some() {
            eprintln!("[vb] argmax_cpu={:.1}ms", t_am0.elapsed().as_secs_f64() * 1e3);
        }

        Ok(())
    }

    /// GDN+conv 상태 GPU 스냅샷 (d2d).
    /// MTP (blk.64) 1스텝 GPU 실행 — CPU mtp_step과 동일 산술 순서.
    /// tok_emb: 토큰 임베딩 행 [n], h: 트렁크 hidden [n], 반환: (argmax, h_next)
    pub fn mtp_step_g(
        &self,
        seq: usize,
        tok_emb: &[f32],
        h_gpu: *mut u8,
        pos: usize,
        with_head: bool,
    ) -> Result<Option<u32>, String> {
        if !self.mtp_on { return Err("mtp_step_gpu: MTP 미로드".into()); }
        let n = self.n_embd;
        let (n_head, n_kv, hd) = (self.n_head, self.n_kv, self.hd);
        let n_ao = n_head * hd; // wo 입력 길이
        assert_eq!(tok_emb.len(), n);
        // 입력 업로드 (h는 GPU 버퍼 직접)
        self.ctx.h2d(self.mtp_e, bytemuck::cast_slice(tok_emb))?;
        let t0s = std::time::Instant::now();
        // enorm → cat[0..n], hnorm → cat[n..2n]
        let en = *self.consts.get("blk.64.nextn.enorm").ok_or("enorm")?;
        let hn = *self.consts.get("blk.64.nextn.hnorm").ok_or("hnorm")?;
        self.rms(self.mtp_e, en, self.mtp_cat, n)?;
        let cat_h = unsafe { self.mtp_cat.add(n * 4) };
        self.rms(h_gpu, hn, cat_h, n)?;
        // eh_proj [2n → n]
        if std::env::var_os("LLM170_MTP_STAGE").is_some() {
            self.ctx.sync()?;
            let mut v = vec![0f32; 2 * n];
            self.ctx.d2h(bytemuck::cast_slice_mut(&mut v).as_mut(), self.mtp_cat)?;
            let (a, b) = v.split_at(n);
            eprintln!("[g] cat e0={:.6} e1={:.6} esum={:.4} | h0={:.6} h1={:.6} hsum={:.4}", a[0], a[1], a.iter().map(|&x| x as f64).sum::<f64>(), b[0], b[1], b.iter().map(|&x| x as f64).sum::<f64>());
        }
        self.quant(self.mtp_cat, self.mtp_xq2, 2 * n)?;
        if std::env::var_os("LLM170_MTP_DBG").is_some() { self.ctx.sync()?; eprintln!("[mtp] quant2 ok"); }
        let (we, te, nie, noe) = self.w("blk.64.nextn.eh_proj.weight")?;
        if std::env::var_os("LLM170_MTP_DBG").is_some() {
            eprintln!("[mtp] eh_proj ty={te} ni={nie} no={noe} w={we:p} xq2={:p} cur={:p} cat={:p}", self.mtp_xq2, self.mtp_cur, self.mtp_cat);
        }
        // RCA 대상: gemv_q8_out 경로가 ni=10240에서만 700 — 직접 launch는 동일 파라미터로
        // 성공(gy 스위프 검증). 동일 직접 경로로 실행 (산술은 gemm_q6k로 동일).
        self.mm_direct(self.mtp_xq2, we, te, nie, noe, self.mtp_cur)?;
        // 진단 덤프 (MTP 헤드 1단계 수치 미러 대조): tok_emb/h/cat/eh
        if let Some(pref) = std::env::var_os("LLM170_MTP_DUMP") {
            let pref = pref.to_string_lossy().to_string();
            self.ctx.sync()?;
            let mut hv = vec![0f32; n];
            self.ctx.d2h(bytemuck::cast_slice_mut(&mut hv).as_mut(), h_gpu)?;
            let mut cv = vec![0f32; 2 * n];
            self.ctx.d2h(bytemuck::cast_slice_mut(&mut cv).as_mut(), self.mtp_cat)?;
            let mut ev = vec![0f32; n];
            self.ctx.d2h(bytemuck::cast_slice_mut(&mut ev).as_mut(), self.mtp_cur)?;
            let mut out = Vec::with_capacity((5 * n + 2 * n) * 4);
            out.extend_from_slice(bytemuck::cast_slice(tok_emb));
            out.extend_from_slice(bytemuck::cast_slice(&hv));
            out.extend_from_slice(bytemuck::cast_slice(&cv));
            out.extend_from_slice(bytemuck::cast_slice(&ev));
            std::fs::write(format!("{pref}.f32"), &out).map_err(|e| e.to_string())?;
            // y(q8) 원시 워드 — quant 레이아웃 검증용 (2n: int8 워드 + 스케일 + qsum)
            // 업로드된 가중치 버퍼 앞부분(GPU) vs 파일 비교용
            let mut wv = vec![0u32; 64];
            self.ctx.d2h(bytemuck::cast_slice_mut(&mut wv).as_mut(), we)?;
            std::fs::write(format!("{pref}.wfirst.u32"), bytemuck::cast_slice(&wv)).map_err(|e| e.to_string())?;
            // 중간/끝 구간도 대조 (부분 업로드 탐지): row 2500 / row 5119 시작
            let off_mid = 2500usize * 40 * 210;
            let off_end = 5119usize * 40 * 210;
            let mut wm = vec![0u32; 64];
            self.ctx.d2h(bytemuck::cast_slice_mut(&mut wm).as_mut(), unsafe { we.add(off_mid) })?;
            std::fs::write(format!("{pref}.wmid.u32"), bytemuck::cast_slice(&wm)).map_err(|e| e.to_string())?;
            let mut we2 = vec![0u32; 64];
            self.ctx.d2h(bytemuck::cast_slice_mut(&mut we2).as_mut(), unsafe { we.add(off_end) })?;
            std::fs::write(format!("{pref}.wend.u32"), bytemuck::cast_slice(&we2)).map_err(|e| e.to_string())?;
            let xq_words = 2 * n / 4 + 2 * n / 32 + 2 * n / 16;
            let mut xv = vec![0u32; xq_words];
            self.ctx.d2h(bytemuck::cast_slice_mut(&mut xv).as_mut(), self.mtp_xq2)?;
            std::fs::write(format!("{pref}.xq.u32"), bytemuck::cast_slice(&xv)).map_err(|e| e.to_string())?;
        }
        if std::env::var_os("LLM170_MTP_STAGE").is_some() {
            self.ctx.sync()?;
            let mut v = vec![0f32; n];
            self.ctx.d2h(bytemuck::cast_slice_mut(&mut v).as_mut(), self.mtp_cur)?;
            eprintln!("[g] eh sum={:.5} x0={:.5} x1={:.5}", v.iter().map(|&x| x as f64).sum::<f64>(), v[0], v[1]);
        }
        // (구 gemv_q8_out 경로 mm_into 호출 제거 — RCA: ni=10240에서 오값이
        //  mm_direct 결과를 덮어써 MTP 초안 품질이 무너졌다. 2026-09-12)
        // attn_norm → q/k/v
        let an = *self.consts.get("blk.64.attn_norm").ok_or("attn_norm")?;
        self.rms(self.mtp_cur, an, self.mtp_e, n)?;
        self.quant(self.mtp_e, self.mtp_xq, n)?;
        let (wq, tq, niq, noq) = self.w("blk.64.attn_q.weight")?;
        self.mm_into(self.mtp_xq, wq, tq, niq, noq, self.aq)?;
        let (wk, tk, nik, nok) = self.w("blk.64.attn_k.weight")?;
        self.mm_into(self.mtp_xq, wk, tk, nik, nok, self.ak)?;
        let (wv, tv, niv, nov) = self.w("blk.64.attn_v.weight")?;
        self.mm_into(self.mtp_xq, wv, tv, niv, nov, self.av)?;
        // q/k norm+rope
        let qn = *self.consts.get("blk.64.attn_q_norm").ok_or("qn")?;
        let kn = *self.consts.get("blk.64.attn_k_norm").ok_or("kn")?;
        let cs = *self.consts.get("cs").ok_or("cs")?;
        {
            let mut qp = self.aq as *mut std::ffi::c_void;
            let mut kp = self.ak as *mut std::ffi::c_void;
            let mut qwp = qn as *mut std::ffi::c_void;
            let mut kwp = kn as *mut std::ffi::c_void;
            let mut csp = cs as *mut std::ffi::c_void;
            let mut ep = self.eps;
            let mut kq = self.kq_scale;
            let mut pp = pos as i32;
            let mut nh = n_head as i32;
            let mut nk = n_kv as i32;
            let mut hh = hd as i32;
            let mut nr = self.n_rot as i32;
            let rows = n_head + n_kv;
            let mut args = vec![Self::p(&mut qp), Self::p(&mut kp), Self::p(&mut qwp), Self::p(&mut kwp), Self::p(&mut csp), Self::p(&mut ep), Self::p(&mut kq), Self::p(&mut pp), Self::p(&mut nh), Self::p(&mut nk), Self::p(&mut hh), Self::p(&mut nr)];
            self.ctx.launch("qk_norm_rope", rows as u32, 1, 32, &mut args)?;
        }
        // MTP KV append
        self.copy(self.ak, self.mtp_kv_k[seq], 0, pos * n_kv * hd, n_kv * hd)?;
        self.copy(self.av, self.mtp_kv_v[seq], 0, pos * n_kv * hd, n_kv * hd)?;
        // flash attention (np = pos+1)
        {
            let n_past = pos + 1;
            let mask = self.consts.get("mask").copied().ok_or("mask")?;
            let mut qp = self.aq as *mut std::ffi::c_void;
            let mut ckp = self.mtp_kv_k[seq] as *mut std::ffi::c_void;
            let mut cvp = self.mtp_kv_v[seq] as *mut std::ffi::c_void;
            let mut mp = mask as *mut std::ffi::c_void;
            let mut op = self.mtp_ao as *mut std::ffi::c_void;
            let mut np_ = n_past as i32;
            let mut nh = n_head as i32;
            let mut nk = n_kv as i32;
            let mut hh = hd as i32;
            let mut tl = 1i32;
            let mut ss = self.ctx_len as i32;
            let mut p0 = pos as i32;
            let mut args = vec![Self::p(&mut qp), Self::p(&mut ckp), Self::p(&mut cvp), Self::p(&mut mp), Self::p(&mut op), Self::p(&mut np_), Self::p(&mut nh), Self::p(&mut nk), Self::p(&mut hh), Self::p(&mut tl), Self::p(&mut ss), Self::p(&mut p0)];
            self.ctx.launch3("qsa_flash", 1, n_head as u32, 1, 256, &mut args)?;
        }
        // wo + 잔차 (입력 길이 = n_head*hd)
        self.quant(self.mtp_ao, self.mtp_xq, n_ao)?;
        let (wo, two, nio, noo) = self.w("blk.64.attn_output.weight")?;
        self.mm_direct(self.mtp_xq, wo, two, nio, noo, self.gout)?;
        if std::env::var_os("LLM170_MTP_STAGE").is_some() {
            self.ctx.sync()?;
            let mut v = vec![0f32; n];
            self.ctx.d2h(bytemuck::cast_slice_mut(&mut v).as_mut(), self.gout)?;
            eprintln!("[g] wo sum={:.5} x0={:.5}", v.iter().map(|&x| x as f64).sum::<f64>(), v[0]);
        }
        self.axpy(self.mtp_cur, self.gout, n)?;
        // FFN
        let pn = *self.consts.get("blk.64.post_attention_norm").ok_or("post_norm")?;
        self.rms(self.mtp_cur, pn, self.mtp_e, n)?;
        self.quant(self.mtp_e, self.mtp_xq, n)?;
        let (wg, tg, nig, nog) = self.w("blk.64.ffn_gate.weight")?;
        self.mm_into(self.mtp_xq, wg, tg, nig, nog, self.fgate)?;
        let (wu, tu, niu, nou) = self.w("blk.64.ffn_up.weight")?;
        self.mm_into(self.mtp_xq, wu, tu, niu, nou, self.fup)?;
        {
            let mut gp = self.fgate as *mut std::ffi::c_void;
            let mut up = self.fup as *mut std::ffi::c_void;
            let mut op = self.fglu as *mut std::ffi::c_void;
            let mut na = self.n_ff as i32;
            let mut args = vec![Self::p(&mut gp), Self::p(&mut up), Self::p(&mut op), Self::p(&mut na)];
            self.ew_l("silu_mul", self.n_ff, &mut args)?;
        }
        self.quant(self.fglu, self.xq_f, self.n_ff)?;
        let (wd, td, nid, nod) = self.w("blk.64.ffn_down.weight")?;
        self.mm_into(self.xq_f, wd, td, nid, nod, self.fdown)?;
        self.axpy(self.mtp_cur, self.fdown, n)?;
        if std::env::var_os("LLM170_MTP_STAGE").is_some() {
            self.ctx.sync()?;
            let mut v = vec![0f32; n];
            self.ctx.d2h(bytemuck::cast_slice_mut(&mut v).as_mut(), self.mtp_cur)?;
            eprintln!("[g] ff sum={:.5} x0={:.5}", v.iter().map(|&x| x as f64).sum::<f64>(), v[0]);
        }
        if !with_head {
            if std::env::var_os("LLM170_SPEC_TIMING").is_some() {
                eprintln!("[mt] step(nohead)={:.2}ms", t0s.elapsed().as_secs_f64() * 1e3);
            }
            return Ok(None);
        }
        // shared head norm → output head → argmax
        let shn = *self.consts.get("blk.64.nextn.shared_head_norm").ok_or("shn")?;
        self.rms(self.mtp_cur, shn, self.mtp_e, n)?;
        let t0h = std::time::Instant::now();
        let am = self.head_argmax_gpu(self.mtp_e)?;
        if std::env::var_os("LLM170_SPEC_TIMING").is_some() {
            eprintln!("[mt] head={:.2}ms", t0s.elapsed().as_secs_f64() * 1e3);
        }
        Ok(Some(am))
    }

    /// MTP 프리필 배치 — blk.64를 t행 한 번에 처리 (t=1 스텝 × 토큰수 대체).
    /// tok_embs: [t][n] 토큰 임베딩, h_shift: [t][n] = [h_prev, h_all[0..t-1]].
    /// 반환: 마지막 행의 드래프트 토큰 (헤드는 마지막 행만).
    pub fn mtp_prefill_batch(
        &self,
        seq: usize,
        tok_embs: &[f32],
        carry_h: &[f32],
        t: usize,
        pos0: usize,
        with_head: bool,
    ) -> Result<u32, String> {
        if !self.mtp_on {
            return Err("mtp_prefill_batch: MTP 미로드".into());
        }
        let n = self.n_embd;
        let (n_head, n_kv, hd, n_rot) = (self.n_head, self.n_kv, self.hd, self.n_rot);
        if t == 0 || t > self.t_max_mtp {
            return Err(format!("mtp_prefill_batch: t={t} 범위 밖"));
        }
        let xq2_w = 2 * n / 4 + 2 * n / 32 + 2 * n / 16;
        let xq_n = n / 4 + n / 32 + n / 16;
        let xq_sf = self.n_ff / 4 + self.n_ff / 32 + self.n_ff / 16;
        let n_ao = n_head * hd; // attn_output 입력 길이
        let xq_sg = n_ao / 4 + n_ao / 32 + n_ao / 16;
        let mask = self.consts.get("mask").copied().ok_or("mask")?;
        let t_mtp = std::time::Instant::now();
        let mtp_time = std::env::var_os("LLM170_MTP_TIMING").is_some();
        let mut mark = |label: &str, last: &mut std::time::Instant| {
            if mtp_time {
                self.ctx.sync().ok();
                eprintln!("[mtpb] {label}: {:.2}ms", last.elapsed().as_secs_f64() * 1e3);
                *last = std::time::Instant::now();
            }
        };
        let mut cp = std::time::Instant::now();
        // KV-only 프리필 제어: 프롬프트 행의 attention/wo/FFN 출력은 쓰이지 않는다
        // (헤드는 마지막 행, 체인은 디코드 h 사용). LLM170_MTP_FULL=1이면 전행.
        let full = std::env::var_os("LLM170_MTP_FULL").is_some();
        let qstride = n_head * 2 * hd;
        let ostride = n_head * hd;
        let nrow_attn = if full { t } else { 1 };
        let qoff = if full { 0 } else { (t - 1) * qstride };
        let ooff = if full { 0 } else { (t - 1) * ostride };
        // ① enorm(tok) ‖ hnorm(h_{p-1}) → cat [t][2n]  (mtp_b_cur/mtp_b_e는 임시)
        if self.mtp_prefetched.swap(false, std::sync::atomic::Ordering::SeqCst) {
            self.ctx.join2()?;   // 사이드 h2d 완료 대기 (메인 프리필과 중첩됨)
        } else {
            self.ctx.h2d(self.mtp_b_e, bytemuck::cast_slice(tok_embs))?;
        }
        // h_shift는 device에서 조립 — src=본체 hidden(xs_t), carry=이전 청크 마지막 행.
        // (호스트 왕복 2×t·n·4B 제거)
        if carry_h.len() != n {
            return Err(format!("mtp_prefill_batch: carry_h len {} != {n}", carry_h.len()));
        }
        self.ctx.h2d(self.mtp_h, bytemuck::cast_slice(carry_h))?;
        {
            let mut sp = self.xs_t as *mut std::ffi::c_void;
            let mut cp2 = self.mtp_h as *mut std::ffi::c_void;
            let mut dp = self.mtp_b_hs as *mut std::ffi::c_void;
            let mut na = n as i32;
            let mut ta = t as i32;
            let mut args = vec![Self::p(&mut sp), Self::p(&mut cp2), Self::p(&mut dp), Self::p(&mut na), Self::p(&mut ta)];
            let gx = ((n * t) as u32).div_ceil(256);
            self.ctx.launch3("row_shift_gather", gx, 1, 1, 256, &mut args)?;
        }
        let en = *self.consts.get("blk.64.nextn.enorm").ok_or("enorm")?;
        let hn = *self.consts.get("blk.64.nextn.hnorm").ok_or("hnorm")?;
        self.rms_rows(self.mtp_b_e, en, self.mtp_b_cur, n, t)?;
        self.rms_rows(self.mtp_b_hs, hn, self.mtp_b_e, n, t)?;
        {
            let mut ep = self.mtp_b_cur as *mut std::ffi::c_void;
            let mut hp = self.mtp_b_e as *mut std::ffi::c_void;
            let mut op = self.mtp_b_cat as *mut std::ffi::c_void;
            let mut na = n as i32;
            let mut ta = t as i32;
            let gx = (n.div_ceil(256)) as u32;
            let mut args = vec![
                Self::p(&mut ep), Self::p(&mut hp), Self::p(&mut op),
                Self::p(&mut na), Self::p(&mut ta),
            ];
            self.ctx.launch3("cat2_rows", gx, t as u32, 1, 256, &mut args)?;
        }
        mark("norms+cat", &mut cp);
        // ② eh_proj [2n → n]
        self.ctx.quant_q8_b(self.mtp_b_cat, self.mtp_b_xq2, 2 * n, xq2_w, t)?;
        let (we, te, nie, noe) = self.w("blk.64.nextn.eh_proj.weight")?;
        self.mm_b2(
            self.mtp_b_cat as *mut u8, self.mtp_b_xq2, xq2_w, we, te, nie, noe,
            self.mtp_b_cur, t,
        )?;
        mark("eh_proj", &mut cp);
        // ③ attn_norm → q/k/v
        let an = *self.consts.get("blk.64.attn_norm").ok_or("attn_norm")?;
        self.rms_rows(self.mtp_b_cur, an, self.mtp_b_e, n, t)?;
        self.ctx.quant_q8_b(self.mtp_b_e, self.mtp_b_xqn, n, xq_n, t)?;
        // q 투영은 attention(=종료 청크)에서만 필요 — KV는 k/v만 적립한다.
        let (wq, tq, niq, noq) = self.w("blk.64.attn_q.weight")?;
        if with_head {
            self.mm_b2(self.mtp_b_e as *mut u8, self.mtp_b_xqn, xq_n, wq, tq, niq, noq, self.aq_t, t)?;
        }
        let (wk, tk, nik, nok) = self.w("blk.64.attn_k.weight")?;
        self.mm_b2(self.mtp_b_e as *mut u8, self.mtp_b_xqn, xq_n, wk, tk, nik, nok, self.ak_t, t)?;
        let (wv, tv, niv, nov) = self.w("blk.64.attn_v.weight")?;
        self.mm_b2(self.mtp_b_e as *mut u8, self.mtp_b_xqn, xq_n, wv, tv, niv, nov, self.av_t, t)?;
        mark("qkv", &mut cp);
        // ④ q/k norm+rope (배치, pos+y) + KV 적립 + flash (배치)
        let qn = *self.consts.get("blk.64.attn_q_norm").ok_or("qn")?;
        let kn = *self.consts.get("blk.64.attn_k_norm").ok_or("kn")?;
        let cs = *self.consts.get("cs").ok_or("cs")?;
        {
            let mut qp = self.aq_t as *mut std::ffi::c_void;
            let mut kp = self.ak_t as *mut std::ffi::c_void;
            let mut qwp = qn as *mut std::ffi::c_void;
            let mut kwp = kn as *mut std::ffi::c_void;
            let mut csp = cs as *mut std::ffi::c_void;
            let mut ep = self.eps;
            let mut kq = self.kq_scale;
            let mut pp = pos0 as i32;
            let mut nh = n_head as i32;
            let mut nk = n_kv as i32;
            let mut h = hd as i32;
            let mut nr = n_rot as i32;
            let rows = n_head + n_kv;
            let mut args = vec![
                Self::p(&mut qp), Self::p(&mut kp), Self::p(&mut qwp), Self::p(&mut kwp),
                Self::p(&mut csp), Self::p(&mut ep), Self::p(&mut kq), Self::p(&mut pp),
                Self::p(&mut nh), Self::p(&mut nk), Self::p(&mut h), Self::p(&mut nr),
            ];
            self.ctx.launch3("qk_norm_rope", rows as u32, t as u32, 1, 32, &mut args)?;
        }
        for (src, dst) in [
            (self.ak_t, self.mtp_kv_k[seq]),
            (self.av_t, self.mtp_kv_v[seq]),
        ] {
            let mut sp = src as *mut std::ffi::c_void;
            let mut dp = dst as *mut std::ffi::c_void;
            let mut na = (n_kv * hd) as i32;
            let mut p0 = pos0 as i32;
            let mut args = vec![Self::p(&mut sp), Self::p(&mut dp), Self::p(&mut na), Self::p(&mut p0)];
            self.ctx.launch3("kv_append_t", (n_kv * hd).div_ceil(64) as u32, t as u32, 1, 64, &mut args)?;
        }
        {
            // KV-only: 원소 i의 MTP층 출력은 (a) 헤드에서 마지막 행만, (b) 체인은
            // 디코드 스텝의 h를 쓰므로 프롬프트 행들의 attention/wo/FFN은 불필요.
            // 인과 구조상 마지막 행의 출력은 앞 행들의 *KV*만 필요하다 (이미 적립).
            // LLM170_MTP_FULL=1이면 종전 전행 경로.
            let mut qp = unsafe { self.aq_t.add(qoff * 4) } as *mut std::ffi::c_void;
            let mut ckp = self.mtp_kv_k[seq] as *mut std::ffi::c_void;
            let mut cvp = self.mtp_kv_v[seq] as *mut std::ffi::c_void;
            let mut mp = mask as *mut std::ffi::c_void;
            let mut op = unsafe { self.aout_t.add(ooff * 4) } as *mut std::ffi::c_void;
            let mut np_ = (pos0 + t) as i32;
            let mut nh = n_head as i32;
            let mut nk = n_kv as i32;
            let mut h = hd as i32;
            let mut tl = nrow_attn as i32;
            let mut ss = self.ctx_len as i32;
            let mut p0 = (pos0 + t - nrow_attn) as i32;
            if std::env::var_os("LLM170_NO_QSA_SPLIT").is_none()
                && np_ > std::env::var("LLM170_QSA_TH").ok().and_then(|v| v.parse::<i32>().ok()).unwrap_or(128)
            {
                let sg = std::env::var("LLM170_QSA_SEG").ok().and_then(|v| v.parse().ok()).unwrap_or(128usize).max(64);
                let nseg = (pos0 + t + sg - 1) / sg;
                let part = self.ctx.scratch(nrow_attn * n_head * nseg * (hd + 2) * 4)?;
                let mut pp2 = part as *mut std::ffi::c_void;
                let mut sg_a = sg as i32;
                let mut args = vec![
                    Self::p(&mut qp), Self::p(&mut ckp), Self::p(&mut cvp), Self::p(&mut mp),
                    Self::p(&mut pp2), Self::p(&mut np_), Self::p(&mut nh), Self::p(&mut nk),
                    Self::p(&mut h), Self::p(&mut tl), Self::p(&mut ss), Self::p(&mut p0),
                    Self::p(&mut sg_a),
                ];
                let wk = nrow_attn > 8 && std::env::var_os("LLM170_NO_WKFLASH").is_none();
                let (kn2, gx) = if wk { ("qsa_flash_wk", (nrow_attn.div_ceil(32)) as u32) } else { ("qsa_flash_split4q4", (nrow_attn.div_ceil(4)) as u32) };
                self.ctx.launch3(kn2, gx, n_head as u32, nseg as u32, 256, &mut args)?;
                let mut margs = vec![
                    Self::p(&mut qp), Self::p(&mut pp2), Self::p(&mut op), Self::p(&mut np_),
                    Self::p(&mut nh), Self::p(&mut h), Self::p(&mut tl), Self::p(&mut sg_a),
                ];
                self.ctx.launch3("qsa_flash_merge", nrow_attn as u32, n_head as u32, 1, 256, &mut margs)?;
            } else {
                let mut args = vec![
                    Self::p(&mut qp), Self::p(&mut ckp), Self::p(&mut cvp), Self::p(&mut mp),
                    Self::p(&mut op), Self::p(&mut np_), Self::p(&mut nh), Self::p(&mut nk),
                    Self::p(&mut h), Self::p(&mut tl), Self::p(&mut ss), Self::p(&mut p0),
                ];
                self.ctx.launch3("qsa_flash", nrow_attn as u32, n_head as u32, 1, 256, &mut args)?;
            }
        }
        mark("attn+kv", &mut cp);
        // 중간 청크(with_head=false)는 KV 적립만 — 마지막 행의 attention/FFN/헤드도
        // 아무도 읽지 않는다 (초안은 프롬프트 종료 청크에서만 필요).
        if !with_head {
            if mtp_time {
                eprintln!("[mtpb] TOTAL {:.2}ms (t={t}, kv-only)", t_mtp.elapsed().as_secs_f64() * 1e3);
            }
            return Ok(0);
        }
        // ⑤⑥ KV-only: 마지막 행만 (앞 행들의 wo/FFN 출력은 아무도 쓰지 않는다)
        let nrow_ffn = if full { t } else { 1 };
        let coff = if full { 0 } else { (t - 1) * n };
        let cur_p = unsafe { self.mtp_b_cur.add(coff * 4) } as *mut u8;
        let aout_p = unsafe { self.aout_t.add(ooff * 4) } as *mut u8;
        let gout_p = unsafe { self.gout_t.add(coff * 4) } as *mut u8;
        // ⑤ attn_output + 잔차
        self.ctx.quant_q8_b(aout_p, self.mtp_b_xqn, n_head * hd, xq_sg, nrow_ffn)?;
        let (wo, two, nio, noo) = self.w("blk.64.attn_output.weight")?;
        self.mm_b2(
            aout_p, self.mtp_b_xqn, xq_sg, wo, two, nio, noo,
            gout_p, nrow_ffn,
        )?;
        self.axpy(cur_p, gout_p, n * nrow_ffn)?;
        // ⑥ FFN + 잔차
        let pn = *self.consts.get("blk.64.post_attention_norm").ok_or("post_norm")?;
        self.rms_rows(cur_p, pn, self.mtp_b_e, n, nrow_ffn)?;
        self.ctx.quant_q8_b(self.mtp_b_e, self.mtp_b_xqn, n, xq_n, nrow_ffn)?;
        let (wg, tg, nig, nog) = self.w("blk.64.ffn_gate.weight")?;
        self.mm_b2(self.mtp_b_e as *mut u8, self.mtp_b_xqn, xq_n, wg, tg, nig, nog, self.fgate_t, nrow_ffn)?;
        let (wu, tu, niu, nou) = self.w("blk.64.ffn_up.weight")?;
        self.mm_b2(self.mtp_b_e as *mut u8, self.mtp_b_xqn, xq_n, wu, tu, niu, nou, self.fup_t, nrow_ffn)?;
        {
            let mut gp = self.fgate_t as *mut std::ffi::c_void;
            let mut up = self.fup_t as *mut std::ffi::c_void;
            let mut op = self.fglu_t as *mut std::ffi::c_void;
            let mut na = (self.n_ff * nrow_ffn) as i32;
            let mut args = vec![Self::p(&mut gp), Self::p(&mut up), Self::p(&mut op), Self::p(&mut na)];
            self.ew_l(
                if std::env::var("LLM170_F32SILU").as_deref() != Ok("0") { "silu_mul_f32" } else { "silu_mul" },
                self.n_ff * nrow_ffn,
                &mut args,
            )?;
        }
        self.ctx.quant_q8_b(self.fglu_t, self.mtp_b_xq2, self.n_ff, xq_sf, nrow_ffn)?;
        let (wd, td, nid, nod) = self.w("blk.64.ffn_down.weight")?;
        self.mm_b2(
            self.fglu_t as *mut u8, self.mtp_b_xq2, xq_sf, wd, td, nid, nod,
            self.fdown_t, nrow_ffn,
        )?;
        self.axpy(cur_p, self.fdown_t, n * nrow_ffn)?;
        mark("ffn", &mut cp);
        // ⑦ 마지막 행만 헤드 — 공유 head norm → output GEMV → argmax
        let shn = *self.consts.get("blk.64.nextn.shared_head_norm").ok_or("shn")?;
        let last = unsafe { self.mtp_b_cur.add((t - 1) * n * 4) };
        self.rms(last, shn, self.mtp_e, n)?;
        let am = self.head_argmax_gpu(self.mtp_e)?;
        if mtp_time {
            eprintln!("[mtpb] TOTAL {:.2}ms (t={})", t_mtp.elapsed().as_secs_f64() * 1e3, t);
        }
        Ok(am)
    }

    /// MTP 1스텝 (호스트 h, head, h_next 회수) — 프리필/디코드 훅용.
    pub fn mtp_step_gpu(        &self,
        seq: usize,
        tok_emb: &[f32],
        h: &[f32],
        pos: usize,
    ) -> Result<(u32, Vec<f32>), String> {
        self.ctx.h2d(self.mtp_h, bytemuck::cast_slice(h))?;
        let am = self
            .mtp_step_g(seq, tok_emb, self.mtp_h, pos, true)?
            .ok_or_else(|| "mtp head".to_string())?;
        let mut h_next = vec![0f32; self.n_embd];
        self.ctx
            .d2h(bytemuck::cast_slice_mut(&mut h_next).as_mut(), self.mtp_cur)?;
        Ok((am, h_next))
    }

    /// MTP 체인 스텝 — h를 내부 mtp_cur(직전 h_next)에서 직접 읽음.
    pub fn mtp_step_chain(
        &self,
        seq: usize,
        tok_emb: &[f32],
        pos: usize,
    ) -> Result<u32, String> {
        self.mtp_step_g(seq, tok_emb, self.mtp_cur, pos, true)?
            .ok_or("mtp head".to_string())
    }

    /// MTP 상태 진행 스텝 (호스트 trunk h, head 없음) — spec 수용 후 KV 동기.
    pub fn mtp_step_adv(
        &self,
        seq: usize,
        tok_emb: &[f32],
        h: &[f32],
        pos: usize,
    ) -> Result<(), String> {
        self.ctx.h2d(self.mtp_h, bytemuck::cast_slice(h))?;
        self.mtp_step_g(seq, tok_emb, self.mtp_h, pos, false)?;
        Ok(())
    }

    /// 정규화 입력 x → output GEMV → GPU argmax
    fn head_argmax_gpu(&self, x: *mut u8) -> Result<u32, String> {
        let n = self.n_embd;
        self.quant(x, self.mtp_xq, n)?;
        let (wo, to, nio, noo) = self.w("output.weight")?;
        self.mm_into(self.mtp_xq, wo, to, nio, noo, self.logits)?;
        if std::env::var_os("LLM170_MTP_STAGE").is_some() {
            self.ctx.sync()?;
            let mut v = vec![0f32; 8];
            self.ctx.d2h(bytemuck::cast_slice_mut(&mut v).as_mut(), self.logits)?;
            let mut hn8 = vec![0f32; 8];
            self.ctx.d2h(bytemuck::cast_slice_mut(&mut hn8).as_mut(), x)?;
            eprintln!("[g] head L0..7={:?} hnorm0..3={:?}", v, &hn8[0..4]);
        }
        let mut xp = self.logits as *mut std::ffi::c_void;
        let mut np_ = noo as i32;
        let mut op = self.ctx.scratch(16)?;
        let mut args = vec![Self::p(&mut xp), Self::p(&mut np_), Self::p(&mut op)];
        self.ctx.launch("argmax64", 1, 1, 64, &mut args)?;
        let mut b8 = [0u8; 8]; // int2: x=최댓값, y=인덱스
        self.ctx.d2h(&mut b8, op)?;
        Ok(u32::from_le_bytes([b8[4], b8[5], b8[6], b8[7]]))
    }

    /// np 배치 디코드 — GEMM/요소커널은 t=n_seqs 행 공유, 상태커널(conv/AR/rope/KV/flash)은
    /// 행 슬라이스로 per-seq 실행 (plans/15). 반환: seq별 logits.
    pub fn step_batch_np(
        &self,
        seqs: &[usize],
        poss: &[u32],
        emb: &[f32],
    ) -> Result<Vec<Vec<f32>>, String> {
        let t = seqs.len();
        let n = self.n_embd;
        debug_assert_eq!(emb.len(), t * n);
        let (n_head, n_kv, hd, n_rot) = (self.n_head, self.n_kv, self.hd, self.n_rot);
        // 레이아웃 (DecodeState 필드 준거): conv_ch = d_inner + 2·n_group·d_state,
        // q/k 행 = n_group·d_state, v 행 = dt_rank·d_state(=d_inner).
        let (conv_ch, k_len, v_len) = (self.conv_ch, self.k_len, self.v_len);
        let d_inner = self.d_inner;
        let xq_sn = n / 4 + n / 32 + n / 16;
        let xq_sf = self.n_ff / 4 + self.n_ff / 32 + self.n_ff / 16;
        let xq_sg = d_inner / 4 + d_inner / 32 + d_inner / 16;
        self.ctx.h2d(self.xs_t, bytemuck::cast_slice(emb))?;
        let mut recr_idx = 0usize;
        let mut full_idx = 0usize;
        let mask = self.consts.get("mask").copied().ok_or("mask")?;
        for il in 0..self.n_layer {
            let wn = *self.consts.get(&format!("blk.{il}.attn_norm")).ok_or("attn_norm")?;
            self.rms_rows(self.xs_t, wn, self.xn_t, n, t)?;
            self.ctx.quant_q8_b(self.xn_t, self.xq_n_t, n, xq_sn, t)?;
            if self.is_recr[il] {
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_qkv.weight"))?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wp, ty, ni, no, self.gqkv_t, t)?;
                let (wg2, tg2, nig2, nog2) = self.w(&format!("blk.{il}.attn_gate.weight"))?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wg2, tg2, nig2, nog2, self.gz_t, t)?;
                let (wb2, tb2, nib2, nob2) = self.w(&format!("blk.{il}.ssm_beta.weight"))?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wb2, tb2, nib2, nob2, self.gb_t, t)?;
                let (wa2, ta2, nia2, noa2) = self.w(&format!("blk.{il}.ssm_alpha.weight"))?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wa2, ta2, nia2, noa2, self.ga_t, t)?;
                let cw = *self.consts.get(&format!("blk.{il}.conv_w")).ok_or("conv_w")?;
                let dtb = *self.consts.get(&format!("blk.{il}.dt_bias")).ok_or("dtb")?;
                let ssa = *self.consts.get(&format!("blk.{il}.ssm_a")).ok_or("ssa")?;
                let snorm = *self.consts.get(&format!("blk.{il}.ssm_norm")).ok_or("ssm_norm")?;
                // conv — per-seq (t=1, 행 슬라이스)
                for s in 0..t {
                    let row = unsafe { self.gqkv_t.add(s * conv_ch * 4) };
                    let mut qp = row as *mut std::ffi::c_void;
                    let mut cp = cw as *mut std::ffi::c_void;
                    let mut sp = self.st_conv[recr_idx][seqs[s]] as *mut std::ffi::c_void;
                    let mut op = unsafe { self.gconv_t.add(s * conv_ch * 4) } as *mut std::ffi::c_void;
                    let mut ch = conv_ch as i32;
                    let mut kk = self.conv_k as i32;
                    let mut tt = 1i32;
                    let mut args = vec![Self::p(&mut qp), Self::p(&mut cp), Self::p(&mut sp), Self::p(&mut op), Self::p(&mut ch), Self::p(&mut kk), Self::p(&mut tt)];
                    self.ctx.launch3("gdn_conv_t", conv_ch as u32, 1, 1, 32, &mut args)?;
                }
                // split3 공유 (행별 요소)
                {
                    let mut sp = self.gconv_t as *mut std::ffi::c_void;
                    let mut q0 = self.gq_t as *mut std::ffi::c_void;
                    let mut q1 = self.gk_t as *mut std::ffi::c_void;
                    let mut q2 = self.gv_t as *mut std::ffi::c_void;
                    let mut n0 = k_len as i32;
                    let mut n1 = k_len as i32;
                    let mut n2 = v_len as i32;
                    let total = (2 * k_len + v_len) * t;
                    let mut args = vec![Self::p(&mut sp), Self::p(&mut q0), Self::p(&mut q1), Self::p(&mut q2), Self::p(&mut n0), Self::p(&mut n1), Self::p(&mut n2)];
                    self.ew_l("split3", total, &mut args)?;
                }
                // l2 공유
                {
                    let scale = 1.0f32 / (self.d_state as f32).sqrt();
                    let mut qp = self.gq_t as *mut std::ffi::c_void;
                    let mut kp = self.gk_t as *mut std::ffi::c_void;
                    let mut ep = self.eps;
                    let mut sc = scale;
                    let mut d = self.d_state as i32;
                    let mut ng = self.n_group as i32;
                    let mut args = vec![Self::p(&mut qp), Self::p(&mut kp), Self::p(&mut ep), Self::p(&mut sc), Self::p(&mut d), Self::p(&mut ng)];
                    self.ctx.launch3("l2_rows2_scale_w", (2 * self.n_group) as u32, t as u32, 1, 32, &mut args)?;
                }
                // beta/e^g 공유
                {
                    let mut bp = self.gb_t as *mut std::ffi::c_void;
                    let mut ap = self.ga_t as *mut std::ffi::c_void;
                    let mut dp = dtb as *mut std::ffi::c_void;
                    let mut sp2 = ssa as *mut std::ffi::c_void;
                    let mut bgp = self.gbg_t as *mut std::ffi::c_void;
                    let mut nh = (self.dt_rank * t) as i32;
                    let mut dr = self.dt_rank as i32;
                    let mut args = vec![Self::p(&mut bp), Self::p(&mut ap), Self::p(&mut dp), Self::p(&mut sp2), Self::p(&mut bgp), Self::p(&mut nh), Self::p(&mut dr)];
                    self.ew_l(if std::env::var("LLM170_F32SILU").as_deref() != Ok("0") { "gdn_beta_g_f32" } else { "gdn_beta_g" }, self.dt_rank * t, &mut args)?;
                }
                // AR — per-seq (t=1 슬라이스)
                for s in 0..t {
                    let mut sp3 = self.st_gdn[recr_idx][seqs[s]] as *mut std::ffi::c_void;
                    let mut qp = unsafe { self.gq_t.add(s * k_len * 4) } as *mut std::ffi::c_void;
                    let mut kp = unsafe { self.gk_t.add(s * k_len * 4) } as *mut std::ffi::c_void;
                    let mut vp = unsafe { self.gv_t.add(s * v_len * 4) } as *mut std::ffi::c_void;
                    let mut bgp = unsafe { self.gbg_t.add(s * self.dt_rank * 2 * 4) } as *mut std::ffi::c_void;
                    let mut op = unsafe { self.go_t.add(s * v_len * 4) } as *mut std::ffi::c_void;
                    let mut d = self.d_state as i32;
                    let mut ks = k_len as i32;
                    let mut vs = v_len as i32;
                    let mut hv = self.dt_rank as i32;
                    let mut hk = self.n_group as i32;
                    let mut asc = 1.0f32 / (self.d_state as f32).sqrt();
                    let mut tt = 1i32;
                    let mut args = vec![Self::p(&mut sp3), Self::p(&mut qp), Self::p(&mut kp), Self::p(&mut vp), Self::p(&mut bgp), Self::p(&mut op), Self::p(&mut d), Self::p(&mut ks), Self::p(&mut vs), Self::p(&mut hv), Self::p(&mut hk), Self::p(&mut asc), Self::p(&mut tt)];
                    self.ctx.launch3("gdn_ar_w", self.dt_rank as u32, self.d_state as u32, 1, 32, &mut args)?;
                }
                if std::env::var_os("LLM170_NP_DBG6").is_some() && il == 0 {
                    self.ctx.sync()?;
                    if std::env::var_os("LLM170_NP_DBG7").is_some() {
                        let mut hq2 = vec![0f32; conv_ch * t];
                        self.ctx.d2h(bytemuck::cast_slice_mut(&mut hq2).as_mut(), self.gqkv_t)?;
                        let mut hc = vec![0f32; conv_ch * t];
                        self.ctx.d2h(bytemuck::cast_slice_mut(&mut hc).as_mut(), self.gconv_t)?;
                        let cs = |v: &[f32]| v.iter().map(|&x| x as f64).sum::<f64>();
                        eprintln!("[g0q] qkv=[{:.4},{:.4}] conv=[{:.4},{:.4}]",
                            cs(&hq2[..conv_ch]), cs(&hq2[conv_ch..]),
                            cs(&hc[..conv_ch]), cs(&hc[conv_ch..]));
                    }
                    let mut hq = vec![0f32; k_len * t];
                    self.ctx.d2h(bytemuck::cast_slice_mut(&mut hq).as_mut(), self.gq_t)?;
                    let mut hk2 = vec![0f32; k_len * t];
                    self.ctx.d2h(bytemuck::cast_slice_mut(&mut hk2).as_mut(), self.gk_t)?;
                    let mut hv2 = vec![0f32; v_len * t];
                    self.ctx.d2h(bytemuck::cast_slice_mut(&mut hv2).as_mut(), self.gv_t)?;
                    let mut ho = vec![0f32; v_len * t];
                    self.ctx.d2h(bytemuck::cast_slice_mut(&mut ho).as_mut(), self.go_t)?;
                    let cs = |v: &[f32]| v.iter().map(|&x| x as f64).sum::<f64>();
                    eprintln!("[g0] q=[{:.4},{:.4}] k=[{:.4},{:.4}] v=[{:.4},{:.4}] ar_out=[{:.4},{:.4}]",
                        cs(&hq[..k_len.min(hq.len())]), cs(&hq[k_len.min(hq.len())..]),
                        cs(&hk2[..k_len.min(hk2.len())]), cs(&hk2[k_len.min(hk2.len())..]),
                        cs(&hv2[..v_len.min(hv2.len())]), cs(&hv2[v_len.min(hv2.len())..]),
                        cs(&ho[..v_len.min(ho.len())]), cs(&ho[v_len.min(ho.len())..]));
                }
                // norm_gated 공유
                {
                    let mut op = self.go_t as *mut std::ffi::c_void;
                    let mut zp = self.gz_t as *mut std::ffi::c_void;
                    let mut wp = snorm as *mut std::ffi::c_void;
                    let mut outp = self.ggated_t as *mut std::ffi::c_void;
                    let mut ep = self.eps;
                    let mut d = self.d_state as i32;
                    let mut nh = self.dt_rank as i32;
                    let mut args = vec![Self::p(&mut op), Self::p(&mut zp), Self::p(&mut wp), Self::p(&mut outp), Self::p(&mut ep), Self::p(&mut d), Self::p(&mut nh)];
                    self.ctx.launch3(if std::env::var("LLM170_F32SILU").as_deref() != Ok("0") { "norm_gated_silu_f32" } else { "norm_gated_silu" }, self.dt_rank as u32, t as u32, 1, 32, &mut args)?;
                }
                self.ctx.quant_q8_b(self.ggated_t, self.xq_g_t, self.d_inner, xq_sg, t)?;
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.ssm_out.weight"))?;
                self.mm_b2(self.ggated_t, self.xq_g_t, xq_sg, wp, ty, ni, no, self.gout_t, t)?;
                recr_idx += 1;
            } else {
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_q.weight"))?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wp, ty, ni, no, self.aq_t, t)?;
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_k.weight"))?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wp, ty, ni, no, self.ak_t, t)?;
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_v.weight"))?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wp, ty, ni, no, self.av_t, t)?;
                let qn = *self.consts.get(&format!("blk.{il}.attn_q_norm")).ok_or("qn")?;
                let kn = *self.consts.get(&format!("blk.{il}.attn_k_norm")).ok_or("kn")?;
                let cs = *self.consts.get("cs").ok_or("cs")?;
                for s in 0..t {
                    let aq_row = unsafe { self.aq_t.add(s * n_head * 2 * hd * 4) };
                    let ak_row = unsafe { self.ak_t.add(s * n_kv * hd * 4) };
                    // rope per-seq (t=1)
                    {
                        let mut qp = aq_row as *mut std::ffi::c_void;
                        let mut kp = ak_row as *mut std::ffi::c_void;
                        let mut qwp = qn as *mut std::ffi::c_void;
                        let mut kwp = kn as *mut std::ffi::c_void;
                        let mut csp = cs as *mut std::ffi::c_void;
                        let mut ep = self.eps;
                        let mut kq = self.kq_scale;
                        let mut pp = poss[s] as i32;
                        let mut nh = n_head as i32;
                        let mut nk = n_kv as i32;
                        let mut h = hd as i32;
                        let mut nr = n_rot as i32;
                        let rows = n_head + n_kv;
                        let mut args = vec![Self::p(&mut qp), Self::p(&mut kp), Self::p(&mut qwp), Self::p(&mut kwp), Self::p(&mut csp), Self::p(&mut ep), Self::p(&mut kq), Self::p(&mut pp), Self::p(&mut nh), Self::p(&mut nk), Self::p(&mut h), Self::p(&mut nr)];
                        self.ctx.launch3("qk_norm_rope", rows as u32, 1, 1, 32, &mut args)?;
                    }
                    // KV append per-seq
                    let pos = poss[s] as usize;
                    let av_row = unsafe { self.av_t.add(s * n_kv * hd * 4) };
                    for (src, table) in [(ak_row, &self.kv_k), (av_row, &self.kv_v)] {
                        let mut sp = src as *mut std::ffi::c_void;
                        let mut dp = table[full_idx][seqs[s]] as *mut std::ffi::c_void;
                        let mut na = (n_kv * hd) as i32;
                        let mut p0 = pos as i32;
                        let mut args = vec![Self::p(&mut sp), Self::p(&mut dp), Self::p(&mut na), Self::p(&mut p0)];
                        self.ctx.launch3("kv_append_t", (n_kv * hd).div_ceil(64) as u32, 1, 1, 64, &mut args)?;
                    }
                    // flash per-seq (t=1)
                    {
                        let mut qp = aq_row as *mut std::ffi::c_void;
                        let mut ckp = self.kv_k[full_idx][seqs[s]] as *mut std::ffi::c_void;
                        let mut cvp = self.kv_v[full_idx][seqs[s]] as *mut std::ffi::c_void;
                        let mut mp = mask as *mut std::ffi::c_void;
                        let mut op = unsafe { self.aout_t.add(s * n_head * hd * 4) } as *mut std::ffi::c_void;
                        let mut np_ = (pos + 1) as i32;
                        let mut nh = n_head as i32;
                        let mut nk = n_kv as i32;
                        let mut h = hd as i32;
                        let mut tl = 1i32;
                        let mut ss = self.ctx_len as i32;
                        let mut p0 = pos as i32;
                        let mut args = vec![Self::p(&mut qp), Self::p(&mut ckp), Self::p(&mut cvp), Self::p(&mut mp), Self::p(&mut op), Self::p(&mut np_), Self::p(&mut nh), Self::p(&mut nk), Self::p(&mut h), Self::p(&mut tl), Self::p(&mut ss), Self::p(&mut p0)];
                        self.ctx.launch3("qsa_flash", 1, n_head as u32, 1, 256, &mut args)?;
                    }
                }
                if !self.grp_mmq(&[format!("blk.{il}.attn_output.weight")], t) {
                    self.ctx.quant_q8_b(self.aout_t, self.xq_g_t, n_head * hd, xq_sg, t)?;
                }
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_output.weight"))?;
                self.mm_b2(self.aout_t, self.xq_g_t, xq_sg, wp, ty, ni, no, self.gout_t, t)?;
                full_idx += 1;
            }
            self.axpy(self.xs_t, self.gout_t, n * t)?;
            // FFN 공유
            let pw = *self.consts.get(&format!("blk.{il}.post_norm")).ok_or("post_norm")?;
            self.rms_rows(self.xs_t, pw, self.xn_t, n, t)?;
            if !self.grp_mmq(
                &[format!("blk.{il}.ffn_gate.weight"), format!("blk.{il}.ffn_up.weight")],
                t,
            ) {
                self.ctx.quant_q8_b(self.xn_t, self.xq_n_t, n, xq_sn, t)?;
            }
            let (wg, tg, nig, nog) = self.w(&format!("blk.{il}.ffn_gate.weight"))?;
            self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wg, tg, nig, nog, self.fgate_t, t)?;
            let (wu, tu, niu, nou) = self.w(&format!("blk.{il}.ffn_up.weight"))?;
            self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wu, tu, niu, nou, self.fup_t, t)?;
            {
                let mut gp = self.fgate_t as *mut std::ffi::c_void;
                let mut up = self.fup_t as *mut std::ffi::c_void;
                let mut op = self.fglu_t as *mut std::ffi::c_void;
                let mut na = (self.n_ff * t) as i32;
                let mut args = vec![Self::p(&mut gp), Self::p(&mut up), Self::p(&mut op), Self::p(&mut na)];
                self.ew_l(if std::env::var("LLM170_F32SILU").as_deref() != Ok("0") { "silu_mul_f32" } else { "silu_mul" }, self.n_ff * t, &mut args)?;
            }
            if !self.grp_mmq(&[format!("blk.{il}.ffn_down.weight")], t) {
                self.ctx.quant_q8_b(self.fglu_t, self.xq_f_t, self.n_ff, xq_sf, t)?;
            }
            let (wd, td, nid, nod) = self.w(&format!("blk.{il}.ffn_down.weight"))?;
            self.mm_b2(self.fglu_t, self.xq_f_t, xq_sf, wd, td, nid, nod, self.fdown_t, t)?;
            self.axpy(self.xs_t, self.fdown_t, n * t)?;
            if std::env::var_os("LLM170_NP_DBG3").is_some() && il % 8 == 0 {
                self.ctx.sync()?;
                let mut hv = vec![0f32; n * t];
                self.ctx.d2h(bytemuck::cast_slice_mut(&mut hv).as_mut(), self.xs_t)?;
                let r0 = &hv[..n];
                let r1 = &hv[n.min(hv.len())..];
                eprintln!(
                    "[npl] L{il} r0={:.4} r1={:.4}",
                    r0.iter().map(|&v| v as f64).sum::<f64>(),
                    r1.iter().map(|&v| v as f64).sum::<f64>()
                );
            }
        }
        // head — 전체 t행 logits (j128 강제 타일) → 행별 d2h
        let wn = *self.consts.get("output_norm").ok_or("output_norm")?;
        self.rms_rows(self.xs_t, wn, self.xn_t, n, t)?;
        self.ctx.quant_q8_b(self.xn_t, self.xq_n_t, n, xq_sn, t)?;
        let (wh, th, nih, noh) = self.w("output.weight")?;
        // 소형 t(2..=4)는 4-토큰 GEMV — 타일 헤드는 128열 고정이라 t=4에서 18%를 먹는다
        // (1.04GB q6_K 헤드를 j128 타일로 읽어 29.7ms/스텝, 실측).
        if (2..=4).contains(&t)
            && th == 14
            && std::env::var_os("LLM170_NO_G4").is_none()
        {
            self.ctx.gemm_g4(
                th,
                self.xq_n_t as *const u8,
                wh as *const u8,
                self.ktab2 as *const u8,
                nih,
                noh,
                xq_sn,
                t,
                self.logits_all,
            )?;
        } else {
            self.ctx.gemm_tile_head(
                self.xq_n_t as *const u8,
                wh as *const u8,
                self.ktab2 as *const u8,
                th,
                nih,
                noh,
                xq_sn,
                t,
                self.logits_all,
            )?;
        }
        let mut out = Vec::with_capacity(t);
        let mut row = vec![0f32; noh];
        for s in 0..t {
            let src = unsafe { self.logits_all.offset((s * noh * 4) as isize) } as *const u8;
            self.ctx.d2h(bytemuck::cast_slice_mut(&mut row).as_mut(), src)?;
            out.push(row.clone());
        }
        Ok(out)
    }

    /// np×spec 병합 verify (plans/18) — seq-major 행 그룹. group_starts[i] = seq_i 그룹의
    /// 첫 행 인덱스, group_starts 끝은 t. 행별 argmax 반환 (logits_all 재사용).
    #[allow(clippy::too_many_arguments)]
    pub fn verify_batch_ms(
        &self,
        seqs: &[usize],
        poss: &[usize],
        group_starts: &[usize],
        emb: &[f32],
        argmaxes: &mut Vec<u32>,
        h_all: &mut Vec<f32>,
    ) -> Result<(), String> {
        let t = emb.len() / self.n_embd;
        if t > 64 {
            return Err(format!("verify_batch_ms t={t} > 64"));
        }
        let n = self.n_embd;
        let (n_head, n_kv, hd, n_rot) = (self.n_head, self.n_kv, self.hd, self.n_rot);
        let conv_ch = self.conv_ch;
        let k_len = self.k_len;
        let v_len = self.v_len;
        let d_inner = self.d_inner;
        let xq_sn = n / 4 + n / 32 + n / 16;
        let xq_sf = self.n_ff / 4 + self.n_ff / 32 + self.n_ff / 16;
        let xq_sg = d_inner / 4 + d_inner / 32 + d_inner / 16;

        // 행 메타데이터 (호스트 조립 → 소형 업로드)
        let mut row_seq = vec![0i32; t];
        let mut row_pos = vec![0i32; t];
        let mut seg_start = vec![0i32; t];
        let mut seg_end = vec![0i32; t];
        let mut row_np = vec![0i32; t];
        for gi in 0..group_starts.len() {
            let (g0, g1) = (group_starts[gi], if gi + 1 < group_starts.len() { group_starts[gi + 1] } else { t });
            for r in g0..g1 {
                row_seq[r] = seqs[gi] as i32;
                row_pos[r] = (poss[gi] + (r - g0)) as i32;
                seg_start[r] = g0 as i32;
                seg_end[r] = g1 as i32;
                row_np[r] = row_pos[r] + 1; // 절대 위치+1 (attention 범위)
            }
        }
        self.h2d_i32_ms(&row_seq, &row_pos, &seg_start, &seg_end, &row_np)?;
        // per-row KV 포인터 (레이어별) — 업로드는 레이어 루프 내 (kv_k[il][seq])
        self.ctx.h2d(self.xs_t, bytemuck::cast_slice(emb))?;
        let mut recr_idx = 0usize;
        let mut full_idx = 0usize;
        let mask = self.consts.get("mask").copied().ok_or("mask")?;
        let cs = *self.consts.get("cs").ok_or("cs")?;
        for il in 0..self.n_layer {
            let wn = *self.consts.get(&format!("blk.{il}.attn_norm")).ok_or("attn_norm")?;
            self.rms_rows(self.xs_t, wn, self.xn_t, n, t)?;
            self.ctx.quant_q8_b(self.xn_t, self.xq_n_t, n, xq_sn, t)?;
            if self.is_recr[il] {
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_qkv.weight"))?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wp, ty, ni, no, self.gqkv_t, t)?;
                let (wg2, tg2, nig2, nog2) = self.w(&format!("blk.{il}.attn_gate.weight"))?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wg2, tg2, nig2, nog2, self.gz_t, t)?;
                let (wb2, tb2, nib2, nob2) = self.w(&format!("blk.{il}.ssm_beta.weight"))?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wb2, tb2, nib2, nob2, self.gb_t, t)?;
                let (wa2, ta2, nia2, noa2) = self.w(&format!("blk.{il}.ssm_alpha.weight"))?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wa2, ta2, nia2, noa2, self.ga_t, t)?;
                if il == 0 {
                    self.trace_rows("ms_gqkv", self.gqkv_t, conv_ch, t)?;
                }
                let cw = *self.consts.get(&format!("blk.{il}.conv_w")).ok_or("conv_w")?;
                let dtb = *self.consts.get(&format!("blk.{il}.dt_bias")).ok_or("dtb")?;
                let ssa = *self.consts.get(&format!("blk.{il}.ssm_a")).ok_or("ssa")?;
                let snorm = *self.consts.get(&format!("blk.{il}.ssm_norm")).ok_or("ssm_norm")?;
                // conv _ms (행별 seq 상태)
                {
                    let mut qp = self.gqkv_t as *mut std::ffi::c_void;
                    let mut cp = cw as *mut std::ffi::c_void;
                    let mut sp = self.ms_conv_ptr(recr_idx, &row_seq)? as *mut std::ffi::c_void;
                    let mut op = self.gconv_t as *mut std::ffi::c_void;
                    let mut ch = conv_ch as i32;
                    let mut kk = self.conv_k as i32;
                    let mut tt = t as i32;
                    let mut rs = self.ms_rowseq as *mut std::ffi::c_void;
                    let mut sg = self.ms_segstart as *mut std::ffi::c_void;
                    let mut args = vec![Self::p(&mut qp), Self::p(&mut cp), Self::p(&mut sp), Self::p(&mut op), Self::p(&mut ch), Self::p(&mut kk), Self::p(&mut tt), Self::p(&mut rs), Self::p(&mut sg)];
                    self.ctx.launch3(if std::env::var("LLM170_F32SILU").as_deref() != Ok("0") { "gdn_conv_t2_ms_f32" } else { "gdn_conv_t2_ms" }, conv_ch.div_ceil(64) as u32, t as u32, 1, 64, &mut args)?;
                    let mut qp2 = self.gqkv_t as *mut std::ffi::c_void;
                    let mut sp2 = self.ms_conv_ptr(recr_idx, &row_seq)? as *mut std::ffi::c_void;
                    let mut ch2 = conv_ch as i32;
                    let mut kk2 = self.conv_k as i32;
                    let mut tt2 = t as i32;
                    let mut rs2 = self.ms_rowseq as *mut std::ffi::c_void;
                    let mut se = self.ms_segend as *mut std::ffi::c_void;
                    let mut args2 = vec![Self::p(&mut qp2), Self::p(&mut sp2), Self::p(&mut ch2), Self::p(&mut kk2), Self::p(&mut tt2), Self::p(&mut rs2), Self::p(&mut se)];
                    self.ctx.launch3("gdn_conv_state_ms", t as u32, conv_ch.div_ceil(64) as u32, 1, 64, &mut args2)?;
                }
                if il == 0 {
                    self.trace_rows("ms_gconv", self.gconv_t, conv_ch, t)?;
                }
                // split3 공유
                {
                    let mut sp = self.gconv_t as *mut std::ffi::c_void;
                    let mut q0 = self.gq_t as *mut std::ffi::c_void;
                    let mut q1 = self.gk_t as *mut std::ffi::c_void;
                    let mut q2 = self.gv_t as *mut std::ffi::c_void;
                    let mut n0 = k_len as i32;
                    let mut n1 = k_len as i32;
                    let mut n2 = v_len as i32;
                    let total = (2 * k_len + v_len) * t;
                    let mut args = vec![Self::p(&mut sp), Self::p(&mut q0), Self::p(&mut q1), Self::p(&mut q2), Self::p(&mut n0), Self::p(&mut n1), Self::p(&mut n2)];
                    self.ew_l("split3", total, &mut args)?;
                }
                // l2 공유
                {
                    let scale = 1.0f32 / (self.d_state as f32).sqrt();
                    let mut qp = self.gq_t as *mut std::ffi::c_void;
                    let mut kp = self.gk_t as *mut std::ffi::c_void;
                    let mut ep = self.eps;
                    let mut sc = scale;
                    let mut d = self.d_state as i32;
                    let mut ng = self.n_group as i32;
                    let mut args = vec![Self::p(&mut qp), Self::p(&mut kp), Self::p(&mut ep), Self::p(&mut sc), Self::p(&mut d), Self::p(&mut ng)];
                    self.ctx.launch3("l2_rows2_scale_w", (2 * self.n_group) as u32, t as u32, 1, 32, &mut args)?;
                }
                // beta/e^g 공유
                {
                    let mut bp = self.gb_t as *mut std::ffi::c_void;
                    let mut ap = self.ga_t as *mut std::ffi::c_void;
                    let mut dp = dtb as *mut std::ffi::c_void;
                    let mut sp2 = ssa as *mut std::ffi::c_void;
                    let mut bgp = self.gbg_t as *mut std::ffi::c_void;
                    let mut nh = (self.dt_rank * t) as i32;
                    let mut dr = self.dt_rank as i32;
                    let mut args = vec![Self::p(&mut bp), Self::p(&mut ap), Self::p(&mut dp), Self::p(&mut sp2), Self::p(&mut bgp), Self::p(&mut nh), Self::p(&mut dr)];
                    self.ew_l(if std::env::var("LLM170_F32SILU").as_deref() != Ok("0") { "gdn_beta_g_f32" } else { "gdn_beta_g" }, self.dt_rank * t, &mut args)?;
                }
                                // AR — per-seq 슬라이스 (gdn_ar_w_ms 비대칭 RCA 회피 — 트렁크 검증 커널 재사용)
                for gi in 0..group_starts.len() {
                    let (g0, g1) = (group_starts[gi], if gi + 1 < group_starts.len() { group_starts[gi + 1] } else { t });
                    let gt = g1 - g0;
                    let sq = seqs[gi];
                    let mut sp3 = self.st_gdn[recr_idx][sq] as *mut std::ffi::c_void;
                    let mut qp = unsafe { self.gq_t.add(g0 * k_len * 4) } as *mut std::ffi::c_void;
                    let mut kp = unsafe { self.gk_t.add(g0 * k_len * 4) } as *mut std::ffi::c_void;
                    let mut vp = unsafe { self.gv_t.add(g0 * v_len * 4) } as *mut std::ffi::c_void;
                    let mut bgp = unsafe { self.gbg_t.add(g0 * self.dt_rank * 2 * 4) } as *mut std::ffi::c_void;
                    let mut op = unsafe { self.go_t.add(g0 * v_len * 4) } as *mut std::ffi::c_void;
                    let mut d = self.d_state as i32;
                    let mut ks = k_len as i32;
                    let mut vs = v_len as i32;
                    let mut hv = self.dt_rank as i32;
                    let mut hk = self.n_group as i32;
                    let mut asc = 1.0f32 / (self.d_state as f32).sqrt();
                    let mut tt = gt as i32;
                    let mut args = vec![Self::p(&mut sp3), Self::p(&mut qp), Self::p(&mut kp), Self::p(&mut vp), Self::p(&mut bgp), Self::p(&mut op), Self::p(&mut d), Self::p(&mut ks), Self::p(&mut vs), Self::p(&mut hv), Self::p(&mut hk), Self::p(&mut asc), Self::p(&mut tt)];
                    self.ctx.launch3("gdn_ar_w", self.dt_rank as u32, self.d_state as u32, 1, 32, &mut args)?;
                }
                if il == 0 {
                    self.trace_rows("ms_go", self.go_t, v_len, t)?;
                }
// norm_gated 공유
                {
                    let mut op = self.go_t as *mut std::ffi::c_void;
                    let mut zp = self.gz_t as *mut std::ffi::c_void;
                    let mut wp = snorm as *mut std::ffi::c_void;
                    let mut outp = self.ggated_t as *mut std::ffi::c_void;
                    let mut ep = self.eps;
                    let mut d = self.d_state as i32;
                    let mut nh = self.dt_rank as i32;
                    let mut args = vec![Self::p(&mut op), Self::p(&mut zp), Self::p(&mut wp), Self::p(&mut outp), Self::p(&mut ep), Self::p(&mut d), Self::p(&mut nh)];
                    self.ctx.launch3(if std::env::var("LLM170_F32SILU").as_deref() != Ok("0") { "norm_gated_silu_f32" } else { "norm_gated_silu" }, self.dt_rank as u32, t as u32, 1, 32, &mut args)?;
                }
                self.ctx.quant_q8_b(self.ggated_t, self.xq_g_t, self.d_inner, xq_sg, t)?;
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.ssm_out.weight"))?;
                self.mm_b2(self.ggated_t, self.xq_g_t, xq_sg, wp, ty, ni, no, self.gout_t, t)?;
                recr_idx += 1;
            } else {
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_q.weight"))?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wp, ty, ni, no, self.aq_t, t)?;
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_k.weight"))?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wp, ty, ni, no, self.ak_t, t)?;
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_v.weight"))?;
                self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wp, ty, ni, no, self.av_t, t)?;
                let qn = *self.consts.get(&format!("blk.{il}.attn_q_norm")).ok_or("qn")?;
                let kn = *self.consts.get(&format!("blk.{il}.attn_k_norm")).ok_or("kn")?;
                                // per-seq 슬라이스 (np 검증 경로 — _ms 의심 바이팩)
                for gi in 0..group_starts.len() {
                    let (g0, g1) = (group_starts[gi], if gi + 1 < group_starts.len() { group_starts[gi + 1] } else { t });
                    let gt = g1 - g0;
                    let sq = seqs[gi];
                    let aq_row = unsafe { self.aq_t.add(g0 * n_head * 2 * hd * 4) };
                    let ak_row = unsafe { self.ak_t.add(g0 * n_kv * hd * 4) };
                    {
                        let mut qp = aq_row as *mut std::ffi::c_void;
                        let mut kp = ak_row as *mut std::ffi::c_void;
                        let mut qwp = qn as *mut std::ffi::c_void;
                        let mut kwp = kn as *mut std::ffi::c_void;
                        let mut csp = cs as *mut std::ffi::c_void;
                        let mut ep = self.eps;
                        let mut kq = self.kq_scale;
                        let mut pp = (poss[gi]) as i32;
                        let mut nh = n_head as i32;
                        let mut nk = n_kv as i32;
                        let mut h = hd as i32;
                        let mut nr = n_rot as i32;
                        let rows = n_head + n_kv;
                        let mut args = vec![Self::p(&mut qp), Self::p(&mut kp), Self::p(&mut qwp), Self::p(&mut kwp), Self::p(&mut csp), Self::p(&mut ep), Self::p(&mut kq), Self::p(&mut pp), Self::p(&mut nh), Self::p(&mut nk), Self::p(&mut h), Self::p(&mut nr)];
                        self.ctx.launch3("qk_norm_rope", rows as u32, gt as u32, 1, 32, &mut args)?;
                    }
                    let pos0 = poss[gi];
                    let av_row = unsafe { self.av_t.add(g0 * n_kv * hd * 4) };
                    for (src, table) in [(ak_row, &self.kv_k), (av_row, &self.kv_v)] {
                        let mut sp = src as *mut std::ffi::c_void;
                        let mut dp = table[full_idx][sq] as *mut std::ffi::c_void;
                        let mut na = (n_kv * hd) as i32;
                        let mut p0 = pos0 as i32;
                        let mut args = vec![Self::p(&mut sp), Self::p(&mut dp), Self::p(&mut na), Self::p(&mut p0)];
                        self.ctx.launch3("kv_append_t", (n_kv * hd).div_ceil(64) as u32, gt as u32, 1, 64, &mut args)?;
                    }
                    {
                        let mut qp = aq_row as *mut std::ffi::c_void;
                        let mut ckp = self.kv_k[full_idx][sq] as *mut std::ffi::c_void;
                        let mut cvp = self.kv_v[full_idx][sq] as *mut std::ffi::c_void;
                        let mut mp = mask as *mut std::ffi::c_void;
                        let mut op = unsafe { self.aout_t.add(g0 * n_head * hd * 4) } as *mut std::ffi::c_void;
                        let mut np_ = (pos0 + gt) as i32;
                        let mut nh = n_head as i32;
                        let mut nk = n_kv as i32;
                        let mut h = hd as i32;
                        let mut tl = gt as i32;
                        let mut ss = self.ctx_len as i32;
                        let mut p0 = pos0 as i32;
                        let mut args = vec![Self::p(&mut qp), Self::p(&mut ckp), Self::p(&mut cvp), Self::p(&mut mp), Self::p(&mut op), Self::p(&mut np_), Self::p(&mut nh), Self::p(&mut nk), Self::p(&mut h), Self::p(&mut tl), Self::p(&mut ss), Self::p(&mut p0)];
                        self.ctx.launch3("qsa_flash", gt as u32, n_head as u32, 1, 256, &mut args)?;
                    }
                }
self.ctx.quant_q8_b(self.aout_t, self.xq_g_t, n_head * hd, xq_sg, t)?;
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_output.weight"))?;
                self.mm_b2(self.aout_t, self.xq_g_t, xq_sg, wp, ty, ni, no, self.gout_t, t)?;
                if full_idx == 1 {
                    self.trace_rows("ms_aout", self.aout_t, n_head * hd, t)?;
                }
                full_idx += 1;
            }
            self.axpy(self.xs_t, self.gout_t, n * t)?;
            // FFN 공유
            let pw = *self.consts.get(&format!("blk.{il}.post_norm")).ok_or("post_norm")?;
            self.rms_rows(self.xs_t, pw, self.xn_t, n, t)?;
            if !self.grp_mmq(
                &[format!("blk.{il}.ffn_gate.weight"), format!("blk.{il}.ffn_up.weight")],
                t,
            ) {
                self.ctx.quant_q8_b(self.xn_t, self.xq_n_t, n, xq_sn, t)?;
            }
            let (wg, tg, nig, nog) = self.w(&format!("blk.{il}.ffn_gate.weight"))?;
            self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wg, tg, nig, nog, self.fgate_t, t)?;
            let (wu, tu, niu, nou) = self.w(&format!("blk.{il}.ffn_up.weight"))?;
            self.mm_b2(self.xn_t, self.xq_n_t, xq_sn, wu, tu, niu, nou, self.fup_t, t)?;
            {
                let mut gp = self.fgate_t as *mut std::ffi::c_void;
                let mut up = self.fup_t as *mut std::ffi::c_void;
                let mut op = self.fglu_t as *mut std::ffi::c_void;
                let mut na = (self.n_ff * t) as i32;
                let mut args = vec![Self::p(&mut gp), Self::p(&mut up), Self::p(&mut op), Self::p(&mut na)];
                self.ew_l(if std::env::var("LLM170_F32SILU").as_deref() != Ok("0") { "silu_mul_f32" } else { "silu_mul" }, self.n_ff * t, &mut args)?;
            }
            if !self.grp_mmq(&[format!("blk.{il}.ffn_down.weight")], t) {
                self.ctx.quant_q8_b(self.fglu_t, self.xq_f_t, self.n_ff, xq_sf, t)?;
            }
            let (wd, td, nid, nod) = self.w(&format!("blk.{il}.ffn_down.weight"))?;
            self.mm_b2(self.fglu_t, self.xq_f_t, xq_sf, wd, td, nid, nod, self.fdown_t, t)?;
            self.axpy(self.xs_t, self.fdown_t, n * t)?;
        }
        // head (j128 강제) + 행별 argmax
        let wn = *self.consts.get("output_norm").ok_or("output_norm")?;
        self.rms_rows(self.xs_t, wn, self.xn_t, n, t)?;
        self.ctx.quant_q8_b(self.xn_t, self.xq_n_t, n, xq_sn, t)?;
        let (wh, th, nih, noh) = self.w("output.weight")?;
        self.ctx.gemm_tile_head(
            self.xq_n_t as *const u8,
            wh as *const u8,
            self.ktab2 as *const u8,
            th,
            nih,
            noh,
            xq_sn,
            t,
            self.logits_all,
        )?;
        argmaxes.clear();
        argmaxes.resize(t, 0);
        let mut all_buf = vec![0f32; t * noh];
        self.ctx.d2h(bytemuck::cast_slice_mut(&mut all_buf).as_mut(), self.logits_all)?;
        for ti in 0..t {
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &v) in all_buf[ti * noh..(ti + 1) * noh].iter().enumerate() {
                if v > bv {
                    bv = v;
                    best = i;
                }
            }
            argmaxes[ti] = best as u32;
        }
        if std::env::var_os("LLM170_MS_LOGITS").is_some() {
            for ti in 0..t {
                let mut top: Vec<(u32, f32)> = all_buf[ti * noh..(ti + 1) * noh]
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (i as u32, v))
                    .collect();
                top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                top.truncate(8);
                eprintln!("[mslg] row{ti}: {}", top.iter()
                    .map(|(i, v)| format!("{i}:{v:.3}")).collect::<Vec<_>>().join(" "));
            }
        }
        h_all.clear();
        h_all.resize(t * self.n_embd, 0.0);
        self.ctx
            .d2h(bytemuck::cast_slice_mut(h_all).as_mut(), self.xs_t)?;
        Ok(())
    }

    /// 행 메타 업로드 (i32 5종) — 고정 ms 버퍼.
    fn h2d_i32_ms(
        &self,
        row_seq: &[i32],
        row_pos: &[i32],
        seg_start: &[i32],
        seg_end: &[i32],
        row_np: &[i32],
    ) -> Result<(), String> {
        self.ctx.h2d(self.ms_rowseq, bytemuck::cast_slice(row_seq))?;
        self.ctx.h2d(self.ms_rowpos, bytemuck::cast_slice(row_pos))?;
        self.ctx.h2d(self.ms_segstart, bytemuck::cast_slice(seg_start))?;
        self.ctx.h2d(self.ms_segend, bytemuck::cast_slice(seg_end))?;
        self.ctx.h2d(self.ms_rownp, bytemuck::cast_slice(row_np))?;
        Ok(())
    }

    /// 레이어별 per-seq 상태 포인터 테이블 업로드 (t 엔트리 — 행 → 자기 seq 상태).
    fn ms_conv_ptr(&self, il: usize, row_seq: &[i32]) -> Result<*mut u8, String> {
        let tbl: Vec<*mut u8> = row_seq.iter().map(|&sq| self.st_conv[il][sq as usize]).collect();
        let raw: Vec<usize> = tbl.iter().map(|&p| p as usize).collect();
        self.ctx.h2d(self.ms_ptrbuf, bytemuck::cast_slice(&raw))?;
        Ok(self.ms_ptrbuf)
    }

    fn ms_gdn_ptr(&self, il: usize, row_seq: &[i32]) -> Result<*mut u8, String> {
        let tbl: Vec<*mut u8> = row_seq.iter().map(|&sq| self.st_gdn[il][sq as usize]).collect();
        let raw: Vec<usize> = tbl.iter().map(|&p| p as usize).collect();
        self.ctx.h2d(self.ms_ptrbuf, bytemuck::cast_slice(&raw))?;
        Ok(self.ms_ptrbuf)
    }

    fn ms_kvk_ptr(&self, il: usize, row_seq: &[i32]) -> Result<*mut u8, String> {
        self.ms_kvk_ptr_to(il, row_seq, self.ms_ptrbuf)
    }

    fn ms_kvk_ptr_to(&self, il: usize, row_seq: &[i32], dst: *mut u8) -> Result<*mut u8, String> {
        let tbl: Vec<*mut u8> = row_seq.iter().map(|&sq| self.kv_k[il][sq as usize]).collect();
        let raw: Vec<usize> = tbl.iter().map(|&p| p as usize).collect();
        self.ctx.h2d(dst, bytemuck::cast_slice(&raw))?;
        Ok(dst)
    }

    fn ms_kvv_ptr_to(&self, il: usize, row_seq: &[i32], dst: *mut u8) -> Result<*mut u8, String> {
        let tbl: Vec<*mut u8> = row_seq.iter().map(|&sq| self.kv_v[il][sq as usize]).collect();
        let raw: Vec<usize> = tbl.iter().map(|&p| p as usize).collect();
        self.ctx.h2d(dst, bytemuck::cast_slice(&raw))?;
        Ok(dst)
    }

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
        if std::env::var_os("LLM170_NO_MMQ").is_none() || only.is_some() {
            if (only.is_none() || only.map_or(false, |m| m & (1u32 << (ty - 12)) != 0)) && matches!(ty, 12 | 13 | 14 | 23) && (ty != 14 || std::env::var_os("LLM170_NO_Q6MMQ").is_none()) && (t >= 32 || (t == 1 && std::env::var_os("LLM170_Q1MMQ").is_some()))
                && super::co_loaded(super::CO_MMQ | super::CO_MMQ2 | super::CO_MMQ3) {
                        return self.ctx.gemm_mmq(ty, y_f32 as *const u8, wp as *const u8, n_in, n_out, t, out);
            }
            // q6_K: dequant→f16 v4 타일 (llama dequant+MFMA 경로 대응, 부록42)
            if ty == 14 && t >= 32 && super::co_loaded(super::CO_MMQ2)
                && std::env::var_os("LLM170_DEQ16").is_some() {
                return self.ctx.gemm_f16_q6(y_f32 as *const u8, wp as *const u8, n_in, n_out, t, out);
            }
        }
        self.mm_b(xq, xq_w, wp, ty, n_in, n_out, out, t)
    }

    /// plans/28 디버그: 버퍼의 행별 L1 노름 덤프 (지연 게이트) — 수치 오염 행 탐지.
    fn trace_rows(&self, label: &str, ptr: *const u8, row_f32: usize, t: usize) -> Result<(), String> {
        if std::env::var_os("LLM170_MS_TRACE").is_none() {
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
            && std::env::var_os("LLM170_NO_G4").is_none()
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
        if ty == 13 && t == 1 && std::env::var_os("LLM170_Q5V2").is_some() {
            return self.ctx.gemv_q8_out_v2(xq as *const u8, wp as *const u8, ty, n_in, n_out, out, xq_w, t);
        }
        // 홀수 타입 타일 (plans/04): odd CO + t>=32에서만
        let odd_v4 = std::env::var_os("LLM170_EXACT").is_none()
            && super::co_loaded(super::CO_ODD) && t >= 32
            && matches!(ty, 20 | 11 | 21);
        // q8_0 타일 (j128): 소형 GEMV 토큰당 재독 제거
        let q8t = ty == 8 && t > 64 && (n_out >= 128 || t >= 256) && std::env::var_os("LLM170_EXACT").is_none()
            && super::co_loaded(super::CO_J128);
        if (matches!(ty, 12 | 13 | 14 | 23) && t > 1 || odd_v4 || q8t) && std::env::var_os("LLM170_NO_TILE").is_none() {
            // 타일 경로 — 가중 1회 독서 (블록=1행, TT 토큰 레지스터)
            return self.ctx.gemm_tile(xq as *const u8, wp as *const u8, self.ktab2 as *const u8, ty, n_in, n_out, xq_w, t, out);
        }
        self.ctx.gemv_q8_out(xq as *const u8, wp as *const u8, self.ktab2 as *const u8, ty, n_in, n_out, out, xq_w, t)
    }
    /// mm_b 사이드 스트림판 — 타일형만 (비타일은 주 스트림 사용)
    #[allow(clippy::too_many_arguments)]
    /// mm_b_s의 MMQ판 — side stream에서 quant+mul_mat_q (부록48).
    fn mm_b2_s(&self, y_f32: *mut u8, xq: *mut u8, xq_w: usize, wp: *mut u8, ty: u32, n_in: usize, n_out: usize, out: *mut u8, t: usize) -> Result<(), String> {
        if matches!(ty, 12 | 13 | 23) && t >= 32 && std::env::var_os("LLM170_NO_MMQ").is_none() && std::env::var_os("LLM170_NO_MMQ_S").is_none()
            && super::co_loaded(super::CO_MMQ | super::CO_MMQ2 | super::CO_MMQ3) {
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
pub fn inject(eng: &mut llm170_core::model::Engine) -> Result<(), String> {
    let t0 = std::time::Instant::now();
    let hp = eng.model.hp.clone();
    let (wnames, cnames): (Vec<String>, Vec<String>) =
        llm170_core::model::rawinject::raw_names(eng);
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
    let consts = llm170_core::model::rawinject::raw_consts(eng, &cnames);
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
