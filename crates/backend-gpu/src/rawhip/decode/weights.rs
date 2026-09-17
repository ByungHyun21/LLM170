//! decode 무게 상주화 — 업로드·레이아웃 준비 (decode/mod.rs에서 이동, plans/78 R2).

use super::*;
use crate::rawhip::env_on;

impl DecodeState {
    fn a(ctx: &RawCtx, bytes: usize) -> Result<*mut u8, String> {
        ctx.alloc(bytes).map_err(|e| e.to_string())
    }

    /// 모델에서 상주 상태 구축 — 가중치 업로드 1회.
    pub fn new(
        ctx: RawCtx,
        hp: &llm170_core::qwen35::hparams::Hparams,
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
        let mut kv_k16 = Vec::with_capacity(n_full);
        let mut kv_v16 = Vec::with_capacity(n_full);
        for _ in 0..n_full {
            let mut ck = Vec::with_capacity(n_seqs);
            let mut cv2 = Vec::with_capacity(n_seqs);
            let mut ck16 = Vec::with_capacity(n_seqs);
            let mut cv16 = Vec::with_capacity(n_seqs);
            for s in 0..n_seqs {
                // f32 원본은 legacy 폴백에서만 필요 — 기본은 f16 미러만(메모리 3배 절감).
                let lf = legacy_f32();
                ck.push(if lf { ctx.alloc(kv_len * 4).map_err(|e| e.to_string())? } else { std::ptr::null_mut() });
                cv2.push(if lf { ctx.alloc(kv_len * 4).map_err(|e| e.to_string())? } else { std::ptr::null_mut() });
                // f16 미러: 초기값은 변환 전까지 읽히지 않는다(항상 [0,n_past) 를 변환)
                ck16.push(ctx.alloc(kv_len * 2).map_err(|e| e.to_string())?);
                cv16.push(ctx.alloc(kv_len * 2).map_err(|e| e.to_string())?);
                if lf {
                    ctx.h2d(ck[s], bytemuck::cast_slice(&zero_k))?;
                    ctx.h2d(cv2[s], bytemuck::cast_slice(&zero_k))?;
                }
            }
            kv_k.push(ck);
            kv_v.push(cv2);
            kv_k16.push(ck16);
            kv_v16.push(cv16);
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
        // np/spec 배치 scores 버퍼 — 행 수 상한은 **배치**(n_seqs×(k+1) ≤ 40,
        // 아래 b_logits_all 주석)이지 프리필 청크가 아니다. 종전엔 t_max(=
        // LLM170_CHUNK, 기본 512)로 잡아 ctx에 곱해져 ctx 32768에서 4.3GB
        // 단일 할당이 됐고 서버 init이 사실상 멈췄다(2026-09-17 실측:
        // ctx 8192 준비 13.3s → ctx 32768 209s에도 미완). 64행 상한은 코드
        // 자체의 문서화된 한계와 같다.
        let b_scores_t = bs(t_max.min(64) * hp.n_head * ctx_len * 4);
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
        let mut v_mtp_k16: Vec<*mut u8> = Vec::new();
        let mut v_mtp_v16: Vec<*mut u8> = Vec::new();
        let (mut v_mtp_k, mut v_mtp_v, b_mtp_cat, b_mtp_cur, b_mtp_qkv, b_mtp_ao, b_mtp_e, b_mtp_h, b_mtp_xq, b_mtp_xq2) = if mtp_on {
            let mut vk = Vec::with_capacity(n_seqs);
            let mut vv = Vec::with_capacity(n_seqs);
            for _ in 0..n_seqs {
                vk.push(ctx.alloc(kv_len * 4).map_err(|e| e.to_string())?);
                vv.push(ctx.alloc(kv_len * 4).map_err(|e| e.to_string())?);
                // MTP 어텐션도 f16 미러를 읽는다(qsa_flash 가 f16).
                v_mtp_k16.push(ctx.alloc(kv_len * 2).map_err(|e| e.to_string())?);
                v_mtp_v16.push(ctx.alloc(kv_len * 2).map_err(|e| e.to_string())?);
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
            let zh = vec![0u8; kv_len * 2];
            for &p in v_mtp_k16.iter() { ctx.h2d(p, &zh)?; }
            for &p in v_mtp_v16.iter() { ctx.h2d(p, &zh)?; }
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
            one, consts: c, weights: wmap, ktab2: kt, n_vocab: hp.vocab, n_vocab_set: true,
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
            mtp_kv_k16: v_mtp_k16,
            mtp_kv_v16: v_mtp_v16,
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
            kv_k, kv_v, kv_k16, kv_v16, st_conv, st_gdn,
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
    pub(super) fn mm_into(&self, xq: *mut u8, wp: *mut u8, ty: u32, n_in: usize, n_out: usize, out: *mut u8) -> Result<(), String> {
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
    pub(super) fn mm_direct(&self, xq: *mut u8, wp: *mut u8, ty: u32, n_in: usize, n_out: usize, out: *mut u8) -> Result<(), String> {
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

    pub(super) fn ew_l(&self, name: &'static str, n: usize, args: &mut [*mut std::ffi::c_void]) -> Result<(), String> {
        self.ctx.launch(name, n.div_ceil(64) as u32, 1, 64, args)
    }
    pub(super) fn p<T>(v: &mut T) -> *mut std::ffi::c_void {
        v as *mut T as *mut std::ffi::c_void
    }
    pub(super) fn rms(&self, x: *mut u8, w: *mut u8, out: *mut u8, n: usize) -> Result<(), String> {
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
    pub(super) fn quant(&self, x: *mut u8, xq: *mut u8, n: usize) -> Result<(), String> {
        self.ctx.quant_q8(x as *const u8, xq, n)
    }
    /// rms+quant 융합 (t=1, n%1024==0) — 3런치 1런치. 산술 미러 동일열.
    fn rms_quant(&self, x: *mut u8, w: *mut u8, xq: *mut u8, n: usize) -> Result<(), String> {
        if !n.is_multiple_of(1024) || env_on("LLM170_RMSQ_SPLIT") {
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
    pub(super) fn axpy(&self, y: *mut u8, x: *mut u8, n: usize) -> Result<(), String> {
        let mut yp = y as *mut std::ffi::c_void;
        let mut xp = x as *mut std::ffi::c_void;
        let mut op = self.one as *mut std::ffi::c_void;
        let mut na = n as i32;
        let mut args = vec![Self::p(&mut yp), Self::p(&mut xp), Self::p(&mut op), Self::p(&mut na)];
        self.ew_l("axpy_scaled", n, &mut args)
    }
    pub(super) fn copy(&self, src: *mut u8, dst: *mut u8, src_off: usize, dst_off: usize, n: usize) -> Result<(), String> {
        let mut sp = src as *mut std::ffi::c_void;
        let mut dp = dst as *mut std::ffi::c_void;
        let mut so = src_off as i32;
        let mut doff = dst_off as i32;
        let mut na = n as i32;
        let mut args = vec![Self::p(&mut sp), Self::p(&mut dp), Self::p(&mut so), Self::p(&mut doff), Self::p(&mut na)];
        self.ew_l("copy_rows", n, &mut args)
    }
    pub(super) fn w(&self, key: &str) -> Result<(*mut u8, u32, usize, usize), String> {
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
        if let Some(m) = only
            && m & (1u32 << (ty - 12)) == 0 {
                return false;
            }
        t >= 32 && self.ctx.co_loaded(super::CO_MMQ | super::CO_MMQ2 | super::CO_MMQ3)
    }

    /// 사이드 스트림(mm_b2_s → gemm_mmq_s)의 f32 직소비 여부 — 조건 미러.
    /// (gemm_mmq_s는 ty14를 다루지 않는다.)
    fn mmq_used_s(&self, ty: u32, t: usize) -> bool {
        matches!(ty, 12 | 13 | 23)
            && t >= 32
           
           
            && self.ctx.co_loaded(super::CO_MMQ | super::CO_MMQ2 | super::CO_MMQ3)
    }

    /// 지정 가중치들이 모두 f32 직소비 경로면 활성 quant를 생략할 수 있다.
    pub(super) fn grp_mmq(&self, names: &[String], t: usize) -> bool {
        let r = names.iter().all(|n| {
            self.weights
                .get(n)
                .is_some_and(|&(_, ty, _, _)| self.mmq_used(ty, t) && self.mmq_used_s(ty, t))
        });
        if env_on("LLM170_QSKIP_DBG") {
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
            if env_on("LLM170_RAWHIP_TRACE") {
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
                } else if env_on("LLM170_DECODE_PAIRS") {
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
                } else if env_on("LLM170_DECODE_PAIRS") {
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
                    self.ew_l("gdn_beta_g_f32", self.dt_rank, &mut args)?;
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
                if env_on("LLM170_RAWHIP_TRACE") && il == 0 {
                    self.ctx.sync()?;
                    let mut ho = vec![0f32; v_len];
                    self.ctx.d2h(bytemuck::cast_slice_mut(&mut ho).as_mut(), self.go)?;
                    let sumo: f64 = ho.iter().map(|&v| v as f64).sum();
                    let mut hq = vec![0f32; k_len];
                    self.ctx.d2h(bytemuck::cast_slice_mut(&mut hq).as_mut(), self.gq)?;
                    let _sumq: f64 = hq.iter().map(|&v| v as f64).sum();
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
                if self.d_state.is_multiple_of(32) && self.d_inner.is_multiple_of(1024) {
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
                    self.ctx.launch("norm_gated_silu_f32", self.dt_rank as u32, 1, 32, &mut args)?;
                    self.quant(self.ggated, self.xq_g, self.d_inner)?;
                }
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.ssm_out.weight"))?;
                self.mm_into(self.xq_g, wp, ty, ni, no, self.gout)?;
                if env_on("LLM170_RAWHIP_TRACE") && il == 0 {
                    self.ctx.sync()?;
                    let mut ho = vec![0f32; n];
                    self.ctx.d2h(bytemuck::cast_slice_mut(&mut ho).as_mut(), self.gout)?;
                    let sumo: f64 = ho.iter().map(|&v| v as f64).sum();
                    eprintln!("#  G0dbg gout sum={sumo:.6} gout[0..4]={:?}", &ho[0..4]);
                }
                recr_idx += 1;
            } else {
                if env_on("LLM170_RAWHIP_TRACE") && il == 3 {
                    self.ctx.sync()?;
                    let mut hn = vec![0f32; n];
                    self.ctx.d2h(bytemuck::cast_slice_mut(&mut hn).as_mut(), self.xn)?;
                    eprintln!("#  A3dbg xn[0..6]={:?}", &hn[0..6]);
                    // 결정적 A/B: 이 xn으로 호스트 미러 av[0] 계산
                    if il == 3 && env_on("LLM170_RAWHIP_HOSTAB") {
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
                if env_on("LLM170_RAWHIP_TRACE") { self.ctx.sync()?; eprintln!("#  aq ok"); }
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_k.weight"))?;
                self.mm_into(self.xq_n, wp, ty, ni, no, self.ak)?;
                if env_on("LLM170_RAWHIP_TRACE") { self.ctx.sync()?; eprintln!("#  ak ok"); }
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_v.weight"))?;
                self.mm_into(self.xq_n, wp, ty, ni, no, self.av)?;
                if env_on("LLM170_RAWHIP_TRACE") { self.ctx.sync()?; eprintln!("#  av ok"); }
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
                    if env_on("LLM170_RAWHIP_TRACE") { self.ctx.sync()?; eprintln!("#  qk_norm ok"); }
                }
                // KV append
                if env_on("LLM170_RAWHIP_TRACE") && il == 3 {
                    self.ctx.sync()?;
                    let mut hk = vec![0f32; n_kv * hd];
                    self.ctx.d2h(bytemuck::cast_slice_mut(&mut hk).as_mut(), self.ak)?;
                    eprintln!("#  A3dbg pos{pos} ak[0..4]={:?}", &hk[0..4]);
                    let mut hck = vec![0f32; (pos + 1) * n_kv * hd];
                    if !legacy_f32() { return Err("dump env needs the f32 KV (set LLM170_NO_GQA2D)".into()); }
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
                if legacy_f32() {
                    self.copy(self.ak, self.kv_k[full_idx][seq], 0, pos * n_kv * hd, n_kv * hd)?;
                    self.copy(self.av, self.kv_v[full_idx][seq], 0, pos * n_kv * hd, n_kv * hd)?;
                }
                let dpos = pos * n_kv * hd;
                kv_to_f16(&self.ctx, self.ak, self.kv_k16[full_idx][seq], 0, dpos, n_kv * hd)?;
                kv_to_f16(&self.ctx, self.av, self.kv_v16[full_idx][seq], 0, dpos, n_kv * hd)?;
                // score
                let mask = *self.consts.get("mask").ok_or("mask")?;
                let flash1 = hd <= 256;
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
                    if env_on("LLM170_RAWHIP_TRACE") { self.ctx.sync()?; eprintln!("#  score ok"); }
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
                    if env_on("LLM170_RAWHIP_TRACE") { self.ctx.sync()?; eprintln!("#  mix ok"); }
                }
                }
                // t=1 fused flash (score/mix2 대체)
                if hd <= 256 {
                    let mut qp = self.aq as *mut std::ffi::c_void;
                    let mut ckp = self.kv_k16[full_idx][seq] as *mut std::ffi::c_void;
                    let mut cvp = self.kv_v16[full_idx][seq] as *mut std::ffi::c_void;
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
                    let gqa_ok = hd <= 256 && n_head % n_kv == 0;
                    if gqa_ok || np_ > (std::env::var("LLM170_T1SEG").ok().and_then(|v| v.parse::<i32>().ok()).unwrap_or(512)) {
                        // 분할 flash — 헤드당 1블록(48블록)은 대역폭 저활용,
                        // 세그먼트 병렬화 (t=1도 nq 가드로 안전, 2026-09-05)
                        // sg = 32 고정은 단문맥 최적(플랫폼 지연 은닉)이지만 장문맥에서
                        // nseg=512@16k 가 되어 merge 가 헤드당 500+ 부분합을 직렬 합산한다
                        // (KTRACE 16k: gqa2d 5.6 + merge 4.2ms/step). plans/73: 문맥에
                        // 비례해 키우면 nseg ≤ 64 로 유지 — 블록 수(4kvh×nseg)는
                        // 256+ 로 충분히 병렬. 단문맥(sg=32 구간)은 수치 순서 불변.
                        let sg = std::env::var("LLM170_T1SG").ok().and_then(|v| v.parse().ok())
                            .unwrap_or_else(|| ((pos + 1) / 64).clamp(32, 256));
                        let nseg = (pos + 1).div_ceil(sg);
                        let part = self.ctx.scratch(n_head * nseg * (hd + 2) * 4)?;
                        let mut pp2 = part as *mut std::ffi::c_void;
                        let mut sg_a = sg as i32;
                        let mut args = vec![Self::p(&mut qp), Self::p(&mut ckp), Self::p(&mut cvp), Self::p(&mut mp), Self::p(&mut pp2), Self::p(&mut np_), Self::p(&mut nh), Self::p(&mut nk), Self::p(&mut h), Self::p(&mut tl), Self::p(&mut ss), Self::p(&mut p0), Self::p(&mut sg_a)];
                        // GQA 공유(t=1): kv-head당 WG 하나가 q-head 전부를 처리 —
                        // K/V 트래픽 1/(q-heads per kv-head). LLM170_NO_GQA=1이면 종전.
                        // 실측 교차점: ctx<768은 종전(더 많은 WG), 그 이상은 GQA 공유가
                        // 이김 (pp512 −1.2%, 1024 +1.6%, 2048 +4.7%, 3072 +8.8%).
                        let gqa = gqa_ok;
                        // v2 기본(2026-09-12): 워프가 키 4개를 전담해 감축을 워프 안에서 끝낸다.
                        // gqa-bench 실측 3314키 292.9 -> 171.5us (1.71x), 최대상대차 5.1e-7.
                        // LLM170_NO_GQA2=1 이면 종전 커널로 복귀.
                        let gqa2 = !env_on("LLM170_NO_GQA2");
                        // v_dot2(f16 KV) 경로가 기본: QK 에 셔플이 없다 (gqa-bench 3314 175.9→106.9us)
                        let gqa2d = gqa2;
                        if gqa2d {
                            // f16 미러 + v_dot2: 인자 순서는 gqa2 와 같고 K/V 만 half 버퍼
                            let mut k16 = self.kv_k16[full_idx][seq] as *mut std::ffi::c_void;
                            let mut v16 = self.kv_v16[full_idx][seq] as *mut std::ffi::c_void;
                            let mut args16: Vec<*mut std::ffi::c_void> = vec![
                                &mut qp as *mut _ as *mut std::ffi::c_void,
                                &mut k16 as *mut _ as *mut std::ffi::c_void,
                                &mut v16 as *mut _ as *mut std::ffi::c_void,
                                &mut mp as *mut _ as *mut std::ffi::c_void,
                                &mut pp2 as *mut _ as *mut std::ffi::c_void,
                                &mut np_ as *mut _ as *mut std::ffi::c_void,
                                &mut nh as *mut _ as *mut std::ffi::c_void,
                                &mut nk as *mut _ as *mut std::ffi::c_void,
                                &mut h as *mut _ as *mut std::ffi::c_void,
                                &mut tl as *mut _ as *mut std::ffi::c_void,
                                &mut ss as *mut _ as *mut std::ffi::c_void,
                                &mut p0 as *mut _ as *mut std::ffi::c_void,
                                &mut sg_a as *mut _ as *mut std::ffi::c_void,
                            ];
                            self.ctx.launch3("qsa_flash_gqa2d", 1, n_kv as u32, nseg as u32, 256, &mut args16)?;
                        } else if gqa && gqa2 {
                            self.ctx.launch3("qsa_flash_gqa2", 1, n_kv as u32, nseg as u32, 256, &mut args)?;
                        } else if gqa {
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
                if env_on("LLM170_RAWHIP_TRACE") && il == 3 {
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
            if env_on("LLM170_RAWHIP_TRACE") {
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
            if env_on("LLM170_DECODE_PAIRS") {
                self.ctx.side_wait_main()?;
                self.mm_into(self.xq_n, wg, tg, nig, nog, self.fgate)?;
                self.mm_into_s(self.xq_n, wu, tu, niu, nou, self.fup)?;
                self.ctx.join2()?;
            } else {
                self.mm_into(self.xq_n, wg, tg, nig, nog, self.fgate)?;
                self.mm_into(self.xq_n, wu, tu, niu, nou, self.fup)?;
            }
            // silu_mul+quant 융합 (t=1, n_ff%2048==0) — 동일 산술열
            if self.n_ff.is_multiple_of(2048) {
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
            if env_on("LLM170_DOWN_PROF") { self.ctx.sync()?; }
            let __td = std::time::Instant::now();
            self.mm_into(self.xq_f, wd, td, nid, nod, self.fdown)?;
            if env_on("LLM170_DOWN_PROF") {
                self.ctx.sync()?;
                eprintln!("# down ty={td} no={nod} {:.3}ms", __td.elapsed().as_secs_f64() * 1e3);
            }
            self.axpy(self.xs, self.fdown, n)?;
            if env_on("LLM170_RAWHIP_TRACE") {
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
        if env_on("LLM170_RAWHIP_TIMING") {
            eprintln!("step gpu={:.2}ms", t0.elapsed().as_secs_f64() * 1e3);
        }
        Ok(Vec::new()) // logits 상주
    }
}
