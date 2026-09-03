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
        let ktab2: Vec<u32> = (0..256u32)
            .map(|b| {
                let lo = llm170_core::KVALUES_IQ4NL[(b & 0xF) as usize] as u8 as u32;
                let hi = llm170_core::KVALUES_IQ4NL[(b >> 4) as usize] as u8 as u32;
                lo | (hi << 8)
            })
            .collect();
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
        // 배치 아레나 (t_max=64)
        let t_max = 128usize;
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
        self.ctx.launch("rms_finish", 1, 1, 32, &mut a2)
    }
    fn quant(&self, x: *mut u8, xq: *mut u8, n: usize) -> Result<(), String> {
        self.ctx.quant_q8(x as *const u8, xq, n)
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

    /// 디코드 1스텝 (t=1, 단일 시퀀스) — logits 반환. xs에 임베딩 h2d 완료 전제.
    #[allow(clippy::too_many_lines)]
    pub fn step(&self, seq: usize, pos: usize) -> Result<Vec<f32>, String> {
        let t0 = std::time::Instant::now();
        let _ = &t0;
        self.ctx.scratch_rewind();
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
            self.rms(self.xs, wn, self.xn, n)?;
            self.quant(self.xn, self.xq_n, n)?;
            if self.is_recr[il] {
                // in_proj 4종
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_qkv.weight"))?;
                self.mm_into(self.xq_n, wp, ty, ni, no, self.gqkv)?;
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_gate.weight"))?;
                self.mm_into(self.xq_n, wp, ty, ni, no, self.gz)?;
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.ssm_beta.weight"))?;
                self.mm_into(self.xq_n, wp, ty, ni, no, self.gb)?;
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.ssm_alpha.weight"))?;
                self.mm_into(self.xq_n, wp, ty, ni, no, self.ga)?;
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
                    self.ctx.launch("gdn_conv", conv_ch as u32, 1, 32, &mut args)?;
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
                    self.ew_l("gdn_beta_g", self.dt_rank, &mut args)?;
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
                    self.ctx.launch("gdn_ar", n_pairs as u32, 1, 128, &mut args)?;
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
                // norm_gated silu (rows = dt_rank, w = ssm_norm 반복)
                let snorm = *self.consts.get(&format!("blk.{il}.ssm_norm")).ok_or("ssm_norm")?;
                {
                    let mut op = self.go as *mut std::ffi::c_void;
                    let mut zp = self.gz as *mut std::ffi::c_void;
                    let mut wp = snorm as *mut std::ffi::c_void;
                    let mut outp = self.ggated as *mut std::ffi::c_void;
                    let mut ep = self.eps;
                    let mut d = self.d_state as i32;
                    let mut nh = self.dt_rank as i32;
                    let mut args = vec![Self::p(&mut op), Self::p(&mut zp), Self::p(&mut wp), Self::p(&mut outp), Self::p(&mut ep), Self::p(&mut d), Self::p(&mut nh)];
                    self.ctx.launch("norm_gated_silu", self.dt_rank as u32, 1, 32, &mut args)?;
                }
                // out proj (ggated → gout) — n_in = d_inner
                self.quant(self.ggated, self.xq_g, self.d_inner)?;
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
            self.rms(self.xs, pw, self.xn, n)?;
            self.quant(self.xn, self.xq_n, n)?;
            let (wg, tg, nig, nog) = self.w(&format!("blk.{il}.ffn_gate.weight"))?;
            self.mm_into(self.xq_n, wg, tg, nig, nog, self.fgate)?;
            let (wu, tu, niu, nou) = self.w(&format!("blk.{il}.ffn_up.weight"))?;
            self.mm_into(self.xq_n, wu, tu, niu, nou, self.fup)?;
            // silu_mul
            {
                let mut gp = self.fgate as *mut std::ffi::c_void;
                let mut up = self.fup as *mut std::ffi::c_void;
                let mut op = self.fglu as *mut std::ffi::c_void;
                let mut na = self.n_ff as i32;
                let mut args = vec![Self::p(&mut gp), Self::p(&mut up), Self::p(&mut op), Self::p(&mut na)];
                self.ew_l("silu_mul", self.n_ff, &mut args)?;
            }
            self.quant(self.fglu, self.xq_f, self.n_ff)?;
            let (wd, td, nid, nod) = self.w(&format!("blk.{il}.ffn_down.weight"))?;
            self.mm_into(self.xq_f, wd, td, nid, nod, self.fdown)?;
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
        let guard = self.st.lock().map_err(|e| e.to_string())?;
        let ds = guard.as_ref().ok_or("raw_decode: 미초기화")?;
        ds.step_batch(seq, pos0, emb)?;
        let r = ds.read_logits();
        if let (Some(path), Ok(v)) = (std::env::var_os("LLM170_DUMP_LOGITS"), r.as_ref()) {
            let _ = std::fs::write(&path, bytemuck::cast_slice(v));
        }
        if std::env::var_os("LLM170_RAWHIP_TIMING").is_some() {
            eprintln!("batch({} tok) wall={:.1}ms", emb.len() / ds.n_embd, t0.elapsed().as_secs_f64() * 1e3);
        }
        r
    }

    fn raw_step(&self, seq: usize, pos: usize, emb: &[f32]) -> Result<Vec<f32>, String> {
        let t0 = std::time::Instant::now();
        let guard = self.st.lock().map_err(|e| e.to_string())?;
        let ds = guard.as_ref().ok_or("raw_decode: 미초기화")?;
        ds.ctx.h2d(ds.xs, bytemuck::cast_slice(emb))?;
        ds.step(seq, pos)?;
        let r = ds.read_logits();
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
            self.ctx.quant_q8_b(self.xn_t, self.xq_n_t, n, xq_sn, t)?;
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
                    self.ctx.side_wait_main()?;
                    self.mm_b_s(self.xq_n_t, xq_sn, wg2, tg2, nig2, nog2, self.gz_t, t)?;
                    self.mm_b(self.xq_n_t, xq_sn, wp, ty, ni, no, self.gqkv_t, t)?;
                    self.mm_b(self.xq_n_t, xq_sn, wb2, tb2, nib2, nob2, self.gb_t, t)?;
                    self.mm_b(self.xq_n_t, xq_sn, wa2, ta2, nia2, noa2, self.ga_t, t)?;
                    self.ctx.join2()?;
                } else {
                    self.mm_b(self.xq_n_t, xq_sn, wp, ty, ni, no, self.gqkv_t, t)?;
                    self.mm_b(self.xq_n_t, xq_sn, wg2, tg2, nig2, nog2, self.gz_t, t)?;
                    self.mm_b(self.xq_n_t, xq_sn, wb2, tb2, nib2, nob2, self.gb_t, t)?;
                    self.mm_b(self.xq_n_t, xq_sn, wa2, ta2, nia2, noa2, self.ga_t, t)?;
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
                        self.ctx.launch3("gdn_conv_t2", conv_ch.div_ceil(64) as u32, t as u32, 1, 64, &mut args)?;
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
gmark("conv", &mut marks);
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
                    self.ew_l("gdn_beta_g", self.dt_rank * t, &mut args)?;
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
                    } else {
                        self.ctx.launch3("gdn_ar_w", self.dt_rank as u32, self.d_state as u32, 1, 32, &mut args)?;
                    }
                }
gmark("ar", &mut marks);
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
                    self.ctx.launch3("norm_gated_silu", self.dt_rank as u32, t as u32, 1, 32, &mut args)?;
                }
gmark("gdn", &mut marks);
gmark("normg", &mut marks);
                // out proj 배치
                self.ctx.quant_q8_b(self.ggated_t, self.xq_g_t, self.d_inner, xq_sg, t)?;
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.ssm_out.weight"))?;
gmark("outproj", &mut marks);
                self.mm_b(self.xq_g_t, xq_sg, wp, ty, ni, no, self.gout_t, t)?;
                recr_idx += 1;
} else {
gmark("attn", &mut marks);
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_q.weight"))?;
                self.mm_b(self.xq_n_t, xq_sn, wp, ty, ni, no, self.aq_t, t)?;
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_k.weight"))?;
                self.mm_b(self.xq_n_t, xq_sn, wp, ty, ni, no, self.ak_t, t)?;
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_v.weight"))?;
                self.mm_b(self.xq_n_t, xq_sn, wp, ty, ni, no, self.av_t, t)?;
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
                // wo 배치
                self.ctx.quant_q8_b(self.aout_t, self.xq_g_t, n_head * hd, xq_sg, t)?;
                let (wp, ty, ni, no) = self.w(&format!("blk.{il}.attn_output.weight"))?;
                self.mm_b(self.xq_g_t, xq_sg, wp, ty, ni, no, self.gout_t, t)?;
                full_idx += 1;
            }
self.axpy(self.xs_t, self.gout_t, n * t)?;
            gmark("proj", &mut marks);
            // FFN 배치
            let pw = *self.consts.get(&format!("blk.{il}.post_norm")).ok_or("post_norm")?;
            self.rms_rows(self.xs_t, pw, self.xn_t, n, t)?;
            self.ctx.quant_q8_b(self.xn_t, self.xq_n_t, n, xq_sn, t)?;
gmark("ffn_quant", &mut marks);
            // 2스트림: gate(사이드) ‖ up(주) — 독립 GEMM, 출력버퍼 분리
            let (wg, tg, nig, nog) = self.w(&format!("blk.{il}.ffn_gate.weight"))?;
            let (wu, tu, niu, nou) = self.w(&format!("blk.{il}.ffn_up.weight"))?;
            let gate_tile = matches!(tg, 12 | 13 | 14 | 23) && t > 64;
            let up_tile = matches!(tu, 12 | 13 | 14 | 23) && t > 64;
            if gate_tile && !up_tile {
                self.mm_b(self.xq_n_t, xq_sn, wu, tu, niu, nou, self.fup_t, t)?;
                self.ctx.side_wait_main()?;
                self.mm_b_s(self.xq_n_t, xq_sn, wg, tg, nig, nog, self.fgate_t, t)?;
                self.ctx.join2()?;
            } else if !gate_tile && up_tile {
                self.ctx.side_wait_main()?;
                self.mm_b_s(self.xq_n_t, xq_sn, wu, tu, niu, nou, self.fup_t, t)?;
                self.mm_b(self.xq_n_t, xq_sn, wg, tg, nig, nog, self.fgate_t, t)?;
                self.ctx.join2()?;
            } else if gate_tile && up_tile {
                // 둘 다 타일 — 하나 사이드
                self.ctx.side_wait_main()?;
                self.mm_b_s(self.xq_n_t, xq_sn, wu, tu, niu, nou, self.fup_t, t)?;
                self.mm_b(self.xq_n_t, xq_sn, wg, tg, nig, nog, self.fgate_t, t)?;
                self.ctx.join2()?;
            } else {
                self.mm_b(self.xq_n_t, xq_sn, wg, tg, nig, nog, self.fgate_t, t)?;
                self.mm_b(self.xq_n_t, xq_sn, wu, tu, niu, nou, self.fup_t, t)?;
            }
gmark("ffn_gate", &mut marks);
gmark("ffn_up", &mut marks);
            {
                let mut gp = self.fgate_t as *mut std::ffi::c_void;
                let mut up = self.fup_t as *mut std::ffi::c_void;
                let mut op = self.fglu_t as *mut std::ffi::c_void;
                let mut na = (self.n_ff * t) as i32;
                let mut args = vec![Self::p(&mut gp), Self::p(&mut up), Self::p(&mut op), Self::p(&mut na)];
                self.ew_l("silu_mul", self.n_ff * t, &mut args)?;
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
            self.ctx.quant_q8_b(self.fglu_t, self.xq_f_t, self.n_ff, xq_sf, t)?;
gmark("ffn_quant2", &mut marks);
            let (wd, td, nid, nod) = self.w(&format!("blk.{il}.ffn_down.weight"))?;
            self.mm_b(self.xq_f_t, xq_sf, wd, td, nid, nod, self.fdown_t, t)?;
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
            self.ctx.launch("rms_finish", 1, 1, 32, &mut a2)?;
        }
        self.ctx.quant_q8(self.xn, self.xq_n, n)?;
        let (wh, th, nih, noh) = self.w("output.weight")?;
        self.mm_into(self.xq_n, wh, th, nih, noh, self.logits)?;
        gmark("head", &mut marks);
        if prof {
            let wall = t0w.elapsed().as_secs_f64() * 1e3;
            let _ = wall;
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
        self.ctx.launch("rms_finish", t as u32, 1, 32, &mut a2)
    }

    /// 배치 GEMV — xq [t][xq_w], out [t][n_out].
    #[allow(clippy::too_many_arguments)]
    fn mm_b(&self, xq: *mut u8, xq_w: usize, wp: *mut u8, ty: u32, n_in: usize, n_out: usize, out: *mut u8, t: usize) -> Result<(), String> {
        // 홀수 타입 타일 (plans/04): CO3 + 타입별 env + t>=32에서만
        let odd_v4 = std::env::var_os("LLM170_EXACT").is_none()
            && std::env::var_os("LLM170_CO3_PATH").is_some() && t >= 32
            && ((ty == 20 && std::env::var_os("LLM170_NLV4").is_some())
                || (ty == 11 && std::env::var_os("LLM170_Q3KV4").is_some())
                || (ty == 21 && std::env::var_os("LLM170_IQ3SV4").is_some()));
        if (matches!(ty, 12 | 13 | 14 | 23) && t > 1 || odd_v4) && std::env::var_os("LLM170_NO_TILE").is_none() {
            // 타일 경로 — 가중 1회 독서 (블록=1행, TT 토큰 레지스터)
            return self.ctx.gemm_tile(xq as *const u8, wp as *const u8, self.ktab2 as *const u8, ty, n_in, n_out, xq_w, t, out);
        }
        self.ctx.gemv_q8_out(xq as *const u8, wp as *const u8, self.ktab2 as *const u8, ty, n_in, n_out, out, xq_w, t)
    }
    /// mm_b 사이드 스트림판 — 타일형만 (비타일은 주 스트림 사용)
    #[allow(clippy::too_many_arguments)]
    fn mm_b_s(&self, xq: *mut u8, xq_w: usize, wp: *mut u8, ty: u32, n_in: usize, n_out: usize, out: *mut u8, t: usize) -> Result<(), String> {
        self.ctx.gemm_tile_s(xq as *const u8, wp as *const u8, self.ktab2 as *const u8, ty, n_in, n_out, xq_w, t, out)
    }

}
