//! 원시 HIP 디코드 실행기 — 1토큰 스텝을 원시 런치열로 구성 (2026-09-03).
//! frame35의 op 순서를 그대로 옮기되 cubecl 프레임(op당 블로킹 제출)을
//! 대체: 영속 버퍼 + 비동기 런치 + 마지막 1회 동기. 수치는 커널 검증
//! 게이트(rawhip-check·미러)를 통과한 산술과 동일.

use super::RawCtx;
use llm170_core::matmul::Weight;

/// 디코드 상주 상태 — 스텝마다 재사용, 해제 없음.
pub struct DecodeState {
    pub ctx: RawCtx,
    // 활성/중간 버퍼 (f32 바이트)
    pub xs: *mut u8,      // 잔차 스트림 [n_embd]
    pub xn: *mut u8,      // norm 출력 [n_embd]
    pub gqkv: *mut u8,    // conv 출력 [conv_ch]
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
    pub xq_g: *mut u8,    // (6144/4 + 6144/32)*4
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
    // 상수 (norm 가중치·conv·cs 테이블·마스크)
    pub consts: std::collections::HashMap<String, *mut u8>,
    // 가중치 (dev 상주 — 업로드 1회)
    pub weights: std::collections::HashMap<String, (*mut u8, u32, usize, usize)>, // (ptr, ty, n_in, n_out)
    pub ktab2: *mut u8,
    // KV/GDN 상태 [seq][...]
    pub kv_k: Vec<*mut u8>,
    pub kv_v: Vec<*mut u8>,
    pub st_conv: Vec<*mut u8>,
    pub st_gdn: Vec<*mut u8>,
    // 하이퍼파라미터
    pub n_embd: usize,
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
        let mut kv_k = Vec::with_capacity(n_seqs);
        let mut kv_v = Vec::with_capacity(n_seqs);
        let mut st_conv = Vec::with_capacity(n_seqs);
        let mut st_gdn = Vec::with_capacity(n_seqs);
        for _ in 0..n_seqs {
            kv_k.push(ctx.alloc(kv_len * 4).map_err(|e| e.to_string())?);
            kv_v.push(ctx.alloc(kv_len * 4).map_err(|e| e.to_string())?);
            st_conv.push(ctx.alloc(conv_len * 4).map_err(|e| e.to_string())?);
            st_gdn.push(ctx.alloc(gdn_len * 4).map_err(|e| e.to_string())?);
        }
        let zero_k = vec![0f32; kv_len];
        let zero_conv = vec![0f32; conv_len];
        let zero_gdn = vec![0f32; gdn_len];
        for s in 0..n_seqs {
            ctx.h2d(kv_k[s], bytemuck::cast_slice(&zero_k))?;
            ctx.h2d(kv_v[s], bytemuck::cast_slice(&zero_k))?;
            ctx.h2d(st_conv[s], bytemuck::cast_slice(&zero_conv))?;
            ctx.h2d(st_gdn[s], bytemuck::cast_slice(&zero_gdn))?;
        }
        let max_rows = (n / 32).max(n_ff.max(g6)).max(hp.n_head + hp.n_kv).max(1);
        let bs = |nb: usize| Self::a(&ctx, nb).unwrap();
        let (b_xs, b_xn, b_gqkv) = (bs(n * 4), bs(n * 4), bs(conv_ch * 4));
        let (b_gz, b_gb, b_ga, b_gbg) = (bs(d_inner * 4), bs(hp.dt_rank * 4), bs(hp.dt_rank * 4), bs(hp.dt_rank * 2 * 4));
        let (b_gq, b_gk, b_gv, b_go) = (bs(k_len * 4), bs(k_len * 4), bs(v_len * 4), bs(v_len * 4));
        let (b_ggated, b_gout) = (bs(d_inner * 4), bs(n * 4));
        let (b_fgate, b_fup, b_fglu, b_fdown) = (bs(n_ff * 4), bs(n_ff * 4), bs(n_ff * 4), bs(n * 4));
        let b_logits = bs(hp.vocab * 4);
        let (b_xqn, b_xqf, b_xqg) = (bs((n / 4 + n / 32) * 4), bs((n_ff / 4 + n_ff / 32) * 4), bs((g6 / 4 + g6 / 32) * 4));
        let (b_aq, b_ak, b_av) = (bs(hp.n_head * 2 * hp.head_dim * 4), bs(hp.n_kv * hp.head_dim * 4), bs(hp.n_kv * hp.head_dim * 4));
        let (b_aout, b_scores, b_p64) = (bs(hp.n_head * hp.head_dim * 4), bs(hp.n_head * ctx_len * 4), bs(max_rows * 32 * 8));
        let ds = DecodeState {
            ctx,
            xs: b_xs, xn: b_xn, gqkv: b_gqkv, gz: b_gz, gb: b_gb,
            ga: b_ga, gbg: b_gbg, gq: b_gq, gk: b_gk,
            gv: b_gv, go: b_go, ggated: b_ggated, gout: b_gout,
            fgate: b_fgate, fup: b_fup, fglu: b_fglu, fdown: b_fdown,
            logits: b_logits, xq_n: b_xqn, xq_f: b_xqf, xq_g: b_xqg,
            aq: b_aq, ak: b_ak, av: b_av, aout: b_aout,
            scores: b_scores, p64: b_p64,
            one, consts: c, weights: wmap, ktab2: kt,
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
    /// ew 헬퍼 — 그리드 계산 포함.
    fn ew(&self, name: &str, n: usize, args: &mut [*mut std::ffi::c_void]) -> Result<(), String> {
        self.ctx.launch(name, n.div_ceil(64) as u32, 1, 64, args)
    }
    fn p<T>(v: &mut T) -> *mut std::ffi::c_void {
        v as *mut T as *mut std::ffi::c_void
    }
    /// rms 2-커널 (x, w, out, n, w_reps) — p64 재사용.
    fn rms(&self, x: *mut u8, w: *mut u8, out: *mut u8, n: usize, w_reps: usize) -> Result<(), String> {
        let rows = 1usize;
        let mut xp = x as *mut std::ffi::c_void;
        let mut pp = self.p64 as *mut std::ffi::c_void;
        let mut na = n as i32;
        let mut a1 = vec![Self::p(&mut xp), Self::p(&mut pp), Self::p(&mut na)];
        self.ctx.launch("rms_part", rows as u32, 1, 32, &mut a1)?;
        let mut wp = w as *mut std::ffi::c_void;
        let mut op = out as *mut std::ffi::c_void;
        let mut ep = self.eps;
        let mut wr = w_reps as i32;
        let mut a2 = vec![
            Self::p(&mut xp), Self::p(&mut wp), Self::p(&mut pp),
            Self::p(&mut op), Self::p(&mut ep), Self::p(&mut na), Self::p(&mut wr),
        ];
        self.ctx.launch("rms_finish", rows as u32, 1, 32, &mut a2)
    }
    /// W4A8 GEMV (quant 완료된 xq 사용) — part는 gemv_q8 내부 alloc 대신
    /// 재사용 불가(소유) — 그대로 gemv_q8 사용.
    fn mm(&self, xq: *mut u8, wkey: &str, out: *mut u8, n_out: usize) -> Result<(), String> {
        let (wp, ty, n_in, _) = *self.weights.get(wkey).ok_or_else(|| format!("weight 없음: {wkey}"))?;
        let g = self.ctx.gemv_q8(xq as *const u8, wp as *const u8, self.ktab2 as *const u8, ty, n_in, n_out)?;
        // out이 임시가 아닌 상주 버퍼에 필요 — gemv_q8는 자체 out 사용.
        // 상주 out에 복사하는 커널 호출 대신: gemv 결과를 h2d? 비효율.
        // → gemv_q8 직접 상주 out에 쓰도록 변형 필요. 임시: d2h+h2d.
        self.ctx.h2d(out, bytemuck::cast_slice(&g))?;
        Ok(())
    }
    /// 활성 양자화 (x → xq 통합버퍼).
    fn quant(&self, x: *mut u8, xq: *mut u8, n: usize) -> Result<(), String> {
        self.ctx.quant_q8(x as *const u8, xq, n)
    }
    fn axpy(&self, y: *mut u8, x: *mut u8, n: usize) -> Result<(), String> {
        let mut yp = y as *mut std::ffi::c_void;
        let mut xp = x as *mut std::ffi::c_void;
        let mut op = self.one as *mut std::ffi::c_void;
        let mut na = n as i32;
        let mut args = vec![Self::p(&mut yp), Self::p(&mut xp), Self::p(&mut op), Self::p(&mut na)];
        self.ew("axpy_scaled", n, &mut args)
    }
    fn copy(&self, src: *mut u8, dst: *mut u8, src_off: usize, dst_off: usize, n: usize) -> Result<(), String> {
        let mut sp = src as *mut std::ffi::c_void;
        let mut dp = dst as *mut std::ffi::c_void;
        let mut so = src_off as i32;
        let mut doff = dst_off as i32;
        let mut na = n as i32;
        let mut args = vec![Self::p(&mut sp), Self::p(&mut dp), Self::p(&mut so), Self::p(&mut doff), Self::p(&mut na)];
        self.ew("copy_rows", n, &mut args)
    }
}

impl DecodeState {
    /// 디코드 1스텝 (t=1, 단일 시퀀스) — logits 반환.
    /// LLM170_RAWHIP=1 게이트로 엔진에서 호출 (배선 다음 단계).
    pub fn step(&self, seq: usize, token: u32, pos: usize) -> Result<Vec<f32>, String> {
        let n = self.n_embd;
        // 0) 임베딩 — 호출부가 xs에 h2d 완료 전제 (호스트 dequant).
        // 1) pre-norm
        let wn = self.consts.get(&format!("blk.{}.attn_norm", 0)).copied().unwrap_or(self.xn);
        let _ = wn;
        // 각 층: is_recr 분기 — 전체 루프는 엔진 배선 시 완성.
        // 현재까지: 커널 전종 + 상태 초기화 + 헬퍼. 다음 커밋에서 층 루프.
        let _ = (seq, token, pos);
        Err("decode step: layer loop pending — next commit".into())
    }
}
