//! CLIP ViT GPU 실행기 (plans/17) — mmproj 가중치 f32 업로드 1회, 커널 파이프라인.
//! 산술은 core::clip (CPU) 미러 — 검증: 행별 최대오차.

use crate::rawhip::RawCtx;
use std::collections::HashMap;

pub struct Vit {
    ctx: std::sync::Arc<RawCtx>,
    n_embd: usize,
    n_head: usize,
    d_head: usize,
    n_ff: usize,
    n_blk: usize,
    eps: f32,
    patch: usize,
    /// 가중치 f32 행major — 이름 → (ptr, rows, ni)
    w: HashMap<String, (*mut u8, usize, usize)>,
    // 버퍼
    b_x: *mut u8,      // [tmax][n_embd]
    b_xn: *mut u8,     // [tmax][n_embd]
    b_qkv: *mut u8,    // [tmax][3·n_embd]
    b_attn: *mut u8,   // [tmax][n_embd]
    b_proj: *mut u8,   // [tmax][n_embd]
    b_mid: *mut u8,    // [tmax][n_ff]
    b_yx: *mut u8,     // [tmax][2] i32
    b_kvs: *mut u8,    // k/v 스테이징 [2][tmax][n_embd]
    t_max: usize,
}

// RawCtx는 Sync (내부 뮤텍스·스트림 단일) — 기존 DecodeState와 동일 가정.
unsafe impl Send for Vit {}
unsafe impl Sync for Vit {}

impl Vit {
    /// weights: (이름, f32 행major 데이터, rows, ni) — 호출자가 mmproj에서 f32 변환해 전달.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: std::sync::Arc<RawCtx>,
        weights: Vec<(String, Vec<f32>, usize, usize)>,
        n_embd: usize,
        n_head: usize,
        n_ff: usize,
        n_blk: usize,
        eps: f32,
        patch: usize,
        t_max: usize,
    ) -> Result<Self, String> {
        let mut w = HashMap::new();
        let ctx2 = ctx.clone();
        for (name, data, rows, ni) in weights {
            let p = ctx2.alloc(data.len() * 4).map_err(|e| e.to_string())?;
            ctx2.h2d(p, bytemuck::cast_slice(&data))?;
            w.insert(name, (p, rows, ni));
        }
        let a = |n: usize| -> Result<*mut u8, String> { ctx2.alloc(n.max(1) * 4).map_err(|e| e.to_string()) };
        Ok(Self {
            ctx,
            n_embd,
            n_head,
            d_head: n_embd / n_head,
            n_ff,
            n_blk,
            eps,
            patch,
            w,
            b_x: a(t_max * n_embd)?,
            b_xn: a(t_max * n_embd)?,
            b_qkv: a(t_max * 3 * n_embd)?,
            b_attn: a(t_max * n_embd)?,
            b_proj: a(t_max * n_embd)?,
            b_mid: a(t_max * n_ff)?,
            b_yx: a(t_max * 2)?,
            b_kvs: a(2 * t_max * n_embd)?,
            t_max,
        })
    }

    fn wt(&self, k: &str) -> Result<(*mut u8, usize, usize), String> {
        self.w
            .get(k)
            .copied()
            .ok_or_else(|| format!("vit weight 없음: {k}"))
    }

    fn time_stage(&self, label: &str, t0: &std::time::Instant) {
        if std::env::var_os("LLM170_VIT_TIME").is_some() {
            let _ = self.ctx.sync();
            eprintln!("[vtime] {label} +{:.2}ms", t0.elapsed().as_secs_f64() * 1e3);
        }
    }

    fn gemm(
        &self,
        x: *mut u8,
        wk: *mut u8,
        bk: *mut u8,
        ni: usize,
        n_out: usize,
        out: *mut u8,
        t: usize,
    ) -> Result<(), String> {
        let mut xp = x as *mut std::ffi::c_void;
        let mut wp = wk as *mut std::ffi::c_void;
        let mut bp = bk as *mut std::ffi::c_void;
        let mut op = out as *mut std::ffi::c_void;
        let mut ni_a = ni as i32;
        let mut no_a = n_out as i32;
        let mut tt = t as i32;
        let mut args = vec![
            arg(&mut xp), arg(&mut wp), arg(&mut bp), arg(&mut op),
            arg(&mut ni_a), arg(&mut no_a), arg(&mut tt),
        ];
        let gx = t.div_ceil(32) as u32;
        let gy = n_out.div_ceil(8) as u32;
        let gz = gy.div_ceil(65535) as u32;
        self.ctx.launch3("gemm_f32t", gx, gy.min(65535) as u32, gz, 256, &mut args)
    }

    /// toks: merge-major [t][n_embd] (conv+pos+bias 적용된 입력), yx: [t][2].
    /// 반환: merger 출력 [t/4][5120].
    pub fn forward(
        &self,
        toks: &[f32],
        yx: &[i32],
        pw: usize,
        ph: usize,
    ) -> Result<Vec<f32>, String> {
        let t = yx.len() / 2; // yx: [t][2]
        assert!(t <= self.t_max);
        if t > 2304 {
            // flash_vit의 smem 점수 버퍼 상한 — 초과 해상도는 호출자 폴백
            return Err(format!("vit: t={t} > 2304 (flash_vit smem 상한)"));
        }
        let (n, nh, dh) = (self.n_embd, self.n_head, self.d_head);
        self.ctx.h2d(self.b_x, bytemuck::cast_slice(toks))?;
        self.ctx.h2d(self.b_yx, bytemuck::cast_slice(yx))?;
        // 입력 → b_xn (작업 버퍼 역할: cur)
        // 입력 복사: b_x → b_xn (pack_strided 재활용: stride=n)
        {
            let mut sp = self.b_x as *mut std::ffi::c_void;
            let mut dp = self.b_xn as *mut std::ffi::c_void;
            let mut na = n as i32;
            let mut ss = n as i32;
            let mut tt = t as i32;
            let mut args = vec![arg(&mut sp), arg(&mut dp), arg(&mut na), arg(&mut ss), arg(&mut tt)];
            self.ctx.launch3("pack_strided", n.div_ceil(64) as u32, t as u32, 1, 64, &mut args)?;
        }
        let tmark = std::time::Instant::now();
        let mut tlast = tmark;
        for il in 0..self.n_blk {
            if il == 1 && std::env::var_os("LLM170_VIT_TIME").is_some() {
                self.time_stage("L0 total", &tlast);
                tlast = std::time::Instant::now();
            }
            // b_x → b_xn (레이어 입력 복사 — q 팩이 b_xn을 덮어쓰므로 매 층 갱신)
            self.pack_strided(self.b_x, 0, n, 1, self.b_xn, t)?;
            // LN1 → qkv
            self.ln(self.b_xn, &format!("v.blk.{il}.ln1"), t)?;
            let (wq, rows, ni) = self.wt(&format!("v.blk.{il}.attn_qkv.weight"))?;
            let bq = self.wt(&format!("v.blk.{il}.attn_qkv.bias"))?.0;
            self.gemm(self.b_xn, wq, bq, ni, rows, self.b_qkv, t)?;
            if il == 0 && std::env::var_os("LLM170_VIT_DBG").is_some() {
                self.ctx.sync()?;
                let mut v = vec![0f32; t * n];
                self.ctx.d2h(bytemuck::cast_slice_mut(&mut v).as_mut(), self.b_xn)?;
                let ssum: f64 = v.iter().map(|&x| x as f64).sum();
                eprintln!("[vit] L0 ln1 sum={ssum:.4} x0={:.6} x1={:.6}", v[0], v[1]);
                let mut q = vec![0f32; 8];
                self.ctx.d2h(bytemuck::cast_slice_mut(&mut q).as_mut(), self.b_qkv)?;
                eprintln!("[vit] L0 qkv q0..7={:?}", &q);
            }
            // rope q (offset 0) · k (offset n)
            for (off, stride) in [(0usize, 3 * n), (n, 3 * n)] {
                let mut qp = unsafe { self.b_qkv.add(off * 4) } as *mut std::ffi::c_void;
                let mut yxp = self.b_yx as *mut std::ffi::c_void;
                let mut nh_a = nh as i32;
                let mut dh_a = dh as i32;
                let mut tt = t as i32;
                let mut st = stride as i32;
                let mut args = vec![
                    arg(&mut qp), arg(&mut yxp), arg(&mut nh_a), arg(&mut dh_a), arg(&mut tt), arg(&mut st),
                ];
                self.ctx.launch3("vit_rope", t as u32, nh as u32, 1, 32, &mut args)?;
            }
            if il == 0 && std::env::var_os("LLM170_VIT_DBG").is_some() {
                self.ctx.sync()?;
                let mut q = vec![0f32; 8];
                let base = unsafe { self.b_qkv.add(0) };
                let _ = base;
                self.ctx.d2h(bytemuck::cast_slice_mut(&mut q).as_mut(), self.b_qkv)?;
                eprintln!("[vit] L0 roped q0..7={:?}", &q);
                let mut k = vec![0f32; 4];
                self.ctx.d2h(bytemuck::cast_slice_mut(&mut k).as_mut(), unsafe { self.b_qkv.add(n * 4) })?;
                eprintln!("[vit] L0 roped k0..3={:?}", &k);
            }
            if il == 1 && std::env::var_os("LLM170_VIT_TIME").is_some() {
                self.time_stage("L: ln1+qkv+rope", &tlast);
                tlast = std::time::Instant::now();
            }
            // attention — q/k/v는 qkv 내 off 0/n/2n, 행 스트라이드 3n → q도 팩 (b_xn 재활용)
            {
                self.pack_strided(self.b_qkv, 0, 3 * n, 1, self.b_xn, t)?;
                let qp = self.b_xn as *mut std::ffi::c_void;
                let kp = unsafe { self.b_qkv.add(n * 4) } as *mut std::ffi::c_void;
                let vp = unsafe { self.b_qkv.add(2 * n * 4) } as *mut std::ffi::c_void;
                // q/k/v가 3n 스트라이드 — flash_vit는 nh·d 스트라이드 가정 →
                // stride 변형 필요. 여기서는 커널에 stride를 추가하지 않고
                // k/v를 b_mid에 압축 복사(행별 n_embd) 후 실행.
                let ks = self.b_kvs;
                let vs = unsafe { self.b_kvs.add(t * n * 4) };
                self.pack_strided(self.b_qkv, n, 3 * n, 1, ks, t)?;
                self.pack_strided(self.b_qkv, 2 * n, 3 * n, 1, vs, t)?;
                let mut q2 = qp;
                let mut k2 = ks as *mut std::ffi::c_void;
                let mut v2 = vs as *mut std::ffi::c_void;
                let mut o2 = self.b_attn as *mut std::ffi::c_void;
                let mut np_ = t as i32;
                let mut nh_a = nh as i32;
                let mut dh_a = dh as i32;
                let mut sc = 1.0f32 / (dh as f32).sqrt();
                let mut args = vec![
                    arg(&mut q2), arg(&mut k2), arg(&mut v2), arg(&mut o2),
                    arg(&mut np_), arg(&mut nh_a), arg(&mut dh_a), arg(&mut sc),
                ];
                self.ctx.launch3("flash_vit", t as u32, nh as u32, 1, 256, &mut args)?;
            }
            if il == 0 && std::env::var_os("LLM170_VIT_DBG").is_some() {
                self.ctx.sync()?;
                let mut a = vec![0f32; 8];
                self.ctx.d2h(bytemuck::cast_slice_mut(&mut a).as_mut(), self.b_attn)?;
                let asum: f64 = {
                    let mut v = vec![0f32; t * n];
                    self.ctx.d2h(bytemuck::cast_slice_mut(&mut v).as_mut(), self.b_attn)?;
                    v.iter().map(|&x| x as f64).sum()
                };
                eprintln!("[vit] L0 attn sum={asum:.4} a0..7={:?}", &a);
            }
            if il == 1 && std::env::var_os("LLM170_VIT_TIME").is_some() {
                self.time_stage("L: attention", &tlast);
                tlast = std::time::Instant::now();
            }
            // attn_out proj + 잔차
            let (ow, orows, oni) = self.wt(&format!("v.blk.{il}.attn_out.weight"))?;
            let ob = self.wt(&format!("v.blk.{il}.attn_out.bias"))?.0;
            self.gemm(self.b_attn, ow, ob, oni, orows, self.b_proj, t)?;
            self.axpy(self.b_x, self.b_proj, t * n)?;
            // LN2 → FFN — 잔차 보존: b_x는 그대로, 정규화분은 b_xn에
            self.pack_strided(self.b_x, 0, n, 1, self.b_xn, t)?;
            self.ln(self.b_xn, &format!("v.blk.{il}.ln2"), t)?;
            let (uw, urows, uni) = self.wt(&format!("v.blk.{il}.ffn_up.weight"))?;
            let ub = self.wt(&format!("v.blk.{il}.ffn_up.bias"))?.0;
            self.gemm(self.b_xn, uw, ub, uni, urows, self.b_mid, t)?;
            {
                let mut mp = self.b_mid as *mut std::ffi::c_void;
                let mut na = (self.n_ff * t) as i32;
                let mut args = vec![arg(&mut mp), arg(&mut na)];
                self.ctx.launch("gelu_t", (self.n_ff * t).div_ceil(64) as u32, 1, 64, &mut args)?;
            }
            let (dw, drows, dni) = self.wt(&format!("v.blk.{il}.ffn_down.weight"))?;
            let db = self.wt(&format!("v.blk.{il}.ffn_down.bias"))?.0;
            self.gemm(self.b_mid, dw, db, dni, drows, self.b_proj, t)?;
            self.axpy(self.b_x, self.b_proj, t * n)?;
        }
        if std::env::var_os("LLM170_VIT_TIME").is_some() {
            self.time_stage("L: ffn+residual (last)", &tlast);
        }
        // post_ln → merger: [t/4][4n] pack → mm0 gelu → mm2
        self.ln(self.b_x, "v.post_ln", t)?;
        let n_out_tok = t / 4;
        let merger_buf = self.ctx.alloc(n_out_tok * 4 * n * 4).map_err(|e| e.to_string())?;
        // 2×2 pack (연속 4토큰 결합) — 호스트에서 하는 게 간단: d2h b_x → pack → h2d.
        let mut hx = vec![0f32; t * n];
        self.ctx.d2h(bytemuck::cast_slice_mut(&mut hx).as_mut(), self.b_x)?;
        let mut cat_rows = vec![0f32; n_out_tok * 4 * n];
        for m in 0..n_out_tok {
            for j in 0..4 {
                let (a0, b0) = ((m * 4 + j) * n, m * 4 * n + j * n);
                cat_rows[b0..b0 + n].copy_from_slice(&hx[a0..a0 + n]);
            }
        }
        let _ = merger_buf;
        // mid: [t/4][4n] — b_proj(n) 부족, b_mid(n_ff=4304) 부족(4608>4304) → alloc
        let mid_buf = self.ctx.alloc(n_out_tok * 4 * n * 4).map_err(|e| e.to_string())?;
        self.ctx.h2d(mid_buf, bytemuck::cast_slice(&cat_rows))?;
        let (m0, m0rows, m0ni) = self.wt("mm.0.weight")?;
        let m0b = self.wt("mm.0.bias")?.0;
        self.gemm(mid_buf, m0, m0b, m0ni, m0rows, merger_buf, n_out_tok)?;
        {
            let mut mp = merger_buf as *mut std::ffi::c_void;
            let mut na = (n_out_tok * 4 * n) as i32;
            let mut args = vec![arg(&mut mp), arg(&mut na)];
            self.ctx.launch("gelu_t", (n_out_tok * 4 * n).div_ceil(64) as u32, 1, 64, &mut args)?;
        }
        let (m2, m2rows, m2ni) = self.wt("mm.2.weight")?;
        let m2b = self.wt("mm.2.bias")?.0;
        let out_buf = self.ctx.alloc(n_out_tok * 5120 * 4).map_err(|e| e.to_string())?;
        self.gemm(merger_buf, m2, m2b, m2ni, m2rows, out_buf, n_out_tok)?;
        let mut out = vec![0f32; n_out_tok * 5120];
        self.ctx.d2h(bytemuck::cast_slice_mut(&mut out).as_mut(), out_buf)?;
        Ok(out)
    }

    fn ln(&self, buf: *mut u8, key: &str, t: usize) -> Result<(), String> {
        let n = self.n_embd;
        let (wp, _, _) = self.wt(&format!("{key}.weight"))?;
        let (bp, _, _) = self.wt(&format!("{key}.bias"))?;
        let mut xp = buf as *mut std::ffi::c_void;
        let mut w2 = wp as *mut std::ffi::c_void;
        let mut b2 = bp as *mut std::ffi::c_void;
        let mut na = n as i32;
        let mut ep = self.eps;
        let mut tt = t as i32;
        let mut args = vec![arg(&mut xp), arg(&mut w2), arg(&mut b2), arg(&mut na), arg(&mut ep), arg(&mut tt)];
        self.ctx.launch3("layernorm_t", t as u32, 1, 1, 32, &mut args)
    }

    fn axpy(&self, y: *mut u8, x: *mut u8, n: usize) -> Result<(), String> {
        // y += x
        let mut yp = y as *mut std::ffi::c_void;
        let mut xp = x as *mut std::ffi::c_void;
        let mut na = n as i32;
        let mut args = vec![arg(&mut yp), arg(&mut xp), arg(&mut na)];
        // axpy_scaled(y, x, one, n) — one 버퍼는 DecodeState 것과 별도 필요 → 간단 커널 대신
        // gemm 없이: 전용 add 커널 사용
        self.ctx.launch("add_f32", n.div_ceil(64) as u32, 1, 64, &mut args)
    }

    /// src [t][src_stride]에서 [t][n] 팩.
    fn pack_strided(
        &self,
        src: *mut u8,
        off: usize,
        src_stride: usize,
        _dst_stride: usize,
        dst: *mut u8,
        t: usize,
    ) -> Result<(), String> {
        let n = self.n_embd;
        let mut sp = unsafe { src.add(off * 4) } as *mut std::ffi::c_void;
        let mut dp = dst as *mut std::ffi::c_void;
        let mut na = n as i32;
        let mut ss = src_stride as i32;
        let mut tt = t as i32;
        let mut args = vec![arg(&mut sp), arg(&mut dp), arg(&mut na), arg(&mut ss), arg(&mut tt)];
        self.ctx.launch3("pack_strided", n.div_ceil(64) as u32, t as u32, 1, 64, &mut args)
    }
}

#[inline]
fn arg<T>(v: &mut T) -> *mut std::ffi::c_void {
    v as *mut T as *mut std::ffi::c_void
}
