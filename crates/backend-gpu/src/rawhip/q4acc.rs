//! qwen4exp(rawhip) 가속기 — Engine4용 `Accelerator` 값 경로 구현.
//!
//! 배경: cubecl 제거(ADR-0018)로 qwen4exp의 유일한 가속기가 사라져
//! 2026-09-12 기준 qwen4exp는 CPU 전용(pp32 1.77 / tg8 0.56 t/s)이었다.
//! 이 모듈은 그 값을 rawhip 위에 복원한다 — 설계는 삭제된 cubecl
//! 구현(backend-gpu/src/lib.rs @921a411^)의 계약을 따르되 커널은 기존
//! rawhip 자산(GEMV 패밀리·quant_q8)과 신규 3종(q5_1·f32·QSA 어텐션)을 쓴다.
//!
//! 계약 (ADR-0014 개정판: 해제 없음):
//! - 무게: mmap 포인터 키로 1회 업로드 후 영구 상주 (`Weight.data.as_ptr()`).
//! - 활성: 값 경로는 호출마다 h2d → q8 양자화 → GEMV/GEMM → d2h.
//!   (프레임 경로는 후속 — device 상주 op 세트)
//! - 산술: 기존 커널과 동일 계열(정수 isum 정확, 그룹 스케일·f64 부분합
//!   순서만 CPU와 상이 — 문서화된 ≤2.4e-7 계열).

use cubecl_hip_sys as hip;
use llm170_gguf::GgmlType;

use super::{ck, RawCtx};

/// 용도별 성장형 디바이스 버퍼 (해제 없음 — ADR-0014).
struct GBuf {
    bytes: usize,
    ptr: *mut u8,
}

impl GBuf {
    const fn new() -> Self {
        GBuf { bytes: 0, ptr: std::ptr::null_mut() }
    }

    fn ensure(&mut self, ctx: &RawCtx, bytes: usize) -> Result<*mut u8, String> {
        if bytes > self.bytes {
            self.ptr = ctx.alloc(bytes)?;
            self.bytes = bytes;
        }
        Ok(self.ptr)
    }
}

pub struct Q4Acc {
    ctx: RawCtx,
    ktab2: *mut u8,
    /// 업로드된 무게 — (mmap ptr → (device ptr, f32 레이아웃 여부)). 해제 없음.
    /// f32 레이아웃 = 무양자화(F32) 또는 업로드 시 f32로 전개한 Bf16/F16
    /// (인덱서 투영 — bf16 24텐서 0.04 GiB, 전개 비용 무시 가능).
    weights: std::sync::Mutex<std::collections::HashMap<usize, (*mut u8, bool)>>,
    wbytes: std::sync::atomic::AtomicUsize,
    xf: std::sync::Mutex<GBuf>,
    xq: std::sync::Mutex<GBuf>,
    yf: std::sync::Mutex<GBuf>,
    qs: std::sync::Mutex<GBuf>,
    ckv: std::sync::Mutex<GBuf>,
    cvv: std::sync::Mutex<GBuf>,
    msk: std::sync::Mutex<GBuf>,
    atn: std::sync::Mutex<GBuf>,
}

// SAFETY: 포인터는 디바이스 주소 — 스레드 간 공유해도 HIP 런타임이 직렬화한다
// (단일 스트림 + 호출부는 decode1을 직렬 호출). VkAcc와 동일한 계약.
unsafe impl Send for Q4Acc {}
unsafe impl Sync for Q4Acc {}

/// 활성 q8 버퍼의 행 스트라이드(워드) — quant_q8과 동일 규약.
fn xq_words(n: usize) -> usize {
    n / 4 + n / 32 + n / 16
}

fn ggml_id(ty: GgmlType) -> u32 {
    ty as u32
}

impl Q4Acc {
    pub fn new() -> Result<Self, String> {
        let ctx = RawCtx::new()?;
        let ktab2 = {
            let p = ctx.alloc(1024)?;
            let kt: Vec<u32> = llm170_core::ktab2_packed();
            ctx.h2d(p, bytemuck::cast_slice(&kt))?;
            p
        };
        Ok(Q4Acc {
            ctx,
            ktab2,
            weights: Default::default(),
            wbytes: Default::default(),
            xf: std::sync::Mutex::new(GBuf::new()),
            xq: std::sync::Mutex::new(GBuf::new()),
            yf: std::sync::Mutex::new(GBuf::new()),
            qs: std::sync::Mutex::new(GBuf::new()),
            ckv: std::sync::Mutex::new(GBuf::new()),
            cvv: std::sync::Mutex::new(GBuf::new()),
            msk: std::sync::Mutex::new(GBuf::new()),
            atn: std::sync::Mutex::new(GBuf::new()),
        })
    }

    /// 업로드 누적 바이트 (진단).
    pub fn uploaded_bytes(&self) -> usize {
        self.wbytes.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// 무게 1회 업로드 후 상주 — mmap 포인터가 키.
    /// 반환: (device ptr, f32 레이아웃 여부). Bf16/F16은 f32로 전개해 올린다
    /// (CPU `dequant_row`와 동일한 bf16_to_f32/half_to_f32 — 비트 동일).
    fn dev_weight(&self, w: &llm170_core::matmul::Weight<'_>) -> Result<(*mut u8, bool), String> {
        let key = w.data.as_ptr() as usize;
        if let Some(v) = self.weights.lock().map_err(|e| e.to_string())?.get(&key) {
            return Ok(*v);
        }
        let (ptr, is_f32) = match w.ty {
            GgmlType::F32 => (self.ctx.alloc(w.data.len().max(1))?, true),
            GgmlType::Bf16 | GgmlType::F16 => {
                let n = w.data.len() / 2;
                let mut v = Vec::with_capacity(n * 4);
                for i in 0..n {
                    let h = u16::from_le_bytes([w.data[i * 2], w.data[i * 2 + 1]]);
                    let f = if w.ty == GgmlType::F16 {
                        llm170_core::quant::half_to_f32(h)
                    } else {
                        llm170_core::quant::bf16_to_f32(h)
                    };
                    v.extend_from_slice(&f.to_le_bytes());
                }
                let p = self.ctx.alloc(v.len().max(1))?;
                self.ctx.h2d(p, &v)?;
                self.wbytes
                    .fetch_add(v.len(), std::sync::atomic::Ordering::Relaxed);
                self.weights
                    .lock()
                    .map_err(|e| e.to_string())?
                    .insert(key, (p, true));
                return Ok((p, true));
            }
            _ => (self.ctx.alloc(w.data.len().max(1))?, false),
        };
        self.ctx.h2d(ptr, w.data)?;
        self.wbytes
            .fetch_add(w.data.len(), std::sync::atomic::Ordering::Relaxed);
        self.weights
            .lock()
            .map_err(|e| e.to_string())?
            .insert(key, (ptr, is_f32));
        Ok((ptr, is_f32))
    }

    /// GEMV/GEMM 1런치 — xq는 이미 업로드·양자화된 활성 포인터.
    fn launch_gemm(
        &self,
        ty: u32,
        xq: *mut u8,
        w: *mut u8,
        n_in: usize,
        n_out: usize,
        xq_w: usize,
        t: usize,
        out: *mut u8,
    ) -> Result<(), String> {
        if ty == ggml_id(GgmlType::Q5_1) {
            let gy = n_out.min(65535) as u32;
            let gz = n_out.div_ceil(65535) as u32;
            let part = self.ctx.scratch(n_out * 64 * 8)?;
            let mut xq_p = xq as *mut std::ffi::c_void;
            let mut w_p = w as *mut std::ffi::c_void;
            let mut part_p = part as *mut std::ffi::c_void;
            let mut o_p = out as *mut std::ffi::c_void;
            let mut ni = n_in as i32;
            let mut no = n_out as i32;
            let mut xw = xq_w as i32;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut xq_p) as *mut _ as *mut std::ffi::c_void,
                (&mut w_p) as *mut _ as *mut std::ffi::c_void,
                (&mut part_p) as *mut _ as *mut std::ffi::c_void,
                (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                (&mut ni) as *mut _ as *mut std::ffi::c_void,
                (&mut no) as *mut _ as *mut std::ffi::c_void,
                (&mut xw) as *mut _ as *mut std::ffi::c_void,
            ];
            return self.ctx.launch3("q4_gemm_q5_1", t as u32, gy, gz, 64, &mut args);
        }
        self.ctx.gemv_q8_out(
            xq as *const u8,
            w as *const u8,
            self.ktab2 as *const u8,
            ty,
            n_in,
            n_out,
            out,
            xq_w,
            t,
        )
    }

    /// f32 무게(라우터 등) GEMV — 양자화 없이 업로드한 활성을 직접 소비.
    fn launch_gemm_f32(
        &self,
        x: *mut u8,
        w: *mut u8,
        n_in: usize,
        n_out: usize,
        t: usize,
        out: *mut u8,
    ) -> Result<(), String> {
        let gy = n_out.min(65535) as u32;
        let gz = n_out.div_ceil(65535) as u32;
        let mut x_p = x as *mut std::ffi::c_void;
        let mut w_p = w as *mut std::ffi::c_void;
        let mut o_p = out as *mut std::ffi::c_void;
        let mut ni = n_in as i32;
        let mut no = n_out as i32;
        let mut st = n_in as i32;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&mut x_p) as *mut _ as *mut std::ffi::c_void,
            (&mut w_p) as *mut _ as *mut std::ffi::c_void,
            (&mut o_p) as *mut _ as *mut std::ffi::c_void,
            (&mut ni) as *mut _ as *mut std::ffi::c_void,
            (&mut no) as *mut _ as *mut std::ffi::c_void,
            (&mut st) as *mut _ as *mut std::ffi::c_void,
        ];
        self.ctx.launch3("q4_gemm_f32", t as u32, gy, gz, 64, &mut args)
    }

    /// 배치 GEMM 본체 — xs [t][n_in] f32 → outs [t][n_out] f32.
    /// `x_start`/`w_off`은 moe_down의 전문가 그룹 런치용 부분 범위.
    fn batch_into(
        &self,
        xs: &[Vec<f32>],
        outs: &mut [Vec<f32>],
        w: &llm170_core::matmul::Weight<'_>,
        w_off_bytes: usize,
    ) -> Result<(), String> {
        let t = xs.len();
        if t == 0 {
            return Ok(());
        }
        let n_in = w.n_in as usize;
        let n_out = w.n_out as usize;
        let ty = ggml_id(w.ty);
        let (w_dev, w_f32) = self.dev_weight(w)?;
        let w_slice = unsafe { w_dev.add(w_off_bytes) };
        // 활성 업로드 (연속 f32)
        let mut xflat = Vec::with_capacity(t * n_in);
        for row in xs {
            if row.len() != n_in {
                return Err(format!("matmul_batch: x({}) != n_in({})", row.len(), n_in));
            }
            xflat.extend_from_slice(row);
        }
        let mut yflat = vec![0.0f32; t * n_out];
        let mut out_dev = None;
        {
            let mut yb = self.yf.lock().map_err(|e| e.to_string())?;
            let ydev = yb.ensure(&self.ctx, t * n_out * 4)?;
            out_dev = Some(ydev);
        }
        let ydev = out_dev.unwrap();
        if w_f32 {
            let xdev = {
                let mut xb = self.xf.lock().map_err(|e| e.to_string())?;
                xb.ensure(&self.ctx, t * n_in * 4)?
            };
            self.ctx.h2d(xdev, bytemuck::cast_slice(&xflat))?;
            self.launch_gemm_f32(xdev, w_slice, n_in, n_out, t, ydev)?;
        } else {
            let xq_w = xq_words(n_in);
            let xq_buf = {
                let mut xb = self.xq.lock().map_err(|e| e.to_string())?;
                xb.ensure(&self.ctx, t * xq_w * 4)?
            };
            let xdev = {
                let mut xb = self.xf.lock().map_err(|e| e.to_string())?;
                xb.ensure(&self.ctx, t * n_in * 4)?
            };
            self.ctx.h2d(xdev, bytemuck::cast_slice(&xflat))?;
            self.ctx.quant_q8_b(xdev, xq_buf, n_in, xq_w, t)?;
            self.launch_gemm(ty, xq_buf, w_slice, n_in, n_out, xq_w, t, ydev)?;
        }
        self.ctx.d2h(bytemuck::cast_slice_mut(&mut yflat), ydev)?;
        for (o, v) in outs.iter_mut().zip(yflat.chunks_exact(n_out)) {
            o.copy_from_slice(v);
        }
        Ok(())
    }
}

impl llm170_core::matmul::FrameState for Q4Acc {}

impl llm170_core::matmul::Accelerator for Q4Acc {
    fn barrier(&self) {
        unsafe {
            let _ = ck(hip::hipDeviceSynchronize(), "hipDeviceSynchronize");
        }
    }

    fn matmul(
        &self,
        x: &[f32],
        w: &llm170_core::matmul::Weight<'_>,
        out: &mut [f32],
    ) -> Result<(), String> {
        let mut o = vec![vec![0.0f32; w.n_out as usize]];
        let xs = [x.to_vec()];
        self.batch_into(&xs, &mut o, w, 0)?;
        out.copy_from_slice(&o[0]);
        Ok(())
    }

    fn matmul_batch(
        &self,
        xs: &[Vec<f32>],
        w: &llm170_core::matmul::Weight<'_>,
        outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        self.batch_into(xs, outs, w, 0)
    }

    fn matmul_group(
        &self,
        xs: &[Vec<f32>],
        ws: &[llm170_core::matmul::Weight<'_>],
        outs: &mut [Vec<Vec<f32>>],
    ) -> Result<(), String> {
        // 동일 입력 — 양자화 1회 후 가중치별 런치. (값 경로 x 업로드 절약)
        if ws.len() != outs.len() {
            return Err(format!("matmul_group: ws({}) != outs({})", ws.len(), outs.len()));
        }
        for (w, o) in ws.iter().zip(outs.iter_mut()) {
            self.batch_into(xs, o, w, 0)?;
        }
        Ok(())
    }

    fn matmul_paired(
        &self,
        xs: &[Vec<f32>],
        ws: &[llm170_core::matmul::Weight<'_>],
        outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        if ws.len() != xs.len() || ws.len() != outs.len() {
            return Err(format!(
                "matmul_paired: 형상 불일치 ws={} xs={} outs={}",
                ws.len(),
                xs.len(),
                outs.len()
            ));
        }
        for ((x, w), o) in xs.iter().zip(ws.iter()).zip(outs.iter_mut()) {
            let one = [x.clone()];
            let mut oo = [std::mem::take(o)];
            self.batch_into(&one, &mut oo, w, 0)?;
            *o = std::mem::take(&mut oo[0]);
        }
        Ok(())
    }

    /// MoE 전문가 스택 배치 — ids로 전문가를 묶어 그룹별 런치.
    /// ids가 가리키는 전문가 슬라이스는 스택에서 연속이므로 바이트 오프셋만
    /// 옮기면 기존 GEMV 커널이 그대로 성립한다(mul_mat_id의 오프셋 형태).
    fn moe_down(
        &self,
        xs: &[Vec<f32>],
        ws: &llm170_core::matmul::Weight<'_>,
        expert_ids: &[u32],
        n_expert_stack: usize,
        outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        let t = xs.len();
        if t != expert_ids.len() || t != outs.len() {
            return Err(format!(
                "moe_down: 형상 불일치 xs={} ids={} outs={}",
                t,
                expert_ids.len(),
                outs.len()
            ));
        }
        if t == 0 {
            return Ok(());
        }
        let per_expert = ws.data.len() / n_expert_stack.max(1);
        let n_in = ws.n_in as usize;
        let n_out = ws.n_out as usize;
        let (w_dev, w_f32) = self.dev_weight(ws)?;
        // 활성 업로드 + 양자화 1회 (전문가 공통)
        let mut xflat = Vec::with_capacity(t * n_in);
        for row in xs {
            if row.len() != n_in {
                return Err(format!("moe_down: x({}) != n_in({})", row.len(), n_in));
            }
            xflat.extend_from_slice(row);
        }
        let mut yflat = vec![0.0f32; t * n_out];
        let xq_w = xq_words(n_in);
        let (xdev, xq_buf, ydev, xdev_f32) = {
            let mut xb = self.xf.lock().map_err(|e| e.to_string())?;
            let xdev = xb.ensure(&self.ctx, t * n_in * 4)?;
            let mut qb = self.xq.lock().map_err(|e| e.to_string())?;
            let xq_buf = qb.ensure(&self.ctx, t * xq_w * 4)?;
            let mut yb = self.yf.lock().map_err(|e| e.to_string())?;
            let ydev = yb.ensure(&self.ctx, t * n_out * 4)?;
            (xdev, xq_buf, ydev, xdev)
        };
        self.ctx.h2d(xdev, bytemuck::cast_slice(&xflat))?;
        if !w_f32 {
            self.ctx.quant_q8_b(xdev, xq_buf, n_in, xq_w, t)?;
        }
        // 전문가별 연속 그룹 런치
        let mut i = 0usize;
        while i < t {
            let e = expert_ids[i];
            let mut j = i + 1;
            while j < t && expert_ids[j] == e {
                j += 1;
            }
            let rows = j - i;
            let xsrc = if w_f32 {
                unsafe { xdev_f32.add(i * n_in * 4) }
            } else {
                unsafe { xq_buf.add(i * xq_w * 4) }
            };
            let wsrc = unsafe { w_dev.add(e as usize * per_expert) };
            let dst = unsafe { ydev.add(i * n_out * 4) };
            if w_f32 {
                self.launch_gemm_f32(xsrc, wsrc, n_in, n_out, rows, dst)?;
            } else {
                self.launch_gemm(ggml_id(ws.ty), xsrc, wsrc, n_in, n_out, xq_w, rows, dst)?;
            }
            i = j;
        }
        self.ctx.d2h(bytemuck::cast_slice_mut(&mut yflat), ydev)?;
        for (o, v) in outs.iter_mut().zip(yflat.chunks_exact(n_out)) {
            o.copy_from_slice(v);
        }
        Ok(())
    }

    /// QSA 마스크드 밀집 GQA (값 경로 브리지) — f32 캐시.
    #[allow(clippy::too_many_arguments)]
    fn qsa_attention(
        &self,
        q: &[f32],
        ck: &[f32],
        cv: &[f32],
        mask: &[u32],
        kq_scale: f32,
        n_past: usize,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
    ) -> Result<Vec<f32>, String> {
        let (qdev, kdev, vdev, mdev, odev) = {
            let mut a = self.qs.lock().map_err(|e| e.to_string())?;
            let qdev = a.ensure(&self.ctx, q.len() * 4)?;
            let mut b = self.ckv.lock().map_err(|e| e.to_string())?;
            let kdev = b.ensure(&self.ctx, ck.len() * 4)?;
            let mut c = self.cvv.lock().map_err(|e| e.to_string())?;
            let vdev = c.ensure(&self.ctx, cv.len() * 4)?;
            let mut d = self.msk.lock().map_err(|e| e.to_string())?;
            let mdev = d.ensure(&self.ctx, mask.len() * 4)?;
            let mut e2 = self.atn.lock().map_err(|e| e.to_string())?;
            let odev = e2.ensure(&self.ctx, t * n_head * hd * 4)?;
            (qdev, kdev, vdev, mdev, odev)
        };
        self.ctx.h2d(qdev, bytemuck::cast_slice(q))?;
        self.ctx.h2d(kdev, bytemuck::cast_slice(ck))?;
        self.ctx.h2d(vdev, bytemuck::cast_slice(cv))?;
        self.ctx.h2d(mdev, bytemuck::cast_slice(mask))?;
        let mut q_p = qdev as *mut std::ffi::c_void;
        let mut k_p = kdev as *mut std::ffi::c_void;
        let mut v_p = vdev as *mut std::ffi::c_void;
        let mut m_p = mdev as *mut std::ffi::c_void;
        let mut o_p = odev as *mut std::ffi::c_void;
        let mut sc = kq_scale;
        let mut np_ = n_past as i32;
        let mut nh = n_head as i32;
        let mut nk = n_kv as i32;
        let mut h = hd as i32;
        let mut tt = t as i32;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&mut q_p) as *mut _ as *mut std::ffi::c_void,
            (&mut k_p) as *mut _ as *mut std::ffi::c_void,
            (&mut v_p) as *mut _ as *mut std::ffi::c_void,
            (&mut m_p) as *mut _ as *mut std::ffi::c_void,
            (&mut o_p) as *mut _ as *mut std::ffi::c_void,
            (&mut sc) as *mut _ as *mut std::ffi::c_void,
            (&mut np_) as *mut _ as *mut std::ffi::c_void,
            (&mut nh) as *mut _ as *mut std::ffi::c_void,
            (&mut nk) as *mut _ as *mut std::ffi::c_void,
            (&mut h) as *mut _ as *mut std::ffi::c_void,
            (&mut tt) as *mut _ as *mut std::ffi::c_void,
        ];
        self.ctx
            .launch3("q4_qsa_attn", t as u32, n_head as u32, 1, 256, &mut args)?;
        let mut out = vec![0.0f32; t * n_head * hd];
        self.ctx.d2h(bytemuck::cast_slice_mut(&mut out), odev)?;
        Ok(out)
    }
}

/// Engine4에 주입할 가속기 생성 — 실패 시 호출부가 CPU로 폴백(경고).
pub fn new_acc() -> Result<std::sync::Arc<dyn llm170_core::matmul::Accelerator>, String> {
    let a = Q4Acc::new()?;
    eprintln!(
        "# q4acc: rawhip 가속기 준비 (무게는 첫 사용 시 업로드·영구 상주, ADR-0014)"
    );
    Ok(std::sync::Arc::new(a))
}

/// q5_1 커널 마이크로 검증 — 합성 블록 1개(d=1.0, m=-0.5, q=i%32)로
/// GPU ↔ CPU 레인 미러를 원소 수준에서 대조한다 (`q4-acc-check micro`).
pub fn micro_check() -> Result<String, String> {
    use llm170_core::matmul::Accelerator;
    let n = 32usize;
    let mut bytes = vec![0u8; 24];
    bytes[0] = 0x00;
    bytes[1] = 0x3C; // d = 1.0
    bytes[2] = 0x00;
    bytes[3] = 0xB8; // m = -0.5
    let mut qh = 0u32;
    for i in 0..n {
        let q = (i % 32) as u32;
        if (q >> 4) & 1 == 1 {
            qh |= 1 << i;
        }
    }
    bytes[4..8].copy_from_slice(&qh.to_le_bytes());
    for j in 0..16usize {
        let lo = ((j) % 32) as u8 & 0xF;
        let hi = ((16 + j) % 32) as u8 & 0xF;
        bytes[8 + j] = lo | (hi << 4);
    }
    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 0.15).collect();
    let w = llm170_core::matmul::Weight { data: &bytes, ty: GgmlType::Q5_1, n_in: n as u64, n_out: 1 };
    let acc = Q4Acc::new()?;
    let mut gpu = vec![vec![0.0f32; 1]];
    acc.matmul_batch(&[x.clone()], &w, &mut gpu)?;
    let y = llm170_core::quant::quantize_row_q8_ref(&x);
    let cpu = llm170_core::quant::dot_row_w4a8_q5_1_lane(&bytes, n as u64, &y);
    Ok(format!(
        "micro q5_1: gpu={:?} cpu={cpu:.6} qh={qh:#010x} block={:02x?}",
        gpu[0][0], bytes
    ))
}

/// `q4-acc-check <model> <tensor> [t] [rows]` — GPU(가속기 값 경로) ↔ CPU
/// W4A8 레인 미러 대조. 계약: 같은 산술 계열이므로 ≤1e-6(비트 일치 기대).
pub fn check_tensor(
    model: &std::path::Path,
    tensor: &str,
    t: usize,
    rows_max: usize,
) -> Result<String, String> {
    use llm170_core::matmul::Accelerator;
    let m = llm170_core::qwen4exp::Model4::load(model).map_err(|e| e.to_string())?;
    let w = m
        .w(tensor)
        .ok_or_else(|| format!("텐서 없음: {tensor}"))?;
    let n_in = w.n_in as usize;
    let (blck, bsize) = w.ty.block_info();
    let row_bytes = (n_in / blck as usize) * bsize as usize;
    let n_out = (w.n_out as usize).min(rows_max.max(1));
    let ws = llm170_core::matmul::Weight {
        data: &w.data[..n_out * row_bytes],
        ty: w.ty,
        n_in: w.n_in,
        n_out: n_out as u64,
    };
    // 결정적 입력 (LCG, ±0.5)
    let mut seed = 0x1234_5678u64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
    };
    let xs: Vec<Vec<f32>> = (0..t).map(|_| (0..n_in).map(|_| lcg()).collect()).collect();
    let acc = Q4Acc::new()?;
    let mut gpu = vec![vec![0.0f32; n_out]; t];
    acc.matmul_batch(&xs, &ws, &mut gpu)?;
    // CPU W4A8 레인 미러
    let mut cpu = vec![vec![0.0f32; n_out]; t];
    for (ti, x) in xs.iter().enumerate() {
        if matches!(ws.ty, GgmlType::F32 | GgmlType::Bf16 | GgmlType::F16) {
            // f32 계열 — 가속기도 f32로 전개해 f32 커널을 쓴다. 환원 순서만
            // 다르므로 f64 기준 대비 상대오차로 판정한다.
            let mut wrow = vec![0.0f32; n_in];
            for (o, out) in cpu[ti].iter_mut().enumerate() {
                llm170_core::quant::dequant_row(
                    ws.ty,
                    &ws.data[o * row_bytes..],
                    0,
                    n_in as u64,
                    &mut wrow,
                );
                let mut s = 0.0f64;
                for i in 0..n_in {
                    s += x[i] as f64 * wrow[i] as f64;
                }
                *out = s as f32;
            }
            continue;
        }
        let y = llm170_core::quant::quantize_row_q8_ref(x);
        for (o, out) in cpu[ti].iter_mut().enumerate() {
            let row = &ws.data[o * row_bytes..(o + 1) * row_bytes];
            *out = match ws.ty {
                GgmlType::Q4K => llm170_core::quant::dot_row_w4a8_q4k_lane(row, ws.n_in, &y),
                GgmlType::Q5K => llm170_core::quant::dot_row_w4a8_q5k_lane(row, ws.n_in, &y),
                GgmlType::Q6K => llm170_core::quant::dot_row_w4a8_q6k_lane(row, ws.n_in, &y),
                GgmlType::Q3K => llm170_core::quant::dot_row_w4a8_q3k_lane(row, ws.n_in, &y),
                GgmlType::Q8_0 => llm170_core::quant::dot_row_w4a8_q8_0_lane(row, ws.n_in, &y),
                GgmlType::Q5_1 => llm170_core::quant::dot_row_w4a8_q5_1_lane(row, ws.n_in, &y),
                GgmlType::Iq4Nl => llm170_core::quant::dot_row_w4a8_iq4nl_lane(row, ws.n_in, &y),
                GgmlType::Iq3S => llm170_core::quant::dot_row_w4a8_iq3s_lane(row, ws.n_in, &y),
                GgmlType::Iq4Xs => llm170_core::quant::dot_row_w4a8_iq4xs_lane(row, ws.n_in, &y),
                other => return Err(format!("q4-acc-check: 미지원 타입 {other:?} — /dev/null")),
            };
        }
    }
    let (mut max_abs, mut max_rel, mut bit_eq, mut n) = (0.0f64, 0.0f64, 0usize, 0usize);
    for (g, c) in gpu.iter().zip(cpu.iter()) {
        for (a, b) in g.iter().zip(c.iter()) {
            let d = (*a - *b).abs() as f64;
            max_abs = max_abs.max(d);
            max_rel = max_rel.max(d / b.abs().max(1e-3) as f64);
            bit_eq += (a.to_bits() == b.to_bits()) as usize;
            n += 1;
        }
    }
    if std::env::var_os("LLM170_Q4ACC_DBG").is_some() {
        eprintln!("# gpu[0][..8] = {:?}", &gpu[0][..8]);
        eprintln!("# cpu[0][..8] = {:?}", &cpu[0][..8]);
    }
    Ok(format!(
        "q4-acc-check {tensor} [{n_out}x{n_in}] ty={:?} t={t}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} bit_eq={}/{} ({:.1}%)",
        ws.ty,
        bit_eq,
        n,
        100.0 * bit_eq as f64 / n as f64
    ))
}
