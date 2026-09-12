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
    name: &'static str,
    bytes: usize,
    ptr: *mut u8,
}

impl GBuf {
    const fn new(name: &'static str) -> Self {
        GBuf { name, bytes: 0, ptr: std::ptr::null_mut() }
    }

    fn ensure(&mut self, ctx: &RawCtx, bytes: usize) -> Result<*mut u8, String> {
        if bytes > self.bytes {
            if std::env::var_os("LLM170_Q4ACC_STATS").is_some() {
                eprintln!("# q4acc: {} 확장 {} → {} B", self.name, self.bytes, bytes);
            }
            self.ptr = ctx.alloc(bytes)?;
            self.bytes = bytes;
        }
        Ok(self.ptr)
    }
}

/// 무게 파일 소스 — mmap 베이스 주소 범위 + 파일 핸들 (staged pread 업로드용).
struct Source {
    base: usize,
    len: usize,
    file: std::fs::File,
}

/// per-op 시간 누적 (LLM170_Q4ACC_TIME=1) — (업로드, 양자화, 런치, d2h, 호출수)
#[derive(Default)]
struct AccTime {
    upload_ns: u64,
    quant_ns: u64,
    launch_ns: u64,
    d2h_ns: u64,
    calls: u64,
}

pub struct Q4Acc {
    ctx: RawCtx,
    ktab2: *mut u8,
    /// 업로드된 무게 — (mmap ptr → (device ptr, f32 레이아웃 여부)). 해제 없음.
    /// f32 레이아웃 = 무양자화(F32) 또는 업로드 시 f32로 전개한 Bf16/F16
    /// (인덱서 투영 — bf16 24텐서 0.04 GiB, 전개 비용 무시 가능).
    weights: std::sync::Mutex<std::collections::HashMap<usize, (*mut u8, bool)>>,
    wbytes: std::sync::atomic::AtomicUsize,
    time: std::sync::Mutex<AccTime>,
    /// 모델 파트 파일 — 있으면 업로드가 mmap 폴트 대신 pread 스테이징을 쓴다.
    sources: Vec<Source>,
    stage: std::sync::Mutex<Vec<u8>>,
    /// 프레임 버퍼 레지스트리 — 핸들 = 인덱스+1 (해제 없음, ADR-0014).
    frames: std::sync::Mutex<Vec<(*mut u8, usize)>>,
    /// 프레임 활성 q8 스크래치 (값 경로 xq와 분리 — 프레임/값 교차 안전).
    fxq: std::sync::Mutex<GBuf>,
    /// rms_part 부분합 스크래치 (rows×32 double).
    fpart: std::sync::Mutex<GBuf>,
    /// 현재 프레임 스텝의 토큰 수 (frame_begin).
    cur_t: std::sync::atomic::AtomicUsize,
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
        Self::new_with_sources(Vec::new())
    }

    /// 파트 소스 지정판 — (`Model4::part_sources`). 비어 있으면 mmap 폴트 폴백.
    pub fn new_with_sources(parts: Vec<(usize, usize, std::path::PathBuf)>) -> Result<Self, String> {
        let ctx = RawCtx::new()?;
        let ktab2 = {
            let p = ctx.alloc(1024)?;
            let kt: Vec<u32> = llm170_core::ktab2_packed();
            ctx.h2d(p, bytemuck::cast_slice(&kt))?;
            p
        };
        let mut sources = Vec::with_capacity(parts.len());
        for (base, len, path) in parts {
            match std::fs::File::open(&path) {
                Ok(file) => sources.push(Source { base, len, file }),
                Err(e) => eprintln!("# q4acc: 파트 열기 실패 {} — mmap 폴백 ({e})", path.display()),
            }
        }
        Ok(Q4Acc {
            ctx,
            ktab2,
            weights: Default::default(),
            wbytes: Default::default(),
            time: Default::default(),
            sources,
            stage: std::sync::Mutex::new(Vec::new()),
            frames: Default::default(),
            fxq: std::sync::Mutex::new(GBuf::new("fxq")),
            fpart: std::sync::Mutex::new(GBuf::new("fpart")),
            cur_t: std::sync::atomic::AtomicUsize::new(1),
            xf: std::sync::Mutex::new(GBuf::new("xf")),
            xq: std::sync::Mutex::new(GBuf::new("xq")),
            yf: std::sync::Mutex::new(GBuf::new("yf")),
            qs: std::sync::Mutex::new(GBuf::new("qs")),
            ckv: std::sync::Mutex::new(GBuf::new("ckv")),
            cvv: std::sync::Mutex::new(GBuf::new("cvv")),
            msk: std::sync::Mutex::new(GBuf::new("msk")),
            atn: std::sync::Mutex::new(GBuf::new("atn")),
        })
    }

    fn note(&self, up: u64, q: u64, k: u64, d: u64) {
        if let Ok(mut t) = self.time.lock() {
            t.upload_ns += up;
            t.quant_ns += q;
            t.launch_ns += k;
            t.d2h_ns += d;
            t.calls += 1;
            if t.calls % 200 == 0 {
                eprintln!(
                    "# q4acc[{}]: upload={:.1}s quant={:.1}s launch={:.1}s d2h={:.1}s (평균 {:.1} ms/호출)",
                    t.calls,
                    t.upload_ns as f64 / 1e9,
                    t.quant_ns as f64 / 1e9,
                    t.launch_ns as f64 / 1e9,
                    t.d2h_ns as f64 / 1e9,
                    (t.upload_ns + t.quant_ns + t.launch_ns + t.d2h_ns) as f64 / 1e6 / t.calls as f64
                );
            }
        }
    }

    /// 업로드 누적 바이트 (진단).
    pub fn uploaded_bytes(&self) -> usize {
        self.wbytes.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// madvise 래퍼 (페이지 정렬 가정 — mmap 베이스 + 4096 배수 오프셋).
    fn advise(addr: usize, len: usize, advice: libc::c_int) {
        if len == 0 {
            return;
        }
        unsafe {
            let _ = libc::madvise(addr as *mut libc::c_void, len, advice);
        }
    }

    /// 대형 텐서 업로드 — 8 MiB 청크로 파이프라인 선반입 + 사용 후 캐시 반납.
    ///
    /// 실측(2026-09-12): mmap 폴트 경로는 20-60 MB/s(페이지 폴트당 4 KB),
    /// 캐시 적중 시 4-24 GB/s. llama도 같은 문제를 pread 스테이징 패치로
    /// 해결했다(83 GB/91 s). 여기서는 (a) 다음 청크에 MADV_WILLNEED를 미리
    /// 걸어 폴트가 선반입된 페이지에 떨어지게 하고, (b) 사용한 청크는
    /// MADV_DONTNEED로 커널에 반납해 30 GiB 호스트 RAM의 캐시 압박을 없앤다.
    /// 반납해도 무게는 VRAM 사본이 정본이므로 재독은 없다.
    /// pread 스테이징 업로드 — 파일에서 직접 8 MiB 청크로 읽어 h2d.
    /// 반환 None = 이 포인터가 알려진 파트 밖(폴백 필요).
    /// 실측: mmap 폴트 20-180 MB/s vs 버퍼드 pread 1.2 GB/s (같은 파일).
    fn staged_upload(&self, dst: *mut u8, ptr: usize, len: usize) -> Option<Result<(), String>> {
        use std::os::unix::fs::FileExt;
        let src = self
            .sources
            .iter()
            .find(|s| ptr >= s.base && ptr.checked_add(len).map(|e| e <= s.base + s.len).unwrap_or(false))?;
        let mut off = (ptr - src.base) as u64;
        let result = (|| -> Result<(), String> {
            const CH: usize = 8 << 20;
            let mut stage = self.stage.lock().map_err(|e| e.to_string())?;
            if stage.len() < CH.min(len) {
                *stage = vec![0u8; CH.min(len)];
            }
            let mut done = 0usize;
            while done < len {
                let n = CH.min(len - done);
                src.file.read_exact_at(&mut stage[..n], off).map_err(|e| format!("pread {off}: {e}"))?;
                self.ctx.h2d(unsafe { dst.add(done) }, &stage[..n])?;
                done += n;
                off += n as u64;
            }
            Ok(())
        })();
        Some(result)
    }

    fn upload_pipelined(&self, dst: *mut u8, data: &[u8]) -> Result<(), String> {
        const CH: usize = 8 << 20;
        let base = data.as_ptr() as usize;
        let n = data.len();
        if base % 4096 != 0 || n < (4 << 20) {
            Self::advise(base & !4095, ((base & 4095) + n + 4095) & !4095, libc::MADV_WILLNEED);
            return self.ctx.h2d(dst, data);
        }
        let mut off = 0usize;
        Self::advise(base, CH.min(n), libc::MADV_WILLNEED | libc::MADV_SEQUENTIAL);
        while off < n {
            let sz = CH.min(n - off);
            if off + sz < n {
                Self::advise(base + off + sz, CH.min(n - off - sz), libc::MADV_WILLNEED | libc::MADV_SEQUENTIAL);
            }
            self.ctx.h2d(unsafe { dst.add(off) }, &data[off..off + sz])?;
            Self::advise(base + off, sz, libc::MADV_DONTNEED);
            off += sz;
        }
        Ok(())
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
        let t0 = std::time::Instant::now();
        if let Some(r) = self.staged_upload(ptr, w.data.as_ptr() as usize, w.data.len()) {
            r?;
        } else {
            self.upload_pipelined(ptr, w.data)?;
        }
        self.wbytes
            .fetch_add(w.data.len(), std::sync::atomic::Ordering::Relaxed);
        if std::env::var_os("LLM170_Q4ACC_STATS").is_some() && w.data.len() >= (1 << 20) {
            let total = self.wbytes.load(std::sync::atomic::Ordering::Relaxed);
            eprintln!(
                "# q4acc: 업로드 {:.1} MiB ({:.1} ms → {:.0} MB/s, 누적 {:.2} GiB)",
                w.data.len() as f64 / (1u64 << 20) as f64,
                t0.elapsed().as_secs_f64() * 1e3,
                w.data.len() as f64 / t0.elapsed().as_secs_f64() / 1e6,
                total as f64 / (1u64 << 30) as f64
            );
        }
        self.weights
            .lock()
            .map_err(|e| e.to_string())?
            .insert(key, (ptr, is_f32));
        Ok((ptr, is_f32))
    }

    // ─── 프레임(활성화 상주) 지원 — plans/64 P1 ───

    fn t_cur(&self) -> usize {
        self.cur_t.load(std::sync::atomic::Ordering::Relaxed).max(1)
    }

    fn fptr(&self, h: u64) -> Result<*mut u8, String> {
        let v = self.frames.lock().map_err(|e| e.to_string())?;
        v.get((h.checked_sub(1).ok_or("frame 핸들 0")?) as usize)
            .map(|(p, _)| *p)
            .ok_or_else(|| format!("frame 핸들 없음: {h}"))
    }

    fn flen(&self, h: u64) -> Result<usize, String> {
        let v = self.frames.lock().map_err(|e| e.to_string())?;
        v.get((h.checked_sub(1).ok_or("frame 핸들 0")?) as usize)
            .map(|(_, l)| *l)
            .ok_or_else(|| format!("frame 핸들 없음: {h}"))
    }

    /// 프레임 활성 q8 준비 — x(프레임 f32) → xq 스크래치. (xq, xq_w)
    fn frame_quant(&self, x: *mut u8, n_in: usize, t: usize) -> Result<(*mut u8, usize), String> {
        let xq_w = xq_words(n_in);
        let buf = {
            let mut b = self.fxq.lock().map_err(|e| e.to_string())?;
            b.ensure(&self.ctx, t * xq_w * 4)?
        };
        self.ctx.quant_q8_b(x, buf, n_in, xq_w, t)?;
        Ok((buf, xq_w))
    }

    /// 프레임 GEMM 1건 — x는 프레임 f32, 무게는 mmap 참조(업로드 캐시).
    fn frame_gemm(&self, x: *mut u8, w: &llm170_core::matmul::Weight<'_>, out: *mut u8, t: usize) -> Result<(), String> {
        let n_in = w.n_in as usize;
        let n_out = w.n_out as usize;
        let (wd, f32w) = self.dev_weight(w)?;
        if f32w {
            return self.launch_gemm_f32(x, wd, n_in, n_out, t, out);
        }
        let (xq, xq_w) = self.frame_quant(x, n_in, t)?;
        self.launch_gemm(ggml_id(w.ty), xq, wd, n_in, n_out, xq_w, t, out)
    }

    /// 프레임 op 런치 헬퍼 — gx/gy/gz + 32/64/128/256 스레드.
    fn kop(
        &self,
        kern: &str,
        gx: u32,
        gy: u32,
        gz: u32,
        block: u32,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<(), String> {
        self.ctx.launch3(kern, gx, gy, gz, block, args)
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
        // t≥16: MMQ 타일 우선 — 가중치 1회 독서 + 토큰 타일 상각(raw 디코더
        // mm_b와 동일 게이트). 타일 커널이 없는 타입은 GEMV 폴백.
        // t≥16: MMQ 타일 우선 — 단 **128토큰 이하로 쪼개서** 호출한다.
        // j128 CO는 gz>1(다중 토큰 사분면)일 때 n_in=6144 형상에서 폴트한다
        // (2026-09-12 실측: t=129 폴트, t=128 정상, GEMV 경로는 비트 동일).
        if t >= 16 && std::env::var_os("LLM170_Q4_NO_TILE").is_none() {
            let mut ok = true;
            for c in 0..t.div_ceil(128) {
                let t0 = c * 128;
                let tc = 128.min(t - t0);
                let xsrc = unsafe { xq.add(t0 * xq_w * 4) };
                let osrc = unsafe { out.add(t0 * n_out * 4) };
                if self
                    .ctx
                    .gemm_tile(xsrc, w, self.ktab2, ty, n_in, n_out, xq_w, tc, osrc)
                    .is_err()
                {
                    ok = false;
                    break;
                }
            }
            if ok {
                return Ok(());
            }
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
    /// 활성 준비 — 업로드(+q8 양자화) 1회. 그룹/전문가 호출이 공유한다.
    /// 반환: (xf 포인터, xq 포인터(양자화 전이면 null), xq_w, t). w_f32면 xf를 쓴다.
    fn prepare_x(
        &self,
        xs: &[Vec<f32>],
        n_in: usize,
        w_f32: bool,
    ) -> Result<(*mut u8, *mut u8, usize, usize), String> {
        let t = xs.len();
        let mut xflat = Vec::with_capacity(t * n_in);
        for row in xs {
            if row.len() != n_in {
                return Err(format!("matmul: x({}) != n_in({n_in})", row.len()));
            }
            xflat.extend_from_slice(row);
        }
        let xdev = {
            let mut xb = self.xf.lock().map_err(|e| e.to_string())?;
            xb.ensure(&self.ctx, t * n_in * 4)?
        };
        self.ctx.h2d(xdev, bytemuck::cast_slice(&xflat))?;
        if w_f32 {
            return Ok((xdev, std::ptr::null_mut(), 0, t));
        }
        let xq_w = xq_words(n_in);
        let xq_buf = {
            let mut xb = self.xq.lock().map_err(|e| e.to_string())?;
            xb.ensure(&self.ctx, t * xq_w * 4)?
        };
        self.ctx.quant_q8_b(xdev, xq_buf, n_in, xq_w, t)?;
        Ok((xdev, xq_buf, xq_w, t))
    }

    /// 준비된 활성으로 1회 런치 + 판독.
    fn run_prepared(
        &self,
        xf: *mut u8,
        xq: *mut u8,
        xq_w: usize,
        t: usize,
        w: &llm170_core::matmul::Weight<'_>,
        w_off_bytes: usize,
        outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        let n_in = w.n_in as usize;
        let n_out = w.n_out as usize;
        let tt = std::env::var_os("LLM170_Q4ACC_TIME").is_some();
        let t_up = std::time::Instant::now();
        let (w_dev, w_f32) = self.dev_weight(w)?;
        let up_ns = t_up.elapsed().as_nanos() as u64;
        let w_slice = unsafe { w_dev.add(w_off_bytes) };
        let ydev = {
            let mut yb = self.yf.lock().map_err(|e| e.to_string())?;
            yb.ensure(&self.ctx, t * n_out * 4)?
        };
        let mut yflat = vec![0.0f32; t * n_out];
        let t_k = std::time::Instant::now();
        if w_f32 {
            self.launch_gemm_f32(xf, w_slice, n_in, n_out, t, ydev)?;
        } else {
            self.launch_gemm(ggml_id(w.ty), xq, w_slice, n_in, n_out, xq_w, t, ydev)?;
        }
        let k_ns = t_k.elapsed().as_nanos() as u64;
        let t_d = std::time::Instant::now();
        self.ctx.d2h(bytemuck::cast_slice_mut(&mut yflat), ydev)?;
        let d_ns = t_d.elapsed().as_nanos() as u64;
        if tt {
            self.note(up_ns, 0, k_ns, d_ns);
        }
        for (o, v) in outs.iter_mut().zip(yflat.chunks_exact(n_out)) {
            o.copy_from_slice(v);
        }
        Ok(())
    }

    /// 배치 GEMM 본체 — xs [t][n_in] f32 → outs [t][n_out] f32.
    fn batch_into(
        &self,
        xs: &[Vec<f32>],
        outs: &mut [Vec<f32>],
        w: &llm170_core::matmul::Weight<'_>,
        w_off_bytes: usize,
    ) -> Result<(), String> {
        if xs.is_empty() {
            return Ok(());
        }
        let n_in = w.n_in as usize;
        // f32 계열은 양자화를 건너뛰므로 준비 단계가 w_f32를 알아야 한다.
        let w_f32 = matches!(w.ty, GgmlType::F32 | GgmlType::Bf16 | GgmlType::F16);
        let (xf, xq, xq_w, t) = self.prepare_x(xs, n_in, w_f32)?;
        self.run_prepared(xf, xq, xq_w, t, w, w_off_bytes, outs)
    }
}

impl llm170_core::matmul::FrameState for Q4Acc {
    fn frame_begin(&self, t: usize) {
        self.cur_t.store(t.max(1), std::sync::atomic::Ordering::Relaxed);
    }

    /// GDN AR (프레임) — qwen35 raw 디코더와 동일 커널(gdn_ar_w_swap).
    /// q는 호출부에서 1/√d 스케일이 끝난 상태 → 커널 scale=1.0.
    #[allow(clippy::too_many_arguments)]
    fn frame_gdn_ar(
        &self,
        q_scaled: u64,
        k: u64,
        v: u64,
        beta_ge: u64,
        states: u64,
        out: u64,
        n_seqs: usize,
        h_k: usize,
        h_v: usize,
        d: usize,
    ) -> Result<(), String> {
        if n_seqs != 1 {
            return Err("q4acc: frame_gdn_ar np 미지원".into());
        }
        let (mut sp, mut qp, mut kp, mut vp, mut bp, mut op_) = (
            self.fptr(states)?,
            self.fptr(q_scaled)?,
            self.fptr(k)?,
            self.fptr(v)?,
            self.fptr(beta_ge)?,
            self.fptr(out)?,
        );
        let mut dd = d as i32;
        let mut ks = (h_k * d) as i32;
        let mut vs = (h_v * d) as i32;
        let mut hv = h_v as i32;
        let mut hk = h_k as i32;
        let mut sc = 1.0f32;
        // t토큰 순차 재귀 — 커널 내부 ti 루프가 상태를 이어간다(1런치).
        let mut tt = self.t_cur() as i32;
        if std::env::var_os("LLM170_Q4_DBG").is_some() {
            eprintln!(
                "# ar-args s={:?} q={:?} k={:?} v={:?} bg={:?} out={:?} d={dd} ks={ks} vs={vs} hv={hv} hk={hk}",
                sp as usize, qp as usize, kp as usize, vp as usize, bp as usize, op_ as usize
            );
        }
        self.ctx.launch3(
            "q4_gdn_ar_w",
            d as u32,
            h_v as u32,
            1,
            32,
            &mut cargs!(&mut sp, &mut qp, &mut kp, &mut vp, &mut bp, &mut op_, &mut dd, &mut ks, &mut vs, &mut hv, &mut hk, &mut sc, &mut tt),
        )
    }

    fn frame_moe_gather(
        &self,
        mix: u64,
        xsel: u64,
        n: usize,
        k_sel: usize,
        t: usize,
    ) -> Result<(), String> {
        let (mp, xs) = (self.fptr(mix)?, self.fptr(xsel)?);
        let total = (t * k_sel * n) as u32;
        let (mut a, mut b) = (mp, xs);
        let (mut nn, mut ks, mut tt) = (n as i32, k_sel as i32, t as i32);
        self.kop("q4_moe_gather", total.div_ceil(128), 1, 1, 128, &mut cargs!(&mut a, &mut b, &mut nn, &mut ks, &mut tt))
    }

    fn frame_moe_scatter(
        &self,
        ys: u64,
        wt: u64,
        out: u64,
        k_sel: usize,
        n: usize,
        t: usize,
    ) -> Result<(), String> {
        let (yp, wp, op_) = (self.fptr(ys)?, self.fptr(wt)?, self.fptr(out)?);
        let (mut a, mut b, mut c) = (yp, wp, op_);
        let (mut ks, mut nn, mut tt) = (k_sel as i32, n as i32, t as i32);
        self.kop("q4_moe_scatter", ((t * n) as u32).div_ceil(128), 1, 1, 128, &mut cargs!(&mut a, &mut b, &mut c, &mut ks, &mut nn, &mut tt))
    }

    /// MoE ids 구동 전문가 GEMM — 스택 + ids(프레임 상주). ids는 행당 u32.
    /// ids 40바이트 판독 후 전문가 연속 그룹으로 기존 커널을 런치한다(그룹당
    /// 1런치, 스택은 1회 업로드). ids-aware 단일 런치는 후속 최적화.
    fn frame_moe_gemm(
        &self,
        x: u64,
        ws: &llm170_core::matmul::Weight<'_>,
        ids: u64,
        out: u64,
        n_expert_stack: usize,
        k_sel: usize,
    ) -> Result<(), String> {
        let n_in = ws.n_in as usize;
        let n_out = ws.n_out as usize / n_expert_stack.max(1);
        // 행 수 = t·k_sel — 버퍼는 t_max 크기라 길이에서 유도할 수 없다.
        let rows = self.t_cur() * k_sel.max(1);
        let mut idv = vec![0u32; rows];
        let idp = self.fptr(ids)?;
        self.ctx.d2h(bytemuck::cast_slice_mut(&mut idv), idp)?;
        let xp = self.fptr(x)?;
        let op_ = self.fptr(out)?;
        let (wd, f32w) = self.dev_weight(ws)?;
        let per_expert = ws.data.len() / n_expert_stack.max(1);
        let (xq, xq_w) = if f32w {
            (std::ptr::null_mut(), 0usize)
        } else {
            self.frame_quant(xp, n_in, rows)?
        };
        let mut i = 0usize;
        while i < rows {
            let e = idv[i];
            let mut j = i + 1;
            while j < rows && idv[j] == e {
                j += 1;
            }
            let r = j - i;
            let xsrc = if f32w {
                unsafe { xp.add(i * n_in * 4) }
            } else {
                unsafe { xq.add(i * xq_w * 4) }
            };
            let wsrc = unsafe { wd.add(e as usize * per_expert) };
            let dst = unsafe { op_.add(i * n_out * 4) };
            if f32w {
                self.launch_gemm_f32(xsrc, wsrc, n_in, n_out, r, dst)?;
            } else {
                self.launch_gemm(ggml_id(ws.ty), xsrc, wsrc, n_in, n_out, xq_w, r, dst)?;
            }
            i = j;
        }
        Ok(())
    }
}

impl Q4Acc {
    /// q4_qsa_attn 커널 런치 본체 — 가드 없음(격리 프로브·진단 전용).
    #[allow(clippy::too_many_arguments)]
    pub fn qsa_attn_raw(
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
        if std::env::var_os("LLM170_Q4_DBG").is_some() {
            let bad = out.iter().filter(|v| !v.is_finite()).count();
            let badq = q.iter().filter(|v| !v.is_finite()).count();
            eprintln!("# qsa_attn t={t} n_past={n_past}: out 비유한={bad}/{} q 비유한={badq}", out.len());
        }
        Ok(out)
    }
}

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
        // 동일 입력 — 업로드·양자화 1회를 그룹 전체가 공유한다(값 경로에서
        // 왕복이 스텝 비용의 대부분이라 그룹 호출당 3→1로 줄인다).
        if ws.len() != outs.len() {
            return Err(format!("matmul_group: ws({}) != outs({})", ws.len(), outs.len()));
        }
        if ws.is_empty() || xs.is_empty() {
            return Ok(());
        }
        let n_in = ws[0].n_in as usize;
        let f32_family = |t: GgmlType| matches!(t, GgmlType::F32 | GgmlType::Bf16 | GgmlType::F16);
        let w_f32 = f32_family(ws[0].ty);
        // 타입 계열·n_in이 섞이면 준비를 공유할 수 없다 — 개별 경로로.
        if ws.iter().any(|w| w.n_in as usize != n_in || f32_family(w.ty) != w_f32) {
            for (w, o) in ws.iter().zip(outs.iter_mut()) {
                self.batch_into(xs, o, w, 0)?;
            }
            return Ok(());
        }
        let (xf, xq, xq_w, t) = self.prepare_x(xs, n_in, w_f32)?;
        for (w, o) in ws.iter().zip(outs.iter_mut()) {
            self.run_prepared(xf, xq, xq_w, t, w, 0, o)?;
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
        // 3D 전문가 스택은 n_out = 전문가수×전문가당 행으로 온다 — 런치·출력
        // 버퍼는 전문가당 행 기준이다(mul_mat_id와 동일한 해석).
        let n_out = ws.n_out as usize / n_expert_stack.max(1);
        let (w_dev, w_f32) = self.dev_weight(ws)?;
        // 활성 업로드 + 양자화 1회 (전문가 공통)
        let (xdev_f32, xq_buf, xq_w, t) = self.prepare_x(xs, n_in, w_f32)?;
        let mut yflat = vec![0.0f32; t * n_out];
        let ydev = {
            let mut yb = self.yf.lock().map_err(|e| e.to_string())?;
            yb.ensure(&self.ctx, t * n_out * 4)?
        };
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
        // t>128 가드 해제(2026-09-13): q4-qsa-check 프로브로 커널이 t=129·200·
        // 512(n_past 512)에서 CPU 미러와 일치함을 확인(최대 2e-5). 기존 가드는
        // 폴백을 유발했지만 호출자(qsa.rs)의 Err 경로가 CPU 재계산 없이 **빈
        // 어텐션 행**을 반환해 어텐션 자체가 누락됐다(양 경로 동일 → 자가일치
        // 검사가 통과). 유일한 강제 폴백: LLM170_QSA_CPU=1.
        if std::env::var_os("LLM170_QSA_CPU").is_some() {
            return Err(format!("q4acc: qsa_attention t={t} CPU 강제(LLM170_QSA_CPU)"));
        }
        self.qsa_attn_raw(q, ck, cv, mask, kq_scale, n_past, n_head, n_kv, hd, t)
    }

    // ─── 프레임(활성화 GPU 상주) — plans/64 P1 ───
    // 계약: core `qwen4exp/frame.rs`의 op 순서·산술 그대로. 프레임 경로는
    // 스텝당 동기를 ~14회로 줄인다(값 경로 ~1300회).

    /// 버퍼 할당 — `len`은 **원소 수**(f32 4바이트/u32 1워드). core frame.rs
    /// 규약(`a(k_len)`, `a(v.len())`)을 따른다.
    fn frame_alloc(&self, len: usize) -> Result<u64, String> {
        let p = self.ctx.alloc((len.max(4)) * 4)?;
        let mut v = self.frames.lock().map_err(|e| e.to_string())?;
        v.push((p, len));
        Ok(v.len() as u64)
    }

    fn frame_free(&self, _h: u64) -> Result<(), String> {
        // 해제 없음 (ADR-0014) — 풀은 영구.
        Ok(())
    }

    fn frame_write(&self, h: u64, data: &[f32]) -> Result<(), String> {
        let p = self.fptr(h)?;
        self.ctx.h2d(p, bytemuck::cast_slice(data))
    }

    fn frame_write_u32(&self, h: u64, data: &[u32]) -> Result<(), String> {
        let p = self.fptr(h)?;
        self.ctx.h2d(p, bytemuck::cast_slice(data))
    }

    fn frame_read(&self, h: u64, out: &mut [f32]) -> Result<(), String> {
        let p = self.fptr(h)?;
        // 동기 hipMemcpy — 공유 핀 스테이징(d2h 헬퍼)의 재사용 상태에 의존하지
        // 않는다. 프레임 판독은 스텝당 몇 회뿐이라 동기 경로 비용이 무의미하다.
        unsafe {
            ck(
                hip::hipMemcpy(
                    out.as_mut_ptr() as *mut std::ffi::c_void,
                    p as *const std::ffi::c_void,
                    out.len() * 4,
                    hip::hipMemcpyKind_hipMemcpyDeviceToHost,
                ),
                "frame_read",
            )
        }
    }

    fn frame_mm(&self, x: u64, w: &llm170_core::matmul::Weight<'_>, out: u64, t: usize) -> Result<(), String> {
        let (xp, op) = (self.fptr(x)?, self.fptr(out)?);
        self.frame_gemm(xp, w, op, t)
    }

    fn frame_mm_group(&self, x: u64, ws: &[llm170_core::matmul::Weight<'_>], outs: &[u64], t: usize) -> Result<(), String> {
        if ws.len() != outs.len() {
            return Err(format!("frame_mm_group: ws({}) != outs({})", ws.len(), outs.len()));
        }
        let xp = self.fptr(x)?;
        // 동일 입력 — 양자화 1회 공유 (f32 계열이 섞이면 개별).
        let f32_family = |ty: GgmlType| matches!(ty, GgmlType::F32 | GgmlType::Bf16 | GgmlType::F16);
        let f32w = f32_family(ws[0].ty);
        if ws.iter().all(|w| w.n_in == ws[0].n_in && f32_family(w.ty) == f32w) && !f32w {
            let (xq, xq_w) = self.frame_quant(xp, ws[0].n_in as usize, t)?;
            for (w, o) in ws.iter().zip(outs) {
                let (wd, _) = self.dev_weight(w)?;
                let op = self.fptr(*o)?;
                self.launch_gemm(ggml_id(w.ty), xq, wd, w.n_in as usize, w.n_out as usize, xq_w, t, op)?;
            }
            return Ok(());
        }
        for (w, o) in ws.iter().zip(outs) {
            let op = self.fptr(*o)?;
            self.frame_gemm(xp, w, op, t)?;
        }
        Ok(())
    }

    /// 상주 elementwise/RoPE/인덱서 연산 — qwen4exp 프레임이 쓰는 변형만 구현.
    fn frame_op(&self, op: &llm170_core::matmul::FrameOp) -> Result<(), String> {
        use llm170_core::matmul::FrameOp as O;
        match *op {
            O::SiluDiv { t, div, n } => {
                let mut p = self.fptr(t)?;
                let mut d = div;
                let mut nn = n as i32;
                self.kop("q4_silu_div", (n as u32).div_ceil(128), 1, 1, 128, &mut cargs!(&mut p, &mut d, &mut nn))
            }
            O::SiluMul { g, u, out, n } => {
                let (mut gp, mut up, mut op) = (self.fptr(g)?, self.fptr(u)?, self.fptr(out)?);
                let mut nn = n as i32;
                self.kop("silu_mul", (n as u32).div_ceil(128), 1, 1, 128, &mut cargs!(&mut gp, &mut up, &mut op, &mut nn))
            }
            O::Sigmoid { t, n } => {
                let mut p = self.fptr(t)?;
                let mut nn = n as i32;
                self.kop("q4_sigmoid", (n as u32).div_ceil(128), 1, 1, 128, &mut cargs!(&mut p, &mut nn))
            }
            O::RmsRows { x, w, out, eps, n, w_reps } => {
                let (xp, wp) = (self.fptr(x)?, self.fptr(w)?);
                let rows = w_reps * self.t_cur();
                let part = {
                    let mut b = self.fpart.lock().map_err(|e| e.to_string())?;
                    b.ensure(&self.ctx, rows * 32 * 8)?
                };
                {
                    let mut xa = xp;
                    let mut pa = part;
                    let mut nn = n as i32;
                    self.kop("rms_part", rows as u32, 1, 1, 32, &mut cargs!(&mut xa, &mut pa, &mut nn))?;
                }
                let mut xa = xp;
                let mut wa = wp;
                let mut pa = part;
                let mut op_ = self.fptr(out)?;
                let mut e = eps;
                let mut nn = n as i32;
                let mut rr = w_reps as i32;
                // 256스레드 = 8 그룹 × 32레인 (raw 디코더와 동일 기하 — 128로
                // 줄이면 행 절반이 미기록)
                self.kop("rms_finish", rows as u32, 1, 1, 256, &mut cargs!(&mut xa, &mut wa, &mut pa, &mut op_, &mut e, &mut nn, &mut rr))
            }
            O::NormGated { o, z, w, out, eps, d, n_h } => {
                let mut op_ = self.fptr(o)?;
                let mut zp = self.fptr(z)?;
                let mut wp = self.fptr(w)?;
                let mut outp = self.fptr(out)?;
                let mut e = eps;
                let mut dd = d as i32;
                let mut nh = n_h as i32;
                let rows = n_h * self.t_cur();
                self.kop("q4_norm_gated_sig", n_h as u32, (rows / n_h.max(1)) as u32, 1, 32, &mut cargs!(&mut op_, &mut zp, &mut wp, &mut outp, &mut e, &mut dd, &mut nh))
            }
            O::L2Rows { x, eps, d } => {
                let mut xp = self.fptr(x)?;
                let mut e = eps;
                let mut dd = d as i32;
                let rows = (d * 0 + self.flen(x)? / d).max(1) as u32;
                self.kop("q4_l2_rows", rows, 1, 1, 32, &mut cargs!(&mut xp, &mut e, &mut dd))
            }
            O::Scale { t, s, n } => {
                let mut p = self.fptr(t)?;
                let mut ss = s;
                let mut nn = n as i32;
                self.kop("q4_scale", (n as u32).div_ceil(128), 1, 1, 128, &mut cargs!(&mut p, &mut ss, &mut nn))
            }
            O::CopyRows { src, dst, src_off, dst_off, n } => {
                let (mut sp, mut dp) = (self.fptr(src)?, self.fptr(dst)?);
                let (mut so, mut dfo) = (src_off as i32, dst_off as i32);
                let mut nn = n as i32;
                self.kop("copy_rows", (n as u32).div_ceil(128), 1, 1, 128, &mut cargs!(&mut sp, &mut dp, &mut so, &mut dfo, &mut nn))
            }
            O::HcGateMean { xn, gate, out, hc, n } => {
                let (mut xp, mut gp, mut op_) = (self.fptr(xn)?, self.fptr(gate)?, self.fptr(out)?);
                let total = n * self.t_cur();
                let mut h = hc as i32;
                let mut nn = n as i32;
                let mut tt = total as i32;
                self.kop("q4_hc_gate_mean", (total as u32).div_ceil(128), 1, 1, 128, &mut cargs!(&mut xp, &mut gp, &mut op_, &mut h, &mut nn, &mut tt))
            }
            O::HcCombine { res, out, inj, hc, n, total: _ } => {
                // 커널은 (토큰,차원)당 1스레드 — op의 total(=hc·n·t)을 범위로 쓰면
                // hc배만큼 범위 밖을 쓴다(실측: hc>1에서 폴트). n·t를 쓴다.
                let (mut rp, mut op_, mut ip) = (self.fptr(res)?, self.fptr(out)?, self.fptr(inj)?);
                let tn = n * self.t_cur();
                let mut h = hc as i32;
                let mut nn = n as i32;
                let mut tt = tn as i32;
                self.kop("q4_hc_combine", (tn as u32).div_ceil(128), 1, 1, 128, &mut cargs!(&mut rp, &mut op_, &mut ip, &mut h, &mut nn, &mut tt))
            }
            O::Split3 { src, d0, d1, d2, n0, n1, n2 } => {
                let (mut sp, mut a0, mut a1, mut a2) = (
                    self.fptr(src)?, self.fptr(d0)?, self.fptr(d1)?, self.fptr(d2)?,
                );
                let (mut x0, mut x1, mut x2) = (n0 as i32, n1 as i32, n2 as i32);
                let total = ((n0 + n1 + n2) * self.t_cur()) as u32;
                self.kop("split3", total.div_ceil(128), 1, 1, 128, &mut cargs!(&mut sp, &mut a0, &mut a1, &mut a2, &mut x0, &mut x1, &mut x2))
            }
            O::GdnBetaG { b, a, dtb, sa, bg, n_h } => {
                let (mut bp, mut ap, mut dp, mut sp, mut gp) = (
                    self.fptr(b)?, self.fptr(a)?, self.fptr(dtb)?, self.fptr(sa)?, self.fptr(bg)?,
                );
                let mut nh = n_h as i32;
                // dt_rank = n_h / t (n_h = dt_rank·t) — t>1에서 n_h를 dt_rank로
                // 넘기면 dtb/sa(길이 dt_rank)를 넘겨 읽어 폴트 (실측 700).
                let mut dr = (n_h / self.t_cur().max(1)) as i32;
                self.kop("gdn_beta_g", (n_h as u32).div_ceil(128), 1, 1, 128, &mut cargs!(&mut bp, &mut ap, &mut dp, &mut sp, &mut gp, &mut nh, &mut dr))
            }
            O::GdnConv { qkv, cw, state, out, ch, k, t_len } => {
                let (qp, cp, stp, op_) = (
                    self.fptr(qkv)?, self.fptr(cw)?, self.fptr(state)?, self.fptr(out)?,
                );
                if t_len == 1 {
                    let (mut q, mut c, mut s_, mut o_) = (qp, cp, stp, op_);
                    let mut chh = ch as i32;
                    let mut kk = k as i32;
                    return self.kop("gdn_conv", (ch as u32).div_ceil(64), 1, 1, 64, &mut cargs!(&mut q, &mut c, &mut s_, &mut o_, &mut chh, &mut kk));
                }
                if t_len >= k - 1 {
                    // 완전 병렬 청크판 (전제 t ≥ k-1) + 링 상태 갱신은 별도 커널
                    // (conv_t2는 상태를 갱신하지 않는다 — raw 디코더도 2런치)
                    {
                        let (mut q, mut c, mut s_, mut o_) = (qp, cp, stp, op_);
                        let mut chh = ch as i32;
                        let mut kk = k as i32;
                        let mut tt = t_len as i32;
                        self.kop("gdn_conv_t2", (ch as u32).div_ceil(64), t_len as u32, 1, 64, &mut cargs!(&mut q, &mut c, &mut s_, &mut o_, &mut chh, &mut kk, &mut tt))?;
                    }
                    let (mut q2, mut s2) = (qp, stp);
                    let mut ch2 = ch as i32;
                    let mut k2 = k as i32;
                    let mut t2 = t_len as i32;
                    return self.kop("gdn_conv_state", (k - 1) as u32, (ch as u32).div_ceil(64), 1, 64, &mut cargs!(&mut q2, &mut s2, &mut ch2, &mut k2, &mut t2));
                }
                // 짧은 꼬리(t < k-1): 토큰별 순차 (t=1 커널 반복, 포인터 전진)
                for ti in 0..t_len {
                    let (mut q, mut c, mut s_, mut o_) = (
                        unsafe { qp.add(ti * ch * 4) },
                        cp,
                        stp,
                        unsafe { op_.add(ti * ch * 4) },
                    );
                    let mut chh = ch as i32;
                    let mut kk = k as i32;
                    self.kop("gdn_conv", (ch as u32).div_ceil(64), 1, 1, 64, &mut cargs!(&mut q, &mut c, &mut s_, &mut o_, &mut chh, &mut kk))?;
                }
                Ok(())
            }
            O::MoeTop10 { route, ids, wt, n_exp, k_sel } => {
                let (mut rp, mut ip, mut wp) = (self.fptr(route)?, self.fptr(ids)?, self.fptr(wt)?);
                let mut ne = n_exp as i32;
                let mut ks = k_sel as i32;
                let t = self.t_cur();
                self.kop("q4_moe_top10", t as u32, 1, 1, 1, &mut cargs!(&mut rp, &mut ip, &mut wp, &mut ne, &mut ks))
            }
            O::MoeWeightedSum { ys, wt, out, k, n } => {
                let (mut yp, mut wp, mut op_) = (self.fptr(ys)?, self.fptr(wt)?, self.fptr(out)?);
                let mut kk = k as i32;
                let mut nn = (n * self.t_cur()) as i32;
                self.kop("q4_moe_weighted_sum", ((n * self.t_cur()) as u32).div_ceil(128), 1, 1, 128, &mut cargs!(&mut yp, &mut wp, &mut op_, &mut kk, &mut nn))
            }
            O::AxpyScaled { y, x, s, n } => {
                let (mut yp, mut xp, mut sp) = (self.fptr(y)?, self.fptr(x)?, self.fptr(s)?);
                let mut nn = n as i32;
                let t = self.t_cur();
                if t <= 1 {
                    self.kop("axpy_scaled", (n as u32).div_ceil(128), 1, 1, 128, &mut cargs!(&mut yp, &mut xp, &mut sp, &mut nn))
                } else {
                    // 토큰 배치: s[t] — per = 토큰당 원소 수
                    let mut pp = (n / t) as i32;
                    self.kop("q4_axpy_scaled_t", (n as u32).div_ceil(128), 1, 1, 128, &mut cargs!(&mut yp, &mut xp, &mut sp, &mut nn, &mut pp))
                }
            }
            ref other => Err(format!("q4acc: 프레임 op 미지원 {other:?}")),
        }
    }
}

/// 프레임 op 인자 벡터 — 로컬 변수의 주소를 c_void로.
macro_rules! cargs {
    ($($e:expr),+ $(,)?) => {{
        let mut v: Vec<*mut std::ffi::c_void> = Vec::new();
        $( v.push($e as *mut _ as *mut std::ffi::c_void); )+
        v
    }};
}
use cargs;

/// Engine4에 주입할 가속기 생성 — 실패 시 호출부가 CPU로 폴백(경고).
pub fn new_acc() -> Result<std::sync::Arc<dyn llm170_core::matmul::Accelerator>, String> {
    new_acc_with_sources(Vec::new())
}

/// 파트 소스 지정판 — 서버 배선이 `Model4::part_sources()`를 넘긴다.
pub fn new_acc_with_sources(
    parts: Vec<(usize, usize, std::path::PathBuf)>,
) -> Result<std::sync::Arc<dyn llm170_core::matmul::Accelerator>, String> {
    let a = Q4Acc::new_with_sources(parts)?;
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

/// `q4-ar-check` — 프레임 AR 커널(q4_gdn_ar_w) ↔ core `gdn_ar_batch` 대조.
/// 합성 입력(결정적 LCG)으로 수치 계약을 직접 확인한다.
#[allow(clippy::many_single_char_names)]
pub fn ar_check() -> Result<String, String> {
    ar_check_t(1)
}

/// t토큰 AR 대조 — t>1은 커널 내부 순차 재귀 경로.
pub fn ar_check_t(t: usize) -> Result<String, String> {
    use llm170_core::matmul::{Accelerator, FrameState};
    let (n_group, dt_rank, d) = (16usize, 48usize, 128usize);
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
    };
    let k_len = n_group * d;
    let v_len = dt_rank * d;
    let q: Vec<f32> = (0..k_len * t).map(|_| lcg()).collect();
    let k: Vec<f32> = (0..k_len * t).map(|_| lcg()).collect();
    let v: Vec<f32> = (0..v_len * t).map(|_| lcg()).collect();
    let beta: Vec<f32> = (0..dt_rank * t).map(|_| lcg()).collect();
    let g: Vec<f32> = (0..dt_rank * t).map(|_| lcg()).collect();
    let st0: Vec<f32> = (0..dt_rank * d * d).map(|_| lcg() * 0.1).collect();

    // CPU 기준 (프레임 규약: β = σ(b), g는 원값 — AR이 exp)
    let beta_sig: Vec<f32> = beta
        .iter()
        .map(|&b| 1.0 / (1.0 + llm170_core::ops::exp_cr(-b)))
        .collect();
    let mut st_cpu = st0.clone();
    let mut o_cpu = vec![0.0f32; v_len * t];
    llm170_core::gdn::gdn_ar_batch(
        &q, &k, &v, &beta_sig, &g, &mut st_cpu, &mut o_cpu, t, n_group, dt_rank,
    );

    // GPU 프레임 (q는 1/√d 선스케일, bg는 인터리브 [σ(b), e^g])
    let acc = Q4Acc::new()?;
    let hq = acc.frame_alloc(k_len * t)?;
    let hk = acc.frame_alloc(k_len * t)?;
    let hv = acc.frame_alloc(v_len * t)?;
    let hbg = acc.frame_alloc(dt_rank * 2 * t)?;
    let hst = acc.frame_alloc(st0.len())?;
    let ho = acc.frame_alloc(v_len * t)?;
    let mut bg = vec![0.0f32; dt_rank * 2 * t];
    for h in 0..dt_rank * t {
        bg[h * 2] = beta_sig[h];
        bg[h * 2 + 1] = llm170_core::ops::exp_cr(g[h]);
    }
    let qs: Vec<f32> = q.iter().map(|x| x / (d as f32).sqrt()).collect();
    acc.frame_write(hq, &qs)?;
    acc.frame_write(hk, &k)?;
    acc.frame_write(hv, &v)?;
    acc.frame_write(hbg, &bg)?;
    acc.frame_write(hst, &st0)?;
    acc.frame_gdn_ar(hq, hk, hv, hbg, hst, ho, 1, n_group, dt_rank, d)?;
    let mut o_gpu = vec![0.0f32; v_len * t];
    acc.frame_read(ho, &mut o_gpu)?;
    let mut st_gpu = vec![0.0f32; st0.len()];
    acc.frame_read(hst, &mut st_gpu)?;
    let rel = |a: &[f32], b: &[f32]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| ((x - y).abs() as f64) / (y.abs().max(1e-3) as f64))
            .fold(0.0f64, f64::max)
    };
    Ok(format!(
        "q4-ar-check t={t} n_group={n_group} dt_rank={dt_rank} d={d}: out rel={:.3e} (cpu[0]={:+.6} gpu[0]={:+.6}), state rel={:.3e}",
        rel(&o_gpu, &o_cpu),
        o_cpu[0],
        o_gpu[0],
        rel(&st_gpu, &st_cpu)
    ))
}

/// `q4-qsa-check [t] [n_past]` — q4_qsa_attn GPU ↔ CPU 미러(합성 Q/K/V·마스크).
/// t>128 결함(오답)의 원인을 좁히기 위한 격리 하네스.
pub fn qsa_check(t: usize, n_past: usize) -> Result<String, String> {
    use llm170_core::ops::exp_cr;
    let (n_head, n_kv, hd) = (24usize, 2usize, 256usize);
    let mut seed = 0x9e37_79b9u64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
    };
    // 인과 + 블록 스파스 마스크: 위치 p는 (tok, p)가 허용될 때만 1.
    let total = n_past;
    let q: Vec<f32> = (0..t * n_head * 2 * hd).map(|_| lcg()).collect();
    let ck: Vec<f32> = (0..total * n_kv * hd).map(|_| lcg()).collect();
    let cv: Vec<f32> = (0..total * n_kv * hd).map(|_| lcg()).collect();
    let mut mask = vec![0u32; t * n_past];
    let base = n_past - t; // 이 배치 이전 위치 수
    for tok in 0..t {
        for p in 0..n_past {
            // 인과: p <= base+tok. 스파스: 4토큰 블록당 최근 2블록만 남기는 흉내.
            let causal = p <= base + tok;
            let blk = p / 4;
            let cur = (base + tok) / 4;
            let keep = blk + 2 > cur;
            mask[tok * n_past + p] = if causal && keep { 1 } else { 0 };
        }
    }
    let kq_scale = 1.0f32;
    let acc = Q4Acc::new()?;
    let gpu = acc.qsa_attn_raw(&q, &ck, &cv, &mask, kq_scale, n_past, n_head, n_kv, hd, t)?;
    // CPU 미러 (core stages::qsa_attention과 같은 산술 구조).
    let mut cpu = vec![0.0f32; t * n_head * hd];
    for tok in 0..t {
        for h in 0..n_head {
            let kvh = h / (n_head / n_kv);
            let qh = &q[(tok * n_head + h) * 2 * hd..(tok * n_head + h) * 2 * hd + hd];
            let gate = &q[(tok * n_head + h) * 2 * hd + hd..(tok * n_head + h) * 2 * hd + 2 * hd];
            let mut m = f32::NEG_INFINITY;
            let mut sc = vec![f32::NEG_INFINITY; n_past];
            for p in 0..n_past {
                if mask[tok * n_past + p] == 0 {
                    continue;
                }
                let k = &ck[p * n_kv * hd + kvh * hd..p * n_kv * hd + kvh * hd + hd];
                let mut s = 0.0f32;
                for i in 0..hd {
                    s += qh[i] * k[i];
                }
                s *= kq_scale;
                sc[p] = s;
                m = m.max(s);
            }
            let mut l = 0.0f32;
            let mut a = [0.0f32; 256];
            for p in 0..n_past {
                if sc[p] == f32::NEG_INFINITY {
                    continue;
                }
                let e = exp_cr(sc[p] - m);
                l += e;
                let v = &cv[p * n_kv * hd + kvh * hd..p * n_kv * hd + kvh * hd + hd];
                for i in 0..hd {
                    a[i] += e * v[i];
                }
            }
            for i in 0..hd {
                let o = if l > 0.0 { a[i] / l } else { 0.0 };
                cpu[(tok * n_head + h) * hd + i] = o * (1.0 / (1.0 + exp_cr(-gate[i])));
            }
        }
    }
    let mut nz = 0usize;
    let mut maxrel = 0.0f64;
    let mut nonfinite = 0usize;
    for (i, (&a, &b)) in gpu.iter().zip(&cpu).enumerate() {
        if !a.is_finite() {
            nonfinite += 1;
            continue;
        }
        let d = ((a - b).abs() as f64) / (b.abs().max(1e-3) as f64);
        if d > 1e-3 {
            nz += 1;
            if nz <= 3 {
                let tok = i / (n_head * hd);
                let h = (i / hd) % n_head;
                let dim = i % hd;
                eprintln!("# qsa diff #{nz} tok={tok} h={h} dim={dim} gpu={a} cpu={b}");
            }
        }
        maxrel = maxrel.max(d);
    }
    Ok(format!(
        "q4-qsa-check t={t} n_past={n_past}: nonfinite={nonfinite} mismatch={nz}/{} maxrel={maxrel:.3e}",
        gpu.len()
    ))
}

/// `q4-hc-check [t] [n] [hc]` — 프레임 HC op(HcGateMean/HcCombine) 격리 검증.
/// 합성 입력으로 GPU ↔ CPU 미러를 대조하고, 폴트 여부를 직접 보고한다.
pub fn hc_check(t: usize, n: usize, hc: usize) -> Result<String, String> {
    use llm170_core::matmul::{Accelerator, FrameOp, FrameState};
    let mut seed = 0x1234_5678u64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
    };
    let acc = Q4Acc::new()?;
    let hxn = acc.frame_alloc(t * hc * n)?;
    let hgate = acc.frame_alloc(t * hc * n)?;
    let hmix = acc.frame_alloc(t * n)?;
    let hres = acc.frame_alloc(t * hc * n)?;
    let hout = acc.frame_alloc(t * n)?;
    let hinj = acc.frame_alloc(t * hc)?;
    let xn: Vec<f32> = (0..t * hc * n).map(|_| lcg()).collect();
    let gate: Vec<f32> = (0..t * hc * n).map(|_| lcg()).collect();
    let res0: Vec<f32> = (0..t * hc * n).map(|_| lcg()).collect();
    let out: Vec<f32> = (0..t * n).map(|_| lcg()).collect();
    let inj: Vec<f32> = (0..t * hc).map(|_| lcg()).collect();
    acc.frame_write(hxn, &xn)?;
    acc.frame_write(hgate, &gate)?;
    acc.frame_write(hres, &res0)?;
    acc.frame_write(hout, &out)?;
    acc.frame_write(hinj, &inj)?;
    acc.frame_begin(t);
    acc.frame_op(&FrameOp::HcGateMean { xn: hxn, gate: hgate, out: hmix, hc, n })?;
    acc.frame_op(&FrameOp::HcCombine { res: hres, out: hout, inj: hinj, hc, n, total: hc * n * t })?;
    let mut mix_gpu = vec![0.0f32; t * n];
    acc.frame_read(hmix, &mut mix_gpu)?;
    let mut res_gpu = vec![0.0f32; t * hc * n];
    acc.frame_read(hres, &mut res_gpu)?;
    // CPU 미러
    let sig = |x: f32| 1.0f32 / (1.0 + llm170_core::ops::exp_cr(-x));
    let mut mix_cpu = vec![0.0f32; t * n];
    for ti in 0..t {
        for i in 0..n {
            let mut a = 0.0f32;
            for s in 0..hc {
                let k = ti * hc * n + s * n + i;
                a += xn[k] * sig(gate[k]);
            }
            mix_cpu[ti * n + i] = a / hc as f32;
        }
    }
    let mut res_cpu = res0.clone();
    for ti in 0..t {
        for i in 0..n {
            for s in 0..hc {
                res_cpu[ti * hc * n + s * n + i] += out[ti * n + i] * 2.0 * sig(inj[ti * hc + s] / hc as f32);
            }
        }
    }
    let rel = |a: &[f32], b: &[f32]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| ((x - y).abs() as f64) / (y.abs().max(1e-3) as f64))
            .fold(0.0f64, f64::max)
    };
    Ok(format!(
        "q4-hc-check t={t} n={n} hc={hc}: mix rel={:.3e} res rel={:.3e}",
        rel(&mix_gpu, &mix_cpu),
        rel(&res_gpu, &res_cpu)
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
