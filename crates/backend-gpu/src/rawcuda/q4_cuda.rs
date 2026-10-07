//! Q4(GGUF Q4_K / UD-Q4_K_XL) CUDA 모듈층 — MMQ GEMV·GEMM + 디양자화
//! (plans/124 G8, 2026-10-04).
//!
//! API 형상은 rawhip q4acc(value.rs launch_gemm → quant_q8 → gemv/tile)의
//! 미러. 3층 분리 원칙(plans/124 §5): 이 층은 가중치 상주 + 계산 API의
//! 단일 진실 — 검증 자산(GGUF 파서·오라클)은 q4_cuda_probe로 금지.
//! 수치 계약 원천은 **core ggml 미러**(crates/core/src/quant/)이며 커널
//! 산술 계약은 assets/exl3_q4.cu 헤더 참조(정합 판정은 값 maxdiff +
//! 비트 일치 — argmax 판정 금지, plans/124 §5).
//!
//! 단일 상주 원칙(2026-10-04 동결 사고): 이 모듈이 디바이스 컨텍스트를
//! 소유한다. Exl3CudaDecoder와 동시 상주는 검증층에서 금지(한 프로세스
//! 모델 1개).
//!
//! [독립 컴파일 계약] 이 파일은 scripts/cuda_probe_shim.rs가 rustc로 단독
//! 컴파일한다(전체 워크스페이스는 Windows에서 llm170-core mmap 결함으로
//! 불가 — G1 원장). 따라서 std 외 크레이트 의존 금지. exl3_cuda.rs의
//! 패턴(자산 리졸버·청크 h2d)을 새 코드로 미러 — 임포트하지 않는다
//! (사용자 지시 2026-10-04: 모듈별 독립 파일 구조).

use crate::rawcuda::ctx::CudaCtx;
use crate::rawcuda::ffi::CUdeviceptr;
use std::collections::HashMap;

/// Q4_K 블록 레이아웃 상수 — crates/gguf/src/types.rs block_info(Q4K)
/// = (256원소, 144B). UD-Q4_K_XL도 GGUF에는 표준 Q4_K 블록으로 저장
/// (assets/exl3_q4.cu 헤더 [UD-Q4_K_XL 형식 판독]).
pub const Q4K_BLCK: usize = 256;
pub const Q4K_BYTES: usize = 144;

/// 활성 q8 버퍼의 행 스트라이드(워드) — rawhip q4acc/mod.rs xq_words
/// (L284-286) 미러: [n/4 팩워드][n/32 d 비트][2·n/32 q16 합].
pub fn xq_words(n: usize) -> usize {
    n / 4 + n / 32 + n / 16
}

/// 상주 Q4_K 선형 — 행 우선 [n_out][n_in] 바이트(행 = n_in/256 블록 × 144B).
pub struct Q4Lin {
    pub n_in: usize,
    pub n_out: usize,
    pub w: CUdeviceptr,
}

/// Q4 CUDA 모듈 — 가중치 상주 + quant/GEMV/GEMM/dequant API.
pub struct Q4Cuda {
    cc: CudaCtx,
    lins: HashMap<String, Q4Lin>,
    dx: CUdeviceptr,
    x_cap: usize,
    dxq: CUdeviceptr,
    xq_cap: usize,
    dout: CUdeviceptr,
    out_cap: usize,
}

/// 대형 h2d 청크 상한(4MB — 페이지 미매핑 가드, G2 원장 패턴 미러).
const H2D_CHUNK: usize = 4 << 20;

impl Q4Cuda {
    /// exl3_q4.fatbin 자산 해석 — LLM170_CUDA_Q4_FATBIN_PATH 오버라이드 우선
    /// (자산 경로 오버라이드일 뿐 계산 경로 분기 아님).
    fn q4_fatbin_bytes() -> Result<Vec<u8>, String> {
        const ENV: &str = "LLM170_CUDA_Q4_FATBIN_PATH";
        const REL: &[&str] = &[
            "crates/backend-gpu/src/rawcuda/assets/exl3_q4.fatbin",
            "src/rawcuda/assets/exl3_q4.fatbin",
        ];
        if let Some(p) = std::env::var_os(ENV) {
            return std::fs::read(&p).map_err(|e| format!("{ENV}({p:?}) 읽기 실패: {e}"));
        }
        for r in REL {
            if let Ok(b) = std::fs::read(r) {
                return Ok(b);
            }
        }
        Err(format!(
            "exl3_q4.fatbin 없음 — scripts/build_cuda.bat 실행 또는 {ENV} 지정 (탐색: {REL:?})"
        ))
    }

    /// 컨텍스트 + Q4 커널 4종 로드(quant/dequant/gemv/gemm).
    pub fn new() -> Result<Self, String> {
        let image = Self::q4_fatbin_bytes()?;
        let mut cc = CudaCtx::new()?;
        let _g = cc.guard()?;
        cc.load_fatbin(
            "q4",
            &image,
            &[
                "q4_quant_q8",
                "q4_dequant_q4k",
                "q4_gemv_q4k",
                "q4_gemm_q4k_m",
            ],
        )?;
        Ok(Q4Cuda {
            cc,
            lins: HashMap::new(),
            dx: 0,
            x_cap: 0,
            dxq: 0,
            xq_cap: 0,
            dout: 0,
            out_cap: 0,
        })
    }

    pub fn device_name(&self) -> &str {
        &self.cc.device_name
    }

    /// 대형 h2d 청크 분할 업로드(G2 원장 패턴 미러 — 4MB 청크).
    fn h2d_chunked(cc: &CudaCtx, dst: CUdeviceptr, src: &[u8]) -> Result<(), String> {
        let mut off = 0usize;
        while off < src.len() {
            let hi = (off + H2D_CHUNK).min(src.len());
            // SAFETY: src는 호출자 소유 슬라이스 — 부분 슬라이스는 호출 내 유효.
            let part = unsafe { std::slice::from_raw_parts(src.as_ptr().add(off), hi - off) };
            cc.h2d(dst + off as u64, part)?;
            off = hi;
        }
        Ok(())
    }

    /// Q4_K 선형 등록(바이트 열 = n_out행 × n_in/256블록 × 144B, 행 우선).
    /// n_in은 256의 배수 계약(블록 경계 정렬 — GGUF ne[0] 규약 준수).
    pub fn add_q4k_bytes(
        &mut self,
        key: &str,
        bytes: &[u8],
        n_in: usize,
        n_out: usize,
    ) -> Result<(), String> {
        if n_in == 0 || !n_in.is_multiple_of(Q4K_BLCK) || n_out == 0 {
            return Err(format!(
                "q4 add: n_in={n_in} n_out={n_out} — n_in>0·256배수·n_out>0 계약 위반"
            ));
        }
        let expect = n_out * (n_in / Q4K_BLCK) * Q4K_BYTES;
        if bytes.len() != expect {
            return Err(format!(
                "q4 add: bytes={} != {} (n_in={n_in} n_out={n_out}) — 행 우선 배치 계약 위반",
                bytes.len(),
                expect
            ));
        }
        let w = self.cc.alloc(expect)?;
        Self::h2d_chunked(&self.cc, w, bytes)?;
        if let Some(old) = self.lins.insert(key.to_string(), Q4Lin { n_in, n_out, w }) {
            // SAFETY: old.w는 이 ctx의 alloc 산출물(이중 해제 금지 — 1회 교체).
            self.cc.free(old.w)?;
        }
        Ok(())
    }

    /// 등록 선형 형상 조회.
    pub fn lin_shape(&self, key: &str) -> Option<(usize, usize)> {
        self.lins.get(key).map(|l| (l.n_in, l.n_out))
    }

    /// 활성 f32 업로드 → q8 양자화(q4_quant_q8) → xq 디바이스 포인터.
    /// rows.len() = t·n_in 계약. 반환 포인터 수명은 다음 quant 호출 전까지.
    fn quant_rows(&mut self, rows: &[f32], n_in: usize, t: usize) -> Result<CUdeviceptr, String> {
        if rows.len() != t * n_in {
            return Err(format!(
                "q4 quant: rows.len={} != t·n_in={}",
                rows.len(),
                t * n_in
            ));
        }
        let nwords = xq_words(n_in);
        if self.dx == 0 || t * n_in > self.x_cap {
            if self.dx != 0 {
                // SAFETY: dx는 이전 alloc 산출물 — 재할당 전 1회 해제.
                self.cc.free(self.dx)?;
            }
            self.dx = self.cc.alloc(t * n_in * 4)?;
            self.x_cap = t * n_in;
        }
        if self.dxq == 0 || t * nwords > self.xq_cap {
            if self.dxq != 0 {
                // SAFETY: dxq는 이전 alloc 산출물 — 재할당 전 1회 해제.
                self.cc.free(self.dxq)?;
            }
            self.dxq = self.cc.alloc(t * nwords * 4)?;
            self.xq_cap = t * nwords;
        }
        // SAFETY: rows는 f32 슬라이스 — 바이트 뷰 변환(길이 일치).
        let xb = unsafe { std::slice::from_raw_parts(rows.as_ptr() as *const u8, rows.len() * 4) };
        Self::h2d_chunked(&self.cc, self.dx, xb)?;
        let f = self.cc.function("q4_quant_q8")?;
        let (mut a0, mut a1) = (self.dx, self.dxq);
        let (mut nn, mut nw) = (n_in as i32, nwords as i32);
        let mut args: [*mut std::ffi::c_void; 4] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut nn) as *mut _ as *mut _,
            (&mut nw) as *mut _ as *mut _,
        ];
        let nblk = n_in / 32;
        self.cc
            .launch(f, nblk.div_ceil(128) as u32, t as u32, 128, &mut args)?;
        Ok(self.dxq)
    }

    /// 출력 스크래치 확보(len f32) — 판독 전 ensure 후 d2h.
    fn ensure_out(&mut self, len: usize) -> Result<(), String> {
        if self.dout == 0 || len > self.out_cap {
            if self.dout != 0 {
                // SAFETY: dout은 이전 alloc 산출물 — 재할당 전 1회 해제.
                self.cc.free(self.dout)?;
            }
            self.dout = self.cc.alloc(len * 4)?;
            self.out_cap = len;
        }
        Ok(())
    }

    /// MMQ GEMV(t=1) — q4_gemv_q4k: quant 1행 → grid(1, n_out)·64스레드.
    /// 환원 순서 = core lane.rs 레인 미러(비트동일 기대 — 프로브 판정).
    pub fn gemv_host(&mut self, key: &str, x: &[f32]) -> Result<Vec<f32>, String> {
        let (n_in, n_out) = self
            .lin_shape(key)
            .ok_or_else(|| format!("q4 gemv: 미등록 선형 {key}"))?;
        if x.len() != n_in {
            return Err(format!("q4 gemv: x.len={} != n_in={n_in}", x.len()));
        }
        if n_out > 65535 {
            return Err(format!(
                "q4 gemv: n_out={n_out} — grid-y 상한 65535 초과(블록 분할 필요)"
            ));
        }
        let xq = self.quant_rows(x, n_in, 1)?;
        let f = self.cc.function("q4_gemv_q4k")?;
        let w = self.lins[key].w;
        self.ensure_out(n_out)?;
        let (mut a0, mut a1, mut op) = (xq, w, self.dout);
        let (mut ni, mut no, mut nw) = (n_in as i32, n_out as i32, xq_words(n_in) as i32);
        // 인자 순서 계약: 커널 시그니처 (xq, w, out, n_in, n_out, xq_w) —
        // out을 ni/no 뒤에 두면 포인터·정수가 교차해 ILLEGAL ADDRESS가
        // 난다(G8 디버그 원장 — gemm은 정순이었음).
        let mut args: [*mut std::ffi::c_void; 6] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut op) as *mut _ as *mut _,
            (&mut ni) as *mut _ as *mut _,
            (&mut no) as *mut _ as *mut _,
            (&mut nw) as *mut _ as *mut _,
        ];
        self.cc.launch(f, 1, n_out as u32, 64, &mut args)?;
        let mut ob = vec![0u8; n_out * 4];
        self.cc.d2h(&mut ob, self.dout)?;
        self.cc.sync()?;
        // SAFETY: d2h·sync 완료 후 재해석(길이·정렬 일치 — G2 판독 패턴).
        Ok(unsafe { std::slice::from_raw_parts(ob.as_ptr() as *const f32, n_out) }.to_vec())
    }

    /// MMQ GEMM 타일(T행) — q4_gemm_q4k_m: quant T행 → grid(ceil(n_out/16),
    /// ceil(t/16))·256스레드. 블록 순차 f32 = core dot_q4k_q8 순서(비트동일
    /// 기대 — 프로브 판정). out 배치 = [t][n_out] 행 우선.
    pub fn gemm_host(&mut self, key: &str, rows: &[f32]) -> Result<Vec<f32>, String> {
        let (n_in, n_out) = self
            .lin_shape(key)
            .ok_or_else(|| format!("q4 gemm: 미등록 선형 {key}"))?;
        if !rows.len().is_multiple_of(n_in) {
            return Err(format!(
                "q4 gemm: rows.len={} % n_in={n_in} != 0 — 행 배치 계약 위반",
                rows.len()
            ));
        }
        let t = rows.len() / n_in;
        if t == 0 || t > 65535 {
            return Err(format!(
                "q4 gemm: t={t} — (0, 65535] 도메인 계약(그리드 y 상한)"
            ));
        }
        let xq = self.quant_rows(rows, n_in, t)?;
        let f = self.cc.function("q4_gemm_q4k_m")?;
        let w = self.lins[key].w;
        self.ensure_out(t * n_out)?;
        let (mut a0, mut a1, mut op) = (xq, w, self.dout);
        let (mut ni, mut no, mut nw, mut tt) =
            (n_in as i32, n_out as i32, xq_words(n_in) as i32, t as i32);
        let mut args: [*mut std::ffi::c_void; 7] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut op) as *mut _ as *mut _,
            (&mut ni) as *mut _ as *mut _,
            (&mut no) as *mut _ as *mut _,
            (&mut nw) as *mut _ as *mut _,
            (&mut tt) as *mut _ as *mut _,
        ];
        self.cc.launch(
            f,
            n_out.div_ceil(16) as u32,
            t.div_ceil(16) as u32,
            256,
            &mut args,
        )?;
        let mut ob = vec![0u8; t * n_out * 4];
        self.cc.d2h(&mut ob, self.dout)?;
        self.cc.sync()?;
        // SAFETY: d2h·sync 완료 후 재해석(길이·정렬 일치 — G2 판독 패턴).
        Ok(unsafe { std::slice::from_raw_parts(ob.as_ptr() as *const f32, t * n_out) }.to_vec())
    }

    /// Q4_K 디양자화 — q4_dequant_q4k: 앞 n_rows행 × n_in f32 (행 우선).
    /// core deq.rs deq_q4_k와 비트동일 계약(-fmad=false 빌드).
    pub fn dequant_host(&mut self, key: &str, n_rows: usize) -> Result<Vec<f32>, String> {
        let (n_in, n_out) = self
            .lin_shape(key)
            .ok_or_else(|| format!("q4 dequant: 미등록 선형 {key}"))?;
        if n_rows == 0 || n_rows > n_out {
            return Err(format!(
                "q4 dequant: n_rows={n_rows} — (0, {n_out}] 도메인 계약 위반"
            ));
        }
        let nsuper = n_in / Q4K_BLCK;
        let f = self.cc.function("q4_dequant_q4k")?;
        let w = self.lins[key].w;
        self.ensure_out(n_rows * n_in)?;
        let (mut a0, mut a1) = (w, self.dout);
        let (mut ns, mut no) = (nsuper as i32, n_rows as i32);
        let mut args: [*mut std::ffi::c_void; 4] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut ns) as *mut _ as *mut _,
            (&mut no) as *mut _ as *mut _,
        ];
        self.cc
            .launch(f, n_rows as u32, nsuper as u32, 256, &mut args)?;
        let mut ob = vec![0u8; n_rows * n_in * 4];
        self.cc.d2h(&mut ob, self.dout)?;
        self.cc.sync()?;
        // SAFETY: d2h·sync 완료 후 재해석(길이·정렬 일치 — G2 판독 패턴).
        Ok(
            unsafe { std::slice::from_raw_parts(ob.as_ptr() as *const f32, n_rows * n_in) }
                .to_vec(),
        )
    }
}

impl Drop for Q4Cuda {
    fn drop(&mut self) {
        // SAFETY: 각 포인터는 이 ctx의 alloc 산출물이며 drop에서 1회 해제.
        let r = (|| {
            for (_, l) in self.lins.drain() {
                self.cc.free(l.w)?;
            }
            if self.dx != 0 {
                self.cc.free(self.dx)?;
            }
            if self.dxq != 0 {
                self.cc.free(self.dxq)?;
            }
            if self.dout != 0 {
                self.cc.free(self.dout)?;
            }
            Ok::<(), String>(())
        })();
        if let Err(e) = r {
            eprintln!("q4_cuda: drop 해제 실패: {e}");
        }
    }
}
