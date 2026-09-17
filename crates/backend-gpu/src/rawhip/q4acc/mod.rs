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

/// MoE 전문가 그룹화 캐시 — 같은 라우팅(ids)에 대한 gate/up/down 3개 투영이
/// 같은 순열을 쓴다. 게이트가 1회만 d2h·정렬·업로드하고 나머지는 재사용한다.
/// (실측: 호출마다 d2h+h2d 동기 → 층당 9회 → MoE 77ms/층, 청크의 59%.)
struct MoeGroup {
    generation: u64,
    rows: usize,
    perm_d: u64,
    inv_d: u64,
    rowexp_d: u64,   // 행→전문가 (순열 후 순서) — 그룹 GEMM용
    perm_pad_d: u64,
    inv_pad_d: u64,
    tilexp_d: u64,
    rows_pad: usize,
    /// 디바이스 rows_pad 포인터(디바이스 그룹화 경로에서만 != 0).
    rows_pad_d: u64,
    /// 디바이스 전문가 오프셋 포인터(폴백이 필요할 때만 사용).
    off_d: u64,
    /// 오프셋의 비동기 d2h 목적지(핀) — 폴백이 읽는다.
    pinned_off: *mut u8,
    off: Vec<usize>,
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
    soff: std::sync::Mutex<GBuf>,
    atn: std::sync::Mutex<GBuf>,
    qsp: std::sync::Mutex<GBuf>,
    /// 출력 스테이징 재사용 버퍼(d2h가 전체를 덮어쓰므로 0-채움 불필요).
    ybuf: std::sync::Mutex<Vec<f32>>,
    /// 컨텍스트 길이(엔진이 주입). KV 풀을 이 크기로 선할당한다.
    ctx_len: std::sync::atomic::AtomicUsize,
    /// plans/67 2a: q/k norm·cs(로프 테이블) 상수용 소형 풀.
    cst: std::sync::Mutex<GBuf>,
    /// QSA KV 상주 풀 [(full_idx, seq)] → (k, v) — plans/67 3단계.
    qsa_kv: std::sync::Mutex<std::collections::HashMap<(usize, usize), (GBuf, GBuf)>>,
    /// 상주 풀 워터마크 [(full_idx, seq)] → 다음 기대 pos — 풀이 값을 쓴 적 없는
    /// 구멍(값 경로 청크·롤백·리줌)을 읽는 사고를 막는다(불일치 → 업로드 폴백).
    qsa_kv_pos: std::sync::Mutex<std::collections::HashMap<(usize, usize), usize>>,
    /// plans/73: 인덱서 k 상주 풀 [(full_idx, seq)] — 디바이스 선택의 원천.
    qsa_idxk: std::sync::Mutex<std::collections::HashMap<(usize, usize), GBuf>>,
    /// 블록키 상주 풀 [(full_idx, seq)] — 증분 갱신(q4_idx_bk_update).
    qsa_bk: std::sync::Mutex<std::collections::HashMap<(usize, usize), GBuf>>,
    /// 인덱서 풀 워터마크(qsa_kv_pos와 동일 규약).
    qsa_idx_pos: std::sync::Mutex<std::collections::HashMap<(usize, usize), usize>>,
    /// 선택 스크래치: iq_rope / 점수 / 선택플래그.
    qsa_iqr: std::sync::Mutex<GBuf>,
    qsa_scr: std::sync::Mutex<GBuf>,
    qsa_selflag: std::sync::Mutex<GBuf>,
    /// 인덱저 상수(콘텐츠 해시로 1회 업로드 캐시).
    qsa_iqw: std::sync::Mutex<(u64, GBuf)>,
    qsa_ikw: std::sync::Mutex<(u64, GBuf)>,
    qsa_csidx: std::sync::Mutex<(usize, usize, GBuf)>,
    /// qk_norm_rope 상수 업로드 캐시 — **(ptr,len) 키 맵**.
    /// 단일 슬롯이던 시절엔 층마다 타일이 달라 매 층 미스 → 24KB+2KB 동기 복사
    /// ×12층 = 3.4ms/층(스텝의 40ms)이 호스트를 세웠다 (2026-09-16 실측).
    qn_map: std::sync::Mutex<std::collections::HashMap<(u64, usize), GBuf>>,
    kn_map: std::sync::Mutex<std::collections::HashMap<(u64, usize), GBuf>>,
    cst_cache: std::sync::Mutex<(usize, usize)>,
    /// plans/73: PLE conv 링 상주 상태 [seq] + 워터마크(접두 되감기 검출).
    ple_ring: std::sync::Mutex<std::collections::HashMap<usize, GBuf>>,
    ple_ring_pos: std::sync::Mutex<std::collections::HashMap<usize, usize>>,
    /// PLE norm/conv 상수(콘텐츠 해시 1회 업로드).
    ple_nk: std::sync::Mutex<(u64, GBuf)>,
    ple_nq: std::sync::Mutex<(u64, GBuf)>,
    ple_nc: std::sync::Mutex<(u64, GBuf)>,
    ple_cw: std::sync::Mutex<(u64, GBuf)>,
    /// MoE 전문가 그룹화 — x 행 gather / 결과 행 산란 / 순열 업로드.
    xperm: std::sync::Mutex<GBuf>,
    yperm: std::sync::Mutex<GBuf>,
    rperm: std::sync::Mutex<GBuf>,
    rexp: std::sync::Mutex<GBuf>,
    texp: std::sync::Mutex<GBuf>,
    gxp: std::sync::Mutex<GBuf>,
    gyp: std::sync::Mutex<GBuf>,
    gp: std::sync::Mutex<GBuf>,
    gi: std::sync::Mutex<GBuf>,
    rperm2: std::sync::Mutex<GBuf>,
    /// 그룹화 캐시(위 MoeGroup) + 무효화 세대(라우팅이 갱신될 때 증가).
    moe_group: std::sync::Mutex<Option<MoeGroup>>,
    /// 활성 양자화 캐시 — 전문가별 GEMM 256회가 같은 행 집합을 재양자화하던 것
    /// (호출당 ~40us)을 1회로. 버퍼(fxq)가 단일 슬롯이라 새 양자화가 곧 교체다.
    quant_cache: std::sync::Mutex<Option<(usize, usize, u64, u64, usize)>>,
    moe_gen: std::sync::atomic::AtomicU64,
}

// SAFETY: 포인터는 디바이스 주소 — 스레드 간 공유해도 HIP 런타임이 직렬화한다
// (단일 스트림 + 호출부는 decode1을 직렬 호출). VkAcc와 동일한 계약.
unsafe impl Send for Q4Acc {}
unsafe impl Sync for Q4Acc {}

/// 활성 q8 버퍼의 행 스트라이드(워드) — quant_q8과 동일 규약.
pub(crate) fn xq_words(n: usize) -> usize {
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
            soff: std::sync::Mutex::new(GBuf::new("soff")),
            atn: std::sync::Mutex::new(GBuf::new("atn")),
            qsp: std::sync::Mutex::new(GBuf::new("qsp")),
            ybuf: std::sync::Mutex::new(Vec::new()),
            ctx_len: std::sync::atomic::AtomicUsize::new(0),
            cst: std::sync::Mutex::new(GBuf::new("cst")),
            qsa_kv: std::sync::Mutex::new(std::collections::HashMap::new()),
            qsa_kv_pos: std::sync::Mutex::new(std::collections::HashMap::new()),
            qsa_idxk: std::sync::Mutex::new(std::collections::HashMap::new()),
            qsa_bk: std::sync::Mutex::new(std::collections::HashMap::new()),
            qsa_idx_pos: std::sync::Mutex::new(std::collections::HashMap::new()),
            qsa_iqr: std::sync::Mutex::new(GBuf::new("qsa_iqr")),
            ple_ring: std::sync::Mutex::new(std::collections::HashMap::new()),
            ple_ring_pos: std::sync::Mutex::new(std::collections::HashMap::new()),
            ple_nk: std::sync::Mutex::new((0, GBuf::new("ple_nk"))),
            ple_nq: std::sync::Mutex::new((0, GBuf::new("ple_nq"))),
            ple_nc: std::sync::Mutex::new((0, GBuf::new("ple_nc"))),
            ple_cw: std::sync::Mutex::new((0, GBuf::new("ple_cw"))),
            qsa_scr: std::sync::Mutex::new(GBuf::new("qsa_scr")),
            qsa_selflag: std::sync::Mutex::new(GBuf::new("qsa_selflag")),
            qsa_iqw: std::sync::Mutex::new((0, GBuf::new("qsa_iqw"))),
            qsa_ikw: std::sync::Mutex::new((0, GBuf::new("qsa_ikw"))),
            qsa_csidx: std::sync::Mutex::new((0, 0, GBuf::new("qsa_csidx"))),
            qn_map: std::sync::Mutex::new(std::collections::HashMap::new()),
            kn_map: std::sync::Mutex::new(std::collections::HashMap::new()),
            cst_cache: std::sync::Mutex::new((0, 0)),
            xperm: std::sync::Mutex::new(GBuf::new("xperm")),
            yperm: std::sync::Mutex::new(GBuf::new("yperm")),
            rperm: std::sync::Mutex::new(GBuf::new("rperm")),
            rexp: std::sync::Mutex::new(GBuf::new("rexp")),
            texp: std::sync::Mutex::new(GBuf::new("texp")),
            gxp: std::sync::Mutex::new(GBuf::new("gxp")),
            gyp: std::sync::Mutex::new(GBuf::new("gyp")),
            gp: std::sync::Mutex::new(GBuf::new("gp")),
            gi: std::sync::Mutex::new(GBuf::new("gi")),
            rperm2: std::sync::Mutex::new(GBuf::new("rperm2")),
            moe_group: std::sync::Mutex::new(None),
            quant_cache: std::sync::Mutex::new(None),
            moe_gen: std::sync::atomic::AtomicU64::new(0),
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
        if !base.is_multiple_of(4096) || n < (4 << 20) {
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

    /// 핸들의 바이트 용량(frame_alloc·frame_slice가 적어 둔 cap).
    /// 판독/기입이 이걸 넘으면 **조용한 OOB**였다 — 이웃 버퍼 내용을 읽거나
    /// 덮어쓰고도 에러가 없어 디버깅을 오도한다(2026-09-17). 명시 에러로 바꾼다.
    fn fcap(&self, h: u64) -> Result<usize, String> {
        let v = self.frames.lock().map_err(|e| e.to_string())?;
        v.get((h.checked_sub(1).ok_or("frame 핸들 0")?) as usize)
            .map(|(_, cap)| *cap)
            .ok_or_else(|| format!("frame 핸들 없음: {h}"))
    }

    /// 판독/기입 공통 경계 검사.
    fn fchk(&self, h: u64, bytes: usize, what: &str) -> Result<(), String> {
        let cap = self.fcap(h)?;
        if bytes > cap {
            return Err(format!("{what} 범위 초과: need {bytes}B > cap {cap}B (핸들 {h})"));
        }
        Ok(())
    }

}

impl llm170_core::matmul::GraphCapture for Q4Acc {

    fn capture_mark(&self, tag: &str) -> Result<(), String> {
        unsafe { crate::rawhip::capture_mark(self.ctx.stream, tag) }
    }
    fn graph_capture_begin(&self) -> Result<(), String> {
        unsafe { crate::rawhip::graph_capture_begin(self.ctx.stream) }
    }
    fn graph_capture_end(&self) -> Result<(), String> {
        unsafe { crate::rawhip::graph_capture_end(self.ctx.stream) }
    }
    fn graph_replay(&self, on: bool) -> Result<(), String> {
        unsafe { crate::rawhip::graph_replay(on) }
    }
    fn graph_abort(&self) {
        crate::rawhip::graph_abort();
    }
    fn pre_pair(&self, on: bool) {
        self.ctx.pre_pair.store(on, std::sync::atomic::Ordering::Relaxed);
    }
    fn pre_mark(&self) -> Result<(), String> {
        self.ctx.pre_mark()
    }
    fn pre_ready(&self) -> bool {
        self.ctx.pre_ready()
    }
    fn pre_join(&self) -> Result<(), String> {
        self.ctx.pre_join()
    }
}


/// FNV-1a f32 슬라이스 해시 — 상수 업로드 캐시 키(plans/73).
fn fnv_hash(data: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &v in data {
        h ^= v.to_bits() as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// 프레임 op 인자 벡터 — 로컬 변수의 주소를 c_void로.
macro_rules! cargs {
    ($($e:expr),+ $(,)?) => {{
        let mut v: Vec<*mut std::ffi::c_void> = Vec::new();
        $( v.push($e as *mut _ as *mut std::ffi::c_void); )+
        v
    }};
}

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
    eprintln!("{}", crate::rawhip::probes::device_report(&a.ctx));
    Ok(std::sync::Arc::new(a))
}

mod checks;
mod frame;
mod moe;
mod qsa;
mod value;

pub use checks::*;
