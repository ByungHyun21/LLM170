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
    qn: std::sync::Mutex<GBuf>,
    kn: std::sync::Mutex<GBuf>,
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
    /// 진단(LLM170_MOE_HASH): moe 최종 출력(out) 해시 — DEV/HOST 경로 비교용.
    fn moe_hash_check(&self, tag: &str, op: *mut u8, rows: usize, n_out: usize) -> Result<(), String> {
        if std::env::var_os("LLM170_MOE_HASH").is_none() {
            return Ok(());
        }
        // out의 행 수는 토큰 수(t) — rows는 t·k_sel이므로 rows/k_sel… 대신
        // 버퍼 규약상 out은 [t][n_out]이고 t = frame t_cur.
        let t = self.t_cur().max(1);
        let n = (t * n_out).min(rows * n_out);
        let mut v = vec![0.0f32; n];
        self.ctx.d2h(bytemuck::cast_slice_mut(&mut v), op as *const u8)?;
        let mut x = 0xcbf29ce484222325u64;
        for f in v.iter() {
            x ^= f.to_bits() as u64;
            x = x.wrapping_mul(0x100000001b3);
        }
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let q = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        eprintln!("# moe-hash {tag} #{q} t={t} n_out={n_out} h={x:016x}");
        Ok(())
    }
    /// 행 순열 gather: dst[g] = src[perm[g]] (row_u32 = 행당 u32 수).
    /// 산란은 역순열을 넘겨 같은 커널로 수행한다.
    /// 디바이스 그룹화 런치 — q4_moe_group_t1(단일 블록·단일 스레드).
    /// 호스트 왕복(동기 d2h + 테이블 빌드 + h2d 3회)을 대체한다. 테이블은
    /// ids의 순수 함수이므로 결과는 호스트판과 동일(비트 동일).
    #[allow(clippy::too_many_arguments)]
    fn moe_group_dev(
        &self,
        ids: u64,
        ne: usize,
        rows: usize,
        off_d: u64,
        perm_d: u64,
        inv_d: u64,
        rowexp_d: u64,
        perm_pad_d: u64,
        inv_pad_d: u64,
        tilexp_d: u64,
        rows_pad_d: u64,
        bound: usize,
    ) -> Result<(), String> {
        let mut ip = self.fptr(ids)?;
        let (mut od, mut pd, mut iv) = (off_d as *mut u8, perm_d as *mut u8, inv_d as *mut u8);
        let (mut rx, mut pp, mut ipd) =
            (rowexp_d as *mut u8, perm_pad_d as *mut u8, inv_pad_d as *mut u8);
        let (mut tx, mut rpd) = (tilexp_d as *mut u8, rows_pad_d as *mut u8);
        let (mut n_e, mut rws) = (ne as i32, rows as i32);
        let mut bnd = bound as i32;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&mut ip) as *mut _ as *mut std::ffi::c_void,
            (&mut n_e) as *mut _ as *mut std::ffi::c_void,
            (&mut rws) as *mut _ as *mut std::ffi::c_void,
            (&mut od) as *mut _ as *mut std::ffi::c_void,
            (&mut pd) as *mut _ as *mut std::ffi::c_void,
            (&mut iv) as *mut _ as *mut std::ffi::c_void,
            (&mut rx) as *mut _ as *mut std::ffi::c_void,
            (&mut pp) as *mut _ as *mut std::ffi::c_void,
            (&mut ipd) as *mut _ as *mut std::ffi::c_void,
            (&mut tx) as *mut _ as *mut std::ffi::c_void,
            (&mut rpd) as *mut _ as *mut std::ffi::c_void,
            (&mut bnd) as *mut _ as *mut std::ffi::c_void,
        ];
        self.ctx.launch3("q4_moe_group_t1", 1, 1, 1, 128, &mut args)
    }


    fn rows_permute(
        &self,
        src: *mut u8,
        perm: &[u32],
        dst: *mut u8,
        row_u32: usize,
        n: usize,
    ) -> Result<(), String> {
        if n == 0 || row_u32 == 0 {
            return Ok(());
        }
        let pd = {
            let mut g = self.rperm.lock().map_err(|e| e.to_string())?;
            g.ensure(&self.ctx, n * 4)?
        };
        self.ctx.h2d(pd, bytemuck::cast_slice(perm))?;
        self.rows_permute_dev(src, pd as *mut u8, dst, row_u32, n)
    }

    /// 디바이스 순열판 — 순열이 이미 GPU에 있으면 h2d/동기 없이 런치만 한다.
    fn rows_permute_dev(
        &self,
        src: *mut u8,
        perm_d: *mut u8,
        dst: *mut u8,
        row_u32: usize,
        n: usize,
    ) -> Result<(), String> {
        if n == 0 || row_u32 == 0 {
            return Ok(());
        }
        let (mut a, mut b, mut c) = (src, perm_d, dst);
        let (mut ru, mut nn) = (row_u32 as i32, n as i32);
        self.kop(
            "q4_rows_permute_u32",
            n as u32,
            1,
            1,
            128,
            &mut cargs!(&mut a, &mut b, &mut c, &mut ru, &mut nn),
        )
    }

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
            qn: std::sync::Mutex::new(GBuf::new("qn")),
            kn: std::sync::Mutex::new(GBuf::new("kn")),
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
        // llama MMQ 경로(부록5: q4_K maxrel 6e-4) — qwen35 raw 디코더가 쓰는
        // 바로 그 mul_mat_q 커널. f32 활성을 직접 양자화하므로 frame_quant를
        // 건너뛴다. 형상은 qwen35와 같은 게이트(t>=32).
        let ty = ggml_id(w.ty);
        if t >= 32
            && matches!(ty, 12 | 13 | 14 | 23)
            && std::env::var_os("LLM170_Q4_MMQ").is_some()
            && self.ctx.gemm_mmq(ty, x as *const u8, wd, n_in, n_out, t, out).is_ok()
        {
            return Ok(());
        }
        // f16 경로는 t=1에서만 검증됨(plans/65 §19): t>1(프리필)은 x 취급이 어긋나
        // 값이 깨진다(f16-map t=4 프로브로 재현). t>1 해결 전에는 배선하지 않는다.
        let (xq, xq_w) = self.frame_quant(x, n_in, t)?;
        self.launch_gemm(ty, xq, wd, n_in, n_out, xq_w, t, out)
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
        // q4_K MMQ급 타일 — 산술은 core dot_q4k_q8과 동일 순서로 썼다.
        // 2026-09-14 재검증: q4k-micro가 max_abs 0.0(CPU 참조와 비트 동일)이고
        // Flash-Next diverse(24토큰 프리필+8디코드)도 기준열과 비트 동일하다 —
        // 과거 "실측 오답(ffn_gate_exps t=20)" 기록은 이후 수정으로 해소됐다.
        // 다만 pp2048 실측이 9,012~9,118 → 8,899~9,155ms로 중립(노이즈 범위)이라
        // 기본은 여전히 끈 상태다: 이득이 아니라 속도 근거로 옵트인 유지.
        if ty == ggml_id(GgmlType::Q4K)
            && t >= 16
            && (std::env::var_os("LLM170_Q4K_MMQ").is_some()
                || std::env::var_os("LLM170_Q4K_OUTS").is_some())
        {
            // 행-배치 타일(plans/65) — 가중치 디퀀트를 행 루프 밖으로.
            if std::env::var_os("LLM170_Q4K_Y").is_some() {
                let rpt: usize = std::env::var("LLM170_Q4K_YRPT")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(16);
                // 커널이 += 누산이므로 출력을 0으로 초기화한다(출력 버퍼는 매
                // 호출 새로 쓰이는 스크래치라 안전).
                unsafe {
                    std::ptr::write_bytes(out as *mut f32, 0, t * n_out);
                }
                let mut xq_p = xq as *mut std::ffi::c_void;
                let mut w_p = w as *mut std::ffi::c_void;
                let mut part_p = self.ctx.scratch(4)? as *mut std::ffi::c_void;
                let mut o_p = out as *mut std::ffi::c_void;
                let (mut ni, mut no, mut xw, mut tt, mut rp) =
                    (n_in as i32, n_out as i32, xq_w as i32, t as i32, rpt as i32);
                let mut args: Vec<*mut std::ffi::c_void> = vec![
                    (&mut xq_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut w_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut part_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut ni) as *mut _ as *mut std::ffi::c_void,
                    (&mut no) as *mut _ as *mut std::ffi::c_void,
                    (&mut xw) as *mut _ as *mut std::ffi::c_void,
                    (&mut tt) as *mut _ as *mut std::ffi::c_void,
                    (&mut rp) as *mut _ as *mut std::ffi::c_void,
                ];
                return self.ctx.launch3(
                    "q4_gemm_q4k_y",
                    n_out.div_ceil(256) as u32,
                    t.div_ceil(rpt) as u32,
                    1,
                    256,
                    &mut args,
                );
            }
            // x-스테이징 타일(plans/65) — 출력별 x 재독 제거. 로직·순서는 _m과 동일.
            if std::env::var_os("LLM170_Q4K_X").is_some() {
                let mut xq_p = xq as *mut std::ffi::c_void;
                let mut w_p = w as *mut std::ffi::c_void;
                let mut part_p = self.ctx.scratch(4)? as *mut std::ffi::c_void;
                let mut o_p = out as *mut std::ffi::c_void;
                let (mut ni, mut no, mut xw, mut tt) =
                    (n_in as i32, n_out as i32, xq_w as i32, t as i32);
                let mut args: Vec<*mut std::ffi::c_void> = vec![
                    (&mut xq_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut w_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut part_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut ni) as *mut _ as *mut std::ffi::c_void,
                    (&mut no) as *mut _ as *mut std::ffi::c_void,
                    (&mut xw) as *mut _ as *mut std::ffi::c_void,
                    (&mut tt) as *mut _ as *mut std::ffi::c_void,
                ];
                return self.ctx.launch3(
                    "q4_gemm_q4k_x",
                    n_out.div_ceil(16) as u32,
                    t.div_ceil(16) as u32,
                    1,
                    256,
                    &mut args,
                );
            }
            // 형상 스윕용 가변 타일(plans/65) — outs/rows를 env로 지정.
            if let (Ok(outs), Ok(rows)) = (
                std::env::var("LLM170_Q4K_OUTS").map(|v| v.parse::<usize>()),
                std::env::var("LLM170_Q4K_ROWS").map(|v| v.parse::<usize>()),
            ) {
                let (outs, rows) = (outs.unwrap_or(16), rows.unwrap_or(16));
                let mut xq_p = xq as *mut std::ffi::c_void;
                let mut w_p = w as *mut std::ffi::c_void;
                let mut part_p = self.ctx.scratch(4)? as *mut std::ffi::c_void;
                let mut o_p = out as *mut std::ffi::c_void;
                let (mut ni, mut no, mut xw, mut tt) =
                    (n_in as i32, n_out as i32, xq_w as i32, t as i32);
                let (mut oo, mut rr) = (outs as i32, rows as i32);
                let mut args: Vec<*mut std::ffi::c_void> = vec![
                    (&mut xq_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut w_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut part_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut ni) as *mut _ as *mut std::ffi::c_void,
                    (&mut no) as *mut _ as *mut std::ffi::c_void,
                    (&mut xw) as *mut _ as *mut std::ffi::c_void,
                    (&mut tt) as *mut _ as *mut std::ffi::c_void,
                    (&mut oo) as *mut _ as *mut std::ffi::c_void,
                    (&mut rr) as *mut _ as *mut std::ffi::c_void,
                ];
                return self.ctx.launch3(
                    "q4_gemm_q4k_g",
                    n_out.div_ceil(outs) as u32,
                    t.div_ceil(rows) as u32,
                    1,
                    (outs * rows) as u32,
                    &mut args,
                );
            }
            let mut xq_p = xq as *mut std::ffi::c_void;
            let mut w_p = w as *mut std::ffi::c_void;
            let mut part_p = self.ctx.scratch(4)? as *mut std::ffi::c_void;
            let mut o_p = out as *mut std::ffi::c_void;
            let mut ni = n_in as i32;
            let mut no = n_out as i32;
            let mut xw = xq_w as i32;
            let mut tt = t as i32;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut xq_p) as *mut _ as *mut std::ffi::c_void,
                (&mut w_p) as *mut _ as *mut std::ffi::c_void,
                (&mut part_p) as *mut _ as *mut std::ffi::c_void,
                (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                (&mut ni) as *mut _ as *mut std::ffi::c_void,
                (&mut no) as *mut _ as *mut std::ffi::c_void,
                (&mut xw) as *mut _ as *mut std::ffi::c_void,
                (&mut tt) as *mut _ as *mut std::ffi::c_void,
            ];
            return self.ctx.launch3(
                "q4_gemm_q4k_m",
                n_out.div_ceil(16) as u32,
                t.div_ceil(16) as u32,
                1,
                256,
                &mut args,
            );
        }
        if ty == ggml_id(GgmlType::Q5_1) {
            // 타일 판은 출력 4개/블록 — 그리드도 4로 나눈다.
            let tiled = std::env::var_os("LLM170_NO_Q5_1_T").is_none();
            let outs_per_block = if tiled { 4usize } else { 1 };
            let nblk = n_out.div_ceil(outs_per_block);
            let gy = nblk.min(65535) as u32;
            let gz = nblk.div_ceil(65535) as u32;
            let part = self.ctx.scratch(n_out * 64 * 8)?;
            let mut xq_p = xq as *mut std::ffi::c_void;
            let mut w_p = w as *mut std::ffi::c_void;
            let mut part_p = part as *mut std::ffi::c_void;
            let mut o_p = out as *mut std::ffi::c_void;
            let mut ni = n_in as i32;
            let mut no = n_out as i32;
            let mut xw = xq_w as i32;
            let mut tt = t as i32;
            // 16행 타일 판(2026-09-13) — 가중치 1회 독서로 상각. 원판은 행마다
            // 같은 가중치 행을 다시 읽어 MoE expert-down(20행 그룹)에서 20배
            // 증폭이었다(실측 2715ms/청크). 산술 순서는 동일 = 비트 동일.
            // MMQ급 판(스레드당 (출력,행) 누산) — 기본. 누산 순서가 달라
            // 비트 동일이 아니지만 q4-acc-check 실측 max_abs 5.96e-8 /
            // max_rel 2.1e-5 (q5_1 양자화 오차 ~1e-2의 1/500)이고 230토큰
            // greedy 스트림이 동일하다 — llama.cpp/vLLM과 같은 허용 오차 계약.
            // 비트 동일 판은 LLM170_Q5_1_EXACT=1로 복귀.
            let mmq = t >= 16
                && std::env::var_os("LLM170_Q5_1_EXACT").is_none()
                && ty == ggml_id(GgmlType::Q5_1);
            let kern = match (mmq, tiled) {
                (true, _) => "q4_gemm_q5_1_m",
                (false, true) => "q4_gemm_q5_1_t",
                (false, false) => "q4_gemm_q5_1",
            };
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut xq_p) as *mut _ as *mut std::ffi::c_void,
                (&mut w_p) as *mut _ as *mut std::ffi::c_void,
                (&mut part_p) as *mut _ as *mut std::ffi::c_void,
                (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                (&mut ni) as *mut _ as *mut std::ffi::c_void,
                (&mut no) as *mut _ as *mut std::ffi::c_void,
                (&mut xw) as *mut _ as *mut std::ffi::c_void,
                (&mut tt) as *mut _ as *mut std::ffi::c_void,
            ];
            if kern.ends_with("_m") {
                let nblk = n_out.div_ceil(16);
                let smem = (16 * (n_in / 32) * 24) as u32;
                return self.ctx.launch3_dyn(
                    kern,
                    nblk.min(65535) as u32,
                    t.div_ceil(16) as u32,
                    1,
                    256,
                    smem,
                    &mut args,
                );
            }
            let gx = if tiled { t.div_ceil(16) as u32 } else { t as u32 };
            return self.ctx.launch3(kern, gx, gy, gz, if tiled { 256 } else { 64 }, &mut args);
        }
        // t≥16: MMQ 타일 우선 — 가중치 1회 독서 + 토큰 타일 상각(raw 디코더
        // mm_b와 동일 게이트). 타일 커널이 없는 타입은 GEMV 폴백.
        // 실측(2026-09-14, q4k-bench 2560x6144): 2.5-2.7 TFLOPS로 **t에 걸쳐 평탄**하다
        // (t=128 1.474ms 2.73, t=512 6.401 2.52, t=1024 12.581 2.56, t=2048 27.203
        // 2.37 TFLOPS). 즉 점유율 문제가 아니라 이 형상의 커널 고유 비용이고
        // 가중치 대역도 0.3-6 GB/s뿐이라 연산·대역폭 어느 쪽도 아니다 — llama.cpp
        // 대비 프리필 1.33x가 사는 곳이다. 27B가 같은 계열로 19.5 TFLOPS를 내는 것은
        // n_in/n_out이 더 큰 형상(5120x17408)이라 행당 상각이 크기 때문이다.
        // 같은 형상에서 q4_K MMQ 타일(LLM170_Q4K_MMQ/Y)은 오히려 느렸고(33-34ms),
        // Q6K/Q4_K f16 융합 dequant도 중립이었다. 남은 방향은 그래프당 dequant 캐시.
        // t≥16: MMQ 타일 우선 — 단 **128토큰 이하로 쪼개서** 호출한다.
        // j128 CO는 gz>1(다중 토큰 사분면)일 때 n_in=6144 형상에서 폴트한다
        // (2026-09-12 실측: t=129 폴트, t=128 정상, GEMV 경로는 비트 동일).
        if t >= 16 && std::env::var_os("LLM170_Q4_NO_TILE").is_none() {
            // j128/v4 계열(=8/12/13/14/23)은 사분면 지원 — 그 외 타입만 128씩 분할.
            let tq_mode = matches!(
                ty,
                x if x == ggml_id(GgmlType::Q8_0)
                    || x == ggml_id(GgmlType::Q4K)
                    || x == ggml_id(GgmlType::Q5K)
                    || x == ggml_id(GgmlType::Q6K)
                    || x == ggml_id(GgmlType::Q3K)
            );
            let mut ok = true;
            for c in 0..if tq_mode { 1 } else { t.div_ceil(128) } {
                let t0 = c * 128;
                let tc = if tq_mode { t } else { 128.min(t - t0) };
                let xsrc = unsafe { xq.add(t0 * xq_w * 4) };
                let osrc = unsafe { out.add(t0 * n_out * 4) };
                if let Err(e) = self
                    .ctx
                    .gemm_tile(xsrc, w, self.ktab2, ty, n_in, n_out, xq_w, tc, osrc)
                {
                    if std::env::var_os("LLM170_Q4_DBG").is_some() {
                        use std::sync::Mutex;
                        use std::sync::OnceLock;
                        static SEEN: OnceLock<Mutex<Vec<(u32, usize, usize, usize)>>> = OnceLock::new();
                        let seen = SEEN.get_or_init(|| Mutex::new(Vec::new()));
                        if let Ok(mut v) = seen.lock() {
                            // (ty, n_in, n_out) 별 1회 + t는 128 단위 구간으로 구분.
                            let key = (ty, n_in, n_out, (tc / 128) * 128);
                            if !v.contains(&key) && v.len() < 24 {
                                v.push(key);
                                eprintln!(
                                    "# gemm_tile 폴백: ty={ty} n_in={n_in} n_out={n_out} t={tc} err={e}"
                                );
                            }
                        }
                    }
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
        let mut x_p = x as *mut std::ffi::c_void;
        let mut w_p = w as *mut std::ffi::c_void;
        let mut o_p = out as *mut std::ffi::c_void;
        let mut ni = n_in as i32;
        let mut no = n_out as i32;
        let mut st = n_in as i32;
        // MMQ급 타일 — 커널 자체는 276→176ms로 빨라지지만(스레드당 40 MAC →
        // 2560 MAC) 엔드투엔드 pp512는 136.5 vs 137.8로 **차이 없음**(파이프라인
        // 뒤에 숨음). 이득 없는 계약 변경이라 기본에서 제외 — 옵트인만 남긴다.
        // f32 MMQ 판 기본(2026-09-13): pp2048 10,781→10,307ms, 토큰 동일.
        if t >= 16 {
            let mut tt = t as i32;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut x_p) as *mut _ as *mut std::ffi::c_void,
                (&mut w_p) as *mut _ as *mut std::ffi::c_void,
                (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                (&mut ni) as *mut _ as *mut std::ffi::c_void,
                (&mut no) as *mut _ as *mut std::ffi::c_void,
                (&mut st) as *mut _ as *mut std::ffi::c_void,
                (&mut tt) as *mut _ as *mut std::ffi::c_void,
            ];
            return self.ctx.launch3(
                "q4_gemm_f32_m",
                n_out.div_ceil(16) as u32,
                t.div_ceil(16) as u32,
                1,
                256,
                &mut args,
            );
        }
        // plans/73: t=1은 워프-퍼-출력판 — 저출력(hc inject [10240→4])·라우터
        // 형상에서 원판 대비 3-6×. 누산 재배열 편차는 게이트로 검증.
        if t == 1 && n_in % 4 == 0 && std::env::var("LLM170_F32W").as_deref() != Ok("0") {
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut x_p) as *mut _ as *mut std::ffi::c_void,
                (&mut w_p) as *mut _ as *mut std::ffi::c_void,
                (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                (&mut ni) as *mut _ as *mut std::ffi::c_void,
                (&mut no) as *mut _ as *mut std::ffi::c_void,
            ];
            return self.ctx.launch3(
                "q4_gemm_f32_w",
                n_out.div_ceil(8) as u32,
                1,
                1,
                256,
                &mut args,
            );
        }
        let gy = n_out.min(65535) as u32;
        let gz = n_out.div_ceil(65535) as u32;
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
            // f16 경로는 128 사분면 경계까지 쓰므로 여유를 둔다(행 < t 만 사용).
            let need = if std::env::var_os("LLM170_F16_ACC").is_some() {
                t.div_ceil(128) * 128 * n_out * 4
            } else {
                t * n_out * 4
            };
            yb.ensure(&self.ctx, need)?
        };
        // 실측(2026-09-14, pp2048): 아래 할당+d2h+행 산포가 청크당 ~0.75s(8%)를
        // 쓴다(LLM170_Q4ACC_TIME으로 d2h=1.5s/400콜). 스테이지가 행 벡터 대신
        // 디바이스 상주 버퍼를 받으면 사라지는 비용 — QSA 선택목록 물질화와 같은 뿌리.
        //
        // 장문맥 디코드(pp8192, t=1)에서는 이 d2h가 **호출 비용의 96%**다:
        // LLM170_Q4ACC_TIME 600콜 기준 평균 5.4ms/호출인데 upload 0.2s·quant 0.0s·
        // launch 0.0s(비동기)이고 d2h만 3.1s(=5.2ms/호출)다. d2h는 앞서 큐에 넣은
        // 디바이스 작업을 기다리므로 이 값은 "그 호출이 동기화하는 디바이스 시간"이다.
        // t=1 GEMV 자체는 수십 us이므로, 스텝 비용을 결정하는 것은 **동기화 횟수**다
        // (스텝당 ~25회 x 5.2ms ≈ 130ms = 프레임 t=1 실측 137ms와 일치).
        // 다음 지렛대: 동기 횟수를 줄이거나(그룹핑) d2h를 뒤로 미루는 것.
        //
        // d2h 대역 자체도 실측했다(q4-d2h-bench): 8MB 0.476ms = **17.6 GB/s**.
        // 따라서 프리필의 호출당 5.2ms는 대부분 출력 전송이다(t=2048 x n_out 6144
        // = 50MB -> 2.8ms + 런치/커널). 즉 **출력을 호스트로 가져오는 한 이 비용은
        // 사라지지 않는다** — 스테이지 API를 디바이스 상주 버퍼로 바꾸는 것이
        // 프리필(8%)과 장문맥 디코드(58% QSA 스테이지)의 공통 해법이다.
        // 출력 스테이징은 **영속 버퍼**를 재사용한다: d2h가 전체를 덮어쓰므로
        // 매 호출 `vec![0.0; t*n_out]`로 할당+0-채움할 필요가 없다(프리필에서
        // 호출당 50MB — 그룹 5회면 250MB의 memset이 사라진다).
        let mut yb = self.ybuf.lock().map_err(|e| e.to_string())?;
        if yb.len() < t * n_out {
            yb.resize(t * n_out, 0.0); // 확장 시에만 0 채움
        }
        let t_k = std::time::Instant::now();
        if w_f32 {
            self.launch_gemm_f32(xf, w_slice, n_in, n_out, t, ydev)?;
        } else {
            // f16 경로 A/B — 실모델 텐서·실활성으로 검증(q4-acc-check가 미러와 대조).
            // 실측(2026-09-14): 게이트를 Q4_K/Q6_K(12/14)로 넓혀도 pp2048 중립이었다
            // (9,087.8 vs 9,043.9ms). 경로는 실제로 타고(F16_DBG=48콜: QSA wq
            // [6144x2560]x12, wk/wv [2560x512]x24) 200토큰 프롬프트 greedy 스트림도
            // 동일했지만, dequant가 **호출마다** 돌아 상각되지 않는다. plans/66 P1의
            // 실제 내용은 "그래프당 1회 dequant 후 캐시"이고 그게 빠져 있다.
            let ty0 = ggml_id(w.ty);
            if t >= 32
                && ty0 == 8
                && std::env::var_os("LLM170_F16_ACC").is_some()
                && self
                    .ctx
                    .gemm_f16_deq(ty0, xf as *const u8, w_slice, n_in, n_out, t, ydev)
                    .is_ok()
            {
                // 임시 진단: LLM170_F16_DBG=1 이면 호출 직후 동기화해 실패 지점을 명명한다.
                if std::env::var_os("LLM170_F16_DBG").is_some() {
                    self.ctx.sync().map_err(|e| format!("f16 sync [{n_in}x{n_out}] t={t}: {e}"))?;
                    eprintln!("f16-deq OK [{n_in}x{n_out}] t={t}");
                }
            } else {
            self.launch_gemm(ty0, xq, w_slice, n_in, n_out, xq_w, t, ydev)?;
            }
        }
        let k_ns = t_k.elapsed().as_nanos() as u64;
        let t_d = std::time::Instant::now();
        let yflat = &mut yb[..t * n_out];
        self.ctx.d2h(bytemuck::cast_slice_mut(yflat), ydev)?;
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
    fn set_ctx_len(&self, n: usize) {
        self.ctx_len.store(n, std::sync::atomic::Ordering::Relaxed);
    }

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
        // gdn_ar_w_swap: 전치 상태 레이아웃(s[dv*d+kdim]) + d=128 고정(레인당
        // kdim 4개). 구 q4_gdn_ar_w의 열 단위 접근은 512B 스트라이드였다.
        self.ctx.launch3(
            "gdn_ar_w_swap",
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
    /// ids는 확률순(전문가순 아님)이라 연속 런이 1행씩 흩어진다 — 실측
    /// t=512·k_sel=10에서 런치 ~4000회/층. 카운팅 정렬로 전문가 순으로 묶어
    /// 런치 수를 전문가 수 수준으로 줄이고(순열은 relu 없이 안정), 결과 행
    /// 순서는 역순열 산란으로 복원한다(가중합이 원래 행 순서를 요구).
    fn frame_moe_gemm(
        &self,
        x: u64,
        ws: &llm170_core::matmul::Weight<'_>,
        ids: u64,
        out: u64,
        n_expert_stack: usize,
        k_sel: usize,
    ) -> Result<(), String> {
        let tm = std::env::var_os("LLM170_MOE_TIME").is_some();
        let t0 = std::time::Instant::now();
        let mut lap = t0;
        let phase = |name: &str, lap: &mut std::time::Instant| {
            if tm {
                let ms = lap.elapsed().as_secs_f64() * 1e3;
                if ms >= 0.05 {
                    eprintln!("# moe-phase {name}={ms:.2}ms");
                }
                *lap = std::time::Instant::now();
            }
        };
        let n_in = ws.n_in as usize;
        let n_out = ws.n_out as usize / n_expert_stack.max(1);
        // 행 수 = t·k_sel — 버퍼는 t_max 크기라 길이에서 유도할 수 없다.
        let rows = self.t_cur() * k_sel.max(1);
        let xp = self.fptr(x)?;
        let op_ = self.fptr(out)?;
        let (wd, f32w) = self.dev_weight(ws)?;
        let per_expert = ws.data.len() / n_expert_stack.max(1);
        phase("weight", &mut lap);
        let gen_q = self.moe_gen.load(std::sync::atomic::Ordering::Relaxed);
        let (xq, xq_w) = if f32w {
            (std::ptr::null_mut(), 0usize)
        } else {
            let key = (xp as usize, rows, gen_q);
            let hit = {
                let c = self.quant_cache.lock().map_err(|e| e.to_string())?;
                c.as_ref()
                    .filter(|(xp0, r0, g0, _, _)| (*xp0, *r0, *g0) == key)
                    .map(|(_, _, _, q, w)| (*q as *mut u8, *w))
            };
            match hit {
                Some(v) => v,
                None => {
                    let (q, w) = self.frame_quant(xp, n_in, rows)?;
                    let mut c = self.quant_cache.lock().map_err(|e| e.to_string())?;
                    *c = Some((key.0, key.1, key.2, q as u64, w));
                    (q, w)
                }
            }
        };
        phase("quant", &mut lap);
        // direct-ids(t=1, LLM170_MOE_DIRECT=1): 그룹화 테이블·gather·scatter를
        // 전부 건너뛰고 커널이 ids[row]를 직접 읽는다. 행 순서가 곧 ids 순서라
        // 가중합(ys[e*n+i])이 그대로 맞고, 호스트 왕복(ids d2h+빌드+h2d)도 없다.
        // t=1에서는 k_sel행이 같은 벡터이므로 스트라이드 0으로 0번 행을 읽는다.
        if self.t_cur() == 1
            && ws.ty == GgmlType::Q4K
            && !f32w
            && std::env::var_os("LLM170_MOE_GROUPED").is_none()
        {
            let idp = self.fptr(ids)?;
            // K-분할: 타일 40블록(=1/CU)이던 점유율을 ksplit배로. 부분합은 part에
            // 남기고 reduce가 k 오름차순 합산(결정적, 순서 재결합만 다른 미세 드리프트).
            let ksplit: u32 = std::env::var("LLM170_MOE_KSPLIT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4)
                .clamp(1, 8);
            let mut x_p = xq as *mut std::ffi::c_void;
            let mut w_p = wd as *mut std::ffi::c_void;
            let part_buf = self
                .ctx
                .scratch(rows * n_out * ksplit as usize * 8)?;
            let mut part_p = part_buf as *mut std::ffi::c_void;
            let mut o_p = self.fptr(out)? as *mut std::ffi::c_void;
            let mut ip = idp as *mut std::ffi::c_void;
            let (mut ni, mut no) = (n_in as i32, n_out as i32);
            let (mut xw, mut tt, mut eb) = (0i32, rows as i32, per_expert as i32);
            let mut rp: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut x_p) as *mut _ as *mut std::ffi::c_void,
                (&mut w_p) as *mut _ as *mut std::ffi::c_void,
                (&mut part_p) as *mut _ as *mut std::ffi::c_void,
                (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                (&mut ip) as *mut _ as *mut std::ffi::c_void,
                (&mut ni) as *mut _ as *mut std::ffi::c_void,
                (&mut no) as *mut _ as *mut std::ffi::c_void,
                (&mut xw) as *mut _ as *mut std::ffi::c_void,
                (&mut tt) as *mut _ as *mut std::ffi::c_void,
                (&mut eb) as *mut _ as *mut std::ffi::c_void,
                (&mut rp) as *mut _ as *mut std::ffi::c_void,
            ];
            // 실측(2026-09-14): 호출당 10전문가 x 0.92MB = 9.2MB를 51us에 옮긴다
            // = 180 GB/s ≈ DRAM(236)의 76%. 이미 최적에 가까워 K-분할(4배 블록,
            // -1.4%), GEMV형 그리드(16배 블록 + 트리 환원, 중립), 접근 패턴
            // 프로브(235-264 GB/s로 평탄)가 모두 중립이었다 — 격차가 아니라 산술이었다.
            self.ctx.launch3(
                "q4_gemm_q4k_ge_ids",
                n_out.div_ceil(16) as u32,
                rows.div_ceil(16) as u32,
                ksplit,
                256,
                &mut args,
            )?;
            if ksplit > 1 {
                let mut pp = part_buf as *mut std::ffi::c_void;
                let mut op2 = self.fptr(out)? as *mut std::ffi::c_void;
                let mut nn = (rows * n_out) as i32;
                let mut ks = ksplit as i32;
                let mut rargs: Vec<*mut std::ffi::c_void> = vec![
                    (&mut pp) as *mut _ as *mut std::ffi::c_void,
                    (&mut op2) as *mut _ as *mut std::ffi::c_void,
                    (&mut nn) as *mut _ as *mut std::ffi::c_void,
                    (&mut ks) as *mut _ as *mut std::ffi::c_void,
                ];
                self.ctx.launch3(
                    "q4_gemm_q4k_ids_reduce",
                    ((rows * n_out) as u32).div_ceil(256),
                    1,
                    1,
                    256,
                    &mut rargs,
                )?;
            }
            return Ok(());
        }
        // plans/73: Q5_1 다운의 direct-ids를 **그룹화 캐시 평가 전에** 올린다 —
        // 종전엔 캐시 미스가 q4_moe_group_t1 커널 + 비동기 d2h를 매층 발사하고
        // 곧바로 direct-ids로 반환해 그 작업이 전부 쓰레기였다(0.034ms × 48층
        // + 스텝당 48회의 d2h_issue).
        if self.t_cur() == 1
            && ws.ty == GgmlType::Q5_1
            && !f32w
            && rows > 0
            && n_in / 32 <= 32
            && std::env::var_os("LLM170_MOE_GROUPED").is_none()
            && std::env::var("LLM170_Q5W").as_deref() != Ok("0")
        {
            let idp = self.fptr(ids)?;
            let mut x_p = xq as *mut std::ffi::c_void;
            let mut w_p = wd as *mut std::ffi::c_void;
            let mut part_p = self.ctx.scratch(4)? as *mut std::ffi::c_void;
            let mut o_p = self.fptr(out)? as *mut std::ffi::c_void;
            let mut ip = idp as *mut std::ffi::c_void;
            let (mut ni, mut no) = (n_in as i32, n_out as i32);
            let (mut xw, mut tt) = (xq_w as i32, rows as i32);
            let mut ew = (per_expert / 4) as i32;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut x_p) as *mut _ as *mut std::ffi::c_void,
                (&mut w_p) as *mut _ as *mut std::ffi::c_void,
                (&mut part_p) as *mut _ as *mut std::ffi::c_void,
                (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                (&mut ip) as *mut _ as *mut std::ffi::c_void,
                (&mut ni) as *mut _ as *mut std::ffi::c_void,
                (&mut no) as *mut _ as *mut std::ffi::c_void,
                (&mut xw) as *mut _ as *mut std::ffi::c_void,
                (&mut tt) as *mut _ as *mut std::ffi::c_void,
                (&mut ew) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx.launch3(
                "q4_gemm_q5_1_w_ids",
                n_out.div_ceil(8) as u32,
                rows as u32,
                1,
                256,
                &mut args,
            )?;
            return Ok(());
        }
        // plans/73: Q8_0 다운 전문가도 direct-ids 워프판으로 — 종전엔 이 층들이
        if self.t_cur() == 1
            && ws.ty == GgmlType::Q8_0
            && !f32w
            && rows > 0
            && n_in / 32 <= 32
            && std::env::var_os("LLM170_MOE_GROUPED").is_none()
            && std::env::var("LLM170_Q8IDS").as_deref() != Ok("0")
        {
            let idp = self.fptr(ids)?;
            let mut x_p = xq as *mut std::ffi::c_void;
            let mut w_p = wd as *mut std::ffi::c_void;
            let mut part_p = self.ctx.scratch(4)? as *mut std::ffi::c_void;
            let mut o_p = self.fptr(out)? as *mut std::ffi::c_void;
            let mut ip = idp as *mut std::ffi::c_void;
            let mut ew = per_expert as i32; // 바이트 — 34B 행 비정렬 오프셋용
            let (mut ni, mut no) = (n_in as i32, n_out as i32);
            let (mut xw, mut tt) = (xq_w as i32, rows as i32);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut x_p) as *mut _ as *mut std::ffi::c_void,
                (&mut w_p) as *mut _ as *mut std::ffi::c_void,
                (&mut part_p) as *mut _ as *mut std::ffi::c_void,
                (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                (&mut ip) as *mut _ as *mut std::ffi::c_void,
                (&mut ni) as *mut _ as *mut std::ffi::c_void,
                (&mut no) as *mut _ as *mut std::ffi::c_void,
                (&mut xw) as *mut _ as *mut std::ffi::c_void,
                (&mut tt) as *mut _ as *mut std::ffi::c_void,
                (&mut ew) as *mut _ as *mut std::ffi::c_void,
            ];
            if std::env::var_os("LLM170_Q8IDS_DBG").is_some() {
                eprintln!("# q8ids launch n_in={n_in} n_out={n_out} rows={rows} per_expert={per_expert}");
            }
            self.ctx.launch3(
                "gemm_q8_0_ids",
                n_out.div_ceil(8) as u32,
                rows as u32,
                1,
                256,
                &mut args,
            )?;
            return Ok(());
        }
        let ne = n_expert_stack.max(1);
        // 그룹화 캐시 — gate/up/down 3개 투영이 같은 라우팅을 공유한다. 게이트가
        // 1회만 d2h(동기)+정렬+순열 업로드하고 나머지는 디바이스 순열을 재사용.
        // 실측: 호출마다 동기하던 시절 층당 9회 → MoE 77ms/층(청크 59%).
        let generation = self.moe_gen.load(std::sync::atomic::Ordering::Relaxed);
        let hit = {
            let c = self.moe_group.lock().map_err(|e| e.to_string())?;
            c.as_ref()
                .filter(|g| g.generation == generation && g.rows == rows)
                .map(|g| (g.perm_d, g.inv_d, g.rowexp_d, g.perm_pad_d, g.inv_pad_d, g.tilexp_d, g.rows_pad, g.rows_pad_d, g.off.clone(), g.off_d, g.pinned_off))
        };
        if tm {
            eprintln!("# moe-cache {}", if hit.is_some() { "HIT" } else { "MISS" });
        }
        let (perm_d, inv_d, rowexp_d, perm_pad_d, inv_pad_d, tilexp_d, rows_pad, rows_pad_d, off, off_d, pinned_off) = match hit {
            Some(v) => v,
            None => {
                // t=1(디코드): 그룹화를 GPU에서 한다. 호스트 왕복(동기 d2h + 테이블
                // 빌드 + h2d 3회)이 스텝의 44%(48층×0.9ms)였다 — 테이블은 ids의
                // 순수 함수라 커널로 옮기면 사라진다. 결과는 호스트판과 동일 순서라
                // 비트 동일. (프리필은 행 수가 커서 기존 호스트 경로 유지.)
                // 기본은 호스트 경로 — 2026-09-14 A/B: 호스트 755.4ms vs
                // 디바이스 1006.9ms(tg8, 같은 바이너리). 디바이스판은 테이블이
                // 비트 동일하고 호스트 빌드·h2d 3회를 없애지만, 추가분(그룹 커널
                // + 상한(rows*16+16) 크기로 커진 gather/scatter + 폴백의 이벤트
                // 대기)이 그보다 커서 +31ms/스텝이다. down(q8_0) 폴백까지 그룹
                // 커널로 덮으면 재평가한다. 옵트인: LLM170_MOE_GROUP_DEV=1.
                // t=1 전용 (프리필은 아직 불가). 2026-09-14 리팩터: 상한을 한 곳에서
                // 계산(rows + 16*ne)해 커널 인자·모든 버퍼에 쓰고, 소비 지점에서
                // 디바이스가 보고한 rows_pad를 검증한다 — 리팩터 전에는 상한이
                // 5곳에 흩어져 이 경로 자체가 잠재 OOB였다(rows*16+16=176 vs 실제
                // ≤8,202). 리팩터 후 t=1은 비트 동일로 검증됨.
                // 프리필(t>1)은 아직 5번째 상한 축이 남아 실패한다(h2d 8MB — 크기는
                // h2d 진단이 보고한다). 켜려면 그 축부터 찾아야 한다.
                // t=1(디코드) 전용 — 프리필(t>1)은 plans/68에서 레이아웃 혼재
                // (패딩/비패딩 gather·scatter·폴백 오프셋)를 전면 교정했으나
                // 잔여 발산(16토큰 중 마지막 1개 플립)과 진단 동기화 시에만
                // 재현되는 폴백 행 수 오염이 남아 기본 경로는 유지한다.
                let pf_exp = std::env::var("LLM170_MOE_GROUP_PF").as_deref() == Ok("1");
                if std::env::var("LLM170_MOE_GROUP_DEV").as_deref() != Ok("0")
                    && (self.t_cur() == 1 || pf_exp)
                    && ne <= 512
                    && rows > 0
                {
                    // **단일 상한**: Σ_e ceil(r_e/16)*16 ≤ rows + 16*ne
                    // (전문가당 ≤15행 패딩). 종전 rows*16+16은 16배 과대였고,
                    // 그 값으로 커널 zero-fill·호스트 버퍼가 어긋나 OOB가 났다.
                    let bound = rows + 16 * ne;
                    let (pd, ivd, rxd) = {
                        let mut a = self.rperm.lock().map_err(|e| e.to_string())?;
                        let pd = a.ensure(&self.ctx, rows * 4)? as u64;
                        let mut b = self.rperm2.lock().map_err(|e| e.to_string())?;
                        let ivd = b.ensure(&self.ctx, rows * 4)? as u64;
                        let mut c = self.rexp.lock().map_err(|e| e.to_string())?;
                        // GEMM이 rows_pad까지 rowexp를 읽는다 → bound 크기.
                        let rxd = c.ensure(&self.ctx, bound * 4)? as u64;
                        (pd, ivd, rxd)
                    };
                    let (ppd, ipd, txd, offd, rpd) = {
                        let mut a = self.gp.lock().map_err(|e| e.to_string())?;
                        let ppd = a.ensure(&self.ctx, (bound + 1) * 4)? as u64;
                        let mut b = self.gi.lock().map_err(|e| e.to_string())?;
                        let ipd = b.ensure(&self.ctx, rows * 4)? as u64;
                        let mut c = self.texp.lock().map_err(|e| e.to_string())?;
                        let txd = c.ensure(&self.ctx, (bound / 16 + 1) * 4)? as u64;
                        let mut d = self.gyp.lock().map_err(|e| e.to_string())?;
                        let base = d.ensure(&self.ctx, (ne + 2) * 4)? as u64;
                        let offd = base;
                        let rpd = base + (ne as u64 + 1) * 4; // off 뒤 4B = rows_pad
                        (ppd, ipd, txd, offd, rpd)
                    };
                    self.moe_group_dev(ids, ne, rows, offd, pd, ivd, rxd, ppd, ipd, txd, rpd, bound)?;
                    if std::env::var_os("LLM170_MOE_GCHECK").is_some() {
                        // 진단: 디바이스 테이블과 호스트 재계산을 비교(첫 불일치 지점 출력).
                        self.ctx.sync().map_err(|e| e.to_string())?;
                        let mut dev_off = vec![0i32; ne + 2];
                        self.ctx.d2h(bytemuck::cast_slice_mut(&mut dev_off), offd as *const u8)?;
                        let mut dev_perm = vec![0u32; rows];
                        self.ctx.d2h(bytemuck::cast_slice_mut(&mut dev_perm), pd as *const u8)?;
                        let mut dev_rowexp = vec![0u32; bound];
                        self.ctx.d2h(bytemuck::cast_slice_mut(&mut dev_rowexp), rxd as *const u8)?;
                        let idp = self.fptr(ids)?;
                        let mut idv = vec![0u32; rows];
                        self.ctx.d2h(bytemuck::cast_slice_mut(&mut idv), idp)?;
                        let mut cnt = vec![0i32; ne];
                        for &e in &idv { cnt[(e as usize).min(ne - 1)] += 1; }
                        let mut hoff = vec![0usize; ne + 1];
                        let mut acc2 = 0;
                        for e in 0..ne { hoff[e] = acc2; acc2 += cnt[e] as usize; }
                        hoff[ne] = acc2;
                        let mut bad = 0;
                        for e in 0..=ne {
                            if dev_off[e] as usize != hoff[e] {
                                eprintln!("# gcheck off[{e}] dev={} host={}", dev_off[e], hoff[e]);
                                bad += 1;
                                if bad > 4 { break; }
                            }
                        }
                        if bad == 0 {
                            let mut cur = hoff[..ne].to_vec();
                            for (i, &e) in idv.iter().enumerate() {
                                let e2 = (e as usize).min(ne - 1);
                                let ppos = cur[e2]; cur[e2] += 1;
                                if dev_perm[ppos] as usize != i { 
                                    eprintln!("# gcheck perm@{ppos} dev={} host={i}", dev_perm[ppos]);
                                    bad += 1;
                                    if bad > 4 { break; }
                                }
                                if dev_rowexp[ppos] as usize != e2 {
                                    eprintln!("# gcheck rowexp@{ppos} dev={} host={e2}", dev_rowexp[ppos]);
                                    bad += 1;
                                    if bad > 4 { break; }
                                }
                            }
                        }
                        eprintln!("# gcheck rows={rows} rows_pad_dev={} bad={bad}", dev_off[ne + 1]);
                    }
                    // 폴백(비 Q4K/Q5_1 타입)용 오프셋. 기본은 비동기로 미리 걸어
                    // 소비 시점(층 하단)까지 gate/up GEMM이 지연을 덮는다.
                    // LLM170_MOE_GROUP_SYNC=1이면 즉시 동기(스트림 드레인) —
                    // 호스트 경로와 같은 순서 조건을 만들어 순서 효과를 검정한다.
                    // 이분법: 비동기 예약 자체를 건너뛴다(폴백은 동기 d2h로).
                    let pinned_off = if std::env::var_os("LLM170_MOE_GROUP_NOD2H").is_some() {
                        std::ptr::null_mut()
                    } else if std::env::var_os("LLM170_MOE_GROUP_SYNC").is_some() {
                        let buf = self.ctx.d2h_issue((ne + 2) * 4, offd as *const u8)?;
                        self.ctx.d2h_wait()?;
                        buf
                    } else {
                        // +4B: 오프셋 뒤에 디바이스가 계산한 rows_pad가 붙어 있다(가드용).
                        self.ctx.d2h_issue((ne + 2) * 4, offd as *const u8)?
                    };
                    let mut c = self.moe_group.lock().map_err(|e| e.to_string())?;
                    *c = Some(MoeGroup {
                        generation, rows, perm_d: pd, inv_d: ivd, rowexp_d: rxd,
                        perm_pad_d: ppd, inv_pad_d: ipd, tilexp_d: txd,
                        rows_pad: bound, rows_pad_d: rpd, off_d: offd, pinned_off, off: Vec::new(),
                    });
                    (pd, ivd, rxd, ppd, ipd, txd, bound, rpd, Vec::new(), offd, pinned_off)
                } else {
                // 그래프 캡처 경계 — 이 블록은 d2h(라우팅 판독)+호스트 정렬+h2d를
                // 하므로 캡처 밖이어야 한다(세그먼트 분할점).
                crate::rawhip::capture_mark(self.ctx.stream, "moe_group_in")?;
                let mut lp = std::time::Instant::now();
                let idp = self.fptr(ids)?;
                let mut idv = vec![0u32; rows];
                self.ctx.d2h(bytemuck::cast_slice_mut(&mut idv), idp)?;
                if tm {
                    let ms = lp.elapsed().as_secs_f64() * 1e3;
                    if ms >= 0.05 { eprintln!("# moe-miss d2h={ms:.2}ms rows={rows}"); }
                    lp = std::time::Instant::now();
                }
                let mut off = vec![0usize; ne + 1];
                for &e in &idv {
                    off[(e as usize).min(ne - 1) + 1] += 1;
                }
                for e in 0..ne {
                    off[e + 1] += off[e];
                }
                let mut cur = off[..ne].to_vec();
                let mut perm = vec![0u32; rows];
                let mut inv = vec![0u32; rows];
                for (i, &e) in idv.iter().enumerate() {
                    let e = (e as usize).min(ne - 1);
                    let p = cur[e];
                    perm[p] = i as u32;
                    inv[i] = p as u32;
                    cur[e] += 1;
                }
                if tm {
                    let ms = lp.elapsed().as_secs_f64() * 1e3;
                    if ms >= 0.05 { eprintln!("# moe-miss sort={ms:.2}ms"); }
                    lp = std::time::Instant::now();
                }
                let (pd, ivd, rxd) = {
                    let mut a = self.rperm.lock().map_err(|e| e.to_string())?;
                    let pd = a.ensure(&self.ctx, rows * 4)? as u64;
                    let mut b = self.rperm2.lock().map_err(|e| e.to_string())?;
                    let ivd = b.ensure(&self.ctx, rows * 4)? as u64;
                    let mut c = self.rexp.lock().map_err(|e| e.to_string())?;
                    let rxd = c.ensure(&self.ctx, rows * 4)? as u64;
                    (pd, ivd, rxd)
                };
                self.ctx.h2d(pd as *mut u8, bytemuck::cast_slice(&perm))?;
                self.ctx.h2d(ivd as *mut u8, bytemuck::cast_slice(&inv))?;
                // rowexp: 순열 후 행 p의 전문가 = idv[perm[p]]
                let mut rowexp = vec![0u32; rows];
                for p in 0..rows {
                    rowexp[p] = idv[(perm[p] as usize).min(rows - 1)].min((ne - 1) as u32);
                }
                self.ctx.h2d(rxd as *mut u8, bytemuck::cast_slice(&rowexp))?;
                let mut off_pad = vec![0usize; ne + 1];
                for e in 0..ne {
                    off_pad[e + 1] = off_pad[e] + (off[e + 1] - off[e]).div_ceil(16) * 16;
                }
                let rows_pad = off_pad[ne].max(16);
                let mut perm_pad = vec![0u32; rows_pad];
                let mut inv_pad = vec![0u32; rows];
                for e in 0..ne {
                    let r = off[e + 1] - off[e];
                    for i in 0..(off_pad[e + 1] - off_pad[e]) {
                        let pd = off_pad[e] + i;
                        if i < r {
                            let src = off[e] + i;
                            perm_pad[pd] = perm[src];
                            inv_pad[perm[src] as usize] = pd as u32;
                        } else {
                            perm_pad[pd] = 0;
                        }
                    }
                }
                let mut tilexp = vec![0u32; rows_pad / 16];
                for e in 0..ne {
                    for tg in off_pad[e] / 16..off_pad[e + 1] / 16 {
                        tilexp[tg] = e as u32;
                    }
                }
                let (ppd, ipd, txd) = {
                    let mut a = self.gp.lock().map_err(|e| e.to_string())?;
                    let ppd = a.ensure(&self.ctx, rows_pad * 4)? as u64;
                    let mut b = self.gi.lock().map_err(|e| e.to_string())?;
                    let ipd = b.ensure(&self.ctx, rows * 4)? as u64;
                    let mut c = self.texp.lock().map_err(|e| e.to_string())?;
                    let txd = c.ensure(&self.ctx, (rows_pad / 16).max(1) * 4)? as u64;
                    (ppd, ipd, txd)
                };
                if std::env::var_os("LLM170_GE5_DBG").is_some() {
                    eprintln!(
                        "# ge5 rows={rows} rows_pad={rows_pad} ne={ne} ppd={ppd} ipd={ipd} txd={txd} \
perm_pad[0..4]={:?} inv_pad[0..4]={:?} tile[0..4]={:?} off[0..4]={:?}",
                        &perm_pad[..perm_pad.len().min(4)],
                        &inv_pad[..inv_pad.len().min(4)],
                        &tilexp[..tilexp.len().min(4)],
                        &off[..off.len().min(4)]
                    );
                }
                self.ctx.h2d(ppd as *mut u8, bytemuck::cast_slice(&perm_pad))?;
                self.ctx.h2d(ipd as *mut u8, bytemuck::cast_slice(&inv_pad))?;
                self.ctx.h2d(txd as *mut u8, bytemuck::cast_slice(&tilexp))?;
                if tm {
                    let ms = lp.elapsed().as_secs_f64() * 1e3;
                    if ms >= 0.05 { eprintln!("# moe-miss h2d={ms:.2}ms"); }
                }
                crate::rawhip::capture_mark(self.ctx.stream, "moe_group_out")?;
                let mut c = self.moe_group.lock().map_err(|e| e.to_string())?;
                *c = Some(MoeGroup { generation, rows, perm_d: pd, inv_d: ivd, rowexp_d: rxd,
                    perm_pad_d: ppd, inv_pad_d: ipd, tilexp_d: txd, rows_pad, rows_pad_d: 0, off_d: 0, pinned_off: std::ptr::null_mut(), off: off.clone() });
                (pd, ivd, rxd, ppd, ipd, txd, rows_pad, 0u64, off, 0u64, std::ptr::null_mut())
                }
            }
        };
        phase("group", &mut lap);
        let row_u32 = if f32w { n_in } else { xq_w };
        // plans/68 레이아웃 실험 플래그 — t=1의 기존(검증된) 동작은 그대로 두고
        // 프리필 디바이스 그룹화 실험에서만 패딩 도메인 레이아웃을 쓴다.
        let pad_layout = rows_pad_d != 0 && self.t_cur() > 1;
        // 디바이스 그룹화 경로의 GEMM은 t = rows_pad로 x를 읽는다(패딩 행의 출력은
        // scatter가 버리므로 값은 무관, 크기만 rows_pad까지 필요).
        let xbuf_rows = if rows_pad_d != 0 { rows + 16 * ne } else { rows };
        let xg = {
            let mut g = self.xperm.lock().map_err(|e| e.to_string())?;
            g.ensure(&self.ctx, xbuf_rows * row_u32 * 4)?
        };
        let yg = {
            let mut g = self.yperm.lock().map_err(|e| e.to_string())?;
            // ★ 5번째 축(plans/68): q4_gemm_q4k_ge는 r < *rows_pad까지
            // out[r·n_out+o]에 기록한다 — 디바이스 그룹화 경로(rows_pad_d≠0)는
            // 패딩 행(≤16·ne)분까지 버퍼를 확보해야 한다. 종전 rows 크기여서
            // 프리필에서 out 끝을 넘는 쓰기 → HIP 700(10차 소거의 정체).
            let ybuf_rows = if rows_pad_d != 0 { rows + 16 * ne } else { rows };
            g.ensure(&self.ctx, ybuf_rows * n_out * 4)?
        };
        let xsrc0 = if f32w { xp } else { xq };
        // plans/68 레이아웃 일관화: 디바이스 그룹화(rows_pad_d≠0)는 GEMM이
        // **패딩 도메인**(r < *rows_pad, rowexp=패딩 인덱스)으로 읽는다 — gather도
        // perm_pad/bound행으로. 호스트 경로는 종전대로 비패딩 perm_d/rows.
        if pad_layout {
            self.rows_permute_dev(xsrc0, perm_pad_d as *mut u8, xg, row_u32, rows + 16 * ne)?;
        } else {
            self.rows_permute_dev(xsrc0, perm_d as *mut u8, xg, row_u32, rows)?;
        }
        phase("gather", &mut lap);
        if llm170_core::qwen4exp::frame::stage_skipped("moe") {
            // 진단용(LLM170_STAGE_SKIP=moe): 전문가 GEMM 생략 — 비용 분해, 출력 무효.
            return Ok(());
        }
        // 그룹 런치(옵트인) — q4_K 전문가를 한 번에: 청크당 런치 7.4만 → 48.
        // 호스트/갭 ~2.7초@pp2048 제거(KTRACE 실측). 산술은 _m과 동일(비트 동일).
        // 기본 경로 — 비트 동일(토큰 검증), pp2048 −1.8%, 런치 7.4만→48/청크.
        // q5_1(다운) 그룹판 — 16배수 패딩 레이아웃으로 타일=전문가, 가중치 재독 1회.
        if ws.ty == GgmlType::Q5_1 && !f32w && rows > 0 {
            // direct-ids(t=1): down도 그룹화 없이 — 게이트/up만 direct로는 down에서
            // 그룹화(d2h+빌드+h2d)가 1회 발생해 이득이 사라진다.
            // 실측(2026-09-14): down은 72 GB/s로 게이트/up의 180에 크게 못 미친다.
            // 원인은 워프가 출력행마다 480B 스트라이드로 읽어 32B 섹터당 4B만
            // 쓰는 8배 증폭. gm의 협조 적재 이식은 **불가능**하다(그 불변식은
            // 타일당 단일 전문가인데 비정렬 direct는 행마다 전문가가 다르다 —
            // o-행 하나에 16전문가 가중치가 필요해 공유 버퍼로 표현 불가, 시도 후 복원).
            // 워프-퍼-행 재설계도 q5_1의 6워드 슈퍼블록 입도 때문에 3배가 한계였다.
            // 즉 6ms는 Q5_1 레이아웃 고유 비용이다.
            if self.t_cur() == 1 && std::env::var_os("LLM170_MOE_GROUPED").is_none() {
                let idp = self.fptr(ids)?;
                let mut x_p = xq as *mut std::ffi::c_void;
                let mut w_p = wd as *mut std::ffi::c_void;
                let mut part_p = self.ctx.scratch(4)? as *mut std::ffi::c_void;
                let mut o_p = self.fptr(out)? as *mut std::ffi::c_void;
                let mut ip = idp as *mut std::ffi::c_void;
                let (mut ni, mut no) = (n_in as i32, n_out as i32);
                let (mut xw, mut tt) = (xq_w as i32, rows as i32);
                let mut ew = (per_expert / 4) as i32;
                let mut args: Vec<*mut std::ffi::c_void> = vec![
                    (&mut x_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut w_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut part_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut ip) as *mut _ as *mut std::ffi::c_void,
                    (&mut ni) as *mut _ as *mut std::ffi::c_void,
                    (&mut no) as *mut _ as *mut std::ffi::c_void,
                    (&mut xw) as *mut _ as *mut std::ffi::c_void,
                    (&mut tt) as *mut _ as *mut std::ffi::c_void,
                    (&mut ew) as *mut _ as *mut std::ffi::c_void,
                ];
                self.ctx.launch3(
                    "q4_gemm_q5_1_gm_ids",
                    n_out.div_ceil(16) as u32,
                    rows.div_ceil(16) as u32,
                    1,
                    256,
                    &mut args,
                )?;
                return Ok(());
            }
            let xgp = {
                let mut g = self.gxp.lock().map_err(|e| e.to_string())?;
                g.ensure(&self.ctx, rows_pad * xq_w * 4)?
            };
            self.rows_permute_dev(xq as *mut u8, perm_pad_d as *mut u8, xgp, xq_w, rows_pad)?;
            let ygp = {
                let mut g = self.gyp.lock().map_err(|e| e.to_string())?;
                g.ensure(&self.ctx, rows_pad * n_out * 4)?
            };
            {
                let mut part_p = self.ctx.scratch(4)? as *mut std::ffi::c_void;
                let mut x_p = xgp as *mut std::ffi::c_void;
                let mut w_p = wd as *mut std::ffi::c_void;
                let mut o_p = ygp as *mut std::ffi::c_void;
                let mut tx_p = tilexp_d as *mut std::ffi::c_void;
                let (mut ni, mut no, mut xw, mut tt, mut ew) = (
                    n_in as i32,
                    n_out as i32,
                    xq_w as i32,
                    rows_pad as i32,
                    (per_expert / 4) as i32,
                );
                let mut args: Vec<*mut std::ffi::c_void> = vec![
                    (&mut x_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut w_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut part_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut tx_p) as *mut _ as *mut std::ffi::c_void,
                    (&mut ni) as *mut _ as *mut std::ffi::c_void,
                    (&mut no) as *mut _ as *mut std::ffi::c_void,
                    (&mut xw) as *mut _ as *mut std::ffi::c_void,
                    (&mut tt) as *mut _ as *mut std::ffi::c_void,
                    (&mut ew) as *mut _ as *mut std::ffi::c_void,
                ];
                let smem = (16 * (n_in / 32) * 24) as u32;
                self.ctx.launch3_dyn(
                    "q4_gemm_q5_1_gm",
                    n_out.div_ceil(16).min(65535) as u32,
                    rows_pad.div_ceil(16) as u32,
                    1,
                    256,
                    smem,
                    &mut args,
                )?;
            }
            self.rows_permute_dev(ygp, inv_pad_d as *mut u8, op_, n_out, rows)?;
            if tm {
                eprintln!("# moe-phase TOTAL={:.2}ms rows={rows} ge5", t0.elapsed().as_secs_f64() * 1e3);
            }
            return Ok(());
        }
        if ws.ty == GgmlType::Q4K && !f32w && rows > 0 {
            let mut part_p = self.ctx.scratch(4)? as *mut std::ffi::c_void;
            let mut x_p = xg as *mut std::ffi::c_void;
            let mut w_p = wd as *mut std::ffi::c_void;
            let mut o_p = yg as *mut std::ffi::c_void;
            let mut rx_p = rowexp_d as *mut std::ffi::c_void;
            // 디바이스 그룹화 경로면 rows_pad를 커널이 디바이스에서 읽는다(가드).
            let mut rpd_p = rows_pad_d as *mut u8;
            let (mut ni, mut no, mut xw, mut tt, mut eb) = (
                n_in as i32,
                n_out as i32,
                xq_w as i32,
                rows as i32,
                per_expert as i32,
            );
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut x_p) as *mut _ as *mut std::ffi::c_void,
                (&mut w_p) as *mut _ as *mut std::ffi::c_void,
                (&mut part_p) as *mut _ as *mut std::ffi::c_void,
                (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                (&mut rx_p) as *mut _ as *mut std::ffi::c_void,
                (&mut ni) as *mut _ as *mut std::ffi::c_void,
                (&mut no) as *mut _ as *mut std::ffi::c_void,
                (&mut xw) as *mut _ as *mut std::ffi::c_void,
                (&mut tt) as *mut _ as *mut std::ffi::c_void,
                (&mut eb) as *mut _ as *mut std::ffi::c_void,
                (&mut rpd_p) as *mut _ as *mut std::ffi::c_void,
            ];
            {
                use std::sync::Mutex;
                use std::sync::OnceLock;
                static SEEN: OnceLock<Mutex<Vec<(usize, usize, usize)>>> = OnceLock::new();
                if std::env::var_os("LLM170_Q4_DBG").is_some() {
                    let seen = SEEN.get_or_init(|| Mutex::new(Vec::new()));
                    if let Ok(mut v) = seen.lock() {
                        let key = (n_in, n_out, rows);
                        if !v.contains(&key) && v.len() < 8 {
                            v.push(key);
                            eprintln!(
                                "# q4_gemm_q4k_ge: n_in={n_in} n_out={n_out} rows={rows} blocks={}x{}",
                                n_out.div_ceil(16),
                                rows.div_ceil(16)
                            );
                        }
                    }
                }
            }
            self.ctx.launch3(
                "q4_gemm_q4k_ge",
                n_out.div_ceil(16) as u32,
                rows_pad.div_ceil(16) as u32,
                1,
                256,
                &mut args,
            )?;
            let scat = if pad_layout { inv_pad_d } else { inv_d };
            self.rows_permute_dev(yg, scat as *mut u8, op_, n_out, rows)?;
            phase("scatter", &mut lap);
            if tm {
                eprintln!("# moe-phase TOTAL={:.2}ms rows={rows}", t0.elapsed().as_secs_f64() * 1e3);
            }
            self.moe_hash_check("ge", op_, rows, n_out)?;
            return Ok(());
        }
        // 디바이스 그룹화 경로(off 비어 있음): 폴백(비 Q4K/Q5_1 타입, 예: down의
        // q8_0)은 전문가별 런치를 위해 오프셋만 읽는다 — 그룹화·테이블 빌드·업로드
        // 왕복은 GPU가 이미 끝냈으므로 여기서는 (ne+1)개 int만 받는다.
        let mut off_d2h;
        // plans/68: 디바이스 그룹화 경로의 xg는 **패딩 도메인** — 폴백(전문가별
        // 런치)도 패딩 오프셋(off_pad)에서 구간을 읽어야 행이 맞는다.
        let off: &[usize] = if off.is_empty() && rows_pad_d != 0 {
            self.ctx.d2h_wait()?;
            // 오프셋 + 그 뒤 4B(디바이스가 계산한 rows_pad)를 함께 읽어 **상한을
            // 검증**한다. 초과하면 크래시(h2d 700) 대신 진단 메시지로 실패시킨다 —
            // 2026-09-14에 상한 가정이 5곳에 흩어져 있어 디버깅이 오래 걸렸다.
            let mut b = vec![0i32; ne + 2];
            if !pinned_off.is_null() {
                unsafe {
                    std::ptr::copy_nonoverlapping(pinned_off as *const u8, b.as_mut_ptr() as *mut u8, (ne + 2) * 4);
                }
            } else {
                self.ctx.d2h(bytemuck::cast_slice_mut(&mut b), off_d as *const u8)?;
            }
            let _ = &b;
            let rows_pad_dev = b[ne + 1].max(0) as usize;
            let bound = self.t_cur() * k_sel.max(1) + 16 * ne;
            if std::env::var_os("LLM170_MOE_BCHECK").is_some() {
                // b(pinned off) 무결성 — r 오염(gemm_q5k gx=1.04억)의 원본 관찰.
                let mut mono_ok = true;
                for i in 0..ne {
                    if b[i] > b[i + 1] { mono_ok = false; break; }
                }
                let total = b[ne];
                if !mono_ok || total < 0 || total as usize > bound || b[..ne.min(8)].iter().any(|&x| x < 0) {
                    eprintln!(
                        "# bcheck BAD rows={rows} ne={ne} mono={mono_ok} total={total} bound={bound} b0..7={:?} rp={}",
                        &b[..8.min(ne)], b[ne + 1]
                    );
                }
            }
            if rows_pad_dev > bound {
                return Err(format!(
                    "moe 그룹화: rows_pad {rows_pad_dev} > bound {bound} (ne={ne}) — 상한 가정 위반"
                ));
            }
            if pad_layout {
                // [프리필 실험] 시작점은 패딩 도메인, 행 수는 실제 카운트 — 패딩
                // 행을 타일 GEMM에 넘기면 블록 단위 처리가 실제 행 결과를 흔든다
                // (해시 국소화로 확인, 2026-09-14). starts = Σ ceil16(cnt).
                let mut starts = vec![0usize; ne + 1];
                let mut accp2 = 0usize;
                for e in 0..ne {
                    starts[e] = accp2;
                    let c = (b[e + 1].max(0) as usize).saturating_sub(b[e].max(0) as usize);
                    accp2 += c.div_ceil(16) * 16;
                }
                starts[ne] = accp2;
                off_d2h = vec![0usize; ne + 1];
                for e in 0..ne {
                    off_d2h[e] = starts[e];
                    off_d2h[e + 1] = starts[e]
                        + (b[e + 1].max(0) as usize).saturating_sub(b[e].max(0) as usize);
                }
            } else {
                // t=1 종전 동작: 비패딩 off를 그대로(gather도 perm_d라 일관).
                off_d2h = b[..ne + 1].iter().map(|&x| x.max(0) as usize).collect();
            }
            &off_d2h
        } else {
            &off
        };
        for e in 0..ne {
            let r = off[e + 1] - off[e];
            if r == 0 {
                continue;
            }
            let start = off[e];
            let xsrc = unsafe { xg.add(start * row_u32 * 4) };
            let wsrc = unsafe { wd.add(e * per_expert) };
            let dst = unsafe { yg.add(start * n_out * 4) };
            if f32w {
                self.launch_gemm_f32(xsrc, wsrc, n_in, n_out, r, dst)?;
            } else {
                self.launch_gemm(ggml_id(ws.ty), xsrc, wsrc, n_in, n_out, xq_w, r, dst)?;
            }
        }
        phase("gemms", &mut lap);
        let scat2 = if pad_layout { inv_pad_d } else { inv_d };
        self.rows_permute_dev(yg, scat2 as *mut u8, op_, n_out, rows)?;
        phase("scatter", &mut lap);
        if tm {
            eprintln!("# moe-phase TOTAL={:.2}ms rows={rows}", t0.elapsed().as_secs_f64() * 1e3);
        }
        self.moe_hash_check("fb", op_, rows, n_out)?;
        Ok(())
    }


}

impl Q4Acc {
    /// q4_qsa_attn_sel 런치 본체 — 선택 목록(오름차순 위치)만 순회한다.
    #[allow(clippy::too_many_arguments)]
    pub fn qsa_attn_sel_raw(
        &self,
        q: &[f32],
        ck: &[f32],
        cv: &[f32],
        sel_idx: &[u32],
        sel_off: &[u32],
        kq_scale: f32,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
    ) -> Result<Vec<f32>, String> {
        let (qdev, kdev, vdev, sdev, odev, ofdev) = {
            let mut a = self.qs.lock().map_err(|e| e.to_string())?;
            let qdev = a.ensure(&self.ctx, q.len() * 4)?;
            // KV는 컨텍스트 전체를 미리 잡는다(엔진이 주입한 ctx_len). 종전에는
            // n_past가 늘 때마다 재할당해 매 스텝 주소가 바뀌었다(실측 48회/세션).
            let kv_floats = self.ctx_len.load(std::sync::atomic::Ordering::Relaxed)
                * n_kv.max(1) * hd.max(1);
            let mut b = self.ckv.lock().map_err(|e| e.to_string())?;
            let kdev = b.ensure(&self.ctx, ck.len().max(kv_floats) * 4)?;
            let mut c = self.cvv.lock().map_err(|e| e.to_string())?;
            let vdev = c.ensure(&self.ctx, cv.len().max(kv_floats) * 4)?;
            let mut d = self.msk.lock().map_err(|e| e.to_string())?;
            let sdev = d.ensure(&self.ctx, sel_idx.len().max(1) * 4)?;
            let mut e2 = self.soff.lock().map_err(|e| e.to_string())?;
            let ofdev = e2.ensure(&self.ctx, sel_off.len().max(1) * 4)?;
            let mut f2 = self.atn.lock().map_err(|e| e.to_string())?;
            let odev = f2.ensure(&self.ctx, t * n_head * hd * 4)?;
            (qdev, kdev, vdev, sdev, odev, ofdev)
        };
        // 주의(실측 2026-09-14): 아래 h2d는 **매 호출 KV 캐시 전체**를 올린다.
        // n_past 8192에서 8192 x 2 x 256 x 4B x 2(K,V) = 33.6MB/층, 12층이면
        // 403MB/스텝이고 d2h 대역 실측(17.6 GB/s)으로 ~23ms/스텝 = 장문맥 스텝의 16%다.
        // 컨텍스트 스케일링 실측이 이를 지지한다: pp2048 103.9ms/스텝 ->
        // pp8192 141.8ms/스텝(+37.9)이고 KV 증가분만 302MB/스텝 ~17ms(증가의 45%)다.
        // 정공법은 KV를 디바이스 상주로 두고 증가분만 올리는 것이다. 다만
        // (ptr, len) 기반 델타 캐시는 **정확하지 않다**: 새 시퀀스의 첫 청크가
        // 이전 캐시 길이보다 길면 len이 커져 델타 경로로 빠지고 낡은 접두가 남는다
        // (짧은 시퀀스 1024 뒤에 긴 시퀀스가 2048로 시작하는 경우). 경계 내용
        // 비교도 동일 내용이면 통과해 버린다. 따라서 **명시적 리셋 신호**가 필요하다:
        // 스테이지는 pos0을 알고 있으므로 qsa_attention_sel 시그니처에 pos0을 넣거나
        // 프레임 begin에서 리셋을 알리는 것이 최소 변경이다(트레이트+CPU 폴백 수정).
        // (_sel4_raw도 같은 블록을 쓴다. 서버 다중 시퀀스·프롬프트 교체 검증 필수.)
        self.ctx.h2d(qdev, bytemuck::cast_slice(q))?;
        self.ctx.h2d(kdev, bytemuck::cast_slice(ck))?;
        self.ctx.h2d(vdev, bytemuck::cast_slice(cv))?;
        self.ctx.h2d(sdev, bytemuck::cast_slice(sel_idx))?;
        self.ctx.h2d(ofdev, bytemuck::cast_slice(sel_off))?;
        let mut q_p = qdev as *mut std::ffi::c_void;
        let mut k_p = kdev as *mut std::ffi::c_void;
        let mut v_p = vdev as *mut std::ffi::c_void;
        let mut si_p = sdev as *mut std::ffi::c_void;
        let mut so_p = ofdev as *mut std::ffi::c_void;
        let mut o_p = odev as *mut std::ffi::c_void;
        let mut sc = kq_scale;
        let mut nh = n_head as i32;
        let mut nk = n_kv as i32;
        let mut h = hd as i32;
        let mut tt = t as i32;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&mut q_p) as *mut _ as *mut std::ffi::c_void,
            (&mut k_p) as *mut _ as *mut std::ffi::c_void,
            (&mut v_p) as *mut _ as *mut std::ffi::c_void,
            (&mut si_p) as *mut _ as *mut std::ffi::c_void,
            (&mut so_p) as *mut _ as *mut std::ffi::c_void,
            (&mut o_p) as *mut _ as *mut std::ffi::c_void,
            (&mut sc) as *mut _ as *mut std::ffi::c_void,
            (&mut nh) as *mut _ as *mut std::ffi::c_void,
            (&mut nk) as *mut _ as *mut std::ffi::c_void,
            (&mut h) as *mut _ as *mut std::ffi::c_void,
            (&mut tt) as *mut _ as *mut std::ffi::c_void,
        ];
        // 블록 16워프 = 16토큰(워프당 1헤드) — 목록만 순회하는 기본판.
        self.ctx.launch3(
            "q4_qsa_attn_sel",
            t.div_ceil(16) as u32,
            n_head as u32,
            1,
            512,
            &mut args,
        )?;
        let mut out = vec![0.0f32; t * n_head * hd];
        self.ctx.d2h(bytemuck::cast_slice_mut(&mut out), odev)?;
        Ok(out)
    }

    /// q4_qsa_attn_sel4 런치 본체 — 선택 목록(오름차순 위치)만 순회한다.
    #[allow(clippy::too_many_arguments)]
    pub fn qsa_attn_sel4_raw(
        &self,
        q: &[f32],
        ck: &[f32],
        cv: &[f32],
        sel_idx: &[u32],
        sel_off: &[u32],
        kq_scale: f32,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
    ) -> Result<Vec<f32>, String> {
        let (qdev, kdev, vdev, sdev, odev, ofdev) = {
            let mut a = self.qs.lock().map_err(|e| e.to_string())?;
            let qdev = a.ensure(&self.ctx, q.len() * 4)?;
            // KV는 컨텍스트 전체를 미리 잡는다(엔진이 주입한 ctx_len). 종전에는
            // n_past가 늘 때마다 재할당해 매 스텝 주소가 바뀌었다(실측 48회/세션).
            let kv_floats = self.ctx_len.load(std::sync::atomic::Ordering::Relaxed)
                * n_kv.max(1) * hd.max(1);
            let mut b = self.ckv.lock().map_err(|e| e.to_string())?;
            let kdev = b.ensure(&self.ctx, ck.len().max(kv_floats) * 4)?;
            let mut c = self.cvv.lock().map_err(|e| e.to_string())?;
            let vdev = c.ensure(&self.ctx, cv.len().max(kv_floats) * 4)?;
            let mut d = self.msk.lock().map_err(|e| e.to_string())?;
            let sdev = d.ensure(&self.ctx, sel_idx.len().max(1) * 4)?;
            let mut e2 = self.soff.lock().map_err(|e| e.to_string())?;
            let ofdev = e2.ensure(&self.ctx, sel_off.len().max(1) * 4)?;
            let mut f2 = self.atn.lock().map_err(|e| e.to_string())?;
            let odev = f2.ensure(&self.ctx, t * n_head * hd * 4)?;
            (qdev, kdev, vdev, sdev, odev, ofdev)
        };
        self.ctx.h2d(qdev, bytemuck::cast_slice(q))?;
        self.ctx.h2d(kdev, bytemuck::cast_slice(ck))?;
        self.ctx.h2d(vdev, bytemuck::cast_slice(cv))?;
        self.ctx.h2d(sdev, bytemuck::cast_slice(sel_idx))?;
        self.ctx.h2d(ofdev, bytemuck::cast_slice(sel_off))?;
        let mut q_p = qdev as *mut std::ffi::c_void;
        let mut k_p = kdev as *mut std::ffi::c_void;
        let mut v_p = vdev as *mut std::ffi::c_void;
        let mut si_p = sdev as *mut std::ffi::c_void;
        let mut so_p = ofdev as *mut std::ffi::c_void;
        let mut o_p = odev as *mut std::ffi::c_void;
        let mut sc = kq_scale;
        let mut nh = n_head as i32;
        let mut nk = n_kv as i32;
        let mut h = hd as i32;
        let mut tt = t as i32;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&mut q_p) as *mut _ as *mut std::ffi::c_void,
            (&mut k_p) as *mut _ as *mut std::ffi::c_void,
            (&mut v_p) as *mut _ as *mut std::ffi::c_void,
            (&mut si_p) as *mut _ as *mut std::ffi::c_void,
            (&mut so_p) as *mut _ as *mut std::ffi::c_void,
            (&mut o_p) as *mut _ as *mut std::ffi::c_void,
            (&mut sc) as *mut _ as *mut std::ffi::c_void,
            (&mut nh) as *mut _ as *mut std::ffi::c_void,
            (&mut nk) as *mut _ as *mut std::ffi::c_void,
            (&mut h) as *mut _ as *mut std::ffi::c_void,
            (&mut tt) as *mut _ as *mut std::ffi::c_void,
        ];
        // 8워프 = 4토큰 × 2헤드묶음. 묶음당 헤드 수는 6이 기본(2026-09-14):
        // 게이트를 레지스터에서 빼면 qr[6][8]+acc[6][8]=96으로 4헤드판과 같은
        // 예산이라 K/V 행 재독이 6회 -> 4회로 준다(프리필 어텐션이 대역폭 바운드:
        // t=2048 콜당 ~34GB/236GB/s ~= 실측 101ms). 12의 배수가 아니면 4헤드판.
        let use6 = n_head % 12 == 0 && std::env::var("LLM170_QSA_H6").as_deref() != Ok("0");
        let (kern, gy) = if use6 {
            ("q4_qsa_attn_sel6", (n_head / 12) as u32)
        } else {
            ("q4_qsa_attn_sel4", (n_head / 8) as u32)
        };
        self.ctx.launch3(kern, t.div_ceil(4) as u32, gy, 1, 256, &mut args)?;
        let mut out = vec![0.0f32; t * n_head * hd];
        self.ctx.d2h(bytemuck::cast_slice_mut(&mut out), odev)?;
        Ok(out)
    }

    /// plans/67 1단계: **디바이스 q판** — q가 wq의 frame_mm_group 출력(디바이스)에
    /// 이미 있을 때 h2d 없이 어텐션을 돌고 결과를 디바이스 out에 쓴다(d2h도 없음).
    /// k/v는 기존 풀 업로드 경로(실측: KV 업로드는 유의미한 비용이 아님).
    /// QSA KV 상주 풀 — ctx_len 전체를 선할당(주소 안정성: ensure 재할당이
    /// 어텐션 커널에 전달된 포인터를 무효화하지 않게 1회 확정). k/v 행은
    /// D2D로 append(왕복 0). 반환 핸들 = 풀 포인터(디바이스 주소).
    fn qsa_kv_dev_impl(
        &self,
        full_idx: usize,
        seq: usize,
        k: u64,
        v: u64,
        t: usize,
        pos0: usize,
        n_kv: usize,
        hd: usize,
    ) -> Result<(u64, u64), String> {
        let ctx_len = self.ctx_len.load(std::sync::atomic::Ordering::Relaxed);
        if ctx_len == 0 {
            return Err("qsa_kv_dev: ctx_len 미주입".into());
        }
        let bytes = ctx_len * n_kv * hd * 4;
        // 워터마크 — 이 풀에 적립된 다음 위치. 규칙:
        //   pos0 == w: 정상 순차 적립.
        //   pos0 <  w: **되감기** — 위치 p의 k/v는 (토큰 접두어, p)의 결정 함수라
        //              접두어가 불변인 되감기(벤치 워밍업 후 재시작, 스펙 롤백,
        //              슬롯 재프리필)에서 [0, pos0)의 기존 값과 새 값이 동일하다.
        //              재구축도 순서대로 돌아 과거 청크가 이번 재구축분을 덮는다.
        //   pos0 >  w: 구멍(값 경로 청크 등) — 읽을 수 없으니 업로드 경로로 폴백.
        {
            let mut wm = self.qsa_kv_pos.lock().map_err(|e| e.to_string())?;
            let w = wm.entry((full_idx, seq)).or_insert(0);
            if pos0 > *w {
                return Err(format!(
                    "qsa_kv_dev: 워터마크 구멍 w={w} pos0={pos0} — 업로드 경로로 폴백"
                ));
            }
            *w = pos0 + t;
        }
        let mut m = self.qsa_kv.lock().map_err(|e| e.to_string())?;
        let ent = m
            .entry((full_idx, seq))
            .or_insert_with(|| (GBuf::new("qsakv_k"), GBuf::new("qsakv_v")));
        if ent.0.bytes < bytes {
            ent.0.ensure(&self.ctx, bytes)?;
            ent.1.ensure(&self.ctx, bytes)?;
        }
        let (kp, vp) = (ent.0.ptr, ent.1.ptr);
        let rows = t * n_kv * hd * 4;
        let ksrc = self.fptr(k)?;
        let vsrc = self.fptr(v)?;
        self.ctx
            .d2d(unsafe { kp.add(pos0 * n_kv * hd * 4) }, ksrc, rows)?;
        self.ctx
            .d2d(unsafe { vp.add(pos0 * n_kv * hd * 4) }, vsrc, rows)?;
        Ok((kp as u64, vp as u64))
    }

    /// plans/73 공용: ik를 idx 풀에 적립하고 완성 블록의 블록키를 증분 갱신한다.
    /// 소스가 디바이스(디코드, f.qsa_ik)면 d2d, 호스트(프리필 청크)면 h2d.
    /// 워터마크 규약은 qsa_kv_dev_impl과 동일(순차 적립/접두어 되감기 허용).
    #[allow(clippy::too_many_arguments)]
    fn qsa_idx_append(
        &self,
        full_idx: usize,
        seq: usize,
        ik_dev: *const u8,
        ik_host: &[f32],
        t: usize,
        pos0: usize,
        idx_dim: usize,
        r: usize,
        ikw: &[f32],
        cs_idx: &[f32],
        eps: f32,
    ) -> Result<(*mut u8, *mut u8), String> {
        let ctx_len = self.ctx_len.load(std::sync::atomic::Ordering::Relaxed);
        if ctx_len == 0 {
            return Err("qsa_idx_append: ctx_len 미주입".into());
        }
        if r == 0 || idx_dim != 128 {
            return Err(format!("qsa_idx_append: 미지원 형상 r={r} idx_dim={idx_dim}"));
        }
        {
            let mut wm = self.qsa_idx_pos.lock().map_err(|e| e.to_string())?;
            let w = wm.entry((full_idx, seq)).or_insert(0);
            if pos0 > *w {
                return Err(format!("qsa_idx_append: 워터마크 구멍 w={w} pos0={pos0}"));
            }
            *w = pos0 + t;
        }
        let nb_max = ctx_len / r + 1;
        let (idxk_p, bk_p) = {
            let mut m = self.qsa_idxk.lock().map_err(|e| e.to_string())?;
            let idxk = m
                .entry((full_idx, seq))
                .or_insert_with(|| GBuf::new("qsa_idxk"));
            idxk.ensure(&self.ctx, ctx_len * idx_dim * 4)?;
            let mut b = self.qsa_bk.lock().map_err(|e| e.to_string())?;
            let bk = b
                .entry((full_idx, seq))
                .or_insert_with(|| GBuf::new("qsa_bk"));
            bk.ensure(&self.ctx, nb_max * idx_dim * 4)?;
            (idxk.ptr, bk.ptr)
        };
        if !ik_dev.is_null() {
            self.ctx.d2d(
                unsafe { idxk_p.add(pos0 * idx_dim * 4) },
                ik_dev,
                t * idx_dim * 4,
            )?;
        } else {
            // hk: h2d는 동기 API — 프리필 청크(드문 경로)라 비용 무의미.
            self.ctx.h2d(
                unsafe { idxk_p.add(pos0 * idx_dim * 4) },
                bytemuck::cast_slice(&ik_host[..t * idx_dim]),
            )?;
        }
        let b0 = pos0 / r;
        let b1 = (pos0 + t) / r;
        if b1 > b0 {
            let ikw_d = self.upload_hashed(&self.qsa_ikw, ikw)?;
            let cs_d = self.upload_by_ptr(&self.qsa_csidx, cs_idx)?;
            let (mut ikp, mut bkp, mut iw, mut cp) = (
                idxk_p as *mut std::ffi::c_void,
                bk_p as *mut std::ffi::c_void,
                ikw_d as *mut std::ffi::c_void,
                cs_d as *mut std::ffi::c_void,
            );
            let (mut e, mut bb0, mut rr, mut dd) =
                (eps, b0 as i32, r as i32, idx_dim as i32);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut ikp) as *mut _ as *mut std::ffi::c_void,
                (&mut bkp) as *mut _ as *mut std::ffi::c_void,
                (&mut iw) as *mut _ as *mut std::ffi::c_void,
                (&mut cp) as *mut _ as *mut std::ffi::c_void,
                (&mut e) as *mut _ as *mut std::ffi::c_void,
                (&mut bb0) as *mut _ as *mut std::ffi::c_void,
                (&mut rr) as *mut _ as *mut std::ffi::c_void,
                (&mut dd) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx
                .launch3("q4_idx_bk_update", (b1 - b0) as u32, 1, 1, 32, &mut args)?;
        }
        Ok((idxk_p, bk_p))
    }

    /// 소형 상수 업로드 캐시 — 내용 FNV 해시가 같으면 재업로드 생략(매 스텝
    /// h2d+sync를 낳던 qn/kn/iqw/ikw 류 제거). 반환 = 디바이스 포인터.
    fn upload_hashed(&self, slot: &std::sync::Mutex<(u64, GBuf)>, data: &[f32]) -> Result<*mut u8, String> {
        let h = fnv_hash(data);
        let mut g = slot.lock().map_err(|e| e.to_string())?;
        if g.0 != h || g.1.ptr.is_null() {
            g.1.ensure(&self.ctx, data.len().max(1) * 4)?;
            self.ctx.h2d(g.1.ptr, bytemuck::cast_slice(data))?;
            g.0 = h;
        }
        Ok(g.1.ptr)
    }

    /// (ptr,len) 키 다중 엔트리 업로드 캐시 — 층별로 다른 상수를 상주시킨다.
    /// 단일 슬롯이면 층마다 미스해 매층 동기 h2d가 발생한다(실측 3.4ms/층).
    fn upload_map(
        &self,
        map: &std::sync::Mutex<std::collections::HashMap<(u64, usize), GBuf>>,
        name: &'static str,
        data: &[f32],
    ) -> Result<*mut u8, String> {
        let key = (data.as_ptr() as u64, data.len());
        let mut m = map.lock().map_err(|e| e.to_string())?;
        if let Some(b) = m.get(&key) {
            return Ok(b.ptr);
        }
        // 상한: 층 수 × 소수 항목이면 충분하다 — 넘치면 비운다(재업로드 비용 < 무한 증가).
        if m.len() >= 64 {
            m.clear();
        }
        let mut b = GBuf::new(name);
        b.ensure(&self.ctx, data.len().max(1) * 4)?;
        self.ctx.h2d(b.ptr, bytemuck::cast_slice(data))?;
        let p = b.ptr;
        m.insert(key, b);
        Ok(p)
    }

    /// 대형 상수(cs 테이블) 업로드 캐시 — (ptr, len) 키. 프레임 필드 벡터는
    /// 스텝 사이 포인터가 안정적이라 해시(4MB)보다 저렴하다.
    fn upload_by_ptr(
        &self,
        slot: &std::sync::Mutex<(usize, usize, GBuf)>,
        data: &[f32],
    ) -> Result<*mut u8, String> {
        let key = (data.as_ptr() as usize, data.len());
        let mut g = slot.lock().map_err(|e| e.to_string())?;
        if g.0 != key.0 || g.1 != key.1 || g.2.ptr.is_null() {
            g.2.ensure(&self.ctx, data.len().max(1) * 4)?;
            self.ctx.h2d(g.2.ptr, bytemuck::cast_slice(data))?;
            (g.0, g.1) = (key.0, key.1);
        }
        Ok(g.2.ptr)
    }

    /// 상주 캐시판 어텐션 — ck/cv가 디바이스 주소(업로드 없음). 커널 선택은
    /// qsa_attention_dev와 동일(t=1 분할 우선).
    fn qsa_attn_res(
        &self,
        q: u64,
        ckp: u64,
        cvp: u64,
        sel_idx: &[u32],
        sel_off: &[u32],
        kq_scale: f32,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
        out: u64,
    ) -> Result<(), String> {
        let use_split = t == 1 && std::env::var("LLM170_QSA_SPLIT").as_deref() != Ok("0");
        let (sdev, ofdev, pdev) = {
            let mut d = self.msk.lock().map_err(|e| e.to_string())?;
            let sdev = d.ensure(&self.ctx, sel_idx.len().max(1) * 4)?;
            let mut e2 = self.soff.lock().map_err(|e| e.to_string())?;
            let ofdev = e2.ensure(&self.ctx, sel_off.len().max(1) * 4)?;
            let n_splits = if use_split {
                let list_len = sel_off
                    .get(1)
                    .copied()
                    .unwrap_or(0)
                    .saturating_sub(sel_off.first().copied().unwrap_or(0)) as usize;
                let cap = std::env::var("LLM170_QSA_SPLITS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(64);
                (list_len / 32).clamp(1, cap.max(1).min(512))
            } else {
                1
            };
            let mut g = self.qsp.lock().map_err(|e| e.to_string())?;
            let pdev = g.ensure(&self.ctx, n_head * n_splits * 32 * 10 * 4)?;
            (sdev, ofdev, pdev)
        };
        self.ctx.h2d(sdev, bytemuck::cast_slice(sel_idx))?;
        self.ctx.h2d(ofdev, bytemuck::cast_slice(sel_off))?;
        let qdev = self.fptr(q)?;
        let odev = self.fptr(out)?;
        if use_split {
            let list_len = sel_off
                .get(1)
                .copied()
                .unwrap_or(0)
                .saturating_sub(sel_off.first().copied().unwrap_or(0)) as usize;
            let cap = std::env::var("LLM170_QSA_SPLITS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(64);
            let n_splits: usize = (list_len / 32).clamp(1, cap.max(1).min(512));
            let mut q_p = qdev as *mut std::ffi::c_void;
            let mut k_p = ckp as *mut std::ffi::c_void;
            let mut v_p = cvp as *mut std::ffi::c_void;
            let mut si_p = sdev as *mut std::ffi::c_void;
            let mut so_p = ofdev as *mut std::ffi::c_void;
            let mut pa_p = pdev as *mut std::ffi::c_void;
            let mut ns = n_splits as i32;
            let mut sc = kq_scale;
            let mut nh = n_head as i32;
            let mut nk = n_kv as i32;
            let mut h = hd as i32;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut q_p) as *mut _ as *mut std::ffi::c_void,
                (&mut k_p) as *mut _ as *mut std::ffi::c_void,
                (&mut v_p) as *mut _ as *mut std::ffi::c_void,
                (&mut si_p) as *mut _ as *mut std::ffi::c_void,
                (&mut so_p) as *mut _ as *mut std::ffi::c_void,
                (&mut pa_p) as *mut _ as *mut std::ffi::c_void,
                (&mut ns) as *mut _ as *mut std::ffi::c_void,
                (&mut sc) as *mut _ as *mut std::ffi::c_void,
                (&mut nh) as *mut _ as *mut std::ffi::c_void,
                (&mut nk) as *mut _ as *mut std::ffi::c_void,
                (&mut h) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx.launch3(
                "q4_qsa_attn_sel4s",
                n_splits.div_ceil(4) as u32,
                (n_head / 12) as u32,
                1,
                256,
                &mut args,
            )?;
            let mut pa_p = pdev as *mut std::ffi::c_void;
            let mut q_p = qdev as *mut std::ffi::c_void;
            let mut o_p = odev as *mut std::ffi::c_void;
            let mut ns = n_splits as i32;
            let mut nh = n_head as i32;
            let mut h = hd as i32;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut pa_p) as *mut _ as *mut std::ffi::c_void,
                (&mut q_p) as *mut _ as *mut std::ffi::c_void,
                (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                (&mut ns) as *mut _ as *mut std::ffi::c_void,
                (&mut nh) as *mut _ as *mut std::ffi::c_void,
                (&mut h) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx.launch3(
                "q4_qsa_attn_sel4s_merge",
                n_head.div_ceil(8) as u32,
                1,
                1,
                256,
                &mut args,
            )?;
            return Ok(());
        }
        // 비분할 — t>3은 sel4(K/V 4헤드 공유), t≤3은 sel. 상동 사유.
        let use6 = false && n_head % 12 == 0;
        let mut q_p = qdev as *mut std::ffi::c_void;
        let mut o_p = odev as *mut std::ffi::c_void;
        let mut k_p = ckp as *mut std::ffi::c_void;
        let mut v_p = cvp as *mut std::ffi::c_void;
        let mut si_p = sdev as *mut std::ffi::c_void;
        let mut so_p = ofdev as *mut std::ffi::c_void;
        let mut sc = kq_scale;
        let mut nh = n_head as i32;
        let mut nk = n_kv as i32;
        let mut h = hd as i32;
        let mut tt = t as i32;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&mut q_p) as *mut _ as *mut std::ffi::c_void,
            (&mut k_p) as *mut _ as *mut std::ffi::c_void,
            (&mut v_p) as *mut _ as *mut std::ffi::c_void,
            (&mut si_p) as *mut _ as *mut std::ffi::c_void,
            (&mut so_p) as *mut _ as *mut std::ffi::c_void,
            (&mut o_p) as *mut _ as *mut std::ffi::c_void,
            (&mut sc) as *mut _ as *mut std::ffi::c_void,
            (&mut nh) as *mut _ as *mut std::ffi::c_void,
            (&mut nk) as *mut _ as *mut std::ffi::c_void,
            (&mut h) as *mut _ as *mut std::ffi::c_void,
            (&mut tt) as *mut _ as *mut std::ffi::c_void,
        ];
        let (kern, gy, blk) = if use6 {
            ("q4_qsa_attn_sel6", (n_head / 12) as u32, 256u32)
        } else {
            ("q4_qsa_attn_sel4", (n_head / 8) as u32, 256u32)
        };
        let gx = t.div_ceil(4) as u32;
        self.ctx.launch3(kern, gx, gy, 1, blk, &mut args)?;
        Ok(())
    }

    /// 산술은 `qsa_attn_sel6_raw`와 동일(같은 커널) → 비트 동일 기대.
    #[allow(clippy::too_many_arguments)]
    pub fn qsa_attn_dev_raw(
        &self,
        q: u64,
        ck: &[f32],
        cv: &[f32],
        sel_idx: &[u32],
        sel_off: &[u32],
        kq_scale: f32,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
        out: u64,
    ) -> Result<(), String> {
        let (qdev, kdev, vdev, sdev, ofdev, _odev) = {
            let mut a = self.qs.lock().map_err(|e| e.to_string())?;
            let qdev = a.ensure(&self.ctx, t.max(1) * n_head * 2 * hd * 4)?;
            let mut b = self.ckv.lock().map_err(|e| e.to_string())?;
            let kdev = b.ensure(&self.ctx, ck.len().max(1) * 4)?;
            let mut c = self.cvv.lock().map_err(|e| e.to_string())?;
            let vdev = c.ensure(&self.ctx, cv.len().max(1) * 4)?;
            let mut d = self.msk.lock().map_err(|e| e.to_string())?;
            let sdev = d.ensure(&self.ctx, sel_idx.len().max(1) * 4)?;
            let mut e2 = self.soff.lock().map_err(|e| e.to_string())?;
            let ofdev = e2.ensure(&self.ctx, sel_off.len().max(1) * 4)?;
            let mut f2 = self.atn.lock().map_err(|e| e.to_string())?;
            let odev = f2.ensure(&self.ctx, t * n_head * hd * 4)?;
            (qdev, kdev, vdev, sdev, ofdev, odev)
        };
        let _ = qdev; // q는 인자로 받은 디바이스 버퍼를 그대로 쓴다(업로드 없음).
        self.ctx.h2d(kdev, bytemuck::cast_slice(ck))?;
        self.ctx.h2d(vdev, bytemuck::cast_slice(cv))?;
        self.ctx.h2d(sdev, bytemuck::cast_slice(sel_idx))?;
        self.ctx.h2d(ofdev, bytemuck::cast_slice(sel_off))?;
        let mut q_p = self.fptr(q)? as *mut std::ffi::c_void;
        let mut o_p = self.fptr(out)? as *mut std::ffi::c_void;
        let mut k_p = kdev as *mut std::ffi::c_void;
        let mut v_p = vdev as *mut std::ffi::c_void;
        let mut si_p = sdev as *mut std::ffi::c_void;
        let mut so_p = ofdev as *mut std::ffi::c_void;
        let mut sc = kq_scale;
        let mut nh = n_head as i32;
        let mut nk = n_kv as i32;
        let mut h = hd as i32;
        let mut tt = t as i32;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&mut q_p) as *mut _ as *mut std::ffi::c_void,
            (&mut k_p) as *mut _ as *mut std::ffi::c_void,
            (&mut v_p) as *mut _ as *mut std::ffi::c_void,
            (&mut si_p) as *mut _ as *mut std::ffi::c_void,
            (&mut so_p) as *mut _ as *mut std::ffi::c_void,
            (&mut o_p) as *mut _ as *mut std::ffi::c_void,
            (&mut sc) as *mut _ as *mut std::ffi::c_void,
            (&mut nh) as *mut _ as *mut std::ffi::c_void,
            (&mut nk) as *mut _ as *mut std::ffi::c_void,
            (&mut h) as *mut _ as *mut std::ffi::c_void,
            (&mut tt) as *mut _ as *mut std::ffi::c_void,
        ];
        // 프리필(t>3)은 sel4 — K/V를 4헤드가 공유(트래픽 1/4, 호스트 경로와
        // 동일 선택). sel6(6헤드)은 K/V를 묶음마다 재독해 t=2048 실측 40ms/런치
        // 까지 올라갔다(2026-09-14 KTRACE, 48런치 1.92s) — 프리필 회귀였음.
        // t≤3도 호스트 규약대로 sel(헤드당 워프)을 쓴다.
        let _ = n_head % 12;
        let (kern, gy, blk) = if t > 3 {
            ("q4_qsa_attn_sel4", (n_head / 8) as u32, 256u32)
        } else {
            ("q4_qsa_attn_sel", (n_head / 4) as u32, 128u32)
        };
        if t > 3 {
            let gx = t.div_ceil(4) as u32;
            self.ctx.launch3(kern, gx, gy, 1, blk, &mut args)?;
        } else {
            // _sel 원본 규격: 블록 16워프=16토큰(워프당 1헤드), gy=n_head.
            self.ctx
                .launch3("q4_qsa_attn_sel", t.div_ceil(16) as u32, n_head as u32, 1, 512, &mut args)?;
        }
        Ok(())
    }

    /// t=1 위치 분할판 — 선택목록을 n_splits로 쪼개 (split, 헤드묶음) 그리드로
    /// 펼친다. `_sel4`는 워프가 목록 전체를 직렬 순회해 t=1에서 지연 바운드다
    /// (실측 1.425ms/콜). 부분 (m,l,acc)를 남기고 2차 커널이 flash 규약으로
    /// 병합한다 — 합산 순서가 분할 경계에서 달라 비트 동일은 아니고 greedy
    /// 스트림 동일성으로 검증한다. LLM170_QSA_SPLITS로 분할 수(기본 64).
    #[allow(clippy::too_many_arguments)]
    pub fn qsa_attn_sel4s_raw(
        &self,
        q: &[f32],
        ck: &[f32],
        cv: &[f32],
        sel_idx: &[u32],
        sel_off: &[u32],
        kq_scale: f32,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
    ) -> Result<Vec<f32>, String> {
        // 분할 수는 목록 길이에 맞춘다 — 짧은 문맥에서는 분할 이득이 없고
        // 부분 버퍼 쓰기·병합 비용만 늘어난다(분할당 최소 32위치).
        let list_len = sel_off
            .get(1)
            .copied()
            .unwrap_or(0)
            .saturating_sub(sel_off.first().copied().unwrap_or(0)) as usize;
        let cap = std::env::var("LLM170_QSA_SPLITS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64);
        let n_splits: usize = (list_len / 32).clamp(1, cap.max(1).min(512));
        let (qdev, kdev, vdev, sdev, ofdev, pdev, odev) = {
            let mut a = self.qs.lock().map_err(|e| e.to_string())?;
            let qdev = a.ensure(&self.ctx, q.len().max(1) * 4)?;
            // KV는 컨텍스트 전체를 미리 잡는다(엔진이 주입한 ctx_len). 종전에는
            // n_past가 늘 때마다 재할당해 매 스텝 주소가 바뀌었다(실측 48회/세션).
            let kv_floats = self.ctx_len.load(std::sync::atomic::Ordering::Relaxed)
                * n_kv.max(1) * hd.max(1);
            let mut b = self.ckv.lock().map_err(|e| e.to_string())?;
            let kdev = b.ensure(&self.ctx, ck.len().max(kv_floats) * 4)?;
            let mut c = self.cvv.lock().map_err(|e| e.to_string())?;
            let vdev = c.ensure(&self.ctx, cv.len().max(kv_floats) * 4)?;
            let mut d = self.msk.lock().map_err(|e| e.to_string())?;
            let sdev = d.ensure(&self.ctx, sel_idx.len().max(1) * 4)?;
            let mut e2 = self.soff.lock().map_err(|e| e.to_string())?;
            let ofdev = e2.ensure(&self.ctx, sel_off.len().max(1) * 4)?;
            let mut g = self.qsp.lock().map_err(|e| e.to_string())?;
            let pdev = g.ensure(&self.ctx, n_head * n_splits * 32 * 10 * 4)?;
            let mut f2 = self.atn.lock().map_err(|e| e.to_string())?;
            let odev = f2.ensure(&self.ctx, t * n_head * hd * 4)?;
            (qdev, kdev, vdev, sdev, ofdev, pdev, odev)
        };
        self.ctx.h2d(qdev, bytemuck::cast_slice(q))?;
        self.ctx.h2d(kdev, bytemuck::cast_slice(ck))?;
        self.ctx.h2d(vdev, bytemuck::cast_slice(cv))?;
        self.ctx.h2d(sdev, bytemuck::cast_slice(sel_idx))?;
        self.ctx.h2d(ofdev, bytemuck::cast_slice(sel_off))?;
        {
            let mut q_p = qdev as *mut std::ffi::c_void;
            let mut k_p = kdev as *mut std::ffi::c_void;
            let mut v_p = vdev as *mut std::ffi::c_void;
            let mut si_p = sdev as *mut std::ffi::c_void;
            let mut so_p = ofdev as *mut std::ffi::c_void;
            let mut pa_p = pdev as *mut std::ffi::c_void;
            let mut ns = n_splits as i32;
            let mut sc = kq_scale;
            let mut nh = n_head as i32;
            let mut nk = n_kv as i32;
            let mut h = hd as i32;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut q_p) as *mut _ as *mut std::ffi::c_void,
                (&mut k_p) as *mut _ as *mut std::ffi::c_void,
                (&mut v_p) as *mut _ as *mut std::ffi::c_void,
                (&mut si_p) as *mut _ as *mut std::ffi::c_void,
                (&mut so_p) as *mut _ as *mut std::ffi::c_void,
                (&mut pa_p) as *mut _ as *mut std::ffi::c_void,
                (&mut ns) as *mut _ as *mut std::ffi::c_void,
                (&mut sc) as *mut _ as *mut std::ffi::c_void,
                (&mut nh) as *mut _ as *mut std::ffi::c_void,
                (&mut nk) as *mut _ as *mut std::ffi::c_void,
                (&mut h) as *mut _ as *mut std::ffi::c_void,
            ];
            // 묶음당 6헤드(블록당 2묶음) — 분할판도 같은 재독 절감.
            self.ctx.launch3(
                "q4_qsa_attn_sel4s",
                n_splits.div_ceil(4) as u32,
                (n_head / 12) as u32,
                1,
                256,
                &mut args,
            )?;
        }
        {
            let mut pa_p = pdev as *mut std::ffi::c_void;
            let mut q_p = qdev as *mut std::ffi::c_void;
            let mut o_p = odev as *mut std::ffi::c_void;
            let mut ns = n_splits as i32;
            let mut nh = n_head as i32;
            let mut h = hd as i32;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut pa_p) as *mut _ as *mut std::ffi::c_void,
                (&mut q_p) as *mut _ as *mut std::ffi::c_void,
                (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                (&mut ns) as *mut _ as *mut std::ffi::c_void,
                (&mut nh) as *mut _ as *mut std::ffi::c_void,
                (&mut h) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx.launch3(
                "q4_qsa_attn_sel4s_merge",
                n_head.div_ceil(8) as u32,
                1,
                1,
                256,
                &mut args,
            )?;
        }
        let mut out = vec![0.0f32; t * n_head * hd];
        self.ctx.d2h(bytemuck::cast_slice_mut(&mut out), odev)?;
        Ok(out)
    }

    /// 분할판의 **디바이스 q·출력판** (plans/67 2c) — q를 wq 출력 버퍼에서 직접
    /// 읽고 어텐션 출력도 프레임 버퍼에 쓴다(왕복 0). 산술은 sel4s와 동일
    /// 커널 쌍이라 greedy 스트림 동일. t=1 장문맥 디코드의 지연 바운드를
    /// 분할로 푼다(142.5→124.4ms/스텝 실측치의 디바이스 상속).
    #[allow(clippy::too_many_arguments)]
    pub fn qsa_attn_sel4s_dev_raw(
        &self,
        q: u64,
        ck: &[f32],
        cv: &[f32],
        sel_idx: &[u32],
        sel_off: &[u32],
        kq_scale: f32,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
        out: u64,
    ) -> Result<(), String> {
        let _ = t; // t==1 규약(호출부가 보장) — 커널은 sel_off로 범위를 안다
        let list_len = sel_off
            .get(1)
            .copied()
            .unwrap_or(0)
            .saturating_sub(sel_off.first().copied().unwrap_or(0)) as usize;
        let cap = std::env::var("LLM170_QSA_SPLITS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64);
        let n_splits: usize = (list_len / 32).clamp(1, cap.max(1).min(512));
        let (kdev, vdev, sdev, ofdev, pdev) = {
            let kv_floats = self.ctx_len.load(std::sync::atomic::Ordering::Relaxed)
                * n_kv.max(1) * hd.max(1);
            let mut b = self.ckv.lock().map_err(|e| e.to_string())?;
            let kdev = b.ensure(&self.ctx, ck.len().max(kv_floats) * 4)?;
            let mut c = self.cvv.lock().map_err(|e| e.to_string())?;
            let vdev = c.ensure(&self.ctx, cv.len().max(kv_floats) * 4)?;
            let mut d = self.msk.lock().map_err(|e| e.to_string())?;
            let sdev = d.ensure(&self.ctx, sel_idx.len().max(1) * 4)?;
            let mut e2 = self.soff.lock().map_err(|e| e.to_string())?;
            let ofdev = e2.ensure(&self.ctx, sel_off.len().max(1) * 4)?;
            let mut g = self.qsp.lock().map_err(|e| e.to_string())?;
            let pdev = g.ensure(&self.ctx, n_head * n_splits * 32 * 10 * 4)?;
            (kdev, vdev, sdev, ofdev, pdev)
        };
        self.ctx.h2d(kdev, bytemuck::cast_slice(ck))?;
        self.ctx.h2d(vdev, bytemuck::cast_slice(cv))?;
        self.ctx.h2d(sdev, bytemuck::cast_slice(sel_idx))?;
        self.ctx.h2d(ofdev, bytemuck::cast_slice(sel_off))?;
        let qdev = self.fptr(q)?;
        let odev = self.fptr(out)?;
        {
            let mut q_p = qdev as *mut std::ffi::c_void;
            let mut k_p = kdev as *mut std::ffi::c_void;
            let mut v_p = vdev as *mut std::ffi::c_void;
            let mut si_p = sdev as *mut std::ffi::c_void;
            let mut so_p = ofdev as *mut std::ffi::c_void;
            let mut pa_p = pdev as *mut std::ffi::c_void;
            let mut ns = n_splits as i32;
            let mut sc = kq_scale;
            let mut nh = n_head as i32;
            let mut nk = n_kv as i32;
            let mut h = hd as i32;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut q_p) as *mut _ as *mut std::ffi::c_void,
                (&mut k_p) as *mut _ as *mut std::ffi::c_void,
                (&mut v_p) as *mut _ as *mut std::ffi::c_void,
                (&mut si_p) as *mut _ as *mut std::ffi::c_void,
                (&mut so_p) as *mut _ as *mut std::ffi::c_void,
                (&mut pa_p) as *mut _ as *mut std::ffi::c_void,
                (&mut ns) as *mut _ as *mut std::ffi::c_void,
                (&mut sc) as *mut _ as *mut std::ffi::c_void,
                (&mut nh) as *mut _ as *mut std::ffi::c_void,
                (&mut nk) as *mut _ as *mut std::ffi::c_void,
                (&mut h) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx.launch3(
                "q4_qsa_attn_sel4s",
                n_splits.div_ceil(4) as u32,
                (n_head / 12) as u32,
                1,
                256,
                &mut args,
            )?;
        }
        {
            let mut pa_p = pdev as *mut std::ffi::c_void;
            let mut q_p = qdev as *mut std::ffi::c_void;
            let mut o_p = odev as *mut std::ffi::c_void;
            let mut ns = n_splits as i32;
            let mut nh = n_head as i32;
            let mut h = hd as i32;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut pa_p) as *mut _ as *mut std::ffi::c_void,
                (&mut q_p) as *mut _ as *mut std::ffi::c_void,
                (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                (&mut ns) as *mut _ as *mut std::ffi::c_void,
                (&mut nh) as *mut _ as *mut std::ffi::c_void,
                (&mut h) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx.launch3(
                "q4_qsa_attn_sel4s_merge",
                n_head.div_ceil(8) as u32,
                1,
                1,
                256,
                &mut args,
            )?;
        }
        Ok(())
    }

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
            // KV는 컨텍스트 전체를 미리 잡는다(엔진이 주입한 ctx_len). 종전에는
            // n_past가 늘 때마다 재할당해 매 스텝 주소가 바뀌었다(실측 48회/세션).
            let kv_floats = self.ctx_len.load(std::sync::atomic::Ordering::Relaxed)
                * n_kv.max(1) * hd.max(1);
            let mut b = self.ckv.lock().map_err(|e| e.to_string())?;
            let kdev = b.ensure(&self.ctx, ck.len().max(kv_floats) * 4)?;
            let mut c = self.cvv.lock().map_err(|e| e.to_string())?;
            let vdev = c.ensure(&self.ctx, cv.len().max(kv_floats) * 4)?;
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
        // 워프-퍼-토큰 커널 — 블록 배리어 없음 + K를 8토큰이 공유(§41).
        // 미러 대조 2.263e-4(구 커널과 동일), 토큰 동일, 프리필 −1.9%@11.75k.
        self.ctx.launch3(
            "q4_qsa_attn_wt",
            t.div_ceil(16) as u32,
            n_head as u32,
            1,
            512,
            &mut args,
        )?;
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

    /// np 행별 conv 1런치 (plans/74 N2) — gdn_conv(t=1) 산술, 상태는 행
    /// 포인터 테이블. qkv/out은 [t][ch] 연속 프레임 버퍼.
    fn frame_gdn_conv_np(
        &self,
        qkv: u64,
        out: u64,
        states: &[u64],
        cw: u64,
        ch: usize,
        k: usize,
    ) -> Result<(), String> {
        let t = states.len();
        if t == 0 {
            return Ok(());
        }
        let mut ptrs: Vec<usize> = Vec::with_capacity(t);
        for &h in states {
            ptrs.push(self.fptr(h)? as usize);
        }
        let tbl = self.ctx.scratch(t * 8)?;
        self.ctx.h2d(tbl, bytemuck::cast_slice(&ptrs))?;
        let (mut q, mut c, mut s_, mut o_) = (
            self.fptr(qkv)?,
            self.fptr(cw)?,
            tbl as *mut std::ffi::c_void,
            self.fptr(out)?,
        );
        let (mut chh, mut kk, mut tt) = (ch as i32, k as i32, t as i32);
        self.kop(
            "gdn_conv_np",
            (ch as u32).div_ceil(64),
            t as u32,
            1,
            64,
            &mut cargs!(&mut q, &mut c, &mut s_, &mut o_, &mut chh, &mut kk, &mut tt),
        )
    }
    /// np 행별 AR 1런치 (plans/74 N2) — gdn_ar_w_swap(t=1) 산술(scale=1,
    /// q는 L2Rows+Scale 로 선스케일), 상태는 행 포인터 테이블.
    #[allow(clippy::too_many_arguments)]
    fn frame_gdn_ar_np(
        &self,
        q: u64,
        k: u64,
        v: u64,
        beta_ge: u64,
        out: u64,
        states: &[u64],
        h_k: usize,
        h_v: usize,
        d: usize,
    ) -> Result<(), String> {
        let t = states.len();
        if t == 0 {
            return Ok(());
        }
        let mut ptrs: Vec<usize> = Vec::with_capacity(t);
        for &h in states {
            ptrs.push(self.fptr(h)? as usize);
        }
        let tbl = self.ctx.scratch(t * 8)?;
        self.ctx.h2d(tbl, bytemuck::cast_slice(&ptrs))?;
        let (mut sp, mut qp, mut kp, mut vp, mut bp, mut op_) = (
            tbl as *mut std::ffi::c_void,
            self.fptr(q)?,
            self.fptr(k)?,
            self.fptr(v)?,
            self.fptr(beta_ge)?,
            self.fptr(out)?,
        );
        let (mut dd, mut ks, mut vs, mut hv, mut hk, mut sc, mut tt) = (
            d as i32,
            (h_k * d) as i32,
            (h_v * d) as i32,
            h_v as i32,
            h_k as i32,
            1.0f32,
            t as i32,
        );
        // gx=h_v(페어 축 — 커널의 blockIdx.x), gy=d(u 축). 27B rawhip 판과
        // 동일 순서(2026-09-16 실수로 (d,h_v)로 바꿔써 GPU 메모리 폴트).
        self.ctx.launch3(
            "gdn_ar_w_np",
            h_v as u32,
            d as u32,
            1,
            32,
            &mut cargs!(&mut sp, &mut qp, &mut kp, &mut vp, &mut bp, &mut op_, &mut dd, &mut ks, &mut vs, &mut hv, &mut hk, &mut sc, &mut tt),
        )
    }
    /// plans/73(np): 프레임 버퍼 행 뷰 — 배치 디코드의 per-seq 상태 op용.
    fn frame_slice(&self, h: u64, off_elems: usize, len: usize) -> Result<u64, String> {
        let mut v = self.frames.lock().map_err(|e| e.to_string())?;
        let idx = (h.checked_sub(1).ok_or("frame 핸들 0")?) as usize;
        let (base, cap) = *v
            .get(idx)
            .ok_or_else(|| format!("frame 핸들 없음: {h}"))?;
        let need = (off_elems + len) * 4;
        if need > cap {
            return Err(format!("frame_slice 범위 초과: need {need} > cap {cap}"));
        }
        let ptr = unsafe { base.add(off_elems * 4) };
        v.push((ptr, len * 4));
        Ok(v.len() as u64)
    }

    fn capture_mark(&self, tag: &str) -> Result<(), String> {
        crate::rawhip::capture_mark(self.ctx.stream, tag)
    }
    fn graph_capture_begin(&self) -> Result<(), String> {
        crate::rawhip::graph_capture_begin(self.ctx.stream)
    }
    fn graph_capture_end(&self) -> Result<(), String> {
        crate::rawhip::graph_capture_end(self.ctx.stream)
    }
    fn graph_replay(&self, on: bool) -> Result<(), String> {
        crate::rawhip::graph_replay(on)
    }

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

    fn frame_qk_norm_rope(
        &self,
        q: u64,
        k: u64,
        q_norm: &[f32],
        k_norm: &[f32],
        cs: &[f32],
        eps: f32,
        pos0: usize,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        n_rot: usize,
        t: usize,
    ) -> Result<(), String> {
        // 상수 3개(qn/kn/cs) — plans/73: 매 호출 h2d(+sync)가 스텝당 36회의
        // 동기를 만들었다. **키는 (ptr,len)** — 내용 해시는 층마다 값이 달라
        // 단일 슬롯 캐시가 매 층 미스했고(24KB+2KB 동기 복사 ×12층 = 40ms/스텝),
        // 프레임이 헤드 타일을 1회 만들어 상주시키므로 포인터가 곧 신원이다.
        // (2026-09-16: LLM170_Q4_TIME 계측 — qsa.mm+rope 3.4ms/층의 전부가 이 복사였다)
        let (qnd, knd, csd) = {
            let qnd = self.upload_map(&self.qn_map, "qn_t", q_norm)?;
            let knd = self.upload_map(&self.kn_map, "kn_t", k_norm)?;
            let qh = q_norm.as_ptr() as u64;
            let kh = k_norm.as_ptr() as u64;
            let _ = (qh, kh);
            let cskey = (cs.as_ptr() as usize, cs.len());
            let mut c = (self.cst.lock().map_err(|e| e.to_string())?, self.cst_cache.lock().map_err(|e| e.to_string())?);
            if *c.1 != cskey || c.0.ptr.is_null() {
                c.0.ensure(&self.ctx, cs.len().max(1) * 4)?;
                self.ctx.h2d(c.0.ptr, bytemuck::cast_slice(cs))?;
                *c.1 = cskey;
            }
            (qnd, knd, c.0.ptr)
        };
        let mut qp = self.fptr(q)? as *mut std::ffi::c_void;
        let mut kp = self.fptr(k)? as *mut std::ffi::c_void;
        let mut qwp = qnd as *mut std::ffi::c_void;
        let mut kwp = knd as *mut std::ffi::c_void;
        let mut csp = csd as *mut std::ffi::c_void;
        // kq_scale은 이 커널의 decode 판은 k에 구워 넣지만(kqs=self.kq_scale),
        // QSA 프레임 경로는 **k를 무척도(1.0)로 둔다** — QSA KV 캐시 규약이
        // 무척도 k이고 qsa_attn_sel6가 q·k에 kq_scale을 곱하기 때문. 초기 구현은
        // 0.0을 넘겨 k를 전부 0으로 만드는 잠복 결함이었음(미호출 경로라 미발견,
        // 2026-09-14 plans/67 2c 연결 시 발견·수정).
        let mut kq = 1.0f32;
        let mut ep = eps;
        let mut pp = pos0 as i32;
        let mut nh = n_head as i32;
        let mut nk = n_kv as i32;
        let mut h = hd as i32;
        let mut nr = n_rot as i32;
        let rows = n_head + n_kv;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&mut qp) as *mut _ as *mut std::ffi::c_void,
            (&mut kp) as *mut _ as *mut std::ffi::c_void,
            (&mut qwp) as *mut _ as *mut std::ffi::c_void,
            (&mut kwp) as *mut _ as *mut std::ffi::c_void,
            (&mut csp) as *mut _ as *mut std::ffi::c_void,
            (&mut ep) as *mut _ as *mut std::ffi::c_void,
            (&mut kq) as *mut _ as *mut std::ffi::c_void,
            (&mut pp) as *mut _ as *mut std::ffi::c_void,
            (&mut nh) as *mut _ as *mut std::ffi::c_void,
            (&mut nk) as *mut _ as *mut std::ffi::c_void,
            (&mut h) as *mut _ as *mut std::ffi::c_void,
            (&mut nr) as *mut _ as *mut std::ffi::c_void,
        ];
        self.ctx.launch3("qk_norm_rope", rows as u32, t as u32, 1, 32, &mut args)
    }

    fn qsa_attention_dev(
        &self,
        q: u64,
        ck: &[f32],
        cv: &[f32],
        sel_idx: &[u32],
        sel_off: &[u32],
        kq_scale: f32,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
        out: u64,
    ) -> Result<(), String> {
        // t=1은 위치 분할판(flash-decoding형) — 디바이스 q·출력판이 같은 커널
        // 쌍을 쓴다. 규약은 호스트 판(qsa_attention_sel)과 동일: LLM170_QSA_SPLIT=0
        // 이면 비분할 sel6/sel4로 돌아간다.
        if t == 1 && std::env::var("LLM170_QSA_SPLIT").as_deref() != Ok("0") {
            self.qsa_attn_sel4s_dev_raw(q, ck, cv, sel_idx, sel_off, kq_scale, n_head, n_kv, hd, t, out)
        } else {
            self.qsa_attn_dev_raw(q, ck, cv, sel_idx, sel_off, kq_scale, n_head, n_kv, hd, t, out)
        }
    }

    fn qsa_kv_dev(
        &self,
        full_idx: usize,
        seq: usize,
        k: u64,
        v: u64,
        t: usize,
        pos0: usize,
        n_kv: usize,
        hd: usize,
    ) -> Result<(u64, u64), String> {
        self.qsa_kv_dev_impl(full_idx, seq, k, v, t, pos0, n_kv, hd)
    }

    fn shexp_gu(
        &self, x: u64, wg: &llm170_core::matmul::Weight, wu: &llm170_core::matmul::Weight,
        h: u64, n_in: usize, n_hidden: usize,
    ) -> Result<(), String> {
        let mut xp = self.fptr(x)? as *mut std::ffi::c_void;
        let (wgd, _) = self.dev_weight(wg)?;
        let (wud, _) = self.dev_weight(wu)?;
        let mut wgp = wgd as *mut std::ffi::c_void;
        let mut wup = wud as *mut std::ffi::c_void;
        let mut hp = self.fptr(h)? as *mut std::ffi::c_void;
        let mut ni = n_in as i32;
        let mut nh = n_hidden as i32;
        let mut args = vec![
            (&mut xp) as *mut _ as *mut std::ffi::c_void,
            (&mut wgp) as *mut _ as *mut std::ffi::c_void,
            (&mut wup) as *mut _ as *mut std::ffi::c_void,
            (&mut hp) as *mut _ as *mut std::ffi::c_void,
            (&mut ni) as *mut _ as *mut std::ffi::c_void,
            (&mut nh) as *mut _ as *mut std::ffi::c_void,
        ];
        // n_hidden=640, warp당 1행 → 640 워프 = 20블록(256스레드=8워프)
        self.ctx.launch3("q4_shexp_gu", n_hidden.div_ceil(8) as u32, 1, 1, 256, &mut args)
    }

    fn shexp_da(
        &self, h: u64, wd: &llm170_core::matmul::Weight, s: u64, mout: u64,
        n_in: usize, n_hidden: usize,
    ) -> Result<(), String> {
        let mut hp = self.fptr(h)? as *mut std::ffi::c_void;
        let (wdd, _) = self.dev_weight(wd)?;
        let mut wdp = wdd as *mut std::ffi::c_void;
        let mut sp = self.fptr(s)? as *mut std::ffi::c_void;
        let mut mp = self.fptr(mout)? as *mut std::ffi::c_void;
        let mut ni = n_in as i32;
        let mut nh = n_hidden as i32;
        let mut args = vec![
            (&mut hp) as *mut _ as *mut std::ffi::c_void,
            (&mut wdp) as *mut _ as *mut std::ffi::c_void,
            (&mut sp) as *mut _ as *mut std::ffi::c_void,
            (&mut mp) as *mut _ as *mut std::ffi::c_void,
            (&mut ni) as *mut _ as *mut std::ffi::c_void,
            (&mut nh) as *mut _ as *mut std::ffi::c_void,
        ];
        // n_in=2560, warp당 1행 → 2560 워프 = 320블록(8워프/블록)
        self.ctx.launch3("q4_shexp_da", n_in.div_ceil(8) as u32, 1, 1, 256, &mut args)
    }

    fn qsa_kv_check(
        &self,
        full_idx: usize,
        seq: usize,
        host_ck: &[f32],
        host_cv: &[f32],
    ) -> Result<(), String> {
        let m = self.qsa_kv.lock().map_err(|e| e.to_string())?;
        let Some((kb, vb)) = m.get(&(full_idx, seq)) else {
            return Err("qsa_kv_check: 풀 없음".into());
        };
        let n = host_ck.len().min(kb.bytes / 4);
        let mut got = vec![0.0f32; n];
        self.ctx
            .d2h(bytemuck::cast_slice_mut(&mut got), kb.ptr as *const u8)
            .map_err(|e| e.to_string())?;
        for (i, (a, b)) in got.iter().zip(host_ck[..n].iter()).enumerate() {
            if a.to_bits() != b.to_bits() {
                return Err(format!(
                    "qsa_kv_check k 불일치 @float {i}: pool={a:e} host={b:e}"
                ));
            }
        }
        let n = host_cv.len().min(vb.bytes / 4);
        let mut got = vec![0.0f32; n];
        self.ctx
            .d2h(bytemuck::cast_slice_mut(&mut got), vb.ptr as *const u8)
            .map_err(|e| e.to_string())?;
        for (i, (a, b)) in got.iter().zip(host_cv[..n].iter()).enumerate() {
            if a.to_bits() != b.to_bits() {
                return Err(format!(
                    "qsa_kv_check v 불일치 @float {i}: pool={a:e} host={b:e}"
                ));
            }
        }
        Ok(())
    }

    fn qsa_attention_dev_res(
        &self,
        q: u64,
        ck: u64,
        cv: u64,
        sel_idx: &[u32],
        sel_off: &[u32],
        kq_scale: f32,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
        out: u64,
    ) -> Result<(), String> {
        self.qsa_attn_res(q, ck, cv, sel_idx, sel_off, kq_scale, n_head, n_kv, hd, t, out)
    }

    fn qsa_sel_dev(
        &self,
        full_idx: usize,
        seq: usize,
        iq: u64,
        ik: u64,
        t: usize,
        pos0: usize,
        idx_heads: usize,
        idx_dim: usize,
        r: usize,
        idx_top_k: usize,
        iqw: &[f32],
        ikw: &[f32],
        cs_idx: &[f32],
        eps: f32,
    ) -> Result<(u64, u64, usize), String> {
        if t != 1 {
            return Err(format!("qsa_sel_dev: t={t} (디코드 전용)"));
        }
        if r == 0 {
            return Err("qsa_sel_dev: r=0".into());
        }
        let n_past = pos0 + t;
        let n_blocks = n_past / r;
        if n_blocks > 8192 {
            return Err(format!("qsa_sel_dev: n_blocks={n_blocks} > 8192 (expand shared)"));
        }
        let iqp = self.fptr(iq)?;
        let ikp = self.fptr(ik)?;
        let (_idxk_p, bk_p) =
            self.qsa_idx_append(full_idx, seq, ikp, &[], t, pos0, idx_dim, r, ikw, cs_idx, eps)?;
        // (1) iq norm+rope — iqr 스크래치.
        let iqr = {
            let mut g = self.qsa_iqr.lock().map_err(|e| e.to_string())?;
            g.ensure(&self.ctx, t * idx_heads * idx_dim * 4)?
        };
        let iqw_d = self.upload_hashed(&self.qsa_iqw, iqw)?;
        let cs_d = self.upload_by_ptr(&self.qsa_csidx, cs_idx)?;
        {
            let (mut qp, mut op, mut iw, mut cp) = (
                iqp as *mut std::ffi::c_void,
                iqr as *mut std::ffi::c_void,
                iqw_d as *mut std::ffi::c_void,
                cs_d as *mut std::ffi::c_void,
            );
            let (mut e, mut pp, mut tt, mut ih, mut dd) = (
                eps,
                pos0 as i32,
                t as i32,
                idx_heads as i32,
                idx_dim as i32,
            );
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut qp) as *mut _ as *mut std::ffi::c_void,
                (&mut op) as *mut _ as *mut std::ffi::c_void,
                (&mut iw) as *mut _ as *mut std::ffi::c_void,
                (&mut cp) as *mut _ as *mut std::ffi::c_void,
                (&mut e) as *mut _ as *mut std::ffi::c_void,
                (&mut pp) as *mut _ as *mut std::ffi::c_void,
                (&mut tt) as *mut _ as *mut std::ffi::c_void,
                (&mut ih) as *mut _ as *mut std::ffi::c_void,
                (&mut dd) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx
                .launch3("q4_idx_q_rope", idx_heads as u32, t as u32, 1, 32, &mut args)?;
        }
        // (2) 블록 점수 — 스레드당 블록.
        let scr = {
            let mut g = self.qsa_scr.lock().map_err(|e| e.to_string())?;
            g.ensure(&self.ctx, n_blocks.max(1) * 4)?
        };
        if n_blocks > 0 {
            let (mut qp, mut bp, mut sp) = (
                iqr as *mut std::ffi::c_void,
                bk_p as *mut std::ffi::c_void,
                scr as *mut std::ffi::c_void,
            );
            let (mut nb, mut ih, mut dd) = (n_blocks as i32, idx_heads as i32, idx_dim as i32);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut qp) as *mut _ as *mut std::ffi::c_void,
                (&mut bp) as *mut _ as *mut std::ffi::c_void,
                (&mut sp) as *mut _ as *mut std::ffi::c_void,
                (&mut nb) as *mut _ as *mut std::ffi::c_void,
                (&mut ih) as *mut _ as *mut std::ffi::c_void,
                (&mut dd) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx.launch3(
                "q4_idx_score",
                (n_blocks as u32).div_ceil(256),
                1,
                1,
                256,
                &mut args,
            )?;
        }
        // (3) top-k 순위 + 목록 전개. n_sel 산술은 stages::qsa_select 패스 B와
        // 동일(usize 정수 — 호스트에서 계산해도 무동기).
        let tail_start = n_blocks * r;
        let tail_cnt = n_past - tail_start;
        let width = n_past.min(idx_top_k + r - 1);
        let n_sel = ((width - tail_cnt) / r).min(n_blocks);
        let list_len = n_sel * r + tail_cnt;
        let (sdev, ofdev) = {
            let mut d = self.msk.lock().map_err(|e| e.to_string())?;
            let sdev = d.ensure(&self.ctx, list_len.max(1) * 4)? as u64;
            let mut e2 = self.soff.lock().map_err(|e| e.to_string())?;
            let ofdev = e2.ensure(&self.ctx, 2 * 4)? as u64;
            (sdev, ofdev)
        };
        if n_blocks > 0 && n_blocks <= 4096 && std::env::var("LLM170_QSA_TOPK").as_deref() != Ok("0")
        {
            // 비토닉 단일 블록판 — rank+expand 콤보 대비 ~20×(0.228 → ~0.01ms).
            let (mut sp, mut si, mut so) = (
                scr as *mut std::ffi::c_void,
                sdev as *mut std::ffi::c_void,
                ofdev as *mut std::ffi::c_void,
            );
            let (mut nb, mut ns, mut rr, mut np) =
                (n_blocks as i32, n_sel as i32, r as i32, n_past as i32);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut sp) as *mut _ as *mut std::ffi::c_void,
                (&mut si) as *mut _ as *mut std::ffi::c_void,
                (&mut so) as *mut _ as *mut std::ffi::c_void,
                (&mut nb) as *mut _ as *mut std::ffi::c_void,
                (&mut ns) as *mut _ as *mut std::ffi::c_void,
                (&mut rr) as *mut _ as *mut std::ffi::c_void,
                (&mut np) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx.launch3("q4_idx_topk", 1, 1, 1, 256, &mut args)?;
            return Ok((sdev, ofdev, list_len));
        }
        {
            let selflag = {
                let mut g = self.qsa_selflag.lock().map_err(|e| e.to_string())?;
                g.ensure(&self.ctx, n_blocks.max(1) * 4)?
            };
            if n_blocks > 0 {
                let (mut sp, mut fp) = (
                    scr as *mut std::ffi::c_void,
                    selflag as *mut std::ffi::c_void,
                );
                let (mut nb, mut ns) = (n_blocks as i32, n_sel as i32);
                let mut args: Vec<*mut std::ffi::c_void> = vec![
                    (&mut sp) as *mut _ as *mut std::ffi::c_void,
                    (&mut fp) as *mut _ as *mut std::ffi::c_void,
                    (&mut nb) as *mut _ as *mut std::ffi::c_void,
                    (&mut ns) as *mut _ as *mut std::ffi::c_void,
                ];
                self.ctx.launch3(
                    "q4_idx_rank",
                    (n_blocks as u32).div_ceil(256),
                    1,
                    1,
                    256,
                    &mut args,
                )?;
            }
            let (mut fp, mut si, mut so) = (
                selflag as *mut std::ffi::c_void,
                sdev as *mut std::ffi::c_void,
                ofdev as *mut std::ffi::c_void,
            );
            let (mut nb, mut ns, mut rr, mut np) =
                (n_blocks as i32, n_sel as i32, r as i32, n_past as i32);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut fp) as *mut _ as *mut std::ffi::c_void,
                (&mut si) as *mut _ as *mut std::ffi::c_void,
                (&mut so) as *mut _ as *mut std::ffi::c_void,
                (&mut nb) as *mut _ as *mut std::ffi::c_void,
                (&mut ns) as *mut _ as *mut std::ffi::c_void,
                (&mut rr) as *mut _ as *mut std::ffi::c_void,
                (&mut np) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx
                .launch3("q4_idx_expand", 1, 1, 1, 256, &mut args)?;
        }
        Ok((sdev, ofdev, list_len))
    }

    fn qsa_idx_append_host(
        &self,
        full_idx: usize,
        seq: usize,
        ik_host: &[f32],
        t: usize,
        pos0: usize,
        idx_dim: usize,
        r: usize,
        ikw: &[f32],
        cs_idx: &[f32],
        eps: f32,
    ) -> Result<(), String> {
        self.qsa_idx_append(
            full_idx,
            seq,
            std::ptr::null(),
            ik_host,
            t,
            pos0,
            idx_dim,
            r,
            ikw,
            cs_idx,
            eps,
        )
        .map(|_| ())
    }

    fn qsa_host_rebuild(
        &self,
        full_idx: usize,
        seq: usize,
        pos: usize,
        kv_row: usize,
        kv_k: &mut [f32],
        kv_v: &mut [f32],
        idx_k: &mut [f32],
        bk: &mut [f32],
        r: usize,
        idx_dim: usize,
    ) -> Result<(), String> {
        let kv_ok = {
            let wm = self.qsa_kv_pos.lock().map_err(|e| e.to_string())?;
            wm.get(&(full_idx, seq)).copied().unwrap_or(0) >= pos
        };
        let idx_ok = {
            let wm = self.qsa_idx_pos.lock().map_err(|e| e.to_string())?;
            wm.get(&(full_idx, seq)).copied().unwrap_or(0) >= pos
        };
        if !kv_ok || !idx_ok {
            return Err(format!(
                "qsa_host_rebuild: 풀 워터마크 부족 kv={kv_ok} idx={idx_ok} pos={pos}"
            ));
        }
        let nb = pos / r;
        {
            let m = self.qsa_kv.lock().map_err(|e| e.to_string())?;
            let ent = m
                .get(&(full_idx, seq))
                .ok_or("qsa_host_rebuild: kv 풀 없음")?;
            self.ctx.d2h(bytemuck::cast_slice_mut(&mut kv_k[..pos * kv_row]), ent.0.ptr)?;
            self.ctx.d2h(bytemuck::cast_slice_mut(&mut kv_v[..pos * kv_row]), ent.1.ptr)?;
        }
        {
            let m = self.qsa_idxk.lock().map_err(|e| e.to_string())?;
            let p = m
                .get(&(full_idx, seq))
                .ok_or("qsa_host_rebuild: idx 풀 없음")?
                .ptr;
            self.ctx
                .d2h(bytemuck::cast_slice_mut(&mut idx_k[..pos * idx_dim]), p)?;
        }
        {
            let m = self.qsa_bk.lock().map_err(|e| e.to_string())?;
            let p = m
                .get(&(full_idx, seq))
                .ok_or("qsa_host_rebuild: bk 풀 없음")?
                .ptr;
            self.ctx
                .d2h(bytemuck::cast_slice_mut(&mut bk[..nb * idx_dim]), p)?;
        }
        Ok(())
    }

    fn qsa_attention_dev_sel(
        &self,
        q: u64,
        ck: u64,
        cv: u64,
        sel_idx: u64,
        sel_off: u64,
        list_len: usize,
        kq_scale: f32,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
        out: u64,
    ) -> Result<(), String> {
        // sel 버퍼가 이미 디바이스에 있다 — 업로드 없이 qsa_attn_res와 동일한
        // 커널 쌍(t=1 분할 우선)을 발사한다.
        if t != 1 || std::env::var("LLM170_QSA_SPLIT").as_deref() == Ok("0") {
            return Err(format!("qsa_attention_dev_sel: t={t} 비분할은 미지원"));
        }
        let cap = std::env::var("LLM170_QSA_SPLITS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64);
        let n_splits: usize = (list_len / 32).clamp(1, cap.max(1).min(512));
        let pdev = {
            let mut g = self.qsp.lock().map_err(|e| e.to_string())?;
            g.ensure(&self.ctx, n_head * n_splits * 32 * 10 * 4)?
        };
        let (qdev, odev) = (self.fptr(q)?, self.fptr(out)?);
        let mut q_p = qdev as *mut std::ffi::c_void;
        let mut k_p = ck as *mut std::ffi::c_void;
        let mut v_p = cv as *mut std::ffi::c_void;
        let mut si_p = sel_idx as *mut std::ffi::c_void;
        let mut so_p = sel_off as *mut std::ffi::c_void;
        let mut pa_p = pdev as *mut std::ffi::c_void;
        let mut ns = n_splits as i32;
        let mut sc = kq_scale;
        let mut nh = n_head as i32;
        let mut nk = n_kv as i32;
        let mut h = hd as i32;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&mut q_p) as *mut _ as *mut std::ffi::c_void,
            (&mut k_p) as *mut _ as *mut std::ffi::c_void,
            (&mut v_p) as *mut _ as *mut std::ffi::c_void,
            (&mut si_p) as *mut _ as *mut std::ffi::c_void,
            (&mut so_p) as *mut _ as *mut std::ffi::c_void,
            (&mut pa_p) as *mut _ as *mut std::ffi::c_void,
            (&mut ns) as *mut _ as *mut std::ffi::c_void,
            (&mut sc) as *mut _ as *mut std::ffi::c_void,
            (&mut nh) as *mut _ as *mut std::ffi::c_void,
            (&mut nk) as *mut _ as *mut std::ffi::c_void,
            (&mut h) as *mut _ as *mut std::ffi::c_void,
        ];
        self.ctx.launch3(
            "q4_qsa_attn_sel4s",
            n_splits.div_ceil(4) as u32,
            (n_head / 12) as u32,
            1,
            256,
            &mut args,
        )?;
        let mut pa2_p = pdev as *mut std::ffi::c_void;
        let mut q2_p = qdev as *mut std::ffi::c_void;
        let mut o2_p = odev as *mut std::ffi::c_void;
        let mut ns2 = n_splits as i32;
        let mut nh2 = n_head as i32;
        let mut h2 = hd as i32;
        let mut margs: Vec<*mut std::ffi::c_void> = vec![
            (&mut pa2_p) as *mut _ as *mut std::ffi::c_void,
            (&mut q2_p) as *mut _ as *mut std::ffi::c_void,
            (&mut o2_p) as *mut _ as *mut std::ffi::c_void,
            (&mut ns2) as *mut _ as *mut std::ffi::c_void,
            (&mut nh2) as *mut _ as *mut std::ffi::c_void,
            (&mut h2) as *mut _ as *mut std::ffi::c_void,
        ];
        self.ctx.launch3(
            "q4_qsa_attn_sel4s_merge",
            n_head.div_ceil(8) as u32,
            1,
            1,
            256,
            &mut margs,
        )?;
        Ok(())
    }

    fn qsa_sel_readback(
        &self,
        sel_idx: u64,
        sel_off: u64,
        list_len: usize,
    ) -> Result<(Vec<u32>, Vec<u32>), String> {
        let mut idx = vec![0u32; list_len];
        let mut off = vec![0u32; 2];
        self.ctx
            .d2h(bytemuck::cast_slice_mut(&mut idx), sel_idx as *const u8)?;
        self.ctx
            .d2h(bytemuck::cast_slice_mut(&mut off), sel_off as *const u8)?;
        Ok((idx, off))
    }

    fn ple_math_dev(
        &self,
        res: u64,
        key: u64,
        value: u64,
        nk: &[f32],
        nq: &[f32],
        nc: &[f32],
        conv_w: &[f32],
        gated: u64,
        conv_out: u64,
        gate_out: u64,
        seq: usize,
        t: usize,
        eps: f32,
        n_embd: usize,
        hc: usize,
        kern: usize,
        dil: usize,
        hist: usize,
        host_ring: &[f32],
    ) -> Result<(), String> {
        if t != 1 {
            return Err(format!("ple_math_dev: t={t} (디코드 전용)"));
        }
        let hc_dim = hc * n_embd;
        let ring_bytes = hist * hc_dim * 4;
        // 링 풀 + 워터마크(되감기면 호스트 링으로 리프레시).
        let rewind = {
            let mut wm = self.ple_ring_pos.lock().map_err(|e| e.to_string())?;
            let w = wm.entry(seq).or_insert(0);
            let rw = *w > t; // pos0=0 재시작(벤치 워밍업 등)
            *w = t;          // t=1: 이번 토큰까지 유효
            rw
        };
        let ring = {
            let mut m = self.ple_ring.lock().map_err(|e| e.to_string())?;
            let g = m.entry(seq).or_insert_with(|| GBuf::new("ple_ring"));
            // 주의: ensure 가 ptr 을 세우므로 최초 판정은 ensure **전**에.
            let fresh = g.ptr.is_null();
            g.ensure(&self.ctx, ring_bytes)?;
            if fresh || rewind {
                // 최초/되감기: 호스트 링(정합 상태)으로 초기화 — 동기 h2d 1회.
                self.ctx.h2d(g.ptr, bytemuck::cast_slice(host_ring))?;
            }
            g.ptr
        };
        let (resp, keyp, valp, gp, cop, gop) = (
            self.fptr(res)?,
            self.fptr(key)?,
            self.fptr(value)?,
            self.fptr(gated)?,
            self.fptr(conv_out)?,
            self.fptr(gate_out)?,
        );
        let nk_d = self.upload_hashed(&self.ple_nk, nk)?;
        let nq_d = self.upload_hashed(&self.ple_nq, nq)?;
        let nc_d = self.upload_hashed(&self.ple_nc, nc)?;
        let cw_d = self.upload_hashed(&self.ple_cw, conv_w)?;
        // (1) gate + 방송 + 그룹 norm — 워프당 (t,s), 레인 0 실행.
        {
            let (mut rp, mut kp, mut vp) = (
                resp as *mut std::ffi::c_void,
                keyp as *mut std::ffi::c_void,
                valp as *mut std::ffi::c_void,
            );
            let (mut nk_, mut nq_, mut nc_) = (
                nk_d as *mut std::ffi::c_void,
                nq_d as *mut std::ffi::c_void,
                nc_d as *mut std::ffi::c_void,
            );
            let (mut gp_, mut gop_) = (gp as *mut std::ffi::c_void, gop as *mut std::ffi::c_void);
            let (mut e, mut ne, mut hcc, mut tt) =
                (eps, n_embd as i32, hc as i32, t as i32);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut rp) as *mut _ as *mut std::ffi::c_void,
                (&mut kp) as *mut _ as *mut std::ffi::c_void,
                (&mut vp) as *mut _ as *mut std::ffi::c_void,
                (&mut nk_) as *mut _ as *mut std::ffi::c_void,
                (&mut nq_) as *mut _ as *mut std::ffi::c_void,
                (&mut nc_) as *mut _ as *mut std::ffi::c_void,
                (&mut gp_) as *mut _ as *mut std::ffi::c_void,
                (&mut gop_) as *mut _ as *mut std::ffi::c_void,
                (&mut e) as *mut _ as *mut std::ffi::c_void,
                (&mut ne) as *mut _ as *mut std::ffi::c_void,
                (&mut hcc) as *mut _ as *mut std::ffi::c_void,
                (&mut tt) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx.launch3(
                "q4_ple_gate",
                hc.div_ceil(8) as u32,
                t as u32,
                1,
                256,
                &mut args,
            )?;
        }
        // (2) dilated conv + silu + 링 갱신.
        {
            let (mut gp_, mut cw_, mut ring_, mut cop_) = (
                gp as *mut std::ffi::c_void,
                cw_d as *mut std::ffi::c_void,
                ring as *mut std::ffi::c_void,
                cop as *mut std::ffi::c_void,
            );
            let (mut hd, mut tt, mut k2, mut d2, mut h2) = (
                hc_dim as i32,
                t as i32,
                kern as i32,
                dil as i32,
                hist as i32,
            );
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut gp_) as *mut _ as *mut std::ffi::c_void,
                (&mut cw_) as *mut _ as *mut std::ffi::c_void,
                (&mut ring_) as *mut _ as *mut std::ffi::c_void,
                (&mut cop_) as *mut _ as *mut std::ffi::c_void,
                (&mut hd) as *mut _ as *mut std::ffi::c_void,
                (&mut tt) as *mut _ as *mut std::ffi::c_void,
                (&mut k2) as *mut _ as *mut std::ffi::c_void,
                (&mut d2) as *mut _ as *mut std::ffi::c_void,
                (&mut h2) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx.launch3(
                "q4_ple_conv",
                hc_dim.div_ceil(256) as u32,
                1,
                1,
                256,
                &mut args,
            )?;
        }
        // (3) 잔차.
        {
            let (mut rp, mut vp, mut gop_, mut cop_) = (
                resp as *mut std::ffi::c_void,
                valp as *mut std::ffi::c_void,
                gop as *mut std::ffi::c_void,
                cop as *mut std::ffi::c_void,
            );
            let (mut ne, mut hcc, mut tt) = (n_embd as i32, hc as i32, t as i32);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut rp) as *mut _ as *mut std::ffi::c_void,
                (&mut vp) as *mut _ as *mut std::ffi::c_void,
                (&mut gop_) as *mut _ as *mut std::ffi::c_void,
                (&mut cop_) as *mut _ as *mut std::ffi::c_void,
                (&mut ne) as *mut _ as *mut std::ffi::c_void,
                (&mut hcc) as *mut _ as *mut std::ffi::c_void,
                (&mut tt) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx.launch3(
                "q4_ple_residual",
                n_embd.div_ceil(256) as u32,
                1,
                1,
                256,
                &mut args,
            )?;
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
        // 전문가 순 그룹화 (2026-09-13): ids는 확률순이라 연속 런이 1행씩
        // 흩어진다(프레임 실측 t=512·k=10 ≈4000런치/층). 카운팅 정렬로 묶어
        // 런치 수를 전문가 수 수준으로 줄인다. x 행은 순열 gather로 모으고,
        // 결과 행 순서는 d2h 후 호스트 산란으로 복원한다(가중합이 원래 행
        // 순서를 요구 — 호스트 비용은 perm 인덱싱뿐).
        let ne = n_expert_stack.max(1);
        let mut off = vec![0usize; ne + 1];
        for &e in expert_ids {
            off[(e as usize).min(ne - 1) + 1] += 1;
        }
        for e in 0..ne {
            off[e + 1] += off[e];
        }
        let mut cur = off[..ne].to_vec();
        let mut perm = vec![0u32; t];
        for (i, &e) in expert_ids.iter().enumerate() {
            let e = (e as usize).min(ne - 1);
            let p = cur[e];
            perm[p] = i as u32;
            cur[e] += 1;
        }
        let row_u32 = if w_f32 { n_in } else { xq_w };
        let xbase = if w_f32 { xdev_f32 } else { xq_buf };
        let xg = {
            let mut g = self.xperm.lock().map_err(|e| e.to_string())?;
            g.ensure(&self.ctx, t * row_u32 * 4)?
        };
        self.rows_permute(xbase, &perm, xg, row_u32, t)?;
        for e in 0..ne {
            let rows = off[e + 1] - off[e];
            if rows == 0 {
                continue;
            }
            let start = off[e];
            let xsrc = unsafe { xg.add(start * row_u32 * 4) };
            let wsrc = unsafe { w_dev.add(e * per_expert) };
            let dst = unsafe { ydev.add(start * n_out * 4) };
            if w_f32 {
                self.launch_gemm_f32(xsrc, wsrc, n_in, n_out, rows, dst)?;
            } else {
                self.launch_gemm(ggml_id(ws.ty), xsrc, wsrc, n_in, n_out, xq_w, rows, dst)?;
            }
        }
        self.ctx.d2h(bytemuck::cast_slice_mut(&mut yflat), ydev)?;
        for (g, v) in yflat.chunks_exact(n_out).enumerate() {
            outs[perm[g] as usize].copy_from_slice(v);
        }
        Ok(())
    }

    /// QSA 마스크드 밀집 GQA (값 경로 브리지) — f32 캐시.
    fn total_mem_bytes(&self) -> u64 {
        let (mut f, mut t) = (0usize, 0usize);
        unsafe {
            if hip::hipMemGetInfo(&mut f, &mut t) != hip::hipError_t_hipSuccess {
                return 0;
            }
        }
        t as u64
    }

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

    #[allow(clippy::too_many_arguments)]
    fn qsa_attention_sel(
        &self,
        q: &[f32],
        ck: &[f32],
        cv: &[f32],
        sel_idx: &[u32],
        sel_off: &[u32],
        kq_scale: f32,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
    ) -> Result<Vec<f32>, String> {
        if std::env::var_os("LLM170_QSA_CPU").is_some() {
            return Err(format!("q4acc: qsa_attention_sel t={t} CPU 강제"));
        }
        // 4헤드-퍼-워프판은 K/V 행을 4헤드가 공유한다(트래픽 1/4) — 프리필에서
        // −20%. 단 t가 작으면(디코드) 블록의 워프 대부분이 놀아 역효과이므로
        // t≤3은 헤드당 워프 1개인 `_sel`로 보낸다. 둘은 비트 동일(프로브 확인).
        // 실측(2026-09-14): 그 강제(LLM170_QSA_SEL4_DEC)는 디코드에서 중립이었다
        // (589.7/562.2 vs 576.3/566.2ms, 토큰 동일) — 점유율 손실이 트래픽 이득을 상쇄.
        // 장문맥 디코드의 실제 비용은 아래와 같다(pp8192, KTRACE):
        //   q4_qsa_attn_sel = 1.425 ms/콜 = 17.1 ms/스텝(커널 합 83.9ms의 20%, 최대 단일)
        //   = 8192위치 x 256 x 2(K,V) x 24헤드 = 403 MB/층 -> 236 GB/s로 1.7ms ≈ 측정치
        // 즉 **헤드 24개가 같은 K/V 행을 각자 다시 읽는 대역폭 문제**다. 4헤드 공유로는
        // 점유율 때문에 안 되고, flash-decoding형(선택목록을 블록 간 분할 + 부분 softmax
        // 병합)으로 K/V를 1회만 읽어야 한다 — 17.1ms -> ~1ms, 스텝의 ~8%.
        // t=1 분할판은 기본 ON이다(장문맥 디코드 142.5 -> 124.4 ms/스텝 = -12.7%,
        // diverse 스트림 완전 동일, 단문맥 무회귀). 비트 동일 경로 복귀는
        // LLM170_QSA_SPLIT=0, 분할 상한은 LLM170_QSA_SPLITS(기본 64, 목록/32로 적응).
        if t == 1 && std::env::var("LLM170_QSA_SPLIT").as_deref() != Ok("0") {
            // 위치 분할(flash-decoding형) — 지연 바운드인 t=1을 (split, 헤드묶음)
            // 그리드로 펼친다. 부분 소프트맥스를 2차 커널이 병합하므로 합산
            // 순서가 달라진다(greedy 스트림 동일성으로 검증, 비트 동일 아님).
            self.qsa_attn_sel4s_raw(q, ck, cv, sel_idx, sel_off, kq_scale, n_head, n_kv, hd, t)
        } else if t <= 3 {
            self.qsa_attn_sel_raw(q, ck, cv, sel_idx, sel_off, kq_scale, n_head, n_kv, hd, t)
        } else {
            self.qsa_attn_sel4_raw(q, ck, cv, sel_idx, sel_off, kq_scale, n_head, n_kv, hd, t)
        }
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

    /// [t][vocab] logits 행별 GPU argmax — np greedy 판정 (plans/74 N1).
    /// argmax64 = CPU greedy와 동일 의미(동률 최저 인덱스).
    fn frame_argmax_rows(&self, logits: u64, t: usize, vocab: usize) -> Result<Vec<u32>, String> {
        let base = self.fptr(logits)?;
        let sc = self.ctx.scratch(t.max(1) * 8)?;
        for s in 0..t {
            let mut xp = unsafe { base.add(s * vocab * 4) } as *mut std::ffi::c_void;
            let mut n2 = vocab as i32;
            let mut op = unsafe { sc.add(s * 8) } as *mut std::ffi::c_void;
            let mut args = vec![
                (&mut xp) as *mut _ as *mut std::ffi::c_void,
                (&mut n2) as *mut _ as *mut std::ffi::c_void,
                (&mut op) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx.launch3("argmax64", 1, 1, 1, 64, &mut args)?;
        }
        let mut r8 = vec![0u8; t * 8];
        self.ctx.d2h(&mut r8, sc)?;
        Ok((0..t)
            .map(|s| {
                let b = &r8[s * 8..s * 8 + 8];
                u32::from_le_bytes([b[4], b[5], b[6], b[7]])
            })
            .collect())
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
            // llama MMQ 우선(옵트인) — 같은 입력을 여러 커널이 공유하는 그룹이라
            // 항목별로 MMQ 가능 타입이면 MMQ를 쓰고 나머지는 기존 타일로 간다.
            let mmq_on = t >= 32 && std::env::var_os("LLM170_Q4_MMQ").is_some();
            let (xq, xq_w) = if mmq_on {
                (std::ptr::null_mut(), 0usize)
            } else {
                self.frame_quant(xp, ws[0].n_in as usize, t)?
            };
            for (w, o) in ws.iter().zip(outs) {
                let (wd, _) = self.dev_weight(w)?;
                let op = self.fptr(*o)?;
                let ty = ggml_id(w.ty);
                let n_in = w.n_in as usize;
                let n_out = w.n_out as usize;
                if mmq_on && matches!(ty, 12 | 13 | 14 | 23)
                    && self.ctx.gemm_mmq(ty, xp as *const u8, wd, n_in, n_out, t, op).is_ok()
                {
                    continue;
                }
                // f16 경로 미검증(위 frame_gemm 주석 참조) — 배선 보류.
                let (xqi, xwi) = if mmq_on { self.frame_quant(xp, n_in, t)? } else { (xq, xq_w) };
                self.launch_gemm(ty, xqi, wd, n_in, n_out, xwi, t, op)?;
            }
            return Ok(());
        }
        // plans/71: q8_0 가중치 + t>=32는 MMQ(int8 dp4a) — f32 활성을 직접 받아
        // 자체 양자화. 종전 j128 타일 대비 측정 이득은 벤치로 검증.
        for (w, o) in ws.iter().zip(outs) {
            let op = self.fptr(*o)?;
            if w.ty == GgmlType::Q8_0 && t >= 32
                && std::env::var("LLM170_Q8MMQ").as_deref() == Ok("1")
            {
                let (wd, _) = self.dev_weight(w)?;
                self.ctx
                    .gemm_mmq(8, xp as *const u8, wd, w.n_in as usize, w.n_out as usize, t, op)
                    .map_err(|e| format!("q8mmq: {e}"))?;
                continue;
            }
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
                // plans/73: 융합 판은 측정 역행(16.78→16.28 t/s) — 옵트인 자산.
                // 워프=행의 320-원소 직렬 f32 체인이 part/finish 의 병렬 2런치보다 느리다.
                if rows <= 32 && std::env::var_os("LLM170_RMSSMALL").is_some() {
                    let mut xa = xp;
                    let mut wa = wp;
                    let mut op_ = self.fptr(out)?;
                    let mut e = eps;
                    let mut nn = n as i32;
                    let (mut rws, mut rr) = (rows as i32, w_reps as i32);
                    return self.kop(
                        "rms_small",
                        rows.div_ceil(8) as u32,
                        1,
                        1,
                        256,
                        &mut cargs!(&mut xa, &mut wa, &mut op_, &mut e, &mut nn, &mut rws, &mut rr),
                    );
                }
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
            O::L2Rows { x, eps, d, n } => {
                let mut xp = self.fptr(x)?;
                let mut e = eps;
                let mut dd = d as i32;
                // 행 수는 *토큰 수*에서 온다. 버퍼 길이(t_max)를 쓰면 t=1에서도
                // t_max행을 처리해 33.7ms/스텝을 낭비한다(2026-09-14 실측).
                let rows = (n / d).max(1) as u32;
                self.kop("q4_l2_rows", rows, 1, 1, 32, &mut cargs!(&mut xp, &mut e, &mut dd))
            }
            O::Scale { t, s, n } => {
                let mut p = self.fptr(t)?;
                let mut ss = s;
                let mut nn = n as i32;
                self.kop("q4_scale", (n as u32).div_ceil(128), 1, 1, 128, &mut cargs!(&mut p, &mut ss, &mut nn))
            }
            O::BcastRows { src, dst, n, rows } => {
                let (mut sp, mut dp) = (self.fptr(src)?, self.fptr(dst)?);
                let (mut nn, mut rr) = (n as i32, rows as i32);
                self.kop("bcast_rows", (n as u32).div_ceil(128), rows as u32, 1, 128, &mut cargs!(&mut sp, &mut dp, &mut nn, &mut rr))
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
                // 라우팅이 새로 쓰였다 — 그룹화 캐시 무효화.
                self.moe_gen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let (mut rp, mut ip, mut wp) = (self.fptr(route)?, self.fptr(ids)?, self.fptr(wt)?);
                let mut ne = n_exp as i32;
                let mut ks = k_sel as i32;
                let t = self.t_cur();
                // 워프 병렬판(2026-09-13) — 원판은 1스레드/토큰이라 디코드에서
                // 0.85ms/호출이었다(토큰당 48콜 = 41ms). 선택 로직은 동일해
                // 결과는 비트 동일.
                self.kop("q4_moe_top10_m", t as u32, 1, 1, 32, &mut cargs!(&mut rp, &mut ip, &mut wp, &mut ne, &mut ks))
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
    eprintln!("{}", crate::rawhip::probes::device_report(&a.ctx));
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
/// `q4-ple-check` — q4_ple_gate 커널 ↔ 호스트 산술 미러 대조(합성 입력).
pub fn ple_gate_check() -> Result<String, String> {
    use std::ffi::c_void;
    let ctx = RawCtx::new()?;
    let (n_embd, hc) = (2560usize, 4usize);
    let hc_dim = hc * n_embd;
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
    };
    let res: Vec<f32> = (0..hc_dim).map(|_| lcg()).collect();
    let key: Vec<f32> = (0..hc_dim).map(|_| lcg()).collect();
    let val: Vec<f32> = (0..n_embd).map(|_| lcg()).collect();
    let nk: Vec<f32> = (0..hc_dim).map(|_| 0.8 + lcg().abs()).collect();
    let nq: Vec<f32> = (0..hc_dim).map(|_| 0.8 + lcg().abs()).collect();
    let nc: Vec<f32> = (0..hc_dim).map(|_| 0.8 + lcg().abs()).collect();
    let rd = ctx.alloc(hc_dim * 4)?;
    let kd = ctx.alloc(hc_dim * 4)?;
    let vd = ctx.alloc(n_embd * 4)?;
    let nkd = ctx.alloc(hc_dim * 4)?;
    let nqd = ctx.alloc(hc_dim * 4)?;
    let ncd = ctx.alloc(hc_dim * 4)?;
    let gd = ctx.alloc(hc_dim * 4)?;
    let god = ctx.alloc(hc * 4)?;
    ctx.h2d(rd, bytemuck::cast_slice(&res))?;
    ctx.h2d(kd, bytemuck::cast_slice(&key))?;
    ctx.h2d(vd, bytemuck::cast_slice(&val))?;
    ctx.h2d(nkd, bytemuck::cast_slice(&nk))?;
    ctx.h2d(nqd, bytemuck::cast_slice(&nq))?;
    ctx.h2d(ncd, bytemuck::cast_slice(&nc))?;
    let (mut rp, mut kp, mut vp, mut nk_, mut nq_, mut nc_, mut gp_, mut gop_) = (
        rd as *mut c_void, kd as *mut c_void, vd as *mut c_void,
        nkd as *mut c_void, nqd as *mut c_void, ncd as *mut c_void,
        gd as *mut c_void, god as *mut c_void,
    );
    let (mut e, mut ne, mut hcc, mut tt) = (1e-6f32, n_embd as i32, hc as i32, 1i32);
    let mut args: Vec<*mut c_void> = vec![
        &mut rp as *mut _ as *mut c_void, &mut kp as *mut _ as *mut c_void,
        &mut vp as *mut _ as *mut c_void, &mut nk_ as *mut _ as *mut c_void,
        &mut nq_ as *mut _ as *mut c_void, &mut nc_ as *mut _ as *mut c_void,
        &mut gp_ as *mut _ as *mut c_void, &mut gop_ as *mut _ as *mut c_void,
        &mut e as *mut _ as *mut c_void, &mut ne as *mut _ as *mut c_void,
        &mut hcc as *mut _ as *mut c_void, &mut tt as *mut _ as *mut c_void,
    ];
    ctx.launch3("q4_ple_gate", hc.div_ceil(8) as u32, 1, 1, 256, &mut args)?;
    ctx.sync()?;
    let mut dgate = vec![0f32; hc];
    let mut dgated = vec![0f32; hc_dim];
    ctx.d2h(bytemuck::cast_slice_mut(&mut dgate).as_mut(), god)?;
    ctx.d2h(bytemuck::cast_slice_mut(&mut dgated).as_mut(), gd)?;
    // 호스트 미러(ple_block 산술)
    let eps = 1e-6f32;
    let mut out = String::new();
    for s in 0..hc {
        let kn = llm170_core::ops::rms_norm(&key[s * n_embd..(s + 1) * n_embd], &nk[s * n_embd..(s + 1) * n_embd], eps);
        let qn = llm170_core::ops::rms_norm(&res[s * n_embd..(s + 1) * n_embd], &nq[s * n_embd..(s + 1) * n_embd], eps);
        let mut dot = 0.0f32;
        for i in 0..n_embd { dot += kn[i] * qn[i]; }
        dot /= (n_embd as f32).sqrt();
        let mag = dot.abs().max(1e-6).sqrt();
        let g = llm170_core::ops::sigmoid(if dot >= 0.0 { mag } else { -mag });
        let mut gated: Vec<f32> = (0..n_embd).map(|i| val[i] * g).collect();
        let sg = {
            let sum = llm170_core::ops::sq_sum(&gated);
            1.0 / ((sum / n_embd as f64 + eps as f64).sqrt() as f32)
        };
        for i in 0..n_embd { gated[i] = gated[i] * sg * nc[s * n_embd + i]; }
        let gmax = gated.iter().zip(dgated[s * n_embd..(s + 1) * n_embd].iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        out += &format!("s{s}: gate dev={:.6} host={:.6} (dot={:.4}) gated max|d-h|={gmax:.2e}\n", dgate[s], g, dot);
    }
    Ok(out)
}

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
    // 프레임 AR은 전치 상태 레이아웃(gdn_ar_w_swap) — 업로드 전치, 판독 후 복원.
    let st0_t = llm170_core::qwen4exp::frame::Frame4::transpose_pairs(&st0, d);
    acc.frame_write(hst, &st0_t)?;
    acc.frame_gdn_ar(hq, hk, hv, hbg, hst, ho, 1, n_group, dt_rank, d)?;
    let mut o_gpu = vec![0.0f32; v_len * t];
    acc.frame_read(ho, &mut o_gpu)?;
    let mut st_gpu_t = vec![0.0f32; st0.len()];
    acc.frame_read(hst, &mut st_gpu_t)?;
    let st_gpu = llm170_core::qwen4exp::frame::Frame4::transpose_pairs(&st_gpu_t, d);
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
    // 선택-목록판 — 같은 마스크에서 오름차순 목록을 만들어 마스크판과 대조한다.
    // 산술 순서가 같으므로 **비트 동일**이 기대값이다(다르면 목록/커널 버그).
    let mut sel_idx: Vec<u32> = Vec::new();
    let mut sel_off: Vec<u32> = vec![0];
    for tok in 0..t {
        for p in 0..n_past {
            if mask[tok * n_past + p] != 0 {
                sel_idx.push(p as u32);
            }
        }
        sel_off.push(sel_idx.len() as u32);
    }
    let gpu_sel = acc.qsa_attn_sel_raw(
        &q, &ck, &cv, &sel_idx, &sel_off, kq_scale, n_head, n_kv, hd, t,
    )?;
    let mut bit_diff = 0usize;
    let mut maxrel_sel = 0.0f64;
    let mut maxrel_cpu = 0.0f64;
    for ((&a, &b), &c) in gpu.iter().zip(&gpu_sel).zip(&cpu) {
        if a != b {
            bit_diff += 1;
        }
        let d = ((a - c).abs() as f64) / (c.abs().max(1e-3) as f64);
        maxrel_sel = maxrel_sel.max(d);
        let d2 = ((b - c).abs() as f64) / (c.abs().max(1e-3) as f64);
        maxrel_cpu = maxrel_cpu.max(d2);
    }
    let gpu_sel4 = acc.qsa_attn_sel4_raw(
        &q, &ck, &cv, &sel_idx, &sel_off, kq_scale, n_head, n_kv, hd, t,
    )?;
    let mut bit_diff4 = 0usize;
    let mut maxrel_sel4 = 0.0f64;
    for (&b, &c) in gpu_sel.iter().zip(&gpu_sel4) {
        if b != c {
            bit_diff4 += 1;
        }
        let d = ((c - b).abs() as f64) / (b.abs().max(1e-3) as f64);
        maxrel_sel4 = maxrel_sel4.max(d);
    }
    Ok(format!(
        "q4-qsa-check t={t} n_past={n_past}: nonfinite={nonfinite} mismatch={nz}/{} maxrel={maxrel:.3e} | sel: bit_diff={bit_diff}/{} sel4_bit_diff={bit_diff4}/{} maxrel_sel4_vs_sel={maxrel_sel4:.3e} maxrel_sel_vs_cpu={maxrel_sel:.3e} maxrel_mask_vs_cpu={maxrel_cpu:.3e} sel_keys={} (스캔 {}키 대비 {:.1}배 적음)",
        gpu.len(),
        gpu_sel.len(),
        gpu_sel4.len(),
        sel_idx.len(),
        t * n_past,
        (t * n_past) as f64 / (sel_idx.len().max(1)) as f64,
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
    if std::env::var_os("LLM170_Q4ACC_ROWDBG").is_some() {
        for &ri in &[0usize, 1, 2, 127, 128, 129, 130, 199, 200, 201, 255] {
            if ri >= gpu.len() {
                continue;
            }
            let m = gpu[ri]
                .iter()
                .zip(&cpu[ri])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!("# row {ri}: maxerr={m:.5}");
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
