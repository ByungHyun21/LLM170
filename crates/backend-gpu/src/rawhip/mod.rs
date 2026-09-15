//! 원시 HIP 실행기 — cubecl을 거치지 않는 직접 경로 (2026-09-03 재작성).
//! hipRTC로 임베디드 HIP C++ 소스를 컴파일하고 hipModuleLaunchKernel로
//! 실행. 버퍼는 영속 아레나(해제 없음, ADR-0014 동일 규칙). 커널 산술은
//! core 미러(dot_row_w4a8_*_lane)와 동일 연산열 — to_bits 검증 게이트.
#![allow(dead_code)] // 프론트 정리(2026-09-14): 레거시·진단 경로 보존

use cubecl_hip_sys as hip;
use std::collections::HashMap;
use std::ffi::CString;

pub mod decode;
pub mod kernels;
pub mod q4acc;
pub mod vit;

/// 로드된 오프라인 타일 코드오브젝트 패밀리 (임베딩 or LLM170_CO*_PATH
/// 오버라이드). RawCtx::new 완료 후 불변. 타일 발사 게이트는 env가 아니라
/// 이 비트를 본다 — 무환경 기본 성능 = 튜닝 성능.
pub const CO_J128: u8 = 1; // w32b.co: *_j128 계열 (t≤128)
pub const CO_V4: u8 = 2; // v4all.co: *_v4 + *_wm 4종
pub const CO_ODD: u8 = 4; // odd_all.co: nl/q3k/iq3s v4 (plans/04)
pub const CO_MMQ: u8 = 8; // mmq.co: llama mul_mat_q<q4_K/q5_K,128> + mmq_quant_y
pub const CO_MMQ2: u8 = 16; // mmq2.co: gemm_f16_v4 (deq-f16 경로)
pub const CO_MMQ3: u8 = 32; // mmq3.co: llama 프로덕션 mul_mat_q<iq4_xs>
pub const CO_MMQ8: u8 = 64; // mmq8.co: ROCm 10 fatbin의 mul_mat_q<q8_0>(plans/71)
pub const CO_QY: u8 = 128; // quanty_new.co: ROCm 10 quantize_mmq_q8_1<D4/DS4>(plans/71)
static CO_FAM: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

pub fn co_loaded(bit: u8) -> bool {
    CO_FAM.load(std::sync::atomic::Ordering::Relaxed) & bit != 0
}


fn name_leak(n: &str) -> &'static str {
    // launch3 호출부의 name은 리터럴 — 그대로 반환 (비-리터럴 경로는 트레이스 스킵 허용)
    unsafe { std::mem::transmute::<&str, &'static str>(n) }
}

pub struct KtraceEv(pub &'static str, pub usize, pub u32);  // name, event, gy
pub static KTRACE: std::sync::Mutex<Option<Vec<KtraceEv>>> = std::sync::Mutex::new(None);
/// MMQ mul_mat_q 동적 smem 상한 설정 캐시 — 런치마다 드라이버 호출하지 않도록.
/// (hipFuncSetAttribute는 커널 로드 갱신을 유발할 수 있어 GEMM마다 부르면 손해)
static MMQ_SMEM_SET: std::sync::LazyLock<std::sync::Mutex<std::collections::HashSet<(usize, i32)>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashSet::new()));
fn ck(status: hip::hipError_t, what: &str) -> Result<(), String> {
    if status == hip::hipError_t_hipSuccess {
        Ok(())
    } else {
        Err(format!("rawhip: {what}: {status:?}"))
    }
}

static AOUT_DUMPED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub fn aout_dumped() -> bool {
    let r = AOUT_DUMPED.swap(true, std::sync::atomic::Ordering::SeqCst);
    !r
}

pub fn ktrace_on() { *KTRACE.lock().unwrap() = Some(Vec::new()); }

// ── 프레임 그래프 캡처(디코드 스텝) ────────────────────────────────────────
// 스텝 내 호스트 왕복(capture_mark)을 경계로 스트림 캡처를 세그먼트로 끊어
// 그래프로 굳히고, 재생 시에는 런치 함수가 즉시 반환되어 커널이 그래프에서
// 실행된다(런치 ~3천 회/스텝 → 세그먼트 수 회). 프로세스 전역 — CLI는 가속기
// 1개, 재생은 스텝 단위 단일 스레드라 전역으로 충분하다.
pub enum GraphMode {
    Off,
    Capture { segs: Vec<hip::hipGraph_t>, open: bool },
    Replay { execs: Vec<hip::hipGraphExec_t>, idx: usize },
}
// SAFETY: 그래프 핸들은 디바이스 객체 — HIP 런타임이 직렬화하며, 재생은 스텝
// 단위로 단일 스레드에서만 일어난다.
unsafe impl Send for GraphMode {}
pub static GRAPH: std::sync::Mutex<GraphMode> = std::sync::Mutex::new(GraphMode::Off);
pub static GRAPH_SKIP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static NOLAUNCH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

#[inline]
pub fn nolaunch_on() -> bool {
    *NOLAUNCH.get_or_init(|| std::env::var_os("LLM170_NOLAUNCH").is_some())
}

/// 캡처 시작 — BeginCapture는 **첫 마크에서** 건다. 마크 이전 구간(임베딩 h2d 등
/// 캡처 불가 연산)을 캡처 밖에 두기 위해서다.
pub fn graph_capture_begin(_stream: hip::hipStream_t) -> Result<(), String> {
    *GRAPH.lock().map_err(|e| e.to_string())? =
        GraphMode::Capture { segs: Vec::new(), open: false };
    Ok(())
}

/// 캡처 종료 — 마지막 세그먼트를 닫고 전부 instantiate, 재생 모드로 전환.
pub fn graph_capture_end(stream: hip::hipStream_t) -> Result<(), String> {
    let mut g = GRAPH.lock().map_err(|e| e.to_string())?;
    let GraphMode::Capture { segs, open } = &mut *g else {
        return Err("graph_capture_end: 캡처 중이 아님".into());
    };
    let mut segs = std::mem::take(segs);
    if *open {
        unsafe {
            let mut gr: hip::hipGraph_t = std::ptr::null_mut();
            ck(hip::hipStreamEndCapture(stream, &mut gr), "EndCapture")?;
            segs.push(gr);
        }
        *open = false;
    }
    let mut execs = Vec::with_capacity(segs.len());
    for g0 in &segs {
        unsafe {
            let mut ex: hip::hipGraphExec_t = std::ptr::null_mut();
            ck(hip::hipGraphInstantiate(&mut ex, *g0, std::ptr::null_mut(), std::ptr::null_mut(), 0), "GraphInstantiate")?;
            execs.push(ex);
        }
    }
    eprintln!("# graph: 세그먼트 {}개 캡처·인스턴스화", execs.len());
    *g = GraphMode::Replay { execs, idx: 0 };
    Ok(())
}

/// 재생 모드 진입/이탈 — 진입 시 런치 함수가 커널 발사를 건너뛴다(그래프가 실행).
pub fn graph_replay(on: bool) -> Result<(), String> {
    if on {
        let mut g = GRAPH.lock().map_err(|e| e.to_string())?;
        if let GraphMode::Replay { idx, .. } = &mut *g {
            *idx = 0;
        }
    }
    GRAPH_SKIP.store(on, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

/// 세그먼트 경계 — 호스트 왕복 직전에 호출. 캡처 중이면 현재 세그먼트를 닫고
/// 다음을 열고, 재생 중이면 앞 세그먼트 그래프를 발사한다.
#[inline]
fn is_end(tag: &str) -> bool {
    tag.ends_with("_in")
}

pub fn capture_mark(stream: hip::hipStream_t, tag: &str) -> Result<(), String> {
    let dbg = std::env::var_os("LLM170_GRAPH_DEBUG").is_some();
    let mut g = GRAPH.lock().map_err(|e| e.to_string())?;
    match &mut *g {
        GraphMode::Off => Ok(()),
        // 규약: `*_in` = 호스트 왕복 *직전* → 현재 세그먼트 종료(그래프 굳힘).
        //       `*_out` = 왕복 *직후* → 다음 세그먼트 시작. 왕복 구간(d2h/h2d)은
        //       어느 그래프에도 들어가지 않는다(캡처 불가 연산).
        GraphMode::Capture { segs, open } => unsafe {
            if is_end(tag) {
                if *open {
                    let mut gr: hip::hipGraph_t = std::ptr::null_mut();
                    let r = hip::hipStreamEndCapture(stream, &mut gr);
                    if r != hip::hipError_t_hipSuccess {
                        return Err(format!("EndCapture 실패({r:?}) tag={tag} seg={}", segs.len()));
                    }
                    segs.push(gr);
                    *open = false;
                    if dbg {
                        eprintln!("# graph-mark {tag} 종료 (seg {})", segs.len());
                    }
                }
            } else if !*open {
                let r = hip::hipStreamBeginCapture(stream, hip::hipStreamCaptureMode_hipStreamCaptureModeThreadLocal);
                if r != hip::hipError_t_hipSuccess {
                    return Err(format!("BeginCapture 실패({r:?}) tag={tag}"));
                }
                *open = true;
                if dbg {
                    eprintln!("# graph-mark {tag} 시작 (seg {})", segs.len());
                }
            }
            Ok(())
        },
        GraphMode::Replay { execs, idx } => {
            // 세그먼트는 `*_in`(종료) 지점에서 발사된다 — 그 그래프가 직전 구간.
            if is_end(tag) && *idx < execs.len() {
                unsafe { ck(hip::hipGraphLaunch(execs[*idx], stream), "GraphLaunch")?; }
                *idx += 1;
            }
            Ok(())
        }
    }
}
pub fn ktrace_dump() -> String {
    let mut g = KTRACE.lock().unwrap();
    // ktrace_on 없이 호출되면(스펙 경로 등) 빈 문자열 — 과거 unwrap 패닉
    let Some(slot) = g.as_mut() else { return String::new() };
    let evs = std::mem::take(slot);
    // 이벤트 핸들 정리 — 파괴하지 않으면 hipEvent 풀이 고갈되어(런치당 2개 생성,
    // 13k 런치) 이후 생성이 실패하고 트레이스에서 통째로 누락된다(2026-09-14 규명).
    struct Evs(Vec<KtraceEv>);
    impl Drop for Evs {
        fn drop(&mut self) {
            unsafe {
                for e in &self.0 {
                    let _ = hip::hipEventDestroy(e.1 as *mut _);
                }
            }
        }
    }
    let _evs_guard = Evs(evs);
    let evs = &_evs_guard.0;
    let mut out = String::new();
    // 쌍 결합: 연속 동일 (name, gy) 두 이벤트가 start/end
    let mut sums: std::collections::HashMap<(&str, u32), (f64, u32)> = std::collections::HashMap::new();
    let mut total = 0.0f64;
    let mut gaps = 0.0f64;
    // 쌍은 **고정 stride-2**다: 런치마다 (start, end)를 정확히 2개 기록한다.
    // 종전 휴리스틱("연속 같은 (name,gy) = 쌍")은 같은 커널이 연속 런치될 때
    // end→start를 한 쌍으로 묶어 합계를 통째로 어긋나게 했다 — 배치 형상에서
    // 흔하고, 27B pp512에서 합계 366ms 대 벽 1,409ms(4배 과소)로 나타났다
    // (2026-09-14 규명). 갭도 같은 순회에서 end(k) → start(k+1)로 잰다.
    let npair = evs.len() / 2;
    unsafe {
        for k in 0..npair {
            let (st, en) = (&evs[2 * k], &evs[2 * k + 1]);
            // 짝이 어긋난 런치(다른 커널과 섞임)면 방어적으로 건너뛴다.
            if st.0 != en.0 || st.2 != en.2 {
                continue;
            }
            let mut ms = 0f32;
            if hip::hipEventElapsedTime(&mut ms, st.1 as *mut _, en.1 as *mut _) == hip::hipError_t_hipSuccess {
                let ent = sums.entry((st.0, st.2)).or_insert((0.0, 0));
                ent.0 += ms as f64;
                ent.1 += 1;
                total += ms as f64;
            }
            if k + 1 < npair {
                let nst = &evs[2 * (k + 1)];
                let mut gm = 0f32;
                if hip::hipEventElapsedTime(&mut gm, en.1 as *mut _, nst.1 as *mut _) == hip::hipError_t_hipSuccess {
                    gaps += gm as f64;
                }
            }
        }
        // 런치 갭: end(N)→start(N+1) 같은 스트림 상 연속
        let mut gap_by_pred: std::collections::HashMap<&str, (f64, u32)> = std::collections::HashMap::new();
        let mut gap_tot = 0.0f64;
        let mut prev_end: Option<(usize, &str)> = None;
        for k in 0..evs.len()/2 {
            let (st, en) = (&evs[2*k], &evs[2*k+1]);
            if let Some((pe, pn)) = prev_end {
                let mut ms = 0f32;
                if hip::hipEventElapsedTime(&mut ms, pe as *mut _, st.1 as *mut _) == hip::hipError_t_hipSuccess && ms > 0.0 {
                    let e2 = gap_by_pred.entry(pn).or_insert((0.0, 0));
                    e2.0 += ms as f64; e2.1 += 1;
                    gap_tot += ms as f64;
                }
            }
            prev_end = Some((en.1, en.0));
        }
        // 런치 **순서** 덤프 (LLM170_KTRACE_SEQ=N): 갭의 주인은 전임자가 아니라
        // (2026-09-14: 이 덤프로 MoE 묶음이 gather → gate GEMM → scatter → up GEMM …
        //  순서임을 확인해 그룹화 호스트 경로를 특정했다)
        // **후속 op의 호스트 비용**이므로(갭이 후속에 따라 달라진다), 순서를 봐야
        // 어떤 op인지 특정된다.
        if let Ok(v) = std::env::var("LLM170_KTRACE_SEQ") {
            let n: usize = v.parse().unwrap_or(64);
            for k in 0..npair.min(n) {
                let (st, en) = (&evs[2 * k], &evs[2 * k + 1]);
                let mut sms = 0f32;
                let _ = hip::hipEventElapsedTime(&mut sms, st.1 as *mut _, en.1 as *mut _);
                out.push_str(&format!("# seq {k:4} {:<28} gy={:<6} {:8.3}ms\n", st.0, st.2, sms));
            }
        }
        for e in evs.iter() { hip::hipEventDestroy(e.1 as *mut _); }
        out.push_str(&format!("LAUNCH GAPS total {:.1}ms\n", gap_tot));
        let mut gv: Vec<_> = gap_by_pred.iter().collect();
        gv.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
        for (n, (ms, c)) in gv.iter().take(12) {
            out.push_str(&format!("  after {:26} {:8.1}ms x{:4}\n", n, ms, c));
        }
    }
    let mut v: Vec<_> = sums.iter().collect();
    v.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
    for ((n, gy), (ms, cnt)) in v.iter().take(40) {
        out.push_str(&format!("{:30} gy={:4} {:9.3}ms x{:4}\n", n, gy, ms, cnt));
    }
    out.push_str(&format!("TOTAL {:.1}ms GAPS {:.1}ms\n", total, gaps));
    out
}

/// 컴파일된 커널 실행기.
pub struct RawCtx {
    module: hip::hipModule_t,
    fns: HashMap<&'static str, hip::hipFunction_t>,
    stream: hip::hipStream_t,
    stream2: hip::hipStream_t,
    /// 크기별 스크래치 풀 — 해제 없는 재사용 (호출마다 신규 할당이
    /// 메모리 고갈→illegal address 유발, 2026-09-03 RCA).
    /// MMQ 전용 y 버퍼 (size, ptr) — 풀 충돌 격리.
    mmq_y: std::sync::Mutex<(usize, *mut u8)>,
    /// side-stream MMQ 전용 y 버퍼 (스트림 레이스 격리).
    mmq_y_s: std::sync::Mutex<(usize, *mut u8)>,
    /// q6→f16 전개 캐시 (w주소 → f16 버퍼).
    f16_cache: std::sync::Mutex<std::collections::HashMap<usize, *mut u8>>,
    /// hipMalloc 범위 등록부 — h2d 실패 시 목적지가 살아있는 할당 안인지 보고한다.
    /// 2026-09-14: 이것으로 "실패한 목적지가 정상 할당 내"임을 한 번에 확인해
    /// 원인을 커널 런치로 좁혔다(해제는 하지 않으므로 목록은 영구, 수백 개 수준).
    allocs: std::sync::Mutex<Vec<(usize, usize)>>,
    ar_cache: std::sync::Mutex<Option<(*mut u8, *mut u8, *mut u8, *mut u8)>>,
    mmq_y_cache: std::sync::Mutex<(u64, usize, usize)>,  // (epoch, y_ptr, y_bytes) — 부록81 (yb 재사용은 호출부)
    /// q6 정준 재배열 캐시.
    canon_q6: std::sync::Mutex<std::collections::HashMap<usize, *mut u8>>,
    /// f16 경로 xq 버퍼 (size, ptr).
    mmq_y2: std::sync::Mutex<(usize, *mut u8)>,
    scratch: std::sync::Mutex<HashMap<usize, Vec<*mut u8>>>,
    cursors: std::sync::Mutex<HashMap<usize, usize>>,
    /// D2H 핀 스테이징 (필요시 성장, 해제 없음 — ADR-0014).
    /// pageable 버퍼로의 hipMemcpyAsync D2H는 슬로패스(1MB에 ~90ms,
    /// 2026-09-05 tg RCA) — 핀 버퍼 경유로 원소복사.
    pinned: std::sync::Mutex<(usize, *mut u8)>,
    /// d2h_issue(비동기) 전용 핀 — 동기 d2h와 버퍼를 공유하면 issue→wait 사이의
    /// 어느 동기 판독이든 내용을 덮어쓴다(plans/68 12차: b가 실수값으로 오염돼
    /// 폴백 행 수 1.04억 → HIP 700. t=1 프로덕션에도 잠복 경쟁이었다).
    pinned_a: std::sync::Mutex<(usize, *mut u8)>,
}

/// 타일 발사 파라미터 (스택 로컬 소유 — args 포인터 유효성 보장).
struct TileLaunch {
    kern: &'static str,
    xp: *mut std::ffi::c_void,
    wp: *mut std::ffi::c_void,
    op: *mut std::ffi::c_void,
    ktp: *mut std::ffi::c_void,
    ni: i32,
    no: i32,
    xw: i32,
    tt: i32,
    gx: u32,
    gz: u32,
    block: u32,
    ktab: bool,
}


impl RawCtx {
    pub fn new() -> Result<Self, String> {
        unsafe {
            ck(hip::hipSetDevice(0), "hipSetDevice")?;
            let _ = hip::hipSetDeviceFlags(hip::hipDeviceScheduleSpin);

            let src = CString::new(kernels::SRC).unwrap();
            let mut prog: hip::hiprtcProgram = std::ptr::null_mut();
            let rs = hip::hiprtcCreateProgram(&mut prog, src.as_ptr(), std::ptr::null(), 0, std::ptr::null_mut(), std::ptr::null_mut());
            if rs != hip::hiprtcResult_HIPRTC_SUCCESS {
                return Err(format!("hiprtcCreateProgram: {rs:?}"));
            }
            // LLM170_HIP_INC: hipconfig 서브프로세스 없이 include 경로를 준다
            // (rocprof 등 서브프로세스를 방해하는 도구 아래에서 필요).
            let inc = match std::env::var_os("LLM170_HIP_INC") {
                Some(v) => v.to_string_lossy().into_owned(),
                None => cubecl_hip_sys::get_hip_include_path().map_err(|e| e.to_string())?,
            };
            let o1 = CString::new(format!("-I{inc}")).unwrap();
            let o2 = CString::new("--std=c++17").unwrap();
            let o3 = CString::new("-O3").unwrap();
            // FMA 수축 차단 — CPU 비트계약 (a+=b*c 축약이 비트 불일치,
            // 2026-09-03 AR xor RCA)
            let o4 = CString::new("-ffp-contract=off").unwrap();
            let o5 = CString::new("-I/opt/rocm/include").unwrap();
            // exp_cr 기본을 디바이스 __expf로 (f64 호너 제거, 2026-09-12).
            // 효과: tg +1.0%, pp +0.55%, judge 16/19 -> 17/19 (llama와 더 가까움).
            // LLM170_EXACTEXP=1이면 glibc 비트일치 f64 경로 복원.
            let ofast = CString::new("-DLLM170_FASTEXP").unwrap();
            let fastexp = std::env::var_os("LLM170_EXACTEXP").is_none();
            let mut opts = vec![o1.as_ptr(), o2.as_ptr(), o3.as_ptr(), o4.as_ptr(), o5.as_ptr()];
            if fastexp { opts.push(ofast.as_ptr()); }
            let rs = hip::hiprtcCompileProgram(prog, opts.len() as i32, opts.as_mut_ptr());
            if rs != hip::hiprtcResult_HIPRTC_SUCCESS {
                let mut sz = 0usize;
                let _ = hip::hiprtcGetProgramLogSize(prog, &mut sz);
                let mut buf = vec![0i8; sz.max(1)];
                let _ = hip::hiprtcGetProgramLog(prog, buf.as_mut_ptr());
                let log = String::from_utf8_lossy(std::slice::from_raw_parts(buf.as_ptr() as *const u8, sz));
                return Err(format!("rawhip 컴파일 실패: {log}"));
            }
            let mut code_sz = 0usize;
            if hip::hiprtcGetCodeSize(prog, &mut code_sz) != hip::hiprtcResult_HIPRTC_SUCCESS {
                return Err("GetCodeSize".into());
            }
            let mut code = vec![0i8; code_sz];
            if hip::hiprtcGetCode(prog, code.as_mut_ptr()) != hip::hiprtcResult_HIPRTC_SUCCESS {
                return Err("GetCode".into());
            }
            if let Some(path) = std::env::var_os("LLM170_DUMP_MODULE") {
                let _ = std::fs::write(&path, std::slice::from_raw_parts(code.as_ptr() as *const u8, code_sz));
            }
            let mut module: hip::hipModule_t = std::ptr::null_mut();
            ck(hip::hipModuleLoadData(&mut module, code.as_ptr() as *const _), "ModuleLoadData")?;
            let mut fns = HashMap::new();
            for name in kernels::NAMES {
                let cname = CString::new(*name).unwrap();
                let mut f: hip::hipFunction_t = std::ptr::null_mut();
                ck(hip::hipModuleGetFunction(&mut f, module, cname.as_ptr()), "GetFunction")?;
                fns.insert(*name, f);
            }
            // 오프라인 코드오브젝트 병행 로드 (wave32 커널 등).
            // 기본: 바이너리 임베딩(crates/.../co/*.co, gfx1151 빌드).
            // LLM170_CO*_PATH가 있으면 그 파일이 우선 (커널 실험 오버라이드).
            // LLM170_NO_CO: 전부 생략 (hipRTC wm/mm + GEMV 폴백 측정용).
            if std::env::var_os("LLM170_NO_CO").is_none() {
                let slots: &[(u8, &str, &[u8], &[&str])] = &[
                    (
                        CO_V4,
                        "LLM170_CO2_PATH",
                        include_bytes!("co/v4all.co"),
                        &["gemm_q5k_v4", "gemm_q4k_v4", "gemm_xs_v4",
                          "gemm_q5k_wm", "gemm_q4k_wm", "gemm_q6k_wm", "gemm_xs_wm"],
                    ),
                    (
                        CO_ODD,
                        "LLM170_CO3_PATH",
                        include_bytes!("co/odd_all.co"),
                        &["gemm_nl_v4", "gemm_q3k_v4", "gemm_iq3s_v4"],
                    ),
                    (
                        CO_MMQ,
                        "LLM170_CO4_PATH",
                        include_bytes!("co/mmq.co"),
                        &["mmq_quant_y",
                          "mmq_quant_y_d4",
                          "_ZL9mul_mat_qIL9ggml_type12ELi128ELb0EEvPKcPKiS4_S4_PfS5_PKf15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_",
                          "_ZL9mul_mat_qIL9ggml_type13ELi128ELb0EEvPKcPKiS4_S4_PfS5_PKf15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_",
                          "_ZL9mul_mat_qIL9ggml_type14ELi128ELb0EEvPKcPKiS4_S4_PfS5_PKf15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_",
                          "_ZL9mul_mat_qIL9ggml_type23ELi128ELb0EEvPKcPKiS4_S4_PfS5_PKf15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_"],
                    ),
                    (
                        CO_MMQ2,
                        "LLM170_CO5_PATH",
                        include_bytes!("co/mmq2.co"),
                        &["gemm_f16_v4"],
                    ),
                    (
                        CO_MMQ3,
                        "LLM170_CO6_PATH",
                        include_bytes!("co/mmq3.co"),
                        &["_ZL9mul_mat_qIL9ggml_type23ELi128ELb0EEvPKcPKiS4_S4_PfS5_PKf15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_"],
                    ),
                    (
                        CO_MMQ8,
                        "LLM170_CO7_PATH",
                        include_bytes!("co/mmq8.co"),
                        &["_ZL9mul_mat_qIL9ggml_type8ELi128ELb0EEvPKcPKiS4_S4_PfS5_PKf15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_"],
                    ),
                    (
                        CO_QY,
                        "LLM170_CO8_PATH",
                        include_bytes!("co/quanty_new.co"),
                        &["_ZL17quantize_mmq_q8_1IL18mmq_q8_1_ds_layout0ELb0EEvPKfPKiPvllllliii",
                          "_ZL17quantize_mmq_q8_1IL18mmq_q8_1_ds_layout1ELb0EEvPKfPKiPvllllliii",
                          "_ZL17quantize_mmq_q8_1IL18mmq_q8_1_ds_layout2ELb0EEvPKfPKiPvllllliii"],
                    ),
                    (
                        CO_J128,
                        "LLM170_CO_PATH",
                        include_bytes!("co/w32b.co"),
                        &["gemm_q5k_j128", "gemm_q4k_j128", "gemm_q6k_j128",
                          "gemm_xs_j128", "gemm_q8_j128"],
                    ),
                ];
                for (bit, env_key, embedded, names) in slots {
                    let bytes: Vec<u8> = match std::env::var_os(env_key) {
                        Some(p) => std::fs::read(&p)
                            .map_err(|e| format!("{env_key} 읽기({p:?}): {e}"))?,
                        None => embedded.to_vec(),
                    };
                    let mut m: hip::hipModule_t = std::ptr::null_mut();
                    ck(hip::hipModuleLoadData(&mut m, bytes.as_ptr() as *const _),
                       &format!("{env_key} ModuleLoadData"))?;
                    let mut loaded = 0u8;
                    for name in *names {
                        let cname = CString::new(*name).unwrap();
                        let mut f: hip::hipFunction_t = std::ptr::null_mut();
                        if hip::hipModuleGetFunction(&mut f, m, cname.as_ptr())
                            == hip::hipError_t_hipSuccess
                        {
                            fns.insert(name, f);
                            loaded |= bit;
                        }
                    }
                    CO_FAM.fetch_or(loaded, std::sync::atomic::Ordering::Relaxed);
                }
            }

            let mut stream: hip::hipStream_t = std::ptr::null_mut();
            ck(hip::hipStreamCreate(&mut stream), "StreamCreate")?;
            let mut stream2: hip::hipStream_t = std::ptr::null_mut();
            ck(hip::hipStreamCreate(&mut stream2), "StreamCreate2")?;
            Ok(RawCtx { module, fns, stream, stream2, mmq_y: std::sync::Mutex::new((0, std::ptr::null_mut())),
            mmq_y_s: std::sync::Mutex::new((0, std::ptr::null_mut())),
            f16_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            allocs: std::sync::Mutex::new(Vec::new()),
            ar_cache: std::sync::Mutex::new(None),
            mmq_y_cache: std::sync::Mutex::new((u64::MAX, 0, 0)),
            canon_q6: std::sync::Mutex::new(std::collections::HashMap::new()),
            mmq_y2: std::sync::Mutex::new((0, std::ptr::null_mut())), scratch: std::sync::Mutex::new(HashMap::new()), cursors: std::sync::Mutex::new(HashMap::new()), pinned: std::sync::Mutex::new((0, std::ptr::null_mut())), pinned_a: std::sync::Mutex::new((0, std::ptr::null_mut())) })
        }
    }

    /// 스크래시 획득 — 같은 크기는 항상 슬롯 0 재사용 (스트림 순서가
    /// 이전 사용 완료를 보장 — 단일 스트림). 호출마다 신규 할당은
    /// 메모리 고갈→illegal address (2026-09-03 RCA).
    pub fn scratch(&self, bytes: usize) -> Result<*mut u8, String> {
        let mut sc = self.scratch.lock().map_err(|e| e.to_string())?;
        let v = sc.entry(bytes).or_default();
        if v.is_empty() {
            let p = self.alloc(bytes)?;
            v.push(p);
        }
        Ok(v[0])
    }


    /// 영속 디바이스 할당 (해제 없음).
    pub fn alloc(&self, bytes: usize) -> Result<*mut u8, String> {
        let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            let r = hip::hipMalloc(&mut p, bytes);
            if r != hip::hipError_t_hipSuccess { eprintln!("alloc {bytes}B → {r:?}"); }
            ck(r, "hipMalloc")?;
        }
        if let Ok(mut v) = self.allocs.lock() {
            v.push((p as usize, p as usize + bytes));
        }
        Ok(p as *mut u8)
    }

    /// `p`가 살아있는 할당 안인지(그리고 몇 바이트 남았는지) — h2d 실패 진단용.
    fn alloc_span(&self, p: usize, need: usize) -> String {
        let Ok(v) = self.allocs.lock() else { return "등록부 잠금 실패".into() };
        for &(s, e) in v.iter() {
            if p >= s && p < e {
                return if p + need <= e {
                    format!("할당 내 [{s:#x},{e:#x})")
                } else {
                    format!("할당 경계 초과! [{s:#x},{e:#x}) +{}B", p + need - e)
                };
            }
        }
        format!("할당 밖 (등록 {}개)", v.len())
    }

    /// 사이드 스트림 비동기 h2d — 메인 스트림 작업과 중첩시킨 뒤 join2로 합류.
    pub fn h2d_async_s(&self, dst: *mut u8, src: &[u8]) -> Result<(), String> {
        unsafe {
            ck(hip::hipMemcpyAsync(dst as *mut _, src.as_ptr() as *const _, src.len(),
                hip::hipMemcpyKind_hipMemcpyHostToDevice, self.stream2), "h2d-async-s")
        }
    }

    /// 메인 스트림 비동기 h2d — 발사 순서가 커널과 같은 큐를 따라야 하는
    /// 소형 업로드(플레 임베딩 등)용. 동기화 없음(plans/73).
    pub fn h2d_async_m(&self, dst: *mut u8, src: &[u8]) -> Result<(), String> {
        unsafe {
            ck(hip::hipMemcpyAsync(dst as *mut _, src.as_ptr() as *const _, src.len(),
                hip::hipMemcpyKind_hipMemcpyHostToDevice, self.stream), "h2d-async-m")
        }
    }
    /// 디바이스→디바이드 복사(메인 스트림) — QSA KV 상주 풀 append용(plans/67 3단계).
    pub fn d2d(&self, dst: *mut u8, src: *const u8, bytes: usize) -> Result<(), String> {
        unsafe {
            ck(hip::hipMemcpyAsync(dst as *mut _, src as *const _, bytes,
                hip::hipMemcpyKind_hipMemcpyDeviceToDevice, self.stream), "d2d")
        }
    }
    pub fn h2d(&self, dst: *mut u8, src: &[u8]) -> Result<(), String> {
        unsafe {
            // 실패 시 크기·목적지·**호출 지점**을 남긴다. 상한 가정이 여러 곳에
            // 흩어진 경로에서 "h2d: 700"만으로는 어느 복사인지 알 수 없었고,
            // 크기만으로도 부족했다(2026-09-14). 백트레이스는 강제로 잡는다
            // (RUST_BACKTRACE 미설정이어도 동작).
            let tag = format!("h2d {}B dst={dst:p}", src.len());
            if std::env::var_os("LLM170_H2D_TRACE").is_some() {
                eprintln!("# h2d {}B dst={dst:p} q0={:?}", src.len(), &src[..src.len().min(2)]);
            }
            // LLM170_MEMDBG: 복사 **직전** 여유 메모리(사후 조회는 sticky 오류로 0/0).
            if std::env::var_os("LLM170_MEMDBG").is_some() && src.len() >= (1 << 20) {
                let (mut fb, mut tb) = (0usize, 0usize);
                let _ = hip::hipMemGetInfo(&mut fb, &mut tb);
                eprintln!("# memdbg before {tag}: free={}MB/{}MB", fb / 1048576, tb / 1048576);
            }
            if let Err(e) = ck(hip::hipMemcpyAsync(dst as *mut _, src.as_ptr() as *const _, src.len(), hip::hipMemcpyKind_hipMemcpyHostToDevice, self.stream), &tag) {
                // 사후 hipMemGetInfo는 sticky 오류 때문에 0을 돌려준다(확인함) —
                // 메모리 진단이 필요하면 이 h2d **전에** 조회해야 한다.
                let span = self.alloc_span(dst as usize, src.len());
                let bt = std::backtrace::Backtrace::force_capture();
                let frames: Vec<String> = format!("{bt}")
                    .lines()
                    .filter(|l| l.contains("llm170") && !l.contains("backtrace"))
                    .take(4)
                    .map(|l| l.trim().to_string())
                    .collect();
                return Err(format!(
                    "{e} | dst {span} | 호출: {}",
                    frames.join(" <- ")
                ));
            }
            self.sync()
        }
    }

    /// 비동기 d2h용 이벤트(스텝 간 재사용) — 한 스트림에 순서대로 걸린다.
    fn d2h_ev(&self) -> Result<hip::hipEvent_t, String> {
        static EV: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let p = *EV.get_or_init(|| {
            let mut e: hip::hipEvent_t = std::ptr::null_mut();
            unsafe {
                let _ = hip::hipEventCreateWithFlags(&mut e, 0);
            }
            e as usize
        });
        Ok(p as hip::hipEvent_t)
    }

    /// 비동기 d2h 예약 — 핀 버퍼로 스트림 복사만 걸고 sync하지 않는다.
    /// 그 사이 커널을 계속 발사해 전송·대기 지연을 다른 연산 뒤에 숨기고,
    /// 필요해지는 지점에서 `d2h_wait()`로 완료를 기다린다. 반환 = 핀 버퍼.
    pub fn d2h_issue(&self, need: usize, src: *const u8) -> Result<*mut u8, String> {
        unsafe {
            let mut pin = self.pinned_a.lock().map_err(|e| e.to_string())?;
            if pin.0 < need {
                let mut p: *mut std::os::raw::c_void = std::ptr::null_mut();
                if hip::hipMallocHost(&mut p, need) == hip::hipError_t_hipSuccess {
                    *pin = (need, p as *mut u8);
                } else {
                    *pin = (0, std::ptr::null_mut());
                }
            }
            let buf = pin.1;
            if buf.is_null() {
                return Ok(std::ptr::null_mut());
            }
            ck(
                hip::hipMemcpyAsync(
                    buf as *mut _,
                    src as *const _,
                    need,
                    hip::hipMemcpyKind_hipMemcpyDeviceToHost,
                    self.stream,
                ),
                "d2h-issue",
            )?;
            let ev = self.d2h_ev()?;
            ck(hip::hipEventRecord(ev, self.stream), "d2h-ev")?;
            Ok(buf)
        }
    }

    /// `d2h_issue` 완료 대기 — **복사 이벤트만** 기다린다(스트림 전체를 비우지
    /// 않으므로 그 뒤에 큐잉된 커널은 계속 진행된다). 파이프라인을 살려 두는
    /// 것이 이 API의 존재 이유다(2026-09-14: 전체 sync는 층마다 파이프라인을
    /// 비워 shared 스테이지가 25.8 → 50.7ms로 두 배가 됐다).
    pub fn d2h_wait(&self) -> Result<(), String> {
        unsafe {
            let ev = self.d2h_ev()?;
            ck(hip::hipEventSynchronize(ev), "d2h-ev-wait")
        }
    }

    pub fn d2h(&self, dst: &mut [u8], src: *const u8) -> Result<(), String> {
        unsafe {
            // pageable 직행은 슬로패스 — 핀 스테이징 경유 (2026-09-05 tg RCA:
            // logits 1MB D2H가 92ms → 핀 경유 시 <1ms 예상)
            let need = dst.len();
            // 핀 스테이징 실패(호스트 메모리 압박 등)는 조용히 페이지어블 경로로
            // 폴백한다 — 폴트 여부를 가리는 대신 판독은 성공시킨다(2026-09-12:
            // 장문 청크에서 hipMallocHost: 700으로 판독이 통째로 실패).
            let mut pin = self.pinned.lock().map_err(|e| e.to_string())?;
            if pin.0 < need {
                let mut p: *mut std::os::raw::c_void = std::ptr::null_mut();
                if hip::hipMallocHost(&mut p, need) == hip::hipError_t_hipSuccess {
                    *pin = (need, p as *mut u8);
                } else {
                    *pin = (0, std::ptr::null_mut());
                }
            }
            if !pin.1.is_null() {
                let buf = pin.1;
                ck(hip::hipMemcpyAsync(buf as *mut _, src as *const _, need, hip::hipMemcpyKind_hipMemcpyDeviceToHost, self.stream), "d2h-pin")?;
                ck(hip::hipStreamSynchronize(self.stream), "d2h-sync")?;
                std::ptr::copy_nonoverlapping(buf, dst.as_mut_ptr(), need);
                return Ok(());
            }
            ck(
                hip::hipMemcpy(
                    dst.as_mut_ptr() as *mut std::os::raw::c_void,
                    src as *const std::os::raw::c_void,
                    need,
                    hip::hipMemcpyKind_hipMemcpyDeviceToHost,
                ),
                "d2h-pageable",
            )
        }
    }
    pub fn sync(&self) -> Result<(), String> {
        unsafe { ck(hip::hipStreamSynchronize(self.stream), "sync") }
    }

    /// KTRACE 전용 이벤트 마커 — launch3를 거치지 않는 직접 런치 경로용.
    fn ktr_mark(&self, name: &'static str, gy: u32) {
        if let Ok(mut g) = KTRACE.lock() {
            if g.is_some() {
                let mut ev: hip::hipEvent_t = std::ptr::null_mut();
                unsafe {
                    hip::hipEventCreateWithFlags(&mut ev, 0);
                    hip::hipEventRecord(ev, self.stream);
                }
                g.as_mut().unwrap().push(KtraceEv(name, ev as usize, gy));
            }
        }
    }

    /// 커널 런치 — args는 각 인자 값에 대한 포인터 배열 (호출자 슬롯 유지).
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        name: &str,
        gx: u32,
        gy: u32,
        block: u32,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<(), String> {
        // 진단(LLM170_NOLAUNCH): 런치를 건너뛰고 호스트 스켈레톤 시간만 측정한다.
        // 그래프 재생 중에도 즉시 반환한다(커널은 그래프가 실행).
        if GRAPH_SKIP.load(std::sync::atomic::Ordering::Relaxed) || nolaunch_on() {
            return Ok(());
        }
        let f = *self.fns.get(name).ok_or_else(|| format!("커널 없음: {name}"))?;
        unsafe {
            if let Ok(mut g) = KTRACE.lock() {
                if g.is_some() {
                    let mut ev0: hip::hipEvent_t = std::ptr::null_mut();
                    hip::hipEventCreateWithFlags(&mut ev0, 0);
                    hip::hipEventRecord(ev0, self.stream);
                    g.as_mut().unwrap().push(KtraceEv(name_leak(name), ev0 as usize, gy));
                }
            }
            ck(hip::hipModuleLaunchKernel(f, gx, gy, 1, block, 1, 1, 0, self.stream, args.as_mut_ptr(), std::ptr::null_mut()), "launch").map_err(|e| format!("{e} kern={name} gx={gx} blk={block}"))?;
            if let Ok(mut g) = KTRACE.lock() {
                if g.is_some() {
                    let mut ev: hip::hipEvent_t = std::ptr::null_mut();
                    hip::hipEventCreateWithFlags(&mut ev, 0);
                    hip::hipEventRecord(ev, self.stream);
                    g.as_mut().unwrap().push(KtraceEv(name_leak(name), ev as usize, gy));
                }
            }
        }
        Ok(())
    }


    /// 3차원 그리드 런치 (qsa용 — gy 추가).
    #[allow(clippy::too_many_arguments)]
    pub fn launch3(
        &self,
        name: &str,
        gx: u32,
        gy: u32,
        gz: u32,
        block: u32,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<(), String> {        if std::env::var_os("LLM170_LAUNCH_BT").is_some() {
            eprintln!("[lbt] {name} gx={gx} gy={gy} gz={gz}");
        }


        if GRAPH_SKIP.load(std::sync::atomic::Ordering::Relaxed) || nolaunch_on() {
            return Ok(());
        }
        if std::env::var_os("LLM170_KT_NAMES").is_some() {
            eprintln!("# KT3 {name} gy={gy}");
        }
        let f = *self.fns.get(name).ok_or_else(|| format!("커널 없음: {name}"))?;
        unsafe {
            if let Ok(mut g) = KTRACE.lock() {
                if g.is_some() {
                    let mut ev0: hip::hipEvent_t = std::ptr::null_mut();
                    hip::hipEventCreateWithFlags(&mut ev0, 0);
                    hip::hipEventRecord(ev0, self.stream);
                    g.as_mut().unwrap().push(KtraceEv(name_leak(name), ev0 as usize, gy));
                }
            }
            ck(hip::hipModuleLaunchKernel(f, gx, gy, gz, block, 1, 1, 0, self.stream, args.as_mut_ptr(), std::ptr::null_mut()), "launch3").map_err(|e| format!("{e} kern={name} gx={gx} gy={gy} gz={gz} blk={block}"))?;
            if let Ok(mut g) = KTRACE.lock() {
                if g.is_some() {
                    let mut ev: hip::hipEvent_t = std::ptr::null_mut();
                    hip::hipEventCreateWithFlags(&mut ev, 0);
                    hip::hipEventRecord(ev, self.stream);
                    g.as_mut().unwrap().push(KtraceEv(name_leak(name), ev as usize, gy));
                }
            }
        }
        Ok(())
    }

    /// 사이드 스트림 발사 (비동기 — join2로 합류)
    /// 동적 shared 64KB 런치 (부록82) — 커널당 1회 속성 설정.
    pub fn launch3_dyn(&self, name: &str, gx: u32, gy: u32, gz: u32, block: u32, smem: u32, args: &mut [*mut std::ffi::c_void]) -> Result<(), String> {
        if GRAPH_SKIP.load(std::sync::atomic::Ordering::Relaxed) || nolaunch_on() {
            return Ok(());
        }
        use std::collections::HashSet;
        use std::sync::OnceLock;
        static SET: OnceLock<std::sync::Mutex<HashSet<usize>>> = OnceLock::new();
        let f = *self.fns.get(name).ok_or_else(|| format!("커널 없음: {name}"))?;
        let set = SET.get_or_init(|| std::sync::Mutex::new(HashSet::new()));
        {
            let mut g = set.lock().map_err(|e| e.to_string())?;
            if g.insert(f as usize) {
                unsafe {
                    let r = hip::hipFuncSetAttribute(f as *const std::ffi::c_void, hip::hipFuncAttribute_hipFuncAttributeMaxDynamicSharedMemorySize, smem as i32);
                    if r != hip::hipError_t_hipSuccess {
                        return Err(format!("smem 속성 실패: {r:?}"));
                    }
                }
            }
        }
        unsafe {
            // KTRACE 이벤트 짝 — launch3와 동일(2026-09-14 plans/69: 이 기록이
            // 없어 qsa_flash_wmma의 실행 시간이 'kv_f16 뒤 갭 12s'로 위장,
            // 8.4× 프리필 격차의 원인 파악을 하루 종일 돌렸다).
            if let Ok(mut g) = KTRACE.lock() {
                if g.is_some() {
                    let mut ev0: hip::hipEvent_t = std::ptr::null_mut();
                    hip::hipEventCreateWithFlags(&mut ev0, 0);
                    hip::hipEventRecord(ev0, self.stream);
                    g.as_mut().unwrap().push(KtraceEv(name_leak(name), ev0 as usize, gy));
                }
            }
            ck(hip::hipModuleLaunchKernel(f, gx, gy, gz, block, 1, 1, smem, self.stream, args.as_mut_ptr(), std::ptr::null_mut()), "launch3_dyn")?;
            if let Ok(mut g) = KTRACE.lock() {
                if g.is_some() {
                    let mut ev: hip::hipEvent_t = std::ptr::null_mut();
                    hip::hipEventCreateWithFlags(&mut ev, 0);
                    hip::hipEventRecord(ev, self.stream);
                    g.as_mut().unwrap().push(KtraceEv(name_leak(name), ev as usize, gy));
                }
            }
        }
        Ok(())
    }

    pub fn launch3s(
        &self,
        name: &str,
        gx: u32,
        gy: u32,
        gz: u32,
        block: u32,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<(), String> {
        let f = *self.fns.get(name).ok_or_else(|| format!("커널 없음: {name}"))?;
        unsafe {
            // KTRACE 훅 — 프레임 경로의 dense GEMM이 전부 이 경로(stream2)를 쓴다.
            // 훅이 없어 프레임 트레이스에서 통째로 누락되던 버그(2026-09-14).
            if let Ok(mut g) = KTRACE.lock() {
                if g.is_some() {
                    let mut ev0: hip::hipEvent_t = std::ptr::null_mut();
                    hip::hipEventCreateWithFlags(&mut ev0, 0);
                    hip::hipEventRecord(ev0, self.stream2);
                    g.as_mut().unwrap().push(KtraceEv(name_leak(name), ev0 as usize, gy));
                }
            }
            ck(hip::hipModuleLaunchKernel(f, gx, gy, gz, block, 1, 1, 0, self.stream2, args.as_mut_ptr(), std::ptr::null_mut()), "launch3s")?;
            if let Ok(mut g) = KTRACE.lock() {
                if g.is_some() {
                    let mut ev: hip::hipEvent_t = std::ptr::null_mut();
                    hip::hipEventCreateWithFlags(&mut ev, 0);
                    hip::hipEventRecord(ev, self.stream2);
                    g.as_mut().unwrap().push(KtraceEv(name_leak(name), ev as usize, gy));
                }
            }
        }
        Ok(())
    }
    /// 사이드 스트림 → 주 스트림 합류: 이벤트 경유
    pub fn join2(&self) -> Result<(), String> {
        unsafe {
            let mut ev: hip::hipEvent_t = std::ptr::null_mut();
            ck(hip::hipEventCreateWithFlags(&mut ev, 0), "evCreate")?;
            ck(hip::hipEventRecord(ev, self.stream2), "evRecord")?;
            ck(hip::hipStreamWaitEvent(self.stream, ev, 0), "evWait")?;
            ck(hip::hipEventDestroy(ev), "evDestroy")?;
        }
        Ok(())
    }
    /// 주 스트림 현재 시점 → 사이드 대기 (사이드 입력 준비 경합 방지)
    pub fn side_wait_main(&self) -> Result<(), String> {
        unsafe {
            let mut ev: hip::hipEvent_t = std::ptr::null_mut();
            ck(hip::hipEventCreateWithFlags(&mut ev, 0), "evCreate2")?;
            ck(hip::hipEventRecord(ev, self.stream), "evRecord2")?;
            ck(hip::hipStreamWaitEvent(self.stream2, ev, 0), "evWait2")?;
            ck(hip::hipEventDestroy(ev), "evDestroy2")?;
        }
        Ok(())
    }

    /// W4A8 t=1 GEMV — 타입별 커널 선택, 부분합 reduce까지 수행.
    /// 반환 [n_out] f32. 수치: dot_row_w4a8_*_lane 미러와 동일열.
    pub fn gemv_q8(
        &self,
        xq: *const u8,
        w: *const u8,
        ktab2: *const u8,
        ty: u32,
        n_in: usize,
        n_out: usize,
    ) -> Result<Vec<f32>, String> {
        let part = self.scratch(n_out * 64 * 8)?;
        let out = self.scratch(n_out * 4)?;
        let gy = n_out.min(65535) as u32;
        let gz = n_out.div_ceil(65535) as u32;
        let kern = match ty {
            23 => "gemm_xs",   // iq4_xs
            13 => "gemm_q5k",  // q5_K
            8 => "gemm_q8_0",  // q8_0
            12 => "gemm_q4k",  // q4_K
            14 => "gemm_q6k",  // q6_K
            20 => "gemm_nl",   // iq4_nl
            11 => "gemm_q3k",  // q3_K
            21 => "gemm_iq3s", // iq3_s
            _ => return Err(format!("미지원 타입 {ty}")),
        };
        let mut xq_p = xq as *mut std::ffi::c_void;
        let mut w_p = w as *mut std::ffi::c_void;
        let mut part_p = part as *mut std::ffi::c_void;
        let mut kt_p = ktab2 as *mut std::ffi::c_void;
        let mut n_in_a = n_in as i32;
        let mut n_out_a = n_out as i32;
        let mut gx_a = 1i32;
        let mut args_v: Vec<*mut std::ffi::c_void> = match ty {
            23 | 20 => vec![
                &mut xq_p as *mut _ as *mut std::ffi::c_void,
                &mut w_p as *mut _ as *mut std::ffi::c_void,
                &mut part_p as *mut _ as *mut std::ffi::c_void,
                &mut kt_p as *mut _ as *mut std::ffi::c_void,
                &mut n_in_a as *mut _ as *mut std::ffi::c_void,
                &mut n_out_a as *mut _ as *mut std::ffi::c_void,
            ],
            _ => vec![
                &mut xq_p as *mut _ as *mut std::ffi::c_void,
                &mut w_p as *mut _ as *mut std::ffi::c_void,
                &mut part_p as *mut _ as *mut std::ffi::c_void,
                &mut n_in_a as *mut _ as *mut std::ffi::c_void,
                &mut n_out_a as *mut _ as *mut std::ffi::c_void,
            ],
        };
        let _ = &mut gx_a;
        let mut out_p0 = out as *mut std::ffi::c_void;
        match ty {
            23 | 20 => args_v.insert(4, &mut out_p0 as *mut _ as *mut std::ffi::c_void),
            _ => args_v.insert(3, &mut out_p0 as *mut _ as *mut std::ffi::c_void),
        }
        let mut xw_a = (n_in / 4 + n_in / 32 + n_in / 16) as i32;
        args_v.push(&mut xw_a as *mut _ as *mut std::ffi::c_void);
        self.launch3(kern, 1, gy, gz, 64, &mut args_v)?;
        let mut res = vec![0f32; n_out];
        self.sync()?;
        self.d2h(bytemuck::cast_slice_mut(&mut res).as_mut(), out as *const u8)?;
        Ok(res)
    }

    /// W4A8 GEMV — reduce 결과를 상주 out에 직접 기록 (왕복 제거).
    /// 수치는 gemv_q8과 동일열 (동일 커널·reduce).
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q8_out_v2(&self, xq: *const u8, w: *const u8, _ty: u32, n_in: usize, n_out: usize, out: *mut u8, xq_w: usize, t: usize) -> Result<(), String> {
        let gy = n_out.min(65535) as u32;
        let gz = n_out.div_ceil(65535) as u32;
        let mut xq_p = xq as *mut std::ffi::c_void;
        let mut w_p = w as *mut std::ffi::c_void;
        let mut o_p = out as *mut std::ffi::c_void;
        let mut ni = n_in as i32;
        let mut no = n_out as i32;
        let mut xw = xq_w as i32;
        let mut tt = t as i32;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&mut xq_p) as *mut _ as *mut std::ffi::c_void,
            (&mut w_p) as *mut _ as *mut std::ffi::c_void,
            (&mut o_p) as *mut _ as *mut std::ffi::c_void,
            (&mut ni) as *mut _ as *mut std::ffi::c_void,
            (&mut no) as *mut _ as *mut std::ffi::c_void,
            (&mut xw) as *mut _ as *mut std::ffi::c_void,
            (&mut tt) as *mut _ as *mut std::ffi::c_void,
        ];
        let f = *self.fns.get("gemm_q5k_v2").ok_or("gemm_q5k_v2 없음")?;
        unsafe {
            ck(hip::hipModuleLaunchKernel(f, t as u32, gy, gz, 64, 1, 1, 0, self.stream, args.as_mut_ptr(), std::ptr::null_mut()), "gemm_q5k_v2")?;
        }
        Ok(())
    }

    pub fn gemv_q8_out(
        &self,
        xq: *const u8,
        w: *const u8,
        ktab2: *const u8,
        ty: u32,
        n_in: usize,
        n_out: usize,
        out: *mut u8,
        xq_w: usize,
        t: usize,
    ) -> Result<(), String> {
        let part = self.scratch(n_out * 64 * 8)?;
        let gy = n_out.min(65535) as u32;
        let _gz = n_out.div_ceil(65535) as u32;
        let kern = match ty {
            23 => "gemm_xs",
            13 => "gemm_q5k",
            8 => "gemm_q8_0",
            12 => "gemm_q4k",
            14 => "gemm_q6k",
            20 => "gemm_nl",
            11 => "gemm_q3k",
            21 => "gemm_iq3s",
            _ => return Err(format!("미지원 타입 {ty}")),
        };
        let mut xq_p = xq as *mut std::ffi::c_void;
        let mut w_p = w as *mut std::ffi::c_void;
        let mut part_p = part as *mut std::ffi::c_void;
        let mut kt_p = ktab2 as *mut std::ffi::c_void;
        let mut n_in_a = n_in as i32;
        let mut n_out_a = n_out as i32;
        let mut args_v: Vec<*mut std::ffi::c_void> = match ty {
            23 | 20 => vec![
                &mut xq_p as *mut _ as *mut std::ffi::c_void,
                &mut w_p as *mut _ as *mut std::ffi::c_void,
                &mut part_p as *mut _ as *mut std::ffi::c_void,
                &mut kt_p as *mut _ as *mut std::ffi::c_void,
                &mut n_in_a as *mut _ as *mut std::ffi::c_void,
                &mut n_out_a as *mut _ as *mut std::ffi::c_void,
            ],
            _ => vec![
                &mut xq_p as *mut _ as *mut std::ffi::c_void,
                &mut w_p as *mut _ as *mut std::ffi::c_void,
                &mut part_p as *mut _ as *mut std::ffi::c_void,
                &mut n_in_a as *mut _ as *mut std::ffi::c_void,
                &mut n_out_a as *mut _ as *mut std::ffi::c_void,
            ],
        };
        // plans/73: vdr=2 판(v2)은 측정 역행(10.83 vs 11.35 t/s) — 옵트인 자산.
        if t == 1 && ty == 13 && std::env::var("LLM170_Q5KV2").as_deref() == Ok("1") {
            return self.gemv_q8_out_v2(xq, w, ty, n_in, n_out, out, xq_w, t);
        }
        let mut out_p0 = out as *mut std::ffi::c_void;
        let mut xw_a = xq_w as i32;
        let mut tt_a = t as i32;
        // plans/73 (2026-09-16): t=1 q8_0 전 형상을 **16레인×4사분면** 판으로 —
        // 종전 64레인/행은 n_sub=80(qkv/gate)에서 62.5%, n_sub=10(hc up)에서 31%
        // 레인 효율이었고 그만큼 대역폭이 깎였다(GDN mm_group 15.0ms = 106GB/s).
        // 산술은 비트 동일(gemm_q8_0_w4 주석의 트리 재구성). 킬스위치 LLM170_Q8W4=0.
        // 멀티토큰 판(2026-09-16, np 배치): t=2..8 q8_0은 무게 행 1회 독서로
        // 토큰별 내적 — grid=(t,n_out) 배치가 토큰마다 무게를 재독하는 것과
        // 달리 가중치 트래픽이 t배 증가하지 않는다. 산술 비트 동일.
        // plans/74 N2: 소형 n_sub(≤32 — hc up/down=10)는 16레인/행 mt16 판 —
        // mt(64레인)은 이 형상에서 레인 효율 15%(np t=4 182us/호출 실측).
        // 킬스위치 LLM170_Q8MT16=0.
        if t >= 2 && t <= 8 && ty == 8 && n_in / 32 <= 32
            && std::env::var("LLM170_Q8MT16").as_deref() != Ok("0")
        {
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &mut xq_p as *mut _ as *mut std::ffi::c_void,
                &mut w_p as *mut _ as *mut std::ffi::c_void,
                &mut part_p as *mut _ as *mut std::ffi::c_void,
                &mut out_p0 as *mut _ as *mut std::ffi::c_void,
                &mut n_in_a as *mut _ as *mut std::ffi::c_void,
                &mut n_out_a as *mut _ as *mut std::ffi::c_void,
                &mut xw_a as *mut _ as *mut std::ffi::c_void,
                &mut tt_a as *mut _ as *mut std::ffi::c_void,
            ];
            return self.launch3(
                "gemm_q8_0_mt16",
                n_out.div_ceil(8) as u32,
                1,
                1,
                128,
                &mut args,
            );
        }
        if t >= 2 && t <= 8 && ty == 8 && std::env::var("LLM170_Q8MT").as_deref() != Ok("0") {
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &mut xq_p as *mut _ as *mut std::ffi::c_void,
                &mut w_p as *mut _ as *mut std::ffi::c_void,
                &mut part_p as *mut _ as *mut std::ffi::c_void,
                &mut out_p0 as *mut _ as *mut std::ffi::c_void,
                &mut n_in_a as *mut _ as *mut std::ffi::c_void,
                &mut n_out_a as *mut _ as *mut std::ffi::c_void,
                &mut xw_a as *mut _ as *mut std::ffi::c_void,
                &mut tt_a as *mut _ as *mut std::ffi::c_void,
            ];
            return self.launch3(
                "gemm_q8_0_mt",
                1,
                n_out.min(65535) as u32,
                n_out.div_ceil(65535) as u32,
                64,
                &mut args,
            );
        }
        // 실측(2026-09-16): w4(사분면, 비트 동일) -9%, w16(연속 매핑) -7% —
        // 둘 다 레인 효율은 100%지만 종전 64레인 판(coalescing·ILP)이 더 빠르다.
        // 따라서 **기본은 종전 커널**, 실험판은 옵트인(LLM170_Q8W4=1 / Q8W16=1).
        if t == 1
            && ty == 8
            && (std::env::var("LLM170_Q8W4").as_deref() == Ok("1")
                || std::env::var("LLM170_Q8W16").as_deref() == Ok("1"))
        {
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &mut xq_p as *mut _ as *mut std::ffi::c_void,
                &mut w_p as *mut _ as *mut std::ffi::c_void,
                &mut part_p as *mut _ as *mut std::ffi::c_void,
                &mut out_p0 as *mut _ as *mut std::ffi::c_void,
                &mut n_in_a as *mut _ as *mut std::ffi::c_void,
                &mut n_out_a as *mut _ as *mut std::ffi::c_void,
                &mut xw_a as *mut _ as *mut std::ffi::c_void,
            ];
            let alt = std::env::var("LLM170_Q8W16").as_deref() == Ok("1");
            return self.launch3(
                if alt { "gemm_q8_0_w16" } else { "gemm_q8_0_w4" },
                n_out.div_ceil(8) as u32,
                t as u32,
                1,
                128,
                &mut args,
            );
        }
        // 실험(2026-09-16): 워프판(32레인/행)을 n_sub>32 형상까지 확대 —
        // n_sub=80에서 레인 효율 83% vs 종전 62.5%. LLM170_Q8W_ALL=1로 옵트인.
        let w_all = t == 1 && ty == 8 && std::env::var("LLM170_Q8W_ALL").as_deref() == Ok("1");
        // 소형 n_sub(≤32) 구간은 w16(16레인/행, 레인 효율 62.5-100% vs 워프판
        // 31%)으로 — FN tg128 17.2 → 18.0 t/s (+4.8%, 2026-09-16 실측, 게이트 동일).
        // 킬스위치 LLM170_Q8W16_SMALL=0.
        if t == 1
            && ty == 8
            && (n_in / 32 <= 32
                || (n_out >= 32768 && std::env::var("LLM170_Q8W16_HEAD").as_deref() == Ok("1")))
            && (std::env::var("LLM170_Q8W16_SMALL").as_deref() != Ok("0")
                || std::env::var("LLM170_Q8W16_HEAD").as_deref() == Ok("1"))
        {
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &mut xq_p as *mut _ as *mut std::ffi::c_void,
                &mut w_p as *mut _ as *mut std::ffi::c_void,
                &mut part_p as *mut _ as *mut std::ffi::c_void,
                &mut out_p0 as *mut _ as *mut std::ffi::c_void,
                &mut n_in_a as *mut _ as *mut std::ffi::c_void,
                &mut n_out_a as *mut _ as *mut std::ffi::c_void,
                &mut xw_a as *mut _ as *mut std::ffi::c_void,
            ];
            return self.launch3(
                "gemm_q8_0_w16",
                n_out.div_ceil(8) as u32,
                t as u32,
                1,
                128,
                &mut args,
            );
        }
        if t == 1 && ty == 8 && (n_in / 32 <= 32 || w_all) && std::env::var("LLM170_Q8W").as_deref() != Ok("0") {
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &mut xq_p as *mut _ as *mut std::ffi::c_void,
                &mut w_p as *mut _ as *mut std::ffi::c_void,
                &mut part_p as *mut _ as *mut std::ffi::c_void,
                &mut out_p0 as *mut _ as *mut std::ffi::c_void,
                &mut n_in_a as *mut _ as *mut std::ffi::c_void,
                &mut n_out_a as *mut _ as *mut std::ffi::c_void,
                &mut xw_a as *mut _ as *mut std::ffi::c_void,
            ];
            return self.launch3(
                "gemm_q8_0_w",
                n_out.div_ceil(8) as u32,
                1,
                1,
                256,
                &mut args,
            );
        }
        let gz2 = n_out.div_ceil(65535) as u32;
        match ty {
            23 | 20 => args_v.insert(4, &mut out_p0 as *mut _ as *mut std::ffi::c_void),
            _ => args_v.insert(3, &mut out_p0 as *mut _ as *mut std::ffi::c_void),
        }
        let xw_ptr = &mut xw_a as *mut _ as *mut std::ffi::c_void;
        args_v.push(xw_ptr);
        self.launch3(kern, t as u32, gy, gz2, 64, &mut args_v)?;
        Ok(())
    }

    /// np 소형 배치(t=2..4) 4-토큰 GEMV — 가중 1회 독서.
    /// y는 [t][xq_w], out은 [t][n_out] (토큰별 독립 누산·환원).
    pub fn gemm_g4(
        &self,
        ty: u32,
        xq: *const u8,
        w: *const u8,
        ktab2: *const u8,
        n_in: usize,
        n_out: usize,
        xq_w: usize,
        t: usize,
        out: *mut u8,
    ) -> Result<(), String> {
        // plans/74 N3: q5_K 은 워프=행(블록 32, 스트라이드 32레인)이 기본 —
        // 원판(블록 64) 대비 np4 마이크로벤치 +4.5%, 토큰 A/B 동일.
        // LLM170_NO_Q5K4W2=1 이면 원판 복귀.
        let w2 = ty == 13 && std::env::var_os("LLM170_NO_Q5K4W2").is_none();
        let kern = match ty {
            12 => "gemm_q4k4",
            13 => if w2 { "gemm_q5k4_w2" } else { "gemm_q5k4" },
            14 => "gemm_q6k4",
            23 => "gemm_xs4",
            _ => return Err(format!("g4 미지원 타입 {ty}")),
        };
        let gy = n_out.min(65535) as u32;
        let gz = n_out.div_ceil(65535) as u32;
        let mut xp = xq as *mut std::ffi::c_void;
        let mut wp = w as *mut std::ffi::c_void;
        let mut op = out as *mut std::ffi::c_void;
        let mut kt = ktab2 as *mut std::ffi::c_void;
        let mut ni = n_in as i32;
        let mut no = n_out as i32;
        let mut xw = xq_w as i32;
        let mut tt = t as i32;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &mut xp as *mut _ as *mut std::ffi::c_void,
            &mut wp as *mut _ as *mut std::ffi::c_void,
        ];
        if ty == 23 {
            args.push(&mut kt as *mut _ as *mut std::ffi::c_void);
        }
        args.push(&mut op as *mut _ as *mut std::ffi::c_void);
        args.push(&mut ni as *mut _ as *mut std::ffi::c_void);
        args.push(&mut no as *mut _ as *mut std::ffi::c_void);
        args.push(&mut xw as *mut _ as *mut std::ffi::c_void);
        args.push(&mut tt as *mut _ as *mut std::ffi::c_void);
        if w2 && std::env::var_os("LLM170_G4_TRACE").is_some() {
            eprintln!("[g4w2] n_in={n_in} n_out={n_out} t={t} gy={gy} gz={gz}");
        }
        let blk: u32 = if w2 { 32 } else { 64 };
        self.launch3(kern, 1, gy, gz, blk, &mut args)
    }

    /// mmq quant_y 캐시 무효화 (y 원본 재기입 직전 호출 — 부록81).
    pub fn mmq_y_bump(&self) {
        if let Ok(mut c) = self.mmq_y_cache.lock() { c.0 = c.0.wrapping_add(1); }
    }

    /// AR 청크 버퍼 조기 확보 (DecodeState init에서 호출 — 조각화 회피).
    pub fn ar_chunk_prealloc(&self, npair: usize, d: usize, nc_max: usize, ch: usize) -> Result<(), String> {
        self.ar_chunk_bufs(npair, d, nc_max, ch).map(|_| ())
    }

    /// AR 청크 스캔 버퍼 (lend/sstart/pgb/pbuf) — b_t_max 기준 1회 할당.
    pub fn ar_chunk_bufs(&self, npair: usize, d: usize, nc_max: usize, ch: usize) -> Result<(*mut u8, *mut u8, *mut u8, *mut u8), String> {
        let mut g = self.ar_cache.lock().map_err(|e| e.to_string())?;
        if let Some(b) = *g { return Ok(b); }
        let lend = self.alloc(npair * d * d * nc_max)?;
        let sstart = self.alloc(npair * d * d * nc_max)?;
        let pgb = self.alloc(npair * d * nc_max)?;
        let pbuf = self.alloc(npair * ch * d * nc_max)?;
        let b = (lend, sstart, pgb, pbuf);
        *g = Some(b);
        Ok(b)
    }

    /// 사이드 스트림판 — 호출자가 side_wait_main 후 발사/ join2로 합류.
    pub fn gemv_q8_out_s(
        &self,
        xq: *const u8,
        w: *const u8,
        ktab2: *const u8,
        ty: u32,
        n_in: usize,
        n_out: usize,
        out: *mut u8,
        xq_w: usize,
        t: usize,
    ) -> Result<(), String> {
        let part = self.scratch(n_out * 64 * 8)?;
        let gy = n_out.min(65535) as u32;
        let _gz = n_out.div_ceil(65535) as u32;
        let kern = match ty {
            23 => "gemm_xs",
            13 => "gemm_q5k",
            8 => "gemm_q8_0",
            12 => "gemm_q4k",
            14 => "gemm_q6k",
            20 => "gemm_nl",
            11 => "gemm_q3k",
            21 => "gemm_iq3s",
            _ => return Err(format!("미지원 타입 {ty}")),
        };
        let mut xq_p = xq as *mut std::ffi::c_void;
        let mut w_p = w as *mut std::ffi::c_void;
        let mut part_p = part as *mut std::ffi::c_void;
        let mut kt_p = ktab2 as *mut std::ffi::c_void;
        let mut n_in_a = n_in as i32;
        let mut n_out_a = n_out as i32;
        let mut args_v: Vec<*mut std::ffi::c_void> = match ty {
            23 | 20 => vec![
                &mut xq_p as *mut _ as *mut std::ffi::c_void,
                &mut w_p as *mut _ as *mut std::ffi::c_void,
                &mut part_p as *mut _ as *mut std::ffi::c_void,
                &mut kt_p as *mut _ as *mut std::ffi::c_void,
                &mut n_in_a as *mut _ as *mut std::ffi::c_void,
                &mut n_out_a as *mut _ as *mut std::ffi::c_void,
            ],
            _ => vec![
                &mut xq_p as *mut _ as *mut std::ffi::c_void,
                &mut w_p as *mut _ as *mut std::ffi::c_void,
                &mut part_p as *mut _ as *mut std::ffi::c_void,
                &mut n_in_a as *mut _ as *mut std::ffi::c_void,
                &mut n_out_a as *mut _ as *mut std::ffi::c_void,
            ],
        };
        let gz = n_out.div_ceil(65535) as u32;
        let mut out_p0 = out as *mut std::ffi::c_void;
        match ty {
            23 | 20 => args_v.insert(4, &mut out_p0 as *mut _ as *mut std::ffi::c_void),
            _ => args_v.insert(3, &mut out_p0 as *mut _ as *mut std::ffi::c_void),
        }
        let mut xw_a = xq_w as i32;
        let xw_ptr = &mut xw_a as *mut _ as *mut std::ffi::c_void;
        args_v.push(xw_ptr);
        self.launch3s(kern, t as u32, gy, gz, 64, &mut args_v)?;
        Ok(())
    }

    fn tile_core(&self, xq: *const u8, w: *const u8, ktab2: *const u8, ty: u32, n_in: usize, n_out: usize, xq_w: usize, t: usize, out: *mut u8) -> Result<TileLaunch, String> {
        let j128 = std::env::var_os("LLM170_EXACT").is_none()
            && co_loaded(CO_J128) && t > 64;
        self.tile_core_inner(xq, w, ktab2, ty, n_in, n_out, xq_w, t, out, j128)
    }

    /// head 강제판 — j128 타일을 t≤64에서도 (n_out 초대형일 때 이득).
    fn tile_core_head(&self, xq: *const u8, w: *const u8, ktab2: *const u8, ty: u32, n_in: usize, n_out: usize, xq_w: usize, t: usize, out: *mut u8) -> Result<TileLaunch, String> {
        let j128 = std::env::var_os("LLM170_EXACT").is_none()
            && co_loaded(CO_J128);
        self.tile_core_inner(xq, w, ktab2, ty, n_in, n_out, xq_w, t, out, j128)
    }

    fn tile_core_inner(&self, xq: *const u8, w: *const u8, ktab2: *const u8, ty: u32, n_in: usize, n_out: usize, xq_w: usize, t: usize, out: *mut u8, j128: bool) -> Result<TileLaunch, String> {
        // wm·mm 상한 64: t>64 무CO는 유효 커널 없음 — 침묵 오답 대신 에러
        if t > 64 && !j128 {
            return Err(format!("타일 미지원: t={t}는 CO 사전컴파일(j128/v4) 필요"));
        }
        let (v4, odd) = (co_loaded(CO_V4), co_loaded(CO_ODD));
        let kern: &'static str = match ty {
            13 => if j128 && v4 { "gemm_q5k_v4" } else if j128 { "gemm_q5k_j128" } else if std::env::var_os("LLM170_EXACT").is_none() && t >= 32 { "gemm_q5k_wm" } else { "gemm_q5k_mm" },
            12 => if j128 && v4 { "gemm_q4k_v4" } else if j128 { "gemm_q4k_j128" } else if std::env::var_os("LLM170_EXACT").is_none() && t >= 32 { "gemm_q4k_wm" } else { "gemm_q4k_mm" },
            14 => if j128 { "gemm_q6k_j128" } else if std::env::var_os("LLM170_EXACT").is_none() && t >= 32 { "gemm_q6k_wm" } else { "gemm_q6k_mm" },
            23 => if j128 { "gemm_xs_j128" } else if v4 && std::env::var_os("LLM170_XS_V4U").is_some() { "gemm_xs_v4u" } else if std::env::var_os("LLM170_XS_MM").is_some() { "gemm_xs_mm" } else if v4 && std::env::var_os("LLM170_EXACT").is_none() && t >= 32 { "gemm_xs_v4" } else if std::env::var_os("LLM170_EXACT").is_none() && t >= 32 { "gemm_xs_wm" } else { "gemm_xs_mm" },
            20 => if odd && std::env::var_os("LLM170_EXACT").is_none() && t >= 32 { "gemm_nl_v4" } else { return Err("타일 미지원 타입 20 (GEMV 경로 사용)".into()) },
            11 => if odd && std::env::var_os("LLM170_EXACT").is_none() && t >= 32 { "gemm_q3k_v4" } else { return Err("타일 미지원 타입 11 (GEMV 경로 사용)".into()) },
            21 => if odd && std::env::var_os("LLM170_EXACT").is_none() && t >= 32 { "gemm_iq3s_v4" } else { return Err("타일 미지원 타입 21 (GEMV 경로 사용)".into()) },
            8 => if j128 { "gemm_q8_j128" } else { return Err("타일 미지원 타입 8 (GEMV 경로 사용)".into()) },
            _ => return Err(format!("타일 미지원 타입 {ty}")),
        };
        if std::env::var_os("LLM170_TILE_SHAPES").is_some() {
            use std::sync::Mutex;
            use std::sync::OnceLock;
            static SEEN: OnceLock<Mutex<Vec<(String, usize, usize, usize)>>> = OnceLock::new();
            let seen = SEEN.get_or_init(|| Mutex::new(Vec::new()));
            if let Ok(mut v) = seen.lock() {
                let key = (kern.to_string(), n_in, n_out, (t / 128) * 128);
                if !v.contains(&key) {
                    v.push(key.clone());
                    eprintln!("# tile-shape {kern} n_in={n_in} n_out={n_out} t~{t}");
                }
            }
        }
        let mm = kern.ends_with("_mm") || kern.ends_with("_wm") || kern.ends_with("_j128") || kern.ends_with("_v4");
        let rows_per_block: usize = if kern.ends_with("_j128") || kern.ends_with("_v4") { 128 } else if mm { 64 } else { 1 };
        let nblocks = n_out.div_ceil(rows_per_block);
        Ok(TileLaunch {
            kern,
            xp: xq as *mut std::ffi::c_void,
            wp: w as *mut std::ffi::c_void,
            op: out as *mut std::ffi::c_void,
            ktp: ktab2 as *mut std::ffi::c_void,
            ni: n_in as i32,
            no: n_out as i32,
            xw: xq_w as i32,
            // z-그리드 토큰 사분면: t≤128이면 gz=1 (무변형). 초과분은 128씩.
            // j128/v4 판은 토큰 사분면(blockIdx.z) 지원 — t를 그대로 넘긴다(§51 수정판).
            tt: if kern.ends_with("_j128") || kern.ends_with("_v4") {
                t as i32
            } else {
                t.min(128) as i32
            },
            gx: nblocks.min(65535) as u32,
            gz: (nblocks.div_ceil(65535) * t.div_ceil(128)) as u32,
            block: if mm { 256 } else { 64 },
            ktab: ty == 23 || kern == "gemm_nl_v4",
        })
    }

    fn tile_args(l: &mut TileLaunch) -> Vec<*mut std::ffi::c_void> {
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&mut l.xp) as *mut _ as *mut std::ffi::c_void,
            (&mut l.wp) as *mut _ as *mut std::ffi::c_void,
            (&mut l.op) as *mut _ as *mut std::ffi::c_void,
        ];
        if l.ktab {
            args.push((&mut l.ktp) as *mut _ as *mut std::ffi::c_void);
        }
        args.push((&mut l.ni) as *mut _ as *mut std::ffi::c_void);
        args.push((&mut l.no) as *mut _ as *mut std::ffi::c_void);
        args.push((&mut l.xw) as *mut _ as *mut std::ffi::c_void);
        args.push((&mut l.tt) as *mut _ as *mut std::ffi::c_void);
        args
    }

    /// spec verify head 전용 — j128/v4 타일 강제 (가중 1회 독서).
    /// 산술은 동일 W4A8이나 환원 순서가 mm 계열와 달라 스트림 비트계약 대상 아님
    /// (spec 내부 draft↔verify 일관성만 요구).
    pub fn gemm_tile_head(&self, xq: *const u8, w: *const u8, ktab2: *const u8, ty: u32, n_in: usize, n_out: usize, xq_w: usize, t: usize, out: *mut u8) -> Result<(), String> {
        let mut l = self.tile_core_head(xq, w, ktab2, ty, n_in, n_out, xq_w, t, out)?;
        let mut args = Self::tile_args(&mut l);
        self.launch3(l.kern, l.gx, 1, l.gz, l.block, &mut args)
    }

    pub fn gemm_tile(&self, xq: *const u8, w: *const u8, ktab2: *const u8, ty: u32, n_in: usize, n_out: usize, xq_w: usize, t: usize, out: *mut u8) -> Result<(), String> {
        let mut l = self.tile_core(xq, w, ktab2, ty, n_in, n_out, xq_w, t, out)?;
        let mut args = Self::tile_args(&mut l);
        let r = self.launch3(l.kern, l.gx, 1, l.gz, l.block, &mut args);
        if std::env::var_os("LLM170_TILE_PROF").is_some() {
            self.sync().ok();
            let ti = std::time::Instant::now();
            self.launch3(l.kern, l.gx, 1, l.gz, l.block, &mut args).ok();
            self.sync().ok();
            eprintln!("tileprof ty={ty} {n_in}x{n_out} t={t} kern={} {:.3}ms", l.kern, ti.elapsed().as_secs_f64()*1e3);
        }
        r
    }

    /// 커널 속성 조회 — (레지스터, 로컬 바이트, 최대 스레드). 점유율 진단용.
    pub fn kern_attrs(&self, name: &str) -> Option<(i32, usize, i32)> {
        let f = *self.fns.get(name)?;
        let mut a: hip::hipFuncAttributes = unsafe { std::mem::zeroed() };
        let r = unsafe { hip::hipFuncGetAttributes(&mut a, f as *const std::ffi::c_void) };
        if r == hip::hipError_t_hipSuccess {
            Some((a.numRegs, a.localSizeBytes, a.maxThreadsPerBlock))
        } else {
            None
        }
    }

    /// gemm_tile의 사이드 스트림판 — 커널 선택·인자 구성은 공용 코어에 위임.
    pub fn gemm_tile_s(&self, xq: *const u8, w: *const u8, ktab2: *const u8, ty: u32, n_in: usize, n_out: usize, xq_w: usize, t: usize, out: *mut u8) -> Result<(), String> {
        let mut l = self.tile_core(xq, w, ktab2, ty, n_in, n_out, xq_w, t, out)?;
        let mut args = Self::tile_args(&mut l);
        self.launch3s(l.kern, l.gx, 1, l.gz, l.block, &mut args)
    }

    /// q6_K → f16 전개 + gemm_f16_v4 (deq-f16 경로, 부록42).
    pub fn gemm_f16_q6(&self, y_f32: *const u8, w: *const u8, n_in: usize, n_out: usize, t: usize, out: *mut u8) -> Result<(), String> {
        let fns = &self.fns;
        let fq = *fns.get("dequant_q6k_f16").ok_or("dequant_q6k_f16 없음")?;
        let fm = *fns.get("gemm_f16_v4").ok_or("gemm_f16_v4 없음")?;
        // f16 전개 버퍼 (지속: w 주소 키 캐시)
        let key = w as usize;
        let wf16 = {
            let mut c = self.f16_cache.lock().map_err(|e| e.to_string())?;
            if let Some(&p) = c.get(&key) { p }
            else {
                let blocks = n_in.div_ceil(256);   // 256요소 슈퍼블록 격자(부분 허용)
                let p = self.alloc(n_out * n_in * 2)? as *mut u8;
                unsafe {
                    let mut a1 = w as *mut std::ffi::c_void;
                    let mut a2 = p as *mut std::ffi::c_void;
                    let mut a3 = blocks as i32;
                    let mut a4 = n_out as i32;
                    let mut args = vec![&mut a1 as *mut _ as *mut _, &mut a2 as *mut _ as *mut _, &mut a3 as *mut _ as *mut _, &mut a4 as *mut _ as *mut _];
                    ck(hip::hipModuleLaunchKernel(fq, n_out as u32, blocks as u32, 1, 256, 1, 1, 0, self.stream, args.as_mut_ptr(), std::ptr::null_mut()), "dequant_q6k_f16")?;
                }
                c.insert(key, p);
                p
            }
        };
        // y: f32 → **llama q8_1**(mmq_quant_y, 144B/128원소) — v4 GEMM은 llama
        // 계열이라 이 레이아웃을 기대한다. 우리 quant_q8(1.375B/원소)을 넣으면
        // 레이아웃이 어긋나 쓰레기 토큰이 나온다(plans/65 §13-14 실측).
        let xq_w = (n_in / 128) * 36;
        let tr = t.div_ceil(128) * 128;   // 커널의 128 단위 사분면 경계 (범위 밖 쓰기 방지)
        let mut xq = self.mmq_y2.lock().map_err(|e| e.to_string())?;
        let xq_p = if xq.0 < xq_w * tr {
            let p = self.alloc(xq_w * tr * 4)? as *mut u8;
            *xq = (xq_w * tr, p);
            p
        } else { xq.1 };
        unsafe {
            let mut a1 = y_f32 as *mut std::ffi::c_void;
            let mut a2 = xq_p as *mut std::ffi::c_void;
            let mut a3 = n_in as i32;
            let mut a4 = xq_w as i32;
            let mut a5 = t as i32;
            let mut args = vec![&mut a1 as *mut _ as *mut _, &mut a2 as *mut _ as *mut _, &mut a3 as *mut _ as *mut _, &mut a4 as *mut _ as *mut _, &mut a5 as *mut _ as *mut _];
            // quant_q8_b: grid(nblk/64, t) block 64 — kernels.rs quant_q8 시그니처 (x, xq, n, xq_w)
            let fq8 = *fns.get("mmq_quant_y").ok_or("mmq_quant_y 없음")?;
            ck(hip::hipModuleLaunchKernel(fq8, (n_in / 128) as u32, t as u32, 1, 32, 1, 1, 0, self.stream, args.as_mut_ptr(), std::ptr::null_mut()), "mmq_quant_y")?;
            let mut b1 = xq_p as *mut std::ffi::c_void;
            let mut b2 = wf16 as *mut std::ffi::c_void;
            let mut b3 = out as *mut std::ffi::c_void;
            let mut b4 = n_in as i32;
            let mut b5 = n_out as i32;
            let mut b6 = xq_w as i32;
            let mut b7 = tr as i32;
            let _args2 = vec![&mut b1 as *mut _ as *mut _, &mut b2 as *mut _ as *mut _, &mut b3 as *mut _ as *mut _, &mut b4 as *mut _ as *mut _, &mut b5 as *mut _ as *mut _, &mut b6 as *mut _ as *mut _, &mut b7 as *mut _ as *mut _];
            // z-그리드 사분면 CO: 단일 런치 (tt=min(t,128), gz=사분면)
            {
              let mut z1 = xq_p as *mut std::ffi::c_void;
              let mut z3 = out as *mut std::ffi::c_void;
              let mut z7 = t.min(128) as i32;
              let mut az: Vec<*mut std::ffi::c_void> = vec![&mut z1 as *mut _ as *mut _, &mut b2 as *mut _ as *mut _, &mut z3 as *mut _ as *mut _,
                  &mut b4 as *mut _ as *mut _, &mut b5 as *mut _ as *mut _, &mut b6 as *mut _ as *mut _, &mut z7 as *mut _ as *mut _];
              ck(hip::hipModuleLaunchKernel(fm, ((n_out + 127) / 128) as u32, 1, (tr / 128) as u32, 256, 1, 1, 0, self.stream, az.as_mut_ptr(), std::ptr::null_mut()), "gemm_f16_v4")?;
            }
        if std::env::var_os("LLM170_DEQ_DUMP").is_some() {
            self.sync().ok();
            let _ = std::fs::write("/tmp/deq_wf16.f16", std::slice::from_raw_parts(wf16 as *const u8, n_out * n_in * 2));
            let _ = std::fs::write("/tmp/deq_w.bin", std::slice::from_raw_parts(w as *const u8, n_out.min(1) * (n_in/256) * 210 + 210));
            let _ = std::fs::write("/tmp/deq_xq.bin", std::slice::from_raw_parts(xq_p as *const u8, xq_w * t * 4));
            eprintln!("DEQ_DUMP: wf16 {}B xq {}B (ni={n_in} no={n_out} t={t} xw={xq_w})", n_out*n_in*2, xq_w*t*4);
            std::process::exit(0);
        }
        }
        Ok(())
    }
    pub fn gemm_f16_deq(&self, ty: u32, y_f32: *const u8, w: *const u8, n_in: usize, n_out: usize, t: usize, out: *mut u8) -> Result<(), String> {
        let fns = &self.fns;
        // f16 전개 커널 선택 — 우리 .co의 GEMM이 소비하는 레이아웃으로 전개한다.
        let (fq, _blk_div) = match ty {
            14 => (*fns.get("dequant_q6k_f16").ok_or("dequant_q6k_f16 없음")?, 1usize),
            12 => (*fns.get("dequant_q4k_f16").ok_or("dequant_q4k_f16 없음")?, 1),
            8 => (*fns.get("dequant_q8_0_f16").ok_or("dequant_q8_0_f16 없음")?, 1),
            _ => return Err(format!("f16 경로 미지원 타입 {ty}")),
        };
        let fm = *fns.get("gemm_f16_v4").ok_or("gemm_f16_v4 없음")?;
        // f16 전개 버퍼 (지속: w 주소 키 캐시)
        let key = (w as usize) ^ ((ty as usize) << 60);
        // 크기 가드: 전문가 스택(수십 GB)은 f16 캐시 불가 → 호출자가 거른다.
        if (n_out as u64) * (n_in as u64) * 2 > 512 * 1024 * 1024 {
            return Err("f16 캐시 상한 초과".into());
        }
        let wf16 = {
            let mut c = self.f16_cache.lock().map_err(|e| e.to_string())?;
            if let Some(&p) = c.get(&key) { p }
            else {
                let _blocks = n_in / 256;
                let p = self.alloc(n_out * n_in * 2)? as *mut u8;
                unsafe {
                    let mut a1 = w as *mut std::ffi::c_void;
                    let mut a2 = p as *mut std::ffi::c_void;
                    let mut a3 = (n_in / 32) as i32;   // 행당 32블록 수 = 행 스트라이드
                    let mut a4 = n_out as i32;
                    let mut a5 = n_in as i32;          // 유효 요소 수(부분 블록 가드)
                    let mut args = vec![&mut a1 as *mut _ as *mut _, &mut a2 as *mut _ as *mut _, &mut a3 as *mut _ as *mut _, &mut a4 as *mut _ as *mut _, &mut a5 as *mut _ as *mut _];
                    ck(hip::hipModuleLaunchKernel(fq, n_out as u32, n_in.div_ceil(256) as u32, 1, 256, 1, 1, 0, self.stream, args.as_mut_ptr(), std::ptr::null_mut()), "dequant_f16")?;
                }
                c.insert(key, p);
                p
            }
        };
        // y: f32 → 우리 xq (quant_q8) — y_f32 에서 직접
        let xq_w = n_in/4 + n_in/32 + n_in/16;
        // 커널은 t를 128 단위 사분면으로 소비하고 드레인도 그 경계까지 쓴다 →
        // 부분 t에서 범위 밖 쓰기가 생긴다. 런치 t를 128 배수로 올려 in-bounds로 만든다
        // (행 < t 만 유효, 호출자가 그만큼만 읽는다).
        let tr = t.div_ceil(128) * 128;
        let mut xq = self.mmq_y2.lock().map_err(|e| e.to_string())?;
        let xq_p = if xq.0 < xq_w * tr {
            let p = self.alloc(xq_w * tr * 4)? as *mut u8;
            *xq = (xq_w * tr, p);
            p
        } else { xq.1 };
        unsafe {
            let mut a1 = y_f32 as *mut std::ffi::c_void;
            let mut a2 = xq_p as *mut std::ffi::c_void;
            let mut a3 = n_in as i32;
            let mut a4 = xq_w as i32;
            let mut a5 = t as i32;
            let mut args = vec![&mut a1 as *mut _ as *mut _, &mut a2 as *mut _ as *mut _, &mut a3 as *mut _ as *mut _, &mut a4 as *mut _ as *mut _, &mut a5 as *mut _ as *mut _];
            // quant_q8_b: grid(nblk/64, t) block 64 — kernels.rs quant_q8 시그니처 (x, xq, n, xq_w)
            let fq8 = *fns.get("quant_q8").ok_or("quant_q8 없음")?;
            ck(hip::hipModuleLaunchKernel(fq8, ((n_in/32).div_ceil(64)) as u32, t as u32, 1, 64, 1, 1, 0, self.stream, args.as_mut_ptr(), std::ptr::null_mut()), "quant_q8")?;
            let mut b1 = xq_p as *mut std::ffi::c_void;
            let mut b2 = wf16 as *mut std::ffi::c_void;
            let mut b3 = out as *mut std::ffi::c_void;
            let mut b4 = n_in as i32;
            let mut b5 = n_out as i32;
            let mut b6 = xq_w as i32;
            let mut b7 = tr as i32;
            let _args2 = vec![&mut b1 as *mut _ as *mut _, &mut b2 as *mut _ as *mut _, &mut b3 as *mut _ as *mut _, &mut b4 as *mut _ as *mut _, &mut b5 as *mut _ as *mut _, &mut b6 as *mut _ as *mut _, &mut b7 as *mut _ as *mut _];
            // z-그리드 사분면 CO: 단일 런치 (tt=min(t,128), gz=사분면)
            {
              let mut z1 = xq_p as *mut std::ffi::c_void;
              let mut z3 = out as *mut std::ffi::c_void;
              let mut z7 = t.min(128) as i32;
              let mut az: Vec<*mut std::ffi::c_void> = vec![&mut z1 as *mut _ as *mut _, &mut b2 as *mut _ as *mut _, &mut z3 as *mut _ as *mut _,
                  &mut b4 as *mut _ as *mut _, &mut b5 as *mut _ as *mut _, &mut b6 as *mut _ as *mut _, &mut z7 as *mut _ as *mut _];
              ck(hip::hipModuleLaunchKernel(fm, ((n_out + 127) / 128) as u32, 1, (tr / 128) as u32, 256, 1, 1, 0, self.stream, az.as_mut_ptr(), std::ptr::null_mut()), "gemm_f16_v4")?;
            }
        if std::env::var_os("LLM170_DEQ_DUMP").is_some() {
            self.sync().ok();
            let _ = std::fs::write("/tmp/deq_wf16.f16", std::slice::from_raw_parts(wf16 as *const u8, n_out * n_in * 2));
            let _ = std::fs::write("/tmp/deq_w.bin", std::slice::from_raw_parts(w as *const u8, n_out.min(1) * (n_in/256) * 210 + 210));
            let _ = std::fs::write("/tmp/deq_xq.bin", std::slice::from_raw_parts(xq_p as *const u8, xq_w * t * 4));
            eprintln!("DEQ_DUMP: wf16 {}B xq {}B (ni={n_in} no={n_out} t={t} xw={xq_w})", n_out*n_in*2, xq_w*t*4);
            std::process::exit(0);
        }
        }
        Ok(())
    }

    /// llama MMQ (mul_mat_q<q4_K/q5_K,128>) — f32 활성 직양자화 + 원형 런치.
    /// 하니스 검증: q4_K maxrel 6e-4, q5_K maxrel 1.5e-3 (plans/27 부록5·14).
    pub fn gemm_mmq(&self, ty: u32, y_f32: *const u8, w: *const u8, n_in: usize, n_out: usize, t: usize, out: *mut u8) -> Result<(), String> {
        let fns = &self.fns;
        // D4 타입(q6_K/iq4_xs)은 f32-d 전용 양자화 (mmq.cuh ds_layout 계약)
        // DS 레이아웃(mmq.cuh): Q6K/IQ4XS/Q8_0 → D4, Q4K/Q5K → DS4.
        // Q8_0(8)도 D4라 기존 quant_y_d4와 포맷 공유를 기대(plans/71 실험).
        let fq = *fns.get(if matches!(ty, 8 | 14 | 23) { "mmq_quant_y_d4" } else { "mmq_quant_y" })
            .ok_or("mmq quant 없음")?;
        let j: usize = if std::env::var_os("LLM170_MMQ64").is_some() { 64 } else { 128 };
        let sym = match ty {
            12 => { let js = if j == 64 { "64" } else { "128" }; format!("_ZL9mul_mat_qIL9ggml_type12ELi{}ELb0EEvPKcPKiS4_S4_PfS5_PKf15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_", js) }
            13 => { let js = if j == 64 { "64" } else { "128" }; format!("_ZL9mul_mat_qIL9ggml_type13ELi{}ELb0EEvPKcPKiS4_S4_PfS5_PKf15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_", js) }
            14 => { let js = if j == 64 { "64" } else { "128" }; format!("_ZL9mul_mat_qIL9ggml_type14ELi{}ELb0EEvPKcPKiS4_S4_PfS5_PKf15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_", js) }
            23 => { let js = if j == 64 { "64" } else { "128" }; format!("_ZL9mul_mat_qIL9ggml_type23ELi{}ELb0EEvPKcPKiS4_S4_PfS5_PKf15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_", js) }
            8 => { let js = if j == 64 { "64" } else { "128" }; format!("_ZL9mul_mat_qIL9ggml_type8ELi{}ELb0EEvPKcPKiS4_S4_PfS5_PKf15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_", js) }
            _ => return Err(format!("MMQ 미지원 타입 {ty}")),
        };
        let fm = *fns.get(&sym[..]).ok_or("mul_mat_q 없음")?;
        // q6_K는 GGUF(=ggml 정준) 레이아웃을 그대로 쓴다. mul_mat_q는 llama.cpp
        // mmq.cuh 직인스턴스화라 정준 블록(ql|qh|scales|d)을 기대한다 — 과거의
        // requant_q6k_canonical(d-first 재배열)은 정준 입력을 오히려 깨뜨려
        // ≥32토큰 프리필에서 쓰레기 토큰을 냈다(2026-09-12 실측). 레거시 경로는
        // LLM170_Q6RQ=1로만 복원.
        let w_eff = if ty == 14 && std::env::var_os("LLM170_Q6RQ").is_some() {
            let key = w as usize ^ 0xdeadbeef;
            let mut c = self.canon_q6.lock().map_err(|e| e.to_string())?;
            if let Some(&p2) = c.get(&key) { p2 }
            else {
                let blocks2 = n_in / 256;
                let p2 = self.alloc(n_out * blocks2 * 210)? as *mut u8;
                let fqr = *fns.get("requant_q6k_canonical").ok_or("requant 없음")?;
                unsafe {
                    let mut a1 = w as *mut std::ffi::c_void;
                    let mut a2 = p2 as *mut std::ffi::c_void;
                    let mut a3 = blocks2 as i32;
                    let mut a4 = n_out as i32;
                    let mut args = vec![&mut a1 as *mut _ as *mut _, &mut a2 as *mut _ as *mut _, &mut a3 as *mut _ as *mut _, &mut a4 as *mut _ as *mut _];
                    ck(hip::hipModuleLaunchKernel(fqr, n_out as u32, blocks2 as u32, 1, 128, 1, 1, 0, self.stream, args.as_mut_ptr(), std::ptr::null_mut()), "requant_q6k_canonical")?;
                }
                if std::env::var_os("LLM170_RQ_DUMP").is_some() {
                    self.sync().ok();
                    let _ = std::fs::write("/tmp/rq_out.bin", unsafe { std::slice::from_raw_parts(p2 as *const u8, 420) });
                    let _ = std::fs::write("/tmp/rq_in.bin", unsafe { std::slice::from_raw_parts(w as *const u8, 420) });
                    eprintln!("RQ_DUMP 완료 (첫 블록 2개)");
                }
                c.insert(key, p2);
                p2
            }
        } else { w as *mut u8 };
        // 전용 y 버퍼 — scratch 풀은 동일 크기 호출에 같은 포인터 반환(비동기
        // 재작성 위험). MMQ y는 단일 소유로 격리.
        let yb = {
            // 마지막 128행 타일은 t를 넘어 읽는다 — llama.cpp도 y 버퍼에
            // J_max*sizeof(block_q8_1_mmq) 슬랙을 둔다(mmq.cu nbytes_src1_q8_1).
            // 슬랙이 없으면 t가 128의 배수가 아닐 때(예: 검증 배치 t=33) OOB read.
            const MMQ_Y_SLACK: usize = 128 * 144;
            let need = (n_in / 128) * t * 144 + MMQ_Y_SLACK;
            let mut sc = self.mmq_y.lock().map_err(|e| e.to_string())?;
            if sc.0 < need {
                if !sc.1.is_null() { unsafe { hip::hipFree(sc.1 as *mut _) }; }
                sc.1 = self.alloc(need)? as *mut u8;
                sc.0 = need;
            }
            sc.1
        };
        let mut yp = yb as *mut std::ffi::c_void;
        let mut ysrc = y_f32 as *const std::ffi::c_void;
        let mut nt = t as i32;
        let mut ni_a = n_in as i32;
        // 부록81: 동일 에포크·동일 y원본이면 재양자화 스킵 (층당 1회).
        let y_key = (y_f32 as usize, (n_in / 128) * t * 144);
        // 회귀 픽스(부록90): 캐시는 메인 yb(mmq_y)만 — 사이드 yb(mmq_y_s)는
        // 별도 버퍼라 히트 시 미초기화 y로 mul_mat_q를 돌렸다 (장문 가비지).
        let this_is_main = yb == { self.mmq_y.lock().map(|c| c.1).unwrap_or(std::ptr::null_mut()) };
        let cached = false && this_is_main && { self.mmq_y_cache.lock().map(|c| *c == (c.0, y_key.0, y_key.1)).unwrap_or(false) };
        if !cached {
            if ty == 8 {
                // plans/71: Q8_0의 y양자화는 신형 quantize_mmq_q8_1<D4,false>
                // (ROCm 10 빌드) — 구형 mmq_quant_y*는 block_q8_1_mmq ABI가 달라
                // 혼합 시 HIP 700. 인자: (x, ids=null, vy, ne00, s01, s02, s03,
                // ne0, ne1, ne2, n_expert_used) 그리드 (t, ceil(n_in/512), 1) 128.
                let fq2 = *self.fns.get("_ZL17quantize_mmq_q8_1IL18mmq_q8_1_ds_layout0ELb0EEvPKfPKiPvllllliii")
                    .ok_or("quantize_mmq_q8_1<D4> 없음")?;
                let mut xp2 = y_f32 as *mut std::ffi::c_void;
                let mut idsp: *mut std::ffi::c_void = std::ptr::null_mut();
                let mut ne00 = n_in as i64;
                let mut s01 = n_in as i64;
                let mut s02 = 0i64;
                let mut s03 = 0i64;
                let mut ne0 = n_in as i64;
                let mut ne1 = t as i32;
                let mut ne2 = 1i32;
                let mut neu = 0i32;
                let mut q2 = vec![
                    &mut xp2 as *mut _ as *mut std::ffi::c_void,
                    &mut idsp as *mut _ as *mut std::ffi::c_void,
                    &mut yp as *mut _ as *mut std::ffi::c_void,
                    &mut ne00 as *mut _ as *mut std::ffi::c_void,
                    &mut s01 as *mut _ as *mut std::ffi::c_void,
                    &mut s02 as *mut _ as *mut std::ffi::c_void,
                    &mut s03 as *mut _ as *mut std::ffi::c_void,
                    &mut ne0 as *mut _ as *mut std::ffi::c_void,
                    &mut ne1 as *mut _ as *mut std::ffi::c_void,
                    &mut ne2 as *mut _ as *mut std::ffi::c_void,
                    &mut neu as *mut _ as *mut std::ffi::c_void,
                ];
                // ne0는 128 배수여야 함(assert 위) — n_in이 128 미만 배수면
                // 상위 경로에서 이미 128 정렬(27B/Flash 폭은 전부 128배수).
                unsafe {
                    let gy = n_in.div_ceil(512) as u32;
                    ck(hip::hipModuleLaunchKernel(fq2, t as u32, gy, 1, 128, 1, 1, 0, self.stream, q2.as_mut_ptr(), std::ptr::null_mut()), "quantize_mmq_q8_1")?;
                }
            } else {
            unsafe {
                let mut qargs = vec![
                    &mut ysrc as *mut _ as *mut std::ffi::c_void,
                    &mut yp as *mut _ as *mut std::ffi::c_void,
                    &mut nt as *mut _ as *mut std::ffi::c_void,
                    &mut ni_a as *mut _ as *mut std::ffi::c_void,
                ];
                self.ktr_mark("mmq_quant_y", t as u32);
                ck(hip::hipModuleLaunchKernel(fq, (n_in / 128) as u32, t as u32, 1, 32, 1, 1, 0, self.stream, qargs.as_mut_ptr(), std::ptr::null_mut()), "mmq_quant_y")?;
                self.ktr_mark("mmq_quant_y", t as u32);
            }
            }
            if let Ok(mut c) = self.mmq_y_cache.lock() { *c = (c.0, y_key.0, y_key.1); }
        }
        fn fd3(d: u32) -> [u32; 3] {
            let mut l = 0u32;
            while l < 32 && (1u32 << l) < d { l += 1; }
            let mp = ((((1u64) << 32) * (((1u64) << l) - d as u64)) / d as u64 + 1) as u32;
            [mp, l, d]
        }
        let j: usize = if std::env::var_os("LLM170_MMQ64").is_some() { 64 } else { 128 };
        // 블록 원소수(qk): K계열 256, Q8_0은 32 — launcher의 ncols_x/qk 계약.
        // n_in/256 하드코딩은 Q8_0에서 8배 작아 인덱싱 붕괴(가비지)였다(plans/71).
        let qk: usize = if ty == 8 { 32 } else { 256 };
        let nbk = (n_in / qk) as u32;
        let mut bpn = fd3(nbk);
        let mut one = fd3(1);
        let j_now: usize = if std::env::var_os("LLM170_MMQ64").is_some() { 64 } else { 128 };
        let mut ntx_fd = fd3(((t + j_now - 1) / j_now) as u32);
        let z3: [u32; 3] = [0, 0, 0];
        let mut ax = w_eff as *mut std::ffi::c_void;
        let mut ay = yb as *mut std::ffi::c_void;
        let mut aid: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut aeb: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut adst = out as *mut std::ffi::c_void;
        let mut afx: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut ays: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut p_nrows = n_out as i32;
        let mut p_ncolsdst = t as i32;
        let mut p_srow = (n_in / qk) as i32;
        let mut p_ncolsy = t as i32;
        let mut p_scol = n_out as i32;
        let smem: i32 = (j * 4 + 128 * 76 * 4 + ((j * 144 + 1023) / 1024) * 1024) as i32;
        unsafe {
            ck(hip::hipFuncSetAttribute(fm as *const _, hip::hipFuncAttribute_hipFuncAttributeMaxDynamicSharedMemorySize, smem), "mmq smem attr")?;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &mut ax as *mut _ as *mut _, &mut ay as *mut _ as *mut _,
                &mut aid as *mut _ as *mut _, &mut aeb as *mut _ as *mut _,
                &mut adst as *mut _ as *mut _, &mut afx as *mut _ as *mut _,
                &mut ays as *mut _ as *mut _,
                bpn.as_mut_ptr() as *mut _, &mut p_nrows as *mut _ as *mut _,
                &mut p_ncolsdst as *mut _ as *mut _, &mut p_srow as *mut _ as *mut _,
                &mut p_ncolsy as *mut _ as *mut _, &mut p_scol as *mut _ as *mut _,
                one.as_mut_ptr() as *mut _, one.as_mut_ptr() as *mut _,
                z3.as_ptr() as *mut _, z3.as_ptr() as *mut _, z3.as_ptr() as *mut _,
                one.as_mut_ptr() as *mut _, one.as_mut_ptr() as *mut _,
                z3.as_ptr() as *mut _, z3.as_ptr() as *mut _, z3.as_ptr() as *mut _,
                ntx_fd.as_mut_ptr() as *mut _,
            ];
            let tag: &'static str = match ty {
                12 => "mmq_q4k",
                13 => "mmq_q5k",
                14 => "mmq_q6k",
                23 => "mmq_xs",
                _ => "mmq_other",
            };
            self.ktr_mark(tag, t as u32);
            ck(hip::hipModuleLaunchKernel(fm, ((n_out + 127) / 128) as u32, ((t + 127) / 128) as u32, 1, 32, 8, 1, smem as u32, self.stream, args.as_mut_ptr(), std::ptr::null_mut()), "mul_mat_q")?;
            self.ktr_mark(tag, t as u32);
        if std::env::var_os("LLM170_MMQ_ARGS").is_some() {
            eprintln!("mmq_args ty={ty} n_in={n_in} n_out={n_out} t={t} grid=({},{},1) blk=(32,8) smem={smem} srow={} scol={} nrows={}",
                (n_out + 127) / 128, (t + 127) / 128, n_in / 256, n_out, n_out);
        }
        }
        Ok(())
    }
    pub fn gemm_mmq_s(&self, ty: u32, y_f32: *const u8, w: *const u8, n_in: usize, n_out: usize, t: usize, out: *mut u8) -> Result<(), String> {
        let fns = &self.fns;
        // D4 타입(q6_K/iq4_xs)은 f32-d 전용 양자화 (mmq.cuh ds_layout 계약)
        // DS 레이아웃(mmq.cuh): Q6K/IQ4XS/Q8_0 → D4, Q4K/Q5K → DS4.
        // Q8_0(8)도 D4라 기존 quant_y_d4와 포맷 공유를 기대(plans/71 실험).
        let fq = *fns.get(if matches!(ty, 8 | 14 | 23) { "mmq_quant_y_d4" } else { "mmq_quant_y" })
            .ok_or("mmq quant 없음")?;
        let _j: usize = if std::env::var_os("LLM170_MMQ64").is_some() { 64 } else { 128 };
        let sym = match ty {
            12 => "_ZL9mul_mat_qIL9ggml_type12ELi128ELb0EEvPKcPKiS4_S4_PfS5_PKf15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_",
            13 => "_ZL9mul_mat_qIL9ggml_type13ELi128ELb0EEvPKcPKiS4_S4_PfS5_PKf15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_",
            14 => "_ZL9mul_mat_qIL9ggml_type14ELi128ELb0EEvPKcPKiS4_S4_PfS5_PKf15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_",
            23 => "_ZL9mul_mat_qIL9ggml_type23ELi128ELb0EEvPKcPKiS4_S4_PfS5_PKf15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_",
            _ => return Err(format!("MMQ 미지원 타입 {ty}")),
        };
        let fm = *fns.get(&sym[..]).ok_or("mul_mat_q 없음")?;
        // q6_K는 GGUF(=ggml 정준) 레이아웃을 그대로 쓴다. mul_mat_q는 llama.cpp
        // mmq.cuh 직인스턴스화라 정준 블록(ql|qh|scales|d)을 기대한다 — 과거의
        // requant_q6k_canonical(d-first 재배열)은 정준 입력을 오히려 깨뜨려
        // ≥32토큰 프리필에서 쓰레기 토큰을 냈다(2026-09-12 실측). 레거시 경로는
        // LLM170_Q6RQ=1로만 복원.
        let w_eff = if ty == 14 && std::env::var_os("LLM170_Q6RQ").is_some() {
            let key = w as usize ^ 0xdeadbeef;
            let mut c = self.canon_q6.lock().map_err(|e| e.to_string())?;
            if let Some(&p2) = c.get(&key) { p2 }
            else {
                let blocks2 = n_in / 256;
                let p2 = self.alloc(n_out * blocks2 * 210)? as *mut u8;
                let fqr = *fns.get("requant_q6k_canonical").ok_or("requant 없음")?;
                unsafe {
                    let mut a1 = w as *mut std::ffi::c_void;
                    let mut a2 = p2 as *mut std::ffi::c_void;
                    let mut a3 = blocks2 as i32;
                    let mut a4 = n_out as i32;
                    let mut args = vec![&mut a1 as *mut _ as *mut _, &mut a2 as *mut _ as *mut _, &mut a3 as *mut _ as *mut _, &mut a4 as *mut _ as *mut _];
                    ck(hip::hipModuleLaunchKernel(fqr, n_out as u32, blocks2 as u32, 1, 128, 1, 1, 0, self.stream2, args.as_mut_ptr(), std::ptr::null_mut()), "requant_q6k_canonical")?;
                }
                if std::env::var_os("LLM170_RQ_DUMP").is_some() {
                    self.sync().ok();
                    let _ = std::fs::write("/tmp/rq_out.bin", unsafe { std::slice::from_raw_parts(p2 as *const u8, 420) });
                    let _ = std::fs::write("/tmp/rq_in.bin", unsafe { std::slice::from_raw_parts(w as *const u8, 420) });
                    eprintln!("RQ_DUMP 완료 (첫 블록 2개)");
                }
                c.insert(key, p2);
                p2
            }
        } else { w as *mut u8 };
        // 전용 y 버퍼 — scratch 풀은 동일 크기 호출에 같은 포인터 반환(비동기
        // 재작성 위험). MMQ y는 단일 소유로 격리.
        let yb = {
            const MMQ_Y_SLACK: usize = 128 * 144;
            let need = (n_in / 128) * t * 144 + MMQ_Y_SLACK;
            let mut sc = self.mmq_y_s.lock().map_err(|e| e.to_string())?;
            if sc.0 < need {
                if !sc.1.is_null() { unsafe { hip::hipFree(sc.1 as *mut _) }; }
                sc.1 = self.alloc(need)? as *mut u8;
                sc.0 = need;
            }
            sc.1
        };
        let mut yp = yb as *mut std::ffi::c_void;
        let mut ysrc = y_f32 as *const std::ffi::c_void;
        let mut nt = t as i32;
        let mut ni_a = n_in as i32;
        unsafe {
            let mut qargs = vec![
                &mut ysrc as *mut _ as *mut std::ffi::c_void,
                &mut yp as *mut _ as *mut std::ffi::c_void,
                &mut nt as *mut _ as *mut std::ffi::c_void,
                &mut ni_a as *mut _ as *mut std::ffi::c_void,
            ];
            ck(hip::hipModuleLaunchKernel(fq, (n_in / 128) as u32, t as u32, 1, 32, 1, 1, 0, self.stream2, qargs.as_mut_ptr(), std::ptr::null_mut()), "mmq_quant_y")?;
        }
        fn fd3(d: u32) -> [u32; 3] {
            let mut l = 0u32;
            while l < 32 && (1u32 << l) < d { l += 1; }
            let mp = ((((1u64) << 32) * (((1u64) << l) - d as u64)) / d as u64 + 1) as u32;
            [mp, l, d]
        }
        let j: usize = if std::env::var_os("LLM170_MMQ64").is_some() { 64 } else { 128 };
        // 블록 원소수(qk): K계열 256, Q8_0은 32 — launcher의 ncols_x/qk 계약.
        // n_in/256 하드코딩은 Q8_0에서 8배 작아 인덱싱 붕괴(가비지)였다(plans/71).
        let qk: usize = if ty == 8 { 32 } else { 256 };
        let nbk = (n_in / qk) as u32;
        let mut bpn = fd3(nbk);
        let mut one = fd3(1);
        let j_now: usize = if std::env::var_os("LLM170_MMQ64").is_some() { 64 } else { 128 };
        let mut ntx_fd = fd3(((t + j_now - 1) / j_now) as u32);
        let z3: [u32; 3] = [0, 0, 0];
        let mut ax = w_eff as *mut std::ffi::c_void;
        let mut ay = yb as *mut std::ffi::c_void;
        let mut aid: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut aeb: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut adst = out as *mut std::ffi::c_void;
        let mut afx: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut ays: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut p_nrows = n_out as i32;
        let mut p_ncolsdst = t as i32;
        let mut p_srow = (n_in / qk) as i32;
        let mut p_ncolsy = t as i32;
        let mut p_scol = n_out as i32;
        let smem: i32 = (j * 4 + 128 * 76 * 4 + ((j * 144 + 1023) / 1024) * 1024) as i32;
        unsafe {
            ck(hip::hipFuncSetAttribute(fm as *const _, hip::hipFuncAttribute_hipFuncAttributeMaxDynamicSharedMemorySize, smem), "mmq smem attr")?;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &mut ax as *mut _ as *mut _, &mut ay as *mut _ as *mut _,
                &mut aid as *mut _ as *mut _, &mut aeb as *mut _ as *mut _,
                &mut adst as *mut _ as *mut _, &mut afx as *mut _ as *mut _,
                &mut ays as *mut _ as *mut _,
                bpn.as_mut_ptr() as *mut _, &mut p_nrows as *mut _ as *mut _,
                &mut p_ncolsdst as *mut _ as *mut _, &mut p_srow as *mut _ as *mut _,
                &mut p_ncolsy as *mut _ as *mut _, &mut p_scol as *mut _ as *mut _,
                one.as_mut_ptr() as *mut _, one.as_mut_ptr() as *mut _,
                z3.as_ptr() as *mut _, z3.as_ptr() as *mut _, z3.as_ptr() as *mut _,
                one.as_mut_ptr() as *mut _, one.as_mut_ptr() as *mut _,
                z3.as_ptr() as *mut _, z3.as_ptr() as *mut _, z3.as_ptr() as *mut _,
                ntx_fd.as_mut_ptr() as *mut _,
            ];
            ck(hip::hipModuleLaunchKernel(fm, ((n_out + 127) / 128) as u32, ((t + 127) / 128) as u32, 1, 32, 8, 1, smem as u32, self.stream2, args.as_mut_ptr(), std::ptr::null_mut()), "mul_mat_q")?;
        if std::env::var_os("LLM170_MMQ_ARGS").is_some() {
            eprintln!("mmq_args ty={ty} n_in={n_in} n_out={n_out} t={t} grid=({},{},1) blk=(32,8) smem={smem} srow={} scol={} nrows={}",
                (n_out + 127) / 128, (t + 127) / 128, n_in / 256, n_out, n_out);
        }
        }
        Ok(())
    }


    /// 활성 양자화 — quantize_row_q8_ref 비트 미러. 출력은 xq 하나:
    /// [0..n/4) 워드 + [n/4..n/4+n/32) d 비트(u32 편승 — 저장 경로 단일화).
    /// 버퍼 크기 (n/4 + n/32)·4 바이트 필요.
    pub fn quant_q8(&self, x: *const u8, xq: *mut u8, n: usize) -> Result<(), String> {
        self.quant_q8_b(x, xq, n, n / 4 + n / 32 + n / 16, 1)
    }

    /// 배치 양자화 — t토큰 [t][n] → [t][xq_w 워드].
    pub fn quant_q8_b(&self, x: *const u8, xq: *mut u8, n: usize, xq_w: usize, t: usize) -> Result<(), String> {
        let nblk = n / 32;
        let mut x_p = x as *mut std::ffi::c_void;
        let mut xq_p = xq as *mut std::ffi::c_void;
        let mut n_a = n as i32;
        let mut xw = xq_w as i32;
        let mut args = vec![
            &mut x_p as *mut _ as *mut std::ffi::c_void,
            &mut xq_p as *mut _ as *mut std::ffi::c_void,
            &mut n_a as *mut _ as *mut std::ffi::c_void,
            &mut xw as *mut _ as *mut std::ffi::c_void,
        ];
        self.launch3("quant_q8", nblk.div_ceil(64) as u32, t as u32, 1, 64, &mut args)
    }
}

pub mod probes;
pub use probes::*;

#[cfg(test)]
mod micro_tests {
    use super::*;

    /// 런치 비용 분해 — 호스트 enqueue와 디바이스 디스패치를 가른다.
    /// 디코드 스텝은 ~3,222런치가 129ms를 쓰는데, 그 37µs/런치가 호스트인지
    /// 디바이스인지가 다음 수(융합 vs 배관)를 정한다.
    #[test]
    fn launch_cost_split() {
        let ctx = RawCtx::new().expect("ctx");
        let n = 1024i32;
        let a = ctx.scratch(4096).expect("a");
        let b = ctx.scratch(4096).expect("b");
        let launch = || {
            let (mut ap, mut bp) = (a as *mut std::ffi::c_void, b as *mut std::ffi::c_void);
            let (mut nn, mut rr) = (n, 1i32);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut ap) as *mut _ as *mut std::ffi::c_void,
                (&mut bp) as *mut _ as *mut std::ffi::c_void,
                (&mut nn) as *mut _ as *mut std::ffi::c_void,
                (&mut rr) as *mut _ as *mut std::ffi::c_void,
            ];
            ctx.launch3("bcast_rows", 8, 1, 1, 128, &mut args).unwrap();
        };
        for _ in 0..20 {
            launch();
        }
        ctx.sync().unwrap();
        const N: usize = 2000;
        let t0 = std::time::Instant::now();
        for _ in 0..N {
            launch();
        }
        let enq = t0.elapsed().as_secs_f64() * 1e3;
        ctx.sync().unwrap();
        let total = t0.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "# micro {N}런치: host-enqueue {enq:.1}ms ({:.2}µs/런치), sync포함 총 {total:.1}ms ({:.2}µs/런치)",
            enq / N as f64 * 1e3,
            total / N as f64 * 1e3
        );

        // 이상 커널 격리 — q4_l2_rows(d=2560, 1블록×32스레드). 디코드에서
        // 0.34ms/회로 관측된 그 커널이 정말 그런지, 아니면 측정 맥락 탓인지.
        let x = ctx.scratch(2560 * 4).expect("x");
        let launch_l2 = || {
            let mut xp = x as *mut std::ffi::c_void;
            let mut e = 1e-6f32;
            let mut d = 2560i32;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut xp) as *mut _ as *mut std::ffi::c_void,
                (&mut e) as *mut _ as *mut std::ffi::c_void,
                (&mut d) as *mut _ as *mut std::ffi::c_void,
            ];
            ctx.launch3("q4_l2_rows", 1, 1, 1, 32, &mut args).unwrap();
        };
        for _ in 0..20 {
            launch_l2();
        }
        ctx.sync().unwrap();
        let t1 = std::time::Instant::now();
        for _ in 0..N {
            launch_l2();
        }
        let enq2 = t1.elapsed().as_secs_f64() * 1e3;
        ctx.sync().unwrap();
        let total2 = t1.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "# micro q4_l2_rows(d=2560) {N}런치: host {enq2:.1}ms ({:.2}µs/런치), 총 {total2:.1}ms ({:.2}µs/런치)",
            enq2 / N as f64 * 1e3,
            total2 / N as f64 * 1e3
        );

        // GDN l2scale 시퀀스 격리 — 디코드에서 0.69ms/층으로 관측된 그 4커널
        // (split3 → l2_rows ×2 → scale). 실제로 그만큼 드는지, 맥락 탓인지 가른다.
        let conv_ch = 4096i32;
        let kv = 1024i32;
        let dstate = 128i32;
        let src = ctx.scratch(conv_ch as usize * 4).expect("src");
        let gq = ctx.scratch(kv as usize * 4).expect("gq");
        let gk = ctx.scratch(kv as usize * 4).expect("gk");
        let gv = ctx.scratch(kv as usize * 4).expect("gv");
        let launch_seq = || {
            let (mut sp, mut p0, mut p1, mut p2) = (
                src as *mut std::ffi::c_void,
                gq as *mut std::ffi::c_void,
                gk as *mut std::ffi::c_void,
                gv as *mut std::ffi::c_void,
            );
            let (mut n0, mut n1, mut n2) = (kv, kv, kv);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut sp) as *mut _ as *mut std::ffi::c_void,
                (&mut p0) as *mut _ as *mut std::ffi::c_void,
                (&mut p1) as *mut _ as *mut std::ffi::c_void,
                (&mut p2) as *mut _ as *mut std::ffi::c_void,
                (&mut n0) as *mut _ as *mut std::ffi::c_void,
                (&mut n1) as *mut _ as *mut std::ffi::c_void,
                (&mut n2) as *mut _ as *mut std::ffi::c_void,
            ];
            ctx.launch3("split3", ((kv * 3) as u32).div_ceil(256), 1, 1, 256, &mut args)
                .unwrap();
            for x in [gq as *mut std::ffi::c_void, gk as *mut std::ffi::c_void] {
                let (mut xp, mut e, mut d) = (x, 1e-6f32, dstate);
                let mut a2: Vec<*mut std::ffi::c_void> = vec![
                    (&mut xp) as *mut _ as *mut std::ffi::c_void,
                    (&mut e) as *mut _ as *mut std::ffi::c_void,
                    (&mut d) as *mut _ as *mut std::ffi::c_void,
                ];
                ctx.launch3("q4_l2_rows", 1, 1, 1, 32, &mut a2).unwrap();
            }
            let (mut gp, mut sc, mut nn) = (gq as *mut std::ffi::c_void, 0.088f32, kv);
            let mut a3: Vec<*mut std::ffi::c_void> = vec![
                (&mut gp) as *mut _ as *mut std::ffi::c_void,
                (&mut sc) as *mut _ as *mut std::ffi::c_void,
                (&mut nn) as *mut _ as *mut std::ffi::c_void,
            ];
            ctx.launch3("q4_scale", (kv as u32).div_ceil(128), 1, 1, 128, &mut a3).unwrap();
        };
        for _ in 0..20 {
            launch_seq();
        }
        ctx.sync().unwrap();
        const M: usize = 2000;
        let t2 = std::time::Instant::now();
        for _ in 0..M {
            launch_seq();
        }
        let enq3 = t2.elapsed().as_secs_f64() * 1e3;
        ctx.sync().unwrap();
        let total3 = t2.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "# micro GDN l2scale 시퀀스(4커널) {M}회: host {enq3:.1}ms ({:.2}µs/회), 총 {total3:.1}ms ({:.2}µs/회)",
            enq3 / M as f64 * 1e3,
            total3 / M as f64 * 1e3
        );

        // DRAM 대역 — 512MB 버퍼를 bcast_rows(읽기 n*4 + 쓰기 n*4)로 훑는다.
        // 디코드의 실 트래픽 추정(129.5ms × 대역)에 필요한 값.
        let big = 512usize * 1024 * 1024 / 4; // f32 개수
        let sb = ctx.scratch(big * 4).expect("src-big");
        let db = ctx.scratch(big * 4).expect("dst-big");
        let launch_big = || {
            let (mut sp, mut dp) = (sb as *mut std::ffi::c_void, db as *mut std::ffi::c_void);
            let (mut nn, mut rr) = (big as i32, 1i32);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut sp) as *mut _ as *mut std::ffi::c_void,
                (&mut dp) as *mut _ as *mut std::ffi::c_void,
                (&mut nn) as *mut _ as *mut std::ffi::c_void,
                (&mut rr) as *mut _ as *mut std::ffi::c_void,
            ];
            ctx.launch3("bcast_rows", (big as u32).div_ceil(128), 1, 1, 128, &mut args)
                .unwrap();
        };
        for _ in 0..3 {
            launch_big();
        }
        ctx.sync().unwrap();
        let t3 = std::time::Instant::now();
        launch_big();
        ctx.sync().unwrap();
        let dt = t3.elapsed().as_secs_f64();
        let gb = (big as f64 * 8.0) / 1e9; // 읽기+쓰기
        eprintln!(
            "# micro DRAM 대역(512MB x2): {:.2}ms → {:.0} GB/s",
            dt * 1e3,
            gb / dt
        );

        // 가중치 접근 패턴 프로브 — 144B 블록 스트라이드에서 실효 대역.
        // (q4_K 가중치를 스칼라로 읽는 현 커널들의 실제 패턴)
        let nb = 4_000_000usize; // 4M 블록 × 144B = 576MB
        let wbuf = ctx.scratch(nb * 144).expect("wbuf");
        let accb = ctx.scratch(4).expect("acc");
        for mode in [0i32, 1, 2, 3, 4, 5] {
            let launch_bw = || {
                let (mut wp, mut ap) = (wbuf as *mut std::ffi::c_void, accb as *mut std::ffi::c_void);
                let (mut n, mut m) = (nb as i32, mode);
                let mut args: Vec<*mut std::ffi::c_void> = vec![
                    (&mut wp) as *mut _ as *mut std::ffi::c_void,
                    (&mut ap) as *mut _ as *mut std::ffi::c_void,
                    (&mut n) as *mut _ as *mut std::ffi::c_void,
                    (&mut m) as *mut _ as *mut std::ffi::c_void,
                ];
                ctx.launch3("bw_strided", ((nb as u32).div_ceil(256)), 1, 1, 256, &mut args)
                    .unwrap();
            };
            launch_bw();
            ctx.sync().unwrap();
            let t = std::time::Instant::now();
            for _ in 0..4 {
                launch_bw();
            }
            ctx.sync().unwrap();
            let dt = t.elapsed().as_secs_f64();
            // 접근한 바이트 수(스칼라 4B/8B, 벡터 16B) 기준 실효 대역
            let bytes = match mode {
                0 => nb as f64 * 4.0,
                1 => nb as f64 * 4.0,
                2 => nb as f64 * 16.0,
                3 => nb as f64 * 2.0 * 8.0,   // f16 8회/스레드
                5 => nb as f64 * 4.0 * 8.0,   // 워드 8회/레인(완전 병합)
                _ => nb as f64 * 2.0 * 8.0,   // 워프 협동도 같은 바이트
            } * 4.0;
            let touched = nb as f64 * 144.0 * 4.0; // 실제로 건드린 메모리 범위
            eprintln!(
                "# micro bw_strided mode{mode}: {:.1}ms → 실사용 {:.1} GB/s (건드린 범위 기준 {:.1} GB/s)",
                dt * 1e3,
                bytes / dt / 1e9,
                touched / dt / 1e9
            );
        }

        // MoE 그룹 커널 + 비동기 d2h 왕복 격리 — 디코드 디바이스 경로가
        // +30ms/스텝을 보이는데, 그 추가분의 실체를 여기서 가른다.
        let ne = 512i32;
        let rows = 10i32;
        // 상한은 프로덕션과 같은 식(rows + 16*ne)을 쓴다 — 커널도 이 값을 인자로
        // 받고, rowexp·x 버퍼가 이 크기를 전제한다(2026-09-14 단일화).
        let bound = (rows + 16 * ne) as usize;
        let ids = ctx.scratch(rows as usize * 4).expect("ids");
        let offb = ctx.scratch((ne as usize + 2) * 4).expect("off");
        let permb = ctx.scratch(bound * 4).expect("perm");
        let invb = ctx.scratch(rows as usize * 4).expect("inv");
        let rexb = ctx.scratch(bound * 4).expect("rex"); // GEMM이 rows_pad까지 읽는다
        let ppb = ctx.scratch(bound * 4).expect("pp");
        let ipb = ctx.scratch(rows as usize * 4).expect("ip");
        let txb = ctx.scratch(bound).expect("tx");
        let rpb = ctx.scratch(4).expect("rp");
        let launch_g = || {
            let (mut a, mut b) = (ids as *mut std::ffi::c_void, offb as *mut std::ffi::c_void);
            let (mut c, mut d) = (permb as *mut std::ffi::c_void, invb as *mut std::ffi::c_void);
            let (mut e, mut f) = (rexb as *mut std::ffi::c_void, ppb as *mut std::ffi::c_void);
            let (mut g, mut h) = (ipb as *mut std::ffi::c_void, txb as *mut std::ffi::c_void);
            let mut i = rpb as *mut std::ffi::c_void;
            let (mut n_e, mut rws) = (ne, rows);
            let mut bnd = bound as i32;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut a) as *mut _ as *mut std::ffi::c_void,
                (&mut n_e) as *mut _ as *mut std::ffi::c_void,
                (&mut rws) as *mut _ as *mut std::ffi::c_void,
                (&mut b) as *mut _ as *mut std::ffi::c_void,
                (&mut c) as *mut _ as *mut std::ffi::c_void,
                (&mut d) as *mut _ as *mut std::ffi::c_void,
                (&mut e) as *mut _ as *mut std::ffi::c_void,
                (&mut f) as *mut _ as *mut std::ffi::c_void,
                (&mut g) as *mut _ as *mut std::ffi::c_void,
                (&mut h) as *mut _ as *mut std::ffi::c_void,
                (&mut i) as *mut _ as *mut std::ffi::c_void,
                (&mut bnd) as *mut _ as *mut std::ffi::c_void,
            ];
            ctx.launch3("q4_moe_group_t1", 1, 1, 1, 128, &mut args).unwrap();
        };
        for _ in 0..20 {
            launch_g();
        }
        ctx.sync().unwrap();
        let tg1 = std::time::Instant::now();
        for _ in 0..N {
            launch_g();
        }
        let enqg = tg1.elapsed().as_secs_f64() * 1e3;
        ctx.sync().unwrap();
        let totg = tg1.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "# micro group kernel {N}회: host {enqg:.1}ms ({:.2}µs), 총 {totg:.1}ms ({:.2}µs)",
            enqg / N as f64 * 1e3,
            totg / N as f64 * 1e3
        );
        // 비동기 d2h + 이벤트 대기 왕복(디바이스 경로가 층마다 하는 것)
        let tg2 = std::time::Instant::now();
        for _ in 0..N {
            let _ = ctx.d2h_issue((ne as usize + 1) * 4, offb as *const u8).unwrap();
            ctx.d2h_wait().unwrap();
        }
        let totd = tg2.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "# micro d2h_issue+wait {N}회: 총 {totd:.1}ms ({:.2}µs/회)",
            totd / N as f64 * 1e3
        );
    }
}
