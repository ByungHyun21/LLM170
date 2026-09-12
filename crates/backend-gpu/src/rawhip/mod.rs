//! 원시 HIP 실행기 — cubecl을 거치지 않는 직접 경로 (2026-09-03 재작성).
//! hipRTC로 임베디드 HIP C++ 소스를 컴파일하고 hipModuleLaunchKernel로
//! 실행. 버퍼는 영속 아레나(해제 없음, ADR-0014 동일 규칙). 커널 산술은
//! core 미러(dot_row_w4a8_*_lane)와 동일 연산열 — to_bits 검증 게이트.

use cubecl_hip_sys as hip;
use std::collections::HashMap;
use std::ffi::CString;

pub mod decode;
pub mod kernels;
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
pub fn ktrace_dump() -> String {
    let mut g = KTRACE.lock().unwrap();
    // ktrace_on 없이 호출되면(스펙 경로 등) 빈 문자열 — 과거 unwrap 패닉
    let Some(slot) = g.as_mut() else { return String::new() };
    let evs = std::mem::take(slot);
    let mut out = String::new();
    // 쌍 결합: 연속 동일 (name, gy) 두 이벤트가 start/end
    let mut sums: std::collections::HashMap<(&str, u32), (f64, u32)> = std::collections::HashMap::new();
    let mut total = 0.0f64;
    let mut gaps = 0.0f64;
    let mut i = 0usize;
    unsafe {
        while i + 1 < evs.len() {
            if evs[i].0 == evs[i+1].0 && evs[i].2 == evs[i+1].2 {
                let mut ms = 0f32;
                if hip::hipEventElapsedTime(&mut ms, evs[i].1 as *mut _, evs[i+1].1 as *mut _) == hip::hipError_t_hipSuccess {
                    let ent = sums.entry((evs[i].0, evs[i].2)).or_insert((0.0, 0));
                    ent.0 += ms as f64; ent.1 += 1;
                    total += ms as f64;
                }
                i += 2;
            } else {
                // 쌍이 아님 → 직전 end에서 이 start까지 갭
                if i > 0 {
                    let mut ms = 0f32;
                    if hip::hipEventElapsedTime(&mut ms, evs[i-1].1 as *mut _, evs[i].1 as *mut _) == hip::hipError_t_hipSuccess {
                        gaps += ms as f64;
                    }
                }
                i += 1;
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
            let inc = cubecl_hip_sys::get_hip_include_path().map_err(|e| e.to_string())?;
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
            ar_cache: std::sync::Mutex::new(None),
            mmq_y_cache: std::sync::Mutex::new((u64::MAX, 0, 0)),
            canon_q6: std::sync::Mutex::new(std::collections::HashMap::new()),
            mmq_y2: std::sync::Mutex::new((0, std::ptr::null_mut())), scratch: std::sync::Mutex::new(HashMap::new()), cursors: std::sync::Mutex::new(HashMap::new()), pinned: std::sync::Mutex::new((0, std::ptr::null_mut())) })
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
        Ok(p as *mut u8)
    }

    /// 사이드 스트림 비동기 h2d — 메인 스트림 작업과 중첩시킨 뒤 join2로 합류.
    pub fn h2d_async_s(&self, dst: *mut u8, src: &[u8]) -> Result<(), String> {
        unsafe {
            ck(hip::hipMemcpyAsync(dst as *mut _, src.as_ptr() as *const _, src.len(),
                hip::hipMemcpyKind_hipMemcpyHostToDevice, self.stream2), "h2d-async-s")
        }
    }
    pub fn h2d(&self, dst: *mut u8, src: &[u8]) -> Result<(), String> {
        unsafe {
            ck(hip::hipMemcpyAsync(dst as *mut _, src.as_ptr() as *const _, src.len(), hip::hipMemcpyKind_hipMemcpyHostToDevice, self.stream), "h2d")?;
            self.sync()
        }
    }

    pub fn d2h(&self, dst: &mut [u8], src: *const u8) -> Result<(), String> {
        unsafe {
            // pageable 직행은 슬로패스 — 핀 스테이징 경유 (2026-09-05 tg RCA:
            // logits 1MB D2H가 92ms → 핀 경유 시 <1ms 예상)
            let need = dst.len();
            let mut pin = self.pinned.lock().map_err(|e| e.to_string())?;
            if pin.0 < need {
                let mut p: *mut std::os::raw::c_void = std::ptr::null_mut();
                ck(hip::hipMallocHost(&mut p, need), "hipMallocHost")?;
                *pin = (need, p as *mut u8);
            }
            let buf = pin.1;
            ck(hip::hipMemcpyAsync(buf as *mut _, src as *const _, need, hip::hipMemcpyKind_hipMemcpyDeviceToHost, self.stream), "d2h-pin")?;
            ck(hip::hipStreamSynchronize(self.stream), "d2h-sync")?;
            std::ptr::copy_nonoverlapping(buf, dst.as_mut_ptr(), need);
            Ok(())
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
    ) -> Result<(), String> {

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
            ck(hip::hipModuleLaunchKernel(f, gx, gy, gz, block, 1, 1, smem, self.stream, args.as_mut_ptr(), std::ptr::null_mut()), "launch3_dyn")?;
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
            ck(hip::hipModuleLaunchKernel(f, gx, gy, gz, block, 1, 1, 0, self.stream2, args.as_mut_ptr(), std::ptr::null_mut()), "launch3s")?;
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
    pub fn gemv_q8_out_v2(&self, xq: *const u8, w: *const u8, ty: u32, n_in: usize, n_out: usize, out: *mut u8, xq_w: usize, t: usize) -> Result<(), String> {
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
        let gz = n_out.div_ceil(65535) as u32;
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
        self.launch3(kern, t as u32, gy, gz, 64, &mut args_v)?;
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
        let kern = match ty {
            12 => "gemm_q4k4",
            13 => "gemm_q5k4",
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
        self.launch3(kern, 1, gy, gz, 64, &mut args)
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
        let gz = n_out.div_ceil(65535) as u32;
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
            tt: t.min(128) as i32,
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
                let blocks = n_in / 256;
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
        // y: f32 → 우리 xq (quant_q8) — y_f32 에서 직접
        let xq_w = n_in/4 + n_in/32 + n_in/16;
        let mut xq = self.mmq_y2.lock().map_err(|e| e.to_string())?;
        let xq_p = if xq.0 < xq_w * t {
            let p = self.alloc(xq_w * t * 4)? as *mut u8;
            *xq = (xq_w * t, p);
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
            let mut b7 = t as i32;
            let mut args2 = vec![&mut b1 as *mut _ as *mut _, &mut b2 as *mut _ as *mut _, &mut b3 as *mut _ as *mut _, &mut b4 as *mut _ as *mut _, &mut b5 as *mut _ as *mut _, &mut b6 as *mut _ as *mut _, &mut b7 as *mut _ as *mut _];
            // z-그리드 사분면 CO: 단일 런치 (tt=min(t,128), gz=사분면)
            {
              let mut z1 = xq_p as *mut std::ffi::c_void;
              let mut z3 = out as *mut std::ffi::c_void;
              let mut z7 = t.min(128) as i32;
              let mut az: Vec<*mut std::ffi::c_void> = vec![&mut z1 as *mut _ as *mut _, &mut b2 as *mut _ as *mut _, &mut z3 as *mut _ as *mut _,
                  &mut b4 as *mut _ as *mut _, &mut b5 as *mut _ as *mut _, &mut b6 as *mut _ as *mut _, &mut z7 as *mut _ as *mut _];
              ck(hip::hipModuleLaunchKernel(fm, ((n_out + 127) / 128) as u32, 1, t.div_ceil(128) as u32, 256, 1, 1, 0, self.stream, az.as_mut_ptr(), std::ptr::null_mut()), "gemm_f16_v4")?;
            }
        if std::env::var_os("LLM170_DEQ_DUMP").is_some() {
            self.sync().ok();
            let _ = std::fs::write("/tmp/deq_wf16.f16", unsafe { std::slice::from_raw_parts(wf16 as *const u8, n_out * n_in * 2) });
            let _ = std::fs::write("/tmp/deq_w.bin", unsafe { std::slice::from_raw_parts(w as *const u8, n_out.min(1) * (n_in/256) * 210 + 210) });
            let _ = std::fs::write("/tmp/deq_xq.bin", unsafe { std::slice::from_raw_parts(xq_p as *const u8, xq_w * t * 4) });
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
        let fq = *fns.get(if matches!(ty, 14 | 23) { "mmq_quant_y_d4" } else { "mmq_quant_y" })
            .ok_or("mmq quant 없음")?;
        let j: usize = if std::env::var_os("LLM170_MMQ64").is_some() { 64 } else { 128 };
        let sym = match ty {
            12 => { let js = if j == 64 { "64" } else { "128" }; format!("_ZL9mul_mat_qIL9ggml_type12ELi{}ELb0EEvPKcPKiS4_S4_PfS5_PKf15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_", js) }
            13 => { let js = if j == 64 { "64" } else { "128" }; format!("_ZL9mul_mat_qIL9ggml_type13ELi{}ELb0EEvPKcPKiS4_S4_PfS5_PKf15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_", js) }
            14 => { let js = if j == 64 { "64" } else { "128" }; format!("_ZL9mul_mat_qIL9ggml_type14ELi{}ELb0EEvPKcPKiS4_S4_PfS5_PKf15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_", js) }
            23 => { let js = if j == 64 { "64" } else { "128" }; format!("_ZL9mul_mat_qIL9ggml_type23ELi{}ELb0EEvPKcPKiS4_S4_PfS5_PKf15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_", js) }
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
            let mut sc = self.mmq_y.lock().map_err(|e| e.to_string())?;
            if sc.0 < (n_in / 128) * t * 144 {
                if !sc.1.is_null() { unsafe { hip::hipFree(sc.1 as *mut _) }; }
                sc.1 = self.alloc((n_in / 128) * t * 144)? as *mut u8;
                sc.0 = (n_in / 128) * t * 144;
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
            if let Ok(mut c) = self.mmq_y_cache.lock() { *c = (c.0, y_key.0, y_key.1); }
        }
        fn fd3(d: u32) -> [u32; 3] {
            let mut l = 0u32;
            while l < 32 && (1u32 << l) < d { l += 1; }
            let mp = ((((1u64) << 32) * (((1u64) << l) - d as u64)) / d as u64 + 1) as u32;
            [mp, l, d]
        }
        let j: usize = if std::env::var_os("LLM170_MMQ64").is_some() { 64 } else { 128 };
        let nbk = (n_in / 256) as u32;
        let mut bpn = fd3(nbk);
        let mut one = fd3(1);
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
        let mut p_srow = (n_in / 256) as i32;
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
                one.as_mut_ptr() as *mut _,
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
        let fq = *fns.get(if matches!(ty, 14 | 23) { "mmq_quant_y_d4" } else { "mmq_quant_y" })
            .ok_or("mmq quant 없음")?;
        let j: usize = if std::env::var_os("LLM170_MMQ64").is_some() { 64 } else { 128 };
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
            let mut sc = self.mmq_y_s.lock().map_err(|e| e.to_string())?;
            if sc.0 < (n_in / 128) * t * 144 {
                if !sc.1.is_null() { unsafe { hip::hipFree(sc.1 as *mut _) }; }
                sc.1 = self.alloc((n_in / 128) * t * 144)? as *mut u8;
                sc.0 = (n_in / 128) * t * 144;
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
        let nbk = (n_in / 256) as u32;
        let mut bpn = fd3(nbk);
        let mut one = fd3(1);
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
        let mut p_srow = (n_in / 256) as i32;
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
                one.as_mut_ptr() as *mut _,
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
