//! RawCtx — 디바이스 컨텍스트(스트림·커널 레지스트리·버퍼 아레나·런치 래퍼).
//! plans/75 P2.2: mod.rs 에서 기계적 이동(내용 무변경).

pub(super) use crate::rawhip::KtraceEv;
pub(super) use crate::rawhip::ck;
pub(super) use crate::rawhip::env_on;
pub(super) use crate::rawhip::kernels;
pub(super) use crate::rawhip::{
    CO_J128, CO_MMQ, CO_MMQ2, CO_MMQ3, CO_MMQ8, CO_ODD, CO_QY, CO_V4, CO_W32F,
};
pub(super) use cubecl_hip_sys as hip;
pub(super) use std::collections::HashMap;
pub(super) use std::ffi::CString;

/// hipGraph 스트림 캡처·재생(plans/121 hip 스케줄링 — 런치 오버헤드 제거).
/// ROCm 10 시그니처(CUDA12형). 그래프 내부 노드는 실행 시점에 디바이스 상태를 읽는다.
pub mod hipgraph {
    use std::ffi::c_void;

    pub type Graph = *mut c_void;
    pub type GraphExec = *mut c_void;

    unsafe extern "C" {
        pub fn hipHostMalloc(ptr: *mut *mut c_void, size: usize, flags: u32) -> i32;
        pub fn hipHostFree(ptr: *mut c_void) -> i32;
        pub fn hipStreamBeginCapture(stream: *mut c_void, mode: i32) -> i32;
        pub fn hipStreamEndCapture(stream: *mut c_void, graph: *mut Graph) -> i32;
        pub fn hipGraphInstantiate(exec: *mut GraphExec, graph: Graph, flags: u64) -> i32;
        pub fn hipGraphLaunch(exec: GraphExec, stream: *mut c_void) -> i32;
        pub fn hipGraphExecDestroy(exec: GraphExec) -> i32;
        pub fn hipGraphDestroy(graph: Graph) -> i32;
        /// L2 플러시 측정(plans/131 S10) — 대형 memset이 L2 내용을 강제 교체.
        pub fn hipMemsetAsync(
            dst: *mut c_void,
            value: i32,
            size: usize,
            stream: *mut c_void,
        ) -> i32;
    }
}

impl RawCtx {
    /// L2 플러시용 대형 memset(스트림 순서) — 측정 프로토콜 전용(plans/131 S10).
    pub fn l2_flush(&self, dst: *mut u8, bytes: usize) -> Result<(), String> {
        let rc = unsafe {
            hipgraph::hipMemsetAsync(
                dst as *mut std::ffi::c_void,
                0,
                bytes,
                self.stream as *mut _,
            )
        };
        if rc != 0 {
            return Err(format!("l2_flush memset: {rc}"));
        }
        Ok(())
    }
}

pub struct RawCtx {
    pub(crate) fns: HashMap<&'static str, hip::hipFunction_t>,
    /// 로드된 코드오브젝트 패밀리 비트(CO_* 상수) — new() 완료 후 불변 (plans/78 R4).
    pub(crate) co_fam: std::sync::atomic::AtomicU16,
    pub(crate) scope: std::sync::atomic::AtomicU8,
    pub(crate) stream: hip::hipStream_t,
    /// plans/115 D: 프리필 그래프 캡처 중 — sync/d2h_wait/ktr_ev 건너뜀.
    pub(crate) capturing: std::sync::atomic::AtomicBool,
    pub(crate) stream2: hip::hipStream_t,
    /// 프리필 전용 스트림 페어 — 프레임 경로(launch3s + join2/side_wait_main)를
    /// 디코드와 겹쳐 돌리기 위한 별도 쌍(plans/74 np4 겹치기).
    pub(crate) stream3: hip::hipStream_t,
    pub(crate) stream4: hip::hipStream_t,
    /// AtomicBool: RawCtx 는 VL 경로에서 Arc 로 공유되므로 Cell 은 Sync 를 깬다.
    pub pre_pair: std::sync::atomic::AtomicBool,
    pub(crate) pre_ev: std::sync::Mutex<Option<hip::hipEvent_t>>,
    /// 크기별 스크래치 풀 — 해제 없는 재사용 (호출마다 신규 할당이
    /// 메모리 고갈→illegal address 유발, 2026-09-03 RCA).
    /// MMQ 전용 y 버퍼 (size, ptr) — 풀 충돌 격리.
    pub(crate) mmq_y: std::sync::Mutex<(usize, *mut u8)>,
    /// side-stream MMQ 전용 y 버퍼 (스트림 레이스 격리).
    pub(crate) mmq_y_s: std::sync::Mutex<(usize, *mut u8)>,
    /// q6→f16 전개 캐시 (w주소 → f16 버퍼).
    pub(crate) f16_cache: std::sync::Mutex<std::collections::HashMap<usize, *mut u8>>,
    /// hipMalloc 범위 등록부 — h2d 실패 시 목적지가 살아있는 할당 안인지 보고한다.
    /// 2026-09-14: 이것으로 "실패한 목적지가 정상 할당 내"임을 한 번에 확인해
    /// 원인을 커널 런치로 좁혔다(해제는 하지 않으므로 목록은 영구, 수백 개 수준).
    pub(crate) allocs: std::sync::Mutex<Vec<(usize, usize)>>,
    pub(crate) ar_cache: std::sync::Mutex<Option<(*mut u8, *mut u8, *mut u8, *mut u8)>>,
    /// q6 정준 재배열 캐시.
    /// f16 경로 xq 버퍼 (size, ptr).
    pub(crate) mmq_y2: std::sync::Mutex<(usize, *mut u8)>,
    pub(crate) scratch: std::sync::Mutex<HashMap<(usize, usize), Vec<*mut u8>>>,
    /// D2H 핀 스테이징 (필요시 성장, 해제 없음 — ADR-0014).
    /// pageable 버퍼로의 hipMemcpyAsync D2H는 슬로패스(1MB에 ~90ms,
    /// 2026-09-05 tg RCA) — 핀 버퍼 경유로 원소복사.
    pub(crate) pinned: std::sync::Mutex<(usize, *mut u8)>,
    /// d2h_issue(비동기) 전용 핀 — 동기 d2h와 버퍼를 공유하면 issue→wait 사이의
    /// 어느 동기 판독이든 내용을 덮어쓴다(plans/68 12차: b가 실수값으로 오염돼
    /// 폴백 행 수 1.04억 → HIP 700. t=1 프로덕션에도 잠복 경쟁이었다).
    pub(crate) pinned_a: std::sync::Mutex<(usize, *mut u8)>,
    /// H2D 스테이지 핀 2버퍼(107 W1.5-3) — 업로드 파이프라인 전용.
    pub(crate) pinned_stage: std::sync::Mutex<(usize, *mut u8)>,
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

/// 프리필 패밀리 핀 (plans/84 A) — step_batch 진입~종료 사이 true.
/// t 2..8 GEMV mt 변형·flash wk 게이트가 패밀리를 갈라 청크 불변을 깨뜨리므로
/// 핀 중에는 large-t 패밀리로 통일한다.
pub(crate) static PREFILL_PIN: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// 스펙 검증 배치 핀 (plans/110 W2) — frame_forward_verify 진입~종료 사이
/// true. t 2..8 Q8_0/f32 GEMV의 mt 변형은 t=1 디코드 커널(w16/w)과 축소
/// 순서가 달라 비트가 갈라진다 — 핀 중에는 행별 t=1 디스패치로 돌려
/// 검증 배치 == 순차 decode1 비트 동일을 보장한다(비용: 행당 무게 재독).
pub(crate) static VERIFY_ROW_PIN: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// 디스패처 모델 스코프 (plans/84 E1) — 전역 디스패처의 패밀리 기본값을
/// 모델별로 분리: 27B(qwen35) 게이트 타이를 뒤집는 저출력 warp GEMV를
/// Flash-Next(qwen4exp/q4acc)에서만 기본 적용하기 위함.
pub const SCOPE_QWEN35: u8 = 0;
pub const SCOPE_FLASHNEXT: u8 = 1;

impl RawCtx {
    /// 코드오브젝트 패밀리 로드 비트 질의 (plans/78 R4 — 전역 static 승계).
    /// 모델 스코프 지정 (q4acc 초기화 시 FlashNext).
    pub fn set_scope(&self, s: u8) {
        self.scope.store(s, std::sync::atomic::Ordering::Relaxed);
    }
    pub fn scope_is_flashnext(&self) -> bool {
        self.scope.load(std::sync::atomic::Ordering::Relaxed) == SCOPE_FLASHNEXT
    }

    pub fn co_loaded(&self, bit: u16) -> bool {
        self.co_fam.load(std::sync::atomic::Ordering::Relaxed) & bit != 0
    }

    /// hipRTC 즉시 컴파일 → 커널 레지스트리 채움 (plans/109 P9 — new() 분리).
    /// # Safety: RawCtx::new 초기화 경로(단일 스레드)에서만 호출.
    unsafe fn compile_rtc(
        fns: &mut HashMap<&'static str, hip::hipFunction_t>,
    ) -> Result<(), String> {
        // SAFETY: 초기화 경로(단일 스레드).
        unsafe {
            if llm170_diag::dump::opts().key("exl3_hipdbg") {
                eprintln!(
                    "  [rtcdbg] NAMES={} src_has_exl3={}",
                    kernels::NAMES.len(),
                    kernels::SRC.contains("exl3_had_in")
                );
            }
            let src = CString::new(kernels::SRC).unwrap();
            let mut prog: hip::hiprtcProgram = std::ptr::null_mut();
            let rs = hip::hiprtcCreateProgram(
                &mut prog,
                src.as_ptr(),
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            if rs != hip::hiprtcResult_HIPRTC_SUCCESS {
                return Err(format!("hiprtcCreateProgram: {rs:?}"));
            }
            // LLM170_HIP_INC: hipconfig 서브프로세스 없이 include 경로를 준다
            // (rocprof 등 서브프로세스를 방해하는 도구 아래에서 필요).
            let inc = match llm170_diag::flag::val("LLM170_HIP_INC") {
                Some(v) => v.to_string(),
                None => cubecl_hip_sys::get_hip_include_path().map_err(|e| e.to_string())?,
            };
            let o1 = CString::new(format!("-I{inc}")).unwrap();
            let o2 = CString::new("--std=c++17").unwrap();
            // -O3 확정(plans/130 원장): -O2 실험에서 exl3_gemv VGPR 112→43·2블록/CU
            // 점유가 개션됐으나 gemv 137→140ms·tg 5.96→5.89로 미세 역행 — -O3의
            // ILP가 실제 이득. 컴퓨트 바닥은 점유가 아니라 연산량(extract+decode
            // ALU 체인, ISA 원장: 정수 곱셈계 ~190명령)임이 확정.
            let o3 = CString::new("-O3").unwrap();
            // FMA 수축 차단 — CPU 비트계약 (a+=b*c 축약이 비트 불일치,
            // 2026-09-03 AR xor RCA)
            let o4 = CString::new("-ffp-contract=off").unwrap();
            let o5 = CString::new("-I/opt/rocm/include").unwrap();
            // exp_cr 기본을 디바이스 __expf로 (f64 호너 제거, 2026-09-12).
            // 효과: tg +1.0%, pp +0.55%, judge 16/19 -> 17/19 (llama와 더 가까움).
            // FASTEXP exp2 근사 경로(측정 승격: tg +1.0%, pp +0.55%, judge
            // 16/19→17/19 — plans/115 env 정리로 상시 고정).
            let ofast = CString::new("-DLLM170_FASTEXP").unwrap();
            let fastexp = true;
            let mut opts = vec![
                o1.as_ptr(),
                o2.as_ptr(),
                o3.as_ptr(),
                o4.as_ptr(),
                o5.as_ptr(),
            ];
            if fastexp {
                opts.push(ofast.as_ptr());
            }
            let rs = hip::hiprtcCompileProgram(prog, opts.len() as i32, opts.as_mut_ptr());
            if rs != hip::hiprtcResult_HIPRTC_SUCCESS {
                let mut sz = 0usize;
                let _ = hip::hiprtcGetProgramLogSize(prog, &mut sz);
                let mut buf = vec![0i8; sz.max(1)];
                let _ = hip::hiprtcGetProgramLog(prog, buf.as_mut_ptr());
                let log = String::from_utf8_lossy(std::slice::from_raw_parts(
                    buf.as_ptr() as *const u8,
                    sz,
                ));
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
            let mut module: hip::hipModule_t = std::ptr::null_mut();
            ck(
                hip::hipModuleLoadData(&mut module, code.as_ptr() as *const _),
                "ModuleLoadData",
            )?;
            for name in kernels::NAMES {
                let cname = CString::new(*name).unwrap();
                let mut f: hip::hipFunction_t = std::ptr::null_mut();
                ck(
                    hip::hipModuleGetFunction(&mut f, module, cname.as_ptr()),
                    "GetFunction",
                )?;
                fns.insert(*name, f);
            }
        }
        Ok(())
    }

    /// 오프라인 CO 패밀리 병행 로드 → fns 병합 + 패밀리 비트.
    /// # Safety: compile_rtc와 동일 초기화 경로.
    unsafe fn load_co_families(
        fns: &mut HashMap<&'static str, hip::hipFunction_t>,
    ) -> Result<u16, String> {
        let mut fam_bits = 0u16;
        // SAFETY: 초기화 경로(단일 스레드).
        unsafe {
            // 오프라인 코드오브젝트 병행 로드 (wave32 커널 등).
            // 기본: 바이너리 임베딩(crates/.../co/*.co, gfx1151 빌드).
            // LLM170_CO*_PATH가 있으면 그 파일이 우선 (커널 실험 오버라이드).
            {
                let slots: &[(u16, &str, &[u8], &[&str])] = &[
                    (
                        CO_V4,
                        "LLM170_CO2_PATH",
                        include_bytes!("../co/v4all.co"),
                        &[
                            "gemm_q5k_v4",
                            "gemm_q4k_v4",
                            "gemm_xs_v4",
                            "gemm_q5k_wm",
                            "gemm_q4k_wm",
                            "gemm_q6k_wm",
                            "gemm_xs_wm",
                        ],
                    ),
                    (
                        CO_ODD,
                        "LLM170_CO3_PATH",
                        include_bytes!("../co/odd_all.co"),
                        &["gemm_nl_v4", "gemm_q3k_v4", "gemm_iq3s_v4"],
                    ),
                    (
                        CO_MMQ,
                        "LLM170_CO4_PATH",
                        include_bytes!("../co/mmq.co"),
                        &[
                            "mmq_quant_y",
                            "mmq_quant_y_d4",
                            "_ZL9mul_mat_qIL9ggml_type12ELi128ELb0EEvPKcPKiS4_S4_PfS5_PKf15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_",
                            "_ZL9mul_mat_qIL9ggml_type13ELi128ELb0EEvPKcPKiS4_S4_PfS5_PKf15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_",
                            "_ZL9mul_mat_qIL9ggml_type14ELi128ELb0EEvPKcPKiS4_S4_PfS5_PKf15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_",
                            "_ZL9mul_mat_qIL9ggml_type23ELi128ELb0EEvPKcPKiS4_S4_PfS5_PKf15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_",
                        ],
                    ),
                    (
                        CO_MMQ2,
                        "LLM170_CO5_PATH",
                        include_bytes!("../co/mmq2.co"),
                        &["gemm_f16_v4"],
                    ),
                    (
                        CO_MMQ3,
                        "LLM170_CO6_PATH",
                        include_bytes!("../co/mmq3.co"),
                        &[
                            "_ZL9mul_mat_qIL9ggml_type23ELi128ELb0EEvPKcPKiS4_S4_PfS5_PKf15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_",
                        ],
                    ),
                    (
                        CO_MMQ8,
                        "LLM170_CO7_PATH",
                        include_bytes!("../co/mmq8.co"),
                        &[
                            "_ZL9mul_mat_qIL9ggml_type8ELi128ELb0EEvPKcPKiS4_S4_PfS5_PKf15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_",
                        ],
                    ),
                    (
                        CO_QY,
                        "LLM170_CO8_PATH",
                        include_bytes!("../co/quanty_new.co"),
                        &[
                            "_ZL17quantize_mmq_q8_1IL18mmq_q8_1_ds_layout0ELb0EEvPKfPKiPvllllliii",
                            "_ZL17quantize_mmq_q8_1IL18mmq_q8_1_ds_layout1ELb0EEvPKfPKiPvllllliii",
                            "_ZL17quantize_mmq_q8_1IL18mmq_q8_1_ds_layout2ELb0EEvPKfPKiPvllllliii",
                        ],
                    ),
                    (
                        CO_J128,
                        "LLM170_CO_PATH",
                        include_bytes!("../co/w32b.co"),
                        &[
                            "gemm_q5k_j128",
                            "gemm_q4k_j128",
                            "gemm_q6k_j128",
                            "gemm_xs_j128",
                            "gemm_q8_j128",
                        ],
                    ),
                    (
                        CO_W32F,
                        "LLM170_CO12_PATH",
                        include_bytes!("../co/w32f.co"),
                        &[
                            "_Z18exl3_gemv_j128_w32ILi4ELb0ELi0ELi1EEvPK6__halfPKtPviii",
                            "_Z18exl3_gemv_j128_w32ILi4ELb0ELi1ELi1EEvPK6__halfPKtPviii",
                            "_Z18exl3_gemv_j128_w32ILi4ELb1ELi0ELi0EEvPK6__halfPKtPviii",
                            "_Z18exl3_gemv_j128_w32ILi4ELb1ELi0ELi1EEvPK6__halfPKtPviii",
                            "_Z18exl3_gemv_j128_w32ILi4ELb1ELi1ELi0EEvPK6__halfPKtPviii",
                            "_Z18exl3_gemv_j128_w32ILi4ELb1ELi1ELi1EEvPK6__halfPKtPviii",
                        ],
                    ),
                ];
                for (bit, env_key, embedded, names) in slots {
                    let bytes: Vec<u8> = match llm170_diag::flag::val(env_key) {
                        Some(p) => {
                            std::fs::read(p).map_err(|e| format!("{env_key} 읽기({p}): {e}"))?
                        }
                        None => embedded.to_vec(),
                    };
                    let mut m: hip::hipModule_t = std::ptr::null_mut();
                    ck(
                        hip::hipModuleLoadData(&mut m, bytes.as_ptr() as *const _),
                        &format!("{env_key} ModuleLoadData"),
                    )?;
                    let mut loaded = 0u16;
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
                    fam_bits |= loaded;
                }
            }
        }
        Ok(fam_bits)
    }

    pub fn new() -> Result<Self, String> {
        unsafe {
            ck(hip::hipSetDevice(0), "hipSetDevice")?;
            let _ = hip::hipSetDeviceFlags(hip::hipDeviceScheduleSpin);
            let mut fns = HashMap::new();
            Self::compile_rtc(&mut fns)?;
            if llm170_diag::dump::opts().key("exl3_hipdbg") {
                eprintln!(
                    "  [rtcdbg] post-compile fns={} exl3: {}",
                    fns.len(),
                    fns.keys().filter(|k| k.contains("exl3")).count()
                );
            }
            let fam_bits = Self::load_co_families(&mut fns)?;
            let mut stream: hip::hipStream_t = std::ptr::null_mut();
            ck(hip::hipStreamCreate(&mut stream), "StreamCreate")?;
            let mut stream2: hip::hipStream_t = std::ptr::null_mut();
            ck(hip::hipStreamCreate(&mut stream2), "StreamCreate2")?;
            let mut stream3: hip::hipStream_t = std::ptr::null_mut();
            ck(hip::hipStreamCreate(&mut stream3), "StreamCreate3")?;
            let mut stream4: hip::hipStream_t = std::ptr::null_mut();
            ck(hip::hipStreamCreate(&mut stream4), "StreamCreate4")?;
            Ok(RawCtx {
                scope: std::sync::atomic::AtomicU8::new(SCOPE_QWEN35),
                fns,
                co_fam: std::sync::atomic::AtomicU16::new(fam_bits),
                stream,
                capturing: std::sync::atomic::AtomicBool::new(false),
                stream2,
                stream3,
                stream4,
                pre_pair: std::sync::atomic::AtomicBool::new(false),
                pre_ev: std::sync::Mutex::new(None),
                mmq_y: std::sync::Mutex::new((0, std::ptr::null_mut())),
                mmq_y_s: std::sync::Mutex::new((0, std::ptr::null_mut())),
                f16_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
                allocs: std::sync::Mutex::new(Vec::new()),
                ar_cache: std::sync::Mutex::new(None),
                mmq_y2: std::sync::Mutex::new((0, std::ptr::null_mut())),
                scratch: std::sync::Mutex::new(HashMap::new()),
                pinned_a: std::sync::Mutex::new((0, std::ptr::null_mut())),
                pinned: std::sync::Mutex::new((0, std::ptr::null_mut())),
                pinned_stage: std::sync::Mutex::new((0, std::ptr::null_mut())),
            })
        }
    }

    /// 스크래시 획득 — 같은 크기는 항상 슬롯 0 재사용 (스트림 순서가
    /// 이전 사용 완료를 보장 — 단일 스트림). 호출마다 신규 할당은
    /// 메모리 고갈→illegal address (2026-09-03 RCA).
    pub fn scratch(&self, bytes: usize) -> Result<*mut u8, String> {
        let mut sc = self.scratch.lock().map_err(|e| e.to_string())?;
        let v = sc
            .entry((
                self.pre_pair.load(std::sync::atomic::Ordering::Relaxed) as usize,
                bytes,
            ))
            .or_default();
        if v.is_empty() {
            let p = self.alloc(bytes)?;
            v.push(p);
        }
        Ok(v[0])
    }

    /// 현재 메인 스트림 — 프리필 페어면 stream3.
    #[inline]
    pub fn cur_stream(&self) -> hip::hipStream_t {
        if self.pre_pair.load(std::sync::atomic::Ordering::Relaxed) {
            self.stream3
        } else {
            self.stream
        }
    }

    /// 현재 사이드 스트림(launch3s / join2 / side_wait_main 대상).
    #[inline]
    pub fn cur_side(&self) -> hip::hipStream_t {
        if self.pre_pair.load(std::sync::atomic::Ordering::Relaxed) {
            self.stream4
        } else {
            self.stream2
        }
    }

    /// 프리필 완료 이벤트 기록(현재 사이드) / 비블로킹 확인 / 메인 합류.
    pub fn pre_mark(&self) -> Result<(), String> {
        let mut g = self.pre_ev.lock().map_err(|e| e.to_string())?;
        unsafe {
            if g.is_none() {
                let mut ev: hip::hipEvent_t = std::ptr::null_mut();
                ck(hip::hipEventCreateWithFlags(&mut ev, 0), "pre-ev")?;
                *g = Some(ev);
            }
            ck(
                hip::hipEventRecord(g.unwrap(), self.cur_side()),
                "pre-ev-rec",
            )?;
        }
        Ok(())
    }
    pub fn pre_ready(&self) -> bool {
        let Ok(g) = self.pre_ev.lock() else {
            return false;
        };
        match *g {
            None => false,
            Some(ev) => unsafe { hip::hipEventQuery(ev) == hip::hipError_t_hipSuccess },
        }
    }
    pub fn pre_join(&self) -> Result<(), String> {
        let g = self.pre_ev.lock().map_err(|e| e.to_string())?;
        if let Some(ev) = *g {
            unsafe {
                ck(hip::hipStreamWaitEvent(self.stream, ev, 0), "pre-join")?;
            }
        }
        Ok(())
    }

    pub fn sync(&self) -> Result<(), String> {
        if self.capturing.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(()); // 캡처 중 sync는 불법 — skip(plans/115 D)
        }
        unsafe { ck(hip::hipStreamSynchronize(self.stream), "sync") }
    }

    /// 사이드 스트림 동기화 — staged_upload 이중버퍼 종료 지점용
    /// (107 W1.5-3: 호스트 스테이지 반납 전 복사 완료 보장).
    pub fn sync2(&self) -> Result<(), String> {
        unsafe { ck(hip::hipStreamSynchronize(self.stream2), "sync2") }
    }

    /// 사이드 스트림 복사 후 이벤트 기록 (이중버퍼 재사용 판정용,
    /// 107 W1.5-3). 호스트는 ev_sync로 해당 복사만 선별 대기한다.
    ///
    /// # Safety
    /// `ev`는 유효한 이벤트 핸들이어야 한다(이 모듈 생성분).
    pub unsafe fn ev_record_s2(&self, ev: hip::hipEvent_t) -> Result<(), String> {
        unsafe { ck(hip::hipEventRecord(ev, self.stream2), "evRecS2") }
    }

    /// 이벤트 생성/파기/호스트 대기 — 스테이지 반납 직전 확인용.
    pub fn ev_create() -> Result<hip::hipEvent_t, String> {
        unsafe {
            let mut ev: hip::hipEvent_t = std::ptr::null_mut();
            ck(hip::hipEventCreateWithFlags(&mut ev, 0), "evCreate")?;
            Ok(ev)
        }
    }

    ///
    /// # Safety
    /// `ev`는 유효한 이벤트 핸들이어야 한다(이 모듈 생성분, 중복 파기 금지).
    pub unsafe fn ev_destroy(ev: hip::hipEvent_t) -> Result<(), String> {
        unsafe { ck(hip::hipEventDestroy(ev), "evDestroy") }
    }
}

mod alloc;
mod copy;
mod gemm;
pub mod launch;
// 마커 gcu
// 마커 pin2
