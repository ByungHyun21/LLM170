//! CUDA 드라이버 API 수동 바인딩 — nvcuda.dll / libcuda.so.1 런타임 해석
//! (plans/124 2026-10-04).
//!
//! 계약: 새 크레이트 금지 → cuda-sys 등 도입 없이 시그니처를 손작성한다.
//! 링크 방침: 정적 extern 링크(`#[link(name = "cuda")]`)는 AMD 전용 기기의
//! 기본 빌드를 깨뜨린다(libcuda.so 없음 — 기본 빌드 녹색 계약). rawhip의
//! cubecl-hip-sys(의존 libc+regex, 링크 의존 없음)와 동일 정책으로 런타임
//! 해석만 한다(Windows: LoadLibraryA, unix: dlopen — libc은 기존 의존).
//! 시그니처는 CUDA 13.4 nvcuda.dll 수출표와 대조 확인(2026-10-04 dumpbin).
//! 계 alloc/memcpy/free는 _v2 심볼에 바인딩 — CUDA 13.4에서 평명(非_v2)
//! 수출은 프라이머리 유지 컨텍스트를 인식하지 못해 CUresult=201로 실패함을
//! 실측 확인(평명 cuMemAlloc=201 vs cuMemAlloc_v2=0, 동일 유효 컨텍스트).
//! CUDA 11+ 헤더에서 cuMemAlloc 등은 _v2의 매크로 별칭 — _v2가 정규 경로.

use std::ffi::{c_char, c_int, c_uint, c_void};
use std::sync::OnceLock;

/// CUdevice — 디바이스 서수 식별자.
pub type CUdevice = c_int;
/// CUdeviceptr — 64비트 디바이스 포인터(부호 없는 64비트 규격).
pub type CUdeviceptr = u64;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;

/// CUresult — 드라이버 API 반환 코드(0 = CUDA_SUCCESS).
pub type CUresult = c_uint;
pub const CUDA_SUCCESS: CUresult = 0;

/// 확실한 코드만 명명 — 그 외는 숫자로 보고(오표기 위험 차단).
pub fn err_text(r: CUresult) -> String {
    match r {
        CUDA_SUCCESS => "성공".into(),
        1 => "잘못된 값(CUDA_ERROR_INVALID_VALUE)".into(),
        2 => "디바이스 메모리 부족(CUDA_ERROR_OUT_OF_MEMORY)".into(),
        3 => "드라이버 미초기화(CUDA_ERROR_NOT_INITIALIZED)".into(),
        4 => "드라이버 해제됨(CUDA_ERROR_DEINITIALIZED)".into(),
        100 => "CUDA 디바이스 없음(CUDA_ERROR_NO_DEVICE)".into(),
        101 => "잘못된 디바이스 서수(CUDA_ERROR_INVALID_DEVICE)".into(),
        400 => "잘못된 핸들(CUDA_ERROR_INVALID_HANDLE)".into(),
        _ => format!("CUresult={r}"),
    }
}

// ── 드라이버 함수 시그니처(손작성 — 헤더 미러, 2026-10-04) ──
pub type CuInitFn = unsafe extern "system" fn(flags: c_uint) -> CUresult;
pub type CuDeviceGetCountFn = unsafe extern "system" fn(count: *mut c_int) -> CUresult;
pub type CuDeviceGetFn = unsafe extern "system" fn(dev: *mut CUdevice, ordinal: c_int) -> CUresult;
pub type CuDeviceGetNameFn =
    unsafe extern "system" fn(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;
pub type CuDevicePrimaryCtxRetainFn =
    unsafe extern "system" fn(ctx: *mut CUcontext, dev: CUdevice) -> CUresult;
pub type CuCtxSetCurrentFn = unsafe extern "system" fn(ctx: CUcontext) -> CUresult;
pub type CuCtxGetCurrentFn = unsafe extern "system" fn(ctx: *mut CUcontext) -> CUresult;
pub type CuModuleLoadDataFn =
    unsafe extern "system" fn(module: *mut CUmodule, image: *const c_void) -> CUresult;
pub type CuModuleGetFunctionFn = unsafe extern "system" fn(
    f: *mut CUfunction,
    module: CUmodule,
    name: *const c_char,
) -> CUresult;
pub type CuMemAllocFn = unsafe extern "system" fn(dptr: *mut CUdeviceptr, bytes: usize) -> CUresult;
pub type CuMemcpyHtoDFn =
    unsafe extern "system" fn(dst: CUdeviceptr, src: *const c_void, bytes: usize) -> CUresult;
pub type CuMemcpyDtoHFn =
    unsafe extern "system" fn(dst: *mut c_void, src: CUdeviceptr, bytes: usize) -> CUresult;
/// cuMemcpyDtoD_v2 — 디바이스 내 복사. 호스트 왕복 없는 디바이스 체인의
/// 필수 요소(plans/cuda-port.md S10): GEMV 입출력을 상주 버퍼 사이에
/// 직접 옮긴다.
pub type CuMemcpyDtoDFn =
    unsafe extern "system" fn(dst: CUdeviceptr, src: CUdeviceptr, bytes: usize) -> CUresult;
pub type CuLaunchKernelFn = unsafe extern "system" fn(
    f: CUfunction,
    gx: c_uint,
    gy: c_uint,
    gz: c_uint,
    bx: c_uint,
    by: c_uint,
    bz: c_uint,
    shared: c_uint,
    stream: CUstream,
    params: *mut *mut c_void,
    extra: *mut c_void,
) -> CUresult;
pub type CuStreamSynchronizeFn = unsafe extern "system" fn(stream: CUstream) -> CUresult;
pub type CuMemFreeFn = unsafe extern "system" fn(dptr: CUdeviceptr) -> CUresult;
/// cuFuncSetAttribute — 함수 속성 설정(G5 scan 동적 공유 61,828B의
/// opt-in: CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES=8).
pub type CuFuncSetAttributeFn =
    unsafe extern "system" fn(f: CUfunction, attrib: c_uint, value: c_int) -> CUresult;

/// cuMemGetInfo_v2 — 가용/전체 디바이스 메모리(바이트). B6(plans/cuda-models.md
/// §4): 리소스 가드의 CUDA 런타임 VRAM 조회용(현재 컨텍스트 기준).
pub type CuMemGetInfoFn =
    unsafe extern "system" fn(free: *mut usize, total: *mut usize) -> CUresult;

/// 해석 완료된 드라이버 함수표 — 전부 순수 함수 포인터(Send+Sync 자동).
pub(crate) struct Driver {
    pub init: CuInitFn,
    pub device_get_count: CuDeviceGetCountFn,
    pub device_get: CuDeviceGetFn,
    pub device_get_name: CuDeviceGetNameFn,
    pub device_primary_ctx_retain: CuDevicePrimaryCtxRetainFn,
    pub ctx_set_current: CuCtxSetCurrentFn,
    pub ctx_get_current: CuCtxGetCurrentFn,
    pub module_load_data: CuModuleLoadDataFn,
    pub module_get_function: CuModuleGetFunctionFn,
    pub mem_alloc: CuMemAllocFn,
    pub memcpy_htod: CuMemcpyHtoDFn,
    pub memcpy_dtoh: CuMemcpyDtoHFn,
    pub memcpy_dtod: CuMemcpyDtoDFn,
    pub launch_kernel: CuLaunchKernelFn,
    pub stream_synchronize: CuStreamSynchronizeFn,
    pub mem_free: CuMemFreeFn,

    /// cuMemGetInfo_v2 — 가드 VRAM 프로브(B6).
    pub mem_get_info: CuMemGetInfoFn,
    pub func_set_attribute: CuFuncSetAttributeFn,
}

impl Driver {
    /// 프로세스 최초 1회 해석(OnceLock 캐시 — 실패 사유도 재보고).
    pub fn get() -> Result<&'static Driver, String> {
        static D: OnceLock<Result<Driver, String>> = OnceLock::new();
        // SAFETY: OnceLock 초기화 경로 — 단일 스레드 진입 보장, 함수표는 이후 불변.
        D.get_or_init(|| unsafe { Self::load() })
            .as_ref()
            .map_err(|e| format!("rawcuda: CUDA 드라이버 로드 실패: {e}"))
    }

    /// 라이브러리 핸들은 해제하지 않는다(프로세스 수명 — ADR-0014 영속 규칙과
    /// 동일 취급, dlclose/FreeLibrary 금지).
    ///
    /// # Safety
    /// 초기화 경로(OnceLock)에서만 호출 — 함수 포인터 transmute 포함.
    unsafe fn load() -> Result<Driver, String> {
        let lib = loader::open()?;
        macro_rules! sym {
            ($name:literal) => {
                loader::sym(lib, $name)?
            };
        }
        // SAFETY: 수출표 대조 완료 심볼 — 시그니처는 cuda.h 미러(위 손작성).
        unsafe {
            Ok(Driver {
                init: std::mem::transmute::<*mut c_void, CuInitFn>(sym!("cuInit")),
                device_get_count: std::mem::transmute::<*mut c_void, CuDeviceGetCountFn>(sym!(
                    "cuDeviceGetCount"
                )),
                device_get: std::mem::transmute::<*mut c_void, CuDeviceGetFn>(sym!("cuDeviceGet")),
                device_get_name: std::mem::transmute::<*mut c_void, CuDeviceGetNameFn>(sym!(
                    "cuDeviceGetName"
                )),
                device_primary_ctx_retain: std::mem::transmute::<
                    *mut c_void,
                    CuDevicePrimaryCtxRetainFn,
                >(sym!("cuDevicePrimaryCtxRetain")),
                ctx_set_current: std::mem::transmute::<*mut c_void, CuCtxSetCurrentFn>(sym!(
                    "cuCtxSetCurrent"
                )),
                ctx_get_current: std::mem::transmute::<*mut c_void, CuCtxGetCurrentFn>(sym!(
                    "cuCtxGetCurrent"
                )),
                module_load_data: std::mem::transmute::<*mut c_void, CuModuleLoadDataFn>(sym!(
                    "cuModuleLoadData"
                )),
                module_get_function: std::mem::transmute::<*mut c_void, CuModuleGetFunctionFn>(
                    sym!("cuModuleGetFunction"),
                ),
                mem_alloc: std::mem::transmute::<*mut c_void, CuMemAllocFn>(sym!("cuMemAlloc_v2")),
                memcpy_htod: std::mem::transmute::<*mut c_void, CuMemcpyHtoDFn>(sym!(
                    "cuMemcpyHtoD_v2"
                )),
                memcpy_dtoh: std::mem::transmute::<*mut c_void, CuMemcpyDtoHFn>(sym!(
                    "cuMemcpyDtoH_v2"
                )),
                memcpy_dtod: std::mem::transmute::<*mut c_void, CuMemcpyDtoDFn>(sym!(
                    "cuMemcpyDtoD_v2"
                )),
                launch_kernel: std::mem::transmute::<*mut c_void, CuLaunchKernelFn>(sym!(
                    "cuLaunchKernel"
                )),
                stream_synchronize: std::mem::transmute::<*mut c_void, CuStreamSynchronizeFn>(
                    sym!("cuStreamSynchronize"),
                ),
                mem_free: std::mem::transmute::<*mut c_void, CuMemFreeFn>(sym!("cuMemFree_v2")),
                func_set_attribute: std::mem::transmute::<*mut c_void, CuFuncSetAttributeFn>(sym!(
                    "cuFuncSetAttribute"
                )),

                mem_get_info: std::mem::transmute::<*mut c_void, CuMemGetInfoFn>(sym!(
                    "cuMemGetInfo_v2"
                )),
            })
        }
    }
}

/// 플랫폼 로더 — Windows kernel32 / unix dlopen(libc 기존 의존).
mod loader {
    use std::ffi::c_void;

    /// NUL 종료 심볼명 편의 — b"cuXxx\0" 인자용.
    pub fn cstr(name: &[u8]) -> *const u8 {
        debug_assert!(name.ends_with(b"\0"));
        name.as_ptr()
    }

    #[cfg(windows)]
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryA(name: *const u8) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
    }

    #[cfg(windows)]
    pub fn open() -> Result<*mut c_void, String> {
        // SAFETY: nvcuda.dll 로드 — 부작용은 모듈 참조수 증가뿐(해제 없음).
        let h = unsafe { LoadLibraryA(cstr(b"nvcuda.dll\0")) };
        if h.is_null() {
            Err("nvcuda.dll 로드 실패 — NVIDIA 드라이버 미설치?".into())
        } else {
            Ok(h)
        }
    }

    #[cfg(windows)]
    pub fn sym(lib: *mut c_void, name: &str) -> Result<*mut c_void, String> {
        let mut b = name.as_bytes().to_vec();
        b.push(0);
        // SAFETY: lib는 open()이 돌려준 유효 핸들, b는 NUL 종료.
        let p = unsafe { GetProcAddress(lib, cstr(&b)) };
        if p.is_null() {
            Err(format!("심볼 없음: {name} (nvcuda.dll 버전 불일치?)"))
        } else {
            Ok(p)
        }
    }

    // unix 수동 dl 바인딩 — std 외 크레이트 금지 계약(plans/124)으로 libc 크레이트를
    // 쓰지 않는다. glibc 2.34+는 dlopen/dlsym이 libc 내장(구분 libdl 폐지)이고
    // std 타깃은 libc에 링크되므로 extern "C" 선언만으로 해석된다(2026-10-07
    // 리눅스 포팅). RTLD_NOW=2는 glibc·musl 공통값.
    #[cfg(unix)]
    const RTLD_NOW: std::ffi::c_int = 2;

    #[cfg(unix)]
    unsafe extern "C" {
        fn dlopen(filename: *const std::ffi::c_char, flags: std::ffi::c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const std::ffi::c_char) -> *mut c_void;
    }

    #[cfg(unix)]
    pub fn open() -> Result<*mut c_void, String> {
        // 바이트 문자열 길이가 달라 [u8; N] 배열 단일형이 불가 — 슬라이스로.
        for cand in [&b"libcuda.so.1\0"[..], &b"libcuda.so\0"[..]] {
            // SAFETY: 드라이버 라이브러리 로드 — 참조수 증가만(해제 없음).
            let h = unsafe { dlopen(cstr(cand) as *const std::ffi::c_char, RTLD_NOW) };
            if !h.is_null() {
                return Ok(h);
            }
        }
        Err("libcuda.so 로드 실패 — NVIDIA 드라이버 미설치?".into())
    }

    #[cfg(unix)]
    pub fn sym(lib: *mut c_void, name: &str) -> Result<*mut c_void, String> {
        let mut b = name.as_bytes().to_vec();
        b.push(0);
        // SAFETY: lib는 open()이 돌려준 유효 핸들, b는 NUL 종료.
        let p = unsafe { dlsym(lib, cstr(&b) as *const std::ffi::c_char) };
        if p.is_null() {
            Err(format!("심볼 없음: {name}"))
        } else {
            Ok(p)
        }
    }
}
