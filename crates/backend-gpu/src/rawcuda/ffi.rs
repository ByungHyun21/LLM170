
use std::ffi::{c_char, c_int, c_uchar, c_uint, c_void};
use std::sync::OnceLock;

pub type CUdevice = c_int;
pub type CUdeviceptr = u64;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUgraph = *mut c_void;
pub type CUgraphExec = *mut c_void;
pub type CUevent = *mut c_void;

pub type CUresult = c_uint;
pub const CUDA_SUCCESS: CUresult = 0;

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
        _ => {
            if let Ok(drv) = Driver::get() {
                let mut p: *const c_char = std::ptr::null();
                let rc = unsafe { (drv.get_error_string)(r, &mut p) };
                if rc == CUDA_SUCCESS && !p.is_null() {
                    let s = unsafe { std::ffi::CStr::from_ptr(p) };
                    return format!("{} (CUresult={r})", s.to_string_lossy());
                }
            }
            format!("CUresult={r}")
        }
    }
}

pub type CuInitFn = unsafe extern "system" fn(flags: c_uint) -> CUresult;
pub type CuDeviceGetCountFn = unsafe extern "system" fn(count: *mut c_int) -> CUresult;
pub type CuDeviceGetFn = unsafe extern "system" fn(dev: *mut CUdevice, ordinal: c_int) -> CUresult;

pub const CU_DEVICE_ATTRIBUTE_CLOCK_RATE: c_int = 13;
pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
pub const CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE: c_int = 36;
pub const CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: c_int = 37;
pub const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: c_int = 97;
pub type CuDeviceGetNameFn =
    unsafe extern "system" fn(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;
pub type CuDeviceGetAttributeFn =
    unsafe extern "system" fn(pi: *mut c_int, attrib: c_int, dev: CUdevice) -> CUresult;
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
pub type CuMemcpyHtoDAsyncFn = unsafe extern "system" fn(
    dst: CUdeviceptr,
    src: *const c_void,
    bytes: usize,
    stream: CUstream,
) -> CUresult;
pub type CuMemcpyDtoDAsyncFn = unsafe extern "system" fn(
    dst: CUdeviceptr,
    src: CUdeviceptr,
    bytes: usize,
    stream: CUstream,
) -> CUresult;
pub type CuStreamCreateFn =
    unsafe extern "system" fn(ph_stream: *mut CUstream, flags: c_uint) -> CUresult;
pub type CuStreamWaitEventFn =
    unsafe extern "system" fn(s: CUstream, ev: CUevent, flags: c_uint) -> CUresult;
pub type CuMemHostAllocFn =
    unsafe extern "system" fn(pp: *mut *mut c_void, bytes: usize, flags: c_uint) -> CUresult;
pub type CuMemFreeHostFn = unsafe extern "system" fn(p: *mut c_void) -> CUresult;
pub type CuMemHostGetDevicePointerFn = unsafe extern "system" fn(
    pdev: *mut CUdeviceptr,
    phost: *mut c_void,
    flags: c_uint,
) -> CUresult;
pub type CuMemcpyDtoHAsyncFn = unsafe extern "system" fn(
    dst: *mut c_void,
    src: CUdeviceptr,
    bytes: usize,
    stream: CUstream,
) -> CUresult;
pub type CuStreamBeginCaptureFn =
    unsafe extern "system" fn(stream: CUstream, mode: c_uint) -> CUresult;
pub type CuStreamEndCaptureFn =
    unsafe extern "system" fn(stream: CUstream, ph_graph: *mut CUgraph) -> CUresult;
pub type CuGraphInstantiateWithFlagsFn =
    unsafe extern "system" fn(ph_exec: *mut CUgraphExec, graph: CUgraph, flags: u64) -> CUresult;
pub type CuGraphLaunchFn =
    unsafe extern "system" fn(exec: CUgraphExec, stream: CUstream) -> CUresult;
pub type CuGraphUploadFn =
    unsafe extern "system" fn(exec: CUgraphExec, stream: CUstream) -> CUresult;
pub type CuGraphExecDestroyFn = unsafe extern "system" fn(exec: CUgraphExec) -> CUresult;
pub type CuGraphDestroyFn = unsafe extern "system" fn(graph: CUgraph) -> CUresult;

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
pub type CuFuncSetAttributeFn =
    unsafe extern "system" fn(f: CUfunction, attrib: c_uint, value: c_int) -> CUresult;

pub type CuEventCreateFn = unsafe extern "system" fn(ev: *mut CUevent, flags: c_uint) -> CUresult;
pub type CuEventRecordFn = unsafe extern "system" fn(ev: CUevent, s: CUstream) -> CUresult;
pub type CuEventElapsedTimeFn =
    unsafe extern "system" fn(ms: *mut f32, a: CUevent, b: CUevent) -> CUresult;
pub type CuEventDestroyFn = unsafe extern "system" fn(ev: CUevent) -> CUresult;

pub type CuEventSynchronizeFn = unsafe extern "system" fn(ev: CUevent) -> CUresult;

pub type CuMemGetInfoFn =
    unsafe extern "system" fn(free: *mut usize, total: *mut usize) -> CUresult;

pub type CuMemHostRegisterFn =
    unsafe extern "system" fn(p: *mut c_void, bytes: usize, flags: c_uint) -> CUresult;
pub type CuMemHostUnregisterFn = unsafe extern "system" fn(p: *mut c_void) -> CUresult;

pub type CuGetErrorStringFn =
    unsafe extern "system" fn(r: CUresult, p_str: *mut *const c_char) -> CUresult;

pub type CuMemsetD8AsyncFn = unsafe extern "system" fn(
    dst: CUdeviceptr,
    uc: c_uchar,
    n: usize,
    stream: CUstream,
) -> CUresult;


pub type CUmemGenericAllocationHandle = u64;

pub const CU_MEM_ALLOCATION_TYPE_PINNED: c_int = 0x1;
pub const CU_MEM_HANDLE_TYPE_NONE: c_int = 0x0;
pub const CU_MEM_LOCATION_TYPE_DEVICE: c_int = 0x1;
pub const CU_MEM_ACCESS_FLAGS_PROT_READWRITE: c_uint = 0x3;
pub const CU_MEM_ALLOC_GRANULARITY_MINIMUM: c_uint = 0x0;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUmemLocation {
    pub ty: c_int,
    pub id: c_int,
}

#[repr(C)]
pub struct CUmemAccessDesc {
    pub location: CUmemLocation,
    pub flags: c_uint,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUmemAllocationPropAllocFlags {
    pub compression_type: u8,
    pub gpu_direct_rdma_capable: u8,
    pub usage: u16,
    pub reserved: [u8; 4],
}

#[repr(C)]
pub struct CUmemAllocationProp {
    pub ty: c_int,
    pub requested_handle_types: c_int,
    pub location: CUmemLocation,
    pub win32_handle_meta_data: *mut c_void,
    pub alloc_flags: CUmemAllocationPropAllocFlags,
}

const _: () = assert!(std::mem::size_of::<CUmemLocation>() == 8);
const _: () = assert!(std::mem::size_of::<CUmemAccessDesc>() == 12);
const _: () = assert!(std::mem::size_of::<CUmemAllocationProp>() == 32);

pub type CuMemCreateFn = unsafe extern "system" fn(
    handle: *mut CUmemGenericAllocationHandle,
    bytes: usize,
    prop: *const CUmemAllocationProp,
    flags: u64,
) -> CUresult;
pub type CuMemAddressReserveFn = unsafe extern "system" fn(
    ptr: *mut CUdeviceptr,
    bytes: usize,
    alignment: usize,
    addr: CUdeviceptr,
    flags: u64,
) -> CUresult;
pub type CuMemMapFn = unsafe extern "system" fn(
    ptr: CUdeviceptr,
    bytes: usize,
    offset: usize,
    handle: CUmemGenericAllocationHandle,
    flags: u64,
) -> CUresult;
pub type CuMemSetAccessFn = unsafe extern "system" fn(
    ptr: CUdeviceptr,
    bytes: usize,
    desc: *const CUmemAccessDesc,
    count: usize,
) -> CUresult;
pub type CuMemUnmapFn = unsafe extern "system" fn(ptr: CUdeviceptr, bytes: usize) -> CUresult;
pub type CuMemReleaseFn =
    unsafe extern "system" fn(handle: CUmemGenericAllocationHandle) -> CUresult;
pub type CuMemAddressFreeFn = unsafe extern "system" fn(ptr: CUdeviceptr, bytes: usize) -> CUresult;
pub type CuMemGetAllocationGranularityFn = unsafe extern "system" fn(
    granularity: *mut usize,
    prop: *const CUmemAllocationProp,
    option: c_uint,
) -> CUresult;

pub(crate) struct Driver {
    pub init: CuInitFn,
    pub device_get_count: CuDeviceGetCountFn,
    pub device_get: CuDeviceGetFn,
    pub device_get_name: CuDeviceGetNameFn,
    pub device_get_attribute: CuDeviceGetAttributeFn,
    pub device_primary_ctx_retain: CuDevicePrimaryCtxRetainFn,
    pub ctx_set_current: CuCtxSetCurrentFn,
    pub ctx_get_current: CuCtxGetCurrentFn,
    pub module_load_data: CuModuleLoadDataFn,
    pub module_get_function: CuModuleGetFunctionFn,
    pub mem_alloc: CuMemAllocFn,
    pub memcpy_htod: CuMemcpyHtoDFn,
    pub memcpy_htod_async: CuMemcpyHtoDAsyncFn,
    pub memcpy_dtoh: CuMemcpyDtoHFn,
    pub memcpy_dtod_async: CuMemcpyDtoDAsyncFn,
    pub launch_kernel: CuLaunchKernelFn,
    pub stream_synchronize: CuStreamSynchronizeFn,
    pub stream_create: CuStreamCreateFn,
    pub stream_wait_event: CuStreamWaitEventFn,
    pub mem_host_alloc: CuMemHostAllocFn,
    pub mem_free_host: CuMemFreeHostFn,
    pub mem_host_get_device_pointer: CuMemHostGetDevicePointerFn,
    pub memcpy_dtoh_async: CuMemcpyDtoHAsyncFn,
    pub stream_begin_capture: CuStreamBeginCaptureFn,
    pub stream_end_capture: CuStreamEndCaptureFn,
    pub graph_instantiate: CuGraphInstantiateWithFlagsFn,
    pub graph_launch: CuGraphLaunchFn,
    pub graph_upload: CuGraphUploadFn,
    pub graph_exec_destroy: CuGraphExecDestroyFn,
    pub graph_destroy: CuGraphDestroyFn,
    pub mem_free: CuMemFreeFn,

    pub mem_get_info: CuMemGetInfoFn,
    pub func_set_attribute: CuFuncSetAttributeFn,

    pub get_error_string: CuGetErrorStringFn,
    pub host_register: CuMemHostRegisterFn,
    pub host_unregister: CuMemHostUnregisterFn,
    pub memset_d8_async: CuMemsetD8AsyncFn,

    pub mem_create: CuMemCreateFn,
    pub mem_address_reserve: CuMemAddressReserveFn,
    pub mem_map: CuMemMapFn,
    pub mem_set_access: CuMemSetAccessFn,
    pub mem_unmap: CuMemUnmapFn,
    pub mem_release: CuMemReleaseFn,
    pub mem_address_free: CuMemAddressFreeFn,
    pub mem_get_allocation_granularity: CuMemGetAllocationGranularityFn,

    pub event_create: CuEventCreateFn,
    pub event_record: CuEventRecordFn,
    pub event_elapsed: CuEventElapsedTimeFn,
    pub event_destroy: CuEventDestroyFn,
    pub event_synchronize: CuEventSynchronizeFn,
}

impl Driver {
    pub fn get() -> Result<&'static Driver, String> {
        static D: OnceLock<Result<Driver, String>> = OnceLock::new();
        D.get_or_init(|| unsafe { Self::load() })
            .as_ref()
            .map_err(|e| format!("rawcuda: CUDA 드라이버 로드 실패: {e}"))
    }

    unsafe fn load() -> Result<Driver, String> {
        let lib = loader::open()?;
        macro_rules! sym {
            ($name:literal) => {
                loader::sym(lib, $name)?
            };
        }
        unsafe {
            Ok(Driver {
                init: std::mem::transmute::<*mut c_void, CuInitFn>(sym!("cuInit")),
                device_get_count: std::mem::transmute::<*mut c_void, CuDeviceGetCountFn>(sym!(
                    "cuDeviceGetCount"
                )),
                device_get: std::mem::transmute::<*mut c_void, CuDeviceGetFn>(sym!("cuDeviceGet")),
                device_get_attribute: std::mem::transmute::<*mut c_void, CuDeviceGetAttributeFn>(
                    sym!("cuDeviceGetAttribute"),
                ),
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
                memcpy_htod_async: std::mem::transmute::<*mut c_void, CuMemcpyHtoDAsyncFn>(sym!(
                    "cuMemcpyHtoDAsync_v2"
                )),
                memcpy_dtoh: std::mem::transmute::<*mut c_void, CuMemcpyDtoHFn>(sym!(
                    "cuMemcpyDtoH_v2"
                )),
                memcpy_dtod_async: std::mem::transmute::<*mut c_void, CuMemcpyDtoDAsyncFn>(sym!(
                    "cuMemcpyDtoDAsync_v2"
                )),
                launch_kernel: std::mem::transmute::<*mut c_void, CuLaunchKernelFn>(sym!(
                    "cuLaunchKernel"
                )),
                stream_synchronize: std::mem::transmute::<*mut c_void, CuStreamSynchronizeFn>(
                    sym!("cuStreamSynchronize"),
                ),
                stream_create: std::mem::transmute::<*mut c_void, CuStreamCreateFn>(sym!(
                    "cuStreamCreate"
                )),
                stream_wait_event: std::mem::transmute::<*mut c_void, CuStreamWaitEventFn>(sym!(
                    "cuStreamWaitEvent"
                )),
                mem_host_alloc: std::mem::transmute::<*mut c_void, CuMemHostAllocFn>(sym!(
                    "cuMemHostAlloc"
                )),
                mem_free_host: std::mem::transmute::<*mut c_void, CuMemFreeHostFn>(sym!(
                    "cuMemFreeHost"
                )),
                mem_host_get_device_pointer: std::mem::transmute::<
                    *mut c_void,
                    CuMemHostGetDevicePointerFn,
                >(
                    loader::sym(lib, "cuMemHostGetDevicePointer_v2")
                        .or_else(|_| loader::sym(lib, "cuMemHostGetDevicePointer"))?,
                ),
                memcpy_dtoh_async: std::mem::transmute::<*mut c_void, CuMemcpyDtoHAsyncFn>(sym!(
                    "cuMemcpyDtoHAsync_v2"
                )),
                stream_begin_capture: std::mem::transmute::<*mut c_void, CuStreamBeginCaptureFn>(
                    sym!("cuStreamBeginCapture"),
                ),
                stream_end_capture: std::mem::transmute::<*mut c_void, CuStreamEndCaptureFn>(sym!(
                    "cuStreamEndCapture"
                )),
                graph_instantiate: std::mem::transmute::<*mut c_void, CuGraphInstantiateWithFlagsFn>(
                    sym!("cuGraphInstantiateWithFlags"),
                ),
                graph_launch: std::mem::transmute::<*mut c_void, CuGraphLaunchFn>(sym!(
                    "cuGraphLaunch"
                )),
                graph_upload: std::mem::transmute::<*mut c_void, CuGraphUploadFn>(sym!(
                    "cuGraphUpload"
                )),
                graph_exec_destroy: std::mem::transmute::<*mut c_void, CuGraphExecDestroyFn>(sym!(
                    "cuGraphExecDestroy"
                )),
                graph_destroy: std::mem::transmute::<*mut c_void, CuGraphDestroyFn>(sym!(
                    "cuGraphDestroy"
                )),
                mem_free: std::mem::transmute::<*mut c_void, CuMemFreeFn>(sym!("cuMemFree_v2")),
                func_set_attribute: std::mem::transmute::<*mut c_void, CuFuncSetAttributeFn>(sym!(
                    "cuFuncSetAttribute"
                )),
                event_create: std::mem::transmute::<*mut c_void, CuEventCreateFn>(sym!(
                    "cuEventCreate"
                )),
                event_record: std::mem::transmute::<*mut c_void, CuEventRecordFn>(sym!(
                    "cuEventRecord"
                )),
                event_elapsed: std::mem::transmute::<*mut c_void, CuEventElapsedTimeFn>(sym!(
                    "cuEventElapsedTime"
                )),
                event_destroy: std::mem::transmute::<*mut c_void, CuEventDestroyFn>(sym!(
                    "cuEventDestroy"
                )),
                event_synchronize: std::mem::transmute::<*mut c_void, CuEventSynchronizeFn>(sym!(
                    "cuEventSynchronize"
                )),

                mem_get_info: std::mem::transmute::<*mut c_void, CuMemGetInfoFn>(sym!(
                    "cuMemGetInfo_v2"
                )),
                get_error_string: std::mem::transmute::<*mut c_void, CuGetErrorStringFn>(sym!(
                    "cuGetErrorString"
                )),
                host_register: std::mem::transmute::<*mut c_void, CuMemHostRegisterFn>(sym!(
                    "cuMemHostRegister_v2"
                )),
                host_unregister: std::mem::transmute::<*mut c_void, CuMemHostUnregisterFn>(sym!(
                    "cuMemHostUnregister"
                )),
                memset_d8_async: std::mem::transmute::<*mut c_void, CuMemsetD8AsyncFn>(sym!(
                    "cuMemsetD8Async"
                )),
                mem_create: std::mem::transmute::<*mut c_void, CuMemCreateFn>(sym!("cuMemCreate")),
                mem_address_reserve: std::mem::transmute::<*mut c_void, CuMemAddressReserveFn>(
                    sym!("cuMemAddressReserve"),
                ),
                mem_map: std::mem::transmute::<*mut c_void, CuMemMapFn>(sym!("cuMemMap")),
                mem_set_access: std::mem::transmute::<*mut c_void, CuMemSetAccessFn>(sym!(
                    "cuMemSetAccess"
                )),
                mem_unmap: std::mem::transmute::<*mut c_void, CuMemUnmapFn>(sym!("cuMemUnmap")),
                mem_release: std::mem::transmute::<*mut c_void, CuMemReleaseFn>(sym!(
                    "cuMemRelease"
                )),
                mem_address_free: std::mem::transmute::<*mut c_void, CuMemAddressFreeFn>(sym!(
                    "cuMemAddressFree"
                )),
                mem_get_allocation_granularity: std::mem::transmute::<
                    *mut c_void,
                    CuMemGetAllocationGranularityFn,
                >(sym!(
                    "cuMemGetAllocationGranularity"
                )),
            })
        }
    }
}

mod loader {
    use std::ffi::c_void;

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
        let p = unsafe { GetProcAddress(lib, cstr(&b)) };
        if p.is_null() {
            Err(format!("심볼 없음: {name} (nvcuda.dll 버전 불일치?)"))
        } else {
            Ok(p)
        }
    }

    #[cfg(unix)]
    const RTLD_NOW: std::ffi::c_int = 2;

    #[cfg(unix)]
    unsafe extern "C" {
        fn dlopen(filename: *const std::ffi::c_char, flags: std::ffi::c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const std::ffi::c_char) -> *mut c_void;
    }

    #[cfg(unix)]
    pub fn open() -> Result<*mut c_void, String> {
        for cand in [&b"libcuda.so.1\0"[..], &b"libcuda.so\0"[..]] {
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
        let p = unsafe { dlsym(lib, cstr(&b) as *const std::ffi::c_char) };
        if p.is_null() {
            Err(format!("심볼 없음: {name}"))
        } else {
            Ok(p)
        }
    }
}
