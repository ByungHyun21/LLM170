
use crate::rawcuda::ffi::{self, CUDA_SUCCESS, CUdeviceptr, CUfunction, CUstream};
use std::collections::HashMap;

pub const PROF_CATS: [&str; 16] = [
    "misc", "norm", "gemv", "gemm", "gdn", "attn", "ew", "head", "moe", "scan", "l2perm", "conv",
    "gate", "grp", "marlin", "h2f",
];

pub static HOST_LAUNCH_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static HOST_LAUNCH_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn prof_cat(name: &str) -> usize {
    if name.contains("h2f") {
        15
    } else if name.contains("marlin") {
        14
    } else if name.contains("scan") {
        9
    } else if name.contains("l2perm") {
        10
    } else if name.contains("conv") {
        11
    } else if name.contains("gate") {
        12
    } else if name.contains("grp") {
        13
    } else if name.contains("moe") {
        8
    } else if name.contains("norm") {
        1
    } else if name.contains("gemv") {
        2
    } else if name.contains("gemm") {
        3
    } else if name.contains("gdn") {
        4
    } else if name.contains("attn") {
        5
    } else if name.contains("ew") || name.contains("axpy") || name.contains("shared_add") {
        6
    } else if name.contains("head") {
        7
    } else {
        0
    }
}

pub struct Prof {
    pub on: bool,
    pub capturing: std::cell::Cell<bool>,
    evs: std::cell::RefCell<Vec<ffi::CUevent>>,
    cats: std::cell::RefCell<Vec<usize>>,
}

pub struct CudaCtx {
    drv: &'static ffi::Driver,
    pub device: ffi::CUdevice,
    pub device_name: String,
    ctx: ffi::CUcontext,
    pub stream: CUstream,
    pub stream2: CUstream,
    pub prof: Prof,
    modules: HashMap<&'static str, ffi::CUmodule>,
    fns: HashMap<&'static str, CUfunction>,
    copy_h2d: std::cell::Cell<(u64, u64, u64)>,
    copy_d2h: std::cell::Cell<(u64, u64, u64)>,
    copy_d2d: std::cell::Cell<(u64, u64, u64)>,
    vmm: std::cell::RefCell<HashMap<CUdeviceptr, (ffi::CUmemGenericAllocationHandle, usize)>>,
    allocs: std::cell::RefCell<Vec<CUdeviceptr>>,
}

pub struct CtxGuard {
    drv: &'static ffi::Driver,
    prev: ffi::CUcontext,
}

impl Drop for CtxGuard {
    fn drop(&mut self) {
        let r = unsafe { (self.drv.ctx_set_current)(self.prev) };
        if r != CUDA_SUCCESS {
            eprintln!("rawcuda: ctx guard 복원 실패: {}", ffi::err_text(r));
        }
    }
}

mod caps;
mod graph;
mod mem;
mod prof;

pub use self::caps::{DeviceCaps, VramSampler, cuda_mem_free, cuda_mem_free_dev};

impl CudaCtx {
    pub fn new_dev(ordinal: i32) -> Result<Self, String> {
        let drv = ffi::Driver::get()?;
        unsafe {
            let r = (drv.init)(0);
            if r != CUDA_SUCCESS {
                return Err(format!("rawcuda: cuInit: {}", ffi::err_text(r)));
            }
            let mut n: ffi::CUdevice = 0;
            let r = (drv.device_get_count)(&mut n);
            if r != CUDA_SUCCESS {
                return Err(format!("rawcuda: cuDeviceGetCount: {}", ffi::err_text(r)));
            }
            if n == 0 {
                return Err("rawcuda: CUDA 디바이스 0개".into());
            }
            let mut dev: ffi::CUdevice = 0;
            let r = (drv.device_get)(&mut dev, ordinal);
            if r != CUDA_SUCCESS {
                return Err(format!(
                    "rawcuda: cuDeviceGet({ordinal}): {}",
                    ffi::err_text(r)
                ));
            }
            let mut namebuf = [0i8; 256];
            let r = (drv.device_get_name)(namebuf.as_mut_ptr(), namebuf.len() as _, dev);
            if r != CUDA_SUCCESS {
                return Err(format!("rawcuda: cuDeviceGetName: {}", ffi::err_text(r)));
            }
            let device_name = {
                let len = namebuf
                    .iter()
                    .position(|&c| c == 0)
                    .unwrap_or(namebuf.len());
                String::from_utf8_lossy(std::slice::from_raw_parts(
                    namebuf.as_ptr() as *const u8,
                    len,
                ))
                .into_owned()
            };
            let mut ctx: ffi::CUcontext = std::ptr::null_mut();
            let r = (drv.device_primary_ctx_retain)(&mut ctx, dev);
            if r != CUDA_SUCCESS {
                return Err(format!(
                    "rawcuda: cuDevicePrimaryCtxRetain: {}",
                    ffi::err_text(r)
                ));
            }
            let r = (drv.ctx_set_current)(ctx);
            if r != CUDA_SUCCESS {
                return Err(format!("rawcuda: cuCtxSetCurrent: {}", ffi::err_text(r)));
            }
            Ok(CudaCtx {
                drv,
                prof: Prof {
                    on: llm170_diag::flag::on_nonzero("LLM170_TIME"),
                    capturing: std::cell::Cell::new(false),
                    evs: std::cell::RefCell::new(Vec::new()),
                    cats: std::cell::RefCell::new(Vec::new()),
                },
                device: dev,
                device_name,
                ctx,
                stream: std::ptr::null_mut(),
                stream2: std::ptr::null_mut(),
                modules: HashMap::new(),
                fns: HashMap::new(),
                copy_h2d: std::cell::Cell::new((0, 0, 0)),
                copy_d2h: std::cell::Cell::new((0, 0, 0)),
                copy_d2d: std::cell::Cell::new((0, 0, 0)),
                vmm: std::cell::RefCell::new(HashMap::new()),
                allocs: std::cell::RefCell::new(Vec::new()),
            })
        }
    }

    pub fn new() -> Result<Self, String> {
        Self::new_dev(0)
    }

    pub fn guard(&self) -> Result<CtxGuard, String> {
        let drv = self.drv;
        unsafe {
            let mut prev: ffi::CUcontext = std::ptr::null_mut();
            let r = (drv.ctx_get_current)(&mut prev);
            if r != CUDA_SUCCESS {
                return Err(format!("rawcuda: cuCtxGetCurrent: {}", ffi::err_text(r)));
            }
            let r = (drv.ctx_set_current)(self.ctx);
            if r != CUDA_SUCCESS {
                return Err(format!("rawcuda: cuCtxSetCurrent: {}", ffi::err_text(r)));
            }
            Ok(CtxGuard { drv, prev })
        }
    }

    pub fn load_fatbin(
        &mut self,
        key: &'static str,
        image: &[u8],
        kernels: &[&'static str],
    ) -> Result<(), String> {
        let drv = self.drv;
        unsafe {
            let mut module: ffi::CUmodule = std::ptr::null_mut();
            let r = (drv.module_load_data)(&mut module, image.as_ptr() as *const _);
            if r != CUDA_SUCCESS {
                return Err(format!(
                    "rawcuda: cuModuleLoadData({key}): {} — fatbin 아키텍처 불일치 의심",
                    ffi::err_text(r)
                ));
            }
            for k in kernels {
                let mut b = k.as_bytes().to_vec();
                b.push(0);
                let mut f: CUfunction = std::ptr::null_mut();
                let r = (drv.module_get_function)(&mut f, module, b.as_ptr() as *const _);
                if r != CUDA_SUCCESS {
                    return Err(format!(
                        "rawcuda: cuModuleGetFunction({key}::{k}): {}",
                        ffi::err_text(r)
                    ));
                }
                self.fns.insert(k, f);
            }
            self.modules.insert(key, module);
        }
        Ok(())
    }

    pub fn sm_count(&self) -> u32 {
        unsafe {
            let mut v: std::os::raw::c_int = 0;
            if (self.drv.device_get_attribute)(
                &mut v,
                ffi::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                self.device,
            ) == CUDA_SUCCESS
            {
                v.max(0) as u32
            } else {
                0
            }
        }
    }

    pub fn smem_optin_max(&self) -> u32 {
        unsafe {
            let mut v: std::os::raw::c_int = 0;
            if (self.drv.device_get_attribute)(
                &mut v,
                ffi::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                self.device,
            ) == CUDA_SUCCESS
            {
                v.max(0) as u32
            } else {
                0
            }
        }
    }

    pub fn function(&self, name: &str) -> Result<CUfunction, String> {
        self.fns
            .get(name)
            .copied()
            .ok_or_else(|| format!("rawcuda: 커널 미로드: {name}"))
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn launch(
        &self,
        f: CUfunction,
        gx: u32,
        gy: u32,
        block: u32,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<(), String> {
        self.launch_on(self.stream, f, gx, gy, block, args)
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn launch_on(
        &self,
        s: CUstream,
        f: CUfunction,
        gx: u32,
        gy: u32,
        block: u32,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<(), String> {
        let t0 = std::time::Instant::now();
        self.prof_mark(f);
        unsafe {
            let r = (self.drv.launch_kernel)(
                f,
                gx,
                gy,
                1,
                block,
                1,
                1,
                0,
                s,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if r != CUDA_SUCCESS {
                return Err(format!("rawcuda: cuLaunchKernel: {}", ffi::err_text(r)));
            }
        }
        HOST_LAUNCH_NS.fetch_add(
            t0.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        HOST_LAUNCH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn set_dynamic_smem(&self, f: CUfunction, bytes: u32) -> Result<(), String> {
        unsafe {
            let r = (self.drv.func_set_attribute)(f, 8, bytes as i32);
            if r != CUDA_SUCCESS {
                return Err(format!(
                    "rawcuda: cuFuncSetAttribute(smem {bytes}B): {}",
                    ffi::err_text(r)
                ));
            }
        }
        Ok(())
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn launch_shared(
        &self,
        f: CUfunction,
        gx: u32,
        gy: u32,
        block: u32,
        shared: u32,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<(), String> {
        self.prof_mark(f);
        unsafe {
            let r = (self.drv.launch_kernel)(
                f,
                gx,
                gy,
                1,
                block,
                1,
                1,
                shared,
                self.stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if r != CUDA_SUCCESS {
                return Err(format!("rawcuda: cuLaunchKernel: {}", ffi::err_text(r)));
            }
        }
        Ok(())
    }


    pub fn create_stream(&mut self) -> Result<(), String> {
        unsafe {
            let mut s: CUstream = std::ptr::null_mut();
            let r = (self.drv.stream_create)(&mut s, 1);
            if r != CUDA_SUCCESS {
                return Err(format!("rawcuda: cuStreamCreate: {}", ffi::err_text(r)));
            }
            self.stream = s;
        }
        Ok(())
    }

    #[inline]
    pub fn capture_guard(&self, what: &str) {
        debug_assert!(
            !self.prof.capturing.get(),
            "캡처 중 버퍼 재할당({what}) — warm_for_capture 선할당 누락(P10)"
        );
    }

    pub fn create_stream2(&mut self) -> Result<(), String> {
        unsafe {
            let mut s: CUstream = std::ptr::null_mut();
            let r = (self.drv.stream_create)(&mut s, 1);
            if r != CUDA_SUCCESS {
                return Err(format!("rawcuda: cuStreamCreate(2): {}", ffi::err_text(r)));
            }
            self.stream2 = s;
        }
        Ok(())
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn stream_wait(&self, s: CUstream, ev: ffi::CUevent) -> Result<(), String> {
        let r = unsafe { (self.drv.stream_wait_event)(s, ev, 0) };
        if r != CUDA_SUCCESS {
            return Err(format!("rawcuda: cuStreamWaitEvent: {}", ffi::err_text(r)));
        }
        Ok(())
    }

    pub fn sync(&self) -> Result<(), String> {
        unsafe {
            let r = (self.drv.stream_synchronize)(self.stream);
            if r != CUDA_SUCCESS {
                return Err(format!(
                    "rawcuda: cuStreamSynchronize: {}",
                    ffi::err_text(r)
                ));
            }
        }
        Ok(())
    }
}
