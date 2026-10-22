use super::*;

pub struct VramSampler {
    drv: &'static ffi::Driver,
    ctxs: Vec<ffi::CUcontext>,
    caps: Vec<DeviceCaps>,
    names: Vec<String>,
}

#[derive(Clone, Copy, Default)]
pub struct DeviceCaps {
    pub sm_count: u32,
    pub sm_clock_khz: u32,
    pub mem_clock_khz: u32,
    pub bus_width_bits: u32,
}

impl DeviceCaps {
    pub fn peak_bw_bytes(&self) -> u64 {
        2 * self.mem_clock_khz as u64 * 1000 * self.bus_width_bits as u64 / 8
    }
}

unsafe impl Send for VramSampler {}

impl VramSampler {
    pub fn new() -> Option<Self> {
        let drv = ffi::Driver::get().ok()?;
        unsafe {
            if (drv.init)(0) != CUDA_SUCCESS {
                return None;
            }
            let mut n: i32 = 0;
            if (drv.device_get_count)(&mut n) != CUDA_SUCCESS || n <= 0 {
                return None;
            }
            let mut ctxs = Vec::new();
            let mut caps = Vec::new();
            let mut names = Vec::new();
            for i in 0..n {
                let mut dev: ffi::CUdevice = 0;
                if (drv.device_get)(&mut dev, i) != CUDA_SUCCESS {
                    continue;
                }
                let mut ctx: ffi::CUcontext = std::ptr::null_mut();
                if (drv.device_primary_ctx_retain)(&mut ctx, dev) != CUDA_SUCCESS {
                    continue;
                }
                let attr = |a: std::os::raw::c_int| -> u32 {
                    let mut v: std::os::raw::c_int = 0;
                    if (drv.device_get_attribute)(&mut v, a, dev) == CUDA_SUCCESS {
                        v.max(0) as u32
                    } else {
                        0
                    }
                };
                caps.push(DeviceCaps {
                    sm_count: attr(ffi::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
                    sm_clock_khz: attr(ffi::CU_DEVICE_ATTRIBUTE_CLOCK_RATE),
                    mem_clock_khz: attr(ffi::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE),
                    bus_width_bits: attr(ffi::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH),
                });
                let mut nbuf = [0i8; 128];
                let nm = if (drv.device_get_name)(nbuf.as_mut_ptr(), nbuf.len() as i32, dev)
                    == CUDA_SUCCESS
                {
                    let bytes: Vec<u8> = nbuf
                        .iter()
                        .take_while(|&&c| c != 0)
                        .map(|&c| c as u8)
                        .collect();
                    String::from_utf8_lossy(&bytes).into_owned()
                } else {
                    String::new()
                };
                names.push(nm);
                ctxs.push(ctx);
            }
            if ctxs.is_empty() {
                return None;
            }
            Some(VramSampler {
                drv,
                ctxs,
                caps,
                names,
            })
        }
    }

    pub fn device_count(&self) -> usize {
        self.ctxs.len()
    }

    pub fn caps(&self) -> &[DeviceCaps] {
        &self.caps
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub fn sample(&self) -> Vec<Option<(u64, u64)>> {
        let mut out = Vec::with_capacity(self.ctxs.len());
        unsafe {
            for &ctx in &self.ctxs {
                if (self.drv.ctx_set_current)(ctx) != CUDA_SUCCESS {
                    out.push(None);
                    continue;
                }
                let (mut f, mut t) = (0usize, 0usize);
                if (self.drv.mem_get_info)(&mut f, &mut t) != CUDA_SUCCESS {
                    out.push(None);
                    continue;
                }
                out.push(Some((f as u64, t as u64)));
            }
        }
        out
    }
}

pub fn cuda_mem_free_dev(ordinal: i32) -> Option<(u64, u64)> {
    let drv = ffi::Driver::get().ok()?;
    unsafe {
        if (drv.init)(0) != CUDA_SUCCESS {
            return None;
        }
        let mut dev: ffi::CUdevice = 0;
        if (drv.device_get)(&mut dev, ordinal) != CUDA_SUCCESS {
            return None;
        }
        let mut ctx: ffi::CUcontext = std::ptr::null_mut();
        if (drv.device_primary_ctx_retain)(&mut ctx, dev) != CUDA_SUCCESS {
            return None;
        }
        if (drv.ctx_set_current)(ctx) != CUDA_SUCCESS {
            return None;
        }
        let (mut free, mut total) = (0usize, 0usize);
        if (drv.mem_get_info)(&mut free, &mut total) != CUDA_SUCCESS {
            return None;
        }
        Some((free as u64, total as u64))
    }
}

pub fn cuda_mem_free() -> Option<(u64, u64)> {
    cuda_mem_free_dev(0)
}
