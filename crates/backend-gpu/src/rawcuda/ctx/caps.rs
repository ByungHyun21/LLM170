use super::*;

pub struct VramSampler {
    drv: &'static ffi::Driver,
    ctxs: Vec<ffi::CUcontext>,
    caps: Vec<DeviceCaps>,
    names: Vec<String>,
}

/// 장치 능력(정적) — 모니터링. CMP 170HX처럼 대역폭/연산 비대칭 카드 판독용.
#[derive(Clone, Copy, Default)]
pub struct DeviceCaps {
    pub sm_count: u32,
    /// SM 최대 클럭(kHz).
    pub sm_clock_khz: u32,
    /// 메모리 클럭(kHz — 실제 클럭, DDR 배수는 peak_bw에서).
    pub mem_clock_khz: u32,
    pub bus_width_bits: u32,
}

impl DeviceCaps {
    /// 이론 피크 대역폭(bytes/s) — DDR: 2 × 메모리클럭 × 버스폭 / 8.
    /// 검증: A100/HBM2e 1215MHz×5120bit → 1.55TB/s · 4090 GDDR6X
    /// 10501MHz×384bit → 1008GB/s — 두 실스펙과 일치하는 공식.
    pub fn peak_bw_bytes(&self) -> u64 {
        2 * self.mem_clock_khz as u64 * 1000 * self.bus_width_bits as u64 / 8
    }
}

// SAFETY: 컨텍스트 핸들은 Send가 아니지만, 이 샘플러는 **단일 스레드**
// (모니터링 샘플러 스레드)가 소유·사용한다 — retain한 컨텍스트는 프로세스
// 수명 유지되고, sample()은 current 전환 후 mem_get_info만 한다.
unsafe impl Send for VramSampler {}

impl VramSampler {
    /// 전체 CUDA 장치 열거 + primary context retain. 실패 장치는 건너뛴다.
    pub fn new() -> Option<Self> {
        let drv = ffi::Driver::get().ok()?;
        // SAFETY: 프로브 경로 — 장치 열거·컨텍스트 유지(프로세스 수명).
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
                // 장치 능력(정적) — 실패 속성은 0 유지.
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

    /// 장치 능력(정적) — 대역폭 피크·SM.
    pub fn caps(&self) -> &[DeviceCaps] {
        &self.caps
    }

    /// 장치 이름.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// 장치별 (free, total) — 실패 장치는 None(게시 측이 0 유지).
    pub fn sample(&self) -> Vec<Option<(u64, u64)>> {
        let mut out = Vec::with_capacity(self.ctxs.len());
        // SAFETY: retain한 컨텍스트로 current 전환 후 mem_get_info.
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

/// cuInit → 디바이스 0 프라이머리 컨텍스트 유지 → cuMemGetInfo_v2.
/// 모듈 로드 없음(가드는 모델 적재 전 단계). 실패 시 None(호출부가 B17 정책
/// 으로 거부 — Option 계약).
pub fn cuda_mem_free() -> Option<(u64, u64)> {
    let drv = ffi::Driver::get().ok()?;
    // SAFETY: 출력은 스택 로컬 — 프로브 경로, 컨텍스트 유지는 프로세스 수명.
    unsafe {
        if (drv.init)(0) != CUDA_SUCCESS {
            return None;
        }
        let mut dev: ffi::CUdevice = 0;
        if (drv.device_get)(&mut dev, 0) != CUDA_SUCCESS {
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
