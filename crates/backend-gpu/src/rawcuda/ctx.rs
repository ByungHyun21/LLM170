//! CudaCtx — CUDA 드라이버 컨텍스트 래퍼(디바이스·컨텍스트·모듈·런치·복사).
//! 스켈레톤 단계(2026-10-04):
//! 단일 디바이스(ordinal 0)·프라이머리 컨텍스트·레거시 기본 스트림.
//! 버퍼는 명시적 alloc/free(스모크 검증용) — 영속 아레나 규칙(ADR-0014)은
//! 실 가중치 상주 단계에서 도입한다.

use crate::rawcuda::ffi::{self, CUDA_SUCCESS, CUdeviceptr, CUfunction, CUstream};
use std::collections::HashMap;

pub struct CudaCtx {
    drv: &'static ffi::Driver,
    pub device: ffi::CUdevice,
    pub device_name: String,
    ctx: ffi::CUcontext,
    /// 레거시 기본 스트림(0). 후속 목표의
    /// 디바이스 체인(드래프트 호스트 왕복 제거)에서
    /// cuStreamCreate 도입 시 교체.
    pub stream: CUstream,
    modules: HashMap<&'static str, ffi::CUmodule>,
    fns: HashMap<&'static str, CUfunction>,
}

/// 컨텍스트 스코프 가드 — 진입 시 현재 컨텍스트를 이 ctx로 전환,
/// 이탈 시 이전 컨텍스트 복원(cuCtxGetCurrent 기반 — 단일 스레드 계약
/// 환경에서 스코프 명시화).
pub struct CtxGuard {
    drv: &'static ffi::Driver,
    prev: ffi::CUcontext,
}

impl Drop for CtxGuard {
    fn drop(&mut self) {
        // SAFETY: guard 생성 시 기록한 이전 컨텍스트로 복원 — 유효한
        // 핸들이거나 null(컨텍스트 없음 상태 복원). 실패는 프로세스 계약
        // 위반 신호로 무시하지 않고 stderr 보고.
        let r = unsafe { (self.drv.ctx_set_current)(self.prev) };
        if r != CUDA_SUCCESS {
            eprintln!("rawcuda: ctx guard 복원 실패: {}", ffi::err_text(r));
        }
    }
}

impl CudaCtx {
    /// 드라이버 초기화 → 디바이스 0 확보 → 프라이머리 컨텍스트 유지·전환.
    pub fn new() -> Result<Self, String> {
        let drv = ffi::Driver::get()?;
        // SAFETY: 초기화 경로(단일 스레드) — 출력 포인터는 모두 스택 로컬.
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
            let r = (drv.device_get)(&mut dev, 0);
            if r != CUDA_SUCCESS {
                return Err(format!("rawcuda: cuDeviceGet(0): {}", ffi::err_text(r)));
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
                device: dev,
                device_name,
                ctx,
                stream: std::ptr::null_mut(),
                modules: HashMap::new(),
                fns: HashMap::new(),
            })
        }
    }

    /// 컨텍스트 가드 — 현재 스코프에서 이 ctx가 current임을 보장.
    pub fn guard(&self) -> Result<CtxGuard, String> {
        let drv = self.drv;
        // SAFETY: 출력은 스택 로컬 — 현재 컨텍스트 판독 후 우리 ctx로 전환.
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

    /// 팻바이니 이미지 로드 → 커널 심볼 해석(fatbin은 cuModuleLoadData가
    /// 아키텍처 섹션을 자동 선택 — sm_80/sm_89 이중 타겋).
    pub fn load_fatbin(
        &mut self,
        key: &'static str,
        image: &[u8],
        kernels: &[&'static str],
    ) -> Result<(), String> {
        let drv = self.drv;
        // SAFETY: 초기화·로드 경로 — image는 호출자 소유 버퍼(수명: 호출 내).
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

    /// 로드된 커널 핸들 조회.
    pub fn function(&self, name: &str) -> Result<CUfunction, String> {
        self.fns
            .get(name)
            .copied()
            .ok_or_else(|| format!("rawcuda: 커널 미로드: {name}"))
    }

    /// 디바이스 메모리 할당.
    pub fn alloc(&self, bytes: usize) -> Result<CUdeviceptr, String> {
        // SAFETY: 출력은 스택 로컬 — 할당 크기만 전달.
        unsafe {
            let mut p: CUdeviceptr = 0;
            let r = (self.drv.mem_alloc)(&mut p, bytes);
            if r != CUDA_SUCCESS {
                return Err(format!(
                    "rawcuda: cuMemAlloc({bytes}B): {}",
                    ffi::err_text(r)
                ));
            }
            Ok(p)
        }
    }

    /// 디바이스 메모리 해제(스모크 검증용 — 영속 아레나 도입 전 임시 규칙).
    pub fn free(&self, p: CUdeviceptr) -> Result<(), String> {
        // SAFETY: p는 alloc이 돌려준 유효 핸들(중복 해제 금지 계약).
        unsafe {
            let r = (self.drv.mem_free)(p);
            if r != CUDA_SUCCESS {
                return Err(format!("rawcuda: cuMemFree: {}", ffi::err_text(r)));
            }
        }
        Ok(())
    }

    /// 호스트→디바이스 복사(동기).
    pub fn h2d(&self, dst: CUdeviceptr, src: &[u8]) -> Result<(), String> {
        // SAFETY: dst는 alloc이 돌려준 유효 할당, src는 호출자 소유(호출 내 수명).
        unsafe {
            let r = (self.drv.memcpy_htod)(dst, src.as_ptr() as *const _, src.len());
            if r != CUDA_SUCCESS {
                return Err(format!(
                    "rawcuda: cuMemcpyHtoD({}B): {}",
                    src.len(),
                    ffi::err_text(r)
                ));
            }
        }
        Ok(())
    }

    /// 디바이스→호스트 복사(동기).
    pub fn d2h(&self, dst: &mut [u8], src: CUdeviceptr) -> Result<(), String> {
        // SAFETY: src는 유효 할당, dst는 호출자 소유 버퍼(길이 일치 계약).
        unsafe {
            let r = (self.drv.memcpy_dtoh)(dst.as_mut_ptr() as *mut _, src, dst.len());
            if r != CUDA_SUCCESS {
                return Err(format!(
                    "rawcuda: cuMemcpyDtoH({}B): {}",
                    dst.len(),
                    ffi::err_text(r)
                ));
            }
        }
        Ok(())
    }

    /// 디바이스→디바이스 복사(동기). 호스트 왕복 없는 체인의 기본 이동.
    pub fn d2d(&self, dst: CUdeviceptr, src: CUdeviceptr, bytes: usize) -> Result<(), String> {
        // SAFETY: 두 포인터 모두 alloc이 돌려준 유효 할당, 범위는 호출자 계약.
        unsafe {
            let r = (self.drv.memcpy_dtod)(dst, src, bytes);
            if r != CUDA_SUCCESS {
                return Err(format!(
                    "rawcuda: cuMemcpyDtoD({bytes}B): {}",
                    ffi::err_text(r)
                ));
            }
        }
        Ok(())
    }

    /// 커널 발사 — launch(f, gx, gy, block, args)
    /// (gz/bz=1 고정, shared=0, extra=null). args 원소는 각 커널 인자값을
    /// 가리키는 포인터(cuLaunchKernel 규격 — 인자 주소 배열).
    /// clippy allow: f는 드라이버 불투명 핸들 — 해드 유효성은 본문 SAFETY 계약
    /// (호출자 보증)과 드라이버 CUresult 검증에 맡긴다. unsafe fn화 시 호출점
    /// 200+곳 러플이 계약 강화 없이 노이즈만 늘린다(2026-10-07 판정).
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn launch(
        &self,
        f: CUfunction,
        gx: u32,
        gy: u32,
        block: u32,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<(), String> {
        // SAFETY: f는 function()이 돌려준 유효 핸들, args 포인터들은
        // 호출 시점까지 유효한 스택 로컬(호출자 계약).
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

    /// 함수 속성: 동적 공유메모리 상한 opt-in(정적 48KB 초과 커널 —
    /// G5 gdn_scan 61,828B). CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_
    /// SIZE_BYTES=8(cuda.h 규약 — sm_80 164KB/SM 상한 내에서만 성공).
    /// clippy allow — launch와 동일 판정(불투명 핸들).
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn set_dynamic_smem(&self, f: CUfunction, bytes: u32) -> Result<(), String> {
        // SAFETY: f는 function()이 돌려준 유효 핸들 — 스칼라 속성값만 전달.
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

    /// 커널 발사(동적 공유메모리 지정) — launch의 shared=0 고정을 푼 변형
    /// (G5 scan). 48KB 초과 분은 set_dynamic_smem 사전 opt-in이 필수.
    /// clippy allow — launch와 동일 판정(불투명 핸들).
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
        // SAFETY: launch와 동일 계약 — f는 유효 핸들, args 포인터들은
        // 호출 시점까지 유효한 스택 로컬(호출자 계약).
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

    /// 기본 스트림 동기화.
    pub fn sync(&self) -> Result<(), String> {
        // SAFETY: stream 필드는 new()가 설정한 값(레거시 기본 스트림 0).
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

/// B6: CUDA 런타임 VRAM 프로브 — 가드 preflight용.
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
