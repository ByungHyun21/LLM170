//! CudaCtx — CUDA 드라이버 컨텍스트 래퍼(디바이스·컨텍스트·모듈·런치·복사).
//! 스켈레톤 단계(2026-10-04):
//! 단일 디바이스(ordinal 0)·프라이머리 컨텍스트·레거시 기본 스트림.
//! 버퍼는 명시적 alloc/free(스모크 검증용) — 영속 아레나 규칙(ADR-0014)은
//! 실 가중치 상주 단계에서 도입한다.

use crate::rawcuda::ffi::{self, CUDA_SUCCESS, CUdeviceptr, CUfunction, CUstream};
use std::collections::HashMap;

/// 진단 타이머 범주(P8) — 커널 심볼명 분류.
pub const PROF_CATS: [&str; 13] = [
    "misc", "norm", "gemv", "gemm", "gdn", "attn", "ew", "head", "moe", "scan", "l2perm", "conv",
    "gate",
];

/// [진단] 호스트 발사 누적 — (ns, calls). launch가 갱신, 보고가 판독.
pub static HOST_LAUNCH_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static HOST_LAUNCH_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// 심볼명 → 범주 인덱스(순서 주의: moe를 gemv보다 먼저 검사).
pub fn prof_cat(name: &str) -> usize {
    if name.contains("scan") {
        9
    } else if name.contains("l2perm") {
        10
    } else if name.contains("conv") {
        11
    } else if name.contains("gate") {
        12
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

/// 진단 타이머 상태 — LLM170_TIME=1일 때만 launch마다 이벤트를 남긴다.
/// 커널은 단일 스트림에 직렬이므로 (ev[i], ev[i+1]) 경과 = i번째 커널 소요.
pub struct Prof {
    pub on: bool,
    /// 캡처 중에는 마킹 금지 — 캡처 스트림의 cuEventRecord는 그래프 노드가
    /// 되어 타이밍이 왜곡되고 캡처 자체를 깨뜨릴 수 있다.
    pub capturing: std::cell::Cell<bool>,
    evs: std::cell::RefCell<Vec<ffi::CUevent>>,
    cats: std::cell::RefCell<Vec<usize>>,
}

pub struct CudaCtx {
    drv: &'static ffi::Driver,
    pub device: ffi::CUdevice,
    pub device_name: String,
    ctx: ffi::CUcontext,
    /// 레거시 기본 스트림(0). 후속 목표의
    /// 디바이스 체인(드래프트 호스트 왕복 제거)에서
    /// cuStreamCreate 도입 시 교체.
    pub stream: CUstream,
    /// 진단 타이머(P8) — LLM170_TIME=1일 때 launch마다 이벤트 1개 기록.
    /// 커널별 소요를 뒤에서 (ev[i], ev[i+1]) 경과로 복원한다(동일 스트림 직렬).
    pub prof: Prof,
    modules: HashMap<&'static str, ffi::CUmodule>,
    fns: HashMap<&'static str, CUfunction>,
    /// 복사 계측(모니터링) — 방향별 (바이트, ns, 호출). "최신 누적값"만.
    /// ns는 API 호출 구간(동기 복사는 전송 시간 포함, 비동기는 스테이징/큐잉
    /// 시간) — 유효 대역폭은 폴러가 벽시계 차분으로 계산한다.
    copy_h2d: std::cell::Cell<(u64, u64, u64)>,
    copy_d2h: std::cell::Cell<(u64, u64, u64)>,
    copy_d2d: std::cell::Cell<(u64, u64, u64)>,
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

mod caps;
mod graph;
mod mem;
mod prof;

pub use self::caps::{DeviceCaps, VramSampler, cuda_mem_free};

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
                prof: Prof {
                    // [2026-10-10 수정] 옵트인(=1) — ne0(부재 시 ON)였던 동안
                    // serve는 prof_report 미호출로 cuEvent를 발사마다 누수했다.
                    on: llm170_diag::flag::on_nonzero("LLM170_TIME"),
                    capturing: std::cell::Cell::new(false),
                    evs: std::cell::RefCell::new(Vec::new()),
                    cats: std::cell::RefCell::new(Vec::new()),
                },
                device: dev,
                device_name,
                ctx,
                stream: std::ptr::null_mut(),
                modules: HashMap::new(),
                fns: HashMap::new(),
                copy_h2d: std::cell::Cell::new((0, 0, 0)),
                copy_d2h: std::cell::Cell::new((0, 0, 0)),
                copy_d2d: std::cell::Cell::new((0, 0, 0)),
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
        // [진단] 호스트 발사 비용·횟수 — LLM170_TIME 보고에 합산(프로파일러가
        // 발사 '전'에만 이벤트를 기록하므로 갭이 직전 커널에 귀속되는 문제의
        // 실체 판정용).
        let t0 = std::time::Instant::now();
        self.prof_mark(f);
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
        HOST_LAUNCH_NS.fetch_add(
            t0.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        HOST_LAUNCH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
        self.prof_mark(f);
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

    // ── CUDA Graph 캡처(체인 1 launch 붕괴 — P1) ──

    /// 실스트림 생성(비차단) + 이 컨텍스트의 기본 스트림으로 교체.
    /// 그래프 캡처는 레거시 기본 스트림(0)에서 불가(STREAM_CAPTURE_UNSUPPORTED).
    /// 이후 모든 발사·비동기 복사가 이 스트림으로 나간다(파괴는 프로세스 수명).
    pub fn create_stream(&mut self) -> Result<(), String> {
        // SAFETY: 출력은 스택 로컬 — 생성 핸들은 필드 보관.
        unsafe {
            let mut s: CUstream = std::ptr::null_mut();
            let r = (self.drv.stream_create)(&mut s, 1); // CU_STREAM_NON_BLOCKING
            if r != CUDA_SUCCESS {
                return Err(format!("rawcuda: cuStreamCreate: {}", ffi::err_text(r)));
            }
            self.stream = s;
        }
        Ok(())
    }

    /// [R12 2026-10-10] 캡처 중 재할당 감지 — debug 빌드 즉시 실패(P10 함정
    /// 자동 검출: warm_for_capture 선할당 누락). release에서는 비용 0.
    #[inline]
    pub fn capture_guard(&self, what: &str) {
        debug_assert!(
            !self.prof.capturing.get(),
            "캡처 중 버퍼 재할당({what}) — warm_for_capture 선할당 누락(P10)"
        );
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
