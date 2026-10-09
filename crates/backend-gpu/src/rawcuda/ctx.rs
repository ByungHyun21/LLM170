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
                    on: llm170_diag::flag::ne0("LLM170_TIME"),
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
        let t0 = std::time::Instant::now();
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
        Self::bump(&self.copy_h2d, src.len(), t0.elapsed().as_nanos() as u64);
        Ok(())
    }

    /// 호스트→디바이스 복사(비동기 — 기본 스트림 순서 계약). 페이지러블
    /// 소스는 드라이버가 반환 전 스테이징하므로 호출 내 수명이면 충분하다.
    /// 목적지가 커널 입력이면 같은 스트림 순서로 보이고, 관측 전에는 동기
    /// d2h/스트림 동기화가 온다.
    pub fn h2d_async(&self, dst: CUdeviceptr, src: &[u8]) -> Result<(), String> {
        let t0 = std::time::Instant::now();
        // SAFETY: dst는 alloc이 돌려준 유효 할당, src는 호출 내 수명(스테이징 계약).
        unsafe {
            let r =
                (self.drv.memcpy_htod_async)(dst, src.as_ptr() as *const _, src.len(), self.stream);
            if r != CUDA_SUCCESS {
                return Err(format!(
                    "rawcuda: cuMemcpyHtoDAsync({}B): {}",
                    src.len(),
                    ffi::err_text(r)
                ));
            }
        }
        Self::bump(&self.copy_h2d, src.len(), t0.elapsed().as_nanos() as u64);
        Ok(())
    }

    /// 디바이스→호스트 복사(동기).
    pub fn d2h(&self, dst: &mut [u8], src: CUdeviceptr) -> Result<(), String> {
        let t0 = std::time::Instant::now();
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
        Self::bump(&self.copy_d2h, dst.len(), t0.elapsed().as_nanos() as u64);
        Ok(())
    }

    /// 디바이스→디바이스 복사(비동기 — 기본 스트림 순서 계약). 동기
    /// DtoD는 호출마다 스트림을 배수해 디바이스 체인(토큰당 수백 회)에서
    /// GP 유휴를 만든다(2026-10-08 실측). 목적지를 관측하는 쪽은 항상
    /// 같은 스트림의 커널 또는 동기 d2h이므로 순서만 보장되면 된다.
    pub fn d2d(&self, dst: CUdeviceptr, src: CUdeviceptr, bytes: usize) -> Result<(), String> {
        let t0 = std::time::Instant::now();
        // SAFETY: 두 포인터 모두 alloc이 돌려준 유효 할당, 범위는 호출자 계약.
        unsafe {
            let r = (self.drv.memcpy_dtod_async)(dst, src, bytes, self.stream);
            if r != CUDA_SUCCESS {
                return Err(format!(
                    "rawcuda: cuMemcpyDtoDAsync({bytes}B): {}",
                    ffi::err_text(r)
                ));
            }
        }
        Self::bump(&self.copy_d2d, bytes, t0.elapsed().as_nanos() as u64);
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

    /// pinned 호스트 할당 — 캡처 가능한 async 복사의 소스/목적지.
    pub fn pinned_alloc(&self, bytes: usize) -> Result<*mut std::ffi::c_void, String> {
        // SAFETY: 출력은 스택 로컬 — 해제는 pinned_free 계약.
        unsafe {
            let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
            let r = (self.drv.mem_host_alloc)(&mut p, bytes, 0);
            if r != CUDA_SUCCESS {
                return Err(format!(
                    "rawcuda: cuMemHostAlloc({bytes}B): {}",
                    ffi::err_text(r)
                ));
            }
            Ok(p)
        }
    }

    /// pinned 해제. clippy allow — launch와 동일 판정(불투명 핸들: 유효성은
    /// 발급 계약과 드라이버 CUresult 검증에 맡긴다).
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn pinned_free(&self, p: *mut std::ffi::c_void) -> Result<(), String> {
        // SAFETY: p는 pinned_alloc이 돌려준 유효 핸들(중복 해제 금지 계약).
        unsafe {
            let r = (self.drv.mem_free_host)(p);
            if r != CUDA_SUCCESS {
                return Err(format!("rawcuda: cuMemFreeHost: {}", ffi::err_text(r)));
            }
        }
        Ok(())
    }

    /// 디바이스→호스트 비동기 복사(pinned dst — 캡처 노드로 기록 가능).
    /// clippy allow — dst는 호출자가 보증하는 pinned 포인터(불투명 핸들 계약).
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn d2h_async(&self, dst: *mut u8, src: CUdeviceptr, bytes: usize) -> Result<(), String> {
        let t0 = std::time::Instant::now();
        // SAFETY: dst는 pinned(호출자 계약), src는 유효 할당, 범위는 호출자 계약.
        unsafe {
            let r =
                (self.drv.memcpy_dtoh_async)(dst as *mut std::ffi::c_void, src, bytes, self.stream);
            if r != CUDA_SUCCESS {
                return Err(format!(
                    "rawcuda: cuMemcpyDtoHAsync({bytes}B): {}",
                    ffi::err_text(r)
                ));
            }
        }
        Self::bump(&self.copy_d2h, bytes, t0.elapsed().as_nanos() as u64);
        Ok(())
    }

    /// 그래프 캡처 개시(THREAD_LOCAL) — 이후 이 스트림의 발사가 그래프 노드로
    /// 기록된다(실행 아님). 캡처 중 동기 복사·alloc·sync는 금지.
    pub fn capture_begin(&self) -> Result<(), String> {
        self.prof.capturing.set(true);
        // SAFETY: stream은 create_stream이 설정한 유효 핸들.
        unsafe {
            let r = (self.drv.stream_begin_capture)(self.stream, 1);
            if r != CUDA_SUCCESS {
                return Err(format!(
                    "rawcuda: cuStreamBeginCapture: {}",
                    ffi::err_text(r)
                ));
            }
        }
        Ok(())
    }

    /// 캡처 종료 → 그래프 핸들. 실패 시에도 캡처 상태는 해제된다.
    pub fn capture_end(&self) -> Result<ffi::CUgraph, String> {
        self.prof.capturing.set(false);
        // SAFETY: 출력은 스택 로컬 — 그래프 핸들은 호출자 소유(graph_destroy).
        unsafe {
            let mut g: ffi::CUgraph = std::ptr::null_mut();
            let r = (self.drv.stream_end_capture)(self.stream, &mut g);
            if r != CUDA_SUCCESS {
                return Err(format!("rawcuda: cuStreamEndCapture: {}", ffi::err_text(r)));
            }
            Ok(g)
        }
    }

    /// 그래프 인스턴스화 — 실행 핸들(캡처 1회·replay N회).
    /// clippy allow — launch와 동일 판정(불투명 핸들).
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn graph_instantiate(&self, g: ffi::CUgraph) -> Result<ffi::CUgraphExec, String> {
        // SAFETY: g는 capture_end가 돌려준 유효 그래프.
        unsafe {
            let mut e: ffi::CUgraphExec = std::ptr::null_mut();
            let r = (self.drv.graph_instantiate)(&mut e, g, 0);
            if r != CUDA_SUCCESS {
                return Err(format!(
                    "rawcuda: cuGraphInstantiateWithFlags: {}",
                    ffi::err_text(r)
                ));
            }
            Ok(e)
        }
    }

    /// 그래프 실행 — 전 노드를 1회 launch로 제출.
    /// clippy allow — launch와 동일 판정(불투명 핸들).
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn graph_launch(&self, e: ffi::CUgraphExec) -> Result<(), String> {
        // SAFETY: e는 graph_instantiate가 돌려준 유효 실행 핸들.
        unsafe {
            let r = (self.drv.graph_launch)(e, self.stream);
            if r != CUDA_SUCCESS {
                return Err(format!("rawcuda: cuGraphLaunch: {}", ffi::err_text(r)));
            }
        }
        Ok(())
    }

    /// 그래프·실행 핸들 해제.
    /// clippy allow — launch와 동일 판정(불투명 핸들).
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn graph_destroy(&self, e: ffi::CUgraphExec, g: ffi::CUgraph) -> Result<(), String> {
        // SAFETY: 두 핸들 모두 본 모듈이 발급한 유효 핸들(중복 해제 금지 계약).
        unsafe {
            if !e.is_null() {
                let r = (self.drv.graph_exec_destroy)(e);
                if r != CUDA_SUCCESS {
                    return Err(format!("rawcuda: cuGraphExecDestroy: {}", ffi::err_text(r)));
                }
            }
            if !g.is_null() {
                let r = (self.drv.graph_destroy)(g);
                if r != CUDA_SUCCESS {
                    return Err(format!("rawcuda: cuGraphDestroy: {}", ffi::err_text(r)));
                }
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
impl CudaCtx {
    /// 복사 계측 접근자 — [h2d, d2h, d2d] × (바이트, ns, 호출).
    pub fn copy_stats(&self) -> [(u64, u64, u64); 3] {
        [
            self.copy_h2d.get(),
            self.copy_d2h.get(),
            self.copy_d2d.get(),
        ]
    }

    /// 계측 누적(내부).
    fn bump(c: &std::cell::Cell<(u64, u64, u64)>, bytes: usize, ns: u64) {
        let (b, t, n) = c.get();
        c.set((b + bytes as u64, t + ns, n + 1));
    }
}

/// 다중 GPU VRAM 샘플러 — 기동 시 장치별 primary context를 **1회** retain하고
/// 이후 `cuMemGetInfo`만 반복한다(스크랩마다 retain 금지: refcount 무한 증가 +
/// 호출 스레드 current ctx 오염 — 모니터링 샘플러 스레드 전용 계약).
///
/// 값 의미론: free/total은 **기기 전체** 값 — 이 프로세스 귀속이 아니다
/// (공유 기기에서는 타 프로세스 포함; 독점 배포 가정).
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

impl CudaCtx {
    /// 이벤트 생성(진단 타이머).
    pub fn event_create(&self) -> Result<ffi::CUevent, String> {
        let mut ev: ffi::CUevent = std::ptr::null_mut();
        // SAFETY: 초기화 경로(단일 스레드) — 출력 포인터는 스택 로컬.
        let r = unsafe { (self.drv.event_create)(&mut ev, 0) };
        if r != CUDA_SUCCESS {
            return Err(format!("rawcuda: cuEventCreate: {}", ffi::err_text(r)));
        }
        Ok(ev)
    }

    /// clippy allow — launch와 동일 판정(불투명 핸들).
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn event_record(&self, ev: ffi::CUevent) -> Result<(), String> {
        // SAFETY: ev는 event_create 산출 핸들(호출자 수명 계약).
        let r = unsafe { (self.drv.event_record)(ev, self.stream) };
        if r != CUDA_SUCCESS {
            return Err(format!("rawcuda: cuEventRecord: {}", ffi::err_text(r)));
        }
        Ok(())
    }

    /// clippy allow — launch와 동일 판정(불투명 핸들).
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn event_destroy(&self, ev: ffi::CUevent) {
        // SAFETY: ev는 본 모듈이 만든 핸들 — 파괴 후 재사용 금지(호출자 계약).
        unsafe {
            let _ = (self.drv.event_destroy)(ev);
        }
    }

    /// 커널 발사 직전 훅 — 함수 포인터를 심볼명(역상)으로 찾아 범주와 함께
    /// 이벤트를 남긴다. fns는 ~40항목이라 런치당 선형탐색이 저렴하다.
    pub fn prof_mark(&self, f: CUfunction) {
        if !self.prof.on || self.prof.capturing.get() {
            return;
        }
        let mut name = "";
        for (k, v) in &self.fns {
            if *v == f {
                name = k;
                break;
            }
        }
        let Ok(ev) = self.event_create() else { return };
        if self.event_record(ev).is_err() {
            return;
        }
        self.prof.evs.borrow_mut().push(ev);
        self.prof.cats.borrow_mut().push(prof_cat(name));
    }

    /// 범주별 합산 리포트(ms/토큰, %). 호출 전 sync 권장 — 여기서 마지막
    /// 센티넬 이벤트를 기록하고 sync한다. 이벤트는 파괴 후 비운다.
    pub fn prof_report(&self, label: &str) -> Result<String, String> {
        if !self.prof.on {
            return Ok(String::new());
        }
        let sentinel = self.event_create()?;
        self.event_record(sentinel)?;
        // SAFETY: 스트림 완료 대기(동기 호출 — 계약: 단일 스레드 사용).
        let r = unsafe { (self.drv.stream_synchronize)(self.stream) };
        if r != CUDA_SUCCESS {
            return Err(format!("rawcuda: prof sync: {}", ffi::err_text(r)));
        }
        let evs = self.prof.evs.borrow();
        let cats = self.prof.cats.borrow();
        let mut acc = [0f64; PROF_CATS.len()];
        let mut total = 0f64;
        for i in 0..evs.len() {
            let a = evs[i];
            let b = if i + 1 < evs.len() {
                evs[i + 1]
            } else {
                sentinel
            };
            let mut ms = 0f32;
            // SAFETY: 두 이벤트 모두 기록 완료(sync 후) — elapsed 유효.
            let r = unsafe { (self.drv.event_elapsed)(&mut ms, a, b) };
            if r == CUDA_SUCCESS {
                acc[cats[i]] += ms as f64;
                total += ms as f64;
            }
        }
        let mut out = format!("prof[{label}] 합 {total:.2}ms");
        for (i, cat) in PROF_CATS.iter().enumerate() {
            if acc[i] > 0.001 {
                out.push_str(&format!(
                    " · {cat} {:.2}ms({:.0}%)",
                    acc[i],
                    100.0 * acc[i] / total.max(0.001)
                ));
            }
        }
        drop(evs);
        drop(cats);
        for ev in self.prof.evs.borrow_mut().drain(..) {
            self.event_destroy(ev);
        }
        self.prof.cats.borrow_mut().clear();
        self.event_destroy(sentinel);
        Ok(out)
    }
}
