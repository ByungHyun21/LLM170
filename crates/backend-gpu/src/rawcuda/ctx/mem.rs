use super::*;

impl CudaCtx {
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

    /// 기존 호스트 버퍼 pin(핀드 DMA 소스, P7-fix) — flags=0.
    /// 실패(주로 RLIMIT_MEMLOCK 초과)는 호출부가 폴백 판단.
    ///
    /// # Safety
    /// `p`는 `bytes` 범위가 유효한 호스트 포인터여야 하고, 해제 전에
    /// [`Self::host_unregister`]로 등록을 풀어야 한다.
    pub unsafe fn host_register(
        &self,
        p: *mut std::ffi::c_void,
        bytes: usize,
    ) -> Result<(), String> {
        // SAFETY: 호출자가 p/bytes 유효성을 보증한다(위 Safety 계약).
        let r = unsafe { (self.drv.host_register)(p, bytes, 0) };
        if r != CUDA_SUCCESS {
            return Err(ffi::err_text(r));
        }
        Ok(())
    }

    /// pin 해제 — 등록 포인터 해제 **전** 호출.
    ///
    /// # Safety
    /// `p`는 [`Self::host_register`]에 성공한 포인터여야 한다.
    pub unsafe fn host_unregister(&self, p: *mut std::ffi::c_void) -> Result<(), String> {
        // SAFETY: 호출자가 host_register 성공 포인터임을 보증한다.
        let r = unsafe { (self.drv.host_unregister)(p) };
        if r != CUDA_SUCCESS {
            return Err(ffi::err_text(r));
        }
        Ok(())
    }

    /// 디바이스 영역 0 채움(비동기 — 기본 스트림 순서 계약).
    /// [2026-10-09 H] 종전 zero_dev의 동기 h2d 4MiB 청크 루프 대체 —
    /// cuMemsetD8Async 1콜(스트림 순서라 후속 커널·판독과 정합).
    pub fn memset0_async(&self, dst: CUdeviceptr, bytes: usize) -> Result<(), String> {
        // SAFETY: dst는 alloc이 돌려준 유효 할당, 범위는 호출자 계약.
        unsafe {
            let r = (self.drv.memset_d8_async)(dst, 0, bytes, self.stream);
            if r != CUDA_SUCCESS {
                return Err(format!(
                    "rawcuda: cuMemsetD8Async({bytes}B): {}",
                    ffi::err_text(r)
                ));
            }
        }
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

    /// 호스트 발사 누적(진단): (ns, calls).
    pub fn host_launch_stats() -> (u64, u64) {
        (
            HOST_LAUNCH_NS.load(std::sync::atomic::Ordering::Relaxed),
            HOST_LAUNCH_CALLS.load(std::sync::atomic::Ordering::Relaxed),
        )
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

    /// 복사 계측 접근자 — [h2d, d2h, d2d] × (바이트, ns, 호출).
    pub fn copy_stats(&self) -> [(u64, u64, u64); 3] {
        [
            self.copy_h2d.get(),
            self.copy_d2h.get(),
            self.copy_d2d.get(),
        ]
    }

    /// 계측 누적(내부).
    pub(super) fn bump(c: &std::cell::Cell<(u64, u64, u64)>, bytes: usize, ns: u64) {
        let (b, t, n) = c.get();
        c.set((b + bytes as u64, t + ns, n + 1));
    }
}
