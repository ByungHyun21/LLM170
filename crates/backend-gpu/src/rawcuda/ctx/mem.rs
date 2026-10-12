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

    /// [VMM 2026-10-12] 디바이스 로컬 PINNED 할당 속성 — 세목 조회와
    /// cuMemCreate가 공유.
    fn vmm_prop(&self) -> ffi::CUmemAllocationProp {
        ffi::CUmemAllocationProp {
            ty: ffi::CU_MEM_ALLOCATION_TYPE_PINNED,
            requested_handle_types: ffi::CU_MEM_HANDLE_TYPE_NONE,
            location: ffi::CUmemLocation {
                ty: ffi::CU_MEM_LOCATION_TYPE_DEVICE,
                id: self.device,
            },
            win32_handle_meta_data: std::ptr::null_mut(),
            alloc_flags: ffi::CUmemAllocationPropAllocFlags {
                compression_type: 0,
                gpu_direct_rdma_capable: 0,
                usage: 0,
                reserved: [0; 4],
            },
        }
    }

    /// [VMM] 최소 할당 세목(바이트) — cuMemGetAllocationGranularity
    /// (MINIMUM) 조회. 값은 하드코딩하지 않는다(디바이스별 상이 — 통상
    /// 2MiB급). alloc_vmm이 요청 크기를 이 배수로 올린다.
    pub fn vmm_granularity(&self) -> Result<usize, String> {
        let prop = self.vmm_prop();
        // SAFETY: 출력은 스택 로컬, prop은 위에서 완전 초기화(포인터 필드 null).
        unsafe {
            let mut g: usize = 0;
            let r = (self.drv.mem_get_allocation_granularity)(
                &mut g,
                &prop,
                ffi::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            );
            if r != CUDA_SUCCESS {
                return Err(format!(
                    "rawcuda: cuMemGetAllocationGranularity: {}",
                    ffi::err_text(r)
                ));
            }
            Ok(g)
        }
    }

    /// [VMM] 가상 주소 기반 디바이스 할당 — granularity 올림 →
    /// cuMemCreate(물리) → cuMemAddressReserve(VA) → cuMemMap →
    /// cuMemSetAccess(디바이스 RW). 반환 ptr은 일반 디바이스 포인터와
    /// 동일하게 h2d/d2d/커널 인자로 쓸 수 있다. 매핑 실측 크기는 올림
    /// 크기로 내부 대장에 기록된다 — 해제는 free_vmm(ptr, 원래 bytes).
    /// 실패 경로는 이미 확보한 자원을 즉시 반납한다(누수 없음).
    pub fn alloc_vmm(&self, bytes: usize) -> Result<CUdeviceptr, String> {
        if bytes == 0 {
            return Err("rawcuda: alloc_vmm(0B) — 0바이트 할당 불가".into());
        }
        let g = self.vmm_granularity()?;
        if g == 0 {
            return Err("rawcuda: alloc_vmm: 세목 조회가 0을 반환".into());
        }
        let size = bytes.div_ceil(g) * g;
        let prop = self.vmm_prop();
        // SAFETY: 출력·prop은 스택 로컬 — 각 단계는 CUresult 검증 후 진행,
        // 실패 시 확보분 선반납(best-effort — 원본 오류 전파).
        unsafe {
            let mut handle: ffi::CUmemGenericAllocationHandle = 0;
            let r = (self.drv.mem_create)(&mut handle, size, &prop, 0);
            if r != CUDA_SUCCESS {
                return Err(format!(
                    "rawcuda: cuMemCreate({bytes}B→{size}B): {}",
                    ffi::err_text(r)
                ));
            }
            let mut ptr: CUdeviceptr = 0;
            let r = (self.drv.mem_address_reserve)(&mut ptr, size, 0, 0, 0);
            if r != CUDA_SUCCESS {
                let _ = (self.drv.mem_release)(handle);
                return Err(format!(
                    "rawcuda: cuMemAddressReserve({size}B): {}",
                    ffi::err_text(r)
                ));
            }
            let r = (self.drv.mem_map)(ptr, size, 0, handle, 0);
            if r != CUDA_SUCCESS {
                let _ = (self.drv.mem_release)(handle);
                let _ = (self.drv.mem_address_free)(ptr, size);
                return Err(format!("rawcuda: cuMemMap({size}B): {}", ffi::err_text(r)));
            }
            let desc = ffi::CUmemAccessDesc {
                location: prop.location,
                flags: ffi::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
            };
            let r = (self.drv.mem_set_access)(ptr, size, &desc, 1);
            if r != CUDA_SUCCESS {
                let _ = (self.drv.mem_unmap)(ptr, size);
                let _ = (self.drv.mem_release)(handle);
                let _ = (self.drv.mem_address_free)(ptr, size);
                return Err(format!(
                    "rawcuda: cuMemSetAccess({size}B): {}",
                    ffi::err_text(r)
                ));
            }
            self.vmm.borrow_mut().insert(ptr, (handle, size));
            Ok(ptr)
        }
    }

    /// [VMM] 해제 — cuMemUnmap(VA 매핑 해제) → cuMemRelease(물리 반납) →
    /// cuMemAddressFree(VA 예약 해제). bytes는 alloc_vmm에 전달했던
    /// 요청값(일관성 검사용 — 실제 언맵 크기는 대장의 올림 크기).
    /// 중간 단계 실패 시 대장을 유지하나 이미 진행된 단계가 있어 재시도가
    /// 무의미할 수 있다 — 오류는 상위로 전파한다(드라이버 오류 = 프로세스
    /// 계약 위반 취급, 통상 경로와 동일).
    pub fn free_vmm(&self, ptr: CUdeviceptr, bytes: usize) -> Result<(), String> {
        let mut m = self.vmm.borrow_mut();
        let Some(&(handle, size)) = m.get(&ptr) else {
            return Err(format!("rawcuda: free_vmm: 미등록 VMM ptr={ptr:#x}"));
        };
        if bytes > size {
            return Err(format!(
                "rawcuda: free_vmm: 요청 {bytes}B > 매핑 {size}B — alloc_vmm 인자 불일치"
            ));
        }
        // SAFETY: ptr/handle/size는 alloc_vmm이 대장에 기록한 값.
        unsafe {
            let r = (self.drv.mem_unmap)(ptr, size);
            if r != CUDA_SUCCESS {
                return Err(format!("rawcuda: cuMemUnmap: {}", ffi::err_text(r)));
            }
            let r = (self.drv.mem_release)(handle);
            if r != CUDA_SUCCESS {
                return Err(format!("rawcuda: cuMemRelease: {}", ffi::err_text(r)));
            }
            let r = (self.drv.mem_address_free)(ptr, size);
            if r != CUDA_SUCCESS {
                return Err(format!("rawcuda: cuMemAddressFree: {}", ffi::err_text(r)));
            }
        }
        m.remove(&ptr);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// VMM 왕복 스모크 — granularity 조회 → create→reserve→map→setAccess →
    /// h2d 기록 → d2h 판독 → unmap→release→addressFree 전주기(2026-10-12).
    /// GPU 없는 기기(기본 빌드 녹색 계약)에서는 컨텍스트 생성 실패를
    /// 조기 통과로 처리한다.
    #[test]
    fn vmm_alloc_roundtrip() {
        let ctx = match CudaCtx::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("vmm_alloc_roundtrip 건너뜀(GPU 없음): {e}");
                return;
            }
        };
        let g = ctx.vmm_granularity().expect("granularity 조회");
        assert!(g > 0, "세목이 0");
        // 요청 크기 < 세목 → 내부 올림 경로 검증(4096B → g 배수).
        let payload: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
        let ptr = ctx.alloc_vmm(payload.len()).expect("alloc_vmm");
        ctx.h2d(ptr, &payload).expect("h2d");
        let mut back = vec![0u8; payload.len()];
        ctx.d2h(&mut back, ptr).expect("d2h");
        assert_eq!(payload, back, "VMM 왕복 데이터 불일치");
        ctx.free_vmm(ptr, payload.len()).expect("free_vmm");
    }
}
