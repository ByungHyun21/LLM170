use super::*;

impl CudaCtx {
    pub fn alloc(&self, bytes: usize) -> Result<CUdeviceptr, String> {
        unsafe {
            let mut p: CUdeviceptr = 0;
            let r = (self.drv.mem_alloc)(&mut p, bytes);
            if r != CUDA_SUCCESS {
                return Err(format!(
                    "rawcuda: cuMemAlloc({bytes}B): {}",
                    ffi::err_text(r)
                ));
            }
            self.allocs.borrow_mut().push(p);
            Ok(p)
        }
    }

    pub fn free_all(&self) {
        unsafe {
            let r = (self.drv.ctx_set_current)(self.ctx);
            if r != CUDA_SUCCESS {
                eprintln!("rawcuda: free_all ctx 전환 실패: {}", ffi::err_text(r));
                return;
            }
            for p in self.allocs.borrow_mut().drain(..) {
                let r = (self.drv.mem_free)(p);
                if r != CUDA_SUCCESS {
                    eprintln!("rawcuda: free_all cuMemFree 실패: {}", ffi::err_text(r));
                }
            }
        }
    }

    pub fn free(&self, p: CUdeviceptr) -> Result<(), String> {
        unsafe {
            let r = (self.drv.mem_free)(p);
            if r != CUDA_SUCCESS {
                return Err(format!("rawcuda: cuMemFree: {}", ffi::err_text(r)));
            }
        }
        self.allocs.borrow_mut().retain(|&x| x != p);
        Ok(())
    }

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

    pub fn vmm_granularity(&self) -> Result<usize, String> {
        let prop = self.vmm_prop();
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

    pub fn h2d(&self, dst: CUdeviceptr, src: &[u8]) -> Result<(), String> {
        let t0 = std::time::Instant::now();
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

    pub fn h2d_async(&self, dst: CUdeviceptr, src: &[u8]) -> Result<(), String> {
        let t0 = std::time::Instant::now();
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



    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn host_register(
        &self,
        p: *mut std::ffi::c_void,
        bytes: usize,
    ) -> Result<(), String> {
        let r = unsafe { (self.drv.host_register)(p, bytes, 0) };
        if r != CUDA_SUCCESS {
            return Err(ffi::err_text(r));
        }
        Ok(())
    }



    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn host_unregister(&self, p: *mut std::ffi::c_void) -> Result<(), String> {
        let r = unsafe { (self.drv.host_unregister)(p) };
        if r != CUDA_SUCCESS {
            return Err(ffi::err_text(r));
        }
        Ok(())
    }

    pub fn memset0_async(&self, dst: CUdeviceptr, bytes: usize) -> Result<(), String> {
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

    pub fn d2h(&self, dst: &mut [u8], src: CUdeviceptr) -> Result<(), String> {
        let t0 = std::time::Instant::now();
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

    pub fn d2d(&self, dst: CUdeviceptr, src: CUdeviceptr, bytes: usize) -> Result<(), String> {
        let t0 = std::time::Instant::now();
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

    pub fn pinned_alloc(&self, bytes: usize) -> Result<*mut std::ffi::c_void, String> {
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

    pub fn pinned_alloc_device(
        &self,
        bytes: usize,
    ) -> Result<(*mut std::ffi::c_void, CUdeviceptr), String> {
        let _g = self.guard()?;
        const CU_MEMHOSTALLOC_DEVICEMAP: std::ffi::c_uint = 0x02;
        unsafe {
            let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
            let r = (self.drv.mem_host_alloc)(&mut p, bytes, CU_MEMHOSTALLOC_DEVICEMAP);
            if r != CUDA_SUCCESS {
                return Err(format!(
                    "rawcuda: cuMemHostAlloc(device-map, {bytes}B): {}",
                    ffi::err_text(r)
                ));
            }
            let mut dev: CUdeviceptr = 0;
            let r2 = (self.drv.mem_host_get_device_pointer)(&mut dev, p, 0);
            if r2 != CUDA_SUCCESS {
                let _ = (self.drv.mem_free_host)(p);
                return Err(format!(
                    "rawcuda: cuMemHostGetDevicePointer({bytes}B): {}",
                    ffi::err_text(r2)
                ));
            }
            Ok((p, dev))
        }
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn pinned_free(&self, p: *mut std::ffi::c_void) -> Result<(), String> {
        unsafe {
            let r = (self.drv.mem_free_host)(p);
            if r != CUDA_SUCCESS {
                return Err(format!("rawcuda: cuMemFreeHost: {}", ffi::err_text(r)));
            }
        }
        Ok(())
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn d2h_async(&self, dst: *mut u8, src: CUdeviceptr, bytes: usize) -> Result<(), String> {
        let t0 = std::time::Instant::now();
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

    pub fn copy_stats(&self) -> [(u64, u64, u64); 3] {
        [
            self.copy_h2d.get(),
            self.copy_d2h.get(),
            self.copy_d2d.get(),
        ]
    }

    pub(super) fn bump(c: &std::cell::Cell<(u64, u64, u64)>, bytes: usize, ns: u64) {
        let (b, t, n) = c.get();
        c.set((b + bytes as u64, t + ns, n + 1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let payload: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
        let ptr = ctx.alloc_vmm(payload.len()).expect("alloc_vmm");
        ctx.h2d(ptr, &payload).expect("h2d");
        let mut back = vec![0u8; payload.len()];
        ctx.d2h(&mut back, ptr).expect("d2h");
        assert_eq!(payload, back, "VMM 왕복 데이터 불일치");
        ctx.free_vmm(ptr, payload.len()).expect("free_vmm");
    }
}
