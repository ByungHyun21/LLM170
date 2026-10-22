
use super::*;

pub(super) fn realloc_fields<const N: usize>(
    mut free: impl FnMut(CUdeviceptr) -> Result<(), String>,
    mut alloc: impl FnMut(usize) -> Result<CUdeviceptr, String>,
    mut fields: [&mut CUdeviceptr; N],
    sizes: [usize; N],
) -> Result<(), String> {
    for f in fields.iter() {
        if **f != 0 {
            free(**f)?;
        }
    }
    for f in fields.iter_mut() {
        **f = 0;
    }
    for (f, sz) in fields.into_iter().zip(sizes) {
        *f = alloc(sz)?;
    }
    Ok(())
}

impl W4a16Dec {
    pub(crate) fn h2d_chunked(cc: &CudaCtx, dst: CUdeviceptr, src: &[u8]) -> Result<(), String> {
        const CH: usize = 8 << 20;
        const INFLIGHT: usize = 64 << 20;
        let mut pending = 0usize;
        for (i, chunk) in src.chunks(CH).enumerate() {
            cc.h2d_async(dst + (i * CH) as u64, chunk)?;
            pending += chunk.len();
            if pending >= INFLIGHT {
                cc.sync()?;
                pending = 0;
            }
            llm170_diag::watchdog::bump();
        }
        cc.sync()
    }

    pub(super) fn zero_dev(cc: &CudaCtx, ptr: CUdeviceptr, len: usize) -> Result<(), String> {
        cc.memset0_async(ptr, len)
    }

    pub fn mem_stats(&self) -> (u64, u64, u64, u64) {
        let kv = if self.attn.kv_arena_bytes > 0 {
            self.attn.kv_arena_bytes
        } else {
            self.attn
                .attn
                .map(|d| {
                    let rows = (self.n_slots as u64) * (d.n_attn as u64) * (d.cap as u64);
                    rows * (d.kv_dim() as u64) + rows * (d.kv_heads as u64) * 4 * 2
                })
                .unwrap_or(0)
        };
        let experts = self.experts_bytes;
        let (w_gpu, w_cpu) = if self.moe.n_experts > 0 && !self.moe.moe_resident {
            (self.weights_bytes, experts)
        } else {
            (self.weights_bytes, 0)
        };
        (w_gpu, kv, w_cpu, 0)
    }

    pub fn active_weight_bytes(&self) -> u64 {
        let experts = self.experts_bytes;
        let base = self.weights_bytes.saturating_sub(self.marlin_bytes);
        if self.moe.n_experts > 0 && self.moe.moe_resident {
            let unsel =
                experts / self.moe.n_experts as u64 * (self.moe.n_experts - self.moe.top_k) as u64;
            base.saturating_sub(unsel)
        } else {
            base
        }
    }

    pub fn copy_stats(&self) -> [(u64, u64, u64); 3] {
        self.cc.copy_stats()
    }

    pub fn alloc_pinned_scratch(&self, bytes: usize) -> Result<*mut std::ffi::c_void, String> {
        self.cc.pinned_alloc(bytes)
    }

    pub fn free_pinned_scratch(&self, p: *mut std::ffi::c_void) -> Result<(), String> {
        self.cc.pinned_free(p)
    }

    pub fn alloc_scratch(&self, bytes: usize) -> Result<CUdeviceptr, String> {
        self.cc.alloc(bytes)
    }

    pub fn free_scratch(&self, p: CUdeviceptr) -> Result<(), String> {
        self.cc.free(p)
    }

    pub fn h2d_scratch(&self, dst: CUdeviceptr, src: &[u8]) -> Result<(), String> {
        self.cc.h2d(dst, src)
    }

    pub fn sync_bench(&self) -> Result<(), String> {
        self.cc.sync()
    }

    pub fn prof_report(&mut self, label: &str) -> Result<String, String> {
        let _g = self.cc.guard()?;
        self.cc.prof_report(label)
    }
}

pub(crate) struct PinnedRing<'a> {
    cc: &'a CudaCtx,
    a: *mut std::ffi::c_void,
    b: *mut std::ffi::c_void,
    ev_a: ffi::CUevent,
    ev_b: ffi::CUevent,
    chunk: usize,
    seq: u64,
}

impl<'a> PinnedRing<'a> {
    pub(crate) fn new(cc: &'a CudaCtx, chunk: usize) -> Result<Self, String> {
        let a = cc.pinned_alloc(chunk)?;
        let b = match cc.pinned_alloc(chunk) {
            Ok(p) => p,
            Err(e) => {
                let _ = cc.pinned_free(a);
                return Err(e);
            }
        };
        unsafe {
            std::ptr::write_bytes(a, 0, chunk);
            std::ptr::write_bytes(b, 0, chunk);
        }
        let ev_a = match cc.event_create_nt() {
            Ok(e) => e,
            Err(e) => {
                let _ = cc.pinned_free(a);
                let _ = cc.pinned_free(b);
                return Err(e);
            }
        };
        let ev_b = match cc.event_create_nt() {
            Ok(e) => e,
            Err(e) => {
                let _ = cc.pinned_free(a);
                let _ = cc.pinned_free(b);
                cc.event_destroy(ev_a);
                return Err(e);
            }
        };
        Ok(Self {
            cc,
            a,
            b,
            ev_a,
            ev_b,
            chunk,
            seq: 0,
        })
    }

    pub(crate) fn send(
        &mut self,
        dst: CUdeviceptr,
        bytes: usize,
        fill: &mut dyn FnMut(u64, &mut [u8]) -> Result<(), String>,
    ) -> Result<(), String> {
        let mut off = 0usize;
        while off < bytes {
            let n = self.chunk.min(bytes - off);
            let (p, ev) = if self.seq.is_multiple_of(2) {
                (self.a, self.ev_a)
            } else {
                (self.b, self.ev_b)
            };
            if self.seq >= 2 {
                self.cc.event_sync(ev)?;
            }
            let buf = unsafe { std::slice::from_raw_parts_mut(p as *mut u8, n) };
            fill(off as u64, buf)?;
            let src = unsafe { std::slice::from_raw_parts(p as *const u8, n) };
            self.cc.h2d_async(dst + off as u64, src)?;
            self.cc.event_record(ev)?;
            self.seq += 1;
            off += n;
            llm170_diag::watchdog::bump();
        }
        Ok(())
    }

    pub(crate) fn sync(&self) -> Result<(), String> {
        self.cc.sync()
    }
}

impl Drop for PinnedRing<'_> {
    fn drop(&mut self) {
        self.cc.event_destroy(self.ev_a);
        self.cc.event_destroy(self.ev_b);
        let _ = self.cc.pinned_free(self.a);
        let _ = self.cc.pinned_free(self.b);
    }
}
