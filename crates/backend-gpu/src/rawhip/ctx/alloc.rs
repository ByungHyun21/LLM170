//! ctx/alloc — 디바이스 아레나 alloc (alloc/alloc_span) (ctx.rs 절단, plans/107 W5; 내용 무변경).

use super::*;

impl RawCtx {
    /// 영속 디바이스 할당 (해제 없음).
    pub fn alloc(&self, bytes: usize) -> Result<*mut u8, String> {
        let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            let r = hip::hipMalloc(&mut p, bytes);
            if r != hip::hipError_t_hipSuccess {
                eprintln!("alloc {bytes}B → {r:?}");
            }
            ck(r, "hipMalloc")?;
        }
        if let Ok(mut v) = self.allocs.lock() {
            v.push((p as usize, p as usize + bytes));
        }
        Ok(p as *mut u8)
    }

    /// `p`가 살아있는 할당 안인지(그리고 몇 바이트 남았는지) — h2d 실패 진단용.
    pub(super) fn alloc_span(&self, p: usize, need: usize) -> String {
        let Ok(v) = self.allocs.lock() else {
            return "등록부 잠금 실패".into();
        };
        for &(s, e) in v.iter() {
            if p >= s && p < e {
                return if p + need <= e {
                    format!("할당 내 [{s:#x},{e:#x})")
                } else {
                    format!("할당 경계 초과! [{s:#x},{e:#x}) +{}B", p + need - e)
                };
            }
        }
        format!("할당 밖 (등록 {}개)", v.len())
    }
}
