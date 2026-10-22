use super::*;

impl CudaCtx {
    pub fn event_create(&self) -> Result<ffi::CUevent, String> {
        let mut ev: ffi::CUevent = std::ptr::null_mut();
        let r = unsafe { (self.drv.event_create)(&mut ev, 0) };
        if r != CUDA_SUCCESS {
            return Err(format!("rawcuda: cuEventCreate: {}", ffi::err_text(r)));
        }
        Ok(ev)
    }

    pub fn event_create_nt(&self) -> Result<ffi::CUevent, String> {
        let mut ev: ffi::CUevent = std::ptr::null_mut();
        let r = unsafe { (self.drv.event_create)(&mut ev, 2) };
        if r != CUDA_SUCCESS {
            return Err(format!("rawcuda: cuEventCreate(nt): {}", ffi::err_text(r)));
        }
        Ok(ev)
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn event_record_on(&self, ev: ffi::CUevent, s: ffi::CUstream) -> Result<(), String> {
        let r = unsafe { (self.drv.event_record)(ev, s) };
        if r != CUDA_SUCCESS {
            return Err(format!("rawcuda: cuEventRecord: {}", ffi::err_text(r)));
        }
        Ok(())
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn event_record(&self, ev: ffi::CUevent) -> Result<(), String> {
        let r = unsafe { (self.drv.event_record)(ev, self.stream) };
        if r != CUDA_SUCCESS {
            return Err(format!("rawcuda: cuEventRecord: {}", ffi::err_text(r)));
        }
        Ok(())
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn event_destroy(&self, ev: ffi::CUevent) {
        unsafe {
            let _ = (self.drv.event_destroy)(ev);
        }
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn event_sync(&self, ev: ffi::CUevent) -> Result<(), String> {
        let r = unsafe { (self.drv.event_synchronize)(ev) };
        if r != CUDA_SUCCESS {
            return Err(format!("rawcuda: cuEventSynchronize: {}", ffi::err_text(r)));
        }
        Ok(())
    }

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

    pub fn prof_report(&self, label: &str) -> Result<String, String> {
        if !self.prof.on {
            return Ok(String::new());
        }
        let sentinel = self.event_create()?;
        self.event_record(sentinel)?;
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
        out.push_str(&format!(
            " · [host] {:.2}ms/{}회(평균 {:.1}µs)",
            HOST_LAUNCH_NS.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6,
            HOST_LAUNCH_CALLS.load(std::sync::atomic::Ordering::Relaxed),
            HOST_LAUNCH_NS.load(std::sync::atomic::Ordering::Relaxed) as f64
                / 1e3
                / HOST_LAUNCH_CALLS
                    .load(std::sync::atomic::Ordering::Relaxed)
                    .max(1) as f64
        ));
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
