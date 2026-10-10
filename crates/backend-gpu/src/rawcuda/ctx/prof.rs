use super::*;

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
