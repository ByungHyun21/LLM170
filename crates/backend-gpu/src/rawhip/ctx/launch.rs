//! ctx/launch — 런치 래퍼·KTRACE 배관·quant_q8 (ctx.rs 절단, plans/107 W5; 내용 무변경).

use super::*;

// plans/115 D: 런치 제출 비용 계측 (LLM170_DUMP=launch_time).
pub static IO_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static IO_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static IO_LAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static LT_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static LT_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn io_time_report() {
    let us = IO_US.swap(0, std::sync::atomic::Ordering::Relaxed);
    let n = IO_N.swap(0, std::sync::atomic::Ordering::Relaxed);
    if n > 0 {
        eprintln!(
            "[iotime] sync-I/O {n} calls {:.1}ms (avg {:.0}us, last {})",
            us as f64 / 1e3,
            us as f64 / n as f64,
            IO_LAST.load(std::sync::atomic::Ordering::Relaxed)
        );
    }
}

pub fn launch_time_report() {
    let us = LT_US.swap(0, std::sync::atomic::Ordering::Relaxed);
    let n = LT_N.swap(0, std::sync::atomic::Ordering::Relaxed);
    if n > 0 {
        eprintln!(
            "[ltime] 제출 {n}회 총 {us}µs (평균 {:.1}µs)",
            us as f64 / n as f64
        );
    }
}

impl RawCtx {
    /// KTRACE 전용 이벤트 마커 — launch3를 거치지 않는 직접 런치 경로용.
    pub(super) fn ktr_mark(&self, name: &'static str, gy: u32) {
        self.ktr_ev(name, gy, self.stream);
    }

    /// KTRACE 이벤트 push — (런치 전) 표준 블록 8복제 통합(plans/109 P9).
    fn ktr_ev(&self, name: &'static str, gy: u32, stream: hip::hipStream_t) {
        if self.capturing.load(std::sync::atomic::Ordering::Relaxed) {
            return; // 캡처 중 이벤트 기록 skip(그래프 노드 불허)
        }
        if let Some(mut g) = crate::rawhip::ktrace_active() {
            let mut ev: hip::hipEvent_t = std::ptr::null_mut();
            unsafe {
                hip::hipEventCreateWithFlags(&mut ev, 0);
                hip::hipEventRecord(ev, stream);
            }
            g.as_mut().unwrap().push(KtraceEv(name, ev as usize, gy));
        }
    }

    /// 커널 런치 — args는 각 인자 값에 대한 포인터 배열 (호출자 슬롯 유지).
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        name: &'static str,
        gx: u32,
        gy: u32,
        block: u32,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<(), String> {
        // 진단(LLM170_NOLAUNCH): 런치를 건너뛰고 호스트 스켈레톤 시간만 측정한다.
        // 그래프 재생 중에도 즉시 반환한다(커널은 그래프가 실행).
        if nolaunch_on() {
            return Ok(());
        }
        let f = *self
            .fns
            .get(name)
            .ok_or_else(|| format!("커널 없음: {name}"))?;
        unsafe {
            self.ktr_ev(name, gy, self.stream);
            ck(
                hip::hipModuleLaunchKernel(
                    f,
                    gx,
                    gy,
                    1,
                    block,
                    1,
                    1,
                    0,
                    self.cur_stream(),
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "launch",
            )
            .map_err(|e| format!("{e} kern={name} gx={gx} blk={block}"))?;
            self.ktr_ev(name, gy, self.stream);
        }
        if llm170_diag::dump::opts().key("launch_time") {
            LT_US.fetch_add(0, std::sync::atomic::Ordering::Relaxed); // 2D는 극소
        }
        Ok(())
    }

    /// 3차원 그리드 런치 (qsa용 — gy 추가).
    #[allow(clippy::too_many_arguments)]
    pub fn launch3(
        &self,
        name: &'static str,
        gx: u32,
        gy: u32,
        gz: u32,
        block: u32,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<(), String> {
        let lt_on = llm170_diag::dump::opts().key("launch_time");
        let lt_t0 = std::time::Instant::now();
        if lt_on {
            _ = lt_t0;
        }
        if env_on("LLM170_LAUNCH_BT") {
            eprintln!("[lbt] {name} gx={gx} gy={gy} gz={gz}");
        }

        if nolaunch_on() {
            return Ok(());
        }
        if env_on("LLM170_KT_NAMES") {
            eprintln!("# KT3 {name} gy={gy}");
        }
        let f = *self
            .fns
            .get(name)
            .ok_or_else(|| format!("커널 없음: {name}"))?;
        unsafe {
            self.ktr_ev(name, gy, self.stream);
            ck(
                hip::hipModuleLaunchKernel(
                    f,
                    gx,
                    gy,
                    gz,
                    block,
                    1,
                    1,
                    0,
                    self.cur_stream(),
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "launch3",
            )
            .map_err(|e| format!("{e} kern={name} gx={gx} gy={gy} gz={gz} blk={block}"))?;
            self.ktr_ev(name, gy, self.stream);
        }
        if lt_on {
            LT_US.fetch_add(
                lt_t0.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            LT_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    }

    /// 사이드 스트림 발사 (비동기 — join2로 합류)
    /// 동적 shared 64KB 런치 (부록82) — 커널당 1회 속성 설정.
    pub fn launch3_dyn(
        &self,
        name: &'static str,
        gx: u32,
        gy: u32,
        gz: u32,
        block: u32,
        smem: u32,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<(), String> {
        let lt_on = llm170_diag::dump::opts().key("launch_time");
        let lt_t0 = std::time::Instant::now();
        if nolaunch_on() {
            return Ok(());
        }
        use std::collections::HashSet;
        use std::sync::OnceLock;
        static SET: OnceLock<std::sync::Mutex<HashSet<usize>>> = OnceLock::new();
        let f = *self
            .fns
            .get(name)
            .ok_or_else(|| format!("커널 없음: {name}"))?;
        let set = SET.get_or_init(|| std::sync::Mutex::new(HashSet::new()));
        {
            let mut g = set.lock().map_err(|e| e.to_string())?;
            if g.insert(f as usize) {
                unsafe {
                    let r = hip::hipFuncSetAttribute(
                        f as *const std::ffi::c_void,
                        hip::hipFuncAttribute_hipFuncAttributeMaxDynamicSharedMemorySize,
                        smem as i32,
                    );
                    if r != hip::hipError_t_hipSuccess {
                        return Err(format!("smem 속성 실패: {r:?}"));
                    }
                }
            }
        }
        unsafe {
            // KTRACE 이벤트 짝 — launch3와 동일(2026-09-14 plans/69: 이 기록이
            // 없어 qsa_flash_wmma의 실행 시간이 'kv_f16 뒤 갭 12s'로 위장,
            // 8.4× 프리필 격차의 원인 파악을 하루 종일 돌렸다).
            self.ktr_ev(name, gy, self.stream);
            ck(
                hip::hipModuleLaunchKernel(
                    f,
                    gx,
                    gy,
                    gz,
                    block,
                    1,
                    1,
                    smem,
                    self.stream,
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "launch3_dyn",
            )?;
            self.ktr_ev(name, gy, self.stream);
        }
        if lt_on {
            LT_US.fetch_add(
                lt_t0.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            LT_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    }

    pub fn launch3s(
        &self,
        name: &'static str,
        gx: u32,
        gy: u32,
        gz: u32,
        block: u32,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<(), String> {
        let f = *self
            .fns
            .get(name)
            .ok_or_else(|| format!("커널 없음: {name}"))?;
        unsafe {
            // KTRACE 훅 — 프레임 경로의 dense GEMM이 전부 이 경로(stream2)를 쓴다.
            // 훅이 없어 프레임 트레이스에서 통째로 누락되던 버그(2026-09-14).
            self.ktr_ev(name, gy, self.cur_side());
            ck(
                hip::hipModuleLaunchKernel(
                    f,
                    gx,
                    gy,
                    gz,
                    block,
                    1,
                    1,
                    0,
                    self.cur_side(),
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "launch3s",
            )?;
            self.ktr_ev(name, gy, self.cur_side());
        }
        Ok(())
    }
    /// 사이드 스트림 → 주 스트림 합류: 이벤트 경유
    pub fn join2(&self) -> Result<(), String> {
        unsafe {
            let mut ev: hip::hipEvent_t = std::ptr::null_mut();
            ck(hip::hipEventCreateWithFlags(&mut ev, 0), "evCreate")?;
            ck(hip::hipEventRecord(ev, self.cur_side()), "evRecord")?;
            ck(hip::hipStreamWaitEvent(self.cur_stream(), ev, 0), "evWait")?;
            ck(hip::hipEventDestroy(ev), "evDestroy")?;
        }
        Ok(())
    }
    /// 주 스트림 현재 시점 → 사이드 대기 (사이드 입력 준비 경합 방지)
    pub fn side_wait_main(&self) -> Result<(), String> {
        unsafe {
            let mut ev: hip::hipEvent_t = std::ptr::null_mut();
            ck(hip::hipEventCreateWithFlags(&mut ev, 0), "evCreate2")?;
            ck(hip::hipEventRecord(ev, self.stream), "evRecord2")?;
            ck(hip::hipStreamWaitEvent(self.stream2, ev, 0), "evWait2")?;
            ck(hip::hipEventDestroy(ev), "evDestroy2")?;
        }
        Ok(())
    }

    /// 활성 양자화 — quantize_row_q8_ref 비트 미러. 출력은 xq 하나:
    /// [0..n/4) 워드 + [n/4..n/4+n/32) d 비트(u32 편승 — 저장 경로 단일화).
    /// 버퍼 크기 (n/4 + n/32)·4 바이트 필요.
    pub fn quant_q8(&self, x: *const u8, xq: *mut u8, n: usize) -> Result<(), String> {
        self.quant_q8_b(x, xq, n, crate::rawhip::q4acc::xq_words(n), 1)
    }

    /// 배치 양자화 — t토큰 [t][n] → [t][xq_w 워드].
    pub fn quant_q8_b(
        &self,
        x: *const u8,
        xq: *mut u8,
        n: usize,
        xq_w: usize,
        t: usize,
    ) -> Result<(), String> {
        let nblk = n / 32;
        let mut x_p = x as *mut std::ffi::c_void;
        let mut xq_p = xq as *mut std::ffi::c_void;
        let mut n_a = n as i32;
        let mut xw = xq_w as i32;
        let mut args = vec![
            &mut x_p as *mut _ as *mut std::ffi::c_void,
            &mut xq_p as *mut _ as *mut std::ffi::c_void,
            &mut n_a as *mut _ as *mut std::ffi::c_void,
            &mut xw as *mut _ as *mut std::ffi::c_void,
        ];
        self.launch3(
            "quant_q8",
            nblk.div_ceil(64) as u32,
            t as u32,
            1,
            64,
            &mut args,
        )
    }
}

/// 진단(LLM170_NOLAUNCH): 런치를 건너뛰고 호스트 스켈레톤 시간만 측정.
pub(crate) fn nolaunch_on() -> bool {
    *NOLAUNCH
}

static NOLAUNCH: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var_os("LLM170_NOLAUNCH").is_some());
