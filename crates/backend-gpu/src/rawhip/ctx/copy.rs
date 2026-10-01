//! ctx/copy — h2d/d2h·비동기·스테이지 복사 + pinned 스테이징 (ctx.rs 절단, plans/107 W5; 내용 무변경).

use super::*;

pub static IO_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static IO_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static IO_LAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl RawCtx {
    /// 사이드 스트림 비동기 h2d — 메인 스트림 작업과 중첩시킨 뒤 join2로 합류.
    pub fn h2d_async_s(&self, dst: *mut u8, src: &[u8]) -> Result<(), String> {
        unsafe {
            ck(
                hip::hipMemcpyAsync(
                    dst as *mut _,
                    src.as_ptr() as *const _,
                    src.len(),
                    hip::hipMemcpyKind_hipMemcpyHostToDevice,
                    self.stream2,
                ),
                "h2d-async-s",
            )
        }
    }

    /// 메인 스트림 비동기 h2d — 발사 순서가 커널과 같은 큐를 따라야 하는
    /// 소형 업로드(플레 임베딩 등)용. 동기화 없음(plans/73).
    pub fn h2d_async_m(&self, dst: *mut u8, src: &[u8]) -> Result<(), String> {
        unsafe {
            ck(
                hip::hipMemcpyAsync(
                    dst as *mut _,
                    src.as_ptr() as *const _,
                    src.len(),
                    hip::hipMemcpyKind_hipMemcpyHostToDevice,
                    self.stream,
                ),
                "h2d-async-m",
            )
        }
    }
    /// 디바이스→디바이드 복사(메인 스트림) — QSA KV 상주 풀 append용(plans/67 3단계).
    pub fn d2d(&self, dst: *mut u8, src: *const u8, bytes: usize) -> Result<(), String> {
        unsafe {
            ck(
                hip::hipMemcpyAsync(
                    dst as *mut _,
                    src as *const _,
                    bytes,
                    hip::hipMemcpyKind_hipMemcpyDeviceToDevice,
                    self.stream,
                ),
                "d2d",
            )
        }
    }
    pub fn h2d(&self, dst: *mut u8, src: &[u8]) -> Result<(), String> {
        unsafe {
            // 실패 시 크기·목적지·**호출 지점**을 남긴다. 상한 가정이 여러 곳에
            // 흩어진 경로에서 "h2d: 700"만으로는 어느 복사인지 알 수 없었고,
            // 크기만으로도 부족했다(2026-09-14). 백트레이스는 강제로 잡는다
            // (RUST_BACKTRACE 미설정이어도 동작).
            let tag = format!("h2d {}B dst={dst:p}", src.len());
            // LLM170_MEMDBG: 복사 **직전** 여유 메모리(사후 조회는 sticky 오류로 0/0).
            if env_on("LLM170_MEMDBG") && src.len() >= (1 << 20) {
                let (mut fb, mut tb) = (0usize, 0usize);
                let _ = hip::hipMemGetInfo(&mut fb, &mut tb);
                eprintln!(
                    "# memdbg before {tag}: free={}MB/{}MB",
                    fb / 1048576,
                    tb / 1048576
                );
            }
            if let Err(e) = ck(
                hip::hipMemcpyAsync(
                    dst as *mut _,
                    src.as_ptr() as *const _,
                    src.len(),
                    hip::hipMemcpyKind_hipMemcpyHostToDevice,
                    self.stream,
                ),
                &tag,
            ) {
                // 사후 hipMemGetInfo는 sticky 오류 때문에 0을 돌려준다(확인함) —
                // 메모리 진단이 필요하면 이 h2d **전에** 조회해야 한다.
                let span = self.alloc_span(dst as usize, src.len());
                let bt = std::backtrace::Backtrace::force_capture();
                let frames: Vec<String> = format!("{bt}")
                    .lines()
                    .filter(|l| l.contains("llm170") && !l.contains("backtrace"))
                    .take(4)
                    .map(|l| l.trim().to_string())
                    .collect();
                return Err(format!("{e} | dst {span} | 호출: {}", frames.join(" <- ")));
            }
            if llm170_diag::dump::opts().key("io_time") {
                let t0 = std::time::Instant::now();
                self.sync()?;
                IO_US.fetch_add(
                    t0.elapsed().as_micros() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
                IO_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                IO_LAST.store(1, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            } else {
                self.sync()
            }
        }
    }

    /// 비동기 d2h용 이벤트(스텝 간 재사용) — 한 스트림에 순서대로 걸린다.
    fn d2h_ev(&self) -> Result<hip::hipEvent_t, String> {
        static EV: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let p = *EV.get_or_init(|| {
            let mut e: hip::hipEvent_t = std::ptr::null_mut();
            unsafe {
                let _ = hip::hipEventCreateWithFlags(&mut e, 0);
            }
            e as usize
        });
        Ok(p as hip::hipEvent_t)
    }

    /// 비동기 d2h 예약 — 핀 버퍼로 스트림 복사만 걸고 sync하지 않는다.
    /// 그 사이 커널을 계속 발사해 전송·대기 지연을 다른 연산 뒤에 숨기고,
    /// 필요해지는 지점에서 `d2h_wait()`로 완료를 기다린다. 반환 = 핀 버퍼.
    pub fn d2h_issue(&self, need: usize, src: *const u8) -> Result<*mut u8, String> {
        unsafe {
            let mut pin = self.pinned_a.lock().map_err(|e| e.to_string())?;
            if pin.0 < need {
                let mut p: *mut std::os::raw::c_void = std::ptr::null_mut();
                if hip::hipMallocHost(&mut p, need) == hip::hipError_t_hipSuccess {
                    *pin = (need, p as *mut u8);
                } else {
                    *pin = (0, std::ptr::null_mut());
                }
            }
            let buf = pin.1;
            if buf.is_null() {
                return Ok(std::ptr::null_mut());
            }
            ck(
                hip::hipMemcpyAsync(
                    buf as *mut _,
                    src as *const _,
                    need,
                    hip::hipMemcpyKind_hipMemcpyDeviceToHost,
                    self.stream,
                ),
                "d2h-issue",
            )?;
            let ev = self.d2h_ev()?;
            ck(hip::hipEventRecord(ev, self.stream), "d2h-ev")?;
            Ok(buf)
        }
    }

    /// `d2h_issue` 완료 대기 — **복사 이벤트만** 기다린다(스트림 전체를 비우지
    /// 않으므로 그 뒤에 큐잉된 커널은 계속 진행된다). 파이프라인을 살려 두는
    /// 것이 이 API의 존재 이유다(2026-09-14: 전체 sync는 층마다 파이프라인을
    /// 비워 shared 스테이지가 25.8 → 50.7ms로 두 배가 됐다).
    pub fn d2h_wait(&self) -> Result<(), String> {
        unsafe {
            let ev = self.d2h_ev()?;
            ck(hip::hipEventSynchronize(ev), "d2h-ev-wait")
        }
    }

    pub fn d2h(&self, dst: &mut [u8], src: *const u8) -> Result<(), String> {
        unsafe {
            // pageable 직행은 슬로패스 — 핀 스테이징 경유 (2026-09-05 tg RCA:
            // logits 1MB D2H가 92ms → 핀 경유 시 <1ms 예상)
            let need = dst.len();
            // 핀 스테이징 실패(호스트 메모리 압박 등)는 조용히 페이지어블 경로로
            // 폴백한다 — 폴트 여부를 가리는 대신 판독은 성공시킨다(2026-09-12:
            // 장문 청크에서 hipMallocHost: 700으로 판독이 통째로 실패).
            let mut pin = self.pinned.lock().map_err(|e| e.to_string())?;
            if pin.0 < need {
                let mut p: *mut std::os::raw::c_void = std::ptr::null_mut();
                if hip::hipMallocHost(&mut p, need) == hip::hipError_t_hipSuccess {
                    *pin = (need, p as *mut u8);
                } else {
                    *pin = (0, std::ptr::null_mut());
                }
            }
            if !pin.1.is_null() {
                let buf = pin.1;
                ck(
                    hip::hipMemcpyAsync(
                        buf as *mut _,
                        src as *const _,
                        need,
                        hip::hipMemcpyKind_hipMemcpyDeviceToHost,
                        self.stream,
                    ),
                    "d2h-pin",
                )?;
                if llm170_diag::dump::opts().key("io_time") {
                    let t0 = std::time::Instant::now();
                    ck(hip::hipStreamSynchronize(self.stream), "d2h-sync")?;
                    super::launch::IO_US.fetch_add(
                        t0.elapsed().as_micros() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    super::launch::IO_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    super::launch::IO_LAST.store(2, std::sync::atomic::Ordering::Relaxed);
                } else {
                    ck(hip::hipStreamSynchronize(self.stream), "d2h-sync")?;
                }
                std::ptr::copy_nonoverlapping(buf, dst.as_mut_ptr(), need);
                return Ok(());
            }
            ck(
                hip::hipMemcpy(
                    dst.as_mut_ptr() as *mut std::os::raw::c_void,
                    src as *const std::os::raw::c_void,
                    need,
                    hip::hipMemcpyKind_hipMemcpyDeviceToHost,
                ),
                "d2h-pageable",
            )
        }
    }
    /// 핀 스테이지 2버퍼(107 W1.5-3) — async H2D가 pageable 경유
    /// 스테이징 없이 디바이스로 직행하도록. 반환 (a, b): 실패 시 null.
    /// 크기는 요청치 이상 보유(성장 재할당). 해제는 Drop이 책임.
    pub fn pinned_stage2(&self, need: usize) -> Result<(*mut u8, *mut u8), String> {
        // SAFETY (107 W8): 핀 스테이지 이중 버퍼 — 재할당은 이전 포인터 hipFreeHost 후; (base, base+need)는 항상 2*need 할당의 두 절반. 호출부는 다음 pinned_stage2(재할당) 전에 사용을 마쳐야 한다.
        unsafe {
            let mut pin = self.pinned_stage.lock().map_err(|e| e.to_string())?;
            if pin.0 < 2 * need {
                if !pin.1.is_null() {
                    let _ = hip::hipFreeHost(pin.1 as *mut _);
                }
                let mut p: *mut std::os::raw::c_void = std::ptr::null_mut();
                ck(hip::hipMallocHost(&mut p, 2 * need), "pinStage")?;
                *pin = (2 * need, p as *mut u8);
            }
            let base = pin.1;
            Ok((base, base.add(need)))
        }
    }

    ///
    /// # Safety
    /// `ev`는 유효한 이벤트 핸들이어야 한다.
    pub unsafe fn ev_sync(ev: hip::hipEvent_t) -> Result<(), String> {
        unsafe { ck(hip::hipEventSynchronize(ev), "evSync") }
    }
}
