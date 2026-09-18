//! KTRACE — 커널별 GPU 시간 계측(자체 hipEvent 페어링). 외부 GPU 프로파일러 금지 정책에
//! 따라 엔진 자체 계측을 쓴다. 켜면 런치마다 이벤트 2개를 기록하고 dump 에서 정리한다.
//!
//! plans/83 C2: 이 모듈은 **diag 이벤트 어댑터**다 — hipEvent 쌍을 resolved
//! `diag::trace::Ev`(start_ms/dur_ms/lane)로 변환해 공유 진단 계층에 맡기고,
//! 출력은 `diag::writer::dump`가 담당한다. 분석·포맷의 단일 진실 공급원.

use crate::rawhip::hip;
use llm170_diag::trace::Ev;

pub fn ktrace_dump() -> String {
    let mut g = KTRACE.lock();
    // ktrace_on 없이 호출되면(스펙 경로 등) 빈 문자열
    let Some(slot) = g.as_mut() else { return String::new() };
    let evs = std::mem::take(slot);
    // 이벤트 핸들 정리 — 파괴하지 않으면 hipEvent 풀이 고갈되어(런치당 2개 생성,
    // 13k 런치) 이후 생성이 실패하고 트레이스에서 통째로 누락된다(2026-09-14 규명).
    struct Evs(Vec<KtraceEv>);
    impl Drop for Evs {
        fn drop(&mut self) {
            unsafe {
                for e in self.0.iter() {
                    hip::hipEventDestroy(e.1 as *mut _);
                }
            }
        }
    }
    let _evs_guard = Evs(evs);
    let evs = &_evs_guard.0;

    // 짝 결합 → Ev 변환. 쌍은 **고정 stride-2**: 런치마다 (start, end) 정확히
    // 2개 기록. 짝이 어긋난 런치(다른 커널과 섞임)는 방어적으로 건너뛴다.
    // 절대 시작 시각: start(k+1) = start(k) + dur(k) + gap(k) 누적 —
    // hipEventElapsedTime은 쌍 간 상대치만 제공하므로.
    let npair = evs.len() / 2;
    let mut out: Vec<Ev> = Vec::with_capacity(npair);
    let mut t = 0.0f64;
    unsafe {
        let mut k = 0usize;
        while k < npair {
            let (st, en) = (&evs[2 * k], &evs[2 * k + 1]);
            k += 1;
            if st.0 != en.0 || st.2 != en.2 {
                continue;
            }
            let mut dur = 0f32;
            if hip::hipEventElapsedTime(&mut dur, st.1 as *mut _, en.1 as *mut _) != hip::hipError_t_hipSuccess {
                continue;
            }
            out.push(Ev {
                name: st.0,
                lane: st.2,
                start_ms: t,
                dur_ms: dur as f64,
                gap_next_ms: None,
                seq_ms: 0.0,
            });
            t += dur as f64;
            // 다음 쌍까지의 갭 적립(짝 어긋남 건너뛰기 포함)
            while k < npair {
                let nst = &evs[2 * k];
                if nst.0 == evs[2 * k + 1].0 && nst.2 == evs[2 * k + 1].2 {
                    let mut gm = 0f32;
                    if hip::hipEventElapsedTime(&mut gm, en.1 as *mut _, nst.1 as *mut _) == hip::hipError_t_hipSuccess {
                        t += gm as f64;
                    }
                    break;
                }
                k += 1;
            }
        }
    }
    let mut evs = out;
    llm170_diag::trace::resolve(&mut evs);
    llm170_diag::writer::dump(&evs, 0)
}

pub fn ktrace_on() { *KTRACE.lock() = Some(Vec::new()); }

pub fn aout_dumped() -> bool {
    let r = AOUT_DUMPED.swap(true, std::sync::atomic::Ordering::SeqCst);
    r
}

static AOUT_DUMPED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub struct KtraceEv(pub &'static str, pub usize, pub u32);  // name, event, gy
pub static KTRACE: parking_lot::Mutex<Option<Vec<KtraceEv>>> = parking_lot::Mutex::new(None);

/// 활성 KTRACE 슬롯 가드 — 런치 훅용. 녹화 중이 아니면 None.
pub fn ktrace_active() -> Option<parking_lot::MutexGuard<'static, Option<Vec<KtraceEv>>>> {
    let g = KTRACE.lock();
    g.is_some().then_some(g)
}
