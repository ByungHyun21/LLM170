//! KTRACE — 커널별 GPU 시간 계측(자체 hipEvent 페어링). 외부 GPU 프로파일러 금지 정책에
//! 따라 엔진 자체 계측을 쓴다. 켜면 런치마다 이벤트 2개를 기록하고 dump 에서 정리한다.

use crate::rawhip::hip;
#[allow(unused_imports)]
use crate::rawhip::ck;

pub fn ktrace_dump() -> String {
    let mut g = KTRACE.lock().unwrap();
    // ktrace_on 없이 호출되면(스펙 경로 등) 빈 문자열 — 과거 unwrap 패닉
    let Some(slot) = g.as_mut() else { return String::new() };
    let evs = std::mem::take(slot);
    // 이벤트 핸들 정리 — 파괴하지 않으면 hipEvent 풀이 고갈되어(런치당 2개 생성,
    // 13k 런치) 이후 생성이 실패하고 트레이스에서 통째로 누락된다(2026-09-14 규명).
    struct Evs(Vec<KtraceEv>);
    impl Drop for Evs {
        fn drop(&mut self) {
            unsafe {
                for e in &self.0 {
                    let _ = hip::hipEventDestroy(e.1 as *mut _);
                }
            }
        }
    }
    let _evs_guard = Evs(evs);
    let evs = &_evs_guard.0;
    let mut out = String::new();
    // 쌍 결합: 연속 동일 (name, gy) 두 이벤트가 start/end
    let mut sums: std::collections::HashMap<(&str, u32), (f64, u32)> = std::collections::HashMap::new();
    let mut total = 0.0f64;
    let mut gaps = 0.0f64;
    // 쌍은 **고정 stride-2**다: 런치마다 (start, end)를 정확히 2개 기록한다.
    // 종전 휴리스틱("연속 같은 (name,gy) = 쌍")은 같은 커널이 연속 런치될 때
    // end→start를 한 쌍으로 묶어 합계를 통째로 어긋나게 했다 — 배치 형상에서
    // 흔하고, 27B pp512에서 합계 366ms 대 벽 1,409ms(4배 과소)로 나타났다
    // (2026-09-14 규명). 갭도 같은 순회에서 end(k) → start(k+1)로 잰다.
    let npair = evs.len() / 2;
    unsafe {
        for k in 0..npair {
            let (st, en) = (&evs[2 * k], &evs[2 * k + 1]);
            // 짝이 어긋난 런치(다른 커널과 섞임)면 방어적으로 건너뛴다.
            if st.0 != en.0 || st.2 != en.2 {
                continue;
            }
            let mut ms = 0f32;
            if hip::hipEventElapsedTime(&mut ms, st.1 as *mut _, en.1 as *mut _) == hip::hipError_t_hipSuccess {
                let ent = sums.entry((st.0, st.2)).or_insert((0.0, 0));
                ent.0 += ms as f64;
                ent.1 += 1;
                total += ms as f64;
            }
            if k + 1 < npair {
                let nst = &evs[2 * (k + 1)];
                let mut gm = 0f32;
                if hip::hipEventElapsedTime(&mut gm, en.1 as *mut _, nst.1 as *mut _) == hip::hipError_t_hipSuccess {
                    gaps += gm as f64;
                }
            }
        }
        // 런치 갭: end(N)→start(N+1) 같은 스트림 상 연속
        let mut gap_by_pred: std::collections::HashMap<&str, (f64, u32)> = std::collections::HashMap::new();
        let mut gap_tot = 0.0f64;
        let mut prev_end: Option<(usize, &str)> = None;
        for k in 0..evs.len()/2 {
            let (st, en) = (&evs[2*k], &evs[2*k+1]);
            if let Some((pe, pn)) = prev_end {
                let mut ms = 0f32;
                if hip::hipEventElapsedTime(&mut ms, pe as *mut _, st.1 as *mut _) == hip::hipError_t_hipSuccess && ms > 0.0 {
                    let e2 = gap_by_pred.entry(pn).or_insert((0.0, 0));
                    e2.0 += ms as f64; e2.1 += 1;
                    gap_tot += ms as f64;
                }
            }
            prev_end = Some((en.1, en.0));
        }
        // 런치 **순서** 덤프 (LLM170_KTRACE_SEQ=N): 갭의 주인은 전임자가 아니라
        // (2026-09-14: 이 덤프로 MoE 묶음이 gather → gate GEMM → scatter → up GEMM …
        //  순서임을 확인해 그룹화 호스트 경로를 특정했다)
        // **후속 op의 호스트 비용**이므로(갭이 후속에 따라 달라진다), 순서를 봐야
        // 어떤 op인지 특정된다.
        if let Ok(v) = std::env::var("LLM170_KTRACE_SEQ") {
            let n: usize = v.parse().unwrap_or(64);
            for k in 0..npair.min(n) {
                let (st, en) = (&evs[2 * k], &evs[2 * k + 1]);
                let mut sms = 0f32;
                let _ = hip::hipEventElapsedTime(&mut sms, st.1 as *mut _, en.1 as *mut _);
                out.push_str(&format!("# seq {k:4} {:<28} gy={:<6} {:8.3}ms\n", st.0, st.2, sms));
            }
        }
        for e in evs.iter() { hip::hipEventDestroy(e.1 as *mut _); }
        out.push_str(&format!("LAUNCH GAPS total {:.1}ms\n", gap_tot));
        let mut gv: Vec<_> = gap_by_pred.iter().collect();
        gv.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
        for (n, (ms, c)) in gv.iter().take(12) {
            out.push_str(&format!("  after {:26} {:8.1}ms x{:4}\n", n, ms, c));
        }
    }
    let mut v: Vec<_> = sums.iter().collect();
    v.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
    for ((n, gy), (ms, cnt)) in v.iter().take(40) {
        out.push_str(&format!("{:30} gy={:4} {:9.3}ms x{:4}\n", n, gy, ms, cnt));
    }
    out.push_str(&format!("TOTAL {:.1}ms GAPS {:.1}ms\n", total, gaps));
    out
}

pub fn ktrace_on() { *KTRACE.lock().unwrap() = Some(Vec::new()); }

pub fn aout_dumped() -> bool {
    let r = AOUT_DUMPED.swap(true, std::sync::atomic::Ordering::SeqCst);
    !r
}

static AOUT_DUMPED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub struct KtraceEv(pub &'static str, pub usize, pub u32);  // name, event, gy
pub static KTRACE: std::sync::Mutex<Option<Vec<KtraceEv>>> = std::sync::Mutex::new(None);

