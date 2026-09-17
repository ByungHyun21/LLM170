//! 원시 HIP 실행기 — cubecl을 거치지 않는 직접 경로 (2026-09-03 재작성).
//! hipRTC로 임베디드 HIP C++ 소스를 컴파일하고 hipModuleLaunchKernel로
//! 실행. 버퍼는 영속 아레나(해제 없음, ADR-0014 동일 규칙). 커널 산술은
//! core 미러(dot_row_w4a8_*_lane)와 동일 연산열 — to_bits 검증 게이트.
#![allow(dead_code)] // 프론트 정리(2026-09-14): 레거시·진단 경로 보존

pub mod graph;
pub mod ktrace;
pub use graph::*;
pub use ktrace::*;

use cubecl_hip_sys as hip;

pub mod decode;
pub mod kernels;
pub mod q4acc;
pub mod vit;

/// 로드된 오프라인 타일 코드오브젝트 패밀리 (임베딩 or LLM170_CO*_PATH
/// 오버라이드). RawCtx::new 완료 후 불변. 타일 발사 게이트는 env가 아니라
/// 이 비트를 본다 — 무환경 기본 성능 = 튜닝 성능.
pub const CO_J128: u8 = 1; // w32b.co: *_j128 계열 (t≤128)
pub const CO_V4: u8 = 2; // v4all.co: *_v4 + *_wm 4종
pub const CO_ODD: u8 = 4; // odd_all.co: nl/q3k/iq3s v4 (plans/04)
pub const CO_MMQ: u8 = 8; // mmq.co: llama mul_mat_q<q4_K/q5_K,128> + mmq_quant_y
pub const CO_MMQ2: u8 = 16; // mmq2.co: gemm_f16_v4 (deq-f16 경로)
pub const CO_MMQ3: u8 = 32; // mmq3.co: llama 프로덕션 mul_mat_q<iq4_xs>
pub const CO_MMQ8: u8 = 64; // mmq8.co: ROCm 10 fatbin의 mul_mat_q<q8_0>(plans/71)
pub const CO_QY: u8 = 128; // quanty_new.co: ROCm 10 quantize_mmq_q8_1<D4/DS4>(plans/71)
static CO_FAM: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

pub fn co_loaded(bit: u8) -> bool {
    CO_FAM.load(std::sync::atomic::Ordering::Relaxed) & bit != 0
}


/// MMQ mul_mat_q 동적 smem 상한 설정 캐시 — 런치마다 드라이버 호출하지 않도록.
/// (hipFuncSetAttribute는 커널 로드 갱신을 유발할 수 있어 GEMM마다 부르면 손해)
static MMQ_SMEM_SET: std::sync::LazyLock<std::sync::Mutex<std::collections::HashSet<(usize, i32)>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashSet::new()));
pub(crate) fn ck(status: hip::hipError_t, what: &str) -> Result<(), String> {
    if status == hip::hipError_t_hipSuccess {
        Ok(())
    } else {
        Err(format!("rawhip: {what}: {status:?}"))
    }
}



// ── 프레임 그래프 캡처(디코드 스텝) ────────────────────────────────────────
// 스텝 내 호스트 왕복(capture_mark)을 경계로 스트림 캡처를 세그먼트로 끊어
// 그래프로 굳히고, 재생 시에는 런치 함수가 즉시 반환되어 커널이 그래프에서
// 실행된다(런치 ~3천 회/스텝 → 세그먼트 수 회). 프로세스 전역 — CLI는 가속기
// 1개, 재생은 스텝 단위 단일 스레드라 전역으로 충분하다.
// SAFETY: 그래프 핸들은 디바이스 객체 — HIP 런타임이 직렬화하며, 재생은 스텝
// 단위로 단일 스레드에서만 일어난다.
unsafe impl Send for GraphMode {}

/// 컴파일된 커널 실행기.
pub mod ctx;
pub use ctx::*;

pub mod probes;
pub use probes::*;

#[cfg(test)]
mod micro_tests {
    use super::*;

    /// 런치 비용 분해 — 호스트 enqueue와 디바이스 디스패치를 가른다.
    /// 디코드 스텝은 ~3,222런치가 129ms를 쓰는데, 그 37µs/런치가 호스트인지
    /// 디바이스인지가 다음 수(융합 vs 배관)를 정한다.
    #[test]
    fn launch_cost_split() {
        let ctx = RawCtx::new().expect("ctx");
        let n = 1024i32;
        let a = ctx.scratch(4096).expect("a");
        let b = ctx.scratch(4096).expect("b");
        let launch = || {
            let (mut ap, mut bp) = (a as *mut std::ffi::c_void, b as *mut std::ffi::c_void);
            let (mut nn, mut rr) = (n, 1i32);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut ap) as *mut _ as *mut std::ffi::c_void,
                (&mut bp) as *mut _ as *mut std::ffi::c_void,
                (&mut nn) as *mut _ as *mut std::ffi::c_void,
                (&mut rr) as *mut _ as *mut std::ffi::c_void,
            ];
            ctx.launch3("bcast_rows", 8, 1, 1, 128, &mut args).unwrap();
        };
        for _ in 0..20 {
            launch();
        }
        ctx.sync().unwrap();
        const N: usize = 2000;
        let t0 = std::time::Instant::now();
        for _ in 0..N {
            launch();
        }
        let enq = t0.elapsed().as_secs_f64() * 1e3;
        ctx.sync().unwrap();
        let total = t0.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "# micro {N}런치: host-enqueue {enq:.1}ms ({:.2}µs/런치), sync포함 총 {total:.1}ms ({:.2}µs/런치)",
            enq / N as f64 * 1e3,
            total / N as f64 * 1e3
        );

        // 이상 커널 격리 — q4_l2_rows(d=2560, 1블록×32스레드). 디코드에서
        // 0.34ms/회로 관측된 그 커널이 정말 그런지, 아니면 측정 맥락 탓인지.
        let x = ctx.scratch(2560 * 4).expect("x");
        let launch_l2 = || {
            let mut xp = x as *mut std::ffi::c_void;
            let mut e = 1e-6f32;
            let mut d = 2560i32;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut xp) as *mut _ as *mut std::ffi::c_void,
                (&mut e) as *mut _ as *mut std::ffi::c_void,
                (&mut d) as *mut _ as *mut std::ffi::c_void,
            ];
            ctx.launch3("q4_l2_rows", 1, 1, 1, 32, &mut args).unwrap();
        };
        for _ in 0..20 {
            launch_l2();
        }
        ctx.sync().unwrap();
        let t1 = std::time::Instant::now();
        for _ in 0..N {
            launch_l2();
        }
        let enq2 = t1.elapsed().as_secs_f64() * 1e3;
        ctx.sync().unwrap();
        let total2 = t1.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "# micro q4_l2_rows(d=2560) {N}런치: host {enq2:.1}ms ({:.2}µs/런치), 총 {total2:.1}ms ({:.2}µs/런치)",
            enq2 / N as f64 * 1e3,
            total2 / N as f64 * 1e3
        );

        // GDN l2scale 시퀀스 격리 — 디코드에서 0.69ms/층으로 관측된 그 4커널
        // (split3 → l2_rows ×2 → scale). 실제로 그만큼 드는지, 맥락 탓인지 가른다.
        let conv_ch = 4096i32;
        let kv = 1024i32;
        let dstate = 128i32;
        let src = ctx.scratch(conv_ch as usize * 4).expect("src");
        let gq = ctx.scratch(kv as usize * 4).expect("gq");
        let gk = ctx.scratch(kv as usize * 4).expect("gk");
        let gv = ctx.scratch(kv as usize * 4).expect("gv");
        let launch_seq = || {
            let (mut sp, mut p0, mut p1, mut p2) = (
                src as *mut std::ffi::c_void,
                gq as *mut std::ffi::c_void,
                gk as *mut std::ffi::c_void,
                gv as *mut std::ffi::c_void,
            );
            let (mut n0, mut n1, mut n2) = (kv, kv, kv);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut sp) as *mut _ as *mut std::ffi::c_void,
                (&mut p0) as *mut _ as *mut std::ffi::c_void,
                (&mut p1) as *mut _ as *mut std::ffi::c_void,
                (&mut p2) as *mut _ as *mut std::ffi::c_void,
                (&mut n0) as *mut _ as *mut std::ffi::c_void,
                (&mut n1) as *mut _ as *mut std::ffi::c_void,
                (&mut n2) as *mut _ as *mut std::ffi::c_void,
            ];
            ctx.launch3("split3", ((kv * 3) as u32).div_ceil(256), 1, 1, 256, &mut args)
                .unwrap();
            for x in [gq as *mut std::ffi::c_void, gk as *mut std::ffi::c_void] {
                let (mut xp, mut e, mut d) = (x, 1e-6f32, dstate);
                let mut a2: Vec<*mut std::ffi::c_void> = vec![
                    (&mut xp) as *mut _ as *mut std::ffi::c_void,
                    (&mut e) as *mut _ as *mut std::ffi::c_void,
                    (&mut d) as *mut _ as *mut std::ffi::c_void,
                ];
                ctx.launch3("q4_l2_rows", 1, 1, 1, 32, &mut a2).unwrap();
            }
            let (mut gp, mut sc, mut nn) = (gq as *mut std::ffi::c_void, 0.088f32, kv);
            let mut a3: Vec<*mut std::ffi::c_void> = vec![
                (&mut gp) as *mut _ as *mut std::ffi::c_void,
                (&mut sc) as *mut _ as *mut std::ffi::c_void,
                (&mut nn) as *mut _ as *mut std::ffi::c_void,
            ];
            ctx.launch3("q4_scale", (kv as u32).div_ceil(128), 1, 1, 128, &mut a3).unwrap();
        };
        for _ in 0..20 {
            launch_seq();
        }
        ctx.sync().unwrap();
        const M: usize = 2000;
        let t2 = std::time::Instant::now();
        for _ in 0..M {
            launch_seq();
        }
        let enq3 = t2.elapsed().as_secs_f64() * 1e3;
        ctx.sync().unwrap();
        let total3 = t2.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "# micro GDN l2scale 시퀀스(4커널) {M}회: host {enq3:.1}ms ({:.2}µs/회), 총 {total3:.1}ms ({:.2}µs/회)",
            enq3 / M as f64 * 1e3,
            total3 / M as f64 * 1e3
        );

        // DRAM 대역 — 512MB 버퍼를 bcast_rows(읽기 n*4 + 쓰기 n*4)로 훑는다.
        // 디코드의 실 트래픽 추정(129.5ms × 대역)에 필요한 값.
        let big = 512usize * 1024 * 1024 / 4; // f32 개수
        let sb = ctx.scratch(big * 4).expect("src-big");
        let db = ctx.scratch(big * 4).expect("dst-big");
        let launch_big = || {
            let (mut sp, mut dp) = (sb as *mut std::ffi::c_void, db as *mut std::ffi::c_void);
            let (mut nn, mut rr) = (big as i32, 1i32);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut sp) as *mut _ as *mut std::ffi::c_void,
                (&mut dp) as *mut _ as *mut std::ffi::c_void,
                (&mut nn) as *mut _ as *mut std::ffi::c_void,
                (&mut rr) as *mut _ as *mut std::ffi::c_void,
            ];
            ctx.launch3("bcast_rows", (big as u32).div_ceil(128), 1, 1, 128, &mut args)
                .unwrap();
        };
        for _ in 0..3 {
            launch_big();
        }
        ctx.sync().unwrap();
        let t3 = std::time::Instant::now();
        launch_big();
        ctx.sync().unwrap();
        let dt = t3.elapsed().as_secs_f64();
        let gb = (big as f64 * 8.0) / 1e9; // 읽기+쓰기
        eprintln!(
            "# micro DRAM 대역(512MB x2): {:.2}ms → {:.0} GB/s",
            dt * 1e3,
            gb / dt
        );

        // 가중치 접근 패턴 프로브 — 144B 블록 스트라이드에서 실효 대역.
        // (q4_K 가중치를 스칼라로 읽는 현 커널들의 실제 패턴)
        let nb = 4_000_000usize; // 4M 블록 × 144B = 576MB
        let wbuf = ctx.scratch(nb * 144).expect("wbuf");
        let accb = ctx.scratch(4).expect("acc");
        for mode in [0i32, 1, 2, 3, 4, 5] {
            let launch_bw = || {
                let (mut wp, mut ap) = (wbuf as *mut std::ffi::c_void, accb as *mut std::ffi::c_void);
                let (mut n, mut m) = (nb as i32, mode);
                let mut args: Vec<*mut std::ffi::c_void> = vec![
                    (&mut wp) as *mut _ as *mut std::ffi::c_void,
                    (&mut ap) as *mut _ as *mut std::ffi::c_void,
                    (&mut n) as *mut _ as *mut std::ffi::c_void,
                    (&mut m) as *mut _ as *mut std::ffi::c_void,
                ];
                ctx.launch3("bw_strided", (nb as u32).div_ceil(256), 1, 1, 256, &mut args)
                    .unwrap();
            };
            launch_bw();
            ctx.sync().unwrap();
            let t = std::time::Instant::now();
            for _ in 0..4 {
                launch_bw();
            }
            ctx.sync().unwrap();
            let dt = t.elapsed().as_secs_f64();
            // 접근한 바이트 수(스칼라 4B/8B, 벡터 16B) 기준 실효 대역
            let bytes = match mode {
                0 => nb as f64 * 4.0,
                1 => nb as f64 * 4.0,
                2 => nb as f64 * 16.0,
                3 => nb as f64 * 2.0 * 8.0,   // f16 8회/스레드
                5 => nb as f64 * 4.0 * 8.0,   // 워드 8회/레인(완전 병합)
                _ => nb as f64 * 2.0 * 8.0,   // 워프 협동도 같은 바이트
            } * 4.0;
            let touched = nb as f64 * 144.0 * 4.0; // 실제로 건드린 메모리 범위
            eprintln!(
                "# micro bw_strided mode{mode}: {:.1}ms → 실사용 {:.1} GB/s (건드린 범위 기준 {:.1} GB/s)",
                dt * 1e3,
                bytes / dt / 1e9,
                touched / dt / 1e9
            );
        }

        // MoE 그룹 커널 + 비동기 d2h 왕복 격리 — 디코드 디바이스 경로가
        // +30ms/스텝을 보이는데, 그 추가분의 실체를 여기서 가른다.
        let ne = 512i32;
        let rows = 10i32;
        // 상한은 프로덕션과 같은 식(rows + 16*ne)을 쓴다 — 커널도 이 값을 인자로
        // 받고, rowexp·x 버퍼가 이 크기를 전제한다(2026-09-14 단일화).
        let bound = (rows + 16 * ne) as usize;
        let ids = ctx.scratch(rows as usize * 4).expect("ids");
        let offb = ctx.scratch((ne as usize + 2) * 4).expect("off");
        let permb = ctx.scratch(bound * 4).expect("perm");
        let invb = ctx.scratch(rows as usize * 4).expect("inv");
        let rexb = ctx.scratch(bound * 4).expect("rex"); // GEMM이 rows_pad까지 읽는다
        let ppb = ctx.scratch(bound * 4).expect("pp");
        let ipb = ctx.scratch(rows as usize * 4).expect("ip");
        let txb = ctx.scratch(bound).expect("tx");
        let rpb = ctx.scratch(4).expect("rp");
        let launch_g = || {
            let (mut a, mut b) = (ids as *mut std::ffi::c_void, offb as *mut std::ffi::c_void);
            let (mut c, mut d) = (permb as *mut std::ffi::c_void, invb as *mut std::ffi::c_void);
            let (mut e, mut f) = (rexb as *mut std::ffi::c_void, ppb as *mut std::ffi::c_void);
            let (mut g, mut h) = (ipb as *mut std::ffi::c_void, txb as *mut std::ffi::c_void);
            let mut i = rpb as *mut std::ffi::c_void;
            let (mut n_e, mut rws) = (ne, rows);
            let mut bnd = bound as i32;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut a) as *mut _ as *mut std::ffi::c_void,
                (&mut n_e) as *mut _ as *mut std::ffi::c_void,
                (&mut rws) as *mut _ as *mut std::ffi::c_void,
                (&mut b) as *mut _ as *mut std::ffi::c_void,
                (&mut c) as *mut _ as *mut std::ffi::c_void,
                (&mut d) as *mut _ as *mut std::ffi::c_void,
                (&mut e) as *mut _ as *mut std::ffi::c_void,
                (&mut f) as *mut _ as *mut std::ffi::c_void,
                (&mut g) as *mut _ as *mut std::ffi::c_void,
                (&mut h) as *mut _ as *mut std::ffi::c_void,
                (&mut i) as *mut _ as *mut std::ffi::c_void,
                (&mut bnd) as *mut _ as *mut std::ffi::c_void,
            ];
            ctx.launch3("q4_moe_group_t1", 1, 1, 1, 128, &mut args).unwrap();
        };
        for _ in 0..20 {
            launch_g();
        }
        ctx.sync().unwrap();
        let tg1 = std::time::Instant::now();
        for _ in 0..N {
            launch_g();
        }
        let enqg = tg1.elapsed().as_secs_f64() * 1e3;
        ctx.sync().unwrap();
        let totg = tg1.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "# micro group kernel {N}회: host {enqg:.1}ms ({:.2}µs), 총 {totg:.1}ms ({:.2}µs)",
            enqg / N as f64 * 1e3,
            totg / N as f64 * 1e3
        );
        // 비동기 d2h + 이벤트 대기 왕복(디바이스 경로가 층마다 하는 것)
        let tg2 = std::time::Instant::now();
        for _ in 0..N {
            let _ = ctx.d2h_issue((ne as usize + 1) * 4, offb as *const u8).unwrap();
            ctx.d2h_wait().unwrap();
        }
        let totd = tg2.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "# micro d2h_issue+wait {N}회: 총 {totd:.1}ms ({:.2}µs/회)",
            totd / N as f64 * 1e3
        );
    }
}

