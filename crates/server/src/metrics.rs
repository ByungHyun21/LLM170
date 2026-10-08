//! 모니터링 스냅샷 — 외부 폴러(JSON) + Prometheus 텍스트 (2026-10-09).
//!
//! 전제(사용자): CMP 4장 1대 · 서버 1~2개 — 외부 폴러가 각 인스턴스를 긁고
//! **시계열 누적은 폴러 쪽**에서 한다. 서버는 **최신 스냅샷 1장**만 유지한다
//! (히스토리·링버퍼 없음, 모든 값은 덮어쓰기).
//!
//! 설계:
//! - **CUDA 호출을 HTTP 경로에서 완전 배제** — 전용 샘플러 스레드(1Hz)가
//!   장치별 VRAM(`VramSampler` — 기동 시 1회 retain)과 /proc(RSS·
//!   MemAvailable·MemTotal)을 읽어 원자값에 게시. 핸들러는 읽기+조립만
//!   (스크랩 빈도와 무관한 비용 — /health와 동급).
//! - 누적 카운터는 SCHED(AtomicU64)의 "최신 누적값" — 스크래퍼가 두 스냅샷
//!   차분으로 처리량을 계산한다(스크래이프 리셋 트릭 없음).
//! - GPU 라벨: **전 장치 열거**(한 프로세스가 여러 장을 쓰는 미래 대비).
//! - VRAM free/total 의미론: **기기 전체** 값 — 이 프로세스 귀속이 아니다
//!   (공유 기기에서는 타 프로세스 포함; 독점 배포 가정).
//! - 인스턴스 식별: `LLM170_INSTANCE`(선택) + 모델/ctx/slots/장치 수.
//! - 메모리: 원자값 u64 몇 개 + 슬롯 뷰(<1KB) — 총 2KB 미만.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// 슬롯 상태 뷰 — slot_loop가 저빈도(≤1Hz)로 게시(최신값만).
#[derive(Clone, Default)]
pub struct SlotView {
    pub id: usize,
    pub is_processing: bool,
    pub n_prompt_tokens: usize,
    pub n_prompt_processed: usize,
    pub n_cache: usize,
    pub n_decoded: usize,
    pub err_count: u32,
}

/// 정적 정보(기동 1회 — 이후 불변).
pub struct Info {
    pub instance: String,
    pub model: String,
    pub ctx: usize,
    pub n_slots: usize,
    /// 모델 파일 총합(디스크 — 분류의 기준점).
    pub model_bytes: u64,
}

/// 메모리 분류 뷰 — slot_loop가 저빈도 게시(모델이 시스템을 쓰는 방식).
#[derive(Clone, Default)]
pub struct MemView {
    /// VRAM 상주 가중치(비전문가 + 상주 전문가).
    pub weights_gpu: u64,
    /// VRAM KV 캐시(현행 전량 — CPU 오프로드 없음).
    pub kv_gpu: u64,
    /// CPU 오프로드 가중치(스트리밍 전문가 — 호스트에서 토큰별 업로드).
    pub offload_cpu: u64,
    /// CPU PLE 오프로드(미구현 W4-2 — 0).
    pub ple_cpu: u64,
    /// 토큰당 활성 가중치 바이트(실효 대역폭 = 이 값 × tg tok/s).
    pub active_per_token: u64,
    /// MoE 배치: "none" | "resident" | "streaming".
    pub moe_mode: String,
}

/// 복사(io) 뷰 — 런타임/적재 분리. 각 항목 = [h2d, d2h, d2d] × (바이트, ns, 호출).
#[derive(Clone, Copy, Default)]
pub struct IoView {
    /// 서빙 구간 누적(토큰당 스트리밍·로짓 판독 등 — 병목 판독용).
    pub runtime: [(u64, u64, u64); 3],
    /// 적재 구간(가중치 업로드 — 1회).
    pub load: [(u64, u64, u64); 3],
}

struct Shared {
    info: Info,
    started: std::time::Instant,
    /// 기동 직후(엔진 적재 전) 프리 VRAM — gpu_used = before − now.
    free_before: Vec<AtomicU64>,
    gpu_free: Vec<AtomicU64>,
    gpu_total: Vec<AtomicU64>,
    /// 장치 능력·이름(정적 — 대역폭 피크·SM).
    caps: Vec<llm170_backend_gpu::DeviceCaps>,
    names: Vec<String>,
    mem: Mutex<MemView>,
    io: Mutex<IoView>,
    host_avail: AtomicU64,
    host_total: AtomicU64,
    rss: AtomicU64,
    slots: Mutex<Vec<SlotView>>,
}

static SNAP: OnceLock<Shared> = OnceLock::new();

/// 기동 시 1회 — 정적 정보 등록 + 샘플러 스레드 시작.
/// 장치 수는 `VramSampler`가 실제 열거한 값(드라이버 실측)을 쓴다.
pub fn init(info: Info) {
    let sampler = llm170_backend_gpu::VramSampler::new();
    let n = sampler.as_ref().map(|s| s.device_count()).unwrap_or(0);
    // 엔진 적재 **전** 프리 VRAM 기록(적재 후 델타 = 이 프로세스 VRAM).
    let before: Vec<u64> = sampler
        .as_ref()
        .map(|s| {
            s.sample()
                .into_iter()
                .map(|v| v.map(|(f, _)| f).unwrap_or(0))
                .collect()
        })
        .unwrap_or_default();
    let shared = Shared {
        info,
        started: std::time::Instant::now(),
        free_before: before.iter().map(|&f| AtomicU64::new(f)).collect(),
        gpu_free: before.iter().map(|&f| AtomicU64::new(f)).collect(),
        gpu_total: (0..n).map(|_| AtomicU64::new(0)).collect(),
        caps: sampler
            .as_ref()
            .map(|s| s.caps().to_vec())
            .unwrap_or_default(),
        names: sampler
            .as_ref()
            .map(|s| s.names().to_vec())
            .unwrap_or_default(),
        mem: Mutex::new(MemView::default()),
        io: Mutex::new(IoView::default()),
        host_avail: AtomicU64::new(0),
        host_total: AtomicU64::new(0),
        rss: AtomicU64::new(0),
        slots: Mutex::new(Vec::new()),
    };
    if SNAP.set(shared).is_err() {
        return; // 중복 init — 무시
    }
    std::thread::Builder::new()
        .name("llm170-metrics".into())
        .spawn(move || sampler_loop(sampler))
        .ok();
}

/// 샘플러 루프 — 1Hz. 실패 장치는 이전 값 유지(0으로 덮지 않음).
fn sampler_loop(sampler: Option<llm170_backend_gpu::VramSampler>) {
    loop {
        let Some(shared) = SNAP.get() else {
            return;
        };
        if let Some(s) = &sampler {
            for (i, v) in s.sample().iter().enumerate() {
                let Some((f, t)) = v else { continue };
                if let Some(a) = shared.gpu_free.get(i) {
                    a.store(*f, Ordering::Relaxed);
                }
                if let Some(a) = shared.gpu_total.get(i) {
                    a.store(*t, Ordering::Relaxed);
                }
            }
        }
        if let Some(v) = crate::resource::host_mem_available() {
            shared.host_avail.store(v, Ordering::Relaxed);
        }
        if let Some(v) = crate::resource::host_mem_total() {
            shared.host_total.store(v, Ordering::Relaxed);
        }
        if let Some(v) = crate::resource::process_rss() {
            shared.rss.store(v, Ordering::Relaxed);
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}

/// 메모리 분류 게시(slot_loop — 최신값 덮어쓰기).
pub fn publish_memory(v: MemView) {
    if let Some(s) = SNAP.get()
        && let Ok(mut g) = s.mem.lock()
    {
        *g = v;
    }
}

/// 복사(io) 게시(slot_loop — 최신값 덮어쓰기).
pub fn publish_io(v: IoView) {
    if let Some(s) = SNAP.get()
        && let Ok(mut g) = s.io.lock()
    {
        *g = v;
    }
}

/// 슬롯 뷰 게시(slot_loop — 최신값 덮어쓰기).
pub fn publish_slots(v: Vec<SlotView>) {
    if let Some(s) = SNAP.get()
        && let Ok(mut g) = s.slots.lock()
    {
        *g = v;
    }
}

/// Prometheus 게이지 1줄(HELP/TYPE 포함).
fn pm_gauge(o: &mut String, name: &str, help: &str, val: String) {
    o.push_str(&format!(
        "# HELP {name} {help}\n# TYPE {name} gauge\n{val}\n"
    ));
}

/// Prometheus 누적 카운터 1줄.
fn pm_counter(o: &mut String, name: &str, help: &str, val: u64) {
    o.push_str(&format!(
        "# HELP {name} {help}\n# TYPE {name} counter\n{name} {val}\n"
    ));
}

/// JSON 문자열 이스케이프(최소 — 따옴표/역슬래시/제어문자).
fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => o.push(' '),
            c => o.push(c),
        }
    }
    o
}

/// JSON 스냅샷 — 외부 폴러용(수 KB). 메모리는 **모델 중심 분류**:
/// 각 컴포넌트가 gpu/cpu 어디에 얼마나 있는지(현행 KV·PLE는 gpu 전량/미구현).
pub fn json() -> String {
    let Some(s) = SNAP.get() else {
        return "{\"error\":\"metrics uninitialized\"}".to_string();
    };
    let dev_used = |i: usize| -> u64 {
        let before = s
            .free_before
            .get(i)
            .map(|a| a.load(Ordering::Relaxed))
            .unwrap_or(0);
        before.saturating_sub(s.gpu_free[i].load(Ordering::Relaxed))
    };
    let gpus: Vec<String> = (0..s.gpu_free.len())
        .map(|i| {
            let caps = s.caps.get(i).copied().unwrap_or_default();
            format!(
                "{{\"idx\":{i},\"name\":\"{}\",\"mem_free\":{},\"mem_total\":{},\"used\":{},\
\"sm_count\":{},\"sm_clock_khz\":{},\"mem_clock_khz\":{},\"bus_width_bits\":{},\
\"peak_bw_bytes_s\":{}}}",
                esc(s.names.get(i).map(String::as_str).unwrap_or("")),
                s.gpu_free[i].load(Ordering::Relaxed),
                s.gpu_total[i].load(Ordering::Relaxed),
                dev_used(i),
                caps.sm_count,
                caps.sm_clock_khz,
                caps.mem_clock_khz,
                caps.bus_width_bits,
                caps.peak_bw_bytes()
            )
        })
        .collect();
    let gpu_total: u64 = (0..s.gpu_total.len())
        .map(|i| s.gpu_total[i].load(Ordering::Relaxed))
        .sum();
    let gpu_used: u64 = (0..s.gpu_free.len()).map(dev_used).sum();
    let mem = s.mem.lock().map(|m| m.clone()).unwrap_or_default();
    let io = s.io.lock().map(|v| *v).unwrap_or_default();
    let workspace = gpu_used
        .saturating_sub(mem.weights_gpu)
        .saturating_sub(mem.kv_gpu);
    let sc = &crate::sched::SCHED;
    let jobs = sc.jobs.load(Ordering::Relaxed);
    let qw = sc.queue_wait_us.load(Ordering::Relaxed);
    let td = sc.ticks_decode.load(Ordering::Relaxed);
    let md = sc.ms_decode.load(Ordering::Relaxed);
    let cp = sc.chunks_prefill.load(Ordering::Relaxed);
    let mp = sc.ms_prefill.load(Ordering::Relaxed);
    let slot_detail: Vec<String> = s
        .slots
        .lock()
        .map(|g| {
            g.iter()
                .map(|v| {
                    format!(
                        "{{\"id\":{},\"processing\":{},\"prompt_tokens\":{},\"prompt_processed\":{},\"cache_tokens\":{},\"decoded\":{},\"err_count\":{}}}",
                        v.id,
                        v.is_processing,
                        v.n_prompt_tokens,
                        v.n_prompt_processed,
                        v.n_cache,
                        v.n_decoded,
                        v.err_count
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    format!(
        "{{\"instance\":\"{}\",\"model\":\"{}\",\"ctx\":{},\"n_slots\":{},\"uptime_s\":{:.1},\"memory\":{{\"model_bytes\":{},\"gpu_total\":{gpu_total},\"gpu_used\":{gpu_used},\"weights\":{{\"gpu\":{},\"cpu\":{}}},\"kv_cache\":{{\"gpu\":{},\"cpu\":0}},\"workspace\":{{\"gpu\":{workspace},\"cpu\":0}},\"ple\":{{\"gpu\":0,\"cpu\":{}}},\"active_per_token\":{},\"host_rss\":{},\"moe_mode\":\"{}\"}},\"gpus\":[{}],\"io\":{{\"h2d_bytes\":{},\"h2d_ms\":{},\"h2d_calls\":{},\"d2h_bytes\":{},\"d2h_ms\":{},\"d2h_calls\":{},\"d2d_bytes\":{},\"d2d_ms\":{},\"d2d_calls\":{},\"load_h2d_bytes\":{},\"load_h2d_ms\":{},\"load_d2h_bytes\":{},\"load_d2d_bytes\":{}}},\"serving\":{{\"jobs\":{},\"queue_wait_ms_avg\":{:.1},\"prefix_reuse_tokens\":{},\"decode_steps\":{},\"decode_ms_avg\":{:.2},\"prefill_chunks\":{},\"prefill_ms_avg\":{:.2},\"slots_active\":{},\"prompt_tokens\":{},\"generation_tokens\":{},\"requests\":{},\"requests_failed\":{}}},\"slot_detail\":[{}]}}",
        esc(&s.info.instance),
        esc(&s.info.model),
        s.info.ctx,
        s.info.n_slots,
        s.started.elapsed().as_secs_f64(),
        s.info.model_bytes,
        mem.weights_gpu,
        mem.offload_cpu,
        mem.kv_gpu,
        mem.ple_cpu,
        mem.active_per_token,
        s.rss.load(Ordering::Relaxed),
        esc(&mem.moe_mode),
        gpus.join(","),
        io.runtime[0].0,
        io.runtime[0].1 / 1_000_000,
        io.runtime[0].2,
        io.runtime[1].0,
        io.runtime[1].1 / 1_000_000,
        io.runtime[1].2,
        io.runtime[2].0,
        io.runtime[2].1 / 1_000_000,
        io.runtime[2].2,
        io.load[0].0,
        io.load[0].1 / 1_000_000,
        io.load[1].0,
        io.load[2].0,
        jobs,
        if jobs > 0 {
            qw as f64 / jobs as f64 / 1e3
        } else {
            0.0
        },
        sc.prefix_tokens.load(Ordering::Relaxed),
        td,
        if td > 0 { md as f64 / td as f64 } else { 0.0 },
        cp,
        if cp > 0 { mp as f64 / cp as f64 } else { 0.0 },
        sc.slots_active.load(Ordering::Relaxed),
        sc.prompt_tokens.load(Ordering::Relaxed),
        sc.gen_tokens.load(Ordering::Relaxed),
        sc.requests.load(Ordering::Relaxed),
        sc.requests_failed.load(Ordering::Relaxed),
        slot_detail.join(",")
    )
}

/// Prometheus 텍스트 — 표준 도구 호환(부가; 주 타깃은 JSON 폴러).
pub fn prometheus() -> String {
    let Some(s) = SNAP.get() else {
        return String::new();
    };
    let sc = &crate::sched::SCHED;
    let mut o = String::with_capacity(2048);
    o.push_str("# HELP llm170_gpu_mem_free_bytes GPU free memory (device-wide; exclusive-box assumption)\n");
    o.push_str("# TYPE llm170_gpu_mem_free_bytes gauge\n");
    for (i, a) in s.gpu_free.iter().enumerate() {
        o.push_str(&format!(
            "llm170_gpu_mem_free_bytes{{gpu=\"{i}\"}} {}\n",
            a.load(Ordering::Relaxed)
        ));
    }
    o.push_str("# TYPE llm170_gpu_mem_total_bytes gauge\n");
    for (i, a) in s.gpu_total.iter().enumerate() {
        o.push_str(&format!(
            "llm170_gpu_mem_total_bytes{{gpu=\"{i}\"}} {}\n",
            a.load(Ordering::Relaxed)
        ));
    }
    pm_gauge(
        &mut o,
        "llm170_host_mem_available_bytes",
        "Host MemAvailable (/proc/meminfo)",
        format!(
            "llm170_host_mem_available_bytes {}",
            s.host_avail.load(Ordering::Relaxed)
        ),
    );
    pm_gauge(
        &mut o,
        "llm170_host_mem_total_bytes",
        "Host MemTotal (/proc/meminfo)",
        format!(
            "llm170_host_mem_total_bytes {}",
            s.host_total.load(Ordering::Relaxed)
        ),
    );
    pm_gauge(
        &mut o,
        "llm170_process_resident_memory_bytes",
        "Process RSS (/proc/self/status VmRSS)",
        format!(
            "llm170_process_resident_memory_bytes {}",
            s.rss.load(Ordering::Relaxed)
        ),
    );
    pm_gauge(
        &mut o,
        "llm170_uptime_seconds",
        "Seconds since metrics init",
        format!(
            "llm170_uptime_seconds {:.1}",
            s.started.elapsed().as_secs_f64()
        ),
    );
    pm_gauge(
        &mut o,
        "llm170_slots_active",
        "Slots with an active job",
        format!(
            "llm170_slots_active {}",
            sc.slots_active.load(Ordering::Relaxed)
        ),
    );
    pm_counter(
        &mut o,
        "llm170_jobs_total",
        "Jobs assigned to slots",
        sc.jobs.load(Ordering::Relaxed),
    );
    pm_counter(
        &mut o,
        "llm170_requests_total",
        "Requests admitted",
        sc.requests.load(Ordering::Relaxed),
    );
    pm_counter(
        &mut o,
        "llm170_requests_failed_total",
        "Requests failed (final)",
        sc.requests_failed.load(Ordering::Relaxed),
    );
    pm_counter(
        &mut o,
        "llm170_prompt_tokens_total",
        "Prompt tokens processed",
        sc.prompt_tokens.load(Ordering::Relaxed),
    );
    pm_counter(
        &mut o,
        "llm170_generation_tokens_total",
        "Generated tokens emitted",
        sc.gen_tokens.load(Ordering::Relaxed),
    );
    pm_counter(
        &mut o,
        "llm170_queue_wait_us_total",
        "Cumulative queue wait (us)",
        sc.queue_wait_us.load(Ordering::Relaxed),
    );
    pm_counter(
        &mut o,
        "llm170_decode_steps_total",
        "Decode steps",
        sc.ticks_decode.load(Ordering::Relaxed),
    );
    pm_counter(
        &mut o,
        "llm170_decode_ms_total",
        "Cumulative decode ms",
        sc.ms_decode.load(Ordering::Relaxed),
    );
    pm_counter(
        &mut o,
        "llm170_prefill_chunks_total",
        "Prefill chunks",
        sc.chunks_prefill.load(Ordering::Relaxed),
    );
    pm_counter(
        &mut o,
        "llm170_prefill_ms_total",
        "Cumulative prefill ms",
        sc.ms_prefill.load(Ordering::Relaxed),
    );
    pm_counter(
        &mut o,
        "llm170_prefix_reuse_tokens_total",
        "Prefix-cache reused tokens",
        sc.prefix_tokens.load(Ordering::Relaxed),
    );
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_escapes_and_shape() {
        // 미초기화 상태 — 폴러가 파싱 가능한 JSON이어야 한다.
        let j = json();
        assert!(j.starts_with('{') && j.ends_with('}'), "{j}");
        // 이스케이프 규칙.
        assert_eq!(esc("a\"b\\c\nd"), "a\\\"b\\\\c d");
    }

    #[test]
    fn prometheus_is_empty_before_init() {
        // init 전에는 빈 문자열(핸들러가 200 + 빈 본문 — 폴러 무해).
        assert!(prometheus().is_empty() || prometheus().contains("llm170_"));
    }
}
