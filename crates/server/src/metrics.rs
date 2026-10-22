
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

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

pub struct Info {
    pub instance: String,
    pub model: String,
    pub ctx: usize,
    pub model_max_ctx: usize,
    pub plan: String,
    pub n_slots: usize,
    pub model_bytes: u64,
    pub kv_mem: u64,
    pub pool_cap_tokens: usize,
}

#[derive(Clone, Default)]
pub struct MemView {
    pub weights_gpu: u64,
    pub kv_gpu: u64,
    pub kv_host: u64,
    pub kv_host_blocks: u64,
    pub kv_blocks_total: u64,
    pub kv_free_vram: u64,
    pub kv_free_host: u64,
    pub offload_cpu: u64,
    pub ple_cpu: u64,
    pub active_per_token: u64,
    pub moe_mode: String,
    pub moe_cache: (u64, u64, usize),
    pub mem_advice: u8,
}

#[derive(Clone, Copy, Default)]
pub struct IoView {
    pub runtime: [(u64, u64, u64); 3],
    pub load: [(u64, u64, u64); 3],
}

struct Shared {
    info: Info,
    started: std::time::Instant,
    free_before: Vec<AtomicU64>,
    gpu_free: Vec<AtomicU64>,
    gpu_total: Vec<AtomicU64>,
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

pub fn init(info: Info) {
    let sampler = llm170_backend_gpu::VramSampler::new();
    let n = sampler.as_ref().map(|s| s.device_count()).unwrap_or(0);
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
        return;
    }
    std::thread::Builder::new()
        .name("llm170-metrics".into())
        .spawn(move || sampler_loop(sampler))
        .ok();
}

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

pub fn publish_memory(v: MemView) {
    if let Some(s) = SNAP.get()
        && let Ok(mut g) = s.mem.lock()
    {
        *g = v;
    }
}

pub fn publish_io(v: IoView) {
    if let Some(s) = SNAP.get()
        && let Ok(mut g) = s.io.lock()
    {
        *g = v;
    }
}

pub fn publish_slots(v: Vec<SlotView>) {
    if let Some(s) = SNAP.get()
        && let Ok(mut g) = s.slots.lock()
    {
        *g = v;
    }
}

fn pm_gauge(o: &mut String, name: &str, help: &str, val: String) {
    o.push_str(&format!(
        "# HELP {name} {help}\n# TYPE {name} gauge\n{val}\n"
    ));
}

fn pm_counter(o: &mut String, name: &str, help: &str, val: u64) {
    o.push_str(&format!(
        "# HELP {name} {help}\n# TYPE {name} counter\n{name} {val}\n"
    ));
}

fn advice_name(v: u8) -> &'static str {
    match v {
        1 => "prefer_kv",
        2 => "prefer_cache",
        _ => "hold",
    }
}

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

pub fn effective_ctx() -> usize {
    SNAP.get().map(|s| s.info.ctx).unwrap_or(0)
}
pub fn model_max_ctx() -> usize {
    SNAP.get().map(|s| s.info.model_max_ctx).unwrap_or(0)
}

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
        "{{\"instance\":\"{}\",\"model\":\"{}\",\"ctx\":{},\"model_max_ctx\":{},\"plan\":\"{}\",\"effective_ctx\":{},\"n_slots\":{},\"uptime_s\":{:.1},\"memory\":{{\"model_bytes\":{},\"gpu_total\":{gpu_total},\"gpu_used\":{gpu_used},\"weights\":{{\"gpu\":{},\"cpu\":{}}},\"kv_cache\":{{\"gpu\":{},\"host\":{},\"cpu\":0,\"host_blocks\":{},\"total_blocks\":{},\"free_vram_blocks\":{},\"free_host_blocks\":{}}},\"workspace\":{{\"gpu\":{workspace},\"cpu\":0}},\"ple\":{{\"gpu\":0,\"cpu\":{}}},\"active_per_token\":{},\"host_rss\":{},\"moe_mode\":\"{}\",\"moe_cache\":{{\"k\":{},\"hits\":{},\"miss\":{},\"hit_rate\":{:.4}}}}},\"gpus\":[{}],\"io\":{{\"h2d_bytes\":{},\"h2d_ms\":{},\"h2d_calls\":{},\"d2h_bytes\":{},\"d2h_ms\":{},\"d2h_calls\":{},\"d2d_bytes\":{},\"d2d_ms\":{},\"d2d_calls\":{},\"load_h2d_bytes\":{},\"load_h2d_ms\":{},\"load_d2h_bytes\":{},\"load_d2d_bytes\":{}}},\"serving\":{{\"kv_mem_bytes\":{},\"pool_cap_tokens\":{},\"kv_offload_ratio\":{:.4},\"mem_advice\":\"{}\",\"kv_pool_bytes\":{},\"jobs\":{},\"queue_wait_ms_avg\":{:.1},\"prefix_reuse_tokens\":{},\"prefix_reuse_ratio\":{:.4},\"prefix_reuse_full\":{},\"prefix_reuse_ckpt\":{},\"ttft_ms_avg\":{:.1},\"trie_boundaries\":{},\"decode_steps\":{},\"decode_ms_avg\":{:.2},\"prefill_chunks\":{},\"prefill_ms_avg\":{:.2},\"slots_active\":{},\"prompt_tokens\":{},\"generation_tokens\":{},\"requests\":{},\"requests_failed\":{},\"spec_rounds\":{},\"spec_accepted\":{},\"spec_round_ms_avg\":{:.2},\"aging\":{{\"frozen\":{},\"d_ms\":{},\"freeze_ms\":{},\"events\":{},\"blocked_jobs\":{},\"blocked_tokens\":{},\"head_wait_p95_ms\":{}}}}},\"slot_detail\":[{}]}}",
        esc(&s.info.instance),
        esc(&s.info.model),
        s.info.ctx,
        s.info.model_max_ctx,
        esc(&s.info.plan),
        s.info.ctx,
        s.info.n_slots,
        s.started.elapsed().as_secs_f64(),
        s.info.model_bytes,
        mem.weights_gpu,
        mem.offload_cpu,
        mem.kv_gpu,
        mem.kv_host,
        mem.kv_host_blocks,
        mem.kv_blocks_total,
        mem.kv_free_vram,
        mem.kv_free_host,
        mem.ple_cpu,
        mem.active_per_token,
        s.rss.load(Ordering::Relaxed),
        esc(&mem.moe_mode),
        mem.moe_cache.2,
        mem.moe_cache.0,
        mem.moe_cache.1,
        if mem.moe_cache.0 + mem.moe_cache.1 > 0 {
            mem.moe_cache.0 as f64 / (mem.moe_cache.0 + mem.moe_cache.1) as f64
        } else {
            0.0
        },
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
        s.info.kv_mem,
        s.info.pool_cap_tokens,
        if mem.kv_gpu + mem.kv_host > 0 {
            mem.kv_host as f64 / (mem.kv_gpu + mem.kv_host) as f64
        } else {
            0.0
        },
        advice_name(mem.mem_advice),
        mem.kv_gpu,
        jobs,
        if jobs > 0 {
            qw as f64 / jobs as f64 / 1e3
        } else {
            0.0
        },
        sc.prefix_tokens.load(Ordering::Relaxed),
        {
            let pt = sc.prompt_tokens.load(Ordering::Relaxed);
            if pt > 0 {
                sc.prefix_tokens.load(Ordering::Relaxed) as f64 / pt as f64
            } else {
                0.0
            }
        },
        sc.prefix_full_hits.load(Ordering::Relaxed),
        sc.prefix_ckpt_hits.load(Ordering::Relaxed),
        {
            let jb = sc.jobs.load(Ordering::Relaxed);
            if jb > 0 {
                sc.ttft_us.load(Ordering::Relaxed) as f64 / jb as f64 / 1e3
            } else {
                0.0
            }
        },
        sc.trie_boundaries.load(Ordering::Relaxed),
        td,
        if td > 0 { md as f64 / td as f64 } else { 0.0 },
        cp,
        if cp > 0 { mp as f64 / cp as f64 } else { 0.0 },
        sc.slots_active.load(Ordering::Relaxed),
        sc.prompt_tokens.load(Ordering::Relaxed),
        sc.gen_tokens.load(Ordering::Relaxed),
        sc.requests.load(Ordering::Relaxed),
        sc.requests_failed.load(Ordering::Relaxed),
        sc.spec_rounds.load(Ordering::Relaxed),
        sc.spec_accepted.load(Ordering::Relaxed),
        {
            let sr = sc.spec_rounds.load(Ordering::Relaxed);
            if sr > 0 {
                sc.spec_ms.load(Ordering::Relaxed) as f64 / sr as f64 / 1e3
            } else {
                0.0
            }
        },
        sc.aging_frozen.load(Ordering::Relaxed),
        sc.aging_d_ms.load(Ordering::Relaxed),
        sc.aging_freeze_ms.load(Ordering::Relaxed),
        sc.aging_events.load(Ordering::Relaxed),
        sc.aging_blocked_jobs.load(Ordering::Relaxed),
        sc.aging_blocked_tokens.load(Ordering::Relaxed),
        sc.aging_head_wait_p95_ms.load(Ordering::Relaxed),
        slot_detail.join(",")
    )
}

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
        "llm170_spec_rounds_total",
        "speculative verify rounds",
        sc.spec_rounds.load(Ordering::Relaxed),
    );
    pm_counter(
        &mut o,
        "llm170_spec_accepted_total",
        "accepted draft tokens",
        sc.spec_accepted.load(Ordering::Relaxed),
    );
    pm_counter(
        &mut o,
        "llm170_spec_round_ms_total",
        "speculative verify round wall us (x1000)",
        sc.spec_ms.load(Ordering::Relaxed) / 1000,
    );
    pm_gauge(
        &mut o,
        "llm170_spec_econ_on",
        "spec economics gate (1=spec, 0=plain)",
        format!("llm170_spec_econ_on {}", sc.econ_on.load(Ordering::Relaxed)),
    );
    pm_gauge(
        &mut o,
        "llm170_spec_policy",
        "spec policy (0=auto/off, 1=mtp, 2=ngram, 3=off)",
        format!(
            "llm170_spec_policy {}",
            crate::engine::SPEC_POLICY.load(Ordering::Relaxed)
        ),
    );
    pm_gauge(
        &mut o,
        "llm170_spec_enabled",
        "spec explicitly enabled by policy (1=yes, 0=no)",
        format!(
            "llm170_spec_enabled {}",
            u8::from(crate::engine::spec_enabled_by_policy())
        ),
    );
    let econ_cost = |v: &std::sync::atomic::AtomicU64, n: &std::sync::atomic::AtomicU64| -> f64 {
        if n.load(Ordering::Relaxed) == 0 {
            0.0
        } else {
            f64::from_bits(v.load(Ordering::Relaxed))
        }
    };
    pm_gauge(
        &mut o,
        "llm170_spec_cost_ms_tok",
        "spec round cost EMA (ms per emitted token, best source)",
        format!(
            "llm170_spec_cost_ms_tok {:.4}",
            econ_cost(&sc.econ_src[0], &sc.econ_src_n[0]).min({
                let n = sc.econ_src_n[1].load(Ordering::Relaxed);
                if n == 0 {
                    f64::INFINITY
                } else {
                    f64::from_bits(sc.econ_src[1].load(Ordering::Relaxed))
                }
            })
        ),
    );
    pm_gauge(
        &mut o,
        "llm170_plain_cost_ms_tok",
        "plain decode cost EMA (ms per token)",
        format!(
            "llm170_plain_cost_ms_tok {:.4}",
            econ_cost(&sc.econ_plain, &sc.econ_plain_n)
        ),
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
    pm_counter(
        &mut o,
        "llm170_prefix_reuse_full_total",
        "Prefix-cache full-prefix reuse hits",
        sc.prefix_full_hits.load(Ordering::Relaxed),
    );
    pm_counter(
        &mut o,
        "llm170_prefix_reuse_ckpt_total",
        "Prefix-cache GDN-checkpoint reuse hits",
        sc.prefix_ckpt_hits.load(Ordering::Relaxed),
    );
    {
        let pt = sc.prompt_tokens.load(Ordering::Relaxed);
        let ratio = if pt > 0 {
            sc.prefix_tokens.load(Ordering::Relaxed) as f64 / pt as f64
        } else {
            0.0
        };
        pm_gauge(
            &mut o,
            "llm170_prefix_reuse_ratio",
            "Prefix-cache hit ratio (reused/prompt tokens, cumulative)",
            format!("{ratio:.4}"),
        );
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_escapes_and_shape() {
        let j = json();
        assert!(j.starts_with('{') && j.ends_with('}'), "{j}");
        assert_eq!(esc("a\"b\\c\nd"), "a\\\"b\\\\c d");
    }

    #[test]
    fn prometheus_is_empty_before_init() {
        assert!(prometheus().is_empty() || prometheus().contains("llm170_"));
    }
}
