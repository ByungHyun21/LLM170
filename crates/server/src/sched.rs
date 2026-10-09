//! 슬롯 스케줄러 (engine.rs에서 순수 이동).
//! 연속 배칭: 디코드 우선·잔여 예산 프리필 청크(llama.cpp 규칙 1:1).
//! [B5 2026-10-09] 청크 가변화(예산−디코드)·라운드로빈·경합 long 캡.
//! 슬롯 = 엔진 시퀀스 id. Engine 열거는 engine.rs(파사드) — 이 층은
//! 배정(slot_loop·assign_slot)과 슬롯 상태만 담는다.
use crate::engine::{Engine, InferResult};
use std::sync::atomic::{AtomicU64, Ordering};

/// [B5 2026-10-09] 프리필 청크 정책 — llama.cpp "잔여 예산" 규칙의 명시화.
/// 틱 예산에서 활성 디코드(슬롯당 1토큰)를 차감한 몫과 엔진 상한의 min.
/// 경합(대기 프리필 복수)에서 긴 프롬프트는 청크를 절반으로 낮춰 라운드로빈
/// 지연을 균등화하고, 청크 상한이 커져도(엔진 확대 전제) 예산이 상한을 지킨다.
const PF_BUDGET_TOKENS: usize = 1024;
/// 엔진 프리필 상한 미러(CHAIN_TMAX=512).
const PF_CHUNK_MAX: usize = 512;
/// 청크 최소.
const PF_CHUNK_MIN: usize = 64;
/// 이 길이 초과 프롬프트는 경합 시 절반 청크(라운드로빈 최대 대기 ↓).
const PF_LONG_PROMPT: usize = 2048;

/// 슬롯 스케줄러 (04) — llama.cpp 규칙 1:1: 디코드 우선, 잔여 예산만
/// 프리필 청크. 슬롯 = 엔진 시퀀스 id. 요청 종료 → 슬롯 반환(reset_seq).
pub struct SlotJob {
    pub tokens: Vec<u32>,
    pub n_predict: usize,
    /// 샘플링 파라미터 (기본 greedy — None이면 GPU argmax 경로 유지).
    pub sampler: Option<llm170_core::sampler::SamplerParams>,
    /// 조기 종료 토큰 (EOS + 채팅 템플릿 종결자).
    pub stops: Vec<u32>,
    /// 토큰별 SSE 스트림 채널.
    pub progress: Option<std::sync::mpsc::Sender<u32>>,
    /// 최종 결과 송신.
    pub out: std::sync::mpsc::Sender<InferResult>,
    /// 108 P5: 큐 진입 시각 — 배정까지 대기(큐 적체·슬롯 부족 계측).
    pub queued: std::time::Instant,
}
struct Slot {
    job: Option<SlotJob>,
    prefilled: usize,
    next: u32,
    generated: u32,
    tokens: Vec<u32>,
    touch: u64,
    /// 클라이언트 절단 — progress 채널 송신 실패로 감지 (SSE flush 실패).
    cancelled: bool,
    /// 접두 캐시 — 상태가 구워진 전체 토큰열 (요청 간 유지).
    cached: Vec<u32>,
    /// 슬롯별 샘플러 (요청에서 생성, 토큰마다 상태 갱신).
    sampler: Option<llm170_core::sampler::Sampler>,
    /// QA-1: 연속 엔진 실패 횟수 — 성공 emit 시 0으로 리셋.
    err_count: u32,
    /// QA-1: 연속 실패 상한(3) 도달 시 확정 실패 사유.
    failed: Option<String>,
}

impl Slot {
    fn free() -> Self {
        Slot {
            job: None,
            prefilled: 0,
            next: 0,
            generated: 0,
            tokens: Vec::new(),
            touch: 0,
            cancelled: false,
            cached: Vec::new(),
            sampler: None,
            err_count: 0,
            failed: None,
        }
    }
}

/// QA-1: 엔진 호출 실패 적립 — 연속 3회 실패 시 슬롯 확정 실패(에러 전파).
/// 일회성 실패는 다음 틱 재시도(종전 동작), 지속 실패만 스피너에서 해방.
fn slot_fail(s: &mut Slot, msg: String) {
    s.err_count += 1;
    if s.err_count >= 3 && s.failed.is_none() {
        eprintln!("# slot 확정 실패(연속 {}회): {}", s.err_count, msg);
        s.failed = Some(msg);
    }
}

/// qwen35 로드 재시도 — 동일.
/// GPU 엔진 로드(간헐 ENOPT 재시도 — 2026-09-01 실측 회복 패턴).
pub(crate) fn load_gpu_retry(
    p: &std::path::Path,
    n_slots: usize,
    ctx: usize,
) -> crate::gpu_engine::GpuEngine {
    for i in 0..5 {
        match crate::gpu_engine::GpuEngine::load(p, n_slots, ctx) {
            Ok(e) => return e,
            Err(err) => {
                eprintln!("# gpu 엔진 로드 재시도 {}/5: {err}", i + 1);
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
        }
    }
    panic!("gpu 엔진 로드 최종 실패: {}", p.display())
}

/// 슬롯 로짓 → 토큰: 활성 샘플러면 sample, 아니면 greedy (동률 최저 인덱스).
fn pick(s: &mut Slot, logits: &[f32]) -> u32 {
    match &mut s.sampler {
        Some(sm) if !sm.is_greedy() => sm.sample(logits),
        _ => llm170_core::qwen35::greedy(logits),
    }
}

/// Q35 np 디코드 — 샘플링 슬롯 포함시 logits 경로(decode), 아니면 GPU argmax 판.
/// [A9-3 2026-10-10] 혼합 워크로드: 그리디 슬롯은 배치(가중 상각), 샘플링
/// 슬롯만 직렬 — 슬롯별 상태 독립이라 순서 무관. 그리디가 1개 이하면 종전 직렬.
fn q35_decode(e: &mut crate::gpu_engine::GpuEngine, slots: &mut [Slot], seqs: &[usize]) {
    // 그리디/샘플링 분할.
    let (mut greedy, mut sampled): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
    for &i in seqs {
        if slots[i].sampler.as_ref().is_some_and(|s| !s.is_greedy()) {
            sampled.push(i);
        } else {
            greedy.push(i);
        }
    }
    if greedy.len() >= 2 {
        let toks: Vec<u32> = greedy.iter().map(|&i| slots[i].next).collect();
        match e.decode_np_greedy(&greedy, &toks) {
            Ok(ts) => {
                for (row, &i) in greedy.iter().enumerate() {
                    slot_emit(&mut slots[i], ts[row]);
                }
            }
            Err(err) => {
                eprintln!("# np-greedy 실패({err}) — 이번 회차 건너뜀");
                for &i2 in &greedy {
                    slot_fail(&mut slots[i2], format!("decode_np_greedy: {err}"));
                }
            }
        }
    } else if let Some(&i) = greedy.first() {
        sampled.push(i); // 그리디 1개 — logits 경로로 합류(직렬).
    }
    if !sampled.is_empty() {
        let toks: Vec<u32> = sampled.iter().map(|&i| slots[i].next).collect();
        match e.decode(&sampled, &toks) {
            Ok(rows) => {
                for (row, &i) in sampled.iter().enumerate() {
                    let t = pick(&mut slots[i], &rows[row]);
                    slot_emit(&mut slots[i], t);
                }
            }
            Err(err) => {
                eprintln!("# decode 실패({err}) — 이번 회차 건너뜀");
                for &i2 in &sampled {
                    slot_fail(&mut slots[i2], format!("decode: {err}"));
                }
            }
        }
    }
}

/// 연속 배칭 루프 (04-2). 매 반복: ① 큐 drain → LRU 가용 슬롯 배정
/// ② 디코드 우선(활성 전 슬롯 — q35는 1배치 호출, q4는 슬롯별 decode1)
/// ③ 디코드한 스텝이 없으면 프리필 1청크. 완료/EOS → 슬롯 반환(reset_seq).
/// 108 P5 — 스케줄러 계측(원자). serve 종료 시 요약.
pub static SCHED: Sched = Sched {
    jobs: AtomicU64::new(0),
    queue_wait_us: AtomicU64::new(0),
    prefix_tokens: AtomicU64::new(0),
    ticks_decode: AtomicU64::new(0),
    ms_decode: AtomicU64::new(0),
    chunks_prefill: AtomicU64::new(0),
    ms_prefill: AtomicU64::new(0),
    spec_rounds: AtomicU64::new(0),
    spec_accepted: AtomicU64::new(0),
    slots_active: AtomicU64::new(0),
    prompt_tokens: AtomicU64::new(0),
    gen_tokens: AtomicU64::new(0),
    requests: AtomicU64::new(0),
    requests_failed: AtomicU64::new(0),
};
pub struct Sched {
    pub jobs: AtomicU64,
    pub queue_wait_us: AtomicU64,
    pub prefix_tokens: AtomicU64,
    pub ticks_decode: AtomicU64,
    pub ms_decode: AtomicU64,
    pub chunks_prefill: AtomicU64,
    pub ms_prefill: AtomicU64,
    /// 스펙 라운드 수/수용 토큰 수 계측(수용률 = acc/rounds).
    pub spec_rounds: AtomicU64,
    pub spec_accepted: AtomicU64,
    // ── 모니터링(2026-10-09) — "최신 누적값"만(히스토리 없음) ──
    /// 활성 작업 슬롯 수(게이지 — slot_loop가 게시).
    pub slots_active: AtomicU64,
    /// 프롬프트 토큰 누적(완료 잡 기준).
    pub prompt_tokens: AtomicU64,
    /// 생성 토큰 누적(방출 기준).
    pub gen_tokens: AtomicU64,
    /// 배정된 요청 누적.
    pub requests: AtomicU64,
    /// 확정 실패 요청 누적.
    pub requests_failed: AtomicU64,
}
impl Sched {
    pub fn summary(&self) -> String {
        let jobs = self.jobs.load(Ordering::Relaxed);
        let qw = self.queue_wait_us.load(Ordering::Relaxed);
        let px = self.prefix_tokens.load(Ordering::Relaxed);
        let td = self.ticks_decode.load(Ordering::Relaxed);
        let md = self.ms_decode.load(Ordering::Relaxed);
        let cp = self.chunks_prefill.load(Ordering::Relaxed);
        let mp = self.ms_prefill.load(Ordering::Relaxed);
        let sr = self.spec_rounds.load(Ordering::Relaxed);
        let sa = self.spec_accepted.load(Ordering::Relaxed);
        format!(
            "[sched] jobs {jobs} | queue-wait avg {:.0}ms | prefix-reuse {px}tok | decode {td}x avg {:.1}ms | prefill {cp}x avg {:.1}ms | spec {sr}r acc {sa} ({:.2}/r)",
            if jobs > 0 {
                qw as f64 / jobs as f64 / 1e3
            } else {
                0.0
            },
            if td > 0 { md as f64 / td as f64 } else { 0.0 },
            if cp > 0 { mp as f64 / cp as f64 } else { 0.0 },
            if sr > 0 { sa as f64 / sr as f64 } else { 0.0 },
        )
    }
}

pub fn slot_loop(mut eng: Engine, rx: std::sync::mpsc::Receiver<SlotJob>, n_slots: usize) {
    // EOS 하드코드 248044 → 모델 메타 파생(Engine::eos).
    let eos = eng.eos();
    // 모니터링 — 복사 계측 기준(가중치 적재분을 load로 분리, 이후는 runtime).
    let copy_base = eng.copy_stats();
    // 기동 워밍업 — 첫 요청이 지연 초기화(raw_init, ctx 비례 수십 초)를
    // 뒤집어쓰지 않도록 여기서 소진하고 상태를 되돌린다. 준비 전에는 /health가
    // 503이라 클라이언트가 계측을 시작하지 않는다.
    {
        let warm: Vec<u32> = vec![1u32; 16];
        let w: Result<(), String> = match &mut eng {
            Engine::Gpu(e) => e.prefill(0, &warm).and_then(|l| {
                let t = llm170_core::qwen35::greedy(&l);
                e.decode_greedy(0, t).map(|_| ())
            }),
        };
        if let Err(err) = w {
            eprintln!("# warmup 실패(치명 아님): {err}");
        }
        match &mut eng {
            Engine::Gpu(e) => {
                let _ = e.reset_states();
            }
        }
    }
    crate::http::READY.store(true, std::sync::atomic::Ordering::Release);
    let mut slots: Vec<Slot> = (0..n_slots).map(|_| Slot::free()).collect();
    let mut tick: u64 = 0;
    let (mut n_dec, mut n_pf) = (0u64, 0u64);
    let (mut ms_dec, mut ms_pf) = (0f64, 0f64);
    let npw = llm170_diag::dump::opts().key("wall_time");
    let t0w = std::time::Instant::now();
    let mut last_wt = std::time::Instant::now();
    // 모니터링 게시 스로틀(≤1Hz — 최신값만, 핫패스 할당 최소화).
    let mut last_pub = std::time::Instant::now();
    loop {
        // S3: 진행 심박 — 엔진이 멈추면(디코드/프리필 교착) 와치독이 보고한다.
        llm170_diag::watchdog::bump();
        if npw {
            let now = std::time::Instant::now();
            let dt = last_wt.elapsed().as_secs_f64();
            if dt > 0.05 {
                eprintln!("[wall] +{dt:.2}s @{}s", t0w.elapsed().as_secs_f64());
            }
            last_wt = now;
        }
        // 회귀 픽스(2026-09-16): 종전엔 try_recv로 꺼낸 뒤 "슬롯 점유"를 발견하면
        // break했다 — 꺼낸 작업이 그대로 버려져(송신측 drop → 수신측 즉시 Err)
        // 동시 요청이 빈 응랍으로 소실됐다(4동시 중 여럿 drop 실측). 점유 검사를
        // try_recv **앞으로** 옮겨 작업을 큐에 남긴다 — 의도된 원래 계약.
        loop {
            if !slots.iter().any(|s| s.job.is_none()) {
                break;
            }
            let Ok(j) = rx.try_recv() else {
                break;
            };
            // 107 W7: 배정 단일 구현으로 위임(접두 캐시 로직 동일).
            assign_slot(&mut slots, &mut eng, j, tick);
        }
        if last_pub.elapsed() >= std::time::Duration::from_millis(900) {
            last_pub = std::time::Instant::now();
            publish_slot_views(&slots);
            publish_stats(&eng, copy_base);
        }
        tick += 1;
        let _it0 = std::time::Instant::now();

        // ② 디코드 우선 — prefill 완료 슬롯 전부
        let active: Vec<usize> = (0..n_slots)
            .filter(|&i| {
                slots[i].job.is_some()
                    && slots[i].prefilled == slots[i].job.as_ref().unwrap().tokens.len()
            })
            .collect();
        let mut decoded = false;
        let mut dec_ms = 0f64;
        if !active.is_empty() {
            decoded = true;
            let _dt = std::time::Instant::now();
            llm170_diag::watchdog::bump();
            match &mut eng {
                Engine::Gpu(e) => q35_decode(e, &mut slots, &active),
            }
            dec_ms = _dt.elapsed().as_secs_f64() * 1e3;
            // 완료 슬롯 정리 — 결과 전송·반환
            for &i in &active {
                finish_slot(&mut slots[i], &mut eng, i, eos);
            }
        }

        // ③ 프리필 1청크 — 디코드가 없었던 회차이거나, 아직 프리필이 남은 대기 슬롯이
        // 있는 경우. (디코드 우선이지만 활성 슬롯의 디코드가 대기 슬롯의 프리필을
        // 영구히 굶기면 서버가 요청을 직렬화한다 — np4 실측 2026-09-12.)
        let pending_prefill = slots
            .iter()
            .any(|s| s.job.is_some() && s.prefilled < s.job.as_ref().unwrap().tokens.len());
        if !decoded || pending_prefill {
            let pf = (0..n_slots)
                .filter(|&i| {
                    slots[i].job.is_some()
                        && slots[i].prefilled < slots[i].job.as_ref().unwrap().tokens.len()
                })
                .min_by_key(|&i| slots[i].touch);
            if let Some(i) = pf {
                let _pft = std::time::Instant::now();
                llm170_diag::watchdog::bump();
                // [B5] 가변 청크 — 잔여 예산(디코드 차감) + 경합 시 long 캡.
                let left = slots[i].job.as_ref().unwrap().tokens.len() - slots[i].prefilled;
                let backlog = slots
                    .iter()
                    .filter(|s| {
                        s.job.is_some() && s.prefilled < s.job.as_ref().unwrap().tokens.len()
                    })
                    .count();
                let mut chunk = PF_BUDGET_TOKENS
                    .saturating_sub(active.len())
                    .clamp(PF_CHUNK_MIN, PF_CHUNK_MAX);
                if backlog > 1 && left > PF_LONG_PROMPT {
                    chunk = chunk.min(PF_CHUNK_MAX / 2);
                }
                let chunk = chunk.min(left);
                // Q4(FN)는 prefill_greedy — 청크마다 어휘 152k
                // 로짓 pageable D2H(슬로패스 수십 ms) 대신 GPU argmax 8B 회수.
                // Q35(27B)는 종전 전사 경로(원시 프리필 내부 d2h).
                let (start, logits) = {
                    let end = (slots[i].prefilled + chunk)
                        .min(slots[i].job.as_ref().unwrap().tokens.len());
                    let part: Vec<u32> =
                        slots[i].job.as_ref().unwrap().tokens[slots[i].prefilled..end].to_vec();
                    // 샘플링 슬롯은 로짓 판(마지막 청크만 판정에 사용) — Q4도
                    // prefill_greedy 대신 prefill. greedy는 종전 최적 경로.
                    let samp = slots[i].sampler.as_ref().is_some_and(|s| !s.is_greedy());
                    let r: Result<u32, String> = match &mut eng {
                        Engine::Gpu(e) => e.prefill(i, &part).map(|l| {
                            if samp {
                                pick(&mut slots[i], &l)
                            } else {
                                llm170_core::qwen35::greedy(&l)
                            }
                        }),
                    };
                    (end, r)
                };
                // [B5] 라운드로빈 — 방금 전진한 슬롯을 최신으로 갱신, 다음
                // 청크는 더 오래 기다린(pending) 슬롯이 받는다. 종전엔 touch가
                // 배정 시점에만 갱신돼 한 슬롯이 전량 프리필을 독점했다.
                slots[i].touch = tick;
                if npw {
                    eprintln!(
                        "[wall] prefill slot{i} {start}tok done @{}s",
                        t0w.elapsed().as_secs_f64()
                    );
                }
                n_pf += 1;
                ms_pf += _pft.elapsed().as_secs_f64() * 1e3;
                SCHED.chunks_prefill.fetch_add(1, Ordering::Relaxed);
                SCHED.ms_prefill.fetch_add(
                    (_pft.elapsed().as_secs_f64() * 1e3) as u64,
                    Ordering::Relaxed,
                );
                match logits {
                    Ok(t) => {
                        slots[i].prefilled = start;
                        if start == slots[i].job.as_ref().unwrap().tokens.len() {
                            slot_emit(&mut slots[i], t);
                        }
                    }
                    // QA-1: 프리필 실패 적립 — 종전엔 무시돼 prefilled가 영구
                    // 갱신되지 않는 스피너였다(매 틱 동일 청크 재시도).
                    Err(err) => slot_fail(&mut slots[i], format!("prefill: {err}")),
                }
                finish_slot(&mut slots[i], &mut eng, i, eos);
            }
        }

        if decoded {
            n_dec += 1;
            ms_dec += dec_ms;
            SCHED.ticks_decode.fetch_add(1, Ordering::Relaxed);
            SCHED.ms_decode.fetch_add(dec_ms as u64, Ordering::Relaxed);
            if llm170_diag::dump::opts().key("srv_time") && n_dec % 32 == 0 {
                eprintln!(
                    "[srv] steps={} decode avg {:.1}ms | prefill {}x avg {:.1}ms | {}",
                    n_dec,
                    ms_dec / n_dec as f64,
                    n_pf,
                    if n_pf > 0 { ms_pf / n_pf as f64 } else { 0.0 },
                    SCHED.summary()
                );
            }
        }
        // 유휴 시 차단 수신 — 종료(송신자 전 소멸) 시 루프 탈출.
        // 차단 전 게시(유휴 상태가 마지막 뷰로 굳지 않게 — 실측 결함).
        let busy = slots.iter().any(|s| s.job.is_some());
        if !busy {
            publish_slot_views(&slots);
            publish_stats(&eng, copy_base);
        }
        if !busy {
            match rx.recv() {
                Ok(j) => {
                    // 107 W7: 배정 단일 구현 위임(전 슬롯 접두 탐색으로 개선 — 종전 slot0 고정).
                    assign_slot(&mut slots, &mut eng, j, tick);
                }
                Err(_) => break,
            }
        }
    }
    eprintln!("{}", SCHED.summary());
}

/// 모니터링 — 메모리 분류 + 복사 계측 게시(최신값만).
fn publish_stats(eng: &Engine, base: [(u64, u64, u64); 3]) {
    let (weights_gpu, kv_gpu, offload_cpu, ple_cpu) = eng.mem_stats();
    crate::metrics::publish_memory(crate::metrics::MemView {
        weights_gpu,
        kv_gpu,
        offload_cpu,
        ple_cpu,
        active_per_token: eng.active_weight_bytes(),
        moe_mode: eng.moe_mode().to_string(),
    });
    let now = eng.copy_stats();
    let sub = |i: usize| -> (u64, u64, u64) {
        (
            now[i].0.saturating_sub(base[i].0),
            now[i].1.saturating_sub(base[i].1),
            now[i].2.saturating_sub(base[i].2),
        )
    };
    crate::metrics::publish_io(crate::metrics::IoView {
        runtime: [sub(0), sub(1), sub(2)],
        load: base,
    });
}

/// 모니터링 게시 — 슬롯 뷰 + 활성 수(최신값 덮어쓰기, 히스토리 없음).
fn publish_slot_views(slots: &[Slot]) {
    let views: Vec<crate::metrics::SlotView> = slots
        .iter()
        .enumerate()
        .map(|(i, s)| crate::metrics::SlotView {
            id: i,
            is_processing: s.job.is_some(),
            n_prompt_tokens: s.job.as_ref().map(|j| j.tokens.len()).unwrap_or(0),
            n_prompt_processed: s.prefilled,
            n_cache: s.cached.len(),
            n_decoded: s.generated as usize,
            err_count: s.err_count,
        })
        .collect();
    SCHED.slots_active.store(
        views.iter().filter(|v| v.is_processing).count() as u64,
        Ordering::Relaxed,
    );
    crate::metrics::publish_slots(views);
}

/// 슬롯 배정 단일 구현 (107 W7: drain/유휴 이중 복제 통합).
/// 접두 캐시 최장 일치 슬롯 선택(전 슬롯 대상 — 종전 유휴 경로는
/// slot0 고정이었음), 재사용 시 reset 생략, 샘플러 시딩 포함.
fn assign_slot(slots: &mut [Slot], eng: &mut Engine, j: SlotJob, tick: u64) {
    // 모니터링 — 배정 단일 지점(두 수신 경로 공통).
    SCHED.requests.fetch_add(1, Ordering::Relaxed);
    SCHED
        .queue_wait_us
        .fetch_add(j.queued.elapsed().as_micros() as u64, Ordering::Relaxed);
    // [H 2026-10-09] 빈 프롬프트 최후 가드 — 표면(oai 400·CLI)이 막지만
    // 스케줄러 직접 진입은 0번 토큰에서 조용히 디코드하게 된다.
    if j.tokens.is_empty() {
        SCHED.requests_failed.fetch_add(1, Ordering::Relaxed);
        let _ = j.out.send(InferResult {
            tokens: Vec::new(),
            error: Some("empty prompt".into()),
        });
        return;
    }
    SCHED.jobs.fetch_add(1, Ordering::Relaxed);
    let prefix_ok = true; // NO_PREFIX 폐기 — 접두 캐시 항시
    let pick = (0..slots.len())
        .filter(|&i| slots[i].job.is_none())
        .map(|i| {
            let l = if prefix_ok {
                slots[i]
                    .cached
                    .iter()
                    .zip(j.tokens.iter())
                    .take_while(|(a, b)| a == b)
                    .count()
            } else {
                0
            };
            let full = l > 0 && l == slots[i].cached.len() && j.tokens.len() > l;
            (i, if full { l } else { 0 })
        })
        .max_by_key(|&(_, l)| l);
    let Some((i, reuse)) = pick else {
        return;
    };
    if reuse == 0 {
        if reuse == 0 {
            eng.reset_seq(i);
        }
    } else {
        SCHED
            .prefix_tokens
            .fetch_add(reuse as u64, Ordering::Relaxed);
        eprintln!("# prefix-cache: slot{i} reuse {reuse}토큰");
    }
    let prev = std::mem::take(&mut slots[i].cached);
    let mut sampler_new = j.sampler.clone().map(llm170_core::sampler::Sampler::new);
    if let Some(sm) = &mut sampler_new
        && !sm.is_greedy()
    {
        sm.push_tokens(j.tokens.iter().copied());
    }
    slots[i] = Slot {
        job: Some(j),
        prefilled: reuse,
        next: 0,
        generated: 0,
        tokens: Vec::new(),
        touch: tick,
        cancelled: false,
        cached: prev,
        sampler: sampler_new,
        err_count: 0,
        failed: None,
    };
}
fn slot_emit(s: &mut Slot, t: u32) {
    s.next = t;
    s.tokens.push(t);
    s.generated += 1;
    s.err_count = 0; // QA-1: 성공 — 연속 실패 카운터 리셋
    if let Some(sm) = &mut s.sampler {
        sm.push_tokens([t]);
    }
    if let Some(j) = &s.job
        && let Some(p) = &j.progress
        && p.send(t).is_err()
    {
        // SSE 수신자 소멸(클라이언트 절단) — 즉시 취소 표시
        s.cancelled = true;
    }
}

fn finish_slot(s: &mut Slot, eng: &mut Engine, i: usize, eos: u32) {
    // QA-1: 확정 실패 슬롯 — 에러 결과 전파 후 반환. 종전엔 토큰 방출이
    // 없어 완료 조건이 영구 거짓 → 서버 수명 동안 재시도하는 스피너였다.
    if let Some(err) = s.failed.take()
        && s.job.is_some()
    {
        if let Some(j) = s.job.take() {
            SCHED.requests_failed.fetch_add(1, Ordering::Relaxed);
            let _ = j.out.send(InferResult {
                tokens: Vec::new(),
                error: Some(err),
            });
        }
        eng.reset_seq(i);
        *s = Slot::free(); // 접두 캐시 폐기 — 엔진 시퀀스 상태 신뢰 불가
        return;
    }
    let done = s.job.as_ref().is_some_and(|j| {
        s.prefilled == j.tokens.len()
            && (s.cancelled
                || s.next == eos
                || j.stops.contains(&s.next)
                || s.generated as usize >= j.n_predict.max(1))
    });
    if done {
        if let Some(j) = s.job.take() {
            let mut toks = s.tokens.clone();
            while toks.last() == Some(&eos) || j.stops.contains(toks.last().unwrap_or(&0)) {
                toks.pop();
            }
            toks.truncate(j.n_predict);
            SCHED
                .prompt_tokens
                .fetch_add(j.tokens.len() as u64, Ordering::Relaxed);
            SCHED
                .gen_tokens
                .fetch_add(toks.len() as u64, Ordering::Relaxed);
            let _ = j.out.send(InferResult {
                tokens: toks.clone(),
                error: None,
            });
            // 접두 캐시 — 상태 유지 (프롬프트+생성 = 구워진 열).
            let mut full = j.tokens.clone();
            full.extend(toks);
            s.cached = full;
        } else {
            eng.reset_seq(i);
        }
        let c = std::mem::take(&mut s.cached);
        *s = Slot::free();
        s.cached = c;
    }
}
