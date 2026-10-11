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
/// [§2 2026-10-11] 혼합 정책 — 타 슬롯이 디코드 중이면 프리필 청크를 이 값으로
/// 캡. 512청크는 27B에서 ~443ms 동안 타 슬롯 디코드를 정지(실측). 단독
/// 프리필(벤치마크·첫 요청)은 종전 최대 청크 유지 — 토큰 산술 불변.
const PF_CHUNK_MIXED: usize = 128;
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
    /// [§7.5] 스펙 적응 게이트 — 연속 무수용 라운드 수(1 도달 시 256토큰마다 재시도).
    spec_miss: u32,
    /// [§7.5] 적응 초안 폭 — 전량 수용이면 확대(≤8=TR 도메인), 부진하면 4로 복귀.
    spec_k: usize,
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
            spec_miss: 0,
            spec_k: SPEC_K,
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
            Ok(mut e) => {
                // [§7.1] 스펙 스냅샷 버퍼 — 실패(VRAM 부족 등) 시 비활성 로그.
                if let Err(err) = e.enable_spec() {
                    eprintln!("# spec 비활성: {err}");
                }
                return e;
            }
            Err(err) => {
                eprintln!("# gpu 엔진 로드 재시도 {}/5: {err}", i + 1);
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
        }
    }
    panic!("gpu 엔진 로드 최종 실패: {}", p.display())
}

/// 슬롯 로짓 → 토큰: 활성 샘플러면 sample, 아니면 greedy (동률 최저 인덱스).
fn pick(s: &mut Slot, logits: &mut [f32]) -> u32 {
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
    // [§7.1 2026-10-11] 스펙 서빙 — 초안이 발화하는 그리디 슬롯은 검증
    // 라운드(직렬). 값 보존: 검증+보정 토큰이 t=1 디코드와 비트동일
    // (스펙 골든 계약 — spec_round 주석 참조). 나머지는 종전 배치/단독.
    let mut g2: Vec<usize> = Vec::with_capacity(greedy.len());
    for &i in &greedy {
        match spec_round(i, &mut slots[i], e) {
            Ok(true) => {}
            Ok(false) => g2.push(i),
            Err(err) => slot_fail(&mut slots[i], format!("spec: {err}")),
        }
    }
    let greedy = g2;
    // [§7 2단계] 샘플드 스펙 — 초안 발화 슬롯은 검증 라운드(직렬), 나머지 배치.
    // [S11 2026-10-11] 혼합 통합을 위해 그리디 배치보다 먼저 처리(슬롯 독립 —
    // 순서 무관).
    let mut s2: Vec<usize> = Vec::with_capacity(sampled.len());
    for &i in &sampled {
        match spec_round_sampled(i, &mut slots[i], e) {
            Ok(true) => {}
            Ok(false) => s2.push(i),
            Err(err) => slot_fail(&mut slots[i], format!("spec(샘플): {err}")),
        }
    }
    let sampled = s2;
    // [S11] 혼합(그리디+샘플) — 한 배치 체인으로 통합(종전 np_k + np_m 2회).
    // 실패 시 분리 경로로 폴백.
    if !greedy.is_empty() && !sampled.is_empty() {
        let mut uslots: Vec<usize> = Vec::with_capacity(greedy.len() + sampled.len());
        let mut utoks: Vec<u32> = Vec::with_capacity(uslots.capacity());
        for &i in greedy.iter().chain(sampled.iter()) {
            uslots.push(i);
            utoks.push(slots[i].next);
        }
        match e.decode_np_mixed(&uslots, &utoks) {
            Ok((toks, mut flat)) => {
                let hn = flat.len() / uslots.len();
                for (row, (chunk, &i)) in flat.chunks_mut(hn).zip(uslots.iter()).enumerate() {
                    let smp = slots[i].sampler.as_ref().is_some_and(|s| !s.is_greedy());
                    let t = if smp {
                        pick(&mut slots[i], chunk)
                    } else {
                        toks[row]
                    };
                    slot_emit(&mut slots[i], t);
                }
                return;
            }
            Err(err) => eprintln!("# 혼합 배치 폴백(분리): {err}"),
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
        // [E-1 후속 2026-10-10] 단독 그리디는 **GPU argmax**(4B d2h) —
        // 종전 logits 경로 합류는 토큰당 로짓 993KB pageable D2H + CPU
        // argmax(248K)를 지불했다(실측 ~0.5ms/토큰). 샘플링 슬롯만 logits 판.
        match e.decode_greedy(i, slots[i].next) {
            Ok(t) => slot_emit(&mut slots[i], t),
            Err(err) => slot_fail(&mut slots[i], format!("decode_greedy: {err}")),
        }
    }
    if !sampled.is_empty() {
        let toks: Vec<u32> = sampled.iter().map(|&i| slots[i].next).collect();
        // [S11 2026-10-11] t≥2 샘플링은 배치 체인 1회(로짓 d2h) — 종전
        // 슬롯별 t=1 직렬(≈19.5ms×N) 대비 대폭 상각. 로짓은 행별 t=1 head와
        // 비트동일이라 호스트 샘플러 토큰열 불변. t=1은 종전 직렬 유지.
        match e.decode_np_logits(&sampled, &toks) {
            Ok(mut flat) => {
                let hn = flat.len() / sampled.len();
                let mut off = 0usize;
                for &i in &sampled {
                    let (row, _rest) = flat[off..].split_at_mut(hn);
                    let t = pick(&mut slots[i], row);
                    slot_emit(&mut slots[i], t);
                    off += hn;
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
    queue_depth: AtomicU64::new(0),
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
    /// [B5/I 2026-10-10] 큐 깊이 게이지 — enqueue 성공 +1, 스케줄러 수신 -1.
    /// 429 어드미션 게이트와 정합(포화 시 try_send 실패라 증가 없음).
    pub queue_depth: AtomicU64,
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
        let qd = self.queue_depth.load(Ordering::Relaxed);
        format!(
            "[sched] jobs {jobs} | queue {qd} | queue-wait avg {:.0}ms | prefix-reuse {px}tok | decode {td}x avg {:.1}ms | prefill {cp}x avg {:.1}ms | spec {sr}r acc {sa} ({:.2}/r)",
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
        let w: Result<(), String> = eng.0.prefill(0, &warm).and_then(|l| {
            let t = llm170_core::qwen35::greedy(&l);
            eng.0.decode_greedy(0, t).map(|_| ())
        });
        if let Err(err) = w {
            eprintln!("# warmup 실패(치명 아님): {err}");
        }
        let _ = eng.0.reset_states();
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
            SCHED.queue_depth.fetch_sub(1, Ordering::Relaxed);
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
            q35_decode(&mut eng.0, &mut slots, &active);
            dec_ms = _dt.elapsed().as_secs_f64() * 1e3;
            // 완료 슬롯 정리 — 결과 전송·반환
            for &i in &active {
                finish_slot(&mut slots[i], &mut eng, i, eos);
            }
        }

        // ③ 프리필 1청크 — 디코드가 없었던 회차이거나, 아직 프리필이 남은 대기 슬롯이
        // 있는 경우. (디코드 우선이지만 활성 슬롯의 디코드가 대기 슬롯의 프리필을
        // 영구히 굶기면 서버가 요청을 직렬화한다 — np4 실측 2026-09-12.)
        // [A1 기각 2026-10-11] 디코드-프리필 2-스트림 병행(별도 스크래치 사본 +
        // 발사/회수 분리)을 구현·실측했으나 무효 — 디코드 최대 갭 143ms vs
        // 기준선 147ms(노이즈). 이유(구조적): 디코드 토큰(②)과 프리필 회수(③)가
        // 같은 틱이라 갭 하한 = 프리필 청크 시간(~132ms) — 병행이 숨기는 건
        // 디코드 몫 15ms뿐(이론 상한 −10%)이고 발사 큐잉(~10ms)과 상쇄된다.
        // 회수를 다음 틱으로 미루는 파이프라이닝은 pf 스크래치 2벌(~112필드·
        // VRAM 2배)이 필요해 상한 대비 비용이 크다. 170HX(DRAM 헤드룸) 재검토 여지.
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
                // [§2] 혼합 — 타 슬롯이 이미 디코드 중(프리필 완료+잔존)이면
                // 청크 캡으로 디코드 정지 시간을 유계화(프리필 총량은 동일).
                let others_decoding = slots.iter().enumerate().any(|(j, s)| {
                    j != i && s.job.is_some() && s.prefilled == s.job.as_ref().unwrap().tokens.len()
                });
                if others_decoding {
                    chunk = chunk.min(PF_CHUNK_MIXED);
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
                    let r: Result<u32, String> = eng.0.prefill(i, &part).map(|mut l| {
                        if samp {
                            pick(&mut slots[i], &mut l)
                        } else {
                            llm170_core::qwen35::greedy(&l)
                        }
                    });
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
                    SCHED.queue_depth.fetch_sub(1, Ordering::Relaxed);
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
    let Some((i, l)) = pick_prefix_slot(slots, &j.tokens) else {
        return;
    };
    // [A2] 재사용 판정 — 전체 접두(상태가 그 끝 위치와 정합) 우선, 부분은
    // 체크포인트(2048 경계 GDN 상태) ≤ l. 부분은 restore로 상태·pos 복원.
    let full = l > 0 && l == slots[i].cached.len();
    let ck = eng.0.checkpoint_pos(i) as usize;
    let mut reuse = if full {
        l
    } else if ck > 0 && ck <= l {
        ck
    } else {
        0
    };
    if reuse == 0 {
        eng.reset_seq(i);
    } else if full {
        SCHED
            .prefix_tokens
            .fetch_add(reuse as u64, Ordering::Relaxed);
        eprintln!("# prefix-cache: slot{i} reuse {reuse}토큰(전체)");
    } else {
        match eng.0.restore_gdn(i) {
            Ok(p) if p as usize == reuse => {
                SCHED
                    .prefix_tokens
                    .fetch_add(reuse as u64, Ordering::Relaxed);
                eprintln!("# prefix-cache: slot{i} reuse {reuse}토큰(체크포인트)");
            }
            _ => {
                eng.reset_seq(i);
                reuse = 0;
            }
        }
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
        spec_miss: 0,
        spec_k: SPEC_K,
    };
}
/// [§7.1 2026-10-11] 스펙 라운드 — n-gram 초안(4-gram·최대 SPEC_K-1) 발화 시
/// 1회. Ok(true)=라운드 수행(토큰 방출), Ok(false)=초안 없음(종전 경로).
/// 값 보존: 검증 배치가 전 위치 토큰을 내고, 수용분+보정(t=1 재계산)이
/// 단독 디코드와 동일한 토큰열을 만든다(프로브 [A-1 완화 ①]과 동일 절차 —
/// 35B/27B 스펙 골든 접두가 판정).
const SPEC_K: usize = 4;
fn spec_round(
    i: usize,
    s: &mut Slot,
    e: &mut crate::gpu_engine::GpuEngine,
) -> Result<bool, String> {
    // [§7.5 2026-10-11] 적응 게이트 — 연속 무수용 2회면 128토큰마다만 재시도.
    // 산문 벤치 실측(서빙 A/B): 초안이 발화하나 수용 0이 반복되면 검증 비용
    // (~0.85틱)이 순손실 — tg −1.6%·np4 −5.7%. 수용 시 즉시 해제.
    if s.spec_miss >= 1 && (s.generated & 255) != 0 {
        return Ok(false);
    }
    let mut hist: Vec<u32> = Vec::new();
    if let Some(j) = &s.job {
        hist.extend_from_slice(&j.tokens);
    }
    hist.extend_from_slice(&s.tokens);
    // [§7.5] 적응 폭 — 전량 수용이면 +1(≤8 — 검증 t≤8이 TR GEMV 도메인),
    // 부진(≤1)이면 4로 복귀. 초안 폭은 수용 판정만 좌우(값 불변).
    let draft = llm170_core::spec::ngram_draft(&hist, 3, 48, s.spec_k - 1);
    if draft.is_empty() {
        return Ok(false);
    }
    let t = 1 + draft.len();
    let mut rows: Vec<f32> = Vec::with_capacity(t * e.n_embd());
    rows.extend_from_slice(&e.embed_row(s.next)?);
    for &tk in &draft {
        rows.extend_from_slice(&e.embed_row(tk)?);
    }
    e.spec_save_state(i)?;
    let toks = e.spec_verify(i, &rows, t)?;
    let mut acc = 0usize;
    while acc < draft.len() && toks[acc] == draft[acc] {
        acc += 1;
    }
    let last = if acc >= 1 { draft[acc - 1] } else { s.next };
    if acc >= 1 {
        e.spec_rollback(i, acc)?;
        e.spec_rewind_pos(i, (t - acc) as u32)?;
    } else {
        e.spec_load_state(i)?;
        e.spec_rewind_pos(i, t as u32)?;
    }
    if acc == 0 {
        s.spec_miss = (s.spec_miss + 1).min(3);
    } else {
        s.spec_miss = 0;
    }
    if acc == draft.len() {
        s.spec_k = (s.spec_k + 1).min(8);
    } else if acc <= 1 {
        s.spec_k = 4;
    }
    let corr = e.decode_greedy(i, last)?;
    for &tk in &draft[..acc] {
        slot_emit(s, tk);
    }
    slot_emit(s, corr);
    SCHED.spec_rounds.fetch_add(1, Ordering::Relaxed);
    SCHED.spec_accepted.fetch_add(acc as u64, Ordering::Relaxed);
    Ok(true)
}

/// [§7 2단계 2026-10-11] 비greedy 스펙 라운드 — target-only 거절샘플링.
/// 수용: coin < p(x_k)(prob_of — 후처리 체인 미러, 수용 초안 가산 페널티).
/// 보정: t=1 재계산 로짓의 표본 — 거절 시 잔차(≠x 재추출 = 조건분포 정확).
/// 방출 분포 = 표적 분포(기대 수용 = E[p]). 그리디 라운드와 롤백 구조 동일.
fn spec_round_sampled(
    i: usize,
    s: &mut Slot,
    e: &mut crate::gpu_engine::GpuEngine,
) -> Result<bool, String> {
    if s.sampler.is_none() || (s.spec_miss >= 1 && (s.generated & 255) != 0) {
        return Ok(false);
    }
    let mut hist: Vec<u32> = Vec::new();
    if let Some(j) = &s.job {
        hist.extend_from_slice(&j.tokens);
    }
    hist.extend_from_slice(&s.tokens);
    let draft = llm170_core::spec::ngram_draft(&hist, 3, 48, s.spec_k - 1);
    if draft.is_empty() {
        return Ok(false);
    }
    let t = 1 + draft.len();
    let mut rows: Vec<f32> = Vec::with_capacity(t * e.n_embd());
    rows.extend_from_slice(&e.embed_row(s.next)?);
    for &tk in &draft {
        rows.extend_from_slice(&e.embed_row(tk)?);
    }
    e.spec_save_state(i)?;
    let lg = e.spec_verify_logits(i, &rows, t)?;
    let vocab = e.vocab();
    if lg.len() != t * vocab {
        return Err(format!(
            "spec(샘플) 로짓 크기 {} != {}",
            lg.len(),
            t * vocab
        ));
    }
    let mut acc = 0usize;
    for k in 0..draft.len() {
        let mut lk = lg[k * vocab..(k + 1) * vocab].to_vec();
        let sm = s.sampler.as_mut().unwrap();
        sm.apply_penalties_ext(&mut lk, &draft[..k]);
        let px = sm.prob_of(&lk, draft[k]);
        let coin = sm.next_coin();
        if coin < px {
            acc += 1;
        } else {
            break;
        }
    }
    let last = if acc >= 1 { draft[acc - 1] } else { s.next };
    if acc >= 1 {
        e.spec_rollback(i, acc)?;
        e.spec_rewind_pos(i, (t - acc) as u32)?;
    } else {
        e.spec_load_state(i)?;
        e.spec_rewind_pos(i, t as u32)?;
    }
    if acc == 0 {
        s.spec_miss = (s.spec_miss + 1).min(3);
    } else {
        s.spec_miss = 0;
    }
    if acc == draft.len() {
        s.spec_k = (s.spec_k + 1).min(8);
    } else if acc <= 1 {
        s.spec_k = 4;
    }
    for &tk in &draft[..acc] {
        slot_emit(s, tk);
    }
    let lg1 = e.decode(&[i], &[last])?.pop().unwrap_or_default();
    if lg1.len() != vocab {
        return Err(format!("spec(샘플) 보정 로짓 {} != {vocab}", lg1.len()));
    }
    let mut corr = {
        let mut l = lg1.clone();
        s.sampler.as_mut().unwrap().sample(&mut l)
    };
    if acc < draft.len() {
        let x = draft[acc];
        let mut guard = 0;
        while corr == x && guard < 64 {
            let mut l = lg1.clone();
            corr = s.sampler.as_mut().unwrap().sample(&mut l);
            guard += 1;
        }
    }
    slot_emit(s, corr);
    SCHED.spec_rounds.fetch_add(1, Ordering::Relaxed);
    SCHED.spec_accepted.fetch_add(acc as u64, Ordering::Relaxed);
    Ok(true)
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

/// [I 2026-10-10] 접두 캐시 재사용 슬롯 선택 — **순수 함수**(테스트 표면).
/// free 슬롯(job 없음) 중 cached와 tokens의 공통 접두가 최장인 곳. 반환
/// (slot, reuse): reuse > 0은 cached **전체**가 접두이고 토큰이 더 길 때만
/// (정확히 같은 길이는 상태가 이미 구워져 있으므로 재사용 아님 — reset 생략
/// 판정은 호출부).
fn pick_prefix_slot(slots: &[Slot], tokens: &[u32]) -> Option<(usize, usize)> {
    // [A2 2026-10-11] 부분 접두까지 l 반환(순수 함수 유지 — 전체/부분/체크포인트
    // 판정은 호출부가 엔진 체크포인트 조회와 함께 수행).
    slots
        .iter()
        .enumerate()
        .filter(|(_, s)| s.job.is_none())
        .map(|(i, s)| {
            let l = s
                .cached
                .iter()
                .zip(tokens.iter())
                .take_while(|(a, b)| a == b)
                .count();
            (i, l)
        })
        .max_by_key(|&(_, l)| l)
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

#[cfg(test)]
mod tests {
    use super::{Slot, pick_prefix_slot};

    fn slot_with(cached: Vec<u32>) -> Slot {
        let mut s = Slot::free();
        s.cached = cached;
        s
    }

    /// [I] 접두 선택 — 빈 슬롯 없음/전부 점유 = None.
    #[test]
    fn pick_none_cases() {
        assert!(pick_prefix_slot(&[], &[1, 2, 3]).is_none());
        let mut busy = slot_with(vec![1, 2]);
        busy.job = None; // job 없음 + cached 유지 = 자유 슬롯(재사용 판정은 아래)
        // 전부 점유(가짜 점유는 job Some 필요 — out 생략 위해 free로 검증 불가하므로
        // "cached 비었고 공통 접두 0" 케이스로 대체).
        let free = slot_with(Vec::new());
        assert_eq!(pick_prefix_slot(&[free], &[1, 2, 3]), Some((0, 0)));
    }

    /// [I] 최장 접두 슬롯 선택 + 전체 접두(더 긴 토큰) 재사용 판정.
    #[test]
    fn pick_longest_prefix() {
        let a = slot_with(vec![1, 2, 3, 4]);
        let b = slot_with(vec![1, 2, 9]);
        let c = slot_with(vec![7, 8]);
        // 공통 접두: a=4(전체+토큰 더 김→재사용), b=2, c=0 → a 선택.
        assert_eq!(pick_prefix_slot(&[a, b, c], &[1, 2, 3, 4, 5]), Some((0, 4)));
    }

    /// [I] 정확히 같은 길이는 재사용 아님(reuse=0 — 상태가 이미 그 열).
    #[test]
    fn pick_exact_length_no_reuse() {
        let a = slot_with(vec![1, 2, 3]);
        assert_eq!(pick_prefix_slot(&[a], &[1, 2, 3]), Some((0, 0)));
    }

    /// [I] 접두 0이어도 슬롯은 반환(reuse=0 — reset 경로).
    #[test]
    fn pick_no_prefix_still_slot() {
        let a = slot_with(vec![9, 9]);
        assert_eq!(pick_prefix_slot(&[a], &[1, 2, 3]), Some((0, 0)));
    }
}
