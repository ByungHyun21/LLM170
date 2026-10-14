//! 슬롯 스케줄러 (engine.rs에서 순수 이동).
//! 연속 배칭: 디코드 우선·잔여 예산 프리필 청크(llama.cpp 규칙 1:1).
//! [B5 2026-10-09] 청크 가변화(예산−디코드)·라운드로빈·경합 long 캡.
//! 슬롯 = 엔진 시퀀스 id. Engine 열거는 engine.rs(파사드) — 이 층은
//! 배정(slot_loop·assign_slot)과 슬롯 상태만 담는다.
use crate::engine::{Engine, InferResult};
use std::sync::atomic::{AtomicU64, Ordering};

mod assign;
mod decode;
mod econ;
mod prefill;
mod publish;
mod spec;
use self::assign::*;
use self::decode::*;
use self::prefill::*;
use self::publish::*;

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
/// [K0 2026-10-12] progress 프레임 — Tok=토큰(SSE 방출), Ping=생존 프로브
/// (프리필 청크마다 송신 — 수신자 drop이면 즉시 취소로 조기 반납).
pub enum ProgressFrame {
    Tok(u32),
    Ping,
}

pub struct SlotJob {
    pub tokens: Vec<u32>,
    pub n_predict: usize,
    /// 샘플링 파라미터 (기본 greedy — None이면 GPU argmax 경로 유지).
    pub sampler: Option<llm170_core::sampler::SamplerParams>,
    /// 조기 종료 토큰 (EOS + 채팅 템플릿 종결자).
    pub stops: Vec<u32>,
    /// 프레임별 SSE 스트림 채널(Tok=토큰·Ping=생존 프로브 — K0).
    pub progress: Option<std::sync::mpsc::Sender<ProgressFrame>>,
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
    kv_budget: Option<u64>,
    kv_reserve: Option<u64>,
) -> crate::gpu_engine::GpuEngine {
    let _ = crate::engine::ARCH.set(llm170_core::qwen35::bind::dir_arch(p).unwrap_or_default());
    for i in 0..5 {
        match crate::gpu_engine::GpuEngine::load(p, n_slots, ctx, kv_budget, kv_reserve) {
            Ok(mut e) => {
                // [§7.1] 스펙 스냅샷 버퍼 — 실패(VRAM 부족 등) 시 비활성 로그.
                if let Err(err) = e.enable_spec() {
                    eprintln!("# spec 비활성: {err}");
                }
                return e;
            }
            Err(err) => {
                eprintln!("# gpu 엔진 로드 재시도 {}/5: {err}", i + 1);
                // [D2 검수 수정 2026-10-13] 재시도는 간헐 클래스(모듈 로드
                // ENOPT 계열) 한정 — 계약 오류(텐서 부재 등)는 즉시 실패로
                // 5회 OOM 연쇄(무의미·오해 소지)를 끊는다. 실패한 시도의
                // 부분 할당은 GpuEngine::load가 free_all_device로 정리한다.
                // [2.J] VRAM 부족은 재시도 무의미 — 백트레이스 대신 안내 종료.
                if err.contains("메모리 부족") || err.contains("OUT_OF_MEMORY") {
                    eprintln!("# gpu 엔진 로드 실패(VRAM 부족): {err}");
                    eprintln!("# --ctx 를 줄이세요(예: 32768).");
                    std::process::exit(1);
                }
                if !(err.contains("ENOPT") || err.contains("cuModule")) {
                    panic!("gpu 엔진 로드 실패(재시도 불가): {} — {err}", p.display());
                }
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
        }
    }
    panic!("gpu 엔진 로드 최종 실패: {}", p.display())
}

/// 연속 배칭 루프 (04-2). 매 반복: ① 큐 drain → LRU 가용 슬롯 배정
/// ② 디코드 우선(활성 전 슬롯 — q35는 1배치 호출, q4는 슬롯별 decode1)
/// ③ 디코드한 스텝이 없으면 프리필 1청크. 완료/EOS → 슬롯 반환(reset_seq).
/// 108 P5 — 스케줄러 계측(원자). serve 종료 시 요약.
pub static SCHED: Sched = Sched {
    jobs: AtomicU64::new(0),
    queue_wait_us: AtomicU64::new(0),
    prefix_tokens: AtomicU64::new(0),
    // [K1 2026-10-12] 재사용 분해 — 전체-접두 / 체크포인트(2048 GDN) 히트 횟수.
    prefix_full_hits: AtomicU64::new(0),
    prefix_ckpt_hits: AtomicU64::new(0),
    trie_boundaries: AtomicU64::new(0),
    ticks_decode: AtomicU64::new(0),
    ms_decode: AtomicU64::new(0),
    chunks_prefill: AtomicU64::new(0),
    ms_prefill: AtomicU64::new(0),
    spec_rounds: AtomicU64::new(0),
    spec_accepted: AtomicU64::new(0),
    spec_ms: AtomicU64::new(0),
    slots_active: AtomicU64::new(0),
    prompt_tokens: AtomicU64::new(0),
    gen_tokens: AtomicU64::new(0),
    requests: AtomicU64::new(0),
    requests_failed: AtomicU64::new(0),
    queue_depth: AtomicU64::new(0),
    // [2.A-④ 2026-10-13] 경제성 게이트 — EMA(f64 bits)·카운트·상태.
    // 소스별/평범/k밴드 비용(ms/토큰) + 스펙 ON/OFF + 프로브 시계.
    econ_plain: AtomicU64::new(0),
    econ_plain_n: AtomicU64::new(0),
    econ_src: [AtomicU64::new(0), AtomicU64::new(0)],
    econ_src_n: [AtomicU64::new(0), AtomicU64::new(0)],
    econ_k: [
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    ],
    econ_k_n: [
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    ],
    econ_on: AtomicU64::new(1),
    econ_kidx: AtomicU64::new(1),
    econ_src_cur: AtomicU64::new(0),
    econ_probe_alt: AtomicU64::new(0),
    econ_gen: AtomicU64::new(0),
    econ_probe_gen: AtomicU64::new(0),
    econ_rounds: AtomicU64::new(0),
};
pub struct Sched {
    pub jobs: AtomicU64,
    pub queue_wait_us: AtomicU64,
    pub prefix_tokens: AtomicU64,
    /// [K1 2026-10-12] 재사용 분해 카운터(횟수) — 히트율 = prefix_tokens/prompt_tokens.
    pub prefix_full_hits: AtomicU64,
    pub prefix_ckpt_hits: AtomicU64,
    /// [§3.4 ③ 2026-10-12] 접두 트리 512 경계 등록 수(모니터링 — metrics json
    /// trie_boundaries). metrics 표기가 선행 반영됨(계측 배선은 병렬 작업분).
    pub trie_boundaries: AtomicU64,
    pub ticks_decode: AtomicU64,
    pub ms_decode: AtomicU64,
    pub chunks_prefill: AtomicU64,
    pub ms_prefill: AtomicU64,
    /// 스펙 라운드 수/수용 토큰 수 계측(수용률 = acc/rounds).
    pub spec_rounds: AtomicU64,
    pub spec_accepted: AtomicU64,
    /// [A6-3] 스펙 라운드 누적 ms(평균 = ms/rounds).
    pub spec_ms: AtomicU64,
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
    // ── [2.A-④ 2026-10-13] 런타임 경제성 게이트 ──
    /// 평범 스텝 비용 EMA(ms/토큰 — f64 bits)·표본 수.
    pub econ_plain: AtomicU64,
    pub econ_plain_n: AtomicU64,
    /// 소스별 라운드 비용 EMA(ms/방출토큰) — [0]=mtp·[1]=ngram.
    pub econ_src: [AtomicU64; 2],
    pub econ_src_n: [AtomicU64; 2],
    /// (소스,k) 라운드 비용 EMA — 인덱스 = 소스*4 + k밴드 인덱스.
    pub econ_k: [AtomicU64; 8],
    pub econ_k_n: [AtomicU64; 8],
    /// 스펙 ON(1)/평범(0) — 히스테리시스 상태.
    pub econ_on: AtomicU64,
    /// 현재 선택된 k밴드 인덱스({2,4,6,8}).
    pub econ_kidx: AtomicU64,
    /// 현재 선택된 소스(0=mtp·1=ngram) — MTP 초안 유지 여부 판정.
    pub econ_src_cur: AtomicU64,
    /// OFF 재프로브의 소스 교차 토글(0↔1).
    pub econ_probe_alt: AtomicU64,
    /// 토큰 시계(프로브 주기 판정) — slot_emit이 증가.
    pub econ_gen: AtomicU64,
    /// 마지막 프로브 시점의 토큰 시계.
    pub econ_probe_gen: AtomicU64,
    /// 스펙 라운드 시계(탐사 주기 — 32라운드마다 밴드 순환 1회).
    pub econ_rounds: AtomicU64,
}
impl Sched {
    pub fn summary(&self) -> String {
        let jobs = self.jobs.load(Ordering::Relaxed);
        let qw = self.queue_wait_us.load(Ordering::Relaxed);
        let px = self.prefix_tokens.load(Ordering::Relaxed);
        let (fh, ch) = (
            self.prefix_full_hits.load(Ordering::Relaxed),
            self.prefix_ckpt_hits.load(Ordering::Relaxed),
        );
        let td = self.ticks_decode.load(Ordering::Relaxed);
        let md = self.ms_decode.load(Ordering::Relaxed);
        let cp = self.chunks_prefill.load(Ordering::Relaxed);
        let mp = self.ms_prefill.load(Ordering::Relaxed);
        let sr = self.spec_rounds.load(Ordering::Relaxed);
        let sa = self.spec_accepted.load(Ordering::Relaxed);
        let sms = self.spec_ms.load(Ordering::Relaxed);
        let qd = self.queue_depth.load(Ordering::Relaxed);
        format!(
            "[sched] jobs {jobs} | queue {qd} | queue-wait avg {:.0}ms | prefix-reuse {px}tok(f{fh}/c{ch}) | decode {td}x avg {:.1}ms | prefill {cp}x avg {:.1}ms | spec {sr}r acc {sa} ({:.2}/r) avg {:.1}ms",
            if jobs > 0 {
                qw as f64 / jobs as f64 / 1e3
            } else {
                0.0
            },
            if td > 0 { md as f64 / td as f64 } else { 0.0 },
            if cp > 0 { mp as f64 / cp as f64 } else { 0.0 },
            if sr > 0 { sa as f64 / sr as f64 } else { 0.0 },
            if sr > 0 {
                sms as f64 / sr as f64 / 1e3
            } else {
                0.0
            },
        )
    }
}

pub fn slot_loop(mut eng: Engine, rx: std::sync::mpsc::Receiver<SlotJob>, n_slots: usize) {
    // [§3.4 ③] 전역 접두 트리 — 슬롯 cached 승격(호스트 조회·무손실).
    let mut ptrie = crate::tree::PrefixTrie::new();
    // EOS 하드코드 248044 → 모델 메타 파생(Engine::eos).
    let eos = eng.eos();
    // [K6-3 2026-10-13] 입장 보류열 — 예약이 풀 여유에 안 맞는 요청(FCFS 대기).
    let mut pending: Vec<SlotJob> = Vec::new();
    // 모니터링 — 복사 계측 기준(가중치 적재분을 load로 분리, 이후는 runtime).
    let copy_base = eng.copy_stats();
    // 기동 워밍업 — 첫 요청이 지연 초기화(raw_init, ctx 비례 수십 초)를
    // 뒤집어쓰지 않도록 여기서 소진하고 상태를 되돌린다. 준비 전에는 /health가
    // 503이라 클라이언트가 계측을 시작하지 않는다.
    {
        let warm: Vec<u32> = vec![1u32; 16];
        // [K6-3 2026-10-13] 온디맨드 풀 — 워밍업 전 슬롯 0 KV 확보(실패는 치명 아님).
        if let Err(e) = eng.0.kv_grow(0, warm.len() + 8) {
            eprintln!("# warmup KV 확보 실패(치명 아님): {e}");
        }
        let w: Result<(), String> = eng.0.prefill(0, &warm).and_then(|l| {
            let t = llm170_core::qwen35::greedy(&l);
            eng.0.decode_greedy(0, t).map(|_| ())
        });
        if let Err(err) = w {
            eprintln!("# warmup 실패(치명 아님): {err}");
        }
        // [K6-3] 워밍업 점유 반납 — reset_seq가 상태 리셋 + KV 블록 해제
        // (방치 시 소형 풀에서 대형 예약 요청이 영원히 대기하는 결함).
        let _ = eng.0.reset_seq(0);
    }
    crate::http::READY.store(true, std::sync::atomic::Ordering::Release);
    let mut slots: Vec<Slot> = (0..n_slots).map(|_| Slot::free()).collect();
    let mut tick: u64 = 0;
    let (mut n_dec, mut n_pf) = (0u64, 0u64);
    let (mut ms_dec, mut ms_pf, mut ms_fin) = (0f64, 0f64, 0f64);
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
        // [R7] ① 유입 — 큐 drain → 슬롯 배정(헬퍼 분리 · 동작 불변).
        slot_intake(&mut slots, &mut eng, &rx, tick, &mut ptrie, &mut pending);
        // [§3.4 ③] 트리 동기화 — free 슬롯의 cached를 트리에 등록/해제.
        // 512 경계만 기록되므로 매 틱 재등록은 중복 없음(insert가 슬롯 중복 체크).
        for (i, s) in slots.iter().enumerate() {
            if s.job.is_none() && s.cached.len() >= 512 {
                ptrie.insert(&s.cached, i);
            }
        }
        // [R7] 게시 스로틀(≤1Hz — 헬퍼 분리 · 동작 불변).
        maybe_publish(&slots, &eng, copy_base, &mut last_pub);
        tick += 1;
        let _it0 = std::time::Instant::now();

        // [R7] ② 디코드 우선 — 활성 전 슬롯(헬퍼 분리 · 동작 불변).
        let (active, decoded, dec_ms, fin_ms) = slot_decode(&mut slots, &mut eng, eos);
        ms_fin += fin_ms;

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
            // [R7] ③ 프리필 1청크(헬퍼 분리 · 동작 불변). None = 클라이언트
            // 절단 — 종전 continue 경로(잔여 카운터·finish 생략) 그대로.
            if let Some(pf_ms) = slot_prefill(&mut eng, &mut slots, &active, tick, eos, npw, &t0w) {
                n_pf += 1;
                ms_pf += pf_ms;
            } else {
                continue;
            }
        }

        if decoded {
            n_dec += 1;
            ms_dec += dec_ms;
            SCHED.ticks_decode.fetch_add(1, Ordering::Relaxed);
            SCHED.ms_decode.fetch_add(dec_ms as u64, Ordering::Relaxed);
            if llm170_diag::dump::opts().key("srv_time") && n_dec % 32 == 0 {
                eprintln!(
                    "[srv] steps={} decode avg {:.1}ms (결과 처리 {:.2}ms) | prefill {}x avg {:.1}ms | {}",
                    n_dec,
                    ms_dec / n_dec as f64,
                    ms_fin / n_dec as f64,
                    n_pf,
                    if n_pf > 0 { ms_pf / n_pf as f64 } else { 0.0 },
                    SCHED.summary()
                );
                // [S13] GPU 범주 분해(LLM170_TIME=1 시 — 캡처 밖 발사만 집계).
                if let Ok(p) = eng.0.prof_report("serve")
                    && !p.is_empty()
                {
                    eprintln!("{p}");
                }
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
                    // [K6-3] 유휴 fast-path도 동일 입장 규약 — 검사 없이 배정하면
                    // 예약 초과가 로드 중 실패로 돌아온다(사용 중 실패 금지 위반).
                    let need = j_need_tokens(&j);
                    if !admit_ok(&eng, &j) {
                        if eng.0.kv_blocks_needed(need) > eng.0.kv_total_blocks() {
                            SCHED.requests_failed.fetch_add(1, Ordering::Relaxed);
                            let _ = j.out.send(InferResult {
                                tokens: Vec::new(),
                                error: Some(format!(
                                    "request exceeds the KV pool: need {need} tokens (prompt + max_tokens) > pool capacity"
                                )),
                            });
                            continue;
                        }
                        // 유휴 = 전 슬롯 캐시 양보 가능 — 회수 후 재시도.
                        while !admit_ok(&eng, &j) && reclaim_idle(&mut eng, &mut slots) {}
                        if !admit_ok(&eng, &j) {
                            SCHED.requests_failed.fetch_add(1, Ordering::Relaxed);
                            let _ = j.out.send(InferResult {
                                tokens: Vec::new(),
                                error: Some(format!(
                                    "request exceeds the KV pool: need {need} tokens (prompt + max_tokens) > pool capacity"
                                )),
                            });
                            continue;
                        }
                    }
                    // 107 W7: 배정 단일 구현 위임(전 슬롯 접두 탐색으로 개선 — 종전 slot0 고정).
                    assign_slot(&mut slots, &mut eng, j, tick, &mut ptrie);
                }
                Err(_) => break,
            }
        }
    }
    eprintln!("{}", SCHED.summary());
}

/// [R7] 틱 ① 유입 — 큐 drain → 가용 슬롯 배정(슬롯 루프에서 순수 이동).
/// 회귀 픽스(2026-09-16): 종전엔 try_recv로 꺼낸 뒤 "슬롯 점유"를 발견하면
/// break했다 — 꺼낸 작업이 그대로 버려져(송신측 drop → 수신측 즉시 Err)
/// 동시 요청이 빈 응답으로 소실됐다(4동시 중 여럿 drop 실측). 점유 검사를
/// try_recv **앞으로** 옮겨 작업을 큐에 남긴다 — 의도된 원래 계약.
fn slot_intake(
    slots: &mut [Slot],
    eng: &mut Engine,
    rx: &std::sync::mpsc::Receiver<SlotJob>,
    tick: u64,
    ptrie: &mut crate::tree::PrefixTrie,
    pending: &mut Vec<SlotJob>,
) {
    // [K6-3 2026-10-13] 입장 보장 — 예약(프롬프트+생성한도+슬랙)이 풀 여유에
    // 맞아야 배정. **보류분 우선 재시도(도착 순서 = 기본 FCFS)** 후 신규 수신.
    let mut i = 0;
    while i < pending.len() {
        if !slots.iter().any(|s| s.job.is_none()) {
            break;
        }
        if admit_ok(eng, &pending[i]) {
            let j = pending.remove(i);
            assign_slot(slots, eng, j, tick, ptrie);
        } else {
            // [K6-3] 용량 자체 초과(무슨 수로도 불가) = 조기 거부(무한 대기 방지).
            let need = j_need_tokens(&pending[i]);
            if eng.0.kv_blocks_needed(need) > eng.0.kv_total_blocks() {
                let j = pending.remove(i);
                SCHED.requests_failed.fetch_add(1, Ordering::Relaxed);
                let _ = j.out.send(InferResult {
                    tokens: Vec::new(),
                    error: Some(format!(
                        "request exceeds the KV pool: need {need} tokens (prompt + max_tokens) > pool capacity"
                    )),
                });
                continue;
            }
            // [K6-3] 유휴 슬롯 접두 캐시 회수(1개) 후 재시도 — 비활성 슬롯은
            // 캐시일 뿐이므로 압박 시 양보(사다리 1단계).
            if reclaim_idle(eng, slots) {
                continue;
            }
            i += 1;
        }
    }
    loop {
        if !slots.iter().any(|s| s.job.is_none()) {
            break;
        }
        let Ok(j) = rx.try_recv() else {
            break;
        };
        SCHED.queue_depth.fetch_sub(1, Ordering::Relaxed);
        if admit_ok(eng, &j) {
            // 107 W7: 배정 단일 구현으로 위임(접두 캐시 로직 동일).
            assign_slot(slots, eng, j, tick, ptrie);
        } else if j_need_tokens(&j) > 0
            && eng.0.kv_blocks_needed(j_need_tokens(&j)) > eng.0.kv_total_blocks()
        {
            // [K6-3] 풀 용량 초과 — 무한 대기 대신 정직 거부.
            SCHED.requests_failed.fetch_add(1, Ordering::Relaxed);
            let _ = j.out.send(InferResult {
                tokens: Vec::new(),
                error: Some(format!(
                    "request exceeds the KV pool: need {} tokens (prompt + max_tokens) > pool capacity",
                    j_need_tokens(&j)
                )),
            });
        } else if reclaim_idle(eng, slots) {
            // [K6-3] 유휴 슬롯 캐시 회수 후 재시도(다음 틱).
            pending.push(j);
        } else {
            // [K6-3] 용량 대기 — 스케줄러 보류열(HTTP 큐로 안 돌림: FCFS 유지).
            pending.push(j);
        }
    }
}

/// [K6-3] 요청 예약 토큰 = 프롬프트 + 생성한도 + 슬랙 2.
fn j_need_tokens(j: &SlotJob) -> usize {
    j.tokens.len() + j.n_predict.max(1) + 2
}

/// [K6-3] 유휴 슬롯(job 없음)의 KV 블록 1개 슬롯 회수 — LRU(가장 오래된
/// touch) 우선. 접두 캐시는 압박 시 양보(사다리 1단계). 반환 = 회수 여부.
fn reclaim_idle(eng: &mut Engine, slots: &mut [Slot]) -> bool {
    let victim = slots
        .iter()
        .enumerate()
        .filter(|(_, s)| s.job.is_none())
        .min_by_key(|(_, s)| s.touch)
        .map(|(i, _)| i);
    match victim {
        Some(i) => {
            let freed = eng.0.kv_release_slot(i);
            if freed > 0 {
                slots[i].cached.clear();
                eprintln!("# kv 압박: 슬롯{i} 접두 캐시 회수({freed}블록)");
                true
            } else {
                false
            }
        }
        None => false,
    }
}

/// [K6-3 2026-10-13] 입장 판정 — 필요 블록(프롬프트+생성한도+슬랙 2행)이
/// 풀 여유 이하. 슬롯 보유 접두 블록은 보수적으로 미계상(과대 판정 = 대기
/// 우선 — 활성 요청 안전). 큐 대기의 단일 기준.
fn admit_ok(eng: &Engine, j: &SlotJob) -> bool {
    let need = j_need_tokens(j);
    eng.0.kv_blocks_needed(need) <= eng.0.kv_free_blocks()
}

#[cfg(test)]
mod tests {
    use super::Slot;

    /// [R14] 스펙 적응 게이트 경계 — 무수용 후 256토큰 재시도 조건.
    #[test]
    fn spec_gate_boundary() {
        let mut s = Slot::free();
        s.spec_miss = 0;
        // 무수용 시 spec_miss 증가(게이트 활성화)
        s.spec_miss += 1;
        assert_eq!(s.spec_miss, 1);
        // 256토큰마다 재시도 = generated & 255 == 0 조건
        s.generated = 256;
        assert_eq!(s.generated & 255, 0, "256 배수에서 재시도 허용");
        s.generated = 257;
        assert_ne!(s.generated & 255, 0, "256 배수 외 재시도 차단");
    }

    /// [R14] 접두 트리 경계 — 512 미만 미스·512 이상 히트·제거 후 미스.
    #[test]
    fn prefix_trie_boundaries() {
        let mut t = crate::tree::PrefixTrie::new();
        let tokens: Vec<u32> = (1..=600).collect();
        t.insert(&tokens, 0);
        // 512 경계 히트
        assert_eq!(t.longest_prefix(&tokens), Some((0, 512)));
        // 512 미만 조회 미스
        let short: Vec<u32> = (1..100).collect();
        assert_eq!(t.longest_prefix(&short), None);
        // 제거 후 미스
        t.remove(0);
        assert_eq!(t.longest_prefix(&tokens), None);
    }
}
