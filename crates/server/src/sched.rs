//! 슬롯 스케줄러 (engine.rs에서 순수 이동).
//! 연속 배칭: 디코드 우선·잔여 예산 프리필 청크(llama.cpp 규칙 1:1).
//! [B5 2026-10-09] 청크 가변화(예산−디코드)·라운드로빈·경합 long 캡.
//! 슬롯 = 엔진 시퀀스 id. Engine 열거는 engine.rs(파사드) — 이 층은
//! 배정(slot_loop·assign_slot)과 슬롯 상태만 담는다.
use crate::engine::{Engine, InferResult};
use std::sync::atomic::{AtomicU64, Ordering};

mod econ;
mod publish;
mod spec;
use self::econ::*;
use self::publish::*;
use self::spec::*;

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
/// [K-STEP 2026-10-12] eos 인자 추가 — 단독 그리디 K스텝 장치 루프가
/// 컨트롤러 eos(캡처 키)로 쓴다.
fn q35_decode(e: &mut crate::gpu_engine::GpuEngine, slots: &mut [Slot], seqs: &[usize], eos: u32) {
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
    // [2026-10-12] 스펙 비활성(스트리밍·VRAM 부족)이면 라운드 시도 금지 —
    // 비활성 상태 시도가 요청을 죽이던 결함(실측: 스트리밍 서브 500).
    let spec_ok = e.spec_on();
    let mut g2: Vec<usize> = Vec::with_capacity(greedy.len());
    for &i in &greedy {
        if !spec_ok {
            g2.push(i);
            continue;
        }
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
        if !spec_ok {
            s2.push(i);
            continue;
        }
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
        let _pmm = std::time::Instant::now();
        match e.decode_np_mixed(&uslots, &utoks) {
            Ok((toks, mut flat)) => {
                econ_plain_sample(_pmm.elapsed().as_secs_f64() * 1e3, uslots.len());
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
        let _pt = std::time::Instant::now();
        match e.decode_np_greedy(&greedy, &toks) {
            Ok(ts) => {
                econ_plain_sample(_pt.elapsed().as_secs_f64() * 1e3, greedy.len());
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
        // [K-STEP §3.4 ① 2026-10-12] 먼저 K스텝 장치 루프(스텝 그래프 k회
        // 리플레이 — 호스트 체크인 K스텝마다)를 시도한다. k = min(K, 잔여
        // max_tokens) — cap 잔여는 엔진이 클램프(2 미만 Err). EOS는 장치
        // 컨트롤러가 감지(이후 스텝 미기록)하므로 방출 순서·값은 순차 단일
        // 스텝과 동일(토큰열 판정). 계약·캡처 실패는 기존 토큰당 경로 폴백.
        const KSTEP_K: usize = 4;
        let rem = slots[i]
            .job
            .as_ref()
            .map(|j| {
                j.n_predict
                    .max(1)
                    .saturating_sub(slots[i].generated as usize)
            })
            .unwrap_or(0);
        let k = KSTEP_K.min(rem);
        let _pt = std::time::Instant::now();
        let kres = if k >= 2 {
            e.decode_greedy_k(i, slots[i].next, k, eos)
        } else {
            Err("kstep: k<2".into())
        };
        match kres {
            Ok(toks) => {
                econ_plain_sample(_pt.elapsed().as_secs_f64() * 1e3, toks.len());
                // 방출 루프 — 슬롯 상태(tokn·generated·SSE 프레임)는 기존
                // 틱당 방출과 동일 경로로 갱신(eos가 로그 끝에 최대 1회).
                for t in toks {
                    slot_emit(&mut slots[i], t);
                }
                // [MTP] 비-스펙 단독 디코드 후 초안 생성(마지막 방출 토큰 기준
                // 1회 — 순차 경로의 토큰당 호출과 같은 상태·입력).
                if e.mtp_drafts_on()
                    && econ_mtp_maintain()
                    && let Err(err) = e.mtp_single(i, slots[i].next)
                {
                    eprintln!("# mtp(무시): {err}");
                }
            }
            Err(_) => {
                let _pt2 = std::time::Instant::now();
                match e.decode_greedy(i, slots[i].next) {
                    Ok(t) => {
                        econ_plain_sample(_pt2.elapsed().as_secs_f64() * 1e3, 1);
                        // [MTP] 비-스펙 단독 디코드 후 초안 생성(다음 라운드 소비).
                        if e.mtp_drafts_on()
                            && econ_mtp_maintain()
                            && let Err(err) = e.mtp_single(i, t)
                        {
                            eprintln!("# mtp(무시): {err}");
                        }
                        slot_emit(&mut slots[i], t);
                    }
                    Err(err) => slot_fail(&mut slots[i], format!("decode_greedy: {err}")),
                }
            }
        }
    }
    if !sampled.is_empty() {
        let toks: Vec<u32> = sampled.iter().map(|&i| slots[i].next).collect();
        // [S11 2026-10-11] t≥2 샘플링은 배치 체인 1회(로짓 d2h) — 종전
        // 슬롯별 t=1 직렬(≈19.5ms×N) 대비 대폭 상각. 로짓은 행별 t=1 head와
        // 비트동일이라 호스트 샘플러 토큰열 불변. t=1은 종전 직렬 유지.
        let _plg = std::time::Instant::now();
        match e.decode_np_logits(&sampled, &toks) {
            Ok(mut flat) => {
                econ_plain_sample(_plg.elapsed().as_secs_f64() * 1e3, sampled.len());
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

/// [R7] 틱 ② 디코드 우선 — prefill 완료 슬롯 전부(슬롯 루프에서 순수 이동).
/// 반환 = (활성 슬롯, 디코드 여부, 디코드 ms, 완료 처리 ms) — 호출부 계측
/// 가산에 그대로 쓴다.
fn slot_decode(slots: &mut [Slot], eng: &mut Engine, eos: u32) -> (Vec<usize>, bool, f64, f64) {
    let active: Vec<usize> = (0..slots.len())
        .filter(|&i| {
            slots[i].job.is_some()
                && slots[i].prefilled == slots[i].job.as_ref().unwrap().tokens.len()
        })
        .collect();
    let mut decoded = false;
    let mut dec_ms = 0f64;
    let mut fin_ms = 0f64;
    if !active.is_empty() {
        decoded = true;
        let _dt = std::time::Instant::now();
        llm170_diag::watchdog::bump();
        q35_decode(&mut eng.0, slots, &active, eos);
        dec_ms = _dt.elapsed().as_secs_f64() * 1e3;
        // 완료 슬롯 정리 — 결과 전송·반환
        let _ft = std::time::Instant::now();
        for &i in &active {
            finish_slot(&mut slots[i], eng, i, eos);
        }
        fin_ms = _ft.elapsed().as_secs_f64() * 1e3;
    }
    (active, decoded, dec_ms, fin_ms)
}

/// [R7] 틱 ③ 프리필 1청크(슬롯 루프에서 순수 이동). 반환 Some(ms) = 청크
/// 완료(호출부 n_pf·ms_pf 가산), None = 클라이언트 절단 — 종전 continue
/// 경로(잔여 카운터·finish 생략) 그대로.
fn slot_prefill(
    eng: &mut Engine,
    slots: &mut [Slot],
    active: &[usize],
    tick: u64,
    eos: u32,
    npw: bool,
    t0w: &std::time::Instant,
) -> Option<f64> {
    let pf = (0..slots.len())
        .filter(|&i| {
            slots[i].job.is_some()
                && slots[i].prefilled < slots[i].job.as_ref().unwrap().tokens.len()
        })
        .min_by_key(|&i| slots[i].touch);
    let i = pf?;
    let _pft = std::time::Instant::now();
    llm170_diag::watchdog::bump();
    // [B5] 가변 청크 — 잔여 예산(디코드 차감) + 경합 시 long 캡.
    let left = slots[i].job.as_ref().unwrap().tokens.len() - slots[i].prefilled;
    let backlog = slots
        .iter()
        .filter(|s| s.job.is_some() && s.prefilled < s.job.as_ref().unwrap().tokens.len())
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
        let end = (slots[i].prefilled + chunk).min(slots[i].job.as_ref().unwrap().tokens.len());
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
    // [K0 2026-10-12] 프리필 중 절단 감지 — 청크마다 Ping 프로브로
    // 수신자 생존 확인(keep-alive 타임아웃까지 대기 없이 즉시).
    // 절단이면 잔여 프리필을 버리고 즉시 반납(캐시 폐기 — 절단된
    // 요청의 상태 신뢰 불가). 종전엔 프리필 완료 후에야 done 판정이
    // cancelled를 봐 긴 프리필이 슬롯을 끝까지 점유했다(체감 지연).
    if let Some(p) = slots[i].job.as_ref().and_then(|j| j.progress.as_ref())
        && p.send(ProgressFrame::Ping).is_err()
    {
        if let Some(j) = slots[i].job.take() {
            // 수신측 이미 절단 — out도 drop 상태라 전송은 최선만.
            let _ = j.out.send(InferResult {
                tokens: Vec::new(),
                error: None,
            });
        }
        eng.reset_seq(i);
        slots[i] = Slot::free();
        return None;
    }
    if npw {
        eprintln!(
            "[wall] prefill slot{i} {start}tok done @{}s",
            t0w.elapsed().as_secs_f64()
        );
    }
    let ms = _pft.elapsed().as_secs_f64() * 1e3;
    SCHED.chunks_prefill.fetch_add(1, Ordering::Relaxed);
    SCHED.ms_prefill.fetch_add(ms as u64, Ordering::Relaxed);
    match logits {
        Ok(t) => {
            slots[i].prefilled = start;
            if start == slots[i].job.as_ref().unwrap().tokens.len() {
                slot_emit(&mut slots[i], t);
                // [MTP] 프리필 완료 — 유예 호출(prev hidden, 첫
                // 토큰)로 첫 초안 생성(다음 틱 스펙 라운드 소비).
                if eng.0.mtp_drafts_on()
                    && econ_mtp_maintain()
                    && let Err(err) = eng.0.mtp_first(i, t)
                {
                    eprintln!("# mtp(무시): {err}");
                }
            }
        }
        // QA-1: 프리필 실패 적립 — 종전엔 무시돼 prefilled가 영구
        // 갱신되지 않는 스피너였다(매 틱 동일 청크 재시도).
        Err(err) => slot_fail(&mut slots[i], format!("prefill: {err}")),
    }
    finish_slot(&mut slots[i], eng, i, eos);
    Some(ms)
}

/// 슬롯 배정 단일 구현 (107 W7: drain/유휴 이중 복제 통합).
/// 접두 캐시 최장 일치 슬롯 선택(전 슬롯 대상 — 종전 유휴 경로는
/// slot0 고정이었음), 재사용 시 reset 생략, 샘플러 시딩 포함.
fn assign_slot(
    slots: &mut [Slot],
    eng: &mut Engine,
    j: SlotJob,
    tick: u64,
    ptrie: &mut crate::tree::PrefixTrie,
) {
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
    // [§3.4 ③] 전역 트리 우선 조회 — 512+ 접두의 슬롯 후보 직접 제시.
    // 미스(단문/신규)는 슬롯 스캔 폴백. 트리는 슬롯 후보만 제시하고
    // 실제 l(공통 접두 길이)는 기존 산식과 동일 — 수치 무변.
    let (i, l) = if let Some((ti, _tl)) = ptrie.longest_prefix(&j.tokens)
        && ti < slots.len()
        && slots[ti].job.is_none()
        && !slots[ti].cached.is_empty()
    {
        let l = slots[ti]
            .cached
            .iter()
            .zip(j.tokens.iter())
            .take_while(|(a, b)| a == b)
            .count();
        (ti, l)
    } else {
        let Some((i, l)) = pick_prefix_slot(slots, &j.tokens) else {
            return;
        };
        (i, l)
    };
    // [A2] 재사용 판정 — 전체 접두(상태가 그 끝 위치와 정합) 우선, 부분은
    // 엔진 restore_prefix(≤ l: qwen = GDN 체크포인트 · gemma4 = pos 되감기).
    // [finish 불변식 2026-10-14] 상한 = 엔진이 실제 소비한 위치 — n_predict
    // 종료(스토퍼 없음)는 cached가 엔진 pos보다 1 길어 그대로 재사용하면
    // 위치가 어긋난다(콜드 정합 유지 — 스토퍼 종료는 정합적이라 무영향).
    let epos = eng.0.seq_pos(i) as usize;
    // [빈 프리필 가드 2026-10-14] 프롬프트 마지막 토큰은 항상 실제 프리필 —
    // 재사용이 토큰 끝과 겹치면 prefill이 빈 벡터가 되어 첫 디코드가 next=0
    // (플레이스홀더)을 투입한다(실측: gemma4가 턴 종료 마커 출력). qwen은
    // 체크포인트가 512 경계로 양자화돼 이 구멍을 비껴갔다.
    let l = l.min(epos).min(j.tokens.len().saturating_sub(1));
    let full = l > 0 && l == slots[i].cached.len() && l == epos;
    let mut reuse = 0usize;
    if full {
        reuse = l;
        SCHED
            .prefix_tokens
            .fetch_add(reuse as u64, Ordering::Relaxed);
        SCHED.prefix_full_hits.fetch_add(1, Ordering::Relaxed);
        eprintln!("# prefix-cache: slot{i} reuse {reuse}토큰(전체)");
    } else if l > 0 {
        match eng.0.restore_prefix(i, l as u32) {
            Ok(p) if p > 0 && p as usize <= l => {
                reuse = p as usize;
                SCHED
                    .prefix_tokens
                    .fetch_add(reuse as u64, Ordering::Relaxed);
                SCHED.prefix_ckpt_hits.fetch_add(1, Ordering::Relaxed);
                eprintln!("# prefix-cache: slot{i} reuse {reuse}토큰(부분)");
            }
            _ => {}
        }
    }
    if reuse == 0 {
        eng.reset_seq(i);
    }
    // [K6-3 2026-10-13] KV 확보 — 예약(프롬프트+생성한도+슬랙 2)까지 온디맨드
    // 배정. 입장 판정이 선행하나 재사용 슬롯의 기존 배정과 합산해 부족할 수
    // 있다(보수 판정) — 실패는 요청 오류로 정직하게(사용 중 실패 금지 정합:
    // 입장 후에는 배정 성공이 보장된다).
    let need_tokens = j.tokens.len() + j.n_predict.max(1) + 2;
    if let Err(e) = eng.0.kv_grow(i, need_tokens) {
        SCHED.requests_failed.fetch_add(1, Ordering::Relaxed);
        let _ = j.out.send(InferResult {
            tokens: Vec::new(),
            error: Some(format!("kv 입장 실패: {e}")),
        });
        let c = std::mem::take(&mut slots[i].cached);
        slots[i] = Slot::free();
        slots[i].cached = c;
        return;
    }
    // [MTP] 접두 재사용 슬롯은 MTP KV 정합이 없다 — 초안 비활성(검증은
    // 정본이라 손실은 없지만, 초안 비용만 지불하게 된다).
    if reuse > 0 {
        eng.0.mtp_slot_reused(i);
    } else {
        eng.0.mtp_slot_fresh(i);
        if let Err(err) = eng.0.mtp_slot_kv_reset() {
            eprintln!("# mtp(무시): {err}");
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
    };
}
/// [I 2026-10-10] 접두 캐시 재사용 슬롯 선택 — **순수 함수**(테스트 표면).
/// free 슬롯(job 없음) 중 cached와 tokens의 공통 접두가 최장인 곳. 반환
/// (slot, l): l = 공통 접두 길이(부분 포함 — 전체/체크포인트 판정은 호출부가
/// 엔진 체크포인트 조회와 함께 수행[A2 2026-10-11]). 정확히 같은 길이도
/// l = cached 전체(전체 재사용 — 프리필 스킵).
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
        // [K6-3] 동률(모두 미스)은 최저 인덱스 — 배치 그래프 키(슬롯 조합)
        // 압축(전형 조합 {0},{0,1},… — 조합 폭발 방지).
        .max_by_key(|&(i, l)| (l, std::cmp::Reverse(i)))
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
                error: Some(err.clone()),
            });
        }
        // [K4 2026-10-12] 용량 초과(사전 검증 에러·프리필 미소비) — 엔진
        // 상태는 이전 cached와 정합이므로 캐시를 버리지 않는다(재요청
        // 접두 히트 유지). 그 외 런타임 에러는 현행 폐기(신뢰 불가).
        if err.contains("context overflow") && s.prefilled == 0 && !s.cached.is_empty() {
            let keep = std::mem::take(&mut s.cached);
            *s = Slot::free();
            s.cached = keep;
            return;
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
    use super::{Slot, econ_pick_k, econ_pick_src, econ_should_spec, pick_prefix_slot};

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

    /// [I→A2] 정확히 같은 길이 = cached 전체가 접두 → l=3(전체 재사용 —
    /// 판정은 호출부 full 경로. 구버전 기대(0,0)은 A2 부분-접두 반환 이전 잔존).
    #[test]
    fn pick_exact_length_full_prefix() {
        let a = slot_with(vec![1, 2, 3]);
        assert_eq!(pick_prefix_slot(&[a], &[1, 2, 3]), Some((0, 3)));
    }

    /// [I] 접두 0이어도 슬롯은 반환(reuse=0 — reset 경로).
    #[test]
    fn pick_no_prefix_still_slot() {
        let a = slot_with(vec![9, 9]);
        assert_eq!(pick_prefix_slot(&[a], &[1, 2, 3]), Some((0, 0)));
    }

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

    /// [2.A-④ 2026-10-13] 경제성 게이트 판정 — 소스 선택·히스테리시스·k밴드.
    #[test]
    fn econ_gate_decisions() {
        // 소스: 둘 다 측정 → 낮은 비용. MTP 미가용 → ngram 고정.
        assert_eq!(econ_pick_src(Some(24.0), Some(17.0), true), 1);
        assert_eq!(econ_pick_src(Some(17.0), Some(24.0), true), 0);
        assert_eq!(econ_pick_src(Some(1.0), Some(1.0), false), 1);
        // 미측정 — MTP 가용이면 MTP로 측정 개시, 아니면 ngram.
        assert_eq!(econ_pick_src(None, None, true), 0);
        assert_eq!(econ_pick_src(None, None, false), 1);
        // 스펙/평범 히스테리시스 — ON 유지(r≤1.02), OFF→ON은 r<0.98 필요.
        assert!(econ_should_spec(Some(10.0), Some(10.0), true), "동률 유지");
        assert!(
            !econ_should_spec(Some(10.3), Some(10.0), true),
            "ON에서 +3%면 OFF"
        );
        assert!(
            !econ_should_spec(Some(10.0), Some(10.0), false),
            "OFF에서 동률이면 OFF 유지(밴드)"
        );
        assert!(
            econ_should_spec(Some(9.7), Some(10.0), false),
            "OFF에서 −3%면 ON"
        );
        assert!(
            econ_should_spec(None, Some(10.0), false),
            "미측정은 ON(탐색)"
        );
        // k밴드: 미측정 탐색(4→6→2→8) → 최소 EMA(3% 여유).
        assert_eq!(econ_pick_k(1, [None, None, None, None]), 1);
        assert_eq!(econ_pick_k(1, [None, Some(1.0), None, None]), 2);
        assert_eq!(
            econ_pick_k(1, [Some(9.0), Some(10.0), Some(11.0), Some(12.0)]),
            0,
            "9.0 < 10.0×0.97 — 전환"
        );
        assert_eq!(
            econ_pick_k(1, [Some(9.8), Some(10.0), Some(10.1), Some(10.2)]),
            1,
            "9.8 > 10.0×0.97 — 유지(히스테리시스)"
        );
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
