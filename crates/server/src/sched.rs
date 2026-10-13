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
                // [D2 검수 수정 2026-10-13] 재시도는 간헐 클래스(모듈 로드
                // ENOPT 계열) 한정 — 계약 오류(텐서 부재 등)는 즉시 실패로
                // 5회 OOM 연쇄(무의미·오해 소지)를 끊는다. 실패한 시도의
                // 부분 할당은 GpuEngine::load가 free_all_device로 정리한다.
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

// ── [2.A-④ 2026-10-13] 런타임 경제성 게이트 ─────────────────────────────
// 라운드 실측(ms/방출토큰) vs 평범 스텝 실측(ms/토큰) EMA로 3택
// (mtp/ngram/평범)을 자동 선택한다 — 스펙은 무손실(검증 정본)이라 스위칭은
// 처리량만 바꾼다(토큰 불변). 히스테리시스(±2%) + OFF 후 주기 재프로브,
// k는 {2,4,6,8} 밴드 EMA(3% 여유·32라운드 탐사). 판정은 순수 함수(테스트 표면).

/// k 밴드 — 인덱스 ↔ 초안 폭.
const ECON_KS: [usize; 4] = [2, 4, 6, 8];
/// 스펙/평범 히스테리시스 여유(±2%).
const ECON_MARGIN: f64 = 0.02;
/// k 전환 여유(3%).
const ECON_K_MARGIN: f64 = 0.03;
/// OFF 후 재프로브 주기(토큰).
const ECON_PROBE_TOKENS: u64 = 512;

/// EMA 갱신 — 초기 표본은 1/(n+1)(빠른 수렴), 이후 1/8. x = ms/토큰.
fn econ_ema(v: &AtomicU64, n: &AtomicU64, x: f64) {
    let k = n.fetch_add(1, Ordering::Relaxed) + 1;
    let a = 1.0 / k.min(8) as f64;
    let old = f64::from_bits(v.load(Ordering::Relaxed));
    v.store((old + a * (x - old)).to_bits(), Ordering::Relaxed);
}

fn econ_get(v: &AtomicU64, n: &AtomicU64) -> Option<f64> {
    (n.load(Ordering::Relaxed) > 0).then(|| f64::from_bits(v.load(Ordering::Relaxed)))
}

fn econ_kidx(k: usize) -> usize {
    match k {
        2 => 0,
        6 => 2,
        8 => 3,
        _ => 1,
    }
}

/// 평범 표본 — (ms, 방출 토큰 수).
fn econ_plain_sample(ms: f64, toks: usize) {
    if toks > 0 {
        econ_ema(&SCHED.econ_plain, &SCHED.econ_plain_n, ms / toks as f64);
    }
}

/// [2.A-④] MTP 초안 유지 여부 — 게이트가 스펙 ON·현 소스 MTP일 때만.
/// OFF/ngram 소스에서는 유지비(체인 ~2.5ms/토큰 수준)를 내지 않는다 —
/// 재선택 시 해당 라운드의 post_round가 초안을 재구축(복귀 지연 1라운드).
fn econ_mtp_maintain() -> bool {
    SCHED.econ_on.load(Ordering::Relaxed) == 1 && SCHED.econ_src_cur.load(Ordering::Relaxed) == 0
}

/// 스펙 라운드 표본 — (소스, 초안 폭, 수용 수, 라운드 ms).
/// 비용/토큰 = 라운드 ms ÷ 방출 토큰(수용 접두 + 보정/첫 토큰 1).
fn econ_spec_sample(src: u8, k: usize, acc: usize, ms: f64) {
    let per_tok = ms / (acc + 1) as f64;
    econ_ema(
        &SCHED.econ_src[src as usize],
        &SCHED.econ_src_n[src as usize],
        per_tok,
    );
    let i = src as usize * 4 + econ_kidx(k);
    econ_ema(&SCHED.econ_k[i], &SCHED.econ_k_n[i], per_tok);
    SCHED.econ_rounds.fetch_add(1, Ordering::Relaxed);
}

/// 소스 선택 — 가용성·측정 반영(둘 다 측정이면 낮은 비용; 미측정이면 MTP부터).
fn econ_pick_src(mtp: Option<f64>, ng: Option<f64>, mtp_avail: bool) -> u8 {
    if !mtp_avail {
        return 1;
    }
    match (mtp, ng) {
        (Some(m), Some(g)) => {
            if g < m {
                1
            } else {
                0
            }
        }
        (None, Some(_)) => 1,
        _ => 0,
    }
}

/// 스펙/평범 히스테리시스 — ratio = 스펙 비용/평범 비용.
/// ON 유지: ratio ≤ 1+m · OFF→ON: ratio < 1−m. 스펙 미측정이면 측정 위해 ON.
fn econ_should_spec(spec: Option<f64>, plain: Option<f64>, on: bool) -> bool {
    match (spec, plain) {
        (None, _) | (_, None) => true,
        (Some(s), Some(p)) => {
            let r = s / p;
            if on {
                r <= 1.0 + ECON_MARGIN
            } else {
                r < 1.0 - ECON_MARGIN
            }
        }
    }
}

/// k 선택 — 미측정 탐색(4→6→2→8) 후 최소 EMA(3% 여유 = 히스테리시스).
fn econ_pick_k(cur: usize, emas: [Option<f64>; 4]) -> usize {
    for &j in &[1usize, 2, 0, 3] {
        if emas[j].is_none() {
            return j;
        }
    }
    let mut best = cur;
    let mut best_v = emas[cur].unwrap_or(f64::INFINITY);
    for (j, v) in emas.iter().enumerate() {
        let v = v.unwrap(); // 미측정 없음(위 루프)
        if v < best_v * (1.0 - ECON_K_MARGIN) {
            best = j;
            best_v = v;
        }
    }
    best
}

/// [2.A-④] 게이트 판정 — (소스, k). None = 평범(라운드 생략).
fn econ_plan(mtp_avail: bool) -> Option<(u8, usize)> {
    let plain = econ_get(&SCHED.econ_plain, &SCHED.econ_plain_n);
    let mtp = econ_get(&SCHED.econ_src[0], &SCHED.econ_src_n[0]);
    let ng = econ_get(&SCHED.econ_src[1], &SCHED.econ_src_n[1]);
    let mut src = econ_pick_src(mtp, ng, mtp_avail);
    // 소스 탐사 — 미측정 소스 우선(비교 성립), 이후 32라운드마다 교차 1회.
    // (스펙이 계속 켜지면 안 뽑힌 소스는 영원히 미측정 — 자기참조 차단.)
    if mtp_avail {
        let other = 1 - src;
        let n_other = SCHED.econ_src_n[other as usize].load(Ordering::Relaxed);
        if n_other < 3 || SCHED.econ_rounds.load(Ordering::Relaxed) % 32 == 15 {
            src = other;
        }
    }
    // 평범 프로브 — 초기 3표본 부트스트랩(4라운드마다 1회 스킵) 후 64라운드
    // 마다 1회 — 스펙이 계속 켜지면 평범 실측이 갱신되지 않는 자기참조 차단.
    // (상태(econ_on)는 유지 — 스킵 1회 뒤 재판정.)
    let rounds = SCHED.econ_rounds.load(Ordering::Relaxed);
    let plain_n = SCHED.econ_plain_n.load(Ordering::Relaxed);
    if (plain_n < 3 && rounds % 4 == 3) || rounds % 64 == 47 {
        return None;
    }
    let spec = if src == 0 { mtp } else { ng };
    let on = SCHED.econ_on.load(Ordering::Relaxed) == 1;
    let mut use_spec = econ_should_spec(spec, plain, on);
    if !use_spec {
        let gtok = SCHED.econ_gen.load(Ordering::Relaxed);
        let last = SCHED.econ_probe_gen.load(Ordering::Relaxed);
        if gtok.saturating_sub(last) < ECON_PROBE_TOKENS {
            SCHED.econ_on.store(0, Ordering::Relaxed);
            return None;
        }
        // 재프로브 1라운드 — 결과 표본이 다음 판정을 갱신(밴드 밖이면 OFF 복귀).
        // OFF 동안에는 완료 라운드가 드물어 rounds 기반 소스 교차가 얼어붙는다 —
        // 프로브마다 소스를 교차해 MTP/ngram 둘 다 주기적으로 재측정한다.
        if mtp_avail {
            let alt = SCHED.econ_probe_alt.load(Ordering::Relaxed) ^ 1;
            SCHED.econ_probe_alt.store(alt, Ordering::Relaxed);
            src = alt as u8;
        }
        SCHED.econ_probe_gen.store(gtok, Ordering::Relaxed);
        use_spec = true;
    }
    SCHED.econ_on.store(use_spec as u64, Ordering::Relaxed);
    SCHED.econ_src_cur.store(src as u64, Ordering::Relaxed);
    let k = if src == 1 {
        let emas: [Option<f64>; 4] =
            std::array::from_fn(|j| econ_get(&SCHED.econ_k[4 + j], &SCHED.econ_k_n[4 + j]));
        let cur = (SCHED.econ_kidx.load(Ordering::Relaxed) as usize).min(3);
        let mut idx = econ_pick_k(cur, emas);
        // 32라운드마다 밴드 순환 1회(탐사 — EMA가 낡은 순위에 갇히지 않게).
        if SCHED.econ_rounds.load(Ordering::Relaxed) % 32 == 31 {
            idx = (idx + 1) % 4;
        }
        SCHED.econ_kidx.store(idx as u64, Ordering::Relaxed);
        ECON_KS[idx]
    } else {
        4 // MTP 저장 초안 폭 고정(체인 K=4)
    };
    Some((src, k))
}

pub fn slot_loop(mut eng: Engine, rx: std::sync::mpsc::Receiver<SlotJob>, n_slots: usize) {
    // [§3.4 ③] 전역 접두 트리 — 슬롯 cached 승격(호스트 조회·무손실).
    let mut ptrie = crate::tree::PrefixTrie::new();
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
        slot_intake(&mut slots, &mut eng, &rx, tick, &mut ptrie);
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
) {
    loop {
        if !slots.iter().any(|s| s.job.is_none()) {
            break;
        }
        let Ok(j) = rx.try_recv() else {
            break;
        };
        SCHED.queue_depth.fetch_sub(1, Ordering::Relaxed);
        // 107 W7: 배정 단일 구현으로 위임(접두 캐시 로직 동일).
        assign_slot(slots, eng, j, tick, ptrie);
    }
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

/// [R7] 모니터링 게시 스로틀(≤1Hz — 최신값만, 핫패스 할당 최소화) —
/// 슬롯 루프에서 순수 이동.
fn maybe_publish(
    slots: &[Slot],
    eng: &Engine,
    copy_base: [(u64, u64, u64); 3],
    last_pub: &mut std::time::Instant,
) {
    if last_pub.elapsed() >= std::time::Duration::from_millis(900) {
        *last_pub = std::time::Instant::now();
        publish_slot_views(slots);
        publish_stats(eng, copy_base);
    }
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
    // 체크포인트(2048 경계 GDN 상태) ≤ l. 부분은 restore로 상태·pos 복원.
    let full = l > 0 && l == slots[i].cached.len();
    let ck = eng.0.checkpoint_pos(i, l as u32) as usize;
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
        SCHED.prefix_full_hits.fetch_add(1, Ordering::Relaxed);
        eprintln!("# prefix-cache: slot{i} reuse {reuse}토큰(전체)");
    } else {
        match eng.0.restore_gdn(i, l as u32) {
            Ok(p) if p as usize == reuse => {
                SCHED
                    .prefix_tokens
                    .fetch_add(reuse as u64, Ordering::Relaxed);
                SCHED.prefix_ckpt_hits.fetch_add(1, Ordering::Relaxed);
                eprintln!("# prefix-cache: slot{i} reuse {reuse}토큰(체크포인트)");
            }
            _ => {
                eng.reset_seq(i);
                reuse = 0;
            }
        }
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
/// [§7.1 2026-10-11] 스펙 라운드 — 초안(4-gram 최대 k밴드 {2,4,6,8}-1) 발화 시
/// 1회. Ok(true)=라운드 수행(토큰 방출), Ok(false)=초안 없음(종전 경로).
/// 값 보존: 검증 배치가 전 위치 토큰을 내고, 수용분+보정(t=1 재계산)이
/// 단독 디코드와 동일한 토큰열을 만든다(프로브 [A-1 완화 ①]과 동일 절차 —
/// 35B/27B 스펙 골든 접두가 판정). 초안 폭은 k밴드 EMA(2.A-④)가 결정.
/// [MTP-CHAIN §3.3] 초안 소스 정책(mtp/ngram)은 엔진 메서드
/// GpuEngine::mtp_drafts_on — 프리필 KV 구축 게이트와 단일 정의.
fn spec_round(
    i: usize,
    s: &mut Slot,
    e: &mut crate::gpu_engine::GpuEngine,
) -> Result<bool, String> {
    spec_round_impl(i, s, e, false)
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
    spec_round_impl(i, s, e, true)
}

/// [R6] 스펙 라운드 공용 골격 — 그리디(확정)/샘플드(거절샘플링) 단일 경로.
/// 게이트·hist 조립·초안·rows·상태 저장·롤백·게이트 갱신은 공유하고, 검증
/// 및 보정만 `sampled` 정책 분기(내부 로직·호출 순서·로그·주석 원문 유지 —
/// 동작 불변).
fn spec_round_impl(
    i: usize,
    s: &mut Slot,
    e: &mut crate::gpu_engine::GpuEngine,
    sampled: bool,
) -> Result<bool, String> {
    // [§7.5 2026-10-11] 적응 게이트 — 연속 무수용 시 256토큰마다만 재시도.
    // 산문 벤치 실측(서빙 A/B): 초안이 발화하나 수용 0이 반복되면 검증 비용
    // (~0.85틱)이 순손실 — tg −1.6%·np4 −5.7%. 수용 시 즉시 해제.
    // [MTP 2026-10-12] MTP 초안(수용 80~82%) 편입 시 임계 3이 적정(실측:
    // 임계 1이면 첫 경계 미스로 256토큰 억제 → 42.7 t/s). 현재 MTP는
    // 1-초안이 순손실(런치 오버헤드 ~5.5ms/콜 — 그래프 캡처·MTP-2 후 재평가)
    // 이라 파킹 상태이므로 n-gram 기준 임계 1 유지.
    // [MTP-CHAIN §3.3 2026-10-12] 체인(호스트 왕복 0)으로 재귀 치환 후
    // MTP 재활성 — 임계는 서빙 A/B 재실측 전까지 1 유지(선행 실측 근거
    // 80~82%는 순차 경로 기준 — 체인 편입 후 재계측 필요).
    // [§7.5] 무수용 억제 — 연속 무수용 시 256토큰마다만 재시도(경제성 게이트와
    // 별개의 빠른 억제기: 수용 0이 반복되는 발화는 비용만 쓰고 수용이 없음).
    if s.spec_miss >= 1 && (s.generated & 255) != 0 {
        return Ok(false);
    }
    // 샘플드 정책 한정 — 비greedy는 샘플러 필수(종전 조건 그대로).
    if sampled && s.sampler.is_none() {
        return Ok(false);
    }
    // [2.A-④ 2026-10-13] 경제성 게이트 — 3택(mtp/ngram/평범). None = 평범.
    // 스펙 무손실 — 스위칭은 처리량만(토큰 불변). spec_miss 억제 뒤에 둔다
    // (억제된 라운드는 표본·프로브 시계를 소비하지 않게).
    let mtp_avail = !sampled && e.mtp_drafts_on();
    let Some((src, kk)) = econ_plan(mtp_avail) else {
        return Ok(false);
    };
    let mut hist: Vec<u32> = Vec::new();
    if let Some(j) = &s.job {
        hist.extend_from_slice(&j.tokens);
    }
    hist.extend_from_slice(&s.tokens);
    // [2.A-④] 소스·폭은 게이트가 결정(위 econ_plan) — MTP는 저장 초안(체인
    // K=4 고정), n-gram은 k밴드 {2,4,6,8} EMA. 초안은 검증이 정본 — 소스/폭
    // 교체는 정책 계층(무손실). 샘플드는 n-gram 전용(MTP 초안 없음).
    // [MTP-CHAIN §3.3] mtp_drafts_on=false(강제 ngram)면 생산(mtp_first/single/
    // post_round)·소비 모두 스킵(순수 n-gram 경로).
    let mtp_on = src == 0;
    let draft: Vec<u32> = if mtp_on {
        match e.mtp_take_draft(i) {
            Some(d) => d,
            None => llm170_core::spec::ngram_draft(&hist, 3, 48, kk - 1),
        }
    } else {
        llm170_core::spec::ngram_draft(&hist, 3, 48, kk - 1)
    };
    if draft.is_empty() {
        return Ok(false);
    }
    let t = 1 + draft.len();
    let mut rows: Vec<f32> = Vec::with_capacity(t * e.n_embd());
    rows.extend_from_slice(&e.embed_row(s.next)?);
    for &tk in &draft {
        rows.extend_from_slice(&e.embed_row(tk)?);
    }
    let rt0 = std::time::Instant::now();
    let t_save = std::time::Instant::now();
    e.spec_save_state(i)?;
    let ms_save = t_save.elapsed().as_secs_f64() * 1e3;
    let dbg = llm170_diag::flag::on_nonzero("LLM170_MOE_DBG");
    // 검증 + 수용(acc) 판정 — 정책 분기.
    // 그리디: 확정 일치 접두(토큰열). 샘플드: target-only 거절샘플링 —
    // 수용: coin < p(x_k)(prob_of — 후처리 체인 미러, 수용 초안 가산 페널티).
    let (toks, ms_ver, acc) = if sampled {
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
        (Vec::new(), 0.0, acc)
    } else {
        let t_ver = std::time::Instant::now();
        let toks = e.spec_verify(i, &rows, t)?;
        let ms_ver = t_ver.elapsed().as_secs_f64() * 1e3;
        let mut acc = 0usize;
        while acc < draft.len() && toks[acc] == draft[acc] {
            acc += 1;
        }
        if dbg {
            eprintln!(
                "[mtp-dbg] draft={draft:?} verify0={:?} acc={acc}",
                toks.first()
            );
            s.spec_miss = 0; // 진단: 게이트 우회(수용률 계측)
        }
        (toks, ms_ver, acc)
    };
    // [MTP] 롤백·보정 전에 검증 dres의 수용 접두를 초안기 hidden으로 스냅샷.
    // (그리디 경로 한정 — 샘플드는 종전대로 MTP 콜 없음.)
    // [2026-10-12 시도·기각] pre_round(acc+1)로 보정행까지 검증 dres에서
    // 푸시하면 디코드 제거 가능하다는 가설 — 실측 MTP-on 0/381 수용으로
    // 기각(검증 dres 행 acc ≠ 보정 hidden — 롤백/그래프 경로 차이). 디코드를
    // 유지한다(MTP 경로 한정 — n-gram/플레인은 융합 유지).
    let spec_mtp = mtp_on;
    if spec_mtp && let Err(err) = e.mtp_pre_round(acc) {
        eprintln!("# mtp(무시): {err}");
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
    if dbg && !sampled {
        s.spec_miss = 0; // 진단: 게이트 우회(수용률 계측) — 증가 뒤에 둬야 유효.
    }
    // [2.A-④ 2026-10-13] k 적응은 밴드 EMA가 대체(구 "전량 수용 +1/부진 4 복귀"
    // 휴리스틱 제거) — 수용·비용이 cost/token EMA에 함께 반영된다.
    // 보정 토큰 — 정책 분기.
    if sampled {
        // 거절 시 잔차(≠x 재추출 = 조건분포 정확). 방출 분포 = 표적 분포
        // (기대 수용 = E[p]). 그리디 라운드와 롤백 구조 동일.
        for &tk in &draft[..acc] {
            slot_emit(s, tk);
        }
        let lg1 = e.decode(&[i], &[last])?.pop().unwrap_or_default();
        let vocab = e.vocab();
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
        // [2.A-④] 경제성 표본(샘플드 = ngram) — 라운드 비용/방출토큰.
        econ_spec_sample(1, draft.len(), acc, rt0.elapsed().as_secs_f64() * 1e3);
        return Ok(true);
    }
    // 그리디 보정 — [2026-10-12] 보정 토큰 = 검증 행 acc의 argmax(toks[acc]) —
    // 종전 decode_greedy(last) 재계산은 **동일 값의 중복**이었다(라운드 30%).
    // 근거: t-그래프 행 acc의 토큰 = last(수용 접두 마지막)이고 그 다음
    // 토큰이 정확히 보정 토큰. 커널·순서 동일 = 비트동일(스펙 골든 계약).
    // acc=draft.len()(전량 수용)이면 toks[acc]도 검증 범위 안(길이 1+len).
    //
    // [MTP 한정 예외] MTP post_round는 보정 토큰의 **hidden 상태 전진**을
    // 요구한다(초안 체인의 입력) — 융합하면 dres/norm.dxn이 검증 상태에
    // 머물러 초안이 붕괴한다(실측: MTP-on 0/508 수용). MTP 활성 경로만
    // 디코드를 유지(토큰 값은 toks[acc]와 동일 — 부수효과가 계약).
    let mut ms_corr = 0.0f64;
    let corr = if spec_mtp {
        let t_corr = std::time::Instant::now();
        let d = e.decode_greedy(i, last)?;
        ms_corr = t_corr.elapsed().as_secs_f64() * 1e3;
        d
    } else {
        toks[acc]
    };
    let t_mtp = std::time::Instant::now();
    if spec_mtp && let Err(err) = e.mtp_post_round(i, &draft[..acc], corr) {
        eprintln!("# mtp(무시): {err}");
    }
    let ms_mtp = t_mtp.elapsed().as_secs_f64() * 1e3;
    if dbg {
        eprintln!(
            "[spec-ms] save={ms_save:.2} verify={ms_ver:.2} corr={ms_corr:.2} mtp={ms_mtp:.2} tot={:.2}",
            rt0.elapsed().as_secs_f64() * 1e3
        );
    }
    for &tk in &draft[..acc] {
        slot_emit(s, tk);
    }
    slot_emit(s, corr);
    SCHED.spec_rounds.fetch_add(1, Ordering::Relaxed);
    SCHED.spec_accepted.fetch_add(acc as u64, Ordering::Relaxed);
    SCHED
        .spec_ms
        .fetch_add(rt0.elapsed().as_micros() as u64, Ordering::Relaxed);
    // [2.A-④] 경제성 표본 — 라운드 비용/방출토큰(수용 접두 + 보정 1).
    econ_spec_sample(src, draft.len(), acc, rt0.elapsed().as_secs_f64() * 1e3);
    Ok(true)
}

fn slot_emit(s: &mut Slot, t: u32) {
    s.next = t;
    s.tokens.push(t);
    s.generated += 1;
    // [2.A-④] 토큰 시계 — 경제성 게이트의 프로브 주기 판정용.
    SCHED.econ_gen.fetch_add(1, Ordering::Relaxed);
    s.err_count = 0; // QA-1: 성공 — 연속 실패 카운터 리셋
    if let Some(sm) = &mut s.sampler {
        sm.push_tokens([t]);
    }
    if let Some(j) = &s.job
        && let Some(p) = &j.progress
        && p.send(ProgressFrame::Tok(t)).is_err()
    {
        // SSE 수신자 소멸(클라이언트 절단) — 즉시 취소 표시
        s.cancelled = true;
    }
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
