//! 엔진 파사드 — qwen35/qwen4exp 통합, 아키텍처 자동 판별.

use std::path::PathBuf;

pub enum BackendSel {
    Cpu,
    Gpu,
    /// Gpu + 런타임 지정 ("hip"|"vulkan") — serve --gpu-runtime (2026-09-01:
    /// HIP가 폴트로 웨지된 경우 Vulkan 회피).
    GpuRuntime(String),
}

#[derive(Clone)]
pub struct InferRequest {
    pub model: PathBuf,
    pub ctx: usize,
}

/// qwen4exp GPU 경로 요청 여부 (plans/64 P1).
/// GPU = `--backend gpu` 명시 시에만 (기본은 CPU golden 경로).
/// `LLM170_Q4_CPU=1` / `LLM170_RAWHIP=0`이면 항상 CPU.
pub fn q4_gpu_env_off() -> bool {
    if std::env::var_os("LLM170_Q4_CPU").is_some() {
        return true;
    }
    std::env::var("LLM170_RAWHIP").map(|v| v == "0").unwrap_or(false)
}

pub fn q4_gpu_wanted(backend: &BackendSel) -> bool {
    if q4_gpu_env_off() {
        return false;
    }
    match backend {
        BackendSel::Cpu => false,
        BackendSel::Gpu => true,
        BackendSel::GpuRuntime(r) => {
            if r != "hip" && r != "vulkan" {
                eprintln!(
                    "# qwen4exp: --gpu-runtime {r}은 미지원(QSA 커널·용량) — HIP로 진행 (plans/64 §7)"
                );
            }
            true
        }
    }
}

/// qwen4exp의 vulkan 런타임 선택 여부 (plans/84 B — 값경로 VkAcc).
pub fn q4_vk_runtime(backend: &BackendSel) -> bool {
    matches!(backend, BackendSel::GpuRuntime(r) if r == "vulkan")
}

/// qwen4exp GPU 요청 판정 — CLI 문자열판 (infer/bench).
pub fn q4_gpu_wanted_str(backend: &str, runtime: &str) -> bool {
    if q4_gpu_env_off() {
        return false;
    }
    if backend != "gpu" {
        return false;
    }
    if runtime != "hip" && runtime != "vulkan" {
        eprintln!(
            "# qwen4exp: --gpu-runtime {runtime}은 미지원(QSA 커널·용량) — HIP로 진행 (plans/64 §7)"
        );
    }
    true
}

/// CLI 문자열판 vulkan 선택 (plans/84 B).
pub fn q4_vk_runtime_str(runtime: &str) -> bool {
    runtime == "vulkan"
}

pub struct InferResult {
    pub tokens: Vec<u32>,
}

/// serve --spec k 전역 (기본 0).
pub static SPEC_K: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

pub enum Engine {
    Q35(Box<llm170_core::qwen35::Engine>),
    Q4(Box<llm170_core::qwen4exp::layers::Engine4>),
}

/// 슬롯 스케줄러 (04) — llama.cpp 규칙 1:1: 디코드 우선, 잔여 예산만
/// 프리필 청크. 슬롯 = 엔진 시퀀스 id. 요청 종료 → 슬롯 반환(reset_seq).
pub struct SlotJob {
    pub tokens: Vec<u32>,
    pub n_predict: usize,
    /// MTP 스펙 k (0=off) — serve --spec.
    pub spec_k: usize,
    /// 샘플링 파라미터 (기본 greedy — None이면 GPU argmax 경로 유지).
    pub sampler: Option<llm170_core::sampler::SamplerParams>,
    /// 조기 종료 토큰 (EOS + 채팅 템플릿 종결자).
    pub stops: Vec<u32>,
    /// 토큰별 SSE 스트림 채널.
    pub progress: Option<std::sync::mpsc::Sender<u32>>,
    /// 최종 결과 송신.
    pub out: std::sync::mpsc::Sender<InferResult>,
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
    /// 접두 캐시 — 상태가 구워진 전체 토큰열 (요청 간 유지, plans/24).
    cached: Vec<u32>,
    /// 슬롯별 샘플러 (요청에서 생성, 토큰마다 상태 갱신).
    sampler: Option<llm170_core::sampler::Sampler>,
}

impl Slot {
    fn free() -> Self {
        Slot { job: None, prefilled: 0, next: 0, generated: 0, tokens: Vec::new(), touch: 0, cancelled: false, cached: Vec::new(), sampler: None }
    }
}


/// qwen4exp 로드 재시도 — transient ENOENT 회복 (최대 5회×1s).
fn load_q4_retry(p: &std::path::Path) -> llm170_core::qwen4exp::Model4 {
    for i in 0..5 {
        match llm170_core::qwen4exp::Model4::load(p) {
            Ok(m) => return m,
            Err(e) => {
                eprintln!("# qwen4exp 로드 재시도 {}/5: {e}", i + 1);
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
        }
    }
    panic!("qwen4exp 로드 최종 실패: {}", p.display())
}

/// qwen35 로드 재시도 — 동일.
fn load_q35_retry(p: &std::path::Path) -> llm170_core::qwen35::Model {
    for i in 0..5 {
        match llm170_core::qwen35::Model::load(p) {
            Ok(m) => return m,
            Err(e) => {
                eprintln!("# qwen35 로드 재시도 {}/5: {e}", i + 1);
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
        }
    }
    panic!("qwen35 로드 최종 실패: {}", p.display())
}

/// GGUF 오픈 재시도 (최대 5회×1s) — transient ENOENT 회복.
fn open_with_retry(p: &std::path::Path) -> Option<llm170_gguf::GgufFile> {
    for i in 0..5 {
        if let Ok(g) = llm170_gguf::GgufFile::open(p) {
            return Some(g);
        }
        eprintln!("# gguf 오픈 재시도 {}/5: {}", i + 1, p.display());
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
    None
}
/// 슬롯 로짓 → 토큰: 활성 샘플러면 sample, 아니면 greedy (동률 최저 인덱스).
fn pick(s: &mut Slot, logits: &[f32]) -> u32 {
    match &mut s.sampler {
        Some(sm) if !sm.is_greedy() => sm.sample(logits),
        _ => llm170_core::qwen35::greedy(logits),
    }
}

/// Q35 np 디코드 — 샘플링 슬롯 포함시 logits 경로(decode), 아니면 GPU argmax 판.
fn q35_decode(e: &mut llm170_core::qwen35::Engine, slots: &mut [Slot], seqs: &[usize]) {
    let toks: Vec<u32> = seqs.iter().map(|&i| slots[i].next).collect();
    if seqs.iter().any(|&i| slots[i].sampler.as_ref().is_some_and(|s| !s.is_greedy())) {
        match e.decode(seqs, &toks) {
            Ok(rows) => {
                for (row, &i) in seqs.iter().enumerate() {
                    let t = pick(&mut slots[i], &rows[row]);
                    slot_emit(&mut slots[i], t);
                }
            }
            Err(err) => eprintln!("# decode 실패({err}) — 이번 회차 건너뜀"),
        }
    } else {
        match e.decode_np_greedy(seqs, &toks) {
            Ok(toks) => {
                for (row, &i) in seqs.iter().enumerate() {
                    slot_emit(&mut slots[i], toks[row]);
                }
            }
            Err(err) => eprintln!("# np-greedy 실패({err}) — 이번 회차 건너뜀"),
        }
    }
}

/// 연속 배칭 루프 (04-2). 매 반복: ① 큐 drain → LRU 가용 슬롯 배정
/// ② 디코드 우선(활성 전 슬롯 — q35는 1배치 호출, q4는 슬롯별 decode1)
/// ③ 디코드한 스텝이 없으면 프리필 1청크. 완료/EOS → 슬롯 반환(reset_seq).
pub fn slot_loop(
    mut eng: Engine,
    rx: std::sync::mpsc::Receiver<SlotJob>,
    n_slots: usize,
) {
    const EOS: u32 = 248044;
    // 기동 워밍업 — 첫 요청이 지연 초기화(raw_init, ctx 비례 수십 초)를
    // 뒤집어쓰지 않도록 여기서 소진하고 상태를 되돌린다. 준비 전에는 /health가
    // 503이라 클라이언트가 계측을 시작하지 않는다.
    {
        let warm: Vec<u32> = vec![1u32; 16];
        let w: Result<(), String> = match &mut eng {
            Engine::Q35(e) => e
                .prefill(0, &warm)
                .and_then(|l| {
                    let t = llm170_core::qwen35::greedy(&l);
                    e.decode_greedy(0, t).map(|_| ())
                })
                .map_err(|e| e.to_string()),
            Engine::Q4(e) => e
                .prefill(0, &warm)
                .and_then(|_| e.decode1(0, 1u32).map(|_| ()))
                .map_err(|e| e.to_string()),
        };
        if let Err(err) = w {
            eprintln!("# warmup 실패(치명 아님): {err}");
        }
        match &mut eng {
            Engine::Q35(e) => e.reset_states(),
            Engine::Q4(e) => e.reset_states(),
        }
    }
    crate::http::READY.store(true, std::sync::atomic::Ordering::Release);
    let mut slots: Vec<Slot> = (0..n_slots).map(|_| Slot::free()).collect();
    let mut tick: u64 = 0;
    let (mut n_dec, mut n_pf) = (0u64, 0u64);
    let (mut ms_dec, mut ms_pf) = (0f64, 0f64);
    let npw = std::env::var_os("LLM170_WALL_TIME").is_some();
    let t0w = std::time::Instant::now();
    let mut last_wt = std::time::Instant::now();
    loop {
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
            let Ok(j) = rx.try_recv() else { break; };
            // 접두 캐시 — cached 전체가 새 프롬프트의 접두면 이어서 프리필.
            let prefix_ok = std::env::var_os("LLM170_NO_PREFIX").is_none();
            let pick = (0..n_slots)
                .filter(|&i| slots[i].job.is_none())
                .map(|i| {
                    let l = if prefix_ok {
                        slots[i].cached.iter().zip(j.tokens.iter()).take_while(|(a, b)| a == b).count()
                    } else { 0 };
                    let full = l > 0 && l == slots[i].cached.len() && j.tokens.len() > l;
                    (i, if full { l } else { 0 })
                })
                .max_by_key(|&(_, l)| l);
            let (i, reuse) = pick.unwrap_or((0, 0));
            if slots[i].job.is_some() { break; }
            if reuse == 0 {
                eng.reset_seq(i);
            }
            let prev_cached = std::mem::take(&mut slots[i].cached);
            let mut sampler_new = j
                .sampler
                .clone()
                .map(llm170_core::sampler::Sampler::new);
            // 프롬프트 토큰으로 패널티 히스토리 시드 (첫 토큰부터 반영)
            if let Some(sm) = &mut sampler_new
                && !sm.is_greedy() {
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
                cached: prev_cached,
                sampler: sampler_new,
            };
            if std::env::var_os("LLM170_SLOT_DBG").is_some() {
                eprintln!("# slot-dbg: job assigned to slot{i} reuse={reuse}");
            }
            if reuse > 0 {
                // 시퀀스 pos는 이미 cached.len() — prefilled=reuse로 잔여만 프리필.
                eprintln!("# prefix-cache: slot{i} reuse {reuse}토큰");
            }
        }
        tick += 1;
        let _it0 = std::time::Instant::now();

        // ② 디코드 우선 — prefill 완료 슬롯 전부
        let active: Vec<usize> = (0..n_slots)
            .filter(|&i| slots[i].job.is_some() && slots[i].prefilled == slots[i].job.as_ref().unwrap().tokens.len())
            .collect();
        let mut decoded = false;
        let mut dec_ms = 0f64;
        if !active.is_empty() {
            decoded = true;
            let _dt = std::time::Instant::now();
            // 샘플링 활성 슬롯 — logits 경로 필요 (GPU argmax 판은 토큰만 회수).
            // greedy 기본은 종전 최적 경로 유지 (게이트 무변화).
            let sampling = |s: &Slot| s.sampler.as_ref().is_some_and(|sm| !sm.is_greedy());
            match &mut eng {
                Engine::Q35(e) => {
                    // 스펙 슬롯 분리 — spec_step 경로 (plans/21). 샘플링 슬롯은
                    // 스펙 제외(스펙 검증은 greedy 판정 전제) — 일반 디코드로.
                    let spec_slots: Vec<usize> = active.iter().copied()
                        .filter(|&i| slots[i].job.as_ref().is_some_and(|j| j.spec_k > 0) && !sampling(&slots[i]))
                        .collect();
                    if !spec_slots.is_empty() && e.has_mtp() && e.raw_decode.is_some() {
                        // np×spec 병합 (plans/18): 스펙 슬롯 2개 이상이면 한 배치로 검증.
                        // 슬롯별 순차 spec_step은 배치 이득을 전부 잃는다 (2026-09-12 측정:
                        // 서버 np4 spec 12.1 vs 비스펙 22.6 t/s agg).
                        let kmin = spec_slots
                            .iter()
                            .map(|&i| slots[i].job.as_ref().unwrap().spec_k.clamp(1, 8))
                            .min()
                            .unwrap_or(1);
                        let mut done_spec: Vec<usize> = Vec::new();
                        if spec_slots.len() > 1 {
                            let ns: Vec<u32> = spec_slots.iter().map(|&i| slots[i].next).collect();
                            if let Ok(accs) = e.spec_step_multi(&spec_slots, &ns, kmin) {
                                for (row, &i) in spec_slots.iter().enumerate() {
                                    let cap = slots[i].job.as_ref().unwrap().n_predict;
                                    for &t in &accs[row] {
                                        if slots[i].generated as usize >= cap {
                                            break;
                                        }
                                        slot_emit(&mut slots[i], t);
                                        if t == EOS {
                                            break;
                                        }
                                    }
                                }
                                done_spec = spec_slots.clone();
                            }
                        }
                        for &i in spec_slots.iter().filter(|&i| !done_spec.contains(i)) {
                            let k = slots[i].job.as_ref().unwrap().spec_k.clamp(1, 8);
                            let next = slots[i].next;
                            let cap = slots[i].job.as_ref().unwrap().n_predict;
                            if let Ok((acc, _tf)) = e.spec_step(i, next, k) {
                                for &t in &acc {
                                    if slots[i].generated as usize >= cap {
                                        break;
                                    }
                                    slot_emit(&mut slots[i], t);
                                    if t == EOS {
                                        break;
                                    }
                                }
                            }
                        }
                        let plain: Vec<usize> = active.iter().copied()
                            .filter(|&i| !spec_slots.contains(&i)).collect();
                        if !plain.is_empty() {
                            q35_decode(e, &mut slots, &plain);
                        }
                    } else {
                        q35_decode(e, &mut slots, &active);
                    }
                }
                Engine::Q4(e) => {
                    // plans/73(np): 활성 2+ 슬롯은 배치 디코드(무게 스트리밍 공유).
                    // 실패 시 decode_batch 내부가 순차 decode1로 폴백한다.
                    // 샘플링 슬롯 포함시 logits 판으로.
                    if active.iter().any(|&i| sampling(&slots[i])) {
                        let toks: Vec<u32> = active.iter().map(|&i| slots[i].next).collect();
                        match e.decode_batch(&active, &toks) {
                            Ok(rows) => {
                                for (row, &i) in active.iter().enumerate() {
                                    let t = pick(&mut slots[i], &rows[row]);
                                    slot_emit(&mut slots[i], t);
                                }
                            }
                            Err(err) => eprintln!("# batch 실패({err}) — 이번 회차 건너뜀"),
                        }
                    } else if active.len() > 1 {
                        let toks: Vec<u32> = active.iter().map(|&i| slots[i].next).collect();
                        match e.decode_batch_greedy(&active, &toks) {
                            Ok(toks) => {
                                for (row, &i) in active.iter().enumerate() {
                                    slot_emit(&mut slots[i], toks[row]);
                                }
                            }
                            Err(err) => eprintln!("# np-greedy 실패({err}) — 이번 회차 건너뜀"),
                        }
                    } else {
                        for &i in &active {
                            if sampling(&slots[i]) {
                                match e.decode1(i, slots[i].next) {
                                    Ok(l) => {
                                        let t = pick(&mut slots[i], &l);
                                        slot_emit(&mut slots[i], t);
                                    }
                                    Err(err) => eprintln!("# decode1 실패({err}) — 이번 회차 건너뜀"),
                                }
                            } else {
                                match e.decode1_greedy(i, slots[i].next) {
                                    Ok(t) => slot_emit(&mut slots[i], t),
                                    Err(err) => eprintln!("# decode1_greedy 실패({err}) — 이번 회차 건너뜀"),
                                }
                            }
                        }
                    }
                }
            }
            dec_ms = _dt.elapsed().as_secs_f64() * 1e3;
            // 완료 슬롯 정리 — 결과 전송·반환
            for &i in &active {
                finish_slot(&mut slots[i], &mut eng, i, EOS);
            }
        }

        // ③ 프리필 1청크 — 디코드가 없었던 회차이거나, 아직 프리필이 남은 대기 슬롯이
        // 있는 경우. (디코드 우선이지만 활성 슬롯의 디코드가 대기 슬롯의 프리필을
        // 영구히 굶기면 서버가 요청을 직렬화한다 — np4 실측 2026-09-12.)
        let pending_prefill = slots.iter().any(|s| {
            s.job.is_some() && s.prefilled < s.job.as_ref().unwrap().tokens.len()
        });
        if !decoded || pending_prefill {
            let pf = (0..n_slots)
                .filter(|&i| {
                    slots[i].job.is_some()
                        && slots[i].prefilled < slots[i].job.as_ref().unwrap().tokens.len()
                })
                .min_by_key(|&i| slots[i].touch);
            // 배치 프리필(plans/74 np4) — 대기 슬롯 N개의 같은 길이 청크를 한 forward 로
            // 묶어 무게 패스를 공유한다(슬롯별이면 4회 읽던 것). 게이트 기본 꺼짐.
            // 실패하면 아래 슬롯별 경로로 폴백(등가성은 prefill_multi 등가 테스트가 보증).
            if std::env::var_os("LLM170_PREFILL_BATCH").is_some() {
                let pend: Vec<usize> = (0..n_slots)
                    .filter(|&i| {
                        slots[i].job.is_some()
                            && slots[i].prefilled < slots[i].job.as_ref().unwrap().tokens.len()
                    })
                    .collect();
                if pend.len() >= 2 {
                    let per = (512usize / pend.len()).max(16);
                    let parts: Vec<Vec<u32>> = pend
                        .iter()
                        .map(|&i| {
                            let j = slots[i].job.as_ref().unwrap();
                            let end = (slots[i].prefilled + per).min(j.tokens.len());
                            j.tokens[slots[i].prefilled..end].to_vec()
                        })
                        .collect();
                    let uniform = parts.iter().all(|p| p.len() == per) && parts.len() == pend.len();
                    if uniform {
                        let flat: Vec<u32> = parts.iter().flatten().copied().collect();
                        if let Engine::Q4(e) = &mut eng {
                            match e.prefill_multi(&pend, &flat, per) {
                                Ok(toks) => {
                                    for (k, &i) in pend.iter().enumerate() {
                                        slots[i].prefilled += per;
                                        let done = slots[i].job.as_ref().is_some_and(|j| {
                                            slots[i].prefilled == j.tokens.len()
                                        });
                                        if done {
                                            slot_emit(&mut slots[i], toks[k]);
                                        }
                                        finish_slot(&mut slots[i], &mut eng, i, EOS);
                                    }
                                    n_pf += 1;
                                    // 이번 회차 프리필 소비 — 슬롯별 경로로 중복 계상 방지.
                                    continue;
                                }
                                Err(err) => eprintln!("# batch-prefill 실패({err}) — 슬롯별 폴백"),
                            }
                        }
                    }
                }
            }
            if let Some(i) = pf {
                let _pft = std::time::Instant::now();
                let chunk = 512usize;
                // plans/74: Q4(FN)는 prefill_greedy — 청크마다 어휘 152k
                // 로짓 pageable D2H(슬로패스 수십 ms) 대신 GPU argmax 8B 회수.
                // Q35(27B)는 종전 전사 경로(원시 프리필 내부 d2h).
                let (start, logits) = {
                    let end = (slots[i].prefilled + chunk).min(slots[i].job.as_ref().unwrap().tokens.len());
                    let part: Vec<u32> = slots[i].job.as_ref().unwrap().tokens[slots[i].prefilled..end].to_vec();
                    // 샘플링 슬롯은 로짓 판(마지막 청크만 판정에 사용) — Q4도
                    // prefill_greedy 대신 prefill. greedy는 종전 최적 경로.
                    let samp = slots[i].sampler.as_ref().is_some_and(|s| !s.is_greedy());
                    let r: Result<u32, String> = match &mut eng {
                        Engine::Q35(e) => e
                            .prefill(i, &part)
                            .map(|l| {
                                if samp {
                                    pick(&mut slots[i], &l)
                                } else {
                                    llm170_core::qwen35::greedy(&l)
                                }
                            })
                            .map_err(|e| e.to_string()),
                        Engine::Q4(e) => {
                            if samp {
                                e.prefill(i, &part).map(|l| pick(&mut slots[i], &l)).map_err(|e| e.to_string())
                            } else {
                                e.prefill_greedy(i, &part).map_err(|e| e.to_string())
                            }
                        }
                    };
                    (end, r)
                };
                if npw {
                    eprintln!("[wall] prefill slot{i} {start}tok done @{}s", t0w.elapsed().as_secs_f64());
                }
                n_pf += 1;
                ms_pf += _pft.elapsed().as_secs_f64() * 1e3;
                if let Ok(t) = logits {
                    slots[i].prefilled = start;
                    if start == slots[i].job.as_ref().unwrap().tokens.len() {
                        slot_emit(&mut slots[i], t);
                    }
                }
                finish_slot(&mut slots[i], &mut eng, i, EOS);
            }
        }

        if decoded {
            n_dec += 1;
            ms_dec += dec_ms;
            if std::env::var_os("LLM170_SRV_TIME").is_some() && n_dec % 32 == 0 {
                eprintln!(
                    "[srv] steps={} decode avg {:.1}ms | prefill {}x avg {:.1}ms",
                    n_dec, ms_dec / n_dec as f64, n_pf,
                    if n_pf > 0 { ms_pf / n_pf as f64 } else { 0.0 }
                );
            }
        }
        // 유휴 시 차단 수신 — 종료(송신자 전 소멸) 시 루프 탈출
        let busy = slots.iter().any(|s| s.job.is_some());
        if !busy {
            match rx.recv() {
                Ok(j) => {
                    // 접두 재사용 — drain 경로와 동일 규칙 (plans/24).
                    let l = slots[0]
                        .cached
                        .iter()
                        .zip(j.tokens.iter())
                        .take_while(|(a, b)| a == b)
                        .count();
                    let reuse = if std::env::var_os("LLM170_NO_PREFIX").is_none()
                        && l > 0
                        && l == slots[0].cached.len()
                        && j.tokens.len() > l
                    {
                        l
                    } else {
                        0
                    };
                    if reuse == 0 {
                        eng.reset_seq(0);
                    } else {
                        eprintln!("# prefix-cache: slot0 reuse {reuse}토큰");
                    }
                    let prev = std::mem::take(&mut slots[0].cached);
                    let mut sampler_new = j
                        .sampler
                        .clone()
                        .map(llm170_core::sampler::Sampler::new);
                    if let Some(sm) = &mut sampler_new
                        && !sm.is_greedy() {
                            sm.push_tokens(j.tokens.iter().copied());
                        }
                    slots[0] = Slot {
                        job: Some(j),
                        prefilled: reuse,
                        next: 0,
                        generated: 0,
                        tokens: Vec::new(),
                        touch: tick,
                        cancelled: false,
                        cached: prev,
                        sampler: sampler_new,
                    };
                }
                Err(_) => break,
            }
        }
    }
}
fn slot_emit(s: &mut Slot, t: u32) {
    s.next = t;
    s.tokens.push(t);
    s.generated += 1;
    if let Some(sm) = &mut s.sampler {
        sm.push_tokens([t]);
    }
    if let Some(j) = &s.job
        && let Some(p) = &j.progress
            && p.send(t).is_err() {
                // SSE 수신자 소멸(클라이언트 절단) — 즉시 취소 표시
                s.cancelled = true;
            }
}

fn finish_slot(s: &mut Slot, eng: &mut Engine, i: usize, eos: u32) {
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
            let _ = j.out.send(InferResult { tokens: toks.clone() });
            // 접두 캐시 — 상태 유지 (프롬프트+생성 = 구워진 열).
            // 스펙 carried가 남으면 GDN이 뒤처짐 — 트렁크 재실행으로 커밋.
            if let Engine::Q35(e) = eng {
                let _ = e.flush_carried(i);
            }
            let mut full = j.tokens.clone();
            full.extend(toks);
            if std::env::var_os("LLM170_NO_PREFIX").is_none() {
                s.cached = full;
            } else {
                eng.reset_seq(i);
            }
        } else {
            eng.reset_seq(i);
        }
        let c = std::mem::take(&mut s.cached);
        *s = Slot::free();
        s.cached = c;
    }
}

/// n_slots 시퀀스로 엔진 구성 (연속 배칭 — 04).
pub fn build_slots(req: InferRequest, backend: BackendSel, n_slots: usize) -> Engine {
    let arch = open_with_retry(&req.model)
        .and_then(|g| g.arch().map(|s| s.to_string()));
    if arch.as_deref() == Some("qwen4exp") {
        let m = load_q4_retry(&req.model);
        let sources = m.part_sources();
        let mut eng = llm170_core::qwen4exp::layers::Engine4::new(m, n_slots, req.ctx);
        // qwen4exp GPU 경로 (rawhip 값 경로) — plans/64 P1. 기본 CPU(정확성
        // 기준); --backend gpu / --gpu-runtime hip일 때만 상주 가속기를 붙인다.
        let want_gpu = q4_gpu_wanted(&backend);
        if want_gpu && q4_vk_runtime(&backend) {
            // plans/84 B — Vulkan 값경로: VkAcc(MatmulHost). 프레임 미구현 →
            // Engine4는 값 경로로 동작(모든 GEMV를 호스트 스테이징).
            match llm170_backend_gpu::new_q4_acc_vk_with_sources(sources) {
                Ok(acc) => {
                    eng = eng.with_acc(acc);
                    eprintln!("# backend: gpu (qwen4exp Vulkan 값경로 — plans/84 B 슬라이스2)");
                }
                Err(e) => {
                    eprintln!("error: qwen4exp Vulkan 가속기 생성 실패 — {e}");
                    eprintln!("error: --backend cpu로 CPU 기준 경로를 쓸 것 (조용한 폴백 금지)");
                }
            }
            return Engine::Q4(Box::new(eng));
        }
        if want_gpu {
            match llm170_backend_gpu::new_q4_acc_with_sources(sources) {
                Ok(acc) => {
                    eng = eng.with_acc(acc);
                    eprintln!(
                        "# backend: gpu (qwen4exp rawhip — 프리필 프레임(기본)/디코드 프레임)"
                    );
                }
                Err(e) => {
                    eprintln!("error: qwen4exp GPU 가속기 생성 실패 — {e}");
                    eprintln!("error: --backend cpu로 CPU 기준 경로를 쓸 것 (조용한 폴백 금지)");
                }
            }
        }
        Engine::Q4(Box::new(eng))
    } else {
        let m = load_q35_retry(&req.model);
        let mut eng = llm170_core::qwen35::Engine::new(m, n_slots, req.ctx);
        // serve --spec — 스펙 의도일 때만 MTP prefill 훅 활성 (plans/22).
        if SPEC_K.get().copied().unwrap_or(0) > 0 {
            eng.mtp_wanted = true;
        }
        if std::env::var("LLM170_RAWHIP").map(|v| v != "0").unwrap_or(true) {
            // plans/29: serve --gpu-runtime vulkan 실제 반영 (지금까지 무시됨).
            let vulkan = match &backend {
                BackendSel::GpuRuntime(r) => r == "vulkan",
                _ => false,
            };
            if vulkan {
                if std::env::var_os("LLM170_VK_ACC").is_some() {
                    match llm170_backend_gpu::rawvk::gemv::VkAcc::new() {
                        Ok(acc) => {
                            eng = eng.with_acc(std::sync::Arc::new(acc));
                            eprintln!("# backend: gpu (vulkan VkAcc)");
                        }
                        Err(e) => eprintln!("vk-acc: {e} (CPU로 진행)"),
                    }
                } else {
                    llm170_backend_gpu::inject_rawvk(&mut eng)
                        .unwrap_or_else(|e| eprintln!("vk-decoder: {e}"));
                }
            } else {
                llm170_backend_gpu::inject_rawhip(&mut eng).unwrap_or_else(|e| eprintln!("rawhip: {e}"));
            }
        }
        let _ = &backend;
        Engine::Q35(Box::new(eng))
    }
}

impl Engine {
    /// 슬롯 단위 리셋 위임.
    pub fn reset_seq(&mut self, seq: usize) {
        match self {
            Engine::Q35(e) => e.reset_seq(seq),
            Engine::Q4(e) => e.reset_seq(seq),
        }
    }

}

/// 멀티바이트 꼬리를 버퍼에 유지하고 완결 접두만 방출.
/// 매핑은 Tokenizer::load의 인코더 변환과 동일 (Ġ/Ċ/latin1/utf8).
pub struct Detok {
    buf: Vec<u8>,
}

impl Detok {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// 토큰 1개 투입 → 지금까지 완결된 텍스트 방출.
    pub fn push(&mut self, tok: u32) -> String {
        let pb = TOKENIZER.get().map(|t| t.piece_bytes(tok)).unwrap_or_default();
        self.buf.extend_from_slice(&pb);
        let mut v = 0usize;
        let b = &self.buf;
        while v < b.len() {
            let ok2 = v + 1 < b.len() && b[v + 1] & 0xC0 == 0x80;
            let ok3 = v + 2 < b.len() && b[v + 1] & 0xC0 == 0x80 && b[v + 2] & 0xC0 == 0x80;
            let ok4 = v + 3 < b.len() && ok3 && b[v + 3] & 0xC0 == 0x80;
            match b[v] {
                x if x < 0x80 => v += 1,
                0xC0..=0xDF if ok2 => v += 2,
                0xE0..=0xEF if ok3 => v += 3,
                0xF0..=0xF7 if ok4 => v += 4,
                _ => break,
            }
        }
        let out = String::from_utf8_lossy(&b[..v]).into_owned();
        self.buf.drain(..v);
        out
    }
}


/// 글로벌 토크나이저 (serve 시 1회 적재).
pub static TOKENIZER: std::sync::OnceLock<crate::tokenize::Tokenizer> = std::sync::OnceLock::new();

pub fn greedy_encode(text: &str) -> Vec<u32> {
    TOKENIZER.get().map(|t| t.encode(text)).unwrap_or_default()
}
