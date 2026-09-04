//! 엔진 파사드 — qwen35/qwen4exp 통합, 아키텍처 자동 판별.

use std::path::PathBuf;
use std::sync::Arc;

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

pub struct InferResult {
    pub tokens: Vec<u32>,
}

/// serve --spec k 전역 (기본 0).
pub static SPEC_K: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

pub enum Engine {
    Q35(llm170_core::model::Engine),
    Q4(llm170_core::qwen4exp::layers::Engine4),
}

/// 슬롯 스케줄러 (04) — llama.cpp 규칙 1:1: 디코드 우선, 잔여 예산만
/// 프리필 청크. 슬롯 = 엔진 시퀀스 id. 요청 종료 → 슬롯 반환(reset_seq).
pub struct SlotJob {
    pub tokens: Vec<u32>,
    pub n_predict: usize,
    /// MTP 스펙 k (0=off) — serve --spec.
    pub spec_k: usize,
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
}

impl Slot {
    fn free() -> Self {
        Slot { job: None, prefilled: 0, next: 0, generated: 0, tokens: Vec::new(), touch: 0, cancelled: false, cached: Vec::new() }
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
fn load_q35_retry(p: &std::path::Path) -> llm170_core::model::Model {
    for i in 0..5 {
        match llm170_core::model::Model::load(p) {
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

/// 연속 배칭 루프 (04-2). 매 반복: ① 큐 drain → LRU 가용 슬롯 배정
/// ② 디코드 우선(활성 전 슬롯 — q35는 1배치 호출, q4는 슬롯별 decode1)
/// ③ 디코드한 스텝이 없으면 프리필 1청크. 완료/EOS → 슬롯 반환(reset_seq).
pub fn slot_loop(
    mut eng: Engine,
    rx: std::sync::mpsc::Receiver<SlotJob>,
    n_slots: usize,
) {
    const EOS: u32 = 248044;
    let mut slots: Vec<Slot> = (0..n_slots).map(|_| Slot::free()).collect();
    let mut tick: u64 = 0;
    loop {
        // ① 새 작업 drain — 전 슬롯 점유 시 큐에 잔류 (bounded: http측 503)
        while let Ok(j) = rx.try_recv() {
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
            slots[i] = Slot {
                job: Some(j),
                prefilled: reuse,
                next: 0,
                generated: 0,
                tokens: Vec::new(),
                touch: tick,
                cancelled: false,
                cached: prev_cached,
            };
            if reuse > 0 {
                // 시퀀스 pos는 이미 cached.len() — prefilled=reuse로 잔여만 프리필.
                eprintln!("# prefix-cache: slot{i} reuse {reuse}토큰");
            }
        }
        tick += 1;

        // ② 디코드 우선 — prefill 완료 슬롯 전부
        let active: Vec<usize> = (0..n_slots)
            .filter(|&i| slots[i].job.is_some() && slots[i].prefilled == slots[i].job.as_ref().unwrap().tokens.len())
            .collect();
        let mut decoded = false;
        if !active.is_empty() {
            decoded = true;
            match &mut eng {
                Engine::Q35(e) => {
                    // 스펙 슬롯 분리 — spec_step 경로 (plans/21).
                    let spec_slots: Vec<usize> = active.iter().copied()
                        .filter(|&i| slots[i].job.as_ref().is_some_and(|j| j.spec_k > 0))
                        .collect();
                    if !spec_slots.is_empty() && e.has_mtp() && e.raw_decode.is_some() {
                        for &i in &spec_slots {
                            let k = slots[i].job.as_ref().unwrap().spec_k.min(8).max(1);
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
                            let toks: Vec<u32> = plain.iter().map(|&i| slots[i].next).collect();
                            let seqs: Vec<usize> = plain.clone();
                            if let Ok(logits) = e.decode(&seqs, &toks) {
                                for (row, &i) in plain.iter().enumerate() {
                                    slot_step(&mut slots[i], &logits[row]);
                                }
                            }
                        }
                    } else {
                        let toks: Vec<u32> = active.iter().map(|&i| slots[i].next).collect();
                        let seqs: Vec<usize> = active.clone();
                        if let Ok(logits) = e.decode(&seqs, &toks) {
                            for (row, &i) in active.iter().enumerate() {
                                slot_step(&mut slots[i], &logits[row]);
                            }
                        }
                    }
                }
                Engine::Q4(e) => {
                    for &i in &active {
                        if let Ok(logits) = e.decode1(i, slots[i].next) {
                            slot_step(&mut slots[i], &logits);
                        }
                    }
                }
            }
            // 완료 슬롯 정리 — 결과 전송·반환
            for &i in &active {
                finish_slot(&mut slots[i], &mut eng, i, EOS);
            }
        }

        // ③ 디코드 스텝이 없었으면 프리필 1청크 (대기 쇼트가 디코드를 굶기지 않음)
        if !decoded {
            let pf = (0..n_slots)
                .filter(|&i| {
                    slots[i].job.is_some()
                        && slots[i].prefilled < slots[i].job.as_ref().unwrap().tokens.len()
                })
                .min_by_key(|&i| slots[i].touch);
            if let Some(i) = pf {
                let chunk = 512usize;
                let (start, logits) = {
                    let end = (slots[i].prefilled + chunk).min(slots[i].job.as_ref().unwrap().tokens.len());
                    let part: Vec<u32> = slots[i].job.as_ref().unwrap().tokens[slots[i].prefilled..end].to_vec();
                    let r: Result<Vec<f32>, String> = match &mut eng {
                        Engine::Q35(e) => e.prefill(i, &part).map_err(|e| e.to_string()),
                        Engine::Q4(e) => e.prefill(i, &part).map_err(|e| e.to_string()),
                    };
                    (end, r)
                };
                if let Ok(l) = logits {
                    slots[i].prefilled = start;
                    if start == slots[i].job.as_ref().unwrap().tokens.len() {
                        let t = llm170_core::model::greedy(&l);
                        slot_emit(&mut slots[i], t);
                    }
                }
                finish_slot(&mut slots[i], &mut eng, i, EOS);
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
                    slots[0] = Slot {
                        job: Some(j),
                        prefilled: reuse,
                        next: 0,
                        generated: 0,
                        tokens: Vec::new(),
                        touch: tick,
                        cancelled: false,
                        cached: prev,
                    };
                }
                Err(_) => break,
            }
        }
    }
}

/// 슬롯 1스텝 — 샘플·스트림·카운트 (finish는 호출부).
fn slot_step(s: &mut Slot, logits: &[f32]) {
    let t = llm170_core::model::greedy(logits);
    slot_emit(s, t);
}
fn slot_emit(s: &mut Slot, t: u32) {
    s.next = t;
    s.tokens.push(t);
    s.generated += 1;
    if let Some(j) = &s.job {
        if let Some(p) = &j.progress {
            if p.send(t).is_err() {
                // SSE 수신자 소멸(클라이언트 절단) — 즉시 취소 표시
                s.cancelled = true;
            }
        }
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

pub fn build(req: InferRequest, backend: BackendSel) -> Engine {
    let slots = std::env::var("LLM170_SLOTS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1)
        .clamp(1, 16);
    build_slots(req, backend, slots)
}

/// n_slots 시퀀스로 엔진 구성 (연속 배칭 — 04).
pub fn build_slots(req: InferRequest, backend: BackendSel, n_slots: usize) -> Engine {
    let arch = open_with_retry(&req.model)
        .and_then(|g| g.arch().map(|s| s.to_string()));
    if arch.as_deref() == Some("qwen4exp") {
        let m = load_q4_retry(&req.model);
        let mut eng = llm170_core::qwen4exp::layers::Engine4::new(m, n_slots, req.ctx);
        let _ = &backend;
        Engine::Q4(eng)
    } else {
        let m = load_q35_retry(&req.model);
        let mut eng = llm170_core::model::Engine::new(m, n_slots, req.ctx);
        // serve --spec — 스펙 의도일 때만 MTP prefill 훅 활성 (plans/22).
        if SPEC_K.get().copied().unwrap_or(0) > 0 {
            eng.mtp_wanted = true;
        }
        if std::env::var("LLM170_RAWHIP").map(|v| v != "0").unwrap_or(true) {
            crate::inject_rawhip(&mut eng).unwrap_or_else(|e| eprintln!("rawhip: {e}"));
        }
        let _ = &backend;
        Engine::Q35(eng)
    }
}

impl Engine {
    /// 무상태 요청 — 시퀀스 상태 초기화 (mmap·가속기 캐시 유지).
    pub fn reset(&mut self) {
        match self {
            Engine::Q35(e) => e.reset_states(),
            Engine::Q4(e) => e.reset_states(),
        }
    }

    /// 슬롯 단위 리셋 위임.
    pub fn reset_seq(&mut self, seq: usize) {
        match self {
            Engine::Q35(e) => e.reset_seq(seq),
            Engine::Q4(e) => e.reset_seq(seq),
        }
    }

    /// prefill + greedy 디코드 → 생성 토큰 전체 (최대 n_predict, EOS 제외).
    /// HTTP는 무상태 — fresh 요청마다 시퀀스 상태 초기화.
    pub fn run(&mut self, tokens: Vec<u32>, n_predict: usize) -> InferResult {
        let r = self.run_inner(tokens, n_predict);
        let mut toks = r.tokens;
        // EOS(248044) 제외 후 n_predict 캡 — OpenAI 규약(n_predict 반환)
        while toks.last() == Some(&248044) {
            toks.pop();
        }
        toks.truncate(n_predict);
        InferResult { tokens: toks }
    }

    /// run_inner + 토큰별 진행 콜백 — SSE가 생성 즉시 전송 (장문 요청이
    /// 완료까지 굳는 것 방지, 2026-09-01).
    pub fn run_with_progress(
        &mut self,
        tokens: Vec<u32>,
        n_predict: usize,
        mut on_token: impl FnMut(u32),
    ) -> InferResult {
        let r = self.run_inner_progress(tokens, n_predict, &mut on_token);
        let mut toks = r.tokens;
        while toks.last() == Some(&248044) {
            toks.pop();
        }
        toks.truncate(n_predict);
        InferResult { tokens: toks }
    }

    fn run_inner_progress(
        &mut self,
        tokens: Vec<u32>,
        n_predict: usize,
        on_token: &mut dyn FnMut(u32),
    ) -> InferResult {
        match self {
            Engine::Q35(e) => {
                let eos = 248044u32;
                let mut out = Vec::new();
                let l = e.prefill(0, &tokens).expect("prefill");
                let mut next = llm170_core::model::greedy(&l);
                out.push(next);
                on_token(next);
                for _ in 0..n_predict {
                    if next == eos {
                        break;
                    }
                    let logits = e.decode(&[0], &[next]).expect("decode");
                    next = llm170_core::model::greedy(&logits[0]);
                    out.push(next);
                    on_token(next);
                }
                InferResult { tokens: out }
            }
            Engine::Q4(e) => {
                let eos = 248044u32;
                let mut out = Vec::new();
                let l = e.prefill(0, &tokens).expect("prefill");
                let mut next = llm170_core::model::greedy(&l);
                out.push(next);
                on_token(next);
                for _ in 0..n_predict {
                    if next == eos {
                        break;
                    }
                    let logits = e.decode1(0, next).expect("decode");
                    next = llm170_core::model::greedy(&logits);
                    out.push(next);
                    on_token(next);
                }
                InferResult { tokens: out }
            }
        }
    }

    fn run_inner(&mut self, tokens: Vec<u32>, n_predict: usize) -> InferResult {
        match self {
            Engine::Q35(e) => {
                let eos = 248044u32;
                let mut out = Vec::new();
                let l = e.prefill(0, &tokens).expect("prefill");
                let mut next = llm170_core::model::greedy(&l);
                out.push(next);
                for _ in 0..n_predict {
                    if next == eos {
                        break;
                    }
                    let logits = e.decode(&[0], &[next]).expect("decode");
                    next = llm170_core::model::greedy(&logits[0]);
                    out.push(next);
                }
                InferResult { tokens: out }
            }
            Engine::Q4(e) => {
                let eos = 248044u32;
                let mut out = Vec::new();
                let l = e.prefill(0, &tokens).expect("prefill");
                let mut next = llm170_core::model::greedy(&l);
                out.push(next);
                for _ in 0..n_predict {
                    if next == eos {
                        break;
                    }
                    let logits = e.decode1(0, next).expect("decode");
                    next = llm170_core::model::greedy(&logits);
                    out.push(next);
                }
                InferResult { tokens: out }
            }
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

pub fn piece_plain(tok: u32) -> String {
    TOKENIZER
        .get()
        .map(|t| t.piece(tok))
        .unwrap_or_default()
}

pub fn piece_escaped(tok: u32) -> String {
    let s = piece_plain(tok);
    s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n").replace('\r', "\\r").replace('\t', "\\t")
}

/// 글로벌 토크나이저 (serve 시 1회 적재).
pub static TOKENIZER: std::sync::OnceLock<crate::tokenize::Tokenizer> = std::sync::OnceLock::new();

pub fn greedy_encode(text: &str) -> Vec<u32> {
    TOKENIZER.get().map(|t| t.encode(text)).unwrap_or_default()
}
