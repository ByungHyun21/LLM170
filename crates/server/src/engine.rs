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

pub enum Engine {
    Q35(llm170_core::model::Engine),
    Q4(llm170_core::qwen4exp::layers::Engine4),
}

/// 슬롯 스케줄러 (04) — llama.cpp 규칙 1:1: 디코드 우선, 잔여 예산만
/// 프리필 청크. 슬롯 = 엔진 시퀀스 id. 요청 종료 → 슬롯 반환(reset_seq).
pub struct SlotJob {
    pub tokens: Vec<u32>,
    pub n_predict: usize,
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
}

impl Slot {
    fn free() -> Self {
        Slot { job: None, prefilled: 0, next: 0, generated: 0, tokens: Vec::new(), touch: 0, cancelled: false }
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
            let free = (0..n_slots)
                .filter(|&i| slots[i].job.is_none())
                .min_by_key(|&i| slots[i].touch);
            let Some(i) = free else { break };
            eng.reset_seq(i);
            slots[i] = Slot {
                job: Some(j),
                prefilled: 0,
                next: 0,
                generated: 0,
                tokens: Vec::new(),
                touch: tick,
                cancelled: false,
            };
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
                    let toks: Vec<u32> = active.iter().map(|&i| slots[i].next).collect();
                    let seqs: Vec<usize> = active.clone();
                    if let Ok(logits) = e.decode(&seqs, &toks) {
                        for (row, &i) in active.iter().enumerate() {
                            slot_step(&mut slots[i], &logits[row]);
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
                    eng.reset_seq(0);
                    slots[0] = Slot {
                        job: Some(j),
                        prefilled: 0,
                        next: 0,
                        generated: 0,
                        tokens: Vec::new(),
                        touch: tick,
                        cancelled: false,
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

/// 완료 조건 검사 — EOS/예산 소진 → 결과 전송·슬롯 반환.
fn finish_slot(s: &mut Slot, eng: &mut Engine, i: usize, eos: u32) {
    let done = s.job.as_ref().is_some_and(|j| {
        s.prefilled == j.tokens.len()
            && (s.cancelled || s.next == eos || s.generated as usize >= j.n_predict.max(1))
    });
    if done {
        if let Some(j) = s.job.take() {
            let mut toks = s.tokens.clone();
            while toks.last() == Some(&eos) {
                toks.pop();
            }
            toks.truncate(j.n_predict);
            let _ = j.out.send(InferResult { tokens: toks });
        }
        eng.reset_seq(i);
        *s = Slot::free();
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
        match &backend {
            BackendSel::GpuRuntime(rt) => {
                let acc: Result<std::sync::Arc<dyn llm170_core::matmul::Accelerator>, String> =
                    if rt == "vulkan" {
                        llm170_backend_gpu::GpuMatmul::new_vulkan()
                            .map(|g| Arc::new(g) as std::sync::Arc<dyn llm170_core::matmul::Accelerator>)
                    } else {
                        llm170_backend_gpu::GpuMatmul::new_hip()
                            .map(|g| Arc::new(g) as std::sync::Arc<dyn llm170_core::matmul::Accelerator>)
                    };
                if let Ok(acc) = acc {
                    eng = eng.with_acc(acc);
                }
            }
            BackendSel::Gpu => {
                if let Ok(acc) = llm170_backend_gpu::GpuMatmul::new_hip() {
                    eng = eng.with_acc(Arc::new(acc));
                }
            }
            BackendSel::Cpu => {}
        }
        Engine::Q4(eng)
    } else {
        let m = load_q35_retry(&req.model);
        let mut eng = llm170_core::model::Engine::new(m, n_slots, req.ctx);
        if std::env::var("LLM170_RAWHIP").map(|v| v != "0").unwrap_or(true) {
            crate::inject_rawhip(&mut eng).unwrap_or_else(|e| eprintln!("rawhip: {e}"));
        }
        match &backend {
            BackendSel::GpuRuntime(rt) => {
                let acc: Result<std::sync::Arc<dyn llm170_core::matmul::Accelerator>, String> =
                    if rt == "vulkan" {
                        llm170_backend_gpu::GpuMatmul::new_vulkan()
                            .map(|g| Arc::new(g) as std::sync::Arc<dyn llm170_core::matmul::Accelerator>)
                    } else {
                        llm170_backend_gpu::GpuMatmul::new_hip()
                            .map(|g| Arc::new(g) as std::sync::Arc<dyn llm170_core::matmul::Accelerator>)
                    };
                if let Ok(acc) = acc {
                    eng = eng.with_acc(acc);
                }
            }
            BackendSel::Gpu => {
                if let Ok(acc) = llm170_backend_gpu::GpuMatmul::new_hip() {
                    eng = eng.with_acc(Arc::new(acc));
                }
            }
            BackendSel::Cpu => {}
        }
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
