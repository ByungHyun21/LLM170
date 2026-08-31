//! 엔진 파사드 — qwen35/qwen4exp 통합, 아키텍처 자동 판별.

use std::path::PathBuf;
use std::sync::Arc;

pub enum BackendSel {
    Cpu,
    Gpu,
}

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

pub fn build(req: InferRequest, backend: BackendSel) -> Engine {
    let arch = llm170_gguf::GgufFile::open(&req.model)
        .ok()
        .and_then(|g| g.arch().map(|s| s.to_string()));
    if arch.as_deref() == Some("qwen4exp") {
        let m = llm170_core::qwen4exp::Model4::load(&req.model).expect("qwen4exp 로드");
        let mut eng = llm170_core::qwen4exp::layers::Engine4::new(m, 1, req.ctx);
        if let BackendSel::Gpu = backend {
            if let Ok(acc) = llm170_backend_gpu::GpuMatmul::new_hip() {
                eng = eng.with_acc(Arc::new(acc));
            }
        }
        Engine::Q4(eng)
    } else {
        let m = llm170_core::model::Model::load(&req.model).expect("qwen35 로드");
        let mut eng = llm170_core::model::Engine::new(m, 1, req.ctx);
        if let BackendSel::Gpu = backend {
            if let Ok(acc) = llm170_backend_gpu::GpuMatmul::new_hip() {
                eng = eng.with_acc(Arc::new(acc));
            }
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
