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

pub fn build(req: InferRequest, backend: BackendSel) -> Engine {
    // 간헐적 파일시스템 ENOENT(디렉터리 목록엔 보이는 transient 결함,
    // 2026-09-01 실측) — 재시도로 회복. mmap 대상 전 파트에 적용.
    let arch = open_with_retry(&req.model)
        .and_then(|g| g.arch().map(|s| s.to_string()));
    if arch.as_deref() == Some("qwen4exp") {
        let m = load_q4_retry(&req.model);
        let mut eng = llm170_core::qwen4exp::layers::Engine4::new(m, 1, req.ctx);
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
        let mut eng = llm170_core::model::Engine::new(m, 1, req.ctx);
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
