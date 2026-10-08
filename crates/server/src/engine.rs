//! 엔진 파사드 — qwen35/qwen4exp 통합, 아키텍처 자동 판별.

use std::path::PathBuf;

pub enum BackendSel {
    Cpu,
    /// 후속(W2/W3): CUDA W4A16 가속 부착 경로에서 사용 예정 — 현 프런트
    /// (serve/infer)는 Cpu 단일이라 아직 생성되지 않는다.
    #[allow(dead_code)]
    Gpu,
}
/// 모델 경로 포맷 판정 — 2026-10-08 단일 트랙:
/// 수용은 **W4A16 디렉터리 단일** — 그 외는 명시 에러로 안내.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ModelFormat {
    /// compressed-tensors(weight_packed 3조)·auto-gptq(qweight/qzeros) 패킹.
    W4A16,
}

/// 포맷 스니핑 — 디렉터리 내 safetensors 내용 기반(index.json 우선, 없으면
/// 첫 샤드 헤더 접두). 그 외 파일·디렉터리는 탈락 에러.
pub fn sniff_format(path: &std::path::Path) -> Result<ModelFormat, String> {
    if path.is_file() {
        return Err(format!(
            "파일 모델 미지원(2026-10-08): W4A16 디렉터리만 지원 — {}",
            path.display()
        ));
    }
    if !path.is_dir() {
        return Err(format!("모델 경로 없음: {}", path.display()));
    }
    let index = path.join("model.safetensors.index.json");
    let hay: Option<String> = if index.is_file() {
        std::fs::read_to_string(&index).ok()
    } else {
        // index 없음: 첫 샤드 헤더(8바이트 길이 + JSON) 접두 판독.
        let shard = std::fs::read_dir(path).ok().and_then(|rd| {
            rd.flatten().map(|e| e.path()).find(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("model") && n.ends_with(".safetensors"))
            })
        });
        shard.and_then(|p| {
            use std::io::Read;
            let mut f = std::fs::File::open(p).ok()?;
            let mut lenb = [0u8; 8];
            f.read_exact(&mut lenb).ok()?;
            let hlen = u64::from_le_bytes(lenb).min(1 << 20) as usize; // 접두 1MB
            let mut buf = vec![0u8; hlen];
            f.read_exact(&mut buf).ok()?;
            String::from_utf8_lossy(&buf).into_owned().into()
        })
    };
    let Some(hay) = hay else {
        // safetensors 없는 디렉터리 — config.json(architectures) 유무로 안내.
        if path.join("config.json").is_file() {
            return Err(format!(
                "미지원 포맷: HF config 배포(architectures) — {} (지원: W4A16 디렉터리)",
                path.display()
            ));
        }
        return Err(format!(
            "모델 디렉터리 인식 불가(model*.safetensors/index.json 없음): {}",
            path.display()
        ));
    };
    if hay.contains(".trellis") {
        return Err(format!(
            "미지원 quant 스키마(.trellis — 2026-10-08): W4A16 디렉터리만 지원 — {}",
            path.display()
        ));
    }
    if hay.contains("weight_packed") || hay.contains("qweight") {
        return Ok(ModelFormat::W4A16);
    }
    if path.join("config.json").is_file() {
        return Err(format!(
            "미지원 포맷: HF config 배포(architectures) — {} (지원: W4A16 디렉터리)",
            path.display()
        ));
    }
    Err(format!(
        "모델 디렉터리 포맷 인식 불가: {} (지원: W4A16 디렉터리)",
        path.display()
    ))
}

#[derive(Clone)]
pub struct InferRequest {
    pub model: PathBuf,
    pub ctx: usize,
}

/// 생성 토큰 싱크 — 명령별 출력(JSONL text 포함/미포함·텍스트 누적) 차이를
/// 흡수한다. eng는 읽기 전용 재차용: 디코드 mutable 차용이 끝난 시점에만
/// 호출된다.
pub trait TokenSink {
    fn on_token(&mut self, s: usize, pos: u32, t: u32, eng: &llm170_core::qwen35::Engine);
}

/// qwen35 greedy 생성 상태 — 호출부가 prefill 결과로 시딩한다.
pub struct GenState {
    pub finished: Vec<bool>,
    pub gen_toks: Vec<Vec<u32>>,
    pub next: Vec<u32>,
    /// 시퀀스별 절대 위치(프롬프트 길이 기준) — 토큰마다 +1.
    pub pos: Vec<u32>,
}

/// qwen35 생성 루프 — 배치 디코드(greedy).
pub fn generate_q35(
    eng: &mut llm170_core::qwen35::Engine,
    st: &mut GenState,
    n_predict: usize,
    eos: u32,
    sink: &mut dyn TokenSink,
) -> Result<(), String> {
    let n = st.next.len();
    for _step in 0..n_predict {
        let active: Vec<usize> = (0..n).filter(|&s| !st.finished[s]).collect();
        if active.is_empty() {
            break;
        }
        let toks: Vec<u32> = active.iter().map(|&s| st.next[s]).collect();
        let logits = eng.decode(&active, &toks).map_err(|e| e.to_string())?;
        for (i, &s) in active.iter().enumerate() {
            let t = llm170_core::qwen35::greedy(&logits[i]);
            st.next[s] = t;
            st.pos[s] += 1;
            sink.on_token(s, st.pos[s], t, eng);
            st.gen_toks[s].push(t);
            if t == eos {
                st.finished[s] = true;
            }
        }
    }
    Ok(())
}

pub struct InferResult {
    pub tokens: Vec<u32>,
    /// QA-1: 엔진 확정 실패 사유 — None이면 정상 종료. 종전엔 에러 필드가
    /// 없어 실패 통보 경로 자체가 없었다(슬롯 스피너 + 클라이언트 영구 대기).
    pub error: Option<String>,
}

pub enum Engine {
    Q35(Box<llm170_core::qwen35::Engine>),
}

impl Engine {
    /// 정지 토큰(F5 — 하드코드 248044 일반화): Q35는 아키텍처 상수.
    pub fn eos(&self) -> u32 {
        match self {
            Engine::Q35(_) => llm170_core::qwen35::EOS_EOT,
        }
    }
}

/// P0-4(§10-2): 기동 배너 1줄 고정 — 스왑 시 "무엇으로 도는지"를 로그만으로
/// 판정한다(B1·B21의 라벨 문제를 계약으로 흡수). offload/attach는 각 엔진
/// 조립 지점의 사실.
fn banner(
    model: &std::path::Path,
    fmt: &str,
    runtime: &str,
    offload: &str,
    attach: &str,
    ctx: usize,
    slots: usize,
) {
    let name = model
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_else(|| model.to_str().unwrap_or("?"));
    eprintln!(
        "# boot: model={name} format={fmt} runtime={runtime} offload={offload} ctx={ctx} slots={slots} attach={attach}"
    );
}

pub fn build_slots(req: InferRequest, _backend: BackendSel, n_slots: usize) -> Engine {
    // W4A16 = qwen35 CPU 경로 단일(가속은 W2 커널 이후 — 그때 attach 재도입).
    let m = load_q35_retry(&req.model);
    let eng = llm170_core::qwen35::Engine::new(m, n_slots, req.ctx);
    banner(
        &req.model,
        "w4a16",
        "cpu",
        "none",
        "off(cpu-backend)",
        req.ctx,
        n_slots,
    );
    Engine::Q35(Box::new(eng))
}

impl Engine {
    /// 슬롯 단위 리셋 위임.
    pub fn reset_seq(&mut self, seq: usize) {
        match self {
            Engine::Q35(e) => e.reset_seq(seq),
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
        let pb = TOKENIZER
            .get()
            .map(|t| t.piece_bytes(tok))
            .unwrap_or_default();
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

use crate::sched::load_q35_retry;
/// 글로벌 토크나이저 (serve 시 1회 적재).
pub use crate::sched::{SlotJob, slot_loop};

pub static TOKENIZER: std::sync::OnceLock<crate::tokenize::Tokenizer> = std::sync::OnceLock::new();

pub fn greedy_encode(text: &str) -> Vec<u32> {
    TOKENIZER.get().map(|t| t.encode(text)).unwrap_or_default()
}
// 마커 eh2
// 마커 eh3
// 마커 eh4
// 마커 eh5
