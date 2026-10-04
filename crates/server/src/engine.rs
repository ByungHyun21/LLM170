//! 엔진 파사드 — qwen35/qwen4exp 통합, 아키텍처 자동 판별.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

pub enum BackendSel {
    Cpu,
    Gpu,
    /// Gpu + 런타임 지정 ("hip"|"vulkan") — serve --gpu-runtime (2026-09-01:
    /// HIP가 폴트로 웨지된 경우 Vulkan 회피).
    GpuRuntime(String),
    /// EXL3 직접 경로 (plans/121 A1) — --model은 EXL3 디렉터리,
    /// --backend exl3로 지정. vk 배치 프리필+순차 디코드.
    Exl3,
    /// EXL3 hip 백엔드(plans/121 exl3-sched) — 단일 슬롯, 배치 프리필+순차 디코드.
    Exl3Hip,
}

#[derive(Clone)]
pub struct InferRequest {
    pub model: PathBuf,
    pub ctx: usize,
    /// 외장 MTP 모듈 경로(plans/109 P15⑤) — None이면 --spec>0 시 자동 탐지.
    pub mtp: Option<PathBuf>,
    /// PLE 테이블 오프로드 모드(plans/111 W4c) — None=auto.
    pub ple_table: Option<String>,
    /// SSD 블록 캐시 예산 MiB(plans/111 W4c) — None=기본 1024.
    pub ple_cache_mib: Option<usize>,
}

/// qwen4exp GPU 경로 요청 여부 (plans/64 P1).
/// GPU = `--backend gpu` 명시 시에만 (기본은 CPU golden 경로).
/// `LLM170_Q4_CPU=1` / `LLM170_RAWHIP=0`이면 항상 CPU.
pub fn q4_gpu_env_off() -> bool {
    if llm170_diag::flag::on("LLM170_Q4_CPU") {
        return true;
    }
    std::env::var("LLM170_RAWHIP")
        .map(|v| v == "0")
        .unwrap_or(false)
}

pub fn q4_gpu_wanted(backend: &BackendSel) -> bool {
    if q4_gpu_env_off() {
        return false;
    }
    match backend {
        BackendSel::Cpu | BackendSel::Exl3 | BackendSel::Exl3Hip => false,
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

/// q4 모델에 외장 MTP 모듈 병합 (plans/109 P15⑤) — `--mtp` 우선, 없으면
/// spec 의도(spec_k>0)일 때 모델 형제의 `mtp-*.gguf` 자동 탐지(Q8_0 우선).
/// 성공/생략은 로그로만 — 실패(명시 지정인데 깨짐)는 Err.
pub fn apply_mtp(
    m: &mut llm170_core::qwen4exp::Model4,
    model_path: &std::path::Path,
    mtp_arg: Option<&std::path::Path>,
    spec_k: usize,
) -> Result<(), String> {
    let pick: Option<std::path::PathBuf> = match mtp_arg {
        Some(p) => Some(p.to_path_buf()),
        None if spec_k > 0 => {
            let dir = model_path.parent().unwrap_or(std::path::Path::new("."));
            let mut hits: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
                .into_iter()
                .flatten()
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.starts_with("mtp-") && n.ends_with(".gguf"))
                })
                .collect();
            hits.sort(); // Q4_K_M < Q8_0 — Q8 우선은 아래에서.
            hits.sort_by_key(|p| {
                // Q8_0 우선(드래프트 품질) — ⑥ 측정 전 임시 기본.
                !p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.contains("Q8"))
            });
            hits.into_iter().next()
        }
        None => None,
    };
    match pick {
        Some(p) => {
            m.load_mtp(&p).map_err(|e| e.to_string())?;
            eprintln!("# mtp: 외장 nextn 모듈 병합 — {}", p.display());
            Ok(())
        }
        None if spec_k > 0 => {
            eprintln!(
                "# mtp: --spec {} 지정이나 mtp-*.gguf 미발견 — 스펙 없이 진행",
                spec_k
            );
            Ok(())
        }
        None => Ok(()),
    }
}

/// CLI 문자열판 vulkan 선택 (plans/84 B).
pub fn q4_vk_runtime_str(runtime: &str) -> bool {
    runtime == "vulkan"
}

/// 백엔드 부착 실패 정책 — serve·vl은 경고 후 CPU 지속, bench·infer 검증은
/// 오류 승격(조용한 CPU 폴백이 GPU 수치로 오인된 사고 이력 — 커밋 참조).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AttachPolicy {
    Warn,
    Strict,
}

/// qwen35 GPU 부착 — 단일 경로 (plans/109 P3). 종전 serve/infer/bench/vl이
/// 각자 베껴 쓰며 serve의 vk-q35 비결정 게이트를 우회했다.
/// 1. LLM170_RAWHIP=0 → 부착 없음(CPU).
/// 2. vulkan && !LLM170_VK_Q35_FORCE → hip 폴백(원장 87/90 비결정 레이스).
/// 3. vulkan 잔여 → LLM170_VK_ACC=1이면 VkAcc, 아니면 VkDecoder.
/// 4. 그 외 → rawhip 디코더. 실패는 정책(Warn=CPU 지속 / Strict=Err)대로.
pub fn attach_q35(
    mut eng: llm170_core::qwen35::Engine,
    vulkan: bool,
    policy: AttachPolicy,
) -> Result<llm170_core::qwen35::Engine, String> {
    // LLM170_RAWHIP=0 → 명시적 CPU.
    if !llm170_diag::flag::ne0("LLM170_RAWHIP") {
        return Ok(eng);
    }
    // 107(원장 87·90): qwen35 vk 디코드 프리필 비결정 레이스(확산형). 조용한
    // 오염 대신 가시 폴백 — LLM170_VK_Q35_FORCE=1로 vk 진단 강행.
    let vk_q35_blocked = vulkan && !llm170_diag::flag::on("LLM170_VK_Q35_FORCE");
    if vk_q35_blocked {
        eprintln!(
            "error: vulkan qwen35 decode is nondeterministic (ledger 87/90) — falling back to hip"
        );
    }
    if vulkan && !vk_q35_blocked {
        if llm170_diag::flag::on("LLM170_VK_ACC") {
            match llm170_backend_gpu::rawvk::vkacc::VkAcc::new() {
                Ok(acc) => {
                    eprintln!("# backend: gpu (vulkan VkAcc)");
                    return Ok(eng.with_acc(std::sync::Arc::new(acc)));
                }
                Err(e) => eprintln!("vk-acc: {e} (CPU로 진행)"),
            }
        } else if let Err(e) = llm170_backend_gpu::inject_rawvk(&mut eng) {
            eprintln!("vk-decoder: {e}");
        } else {
            eprintln!("# backend: gpu (vulkan VkDecoder)");
            return Ok(eng);
        }
        return Ok(eng);
    }
    match llm170_backend_gpu::inject_rawhip(&mut eng) {
        Ok(()) => {
            eprintln!("# backend: gpu (qwen35 rawhip decode)");
            Ok(eng)
        }
        Err(e) => {
            eprintln!("rawhip: {e}");
            match policy {
                AttachPolicy::Warn => Ok(eng),
                AttachPolicy::Strict => Err(e),
            }
        }
    }
}

/// qwen4exp GPU 부착 — 단일 경로 (plans/109 P3). vk·hip 가속기 실패는 정책대로.
///
/// res_hc f16 버스(res_f16)는 호출부가 명시한다 — 현재 원장 상태:
/// - serve(build_slots): `want_gpu && !vk` (원장 105 — hip f16 버스 승격,
///   serve 슬롯 경로에서 +4.3% 웜·토큰 불변 실측).
/// - infer/bench: `false` — charhash 스테이지 해시·FN 토큰 골든이 f32 버스로
///   캡처됐다(2026-09-29 실측: infer에서 f16 설정 시 골든 발산·토큰 열화).
///   통일은 산술 클래스 변경(규칙 10: 승인+재캡처 필요) — 별도 승인 전까지
///   호출부 현행 값을 유지한다(B1 잔여, 의도된 발산으로 문서화).
pub fn attach_q4(
    eng: llm170_core::qwen4exp::layers::Engine4,
    sources: Vec<(usize, usize, PathBuf)>,
    want_gpu: bool,
    vk: bool,
    res_f16: bool,
    policy: AttachPolicy,
) -> Result<llm170_core::qwen4exp::layers::Engine4, String> {
    llm170_core::qwen4exp::frame::set_backend_res_f16(res_f16);
    if !want_gpu {
        return Ok(eng);
    }
    if vk {
        // plans/84 B — Vulkan 값경로: VkAcc(MatmulHost). 프레임 미구현 →
        // Engine4는 값 경로로 동작(모든 GEMV를 호스트 스테이징).
        return match llm170_backend_gpu::new_q4_acc_vk_with_sources(sources) {
            Ok(acc) => {
                eprintln!("# backend: gpu (qwen4exp Vulkan 값경로 — plans/84 B)");
                Ok(eng.with_acc(acc))
            }
            Err(e) => {
                eprintln!("error: qwen4exp Vulkan 가속기 생성 실패 — {e}");
                match policy {
                    AttachPolicy::Warn => Ok(eng),
                    AttachPolicy::Strict => Err(e),
                }
            }
        };
    }
    match llm170_backend_gpu::new_q4_acc_with_sources(sources) {
        Ok(acc) => {
            eprintln!("# backend: gpu (qwen4exp rawhip)");
            Ok(eng.with_acc(acc))
        }
        Err(e) => {
            eprintln!("error: qwen4exp GPU 가속기 생성 실패 — {e}");
            eprintln!("error: --backend cpu로 CPU 기준 경로를 쓸 것 (조용한 폴백 금지)");
            match policy {
                AttachPolicy::Warn => Ok(eng),
                AttachPolicy::Strict => Err(e),
            }
        }
    }
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

/// 스펙 통계 — 요약 eprintln은 호출부 담당.
#[derive(Default)]
pub struct SpecStats {
    pub cycles: usize,
    pub accepted: usize,
    pub target_forwards: usize,
}

/// qwen35 생성 루프 단일 구현 (plans/109 P4) — 종전 infer/vl이 3모드
/// (spec-multi / spec-single / batch)를 각자 손베껴 썼다. 모드 선택:
/// spec_k>0 && has_mtp && LLM170_SPEC_GPU → n>1: "spec-multi", n==1:
/// "spec", 아니면 "batch". --spec 무시 안내(eos·MTP 부재)도 여기서.
pub fn generate_q35(
    eng: &mut llm170_core::qwen35::Engine,
    st: &mut GenState,
    n_predict: usize,
    spec_k: usize,
    eos: u32,
    sink: &mut dyn TokenSink,
) -> Result<(&'static str, SpecStats), String> {
    let n = st.next.len();
    let mut stats = SpecStats::default();
    let spec_on = spec_k > 0 && eng.has_mtp() && std::env::var_os("LLM170_SPEC_GPU").is_some();
    if spec_k > 0 && !eng.has_mtp() {
        eprintln!("# --spec 무시: MTP(nextn) 텐서 없음");
    }
    if spec_on && n > 1 {
        // np×spec 병합 (plans/18)
        let mut min_gen = st.gen_toks.iter().map(|g| g.len()).min().unwrap_or(0);
        while min_gen <= n_predict {
            let active: Vec<usize> = (0..n).filter(|&s| !st.finished[s]).collect();
            if active.is_empty() {
                break;
            }
            let nexts: Vec<u32> = active.iter().map(|&s| st.next[s]).collect();
            let acc = eng
                .spec_step_multi(&active, &nexts, spec_k)
                .map_err(|e| e.to_string())?;
            stats.cycles += 1;
            let mut any = false;
            for (i, &s) in active.iter().enumerate() {
                for &t in &acc[i] {
                    if st.gen_toks[s].len() > n_predict {
                        break;
                    }
                    st.pos[s] += 1;
                    sink.on_token(s, st.pos[s], t, eng);
                    st.gen_toks[s].push(t);
                    st.next[s] = t;
                    stats.accepted += 1;
                    if t == eos {
                        st.finished[s] = true;
                    }
                    any = true;
                }
            }
            if !any {
                break;
            }
            min_gen = usize::MAX;
            for (s, g) in st.gen_toks.iter().enumerate() {
                if !st.finished[s] {
                    min_gen = min_gen.min(g.len());
                }
            }
        }
        return Ok(("spec-multi", stats));
    }
    if spec_on {
        let s = 0usize;
        while st.gen_toks[s].len() <= n_predict && !st.finished[s] {
            let (acc_toks, tf) = eng
                .spec_step(s, st.next[s], spec_k)
                .map_err(|e| e.to_string())?;
            stats.cycles += 1;
            stats.target_forwards += tf;
            for &t in &acc_toks {
                if st.gen_toks[s].len() > n_predict {
                    break;
                }
                st.pos[s] += 1;
                sink.on_token(s, st.pos[s], t, eng);
                st.gen_toks[s].push(t);
                st.next[s] = t;
                stats.accepted += 1;
                if t == eos {
                    st.finished[s] = true;
                }
            }
        }
        return Ok(("spec", stats));
    }
    // 배치 디코드 — 활성 시퀀스 묶어 1스텝 (np 상호검증 대상 경로)
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
    Ok(("batch", stats))
}

pub struct InferResult {
    pub tokens: Vec<u32>,
    /// QA-1: 엔진 확정 실패 사유 — None이면 정상 종료. 종전엔 에러 필드가
    /// 없어 실패 통보 경로 자체가 없었다(슬롯 스피너 + 클라이언트 영구 대기).
    pub error: Option<String>,
}

/// serve --spec k 전역 (기본 0).
pub static SPEC_K: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

pub enum Engine {
    Q35(Box<llm170_core::qwen35::Engine>),
    Q4(Box<llm170_core::qwen4exp::layers::Engine4>),
    /// EXL3 직접 경로 (plans/121 A1) — TrellisResident + 슬롯 SeqState.
    Exl3(Box<crate::exl3_engine::Exl3Engine>),
    Exl3Hip(Box<crate::exl3_hip_engine::Exl3HipEngine>),
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
    /// 접두 캐시 — 상태가 구워진 전체 토큰열 (요청 간 유지, plans/24).
    cached: Vec<u32>,
    /// 슬롯별 샘플러 (요청에서 생성, 토큰마다 상태 갱신).
    sampler: Option<llm170_core::sampler::Sampler>,
    /// QA-1: 연속 엔진 실패 횟수 — 성공 emit 시 0으로 리셋.
    err_count: u32,
    /// QA-1: 연속 실패 상한(3) 도달 시 확정 실패 사유.
    failed: Option<String>,
    /// plans/115 D2/P1-3: 이 잡 프리필의 시작 위치(0=신규, cp=접두 복원,
    /// l=완전 재사용) — h행 세션·mtp_draft_prefill base_pos.
    pf_base: usize,
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
            pf_base: 0,
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

/// 슬롯 로짓 → 토큰 (top-k 후보 경로) — plans/115 A-2.
fn pick_cands(s: &mut Slot, cands: &[(f32, u32)]) -> u32 {
    match &mut s.sampler {
        Some(sm) if !sm.is_greedy() => sm.sample_cands(cands),
        _ => cands
            .iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|&(_, i)| i)
            .unwrap_or(0),
    }
}

/// Q4 배치/순차 디코드(종전 Q4 arm 본체 — P15⑤ 스펙 분기로부터 분리).
fn q4_plain_decode(
    e: &mut Box<llm170_core::qwen4exp::layers::Engine4>,
    slots: &mut [Slot],
    active: &[usize],
) {
    if active
        .iter()
        .any(|&i| slots[i].sampler.as_ref().is_some_and(|sm| !sm.is_greedy()))
    {
        let toks: Vec<u32> = active.iter().map(|&i| slots[i].next).collect();
        // plans/115 A-2: GPU top-k 후보 경로 — 미지원 백엔드는 전체 로짓 폴백
        match e.decode_batch_topk(active, &toks) {
            Ok(rows) => {
                for (row, &i) in active.iter().enumerate() {
                    let t = pick_cands(&mut slots[i], &rows[row]);
                    slot_emit(&mut slots[i], t);
                }
            }
            Err(err) => {
                eprintln!("# batch 실패({err}) — 이번 회차 건너뜀");
                for &i2 in active {
                    slot_fail(&mut slots[i2], format!("decode_batch: {err}"));
                }
            }
        }
    } else if active.len() > 1 {
        let toks: Vec<u32> = active.iter().map(|&i| slots[i].next).collect();
        match e.decode_batch_greedy(active, &toks) {
            Ok(toks) => {
                for (row, &i) in active.iter().enumerate() {
                    slot_emit(&mut slots[i], toks[row]);
                }
            }
            Err(err) => {
                eprintln!("# np-greedy 실패({err}) — 이번 회차 건너뜀");
                for &i2 in active {
                    slot_fail(&mut slots[i2], format!("decode_batch_greedy: {err}"));
                }
            }
        }
    } else {
        for &i in active {
            if slots[i].sampler.as_ref().is_some_and(|sm| !sm.is_greedy()) {
                match e.decode1(i, slots[i].next) {
                    Ok(l) => {
                        let t = pick(&mut slots[i], &l);
                        slot_emit(&mut slots[i], t);
                    }
                    Err(err) => {
                        eprintln!("# decode1 실패({err})");
                        slot_fail(&mut slots[i], format!("decode1: {err}"));
                    }
                }
            } else {
                match e.decode1_greedy(i, slots[i].next) {
                    Ok(t) => slot_emit(&mut slots[i], t),
                    Err(err) => {
                        eprintln!("# decode1g 실패({err})");
                        slot_fail(&mut slots[i], format!("decode1_greedy: {err}"));
                    }
                }
            }
        }
    }
}

/// Q35 np 디코드 — 샘플링 슬롯 포함시 logits 경로(decode), 아니면 GPU argmax 판.
fn q35_decode(e: &mut llm170_core::qwen35::Engine, slots: &mut [Slot], seqs: &[usize]) {
    let toks: Vec<u32> = seqs.iter().map(|&i| slots[i].next).collect();
    if seqs
        .iter()
        .any(|&i| slots[i].sampler.as_ref().is_some_and(|s| !s.is_greedy()))
    {
        match e.decode(seqs, &toks) {
            Ok(rows) => {
                for (row, &i) in seqs.iter().enumerate() {
                    let t = pick(&mut slots[i], &rows[row]);
                    slot_emit(&mut slots[i], t);
                }
            }
            Err(err) => {
                eprintln!("# decode 실패({err}) — 이번 회차 건너뜀");
                for &i2 in seqs {
                    slot_fail(&mut slots[i2], format!("decode: {err}"));
                }
            }
        }
    } else {
        match e.decode_np_greedy(seqs, &toks) {
            Ok(toks) => {
                for (row, &i) in seqs.iter().enumerate() {
                    slot_emit(&mut slots[i], toks[row]);
                }
            }
            Err(err) => {
                eprintln!("# np-greedy 실패({err}) — 이번 회차 건너뜀");
                for &i2 in seqs {
                    slot_fail(&mut slots[i2], format!("decode_np_greedy: {err}"));
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
};
pub struct Sched {
    pub jobs: AtomicU64,
    pub queue_wait_us: AtomicU64,
    pub prefix_tokens: AtomicU64,
    pub ticks_decode: AtomicU64,
    pub ms_decode: AtomicU64,
    pub chunks_prefill: AtomicU64,
    pub ms_prefill: AtomicU64,
    /// plans/115 D2 계측 — 스펙 라운드 수/수용 토큰 수(수용률 = acc/rounds).
    pub spec_rounds: AtomicU64,
    pub spec_accepted: AtomicU64,
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
        format!(
            "[sched] jobs {jobs} | queue-wait avg {:.0}ms | prefix-reuse {px}tok | decode {td}x avg {:.1}ms | prefill {cp}x avg {:.1}ms | spec {sr}r acc {sa} ({:.2}/r)",
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
            Engine::Exl3(e) => e
                .prefill(0, &warm)
                .and_then(|l| {
                    let t = llm170_core::qwen35::greedy(&l);
                    e.decode1(0, t).map(|_| ())
                })
                .map_err(|e| e.to_string()),
            Engine::Exl3Hip(e) => e
                .prefill(&warm)
                .and_then(|l| {
                    let t = llm170_core::qwen35::greedy(&l);
                    e.decode1(t).map(|_| ())
                })
                .map_err(|e| e.to_string()),
        };
        if let Err(err) = w {
            eprintln!("# warmup 실패(치명 아님): {err}");
        }
        match &mut eng {
            Engine::Q35(e) => e.reset_states(),
            Engine::Q4(e) => e.reset_states(),
            Engine::Exl3(e) => e.reset_states(),
            Engine::Exl3Hip(e) => {
                if let Err(err) = e.reset_seq() {
                    eprintln!("# hip 리셋 실패: {err}");
                }
            }
        }
    }
    crate::http::READY.store(true, std::sync::atomic::Ordering::Release);
    let mut slots: Vec<Slot> = (0..n_slots).map(|_| Slot::free()).collect();
    let mut tick: u64 = 0;
    let (mut n_dec, mut n_pf) = (0u64, 0u64);
    let (mut ms_dec, mut ms_pf) = (0f64, 0f64);
    let npw = llm170_diag::dump::opts().key("wall_time");
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
            let Ok(j) = rx.try_recv() else {
                break;
            };
            // 107 W7: 배정 단일 구현으로 위임(접두 캐시 로직 동일).
            assign_slot(&mut slots, &mut eng, j, tick);
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
            // 샘플링 활성 슬롯 — logits 경로 필요 (GPU argmax 판은 토큰만 회수).
            // greedy 기본은 종전 최적 경로 유지 (게이트 무변화).
            let sampling = |s: &Slot| s.sampler.as_ref().is_some_and(|sm| !sm.is_greedy());
            match &mut eng {
                Engine::Q35(e) => {
                    // 스펙 슬롯 분리 — spec_step 경로 (plans/21). 샘플링 슬롯은
                    // 스펙 제외(스펙 검증은 greedy 판정 전제) — 일반 디코드로.
                    let spec_slots: Vec<usize> = active
                        .iter()
                        .copied()
                        .filter(|&i| {
                            slots[i].job.as_ref().is_some_and(|j| j.spec_k > 0)
                                && !sampling(&slots[i])
                        })
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
                            match e.spec_step(i, next, k) {
                                Ok((acc, _tf)) => {
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
                                Err(err) => {
                                    eprintln!("# spec 실패({err})");
                                    slot_fail(&mut slots[i], format!("spec_step: {err}"));
                                }
                            }
                        }
                        let plain: Vec<usize> = active
                            .iter()
                            .copied()
                            .filter(|&i| !spec_slots.contains(&i))
                            .collect();
                        if !plain.is_empty() {
                            q35_decode(e, &mut slots, &plain);
                        }
                    } else {
                        q35_decode(e, &mut slots, &active);
                    }
                }
                Engine::Q4(e) => {
                    // plans/109 P15⑤: MTP 스펙 슬롯 우선 — mtp_spec_step으로
                    // k토근 제안·검증(수용분 emit). 잔여 슬롯은 종전 배치 디코드.
                    // CPU 참조 드래프트 — ④ GPU화 전까지 느리다(스펙 슬롯만).
                    // plans/113(sglang P2-1): 스펙은 단독 활성 슬롯에서만 — 다중
                    // 활성 시 검증 무게(전상태 스냅샷+수용분 재실행)가 배치 이득을
                    // 상쇄해 순손실(serve MTP+np4 6.09 vs np4 27.03 t/s, 원장 128).
                    // spec_k를 무시하고 전원 plain np 배치로.
                    let spec_on = active.len() == 1;
                    let spec_slots: Vec<usize> = if spec_on {
                        active
                            .iter()
                            .copied()
                            .filter(|&i| {
                                slots[i].job.as_ref().is_some_and(|j| j.spec_k > 0)
                                    && !sampling(&slots[i])
                            })
                            .collect()
                    } else {
                        Vec::new()
                    };
                    if !spec_slots.is_empty() {
                        // plans/115 P12: --spec은 MTP 헤드 없이도 유효 — 서픽스 드래프터(비용 0).
                        // plans/110 W5(실험, LLM170_SPEC_MULTI=1): 다중 스펙
                        // 슬롯의 라운드 시작 decode1을 1회 np 배치로 병합. 잔여
                        // 과제: 동일 프롬프트 2슬롯 스트림이 서로 갈라진다(np
                        // 다중 슬롯 결정성 — 검증 전 기본 OFF).
                        let kmin = spec_slots
                            .iter()
                            .map(|&i| slots[i].job.as_ref().unwrap().spec_k.clamp(1, 8))
                            .min()
                            .unwrap_or(1);
                        let mut done_multi = false;
                        if e.model.has_mtp()
                            && llm170_diag::flag::on("LLM170_SPEC_MULTI")
                            && spec_slots.len() >= 2
                            && kmin >= 2
                        {
                            let ns: Vec<u32> = spec_slots.iter().map(|&i| slots[i].next).collect();
                            match e.mtp_spec_step_multi(&spec_slots, &ns, kmin) {
                                Ok((accs, _fw)) => {
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
                                    done_multi = true;
                                }
                                Err(err) => {
                                    eprintln!("# mtp spec-multi 실패({err}) — 순차로");
                                }
                            }
                        }
                        if !done_multi {
                            for &i in &spec_slots {
                                let k = slots[i].job.as_ref().unwrap().spec_k.clamp(1, 8);
                                let next = slots[i].next;
                                let cap = slots[i].job.as_ref().unwrap().n_predict;
                                // 드래프터 체인(plans/115 P12+D2): 서픽스(비용 0)
                                // 단독 — 제안 없으면 plain greedy. 콜드 MTP 폴백은
                                // 드래프트 KV가 없어 기각 일변(검증 낭비)이라 삭제.
                                let drafts = llm170_core::qwen4exp::layers::Engine4::suffix_drafts(
                                    &slots[i].tokens,
                                    k,
                                );
                                let round = if !drafts.is_empty() {
                                    e.suffix_spec_step(i, next, &drafts)
                                } else {
                                    e.decode1_greedy(i, next).map(|t| (vec![t], 1))
                                };
                                match round {
                                    Ok((acc, _fwd)) => {
                                        SCHED.spec_rounds.fetch_add(1, Ordering::Relaxed);
                                        SCHED
                                            .spec_accepted
                                            .fetch_add(acc.len() as u64, Ordering::Relaxed);
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
                                    Err(err) => {
                                        eprintln!("# mtp spec 실패({err}) — 일반 디코드로");
                                        if let Ok(l) = e.decode1(i, next) {
                                            let t = llm170_core::qwen35::greedy(&l);
                                            slot_emit(&mut slots[i], t);
                                        } else {
                                            slot_fail(
                                                &mut slots[i],
                                                format!("mtp_spec+decode1: {err}"),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                        let plain: Vec<usize> = active
                            .iter()
                            .copied()
                            .filter(|&i| !spec_slots.contains(&i))
                            .collect();
                        if !plain.is_empty() {
                            q4_plain_decode(e, &mut slots, &plain);
                        }
                    } else {
                        q4_plain_decode(e, &mut slots, &active);
                    }
                }
                Engine::Exl3Hip(e) => {
                    // hip 기본 경로(단일 슬롯 — plans/121 exl3-sched). active는 0성분.
                    for &i in &active {
                        let next = slots[i].next;
                        let r = e.decode1(next).map(|l| {
                            if slots[i].sampler.as_ref().is_some_and(|sm| !sm.is_greedy()) {
                                pick(&mut slots[i], &l)
                            } else {
                                llm170_core::qwen35::greedy(&l)
                            }
                        });
                        match r {
                            Ok(t) => slot_emit(&mut slots[i], t),
                            Err(err) => {
                                eprintln!("# hip decode 실패({err})");
                                slot_fail(&mut slots[i], format!("hip decode1: {err}"));
                            }
                        }
                    }
                }
                Engine::Exl3(e) => {
                    // EXL3 (plans/121 A1) — 슬롯별 순차 디코드(tg 4.69 t/s).
                    // np 배치·스펙 미보유 — greedy 최적, 샘플링 슬롯은 로짓 판.
                    for &i in &active {
                        let next = slots[i].next;
                        let r = e.decode1(i, next).map(|l| {
                            if slots[i].sampler.as_ref().is_some_and(|sm| !sm.is_greedy()) {
                                pick(&mut slots[i], &l)
                            } else {
                                llm170_core::qwen35::greedy(&l)
                            }
                        });
                        match r {
                            Ok(t) => slot_emit(&mut slots[i], t),
                            Err(err) => {
                                eprintln!("# exl3 decode 실패({err})");
                                slot_fail(&mut slots[i], format!("exl3 decode1: {err}"));
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
            // 배치 프리필(plans/74 np4) — 대기 슬롯 N개의 같은 길이 청크를 한 forward 로
            // 묶어 무게 패스를 공유한다(슬롯별이면 4회 읽던 것). plans/110 W8:
            // 기본 ON(등가성은 prefill_multi 등가 테스트가 보증, 실패 시 아래
            // 슬롯별 경로 폴백). (plans/115 env 정리: 킬스위치 폐기 — 항시.)
            {
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
                                        // plans/115 P1-3: 청크 경계 체크포인트 캡처.
                                        if let Engine::Q4(e) = &mut eng {
                                            e.ckpt_capture(i, slots[i].prefilled);
                                        }
                                        let done = slots[i]
                                            .job
                                            .as_ref()
                                            .is_some_and(|j| slots[i].prefilled == j.tokens.len());
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
                    let end = (slots[i].prefilled + chunk)
                        .min(slots[i].job.as_ref().unwrap().tokens.len());
                    let part: Vec<u32> =
                        slots[i].job.as_ref().unwrap().tokens[slots[i].prefilled..end].to_vec();
                    // plans/115 D2: 잡 첫 청크 — h행 세션 리셋. last_res_hc_rows
                    // 는 청크마다 extend라 리셋 없으면 잡을 넘어 무한 증가한다
                    // (종전엔 드래프트 프리필 len 검사 파탄의 원인이기도 했다).
                    if slots[i].prefilled == slots[i].pf_base
                        && let Engine::Q4(e) = &mut eng
                    {
                        e.hrows_reset(i);
                    }
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
                            let r = if samp {
                                e.prefill(i, &part)
                                    .map(|l| pick(&mut slots[i], &l))
                                    .map_err(|e| e.to_string())
                            } else {
                                e.prefill_greedy(i, &part).map_err(|e| e.to_string())
                            };
                            // plans/115 D2 측정(원장 141): serve에서 MTP 드래프트
                            // 프리필 재생([pf_base..)×~4ms/토큰)은 수용 이득 0 —
                            // 반복·패턴 수용은 서픽스 드래프터(P12, 비용 0)가
                            // 전부 담당, 비반복엔 MTP도 1.0-1.22(원장 128)이라
                            // 재생비용이 항상 우세한다. 재생 삭제 — MTP는 infer
                            // 단일스트림(측정 승리) 전용.
                            r
                        }
                        Engine::Exl3Hip(e) if i == 0 => e.prefill(&part).map(|l| {
                            if samp {
                                pick(&mut slots[i], &l)
                            } else {
                                llm170_core::qwen35::greedy(&l)
                            }
                        }),
                        Engine::Exl3Hip(_) => Err("hip 단일 슬롯: 슬롯>0 미지원".to_string()),
                        Engine::Exl3(e) => e
                            .prefill(i, &part)
                            .map(|l| {
                                if samp {
                                    pick(&mut slots[i], &l)
                                } else {
                                    llm170_core::qwen35::greedy(&l)
                                }
                            })
                            .map_err(|e| e.to_string()),
                    };
                    (end, r)
                };
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
                        // plans/115 P1-3: 청크 경계 체크포인트 캡처.
                        if let Engine::Q4(e) = &mut eng {
                            e.ckpt_capture(i, start);
                        }
                        if start == slots[i].job.as_ref().unwrap().tokens.len() {
                            slot_emit(&mut slots[i], t);
                        }
                    }
                    // QA-1: 프리필 실패 적립 — 종전엔 무시돼 prefilled가 영구
                    // 갱신되지 않는 스피너였다(매 틱 동일 청크 재시도).
                    Err(err) => slot_fail(&mut slots[i], format!("prefill: {err}")),
                }
                finish_slot(&mut slots[i], &mut eng, i, EOS);
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
        // 유휴 시 차단 수신 — 종료(송신자 전 소멸) 시 루프 탈출
        let busy = slots.iter().any(|s| s.job.is_some());
        if !busy {
            match rx.recv() {
                Ok(j) => {
                    // 107 W7: 배정 단일 구현 위임(전 슬롯 접두 탐색으로 개선 — 종전 slot0 고정).
                    assign_slot(&mut slots, &mut eng, j, tick);
                }
                Err(_) => break,
            }
        }
    }
    eprintln!("{}", SCHED.summary());
}

/// 슬롯 배정 단일 구현 (107 W7: drain/유휴 이중 복제 통합).
/// 접두 캐시 최장 일치 슬롯 선택(전 슬롯 대상 — 종전 유휴 경로는
/// slot0 고정이었음), 재사용 시 reset 생략, 샘플러 시딩 포함.
fn assign_slot(slots: &mut [Slot], eng: &mut Engine, j: SlotJob, tick: u64) {
    SCHED
        .queue_wait_us
        .fetch_add(j.queued.elapsed().as_micros() as u64, Ordering::Relaxed);
    SCHED.jobs.fetch_add(1, Ordering::Relaxed);
    let prefix_ok = true; // plans/115 env 정리: NO_PREFIX 폐기 — 접두 캐시 항시
    let pick = (0..slots.len())
        .filter(|&i| slots[i].job.is_none())
        .map(|i| {
            let l = if prefix_ok {
                slots[i]
                    .cached
                    .iter()
                    .zip(j.tokens.iter())
                    .take_while(|(a, b)| a == b)
                    .count()
            } else {
                0
            };
            let full = l > 0 && l == slots[i].cached.len() && j.tokens.len() > l;
            (i, if full { l } else { 0 })
        })
        .max_by_key(|&(_, l)| l);
    let Some((mut i, mut reuse)) = pick else {
        return;
    };
    if reuse == 0 {
        // plans/115 P1-3: 부분 접두 재사용 — 캐시가 새 프롬프트의 **접두**이기만
        // 하면(l < cached.len() 포함) 체크포인트(512 청크 경계)로 되감아 잔여를
        // 재프리필한다. GDN 상태 = 체크포인트 CPU 클론(dirty 전사), QSA KV/idx
        // 는 pos 인덱스 쓰기라 멱등 — 재프리필이 동일 행을 다시 쓴다.
        if prefix_ok {
            let mut best: Option<(usize, usize)> = None;
            for i2 in 0..slots.len() {
                if slots[i2].job.is_some() {
                    continue;
                }
                let l = slots[i2]
                    .cached
                    .iter()
                    .zip(j.tokens.iter())
                    .take_while(|(a, b)| a == b)
                    .count();
                if l >= 512 && best.is_none_or(|(_, bl)| l > bl) {
                    best = Some((i2, l));
                }
            }
            if let Some((i2, l)) = best
                && let Engine::Q4(e) = eng
                && let Some(cp) = e.ckpt_restore_upto(i2, l)
            {
                SCHED.prefix_tokens.fetch_add(cp as u64, Ordering::Relaxed);
                eprintln!(
                    "# prefix-cache: slot{i2} partial reuse {l}토큰 (ckpt {cp}에서 재프리필)"
                );
                i = i2;
                reuse = cp; // prefilled = cp — [cp..len) 재프리필(부분 접두 포함)
            }
        }
        if reuse == 0 {
            eng.reset_seq(i);
        }
    } else {
        SCHED
            .prefix_tokens
            .fetch_add(reuse as u64, Ordering::Relaxed);
        eprintln!("# prefix-cache: slot{i} reuse {reuse}토큰");
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
        pf_base: reuse,
    };
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

fn finish_slot(s: &mut Slot, eng: &mut Engine, i: usize, eos: u32) {
    // QA-1: 확정 실패 슬롯 — 에러 결과 전파 후 반환. 종전엔 토큰 방출이
    // 없어 완료 조건이 영구 거짓 → 서버 수명 동안 재시도하는 스피너였다.
    if let Some(err) = s.failed.take()
        && s.job.is_some()
    {
        if let Some(j) = s.job.take() {
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
            let _ = j.out.send(InferResult {
                tokens: toks.clone(),
                error: None,
            });
            // 접두 캐시 — 상태 유지 (프롬프트+생성 = 구워진 열).
            // 스펙 carried가 남으면 GDN이 뒤처짐 — 트렁크 재실행으로 커밋.
            if let Engine::Q35(e) = eng {
                let _ = e.flush_carried(i);
            }
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

/// n_slots 시퀀스로 엔진 구성 (연속 배칭 — 04).
pub fn build_slots(req: InferRequest, backend: BackendSel, n_slots: usize) -> Engine {
    // EXL3 직접 경로 (plans/121 A1) — --model은 EXL3 디렉터리.
    if matches!(backend, BackendSel::Exl3Hip) {
        let dir = req.model.to_string_lossy().into_owned();
        let eng = crate::exl3_hip_engine::Exl3HipEngine::load(&dir, 1, req.ctx)
            .unwrap_or_else(|e| panic!("exl3-hip 엔진 로드 실패: {e}"));
        return Engine::Exl3Hip(Box::new(eng));
    }
    if matches!(backend, BackendSel::Exl3) {
        let dir = req.model.to_string_lossy().into_owned();
        let eng = crate::exl3_engine::Exl3Engine::load(&dir, n_slots, req.ctx)
            .unwrap_or_else(|e| panic!("exl3 엔진 로드 실패: {e}"));
        return Engine::Exl3(Box::new(eng));
    }
    // plans/111 W4c: PLE 테이블 오프로드 모드(서빙 옵션 → 백엔드 전역).
    if let Some(m) = req.ple_table.as_deref()
        && let Err(e) = llm170_backend_gpu::set_ple_table_mode_by_str(m)
    {
        eprintln!("error: {e}");
    }
    if let Some(mib) = req.ple_cache_mib {
        llm170_backend_gpu::set_ple_ssd_cache_mib(mib);
    }
    let arch = open_with_retry(&req.model).and_then(|g| g.arch().map(|s| s.to_string()));
    if arch.as_deref() == Some("qwen4exp") {
        // qwen4exp GPU 경로 — plans/64 P1: 기본 CPU(정확성 기준); --backend gpu
        // 명시 시에만 상주 가속기 부착(attach_q4가 res_f16 원장 105 규칙 적용).
        let mut m = load_q4_retry(&req.model);
        if let Err(e) = apply_mtp(
            &mut m,
            &req.model,
            req.mtp.as_deref(),
            SPEC_K.get().copied().unwrap_or(0),
        ) {
            eprintln!("error: {e}");
        }
        let sources = m.part_sources();
        let eng = llm170_core::qwen4exp::layers::Engine4::new(m, n_slots, req.ctx);
        let eng = attach_q4(
            eng,
            sources,
            q4_gpu_wanted(&backend),
            q4_vk_runtime(&backend),
            // plans/115: f16 버스 기본 박탈(원장 105 승격 회수) — serve hip에서
            // 토큰 전수 파괴 실측(2026-09-30): [760,6511]→가비지 vs f32 버스로는
            // infer 골든과 완전 일치. B1 잔여(infer f16 골든 발산)와 동일 결함.
            // 산술 클래스는 f32(골든 캡처본)로 통일.
            false,
            AttachPolicy::Warn,
        )
        .unwrap_or_else(|_| unreachable!("Warn policy cannot fail"));
        Engine::Q4(Box::new(eng))
    } else {
        let m = load_q35_retry(&req.model);
        let eng = llm170_core::qwen35::Engine::new(m, n_slots, req.ctx);
        // serve --spec — 스펙 의도일 때만 MTP prefill 훅 활성 (plans/22).
        let mut eng = if SPEC_K.get().copied().unwrap_or(0) > 0 {
            let mut e = eng;
            e.mtp_wanted = true;
            e
        } else {
            eng
        };
        // plans/29: --gpu-runtime vulkan 실제 반영.
        // QA-17: --backend cpu는 부착 생략 — 종전 무조건 부착으로 라벨과
        // 실제 백엔드가 어긋났다(q4 판 q4_gpu_wanted와 대칭 계약).
        let vulkan = matches!(&backend, BackendSel::GpuRuntime(r) if r == "vulkan");
        if !matches!(&backend, BackendSel::Cpu) {
            eng = attach_q35(eng, vulkan, AttachPolicy::Warn)
                .unwrap_or_else(|_| unreachable!("Warn policy cannot fail"));
        }
        Engine::Q35(Box::new(eng))
    }
}

impl Engine {
    /// 슬롯 단위 리셋 위임.
    pub fn reset_seq(&mut self, seq: usize) {
        match self {
            Engine::Q35(e) => e.reset_seq(seq),
            Engine::Q4(e) => e.reset_seq(seq),
            Engine::Exl3(e) => e.reset_seq(seq),
            Engine::Exl3Hip(e) => {
                if let Err(err) = e.reset_seq() {
                    eprintln!("# hip 슬롯 리셋 오류: {err}");
                }
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

/// 글로벌 토크나이저 (serve 시 1회 적재).
pub static TOKENIZER: std::sync::OnceLock<crate::tokenize::Tokenizer> = std::sync::OnceLock::new();

pub fn greedy_encode(text: &str) -> Vec<u32> {
    TOKENIZER.get().map(|t| t.encode(text)).unwrap_or_default()
}
// 마커 eh2
// 마커 eh3
// 마커 eh4
// 마커 eh5
