//! 엔진 파사드 — qwen35/qwen4exp 통합, 아키텍처 자동 판별.

use std::path::PathBuf;

pub enum BackendSel {
    Cpu,
    /// 후속(W2/W3): CUDA W4A16 가속 부착 경로에서 사용 예정 — 현 프런트
    /// (serve/infer)는 Cpu 단일이라 아직 생성되지 않는다.
    #[allow(dead_code)]
    Gpu,
    /// Gpu + 런타임 지정 ("hip"|"vulkan") — serve --gpu-runtime (2026-09-01:
    /// HIP가 폴트로 웨지된 경우 Vulkan 회피).
    #[allow(dead_code)]
    GpuRuntime(String),
}
/// 모델 경로 포맷 판정 — 2026-10-08 방향 전환(plans/w4a16-cuda.md §5):
/// 수용은 **W4A16 디렉터리 단일**. EXL3·GGUF는 탈락 — 명시 에러로 안내.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ModelFormat {
    /// compressed-tensors(weight_packed 3조)·auto-gptq(qweight/qzeros) 패킹.
    W4A16,
}

/// 포맷 스니핑 — 디렉터리 내 safetensors 내용 기반(index.json 우선, 없으면
/// 첫 샤드 헤더 접두). GGUF 파일·EXL3 trellis 디렉터리는 탈락 에러.
pub fn sniff_format(path: &std::path::Path) -> Result<ModelFormat, String> {
    if path.is_file() {
        return Err(format!(
            "GGUF는 탈락(2026-10-08 — plans/w4a16-cuda.md §5): W4A16 디렉터리만 지원 — {}",
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
                "미지원 포맷: HF config 배포(architectures) — {} (지원: GGUF 파일·EXL3 trellis 디렉터리)",
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
            "EXL3는 탈락(2026-10-08 — plans/w4a16-cuda.md §5): W4A16 디렉터리만 지원 — {}",
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
    llm170_diag::flag::val("LLM170_RAWHIP") == Some("0")
}

pub fn q4_gpu_wanted(backend: &BackendSel) -> bool {
    if q4_gpu_env_off() {
        return false;
    }
    match backend {
        BackendSel::Cpu => false,
        BackendSel::Gpu => true,
        BackendSel::GpuRuntime(r) => {
            if r != "hip" && r != "vulkan" && r != "cuda" {
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

/// qwen4exp의 cuda 런타임 선택 여부 (plans/cuda-port.md §1.3 S6 — 값경로
/// Q4AccCuda).
pub fn q4_cuda_runtime(backend: &BackendSel) -> bool {
    matches!(backend, BackendSel::GpuRuntime(r) if r == "cuda")
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

/// 백엔드 부착 실패 정책 — serve는 경고 후 CPU 지속(조용한 CPU 폴백이 GPU
/// 수치로 오인된 사고 이력 — 커밋 참조).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AttachPolicy {
    Warn,
    /// 현 프런트는 Warn만 생성 — Strict는 검증 하네스 후속용으로 존치.
    #[allow(dead_code)]
    Strict,
}

/// qwen35 GPU 부착 — rawhip·rawvk 탈락(2026-10-08, plans/w4a16-cuda.md §5):
/// 잔존 GPU 부착은 CUDA 값경로뿐. 실패는 정책대로(Warn=CPU 지속 / Strict=Err).
pub fn attach_q35(
    eng: llm170_core::qwen35::Engine,
    _vulkan: bool,
    cuda: bool,
    policy: AttachPolicy,
) -> Result<llm170_core::qwen35::Engine, String> {
    if cuda {
        // plans/cuda-port.md §1.3 S6 — CUDA 값경로: Q4AccCuda(MatmulHost).
        match llm170_backend_gpu::new_q4_acc_cuda() {
            Ok(acc) => {
                eprintln!("# backend: gpu (qwen35 CUDA 값경로 — plans/cuda-port.md §1.3 S6)");
                return Ok(eng.with_acc(acc));
            }
            Err(e) => {
                eprintln!("q4acc-cuda: {e}");
                match policy {
                    AttachPolicy::Warn => return Ok(eng),
                    AttachPolicy::Strict => return Err(e),
                }
            }
        }
    }
    Ok(eng)
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
    cuda: bool,
    res_f16: bool,
    policy: AttachPolicy,
) -> Result<llm170_core::qwen4exp::layers::Engine4, String> {
    llm170_core::qwen4exp::frame::set_backend_res_f16(res_f16);
    if !want_gpu {
        // B21/P0-4(§10-3): 무음 CPU 금지 — env 게이트가 GPU를 강제로 껐으면 로그로.
        if q4_gpu_env_off() {
            eprintln!("# backend: cpu (env 게이트 LLM170_Q4_CPU=1/LLM170_RAWHIP=0 — 명시적 CPU)");
        }
        return Ok(eng);
    }
    if cuda {
        // plans/cuda-port.md §1.3 S6 — CUDA 값경로: Q4AccCuda(MatmulHost).
        // 프레임 미구현 → Engine4는 값 경로로 동작(모든 GEMV를 호스트
        // 스테이징). sources는 미소비 — Q4AccCuda는 Weight 식별 등록판이라
        // 텐서 사전 적재가 없다(첫 matmul에서 디바이스 적재).
        return match llm170_backend_gpu::new_q4_acc_cuda() {
            Ok(acc) => {
                eprintln!("# backend: gpu (qwen4exp CUDA 값경로 — plans/cuda-port.md §1.3 S6)");
                Ok(eng.with_acc(acc))
            }
            Err(e) => {
                eprintln!("error: qwen4exp CUDA 가속기 생성 실패 — {e}");
                match policy {
                    AttachPolicy::Warn => Ok(eng),
                    AttachPolicy::Strict => Err(e),
                }
            }
        };
    }
    if vk {
        // rawvk 탈락(2026-10-08) — vk 부착 경로 없음: CPU 유지.
        return Ok(eng);
    }
    // rawhip 탈락(2026-10-08) — hip 부착 경로 없음: CPU 유지(무음 금지 로그).
    let _ = sources;
    eprintln!("# backend: cpu (hip 부착 탈락 — plans/w4a16-cuda.md §5)");
    Ok(eng)
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
    let spec_on = spec_k > 0 && eng.has_mtp() && llm170_diag::flag::on("LLM170_SPEC_GPU");
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
}

impl Engine {
    /// 정지 토큰(plans/130 F5 — 하드코드 248044 일반화): Q4는 GGUF 메타,
    /// Q35는 아키텍처 상수.
    pub fn eos(&self) -> u32 {
        match self {
            Engine::Q4(e) => e.model.eos,
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

pub fn build_slots(req: InferRequest, backend: BackendSel, n_slots: usize) -> Engine {
    // 배너 format 판정(§10-2): W4A16 디렉터리는 w4a16 — 그 외는 gguf/엔진별.
    let fmt_str =
        if req.model.is_dir() && matches!(sniff_format(&req.model), Ok(ModelFormat::W4A16)) {
            "w4a16"
        } else {
            "gguf"
        };
    // PLE 플래그는 rawhip 전역 구현이었음 — hip 탈락(2026-10-08, §5)으로
    // W4A16 경로에서는 무의미. 지정 시 정직 고지.
    if req.ple_table.is_some() || req.ple_cache_mib.is_some() {
        eprintln!(
            "# ple-table/ple-cache: hip 탈락(2026-10-08) — 플래그 무시(plans/w4a16-cuda.md §5)"
        );
    }
    // 아키텍처 판별은 GGUF 파일 전용 — W4A16 디렉터리는 곧장 qwen35 경로.
    let arch = if req.model.is_dir() {
        None
    } else {
        open_with_retry(&req.model).and_then(|g| g.arch().map(|s| s.to_string()))
    };
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
            q4_cuda_runtime(&backend),
            // plans/115: f16 버스 기본 박탈(원장 105 승격 회수) — serve hip에서
            // 토큰 전수 파괴 실측(2026-09-30): [760,6511]→가비지 vs f32 버스로는
            // infer 골든과 완전 일치. B1 잔여(infer f16 골든 발산)와 동일 결함.
            // 산술 클래스는 f32(골든 캡처본)로 통일.
            false,
            AttachPolicy::Warn,
        )
        .unwrap_or_else(|_| unreachable!("Warn policy cannot fail"));
        // 배너(§10-2): attach는 acc 실재 여부, offload는 런타임별 실제.
        let runtime = if q4_cuda_runtime(&backend) {
            "cuda"
        } else if q4_vk_runtime(&backend) {
            "vulkan"
        } else if q4_gpu_wanted(&backend) {
            "hip"
        } else {
            "cpu"
        };
        let (attach, offload) = if eng.acc.is_some() {
            (
                "on",
                match runtime {
                    "cuda" => "partial(Q4K)", // B1: Q4K만 오프로드
                    "cpu" => "none",
                    _ => "full",
                },
            )
        } else if runtime == "cpu" {
            ("off(cpu-backend)", "none")
        } else {
            ("off(env-gate-or-create-failed)", "none")
        };
        banner(
            &req.model, fmt_str, runtime, offload, attach, req.ctx, n_slots,
        );
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
        let cuda = matches!(&backend, BackendSel::GpuRuntime(r) if r == "cuda");
        if !matches!(&backend, BackendSel::Cpu) {
            eng = attach_q35(eng, vulkan, cuda, AttachPolicy::Warn)
                .unwrap_or_else(|e| panic!("gpu attach: {e}"));
        }
        // 배너(§10-2) — Q4와 동일 계약.
        let runtime = if cuda {
            "cuda"
        } else if vulkan {
            "vulkan"
        } else if !matches!(&backend, BackendSel::Cpu) {
            "hip"
        } else {
            "cpu"
        };
        let (attach, offload) = if eng.acc.is_some() || eng.raw_decode.is_some() {
            (
                "on",
                match runtime {
                    "cuda" => "partial(Q4K)",
                    "cpu" => "none",
                    _ => "full",
                },
            )
        } else if runtime == "cpu" {
            ("off(cpu-backend)", "none")
        } else {
            ("off(env-gate-or-create-failed)", "none")
        };
        banner(
            &req.model, fmt_str, runtime, offload, attach, req.ctx, n_slots,
        );
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
pub use crate::sched::{SlotJob, slot_loop};
use crate::sched::{load_q4_retry, load_q35_retry, open_with_retry};

pub static TOKENIZER: std::sync::OnceLock<crate::tokenize::Tokenizer> = std::sync::OnceLock::new();

pub fn greedy_encode(text: &str) -> Vec<u32> {
    TOKENIZER.get().map(|t| t.encode(text)).unwrap_or_default()
}
// 마커 eh2
// 마커 eh3
// 마커 eh4
// 마커 eh5
