//! EXL3 CUDA 모듈층 공유 글루 — 디코더 레지스트리·선형 상주 로드·
//! safetensors/config 최소 파서·fatbin 리졸버(G2-G7 공통 — plans/124
//! G10 파일 분할 2026-10-04).
//! 모듈별 임플은 분리 파일: gemv_cuda(순차 선형 체인)·norm_cuda(norm_resid)·
//! gemm2_cuda(배치 GEMM)·gdn_cuda(GDN 체인)·attn_cuda(어텐션 prep/fwd3s)·
//! ew_argmax_cuda(ew·argmax). q4_cuda·mtp_cuda는 G8·G9 독립 파일.
//! Exl3CudaDecoder 임플 블록이 모듈 파일에 분산 상주한다(동일 타입 —
//! 순수 파일 재배치, 산술 무변경 G10 계약).
//! API 형상은 rawhip/exl3_hip.rs 미러(load·gemv_chain·gemv_host·norm).
//! 3층 분리 원칙(plans/124 §5): 이 층은 가중치 상주 + 상태 + forward API의
//! 단일 진실 — 검증 자산(덤프·사다리 인자)은 exl3_cuda_probe로 금지.
//! 트렐리스 비트 조작식은 커널 소스(assets/exl3_gemv.cu ← src_exl3.hip
//! 직이식)에만 존재한다 — Rust층 파생 금지(2회 사고 원장, plans/124 §2).
//! MTP·배치·그래프는 후속 목표(G7+) — 관련 필드는 유효한
//! 디바이스 포인터를 담지 않는다(0 유지 계약; dkc/dvc/dpp는 G6부터
//! 어텐션 KV 캐시·pos 소유).
//!
//! 독립 컴파일 계약: 이 파일(및 분리 모듈 전체)은 scripts/cuda_probe_shim.rs가
//! rustc로 단독 컴파일한다(전체 워크스페이스는 Windows에서 llm170-core mmap
//! 결함으로 불가 — G1 원장). 따라서 std 외 크레이트 의존 금지 — safetensors
//! 파싱은 llm170_exl3::StArchive의 최소 미러를 내장한다(serde 금지 규약과
//! 동일 노선).

use crate::rawcuda::attn_cuda::AttnDims;
use crate::rawcuda::ctx::CudaCtx;
use crate::rawcuda::ffi::CUdeviceptr;
use crate::rawcuda::gdn_cuda::GdnDims;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// GEMV nseg — [nseg=16][n] 분할 부분합(원장: nseg 4→16 plans/120 A1).
pub const GEMV_NSEG: usize = 16;

/// 선형 1개의 상주 사양 — HipLin 미러(k·n·krate + 트렐리스 가중 3중).
pub struct CudaLin {
    pub k: usize,
    pub n: usize,
    pub krate: u32,
    /// had_in 스케일·팩 계수(선형별 — k값 탐색 금지, 결함 1호).
    /// f16 LE [k] = u32쌍팩 [k/2] — 커널 suh[ki>>1] 규약.
    pub suh: CUdeviceptr,
    /// 트렐리스 코드북 u32 워드 [kt][nt][8K] — 비트 조작식은 원본 1:1.
    pub tre: CUdeviceptr,
    /// had_out 역변환 계수 f16 LE [n](H⁻¹⊙svh — 생략 시 전면 붕괴, 결함 3호).
    pub svh: CUdeviceptr,
}

/// EXL3 CUDA 디코더 — Exl3HipDecoder 미러.
/// G2 상태: lin 레지스트리·GEMV 체인 버퍼(dah/dsb/dyb/dx) 유효.
/// S5(plans/cuda-port.md §3): load는 임베딩 호스트 상주 + 노름·GDN·어텐션
/// 상수 등록. load_keys는 검증용 선형 전용 경로로 유지한다.
pub struct Exl3CudaDecoder {
    /// 디바이스 컨텍스트(디바이스·스트림·커널 레지스트리).
    pub cc: CudaCtx,
    /// 선형 레지스트리(이름 → 상주 사양).
    pub lin: HashMap<String, CudaLin>,
    pub hidden: usize,
    pub n_layers: usize,
    pub loaded_layers: usize,
    pub pos: u32,
    /// plans/cuda-port.md S5: 전층 선형 VRAM 예산 때문에 임베딩은 RAM에 유지.
    /// [vocab][hidden] f32; 디코드 시 해당 행만 장치로 옮긴다.
    pub embed: Vec<f32>,
    pub vocab: usize,
    /// 구형 디바이스 임베딩 슬롯 — S5 호스트 스테이징에서는 0 유지.
    pub dembed: CUdeviceptr,
    /// 현 스텝 잔류 스트림 x — GEMV 체인 입력 스테이징 겸용(G2 유효).
    pub dx: CUdeviceptr,
    /// 노름 가중합 xn = rms_norm(x+ab)·w — G3 유효.
    pub dxn: CUdeviceptr,
    /// 노름 가산 분기(ab)·가중 배열(nw) 상주 — G3 유효. nw는 행 포인터
    /// 계약(오프셋 w·hidden — 누락이 결함 2호: 전 노름 L0 행 판독).
    pub dab: CUdeviceptr,
    pub dnw: CUdeviceptr,
    pub dzero: CUdeviceptr,
    /// had_in 출력 f16 H도메인 u32쌍팩 [kmax/2] — G2 유효.
    pub dah: CUdeviceptr,
    /// GEMV 부분합 sb([nseg][nmax] — had_out이 nseg 합산) — G2 유효.
    pub dsb: CUdeviceptr,
    /// had_out 출력 [nmax] — G2 유효.
    pub dyb: CUdeviceptr,
    /// 배치 had_in 출력 f16쌍팩 [t][k/2] u32 — G4 유효.
    pub daht: CUdeviceptr,
    /// 배치 gemm2+had_out 출력 [t][n] f32 — G4 유효.
    pub dyt: CUdeviceptr,
    /// kseg 부분합 [T≤8][kseg=8][n] f32 — G4 유효(had_out이 합산).
    pub dbat: CUdeviceptr,
    // ── GDN 체인(conv→l2perm→scan→gate, §3.3) G5 유효 ──
    /// GDN 형상(set_gdn 등록 — None이면 미초기화).
    pub gdn: Option<GdnDims>,
    /// conv 입력 qkv [t][conv_ch] — G5.
    pub dqkv: CUdeviceptr,
    /// in_proj_z 산출 z [t][v_len] — G5.
    pub dzv: CUdeviceptr,
    /// conv 산출 q/k/v — G5.
    pub dgq: CUdeviceptr,
    pub dgk: CUdeviceptr,
    pub dgv: CUdeviceptr,
    /// l2perm 입력 xn(=xtb [t][hidden]) — G5.
    pub dgxn: CUdeviceptr,
    /// l2perm 산출 q/k(L2)·v(lc)·bg — G5.
    pub dq2: CUdeviceptr,
    pub dk2: CUdeviceptr,
    pub dv2: CUdeviceptr,
    pub dbg: CUdeviceptr,
    /// scan 산출 o_lc [t][v_len] — G5.
    pub dgo: CUdeviceptr,
    /// gate 산출 gated [t][v_len](HF) — G5.
    pub dgate: CUdeviceptr,
    /// conv 3탭 링 [n_gdn][3][conv_ch](커널이 r/w — T행 순차 계약,
    /// 상주) — G5.
    pub dring: CUdeviceptr,
    /// GDN 상태 [n_gdn][h_v][128·128](커널 r/w — S0≠0 경로 의무) — G5.
    pub dgst: CUdeviceptr,
    /// GDN 상수 상주: conv 가중 [n_gdn][conv_ch][4] · a/b
    /// [n_gdn][2][h_v][hidden] · alog/dtb [n_gdn][h_v] · 노름가중
    /// [n_gdn][128] — G5.
    pub dcw: CUdeviceptr,
    pub dab_c: CUdeviceptr,
    pub dalog: CUdeviceptr,
    pub ddtb: CUdeviceptr,
    pub dnwg: CUdeviceptr,
    pub dkc: CUdeviceptr,
    pub dvc: CUdeviceptr,
    /// KV 인덱스 — pp[0] 디바이스 판독 계약(결함 4호). prep/fwd3s가
    /// 발사 인자가 아니라 이 버퍼에서 pos를 판독한다(그래프/루프 설계
    /// 핵심 — 그래프 내 전진은 exl3_attn_pos_bump) — G6.
    pub dpp: CUdeviceptr,
    // ── 어텐션 체인(prep + fwd3s, §3.4/q_norm·rope base 1e7) G6 유효 ──
    /// 어텐션 형상(set_attn 등록 — None이면 미초기화).
    pub attn: Option<AttnDims>,
    /// q/k 노름 상주 [n_attn][256](저장소 w−1 → +1한 값 등록 규약) — G6.
    pub dqnw_a: CUdeviceptr,
    pub dknw_a: CUdeviceptr,
    /// qg 스테이징 [t][qg_dim] · kin/vin 스테이징 [t][kv_dim] — G6.
    pub dqg_a: CUdeviceptr,
    pub dkin_a: CUdeviceptr,
    pub dvin_a: CUdeviceptr,
    /// prep 산출 qh · fwd3s 산출 outv [t][q_dim] — G6.
    pub dqh_a: CUdeviceptr,
    pub doutv_a: CUdeviceptr,
    // ── ew(silu·mul)·argmax 체인 G7 유효 ──
    /// ew 스테이징 g·u [n] 및 출력 y [n](hip dew 명명 계승) — G7.
    pub dewg: CUdeviceptr,
    pub dewu: CUdeviceptr,
    pub dew: CUdeviceptr,
    /// argmax 로짓 스테이징 [n] — G7.
    pub dlgmax: CUdeviceptr,
    // ── MTP 드래프트(§3.4 — 노름 w−1 저장 규약: constant_bias=1.0만) G4+ ──
    pub mtp_norms: Vec<Vec<f32>>,
    pub mtp_kv_k: Vec<f32>,
    pub mtp_kv_v: Vec<f32>,
    pub mtp_kv_len: usize,
    pub dmtpin: CUdeviceptr,
    pub dmtpk: CUdeviceptr,
    pub dmtpv: CUdeviceptr,
    pub dmtpp: CUdeviceptr,
    /// 배치 h 곡선 디버그 — 배치·순차 h 대조 프로브 우선 과제(§7.1).
    pub dbg_layers: bool,
    pub dbg_hcurve: bool,
    pub hcurve: Vec<(usize, Vec<f32>)>,
    /// CUDA 그래프 실행 핸들(캡처 T와 쌍) — cuStreamBeginCapture 도입 시.
    pub gexec: Option<(usize, *mut std::ffi::c_void)>,
    /// argmax 출력 토큰 [1]u32 — G7(ensure_argmax_buf가 최초 할당).
    pub dargmax: CUdeviceptr,
    /// GEMV 체인 작업 버퍼 상한(ensure_bufs가 갱신 — 재할당 최소화).
    pub(crate) kmax: usize,
    pub(crate) nmax: usize,
    /// 잔차 스트림 dx 용량(원소 수 — GEMV 스테이징·노름 공유 확장).
    pub(crate) x_cap: usize,
    /// 노름 버퍼(dab·dxn) 용량(원소 수)·nw 등록 행 수.
    pub(crate) norm_cap: usize,
    pub(crate) norm_w_rows: usize,
    /// 배치 GEMM 버퍼 용량: daht 바이트·dyt 바이트·dbat n 폭.
    pub(crate) gemm_ah_cap: usize,
    pub(crate) gemm_y_cap: usize,
    pub(crate) gemm_bat_n: usize,
    /// GDN 작업 버퍼 t 상한(행수 — 확장 시에만 재할당, set_gdn 리셋).
    pub(crate) gdn_t_cap: usize,
    /// 어텐션 작업 버퍼 t 상한(행수 — 확장 시에만 재할당, set_attn 리셋).
    pub(crate) attn_t_cap: usize,
    /// ew 작업 버퍼 원소 상한 · argmax 로짓 스테이징 상한(원소 수) — G7.
    pub(crate) ew_cap: usize,
    pub(crate) lg_cap: usize,
}

// SAFETY: CudaCtx·CUdeviceptr 소유 — 단일 스레드 사용 계약(rawhip
// Exl3HipDecoder와 동일 — 서버 slot_loop 단일 소유).
unsafe impl Send for Exl3CudaDecoder {}

// ── 최소 JSON 파서(safetensors 헤더·config.json 전용 — serde 금지 규약) ──
/// llm170_exl3::Json(crates/exl3/src/json.rs)의 축소 미러 — 순수 std.
#[derive(Debug, Clone)]
pub(crate) enum JVal {
    Num(f64),
    Str(String),
    Arr(Vec<JVal>),
    Obj(Vec<(String, JVal)>),
    Bool(bool),
    Nul,
}

impl JVal {
    pub(crate) fn get(&self, key: &str) -> Option<&JVal> {
        match self {
            JVal::Obj(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    pub(crate) fn as_f64(&self) -> Option<f64> {
        match self {
            JVal::Num(v) => Some(*v),
            _ => None,
        }
    }
    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            JVal::Str(s) => Some(s),
            _ => None,
        }
    }
    pub(crate) fn as_arr(&self) -> Option<&[JVal]> {
        match self {
            JVal::Arr(v) => Some(v),
            _ => None,
        }
    }
    pub(crate) fn as_obj(&self) -> Option<&[(String, JVal)]> {
        match self {
            JVal::Obj(m) => Some(m),
            _ => None,
        }
    }
}

pub(crate) struct JParser<'a> {
    pub(crate) b: &'a [u8],
    pub(crate) p: usize,
}

impl JParser<'_> {
    pub(crate) fn parse(mut self) -> Result<JVal, String> {
        self.value(0)
    }
    fn ws(&mut self) {
        while self.p < self.b.len() && matches!(self.b[self.p], b' ' | b'\t' | b'\n' | b'\r') {
            self.p += 1;
        }
    }
    fn value(&mut self, depth: u32) -> Result<JVal, String> {
        if depth > 32 {
            return Err("JSON 깊이 상한 초과".into());
        }
        self.ws();
        match self.b.get(self.p) {
            Some(b'{') => {
                self.p += 1;
                let mut m = Vec::new();
                loop {
                    self.ws();
                    match self.b.get(self.p) {
                        Some(b'}') => {
                            self.p += 1;
                            break;
                        }
                        Some(b',') if !m.is_empty() => self.p += 1,
                        Some(b'"') => {
                            let k = self.string()?;
                            self.ws();
                            if self.b.get(self.p) != Some(&b':') {
                                return Err("JSON ':' 없음".into());
                            }
                            self.p += 1;
                            let v = self.value(depth + 1)?;
                            m.push((k, v));
                        }
                        _ => return Err("JSON 객체 문법 오류".into()),
                    }
                }
                Ok(JVal::Obj(m))
            }
            Some(b'[') => {
                self.p += 1;
                let mut a = Vec::new();
                loop {
                    self.ws();
                    match self.b.get(self.p) {
                        Some(b']') => {
                            self.p += 1;
                            break;
                        }
                        Some(b',') if !a.is_empty() => self.p += 1,
                        _ => {
                            let v = self.value(depth + 1)?;
                            a.push(v);
                        }
                    }
                }
                Ok(JVal::Arr(a))
            }
            Some(b'"') => Ok(JVal::Str(self.string()?)),
            Some(b't') if self.b[self.p..].starts_with(b"true") => {
                self.p += 4;
                Ok(JVal::Bool(true))
            }
            Some(b'f') if self.b[self.p..].starts_with(b"false") => {
                self.p += 5;
                Ok(JVal::Bool(false))
            }
            Some(b'n') if self.b[self.p..].starts_with(b"null") => {
                self.p += 4;
                Ok(JVal::Nul)
            }
            Some(_) => {
                let s = self.p;
                if self.b.get(s) == Some(&b'-') {
                    self.p += 1;
                }
                let d0 = self.p;
                while self.b.get(self.p).is_some_and(|c| c.is_ascii_digit()) {
                    self.p += 1;
                }
                if self.p == d0 {
                    return Err("JSON 숫자 없음".into());
                }
                if self.b.get(self.p) == Some(&b'.') {
                    self.p += 1;
                    while self.b.get(self.p).is_some_and(|c| c.is_ascii_digit()) {
                        self.p += 1;
                    }
                }
                if matches!(self.b.get(self.p), Some(b'e') | Some(b'E')) {
                    self.p += 1;
                    if matches!(self.b.get(self.p), Some(b'+') | Some(b'-')) {
                        self.p += 1;
                    }
                    while self.b.get(self.p).is_some_and(|c| c.is_ascii_digit()) {
                        self.p += 1;
                    }
                }
                let txt = std::str::from_utf8(&self.b[s..self.p]).map_err(|_| "JSON 인코딩")?;
                txt.parse::<f64>()
                    .map(JVal::Num)
                    .map_err(|e| format!("JSON 숫자 {txt}: {e}"))
            }
            None => Err("JSON 예기치 않은 끝".into()),
        }
    }
    fn string(&mut self) -> Result<String, String> {
        if self.b.get(self.p) != Some(&b'"') {
            return Err("JSON 문자열 아님".into());
        }
        self.p += 1;
        let mut out = String::new();
        loop {
            match self.b.get(self.p) {
                Some(b'"') => {
                    self.p += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.p += 1;
                    match self.b.get(self.p) {
                        Some(b'"') => out.push('"'),
                        Some(b'\\') => out.push('\\'),
                        Some(b'/') => out.push('/'),
                        Some(b'b') => out.push('\u{8}'),
                        Some(b'f') => out.push('\u{c}'),
                        Some(b'n') => out.push('\n'),
                        Some(b'r') => out.push('\r'),
                        Some(b't') => out.push('\t'),
                        Some(b'u') => {
                            if self.p + 4 >= self.b.len() {
                                return Err("JSON \\u 종결".into());
                            }
                            let hex = std::str::from_utf8(&self.b[self.p + 1..self.p + 5])
                                .map_err(|_| "JSON \\u 인코딩")?;
                            let cp = u32::from_str_radix(hex, 16).map_err(|_| "JSON \\u 자릿수")?;
                            out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                            self.p += 4;
                        }
                        _ => return Err("JSON 이스케이프 미지원".into()),
                    }
                    self.p += 1;
                }
                Some(&c) if c < 0x80 => {
                    out.push(c as char);
                    self.p += 1;
                }
                Some(_) => {
                    // 다중바이트 UTF-8 통과(필요 키·문자열은 ASCII지만 방어적 수용).
                    let s = self.p;
                    let mut e = self.p + 1;
                    while e < self.b.len() && (self.b[e] & 0xC0) == 0x80 {
                        e += 1;
                    }
                    out.push_str(std::str::from_utf8(&self.b[s..e]).map_err(|_| "JSON UTF-8")?);
                    self.p = e;
                }
                None => return Err("JSON 문자열 종결 없음".into()),
            }
        }
    }
}

// ── safetensors 최소 리더(llm170_exl3::StArchive 미러 — 선형 3중 전용) ──
pub(crate) struct StEntry {
    begin: u64,
    end: u64,
    shard: usize,
    shape: Vec<u64>,
    /// dtype 코드(0 F32 · 1 F16 · 2 BF16 · 3 I16 · 4 I32 · 5 I64 · 6 U8).
    pub(crate) dt: u8,
}

pub(crate) struct StArchive {
    shards: Vec<PathBuf>,
    data_base: Vec<u64>,
    entries: HashMap<String, StEntry>,
}

impl StArchive {
    /// `.safetensors` 파일 하나 또는 샤드 디렉터리(index.json 우선).
    pub(crate) fn open(path: &Path) -> Result<Self, String> {
        if path.is_dir() {
            Self::open_dir(path)
        } else {
            let (db, es) = Self::parse_shard(path, 0)?;
            Ok(Self {
                shards: vec![path.to_path_buf()],
                data_base: vec![db],
                entries: es,
            })
        }
    }

    fn open_dir(dir: &Path) -> Result<Self, String> {
        let idx = dir.join("model.safetensors.index.json");
        let mut names: Vec<String> = Vec::new();
        let mut index_map: Option<HashMap<String, String>> = None;
        if idx.exists() {
            let raw = std::fs::read_to_string(&idx).map_err(|e| e.to_string())?;
            let v = JParser {
                b: raw.as_bytes(),
                p: 0,
            }
            .parse()?;
            let wm = v
                .get("weight_map")
                .and_then(JVal::as_obj)
                .ok_or("index.json: weight_map 없음")?;
            let mut m = HashMap::new();
            for (k, val) in wm {
                let f = val.as_str().ok_or("weight_map 값이 문자열 아님")?;
                if !names.iter().any(|s| s == f) {
                    names.push(f.to_string());
                }
                m.insert(k.clone(), f.to_string());
            }
            index_map = Some(m);
        } else {
            for e in std::fs::read_dir(dir).map_err(|e| e.to_string())? {
                let e = e.map_err(|e| e.to_string())?;
                let name = e.file_name().to_string_lossy().into_owned();
                if name.ends_with(".safetensors") && name.starts_with("model") {
                    names.push(name);
                }
            }
            if names.is_empty() {
                return Err(format!("model*.safetensors 없음: {}", dir.display()));
            }
        }
        names.sort();
        let mut shards = Vec::new();
        let mut data_base = Vec::new();
        let mut entries = HashMap::new();
        for (si, fname) in names.iter().enumerate() {
            let p = dir.join(fname);
            let (db, es) = Self::parse_shard(&p, si)?;
            shards.push(p);
            data_base.push(db);
            for (name, mut e) in es {
                e.shard = si;
                let take = match &index_map {
                    Some(m) => m.get(&name).is_some_and(|f| f == fname),
                    None => true,
                };
                if take {
                    entries.insert(name, e);
                }
            }
        }
        Ok(Self {
            shards,
            data_base,
            entries,
        })
    }

    /// 단일 샤드 헤더 파싱 — 데이터 오프셋·형상·dtype 바이트 수 대조
    /// (llm170_exl3 파손 방어 미러).
    fn parse_shard(path: &Path, shard: usize) -> Result<(u64, HashMap<String, StEntry>), String> {
        let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut lenb = [0u8; 8];
        f.read_exact(&mut lenb).map_err(|e| e.to_string())?;
        let hlen = u64::from_le_bytes(lenb);
        if hlen == 0 || hlen > (1 << 30) {
            return Err(format!("{}: 헤더 길이 {hlen}", path.display()));
        }
        let mut hb = vec![0u8; hlen as usize];
        f.read_exact(&mut hb).map_err(|e| e.to_string())?;
        let data_base = 8 + hlen;
        let v = JParser { b: &hb, p: 0 }.parse()?;
        let obj = v.as_obj().ok_or("헤더가 객체 아님")?;
        let mut out = HashMap::new();
        for (name, tv) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dt = tv
                .get("dtype")
                .and_then(JVal::as_str)
                .ok_or_else(|| format!("{name}: dtype 없음"))?;
            let (nb, dtc) = match dt {
                "F16" => (2u64, 1u8),
                "BF16" => (2, 2),
                "I16" => (2, 3),
                "F32" => (4, 0),
                "I32" => (4, 4),
                "I64" => (8, 5),
                "U8" => (1, 6),
                other => return Err(format!("{name}: 미지원 dtype {other}")),
            };
            let shape = tv
                .get("shape")
                .and_then(JVal::as_arr)
                .ok_or_else(|| format!("{name}: shape 없음"))?
                .iter()
                .map(|s| s.as_f64().unwrap_or(0.0) as u64)
                .collect::<Vec<u64>>();
            let offs = tv
                .get("data_offsets")
                .and_then(JVal::as_arr)
                .ok_or_else(|| format!("{name}: data_offsets 없음"))?;
            let (b, e) = (
                offs.first().and_then(JVal::as_f64),
                offs.get(1).and_then(JVal::as_f64),
            );
            let (Some(b), Some(e)) = (b, e) else {
                return Err(format!("{name}: data_offsets 값 오류"));
            };
            let numel: u64 = shape.iter().product();
            if (e - b) as u64 != numel * nb {
                return Err(format!("{name}: 바이트 {} != numel {numel} × {nb}", e - b));
            }
            out.insert(
                name.clone(),
                StEntry {
                    begin: b as u64,
                    end: e as u64,
                    shard,
                    shape,
                    dt: dtc,
                },
            );
        }
        Ok((data_base, out))
    }

    /// 텐서 원시 바이트 직독.
    pub(crate) fn read(&self, name: &str) -> Result<Vec<u8>, String> {
        let e = self
            .entries
            .get(name)
            .ok_or_else(|| format!("텐서 없음: {name}"))?;
        let mut f = std::fs::File::open(&self.shards[e.shard]).map_err(|err| err.to_string())?;
        f.seek(SeekFrom::Start(self.data_base[e.shard] + e.begin))
            .map_err(|err| err.to_string())?;
        let n = (e.end - e.begin) as usize;
        let mut buf = vec![0u8; n];
        f.read_exact(&mut buf).map_err(|err| err.to_string())?;
        Ok(buf)
    }

    /// plans/cuda-port.md S5: 지정 형상·부동 dtype을 확인한 뒤 원시 텐서를
    /// f32로 확장한다. 프로브의 st_to_f32와 독립된 생산 경로(정수 오인 금지).
    fn read_f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let entry = self
            .entries
            .get(name)
            .ok_or_else(|| format!("텐서 없음: {name}"))?;
        if entry.shape.len() != shape.len()
            || !entry
                .shape
                .iter()
                .zip(shape)
                .all(|(&got, &want)| got == want as u64)
        {
            return Err(format!("{name}: shape {:?} != {shape:?}", entry.shape));
        }
        let n = shape
            .iter()
            .try_fold(1usize, |a, b| a.checked_mul(*b))
            .ok_or_else(|| format!("{name}: 원소 수 범위 초과"))?;
        let stride = match entry.dt {
            0 => 4usize, // F32
            1 | 2 => 2,  // F16 / BF16
            dt => return Err(format!("{name}: 부동 dtype 필요, 코드 {dt}")),
        };
        let bytes = n
            .checked_mul(stride)
            .ok_or_else(|| format!("{name}: 바이트 수 범위 초과"))?;
        if entry.end - entry.begin != bytes as u64 {
            return Err(format!("{name}: 원소 수/바이트 길이 불일치"));
        }
        let raw = self.read(name)?;
        let mut out = Vec::with_capacity(n);
        match entry.dt {
            0 => {
                for b in raw.as_chunks::<4>().0 {
                    out.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
                }
            }
            1 => {
                for b in raw.as_chunks::<2>().0 {
                    out.push(exl3_f16_to_f32(u16::from_le_bytes([b[0], b[1]])));
                }
            }
            2 => {
                for b in raw.as_chunks::<2>().0 {
                    out.push(f32::from_bits(
                        (u16::from_le_bytes([b[0], b[1]]) as u32) << 16,
                    ));
                }
            }
            _ => unreachable!(),
        }
        Ok(out)
    }

    /// 텐서 dtype 코드(0 F32 · 1 F16 · 2 BF16 — 상수 해독용).
    pub(crate) fn dtype_of(&self, name: &str) -> Option<u8> {
        self.entries.get(name).map(|e| e.dt)
    }

    /// 텐서 형상(선형 k·n·krate 산출용).
    pub(crate) fn shape_of(&self, name: &str) -> Option<&[u64]> {
        self.entries.get(name).map(|e| e.shape.as_slice())
    }

    /// 전 텐서 열거 (이름, dtype 코드, shape) — Flash-Next 인벤토리
    /// 프로브용(FNA 2026-10-05). dtype 코드 계약은 parse_shard 참조
    /// (0 F32류·1 F16·2 BF16·3 기타 2B).
    pub(crate) fn each_tensor(&self, mut f: impl FnMut(&str, u8, &[u64])) {
        for (name, e) in &self.entries {
            f(name, e.dt, &e.shape);
        }
    }

    /// 등록 텐서 수·샤드 수 — 인벤토리 요약용(FNA).
    pub(crate) fn tensor_count(&self) -> usize {
        self.entries.len()
    }
    pub(crate) fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// `.trellis` 접미 키 전체(trellis 텐서명 → 선형 기저명).
    fn linear_base_keys(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .entries
            .keys()
            .filter(|k| k.ends_with(".trellis"))
            .map(|k| k.trim_end_matches(".trellis").to_string())
            .collect();
        v.sort();
        v
    }
}

/// plans/cuda-port.md S5: 반정밀 형상/서브노멀까지 보존하는 std 전용 변환.
/// 검증층(exl3_cuda_probe.rs) 함수를 참조하지 않는 생산 경로.
fn exl3_f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut m = mant;
            let mut shift = 0u32;
            while m & 0x0400 == 0 {
                m <<= 1;
                shift += 1;
            }
            sign | ((113 - shift) << 23) | ((m & 0x03ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

// ── 모듈층 본체 ──
impl Exl3CudaDecoder {
    /// exl3_gemv.fatbin 자산 해석 — LLM170_CUDA_EXL3_FATBIN_PATH 오버라이드
    /// 우선(G1 smoke 리졸버 미러 — 자산 경로 오버라이드일 뿐 계산 경로
    /// 분기 아님). 후보는 실행 기준 상대 경로 2종(저장소 루트·크레이트 루트).
    fn exl3_fatbin_bytes() -> Result<Vec<u8>, String> {
        const ENV: &str = "LLM170_CUDA_EXL3_FATBIN_PATH";
        const REL: &[&str] = &[
            "crates/backend-gpu/src/rawcuda/assets/exl3_gemv.fatbin",
            "src/rawcuda/assets/exl3_gemv.fatbin",
        ];
        if let Some(p) = std::env::var_os(ENV) {
            return std::fs::read(&p).map_err(|e| format!("{ENV}({p:?}) 읽기 실패: {e}"));
        }
        for r in REL {
            if let Ok(b) = std::fs::read(r) {
                return Ok(b);
            }
        }
        Err(format!(
            "exl3_gemv.fatbin 없음 — scripts/build_cuda.bat 실행 또는 {ENV} 지정 (탐색: {REL:?})"
        ))
    }

    /// exl3_norm.fatbin 자산 해석 — LLM170_CUDA_EXL3_NORM_FATBIN_PATH
    /// 오버라이드 우선(자산 경로 오버라이드일 뿐 계산 경로 분기 아님).
    fn exl3_norm_fatbin_bytes() -> Result<Vec<u8>, String> {
        const ENV: &str = "LLM170_CUDA_EXL3_NORM_FATBIN_PATH";
        const REL: &[&str] = &[
            "crates/backend-gpu/src/rawcuda/assets/exl3_norm.fatbin",
            "src/rawcuda/assets/exl3_norm.fatbin",
        ];
        if let Some(p) = std::env::var_os(ENV) {
            return std::fs::read(&p).map_err(|e| format!("{ENV}({p:?}) 읽기 실패: {e}"));
        }
        for r in REL {
            if let Ok(b) = std::fs::read(r) {
                return Ok(b);
            }
        }
        Err(format!(
            "exl3_norm.fatbin 없음 — scripts/build_cuda.bat 실행 또는 {ENV} 지정 (탐색: {REL:?})"
        ))
    }

    /// exl3_gemm2.fatbin 자산 해석 — LLM170_CUDA_EXL3_GEMM2_FATBIN_PATH
    /// 오버라이드 우선(자산 경로 오버라이드일 뿐 계산 경로 분기 아님).
    fn exl3_gemm2_fatbin_bytes() -> Result<Vec<u8>, String> {
        const ENV: &str = "LLM170_CUDA_EXL3_GEMM2_FATBIN_PATH";
        const REL: &[&str] = &[
            "crates/backend-gpu/src/rawcuda/assets/exl3_gemm2.fatbin",
            "src/rawcuda/assets/exl3_gemm2.fatbin",
        ];
        if let Some(p) = std::env::var_os(ENV) {
            return std::fs::read(&p).map_err(|e| format!("{ENV}({p:?}) 읽기 실패: {e}"));
        }
        for r in REL {
            if let Ok(b) = std::fs::read(r) {
                return Ok(b);
            }
        }
        Err(format!(
            "exl3_gemm2.fatbin 없음 — scripts/build_cuda.bat 실행 또는 {ENV} 지정 (탐색: {REL:?})"
        ))
    }

    /// exl3_gdn.fatbin 자산 해석 — LLM170_CUDA_EXL3_GDN_FATBIN_PATH
    /// 오버라이드 우선(자산 경로 오버라이드일 뿐 계산 경로 분기 아님).
    fn exl3_gdn_fatbin_bytes() -> Result<Vec<u8>, String> {
        const ENV: &str = "LLM170_CUDA_EXL3_GDN_FATBIN_PATH";
        const REL: &[&str] = &[
            "crates/backend-gpu/src/rawcuda/assets/exl3_gdn.fatbin",
            "src/rawcuda/assets/exl3_gdn.fatbin",
        ];
        if let Some(p) = std::env::var_os(ENV) {
            return std::fs::read(&p).map_err(|e| format!("{ENV}({p:?}) 읽기 실패: {e}"));
        }
        for r in REL {
            if let Ok(b) = std::fs::read(r) {
                return Ok(b);
            }
        }
        Err(format!(
            "exl3_gdn.fatbin 없음 — scripts/build_cuda.bat 실행 또는 {ENV} 지정 (탐색: {REL:?})"
        ))
    }

    /// exl3_attn.fatbin 자산 해석 — LLM170_CUDA_EXL3_ATTN_FATBIN_PATH
    /// 오버라이드 우선(자산 경로 오버라이드일 뿐 계산 경로 분기 아님).
    fn exl3_attn_fatbin_bytes() -> Result<Vec<u8>, String> {
        const ENV: &str = "LLM170_CUDA_EXL3_ATTN_FATBIN_PATH";
        const REL: &[&str] = &[
            "crates/backend-gpu/src/rawcuda/assets/exl3_attn.fatbin",
            "src/rawcuda/assets/exl3_attn.fatbin",
        ];
        if let Some(p) = std::env::var_os(ENV) {
            return std::fs::read(&p).map_err(|e| format!("{ENV}({p:?}) 읽기 실패: {e}"));
        }
        for r in REL {
            if let Ok(b) = std::fs::read(r) {
                return Ok(b);
            }
        }
        Err(format!(
            "exl3_attn.fatbin 없음 — scripts/build_cuda.bat 실행 또는 {ENV} 지정 (탐색: {REL:?})"
        ))
    }

    /// exl3_ew.fatbin 자산 해석 — LLM170_CUDA_EXL3_EW_FATBIN_PATH 오버라이드
    /// 우선(자산 경로 오버라이드일 뿐 계산 경로 분기 아님).
    fn exl3_ew_fatbin_bytes() -> Result<Vec<u8>, String> {
        const ENV: &str = "LLM170_CUDA_EXL3_EW_FATBIN_PATH";
        const REL: &[&str] = &[
            "crates/backend-gpu/src/rawcuda/assets/exl3_ew.fatbin",
            "src/rawcuda/assets/exl3_ew.fatbin",
        ];
        if let Some(p) = std::env::var_os(ENV) {
            return std::fs::read(&p).map_err(|e| format!("{ENV}({p:?}) 읽기 실패: {e}"));
        }
        for r in REL {
            if let Ok(b) = std::fs::read(r) {
                return Ok(b);
            }
        }
        Err(format!(
            "exl3_ew.fatbin 없음 — scripts/build_cuda.bat 실행 또는 {ENV} 지정 (탐색: {REL:?})"
        ))
    }

    /// 컨텍스트 + GEMV·노름·배치 GEMM·GDN·어텐션·ew/argmax 커널 로드(공용).
    fn open_ctx() -> Result<CudaCtx, String> {
        let image = Self::exl3_fatbin_bytes()?;
        let image_n = Self::exl3_norm_fatbin_bytes()?;
        let image_g = Self::exl3_gemm2_fatbin_bytes()?;
        let image_d = Self::exl3_gdn_fatbin_bytes()?;
        let image_a = Self::exl3_attn_fatbin_bytes()?;
        let image_e = Self::exl3_ew_fatbin_bytes()?;
        let mut cc = CudaCtx::new()?;
        let _g = cc.guard()?;
        cc.load_fatbin(
            "exl3gemv",
            &image,
            &["exl3_had_in", "exl3_gemv", "exl3_had_out"],
        )?;
        cc.load_fatbin("exl3norm", &image_n, &["exl3_norm_resid"])?;
        cc.load_fatbin("exl3gemm2", &image_g, &["exl3_gemm2", "exl3_gemm2_kseg"])?;
        cc.load_fatbin(
            "exl3gdn",
            &image_d,
            &[
                "exl3_gdn_conv",
                "exl3_gdn_l2perm",
                // exl3_gdn_l2perm_gather은 음성대조 계기(원장 17호) —
                // gdn_chain_host_gather_l2perm 검증 경로만 발사.
                "exl3_gdn_l2perm_gather",
                "exl3_gdn_scan",
                "exl3_gdn_gate",
            ],
        )?;
        cc.load_fatbin(
            "exl3attn",
            &image_a,
            &[
                "exl3_attn_prep",
                // exl3_attn_prep_hostpos은 음성대조 계기(결함 4호 재현,
                // 원장 17호) — attn_chain_host_hostpos 검증 경로만 발사.
                "exl3_attn_prep_hostpos",
                "exl3_attn_fwd3s",
                "exl3_attn_pos_bump",
            ],
        )?;
        cc.load_fatbin("exl3ew", &image_e, &["exl3_ew", "exl3_argmax"])?;
        Ok(cc)
    }

    /// 빈 디코더(커널만 로드) — 검증층 합성 선형 등록 경로.
    pub fn empty() -> Result<Self, String> {
        let cc = Self::open_ctx()?;
        Ok(Self {
            cc,
            lin: HashMap::new(),
            hidden: 0,
            n_layers: 0,
            loaded_layers: 0,
            pos: 0,
            embed: Vec::new(),
            vocab: 0,
            dembed: 0,
            dx: 0,
            dxn: 0,
            dab: 0,
            dnw: 0,
            dzero: 0,
            dah: 0,
            dsb: 0,
            dyb: 0,
            daht: 0,
            dyt: 0,
            dbat: 0,
            gdn: None,
            dqkv: 0,
            dzv: 0,
            dgq: 0,
            dgk: 0,
            dgv: 0,
            dgxn: 0,
            dq2: 0,
            dk2: 0,
            dv2: 0,
            dbg: 0,
            dgo: 0,
            dgate: 0,
            dring: 0,
            dgst: 0,
            dcw: 0,
            dab_c: 0,
            dalog: 0,
            ddtb: 0,
            dnwg: 0,
            dkc: 0,
            dvc: 0,
            dpp: 0,
            attn: None,
            dqnw_a: 0,
            dknw_a: 0,
            dqg_a: 0,
            dkin_a: 0,
            dvin_a: 0,
            dqh_a: 0,
            doutv_a: 0,
            dewg: 0,
            dewu: 0,
            dew: 0,
            dlgmax: 0,
            mtp_norms: Vec::new(),
            mtp_kv_k: Vec::new(),
            mtp_kv_v: Vec::new(),
            mtp_kv_len: 0,
            dmtpin: 0,
            dmtpk: 0,
            dmtpv: 0,
            dmtpp: 0,
            dbg_layers: false,
            dbg_hcurve: false,
            hcurve: Vec::new(),
            gexec: None,
            dargmax: 0,
            kmax: 0,
            nmax: 0,
            x_cap: 0,
            norm_cap: 0,
            norm_w_rows: 0,
            gemm_ah_cap: 0,
            gemm_y_cap: 0,
            gemm_bat_n: 0,
            gdn_t_cap: 0,
            attn_t_cap: 0,
            ew_cap: 0,
            lg_cap: 0,
        })
    }

    /// 대형 pageable h2d는 페이지 미매핑 사례(hip 원장 2026-10-04) 방지 —
    /// 4MB 청크 분할(Exl3HipDecoder::h2d_chunked 미러).
    pub(crate) fn h2d_chunked(cc: &CudaCtx, dst: CUdeviceptr, src: &[u8]) -> Result<(), String> {
        const CH: usize = 4 << 20;
        for off in (0..src.len()).step_by(CH) {
            let end = (off + CH).min(src.len());
            cc.h2d(dst + off as u64, &src[off..end])?;
        }
        Ok(())
    }

    /// 원시 선형 3중 등록(합성·실가중 공용 업로드 경로 — 상주는 모듈층 소유).
    /// suh: f16 LE [k], svh: f16 LE [n], tre: u16 LE [kt][nt][16K].
    pub fn add_linear_bytes(
        &mut self,
        key: &str,
        k: usize,
        n: usize,
        krate: u32,
        suh: &[u8],
        tre: &[u8],
        svh: &[u8],
    ) -> Result<(), String> {
        if self.lin.contains_key(key) {
            return Err(format!("{key}: 중복 등록"));
        }
        if k == 0 || !k.is_multiple_of(128) || n == 0 || !n.is_multiple_of(128) {
            return Err(format!("{key}: k={k} n={n} — 128 배수 계약 위반"));
        }
        if !(1..=6).contains(&krate) {
            return Err(format!(
                "{key}: krate={krate} — 커널 스테이징 상한(stg[8*48])은 K≤6"
            ));
        }
        let expect = (k / 16) * (n / 16) * 16 * krate as usize * 2;
        if tre.len() != expect {
            return Err(format!("{key}: trellis {}B != {expect}B", tre.len()));
        }
        if suh.len() != k * 2 || svh.len() != n * 2 {
            return Err(format!(
                "{key}: suh {}B != {}B · svh {}B != {}B",
                suh.len(),
                k * 2,
                svh.len(),
                n * 2
            ));
        }
        let dsuh = self.cc.alloc(suh.len())?;
        let dtre = self.cc.alloc(tre.len())?;
        let dsvh = self.cc.alloc(svh.len())?;
        Self::h2d_chunked(&self.cc, dsuh, suh)?;
        Self::h2d_chunked(&self.cc, dtre, tre)?;
        Self::h2d_chunked(&self.cc, dsvh, svh)?;
        self.lin.insert(
            key.to_string(),
            CudaLin {
                k,
                n,
                krate,
                suh: dsuh,
                tre: dtre,
                svh: dsvh,
            },
        );
        Ok(())
    }

    /// 아카이브에서 선형 1개 적재(형상은 trellis 엔트리에서 산출).
    /// 반환: (k, n, krate, suh, tre, svh) — add_linear_bytes로 업로드.
    fn load_linear_from_archive(
        &self,
        ar: &StArchive,
        key: &str,
    ) -> Result<(usize, usize, u32, Vec<u8>, Vec<u8>, Vec<u8>), String> {
        let shape = ar
            .shape_of(&format!("{key}.trellis"))
            .ok_or_else(|| format!("{key}.trellis 없음"))?
            .to_vec();
        if shape.len() != 3 {
            return Err(format!("{key}: trellis dim {}", shape.len()));
        }
        let (kt, nt, tw) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
        // 반정수 bpw(16K+8) 미지원 — vk 체커와 동일 규약.
        if tw % 16 != 0 {
            return Err(format!("{key}: 반정수 bpw(tw={tw}) 미지원"));
        }
        let (k, n, krate) = (kt * 16, nt * 16, (tw / 16) as u32);
        let tre = ar.read(&format!("{key}.trellis"))?;
        let suh = ar.read(&format!("{key}.suh"))?;
        let svh = ar.read(&format!("{key}.svh"))?;
        Ok((k, n, krate, suh, tre, svh))
    }

    /// 지정 키만 적재(검증 사다리·단일 선형 프로브용 — VRAM 예산 제어).
    pub fn load_keys(dir: &str, keys: &[&str]) -> Result<Self, String> {
        let mut d = Self::empty()?;
        let ar = StArchive::open(Path::new(dir))?;
        for key in keys {
            let (k, n, krate, suh, tre, svh) = d.load_linear_from_archive(&ar, key)?;
            d.add_linear_bytes(key, k, n, krate, &suh, &tre, &svh)?;
        }
        Ok(d)
    }

    /// 가중치 상주 업로드 — 기존 2인자 API는 검증/서버 호환용 1024 KV 상한.
    /// 컨텍스트가 1024를 넘는 서빙은 load_with_ctx를 사용한다(plans/128 P0).
    pub fn load(dir: &str, lim_layers: usize) -> Result<Self, String> {
        Self::load_with_ctx(dir, lim_layers, crate::rawcuda::attn_cuda::ATTN_KV_CAP)
    }

    /// plans/cuda-port.md S5: 선형은 lim_layers 접두만 업로드하되, set_gdn /
    /// set_attn은 모델 전층 배열로 색인하므로 일반 텐서 메타·상수는 모두 조립.
    /// 0 또는 ≥n_layers는 기존대로 모든 선형(본체+mtp)을 적재한다.
    /// kvcap=0은 hip 어댑터와 같은 기본 4096, 그 밖은 64..32768 클램프.
    pub fn load_with_ctx(dir: &str, lim_layers: usize, kvcap: usize) -> Result<Self, String> {
        let cfg = std::fs::read_to_string(format!("{dir}/config.json"))
            .map_err(|e| format!("config.json: {e}"))?;
        let v = JParser {
            b: cfg.as_bytes(),
            p: 0,
        }
        .parse()?;
        let tc = v.get("text_config").unwrap_or(&v);
        let hidden = tc
            .get("hidden_size")
            .and_then(JVal::as_f64)
            .ok_or("config.json: hidden_size 없음")? as usize;
        let n_layers = tc
            .get("num_hidden_layers")
            .and_then(JVal::as_f64)
            .ok_or("config.json: num_hidden_layers 없음")? as usize;
        // 현재 모듈 체인 인덱스는 il%4==3 / il/4에 고정되어 있다.
        let interval = tc
            .get("full_attention_interval")
            .and_then(JVal::as_f64)
            .unwrap_or(4.0);
        if n_layers == 0 || !n_layers.is_multiple_of(4) || interval != 4.0 {
            return Err(format!(
                "EXL3 CUDA: n_layers={n_layers}, attention interval={interval} — 4층 주기 필요"
            ));
        }
        let ar = StArchive::open(Path::new(dir))?;
        let need: Vec<String> = if lim_layers == 0 || lim_layers >= n_layers {
            ar.linear_base_keys()
        } else {
            // hip load의 층별 키 전개 미러(어텐션 il%4==3 / 나머지 GDN).
            let mut v2 = Vec::new();
            for il in 0..lim_layers {
                let lp = format!("model.language_model.layers.{il}");
                if il % 4 == 3 {
                    for nm in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                        v2.push(format!("{lp}.self_attn.{nm}"));
                    }
                } else {
                    for nm in ["in_proj_qkv", "in_proj_z", "out_proj"] {
                        v2.push(format!("{lp}.linear_attn.{nm}"));
                    }
                }
                for nm in ["gate_proj", "up_proj", "down_proj"] {
                    v2.push(format!("{lp}.mlp.{nm}"));
                }
            }
            v2.push("lm_head".to_string());
            v2
        };
        let mut d = Self::empty()?;
        d.hidden = hidden;
        d.n_layers = n_layers;
        // 0·상한초과도 전체 적재 — 런타임 반복 상한과 레지스트리 일치.
        d.loaded_layers = if lim_layers == 0 {
            n_layers
        } else {
            lim_layers.min(n_layers)
        };
        for key in &need {
            let (k, n, krate, suh, tre, svh) = d.load_linear_from_archive(&ar, key)?;
            d.add_linear_bytes(key, k, n, krate, &suh, &tre, &svh)?;
        }
        d.load_plain_tensors(&ar, &cfg, kvcap)?;
        Ok(d)
    }

    /// S5 원장: embedding은 24GB 선형 상주와 중복 VRAM 상주 금지.
    /// BF16 norm(w−1)만 +1하고 GDN A_log·dt_bias·conv·in_proj는 원값 유지
    /// (rawvk/exl3/{resident,gdn,frame}.rs 색인/편향 규약 미러).
    fn load_plain_tensors(
        &mut self,
        ar: &StArchive,
        cfg: &str,
        kvcap: usize,
    ) -> Result<(), String> {
        let key = "model.language_model.embed_tokens.weight";
        let shape = ar
            .shape_of(key)
            .ok_or_else(|| format!("텐서 없음: {key}"))?;
        if shape.len() != 2 || shape[0] == 0 || shape[1] != self.hidden as u64 {
            return Err(format!(
                "{key}: shape {shape:?} != [vocab, {}]",
                self.hidden
            ));
        }
        self.vocab = usize::try_from(shape[0]).map_err(|_| format!("{key}: vocab 범위 초과"))?;
        self.embed = ar.read_f32(key, &[self.vocab, self.hidden])?;

        let mut nw = Vec::with_capacity((2 * self.n_layers + 1) * self.hidden);
        for il in 0..self.n_layers {
            let lp = format!("model.language_model.layers.{il}");
            for name in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
                let k = format!("{lp}.{name}");
                let mut w = ar.read_f32(&k, &[self.hidden])?;
                if ar.dtype_of(&k) == Some(2) {
                    for v in &mut w {
                        *v += 1.0;
                    }
                }
                nw.extend(w);
            }
        }
        let k = "model.language_model.norm.weight";
        let mut w = ar.read_f32(k, &[self.hidden])?;
        if ar.dtype_of(k) == Some(2) {
            for v in &mut w {
                *v += 1.0;
            }
        }
        nw.extend(w);
        let mut nwb = Vec::with_capacity(nw.len() * 4);
        for f in nw {
            nwb.extend_from_slice(&f.to_le_bytes());
        }
        self.set_norm_weights(&nwb, 2 * self.n_layers + 1)?;

        let gd = GdnDims::from_config(cfg)?;
        if gd.hidden != self.hidden || gd.n_gdn != self.n_layers - self.n_layers / 4 {
            return Err("GDN: config 형상/층수 불일치".into());
        }
        let mut cw = Vec::with_capacity(gd.n_gdn * gd.conv_ch() * 4);
        let mut ab = Vec::with_capacity(gd.n_gdn * 2 * gd.h_v * self.hidden);
        let mut alog = Vec::with_capacity(gd.n_gdn * gd.h_v);
        let mut dtb = Vec::with_capacity(gd.n_gdn * gd.h_v);
        let mut gnw = Vec::with_capacity(gd.n_gdn * gd.d);
        for il in 0..self.n_layers {
            if il % 4 == 3 {
                continue;
            }
            let lp = format!("model.language_model.layers.{il}.linear_attn");
            cw.extend(ar.read_f32(&format!("{lp}.conv1d.weight"), &[gd.conv_ch(), 1, 4])?);
            for nm in ["in_proj_a.weight", "in_proj_b.weight"] {
                ab.extend(ar.read_f32(&format!("{lp}.{nm}"), &[gd.h_v, self.hidden])?);
            }
            alog.extend(ar.read_f32(&format!("{lp}.A_log"), &[gd.h_v])?);
            dtb.extend(ar.read_f32(&format!("{lp}.dt_bias"), &[gd.h_v])?);
            gnw.extend(ar.read_f32(&format!("{lp}.norm.weight"), &[gd.d])?);
        }
        self.set_gdn(gd, &cw, &ab, &alog, &dtb, &gnw)?;

        let mut ad = AttnDims::from_config(cfg)?;
        if ad.n_attn != self.n_layers / 4 {
            return Err("attn: config 층수 불일치".into());
        }
        ad.cap = if kvcap == 0 {
            4096
        } else {
            kvcap.clamp(64, 32768)
        };
        let mut qnw = Vec::with_capacity(ad.n_attn * ad.d);
        let mut knw = Vec::with_capacity(ad.n_attn * ad.d);
        for ai in 0..ad.n_attn {
            let lp = format!("model.language_model.layers.{}.self_attn", ai * 4 + 3);
            for (name, out) in [("q_norm.weight", &mut qnw), ("k_norm.weight", &mut knw)] {
                let k = format!("{lp}.{name}");
                let mut w = ar.read_f32(&k, &[ad.d])?;
                if ar.dtype_of(&k) == Some(2) {
                    for v in &mut w {
                        *v += 1.0;
                    }
                }
                out.extend(w);
            }
        }
        self.set_attn(ad, &qnw, &knw)
    }

    /// 디바이스 명(프로브 보고용).
    pub fn device_name(&self) -> &str {
        &self.cc.device_name
    }

    /// 등록 선형 형상(프로브 보고용 — 결함 1호: 형상은 키로 조회).
    pub fn lin_shape(&self, key: &str) -> Option<(usize, usize, u32)> {
        self.lin.get(key).map(|l| (l.k, l.n, l.krate))
    }

    /// 소유 포인터 사본(대여 회피 — hip clone_shallow 미러).
    pub(crate) fn lin_copy(&self, key: &str) -> Result<CudaLin, String> {
        let l = self
            .lin
            .get(key)
            .ok_or_else(|| format!("선형 없음: {key}"))?;
        Ok(CudaLin {
            k: l.k,
            n: l.n,
            krate: l.krate,
            suh: l.suh,
            tre: l.tre,
            svh: l.svh,
        })
    }

    /// 잔차 스트림 x 버퍼 보장(GEMV 스테이징·노름 공유 — 확장 시에만 재할당).
    pub(crate) fn ensure_x(&mut self, elems: usize) -> Result<(), String> {
        if elems > self.x_cap {
            if self.dx != 0 {
                self.cc.free(self.dx)?;
            }
            self.dx = self.cc.alloc(elems * 4)?;
            self.x_cap = elems;
        }
        Ok(())
    }

    /// 등록 선형의 원시 3중 판독(검증층 오라클 소스 — d2h 사본).
    pub fn readback_linear(
        &self,
        key: &str,
    ) -> Result<(usize, usize, u32, Vec<u8>, Vec<u8>, Vec<u8>), String> {
        let l = self
            .lin
            .get(key)
            .ok_or_else(|| format!("선형 없음: {key}"))?;
        let mut suh = vec![0u8; l.k * 2];
        let mut tre = vec![0u8; (l.k / 16) * (l.n / 16) * 16 * l.krate as usize * 2];
        let mut svh = vec![0u8; l.n * 2];
        self.cc.d2h(&mut suh, l.suh)?;
        self.cc.d2h(&mut tre, l.tre)?;
        self.cc.d2h(&mut svh, l.svh)?;
        self.cc.sync()?;
        Ok((l.k, l.n, l.krate, suh, tre, svh))
    }

    /// 토큰 임베딩 → 로짓 1스텝(plans/cuda-port.md S5 순차 디코드).
    pub fn forward_tok(&mut self, tok: u32) -> Result<Vec<f32>, String> {
        let row = self.embed_row_host(tok);
        if row.len() != self.hidden {
            return Err(format!("exl3-cuda: 임베딩 토큰 {tok} 범위 밖 또는 미적재"));
        }
        self.forward(&row).map(|(logits, _)| logits)
    }

    /// 임베딩은 RAM 상주: 토큰에 해당하는 20KB 행만 복사한다(S5).
    /// 범위 밖이면 빈 행을 반환해 forward_tok이 상태 변경 전에 거부한다.
    pub fn embed_row_host(&mut self, tok: u32) -> Vec<f32> {
        if tok as usize >= self.vocab {
            return Vec::new();
        }
        let start = tok as usize * self.hidden;
        self.embed[start..start + self.hidden].to_vec()
    }

    /// 1스텝 greedy: argmax 스캔 폭은 로짓 전체 길이(결함 8호).
    /// forward와 argmax가 같은 컨텍스트 스코프를 써야 하므로 가드를
    /// 여기로 올린다(슬롯 스레드에 전파되지 않는 current 컨텍스트 —
    /// plans/cuda-port.md S5). 중첩 가드는 재진입 가능(prev=자신).
    pub fn step_tok(&mut self, tok: u32) -> Result<u32, String> {
        let _g = self.cc.guard()?;
        let logits = self.forward_tok(tok)?;
        self.argmax_host(&logits)
    }

    /// MTP 드래프트(호스트 경로, §4.13 참고) — G4+.
    pub fn mtp_draft(&mut self, _token: u32, _h_in: &[f32], _pos: u32) -> Result<Vec<f32>, String> {
        Err("TODO(plans/124 G4+): mtp_draft 미구현".into())
    }

    /// MTP 드래프트(디바이스 체인 — 호스트 왕복 없음이 목표, §4.13) — G4+.
    pub fn mtp_draft_gpu(&mut self, _token: u32, _h_in: &[f32], _pos: u32) -> Result<u32, String> {
        Err("TODO(plans/124 G4+): mtp_draft_gpu 미구현".into())
    }

    /// 배치 프리필(gemm2/kseg 경로 — h 대조 프로브 병행 의무, §7.1) — G3+.
    pub fn forward_batch(
        &mut self,
        _rows: &[Vec<f32>],
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>), String> {
        Err("TODO(plans/124 G3+): forward_batch 미구현".into())
    }

    /// 배치 그래프 캡처(판독 배리어·원시 복사 계약 — §4.12·§4.16) — G3+.
    pub fn capture_batch(&mut self, _t: usize) -> Result<(), String> {
        Err("TODO(plans/124 G3+): capture_batch 미구현".into())
    }

    /// 캡처 그래프 재생(캡처 T와 rows 길이 일치 계약) — G3+.
    pub fn replay_batch(
        &mut self,
        _rows: &[Vec<f32>],
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>), String> {
        Err("TODO(plans/124 G3+): replay_batch 미구현".into())
    }

    /// 배치 + MTP 드래프트(종료 시점: 최종 잔차 t행 — §3.4) — G4+.
    pub fn forward_batch_with_mtp(
        &mut self,
        _rows: &[Vec<f32>],
        _toks: &[u32],
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>), String> {
        Err("TODO(plans/124 G4+): forward_batch_with_mtp 미구현".into())
    }

    /// 임베딩 행 1개 → (로짓, 마지막 노름 이전 잔차) — plans/cuda-port.md S5.
    pub fn forward(&mut self, embed_row: &[f32]) -> Result<(Vec<f32>, Vec<f32>), String> {
        self.forward_host_staged(embed_row)
    }
}
