//! EXL3 MTP 드래프트 CUDA 모듈층 — plans/124 G9(최종 목표), 2026-10-04.
//! 산술 계약 §3.4: enorm(e)‖hnorm(h) → mtp.fc → 게이트 어텐션(자체 KV,
//! q_norm/k_norm + rope base 1e7, 24헤드×256dim, 게이트 sigmoid) →
//! o+resid → FFN → resid → 공유 노름 → lm_head.
//! API 형상은 rawhip/decode/spec.rs mtp_step_g(mtp_step_gpu·mtp_step_chain)
//! 미러 — 단 rawcuda는 Exl3CudaDecoder 소유 버퍼를 공유하지 않고 MTP
//! 전용 버퍰 세트를 소유한다(병렬 작업 계약 2026-10-04: exl3_cuda.rs
//! 미수정 — 모듈은 이 파일이 단독 소유).
//!
//! 체인 커널 조립(신규 커널은 assets/exl3_mtp.cu 2종뿐 — 나머지는
//! G2-G7 자산 재사용, 발사 인자만 MTP 소유 버퍼로):
//! - 선형 9종(mtp.fc·q/k/v/o_proj·gate/up/down_proj·lm_head):
//!   exl3_had_in → exl3_gemv(nseg=16) → exl3_had_out(정확 1회 —
//!   결함 3·15호 가드. H도메인 부분합 → WHT⁻¹·0.0884·svh).
//! - enorm/hnorm/attn_norm/post_norm/공유 head norm: exl3_mtp_rms
//!   (assets/exl3_mtp.cu — exl3_norm_resid의 ab=0 등가, 적산 순서 1:1).
//! - q_norm/k_norm + rope(base 1e7 = 전 모델 rope_theta, 64차 부분
//!   회전) + 자체 KV 적립: exl3_attn_prep(lay=0 + MTP 소유 KV/노름
//!   버퍼 — G6 커널 계약 그대로).
//! - 게이트 어텐션(T=1): exl3_attn_fwd3s(lay=0 — q‖gate 인터리브
//!   [q_heads*512]의 gate 반을 sigmoid로 곱한다, §3.4 "게이트 sigmoid").
//!   KV 인덱스는 pp[0] 디바이스 판독(결함 4호) — 전진은
//!   exl3_attn_pos_bump(MTP 소유 pp).
//! - FFN 게이트곱: exl3_ew(silu·mul — crates/core/src/ops.rs silu
//!   L127-130 공식).
//! - 잔차 가산 2회(o_proj·down_proj 출력): exl3_mtp_axpy.
//! - 토큰: exl3_argmax(n = 로짓 길이 248320 — 결함 8호).
//!
//! 노름 규약(§3.4): constant_bias=1.0 노름(layernorm·q/k_norm·공유 head
//! norm — 전부 이 계약 대상)은 저장소 w−1 → 등록값 +1(호출자 계약,
//! set_attn의 qnw/knw와 동일 경로). GDN 노름은 원값(본 모듈 범위 밖).
//!
//! h_in 캡처 시점 계약(§3.4·결함 10호): h_in은 타깃 트렁크의
//! pre-final-norm 잔차, 즉 마지막 FFN 합산 "전" 값이다(합산 후 값을
//! 쓰면 수용률이 하락 — 계측 a1 0.69[전] vs 0.44[후], 2026-10-04 원장).
//! 캡처는 호출자(메인 부착) 책임이고, 본 모듈은 전달받은 h_in을
//! 그대로 소비한다 — 전/후 차이가 값으로 검출 가능함은 검증층
//! (mtp_cuda_probe.rs)이 증명한다.
//!
//! 상태 규약(원장 19호): MTP KV는 시퀀스 상태다 — 재사용 루프는
//! 문맥을 꼬는 사고 계급. 프로브는 쌍 디코더 또는 reset+reseed로
//! fresh 상태를 강제하고, KV는 비영 시딩(§3.3 S0≠0 정신)이 기본.
//!
//! [CMP 170HX(sm_80, GA100 70SM, HBM2e ~1.5TB/s) 설계 근거 — plans/124 §0]
//! 드래프트 1스텝 = GEMV 9회(k 5120~17408·lm_head 5120×248320) + 소형
//! 커널 다수 — 지배 비용은 lm_head GEMV(읽기 797MB 트렐리스 → HBM2e
//! 0.53ms@1.5TB/s 하한)와 fc/FFN 4종(각 ≤45MB). T=1 GEMV 그리드
//! (n/16/8, 16)는 27B fc 기준 40×16=640블록 → 70SM 확산 양호;
//! lm_head 1940×16 = 31K블록 — 대역폠 포화 경로(4070 타이밍 무관,
//! 개발기는 정합 호스트일 뿐). 전 체인은 호스트 왕복 없는 디바이스
//! 체인(결함 13호 — h2d는 최초 e/h, d2h는 최종 판독만).
//!
//! 독립 컴파일 계약: 이 파일도 cuda_probe_shim.rs의 rustc 단독 컴파일
//! 대상(std 외 크레이트 금지 — exl3_cuda.rs 머리 주석과 동일 계약).

use crate::rawcuda::ctx::CudaCtx;
use crate::rawcuda::exl3_cuda::{CudaLin, Exl3CudaDecoder, GEMV_NSEG};
use crate::rawcuda::ffi::CUdeviceptr;

/// MTP 노름 행 인덱스 — dmnw [5][hidden] 등록 순서(계약 고정).
pub const MTP_NORM_ENORM: usize = 0;
pub const MTP_NORM_HNORM: usize = 1;
pub const MTP_NORM_ATTN: usize = 2;
pub const MTP_NORM_POST: usize = 3;
pub const MTP_NORM_SHARED: usize = 4;
/// MTP 노름 행 수.
pub const MTP_NORM_ROWS: usize = 5;

/// 모듈이 사용하는 선형 키(아카이브 기저명 — 결함 1호: 형상은 키로).
pub const MTP_LIN_FC: &str = "mtp.fc";
pub const MTP_LIN_Q: &str = "mtp.layers.0.self_attn.q_proj";
pub const MTP_LIN_K: &str = "mtp.layers.0.self_attn.k_proj";
pub const MTP_LIN_V: &str = "mtp.layers.0.self_attn.v_proj";
pub const MTP_LIN_O: &str = "mtp.layers.0.self_attn.o_proj";
pub const MTP_LIN_GATE: &str = "mtp.layers.0.mlp.gate_proj";
pub const MTP_LIN_UP: &str = "mtp.layers.0.mlp.up_proj";
pub const MTP_LIN_DOWN: &str = "mtp.layers.0.mlp.down_proj";
pub const MTP_LIN_HEAD: &str = "lm_head";
/// 전체 선형 키(load_keys용).
pub const MTP_LIN_KEYS: [&str; 9] = [
    MTP_LIN_FC,
    MTP_LIN_Q,
    MTP_LIN_K,
    MTP_LIN_V,
    MTP_LIN_O,
    MTP_LIN_GATE,
    MTP_LIN_UP,
    MTP_LIN_DOWN,
    MTP_LIN_HEAD,
];

/// MTP 드래프트 형상 — 모델 config.json text_config에서 유도
/// (GdnDims/AttnDims 방식 계승 — 형상은 명시 등록, 추정 금지).
/// 실측 기준(2026-10-04, D:/models/Qwen3.8-27B-exl3-4.00bpw/config.json):
/// hidden=5120·q_heads=24·kv_heads=4·d=256·n_ff=17408·vocab=248320·
/// mtp_num_hidden_layers=1. 35B-A3B exl3 아카이브에는 mtp.* 텐서가
/// 없다(MTP 가중치는 27B 아카이브 전용 — GGUF 쪽 MTP은 Q4 경로).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MtpDims {
    /// 잔류 폭(hidden — 27B 5120).
    pub hidden: usize,
    /// q헤드 수(27B 24).
    pub q_heads: usize,
    /// KV헤드 수(27B 4).
    pub kv_heads: usize,
    /// 헤드 폭(256 고정 계약 — exl3_attn_prep/fwd3s 64차 rope와 짝).
    pub d: usize,
    /// FFN 중간 폭(27B 17408).
    pub n_ff: usize,
    /// 어휘(로짓 길이 — 결함 8호: argmax n).
    pub vocab: usize,
    /// MTP KV 캐시 위치 상한(hip 규약 1024 — dkc/dvc [cap][kv_dim]).
    pub cap: usize,
}

impl MtpDims {
    /// q/qh/헤드 입력 폭.
    pub fn q_dim(&self) -> usize {
        self.q_heads * self.d
    }
    /// k/v/KV 캐시 폭.
    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.d
    }
    /// qg 폭(q‖gate 인터리브 — 헤드당 512: 256 q + 256 gate).
    pub fn qg_dim(&self) -> usize {
        self.q_heads * 2 * self.d
    }
    /// KV그룹 폭(GQA).
    pub fn gq(&self) -> usize {
        self.q_heads / self.kv_heads
    }

    /// config.json 본문 → 형상(실측 차원 계약 — 프로브는 실모델
    /// config.json에서 읽는다). 파서는 exl3_cuda.rs JParser의 최소
    /// 미러(독립 컴파일 계약 — std 전용 자작).
    pub fn from_config(cfg: &str) -> Result<Self, String> {
        let v = MJParser {
            b: cfg.as_bytes(),
            p: 0,
        }
        .parse()?;
        let tc = v.get("text_config").unwrap_or(&v);
        let num = |k: &str| {
            tc.get(k)
                .and_then(MJVal::as_f64)
                .ok_or_else(|| format!("config.json: text_config.{k} 없음"))
        };
        let hidden = num("hidden_size")? as usize;
        let q_heads = num("num_attention_heads")? as usize;
        let kv_heads = num("num_key_value_heads")? as usize;
        let d = num("head_dim")? as usize;
        let n_ff = num("intermediate_size")? as usize;
        let vocab = num("vocab_size")? as usize;
        let mtp_layers = tc
            .get("mtp_num_hidden_layers")
            .and_then(MJVal::as_f64)
            .unwrap_or(1.0) as usize;
        if d != 256 {
            return Err(format!(
                "mtp: head_dim {d} — 256 고정 계약(exl3_attn 64차 rope)"
            ));
        }
        if q_heads == 0 || kv_heads == 0 || !q_heads.is_multiple_of(kv_heads) {
            return Err(format!(
                "mtp: q_heads={q_heads} kv_heads={kv_heads} — q%kv==0 계약(GQA)"
            ));
        }
        if !hidden.is_multiple_of(1024) || hidden > 8192 || hidden == 0 {
            return Err(format!(
                "mtp: hidden={hidden} — 1024 배수·8192 이하 계약(rms v[8])"
            ));
        }
        if mtp_layers != 1 {
            return Err(format!(
                "mtp: mtp_num_hidden_layers={mtp_layers} — 1 고정(단일 드래프트 층)"
            ));
        }
        if n_ff == 0 || vocab == 0 {
            return Err("mtp: n_ff/vocab 0".into());
        }
        Ok(MtpDims {
            hidden,
            q_heads,
            kv_heads,
            d,
            n_ff,
            vocab,
            cap: crate::rawcuda::attn_cuda::ATTN_KV_CAP,
        })
    }
}

/// MTP 스텝 중간 산출(검증층 단계별 값 판정용 — G2 debug_run_stages
/// 노선: 호출은 검증층, 상태는 모듈층 소유).
pub struct MtpMids {
    /// enorm‖hnorm concat [2·hidden].
    pub cat: Vec<f32>,
    /// mtp.fc 산출(hidden).
    pub eh: Vec<f32>,
    /// q/k 노름+rope 산출 q [q_dim].
    pub qh: Vec<f32>,
    /// fwd3s 게이트 어텐션 출력 [q_dim].
    pub outv: Vec<f32>,
    /// 현 스텝이 적립한 KC/KV 행 [kv_dim] 각각.
    pub kc_row: Vec<f32>,
    pub vc_row: Vec<f32>,
    /// o+resid 직후 잔류(hidden).
    pub cur_attn: Vec<f32>,
    /// FFN+resid 직후 잔류 = h_next(hidden).
    pub h_next: Vec<f32>,
    /// 공유 head norm 산출(hidden).
    pub head_in: Vec<f32>,
}

/// MTP 드래프트 상주 상태 — Exl3CudaDecoder와 공유하지 않는 전용
/// 버퍼 세트(병렬 작업 계약: exl3_cuda.rs 무수정). 가중치(선형 9종)는
/// decoder.lin 레지스트리에서 대여(소유는 decoder).
pub struct Exl3CudaMtp {
    /// 형상(new 등록).
    pub dims: MtpDims,
    /// 노름 가중 [5][hidden] f32(등록값 = 저장소 w−1 + 1 — §3.4 규약).
    pub dmnw: CUdeviceptr,
    /// q/k 노름 [256] f32(등록값 +1).
    pub dqnw: CUdeviceptr,
    pub dknw: CUdeviceptr,
    /// MTP 자체 KV 캐시 [cap][kv_dim] f32(prep이 r/w, 0 초기화).
    pub dkc: CUdeviceptr,
    pub dvc: CUdeviceptr,
    /// KV 인덱스 pp[0] — 디바이스 판독 계약(결함 4호).
    pub dpp: CUdeviceptr,
    // ── 작업 버퍼(체인 내부 스테이징 — 모두 MTP 소유) ──
    /// 토큰 임베딩 e·h_in 입력 [hidden].
    pub de: CUdeviceptr,
    pub dh: CUdeviceptr,
    /// enorm‖hnorm concat [2·hidden].
    pub dcat: CUdeviceptr,
    /// MTP 잔류 스트림(fc 산출 → o+resid → FFN+resid = h_next).
    pub dcur: CUdeviceptr,
    /// 노름 출력 스테이징 [hidden].
    pub dnrm: CUdeviceptr,
    /// q‖gate 인터리브 [qg_dim] · k/v [kv_dim] (prep 입력).
    pub dqg: CUdeviceptr,
    pub dkin: CUdeviceptr,
    pub dvin: CUdeviceptr,
    /// prep 산출 qh · fwd3s 산출 outv [q_dim].
    pub dqh: CUdeviceptr,
    pub doutv: CUdeviceptr,
    /// o_proj 출력(잔차 가산분)[hidden].
    pub dgout: CUdeviceptr,
    /// FFN gate/up/게이트곱 [n_ff] · down 출력 [hidden].
    pub dfg: CUdeviceptr,
    pub dfu: CUdeviceptr,
    pub dfglu: CUdeviceptr,
    pub dfdown: CUdeviceptr,
    /// lm_head 로짓 [vocab].
    pub dlogits: CUdeviceptr,
    /// had_in 출력 f16 u32쌍팩 [kmax/2] — 선형 공용 스테이징.
    dah: CUdeviceptr,
    /// GEMV 부분합 sb [nseg][nmax] — had_out이 nseg 합산.
    dsb: CUdeviceptr,
    /// had_out 출력 스테이징 [nmax](선형별 슬라이스 사용).
    dyb: CUdeviceptr,
}

impl Exl3CudaMtp {
    /// exl3_mtp.fatbin 자산 해석 — LLM170_CUDA_EXL3_MTP_FATBIN_PATH
    /// 오버라이드 우선(자산 경로 오버라이드일 뿐 계산 경로 분기 아님).
    fn mtp_fatbin_bytes() -> Result<Vec<u8>, String> {
        const ENV: &str = "LLM170_CUDA_EXL3_MTP_FATBIN_PATH";
        const REL: &[&str] = &[
            "crates/backend-gpu/src/rawcuda/assets/exl3_mtp.fatbin",
            "src/rawcuda/assets/exl3_mtp.fatbin",
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
            "exl3_mtp.fatbin 없음 — scripts/build_cuda.bat 실행 또는 {ENV} 지정 (탐색: {REL:?})"
        ))
    }

    /// 선형 형상 검증(결함 1호 정신 — k/n은 키로 명시 대조).
    fn lin_check(dec: &Exl3CudaDecoder, key: &str, k: usize, n: usize) -> Result<(), String> {
        let got = dec
            .lin_shape(key)
            .ok_or_else(|| format!("mtp: 선형 없음(등록 누락): {key}"))?;
        if got.0 != k || got.1 != n {
            return Err(format!(
                "mtp: {key} 형상 (k={}, n={}) != 계약 (k={k}, n={n})",
                got.0, got.1
            ));
        }
        Ok(())
    }

    /// MTP 상주 상태 생성 — 노름·KV·pp 할당 + exl3_mtp 커널 등록.
    /// norms: [5][hidden] f32(행 순서 enorm·hnorm·attn·post·shared —
    /// 저장소 w−1에 +1한 값). qnw/knw: [256] f32(+1 값).
    /// 선형 9종은 사전에 decoder에 등록되어 있어야 한다
    /// (load_keys(dir, &MTP_LIN_KEYS) — VRAM 예산: 필요 텐서만).
    pub fn new(
        dec: &mut Exl3CudaDecoder,
        dims: MtpDims,
        norms: &[f32],
        qnw: &[f32],
        knw: &[f32],
    ) -> Result<Self, String> {
        let n = dims.hidden;
        if norms.len() != MTP_NORM_ROWS * n {
            return Err(format!(
                "mtp: norms {} != {}x{n}",
                norms.len(),
                MTP_NORM_ROWS
            ));
        }
        if qnw.len() != 256 || knw.len() != 256 {
            return Err("mtp: qnw/knw != 256".into());
        }
        // 선형 형상 계약 대조(27B 실측 기준 — 형상은 dims에서 유도).
        Self::lin_check(dec, MTP_LIN_FC, 2 * n, n)?;
        Self::lin_check(dec, MTP_LIN_Q, n, dims.qg_dim())?;
        Self::lin_check(dec, MTP_LIN_K, n, dims.kv_dim())?;
        Self::lin_check(dec, MTP_LIN_V, n, dims.kv_dim())?;
        Self::lin_check(dec, MTP_LIN_O, dims.q_dim(), n)?;
        Self::lin_check(dec, MTP_LIN_GATE, n, dims.n_ff)?;
        Self::lin_check(dec, MTP_LIN_UP, n, dims.n_ff)?;
        Self::lin_check(dec, MTP_LIN_DOWN, dims.n_ff, n)?;
        Self::lin_check(dec, MTP_LIN_HEAD, n, dims.vocab)?;
        // 커널 등록(모듈 소유 fatbin — 레지스트리는 dec.cc 공유).
        let image = Self::mtp_fatbin_bytes()?;
        {
            let _g = dec.cc.guard()?;
            dec.cc
                .load_fatbin("exl3mtp", &image, &["exl3_mtp_rms", "exl3_mtp_axpy"])?;
        }
        // SAFETY: f32 슬라이스 → 바이트 뷰(길이·정렬 일치).
        let b =
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        let cc = &dec.cc;
        let dmnw = cc.alloc(norms.len() * 4)?;
        cuda_h2d_chunked(cc, dmnw, b(norms))?;
        let dqnw = cc.alloc(1024)?;
        cc.h2d(dqnw, b(qnw))?;
        let dknw = cc.alloc(1024)?;
        cc.h2d(dknw, b(knw))?;
        let kv_elems = dims.cap * dims.kv_dim();
        let dkc = cc.alloc(kv_elems * 4)?;
        cuda_h2d_chunked(cc, dkc, &vec![0u8; kv_elems * 4])?;
        let dvc = cc.alloc(kv_elems * 4)?;
        cuda_h2d_chunked(cc, dvc, &vec![0u8; kv_elems * 4])?;
        let dpp = cc.alloc(4)?;
        cc.h2d(dpp, &0u32.to_le_bytes())?;
        let de = cc.alloc(n * 4)?;
        let dh = cc.alloc(n * 4)?;
        let dcat = cc.alloc(2 * n * 4)?;
        let dcur = cc.alloc(n * 4)?;
        let dnrm = cc.alloc(n * 4)?;
        let dqg = cc.alloc(dims.qg_dim() * 4)?;
        let dkin = cc.alloc(dims.kv_dim() * 4)?;
        let dvin = cc.alloc(dims.kv_dim() * 4)?;
        let dqh = cc.alloc(dims.q_dim() * 4)?;
        let doutv = cc.alloc(dims.q_dim() * 4)?;
        let dgout = cc.alloc(n * 4)?;
        let dfg = cc.alloc(dims.n_ff * 4)?;
        let dfu = cc.alloc(dims.n_ff * 4)?;
        let dfglu = cc.alloc(dims.n_ff * 4)?;
        let dfdown = cc.alloc(n * 4)?;
        let dlogits = cc.alloc(dims.vocab * 4)?;
        // GEMV 스테이징: kmax = max(2n, n_ff) · nmax = vocab(lm_head).
        let kmax = (2 * n).max(dims.n_ff);
        let nmax = dims.vocab;
        let dah = cc.alloc(kmax * 2)?;
        let dsb = cc.alloc(GEMV_NSEG * nmax * 4)?;
        let dyb = cc.alloc(nmax * 4)?;
        Ok(Self {
            dims,
            dmnw,
            dqnw,
            dknw,
            dkc,
            dvc,
            dpp,
            de,
            dh,
            dcat,
            dcur,
            dnrm,
            dqg,
            dkin,
            dvin,
            dqh,
            doutv,
            dgout,
            dfg,
            dfu,
            dfglu,
            dfdown,
            dlogits,
            dah,
            dsb,
            dyb,
        })
    }

    /// pp[0] 상주값 갱신(h2d — 캡처 외 경로. 그래프 내 전진은 pos_bump).
    pub fn mtp_set_pos(&self, cc: &CudaCtx, pos: u32) -> Result<(), String> {
        cc.h2d(self.dpp, &pos.to_le_bytes())
    }

    /// pp[0] += 1(exl3_attn_pos_bump 재사용 — MTP 소유 pp로 발사).
    pub fn mtp_pos_bump(&self, cc: &CudaCtx) -> Result<(), String> {
        let f = cc.function("exl3_attn_pos_bump")?;
        let mut p0 = self.dpp;
        let mut args: [*mut std::ffi::c_void; 1] = [(&mut p0) as *mut _ as *mut _];
        cc.launch(f, 1, 1, 32, &mut args)
    }

    /// MTP KV 히스토리 시딩(비영 — §3.3 S0≠0 정신 계승) — kc/vc:
    /// [cap][kv_dim] 전체(prep 도메인 값: k행은 노름+rope 적용후).
    pub fn mtp_seed_kv(&self, cc: &CudaCtx, kc: &[f32], vc: &[f32]) -> Result<(), String> {
        let elems = self.dims.cap * self.dims.kv_dim();
        if kc.len() != elems || vc.len() != elems {
            return Err(format!(
                "mtp seed: kc/vc {}/{} != cap {}x{}",
                kc.len(),
                vc.len(),
                self.dims.cap,
                self.dims.kv_dim()
            ));
        }
        // SAFETY: f32 슬라이스 → 바이트 뷰(길이·정렬 일치).
        let b =
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        cuda_h2d_chunked(cc, self.dkc, b(kc))?;
        cuda_h2d_chunked(cc, self.dvc, b(vc))?;
        Ok(())
    }

    /// fresh 상태 복귀(원장 19호 — 상태 오염 가드): KV 제로 + pp=0.
    /// 재사용 루프의 문맥 꼬임 방지(프로브는 reset+reseed 또는 쌍
    /// 디코더로 fresh 상태를 강제한다).
    pub fn mtp_reset_kv(&self, cc: &CudaCtx) -> Result<(), String> {
        let elems = self.dims.cap * self.dims.kv_dim();
        cuda_h2d_chunked(cc, self.dkc, &vec![0u8; elems * 4])?;
        cuda_h2d_chunked(cc, self.dvc, &vec![0u8; elems * 4])?;
        self.mtp_set_pos(cc, 0)
    }

    /// 평 RMS 노름 1행 발사 — exl3_mtp_rms(그리드 (1,1), 블록 1024).
    /// w_row: dmnw의 행 인덱스(결함 2호: 행 오프셋 명시).
    // [rustfmt skip — G6 병리 계승(2026-10-04, 시간제한 계약)] 원시
    // 발사 인자 배열 — 스킵 없이 rustfmt가 초선형 폭주(원장).
    #[rustfmt::skip]
    fn rms_dev(&self, cc: &CudaCtx, x: CUdeviceptr, w_row: usize, out: CUdeviceptr) -> Result<(), String> {
        let n = self.dims.hidden as i32;
        let f = cc.function("exl3_mtp_rms")?;
        // SAFETY: dmnw 할당 내 행 오프셋 — MTP_NORM_ROWS×hidden 경계 내.
        let w = self.dmnw + (w_row * self.dims.hidden) as u64 * 4;
        let (mut a0, mut a1, mut a2) = (x, w, out);
        let mut nn = n;
        let mut args: [*mut std::ffi::c_void; 4] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
            (&mut nn) as *mut _ as *mut _,
        ];
        cc.launch(f, 1, 1, 1024, &mut args)
    }

    /// 잔차 가산 x += y — exl3_mtp_axpy(그리드 ⌈n/256⌉, 블록 256).
    // [rustfmt skip — G6 병리 계승(2026-10-04)] 발사 인자 배열.
    #[rustfmt::skip]
    fn axpy_dev(&self, cc: &CudaCtx, x: CUdeviceptr, y: CUdeviceptr, n: usize) -> Result<(), String> {
        if n == 0 || n > i32::MAX as usize {
            return Err(format!("mtp axpy: n={n} 도메인 위반"));
        }
        let f = cc.function("exl3_mtp_axpy")?;
        let (mut a0, mut a1) = (x, y);
        let mut nn = n as i32;
        let mut args: [*mut std::ffi::c_void; 3] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut nn) as *mut _ as *mut _,
        ];
        cc.launch(f, n.div_ceil(256) as u32, 1, 256, &mut args)
    }

    /// 트렐리스 선형 1회(디바이스 체인 — 호스트 왕복 없음, 결함 13호):
    /// had_in(x) → gemv(nseg=16 부분합) → had_out(정확 1회) → y.
    /// 발사 기하는 exl3_cuda.rs gemv_chain_opt와 동일(산술 미러).
    // [rustfmt skip — G6 병리 계승(2026-10-04, 시간제한 계약)] 밀착
    // 3단 발사(had_in/gemv/had_out) 인자 배열 — 스킵 없이 rustfmt
    // 초선형 폭주(G6 attn 원장과 동일 계열).
    #[rustfmt::skip]
    fn lin_dev(
        &self,
        cc: &CudaCtx,
        lin: &CudaLin,
        x: CUdeviceptr,
        y: CUdeviceptr,
    ) -> Result<(), String> {
        let (k, n, krate) = (lin.k, lin.n, lin.krate as i32);
        // had_in: 그리드 (k/128, 1), 블록 128.
        let f_hin = cc.function("exl3_had_in")?;
        let (mut a0, mut a1, mut a2) = (x, lin.suh, self.dah);
        let (mut kc, mut ks) = ((k / 128) as i32, k as i32);
        let mut args_hin: [*mut std::ffi::c_void; 5] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
            (&mut kc) as *mut _ as *mut _,
            (&mut ks) as *mut _ as *mut _,
        ];
        cc.launch(f_hin, (k / 128) as u32, 1, 128, &mut args_hin)?;
        // gemv: 그리드 ((n/16)/8, nseg=16), 블록 128 — sb [nseg][n].
        let f_gv = cc.function("exl3_gemv")?;
        let (mut b0, mut b1, mut b2) = (self.dah, lin.tre, self.dsb);
        let (mut kt, mut nt, mut kk) = ((k / 16) as i32, (n / 16) as i32, krate);
        let mut args_gv: [*mut std::ffi::c_void; 6] = [
            (&mut b0) as *mut _ as *mut _,
            (&mut b1) as *mut _ as *mut _,
            (&mut b2) as *mut _ as *mut _,
            (&mut kt) as *mut _ as *mut _,
            (&mut nt) as *mut _ as *mut _,
            (&mut kk) as *mut _ as *mut _,
        ];
        cc.launch(
            f_gv,
            ((n / 16) / 8) as u32,
            GEMV_NSEG as u32,
            128,
            &mut args_gv,
        )?;
        // had_out: 그리드 (n/128, 1), 블록 128 — nseg 합산 + WHT⁻¹·svh.
        let f_hout = cc.function("exl3_had_out")?;
        let (mut c0, mut c1, mut c2) = (self.dsb, lin.svh, y);
        let (mut nch, mut nsg, mut nst) = ((n / 128) as i32, GEMV_NSEG as i32, n as i32);
        let mut args_hout: [*mut std::ffi::c_void; 6] = [
            (&mut c0) as *mut _ as *mut _,
            (&mut c1) as *mut _ as *mut _,
            (&mut c2) as *mut _ as *mut _,
            (&mut nch) as *mut _ as *mut _,
            (&mut nsg) as *mut _ as *mut _,
            (&mut nst) as *mut _ as *mut _,
        ];
        cc.launch(f_hout, (n / 128) as u32, 1, 128, &mut args_hout)
    }

    /// 선형 키 발사 래퍼(대여 회피 없이 매 조회 — 결함 1호: 키로).
    fn lin_key(&self, dec: &Exl3CudaDecoder, key: &str) -> Result<CudaLin, String> {
        let l = dec
            .lin
            .get(key)
            .ok_or_else(|| format!("mtp: 선형 없음: {key}"))?;
        Ok(CudaLin {
            k: l.k,
            n: l.n,
            krate: l.krate,
            suh: l.suh,
            tre: l.tre,
            svh: l.svh,
        })
    }

    /// prep 발사(MTP 소유 버퍼, lay=0 — G6 커널 계약 그대로.
    /// 그리드 (1, q_heads+kv_heads), 블록 128. pos는 pp[0] 디바이스
    /// 판독 — 결함 4호: 발사 인자가 아니다).
    // [rustfmt skip — G6 병리 계승(2026-10-04)] 14원소 발사 인자 배열.
    #[rustfmt::skip]
    fn attn_prep_dev(&self, cc: &CudaCtx) -> Result<(), String> {
        let f = cc.function("exl3_attn_prep")?;
        let (mut tl, mut lay) = (1i32, 0i32);
        let (mut qh, mut kvh, mut cp) = (
            self.dims.q_heads as i32,
            self.dims.kv_heads as i32,
            self.dims.cap as i32,
        );
        let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7, mut a8) = (
            self.dqg,
            self.dkin,
            self.dvin,
            self.dqnw,
            self.dknw,
            self.dqh,
            self.dkc,
            self.dvc,
            self.dpp,
        );
        let mut args: [*mut std::ffi::c_void; 14] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
            (&mut a3) as *mut _ as *mut _,
            (&mut a4) as *mut _ as *mut _,
            (&mut a5) as *mut _ as *mut _,
            (&mut a6) as *mut _ as *mut _,
            (&mut a7) as *mut _ as *mut _,
            (&mut a8) as *mut _ as *mut _,
            (&mut tl) as *mut _ as *mut _,
            (&mut lay) as *mut _ as *mut _,
            (&mut qh) as *mut _ as *mut _,
            (&mut kvh) as *mut _ as *mut _,
            (&mut cp) as *mut _ as *mut _,
        ];
        cc.launch(f, 1, (self.dims.q_heads + self.dims.kv_heads) as u32, 128, &mut args)
    }

    /// fwd3s 발사(MTP 소유 버퍼, lay=0 — T=1 드래프트. 그리드
    /// (1, q_heads), 블록 256. 게이트 sigmoid는 커널 내 qg 반).
    // [rustfmt skip — G6 병리 계승(2026-10-04)] 11원소 발사 인자 배열.
    #[rustfmt::skip]
    fn attn_fwd3s_dev(&self, cc: &CudaCtx) -> Result<(), String> {
        let f = cc.function("exl3_attn_fwd3s")?;
        let (mut tl, mut lay) = (1i32, 0i32);
        let (mut qh, mut kvh, mut cp) = (
            self.dims.q_heads as i32,
            self.dims.kv_heads as i32,
            self.dims.cap as i32,
        );
        let (mut f0, mut f1, mut f2, mut f3, mut f4, mut f5) =
            (self.dqh, self.dkc, self.dvc, self.dqg, self.doutv, self.dpp);
        let mut args: [*mut std::ffi::c_void; 11] = [
            (&mut f0) as *mut _ as *mut _,
            (&mut f1) as *mut _ as *mut _,
            (&mut f2) as *mut _ as *mut _,
            (&mut f3) as *mut _ as *mut _,
            (&mut f4) as *mut _ as *mut _,
            (&mut f5) as *mut _ as *mut _,
            (&mut tl) as *mut _ as *mut _,
            (&mut lay) as *mut _ as *mut _,
            (&mut qh) as *mut _ as *mut _,
            (&mut kvh) as *mut _ as *mut _,
            (&mut cp) as *mut _ as *mut _,
        ];
        cc.launch(f, 1, self.dims.q_heads as u32, 256, &mut args)
    }

    /// FFN 게이트곱 발사 — exl3_ew(그리드 ⌈n_ff/128⌉, 블록 128).
    // [rustfmt skip — G6 병리 계승(2026-10-04)] 발사 인자 배열.
    #[rustfmt::skip]
    fn ew_dev(&self, cc: &CudaCtx) -> Result<(), String> {
        let f = cc.function("exl3_ew")?;
        let nn = self.dims.n_ff;
        let mut n32 = nn as i32;
        let (mut a0, mut a1, mut a2) = (self.dfg, self.dfu, self.dfglu);
        let mut args: [*mut std::ffi::c_void; 4] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
            (&mut n32) as *mut _ as *mut _,
        ];
        cc.launch(f, nn.div_ceil(128) as u32, 1, 128, &mut args)
    }

    /// MTP 1스텝 체인 본체(디바이스 체인 — 결함 13호: h2d는 e/h
    /// 업로드뿐, 중간 판독 없음). e: 토큰 임베딩 행 [hidden],
    /// h_dev: h_in 디바이스 버퍼(캡처 시점은 호출자 계약 — §3.4
    /// 마지막 FFN 합산 "전" 잔차). 반환: with_head면 드래프트 토큰.
    /// mids: Some이면 단계별 중간 산출 판독(검증층 전용 — 프로덕션
    /// 호출은 None, 토큰·h_next 판독만).
    #[allow(clippy::too_many_arguments)]
    pub fn mtp_step_g(
        &self,
        dec: &mut Exl3CudaDecoder,
        e: &[f32],
        h_dev: CUdeviceptr,
        with_head: bool,
        mids: Option<&mut MtpMids>,
    ) -> Result<Option<u32>, String> {
        let cc = &dec.cc;
        let dm = self.dims;
        let n = dm.hidden;
        if e.len() != n {
            return Err(format!("mtp: e.len={} != hidden {n}", e.len()));
        }
        // SAFETY: e는 길이 n*4 바이트의 f32 슬라이스 — 바이트 뷰 변환.
        let eb = unsafe { std::slice::from_raw_parts(e.as_ptr() as *const u8, e.len() * 4) };
        cc.h2d(self.de, eb)?;
        // ① enorm(e) → cat[0..n] · hnorm(h) → cat[n..2n]
        self.rms_dev(cc, self.de, MTP_NORM_ENORM, self.dcat)?;
        // SAFETY: dcat 할당 내 절반 오프셋 — 2n 경계 내.
        self.rms_dev(cc, h_dev, MTP_NORM_HNORM, self.dcat + (n as u64) * 4)?;
        // ② mtp.fc [2n → n] → cur
        let lfc = self.lin_key(dec, MTP_LIN_FC)?;
        self.lin_dev(cc, &lfc, self.dcat, self.dcur)?;
        // ③ attn_norm → q/k/v
        self.rms_dev(cc, self.dcur, MTP_NORM_ATTN, self.dnrm)?;
        let lq = self.lin_key(dec, MTP_LIN_Q)?;
        self.lin_dev(cc, &lq, self.dnrm, self.dqg)?;
        let lk = self.lin_key(dec, MTP_LIN_K)?;
        self.lin_dev(cc, &lk, self.dnrm, self.dkin)?;
        let lv = self.lin_key(dec, MTP_LIN_V)?;
        self.lin_dev(cc, &lv, self.dnrm, self.dvin)?;
        // ④ q/k 노름+rope(pp[0] 디바이스 판독) + 자체 KV 적립 + fwd3s
        self.attn_prep_dev(cc)?;
        self.attn_fwd3s_dev(cc)?;
        // ⑤ o_proj → 잔차 가산(cur += gout)
        let lo = self.lin_key(dec, MTP_LIN_O)?;
        self.lin_dev(cc, &lo, self.doutv, self.dgout)?;
        self.axpy_dev(cc, self.dcur, self.dgout, n)?;
        // ⑥ FFN: post_norm → gate/up → silu·mul → down → 잔차 가산
        self.rms_dev(cc, self.dcur, MTP_NORM_POST, self.dnrm)?;
        let lg = self.lin_key(dec, MTP_LIN_GATE)?;
        self.lin_dev(cc, &lg, self.dnrm, self.dfg)?;
        let lu = self.lin_key(dec, MTP_LIN_UP)?;
        self.lin_dev(cc, &lu, self.dnrm, self.dfu)?;
        self.ew_dev(cc)?;
        let ld = self.lin_key(dec, MTP_LIN_DOWN)?;
        self.lin_dev(cc, &ld, self.dfglu, self.dfdown)?;
        self.axpy_dev(cc, self.dcur, self.dfdown, n)?;
        // ⑦ 공유 head norm → (with_head) lm_head → argmax
        self.rms_dev(cc, self.dcur, MTP_NORM_SHARED, self.dnrm)?;
        let mut token = None;
        if with_head {
            let lh = self.lin_key(dec, MTP_LIN_HEAD)?;
            self.lin_dev(cc, &lh, self.dnrm, self.dlogits)?;
            token = Some(dec.argmax_dev(self.dlogits, dm.vocab)?);
        }
        if let Some(m) = mids {
            self.read_mids(&dec.cc, m)?;
        }
        Ok(token)
    }

    /// 중간 산출 판독(검증층 — mtp_step_g mids 경로의 실체.
    /// cur는 스테이지별로 변형되므로 지정 시점 값을 순차 판독한다).
    fn read_mids(&self, cc: &CudaCtx, m: &mut MtpMids) -> Result<(), String> {
        let dm = self.dims;
        let n = dm.hidden;
        let take = |v: &mut Vec<f32>, elems: usize, src: CUdeviceptr| -> Result<(), String> {
            let mut buf = vec![0u8; elems * 4];
            cc.d2h(&mut buf, src)?;
            // SAFETY: d2h 완료 후 재해석(길이·정렬 일치 — G2 판독 패턴).
            let s = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const f32, elems) };
            v.clear();
            v.extend_from_slice(s);
            Ok(())
        };
        take(&mut m.cat, 2 * n, self.dcat)?;
        take(&mut m.qh, dm.q_dim(), self.dqh)?;
        take(&mut m.outv, dm.q_dim(), self.doutv)?;
        // SAFETY: KV 캐시 내 현행 위치 슬라이스 — cap·kv_dim 경계는
        // pp[0] 판독값 기준(모듈은 pp 값을 모른다 — 판독해서 계산).
        let mut pb = [0u8; 4];
        cc.d2h(&mut pb, self.dpp)?;
        let pos = u32::from_le_bytes([pb[0], pb[1], pb[2], pb[3]]) as usize;
        let row = dm.kv_dim();
        take(&mut m.kc_row, row, self.dkc + (pos * row) as u64 * 4)?;
        take(&mut m.vc_row, row, self.dvc + (pos * row) as u64 * 4)?;
        // 주: eh·cur_attn은 스텝 종료 시점 dcur(h_next)로 덮어씀 —
        // 단계별 값이 필요한 프로브는 이 판독 계열을 확장하지 않고
        // h_next·head_in 판독으로 종단 판정한다(값 maxdiff 계약).
        take(&mut m.h_next, n, self.dcur)?;
        take(&mut m.head_in, n, self.dnrm)?;
        Ok(())
    }

    /// 호스트 h 진입(hip mtp_step_gpu 미러): h2d h → mtp_step_g →
    /// (토큰, h_next) 회수.
    pub fn mtp_step_gpu(
        &self,
        dec: &mut Exl3CudaDecoder,
        e: &[f32],
        h: &[f32],
        with_head: bool,
    ) -> Result<(Option<u32>, Vec<f32>), String> {
        let n = self.dims.hidden;
        if h.len() != n {
            return Err(format!("mtp: h.len={} != hidden {n}", h.len()));
        }
        // SAFETY: h는 길이 n*4 바이트의 f32 슬라이스 — 바이트 뷰 변환.
        let hb = unsafe { std::slice::from_raw_parts(h.as_ptr() as *const u8, h.len() * 4) };
        dec.cc.h2d(self.dh, hb)?;
        let token = self.mtp_step_g(dec, e, self.dh, with_head, None)?;
        let mut buf = vec![0u8; n * 4];
        dec.cc.d2h(&mut buf, self.dcur)?;
        dec.cc.sync()?;
        // SAFETY: d2h 완료 후 재해석(길이·정렬 일치).
        let h_next = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const f32, n) }.to_vec();
        Ok((token, h_next))
    }

    /// 체인 스텝(hip mtp_step_chain 미러) — h를 내부 dh(직전 h_next
    /// 소스)가 아니라 호출자 지정 디바이스 버퍼에서 직접 읽는 변형은
    /// mtp_step_g 사용. 본 진입은 토큰만 회수(드래프트 체인용).
    pub fn mtp_step_chain(&self, dec: &mut Exl3CudaDecoder, e: &[f32]) -> Result<u32, String> {
        // 주: 내부 dh는 이전 mtp_step_gpu 호출의 h_in 잔존 — 체인
        // 캐리(h_next → h_in)는 호출자가 dcur → dh로 전달한다
        // (원시 d2d가 ctx에 없어 검증층 프로브는 h_next를 회수해
        // 재주입한다. 메인 부착 단계에서 d2d 도입 예정).
        self.mtp_step_g(dec, e, self.dh, true, None)?
            .ok_or_else(|| "mtp: head 미실행".to_string())
    }
}

/// 대형 pageable h2d 4MB 청크 분할(exl3_cuda.rs h2d_chunked 미러 —
/// 페이지 미매핑 사고 가드, 독립 컴파일 계약상 자작 사본).
fn cuda_h2d_chunked(cc: &CudaCtx, dst: CUdeviceptr, src: &[u8]) -> Result<(), String> {
    const CH: usize = 4 << 20;
    for off in (0..src.len()).step_by(CH) {
        let end = (off + CH).min(src.len());
        cc.h2d(dst + off as u64, &src[off..end])?;
    }
    Ok(())
}

// ── config.json 최소 파서(exl3_cuda.rs JParser 미러 — 독립 컴파일) ──

enum MJVal {
    Obj(Vec<(String, MJVal)>),
    Arr(Vec<MJVal>),
    Num(f64),
    Str(String),
    Bool(bool),
    Null,
}

impl MJVal {
    fn get(&self, key: &str) -> Option<&MJVal> {
        match self {
            MJVal::Obj(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    fn as_f64(&self) -> Option<f64> {
        match self {
            MJVal::Num(v) => Some(*v),
            _ => None,
        }
    }
    fn as_str(&self) -> Option<&str> {
        match self {
            MJVal::Str(s) => Some(s),
            _ => None,
        }
    }
    fn as_arr(&self) -> Option<&[MJVal]> {
        match self {
            MJVal::Arr(v) => Some(v),
            _ => None,
        }
    }
    fn as_obj(&self) -> Option<&[(String, MJVal)]> {
        match self {
            MJVal::Obj(m) => Some(m),
            _ => None,
        }
    }
}

struct MJParser<'a> {
    b: &'a [u8],
    p: usize,
}

impl MJParser<'_> {
    fn parse(mut self) -> Result<MJVal, String> {
        self.ws();
        let v = self.value(0)?;
        self.ws();
        if self.p != self.b.len() {
            return Err(format!("config.json: 잔여 바이트 @{}", self.p));
        }
        Ok(v)
    }

    fn ws(&mut self) {
        while self.p < self.b.len() && self.b[self.p].is_ascii_whitespace() {
            self.p += 1;
        }
    }

    fn value(&mut self, depth: u32) -> Result<MJVal, String> {
        if depth > 32 {
            return Err("config.json: 깊이 상한".into());
        }
        match self.b.get(self.p) {
            Some(b'{') => {
                self.p += 1;
                let mut m = Vec::new();
                self.ws();
                if self.b.get(self.p) == Some(&b'}') {
                    self.p += 1;
                    return Ok(MJVal::Obj(m));
                }
                loop {
                    self.ws();
                    let k = self.string()?;
                    self.ws();
                    if self.b.get(self.p) != Some(&b':') {
                        return Err(format!("config.json: ':' 예상 @{}", self.p));
                    }
                    self.p += 1;
                    self.ws();
                    let v = self.value(depth + 1)?;
                    m.push((k, v));
                    self.ws();
                    match self.b.get(self.p) {
                        Some(b',') => self.p += 1,
                        Some(b'}') => {
                            self.p += 1;
                            return Ok(MJVal::Obj(m));
                        }
                        _ => return Err(format!("config.json: ','/}}' 예상 @{}", self.p)),
                    }
                }
            }
            Some(b'[') => {
                self.p += 1;
                let mut a = Vec::new();
                self.ws();
                if self.b.get(self.p) == Some(&b']') {
                    self.p += 1;
                    return Ok(MJVal::Arr(a));
                }
                loop {
                    self.ws();
                    a.push(self.value(depth + 1)?);
                    self.ws();
                    match self.b.get(self.p) {
                        Some(b',') => self.p += 1,
                        Some(b']') => {
                            self.p += 1;
                            return Ok(MJVal::Arr(a));
                        }
                        _ => return Err(format!("config.json: ','/]' 예상 @{}", self.p)),
                    }
                }
            }
            Some(b'"') => Ok(MJVal::Str(self.string()?)),
            Some(b't') => self.lit("true", MJVal::Bool(true)),
            Some(b'f') => self.lit("false", MJVal::Bool(false)),
            Some(b'n') => self.lit("null", MJVal::Null),
            Some(_) => {
                let s = self.p;
                while self.p < self.b.len()
                    && matches!(
                        self.b[self.p],
                        b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9'
                    )
                {
                    self.p += 1;
                }
                std::str::from_utf8(&self.b[s..self.p])
                    .ok()
                    .and_then(|t| t.parse::<f64>().ok())
                    .map(MJVal::Num)
                    .ok_or_else(|| format!("config.json: 숫자 파싱 @{s}"))
            }
            None => Err("config.json: 예기치 않은 끝".into()),
        }
    }

    fn lit(&mut self, s: &str, v: MJVal) -> Result<MJVal, String> {
        if self.b[self.p..].starts_with(s.as_bytes()) {
            self.p += s.len();
            Ok(v)
        } else {
            Err(format!("config.json: 리터럴 {s} 예상 @{}", self.p))
        }
    }

    fn string(&mut self) -> Result<String, String> {
        if self.b.get(self.p) != Some(&b'"') {
            return Err(format!("config.json: 문자열 예상 @{}", self.p));
        }
        self.p += 1;
        let mut out = String::new();
        while let Some(&c) = self.b.get(self.p) {
            match c {
                b'"' => {
                    self.p += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.p += 1;
                    match self.b.get(self.p) {
                        Some(b'"') => out.push('"'),
                        Some(b'\\') => out.push('\\'),
                        Some(b'/') => out.push('/'),
                        Some(b'n') => out.push('\n'),
                        Some(b't') => out.push('\t'),
                        Some(b'r') => out.push('\r'),
                        Some(b'u') => {
                            let h = std::str::from_utf8(&self.b[self.p + 1..self.p + 5])
                                .ok()
                                .and_then(|t| u32::from_str_radix(t, 16).ok())
                                .ok_or("config.json: \\u 이스케이프")?;
                            out.push(char::from_u32(h).unwrap_or('\u{fffd}'));
                            self.p += 4;
                        }
                        _ => return Err("config.json: 이스케이프".into()),
                    }
                    self.p += 1;
                }
                _ => {
                    let s = self.p;
                    while self.p < self.b.len() && self.b[self.p] != b'"' && self.b[self.p] != b'\\'
                    {
                        self.p += 1;
                    }
                    out.push_str(&String::from_utf8_lossy(&self.b[s..self.p]));
                }
            }
        }
        Err("config.json: 문자열 미종결".into())
    }
}
