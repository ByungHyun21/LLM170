//! DeepSeek-V4 DSpark MTP 드래프트 스테이지 모듈층 — plans/130 B4, 2026-10-05.
//!
//! [계약] crates/core/src/deepseek4/frame.rs DSpark 코어 함수들이 산술
//! 원천(유일 기준 — plans/124 §6). 본 모듈은 드래프트 스테이지 전체를
//! 미러한다(보고서 §7 — core frame.rs 가 운영 계약):
//! - dspark_main_x L208-230: main_proj FP8-sim GEMM(활성 12288, 128블록)
//!   → bf16 → main_norm 가중 RMS — 본 모듈 커널(ds4_mtp.fatbin).
//! - dspark_draft_ids L233-239: [token, noise×(block-1)] — 호스트.
//! - dspark_window_warm L242-263: main_x 전 토큰 project_kv(ds4_attn
//!   모듈 API 재사용 — attn.rs project_kv L137-158) → 링 [win×512].
//! - dspark_decode_attn L267-345: 메인 토큰 kv(1토큰 project_kv — rope
//!   행 0 계약) 링 pos%win 기입 · 드래프트 q/kv 로프 위치 pos+1..pos+b
//!   (시프트 로프표로 ds4_attn stage_project_q/kv 재사용 — project_q/kv
//!   산술과 project_kv_unroped L349-371+수동 로프가 동일 비트) ·
//!   [링|드래프트 kv] 스파스 어텐션+싱크(idx 행 = [0..min(win,pos+1)]++
//!   [win..win+b] — 전 드래프트 토큰 동일) → 비회전 rope^-1(pos+1..) →
//!   그룹 출력(ds4_attn stage_output) — 어텐션 내부 커널은 본 모듈
//!   ds4_mtp.fatbin(sparse/derot — ds4_attn.cu 리터럴 재이식).
//! - 드래프트 블록: layers.rs block_forward L28-121 배선(mHC→attn_norm→
//!   어텐션→mHC→ffn_norm→MoE→mHC) — hc/moe 내부는 착지 모듈 API 재사용
//!   (ds4_hc_cuda/ds4_moe_cuda), norm 은 본 모듈 rmsw 커널.
//! - 종결: hc_head(ds4_hc) → norm(본 모듈) → 공유 head 스트립 gemv
//!   (frame.rs head_gemv_stripwise L41-77 — 호스트 trellis 스트립 디양자화
//!   + 본 모듈 head_strip 커널, k블록 부분합 순서) → 마르코프 바이어스
//!     (markov_logits_bias L373-383) → 신뢰도(confidence_score L386-397).
//!
//! [모듈 조립 계약] 블록 내부 산술이 착지 모듈과 일치하는 곳은 전부 그
//! 모듈 API 로 재사용한다(ds4_attn/ds4_hc/ds4_moe — 커널 재구현 금지).
//! ds4_attn 인스턴스는 mtp.0 attn 가중치를 IL_MAIN/IL_SHIFT 이중 등록한다:
//! IL_MAIN(=n_layers) 은 정상 로프표(워밍 t 토큰·메인 토큰 kv 행 0·그룹
//! 출력), IL_SHIFT(=n_layers+1) 은 시프트 로프표(행 i = 절대 위치 pos+1+i
//! — 드래프트 q/kv). 두 슬롯 모두 compress_ratios 꼬리 0(SWA) — 형상
//! 검증에서 확인. 각 모듈 인스턴스는 CudaCtx 를 단독 소유하되 전부 동일
//! 디바이스 프라이머리 컨텍스트(device_primary_ctx_retain — 공존 계약).
//!
//! [정합 목표 — 비트동일] main_x·링·q/kv·스파스/비회전/그룹 출력·mHC·
//! norm·스트립 gemv·마르코프·신뢰도 전부 f32 연산별 반올림 + f64 트윈
//! (exp_cr) — 코어 미러와 비트동일. MoE 구간만 ds4-moe 계급의 라우트
//! 가중치 정밀도를 승계(프루브 게이트 문서화 — 검증층 원장).
//!
//! [가중치 소스] EXL3 Vision-Exp(mtp.{0..2}) — main_proj(mtp.0, trellis)
//! ·main_norm(mtp.0)·hc_{attn,ffn}(mtp.0)·attn_norm/ffn_norm(mtp.0)·
//! 게이트/공유/전문가(mtp.0.ffn — noaux_tc 라우티드, tid2eid 없음)·
//! hc_head/norm/markov_w1/markov_w2/confidence(mtp.2 — loader.rs
//! dspark_head_parts L439-449 텐서명 계약). 전부 오프셋 직독(검증층).
//!
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 금지 —
//! scripts/cuda_probe_shim.rs 단독 컴파일 대상.

use crate::rawcuda::ctx::CudaCtx;
use crate::rawcuda::ds4_attn_cuda::{Ds4AttnCuda, Ds4AttnDims, Ds4LayerF32};
use crate::rawcuda::ds4_hc_cuda::{Ds4HcCuda, Ds4HcDims};
use crate::rawcuda::ds4_moe_cuda::{Ds4MoeCuda, Ds4MoeDims};
use crate::rawcuda::exl3_cuda::{Exl3CudaDecoder, JParser, JVal};
use crate::rawcuda::ffi::CUdeviceptr;

/// DSpark 형상 — config.json DSpark+MoE 서브셋(config.rs dspark_* 대응,
/// 값은 픽스처 config에서 — 추정 금지, 결함 1호 정신).
#[derive(Debug, Clone)]
pub struct Ds4MtpDims {
    pub dim: usize,
    pub vocab: usize,
    pub hc: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub window: usize,
    pub rms_eps: f32,
    /// MoE 서브셋(noaux_tc 라우티드 — mtp.*엔 tid2eid 없음, 실측).
    pub moe: Ds4MoeDims,
    /// num_nextn_predict_layers(픽스처 실측 3).
    pub n_mtp_layers: usize,
    /// dspark_block_size(5).
    pub dspark_block: usize,
    /// dspark_noise_token_id(128799).
    pub dspark_noise_token: u32,
    /// dspark_markov_rank(256).
    pub markov_rank: usize,
}

impl Ds4MtpDims {
    /// config.json 본문 → 형상. 키 누락은 Err(DSpark 키는 전부 필수).
    pub fn from_config(cfg: &str) -> Result<Self, String> {
        let v = JParser {
            b: cfg.as_bytes(),
            p: 0,
        }
        .parse()?;
        let num = |k: &str| -> Result<f64, String> {
            v.get(k)
                .and_then(JVal::as_f64)
                .ok_or_else(|| format!("config.json: {k} 없음"))
        };
        let u = |k: &str| -> Result<usize, String> {
            let x = num(k)?;
            if x < 0.0 || x.fract() != 0.0 {
                return Err(format!("config.json: {k}={x} — 양의 정수 계약"));
            }
            Ok(x as usize)
        };
        let moe = Ds4MoeDims {
            dim: u("hidden_size")?,
            n_routed: u("n_routed_experts")?,
            n_active: u("num_experts_per_tok")?,
            inter: u("moe_intermediate_size")?,
            route_scale: num("routed_scaling_factor")? as f32,
            swiglu_limit: num("swiglu_limit")? as f32,
        };
        let dims = Ds4MtpDims {
            dim: moe.dim,
            vocab: u("vocab_size")?,
            hc: u("hc_mult")?,
            n_heads: u("num_attention_heads")?,
            head_dim: u("head_dim")?,
            rope_head_dim: u("qk_rope_head_dim")?,
            window: u("sliding_window")?,
            rms_eps: num("rms_norm_eps")? as f32,
            moe,
            n_mtp_layers: u("num_nextn_predict_layers")?,
            dspark_block: u("dspark_block_size")?,
            dspark_noise_token: u("dspark_noise_token_id")? as u32,
            markov_rank: u("dspark_markov_rank")?,
        };
        if dims.dspark_block == 0 || dims.dspark_block > 32 {
            return Err(format!(
                "ds4-mtp: dspark_block={} — (0, 32] 도메인(스파스 인덱스 행)",
                dims.dspark_block
            ));
        }
        if dims.markov_rank == 0 || dims.markov_rank > 4096 {
            return Err(format!(
                "ds4-mtp: markov_rank={} — (0, 4096] 도메인",
                dims.markov_rank
            ));
        }
        if dims.dspark_noise_token as usize >= dims.vocab {
            return Err(format!(
                "ds4-mtp: noise={} ≥ vocab={}",
                dims.dspark_noise_token, dims.vocab
            ));
        }
        if !dims.dim.is_multiple_of(128) || !dims.vocab.is_multiple_of(128) {
            return Err(format!(
                "ds4-mtp: dim={} vocab={} — 128 배수 계약",
                dims.dim, dims.vocab
            ));
        }
        Ok(dims)
    }
}

/// Ds4MtpCuda — DSpark 드래프트 스테이지 모듈. 소유 서브모듈(attn/hc/moe)
/// 은 pub 로 노출(가중치 등록·전문가 적재는 호출부 계약 — 프로브 원장).
pub struct Ds4MtpCuda {
    /// 디바이스 컨텍스트(본 모듈 커널 소유 — 서브모듈과 프라이머리 공유).
    pub cc: CudaCtx,
    pub dims: Ds4MtpDims,
    /// 착지 어텐션 모듈 — mtp.0 attn 가중치 IL_MAIN/IL_SHIFT 이중 등록.
    pub attn: Ds4AttnCuda,
    /// 착지 mHC 모듈 — hc_pre/hc_post(IL_MAIN)·hc_head(HEAD 키).
    pub hc: Ds4HcCuda,
    /// 착지 MoE 모듈 — noaux_tc 라우티드(전문가 상주는 호출부).
    pub moe: Ds4MoeCuda,
    /// mtp.0 attn 정상 로프표 등록 슬롯(=n_layers).
    pub il_main: usize,
    /// 시프트 로프표 등록 슬롯(=n_layers+1 — 드래프트 q/kv).
    pub il_shift: usize,
    // ── 상주 가중치(본 모듈 커널 — 0=미등록) ──
    d_proj: CUdeviceptr,       // main_proj f32 k-major [3d][d]
    d_main_norm: CUdeviceptr,  // [d]
    d_attn_norm: CUdeviceptr,  // [d]
    d_ffn_norm: CUdeviceptr,   // [d]
    d_final_norm: CUdeviceptr, // [d]
    d_w1: CUdeviceptr,         // markov_w1 f32 [vocab][rank]
    d_w2: CUdeviceptr,         // markov_w2 f32 k-major [rank][vocab]
    d_conf_proj: CUdeviceptr,  // confidence proj [d+rank]
    d_sink: CUdeviceptr,       // 어텐션 싱크 z' [n_heads](mtp.0.attn.attn_sink)
    // ── 호스트 로프표 사본(시프트표/derot 소스) ──
    rope_cs: Vec<f32>,
    rope_half: usize,
    // ── 작업 버퍼(grow-only) ──
    dx3: CUdeviceptr,    // [t][3d] main_hidden 스테이징(fp8 제자리)
    d_gout: CUdeviceptr, // [rows][d] gemm/norm 입출력 공용
    d_rows: CUdeviceptr, // [rows][d] norm 출력
    dq: CUdeviceptr,     // [b][nh·hd] 드래프트 q(로프 완료)
    dkv: CUdeviceptr,    // [(win+b)][hd] 링|드래프트 kv
    didx: CUdeviceptr,   // [b][win+b] 스파스 인덱스
    do_: CUdeviceptr,    // [b][nh·hd] 어텐션 출력(비회전 완료)
    drope: CUdeviceptr,  // [b][half·2] 시프트 로프표(derot)
    dheadw: CUdeviceptr, // [d][128] 헤드 스트립 가중
    dheadx: CUdeviceptr, // [b][d] 헤드 입력 h
    dheady: CUdeviceptr, // [b][128] 스트립 로짓
    dbias: CUdeviceptr,  // [b][vocab] 마르코프 바이어스
    dconf: CUdeviceptr,  // [b] 신뢰도 출력
    cap_rows: usize,
    cap_b: usize,
}

/// f32 슬라이스 → LE 바이트(모듈 h2d 규격).
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

/// i32 슬라이스 → LE 바이트.
fn i32_bytes(v: &[i32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

/// u32 슬라이스 → LE 바이트.
fn u32_bytes(v: &[u32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

impl Ds4MtpCuda {
    /// ds4_mtp.fatbin 자산 해석 — LLM170_CUDA_DS4_MTP_FATBIN_PATH 오버라이드
    /// 우선(자산 경로 오버라이드일 뿐 계산 경로 분기 아님).
    fn fatbin_bytes() -> Result<Vec<u8>, String> {
        const ENV: &str = "LLM170_CUDA_DS4_MTP_FATBIN_PATH";
        const REL: &[&str] = &[
            "crates/backend-gpu/src/rawcuda/assets/ds4_mtp.fatbin",
            "src/rawcuda/assets/ds4_mtp.fatbin",
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
            "ds4_mtp.fatbin 없음 — scripts/build_cuda.bat 실행 또는 {ENV} 지정 (탐색: {REL:?})"
        ))
    }

    /// 개방 — config.json 본문으로 형상 확정 + 서브모듈 + ds4_mtp.fatbin.
    /// IL_MAIN/IL_SHIFT 슬롯의 compress_ratios 가 0(SWA)임을 검증(추정 금지).
    pub fn open(cfg: &str) -> Result<Self, String> {
        let dims = Ds4MtpDims::from_config(cfg)?;
        let attn_dims = Ds4AttnDims::from_config(cfg)?;
        let hc_dims = Ds4HcDims::from_config(cfg)?;
        let n_layers = attn_dims.n_layers;
        let (il_main, il_shift) = (n_layers, n_layers + 1);
        if attn_dims.compress_ratios.len() <= il_shift {
            return Err(format!(
                "ds4-mtp: compress_ratios 길이 {} ≤ il_shift {il_shift} — MTP 꼬리 필요",
                attn_dims.compress_ratios.len()
            ));
        }
        for il in [il_main, il_shift] {
            if attn_dims.compress_ratios[il] != 0 {
                return Err(format!(
                    "ds4-mtp: compress_ratios[{il}]={} — DSpark 슬롯은 0(SWA) 계약",
                    attn_dims.compress_ratios[il]
                ));
            }
        }
        let image = Self::fatbin_bytes()?;
        let mut cc = CudaCtx::new()?;
        {
            let _g = cc.guard()?;
            cc.load_fatbin(
                "ds4_mtp",
                &image,
                &[
                    "llm170_ds4m_fp8_rows",
                    "llm170_ds4m_gemm",
                    "llm170_ds4m_bf16_rows",
                    "llm170_ds4m_rmsw_rows",
                    "llm170_ds4m_sparse_attn",
                    "llm170_ds4m_derot",
                    "llm170_ds4m_head_strip",
                    "llm170_ds4m_markov",
                    "llm170_ds4m_conf",
                ],
            )?;
        }
        let attn = Ds4AttnCuda::new(attn_dims)?;
        let hc = Ds4HcCuda::new(hc_dims)?;
        let moe = Ds4MoeCuda::new(dims.moe.clone())?;
        Ok(Ds4MtpCuda {
            cc,
            dims,
            attn,
            hc,
            moe,
            il_main,
            il_shift,
            d_proj: 0,
            d_main_norm: 0,
            d_attn_norm: 0,
            d_ffn_norm: 0,
            d_final_norm: 0,
            d_w1: 0,
            d_w2: 0,
            d_conf_proj: 0,
            d_sink: 0,
            rope_cs: Vec::new(),
            rope_half: 0,
            dx3: 0,
            d_gout: 0,
            d_rows: 0,
            dq: 0,
            dkv: 0,
            didx: 0,
            do_: 0,
            drope: 0,
            dheadw: 0,
            dheadx: 0,
            dheady: 0,
            dbias: 0,
            dconf: 0,
            cap_rows: 0,
            cap_b: 0,
        })
    }

    /// 디바이스 이름(프로브 보고용).
    pub fn device_name(&self) -> &str {
        &self.cc.device_name
    }

    // ── 등록 API(형상 전부 검증 — 형상 추정 금지, 결함 1호) ──

    /// 상주 슬롯 교체(기존 있으면 해제 후 재할당·업로드 — 대형은 청크).
    fn replace_w(
        name: &str,
        cc: &CudaCtx,
        slot: &mut CUdeviceptr,
        bytes: &[u8],
    ) -> Result<(), String> {
        if *slot != 0 {
            // SAFETY: 이전 alloc 산출물 — 교체 시 1회 해제.
            cc.free(*slot)?;
        }
        let p = cc.alloc(bytes.len())?;
        if bytes.len() > (4 << 20) {
            Exl3CudaDecoder::h2d_chunked(cc, p, bytes)?;
        } else {
            cc.h2d(p, bytes)?;
        }
        *slot = p;
        let _ = name;
        Ok(())
    }

    /// 드래프트 층 어텐션 가중치 등록 — mtp.0.attn(Ds4LayerF32, SWA —
    /// comp/idx 불요). IL_MAIN/IL_SHIFT 이중 등록(로프표만 상이).
    pub fn register_draft_layer(&mut self, w: &Ds4LayerF32) -> Result<(), String> {
        self.attn.register_layer(self.il_main, w)?;
        self.attn.register_layer(self.il_shift, w)
    }

    /// 정상 로프표 등록 — cs: [len][half][2] f32(호스트 RopeTable::build
    /// 산출물, 윈도우 계열 base). 길이는 워밍 t + 여유 확보(호출부 계약).
    /// 시프트표는 dspark_decode_attn 이 본 사본에서 행 단위 절출해 구축.
    pub fn set_rope_main(&mut self, cs: &[f32], half: usize) -> Result<(), String> {
        if half == 0 || !cs.len().is_multiple_of(half * 2) {
            return Err(format!(
                "ds4-mtp: rope 표 {} × half {half} 정합 오류",
                cs.len()
            ));
        }
        self.rope_cs = cs.to_vec();
        self.rope_half = half;
        let len = cs.len() / (half * 2);
        self.attn.set_rope(self.il_main, cs, half)?;
        let _ = len;
        Ok(())
    }

    /// 드래프트 층 어텐션 싱크 등록 — mtp.0.attn.attn_sink F32 [n_heads].
    /// (ds4_attn 모듈이 sink 포인터를 노출하지 않으므로 본 모듈에 상주
    /// — 스파스 어텐션 커널 인자.)
    pub fn register_sink(&mut self, sink: &[f32]) -> Result<(), String> {
        if sink.len() != self.dims.n_heads {
            return Err(format!(
                "ds4-mtp: sink {} != n_heads {}",
                sink.len(),
                self.dims.n_heads
            ));
        }
        let _g = self.cc.guard()?;
        Self::replace_w("sink", &self.cc, &mut self.d_sink, &f32_bytes(sink))
    }

    /// mHC 파라미터 등록 — mtp.0 hc_{attn,ffn}_{fn,base,scale}(F32 plain).
    pub fn register_hc(
        &mut self,
        attn_fn: &[f32],
        attn_base: &[f32],
        attn_scale: &[f32],
        ffn_fn: &[f32],
        ffn_base: &[f32],
        ffn_scale: &[f32],
    ) -> Result<(), String> {
        self.hc
            .register(self.il_main, "attn", attn_fn, attn_base, attn_scale)?;
        self.hc
            .register(self.il_main, "ffn", ffn_fn, ffn_base, ffn_scale)
    }

    /// 종결 hc_head 등록 — mtp.{last}.hc_head_{fn,base,scale}.
    pub fn register_hc_head(
        &mut self,
        fns: &[f32],
        base: &[f32],
        scale: &[f32],
    ) -> Result<(), String> {
        self.hc.register(
            crate::rawcuda::ds4_hc_cuda::DS4_HC_HEAD_IL,
            "head",
            fns,
            base,
            scale,
        )
    }

    /// 노름 가중 등록 — main_norm·attn_norm·ffn_norm(mtp.0)·종결 norm
    /// (mtp.{last}). 전부 [d].
    pub fn register_norms(
        &mut self,
        main_norm: &[f32],
        attn_norm: &[f32],
        ffn_norm: &[f32],
        final_norm: &[f32],
    ) -> Result<(), String> {
        let d = self.dims.dim;
        for (nm, w) in [
            ("main_norm", main_norm),
            ("attn_norm", attn_norm),
            ("ffn_norm", ffn_norm),
            ("final_norm", final_norm),
        ] {
            if w.len() != d {
                return Err(format!("ds4-mtp: {nm} {} != dim {d}", w.len()));
            }
        }
        let _g = self.cc.guard()?;
        Self::replace_w(
            "main_norm",
            &self.cc,
            &mut self.d_main_norm,
            &f32_bytes(main_norm),
        )?;
        Self::replace_w(
            "attn_norm",
            &self.cc,
            &mut self.d_attn_norm,
            &f32_bytes(attn_norm),
        )?;
        Self::replace_w(
            "ffn_norm",
            &self.cc,
            &mut self.d_ffn_norm,
            &f32_bytes(ffn_norm),
        )?;
        Self::replace_w(
            "final_norm",
            &self.cc,
            &mut self.d_final_norm,
            &f32_bytes(final_norm),
        )?;
        Ok(())
    }

    /// main_proj 등록 — f32 k-major [3d][d](trellis 디양자화값, mtp.0).
    pub fn register_main_proj(&mut self, w: &[f32]) -> Result<(), String> {
        let d = self.dims.dim;
        if w.len() != 3 * d * d {
            return Err(format!("ds4-mtp: main_proj {} != 3·{d}·{d}", w.len()));
        }
        let _g = self.cc.guard()?;
        Self::replace_w("main_proj", &self.cc, &mut self.d_proj, &f32_bytes(w))
    }

    /// 마르코크/신뢰도 등록 — w1 [vocab][rank]·w2 k-major [rank][vocab]
    /// (loader.rs plain_kmat 전치값)·confidence proj [d+rank](mtp.{last}).
    pub fn register_markov(&mut self, w1: &[f32], w2: &[f32], conf: &[f32]) -> Result<(), String> {
        let (d, rank, vocab) = (self.dims.dim, self.dims.markov_rank, self.dims.vocab);
        if w1.len() != vocab * rank {
            return Err(format!("ds4-mtp: markov_w1 {} != {vocab}·{rank}", w1.len()));
        }
        if w2.len() != rank * vocab {
            return Err(format!("ds4-mtp: markov_w2 {} != {rank}·{vocab}", w2.len()));
        }
        if conf.len() != d + rank {
            return Err(format!("ds4-mtp: conf {} != {d}+{rank}", conf.len()));
        }
        let _g = self.cc.guard()?;
        Self::replace_w("markov_w1", &self.cc, &mut self.d_w1, &f32_bytes(w1))?;
        Self::replace_w("markov_w2", &self.cc, &mut self.d_w2, &f32_bytes(w2))?;
        Self::replace_w(
            "conf_proj",
            &self.cc,
            &mut self.d_conf_proj,
            &f32_bytes(conf),
        )?;
        Ok(())
    }

    /// 행 버퍼 보장([cap_rows][d] — main_x t 행·드래프트 b 행 공용).
    fn ensure_rows(&mut self, rows: usize) -> Result<(), String> {
        if rows <= self.cap_rows {
            return Ok(());
        }
        let _g = self.cc.guard()?;
        if self.cap_rows > 0 {
            for p in [self.dx3, self.d_gout, self.d_rows] {
                if p != 0 {
                    self.cc.free(p)?;
                }
            }
        }
        let d = self.dims.dim;
        self.dx3 = self.cc.alloc(rows * 3 * d * 4)?;
        self.d_gout = self.cc.alloc(rows * d * 4)?;
        self.d_rows = self.cc.alloc(rows * d * 4)?;
        self.cap_rows = rows;
        Ok(())
    }

    /// 드래프트 버퍼 보장([cap_b] 축 — q/idx/kv/o/rope/head/bias/conf).
    fn ensure_b(&mut self, b: usize) -> Result<(), String> {
        if b <= self.cap_b {
            return Ok(());
        }
        let dm = &self.dims;
        let (nh, hd, d, vocab, rank) = (dm.n_heads, dm.head_dim, dm.dim, dm.vocab, dm.markov_rank);
        let _g = self.cc.guard()?;
        if self.cap_b > 0 {
            for p in [
                self.dq,
                self.dkv,
                self.didx,
                self.do_,
                self.drope,
                self.dheadw,
                self.dheadx,
                self.dheady,
                self.dbias,
                self.dconf,
            ] {
                if p != 0 {
                    self.cc.free(p)?;
                }
            }
        }
        let rows_kv = dm.window + b;
        self.dq = self.cc.alloc(b * nh * hd * 4)?;
        self.dkv = self.cc.alloc(rows_kv * hd * 4)?;
        self.didx = self.cc.alloc(b * rows_kv * 4)?;
        self.do_ = self.cc.alloc(b * nh * hd * 4)?;
        self.drope = self.cc.alloc(b * dm.rope_head_dim * 4)?;
        self.dheadw = self.cc.alloc(d * 128 * 4)?;
        self.dheadx = self.cc.alloc(b * d * 4)?;
        self.dheady = self.cc.alloc(b * 128 * 4)?;
        self.dbias = self.cc.alloc(b * vocab * 4)?;
        // confidence 출력 [b] + 마르코프 임베딩 스테이징 [b][rank]
        // (confidence()가 dconf+b·4 에 b·rank·4 바이트를 h2d 한다 —
        //  기존 rank·4 는 1토큰분이라 다 토큰 스테이징에서 오버플로).
        self.dconf = self.cc.alloc(b * 4 + b * rank * 4)?;
        self.cap_b = b;
        Ok(())
    }

    // ── 커널 런치 헬퍼 ──

    fn k_fp8(
        &self,
        buf: CUdeviceptr,
        rows: usize,
        stride: usize,
        cols: usize,
        block: usize,
    ) -> Result<(), String> {
        let nblk = cols.div_ceil(block);
        let total = (rows * nblk) as u32;
        let (mut a0, mut a1) = (buf, rows as i32);
        let (mut a2, mut a3, mut a4) = (stride as i32, cols as i32, block as i32);
        let mut args: [*mut std::ffi::c_void; 5] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
            (&mut a3) as *mut _ as *mut _,
            (&mut a4) as *mut _ as *mut _,
        ];
        let f = self.cc.function("llm170_ds4m_fp8_rows")?;
        self.cc.launch(f, total.div_ceil(128), 1, 128, &mut args)
    }

    fn k_gemm(
        &self,
        x: CUdeviceptr,
        w: CUdeviceptr,
        y: CUdeviceptr,
        t: usize,
        k: usize,
        n: usize,
    ) -> Result<(), String> {
        let total = (t * n) as u64;
        let (mut a0, mut a1, mut a2) = (x, w, y);
        let (mut a3, mut a4, mut a5) = (t as i32, k as i32, n as i32);
        let (mut a6, mut a7) = (k as i32, n as i32);
        let mut args: [*mut std::ffi::c_void; 8] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
            (&mut a3) as *mut _ as *mut _,
            (&mut a4) as *mut _ as *mut _,
            (&mut a5) as *mut _ as *mut _,
            (&mut a6) as *mut _ as *mut _,
            (&mut a7) as *mut _ as *mut _,
        ];
        let f = self.cc.function("llm170_ds4m_gemm")?;
        self.cc
            .launch(f, total.div_ceil(256) as u32, 1, 256, &mut args)
    }

    fn k_bf16(&self, buf: CUdeviceptr, n: usize) -> Result<(), String> {
        let (mut a0, mut a1) = (buf, n as i64);
        let mut args: [*mut std::ffi::c_void; 2] =
            [(&mut a0) as *mut _ as *mut _, (&mut a1) as *mut _ as *mut _];
        let f = self.cc.function("llm170_ds4m_bf16_rows")?;
        self.cc
            .launch(f, (n as u64).div_ceil(256) as u32, 1, 256, &mut args)
    }

    fn k_rmsw(
        &self,
        x: CUdeviceptr,
        w: CUdeviceptr,
        y: CUdeviceptr,
        rows: usize,
    ) -> Result<(), String> {
        let d = self.dims.dim;
        let eps = self.dims.rms_eps;
        let (mut a0, mut a1, mut a2) = (x, w, y);
        let (mut a3, mut a4, mut a5) = (rows as i32, d as i32, eps);
        let mut args: [*mut std::ffi::c_void; 6] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
            (&mut a3) as *mut _ as *mut _,
            (&mut a4) as *mut _ as *mut _,
            (&mut a5) as *mut _ as *mut _,
        ];
        let f = self.cc.function("llm170_ds4m_rmsw_rows")?;
        self.cc
            .launch(f, rows.div_ceil(64) as u32, 1, 64, &mut args)
    }

    fn read_f32(&self, ptr: CUdeviceptr, n: usize) -> Result<Vec<f32>, String> {
        let mut b = vec![0u8; n * 4];
        self.cc.d2h(&mut b, ptr)?;
        self.cc.sync()?;
        // SAFETY: d2h 완료 버퍼 재해석(길이·정렬 일치 — G2 판독 패턴).
        Ok(unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, n) }.to_vec())
    }

    /// 노름 체인(호스트 행렬 입력 → 디바이스 norm → 판독) — attn_norm/
    /// ffn_norm/final_norm 공용.
    fn norm_rows(&mut self, x: &[f32], rows: usize, w: CUdeviceptr) -> Result<Vec<f32>, String> {
        let d = self.dims.dim;
        if x.len() != rows * d {
            return Err(format!("ds4-mtp: norm 입력 {} != {rows}·{d}", x.len()));
        }
        self.ensure_rows(rows)?;
        let _g = self.cc.guard()?;
        self.cc.h2d(self.d_gout, &f32_bytes(x))?;
        self.k_rmsw(self.d_gout, w, self.d_rows, rows)?;
        self.cc.sync()?;
        self.read_f32(self.d_rows, rows * d)
    }

    /// 노름 체인 공개 API(검증층 중간 판정용) — x [rows][dim] → 노름 출력.
    /// kind: "attn"|"ffn"|"final"(등록 슬롯 선택).
    pub fn norm_rows_pub(
        &mut self,
        x: &[f32],
        rows: usize,
        kind: &str,
    ) -> Result<Vec<f32>, String> {
        let w = match kind {
            "attn" => self.d_attn_norm,
            "ffn" => self.d_ffn_norm,
            "final" => self.d_final_norm,
            other => return Err(format!("ds4-mtp: norm kind={other} — attn|ffn|final 계약")),
        };
        if w == 0 {
            return Err(format!("ds4-mtp: {kind}_norm 미등록"));
        }
        self.norm_rows(x, rows, w)
    }

    // ── DSpark 코어 fn 미러(frame.rs 줄번호 — 헤드 [계약]) ──

    /// dspark_main_x 미러(L208-230) — main_hidden [t][3d](트렁크 타깃
    /// 스트림 평균 인터리브) → main_x [t][d]. fp8_sim 활성 → main_proj
    /// gemm → bf16 → main_norm 가중 RMS.
    pub fn dspark_main_x(&mut self, main_hidden: &[f32]) -> Result<Vec<f32>, String> {
        let d = self.dims.dim;
        if self.d_proj == 0 || self.d_main_norm == 0 {
            return Err("ds4-mtp: main_proj/main_norm 미등록".into());
        }
        if main_hidden.is_empty() || !main_hidden.len().is_multiple_of(3 * d) {
            return Err(format!(
                "ds4-mtp: main_hidden {} — t·3·{d} 정합 오류",
                main_hidden.len()
            ));
        }
        let t = main_hidden.len() / (3 * d);
        self.ensure_rows(t)?;
        let _g = self.cc.guard()?;
        self.cc.h2d(self.dx3, &f32_bytes(main_hidden))?;
        self.k_fp8(self.dx3, t, 3 * d, 3 * d, 128)?;
        self.k_gemm(self.dx3, self.d_proj, self.d_gout, t, 3 * d, d)?;
        self.k_bf16(self.d_gout, t * d)?;
        self.k_rmsw(self.d_gout, self.d_main_norm, self.d_rows, t)?;
        self.cc.sync()?;
        self.read_f32(self.d_rows, t * d)
    }

    /// dspark_draft_ids 미러(L233-239) — [token, noise×(block-1)].
    pub fn dspark_draft_ids(&self, token: u32) -> Vec<u32> {
        let mut v = vec![self.dims.dspark_noise_token; self.dims.dspark_block];
        v[0] = token;
        v
    }

    /// dspark_window_warm 미러(L242-263) — main_x [t][d] → 링 [win][hd].
    /// project_kv 는 ds4_attn 모듈(정상 로프표, 위치 0..t-1).
    pub fn window_warm(&mut self, main_x: &[f32]) -> Result<Vec<f32>, String> {
        let (d, hd, win) = (self.dims.dim, self.dims.head_dim, self.dims.window);
        if main_x.is_empty() || !main_x.len().is_multiple_of(d) {
            return Err(format!(
                "ds4-mtp: main_x {} — t·{d} 정합 오류",
                main_x.len()
            ));
        }
        let t = main_x.len() / d;
        let kv = self.attn.stage_project_kv(self.il_main, main_x)?;
        let mut ring = vec![0.0f32; win * hd];
        if t <= win {
            ring[..t * hd].copy_from_slice(&kv);
        } else {
            // model.py prefill 분기 — 마지막 win개를 링 순서로 회전 배치.
            let cutoff = t % win;
            let tail = &kv[(t - win) * hd..];
            ring[cutoff * hd..].copy_from_slice(&tail[..(win - cutoff) * hd]);
            ring[..cutoff * hd].copy_from_slice(&tail[(win - cutoff) * hd..]);
        }
        Ok(ring)
    }

    /// dspark_decode_attn 미러(L267-345) — 드래프트 b토큰 어텐션.
    /// x_draft [b][d](attn_norm 출력) · main_x_tok [d](현재 메인 토큰
    /// hidden — project_kv 가 1토큰이므로 로프 행 0 계약) · ring [win·hd]
    /// (pos%win 슬롯에 메인 kv 기입 — 변경사본 반환) · pos = 메인 토큰
    /// 절대 위치. 반환 [b][d](그룹 출력 완료).
    pub fn dspark_decode_attn(
        &mut self,
        x_draft: &[f32],
        main_x_tok: &[f32],
        ring: &mut [f32],
        pos: usize,
    ) -> Result<Vec<f32>, String> {
        let dm = &self.dims;
        let (d, hd, nh, win, rd) = (dm.dim, dm.head_dim, dm.n_heads, dm.window, dm.rope_head_dim);
        let b = if x_draft.is_empty() {
            0
        } else {
            x_draft.len() / d
        };
        if b == 0 || x_draft.len() != b * d {
            return Err(format!(
                "ds4-mtp: x_draft {} — b·{d} 정합 오류",
                x_draft.len()
            ));
        }
        if main_x_tok.len() != d {
            return Err(format!("ds4-mtp: main_x_tok {} != {d}", main_x_tok.len()));
        }
        if ring.len() != win * hd {
            return Err(format!("ds4-mtp: ring {} != {win}·{hd}", ring.len()));
        }
        let half = self.rope_half;
        if self.rope_cs.is_empty() || half == 0 {
            return Err("ds4-mtp: 정상 로프표 미등록 — set_rope_main 먼저".into());
        }
        if self.d_sink == 0 {
            return Err("ds4-mtp: 싱크 미등록 — register_sink 먼저".into());
        }
        let rope_len = self.rope_cs.len() / (half * 2);
        if rope_len < pos + 1 + b {
            return Err(format!(
                "ds4-mtp: 로프표 len {rope_len} < pos+1+b {}",
                pos + 1 + b
            ));
        }
        // 1) 메인 토큰 kv — project_kv(1토큰 → 로프 행 0) 후 링 기입.
        let mkv = self.attn.stage_project_kv(self.il_main, main_x_tok)?;
        ring[pos % win * hd..(pos % win + 1) * hd].copy_from_slice(&mkv);
        // 2) 시프트 로프표(행 i = 절대 위치 pos+1+i) — IL_SHIFT + derot 용.
        let mut shift_cs = vec![0.0f32; b * half * 2];
        for i in 0..b {
            let src = (pos + 1 + i) * half * 2;
            shift_cs[i * half * 2..(i + 1) * half * 2]
                .copy_from_slice(&self.rope_cs[src..src + half * 2]);
        }
        self.attn.set_rope(self.il_shift, &shift_cs, half)?;
        // 3) 드래프트 q/kv — 시프트 표로 project_q/kv(코어 수동 로프와 동일 비트).
        let (_, q) = self.attn.stage_project_q(self.il_shift, x_draft)?;
        let kvd = self.attn.stage_project_kv(self.il_shift, x_draft)?;
        // 4) [링|드래프트 kv] 스파스 어텐션 — idx 행 = [0..min(win,pos+1)]
        //    ++ [win..win+b](전 드래프트 토큰 동일 행 — L313-315).
        self.ensure_b(b)?;
        let _g = self.cc.guard()?;
        let mut kv_all = Vec::with_capacity((win + b) * hd);
        kv_all.extend_from_slice(ring);
        kv_all.extend_from_slice(&kvd);
        self.cc.h2d(self.dkv, &f32_bytes(&kv_all))?;
        self.cc.h2d(self.dq, &f32_bytes(&q))?;
        let mut idxs: Vec<i32> = (0..win.min(pos + 1)).map(|v| v as i32).collect();
        idxs.extend((0..b).map(|v| (win + v) as i32));
        let stride = idxs.len();
        let mut idx_rows = vec![0i32; b * stride];
        for i in 0..b {
            idx_rows[i * stride..(i + 1) * stride].copy_from_slice(&idxs);
        }
        self.cc.h2d(self.didx, &i32_bytes(&idx_rows))?;
        self.cc.h2d(self.drope, &f32_bytes(&shift_cs))?;
        {
            let total = (b * nh) as u32;
            let (mut a0, mut a1, mut a2, mut a3) = (self.dq, self.dkv, self.didx, self.d_sink);
            let (mut a4, mut a5, mut a6, mut a7) = (self.do_, b as i32, nh as i32, hd as i32);
            let mut a8 = stride as i32;
            let mut args: [*mut std::ffi::c_void; 9] = [
                (&mut a0) as *mut _ as *mut _,
                (&mut a1) as *mut _ as *mut _,
                (&mut a2) as *mut _ as *mut _,
                (&mut a3) as *mut _ as *mut _,
                (&mut a4) as *mut _ as *mut _,
                (&mut a5) as *mut _ as *mut _,
                (&mut a6) as *mut _ as *mut _,
                (&mut a7) as *mut _ as *mut _,
                (&mut a8) as *mut _ as *mut _,
            ];
            let f = self.cc.function("llm170_ds4m_sparse_attn")?;
            self.cc.launch(f, total.div_ceil(64), 1, 64, &mut args)?;
        }
        {
            let total = (b * nh) as u32;
            let (mut a0, mut a1, mut a2, mut a3) = (self.do_, b as i32, nh as i32, hd as i32);
            let (mut a4, mut a5, mut a6) = (rd as i32, self.drope, half as i32);
            let mut args: [*mut std::ffi::c_void; 7] = [
                (&mut a0) as *mut _ as *mut _,
                (&mut a1) as *mut _ as *mut _,
                (&mut a2) as *mut _ as *mut _,
                (&mut a3) as *mut _ as *mut _,
                (&mut a4) as *mut _ as *mut _,
                (&mut a5) as *mut _ as *mut _,
                (&mut a6) as *mut _ as *mut _,
            ];
            let f = self.cc.function("llm170_ds4m_derot")?;
            self.cc.launch(f, total.div_ceil(64), 1, 64, &mut args)?;
        }
        self.cc.sync()?;
        let o = self.read_f32(self.do_, b * nh * hd)?;
        // 5) 그룹 출력 — ds4_attn stage_output(wo_a/wo_b, 로프 무관).
        self.attn.stage_output(self.il_main, &o)
    }

    /// 드래프트 블록 미러(layers.rs block_forward L28-121 — 어텐션만
    /// dspark_decode_attn) — x [b][hc·d](임베드 방송 상태) → 동일 형상.
    /// MoE 전문가는 호출부가 미리 상주시킨다(moe.add_expert_f32).
    pub fn draft_block(
        &mut self,
        x: &[Vec<f32>],
        main_x_tok: &[f32],
        ring: &mut [f32],
        pos: usize,
    ) -> Result<Vec<Vec<f32>>, String> {
        let (d, hc) = (self.dims.dim, self.dims.hc);
        let t = x.len();
        if t == 0 {
            return Err("ds4-mtp: 빈 드래프트 입력".into());
        }
        for (ti, r) in x.iter().enumerate() {
            if r.len() != hc * d {
                return Err(format!("ds4-mtp: x[{ti}] {} != hc·{d}", r.len()));
            }
        }
        if self.d_attn_norm == 0 || self.d_ffn_norm == 0 {
            return Err("ds4-mtp: attn_norm/ffn_norm 미등록".into());
        }
        // 어텐션 서브블록 — hc_pre → attn_norm → dspark 어텐션 → hc_post.
        let (y, posts, combs) = self.hc.hc_pre(self.il_main, "attn", x)?;
        let mut yflat = Vec::with_capacity(t * d);
        for r in &y {
            yflat.extend_from_slice(r);
        }
        let attn_norm = self.d_attn_norm;
        let xn = self.norm_rows(&yflat, t, attn_norm)?;
        let a = self.dspark_decode_attn(&xn, main_x_tok, ring, pos)?;
        let a_rows: Vec<Vec<f32>> = a.chunks_exact(d).map(|c| c.to_vec()).collect();
        let x2 = self.hc.hc_post(&a_rows, x, &posts, &combs)?;
        // FFN 서브블록 — hc_pre → ffn_norm → MoE(라우티드) → hc_post.
        let (y2, posts2, combs2) = self.hc.hc_pre(self.il_main, "ffn", &x2)?;
        let mut y2flat = Vec::with_capacity(t * d);
        for r in &y2 {
            y2flat.extend_from_slice(r);
        }
        let ffn_norm = self.d_ffn_norm;
        let xn2 = self.norm_rows(&y2flat, t, ffn_norm)?;
        let xn2_rows: Vec<Vec<f32>> = xn2.chunks_exact(d).map(|c| c.to_vec()).collect();
        let f = self.moe.moe_ffn_routed(&xn2_rows)?;
        self.hc.hc_post(&f, &x2, &posts2, &combs2)
    }

    /// 종결 hc_head — mtp.{last}.hc_head(자체 fn/base/scale).
    pub fn draft_head_hmix(&mut self, x_out: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
        self.hc.hc_head(x_out)
    }

    /// 종결 norm — mtp.{last}.norm(hmix [b][d] → h [b][d]).
    pub fn draft_final_norm(&mut self, hmix: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
        let d = self.dims.dim;
        if self.d_final_norm == 0 {
            return Err("ds4-mtp: final_norm 미등록".into());
        }
        let mut flat = Vec::with_capacity(hmix.len() * d);
        for r in hmix {
            if r.len() != d {
                return Err(format!("ds4-mtp: hmix 행 {} != {d}", r.len()));
            }
            flat.extend_from_slice(r);
        }
        let w = self.d_final_norm;
        let out = self.norm_rows(&flat, hmix.len(), w)?;
        Ok(out.chunks_exact(d).map(|c| c.to_vec()).collect())
    }

    /// 헤드 스트립 gemv — frame.rs head_gemv_stripwise L41-77 미러.
    /// h [b][d] · w_strip [d][128](호스트 trellis 스트립 디양자화·재팩)
    /// → 로짓 [b][128](k블록 부분합 순서, 반올림 없음). 스트립 루프는
    /// 호출부(2MB 스트립 상주 — 전량 2.1GB 적재 회피).
    pub fn head_strip_gemv(&mut self, h: &[f32], w_strip: &[f32]) -> Result<Vec<f32>, String> {
        let d = self.dims.dim;
        let b = if h.is_empty() { 0 } else { h.len() / d };
        if b == 0 || h.len() != b * d {
            return Err(format!("ds4-mtp: head h {} — b·{d} 정합 오류", h.len()));
        }
        if w_strip.len() != d * 128 {
            return Err(format!("ds4-mtp: head 스트립 {} != {d}·128", w_strip.len()));
        }
        self.ensure_b(b)?;
        let _g = self.cc.guard()?;
        self.cc.h2d(self.dheadw, &f32_bytes(w_strip))?;
        self.cc.h2d(self.dheadx, &f32_bytes(h))?;
        let total = (b * 128) as u32;
        let (mut a0, mut a1, mut a2) = (self.dheadx, self.dheadw, self.dheady);
        let (mut a3, mut a4, mut a5) = (b as i32, d as i32, 128i32);
        let mut args: [*mut std::ffi::c_void; 6] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
            (&mut a3) as *mut _ as *mut _,
            (&mut a4) as *mut _ as *mut _,
            (&mut a5) as *mut _ as *mut _,
        ];
        let f = self.cc.function("llm170_ds4m_head_strip")?;
        self.cc.launch(f, total.div_ceil(128), 1, 128, &mut args)?;
        self.cc.sync()?;
        self.read_f32(self.dheady, b * 128)
    }

    /// markov_logits_bias 미러(L373-383) — out_ids [b] → 바이어스
    /// [b][vocab] = w1행 게더 · w2 gemv(k 오름차순 f32).
    pub fn markov_logits_bias(&mut self, out_ids: &[u32]) -> Result<Vec<f32>, String> {
        let (vocab, rank) = (self.dims.vocab, self.dims.markov_rank);
        let b = out_ids.len();
        if b == 0 || b > self.dims.dspark_block * 2 {
            return Err(format!(
                "ds4-mtp: out_ids {b} — (0, {}] 도메인",
                self.dims.dspark_block * 2
            ));
        }
        if self.d_w1 == 0 || self.d_w2 == 0 {
            return Err("ds4-mtp: markov w1/w2 미등록".into());
        }
        for (i, &id) in out_ids.iter().enumerate() {
            if id as usize >= vocab {
                return Err(format!("ds4-mtp: out_ids[{i}]={id} ≥ vocab {vocab}"));
            }
        }
        self.ensure_b(b)?;
        let _g = self.cc.guard()?;
        // ids 를 dbias 버퍼 선두에 스테이징(별도 ids 버퍼 없음 — b·4 바이트).
        self.cc.h2d(self.dconf, &u32_bytes(out_ids))?;
        let total = (b * vocab) as u64;
        let (mut a0, mut a1, mut a2, mut a3) = (self.dconf, self.d_w1, self.d_w2, self.dbias);
        let (mut a4, mut a5, mut a6) = (b as i32, rank as i32, vocab as i32);
        let mut args: [*mut std::ffi::c_void; 7] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
            (&mut a3) as *mut _ as *mut _,
            (&mut a4) as *mut _ as *mut _,
            (&mut a5) as *mut _ as *mut _,
            (&mut a6) as *mut _ as *mut _,
        ];
        let f = self.cc.function("llm170_ds4m_markov")?;
        self.cc
            .launch(f, total.div_ceil(128) as u32, 1, 128, &mut args)?;
        self.cc.sync()?;
        self.read_f32(self.dbias, b * vocab)
    }

    /// confidence_score 미러(L386-397) — h [b][d] · markov_embed [b][rank]
    /// → 신뢰도 [b](chain 순서 f32 순차).
    pub fn confidence(&mut self, h: &[f32], markov_embed: &[f32]) -> Result<Vec<f32>, String> {
        let (d, rank) = (self.dims.dim, self.dims.markov_rank);
        let b = if h.is_empty() { 0 } else { h.len() / d };
        if b == 0 || h.len() != b * d || markov_embed.len() != b * rank {
            return Err(format!(
                "ds4-mtp: conf h {} / markov {} — b·{d}/b·{rank} 정합 오류",
                h.len(),
                markov_embed.len()
            ));
        }
        if self.d_conf_proj == 0 {
            return Err("ds4-mtp: conf proj 미등록".into());
        }
        self.ensure_b(b)?;
        let _g = self.cc.guard()?;
        // h 는 dheadx 재사용, markov 는 dconf 뒤 구간에 스테이징.
        self.cc.h2d(self.dheadx, &f32_bytes(h))?;
        self.cc
            .h2d(self.dconf + (b * 4) as u64, &f32_bytes(markov_embed))?;
        let m_off = self.dconf + (b * 4) as u64;
        let (mut a0, mut a1, mut a2, mut a3) = (self.dheadx, m_off, self.d_conf_proj, self.dconf);
        let (mut a4, mut a5, mut a6) = (b as i32, d as i32, rank as i32);
        let mut args: [*mut std::ffi::c_void; 7] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
            (&mut a3) as *mut _ as *mut _,
            (&mut a4) as *mut _ as *mut _,
            (&mut a5) as *mut _ as *mut _,
            (&mut a6) as *mut _ as *mut _,
        ];
        let f = self.cc.function("llm170_ds4m_conf")?;
        self.cc.launch(f, b as u32, 1, 1, &mut args)?;
        self.cc.sync()?;
        self.read_f32(self.dconf, b)
    }
}
