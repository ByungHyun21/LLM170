//! DeepSeek-V4-Flash 어텐션 스테이지 모듈층 — plans/130 B3, 2026-10-05.
//!
//! [계약] crates/core/src/deepseek4/stages/attn.rs 가 산술 원천(유일 기준 —
//! plans/124 §6 계승). 본 모듈은 SWA/CSA/HCA 3층 유형의 어텐션 프리필 체인
//! (Q/KV 프로젝션 + 컴프레서 + 인덱서 + 싱크 스파스 어텐션 + 출력 비회전 +
//! 그룹 출력)을 커널 체인(assets/ds4_attn.cu)으로 미러한다. 산술 헬퍼는
//! 전부 .cu 쪽 비트 미러(bf16 RNE·FP8/FP4 QAT 시뮬·exp_cr 트윈 — ops.rs
//! 직이식), RoPE cos/sin 은 호스트에서 ops.rs RopeTable::build L280-325
//! 대로 구축한 표를 `set_rope` 로 등록해 커널이 읽는다(테이블 구동 — 장치
//! 초월함수 전면 배제, 양측 동일 비트).
//!
//! [정합 목표 — 비트동일] 본 스테이지 산술은 f32 연산별 반올림(-fmad=false)
//! + f64 트윈 exp(ops.rs exp_cr)+호스트 rope 표라 core 미러와 비트동일이
//! 가능하다(판정 기준 bitdiff=0 — ds4_attn_cuda_probe.rs 원장).
//!
//! [층 유형 — config.rs layer map] compress_ratios[il] 0=SWA(윈도우 128) /
//! 4=CSA(겹침 압축+인덱서) / 128=HCA(비겹침 밀도). 컴프레서는 L2+ 전층,
//! 인덱서는 CSA 만(odd HCA 없음).
//!
//! [CMP 170HX(sm_80) 설계 근거 — plans/124 §0] 정합 우선 설계(1스레드=1출력
//! gemm·1스레드=1(t,h) 스파스 어텐션 — G6 fwd3s 원장 계급). t-블록 tiled
//! 변형은 sm_80 실측 후 재판정. 속도 칸 '측정 대기 sm_80'(개발기 RTX 4070
//! SUPER 타이밍 금지 — plans/130 §0).
//!
//! [음성대조] Ds4Neg 3종(top-k 인과 off-by-one·ape 미스얼라인·싱크 누락)은
//! 각각 .cu 의 쌍둥이 커널(llm170_ds4_topk_vis1·pool_apeoff·
//! sparse_attn_nosink)로 실경로 재현 — 검증층(`*_neg`) 전용(원장 17호).
//!
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 금지 —
//! scripts/cuda_probe_shim.rs 단독 컴파일 대상.

use crate::rawcuda::ctx::CudaCtx;
use crate::rawcuda::exl3_cuda::{Exl3CudaDecoder, JParser, JVal};
use crate::rawcuda::ffi::CUdeviceptr;
use std::collections::HashMap;

/// 어텐션 스파스 행 상한 — 윈도우 128 + top-k 512(ds4_attn.cu s[640] 로컬
/// 상한과 동일 값. 어텐션 결함 4호와 무관한 정적 상한 — 초과 시 모듈 Err).
pub const DS4_SPARSE_EMAX: usize = 640;

/// 층 유형 — config.rs LayerKind L8-18 미러.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ds4Kind {
    /// ratio 0 — 순수 슬라이딩 윈도우.
    Swa,
    /// ratio 4 — 겹침 압축(CSA) + 인덱서.
    Csa,
    /// ratio 128 — 비겹침 밀도 압축(HCA).
    Hca,
}

/// 음성대조 모드(원장 17호 — 검증층 전용, 프로덕션 경로 None 고정).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ds4Neg {
    /// top-k 인과 가시경계 off-by-one(visible+1).
    TopkVis1,
    /// 컴프레서 ape 행 미스얼라인(j→(j+1)%ratio).
    ApeOff,
    /// 싱크 로짓 누락(denom 에서 z' 항 제외).
    SinkDropped,
}

/// ds4 어텐션 형상 — config.json 본문에서 유도(추정 금지 — 결함 1호 정신).
/// Vision-Exp 실측(2026-10-05): dim 4096·헤드 64×512·rope 64·q_lora 1024·
/// o_lora 1024·o_groups 8·window 128·index 64×128 top-512·rms_eps 1e-20.
#[derive(Debug, Clone)]
pub struct Ds4AttnDims {
    pub n_layers: usize,
    pub dim: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub window: usize,
    pub rms_eps: f32,
    pub compress_ratios: Vec<i64>,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
}

impl Ds4AttnDims {
    /// 층 il 압축 비율.
    pub fn ratio(&self, il: usize) -> usize {
        self.compress_ratios[il] as usize
    }

    /// 층 유형 — 0/4/128 외 거부(config.rs kind L66-75 미러).
    pub fn kind(&self, il: usize) -> Result<Ds4Kind, String> {
        match self.ratio(il) {
            0 => Ok(Ds4Kind::Swa),
            4 => Ok(Ds4Kind::Csa),
            128 => Ok(Ds4Kind::Hca),
            other => Err(format!("ds4: compress_ratios[{il}]={other} — 0/4/128 외")),
        }
    }

    /// config.json 본문 → 형상.
    pub fn from_config(cfg: &str) -> Result<Self, String> {
        let v = JParser {
            b: cfg.as_bytes(),
            p: 0,
        }
        .parse()?;
        let u = |k: &str| -> Result<usize, String> {
            v.get(k)
                .and_then(JVal::as_f64)
                .map(|x| x as usize)
                .ok_or_else(|| format!("config.json: {k} 없음"))
        };
        let ratios = v
            .get("compress_ratios")
            .and_then(JVal::as_arr)
            .ok_or("config.json: compress_ratios 없음")?
            .iter()
            .filter_map(JVal::as_f64)
            .map(|x| x as i64)
            .collect::<Vec<i64>>();
        let n_layers = u("num_hidden_layers")?;
        if ratios.len() < n_layers {
            return Err(format!(
                "ds4: compress_ratios 길이 {} < n_layers {n_layers}",
                ratios.len()
            ));
        }
        Ok(Ds4AttnDims {
            n_layers,
            dim: u("hidden_size")?,
            n_heads: u("num_attention_heads")?,
            head_dim: u("head_dim")?,
            rope_head_dim: u("qk_rope_head_dim")?,
            q_lora_rank: u("q_lora_rank")?,
            o_lora_rank: u("o_lora_rank")?,
            o_groups: u("o_groups")?,
            window: u("sliding_window")?,
            rms_eps: v.get("rms_norm_eps").and_then(JVal::as_f64).unwrap_or(1e-6) as f32,
            compress_ratios: ratios,
            index_n_heads: u("index_n_heads")?,
            index_head_dim: u("index_head_dim")?,
            index_topk: u("index_topk")?,
        })
    }
}

/// 컴프레서 가중치(호스트 f32 k-major — AttnWeights 계열과 동치, loader.rs
/// comp_weights L303-328 산출물 형상).
pub struct Ds4CompF32 {
    pub wkv: Vec<f32>,
    pub wgate: Vec<f32>,
    pub ape: Vec<f32>,
    pub norm: Vec<f32>,
    pub head_dim: usize,
    pub ratio: usize,
    pub rotate: bool,
}

impl Ds4CompF32 {
    /// coff = 1 + 겹침(ratio 4 → 2) — attn.rs CompressorWeights::coff L59-61.
    pub fn coff(&self) -> usize {
        1 + usize::from(self.ratio == 4)
    }
}

/// 인덱서 가중치 — attn.rs IndexerWeights L67-72 미러.
pub struct Ds4IndexerF32 {
    pub wq_b: Vec<f32>,
    pub weights_proj: Vec<f32>,
    pub comp: Ds4CompF32,
}

/// 1층 어텐션 가중치 세트 — attn.rs AttnWeights L33-45 + comp/idx.
pub struct Ds4LayerF32 {
    pub wq_a: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub wq_b: Vec<f32>,
    pub wkv: Vec<f32>,
    pub kv_norm: Vec<f32>,
    pub sink: Vec<f32>,
    pub wo_a: Vec<Vec<f32>>,
    pub wo_b: Vec<f32>,
    pub comp: Option<Ds4CompF32>,
    pub idx: Option<Ds4IndexerF32>,
}

/// 디바이스 상주 컴프레서/인덱서/층 가중치(해제 없이 누적 — 프로브 수명).
struct Ds4CompDev {
    wkv: CUdeviceptr,
    wgate: CUdeviceptr,
    ape: CUdeviceptr,
    norm: CUdeviceptr,
    head_dim: usize,
    ratio: usize,
    rotate: bool,
}

struct Ds4IndexerDev {
    wq_b: CUdeviceptr,
    wproj: CUdeviceptr,
    comp: Ds4CompDev,
}

struct Ds4LayerDev {
    wq_a: CUdeviceptr,
    q_norm: CUdeviceptr,
    wq_b: CUdeviceptr,
    wkv: CUdeviceptr,
    kv_norm: CUdeviceptr,
    sink: CUdeviceptr,
    wo_a: Vec<CUdeviceptr>,
    wo_b: CUdeviceptr,
    comp: Option<Ds4CompDev>,
    idx: Option<Ds4IndexerDev>,
}

/// ds4 어텐션 모듈 — CudaCtx 단일 소유(단일 상주 원칙).
pub struct Ds4AttnCuda {
    /// 디바이스 컨텍스트(모듈 단독 소유 — 병렬 작업 계약).
    pub cc: CudaCtx,
    pub dims: Ds4AttnDims,
    layers: HashMap<usize, Ds4LayerDev>,
    /// RoPE 표 [len][half][2] f32(호스트 RopeTable::build 산출물) — 층별.
    ropes: HashMap<usize, (CUdeviceptr, usize, usize)>,
    // ── 작업 버퍼(토큰 용량 cap_t 까지 재사용 — ensure 계약) ──
    dx: CUdeviceptr,      // [t][dim] 원본 x
    dxq8: CUdeviceptr,    // [t][dim] fp8-sim 사본
    dc: CUdeviceptr,      // [t][qrank] c / c_Q
    dc2: CUdeviceptr,     // [t][qrank] q_norm 적용 후
    dcq8: CUdeviceptr,    // [t][qrank] fp8-sim 사본
    dq: CUdeviceptr,      // [t][nh*hd] q(로프 완료)
    dkv: CUdeviceptr,     // [t][hd] kv 프로젝션
    dkv2: CUdeviceptr,    // [t][hd] kv_norm+rope+fp8 완료
    dkvc: CUdeviceptr,    // [t][1024] 컴프레서 wkv 출력(coff·hd 상한)
    dscc: CUdeviceptr,    // [t][1024] wgate 출력
    dpool: CUdeviceptr,   // [t][512] 풀/종결(본체 컴프레서)
    dpooli: CUdeviceptr,  // [t][128] 인덱서 컴프레서 풀/종결
    diq: CUdeviceptr,     // [t][ih*id] 인덱서 qI
    diw: CUdeviceptr,     // [t][ih] 헤드 가중치
    dscores: CUdeviceptr, // [t][t] 인덱서 스코어(nb 상한 = t)
    dsel: CUdeviceptr,    // [t][topk] 선택(-1 패드)
    didx: CUdeviceptr,    // [t][win+topk] 스파스 인덱스(-1 패드)
    dkvall: CUdeviceptr,  // [t+nb][hd] 윈도우 kv + 압축 엔트리
    do_: CUdeviceptr,     // [t][nh*hd] 어텐션 출력(비회전 완료)
    dlat: CUdeviceptr,    // [t][g*r] latents
    dy: CUdeviceptr,      // [t][dim] 층 출력
    cap_t: usize,
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

impl Ds4AttnCuda {
    /// ds4_attn.fatbin 자산 해석 — LLM170_CUDA_DS4_ATTN_FATBIN_PATH
    /// 오버라이드 우선(자산 경로 오버라이드일 뿐 계산 경로 분기 아님).
    fn fatbin_bytes() -> Result<Vec<u8>, String> {
        const ENV: &str = "LLM170_CUDA_DS4_ATTN_FATBIN_PATH";
        const REL: &[&str] = &[
            "crates/backend-gpu/src/rawcuda/assets/ds4_attn.fatbin",
            "src/rawcuda/assets/ds4_attn.fatbin",
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
            "ds4_attn.fatbin 없음 — scripts/build_cuda.bat 실행 또는 {ENV} 지정 (탐색: {REL:?})"
        ))
    }

    /// 개방 — 컨텍스트 + ds4_attn.fatbin 적재(커널 15종 + 음성 쌍둥이 3종).
    pub fn new(dims: Ds4AttnDims) -> Result<Self, String> {
        let image = Self::fatbin_bytes()?;
        let mut cc = CudaCtx::new()?;
        let _g = cc.guard()?;
        cc.load_fatbin(
            "ds4attn",
            &image,
            &[
                "llm170_ds4_fp8_rows",
                "llm170_ds4_gemm",
                "llm170_ds4_bf16_rows",
                "llm170_ds4_rmsw_rows",
                "llm170_ds4_head_rms",
                "llm170_ds4_rope_tail",
                "llm170_ds4_pool",
                "llm170_ds4_comp_finish",
                "llm170_ds4_indexer_q",
                "llm170_ds4_iw_scale",
                "llm170_ds4_indexer_scores",
                "llm170_ds4_fill_i32",
                "llm170_ds4_topk",
                "llm170_ds4_sparse_attn",
                "llm170_ds4_derot",
                // 음성대조 쌍둥이(원장 17호) — *_neg 검증 경로만 발사.
                "llm170_ds4_topk_vis1",
                "llm170_ds4_pool_apeoff",
                "llm170_ds4_sparse_attn_nosink",
            ],
        )?;
        Ok(Ds4AttnCuda {
            cc,
            dims,
            layers: HashMap::new(),
            ropes: HashMap::new(),
            dx: 0,
            dxq8: 0,
            dc: 0,
            dc2: 0,
            dcq8: 0,
            dq: 0,
            dkv: 0,
            dkv2: 0,
            dkvc: 0,
            dscc: 0,
            dpool: 0,
            dpooli: 0,
            diq: 0,
            diw: 0,
            dscores: 0,
            dsel: 0,
            didx: 0,
            dkvall: 0,
            do_: 0,
            dlat: 0,
            dy: 0,
            cap_t: 0,
        })
    }

    /// 디바이스 이름(프로브 보고용).
    pub fn device_name(&self) -> &str {
        &self.cc.device_name
    }

    /// RoPE 표 등록 — cs: [len][half][2] f32(호스트 RopeTable::build 산출물).
    /// half = rope_head_dim/2. 층 재등록은 이전 표 해제 후 교체.
    pub fn set_rope(&mut self, il: usize, cs: &[f32], half: usize) -> Result<(), String> {
        if half == 0 || cs.len() % (half * 2) != 0 {
            return Err(format!("ds4: rope 표 {} × half {half} 정합 오류", cs.len()));
        }
        let len = cs.len() / (half * 2);
        let _g = self.cc.guard()?;
        if let Some(old) = self.ropes.remove(&il) {
            self.cc.free(old.0)?;
        }
        let d = self.cc.alloc(cs.len() * 4)?;
        Exl3CudaDecoder::h2d_chunked(&self.cc, d, &f32_bytes(cs))?;
        self.ropes.insert(il, (d, half, len));
        Ok(())
    }

    fn rope_of(&self, il: usize) -> Result<(CUdeviceptr, usize, usize), String> {
        self.ropes
            .get(&il)
            .copied()
            .ok_or_else(|| format!("ds4: 층 {il} rope 표 미등록(set_rope 먼저)"))
    }

    /// 컴프레서 가중치 업로드(형상 검증 포함 — 추정 금지).
    fn up_comp(cc: &CudaCtx, cw: &Ds4CompF32, dim: usize) -> Result<Ds4CompDev, String> {
        let cd = cw.coff() * cw.head_dim;
        if cw.wkv.len() != dim * cd || cw.wgate.len() != dim * cd {
            return Err(format!(
                "ds4 comp: wkv/wgate {}/{} != {dim}×{cd}",
                cw.wkv.len(),
                cw.wgate.len()
            ));
        }
        let expected_ape = cw.ratio * cd;
        if cw.ape.len() != expected_ape {
            return Err(format!(
                "ds4 comp: ape {} != ratio {}×{cd}",
                cw.ape.len(),
                cw.ratio
            ));
        }
        if cw.norm.len() != cw.head_dim {
            return Err(format!(
                "ds4 comp: norm {} != head_dim {}",
                cw.norm.len(),
                cw.head_dim
            ));
        }
        let upl = |v: &[f32]| -> Result<CUdeviceptr, String> {
            let d = cc.alloc(v.len() * 4)?;
            Exl3CudaDecoder::h2d_chunked(cc, d, &f32_bytes(v))?;
            Ok(d)
        };
        Ok(Ds4CompDev {
            wkv: upl(&cw.wkv)?,
            wgate: upl(&cw.wgate)?,
            ape: upl(&cw.ape)?,
            norm: upl(&cw.norm)?,
            head_dim: cw.head_dim,
            ratio: cw.ratio,
            rotate: cw.rotate,
        })
    }

    /// 층 가중치 등록(형상 전부 검증 — 형상 추정 금지, 결함 1호). 재등록은
    /// 이전 디바이스 버퍼 해제 후 교체.
    pub fn register_layer(&mut self, il: usize, w: &Ds4LayerF32) -> Result<(), String> {
        let d = self.dims.clone();
        let (dim, qr, nh, hd) = (d.dim, d.q_lora_rank, d.n_heads, d.head_dim);
        let kind = d.kind(il)?;
        if w.wq_a.len() != dim * qr || w.wq_b.len() != qr * nh * hd || w.wkv.len() != dim * hd {
            return Err(format!(
                "ds4 L{il}: wq_a/wq_b/wkv 형상 오류 {}/{}/{}",
                w.wq_a.len(),
                w.wq_b.len(),
                w.wkv.len()
            ));
        }
        if w.q_norm.len() != qr || w.kv_norm.len() != hd || w.sink.len() != nh {
            return Err(format!(
                "ds4 L{il}: q_norm/kv_norm/sink {}/{}/{}",
                w.q_norm.len(),
                w.kv_norm.len(),
                w.sink.len()
            ));
        }
        let gr = d.o_groups * d.o_lora_rank;
        if w.wo_a.len() != d.o_groups
            || w.wo_a
                .iter()
                .any(|g| g.len() != (nh * hd / d.o_groups) * d.o_lora_rank)
            || w.wo_b.len() != gr * dim
        {
            return Err(format!("ds4 L{il}: wo_a/wo_b 형상 오류",));
        }
        match kind {
            Ds4Kind::Swa => {
                if w.comp.is_some() || w.idx.is_some() {
                    return Err(format!("ds4 L{il}: SWA 층에 comp/idx 불필요"));
                }
            }
            Ds4Kind::Csa => {
                if w.comp.is_none() || w.idx.is_none() {
                    return Err(format!("ds4 L{il}: CSA 층에 comp+idx 필수"));
                }
            }
            Ds4Kind::Hca => {
                if w.comp.is_none() || w.idx.is_some() {
                    return Err(format!("ds4 L{il}: HCA 층은 comp 만"));
                }
            }
        }
        let _g = self.cc.guard()?;
        if let Some(old) = self.layers.remove(&il) {
            self.cc.free(old.wq_a)?;
            self.cc.free(old.q_norm)?;
            self.cc.free(old.wq_b)?;
            self.cc.free(old.wkv)?;
            self.cc.free(old.kv_norm)?;
            self.cc.free(old.sink)?;
            for g in old.wo_a {
                self.cc.free(g)?;
            }
            self.cc.free(old.wo_b)?;
        }
        let upl = |v: &[f32]| -> Result<CUdeviceptr, String> {
            let dp = self.cc.alloc(v.len() * 4)?;
            Exl3CudaDecoder::h2d_chunked(&self.cc, dp, &f32_bytes(v))?;
            Ok(dp)
        };
        let comp = match &w.comp {
            Some(cw) => Some(Self::up_comp(&self.cc, cw, dim)?),
            None => None,
        };
        let idx = match &w.idx {
            Some(iw) => {
                if iw.wq_b.len() != qr * d.index_n_heads * d.index_head_dim {
                    return Err(format!(
                        "ds4 L{il}: indexer wq_b {} != {qr}×{}×{}",
                        iw.wq_b.len(),
                        d.index_n_heads,
                        d.index_head_dim
                    ));
                }
                if iw.weights_proj.len() != dim * d.index_n_heads {
                    return Err(format!(
                        "ds4 L{il}: weights_proj {} != {dim}×{}",
                        iw.weights_proj.len(),
                        d.index_n_heads
                    ));
                }
                Some(Ds4IndexerDev {
                    wq_b: upl(&iw.wq_b)?,
                    wproj: upl(&iw.weights_proj)?,
                    comp: Self::up_comp(&self.cc, &iw.comp, dim)?,
                })
            }
            None => None,
        };
        let wo_a = w
            .wo_a
            .iter()
            .map(|g| upl(g))
            .collect::<Result<Vec<_>, _>>()?;
        self.layers.insert(
            il,
            Ds4LayerDev {
                wq_a: upl(&w.wq_a)?,
                q_norm: upl(&w.q_norm)?,
                wq_b: upl(&w.wq_b)?,
                wkv: upl(&w.wkv)?,
                kv_norm: upl(&w.kv_norm)?,
                sink: upl(&w.sink)?,
                wo_a,
                wo_b: upl(&w.wo_b)?,
                comp,
                idx,
            },
        );
        Ok(())
    }

    fn layer(&self, il: usize) -> Result<&Ds4LayerDev, String> {
        self.layers
            .get(&il)
            .ok_or_else(|| format!("ds4: 층 {il} 미등록(register_layer 먼저)"))
    }

    /// 토큰 용량 버퍼 보장(확장 시에만 재할당).
    fn ensure(&mut self, t: usize) -> Result<(), String> {
        if t <= self.cap_t {
            return Ok(());
        }
        let d = self.dims.clone();
        let _g = self.cc.guard()?;
        if self.cap_t > 0 {
            for p in [
                self.dx,
                self.dxq8,
                self.dc,
                self.dc2,
                self.dcq8,
                self.dq,
                self.dkv,
                self.dkv2,
                self.dkvc,
                self.dscc,
                self.dpool,
                self.dpooli,
                self.diq,
                self.diw,
                self.dscores,
                self.dsel,
                self.didx,
                self.dkvall,
                self.do_,
                self.dlat,
                self.dy,
            ] {
                if p != 0 {
                    self.cc.free(p)?;
                }
            }
        }
        let cd_max = 2 * d.head_dim; // CSA coff=2 상한(인덱서 256 포함)
        let nh = d.n_heads;
        let hd = d.head_dim;
        self.dx = self.cc.alloc(t * d.dim * 4)?;
        self.dxq8 = self.cc.alloc(t * d.dim * 4)?;
        self.dc = self.cc.alloc(t * d.q_lora_rank * 4)?;
        self.dc2 = self.cc.alloc(t * d.q_lora_rank * 4)?;
        self.dcq8 = self.cc.alloc(t * d.q_lora_rank * 4)?;
        self.dq = self.cc.alloc(t * nh * hd * 4)?;
        self.dkv = self.cc.alloc(t * hd * 4)?;
        self.dkv2 = self.cc.alloc(t * hd * 4)?;
        self.dkvc = self.cc.alloc(t * cd_max * 4)?;
        self.dscc = self.cc.alloc(t * cd_max * 4)?;
        self.dpool = self.cc.alloc(t * hd * 4)?;
        self.dpooli = self.cc.alloc(t * d.index_head_dim * 4)?;
        self.diq = self.cc.alloc(t * d.index_n_heads * d.index_head_dim * 4)?;
        self.diw = self.cc.alloc(t * d.index_n_heads * 4)?;
        self.dscores = self.cc.alloc(t * t * 4)?;
        self.dsel = self.cc.alloc(t * d.index_topk * 4)?;
        self.didx = self.cc.alloc(t * (d.window + d.index_topk) * 4)?;
        self.dkvall = self.cc.alloc(2 * t * hd * 4)?;
        self.do_ = self.cc.alloc(t * nh * hd * 4)?;
        self.dlat = self.cc.alloc(t * d.o_groups * d.o_lora_rank * 4)?;
        self.dy = self.cc.alloc(t * d.dim * 4)?;
        self.cap_t = t;
        Ok(())
    }

    // ── 커널 런치 헬퍼(공통 인자 배열 조립 — 호출부 중복 최소) ──

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
        let f = self.cc.function("llm170_ds4_fp8_rows")?;
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
        x_stride: usize,
        y_stride: usize,
    ) -> Result<(), String> {
        let total = (t * n) as u64;
        let (mut a0, mut a1, mut a2) = (x, w, y);
        let (mut a3, mut a4, mut a5) = (t as i32, k as i32, n as i32);
        let (mut a6, mut a7) = (x_stride as i32, y_stride as i32);
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
        let f = self.cc.function("llm170_ds4_gemm")?;
        self.cc
            .launch(f, total.div_ceil(256) as u32, 1, 256, &mut args)
    }

    fn k_bf16(&self, buf: CUdeviceptr, n: usize) -> Result<(), String> {
        let (mut a0, mut a1) = (buf, n as i64);
        let mut args: [*mut std::ffi::c_void; 2] =
            [(&mut a0) as *mut _ as *mut _, (&mut a1) as *mut _ as *mut _];
        let f = self.cc.function("llm170_ds4_bf16_rows")?;
        self.cc
            .launch(f, (n as u64).div_ceil(256) as u32, 1, 256, &mut args)
    }

    fn k_rmsw(
        &self,
        x: CUdeviceptr,
        w: CUdeviceptr,
        y: CUdeviceptr,
        rows: usize,
        cols: usize,
    ) -> Result<(), String> {
        let eps = self.dims.rms_eps;
        let (mut a0, mut a1, mut a2) = (x, w, y);
        let (mut a3, mut a4, mut a5) = (rows as i32, cols as i32, eps);
        let mut args: [*mut std::ffi::c_void; 6] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
            (&mut a3) as *mut _ as *mut _,
            (&mut a4) as *mut _ as *mut _,
            (&mut a5) as *mut _ as *mut _,
        ];
        let f = self.cc.function("llm170_ds4_rmsw_rows")?;
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

    // ── 스테이지 진입(attn.rs 스테이지 fn 미러 — 검증층 판정점) ──

    /// Q 프로젝션 + RoPE — attn.rs project_q L84-120 + rope_q L122-135.
    /// 반환 (c_Q [t×qrank] bf16값, q [t×nh×hd] 로프 완료 bf16값).
    pub fn stage_project_q(
        &mut self,
        il: usize,
        x: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let d = self.dims.clone();
        let t = x.len() / d.dim;
        if t == 0 || x.len() != t * d.dim {
            return Err(format!("ds4: x.len {} — t×dim 정합 오류", x.len()));
        }
        let lay = self.layer(il)?;
        let (dwq_a, dq_norm, dwq_b) = (lay.wq_a, lay.q_norm, lay.wq_b);
        let (drope, dhalf, dlen) = self.rope_of(il)?;
        if dlen < t {
            return Err(format!("ds4: rope 표 len {dlen} < t {t}"));
        }
        self.ensure(t)?;
        let _g = self.cc.guard()?;
        let xb = f32_bytes(x);
        Exl3CudaDecoder::h2d_chunked(&self.cc, self.dx, &xb)?;
        // fp8-sim 활성 → wq_a gemm → bf16 → q_norm 가중 RMS → dc2.
        Exl3CudaDecoder::h2d_chunked(&self.cc, self.dxq8, &xb)?;
        self.k_fp8(self.dxq8, t, d.dim, d.dim, 128)?;
        self.k_gemm(
            self.dxq8,
            dwq_a,
            self.dc,
            t,
            d.dim,
            d.q_lora_rank,
            d.dim,
            d.q_lora_rank,
        )?;
        self.k_bf16(self.dc, t * d.q_lora_rank)?;
        self.k_rmsw(self.dc, dq_norm, self.dc2, t, d.q_lora_rank)?;
        // fp8-sim → wq_b gemm → bf16 → 헤드 RMS → rope 꼬리.
        let cb = self.read_f32(self.dc2, t * d.q_lora_rank)?;
        Exl3CudaDecoder::h2d_chunked(&self.cc, self.dcq8, &f32_bytes(&cb))?;
        self.k_fp8(self.dcq8, t, d.q_lora_rank, d.q_lora_rank, 128)?;
        self.k_gemm(
            self.dcq8,
            dwq_b,
            self.dq,
            t,
            d.q_lora_rank,
            d.n_heads * d.head_dim,
            d.q_lora_rank,
            d.n_heads * d.head_dim,
        )?;
        self.k_bf16(self.dq, t * d.n_heads * d.head_dim)?;
        {
            let total = (t * d.n_heads) as u32;
            let (mut a0, mut a1, mut a2, mut a3) =
                (self.dq, t as i32, d.n_heads as i32, d.head_dim as i32);
            let mut a4 = self.dims.rms_eps;
            let mut args: [*mut std::ffi::c_void; 5] = [
                (&mut a0) as *mut _ as *mut _,
                (&mut a1) as *mut _ as *mut _,
                (&mut a2) as *mut _ as *mut _,
                (&mut a3) as *mut _ as *mut _,
                (&mut a4) as *mut _ as *mut _,
            ];
            let f = self.cc.function("llm170_ds4_head_rms")?;
            self.cc.launch(f, total.div_ceil(128), 1, 128, &mut args)?;
        }
        {
            let total = (t * d.n_heads) as u32;
            let (mut a0, mut a1, mut a2, mut a3) =
                (self.dq, t as i32, d.n_heads as i32, d.head_dim as i32);
            let (mut a4, mut a5, mut a6) = (d.rope_head_dim as i32, drope, dhalf as i32);
            let mut args: [*mut std::ffi::c_void; 7] = [
                (&mut a0) as *mut _ as *mut _,
                (&mut a1) as *mut _ as *mut _,
                (&mut a2) as *mut _ as *mut _,
                (&mut a3) as *mut _ as *mut _,
                (&mut a4) as *mut _ as *mut _,
                (&mut a5) as *mut _ as *mut _,
                (&mut a6) as *mut _ as *mut _,
            ];
            let f = self.cc.function("llm170_ds4_rope_tail")?;
            self.cc.launch(f, total.div_ceil(128), 1, 128, &mut args)?;
        }
        self.cc.sync()?;
        let q = self.read_f32(self.dq, t * d.n_heads * d.head_dim)?;
        Ok((cb, q))
    }

    /// KV 프로젝션 — attn.rs project_kv L137-158.
    /// kv [t×hd]: norm → rope 꼬리 → 비로프 448 FP8-sim.
    pub fn stage_project_kv(&mut self, il: usize, x: &[f32]) -> Result<Vec<f32>, String> {
        let d = self.dims.clone();
        let t = x.len() / d.dim;
        let lay = self.layer(il)?;
        let (dwkv, dknorm) = (lay.wkv, lay.kv_norm);
        let (drope, dhalf, dlen) = self.rope_of(il)?;
        if dlen < t {
            return Err(format!("ds4: rope 표 len {dlen} < t {t}"));
        }
        self.ensure(t)?;
        let _g = self.cc.guard()?;
        let xb = f32_bytes(x);
        Exl3CudaDecoder::h2d_chunked(&self.cc, self.dxq8, &xb)?;
        self.k_fp8(self.dxq8, t, d.dim, d.dim, 128)?;
        self.k_gemm(
            self.dxq8, dwkv, self.dkv, t, d.dim, d.head_dim, d.dim, d.head_dim,
        )?;
        self.k_bf16(self.dkv, t * d.head_dim)?;
        self.k_rmsw(self.dkv, dknorm, self.dkv2, t, d.head_dim)?;
        {
            let total = t as u32;
            let (mut a0, mut a1, mut a2, mut a3) = (self.dkv2, t as i32, 1i32, d.head_dim as i32);
            let (mut a4, mut a5, mut a6) = (d.rope_head_dim as i32, drope, dhalf as i32);
            let mut args: [*mut std::ffi::c_void; 7] = [
                (&mut a0) as *mut _ as *mut _,
                (&mut a1) as *mut _ as *mut _,
                (&mut a2) as *mut _ as *mut _,
                (&mut a3) as *mut _ as *mut _,
                (&mut a4) as *mut _ as *mut _,
                (&mut a5) as *mut _ as *mut _,
                (&mut a6) as *mut _ as *mut _,
            ];
            let f = self.cc.function("llm170_ds4_rope_tail")?;
            self.cc.launch(f, total.div_ceil(64), 1, 64, &mut args)?;
        }
        // 비로프 d−rd 폭만 FP8-sim(64블록 — 로프 딤 bf16 유지).
        self.k_fp8(self.dkv2, t, d.head_dim, d.head_dim - d.rope_head_dim, 64)?;
        self.cc.sync()?;
        self.read_f32(self.dkv2, t * d.head_dim)
    }

    /// 컴프레서(본체) — attn.rs attention_forward L616-630(compressor_pool_
    /// prefill + compressor_finish). 반환 압축 엔트리 [nb][hd].
    /// neg=ApeOff 면 음성 쌍둥이(pool_apeoff)로 풀 계산(원장 17호).
    pub fn stage_compress(
        &mut self,
        il: usize,
        x: &[f32],
        neg: Option<Ds4Neg>,
    ) -> Result<Vec<f32>, String> {
        let d = self.dims.clone();
        let t = x.len() / d.dim;
        let lay = self.layer(il)?;
        let Some(comp) = &lay.comp else {
            return Err(format!("ds4 L{il}: 컴프레서 없음(SWA)"));
        };
        let (cwkv, cwgate, cape, cnorm, cratio, chd, crotate) = (
            comp.wkv,
            comp.wgate,
            comp.ape,
            comp.norm,
            comp.ratio,
            comp.head_dim,
            comp.rotate,
        );
        let (ratio, hd, coff) = (cratio, chd, 1 + usize::from(cratio == 4));
        let cd = coff * hd;
        let (drope, dhalf, dlen) = self.rope_of(il)?;
        let nb = t / ratio;
        let max_pos = nb.saturating_sub(1) * ratio;
        if dlen <= max_pos {
            return Err(format!("ds4: rope 표 len {dlen} ≤ 블록 위치 {max_pos}"));
        }
        self.ensure(t)?;
        let _g = self.cc.guard()?;
        Exl3CudaDecoder::h2d_chunked(&self.cc, self.dx, &f32_bytes(x))?;
        self.k_gemm(self.dx, cwkv, self.dkvc, t, d.dim, cd, d.dim, cd)?;
        self.k_gemm(self.dx, cwgate, self.dscc, t, d.dim, cd, d.dim, cd)?;
        // 풀 — neg=ApeOff 만 쌍둥이(나머지 neg 는 다른 스테이지 소관).
        {
            let total = (nb * hd) as u32;
            let (mut a0, mut a1, mut a2, mut a3) = (self.dkvc, self.dscc, cape, self.dpool);
            let (mut a4, mut a5, mut a6, mut a7) = (
                nb as i32,
                hd as i32,
                ratio as i32,
                usize::from(coff == 2) as i32,
            );
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
            let name = if neg == Some(Ds4Neg::ApeOff) {
                "llm170_ds4_pool_apeoff"
            } else {
                "llm170_ds4_pool"
            };
            let f = self.cc.function(name)?;
            self.cc.launch(f, total.div_ceil(128), 1, 128, &mut args)?;
        }
        {
            let (mut a0, mut a1, mut a2, mut a3) = (self.dpool, cnorm, nb as i32, hd as i32);
            let (mut a4, mut a5, mut a6) = (
                d.rope_head_dim as i32,
                ratio as i32,
                usize::from(crotate) as i32,
            );
            let (mut a7, mut a8, mut a9) = (drope, dhalf as i32, self.dims.rms_eps);
            let mut args: [*mut std::ffi::c_void; 10] = [
                (&mut a0) as *mut _ as *mut _,
                (&mut a1) as *mut _ as *mut _,
                (&mut a2) as *mut _ as *mut _,
                (&mut a3) as *mut _ as *mut _,
                (&mut a4) as *mut _ as *mut _,
                (&mut a5) as *mut _ as *mut _,
                (&mut a6) as *mut _ as *mut _,
                (&mut a7) as *mut _ as *mut _,
                (&mut a8) as *mut _ as *mut _,
                (&mut a9) as *mut _ as *mut _,
            ];
            let f = self.cc.function("llm170_ds4_comp_finish")?;
            self.cc
                .launch(f, nb.div_ceil(32) as u32, 1, 32, &mut args)?;
        }
        self.cc.sync()?;
        self.read_f32(self.dpool, nb * hd)
    }

    /// 인덱서 체인 — attn.rs indexer_q L394-427 · indexer_k L429-450 ·
    /// indexer_weights L452-463 · indexer_scores L465-492 · indexer_topk
    /// L494-529. 반환 (qI, kI, w, scores[t][nb], sel[t][topk]).
    /// neg=TopkVis1 면 top-k 쌍둥이(vis1)로 선택(원장 17호).
    pub fn stage_indexer(
        &mut self,
        il: usize,
        c_q: &[f32],
        x: &[f32],
        neg: Option<Ds4Neg>,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<i32>), String> {
        let d = self.dims.clone();
        let t = x.len() / d.dim;
        let lay = self.layer(il)?;
        let Some(idx) = &lay.idx else {
            return Err(format!("ds4 L{il}: 인덱서 없음(CSA 전용)"));
        };
        let (iwq_b, iwproj, icwkv, icwgate, icape, icnorm, icratio, ichd) = (
            idx.wq_b,
            idx.wproj,
            idx.comp.wkv,
            idx.comp.wgate,
            idx.comp.ape,
            idx.comp.norm,
            idx.comp.ratio,
            idx.comp.head_dim,
        );
        let (ih, id, rd) = (d.index_n_heads, d.index_head_dim, d.rope_head_dim);
        let (ratio, coff) = (icratio, 1 + usize::from(icratio == 4));
        let cd = coff * ichd;
        let nb = t / ratio;
        let (drope, dhalf, dlen) = self.rope_of(il)?;
        if dlen < t || dlen <= nb.saturating_sub(1) * ratio {
            return Err(format!("ds4: rope 표 len {dlen} < t {t}/블록 위치"));
        }
        self.ensure(t)?;
        let _g = self.cc.guard()?;
        // qI: c_Q fp8-sim → wq_b gemm → bf16 → rope+had+fp4(커널 9).
        Exl3CudaDecoder::h2d_chunked(&self.cc, self.dcq8, &f32_bytes(c_q))?;
        self.k_fp8(self.dcq8, t, d.q_lora_rank, d.q_lora_rank, 128)?;
        self.k_gemm(
            self.dcq8,
            iwq_b,
            self.diq,
            t,
            d.q_lora_rank,
            ih * id,
            d.q_lora_rank,
            ih * id,
        )?;
        self.k_bf16(self.diq, t * ih * id)?;
        {
            let total = (t * ih) as u32;
            let (mut a0, mut a1, mut a2, mut a3) = (self.diq, t as i32, ih as i32, id as i32);
            let (mut a4, mut a5, mut a6) = (rd as i32, drope, dhalf as i32);
            let mut args: [*mut std::ffi::c_void; 7] = [
                (&mut a0) as *mut _ as *mut _,
                (&mut a1) as *mut _ as *mut _,
                (&mut a2) as *mut _ as *mut _,
                (&mut a3) as *mut _ as *mut _,
                (&mut a4) as *mut _ as *mut _,
                (&mut a5) as *mut _ as *mut _,
                (&mut a6) as *mut _ as *mut _,
            ];
            let f = self.cc.function("llm170_ds4_indexer_q")?;
            self.cc.launch(f, total.div_ceil(64), 1, 64, &mut args)?;
        }
        // kI: 전용 컴프레서 gemm(원시 x) → 풀 → 종결(rotate).
        Exl3CudaDecoder::h2d_chunked(&self.cc, self.dx, &f32_bytes(x))?;
        self.k_gemm(self.dx, icwkv, self.dkvc, t, d.dim, cd, d.dim, cd)?;
        self.k_gemm(self.dx, icwgate, self.dscc, t, d.dim, cd, d.dim, cd)?;
        {
            let total = (nb * id) as u32;
            let (mut a0, mut a1, mut a2, mut a3) = (self.dkvc, self.dscc, icape, self.dpooli);
            let (mut a4, mut a5, mut a6, mut a7) = (
                nb as i32,
                id as i32,
                ratio as i32,
                usize::from(coff == 2) as i32,
            );
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
            let f = self.cc.function("llm170_ds4_pool")?;
            self.cc.launch(f, total.div_ceil(128), 1, 128, &mut args)?;
        }
        {
            let (mut a0, mut a1, mut a2, mut a3) = (self.dpooli, icnorm, nb as i32, id as i32);
            let (mut a4, mut a5, mut a6) = (rd as i32, ratio as i32, 1i32);
            let (mut a7, mut a8, mut a9) = (drope, dhalf as i32, self.dims.rms_eps);
            let mut args: [*mut std::ffi::c_void; 10] = [
                (&mut a0) as *mut _ as *mut _,
                (&mut a1) as *mut _ as *mut _,
                (&mut a2) as *mut _ as *mut _,
                (&mut a3) as *mut _ as *mut _,
                (&mut a4) as *mut _ as *mut _,
                (&mut a5) as *mut _ as *mut _,
                (&mut a6) as *mut _ as *mut _,
                (&mut a7) as *mut _ as *mut _,
                (&mut a8) as *mut _ as *mut _,
                (&mut a9) as *mut _ as *mut _,
            ];
            let f = self.cc.function("llm170_ds4_comp_finish")?;
            self.cc
                .launch(f, nb.div_ceil(32) as u32, 1, 32, &mut args)?;
        }
        // 헤드 가중치: gemm → bf16(bf16·c) 이중 경계.
        self.k_gemm(self.dx, iwproj, self.diw, t, d.dim, ih, d.dim, ih)?;
        {
            let c = 1.0f32 / (id as f32).sqrt() / (ih as f32).sqrt();
            let n = (t * ih) as i64;
            let (mut a0, mut a1, mut a2) = (self.diw, n, c);
            let mut args: [*mut std::ffi::c_void; 3] = [
                (&mut a0) as *mut _ as *mut _,
                (&mut a1) as *mut _ as *mut _,
                (&mut a2) as *mut _ as *mut _,
            ];
            let f = self.cc.function("llm170_ds4_iw_scale")?;
            self.cc
                .launch(f, (n as u64).div_ceil(256) as u32, 1, 256, &mut args)?;
        }
        // 스코어 → top-k 선택.
        {
            let total = (t * nb) as u32;
            let (mut a0, mut a1, mut a2, mut a3) = (self.diq, self.dpooli, self.diw, self.dscores);
            let (mut a4, mut a5, mut a6, mut a7) = (t as i32, nb as i32, ih as i32, id as i32);
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
            let f = self.cc.function("llm170_ds4_indexer_scores")?;
            self.cc.launch(f, total.div_ceil(128), 1, 128, &mut args)?;
        }
        {
            let n = (t * d.index_topk) as i64;
            let (mut a0, mut a1, mut a2) = (self.dsel, -1i32, n);
            let mut args: [*mut std::ffi::c_void; 3] = [
                (&mut a0) as *mut _ as *mut _,
                (&mut a1) as *mut _ as *mut _,
                (&mut a2) as *mut _ as *mut _,
            ];
            let f = self.cc.function("llm170_ds4_fill_i32")?;
            self.cc
                .launch(f, (n as u64).div_ceil(256) as u32, 1, 256, &mut args)?;
        }
        {
            let total = (t * nb) as u32;
            let (mut a0, mut a1, mut a2, mut a3) = (self.dscores, self.dsel, t as i32, nb as i32);
            let (mut a4, mut a5, mut a6) = (ratio as i32, d.index_topk as i32, t as i32);
            let mut args: [*mut std::ffi::c_void; 7] = [
                (&mut a0) as *mut _ as *mut _,
                (&mut a1) as *mut _ as *mut _,
                (&mut a2) as *mut _ as *mut _,
                (&mut a3) as *mut _ as *mut _,
                (&mut a4) as *mut _ as *mut _,
                (&mut a5) as *mut _ as *mut _,
                (&mut a6) as *mut _ as *mut _,
            ];
            let name = if neg == Some(Ds4Neg::TopkVis1) {
                "llm170_ds4_topk_vis1"
            } else {
                "llm170_ds4_topk"
            };
            let f = self.cc.function(name)?;
            self.cc.launch(f, total.div_ceil(128), 1, 128, &mut args)?;
        }
        self.cc.sync()?;
        let iq = self.read_f32(self.diq, t * ih * id)?;
        let ki = self.read_f32(self.dpooli, nb * id)?;
        let w = self.read_f32(self.diw, t * ih)?;
        let scores = self.read_f32(self.dscores, t * nb)?;
        let mut sb = vec![0u8; t * d.index_topk * 4];
        self.cc.d2h(&mut sb, self.dsel)?;
        self.cc.sync()?;
        // SAFETY: d2h 완료 버퍼 재해석(길이·정렬 일치).
        let sel =
            unsafe { std::slice::from_raw_parts(sb.as_ptr() as *const i32, t * d.index_topk) }
                .to_vec();
        Ok((iq, ki, w, scores, sel))
    }

    /// HCA 밀도 선택 — attn.rs compress_idx_dense L175-179(호스트 산출).
    fn dense_sel(&self, t: usize, ratio: usize, offset: usize) -> Vec<Vec<i32>> {
        (0..t)
            .map(|ti| (0..(ti + 1) / ratio).map(|b| (b + offset) as i32).collect())
            .collect()
    }

    /// 프리필 윈도우 인덱스 — attn.rs window_idx_prefill L160-165(호스트 산출).
    fn window_idx(&self, ti: usize, win: usize) -> Vec<i32> {
        let start = ti.saturating_sub(win - 1);
        (start..=ti).map(|v| v as i32).collect()
    }

    /// 스파스 어텐션 + 출력 비회전 — attn.rs sparse_attn_one L531-580 +
    /// attention_forward L646-658. q [t×nh×hd]·kv_all [(t+nb)×hd]·
    /// sel [t][topk](-1 패드 허용). 반환 o [t×nh×hd](비회전 완료).
    /// neg=SinkDropped 면 nosink 쌍둥이(원장 17호).
    pub fn stage_sparse(
        &mut self,
        il: usize,
        q: &[f32],
        kv_all: &[f32],
        sel: &[Vec<i32>],
        neg: Option<Ds4Neg>,
    ) -> Result<Vec<f32>, String> {
        let d = self.dims.clone();
        let (nh, hd) = (d.n_heads, d.head_dim);
        let t = q.len() / (nh * hd);
        let rows_tot = kv_all.len() / hd;
        let lay = self.layer(il)?;
        let dsink = lay.sink;
        let (drope, dhalf, dlen) = self.rope_of(il)?;
        if dlen < t {
            return Err(format!("ds4: rope 표 len {dlen} < t {t}"));
        }
        if sel.len() != t {
            return Err(format!("ds4: sel.len {} != t {t}", sel.len()));
        }
        let stride = d.window + d.index_topk;
        if stride > DS4_SPARSE_EMAX {
            return Err(format!(
                "ds4: 스파스 행 상한 {DS4_SPARSE_EMAX} < win+topk {stride}"
            ));
        }
        self.ensure(t)?;
        let _g = self.cc.guard()?;
        Exl3CudaDecoder::h2d_chunked(&self.cc, self.dq, &f32_bytes(q))?;
        Exl3CudaDecoder::h2d_chunked(&self.cc, self.dkvall, &f32_bytes(kv_all))?;
        // idxs 조립(호스트 — 윈도우 ++ sel, -1 패드).
        let mut idxs = vec![-1i32; t * stride];
        for (ti, srow) in sel.iter().enumerate() {
            let win = self.window_idx(ti, d.window);
            let mut n = 0usize;
            for &v in win.iter().chain(srow.iter()) {
                if n >= stride {
                    return Err(format!("ds4: t={ti} 인덱스 수 초과(win+sel > {stride})"));
                }
                if (v as i64) < -1 || v as usize >= rows_tot {
                    return Err(format!("ds4: t={ti} 인덱스 {v} 범위 외(행 {rows_tot})"));
                }
                idxs[ti * stride + n] = v;
                n += 1;
            }
        }
        Exl3CudaDecoder::h2d_chunked(&self.cc, self.didx, &i32_bytes(&idxs))?;
        {
            let total = (t * nh) as u32;
            let (mut a0, mut a1, mut a2, mut a3) = (self.dq, self.dkvall, self.didx, dsink);
            let (mut a4, mut a5, mut a6, mut a7) = (self.do_, t as i32, nh as i32, hd as i32);
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
            let name = if neg == Some(Ds4Neg::SinkDropped) {
                "llm170_ds4_sparse_attn_nosink"
            } else {
                "llm170_ds4_sparse_attn"
            };
            let f = self.cc.function(name)?;
            self.cc.launch(f, total.div_ceil(64), 1, 64, &mut args)?;
        }
        {
            let total = (t * nh) as u32;
            let (mut a0, mut a1, mut a2, mut a3) = (self.do_, t as i32, nh as i32, hd as i32);
            let (mut a4, mut a5, mut a6) = (d.rope_head_dim as i32, drope, dhalf as i32);
            let mut args: [*mut std::ffi::c_void; 7] = [
                (&mut a0) as *mut _ as *mut _,
                (&mut a1) as *mut _ as *mut _,
                (&mut a2) as *mut _ as *mut _,
                (&mut a3) as *mut _ as *mut _,
                (&mut a4) as *mut _ as *mut _,
                (&mut a5) as *mut _ as *mut _,
                (&mut a6) as *mut _ as *mut _,
            ];
            let f = self.cc.function("llm170_ds4_derot")?;
            self.cc.launch(f, total.div_ceil(64), 1, 64, &mut args)?;
        }
        self.cc.sync()?;
        self.read_f32(self.do_, t * nh * hd)
    }

    /// 그룹 출력 — attn.rs attention_output L582-599. o [t×nh×hd] → y [t×dim].
    pub fn stage_output(&mut self, il: usize, o: &[f32]) -> Result<Vec<f32>, String> {
        let d = self.dims.clone();
        let (nh, hd, g, r) = (d.n_heads, d.head_dim, d.o_groups, d.o_lora_rank);
        let t = o.len() / (nh * hd);
        let gdim = nh * hd / g;
        let lay = self.layer(il)?;
        let (dwo_a, dwo_b) = (lay.wo_a.clone(), lay.wo_b);
        self.ensure(t)?;
        let _g = self.cc.guard()?;
        Exl3CudaDecoder::h2d_chunked(&self.cc, self.do_, &f32_bytes(o))?;
        for gi in 0..g {
            // o 열 슬라이스 [t, gi·gdim .. +gdim) — x_stride=nh·hd(strided gemm).
            let xoff = self.do_ + (gi * gdim) as u64 * 4;
            let yoff = self.dlat + (gi * r) as u64 * 4;
            self.k_gemm(xoff, dwo_a[gi], yoff, t, gdim, r, nh * hd, g * r)?;
        }
        self.k_bf16(self.dlat, t * g * r)?;
        self.k_fp8(self.dlat, t, g * r, g * r, 128)?;
        self.k_gemm(self.dlat, dwo_b, self.dy, t, g * r, d.dim, g * r, d.dim)?;
        self.k_bf16(self.dy, t * d.dim)?;
        self.cc.sync()?;
        self.read_f32(self.dy, t * d.dim)
    }

    /// 프리필 어텐션 전체 — attn.rs attention_forward L601-660 미러.
    /// x [t×dim] → 출력 [t×dim]. neg 는 음성대조 경로(검증층 전용).
    pub fn attention_forward_ex(
        &mut self,
        il: usize,
        x: &[f32],
        neg: Option<Ds4Neg>,
    ) -> Result<Vec<f32>, String> {
        let d = self.dims.clone();
        let t = x.len() / d.dim;
        let kind = d.kind(il)?;
        let (c_q, q) = self.stage_project_q(il, x)?;
        let kv = self.stage_project_kv(il, x)?;
        let mut kv_all = kv;
        let comp_sel: Vec<Vec<i32>> = match kind {
            Ds4Kind::Swa => vec![Vec::new(); t],
            Ds4Kind::Csa | Ds4Kind::Hca => {
                let entries = self.stage_compress(il, x, neg)?;

                kv_all.extend(entries);
                let offset = t;
                match kind {
                    Ds4Kind::Csa => {
                        let (_, _, _, _, sel) = self.stage_indexer(il, &c_q, x, neg)?;
                        sel.chunks(d.index_topk)
                            .map(|row| row.iter().copied().filter(|&v| v >= 0).collect())
                            .collect()
                    }
                    Ds4Kind::Hca => self.dense_sel(t, d.ratio(il), offset),
                    Ds4Kind::Swa => unreachable!("위에서 처리"),
                }
            }
        };
        let o = self.stage_sparse(il, &q, &kv_all, &comp_sel, neg)?;
        self.stage_output(il, &o)
    }

    /// 프리필 어텐션 전체(정상 경로 — neg 없음).
    pub fn attention_forward(&mut self, il: usize, x: &[f32]) -> Result<Vec<f32>, String> {
        self.attention_forward_ex(il, x, None)
    }
}
