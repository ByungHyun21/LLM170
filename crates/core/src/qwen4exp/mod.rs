//! qwen4exp (Qwen3.8-Flash-Next 125B-A6B) CPU 참조 엔진.
//!
//! 그래프 배선: `~/local_llm-runtimes/qwen4exp/src/models/qwen4exp.cpp` + llama-graph.cpp
//! build_moe_ffn (2026-08-30 판). 스펙: docs/models/qwen4exp.md.
//! - 48층 = 12×(3×GDN→MoE + 1×QSA→MoE) — compress_ratios[il]==0이 GDN, 4가 QSA
//! - 잔차 = hyper-connection 4스트림 (모든 norm 대체): mix = grouped RMSNorm(γ=(1+w) 폴딩)
//!   ·저랭크 게이트·스트림 평균 / combine = s += out·2σ(inject/4)
//! - GDN: qwen35 동일 모듈, z-gate가 sigmoid
//! - QSA: 게이트드 GQA(24Q/2KV/256d, IMROPE 64) + indexer(4q/1k/128d) 블록(4) 상위 top_k=2048 마스크
//! - MoE: 512전문가 top-10(softmax→정규화) + shared 1개(sigmoid 게이트)
//! - PLE(blk.1): n-gram 해시(호스트 u64) → 16행×160 gather → key/value → sgn√|s| 게이트
//!   → 4스트림 방송 → dilated(3) depthwise conv(4) → 잔차 2경로. 테이블 26.8GiB mmap 오프로드.
//! - 4-split GGUF: part1=메타 전용, parts2-4가 텐서 1224개 분산 보관.

pub mod layers;
pub mod stages;

use crate::matmul::Weight;
use crate::quant::dequant_row;
use llm170_gguf::{GgufFile, Value};
use memmap2::{Advice, Mmap, MmapOptions};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum Q4Error {
    MissingTensor(String),
    BadMeta(&'static str),
    Io(String),
}

impl std::fmt::Display for Q4Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Q4Error::MissingTensor(n) => write!(f, "missing tensor: {n}"),
            Q4Error::BadMeta(w) => write!(f, "bad metadata: {w}"),
            Q4Error::Io(e) => write!(f, "io: {e}"),
        }
    }
}

impl std::error::Error for Q4Error {}

/// qwen4exp 하이퍼파라미터 (GGUF 메타 실측값).
#[derive(Debug, Clone)]
pub struct Hparams4 {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub n_rot: usize,
    pub rope_sections: [i32; 4],
    pub rope_base: f32,
    pub eps: f32,
    pub vocab: usize,
    // GDN
    pub d_inner: usize,
    pub dt_rank: usize,
    pub d_state: usize,
    pub n_group: usize,
    pub conv_k: usize,
    // MoE
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub n_ff_shared: usize,
    // hyper-connection
    pub hc: usize,
    pub hc_low_rank: usize,
    // QSA indexer
    pub idx_heads: usize,
    pub idx_dim: usize,
    pub idx_top_k: usize,
    /// Arc 슬라이스 — hp.clone()이 스택 복사(핫패스 스테이지당 1회씩 복제
    /// 중이라 할당 제거, 2026-09-01).
    pub compress: std::sync::Arc<[i32]>, // [n_layer]
    // PLE
    pub ple_layers: std::sync::Arc<[usize]>,
    pub ple_ngram: usize,
    pub ple_heads_per_ngram: usize,
    pub ple_conv_k: usize,
    pub ple_head_dim: usize, // embedding_length_per_layer_input
    pub ple_multipliers: std::sync::Arc<[u64]>,
    pub ple_head_offsets: std::sync::Arc<[u64]>,
    pub ple_head_vocab_sizes: std::sync::Arc<[u64]>,
    pub ple_eos: u32,
    pub ple_image: u32,
}

impl Hparams4 {
    pub fn is_recr(&self, il: usize) -> bool {
        self.compress[il] == 0
    }
    pub fn is_ple(&self, il: usize) -> bool {
        self.ple_layers.contains(&il)
    }
    /// kq 스케일 — 1/√head_dim (f_attention_scale 기본 0).
    pub fn kq_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
}

/// 4-split 로드 모델. 가중치는 파트별 mmap — PLE 테이블 파트는 MADV_RANDOM.
pub struct Model4 {
    pub hp: Hparams4,
    pub parts: Vec<PartMap>,
    /// 텐서 이름 → 파트 인덱스·텐서 정보 인덱스
    index: HashMap<String, (usize, usize)>,
    /// f32 소형 벡터(norm·a·dt_bias·conv 등) 디양자화 캐시 — 호출마다
    /// 전량 재디양자화하던 것을 1회로 (Send, 단일 엔진 스레드에서만 접근).
    f32_cache: std::cell::RefCell<HashMap<String, Vec<f32>>>,
    pub token_pieces: Vec<String>,
    pub eos: u32,
}

pub struct PartMap {
    pub path: PathBuf,
    pub data_offset: u64,
    pub tensors: Vec<llm170_gguf::TensorInfo>,
    pub mmap: Mmap,
}

/// File::open 대기 버전 — mmap 핸들 열기도 윈도우 대기.
fn open_file_wait(path: &Path) -> Result<std::fs::File, Q4Error> {
    let wait: u64 = std::env::var("LLM170_OPEN_WAIT_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(wait);
    loop {
        match std::fs::File::open(path) {
            Ok(f) => return Ok(f),
            Err(e) => {
                if wait == 0 || std::time::Instant::now() >= deadline {
                    return Err(Q4Error::Io(e.to_string()));
                }
                std::thread::sleep(std::time::Duration::from_millis(1000));
            }
        }
    }
}

impl Model4 {
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let g1 = GgufFile::open(path)?;
        if g1.kv_str("general.architecture").as_deref() != Some("qwen4exp") {
            return Err("not a qwen4exp model".into());
        }
        let split_count = g1.kv_u64("split.count").unwrap_or(1) as u32;
        let mut paths = vec![path.to_path_buf()];
        if split_count > 1 {
            // 파일명 패턴: ...-00001-of-00004.gguf — 보존 필수(스펙)
            let stem = path
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or("bad path")?;
            for i in 2..=split_count {
                let pat = format!("-{:05}-of-{:05}", 1, split_count);
                let rep = format!("-{:05}-of-{:05}", i, split_count);
                let name = stem.replace(&pat, &rep);
                paths.push(path.with_file_name(name));
            }
        }

        // 파트 mmap + 텐서 색인 (part1은 메타 전용 — 텐서 0)
        let mut parts = Vec::with_capacity(paths.len());
        let mut index = HashMap::new();
        for (pi, p) in paths.iter().enumerate() {
            let g = if pi == 0 { &g1 } else { &GgufFile::open(p)? };
            let file = std::fs::File::open(p).map_err(|e| Q4Error::Io(e.to_string()))?;
            // SAFETY: 읽기 전용 무게 매핑 — 수정하지 않는다
            // SAFETY: 읽기 전용 매핑 — 수정하지 않는다
            let mmap = unsafe { MmapOptions::new().map(&file)? };
            for (ti, t) in g.tensors.iter().enumerate() {
                index.insert(t.name.clone(), (pi, ti));
            }
            if pi > 0 {
                // PLE 랜덤 GET_ROWS — 순차 스트리밍과 반대 어드바이스.
                // 테이블이 이 파트에 있는지 확인해 Random 지정 (run.sh 원형).
                let has_ple = g.tensors.iter().any(|t| t.name == "per_layer_token_embd.weight");
                if has_ple {
                    // SAFETY: 어드바이스는 힌트 — 잘못돼도 안전
                    unsafe { mmap.advise(Advice::Random) }.ok();
                }
            }
            parts.push(PartMap {
                path: p.clone(),
                data_offset: g.data_offset,
                tensors: g.tensors.clone(),
                mmap,
            });
        }

        let hp = Self::hparams(&g1)?;
        let mut token_pieces = Vec::new();
        if let Some(Value::Array(_, a)) = g1.kv("tokenizer.ggml.tokens") {
            for t in a {
                token_pieces.push(t.as_str().unwrap_or("").to_string());
            }
        }
        let vocab = parts
            .iter()
            .find_map(|p| p.tensors.iter().find(|t| t.name == "token_embd.weight").map(|t| t.ne[1] as usize))
            .unwrap_or(0);
        let hp = Hparams4 { vocab, ..hp };
        let eos = hp.ple_eos;
        let m = Model4 {
            hp,
            parts,
            index,
            f32_cache: std::cell::RefCell::new(HashMap::new()),
            token_pieces,
            eos,
        };
        for name in ["token_embd.weight", "output.weight"] {
            m.w(name).ok_or(Q4Error::MissingTensor(name.into()))?;
        }
        Ok(m)
    }

    fn hparams(g: &GgufFile) -> Result<Hparams4, Box<dyn std::error::Error>> {
        let u = |k: &str| g.arch_kv_u64(k);
        let s32 = |k: &str| -> Option<Vec<i32>> {
            g.arch_kv(k).and_then(|v| match v {
                Value::Array(_, a) => a
                    .iter()
                    .map(|v| match v {
                        Value::I32(x) => Some(*x),
                        _ => None,
                    })
                    .collect::<Option<Vec<_>>>(),
                _ => None,
            })
        };
        let u64a = |k: &str| -> Option<Vec<u64>> {
            g.arch_kv(k).and_then(|v| match v {
                Value::Array(_, a) => a
                    .iter()
                    .map(|v| match v {
                        Value::U64(x) => Some(*x),
                        _ => None,
                    })
                    .collect::<Option<Vec<_>>>(),
                _ => None,
            })
        };
        let compress =
            s32("attention.compress_ratios").ok_or(Q4Error::BadMeta("compress_ratios"))?;
        let ple_layers = s32("ple.layers").ok_or(Q4Error::BadMeta("ple.layers"))?;
        let ple_multipliers =
            u64a("ple.layer_multipliers").ok_or(Q4Error::BadMeta("layer_multipliers"))?;
        let ple_head_offsets =
            u64a("ple.head_offsets").ok_or(Q4Error::BadMeta("head_offsets"))?;
        let ple_head_vocab_sizes =
            u64a("ple.head_vocab_sizes").ok_or(Q4Error::BadMeta("head_vocab_sizes"))?;
        let sections = s32("rope.dimension_sections").unwrap_or([11, 11, 10, 0].to_vec());
        let n_layer = u("block_count").ok_or(Q4Error::BadMeta("block_count"))? as usize;
        if compress.len() != n_layer {
            return Err(Q4Error::BadMeta("compress_ratios 길이").into());
        }
        Ok(Hparams4 {
            n_layer,
            n_embd: u("embedding_length").ok_or(Q4Error::BadMeta("embedding_length"))? as usize,
            n_head: u("attention.head_count").ok_or(Q4Error::BadMeta("head_count"))? as usize,
            n_kv: u("attention.head_count_kv").ok_or(Q4Error::BadMeta("head_count_kv"))? as usize,
            head_dim: u("attention.key_length").ok_or(Q4Error::BadMeta("key_length"))? as usize,
            n_rot: u("rope.dimension_count").ok_or(Q4Error::BadMeta("dimension_count"))? as usize,
            rope_sections: [sections[0], sections[1], sections[2], sections[3]],
            rope_base: g
                .arch_kv("rope.freq_base")
                .and_then(Value::as_f64)
                .unwrap_or(1e7) as f32,
            eps: g
                .arch_kv("attention.layer_norm_rms_epsilon")
                .and_then(Value::as_f64)
                .unwrap_or(1e-6) as f32,
            vocab: 0,
            d_inner: u("ssm.inner_size").ok_or(Q4Error::BadMeta("inner_size"))? as usize,
            dt_rank: u("ssm.time_step_rank").ok_or(Q4Error::BadMeta("time_step_rank"))? as usize,
            d_state: u("ssm.state_size").ok_or(Q4Error::BadMeta("state_size"))? as usize,
            n_group: u("ssm.group_count").ok_or(Q4Error::BadMeta("group_count"))? as usize,
            conv_k: u("ssm.conv_kernel").ok_or(Q4Error::BadMeta("conv_kernel"))? as usize,
            n_expert: u("expert_count").ok_or(Q4Error::BadMeta("expert_count"))? as usize,
            n_expert_used: u("expert_used_count").ok_or(Q4Error::BadMeta("expert_used_count"))? as usize,
            n_ff_exp: u("expert_feed_forward_length").ok_or(Q4Error::BadMeta("expert_ffn"))? as usize,
            n_ff_shared: u("expert_shared_feed_forward_length").ok_or(Q4Error::BadMeta("expert_shared_ffn"))? as usize,
            hc: u("hyper_connection.count").ok_or(Q4Error::BadMeta("hc.count"))? as usize,
            hc_low_rank: u("hyper_connection.low_rank").ok_or(Q4Error::BadMeta("hc.low_rank"))? as usize,
            idx_heads: u("attention.indexer.head_count").ok_or(Q4Error::BadMeta("idx.heads"))? as usize,
            idx_dim: u("attention.indexer.key_length").ok_or(Q4Error::BadMeta("idx.dim"))? as usize,
            idx_top_k: u("attention.indexer.top_k").ok_or(Q4Error::BadMeta("idx.top_k"))? as usize,
            compress: compress.into(),
            ple_layers: ple_layers.iter().map(|&v| v as usize).collect::<Vec<_>>().into(),
            ple_ngram: u("ple.ngram_size").ok_or(Q4Error::BadMeta("ngram_size"))? as usize,
            ple_heads_per_ngram: u("ple.heads_per_ngram").ok_or(Q4Error::BadMeta("heads_per_ngram"))? as usize,
            ple_conv_k: u("ple.conv_kernel").ok_or(Q4Error::BadMeta("ple.conv"))? as usize,
            ple_head_dim: u("embedding_length_per_layer_input").ok_or(Q4Error::BadMeta("per_layer_input"))? as usize,
            ple_multipliers: ple_multipliers.into(),
            ple_head_offsets: ple_head_offsets.into(),
            ple_head_vocab_sizes: ple_head_vocab_sizes.into(),
            ple_eos: u("ple.eos_token_id").ok_or(Q4Error::BadMeta("ple.eos"))? as u32,
            ple_image: u("ple.image_token_id").unwrap_or(0) as u32,
        })
    }

    /// 무게 뷰 — split 파트 걸침 해석.
    pub fn w(&self, name: &str) -> Option<Weight<'_>> {
        let (pi, ti) = *self.index.get(name)?;
        let part = &self.parts[pi];
        let t = &part.tensors[ti];
        let (start, end) = t.file_range(part.data_offset)?;
        Some(Weight {
            data: &part.mmap[start as usize..end as usize],
            ty: t.ty,
            n_in: t.ne[0],
            n_out: t.ne[1] * t.ne[2] * t.ne[3],
        })
    }

    pub fn w4(&self, name: &str) -> Result<Weight<'_>, Q4Error> {
        self.w(name).ok_or_else(|| Q4Error::MissingTensor(name.into()))
    }

    /// f32 벡터 텐서 디양자화 — 캐시. 대상(norm 가중치 등)은 결정적이라
    /// 첫 호출 1회 디양자화 후 재사용 (수치 불변).
    pub fn f32_vec4(&self, name: &str) -> Result<Vec<f32>, Q4Error> {
        if let Some(v) = self.f32_cache.borrow().get(name) {
            return Ok(v.clone());
        }
        let v = self.w4(name)?.dequant_f32_vec();
        self.f32_cache.borrow_mut().insert(name.to_string(), v.clone());
        Ok(v)
    }

    /// 전문가 스택 3D 텐서 [ff, n_embd, n_expert]의 전문가 e 슬라이스 뷰.
    pub fn expert_w(&self, name: &str, e: usize) -> Result<Weight<'_>, Q4Error> {
        let (pi, ti) = *self.index.get(name).ok_or_else(|| Q4Error::MissingTensor(name.into()))?;
        let part = &self.parts[pi];
        let t = &part.tensors[ti];
        let (blck, bsize) = t.ty.block_info();
        let per_expert_bytes = (t.ne[0] / blck * bsize) as usize * t.ne[1] as usize;
        let (start, _end) = t.file_range(part.data_offset).ok_or(Q4Error::BadMeta("expert range"))?;
        let s = start as usize + e * per_expert_bytes;
        Ok(Weight {
            data: &part.mmap[s..s + per_expert_bytes],
            ty: t.ty,
            n_in: t.ne[0],
            n_out: t.ne[1],
        })
    }

    /// PLE 행 gather: 16행 × ple_head_dim을 1차원 f32(2560)로 — get_rows+flatten.
    /// 행은 (head h, row r∈[0,160)) → 전역 행 = h*160 + r (레이아웃: [160, 320M] 행 우선,
    /// head들이 0..320M 행을 오프셋으로 분할 — 변환 스크립트 flatten 축).
    pub fn ple_gather(&self, rows: &[u32], out: &mut [f32]) -> Result<(), Q4Error> {
        let table = self.w4("per_layer_token_embd.weight")?;
        let hd = self.hp.ple_head_dim;
        debug_assert_eq!(out.len(), rows.len() * hd);
        let (blck, bsize) = table.ty.block_info();
        for (hi, &row) in rows.iter().enumerate() {
            let base = row as u64 * hd as u64;
            let byte_off = base / blck * bsize;
            let k = base % blck;
            // 행 시작이 블록 경계(32원소 iq4_nl)에 정렬 — hd=160, blck=32 → 항상 정렬.
            let mut tmp = [0.0f32; 512];
            let n_blocks = (hd as u64).div_ceil(blck) as usize;
            for b in 0..n_blocks {
                dequant_row(
                    table.ty,
                    &table.data[byte_off as usize + b * bsize as usize..],
                    0,
                    blck,
                    &mut tmp[b * blck as usize..(b + 1) * blck as usize],
                );
            }
            out[hi * hd..(hi + 1) * hd].copy_from_slice(&tmp[..hd]);
            let _ = k;
        }
        Ok(())
    }

    /// 표면형 근사 디토크.
    pub fn piece(&self, tok: u32) -> String {
        self.token_pieces
            .get(tok as usize)
            .map(String::as_str)
            .unwrap_or("")
            .replace('Ġ', " ")
            .replace('Ċ', "\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL: &str = "/home/yoon/local_llm/models/qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf";

    /// 실측 모델 로더 계약: split 4파일 병합·hparams·PLE 행 gather·전문가 슬라이스.
    /// 파일 없으면 skip (real_models.rs 관례).
    #[test]
    fn loader_split_contract() {
        if !std::path::Path::new(MODEL).exists() {
            eprintln!("skip: {MODEL} 없음");
            return;
        }
        let m = Model4::load(std::path::Path::new(MODEL)).expect("load");
        let hp = &m.hp;
        assert_eq!((hp.n_layer, hp.n_embd, hp.n_head, hp.n_kv, hp.head_dim), (48, 2560, 24, 2, 256));
        assert_eq!((hp.n_expert, hp.n_expert_used, hp.n_ff_exp), (512, 10, 640));
        assert_eq!((hp.hc, hp.hc_low_rank), (4, 320));
        assert_eq!((hp.idx_heads, hp.idx_dim, hp.idx_top_k), (4, 128, 2048));
        assert_eq!(hp.compress.iter().filter(|&&c| c == 4).count(), 12);
        assert!(hp.is_ple(1));
        assert_eq!(hp.ple_head_vocab_sizes.len(), 16);
        // 전체 텐서 병합 수 = split.tensors.count 1224
        let n = m.parts.iter().map(|p| p.tensors.len()).sum::<usize>();
        assert_eq!(n, 1224, "split.tensors.count");
        // 텐서 뷰: 대표 몇 개
        let w = m.w("token_embd.weight").expect("embd");
        assert_eq!((w.n_in, w.n_out), (2560, 248320));
        let ple = m.w("per_layer_token_embd.weight").expect("ple");
        assert_eq!(ple.ty, llm170_gguf::GgmlType::Iq4Nl);
        // PLE gather: 행 3개 — 유한값·결정성
        let mut out = vec![0.0f32; 3 * hp.ple_head_dim];
        m.ple_gather(&[0, 1, 1_000_000], &mut out);
        assert!(out.iter().all(|v| v.is_finite()));
        let mut out2 = out.clone();
        m.ple_gather(&[0, 1, 1_000_000], &mut out2);
        assert_eq!(out, out2, "gather 결정성");
        // 전문가 슬라이스: 형상 [640, 2560]
        let e0 = m.expert_w("blk.0.ffn_up_exps.weight", 0).expect("expert");
        assert_eq!((e0.n_in, e0.n_out), (2560, 640));
        let e511 = m.expert_w("blk.0.ffn_up_exps.weight", 511).expect("expert 511");
        assert_eq!((e511.n_in, e511.n_out), (2560, 640));
    }
}
