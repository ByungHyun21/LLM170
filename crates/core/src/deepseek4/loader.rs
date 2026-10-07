//! EXL3 픽스처 로더 — 오프셋 읽기 전용(아카이브 110GB, 전체 적재 금지).
//!
//! `llm170_exl3::StArchive`(safetensors 샤드 인덱스) + `Exl3Linear`
//! (trellis 참조 디코드, plans/118 §6 검증 파이프라인) 재사용 —
//! qwen4exp `Model4`의 mmap 오프셋 패턴을 EXL3로 이식(plans/130 B2).
//! 텐서명 계약(EXL3 Vision-Exp 실측):
//! - `layers.{il}.attn.{wq_a,wq_b,wkv,wo_b}` / `wo_a.slice.{0..7}` trellis 쿼드
//! - `layers.{il}.attn.{q_norm,kv_norm,compressor.norm}.weight` BF16
//! - `layers.{il}.attn.attn_sink` F32[64]
//! - `layers.{il}.attn.compressor.{wkv,wgate}` trellis / `ape` F32 / norm BF16
//! - `layers.{il}.attn.indexer.*` (CSA=ratio 4 층만)
//! - `layers.{il}.hc_{attn,ffn}_{fn,base,scale}` F32
//! - `layers.{il}.ffn.gate.{weight F16, bias F16, tid2eid I64(해시 층)}`
//!   — `bias_vl`(F32)은 Vision-Exp 전용 텍스트 경로 미사용(무시).
//! - `layers.{il}.ffn.experts.{e}.{w1,w2,w3}` trellis(지연 적재),
//!   `shared_experts.{w1,w2,w3}`
//! - `embed.weight` BF16[129280,4096] 행 슬라이스 읽기, `head` trellis,
//!   `hc_head_{fn,base,scale}`, `norm.weight`
//! - `mtp.0.main_proj`(trellis)·`main_norm`, `mtp.2.markov_head.*`,
//!   `mtp.2.confidence_head.proj`, `mtp.2.hc_head_*`·`norm`
//!
//! trellis 디양자화는 128행 스트립 단위 스레드 병렬(결정적 — 스트립별
//! 독립 기입, 블록 내부는 참조 디코드 순서 그대로).

use crate::deepseek4::config::{Deepseek4Config, LayerKind};
use crate::deepseek4::layers::BlockWeights;
use crate::deepseek4::stages::attn::{AttnWeights, CompressorWeights, IndexerWeights, LayerAttn};
use crate::deepseek4::stages::hc::HcParams;
use crate::deepseek4::stages::moe::{ExpertWeights, GateWeights};
use llm170_exl3::{Exl3Linear, StArchive, StDtype};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// f16(Half) → f32 — core에 half 의존 없이 자체 구현(e4m3/e2m1과 별개).
#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let e = ((bits >> 10) & 0x1F) as u32;
    let m = (bits & 0x3FF) as u32;
    let v = if e == 0 {
        // 비정규: m/1024·2^-14.
        (m as f32) * (2.0f64).powi(-24) as f32
    } else if e == 31 {
        return f32::NAN;
    } else {
        f32::from_bits(sign | ((e - 15 + 127) << 23) | (m << 13))
    };
    v.copysign(f32::from_bits(sign))
}

/// bf16 → f32 (비트 확장, 정확).
#[inline]
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// 디양자화 완료 선형 — w는 k-major [k][n] f32.
#[derive(Debug, Clone)]
pub struct LinearW {
    pub k: usize,
    pub n: usize,
    pub w: Vec<f32>,
}

#[derive(Debug)]
pub enum Ds4Error {
    Missing(String),
    BadTensor(String),
    Exl3(llm170_exl3::Exl3Error),
    Io(std::io::Error),
}

impl std::fmt::Display for Ds4Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Ds4Error::Missing(n) => write!(f, "missing tensor: {n}"),
            Ds4Error::BadTensor(w) => write!(f, "bad tensor: {w}"),
            Ds4Error::Exl3(e) => write!(f, "exl3: {e}"),
            Ds4Error::Io(e) => write!(f, "io: {e}"),
        }
    }
}

impl std::error::Error for Ds4Error {}

impl From<llm170_exl3::Exl3Error> for Ds4Error {
    fn from(e: llm170_exl3::Exl3Error) -> Self {
        Ds4Error::Exl3(e)
    }
}

impl From<std::io::Error> for Ds4Error {
    fn from(e: std::io::Error) -> Self {
        Ds4Error::Io(e)
    }
}

type R<T> = Result<T, Ds4Error>;

/// EXL3 디렉터리 로더 — config.json + 샤드 아카이브.
pub struct Ds4Loader {
    pub dir: PathBuf,
    pub cfg: Deepseek4Config,
    ar: StArchive,
}

impl Ds4Loader {
    pub fn open(dir: &Path) -> R<Self> {
        let text = std::fs::read_to_string(dir.join("config.json"))?;
        let cfg = Deepseek4Config::from_json(&text).map_err(Ds4Error::BadTensor)?;
        let ar = StArchive::open(dir)?;
        Ok(Ds4Loader {
            dir: dir.to_path_buf(),
            cfg,
            ar,
        })
    }

    /// 바이트 → f32 (dtype 분기).
    fn conv(raw: &[u8], dt: StDtype) -> R<Vec<f32>> {
        Ok(match dt {
            StDtype::F32 => raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
            StDtype::Bf16 => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| bf16_to_f32(u16::from_le_bytes(*c)))
                .collect(),
            StDtype::F16 => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f16_to_f32(u16::from_le_bytes(*c)))
                .collect(),
            other => {
                return Err(Ds4Error::BadTensor(format!(
                    "plain f32 불가 dtype {other:?}"
                )));
            }
        })
    }

    /// plain 텐서 전체 → f32.
    pub fn plain_f32(&self, name: &str) -> R<Vec<f32>> {
        let e = self
            .ar
            .entry(name)
            .ok_or_else(|| Ds4Error::Missing(name.into()))?;
        let raw = self.ar.read(name)?;
        Self::conv(&raw, e.dtype)
    }

    /// plain 텐서 행 범위 부분 읽기 — embed 슬라이스·markov 행 조회용.
    /// 전체 적재 없이 해당 행만 pread한다(아카이브 계약).
    pub fn plain_rows_f32(&self, name: &str, row_lo: u64, row_hi: u64) -> R<Vec<f32>> {
        let e = self
            .ar
            .entry(name)
            .ok_or_else(|| Ds4Error::Missing(name.into()))?;
        if e.shape.len() != 2 {
            return Err(Ds4Error::BadTensor(format!("{name}: 2차원 아님")));
        }
        let cols = e.shape[1];
        let row_bytes = cols * e.dtype.nbytes();
        let mut f = std::fs::File::open(
            self.ar
                .shard_paths()
                .get(e.shard)
                .ok_or_else(|| Ds4Error::BadTensor("샤드 경로 없음".into()))?,
        )?;
        let base = self
            .ar
            .shard_data_base(e.shard)
            .ok_or_else(|| Ds4Error::BadTensor("샤드 베이스 없음".into()))?;
        f.seek(SeekFrom::Start(base + e.begin + row_lo * row_bytes))?;
        let mut raw = vec![0u8; ((row_hi - row_lo) * row_bytes) as usize];
        f.read_exact(&mut raw)?;
        Self::conv(&raw, e.dtype)
    }

    /// tid2eid 표 — [vocab·k] i64 (해시 층 3개분만 적재, 6.2MB).
    pub fn tid2eid(&self, il: usize) -> R<Vec<i64>> {
        let name = format!("layers.{il}.ffn.gate.tid2eid");
        let raw = self.ar.read(&name).map_err(Ds4Error::Exl3)?;
        if raw.len() % 8 != 0 {
            return Err(Ds4Error::BadTensor(format!("{name}: i64 정렬 아님")));
        }
        Ok(raw
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c))
            .collect())
    }

    /// trellis 선형 디양자화 — 스트립 병렬, [k][n] k-major f32.
    /// k·n은 128의 배수여야 한다(픽스처 실측 전부 해당).
    pub fn linear(&self, key: &str) -> R<LinearW> {
        let lin = Exl3Linear::load(&self.ar, key)?;
        let (k, n) = (lin.k, lin.n);
        if k % 128 != 0 || n % 128 != 0 {
            return Err(Ds4Error::BadTensor(format!(
                "{key}: k={k} n={n} — 128 배수 아님"
            )));
        }
        let mut w = vec![0.0f32; k * n];
        let view = lin.view();
        let threads = std::thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(1)
            .min(k / 128)
            .max(1);
        let per = (k / 128).div_ceil(threads) * 128; // 행 스트립 폭(128배수).
        std::thread::scope(|sc| {
            let v = &view;
            let mut off = 0usize;
            let mut rest = w.as_mut_slice();
            while off < k {
                let rows_here = per.min(k - off);
                let (part, tail) = rest.split_at_mut(rows_here * n);
                rest = tail;
                let k0 = off;
                sc.spawn(move || {
                    // 스트립을 128행 k-청크 단위로 — dequant_block_f64의
                    // k0%128==0 계약에 맞춘다(청크당 [128k × 128n]).
                    for (ci, chunk) in part.chunks_mut(128 * n).enumerate() {
                        let kc = k0 + ci * 128;
                        for n0 in (0..n).step_by(128) {
                            let blk = v.dequant_block_f64(kc, n0, 128, 128);
                            for i in 0..128 {
                                let src = &blk[i * 128..(i + 1) * 128];
                                let dst = &mut chunk[i * n + n0..i * n + n0 + 128];
                                for (j, &vv) in src.iter().enumerate() {
                                    dst[j] = vv as f32;
                                }
                            }
                        }
                    }
                });
                off += rows_here;
            }
        });
        Ok(LinearW { k, n, w })
    }

    /// 행 우선 [out][in] plain → k-major 전치 (gate.weight·markov_w2).
    pub fn plain_kmat(&self, name: &str) -> R<LinearW> {
        let e = self
            .ar
            .entry(name)
            .ok_or_else(|| Ds4Error::Missing(name.into()))?;
        if e.shape.len() != 2 {
            return Err(Ds4Error::BadTensor(format!("{name}: 2차원 아님")));
        }
        let (n, k) = (e.shape[0] as usize, e.shape[1] as usize);
        let src = self.plain_f32(name)?;
        let mut w = vec![0.0f32; k * n];
        for o in 0..n {
            for kk in 0..k {
                w[kk * n + o] = src[o * k + kk];
            }
        }
        Ok(LinearW { k, n, w })
    }

    /// hc 파라미터 — fn/base/scale F32.
    fn hc(&self, prefix: &str) -> R<HcParams> {
        Ok(HcParams {
            fns: self.plain_f32(&format!("{prefix}_fn"))?,
            base: self.plain_f32(&format!("{prefix}_base"))?,
            scale: self.plain_f32(&format!("{prefix}_scale"))?,
        })
    }

    fn comp_weights(
        &self,
        prefix: &str,
        head_dim: usize,
        ratio: usize,
        rotate: bool,
    ) -> R<CompressorWeights> {
        let wkv = self.linear(&format!("{prefix}.wkv"))?;
        let wgate = self.linear(&format!("{prefix}.wgate"))?;
        let expected = (1 + usize::from(ratio == 4)) * head_dim;
        if wkv.n != expected || wgate.n != expected {
            return Err(Ds4Error::BadTensor(format!(
                "{prefix}: wkv/wgate n={} 기대 {expected}",
                wkv.n
            )));
        }
        Ok(CompressorWeights {
            wkv: wkv.w,
            wgate: wgate.w,
            ape: self.plain_f32(&format!("{prefix}.ape"))?,
            norm: self.plain_f32(&format!("{prefix}.norm.weight"))?,
            head_dim,
            ratio,
            rotate,
        })
    }

    /// il번 블록 비-전문가 가중치 (전문가는 `expert`로 지연).
    pub fn block(&self, il: usize) -> R<BlockWeights> {
        let cfg = &self.cfg;
        let kind = cfg.kind(il);
        let ratio = cfg.ratio(il);
        let p = format!("layers.{il}");
        let wo_a = (0..cfg.o_groups)
            .map(|g| Ok(self.linear(&format!("{p}.attn.wo_a.slice.{g}"))?.w))
            .collect::<Result<Vec<_>, Ds4Error>>()?;
        let wq_a = self.linear(&format!("{p}.attn.wq_a"))?;
        let wq_b = self.linear(&format!("{p}.attn.wq_b"))?;
        let wkv = self.linear(&format!("{p}.attn.wkv"))?;
        let w = AttnWeights {
            wq_a: wq_a.w,
            q_norm: self.plain_f32(&format!("{p}.attn.q_norm.weight"))?,
            wq_b: wq_b.w,
            wkv: wkv.w,
            kv_norm: self.plain_f32(&format!("{p}.attn.kv_norm.weight"))?,
            sink: self.plain_f32(&format!("{p}.attn.attn_sink"))?,
            wo_a,
            wo_b: self.linear(&format!("{p}.attn.wo_b"))?.w,
        };
        let (comp, idx) = match kind {
            LayerKind::Swa => (None, None),
            LayerKind::Csa | LayerKind::Hca => {
                let comp = Some(self.comp_weights(
                    &format!("{p}.attn.compressor"),
                    cfg.head_dim,
                    ratio,
                    false,
                )?);
                let idx = if kind == LayerKind::Csa {
                    Some(IndexerWeights {
                        wq_b: self.linear(&format!("{p}.attn.indexer.wq_b"))?.w,
                        weights_proj: self
                            .plain_kmat(&format!("{p}.attn.indexer.weights_proj.weight"))?
                            .w,
                        comp: self.comp_weights(
                            &format!("{p}.attn.indexer.compressor"),
                            cfg.index_head_dim,
                            ratio,
                            true,
                        )?,
                    })
                } else {
                    None
                };
                (comp, idx)
            }
        };
        let is_hash = cfg.is_hash(il);
        let gate = GateWeights {
            weight: self.plain_kmat(&format!("{p}.ffn.gate.weight"))?.w,
            // 해시 층은 bias 무시(model.py Gate: hash → bias=None) —
            // bias_vl은 Vision-Exp 전용, 텍스트 경로 미사용.
            bias: if is_hash {
                None
            } else {
                Some(self.plain_f32(&format!("{p}.ffn.gate.bias"))?)
            },
            tid2eid: if is_hash {
                Some(self.tid2eid(il)?)
            } else {
                None
            },
        };
        let shared = self.shared(il)?;
        Ok(BlockWeights {
            attn_norm: self.plain_f32(&format!("{p}.attn_norm.weight"))?,
            ffn_norm: self.plain_f32(&format!("{p}.ffn_norm.weight"))?,
            hc_attn: self.hc(&format!("{p}.hc_attn"))?,
            hc_ffn: self.hc(&format!("{p}.hc_ffn"))?,
            attn: LayerAttn {
                kind,
                ratio,
                w,
                comp,
                idx,
            },
            gate,
            shared,
            is_hash,
        })
    }

    fn shared(&self, il: usize) -> R<ExpertWeights> {
        self.expert_at(&format!("layers.{il}.ffn.shared_experts"))
    }

    /// 라우티드 전문가 e (지연 디양자화).
    pub fn expert(&self, il: usize, e: usize) -> R<ExpertWeights> {
        self.expert_at(&format!("layers.{il}.ffn.experts.{e}"))
    }

    fn expert_at(&self, base: &str) -> R<ExpertWeights> {
        Ok(ExpertWeights {
            w1: self.linear(&format!("{base}.w1"))?.w,
            w2: self.linear(&format!("{base}.w2"))?.w,
            w3: self.linear(&format!("{base}.w3"))?.w,
        })
    }

    /// embed 행 슬라이스 — 토큰 id 목록 → [id][dim] f32(bf16 격자값).
    pub fn embed_rows(&self, ids: &[u32]) -> R<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(ids.len());
        for &id in ids {
            out.push(self.plain_rows_f32("embed.weight", id as u64, id as u64 + 1)?);
        }
        Ok(out)
    }

    /// 최종부 — norm·hc_head·head(스트립 gemv용 raw 선형).
    pub fn final_norm(&self) -> R<Vec<f32>> {
        self.plain_f32("norm.weight")
    }

    pub fn final_hc_head(&self) -> R<HcParams> {
        self.hc("hc_head")
    }

    pub fn head_linear(&self) -> R<Exl3Linear> {
        Exl3Linear::load(&self.ar, "head").map_err(Into::into)
    }

    /// DSpark 스테이지 0 — main_proj·main_norm (mtp.0).
    pub fn dspark_main(&self) -> R<(LinearW, Vec<f32>)> {
        Ok((
            self.linear("mtp.0.main_proj")?,
            self.plain_f32("mtp.0.main_norm.weight")?,
        ))
    }

    /// DSpark 종결부 — mtp.{last} hc_head·norm·markov·confidence.
    pub fn dspark_head_parts(&self) -> R<(HcParams, Vec<f32>, LinearW, Vec<f32>)> {
        let last = self.cfg.n_mtp_layers.saturating_sub(1);
        let p = format!("mtp.{last}");
        Ok((
            self.hc(&format!("{p}.hc_head"))?,
            self.plain_f32(&format!("{p}.norm.weight"))?,
            self.plain_kmat(&format!("{p}.markov_head.markov_w2.weight"))?,
            self.plain_f32(&format!("{p}.confidence_head.proj.weight"))?,
        ))
    }

    /// markov_w1 행 조회 — 토큰 id → [rank] f32.
    pub fn markov_w1_row(&self, id: u32) -> R<Vec<f32>> {
        let rank = self.cfg.dspark_markov_rank as u64;
        let last = self.cfg.n_mtp_layers.saturating_sub(1);
        let row = self.plain_rows_f32(
            &format!("mtp.{last}.markov_head.markov_w1.weight"),
            id as u64,
            id as u64 + 1,
        )?;
        if row.len() as u64 != rank {
            return Err(Ds4Error::BadTensor(format!(
                "markov_w1 행 {} ≠ rank {rank}",
                row.len()
            )));
        }
        Ok(row)
    }

    /// 아카이브 엔트리 수 — 프루브 검증용.
    pub fn entry_count(&self) -> usize {
        self.ar.entries().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 픽스처 없으면 skip(관례). 실측 계약: 엔트리 수·plain/linear 형상.
    #[test]
    fn loader_contract() {
        let Ok(dir) = std::env::var("LLM170_DS4_EXL3") else {
            eprintln!("skip: LLM170_DS4_EXL3 없음");
            return;
        };
        let dir = PathBuf::from(dir);
        if !dir.exists() {
            eprintln!("skip: {dir:?} 없음");
            return;
        }
        let l = Ds4Loader::open(&dir).expect("open");
        assert_eq!(l.cfg.n_layers, 43);
        assert_eq!(l.cfg.vocab, 129280);
        assert_eq!(l.cfg.rms_eps, 1e-20); // Vision-Exp LLM 타워 (보고서 §8)
        assert!(l.entry_count() > 100_000);
        // plain: norm 4096, sink 64.
        assert_eq!(l.plain_f32("norm.weight").unwrap().len(), 4096);
        assert_eq!(l.plain_f32("layers.0.attn.attn_sink").unwrap().len(), 64);
        // linear: wq_a 4096→1024, 유한값.
        let w = l.linear("layers.0.attn.wq_a").unwrap();
        assert_eq!((w.k, w.n), (4096, 1024));
        assert!(w.w.iter().all(|v| v.is_finite()));
        assert!(w.w.iter().any(|&v| v != 0.0));
        // tid2eid: 6·vocab, 범위 내 전문가 id.
        let t = l.tid2eid(0).unwrap();
        assert_eq!(t.len(), 6 * 129280);
        assert!(t.iter().all(|&e| (0..256).contains(&(e as usize))));
        // embed 행 부분 읽기 — 유한.
        let rows = l.embed_rows(&[0, 1, 128799]).unwrap();
        assert_eq!(rows.len(), 3);
        assert!(
            rows.iter()
                .all(|r| r.len() == 4096 && r.iter().all(|v| v.is_finite()))
        );
    }
}
