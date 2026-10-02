//! EXL3 모델 레지스트리 — 엔진 적재용 mmap 뷰 + qwen35 하이퍼파라미터 +
//! 텐서 완전성 검증 (plans/118 §7).
//!
//! 검증 참조 경로(Exl3Linear, 바이트 복사)와 달리 엔진 경로는 샤드를
//! mmap으로 상주시키고 텐서를 (샤드, 오프셋, 길이) 뷰로 참조한다.

use crate::json::Json;
use crate::{Exl3Error, Result, StArchive, StDtype};
use std::path::Path;

/// qwen35 관련 하이퍼파라미터 (config.json text_config).
#[derive(Debug, Clone)]
pub struct Exl3Config {
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    /// full_attention_interval — 4 (il%4==3이 full-attn).
    pub full_attention_interval: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    /// linear_attn (GDN): [num_k_heads, num_v_heads, head_dim, conv_kernel].
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_conv_kernel: usize,
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
    pub mrope_section: [usize; 3],
}

/// 트렐리스 선형의 mmap 뷰 — (샤드 인덱스, 데이터섹션 시작 오프셋).
#[derive(Debug, Clone)]
pub struct LinearRef {
    pub key: String,
    /// k=in, n=out (엔진 계약: GEMM A[m,k]×B[k,n]).
    pub k: usize,
    pub n: usize,
    pub krate: u32,
    /// trellis/suh/svh의 (shard, begin, len) — MmapSlab에서 슬라이스.
    pub tre: Slab,
    pub suh: Slab,
    pub svh: Slab,
}

/// 무양자화(BF16/F16) 텐서 뷰.
#[derive(Debug, Clone)]
pub struct PlainRef {
    pub key: String,
    pub dtype: StDtype,
    pub shape: Vec<u64>,
    pub slab: Slab,
}

#[derive(Debug, Clone, Copy)]
pub struct Slab {
    pub shard: usize,
    pub begin: u64,
    pub len: u64,
}

/// 언어 모델 EXL3 모델 — visual·mtp 제외(plans/118 §7.1, v1).
pub struct Exl3Model {
    pub cfg: Exl3Config,
    pub linears: Vec<LinearRef>,
    pub plains: Vec<PlainRef>,
    /// mmap 슬래브 — 라이프타임 보유자.
    pub slabs: Vec<memmap2::Mmap>,
}

impl Exl3Model {
    /// 디렉터리(config.json + 샤드)에서 레지스트리 구축 + 완전성 검증.
    pub fn open(dir: &Path) -> Result<Self> {
        let cfg = load_config(&dir.join("config.json"))?;
        let ar = StArchive::open(dir)?;
        let (slabs, shard_bases) = mmap_all(&ar)?;

        let mut linears = Vec::new();
        let mut plains = Vec::new();
        for (name, e) in ar.entries() {
            if name.contains("model.visual.") || name.starts_with("mtp.") {
                continue; // v1 언어 모델만 (§7.1)
            }
            let slab = |off: u64| Slab {
                shard: e.shard,
                begin: shard_bases[e.shard] + off,
                len: e.nbytes(),
            };
            if name.ends_with(".trellis") {
                let base = &name[..name.len() - 8];
                let (kt, nt, tw) = (e.shape[0], e.shape[1], e.shape[2]);
                if tw % 16 != 0 {
                    return Err(Exl3Error::BadTensor(format!(
                        "{name}: 반정수 bpw 미지원(현 타깃 없음)"
                    )));
                }
                let (suh, svh) = (
                    ar.entry(&format!("{base}.suh"))
                        .ok_or_else(|| Exl3Error::TensorNotFound(format!("{base}.suh")))?,
                    ar.entry(&format!("{base}.svh"))
                        .ok_or_else(|| Exl3Error::TensorNotFound(format!("{base}.svh")))?,
                );
                if suh.nbytes() != kt * 16 * 2 || svh.nbytes() != nt * 16 * 2 {
                    return Err(Exl3Error::BadTensor(format!("{base}: suh/svh 차원 불일치")));
                }
                linears.push(LinearRef {
                    key: base.to_string(),
                    k: (kt * 16) as usize,
                    n: (nt * 16) as usize,
                    krate: (tw / 16) as u32,
                    tre: slab(e.begin),
                    suh: Slab {
                        shard: suh.shard,
                        begin: shard_bases[suh.shard] + suh.begin,
                        len: suh.nbytes(),
                    },
                    svh: Slab {
                        shard: svh.shard,
                        begin: shard_bases[svh.shard] + svh.begin,
                        len: svh.nbytes(),
                    },
                });
            } else if matches!(e.dtype, StDtype::F16 | StDtype::Bf16) {
                plains.push(PlainRef {
                    key: name.clone(),
                    dtype: e.dtype,
                    shape: e.shape.clone(),
                    slab: slab(e.begin),
                });
            }
        }
        let m = Self {
            cfg,
            linears,
            plains,
            slabs,
        };
        m.verify_language()?;
        Ok(m)
    }

    pub fn linear(&self, key: &str) -> Option<&LinearRef> {
        self.linears.iter().find(|l| l.key == key)
    }

    pub fn plain(&self, key: &str) -> Option<&PlainRef> {
        self.plains.iter().find(|p| p.key == key)
    }

    /// 슬래브에서 텐서 원시 바이트 뷰.
    pub fn slice(&self, s: &Slab) -> &[u8] {
        // SAFETY: mmap 상주 — 슬래브 라이프타임 내 읽기 전용.
        &self.slabs[s.shard][s.begin as usize..(s.begin + s.len) as usize]
    }

    /// §7.1 완전성 — 언어 모델 전 경로 텐서 존재·차원 검증.
    fn verify_language(&self) -> Result<()> {
        let c = &self.cfg;
        let need_plain = |m: &Self, key: &str, want: &[u64]| -> Result<()> {
            let p = m
                .plain(key)
                .ok_or_else(|| Exl3Error::TensorNotFound(format!("완전성: {key} 부재")))?;
            if p.shape != want {
                return Err(Exl3Error::BadTensor(format!(
                    "{key}: 형상 {:?} != 기대 {want:?}",
                    p.shape
                )));
            }
            Ok(())
        };
        let need_lin = |m: &Self, key: &str, k: usize, n: usize| -> Result<()> {
            let l = m
                .linear(key)
                .ok_or_else(|| Exl3Error::TensorNotFound(format!("완전성: {key} 부재")))?;
            if l.k != k || l.n != n {
                return Err(Exl3Error::BadTensor(format!(
                    "{key}: {}×{} != 기대 {k}×{n}",
                    l.k, l.n
                )));
            }
            Ok(())
        };
        need_plain(
            self,
            "model.language_model.embed_tokens.weight",
            &[c.vocab_size as u64, c.hidden_size as u64],
        )?;
        need_plain(
            self,
            "model.language_model.norm.weight",
            &[c.hidden_size as u64],
        )?;
        need_lin(self, "lm_head", c.hidden_size, c.vocab_size)?;
        for il in 0..c.num_hidden_layers {
            let l = format!("model.language_model.layers.{il}");
            need_plain(
                self,
                &format!("{l}.input_layernorm.weight"),
                &[c.hidden_size as u64],
            )?;
            need_plain(
                self,
                &format!("{l}.post_attention_layernorm.weight"),
                &[c.hidden_size as u64],
            )?;
            let full = il % c.full_attention_interval == c.full_attention_interval - 1;
            if full {
                let (nh, kv, hd) = (c.num_attention_heads, c.num_key_value_heads, c.head_dim);
                need_lin(
                    self,
                    &format!("{l}.self_attn.q_proj"),
                    c.hidden_size,
                    nh * hd * 2,
                )?;
                need_lin(
                    self,
                    &format!("{l}.self_attn.k_proj"),
                    c.hidden_size,
                    kv * hd,
                )?;
                need_lin(
                    self,
                    &format!("{l}.self_attn.v_proj"),
                    c.hidden_size,
                    kv * hd,
                )?;
                need_lin(
                    self,
                    &format!("{l}.self_attn.o_proj"),
                    nh * hd,
                    c.hidden_size,
                )?;
                need_plain(self, &format!("{l}.self_attn.q_norm.weight"), &[hd as u64])?;
                need_plain(self, &format!("{l}.self_attn.k_norm.weight"), &[hd as u64])?;
            } else {
                let (nk, nv, hd) = (
                    c.linear_num_key_heads,
                    c.linear_num_value_heads,
                    c.linear_value_head_dim(),
                );
                need_lin(
                    self,
                    &format!("{l}.linear_attn.in_proj_qkv"),
                    c.hidden_size,
                    (2 * nk + nv) * hd,
                )?;
                need_lin(
                    self,
                    &format!("{l}.linear_attn.in_proj_z"),
                    c.hidden_size,
                    nv * hd,
                )?;
                need_lin(
                    self,
                    &format!("{l}.linear_attn.out_proj"),
                    nv * hd,
                    c.hidden_size,
                )?;
                need_plain(self, &format!("{l}.linear_attn.A_log"), &[nv as u64])?;
                need_plain(self, &format!("{l}.linear_attn.dt_bias"), &[nv as u64])?;
                let qkv_dim = (2 * nk + nv) * hd;
                need_plain(
                    self,
                    &format!("{l}.linear_attn.conv1d.weight"),
                    &[qkv_dim as u64, 1, c.linear_conv_kernel as u64],
                )?;
                need_plain(
                    self,
                    &format!("{l}.linear_attn.norm.weight"),
                    &[c.linear_value_head_dim() as u64],
                )?;
                // GDN a/b는 모두 V헤드 스케일러 [nv, hidden] (§7.1).
                need_plain(
                    self,
                    &format!("{l}.linear_attn.in_proj_a.weight"),
                    &[nv as u64, c.hidden_size as u64],
                )?;
                need_plain(
                    self,
                    &format!("{l}.linear_attn.in_proj_b.weight"),
                    &[nv as u64, c.hidden_size as u64],
                )?;
            }
            need_lin(
                self,
                &format!("{l}.mlp.gate_proj"),
                c.hidden_size,
                c.intermediate_size,
            )?;
            need_lin(
                self,
                &format!("{l}.mlp.up_proj"),
                c.hidden_size,
                c.intermediate_size,
            )?;
            need_lin(
                self,
                &format!("{l}.mlp.down_proj"),
                c.intermediate_size,
                c.hidden_size,
            )?;
        }
        Ok(())
    }
}

impl Exl3Config {
    /// linear_attn V헤드 차원 — qwen3_5 계열은 K/V 헤드차원 동일(128).
    fn linear_value_head_dim(&self) -> usize {
        self.linear_key_head_dim
    }
}

fn load_config(path: &Path) -> Result<Exl3Config> {
    let raw = std::fs::read_to_string(path)?;
    let v = Json::parse(&raw)?;
    let t = v
        .get("text_config")
        .ok_or_else(|| Exl3Error::BadHeader("config.json: text_config 부재".into()))?;
    let getn = |k: &str| -> Result<u64> {
        t.get(k)
            .and_then(|v| match v {
                Json::Num(n) if *n >= 0.0 && n.fract() == 0.0 => Some(*n as u64),
                _ => None,
            })
            .ok_or_else(|| Exl3Error::BadHeader(format!("config.json: {k}")))
    };
    let mrope = t
        .get("rope_parameters")
        .and_then(|r| r.get("mrope_section"))
        .and_then(Json::as_num_array)
        .map(|a| {
            [
                a.first().copied().unwrap_or(0) as usize,
                a.get(1).copied().unwrap_or(0) as usize,
                a.get(2).copied().unwrap_or(0) as usize,
            ]
        })
        .unwrap_or([0, 0, 0]);
    Ok(Exl3Config {
        num_hidden_layers: getn("num_hidden_layers")? as usize,
        hidden_size: getn("hidden_size")? as usize,
        intermediate_size: getn("intermediate_size")? as usize,
        num_attention_heads: getn("num_attention_heads")? as usize,
        num_key_value_heads: getn("num_key_value_heads")? as usize,
        head_dim: getn("head_dim")? as usize,
        full_attention_interval: getn("full_attention_interval").unwrap_or(4) as usize,
        vocab_size: getn("vocab_size")? as usize,
        rms_norm_eps: t.get("rms_norm_eps").and_then(Json::as_f64).unwrap_or(1e-6) as f32,
        linear_key_head_dim: getn("linear_key_head_dim")? as usize,
        linear_num_key_heads: getn("linear_num_key_heads")? as usize,
        linear_num_value_heads: getn("linear_num_value_heads")? as usize,
        linear_conv_kernel: getn("linear_conv_kernel_dim")? as usize,
        rope_theta: t
            .get("rope_parameters")
            .and_then(|r| r.get("rope_theta"))
            .and_then(Json::as_f64)
            .unwrap_or(1e7),
        partial_rotary_factor: t
            .get("partial_rotary_factor")
            .and_then(Json::as_f64)
            .unwrap_or(1.0),
        mrope_section: mrope,
    })
}

fn mmap_all(ar: &StArchive) -> Result<(Vec<memmap2::Mmap>, Vec<u64>)> {
    let shard_paths = ar.shard_paths();
    let mut slabs = Vec::with_capacity(shard_paths.len());
    let mut bases = Vec::with_capacity(shard_paths.len());
    for (i, p) in shard_paths.iter().enumerate() {
        let f = std::fs::File::open(p)?;
        // SAFETY: 읽기 전용 mmap — 파일 크기는 StArchive 파싱 시점과 동일.
        let m = unsafe { memmap2::Mmap::map(&f)? };
        let base = ar
            .shard_data_base(i)
            .ok_or_else(|| Exl3Error::BadHeader(format!("샤드 {i} 데이터 오프셋 유실")))?;
        bases.push(base);
        slabs.push(m);
    }
    Ok((slabs, bases))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// mmap 뷰 ↔ 파일 판독(Exl3Linear) 교차검증 — 픽스처 게이트.
    #[test]
    fn mmap_matches_file_read() {
        let Ok(dir) = std::env::var("LLM170_M27_EXL3") else {
            return;
        };
        let m = Exl3Model::open(std::path::Path::new(&dir)).unwrap();
        let ar = StArchive::open(std::path::Path::new(&dir)).unwrap();
        let key = "model.language_model.layers.0.mlp.gate_proj";
        let lref = m.linear(key).unwrap();
        let lin = crate::Exl3Linear::load(&ar, key).unwrap();

        // trellis 타일(0,0) 워드 비교 — mmap 슬라이스→u16→tile_words.
        let tw = 16 * lref.krate as usize;
        let bytes = m.slice(&lref.tre);
        let (chunks, _) = bytes[..tw * 2].as_chunks::<2>();
        let u16s: Vec<u16> = chunks.iter().map(|c| u16::from_le_bytes(*c)).collect();
        let mut w_mmap = [0u16; 256];
        crate::tile_words(&u16s, lref.krate, &mut w_mmap);
        let mut w_file = [0u16; 256];
        {
            let tw2 = 16 * lin.krate as usize;
            let (chunks, _) = lin.trellis[..tw2 * 2].as_chunks::<2>();
            let u16f: Vec<u16> = chunks.iter().map(|c| u16::from_le_bytes(*c)).collect();
            crate::tile_words(&u16f, lin.krate, &mut w_file);
        }
        assert_eq!(w_mmap, w_file);

        // suh/svh f16 비트 일치.
        let suh_bytes = m.slice(&lref.suh);
        for i in 0..lref.k {
            let a = u16::from_le_bytes([suh_bytes[2 * i], suh_bytes[2 * i + 1]]);
            assert_eq!(a, lin.suh[i].to_bits());
        }
        let svh_bytes = m.slice(&lref.svh);
        for i in 0..lref.n {
            let a = u16::from_le_bytes([svh_bytes[2 * i], svh_bytes[2 * i + 1]]);
            assert_eq!(a, lin.svh[i].to_bits());
        }
    }
}
