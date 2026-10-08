//! EXL3 → F16 GGUF 변환 (plans/118 §7-3②) — 품질 게이트 지름길.
//!
//! 트렐리스를 원 기저로 디퀀트(W = diag(suh)·(H·Wq·H)/128·diag(svh))해
//! F16 GGUF로 상재 변환 — 기존 qwen35 엔진을 무수정 구동. GGUF 데이터
//! 배치 = torch [out][in] 행 우선 = 우리 W[k][n]의 **전치**.

use crate::gguf_out::{GgufWriter, Kv};
use crate::model::{Exl3Model, LinearRef};
use crate::trellis::LinearView;
use crate::{Json, Result};
use half::f16;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct ConvertStats {
    pub tensors: usize,
    pub bytes: u64,
    pub elapsed_s: f64,
}

/// GDN V헤드 순열 — llama.cpp(subhead-major) ↔ HF(group-major):
/// gguf 블록 i ← 원본 블록 3·(i%16) + i/16 (48 V헤드, 실측 확정 —
/// beta 지문 corr 1.000·ssm_out 블록 corr 0.999).
fn vperm(i: usize) -> usize {
    3 * (i % 16) + i / 16
}

enum Plan {
    /// BF16/F16 직접(2D torch 행우선 == GGUF 배치) → F16.
    PlainF16 { key: String, ne: [u64; 2] },
    /// F16 2D — 행(헤드) V순열 (alpha/beta [in, 48헤드]).
    PlainF16VRows {
        key: String,
        ne: [u64; 2],
        vrows: u64,
    },
    /// conv1d F16 — v채널 블록 순열 (torch [ch,1,k] == GGUF ne=[k,ch]).
    PlainF16VCh {
        key: String,
        ne: [u64; 2],
        vbase: u64,
    },
    /// 1D BF16 → F32 (norm — 무순열).
    PlainF32 { key: String, n: u64 },
    /// 잔차 RMSNorm γ = 1+w 폴딩(엔진·llama.cpp 규약 — 실측: q4 = exl3+1).
    NormPlus1 { key: String, n: u64 },
    /// 1D BF16 → F32 V순열 (dt_bias).
    PlainF32V { key: String, n: u64 },
    /// A_log → ssm_a = -exp(A_log) V순열 (F32).
    SsmA { key: String },
    /// 트렐리스 → 원 기저 F16 전치. n_base/k_base 이후 128블록 V순열.
    Trellis {
        key: String,
        k: usize,
        n: usize,
        n_base: usize,
        k_base: usize,
    },
}

/// `exl3_to_gguf(dir, out, max_layers)` — qwen35 F16 GGUF 생성.
/// max_layers = Some(n): 처음 n층만(배선 스모크용 절단 모델 — 하이퍼파라미터는
/// 전체 기준 그대로라 엔진은 64층 기대 → 절단 시 block_count도 n으로).
pub fn exl3_to_gguf(
    dir: &Path,
    out_path: &Path,
    max_layers: Option<usize>,
) -> Result<ConvertStats> {
    let t0 = std::time::Instant::now();
    let m = Exl3Model::open(dir)?;
    let pieces = load_token_pieces(dir)?;
    let c = &m.cfg;

    // ── kv ──
    let mut w = GgufWriter::new();
    w.kv("general.architecture", Kv::Str("qwen35".into()));
    w.kv("qwen35.embedding_length", Kv::U32(c.hidden_size as u32));
    let n_layers = max_layers
        .unwrap_or(c.num_hidden_layers)
        .min(c.num_hidden_layers);
    w.kv("qwen35.block_count", Kv::U32(n_layers as u32));
    w.kv(
        "qwen35.attention.head_count",
        Kv::U32(c.num_attention_heads as u32),
    );
    w.kv(
        "qwen35.attention.head_count_kv",
        Kv::U32(c.num_key_value_heads as u32),
    );
    w.kv("qwen35.attention.key_length", Kv::U32(c.head_dim as u32));
    w.kv(
        "qwen35.ssm.state_size",
        Kv::U32(c.linear_key_head_dim as u32),
    );
    w.kv(
        "qwen35.ssm.group_count",
        Kv::U32(c.linear_num_key_heads as u32),
    );
    w.kv(
        "qwen35.ssm.time_step_rank",
        Kv::U32(c.linear_num_value_heads as u32),
    );
    w.kv(
        "qwen35.ssm.inner_size",
        Kv::U32((c.linear_num_value_heads * c.linear_key_head_dim) as u32),
    );
    w.kv(
        "qwen35.feed_forward_length",
        Kv::U32(c.intermediate_size as u32),
    );
    w.kv(
        "qwen35.rope.dimension_count",
        Kv::U32((c.head_dim as f64 * c.partial_rotary_factor) as u32),
    );
    w.kv("qwen35.rope.freq_base", Kv::F64(c.rope_theta));
    w.kv(
        "qwen35.attention.layer_norm_rms_epsilon",
        Kv::F32(c.rms_norm_eps),
    );
    w.kv(
        "qwen35.full_attention_interval",
        Kv::U32(c.full_attention_interval as u32),
    );
    w.kv(
        "qwen35.ssm.conv_kernel",
        Kv::U32(c.linear_conv_kernel as u32),
    );
    w.kv("qwen35.context_length", Kv::U64(262144));
    w.kv("tokenizer.ggml.tokens", Kv::StrArray(pieces));

    // ── 텐서 플랜 (기입 순서) ──
    let mut plans: Vec<(String, Plan)> = Vec::new(); // (gguf_name, plan)
    let h = c.hidden_size as u64;
    let ff = c.intermediate_size as u64;
    let v = c.vocab_size as u64;
    plans.push((
        "token_embd.weight".into(),
        Plan::PlainF16 {
            key: "model.language_model.embed_tokens.weight".into(),
            ne: [h, v],
        },
    ));
    plans.push((
        "output_norm.weight".into(),
        Plan::NormPlus1 {
            key: "model.language_model.norm.weight".into(),
            n: h,
        },
    ));
    plans.push((
        "output.weight".into(),
        Plan::Trellis {
            key: "lm_head".into(),
            k: c.hidden_size,
            n: c.vocab_size,
            n_base: usize::MAX,
            k_base: usize::MAX,
        },
    ));
    let nv = c.linear_num_value_heads as u64;
    let nk = c.linear_num_key_heads as u64;
    let hd = c.linear_key_head_dim as u64;
    let nh = c.num_attention_heads as u64;
    let nkv = c.num_key_value_heads as u64;
    let ahd = c.head_dim as u64;
    for il in 0..n_layers {
        let l = format!("model.language_model.layers.{il}");
        let g = format!("blk.{il}");
        plans.push((
            format!("{g}.attn_norm.weight"),
            Plan::NormPlus1 {
                key: format!("{l}.input_layernorm.weight"),
                n: h,
            },
        ));
        let full = il % c.full_attention_interval == c.full_attention_interval - 1;
        if full {
            plans.push((
                format!("{g}.attn_q.weight"),
                Plan::Trellis {
                    key: format!("{l}.self_attn.q_proj"),
                    k: h as usize,
                    n: (nh * ahd * 2) as usize,
                    n_base: usize::MAX,
                    k_base: usize::MAX,
                },
            ));
            plans.push((
                format!("{g}.attn_k.weight"),
                Plan::Trellis {
                    key: format!("{l}.self_attn.k_proj"),
                    k: h as usize,
                    n: (nkv * ahd) as usize,
                    n_base: usize::MAX,
                    k_base: usize::MAX,
                },
            ));
            plans.push((
                format!("{g}.attn_v.weight"),
                Plan::Trellis {
                    key: format!("{l}.self_attn.v_proj"),
                    k: h as usize,
                    n: (nkv * ahd) as usize,
                    n_base: usize::MAX,
                    k_base: usize::MAX,
                },
            ));
            plans.push((
                format!("{g}.attn_q_norm.weight"),
                Plan::NormPlus1 {
                    key: format!("{l}.self_attn.q_norm.weight"),
                    n: ahd,
                },
            ));
            plans.push((
                format!("{g}.attn_k_norm.weight"),
                Plan::NormPlus1 {
                    key: format!("{l}.self_attn.k_norm.weight"),
                    n: ahd,
                },
            ));
            plans.push((
                format!("{g}.attn_output.weight"),
                Plan::Trellis {
                    key: format!("{l}.self_attn.o_proj"),
                    k: (nh * ahd) as usize,
                    n: h as usize,
                    n_base: usize::MAX,
                    k_base: usize::MAX,
                },
            ));
        } else {
            let qkv = ((2 * nk + nv) * hd) as usize;
            let z = (nv * hd) as usize;
            // V헤드 순열 축: qkv 출력 v부(2·nk·hd..), z 전체, out_proj 입력 전체.
            plans.push((
                format!("{g}.attn_qkv.weight"),
                Plan::Trellis {
                    key: format!("{l}.linear_attn.in_proj_qkv"),
                    k: h as usize,
                    n: qkv,
                    n_base: 2 * nk as usize * hd as usize,
                    k_base: usize::MAX,
                },
            ));
            plans.push((
                format!("{g}.attn_gate.weight"),
                Plan::Trellis {
                    key: format!("{l}.linear_attn.in_proj_z"),
                    k: h as usize,
                    n: z,
                    n_base: 0,
                    k_base: usize::MAX,
                },
            ));
            plans.push((
                format!("{g}.ssm_conv1d.weight"),
                Plan::PlainF16VCh {
                    key: format!("{l}.linear_attn.conv1d.weight"),
                    ne: [c.linear_conv_kernel as u64, qkv as u64],
                    vbase: 2 * nk * hd,
                },
            ));
            plans.push((
                format!("{g}.ssm_dt.bias"),
                Plan::PlainF32V {
                    key: format!("{l}.linear_attn.dt_bias"),
                    n: nv,
                },
            ));
            plans.push((
                format!("{g}.ssm_a"),
                Plan::SsmA {
                    key: format!("{l}.linear_attn.A_log"),
                },
            ));
            plans.push((
                format!("{g}.ssm_alpha.weight"),
                Plan::PlainF16VRows {
                    key: format!("{l}.linear_attn.in_proj_a.weight"),
                    ne: [h, nv],
                    vrows: nv,
                },
            ));
            plans.push((
                format!("{g}.ssm_beta.weight"),
                Plan::PlainF16VRows {
                    key: format!("{l}.linear_attn.in_proj_b.weight"),
                    ne: [h, nv],
                    vrows: nv,
                },
            ));
            plans.push((
                format!("{g}.ssm_norm.weight"),
                Plan::PlainF32 {
                    key: format!("{l}.linear_attn.norm.weight"),
                    n: hd,
                },
            ));
            plans.push((
                format!("{g}.ssm_out.weight"),
                Plan::Trellis {
                    key: format!("{l}.linear_attn.out_proj"),
                    k: z,
                    n: h as usize,
                    n_base: usize::MAX,
                    k_base: 0,
                },
            ));
        }
        plans.push((
            format!("{g}.post_attention_norm.weight"),
            Plan::NormPlus1 {
                key: format!("{l}.post_attention_layernorm.weight"),
                n: h,
            },
        ));
        plans.push((
            format!("{g}.ffn_gate.weight"),
            Plan::Trellis {
                key: format!("{l}.mlp.gate_proj"),
                k: h as usize,
                n: ff as usize,
                n_base: usize::MAX,
                k_base: usize::MAX,
            },
        ));
        plans.push((
            format!("{g}.ffn_up.weight"),
            Plan::Trellis {
                key: format!("{l}.mlp.up_proj"),
                k: h as usize,
                n: ff as usize,
                n_base: usize::MAX,
                k_base: usize::MAX,
            },
        ));
        plans.push((
            format!("{g}.ffn_down.weight"),
            Plan::Trellis {
                key: format!("{l}.mlp.down_proj"),
                k: ff as usize,
                n: h as usize,
                n_base: usize::MAX,
                k_base: usize::MAX,
            },
        ));
    }

    // 등록 — 플랜 순서 = 데이터 기입 순서.
    let mut total_bytes = 0u64;
    for (name, p) in &plans {
        match p {
            Plan::PlainF16 { ne, .. }
            | Plan::PlainF16VRows { ne, .. }
            | Plan::PlainF16VCh { ne, .. } => {
                w.tensor_f16(name, ne);
                total_bytes += ne[0] * ne[1] * 2;
            }
            Plan::PlainF32 { n, .. } | Plan::PlainF32V { n, .. } | Plan::NormPlus1 { n, .. } => {
                w.tensor_f32(name, &[*n]);
                total_bytes += n * 4;
            }
            Plan::SsmA { .. } => {
                w.tensor_f32(name, &[nv]);
                total_bytes += nv * 4;
            }
            Plan::Trellis { k, n, .. } => {
                w.tensor_f16(name, &[*k as u64, *n as u64]);
                total_bytes += (*k as u64) * (*n as u64) * 2;
            }
        }
    }

    let nthreads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
        .max(1);
    let file = std::fs::File::create(out_path)?;
    let mut bw = std::io::BufWriter::with_capacity(16 << 20, file);
    let mut written = 0u64;
    let plans_ref = &plans;
    let model = &m;
    w.write(&mut bw, |name, _off, len, out| {
        let (gname, plan) = plans_ref.iter().find(|(n, _)| n == name).unwrap();
        let _ = gname;
        let buf = materialize(model, plan, nthreads)?;
        assert_eq!(buf.len() as u64, len, "{name} 길이 불일치");
        out.write_all(&buf)?;
        written += len;
        if written % (1 << 30) < len {
            eprintln!(
                "  [conv] {:.1}/{} GB",
                written as f64 / 1e9,
                total_bytes as f64 / 1e9
            );
        }
        Ok(())
    })?;
    bw.flush()?;
    Ok(ConvertStats {
        tensors: plans.len(),
        bytes: total_bytes,
        elapsed_s: t0.elapsed().as_secs_f64(),
    })
}

fn materialize(m: &Exl3Model, plan: &Plan, nthreads: usize) -> Result<Vec<u8>> {
    match plan {
        Plan::PlainF16 { key, ne } => {
            let p = m
                .plain(key)
                .ok_or_else(|| crate::Exl3Error::TensorNotFound(key.clone()))?;
            let bytes = m.slice(&p.slab);
            let n: usize = (ne[0] * ne[1]) as usize;
            let mut out = vec![0u8; n * 2];
            let (chunks, _) = bytes.as_chunks::<2>();
            for (i, c) in chunks.iter().take(n).enumerate() {
                let f = if p.dtype == crate::StDtype::Bf16 {
                    bf16_to_f32(c[0], c[1])
                } else {
                    f16::from_le_bytes(*c).to_f32()
                };
                let h = f16::from_f32(f);
                out[2 * i..2 * i + 2].copy_from_slice(&h.to_le_bytes());
            }
            Ok(out)
        }
        Plan::PlainF16VRows { key, ne, vrows } => {
            let p = m
                .plain(key)
                .ok_or_else(|| crate::Exl3Error::TensorNotFound(key.clone()))?;
            let bytes = m.slice(&p.slab);
            let (rows, cols) = (ne[1] as usize, ne[0] as usize); // 헤드행 × in
            let (chunks, _) = bytes.as_chunks::<2>();
            let mut out = vec![0u8; rows * cols * 2];
            for i in 0..rows {
                let src = if (i as u64) < *vrows { vperm(i) } else { i };
                for c in 0..cols {
                    let v = if p.dtype == crate::StDtype::Bf16 {
                        bf16_to_f32(chunks[src * cols + c][0], chunks[src * cols + c][1])
                    } else {
                        f16::from_le_bytes(chunks[src * cols + c]).to_f32()
                    };
                    let h = f16::from_f32(v);
                    out[(i * cols + c) * 2..(i * cols + c) * 2 + 2]
                        .copy_from_slice(&h.to_le_bytes());
                }
            }
            Ok(out)
        }
        Plan::PlainF16VCh { key, ne, vbase } => {
            // conv1d — 채널행 블록(128) 순열: GGUF 행=채널, ne0=kernel.
            let p = m
                .plain(key)
                .ok_or_else(|| crate::Exl3Error::TensorNotFound(key.clone()))?;
            let bytes = m.slice(&p.slab);
            let (ch, kk) = (ne[1] as usize, ne[0] as usize);
            let (chunks, _) = bytes.as_chunks::<2>();
            let mut out = vec![0u8; ch * kk * 2];
            let nb = (ch - *vbase as usize) / 128;
            // q/k 채널 [0..vbase) 직접 복사(V순열 무관 — v이전은 무순열).
            for r in 0..*vbase as usize {
                for c in 0..kk {
                    let v = if p.dtype == crate::StDtype::Bf16 {
                        bf16_to_f32(chunks[r * kk + c][0], chunks[r * kk + c][1])
                    } else {
                        f16::from_le_bytes(chunks[r * kk + c]).to_f32()
                    };
                    let h = f16::from_f32(v);
                    out[(r * kk + c) * 2..(r * kk + c) * 2 + 2].copy_from_slice(&h.to_le_bytes());
                }
            }
            for blk in 0..nb {
                // src 채널 = vbase(채널 단위) + vperm(블록)·128 + r — 단위 혼합 주의.
                let src_base = *vbase as usize + vperm(blk) * 128;
                for r in 0..128 {
                    let dst = *vbase as usize + blk * 128 + r;
                    for c in 0..kk {
                        let v = if p.dtype == crate::StDtype::Bf16 {
                            bf16_to_f32(
                                chunks[(src_base + r) * kk + c][0],
                                chunks[(src_base + r) * kk + c][1],
                            )
                        } else {
                            f16::from_le_bytes(chunks[(src_base + r) * kk + c]).to_f32()
                        };
                        let h = f16::from_f32(v);
                        out[(dst * kk + c) * 2..(dst * kk + c) * 2 + 2]
                            .copy_from_slice(&h.to_le_bytes());
                    }
                }
            }
            Ok(out)
        }
        Plan::PlainF32V { key, n } => {
            let p = m
                .plain(key)
                .ok_or_else(|| crate::Exl3Error::TensorNotFound(key.clone()))?;
            let bytes = m.slice(&p.slab);
            let (chunks, _) = bytes.as_chunks::<2>();
            let mut out = vec![0u8; *n as usize * 4];
            for i in 0..*n as usize {
                let src = vperm(i);
                let f = if p.dtype == crate::StDtype::Bf16 {
                    bf16_to_f32(chunks[src][0], chunks[src][1])
                } else {
                    f16::from_le_bytes(chunks[src]).to_f32()
                };
                out[4 * i..4 * i + 4].copy_from_slice(&f.to_le_bytes());
            }
            Ok(out)
        }
        Plan::PlainF32 { key, n } | Plan::NormPlus1 { key, n } => {
            let plus1 = matches!(plan, Plan::NormPlus1 { .. });
            let p = m
                .plain(key)
                .ok_or_else(|| crate::Exl3Error::TensorNotFound(key.clone()))?;
            let bytes = m.slice(&p.slab);
            let (chunks, _) = bytes.as_chunks::<2>();
            let mut out = vec![0u8; *n as usize * 4];
            for i in 0..*n as usize {
                let mut f = if p.dtype == crate::StDtype::Bf16 {
                    bf16_to_f32(chunks[i][0], chunks[i][1])
                } else {
                    f16::from_le_bytes(chunks[i]).to_f32()
                };
                if plus1 {
                    f += 1.0;
                }
                out[4 * i..4 * i + 4].copy_from_slice(&f.to_le_bytes());
            }
            Ok(out)
        }
        Plan::SsmA { key } => {
            let p = m
                .plain(key)
                .ok_or_else(|| crate::Exl3Error::TensorNotFound(key.clone()))?;
            let bytes = m.slice(&p.slab);
            let (chunks, _) = bytes.as_chunks::<2>();
            let mut out = Vec::with_capacity(chunks.len() * 4);
            for (i, _) in chunks.iter().enumerate() {
                let a_log = bf16_to_f32(chunks[vperm(i)][0], chunks[vperm(i)][1]);
                let a = -a_log.exp();
                out.extend_from_slice(&a.to_le_bytes());
            }
            Ok(out)
        }
        Plan::Trellis {
            key,
            k,
            n,
            n_base,
            k_base,
        } => {
            let lr: &LinearRef = m
                .linear(key)
                .ok_or_else(|| crate::Exl3Error::TensorNotFound(key.clone()))?;
            let view = LinearView {
                tre: m.slice(&lr.tre),
                suh: m.slice(&lr.suh),
                svh: m.slice(&lr.svh),
                k: *k,
                n: *n,
                krate: lr.krate,
            };
            // 전치 F16: out[(n0+j)*k + (k0+i)] = f16(W[k0+i][n0+j]).
            let mut out = vec![0u8; k * n * 2];
            let tasks: Vec<(usize, usize)> = (0..*n / 128)
                .flat_map(|nc| (0..*k / 128).map(move |kc| (nc, kc)))
                .collect();
            let next = AtomicUsize::new(0);
            // SAFETY(Send): 원시 포인터의 스레드 공유 — 태스크별 상호 배타적
            // 영역 기입 계약(위 주석)으로 데이터 레이스 없음.
            struct SendPtr(*mut u8);
            // SAFETY(Send+Sync): 태스크별 상호 배타적 기입 — 공유 자체 무해.
            unsafe impl Send for SendPtr {}
            unsafe impl Sync for SendPtr {}
            let base = std::sync::Arc::new(SendPtr(out.as_mut_ptr()));
            std::thread::scope(|s| {
                for _ in 0..nthreads {
                    s.spawn(|| {
                        loop {
                            let t = next.fetch_add(1, Ordering::Relaxed);
                            if t >= tasks.len() {
                                break;
                            }
                            let (nc, kc) = tasks[t];
                            // V헤드 순열 — 목표 블록 → 원본 블록(128 단위).
                            let map = |b: usize, base: usize, span: usize| -> usize {
                                if base == usize::MAX || span == 0 || b * 128 < base {
                                    b
                                } else {
                                    let off = (b * 128 - base) / 128;
                                    if off < span {
                                        (base + vperm(off) * 128) / 128
                                    } else {
                                        b
                                    }
                                }
                            };
                            let span_n = if *n_base == usize::MAX {
                                0
                            } else {
                                (n - n_base) / 128
                            };
                            let span_k = if *k_base == usize::MAX {
                                0
                            } else {
                                (k - k_base) / 128
                            };
                            let blk = view.dequant_block_f64(
                                map(kc, *k_base, span_k) * 128,
                                map(nc, *n_base, span_n) * 128,
                                128,
                                128,
                            );
                            for j in 0..128usize {
                                let row = (nc * 128 + j) * k + kc * 128;
                                for i in 0..128usize {
                                    let h = f16::from_f64(blk[i * 128 + j]);
                                    // SAFETY: 태스크 (nc,kc)는 상호 배타적
                                    // 영역(행 nc*128..+128, 열 kc*128..+128)에만
                                    // 기입 — 데이터 레이스 없음. base는 out 수명.
                                    unsafe {
                                        std::ptr::copy_nonoverlapping(
                                            h.to_le_bytes().as_ptr(),
                                            base.0.add((row + i) * 2),
                                            2,
                                        );
                                    }
                                }
                            }
                        }
                    });
                }
            });
            Ok(out)
        }
    }
}

/// BF16 하위바이트·상위바이트 → f32.
fn bf16_to_f32(lo: u8, hi: u8) -> f32 {
    f32::from_bits(((hi as u32) << 24) | ((lo as u32) << 16))
}

/// vocab.json {"piece": id} → id 순 토큰 조각표. (W4A16 변환기 공용 — pub)
pub fn load_token_pieces(dir: &Path) -> Result<Vec<String>> {
    let raw = std::fs::read_to_string(dir.join("vocab.json"))?;
    let v = Json::parse(&raw)?;
    let obj = v
        .as_object()
        .ok_or_else(|| crate::Exl3Error::BadHeader("vocab.json: 객체 아님".into()))?;
    let mut pairs: Vec<(u32, String)> = obj
        .iter()
        .filter_map(|(piece, id)| match id {
            Json::Num(n) if *n >= 0.0 && n.fract() == 0.0 => Some((*n as u32, piece.clone())),
            _ => None,
        })
        .collect();
    pairs.sort_by_key(|(id, _)| *id);
    Ok(pairs.into_iter().map(|(_, p)| p).collect())
}
