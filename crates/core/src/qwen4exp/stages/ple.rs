//! ple(n-gram 해시 임베딩) 스테이지 — Engine4에서 분리 (리팩토링 P1, 2026-09-01).
//! 수치 경로 불변 — 이동만. Ctx 기반 백엔드 독립 (CPU/GPU 동일 코드).

use super::super::Q4Error;
use super::Ctx;
use super::super::layers::SeqState4;
use crate::ops::{rms_norm, sigmoid, silu};
use llm170_profiler::profile_span;

    /// PLE 블록 — 해시 gather→key/value→게이트→방송→dilated conv→잔차 2경로.
    pub fn ple_block(
        ctx: &Ctx,
        seq: &mut SeqState4,
        il: usize,
        res_hc: &mut Vec<Vec<f32>>,
        rows: &[u32],
    ) -> Result<(), Q4Error> {
        profile_span!("q4::ple");
        let hp = ctx.model.hp.clone();
        let (n_embd, hc) = (hp.n_embd, hp.hc);
        let hc_dim = hc * n_embd;
        let t = res_hc.len();
        let w_key = ctx.model.w4(&format!("blk.{il}.ple_key.weight"))?;
        let w_value = ctx.model.w4(&format!("blk.{il}.ple_value.weight"))?;
        let n_key = ctx.model.f32_vec4(&format!("blk.{il}.ple_norm_key.weight"))?;
        let n_query = ctx.model.f32_vec4(&format!("blk.{il}.ple_norm_query.weight"))?;
        let n_conv = ctx.model.f32_vec4(&format!("blk.{il}.ple_norm_conv.weight"))?;
        let conv_w = ctx.model.f32_vec4(&format!("blk.{il}.ple_conv1d.weight"))?;

        // emb gather [t][2560]
        let heads = hp.ple_heads_per_ngram * 2; // bigram+trigram = 16
        let emb_w = heads * hp.ple_head_dim; // 16×160 = 2560
        let mut emb = vec![vec![0.0f32; emb_w]; t];
        for (ti, r) in rows.chunks(heads).enumerate() {
            let mut flat = vec![0.0f32; emb_w];
            ctx.model.ple_gather(r, &mut flat)?;
            emb[ti] = flat;
        }
        // key/value 프로젝션
        let mut key = vec![vec![0.0f32; w_key.n_out as usize]; t];
        ctx.mm_batch(&emb, &w_key, &mut key)?;
        let mut value = vec![vec![0.0f32; w_value.n_out as usize]; t];
        ctx.mm_batch(&emb, &w_value, &mut value)?;

        let mut gated_hist: Vec<Vec<f32>> = Vec::with_capacity(t);
        for ti in 0..t {
            // grouped norm key / query — 감마는 전체 [hc_dim] 폭
            let mut k_n = vec![0.0f32; key[ti].len().max(hc_dim)];
            let kl = key[ti].len();
            debug_assert!(kl == hc_dim);
            for s in 0..hc {
                let head = key[ti][s * n_embd..(s + 1) * n_embd].to_vec();
                k_n[s * n_embd..(s + 1) * n_embd]
                    .copy_from_slice(&rms_norm(&head, &n_key[s * n_embd..(s + 1) * n_embd], hp.eps));
            }
            let mut q_n = vec![0.0f32; hc_dim];
            for s in 0..hc {
                let head = res_hc[ti][s * n_embd..(s + 1) * n_embd].to_vec();
                q_n[s * n_embd..(s + 1) * n_embd]
                    .copy_from_slice(&rms_norm(&head, &n_query[s * n_embd..(s + 1) * n_embd], hp.eps));
            }
            // per-stream s = Σ key·query / √n_embd → sigmoid(sgn·√|s|)
            let mut gate = vec![0.0f32; hc];
            for s in 0..hc {
                let mut dot = 0.0f32;
                for i in 0..n_embd {
                    dot += k_n[s * n_embd + i] * q_n[s * n_embd + i];
                }
                dot /= (n_embd as f32).sqrt();
                let mag = dot.abs().max(1e-6).sqrt();
                gate[s] = sigmoid(if dot >= 0.0 { mag } else { -mag });
            }
            // value 방송 × 게이트 → grouped norm
            let mut gated = vec![0.0f32; hc_dim];
            for s in 0..hc {
                for i in 0..n_embd {
                    gated[s * n_embd + i] = value[ti][i] * gate[s];
                }
            }
            let mut normalized = vec![0.0f32; hc_dim];
            for s in 0..hc {
                let head = gated[s * n_embd..(s + 1) * n_embd].to_vec();
                normalized[s * n_embd..(s + 1) * n_embd].copy_from_slice(
                    &rms_norm(&head, &n_conv[s * n_embd..(s + 1) * n_embd], hp.eps),
                );
            }
            gated_hist.push(normalized);
        }

        // dilated depthwise conv (kern 4, dil 3, hist 9) — 시퀀스 상태 이용
        let kern = hp.ple_conv_k;
        let dil = hp.ple_ngram;
        let hist = (kern - 1) * dil;
        let st = &mut seq.ple_conv;
        // padded = hist(상태) + t열 → conv 출력 t열 → 상태 tail 갱신
        let mut padded: Vec<Vec<f32>> = Vec::with_capacity(hist + t);
        for j in 0..hist {
            padded.push(st[j * hc_dim..(j + 1) * hc_dim].to_vec());
        }
        for g in gated_hist.iter() {
            padded.push(g.clone());
        }
        let mut conv_out = vec![vec![0.0f32; hc_dim]; t];
        for ti in 0..t {
            for k in 0..kern {
                let start = hist + ti - (kern - 1 - k) * dil;
                let src = &padded[start];
                for c in 0..hc_dim {
                    conv_out[ti][c] += conv_w[c * kern + k] * src[c];
                }
            }
            for c in 0..hc_dim {
                conv_out[ti][c] = silu(conv_out[ti][c]);
            }
        }
        // 상태 갱신: 마지막 hist 열
        for j in 0..hist {
            let src = &padded[t + j];
            st[j * hc_dim..(j + 1) * hc_dim].copy_from_slice(src);
        }

        // 잔차: hidden + gated(norm 전 방송값) + conv_out — build_ple 반환식 그대로.
        // gated_pre는 conv 블록 위에서 이미 계산했으므로 재계산 대신 저장 구조 사용:
        // (value·gate 방송은 위 루프에서 `gated`로 존재했으나 norm에 덮어씀 — 재계산)
        for ti in 0..t {
            // 게이트 재계산 (결정적 동일값)
            let mut k_n = vec![0.0f32; hc_dim];
            for s in 0..hc {
                let head = key[ti][s * n_embd..(s + 1) * n_embd].to_vec();
                k_n[s * n_embd..(s + 1) * n_embd]
                    .copy_from_slice(&rms_norm(&head, &n_key[s * n_embd..(s + 1) * n_embd], hp.eps));
            }
            let mut q_n = vec![0.0f32; hc_dim];
            for s in 0..hc {
                let head = res_hc[ti][s * n_embd..(s + 1) * n_embd].to_vec();
                q_n[s * n_embd..(s + 1) * n_embd]
                    .copy_from_slice(&rms_norm(&head, &n_query[s * n_embd..(s + 1) * n_embd], hp.eps));
            }
            for s in 0..hc {
                let mut dot = 0.0f32;
                for i in 0..n_embd {
                    dot += k_n[s * n_embd + i] * q_n[s * n_embd + i];
                }
                dot /= (n_embd as f32).sqrt();
                let mag = dot.abs().max(1e-6).sqrt();
                let g = sigmoid(if dot >= 0.0 { mag } else { -mag });
                for i in 0..n_embd {
                    res_hc[ti][s * n_embd + i] += value[ti][i] * g + conv_out[ti][s * n_embd + i];
                }
            }
        }
        Ok(())
    }

    /// PLE n-gram 해시 — 호스트 u64 (ctx[s]=직전 s토큰, EOS 절단).
    pub fn ple_hash(ctx: &Ctx, seq: &mut SeqState4, tokens: &[u32]) -> Vec<u32> {
        let hp = ctx.model.hp.clone();
        let ngram = hp.ple_ngram;
        let heads = hp.ple_heads_per_ngram * 2; // bigram+trigram = 16
        let eos = hp.ple_eos;
        let hist0: Vec<u32> = seq.ple_hist.clone();
        let hist_valid = seq.ple_next_pos == seq.pos;
        let mut hist: Vec<u32> = if hist_valid { hist0 } else { vec![eos; ngram - 1] };
        let mut rows = Vec::with_capacity(tokens.len() * heads);
        for (i, &tok) in tokens.iter().enumerate() {
            let mut ctx = vec![tok as u64; ngram];
            let mut cut = false;
            for s in 1..ngram {
                let j = i as i64 - s as i64;
                let prev: u64 = if j >= 0 {
                    tokens[j as usize] as u64
                } else {
                    let back = s as i64 - i as i64;
                    let k = hist.len() as i64 - back;
                    if k >= 0 && (k as usize) < hist.len() {
                        hist[k as usize] as u64
                    } else {
                        eos as u64
                    }
                };
                ctx[s] = if cut { eos as u64 } else { prev };
                if ctx[s] == eos as u64 {
                    cut = true;
                }
            }
            for n in 2..=ngram {
                let mut mixed = ctx[0].wrapping_mul(hp.ple_multipliers[0]);
                for j in 1..n {
                    mixed ^= ctx[j].wrapping_mul(hp.ple_multipliers[j]);
                }
                let base = (n - 2) * hp.ple_heads_per_ngram;
                for g in 0..hp.ple_heads_per_ngram {
                    let h = base + g;
                    rows.push(
                        (mixed % hp.ple_head_vocab_sizes[h] + hp.ple_head_offsets[h]) as u32,
                    );
                }
            }
            hist.push(tok);
            if hist.len() > ngram - 1 {
                let cut = hist.len() - (ngram - 1);
                hist.drain(..cut);
            }
        }
        seq.ple_hist = hist;
        seq.ple_next_pos = seq.pos + tokens.len() as u32;
        rows
    }

