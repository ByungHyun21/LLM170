//! 프레임 — 43블록 프리필 순방향 + hc_head → norm → head + DSpark MTP 코어.
//!
//! model.py `Transformer.forward` 재현(프리필 start_pos=0 경로):
//! embed → 4스트림 방송 → 블록×43 → hc_head → norm → head(fp32 로짓).
//! 트렁크 40-42층의 4스트림 평균을 DSpark main_hidden으로 수집(§7).
//!
//! DSpark(보고서 §7, B4 계약 — 코어 함수만 이곳):
//! - main_x = main_norm(main_proj(concat(트렁크 스트림 평균)))
//! - 드래프트 입력 [입력토큰, noise×(block-1)] — 공유 embed, 4스트림 방송
//! - 프리필: main_x로 윈도우 KV 워밍(링). 디코드: [윈도우 | 드래프트 kv]
//!   어텐션+싱크+비회전 그룹 출력 — compress_ratio 0 강제(윈도우만)
//! - 종결: hc_head → norm → 공유 head → 마르코프 바이어스
//!   logits[:,i] += w2(w1(out_ids[:,i])) (rank 256) → 6토큰;
//!   신뢰도 proj [4096+256→1] fp32.

use crate::deepseek4::config::Deepseek4Config;
use crate::deepseek4::loader::{Ds4Error, Ds4Loader, LinearW};
use crate::deepseek4::ops::{
    RopeTable, bf16_round, fp8_sim, gemm_nt, rms_norm_weighted, rope_apply,
};
use crate::deepseek4::stages::attn::{
    LayerAttn, attention_output, project_kv, project_q, sparse_attn_one,
};
use crate::deepseek4::stages::hc::hc_head;
use llm170_exl3::Exl3Linear;

/// 프레임 순방향 결과.
pub struct ForwardOut {
    /// 마지막 토큰 로짓 [vocab] — f32(반올림 없음, 참조 head와 동일).
    pub logits_last: Vec<f32>,
    /// 트렁크 40-42 스트림 평균 concat [t × 3·dim] — DSpark main 입력.
    pub main_hidden: Vec<f32>,
    /// 최종 hc_head 직후 은닉(마지막 토큰) — 디버그/프루브용.
    pub last_hidden: Vec<f32>,
}

/// 헤드 스트립 gemv — head(129280×4096) 전체 재료화 회피.
/// n스트립(128열)별로 k블록을 순차 디양자화해 부분합: y[j]에 k블록
/// 오름차순, 블록 내 행 오름차순으로 가산(스트립 gemv 계약 — CUDA 헤드
/// 모듈이 이 순서를 미러링한다). 스레드 병렬(열 스트립 독립).
pub fn head_gemv_stripwise(lin: &Exl3Linear, x: &[f32]) -> Vec<f32> {
    let (k, n) = (lin.k, lin.n);
    assert_eq!(x.len(), k);
    let mut y = vec![0.0f32; n];
    let view = lin.view();
    let threads = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1)
        .min(n / 128)
        .max(1);
    // 열 스트립(128열)을 스레드별 그룹으로 분할 — 각 &mut 청크 1회 이동.
    // 스트립 내 누산: k블록 오름차순, 블록 내 행 오름차순(계약 순서).
    let nstrips = n / 128;
    let per = nstrips.div_ceil(threads);
    let chunks: Vec<&mut [f32]> = y.chunks_mut(128).collect();
    let mut groups: Vec<Vec<&mut [f32]>> = (0..threads).map(|_| Vec::new()).collect();
    for (ci, c) in chunks.into_iter().enumerate() {
        groups[ci * threads / nstrips].push(c);
    }
    std::thread::scope(|sc| {
        let v = &view;
        let mut handles = Vec::new();
        for (gi, group) in groups.into_iter().enumerate() {
            let handle = sc.spawn(move || {
                for (si, ycol) in group.into_iter().enumerate() {
                    let strip = gi * per + si;
                    if strip >= nstrips {
                        break;
                    }
                    let n0 = strip * 128;
                    for k0 in (0..k).step_by(128) {
                        let blk = v.dequant_block_f64(k0, n0, 128, 128);
                        for j in 0..128 {
                            let mut acc = 0.0f32;
                            for i in 0..128 {
                                acc += x[k0 + i] * blk[i * 128 + j] as f32;
                            }
                            ycol[j] += acc;
                        }
                    }
                }
            });
            handles.push(handle);
        }
        for h in handles {
            h.join().expect("head gemv 스레드");
        }
    });
    y
}

/// Ds4Frame — 로더 위 순방향 오케스트레이션.
pub struct Ds4Frame<'a> {
    pub loader: &'a Ds4Loader,
}

impl Ds4Frame<'_> {
    pub fn new(loader: &Ds4Loader) -> Ds4Frame<'_> {
        Ds4Frame { loader }
    }

    /// 층별 rope 테이블 2종 생성 — 윈도우(10000·비YaRN) / 압축(160000·YaRN).
    fn rope_tables(&self, len: usize) -> (RopeTable, RopeTable) {
        let c = &self.loader.cfg;
        let win = RopeTable::build(
            c.rope_head_dim,
            len,
            c.rope_theta,
            false,
            c.yarn_factor,
            c.yarn_orig_len,
            c.yarn_beta_fast,
            c.yarn_beta_slow,
        );
        let cmp = RopeTable::build(
            c.rope_head_dim,
            len,
            c.compress_rope_theta,
            true,
            c.yarn_factor,
            c.yarn_orig_len,
            c.yarn_beta_fast,
            c.yarn_beta_slow,
        );
        (win, cmp)
    }

    /// 43블록 프리필 — 블록별 가중치 지연 적재(사용 후 해제).
    pub fn forward(&self, tokens: &[u32]) -> Result<ForwardOut, Ds4Error> {
        let l = self.loader;
        let cfg = &l.cfg;
        let (d, hc, t) = (cfg.dim, cfg.hc_mult, tokens.len());
        let (rope_win, rope_cmp) = self.rope_tables(t);
        // embed → 4스트림 방송.
        let rows = l.embed_rows(tokens)?;
        let mut x = vec![0.0f32; t * hc * d];
        for (i, row) in rows.iter().enumerate() {
            for j in 0..hc {
                x[i * hc * d + j * d..i * hc * d + (j + 1) * d].copy_from_slice(row);
            }
        }
        let mut main_hiddens: Vec<Vec<f32>> = Vec::new();
        for il in 0..cfg.n_layers {
            let bw = l.block(il)?;
            let rope = if cfg.ratio(il) == 0 {
                &rope_win
            } else {
                &rope_cmp
            };
            let x2 = crate::deepseek4::layers::block_forward(&bw, &x, tokens, cfg, rope, |e| {
                l.expert(il, e).expect("전문가 디양자화")
            });
            // 트렁크 타깃 층 — 스트림 평균(bf16 경계 — h.mean(dim=2) type_as).
            if cfg.dspark_targets.contains(&il) {
                let mut m = vec![0.0f32; t * d];
                for i in 0..t {
                    for j in 0..hc {
                        for dd in 0..d {
                            m[i * d + dd] += x2[i * hc * d + j * d + dd];
                        }
                    }
                    for dd in 0..d {
                        m[i * d + dd] = bf16_round(m[i * d + dd] / hc as f32);
                    }
                }
                main_hiddens.push(m);
            }
            x = x2;
            if x.iter().any(|v| !v.is_finite()) {
                return Err(Ds4Error::BadTensor(format!("블록 {il} 비유한 출력")));
            }
        }
        // hc_head → norm → head(마지막 토큰) — hc_head는 토큰 단위 함수.
        let hh = l.final_hc_head()?;
        let norm_w = l.final_norm()?;
        let mut hmix = vec![0.0f32; t * d];
        for i in 0..t {
            let yi = hc_head(
                &x[i * hc * d..(i + 1) * hc * d],
                d,
                hc,
                &hh,
                cfg.rms_eps,
                cfg.hc_eps,
            );
            hmix[i * d..(i + 1) * d].copy_from_slice(&yi);
        }
        let last = rms_norm_weighted(&hmix[(t - 1) * d..t * d], &norm_w, cfg.rms_eps);
        let head = l.head_linear()?;
        let logits_last = head_gemv_stripwise(&head, &last);
        let mut main_hidden = vec![0.0f32; t * d * main_hiddens.len()];
        for (li, mh) in main_hiddens.iter().enumerate() {
            for i in 0..t * d {
                main_hidden[i * main_hiddens.len() + li] = mh[i];
            }
        }
        Ok(ForwardOut {
            logits_last,
            main_hidden,
            last_hidden: last.to_vec(),
        })
    }
}

// ---------------- DSpark MTP 코어 (B4 계약, plans/130 §4) ----------------

/// main_x = main_norm(main_proj(main_hidden)) — FP8-sim 활성 main_proj.
pub fn dspark_main_x(
    main_hidden: &[f32],
    proj: &LinearW,
    norm_w: &[f32],
    cfg: &Deepseek4Config,
) -> Vec<f32> {
    let d = cfg.dim;
    let t = main_hidden.len() / proj.k;
    let mut xq = main_hidden.to_vec();
    fp8_sim(&mut xq, proj.k, 128);
    let mut y = vec![0.0f32; t * d];
    gemm_nt(&xq, &proj.w, proj.k, d, &mut y);
    crate::deepseek4::ops::bf16_round_slice(&mut y);
    let mut out = Vec::with_capacity(t * d);
    for i in 0..t {
        out.extend_from_slice(&rms_norm_weighted(
            &y[i * d..(i + 1) * d],
            norm_w,
            cfg.rms_eps,
        ));
    }
    out
}

/// 드래프트 입력 토큰 — [token, noise×(block-1)] (block_size 5 → 5개).
pub fn dspark_draft_ids(token: u32, cfg: &Deepseek4Config) -> Vec<u32> {
    let mut v = vec![cfg.dspark_noise_token; cfg.dspark_block];
    v[0] = token;
    v
}

/// DSpark 프리필 워밍 — main_x로 윈도우 링 채움(t ≤ win 가정; 초과 시
/// 마지막 win개를 링 순서로 회전 배치 — model.py prefill 분기).
/// 반환: 링 [win × head_dim].
pub fn dspark_window_warm(
    la: &LayerAttn,
    main_x: &[f32],
    cfg: &Deepseek4Config,
    rope: &RopeTable,
) -> Vec<f32> {
    let (d, hd, win) = (cfg.dim, cfg.head_dim, cfg.window);
    let t = main_x.len() / d;
    let kv = project_kv(&la.w, main_x, cfg, rope);
    let mut ring = vec![0.0f32; win * hd];
    if t <= win {
        ring[..t * hd].copy_from_slice(&kv);
    } else {
        let cutoff = t % win;
        // cache[cutoff:win] = kv[-win:][..win-cutoff]; cache[:cutoff] = 나머지.
        let tail = &kv[(t - win) * hd..];
        ring[cutoff * hd..].copy_from_slice(&tail[..(win - cutoff) * hd]);
        ring[..cutoff * hd].copy_from_slice(&tail[(win - cutoff) * hd..]);
    }
    ring
}

/// DSpark 디코드 어텐션 — 드래프트 b토큰(블록) 처리. 현재 메인 토큰의 kv를
/// 링 pos%win 슬롯에 기입 후 [링 | 드래프트 kv]에 대해 윈도우+싱크 어텐션.
/// 드래프트 토큰 위치는 pos+1..pos+b(rope). 반환 [b × dim].
pub fn dspark_decode_attn(
    la: &LayerAttn,
    x_draft: &[f32],
    main_x_tok: &[f32],
    ring: &mut [f32],
    pos: usize,
    cfg: &Deepseek4Config,
    rope: &RopeTable,
) -> Vec<f32> {
    let (d, hd, nh, win, rd) = (
        cfg.dim,
        cfg.head_dim,
        cfg.n_heads,
        cfg.window,
        cfg.rope_head_dim,
    );
    let b = x_draft.len() / d;
    // 메인 토큰 kv — 링 기입.
    let mkv = project_kv(&la.w, main_x_tok, cfg, rope);
    ring[pos % win * hd..(pos % win + 1) * hd].copy_from_slice(&mkv);
    // 드래프트 q/kv — 위치 pos+1..pos+b.
    let (_, mut q) = project_q(&la.w, x_draft, cfg);
    for i in 0..b {
        for h in 0..nh {
            let base = i * nh * hd + h * hd;
            rope_apply(
                &mut q[base + hd - rd..base + hd],
                rope.at(pos + 1 + i),
                false,
            );
            for v in q[base + hd - rd..base + hd].iter_mut() {
                *v = bf16_round(*v);
            }
        }
    }
    // 드래프트 kv — norm까지만 프로젝션 후 위치 오프셋(pos+1..pos+b) 로프
    // 수동 적용(project_kv는 0..b 위치를 쓰므로 언로프 프로젝션 사용).
    let mut kv_all = ring.to_vec();
    let mut kvd = project_kv_unroped(&la.w, x_draft, cfg);
    for i in 0..b {
        rope_apply(
            &mut kvd[i * hd + hd - rd..(i + 1) * hd],
            rope.at(pos + 1 + i),
            false,
        );
        for v in kvd[i * hd + hd - rd..(i + 1) * hd].iter_mut() {
            *v = bf16_round(*v);
        }
    }
    kv_all.extend_from_slice(&kvd);
    // topk = [0..min(win, pos+1)] ++ [win + 0..b].
    let mut idxs: Vec<i32> = (0..win.min(pos + 1)).map(|v| v as i32).collect();
    idxs.extend((0..b).map(|v| (win + v) as i32));
    let mut out = vec![0.0f32; b * d];
    let mut o = vec![0.0f32; nh * hd];
    for i in 0..b {
        let ot = sparse_attn_one(
            &q[i * nh * hd..(i + 1) * nh * hd],
            &kv_all,
            &idxs,
            &la.w.sink,
            cfg,
        );
        o.copy_from_slice(&ot);
        for h in 0..nh {
            let base = h * hd;
            rope_apply(
                &mut o[base + hd - rd..base + hd],
                rope.at(pos + 1 + i),
                true,
            );
            for v in o[base + hd - rd..base + hd].iter_mut() {
                *v = bf16_round(*v);
            }
        }
        let y = attention_output(&o, &la.w, cfg);
        out[i * d..(i + 1) * d].copy_from_slice(&y);
    }
    out
}

/// 로프 없는 KV 프로젝션(norm까지만) — DSpark 디코드용(위치 오프셋 수동).
fn project_kv_unroped(
    w: &crate::deepseek4::stages::attn::AttnWeights,
    x: &[f32],
    cfg: &Deepseek4Config,
) -> Vec<f32> {
    let (d, hd) = (cfg.dim, cfg.head_dim);
    let t = x.len() / d;
    let mut xq = x.to_vec();
    fp8_sim(&mut xq, d, 128);
    let mut kv = vec![0.0f32; t * hd];
    gemm_nt(&xq, &w.wkv, d, hd, &mut kv);
    crate::deepseek4::ops::bf16_round_slice(&mut kv);
    let mut out = Vec::with_capacity(t * hd);
    for i in 0..t {
        out.extend_from_slice(&rms_norm_weighted(
            &kv[i * hd..(i + 1) * hd],
            &w.kv_norm,
            cfg.rms_eps,
        ));
    }
    out
}

/// 마르코프 바이어스 — logits_bias = w2 · w1행 [rank], f32 gemv.
pub fn markov_logits_bias(markov_embed: &[f32], w2: &LinearW) -> Vec<f32> {
    assert_eq!(markov_embed.len(), w2.k);
    let mut y = vec![0.0f32; w2.n];
    for (kk, &e) in markov_embed.iter().enumerate() {
        let row = &w2.w[kk * w2.n..(kk + 1) * w2.n];
        for (yi, &wv) in y.iter_mut().zip(row.iter()) {
            *yi += e * wv;
        }
    }
    y
}

/// 신뢰도 스코어 — proj [4096+256 → 1] f32, 입력 cat(x, markov_embed).
pub fn confidence_score(x: &[f32], markov_embed: &[f32], proj: &[f32]) -> f32 {
    assert_eq!(proj.len(), x.len() + markov_embed.len());
    let mut acc = 0.0f32;
    for (&v, &p) in x.iter().chain(markov_embed.iter()).zip(proj.iter()) {
        acc += v * p;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_dir() -> Option<std::path::PathBuf> {
        let p = llm170_diag::flag::val("LLM170_DS4_EXL3")
            .unwrap_or("D:/models/DeepSeek-V4-Flash-Vision-Exp-exl3-3.04bpw");
        let p = std::path::PathBuf::from(p);
        p.exists().then_some(p)
    }

    /// 스모크 1 — L0(해시-SWA) 블록 실측 가중치 1블록 통과.
    /// 유한 + 상대 크기 상한. #[ignore] — 실측 픽스처(수 분).
    #[test]
    #[ignore]
    fn smoke_block_l0() {
        let Some(dir) = fixture_dir() else {
            eprintln!("skip: 픽스처 없음");
            return;
        };
        let l = Ds4Loader::open(&dir).expect("open");
        let cfg = &l.cfg;
        let bw = l.block(0).expect("block 0");
        assert!(bw.is_hash);
        let toks = [5u32, 1000, 99999, 128799];
        let rows = l.embed_rows(&toks).expect("embed");
        let (hc, d) = (cfg.hc_mult, cfg.dim);
        let mut x = vec![0.0f32; toks.len() * hc * d];
        for (i, r) in rows.iter().enumerate() {
            for j in 0..hc {
                x[i * hc * d + j * d..i * hc * d + (j + 1) * d].copy_from_slice(r);
            }
        }
        let rope = RopeTable::build(
            cfg.rope_head_dim,
            toks.len(),
            cfg.rope_theta,
            false,
            cfg.yarn_factor,
            cfg.yarn_orig_len,
            cfg.yarn_beta_fast,
            cfg.yarn_beta_slow,
        );
        let y = crate::deepseek4::layers::block_forward(&bw, &x, &toks, cfg, &rope, |e| {
            l.expert(0, e).expect("expert")
        });
        assert!(y.iter().all(|v| v.is_finite()));
        let mx = y.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        assert!(mx < 1e3, "L0 출력 크기 비정상: {mx}");
        eprintln!(
            "smoke_block_l0: |x|max={mx:.3}, 유한 OK ({} 토큰)",
            toks.len()
        );
    }

    /// 스모크 2 — 43층 전체 순방향(8토큰) + 마지막 토큰 로짓.
    /// 로짓 유한·상대 크기, main_hidden 형상. #[ignore] — 수십 분 가능(release 권장).
    #[test]
    #[ignore]
    fn smoke_forward_43l() {
        let Some(dir) = fixture_dir() else {
            eprintln!("skip: 픽스처 없음");
            return;
        };
        let l = Ds4Loader::open(&dir).expect("open");
        let cfg = &l.cfg;
        let toks: Vec<u32> = [1u32, 2, 3, 4, 5, 6, 7, 8].to_vec();
        let t0 = std::time::Instant::now();
        let out = Ds4Frame::new(&l).forward(&toks).expect("forward");
        eprintln!("43L forward {}초", t0.elapsed().as_secs_f64());
        assert_eq!(out.logits_last.len(), cfg.vocab);
        assert!(out.logits_last.iter().all(|v| v.is_finite()));
        let mx = out.logits_last.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        assert!(mx < 1e3, "로짓 크기 비정상: {mx}");
        // argmax(동점 낮은 인덱스 — 계약) + 상위 3.
        let mut idx: Vec<usize> = (0..cfg.vocab).collect();
        idx.sort_by(|&a, &b| {
            out.logits_last[b]
                .partial_cmp(&out.logits_last[a])
                .unwrap()
                .then(a.cmp(&b))
        });
        eprintln!(
            "smoke_43l: argmax={} top3={:?} logits={:?} max|logit|={mx:.2}",
            idx[0],
            &idx[..3],
            idx[..3]
                .iter()
                .map(|&i| out.logits_last[i])
                .collect::<Vec<_>>(),
        );
        assert_eq!(
            out.main_hidden.len(),
            toks.len() * cfg.dim * cfg.dspark_targets.len()
        );
        assert!(out.main_hidden.iter().all(|v| v.is_finite()));
    }

    /// 스모크 3 — DSpark 코어 함수(실측 가중치): main_proj/norm + markov +
    /// confidence. mtp.0/mtp.2 소형 텐서만 사용. #[ignore].
    #[test]
    #[ignore]
    fn smoke_dspark_core() {
        let Some(dir) = fixture_dir() else {
            eprintln!("skip: 픽스처 없음");
            return;
        };
        let l = Ds4Loader::open(&dir).expect("open");
        let cfg = &l.cfg;
        // main_x: main_proj 12288→4096 + main_norm.
        let (proj, norm_w) = l.dspark_main().expect("dspark main");
        assert_eq!((proj.k, proj.n), (3 * cfg.dim, cfg.dim));
        let mh = vec![0.05f32; 2 * proj.k];
        let mx = dspark_main_x(&mh, &proj, &norm_w, cfg);
        assert_eq!(mx.len(), 2 * cfg.dim);
        assert!(mx.iter().all(|v| v.is_finite()));
        // markov w1 행 + w2 gemv + confidence.
        let row = l.markov_w1_row(7).expect("w1 행");
        assert_eq!(row.len(), cfg.dspark_markov_rank);
        let (_, _, w2, proj_c) = l.dspark_head_parts().expect("dspark head");
        assert_eq!((w2.k, w2.n), (cfg.dspark_markov_rank, cfg.vocab));
        let bias = markov_logits_bias(&row, &w2);
        assert_eq!(bias.len(), cfg.vocab);
        assert!(bias.iter().all(|v| v.is_finite()));
        assert_eq!(proj_c.len(), cfg.dim + cfg.dspark_markov_rank);
        let c = confidence_score(&vec![0.01; cfg.dim], &row, &proj_c);
        assert!(c.is_finite());
        eprintln!(
            "smoke_dspark: main_x|max|={:.3} bias|max|={:.3} conf={c:.4}",
            mx.iter().fold(0.0f32, |a, &v| a.max(v.abs())),
            bias.iter().fold(0.0f32, |a, &v| a.max(v.abs())),
        );
    }

    /// DSpark 순수 함수 — 합성 가중치로 형상/인과 검증(픽스처 불필요).
    #[test]
    fn dspark_fns_synthetic() {
        let mut cfg = crate::deepseek4::config::test_cfg();
        cfg.dspark_block = 5;
        cfg.dspark_noise_token = 99;
        // 드래프트 id: [tok, noise×4].
        assert_eq!(dspark_draft_ids(7, &cfg), vec![7, 99, 99, 99, 99]);
    }
}
