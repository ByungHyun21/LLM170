//! ple(n-gram 해시 임베딩) 스테이지 — Engine4에서 분리 (리팩토링 P1, 2026-09-01).
//! 수치 경로 불변 — 이동만. Ctx 기반 백엔드 독립 (CPU/GPU 동일 코드).

use super::super::Q4Error;
use super::Ctx;
use super::super::layers::SeqState4;
use crate::ops::{rms_norm, sigmoid, silu};
use llm170_diag::profile_span;

    /// PLE 블록 — 해시 gather→key/value→게이트→방송→dilated conv→잔차 2경로.
    /// 순수 gather 래퍼 — 스레드에서 쓰기 위한 별칭(ple_gather_parts는 Sync).
    fn plo_gather(data: &[u8], ty: llm170_gguf::GgmlType, hd: usize, rows: &[u32], out: &mut [f32]) {
        crate::qwen4exp::ple_gather_parts(data, ty, hd, rows, out)
    }

    /// 호스트 PLE 블록의 토큰 병렬 폭 — 토큰별 산술 순서는 그대로라 수치 불변.
    fn ple_threads(t: usize) -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(t.max(1))
            .min(32)
    }

    pub fn ple_block(
        ctx: &Ctx,
        seq: &mut SeqState4,
        il: usize,
        res_hc: &mut [Vec<f32>],
        rows: &[u32],
        prefetched: Option<Vec<Vec<f32>>>,
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
        // ── PLE 하위 스테이지 계측 (05-1) — LLM170_Q4_TIME=1:
        // ple_gather(mmap 랜덤읽기+iq4_nl 디양자화)와 key/value 투영(mm) 분해.
        // 파이프라인 설계(05-2)의 병목 실측 입력.
        let tm_on = std::env::var_os("LLM170_Q4_TIME").is_some();
        let t_g0 = std::time::Instant::now();

        // emb gather [t][heads·ple_head_dim]
        let heads = hp.ple_heads_per_ngram * 2; // bigram+trigram = 16
        let emb_w = heads * hp.ple_head_dim; // 16×160 = 2560
        let mut emb = vec![vec![0.0f32; emb_w]; t];
        let hit = prefetched
            .as_ref()
            .is_some_and(|p| p.len() == t && p.iter().all(|r| r.len() == emb_w));
        let nt = ple_threads(t);
        let per = t.div_ceil(nt);
        if !hit {
            // mmap 랜덤 읽기 + 디양자화 — 토큰별 독립이므로 스레드로 나눈다
            // (수치 불변: 토큰별 결과가 그대로 emb[ti]).
            // 테이블 뷰를 미리 해석해 순수 함수(ple_gather_parts)로 호출 —
            // Model4는 내부 RefCell 캐시가 있어 스레드 간 공유가 불가능하다.
            let (tptr, tlen, tty, thd) = ctx.model.ple_table_view()?;
            let tdata: &[u8] = unsafe { std::slice::from_raw_parts(tptr as *const u8, tlen) };
            std::thread::scope(|sc| {
                let mut rest: &mut [Vec<f32>] = &mut emb;
                let mut base = 0usize;
                while base < t {
                    let take = per.min(t - base);
                    let (head, tail) = rest.split_at_mut(take);
                    rest = tail;
                    let b = base;
                    sc.spawn(move || {
                        for (i, out) in head.iter_mut().enumerate() {
                            let r = &rows[(b + i) * heads..(b + i + 1) * heads];
                            let mut flat = vec![0.0f32; emb_w];
                            plo_gather(tdata, tty, thd, r, &mut flat);
                            *out = flat;
                        }
                    });
                    base += take;
                }
            });
        } else {
            emb = prefetched.unwrap();
            if std::env::var_os("LLM170_PLE_VERIFY").is_some() {
                for (ti, r) in rows.chunks(heads).enumerate() {
                    let mut chk = vec![0.0f32; emb_w];
                    ctx.model.ple_gather(r, &mut chk)?;
                    for i in 0..emb_w {
                        if (chk[i] - emb[ti][i]).abs() > 1e-6 {
                            eprintln!("# PLE VERIFY FAIL t={t} ti={ti} i={i} pre={} chk={}", emb[ti][i], chk[i]);
                            break;
                        }
                    }
                }
                eprintln!("# PLE VERIFY t={t} done");
            }
            if std::env::var_os("LLM170_Q4_TIME").is_some() {
                eprintln!("# ple-stage t={t}: prefetch HIT (gather 스킵)");
            }
        }
        let t_gather = t_g0.elapsed();
        let t_m0 = std::time::Instant::now();
        // key/value 프로젝션
        let mut key = vec![vec![0.0f32; w_key.n_out as usize]; t];
        ctx.mm_batch(&emb, &w_key, &mut key)?;
        let mut value = vec![vec![0.0f32; w_value.n_out as usize]; t];
        ctx.mm_batch(&emb, &w_value, &mut value)?;
        if tm_on {
            eprintln!(
                "# ple-stage t={t}: gather={:.1}ms proj={:.1}ms",
                t_gather.as_secs_f64() * 1e3,
                t_m0.elapsed().as_secs_f64() * 1e3
            );
        }
        ple_stage_hash("emb", &emb);
        ple_stage_hash("key", &key);
        ple_stage_hash("value", &value);

        let mut gated_hist: Vec<Vec<f32>> = vec![Vec::new(); t];
        {
        let res_hc_ro: &[Vec<f32>] = res_hc;
        std::thread::scope(|sc| {
            let mut rest: &mut [Vec<f32>] = &mut gated_hist;
            let mut base = 0usize;
            while base < t {
                let take = per.min(t - base);
                let (head, tail) = rest.split_at_mut(take);
                rest = tail;
                let b = base;
                let (key, value) = (&key, &value);
                let (n_key, n_query, n_conv) = (&n_key, &n_query, &n_conv);
                let (res_hc_v, hp) = (res_hc_ro, &hp);
                sc.spawn(move || {
                    for (i, out) in head.iter_mut().enumerate() {
                        let ti = b + i;
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
                let head = res_hc_v[ti][s * n_embd..(s + 1) * n_embd].to_vec();
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
                        *out = normalized;
                    }
                });
                base += take;
            }
        });
        ple_stage_hash("gated", &gated_hist);
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
        std::thread::scope(|sc| {
            let mut rest: &mut [Vec<f32>] = &mut conv_out;
            let mut base = 0usize;
            while base < t {
                let take = per.min(t - base);
                let (head, tail) = rest.split_at_mut(take);
                rest = tail;
                let b = base;
                let (padded, conv_w) = (&padded, &conv_w);
                sc.spawn(move || {
                    for (i, out) in head.iter_mut().enumerate() {
                        let ti = b + i;
                        for k in 0..kern {
                            let start = hist + ti - (kern - 1 - k) * dil;
                            let src = &padded[start];
                            for c in 0..hc_dim {
                                out[c] += conv_w[c * kern + k] * src[c];
                            }
                        }
                        for c in 0..hc_dim {
                            out[c] = silu(out[c]);
                        }
                    }
                });
                base += take;
            }
        });
        ple_stage_hash("conv", &conv_out);
        // 상태 갱신: 마지막 hist 열
        for j in 0..hist {
            let src = &padded[t + j];
            st[j * hc_dim..(j + 1) * hc_dim].copy_from_slice(src);
        }

        // 잔차: hidden + gated(norm 전 방송값) + conv_out — build_ple 반환식 그대로.
        // gated_pre는 conv 블록 위에서 이미 계산했으므로 재계산 대신 저장 구조 사용:
        // (value·gate 방송은 위 루프에서 `gated`로 존재했으나 norm에 덮어씀 — 재계산)
        std::thread::scope(|sc| {
            let mut rest: &mut [Vec<f32>] = &mut res_hc[..];
            let mut base = 0usize;
            while base < t {
                let take = per.min(t - base);
                let (head, tail) = rest.split_at_mut(take);
                rest = tail;
                let b = base;
                let (key, value, conv_out) = (&key, &value, &conv_out);
                let (n_key, n_query, hp) = (&n_key, &n_query, &hp);
                sc.spawn(move || {
                    for (i, row) in head.iter_mut().enumerate() {
            let ti = b + i;
            // 게이트 재계산 (결정적 동일값)
            let mut k_n = vec![0.0f32; hc_dim];
            for s in 0..hc {
                let head = key[ti][s * n_embd..(s + 1) * n_embd].to_vec();
                k_n[s * n_embd..(s + 1) * n_embd]
                    .copy_from_slice(&rms_norm(&head, &n_key[s * n_embd..(s + 1) * n_embd], hp.eps));
            }
            let mut q_n = vec![0.0f32; hc_dim];
            for s in 0..hc {
                let head = row[s * n_embd..(s + 1) * n_embd].to_vec();
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
                    row[s * n_embd + i] += value[ti][i] * g + conv_out[ti][s * n_embd + i];
                }
            }
                    }
                });
                base += take;
            }
        });
        Ok(())
    }

    /// 진단(plans/80): ple_block 하위 스테이지 산출물 FNV 해시 — 청크 불변
    /// 결함의 하위 스테이지 특정용. `LLM170_PLE_DUMP=1`.
    fn ple_stage_hash(tag: &str, rows: &[Vec<f32>]) {
        if std::env::var_os("LLM170_PLE_DUMP").is_none() {
            return;
        }
        let mut h = [0xcbf29ce484222325u64; 4];
        for (i, r) in rows.iter().enumerate().take(64) {
            let w = i / 16;
            for v in r.iter() {
                h[w] = h[w].wrapping_mul(0x100000001b3) ^ (v.to_bits() as u64);
            }
        }
        eprintln!(
            "[pleh] {tag} n={} h1={:016x} h2={:016x} h3={:016x} h4={:016x}",
            rows.len(), h[0], h[1], h[2], h[3]
        );
    }
    /// PLE n-gram 해시 — 호스트 u64 (ctx[s]=직전 s토큰, EOS 절단).
    pub fn ple_hash(ctx: &Ctx, seq: &mut SeqState4, tokens: &[u32]) -> Vec<u32> {
        if std::env::var_os("LLM170_PLE_DUMP").is_some() {
            eprintln!(
                "[plehash] pos={} next_pos={} hist={:?} hist_valid={}",
                seq.pos, seq.ple_next_pos, seq.ple_hist, seq.ple_next_pos == seq.pos
            );
        }
        let hp = ctx.model.hp.clone();
        let ngram = hp.ple_ngram;
        let heads = hp.ple_heads_per_ngram * 2; // bigram+trigram = 16
        let eos = hp.ple_eos;
        let hist0: Vec<u32> = seq.ple_hist.clone();
        let hist_valid = seq.ple_next_pos == seq.pos;
        let mut hist: Vec<u32> = if hist_valid { hist0.clone() } else { vec![eos; ngram - 1] };
        let mut rows = Vec::with_capacity(tokens.len() * heads);
        for (i, &tok) in tokens.iter().enumerate() {
            let mut ctx = vec![tok as u64; ngram];
            let mut cut = false;
            for s in 1..ngram {
                let j = i as i64 - s as i64;
                let prev: u64 = if j >= 0 {
                    tokens[j as usize] as u64
                } else {
                    // 청크 경계 lookback은 **호출 시작 스냅샷**(hist0)에서 읽는다.
                    // 라이브 hist엔 이번 호출의 토큰이 이미 push되어 있어, 경계
                    // 토큰의 trigram이 잘못된 선행 토큰을 참조한다(2026-09-18,
                    // plans/80 — 청크 불변성 결함의 뿌리).
                    let back = s as i64 - i as i64;
                    let k = hist0.len() as i64 - back;
                    if k >= 0 && (k as usize) < hist0.len() {
                        hist0[k as usize] as u64
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
        if std::env::var_os("LLM170_PLE_DUMP").is_some() {
            eprintln!("[plerows] {:?}", &rows[..rows.len().min(3 * heads)]);
            if rows.len() > 16 * heads {
                let w: Vec<u32> = rows[16 * heads..16 * heads + 3 * heads].to_vec();
                eprintln!("[plerows16] {w:?}");
            }
        }
        seq.ple_hist = hist;
        seq.ple_next_pos = seq.pos + tokens.len() as u32;
        rows
    }


