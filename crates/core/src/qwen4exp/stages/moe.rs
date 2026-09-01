//! moe(512 전문가 top-10 + shared) 스테이지 — Engine4에서 분리 (리팩토링 P1, 2026-09-01).
//! 수치 경로 불변 — 이동만. Ctx 기반 백엔드 독립 (CPU/GPU 동일 코드).

use super::super::Q4Error;
use super::Ctx;
use crate::ops::{sigmoid, silu};
use llm170_profiler::profile_span;

    /// MoE FFN — top-10 라우팅(softmax→정규화) + shared(sigmoid 게이트).
    /// MoE FFN — 토큰-전문가 그룬핑 배치: 라우터는 전 토큰 배치, 각 전문가는
    /// 자기 토큰 서브배치로 3 role 배치 GEMM. 호출 수가 토큰 수와 무관하게
    /// 전문가 수(512)×3 + 공유 3 + 라우터 2로 고정 — GPU 장문 prefill의 핵심.
    pub fn moe_ffn(ctx: &Ctx, il: usize, xs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, Q4Error> {
        profile_span!("q4::moe");
        let hp = ctx.model.hp.clone();
        let w_route = ctx.model.w4(&format!("blk.{il}.ffn_gate_inp.weight"))?;
        let w_route_sh = ctx.model.w4(&format!("blk.{il}.ffn_gate_inp_shexp.weight"))?;
        let sh_up = ctx.model.w4(&format!("blk.{il}.ffn_up_shexp.weight"))?;
        let sh_gate = ctx.model.w4(&format!("blk.{il}.ffn_gate_shexp.weight"))?;
        let sh_down = ctx.model.w4(&format!("blk.{il}.ffn_down_shexp.weight"))?;
        let n_exp = hp.n_expert;
        let n_used = hp.n_expert_used;
        let n_ff = hp.n_ff_exp;
        let n_embd = hp.n_embd;
        let t = xs.len();

        // 서브스테이지 계량 (LLM170_Q4_TIME) — 190ms의 내부 분해용 (2026-09-01).
        let tm = std::env::var_os("LLM170_Q4_TIME").is_some();
        let t_route0 = std::time::Instant::now();
        // 1) 라우팅 — 전 토큰 배치 1회
        let mut route = vec![vec![0.0f32; n_exp]; t];
        ctx.mm_batch(xs, &w_route, &mut route)?;
        let mut sgate_all = vec![vec![0.0f32; 1]; t];
        ctx.mm_batch(xs, &w_route_sh, &mut sgate_all)?;

        let t_route = t_route0.elapsed();
        let t_sel0 = std::time::Instant::now();
        // 2) 선택 — 전문가별 (토큰, 가중치) 리스트
        let mut by_expert: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_exp];
        for (ti, logits) in route.iter_mut().enumerate() {
            let mx = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut zs = 0.0f32;
            for v in logits.iter_mut() {
                *v = (*v - mx).exp();
                zs += *v;
            }
            for v in logits.iter_mut() {
                *v /= zs;
            }
            let mut idx: Vec<usize> = (0..n_exp).collect();
            idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
            let sel = &idx[..n_used];
            let mut wsum: f32 = sel.iter().map(|&e| logits[e]).sum();
            wsum = wsum.max(6.103515625e-5);
            for &e in sel {
                let w = logits[e] / wsum;
                if w != 0.0 {
                    by_expert[e].push((ti, w));
                }
            }
        }

        let t_sel = t_sel0.elapsed();
        let t_gemm0 = std::time::Instant::now();
        // 3) 전문가별 서브배치 — 512×3 배치 GEMM (빈 전문가 스킵)
        let mut out = vec![vec![0.0f32; n_embd]; t];
        let trace = std::env::var_os("LLM170_Q4_TRACE").is_some();
        if trace && route.iter().flatten().any(|x| !x.is_finite()) {
            eprintln!("# NaN route logits (입력은 finite여야 함)");
            std::process::exit(101);
        }
        // 디코드 t=1 빠른 경로: 선택 전문가들의 gate·up가 동일 입력 — 그룹 1호출로
        // 2×n_used회 왕복을 1회로 (실측 병목: 전문가당 GPU 왕복 1,440회/스텝).
        let nofast = std::env::var_os("LLM170_Q4_NOFAST").is_some();
        if t == 1 && !nofast {
            let sel: Vec<usize> = (0..n_exp).filter(|&e| !by_expert[e].is_empty()).collect();
            let n_sel = sel.len();
            let mut gate_y = vec![vec![0.0f32; n_ff]; n_sel];
            let mut up_y = vec![vec![0.0f32; n_ff]; n_sel];
            {
                let mut ws: Vec<crate::matmul::Weight> = Vec::with_capacity(2 * n_sel);
                for &e in &sel {
                    ws.push(ctx.model.expert_w(&format!("blk.{il}.ffn_gate_exps.weight"), e)?);
                    ws.push(ctx.model.expert_w(&format!("blk.{il}.ffn_up_exps.weight"), e)?);
                }
                let mut gu: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0f32; n_ff]; 1]; 2 * n_sel];
                ctx.mm_group(xs, &ws, &mut gu)?;
                for i in 0..n_sel {
                    gate_y[i] = gu[2 * i][0].clone();
                    up_y[i] = gu[2 * i + 1][0].clone();
                }
            }
            for r in gate_y.iter_mut() {
                for i in 0..n_ff {
                    r[i] = silu(r[i]);
                }
            }
            for (r, u) in gate_y.iter_mut().zip(up_y.iter()) {
                for i in 0..n_ff {
                    r[i] *= u[i];
                }
            }
            {
                let mut eos = vec![vec![0.0f32; n_embd]; n_sel];
                // P2-2: K전문가 down 1런치 (스택 뷰). 미지원/호스트 폴백은 짝으로.
                let mut batched = false;
                // 배치 down은 스택 전체(345MB/층) 상주를 요구해 예산을 악화시킴
                // (실측: moe 190→720ms, 터치 슬라이스 합계 ~1.2GB인데 스택은
                // 16.5GB). 기본 끔 — LLM170_MOE_BATCH=1 (전체 상주 가능한
                // CMP 40GB+ 또는 LRU 스트리밍 도입시).
                let batch_on = std::env::var_os("LLM170_MOE_BATCH").is_some()
                    && std::env::var_os("LLM170_MOE_CPU").is_none();
                if batch_on {
                    if let Some(acc) = ctx.acc {
                        let stack = ctx.model.w4(&format!("blk.{il}.ffn_down_exps.weight"))?;
                        let ids: Vec<u32> = sel.iter().map(|&e| e as u32).collect();
                        if acc.moe_down(&gate_y[..n_sel], &stack, &ids, n_exp, &mut eos).is_ok() {
                            batched = true;
                        }
                    }
                }
                // mm_paired 폴백은 batch_on과 무관하게 항상 대기 — ed89e84가
                // 이 블록을 if batch_on 내부로 잘못 중첩해 기본(t=1 fast) 경로의
                // 라우팅 전문가 기여가 0이던 회귀 (2026-09-01 프레임 대조로 발견).
                if !batched {
                    let mut wds = Vec::with_capacity(n_sel);
                    for &e in &sel {
                        wds.push(ctx.model.expert_w(&format!("blk.{il}.ffn_down_exps.weight"), e)?);
                    }
                    ctx.mm_paired(&gate_y[..n_sel], &wds, &mut eos)?;
                }
                for (k, &e) in sel.iter().enumerate() {
                    let (ti, w) = by_expert[e][0];
                    let o = &mut out[ti];
                    for i in 0..n_embd {
                        o[i] += w * eos[k][i];
                    }
                }
            }
            // shared 전문가 — gate·up 동일 입력 xs 그룹 1호출
            let mut sh_gate_y = vec![vec![0.0f32; n_ff]; t];
            let mut sh_up_y = vec![vec![0.0f32; n_ff]; t];
            {
                let mut gi = vec![
                    std::mem::take(&mut sh_gate_y),
                    std::mem::take(&mut sh_up_y),
                ];
                ctx.mm_group(xs, &[sh_gate, sh_up], &mut gi)?;
                sh_gate_y = std::mem::take(&mut gi[0]);
                sh_up_y = std::mem::take(&mut gi[1]);
            }
            for r in sh_gate_y.iter_mut() {
                for i in 0..n_ff {
                    r[i] = silu(r[i]);
                }
            }
            for (r, u) in sh_gate_y.iter_mut().zip(sh_up_y.iter()) {
                for i in 0..n_ff {
                    r[i] *= u[i];
                }
            }
            let mut shout = vec![vec![0.0f32; n_embd]; t];
            ctx.mm_batch(&sh_gate_y, &sh_down, &mut shout)?;
            for (ti, o) in out.iter_mut().enumerate() {
                let sh_w = sigmoid(sgate_all[ti][0]);
                for i in 0..n_embd {
                    o[i] += sh_w * shout[ti][i];
                }
            }
            return Ok(out);
        }
        // 프리필 그룹 경로 (03 §3.3): 토큰-메이저 (ti,e,w) 페어로 gate·up·down
        // 스택 GEMM 3런치 — 전문가별 서브배치(≤n_exp×3회 왕복) 대신.
        // LLM170_MOE_BATCH=1(스택 상주 예산 확보)에서만 — 페어 정렬은
        // 토큰 메이저·전문가 오름차순(전문가별 경로의 토큰별 누산 순서와 동일).
        let mut grouped_done = false;
        let batch_on = std::env::var_os("LLM170_MOE_BATCH").is_some()
            && std::env::var_os("LLM170_MOE_CPU").is_none();
        if t > 1 && batch_on {
            if let Some(acc) = ctx.acc {
                let mut pairs: Vec<(usize, usize, f32)> = Vec::with_capacity(t * n_used);
                for e in 0..n_exp {
                    for &(ti, w) in &by_expert[e] {
                        pairs.push((ti, e, w));
                    }
                }
                pairs.sort_by(|&a, &b| (a.0, a.1).cmp(&(b.0, b.1)));
                let np_ = pairs.len();
                let ids: Vec<u32> = pairs.iter().map(|&(_, e, _)| e as u32).collect();
                let xp: Vec<Vec<f32>> = pairs.iter().map(|&(ti, _, _)| xs[ti].clone()).collect();
                let wg_stack = ctx.model.w4(&format!("blk.{il}.ffn_gate_exps.weight"))?;
                let wu_stack = ctx.model.w4(&format!("blk.{il}.ffn_up_exps.weight"))?;
                let wd_stack = ctx.model.w4(&format!("blk.{il}.ffn_down_exps.weight"))?;
                let mut gate_y = vec![vec![0.0f32; n_ff]; np_];
                let mut up_y = vec![vec![0.0f32; n_ff]; np_];
                if acc.moe_down(&xp, &wg_stack, &ids, n_exp, &mut gate_y).is_ok()
                    && acc.moe_down(&xp, &wu_stack, &ids, n_exp, &mut up_y).is_ok()
                {
                    for r in gate_y.iter_mut() {
                        for i in 0..n_ff {
                            r[i] = silu(r[i]);
                        }
                    }
                    for (r, u) in gate_y.iter_mut().zip(up_y.iter()) {
                        for i in 0..n_ff {
                            r[i] *= u[i];
                        }
                    }
                    let mut y = vec![vec![0.0f32; n_embd]; np_];
                    if acc.moe_down(&gate_y, &wd_stack, &ids, n_exp, &mut y).is_ok() {
                        for ((ti, _, w), yo) in pairs.iter().zip(y.iter()) {
                            let o = &mut out[*ti];
                            for i in 0..n_embd {
                                o[i] += w * yo[i];
                            }
                        }
                        grouped_done = true;
                    }
                }
            }
        }
        if !grouped_done {
        for e in 0..n_exp {
            let list = &by_expert[e];
            if list.is_empty() {
                continue;
            }
            let sub: Vec<Vec<f32>> = list.iter().map(|&(ti, _)| xs[ti].clone()).collect();
            let mut gate_y = vec![vec![0.0f32; n_ff]; list.len()];
            let mut up_y = vec![vec![0.0f32; n_ff]; list.len()];
            let wg = ctx.model.expert_w(&format!("blk.{il}.ffn_gate_exps.weight"), e)?;
            let wu = ctx.model.expert_w(&format!("blk.{il}.ffn_up_exps.weight"), e)?;
            let wd = ctx.model.expert_w(&format!("blk.{il}.ffn_down_exps.weight"), e)?;
            ctx.mm_batch(&sub, &wg, &mut gate_y)?;
            ctx.mm_batch(&sub, &wu, &mut up_y)?;
            for r in gate_y.iter_mut() {
                for i in 0..n_ff {
                    r[i] = silu(r[i]);
                }
            }
            for (r, u) in gate_y.iter_mut().zip(up_y.iter()) {
                for i in 0..n_ff {
                    r[i] *= u[i];
                }
            }
            let mut eout = vec![vec![0.0f32; n_embd]; list.len()];
            ctx.mm_batch(&gate_y, &wd, &mut eout)?;
            if trace {
                if gate_y.iter().flatten().any(|x| !x.is_finite()) {
                    eprintln!("# NaN expert e={e} gate_y (t={})", gate_y.len());
                    std::process::exit(101);
                }
                if eout.iter().flatten().any(|x| !x.is_finite()) {
                    eprintln!("# NaN expert e={e} eout");
                    std::process::exit(101);
                }
            }
            for ((ti, w), eo) in list.iter().zip(eout.iter()) {
                let o = &mut out[*ti];
                for i in 0..n_embd {
                    o[i] += w * eo[i];
                }
            }
        }
        }

        let t_gemm = t_gemm0.elapsed();
        let t_sh0 = std::time::Instant::now();
        // 4) shared 전문가 — 전 토큰 배치, gate·up 동일 입력 그룹 1호출
        let mut sh_gate_y = vec![vec![0.0f32; n_ff]; t];
        let mut sh_up_y = vec![vec![0.0f32; n_ff]; t];
        {
            let mut gi = vec![
                std::mem::take(&mut sh_gate_y),
                std::mem::take(&mut sh_up_y),
            ];
            ctx.mm_group(xs, &[sh_gate, sh_up], &mut gi)?;
            sh_gate_y = std::mem::take(&mut gi[0]);
            sh_up_y = std::mem::take(&mut gi[1]);
        }
        for r in sh_gate_y.iter_mut() {
            for i in 0..n_ff {
                r[i] = silu(r[i]);
            }
        }
        for (r, u) in sh_gate_y.iter_mut().zip(sh_up_y.iter()) {
            for i in 0..n_ff {
                r[i] *= u[i];
            }
        }
        let mut shout = vec![vec![0.0f32; n_embd]; t];
        ctx.mm_batch(&sh_gate_y, &sh_down, &mut shout)?;
        if trace {
            if sh_gate_y.iter().flatten().any(|x| !x.is_finite()) {
                eprintln!("# NaN shared sh_gate_y t={t}");
                // 재현 덤프: 입력 xs 비트 + 층 정보
                let mut buf: Vec<u8> = Vec::with_capacity(8 + t * n_embd * 4);
                buf.extend_from_slice(&(t as u64).to_le_bytes());
                buf.extend_from_slice(&(n_embd as u64).to_le_bytes());
                for row in xs {
                    for v in row {
                        buf.extend_from_slice(&v.to_le_bytes());
                    }
                }
                let _ = std::fs::write("/tmp/q4_nan_dump.bin", buf);
                eprintln!("# xs 덤프 → /tmp/q4_nan_dump.bin (layer {il})");
                std::process::exit(101);
            }
            if shout.iter().flatten().any(|x| !x.is_finite()) {
                eprintln!("# NaN shared shout t={t}");
                std::process::exit(101);
            }
        }
        let t_sh = t_sh0.elapsed();
        for (ti, o) in out.iter_mut().enumerate() {
            let sh_w = sigmoid(sgate_all[ti][0]);
            for i in 0..n_embd {
                o[i] += sh_w * shout[ti][i];
            }
        }
        if tm {
            eprintln!(
                "# moe-sub: route {:.0}µs sel {:.0}µs experts {:.0}µs shared {:.0}µs (t={t})",
                t_route.as_micros(),
                t_sel.as_micros(),
                t_gemm.as_micros(),
                t_sh.as_micros()
            );
        }
        Ok(out)
    }

