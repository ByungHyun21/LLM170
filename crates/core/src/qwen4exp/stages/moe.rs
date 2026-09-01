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
                let mut wds = Vec::with_capacity(n_sel);
                for &e in &sel {
                    wds.push(ctx.model.expert_w(&format!("blk.{il}.ffn_down_exps.weight"), e)?);
                }
                let mut eos = vec![vec![0.0f32; n_embd]; n_sel];
                ctx.mm_paired(&gate_y[..n_sel], &wds, &mut eos)?;
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

