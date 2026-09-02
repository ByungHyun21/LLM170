//! Gated DeltaNet — llama.cpp 대응 구현.
//!
//! - `gdn_chunk_seq`: 청크(CS=64) prefill — **AR과 구조 동일** 수학(d = (I+A)⁻¹(βv − M)).
//!   llama CPU fused 커널(ops.cpp one_chunk, 순차 AR)과 동일 의미론임을
//!   소형 랜덤 모델 + `-ub 1` 교차검증으로 확인 (2026-08-30).
//!   참고: llama의 그래프 청크 경도(build_delta_net_chunking)는 S-항 적용 순서가
//!   달라 순차 AR과 미세하게 다른 수치를 냄 — 평탄한 랜덤 로짓에서 argmax가 갈렸음.
//!   본 엔진은 의미론 기준(순차 AR)을 따르고 실모델 토큰 스트림으로 검증.
//! - `gdn_ar_batch`: 자기회귀 디코드, 배치 = (시퀀스 × 토큰 1개).
//!
//! 헤드 매핑: GGUF V-헤드는 tiled 재배열 → **V 헤드 h ↔ K 헤드 h % H_k**
//! (fused 커널 `ik1 = iv1 % nek1`, delta-net-base repeat 모두 동일).
//!
//! 상태 S[kdim, vdim] — 두 경로 동일 레이아웃.

use llm170_profiler::profile_span;

pub const CS: usize = 64; // chunk size (비-KDA: 64)

/// 한 시퀀스 prefill. 레이아웃: q/k `[T][H_k][d]`, v `[T][H_v][d]`, beta/g `[T][H_v]`,
/// state `[H_v][d*d]` (입출), out `[T][H_v][d]`. V 헤드별 스레드 병렬.
pub fn gdn_chunk_seq(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: &[f32],
    g: &[f32],
    state: &mut [f32],
    out: &mut [f32],
    t_len: usize,
    h_k: usize,
    h_v: usize,
) {
    profile_span!("cpu::gdn_chunk");
    let d = q.len() / (h_k * t_len); // d_state (kdim == vdim)
    debug_assert!(q.len() % (h_k * t_len) == 0);
    let v_stride = h_v * d;
    let mut local_outs: Vec<Vec<f32>> = vec![vec![0.0f32; t_len * d]; h_v];
    {
        let state_chunks: Vec<&mut [f32]> = state.chunks_mut(d * d).collect();
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (h, (st, lo)) in state_chunks
                .into_iter()
                .zip(local_outs.iter_mut())
                .enumerate()
            {
                handles.push(scope.spawn(move || {
                    gdn_chunk_head(q, k, v, beta, g, st, lo, t_len, h % h_k, h, h_k, h_v, d);
                }));
            }
            for hd in handles {
                hd.join().unwrap();
            }
        });
    }
    for h in 0..h_v {
        for t in 0..t_len {
            out[t * v_stride + h * d..t * v_stride + (h + 1) * d]
                .copy_from_slice(&local_outs[h][t * d..(t + 1) * d]);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn gdn_chunk_head(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: &[f32],
    g: &[f32],
    state: &mut [f32],
    out: &mut [f32],
    t_len: usize,
    kh: usize,
    h: usize,
    h_k: usize,
    h_v: usize,
    d: usize,
) {
    let scale = 1.0f32 / (d as f32).sqrt();
    let k_stride = h_k * d;
    let v_stride = h_v * d;
    let n_chunks = t_len.div_ceil(CS);

    let st = &mut state[..d * d]; // [kdim, vdim]
    let mut qp = vec![0.0f32; CS * d];
    let mut kp = vec![0.0f32; CS * d];
    let mut vp = vec![0.0f32; CS * d];
    let mut bp = vec![0.0f32; CS];
    let mut gp = vec![0.0f32; CS];
    let mut gcs = vec![0.0f32; CS];
    let mut d_out = vec![0.0f32; CS * d];
    let mut oi = vec![0.0f32; d];

    for c in 0..n_chunks {
        let t0 = c * CS;
        let n = (t0 + CS).min(t_len) - t0;

        // 제로 패딩 복사 (delta-net-base.cpp:63-70)
        for t in 0..n {
            let src = t0 + t;
            qp[t * d..t * d + d]
                .copy_from_slice(&q[src * k_stride + kh * d..src * k_stride + kh * d + d]);
            kp[t * d..t * d + d]
                .copy_from_slice(&k[src * k_stride + kh * d..src * k_stride + kh * d + d]);
            vp[t * d..t * d + d]
                .copy_from_slice(&v[src * v_stride + h * d..src * v_stride + h * d + d]);
            bp[t] = beta[src * h_v + h];
            gp[t] = g[src * h_v + h];
        }
        for t in n..CS {
            for x in qp[t * d..t * d + d].iter_mut() {
                *x = 0.0;
            }
            for x in kp[t * d..t * d + d].iter_mut() {
                *x = 0.0;
            }
            for x in vp[t * d..t * d + d].iter_mut() {
                *x = 0.0;
            }
            bp[t] = 0.0;
            gp[t] = 0.0;
        }

        for x in qp.iter_mut() {
            *x *= scale;
        }
        let mut acc = 0.0f32;
        for t in 0..CS {
            acc += gp[t];
            gcs[t] = acc;
        }
        let g_last = gcs[CS - 1];

        // ==== 청크 수학 — AR과 구조 동일 (d = (I+A)⁻¹(βv − M)) ====
        // A[i,j] = β_i·(k_i·k_j)·e^{gcs_i−gcs_j} (i>j, 엄격 하삼각)
        // M_i    = β_i·e^{gcs_i}·(k_i·S_prev)
        // o_i    = e^{gcs_i}·(S_prevᵀ q_i) + Σ_{j≤i} (q_i·k_j)·e^{gcs_i−gcs_j}·d_j
        // S_new  = S_prev·e^{g_last} + Σ_j k_j·e^{g_last−gcs_j} ⊗ d_j
        for i in 0..n {
            let beta_i = bp[i];
            // rhs_i = β_i·v_i − β_i·e^{gcs_i}·(k_i·S_prev)
            for dv in 0..d {
                oi[dv] = beta_i * vp[i * d + dv];
            }
            if beta_i != 0.0 {
                let w0 = beta_i * gcs[i].exp();
                for s2 in 0..d {
                    let ks = kp[i * d + s2];
                    if ks == 0.0 {
                        continue;
                    }
                    let w = w0 * ks;
                    for dv in 0..d {
                        oi[dv] -= w * st[s2 * d + dv];
                    }
                }
            }
            let dbase = i * d;
            for dv in 0..d {
                d_out[dbase + dv] = oi[dv];
            }
            // 전진 대입: d_i = rhs_i − Σ_{j<i} A[i,j]·d_j
            for j in 0..i {
                let dot: f32 = (0..d).map(|s2| kp[i * d + s2] * kp[j * d + s2]).sum();
                let aij = dot * beta_i * (gcs[i] - gcs[j]).exp();
                if aij == 0.0 {
                    continue;
                }
                for dv in 0..d {
                    d_out[dbase + dv] -= aij * d_out[j * d + dv];
                }
            }
            // o_i = e^{gcs_i}·(S_prev·q_i) + Σ_{j≤i} kq[i,j]·d_j
            for dv in 0..d {
                oi[dv] = 0.0;
            }
            let qi_exp = gcs[i].exp();
            for s2 in 0..d {
                let qv = qp[i * d + s2];
                if qv == 0.0 {
                    continue;
                }
                let w = qi_exp * qv;
                for dv in 0..d {
                    oi[dv] += w * st[s2 * d + dv];
                }
            }
            for j in 0..=i {
                let dot: f32 = (0..d).map(|s2| qp[i * d + s2] * kp[j * d + s2]).sum();
                let kqij = dot * (gcs[i] - gcs[j]).exp();
                if kqij == 0.0 {
                    continue;
                }
                for dv in 0..d {
                    oi[dv] += kqij * d_out[j * d + dv];
                }
            }
            out[(t0 + i) * d..(t0 + i) * d + d].copy_from_slice(&oi);
        }
        // 상태 갱신: S ← S·e^{g_last} + Σ_j k_j·e^{g_last−gcs_j}⊗d_j
        let gl_exp = g_last.exp();
        for xv in st.iter_mut() {
            *xv *= gl_exp;
        }
        for j in 0..n {
            let w = (g_last - gcs[j]).exp();
            for s2 in 0..d {
                let kv = kp[j * d + s2] * w;
                for dv in 0..d {
                    st[s2 * d + dv] += kv * d_out[j * d + dv];
                }
            }
        }
    }
}

/// 배치 디코드: 토큰 1개 × n_seqs. (build_delta_net_autoregressive / fused one_chunk)
/// 레이아웃: q/k `[B][H_k][d]`, v `[B][H_v][d]`, beta/g `[B][H_v]`,
/// states `[B][H_v][d*d]`, out `[B][H_v][d]`. (seq, v-head) 쌍별 병렬.
pub fn gdn_ar_batch(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: &[f32],
    g: &[f32],
    states: &mut [f32],
    out: &mut [f32],
    n_seqs: usize,
    h_k: usize,
    h_v: usize,
) {
    profile_span!("cpu::gdn_ar");
    let d = q.len() / (h_k * n_seqs);
    let scale = 1.0f32 / (d as f32).sqrt();
    let k_stride = h_k * d;
    let v_stride = h_v * d;

    let mut local_outs: Vec<Vec<f32>> = vec![vec![0.0f32; d]; n_seqs * h_v];
    {
        let state_chunks: Vec<&mut [f32]> = states.chunks_mut(d * d).collect();
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (pair, (st, lo)) in state_chunks
                .into_iter()
                .zip(local_outs.iter_mut())
                .enumerate()
            {
                handles.push(scope.spawn(move || {
                    let (b, h) = (pair / h_v, pair % h_v);
                    let kh = h % h_k;
                    let qs = &q[b * k_stride + kh * d..b * k_stride + kh * d + d];
                    let ks = &k[b * k_stride + kh * d..b * k_stride + kh * d + d];
                    let vs = &v[b * v_stride + h * d..b * v_stride + h * d + d];
                    let beta_h = beta[b * h_v + h];
                    let g_exp = crate::ops::exp_cr(g[b * h_v + h]);

                    // S ← S·e^g;  sk[dv] = Σ_kdim S[kdim,dv]·k[kdim]
                    let mut sk = vec![0.0f32; d];
                    for kdim in 0..d {
                        let kk = ks[kdim];
                        for dv in 0..d {
                            let s = &mut st[kdim * d + dv];
                            *s *= g_exp;
                            sk[dv] += *s * kk;
                        }
                    }
                    // delta[dv] = (v[dv] − sk[dv])·β;  S += k⊗delta
                    for dv in 0..d {
                        let delta = (vs[dv] - sk[dv]) * beta_h;
                        for kdim in 0..d {
                            st[kdim * d + dv] += ks[kdim] * delta;
                        }
                    }
                    // o[dv] = Σ_kdim S[kdim,dv]·(q[kdim]·scale)
                    for dv in 0..d {
                        let mut o = 0.0f32;
                        for kdim in 0..d {
                            o += st[kdim * d + dv] * qs[kdim] * scale;
                        }
                        lo[dv] = o;
                    }
                }));
            }
            for hd in handles {
                hd.join().unwrap();
            }
        });
    }
    for pair in 0..n_seqs * h_v {
        let (b, h) = (pair / h_v, pair % h_v);
        out[b * v_stride + h * d..b * v_stride + (h + 1) * d].copy_from_slice(&local_outs[pair]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 강한 내부 정합성: AR(토큰별)과 chunked 결과가 일치해야 한다.
    #[test]
    fn chunked_matches_ar() {
        let h_k = 2;
        let h_v = 6;
        let d = 128;
        let t = 100;
        let mut rng = 0x1234_5678u64;
        let mut rnd = || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };

        let mut q = vec![0.0f32; t * h_k * d];
        let mut k = vec![0.0f32; t * h_k * d];
        let mut v = vec![0.0f32; t * h_v * d];
        let mut beta = vec![0.0f32; t * h_v];
        let mut g = vec![0.0f32; t * h_v];
        for x in q.iter_mut() {
            *x = rnd();
        }
        for x in k.iter_mut() {
            *x = rnd();
        }
        for x in v.iter_mut() {
            *x = rnd();
        }
        // 실제 모델 전제: q,k는 L2 정규화, β∈(0,1) sigmoid, g≤0
        for ti in 0..t {
            for h in 0..h_k {
                let b = ti * h_k * d + h * d;
                let head: Vec<f32> = q[b..b + d].to_vec();
                let nn = crate::ops::l2_norm(&head, 1e-6);
                q[b..b + d].copy_from_slice(&nn);
                let headk: Vec<f32> = k[b..b + d].to_vec();
                let nk = crate::ops::l2_norm(&headk, 1e-6);
                k[b..b + d].copy_from_slice(&nk);
            }
        }
        for (i, x) in beta.iter_mut().enumerate() {
            *x = crate::ops::sigmoid(1.5 * rnd() + ((i % 5) as f32) * 0.2 - 0.5);
        }
        for x in g.iter_mut() {
            *x = -0.1 - rnd().abs();
        }

        let mut state_ar = vec![0.0f32; h_v * d * d];
        let mut out_ar = vec![0.0f32; t * h_v * d];
        for ti in 0..t {
            gdn_ar_batch(
                &q[ti * h_k * d..(ti + 1) * h_k * d],
                &k[ti * h_k * d..(ti + 1) * h_k * d],
                &v[ti * h_v * d..(ti + 1) * h_v * d],
                &beta[ti * h_v..(ti + 1) * h_v],
                &g[ti * h_v..(ti + 1) * h_v],
                &mut state_ar,
                &mut out_ar[ti * h_v * d..(ti + 1) * h_v * d],
                1,
                h_k,
                h_v,
            );
        }

        let mut state_ch = vec![0.0f32; h_v * d * d];
        let mut out_ch = vec![0.0f32; t * h_v * d];
        gdn_chunk_seq(
            &q,
            &k,
            &v,
            &beta,
            &g,
            &mut state_ch,
            &mut out_ch,
            t,
            h_k,
            h_v,
        );

        let mut max_diff = 0.0f32;
        for i in 0..out_ar.len() {
            max_diff = max_diff.max((out_ar[i] - out_ch[i]).abs());
        }
        let mut max_state_diff = 0.0f32;
        for i in 0..state_ar.len() {
            max_state_diff = max_state_diff.max((state_ar[i] - state_ch[i]).abs());
        }
        let scale_ref = out_ch.iter().fold(0.0f32, |a, &b| a.max(b.abs())).max(1.0);
        assert!(
            max_diff / scale_ref < 2e-3,
            "출력 불일치: max_diff={max_diff}"
        );
        assert!(max_state_diff < 5e-2, "상태 불일치: {max_state_diff}");
    }
}
