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

use llm170_diag::profile_span;

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
    debug_assert!(q.len().is_multiple_of(h_k * t_len));
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
            d_out[dbase..(d + dbase)].copy_from_slice(&oi[..d]);
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

// ── GDN AR 헤드 병렬 풀 (plans/120 A1) ──
// exl3 직접 디코드 계측: gdn:delta 79.6ms/토큰, 그중 상당분이 토큰당
// 48층 × 48헤드 = 2304회 thread::scope OS 스폰이었다. 상수 풀로 스폰 비용
// 제거. 잡이 'static이어야 하므로 호출자 소유 버퍼는 원시 포인터로 캡처하고
// run_par가 완료 카운터 도달 시에만 반환함으로써 수명을 증명한다(아래 SAFETY).
pub mod ar_pool {
    use std::collections::VecDeque;
    use std::sync::{Arc, Condvar, Mutex, OnceLock};

    type Job = Box<dyn FnOnce() + Send + 'static>;

    struct Queue {
        jobs: Mutex<VecDeque<Job>>,
        cv: Condvar,
    }
    static QUEUE: OnceLock<Queue> = OnceLock::new();

    fn queue() -> &'static Queue {
        QUEUE.get_or_init(|| {
            let n = std::thread::available_parallelism()
                .map(|v| v.get())
                .unwrap_or(8)
                .min(64);
            for _ in 0..n {
                let worker = std::thread::Builder::new().name("gdn-ar-pool".into());
                if worker.spawn(worker_loop).is_err() {
                    break; // 잔여 워커로 진행 — 전멸 시에만 교착 가능
                }
            }
            Queue {
                jobs: Mutex::new(VecDeque::new()),
                cv: Condvar::new(),
            }
        })
    }

    fn worker_loop() {
        let Some(q) = QUEUE.get() else { return };
        loop {
            let job = {
                let mut g = q.jobs.lock().unwrap_or_else(|e| e.into_inner());
                while g.is_empty() {
                    g = q.cv.wait(g).unwrap_or_else(|e| e.into_inner());
                }
                match g.pop_front() {
                    Some(j) => j,
                    None => continue,
                }
            };
            job();
        }
    }

    /// n_jobs개 잡 분배 후 전부 완료까지 대기. 잡 패닉은 페이로드를 보관해
    /// 호출자 스레드에서 재개한다(thread::scope + join().unwrap() 의미 보존).
    pub fn run_par(n_jobs: usize, make: impl Fn(usize) -> Job) {
        let q = queue();
        let done = Arc::new((Mutex::new(0usize), Condvar::new()));
        let panic_payload: Arc<Mutex<Option<Box<dyn std::any::Any + Send>>>> =
            Arc::new(Mutex::new(None));
        for i in 0..n_jobs {
            let job = make(i);
            let d = Arc::clone(&done);
            let pp = Arc::clone(&panic_payload);
            let mut g = q.jobs.lock().unwrap_or_else(|e| e.into_inner());
            g.push_back(Box::new(move || {
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
                if let Err(p) = r
                    && let Ok(mut slot) = pp.lock()
                    && slot.is_none()
                {
                    *slot = Some(p);
                }
                let (m, c) = &*d;
                let mut g2 = m.lock().unwrap_or_else(|e| e.into_inner());
                *g2 += 1;
                c.notify_all();
            }));
            drop(g);
            q.cv.notify_one();
        }
        let (m, c) = &*done;
        let mut g = m.lock().unwrap_or_else(|e| e.into_inner());
        while *g < n_jobs {
            g = c.wait(g).unwrap_or_else(|e| e.into_inner());
        }
        drop(g);
        if let Ok(mut slot) = panic_payload.lock()
            && let Some(p) = slot.take()
        {
            std::panic::resume_unwind(p);
        }
    }
}

/// 원시 포인터의 usize 보관 래퍼 — ar_pool 잡 캡처용.
/// (usize 필드는 Send — 에디션 2021 정밀 캡처가 구조체가 아닌 필드를
/// 캡처하므로 *mut 을 직접 넣으면 Send가 깨진다.)
/// SAFETY: 주소가 가리키는 버퍼는 run_par 반환 전까지 유효하고(완료
/// 카운터 증명) pair별 접근 영역은 서로 격리된다 — gdn_ar_batch의 SAFETY 참조.
#[derive(Clone, Copy)]
struct SendPtr(usize);

/// 단일 (seq, v-head) AR 스텝. 비트동일 보존 전제(plans/120 A1):
/// · sk/o 순회를 행(kdim) 우량으로 전환하되 kdim 누적 순서와 항 표현식
///   `(s·q)·scale`·`(s·k)` 을 원문과 동일하게 유지한다(열 우량 그대로면
///   스트라이드 d 접근으로 캐시·벡터화 모두 실패).
/// · delta 갱신은 각 원소에 +1회 — 열→행 전환은 값 불변(비트동일).
#[allow(clippy::too_many_arguments)]
fn gdn_ar_head(
    st: &mut [f32],
    qs: &[f32],
    ks: &[f32],
    vs: &[f32],
    beta_h: f32,
    g_exp: f32,
    scale: f32,
    lo: &mut [f32],
) {
    let d = qs.len();
    let mut sk = vec![0.0f32; d];
    for kdim in 0..d {
        let kk = ks[kdim];
        let row = &mut st[kdim * d..kdim * d + d];
        for dv in 0..d {
            row[dv] *= g_exp;
            sk[dv] += row[dv] * kk;
        }
    }
    let mut delta = vec![0.0f32; d];
    for dv in 0..d {
        delta[dv] = (vs[dv] - sk[dv]) * beta_h;
    }
    for kdim in 0..d {
        let kd = ks[kdim];
        let row = &mut st[kdim * d..kdim * d + d];
        for dv in 0..d {
            row[dv] += kd * delta[dv];
        }
    }
    let mut o = vec![0.0f32; d];
    for kdim in 0..d {
        let qq = qs[kdim];
        let row = &st[kdim * d..kdim * d + d];
        for dv in 0..d {
            o[dv] += row[dv] * qq * scale;
        }
    }
    lo.copy_from_slice(&o);
}

// (행별 도트 풀 병렬 헬퍼 2종은 제거 — 잡당 고정비 ~3µs가 3584-길이 도트를
// 못 이겨 스칼라 대비 무차. 2026-10-02 실험, plans/120 A1 기록.)

/// 배치 디코드: 토큰 1개 × n_seqs. (build_delta_net_autoregressive / fused one_chunk)
/// 레이아웃: q/k `[B][H_k][d]`, v `[B][H_v][d]`, beta/g `[B][H_v]`,
/// states `[B][H_v][d*d]`, out `[B][H_v][d]`. (seq, v-head) 쌍별 병렬
/// (ar_pool 상수 풀 — plans/120 A1, 스폰 2304회/토큰 제거).
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

    let n_pairs = n_seqs * h_v;
    let mut local_outs = vec![0.0f32; n_pairs * d];
    {
        // SAFETY: 잡은 원시 포인터(SendPtr)만 캡처한다.
        // (1) states·local_outs·q·k·v·beta·g는 run_par 반환 전까지 유효 —
        //     run_par는 완료 카운터가 n_pairs에 도달해야만 반환한다.
        // (2) pair별 영역은 서로 겹치지 않는다: 상태는 d*d 청크, 로컬 출력은
        //     d 청크(기존 chunks_mut 분할과 동일), 입력은 읽기 전용 공유.
        let st_base = SendPtr(states.as_mut_ptr() as usize);
        let lo_base = SendPtr(local_outs.as_mut_ptr() as usize);
        let (qb, kb, vb, bb, gb) = (
            SendPtr(q.as_ptr() as usize),
            SendPtr(k.as_ptr() as usize),
            SendPtr(v.as_ptr() as usize),
            SendPtr(beta.as_ptr() as usize),
            SendPtr(g.as_ptr() as usize),
        );
        ar_pool::run_par(n_pairs, move |pair| {
            Box::new(move || unsafe {
                let st = std::slice::from_raw_parts_mut(
                    (st_base.0 as *mut f32).add(pair * d * d),
                    d * d,
                );
                let lo = std::slice::from_raw_parts_mut((lo_base.0 as *mut f32).add(pair * d), d);
                let (b, h) = (pair / h_v, pair % h_v);
                let kh = h % h_k;
                let qs =
                    std::slice::from_raw_parts((qb.0 as *const f32).add(b * k_stride + kh * d), d);
                let ks =
                    std::slice::from_raw_parts((kb.0 as *const f32).add(b * k_stride + kh * d), d);
                let vs =
                    std::slice::from_raw_parts((vb.0 as *const f32).add(b * v_stride + h * d), d);
                let beta_h = *(bb.0 as *const f32).add(b * h_v + h);
                let g_exp = crate::ops::exp_cr(*(gb.0 as *const f32).add(b * h_v + h));
                gdn_ar_head(st, qs, ks, vs, beta_h, g_exp, scale, lo);
            })
        });
    }
    for pair in 0..n_pairs {
        let (b, h) = (pair / h_v, pair % h_v);
        out[b * v_stride + h * d..b * v_stride + (h + 1) * d]
            .copy_from_slice(&local_outs[pair * d..(pair + 1) * d]);
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
