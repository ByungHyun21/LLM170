
use llm170_diag::profile_span;

pub const CS: usize = 64;

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
    let d = q.len() / (h_k * t_len);
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

    let st = &mut state[..d * d];
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

        for i in 0..n {
            let beta_i = bp[i];
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

pub mod ar_pool {
    use std::collections::VecDeque;
    use std::sync::{Arc, Condvar, Mutex, OnceLock};

    type Job = Box<dyn FnOnce() + Send + 'static>;

    struct Queue {
        jobs: Mutex<VecDeque<Job>>,
        cv: Condvar,
    }
    static QUEUE: OnceLock<Queue> = OnceLock::new();
    static SPAWNED: std::sync::Once = std::sync::Once::new();
    static SPAWNED_N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    fn queue() -> &'static Queue {
        let q = QUEUE.get_or_init(|| Queue {
            jobs: Mutex::new(VecDeque::new()),
            cv: Condvar::new(),
        });
        SPAWNED.call_once(|| {
            let n = std::thread::available_parallelism()
                .map(|v| v.get())
                .unwrap_or(8)
                .min(64);
            for _ in 0..n {
                let worker = std::thread::Builder::new().name("gdn-ar-pool".into());
                match worker.spawn(move || worker_loop(q)) {
                    Ok(_) => {
                        SPAWNED_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    Err(_) => break,
                }
            }
        });
        q
    }

    pub fn spawned_workers() -> usize {
        SPAWNED_N.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn worker_loop(q: &'static Queue) {
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

    pub fn run_par(n_jobs: usize, make: impl Fn(usize) -> Job) {
        let q = queue();
        if spawned_workers() == 0 {
            for i in 0..n_jobs {
                (make(i))();
            }
            return;
        }
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

#[derive(Clone, Copy)]
struct SendPtr(usize);

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

    #[test]
    fn ar_pool_spawns_workers_and_completes() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let done = Arc::new(AtomicUsize::new(0));
        let d = Arc::clone(&done);
        let h = std::thread::spawn(move || {
            ar_pool::run_par(64, move |i| {
                let d = Arc::clone(&d);
                Box::new(move || {
                    d.fetch_add(i + 1, Ordering::Relaxed);
                })
            });
        });
        let t0 = std::time::Instant::now();
        while !h.is_finished() {
            assert!(
                t0.elapsed().as_secs() < 30,
                "ar_pool 교착 — 워커 전멸 의심(C1 회귀)"
            );
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        h.join().unwrap();
        assert_eq!(done.load(Ordering::Relaxed), 64 * 65 / 2);
        assert!(
            ar_pool::spawned_workers() >= 1,
            "워커 0 — set-then-spawn 결함"
        );
    }

    #[test]
    fn ar_pool_concurrent_run_par() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let mut hs = Vec::new();
        for _ in 0..4 {
            hs.push(std::thread::spawn(|| {
                let done = Arc::new(AtomicUsize::new(0));
                let d = Arc::clone(&done);
                ar_pool::run_par(16, move |_| {
                    let d = Arc::clone(&d);
                    Box::new(move || {
                        d.fetch_add(1, Ordering::Relaxed);
                    })
                });
                assert_eq!(done.load(Ordering::Relaxed), 16);
            }));
        }
        for h in hs {
            h.join().unwrap();
        }
    }
}
