//! EXL3 GDN/어텐션 배선 (R4 — 순수 이동). GPU 체크는 gdn.rs/attn.rs(검증층),
//! 이 파일은 decode/prefill 오케스트라가 부르는 CPU-side 배선이다.
use super::cpu::SeqState;
use super::cpu::*;
use super::resident::TrellisResident;

pub(super) fn gdn_forward(
    tr: &mut TrellisResident,
    seq: &mut SeqState,
    il: usize,
    x_normed: &[f32],
) -> Result<Vec<f32>, String> {
    let lp = format!("model.language_model.layers.{il}.linear_attn");
    let d_state = 128usize;
    let n_k = 16usize; // K heads
    let n_v = 48usize; // V heads
    let d_inner = n_v * d_state; // 6144
    let k_len = n_k * d_state; // 2048
    let v_len = n_v * d_state; // 6144
    let conv_k = 4usize;
    let conv_ch = k_len * 2 + v_len; // 10240
    let eps = 1e-6f32;

    // 선형 투영 (vk GEMV) — alpha/beta도 선형이지만 크기가 작아 inline 계산
    // plans/120 A1: qkvz를 지연 패턴으로 제출(DBUF=1 시 비동기)하고
    // alpha/beta dot·ssm 수학은 qkvz 출력과 무독립이라 GPU 실행과 중첩한
    // 뒤 fetch로 합류 — 간극 −13ms 목표. DBUF 부재 시 의미 불변.
    let _gq = ph("gdn:lin_qkvz");
    let (pq, pz) = tr.linear_pair_deferred(
        &format!("{lp}.in_proj_qkv"),
        &format!("{lp}.in_proj_z"),
        x_normed,
    )?;
    // alpha: hidden → 48 (V헤드별 스케일러) — 노름에서 읽기
    let a_proj = tr
        .norm(&format!("{lp}.in_proj_a.weight"))
        .ok_or("alpha missing")?;
    let b_proj = tr
        .norm(&format!("{lp}.in_proj_b.weight"))
        .ok_or("beta missing")?;

    // a[b] = dot(x_normed, a_proj[b]) for each V head b
    // (풀 병렬화는 잡당 고정비 ~3µs에 묻혔 materially 무차 — 스칼라 유지,
    // plans/120 A1 실험 기록)
    let _ga = ph("gdn:ab_dot");
    let mut a_vals = vec![0f32; n_v];
    for (h, a_vals_h) in a_vals.iter_mut().enumerate() {
        let row = &a_proj[h * x_normed.len()..(h + 1) * x_normed.len()];
        *a_vals_h = x_normed.iter().zip(row.iter()).map(|(&x, &w)| x * w).sum();
    }
    let mut b_vals = vec![0f32; n_v];
    for (h, b_vals_h) in b_vals.iter_mut().enumerate() {
        let row = &b_proj[h * x_normed.len()..(h + 1) * x_normed.len()];
        *b_vals_h = x_normed.iter().zip(row.iter()).map(|(&x, &w)| x * w).sum();
    }
    drop(_ga);

    // ssm_a = -exp(A_log) — HF 순서 그대로 (직접 경로는 전체 HF 일관).
    // (GGUF 변환 시에만 V헤드 순열 필요 — §7.1b)
    let a_log = tr.norm(&format!("{lp}.A_log")).ok_or("A_log missing")?;
    let ssm_a: Vec<f32> = (0..n_v)
        .map(|i| -a_log[i.min(a_log.len() - 1)].exp())
        .collect();
    let dt_bias = tr
        .norm(&format!("{lp}.dt_bias"))
        .ok_or("dt_bias missing")?
        .to_vec(); // fetch(가변) 이후에도 쓴다 — 소유 복사(값 불변)

    // 지연 판독 합류 — 이 시점까지 GPU는 qkvz를 실행했다(DBUF 시).
    let qkv = tr.fetch(pq)?;
    let z = tr.fetch(pz)?;
    drop(_gq);
    if il == 0 && seq.pos == 0 {
        let rms = (qkv.iter().map(|v| v * v).sum::<f32>() / qkv.len() as f32).sqrt();
        eprintln!(
            "  [dbg] L0 qkv rms={rms:.4} qkv[0]={:.6} qkv[1]={:.6}",
            qkv[0],
            qkv.get(1).copied().unwrap_or(0.0)
        );
    }

    // beta, g
    let _gm = ph("gdn:ab_math");
    let beta_all: Vec<f32> = (0..n_v).map(|h| sigmoid(b_vals[h])).collect();
    let g_all: Vec<f32> = (0..n_v)
        .map(|h| softplus(a_vals[h] + dt_bias[h]) * ssm_a[h])
        .collect();
    drop(_gm);

    // conv1d (t=1)
    let conv_w = tr
        .norm(&format!("{lp}.conv1d.weight"))
        .ok_or("conv1d missing")?;
    let st = &mut seq.gdn[il];
    let _gc = ph("gdn:conv");
    let mut q_all = vec![0f32; k_len];
    let mut k_all = vec![0f32; k_len];
    let mut v_all = vec![0f32; v_len];
    for c in 0..conv_ch {
        let mut sum = conv_w[c * conv_k + (conv_k - 1)] * qkv[c];
        for j in 0..conv_k - 1 {
            sum += conv_w[c * conv_k + j] * st.conv[j * conv_ch + c];
        }
        let out_c = silu(sum);
        for j in 0..conv_k - 2 {
            st.conv[j * conv_ch + c] = st.conv[(j + 1) * conv_ch + c];
        }
        st.conv[(conv_k - 2) * conv_ch + c] = qkv[c];
        if c < k_len {
            q_all[c] = out_c;
        } else if c < 2 * k_len {
            k_all[c - k_len] = out_c;
        } else {
            v_all[c - 2 * k_len] = out_c;
        }
    }

    // L2 norm on q, k per K head
    for h in 0..n_k {
        let b0 = h * d_state;
        let head: Vec<f32> = q_all[b0..b0 + d_state].to_vec();
        q_all[b0..b0 + d_state].copy_from_slice(&l2_norm(&head, eps));
        let headk: Vec<f32> = k_all[b0..b0 + d_state].to_vec();
        k_all[b0..b0 + d_state].copy_from_slice(&l2_norm(&headk, eps));
    }
    drop(_gc);

    // GDN delta rule — core::gdn::gdn_ar_batch 재사용 (헤드 병렬).
    // HF 순서 → llama.cpp 순서 (h%h_k 매핑용) → 역순열로 복귀.
    // plans/120 A1: 상태는 llama.cpp 순서로 영구 저장 — 층당 3MB×2 순열
    // 복사(토큰당 288MB) 제거. 순열은 v/beta/g/o(6144f32)에만 적용.
    let hf_to_lc = |h: usize| -> usize { 3 * (h % 16) + h / 16 };

    let _gd = ph("gdn:delta");
    let mut v_lc = vec![0f32; v_len]; // llama.cpp 순서 v
    let mut beta_lc = vec![0f32; n_v];
    let mut g_lc = vec![0f32; n_v];
    let mut o_lc = vec![0f32; v_len];
    for i in 0..n_v {
        let j = hf_to_lc(i);
        v_lc[i * d_state..(i + 1) * d_state]
            .copy_from_slice(&v_all[j * d_state..(j + 1) * d_state]);
        beta_lc[i] = beta_all[j];
        g_lc[i] = g_all[j];
    }

    llm170_core::gdn::gdn_ar_batch(
        &q_all,
        &k_all,
        &v_lc,
        &beta_lc,
        &g_lc,
        &mut st.states,
        &mut o_lc,
        1,
        n_k,
        n_v,
    );

    // 결과 역순열 (llama.cpp → HF)
    let mut o_all = vec![0f32; v_len];
    for i in 0..n_v {
        let j = hf_to_lc(i);
        o_all[j * d_state..(j + 1) * d_state]
            .copy_from_slice(&o_lc[i * d_state..(i + 1) * d_state]);
    }
    drop(_gd);

    // norm_gated: rms_norm(o) * silu(z) per V head
    let ssm_norm_w = tr
        .norm(&format!("{lp}.norm.weight"))
        .ok_or("ssm_norm missing")?;
    let mut gated = vec![0f32; d_inner];
    let _gg = ph("gdn:gate");
    for h in 0..n_v {
        let b0 = h * d_state;
        let head: Vec<f32> = o_all[b0..b0 + d_state].to_vec();
        let n = rms_norm(&head, ssm_norm_w, eps);
        for i in 0..d_state {
            gated[b0 + i] = n[i] * silu(z[b0 + i]);
        }
    }
    drop(_gg);

    // out_proj (vk GEMV)
    let _go = ph("gdn:lin_out");
    let r = tr.linear(&format!("{lp}.out_proj"), &gated)?;
    drop(_go);
    Ok(r)
}

// ── Full attention (t=1) ──

/// T-행 full-attention 층: 배치 선형(q/k/v) + 행별 norm·rope·KV 적립 +
/// 인과 어텐션(행 병렬) + 게이트 + o_proj 배치.
pub(super) fn attn_forward(
    tr: &mut TrellisResident,
    seq: &mut SeqState,
    il: usize,
    attn_il: usize, // attention 층 인덱스 (kv 캐시 배열용)
    x_normed: &[f32],
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
) -> Result<Vec<f32>, String> {
    let lp = format!("model.language_model.layers.{il}.self_attn");
    let n_rot = 64; // partial_rotary_factor 0.25 * 256
    let rope_base = 1e7f32;

    // q/k/v (vk GEMV) — q는 gate 퓨전 [n_head * head_dim * 2]
    // plans/120 A1: 3회 개별 배치 → linear_triple 1배치 (동기 3→1).
    let _g1 = ph("attn:lin_qkv");
    let (q_gate, k, v) = tr.linear_triple(
        &format!("{lp}.q_proj"),
        &format!("{lp}.k_proj"),
        &format!("{lp}.v_proj"),
        x_normed,
    )?;
    drop(_g1);

    // q_norm, k_norm
    let q_norm_w = tr
        .norm(&format!("{lp}.q_norm.weight"))
        .ok_or("q_norm missing")?;
    let k_norm_w = tr
        .norm(&format!("{lp}.k_norm.weight"))
        .ok_or("k_norm missing")?;

    // q 디인터리브 (q‖gate) + norm + rope
    let pos = seq.pos;
    let kv = &mut seq.kv[attn_il];
    let kv_cap = kv.k.len() / (n_kv * head_dim);
    let _g2 = ph("attn:qknorm_rope");

    // q: 헤드별 [q(256), gate(256)] 인터리브 → 분리
    let mut q_heads = vec![0f32; n_head * head_dim];
    let mut gate_heads = vec![0f32; n_head * head_dim];
    for h in 0..n_head {
        let src = h * head_dim * 2;
        q_heads[h * head_dim..(h + 1) * head_dim].copy_from_slice(&q_gate[src..src + head_dim]);
        gate_heads[h * head_dim..(h + 1) * head_dim]
            .copy_from_slice(&q_gate[src + head_dim..src + head_dim * 2]);
    }

    // per-head q norm + rope
    for h in 0..n_head {
        let b0 = h * head_dim;
        let head: Vec<f32> = q_heads[b0..b0 + head_dim].to_vec();
        let n = rms_norm(&head, q_norm_w, 1e-6);
        q_heads[b0..b0 + head_dim].copy_from_slice(&n);
        // rope (in-place)
        let mut h_rot = q_heads[b0..b0 + head_dim].to_vec();
        rope(&mut h_rot, pos, n_rot, rope_base);
        q_heads[b0..b0 + head_dim].copy_from_slice(&h_rot);
    }

    // k norm + rope → KV 캐시 추가
    // A15(plans/129): KV 만석은 Err — 종전 디코드 무침입 스킵이 프리필의 Err과
    // 비대칭으로 조용한 품질 오염(마지막 토큰만 넣고 계속 진행)이었다.
    if kv.len >= kv_cap {
        return Err(format!(
            "exl3 KV full: pos {} >= cap {kv_cap} — ctx 상향 필요",
            kv.len
        ));
    }
    {
        let k_base = kv.len * n_kv * head_dim;
        for h in 0..n_kv {
            let b0 = h * head_dim;
            let head: Vec<f32> = k[b0..b0 + head_dim].to_vec();
            let n = rms_norm(&head, k_norm_w, 1e-6);
            let mut h_rot = n;
            rope(&mut h_rot, pos, n_rot, rope_base);
            kv.k[k_base + b0..k_base + b0 + head_dim].copy_from_slice(&h_rot);
            kv.v[k_base + b0..k_base + b0 + head_dim].copy_from_slice(&v[b0..b0 + head_dim]);
        }
        kv.len += 1;
    }
    drop(_g2);

    // attention: GQA — 각 q 헤드가 n_head/n_kv개의 kv 헤드를 공유
    // plans/120 A1: 헤드별 풀 병렬 + 가중치 사전계산(비트동일 — w[t]는
    // 원문과 동일 표현식으로 1회 계산, t 누적 순서 보존).
    let _g3 = ph("attn:core");
    let scale = 1.0 / (head_dim as f32).sqrt();
    let n_rep = n_head / n_kv;
    let mut attn_out = vec![0f32; n_head * head_dim];
    {
        let kv_len = kv.len;
        // SAFETY: 잡은 run_par 완료 대기 내에서만 접근 — q_heads·kv·attn_out은
        // 이 스코프 내 유효, 헤드별 출력 영역은 서로 분리된다.
        let (qp, kp, vp, op) = (
            SendP(q_heads.as_ptr() as usize),
            SendP(kv.k.as_ptr() as usize),
            SendP(kv.v.as_ptr() as usize),
            SendP(attn_out.as_mut_ptr() as usize),
        );
        llm170_core::gdn::ar_pool::run_par(n_head, move |h| {
            Box::new(move || unsafe {
                let kv_h = h / n_rep;
                let q =
                    std::slice::from_raw_parts((qp.0 as *const f32).add(h * head_dim), head_dim);
                let (kb, vb) = (kp.0 as *const f32, vp.0 as *const f32);
                let ob = (op.0 as *mut f32).add(h * head_dim);

                // 점수 계산
                let mut scores = vec![0f32; kv_len];
                for t in 0..kv_len {
                    let k_base = t * n_kv * head_dim + kv_h * head_dim;
                    let mut dot = 0f32;
                    for d in 0..head_dim {
                        dot += q[d] * *kb.add(k_base + d);
                    }
                    scores[t] = dot * scale;
                }

                // softmax
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scores.iter().map(|&s| (s - max_s).exp()).sum();
                let inv_sum = 1.0 / exp_sum;
                let w: Vec<f32> = scores
                    .iter()
                    .map(|&s| (s - max_s).exp() * inv_sum)
                    .collect();

                // 가중 합
                for d in 0..head_dim {
                    let mut sum = 0f32;
                    for t in 0..kv_len {
                        let v_base = t * n_kv * head_dim + kv_h * head_dim;
                        sum += w[t] * *vb.add(v_base + d);
                    }
                    *ob.add(d) = sum;
                }
            })
        });
    }

    // output gate: attn_out * sigmoid(gate)
    for h in 0..n_head {
        let b0 = h * head_dim;
        for d in 0..head_dim {
            attn_out[b0 + d] *= sigmoid(gate_heads[b0 + d]);
        }
    }
    drop(_g3);

    // o_proj (vk GEMV)
    let _g4 = ph("attn:lin_o");
    let r = tr.linear(&format!("{lp}.o_proj"), &attn_out)?;
    drop(_g4);
    Ok(r)
}

pub(super) fn attn_batch(
    tr: &mut TrellisResident,
    seq: &mut SeqState,
    il: usize,
    attn_il: usize,
    t_rows: usize,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
) -> Result<Vec<f32>, String> {
    let lp = format!("model.language_model.layers.{il}.self_attn");
    let n_rot = 64usize;
    let rope_base = 1e7f32;
    let eps = 1e-6f32;

    // GPU 경로(plans/121 F2b): qkv GEMM → prep(norm+rope+KV 적립) → fwd3
    // (인과 어텐션+게이트→xtb) → o_proj 직결. KV CPU 사본은 프리필 말미 벌크.
    if t_rows > 8 {
        let slots = tr.linear_batch_multi_gpu(
            &[
                &format!("{lp}.q_proj"),
                &format!("{lp}.k_proj"),
                &format!("{lp}.v_proj"),
            ],
            t_rows,
        )?;
        let pos0 = seq.pos;
        tr.attn_layer_gpu(attn_il, t_rows, pos0, slots[0].0, slots[1].0, slots[2].0)?;
        seq.kv[attn_il].len += t_rows;
        let out = tr.linear_out_gpu(&format!("{lp}.o_proj"), t_rows)?;
        return Ok(out);
    }

    // 계약: xn은 호출자가 stage_f32 버퍼에 [T][h]로 미리 스테이징했다.
    let _a0 = ph("ppa:lin_qkv");
    let mut outs = tr.linear_batch_multi_staged(
        &[
            &format!("{lp}.q_proj"),
            &format!("{lp}.k_proj"),
            &format!("{lp}.v_proj"),
        ],
        t_rows,
    )?;
    let v = outs.pop().ok_or("qkv 결과 유실")?;
    let k = outs.pop().ok_or("qkv 결과 유실")?;
    let q_gate = outs.pop().ok_or("qkv 결과 유실")?; // [T][n_head*head_dim*2]
    drop(_a0);

    let q_norm_w = tr
        .norm(&format!("{lp}.q_norm.weight"))
        .ok_or("q_norm missing")?
        .to_vec();
    let k_norm_w = tr
        .norm(&format!("{lp}.k_norm.weight"))
        .ok_or("k_norm missing")?
        .to_vec();

    // 행별: q 디인터리브 + norm + rope, k norm+rope → KV 적립(순서 보장).
    let _a1 = ph("ppa:kv");
    let pos0 = seq.pos;
    let mut q_heads = vec![0f32; t_rows * n_head * head_dim];
    let mut gate_heads = vec![0f32; t_rows * n_head * head_dim];
    {
        let kv = &mut seq.kv[attn_il];
        let kv_cap = kv.k.len() / (n_kv * head_dim);
        if kv.len + t_rows > kv_cap {
            return Err(format!(
                "attn_batch: kv 용량 초과 ({}+{} > {kv_cap})",
                kv.len, t_rows
            ));
        }
        for t in 0..t_rows {
            // q: 헤드별 [q(256), gate(256)] 인터리브 → 분리
            for hh in 0..n_head {
                let src = t * n_head * head_dim * 2 + hh * head_dim * 2;
                q_heads[t * n_head * head_dim + hh * head_dim
                    ..t * n_head * head_dim + (hh + 1) * head_dim]
                    .copy_from_slice(&q_gate[src..src + head_dim]);
                gate_heads[t * n_head * head_dim + hh * head_dim
                    ..t * n_head * head_dim + (hh + 1) * head_dim]
                    .copy_from_slice(&q_gate[src + head_dim..src + head_dim * 2]);
            }
            // per-head q norm + rope
            for hh in 0..n_head {
                let b0 = t * n_head * head_dim + hh * head_dim;
                let head: Vec<f32> = q_heads[b0..b0 + head_dim].to_vec();
                let n = rms_norm(&head, &q_norm_w, eps);
                let mut h_rot = n;
                rope(&mut h_rot, pos0 + t as u32, n_rot, rope_base);
                q_heads[b0..b0 + head_dim].copy_from_slice(&h_rot);
            }
            // k norm + rope → 캐시 적립, v 적립
            let k_base = kv.len * n_kv * head_dim;
            for hh in 0..n_kv {
                let b0 = t * n_kv * head_dim + hh * head_dim;
                let head: Vec<f32> = k[b0..b0 + head_dim].to_vec();
                let n = rms_norm(&head, &k_norm_w, eps);
                let mut h_rot = n;
                rope(&mut h_rot, pos0 + t as u32, n_rot, rope_base);
                kv.k[k_base + hh * head_dim..k_base + (hh + 1) * head_dim].copy_from_slice(&h_rot);
                kv.v[k_base + hh * head_dim..k_base + (hh + 1) * head_dim]
                    .copy_from_slice(&v[b0..b0 + head_dim]);
            }
            kv.len += 1;
        }
    }

    drop(_a1);
    // 인과 어텐션 — 행 병렬(행 t는 kv[0..=pos0+t] 만 본다).
    let _a2 = ph("ppa:core");
    let scale = 1.0 / (head_dim as f32).sqrt();
    let n_rep = n_head / n_kv;
    // 어텐션 출력은 스테이징 버퍼에 직접 기록(o_proj가 곧 소비 — 원장 #3).
    let stage = tr.stage_f32()?;
    {
        let kv = &seq.kv[attn_il];
        let (qp, kp, vp, gp, op) = (
            PP(q_heads.as_ptr() as usize),
            PP(kv.k.as_ptr() as usize),
            PP(kv.v.as_ptr() as usize),
            PP(gate_heads.as_ptr() as usize),
            PP(stage as usize),
        );
        // SAFETY: 행별 출력 영역 분리, kv/q/gate는 읽기 공유(scope join 증명).
        par_rows(t_rows, move |t| unsafe {
            let kv_len = (pos0 as usize) + t + 1;
            // 행당 버퍼 1회 — 이전 (t,h)잙당 scores/w Vec 2개 할당 제거.
            let mut scores = vec![0f32; kv_len];
            let mut acc = vec![0f32; head_dim];
            for hh in 0..n_head {
                let kv_h = hh / n_rep;
                let q = std::slice::from_raw_parts(
                    (qp.0 as *const f32).add(t * n_head * head_dim + hh * head_dim),
                    head_dim,
                );
                let (kb, vb) = (kp.0 as *const f32, vp.0 as *const f32);
                let ob = (op.0 as *mut f32).add(t * n_head * head_dim + hh * head_dim);
                for tt in 0..kv_len {
                    let k_base = tt * n_kv * head_dim + kv_h * head_dim;
                    let mut dot = 0f32;
                    for d in 0..head_dim {
                        dot += q[d] * *kb.add(k_base + d);
                    }
                    scores[tt] = dot * scale;
                }
                let max_s = scores[..kv_len]
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                // t-우선 가중합 — v 행 연속 판독(이전 d-우선은 열 스트라이드
                // 4KB 산재 판독으로 ppa:core 34% 병목의 주원인, 2026-10-03
                // 원장 #1). 정규화는 누적 후 1회(환원 순서 변경 — 10a).
                let mut wsum = 0f32;
                for d in 0..head_dim {
                    acc[d] = 0.0;
                }
                for tt in 0..kv_len {
                    let v_base = tt * n_kv * head_dim + kv_h * head_dim;
                    let wtt = (scores[tt] - max_s).exp();
                    wsum += wtt;
                    for d in 0..head_dim {
                        acc[d] += wtt * *vb.add(v_base + d);
                    }
                }
                let inv = 1.0 / wsum;
                // 게이트: attn_out · sigmoid(gate)
                for d in 0..head_dim {
                    let gv = *(gp.0 as *const f32).add(t * n_head * head_dim + hh * head_dim + d);
                    *ob.add(d) = acc[d] * inv * sigmoid(gv);
                }
            }
        });
    }

    drop(_a2);
    // o_proj 배치(스테이징 — 어텐션 출력을 stage에 직접 기록했다)
    let _a3 = ph("ppa:out");
    let o = tr.linear_batch_staged(&format!("{lp}.o_proj"), t_rows)?;
    drop(_a3);
    Ok(o)
}

/// T-행 GDN 층: 배치 선형(qkv+z) + conv1d(T) + 청크 스캔 + norm_gated +
/// out_proj 배치. 상태(gdn states·conv 링)는 순차 경로와 동일 형식으로
/// 갱신 — 이후 decode_step 연속 가능.
pub(super) fn gdn_batch(
    tr: &mut TrellisResident,
    seq: &mut SeqState,
    il: usize,
    xn: &[f32],
    t_rows: usize,
) -> Result<Vec<f32>, String> {
    let h = tr.hidden;
    let lp = format!("model.language_model.layers.{il}.linear_attn");
    let d_state = 128usize;
    let n_k = 16usize;
    let n_v = 48usize;
    let d_inner = n_v * d_state; // 6144
    let k_len = n_k * d_state; // 2048
    let conv_k = 4usize;
    let conv_ch = k_len * 2 + d_inner; // 10240
    let eps = 1e-6f32;

    // ── F1 GPU 경로(plans/121): T>8에서 전 비선형 GPU 상주 ──
    // qkv+z를 yb에 남기고 → conv→l2perm→scan→gate 4커널 → gated가 xtb에.
    if t_rows > 8 {
        tr.gdn_frame_init()?;
        let gdn_il = (0..il).filter(|i| i % 4 != 3).count();
        // GPU 상태/ring 버퍼는 alloc_host_cached로 제로 보장 없음 — CPU 상태를
        // 업로드(초기 전부 0, 이후 gdn_state_sync가 갱신된 값 유지).
        {
            let _gu = ph("ppg:upload");
            let g = &seq.gdn[il];
            tr.gdn_state_upload(gdn_il, &g.states, &g.conv)?;
        }
        let _gf = ph("ppg:gpu_layer");
        let slots = {
            let _gl = ph("ppg:lin_qkvz");
            tr.linear_batch_multi_gpu(
                &[&format!("{lp}.in_proj_qkv"), &format!("{lp}.in_proj_z")],
                t_rows,
            )?
        };
        let (yb0, _n0) = slots[0];
        let (yb1, _n1) = slots[1];
        {
            let _gk = ph("ppg:kern");
            tr.gdn_layer_gpu(gdn_il, t_rows, std::ptr::null_mut(), yb0, yb1)?;
        }
        drop(_gf);
        // out_proj GPU 직결: gate가 xtb에 기록한 출력을 왕복 없이 소비
        // (plans/121 F2 스케줄 — invalidate_xtab/flush/재업로드 제거).
        let out = tr.linear_out_gpu(&format!("{lp}.out_proj"), t_rows)?;
        return Ok(out);
    }

    // ── CPU 경로(기존) — T≤8 스펙 라운드·소형 배치용 ──
    let _g0 = ph("ppg:lin_qkvz");
    let mut outs = tr.linear_batch_multi_staged(
        &[&format!("{lp}.in_proj_qkv"), &format!("{lp}.in_proj_z")],
        t_rows,
    )?;
    drop(_g0);
    let z = outs.pop().ok_or("qkv/z 결과 유실")?; // [T][6144]
    let qkv = outs.pop().ok_or("qkv/z 결과 유실")?; // [T][10240]

    // 무양자화 가중치 — 선형 호출 전 소유 복사(값 불변, borrow 분리)
    let a_proj = tr
        .norm(&format!("{lp}.in_proj_a.weight"))
        .ok_or("alpha missing")?
        .to_vec();
    let b_proj = tr
        .norm(&format!("{lp}.in_proj_b.weight"))
        .ok_or("beta missing")?
        .to_vec();
    let a_log = tr
        .norm(&format!("{lp}.A_log"))
        .ok_or("A_log missing")?
        .to_vec();
    let dt_bias = tr
        .norm(&format!("{lp}.dt_bias"))
        .ok_or("dt_bias missing")?
        .to_vec();
    let conv_w = tr
        .norm(&format!("{lp}.conv1d.weight"))
        .ok_or("conv1d missing")?
        .to_vec();
    let ssm_norm_w = tr
        .norm(&format!("{lp}.norm.weight"))
        .ok_or("ssm_norm missing")?
        .to_vec();

    // ssm_a = -exp(A_log) — 행 준비 루프에서 인라인 계산(환원 불변)

    // alpha/beta dot — 행 병렬(스칼라 dot는 120 무차 확정, 행 단위 병렬만).
    let _ga = ph("ppg:ab");
    let mut a_vals = vec![0f32; t_rows * n_v];
    let mut b_vals = vec![0f32; t_rows * n_v];
    {
        let (xp, ap, bp) = (
            PP(xn.as_ptr() as usize),
            PP(a_proj.as_ptr() as usize),
            PP(b_proj.as_ptr() as usize),
        );
        let (av, bv) = (
            PP(a_vals.as_mut_ptr() as usize),
            PP(b_vals.as_mut_ptr() as usize),
        );
        // SAFETY: 쓰기 영역은 행별 분리(t*n_v+hv), 읽기는 공유 불변.
        par_rows(t_rows, move |t| unsafe {
            let xr = std::slice::from_raw_parts((xp.0 as *const f32).add(t * h), h);
            for hv in 0..n_v {
                let ar = std::slice::from_raw_parts((ap.0 as *const f32).add(hv * h), h);
                *(av.0 as *mut f32).add(t * n_v + hv) =
                    xr.iter().zip(ar.iter()).map(|(&x, &w)| x * w).sum();
                let br = std::slice::from_raw_parts((bp.0 as *const f32).add(hv * h), h);
                *(bv.0 as *mut f32).add(t * n_v + hv) =
                    xr.iter().zip(br.iter()).map(|(&x, &w)| x * w).sum();
            }
        });
    }

    drop(_ga);
    // conv1d(T) — 채널별 인과 콘볼루션 + 링 갱신. 채널 병렬(쓰기 분리).
    let _gv = ph("ppg:conv");
    let mut q_all = vec![0f32; t_rows * k_len];
    let mut k_all = vec![0f32; t_rows * k_len];
    let mut v_all = vec![0f32; t_rows * d_inner];
    // conv 직후 원점 덤프(디버그) — GPU gqr와 비교용.
    let mut q_raw_dbg: Vec<f32> = Vec::new();
    {
        let st = &mut seq.gdn[il];
        let (qp, kp, vp) = (
            PP(q_all.as_mut_ptr() as usize),
            PP(k_all.as_mut_ptr() as usize),
            PP(v_all.as_mut_ptr() as usize),
        );
        let (ringp, qkvp, cwp) = (
            PP(st.conv.as_mut_ptr() as usize),
            PP(qkv.as_ptr() as usize),
            PP(conv_w.as_ptr() as usize),
        );
        // SAFETY: 채널 c의 출력/링 쓰기는 채널별 분리, qkv/conv_w는 읽기 공유.
        par_rows(conv_ch, move |c| unsafe {
            let w0 = *(cwp.0 as *const f32).add(c * conv_k);
            let w1 = *(cwp.0 as *const f32).add(c * conv_k + 1);
            let w2 = *(cwp.0 as *const f32).add(c * conv_k + 2);
            let w3 = *(cwp.0 as *const f32).add(c * conv_k + 3);
            let mut h0 = *(ringp.0 as *mut f32).add(c);
            let mut h1 = *(ringp.0 as *mut f32).add(conv_ch + c);
            let mut h2 = *(ringp.0 as *mut f32).add(2 * conv_ch + c);
            for t in 0..t_rows {
                let xt = *(qkvp.0 as *const f32).add(t * conv_ch + c);
                let o = silu(w3 * xt + w0 * h0 + w1 * h1 + w2 * h2);
                if c < k_len {
                    *(qp.0 as *mut f32).add(t * k_len + c) = o;
                } else if c < 2 * k_len {
                    *(kp.0 as *mut f32).add(t * k_len + (c - k_len)) = o;
                } else {
                    *(vp.0 as *mut f32).add(t * d_inner + (c - 2 * k_len)) = o;
                }
                h0 = h1;
                h1 = h2;
                h2 = xt;
            }
            *(ringp.0 as *mut f32).add(c) = h0;
            *(ringp.0 as *mut f32).add(conv_ch + c) = h1;
            *(ringp.0 as *mut f32).add(2 * conv_ch + c) = h2;
        });
    }

    drop(_gv);
    // conv 직후 원점 캡처(L2 전) — GPU gqr와 비교.
    if il == 0 {
        q_raw_dbg = q_all[..5.min(q_all.len())].to_vec();
    }

    // 행 준비(병렬): L2 q/k + beta/g + lc 순열(v/beta/g).
    let _gp1 = ph("ppg:prep");
    let hf_to_lc = |hh: usize| -> usize { 3 * (hh % 16) + hh / 16 };
    let mut beta_lc = vec![0f32; t_rows * n_v];
    let mut g_lc = vec![0f32; t_rows * n_v];
    let mut v_lc = vec![0f32; t_rows * d_inner];
    {
        let (qp, kp, blp, glp, av, bv, alp, dbp) = (
            PP(q_all.as_mut_ptr() as usize),
            PP(k_all.as_mut_ptr() as usize),
            PP(beta_lc.as_mut_ptr() as usize),
            PP(g_lc.as_mut_ptr() as usize),
            PP(a_vals.as_ptr() as usize),
            PP(b_vals.as_ptr() as usize),
            PP(a_log.as_ptr() as usize),
            PP(dt_bias.as_ptr() as usize),
        );
        // SAFETY: 행별 분리 쓰기(q_all/k_all/beta_lc/g_lc), 읽기 공유.
        par_rows(t_rows, move |t| unsafe {
            // L2 norm q/k per K head
            for hh in 0..n_k {
                let b0 = t * k_len + hh * d_state;
                let head =
                    std::slice::from_raw_parts((qp.0 as *const f32).add(b0), d_state).to_vec();
                let n = l2_norm(&head, eps);
                std::ptr::copy_nonoverlapping(n.as_ptr(), (qp.0 as *mut f32).add(b0), d_state);
                let headk =
                    std::slice::from_raw_parts((kp.0 as *const f32).add(b0), d_state).to_vec();
                let nk = l2_norm(&headk, eps);
                std::ptr::copy_nonoverlapping(nk.as_ptr(), (kp.0 as *mut f32).add(b0), d_state);
            }
            // beta/g — lc 헤드 i는 HF 헤드 j=hf_to_lc(i)의 값(decode의
            // beta_lc[i]=beta_all[j] 미러). a_log/dt_bias도 HF j로 인덱스.
            for i in 0..n_v {
                let j = hf_to_lc(i);
                let a_v = *(av.0 as *const f32).add(t * n_v + j);
                let b_v = *(bv.0 as *const f32).add(t * n_v + j);
                let al = *(alp.0 as *const f32).add(j.min(n_v - 1));
                let dtb = *(dbp.0 as *const f32).add(j);
                *(blp.0 as *mut f32).add(t * n_v + i) = sigmoid(b_v);
                *(glp.0 as *mut f32).add(t * n_v + i) = softplus(a_v + dtb) * -al.exp();
            }
        });
    }
    drop(_gp1);
    // L2+beta/g 완료 후 디버그 — GPU gq/gbg와 비교.
    if il == 0 {
        eprintln!(
            "  [f1dbg] L0 CPU L2q: {:?} beta: {:?} g: {:?}",
            &q_all[..5],
            &beta_lc[..5],
            &g_lc[..5]
        );
    }

    // v 순열 복사(행 병렬 — v_all(HF) → v_lc(llama.cpp 헤드 순서))
    let _gp2 = ph("ppg:vperm");
    {
        let (vap, vlp) = (PP(v_all.as_ptr() as usize), PP(v_lc.as_mut_ptr() as usize));
        // SAFETY: 행·헤드별 분리 쓰기.
        par_rows(t_rows, move |t| unsafe {
            for i in 0..n_v {
                let j = hf_to_lc(i);
                std::ptr::copy_nonoverlapping(
                    (vap.0 as *const f32).add(t * d_inner + j * d_state),
                    (vlp.0 as *mut f32).add(t * d_inner + i * d_state),
                    d_state,
                );
            }
        });
    }

    drop(_gp2);

    // 청크 스캔 — core::gdn 재사용(CS=64, AR 등가 검증). 소형 T(≤8)는
    // gdn_chunk_seq의 층당 스레드 스폰(48헤드×스코프)이 3.35ms/층의
    // 주벽이었다(2026-10-03 스펙 라운드 진단) — ar_pool 순차 AR로 대체:
    // 디코드 경로와 동일 의미론(비트 일치) + 스폰 비용 0.
    let mut o_lc = vec![0f32; t_rows * d_inner];
    let _gc2 = ph("ppg:chunk");
    if t_rows <= 8 {
        let k_stride = n_k * d_state;
        let v_stride = n_v * d_state;
        for t in 0..t_rows {
            // SAFETY: par 잡은 run_par 완료 대기 내 유효 — 행별 분리 입력.
            let (q1, k1, v1) = unsafe {
                (
                    std::slice::from_raw_parts(q_all.as_ptr().add(t * k_stride), k_stride),
                    std::slice::from_raw_parts(k_all.as_ptr().add(t * k_stride), k_stride),
                    std::slice::from_raw_parts(v_lc.as_ptr().add(t * v_stride), v_stride),
                )
            };
            let (b1, g1) = unsafe {
                (
                    std::slice::from_raw_parts(beta_lc.as_ptr().add(t * n_v), n_v),
                    std::slice::from_raw_parts(g_lc.as_ptr().add(t * n_v), n_v),
                )
            };
            let o1 = &mut o_lc[t * v_stride..(t + 1) * v_stride];
            llm170_core::gdn::gdn_ar_batch(
                q1,
                k1,
                v1,
                b1,
                g1,
                &mut seq.gdn[il].states,
                o1,
                1,
                n_k,
                n_v,
            );
        }
    } else {
        // plans/122 II-1 GPU 청크 v1 — 측정 부정(2026-10-03: pp512 62.2 vs CPU 66.7,
        // 전송 ~2GB/런 + 48WG 저점유) — 원복. 정합 자체는 corr 1.0000/o_maxd 9e-4
        // 달성(교훈: staged 호출의 wait_pending 누락이 스테일 판독 범인이었다).
        // 재시도 설계: 2단계 — d 삼각해는 상태 무독립(헤드×청크 전체 병렬) +
        // 상태 캐리/o만 순차 패스.
        llm170_core::gdn::gdn_chunk_seq(
            &q_all,
            &k_all,
            &v_lc,
            &beta_lc,
            &g_lc,
            &mut seq.gdn[il].states,
            &mut o_lc,
            t_rows,
            n_k,
            n_v,
        );
    }

    // 역순열 + norm_gated(rms(o)·silu(z)) — 행 병렬, 스테이징 직접 기록
    // (out_proj 입력이 곧 소비되므로 중간 Vec 없이 stage에 쓴다 — 원장 #3).
    let _gg = ph("ppg:gate");
    let stage = tr.stage_f32()?;
    {
        let (op, zp, gp, nwp) = (
            PP(o_lc.as_ptr() as usize),
            PP(z.as_ptr() as usize),
            PP(stage as usize),
            PP(ssm_norm_w.as_ptr() as usize),
        );
        // SAFETY: 행·헤드별 분리 쓰기(gated).
        par_rows(t_rows, move |t| unsafe {
            // lc 헤드 i → HF 헤드 j=hf_to_lc(i): o는 lc 위치 i에서, z/gated는
            // HF 위치 j에서 (decode의 o_all[j]=o_lc[i]·z HF 미러 — hf_to_lc는
            // 비대합이므로 방향이 중요하다).
            for i in 0..n_v {
                let j = hf_to_lc(i);
                let src = t * d_inner + i * d_state;
                let dst = t * d_inner + j * d_state;
                let head =
                    std::slice::from_raw_parts((op.0 as *const f32).add(src), d_state).to_vec();
                let nw = std::slice::from_raw_parts(nwp.0 as *const f32, d_state);
                let n = rms_norm(&head, nw, eps);
                for d in 0..d_state {
                    let zv = *(zp.0 as *const f32).add(dst + d);
                    *(gp.0 as *mut f32).add(dst + d) = n[d] * silu(zv);
                }
            }
        });
    }

    drop(_gg);
    // out_proj 배치(스테이징)
    let _go = ph("ppg:out");
    let r = tr.linear_batch_staged(&format!("{lp}.out_proj"), t_rows)?;
    drop(_go);
    if il == 0 {
        eprintln!(
            "  [f1dbg] L0 out_proj CPU: {:?} conv_q: {:?}",
            &r[..5.min(r.len())],
            q_raw_dbg
        );
    }
    Ok(r)
}
