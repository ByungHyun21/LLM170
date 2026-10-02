//! EXL3 레이어 스트리밍 단일 토큰 디코드 (§3-2) — 전방향 실구현.
//! TrellisResident 위에 qwen35 전방향: 선형=vk GEMV, 비선형=CPU.
//!
//! 기존 core의 gdn_ar_batch·gdn_norm_gated·rope_head 함수를 재사용 —
//! 비선형 로직을 재발명하지 않고 검증된 구현을 호출한다.

use super::exl3_resident::TrellisResident;

// ── 위상 프로파일러 (LLM170_DUMP=exl3_phase, 원장 89: dump 키로만) ──
// profile_span은 release no-op이라 exl3 프로브 전용 경량 계측.
// 단일 스레드 오케스트레이션 전제(thread_local) — gdn_ar_batch 내부 스레드는 미계측.
use std::cell::RefCell;
use std::collections::HashMap;

struct PhaseAgg {
    count: u64,
    ns: u64,
}
thread_local! {
    static PHASES: RefCell<HashMap<&'static str, PhaseAgg>> = RefCell::new(HashMap::new());
}
fn phase_on() -> bool {
    llm170_diag::dump::opts().key("exl3_phase")
}

/// usize 포인터 래퍼 — core::gdn::ar_pool 잡 캡처용(Send).
/// SAFETY: 주소의 생명은 run_par 완료 대기로 증명(호출 스코프 내).
#[derive(Clone, Copy)]
struct SendP(usize);
struct PhaseGuard(std::time::Instant, &'static str);
impl Drop for PhaseGuard {
    fn drop(&mut self) {
        let ns = self.0.elapsed().as_nanos() as u64;
        PHASES.with(|m| {
            if let Ok(mut m) = m.try_borrow_mut() {
                let e = m.entry(self.1).or_insert(PhaseAgg { count: 0, ns: 0 });
                e.count += 1;
                e.ns += ns;
            }
        });
    }
}
fn ph(name: &'static str) -> Option<PhaseGuard> {
    phase_on().then(|| PhaseGuard(std::time::Instant::now(), name))
}
fn phase_report() {
    if !phase_on() {
        return;
    }
    PHASES.with(|m| {
        if let Ok(m) = m.try_borrow() {
            let mut v: Vec<_> = m.iter().collect();
            v.sort_by_key(|(_, a)| std::cmp::Reverse(a.ns));
            eprintln!("=== exl3 phase (wall — 중첩 포함: *_fwd 값은 하위 위상 합 포함) ===");
            for (k, a) in v {
                eprintln!(
                    "  {k:20} {:>7}회 {:>10.1}ms  평균 {:>8.3}ms",
                    a.count,
                    a.ns as f64 / 1e6,
                    a.ns as f64 / a.count as f64 / 1e6
                );
            }
        }
    });
}

/// GDN 시퀀스 상태 (per-layer per-head 128×128).
pub struct GdnState {
    /// [48 heads][128*128] — core::gdn 형식, **llama.cpp 헤드 순서**로 저장
    /// (hf_to_lc 순열을 초기화 시 고정 — plans/120 A1, 매 토큰 순열복사 제거).
    pub states: Vec<f32>,
    /// conv1d 링 [conv_k-1][conv_ch].
    pub conv: Vec<f32>,
}

/// KV 캐시 (full-attn 층용).
pub struct KvCache {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub len: usize,
}

/// 전체 시퀀스 상태.
pub struct SeqState {
    pub gdn: Vec<GdnState>,
    pub kv: Vec<KvCache>,
    pub pos: u32,
}

// ── 유틸리티 ──

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let ss: f32 = x.iter().map(|&v| v * v).sum();
    let inv = 1.0 / ((ss / n as f32 + eps).sqrt());
    x.iter().zip(w.iter()).map(|(&v, &g)| v * inv * g).collect()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn l2_norm(x: &[f32], eps: f32) -> Vec<f32> {
    let ss: f32 = x.iter().map(|&v| v * v).sum();
    let inv = 1.0 / ((ss + eps).sqrt());
    x.iter().map(|&v| v * inv).collect()
}

/// RoPE — core::ops::rope_head 위임.
fn rope(head: &mut [f32], pos: u32, n_rot: usize, base: f32) {
    // core의 rope_head는 half-split 방식 — 직접 호출.
    // SAFETY: head가 n_rot보다 크거나 같음을 보장.
    llm170_core::ops::rope_head(head, pos, n_rot, base);
}

// ── GDN 단일 토큰 (t=1) ──

fn gdn_forward(
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

fn attn_forward(
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
    if kv.len < kv_cap {
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

// ── 메인 디코드 ──

pub fn decode_step(
    tr: &mut TrellisResident,
    seq: &mut SeqState,
    token: u32,
) -> Result<Vec<f32>, String> {
    let h = tr.hidden;
    let eps = 1e-6f32;

    // 임베딩
    let mut x = tr.embed_row(token).to_vec();
    if seq.pos == 0 {
        let rms = (x.iter().map(|v| v * v).sum::<f32>() / h as f32).sqrt();
        eprintln!("  [dbg] embed rms={rms:.4}");
        // 첫 층 norm 가중치 확인
        if let Some(nw) = tr.norm("model.language_model.layers.0.input_layernorm.weight") {
            eprintln!("  [dbg] attn_norm[0..3] = {:?}", &nw[..3.min(nw.len())]);
        }
    }

    let mut attn_count = 0;
    for il in 0..tr.n_layers {
        let lp = format!("model.language_model.layers.{il}");
        let full = il % 4 == 3;

        // attn_norm
        let norm_w = tr
            .norm(&format!("{lp}.input_layernorm.weight"))
            .ok_or("norm missing")?;
        let xn = rms_norm(&x, norm_w, eps);

        let mut attn_out = if full {
            let _g = ph("attn_fwd");
            let out = attn_forward(tr, seq, il, attn_count, &xn, 24, 4, 256)?;
            drop(_g);
            attn_count += 1;
            out
        } else {
            let _g = ph("gdn_fwd");
            let r = gdn_forward(tr, seq, il, &xn)?;
            drop(_g);
            r
        };
        // 디버그: attention/GDN 출력 제거 — FFN만 남겨 격리.
        if std::env::var_os("LLM170_EXL3_DBG")
            .map(|v| v == "attn_skip")
            .unwrap_or(false)
        {
            for v in attn_out.iter_mut() {
                *v = 0.0;
            }
        }

        // 잔차
        for i in 0..h {
            x[i] += attn_out.get(i).copied().unwrap_or(0.0);
        }

        // FFN (post_attention_norm)
        let ffn_norm_w = tr
            .norm(&format!("{lp}.post_attention_layernorm.weight"))
            .ok_or("ffn norm missing")?;
        let xf = rms_norm(&x, ffn_norm_w, eps);
        // plans/120 A1: FFN 3선형 + GPU ew(silu·mul) 단일 배치 — 게이트/업
        // 판독·CPU 활성화·업로드 제거(간극 감소). ew GPU exp는 10a.
        let _gf = ph("ffn_all");
        let ffn_out = tr.ffn_triple(
            &format!("{lp}.mlp.gate_proj"),
            &format!("{lp}.mlp.up_proj"),
            &format!("{lp}.mlp.down_proj"),
            &xf,
        )?;
        drop(_gf);
        for i in 0..h {
            x[i] += ffn_out.get(i).copied().unwrap_or(0.0);
        }
    }

    // 활성화 크기 디버그 (첫 스텝만)
    if seq.pos == 0 {
        let rms = (x.iter().map(|v| v * v).sum::<f32>() / h as f32).sqrt();
        eprintln!(
            "  [dbg] 최종 x rms={rms:.4} max={:.4}",
            x.iter().cloned().fold(f32::MIN, f32::max)
        );
    }
    seq.pos += 1;

    // output_norm + lm_head
    let out_norm_w = tr
        .norm("model.language_model.norm.weight")
        .ok_or("output norm missing")?;
    let xn = rms_norm(&x, out_norm_w, eps);
    let _gh = ph("head");
    let r = tr.linear("lm_head", &xn)?;
    drop(_gh);
    Ok(r)
}

/// 시퀀스 상태 초기화.
pub fn new_seq_state(n_layers: usize, ctx_len: usize) -> SeqState {
    let n_v = 48;
    let d_state = 128;
    let conv_ch = 10240;
    let conv_k = 4;
    let n_full_attn = 16; // 64/4
    let _n_head = 24;

    SeqState {
        gdn: (0..n_layers)
            .map(|_| GdnState {
                states: vec![0f32; n_v * d_state * d_state],
                conv: vec![0f32; (conv_k - 1) * conv_ch],
            })
            .collect(),
        kv: (0..n_full_attn)
            .map(|_| KvCache {
                k: vec![0f32; ctx_len * 4 * 256],
                v: vec![0f32; ctx_len * 4 * 256],
                len: 0,
            })
            .collect(),
        pos: 0,
    }
}

/// `llm170 exl3-decode <dir> <token_ids_comma> [n_predict]`
pub fn exl3_decode(dir: &str, tokens_str: &str, n_predict: usize) -> Result<String, String> {
    let prompt: Vec<u32> = tokens_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if prompt.is_empty() {
        return Err("토큰 ID 필요 (쉼표 구분)".into());
    }
    let t0 = std::time::Instant::now();
    eprintln!("  [exl3-decode] 상주 적재 중...");
    let mut tr = TrellisResident::load(dir)?;
    eprintln!(
        "  [exl3-decode] 적재 완료 {}선형 {:.1}s — 디코드 시작",
        tr.linears.len(),
        t0.elapsed().as_secs_f64()
    );

    let mut seq = new_seq_state(tr.n_layers, 512);
    let t1 = std::time::Instant::now();
    let mut logits = Vec::new();
    // 프리필 (순차 디코드)
    for (i, &tok) in prompt.iter().enumerate() {
        logits = decode_step(&mut tr, &mut seq, tok)?;
        if i == 0 {
            let rms = (0..1).map(|_| 0f32).sum::<f32>(); // suppress warning
            let _ = rms;
        }
    }
    // 생성
    let mut out_tokens = Vec::new();
    for step in 0..n_predict {
        let (best, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((0, &0.0));
        out_tokens.push(best as u32);
        if step < 6 {
            eprintln!(
                "  [exl3-decode] step {step}: token={}",
                out_tokens.last().copied().unwrap_or(0)
            );
        }
        if step + 1 < n_predict {
            logits = decode_step(&mut tr, &mut seq, out_tokens[out_tokens.len() - 1])?;
        }
    }
    let decode_s = t1.elapsed().as_secs_f64();
    // 진단 덤프 — VK_TS: GPU 디스패치 집계(프레임 경로와 동일 ts 원장),
    // exl3_phase: CPU 위상 분해. decode_s 측정 후 호출(쿼리 WAIT 제외).
    if llm170_diag::flag::on("LLM170_VK_TS") {
        tr.ctx.ts_report();
    }
    phase_report();
    let total = prompt.len() + n_predict;
    let tps = total as f64 / decode_s;
    Ok(format!(
        "exl3-decode: prompt {} + gen {} in {:.1}s ({:.2} t/s) — gen {:?}",
        prompt.len(),
        n_predict,
        decode_s,
        tps,
        out_tokens.as_slice()
    ))
}

// ── T-배치 프리필 (plans/121 A1-pp) ─────────────────────────────────────
//
// 순차 디코드 대비: 선형은 T-배치 GEMM(연산강도 32배 — 메모리 벽 탈출),
// 비선형은 core 자산 재사용 — GDN 청크 스캔(core::gdn::gdn_chunk_seq,
// AR 등가 검증됨), 어텐션/활성화는 행 병렬. 수치 클래스는 순차 경로와
// 동일(f16 쌍 누산 + FOLD=4 케이던스 — gemm 커널 주석 참조).

/// 원시 포인터 usize 래퍼 — par_rows 잡 캡처용(gdn.rs SendPtr와 동일 계약).
#[derive(Clone, Copy)]
struct PP(usize);

/// 행 병렬 — thread::scope 청크. body는 Sync(공유 읽기)이며 행별 분리
/// 쓰기는 원시 포인터(PP)로 수행한다(호출 스코프 내 유효 — scope join 증명).
fn par_rows(t_rows: usize, body: impl Fn(usize) + Sync + Send) {
    if t_rows == 0 {
        return;
    }
    let nt = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(8)
        .min(t_rows);
    let per = t_rows.div_ceil(nt);
    std::thread::scope(|s| {
        for lo in (0..t_rows).step_by(per) {
            let hi = (lo + per).min(t_rows);
            let body = &body;
            s.spawn(move || (lo..hi).for_each(body));
        }
    });
}

/// T-행 GDN 층: 배치 선형(qkv+z) + conv1d(T) + 청크 스캔 + norm_gated +
/// out_proj 배치. 상태(gdn states·conv 링)는 순차 경로와 동일 형식으로
/// 갱신 — 이후 decode_step 연속 가능.
fn gdn_batch(
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

    // 선형: qkv+z 공유 입력 1배치
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

    // 청크 스캔 — core::gdn 재사용(CS=64, AR 등가 검증).
    let mut o_lc = vec![0f32; t_rows * d_inner];
    let _gc2 = ph("ppg:chunk");
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
    Ok(r)
}

/// T-행 full-attention 층: 배치 선형(q/k/v) + 행별 norm·rope·KV 적립 +
/// 인과 어텐션(행 병렬) + 게이트 + o_proj 배치.
fn attn_batch(
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

/// 배치 프리필: 전체 프롬프트를 T-배치로 통과(>512 청크 분할) — 마지막
/// 토큰 logits 반환. 상태는 decode_step 연속 가능 형식으로 적립.
/// 산술: 10a(환원 순서·청크 분해 — 골든/토큰 기준 갱신 대상).
pub fn prefill_batch(
    tr: &mut TrellisResident,
    seq: &mut SeqState,
    tokens: &[u32],
) -> Result<Vec<f32>, String> {
    let h = tr.hidden;
    let eps = 1e-6f32;
    if tokens.is_empty() {
        return Err("prefill_batch: 빈 프롬프트".into());
    }
    let mut logits = Vec::new();
    for chunk_toks in tokens.chunks(super::exl3_resident::BATCH_TMAX) {
        let t_rows = chunk_toks.len();
        // 임베딩 행 조립
        let _e0 = ph("pp:embed");
        let mut x = vec![0f32; t_rows * h];
        for (t, &tok) in chunk_toks.iter().enumerate() {
            x[t * h..(t + 1) * h].copy_from_slice(tr.embed_row(tok));
        }
        drop(_e0);
        // 스테이징 버퍼 — 이 청크의 모든 선형 입력이 여기에 병렬 직접 기록된다.
        let stage = tr.stage_f32()?;
        let mut attn_count = 0;
        for il in 0..tr.n_layers {
            let lp = format!("model.language_model.layers.{il}");
            let full = il % 4 == 3;
            let norm_w = tr
                .norm(&format!("{lp}.input_layernorm.weight"))
                .ok_or("norm missing")?
                .to_vec();
            // xn 행 rms — 병렬 이중 기록(CPU 슬라이스 + 스테이징 —
            // alpha/beta dot용 xn은 CPU에도 필요).
            let _n0 = ph("pp:norm_x");
            let mut xn = vec![0f32; t_rows * h];
            {
                let (xp, np, op, sp) = (
                    PP(x.as_ptr() as usize),
                    PP(norm_w.as_ptr() as usize),
                    PP(xn.as_mut_ptr() as usize),
                    PP(stage as usize),
                );
                // SAFETY: 행별 분리 쓰기(xn·스테이징).
                par_rows(t_rows, move |t| unsafe {
                    let xr = std::slice::from_raw_parts((xp.0 as *const f32).add(t * h), h);
                    let ss: f32 = xr.iter().map(|&v| v * v).sum();
                    let inv = 1.0 / ((ss / h as f32 + eps).sqrt());
                    let ob = (op.0 as *mut f32).add(t * h);
                    let sb = (sp.0 as *mut f32).add(t * h);
                    let nw = np.0 as *const f32;
                    for i in 0..h {
                        let v = *xr.get_unchecked(i) * inv * *nw.add(i);
                        *ob.add(i) = v;
                        *sb.add(i) = v;
                    }
                });
            }
            drop(_n0);
            let attn_out = if full {
                let _g = ph("pp:attn");
                let r = attn_batch(tr, seq, il, attn_count, t_rows, 24, 4, 256)?;
                drop(_g);
                attn_count += 1;
                r
            } else {
                let _g = ph("pp:gdn");
                let r = gdn_batch(tr, seq, il, &xn, t_rows)?;
                drop(_g);
                r
            };
            // 잔차 x += attn_out
            let _r0 = ph("pp:resid");
            for (a, b) in x.iter_mut().zip(attn_out.iter()) {
                *a += b;
            }
            drop(_r0);
            let ffn_norm_w = tr
                .norm(&format!("{lp}.post_attention_layernorm.weight"))
                .ok_or("ffn norm missing")?
                .to_vec();
            // xf 행 rms — 스테이징 단일 기록(FFN 트리오가 곧 소비).
            let _n1 = ph("pp:norm_f");
            {
                let (xp, np, sp) = (
                    PP(x.as_ptr() as usize),
                    PP(ffn_norm_w.as_ptr() as usize),
                    PP(stage as usize),
                );
                // SAFETY: 행별 분리 쓰기(스테이징).
                par_rows(t_rows, move |t| unsafe {
                    let xr = std::slice::from_raw_parts((xp.0 as *const f32).add(t * h), h);
                    let ss: f32 = xr.iter().map(|&v| v * v).sum();
                    let inv = 1.0 / ((ss / h as f32 + eps).sqrt());
                    let sb = (sp.0 as *mut f32).add(t * h);
                    let nw = np.0 as *const f32;
                    for i in 0..h {
                        *sb.add(i) = *xr.get_unchecked(i) * inv * *nw.add(i);
                    }
                });
            }
            drop(_n1);
            // FFN 트리오(단일 배치): gate/up gemm → ew_t(GPU) → down gemm.
            let _gf = ph("pp:ffn");
            let ffn_out = tr.ffn_trio_batch(
                &format!("{lp}.mlp.gate_proj"),
                &format!("{lp}.mlp.up_proj"),
                &format!("{lp}.mlp.down_proj"),
                t_rows,
            )?;
            drop(_gf);
            let _r1 = ph("pp:resid2");
            for (a, b) in x.iter_mut().zip(ffn_out.iter()) {
                *a += b;
            }
            drop(_r1);
        }
        seq.pos += t_rows as u32;
        // 마지막 청크: output_norm + lm_head(마지막 행만)
        let out_norm_w = tr
            .norm("model.language_model.norm.weight")
            .ok_or("output norm missing")?
            .to_vec();
        let last = &x[(t_rows - 1) * h..t_rows * h];
        let xn_last = rms_norm(last, &out_norm_w, eps);
        logits = tr.linear("lm_head", &xn_last)?;
    }
    Ok(logits)
}

/// `llm170 exl3-pp <dir> <token_ids> [n_predict]` — 배치 프리필 검증+벤치:
/// ① 순차 프리필 기준 로짓 확보 ② 배치 프리필 로짓 비교(상관·argmax)
/// ③ 이어서 greedy 생성 토큰 비교(배치→순차 디코드 전환 정합)
/// ④ 배치 pp 3회 중앙값 t/s.
pub fn exl3_pp(dir: &str, tokens_str: &str, n_predict: usize) -> Result<String, String> {
    let prompt: Vec<u32> = tokens_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if prompt.is_empty() {
        return Err("토큰 ID 필요 (쉼표 구분)".into());
    }
    eprintln!("  [exl3-pp] 상주 적재 중...");
    let t0 = std::time::Instant::now();
    let mut tr = TrellisResident::load(dir)?;
    eprintln!(
        "  [exl3-pp] 적재 완료 {:.1}s — 프롬프트 {} 토큰",
        t0.elapsed().as_secs_f64(),
        prompt.len()
    );
    let ctx_len = (prompt.len() + n_predict + 64).max(512);
    // 벤치 전용 모드 — 순차 기준·비교 생략(진단 루프용, 검증은 전체 모드로)
    let bench_only = llm170_diag::flag::on("LLM170_PP_BENCH_ONLY");

    // ① 순차 기준
    let mut seq1 = new_seq_state(tr.n_layers, ctx_len);
    let mut logits_seq = Vec::new();
    let mut seq_s = 0f64;
    if !bench_only {
        let t1 = std::time::Instant::now();
        for &tok in &prompt {
            logits_seq = decode_step(&mut tr, &mut seq1, tok)?;
        }
        seq_s = t1.elapsed().as_secs_f64();
    }

    // ② 배치 프리필
    let mut seq2 = new_seq_state(tr.n_layers, ctx_len);
    let t2 = std::time::Instant::now();
    let logits_b = prefill_batch(&mut tr, &mut seq2, &prompt)?;
    let bat_s = t2.elapsed().as_secs_f64();

    // 비교: 상관·최대차·argmax
    let n = logits_seq.len().min(logits_b.len());
    let (mut sxy, mut sx, mut sy, mut sxx, mut syy) = (0f64, 0f64, 0f64, 0f64, 0f64);
    let mut maxd = 0f32;
    for i in 0..n {
        let (a, b) = (logits_seq[i] as f64, logits_b[i] as f64);
        sxy += a * b;
        sx += a;
        sy += b;
        sxx += a * a;
        syy += b * b;
        maxd = maxd.max((logits_seq[i] - logits_b[i]).abs());
    }
    let corr = (n as f64 * sxy - sx * sy)
        / ((n as f64 * sxx - sx * sx) * (n as f64 * syy - sy * sy)).sqrt();
    let am_seq = logits_seq
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let am_b = logits_b
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    // ③ greedy 생성 비교(각 상태에서 n_predict)
    let gen_from = |tr: &mut TrellisResident, seq: &mut SeqState, lg: &[f32]| -> Vec<u32> {
        let mut out = Vec::new();
        let mut logits = lg.to_vec();
        for step in 0..n_predict {
            let (best, _) = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap_or((0, &0.0));
            out.push(best as u32);
            if step + 1 < n_predict {
                logits = match decode_step(tr, seq, best as u32) {
                    Ok(v) => v,
                    Err(_) => break,
                };
            }
        }
        out
    };
    let gen_seq = if bench_only {
        Vec::new()
    } else {
        gen_from(&mut tr, &mut seq1, &logits_seq)
    };
    let gen_b = gen_from(&mut tr, &mut seq2, &logits_b);

    // ④ 배치 pp 타이밍 3회 중앙값(상태 할당 제외)
    let mut times: Vec<f64> = Vec::new();
    for _ in 0..3 {
        let mut s = new_seq_state(tr.n_layers, ctx_len);
        let t = std::time::Instant::now();
        prefill_batch(&mut tr, &mut s, &prompt)?;
        times.push(t.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = times[times.len() / 2];

    // 진단 덤프 — VK_TS: GPU 디스패치 집계, exl3_phase: CPU 위상 분해.
    if llm170_diag::flag::on("LLM170_VK_TS") {
        tr.ctx.ts_report();
    }
    phase_report();

    let seq_tps = prompt.len() as f64 / seq_s;
    let bat_tps = prompt.len() as f64 / med;
    Ok(format!(
        "exl3-pp: 프롬프트 {} — 순차 {seq_tps:.2} t/s vs 배치 {bat_tps:.2} t/s (3회 중앙값, 배치 최초 {bat_s:.2}s)\n  로짓 corr={corr:.6} maxdiff={maxd:.4} argmax {}=={}{}\n  생성 {}토큰: 순차 {:?} 배치 {:?} {}",
        prompt.len(),
        am_seq,
        am_b,
        if am_seq == am_b {
            "일치"
        } else {
            "불일치"
        },
        n_predict,
        gen_seq,
        gen_b,
        if gen_seq == gen_b {
            "일치"
        } else {
            "불일치"
        },
    ))
}
