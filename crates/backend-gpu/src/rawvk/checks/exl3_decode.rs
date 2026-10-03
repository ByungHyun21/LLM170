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
    /// MTP 드래프트 층 자체 KV (plans/121 A2) — 1층.
    pub mtp_kv: Vec<KvCache>,
    /// 본체 잔차 hidden(output_norm 전) — MTP h_in 스냅샷(qwen35 mtp_h 관례).
    pub last_h: Vec<f32>,
    /// 마지막 타깃 로짓(스펙 검증 기준 — 불필요 시 비움).
    pub last_logits: Vec<f32>,
    pub last_tok: u32,
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
    seq.last_tok = token;

    // MTP 스냅샷(plans/121 A2) — h_in은 output_norm 전 잔차(qwen35 mtp_h 관례).
    seq.last_h.clear();
    seq.last_h.extend_from_slice(&x);

    // output_norm + lm_head
    let out_norm_w = tr
        .norm("model.language_model.norm.weight")
        .ok_or("output norm missing")?;
    let xn = rms_norm(&x, out_norm_w, eps);
    let _gh = ph("head");
    let r = tr.linear("lm_head", &xn)?;
    drop(_gh);
    seq.last_logits.clear();
    seq.last_logits.extend_from_slice(&r);
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
        mtp_kv: (0..1)
            .map(|_| KvCache {
                k: vec![0f32; ctx_len * 4 * 256],
                v: vec![0f32; ctx_len * 4 * 256],
                len: 0,
            })
            .collect(),
        last_h: Vec::new(),
        last_logits: Vec::new(),
        last_tok: 0,
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
    // 스레드 수: 전 코어(SMT 포함) 최적 — 2026-10-03 A/B 측정(pp512 3회
    // 중앙값): 32=60.53 > 24=59.66 > 16=59.31 t/s. 물리코어 제한 가설 기각.
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

    // ── F1 GPU 경로(plans/121): T>8에서 전 비선형 GPU 상주 ──
    // qkv+z를 yb에 남기고 → conv→l2perm→scan→gate 4커널 → gated가 xtb에.
    if t_rows > 8 {
        tr.gdn_frame_init()?;
        let gdn_il = (0..il).filter(|i| i % 4 != 3).count();
        // GPU 상태/ring 버퍼는 alloc_host_cached로 제로 보장 없음 — CPU 상태를
        // 업로드(초기 전부 0, 이후 gdn_state_sync가 갱신된 값 유지).
        {
            let g = &seq.gdn[il];
            tr.gdn_state_upload(gdn_il, &g.states, &g.conv)?;
        }
        let _gf = ph("ppg:gpu_layer");
        let slots = tr.linear_batch_multi_gpu(
            &[&format!("{lp}.in_proj_qkv"), &format!("{lp}.in_proj_z")],
            t_rows,
        )?;
        let (yb0, _n0) = slots[0];
        let (yb1, _n1) = slots[1];
        if il == 0 {
            let qkv_head = tr.read_yb_head(0, 5);
            let z_head = tr.read_yb_head(1, 5);
            eprintln!("  [f1dbg] L0 yb0(qkv): {:?} yb1(z): {:?}", qkv_head, z_head);
        }
        tr.gdn_layer_gpu(gdn_il, t_rows, std::ptr::null_mut(), yb0, yb1)?;
        if il == 0 {
            let gqr_head = tr.read_gqr_head(5);
            let gq_head = tr.read_gq_head(5);
            let gbg_head = tr.read_gbg_head(10);
            eprintln!("  [f1dbg] L0 gqr(conv q): {:?}", gqr_head);
            eprintln!("  [f1dbg] L0 gq(L2 q): {:?}", gq_head);
            eprintln!("  [f1dbg] L0 gbg(beta|g): {:?}", gbg_head);
        }
        drop(_gf);
        // gate가 xtb에 기록한 gated를 호스트 가시화 — 이후 flush가 올바른
        // 데이터를 GPU에 밀게 한다(invalidate 없으면 스테일 xn이 덮어씀).
        tr.invalidate_xtb(t_rows * 6144 * 4);
        // GPU 상태 → SeqState 동기화(차기 디코드 정합): 상태 다운로드.
        {
            let (states, conv) = {
                let g = &mut seq.gdn[il];
                (&mut g.states, &mut g.conv)
            };
            tr.gdn_state_sync(gdn_il, states, conv)?;
        }
        // out_proj: gated가 xtb에 있으므로 staged 호출로 결과 반환.
        let out = tr.linear_batch_staged(&format!("{lp}.out_proj"), t_rows)?;
        if il == 0 {
            eprintln!("  [f1dbg] L0 out_proj GPU: {:?}", &out[..5.min(out.len())]);
        }
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
    if il == 0 {
        eprintln!("  [f1dbg] L0 CPU qkv: {:?} z: {:?}", &qkv[..5], &z[..5]);
    }

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
            &q_raw_dbg
        );
    }
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
        seq.last_tok = *chunk_toks.last().ok_or("빈 청크")?;
        // MTP h 스냅샷 — 청크 마지막 행의 잔차.
        seq.last_h.clear();
        seq.last_h
            .extend_from_slice(&x[(t_rows - 1) * h..t_rows * h]);
        // 마지막 청크: output_norm + lm_head(마지막 행만)
        let out_norm_w = tr
            .norm("model.language_model.norm.weight")
            .ok_or("output norm missing")?
            .to_vec();
        let last = &x[(t_rows - 1) * h..t_rows * h];
        let xn_last = rms_norm(last, &out_norm_w, eps);
        logits = tr.linear("lm_head", &xn_last)?;
        seq.last_logits.clear();
        seq.last_logits.extend_from_slice(&logits);
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

// ── MTP 드래프트 층 (plans/121 A2) ─────────────────────────────────────
// qwen35 spec.rs 수학의 트렐리스 판: eh_proj=mtp.fc, enorm/hnorm=pre_fc_norm_*,
// attn_norm=input_layernorm, shared_head_norm=mtp.norm, head=lm_head 공유.
// KV 규약 "슬롯=pos"(QA-27): 드래프트가 쓴 슬롯은 수용 후 타깃 훅(h=본체
// 잔차)이 다시 쓴다 — 훅이 드래프트 로짓도 함께 낸다(Q35 GPU 경로 패턴).

/// MTP 1스텝: (token, h_in) → (logits, mtp_hidden). pos는 이 토큰의 위치.
fn mtp_step(
    tr: &mut TrellisResident,
    seq: &mut SeqState,
    token: u32,
    h_in: &[f32],
    pos: u32,
    with_logits: bool,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let h = tr.hidden;
    let eps = 1e-6f32;
    let enorm = tr
        .norm("mtp.pre_fc_norm_embedding.weight")
        .ok_or("mtp enorm")?;
    let hnorm = tr
        .norm("mtp.pre_fc_norm_hidden.weight")
        .ok_or("mtp hnorm")?;
    let e = tr.embed_row(token);
    let e_n = rms_norm(e, enorm, eps);
    let h_n = rms_norm(h_in, hnorm, eps);
    let mut cat = Vec::with_capacity(2 * h);
    cat.extend_from_slice(&e_n);
    cat.extend_from_slice(&h_n);
    let mut cur = tr.linear("mtp.fc", &cat)?;

    // gated attention — input_layernorm → q/k/v → norm+rope → 자체 KV → o_proj
    let lp = "mtp.layers.0.self_attn";
    let attn_norm_w = tr
        .norm("mtp.layers.0.input_layernorm.weight")
        .ok_or("mtp attn_norm")?;
    let xn = rms_norm(&cur, attn_norm_w, eps);
    let (q_gate, k, v) = tr.linear_triple(
        &format!("{lp}.q_proj"),
        &format!("{lp}.k_proj"),
        &format!("{lp}.v_proj"),
        &xn,
    )?;
    let q_norm_w = tr
        .norm(&format!("{lp}.q_norm.weight"))
        .ok_or("mtp q_norm")?;
    let k_norm_w = tr
        .norm(&format!("{lp}.k_norm.weight"))
        .ok_or("mtp k_norm")?;
    let (n_head, n_kv, head_dim, n_rot) = (24usize, 4usize, 256usize, 64usize);
    let rope_base = 1e7f32;
    let mut q_heads = vec![0f32; n_head * head_dim];
    let mut gate_heads = vec![0f32; n_head * head_dim];
    {
        let kv = &mut seq.mtp_kv[0];
        let kv_cap = kv.k.len() / (n_kv * head_dim);
        if pos as usize >= kv_cap {
            return Err(format!("mtp_step: kv 용량 초과(pos {pos})"));
        }
        for hh in 0..n_head {
            let src = hh * head_dim * 2;
            q_heads[hh * head_dim..(hh + 1) * head_dim]
                .copy_from_slice(&q_gate[src..src + head_dim]);
            gate_heads[hh * head_dim..(hh + 1) * head_dim]
                .copy_from_slice(&q_gate[src + head_dim..src + head_dim * 2]);
        }
        for hh in 0..n_head {
            let b0 = hh * head_dim;
            let head: Vec<f32> = q_heads[b0..b0 + head_dim].to_vec();
            let n = rms_norm(&head, q_norm_w, eps);
            let mut h_rot = n;
            rope(&mut h_rot, pos, n_rot, rope_base);
            q_heads[b0..b0 + head_dim].copy_from_slice(&h_rot);
        }
        let k_base = pos as usize * n_kv * head_dim;
        for hh in 0..n_kv {
            let b0 = hh * head_dim;
            let head: Vec<f32> = k[b0..b0 + head_dim].to_vec();
            let n = rms_norm(&head, k_norm_w, eps);
            let mut h_rot = n;
            rope(&mut h_rot, pos, n_rot, rope_base);
            kv.k[k_base + hh * head_dim..k_base + (hh + 1) * head_dim].copy_from_slice(&h_rot);
            kv.v[k_base + hh * head_dim..k_base + (hh + 1) * head_dim]
                .copy_from_slice(&v[b0..b0 + head_dim]);
        }
        if kv.len < pos as usize + 1 {
            kv.len = pos as usize + 1;
        }
    }
    let scale = 1.0 / (head_dim as f32).sqrt();
    let n_rep = n_head / n_kv;
    let mut attn_out = vec![0f32; n_head * head_dim];
    {
        let kv = &seq.mtp_kv[0];
        let kv_len = (pos as usize + 1).min(kv.len);
        for hh in 0..n_head {
            let kv_h = hh / n_rep;
            let b0 = hh * head_dim;
            let mut scores = vec![0f32; kv_len];
            for (tt, sc) in scores.iter_mut().enumerate() {
                let kb = tt * n_kv * head_dim + kv_h * head_dim;
                let mut d = 0f32;
                for i in 0..head_dim {
                    d += q_heads[b0 + i] * kv.k[kb + i];
                }
                *sc = d * scale;
            }
            let maxv = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for sc in scores.iter_mut() {
                *sc = (*sc - maxv).exp();
                sum += *sc;
            }
            for tt in 0..kv_len {
                let w = scores[tt] / sum;
                let vb = tt * n_kv * head_dim + kv_h * head_dim;
                for i in 0..head_dim {
                    attn_out[b0 + i] += w * kv.v[vb + i];
                }
            }
            for i in 0..head_dim {
                attn_out[b0 + i] *= sigmoid(gate_heads[b0 + i]);
            }
        }
    }
    let o = tr.linear(&format!("{lp}.o_proj"), &attn_out)?;
    for i in 0..h {
        cur[i] += o[i];
    }

    // FFN — ffn_triple(GEMV 배치 3커널)
    let ffn_norm_w = tr
        .norm("mtp.layers.0.post_attention_layernorm.weight")
        .ok_or("mtp ffn_norm")?;
    let xf = rms_norm(&cur, ffn_norm_w, eps);
    let ffn_out = tr.ffn_triple(
        "mtp.layers.0.mlp.gate_proj",
        "mtp.layers.0.mlp.up_proj",
        "mtp.layers.0.mlp.down_proj",
        &xf,
    )?;
    for i in 0..h {
        cur[i] += ffn_out[i];
    }

    if !with_logits {
        return Ok((Vec::new(), cur));
    }
    let sh_norm = tr.norm("mtp.norm.weight").ok_or("mtp shared norm")?;
    let hn = rms_norm(&cur, sh_norm, eps);
    let logits = tr.linear("lm_head", &hn)?;
    Ok((logits, cur))
}

fn argmax32(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// \`llm170 exl3-mtp <dir> <token_ids> [n_predict]\` — MTP 드래프트 수용률·
/// 비용 실측(plans/121 A2 판정 근거): 타깃 순차 기준 생성과 교차 검증해
/// a1/a2(조건부) 수용률·mtp 스텝 벽시간을 보고한다.
pub fn exl3_mtp(dir: &str, tokens_str: &str, n_predict: usize) -> Result<String, String> {
    let prompt: Vec<u32> = tokens_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if prompt.is_empty() {
        return Err("토큰 ID 필요 (쉼표 구분)".into());
    }
    eprintln!("  [exl3-mtp] 상주 적재 중...");
    let t0 = std::time::Instant::now();
    let mut tr = TrellisResident::load(dir)?;
    eprintln!("  [exl3-mtp] 적재 완료 {:.1}s", t0.elapsed().as_secs_f64());
    let ctx_len = (prompt.len() + n_predict + 64).max(512);
    let mut seq = new_seq_state(tr.n_layers, ctx_len);

    // 프리필 — 타깃 순차 + mtp 훅(KV 적립·마지막 토큰만 로짓)
    let mut draft_logits = Vec::new();
    let mut mtp_hook_ms = 0f64;
    let t1 = std::time::Instant::now();
    for (i, &tok) in prompt.iter().enumerate() {
        let _ = decode_step(&mut tr, &mut seq, tok)?;
        let last = i + 1 == prompt.len();
        let t = std::time::Instant::now();
        let h_snap = seq.last_h.clone();
        let (dl, _) = mtp_step(&mut tr, &mut seq, tok, &h_snap, i as u32, last)?;
        mtp_hook_ms += t.elapsed().as_secs_f64() * 1e3;
        if last {
            draft_logits = dl;
        }
    }
    let pf_s = t1.elapsed().as_secs_f64();

    // 생성 — 매 스텝: d0(훅 드래프트) vs 타깃 greedy; d1 체인(조건부)
    let (mut a1n, mut a1d, mut a2n, mut a2d) = (0u64, 0u64, 0u64, 0u64);
    let (mut corr_acc, mut corr_n) = (0f64, 0u64);
    let (mut dec_ms, mut mtp2_ms) = (0f64, 0f64);
    let t2 = std::time::Instant::now();
    let mut out_tokens = Vec::new();
    let mut prev_d0: Option<u32> = None;
    let mut prev_d1: Option<u32> = None;
    for step in 0..n_predict {
        let t0_ = greedy_ref(&seq.last_logits);
        let d0 = argmax32(&draft_logits);
        if d0 == t0_ {
            a1n += 1;
        }
        a1d += 1;
        // k=2 체인 드래프트(이전 스텝 수용 조건부 측정)
        if prev_d0.is_some() && prev_d0 == Some(t0_) {
            // 직전 d0이 수용됨 — prev_d1 비교는 이전 루프에서 이미 예약됨
        }
        out_tokens.push(t0_);
        // 디버그: 드래프트↔타깃 로짓 상관(정렬/산술 판별)
        {
            let n = draft_logits.len().min(seq.last_logits.len());
            let (mut sxy, mut sx, mut sy, mut sxx, mut syy) = (0f64, 0f64, 0f64, 0f64, 0f64);
            for i in 0..n {
                let (a, b) = (draft_logits[i] as f64, seq.last_logits[i] as f64);
                sxy += a * b;
                sx += a;
                sy += b;
                sxx += a * a;
                syy += b * b;
            }
            let c = (n as f64 * sxy - sx * sy)
                / ((n as f64 * sxx - sx * sx) * (n as f64 * syy - sy * sy)).sqrt();
            corr_acc += c;
            corr_n += 1;
            if step < 3 {
                eprintln!(
                    "  [mtp-dbg] step{step}: d0={d0} t0={t0_} corr={c:.4} dl0={:.3} tl0={:.3}",
                    draft_logits[0], seq.last_logits[0]
                );
            }
        }
        if step + 1 < n_predict {
            let t = std::time::Instant::now();
            let _ = decode_step(&mut tr, &mut seq, t0_)?;
            dec_ms += t.elapsed().as_secs_f64() * 1e3;
            // 훅: 수용 토큰 KV 재기입 + 다음 드래프트 + (수용 시) d1 체인
            let t = std::time::Instant::now();
            let h_snap = seq.last_h.clone();
            let pos_snap = seq.pos - 1;
            let (dl, hm) = mtp_step(&mut tr, &mut seq, t0_, &h_snap, pos_snap, true)?;
            mtp2_ms += t.elapsed().as_secs_f64() * 1e3;
            draft_logits = dl;
            prev_d0 = Some(d0);
            prev_d1 = None;
            let hm_kept = hm;
            if d0 == t0_ && step + 2 < n_predict {
                // d1: 수용된 d0으로 체인 1스텝 더
                let pos_next = seq.pos;
                let (l1, _) = mtp_step(&mut tr, &mut seq, d0, &hm_kept, pos_next, true)?;
                prev_d1 = Some(argmax32(&l1));
            }
            // 다음 스텝에서 a2 판정: prev_d1 == 다음 타깃 greedy && prev_d0 수용
            if prev_d1.is_some() && prev_d0 == Some(d0) {
                // a2 판정 예약 — 실제 비교는 루프 끝 a2 블록에서.
            }
        }
        // 간이 a2 측정: 이전 스텝 d0 수용 && d1 존재 → 현재 t0_와 비교
        if let Some(pd1) = prev_d1.take() {
            a2d += 1;
            if pd1 == t0_ {
                a2n += 1;
            }
        }
    }
    let gen_s = t2.elapsed().as_secs_f64();

    let a1 = a1n as f64 / a1d.max(1) as f64;
    let a2 = a2n as f64 / a2d.max(1) as f64;
    let corr = corr_acc / corr_n.max(1) as f64;
    let dec_avg = dec_ms / n_predict.max(1) as f64;
    let mtp_avg = mtp2_ms / n_predict.max(1) as f64;
    let base_tps = n_predict as f64 / gen_s;
    Ok(format!(
        "exl3-mtp: 프롬프트 {}({:.1}s, mtp훅 {:.1}ms/tok) + {}스텝 | a1={:.3}({}/{}) a2={:.3}({}/{}) corr={:.4} | 타깃 {:.1}ms/스텝 mtp {:.1}ms/스텝 | 순차기준 {:.2} t/s",
        prompt.len(),
        pf_s,
        mtp_hook_ms / prompt.len().max(1) as f64,
        n_predict,
        a1,
        a1n,
        a1d,
        a2,
        a2n,
        a2d,
        corr,
        dec_avg,
        mtp_avg,
        base_tps,
    ))
}

fn greedy_ref(v: &[f32]) -> u32 {
    argmax32(v)
}

// ── 스펙 라운드 (plans/121 A2) ─────────────────────────────────────────
// k 드래프트(mtp 체인) → 상태 스냅샷 → [d0..d_{k-1}] T=k 배치 타깃 forward
// → 행별 argmax 검증 → 발산 시 롤백+수용 접두 재실행. GDN 201MB 클론이
// 라운드당 ~25ms(7% 수준) — KV는 len 절단만(멱등 재기입).

/// 스펙 프리필 변형 — 행별 argmax 반환(검증용). 마지막 hidden/logits도
/// 갱신(prefill_batch와 동일 계약).
pub fn prefill_batch_spec(
    tr: &mut TrellisResident,
    seq: &mut SeqState,
    tokens: &[u32],
) -> Result<Vec<u32>, String> {
    let h = tr.hidden;
    let eps = 1e-6f32;
    let mut argmaxes = Vec::new();
    for chunk_toks in tokens.chunks(super::exl3_resident::BATCH_TMAX) {
        let t_rows = chunk_toks.len();
        let _e0 = ph("pp:embed");
        let mut x = vec![0f32; t_rows * h];
        for (t, &tok) in chunk_toks.iter().enumerate() {
            x[t * h..(t + 1) * h].copy_from_slice(tr.embed_row(tok));
        }
        drop(_e0);
        let stage = tr.stage_f32()?;
        let mut attn_count = 0;
        for il in 0..tr.n_layers {
            let lp = format!("model.language_model.layers.{il}");
            let full = il % 4 == 3;
            let norm_w = tr
                .norm(&format!("{lp}.input_layernorm.weight"))
                .ok_or("norm missing")?
                .to_vec();
            let _n0 = ph("pp:norm_x");
            let mut xn = vec![0f32; t_rows * h];
            {
                let (xp, np, op, sp) = (
                    PP(x.as_ptr() as usize),
                    PP(norm_w.as_ptr() as usize),
                    PP(xn.as_mut_ptr() as usize),
                    PP(stage as usize),
                );
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
            let _r0 = ph("pp:resid");
            for (a, b) in x.iter_mut().zip(attn_out.iter()) {
                *a += b;
            }
            drop(_r0);
            let ffn_norm_w = tr
                .norm(&format!("{lp}.post_attention_layernorm.weight"))
                .ok_or("ffn norm missing")?
                .to_vec();
            let _n1 = ph("pp:norm_f");
            {
                let (xp, np, sp) = (
                    PP(x.as_ptr() as usize),
                    PP(ffn_norm_w.as_ptr() as usize),
                    PP(stage as usize),
                );
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
        seq.last_h.clear();
        seq.last_h
            .extend_from_slice(&x[(t_rows - 1) * h..t_rows * h]);
        // 행별 head — output_norm 행별 후 lm_head 배치.
        let out_norm_w = tr
            .norm("model.language_model.norm.weight")
            .ok_or("output norm missing")?
            .to_vec();
        {
            let (xp, np, sp) = (
                PP(x.as_ptr() as usize),
                PP(out_norm_w.as_ptr() as usize),
                PP(stage as usize),
            );
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
        let ys = tr.linear_batch_multi_staged(&["lm_head"], t_rows)?;
        let vocab = ys[0].len() / t_rows;
        for t in 0..t_rows {
            argmaxes.push(argmax32(&ys[0][t * vocab..(t + 1) * vocab]));
        }
        seq.last_logits.clear();
        let last_row = t_rows - 1;
        seq.last_logits
            .extend_from_slice(&ys[0][last_row * vocab..(last_row + 1) * vocab]);
    }
    Ok(argmaxes)
}

/// 스펙 상태 스냅샷 — GDN/conv 클론 + KV len + pos + hidden/로짓.
pub struct SpecSnap {
    gdn_states: Vec<Vec<f32>>,
    conv: Vec<f32>,
    kv_lens: Vec<usize>,
    /// 발산 롤백용 — 드래프트가 늘린 mtp KV len도 되돌린다(잔여
    /// 슬롯 내용은 헤드 재기입이 덮는다).
    mtp_kv_lens: Vec<usize>,
    pos: u32,
    last_h: Vec<f32>,
    last_logits: Vec<f32>,
}

fn spec_snap(seq: &SeqState) -> SpecSnap {
    SpecSnap {
        gdn_states: seq.gdn.iter().map(|g| g.states.clone()).collect(),
        conv: seq
            .gdn
            .iter()
            .flat_map(|g| g.conv.iter().copied())
            .collect(),
        kv_lens: seq.kv.iter().map(|k| k.len).collect(),
        mtp_kv_lens: seq.mtp_kv.iter().map(|k| k.len).collect(),
        pos: seq.pos,
        last_h: seq.last_h.clone(),
        last_logits: seq.last_logits.clone(),
    }
}

fn spec_restore(seq: &mut SeqState, snap: &SpecSnap) {
    for (g, st) in seq.gdn.iter_mut().zip(snap.gdn_states.iter()) {
        g.states.copy_from_slice(st);
    }
    let mut off = 0usize;
    for g in seq.gdn.iter_mut() {
        let n = g.conv.len();
        g.conv.copy_from_slice(&snap.conv[off..off + n]);
        off += n;
    }
    for (k, &l) in seq.kv.iter_mut().zip(snap.kv_lens.iter()) {
        k.len = l;
    }
    for (k, &l) in seq.mtp_kv.iter_mut().zip(snap.mtp_kv_lens.iter()) {
        k.len = l;
    }
    seq.pos = snap.pos;
    seq.last_h.copy_from_slice(&snap.last_h);
    seq.last_logits.copy_from_slice(&snap.last_logits);
}

/// 스펙 1라운드: 반환 (수용 토큰열, 타깃 forward 수). 계약 — seq는 마지막
/// 확정 토큰까지 처리된 상태(last_logits/last_h/last_tok 유효). 수용 토큰은
/// 최대 k+1(전 수용 시 선행 1 토큰 포함), 최소 1(발산 보정 토큰).
pub fn exl3_spec_step(
    tr: &mut TrellisResident,
    seq: &mut SeqState,
    k: usize,
) -> Result<(Vec<u32>, usize), String> {
    let k = k.clamp(1, 4);
    // ① 드래프트 체인 — 시드 (last_tok, last_h), mtp KV 슬롯 = pos+i.
    let mut drafts = Vec::with_capacity(k);
    let mut tok = seq.last_tok;
    let mut h = seq.last_h.clone();
    for i in 0..k {
        let pos = seq.pos + i as u32;
        let (lgt, hm) = mtp_step(tr, seq, tok, &h, pos, true)?;
        drafts.push(argmax32(&lgt));
        tok = drafts[i];
        h = hm;
    }
    // ② 검증 기준 g0(스냅샷 전 last_logits) + 상태 스냅샷.
    let g0 = argmax32(&seq.last_logits);
    let snap = spec_snap(seq);
    // ③ [d0..d_{k-1}] T=k 배치 타깃 forward — 행별 argmax.
    let row_am = prefill_batch_spec(tr, seq, &drafts)?;
    // ④ 수용 보행.
    let mut accepted: Vec<u32> = Vec::with_capacity(k + 1);
    let mut diverged = false;
    for i in 0..k {
        let gi = if i == 0 { g0 } else { row_am[i - 1] };
        if drafts[i] == gi {
            accepted.push(drafts[i]);
        } else {
            accepted.push(gi);
            diverged = true;
            break;
        }
    }
    let forwards = if diverged {
        // ⑤ 롤백 + 수용 접두(보정 토큰 포함) 재실행.
        spec_restore(seq, &snap);
        let re_accepted = accepted.clone();
        prefill_batch_spec(tr, seq, &re_accepted)?;
        2
    } else {
        accepted.push(row_am[k - 1]); // 전 수용 — 선행 보너스 토큰
        1
    };
    // ⑥ MTP 훅 — 마지막 수용 토큰만 (본체 h) 재기입 + 다음 드래프트 로짓.
    //    중간 슬롯은 드래프트 시점 기입 (tok==수용 토큰) 상태로 둔다 —
    //    k/v는 (토큰, mtp_h) 쌍이고 수용 토큰이 같으므로 근사 오차는
    //    다음 라운드 검증이 흡수한다(원장 128 수용률 수준과 동일 논리).
    {
        // 훅 입력은 '마지막 처리 토큰' — 전수용 경로의 accepted 마지막은
        // 선행 보너스(미처리)일 수 있어 last_tok을 쓴다(슬롯=seq.pos).
        let h_last = seq.last_h.clone();
        let _ = mtp_step(tr, seq, seq.last_tok, &h_last, seq.pos, true)?;
    }
    Ok((accepted, forwards))
}

/// `llm170 exl3-mtp2 <dir> <token_ids> [n_predict] [k]` — 스펙 라운드 실측:
/// 유효 t/s(스냅샷·롤백·훅 비용 포함) + 수용률 + forward 수. 그리디 기준
/// 토큰열과의 동일성도 보고(스펙 경로 정합 게이트).
pub fn exl3_mtp2(
    dir: &str,
    tokens_str: &str,
    n_predict: usize,
    k: usize,
) -> Result<String, String> {
    let prompt: Vec<u32> = tokens_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if prompt.is_empty() {
        return Err("토큰 ID 필요 (쉼표 구분)".into());
    }
    eprintln!("  [exl3-mtp2] 상주 적재 중...");
    let t0 = std::time::Instant::now();
    let mut tr = TrellisResident::load(dir)?;
    eprintln!("  [exl3-mtp2] 적재 완료 {:.1}s", t0.elapsed().as_secs_f64());
    let ctx_len = (prompt.len() + n_predict + 64).max(512);
    let mut seq = new_seq_state(tr.n_layers, ctx_len);

    // 프리필(배치) — 마지막 토큰 훅으로 다음 드래프트 시드.
    let am = prefill_batch_spec(&mut tr, &mut seq, &prompt)?;
    {
        let h_last = seq.last_h.clone();
        let pos = seq.pos - 1;
        let tok = seq.last_tok;
        let _ = mtp_step(&mut tr, &mut seq, tok, &h_last, pos, true)?;
        let _ = am;
    }

    // 스펙 루프
    let t2 = std::time::Instant::now();
    let (mut toks, mut fwds, mut rounds) = (0u64, 0u64, 0u64);
    let mut spec_tokens = Vec::new();
    while spec_tokens.len() < n_predict {
        let (acc, fw) = exl3_spec_step(&mut tr, &mut seq, k)?;
        toks += acc.len() as u64;
        fwds += fw as u64;
        rounds += 1;
        for t in acc {
            if spec_tokens.len() < n_predict {
                spec_tokens.push(t);
            }
        }
    }
    let spec_s = t2.elapsed().as_secs_f64();
    // 진단 덤프 — exl3_phase 위상 분해(exl3_pp와 동일 원장 89 키).
    phase_report();
    let tps = toks as f64 / spec_s;

    // 정합 — 같은 상태에서 순차 greedy 재현(토큰 동일성 게이트).
    let mut seq2 = new_seq_state(tr.n_layers, ctx_len);
    let mut logits = prefill_batch(&mut tr, &mut seq2, &prompt)?;
    let mut plain_tokens = Vec::new();
    for _ in 0..n_predict {
        let t = argmax32(&logits);
        plain_tokens.push(t);
        logits = decode_step(&mut tr, &mut seq2, t)?;
    }
    let same = spec_tokens == plain_tokens;
    let per_round = toks as f64 / rounds as f64;
    Ok(format!(
        "exl3-mtp2(k={k}): {}라운드 {}토큰(평균 {per_round:.2}/라운드) {}forward | 유효 {:.2} t/s | 토큰 정합 {} ({}/{} 일치)
  spec {:?}
  plain {:?}
  초 forward {:.0}ms/forward",
        rounds,
        toks,
        fwds,
        tps,
        if same { "통과" } else { "불일치" },
        spec_tokens
            .iter()
            .zip(plain_tokens.iter())
            .filter(|(a, b)| a == b)
            .count(),
        n_predict,
        &spec_tokens[..12.min(spec_tokens.len())],
        &plain_tokens[..12.min(plain_tokens.len())],
        (spec_s / fwds.max(1) as f64) * 1e3,
    ))
}
