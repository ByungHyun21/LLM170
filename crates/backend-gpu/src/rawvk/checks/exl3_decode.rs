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
    /// [48 heads][128*128] — 기씨 core::gdn 형식과 동일.
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
    // plans/120 A1: qkv·z 동일 입력 → linear_pair 1배치 (동기 2→1).
    let _gq = ph("gdn:lin_qkvz");
    let (qkv, z) = tr.linear_pair(
        &format!("{lp}.in_proj_qkv"),
        &format!("{lp}.in_proj_z"),
        x_normed,
    )?;
    drop(_gq);
    if il == 0 && seq.pos == 0 {
        let rms = (qkv.iter().map(|v| v * v).sum::<f32>() / qkv.len() as f32).sqrt();
        eprintln!(
            "  [dbg] L0 qkv rms={rms:.4} qkv[0]={:.6} qkv[1]={:.6}",
            qkv[0],
            qkv.get(1).copied().unwrap_or(0.0)
        );
    }
    // alpha: hidden → 48 (V헤드별 스케일러) — 노름에서 읽기
    let a_proj = tr
        .norm(&format!("{lp}.in_proj_a.weight"))
        .ok_or("alpha missing")?;
    let b_proj = tr
        .norm(&format!("{lp}.in_proj_b.weight"))
        .ok_or("beta missing")?;

    // a[b] = dot(x_normed, a_proj[b]) for each V head b
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
    let dt_bias = tr.norm(&format!("{lp}.dt_bias")).ok_or("dt_bias missing")?;

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
    let hf_to_lc = |h: usize| -> usize { 3 * (h % 16) + h / 16 };

    let _gd = ph("gdn:delta");
    let mut v_lc = vec![0f32; v_len]; // llama.cpp 순서 v
    let mut beta_lc = vec![0f32; n_v];
    let mut g_lc = vec![0f32; n_v];
    let mut o_lc = vec![0f32; v_len];
    // 상태도 순열 — state[llama.cpp head i] = state[HV head hf_to_lc(i)]
    let mut st_lc = vec![0f32; n_v * d_state * d_state];
    for i in 0..n_v {
        let j = hf_to_lc(i);
        v_lc[i * d_state..(i + 1) * d_state]
            .copy_from_slice(&v_all[j * d_state..(j + 1) * d_state]);
        beta_lc[i] = beta_all[j];
        g_lc[i] = g_all[j];
        st_lc[i * d_state * d_state..(i + 1) * d_state * d_state]
            .copy_from_slice(&st.states[j * d_state * d_state..(j + 1) * d_state * d_state]);
    }

    llm170_core::gdn::gdn_ar_batch(
        &q_all, &k_all, &v_lc, &beta_lc, &g_lc, &mut st_lc, &mut o_lc, 1, n_k, n_v,
    );

    // 결과·상태 역순열 (llama.cpp → HF)
    let mut o_all = vec![0f32; v_len];
    for i in 0..n_v {
        let j = hf_to_lc(i);
        o_all[j * d_state..(j + 1) * d_state]
            .copy_from_slice(&o_lc[i * d_state..(i + 1) * d_state]);
        st.states[j * d_state * d_state..(j + 1) * d_state * d_state]
            .copy_from_slice(&st_lc[i * d_state * d_state..(i + 1) * d_state * d_state]);
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
    let _g3 = ph("attn:core");
    let scale = 1.0 / (head_dim as f32).sqrt();
    let n_rep = n_head / n_kv;
    let mut attn_out = vec![0f32; n_head * head_dim];

    for h in 0..n_head {
        let kv_h = h / n_rep;
        let q = &q_heads[h * head_dim..(h + 1) * head_dim];

        // 점수 계산
        let mut scores = vec![0f32; kv.len];
        for t in 0..kv.len {
            let k_base = t * n_kv * head_dim + kv_h * head_dim;
            let mut dot = 0f32;
            for d in 0..head_dim {
                dot += q[d] * kv.k[k_base + d];
            }
            scores[t] = dot * scale;
        }

        // softmax
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = scores.iter().map(|&s| (s - max_s).exp()).sum();
        let inv_sum = 1.0 / exp_sum;

        // 가중 합
        for d in 0..head_dim {
            let mut sum = 0f32;
            for t in 0..kv.len {
                let w = (scores[t] - max_s).exp() * inv_sum;
                let v_base = t * n_kv * head_dim + kv_h * head_dim;
                sum += w * kv.v[v_base + d];
            }
            attn_out[h * head_dim + d] = sum;
        }
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
        let _gu = ph("ffn:lin_gu");
        // plans/120 A1: gate·up 동일 입력 → linear_pair 1배치.
        let (gate, up) = tr.linear_pair(
            &format!("{lp}.mlp.gate_proj"),
            &format!("{lp}.mlp.up_proj"),
            &xf,
        )?;
        drop(_gu);
        let _ga = ph("ffn:act");
        let hidden_act: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(&a, &b)| silu(a) * b)
            .collect();
        drop(_ga);
        let _gd = ph("ffn:lin_down");
        let ffn_out = tr.linear(&format!("{lp}.mlp.down_proj"), &hidden_act)?;
        drop(_gd);
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
