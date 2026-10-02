//! EXL3 레이어 스트리밍 단일 토큰 디코드 (§3-2 최소 구현).
//! TrellisResident 위에 qwen35 전방향: 선형=vk GEMV, 비선형=CPU.

use super::exl3_resident::TrellisResident;

/// RMSNorm: x * w / rms(x).
fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let ss: f32 = x.iter().map(|&v| v * v).sum();
    let inv = 1.0 / ((ss / n as f32 + eps).sqrt());
    x.iter().zip(w.iter()).map(|(&v, &g)| v * inv * g).collect()
}

/// SiLU.
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SwiGLU FFN: down(silu(gate(x)) * up(x)).
fn fwn(tr: &mut TrellisResident, layer: usize, x: &[f32]) -> Result<Vec<f32>, String> {
    let p = format!("model.language_model.layers.{layer}.mlp");
    let g = tr.linear(&format!("{p}.gate_proj"), x)?;
    let u = tr.linear(&format!("{p}.up_proj"), x)?;
    let hidden: Vec<f32> = g.iter().zip(u.iter()).map(|(&a, &b)| silu(a) * b).collect();
    tr.linear(&format!("{p}.down_proj"), &hidden)
}

/// 단일 토큰 전방향 — 결과 logits.
pub fn decode_step(tr: &mut TrellisResident, token: u32, _pos: u32) -> Result<Vec<f32>, String> {
    let h = tr.hidden;
    let eps = 1e-6f32;

    // 임베딩
    let mut x = tr.embed_row(token).to_vec();

    for il in 0..tr.n_layers {
        let lp = format!("model.language_model.layers.{il}");
        let full = il % 4 == 3;

        // attn_norm
        let norm_w = tr
            .norm(&format!("{lp}.input_layernorm.weight"))
            .ok_or("norm missing")?;
        let xn = rms_norm(&x, norm_w, eps);

        // 선형 투영 (전부 vk GEMV)
        let attn_out = if full {
            // Full attention: q/k/v/o
            let _q = tr.linear(&format!("{lp}.self_attn.q_proj"), &xn)?;
            let _k = tr.linear(&format!("{lp}.self_attn.k_proj"), &xn)?;
            let _v = tr.linear(&format!("{lp}.self_attn.v_proj"), &xn)?;
            // TODO: rope + attention + KV cache
            // 임시: v를 그대로 통과 (attention 미구현 — 검증용)
            let _ = (_q, _k);
            _v
        } else {
            // GDN: qkv → conv → scan → z-gate → norm → out
            let qkv = tr.linear(&format!("{lp}.linear_attn.in_proj_qkv"), &xn)?;
            let z = tr.linear(&format!("{lp}.linear_attn.in_proj_z"), &xn)?;
            // TODO: conv1d, GDN scan, z-gate, norm_gated
            // 임시: qkv의 v부분을 silu(z)로 게이트
            let nv = 48 * 128; // d_inner
            let mut gated = vec![0f32; nv];
            for i in 0..nv.min(qkv.len()) {
                let idx = 2 * 16 * 128 + i; // q(2048)+k(2048)+v 시작
                if idx < qkv.len() {
                    gated[i] = qkv[idx] * silu(z.get(i).copied().unwrap_or(0.0));
                }
            }
            let norm_w = tr
                .norm(&format!("{lp}.linear_attn.norm.weight"))
                .unwrap_or(&[1.0f32; 128]);
            let _ = norm_w;
            tr.linear(&format!("{lp}.linear_attn.out_proj"), &gated)?
        };

        // 잔차
        for i in 0..h {
            x[i] += attn_out.get(i).copied().unwrap_or(0.0);
        }

        // FFN (post_attention_norm)
        let ffn_norm_w = tr
            .norm(&format!("{lp}.post_attention_layernorm.weight"))
            .ok_or("ffn norm missing")?;
        let xf = rms_norm(&x, ffn_norm_w, eps);
        let ffn_out = fwn(tr, il, &xf)?;
        for i in 0..h {
            x[i] += ffn_out.get(i).copied().unwrap_or(0.0);
        }
    }

    // output_norm + lm_head
    let out_norm_w = tr
        .norm("model.language_model.norm.weight")
        .ok_or("output norm missing")?;
    let xn = rms_norm(&x, out_norm_w, eps);
    let logits = tr.linear("lm_head", &xn)?;
    Ok(logits)
}

/// `llm170 exl3-decode <dir> <token> [n_predict]`
pub fn exl3_decode(dir: &str, token: u32, n_predict: usize) -> Result<String, String> {
    let t0 = std::time::Instant::now();
    eprintln!("  [exl3-decode] 상주 적재 중...");
    let mut tr = TrellisResident::load(dir)?;
    let load_s = t0.elapsed().as_secs_f64();
    eprintln!(
        "  [exl3-decode] 선형 {}개 · 노름 {}개 · {}층 · {:.1}s",
        tr.linears.len(),
        tr.norms.len(),
        tr.n_layers,
        load_s
    );

    let t1 = std::time::Instant::now();
    let mut cur = token;
    let mut tokens = vec![cur];
    for step in 0..n_predict {
        let logits = decode_step(&mut tr, cur, step as u32 + 1)?;
        // greedy argmax
        let (best, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((0, &0.0));
        cur = best as u32;
        tokens.push(cur);
        if step < 4 {
            eprintln!("  [exl3-decode] step {step}: token={cur}");
        }
    }
    let decode_s = t1.elapsed().as_secs_f64();
    let tps = n_predict as f64 / decode_s;
    Ok(format!(
        "exl3-decode: {} tokens in {:.1}s ({:.2} t/s) — {:?}",
        n_predict,
        decode_s,
        tps,
        &tokens[..tokens.len().min(8)]
    ))
}
