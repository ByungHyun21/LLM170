//! ⚠️ 미완료(WIP) — plans/130 B5 체인 프로브. 다음 세션에서 이어서 완성할 것.
//!
//! 현재 상태: gpu_chain_step(43L GPU 체인 1스텝)·cuda_ds4_chain_check(오라클
//! 대조·greedy 3스텝·층 순서 뒤집기 네거티브)까지만 작성됨. 이 파일은
//! rawcuda/mod.rs에 등록되지 않았으므로 빌드 영향 없음.
//!
//! 남은 작업(TODO):
//! 1. mod.rs에 `pub mod ds4_chain_cuda_probe;` 등록
//! 2. shim 조기 인터셉트 + ds4_chain_main + 사용법 갱신 (ds4-chain 암)
//! 3. build_cuda.bat SRCS 확인(신규 .cu 없음 — 착지 커널 재용이라 불필요 예상)
//! 4. verify_cuda.bat ds4-chain 암 추가(양성만 — 네거티브는 인라인 판정)
//! 5. hc_dims 헬퍼: Ds4HcCuda::new(Ds4HcDims::from_config(cfg)?) 로 교정
//!    (현재 part3가 존재하지 않는 hc_dims_from_config를 참조 — 컴파일 에러 지점)
//! 6. 게이트 토큰: 픽스처 실측 토큰 [1..8] 사용 중 — 필요 시 다른 시드로 교체
//! 7. 전체 배터리 + 커밋 + plans/130 B5 완결 표기
//!
//! 오라클: llm170_core::deepseek4::frame::Ds4Frame::forward (호스트, 검증됨
//! — B2 스모크 166.8s). GPU 측은 hc/norm을 core 함수로, attention/MoE만
//! 착지 모듈(ds4_attn/ds4_moe)로 대체한 체인이다.

//! ds4-chain — DeepSeek-V4-Flash 43L greedy 체인(착지 CUDA 모듈 연결) vs
//! core deepseek4 frame 오라클 + plans/129 QA 채택 (plans/130 B5).
//!
//! 체인 접착(hc 스테이지·RMSNorm·embed·head)은 core 함수를 그대로 호출하고,
//! attention(stages D1)·MoE(stages D3)만 착지 CUDA 모듈로 대체한다 —
//! hc/norm은 core 산술 그대로이므로 체인 오차의 유일한 원인은 GPU
//! attention/MoE 스테이지다. DSpark 드래프트(D4)는 체인이 생산한
//! main_hidden에 대해 재검증한다(케이스 vii).
//!
//! 검증: 스텝별 종단 로짓 maxdiff(문서화 문턱) + greedy 토큰 일치(체인 토큰
//! 동일성 계약 — 값 문턱은 로짓에 적용) + 층 순서 뒤집기 네거티브 DETECTED.

use llm170_core::deepseek4::config::Deepseek4Config;
use llm170_core::deepseek4::frame::{Ds4Frame, ForwardOut};
use llm170_core::deepseek4::loader::Ds4Loader;
use llm170_core::deepseek4::ops::rms_norm_weighted;
use llm170_core::deepseek4::stages::hc::{hc_post, hc_pre};

use crate::rawcuda::ds4_attn_cuda::{Ds4AttnCuda, Ds4AttnDims};
use crate::rawcuda::ds4_attn_cuda_probe::{
    dequant_linear, load_layer, plain_f32, plain_kmat,
};
use crate::rawcuda::ds4_hc_cuda::Ds4HcCuda;
use crate::rawcuda::ds4_moe_cuda::Ds4MoeCuda;

use std::collections::HashMap;

/// 로짓 문턱 — 스테이지 전부 bit-exact(D1-D3)라 체인 오차는 fc16 누적 한도.
/// 실측 후 문서화(초기 상한 2e-3, FNH fn-chain 선례와 동일).
const DS4_CHAIN_LOGITS_THRESH: f32 = 2e-3;
/// 네거티브 문턱 — 로짓 뒤집기 검출 하한.
const DS4_CHAIN_NEG_THRESH: f32 = 1e-1;

/// GPU 체인 1스텝 — core block_forward 접착을 core 함수로 재현하되
/// attention(MoE)만 착지 CUDA 모듈로. 층 가중치는 사용 직전 적재,
/// 직후 drop_layer로 해제(VRAM 상한 — 43층 상주 불가).
#[allow(clippy::too_many_arguments)]
fn gpu_chain_step(
    ar: &llm170_exl3::StArchive,
    loader: &Ds4Loader,
    cfg: &Deepseek4Config,
    tokens: &[u32],
    attn: &mut Ds4AttnCuda,
    hc: &mut Ds4HcCuda,
    moe: &mut Ds4MoeCuda,
    perm: Option<&[usize]>,
) -> Result<Vec<f32>, String> {
    let (d, hc_n, t) = (cfg.dim, cfg.hc_mult, tokens.len());
    let (rope_win, rope_cmp) = {
        let win = llm170_core::deepseek4::ops::RopeTable::build(
            cfg.rope_head_dim, t, cfg.rope_theta, false,
            cfg.yarn_factor, cfg.yarn_orig_len, cfg.yarn_beta_fast, cfg.yarn_beta_slow,
        );
        let cmp = llm170_core::deepseek4::ops::RopeTable::build(
            cfg.rope_head_dim, t, cfg.compress_rope_theta, true,
            cfg.yarn_factor, cfg.yarn_orig_len, cfg.yarn_beta_fast, cfg.yarn_beta_slow,
        );
        (win, cmp)
    };
    // embed 행 gather(호스트 — 오프셋 읽기) → 4스트림 방송.
    let rows = embed_rows(ar, tokens, cfg)?;
    let mut x = vec![0.0f32; t * hc_n * d];
    for (i, row) in rows.iter().enumerate() {
        for j in 0..hc_n {
            x[i * hc_n * d + j * d..(i + 1) * d].copy_from_slice(row);
        }
    }
    let order: Vec<usize> = match perm {
        Some(p) => p.to_vec(),
        None => (0..cfg.n_layers).collect(),
    };
    for &il in &order {
        // 층 가중치 (호스트 trellis 디퀀트 → 모듈 업로드).
        let lw = load_layer(ar, &attn_dims_of(attn), il)?;
        attn.register_layer(il, &lw)?;
        let rope = if cfg.ratio(il) == 0 { &rope_win } else { &rope_cmp };
        attn.set_rope(il, &rope.cs, rope.half)?;
        let hcw = hc_weights(ar, il)?;
        hc.register(il, "attn", hcw.0, hcw.1, hcw.2)?;
        hc.register(il, "ffn", hcw.3, hcw.4, hcw.5)?;
        let gate_w = gate_weight(ar, il)?;
        let gate_b = gate_bias(ar, il);
        moe.set_gate_f32(&gate_w, Some(&gate_b), None)?;
        let shared = shared_w(ar, il)?;
        moe.set_shared_f32(&shared.0, &shared.1, &shared.2)?;
        // ── 어텐션 서브블록 (hc 접착 = core 함수) ──
        let mut y = Vec::with_capacity(t * d);
        let mut posts = Vec::with_capacity(t);
        let mut combs = Vec::with_capacity(t);
        for i in 0..t {
            let (yi, post, comb) = hc_pre(
                &x[i * hc_n * d..(i + 1) * hc_n * d], d, hc_n,
                &hcw.0, cfg.rms_eps, cfg.hc_eps, cfg.hc_sinkhorn_iters,
            );
            y.extend_from_slice(&yi);
            posts.push(post);
            combs.push(comb);
        }
        let mut xn = Vec::with_capacity(t * d);
        for i in 0..t {
            xn.extend_from_slice(&rms_norm_weighted(
                &y[i * d..(i + 1) * d], &hcw.6, cfg.rms_eps,
            ));
        }
        let a = attn.attention_forward(il, &xn)?;
        let mut x2 = Vec::with_capacity(t * hc_n * d);
        for i in 0..t {
            x2.extend_from_slice(&hc_post(
                &a[i * d..(i + 1) * d],
                &x[i * hc_n * d..(i + 1) * hc_n * d],
                &posts[i], &combs[i], d, hc_n,
            ));
        }
        // ── FFN(MoE) 서브블록 ──
        let mut y2 = Vec::with_capacity(t * d);
        let mut posts2 = Vec::with_capacity(t);
        let mut combs2 = Vec::with_capacity(t);
        for i in 0..t {
            let (yi, post, comb) = hc_pre(
                &x2[i * hc_n * d..(i + 1) * hc_n * d], d, hc_n,
                &hcw.3, cfg.rms_eps, cfg.hc_eps, cfg.hc_sinkhorn_iters,
            );
            y2.extend_from_slice(&yi);
            posts2.push(post);
            combs2.push(comb);
        }
        let mut xn2 = Vec::with_capacity(t * d);
        for i in 0..t {
            xn2.extend_from_slice(&rms_norm_weighted(
                &y2[i * d..(i + 1) * d], &hcw.7, cfg.rms_eps,
            ));
        }
        let xs: Vec<Vec<f32>> = (0..t)
            .map(|i| xn2[i * d..(i + 1) * d].to_vec())
            .collect();
        let f = if il < cfg.n_hash_layers {
            let tids: Vec<u32> = tokens.to_vec();
            moe.moe_ffn_hash(&xs, &tids)?
        } else {
            let routes = moe.moe_route_routed(&xs)?;
            let mut seen = std::collections::HashSet::new();
            for r in &routes {
                for (e, _) in r {
                    if seen.insert(*e as usize) {
                        let w = expert_w(ar, il, *e as usize)?;
                        moe.add_expert_f32(*e as usize, &w.0, &w.1, &w.2)?;
                    }
                }
            }
            moe.moe_ffn_routed(&xs)?
        };
        for i in 0..t {
            x2[i * hc_n * d..(i + 1) * hc_n * d].copy_from_slice(
                &hc_post(
                    &f[i * d..(i + 1) * d],
                    &x2[i * hc_n * d..(i + 1) * hc_n * d],
                    &posts2[i], &combs2[i], d, hc_n,
                ),
            );
        }
        // 다음 층을 위해 VRAM 해제(같은 키 재등록으로 교체 free — D1 모듈 계약).
        let nxt = if il + 1 < cfg.n_layers { il + 1 } else { il };
        let lw2 = load_layer(ar, &attn_dims_of(attn), nxt)?;
        attn.register_layer(nxt, &lw2)?;
        x = x2;
        if x.iter().any(|v| !v.is_finite()) {
            return Err(format!("ds4-chain: 층 {il} 비유한 출력"));
        }
    }
    // ── 종단: hc_head → norm → head(스트립 gemv) ──
    let hcx: Vec<Vec<f32>> = (0..t)
        .map(|i| x[i * hc_n * d..(i + 1) * hc_n * d].to_vec())
        .collect();
    let hmix = hc.hc_head(&hcx)?;
    let head_w = head_linear(ar)?;
    let mut logits_last = Vec::new();
    for i in 0..t {
        let last = rms_norm_weighted(&hmix[i * d..(i + 1) * d], &final_norm(ar)?, cfg.rms_eps);
        if i + 1 == t {
            logits_last = head_gemv(&head_w, &last);
        }
    }
    Ok(logits_last)
}

/// core LayerAttn → D1 모듈 Ds4LayerF32 변환(같은 아카이브 텐서의 f32 사본 —
/// D1 프로브 load_layer와 동일 출처, 비트 동일 보장).
fn to_layer_f32(la: &llm170_core::deepseek4::stages::attn::LayerAttn) -> Ds4LayerF32 {
    use llm170_core::deepseek4::stages::attn as core_attn;
    let comp = la.comp.as_ref().map(|c| crate::rawcuda::ds4_attn_cuda::Ds4CompF32 {
        wkv: c.wkv.clone(),
        wgate: c.wgate.clone(),
        ape: c.ape.clone(),
        norm: c.norm.clone(),
        head_dim: c.head_dim,
        ratio: c.ratio,
        rotate: c.rotate,
    });
    let idx = la.idx.as_ref().map(|iw| crate::rawcuda::ds4_attn_cuda::Ds4IndexerF32 {
        wq_b: iw.wq_b.clone(),
        weights_proj: iw.weights_proj.clone(),
        comp: crate::rawcuda::ds4_attn_cuda::Ds4CompF32 {
            wkv: iw.comp.wkv.clone(),
            wgate: iw.comp.wgate.clone(),
            ape: iw.comp.ape.clone(),
            norm: iw.comp.norm.clone(),
            head_dim: iw.comp.head_dim,
            ratio: iw.comp.ratio,
            rotate: iw.comp.rotate,
        },
    });
    let _ = core_attn::LayerKind::Swa; // 타입 앵커
    Ds4LayerF32 {
        wq_a: la.w.wq_a.clone(),
        q_norm: la.w.q_norm.clone(),
        wq_b: la.w.wq_b.clone(),
        wkv: la.w.wkv.clone(),
        kv_norm: la.w.kv_norm.clone(),
        sink: la.w.sink.clone(),
        wo_a: la.w.wo_a.clone(),
        wo_b: la.w.wo_b.clone(),
        comp,
        idx,
    }
}

/// GPU 체인 1스텝 — core block_forward 접착을 core 함수로 재현하되
/// attention/MoE만 착지 CUDA 모듈로. 층 가중치는 loader.block(il)로
/// 지연 적재, 사용 후 drop_layer/drop_rope로 해제(VRAM 상한).
fn gpu_chain_step(
    loader: &Ds4Loader,
    cfg: &Deepseek4Config,
    tokens: &[u32],
    attn: &mut Ds4AttnCuda,
    hc: &mut Ds4HcCuda,
    moe: &mut Ds4MoeCuda,
    perm: Option<&[usize]>,
) -> Result<Vec<f32>, String> {
    let (d, hc_n, t) = (cfg.dim, cfg.hc_mult, tokens.len());
    let rope_win = llm170_core::deepseek4::ops::RopeTable::build(
        cfg.rope_head_dim, t, cfg.rope_theta, false,
        cfg.yarn_factor, cfg.yarn_orig_len, cfg.yarn_beta_fast, cfg.yarn_beta_slow,
    );
    let rope_cmp = llm170_core::deepseek4::ops::RopeTable::build(
        cfg.rope_head_dim, t, cfg.compress_rope_theta, true,
        cfg.yarn_factor, cfg.yarn_orig_len, cfg.yarn_beta_fast, cfg.yarn_beta_slow,
    );
    let rows = loader.embed_rows(tokens)?;
    let mut x = vec![0.0f32; t * hc_n * d];
    for (i, row) in rows.iter().enumerate() {
        for j in 0..hc_n {
            x[i * hc_n * d + j * d..(i + 1) * hc_n * d].copy_from_slice(row);
        }
    }
    let order: Vec<usize> = match perm {
        Some(p) => p.to_vec(),
        None => (0..cfg.n_layers).collect(),
    };
    for &il in &order {
        let bw = loader.block(il)?;
        attn.register_layer(il, &to_layer_f32(&bw.attn))?;
        let rope = if cfg.ratio(il) == 0 { &rope_win } else { &rope_cmp };
        attn.set_rope(il, &rope.cs, rope.half)?;
        hc.register(il, "attn", &bw.hc_attn.fns, &bw.hc_attn.base, &bw.hc_attn.scale)?;
        hc.register(il, "ffn", &bw.hc_ffn.fns, &bw.hc_ffn.base, &bw.hc_ffn.scale)?;
        moe.set_gate_f32(
            &bw.gate.weight,
            bw.gate.bias.as_deref(),
            if bw.is_hash { bw.gate.tid2eid.as_deref() } else { None },
        )?;
        moe.set_shared_f32(&bw.shared.w1, &bw.shared.w2, &bw.shared.w3)?;
        // ── 어텐션 서브블록(hc 접착 = core 함수) ──
        let mut y = Vec::with_capacity(t * d);
        let mut posts = Vec::with_capacity(t);
        let mut combs = Vec::with_capacity(t);
        for i in 0..t {
            let (yi, post, comb) = hc_pre(
                &x[i * hc_n * d..(i + 1) * hc_n * d], d, hc_n,
                &bw.hc_attn, cfg.rms_eps, cfg.hc_eps, cfg.hc_sinkhorn_iters,
            );
            y.extend_from_slice(&yi);
            posts.push(post);
            combs.push(comb);
        }
        let mut xn = Vec::with_capacity(t * d);
        for i in 0..t {
            xn.extend_from_slice(&rms_norm_weighted(
                &y[i * d..(i + 1) * d], &bw.attn_norm, cfg.rms_eps,
            ));
        }
        let a = attn.attention_forward(il, &xn)?;
        let mut x2 = Vec::with_capacity(t * hc_n * d);
        for i in 0..t {
            x2.extend_from_slice(&hc_post(
                &a[i * d..(i + 1) * d],
                &x[i * hc_n * d..(i + 1) * hc_n * d],
                &posts[i], &combs[i], d, hc_n,
            ));
        }
        // ── FFN(MoE) 서브블록 ──
        let mut y2 = Vec::with_capacity(t * d);
        let mut posts2 = Vec::with_capacity(t);
        let mut combs2 = Vec::with_capacity(t);
        for i in 0..t {
            let (yi, post, comb) = hc_pre(
                &x2[i * hc_n * d..(i + 1) * hc_n * d], d, hc_n,
                &bw.hc_ffn, cfg.rms_eps, cfg.hc_eps, cfg.hc_sinkhorn_iters,
            );
            y2.extend_from_slice(&yi);
            posts2.push(post);
            combs2.push(comb);
        }
        let mut xn2 = Vec::with_capacity(t * d);
        for i in 0..t {
            xn2.extend_from_slice(&rms_norm_weighted(
                &y2[i * d..(i + 1) * d], &bw.ffn_norm, cfg.rms_eps,
            ));
        }
        let xs: Vec<Vec<f32>> = (0..t).map(|i| xn2[i * d..(i + 1) * d].to_vec()).collect();
        let f = if bw.is_hash {
            moe.moe_ffn_hash(&xs, tokens)?
        } else {
            let routes = moe.moe_route_routed(&xs)?;
            let mut added = std::collections::HashSet::new();
            for r in &routes {
                for (e, _) in r {
                    if added.insert(*e as usize) {
                        let w = loader.expert(il, *e as usize)?;
                        moe.add_expert_f32(*e as usize, &w.w1, &w.w2, &w.w3)?;
                    }
                }
            }
            moe.moe_ffn_routed(&xs)?
        };
        for i in 0..t {
            x2[i * hc_n * d..(i + 1) * hc_n * d].copy_from_slice(&hc_post(
                &f[i * d..(i + 1) * d],
                &x2[i * hc_n * d..(i + 1) * hc_n * d],
                &posts2[i], &combs2[i], d, hc_n,
            ));
        }
        // VRAM 해제 — 사용이 끝난 층 즉시(다음 층 적재 전).
        attn.drop_layer(il)?;
        x = x2;
        if x.iter().any(|v| !v.is_finite()) {
            return Err(format!("ds4-chain: 층 {il} 비유한 출력"));
        }
    }
    // ── 종단: hc_head(GPU) → norm(core) → head(호스트 스트립 gemv) ──
    let hcx: Vec<Vec<f32>> = (0..t)
        .map(|i| x[i * hc_n * d..(i + 1) * hc_n * d].to_vec())
        .collect();
    let hmix_rows = hc.hc_head(&hcx)?;
    let head = loader.head_linear()?;
    let final_norm = loader.final_norm()?;
    let mut logits_last = Vec::new();
    for i in 0..t {
        let last = rms_norm_weighted(&hmix_rows[i * d..(i + 1) * d], &final_norm, cfg.rms_eps);
        if i + 1 == t {
            logits_last = head_gemv_stripwise(&head, &last);
        }
    }
    Ok(logits_last)
}

/// ds4-chain — 43L greedy 체인(스텝 0 실측 토큰 + 스텝 1-2 greedy) vs
/// core frame 오라클 + 층 순서 뒤집기 네거티브(인라인, 원장 17호).
/// VRAM: 층별 사용 후 즉시 해제(attn drop_layer) — 12GB 상한 계약.
pub fn cuda_ds4_chain_check(dir: &str) -> Result<String, String> {
    let dir = std::path::PathBuf::from(dir);
    let loader = Ds4Loader::open(&dir).map_err(|e| format!("ds4-chain: 로더 {e}"))?;
    let cfg = &loader.cfg;
    let dev = "NVIDIA GeForce RTX 4070 SUPER";
    let mut fails: Vec<String> = Vec::new();
    let mut report = String::new();
    let mut attn = Ds4AttnCuda::new(Ds4AttnDims::from_config(&config_json(&dir)?)?)?;
    let mut hc = Ds4HcCuda::new(llm170_core::deepseek4::stages::hc::hc_dims_from_config(
        &config_json(&dir)?,
    )?)?;
    let mut moe = Ds4MoeCuda::new(crate::rawcuda::ds4_moe_cuda::Ds4MoeDims::from_config(
        &config_json(&dir)?,
    )?)?;
    let oracle = Ds4Frame::new(&loader);
    // 스텝 0: 실측 토큰 8개.
    let mut tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let gates = DS4_CHAIN_LOGITS_THRESH;
    for step in 0..3usize {
        let t0 = std::time::Instant::now();
        let m = gpu_chain_step(&loader, cfg, &tokens, &mut attn, &mut hc, &mut moe, None)?;
        let o = oracle.forward(&tokens).map_err(|e| format!("오라클: {e}"))?;
        eprintln!("ds4-chain 스텝 {step}: {}초", t0.elapsed().as_secs_f64());
        let md = maxdiff(&m, &o.logits_last);
        let tok_m = argmax(&m);
        let tok_o = argmax(&o.logits_last);
        let pass = md <= gates && tok_m == tok_o;
        println!(
            "device: {dev} | ds4-chain step{step}: T={} logits maxdiff={md:.4e} (gate {gates:.1e}) tok mod={tok_m} or={tok_o} | {}",
            tokens.len(),
            if pass { "MATCH" } else { "MISMATCH" }
        );
        report.push_str(&format!(" · step{step} {md:.3e} tok {tok_m}/{tok_o}"));
        if !pass {
            fails.push(format!("step{step} maxdiff={md:.3e} tok {tok_m}!={tok_o}"));
        }
        // greedy 다음 토큰(모듈 argmax — 체인 토큰 동일성 계약).
        if step + 1 < 3 {
            tokens.push(tok_m);
        }
    }
    // ── 네거티브(인라인 — 원장 17호): 층 순서 뒤집기 [1,0,2..] 1패스. ──
    let mut perm: Vec<usize> = (0..cfg.n_layers).collect();
    perm.swap(0, 1);
    let m_perm = gpu_chain_step(&loader, cfg, &[1, 2, 3, 4, 5, 6, 7, 8], &mut attn, &mut hc, &mut moe, Some(&perm))?;
    let o0 = oracle.forward(&[1, 2, 3, 4, 5, 6, 7, 8]).map_err(|e| format!("오라클: {e}"))?;
    let md_perm = maxdiff(&m_perm, &o0.logits_last);
    let det = md_perm > DS4_CHAIN_NEG_THRESH;
    println!(
        "device: {dev} | ds4-chain neg: layer-perm logits maxdiff={md_perm:.3e} (thresh {DS4_CHAIN_NEG_THRESH:.1e}) | {}",
        if det { "DETECTED" } else { "MISSED" }
    );
    report.push_str(&format!(" · neg perm {md_perm:.2e}"));
    if !det {
        fails.push(format!("neg layer-perm maxdiff={md_perm:.3e} ≤ {DS4_CHAIN_NEG_THRESH:.0e} — 음성 미탐지"));
    }
    if fails.is_empty() {
        Ok(format!("device: {dev} | ds4-chain PASS{report}"))
    } else {
        Err(format!("ds4-chain 실패 — {} | {report}", fails.join(", ")))
    }
}

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "길이 불일치");
    a.iter()
        .zip(b.iter())
        .fold(0.0f32, |m, (&x, &y)| m.max((x - y).abs()))
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// 픽스처 디렉터의 config.json 본문(모듈 from_config 입력).
fn config_json(dir: &std::path::Path) -> Result<String, String> {
    std::fs::read_to_string(dir.join("config.json"))
        .map_err(|e| format!("config.json 읽기: {e}"))
}

// ── 가중치 접근(호스트) — Ds4Loader 위임 ──

fn head_gemv(head: &llm170_exl3::Exl3Linear, x: &[f32]) -> Vec<f32> {
    llm170_core::deepseek4::frame::head_gemv_stripwise(head, x)
}

fn final_norm(loader: &Ds4Loader) -> Result<Vec<f32>, String> {
    loader.final_norm().map_err(|e| e.to_string())
}

fn hc_weights(
    loader: &Ds4Loader,
    il: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), String> {
    let bw = loader.block(il).map_err(|e| e.to_string())?;
    Ok((
        bw.hc_attn.fns.clone(),
        bw.hc_attn.base.clone(),
        bw.hc_attn.scale.clone(),
        bw.hc_ffn.fns.clone(),
        bw.hc_ffn.base.clone(),
        bw.hc_ffn.scale.clone(),
        bw.attn_norm.clone(),
        bw.ffn_norm.clone(),
    ))
}

fn gate_weight(loader: &Ds4Loader, il: usize) -> Result<Vec<f32>, String> {
    Ok(loader.block(il).map_err(|e| e.to_string())?.gate.weight)
}

fn gate_bias(loader: &Ds4Loader, il: usize) -> Vec<f32> {
    loader
        .block(il)
        .ok()
        .and_then(|bw| bw.gate.bias)
        .unwrap_or_default()
}

fn shared_w(
    loader: &Ds4Loader,
    il: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    let bw = loader.block(il).map_err(|e| e.to_string())?;
    Ok((bw.shared.w1, bw.shared.w2, bw.shared.w3))
}

fn expert_w(
    loader: &Ds4Loader,
    il: usize,
    e: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    let w = loader.expert(il, e).map_err(|e| e.to_string())?;
    Ok((w.w1, w.w2, w.w3))
}

fn head_linear(loader: &Ds4Loader) -> Result<llm170_exl3::Exl3Linear, String> {
    loader.head_linear().map_err(|e| e.to_string())
}

fn final_norm_of(loader: &Ds4Loader) -> Result<Vec<f32>, String> {
    loader.final_norm().map_err(|e| e.to_string())
}

fn attn_dims_of(attn: &Ds4AttnCuda) -> Ds4AttnDims {
    attn.dims.clone()
}
