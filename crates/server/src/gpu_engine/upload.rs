//! gpu_engine.rs에서 분해(순수 이동 — 코드 무변).
use llm170_backend_gpu::{AttnDims, GdnDims};

/// MoE(35B-A3B) 상주 배선 — 플레인 bf16 업로드 + 전문가 슬라이스 테이블.
/// GpuEngine·w4a16-gpu 프로브 공용.
///
/// SAFETY(전문가 테이블): 항목은 모델 스토어의 mmap 슬라이스 주소다 —
/// 호출자가 모델을 디코더와 함께 보유하는 수명 계약(디코더가 먼저 drop).
pub(crate) fn upload_moe(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::qwen35::Model,
    kv_reserve: u64,
) -> Result<(), String> {
    let hp = model.hp.clone();
    let t0 = std::time::Instant::now();
    for name in model.engine_names() {
        // token_embd(호스트 embed_row)·output(head 전용 업로드)는 체인이
        // 선형으로 쓰지 않는다 — VRAM 절약(각 ~970MiB @35B).
        if name == "token_embd.weight" || name == "output.weight" {
            continue;
        }
        if let Some(w) = model.w_raw(&name)
            && w.ty == llm170_core::wtype::WType::Bf16
        {
            dec.upload_plain(&name, w.data, w.n_out as usize, w.n_in as usize)?;
        }
    }
    let plain_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t1 = std::time::Instant::now();
    dec.set_plain_mode(true);
    let (group, scale_bf16) = model.expert_quant();
    dec.set_moe(
        hp.n_experts,
        hp.top_k,
        hp.moe_ffn,
        hp.shared_ffn,
        group,
        scale_bf16,
        // [gemma4 P2 2026-10-14] 전문가 GLU 활성 — qwen은 silu(기본), gemma4는
        // gelu_tanh(배선 시 true). 모델별 단일 경로(런타임 수치 플래그 아님).
        false,
    );
    let mut tab: Vec<(u64, u64, u64, u64)> = Vec::with_capacity(hp.n_layer * hp.n_experts * 3);
    for il in 0..hp.n_layer {
        for e in 0..hp.n_experts {
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                let (q, s, _, _) = model
                    .expert_slice(il, e, proj)
                    .ok_or_else(|| format!("전문가 슬라이스 부재: L{il} e{e} {proj}"))?;
                tab.push((
                    q.as_ptr() as u64,
                    q.len() as u64,
                    s.as_ptr() as u64,
                    s.len() as u64,
                ));
            }
        }
    }
    // 상주 가능(170HX 64GB 등)이면 전문가 전량 VRAM — 토큰당 PCIe 스트리밍 제거.
    // 판정: 전문가 바이트 + 1GiB 여유. 불가면 스트리밍 테이블(호스트 mmap).
    let expert_bytes: u64 = tab.iter().map(|e| e.1 + e.3).sum();
    let free = llm170_backend_gpu::cuda_mem_free()
        .map(|(f, _)| f)
        .unwrap_or(0);
    // [2.J] CLI 정책 우선 — 1=resident 강제 · 2=streaming 강제 · 0=auto(env/여유).
    // [2.M 사다리] auto는 **KV 우선**: KV 풀 몫(kv_reserve — 서버 산출)을 떼고
    // 남는 VRAM으로만 상주 판정. KV가 ctx를 보장한 뒤 남는 잔여가 전문가 핫셋.
    // (kv_reserve=0 = 종전 규칙 — 프로브 경로.)
    let resident = match crate::engine::MOE_POLICY.load(std::sync::atomic::Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => match llm170_diag::flag::val("LLM170_MOE_RESIDENT") {
            Some("0") => false,
            Some("1") => true,
            _ => free > expert_bytes + kv_reserve + (1 << 30),
        },
    };
    let tab_ms = t1.elapsed().as_secs_f64() * 1e3;
    let t2 = std::time::Instant::now();
    if resident {
        dec.upload_experts_resident(&tab)?;
    } else {
        dec.set_expert_table(tab);
    }
    eprintln!(
        "[moe] 플레인 {plain_ms:.0}ms · 테이블 {tab_ms:.0}ms · 전문가 {:.0}ms ({:.1}GB/s) — 전문가 {:.2}GiB · 여유 {:.2}GiB − KV 예약 {:.2}GiB → {}",
        t2.elapsed().as_secs_f64() * 1e3,
        (expert_bytes as f64 * 1e-9) / t2.elapsed().as_secs_f64().max(1e-9),
        expert_bytes as f64 / (1u64 << 30) as f64,
        free as f64 / (1u64 << 30) as f64,
        kv_reserve as f64 / (1u64 << 30) as f64,
        if resident {
            "전량 상주"
        } else {
            "호스트 스트리밍"
        }
    );
    Ok(())
}

/// 모델 상주 업로드(1~4단계) — GpuEngine·w4a16-gpu 프로브 공용.
/// head는 호출부 소관(프로브는 --no-head/스테이징 분기가 있다).
pub(crate) fn upload_model(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::qwen35::Model,
    ctx: usize,
    kv_reserve: u64,
) -> Result<(), String> {
    let hp = model.hp.clone();
    let interval = hp.full_attn_interval.max(1);
    if interval != 4 {
        return Err(format!(
            "디코더는 4층 주기(full_attention_interval) 전용 — interval={interval}"
        ));
    }
    let moe = hp.n_experts > 0;
    // 1) 선형 상주 — GPU 체인은 원본(HF) 무게(순열은 커널 내부 처리).
    if moe {
        upload_moe(dec, model, kv_reserve)?;
    } else {
        for name in model.engine_names() {
            if let Some(w) = model.w_raw(&name)
                && w.ty == llm170_core::wtype::WType::W4a16Split
            {
                let s = w.aux.ok_or_else(|| format!("{name}: aux 부재"))?;
                dec.upload_lin(&name, w.data, s, w.n_out as usize, w.n_in as usize)?;
            }
        }
    }
    // 2) 노름 nw [2L+1][hidden] (+1 보정 — f32_vec).
    let mut nw: Vec<f32> = Vec::new();
    for il in 0..hp.n_layer {
        nw.extend(
            model
                .f32_vec(&format!("blk.{il}.attn_norm.weight"))
                .map_err(|e| e.to_string())?,
        );
        nw.extend(
            model
                .f32_vec(&format!("blk.{il}.post_attention_norm.weight"))
                .map_err(|e| e.to_string())?,
        );
    }
    nw.extend(
        model
            .f32_vec("output_norm.weight")
            .map_err(|e| e.to_string())?,
    );
    dec.set_norm_weights(&nw, 2 * hp.n_layer + 1)?;
    // 3) GDN 상수 — 원본(HF) 순서(a 먼저 b 다음 — 커널 ab 색인).
    let n_gdn = hp.n_layer - hp.n_layer / interval;
    let gd = GdnDims {
        n_gdn,
        hidden: hp.n_embd,
        h_k: hp.n_group,
        h_v: hp.dt_rank,
        d: hp.d_state,
    };
    let (mut cw, mut ab, mut alog, mut dtb, mut gnw): (
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
    ) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for il in 0..hp.n_layer {
        if (il + 1) % interval == 0 {
            continue;
        }
        cw.extend(
            model
                .raw_f32_vec(&format!("blk.{il}.ssm_conv1d.weight"))
                .map_err(|e| e.to_string())?,
        );
        ab.extend(
            model
                .raw_f32_vec(&format!("blk.{il}.ssm_alpha.weight"))
                .map_err(|e| e.to_string())?,
        );
        ab.extend(
            model
                .raw_f32_vec(&format!("blk.{il}.ssm_beta.weight"))
                .map_err(|e| e.to_string())?,
        );
        alog.extend(
            model
                .raw_f32_vec(&format!("blk.{il}.ssm_a"))
                .map_err(|e| e.to_string())?,
        );
        dtb.extend(
            model
                .raw_f32_vec(&format!("blk.{il}.ssm_dt.bias"))
                .map_err(|e| e.to_string())?,
        );
        gnw.extend(
            model
                .raw_f32_vec(&format!("blk.{il}.ssm_norm.weight"))
                .map_err(|e| e.to_string())?,
        );
    }
    dec.set_gdn(gd, &cw, &ab, &alog, &dtb, &gnw)?;
    // 4) 어텐션 q/k 노름(+1 — f32_vec 보정).
    // [MTP 2026-10-12] MTP 존재 시 가상 전체-어텐션 층 1개 추가(n_attn+1,
    // 마지막 인덱스) — KV 1층 확장 + q/k_norm 행 추가. 커널 무수정.
    let mtp_attn = model
        .w_raw("mtp.layers.0.self_attn.q_proj.weight")
        .is_some();
    let n_attn = hp.n_layer / interval + usize::from(mtp_attn);
    let ad = AttnDims {
        n_attn,
        q_heads: hp.n_head,
        kv_heads: hp.n_kv,
        d: hp.head_dim,
        cap: ctx,
        // [A11] 모델 config에서 동적(종전 하드코딩 4) — 신규 비율 모델 대응.
        interval,
        sliding_window: 0,
    };
    let (mut qnw, mut knw): (Vec<f32>, Vec<f32>) = (Vec::new(), Vec::new());
    for ai in 0..n_attn - usize::from(mtp_attn) {
        let il = ai * interval + interval - 1;
        qnw.extend(
            model
                .f32_vec(&format!("blk.{il}.attn_q_norm.weight"))
                .map_err(|e| e.to_string())?,
        );
        knw.extend(
            model
                .f32_vec(&format!("blk.{il}.attn_k_norm.weight"))
                .map_err(|e| e.to_string())?,
        );
    }
    if mtp_attn {
        qnw.extend(
            model
                .f32_vec("mtp.layers.0.self_attn.q_norm.weight")
                .map_err(|e| e.to_string())?,
        );
        knw.extend(
            model
                .f32_vec("mtp.layers.0.self_attn.k_norm.weight")
                .map_err(|e| e.to_string())?,
        );
    }
    dec.set_attn(ad, Vec::new(), &qnw, &knw)?;
    Ok(())
}
