//! gpu_engine.rs에서 분해(순수 이동 — 코드 무변).

/// [gemma4 P2-3 2026-10-14] gemma4 v0 체인(호스트 스테이징) 호스트 설정 —
/// 서버 프로브가 층 산술(노름·잔차·layer_scalar)을 조립하는 데 필요한 것 전부.
/// 행 순서·계약 출처 = CPU 오라클(`crates/core/src/gemma4/`).
pub(crate) struct G4HostCfg {
    pub softcap: f32,
    /// 노름 행 8L+1 × hidden — [input, post_attn, pre_ff, post_ff1, pre_ff2,
    /// post_ff2, post_ff, router(=moe_scale·h^-0.5)] × 층 + output_norm.
    pub nw: Vec<f32>,
    /// 층별 layer_scalar(출력 곱).
    pub layer_scalar: Vec<f32>,
    /// [L][n_experts] per-expert 사전 스케일(μ_e).
    pub pes: Vec<f32>,
    /// QK 노름 [L][d_max](층별 d까지 유효 — 나머지 0 패딩).
    pub qnw: Vec<f32>,
    pub knw: Vec<f32>,
    pub d_max: usize,
}

impl G4HostCfg {
    pub fn nw_row(&self, il: usize, k: usize, hidden: usize) -> &[f32] {
        &self.nw[(il * 8 + k) * hidden..(il * 8 + k + 1) * hidden]
    }

    pub fn out_norm_row(&self, n_layer: usize, hidden: usize) -> &[f32] {
        &self.nw[n_layer * 8 * hidden..(n_layer * 8 + 1) * hidden]
    }

    pub fn qk_rows(&self, il: usize, d: usize) -> (&[f32], &[f32]) {
        let b = il * self.d_max;
        (&self.qnw[b..b + d], &self.knw[b..b + d])
    }

    pub fn pes_row(&self, il: usize, n_exp: usize) -> &[f32] {
        &self.pes[il * n_exp..(il + 1) * n_exp]
    }
}

/// [gemma4 P2-3 2026-10-14] gemma4 상주 업로드 — qwen35 `upload_model` 미러.
/// ① 양자화 선형(층별 q/k/(v)/o + dense gate/up/down + 라우터) ② 노름 8행/층
/// (라우터 입력 행 = moe_scale·h^-0.5 — CPU route의 가중 RMS와 동일 산식)
/// ③ QK 노름(d_max 패딩) + AttnLayer 테이블(층별 d/kv/sliding/k_eq_v)
/// ④ MoE(무 shared·gelu 전문가·g32 테이블) ⑤ head(원본 embed — 행별 int8)
/// ⑥ 호스트 설정 반환(노름 사본·rope·layer_scalar·pes).
pub(crate) fn upload_model_g4(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::gemma4::Model,
    ctx: usize,
) -> Result<G4HostCfg, String> {
    use llm170_core::wtype::WType;
    let hp = model.hp.clone();
    let h = hp.n_embd;
    let t0 = std::time::Instant::now();
    // ① 선형 — 밀집(양자화) + 라우터(플레인/양자화 무관 분기).
    // [W4 2026-10-16] O_DIRECT 직결 스트림(핀드 링 — mmap 슬라이스 우회).
    let mut ld = super::upload::DirectLoad::new(model.store());
    for il in 0..hp.n_layer {
        for suf in [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
            "moe_gate.weight",
        ] {
            let name = format!("blk.{il}.{suf}");
            let Some(w) = model.w(&name) else {
                continue; // k_eq_v 층의 v_proj 부재
            };
            let (tn, sn) = model
                .w_names(&name)
                .ok_or_else(|| format!("{name}: 원시 텐서 이름 부재"))?;
            match w.ty {
                WType::W4a16Split => {
                    let sn = sn.ok_or_else(|| format!("{name}: scale 이름 부재"))?;
                    let (ql, sl) = (ld.tensor_len(&tn)? as usize, ld.tensor_len(&sn)? as usize);
                    let mut f = |which: usize, off: u64, buf: &mut [u8]| {
                        let n = if which == 0 { tn.as_str() } else { sn.as_str() };
                        ld.read_into(n, off, buf)
                    };
                    dec.upload_lin_g_stream(
                        &name,
                        w.n_out as usize,
                        w.n_in as usize,
                        w.group,
                        w.scale_bf16,
                        ql,
                        sl,
                        &mut f,
                    )?;
                }
                WType::Bf16 => {
                    let len = ld.tensor_len(&tn)? as usize;
                    let mut f = |off: u64, buf: &mut [u8]| ld.read_into(&tn, off, buf);
                    dec.upload_plain_stream(&name, w.n_out as usize, w.n_in as usize, len, &mut f)?;
                }
                _ => {}
            }
        }
    }
    // ② 노름 nw — 8행/층 + 최종 1행. 라우터 행 = moe_scale·h^-0.5(가중 RMS).
    let inv_h = 1.0f32 / (h as f32).sqrt();
    let mut nw: Vec<f32> = Vec::with_capacity((8 * hp.n_layer + 1) * h);
    for il in 0..hp.n_layer {
        for suf in [
            "attn_norm.weight",
            "post_attn_norm.weight",
            "pre_ff_norm.weight",
            "post_ff_norm1.weight",
            "pre_ff_norm2.weight",
            "post_ff_norm2.weight",
            "post_ff_norm.weight",
        ] {
            let v = model
                .f32_vec(&format!("blk.{il}.{suf}"))
                .map_err(|e| e.to_string())?;
            if v.len() != h {
                return Err(format!("g4 노름 형상 위반 blk.{il}.{suf}: {}", v.len()));
            }
            nw.extend(v);
        }
        let sc = model
            .f32_vec(&format!("blk.{il}.moe_scale"))
            .map_err(|e| e.to_string())?;
        if sc.len() != h {
            return Err(format!("g4 moe_scale 형상 위반 L{il}: {}", sc.len()));
        }
        nw.extend(sc.iter().map(|v| v * inv_h));
    }
    nw.extend(
        model
            .f32_vec("output_norm.weight")
            .map_err(|e| e.to_string())?,
    );
    dec.set_norm_weights(&nw, 8 * hp.n_layer + 1)?;
    // ③ QK 노름(d_max 패딩) + 어텐션 층 테이블(KV 아레나 = 층별 최대 클래스).
    let d_max = hp.head_dim.iter().copied().max().unwrap_or(0);
    let mut qnw = vec![0f32; hp.n_layer * d_max];
    let mut knw = vec![0f32; hp.n_layer * d_max];
    for il in 0..hp.n_layer {
        let d = hp.head_dim[il];
        let q = model
            .f32_vec(&format!("blk.{il}.attn_q_norm.weight"))
            .map_err(|e| e.to_string())?;
        let k = model
            .f32_vec(&format!("blk.{il}.attn_k_norm.weight"))
            .map_err(|e| e.to_string())?;
        if q.len() != d || k.len() != d {
            return Err(format!(
                "g4 QK 노름 형상 L{il}: q={} k={} d={d}",
                q.len(),
                k.len()
            ));
        }
        qnw[il * d_max..il * d_max + d].copy_from_slice(&q);
        knw[il * d_max..il * d_max + d].copy_from_slice(&k);
    }
    let layers: Vec<llm170_backend_gpu::AttnLayer> = (0..hp.n_layer)
        .map(|il| llm170_backend_gpu::AttnLayer {
            kv_heads: hp.n_kv[il],
            d: hp.head_dim[il],
            sliding: !hp.full[il],
            k_eq_v: model.kv_eq_v(il),
            kq_scale: 1.0, // gemma4 = 1/√d 없음(노름이 스케일 흡수)
        })
        .collect();
    dec.set_attn(
        llm170_backend_gpu::AttnDims {
            n_attn: hp.n_layer,
            q_heads: hp.n_head,
            kv_heads: hp.n_kv[0],
            d: hp.head_dim[0],
            cap: ctx,
            interval: 1,
            sliding_window: hp.sw,
        },
        layers,
        &qnw,
        &knw,
    )?;
    let attn_ms = t0.elapsed().as_secs_f64() * 1e3;
    // ④ MoE — 전문가 테이블(무 shared·gelu GLU) + 상주/스트리밍 판정.
    let t1 = std::time::Instant::now();
    if hp.n_experts > 0 {
        let (group, scale_bf16) = model.expert_quant();
        dec.set_moe(
            hp.n_experts,
            hp.top_k,
            hp.moe_ffn,
            0,
            group,
            scale_bf16,
            true,
        );
        // 스트리밍 MoE 스테이징(dchain) — gemma4 체인엔 qwen의 ensure_chain_bufs
        // 경로가 없어 여기서 보장(비상주 시 moe_experts_streaming이 사용).
        dec.ensure_moe_stage_chain()?;
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
        let expert_bytes: u64 = tab.iter().map(|e| e.1 + e.3).sum();
        let free = llm170_backend_gpu::cuda_mem_free()
            .map(|(f, _)| f)
            .unwrap_or(0);
        let resident = match crate::engine::MOE_POLICY.load(std::sync::atomic::Ordering::Relaxed) {
            1 => true,
            2 => false,
            _ => match llm170_diag::flag::val("LLM170_MOE_RESIDENT") {
                Some("0") => false,
                Some("1") => true,
                // qwen upload_moe와 달리 KV 차감 없음 — 여기는 set_attn(KV 실할당)이
                // 판정보다 먼저라 free가 이미 KV 풀을 제외한 값이다. kv_reserve를
                // 또 빼면 이중차감(상주→스트리밍 오판)이라 고의로 더하지 않는다.
                _ => free > expert_bytes + (1 << 30),
            },
        };
        if resident {
            // [W4 2026-10-16] 전문가 상주 — O_DIRECT 직결 스트림(셀 단위).
            let mut lens: Vec<(usize, usize)> = Vec::with_capacity(tab.len());
            let mut names: Vec<(String, String)> = Vec::with_capacity(tab.len());
            for il in 0..hp.n_layer {
                for e in 0..hp.n_experts {
                    for proj in ["gate_proj", "up_proj", "down_proj"] {
                        let (qn, sn) = model.expert_names(il, e, proj);
                        lens.push((ld.tensor_len(&qn)? as usize, ld.tensor_len(&sn)? as usize));
                        names.push((qn, sn));
                    }
                }
            }
            let mut f = |cell: usize, which: usize, off: u64, buf: &mut [u8]| {
                let n = if which == 0 {
                    &names[cell].0
                } else {
                    &names[cell].1
                };
                ld.read_into(n, off, buf)
            };
            dec.upload_experts_resident_streamed(&lens, &mut f)?;
        } else {
            dec.set_expert_table(tab);
        }
        eprintln!(
            "[g4-moe] 전문가 {:.2}GiB · 여유 {:.2}GiB → {} (g{group}{})",
            expert_bytes as f64 / (1u64 << 30) as f64,
            free as f64 / (1u64 << 30) as f64,
            if resident {
                "전량 상주"
            } else {
                "호스트 스트리밍"
            },
            if scale_bf16 { "/bf16" } else { "" }
        );
    }
    let moe_ms = t1.elapsed().as_secs_f64() * 1e3;
    // ⑤ head — 원본 embed(tied) 행별 int8 W8A16(단일 head 계약).
    // [W4] O_DIRECT 직결 스트림(행-블록 양자화 — 호스트 RAM 상수).
    let t2 = std::time::Instant::now();
    {
        let hw = model.wchk("output.weight").map_err(|e| e.to_string())?;
        let (tn, _) = model
            .w_names("output.weight")
            .ok_or_else(|| "output.weight: 원시 텐서 이름 부재".to_string())?;
        let len = ld.tensor_len(&tn)? as usize;
        let mut f = |off: u64, buf: &mut [u8]| ld.read_into(&tn, off, buf);
        dec.upload_head_stream(hw.n_out as usize, hw.n_in as usize, len, &mut f)?;
    }
    ld.log_summary();
    let head_ms = t2.elapsed().as_secs_f64() * 1e3;
    // ⑥ 호스트 설정(rope·scalar·pes — 층 산술 조립용) + rope 테이블 등록.
    let mut layer_scalar = Vec::with_capacity(hp.n_layer);
    let mut pes: Vec<f32> = Vec::with_capacity(hp.n_layer * hp.n_experts);
    let mut rp: Vec<f32> = Vec::with_capacity(hp.n_layer * 7);
    for il in 0..hp.n_layer {
        let ls = model
            .f32_vec(&format!("blk.{il}.layer_scalar"))
            .map_err(|e| e.to_string())?;
        layer_scalar.push(
            *ls.first()
                .ok_or_else(|| format!("layer_scalar 부재 L{il}"))?,
        );
        pes.extend(
            model
                .f32_vec(&format!("blk.{il}.moe_per_expert_scale"))
                .map_err(|e| e.to_string())?,
        );
        // 커널 rp 규약: {d, pairs, span, denom, base, vn_on, k_eq_v}.
        let d = hp.head_dim[il];
        rp.extend([
            d as f32,
            (hp.n_rot[il] / 2) as f32,
            (d / 2) as f32,
            d as f32,
            hp.rope_base[il],
            1.0, // v_norm(무가중) — gemma4 전 층
            if model.kv_eq_v(il) { 1.0 } else { 0.0 },
        ]);
    }
    dec.set_g4_rope(&rp)?;
    // 층 스칼라·pes 등록은 여기서 단일화 — 디바이스 청크(엔진·eval·프로브
    // 공용)가 로드 시점 계약을 갖는다(set_g4_layer_scalar 누락 = 청크 실패).
    dec.set_g4_layer_scalar(&layer_scalar)?;
    eprintln!("[g4] 업로드 — 선형·노름 {attn_ms:.0}ms · MoE {moe_ms:.0}ms · head {head_ms:.0}ms");
    Ok(G4HostCfg {
        softcap: hp.softcap,
        nw,
        layer_scalar,
        pes,
        qnw,
        knw,
        d_max,
    })
}
