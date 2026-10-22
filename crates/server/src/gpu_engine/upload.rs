use llm170_backend_gpu::{AttnDims, GdnDims};
use llm170_core::w4a16::{DirectReader, W4a16Model};

pub(crate) struct DirectLoad<'m> {
    store: &'m W4a16Model,
    paths: Vec<&'m std::path::Path>,
    readers: Vec<Option<DirectReader>>,
    pub chunks: u64,
    pub bytes: u64,
    logged: bool,
}

impl<'m> DirectLoad<'m> {
    pub(crate) fn new(store: &'m W4a16Model) -> Self {
        Self {
            store,
            paths: store.shard_paths(),
            readers: (0..store.shard_count()).map(|_| None).collect(),
            chunks: 0,
            bytes: 0,
            logged: false,
        }
    }

    fn reader(&mut self, shard: usize) -> Result<&mut DirectReader, String> {
        if self.readers[shard].is_none() {
            let p = *self
                .paths
                .get(shard)
                .ok_or_else(|| format!("샤드 {shard} 경로 부재"))?;
            self.readers[shard] = Some(DirectReader::open(p).map_err(|e| e.to_string())?);
        }
        Ok(self.readers[shard].as_mut().expect("직전에 채움"))
    }

    pub(crate) fn tensor_len(&self, name: &str) -> Result<u64, String> {
        self.store
            .tensor_range(name)
            .map(|(_, _, len)| len)
            .ok_or_else(|| format!("적재: 텐서 범위 부재 {name}"))
    }

    pub(crate) fn read_into(&mut self, name: &str, off: u64, buf: &mut [u8]) -> Result<(), String> {
        let (shard, base, len) = self
            .store
            .tensor_range(name)
            .ok_or_else(|| format!("적재: 텐서 범위 부재 {name}"))?;
        if off + buf.len() as u64 > len {
            return Err(format!(
                "적재: 범위 밖 {name} off={off} len={} 텐서={len}",
                buf.len()
            ));
        }
        self.chunks += 1;
        self.bytes += buf.len() as u64;
        let r = self.reader(shard)?;
        r.read_into(base + off, buf).map_err(|e| e.to_string())
    }

    pub(crate) fn log_summary(&mut self) {
        if self.logged {
            return;
        }
        self.logged = true;
        if self.chunks == 0 {
            return;
        }
        let align = self.readers.iter().flatten().map(|r| r.align()).max();
        let direct = self.readers.iter().flatten().all(|r| r.direct);
        eprintln!(
            "# load: {} 텐서청크 {}·{:.2}GiB{}",
            if direct {
                "O_DIRECT 직결(mmap 우회)"
            } else {
                "버퍼 읽기 폴백(O_DIRECT 미지원 FS)"
            },
            self.chunks,
            self.bytes as f64 / (1u64 << 30) as f64,
            align.map(|a| format!(" · align {a}B")).unwrap_or_default(),
        );
    }
}

pub(crate) fn upload_moe(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::qwen35::Model,
    kv_reserve: u64,
) -> Result<(), String> {
    let hp = model.hp.clone();
    let t0 = std::time::Instant::now();
    let mut ld = DirectLoad::new(model.store());
    for name in model.engine_names() {
        if name == "token_embd.weight" || name == "output.weight" {
            continue;
        }
        if let Some(w) = model.w_raw(&name)
            && w.ty == llm170_core::wtype::WType::Bf16
        {
            let (tn, _) = model
                .w_raw_names(&name)
                .ok_or_else(|| format!("{name}: 원시 텐서 이름 부재"))?;
            let len = ld.tensor_len(&tn)? as usize;
            let mut f = |off: u64, buf: &mut [u8]| ld.read_into(&tn, off, buf);
            dec.upload_plain_stream(&name, w.n_out as usize, w.n_in as usize, len, &mut f)?;
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
    let expert_bytes: u64 = tab.iter().map(|e| e.1 + e.3).sum();
    let free = dec.free_vram().map(|(f, _)| f).unwrap_or(0);
    let margin = 1u64 << 30;
    let qsz = hp.n_embd * hp.moe_ffn / 2;
    let ssz = hp.n_embd * hp.moe_ffn / group.max(32) * 2;
    let per_slot = 3 * (qsz + ssz);
    let per_total = (hp.n_layer * per_slot).max(1) as u64;
    let free_ps = free.saturating_sub(kv_reserve);
    let k_cache = 96usize.min((free_ps / 2 / per_total) as usize);
    let cache_bytes = k_cache as u64 * per_total;
    let budget = free
        .saturating_sub(kv_reserve)
        .saturating_sub(margin)
        .saturating_sub(cache_bytes);
    let n_cells = (hp.n_layer * hp.n_experts) as u64;
    let per_expert = expert_bytes.checked_div(n_cells).unwrap_or(0);
    let hot_fit = budget
        .checked_div(per_expert)
        .and_then(|v| v.checked_div(hp.n_layer as u64))
        .map_or(0, |v| v.min(hp.n_experts as u64) as usize);
    enum MoePlace {
        Resident,
        Hybrid(usize),
        Streaming,
    }
    let place = match crate::engine::MOE_POLICY.load(std::sync::atomic::Ordering::Relaxed) {
        1 => MoePlace::Resident,
        2 => MoePlace::Streaming,
        3 => {
            if hot_fit >= hp.n_experts {
                MoePlace::Resident
            } else if hot_fit > 0 && k_cache > 0 {
                MoePlace::Hybrid(hot_fit)
            } else {
                eprintln!("# moe: hybrid 강제 — VRAM 예산 없음(핫 0) → 스트리밍");
                MoePlace::Streaming
            }
        }
        _ => match llm170_diag::flag::val("LLM170_MOE_RESIDENT") {
            Some("0") => MoePlace::Streaming,
            Some("1") => MoePlace::Resident,
            _ => {
                if free > expert_bytes + kv_reserve + margin {
                    MoePlace::Resident
                } else if hot_fit > 0 && k_cache > 0 {
                    MoePlace::Hybrid(hot_fit)
                } else {
                    MoePlace::Streaming
                }
            }
        },
    };
    let tab_ms = t1.elapsed().as_secs_f64() * 1e3;
    let t2 = std::time::Instant::now();
    match place {
        MoePlace::Resident => {
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
        }
        MoePlace::Hybrid(hot) => {
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
            dec.upload_experts_hybrid_streamed(&lens, &mut f, hot, k_cache)?;
            dec.set_expert_table_hybrid(tab, hot);
            let hot_bytes = per_expert * hot as u64 * hp.n_layer as u64;
            eprintln!(
                "# moe: 하이브리드 — 핫 {hot}/{}전문가/층 VRAM 직행({:.2}GiB, O_DIRECT) · 비상주 {:.2}GiB만 RAM 스테이징 · 캐시 {k_cache}슬롯/층 예약",
                hp.n_experts,
                hot_bytes as f64 / (1u64 << 30) as f64,
                (expert_bytes - hot_bytes) as f64 / (1u64 << 30) as f64,
            );
        }
        MoePlace::Streaming => {
            dec.set_expert_table(tab);
        }
    }
    ld.log_summary();
    eprintln!(
        "[moe] 플레인 {plain_ms:.0}ms · 테이블 {tab_ms:.0}ms · 전문가 {:.0}ms ({:.1}GB/s) — 전문가 {:.2}GiB · 여유 {:.2}GiB − KV 예약 {:.2}GiB → {}",
        t2.elapsed().as_secs_f64() * 1e3,
        (expert_bytes as f64 * 1e-9) / t2.elapsed().as_secs_f64().max(1e-9),
        expert_bytes as f64 / (1u64 << 30) as f64,
        free as f64 / (1u64 << 30) as f64,
        kv_reserve as f64 / (1u64 << 30) as f64,
        match place {
            MoePlace::Resident => "전량 상주".to_string(),
            MoePlace::Hybrid(h) => format!("하이브리드 핫 {h}/{}층당", hp.n_experts),
            MoePlace::Streaming => "호스트 스트리밍".to_string(),
        }
    );
    Ok(())
}

pub(crate) fn upload_model(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::qwen35::Model,
    ctx: usize,
    kv_reserve: u64,
    window: Option<(usize, usize)>,
) -> Result<(), String> {
    let hp = model.hp.clone();
    let interval = hp.full_attn_interval.max(1);
    if interval != 4 {
        return Err(format!(
            "디코더는 4층 주기(full_attention_interval) 전용 — interval={interval}"
        ));
    }
    let moe = hp.n_experts > 0;
    if moe {
        if window.is_some() {
            return Err("PP 샤딩: MoE 모델은 1차 미지원(dense 27B 전용)".into());
        }
        upload_moe(dec, model, kv_reserve)?;
    } else {
        let mut ld = DirectLoad::new(model.store());
        for name in model.engine_names() {
            if let Some((ws, we)) = window
                && let Some(il) = blk_layer_of(&name)
                && !(ws..we).contains(&il)
            {
                continue;
            }
            if let Some(w) = model.w_raw(&name)
                && w.ty == llm170_core::wtype::WType::W4a16Split
            {
                let (qn, sn) = model
                    .w_raw_names(&name)
                    .ok_or_else(|| format!("{name}: 원시 텐서 이름 부재"))?;
                let sn = sn.ok_or_else(|| format!("{name}: scale 이름 부재"))?;
                let (ql, sl) = (ld.tensor_len(&qn)? as usize, ld.tensor_len(&sn)? as usize);
                let mut f = |which: usize, off: u64, buf: &mut [u8]| {
                    let n = if which == 0 { qn.as_str() } else { sn.as_str() };
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
        }
        ld.log_summary();
    }
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
    let mtp_attn = model
        .w_raw("mtp.layers.0.self_attn.q_proj.weight")
        .is_some();
    let owns_final = window.is_none_or(|(_, we)| we == hp.n_layer);
    dec.set_mtp_attn_planned(
        owns_final && mtp_attn && crate::engine::mtp_upload_on(window.is_some()),
    );
    let n_attn = hp.n_layer / interval + usize::from(mtp_attn);
    let ad = AttnDims {
        n_attn,
        q_heads: hp.n_head,
        kv_heads: hp.n_kv,
        d: hp.head_dim,
        cap: ctx,
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

fn blk_layer_of(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("blk.")?;
    let digits = rest.split('.').next()?;
    digits.parse().ok()
}

#[cfg(test)]
mod pp_tests {
    use super::blk_layer_of;

    #[test]
    fn blk_layer_of_names() {
        assert_eq!(blk_layer_of("blk.0.attn_q.weight"), Some(0));
        assert_eq!(blk_layer_of("blk.31.ffn_down.weight"), Some(31));
        assert_eq!(blk_layer_of("token_embd.weight"), None);
        assert_eq!(blk_layer_of("output.weight"), None);
        assert_eq!(blk_layer_of("mtp.layers.0.self_attn.q_proj.weight"), None);
        assert_eq!(blk_layer_of("blk.12x.weight"), None);
    }
}
