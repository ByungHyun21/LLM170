//! gpu_engine.rs에서 분해(순수 이동 — 코드 무변).
use llm170_backend_gpu::{AttnDims, GdnDims};
use llm170_core::w4a16::{DirectReader, W4a16Model};

/// [2026-10-16] 상주 적재 바이트 소스 — O_DIRECT 직접 리더(샤드별 lazy).
/// mmap 우회로 **페이지 캐시 잔류 0**(모델 > RAM도 성립) — 백엔드 핀드 링이
/// `read_into`를 fill로 소비해 VRAM으로 스트리밍한다. CPU 참조·MoE 스트리밍
/// (비상주) 소스는 호스트가 계층이라 mmap을 유지한다(캐시가 자산).
/// O_DIRECT 미지원 FS는 리더가 **자동으로 버퍼 읽기로 폴백**한다(DirectReader —
/// 경로 선택 플래그 없음: env는 진단 전용 규율, 적재는 수치 불변).
pub(crate) struct DirectLoad<'m> {
    store: &'m W4a16Model,
    paths: Vec<&'m std::path::Path>,
    readers: Vec<Option<DirectReader>>,
    /// 진단 누적 — fill 호출 수·바이트.
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

    /// 텐서 전체 길이(바이트) — 스트림 호출의 len 인자.
    pub(crate) fn tensor_len(&self, name: &str) -> Result<u64, String> {
        self.store
            .tensor_range(name)
            .map(|(_, _, len)| len)
            .ok_or_else(|| format!("적재: 텐서 범위 부재 {name}"))
    }

    /// 텐서 `[off, off+buf.len())` 채움 — 백엔드 fill 계약(전량 기록).
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

    /// 로드 종료 1회 로그 — 직결/폴백(리더 실측)·누적(운영 가시화).
    pub(crate) fn log_summary(&mut self) {
        if self.logged {
            return;
        }
        self.logged = true;
        if self.chunks == 0 {
            return;
        }
        let align = self.readers.iter().flatten().map(|r| r.align()).max();
        // 리더 실측 모드(FS 미지원이면 버퍼 폴백 — 숨기지 않는다).
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
    // [2026-10-16] 플레인 가중 — O_DIRECT 직결 스트림(mmap 슬라이스 우회).
    let mut ld = DirectLoad::new(model.store());
    for name in model.engine_names() {
        // token_embd(호스트 embed_row)·output(head 전용 업로드)는 체인이
        // 선형으로 쓰지 않는다 — VRAM 절약(각 ~970MiB @35B).
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
    // [2.J] CLI 정책 우선 — 1=resident 강제 · 2=streaming 강제 · 3=hybrid 강제 ·
    // 0=auto(env/여유). [2026-10-16 #11] hybrid = 핫 몫만 VRAM 직행(O_DIRECT),
    // 비상주 몫만 RAM 스테이징 — 부분-fit에서 전량 RAM 복사/이중 패스 제거.
    // [2.M 사다리] auto는 **KV 우선**: KV 풀 몫(kv_reserve — 서버 산출)을 떼고
    // 남는 VRAM으로만 상주 판정. KV가 ctx를 보장한 뒤 남는 잔여가 전문가 핫셋.
    // (kv_reserve=0 = 종전 규칙 — 프로브 경로.)
    let margin = 1u64 << 30;
    // [2026-10-16 #11] 캐시 우선 예약 — 하이브리드에서도 mcache가 순수 스트리밍과
    // 같은 크기(k = min(96, (여유−KV)/2/층별슬롯))를 갖도록 예약하고, 핫 몫은
    // 그 나머지로만 산정한다(프리필 distinct 가드 = k 계약 유지).
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
    // 핫 몫(층당) = 예산 ÷ (층×전문가 1개 바이트). v0 정적 배분(층 균등) —
    // 런타임 통계 시드·교체는 W6 액추에이터 축.
    let n_cells = (hp.n_layer * hp.n_experts) as u64;
    let per_expert = if n_cells > 0 {
        expert_bytes / n_cells
    } else {
        0
    };
    let hot_fit = if per_expert > 0 {
        (budget / per_expert / hp.n_layer as u64).min(hp.n_experts as u64) as usize
    } else {
        0
    };
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
            // [2026-10-16] 전문가 상주 — O_DIRECT 직결 스트림(셀 단위).
            // 셀 이름·길이는 expert_slice와 동일 규칙(bind::expert_base).
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
            // [2026-10-16 #11] 핫 몫 = O_DIRECT 직행(디스크→VRAM), 비상주 몫만
            // 호스트 RAM 스테이징(mcache 재업로드 소스).
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
            // 비상주 = 호스트 mmap 스트리밍(설계: 스트리밍 소스는 캐시가 자산).
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
        // [2026-10-16] 양자화 선형 — O_DIRECT 직결 스트림(핀드 링).
        let mut ld = DirectLoad::new(model.store());
        for name in model.engine_names() {
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
