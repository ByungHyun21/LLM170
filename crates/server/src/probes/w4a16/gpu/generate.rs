//! probes/w4a16/gpu.rs에서 분해(순수 이동 — 코드 무변).

/// 5) 프롬프트 순차 prefill + greedy 생성(head는 CPU 참조 경로) + [A-1]
///    스페큘러티브 디코딩 — [R9] 순수 이동(gpu_run 후반부).
#[allow(clippy::too_many_arguments)]
pub(super) fn run_generate(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::qwen35::Model,
    prompt: &[u32],
    no_head: bool,
    n_predict: usize,
    spec_k: usize,
    spec_check: bool,
    upload_ms: f64,
    dir: &str,
    n_lin: usize,
    head_ready: bool,
) -> Result<String, String> {
    let hp = model.hp.clone();
    let head = model
        .w("output.weight")
        .ok_or_else(|| "output.weight 부재".to_string())?;
    let head_logits = |xn: &[f32]| -> Vec<f32> {
        if no_head {
            // head 비용 분리 계측 전용 — 체인 시간만 재기 위한 가짜 로짓.
            return xn[..5120].to_vec();
        }
        let mut lg = vec![0.0f32; head.n_out as usize];
        llm170_core::matmul::matmul(xn, &head, &mut lg);
        lg
    };
    // GPU head — bf16 output.weight를 상주 업로드(--no-head는 CPU 참조).
    // [검수 통합 2026-10-14] head_ready(다중 시나리오 2회차+)면 재업로드 생략.
    // [W4 2026-10-16] O_DIRECT 직결 스트림(mmap 우회 — 프로브 로드도 캐시 잔류 0).
    let mut head_gpu = head_ready;
    if !no_head && !head_ready && head.ty == llm170_core::wtype::WType::Bf16 {
        let mut ld = crate::gpu_engine::DirectLoad::new(model.store());
        let (tn, _) = model
            .w_raw_names("output.weight")
            .ok_or_else(|| "output.weight: 원시 텐서 이름 부재".to_string())?;
        let len = ld.tensor_len(&tn)? as usize;
        let mut f = |off: u64, buf: &mut [u8]| ld.read_into(&tn, off, buf);
        dec.upload_head_stream(head.n_out as usize, head.n_in as usize, len, &mut f)?;
        ld.log_summary();
        head_gpu = true;
    }
    let t1 = std::time::Instant::now();
    let mut out: Vec<u32> = Vec::new();
    let mut next = 0u32;
    // 프롬프트는 t≤8 배치 청크로 — 골든 판정이 배치 경로(GEMM t≥2)를 지난다.
    {
        let h = hp.n_embd;
        let mut i = 0usize;
        while i < prompt.len() {
            // [2026-10-09] MoE 상주는 32(기본), 스트리밍은 1. [P10] 상한 128 —
            // LLM170_PREFILL_T로 실측 오버라이드(엔진 prefill과 동일 규칙).
            let cap = if hp.n_experts > 0 && !dec.moe_experts_resident() {
                1
            } else {
                llm170_diag::flag::val("LLM170_PREFILL_T")
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(512) // 청크 확대
                    .clamp(1, 512)
            };
            let t = (prompt.len() - i).min(cap);
            let mut rows: Vec<f32> = Vec::with_capacity(t * h);
            for &tok in &prompt[i..i + t] {
                rows.extend_from_slice(&model.embed_row(tok).map_err(|e| e.to_string())?);
            }
            let last = i + t == prompt.len();
            if head_gpu {
                let lg = dec.forward_prefill(0, &rows, t, last)?;
                if last {
                    next = llm170_core::matmul::greedy_from(&lg);
                }
            } else {
                let xn = dec.forward_prefill(0, &rows, t, false)?;
                if last {
                    next = llm170_core::matmul::greedy_from(&head_logits(&xn));
                }
            }
            i += t;
        }
    }
    out.push(next);
    if llm170_diag::flag::ne0("LLM170_TIME") {
        eprintln!("{}", dec.prof_report("prefill")?);
    }
    // [A-1] 스페큘러티브 디코딩 — n-gram 초안 + 배치 검증 + GDN 롤백.
    // 출력 토큰 스트림은 비스펙(그리디)과 **정확히 동일**해야 한다(골든 판정).
    let mut hist: Vec<u32> = prompt.to_vec();
    if spec_k >= 2 && head_gpu {
        dec.enable_spec()?;
    }
    if spec_check && head_gpu {
        // [A-1 진단] 검증(t=4 배치) vs t=1 디코드의 토큰 대조.
        // S0 저장 → t=1 4회(T1..T4) → S0 복원+pos 되감기 → 검증 [next,T1,T2,T3].
        dec.enable_spec()?;
        dec.spec_save_state(0)?;
        let mut t1_4: Vec<u32> = Vec::new();
        {
            let mut cur = next;
            for _ in 0..4 {
                let row = model.embed_row(cur).map_err(|e| e.to_string())?;
                cur = dec.forward_device_argmax(0, &row)?;
                t1_4.push(cur);
            }
        }
        // trio(4×t=1)의 최종 상태 저장 — spec scan 대조의 정답 참조.
        let s_trio = dec.spec_dump_state(0)?;
        dec.spec_load_state(0)?;
        dec.spec_rewind_pos(0, 4)?;
        let mut rows: Vec<f32> = Vec::with_capacity(4 * hp.n_embd);
        rows.extend_from_slice(&model.embed_row(next).map_err(|e| e.to_string())?);
        for &tk in &t1_4[..3] {
            rows.extend_from_slice(&model.embed_row(tk).map_err(|e| e.to_string())?);
        }
        let toks = dec.spec_verify(0, &rows, 4)?;
        eprintln!("[spec-check] t1={t1_4:?} verify={toks:?}");
        // o_lc 대조 — spec scan on vs off(같은 입력·같은 상태).
        let nv = 4 * 32 * 128; // h_v=32 고정 가정(35B)
        let a = dec.spec_dump_outv(nv)?;
        dec.spec_load_state(0)?;
        dec.spec_rewind_pos(0, 4)?;
        dec.spec_set_scan(false);
        let _ = dec.spec_verify(0, &rows, 4)?;
        let b = dec.spec_dump_outv(nv)?;
        dec.spec_set_scan(true);
        let mut mx = 0.0f32;
        let mut mi = 0usize;
        for i in 0..nv {
            let d = (a[i] - b[i]).abs();
            if d > mx {
                mx = d;
                mi = i;
            }
        }
        eprintln!(
            "[spec-check] o_lc maxdiff(on-off) = {mx:.6e} @i={mi}(t={} h={} d={}) on={:.6e} off={:.6e}",
            mi / (32 * 128),
            (mi / 128) % 32,
            mi % 128,
            a[mi],
            b[mi]
        );
        for t in 0..4 {
            let mut tm = 0.0f32;
            for i in t * 32 * 128..(t + 1) * 32 * 128 {
                tm = tm.max((a[i] - b[i]).abs());
            }
            eprintln!("[spec-check] t={t} maxdiff={tm:.6e}");
        }
        // [핵심 대조] spec scan(t=4) 최종 상태 vs trio(4×t=1) 최종 상태.
        // + 정상 scan(WY) 경로도 같은 비교 — 공식 차이 규모 판별.
        dec.spec_load_state(0)?;
        dec.spec_rewind_pos(0, 4)?;
        dec.spec_set_scan(false);
        let _ = dec.spec_verify(0, &rows, 4)?;
        let s_wy = dec.spec_dump_state(0)?;
        dec.spec_set_scan(true);
        dec.spec_load_state(0)?;
        dec.spec_rewind_pos(0, 4)?;
        let _ = dec.spec_verify(0, &rows, 4)?;
        let mut wmx = 0.0f32;
        for i in 0..s_trio.len().min(s_wy.len()) {
            wmx = wmx.max((s_trio[i] - s_wy[i]).abs());
        }
        eprintln!("[spec-check] state maxdiff(trio-WY) = {wmx:.6e}");
        let s_spec = dec.spec_dump_state(0)?;
        let mut smx = 0.0f32;
        let mut smi = 0usize;
        for i in 0..s_trio.len().min(s_spec.len()) {
            let d = (s_trio[i] - s_spec[i]).abs();
            if d > smx {
                smx = d;
                smi = i;
            }
        }
        eprintln!(
            "[spec-check] state maxdiff(trio-spec) = {smx:.6e} @i={smi} (층 {}) trio={:.6e} spec={:.6e}",
            smi / (32 * 128 * 128),
            s_trio[smi],
            s_spec[smi]
        );
        // [TreeWY 2.B 2026-10-12] WY 재구성 롤백 대조 — verify(4) 후
        // rollback(k) 상태 ≈ 순차 t=1 디코드 k회 상태(k=1..4 전 접두).
        // 참조 = S0 복원 → t=1 디코드마다 덤프(순차 스캔이 정답). 기대
        // 오차 ~1e-7급(PoC eval 클래스 — 골든 토큰열 판정은 H 섹션이 담당).
        let mut refs: Vec<Vec<f32>> = Vec::new();
        {
            dec.spec_load_state(0)?;
            dec.spec_rewind_pos(0, 4)?;
            let mut cur = next;
            for _ in 0..4 {
                let row = model.embed_row(cur).map_err(|e| e.to_string())?;
                cur = dec.forward_device_argmax(0, &row)?;
                refs.push(dec.spec_dump_state(0)?);
            }
        }
        let mut wy_mx = 0.0f32;
        let dm_hv = dec.spec_hv().max(1);
        for k in 1..=4usize {
            dec.spec_load_state(0)?;
            dec.spec_rewind_pos(0, 4)?;
            let _ = dec.spec_verify(0, &rows, 4)?;
            dec.spec_rollback(0, k)?;
            let s = dec.spec_dump_state(0)?;
            let mut mx = 0.0f32;
            let mut mi = 0usize;
            for i in 0..s.len().min(refs[k - 1].len()) {
                let dv = (s[i] - refs[k - 1][i]).abs();
                if dv > mx {
                    mx = dv;
                    mi = i;
                }
            }
            eprintln!(
                "[spec-check] WY rollback k={k} state maxdiff(seq-wy) = {mx:.6e} @i={mi} (층 {} 헤드{} 행{} 열{}) wy={:.6e} seq={:.6e}",
                mi / (dm_hv * 128 * 128),
                (mi / (128 * 128)) % dm_hv,
                (mi / 128) % 128,
                mi % 128,
                s[mi],
                refs[k - 1][mi]
            );
            wy_mx = wy_mx.max(mx);
        }
        eprintln!("[spec-check] WY rollback max(k=1..4) = {wy_mx:.6e}");
        // [TreeWY 진단] **같은 verify(4) 라운드 내** 대조 — 순차 스캔 최종
        // 상태(S_4) vs 같은 라운드의 wy 데이터로 재구성한 S_4. 입력·pos·
        // 참조 경로 교란 원천이 없는 순수 커널 대조(합성 단위 테스트
        // wy_reconstruct_matches_sequential의 실모델 버전 — 기대 ~1e-6).
        {
            dec.spec_load_state(0)?;
            dec.spec_rewind_pos(0, 4)?;
            let _ = dec.spec_verify(0, &rows, 4)?;
            let s_batch4 = dec.spec_dump_state(0)?;
            dec.spec_rollback(0, 4)?;
            let s_wy4b = dec.spec_dump_state(0)?;
            let mut mx = 0.0f32;
            let mut mi = 0usize;
            for i in 0..s_batch4.len().min(s_wy4b.len()) {
                let dv = (s_batch4[i] - s_wy4b[i]).abs();
                if dv > mx {
                    mx = dv;
                    mi = i;
                }
            }
            eprintln!(
                "[spec-check] same-round WY(4) vs scan-final maxdiff = {mx:.6e} @i={mi} wy={:.6e} scan={:.6e}",
                s_wy4b[mi], s_batch4[mi]
            );
        }
        dec.spec_load_state(0)?;
        dec.spec_rewind_pos(0, 4)?;
    }
    let (mut spec_rounds, mut spec_drafted, mut spec_acc) = (0u64, 0u64, 0u64);
    let spec_t0 = std::time::Instant::now();
    // 원 루프(for _ in 0..n_predict) = n_predict회 **추가** — out은 프리필
    // 첫 토큰을 이미 담고 있다(오프바이원 실측: 35B 4000 골든 8토큰).
    while out.len() <= n_predict {
        if spec_k >= 2 && head_gpu {
            // 초안 = 최대 spec_k-1개(검증 배치 = 1+초안 ≤ spec_k).
            // n=4 — 12는 우리 산문 테스트에서 발화 0(실측). 짧은 패턴이
            // 반복 구조(리스트·코드)를 더 자주 잡는다(수용률은 검증이 판정).
            let draft = llm170_core::spec::ngram_draft(&hist, 4, 48, spec_k - 1);
            if !draft.is_empty() {
                let t = 1 + draft.len();
                let mut rows: Vec<f32> = Vec::with_capacity(t * hp.n_embd);
                rows.extend_from_slice(&model.embed_row(next).map_err(|e| e.to_string())?);
                for &tk in &draft {
                    rows.extend_from_slice(&model.embed_row(tk).map_err(|e| e.to_string())?);
                }
                // [dsave 대체 2026-10-14] 라운드 전 저장 폐지 — 롤백 재료는
                // WY solve + conv 링 스냅샷(프로덕션 sched와 동일 계약).
                let toks = dec.spec_verify(0, &rows, t)?;
                let mut acc = 0usize;
                while acc < draft.len() && toks[acc] == draft[acc] {
                    acc += 1;
                }
                spec_rounds += 1;
                spec_drafted += draft.len() as u64;
                spec_acc += acc as u64;
                // 상태·pos를 마지막 수용 토큰 **직전**으로 되감고 t=1 디코드.
                let last = if acc >= 1 { draft[acc - 1] } else { next };
                if acc >= 1 {
                    dec.spec_rollback(0, acc)?;
                    dec.spec_rewind_pos(0, (t - acc) as u32)?;
                } else {
                    dec.spec_rollback(0, 0)?;
                    dec.spec_rewind_pos(0, t as u32)?;
                }
                let lrow = model.embed_row(last).map_err(|e| e.to_string())?;
                let corr = dec.forward_device_argmax(0, &lrow)?;
                for &tk in &draft[..acc] {
                    out.push(tk);
                    hist.push(tk);
                }
                out.push(corr);
                hist.push(corr);
                next = corr;
                continue;
            }
        }
        let row = model.embed_row(next).map_err(|e| e.to_string())?;
        if head_gpu {
            // [P3] 디코드는 디바이스 argmax(4B d2h) — 토큰 스트림은 종전
            // 로짓+CPU greedy_from과 동일해야 한다(골든이 동등성 판정).
            next = dec.forward_device_argmax(0, &row)?;
        } else {
            let xn = dec.forward_device(0, &row)?;
            next = llm170_core::matmul::greedy_from(&head_logits(&xn));
        }
        out.push(next);
        hist.push(next);
    }
    if llm170_diag::flag::ne0("LLM170_TIME") {
        // 진단(P8): 커널 범주별 소요 — 그래프 캡처 중에는 마킹이 꺼지므로
        // LLM170_GRAPH=0 직접 경로에서 의미가 있다.
        eprintln!("{}", dec.prof_report(&format!("decode {n_predict}토큰"))?);
    }
    if spec_rounds > 0 {
        eprintln!(
            "[spec] 라운드 {spec_rounds} · 초안 {spec_drafted} · 수용 {spec_acc} (수용률 {:.0}%) · {:.1}ms",
            spec_acc as f64 / spec_drafted.max(1) as f64 * 100.0,
            spec_t0.elapsed().as_secs_f64() * 1e3
        );
    }
    let gen_ms = t1.elapsed().as_secs_f64() * 1e3;
    let csv = out
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join(",");
    // MoE는 플레인(비양자화) 업로드 실측 — dense는 split 선형 수 기반 추정.
    let plain_gb = if hp.n_experts > 0 {
        model
            .engine_names()
            .iter()
            .filter(|n| n.as_str() != "token_embd.weight" && n.as_str() != "output.weight")
            .filter_map(|n| model.w_raw(n))
            .filter(|w| w.ty == llm170_core::wtype::WType::Bf16)
            .map(|w| w.data.len() as f64 * 1e-9)
            .sum::<f64>()
    } else {
        n_lin as f64 * 44.6e-3
    };
    Ok(format!(
        "w4a16-gpu {dir} — GPU 체인(디바이스 상주{} · {})\n  업로드 {upload_ms:.0}ms · 생성 {}토큰 {gen_ms:.0}ms ({:.1}ms/토큰)\n tokens: {csv}",
        if head_gpu {
            " + GPU head"
        } else {
            " + CPU head"
        },
        if hp.n_experts > 0 {
            format!("MoE 플레인 {plain_gb:.1}GB + 전문가 스트리밍/상주")
        } else {
            format!("선형 {n_lin}개 · {plain_gb:.1}GB급")
        },
        out.len(),
        gen_ms / out.len() as f64,
    ))
}
