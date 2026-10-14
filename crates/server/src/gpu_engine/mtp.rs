//! gpu_engine.rs에서 분해(순수 이동 — 코드 무변).
use super::QwenEngine;

impl QwenEngine {
    /// [MTP-CHAIN §3.3 2026-10-12] 초안 소스 정책 — MTP 헤드 가용 + 강제
    /// n-gram 아님. LLM170_SPEC_DRAFT_SOURCE: "mtp"(기본/미설정 — MTP 가용
    /// 시 MTP, 스테일·부재 시 n-gram 폴백) | "ngram"(강제 n-gram — MTP
    /// 상태 구축[프리필 KV]·생산[mtp_first/single/post_round] 전면 스킵,
    /// 종전 파킹 경로와 동일). 초안은 검증이 정본이라 이 게이트는 무손실
    /// (운영 정책 — 수치 경로 분기 아님, 단일화 방침 준수).
    pub fn mtp_drafts_on(&self) -> bool {
        // [2.J] CLI 정책 우선 — 1=mtp 강제(가용 시) · 2=ngram 강제 · 0=auto(env 진단).
        match crate::engine::SPEC_POLICY.load(std::sync::atomic::Ordering::Relaxed) {
            1 => self.mtp,
            2 => false,
            _ => self.mtp && llm170_diag::flag::val("LLM170_SPEC_DRAFT_SOURCE") != Some("ngram"),
        }
    }

    /// [MTP] 라운드 전 — 검증 체인 dres의 수용 접두 acc행을 hidden 어셈블리로
    /// (롤백·보정 디코드 전에 호출해야 dres가 살아있다).
    pub fn mtp_pre_round(&mut self, acc: usize) -> Result<(), String> {
        self.dec.mtp_reset_hidden();
        for r in 0..acc {
            self.dec.mtp_push_hidden(r)?;
        }
        Ok(())
    }

    /// [MTP-2] 초안 폭 — 총 초안 수(첫 초안 + 재귀). 검증 t = 1+k ≤ 8(TR GEMV).
    const MTP_K: usize = 4;

    /// [MTP-CHAIN §3.3 ② 2026-10-12] 앵커 직후 k-1 재귀 스텝을 장치 체인
    /// (그래프 리플레이 — 호스트 왕복 0)으로 치환해 초안 [d_1..d_k] 반환.
    /// t>1 앵커는 진입 계약 시드(mtp_chain_seed — 토큰 레지스터·dpp)가
    /// 선행, t=1은 멱등 재기입. 체인 실패(계약·캡처 — 리플레이 前 판정)는
    /// Err → 호출부가 순차 재귀(mtp_recur_step, 스텝당 sync+회수 왕복)로
    /// 폴백. 순차·체인 두 경로의 소비 입력(ptr5 행0·d_1·위치)은 동일 —
    /// 토큰열 판정 동일(모듈 계약 승계).
    fn mtp_chain_round(&mut self, slot: usize, d1: u32, base_pos: u32) -> Result<Vec<u32>, String> {
        self.dec.mtp_chain_seed(slot, d1, base_pos)?;
        self.dec.mtp_chain_drafts(slot, Self::MTP_K, base_pos)
    }

    /// [MTP] 라운드 후 — 보정 행 hidden 추가 + t행 순전파 + **MTP-2 재귀**.
    /// pos_start = slot_pos - (수용+1) — 수용 초안·보정 토큰이 그 위치들이다.
    /// 첫 초안(draft_0)은 (P, P+1)→P+2 규칙 그대로, 이후 k-1개는 직전 fin을
    /// hidden으로 하는 재귀(각 런이 자기 위치 KV를 갱신).
    pub fn mtp_post_round(
        &mut self,
        slot: usize,
        accepted: &[u32],
        corrected: u32,
    ) -> Result<(), String> {
        if !self.dec.mtp_slot_enabled(slot) {
            return Ok(());
        }
        // 보정행 hidden = 액티브 경로가 전진시킨 현재 dres 행 0.
        // (pre_round(acc+1) 융합은 실측 기각 — sched 주석 참조.)
        self.dec.mtp_push_hidden(0)?;
        let mut toks: Vec<u32> = accepted.to_vec();
        toks.push(corrected);
        let mut embeds: Vec<f32> = Vec::with_capacity(toks.len() * self.model.hp.n_embd);
        for &tk in &toks {
            embeds.extend_from_slice(&self.model.embed_row(tk).map_err(|e| e.to_string())?);
        }
        let t = toks.len() as u32;
        let pos_start = self.dec.slot_pos_of(slot).saturating_sub(t);
        let draft = self.dec.mtp_run(slot, &toks, &embeds, pos_start)?;
        // [2.A-③ 2026-10-13] 재귀 시드 = 앵커 마지막 행(t-1)의 fin — 체인/순차
        // 재귀의 "ptr5 행 0 복사" 계약에 맞춰 정규화한다(종전 t≥2에서 행 0을
        // 읽어 d_2+가 스테일 조건에서 생성되던 결함). t=1은 무연산.
        self.dec.mtp_fin_seed(t as usize - 1)?;
        // [MTP-CHAIN §3.3] 재귀 k-1스텝 = 체인 1회(호스트 왕복 0 — 종전
        // 4콜 ≈10.4ms/라운드 → 그래프 리플레이). base_pos = 앵커 마지막 행
        // 위치(pos_start+t-1): 리플레이의 dpp+1 → 첫 스텝 pos_start+t =
        // 순차 경로와 동일 위치. 실패(계약·캡처) 시 순차 재귀 폴백.
        let base_pos = pos_start + t - 1;
        let drafts = match self.mtp_chain_round(slot, draft, base_pos) {
            Ok(ds) => ds,
            Err(err) => {
                eprintln!("# mtp-chain(폴백 순차): {err}");
                let mut ds = vec![draft];
                let mut tok = draft;
                for i in 0..(Self::MTP_K.saturating_sub(1)) {
                    let emb = self.model.embed_row(tok).map_err(|e| e.to_string())?;
                    let d = self
                        .dec
                        .mtp_recur_step(slot, tok, &emb, pos_start + t + i as u32)?;
                    ds.push(d);
                    tok = d;
                }
                ds
            }
        };
        // lim=pos+t+1 슬랙이 읽는 pos+1 행의 이전 라운드 잔존분 제거.
        if let Err(err) = self
            .dec
            .mtp_zero_row(self.dec.slot_pos_of(slot) as usize + 1)
        {
            eprintln!("# mtp(무시): {err}");
        }
        self.dec.mtp_store_drafts(slot, &drafts);
        Ok(())
    }

    /// [MTP] 비-스펙 단독 디코드 후 — 리셋 + 1행 순전파 + 초안 보관.
    /// (슬롯 off면 no-op — 접두 재사용 슬롯.)
    pub fn mtp_single(&mut self, slot: usize, tok: u32) -> Result<(), String> {
        if !self.dec.mtp_slot_enabled(slot) {
            return Ok(());
        }
        self.mtp_pre_round(0)?;
        self.mtp_post_round(slot, &[], tok)
    }

    /// [MTP] 프리필 직후 첫 초안 — 유예 호출 (prev hidden, 첫 토큰) → 초안.
    pub fn mtp_first(&mut self, slot: usize, tok: u32) -> Result<(), String> {
        if !self.dec.mtp_slot_enabled(slot) {
            return Ok(());
        }
        let emb = self.model.embed_row(tok).map_err(|e| e.to_string())?;
        let d = self.dec.mtp_run_prev(slot, tok, &emb)?;
        // [MTP-CHAIN §3.3] 유예 앵커(t=1) 위치 = 저장 시점 slot_pos-1
        // (프리필 완료 직후 호출 계약 — 디코드가 slot_pos를 전진시키기 전).
        // base = 그 위치 → 리플레이 dpp+1 = 첫 스텝 slot_pos(순차 base+0과
        // 동일). 실패 시 순차 재귀 폴백.
        let base_pos = self.dec.slot_pos_of(slot).saturating_sub(1);
        let drafts = match self.mtp_chain_round(slot, d, base_pos) {
            Ok(ds) => ds,
            Err(err) => {
                eprintln!("# mtp-chain(폴백 순차): {err}");
                let mut ds = vec![d];
                let mut tk = d;
                for i in 0..(Self::MTP_K.saturating_sub(1)) {
                    let eb = self.model.embed_row(tk).map_err(|e| e.to_string())?;
                    let nd = self
                        .dec
                        .mtp_recur_step(slot, tk, &eb, base_pos + 1 + i as u32)?;
                    ds.push(nd);
                    tk = nd;
                }
                ds
            }
        };
        self.dec.mtp_store_drafts(slot, &drafts);
        Ok(())
    }

    /// [MTP] 슬롯 재사용(접두 캐시) — KV 정합 없음: 초안 비활성/재개.
    pub fn mtp_slot_reused(&mut self, slot: usize) {
        self.dec.mtp_slot_off(slot);
    }

    pub fn mtp_slot_fresh(&mut self, slot: usize) {
        self.dec.mtp_slot_on(slot);
    }

    /// [MTP] 슬롯 KV 초기화(새 시퀀스 — 잔존 행 제거).
    pub fn mtp_slot_kv_reset(&mut self) -> Result<(), String> {
        self.dec.mtp_kv_reset()
    }

    /// [MTP] 초안 소비(현 위치 유효분만).
    pub fn mtp_take_draft(&mut self, slot: usize) -> Option<Vec<u32>> {
        self.dec.mtp_take_draft(slot)
    }
}

/// [MTP 2026-10-12] MTP 블록 업로드 — 27B dense 한정(v1). 35B MoE MTP는
/// 전문가 스택(3D bf16)이라 차기. 성공 시 true.
pub(crate) fn upload_mtp(
    dec: &mut llm170_backend_gpu::W4a16Dec,
    model: &llm170_core::qwen35::Model,
) -> Result<bool, String> {
    // [MTP 2026-10-12 — 파킹(정확성 수정 완료)] 정확성 결함 3건을 모두
    // 수정해 **수용률 80~82% 실측**(초안=정답 일치). 수정 내역:
    //  (1) 꼬리(8~11단계: o_proj→post_attention_layernorm→MLP→mtp.norm)
    //      미구현 — head가 어텐션 前 x를 소비하던 것(CPU 대조 rel≈1e-7 일치).
    //  (2) norm_resid의 ab 판독이 t행 전체인데 zero 버퍼가 1행(h)이라 t≥2
    //      에서 OOB(행1+ NaN) → 프리필 K/V 오염 → 전 어텐션 NaN.
    //  (3) bind의 노름 매칭 ends_with("norm.weight")가 pre_fc_norm_hidden/
    //      embedding을 놓쳐 +1 없이 raw 사용(그 두 텐서만 zero-centered).
    //  (4) hidden 소스 = pre-norm 잔차(dres) — vLLM speculator.py 298
    //      "(logits_hidden, feedback_hidden) for final-norm correctness".
    // 파킹 사유: **1-초안 스펙이 순손실** — MTP 콜당 런치 오버헤드 ~5.5ms
    // (연산 0.2ms, ~20런치×250µs) + 검증/보정 고정비 → 서빙 38 t/s
    // (평범 디코드 ~50+). **MTP-2(k=4) 구현·실측 완료**: 라운드당 2.62토큰
    // (부분수용 71%)이나 콜 4회 오버헤드(+18ms)로 67.7ms/라운드 = 38.7 t/s.
    // **[MTP-GRAPH·구간 실측 2026-10-12 — 최종]** t=1..8 캡처 인프라 완비
    // (핀 임베드·argmax 핀 회수·mtp_fwd_launch 분리·invalidate take·dx32/16
    // 사전 보장). 라운드 구간 실측(27B 프로즈, k=4): **save 0.00ms · 검증
    // t=5 37.1ms · 보정 t=1 19.8ms · MTP 4콜 10.4ms · 합 67.4ms**. 즉
    // (a) GDN 스냅샷은 d2d async라 사실상 무료(추정 15ms는 오판),
    // (b) 검증 t=5가 t=1(17.9)의 **2.07×**로 지배, (c) 그래프 후 MTP 콜은
    // 2.6ms/call이나 cuGraphLaunch 자체가 느린 기기라 총합 이득 ~0.
    // k=7(검증 t=8 풀블록 가설) 실측: 검증 61.7ms로 오히려 악화(블록 완충
    // 가설 기각), 4.82s/128토큰 = 26.6 t/s. k=4 = 3.4s(37.6 t/s) — 평범
    // (~50) 대비 여전히 순손실. **정직한 결론: 27B는 가중치 바운드라
    // 검증(t=k+1)+보정(t=1) 두 체인 고정비가 스펙 수익을 잠식 — 수익화는
    // ①검증 경로 marlin급(가중치 스트리밍 절반) ②검증+보정 체인 통합
    // ③MTP-2 단일 그래프 디바이스 체인(호스트 왕복 제거)이 선행돼야 한다.
    // **[보정 융합 채택 2026-10-12]** 스펙 라운드의 보정 토큰 = 검증 행 acc
    // argmax(toks[acc]) 재사용 — decode_greedy 중복 제거. n-gram 스펙 서빙
    // **2.6s→2.12s/128토큰(49→60 t/s, +22%)** · 라운드 53.2→32.0ms ·
    // 스펙 골든 불변(비트 계약 유지). MTP 경로만 디코드 유지(보정 hidden
    // 상태 전진이 초안 체인 입력 — 융합 시 0% 실측). MTP-on(k=4)은 3.07s
    // (41.7 t/s)로 여전히 평범 미달 — 파킹 유지(MTP_ENABLED=false).
    // **[MTP-CHAIN §3.3 재활성 2026-10-12]** 파킹 사유의 하나였던 재귀
    // 4콜 호스트 왕복(≈10.4ms/라운드)을 장치 체인(mtp_chain_drafts —
    // 그래프 리플레이 k-1회, 중간 왕복 0)으로 치환해 재활성. 초안 소스
    // 정책은 sched 계층(LLM170_SPEC_DRAFT_SOURCE="ngram" 강제 시 n-gram
    // 전용, 기본 MTP 가용 시 MTP — 35B MoE MTP는 여기서 early-return
    // false라 자동 n-gram). 순차 재귀는 체인 실패(계약·캡처) 폴백으로
    // 유지. 순손실 재발 시 이 자리에서 재파킹(실측 수치 갱신).
    const MTP_ENABLED: bool = true;
    if !MTP_ENABLED || model.w_raw("mtp.fc.weight").is_none() {
        return Ok(false);
    }
    // [D1 검수 수정 2026-10-13] dense MTP 마커 필수 — 종전
    // `experts.gate_up_proj` 검사는 bind가 None(비 .weight 이름)이라 w_raw가
    // 볼 수 없어 **무력**했고(35B MoE MTP가 dense 경로로 새어
    // `mtp.layers.0.mlp.gate_proj.weight 부재`로 로드 실패), 이 자리가
    // MoE MTP의 스킵 지점이다(자동 n-gram 폴백). MoE MTP 지원은 차기.
    if model.w_raw("mtp.layers.0.mlp.gate_proj.weight").is_none() {
        eprintln!("# mtp 스킵: dense MTP 아님(MoE MTP는 차기) — n-gram 폴백");
        return Ok(false);
    }
    let mut q_out = 0usize;
    for name in [
        "mtp.fc.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.v_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.mlp.gate_proj.weight",
        "mtp.layers.0.mlp.up_proj.weight",
        "mtp.layers.0.mlp.down_proj.weight",
    ] {
        let w = model
            .w_raw(name)
            .ok_or_else(|| format!("mtp: {name} 부재"))?;
        if name.ends_with("attn.q_proj.weight") {
            q_out = w.n_out as usize;
        }
        dec.upload_plain(name, w.data, w.n_out as usize, w.n_in as usize)?;
    }
    let mut norms: Vec<f32> = Vec::new();
    for name in [
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.norm.weight",
    ] {
        norms.extend(model.f32_vec(name).map_err(|e| e.to_string())?);
    }
    let hp = &model.hp;
    let ffn = model
        .w_raw("mtp.layers.0.mlp.gate_proj.weight")
        .map(|w| w.n_out as usize)
        .unwrap_or(0);
    dec.set_mtp(&norms, q_out, hp.n_kv * hp.head_dim, ffn)?;
    let _ = hp;
    Ok(true)
}
