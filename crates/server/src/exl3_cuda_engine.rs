//! EXL3 cuda 엔진 어댑터 — Exl3CudaDecoder 다중 슬롯(rawcuda 포팅, plans/124).
//!
//! S5 배선 계약(plans/cuda-port.md): ctx 기반 KV 할당 + 순차 프리필·디코드.
//! S8: 슬롯은 디코더가 GDN 링/스캔 상태·KV 캐시·pos를 슬롯별로 보유한다
//! (가중치는 전 슬롯 공유). hip 어댑터(exl3_hip_engine.rs) 미러.
//!
//! P0-3(plans/cuda-models.md §3.2·B9, 2026-10-08): MTP 스펙 디코드 개방.
//! M4 임계 종결(mtp 프로브 ALL PASS — 근원은 미러 FMA 미모델링, 커널 무결)
//! 후 mtp_cuda(Exl3CudaMtp) 배선. 슬롯별 드래프트 상태(자체 KV)를 두고
//! spec_round는 hip 판(D2 — 스냅샷→배치 검증→수용/롤백 재처리)을 미러한다.
//! 프리필은 verify 배치로 MTP KV를 전 문맥 적립(빈 문맥 시작이 수용률
//! 붕괴 — qwen4exp P15④ 실측 계약).

use crate::exl3_engine::exl3_eos_of;
use llm170_backend_gpu::rawcuda::exl3_cuda::Exl3CudaDecoder;
use llm170_backend_gpu::rawcuda::mtp_cuda::Exl3CudaMtp;

pub struct Exl3CudaEngine {
    dec: Exl3CudaDecoder,
    /// 정지 토큰 — tokenizer_config.json eos_token_id 파생(exl3_eos_of 공용).
    pub eos: u32,
    /// P0-3: 슬롯별 MTP 드래프트 상태 — 스펙 의도(--spec k>0)일 때만 조립.
    /// 빈 벡터 = 아카이브에 mtp.* 부재(35B 계열) 또는 스펙 미요청.
    mtps: Vec<Exl3CudaMtp>,
    /// 직전 커밋 토큰의 pre-final-norm 잔차(드래프트 h 입력).
    mtp_h: Vec<Option<Vec<f32>>>,
    /// 타깃 자신의 다음 예측(미처리 토큰 — 검증 배치 행 0).
    mtp_pp: Vec<u32>,
    /// MTP KV 적립 행 수(드래프트 자체 pos 공간).
    mtp_kv: Vec<usize>,
}

// SAFETY: hip 어댑터(exl3_hip_engine.rs)와 동일 근거 — 모든 GPU 접근은
// slot_loop 단일 스레드에서 직렬 실행(디코더 버퍼 단일 소유).
unsafe impl Send for Exl3CudaEngine {}

impl Exl3CudaEngine {
    /// n_slots개 동시 시퀀스. 슬롯당 VRAM이 선형으로 증가하므로
    /// (27B·ctx 4096 기준 GDN 157MB + KV 536MB) 상한을 넘어가면
    /// cuMemAlloc이 조용히 실패하지 않고 Err로 거절한다 — 세그먼트
    /// 폴트 대신 우아한 거절이 계약이다(plans/128 P0와 동일 논리).
    pub fn load(dir: &str, n_slots: usize, ctx_len: usize) -> Result<Self, String> {
        let slots = n_slots.max(1);
        // plans/cuda-port.md S5: hip와 동일한 ctx 범위로 KV 용량을 정한다.
        let kvcap = if ctx_len == 0 {
            4096
        } else {
            ctx_len.clamp(64, 32768)
        };
        if kvcap != ctx_len {
            eprintln!("# cuda kvcap: ctx {ctx_len} → {kvcap} (범위 [64, 32768]로 클램프)");
        }
        // plans/cuda-port.md S12: 옛엔 여기서 fwd3s 점수 scratch가 공유메모리
        // 1024행이라 ctx>1024를 로드 시점에 경고했다(S9). 커널이 위치 청크
        // 온라인 소프트맥스로 바뀌면서 그 상한이 사라졌으므로 경고도 함께
        // 걷는다 — 이제 위치축 상한은 kvcap(=clamp된 ctx) 자체다.
        let mut dec = Exl3CudaDecoder::load_slots(dir, usize::MAX, kvcap, slots)?;
        // P0-3: 스펙 의도일 때만 MTP 조립(비스펙 프리필의 +드래프트 비용 0).
        let spec_k = crate::engine::SPEC_K.get().copied().unwrap_or(0);
        let mut mtps = Vec::new();
        if spec_k > 0 {
            match Exl3CudaMtp::from_dir(&mut dec, dir) {
                Ok(m0) => {
                    eprintln!("# mtp: EXL3 CUDA 드래프트 조립(슬롯 {slots}, k={spec_k})");
                    mtps.push(m0);
                    for _ in 1..slots {
                        mtps.push(Exl3CudaMtp::from_dir(&mut dec, dir)?);
                    }
                }
                Err(e) => {
                    // 35B 계열(mtp.* 부재) 등 — 경고 후 비스펙 진행(q35 계약).
                    eprintln!("# mtp: EXL3 CUDA 드래프트 조립 실패 — --spec 무시: {e}");
                }
            }
        }
        Ok(Self {
            dec,
            eos: exl3_eos_of(dir),
            mtp_h: vec![None; slots],
            mtp_pp: vec![0; slots],
            mtp_kv: vec![0; slots],
            mtps,
        })
    }

    /// MTP 드래프트 사용 가능(스펙 라운드 진입 계약).
    pub fn has_mtp(&self) -> bool {
        !self.mtps.is_empty()
    }

    /// 1토큰 순차 디코드 — 반환 로짓(디바이스 상주 경로, S10).
    pub fn decode1(&mut self, slot: usize, tok: u32) -> Result<Vec<f32>, String> {
        self.dec.forward_tok_device(slot, tok)
    }

    /// 프리필 — 배치 경로(S11)로 T≤8토큰씩 처리해 마지막 로짓을 반환한다.
    ///
    /// [왜 배치인가] T=1 순차는 토큰당 64층을 한 번씩 돈다. T행으로 넘기면
    /// GEMM2·norm이 행 병렬로 처리하고 lm_head도 T행을 한 번에 돈다.
    /// T 상한은 어텐션 fwd3s의 ATTN_F3S_TMAX(8) — 그보다 큰 청크는
    /// 커널이 거부한다.
    ///
    /// [정합] S11 동치성 게이트(cuda_probe s11)가 배치가 T=1과 같은 토큰을
    /// 고름을 실물로 확인했다(어텐션 0.000e0·GDN 7.5e-4·fwd argmax 일치).
    /// f16 mma 누산 차이로 logit maxdiff는 ~1.6e-2지만 argmax는 같고,
    /// 그게 실사용 판정이다(게이트 스크립트와 동일 기준).
    ///
    /// [P0-3] MTP 적재 시 같은 배치로 드래프트 KV를 전 문맥 적립한다
    /// (verify_batch_with_mtp — 타깃 로짓 불변, MTP는 자체 KV에만 쓴다).
    /// 스펙 상태 시딩: h = 마지막 행 pre-final-norm 잔차, pp = 다음 예측.
    pub fn prefill(&mut self, slot: usize, tokens: &[u32]) -> Result<Vec<f32>, String> {
        if tokens.is_empty() {
            return Err("빈 프리필".into());
        }
        let tmax = llm170_backend_gpu::rawcuda::attn_cuda::ATTN_F3S_TMAX;
        let mut last = Vec::new();
        let mut h_carry: Option<Vec<f32>> = None;
        let use_mtp = slot < self.mtps.len();
        for (ci, chunk) in tokens.chunks(tmax).enumerate() {
            let mut rows: Vec<f32> = Vec::with_capacity(chunk.len() * self.dec.hidden);
            for &tok in chunk {
                let row = self.dec.embed_row_host(tok);
                if row.len() != self.dec.hidden {
                    return Err(format!("exl3-cuda: 임베딩 토큰 {tok} 범위 밖 또는 미적재"));
                }
                rows.extend_from_slice(&row);
            }
            if use_mtp {
                // 첫 청크 행 0은 h_{-1} 부재로 스킵(None) — 이후 청크는
                // 직전 청크 마지막 잔차를 h0로 이어 받는다.
                let h0 = if ci == 0 { None } else { h_carry.as_deref() };
                let (ams, hnew, lg_last) = {
                    let mtp = &self.mtps[slot];
                    self.dec.verify_batch_with_mtp(slot, &rows, mtp, h0)?
                };
                // MTP KV 적립 행 수: 첫 청크 t-1(행 0 스킵), 이후 t.
                self.mtp_kv[slot] += chunk.len() - usize::from(ci == 0);
                let _ = ams;
                h_carry = Some(hnew);
                last = lg_last;
                continue;
            }
            let (lg, h) = self.dec.forward_batch_device(slot, &rows)?;
            last = lg;
            h_carry = Some(h);
        }
        if use_mtp {
            self.mtp_h[slot] = h_carry;
            self.mtp_pp[slot] = llm170_core::qwen35::greedy(&last);
            return Ok(last);
        }
        self.mtp_h[slot] = h_carry;
        self.mtp_pp[slot] = llm170_core::qwen35::greedy(&last);
        Ok(last)
    }

    /// 1토큰 순차 디코드(greedy) — 디바이스 상주 경로(S10). 서버 기본 디코드
    /// 경로다: 토큰당 왕복이 임베딩 업로드·로짓 판독 2회뿐이다(호스트
    /// 스테이징은 GEMV마다 d2h→h2d를 반복해 층당 ~8회).
    ///
    /// [S10 게이트] 두 경로의 종단 토큰열이 동일한 것을 프로브
    /// (cuda_probe s10)가 실측 검증한다 — 산술이 아니라 값으로 증명한다.
    pub fn step_tok_device(&mut self, slot: usize, tok: u32) -> Result<u32, String> {
        // argmax_host가 1MB(로짓 벡터) 장치 버퍼를 cuMemAlloc하므로
        // forward의 가드 밖에서 부르면 INVALID_CONTEXT로 죽는다. 슬롯
        // 스레드에 current 컨텍스트가 전파되지 않기 때문이다(plans/cuda-port.md S5).
        let _g = self.dec.cc.guard()?;
        let logits = self.decode1(slot, tok)?;
        self.dec.argmax_host(&logits)
    }

    /// 슬롯 간 배치 디코드(greedy) — 활성 greedy 슬롯을 T≤8 청크로 나눠
    /// 한 번에 64층 forward 한다(plans/cuda-port.md §1). 트렐리스 선형이
    /// 행병렬 gemv_t 체인이라 행별 출력이 step_tok_device(T=1)와
    /// **비트동일**하다 — 동시 요청 토큰 스트림이 직렬과 같다
    /// (scripts/verify_cuda_slots.py가 종단 판정).
    ///
    /// 청크 분할을 엔진 안에서 한다 — 호출자(sched)는 상한을 몰라도
    /// 된다. 반환 순서 = items 순서.
    pub fn step_batch(&mut self, items: &[(usize, u32)]) -> Result<Vec<u32>, String> {
        if items.is_empty() {
            return Ok(Vec::new());
        }
        let _g = self.dec.cc.guard()?;
        let tmax =
            llm170_backend_gpu::rawcuda::attn_cuda::ATTN_F3S_TMAX.min(Exl3CudaDecoder::GEMV_T_TMAX);
        let mut out = Vec::with_capacity(items.len());
        for ch in items.chunks(tmax) {
            out.extend(self.dec.decode_batch_slots(ch)?);
        }
        Ok(out)
    }

    /// P0-3: MTP 스펙 라운드(hip spec_round D2 미러). greedy 전용.
    /// 반환 = 이번 라운드 새로 확정 토큰(수용 드래프트 + 교정;
    /// pp 자체는 호출자가 이미 배출). 거부 시 GDN 스냅샷 복원 + 수용
    /// 접두+교정 재처리로 진짜 상태 정렬(무롤백 재사용 오염 없음).
    ///
    /// [산술 계약] 검증 배치는 gemm2 T행(gemm2/fwd3s 청크 온라인 소프트맥스)
    /// — 직렬 디코드(gemv_t T=1)와 로짓 maxdiff ~1.6e-2 클래스라 근접
    /// 동률에서 argmax가 갈릴 수 있다(S11 게이트와 동일 계급: prefill 배치
    /// 이미 이 산술로 서빙 중). 스펙 스트림은 검증 산술에 자기일관적이다.
    pub fn spec_round(&mut self, slot: usize, k: usize) -> Result<Vec<u32>, String> {
        if slot >= self.mtps.len() {
            return Err("spec_round: MTP 미적재(아카이브에 mtp.* 부재 또는 --spec 0)".into());
        }
        if k == 0 || k + 1 > llm170_backend_gpu::rawcuda::attn_cuda::ATTN_F3S_TMAX {
            return Err(format!("spec_round: k={k} — 검증 배치 T=k+1≤8 계약"));
        }
        let _g = self.dec.cc.guard()?;
        let h = self.mtp_h[slot]
            .clone()
            .ok_or("spec_round: prefill 선행 필요")?;
        let pp = self.mtp_pp[slot];
        let pos_now = self.dec.slot_pos[slot];
        let mtp_kv0 = self.mtp_kv[slot];

        // ── 드래프트 k체인 — MTP가 pp를 자체 처리해 pp+1..을 예측.
        // 체인 h 캐리는 디바이스(mtp.dcur)에 상주(호스트 왕복 없음).
        {
            let mtp = &self.mtps[slot];
            mtp.mtp_set_pos(&self.dec.cc, mtp_kv0 as u32)?;
        }
        let mut drafts: Vec<u32> = Vec::with_capacity(k);
        let e_pp = self.dec.embed_row_host(pp);
        let mut tok = {
            let mtp = &self.mtps[slot];
            let (d0, _) = mtp.mtp_step_gpu(&mut self.dec, &e_pp, &h, true)?;
            mtp.mtp_pos_bump(&self.dec.cc)?;
            d0.unwrap_or(0)
        };
        for _ in 1..k {
            let e = self.dec.embed_row_host(tok);
            let d = {
                let mtp = &self.mtps[slot];
                // 체인 h 입력 = 직전 h_next(mtp.dcur 상주 — d2d 없이 직독).
                mtp.mtp_step_g(&mut self.dec, &e, mtp.dcur(), true, None)?
                    .ok_or("mtp: head 미실행")?
            };
            let _ = &self.mtps[slot].mtp_pos_bump(&self.dec.cc);
            drafts.push(tok);
            tok = d;
        }
        drafts.push(tok);

        // ── 검증: [pp, d_0..d_{k-1}] 배치 + MTP KV 적립(드래프트가 쓴
        // 위치를 같은 규칙으로 덮어쓴다 — 위치 색인 자가치유).
        let mut rows: Vec<f32> = Vec::with_capacity((k + 1) * self.dec.hidden);
        for &t in std::iter::once(&pp).chain(drafts.iter()) {
            rows.extend_from_slice(&self.dec.embed_row_host(t));
        }
        // 검증 MTP 적립은 드래프트가 쓴 위치부터 다시(位置 되감기 —
        // 드래프트 k스텝이 dpp를 전진시켰다). 위치 색인 쓰기라 같은 칸
        // 재기입이 자명하다.
        {
            let mtp = &self.mtps[slot];
            mtp.mtp_set_pos(&self.dec.cc, mtp_kv0 as u32)?;
        }
        self.dec.gdn_snapshot_slot(slot)?;
        let (ams, hnew, _lg0) = {
            let mtp = &self.mtps[slot];
            self.dec
                .verify_batch_with_mtp(slot, &rows, mtp, Some(&h))?
        };
        if llm170_diag::dump::opts().key("spec_accept") {
            eprintln!(
                "# spec-accept pos={pos_now} kv={mtp_kv0} pp={pp} drafts={drafts:?} verify={ams:?}"
            );
        }
        // 수용 판정: ams[j] == drafts[j](j<k).
        let mut rejected_at: Option<usize> = None;
        for j in 0..k {
            if ams[j] != drafts[j] {
                rejected_at = Some(j);
                break;
            }
        }
        let mut out: Vec<u32> = Vec::with_capacity(k + 1);
        match rejected_at {
            None => {
                // 전부 수용 — 마지막 검증 행(k)의 예측이 다음 pp(보너스).
                out.extend_from_slice(&drafts);
                out.push(ams[k]);
                self.mtp_pp[slot] = ams[k];
                self.mtp_h[slot] = Some(hnew);
                self.mtp_kv[slot] = mtp_kv0 + k + 1;
                Ok(out)
            }
            Some(j) => {
                // 롤백: GDN 복원 + 타깃 pos 되감기 + MTP pos 되감기 후
                // [pp, 수용 접두 d_0..d_{j-1}, 교정 ams[j]] 재처리 = 진짜 상태.
                self.dec.gdn_restore_slot(slot)?;
                self.dec.slot_pos[slot] = pos_now;
                self.dec.attn_set_pos(slot, pos_now)?;
                {
                    let mtp = &self.mtps[slot];
                    mtp.mtp_set_pos(&self.dec.cc, mtp_kv0 as u32)?;
                }
                let mut replay: Vec<u32> = Vec::with_capacity(j + 2);
                replay.push(pp);
                replay.extend_from_slice(&drafts[..j]);
                replay.push(ams[j]);
                let mut rrows: Vec<f32> = Vec::with_capacity(replay.len() * self.dec.hidden);
                for &t in &replay {
                    rrows.extend_from_slice(&self.dec.embed_row_host(t));
                }
                let (rams, hr, _lg1) = {
                    let mtp = &self.mtps[slot];
                    self.dec
                        .verify_batch_with_mtp(slot, &rrows, mtp, Some(&h))?
                };
                out.extend_from_slice(&drafts[..j]);
                out.push(ams[j]);
                self.mtp_pp[slot] = *rams.last().unwrap_or(&0);
                self.mtp_h[slot] = Some(hr);
                self.mtp_kv[slot] = mtp_kv0 + replay.len();
                Ok(out)
            }
        }
    }

    /// 슬롯 제자리 리셋 — GDN 링/스캔 상태와 pos를 디코더에서 함께 초기화.
    /// MTP 상태(자체 KV·h·pp)도 함께 영점화(원장 19호 — 상태 오염 가드).
    pub fn reset_seq(&mut self, slot: usize) -> Result<(), String> {
        if slot < self.mtps.len() {
            let _g = self.dec.cc.guard()?;
            self.mtps[slot].mtp_reset_kv(&self.dec.cc)?;
            self.mtp_h[slot] = None;
            self.mtp_pp[slot] = 0;
            self.mtp_kv[slot] = 0;
        }
        self.dec.reset_state(slot)
    }

    /// 전 슬롯 리셋(워밍업 종료 후).
    pub fn reset_states(&mut self) -> Result<(), String> {
        let slots = self.mtps.len();
        for s in 0..slots {
            let _g = self.dec.cc.guard()?;
            self.mtps[s].mtp_reset_kv(&self.dec.cc)?;
            self.mtp_h[s] = None;
            self.mtp_pp[s] = 0;
            self.mtp_kv[s] = 0;
        }
        self.dec.reset_states()
    }
}
