//! [K-STEP 2026-10-12 §3.4 ① 트랜셰 1] 디바이스 런타임 K-스텝 디코드 루프.
//!
//! 문제: 현행 디코드 틱은 토큰마다 호스트가 그래프 리플레이 → 로짓/argmax
//! 판독 → EOS/max_tokens 검사 → 임베딩 행 조립·h2d → dpp 갱신을 반복한다
//! (호스트-GPU 동기화 = 토큰당 1회). K-스텝 루프는 체인 t=1 스텝 그래프를
//! **k회 연속 리플레이**해 디바이스에서 자율 진행하고 호스트는 K스텝마다
//! 체크인한다(방출 토큰 로그·카운터·정지 플래그 회수 1회).
//!
//! 설계(§3.4 ① — MTP 체인(§3.3 ②) 메커니즘의 본체 디코드 전용화):
//! 스텝 그래프 1회 캡처 = [gather → 체인 본체 → head → argmax → 컨트롤러].
//!  - 토큰 레지스터 = argmax_out(4B). 스텝 j의 argmax가 d_{j+1}을 기록하면
//!    스텝 j+1의 w4a16_embed_gather가 그 토큰의 임베딩 행을 dres로 조립한다
//!    (h2f/b2f = 호스트 디코드와 비트동일 — embed.cu 계약, 순차 경로의
//!    핀드 행 h2d와 값 동일 → 토큰열 판정 동일).
//!  - pos = dpp[slot] 디바이스 판독(어텐션 커널 계약). 컨트롤러
//!    (w4a16_kstep_ctrl)가 리플레이마다 dpp[slot]을 +1 전진한다.
//!  - EOS: 컨트롤러가 정지 플래그를 세우고 이후 리플레이는 부기·전진 없이
//!    같은 pos에서 재실행(비조건 그래프 — 노드 스킵 불가). 잔여 스텝의 KV
//!    오염은 인과적 고립(종료된 시퀀스의 쓰레기 행 — 컨텍스트 상한 내)으로
//!    무해하다. EOS 토큰 자체는 소비되지 않는다(호스트 루프가 방출 후
//!    중단하는 것과 동일 의미론).
//!  - 스트림 순서: 같은 스트림의 그래프 런치는 전체 그래프 완료 후에만
//!    시작된다 — 스텝 j의 argmax 기록이 스텝 j+1의 gather·부기 판독에
//!    선행한다(mtp_chain_drafts k-1 리플레이와 동일 보장).
//!
//! 수치: 본체는 순차 t=1 경로와 동일 커널열(입력 소스만 h2d→gather 교체)
//! — 판정은 토큰열(모듈 계약 승계, kstep_decode_matches_sequential 테스트).
//!
//! 폴백: 계약 위반·캡처 실패(kstep.failed 고정)는 Err 반환 — 호출부(sched
//! 단독 그리디)가 기존 토큰당 단일 스텝 경로로 폴백한다.

use super::*;

impl W4a16Dec {
    /// [K-STEP §3.4 ①] K-스텝 greedy 디코드 — seed_tok(직전 방출 토큰)을
    /// pos=slot_pos부터 k스텝 자율 소비하고 방출 토큰열(최대 k개, EOS 포함
    /// EOS까지)을 반환한다. slot_pos/dpp는 ctr(실제 방출 수)만큼 전진한다.
    ///
    /// 진입 계약: (1) 임베딩 상주(upload_embed — gather 원본) · head/argmax
    /// 등록 · head_n ≤ emb_n(방출 토큰의 임베딩 행 존재). (2) seed_tok ≠
    /// eos(호스트 루프가 EOS에서 이미 중단 — 위반 시 Err). (3) k∈2..=
    /// KSTEP_KMAX이며 pos+k ≤ cap(호출부가 max_tokens 잔여로 클램프; 본
    /// 함수가 cap 잔여로 추가 클램프 — 2 미만이면 Err 폴백). (4) 그래프
    /// 모드 가능(graph_ok — debug_layers·스트리밍 MoE 등은 캡처 불가).
    pub fn decode_k_greedy(
        &mut self,
        slot: usize,
        seed_tok: u32,
        k: usize,
        eos: u32,
    ) -> Result<Vec<u32>, String> {
        let _g = self.cc.guard()?;
        if slot >= self.n_slots {
            return Err(format!("kstep: 슬롯 {slot} >= {}", self.n_slots));
        }
        if !(2..=KSTEP_KMAX).contains(&k) {
            return Err(format!("kstep: k={k} — 2..={KSTEP_KMAX} 전용"));
        }
        if !self.embed_ready() {
            return Err("kstep: 임베딩 미등록(upload_embed 선행)".into());
        }
        if self.head_w == 0 || self.argmax_out == 0 {
            return Err("kstep: head/argmax 미등록".into());
        }
        if self.head_n > self.emb_n {
            return Err(format!(
                "kstep: head_n {} > emb_n {} — 방출 토큰 임베딩 행 부재",
                self.head_n, self.emb_n
            ));
        }
        if seed_tok as usize >= self.emb_n || seed_tok == eos {
            // 시드가 EOS/범위 밖이면 첫 스텝부터 무의미 — 호스트가 이미
            // 중단했어야 한다(방어).
            return Err(format!("kstep: 시드 토큰 {seed_tok} 부적합(범위/EOS)"));
        }
        let dm = self.attn.attn.ok_or("kstep: attn 미등록")?;
        let pos = self.slot_pos[slot];
        // cap 잔여 클램프 — 컨트롤러 in-kernel 검사는 본체 뒤라 KV 기록을
        // 막지 못한다(커널 주석 참조). 정지 후 재실행도 같은 pos에 머무르므로
        // 실제 사용 상한은 pos+k-1 < cap.
        let k = k.min(dm.cap.saturating_sub(pos as usize));
        if k < 2 {
            return Err(format!("kstep: cap 잔여 부족 pos{pos}/cap{}", dm.cap));
        }
        if !self.graph_ok() || self.kstep.failed {
            return Err("kstep: 그래프 모드 불가(폴백)".into());
        }
        let exec = self.ensure_kstep_graph(slot, eos)?;
        // 부기 리셋(ctr·stop 8B 0) + 시드(토큰 레지스터·dpp) — 같은 스트림
        // 이므로 첫 리플레이 전에 완료된다(sync 불요 — mtp_chain_drafts와
        // 동일 패턴, 호출 내 완료되는 스택 소스).
        let rst = [0u8; 8];
        self.cc.h2d_async(self.kstep.buf, &rst)?;
        self.cc
            .h2d_async(self.argmax_out, &seed_tok.to_le_bytes())?;
        self.cc
            .h2d_async(self.attn.dpp + (slot as u64) * 4, &pos.to_le_bytes())?;
        // k회 연속 리플레이 — 중간 호스트 왕복 0.
        for _ in 0..k {
            self.cc.graph_launch(exec)?;
        }
        // 로그 회수(ctr·stop·out[0..k]) — 그래프 완료 뒤 같은 스트림 d2h.
        let mut ob = vec![0u8; 8 + k * 4];
        self.cc
            .d2h_async(ob.as_mut_ptr(), self.kstep.buf, 8 + k * 4)?;
        self.cc.sync()?;
        let ctr = u32::from_le_bytes([ob[0], ob[1], ob[2], ob[3]]) as usize;
        // ctr ≥ 1은 첫 리플레이 컨트롤러 기록으로 보장(시드 ≠ eos 검사됨).
        if ctr == 0 || ctr > k {
            return Err(format!(
                "kstep: 카운터 부적합 ctr={ctr}(k={k}) — 컨트롤러 계약 위반"
            ));
        }
        let toks: Vec<u32> = ob[8..8 + ctr * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        // 호스트 미러 갱신 — ctr = 실제 방출 수(EOS 행 포함). dpp는 EOS 정지
        // 시 ctr-1까지 전진했으므로 attn_set_pos로 미러와 정합 복구(다음
        // 요청·프리필의 cap 검사 일관성).
        self.slot_pos[slot] = pos + ctr as u32;
        self.attn_set_pos(slot, pos + ctr as u32)?;
        Ok(toks)
    }

    /// [K-STEP] 스텝 그래프 확보 — (슬롯, eos) 키 1회 캡처. eos는 커널 인자로
    /// 베이크되므로 캐시 엔트리의 eos와 불일치하면 폐기 후 재캡처(실전은
    /// 모델당 상수라 재캡처 없음 — 테스트가 서로 다른 eos로 검증). 버퍼
    /// ensure는 전부 캡처 밖(캡처 중 cuMemAlloc 금지 계약 — ensure_graph·
    /// ensure_mtp_chain_graph 미러).
    fn ensure_kstep_graph(&mut self, slot: usize, eos: u32) -> Result<ffi::CUgraphExec, String> {
        if slot >= self.kstep.graphs.len() {
            self.kstep.graphs.resize(slot + 1, None);
        }
        if let Some((e_eos, e, h)) = self.kstep.graphs[slot].take() {
            if e_eos == eos {
                self.kstep.graphs[slot] = Some((e_eos, e, h));
                return Ok(e);
            }
            // eos 불일치 — 옛 그래프 폐기 후 재캡처.
            let _ = self.cc.graph_destroy(e, h);
        }
        // 체인 지연 할당원 전량 선할당(CHAIN_TMAX 기준 — warm_for_capture
        // 계약, 기존 t=1 그래프와 동일 보장).
        self.warm_for_capture()?;
        // 최초 캡처 전 실스트림 확보(레거시 0 BeginCapture = CUresult 900).
        if self.cc.stream.is_null() {
            self.cc.create_stream()?;
        }
        // 부기 버퍼 — [ctr 4B][stop 4B][out KSTEP_KMAX×4B] 0 초기화.
        if self.kstep.buf == 0 {
            let b = self.cc.alloc(8 + KSTEP_KMAX * 4)?;
            Self::zero_dev(&self.cc, b, 8 + KSTEP_KMAX * 4)?;
            self.kstep.buf = b;
        }
        self.cc.capture_begin()?;
        // capture 게이트 — 캡처 본문의 attn_set_pos(h2d 스택 임시 소스)를
        // 억제한다(ensure_mtp_chain_graph 미러 — 미설정 시 리플레이가 무효
        // 주소 판독, CUresult 700).
        self.graph.capture_pinned_src = true;
        let cap = self.kstep_launch(slot, eos);
        self.graph.capture_pinned_src = false;
        if let Err(e) = cap {
            let _ = self.cc.capture_end();
            self.kstep.failed = true;
            return Err(format!("kstep 캡처 본문: {e}"));
        }
        let g = self.cc.capture_end()?;
        let e = self.cc.graph_instantiate(g)?;
        self.cc.graph_upload(e)?; // [§4.3] implicit upload spin 제거
        self.kstep.graphs[slot] = Some((eos, e, g));
        eprintln!("[kstep] captured slot={slot} eos={eos}");
        Ok(e)
    }

    /// [K-STEP] 스텝 그래프 본문 — 캡처 전용(직접 발사 없음). 리플레이 1회 =
    /// 디코드 1스텝: gather → 체인 본체 → head → argmax(토큰 레지스터 자기
    /// 연결) → 컨트롤러(부기·정지·dpp 전진).
    fn kstep_launch(&mut self, slot: usize, eos: u32) -> Result<(), String> {
        // 1) 임베딩 gather — 토큰 레지스터(argmax_out) → dres(1행).
        self.embed_gather_launch(self.argmax_out, self.dres, 1)?;
        // 2) 체인 본체(t=1) — 임베딩 h2d 없음(위 gather가 dres 기입).
        let xn = self.chain_device_body(slot)?;
        // 3) GPU head → 로짓.
        self.head_gemv_launch(xn, self.head_out)?;
        // 4) argmax → 토큰 레지스터 덮어쓰기(다음 스텝 입력).
        let fa = self.cc.function("w4a16_argmax_min")?;
        let (mut p_l, mut p_n, mut p_o) = (self.head_out, self.head_n as i32, self.argmax_out);
        self.cc.launch(
            fa,
            1,
            1,
            1024,
            &mut crate::rawcuda::args::l3(&mut p_l, &mut p_n, &mut p_o),
        )?;
        // 5) 컨트롤러 — 방출 로그·카운터·EOS 정지·dpp 전진.
        let fc = self.cc.function("w4a16_kstep_ctrl")?;
        let (mut p_t, mut p_o, mut p_c) = (self.argmax_out, self.kstep.buf + 8, self.kstep.buf);
        let (mut p_s, mut p_p) = (self.kstep.buf + 4, self.attn.dpp);
        let (mut p_sl, mut p_e) = (slot as i32, eos as i32);
        self.cc.launch(
            fc,
            1,
            1,
            32,
            &mut crate::rawcuda::args::l7(
                &mut p_t, &mut p_o, &mut p_c, &mut p_s, &mut p_p, &mut p_sl, &mut p_e,
            ),
        )
    }
}
