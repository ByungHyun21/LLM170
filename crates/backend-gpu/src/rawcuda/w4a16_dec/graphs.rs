use super::*;

impl W4a16Dec {
    /// 그래프 모드 가능 여부 — **기본 ON**(LLM170_GRAPH=0으로 끔).
    /// 캡처 실패(에러 반환) 후에는 직접 경로 고정(매 토큰 재시도 방지).
    /// debug_layers는 캡처 중 d2h/sync를 하므로 그래프 불가.
    ///
    /// [실측 2026-10-08, 27B·4090] 밀집 디코드는 **중립**(그래프 39.9 vs
    /// 직접 40.8 ms/토큰, n=24) — 손해가 없어 기본 ON으로 간다. 이득 상한은
    /// 호스트 enqueue(12ms)이고, GPU가 CPU를 기다릴 때만 회수된다. 값 하는
    /// 곳은 런치 바운드: MoE 전문가 소형 커널(35B-A3B 256×40)·오프로드·
    /// 다중 슬롯 — W4-1에서 데이터 주도 디스패치와 함께 재검한다.
    pub(super) fn graph_ok(&self) -> bool {
        llm170_diag::flag::ne0("LLM170_GRAPH")
            && !self.debug_layers
            && !self.graph_failed
            // [2026-10-09 P1] MoE 상주 경로는 캡처 가능해졌다: 라우터 top-k가
            // 디바이스(w4a16_moe_topk — 호스트 왕복 0), 전문가 디스패치가
            // 데이터 주도(디바이스 idx 포인터 테이블 — 포인터는 고정), h2d/sync
            // 없음(P1·P2). 스트리밍은 전문가 파일 스테이징(h2d_chunked sync)이라
            // 여전히 불가. 미검증 경로는 직접 경로 폴백(graph_failed)이 덮는다.
            && (self.n_experts == 0 || self.moe_resident)
    }

    /// 캡처 그래프 무효화 — 버퍼 재할당 시 옛 포인터 replay를 차단한다
    /// (다음 디코드가 재캡처). 재할당은 캡처 밖(프리필·업로드)에서만 일어난다.
    /// [P10 실측 2026-10-09] 워밍업 t=16 캡처 → t=128 프리필이 norm 버퍼를
    /// 재할당 → replay가 해제 주소에 기록(norm_resid OOB) → CUDA 700.
    pub(super) fn graph_invalidate(&mut self) {
        // [A8] 캐시 전량 폐기 — 옛 포인터를 기록한 exec는 replay 금지.
        for e in self.graph_cache.drain(..) {
            let _ = self.cc.graph_destroy(e.exec, e.handle);
        }
        // [A9] 배치(슬롯집합) 그래프도 동일 계약.
        for (_, e, g) in self.batch_graphs.drain(..) {
            let _ = self.cc.graph_destroy(e, g);
        }
    }

    /// 그래프 준비 — 캐시 히트면 즉시 exec 반환(A8, 슬롯/모드별 1회 캡처).
    /// 미스 시 버퍼 워밍업 → 캡처 → 인스턴스화 → 캐시 등록.
    /// 캡처 중 금지 API(동기 복사·alloc)를 배제하기 위해 ensure_*를 선행한다.
    pub(super) fn ensure_graph(
        &mut self,
        slot: usize,
        head: bool,
        argmax: bool,
    ) -> Result<ffi::CUgraphExec, String> {
        // [A8] 캐시 히트 — 슬롯/모드 전환 재캡처 제거.
        if let Some(e) = self
            .graph_cache
            .iter()
            .find(|e| e.slot == slot && e.head == head && e.argmax == argmax)
        {
            return Ok(e.exec);
        }
        self.warm_for_capture()?;
        // 2) 실스트림·pinned 1회 준비.
        if self.pin_embed.is_null() {
            self.cc.create_stream()?;
            self.pin_embed = self.cc.pinned_alloc(self.hidden * 4)?;
            self.pin_pos = self.cc.pinned_alloc(self.n_slots * 4)?;
        }
        let out_len = if head {
            self.head_n * 4
        } else {
            self.hidden * 4
        };
        if self.pin_out.is_null() || self.pin_out_len < out_len {
            // [A8 잠복 수정 2026-10-10] 핀드 출력 재할당 = 기존 exec들의 d2h
            // 목적지 해제 — 캐시된 그래프 전량 폐기(모드 교대에서 해제 주소
            // 기록 UAF, 할당자 재사용으로 잠복했던 실측 결함).
            self.graph_invalidate();
            if !self.pin_out.is_null() {
                let _ = self.cc.pinned_free(self.pin_out);
            }
            self.pin_out = self.cc.pinned_alloc(out_len)?;
            self.pin_out_len = out_len;
        }
        // 3) 캡처(실행 없음 — 기록만).
        self.cc.capture_begin()?;
        self.capture_pinned_src = true;
        let cap = (|| -> Result<(), String> {
            let row = vec![0f32; self.hidden]; // 내용 무의미(캡처는 실행 아님).
            let xn = self.chain_device(slot, &row)?;
            if head {
                let f = self.cc.function("head_bf16")?;
                let (mut p_w, mut p_x, mut p_o) = (self.head_w, xn, self.head_out);
                let (mut p_n, mut p_k) = (self.head_n as i32, self.head_k as i32);
                let mut args: [*mut std::ffi::c_void; 5] = [
                    (&mut p_w) as *mut _ as *mut _,
                    (&mut p_x) as *mut _ as *mut _,
                    (&mut p_o) as *mut _ as *mut _,
                    (&mut p_n) as *mut _ as *mut _,
                    (&mut p_k) as *mut _ as *mut _,
                ];
                self.cc
                    .launch(f, self.head_n.div_ceil(4 * 256) as u32, 1, 256, &mut args)?;
                if argmax {
                    // [P3] argmax 커널 + 4B d2h — 로짓 전량 readback 제거.
                    let fa = self.cc.function("w4a16_argmax_min")?;
                    let (mut p_l, mut p_n, mut p_o) =
                        (self.head_out, self.head_n as i32, self.argmax_out);
                    let mut aa: [*mut std::ffi::c_void; 3] = [
                        (&mut p_l) as *mut _ as *mut _,
                        (&mut p_n) as *mut _ as *mut _,
                        (&mut p_o) as *mut _ as *mut _,
                    ];
                    self.cc.launch(fa, 1, 1, 1024, &mut aa)?;
                    self.cc
                        .d2h_async(self.pin_out as *mut u8, self.argmax_out, 4)?;
                } else {
                    self.cc
                        .d2h_async(self.pin_out as *mut u8, self.head_out, self.head_n * 4)?;
                }
            } else {
                self.cc
                    .d2h_async(self.pin_out as *mut u8, xn, self.hidden * 4)?;
            }
            Ok(())
        })();
        self.capture_pinned_src = false;
        if let Err(e) = cap {
            let _ = self.cc.capture_end(); // 캡처 상태 정리(그래프 폐기).
            return Err(format!("캡처 본문: {e}"));
        }
        let g = self.cc.capture_end()?;
        let e = self.cc.graph_instantiate(g)?;
        // [A8] 캐시 등록 — 상한 = 슬롯×모드 3종(방어적으로 초과 시 최古 폐기).
        let cap = self.n_slots.max(1) * 3;
        if self.graph_cache.len() >= cap {
            let old = self.graph_cache.remove(0);
            let _ = self.cc.graph_destroy(old.exec, old.handle);
        }
        self.graph_cache.push(GraphEntry {
            exec: e,
            handle: g,
            slot,
            head,
            argmax,
        });
        // 캡처 성공 1회 로그 — 경로 가시화(MoE 상주 = P1 개방분 포함).
        eprintln!(
            "[graph] captured slot={slot} head={head} argmax={argmax} moe_resident={}",
            self.moe_resident
        );
        Ok(e)
    }

    /// 그래프 replay — pinned 입력 기입 → dpp 1회 갱신 → launch 1회 → sync →
    /// pinned 출력 회수. 반환: head 모드면 로짓, 아니면 xn.
    pub(super) fn graph_replay(
        &mut self,
        exec: ffi::CUgraphExec,
        slot: usize,
        embed_row: &[f32],
    ) -> Result<Vec<f32>, String> {
        let pos = self.slot_pos[slot];
        // SAFETY: pinned 버퍼는 hidden*4/n_slots*4 크기 계약(ensure_graph 할당).
        unsafe {
            std::ptr::copy_nonoverlapping(
                embed_row.as_ptr() as *const u8,
                self.pin_embed as *mut u8,
                self.hidden * 4,
            );
            std::ptr::copy_nonoverlapping(
                pos.to_le_bytes().as_ptr(),
                (self.pin_pos as *mut u8).add(slot * 4),
                4,
            );
        }
        // dpp 갱신은 그래프 밖·같은 스트림(그래프보다 먼저 실행 — 순서 보장).
        let posb =
            unsafe { std::slice::from_raw_parts(self.pin_pos as *const u8, self.n_slots * 4) };
        self.cc
            .h2d_async(self.dpp + (slot as u64) * 4, &posb[slot * 4..slot * 4 + 4])?;
        self.cc.graph_launch(exec)?;
        self.cc.sync()?;
        let out =
            unsafe { std::slice::from_raw_parts(self.pin_out as *const f32, self.pin_out_len / 4) };
        let v = out.to_vec();
        self.slot_pos[slot] = pos + 1;
        Ok(v)
    }

    /// [P3] 그래프 replay(argmax 모드) — 4B 인덱스 회수. graph_replay와 동일
    /// 계약(입력 pinned 기입 → dpp 1회 → launch 1회 → sync).
    pub(super) fn graph_replay_argmax(
        &mut self,
        exec: ffi::CUgraphExec,
        slot: usize,
        embed_row: &[f32],
    ) -> Result<u32, String> {
        let pos = self.slot_pos[slot];
        // SAFETY: pinned 버퍼는 hidden*4/n_slots*4 크기 계약(ensure_graph 할당).
        unsafe {
            std::ptr::copy_nonoverlapping(
                embed_row.as_ptr() as *const u8,
                self.pin_embed as *mut u8,
                self.hidden * 4,
            );
            std::ptr::copy_nonoverlapping(
                pos.to_le_bytes().as_ptr(),
                (self.pin_pos as *mut u8).add(slot * 4),
                4,
            );
        }
        let posb =
            unsafe { std::slice::from_raw_parts(self.pin_pos as *const u8, self.n_slots * 4) };
        self.cc
            .h2d_async(self.dpp + (slot as u64) * 4, &posb[slot * 4..slot * 4 + 4])?;
        self.cc.graph_launch(exec)?;
        self.cc.sync()?;
        let ob = unsafe { std::slice::from_raw_parts(self.pin_out as *const u8, 4) };
        self.slot_pos[slot] = pos + 1;
        Ok(u32::from_le_bytes([ob[0], ob[1], ob[2], ob[3]]))
    }

    /// [P3] 디코드 argmax — head 로짓을 디바이스에서 argmax(그래프 = 4B d2h).
    /// 반환 = 토큰 인덱스(min-index-on-tie = CPU greedy_from 계약 미러).
    pub fn forward_device_argmax(&mut self, slot: usize, embed_row: &[f32]) -> Result<u32, String> {
        if self.head_w == 0 || self.argmax_out == 0 {
            return Err("forward_device_argmax: head/argmax 미등록 — upload_head 선행".into());
        }
        let _g = self.cc.guard()?;
        if self.graph_ok() {
            match self
                .ensure_graph(slot, true, true)
                .and_then(|exec| self.graph_replay_argmax(exec, slot, embed_row))
            {
                Ok(t) => return Ok(t),
                Err(e) => {
                    eprintln!("[graph] argmax 경로 실패 — 직접 경로 폴백: {e}");
                    self.graph_failed = true;
                }
            }
        }
        let pos = self.slot_pos[slot];
        let xn = self.chain_device(slot, embed_row)?;
        let f = self.cc.function("head_bf16")?;
        let (mut p_w, mut p_x, mut p_o) = (self.head_w, xn, self.head_out);
        let (mut p_n, mut p_k) = (self.head_n as i32, self.head_k as i32);
        let mut args: [*mut std::ffi::c_void; 5] = [
            (&mut p_w) as *mut _ as *mut _,
            (&mut p_x) as *mut _ as *mut _,
            (&mut p_o) as *mut _ as *mut _,
            (&mut p_n) as *mut _ as *mut _,
            (&mut p_k) as *mut _ as *mut _,
        ];
        self.cc
            .launch(f, self.head_n.div_ceil(4 * 256) as u32, 1, 256, &mut args)?;
        let fa = self.cc.function("w4a16_argmax_min")?;
        let (mut p_l, mut p_nn, mut p_a) = (self.head_out, self.head_n as i32, self.argmax_out);
        let mut aa: [*mut std::ffi::c_void; 3] = [
            (&mut p_l) as *mut _ as *mut _,
            (&mut p_nn) as *mut _ as *mut _,
            (&mut p_a) as *mut _ as *mut _,
        ];
        self.cc.launch(fa, 1, 1, 1024, &mut aa)?;
        let mut ob = [0u8; 4];
        self.cc.d2h_async(ob.as_mut_ptr(), self.argmax_out, 4)?;
        self.cc.sync()?;
        self.slot_pos[slot] = pos + 1;
        self.attn_set_pos(slot, pos + 1)?;
        Ok(u32::from_le_bytes(ob))
    }

    /// [A9 2026-10-10] 배치 greedy 디코드 — 슬롯별 임베딩 행(t=n_active)을
    /// 단일 체인으로 통과, 슬롯 순서의 다음 토큰 반환. 호출부가 전제
    /// (전 슬롯 greedy·t≤BATCH_DEC_MAX)를 보장한다.
    pub fn forward_device_argmax_batch(
        &mut self,
        slots: &[usize],
        rows: &[f32],
    ) -> Result<Vec<u32>, String> {
        let t = slots.len();
        if !(2..=BATCH_DEC_MAX).contains(&t) {
            return Err("batch: t 2..=BATCH_DEC_MAX 전용".into());
        }
        let _g = self.cc.guard()?;
        if rows.len() != t * self.hidden
            || (self.n_experts > 0 && !self.moe_resident)
            || self.head_w == 0
            || slots.iter().any(|&s| s >= self.n_slots)
        {
            return Err(format!("batch: 전제 위반 t={t}"));
        }
        let cap = self.attn.ok_or("attn: 형상 미등록")?.cap;
        for &s in slots {
            if self.slot_pos[s] as usize + 1 > cap {
                return Err(format!("batch: slot{s} 컨텍스트 초과"));
            }
        }
        self.ensure_batch_bufs()?;
        // 핀드 행/pos 기입(그래프 h2d 노드가 replay 시점에 읽는다).
        unsafe {
            std::ptr::copy_nonoverlapping(
                rows.as_ptr() as *const u8,
                self.pin_batch_in as *mut u8,
                t * self.hidden * 4,
            );
            let pp = self.pin_batch_pos as *mut u8;
            for s in 0..self.n_slots {
                let pos = self.slot_pos[s];
                std::ptr::copy_nonoverlapping(pos.to_le_bytes().as_ptr(), pp.add(s * 4), 4);
            }
        }
        let key: Vec<usize> = slots.to_vec();
        if let Some(k) = self.batch_graphs.iter().position(|(ks, _, _)| *ks == key) {
            let exec = self.batch_graphs[k].1;
            self.cc.graph_launch(exec)?;
            self.cc.sync()?;
            let out = self.batch_read_tokens(t);
            for &s in slots {
                self.slot_pos[s] += 1;
            }
            return Ok(out);
        }
        // 미스: 미캡처 실발사(정답) → 캡처(미실행)로 다음부터 replay.
        self.batch_launch(slots, t)?;
        self.cc.sync()?;
        let out = self.batch_read_tokens(t);
        if !self.batch_capture_failed {
            match self.batch_capture(&key, t) {
                Ok(()) => {}
                Err(e) => {
                    eprintln!("# batch 그래프 캡처 실패 — 직접 발사 유지: {e}");
                    self.batch_capture_failed = true;
                    self.cc.sync().ok();
                }
            }
        }
        for &s in slots {
            self.slot_pos[s] += 1;
        }
        Ok(out)
    }

    /// [A9] 배치 그래프 캡처 — 슬롯집합 키. 캡처는 실행하지 않는다(상태 불변).
    pub(super) fn batch_capture(&mut self, key: &[usize], t: usize) -> Result<(), String> {
        if self.batch_graphs.len() >= 8 {
            let (_, e, g) = self.batch_graphs.remove(0);
            let _ = self.cc.graph_destroy(e, g);
        }
        self.cc.capture_begin()?;
        if let Err(e) = self.batch_launch(key, t) {
            let _ = self.cc.capture_end();
            return Err(e);
        }
        let g = self.cc.capture_end()?;
        let e = self.cc.graph_instantiate(g)?;
        self.batch_graphs.push((key.to_vec(), e, g));
        eprintln!("[batch-graph] captured slots={key:?}");
        Ok(())
    }

    /// 프리필 청크 진입(공개) — t=1은 기존 경로(그래프 포함), t≥2는 배치 체인.
    pub fn forward_prefill(
        &mut self,
        slot: usize,
        rows: &[f32],
        t: usize,
        head: bool,
    ) -> Result<Vec<f32>, String> {
        if t == 1 {
            let row = &rows[..self.hidden];
            return if head {
                self.forward_device_head(slot, row)
            } else {
                self.forward_device(slot, row)
            };
        }
        self.chain_device_t(slot, rows, t, head)
    }

    /// 1토큰 forward(디바이스 체인) — 최종 xn까지 판독(CPU head용).
    /// guard는 래퍼 전체를 덮는다 — 체인 이후의 d2h/h2d도 같은 스레드
    /// 컨텍스트가 필요하다(CtxGuard는 드랍 시 이전 컨텍스트로 복원).
    pub fn forward_device(&mut self, slot: usize, embed_row: &[f32]) -> Result<Vec<f32>, String> {
        let _g = self.cc.guard()?;
        if self.graph_ok() {
            match self
                .ensure_graph(slot, false, false)
                .and_then(|exec| self.graph_replay(exec, slot, embed_row))
            {
                Ok(xn) => {
                    llm170_diag::fp::fp_record("gpu.xn", &xn);
                    return Ok(xn);
                }
                Err(e) => {
                    eprintln!("[graph] xn 경로 실패 — 직접 경로 폴백: {e}");
                    self.graph_failed = true;
                }
            }
        }
        let pos = self.slot_pos[slot];
        let xn_final = self.chain_device(slot, embed_row)?;
        let mut ob = vec![0u8; self.hidden * 4];
        self.cc
            .d2h_async(ob.as_mut_ptr(), xn_final, self.hidden * 4)?;
        self.cc.sync()?;
        self.slot_pos[slot] = pos + 1;
        self.attn_set_pos(slot, pos + 1)?;
        let xn =
            unsafe { std::slice::from_raw_parts(ob.as_ptr() as *const f32, self.hidden) }.to_vec();
        // S5: 지문 파이프라인 부활 — LLM170_FP_FILE 시 스테이지 해시 기록
        // (`diag diff`로 실행 2개의 최초 발산 스테이지 추적).
        llm170_diag::fp::fp_record("gpu.xn", &xn);
        Ok(xn)
    }

    /// 1토큰 forward + GPU head — xn 판독 없이 로짓만 회수(왕복 1회).
    pub fn forward_device_head(
        &mut self,
        slot: usize,
        embed_row: &[f32],
    ) -> Result<Vec<f32>, String> {
        if self.head_w == 0 {
            return Err("forward_device_head: head 미등록 — upload_head 선행".into());
        }
        let _g = self.cc.guard()?;
        if self.graph_ok() {
            match self
                .ensure_graph(slot, true, false)
                .and_then(|exec| self.graph_replay(exec, slot, embed_row))
            {
                Ok(lg) => {
                    llm170_diag::fp::fp_record("gpu.logits", &lg);
                    return Ok(lg);
                }
                Err(e) => {
                    eprintln!("[graph] head 경로 실패 — 직접 경로 폴백: {e}");
                    self.graph_failed = true;
                }
            }
        }
        let pos = self.slot_pos[slot];
        let xn = self.chain_device(slot, embed_row)?;
        let f = self.cc.function("head_bf16")?;
        let (mut p_w, mut p_x, mut p_o) = (self.head_w, xn, self.head_out);
        let (mut p_n, mut p_k) = (self.head_n as i32, self.head_k as i32);
        let mut args: [*mut std::ffi::c_void; 5] = [
            (&mut p_w) as *mut _ as *mut _,
            (&mut p_x) as *mut _ as *mut _,
            (&mut p_o) as *mut _ as *mut _,
            (&mut p_n) as *mut _ as *mut _,
            (&mut p_k) as *mut _ as *mut _,
        ];
        self.cc
            .launch(f, self.head_n.div_ceil(4 * 256) as u32, 1, 256, &mut args)?;
        let mut ob = vec![0u8; self.head_n * 4];
        self.cc
            .d2h_async(ob.as_mut_ptr(), self.head_out, self.head_n * 4)?;
        self.cc.sync()?;
        self.slot_pos[slot] = pos + 1;
        self.attn_set_pos(slot, pos + 1)?;
        let lg: Vec<f32> = ob
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        llm170_diag::fp::fp_record("gpu.logits", &lg);
        Ok(lg)
    }
}
