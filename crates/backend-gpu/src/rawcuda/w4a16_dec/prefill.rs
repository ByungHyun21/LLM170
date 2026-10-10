//! 배치/프리필 체인(t≥2) — [R5 2026-10-10] chains.rs에서 분리.
//! chain_t가 프리필 청크(GEMM/mma 경로)와 스펙 검증(all=true)·eval
//! 로짓 모드를 소유한다. 디코드(t=1) 체인은 chains.rs.

use super::*;

impl W4a16Dec {
    // ── CUDA Graph(체인 캡처 — P1) ──

    /// 배치 체인(t∈2..=8) — 프리필 청크. GEMM(t≥2) 경로 + 배치 버퍼.
    /// 반환: 마지막 행의 xn(또는 head면 로짓). t≥2 GEMM은 w4a16-gemm
    /// 게이트가 비트 판정(행별 64레인·tree64 동일) — t=1 경로와 계약 동일.
    pub(super) fn chain_device_t(
        &mut self,
        slot: usize,
        rows: &[f32],
        t: usize,
        head: bool,
    ) -> Result<Vec<f32>, String> {
        self.chain_t(slot, rows, t, head, false, false)
    }

    /// [A-1] 스펙 검증 체인 — `all`=true면 **전 위치** 최종 노름+배치 head+
    /// 배치 argmax로 토큰 t개(id를 f32로)를 반환(상태·pos는 전진).
    /// 롤백은 호출부(상태 복원 + pos 되감기) 소관.
    /// [eval 2026-10-10] `logits`=true(all과 함께)면 argmax 대신 [t][head_n]
    /// 로짓을 회수한다(청크 프리필 PPL 가속 — eval 프로브 전용).
    pub(super) fn chain_t(
        &mut self,
        slot: usize,
        rows: &[f32],
        t: usize,
        head: bool,
        all: bool,
        logits: bool,
    ) -> Result<Vec<f32>, String> {
        let _g = self.cc.guard()?;
        if !(2..=CHAIN_TMAX).contains(&t) || rows.len() != t * self.hidden || slot >= self.n_slots {
            return Err(format!(
                "chain_device_t: t={t} rows={} 계약 위반",
                rows.len()
            ));
        }
        let pos = self.slot_pos[slot];
        let cap = self.attn.ok_or("attn: 형상 미등록")?.cap;
        if pos as usize + t > cap {
            return Err(format!("context overflow: pos{pos}+T{t} > kvcap{cap}"));
        }
        if self.norm_w_rows < 2 * self.n_layers + 1 || self.gdn.is_none() {
            return Err("chain_device_t: 디코더 상수 미등록".into());
        }
        self.ensure_chain_bufs()?;
        self.ensure_norm_bufs(t)?;
        self.ensure_gdn_bufs(t)?;
        self.ensure_attn_bufs(t)?;
        let dyt = self.ensure_dyt(t)?;
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
        let rb = unsafe { std::slice::from_raw_parts(rows.as_ptr() as *const u8, rows.len() * 4) };
        self.cc.h2d_async(self.dres, rb)?;
        let [s0, s1, s1b, s2, s3] = self.dchain;
        let (w2, h) = (self.stg_w2, self.hidden);
        let mut ab = self.dab_dev;
        let mut gi = 0usize;
        // 플레인(MoE) 모드 — bf16 GEMM(x 원시 f32), FFN은 MoE 배치.
        let plain = self.plain_weights;
        // [P10] GEMM 상한 — mma 경로(TC ON·t≥16)는 t 무제한, FFMA 폴백
        // 커널(G4_GTMAX/G4_TMAX2=32)만 32 상한. 폴백으로 t>32를 태우지 않는다.
        let gemm_mma = t >= 16 && llm170_diag::flag::ne0("LLM170_TC");
        if t > GEMM_FFMA_TMAX && !gemm_mma {
            return Err(format!(
                "chain_device_t: t={t} > {GEMM_FFMA_TMAX} — FFMA 폴백 상한(TC=0 진단 또는 t<16)"
            ));
        }
        for il in 0..self.n_layers {
            let xh = if plain { 0 } else { self.ensure_dx32(t * h)? };
            let xn = self
                .norm_resid_dev(2 * il, self.dres, ab, t, xh)
                .map_err(|e| format!("T{il} input norm: {e}"))?;
            let interval = self.attn.ok_or("attn: 형상 미등록")?.interval;
            let branch = if (il + 1) % interval == 0 {
                // [R10] 4변형 → lin_forward 단일 디스패치(Gemm 경로).
                let xin = if plain { xn } else { xh };
                self.lin_forward(
                    plain,
                    LinPath::Gemm,
                    &format!("blk.{il}.attn_q.weight"),
                    xin,
                    s0,
                    t,
                )?;
                self.lin_forward(
                    plain,
                    LinPath::Gemm,
                    &format!("blk.{il}.attn_k.weight"),
                    xin,
                    s1,
                    t,
                )?;
                self.lin_forward(
                    plain,
                    LinPath::Gemm,
                    &format!("blk.{il}.attn_v.weight"),
                    xin,
                    s1b,
                    t,
                )?;
                self.attn_chain_dev_run(slot, il / interval, t, s0, s1, s1b)
                    .map_err(|e| format!("T{il} attn: {e}"))?
            } else {
                if plain {
                    self.lin_forward(
                        plain,
                        LinPath::Gemm,
                        &format!("blk.{il}.attn_qkv.weight"),
                        xn,
                        s0,
                        t,
                    )?;
                    self.lin_forward(
                        plain,
                        LinPath::Gemm,
                        &format!("blk.{il}.attn_gate.weight"),
                        xn,
                        s1,
                        t,
                    )?;
                } else {
                    // [2026-10-09 P6] xh(=self.dx32.ptr)는 norm_resid_dev가 이미
                    // h2f(f2h(xn)) 융합 기록(norm.cu xn32 — cast_x32와 비트 동일
                    // 계약, 실측 근거 주석 포함). 종전 cast_x32 재계산은 중복
                    // 런치였다. 값 불변(골든 검증).
                    self.lin_forward(
                        plain,
                        LinPath::Gemm,
                        &format!("blk.{il}.attn_qkv.weight"),
                        xh,
                        s0,
                        t,
                    )?;
                    self.lin_forward(
                        plain,
                        LinPath::Gemm,
                        &format!("blk.{il}.attn_gate.weight"),
                        xh,
                        s1,
                        t,
                    )?;
                }
                let g = self
                    .gdn_chain_dev_run(slot, gi, t, xn, s0, s1)
                    .map_err(|e| format!("T{il} gdn: {e}"))?;
                gi += 1;
                g
            };
            let lo = if (il + 1) % interval == 0 {
                format!("blk.{il}.attn_output.weight")
            } else {
                format!("blk.{il}.ssm_out.weight")
            };
            if plain {
                self.lin_forward(plain, LinPath::Gemm, &lo, branch, dyt, t)?;
            } else {
                let (_, _, _, ko) = self.lin_spec(&lo)?;
                let xh2 = self.cast_x32(branch, t * ko)?;
                self.lin_forward(plain, LinPath::Gemm, &lo, xh2, dyt, t)?;
            }
            // [P10] cast_x32(t×ko)가 dx32를 재할당했을 수 있다 — 노름 융합
            // 기록(xn32)은 **현재** 포인터를 다시 확인한다. 옛 포인터를 계속
            // 쓰면 비행 커널이 해제 버퍼에 기록한다(새니타이저 OOB 실측).
            let xh = self.ensure_dx32(t * h)?;
            let xn2 = self
                .norm_resid_dev(2 * il + 1, self.dres, dyt, t, xh)
                .map_err(|e| format!("T{il} post norm: {e}"))?;
            if plain {
                ab = self
                    .moe_ffn_dev_t(il, xn2, t)
                    .map_err(|e| format!("T{il} moe: {e}"))?;
            } else {
                let xh3 = xh; // 노름 융합 기록
                let _ = xn2;
                self.lin_forward(
                    false,
                    LinPath::Gemm,
                    &format!("blk.{il}.ffn_gate.weight"),
                    xh3,
                    s0,
                    t,
                )?;
                self.lin_forward(
                    false,
                    LinPath::Gemm,
                    &format!("blk.{il}.ffn_up.weight"),
                    xh3,
                    s1,
                    t,
                )?;
                self.ew_dev(s0, s1, s2, t * w2)?;
                let dn = format!("blk.{il}.ffn_down.weight");
                let (_, _, _, kd) = self.lin_spec(&dn)?;
                let xh4 = self.cast_x32(s2, t * kd)?;
                self.lin_forward(false, LinPath::Gemm, &dn, xh4, s3, t)?;
                ab = s3;
            }
        }
        if all {
            // [A-1] 전 위치 최종 노름 → 배치 head → 배치 argmax → 토큰 t개.
            let xn_all = self
                .norm_resid_dev(2 * self.n_layers, self.dres, ab, t, 0)
                .map_err(|e| format!("S final norm: {e}"))?;
            self.ensure_batch_bufs()?;
            if logits && t > BATCH_DEC_MAX {
                // [eval] t>8 — head GEMV-T 상한(G4_TMAX) 초과: 행별 표준 GEMV
                // (행=블록 DRAM 포화 경로 — eval 전용).
                for r in 0..t {
                    self.head_gemv_launch(
                        xn_all + (r * self.hidden) as u64 * 4,
                        self.dbatch_lg + (r * self.head_n) as u64 * 4,
                    )?;
                }
            } else {
                self.head_gemv_t_launch(xn_all, self.dbatch_lg, t)?;
            }
            if logits {
                // [eval] 로짓 회수 모드 — argmax 대신 [t][head_n] d2h.
                // (행별 표준 GEMV 대안은 실측 더 느림 — head_gemv_t 유지.)
                let mut ob = vec![0u8; t * self.head_n * 4];
                self.cc
                    .d2h_async(ob.as_mut_ptr(), self.dbatch_lg, ob.len())?;
                self.cc.sync()?;
                let pos_after = pos + t as u32;
                self.slot_pos[slot] = pos_after;
                self.attn_set_pos(slot, pos_after)?;
                return Ok(ob
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect());
            }
            {
                let fa = self.cc.function("w4a16_argmax_min_t")?;
                let (mut p_l, mut p_n, mut p_t, mut p_a) =
                    (self.dbatch_lg, self.head_n as i32, t as i32, self.dbatch_am);
                self.cc.launch(
                    fa,
                    t as u32,
                    1,
                    1024,
                    &mut crate::rawcuda::args::l4(&mut p_l, &mut p_n, &mut p_t, &mut p_a),
                )?;
            }
            // SAFETY: self 소유 pinned 스크래치 단독 가변 접근(수명=self).
            let tb =
                unsafe { std::slice::from_raw_parts_mut(self.pin_batch_tok as *mut u8, t * 4) };
            self.cc.d2h_async(tb.as_mut_ptr(), self.dbatch_am, t * 4)?;
            self.cc.sync()?;
            let pos_after = pos + t as u32;
            self.slot_pos[slot] = pos_after;
            self.attn_set_pos(slot, pos_after)?;
            return Ok((0..t)
                .map(|k| {
                    u32::from_le_bytes([tb[k * 4], tb[k * 4 + 1], tb[k * 4 + 2], tb[k * 4 + 3]])
                        as f32
                })
                .collect());
        }
        // 마지막 행만 최종 노름(+head) — 중간 행 로짓은 불필요(상각).
        let last = (t - 1) as u64 * (h as u64) * 4;
        let xn_last = self
            .norm_resid_dev(2 * self.n_layers, self.dres + last, ab + last, 1, 0)
            .map_err(|e| format!("T final norm: {e}"))?;
        let mut ob = vec![0u8; if head { self.head_n * 4 } else { h * 4 }];
        if head {
            if self.head_w == 0 {
                return Err("chain_device_t: head 미등록".into());
            }
            self.head_gemv_launch(xn_last, self.head_out)?;
            // 동기 d2h 금지 — 그래프 캡처가 만든 커스텀(비차단) 스트림과
            // 경합한다(실측: serve 배치 프리필 쓰레기 토큰). 스트림 순서 복사.
            self.cc
                .d2h_async(ob.as_mut_ptr(), self.head_out, self.head_n * 4)?;
        } else {
            self.cc.d2h_async(ob.as_mut_ptr(), xn_last, h * 4)?;
        }
        self.cc.sync()?;
        let pos_after = pos + t as u32;
        self.slot_pos[slot] = pos_after;
        self.attn_set_pos(slot, pos_after)?;
        let v: Vec<f32> = ob
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        if head {
            llm170_diag::fp::fp_record("gpu.logits", &v);
        } else {
            llm170_diag::fp::fp_record("gpu.xn", &v);
        }
        Ok(v)
    }
}
