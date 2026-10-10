use super::*;

impl W4a16Dec {
    /// 1토큰 forward(호스트 스테이징) — 최종 노름 입력(head 입력)을 반환.
    /// KV·GDN 상태를 슬롯 위치만큼 전진시킨다.
    pub fn forward(&mut self, slot: usize, embed_row: &[f32]) -> Result<Vec<f32>, String> {
        let _g = self.cc.guard()?;
        if embed_row.len() != self.hidden || slot >= self.n_slots {
            return Err("forward: 임베딩 폭/슬롯 계약 위반".into());
        }
        let pos = self.slot_pos[slot];
        let cap = self.attn.ok_or("attn: 형상 미등록")?.cap;
        if pos as usize >= cap {
            return Err(format!(
                "context overflow: slot{slot} pos={pos} >= kvcap={cap}"
            ));
        }
        if self.norm_w_rows < 2 * self.n_layers + 1 || self.gdn.is_none() {
            return Err("forward: 디코더 상수 미등록".into());
        }
        let dres = self.cc.alloc(self.hidden * 4)?;
        let result = (|| {
            let row = unsafe {
                std::slice::from_raw_parts(embed_row.as_ptr() as *const u8, self.hidden * 4)
            };
            self.cc.h2d(dres, row)?;
            self.forward_resident(slot, pos, dres)
        })();
        let freed = self.cc.free(dres);
        match (result, freed) {
            (Err(e), _) => Err(e),
            (_, Err(e)) => Err(e),
            (Ok(v), Ok(())) => Ok(v),
        }
    }

    pub(super) fn forward_resident(
        &mut self,
        slot: usize,
        pos: u32,
        dres: CUdeviceptr,
    ) -> Result<Vec<f32>, String> {
        let mut ab = vec![0.0f32; self.hidden];
        let mut gi = 0usize;
        let interval = self.attn.map(|a| a.interval).unwrap_or(4);
        for il in 0..self.n_layers {
            let xn = self
                .norm_resid_staged(2 * il, dres, &ab)
                .map_err(|e| format!("L{il} input norm: {e}"))?;
            if il == 0 && self.debug_layers {
                eprintln!("  G0 xn[0..4]={:?}", &xn[..4]);
            }
            ab = if (il + 1) % interval == 0 {
                let ai = il / interval;
                let q = self
                    .gemv_host(&format!("blk.{il}.attn_q.weight"), &xn)
                    .map_err(|e| format!("L{il} q: {e}"))?;
                let k = self
                    .gemv_host(&format!("blk.{il}.attn_k.weight"), &xn)
                    .map_err(|e| format!("L{il} k: {e}"))?;
                let v = self
                    .gemv_host(&format!("blk.{il}.attn_v.weight"), &xn)
                    .map_err(|e| format!("L{il} v: {e}"))?;
                let out = self
                    .attn_chain_host(slot, ai, 1, &q, &k, &v, pos)
                    .map_err(|e| format!("L{il} attn: {e}"))?;
                self.gemv_host(&format!("blk.{il}.attn_output.weight"), &out)
                    .map_err(|e| format!("L{il} o: {e}"))?
            } else {
                let qkv = self
                    .gemv_host(&format!("blk.{il}.attn_qkv.weight"), &xn)
                    .map_err(|e| format!("L{il} qkv: {e}"))?;
                let z = self
                    .gemv_host(&format!("blk.{il}.attn_gate.weight"), &xn)
                    .map_err(|e| format!("L{il} z: {e}"))?;
                if il == 0 && self.debug_layers {
                    eprintln!("  G0 qkv[0..4]={:?}", &qkv[..4]);
                    eprintln!("  G0 z[0..4]={:?}", &z[..4]);
                }
                let gated = self
                    .gdn_chain_host(slot, gi, 1, &xn, &qkv, &z)
                    .map_err(|e| format!("L{il} gdn: {e}"))?;
                gi += 1;
                if il == 0 && self.debug_layers {
                    eprintln!("  G0 gated[0..4]={:?}", &gated[..4]);
                }
                let o = self
                    .gemv_host(&format!("blk.{il}.ssm_out.weight"), &gated)
                    .map_err(|e| format!("L{il} ssm_out: {e}"))?;
                if il == 0 && self.debug_layers {
                    eprintln!("  G0 out[0..4]={:?}", &o[..4]);
                }
                o
            };
            let xn = self
                .norm_resid_staged(2 * il + 1, dres, &ab)
                .map_err(|e| format!("L{il} post norm: {e}"))?;
            let gate = self
                .gemv_host(&format!("blk.{il}.ffn_gate.weight"), &xn)
                .map_err(|e| format!("L{il} gate: {e}"))?;
            let up = self
                .gemv_host(&format!("blk.{il}.ffn_up.weight"), &xn)
                .map_err(|e| format!("L{il} up: {e}"))?;
            let act = self
                .ew_host(&gate, &up)
                .map_err(|e| format!("L{il} ew: {e}"))?;
            ab = self
                .gemv_host(&format!("blk.{il}.ffn_down.weight"), &act)
                .map_err(|e| format!("L{il} down: {e}"))?;
            if self.debug_layers {
                let mut db = vec![0u8; self.hidden * 4];
                self.cc.d2h(&mut db, dres)?;
                self.cc.sync()?;
                let d: f64 = db
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c) as f64)
                    .sum();
                let a: f64 = ab.iter().map(|&v| v as f64).sum();
                eprintln!(
                    "  G{il:>2} recr={} sum={:.6}",
                    (il + 1) % interval != 0,
                    d + a
                );
            }
        }
        let xn = self
            .norm_resid_staged(2 * self.n_layers, dres, &ab)
            .map_err(|e| format!("final norm: {e}"))?;
        self.slot_pos[slot] = pos + 1;
        self.attn_set_pos(slot, pos + 1)?;
        Ok(xn)
    }

    /// 슬롯 상태 0화(GDN 링/스캔 + pp + pos).
    pub fn reset_state(&mut self, slot: usize) -> Result<(), String> {
        let _g = self.cc.guard()?;
        if slot >= self.n_slots {
            return Err(format!("reset: slot={slot} >= {}", self.n_slots));
        }
        self.slot_pos[slot] = 0;
        self.attn_set_pos(slot, 0)?;
        if let Some(dims) = self.gdn {
            let ring_bytes = dims.n_gdn * 3 * dims.conv_ch() * 4;
            let st_bytes = dims.n_gdn * dims.h_v * 128 * 128 * 4;
            let ring_off = (slot * dims.n_gdn * 3 * dims.conv_ch()) as u64 * 4;
            let st_off = (slot * dims.n_gdn * dims.h_v * 128 * 128) as u64 * 4;
            Self::zero_dev(&self.cc, self.dring + ring_off, ring_bytes)?;
            Self::zero_dev(&self.cc, self.dgst + st_off, st_bytes)?;
        }
        Ok(())
    }

    // ── 디바이스 체인(S10 — 왕복 제거) ──

    /// 1토큰 forward(디바이스 체인) — 왕복은 임베딩 업로드 1회 + 최종 xn
    /// 판독 1회뿐. 산술은 스테이징 경로와 같은 커널·같은 순서(층 4주기).
    pub(super) fn chain_device(
        &mut self,
        slot: usize,
        embed_row: &[f32],
    ) -> Result<CUdeviceptr, String> {
        let _g = self.cc.guard()?;
        if embed_row.len() != self.hidden || slot >= self.n_slots {
            return Err("forward_device: 임베딩 폭/슬롯 계약 위반".into());
        }
        let pos = self.slot_pos[slot];
        let cap = self.attn.ok_or("attn: 형상 미등록")?.cap;
        if pos as usize >= cap {
            return Err(format!(
                "context overflow: slot{slot} pos={pos} >= kvcap={cap}"
            ));
        }
        if self.norm_w_rows < 2 * self.n_layers + 1 || self.gdn.is_none() {
            return Err("forward_device: 디코더 상수 미등록".into());
        }
        self.ensure_chain_bufs()?;
        let row = if self.capture_pinned_src {
            // 캡처 중: pageable async 복사는 캡처 불가 — pinned 버퍼를 소스로
            // 기록하고 replay가 실행 직전에 내용을 채운다.
            unsafe { std::slice::from_raw_parts(self.pin_embed as *const u8, self.hidden * 4) }
        } else {
            unsafe { std::slice::from_raw_parts(embed_row.as_ptr() as *const u8, self.hidden * 4) }
        };
        self.cc.h2d_async(self.dres, row)?;
        let [s0, s1, s1b, s2, s3] = self.dchain;
        let (w0, w1, w2) = (self.stg_w0, self.stg_w1, self.stg_w2);
        let mut ab = self.dab_dev;
        let mut gi = 0usize;
        // 플레인 모드(MoE 모델) — GEMV는 bf16(head_bf16), x는 원시 f32
        // (h2f 왕복 없음 — CPU 플레인 matmul 계약과 동일). FFN은 MoE.
        let plain = self.plain_weights;
        for il in 0..self.n_layers {
            // 노름이 x32를 융합 기록(cast_x32 노드 제거) — q/k/v(또는 qkv/z) 공유.
            let x32 = if plain {
                0
            } else {
                self.ensure_dx32(self.hidden)?
            };
            let xn = self
                .norm_resid_dev(2 * il, self.dres, ab, 1, x32)
                .map_err(|e| format!("L{il} input norm: {e}"))?;
            let interval = self.attn.ok_or("attn: 형상 미등록")?.interval;
            let branch = if (il + 1) % interval == 0 {
                if plain {
                    self.plain_stage_x32(&format!("blk.{il}.attn_q.weight"), xn, s0, w0)
                        .map_err(|e| format!("L{il} q: {e}"))?;
                    self.plain_stage_x32(&format!("blk.{il}.attn_k.weight"), xn, s1, w1)
                        .map_err(|e| format!("L{il} k: {e}"))?;
                    self.plain_stage_x32(&format!("blk.{il}.attn_v.weight"), xn, s1b, w1)
                        .map_err(|e| format!("L{il} v: {e}"))?;
                } else {
                    self.gemv_stage_x32(&format!("blk.{il}.attn_q.weight"), x32, s0, w0)
                        .map_err(|e| format!("L{il} q: {e}"))?;
                    self.gemv_stage_x32(&format!("blk.{il}.attn_k.weight"), x32, s1, w1)
                        .map_err(|e| format!("L{il} k: {e}"))?;
                    self.gemv_stage_x32(&format!("blk.{il}.attn_v.weight"), x32, s1b, w1)
                        .map_err(|e| format!("L{il} v: {e}"))?;
                }
                self.attn_chain_dev_run(slot, il / interval, 1, s0, s1, s1b)
                    .map_err(|e| format!("L{il} attn: {e}"))?
            } else {
                if plain {
                    self.plain_stage_x32(&format!("blk.{il}.attn_qkv.weight"), xn, s0, w0)
                        .map_err(|e| format!("L{il} qkv: {e}"))?;
                    self.plain_stage_x32(&format!("blk.{il}.attn_gate.weight"), xn, s1, w1)
                        .map_err(|e| format!("L{il} z: {e}"))?;
                } else {
                    self.gemv_stage_x32(&format!("blk.{il}.attn_qkv.weight"), x32, s0, w0)
                        .map_err(|e| format!("L{il} qkv: {e}"))?;
                    self.gemv_stage_x32(&format!("blk.{il}.attn_gate.weight"), x32, s1, w1)
                        .map_err(|e| format!("L{il} z: {e}"))?;
                }
                let g = self
                    .gdn_chain_dev_run(slot, gi, 1, xn, s0, s1)
                    .map_err(|e| format!("L{il} gdn: {e}"))?;
                gi += 1;
                g
            };
            let lo = if (il + 1) % interval == 0 {
                format!("blk.{il}.attn_output.weight")
            } else {
                format!("blk.{il}.ssm_out.weight")
            };
            let out = if plain {
                self.plain_gemv_dev(&lo, branch)
                    .map_err(|e| format!("L{il} {lo}: {e}"))?
            } else {
                let (_, _, _, ko) = self.lin_spec(&lo)?;
                let x32b = self
                    .cast_x32(branch, ko)
                    .map_err(|e| format!("L{il} branch cast: {e}"))?;
                self.gemv_dev_x32(&lo, x32b)
                    .map_err(|e| format!("L{il} {lo}: {e}"))?
            };
            // [P10] cast_x32(ko) 재할당 대비 — 현재 dx32 재확인(위 주석 참조).
            let x32 = self.ensure_dx32(self.hidden)?;
            let xn2 = self
                .norm_resid_dev(2 * il + 1, self.dres, out, 1, x32)
                .map_err(|e| format!("L{il} post norm: {e}"))?;
            if plain {
                // MoE FFN(35B-A3B) — 잔차 ab = MoE 출력(dmo).
                ab = self
                    .moe_ffn_dev(il, xn2)
                    .map_err(|e| format!("L{il} moe: {e}"))?;
            } else {
                let x32n = x32; // 노름 융합 기록
                let _ = xn2;
                self.gemv_stage_x32(&format!("blk.{il}.ffn_gate.weight"), x32n, s0, w0)
                    .map_err(|e| format!("L{il} gate: {e}"))?;
                self.gemv_stage_x32(&format!("blk.{il}.ffn_up.weight"), x32n, s1, w1)
                    .map_err(|e| format!("L{il} up: {e}"))?;
                self.ew_dev(s0, s1, s2, w2)?;
                let dn = format!("blk.{il}.ffn_down.weight");
                let (_, _, _, kd) = self.lin_spec(&dn)?;
                let x32d = self
                    .cast_x32(s2, kd)
                    .map_err(|e| format!("L{il} down cast: {e}"))?;
                // down은 s3 직접 쓰기 — dy 경유 d2d 제거.
                self.gemv_launch(&dn, x32d, s3)
                    .map_err(|e| format!("L{il} down: {e}"))?;
                ab = s3;
            }
            if self.debug_layers {
                let mut db = vec![0u8; self.hidden * 4];
                let mut abv = vec![0u8; self.hidden * 4];
                self.cc.d2h(&mut db, self.dres)?;
                self.cc.d2h(&mut abv, ab)?;
                self.cc.sync()?;
                let d: f64 = db
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c) as f64)
                    .sum();
                let a: f64 = abv
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c) as f64)
                    .sum();
                eprintln!("  G{il:>2} recr={} sum={:.6}", (il + 1) % 4 != 0, d + a);
            }
        }
        let xn_final = self
            .norm_resid_dev(2 * self.n_layers, self.dres, ab, 1, 0)
            .map_err(|e| format!("final norm: {e}"))?;
        Ok(xn_final)
    }

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
        if t > 32 && !gemm_mma {
            return Err(format!(
                "chain_device_t: t={t} > 32 — FFMA 폴백 상한(TC=0 진단 또는 t<16)"
            ));
        }
        for il in 0..self.n_layers {
            let xh = if plain { 0 } else { self.ensure_dx32(t * h)? };
            let xn = self
                .norm_resid_dev(2 * il, self.dres, ab, t, xh)
                .map_err(|e| format!("T{il} input norm: {e}"))?;
            let interval = self.attn.ok_or("attn: 형상 미등록")?.interval;
            let branch = if (il + 1) % interval == 0 {
                if plain {
                    self.plain_gemm_launch(&format!("blk.{il}.attn_q.weight"), xn, s0, t)?;
                    self.plain_gemm_launch(&format!("blk.{il}.attn_k.weight"), xn, s1, t)?;
                    self.plain_gemm_launch(&format!("blk.{il}.attn_v.weight"), xn, s1b, t)?;
                } else {
                    self.gemm_launch(&format!("blk.{il}.attn_q.weight"), xh, s0, t)?;
                    self.gemm_launch(&format!("blk.{il}.attn_k.weight"), xh, s1, t)?;
                    self.gemm_launch(&format!("blk.{il}.attn_v.weight"), xh, s1b, t)?;
                }
                self.attn_chain_dev_run(slot, il / interval, t, s0, s1, s1b)
                    .map_err(|e| format!("T{il} attn: {e}"))?
            } else {
                if plain {
                    self.plain_gemm_launch(&format!("blk.{il}.attn_qkv.weight"), xn, s0, t)?;
                    self.plain_gemm_launch(&format!("blk.{il}.attn_gate.weight"), xn, s1, t)?;
                } else {
                    // [2026-10-09 P6] xh(=self.dx32)는 norm_resid_dev가 이미
                    // h2f(f2h(xn)) 융합 기록(norm.cu xn32 — cast_x32와 비트 동일
                    // 계약, 실측 근거 주석 포함). 종전 cast_x32 재계산은 중복
                    // 런치였다. 값 불변(골든 검증).
                    self.gemm_launch(&format!("blk.{il}.attn_qkv.weight"), xh, s0, t)?;
                    self.gemm_launch(&format!("blk.{il}.attn_gate.weight"), xh, s1, t)?;
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
                self.plain_gemm_launch(&lo, branch, dyt, t)?;
            } else {
                let (_, _, _, ko) = self.lin_spec(&lo)?;
                let xh2 = self.cast_x32(branch, t * ko)?;
                self.gemm_launch(&lo, xh2, dyt, t)?;
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
                self.gemm_launch(&format!("blk.{il}.ffn_gate.weight"), xh3, s0, t)?;
                self.gemm_launch(&format!("blk.{il}.ffn_up.weight"), xh3, s1, t)?;
                self.ew_dev(s0, s1, s2, t * w2)?;
                let dn = format!("blk.{il}.ffn_down.weight");
                let (_, _, _, kd) = self.lin_spec(&dn)?;
                let xh4 = self.cast_x32(s2, t * kd)?;
                self.gemm_launch(&dn, xh4, s3, t)?;
                ab = s3;
            }
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
            let f = self.cc.function("head_bf16")?;
            let (mut p_w, mut p_x, mut p_o) = (self.head_w, xn_last, self.head_out);
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

    /// [A9 2026-10-10] 배치 디코드 단일 체인 — 슬롯별 1토큰(t=n_active)을
    /// 한 번에 통과한다. 가중 커널(GEMM·MoE·head)은 t행 배치로 상각하고,
    /// 상태 커널(attn KV/pos·GDN 링/트리오)은 토큰별 슬롯 자원을 쓴다.
    /// 반환: 슬롯 순서의 다음 토큰(argmax, greedy 전용).
    /// 전제(호출부 가드): t∈2..=BATCH_DEC_MAX · 전 슬롯 greedy · !kvq ·
    /// MoE면 상주 모드 · head/argmax 등록. 그래프 밖(직접 발사) 경로.
    /// [A9] 배치 체인 1회 발사 — 행/pos는 핀드(pin_batch_in/pos)에서 읽어
    /// h2d 노드로 기록된다(캡처 가능). sync·slot_pos 갱신은 호출부 소관.
    pub(super) fn batch_launch(&mut self, slots: &[usize], t: usize) -> Result<(), String> {
        if self.norm_w_rows < 2 * self.n_layers + 1 || self.gdn.is_none() {
            return Err("batch_launch: 디코더 상수 미등록".into());
        }
        self.ensure_chain_bufs()?;
        self.ensure_norm_bufs(t)?;
        self.ensure_gdn_bufs(t)?;
        self.ensure_attn_bufs(t)?;
        self.ensure_batch_bufs()?;
        let dyt = self.ensure_dyt(t)?;
        {
            let rb = unsafe {
                std::slice::from_raw_parts(self.pin_batch_in as *const u8, t * self.hidden * 4)
            };
            self.cc.h2d_async(self.dres, rb)?;
            let pb = unsafe {
                std::slice::from_raw_parts(self.pin_batch_pos as *const u8, self.n_slots * 4)
            };
            self.cc.h2d_async(self.dpp, pb)?;
        }
        let [s0, s1, s1b, s2, s3] = self.dchain;
        let (w2, h) = (self.stg_w2, self.hidden);
        let mut ab = self.dab_dev;
        let mut gi = 0usize;
        let plain = self.plain_weights;
        if t > 32 {
            return Err(format!("chain_device_batch: t={t} > 32 FFMA 상한"));
        }
        for il in 0..self.n_layers {
            let xh = if plain { 0 } else { self.ensure_dx32(t * h)? };
            let xn = self
                .norm_resid_dev(2 * il, self.dres, ab, t, xh)
                .map_err(|e| format!("B{il} input norm: {e}"))?;
            let interval = self.attn.ok_or("attn: 형상 미등록")?.interval;
            let branch = if (il + 1) % interval == 0 {
                if plain {
                    self.plain_gemv_t_launch(&format!("blk.{il}.attn_q.weight"), xn, s0, t)?;
                    self.plain_gemv_t_launch(&format!("blk.{il}.attn_k.weight"), xn, s1, t)?;
                    self.plain_gemv_t_launch(&format!("blk.{il}.attn_v.weight"), xn, s1b, t)?;
                } else {
                    self.gemv_t_launch(&format!("blk.{il}.attn_q.weight"), xh, s0, t)?;
                    self.gemv_t_launch(&format!("blk.{il}.attn_k.weight"), xh, s1, t)?;
                    self.gemv_t_launch(&format!("blk.{il}.attn_v.weight"), xh, s1b, t)?;
                }
                self.attn_chain_dev_batch(slots, il / interval, t, s0, s1, s1b)
                    .map_err(|e| format!("B{il} attn: {e}"))?
            } else {
                if plain {
                    self.plain_gemv_t_launch(&format!("blk.{il}.attn_qkv.weight"), xn, s0, t)?;
                    self.plain_gemv_t_launch(&format!("blk.{il}.attn_gate.weight"), xn, s1, t)?;
                } else {
                    self.gemv_t_launch(&format!("blk.{il}.attn_qkv.weight"), xh, s0, t)?;
                    self.gemv_t_launch(&format!("blk.{il}.attn_gate.weight"), xh, s1, t)?;
                }
                let g = self
                    .gdn_chain_dev_batch(slots, gi, t, xn, s0, s1)
                    .map_err(|e| format!("B{il} gdn: {e}"))?;
                gi += 1;
                g
            };
            let lo = if (il + 1) % interval == 0 {
                format!("blk.{il}.attn_output.weight")
            } else {
                format!("blk.{il}.ssm_out.weight")
            };
            if plain {
                self.plain_gemv_t_launch(&lo, branch, dyt, t)?;
            } else {
                let (_, _, _, ko) = self.lin_spec(&lo)?;
                let xh2 = self.cast_x32(branch, t * ko)?;
                self.gemv_t_launch(&lo, xh2, dyt, t)?;
            }
            let xh = self.ensure_dx32(t * h)?;
            let xn2 = self
                .norm_resid_dev(2 * il + 1, self.dres, dyt, t, xh)
                .map_err(|e| format!("B{il} post norm: {e}"))?;
            if plain {
                ab = self
                    .moe_ffn_dev_t(il, xn2, t)
                    .map_err(|e| format!("B{il} moe: {e}"))?;
            } else {
                let xh3 = xh;
                let _ = xn2;
                self.gemv_t_launch(&format!("blk.{il}.ffn_gate.weight"), xh3, s0, t)?;
                self.gemv_t_launch(&format!("blk.{il}.ffn_up.weight"), xh3, s1, t)?;
                self.ew_dev(s0, s1, s2, t * w2)?;
                let dn = format!("blk.{il}.ffn_down.weight");
                let (_, _, _, kd) = self.lin_spec(&dn)?;
                let xh4 = self.cast_x32(s2, t * kd)?;
                self.gemv_t_launch(&dn, xh4, s3, t)?;
                ab = s3;
            }
        }
        // 전 행 최종 노름 → 배치 head(가중 판독 상각) → 배치 argmax → 1회 d2h.
        let xn_all = self
            .norm_resid_dev(2 * self.n_layers, self.dres, ab, t, 0)
            .map_err(|e| format!("B final norm: {e}"))?;
        {
            let f = self.cc.function("head_bf16_t")?;
            let (mut p_w, mut p_x, mut p_o) = (self.head_w, xn_all, self.dbatch_lg);
            let (mut p_n, mut p_k, mut p_t) = (self.head_n as i32, self.head_k as i32, t as i32);
            let mut args: [*mut std::ffi::c_void; 6] = [
                (&mut p_w) as *mut _ as *mut _,
                (&mut p_x) as *mut _ as *mut _,
                (&mut p_o) as *mut _ as *mut _,
                (&mut p_n) as *mut _ as *mut _,
                (&mut p_k) as *mut _ as *mut _,
                (&mut p_t) as *mut _ as *mut _,
            ];
            self.cc
                .launch(f, self.head_n.div_ceil(4 * 256) as u32, 1, 256, &mut args)?;
        }
        {
            let fa = self.cc.function("w4a16_argmax_min_t")?;
            let (mut p_l, mut p_n, mut p_t, mut p_a) =
                (self.dbatch_lg, self.head_n as i32, t as i32, self.dbatch_am);
            let mut aa: [*mut std::ffi::c_void; 4] = [
                (&mut p_l) as *mut _ as *mut _,
                (&mut p_n) as *mut _ as *mut _,
                (&mut p_t) as *mut _ as *mut _,
                (&mut p_a) as *mut _ as *mut _,
            ];
            self.cc.launch(fa, t as u32, 1, 1024, &mut aa)?;
        }
        let tb = unsafe { std::slice::from_raw_parts_mut(self.pin_batch_tok as *mut u8, t * 4) };
        self.cc.d2h_async(tb.as_mut_ptr(), self.dbatch_am, t * 4)?;
        Ok(())
    }

    /// [A9] 핀드 토큰 판독(sync 후).
    pub(super) fn batch_read_tokens(&self, t: usize) -> Vec<u32> {
        let tb = unsafe { std::slice::from_raw_parts(self.pin_batch_tok as *const u8, t * 4) };
        (0..t)
            .map(|k| u32::from_le_bytes([tb[k * 4], tb[k * 4 + 1], tb[k * 4 + 2], tb[k * 4 + 3]]))
            .collect()
    }
}
