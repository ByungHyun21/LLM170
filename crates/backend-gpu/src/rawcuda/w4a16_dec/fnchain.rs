//! [FN M1 2026-10-20] Qwen3.8-Flash-Next 디바이스 체인 — 하이퍼커넥션·PLE·
//! GDN(sigmoid 게이트)·풀(dense)·MoE(CT g32 마린) 48층 루프.
//! 층 순서 = CPU 오라클(oracle.rs forward)과 1:1 — PLE(멀티스트림 직접 가산)
//! → HC mix → 어텐션(GDN|full) → HC write → HC mix → MoE → HC write →
//! 최종 mixer(use_combine=false) → head.
//! M1 판정 = 속도 지도(풀층 dense — 인덱서 미구현으로 골든 불가·오라클
//! 수식 미러는 블록 단위 단위테스트로). 수치 클래스: 체인 f32(커널 주석 참조).

use super::*;

impl W4a16Dec {
    /// FN 체인 등록 — hc_norm 가중(행 = attn(il)·mlp(il)×L + mixer, 각
    /// [hc·hidden])·PLE 가중(선택)·GDN sigmoid 게이트. 상태(PLE conv)는
    /// 슬롯수만큼 할당. 호출 후 `fn_.on = true`.
    pub fn set_fn(
        &mut self,
        hc: usize,
        lowrank: usize,
        eps: f32,
        n_layers: usize,
        hcn: &[f32],
        ple: Option<(usize, &[f32], &[f32], &[f32], &[f32])>,
        sig_gate: bool,
    ) -> Result<(), String> {
        let _g = self.cc.guard()?;
        let hh = hc * self.hidden;
        let rows = 2 * n_layers + 1;
        if hc == 0 || hcn.len() != rows * hh {
            return Err(format!("set_fn: hcn {} != {rows}×{hh}", hcn.len()));
        }
        if let Some((_, w, nk, nq, nc)) = &ple {
            let n = hc * self.hidden;
            if w.len() != n * 4 || nk.len() != n || nq.len() != n || nc.len() != n {
                return Err("set_fn: PLE 가중 형상 위반(w[n*4]·nk/nq/nc[n])".into());
            }
        }
        let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        self.fn_.hc = hc;
        self.fn_.lowrank = lowrank;
        self.fn_.eps = eps;
        self.fn_.ple_il = ple.as_ref().map(|(il, ..)| *il);
        let d = self.cc.alloc(hcn.len() * 4)?;
        self.cc.h2d(d, &f32b(hcn))?;
        self.fn_.d_hcn = d;
        if let Some((_, w, nk, nq, nc)) = ple {
            let dw = self.cc.alloc(w.len() * 4)?;
            self.cc.h2d(dw, &f32b(w))?;
            self.fn_.d_ple_w = dw;
            for (dst, src) in [
                (&mut self.fn_.d_ple_nk, nk),
                (&mut self.fn_.d_ple_nq, nq),
                (&mut self.fn_.d_ple_nc, nc),
            ] {
                let q = self.cc.alloc(src.len() * 4)?;
                self.cc.h2d(q, &f32b(src))?;
                *dst = q;
            }
            // PLE conv 상태 [슬롯][9][hc·hidden] f32 0 초기화.
            let bytes = self.n_slots * 9 * hh * 4;
            let st = self.cc.alloc(bytes)?;
            self.cc.memset0_async(st, bytes)?;
            self.fn_.d_ple_st = st;
            self.fn_.ple_init = vec![false; self.n_slots];
        }
        self.gdn.sig_gate = sig_gate;
        self.fn_.on = true;
        Ok(())
    }

    /// FN 스크래치 확보(t행 기준 — 재할당 패턴).
    fn fn_ensure_bufs(&mut self, t: usize) -> Result<(), String> {
        if self.fn_.warm_t >= t {
            return Ok(());
        }
        let (h, hh, hc, lr) = (
            self.hidden,
            self.fn_.hc * self.hidden,
            self.fn_.hc,
            self.fn_.lowrank,
        );
        let alloc = |b: &mut Buf, bytes: usize| -> Result<(), String> {
            if b.cap < bytes {
                if b.ptr != 0 {
                    self.cc.free(b.ptr)?;
                }
                b.ptr = self.cc.alloc(bytes)?;
                b.cap = bytes;
            }
            Ok(())
        };
        let f32n = t * hh; // R·xn·g·ple_k·ple_gd·ple_nm
        alloc(&mut self.fn_.r, f32n * 4)?;
        alloc(&mut self.fn_.xn, f32n * 4)?;
        alloc(&mut self.fn_.g, f32n * 4)?;
        alloc(&mut self.fn_.x, t * h * 4)?;
        alloc(&mut self.fn_.lora, t * lr * 4)?;
        alloc(&mut self.fn_.y, t * h * 4)?;
        alloc(&mut self.fn_.inj, t * hc * 4)?;
        alloc(&mut self.fn_.ple_e, t * h * 4)?;
        alloc(&mut self.fn_.ple_k, f32n * 4)?;
        alloc(&mut self.fn_.ple_v, t * h * 4)?;
        alloc(&mut self.fn_.ple_gd, f32n * 4)?;
        alloc(&mut self.fn_.ple_nm, f32n * 4)?;
        self.fn_.warm_t = t;
        Ok(())
    }

    /// HC mix 한 사이클 — norm(행 w_row) → lora(W_down) → silu → g(W_up) →
    /// mix(x) + inj(W_inj 원시). 출력: x(self.fn_.x)·inj(self.fn_.inj).
    /// 계약: lnm("blk.N.hc_dn")·lup("blk.N.hc_up")·linj("blk.N.hc_inj")
    /// plain 선형 등록([lr,hh]·[hh,lr]·[hc,hh]).
    fn fn_hc_cycle(
        &mut self,
        w_row: usize,
        t: usize,
        lnm: &str,
        lup: &str,
        linj: &str,
    ) -> Result<(), String> {
        let (hh, eps) = (self.fn_.hc * self.hidden, self.fn_.eps);
        let f = self.cc.function("fn_hc_norm")?;
        let (mut a0, mut a1, mut a2) = (
            self.fn_.r.ptr,
            self.fn_.d_hcn + (w_row * hh * 4) as u64,
            self.fn_.xn,
        );
        let (mut tl, mut ep) = (t as i32, eps);
        self.cc.launch(
            f,
            t as u32,
            1,
            256,
            &mut crate::rawcuda::args::l5(&mut a0, &mut a1, &mut a2, &mut tl, &mut ep),
        )?;
        // lora = W_down·xn — plain GEMM(t행).
        self.lin_forward(
            true,
            LinPath::Gemm,
            lnm,
            self.fn_.xn.ptr,
            self.fn_.lora.ptr,
            t,
        )?;
        let f = self.cc.function("fn_hc_silu")?;
        let n = t * self.fn_.lowrank;
        let (mut a0, mut a1, mut nn) = (self.fn_.lora.ptr, self.fn_.lora.ptr, n as i32);
        self.cc.launch(
            f,
            n.div_ceil(256) as u32,
            1,
            256,
            &mut crate::rawcuda::args::l3(&mut a0, &mut a1, &mut nn),
        )?;
        // g = W_up·silu(lora) → [t][hh].
        self.lin_forward(
            true,
            LinPath::Gemm,
            lup,
            self.fn_.lora.ptr,
            self.fn_.g.ptr,
            t,
        )?;
        // x = mix(xn, g) — 그리드 (t, hidden/256).
        let f = self.cc.function("fn_hc_mix")?;
        let (mut b0, mut b1, mut b2) = (self.fn_.xn.ptr, self.fn_.g.ptr, self.fn_.x);
        self.cc.launch(
            f,
            t as u32,
            self.hidden.div_ceil(256) as u32,
            256,
            &mut crate::rawcuda::args::l3(&mut b0, &mut b1, &mut b2),
        )?;
        // inj = W_inj·xn — [t][hc](원시 — hc_write에서 스케일).
        self.lin_forward(
            true,
            LinPath::Gemm,
            linj,
            self.fn_.xn.ptr,
            self.fn_.inj.ptr,
            t,
        )
    }

    /// HC 쓰기 — R += y·2σ(inj/4). fn_.y·fn_.inj 소비.
    fn fn_hc_write(&mut self, t: usize) -> Result<(), String> {
        let f = self.cc.function("fn_hc_write")?;
        let (mut a0, mut a1, mut a2) = (self.fn_.r.ptr, self.fn_.y.ptr, self.fn_.inj);
        let mut tl = t as i32;
        self.cc.launch(
            f,
            t as u32,
            self.fn_.hc as u32,
            256,
            &mut crate::rawcuda::args::l4(&mut a0, &mut a1, &mut a2, &mut tl),
        )
    }

    /// FN 청크 forward — d_emb에 [t][hidden] f32 임베딩 행이 이미 기록된
    /// 상태 진입(호출부 h2d). PLE 스테이징(ple_e_dev)도 호출부가 준비
    /// (호스트 ngram 게더 → h2d). pos==0이면 R 초기화, 이후 R 지속.
    /// 반환 = 마지막 행 sample [hidden] f32(head 입력).
    pub fn fn_chain_dev_t(
        &mut self,
        slot: usize,
        pos: u32,
        t: usize,
        d_emb: CUdeviceptr,
        ple_e_dev: Option<CUdeviceptr>,
    ) -> Result<Vec<f32>, String> {
        let _g = self.cc.guard()?;
        if !self.fn_.on {
            return Err("fn_chain: set_fn 미등록".into());
        }
        self.fn_ensure_bufs(t)?;
        let (h, hh, hc) = (self.hidden, self.fn_.hc * self.hidden, self.fn_.hc);
        if pos == 0 {
            // R 초기화 — 임베딩 행을 hc 스트림으로 복제.
            let f = self.cc.function("fn_hc_init")?;
            let n = t * hh;
            let (mut a0, mut a1, mut nn) = (d_emb, self.fn_.r.ptr, n as i32);
            self.cc.launch(
                f,
                n.div_ceil(256) as u32,
                1,
                256,
                &mut crate::rawcuda::args::l3(&mut a0, &mut a1, &mut nn),
            )?;
        }
        let interval = self.attn.attn.ok_or("fn_chain: attn 미등록")?.interval;
        let n_layers = self.n_layers;
        let mut gi = 0usize;
        for il in 0..n_layers {
            // PLE — HC mix 앞에서 멀티스트림 상태에 직접 가산.
            if self.fn_.ple_il == Some(il) {
                let pe = ple_e_dev.ok_or("fn_chain: PLE 스테이징 부재")?;
                // key/value = kv_proj(ple_e) — [hh + h] 팩 GEMM(ple_k 단일 버퍼).
                self.lin_forward(
                    true,
                    LinPath::Gemm,
                    &format!("blk.{il}.ple_kv"),
                    pe,
                    self.fn_.ple_k.ptr,
                    t,
                )?;
                let f = self.cc.function("fn_ple_gate")?;
                let (mut a0, mut a1) =
                    (self.fn_.ple_k.ptr, self.fn_.ple_k.ptr + (t * hh * 4) as u64);
                let (mut a2, mut a3, mut a4, mut a5, mut a6, mut a7) = (
                    self.fn_.r.ptr,
                    self.fn_.d_ple_nk,
                    self.fn_.d_ple_nq,
                    self.fn_.d_ple_nc,
                    self.fn_.ple_gd.ptr,
                    self.fn_.ple_nm.ptr,
                );
                let (mut tl, mut ep) = (t as i32, self.fn_.eps);
                self.cc.launch(
                    f,
                    t as u32,
                    hc as u32,
                    256,
                    &mut crate::rawcuda::args::l10(
                        &mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut a5, &mut a6, &mut a7,
                        &mut tl, &mut ep,
                    ),
                )?;
                // conv + 잔차 가산 + 상태 갱신.
                let has = self.fn_.ple_init.get(slot).copied().unwrap_or(false);
                let f = self.cc.function("fn_ple_conv")?;
                let (mut c0, mut c1, mut c2, mut c3, mut c4, mut c5, mut c6) = (
                    self.fn_.ple_nm.ptr,
                    self.fn_.ple_gd.ptr,
                    self.fn_.r.ptr,
                    self.fn_.d_ple_w,
                    self.fn_.d_ple_st + (slot * 9 * hh * 4) as u64,
                    t as i32,
                    has as i32,
                );
                self.cc.launch(
                    f,
                    t as u32,
                    hh.div_ceil(256) as u32,
                    256,
                    &mut crate::rawcuda::args::l7(
                        &mut c0, &mut c1, &mut c2, &mut c3, &mut c4, &mut c5, &mut c6,
                    ),
                )?;
                let f = self.cc.function("fn_ple_conv_wb")?;
                let (mut w0, mut w1, mut w2, mut w3) = (
                    self.fn_.ple_nm.ptr,
                    self.fn_.d_ple_st + (slot * 9 * hh * 4) as u64,
                    t as i32,
                    has as i32,
                );
                self.cc.launch(
                    f,
                    1,
                    hh.div_ceil(256) as u32,
                    256,
                    &mut crate::rawcuda::args::l4(&mut w0, &mut w1, &mut w2, &mut w3),
                )?;
                self.fn_.ple_init[slot] = true;
            }
            // ── 어텐션 블록: HC mix → (GDN | full) → out_proj → HC write.
            self.fn_hc_cycle(
                2 * il,
                t,
                &format!("blk.{il}.hc_dn_a"),
                &format!("blk.{il}.hc_up_a"),
                &format!("blk.{il}.hc_inj_a"),
            )?;
            let is_full = (il + 1).is_multiple_of(interval);
            let branch = if is_full {
                self.lin_forward(
                    true,
                    LinPath::Gemm,
                    &format!("blk.{il}.attn_q"),
                    self.fn_.x.ptr,
                    self.dchain[0],
                    t,
                )?;
                self.lin_forward(
                    true,
                    LinPath::Gemm,
                    &format!("blk.{il}.attn_k"),
                    self.fn_.x.ptr,
                    self.dchain[1],
                    t,
                )?;
                self.lin_forward(
                    true,
                    LinPath::Gemm,
                    &format!("blk.{il}.attn_v"),
                    self.fn_.x.ptr,
                    self.dchain[2],
                    t,
                )?;
                self.attn_chain_dev_run(
                    slot,
                    il / interval,
                    t,
                    self.dchain[0],
                    self.dchain[1],
                    self.dchain[2],
                )?
            } else {
                self.lin_forward(
                    true,
                    LinPath::Gemm,
                    &format!("blk.{il}.in_qkv"),
                    self.fn_.x.ptr,
                    self.dchain[0],
                    t,
                )?;
                self.lin_forward(
                    true,
                    LinPath::Gemm,
                    &format!("blk.{il}.in_z"),
                    self.fn_.x.ptr,
                    self.dchain[1],
                    t,
                )?;
                let g = self.gdn_chain_dev_run(
                    slot,
                    gi,
                    t,
                    self.fn_.x.ptr,
                    self.dchain[0],
                    self.dchain[1],
                )?;
                gi += 1;
                g
            };
            let lo = if is_full {
                format!("blk.{il}.attn_o")
            } else {
                format!("blk.{il}.ssm_out")
            };
            self.lin_forward(true, LinPath::Gemm, &lo, branch, self.fn_.y.ptr, t)?;
            self.fn_hc_write(t)?;
            // ── MoE 블록: HC mix → MoE → HC write.
            self.fn_hc_cycle(
                2 * il + 1,
                t,
                &format!("blk.{il}.hc_dn_m"),
                &format!("blk.{il}.hc_up_m"),
                &format!("blk.{il}.hc_inj_m"),
            )?;
            let y2 = self.moe_ffn_dev_t(il, self.fn_.x.ptr, t)?;
            // moe 출력 → y 버퍼 d2d(프로브급 — 이후 커널 직접 소비로 최적화).
            self.cc.d2d(self.fn_.y.ptr, y2, t * h * 4)?;
            self.fn_hc_write(t)?;
        }
        // ── 최종 mixer — norm(행 2L) + mix(head 입력 sample).
        self.fn_hc_cycle(
            2 * n_layers,
            t,
            "blk.final_hc_dn",
            "blk.final_hc_up",
            "blk.final_hc_inj",
        )?;
        // 마지막 행 sample 회수.
        let mut ob = vec![0u8; h * 4];
        self.cc
            .d2h(&mut ob, self.fn_.x.ptr + ((t - 1) * h * 4) as u64)?;
        // SAFETY: ob는 h*4 바이트(정렬 4) — f32 뷰 재해석.
        let out: Vec<f32> =
            unsafe { std::slice::from_raw_parts(ob.as_ptr() as *const f32, h).to_vec() };
        Ok(out)
    }
}
