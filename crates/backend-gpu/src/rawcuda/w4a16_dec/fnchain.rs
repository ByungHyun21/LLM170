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
        idx_heads: usize,
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
        self.fn_.idx_heads = idx_heads;
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
        // [QSA] 선택 리스트 상한 = 청크 t × (예산 블록×ratio + ratio) 최악.
        let sel_max = t * (self.fn_.idx_heads * 512 + 8);
        alloc(&mut self.fn_.sel, sel_max * 4)?;
        alloc(&mut self.fn_.nsel, t * 4)?;
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

    /// 청크 입력 스테이징 — emb[t][hidden]·ple[t][hidden] f32 h2d.
    /// 반환 = (d_emb, Some(d_ple)) — fn_chain_dev_t 인자.
    pub fn fn_stage(
        &mut self,
        emb: &[f32],
        ple: Option<&[f32]>,
        t: usize,
    ) -> Result<(CUdeviceptr, Option<CUdeviceptr>), String> {
        let _g = self.cc.guard()?;
        self.fn_ensure_bufs(t)?;
        let h = self.hidden;
        // SAFETY: emb/ple 로컬 슬라이스 유효 수명 내 바이트 뷰(길이 = 원소×4).
        let eb = unsafe { std::slice::from_raw_parts(emb.as_ptr() as *const u8, t * h * 4) };
        self.cc.h2d(self.fn_.x.ptr, eb)?;
        let p = match ple {
            Some(p) => {
                let pb = unsafe { std::slice::from_raw_parts(p.as_ptr() as *const u8, t * h * 4) };
                self.cc.h2d(self.fn_.ple_e.ptr, pb)?;
                Some(self.fn_.ple_e.ptr)
            }
            None => None,
        };
        Ok((self.fn_.x.ptr, p))
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
        // 전체창 진입(emb 엔트리·final 출구) — 단일 GPU 경로 불변.
        self.fn_chain_run_inner(slot, pos, t, d_emb, ple_e_dev, None, None)
    }

    /// [PP2-FN 2026-10-20] 창 일반 체인 — self.window 순회. 비최종 창은
    /// R[t][hc·hidden] 전체를 반환(경계 잔차 — PP 샤드B가 h2d해 이어받는다),
    /// 최종 창은 mixer+sample. r_in_dev(Some) = 이전 샤드의 R(emb 무시).
    #[allow(clippy::type_complexity)]
    pub fn fn_chain_run(
        &mut self,
        slot: usize,
        pos: u32,
        t: usize,
        d_emb: CUdeviceptr,
        ple_e_dev: Option<CUdeviceptr>,
        r_in_dev: Option<CUdeviceptr>,
    ) -> Result<Vec<f32>, String> {
        self.fn_chain_run_inner(slot, pos, t, d_emb, ple_e_dev, r_in_dev, None)
    }

    /// [QSA] 콜백판 — 풀층에서 호스트 선택(오라클 코드)을 끼워 넣는 진입.
    /// [QSA] 콜밭판 — 체인이 idx_qk GEMM·d2h·h2d를 수행하고 호스트 콜백은
    /// 순수 선택(오라클 코드)만: (il, t, idx_rows[t][640]) → (sel, nsel).
    #[allow(clippy::type_complexity)]
    pub fn fn_chain_run_qsa(
        &mut self,
        slot: usize,
        pos: u32,
        t: usize,
        d_emb: CUdeviceptr,
        ple_e_dev: Option<CUdeviceptr>,
        r_in_dev: Option<CUdeviceptr>,
        qsa_sel: &mut dyn FnMut(usize, usize, &[f32]) -> (Vec<i32>, Vec<i32>),
    ) -> Result<Vec<f32>, String> {
        self.fn_chain_run_inner(slot, pos, t, d_emb, ple_e_dev, r_in_dev, Some(qsa_sel))
    }

    fn fn_chain_run_inner(
        &mut self,
        slot: usize,
        pos: u32,
        t: usize,
        d_emb: CUdeviceptr,
        ple_e_dev: Option<CUdeviceptr>,
        r_in_dev: Option<CUdeviceptr>,
        mut qsa_sel: Option<&mut dyn FnMut(usize, usize, &[f32]) -> (Vec<i32>, Vec<i32>)>,
    ) -> Result<Vec<f32>, String> {
        let _g = self.cc.guard()?;
        if !self.fn_.on {
            return Err("fn_chain: set_fn 미등록".into());
        }
        // [fix 2026-10-20] KV 온디맨드 확보 — chain_t와 동일(누락 시 K6 블록
        // 테이블 기저 0 → attn_prep_q 불법 기록).
        self.kv_ensure(slot, pos as usize + t + 2)?;
        self.fn_ensure_bufs(t)?;
        // [fix 2026-10-20] classic 체인 스크래치(dchain·GDN·attn·MoE) 보장 —
        // chain_t와 동일 전제(커널 내부 ensure가 없는 버퍼들).
        self.ensure_chain_bufs()?;
        self.ensure_gdn_bufs(t)?;
        self.ensure_attn_bufs(t)?;
        self.ensure_dyt(t)?;
        let (h, hh, hc) = (self.hidden, self.fn_.hc * self.hidden, self.fn_.hc);
        let win = self.window;
        if let Some(ri) = r_in_dev {
            // [PP2] 샤드B 진입 — 이전 샤드 R을 적재(d2d — 동일 버퍼면 무해).
            self.cc.d2d(self.fn_.r.ptr, ri, t * hh * 4)?;
        } else {
            // R 초기화 — **매 청크**(멀티스트림 잔차는 토큰별 — 청크가 토큰을
            // 완결 처리. oracle forward 계약. PP2 비최종 샤드A도 emb 진입).
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
        if self.debug_layers {
            let mut rb = vec![0u8; hh * 4];
            self.cc.d2h(&mut rb, self.fn_.r.ptr)?;
            let row: Vec<f32> = rb
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            let mx = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            eprintln!("fn[gpu] init R0max={mx:.4}");
        }
        let interval = self.attn.attn.ok_or("fn_chain: attn 미등록")?.interval;
        let mut gi = win.gdn_start(interval);
        for off in 0..win.n() {
            let il = win.abs(off);
            // PLE — HC mix 앞에서 멀티스트림 상태에 직접 가산.
            if self.fn_.ple_il == Some(il) {
                let pe = ple_e_dev.ok_or("fn_chain: PLE 스테이징 부재")?;
                // key/value = kv_proj(ple_e) — [hh + h] 팩 GEMM(ple_k 단일 버퍼).
                self.lin_forward(
                    true,
                    LinPath::Gemm,
                    &format!("blk.{il}.ple_kv.weight"),
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
                &format!("blk.{il}.hc_dn_a.weight"),
                &format!("blk.{il}.hc_up_a.weight"),
                &format!("blk.{il}.hc_inj_a.weight"),
            )?;
            let is_full = (il + 1).is_multiple_of(interval);
            #[allow(clippy::too_many_arguments)]
            let branch = if let Some(cb) = qsa_sel.as_mut().filter(|_| is_full) {
                // [QSA 2026-10-20] 풀층 선택 집합 경로 — idx_qk GEMM(체인) →
                // d2h → 호스트 순수 선택(오라클 코드·집합 정합) → h2d →
                // q/k/v GEMM + sel 어텐션(attn_qsa_part).
                self.lin_forward(
                    true,
                    LinPath::Gemm,
                    &format!("blk.{il}.idx_qk.weight"),
                    self.fn_.x.ptr,
                    self.fn_.g.ptr,
                    t,
                )?;
                let ihd640 = (self.fn_.idx_heads + 1) * 128;
                let mut ib = vec![0u8; t * ihd640 * 4];
                self.cc.d2h(&mut ib, self.fn_.g.ptr)?;
                // SAFETY: ib는 t*640*4 바이트(정렬 4) — f32 뷰 재해석.
                let rows: Vec<f32> = unsafe {
                    std::slice::from_raw_parts(ib.as_ptr() as *const f32, t * ihd640).to_vec()
                };
                let (sel, nsel) = cb(il, t, &rows);
                let stride = sel.len() / t.max(1);
                let selb: Vec<u8> = sel.iter().flat_map(|v| v.to_le_bytes()).collect();
                let nselb: Vec<u8> = nsel.iter().flat_map(|v| v.to_le_bytes()).collect();
                self.cc.h2d(self.fn_.sel.ptr, &selb)?;
                self.cc.h2d(self.fn_.nsel.ptr, &nselb)?;
                self.lin_forward(
                    true,
                    LinPath::Gemm,
                    &format!("blk.{il}.attn_q.weight"),
                    self.fn_.x.ptr,
                    self.dchain[0],
                    t,
                )?;
                self.lin_forward(
                    true,
                    LinPath::Gemm,
                    &format!("blk.{il}.attn_k.weight"),
                    self.fn_.x.ptr,
                    self.dchain[1],
                    t,
                )?;
                self.lin_forward(
                    true,
                    LinPath::Gemm,
                    &format!("blk.{il}.attn_v.weight"),
                    self.fn_.x.ptr,
                    self.dchain[2],
                    t,
                )?;
                self.attn_qsa_launch(
                    slot,
                    il / interval,
                    t,
                    self.dchain[0],
                    self.dchain[1],
                    self.dchain[2],
                    self.fn_.sel.ptr,
                    self.fn_.nsel.ptr,
                    stride,
                )?;
                self.attn.doutv_a
            } else if is_full {
                self.lin_forward(
                    true,
                    LinPath::Gemm,
                    &format!("blk.{il}.attn_q.weight"),
                    self.fn_.x.ptr,
                    self.dchain[0],
                    t,
                )?;
                self.lin_forward(
                    true,
                    LinPath::Gemm,
                    &format!("blk.{il}.attn_k.weight"),
                    self.fn_.x.ptr,
                    self.dchain[1],
                    t,
                )?;
                self.lin_forward(
                    true,
                    LinPath::Gemm,
                    &format!("blk.{il}.attn_v.weight"),
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
                    &format!("blk.{il}.attn_qkv.weight"),
                    self.fn_.x.ptr,
                    self.dchain[0],
                    t,
                )?;
                self.lin_forward(
                    true,
                    LinPath::Gemm,
                    &format!("blk.{il}.attn_gate.weight"),
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
                format!("blk.{il}.attn_o.weight")
            } else {
                format!("blk.{il}.ssm_out.weight")
            };
            self.lin_forward(true, LinPath::Gemm, &lo, branch, self.fn_.y.ptr, t)?;
            if self.debug_layers {
                let mut yb = vec![0u8; h * 4];
                self.cc.d2h(&mut yb, self.fn_.y.ptr)?;
                let yr: Vec<f32> = yb
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect();
                let ym = yr.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                eprintln!("fn[gpu] layer {il:>2} y0max={ym:.4}");
            }
            self.fn_hc_write(t)?;
            if self.debug_layers {
                let mut rb = vec![0u8; hh * 4];
                self.cc.d2h(&mut rb, self.fn_.r.ptr)?;
                let row: Vec<f32> = rb
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect();
                let mx = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                eprintln!("fn[gpu] layer {il:>2} attn R0max={mx:.4}");
            }
            // ── MoE 블록: HC mix → MoE → HC write.
            self.fn_hc_cycle(
                2 * il + 1,
                t,
                &format!("blk.{il}.hc_dn_m.weight"),
                &format!("blk.{il}.hc_up_m.weight"),
                &format!("blk.{il}.hc_inj_m.weight"),
            )?;
            let y2 = self.moe_ffn_dev_t(il, self.fn_.x.ptr, t)?;
            // moe 출력 → y 버퍼 d2d(프로브급 — 이후 커널 직접 소비로 최적화).
            self.cc.d2d(self.fn_.y.ptr, y2, t * h * 4)?;
            self.fn_hc_write(t)?;
            // [디버그 배선 2026-10-20] 층별 지문 — R 행 0(오라클 r[0] 대응).
            // debug_layers가 켜지면 그래프 무효화(기존 g4_graph_ok 계약).
            if self.debug_layers {
                let mut rb = vec![0u8; hh * 4];
                self.cc.d2h(&mut rb, self.fn_.r.ptr)?;
                let row: Vec<f32> = rb
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect();
                let mx = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let nan = row.iter().any(|v| v.is_nan());
                eprintln!("fn[gpu] layer {il:>2} R0max={mx:.4} nan={nan}");
            }
        }
        // [PP2] 비최종 창 — R 전체를 경계 잔차로 반환.
        let (win, n_layers) = (self.window, self.n_layers);
        if !win.owns_final(n_layers) {
            let mut rb = vec![0u8; t * hh * 4];
            self.cc.d2h(&mut rb, self.fn_.r.ptr)?;
            // SAFETY: rb는 t*hh*4 바이트(정렬 4) — f32 뷰 재해석.
            return Ok(unsafe {
                std::slice::from_raw_parts(rb.as_ptr() as *const f32, t * hh).to_vec()
            });
        }
        // ── 최종 mixer — norm(행 2L) + mix(head 입력 sample).
        self.fn_hc_cycle(
            2 * n_layers,
            t,
            "blk.final_hc_dn.weight",
            "blk.final_hc_up.weight",
            "blk.final_hc_inj.weight",
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

    /// [gdn-check] GDN 0층 고립 실행 — x[t][hidden] 스테이징 → qkv/z GEMM →
    /// gdn 체인 → gated 출력 d2h 반환(헤드 재배치·o_proj 전).
    pub fn fn_gdn_probe(&mut self, x: &[f32], t: usize) -> Result<Vec<f32>, String> {
        let _g = self.cc.guard()?;
        let h = self.hidden;
        if x.len() != t * h {
            return Err(format!("fn_gdn_probe: x={} != t{t}×h{h}", x.len()));
        }
        self.fn_ensure_bufs(t)?;
        self.ensure_chain_bufs()?;
        self.ensure_gdn_bufs(t)?;
        // SAFETY: x 로컬 슬라이스 유효 수명 내 바이트 뷰.
        let xb = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, t * h * 4) };
        self.cc.h2d(self.fn_.x.ptr, xb)?;
        self.lin_forward(
            true,
            LinPath::Gemm,
            "blk.0.attn_qkv.weight",
            self.fn_.x.ptr,
            self.dchain[0],
            t,
        )?;
        self.lin_forward(
            true,
            LinPath::Gemm,
            "blk.0.attn_gate.weight",
            self.fn_.x.ptr,
            self.dchain[1],
            t,
        )?;
        let g = self.gdn_chain_dev_run(0, 0, t, self.fn_.x.ptr, self.dchain[0], self.dchain[1])?;
        // [gated] 코어 출력(o_proj 전) — dgate = gdn 체인 gated 버퍼.
        let mut ob = vec![0u8; t * h * 4];
        // SAFETY: ob 로컬 버퍼 — d2h 대상 길이 계약.
        self.cc.d2h(&mut ob, g)?;
        // [gdn-check 미러] 코어 입력 버퍼 행0 지문 — 오라클 core-in 대조.
        if llm170_diag::dump::opts().key("debug_layers") {
            let dm = self.gdn.gdn.ok_or("GDN: 형상 미등록")?;
            let nb = |n: usize| vec![0u8; n * 4];
            let rd = |buf: CUdeviceptr, n: usize| -> Result<Vec<f32>, String> {
                let mut b = nb(n);
                self.cc.d2h(&mut b, buf)?;
                Ok(b.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect())
            };
            let mx = |v: &[f32]| v.iter().fold(0f32, |a, x| a.max(x.abs()));
            // bg = [beta 48][g 48] 엔진 순서(커널 내부 vperm)·o = 스캔 출력.
            // 실측 2026-10-21: bg·conv v는 오라클과 원소 정합 — 발산은
            // 스캔(oN CPU 0.0966 vs GPU 0.0592) 내부에 국소화.
            let bgr = rd(self.gdn.dbg, 2 * dm.h_v)?;
            let f2 = |v: &[f32]| -> String {
                v.iter()
                    .map(|x| format!("{x:.3}"))
                    .collect::<Vec<_>>()
                    .join(",")
            };
            let o0 = rd(self.gdn.dgo, dm.v_len())?;
            eprintln!("fn[gpu] core-in L0 bgN={} oN={:.4}", f2(&bgr), mx(&o0),);
        }
        Ok(ob
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    /// [PP2] 샤드B 진입 R 스테이징 — r[t][hc·hidden] h2d → fn_.r.
    pub fn fn_stage_r(&mut self, r: &[f32], t: usize) -> Result<CUdeviceptr, String> {
        let _g = self.cc.guard()?;
        self.fn_ensure_bufs(t)?;
        let hh = self.fn_.hc * self.hidden;
        // SAFETY: r 로컬 슬라이스 유효 수명 내 바이트 뷰(길이 = 원소×4).
        let rb = unsafe { std::slice::from_raw_parts(r.as_ptr() as *const u8, t * hh * 4) };
        self.cc.h2d(self.fn_.r.ptr, rb)?;
        Ok(self.fn_.r.ptr)
    }

    /// head(plain BF16 "blk.head.weight") — sample[hidden] 1행 → 로짓 [vocab].
    /// M1 프로브용(호출당 할당 — 디코드 상수화는 최적화 단계).
    pub fn fn_head(&mut self, sample: &[f32]) -> Result<Vec<f32>, String> {
        let _g = self.cc.guard()?;
        let h = self.hidden;
        let (_, vocab, _) = self.plain_spec("blk.head.weight")?;
        let dx = self.cc.alloc(h * 4)?;
        let dy = self.cc.alloc(vocab * 4)?;
        // SAFETY: sample 로컬 슬라이스 유효 수명 내 바이트 뷰.
        let sb = unsafe { std::slice::from_raw_parts(sample.as_ptr() as *const u8, h * 4) };
        self.cc.h2d(dx, sb)?;
        self.lin_forward(true, LinPath::GemvT, "blk.head.weight", dx, dy, 1)?;
        let mut ob = vec![0u8; vocab * 4];
        self.cc.d2h(&mut ob, dy)?;
        self.cc.free(dx)?;
        self.cc.free(dy)?;
        // SAFETY: ob는 vocab*4 바이트(정렬 4) — f32 뷰 재해석.
        Ok(unsafe { std::slice::from_raw_parts(ob.as_ptr() as *const f32, vocab).to_vec() })
    }
}
