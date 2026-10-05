//! Q4Acc FrameHost 구현 — np 프리미티브·버퍼 관리·mm/op 디스패치 (R7 이동).
use super::*;

impl llm170_core::matmul::FrameHost for Q4Acc {
    /// np 행별 conv 1런치 (plans/74 N2) — gdn_conv(t=1) 산술, 상태는 행
    /// 포인터 테이블. qkv/out은 [t][ch] 연속 프레임 버퍼.
    fn frame_gdn_conv_np(
        &self,
        qkv: u64,
        out: u64,
        states: &[u64],
        cw: u64,
        ch: usize,
        k: usize,
    ) -> Result<(), String> {
        let t = states.len();
        if t == 0 {
            return Ok(());
        }
        // plans/115 U0: 캐시된 테이블 — 매 층 동기 h2d(풀 드레인) 제거.
        let tbl = self.np_state_tbl_cached(states)?;
        let (mut q, mut c, mut s_, mut o_) = (
            self.fptr(qkv)?,
            self.fptr(cw)?,
            tbl as *mut std::ffi::c_void,
            self.fptr(out)?,
        );
        let (mut chh, mut kk, mut tt) = (ch as i32, k as i32, t as i32);
        self.kop(
            "gdn_conv_np",
            (ch as u32).div_ceil(64),
            t as u32,
            1,
            64,
            &mut cargs!(&mut q, &mut c, &mut s_, &mut o_, &mut chh, &mut kk, &mut tt),
        )
    }
    /// np 행별 AR 1런치 (plans/74 N2) — gdn_ar_w_swap(t=1) 산술(scale=1,
    /// q는 L2Rows+Scale 로 선스케일), 상태는 행 포인터 테이블.
    #[allow(clippy::too_many_arguments)]
    fn frame_gdn_ar_np(
        &self,
        q: u64,
        k: u64,
        v: u64,
        beta_ge: u64,
        out: u64,
        states: &[u64],
        h_k: usize,
        h_v: usize,
        d: usize,
    ) -> Result<(), String> {
        let t = states.len();
        if t == 0 {
            return Ok(());
        }
        // plans/115 U0: 캐시된 테이블 — 매 층 동기 h2d(풀 드레인) 제거.
        let tbl = self.np_state_tbl_cached(states)?;
        let (mut sp, mut qp, mut kp, mut vp, mut bp, mut op_) = (
            tbl as *mut std::ffi::c_void,
            self.fptr(q)?,
            self.fptr(k)?,
            self.fptr(v)?,
            self.fptr(beta_ge)?,
            self.fptr(out)?,
        );
        let (mut dd, mut ks, mut vs, mut hv, mut hk, mut sc, mut tt) = (
            d as i32,
            (h_k * d) as i32,
            (h_v * d) as i32,
            h_v as i32,
            h_k as i32,
            1.0f32,
            t as i32,
        );
        // gx=h_v(페어 축 — 커널의 blockIdx.x), gy=d(u 축). 27B rawhip 판과
        // 동일 순서(2026-09-16 실수로 (d,h_v)로 바꿔써 GPU 메모리 폴트).
        self.ctx.launch3(
            "gdn_ar_w_np",
            h_v as u32,
            d as u32,
            1,
            32,
            &mut cargs!(
                &mut sp, &mut qp, &mut kp, &mut vp, &mut bp, &mut op_, &mut dd, &mut ks, &mut vs,
                &mut hv, &mut hk, &mut sc, &mut tt
            ),
        )
    }
    /// plans/73(np): 프레임 버퍼 행 뷰 — 배치 디코드의 per-seq 상태 op용.
    fn shexp_gu_t(
        &self,
        x: u64,
        wg: &llm170_core::matmul::Weight,
        wu: &llm170_core::matmul::Weight,
        h: u64,
        n_in: usize,
        n_hidden: usize,
        t: usize,
    ) -> Result<(), String> {
        let mut xp = self.fptr(x)? as *mut std::ffi::c_void;
        let (wgd, _) = self.dev_weight(wg)?;
        let (wud, _) = self.dev_weight(wu)?;
        let mut wgp = wgd as *mut std::ffi::c_void;
        let mut wup = wud as *mut std::ffi::c_void;
        let mut hp = self.fptr(h)? as *mut std::ffi::c_void;
        let (mut ni, mut nh) = (n_in as i32, n_hidden as i32);
        let mut args = vec![
            (&mut xp) as *mut _ as *mut std::ffi::c_void,
            (&mut wgp) as *mut _ as *mut std::ffi::c_void,
            (&mut wup) as *mut _ as *mut std::ffi::c_void,
            (&mut hp) as *mut _ as *mut std::ffi::c_void,
            (&mut ni) as *mut _ as *mut std::ffi::c_void,
            (&mut nh) as *mut _ as *mut std::ffi::c_void,
        ];
        self.ctx.launch3(
            "q4_shexp_gu_t",
            n_hidden.div_ceil(8) as u32,
            t as u32,
            1,
            256,
            &mut args,
        )
    }

    fn shexp_da_t(
        &self,
        h: u64,
        wd: &llm170_core::matmul::Weight,
        s: u64,
        mout: u64,
        n_in: usize,
        n_hidden: usize,
        t: usize,
    ) -> Result<(), String> {
        let mut hp = self.fptr(h)? as *mut std::ffi::c_void;
        let (wdd, _) = self.dev_weight(wd)?;
        let mut wdp = wdd as *mut std::ffi::c_void;
        let mut sp = self.fptr(s)? as *mut std::ffi::c_void;
        let mut mp = self.fptr(mout)? as *mut std::ffi::c_void;
        let (mut ni, mut nh) = (n_in as i32, n_hidden as i32);
        let mut args = vec![
            (&mut hp) as *mut _ as *mut std::ffi::c_void,
            (&mut wdp) as *mut _ as *mut std::ffi::c_void,
            (&mut sp) as *mut _ as *mut std::ffi::c_void,
            (&mut mp) as *mut _ as *mut std::ffi::c_void,
            (&mut ni) as *mut _ as *mut std::ffi::c_void,
            (&mut nh) as *mut _ as *mut std::ffi::c_void,
        ];
        self.ctx.launch3(
            "q4_shexp_da_t",
            n_in.div_ceil(8) as u32,
            t as u32,
            1,
            256,
            &mut args,
        )
    }

    fn frame_slice(&self, h: u64, off_elems: usize, len: usize) -> Result<u64, String> {
        let mut v = self.frames.lock().map_err(|e| e.to_string())?;
        let idx = (h.checked_sub(1).ok_or("frame 핸들 0")?) as usize;
        let (base, cap) = *v.get(idx).ok_or_else(|| format!("frame 핸들 없음: {h}"))?;
        let need = (off_elems + len) * 4;
        if need > cap {
            return Err(format!("frame_slice 범위 초과: need {need} > cap {cap}"));
        }
        let ptr = unsafe { base.add(off_elems * 4) };
        v.push((ptr, len * 4));
        Ok(v.len() as u64)
    }

    fn frame_qk_norm_rope(
        &self,
        q: u64,
        k: u64,
        q_norm: &[f32],
        k_norm: &[f32],
        cs: &[f32],
        eps: f32,
        pos0: usize,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        n_rot: usize,
        t: usize,
    ) -> Result<(), String> {
        // 상수 3개(qn/kn/cs) — plans/73: 매 호출 h2d(+sync)가 스텝당 36회의
        // 동기를 만들었다. **키는 (ptr,len)** — 내용 해시는 층마다 값이 달라
        // 단일 슬롯 캐시가 매 층 미스했고(24KB+2KB 동기 복사 ×12층 = 40ms/스텝),
        // 프레임이 헤드 타일을 1회 만들어 상주시키므로 포인터가 곧 신원이다.
        // (2026-09-16 실측 — qsa.mm+rope 3.4ms/층의 전부가 이 복사였다)
        let (qnd, knd, csd) = {
            let qnd = self.upload_map(&self.qn_map, "qn_t", q_norm)?;
            let knd = self.upload_map(&self.kn_map, "kn_t", k_norm)?;
            let qh = q_norm.as_ptr() as u64;
            let kh = k_norm.as_ptr() as u64;
            let _ = (qh, kh);
            let cskey = (cs.as_ptr() as usize, cs.len());
            let mut c = (
                self.cst.lock().map_err(|e| e.to_string())?,
                self.cst_cache.lock().map_err(|e| e.to_string())?,
            );
            if *c.1 != cskey || c.0.ptr.is_null() {
                c.0.ensure(&self.ctx, cs.len().max(1) * 4)?;
                self.ctx.h2d(c.0.ptr, bytemuck::cast_slice(cs))?;
                *c.1 = cskey;
            }
            (qnd, knd, c.0.ptr)
        };
        let mut qp = self.fptr(q)? as *mut std::ffi::c_void;
        let mut kp = self.fptr(k)? as *mut std::ffi::c_void;
        let mut qwp = qnd as *mut std::ffi::c_void;
        let mut kwp = knd as *mut std::ffi::c_void;
        let mut csp = csd as *mut std::ffi::c_void;
        // kq_scale은 이 커널의 decode 판은 k에 구워 넣지만(kqs=self.kq_scale),
        // QSA 프레임 경로는 **k를 무척도(1.0)로 둔다** — QSA KV 캐시 규약이
        // 무척도 k이고 qsa_attn_sel6가 q·k에 kq_scale을 곱하기 때문. 초기 구현은
        // 0.0을 넘겨 k를 전부 0으로 만드는 잠복 결함이었음(미호출 경로라 미발견,
        // 2026-09-14 plans/67 2c 연결 시 발견·수정).
        let mut kq = 1.0f32;
        let mut ep = eps;
        let mut pp = pos0 as i32;
        let mut nh = n_head as i32;
        let mut nk = n_kv as i32;
        let mut h = hd as i32;
        let mut nr = n_rot as i32;
        let rows = n_head + n_kv;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&mut qp) as *mut _ as *mut std::ffi::c_void,
            (&mut kp) as *mut _ as *mut std::ffi::c_void,
            (&mut qwp) as *mut _ as *mut std::ffi::c_void,
            (&mut kwp) as *mut _ as *mut std::ffi::c_void,
            (&mut csp) as *mut _ as *mut std::ffi::c_void,
            (&mut ep) as *mut _ as *mut std::ffi::c_void,
            (&mut kq) as *mut _ as *mut std::ffi::c_void,
            (&mut pp) as *mut _ as *mut std::ffi::c_void,
            (&mut nh) as *mut _ as *mut std::ffi::c_void,
            (&mut nk) as *mut _ as *mut std::ffi::c_void,
            (&mut h) as *mut _ as *mut std::ffi::c_void,
            (&mut nr) as *mut _ as *mut std::ffi::c_void,
        ];
        self.ctx
            .launch3("qk_norm_rope", rows as u32, t as u32, 1, 32, &mut args)
    }

    // ─── 프레임(활성화 GPU 상주) — plans/64 P1 ───
    // 계약: core `qwen4exp/frame.rs`의 op 순서·산술 그대로. 프레임 경로는
    // 스텝당 동기를 ~14회로 줄인다(값 경로 ~1300회).

    /// 버퍼 할당 — `len`은 **원소 수**(f32 4바이트/u32 1워드). core frame.rs
    /// 규약(`a(k_len)`, `a(v.len())`)을 따른다.
    fn frame_alloc(&self, len: usize) -> Result<u64, String> {
        let p = self.ctx.alloc((len.max(4)) * 4)?;
        let mut v = self.frames.lock().map_err(|e| e.to_string())?;
        // cap은 **바이트**다 — frame_slice가 need=(off+len)*4와 비교하고 슬라이스
        // 핸들도 len*4로 적는다. 종전엔 원소 수를 적어 슬라이스 상한이 실제
        // 버퍼의 1/4행이었다(np 1행 뷰는 안 걸렸고 다중 프리필 행 대역이 걸림,
        // 2026-09-17). 검사만 완화되므로 기존 경로는 불변.
        v.push((p, len * 4));
        Ok(v.len() as u64)
    }

    fn frame_free(&self, _h: u64) -> Result<(), String> {
        // 해제 없음 (ADR-0014) — 풀은 영구.
        Ok(())
    }

    fn frame_write(&self, h: u64, data: &[f32]) -> Result<(), String> {
        self.fchk(h, data.len() * 4, "frame_write")?;
        let p = self.fptr(h)?;
        self.ctx.h2d(p, bytemuck::cast_slice(data))
    }

    fn frame_write_u32(&self, h: u64, data: &[u32]) -> Result<(), String> {
        self.fchk(h, data.len() * 4, "frame_write_u32")?;
        let p = self.fptr(h)?;
        self.ctx.h2d(p, bytemuck::cast_slice(data))
    }

    fn frame_read(&self, h: u64, out: &mut [f32]) -> Result<(), String> {
        self.fchk(h, out.len() * 4, "frame_read")?;
        let p = self.fptr(h)?;
        // 동기 hipMemcpy — 공유 핀 스테이징(d2h 헬퍼)의 재사용 상태에 의존하지
        // 않는다. 프레임 판독은 스텝당 몇 회뿐이라 동기 경로 비용이 무의미하다.
        unsafe {
            if llm170_diag::dump::opts().key("io_time") {
                let t0 = std::time::Instant::now();
                let r = ck(
                    hip::hipMemcpy(
                        out.as_mut_ptr() as *mut std::ffi::c_void,
                        p as *const std::ffi::c_void,
                        out.len() * 4,
                        hip::hipMemcpyKind_hipMemcpyDeviceToHost,
                    ),
                    "frame_read",
                );
                crate::rawhip::ctx::launch::IO_US.fetch_add(
                    t0.elapsed().as_micros() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
                crate::rawhip::ctx::launch::IO_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                crate::rawhip::ctx::launch::IO_LAST.store(3, std::sync::atomic::Ordering::Relaxed);
                r
            } else {
                ck(
                    hip::hipMemcpy(
                        out.as_mut_ptr() as *mut std::ffi::c_void,
                        p as *const std::ffi::c_void,
                        out.len() * 4,
                        hip::hipMemcpyKind_hipMemcpyDeviceToHost,
                    ),
                    "frame_read",
                )
            }
        }
    }

    /// KTRACE 진단 훅 — 스텝 단위 덤프+재시작(core→백엔드 의존 방향 존중).
    /// plans/111: LLM170_KTRACE 플래그로 게이트 — 종전 무조건 ktrace_on()이라
    /// plans/88 이후 env와 무관하게 상시 녹화됐다(런치당 hipEvent 2개 기록 +
    /// 스텝마다 덤프가 벤치·서빙 전 경로에 부과). 계약(AGENTS.md)대로 옵트인.
    fn ktrace_tick(&self) {
        crate::rawhip::ctx::launch::launch_time_report();
        crate::rawhip::ctx::launch::io_time_report();
        if !llm170_diag::flag::on("LLM170_KTRACE") {
            return;
        }
        eprintln!("{}", crate::rawhip::ktrace_dump());
        crate::rawhip::ktrace_on();
    }

    /// 슬롯 반납 — PLE 링/워터마크 제거(2026-09-16 RCA: reset_seq 이 링을
    /// 못 지워 새 대화가 이전 대화의 n-gram 링을 읽었다 — np4 잔여 비결정성
    /// 및 슬롯 재사용 오염의 원인).
    fn acc_reset_seq(&self, seq: usize) {
        if let Ok(mut m) = self.ple_ring.lock() {
            m.remove(&seq);
        }
        if let Ok(mut wm) = self.ple_ring_pos.lock() {
            wm.remove(&seq);
        }
    }

    /// plans/110 W2 — 검증 배치 핀: t=2..8 Q8_0/f32 GEMV의 행별 t=1
    /// 디스패치 강제(VERIFY_ROW_PIN).
    fn frame_verify_rows(&self, on: bool) {
        crate::rawhip::ctx::VERIFY_ROW_PIN.store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// [t][vocab] logits 행별 GPU argmax — np greedy 판정 (plans/74 N1).
    /// argmax64 = CPU greedy와 동일 의미(동률 최저 인덱스).
    fn frame_argmax_rows(&self, logits: u64, t: usize, vocab: usize) -> Result<Vec<u32>, String> {
        let base = self.fptr(logits)?;
        // 병렬 2단계(plans/74 N3) — argmax64 1블록 판은 vocab 248k 에서
        // ~1.2ms/행 직렬 꼬리(FN np4 KTRACE 4.6ms/step).
        let nblk = (vocab / 4096).clamp(1, 64) as u32;
        let part = self.ctx.scratch(t.max(1) * nblk as usize * 8)?;
        let outb = self
            .ctx
            .scratch(t.max(1) * 8 + 8 * nblk as usize * t.max(1))?;
        let out = unsafe { outb.add(t.max(1) * nblk as usize * 8) };
        {
            let mut xp = base as *mut std::ffi::c_void;
            let mut vb = vocab as i32;
            let mut pp = part as *mut std::ffi::c_void;
            let mut nb = nblk as i32;
            let mut args = vec![
                (&mut xp) as *mut _ as *mut std::ffi::c_void,
                (&mut vb) as *mut _ as *mut std::ffi::c_void,
                (&mut pp) as *mut _ as *mut std::ffi::c_void,
                (&mut nb) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx
                .launch3("argmax_rows_s1", nblk, t.max(1) as u32, 1, 256, &mut args)?;
        }
        {
            let mut pp = part as *mut std::ffi::c_void;
            let mut op = out as *mut std::ffi::c_void;
            let mut nb = nblk as i32;
            let mut args = vec![
                (&mut pp) as *mut _ as *mut std::ffi::c_void,
                (&mut op) as *mut _ as *mut std::ffi::c_void,
                (&mut nb) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx
                .launch3("argmax_rows_s2", t.max(1) as u32, 1, 1, 64, &mut args)?;
        }
        let mut r8 = vec![0u8; t * 8];
        self.ctx.d2h(&mut r8, out)?;
        Ok((0..t)
            .map(|s| {
                let b = &r8[s * 8..s * 8 + 8];
                u32::from_le_bytes([b[4], b[5], b[6], b[7]])
            })
            .collect())
    }

    fn frame_mm(
        &self,
        x: u64,
        w: &llm170_core::matmul::Weight<'_>,
        out: u64,
        t: usize,
    ) -> Result<(), String> {
        let (xp, op) = (self.fptr(x)?, self.fptr(out)?);
        self.frame_gemm(xp, w, op, t)
    }

    fn frame_mm_group(
        &self,
        x: u64,
        ws: &[llm170_core::matmul::Weight<'_>],
        outs: &[u64],
        t: usize,
    ) -> Result<(), String> {
        if ws.len() != outs.len() {
            return Err(format!(
                "frame_mm_group: ws({}) != outs({})",
                ws.len(),
                outs.len()
            ));
        }
        // plans/110 W2: 검증 배치 핀 — 행별 t=1 재귀. t=2..8의 배치 디스패치
        // (mt 커널·듀얼 없음)는 t=1 디코드(듀얼/dmmv 포함)와 산술이 갈라진다.
        // 행 뷰(메모)로 재귀하면 decode1이 쓰는 t=1 코드 경로를 그대로 탄다.
        if (2..=8).contains(&t)
            && crate::rawhip::ctx::VERIFY_ROW_PIN.load(std::sync::atomic::Ordering::Relaxed)
        {
            let n_in = ws[0].n_in as usize;
            for r in 0..t {
                let x_row = self.vview(x, r * n_in, n_in)?;
                let mut out_rows = Vec::with_capacity(outs.len());
                for (i, w) in ws.iter().enumerate() {
                    let no = w.n_out as usize;
                    out_rows.push(self.vview(outs[i], r * no, no)?);
                }
                self.frame_mm_group(x_row, ws, &out_rows, 1)?;
            }
            return Ok(());
        }
        let xp = self.fptr(x)?;
        // 동일 입력 — 양자화 1회 공유 (f32 계열이 섞이면 개별).
        let f32_family =
            |ty: GgmlType| matches!(ty, GgmlType::F32 | GgmlType::Bf16 | GgmlType::F16);
        let f32w = f32_family(ws[0].ty);
        // 108 P7: t=1 q8_0 전용 그룹은 dmmv — 프레임 활성 양자화를 건너뛰고
        // f32 활성 × 커널 내 디양자화 가중 직접 dot(승인된 산술 클래스
        // 변경, 원장 118). 킬스위치 LLM170_HIP_DMMV_OFF=1.
        let dmmv_on = t == 1 && !f32w && ws.iter().all(|w| ggml_id(w.ty) == 8);
        if ws
            .iter()
            .all(|w| w.n_in == ws[0].n_in && f32_family(w.ty) == f32w)
            && !f32w
            && !dmmv_on
        {
            let (xq, xq_w) = self.frame_quant(xp, ws[0].n_in as usize, t)?;
            // plans/83 D2: t=1 디코드에서 그룹 내 q8_0 인접쌍을 듀얼 커널로
            // 융합 — 런치 수 절반, 블록 수 합산(점유 개선). dual 커널의 행
            // 산술은 원판 gemm_q8_0과 동일 트리 → 비트 불변.
            let dual_ok = t == 1;
            let mut idx = 0usize;
            while idx < ws.len() {
                let w = &ws[idx];
                let ty = ggml_id(w.ty);
                if dual_ok && ty == 8 && idx + 1 < ws.len() && ggml_id(ws[idx + 1].ty) == 8 {
                    let (wd1, _) = self.dev_weight(w)?;
                    let (wd2, _) = self.dev_weight(&ws[idx + 1])?;
                    let o1 = self.fptr(outs[idx])?;
                    let o2 = self.fptr(outs[idx + 1])?;
                    let n_in = w.n_in as usize;
                    let no1 = w.n_out as usize;
                    let no2 = ws[idx + 1].n_out as usize;
                    self.gemm_q8_dual(xq, wd1, no1, o1, wd2, no2, o2, n_in)?;
                    idx += 2;
                    continue;
                }
                let (wd, _) = self.dev_weight(w)?;
                let op = self.fptr(outs[idx])?;
                let n_in = w.n_in as usize;
                let n_out = w.n_out as usize;
                self.launch_gemm(ty, xq, wd, n_in, n_out, xq_w, t, op)?;
                idx += 1;
            }
            return Ok(());
        }
        if dmmv_on {
            for (w, &o) in ws.iter().zip(outs.iter()) {
                let (wd, _) = self.dev_weight(w)?;
                let op = self.fptr(o)?;
                self.ctx
                    .gemv_q8_dmmv_out(xp, wd, w.n_in as usize, w.n_out as usize, op)?;
            }
            return Ok(());
        }
        // plans/71: q8_0 가중치 + t>=32는 MMQ(int8 dp4a) — f32 활성을 직접 받아
        // 자체 양자화. 종전 j128 타일 대비 측정 이득은 벤치로 검증.
        // 혼합 패밀리 그룹(GDN [q8,q8,f32,f32] 등) — t=1에서 인접 q8_0 쌍을
        // 듀얼로 융합(plans/83 D2). 첫 분기의 동일-패밀리 조건에 걸리지 않는
        // 그룹의 q8_0 쌍도 같은 이득을 받는다. 비트 불변(행 산술 동일).
        // plans/83 D2(계속): f32 인접쌍(β/α)은 f32 듀얼로, (q8,f32) 인접쌍
        // (hc down+inject)은 혼합 듀얼로 — 각 1런치. 행 산술은 소스 커널과
        // 동일 → 비트 불변.
        let f32fam = |ty: GgmlType| matches!(ty, GgmlType::F32 | GgmlType::Bf16 | GgmlType::F16);
        let dual_any = t == 1;
        let mut idx = 0usize;
        while idx < ws.len() {
            let w = &ws[idx];
            if dual_any && idx + 1 < ws.len() && ws[idx + 1].n_in == w.n_in {
                let a8 = w.ty == GgmlType::Q8_0;
                let b8 = ws[idx + 1].ty == GgmlType::Q8_0;
                let af = f32fam(w.ty);
                let bf = f32fam(ws[idx + 1].ty);
                if a8 && b8 {
                    let (wd1, _) = self.dev_weight(w)?;
                    let (wd2, _) = self.dev_weight(&ws[idx + 1])?;
                    let o1 = self.fptr(outs[idx])?;
                    let o2 = self.fptr(outs[idx + 1])?;
                    let (xq, _xw) = self.frame_quant(xp, w.n_in as usize, t)?;
                    self.gemm_q8_dual(
                        xq,
                        wd1,
                        w.n_out as usize,
                        o1,
                        wd2,
                        ws[idx + 1].n_out as usize,
                        o2,
                        w.n_in as usize,
                    )?;
                    idx += 2;
                    continue;
                }
                if af && bf {
                    let (wd1, _) = self.dev_weight(w)?;
                    let (wd2, _) = self.dev_weight(&ws[idx + 1])?;
                    let o1 = self.fptr(outs[idx])?;
                    let o2 = self.fptr(outs[idx + 1])?;
                    self.gemm_f32_dual(
                        xp as *const u8,
                        wd1,
                        w.n_out as usize,
                        o1,
                        wd2,
                        ws[idx + 1].n_out as usize,
                        o2,
                        w.n_in as usize,
                    )?;
                    idx += 2;
                    continue;
                }
                if a8 && bf {
                    let (wd1, _) = self.dev_weight(w)?;
                    let (wd2, _) = self.dev_weight(&ws[idx + 1])?;
                    let o1 = self.fptr(outs[idx])?;
                    let o2 = self.fptr(outs[idx + 1])?;
                    let ni = w.n_in as usize;
                    let (xq, _xw) = self.frame_quant(xp, ni, t)?;
                    self.gemm_mix_dual(
                        xq,
                        wd1,
                        w.n_out as usize,
                        o1,
                        xp as u64,
                        wd2,
                        ws[idx + 1].n_out as usize,
                        o2,
                        ni,
                    )?;
                    idx += 2;
                    continue;
                }
            }
            if t == 1
                && w.ty == GgmlType::Q8_0
                && idx + 1 < ws.len()
                && ws[idx + 1].ty == GgmlType::Q8_0
                && ws[idx + 1].n_in == w.n_in
            {
                let (wd1, _) = self.dev_weight(w)?;
                let (wd2, _) = self.dev_weight(&ws[idx + 1])?;
                let o1 = self.fptr(outs[idx])?;
                let o2 = self.fptr(outs[idx + 1])?;
                let (xq, xq_w) = self.frame_quant(xp, w.n_in as usize, t)?;
                let _ = xq_w;
                self.gemm_q8_dual(
                    xq,
                    wd1,
                    w.n_out as usize,
                    o1,
                    wd2,
                    ws[idx + 1].n_out as usize,
                    o2,
                    w.n_in as usize,
                )?;
                idx += 2;
                continue;
            }
            let op = self.fptr(outs[idx])?;
            self.frame_gemm(xp, w, op, t)?;
            idx += 1;
        }
        Ok(())
    }

    /// 상주 elementwise/RoPE/인덱서 연산 — qwen4exp 프레임이 쓰는 변형만 구현.
    fn frame_op(&self, op: &llm170_core::matmul::FrameOp) -> Result<(), String> {
        use llm170_core::matmul::FrameOp as O;
        match *op {
            O::SiluDiv { t, div, n } => {
                let mut p = self.fptr(t)?;
                let mut d = div;
                let mut nn = n as i32;
                self.kop(
                    "q4_silu_div",
                    (n as u32).div_ceil(128),
                    1,
                    1,
                    128,
                    &mut cargs!(&mut p, &mut d, &mut nn),
                )
            }
            O::SiluMul { g, u, out, n } => {
                let (mut gp, mut up, mut op) = (self.fptr(g)?, self.fptr(u)?, self.fptr(out)?);
                let mut nn = n as i32;
                self.kop(
                    "silu_mul",
                    (n as u32).div_ceil(128),
                    1,
                    1,
                    128,
                    &mut cargs!(&mut gp, &mut up, &mut op, &mut nn),
                )
            }
            O::Sigmoid { t, n } => {
                let mut p = self.fptr(t)?;
                let mut nn = n as i32;
                self.kop(
                    "q4_sigmoid",
                    (n as u32).div_ceil(128),
                    1,
                    1,
                    128,
                    &mut cargs!(&mut p, &mut nn),
                )
            }
            O::RmsRows {
                x,
                w,
                out,
                eps,
                n,
                w_reps,
            } => {
                let (xp, wp) = (self.fptr(x)?, self.fptr(w)?);
                let rows = w_reps * self.t_cur();
                let part = {
                    let mut b = self.fpart.lock().map_err(|e| e.to_string())?;
                    b.ensure(&self.ctx, rows * 32 * 8)?
                };
                {
                    let mut xa = xp;
                    let mut pa = part;
                    let mut nn = n as i32;
                    self.kop(
                        "rms_part",
                        rows as u32,
                        1,
                        1,
                        32,
                        &mut cargs!(&mut xa, &mut pa, &mut nn),
                    )?;
                }
                let mut xa = xp;
                let mut wa = wp;
                let mut pa = part;
                let mut op_ = self.fptr(out)?;
                let mut e = eps;
                let mut nn = n as i32;
                let mut rr = w_reps as i32;
                // 256스레드 = 8 그룹 × 32레인 (raw 디코더와 동일 기하 — 128로
                // 줄이면 행 절반이 미기록)
                self.kop(
                    "rms_finish",
                    rows as u32,
                    1,
                    1,
                    256,
                    &mut cargs!(
                        &mut xa, &mut wa, &mut pa, &mut op_, &mut e, &mut nn, &mut rr
                    ),
                )
            }
            O::NormGated {
                o,
                z,
                w,
                out,
                eps,
                d,
                n_h,
            } => {
                let mut op_ = self.fptr(o)?;
                let mut zp = self.fptr(z)?;
                let mut wp = self.fptr(w)?;
                let mut outp = self.fptr(out)?;
                let mut e = eps;
                let mut dd = d as i32;
                let mut nh = n_h as i32;
                let rows = n_h * self.t_cur();
                self.kop(
                    "q4_norm_gated_sig",
                    n_h as u32,
                    (rows / n_h.max(1)) as u32,
                    1,
                    32,
                    &mut cargs!(
                        &mut op_, &mut zp, &mut wp, &mut outp, &mut e, &mut dd, &mut nh
                    ),
                )
            }
            O::L2Rows { x, eps, d, n } => {
                let mut xp = self.fptr(x)?;
                let mut e = eps;
                let mut dd = d as i32;
                // 행 수는 *토큰 수*에서 온다. 버퍼 길이(t_max)를 쓰면 t=1에서도
                // t_max행을 처리해 33.7ms/스텝을 낭비한다(2026-09-14 실측).
                let rows = (n / d).max(1) as u32;
                self.kop(
                    "q4_l2_rows",
                    rows,
                    1,
                    1,
                    32,
                    &mut cargs!(&mut xp, &mut e, &mut dd),
                )
            }
            O::Scale { t, s, n } => {
                let mut p = self.fptr(t)?;
                let mut ss = s;
                let mut nn = n as i32;
                self.kop(
                    "q4_scale",
                    (n as u32).div_ceil(128),
                    1,
                    1,
                    128,
                    &mut cargs!(&mut p, &mut ss, &mut nn),
                )
            }
            O::BcastRows { src, dst, n, rows } => {
                let (mut sp, mut dp) = (self.fptr(src)?, self.fptr(dst)?);
                let (mut nn, mut rr) = (n as i32, rows as i32);
                self.kop(
                    "bcast_rows",
                    (n as u32).div_ceil(128),
                    rows as u32,
                    1,
                    128,
                    &mut cargs!(&mut sp, &mut dp, &mut nn, &mut rr),
                )
            }
            O::CopyRows {
                src,
                dst,
                src_off,
                dst_off,
                n,
            } => {
                let (mut sp, mut dp) = (self.fptr(src)?, self.fptr(dst)?);
                let (mut so, mut dfo) = (src_off as i32, dst_off as i32);
                let mut nn = n as i32;
                self.kop(
                    "copy_rows",
                    (n as u32).div_ceil(128),
                    1,
                    1,
                    128,
                    &mut cargs!(&mut sp, &mut dp, &mut so, &mut dfo, &mut nn),
                )
            }
            O::HcGateMean {
                xn,
                gate,
                out,
                hc,
                n,
            } => {
                let (mut xp, mut gp, mut op_) = (self.fptr(xn)?, self.fptr(gate)?, self.fptr(out)?);
                let total = n * self.t_cur();
                let mut h = hc as i32;
                let mut nn = n as i32;
                let mut tt = total as i32;
                self.kop(
                    "q4_hc_gate_mean",
                    (total as u32).div_ceil(128),
                    1,
                    1,
                    128,
                    &mut cargs!(&mut xp, &mut gp, &mut op_, &mut h, &mut nn, &mut tt),
                )
            }
            O::HcCombine {
                res,
                out,
                inj,
                hc,
                n,
                total: _,
            } => {
                // 커널은 (토큰,차원)당 1스레드 — op의 total(=hc·n·t)을 범위로 쓰면
                // hc배만큼 범위 밖을 쓴다(실측: hc>1에서 폴트). n·t를 쓴다.
                // plans/116-1: g(2σ(inj/hc))는 (t,hc)에만 의존 — 결합 전 t·hc
                // 스레드 사전계산 커널로 exp_cr(f64 호너, 스레드당 hc회) 중복을
                // 제거한다. 산술 동일식 → 비트 동일(charhash 무변경 기대).
                let t = self.t_cur();
                let gp = self.ctx.scratch(t * hc * 4)?;
                {
                    let (mut ip, mut g_) = (self.fptr(inj)?, gp as *mut std::ffi::c_void);
                    let mut h = hc as i32;
                    let mut tt = t as i32;
                    self.kop(
                        "q4_hc_gate",
                        ((t * hc) as u32).div_ceil(128),
                        1,
                        1,
                        128,
                        &mut cargs!(&mut ip, &mut g_, &mut h, &mut tt),
                    )?;
                }
                let (mut rp, mut op_, mut gp_) = (
                    self.fptr(res)?,
                    self.fptr(out)?,
                    gp as *mut std::ffi::c_void,
                );
                let tn = n * t;
                let mut h = hc as i32;
                let mut nn = n as i32;
                let mut tt = tn as i32;
                self.kop(
                    "q4_hc_combine",
                    (tn as u32).div_ceil(128),
                    1,
                    1,
                    128,
                    &mut cargs!(&mut rp, &mut op_, &mut gp_, &mut h, &mut nn, &mut tt),
                )
            }
            O::Split3 {
                src,
                d0,
                d1,
                d2,
                n0,
                n1,
                n2,
            } => {
                let (mut sp, mut a0, mut a1, mut a2) = (
                    self.fptr(src)?,
                    self.fptr(d0)?,
                    self.fptr(d1)?,
                    self.fptr(d2)?,
                );
                let (mut x0, mut x1, mut x2) = (n0 as i32, n1 as i32, n2 as i32);
                let total = ((n0 + n1 + n2) * self.t_cur()) as u32;
                self.kop(
                    "split3",
                    total.div_ceil(128),
                    1,
                    1,
                    128,
                    &mut cargs!(
                        &mut sp, &mut a0, &mut a1, &mut a2, &mut x0, &mut x1, &mut x2
                    ),
                )
            }
            O::GdnBetaG {
                b,
                a,
                dtb,
                sa,
                bg,
                n_h,
            } => {
                let (mut bp, mut ap, mut dp, mut sp, mut gp) = (
                    self.fptr(b)?,
                    self.fptr(a)?,
                    self.fptr(dtb)?,
                    self.fptr(sa)?,
                    self.fptr(bg)?,
                );
                let mut nh = n_h as i32;
                // dt_rank = n_h / t (n_h = dt_rank·t) — t>1에서 n_h를 dt_rank로
                // 넘기면 dtb/sa(길이 dt_rank)를 넘겨 읽어 폴트 (실측 700).
                let mut dr = (n_h / self.t_cur().max(1)) as i32;
                self.kop(
                    "gdn_beta_g",
                    (n_h as u32).div_ceil(128),
                    1,
                    1,
                    128,
                    &mut cargs!(
                        &mut bp, &mut ap, &mut dp, &mut sp, &mut gp, &mut nh, &mut dr
                    ),
                )
            }
            O::GdnConv {
                qkv,
                cw,
                state,
                out,
                ch,
                k,
                t_len,
            } => {
                let (qp, cp, stp, op_) = (
                    self.fptr(qkv)?,
                    self.fptr(cw)?,
                    self.fptr(state)?,
                    self.fptr(out)?,
                );
                if t_len == 1 {
                    let (mut q, mut c, mut s_, mut o_) = (qp, cp, stp, op_);
                    let mut chh = ch as i32;
                    let mut kk = k as i32;
                    return self.kop(
                        "gdn_conv",
                        (ch as u32).div_ceil(64),
                        1,
                        1,
                        64,
                        &mut cargs!(&mut q, &mut c, &mut s_, &mut o_, &mut chh, &mut kk),
                    );
                }
                if t_len >= k - 1 {
                    // 완전 병렬 청크판 (전제 t ≥ k-1) + 링 상태 갱신은 별도 커널
                    // (conv_t2는 상태를 갱신하지 않는다 — raw 디코더도 2런치)
                    {
                        let (mut q, mut c, mut s_, mut o_) = (qp, cp, stp, op_);
                        let mut chh = ch as i32;
                        let mut kk = k as i32;
                        let mut tt = t_len as i32;
                        self.kop(
                            "gdn_conv_t2",
                            (ch as u32).div_ceil(64),
                            t_len as u32,
                            1,
                            64,
                            &mut cargs!(
                                &mut q, &mut c, &mut s_, &mut o_, &mut chh, &mut kk, &mut tt
                            ),
                        )?;
                    }
                    let (mut q2, mut s2) = (qp, stp);
                    let mut ch2 = ch as i32;
                    let mut k2 = k as i32;
                    let mut t2 = t_len as i32;
                    return self.kop(
                        "gdn_conv_state",
                        (k - 1) as u32,
                        (ch as u32).div_ceil(64),
                        1,
                        64,
                        &mut cargs!(&mut q2, &mut s2, &mut ch2, &mut k2, &mut t2),
                    );
                }
                // 짧은 꼬리(t < k-1): 토큰별 순차 (t=1 커널 반복, 포인터 전진)
                for ti in 0..t_len {
                    let (mut q, mut c, mut s_, mut o_) =
                        (unsafe { qp.add(ti * ch * 4) }, cp, stp, unsafe {
                            op_.add(ti * ch * 4)
                        });
                    let mut chh = ch as i32;
                    let mut kk = k as i32;
                    self.kop(
                        "gdn_conv",
                        (ch as u32).div_ceil(64),
                        1,
                        1,
                        64,
                        &mut cargs!(&mut q, &mut c, &mut s_, &mut o_, &mut chh, &mut kk),
                    )?;
                }
                Ok(())
            }
            O::MoeTop10 {
                route,
                ids,
                wt,
                n_exp,
                k_sel,
            } => {
                // 라우팅이 새로 쓰였다 — 그룹화 캐시 무효화.
                self.moe_gen
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let (mut rp, mut ip, mut wp) = (self.fptr(route)?, self.fptr(ids)?, self.fptr(wt)?);
                let mut ne = n_exp as i32;
                let mut ks = k_sel as i32;
                let t = self.t_cur();
                // 워프 병렬판(2026-09-13) — 원판은 1스레드/토큰이라 디코드에서
                // 0.85ms/호출이었다(토큰당 48콜 = 41ms). 선택 로직은 동일해
                // 결과는 비트 동일.
                self.kop(
                    "q4_moe_top10_m",
                    t as u32,
                    1,
                    1,
                    32,
                    &mut cargs!(&mut rp, &mut ip, &mut wp, &mut ne, &mut ks),
                )
            }
            O::MoeWeightedSum { ys, wt, out, k, n } => {
                let (mut yp, mut wp, mut op_) = (self.fptr(ys)?, self.fptr(wt)?, self.fptr(out)?);
                let mut kk = k as i32;
                let mut nn = (n * self.t_cur()) as i32;
                self.kop(
                    "q4_moe_weighted_sum",
                    ((n * self.t_cur()) as u32).div_ceil(128),
                    1,
                    1,
                    128,
                    &mut cargs!(&mut yp, &mut wp, &mut op_, &mut kk, &mut nn),
                )
            }
            O::AxpyScaled { y, x, s, n } => {
                let (mut yp, mut xp, mut sp) = (self.fptr(y)?, self.fptr(x)?, self.fptr(s)?);
                let mut nn = n as i32;
                let t = self.t_cur();
                if t <= 1 {
                    self.kop(
                        "axpy_scaled",
                        (n as u32).div_ceil(128),
                        1,
                        1,
                        128,
                        &mut cargs!(&mut yp, &mut xp, &mut sp, &mut nn),
                    )
                } else {
                    // 토큰 배치: s[t] — per = 토큰당 원소 수
                    let mut pp = (n / t) as i32;
                    self.kop(
                        "q4_axpy_scaled_t",
                        (n as u32).div_ceil(128),
                        1,
                        1,
                        128,
                        &mut cargs!(&mut yp, &mut xp, &mut sp, &mut nn, &mut pp),
                    )
                }
            }
            ref other => Err(format!("q4acc: 프레임 op 미지원 {other:?}")),
        }
    }
}
