//! vkacc::ple — EwOps·PLE 원소연산 계열. (plans/90 B1b: gemv.rs 순수 이동)

use super::*;


impl llm170_core::matmul::EwOps for VkAcc {

    fn rms_norm(
        &self,
        xs: &[Vec<f32>],
        w: &[f32],
        eps: f32,
        outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        self.rms_norm_gpu(xs, w, eps, outs)
    }

    fn silu_mul(
        &self,
        gs: &[Vec<f32>],
        us: &[Vec<f32>],
        outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        self.silu_mul_gpu(gs, us, outs)
    }

    fn ffn_chain(
        &self,
        xs: &[Vec<f32>],
        gate_w: &Weight,
        up_w: &Weight,
        down_w: &Weight,
        xs_out: &mut [Vec<f32>],
    ) -> Result<(), String> {
        self.ffn_chain_gpu(xs, gate_w, up_w, down_w, xs_out)
    }

    /// plans/85 §1 — 디코드(t=1) shared expert gate+up: quant 1회 → gemv 2회
    /// → SiluMul. 스크래치 gh/uh는 풀 기반 frame_alloc/frame_free.
    /// frame_mm_group/frame_op가 각자 ctx를 잠그므로 여기엔 중첩 잠금이
    /// 없다(직전 시도의 self-deadlock 원인 — ctx.lock 보유 중 frame_alloc).
    fn shexp_gu(
        &self, x: u64, wg: &Weight, wu: &Weight, h: u64,
        _n_in: usize, n_hidden: usize,
    ) -> Result<(), String> {
        if std::env::var_os("LLM170_VK_SHEXP").is_some_and(|v| v == "0") {
            return Err("shexp_gu: 진단 킬스위치".into());
        }
        let gh = self.frame_alloc(n_hidden)?;
        let uh = self.frame_alloc(n_hidden)?;
        let r = self
            .frame_mm_group(x, &[*wg, *wu], &[gh, uh], 1)
            .and_then(|_| {
                self.frame_op(&llm170_core::matmul::FrameOp::SiluMul {
                    g: gh, u: uh, out: h, n: n_hidden,
                })
            });
        let _ = self.frame_free(gh);
        let _ = self.frame_free(uh);
        r
    }

    /// plans/85 §1 — 디코드(t=1) shared expert down+가산: gemv 1회 →
    /// mout += σ·dh (AxpyScaled — t=1이라 s[0] 판독과 정합).
    fn shexp_da(
        &self, h: u64, wd: &Weight, s: u64, mout: u64,
        n_in: usize, _n_hidden: usize,
    ) -> Result<(), String> {
        if self.frame_t.load(std::sync::atomic::Ordering::Relaxed) != 1 {
            return Err("shexp_da: t=1 전용 (frame_t≠1)".into());
        }
        let dh = self.frame_alloc(n_in)?;
        let r = self
            .frame_mm_group(h, &[*wd], &[dh], 1)
            .and_then(|_| {
                self.frame_op(&llm170_core::matmul::FrameOp::AxpyScaled {
                    y: mout, x: dh, s, n: n_in,
                })
            });
        let _ = self.frame_free(dh);
        r
    }

    /// plans/89 P1.4 — PLE 수학 디바이스판(디코드 t=1): hip q4_ple_* 3커널의
    /// VkAcc 발사. 링/워터마크·상수 캐시 (ptr,len) 동일 규약. 프리필(t>1)은
    /// Err → 엔진이 종전 호스트 브리지로.
    #[allow(clippy::too_many_arguments)]
    fn ple_math_dev(
        &self,
        res: u64,
        key: u64,
        value: u64,
        nk: &[f32],
        nq: &[f32],
        nc: &[f32],
        conv_w: &[f32],
        gated: u64,
        conv_out: u64,
        gate_out: u64,
        seq: usize,
        t: usize,
        eps: f32,
        n_embd: usize,
        hc: usize,
        kern: usize,
        dil: usize,
        hist: usize,
        host_ring: &[f32],
    ) -> Result<(), String> {
        if t != 1 {
            return Err("ple_math_dev: t=1 전용".into());
        }
        let hc_dim = hc * n_embd;
        let ring_bytes = hist * hc_dim * 4;
        let mut ctx = self.ctx.lock();
        self.frame_resume_batch(&mut ctx);
        // 링 + 워터마크(되감기면 호스트 링으로 리프레시).
        let rewind;
        let ringb;
        {
            let mut m = self.ple_rings.lock();
            let e = m.entry(seq).or_insert_with(|| (vkbuf_null(), 0));
            rewind = e.1 > t || e.0.ptr.is_null();
            e.1 = t;
            if e.0.ptr.is_null() {
                e.0 = ctx.alloc_host(ring_bytes)?;
            }
            if rewind {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        host_ring.as_ptr() as *const u8,
                        e.0.ptr,
                        hist * hc_dim * 4,
                    );
                }
            }
            ringb = e.0.buf;
        }
        // 상수 캐시 — 모델 가중 뷰(ptr,len 안정).
        let upload = |ctx: &mut VkCtx, s: &[f32]| -> Result<vk::Buffer, String> {
            let key = (s.as_ptr() as usize, s.len());
            let mut c = self.ple_consts.lock();
            if let Some(b) = c.get(&key) {
                return Ok(b.buf);
            }
            let b = ctx.alloc_host(s.len() * 4)?;
            unsafe { std::ptr::copy_nonoverlapping(s.as_ptr() as *const u8, b.ptr, s.len() * 4) };
            let buf = b.buf;
            c.insert(key, b);
            Ok(buf)
        };
        let nkb = upload(&mut ctx, nk)?;
        let nqb = upload(&mut ctx, nq)?;
        let ncb = upload(&mut ctx, nc)?;
        let cwb = upload(&mut ctx, conv_w)?;
        let rb = self.fbuf(res)?;
        let kb = self.fbuf(key)?;
        let vb = self.fbuf(value)?;
        let gb = self.fbuf(gated)?;
        let cob = self.fbuf(conv_out)?;
        let gob = self.fbuf(gate_out)?;
        // (1) gate+방송+그룹 norm.
        {
            let p = self.pipeline(&mut ctx, Slot::FnPleGate)?;
            let ds2 = ctx.bind_ds(&p, &[rb, kb, vb, nkb, nqb, ncb, gb, gob])?;
            let push = push_u32s(&[n_embd as u32, hc as u32, t as u32]);
            let mut p16 = eps.to_le_bytes().to_vec();
            p16.extend_from_slice(&push);
            ctx.run(p.pl, ds2, p.pipe, &p16, hc.div_ceil(8) as u32, t as u32, 1)?;
        }
        // (2) dilated conv + silu + 링 갱신.
        {
            let p = self.pipeline(&mut ctx, Slot::FnPleConv)?;
            let ds2 = ctx.bind_ds(&p, &[gb, cwb, ringb, cob])?;
            let push = push_u32s(&[
                hc_dim as u32, t as u32, kern as u32, dil as u32, hist as u32,
            ]);
            ctx.run(p.pl, ds2, p.pipe, &push, hc_dim.div_ceil(256) as u32, 1, 1)?;
        }
        // (3) 잔차.
        {
            let p = self.pipeline(&mut ctx, Slot::FnPleRes)?;
            let ds2 = ctx.bind_ds(&p, &[rb, vb, gob, cob])?;
            let push = push_u32s(&[n_embd as u32, hc as u32, t as u32]);
            ctx.run(p.pl, ds2, p.pipe, &push, n_embd.div_ceil(256) as u32, 1, 1)?;
        }
        Ok(())
    }
}



