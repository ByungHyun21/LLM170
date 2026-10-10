use super::*;

impl W4a16Dec {
    /// 배치 전문가 GEMV(상주) — tab 간접 참조, grid = n × nslots.
    /// xstride = 0(x 공통) 또는 k(슬롯/토큰별 활성), sp = 토큰당 슬롯 수
    /// (프리필 = top_k → x는 토큰 단위 [t][k], 디코드 = 1 → 슬롯별 [nslots][k]).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gemv_experts_launch(
        &self,
        base: usize,
        nslots: usize,
        x_dev: CUdeviceptr,
        xstride: usize,
        sp: usize,
        out_dev: CUdeviceptr,
        n: usize,
        k: usize,
    ) -> Result<(), String> {
        let f = self.cc.function("w4a16_gemv_experts_g32_bf16")?;
        let (mut p_t, mut p_b, mut p_i, mut p_ns) =
            (self.moe_dev_tab, base as i32, self.moe_idx, nslots as i32);
        let (mut p_x, mut p_xs, mut p_sp, mut p_o, mut p_n, mut p_k) = (
            x_dev,
            xstride as i32,
            sp as i32,
            out_dev,
            n as i32,
            k as i32,
        );
        self.cc.launch(
            f,
            (n * nslots) as u32,
            1,
            64,
            &mut crate::rawcuda::args::l10(
                &mut p_t, &mut p_b, &mut p_i, &mut p_ns, &mut p_x, &mut p_xs, &mut p_sp, &mut p_o,
                &mut p_n, &mut p_k,
            ),
        )
    }

    /// [P11] 전문가-우선 슬롯 순열 발사(프리필 전용) — w4a16_moe_align.
    /// n_exp ≤ 1024 계약(초과 시 호출부가 종전 경로로 폴백).
    pub(super) fn moe_align_launch(&self, nslots: usize) -> Result<(), String> {
        let f = self.cc.function("w4a16_moe_align")?;
        let (mut p_i, mut p_ns) = (self.moe_idx, nslots as i32);
        let (mut p_g, mut p_c, mut p_o) = (self.moe_gslot, self.moe_gcnt, self.moe_goff);
        let mut p_ne = self.n_experts as i32;
        self.cc.launch(
            f,
            1,
            1,
            256,
            &mut crate::rawcuda::args::l6(
                &mut p_i, &mut p_ns, &mut p_g, &mut p_c, &mut p_o, &mut p_ne,
            ),
        )
    }

    /// [P11] 그룹 mma GEMM(g32) 발사 — grid (전문가 × n타일), M=전문가 슬롯 수.
    pub(super) fn gemm_g32_mma_grp_launch(
        &self,
        base: usize,
        x_dev: CUdeviceptr,
        xstride: usize,
        sp: usize,
        out_dev: CUdeviceptr,
        n: usize,
        k: usize,
    ) -> Result<(), String> {
        let f = self.cc.function("w4a16_gemm_g32_mma_grp")?;
        let (mut p_t, mut p_b) = (self.moe_dev_tab, base as i32);
        let (mut p_g, mut p_c, mut p_of) = (self.moe_gslot, self.moe_gcnt, self.moe_goff);
        let (mut p_x, mut p_xs, mut p_sp, mut p_o, mut p_n, mut p_k) = (
            x_dev,
            xstride as i32,
            sp as i32,
            out_dev,
            n as i32,
            k as i32,
        );
        self.cc.launch(
            f,
            self.n_experts as u32,
            n.div_ceil(GEMM_GRP_N) as u32,
            256,
            &mut crate::rawcuda::args::l11(
                &mut p_t, &mut p_b, &mut p_g, &mut p_c, &mut p_of, &mut p_x, &mut p_xs, &mut p_sp,
                &mut p_o, &mut p_n, &mut p_k,
            ),
        )
    }

    /// 선택 순서 가중 누적 — y[ti] = Σ_{s∈ti} w[s]·d[s][i] (sp = 토큰당 슬롯).
    pub(super) fn moe_accum_dev(
        &mut self,
        w_dev: CUdeviceptr,
        d_dev: CUdeviceptr,
        y_dev: CUdeviceptr,
        sp: usize,
        nslots: usize,
        n: usize,
    ) -> Result<(), String> {
        let f = self.cc.function("w4a16_moe_accum")?;
        let (mut p_w, mut p_d, mut p_y) = (w_dev, d_dev, y_dev);
        let (mut p_sp, mut p_ns, mut nn) = (sp as i32, nslots as i32, n as i32);
        self.cc.launch(
            f,
            n.div_ceil(256) as u32,
            1,
            256,
            &mut crate::rawcuda::args::l6(
                &mut p_w, &mut p_d, &mut p_y, &mut p_sp, &mut p_ns, &mut nn,
            ),
        )
    }

    /// MoE FFN(35B-A3B) — 라우터(bf16 GEMV→호스트 top-k) + 전문가 스트리밍
    /// GEMV + shared. 반환 = self.dmo(잔차 ab로 소비).
    /// 시맨틱은 CPU 스테이지(core qwen35::stages::moe)와 동일: 라우터 전체
    /// softmax → top-k → 재정규화, shared = sigmoid(sgate)·MLP.
    pub(super) fn moe_ffn_dev(
        &mut self,
        il: usize,
        xn: CUdeviceptr,
    ) -> Result<CUdeviceptr, String> {
        let _g = self.cc.guard()?;
        let n_exp = self.n_experts;
        if n_exp == 0 || self.moe_tab.len() < (il + 1) * n_exp * 3 {
            return Err("moe: 구성/전문가 테이블 미등록".into());
        }
        self.ensure_moe_bufs()?;
        // [2026-10-09 P1] 상주는 라우터를 디바이스에서 마무리(top-k 디바이스) —
        // d2h+sync+호스트 moe_topk+h2d 왕복(층당 1회) 제거, 그래프 캡처 가능화.
        // 스트리밍은 전문가 파일 스테이징에 호스트 선택이 필요해 종전 경로 유지
        // (zero_dev도 axpy 누적 전용 — 상주는 moe_accum이 전 행 덮어씀, P2).
        if self.moe_resident {
            self.moe_route_dev(il, xn)?;
            self.moe_experts_batch(il, xn, self.top_k)?;
        } else {
            let sel = self.moe_route(il, xn)?;
            Self::zero_dev(&self.cc, self.dmo, self.hidden * 4)?;
            self.moe_experts_streaming(il, xn, &sel)?;
        }
        self.moe_shared(il, xn)?;
        Ok(self.dmo)
    }

    /// 라우터 디바이스 상주(P1) — gate GEMV → w4a16_moe_topk(idx/wt 디바이스).
    pub(super) fn moe_route_dev(&mut self, il: usize, xn: CUdeviceptr) -> Result<(), String> {
        self.plain_gemv_launch(&format!("blk.{il}.moe_gate.weight"), xn, self.drt)?;
        let f = self.cc.function("w4a16_moe_topk")?;
        let (mut p_lg, mut p_ix, mut p_wt) = (self.drt, self.moe_idx, self.moe_wt);
        let (mut nn, mut kk) = (self.n_experts as i32, self.top_k as i32);
        self.cc.launch(
            f,
            1,
            1,
            32,
            &mut crate::rawcuda::args::l5(&mut p_lg, &mut p_ix, &mut p_wt, &mut nn, &mut kk),
        )
    }

    /// 라우터 — bf16 GEMV(원시 xn) → 로짓 판독 → 호스트 top-k 선택.
    pub(super) fn moe_route(
        &mut self,
        il: usize,
        xn: CUdeviceptr,
    ) -> Result<Vec<(usize, f32)>, String> {
        let n_exp = self.n_experts;
        self.plain_gemv_launch(&format!("blk.{il}.moe_gate.weight"), xn, self.drt)?;
        let mut lb = vec![0u8; n_exp * 4];
        self.cc.d2h_async(lb.as_mut_ptr(), self.drt, n_exp * 4)?;
        self.cc.sync()?;
        let logits: Vec<f32> = lb
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        Ok(moe_topk(&logits, self.top_k))
    }

    /// 전문가 배치(상주) — 테이블 간접 GEMV + 가중 누적.
    /// [P1] idx/wt는 호출자가 디바이스에 기록한다(moe_route_dev) — 종전
    /// h2d 2회(층당) 제거. gDN 경로(moe_ffn_dev_t)는 호스트 선택이 남아
    /// 이 함수 앞에서 moe_idx/moe_wt를 h2d로 채운 뒤 ns를 넘긴다.
    pub(super) fn moe_experts_batch(
        &mut self,
        il: usize,
        xn: CUdeviceptr,
        ns: usize,
    ) -> Result<(), String> {
        let n_exp = self.n_experts;
        let h = self.hidden;
        let n_ff = self.moe_ffn;
        // gate/up 배치(x 공통) → ew → down 배치(x 슬롯별) → 가중 누적.
        let base = il * n_exp * 3;
        self.gemv_experts_launch(base, ns, xn, 0, ns, self.dexp_gate, n_ff, h)?;
        self.gemv_experts_launch(base + 1, ns, xn, 0, ns, self.dexp_up, n_ff, h)?;
        self.ew_dev(self.dexp_gate, self.dexp_up, self.dexp_act, ns * n_ff)?;
        self.gemv_experts_launch(base + 2, ns, self.dexp_act, n_ff, 1, self.dexp_dn, h, n_ff)?;
        self.moe_accum_dev(self.moe_wt, self.dexp_dn, self.dmo, ns, ns, h)
    }

    /// 전문가 스트리밍(비상주) — 스테이징 1쌍 재사용 h2d + 개별 GEMV.
    pub(super) fn moe_experts_streaming(
        &mut self,
        il: usize,
        xn: CUdeviceptr,
        sel: &[(usize, f32)],
    ) -> Result<(), String> {
        let n_exp = self.n_experts;
        let h = self.hidden;
        let n_ff = self.moe_ffn;
        let [s0, s1, _s1b, s2, s3] = self.dchain;
        if sel.is_empty() {
            return Ok(());
        }
        // [P7 2026-10-09] 층 단위 배치 스테이징 — (top_k×3) q/s 슬라이스를
        // async 복사로 전부 올린 뒤 sync 1회(종전 proj마다 h2d_chunked —
        // proj당 sync 2회 × 3 × top_k = 층당 수십 sync). 소스는 호스트
        // 스테이징(C2)·mmap — 둘 다 호출 내 수명이면 충분(드라이버 스테이징).
        let qsz = h * n_ff / 2;
        let ssz = h * n_ff / self.moe_group * 2;
        debug_assert!(
            self.dstg_cap.0 >= qsz * sel.len() * 3 && self.dstg_cap.1 >= ssz * sel.len() * 3
        );
        for (j, &(e, _)) in sel.iter().enumerate() {
            let base = (il * n_exp + e) * 3;
            for p in 0..3 {
                let (qp, ql, sp, sl) = self.moe_tab[base + p];
                let qd = self.dstg_q + ((j * 3 + p) * qsz) as u64;
                let sd = self.dstg_s + ((j * 3 + p) * ssz) as u64;
                // SAFETY: tab 항목은 set_expert_table 수명 계약(mmap/호스트
                // 스테이징), ql/sl은 위 qsz/ssz 형상 계약을 따른다.
                let qb = unsafe { std::slice::from_raw_parts(qp as *const u8, ql as usize) };
                let sb = unsafe { std::slice::from_raw_parts(sp as *const u8, sl as usize) };
                self.cc.h2d_async(qd, qb)?;
                self.cc.h2d_async(sd, sb)?;
            }
        }
        self.cc.sync()?;
        for (j, &(_e, w)) in sel.iter().enumerate() {
            let q0 = self.dstg_q + ((j * 3) * qsz) as u64;
            let p0 = self.dstg_s + ((j * 3) * ssz) as u64;
            let q1 = q0 + qsz as u64;
            let p1 = p0 + ssz as u64;
            let q2 = q0 + (2 * qsz) as u64;
            let p2 = p0 + (2 * ssz) as u64;
            // proj 순서 = 테이블 조립 순서(gate, up, down) — 스테이징과 일치.
            self.gemv_launch_raw(q0, p0, n_ff, h, xn, s0)?;
            self.gemv_launch_raw(q1, p1, n_ff, h, xn, s1)?;
            self.ew_dev(s0, s1, s2, n_ff)?;
            self.gemv_launch_raw(q2, p2, h, n_ff, s2, s3)?;
            self.axpy_dev(w, s3, self.dmo, h)?;
        }
        Ok(())
    }

    /// shared 전문가 — sigmoid(sgate·xn)·down(silu(gate·xn)·up·xn).
    pub(super) fn moe_shared(&mut self, il: usize, xn: CUdeviceptr) -> Result<(), String> {
        if self.shared_ffn == 0 {
            return Ok(());
        }
        let h = self.hidden;
        let sf = self.shared_ffn;
        let [s0, s1, _s1b, s2, s3] = self.dchain;
        self.plain_gemv_launch(&format!("blk.{il}.moe_shared_gate.weight"), xn, s0)?;
        self.plain_gemv_launch(&format!("blk.{il}.moe_shared_up.weight"), xn, s1)?;
        self.ew_dev(s0, s1, s2, sf)?;
        self.plain_gemv_launch(&format!("blk.{il}.moe_shared_down.weight"), s2, s3)?;
        self.plain_gemv_launch(&format!("blk.{il}.moe_shared_sgate.weight"), xn, self.drt)?;
        self.shared_add_dev(self.drt, s3, self.dmo, h, 1)
    }

    /// MoE FFN 배치(t≤8) — 라우터 플레인 GEMM → 디바이스 top-k(t×top_k 슬롯,
    /// 토큰 우선) → 전문가 배치 GEMV(gate/up x=토큰 단위 sp=top_k, down x=슬롯
    /// 단위 sp=1) → 토큰별 누적 → shared. **상주 모드 전용**(스트리밍 프리필은
    /// 미구현 — 호출부가 t=1로 떨어뜨린다).
    pub(super) fn moe_ffn_dev_t(
        &mut self,
        il: usize,
        xn: CUdeviceptr,
        t: usize,
    ) -> Result<CUdeviceptr, String> {
        let _g = self.cc.guard()?;
        let n_exp = self.n_experts;
        let h = self.hidden;
        let n_ff = self.moe_ffn;
        let tk = self.top_k;
        if n_exp == 0 || self.moe_tab.len() < (il + 1) * n_exp * 3 {
            return Err("moe: 구성/전문가 테이블 미등록".into());
        }
        if !self.moe_resident {
            return Err("moe t>1: 상주 모드 전용(스트리밍 프리필 미구현)".into());
        }
        self.ensure_moe_bufs()?;
        // 1) 라우터 [t][n_exp] — 플레인 GEMM → **디바이스 top-k**(P11).
        // 종전: d2h(32KB)+sync+호스트 전체 정렬(512×t)이 층·청크마다 — 프리필의
        // ~10%. 시맨틱은 moe_topk 미러(softmax→k라운드→재정규화).
        // 실측(2026-10-09): 35B 512토큰 프리필 1294→1088ms.
        self.plain_gemm_launch(&format!("blk.{il}.moe_gate.weight"), xn, self.drt, t)?;
        {
            let f = self.cc.function("w4a16_moe_topk_t")?;
            let (mut p_lg, mut p_ix, mut p_wt) = (self.drt, self.moe_idx, self.moe_wt);
            let (mut p_t, mut p_n, mut p_k) = (t as i32, n_exp as i32, tk as i32);
            self.cc.launch(
                f,
                t.div_ceil(8) as u32,
                1,
                256,
                &mut crate::rawcuda::args::l6(
                    &mut p_lg, &mut p_ix, &mut p_wt, &mut p_t, &mut p_n, &mut p_k,
                ),
            )?;
        }
        let ns = t * tk;
        // 2) 전문가 배치.
        if llm170_diag::flag::on_nonzero("LLM170_MOE_DBG") {
            // xn(정규화 출력) 행별 NaN — 업스트림 vs 전문가 GEMV 판별.
            let mut vb = vec![0u8; t * h * 4];
            self.cc.d2h_async(vb.as_mut_ptr(), xn, t * h * 4)?;
            self.cc.sync()?;
            let v: Vec<f32> = vb
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            let rows: Vec<usize> = (0..t)
                .filter(|&r| v[r * h..(r + 1) * h].iter().any(|x| x.is_nan()))
                .collect();
            eprintln!("[t-dbg] xn nan-rows={rows:?} t={t}");
        }
        // (P11) idx/wt는 이미 디바이스에 있다(topk_t) — h2d 없음.
        let base = il * n_exp * 3;
        // [P11] 프리필(t>1)은 전문가-우선 정렬 + 그룹 GEMV — 같은 전문가의
        // 슬롯을 연속 처리해 가중치 행을 L2 재사용(슬롯별 산술 동일 = 비트 동일).
        let group = t > 1 && n_exp <= 1024;
        if group {
            // [P11] 그룹 mma GEMM — 전문가별 슬롯 묶음(M=슬롯 수), T1 계약 미러.
            self.moe_align_launch(ns)?;
            self.gemm_g32_mma_grp_launch(base, xn, h, tk, self.dexp_gate, n_ff, h)?;
            self.gemm_g32_mma_grp_launch(base + 1, xn, h, tk, self.dexp_up, n_ff, h)?;
        } else {
            self.gemv_experts_launch(base, ns, xn, h, tk, self.dexp_gate, n_ff, h)?;
            self.gemv_experts_launch(base + 1, ns, xn, h, tk, self.dexp_up, n_ff, h)?;
        }
        self.ew_dev(self.dexp_gate, self.dexp_up, self.dexp_act, ns * n_ff)?;
        if llm170_diag::flag::on_nonzero("LLM170_MOE_DBG") {
            let mut vb = vec![0u8; ns * n_ff * 4];
            self.cc
                .d2h_async(vb.as_mut_ptr(), self.dexp_act, ns * n_ff * 4)?;
            self.cc.sync()?;
            let v: Vec<f32> = vb
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            let bad: Vec<usize> = (0..ns)
                .filter(|&s| v[s * n_ff..(s + 1) * n_ff].iter().any(|x| x.is_nan()))
                .collect();
            eprintln!("[t-dbg] act nan-slots={bad:?} ns={ns}");
        }
        if group {
            self.gemm_g32_mma_grp_launch(base + 2, self.dexp_act, n_ff, 1, self.dexp_dn, h, n_ff)?;
        } else {
            self.gemv_experts_launch(base + 2, ns, self.dexp_act, n_ff, 1, self.dexp_dn, h, n_ff)?;
        }
        self.moe_accum_dev(self.moe_wt, self.dexp_dn, self.dmo, tk, ns, h)?;
        self.moe_shared_t(il, xn, t)?;
        Ok(self.dmo)
    }

    /// shared 전문가 배치(t≤8) — 플레인 GEMM ×3 + 토큰별 게이트 가산.
    pub(super) fn moe_shared_t(
        &mut self,
        il: usize,
        xn: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        if self.shared_ffn == 0 {
            return Ok(());
        }
        let h = self.hidden;
        let sf = self.shared_ffn;
        let [s0, s1, _s1b, s2, s3] = self.dchain;
        self.plain_gemm_launch(&format!("blk.{il}.moe_shared_gate.weight"), xn, s0, t)?;
        self.plain_gemm_launch(&format!("blk.{il}.moe_shared_up.weight"), xn, s1, t)?;
        self.ew_dev(s0, s1, s2, t * sf)?;
        self.plain_gemm_launch(&format!("blk.{il}.moe_shared_down.weight"), s2, s3, t)?;
        self.plain_gemm_launch(
            &format!("blk.{il}.moe_shared_sgate.weight"),
            xn,
            self.drt,
            t,
        )?;
        self.shared_add_dev(self.drt, s3, self.dmo, h, t)
    }
}
