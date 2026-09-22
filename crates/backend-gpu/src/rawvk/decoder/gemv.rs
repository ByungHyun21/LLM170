//! vk decoder 런치·양자화·GEMV/GEMM 유틸 (plans/79 B).

use super::*;

impl DecoderState {

    /// 파이프라인 지연 생성 캐시.
    pub(super) fn pipe(&mut self, name: &'static str, spv: &[u8], n_buf: u32, pb: u32) -> Result<&Pipes, String> {
        if !self.pipes.contains_key(name) {
            let p = self.ctx.pipeline_pipes(spv, n_buf, pb)?;
            self.pipes.insert(name, p);
        }
        Ok(self.pipes.get(name).unwrap())
    }

    /// 바인딩+런치 (배치 모드 자동 — fresh ds). 배리어 포함.
    pub(super) fn run_pipe(&mut self, name: &'static str, spv: &[u8], n_buf: u32, pb: u32, bufs: &[vk::Buffer], push: &[u8], gx: u32, gy: u32, gz: u32) -> Result<(), String> {
        self.run_pipe_b(name, spv, n_buf, pb, bufs, push, gx, gy, gz, true)
    }

    /// 바인딩+런치 — bar=false면 직후 배리어 생략 (독립 병렬 그룹 내부).
    /// 그룹 마지막 디스패치는 반드시 bar=true로 종결해야 소비자가 안전하다.
    pub(super) fn run_pipe_b(&mut self, name: &'static str, spv: &[u8], n_buf: u32, pb: u32, bufs: &[vk::Buffer], push: &[u8], gx: u32, gy: u32, gz: u32, bar: bool) -> Result<(), String> {
        let t0k = std::time::Instant::now();
        // 배치 자동 분할 — 디스패치 2048마다 제출·대기·재시작 (세트/CMDBUF 누적 방지;
        // 풀 상한 4096세트 이내. 512→2048: 스텝당 중간 드레인 제거, plans/36 G3).
        if self.ctx.batching.load(std::sync::atomic::Ordering::Relaxed) {
            self.split_ctr += 1;
            if self.split_ctr >= std::env::var("LLM170_VK_SPLIT").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(2048) {
                self.split_ctr = 0;
                if std::env::var_os("LLM170_DBG_REC").is_some() {
                    let td = std::time::Instant::now();
                    self.ctx.end_batch_wait()?;
                    self.dbg_drain_ms += td.elapsed().as_secs_f64() * 1e3;
                } else {
                    self.ctx.end_batch_wait()?;
                }
                self.ctx.begin_batch()?;
            }
        }
        self.ctx.nobar_next.set(!bar);
        crate::rawvk::context::site::set_tag(name);
        let p = *self.pipe(name, spv, n_buf, pb)?;
        let ds = self.ctx.bind_ds(&p, bufs)?;
        let r = self.ctx.run(p.pl, ds, p.pipe, push, gx, gy, gz);
        if self.ktime {
            let e = t0k.elapsed().as_secs_f64() * 1e3;
            let key = self.kkey.borrow_mut().take().unwrap_or_else(|| name.to_string());
            let ent = self.ktimes.entry(key).or_insert((0.0f64, 0u64));
            ent.0 += e;
            ent.1 += 1;
        }
        r
    }

    pub(super) fn push_u32s(vals: &[u32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// quant_f16: [t][n] f32 → f16 (f16-B 타일 경로, plans/46).
    pub(super) fn quant_f16(&mut self, src: vk::Buffer, dst: vk::Buffer, n: usize, t: usize) -> Result<(), String> {
        let push = Self::push_u32s(&[n as u32, t as u32]);
        self.run_pipe("quant_f16", QUANT_F16_SPV, 2, 8,
            &[src, dst], &push, (n * t).div_ceil(64) as u32, 1, 1)
    }

    /// quant: [t][n] f32 → xq (q8 레이아웃).
    pub(super) fn quant(&mut self, src: vk::Buffer, xq: vk::Buffer, n: usize, t: usize) -> Result<(), String> {
        let xq_w = crate::rawvk::gemv::xq_words(n);
        let push = Self::push_u32s(&[n as u32, t as u32, xq_w as u32]);
        self.run_pipe("quant", crate::rawvk::gemv::QUANT_SPV, 2, 12,
            &[src, xq], &push, (n / 32 + 63) as u32 / 64, t as u32, 1)
    }


    /// gemv8_q5 (plans/33) — llama mul_mat_vec_q5_k 완전 포트 (typed u16 로드,
    /// SIMD-in-register 니블, fma 체인). 웜 162GB/s (역대 최고). LLM170_G8=1.
    /// bar=false: 독립 병렬 그룹 내부 (직후 배리어 생략).
    pub(super) fn gemv8_q5(&mut self, xn: vk::Buffer, wkey: &str, out: vk::Buffer, t: usize, bar: bool) -> Result<(), String> {
        if std::env::var_os("LLM170_DBG_G8").is_some() {
            eprintln!("[dbg_g8] t={t} {wkey}");
        }
        let (wbufs, ty, ni, no) = self.w.get(wkey).cloned().ok_or(format!("가중치 없음: {wkey}"))?;
        if ty != 13 && ty != 12 && ty != 23 && ty != 11 && ty != 14 && ty != 8 && ty != 20 && ty != 21 {
            return Err("gemv8: q3_K/q4_K/q5_K/q6_K/q8_0/iq4_xs/iq4_nl/iq3_s만".into());
        }

        let mut binds: Vec<vk::Buffer> = wbufs.iter().map(|b| b.buf).collect();
        while binds.len() < 8 {
            binds.push(self.dummy.buf);
        }
        binds.push(xn);
        binds.push(out);
        if ty == 23 {
            binds.push(self.ktab.buf);
        } else if ty == 21 {
            binds.push(self.grid3s.buf);   // IQ3S_GRID 512워드 (iq4_nl ktab과 별개)
        }
        // nlb (plans/46) — IQ4_NL 전용 (구 폴백 대체, t=1 스테디 ~2.8ms/토큰 절감).
        if ty == 20 && std::env::var("LLM170_VK_NLB").map(|v| v == "0").unwrap_or(true) {
            let push = Self::push_u32s(&[ni as u32, no as u32, t as u32, 0, 0, 2]);
            return self.run_pipe_b("gemv8_nlb", GEMV8_NLB_SPV, 10, 24, &binds, &push,
                1, no.div_ceil(2) as u32, t as u32, bar);
        }
        // i3s (plans/46) — IQ3_S 전용 (마지막 폴백 제거, quant.rs deq_iq3_s 미러).
        if ty == 21 && std::env::var("LLM170_VK_I3S").map(|v| v == "0").unwrap_or(true) {
            let push = Self::push_u32s(&[ni as u32, no as u32, t as u32, 0, 0, 2]);
            return self.run_pipe_b("gemv8_i3s", GEMV8_I3S_SPV, 11, 24, &binds, &push,
                1, no.div_ceil(2) as u32, t as u32, bar);
        }
        // xsb (plans/40) — llama generic dmmv 구조 × 검증 xs 디코드: 125→182GB/s.
        if ty == 23 && std::env::var("LLM170_VK_XSB").map(|v| v == "0").unwrap_or(true) {
            let push = Self::push_u32s(&[ni as u32, no as u32, t as u32, 0, 0, 2]);
            return self.run_pipe_b("gemv8_xsb", GEMV8_XSB_SPV, 11, 24, &binds, &push,
                1, no.div_ceil(2) as u32, t as u32, bar);
        }
        let rpf: u32 = if no < 4096 { 1 } else { 2 };   // llama NUM_ROWS=2
        if ty == 8 {
            // q8b (plans/40) — llama generic dmmv 구조: 87→329GB/s. LLM170_VK_Q8B=0 옵트아웃.
            if std::env::var("LLM170_VK_Q8B").map(|v| v == "0").unwrap_or(true) {
                let push = Self::push_u32s(&[ni as u32, no as u32, t as u32, 0, 0, 2]);
                return self.run_pipe_b("gemv8_q8b", GEMV8_Q8B_SPV, 10, 24, &binds, &push,
                    1, no.div_ceil(2) as u32, t as u32, bar);
            }
            // q8_0 — 34B 블록 (plans/40: 소형 straggler 레이턴시 해소)
            let cw = wbufs.first().map(|b| b.bytes / 4).unwrap_or(1) as u32;
            let cw = cw.next_power_of_two();
            let cw_log2 = 31u32 - cw.leading_zeros();
            let cw_mask = cw - 1u32;
            let push = Self::push_u32s(&[ni as u32, no as u32, t as u32, cw_log2, cw_mask, rpf]);
            return self.run_pipe_b("gemv8_q8", GEMV8_Q8_SPV, 10, 24, &binds, &push,
                1, no.div_ceil(rpf as usize) as u32, t as u32, bar);
        }
        let (pname, spv8, n_kb8) = match ty {
            23 => ("gemv8_xs", GEMV8_XS_SPV, 11),
            11 => ("gemv8_q3", GEMV8_Q3_SPV, 10),
            _ => ("", &[][..], 0),
        };
        if ty == 11 && std::env::var("LLM170_VK_Q3B").map(|v| v == "0").unwrap_or(true) {
            // q3b (plans/40) — llama dmmv 구조: 62→122GB/s, max|D|=0.
            let push = Self::push_u32s(&[ni as u32, no as u32, t as u32, 0, 0, 2]);
            return self.run_pipe_b("gemv8_q3b", GEMV8_Q3B_SPV, 10, 24, &binds, &push,
                1, no.div_ceil(2) as u32, t as u32, bar);
        }
        if ty == 23 || ty == 11 {
            let cw = wbufs.first().map(|b| b.bytes / 4).unwrap_or(1) as u32;
            let cw = cw.next_power_of_two();
            let cw_log2 = 31u32 - cw.leading_zeros();
            let cw_mask = cw - 1;
            let push = Self::push_u32s(&[ni as u32, no as u32, t as u32, cw_log2, cw_mask, rpf]);
            return self.run_pipe_b(pname, spv8, n_kb8, 24, &binds, &push,
                1, no.div_ceil(rpf as usize) as u32, t as u32, bar);
        }
        if ty == 12 {
            // q4b (plans/40) — llama mul_mat_vec_q4_k 이식: 152→272GB/s. LLM170_VK_Q4B=0 옵트아웃.
            if std::env::var("LLM170_VK_Q4B").map(|v| v == "0").unwrap_or(true) {
                let push = Self::push_u32s(&[ni as u32, no as u32, t as u32, 0, 0, 2]);
                return self.run_pipe_b("gemv8_q4b", GEMV8_Q4B_SPV, 10, 24, &binds, &push,
                    1, no.div_ceil(2) as u32, t as u32, bar);
            }
            // q4 — u32 워드 단위 (동일 WG 워커)
            let cw = wbufs.first().map(|b| b.bytes / 4).unwrap_or(1) as u32;
            let cw = cw.next_power_of_two();
            let cw_log2 = 31u32 - cw.leading_zeros();
            let cw_mask = cw - 1;
            let push = Self::push_u32s(&[ni as u32, no as u32, t as u32, cw_log2, cw_mask, rpf]);
            return self.run_pipe_b("gemv8_q4", GEMV8_Q4_SPV, 10, 24, &binds, &push,
                1, no.div_ceil(rpf as usize) as u32, t as u32, bar);
        }
        if ty == 14 {
            // q6b (plans/40) — llama mul_mat_vec_q6_k 충실 이식(sccache): +35%. LLM170_VK_Q6B=0 옵트아웃.
            if std::env::var("LLM170_VK_Q6B").map(|v| v == "0").unwrap_or(true) {
                let push = Self::push_u32s(&[ni as u32, no as u32, t as u32, 0, 0, 2]);
                return self.run_pipe_b("gemv8_q6b", GEMV8_Q6B_SPV, 10, 24, &binds, &push,
                    1, no.div_ceil(2) as u32, t as u32, bar);
            }
            // q6 — u16 뷰 (105 u16/블록), llama mul_mat_vec_q6_k 직역 (plans/36 G1)
            let cw2 = wbufs.first().map(|b| b.bytes / 2).unwrap_or(1) as u32;
            let cw2 = cw2.next_power_of_two();
            let cw2_log2 = 31u32 - cw2.leading_zeros();
            let cw2_mask = cw2 - 1;
            let push = Self::push_u32s(&[ni as u32, no as u32, t as u32, cw2_log2, cw2_mask, rpf]);
            return self.run_pipe_b("gemv8_q6", GEMV8_Q6_SPV, 10, 24, &binds, &push,
                1, no.div_ceil(rpf as usize) as u32, t as u32, bar);
        }
        // q5b (plans/40) — llama mul_mat_vec_q5_k 충실 이식 (64스레드·2행·vec4).
        // 단일 청크 typed 뷰 — 143→225GB/s. LLM170_VK_Q5B=0 옵트아웃.
        if std::env::var("LLM170_VK_Q5B").map(|v| v == "0").unwrap_or(true) {
            // plans/46: NUM_ROWS 실험 — llama GCN은 rm_kq=4. t=1이 지연 바운드(f16 2배
            // 바이트에 -1.8%뿐)이므로 행/WG 증가로 ILP 상향. 기본 2, LLM170_VK_NR로 변경.
            let nr: u32 = std::env::var("LLM170_VK_NR").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
            let nr = nr.clamp(1, 4);
            let push = Self::push_u32s(&[ni as u32, no as u32, t as u32, 0, 0, nr]);
            return self.run_pipe_b("gemv8_q5b", GEMV8_Q5B_SPV, 10, 24, &binds, &push,
                1, no.div_ceil(nr as usize) as u32, t as u32, bar);
        }
        // q5 — u16 단위 청크 상수 (typed 뷰), 첫 버퍼 실측 크기 → pow2ceil
        let cw2 = wbufs.first().map(|b| b.bytes / 2).unwrap_or(1) as u32;
        let cw2 = cw2.next_power_of_two();
        let cw2_log2 = 31u32 - cw2.leading_zeros();
        let cw2_mask = cw2 - 1;
        let push = Self::push_u32s(&[ni as u32, no as u32, t as u32, cw2_log2, cw2_mask, rpf]);
        self.run_pipe_b("gemv8_q5", GEMV8_Q5_SPV, 10, 24, &binds, &push,
            1, no.div_ceil(rpf as usize) as u32, t as u32, bar)
    }
    /// gemv 래우터 — t<16은 gemv8(f32 직결, llama 포트), 그 외·미지원 타입은
    /// quant+gemv3(범용 정수 경로). LLM170_G8=0 킬스위치.
    /// 2026-09-08 A/B: q6_K도 gemv3+quant가 gemv6_q6보다 우위(tg32 7.06 vs 6.71) —
    /// gemv4/5/6/7 세대 전원 삭제(plans/35 P2).
    pub(super) fn gemv_w(&mut self, qsrc: vk::Buffer, xq: vk::Buffer, wkey: &str, out: vk::Buffer, t: usize, nq: usize) -> Result<(), String> {
        let g8_off = std::env::var("LLM170_G8").map(|v| v == "0").unwrap_or(false);
        if t < 16 && !g8_off
            && self.gemv8_q5(qsrc, wkey, out, t, true).is_ok() {
                return Ok(());
            }
        // plans/46: N토큰 gemv (q5_K, t≥2, LLM170_VK_Q5N=1 옵트인) — f32 활성 직접
        // 사용, 가중 WG당 1회 판독(그리드 z=토큰블록×8 — 서로 다른 z가 같은 행을
        // 동시에 읽어 L2 병합). coopmat 타일 대신 gemv8 접근의 f32 누산.
        if t >= 2 && std::env::var("LLM170_VK_Q5N").map(|v| v == "1").unwrap_or(false)
            && let Some((wbufs2, ty2, ni2, no2)) = self.w.get(wkey).cloned()
                && ty2 == 13 {
                    let mut binds2: Vec<vk::Buffer> = wbufs2.iter().map(|b| b.buf).collect();
                    while binds2.len() < 8 {
                        binds2.push(self.dummy.buf);
                    }
                    binds2.push(qsrc);
                    binds2.push(out);
                    let push2 = Self::push_u32s(&[ni2 as u32, no2 as u32, t as u32, 0, 0, 2]);
                    let tb = 8usize;
                    return self.run_pipe_b("gemv8_q5n", GEMV8_Q5N_SPV, 10, 24, &binds2, &push2,
                        1, no2.div_ceil(2) as u32, t.div_ceil(tb) as u32, true);
                }
        // plans/46 f16-B: q5 프리필을 f16 활성 직독 타일로 (quant f16화 + load_b 직독).
        if t >= 16 && std::env::var("LLM170_VK_F16B").map(|v| v == "1").unwrap_or(false)
            && let Some((wbufs2, ty2, ni2, _no2)) = self.w.get(wkey).cloned()
                && ty2 == 13 {
                    self.quant_f16(qsrc, self.b_xf16.buf, nq, t)?;
                    let mut binds2: Vec<vk::Buffer> = wbufs2.iter().map(|b| b.buf).collect();
                    while binds2.len() < 8 {
                        binds2.push(self.dummy.buf);
                    }
                    binds2.push(self.dummy.buf);   // binding 8: q8 뷰(사용안함)
                    binds2.push(out);                 // binding 9
                    binds2.push(self.b_xf16.buf);    // binding 10: f16 뷰
                    // push: [n_in, n_out, xq_w=n_in(f16 스트라이드), t, tok_base]
                    let nrows = (_no2 as u32).div_ceil(64);
                    let gy = (t as u32).div_ceil(64);
                    let push2 = Self::push_u32s(&[ni2 as u32, _no2 as u32, ni2 as u32, 64u32, 0u32]);
                    return self.run_pipe_b("tile_ms4gy_f16b", TILE_MS4GY_F16B_SPV, 11, 20,
                        &binds2, &push2, gy, nrows, 1, true);
                }
        self.quant(qsrc, xq, nq, t)?;
        self.gemv(xq, wkey, out, t)
    }


    /// HIP 기본 WMMA와 동일 정확도 클래스 maxrel ~4.9e-4, argmax 안정).
    /// LLM170_VK_NOTILE=1이면 항상 gemv3 정밀 경로.
    pub(super) fn gemv(&mut self, xq: vk::Buffer, wkey: &str, out: vk::Buffer, t: usize) -> Result<(), String> {
        self.gemv_bar(xq, wkey, out, t, true)
    }

    /// bar=false: 독립 그룹 내부 — 최종 디스패치 직후 배리어 생략.
    pub(super) fn gemv_bar(&mut self, xq: vk::Buffer, wkey: &str, out: vk::Buffer, t: usize, bar: bool) -> Result<(), String> {
        let (_, ty, _, no) = self.w.get(wkey).cloned().ok_or(format!("가중치 없음: {wkey}"))?;
        if std::env::var_os("LLM170_DBG_TILE").is_some() {
            eprintln!("[dbg_tile] t={t} ty={ty} no={no} {wkey}");
        }
        let tile_min: usize = if std::env::var_os("LLM170_VK_TILE1").is_some() { 1 } else { 16 };
        // f16 캐시 경로 (plans/39) — 루프 내 디양자화 없는 통일 타일
        if t >= tile_min
            && std::env::var_os("LLM170_VK_NOTILE").is_none()
            && std::env::var_os("LLM170_VK_NOF16W").is_none()
            && self.f16w.contains_key(wkey)
        {
            let xq_w = no; // 자리표시 — 아래에서 ni 기반 재계산
            let _ = xq_w;
            let ni_f = self.w.get(wkey).map(|e| e.2).unwrap_or(0);
            let xq_wf = crate::rawvk::gemv::xq_words(ni_f);
            let fbuf = self.f16w.get(wkey).cloned().unwrap();
            let gx = (no as u32).div_ceil(128);
            for tb in (0..t).step_by(128) {
                let nt = (t - tb).min(128) as u32;
                let last = tb + 128 >= t && bar;
                let push = Self::push_u32s(&[ni_f as u32, no as u32, xq_wf as u32, nt]);
                self.run_pipe_b("tile_f16", TILE_F16_SPV, 3, 16,
                    &[fbuf.buf, xq, out], &push, gx, 1, 1, last)?;
            }
            return Ok(());
        }
        // 타일(coopmat f16) 기본 경로 (2026-09-08 judge TILE 19/19 수용 —
        // llama 자체 pp가 동일 f16-닷 품질계약). 킬스위치 LLM170_VK_NOTILE=1.
        if t >= tile_min && std::env::var_os("LLM170_VK_NOTILE").is_none()
            && (ty == 11 || ty == 12 || ty == 13 || ty == 14 || ty == 20 || ty == 21 || ty == 23 || ty == 8) {
            return self.gemv_tile(xq, wkey, out, t, bar);
        }
        self.gemv_xq(xq, wkey, out, t, bar)
    }

    /// 타일(coopmat) 경로 — 프리필 전용. plans/32.
    #[allow(unreachable_code)] // 마지막 타일 경로가 무조건 return (2026-09-14 경고 정리)
    pub(super) fn gemv_tile(&mut self, xq: vk::Buffer, wkey: &str, out: vk::Buffer, t: usize, bar: bool) -> Result<(), String> {
        let (wbufs, ty, ni, no) = self.w.get(wkey).cloned().ok_or(format!("가중치 없음: {wkey}"))?;
        if std::env::var_os("LLM170_VK_SHAPES").is_some() {
            eprintln!("[shape] {wkey} ty={ty} ni={ni} no={no} gx={}", no.div_ceil(128));
        }
        // plans/30→32: tile128(coopmat f16)은 t=1 gemv와 수치계열이 다르나
        // 프리필 전용(t≥TILE_MIN)이면 spec 검증 배치(t≤5)와 무관 — 불변식 유지.
        // 실측 pp512 11.18→17.45 t/s (+56%). 옵트인 LLM170_VK_TILE=1.
        {
            let xq_w = crate::rawvk::gemv::xq_words(ni);
            let mut binds: Vec<vk::Buffer> = wbufs.iter().map(|b| b.buf).collect();
            while binds.len() < 8 {
                binds.push(self.dummy.buf);
            }
            binds.push(xq);
            binds.push(out);
            let gx = (no as u32).div_ceil(128);
            // tile_msALL (plans/40): 전 타입 ms 골격 — 기본 경로 승격(verify 22/3, pp64
            // 140→177). plans/79 B2: MSALL/MS_TYPES 바이섹트·MS128 모드(1/ffn/split)·
            // GY/GY2 옵트는 종결 실험 게이트로 폐기 — 기본 디스패치 확정.
            let bn128_on = t >= 128;
            let ms_spv: Option<(&str, &[u8], u32)> = match ty {
                13 => {
                    if bn128_on {
                        Some(("tile_ms128", TILE_MS128_SPV, 10))
                    } else {
                        Some(("tile_ms4gy", TILE_MS4GY_SPV, 10))
                    }
                }
                12 if bn128_on => Some(("tile_q4k128", TILE_Q4K128_SPV, 10)),
                14 if bn128_on => Some(("tile_q6k128", TILE_Q6K128_SPV, 10)),
                11 if bn128_on => Some(("tile_q3k128", TILE_Q3K128_SPV, 10)),
                8 if bn128_on => Some(("tile_q8128", TILE_Q8128_SPV, 10)),
                20 if bn128_on => Some(("tile_nl128", TILE_NL128_SPV, 11)),
                _ if ty != 21 && bn128_on => Some(("tile_xs128", TILE_XS128_SPV, 11)),
                12 => Some(("tile_q4kms", TILE_Q4KMS_SPV, 10)),
                14 => Some(("tile_q6kms", TILE_Q6KMS_SPV, 10)),
                11 => Some(("tile_q3kms", TILE_Q3KMS_SPV, 10)),
                8 => Some(("tile_q8ms", TILE_Q8MS_SPV, 10)),
                20 => Some(("tile_nlms", TILE_NLMS_SPV, 11)),
                _ if ty != 21 => Some(("tile_xsms", TILE_XSMS_SPV, 11)),
                _ => None,
            };
            if let Some((nm, spv, nkb)) = ms_spv {
                if nkb == 11 {
                    binds.push(self.ktab.buf);   // xs/nl LUT (구경로와 동일)
                }
                let step: usize = if bn128_on { 128 } else { 64 };
                let gx_ms = (no as u32).div_ceil(64);
                // plans/40 gy: q5_K t<128은 토큰 슬래브를 gy로 병렬 (tile_ms4gy).
                let use_gy = ty == 13 && !bn128_on;
                if use_gy {
                    // plans/40 gy: 토큰 슬래브를 gy로 병렬 — 단일 디스패치 L2 가중 재사용.
                    // plans/42: GYGRP=n이면 n토큰 그룹으로 분할 디스패치 (예: 128 → gy=2,
                    // 하네스 실측 병합 한계 내). 미설정 시 전 토큰 단일 디스패치.
                    let grp: usize = std::env::var("LLM170_VK_GYGRP").ok()
                        .and_then(|v| v.parse().ok()).filter(|&g| g >= 64).unwrap_or(t);
                    let nrows = (no as u32).div_ceil(64);
                    for g0 in (0..t).step_by(grp) {
                        let gt = (t - g0).min(grp);
                        let gy = (gt as u32).div_ceil(64);
                        let last = g0 + grp >= t && bar;
                        let push = Self::push_u32s(&[ni as u32, no as u32, xq_w as u32, 64u32, g0 as u32]);
                        self.run_pipe_b("tile_ms4gy", TILE_MS4GY_SPV, 10, 20, &binds, &push,
                            gy, nrows, 1, last)?;
                    }
                    return Ok(());
                }
                for tb in (0..t).step_by(step) {
                    let nt = (t - tb).min(step) as u32;
                    let last = tb + step >= t && bar;
                    // plans/41 슬래브 토큰 기저 — 커널이 tok_base..tok_base+nt를 처리
                    // ms128 계열은 row_off까지 6필드 (pb=24)
                    let ms128fam2 = nm.ends_with("128");
                    let (push, pb) = if ms128fam2 {
                        (Self::push_u32s(&[ni as u32, no as u32, xq_w as u32, nt, 0u32, tb as u32]), 24)
                    } else {
                        (Self::push_u32s(&[ni as u32, no as u32, xq_w as u32, nt, tb as u32]), 20)
                    };
                    self.run_pipe_b(nm, spv, nkb, pb, &binds, &push, gx_ms, 1, 1, last)?;
                }
                return Ok(());
            }
            // tile128o (점유 변형, plans/39): 64토큰/1-sb/LDS 29.7KB → 2 WG/CU
            if ty == 13 && std::env::var("LLM170_TILE_OCC").map(|v| v == "1").unwrap_or(false) {
                for tb in (0..t).step_by(64) {
                    let nt = (t - tb).min(64) as u32;
                    let last = tb + 64 >= t && bar;
                    let push = Self::push_u32s(&[ni as u32, no as u32, xq_w as u32, nt]);
                    self.run_pipe_b("tile128o", TILE128O_SPV, 10, 16, &binds, &push, gx, 1, 1, last)?;
                }
                return Ok(());
            }
            let step = if ty == 21 { 64 } else { 128 };   // 재생성 패밀리 128토큰, iq3s 구형 (plans/39)
            let n_tb = t.div_ceil(step);
            for (tbi, tb) in (0..t).step_by(step).enumerate() {
                let nt = (t - tb).min(step) as u32;
                let last = tbi + 1 == n_tb && bar;
                if ty == 13 {
                    let push = Self::push_u32s(&[ni as u32, no as u32, xq_w as u32, nt]);
                    self.run_pipe_b("tile128", TILE128_SPV, 10, 16, &binds, &push, gx, 1, 1, last)?;
                } else if ty == 12 {
                    // tile_q4k: ql@16 128B, qh 없음 — 시프트 청크 push
                    let cw = wbufs.first().map(|b| b.bytes.next_power_of_two() / 4).unwrap_or(1) as u32;
                    let cw_log2 = 31u32 - cw.leading_zeros();
                    let cw_mask = (1u32 << cw_log2) - 1u32;
                    let push = Self::push_u32s(&[ni as u32, no as u32, xq_w as u32, nt, cw_log2, cw_mask]);
                    self.run_pipe_b("tile_q4k", TILE_Q4K_SPV, 10, 24, &binds, &push, gx, 1, 1, last)?;
                } else if ty == 14 {
                    // tile_q6k: ql 니블 + qh 2비트 + i8 스케일 + d @208
                    let cw = wbufs.first().map(|b| b.bytes.next_power_of_two() / 4).unwrap_or(1) as u32;
                    let cw_log2 = 31u32 - cw.leading_zeros();
                    let cw_mask = (1u32 << cw_log2) - 1u32;
                    let push = Self::push_u32s(&[ni as u32, no as u32, xq_w as u32, nt, cw_log2, cw_mask]);
                    self.run_pipe_b("tile_q6k", TILE_Q6K_SPV, 10, 24, &binds, &push, gx, 1, 1, last)?;
                } else if ty == 11 {
                    // q3_K — 구 tile_q3k 디코드 결함(스케일 tmp 3바이트/하프 인덱스 — plans/40)
                    // → 검증된 tile_q3kms(ms 골격)로 영구 전환. maxrel 0.0033.
                    let gx_q3 = (no as u32).div_ceil(64);
                    let push = Self::push_u32s(&[ni as u32, no as u32, xq_w as u32, nt, tb as u32]);
                    self.run_pipe_b("tile_q3kms", TILE_Q3KMS_SPV, 10, 20, &binds, &push, gx_q3, 1, 1, last)?;
                } else if ty == 8 {
                    // tile_q8 (plans/32): q8_0 coopmat — 소형(beta/alpha)도 포함
                    // (gemv3 t≥16 소형은 0.2GB/s급 병목 — ts 프로파일 2026-09-08)
                    let cw = wbufs.first().map(|b| b.bytes.next_power_of_two() / 4).unwrap_or(1) as u32;
                    let cw_log2 = 31u32 - cw.leading_zeros();
                    let cw_mask = (1u32 << cw_log2) - 1u32;
                    let push = Self::push_u32s(&[ni as u32, no as u32, xq_w as u32, nt, cw_log2, cw_mask]);
                    self.run_pipe_b("tile_q8", TILE_Q8_SPV, 10, 24, &binds, &push, gx, 1, 1, last)?;
                } else if ty == 20 {
                    // tile_nl: iq4_nl 18B 블록 — ktab 니블 LUT (iq4_xs와 공유)
                    let cw = wbufs.first().map(|b| b.bytes.next_power_of_two() / 4).unwrap_or(1) as u32;
                    let cw_log2 = 31u32 - cw.leading_zeros();
                    let cw_mask = (1u32 << cw_log2) - 1u32;
                    binds.push(self.ktab.buf);
                    let push = Self::push_u32s(&[ni as u32, no as u32, xq_w as u32, nt, cw_log2, cw_mask]);
                    self.run_pipe_b("tile_nl", TILE_NL_SPV, 11, 24, &binds, &push, gx, 1, 1, last)?;
                    binds.pop();
                } else if ty == 21 {
                    // tile_iq3s: 110B 블록 + IQ3S_GRID 512워드 바인딩
                    let cw = wbufs.first().map(|b| b.bytes.next_power_of_two() / 4).unwrap_or(1) as u32;
                    let cw_log2 = 31u32 - cw.leading_zeros();
                    let cw_mask = (1u32 << cw_log2) - 1u32;
                    binds.push(self.grid3s.buf);
                    let push = Self::push_u32s(&[ni as u32, no as u32, xq_w as u32, nt, cw_log2, cw_mask]);
                    self.run_pipe_b("tile_iq3s", TILE_IQ3S_SPV, 11, 24, &binds, &push, gx, 1, 1, last)?;
                    binds.pop();
                } else {
                    // tile_xs (plans/32): iq4_xs coopmat — ktab 바인딩, 시프트 청크
                    let cw = wbufs.first().map(|b| b.bytes.next_power_of_two() / 4).unwrap_or(1) as u32;
                    let cw_log2 = 31u32 - cw.leading_zeros();
                    let cw_mask = (1u32 << cw_log2) - 1u32;
                    binds.push(self.ktab.buf);
                    let push = Self::push_u32s(&[ni as u32, no as u32, xq_w as u32, nt, cw_log2, cw_mask]);
                    self.run_pipe_b("tile_xs", TILE_XS_SPV, 11, 24, &binds, &push, gx, 1, 1, last)?;
                    binds.pop();
                }
            }
            return Ok(());
        }
        // 진입 불가: 위에서 지원 타입 전부가 return 됨(2026-09-14 경고 정리).
        unreachable!("gemv_tile: 타입 미적용")
    }

    /// 비타일 gemv (원본 경로).
    pub(super) fn gemv_xq(&mut self, xq: vk::Buffer, wkey: &str, out: vk::Buffer, t: usize, bar: bool) -> Result<(), String> {
        let (wbufs, ty, ni, no) = self.w.get(wkey).cloned().ok_or(format!("가중치 없음: {wkey}"))?;
        if self.ktime || self.ctx.ts.is_some() {
            *self.kkey.borrow_mut() = Some(format!("gemv:ty{ty}:{wkey}"));
        }
        let xq_w = crate::rawvk::gemv::xq_words(ni);
        let mut binds: Vec<vk::Buffer> = wbufs.iter().map(|b| b.buf).collect();
        while binds.len() < 8 {
            binds.push(self.dummy.buf);
        }
        binds.push(xq);
        binds.push(out);
        binds.push(self.ktab.buf);
        binds.push(self.grid3s.buf);
        // plans/29: 균일 청크 워드 수 전달 (마지막 청크만 부분 — WG 호산소 정합).
        // 단일 청크면 전체/4 → c 항상 0.
        let chunk_words = wbufs.first().map(|b| b.bytes / 4).unwrap_or(1) as u32;
        let push = Self::push_u32s(&[ni as u32, no as u32, xq_w as u32, ty, t as u32, chunk_words]);
        self.run_pipe_b("gemv", crate::rawvk::gemv::GEMV_SPV, 12, 24, &binds, &push, no as u32, 1, 1, bar)
    }

    /// v2 (mlx식) — per-row 스케일: quant_b8v2 + gemm_i8v2. LLM170_VK_I8=2.
    pub(super) fn gemm_i8v2(&mut self, wkey: &str, out: vk::Buffer, t: usize, bar: bool) -> Result<(), String> {
        let e = self.i8w.get(wkey).ok_or(format!("i8w 없음: {wkey}"))?;
        let (wbuf, ni, no) = (e.w.clone(), e.n_in, e.n_out);
        let wsr = self.wsr.get(wkey).cloned().ok_or("wsr 없음")?;
        // quant_b8v2: b_xn → b8 + ydr (ydb 첫 t 슬롯 재사용)
        let push = Self::push_u32s(&[ni as u32, t as u32]);
        self.run_pipe("quant_b8v2", QUANT_B8V2_SPV, 3, 8,
            &[self.b_xn.buf, self.b8.buf, self.ydb.buf], &push,
            1, t as u32, 1)?;
        let mut binds: Vec<vk::Buffer> = vec![wbuf.buf; 1];
        while binds.len() < 8 { binds.push(self.dummy.buf); }
        binds.push(self.b8.buf);
        binds.push(out);
        binds.push(wsr.buf);
        binds.push(self.ydb.buf);
        let push2 = Self::push_u32s(&[ni as u32, no as u32, t as u32]);
        self.run_pipe_b("gemm_i8v2", GEMM_I8V2_SPV, 12, 12, &binds, &push2,
            (no as u32).div_ceil(16), 1, 1, bar)
    }

    /// gemm_i8 (plans/23) — q5_K 사전 언패분으로 t행 GEMM.
    pub(super) fn gemm_i8(&mut self, wkey: &str, out: vk::Buffer, t: usize, bar: bool) -> Result<(), String> {
        let e = self.i8w.get(wkey).ok_or(format!("i8w 없음: {wkey}"))?;
        let (wbuf, wspbuf, wsmbuf, ni, no) = (e.w.clone(), e.wsp.clone(), e.wsm.clone(), e.n_in, e.n_out);
        let n_sub = ni / 32;
        let mut binds: Vec<vk::Buffer> = vec![wbuf.buf; 1];
        while binds.len() < 8 {
            binds.push(self.dummy.buf);
        }
        binds.push(self.b8.buf);
        binds.push(out);
        binds.push(wspbuf.buf);
        binds.push(wsmbuf.buf);
        binds.push(self.ydb.buf);
        binds.push(self.qsb.buf);
        binds.push(self.ishs.buf);
        binds.push(self.faccs.buf);
        let push = Self::push_u32s(&[ni as u32, no as u32, t as u32, n_sub as u32]);
        self.run_pipe_b("gemm_i8", GEMM_I8_SPV, 16, 16, &binds, &push,
            (no as u32).div_ceil(16), 1, 1, bar)
    }

    /// 단계 공유 GEMV 그룹 — 잡들은 상호 독립(동일 입력·상이 출력)이라
    /// 그룹 내부 배리어 생략, 마지막 잡이 배리어로 종결.
    /// xq 양자화는 실제 폴백 잡이 있을 때만 수행 (gemv8 직결 사이트의 dead
    /// quant 스킵 — plans/36 G2). i8 잡(t≥2·VK_I8ON)은 gemm_i8.
    pub(super) fn gemv_stage(&mut self, n: usize, t: usize, jobs: &[(String, vk::Buffer, vk::Buffer)]) -> Result<(), String> {
        let g8 = t < 16 && std::env::var("LLM170_G8").map(|v| v != "0").unwrap_or(true);
        let i8_on = t >= 2 && std::env::var_os("LLM170_VK_I8ON").is_some();
        let v2 = std::env::var("LLM170_VK_I8").map(|v| v == "2").unwrap_or(false);
        // 사전 판정 (self 대여 분리 — 클로저로 두면 mut 대여와 충돌)
        let elig: Vec<bool> = jobs
            .iter()
            .map(|(k, _, _)| matches!(self.w.get(k).map(|e| e.1), Some(8 | 11 | 12 | 13 | 14 | 20 | 21 | 23)))
            .collect();
        let i8s: Vec<bool> = jobs.iter().map(|(k, _, _)| i8_on && self.i8w.contains_key(k)).collect();
        // xq 필요 조건: gemv8/타일 외 폴백 잡이 하나라도 있을 때
        let need_xq = (0..jobs.len()).any(|ji| !i8s[ji] && (!g8 || !elig[ji]));
        if need_xq {
            self.quant(self.b_xn.buf, self.b_xq_n.buf, n, t)?;
        }
        let last = jobs.len() - 1;
        for (ji, (k, xq, out)) in jobs.iter().enumerate() {
            let bar = ji == last;
            if i8s[ji] {
                if v2 {
                    self.gemm_i8v2(k, *out, t, bar)?;
                } else {
                    self.gemm_i8(k, *out, t, bar)?;
                }
            } else if g8 && elig[ji] {
                // spec 검증 배치(t≤5)도 gemv8 수치계열로 — 불변식 회복 (plans/33)
                self.gemv8_q5(self.b_xn.buf, k, *out, t, bar)?;
            } else {
                self.gemv_bar(*xq, k, *out, t, bar)?;
            }
        }
        Ok(())
    }

    /// quant_b8: src f32 [t][n] → b8/yd/qs (gemm_i8 입력).
    pub(super) fn quant_b8(&mut self, src: vk::Buffer, n: usize, t: usize) -> Result<(), String> {
        let n_sub = n / 32;
        let push = Self::push_u32s(&[n as u32, t as u32, n_sub as u32]);
        self.run_pipe("quant_b8", QUANT_B8_SPV, 4, 12,
            &[src, self.b8.buf, self.ydb.buf, self.qsb.buf], &push,
            (n / 32 + 63) as u32 / 64, t as u32, 1)
    }

    /// rms_norm (t행) — 상수 가중치 (consts).
    pub(super) fn rms(&mut self, src: vk::Buffer, wkey: &str, out: vk::Buffer, n: usize, t: usize) -> Result<(), String> {
        let wbuf = self.consts.get(wkey).cloned().ok_or(format!("상수 없음: {wkey}"))?;
        let eps = self.eps;
        // plans/84 B: rms 셰이더에 w_reps 필드 추가(프레임 경로) — 디코더는 1로
        // 고정(산술 불변), push 16B로 맞춘다.
        let mut push = Self::push_u32s(&[n as u32, t as u32, 1u32]);
        push.extend_from_slice(&eps.to_le_bytes());
        self.run_pipe("rms", crate::rawvk::gemv::RMS_SPV, 3, 16,
            &[src, wbuf.buf, out], &push, t as u32, 1, 1)
    }

    /// 잔차 덧셈 + rms_norm 융합 (plans/36 G2) — y += x·1.0 결과를 그대로
    /// 세그먼트 축소하므로 axpy+rms 2디스패치와 비트동일, 런치만 절감.
    pub(super) fn addrms(&mut self, y: vk::Buffer, x: vk::Buffer, wkey: &str, out: vk::Buffer, n: usize, t: usize) -> Result<(), String> {
        let wbuf = self.consts.get(wkey).cloned().ok_or(format!("상수 없음: {wkey}"))?;
        let eps = self.eps;
        let mut push = Self::push_u32s(&[n as u32, t as u32]);
        push.extend_from_slice(&eps.to_le_bytes());
        self.run_pipe("addrms", ADDRMS_SPV, 4, 12,
            &[y, x, wbuf.buf, out], &push, t as u32, 1, 1)
    }


    /// axpy: y += x·s[0] (s=one 버퍼).
    pub(super) fn axpy(&mut self, y: vk::Buffer, x: vk::Buffer, n: usize) -> Result<(), String> {
        // one 버퍼 필요 — dummy는 0이므로 별도 1.0 버퍼 (init에서 만들었으면 재사용)
        let one = self.consts.get("one").ok_or("one 버퍼 없음")?;
        let push = (n as u32).to_le_bytes().to_vec();
        self.run_pipe("axpy", crate::rawvk::AXPY_SPV, 3, 4,
            &[y, x, one.buf], &push, n.div_ceil(256) as u32, 1, 1)
    }

    /// silu_mul g·u.
    pub(super) fn silu_mul(&mut self, g: vk::Buffer, u: vk::Buffer, o: vk::Buffer, total: usize) -> Result<(), String> {
        let push = (total as u32).to_le_bytes().to_vec();
        self.run_pipe("silu", crate::rawvk::gemv::SILU_SPV, 3, 4,
            &[g, u, o], &push, total.div_ceil(256) as u32, 1, 1)
    }

}
