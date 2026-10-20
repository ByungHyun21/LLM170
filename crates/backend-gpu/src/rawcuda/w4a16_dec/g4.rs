//! [gemma4 P2-3 2026-10-14] gemma4 v0 체인 조각 — 호스트 스테이징.
//!
//! qwen35 체인(`chains.rs`)과 달리 gemma4의 층 산술(노름·잔차·layer_scalar)은
//! **호스트 코어 계약**(`llm170_core::ops`)으로 조립한다 — v0 목적은 GPU 커널
//! (GEMV·어텐션·전문가·head)의 토큰 수준 검증(P2 게이트)이다. 디코더가
//! 제공하는 것은 내부 기계가 필요한 두 조각뿐:
//!
//! - `g4_moe_host`: 라우터 입력과 전문가 입력이 **다른** gemma4 MoE 계약
//!   (라우터 = 가중 RMS 행 = moe_scale·h^-0.5, 전문가 = pre_ff_norm2) + pes
//!   (per-expert μ_e) + shared 없음(gelu 전문가).
//! - `head_logits_host`: 최종 노름 hidden → GPU head(행별 int8 W8A16) 로짓.
//!
//! 계약 출처: `crates/core/src/gemma4/{mod.rs,stages/ffn.rs}`(CPU 오라클) —
//! 시퀀스: 라우터 GEMV → 디바이스 top-k(전문가 전체 softmax → top-k →
//! 재정규화) → pes → 전문가 gate/up(GLU_gelu)+down → 가중 누적.

use super::*;

impl W4a16Dec {
    /// gemma4 MoE 스텝 — `hn` = 라우터 입력(가중 RMS 행 적용 후), `xn` = 전문가
    /// 입력(pre_ff_norm2), `pes` = 층의 per-expert 스케일(μ_e). 반환 = 전문가
    /// 출력(hidden f32, 호스트). shared 없음(`set_moe(shared_ffn=0)` 전제).
    pub fn g4_moe_host(
        &mut self,
        il: usize,
        hn: &[f32],
        xn: &[f32],
        pes: &[f32],
    ) -> Result<Vec<f32>, String> {
        let h = self.hidden;
        let n_exp = self.moe.n_experts;
        if hn.len() != h || xn.len() != h {
            return Err(format!(
                "g4_moe: 입력 계약 위반 hn={} xn={} != hidden={h}",
                hn.len(),
                xn.len()
            ));
        }
        if n_exp == 0 || pes.len() != n_exp {
            return Err(format!(
                "g4_moe: pes={} != n_experts={n_exp} (set_moe 선행 확인)",
                pes.len()
            ));
        }
        let _g = self.cc.guard()?;
        self.ensure_moe_bufs()?;
        let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let dh = self.cc.alloc(h * 4)?;
        let dx = self.cc.alloc(h * 4)?;
        let dp = self.cc.alloc(n_exp * 4)?;
        let r = (|| -> Result<Vec<f32>, String> {
            self.cc.h2d(dh, &f32b(hn))?;
            self.cc.h2d(dx, &f32b(xn))?;
            self.cc.h2d(dp, &f32b(pes))?;
            // ① 라우터 — 업로드 방식(GEMV 상주/플레인)에 따라 심볼 분기.
            let rn = format!("blk.{il}.moe_gate.weight");
            if self.lins.contains_key(&rn) {
                self.gemv_launch(&rn, dh, self.moe.drt.ptr)?;
            } else {
                self.plain_gemv_launch(&rn, dh, self.moe.drt.ptr)?;
            }
            // ② top-k(디바이스) — 전체 softmax → k선택(내림차순·동률 낮은 idx)
            //    → 재정규화. moe_idx/moe_wt 기록(CPU route 미러).
            self.moe_topk_dev()?;
            // ③ pes — 재정규화 후 가중에 μ_e(인플레이스 — HF Gemma4TextRouter 순서).
            {
                let f = self.cc.function("w4a16_moe_pes_scale")?;
                let (mut p_w, mut p_i, mut p_p) = (self.moe.moe_wt, self.moe.moe_idx, dp);
                let mut p_n = self.moe.top_k as i32;
                self.cc.launch(
                    f,
                    (self.moe.top_k as u32).div_ceil(256).max(1),
                    1,
                    256,
                    &mut crate::rawcuda::args::l4(&mut p_w, &mut p_i, &mut p_p, &mut p_n),
                )?;
            }
            // ④ 전문가 — 상주: 전 슬롯 배치 + accum / 스트리밍: 자체 누적(axpy).
            if self.moe.moe_resident {
                self.moe_experts_batch(il, dx, self.moe.top_k)?;
                self.moe_accum_dev(
                    self.moe.moe_wt,
                    self.moe.exp.dn,
                    self.moe.dmo.ptr,
                    0,
                    self.moe.top_k,
                    h,
                )?;
            } else {
                let sel = self.moe_read_selection()?;
                Self::zero_dev(&self.cc, self.moe.dmo.ptr, h * 4)?;
                self.moe_experts_streaming(il, dx, &sel)?;
            }
            self.cc.sync()?;
            let mut ob = vec![0u8; h * 4];
            self.cc
                .d2h_async(ob.as_mut_ptr(), self.moe.dmo.ptr, h * 4)?;
            self.cc.sync()?;
            Ok(ob
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect())
        })();
        let _ = self.cc.free(dh);
        let _ = self.cc.free(dx);
        let _ = self.cc.free(dp);
        r
    }

    /// [gemma4 P3 2026-10-14] t행 GEMM 호스트 래퍼 — [t][k] f32 in → [t][n] out.
    /// `gemm_launch`(그룹 심볼 — g32 FFMA) + 사설 스크래치(1회 할당).
    pub fn g4_gemm_host(
        &mut self,
        name: &str,
        x: &[f32],
        t: usize,
        k: usize,
    ) -> Result<Vec<f32>, String> {
        let _g = self.cc.guard()?;
        let (_, _, n, kk) = self.lin_spec(name)?;
        if t == 0 || k != kk || x.len() != t * k {
            return Err(format!(
                "g4_gemm: {name} 계약 위반 t={t} k={k}(선형 {kk}) x={}",
                x.len()
            ));
        }
        if self.g4_a.cap < t * k {
            if self.g4_a.ptr != 0 {
                self.cc.free(self.g4_a.ptr)?;
            }
            self.g4_a.ptr = self.cc.alloc(t * k * 4)?;
            self.g4_a.cap = t * k;
        }
        if self.g4_y.cap < t * n {
            if self.g4_y.ptr != 0 {
                self.cc.free(self.g4_y.ptr)?;
            }
            self.g4_y.ptr = self.cc.alloc(t * n * 4)?;
            self.g4_y.cap = t * n;
        }
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
        let xb = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) };
        self.cc.h2d(self.g4_a.ptr, xb)?;
        self.gemm_launch(name, self.g4_a.ptr, self.g4_y.ptr, t)?;
        self.cc.sync()?;
        let mut ob = vec![0u8; t * n * 4];
        self.cc
            .d2h_async(ob.as_mut_ptr(), self.g4_y.ptr, t * n * 4)?;
        self.cc.sync()?;
        Ok(ob
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    /// [gemma4 P3 2026-10-14] 청크 디바이스 체인 — 스크래치 슬롯 확보.
    /// 재할당 시 캡처 그래프 전량 폐기(옛 포인터 replay 차단 — realloc_buf 동형).
    /// [⑮② 2026-10-18] gemma4 다중세그먼트 테이블 — 층별 [q,k(,v)]·[gate,up]
    /// 그룹을 dmulti/dmulti_meta에 (q,s,out,n,k) u64×5로 기입. out = G4_B1/
    /// B2/B3 슬롯(호스트 체인과 동일 소비 위치). 슬롯은 G4_CHUNK_MAX×최대 n
    /// 으로 선확보 — 이후 런타임 ensure가 재할당하지 않는다(그래프 정적 계약).
    /// gemma4는 ensure_chain_bufs(qwen 테이블)를 부르지 않으므로 dmulti의
    /// 단일 작성자다. 호출 = upload_g4 끝(층 등록 후 1회).
    pub fn build_multi_tables_g4(&mut self) -> Result<(), String> {
        let nl = self.n_layers;
        let dm = self.attn.attn.ok_or("g4 multi: attn 미등록")?;
        let tmax = G4_CHUNK_MAX;
        let qd_max = self
            .attn
            .layers
            .iter()
            .map(|l| dm.q_heads * l.d)
            .max()
            .unwrap_or(0);
        let mut nff_max = 0usize;
        for il in 0..nl {
            let (_, _, n, _) = self.lin_spec(&format!("blk.{il}.ffn_gate.weight"))?;
            nff_max = nff_max.max(n);
        }
        let wmax = qd_max.max(nff_max);
        let (b1, b2, b3) = (
            self.g4_scratch_slot(Self::G4_B1, tmax * wmax * 4)?,
            self.g4_scratch_slot(Self::G4_B2, tmax * wmax * 4)?,
            self.g4_scratch_slot(Self::G4_B3, tmax * wmax * 4)?,
        );
        let mut tab: Vec<u64> = vec![0; nl * 25]; // 최대 5세그먼트(q,k,v,gu)×5 u64/층
        let mut meta: Vec<[u32; 8]> = vec![[0u32; 8]; nl];
        for il in 0..nl {
            let keqv = self.attn.layers[il].k_eq_v;
            let mut segs0: Vec<(String, u64)> = vec![
                (format!("blk.{il}.attn_q.weight"), b1),
                (format!("blk.{il}.attn_k.weight"), b2),
            ];
            if !keqv {
                segs0.push((format!("blk.{il}.attn_v.weight"), b3));
            }
            let segs1: Vec<(String, u64)> = vec![
                (format!("blk.{il}.ffn_gate.weight"), b1),
                (format!("blk.{il}.ffn_up.weight"), b2),
            ];
            let mut wr = il * 25;
            let mut gm = [[0u32; 4]; 2];
            for (gi, segs) in [&segs0, &segs1].into_iter().enumerate() {
                let off = wr as u32;
                let (mut total, mut tgrid, mut cnt) = (0usize, 0usize, 0u32);
                for (name, out) in segs {
                    let (p0, p1, n, k) = self.lin_spec(name)?;
                    if n > wmax {
                        return Err(format!("g4 multi({name}): n={n} > 슬롯 {wmax}"));
                    }
                    tab[wr] = p0;
                    wr += 1;
                    tab[wr] = p1;
                    wr += 1;
                    tab[wr] = *out;
                    wr += 1;
                    tab[wr] = n as u64;
                    wr += 1;
                    tab[wr] = k as u64;
                    wr += 1;
                    total += n;
                    tgrid += n.div_ceil(GEMV_TR);
                    cnt += 1;
                }
                gm[gi] = [off, cnt, total as u32, tgrid as u32];
            }
            meta[il] = [
                gm[0][0], gm[0][1], gm[0][2], gm[1][0], gm[1][1], gm[1][2], gm[0][3], gm[1][3],
            ];
        }
        let bytes = tab.len() * 8;
        // SAFETY: 로컬 Vec<u64>의 유효 수명 내 바이트 뷰.
        let buf = unsafe { std::slice::from_raw_parts(tab.as_ptr() as *const u8, bytes) };
        if self.dmulti == 0 {
            self.dmulti = self.cc.alloc(bytes)?;
        }
        self.cc.h2d(self.dmulti, buf)?;
        self.cc.sync()?;
        self.dmulti_meta = meta;
        Ok(())
    }

    pub fn g4_scratch_slot(&mut self, idx: usize, bytes: usize) -> Result<CUdeviceptr, String> {
        if self.g4_pool.len() <= idx {
            self.g4_pool.resize(idx + 1, Buf { ptr: 0, cap: 0 });
        }
        let (ptr, cap) = {
            let b = &self.g4_pool[idx];
            (b.ptr, b.cap)
        };
        if cap >= bytes {
            return Ok(ptr);
        }
        self.cc.capture_guard("g4_slot"); // 캡처 중 재할당 = warm 누락(즉시 실패)
        self.graph_invalidate(); // [P10] 재할당 — 캡처 옛 포인터 차단.
        self.cc.sync()?; // [P10] 비행 커널의 해제 버퍼 사용 차단.
        let newp = self.cc.alloc(bytes)?;
        if ptr != 0 {
            self.cc.free(ptr)?;
        }
        self.g4_pool[idx] = Buf {
            ptr: newp,
            cap: bytes,
        };
        Ok(newp)
    }

    /// t행 GEMM → 지정 슬롯(디바이스 in/out — 중간 왕복 0). 반환 = 슬롯 포인터.
    pub fn g4_dev_gemm(
        &mut self,
        name: &str,
        x_dev: CUdeviceptr,
        t: usize,
        k: usize,
        y_slot: usize,
    ) -> Result<CUdeviceptr, String> {
        let _g = self.cc.guard()?;
        let (_, _, n, kk) = self.lin_spec(name)?;
        if t == 0 || k != kk {
            return Err(format!("g4_dev_gemm: {name} k={k} != {kk}"));
        }
        let y = self.g4_scratch_slot(y_slot, t * n * 4)?;
        // [gemma4 P3 2026-10-14] t행 GEMV(그룹 디스패치 — g32/bf16 포함):
        // t≤8에서 가중 판독 1회 공유 + t=1 GEMV와 동일 클래스(FFMA gemm 대체 —
        // ncu 실측: FFMA는 DRAM 4.4%·SM 49.4% 지연 정체).
        // [2.N 2026-10-15] t≥16 mma 대체 후보는 실측 기각(pp −6.7/−8.9% —
        // `kernels/gemm.rs` 기록) — t행 GEMV 유지.
        // [A2 2026-10-15] 청크 t>G4_TROW_MAX = **행 분할 호출**(t행 커널은 ≤16
        // 스위치 계약) — 행별 산술 독립이라 단일 호출과 비트동일.
        // [2.N 2026-10-15 dense mma 파일럿 — 정합 OK·채택 보류(기록)] t≥64를
        // 포장 A hi/lo 2-pass mma(grp2 이식)로 바꾼 파일럿: 커널은 **원소 게이트
        // 클래스 수준**(maxdiff 6.7e-3)으로 정합했고 성능도 pp256 +15%였으나,
        // **품질이 5코퍼스에서 혼합·대형**이었다 — en −21.7%·ko −4.4%·long −4.1%
        // vs **code +3.0%·json +7.7%**(방향 = 궤적 운). 기제 = 근접동률 MoE
        // top-8 라우팅 플립 캐스케이드(1e-3급 섭동 → 161/1075(15%) argmax 플립,
        // 평균 NLL ±0.2 nat급 이동) — 즉 **dense 경로의 클래스 변경 = 라우터를
        // 직접 흔들어 모델을 사실상 바꾼다**(전문가 mma의 ±1%와 질적으로 다름).
        // 교훈(오독 기록): PPL%는 NLL의 지수라 −21.7% = NLL −0.244 nats(소량) —
        // **PPL% 단독으로 버그/개선을 판정하지 말 것**: ① mma 커널은 원소 게이트
        // (mma vs t행 GEMV 분할 대조) 먼저, ② 클래스 효과는 NLL nats/토큰 +
        // agree 플립 수 + **다코퍼스**로 읽는다. 재도전 조건 = 비트계약 FFMA
        // 타일 GEMM(클래스 보존 — 품질 불변) 또는 라우팅 강건성 근거.
        // [2026-10-16 재판정 — 재기각(기록)] 파킹 골격(포장 A hi|lo 2-pass)을
        // t≥64에 재배선해 현행 하네스·청크512로 재실행: 원소 게이트
        // (`w4a16-gemm --mma`) maxdiff 2.6e-6~2.3e-5(파일럿 6.7e-3보다 정밀)에도
        // 품질은 파일럿과 **동일 계열** — en NLL −0.2585 nats/토큰(ppl −22.78%)·
        // ko +0.0053·code +0.0104·json +0.0433 · agree 82.4/93.5/90.3/91.5%
        // (실질 플립 3/0/0/0) · 라우팅 Jaccard 0.922(en)·0.950(ko)·플립 28.2/19.7%.
        // 성능(A hi|lo 결합, 같은-세션 ab-bench): pp256/1024/4096
        // +76.6/+88.4/+72.6% — **원소 1e-5급 섭동도 라우터를 통해 모델을 바꾼다**
        // (A 정밀도로 회피 불가). 기각 유지·코드 삭제(단일 경로). 재개 조건
        // 불변 = 비트계약 FFMA 타일 GEMM.
        self.gemv_t_rows(name, x_dev, y, t)?;
        Ok(y)
    }

    /// [A2→P8 2026-10-15] t행 GEMV — t>16은 gemv_t_launch 내부의 tall 단일
    /// 런치(+tail 스위치)가 처리(가중 DRAM 스트림 1회). 종전의 16행 선분할
    /// 루프는 t≤16 폴백만 — 행별 산술 비트동일(레인·j-시퀀스·tree64 행 독립).
    fn gemv_t_rows(
        &mut self,
        name: &str,
        x_dev: CUdeviceptr,
        y_dev: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        self.gemv_t_launch(name, x_dev, y_dev, t)
    }

    /// [A2 2026-10-15] 플레인(bf16) t행 GEMV 행 분할 — 라우터 전용(행별 비트동일).
    fn plain_gemv_t_rows(
        &mut self,
        name: &str,
        x_dev: CUdeviceptr,
        y_dev: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        let (_, n, k) = self.plain_spec(name)?;
        let mut r0 = 0usize;
        while r0 < t {
            let rows = G4_TROW_MAX.min(t - r0);
            self.plain_gemv_t_launch(
                name,
                x_dev + (r0 * k * 4) as u64,
                y_dev + (r0 * n * 4) as u64,
                rows,
            )?;
            r0 += rows;
        }
        Ok(())
    }

    /// [gemma4 P3 2026-10-14] ew_gelu 디바이스 — act = gelu(g)·u → y(슬롯).
    /// t=1 경로(`ew_gelu_host`)와 동일 커널 심볼(수치 정합).
    pub fn g4_dev_gelu(
        &mut self,
        g_dev: CUdeviceptr,
        u_dev: CUdeviceptr,
        y_slot: usize,
        n: usize,
    ) -> Result<CUdeviceptr, String> {
        if n == 0 {
            return Err("g4_dev_gelu: n=0".into());
        }
        let _g = self.cc.guard()?;
        let y = self.g4_scratch_slot(y_slot, n * 4)?;
        let f = self.cc.function("ew_gelu")?;
        let mut nn = n as i32;
        let (mut a0, mut a1, mut a2, mut a3, mut a4) = (g_dev, u_dev, y, 0u64, 0u64);
        self.cc.launch(
            f,
            n.div_ceil(128) as u32,
            1,
            128,
            &mut crate::rawcuda::args::l6(&mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut nn),
        )?;
        Ok(y)
    }

    /// 슬롯 포인터(할당 보장 없음 — g4_scratch_slot 선행 계약).
    pub fn g4_dev_ptr(&self, slot: usize) -> CUdeviceptr {
        self.g4_pool.get(slot).map(|b| b.ptr).unwrap_or(0)
    }

    /// [gemma4 P3 2026-10-14] g4 어텐션 **디바이스·행별 테이블** 런처 — q/k/v·
    /// 출력 모두 디바이스(중간 d2h 0). 행 i = 슬롯 slots[i]·위치 pos[i](연속일
    /// 필요 없음 — 배치 디코드 지원). KV는 각 슬롯 아레나 기록. 반환 = outv 슬롯.
    /// qnw/knw = set_attn d_max 패딩 배열의 이 층 행(커널 행 포인터 계약 정합).
    #[allow(clippy::too_many_arguments)]
    pub fn attn_g4_dev(
        &mut self,
        slots: &[usize],
        pos: &[u32],
        layer: usize,
        q_dev: CUdeviceptr,
        k_dev: CUdeviceptr,
        v_dev: CUdeviceptr,
        q_heads: usize,
        kv_heads: usize,
        d: usize,
        sw: usize,
        kq_scale: f32,
        splits: usize,
        qh_slot: usize,
        part_slot: usize,
        out_slot: usize,
    ) -> Result<CUdeviceptr, String> {
        let t = slots.len();
        if t == 0 || pos.len() != t {
            return Err(format!("attn_g4_dev: t={t} pos={}", pos.len()));
        }
        let _g = self.cc.guard()?;
        let dm = self.attn.attn.ok_or("attn: 형상 미등록")?;
        if self.attn.g4_rp == 0 || layer >= dm.n_attn {
            return Err(format!("attn_g4_dev: layer={layer} n_attn={}", dm.n_attn));
        }
        let qn = t * q_heads * d;
        let qh = self.g4_scratch_slot(qh_slot, qn * 4)?;
        let part = self.g4_scratch_slot(part_slot, t * q_heads * splits * (d + 2) * 4)?;
        let outv = self.g4_scratch_slot(out_slot, qn * 4)?;
        // 행별 (pos, 블록 테이블) 포인터 테이블 — 커널 row_blk/row_pp 계약.
        // **층 무관**(커널이 층 오프셋 가산) — (slots,pos) 동일하면 업로드 생략.
        // [주의] 테이블 슬롯(10/11/12)은 청크 데이터 슬롯과 **분리 유지** —
        // 재사용하면 청크 중간값이 테이블을 덮어 불법 접근(전면 상주 실험 실측).
        // [gemma4 P3 v2 2026-10-14] 업로드 자체는 g4_atab_prepare로 분리
        // (청크 프롤로그 1회 · h2d_async — pageable 동기 h2d의 파이프라인
        // 드레인 회피). 그래프 캡처에서도 테이블 갱신은 캡처 밖 몫이다.
        self.g4_atab_prepare(slots, pos)?;
        let pos_dev = self.g4_scratch_slot(Self::G4_ATAB_POS, t * 4)?;
        let rpp_dev = self.g4_scratch_slot(Self::G4_ATAB_RPP, t * 8)?;
        let rblk_dev = self.g4_scratch_slot(Self::G4_ATAB_RBLK, t * 8)?;
        let dpp = pos_dev; // null 경로 비사용(호환 인자)
        let d_max = self.attn.layers.iter().map(|l| l.d).max().unwrap_or(d);
        let qnw = self.attn.dqnw_a + (layer * d_max * 4) as u64;
        let knw = self.attn.dknw_a + (layer * d_max * 4) as u64;
        let dblk = self.attn.dblk_tab; // 비사용(행 테이블 우선) — 호환 인자
        let drp = self.attn.g4_rp;
        // ① prep_g4 — grid (t, q_heads+kv_heads) · 128스레드.
        {
            let f = self.cc.function("attn_prep_g4")?;
            let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7, mut a8) =
                (q_dev, k_dev, v_dev, qnw, knw, qh, dblk, dpp, drp);
            let (mut rblk, mut rpp) = (rblk_dev, rpp_dev); // 행별 테이블
            let (mut tl, mut lay, mut qh_, mut kvh, mut cp) = (
                t as i32,
                layer as i32,
                q_heads as i32,
                kv_heads as i32,
                dm.cap as i32,
            );
            self.cc.launch(
                f,
                t as u32,
                (q_heads + kv_heads) as u32,
                128,
                &mut crate::rawcuda::args::l16(
                    &mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut a5, &mut a6, &mut a7,
                    &mut rblk, &mut rpp, &mut a8, &mut tl, &mut lay, &mut qh_, &mut kvh, &mut cp,
                ),
            )?;
        }
        // ② [FA2 2026-10-19] t>8 연속 단일 슬롯 프리필 — part_g4+merge 대신
        // FA2 타일 경로(qwen attn_fa2_launch 미러 — kq_scale=1.0·no_gate=1·
        // sw 창 인자는 2026-10-14부터 커널 준비됨). d=256(슬라이딩 25층) =
        // attn_fa2_qg(BR=BC=32) · d=512(풀층 5개) = attn_fa2_g4_512(BR=BC=16 —
        // KD 512 정적 smem 44.8KB 한도 내 타일 축소, PV 4워프 d-쿼터).
        // 혼합 슬롯·t≤8(디코드·스펙 검증)은 기존 part_g4+merge 단일 경로.
        let contig =
            slots.windows(2).all(|w| w[0] == w[1]) && pos.windows(2).all(|w| w[1] == w[0] + 1);
        if t > 8 && (d == 256 || d == 512) && contig {
            let f = self.cc.function(if d == 512 {
                "attn_fa2_g4_512"
            } else {
                "attn_fa2_qg"
            })?;
            let rows = if d == 512 { 16 } else { 32 };
            let (mut a0, mut a1, mut a2, mut a3) = (qh, self.attn_blk_ptr(slots[0]), 0u64, outv);
            let (mut a4, mut a5, mut a6) = (0u64, 0u64, pos_dev); // qg·out32·out16 미사용
            let (mut tl, mut lay, mut qh_, mut kvh, mut cp) = (
                t as i32,
                layer as i32,
                q_heads as i32,
                kv_heads as i32,
                dm.cap as i32,
            );
            let (mut swv, mut ngate) = (sw as i32, 1i32);
            let mut scl = kq_scale;
            self.cc.launch(
                f,
                t.div_ceil(rows) as u32,
                q_heads as u32,
                128,
                &mut crate::rawcuda::args::l15(
                    &mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut a5, &mut a6, &mut tl,
                    &mut lay, &mut qh_, &mut kvh, &mut cp, &mut scl, &mut swv, &mut ngate,
                ),
            )?;
            return Ok(outv);
        }
        // ② f3s part_g4 — grid (t, q_heads×splits) · d스레드.
        {
            let f = self.cc.function("attn_fwd3s_part_g4")?;
            let (mut a0, mut a1, mut a2, mut a3) = (qh, dblk, part, dpp);
            let (mut rblk, mut rpp) = (rblk_dev, rpp_dev); // 행별 테이블
            let (mut tl, mut lay, mut qh_, mut kvh, mut cp) = (
                t as i32,
                layer as i32,
                q_heads as i32,
                kv_heads as i32,
                dm.cap as i32,
            );
            let (mut sp, mut dd, mut swv) = (splits as i32, d as i32, sw as i32);
            let mut scl = kq_scale;
            self.cc.launch(
                f,
                t as u32,
                (q_heads * splits) as u32,
                d as u32,
                &mut crate::rawcuda::args::l15(
                    &mut a0, &mut a1, &mut a2, &mut a3, &mut rblk, &mut rpp, &mut tl, &mut lay,
                    &mut qh_, &mut kvh, &mut cp, &mut sp, &mut dd, &mut swv, &mut scl,
                ),
            )?;
        }
        // ③ merge_g4 — grid (t, q_heads) · d스레드.
        {
            let f = self.cc.function("attn_fwd3s_merge_g4")?;
            let (mut a0, mut a1, mut a2, mut a3) = (part, outv, 0u64, 0u64);
            let (mut tl, mut qh_, mut sp, mut dd) =
                (t as i32, q_heads as i32, splits as i32, d as i32);
            self.cc.launch(
                f,
                t as u32,
                q_heads as u32,
                d as u32,
                &mut crate::rawcuda::args::l8(
                    &mut a0, &mut a1, &mut a2, &mut a3, &mut tl, &mut qh_, &mut sp, &mut dd,
                ),
            )?;
        }
        Ok(outv)
    }

    /// [gemma4 P3 2026-10-14] 층별 pes(μ_e) 디바이스 상주 — 1회 업로드 캐시.
    pub fn set_g4_pes(&mut self, il: usize, pes: &[f32]) -> Result<(), String> {
        let _g = self.cc.guard()?;
        if self.g4_pes.len() <= il {
            self.g4_pes.resize(il + 1, 0);
        }
        if self.g4_pes[il] == 0 {
            let bytes: Vec<u8> = pes.iter().flat_map(|x| x.to_le_bytes()).collect();
            let p = self.cc.alloc(bytes.len())?;
            self.cc.h2d(p, &bytes)?;
            self.g4_pes[il] = p;
        }
        Ok(())
    }

    /// 층 pes 디바이스 포인터(0 = 미등록).
    pub fn g4_pes_dev(&self, il: usize) -> CUdeviceptr {
        self.g4_pes.get(il).copied().unwrap_or(0)
    }

    /// [gemma4 P3 v2 2026-10-14] 층별 layer_scalar 등록(디바이스 청크 전용 —
    /// 호스트 스칼라 인자, 디바이스 버퍼 아님).
    pub fn set_g4_layer_scalar(&mut self, ls: &[f32]) -> Result<(), String> {
        if ls.len() != self.n_layers {
            return Err(format!(
                "g4 layer_scalar: {} != n_layer={}",
                ls.len(),
                self.n_layers
            ));
        }
        self.g4_ls = ls.to_vec();
        Ok(())
    }

    /// [gemma4 P3 v2 2026-10-14] 어텐션 행별 (pos, 블록 테이블) 포인터 테이블
    /// (슬롯 10/11/12) 준비 — (slots,pos) 동일 시 무동작. 층 무관(커널이 층
    /// 오프셋 가산)이라 **청크당 1회**. h2d_async(같은 스트림 순서) —
    /// pageable 동기 복사는 디바이스 전체를 드레인한다(전면 상주 실험 실측).
    /// 그래프 캡처에서도 갱신은 캡처 밖(프롤로그) 몫 — 본문은 값만 읽는다.
    pub fn g4_atab_prepare(&mut self, slots: &[usize], pos: &[u32]) -> Result<(), String> {
        let t = slots.len();
        if t == 0 || pos.len() != t || t > G4_CHUNK_MAX {
            return Err(format!("g4_atab: t={t} pos={} (1..=16)", pos.len()));
        }
        if self.g4_atab.0 == slots && self.g4_atab.1 == pos {
            return Ok(());
        }
        let dm = self.attn.attn.ok_or("attn: 형상 미등록")?;
        if slots.iter().any(|&sl| sl >= self.n_slots) {
            return Err("g4_atab: 슬롯 범위 밖".into());
        }
        let mb = dm.cap.div_ceil(512);
        let pos_dev = self.g4_scratch_slot(Self::G4_ATAB_POS, t * 4)?;
        let rpp_dev = self.g4_scratch_slot(Self::G4_ATAB_RPP, t * 8)?;
        let rblk_dev = self.g4_scratch_slot(Self::G4_ATAB_RBLK, t * 8)?;
        // 캡처 중 업로드 = 프롤로그 누락(캡처엔 pageable 복사 노드가 못 들어간다).
        self.cc.capture_guard("g4_atab");
        let mut pb: Vec<u8> = Vec::with_capacity(t * 4);
        for &pv in pos {
            pb.extend_from_slice(&pv.to_le_bytes());
        }
        self.cc.h2d_async(pos_dev, &pb)?;
        let mut rppb: Vec<u8> = Vec::with_capacity(t * 8);
        let mut rblkb: Vec<u8> = Vec::with_capacity(t * 8);
        for (i, &sl) in slots.iter().enumerate() {
            rppb.extend_from_slice(&(pos_dev + (i * 4) as u64).to_le_bytes());
            let base = self.attn.dblk_tab + (sl * dm.n_attn * mb * 4 * 8) as u64;
            rblkb.extend_from_slice(&base.to_le_bytes());
        }
        self.cc.h2d_async(rpp_dev, &rppb)?;
        self.cc.h2d_async(rblk_dev, &rblkb)?;
        self.g4_atab = (slots.to_vec(), pos.to_vec());
        Ok(())
    }

    /// [gemma4 P3 v2 2026-10-14] 청크 슬롯 지도 — g4_scratch_slot 인덱스.
    /// 데이터 슬롯과 어텐션 테이블(10/11/12)은 분리(위 g4_atab_prepare 주석).
    const G4_X: usize = 0;
    const G4_A: usize = 1;
    const G4_B1: usize = 2;
    const G4_B2: usize = 3;
    const G4_B3: usize = 4;
    const G4_ATAB_POS: usize = 10;
    const G4_ATAB_RPP: usize = 11;
    const G4_ATAB_RBLK: usize = 12;
    const G4_HN: usize = 13;
    const G4_NM: usize = 14;
    const G4_MOE: usize = 15;
    const G4_H2: usize = 16;
    const G4_HF: usize = 17;
    const G4_HEADX: usize = 18;
    const G4_HEADOUT: usize = 19;

    /// 슬롯 전량을 t_max 행 기준으로 선할당 — 청크 중 재할당(캡처 무효화·
    /// 도중 alloc 금지)을 막는다. 크기 = 층 최대 형상(어텐션 qd/분할 포함).
    pub fn g4_warm_slots(&mut self, t_max: usize) -> Result<(), String> {
        let h = self.hidden;
        let dm = self.attn.attn.ok_or("g4_warm: attn 미등록")?;
        let mut qd_max = 0usize;
        let mut part_max = 0usize;
        for l in &self.attn.layers {
            qd_max = qd_max.max(dm.q_heads * l.d);
            part_max = part_max.max(dm.q_heads * 2 * (l.d + 2)); // splits=2
        }
        let th = t_max * h * 4;
        for s in [
            Self::G4_X,
            Self::G4_A,
            Self::G4_HEADX,
            Self::G4_HN,
            Self::G4_NM,
            Self::G4_MOE,
            Self::G4_H2,
            Self::G4_HF,
        ] {
            self.g4_scratch_slot(s, th)?;
        }
        // 어텐션 중간(o·hh·h1 = h 폭)은 A/B 슬롯과 같은 폭을 쓴다.
        for s in [Self::G4_B1, Self::G4_B2, Self::G4_B3] {
            self.g4_scratch_slot(s, t_max * qd_max.max(h) * 4)?;
        }
        self.g4_scratch_slot(5, t_max * qd_max * 4)?; // qh
        self.g4_scratch_slot(6, t_max * part_max * 4)?; // part
        self.g4_scratch_slot(7, t_max * qd_max * 4)?; // outv
        self.g4_scratch_slot(8, th)?; // E(o_proj / ffn_down — h 폭)
        self.g4_scratch_slot(9, th)?; // F(hh→h1→merged — h 폭)
        self.g4_scratch_slot(Self::G4_HEADOUT, t_max * self.head_n * 4)?;
        // [A2b 2026-10-15] 그룹 mma A packed(hi|lo) 미러 상한 — cast_hilo가
        // 캡처 중 재할당하면 capture_guard로 즉시 실패(g4_capture_failed 고정).
        if self.moe.moe_ffn > 0 && self.moe.n_experts > 0 {
            let ns_max = t_max * self.moe.top_k.max(1);
            self.ensure_dxp(ns_max * self.moe.moe_ffn.max(h))?;
        }
        Ok(())
    }

    /// [gemma4 P3 v2 2026-10-14] 청크 본문 — 층 루프 전량 디바이스(중간 왕복 0).
    /// x = 슬롯 0(G4_X)의 잔차 스트림. head_last = 프리필(마지막 행 로짓만).
    /// 캡처 대상 몫: 호스트 왕복·할당·동기 없음(전부 같은 스트림 발사).
    fn g4_chunk_body(
        &mut self,
        slots: &[usize],
        pos: &[u32],
        t: usize,
        head_last: bool,
        nw_per_layer: usize,
        eps: f32,
    ) -> Result<(), String> {
        let h = self.hidden;
        let n = t * h;
        let dm = self.attn.attn.ok_or("g4_chunk_dev: attn 미등록")?;
        let x = self.g4_scratch_slot(Self::G4_X, n * 4)?;
        for il in 0..self.n_layers {
            let l = self.attn.layers[il];
            let d = l.d;
            let qd = dm.q_heads * d;
            let nb = il * nw_per_layer;
            let sw = if l.sliding { dm.sliding_window } else { 0 };
            // ① input_layernorm(디바이스) → q/k/(v) GEMV.
            let a = self.g4_scratch_slot(Self::G4_A, n * 4)?;
            self.rms_rows_dev(nb, x, a, t, eps)?;
            // [⑮② 2026-10-18] q/k(/v) 다중세그먼트 1런치(테이블 = build_
            // multi_tables_g4). t>16은 tall 폴백(개별 호출 — 종전).
            let (q, k, v) = if t <= 16 {
                self.gemv_multi_g32_t_dev(il, 0, a, t)?;
                let q = self.g4_scratch_slot(Self::G4_B1, t * qd * 4)?;
                let k = self.g4_scratch_slot(Self::G4_B2, t * qd * 4)?;
                let v = if l.k_eq_v {
                    k
                } else {
                    self.g4_scratch_slot(Self::G4_B3, t * qd * 4)?
                };
                (q, k, v)
            } else {
                let q =
                    self.g4_dev_gemm(&format!("blk.{il}.attn_q.weight"), a, t, h, Self::G4_B1)?;
                let k =
                    self.g4_dev_gemm(&format!("blk.{il}.attn_k.weight"), a, t, h, Self::G4_B2)?;
                let v = if l.k_eq_v {
                    k // v = k 원시(prep_g4 vn_on 계약 — 호스트 경로와 동일)
                } else {
                    self.g4_dev_gemm(&format!("blk.{il}.attn_v.weight"), a, t, h, Self::G4_B3)?
                };
                (q, k, v)
            };
            // ② 어텐션(디바이스 — 행별 pos/테이블) → o_proj → post_attn 잔차.
            // [E1-mirror 2026-10-20] t≤8(디코드·스펙 검증)은 분할 = ATTN_SPLITS_D(64)
            // — qwen 디코드 선례(+10% tg) 미러. ncu 실측: splits=2에서 점유 12.5%·
            // SM/메모리 SOL 5~7%(순수 레이턴시 바운드 — 32블록×8워프만 활성).
            // 활성 분할은 커널이 grain(32)로 상한 자판(초과분 = 빈 partial 기록).
            // 분할 경계 변화 = ulp급 eval 클래스(E1 노트 참조) — span≤256은
            // eff=1로 불변(단문 골든·지문 비트동일 유지).
            let sp_dec = if t <= 8 { ATTN_SPLITS_D } else { 2 };
            let outv = self.attn_g4_dev(
                slots, pos, il, q, k, v, dm.q_heads, l.kv_heads, d, sw, l.kq_scale, sp_dec, 5, 6, 7,
            )?;
            let o = self.g4_dev_gemm(&format!("blk.{il}.attn_output.weight"), outv, t, qd, 8)?;
            let hh = self.g4_scratch_slot(9, n * 4)?;
            self.rms_rows_dev(nb + 1, o, hh, t, eps)?;
            self.axpy_dev(1.0, hh, x, n)?;
            // ③ dense MLP — pre_ff 노름 → gate/up → gelu → down → post_ff1.
            self.rms_rows_dev(nb + 2, x, a, t, eps)?;
            let (_, _, _, n_ff) = self.lin_spec(&format!("blk.{il}.ffn_down.weight"))?;
            // [⑮②] gate/up 다중세그먼트 1런치(t>16은 tall 폴백).
            let (g, u) = if t <= 16 {
                self.gemv_multi_g32_t_dev(il, 1, a, t)?;
                (
                    self.g4_scratch_slot(Self::G4_B1, t * n_ff * 4)?,
                    self.g4_scratch_slot(Self::G4_B2, t * n_ff * 4)?,
                )
            } else {
                (
                    self.g4_dev_gemm(&format!("blk.{il}.ffn_gate.weight"), a, t, h, Self::G4_B1)?,
                    self.g4_dev_gemm(&format!("blk.{il}.ffn_up.weight"), a, t, h, Self::G4_B2)?,
                )
            };
            let act = self.g4_dev_gelu(g, u, Self::G4_B3, t * n_ff)?;
            let m = self.g4_dev_gemm(&format!("blk.{il}.ffn_down.weight"), act, t, n_ff, 8)?;
            let h1 = self.g4_scratch_slot(9, n * 4)?; // hh 소비 후 재사용
            self.rms_rows_dev(nb + 3, m, h1, t, eps)?;
            // ④ MoE — 라우터(hn)·전문가(nm) 입력 분리 노름 → 디바이스 MoE.
            let hn = self.g4_scratch_slot(Self::G4_HN, n * 4)?;
            let nm = self.g4_scratch_slot(Self::G4_NM, n * 4)?;
            self.rms_rows_dev(nb + 7, x, hn, t, eps)?;
            self.rms_rows_dev(nb + 4, x, nm, t, eps)?;
            let pes = self.g4_pes_dev(il);
            if pes == 0 {
                return Err(format!("g4_chunk_dev: L{il} pes 미등록(set_g4_pes 선행)"));
            }
            let moe = self.g4_moe_dev_t(il, hn, nm, pes, t, Self::G4_MOE)?;
            let h2 = self.g4_scratch_slot(Self::G4_H2, n * 4)?;
            self.rms_rows_dev(nb + 5, moe, h2, t, eps)?;
            // ⑤ 병합(h1+h2) → post_ff 노름 → 잔차 → layer_scalar.
            self.axpy_dev(1.0, h2, h1, n)?;
            let hf = self.g4_scratch_slot(Self::G4_HF, n * 4)?;
            self.rms_rows_dev(nb + 6, h1, hf, t, eps)?;
            self.axpy_dev(1.0, hf, x, n)?;
            let ls = *self
                .g4_ls
                .get(il)
                .ok_or("g4_chunk_dev: layer_scalar 미등록(set_g4_layer_scalar)")?;
            self.scale_dev(x, ls, n)?;
            // [디버그 배선 2026-10-15] 층별 지문 — CPU 오라클(`gemma4 layer`)·
            // 호스트 체인(`gemma4[gpu]`)과 같은 형식으로 최초 발산 층을 추적한다.
            // 행 0 = 청크 첫 토큰(CPU xs[0] 기준과 동일 계보). debug_layers가
            // 켜지면 g4_graph_ok가 그래프를 끄므로 직접 경로에서만 실행된다.
            if self.debug_layers {
                let mut rb = vec![0u8; h * 4];
                self.cc.d2h_async(rb.as_mut_ptr(), x, h * 4)?;
                self.cc.sync()?;
                let row: Vec<f32> = rb
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect();
                let mx = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let nan = row.iter().any(|v| v.is_nan());
                eprintln!("gemma4[gpu-chunk] layer {il:>2} max|x|={mx:.4} nan={nan}");
            }
        }
        // ⑥ 최종 노름 + head(TR t행) — 프리필은 마지막 행만(호스트 경로 계약).
        let hx = self.g4_scratch_slot(Self::G4_HEADX, n * 4)?;
        let out_row = self.n_layers * nw_per_layer;
        if head_last {
            let xr = x + ((t - 1) * h * 4) as u64;
            self.rms_rows_dev(out_row, xr, hx, 1, eps)?;
            self.g4_head_dev_t(hx, 1, Self::G4_HEADOUT)?;
        } else {
            self.rms_rows_dev(out_row, x, hx, t, eps)?;
            self.g4_head_dev_t(hx, t, Self::G4_HEADOUT)?;
        }
        Ok(())
    }

    /// x = x·s (제자리 — gemma4 layer_scalar, `w4a16_scale`).
    fn scale_dev(&mut self, y: CUdeviceptr, s: f32, n: usize) -> Result<(), String> {
        let f = self.cc.function("w4a16_scale")?;
        let mut ss = s;
        let mut p_y = y;
        let mut nn = n as i32;
        self.cc.launch(
            f,
            n.div_ceil(256) as u32,
            1,
            256,
            &mut crate::rawcuda::args::l3(&mut p_y, &mut ss, &mut nn),
        )
    }

    /// t행 MoE(≤16)·전량 디바이스 — `g4_moe_host_t`의 디바이스 i/o 판.
    /// 입력(라우터 hn·전문가 nm)이 이미 디바이스에 있고(상주 청크 노름 출력),
    /// 출력은 지정 슬롯에 남는다(h2d/d2h 0). 커널 시퀀스·상주 계약은
    /// `g4_moe_host_t`와 동일(라우터 → topk_t → pes → glu전문가 → down → accum).
    pub fn g4_moe_dev_t(
        &mut self,
        il: usize,
        dhn: CUdeviceptr,
        dxn: CUdeviceptr,
        pes_dev: CUdeviceptr,
        t: usize,
        out_slot: usize,
    ) -> Result<CUdeviceptr, String> {
        let h = self.hidden;
        let n_exp = self.moe.n_experts;
        let tk = self.moe.top_k.max(1);
        let n_ff = self.moe.moe_ffn;
        if t == 0 || t > G4_CHUNK_MAX || dhn == 0 || dxn == 0 || pes_dev == 0 {
            return Err(format!(
                "g4_moe_dev: 계약 위반 t={t} dhn={dhn} dxn={dxn} pes={pes_dev}"
            ));
        }
        if n_exp == 0 || !self.moe.moe_resident {
            return Err("g4_moe_dev: 전문가 비상주(스트리밍) — t행 미지원(호출부 t=1 폴백)".into());
        }
        let _g = self.cc.guard()?;
        self.ensure_moe_bufs()?;
        // ① 라우터 → drt [t][n_exp]. [A2 2026-10-15] t>16은 행 분할(위 동형).
        let rn = format!("blk.{il}.moe_gate.weight");
        if self.lins.contains_key(&rn) {
            self.gemv_t_rows(&rn, dhn, self.moe.drt.ptr, t)?;
        } else {
            // 플레인 라우터 — t행 플레인 GEMV(행별 산술 = t=1 GEMV와 비트동일).
            self.plain_gemv_t_rows(&rn, dhn, self.moe.drt.ptr, t)?;
        }
        // ② top-k(디바이스, t행) — softmax 전체 → top-k → 재정규화.
        {
            let f = self.cc.function("w4a16_moe_topk_t")?;
            let (mut p_lg, mut p_ix, mut p_wt) =
                (self.moe.drt.ptr, self.moe.moe_idx, self.moe.moe_wt);
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
        // [P4 2026-10-15] 라우팅 캡처(eval 진단 전용) — top-k 선택(id)을 층마다
        // 호스트로 복사(판독만 — 수치 불변). 세그먼트 = [t][k] u32, 청크×층
        // 실행 순서로 append(소비자 = w4a16-eval --route).
        if self.route_capture {
            let mut buf = vec![0u32; t * tk];
            self.cc
                .d2h_async(buf.as_mut_ptr() as *mut u8, self.moe.moe_idx, buf.len() * 4)?;
            self.cc.sync()?;
            self.route_out.extend_from_slice(&buf);
        }
        // ③ pes(μ_e) — 인플레이스(가중 [t][top_k] 전량).
        {
            let ns = t * tk;
            let f = self.cc.function("w4a16_moe_pes_scale")?;
            let (mut p_w, mut p_i, mut p_p) = (self.moe.moe_wt, self.moe.moe_idx, pes_dev);
            let mut p_n = ns as i32;
            self.cc.launch(
                f,
                (ns as u32).div_ceil(256).max(1),
                1,
                256,
                &mut crate::rawcuda::args::l4(&mut p_w, &mut p_i, &mut p_p, &mut p_n),
            )?;
        }
        // ④ 전문가 — 상주 경로 2종:
        //  [⑥ 2026-10-18] t>G4_TROW_MAX(프리필 청크) = **그룹드 mma A 단일
        //  f16 패스** (CMP 방향 — GA100-class는 FP32 약·TC 강·HBM 넓음):
        //  전문가-우선 정렬 후 가중을 전문가당 1회만 판독하고, A는 단일
        //  f16(RNE) — [A2b] hi/lo 2분해는 ncu wait 스톨(고정지연 의존) 지배로
        //  제거(moepp.cu). 잔여 오차 = A f16 반올림 + f32 누산 순서 — eval
        //  게이트 판정(과거 실측 en PPL +0.17%·agree 98.6% 밴드).
        //  t≤16(디코드 배치·단문 청크) = 융합 GEMV(f32 비트동일 기준선).
        let ns = t * tk;
        let base = il * n_exp * 3;
        let out = self.g4_scratch_slot(out_slot, t * h * 4)?;
        // [2026-10-19] 마린 MoE 시도(상주 전용) — gemma4 = gelu_tanh 활성.
        // 정렬(align2)이 공통 전제 — 마린·레거시 모두 먼저 수행.
        let mut marlin_done = false;
        if t > G4_TROW_MAX && n_exp <= 1024 {
            self.moe_align_launch(ns)?;
            marlin_done = self.marlin_moe_ffn_dev(il, dxn, t, h, n_ff, tk, true, out)?;
            if !marlin_done {
                let xp = self.cast_f16_dev(dxn, t * h)?;
                self.gemm_g32_mma_grp2_launch(
                    base,
                    xp,
                    h,
                    tk,
                    self.moe.exp.gate,
                    self.moe.exp.up,
                    n_ff,
                    h,
                )?;
                self.ew_gelu_dev(
                    self.moe.exp.gate,
                    self.moe.exp.up,
                    self.moe.exp.act,
                    ns * n_ff,
                )?;
                let ap = self.cast_f16_dev(self.moe.exp.act, ns * n_ff)?;
                self.gemm_g32_mma_grp_d64_launch(base + 2, ap, n_ff, 1, self.moe.exp.dn, h, n_ff)?;
            }
        } else {
            self.gemv_experts_glu_launch(base, ns, dxn, h, tk, self.moe.exp.act, n_ff, h)?;
            self.gemv_experts_launch(
                base + 2,
                ns,
                self.moe.exp.act,
                n_ff,
                1,
                self.moe.exp.dn,
                h,
                n_ff,
            )?;
        }
        if !marlin_done {
            self.moe_accum_dev(self.moe.moe_wt, self.moe.exp.dn, out, tk, ns, h)?;
        }
        Ok(out)
    }

    /// t행 head(TR int8) → 슬롯 — 입력 디바이스(xn [t][hidden]), d2h는 호출부.
    pub fn g4_head_dev_t(
        &mut self,
        x_dev: CUdeviceptr,
        t: usize,
        out_slot: usize,
    ) -> Result<CUdeviceptr, String> {
        if self.head_w == 0 {
            return Err("g4_head_dev: head 미등록".into());
        }
        if t == 0 || t > G4_CHUNK_MAX || x_dev == 0 {
            return Err(format!("g4_head_dev: t={t} x={x_dev}"));
        }
        let y = self.g4_scratch_slot(out_slot, t * self.head_n * 4)?;
        // [A2 2026-10-15] t>G4_TROW_MAX = 행 분할(head i8_t 스위치 ≤16 계약) —
        // 프리필은 t=1(마지막 행)·디코드는 t≤8이라 실사용은 단일 호출뿐이지만,
        // 청크 상한 상향(64)에 대비해 계약을 지킨다.
        let mut r0 = 0usize;
        while r0 < t {
            let rows = G4_TROW_MAX.min(t - r0);
            self.head_gemv_t_launch(
                x_dev + (r0 * self.hidden * 4) as u64,
                y + (r0 * self.head_n * 4) as u64,
                rows,
            )?;
            r0 += rows;
        }
        Ok(y)
    }

    /// [gemma4 P3 v2 2026-10-14] 청크 진입 — 디바이스 상주 경로(프롤로그 →
    /// 본문 → head d2h). 반환 = 행별 로짓(raw — softcap은 호출부 계약).
    /// 프롤로그: 스트림·핀드 출력 확보, 슬롯 선할당, x 업로드, 어텐션 테이블.
    /// 본문: 전부 같은 스트림 발사(중간 d2h 0) — **그래프 캡처 대상**.
    /// 그래프 히트면 본문+head d2h가 리플레이 1런치로 붕괴한다(캡처 키 =
    /// t·슬롯집합·head_last — pos는 테이블(프롤로그)로 매 청크 갱신).
    pub fn g4_chunk_dev(
        &mut self,
        slots: &[usize],
        pos: &[u32],
        rows: &[f32],
        t: usize,
        head_last: bool,
        nw_per_layer: usize,
        eps: f32,
    ) -> Result<Vec<Vec<f32>>, String> {
        let h = self.hidden;
        if t == 0 || t > G4_CHUNK_MAX || slots.len() != t || pos.len() != t || rows.len() != t * h {
            return Err(format!(
                "g4_chunk_dev: t={t} rows={} slots={} pos={} (1..=16 · t*h 계약)",
                rows.len(),
                slots.len(),
                pos.len()
            ));
        }
        if self.head_w == 0 {
            return Err("g4_chunk_dev: head 미등록(upload_head 선행)".into());
        }
        let _g = self.cc.guard()?;
        // ① 프롤로그 — 그래프 밖(할당·복사·캐시 준비).
        if self.cc.stream.is_null() {
            self.cc.create_stream()?; // 그래프 캡처는 기본 스트림 불가
        }
        self.g4_warm_slots(G4_CHUNK_MAX)?;
        self.ensure_moe_bufs()?;
        self.g4_atab_prepare(slots, pos)?;
        let x = self.g4_scratch_slot(Self::G4_X, t * h * 4)?;
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
        let xb = unsafe { std::slice::from_raw_parts(rows.as_ptr() as *const u8, rows.len() * 4) };
        self.cc.h2d_async(x, xb)?;
        self.g4_ensure_pin_out(G4_CHUNK_MAX * self.head_n * 4)?;
        // ② 본문(디바이스 — 캡처/리플레이 또는 직접 발사).
        if self.g4_graph_ok() {
            if let Err(e) = self.g4_graph_run(slots, pos, t, head_last, nw_per_layer, eps) {
                eprintln!("[g4-graph] 실패 — 직접 경로 폴백: {e}");
                self.graph.g4_capture_failed = true;
                let _ = self.cc.sync();
                self.g4_run_direct(slots, pos, t, head_last, nw_per_layer, eps)?;
            }
        } else {
            self.g4_run_direct(slots, pos, t, head_last, nw_per_layer, eps)?;
        }
        self.cc.sync()?;
        // SAFETY: pin_g4_out는 g4_ensure_pin_out의 할당 계약(≥ nrow*head_n*4).
        let nrow = if head_last { 1 } else { t };
        let flat = unsafe {
            std::slice::from_raw_parts(self.graph.pin_g4_out as *const f32, nrow * self.head_n)
        };
        Ok(flat
            .chunks(self.head_n)
            .map(|r| r.to_vec())
            .collect::<Vec<_>>())
    }

    /// 그래프 모드 가능 여부 — LLM170_GRAPH(기본 ON)·debug_layers 아님·
    /// 상주 MoE(비상주는 g4_moe_dev_t가 거부)·캡처 실패 후 고정.
    fn g4_graph_ok(&self) -> bool {
        llm170_diag::flag::ne0("LLM170_GRAPH")
            && !self.debug_layers
            && !self.route_capture
            && !self.graph.g4_capture_failed
            && self.moe.n_experts > 0
            && self.moe.moe_resident
    }

    /// 본문 + head d2h(핀드) — 직접 발사 몫(캡처 히트/미스 공용 조각).
    fn g4_run_direct(
        &mut self,
        slots: &[usize],
        pos: &[u32],
        t: usize,
        head_last: bool,
        nw_per_layer: usize,
        eps: f32,
    ) -> Result<(), String> {
        self.g4_chunk_body(slots, pos, t, head_last, nw_per_layer, eps)?;
        self.g4_out_d2h(t, head_last)
    }

    /// head 로짓 d2h(핀드 목적지 — 캡처 노드로 기록 가능).
    fn g4_out_d2h(&mut self, t: usize, head_last: bool) -> Result<(), String> {
        let nrow = if head_last { 1 } else { t };
        let hd = self.g4_scratch_slot(Self::G4_HEADOUT, t * self.head_n * 4)?;
        self.cc
            .d2h_async(self.graph.pin_g4_out as *mut u8, hd, nrow * self.head_n * 4)
    }

    /// 그래프 히트면 리플레이(1런치), 미스면 직접 실행 + 캡처 등록(미실행).
    fn g4_graph_run(
        &mut self,
        slots: &[usize],
        pos: &[u32],
        t: usize,
        head_last: bool,
        nw_per_layer: usize,
        eps: f32,
    ) -> Result<(), String> {
        if let Some(k) = self
            .graph
            .g4_graphs
            .iter()
            .position(|e| e.t == t && e.head_last == head_last && e.slots == slots)
        {
            let exec = self.graph.g4_graphs[k].exec;
            return self.cc.graph_launch(exec);
        }
        // 미스 — 직접 실행이 정답(KV·pos 소비) → 캡처(실행 아님) 등록.
        self.g4_run_direct(slots, pos, t, head_last, nw_per_layer, eps)?;
        self.cc.sync()?; // 캡처 전 파이프라인 정리(캡처는 유휴 스트림에서).
        if !self.graph.g4_capture_failed {
            match self.g4_capture(slots, pos, t, head_last, nw_per_layer, eps) {
                Ok(()) => {}
                Err(e) => {
                    eprintln!("# g4 그래프 캡처 실패 — 직접 발사 유지: {e}");
                    self.graph.g4_capture_failed = true;
                    let _ = self.cc.capture_end(); // 캡처 상태 정리(멱등 아님 — 실패 시에만)
                    let _ = self.cc.sync();
                }
            }
        }
        Ok(())
    }

    /// 청크 본문 캡처 — 슬롯집합·t·head_last 키. 캡처는 **실행하지 않는다**
    /// (상태 불변 — KV/pos/출력은 직전 직접 실행분이 이미 반영).
    fn g4_capture(
        &mut self,
        slots: &[usize],
        pos: &[u32],
        t: usize,
        head_last: bool,
        nw_per_layer: usize,
        eps: f32,
    ) -> Result<(), String> {
        let tt = std::time::Instant::now();
        if self.graph.g4_graphs.len() >= 32 {
            let old = self.graph.g4_graphs.remove(0);
            let _ = self.cc.graph_destroy(old.exec, old.handle);
        }
        self.cc.capture_begin()?;
        let r = self
            .g4_chunk_body(slots, pos, t, head_last, nw_per_layer, eps)
            .and_then(|_| self.g4_out_d2h(t, head_last));
        if let Err(e) = r {
            let _ = self.cc.capture_end();
            return Err(e);
        }
        let g = self.cc.capture_end()?;
        let e = match self.cc.graph_instantiate(g) {
            Ok(e) => e,
            Err(err) => {
                // 핸들 유실 방지(캡처 1회·실패 시 재시도 없음 — 고정 플래그).
                let _ = self.cc.graph_destroy(std::ptr::null_mut(), g);
                return Err(err);
            }
        };
        if let Err(err) = self.cc.graph_upload(e) {
            let _ = self.cc.graph_destroy(e, g);
            return Err(err);
        }
        self.graph.g4_graphs.push(G4GraphEntry {
            t,
            slots: slots.to_vec(),
            head_last,
            exec: e,
            handle: g,
        });
        eprintln!(
            "[g4-graph] captured t={t} slots={slots:?} head_last={head_last} ({:.2}ms)",
            tt.elapsed().as_secs_f64() * 1e3
        );
        Ok(())
    }

    /// 청크 head 판독 핀드 버퍼 확보 — 부족 시 재할당(캡처 그래프 전량 폐기).
    fn g4_ensure_pin_out(&mut self, bytes: usize) -> Result<(), String> {
        if !self.graph.pin_g4_out.is_null() && self.graph.pin_g4_out_len >= bytes {
            return Ok(());
        }
        self.graph_invalidate();
        if !self.graph.pin_g4_out.is_null() {
            let _ = self.cc.pinned_free(self.graph.pin_g4_out);
        }
        self.graph.pin_g4_out = self.cc.pinned_alloc(bytes)?;
        self.graph.pin_g4_out_len = bytes;
        Ok(())
    }

    /// 최종 노름 hidden → GPU head 로짓(호스트 회수). `upload_head` 선행 계약
    /// (행별 int8 W8A16 — W4A16 단일 head 계약).
    pub fn head_logits_host(&mut self, xn: &[f32]) -> Result<Vec<f32>, String> {
        if self.head_w == 0 {
            return Err("head_logits_host: head 미등록 — upload_head 선행".into());
        }
        if xn.len() != self.hidden {
            return Err(format!(
                "head_logits_host: xn={} != hidden={}",
                xn.len(),
                self.hidden
            ));
        }
        let _g = self.cc.guard()?;
        let dx = self.cc.alloc(self.hidden * 4)?;
        let r = (|| -> Result<Vec<f32>, String> {
            let b: Vec<u8> = xn.iter().flat_map(|x| x.to_le_bytes()).collect();
            self.cc.h2d(dx, &b)?;
            self.head_gemv_launch(dx, self.head_out)?;
            let mut ob = vec![0u8; self.head_n * 4];
            self.cc
                .d2h_async(ob.as_mut_ptr(), self.head_out, self.head_n * 4)?;
            self.cc.sync()?;
            Ok(ob
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect())
        })();
        let _ = self.cc.free(dx);
        r
    }
}
