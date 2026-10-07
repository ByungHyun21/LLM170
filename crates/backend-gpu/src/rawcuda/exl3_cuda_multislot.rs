// ============================================================================
// 슬롯 간 배치 — 배선·검증 완료(2026-10-08, plans/cuda-port.md §1 착수조건 1)
// ============================================================================
// [경과 요약 — 원장 계승]
// 1. (2026-10-07) `cuda_probe ms`(a9c7158)로 원시 자체는 64층 전체
//    T=2≡T=1 검증 완료. 단 그 판정은 **같은 배치 경로 안에서의 T 비교**
//    라 상태 격리만 본다.
// 2. (2026-10-07) 실측 원장: T=4 동시 디코드 24스텝에서 토큰 2/4가
//    T=1 기준과 어긋났다. 당시 "gemm2 환원 순서" 진단은 **오진** —
//    msrv(서버 타임라인 재현)가 gemv_t(비트동일 선형)로도 같은 증상을
//    재현해 반증됐다.
// 3. (2026-10-08) 진짜 원인: 배치 경로가 어텐션 k/v 스테이징 행 폭을
//    GDN v_len로 썼다. 어텐션 k/v 폭은 **kv_dim**(kv_heads·256)이며
//    v_len(GDN v_heads 폭)과 다르다 — 0행(오프셋 0)만 우연히 맞고 1행부터
//    엉뚱한 오프셋을 읽었다. ms4b(1스텝 로짓 to_bits) lim 스윕으로 국소화:
//    GDN 전수(≤3층) 비트일치 · 최초 어텐션 층(4층)부터 rows≥1 전체
//    로짓 maxdiff ~2.3e-2. kv_dim 수정 후 1스텝 비트일치(전 층).
// 4. (2026-10-08) 종단 검증 전부 PASS — gemv_t T=1..8 to_bits 0불일치,
//    ms(T=2 1스텝)·ms4(동일 프롬프트 4슬롯 순수 T=4 24스텝)·msrv(서버
//    타임라인 24스텝)·verify_cuda_slots.py(슬롯 4 동시 vs 슬롯 1 직렬,
//    겹침배율 3.96x, 4 프롬프트 길이 41/82/123/205 전부 24토큰 일치).
// 5. (2026-10-08) 실측 배율: bench_cuda_slots.py 4슬롯 집합 39.3 tok/s =
//    단일 24.2 tok/s의 **1.62배**(목표 ≥3x 미달). ms4 스텝 시간 분해:
//    직렬 33.0ms · 배치 T=4 67.6ms — 이상적(가중치 1회 판독)은 ~34ms.
// 6. (2026-10-08) §1.2 그래프 캡처 사전 측정(nsys, 0 vs 6 배치 스텝
//    차집합) — **구현 보류 판정**: 배치 스텝당 launch 2,484회 · API 시간
//    3.2ms · GPU busy 62.1ms / wall 67.6ms = idle 8%라 캡처 이득 상한이
//    ~8%다. 시간은 커널 실행이 먹는다: gemv_t 45.7ms(74%, 401회 ×
//    avg 114µs — T=1 gemv ~45µs의 2.5배, 8행 누산기 배열의 레지스터
//    압박으로 점유율 하락 추정), GDN 행체인 13.3ms(scan 10.8 지배),
//    어텐션 1.0ms, 소형 커널(had/norm/ew) 3.5ms. ≥3x 레버 순위:
//    (a) gemv_t 점유율 복원 — T별 커널 인스턴스(컴파일타임 T)로 누산기
//    폭을 T에 맞춘다(FOLD 케이던스·행별 순서 유지 — 비트계약 재검증),
//    (b) GDN scan 행배치(행마다 슬롯 인덱스 배열 인자 — 커널 변경).
// ============================================================================

//! 슬롯 간 디코드 배치 — 활성 슬롯 N개를 T=N으로 한 번에 디코드한다.
//!
//! [왜 필요한가 — plans/cuda-port.md §1 우선순위 1의 근거]
//! S8는 다중 슬롯을 열었지만 **연속배치가 아니었다**. `sched.rs`의 CUDA
//! 분기는 `for &i in &active { e.step_tok_device(i, next) }`로 슬롯마다
//! 64층 forward를 통째로 따로 돌린다 — 시간분할이다. 실측이 증명한다
//! (`scripts/bench_cuda_slots.py`, RTX 4090 sm_89 · 27B EXL3 · ctx 1024):
//!
//! ```text
//! slots=1 conc=1: 128토큰 → 집합 24.55 tok/s
//! slots=4 conc=4: 512토큰 → 집합 24.48 tok/s  = 1.00배 (이상적 4배)
//! ```
//!
//! 원인은 가중치를 슬롯 간에 공유하지 않는 것이다. EXL3 5.00bpw는 행마다
//! 트렐리스 역양자화가 필요하므로, GEMV를 T회 따로 부르면 역양자화도 T번
//! 한다. `gemv_t`는 스레드당 추출·디코드 1회를 T행이 공유하므로 T행이
//! 한 번에 들어가면 그만큼 절약된다.
//!
//! [구조 — 커널 변경 없음(어텐션·GDN) + gemv_t(선형)]
//! S11의 `forward_batch_device`(슬롯 **내** 프리필 T≤8) 구조를 재사용하되
//! 트렐리스 선형은 전부 gemv_t 체인을 쓴다(bstage_gt·bgemv_gt).
//! 어텐션·GDN만 행별(T=1씩 T회)인데, 커널은 상태를 슬롯 오프셋으로
//! 찾으므로(원장 S8) 행마다 다른 슬롯 인자를 넘기는 것만으로 정확하다.
//! 잔차 `dres`도 문제가 아니다 — `norm_resid_dev`가 `dres`를 **인자**로
//! 받으므로 T행 전체를 한 번에 잔차 누산할 수 있다.
//!
//! 층 루프(행 i = items[i]):
//! 1. `norm_resid_dev(..., T)` — 배치. 잔차 누산이 행 독립이므로 한 번에.
//! 2. q/k/v·in_proj — `bstage_gt`(gemv_t T행) **배치**. 역양자화 공유.
//! 3. 어텐션·GDN — **행별**(T회). 슬롯 인자가 다르므로 나눌 수밖에 없다.
//!    산출을 [T][q_dim]/[T][v_len]로 모아 다음 단계를 배치화한다.
//! 4. o_proj·out_proj·down — `bgemv_gt`(gemv_t) — 행 폭이 곧 k라 직통.
//! 5. lm_head — `gemv_t` T행 → 행별 argmax.
//!
//! [정합 — 비트계약, plans/cuda-port.md §1 착수조건 1]
//! 행 i의 계산은 (a) T=1 순차 경로(forward_device)와 같은 커널·같은
//! 인자,(b) 같은 슬롯 상태,(c) 같은 입력 행을 쓴다. 선형은 gemv_t가
//! 행별로 T=1 GEMV와 **비트동일**하고(위 커널 주석), 노름·ew·had_in/
//! had_out은 행 인덱스 커널이라 T와 무관하게 행별 동일하다. 따라서 배치
//! 디코드의 토큰 스트림은 슬롯 1 직렬 디코드와 정확히 같다 —
//! `scripts/verify_cuda_slots.py`(슬롯 N 동시 vs **슬롯 1** 직렬)가
//! 종단 판정한다.
//!
//! [도메인] T ≤ GEMV_T_TMAX(=8, 어텐션 fwd3s 상한과 동일) — 활성 슬롯이
//! 넘으면 호출자가 청크로 나눠야 한다(에러로 알린다 — 조용한 절단 금지).

use crate::rawcuda::exl3_cuda::Exl3CudaDecoder;

/// 슬롯 간 배치가 행별 어텐션·GDN 산출을 모아 두는 스테이징 버퍼 인덱스.
///
/// 0~4는 S11 배치가 이미 전부 점유한다(0·1·2 = q/k/v·qkv/z 스테이징,
/// 3 = ew 산출, 4 = o_proj·out_proj·down 산출). 행 수집용으로 **6번째
/// 버퍼를 새로 잡았다** — 기존 슬롯을 재사용하면 그 슬롯을 읽는 커널과
/// 충돌한다(원장 S11 "커널이 이전 포인터를 읽는 창"과 같은 부류).
const ROWBUF: usize = 5;

impl Exl3CudaDecoder {
    /// `(슬롯, 토큰)` 목록을 T행으로 묶어 한 번에 디코드한다.
    ///
    /// 스케줄러 배선은 `crates/server/src/sched.rs`의 Exl3Cuda 분기가
    /// `Exl3CudaEngine::step_batch`로 호출한다. 활성 슬롯이 상한을 넘으면
    /// 엔진 쪽에서 청크로 나눈다.
    ///
    /// 반환은 각 행의 argmax 토큰(행 순서 = `items` 순서). 전 슬롯의
    /// 위치가 1씩 전진한다. `items` 안의 슬롯은 **중복 불가** — 같은 슬롯을
    /// 두 번 넣으면 그 슬롯의 GDN/KV 상태가 한 스텝에 두 번 갱신되어
    /// 상태가 깨진다(조용한 오염 금지이므로 에러로 거절한다).
    pub fn decode_batch_slots(&mut self, items: &[(usize, u32)]) -> Result<Vec<u32>, String> {
        let (all, n_head) = self.decode_batch_slots_logits(items)?;
        let t = items.len();
        let mut toks = Vec::with_capacity(t);
        for r in 0..t {
            let row = &all[r * n_head..(r + 1) * n_head];
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &v) in row.iter().enumerate() {
                if v > bv {
                    bv = v;
                    best = i;
                }
            }
            toks.push(best as u32);
        }
        Ok(toks)
    }

    /// `decode_batch_slots`의 로짓 판 — 상태 전진·계약은 동일하고 반환만
    /// [T][n_head] f32 로짓(행 우선)이다. 비트 판정 탐침(ms4b) 전용.
    pub fn decode_batch_slots_logits(
        &mut self,
        items: &[(usize, u32)],
    ) -> Result<(Vec<f32>, usize), String> {
        let (h, n_slots) = (self.hidden, self.n_slots.max(1));
        let t = items.len();
        if h == 0 || t == 0 {
            return Err("exl3-cuda: 배치 디코드 — 빈 입력".into());
        }
        let tmax = Exl3CudaDecoder::GEMV_T_TMAX;
        if t > tmax {
            return Err(format!(
                "exl3-cuda: T={t} > 슬롯 간 배치 상한 {tmax} — 활성 슬롯을 \
                 {tmax}개 단위로 나눠 호출할 것"
            ));
        }
        if self.norm_w_rows < 2 * self.n_layers + 1 || self.gdn.is_none() {
            return Err("exl3-cuda: 디코더 상수 미등록 — load_slots 필요".into());
        }
        let cap = self.attn_dims()?.cap;
        let mut seen = vec![false; n_slots];
        let mut pos_of = vec![0u32; t];
        for (i, &(slot, _tok)) in items.iter().enumerate() {
            if slot >= n_slots {
                return Err(format!("exl3-cuda: slot={slot} >= n_slots={n_slots}"));
            }
            if seen[slot] {
                return Err(format!(
                    "exl3-cuda: 슬롯 {slot}이(가) 배치 목록에 2번 — 한 스텝에 \
                     상태를 두 번 갱신하므로 금지"
                ));
            }
            seen[slot] = true;
            let pos = self.slot_pos[slot];
            if pos as usize + 1 > cap {
                return Err(format!(
                    "context overflow: slot{slot} pos={pos}+1 > kvcap={cap} (--ctx 상향 필요)"
                ));
            }
            pos_of[i] = pos;
        }

        let _g = self.cc.guard()?;
        self.ensure_chain_probe_bufs_pub()?;
        self.prewarm_chain_bufs()?;
        // gemv_t 체인 버퍼(dx·daht·dsbt·dyt)도 최대 형상으로 선확보 —
        // 진행 중 재할당 금지(원장 S10·S11, lm_head 어휘 폭 포함).
        self.prewarm_gemv_t_bufs(t)?;

        // 임베딩 [T][hidden] 한 번에 업로드(forward당 유일한 대량 h2d).
        let mut embed = Vec::with_capacity(t * h);
        for &(_, tok) in items {
            embed.extend_from_slice(&self.embed_row_host(tok));
        }
        let eb =
            unsafe { std::slice::from_raw_parts(embed.as_ptr() as *const u8, embed.len() * 4) };
        self.cc
            .h2d(self.dres, eb)
            .map_err(|e| format!("슬롯 간 배치 임베딩 {t}행 업로드: {e}"))?;

        let q_dim = self.attn_dims()?.q_dim();
        let kv_dim = self.attn_dims()?.kv_dim();
        let v_len = self.gdn_dims()?.v_len();
        let conv_ch = self.gdn_dims()?.conv_ch();
        let qg_dim = self.attn_dims()?.qg_dim();
        // 행 수집 버퍼: 어텐션 산출 [T][q_dim] 또는 GDN 산출 [T][v_len].
        let row_w = q_dim.max(v_len);
        self.ensure_stage(ROWBUF, row_w, t)?;
        let rowbuf = self.bchain(ROWBUF)?;

        let mut ab = self.dab_dev;
        let mut gi = 0usize;
        for il in 0..self.loaded_layers.min(self.n_layers) {
            let lp = format!("model.language_model.layers.{il}");
            // 배치 노름 — 잔차 누산이 행 독립이라 T행 한 번에.
            let xn = self.norm_resid_dev(2 * il, self.dres, ab, t)?;
            let branch = if il % 4 == 3 {
                let att = format!("{lp}.self_attn");
                self.bstage_gt(&format!("{att}.q_proj"), xn, 0, t)?;
                self.bstage_gt(&format!("{att}.k_proj"), xn, 1, t)?;
                self.bstage_gt(&format!("{att}.v_proj"), xn, 2, t)?;
                let (c0, c1, c2) = (self.bchain(0)?, self.bchain(1)?, self.bchain(2)?);
                self.ensure_attn_bufs_pub(1)?;
                for (i, &(slot, _)) in items.iter().enumerate() {
                    // qg는 q_heads·512, kin/vin은 **어텐션 kv_dim**(kv_heads·256)
                    // — 행 폭은 각 선형의 n이다. v_len(GDN v_head 폭)을 쓰면
                    // 0행만 우연히 맞고 1행부터 엉뚱한 오프셋을 읽는다
                    // (ms4b lim=4 실측 — 2026-10-08).
                    let out = self.attn_chain_dev_run(
                        slot,
                        il / 4,
                        1,
                        c0 + (i * qg_dim) as u64 * 4,
                        c1 + (i * kv_dim) as u64 * 4,
                        c2 + (i * kv_dim) as u64 * 4,
                    )?;
                    self.cc
                        .d2d(rowbuf + (i * q_dim) as u64 * 4, out, q_dim * 4)?;
                }
                rowbuf
            } else {
                let att = format!("{lp}.linear_attn");
                self.bstage_gt(&format!("{att}.in_proj_qkv"), xn, 0, t)?;
                self.bstage_gt(&format!("{att}.in_proj_z"), xn, 1, t)?;
                let (c0, c1) = (self.bchain(0)?, self.bchain(1)?);
                self.ensure_gdn_bufs_pub(1)?;
                for (i, &(slot, _)) in items.iter().enumerate() {
                    let out = self.gdn_chain_dev_run(
                        slot,
                        gi,
                        1,
                        xn + (i * h) as u64 * 4,
                        c0 + (i * conv_ch) as u64 * 4,
                        c1 + (i * v_len) as u64 * 4,
                    )?;
                    self.cc
                        .d2d(rowbuf + (i * v_len) as u64 * 4, out, v_len * 4)?;
                }
                gi += 1;
                rowbuf
            };
            let lo = if il % 4 == 3 {
                format!("{lp}.self_attn.o_proj")
            } else {
                format!("{lp}.linear_attn.out_proj")
            };
            // branch의 행 폭(어텐션 q_dim / GDN v_len)은 곧 해당 선형의
            // k라 [T][k] 연속 입력으로 직통한다(원장 S11 stride 결함 가드).
            let out = self.bgemv_gt(&lo, branch, t)?;
            let xn2 = self.norm_resid_dev(2 * il + 1, self.dres, out, t)?;
            let mlp = format!("{lp}.mlp");
            self.bstage_gt(&format!("{mlp}.gate_proj"), xn2, 0, t)?;
            self.bstage_gt(&format!("{mlp}.up_proj"), xn2, 1, t)?;
            let (c0, c1) = (self.bchain(0)?, self.bchain(1)?);
            self.ew_batch(c0, c1, t)?;
            let act = self.bchain(3)?;
            ab = self.bgemv_gt(&format!("{mlp}.down_proj"), act, t)?;
        }

        // lm_head — [T][hidden] → [T][n_head] 한 번에(gemv_t가 역양자화
        // 공유 · 행별 T=1 비트동일). xn_final(dxn)을 직통 건다.
        let xn_final = self.norm_resid_dev(2 * self.n_layers, self.dres, ab, t)?;
        let n_head = self.lin_copy("lm_head").map(|l| l.n)?;
        let logits_ptr = self.gemv_t_dev("lm_head", xn_final, t)?;
        let mut lb = vec![0u8; t * n_head * 4];
        self.cc.d2h(&mut lb, logits_ptr)?;
        self.cc.sync()?;
        // SAFETY: d2h 동기 완료 — lb는 t*n_head개의 f32 LE 값.
        let all = unsafe { std::slice::from_raw_parts(lb.as_ptr() as *const f32, t * n_head) };
        // 위치 전진은 행마다 1씩 — 슬롯 상태(attn_set_pos)가 이미 갱신됐지만
        // 호스트 pos도 맞춰야 다음 스텝이 맞는 위치에서 시작한다.
        for (i, &(slot, _)) in items.iter().enumerate() {
            self.slot_pos[slot] = pos_of[i] + 1;
            self.attn_set_pos(slot, pos_of[i] + 1)?;
        }
        Ok((all.to_vec(), n_head))
    }
}
