// ============================================================================
// ⚠️ 미배선 — 결함 있음. 서버에 연결하지 말 것 (2026-10-07)
// ============================================================================
// 구현·검증까지 갔으나 **행 간 상태 오염**이 확인돼 배선을 되돌렸다.
// 슬롯 N개가 모두 살아 있는 상태에서 이 경로를 켜면 동시 요청이 서로의
// 상태를 읽는다. 증거(원자적 재현):
//   - 슬롯 4 동시 vs 직렬 비교에서 **서로 다른 두 프롬프트가 idx=9부터
//     완전히 동일한 24토큰열**을 냈다(`25 198 16 13 48455 264 …`).
//   - 실행마다 어긋나는 프롬프트는 바뀌지만(슬롯 배정 순서가 비결정) 같은
//     실행 안에서는 완전히 재현된다 → 상태 공유이지 환원 순서 차이가 아니다.
//     f16 mma 누산(§6-3)이 만든 argmax 뒤집힘은 두 프롬프트가 같은 출력을
//     낼 수 없다.
//   - T=1은 게이트 기준선 그대로 PASS한다(배치 산술 자체는 맞다). T>1에서만
//     깨진다.
// 남은 범인 후보(다음 세션이 이 순서로 좁힐 것):
//   1. 스케줄러의 슬롯 배정 — 동시 도착 시 두 작업이 한 슬롯을 공유할 수
//      있는지. 이 경로는 중복 슬롯을 Err로 막지만, 그 Err 발생 **이전**에
//      something이 오염될 경로가 있다.
//   2. `attn_set_pos`/`slot_pos` 갱신 시점 — 배치 종료 시 pos를 한 번에
//      올리는 지금 방식이, 행별로 다른 pos에서 fwd3s를 돌리는 동안 pp[0]을
//      엮지 않는지.
//   3. 행 오프셋 계산 — `xn + i*hidden`, `c0 + i*conv_ch`, `rowbuf + i*q_dim`
//      가운데 하나가 stride 규약과 어긋났을 가능성. 어텐션 qg 행 폭은
//      q_heads*512(12288)이고 kin/vin은 kv_heads*256(1024)이며 v_len과 다르다.
// 도구: `load_slots(dir, lim_layers, kvcap, slots)`의 `lim_layers`로 층을
// 줄여(4·8·16…) 슬롯 2개 배치 vs 직렬 디코드의 최초 발산층을 이분 탐색하면
// 1·2·3 중 어느 것인지 확정된다. 그 게이트를 통과하기 전까지 배선 금지.
// ============================================================================

//! 슬롯 간 디코드 배치 — 활성 슬롯 N개를 T=N으로 한 번에 디코드한다.
//!
//! [왜 필요한가 — plans/cuda-port.md §5 우선순위 1의 근거)
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
//! 한다. `gemm2`(=배치 GEMM)는 타일 역양자화를 **한 번** 하고 T행에
//! 적용하므로, T행이 한 번에 들어가면 그만큼 절약된다.
//!
//! [구조 — 커널 변경 없음]
//! S11의 `forward_batch_device`(슬롯 **내** 프리필 T≤8)를 그대로 재사용한다.
//! 유일한 차이는 어텐션·GDN 호출을 슬롯별로 나눈다는 것인데, 커널은 상태를
//! 슬롯 오프셋으로寻기 때문에(원장 S8) 행마다 다른 슬롯 인자를 넘기는 것만
//!으로 정확하다. 잔차 `dres`도 문제가 아니다 — `norm_resid_dev`가 `dres`를
//! **인자**로 받으므로 T행 전체를 한 번에 잔차 누산할 수 있고, 행 포인터
//! 오프셋은 `dxn`(=[T][hidden]) 쪽만 하면 된다.
//!
//! 층 루프(행 i = items[i]):
//! 1. `norm_resid_dev(..., T)` — 배치. 잔차 누산이 행 독립이므로 한 번에.
//! 2. q/k/v·in_proj — `bstage`(gemm2 T행) **배치**. 트렐리스 역양자화 공유.
//! 3. 어텐션·GDN — **행별**(T회). 슬롯 인자가 다르므로 나눌 수밖에 없다.
//!    산출을 [T][q_dim]/[T][v_len]로 모아 다음 단계를 배치화한다.
//! 4. o_proj·out_proj·gate/up·down — `bgemv`/`bgemv_down`(행별 T회) + `bstage`.
//! 5. lm_head — `gemm2` T행 → 행별 argmax.
//!
//! [정합 — 이 경로가 기존 결과를 바꾸지 않아야 한다]
//! 행 i의 계산은 (a) 같은 커널,(b) 같은 슬롯 상태,(c) 같은 입력 행을 쓴다.
//! 배치화는 T행을 한 번의 커널에 넣을 뿐 산술 순서를 바꾸지 않는다. 단
//! `gemm2` 경로의 f16 mma 누산 때문에 T>1 로짓은 T=1과 ulp 수준으로
//! 다르다(S11이 이미 기록한 사실 · 1.6e-2). 판정은 **argmax 일치**다 —
//! 게이트 스크립트와 같은 기준. `scripts/verify_cuda_slots.py`가 이
//! 경로의 슬롯 격리를 토큰열로 검증한다.
//!
//! [도메인] T ≤ ATTN_F3S_TMAX(8) — 어텐션 fwd3s 소형 전용. 활성 슬롯이
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
    /// 스케줄러 배선은 `crates/server/src/sched.rs`의 Exl3Cuda 분기다.
    /// 활성 슬롯이 `ATTN_F3S_TMAX`(8)를 넘으면 호출자가 청크로 나눈다.
    ///
    /// 반환은 각 행의 argmax 토큰(행 순서 = `items` 순서). 전 슬롯의
    /// 위치가 1씩 전진한다. `items` 안의 슬롯은 **중복 불가** — 같은 슬롯을
    /// 두 번 넣으면 그 슬롯의 GDN/KV 상태가 한 스텝에 두 번 갱신되어
    /// 상태가 깨진다(조용한 오염 금지이므로 에러로 거절한다).
    pub fn decode_batch_slots(&mut self, items: &[(usize, u32)]) -> Result<Vec<u32>, String> {
        let (h, n_slots) = (self.hidden, self.n_slots.max(1));
        let t = items.len();
        if h == 0 || t == 0 {
            return Err("exl3-cuda: 배치 디코드 — 빈 입력".into());
        }
        let tmax = crate::rawcuda::attn_cuda::ATTN_F3S_TMAX;
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
                self.bstage(&format!("{att}.q_proj"), xn, 0, t)?;
                self.bstage(&format!("{att}.k_proj"), xn, 1, t)?;
                self.bstage(&format!("{att}.v_proj"), xn, 2, t)?;
                let (c0, c1, c2) = (self.bchain(0)?, self.bchain(1)?, self.bchain(2)?);
                self.ensure_attn_bufs_pub(1)?;
                for (i, &(slot, _)) in items.iter().enumerate() {
                    // qg는 q_heads*512, kin/vin은 kv_heads*256 — 행 폭이 다르다.
                    let out = self.attn_chain_dev_run(
                        slot,
                        il / 4,
                        1,
                        c0 + (i * qg_dim) as u64 * 4,
                        c1 + (i * v_len) as u64 * 4,
                        c2 + (i * v_len) as u64 * 4,
                    )?;
                    self.cc
                        .d2d(rowbuf + (i * q_dim) as u64 * 4, out, q_dim * 4)?;
                }
                rowbuf
            } else {
                let att = format!("{lp}.linear_attn");
                self.bstage(&format!("{att}.in_proj_qkv"), xn, 0, t)?;
                self.bstage(&format!("{att}.in_proj_z"), xn, 1, t)?;
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
            // 입력 행 폭을 층 종류별로 정확히 넘긴다(원장 S11 stride 결함).
            let bstride = if il % 4 == 3 { q_dim } else { v_len };
            let out = self.bgemv(&lo, branch, bstride, t)?;
            let xn2 = self.norm_resid_dev(2 * il + 1, self.dres, out, t)?;
            let mlp = format!("{lp}.mlp");
            self.bstage(&format!("{mlp}.gate_proj"), xn2, 0, t)?;
            self.bstage(&format!("{mlp}.up_proj"), xn2, 1, t)?;
            let (c0, c1) = (self.bchain(0)?, self.bchain(1)?);
            self.ew_batch(c0, c1, t)?;
            let act = self.bchain(3)?;
            ab = self.bgemv_down(&format!("{mlp}.down_proj"), act, t)?;
        }

        // lm_head — [T][hidden] → [T][n_head] 한 번에(gemm2가 역양자화 공유).
        let xn_final = self.norm_resid_dev(2 * self.n_layers, self.dres, ab, t)?;
        let n_head = self.lin_copy("lm_head").map(|l| l.n)?;
        let hd_src = self.bchain(3)?;
        self.cc.d2d(hd_src, xn_final, t * self.hidden * 4)?;
        let logits_ptr = self.gemm2_dev("lm_head", hd_src, t)?;
        let mut lb = vec![0u8; t * n_head * 4];
        self.cc.d2h(&mut lb, logits_ptr)?;
        self.cc.sync()?;
        // SAFETY: d2h 동기 완료 — lb는 t*n_head개의 f32 LE 값.
        let all = unsafe { std::slice::from_raw_parts(lb.as_ptr() as *const f32, t * n_head) };
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
        // 위치 전진은 행마다 1씩 — 슬롯 상태(attn_set_pos)가 이미 갱신됐지만
        // 호스트 pos도 맞춰야 다음 스텝이 맞는 위치에서 시작한다.
        for (i, &(slot, _)) in items.iter().enumerate() {
            self.slot_pos[slot] = pos_of[i] + 1;
            self.attn_set_pos(slot, pos_of[i] + 1)?;
        }
        Ok(toks)
    }
}
