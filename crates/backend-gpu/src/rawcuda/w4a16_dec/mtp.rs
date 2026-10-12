//! [MTP 2026-10-12] 모델 자체 MTP 헤드 스펙 초안기 — 1레이어(전체 어텐션).
//!
//! 배경: 두 타깃 모델(27B·35B)이 `mtp.*` 블록을 BF16 원본으로 동봉하는데
//! 로더가 개수만 세고 미사용이었다(스펙 초안 = n-gram, 산문 발화 4/257).
//! MTP = pre_fc_norm(embed·hidden) → fc(cat) → 1레이어(전체 어텐션) → norm
//! → 공유 head argmax → 다음+1 토큰 초안. 시맨틱은 vLLM qwen3_5_mtp.py 미러
//! (concat 순서 = [embed_norm, hidden_norm]).
//!
//! 검증은 기존 스펙 체인(spec_verify t≤8·GDN 롤백)이 담당 — **무손실**
//! (초안은 검증 통과분만 방출). 이 모듈은 초안 생산만 한다.
//!
//! 어텐션: MTP 레이어를 **가상 전체-어텐션 층**(set_attn의 n_attn+1, 마지막
//! 인덱스)으로 편입 — qnw/knw 배열에 MTP q/k_norm 행이 추가되고 KV도 1층분
//! 확장된다(서버측). 커널 수정 0.
//!
//! 범위(v1): 27B dense MTP만. 35B MoE MTP(전문가 256 bf16)는 차기 —
//! mtp_ok=false로 두면 n-gram 폴백이 유지된다.
//!
//! t행 계약: 라운드 수용분(acc행)+보정 1행을 **한 번에**(t=acc+1) 처리해
//! 가중 판독을 상각한다(행별 산술은 t=1과 동일 계열 — plain_gemv_t 비트동일).
//! 수용분의 hidden은 검증 체인의 dres 행(수용 접두)에서 스냅샷한다.

use super::*;

/// MTP 아레나 행 상한 — 디코드 t≤8, 프리필 서브청크 64(가중 상각).
const MTP_TMAX: usize = 64;
/// 프리필 서브청크(행).
const MTP_PF_CHUNK: usize = 64;

impl super::W4a16Dec {
    /// MTP 활성 여부.
    pub fn mtp_enabled(&self) -> bool {
        self.mtp_ok
    }

    /// 슬롯 현재 위치(호스트 미러) — MTP pos_start 계산용.
    pub fn slot_pos_of(&self, slot: usize) -> u32 {
        self.slot_pos.get(slot).copied().unwrap_or(0)
    }

    /// MTP 등록 — 어텐션 확장(n_attn+1 가정) 후 호출. norms = 5행×hidden f32
    /// (0=pre_fc_norm_hidden 1=pre_fc_norm_embedding 2=input_layernorm
    ///  3=post_attention_layernorm 4=mtp.norm).
    pub fn set_mtp(
        &mut self,
        norms: &[f32],
        q_out: usize,
        kv_dim: usize,
        ffn: usize,
    ) -> Result<(), String> {
        let h = self.hidden;
        let dm = self.attn.ok_or("mtp: attn 미등록")?;
        if norms.len() != 5 * h {
            return Err(format!("mtp: norms != 5x{h}"));
        }
        if dm.n_attn == 0 {
            return Err("mtp: 전체-어텐션 층 없음".into());
        }
        let tmax = MTP_TMAX;
        // 아레나 — 서브 버퍼 오프셋(원소 단위).
        let (o_emb, o_hid, o_cat) = (0, tmax * h, 2 * tmax * h);
        let o_fc = o_cat + 2 * tmax * h;
        let (o_res, o_x, o_o) = (o_fc + tmax * h, o_fc + 2 * tmax * h, o_fc + 3 * tmax * h);
        let o_qg = o_fc + 4 * tmax * h;
        let o_kin = o_qg + tmax * q_out;
        let o_vin = o_kin + tmax * kv_dim;
        let o_g = o_vin + tmax * kv_dim;
        let o_u = o_g + tmax * ffn;
        let o_act = o_u + tmax * ffn;
        let o_dn = o_act + tmax * ffn;
        let o_lg = o_dn + tmax * h; // 로짓은 8행(헤드 t≤8) — 아래서 8로 고정
        let o_xn_e = o_lg + 8 * self.head_n;
        let o_xn_h = o_xn_e + tmax * h;
        let o_prev = o_xn_h + tmax * h;
        let total = o_prev + self.n_slots * h;
        // 기존 아레나 해제.
        if self.mtp_buf != 0 {
            self.cc.free(self.mtp_buf)?;
            self.mtp_buf = 0;
        }
        if self.mtp_nw != 0 {
            self.cc.free(self.mtp_nw)?;
            self.mtp_nw = 0;
        }
        if self.mtp_zero != 0 {
            self.cc.free(self.mtp_zero)?;
            self.mtp_zero = 0;
        }
        let nw = self.cc.alloc(5 * h * 4)?;
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
        let nb =
            unsafe { std::slice::from_raw_parts(norms.as_ptr() as *const u8, norms.len() * 4) };
        if let Err(e) = Self::h2d_chunked(&self.cc, nw, nb) {
            let _ = self.cc.free(nw);
            return Err(e);
        }
        // [정정 2026-10-12] zero ab는 **tmax행**이어야 한다 — norm_resid는
        // ab[e](e=행×hidden+…)로 t행 전체를 읽는다. h(1행)짜리면 t≥2에서
        // OOB 판독(행1+ = 0 또는 NaN) → 프리필 노름 오염 → K/V NaN → 이후
        // 전 어텐션 NaN(초안 0%의 실제 뿌리 — t=1 디코드만 무사했던 이유).
        let zero = self.cc.alloc(tmax * h * 4)?;
        Self::zero_dev(&self.cc, zero, tmax * h * 4)?;
        let buf = self.cc.alloc(total * 4)?;
        self.mtp_nw = nw;
        self.mtp_zero = zero;
        self.mtp_buf = buf;
        self.mtp_off = [
            o_emb, o_hid, o_cat, o_fc, o_res, o_x, o_o, o_qg, o_kin, o_vin, o_g, o_u, o_act, o_dn,
            o_lg, o_xn_e, o_xn_h,
        ];
        self.mtp_prev_off = o_prev;
        self.mtp_layer = dm.n_attn - 1; // 가상 층(서버가 +1로 등록)
        self.mtp_rows = 0;
        self.mtp_ffn = ffn;
        self.mtp_draft = vec![Vec::new(); self.n_slots];
        self.mtp_pos = vec![u32::MAX; self.n_slots];
        self.mtp_prev_pos = vec![u32::MAX; self.n_slots];
        self.mtp_off_slot = vec![false; self.n_slots];
        self.mtp_ok = true;
        eprintln!(
            "[mtp] 등록 — q_out {q_out}·kv {kv_dim}·ffn {ffn}·아레나 {:.1}MiB·층 {}",
            (total * 4) as f64 / (1 << 20) as f64,
            self.mtp_layer
        );
        {
            let dm = self.attn.ok_or("mtp: attn")?;
            let arena = (buf, buf + total as u64 * 4);
            let nwr = (nw, nw + 5 * h as u64 * 4);
            let zr = (zero, zero + h as u64 * 4);
            let ksc0 = self.attn_ksc_ptr(0);
            let kv0 = self.attn_kv_ptr(0);
            let ksce = ksc0 + (dm.n_attn * dm.cap * dm.kv_heads * 4) as u64;
            let kve = kv0 + (dm.n_attn * dm.cap * dm.kv_dim()) as u64;
            let dq = self.dqh_a;
            let dqe = dq + (dm.q_dim() * 8) as u64 * 4;
            eprintln!(
                "[mtp-ptr] arena={arena:?} nw={nwr:?} zero={zr:?} ksc={ksc0}..{ksce} kv={kv0}..{kve} dqh={dq}..{dqe} ksc_in_arena={} nw_in_ksc={}",
                arena.0 <= ksc0 && ksc0 < arena.1,
                ksc0 <= nwr.0 && nwr.0 < ksce
            );
        }
        Ok(())
    }

    fn mtp_ptr(&self, idx: usize) -> CUdeviceptr {
        self.mtp_buf + (self.mtp_off[idx] as u64) * 4
    }

    /// 체인 최종 잔차(dres, pre-norm)의 `row`행을 MTP hidden 어셈블리에 추가.
    /// [정정2 2026-10-12] vLLM 원문(speculator.py 298: "logits_hidden,
    /// feedback_hidden for final-norm correctness") — MTP 입력 = **pre-norm
    /// 잔차(feedback_hidden)**. 타깃의 최종 norm 출력이 아니다(그 오판이
    /// 초안 퇴화의 원인이었다 — MTP가 pre_fc_norm_hidden으로 다시 정규화).
    /// 호출 순서: 검증 직후 수용 접두 행들 → 보정 디코드 직후 1행.
    pub fn mtp_push_hidden(&mut self, row: usize) -> Result<(), String> {
        if !self.mtp_ok {
            return Err("mtp: 미등록".into());
        }
        if self.mtp_rows >= MTP_TMAX {
            return Err("mtp: hidden 행 상한".into());
        }
        let h = self.hidden;
        let src = self.dres + (row * h) as u64 * 4;
        let dst = self.mtp_ptr(1) + (self.mtp_rows * h) as u64 * 4;
        self.copy_dev(dst, src, (h * 4) as u64)?;
        self.mtp_rows += 1;
        Ok(())
    }

    /// 어셈블리 리셋(라운드 시작).
    pub fn mtp_reset_hidden(&mut self) {
        self.mtp_rows = 0;
    }

    /// [진단] 슬롯별 초안 보관(다음 스펙 라운드가 소비).
    pub fn mtp_store_drafts(&mut self, slot: usize, toks: &[u32]) {
        if slot < self.mtp_draft.len() {
            self.mtp_draft[slot] = toks.to_vec();
            // 초안열 i번째 = **slot_pos+1+i 위치**의 토큰(페어링 (P,P+1)→P+2,
            // P = slot_pos-1). 소비 검증도 같은 식.
            self.mtp_pos[slot] = self.slot_pos[slot] + 1;
        }
    }

    /// [MTP-2] 재귀 1스텝 — 직전 런의 fin(ptr5, t=1이므로 행 0)을 hidden으로
    /// 옮기고 (토큰, pos)로 t=1 런. 반환 = 다음 초안.
    /// KV는 이 위치에 MTP 자신의 k/v가 기록된다(검증 수용분은 이후
    /// post_round가 같은 위치를 올바른 토큰으로 재기록 — 자기 치유).
    pub fn mtp_recur_step(
        &mut self,
        slot: usize,
        tok: u32,
        embed: &[f32],
        pos: u32,
    ) -> Result<u32, String> {
        let h = self.hidden;
        if embed.len() != h {
            return Err("mtp: 재귀 임베딩 폭".into());
        }
        // 직전 fin(ptr5 행 0) → hidden(ptr1 행 0), rows=1로 고정.
        self.copy_dev(self.mtp_ptr(1), self.mtp_ptr(5), (h * 4) as u64)?;
        self.mtp_rows = 1;
        self.mtp_run_h(slot, &[tok], embed, pos, true)
    }

    /// [MTP-2] 미래 잔존 행 위생 — pos+1 행(lim 슬랙이 읽는 한 행)의
    /// k/v/스케일을 0으로(이전 라운드 재귀 잔존분이 lim=pos+t+1 판독에
    /// 섞이는 것 차단). 초안 정확도만 영향(검증이 정본).
    pub fn mtp_zero_row(&mut self, pos: usize) -> Result<(), String> {
        if !self.mtp_ok {
            return Ok(());
        }
        let dm = self.attn.ok_or("mtp: attn")?;
        if pos >= dm.cap {
            return Ok(());
        }
        let lay = self.mtp_layer;
        for slot in 0..self.n_slots {
            let rowoff = (lay * dm.cap + pos) as u64;
            let kvc = rowoff * dm.kv_dim() as u64;
            let scc = rowoff * dm.kv_heads as u64 * 4;
            Self::zero_dev(&self.cc, self.attn_kv_ptr(slot) + kvc, dm.kv_dim())?;
            Self::zero_dev(&self.cc, self.attn_vc_ptr(slot) + kvc, dm.kv_dim())?;
            Self::zero_dev(&self.cc, self.attn_ksc_ptr(slot) + scc, dm.kv_heads * 4)?;
            Self::zero_dev(&self.cc, self.attn_vsc_ptr(slot) + scc, dm.kv_heads * 4)?;
        }
        Ok(())
    }

    /// 초안열 소비 — 유효(현 위치)일 때만 반환.
    pub fn mtp_take_draft(&mut self, slot: usize) -> Option<Vec<u32>> {
        if slot >= self.mtp_draft.len() {
            return None;
        }
        if self.mtp_pos[slot] != self.slot_pos[slot] + 1 {
            return None; // 스테일(위치 불일치)
        }
        if self.mtp_draft[slot].is_empty() {
            return None;
        }
        Some(self.mtp_draft[slot].clone())
    }

    /// MTP 순전파(t행) — tokens[i]의 임베딩 + 어셈블된 hidden 행들로
    /// pos_start..pos_start+t 위치를 처리하고, 마지막 행의 argmax(다음+1
    /// 초안)를 반환한다. embeds = [t][hidden] f32(호스트 임베딩 행).
    pub fn mtp_run(
        &mut self,
        slot: usize,
        tokens: &[u32],
        embeds: &[f32],
        pos_start: u32,
    ) -> Result<u32, String> {
        self.mtp_run_h(slot, tokens, embeds, pos_start, true)
    }

    /// head=false면 로짓·argmax 생략(프리필 서브청크 — KV 구축 전용, 0 반환).
    pub fn mtp_run_h(
        &mut self,
        slot: usize,
        tokens: &[u32],
        embeds: &[f32],
        pos_start: u32,
        head: bool,
    ) -> Result<u32, String> {
        let _g = self.cc.guard()?;
        if !self.mtp_ok {
            return Err("mtp: 미등록".into());
        }
        let h = self.hidden;
        let t = tokens.len();
        if !(1..=MTP_TMAX).contains(&t) || self.mtp_rows != t || embeds.len() != t * h {
            return Err(format!(
                "mtp: t={t} rows={} embeds={} 계약 위반",
                self.mtp_rows,
                embeds.len()
            ));
        }
        let pos_end = pos_start as usize + t;
        let cap = self.attn.ok_or("mtp: attn")?.cap;
        if pos_end > cap {
            return Err(format!("mtp: pos{pos_start}+{t} > cap{cap}"));
        }
        self.ensure_chain_bufs()?;
        self.ensure_attn_bufs(t)?;
        self.ensure_batch_bufs()?; // dbatch_am(argmax d2h 목적지)
        // [MTP-GRAPH 2026-10-12] 핀드 스테이징(캡처/직접 공용 — pageable 제거).
        if self.mtp_pin_emb.is_null() {
            self.mtp_pin_emb = self.cc.pinned_alloc(MTP_TMAX * h * 4)?;
            self.mtp_pin_out = self.cc.pinned_alloc(8 * 4)?;
        }
        // SAFETY: 호출자 슬라이스(t*h f32) 옥 → 핀드(MTP_TMAX*h*4) 기입.
        unsafe {
            std::ptr::copy_nonoverlapping(
                embeds.as_ptr() as *const u8,
                self.mtp_pin_emb as *mut u8,
                t * h * 4,
            );
        }
        // 그래프 경로 — head=true·t≤8·캡처 성공 시 런치 1회(호스트 일감은
        // 핀 기입 + dpp 1회 + sync + argmax 4B 회수뿐).
        if head && t <= 8 && !self.mtp_graph_failed {
            match self.ensure_mtp_graph(slot, t) {
                Ok(exec) => {
                    let save = self.slot_pos[slot];
                    self.slot_pos[slot] = pos_start;
                    self.attn_set_pos(slot, pos_start)?;
                    self.cc.graph_launch(exec)?;
                    self.cc.sync()?;
                    self.slot_pos[slot] = save;
                    // SAFETY: pinned argmax(t*4 기입).
                    let ob =
                        unsafe { std::slice::from_raw_parts(self.mtp_pin_out as *const u8, t * 4) };
                    let mut b = [0u8; 4];
                    b.copy_from_slice(&ob[(t - 1) * 4..t * 4]);
                    return Ok(u32::from_le_bytes(b));
                }
                Err(e) => {
                    eprintln!("[mtp-graph] 캡처 실패 — 직접 경로: {e}");
                    self.mtp_graph_failed = true;
                }
            }
        }
        // 직접 경로(프리필 head=false · t>8 · 캡처 실패).
        let save = self.slot_pos[slot];
        self.slot_pos[slot] = pos_start;
        let r = self.mtp_fwd_launch(slot, t, head);
        self.slot_pos[slot] = save;
        r?;
        if !head {
            return Ok(0);
        }
        self.cc.sync()?;
        // SAFETY: pinned argmax 회수(t≤8 = t행, t>8 = 마지막 1행 4B).
        let n = if t > 8 { 4 } else { t * 4 };
        let ob = unsafe { std::slice::from_raw_parts(self.mtp_pin_out as *const u8, n) };
        let mut b = [0u8; 4];
        b.copy_from_slice(&ob[n - 4..n]);
        Ok(u32::from_le_bytes(b))
    }

    /// [MTP-GRAPH] t별 캡처 확보 — 실패 시 호출부가 직접 경로로 폴백.
    fn ensure_mtp_graph(&mut self, slot: usize, t: usize) -> Result<ffi::CUgraphExec, String> {
        if let Some((e, _)) = self.mtp_graphs[t] {
            return Ok(e);
        }
        self.ensure_chain_bufs()?;
        self.ensure_attn_bufs(t)?;
        self.ensure_batch_bufs()?;
        // fwd3s merge의 dx32/dx16 융합 기록 보장(캡처 중 alloc 금지 계약).
        {
            let dm = self.attn.ok_or("mtp: attn")?;
            let nm = t * dm.q_heads * 256;
            self.ensure_dx32(nm)?;
            self.ensure_dx16(nm)?;
        }
        self.cc.capture_begin()?;
        self.capture_pinned_src = true;
        let cap = self.mtp_fwd_launch(slot, t, true);
        self.capture_pinned_src = false;
        if let Err(e) = cap {
            let _ = self.cc.capture_end();
            return Err(format!("mtp 캡처 본문: {e}"));
        }
        let g = self.cc.capture_end()?;
        let e = self.cc.graph_instantiate(g)?;
        self.cc.graph_upload(e)?;
        self.mtp_graphs[t] = Some((e, g));
        eprintln!("[mtp-graph] captured t={t}");
        Ok(e)
    }

    /// [MTP] 순전파 발사부 — 캡처/직접 공용. 임베드 h2d는 pinned 소스에서,
    /// argmax d2h는 pinned 목적지로(회수·sync는 호출부). pos는 호출부가
    /// slot_pos/dpp로 확정한다(커널 인자 아님 — 캡처 안전).
    fn mtp_fwd_launch(&mut self, slot: usize, t: usize, head: bool) -> Result<(), String> {
        let h = self.hidden;
        // 1) 임베딩 h2d ← pinned(호출부가 t*h*4 기입).
        // SAFETY: pinned 버퍼는 MTP_TMAX*h*4 계약.
        let eb = unsafe { std::slice::from_raw_parts(self.mtp_pin_emb as *const u8, t * h * 4) };
        self.cc.h2d_async(self.mtp_ptr(0), eb)?;
        // 2) 두 노름 — embed→xn_e, hidden→xn_h (x 제자리 +0, ab=zero).
        self.mtp_norm(1, self.mtp_ptr(0), self.mtp_ptr(15), t)?;
        self.mtp_norm(0, self.mtp_ptr(1), self.mtp_ptr(16), t)?;
        // 3) concat [embed_norm, hidden_norm] — 행별 2회 d2d(vLLM cat 순서).
        for i in 0..t {
            let cat = self.mtp_ptr(2) + (i * 2 * h) as u64 * 4;
            let xe = self.mtp_ptr(15) + (i * h) as u64 * 4;
            let xh = self.mtp_ptr(16) + (i * h) as u64 * 4;
            self.copy_dev(cat, xe, (h * 4) as u64)?;
            self.copy_dev(cat + (h * 4) as u64, xh, (h * 4) as u64)?;
        }
        // 4) fc: [t][2h] → [t][h]
        self.mtp_plain_gemv_t("mtp.fc.weight", self.mtp_ptr(2), self.mtp_ptr(3), t)?;
        // 5) 잔차 시작: res=0 → (res += fc) 노름(row2) → x
        Self::zero_dev(&self.cc, self.mtp_ptr(4), t * h * 4)?;
        self.mtp_norm_acc(2, self.mtp_ptr(4), self.mtp_ptr(3), self.mtp_ptr(5), t)?;
        // 6) q/k/v — x → qg/kin/vin
        self.mtp_plain_gemv_t(
            "mtp.layers.0.self_attn.q_proj.weight",
            self.mtp_ptr(5),
            self.mtp_ptr(7),
            t,
        )?;
        self.mtp_plain_gemv_t(
            "mtp.layers.0.self_attn.k_proj.weight",
            self.mtp_ptr(5),
            self.mtp_ptr(8),
            t,
        )?;
        self.mtp_plain_gemv_t(
            "mtp.layers.0.self_attn.v_proj.weight",
            self.mtp_ptr(5),
            self.mtp_ptr(9),
            t,
        )?;
        // 7) 어텐션(가상 층) — pos는 호출부가 slot_pos/dpp로 확정.
        let dout = self.attn_chain_dev_run(
            slot,
            self.mtp_layer,
            t,
            self.mtp_ptr(7),
            self.mtp_ptr(8),
            self.mtp_ptr(9),
        )?;
        if !head {
            return Ok(());
        }
        // 8) o_proj — 어텐션 출력(doutv [t][q_dim]) → ptr6.
        self.mtp_plain_gemv_t(
            "mtp.layers.0.self_attn.o_proj.weight",
            dout,
            self.mtp_ptr(6),
            t,
        )?;
        // 9) 잔차 += o_proj → post_attention_layernorm(행3) → x(ptr5).
        self.mtp_norm_acc(3, self.mtp_ptr(4), self.mtp_ptr(6), self.mtp_ptr(5), t)?;
        // 10) MLP — gate/up(ptr5) → silu·mul(ptr12) → down(ptr13).
        self.mtp_plain_gemv_t(
            "mtp.layers.0.mlp.gate_proj.weight",
            self.mtp_ptr(5),
            self.mtp_ptr(10),
            t,
        )?;
        self.mtp_plain_gemv_t(
            "mtp.layers.0.mlp.up_proj.weight",
            self.mtp_ptr(5),
            self.mtp_ptr(11),
            t,
        )?;
        self.ew_dev(
            self.mtp_ptr(10),
            self.mtp_ptr(11),
            self.mtp_ptr(12),
            t * self.mtp_ffn,
        )?;
        self.mtp_plain_gemv_t(
            "mtp.layers.0.mlp.down_proj.weight",
            self.mtp_ptr(12),
            self.mtp_ptr(13),
            t,
        )?;
        // 11) 잔차 += down → mtp.norm(행4) → 최종 xn(ptr5) = head 입력.
        // [정정 2026-10-12] 종전엔 꼬리(8~11)가 미구현이라 head가 어텐션 前
        // x에 적용됐다 — 초안 전량 오답(수용 0/62)의 실제 원인.
        self.mtp_norm_acc(4, self.mtp_ptr(4), self.mtp_ptr(13), self.mtp_ptr(5), t)?;
        // 12) 공유 head + argmax(마지막 행) → d2h.
        // t>8(프리필 서브청크)은 마지막 행만 t=1 head — 로짓 버퍼 절약.
        if t > 8 {
            let xl = self.mtp_ptr(5) + ((t - 1) * h) as u64 * 4;
            self.head_gemv_launch(xl, self.mtp_ptr(14))?;
            let fa = self.cc.function("w4a16_argmax_min")?;
            let (mut p_l, mut p_nn, mut p_a) =
                (self.mtp_ptr(14), self.head_n as i32, self.dbatch_am);
            self.cc.launch(
                fa,
                1,
                1,
                1024,
                &mut crate::rawcuda::args::l3(&mut p_l, &mut p_nn, &mut p_a),
            )?;
            // SAFETY: pinned 목적지(8*4 계약, 4B 기입).
            self.cc
                .d2h_async(self.mtp_pin_out as *mut u8, self.dbatch_am, 4)?;
            return Ok(());
        }
        self.head_gemv_t_launch(self.mtp_ptr(5), self.mtp_ptr(14), t)?;
        {
            let fa = self.cc.function("w4a16_argmax_min_t")?;
            let (mut p_l, mut p_n, mut p_t, mut p_a) = (
                self.mtp_ptr(14),
                self.head_n as i32,
                t as i32,
                self.dbatch_am,
            );
            self.cc.launch(
                fa,
                t as u32,
                1,
                1024,
                &mut crate::rawcuda::args::l4(&mut p_l, &mut p_n, &mut p_t, &mut p_a),
            )?;
        }
        // SAFETY: pinned 목적지(8*4 계약, t*4 기입).
        self.cc
            .d2h_async(self.mtp_pin_out as *mut u8, self.dbatch_am, t * 4)?;
        Ok(())
    }

    /// [MTP 프리필] 호출열 = (P의 hidden, P+1의 토큰) — 토큰은 **시프트된
    /// 열**(엔진이 조립). hidden은 체인 최종 norm(dxn) 행 0..calls-1을 푸시.
    /// save_last_row=Some(r)이면 그 행을 prev로 저장(마지막 P 유예 — 첫
    /// 샘플 후 mtp_run_prev가 소비). 로짓·초안은 만들지 않는다(프리필 중
    /// 초안 무의미).
    pub fn mtp_prefill_chunk(
        &mut self,
        slot: usize,
        call_tokens: &[u32],
        embeds: &[f32],
        pos_start: u32,
        save_last_row: Option<usize>,
    ) -> Result<(), String> {
        let h = self.hidden;
        if !self.mtp_ok {
            return Err("mtp: 미등록".into());
        }
        if embeds.len() != call_tokens.len() * h {
            return Err("mtp: embeds 계약".into());
        }
        let mut r = 0usize;
        while r < call_tokens.len() {
            let t = (call_tokens.len() - r).min(MTP_PF_CHUNK);
            self.mtp_reset_hidden();
            for k in 0..t {
                self.mtp_push_hidden(r + k)?;
            }
            let eb = &embeds[r * h..(r + t) * h];
            let ps = pos_start + r as u32;
            self.mtp_run_h(slot, &call_tokens[r..r + t], eb, ps, false)?;
            r += t;
        }
        if let Some(row) = save_last_row {
            self.mtp_save_prev(slot, row)?;
        }
        Ok(())
    }

    /// 프리필 마지막 hidden 저장(유예) — 첫 샘플 후 mtp_run_prev가 소비.
    pub fn mtp_save_prev(&mut self, slot: usize, row: usize) -> Result<(), String> {
        if !self.mtp_ok || slot >= self.n_slots {
            return Err("mtp: 미등록/슬롯".into());
        }
        let h = self.hidden;
        let src = self.dres + (row * h) as u64 * 4;
        let dst = self.mtp_buf + (self.mtp_prev_off + slot * h) as u64 * 4;
        self.copy_dev(dst, src, (h * 4) as u64)?;
        self.mtp_prev_pos[slot] = self.slot_pos_of(slot).saturating_sub(1);
        Ok(())
    }

    /// 유예 호출 — (prev hidden, tok) t=1 → 초안. 유예 없으면 Err.
    pub fn mtp_run_prev(&mut self, slot: usize, tok: u32, embed: &[f32]) -> Result<u32, String> {
        if self.mtp_prev_pos[slot] == u32::MAX {
            return Err("mtp: 유예 호출 없음".into());
        }
        let h = self.hidden;
        self.mtp_reset_hidden();
        let src = self.mtp_buf + (self.mtp_prev_off + slot * h) as u64 * 4;
        self.copy_dev(self.mtp_ptr(1), src, (h * 4) as u64)?;
        self.mtp_rows = 1;
        let pos = self.mtp_prev_pos[slot];
        let out = self.mtp_run_h(slot, &[tok], embed, pos, true)?;
        self.mtp_prev_pos[slot] = u32::MAX;
        Ok(out)
    }

    /// [MTP] KV 리셋 — 상태 리셋(웜업 종료·슬롯 재사용) 시 잔존 행 제거.
    /// 본체 KV는 reset_state가 지우지만 MTP 층 슬라이스는 별도라 방치되면
    /// t=1 커널의 lim=pos+2 판독이 **웜업 잔존 행**을 읽는다(실측 결함 의심).
    pub fn mtp_kv_reset(&mut self) -> Result<(), String> {
        if !self.mtp_ok {
            return Ok(());
        }
        let dm = self.attn.ok_or("mtp: attn")?;
        let layer = self.mtp_layer;
        for slot in 0..self.n_slots {
            let off = (layer * dm.cap) as u64;
            let kvc = off * dm.kv_dim() as u64;
            let scc = off * dm.kv_heads as u64 * 4;
            Self::zero_dev(&self.cc, self.attn_kv_ptr(slot) + kvc, dm.cap * dm.kv_dim())?;
            Self::zero_dev(&self.cc, self.attn_vc_ptr(slot) + kvc, dm.cap * dm.kv_dim())?;
            Self::zero_dev(
                &self.cc,
                self.attn_ksc_ptr(slot) + scc,
                dm.cap * dm.kv_heads * 4,
            )?;
            Self::zero_dev(
                &self.cc,
                self.attn_vsc_ptr(slot) + scc,
                dm.cap * dm.kv_heads * 4,
            )?;
        }
        Ok(())
    }

    /// 슬롯 MTP 비활성/재개(접두 재사용 슬롯 = KV 정합 없음 — 초안 금지).
    pub fn mtp_slot_off(&mut self, slot: usize) {
        if slot < self.mtp_off_slot.len() {
            self.mtp_off_slot[slot] = true;
            self.mtp_prev_pos[slot] = u32::MAX;
        }
    }

    pub fn mtp_slot_on(&mut self, slot: usize) {
        if slot < self.mtp_off_slot.len() {
            self.mtp_off_slot[slot] = false;
        }
    }

    pub fn mtp_slot_enabled(&self, slot: usize) -> bool {
        self.mtp_ok && slot < self.mtp_off_slot.len() && !self.mtp_off_slot[slot]
    }

    /// MTP 노름 — x는 제자리(+0), xn은 별도 기록(xn32/xn16 없음).
    fn mtp_norm(
        &mut self,
        row: usize,
        x: CUdeviceptr,
        xn: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        let ab = self.mtp_zero;
        self.mtp_norm_acc(row, x, ab, xn, t)
    }

    /// MTP 노름(누적) — res(x) += ab 후 norm → xn.
    fn mtp_norm_acc(
        &mut self,
        row: usize,
        x: CUdeviceptr,
        ab: CUdeviceptr,
        xn: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        let f = self.cc.function("norm_resid")?;
        // 주의: 커널은 nw[w_off + i]로 읽는다 — 포인터는 **베이스**,
        // 행 선택은 w_off로만(이중 오프셋 = 범위 밖 판독 — 실측 결함).
        let (mut p_x, mut p_w) = (x, self.mtp_nw);
        let mut p_ab = ab;
        let mut p_xn = xn;
        let (mut p_z1, mut p_z2) = (0u64, 0u64);
        let (mut p_t, mut p_wo, mut p_h) =
            (t as i32, (row * self.hidden) as i32, self.hidden as i32);
        self.cc.launch(
            f,
            t as u32,
            1,
            1024,
            &mut crate::rawcuda::args::l9(
                &mut p_x, &mut p_w, &mut p_ab, &mut p_xn, &mut p_z1, &mut p_z2, &mut p_t,
                &mut p_wo, &mut p_h,
            ),
        )
    }

    /// MTP 선형 — t=1 plain_gemv(행=블록) · t≤8 t행 GEMV(비트동일 계열) ·
    /// t≥9 plain_gemm(mma/bf16 GEMM — gemv_t 커널은 t≤8 전용).
    fn mtp_plain_gemv_t(
        &mut self,
        name: &str,
        x: CUdeviceptr,
        y: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        if t == 1 {
            self.plain_gemv_launch(name, x, y)
        } else if t <= 8 {
            self.plain_gemv_t_launch(name, x, y, t)
        } else {
            self.plain_gemm_launch(name, x, y, t)
        }
    }
}
