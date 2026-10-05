//! qwen4exp 시퀀스 상태·체크포인트 (plans/129 R5 — layers.rs 순수 이동).
//! SeqState4(값경로 CPU 상태)·SeqCkpt(부분 접두 체크포인트 예산)와 그 Engine4
//! 메서드군. 상태 수명 규칙(첫 GPU 디코드 시 Frame4 전환·resync)은 여기가 진실.
use super::super::Hparams4;
use super::Engine4;

#[derive(Clone)]
pub struct SeqState4 {
    pub pos: u32,
    /// GDN층 S 상태 [dt_rank×d_state×d_state] (순환층 순서)
    pub gdn_s: Vec<Vec<f32>>,
    /// GDN depthwise conv 링 [(conv_k-1)×conv_ch]
    pub conv: Vec<Vec<f32>>,
    /// QSA층 KV [ctx][n_kv×hd] ×2 (full_idx 순서)
    pub kv_k: Vec<Vec<f32>>,
    pub kv_v: Vec<Vec<f32>>,
    /// QSA 인덱서 raw k 캐시 [ctx][idx_dim]
    pub idx_k: Vec<Vec<f32>>,
    /// QSA 인덱서 블록 키 캐시 [QSA층][n_blocks·idx_dim] — pooled+norm+rope
    /// 된 블록 키를 1회 계산해 전 토큰 재사용 (O(T²)→O(T), 2026-09-01).
    pub idx_bk: Vec<Vec<f32>>,
    /// PLE dilated conv 히스토리 [(kern-1)*dil][hc_dim]
    pub ple_conv: Vec<f32>,
    /// PLE n-gram 직전 토큰 히스토리 (최대 ngram-1개, 오래된 것이 앞)
    pub ple_hist: Vec<u32>,
    pub ple_next_pos: u32,
    /// plans/73: 디코드가 QSA 선택을 디바이스에서 수행해 호스트 kv/idx 캐시
    /// 갱신을 건너뛰었음. 프리필(t>1) 진입 시 풀에서 1회 재구축한다.
    pub qsa_host_stale: bool,
}

impl SeqState4 {
    pub fn new(hp: &Hparams4, ctx: usize) -> Self {
        let n_recr = (0..hp.n_layer).filter(|&il| hp.is_recr(il)).count();
        let n_full = hp.n_layer - n_recr;
        let state_size = hp.dt_rank * hp.d_state * hp.d_state;
        let conv_len = (hp.conv_k - 1) * (hp.n_group * hp.d_state * 2 + hp.dt_rank * hp.d_state);
        let ple_hist_len = (hp.ple_conv_k - 1) * hp.ple_ngram;
        let has_ple = hp.is_ple(1);
        SeqState4 {
            pos: 0,
            gdn_s: vec![vec![0.0; state_size]; n_recr],
            conv: vec![vec![0.0; conv_len]; n_recr],
            kv_k: vec![vec![0.0; ctx * hp.n_kv * hp.head_dim]; n_full],
            kv_v: vec![vec![0.0; ctx * hp.n_kv * hp.head_dim]; n_full],
            idx_k: vec![vec![0.0; ctx * hp.idx_dim]; n_full],
            idx_bk: vec![Vec::new(); n_full],
            ple_conv: vec![
                0.0;
                if has_ple {
                    ple_hist_len * hp.hc * hp.n_embd
                } else {
                    0
                }
            ],
            ple_hist: Vec::new(),
            ple_next_pos: 0,
            qsa_host_stale: false,
        }
    }
}

impl SeqState4 {
    /// MTP 드래프트 전용 상태 (plans/109 P15②) — KV/idx 슬롯을 풀어텐션층
    /// +1(블록 n_layer)로 잡는다. GDN/PLE 슬롯은 미사용(0 유지).
    pub fn new_mtp(hp: &Hparams4, ctx: usize) -> Self {
        let mut st = SeqState4::new(hp, ctx);
        let n_full_mtp = st.kv_k.len() + 1;
        st.kv_k = vec![vec![0.0; ctx * hp.n_kv * hp.head_dim]; n_full_mtp];
        st.kv_v = vec![vec![0.0; ctx * hp.n_kv * hp.head_dim]; n_full_mtp];
        st.idx_k = vec![vec![0.0; ctx * hp.idx_dim]; n_full_mtp];
        st.idx_bk = vec![Vec::new(); n_full_mtp];
        st
    }
}

/// plans/115 P1-3: 접두 체크포인트 — 순환 상태 스냅샷.
/// kv_k/kv_v/idx_k는 pos 인덱스 쓰기(되감기 재프리필이 동일 행을 다시 쓴다
/// — 디바이스 풀 워터마크 계약, common/qsa.rs wm_advance)라 제외했다.
/// idx_bk는 블록 완결 길이만 저장(복원 시 truncate → 재프리필이 이어 재계산).
pub struct SeqCkpt {
    pub pos: u32,
    /// GDN/conv 스냅샷의 디바이스 링 슬롯(frame.ckpt_dev[seq] 인덱스).
    pub dev: usize,
    pub ple_conv: Vec<f32>,
    pub ple_hist: Vec<u32>,
    pub ple_next_pos: u32,
    /// [n_full] idx_bk.len() — 복원 truncate 기준
    pub idx_bk_lens: Vec<usize>,
    pub qsa_host_stale: bool,
}

impl Engine4 {
    /// plans/115 P1-3: 접두 체크포인트 캡처 — 프리필 청크 경계 호출.
    /// 간격 게이트(≥512토큰): 배치 프리필은 per=16..256 청크로 잘리므로 매
    /// 청크 클론(≈112MB)이면 TTFT를 갉아먹는다. 프레임 경로는 CPU gdn_s/conv
    /// 을 갱신하지 않으므로 캡처 직전 풀백(D2H)이 필수다(dirty면 CPU 권위 —
    /// 풀백 불요). 실패 시 캡처 생략(부분 재사용만 늦춰질 뿐 정확성 무영향).
    pub fn ckpt_capture(&mut self, seq: usize, pos: usize) {
        const MIN_SPACING: usize = 512;
        if pos < MIN_SPACING {
            return;
        }
        if let Some(c) = self.ckpt[seq].back() {
            let last = c.pos as usize;
            if pos <= last || pos - last < MIN_SPACING {
                return; // 후퇴/중복/과밀 캡처 무시
            }
        }
        // GDN/conv은 디바이스 D2D 스냅샷(스트림 순서 — 청크 직후 상태 그대로).
        // 프레임 없음(CPU 서빙)은 체크포인트 없음 — 부분 재사용 불가(폴백 reset).
        let t0 = std::time::Instant::now();
        let Some(acc) = self.acc.clone() else { return };
        let Some(f) = self.frame.as_mut() else { return };
        // 링 슬롯 확보 — free-list 우선, 부족하면 신규 할당, 상한이면 최旧
        // 체크포인트를 퇴출해 슬롯을 물려받는다.
        let slot = if let Some(g) = f.ckpt_free[seq].pop() {
            g
        } else if f.ckpt_dev[seq].len() < self.ckpt_keep {
            let g = f.ckpt_dev[seq].len();
            let mut gs = Vec::with_capacity(f.st_gdn[seq].len());
            let mut cs = Vec::with_capacity(f.st_conv[seq].len());
            for _ in 0..f.st_gdn[seq].len() {
                match acc.frame_alloc(f.gdn_state_len) {
                    Ok(h) => gs.push(h),
                    Err(e) => {
                        eprintln!("# ckpt: gdn 버퍼 할당 실패 — 캡처 중단({e})");
                        return;
                    }
                }
            }
            for _ in 0..f.st_conv[seq].len() {
                match acc.frame_alloc(f.conv_state_len) {
                    Ok(h) => cs.push(h),
                    Err(e) => {
                        eprintln!("# ckpt: conv 버퍼 할당 실패 — 캡처 중단({e})");
                        return;
                    }
                }
            }
            f.ckpt_dev[seq].push((gs, cs));
            g
        } else if let Some(old) = self.ckpt[seq].pop_front() {
            old.dev
        } else {
            return;
        };
        let (gd, cv) = f.ckpt_dev[seq][slot].clone();
        let mut pairs: Vec<(u64, u64, usize)> = Vec::with_capacity(gd.len() + cv.len());
        for (d, &src) in gd.iter().zip(f.st_gdn[seq].iter()) {
            pairs.push((*d, src, f.gdn_state_len * 4));
        }
        for (d, &src) in cv.iter().zip(f.st_conv[seq].iter()) {
            pairs.push((*d, src, f.conv_state_len * 4));
        }
        if let Err(e) = acc.frame_copy_states(&pairs) {
            eprintln!("# ckpt: D2D 캡처 실패 — 생략({e})");
            f.ckpt_free[seq].push(slot);
            return;
        }
        let st = &self.seqs[seq];
        self.ckpt[seq].push_back(SeqCkpt {
            pos: pos as u32,
            dev: slot,
            ple_conv: st.ple_conv.clone(),
            ple_hist: st.ple_hist.clone(),
            ple_next_pos: st.ple_next_pos,
            idx_bk_lens: st.idx_bk.iter().map(|v| v.len()).collect(),
            qsa_host_stale: st.qsa_host_stale,
        });
        if llm170_diag::dump::opts().key("ckpt_time") {
            eprintln!(
                "# ckpt capture seq{seq} pos{pos}: {:.1}ms",
                t0.elapsed().as_secs_f64() * 1e3
            );
        }
    }

    /// plans/115 P1-3: l 이하 최신 체크포인트로 되감기 — 복원 위치 반환(없으면
    /// None, 상태 무변). kv/idx_k 호스트 행은 접두 불변(그대로), idx_bk는 블록
    /// 길이까지 truncate — 재프리필 [cp..)이 pos 인덱스로 동일 행을 다시 쓴다.
    /// GDN은 dirty=true → 다음 프리필 sync_states가 CPU 클론을 디바이스로 전사.
    /// PLE 링 캐시 무효(acc_reset_seq — pos 기반 되감기 계약).
    pub fn ckpt_restore_upto(&mut self, seq: usize, l: usize) -> Option<usize> {
        let idx = self.ckpt[seq].iter().rposition(|c| (c.pos as usize) <= l)?;
        let ck = self.ckpt[seq].remove(idx).unwrap();
        // 복원 시점 이후 체크포인트는 무효(이후 토큰열이 갈린다) — 슬롯 회수.
        for dropped in self.ckpt[seq].drain(idx..) {
            if let Some(f) = self.frame.as_mut() {
                f.ckpt_free[seq].push(dropped.dev);
            }
        }
        let acc = self.acc.clone()?;
        let f = self.frame.as_mut()?;
        // GDN/conv D2D 되감기 — 디바이스가 곧 권위다(dirty=false 유지).
        // CPU gdn_s는 stale가 되는데 이는 통상 디코드 체제와 동일 — 값경로
        // 진입 시 frame_pullback_cpu가 디바이스에서 지연 갱신한다.
        let (gd, cv) = f.ckpt_dev[seq][ck.dev].clone();
        let mut pairs: Vec<(u64, u64, usize)> = Vec::with_capacity(gd.len() + cv.len());
        for (&d, &src) in gd.iter().zip(f.st_gdn[seq].iter()) {
            pairs.push((src, d, f.gdn_state_len * 4));
        }
        for (&d, &src) in cv.iter().zip(f.st_conv[seq].iter()) {
            pairs.push((src, d, f.conv_state_len * 4));
        }
        let pos = ck.pos as usize;
        f.ckpt_free[seq].push(ck.dev);
        if acc.frame_copy_states(&pairs).is_err() {
            return None; // 상태 무변 — 호출부는 reset 폴백
        }
        acc.acc_reset_seq(seq);
        f.dirty[seq] = false; // 디바이스 GDN = 체크포인트 상태(권위)
        f.last_res_hc_rows = Vec::new(); // MTP h 행 — 새 프리필이 재적립
        self.spec_h_prev[seq] = Vec::new();
        self.last_res_hc_rows = Vec::new();
        // 드래프트 KV는 pos 인덱스 쓰기·접두 불변 — [0..cp)가 그대로 유효하므로
        // pos만 되감는다(신규 초기화보다 낫다: 문맥 보존 → 수용률 유지).
        // [cp..]의 재생은 mtp_draft_prefill(base_pos=cp)이 맡는다.
        if self.model.has_mtp() && !self.mtp_seqs.is_empty() {
            self.mtp_seqs[seq].pos = ck.pos;
        }
        let st = &mut self.seqs[seq];
        st.pos = ck.pos;
        st.ple_conv = ck.ple_conv;
        st.ple_hist = ck.ple_hist;
        st.ple_next_pos = ck.ple_next_pos;
        st.qsa_host_stale = ck.qsa_host_stale;
        for (v, &len) in st.idx_bk.iter_mut().zip(ck.idx_bk_lens.iter()) {
            v.truncate(len);
        }
        Some(pos)
    }

    /// 접두 체크포인트 전체 폐기 — reset_seq/reset_states에서 호출(새 대화의
    /// 체크포인트가 이전 토큰열을 역참조하지 않게).
    pub(super) fn ckpt_clear(&mut self, seq: Option<usize>) {
        match seq {
            Some(s) => self.ckpt[s].clear(),
            None => {
                for q in &mut self.ckpt {
                    q.clear();
                }
            }
        }
    }
}
