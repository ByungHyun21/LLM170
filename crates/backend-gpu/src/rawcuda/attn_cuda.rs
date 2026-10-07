//! EXL3 어텐션(prep + fwd3s) 모듈층(plans/124 G6, G10 파일 분할
//! 2026-10-04). Exl3CudaDecoder의 어텐션 임플 블록 + AttnDims/AttnOut
//! 형상·판독 자료형 — 컨텍스트·fatbin 리졸버는 exl3_cuda.rs 공유 글루.
//!
//! [용도] full-attn 층(il%4==3, 27B 16층·35B 10층) 체인: prep(디인터리브 +
//! q/k_norm w−1 저장 규약 + rope base 1e7 부분회전 64차=불32쌍) + fwd3s
//! (T≤8 소형 전용 게이트 어텐션 — 모듈 Err + 커널 조기복귀 이중 강제).
//! KV 캐시·pp 상주. KV 인덱스는 pp[0] 디바이스 판독(결함 4호 — 발사
//! 인자가 아닌 버퍼 판독이 시그니처 수준 계약). 형상은 AttnDims
//! (config.json 유도, 27B q24/kv4·35B q16/kv2). 산술 계약은
//! assets/exl3_attn.cu(src_exl3.hip exl3_attn_prep L607-687·exl3_attn_fwd3s
//! L799-853 1:1 직이식, -fmad=false 빌드).
//!
//! [정합 — plans/129-cuda C2 원장, sm_89 실측 2026-10-04] (i) 27B q24/kv4
//! lay=15 pos0=33 T=1: qh·KC·VC·outv 전 단계 maxdiff 0.000e0(오라클과
//! 비트동일) · (ib) 27B T=8: 0.000e0 · (ii) 35B-A3B q16/kv2 lay=9 T=8:
//! 0.000e0 — 임계 2e-7(전 모듈 최tight) 대비 무한대 여유. (iii) T=9:
//! 모듈 진입 Err(도메인 이중 강제). 음성대조: 호스트 pos 사본 경로 종단
//! 1.498e-1 > 2e-7 → NEG-DETECTED(결함 4호 값 검증력 입증).
//!
//! [속도] 측정 대기 sm_80 — CMP 170HX 미도착(plans/124 §0). sm_80 자원
//! 증거: exl3_attn_prep REG:26 SHARED:1536 → 12블록/SM · exl3_attn_fwd3s
//! REG:34 SHARED:6144 → 6블록/SM(와프 상한).
//!
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 의존 금지 — scripts/cuda_probe_shim.rs 단독 컴파일.

use crate::rawcuda::exl3_cuda::{Exl3CudaDecoder, JParser, JVal};
use crate::rawcuda::ffi::CUdeviceptr;

/// fwd3s T 상한(plans/124 §1 "fwd3s는 T≤8 소형 전용" — 모듈 Err·커널
/// 조기복귀 이중 강제, assets/exl3_attn.cu EXL3_ATTN_TMAX와 동일 값).
pub const ATTN_F3S_TMAX: usize = 8;
/// fwd3s 점수 scratch 공유메모리 행 수(= assets/exl3_attn.cu EXL3_ATTN_SCAP).
/// 위치축 상한을 이 값이 결정한다: pos+1이 이를 넘으면 공유메모리 범위를
/// 벗어난다(plans/cuda-port.md S9 — cap 1024가 실질 정합 상한의 원인).
pub const ATTN_SCORE_SCAP: usize = 1024;
/// KV 캐시 층당 위치 상한(hip 규약 — dkc/dvc [n_attn][cap][kv_dim]).
pub const ATTN_KV_CAP: usize = 1024;

/// 어텐션 형상 — 모델 config.json text_config에서 유도(27B/35B 상이).
/// d=256 고정(양 모델 head_dim=256 실측 2026-10-04). rope는 부분 회전
/// 64차원=불32쌍(partial_rotary_factor 0.25×256 — 커널 attn_theta가
/// 64차 기준 고정이라 d≠256이면 형상 계약 위반).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttnDims {
    /// full-attn 층수(il%interval==interval-1, 어텐션 색인 ai=il/interval
    /// — 27B 16, 35B 10).
    pub n_attn: usize,
    /// q헤드 수(27B 24, 35B 16).
    pub q_heads: usize,
    /// KV헤드 수(27B 4, 35B 2).
    pub kv_heads: usize,
    /// 헤드 폭(256 고정 계약).
    pub d: usize,
    /// KV 캐시 층당 위치 상한(hip 규약 1024).
    pub cap: usize,
}

impl AttnDims {
    /// KV그룹 폭(GQA — q헤드당 KV헤드 비율. 27B 6, 35B 8).
    pub fn gq(&self) -> usize {
        self.q_heads / self.kv_heads
    }
    /// q/qh/outv 폭.
    pub fn q_dim(&self) -> usize {
        self.q_heads * self.d
    }
    /// k/v/KV 캐시 폭.
    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.d
    }
    /// qg 폭(q‖gate 인터리브 — 헤드당 512: 256 q + 256 gate).
    pub fn qg_dim(&self) -> usize {
        self.q_heads * 2 * self.d
    }
    /// 슬롯 s의 KV 캐시 오프셋(원소) — 레이아웃 [n_slots][n_attn][cap][kv_dim]
    /// (plans/cuda-port.md S8). 커널 layer 인덱스는 0..n_attn을 유지하므로
    /// 슬롯 구분은 이 오프셋으로만 한다.
    pub fn kv_slot_elems(&self, slot: usize) -> usize {
        slot * self.n_attn * self.cap * self.kv_dim()
    }

    /// config.json 본문 → 형상(GdnDims와 동일 파서·규약 — 실측 차원).
    pub fn from_config(cfg: &str) -> Result<Self, String> {
        let v = JParser {
            b: cfg.as_bytes(),
            p: 0,
        }
        .parse()?;
        let tc = v.get("text_config").unwrap_or(&v);
        let num = |k: &str| {
            tc.get(k)
                .and_then(JVal::as_f64)
                .ok_or_else(|| format!("config.json: text_config.{k} 없음"))
        };
        let q_heads = num("num_attention_heads")? as usize;
        let kv_heads = num("num_key_value_heads")? as usize;
        let d = num("head_dim")? as usize;
        let n_layers = num("num_hidden_layers")? as usize;
        let interval = tc
            .get("full_attention_interval")
            .and_then(JVal::as_f64)
            .unwrap_or(4.0) as usize;
        if d != 256 {
            return Err(format!(
                "attn: head_dim {d} — 256 고정 계약(attn_theta 64차 rope)"
            ));
        }
        if q_heads == 0 || kv_heads == 0 || !q_heads.is_multiple_of(kv_heads) {
            return Err(format!(
                "attn: q_heads={q_heads} kv_heads={kv_heads} — q%kv==0 계약(GQA 그룹)"
            ));
        }
        let n_attn = n_layers / interval.max(1);
        if n_attn == 0 {
            return Err("attn: n_attn=0".into());
        }
        Ok(AttnDims {
            n_attn,
            q_heads,
            kv_heads,
            d,
            cap: ATTN_KV_CAP,
        })
    }
}

/// 어텐션 체인 중간 산출 판독(검증층 — 단계별 값 판정용).
pub struct AttnOut {
    /// prep 산출 q(디인터리브+rms+rope) [t][q_dim].
    pub qh: Vec<f32>,
    /// fwd3s 산출 게이트 어텐션 출력 [t][q_dim].
    pub outv: Vec<f32>,
    /// 현 체인이 기록한 KC 신규 행 [t][kv_dim].
    pub kc_rows: Vec<f32>,
    /// 현 체인이 기록한 VC 신규 행 [t][kv_dim].
    pub vc_rows: Vec<f32>,
}

impl Exl3CudaDecoder {
    // ── 어텐션 체인(G6 — plans/124 §1, rawhip exl3_hip.rs 체인 구조 미러) ──

    /// 등록 어텐션 형상(미등록이면 Err).
    pub fn attn_dims(&self) -> Result<AttnDims, String> {
        self.attn
            .ok_or_else(|| "attn: 형상 미등록(set_attn)".to_string())
    }

    /// 어텐션 형상·q/k 노름 상주 등록 + KV 캐시·pp 상주 할당(제로 초기화).
    /// qnw/knw: [n_attn][256] f32 — 아카이브는 w−1 저장 규약(§3.4,
    /// constant_bias=1.0)이므로 호출자가 +1한 값을 등록한다(hip
    /// attn_norms_dump와 동일 경로). KV 캐시 [n_attn][cap][kv_dim]은
    /// 0 기록(프리필 히스토리는 attn_seed_kv로 시딩).
    pub fn set_attn(&mut self, dims: AttnDims, qnw: &[f32], knw: &[f32]) -> Result<(), String> {
        let n = dims.n_attn;
        if qnw.len() != n * 256 || knw.len() != n * 256 {
            return Err(format!(
                "attn: qnw/knw {}/{} != {n}x256",
                qnw.len(),
                knw.len()
            ));
        }
        let slots = self.n_slots.max(1);
        // plans/cuda-port.md S8: KV 캐시는 슬롯 최외곽. 27B·ctx 4096 기준
        // 슬롯당 16층×4096×1024×4B×2(K·V) ≈ 536MB — 슬롯 수의 실질 상한.
        let kv_elems = slots * n * dims.cap * dims.kv_dim();
        for q in [
            self.dqnw_a,
            self.dknw_a,
            self.dkc,
            self.dvc,
            self.dpp,
            self.dqg_a,
            self.dkin_a,
            self.dvin_a,
            self.dqh_a,
            self.doutv_a,
        ] {
            if q != 0 {
                self.cc.free(q)?;
            }
        }
        (self.dqnw_a, self.dknw_a, self.dkc, self.dvc, self.dpp) = (0, 0, 0, 0, 0);
        (
            self.dqg_a,
            self.dkin_a,
            self.dvin_a,
            self.dqh_a,
            self.doutv_a,
        ) = (0, 0, 0, 0, 0);
        self.attn_t_cap = 0;
        // SAFETY: f32 슬라이스 → 바이트 뷰(길이·정렬 일치).
        let b =
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        let dq = self.cc.alloc(qnw.len() * 4)?;
        Self::h2d_chunked(&self.cc, dq, b(qnw))?;
        let dk = self.cc.alloc(knw.len() * 4)?;
        Self::h2d_chunked(&self.cc, dk, b(knw))?;
        let dkc = self.cc.alloc(kv_elems * 4)?;
        Self::h2d_chunked(&self.cc, dkc, &vec![0u8; kv_elems * 4])?;
        let dvc = self.cc.alloc(kv_elems * 4)?;
        Self::h2d_chunked(&self.cc, dvc, &vec![0u8; kv_elems * 4])?;
        let dpp = self.cc.alloc(slots * 4)?;
        Self::h2d_chunked(&self.cc, dpp, &vec![0u8; slots * 4])?;
        self.dqnw_a = dq;
        self.dknw_a = dk;
        self.dkc = dkc;
        self.dvc = dvc;
        self.dpp = dpp;
        self.attn = Some(dims);
        Ok(())
    }

    /// 어텐션 작업 버퍼 보장(t 상한 확장 시에만 재할당).
    fn ensure_attn_bufs(&mut self, t_len: usize) -> Result<(), String> {
        if t_len <= self.attn_t_cap {
            return Ok(());
        }
        let dm = self.attn_dims()?;
        for q in [
            self.dqg_a,
            self.dkin_a,
            self.dvin_a,
            self.dqh_a,
            self.doutv_a,
        ] {
            if q != 0 {
                self.cc.free(q)?;
            }
        }
        self.dqg_a = self.cc.alloc(t_len * dm.qg_dim() * 4)?;
        self.dkin_a = self.cc.alloc(t_len * dm.kv_dim() * 4)?;
        self.dvin_a = self.cc.alloc(t_len * dm.kv_dim() * 4)?;
        self.dqh_a = self.cc.alloc(t_len * dm.q_dim() * 4)?;
        self.doutv_a = self.cc.alloc(t_len * dm.q_dim() * 4)?;
        self.attn_t_cap = t_len;
        Ok(())
    }

    /// pp[0] 상주값 갱신(h2d — 결함 4호: pos의 진실은 장치 버퍼. 캡처
    /// 그래프 내 전진은 attn_pos_bump 커널 — h2d는 캡처 무효화, 결함 16호).
    pub fn attn_set_pos(&mut self, slot: usize, pos: u32) -> Result<(), String> {
        if self.dpp == 0 {
            return Err("attn: pp 미할당(set_attn 먼저)".into());
        }
        if slot >= self.n_slots.max(1) {
            return Err(format!("attn slot={slot} >= n_slots={}", self.n_slots));
        }
        // SAFETY: 슬롯 s는 dpp[s] (4B) — n_slots×4B 할당 경계 내(S8).
        self.cc
            .h2d(self.dpp + (slot as u64) * 4, &pos.to_le_bytes())
    }

    /// 슬롯별 pp 슬라이스 포인터 — 커널은 pp[0]만 읽으므로 슬롯 오프셋을
    /// 넘긴다(plans/cuda-port.md S8: 가중치 색인과 무관한 유일한 예외).
    fn attn_pp_ptr(&self, slot: usize) -> CUdeviceptr {
        self.dpp + (slot as u64) * 4
    }

    /// 슬롯별 KV 캐시 포인터(바이트 오프셋 적용) — set_attn이 슬롯
    /// 최외곽 레이아웃으로 할당한다.
    fn attn_kv_ptr(&self, slot: usize) -> CUdeviceptr {
        match self.attn {
            Some(dm) => self.dkc + (dm.kv_slot_elems(slot) as u64) * 4,
            None => self.dkc,
        }
    }

    /// 슬롯별 VC 캐시 포인터(attn_kv_ptr의 V 대응).
    fn attn_vc_ptr(&self, slot: usize) -> CUdeviceptr {
        match self.attn {
            Some(dm) => self.dvc + (dm.kv_slot_elems(slot) as u64) * 4,
            None => self.dvc,
        }
    }

    /// pp[0] += 1 커널 발사(exl3_attn_pos_bump) — 장치 내 pos 전진.
    pub fn attn_pos_bump(&mut self, slot: usize) -> Result<(), String> {
        let f = self.cc.function("exl3_attn_pos_bump")?;
        let mut p0 = self.attn_pp_ptr(slot);
        let mut args: [*mut std::ffi::c_void; 1] = [(&mut p0) as *mut _ as *mut _];
        self.cc.launch(f, 1, 1, 32, &mut args)
    }

    /// 어텐션 층 KV 캐시 시딩(프리필 히스토리) — kc/vc: [cap][kv_dim].
    pub fn attn_seed_kv(
        &mut self,
        slot: usize,
        layer: usize,
        kc: &[f32],
        vc: &[f32],
    ) -> Result<(), String> {
        let dm = self.attn_dims()?;
        if layer >= dm.n_attn {
            return Err(format!("attn layer={layer} >= n_attn={}", dm.n_attn));
        }
        if slot >= self.n_slots.max(1) {
            return Err(format!("attn slot={slot} >= n_slots={}", self.n_slots));
        }
        let elems = dm.cap * dm.kv_dim();
        if kc.len() != elems || vc.len() != elems {
            return Err(format!(
                "attn seed: kc/vc {}/{} != cap {}x{}",
                kc.len(),
                vc.len(),
                dm.cap,
                dm.kv_dim()
            ));
        }
        // SAFETY: f32 슬라이스 → 바이트 뷰(길이·정렬 일치).
        let b =
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        // SAFETY: 슬롯×층 슬라이스 오프셋 — n_slots×n_attn×cap×kv_dim 경계 내(S8).
        let off_k = self.dkc + ((slot * dm.n_attn + layer) * elems) as u64 * 4;
        Self::h2d_chunked(&self.cc, off_k, b(kc))?;
        // SAFETY: 슬롯×층 슬라이스 오프셋 — 경계 내(S8).
        let off_v = self.dvc + ((slot * dm.n_attn + layer) * elems) as u64 * 4;
        Self::h2d_chunked(&self.cc, off_v, b(vc))?;
        Ok(())
    }

    /// prep 발사(내부) — 그리드 (t_len, q_heads+kv_heads), 블록 128.
    /// hostpos=true는 음성대조 쌍둥이(exl3_attn_prep_hostpos — 호스트
    /// pos 파라미터, 결함 4호 재현). 검증 경로 외 발사 금지.
    fn attn_prep_launch(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        hostpos: bool,
        pos0_host: u32,
    ) -> Result<(), String> {
        let dm = self.attn_dims()?;
        // S8: KV·pp만 슬롯 오프셋. qnw/knw는 슬롯 공유 가중치라
        // layer 인덱스를 그대로 둔다(오염 시 전 층 가중치 오독).
        let kv = self.attn_kv_ptr(slot);
        let f = if hostpos {
            self.cc.function("exl3_attn_prep_hostpos")?
        } else {
            self.cc.function("exl3_attn_prep")?
        };
        let (mut tl, mut lay) = (t_len as i32, layer as i32);
        let (mut qh, mut kvh, mut cp) = (dm.q_heads as i32, dm.kv_heads as i32, dm.cap as i32);
        let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7, mut a8) = (
            self.dqg_a,
            self.dkin_a,
            self.dvin_a,
            self.dqnw_a,
            self.dknw_a,
            self.dqh_a,
            kv,
            self.attn_vc_ptr(slot),
            self.attn_pp_ptr(slot),
        );
        let grid_y = (dm.q_heads + dm.kv_heads) as u32;
        if hostpos {
            let mut p0h = pos0_host as i32;
            let mut args: [*mut std::ffi::c_void; 15] = [
                (&mut a0) as *mut _ as *mut _,
                (&mut a1) as *mut _ as *mut _,
                (&mut a2) as *mut _ as *mut _,
                (&mut a3) as *mut _ as *mut _,
                (&mut a4) as *mut _ as *mut _,
                (&mut a5) as *mut _ as *mut _,
                (&mut a6) as *mut _ as *mut _,
                (&mut a7) as *mut _ as *mut _,
                (&mut a8) as *mut _ as *mut _,
                (&mut tl) as *mut _ as *mut _,
                (&mut lay) as *mut _ as *mut _,
                (&mut qh) as *mut _ as *mut _,
                (&mut kvh) as *mut _ as *mut _,
                (&mut cp) as *mut _ as *mut _,
                (&mut p0h) as *mut _ as *mut _,
            ];
            self.cc.launch(f, t_len as u32, grid_y, 128, &mut args)
        } else {
            let mut args: [*mut std::ffi::c_void; 14] = [
                (&mut a0) as *mut _ as *mut _,
                (&mut a1) as *mut _ as *mut _,
                (&mut a2) as *mut _ as *mut _,
                (&mut a3) as *mut _ as *mut _,
                (&mut a4) as *mut _ as *mut _,
                (&mut a5) as *mut _ as *mut _,
                (&mut a6) as *mut _ as *mut _,
                (&mut a7) as *mut _ as *mut _,
                (&mut a8) as *mut _ as *mut _,
                (&mut tl) as *mut _ as *mut _,
                (&mut lay) as *mut _ as *mut _,
                (&mut qh) as *mut _ as *mut _,
                (&mut kvh) as *mut _ as *mut _,
                (&mut cp) as *mut _ as *mut _,
            ];
            self.cc.launch(f, t_len as u32, grid_y, 128, &mut args)
        }
    }

    /// fwd3s 발사(내부) — 그리드 (t_len, q_heads), 블록 256. T≤8 도메인
    /// 사전 강제(Err — 커널 미발사; 커널 내 조기복귀와 이중 계약).
    fn attn_fwd3s_launch(&mut self, slot: usize, layer: usize, t_len: usize) -> Result<(), String> {
        let dm = self.attn_dims()?;
        if t_len == 0 || t_len > ATTN_F3S_TMAX {
            return Err(format!(
                "attn fwd3s: T={t_len} — 소형 전용 도메인(1..={ATTN_F3S_TMAX}) 위반, 거부"
            ));
        }
        let f = self.cc.function("exl3_attn_fwd3s")?;
        let (mut tl, mut lay) = (t_len as i32, layer as i32);
        let (mut qh, mut kvh, mut cp) = (dm.q_heads as i32, dm.kv_heads as i32, dm.cap as i32);
        let (mut f0, mut f1, mut f2, mut f3, mut f4, mut f5) = (
            self.dqh_a,
            self.attn_kv_ptr(slot),
            self.attn_vc_ptr(slot),
            self.dqg_a,
            self.doutv_a,
            self.attn_pp_ptr(slot),
        );
        let mut args: [*mut std::ffi::c_void; 11] = [
            (&mut f0) as *mut _ as *mut _,
            (&mut f1) as *mut _ as *mut _,
            (&mut f2) as *mut _ as *mut _,
            (&mut f3) as *mut _ as *mut _,
            (&mut f4) as *mut _ as *mut _,
            (&mut f5) as *mut _ as *mut _,
            (&mut tl) as *mut _ as *mut _,
            (&mut lay) as *mut _ as *mut _,
            (&mut qh) as *mut _ as *mut _,
            (&mut kvh) as *mut _ as *mut _,
            (&mut cp) as *mut _ as *mut _,
        ];
        self.cc
            .launch(f, t_len as u32, dm.q_heads as u32, 256, &mut args)
    }

    /// 어텐션 체인 입력 공통 검증·업로드(dqg/dkin/dvin — pos는 기입하지
    /// 않는다: 결함 4호 계약상 pos는 attn_set_pos/pos_bump가 별도 소유).
    pub fn attn_upload(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        qg: &[f32],
        kin: &[f32],
        vin: &[f32],
    ) -> Result<(), String> {
        let dm = self.attn_dims()?;
        if layer >= dm.n_attn {
            return Err(format!("attn layer={layer} >= n_attn={}", dm.n_attn));
        }
        if slot >= self.n_slots.max(1) {
            return Err(format!("attn slot={slot} >= n_slots={}", self.n_slots));
        }
        if qg.len() != t_len * dm.qg_dim()
            || kin.len() != t_len * dm.kv_dim()
            || vin.len() != t_len * dm.kv_dim()
        {
            return Err(format!(
                "attn: qg={} kin={} vin={} — t={t_len} qg_dim={} kv_dim={} 계약 위반",
                qg.len(),
                kin.len(),
                vin.len(),
                dm.qg_dim(),
                dm.kv_dim()
            ));
        }
        self.ensure_attn_bufs(t_len)?;
        // SAFETY: f32 슬라이스 → 바이트 뷰(길이·정렬 일치).
        let b =
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        self.cc.h2d(self.dqg_a, b(qg))?;
        self.cc.h2d(self.dkin_a, b(kin))?;
        self.cc.h2d(self.dvin_a, b(vin))
    }

    /// 어텐션 체인 디바이스 상주(S10): qg·kin·vin이 이미 디바이스에 있을 때
    /// prep → fwd3s를 발사하고 outv(디바이스)를 반환한다. 호스트 왕복 0.
    /// pos는 attn_set_pos로 설정된 장치 pp[0]을 그대로 쓴다(결함 4호).
    /// 반환 포인터는 self.doutv_a(임시 — 다음 호출이 덮어쓴다).
    pub fn attn_chain_dev_run(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        qg_dev: CUdeviceptr,
        kin_dev: CUdeviceptr,
        vin_dev: CUdeviceptr,
    ) -> Result<CUdeviceptr, String> {
        let dm = self.attn_dims()?;
        if t_len == 0 || t_len > ATTN_F3S_TMAX {
            return Err(format!(
                "attn: T={t_len} — fwd3s 소형 전용 도메인(1..={ATTN_F3S_TMAX}) 위반, 거부"
            ));
        }
        if layer >= dm.n_attn {
            return Err(format!("attn layer={layer} >= n_attn={}", dm.n_attn));
        }
        if slot >= self.n_slots.max(1) {
            return Err(format!("attn slot={slot} >= n_slots={}", self.n_slots));
        }
        // 위치축 한계(S9) — 커널 조기복귀 대신 여기서 명시적 거부.
        let pos = self.slot_pos.get(slot).copied().unwrap_or(0);
        if pos as usize + t_len > ATTN_SCORE_SCAP {
            return Err(format!(
                "attn: 위치 {} > fwd3s 공유메모리 한계 {ATTN_SCORE_SCAP} (S9)",
                pos as usize + t_len
            ));
        }
        // **pp[0]을 현재 위치로 기입한다.** prep/fwd3s는 발사 인자가 아니라
        // 장치 pp[0]에서 pos를 읽는다(결함 4호). 호스트 경로는
        // attn_chain_host가 attn_set_pos로 매 층 세팅하지만, 디바이스
        // 경로는 세팅이 없어 pp[0]이 0에 머물렀다 — 0번 토큰은 우연히
        // 맞고 2번째부터 KV 기록 위치가 밀려 어텐션이 틀어진다(원장 S10).
        self.attn_set_pos(slot, pos)?;
        self.ensure_attn_bufs(t_len)?;
        // qg/kin/vin을 상주 스테이징으로 복사한다 — attn_upload이 하던
        // h2d를 대신한다. 빠지면 이전 층의 값이 남는다(원장 S10).
        // prep 커널이 읽는 **모듈 작업 버퍼**로 복사한다 — 상주 스테이징에
        // 만 두면 prep는 이전 층의 값을 본다(원장 S10).
        self.cc.d2d(self.dqg_a, qg_dev, t_len * dm.qg_dim() * 4)?;
        self.cc.d2d(self.dkin_a, kin_dev, t_len * dm.kv_dim() * 4)?;
        self.cc.d2d(self.dvin_a, vin_dev, t_len * dm.kv_dim() * 4)?;
        self.attn_prep_launch(slot, layer, t_len, false, 0)?;
        self.attn_fwd3s_launch(slot, layer, t_len)?;
        Ok(self.doutv_a)
    }

    /// 어텐션 체인 본체(pp 미개입 — 현장치값 사용): prep(디바이스 pos
    /// 판독) → fwd3s → qh·outv·KC/VC 신규 행 판독. T 도메인(1..=8)과
    /// 캐시 상한을 검사한다(pos0는 경계 판정용 판독값 — fwd3s 산출은
    /// 전적으로 디바이스 pp[0]에 의존, 결함 4호).
    fn attn_chain_dev(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        pos0: u32,
    ) -> Result<AttnOut, String> {
        let dm = self.attn_dims()?;
        if t_len == 0 || t_len > ATTN_F3S_TMAX {
            return Err(format!(
                "attn: T={t_len} — fwd3s 소형 전용 도메인(1..={ATTN_F3S_TMAX}) 위반, 거부"
            ));
        }
        if layer >= dm.n_attn {
            return Err(format!("attn layer={layer} >= n_attn={}", dm.n_attn));
        }
        if pos0 as usize + t_len > dm.cap {
            return Err(format!(
                "attn: pos0={pos0} + T={t_len} > cap={}(KV 캐시 상한)",
                dm.cap
            ));
        }
        // fwd3s의 공유메모리 sarr[ATTN_SCORE_SCAP] 한계(pos축). 커널도
        // 조기복귀하지만 그건 "조용히 오답"이므로 여기서 runtime Err로 명시적
        // 거부한다(plans/cuda-port.md S9). 진짜 해법은 위치 청크 분할 온라인
        // 소프트맥스 재작성 — 그전까지 cap 1024가 실질 정합 상한이다.
        let end_pos = pos0 as usize + t_len;
        if end_pos > ATTN_SCORE_SCAP {
            return Err(format!(
                "attn: 위치 {end_pos} > fwd3s 공유메모리 한계 {ATTN_SCORE_SCAP} — \
                 CUDA 어텐션은 cap {ATTN_SCORE_SCAP}까지만 정합(위치 청크 미구현, S9)"
            ));
        }
        self.attn_prep_launch(slot, layer, t_len, false, 0)?;
        self.attn_fwd3s_launch(slot, layer, t_len)?;
        let take = |n: usize, src: CUdeviceptr| -> Result<Vec<f32>, String> {
            let mut buf = vec![0u8; n * 4];
            self.cc.d2h(&mut buf, src)?;
            // SAFETY: d2h 완료 후 재해석(길이·정렬 일치 — G2 판독 패턴).
            Ok(unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const f32, n).to_vec() })
        };
        let qh = take(t_len * dm.q_dim(), self.dqh_a)?;
        let outv = take(t_len * dm.q_dim(), self.doutv_a)?;
        // SAFETY: 슬롯×층×위치 슬라이스 오프셋 — 경계 내(S8).
        let row = t_len * dm.kv_dim();
        let kc_rows = take(
            row,
            self.attn_kv_ptr(slot) + ((layer * dm.cap + pos0 as usize) * dm.kv_dim()) as u64 * 4,
        )?;
        let vc_rows = take(
            row,
            self.attn_vc_ptr(slot) + ((layer * dm.cap + pos0 as usize) * dm.kv_dim()) as u64 * 4,
        )?;
        self.cc.sync()?;
        Ok(AttnOut {
            qh,
            outv,
            kc_rows,
            vc_rows,
        })
    }

    /// 어텐션 체인 호스트 진입(hip 검증 프로브 흐름 미러): 업로드 →
    /// pp[0]=pos0(h2d) → prep(디바이스 pos 판독) → fwd3s → qh·outv·KC/VC
    /// 신규 행 판독. T>8이면 Err(도메인 거부 — 커널 미발사).
    pub fn attn_chain_host(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        qg: &[f32],
        kin: &[f32],
        vin: &[f32],
        pos0: u32,
    ) -> Result<AttnOut, String> {
        self.attn_upload(slot, layer, t_len, qg, kin, vin)?;
        self.attn_set_pos(slot, pos0)?;
        self.attn_chain_dev(slot, layer, t_len, pos0)
    }

    /// 음성대조 계기(원장 17호 — 결함 4호 재현): prep을 호스트 pos 사본
    /// (pos0_host)으로 실행하고 fwd3s는 실경로(디바이스 pp[0] 판독)로
    /// 실행해 종단 outv를 판독. pp[0]은 이 호출이 기입하지 않는다 —
    /// 호출자가 attn_set_pos/pos_bump로 만든 장치 상태 그대로(시나리오
    /// 보존). 호스트 사본이 낡은 값이면 KV 기록 위치가 어긋나 종단이
    /// 이격된다 — 그 이격이 값 판정으로 잡히는지가 검증 대상.
    /// 검증층 전용 API(정상 호출 금지 — gdn gather 계기와 동일 계열).
    pub fn attn_chain_host_hostpos(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        qg: &[f32],
        kin: &[f32],
        vin: &[f32],
        pos0_host: u32,
    ) -> Result<Vec<f32>, String> {
        self.attn_upload(slot, layer, t_len, qg, kin, vin)?;
        self.attn_prep_launch(slot, layer, t_len, true, pos0_host)?;
        self.attn_fwd3s_launch(slot, layer, t_len)?;
        let dm = self.attn_dims()?;
        let mut buf = vec![0u8; t_len * dm.q_dim() * 4];
        self.cc.d2h(&mut buf, self.doutv_a)?;
        self.cc.sync()?;
        // SAFETY: d2h 완료 후 재해석(길이·정렬 일치).
        Ok(unsafe {
            std::slice::from_raw_parts(buf.as_ptr() as *const f32, t_len * dm.q_dim()).to_vec()
        })
    }
}
