//! EXL3 GDN 체인(conv→l2perm→scan→gate) 모듈층(plans/124 G5, G10 파일
//! 분할 2026-10-04). Exl3CudaDecoder의 GDN 임플 블록 + GdnDims/GdnMids
//! 형상·판독 자료형 — 컨텍스트·fatbin 리졸버는 exl3_cuda.rs 공유 글루.
//!
//! [용도] GDN 층(27B 48층·35B-A3B 30층 — full-attn il%4==3 제외) 체인:
//! conv 채널별 3탭 링(순차 회전) → l2perm scatter(j=(h%3)*16+h/3 — gather
//! 아님, §3.3) → scan FLA 청크 알고리즘(후진 소거, S0≠0 상태 경로 의무) →
//! gate. 형상은 GdnDims(config.json 유도, 27B hv=48·35B hv=32 — 결함 1호
//! 정신: 형상은 명시 등록, 추정 금지). q는 이미 l2 정규화됨(중복 스케일
//! 금지). 산술 계약은 assets/exl3_gdn.cu(src_exl3.hip 1:1 직이식,
//! -fmad=false 빌드) — 결함 5호 가드(그리드 (h_v, T)).
//!
//! [정합 — plans/129-cuda C2 원장, sm_89 실측 2026-10-04] (i) 27B lay=47
//! T=32 S0≠0: 종단·전 단계 maxdiff 0.000e0(오라클과 비트동일) · rel>5%
//! 0/196608 · (ii) 35B-A3B lay=29 hidden=2048 hv=32 conv_ch=8192: 0.000e0 ·
//! rel 0/131072 — 미러 계약 실증(임계 2e-4). 음성대조 (a) l2perm gather:
//! 2.706e-1 · (b) S0=0: 1.150e0 → NEG-DETECTED(방향 결함·상태 경로 무시
//! 모두 탐지).
//!
//! [속도] 측정 대기 sm_80 — CMP 170HX 미도착(plans/124 §0). hip 8060S
//! 참고치 T=32 1.0ms; sm_80 자원 증거: exl3_gdn_scan REG:48 + 동적 smem
//! 61,828B opt-in → 2블록/SM 상한(그리드 h_v=48 블록 — 청크 순차 상태
//! 의존으로 헤드 이상 병렬화 불가).
//!
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 의존 금지 — scripts/cuda_probe_shim.rs 단독 컴파일.

use crate::rawcuda::exl3_cuda::{Exl3CudaDecoder, JParser, JVal};
use crate::rawcuda::ffi::CUdeviceptr;

/// GDN scan 커널 동적 공유메모리 바이트(assets/exl3_gdn.cu 계약 —
/// sk/sv/KS/QS 8KB×4 + A/KQ 2KB×2 + dc 16KB + Stile 8KB + bp/gcs/wsm
/// 388B. CUDA 정적 __shared__ 한계 48KB 초과 → opt-in 동적 할당).
pub const GDN_SCAN_SMEM: u32 = 61_828;

/// GDN 체인 형상 — 모델 config.json text_config에서 유도(27B와
/// 35B-A3B가 상이 — 결함 1호 정신: 형상은 명시 등록, 추정 금지).
/// d=128 고정(양 모델 linear_*_head_dim=128 실측 2026-10-04).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GdnDims {
    /// GDN 층수(full-attn il%4==3 제외 — 27B 48, 35B 30).
    pub n_gdn: usize,
    /// 잔류 폭(xn=xtb 행 폭 — 27B 5120, 35B 2048).
    pub hidden: usize,
    /// k(q/k) 헤드 수(양 모델 16).
    pub h_k: usize,
    /// v 헤드 수(27B 48, 35B 32).
    pub h_v: usize,
    /// 헤드 폭(128 고정 계약).
    pub d: usize,
}

impl GdnDims {
    /// conv 채널 수(2·k_len + v_len — 27B 10240, 35B 8192).
    pub fn conv_ch(&self) -> usize {
        (2 * self.h_k + self.h_v) * self.d
    }
    /// q/k 폭(헤드 폭 합).
    pub fn k_len(&self) -> usize {
        self.h_k * self.d
    }
    /// v/o/gated 폭.
    pub fn v_len(&self) -> usize {
        self.h_v * self.d
    }
    /// bg 폭(beta‖g, lc 순열 — 2·h_v).
    pub fn bg_len(&self) -> usize {
        2 * self.h_v
    }
    /// config.json 본문 → 형상(load와 동일 파서·text_config 규약).
    /// G5 프로브는 실모델 config에서 형상을 읽는다(계약: 실측 차원).
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
        let hidden = num("hidden_size")? as usize;
        let h_k = num("linear_num_key_heads")? as usize;
        let h_v = num("linear_num_value_heads")? as usize;
        let d = num("linear_key_head_dim")? as usize;
        let dv = num("linear_value_head_dim")? as usize;
        let n_layers = num("num_hidden_layers")? as usize;
        let interval = tc
            .get("full_attention_interval")
            .and_then(JVal::as_f64)
            .unwrap_or(4.0) as usize;
        let n_gdn = n_layers - n_layers / interval.max(1);
        if d != 128 || dv != 128 {
            return Err(format!("GDN: head_dim {d}/{dv} — 128 고정 계약"));
        }
        if hidden % 128 != 0 || hidden < 128 {
            return Err(format!("GDN: hidden={hidden} — 128 배수 계약"));
        }
        if h_v % h_k != 0 || h_k == 0 {
            return Err(format!(
                "GDN: h_v={h_v} h_k={h_k} — h_v%h_k==0 계약(lc 순열 전치)"
            ));
        }
        if n_gdn == 0 {
            return Err("GDN: n_gdn=0".into());
        }
        Ok(GdnDims {
            n_gdn,
            hidden,
            h_k,
            h_v,
            d,
        })
    }
}

/// GDN 체인 중간 산출 판독(검증층 진단 — 단계별 값 판정용).
pub struct GdnMids {
    /// conv 산출 q [t][k_len].
    pub conv_q: Vec<f32>,
    pub conv_k: Vec<f32>,
    pub conv_v: Vec<f32>,
    /// l2perm 산출 q/k(L2)·v(lc) [t][...] · bg [t][2·h_v].
    pub q2: Vec<f32>,
    pub k2: Vec<f32>,
    pub v2: Vec<f32>,
    pub bg: Vec<f32>,
    /// scan 산출 o_lc [t][v_len].
    pub o_lc: Vec<f32>,
    /// T행 처리 후 링 [3][conv_ch](층 슬라이스).
    pub ring_post: Vec<f32>,
    /// T행 처리 후 상태 [h_v][128·128](층 슬라이스).
    pub st_post: Vec<f32>,
}

impl Exl3CudaDecoder {
    // ── GDN 체인(G5 — plans/124 §3.3, rawhip exl3_hip.rs 체인 구조 미러) ──

    /// 등록 GDN 형상(미등록이면 Err).
    pub fn gdn_dims(&self) -> Result<GdnDims, String> {
        self.gdn
            .ok_or_else(|| "GDN: 형상 미등록(set_gdn)".to_string())
    }

    /// GDN 형상·상수 등록 + 링/상태 상주 할당(제로 초기화 — hip load
    /// 미러). 인자는 전층 배열(커널 lay 인자 인덱싱): cw [n_gdn][conv_ch][4]
    /// · ab [n_gdn][2][h_v][hidden] · alog/dtb [n_gdn][h_v] ·
    /// nw [n_gdn][128]. alog는 A_log 원값(ssm_a=-exp는 커널이 산출).
    pub fn set_gdn(
        &mut self,
        dims: GdnDims,
        cw: &[f32],
        ab: &[f32],
        alog: &[f32],
        dtb: &[f32],
        nw: &[f32],
    ) -> Result<(), String> {
        let (n, cch, hv, hd) = (dims.n_gdn, dims.conv_ch(), dims.h_v, dims.hidden);
        if cw.len() != n * cch * 4 {
            return Err(format!("GDN cw {} != {n}x{cch}x4", cw.len()));
        }
        if ab.len() != n * 2 * hv * hd {
            return Err(format!("GDN ab {} != {n}x2x{hv}x{hd}", ab.len()));
        }
        if alog.len() != n * hv || dtb.len() != n * hv {
            return Err(format!(
                "GDN alog/dtb {}/{} != {n}x{hv}",
                alog.len(),
                dtb.len()
            ));
        }
        if nw.len() != n * 128 {
            return Err(format!("GDN nw {} != {n}x128", nw.len()));
        }
        // 기존 상주 해제(형상 변경 대응 — 작업 버퍼도 리셋).
        for q in [
            self.dcw, self.dab_c, self.dalog, self.ddtb, self.dnwg, self.dring, self.dgst,
            self.dqkv, self.dzv, self.dgxn, self.dgq, self.dgk, self.dgv, self.dq2, self.dk2,
            self.dv2, self.dbg, self.dgo, self.dgate,
        ] {
            if q != 0 {
                self.cc.free(q)?;
            }
        }
        (self.dcw, self.dab_c, self.dalog, self.ddtb, self.dnwg) = (0, 0, 0, 0, 0);
        (self.dring, self.dgst) = (0, 0);
        (self.dqkv, self.dzv, self.dgxn) = (0, 0, 0);
        (self.dgq, self.dgk, self.dgv) = (0, 0, 0);
        (self.dq2, self.dk2, self.dv2) = (0, 0, 0);
        (self.dbg, self.dgo, self.dgate) = (0, 0, 0);
        self.gdn_t_cap = 0;
        // SAFETY: f32 슬라이스 → 바이트 뷰(길이·정렬 일치).
        let b =
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        let dcw = self.cc.alloc(cw.len() * 4)?;
        Self::h2d_chunked(&self.cc, dcw, b(cw))?;
        let dab = self.cc.alloc(ab.len() * 4)?;
        Self::h2d_chunked(&self.cc, dab, b(ab))?;
        let dal = self.cc.alloc(alog.len() * 4)?;
        Self::h2d_chunked(&self.cc, dal, b(alog))?;
        let ddt = self.cc.alloc(dtb.len() * 4)?;
        Self::h2d_chunked(&self.cc, ddt, b(dtb))?;
        let dnw = self.cc.alloc(nw.len() * 4)?;
        Self::h2d_chunked(&self.cc, dnw, b(nw))?;
        let dring = self.cc.alloc(n * 3 * cch * 4)?;
        Self::h2d_chunked(&self.cc, dring, &vec![0u8; n * 3 * cch * 4])?;
        let dgst = self.cc.alloc(n * hv * 128 * 128 * 4)?;
        Self::h2d_chunked(&self.cc, dgst, &vec![0u8; n * hv * 128 * 128 * 4])?;
        (self.dcw, self.dab_c, self.dalog, self.ddtb, self.dnwg) = (dcw, dab, dal, ddt, dnw);
        self.dring = dring;
        self.dgst = dgst;
        self.gdn = Some(dims);
        Ok(())
    }

    /// GDN 작업 버퍼 보장(t 상한 확장 시에만 재할당).
    fn ensure_gdn_bufs(&mut self, t_len: usize) -> Result<(), String> {
        if t_len <= self.gdn_t_cap {
            return Ok(());
        }
        let dm = self.gdn_dims()?;
        let (hd, cch, kl, vl) = (dm.hidden, dm.conv_ch(), dm.k_len(), dm.v_len());
        for q in [
            self.dqkv, self.dzv, self.dgxn, self.dgq, self.dgk, self.dgv, self.dq2, self.dk2,
            self.dv2, self.dbg, self.dgo, self.dgate,
        ] {
            if q != 0 {
                self.cc.free(q)?;
            }
        }
        self.dqkv = self.cc.alloc(t_len * cch * 4)?;
        self.dzv = self.cc.alloc(t_len * vl * 4)?;
        self.dgxn = self.cc.alloc(t_len * hd * 4)?;
        self.dgq = self.cc.alloc(t_len * kl * 4)?;
        self.dgk = self.cc.alloc(t_len * kl * 4)?;
        self.dgv = self.cc.alloc(t_len * vl * 4)?;
        self.dq2 = self.cc.alloc(t_len * kl * 4)?;
        self.dk2 = self.cc.alloc(t_len * kl * 4)?;
        self.dv2 = self.cc.alloc(t_len * vl * 4)?;
        self.dbg = self.cc.alloc(t_len * dm.bg_len() * 4)?;
        self.dgo = self.cc.alloc(t_len * vl * 4)?;
        self.dgate = self.cc.alloc(t_len * vl * 4)?;
        self.gdn_t_cap = t_len;
        Ok(())
    }

    /// GDN 체인 디바이스 4발사(conv → l2perm → scan → gate) — 상태·링은
    /// 상주 버퍼(r/w). gather=true는 음성대조 계기(l2perm 방향 반전,
    /// 결함류: 방향 — 원장 17호. 정상 호출 금지). 그리드 계약(결함 5호):
    /// l2perm/gate는 (h_v, T), conv는 (conv_ch/128, 1), scan은 (h_v, 1).
    fn gdn_chain_dev(&mut self, layer: usize, t_len: usize, gather: bool) -> Result<(), String> {
        let dm = self.gdn_dims()?;
        if layer >= dm.n_gdn {
            return Err(format!("GDN layer={layer} >= n_gdn={}", dm.n_gdn));
        }
        if t_len == 0 {
            return Err("GDN t_len=0".into());
        }
        self.ensure_gdn_bufs(t_len)?;
        let (mut tl, mut lay) = (t_len as i32, layer as i32);
        let (mut hk, mut hv, mut dd) = (dm.h_k as i32, dm.h_v as i32, dm.d as i32);
        let (mut hd, mut kl, mut vl, mut cch) = (
            dm.hidden as i32,
            dm.k_len() as i32,
            dm.v_len() as i32,
            dm.conv_ch() as i32,
        );

        // conv — 채널축 1D 그리드(블록 128 = 완전 coalesce), T행은
        // 커널이 순차 회전(링 계약 — 한 런치).
        let f = self.cc.function("exl3_gdn_conv")?;
        let (mut c0, mut c1, mut c2, mut c3, mut c4, mut c5) = (
            self.dqkv, self.dcw, self.dring, self.dgq, self.dgk, self.dgv,
        );
        let mut ac: [*mut std::ffi::c_void; 11] = [
            (&mut c0) as *mut _ as *mut _,
            (&mut c1) as *mut _ as *mut _,
            (&mut c2) as *mut _ as *mut _,
            (&mut c3) as *mut _ as *mut _,
            (&mut c4) as *mut _ as *mut _,
            (&mut c5) as *mut _ as *mut _,
            (&mut tl) as *mut _ as *mut _,
            (&mut lay) as *mut _ as *mut _,
            (&mut kl) as *mut _ as *mut _,
            (&mut vl) as *mut _ as *mut _,
            (&mut cch) as *mut _ as *mut _,
        ];
        self.cc
            .launch(f, (dm.conv_ch() / 128) as u32, 1, 128, &mut ac)?;

        // l2perm — grid (h_v, T)(결함 5호: t=blockIdx.y), 블록 128.
        let f = if gather {
            self.cc.function("exl3_gdn_l2perm_gather")?
        } else {
            self.cc.function("exl3_gdn_l2perm")?
        };
        let (
            mut l0,
            mut l1,
            mut l2,
            mut l3,
            mut l4,
            mut l5,
            mut l6,
            mut l7,
            mut l8,
            mut l9,
            mut l10,
        ) = (
            self.dgq, self.dgk, self.dgv, self.dgxn, self.dab_c, self.dalog, self.ddtb, self.dq2,
            self.dk2, self.dv2, self.dbg,
        );
        let mut al: [*mut std::ffi::c_void; 16] = [
            (&mut l0) as *mut _ as *mut _,
            (&mut l1) as *mut _ as *mut _,
            (&mut l2) as *mut _ as *mut _,
            (&mut l3) as *mut _ as *mut _,
            (&mut l4) as *mut _ as *mut _,
            (&mut l5) as *mut _ as *mut _,
            (&mut l6) as *mut _ as *mut _,
            (&mut l7) as *mut _ as *mut _,
            (&mut l8) as *mut _ as *mut _,
            (&mut l9) as *mut _ as *mut _,
            (&mut l10) as *mut _ as *mut _,
            (&mut tl) as *mut _ as *mut _,
            (&mut lay) as *mut _ as *mut _,
            (&mut hk) as *mut _ as *mut _,
            (&mut hv) as *mut _ as *mut _,
            (&mut hd) as *mut _ as *mut _,
        ];
        self.cc
            .launch(f, dm.h_v as u32, t_len as u32, 128, &mut al)?;

        // scan — grid (h_v, 1), 블록 128, 동적 공유 61,828B(opt-in).
        let f = self.cc.function("exl3_gdn_scan")?;
        self.cc.set_dynamic_smem(f, GDN_SCAN_SMEM)?;
        let (mut s0, mut s1, mut s2, mut s3, mut s4, mut s5) =
            (self.dq2, self.dk2, self.dv2, self.dbg, self.dgst, self.dgo);
        let mut as_: [*mut std::ffi::c_void; 11] = [
            (&mut s0) as *mut _ as *mut _,
            (&mut s1) as *mut _ as *mut _,
            (&mut s2) as *mut _ as *mut _,
            (&mut s3) as *mut _ as *mut _,
            (&mut s4) as *mut _ as *mut _,
            (&mut s5) as *mut _ as *mut _,
            (&mut tl) as *mut _ as *mut _,
            (&mut hk) as *mut _ as *mut _,
            (&mut hv) as *mut _ as *mut _,
            (&mut dd) as *mut _ as *mut _,
            (&mut lay) as *mut _ as *mut _,
        ];
        self.cc
            .launch_shared(f, dm.h_v as u32, 1, 128, GDN_SCAN_SMEM, &mut as_)?;

        // gate — grid (h_v, T), 블록 128. gated는 별도 버퍼(헤드 순열
        // r/w 교차로 제자리 불가).
        let f = self.cc.function("exl3_gdn_gate")?;
        let (mut g0, mut g1, mut g2, mut g3) = (self.dgo, self.dzv, self.dnwg, self.dgate);
        let mut ag: [*mut std::ffi::c_void; 8] = [
            (&mut g0) as *mut _ as *mut _,
            (&mut g1) as *mut _ as *mut _,
            (&mut g2) as *mut _ as *mut _,
            (&mut g3) as *mut _ as *mut _,
            (&mut tl) as *mut _ as *mut _,
            (&mut lay) as *mut _ as *mut _,
            (&mut hk) as *mut _ as *mut _,
            (&mut hv) as *mut _ as *mut _,
        ];
        self.cc
            .launch(f, dm.h_v as u32, t_len as u32, 128, &mut ag)?;
        Ok(())
    }

    /// GDN 체인 호스트 진입(hip 검증 프로브 흐름 미러): xn·qkv·z 업로드
    /// → 4커널 → gated [t][v_len] 판독. s0/ring0 = Some이면 해당 층의
    /// 상태/링을 시드로 선업로드(S0≠0 검증 의무 — None이면 상주값 사용,
    /// 최초는 제로). 상태·링은 체인 후 갱신된 채 상주(순차 디코드 계약).
    pub fn gdn_chain_host(
        &mut self,
        layer: usize,
        t_len: usize,
        xn: &[f32],
        qkv: &[f32],
        z: &[f32],
        s0: Option<&[f32]>,
        ring0: Option<&[f32]>,
    ) -> Result<Vec<f32>, String> {
        let dm = self.gdn_dims()?;
        if xn.len() != t_len * dm.hidden
            || qkv.len() != t_len * dm.conv_ch()
            || z.len() != t_len * dm.v_len()
        {
            return Err(format!(
                "GDN: xn={} qkv={} z={} — t={t_len} hidden={} conv_ch={} v_len={} 계약 위반",
                xn.len(),
                qkv.len(),
                z.len(),
                dm.hidden,
                dm.conv_ch(),
                dm.v_len()
            ));
        }
        self.ensure_gdn_bufs(t_len)?;
        // SAFETY: f32 슬라이스 → 바이트 뷰(길이·정렬 일치).
        let b =
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        self.cc.h2d(self.dgxn, b(xn))?;
        self.cc.h2d(self.dqkv, b(qkv))?;
        self.cc.h2d(self.dzv, b(z))?;
        if let Some(s) = s0 {
            if s.len() != dm.h_v * 128 * 128 {
                return Err(format!("GDN s0 {} != {}x16384", s.len(), dm.h_v));
            }
            // SAFETY: 층 슬라이스 오프셋 — n_gdn×hv×16384 경계 내.
            let off = self.dgst + (layer * dm.h_v * 128 * 128) as u64 * 4;
            self.cc.h2d(off, b(s))?;
        }
        if let Some(r) = ring0 {
            if r.len() != 3 * dm.conv_ch() {
                return Err(format!("GDN ring0 {} != 3x{}", r.len(), dm.conv_ch()));
            }
            // SAFETY: 층 슬라이스 오프셋 — n_gdn×3×conv_ch 경계 내.
            let off = self.dring + (layer * 3 * dm.conv_ch()) as u64 * 4;
            self.cc.h2d(off, b(r))?;
        }
        self.gdn_chain_dev(layer, t_len, false)?;
        let mut ob = vec![0u8; t_len * dm.v_len() * 4];
        self.cc.d2h(&mut ob, self.dgate)?;
        self.cc.sync()?;
        // SAFETY: d2h 완료 후 재해석(길이·정렬 일치 — G2 판독 패턴).
        Ok(unsafe {
            std::slice::from_raw_parts(ob.as_ptr() as *const f32, t_len * dm.v_len()).to_vec()
        })
    }

    /// 음성대조 계기(원장 17호 — 계기 자체 검증): l2perm을 gather 방향
    /// (CPU 미러 쪽 — 결함류: 방향)으로 실행한 체인. 검증층 전용 API.
    pub fn gdn_chain_host_gather_l2perm(
        &mut self,
        layer: usize,
        t_len: usize,
        xn: &[f32],
        qkv: &[f32],
        z: &[f32],
        s0: Option<&[f32]>,
        ring0: Option<&[f32]>,
    ) -> Result<Vec<f32>, String> {
        let dm = self.gdn_dims()?;
        if xn.len() != t_len * dm.hidden
            || qkv.len() != t_len * dm.conv_ch()
            || z.len() != t_len * dm.v_len()
        {
            return Err("GDN: 입력 길이 계약 위반(gather 진입)".into());
        }
        self.ensure_gdn_bufs(t_len)?;
        // SAFETY: f32 슬라이스 → 바이트 뷰(길이·정렬 일치).
        let b =
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        self.cc.h2d(self.dgxn, b(xn))?;
        self.cc.h2d(self.dqkv, b(qkv))?;
        self.cc.h2d(self.dzv, b(z))?;
        if let Some(s) = s0 {
            // SAFETY: 층 슬라이스 오프셋 — 경계 내.
            let off = self.dgst + (layer * dm.h_v * 128 * 128) as u64 * 4;
            self.cc.h2d(off, b(s))?;
        }
        if let Some(r) = ring0 {
            // SAFETY: 층 슬라이스 오프셋 — 경계 내.
            let off = self.dring + (layer * 3 * dm.conv_ch()) as u64 * 4;
            self.cc.h2d(off, b(r))?;
        }
        self.gdn_chain_dev(layer, t_len, true)?;
        let mut ob = vec![0u8; t_len * dm.v_len() * 4];
        self.cc.d2h(&mut ob, self.dgate)?;
        self.cc.sync()?;
        // SAFETY: d2h 완료 후 재해석.
        Ok(unsafe {
            std::slice::from_raw_parts(ob.as_ptr() as *const f32, t_len * dm.v_len()).to_vec()
        })
    }

    /// GDN 체인 중간 산출 전량 판독(검증층 진단 — 단계별 값 판정·링
    /// 회전·순열 방향·소거 순서 국소화). 직전 gdn_chain_host 실행의
    /// 잔류 버퍼를 읽는다(3층 분리: 판독만, 계산 없음).
    pub fn gdn_mids_host(&mut self, layer: usize, t_len: usize) -> Result<GdnMids, String> {
        let dm = self.gdn_dims()?;
        if self.gdn_t_cap < t_len {
            return Err(format!(
                "GDN: mids 판독 전 체인 실행 필요(t={t_len} > cap={})",
                self.gdn_t_cap
            ));
        }
        let (kl, vl, cch) = (dm.k_len(), dm.v_len(), dm.conv_ch());
        let take = |n: usize, src: CUdeviceptr| -> Result<Vec<f32>, String> {
            let mut buf = vec![0u8; n * 4];
            self.cc.d2h(&mut buf, src)?;
            // SAFETY: d2h 완료 후 재해석(길이·정렬 일치).
            Ok(unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const f32, n).to_vec() })
        };
        let conv_q = take(t_len * kl, self.dgq)?;
        let conv_k = take(t_len * kl, self.dgk)?;
        let conv_v = take(t_len * vl, self.dgv)?;
        let q2 = take(t_len * kl, self.dq2)?;
        let k2 = take(t_len * kl, self.dk2)?;
        let v2 = take(t_len * vl, self.dv2)?;
        let bg = take(t_len * dm.bg_len(), self.dbg)?;
        let o_lc = take(t_len * vl, self.dgo)?;
        // SAFETY: 층 슬라이스 오프셋 — 경계 내.
        let ring_post = take(3 * cch, self.dring + (layer * 3 * cch) as u64 * 4)?;
        // SAFETY: 층 슬라이스 오프셋 — 경계 내.
        let st_post = take(
            dm.h_v * 128 * 128,
            self.dgst + (layer * dm.h_v * 128 * 128) as u64 * 4,
        )?;
        self.cc.sync()?;
        Ok(GdnMids {
            conv_q,
            conv_k,
            conv_v,
            q2,
            k2,
            v2,
            bg,
            o_lc,
            ring_post,
            st_post,
        })
    }
}
