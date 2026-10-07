//! DeepSeek-V4 mHC(멀티 하이퍼커넥션)+Sinkhorn 모듈층 — plans/130 B3,
//! 2026-10-05.
//!
//! [계약] crates/core/src/deepseek4/stages/hc.rs 가 산술 원천(유일 기준
//! — plans/124 §6). 본 모듈은 mHC 4스트림 스테이지를 커널 체인
//! (assets/ds4_hc.cu)으로 미러한다:
//! - hc_pre L115-152: mix GEMV(fn[24,16384]·X) → rsqrt(rms_scale — 정규화는
//!   **믹스에만** 곱한다) → 스플릿+Sinkhorn(20iter) → y=Σ pre_j·X[j]
//!   (원본 X, bf16 경계) → (y, post, comb) 반환.
//! - hc_post L155-181: X'[j] = post_j·F + Σ_k comb[j,k]·X[k] (bf16 경계).
//! - hc_head L185-214(frame.rs L173-184 사용): fn[4,16384]·base[4]·
//!   scale[1] 헤드 변형 → out=Σ pre_j·X_j.
//!
//! [정합 목표 — 비트동일] hc 산술은 전부 f32 연산별 반올림 + f64 트윈
//! exp(crate::ops::exp_cr)이라 core 미러와 비트동일이 가능하다(판정은
//! 값 maxdiff + 비트 불일치 수 — argmax 금지 §6):
//! - mix 점적산·rms 제곱합·적용 누산: 1스레드 순차 f32(mul+add 무-FMA
//!   — assets/ds4_hc.cu 헤드 [빌드 계약] -fmad=false 블록).
//! - rms_scale(deepseek4/ops.rs L48-55)은 **순수 f32 순차 제곱합** —
//!   Flash-Next grouped_rms(qwen4exp sq_sum f64 32세그먼트)와 다른
//!   deepseek4 계약. 커널도 트리 환원 없이 1스레드 순차로 미러.
//! - Sinkhorn(hc_split_sinkhorn L31-112): 토큰당 1스레드 스칼라 직이식
//!   — softmax_rows → +eps → /(col+eps) → 19×{/(row+eps); /(col+eps)}
//!   순서 그대로(순서 민감성은 프루브 음성대조로 고정).
//! - bf16 경계: deepseek4/ops.rs bf16_round L17-22 비트 경로 트윈.
//!
//! [토큰축 배치] 전 토큰 1회 체인 — 호출당 h2d(잔류 전체) 1회 + 커널
//! 4-5회 + d2h(y·post·comb). 그리드 토큰축 gy=blockIdx.y(결함 5호 —
//! T>1 행 미실행 가드). Sinkhorn·rsqrt 는 1블록=1토큰(스칼라 순차 —
//! 비트동일 우선, k-분할은 sm_80 실측 후 재판정 — 원장 18호 계급).
//!
//! [가중치 소스] EXL3 Vision-Exp(D:/models/DeepSeek-V4-Flash-Vision-Exp
//! -exl3-3.04bpw)의 hc_{attn,ffn}_{fn,base,scale}·hc_head_{fn,base,scale}
//! 은 전부 F32 plain 텐서(실측 fn[24,16384]·base[24]·scale[3],
//! 헤드 fn[4,16384]·base[4]·scale[1]) — trellis 없이 오프셋 직독
//! (프루브 ds4_hc_cuda_probe). fp32 전용 스테이지(보고서 §9.4).
//!
//! [CMP 170HX(sm_80, GA100 70SM, HBM2e ~1.5TB/s) 설계 근거 — plans/124 §0]
//! 층당·kind당·토큰당 fn[24×16384] f32 ≈1.57MB 스트리밍 → HBM2e 하한
//! ~1.0µs. 소형 런치(mix 24출력·sinkhorn 1스레드)는 정합 우선 — 개발기
//! (RTX 4070 SUPER) 타이밍 금지, 속도 칸 '측정 대기 sm_80'.
//!
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 금지 —
//! scripts/cuda_probe_shim.rs 단독 컴파일 대상.

use crate::rawcuda::ctx::CudaCtx;
use crate::rawcuda::exl3_cuda::{JParser, JVal};
use crate::rawcuda::ffi::CUdeviceptr;
use std::collections::HashMap;

/// hc_head 등록 키 il 자리(센티넬 — hc_head_* 는 층 무관 최상위 가중치).
pub const DS4_HC_HEAD_IL: usize = usize::MAX;

/// mHC 형상 — config.json(Deepseek4Config 계약 키)에서 유도. 형상은
/// 명시 등록, 추정 금지(MtpDims/HcDims 방식 계승). Vision-Exp 실측
/// (2026-10-05): hc_mult=4 · hidden_size=4096 · rms_norm_eps=1e-20 ·
/// hc_eps=1e-6 · hc_sinkhorn_iters=20.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ds4HcDims {
    /// 스트림 수(hc_mult).
    pub hc: usize,
    /// 토큰 폭(hidden_size).
    pub n_embd: usize,
    /// RMSNorm eps(rms_norm_eps — **hc_eps 와 별개**, Vision-Exp 1e-20).
    pub norm_eps: f32,
    /// Sinkhorn/시그모이드 eps(hc_eps, 1e-6).
    pub hc_eps: f32,
    /// Sinkhorn 반복 수(hc_sinkhorn_iters, 20).
    pub iters: usize,
}

impl Ds4HcDims {
    /// hc_dim = hc·n_embd(잔류 행 폭·믹스 입력 폭).
    pub fn hc_dim(&self) -> usize {
        self.hc * self.n_embd
    }

    /// 본체 믹스 행 수 (2+hc)·hc(=24 @hc4).
    pub fn mix_rows(&self) -> usize {
        (2 + self.hc) * self.hc
    }

    /// config.json 본문 → 형상. 키 누락은 Err(추정 금지 — 결함 1호 정신).
    pub fn from_config(cfg: &str) -> Result<Self, String> {
        let v = JParser {
            b: cfg.as_bytes(),
            p: 0,
        }
        .parse()?;
        let num = |k: &str| {
            v.get(k)
                .and_then(JVal::as_f64)
                .ok_or_else(|| format!("config.json: {k} 없음"))
        };
        let hc = num("hc_mult")? as usize;
        let n_embd = num("hidden_size")? as usize;
        let norm_eps = num("rms_norm_eps")? as f32;
        let hc_eps = num("hc_eps")? as f32;
        let iters = num("hc_sinkhorn_iters")? as usize;
        if hc == 0 || hc > 8 {
            return Err(format!(
                "ds4-hc: 스트림 수 {hc} — 1..=8 계약(커널 스칼라 배열 상한)"
            ));
        }
        if n_embd == 0 {
            return Err(format!("ds4-hc: hidden_size={n_embd} — 양수 계약"));
        }
        if hc_eps <= 0.0 || norm_eps < 0.0 {
            return Err(format!(
                "ds4-hc: eps hc={hc_eps} norm={norm_eps} — 양수 계약"
            ));
        }
        if iters == 0 || iters > 1000 {
            return Err(format!("ds4-hc: iters={iters} — 1..=1000 계약"));
        }
        Ok(Ds4HcDims {
            hc,
            n_embd,
            norm_eps,
            hc_eps,
            iters,
        })
    }
}

/// 등록된 hc 파라미터 1세트(디바이스 상주 f32 — 해제 없이 누적,
/// 프로브 수명 계약). base/scale 까지 전부 디바이스 상주.
struct Ds4HcP {
    dfn: CUdeviceptr,    // [mix_rows·hc_dim] 또는 [hc·hc_dim](헤드)
    dbase: CUdeviceptr,  // [24] / [4]
    dscale: CUdeviceptr, // [3] / [1]
    mix_rows: usize,     // 24(본체) / hc(헤드)
}

/// DeepSeek-V4 mHC 모듈 — CudaCtx 단일 소유(단일 상주 원칙). 커널은
/// assets/ds4_hc.fatbin.
pub struct Ds4HcCuda {
    /// 디바이스 컨텍스트(모듈 단독 소유 — 병렬 작업 계약상 타 모듈
    /// 파일과 버퍼를 공유하지 않는다).
    pub cc: CudaCtx,
    /// 형상(new 등록).
    pub dims: Ds4HcDims,
    /// 파라미터 레지스트리 — 키 (il, kind). kind: "attn"|"ffn"|
    /// "head"((DS4_HC_HEAD_IL,"head")).
    params: HashMap<(usize, String), Ds4HcP>,
    // ── 작업 버퍼(토큰 용량 cap_t 까지 재사용 — G2 ensure 계약) ──
    dx: CUdeviceptr,     // [t][hc_dim] 잔류/헤드 입력(비정규 원본)
    dmix: CUdeviceptr,   // [t][mix_rows] 원시 점적산
    drsqrt: CUdeviceptr, // [t] RMS 배율
    dpre: CUdeviceptr,   // [t][hc]
    dpost: CUdeviceptr,  // [t][hc]
    dcomb: CUdeviceptr,  // [t][hc·hc]
    dy: CUdeviceptr,     // [t][n_embd] hc_pre/hc_head 출력
    df: CUdeviceptr,     // [t][n_embd] hc_post 서브층 출력 F
    dyhc: CUdeviceptr,   // [t][hc_dim] hc_post 출력
    cap_t: usize,
}

/// f32 슬라이스 → LE 바이트(모듈 h2d 규격 — hc_cuda f32_bytes 패턴).
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

impl Ds4HcCuda {
    /// ds4_hc.fatbin 자산 해석 — LLM170_CUDA_DS4_HC_FATBIN_PATH 오버라이드
    /// 우선(자산 경로 오버라이드일 뿐 계산 경로 분기 아님).
    fn ds4_hc_fatbin_bytes() -> Result<Vec<u8>, String> {
        const ENV: &str = "LLM170_CUDA_DS4_HC_FATBIN_PATH";
        const REL: &[&str] = &[
            "crates/backend-gpu/src/rawcuda/assets/ds4_hc.fatbin",
            "src/rawcuda/assets/ds4_hc.fatbin",
        ];
        if let Some(p) = std::env::var_os(ENV) {
            return std::fs::read(&p).map_err(|e| format!("{ENV}({p:?}) 읽기 실패: {e}"));
        }
        for r in REL {
            if let Ok(b) = std::fs::read(r) {
                return Ok(b);
            }
        }
        Err(format!(
            "ds4_hc.fatbin 없음 — scripts/build_cuda.bat 실행 또는 {ENV} 지정 (탐색: {REL:?})"
        ))
    }

    /// 개방 — 컨텍스트 + ds4_hc.fatbin 적재(커널 5종).
    pub fn new(dims: Ds4HcDims) -> Result<Self, String> {
        let image = Self::ds4_hc_fatbin_bytes()?;
        let mut cc = CudaCtx::new()?;
        let _g = cc.guard()?;
        cc.load_fatbin(
            "ds4hc",
            &image,
            &[
                "llm170_ds4_hc_mix_gemv",
                "llm170_ds4_hc_rms_scale",
                "llm170_ds4_hc_split_sinkhorn",
                "llm170_ds4_hc_pre_apply",
                "llm170_ds4_hc_post_apply",
            ],
        )?;
        Ok(Ds4HcCuda {
            cc,
            dims,
            params: HashMap::new(),
            dx: 0,
            dmix: 0,
            drsqrt: 0,
            dpre: 0,
            dpost: 0,
            dcomb: 0,
            dy: 0,
            df: 0,
            dyhc: 0,
            cap_t: 0,
        })
    }

    /// 디바이스 이름(프로브 보고용).
    pub fn device_name(&self) -> &str {
        &self.cc.device_name
    }

    /// hc 파라미터 등록 — fn · base · scale(전부 f32). kind="attn"|"ffn"
    /// → fn[(2+hc)·hc·hc_dim]·base[24]·scale[3], kind="head" →
    /// fn[hc·hc_dim]·base[hc]·scale[1]. 길이는 등록 시 전부 검증(형상
    /// 추정 금지). 재등록은 이전 디바이스 버퍼를 해제 후 교체.
    pub fn register(
        &mut self,
        il: usize,
        kind: &str,
        fns: &[f32],
        base: &[f32],
        scale: &[f32],
    ) -> Result<(), String> {
        let (hc, n) = (self.dims.hc, self.dims.n_embd);
        let hcd = hc * n;
        let head = kind == "head";
        if !head && kind != "attn" && kind != "ffn" {
            return Err(format!("ds4-hc: kind={kind} — attn|ffn|head 계약"));
        }
        let mix_rows = if head { hc } else { (2 + hc) * hc };
        let (want_fn, want_base, want_scale) = if head {
            (hc * hcd, hc, 1)
        } else {
            ((2 + hc) * hc * hcd, (2 + hc) * hc, 3)
        };
        if fns.len() != want_fn {
            return Err(format!("ds4-hc: fn {} != {want_fn}", fns.len()));
        }
        if base.len() != want_base {
            return Err(format!("ds4-hc: base {} != {want_base}", base.len()));
        }
        if scale.len() != want_scale {
            return Err(format!("ds4-hc: scale {} != {want_scale}", scale.len()));
        }
        let _g = self.cc.guard()?;
        if let Some(old) = self.params.remove(&(il, kind.to_string())) {
            self.cc.free(old.dfn)?;
            self.cc.free(old.dbase)?;
            self.cc.free(old.dscale)?;
        }
        let upl = |v: &[f32]| -> Result<CUdeviceptr, String> {
            let d = self.cc.alloc(v.len() * 4)?;
            // SAFETY: v는 f32 슬라이스 — 길이 일치 바이트 뷰 변환.
            let b = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
            self.cc.h2d(d, b)?;
            Ok(d)
        };
        self.params.insert(
            (il, kind.to_string()),
            Ds4HcP {
                dfn: upl(fns)?,
                dbase: upl(base)?,
                dscale: upl(scale)?,
                mix_rows,
            },
        );
        Ok(())
    }

    /// 토큰 용량 버퍼 보장(확장 시에만 재할당 — ensure_norm_bufs 계약).
    fn ensure_bufs(&mut self, t: usize) -> Result<(), String> {
        if t <= self.cap_t {
            return Ok(());
        }
        let _g = self.cc.guard()?;
        if self.cap_t > 0 {
            for d in [
                self.dx,
                self.dmix,
                self.drsqrt,
                self.dpre,
                self.dpost,
                self.dcomb,
                self.dy,
                self.df,
                self.dyhc,
            ] {
                self.cc.free(d)?;
            }
        }
        let (hc, n) = (self.dims.hc, self.dims.n_embd);
        self.dx = self.cc.alloc(t * hc * n * 4)?;
        self.dmix = self.cc.alloc(t * self.dims.mix_rows() * 4)?;
        self.drsqrt = self.cc.alloc(t * 4)?;
        self.dpre = self.cc.alloc(t * hc * 4)?;
        self.dpost = self.cc.alloc(t * hc * 4)?;
        self.dcomb = self.cc.alloc(t * hc * hc * 4)?;
        self.dy = self.cc.alloc(t * n * 4)?;
        self.df = self.cc.alloc(t * n * 4)?;
        self.dyhc = self.cc.alloc(t * hc * n * 4)?;
        self.cap_t = t;
        Ok(())
    }

    /// 믹스 체인 공통(hc_pre·hc_head) — 점적산 → rsqrt → 스플릿.
    /// dmix/drsqrt/dpre/dpost/dcomb 까지 기입하고 돌아온다(호출부가
    /// pre_apply 로 마무리).
    fn mix_chain(&mut self, key: &(usize, String), x: &[Vec<f32>]) -> Result<usize, String> {
        let (hc, n) = (self.dims.hc, self.dims.n_embd);
        let hcd = hc * n;
        let t = x.len();
        if t == 0 {
            return Err("ds4-hc: 빈 x".into());
        }
        for (ti, r) in x.iter().enumerate() {
            if r.len() != hcd {
                return Err(format!("ds4-hc: x[{ti}].len={} != hc_dim {hcd}", r.len()));
            }
        }
        let Some(p) = self.params.get(key) else {
            return Err(format!(
                "ds4-hc: 미등록 파라미터 (il={}, kind={}) — register 먼저",
                key.0, key.1
            ));
        };
        let (dfn, dbase, dscale, mix_rows) = (p.dfn, p.dbase, p.dscale, p.mix_rows);
        let head = mix_rows == hc;
        self.ensure_bufs(t)?;
        let _g = self.cc.guard()?;
        // 잔류 업로드 [t][hcd](1회 — 토큰축 배치 원장).
        let mut flat = Vec::with_capacity(t * hcd);
        for r in x {
            flat.extend_from_slice(r);
        }
        // SAFETY: flat은 f32 벡터 — 바이트 뷰 변환(길이 일치).
        let xb = unsafe { std::slice::from_raw_parts(flat.as_ptr() as *const u8, flat.len() * 4) };
        self.cc.h2d(self.dx, xb)?;

        // 1) mix GEMV — grid(ceil(mix_rows/128), t) × 128.
        let (mut a0, mut a1, mut a2) = (self.dx, dfn, self.dmix);
        let (mut a3, mut a4) = (hcd as i32, mix_rows as i32);
        let mut am: [*mut std::ffi::c_void; 5] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
            (&mut a3) as *mut _ as *mut _,
            (&mut a4) as *mut _ as *mut _,
        ];
        self.cc.launch(
            self.cc.function("llm170_ds4_hc_mix_gemv")?,
            mix_rows.div_ceil(128) as u32,
            t as u32,
            128,
            &mut am,
        )?;

        // 2) RMS 배율 — grid(t) × 1(순수 f32 순차 — ops.rs rms_scale 미러).
        let (mut b0, mut b1) = (self.dx, self.drsqrt);
        let (mut b2, mut b3) = (hcd as i32, self.dims.norm_eps);
        let mut ar: [*mut std::ffi::c_void; 4] = [
            (&mut b0) as *mut _ as *mut _,
            (&mut b1) as *mut _ as *mut _,
            (&mut b2) as *mut _ as *mut _,
            (&mut b3) as *mut _ as *mut _,
        ];
        self.cc.launch(
            self.cc.function("llm170_ds4_hc_rms_scale")?,
            t as u32,
            1,
            1,
            &mut ar,
        )?;

        // 3) 스플릿+Sinkhorn — grid(t) × 1(토큰당 1스레드 스칼라 직이식).
        let (mut c0, mut c1, mut c2, mut c3) = (self.dmix, self.drsqrt, dscale, dbase);
        let (mut c4, mut c5, mut c6) = (self.dpre, self.dpost, self.dcomb);
        let (mut c7, mut c8, mut c9, mut c10, mut c11) = (
            hc as i32,
            self.dims.iters as i32,
            self.dims.hc_eps,
            i32::from(head),
            mix_rows as i32,
        );
        let mut as_: [*mut std::ffi::c_void; 12] = [
            (&mut c0) as *mut _ as *mut _,
            (&mut c1) as *mut _ as *mut _,
            (&mut c2) as *mut _ as *mut _,
            (&mut c3) as *mut _ as *mut _,
            (&mut c4) as *mut _ as *mut _,
            (&mut c5) as *mut _ as *mut _,
            (&mut c6) as *mut _ as *mut _,
            (&mut c7) as *mut _ as *mut _,
            (&mut c8) as *mut _ as *mut _,
            (&mut c9) as *mut _ as *mut _,
            (&mut c10) as *mut _ as *mut _,
            (&mut c11) as *mut _ as *mut _,
        ];
        self.cc.launch(
            self.cc.function("llm170_ds4_hc_split_sinkhorn")?,
            t as u32,
            1,
            1,
            &mut as_,
        )?;
        Ok(t)
    }

    /// hc_pre 미러(hc.rs L115-152) — 층 입력 믹스. 반환 (y[t][n],
    /// post[t][hc], comb[t][hc·hc] row-major). y 는 bf16 경계값.
    pub fn hc_pre(
        &mut self,
        il: usize,
        kind: &str,
        x: &[Vec<f32>],
    ) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>), String> {
        if kind != "attn" && kind != "ffn" {
            return Err(format!("ds4-hc: kind={kind} — attn|ffn 계약"));
        }
        let (hc, n) = (self.dims.hc, self.dims.n_embd);
        let t = self.mix_chain(&(il, kind.to_string()), x)?;
        let _g = self.cc.guard()?;
        // 4) y = Σ pre_j·X[j] — grid(ceil(n/256), t) × 256.
        let (mut d0, mut d1, mut d2) = (self.dx, self.dpre, self.dy);
        let (mut d3, mut d4) = (n as i32, hc as i32);
        let mut ay: [*mut std::ffi::c_void; 5] = [
            (&mut d0) as *mut _ as *mut _,
            (&mut d1) as *mut _ as *mut _,
            (&mut d2) as *mut _ as *mut _,
            (&mut d3) as *mut _ as *mut _,
            (&mut d4) as *mut _ as *mut _,
        ];
        self.cc.launch(
            self.cc.function("llm170_ds4_hc_pre_apply")?,
            n.div_ceil(256) as u32,
            t as u32,
            256,
            &mut ay,
        )?;
        self.cc.sync()?;
        // 판독 — y[t][n] + post[t][hc] + comb[t][hc·hc].
        let mut yb = vec![0u8; t * n * 4];
        self.cc.d2h(&mut yb, self.dy)?;
        // SAFETY: d2h 완료 버퍼 재해석(길이·정렬 일치 — G2 판독 패턴).
        let yf = unsafe { std::slice::from_raw_parts(yb.as_ptr() as *const f32, t * n) };
        let y: Vec<Vec<f32>> = yf.chunks_exact(n).map(|c| c.to_vec()).collect();
        let mut pb = vec![0u8; t * hc * 4];
        self.cc.d2h(&mut pb, self.dpost)?;
        // SAFETY: d2h 완료 버퍼 재해석(길이·정렬 일치).
        let pf = unsafe { std::slice::from_raw_parts(pb.as_ptr() as *const f32, t * hc) };
        let post: Vec<Vec<f32>> = pf.chunks_exact(hc).map(|c| c.to_vec()).collect();
        let mut cb = vec![0u8; t * hc * hc * 4];
        self.cc.d2h(&mut cb, self.dcomb)?;
        // SAFETY: d2h 완료 버퍼 재해석(길이·정렬 일치).
        let cf = unsafe { std::slice::from_raw_parts(cb.as_ptr() as *const f32, t * hc * hc) };
        let comb: Vec<Vec<f32>> = cf.chunks_exact(hc * hc).map(|c| c.to_vec()).collect();
        Ok((y, post, comb))
    }

    /// hc_post 미러(hc.rs L155-181) — 층 출력 재확장. f[t][n] 서브층
    /// 출력, residual[t][hcd] 비정규 원본, post/comb 는 hc_pre 산출.
    /// 반환 X'[t][hcd](bf16 경계값).
    pub fn hc_post(
        &mut self,
        f: &[Vec<f32>],
        residual: &[Vec<f32>],
        post: &[Vec<f32>],
        comb: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, String> {
        let (hc, n) = (self.dims.hc, self.dims.n_embd);
        let hcd = hc * n;
        let t = f.len();
        if t == 0 || residual.len() != t || post.len() != t || comb.len() != t {
            return Err(format!(
                "ds4-hc: hc_post 형상 f={} res={} post={} comb={t}",
                f.len(),
                residual.len(),
                post.len()
            ));
        }
        for ti in 0..t {
            if f[ti].len() != n || residual[ti].len() != hcd {
                return Err(format!(
                    "ds4-hc: hc_post[{ti}] f={} res={} != {n}/{hcd}",
                    f[ti].len(),
                    residual[ti].len()
                ));
            }
            if post[ti].len() != hc || comb[ti].len() != hc * hc {
                return Err(format!(
                    "ds4-hc: hc_post[{ti}] post={} comb={} != {hc}/{}",
                    post[ti].len(),
                    comb[ti].len(),
                    hc * hc
                ));
            }
        }
        self.ensure_bufs(t)?;
        let _g = self.cc.guard()?;
        let mut fflat = Vec::with_capacity(t * n);
        let mut rflat = Vec::with_capacity(t * hcd);
        let mut pflat = Vec::with_capacity(t * hc);
        let mut cflat = Vec::with_capacity(t * hc * hc);
        for ti in 0..t {
            fflat.extend_from_slice(&f[ti]);
            rflat.extend_from_slice(&residual[ti]);
            pflat.extend_from_slice(&post[ti]);
            cflat.extend_from_slice(&comb[ti]);
        }
        // SAFETY: f32 벡터 — 바이트 뷰 변환(길이 일치).
        let fb =
            unsafe { std::slice::from_raw_parts(fflat.as_ptr() as *const u8, fflat.len() * 4) };
        let rb =
            unsafe { std::slice::from_raw_parts(rflat.as_ptr() as *const u8, rflat.len() * 4) };
        let pb =
            unsafe { std::slice::from_raw_parts(pflat.as_ptr() as *const u8, pflat.len() * 4) };
        let cb =
            unsafe { std::slice::from_raw_parts(cflat.as_ptr() as *const u8, cflat.len() * 4) };
        self.cc.h2d(self.df, fb)?;
        self.cc.h2d(self.dx, rb)?;
        self.cc.h2d(self.dpost, pb)?;
        self.cc.h2d(self.dcomb, cb)?;

        // X'[j] = post_j·F + Σ_k comb[j,k]·X[k] — grid(ceil(hcd/256), t).
        let (mut e0, mut e1, mut e2, mut e3) = (self.df, self.dx, self.dpost, self.dcomb);
        let (mut e4, mut e5, mut e6) = (self.dyhc, n as i32, hc as i32);
        let mut ao: [*mut std::ffi::c_void; 7] = [
            (&mut e0) as *mut _ as *mut _,
            (&mut e1) as *mut _ as *mut _,
            (&mut e2) as *mut _ as *mut _,
            (&mut e3) as *mut _ as *mut _,
            (&mut e4) as *mut _ as *mut _,
            (&mut e5) as *mut _ as *mut _,
            (&mut e6) as *mut _ as *mut _,
        ];
        self.cc.launch(
            self.cc.function("llm170_ds4_hc_post_apply")?,
            hcd.div_ceil(256) as u32,
            t as u32,
            256,
            &mut ao,
        )?;
        self.cc.sync()?;
        let mut ob = vec![0u8; t * hcd * 4];
        self.cc.d2h(&mut ob, self.dyhc)?;
        // SAFETY: d2h 완료 버퍼 재해석(길이·정렬 일치).
        let of = unsafe { std::slice::from_raw_parts(ob.as_ptr() as *const f32, t * hcd) };
        Ok(of.chunks_exact(hcd).map(|c| c.to_vec()).collect())
    }

    /// hc_head 미러(hc.rs L185-214 — frame.rs L173-184 종결부). 등록 키
    /// (DS4_HC_HEAD_IL, "head"). 반환 y[t][n](bf16 경계값).
    pub fn hc_head(&mut self, x: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
        let (hc, n) = (self.dims.hc, self.dims.n_embd);
        let t = self.mix_chain(&(DS4_HC_HEAD_IL, "head".to_string()), x)?;
        let _g = self.cc.guard()?;
        let (mut d0, mut d1, mut d2) = (self.dx, self.dpre, self.dy);
        let (mut d3, mut d4) = (n as i32, hc as i32);
        let mut ay: [*mut std::ffi::c_void; 5] = [
            (&mut d0) as *mut _ as *mut _,
            (&mut d1) as *mut _ as *mut _,
            (&mut d2) as *mut _ as *mut _,
            (&mut d3) as *mut _ as *mut _,
            (&mut d4) as *mut _ as *mut _,
        ];
        self.cc.launch(
            self.cc.function("llm170_ds4_hc_pre_apply")?,
            n.div_ceil(256) as u32,
            t as u32,
            256,
            &mut ay,
        )?;
        self.cc.sync()?;
        let mut yb = vec![0u8; t * n * 4];
        self.cc.d2h(&mut yb, self.dy)?;
        // SAFETY: d2h 완료 버퍼 재해석(길이·정렬 일치).
        let yf = unsafe { std::slice::from_raw_parts(yb.as_ptr() as *const f32, t * n) };
        Ok(yf.chunks_exact(n).map(|c| c.to_vec()).collect())
    }
}
