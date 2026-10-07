//! Flash-Next HC(hyper-connection mix) 모듈층 — plans/124 G001(FNA)→FNC,
//! 2026-10-05.
//!
//! [계약] crates/core/src/qwen4exp/stages/hc.rs 가 산술 원천(유일 기준 —
//! plans/124 §6). 본 모듈은 hc_mix·hc_mix_head·hc_mix_nextn_head 3종
//! 진입점과 grouped RMSNorm 을 커널 체인(assets/exl3_fn_hc.cu)으로 미러
//! 한다. combine(res[s] += out·2σ(inject_s/4) — layers.rs hc_combine
//! L2577-2587)는 layers/forward 체인 소유(FNH 범위) — 본 모듈은
//! hc_mix 계약대로 (mixed, inject) 만 반환한다.
//!
//! [정합 목표 — 비트동일] hc 산술은 전부 f32 연산별 반올림 + f64 트윈
//! exp(ops.rs exp_cr)이라 core 미러와 비트동일이 가능하다:
//! - grouped rms: ops.rs sq_sum L11-31 의 32세그먼트 구조를 커널이
//!   정확히 재현(세그먼트 f32 순차 누산 → f64 순차 결합 — G3 norm 의
//!   트리 환원 계급과 달리 세그먼트 구조 자체를 미러).
//! - linear(down/up/inject): cpu.rs matmul L64-76 행별 f32 순차 누산
//!   (mul+add 무-FMA) — 1스레드=1출력 순차 루프와 동일 비트.
//! - silu(lo/hc)·게이트 sigmoid·스트림 평균: stages/hc.rs L58-79 ·
//!   ops.rs L127-133 공식 그대로(-fmad=false 빌드 — assets/exl3_fn_hc.cu
//!   헤드 [빌드 계약]).
//!   판정은 값 maxdiff(비트 불일치 수 보고) — argmax 판정 금지(§6).
//!
//! [토큰축 배치] hc.rs 헤드 원장: down/up/inject 전 토큰 1회 체인 —
//! 호출당 h2d(잔류 전체) 1회 + 커널 5-6회 + d2h 2회(mixed·inject).
//! 그리드 토큰축 gy=blockIdx.y(결함 5호 — T>1 행 미실행 가드).
//!
//! [가중치 소스] 등록은 f32 호스트 슬라이스(행우선 [n_out][n_in] —
//! ggml ne0=n_in 규약). EXL3 5.05bpw 의 실측 픽스처는
//! mtp_hyper_connection_mixer_patch.safetensors(F16 norm[10240]·
//! down[320,10240]·up[10240,320] — 헤더 실측 2026-10-05): F16→f32 는
//! 정확 변환이라 비트동일 판정을 유지한다. 본체 blk.{il}.hc_{kind}_*
//! trellis 선형(krate 7 — G2 상한 초과)은 REUSE(조건부) 항으로 FNB+
//! 스테이징 확정 후 등록 경로만 교체한다(산술 계약 불변).
//!
//! [CMP 170HX(sm_80, GA100 70SM, HBM2e ~1.5TB/s) 설계 근거 — plans/124 §0]
//! 층당·kind당·토큰당 가중 스트리밍 ≈26.4MB → HBM2e 하한 ~17.6µs.
//! up gemv (80,T)×128 은 T=1 에도 70SM 1웨이브 근접; down (3,T)·
//! inject (1,T) 소형 런치는 원장 18호 계급(정합 우선 — k-분할은
//! sm_80 실측 후 재판정). 속도 칸 '측정 대기 sm_80'(개발기 타이밍은
//! 판단 근거 아님).
//!
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 금지 —
//! scripts/cuda_probe_shim.rs 단독 컴파일 대상.

use crate::rawcuda::ctx::CudaCtx;
use crate::rawcuda::exl3_cuda::{JParser, JVal};
use crate::rawcuda::ffi::CUdeviceptr;
use std::collections::HashMap;

/// hc_mix_head 등록 키 il 자리(DI 센티널 — hc_mix_head 는 층 무관
/// output_hc_* 가중치라 모듈 등록 키만을 위해 존재).
pub const HC_HEAD_IL: usize = usize::MAX;

/// HC 형상 — config.json text_config 에서 유도(형상은 명시 등록,
/// 추정 금지 — MtpDims 방식 계승). Flash-Next 실측(2026-10-05,
/// D:/models/Qwen3.8-Flash-Next-exl3-5.05bpw/config.json):
/// hc_count=4 · hc_lowrank=320 · hidden_size=2560 · eps=1e-6.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HcDims {
    /// 스트림 수(hyper_connection.count).
    pub hc: usize,
    /// 토큰 폭(hidden_size).
    pub n_embd: usize,
    /// 저랭크 폭(hyper_connection.low_rank).
    pub low_rank: usize,
    /// RMSNorm eps(rms_norm_eps).
    pub eps: f32,
}

impl HcDims {
    /// hc_dim = hc·n_embd(잔류 행 폭·up 출력·norm 길이).
    pub fn hc_dim(&self) -> usize {
        self.hc * self.n_embd
    }

    /// config.json 본문 → 형상. 키 누락은 Err(추정 금지 — 결함 1호 정신).
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
        let hc = num("hc_count")? as usize;
        let low_rank = num("hc_lowrank")? as usize;
        let n_embd = num("hidden_size")? as usize;
        let eps = num("rms_norm_eps")? as f32;
        if hc == 0 || hc > 8 {
            return Err(format!("hc: 스트림 수 {hc} — 1..=8 계약(블록 32세그먼트)"));
        }
        if n_embd == 0 || !n_embd.is_multiple_of(32) {
            return Err(format!(
                "hc: n_embd={n_embd} — 32 배수 계약(sq_sum 세그먼트)"
            ));
        }
        if low_rank == 0 || eps <= 0.0 {
            return Err(format!("hc: low_rank={low_rank} eps={eps} — 양수 계약"));
        }
        Ok(HcDims {
            hc,
            n_embd,
            low_rank,
            eps,
        })
    }
}

/// 등록된 믹서 1세트(디바이스 상주 f32 — 해제 없이 누적, 프로브 수명 계약).
struct HcMixer {
    dnorm: CUdeviceptr,   // [hc_dim]
    ddown: CUdeviceptr,   // [low_rank][hc_dim]
    dup: CUdeviceptr,     // [hc_dim][low_rank]
    dinject: CUdeviceptr, // [hc][hc_dim] — 0이면 inject 없음(head·nextn_head)
}

/// Flash-Next HC 믹서 모듈 — CudaCtx 단일 소유(단일 상주 원칙,
/// 2026-10-04 동결 사고 계약). 커널은 assets/exl3_fn_hc.fatbin.
pub struct HcCuda {
    /// 디바이스 컨텍스트(모듈 단독 소유 — 병렬 작업 계약상 타 모듈
    /// 파일과 버퍼를 공유하지 않는다).
    pub cc: CudaCtx,
    /// 형상(new 등록).
    pub dims: HcDims,
    /// 믹서 레지스트리 — 키 (il, kind). kind: "attn"|"ffn"|
    /// "nextn_head"|(HC_HEAD_IL,"head").
    mixers: HashMap<(usize, String), HcMixer>,
    // ── 작업 버퍼(토큰 용량 cap_t 까지 재사용 — G2 ensure 계약) ──
    dx: CUdeviceptr,    // [t][hc_dim] 잔류 입력
    dxn: CUdeviceptr,   // [t][hc_dim] grouped rms 산출
    dlo: CUdeviceptr,   // [t][low_rank] down 산출(→silu 제자리)
    dinj: CUdeviceptr,  // [t][hc] inject 산출
    dgate: CUdeviceptr, // [t][hc_dim] up 산출(→게이트 적산 제자리)
    dmix: CUdeviceptr,  // [t][n_embd] 스트림 평균 산출
    cap_t: usize,
}

/// f32 슬라이스 → LE 바이트(모듈 h2d 규격 — norm nw_bytes 패턴).
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

impl HcCuda {
    /// exl3_fn_hc.fatbin 자산 해석 — LLM170_CUDA_FN_HC_FATBIN_PATH
    /// 오버라이드 우선(자산 경로 오버라이드일 뿐 계산 경로 분기 아님).
    fn hc_fatbin_bytes() -> Result<Vec<u8>, String> {
        const ENV: &str = "LLM170_CUDA_FN_HC_FATBIN_PATH";
        const REL: &[&str] = &[
            "crates/backend-gpu/src/rawcuda/assets/exl3_fn_hc.fatbin",
            "src/rawcuda/assets/exl3_fn_hc.fatbin",
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
            "exl3_fn_hc.fatbin 없음 — scripts/build_cuda.bat 실행 또는 {ENV} 지정 (탐색: {REL:?})"
        ))
    }

    /// 개방 — 컨텍스트 + exl3_fn_hc.fatbin 적재(커널 4종).
    pub fn new(dims: HcDims) -> Result<Self, String> {
        let image = Self::hc_fatbin_bytes()?;
        let mut cc = CudaCtx::new()?;
        let _g = cc.guard()?;
        cc.load_fatbin(
            "exl3fnhc",
            &image,
            &[
                "llm170_fn_hc_grouped_rms",
                "llm170_fn_hc_gemv",
                "llm170_fn_hc_silu_scaled",
                "llm170_fn_hc_gate_mean",
            ],
        )?;
        Ok(HcCuda {
            cc,
            dims,
            mixers: HashMap::new(),
            dx: 0,
            dxn: 0,
            dlo: 0,
            dinj: 0,
            dgate: 0,
            dmix: 0,
            cap_t: 0,
        })
    }

    /// 디바이스 이름(프로브 보고용).
    pub fn device_name(&self) -> &str {
        &self.cc.device_name
    }

    /// 믹서 등록 — norm [hc_dim] · down [low_rank][hc_dim] ·
    /// up [hc_dim][low_rank] · inject Option [hc][hc_dim](없으면
    /// hc_mix_head/hc_mix_nextn_head 계약). 길이는 등록 시 전부 검증
    /// (형상 추정 금지). 재등록은 이전 디바이스 버퍼를 해제 후 교체.
    pub fn register(
        &mut self,
        il: usize,
        kind: &str,
        norm: &[f32],
        down: &[f32],
        up: &[f32],
        inject: Option<&[f32]>,
    ) -> Result<(), String> {
        let (hc, n, lr) = (self.dims.hc, self.dims.n_embd, self.dims.low_rank);
        let hcn = hc * n;
        if norm.len() != hcn {
            return Err(format!("hc: norm {} != hc_dim {hcn}", norm.len()));
        }
        if down.len() != lr * hcn {
            return Err(format!("hc: down {} != {lr}×{hcn}", down.len()));
        }
        if up.len() != hcn * lr {
            return Err(format!("hc: up {} != {hcn}×{lr}", up.len()));
        }
        if let Some(wi) = inject
            && wi.len() != hc * hcn
        {
            return Err(format!("hc: inject {} != {hc}×{hcn}", wi.len()));
        }
        let _g = self.cc.guard()?;
        if let Some(old) = self.mixers.remove(&(il, kind.to_string())) {
            self.cc.free(old.dnorm)?;
            self.cc.free(old.ddown)?;
            self.cc.free(old.dup)?;
            if old.dinject != 0 {
                self.cc.free(old.dinject)?;
            }
        }
        let upl = |v: &[f32]| -> Result<CUdeviceptr, String> {
            let d = self.cc.alloc(v.len() * 4)?;
            // SAFETY: v는 f32 슬라이스 — 길이 일치 바이트 뷰 변환.
            let b = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
            self.cc.h2d(d, b)?;
            Ok(d)
        };
        let dinject = match inject {
            Some(wi) => upl(wi)?,
            None => 0,
        };
        self.mixers.insert(
            (il, kind.to_string()),
            HcMixer {
                dnorm: upl(norm)?,
                ddown: upl(down)?,
                dup: upl(up)?,
                dinject,
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
                self.dx, self.dxn, self.dlo, self.dinj, self.dgate, self.dmix,
            ] {
                self.cc.free(d)?;
            }
        }
        let (hc, n, lr) = (self.dims.hc, self.dims.n_embd, self.dims.low_rank);
        self.dx = self.cc.alloc(t * hc * n * 4)?;
        self.dxn = self.cc.alloc(t * hc * n * 4)?;
        self.dlo = self.cc.alloc(t * lr * 4)?;
        self.dinj = self.cc.alloc(t * hc * 4)?;
        self.dgate = self.cc.alloc(t * hc * n * 4)?;
        self.dmix = self.cc.alloc(t * n * 4)?;
        self.cap_t = t;
        Ok(())
    }

    /// grouped RMSNorm 1행 — stages/hc.rs grouped_rms L14-21 미러.
    /// x=[hc_dim], w=[hc_dim](스트림 s 슬라이스 w[s·n,(s+1)·n)) →
    /// xn=[hc_dim]. FNA 스켈레톤 API(hc_grouped_rms(x, w))와 동일 형상.
    pub fn hc_grouped_rms(&mut self, x: &[f32], w: &[f32]) -> Result<Vec<f32>, String> {
        let (hc, n) = (self.dims.hc, self.dims.n_embd);
        if x.len() != hc * n {
            return Err(format!("hc: x.len={} != hc_dim {}", x.len(), hc * n));
        }
        if w.len() != hc * n {
            return Err(format!("hc: w.len={} != hc_dim {}", w.len(), hc * n));
        }
        self.ensure_bufs(1)?;
        let _g = self.cc.guard()?;
        let dnorm = self.cc.alloc(w.len() * 4)?;
        // SAFETY: x·w는 f32 슬라이스 — 바이트 뷰 변환(길이 일치).
        let xb = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) };
        let wb = unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len() * 4) };
        self.cc.h2d(self.dx, xb)?;
        self.cc.h2d(dnorm, wb)?;
        let (mut a0, mut a1, mut a2) = (self.dx, dnorm, self.dxn);
        let (mut a3, mut a4, mut a5) = (n as i32, hc as i32, self.dims.eps);
        let mut args: [*mut std::ffi::c_void; 6] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
            (&mut a3) as *mut _ as *mut _,
            (&mut a4) as *mut _ as *mut _,
            (&mut a5) as *mut _ as *mut _,
        ];
        self.cc.launch(
            self.cc.function("llm170_fn_hc_grouped_rms")?,
            1,
            hc as u32,
            32,
            &mut args,
        )?;
        self.cc.sync()?;
        let mut ob = vec![0u8; x.len() * 4];
        self.cc.d2h(&mut ob, self.dxn)?;
        self.cc.free(dnorm)?;
        // SAFETY: d2h 완료 버퍼 재해석(길이·정렬 일치 — G2 판독 패턴).
        Ok(unsafe { std::slice::from_raw_parts(ob.as_ptr() as *const f32, x.len()) }.to_vec())
    }

    /// hc_mix_ex 미러(stages/hc.rs L25-86) — 디바이스 체인:
    /// grouped rms(전 토큰) → down(+inject 동일 입력) → silu(lo/hc) →
    /// up → 게이트 적용+스트림 평균. h2d 1회 + d2h 2회(mixed·inject).
    fn mix_ex(
        &mut self,
        key: &(usize, String),
        res_hc: &[Vec<f32>],
    ) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), String> {
        let (hc, n, lr) = (self.dims.hc, self.dims.n_embd, self.dims.low_rank);
        let hcn = hc * n;
        let t = res_hc.len();
        if t == 0 {
            return Err("hc: 빈 res_hc".into());
        }
        for (ti, r) in res_hc.iter().enumerate() {
            if r.len() != hcn {
                return Err(format!("hc: res_hc[{ti}].len={} != hc_dim {hcn}", r.len()));
            }
        }
        let Some(m) = self.mixers.get(key) else {
            return Err(format!(
                "hc: 미등록 믹서 (il={}, kind={}) — register 먼저",
                key.0, key.1
            ));
        };
        let (dnorm, ddown, dup, dinject) = (m.dnorm, m.ddown, m.dup, m.dinject);
        self.ensure_bufs(t)?;
        let _g = self.cc.guard()?;
        // 잔류 업로드 [t][hc_dim](1회 — 토큰축 배치 원장).
        let mut flat = Vec::with_capacity(t * hcn);
        for r in res_hc {
            flat.extend_from_slice(r);
        }
        // SAFETY: flat은 f32 벡터 — 바이트 뷰 변환(길이 일치).
        let xb = unsafe { std::slice::from_raw_parts(flat.as_ptr() as *const u8, flat.len() * 4) };
        self.cc.h2d(self.dx, xb)?;

        // 1) grouped RMSNorm — grid(t, hc) × 32.
        let (mut b0, mut b1, mut b2) = (self.dx, dnorm, self.dxn);
        let (mut b3, mut b4, mut b5) = (n as i32, hc as i32, self.dims.eps);
        let mut args: [*mut std::ffi::c_void; 6] = [
            (&mut b0) as *mut _ as *mut _,
            (&mut b1) as *mut _ as *mut _,
            (&mut b2) as *mut _ as *mut _,
            (&mut b3) as *mut _ as *mut _,
            (&mut b4) as *mut _ as *mut _,
            (&mut b5) as *mut _ as *mut _,
        ];
        self.cc.launch(
            self.cc.function("llm170_fn_hc_grouped_rms")?,
            t as u32,
            hc as u32,
            32,
            &mut args,
        )?;

        // 2) down: xn[t][hc_dim]·W_down[lr][hc_dim] → lo[t][lr].
        let (mut c0, mut c1, mut c2) = (self.dxn, ddown, self.dlo);
        let (mut c3, mut c4) = (hcn as i32, lr as i32);
        let mut ad: [*mut std::ffi::c_void; 5] = [
            (&mut c0) as *mut _ as *mut _,
            (&mut c1) as *mut _ as *mut _,
            (&mut c2) as *mut _ as *mut _,
            (&mut c3) as *mut _ as *mut _,
            (&mut c4) as *mut _ as *mut _,
        ];
        self.cc.launch(
            self.cc.function("llm170_fn_hc_gemv")?,
            lr.div_ceil(128) as u32,
            t as u32,
            128,
            &mut ad,
        )?;
        // 2b) inject(있으면): 동일 입력 xn — hc.rs D4 그룹 1호출 계약의
        // 모듈층 재현(독립 버퍼 발사 — 값은 동일).
        if dinject != 0 {
            let (mut d0, mut d1, mut d2) = (self.dxn, dinject, self.dinj);
            let (mut d3, mut d4) = (hcn as i32, hc as i32);
            let mut ai: [*mut std::ffi::c_void; 5] = [
                (&mut d0) as *mut _ as *mut _,
                (&mut d1) as *mut _ as *mut _,
                (&mut d2) as *mut _ as *mut _,
                (&mut d3) as *mut _ as *mut _,
                (&mut d4) as *mut _ as *mut _,
            ];
            self.cc.launch(
                self.cc.function("llm170_fn_hc_gemv")?,
                hc.div_ceil(128) as u32,
                t as u32,
                128,
                &mut ai,
            )?;
        }

        // 3) silu(lo/hc) — 제자리, grid(t·lr).
        let (mut e0, mut e1, mut e2) = (self.dlo, (t * lr) as i32, hc as f32);
        let mut as_: [*mut std::ffi::c_void; 3] = [
            (&mut e0) as *mut _ as *mut _,
            (&mut e1) as *mut _ as *mut _,
            (&mut e2) as *mut _ as *mut _,
        ];
        self.cc.launch(
            self.cc.function("llm170_fn_hc_silu_scaled")?,
            (t * lr).div_ceil(256) as u32,
            1,
            256,
            &mut as_,
        )?;

        // 4) up: lo[t][lr]·W_up[hc_dim][lr] → gate[t][hc_dim].
        let (mut f0, mut f1, mut f2) = (self.dlo, dup, self.dgate);
        let (mut f3, mut f4) = (lr as i32, hcn as i32);
        let mut au: [*mut std::ffi::c_void; 5] = [
            (&mut f0) as *mut _ as *mut _,
            (&mut f1) as *mut _ as *mut _,
            (&mut f2) as *mut _ as *mut _,
            (&mut f3) as *mut _ as *mut _,
            (&mut f4) as *mut _ as *mut _,
        ];
        self.cc.launch(
            self.cc.function("llm170_fn_hc_gemv")?,
            hcn.div_ceil(128) as u32,
            t as u32,
            128,
            &mut au,
        )?;

        // 5) 게이트 적용 + 스트림 평균 — grid(ceil(n/256), t) × 256.
        let (mut g0, mut g1, mut g2) = (self.dxn, self.dgate, self.dmix);
        let (mut g3, mut g4) = (n as i32, hc as i32);
        let mut ag: [*mut std::ffi::c_void; 5] = [
            (&mut g0) as *mut _ as *mut _,
            (&mut g1) as *mut _ as *mut _,
            (&mut g2) as *mut _ as *mut _,
            (&mut g3) as *mut _ as *mut _,
            (&mut g4) as *mut _ as *mut _,
        ];
        self.cc.launch(
            self.cc.function("llm170_fn_hc_gate_mean")?,
            n.div_ceil(256) as u32,
            t as u32,
            256,
            &mut ag,
        )?;
        self.cc.sync()?;

        // 판독 — mixed [t][n] + inject [t][hc](dinject==0이면 빈 행).
        let mut mb = vec![0u8; t * n * 4];
        self.cc.d2h(&mut mb, self.dmix)?;
        // SAFETY: d2h 완료 버퍼 재해석(길이·정렬 일치).
        let mf = unsafe { std::slice::from_raw_parts(mb.as_ptr() as *const f32, t * n) };
        let mixed: Vec<Vec<f32>> = mf.chunks_exact(n).map(|c| c.to_vec()).collect();
        let inject: Vec<Vec<f32>> = if dinject != 0 {
            let mut ib = vec![0u8; t * hc * 4];
            self.cc.d2h(&mut ib, self.dinj)?;
            // SAFETY: d2h 완료 버퍼 재해석(길이·정렬 일치).
            let iarr = unsafe { std::slice::from_raw_parts(ib.as_ptr() as *const f32, t * hc) };
            iarr.chunks_exact(hc).map(|c| c.to_vec()).collect()
        } else {
            vec![Vec::new(); t]
        };
        Ok((mixed, inject))
    }

    /// hc_mix 진입(attn|ffn) — stages/hc.rs hc_mix L89-106 미러.
    /// 반환 (mixed, inject) — combine(2σ(inject/4) 가중 잔차 가산)은
    /// layers.rs hc_combine L2577-2588 소유(FNH 체인).
    pub fn hc_mix(
        &mut self,
        il: usize,
        kind: &str,
        res_hc: &[Vec<f32>],
    ) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), String> {
        if kind != "attn" && kind != "ffn" {
            return Err(format!("hc: kind={kind} — attn|ffn 계약(hc.rs L95)"));
        }
        self.mix_ex(&(il, kind.to_string()), res_hc)
    }

    /// 출력 헤드용 HC mix(inject 없음) — stages/hc.rs hc_mix_head
    /// L126-133(output_hc_{norm,down,up}) 미러. 등록 키 (HC_HEAD_IL,
    /// "head").
    pub fn hc_mix_head(&mut self, res_hc: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
        let (out, _) = self.mix_ex(&(HC_HEAD_IL, "head".to_string()), res_hc)?;
        Ok(out)
    }

    /// MTP 드래프트 종결 HC mix — stages/hc.rs hc_mix_nextn_head
    /// L108-124(blk.{il}.nextn.hc_head_{norm,down,up} — 본체 output_hc
    /// 와 별개 가중치, plans/109 P15②) 미러. 등록 키 (il, "nextn_head").
    /// EXL3 픽스처는 mtp_hyper_connection_mixer_patch.safetensors
    /// (mtp.hyper_connection_mixer.* 3텐서 — F16).
    pub fn hc_mix_nextn_head(
        &mut self,
        il: usize,
        res_hc: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, String> {
        let (out, _) = self.mix_ex(&(il, "nextn_head".to_string()), res_hc)?;
        Ok(out)
    }
}
