//! Flash-Next MoE FFN CUDA 모듈층 — plans/124 G001(FNA)→FNE, 2026-10-05.
//! 라우팅(softmax→top-k→정규화) + 전문가 디스패치·연산 + shared 결합의
//! 단일 진실. 원천 산술 계약: crates/core/src/qwen4exp/stages/moe.rs
//! moe_ffn(CPU 황금 기준) — 커널 산술 계약은 assets/exl3_fn_moe.cu 헤더.
//!
//! 3층 분리 원칙(plans/124 §5): 이 층은 가중치 상주 + 계산 API. 검증 자산
//! (오라클·픽스처·판정)은 moe_cuda_probe.rs — 모듈 파일 오염 금지(원장).
//!
//! [호출 규약 — core Ctx 디스패치 미러(stages/mod.rs)]
//! - 라우터·shared 게이트·shared 전문가: 배치(mm_batch 계급 L105-131 —
//!   전 토큰 동일 가중치) → wptr 전 원소 동일 포인터의 fn_moe_gemm_f16_ptr.
//! - 전문가 3 role: **mm_paired 계급**(L50-75 — "가중치마다 다른 1행 입력
//!   (MoE down)"): 토큰-메이저 (ti,e,w) 페어(moe.rs L198-216)로 xp 게더
//!   후 페어별 전문가 포인터 배열 GEMM. moe_ffn의 mm_paired 주소원
//!   소비자 계약(과제 지정)의 CUDA 미러다.
//! - 전문가별 상주: 포인터 배열 디스패치(전문가별 버퍼 + wptr[p])는
//!   mm_paired의 "전문가별 Weight 리스트" 직접 미러. 스택[512]×3 상주 +
//!   ids 색인(moe_down 계급)은 통합 시 상주 예산 판정(fn_moe_cuda 스켈레
//!   톤 계약 지도 "MoE 전문가" 항·moe.rs L92-95 스택 345MB/층 계약) —
//!   스택 경로는 wptr[p] = stack + e·stride로 **동일 산술**(커널 불변).
//!
//! [가중치 형식] F16 행 우선. EXL3 라우터 mlp.gate.weight F16[512,2560]
//! (트렐리스 아님 — fn_support.rs 계약 지도 실측 2026-10-05) 계급 —
//! 픽스처 F16은 검증 형식이며 GGUF Q4_K(G8 MMQ)·EXL3 트렐리스 전문가
//! 상주는 통합 단계의 역할(본 모듈 범위는 스테이지 산술).
//!
//! 단일 상주 원칙(2026-10-04 동결 사고): 이 모듈이 CudaCtx를 소유한다 —
//! 한 프로세스에 모델 1개(Exl3CudaDecoder·Q4Cuda와 동시 상주 금지).
//!
//! [정합 — 검증 원장, sm_89 실측 2026-10-05] 3혈상 전부 비트동일(maxdiff
//! 0.000e0 — 임계 3e-4): (i) 27B 폭 5120·512e top10 640/640 np=31 ·
//! (ii-a) 35B-A3B cfg 2048·256e top8 512/512 np=25 · (ii-b) Flash-Next cfg
//! 2560·512e top10 640/640 np=31. 라우팅 이산 선택 exact(순서 포함)·
//! 음성대조 2계급 NEG-DETECTED — 세부 원장은 moe_cuda_probe.rs 머리.
//!
//! [속도] 측정 대기 sm_80 — CMP 170HX 미도착(plans/124 §0). sm_80 설계
//! 근거는 assets/exl3_fn_moe.cu 헤더(개발기 sm_89는 정합 호스트일 뿐).
//!
//! [독립 컴파일 계약] scripts/cuda_probe_shim.rs가 rustc로 단독 컴파일 —
//! std 외 크레이트 금지(plans/124 G1).

use crate::rawcuda::ctx::CudaCtx;
use crate::rawcuda::ffi::CUdeviceptr;

/// MoE 형상 — Hparams4 MoE 서브셋(n_embd·n_expert·n_expert_used·n_ff_exp·
/// n_ff_shared — core mod.rs Hparams4 L51 대응, 값은 픽스처 config에서).
#[derive(Debug, Clone, PartialEq)]
pub struct MoeDims {
    pub n_embd: usize,
    pub n_expert: usize,
    pub n_used: usize,
    pub n_ff: usize,
    pub n_ff_sh: usize,
}

/// 라우팅 정규화 하한 — 원천: stages/moe.rs L62(wsum.max(6.103_515_6e-5)).
pub const MOE_WSUM_FLOOR: f32 = 6.103_515_6e-5;
/// 라우팅 선택 상한 — 커널 sel[16] 로컬 슬롯 도메인(실측 계급 8/10).
pub const MOE_USED_MAX: usize = 16;

/// 대형 h2d 청크 상한(4MB — 페이지 미매핑 가드, G2 원장 패턴 미러).
const H2D_CHUNK: usize = 4 << 20;

/// 재할당 가능 디바이스 버퍼(용량 원소 수·바이트 단위 grow-only).
struct DevBuf {
    p: CUdeviceptr,
    cap: usize,
}

impl DevBuf {
    fn new() -> Self {
        DevBuf { p: 0, cap: 0 }
    }
    /// bytes 확보(확장 시에만 재할당 — 이전 할당 1회 해제).
    fn ensure(&mut self, cc: &CudaCtx, bytes: usize) -> Result<CUdeviceptr, String> {
        if self.p != 0 && bytes <= self.cap {
            return Ok(self.p);
        }
        if self.p != 0 {
            // SAFETY: p는 이 ctx의 alloc 산출물 — 재할당 전 1회 해제.
            cc.free(self.p)?;
        }
        self.p = cc.alloc(bytes)?;
        self.cap = bytes;
        Ok(self.p)
    }
}

/// Flash-Next MoE FFN CUDA 모듈 — 가중치 상주 + moe_route/moe_ffn API.
pub struct MoeCuda {
    cc: CudaCtx,
    dims: MoeDims,
    /// 라우터 f16 [n_expert][n_embd] — 원천: ffn_gate_inp(moe.rs L25).
    w_route: CUdeviceptr,
    /// shared 게이트 라우터 f16 [n_embd] — 원천: ffn_gate_inp_sh(L26-28).
    w_route_sh: CUdeviceptr,
    /// shared gate f16 [n_ff_sh][n_embd] — 원천: ffn_gate_shexp(L29).
    sh_gate: CUdeviceptr,
    /// shared up f16 [n_ff_sh][n_embd] — 원천: ffn_up_shexp(L30).
    sh_up: CUdeviceptr,
    /// shared down f16 [n_embd][n_ff_sh] — 원천: ffn_down_shexp(L31-32).
    sh_down: CUdeviceptr,
    /// 전문가 gate f16 [n_ff][n_embd] — 인덱스=전문가 id(0=미상주).
    eg: Vec<CUdeviceptr>,
    /// 전문가 up f16 [n_ff][n_embd].
    eu: Vec<CUdeviceptr>,
    /// 전문가 down f16 [n_embd][n_ff].
    ed: Vec<CUdeviceptr>,
    // ── 작업 버퍼(grow-only — 호출 간 유지) ──
    /// 입력 스테이징 f32 [t][n_embd].
    dx: DevBuf,
    /// 라우터 로짓 f32 [t][n_expert].
    dlogits: DevBuf,
    /// 라우팅 선택 id i32 [t][n_used].
    dids: DevBuf,
    /// 라우팅 가중치 f32 [t][n_used].
    dwts: DevBuf,
    /// 토큰별 유지 원소 수 i32 [t].
    dcnt: DevBuf,
    /// 페어 게더 입력 f32 [np][n_embd] — 원천: moe.rs L203 xp.
    dxp: DevBuf,
    /// 페어 가중치 포인터 배열 u64 [np].
    dwptr: DevBuf,
    /// 중간(role 게이트·업) f32 — 최대(np·n_ff, t·n_ff_sh).
    da: DevBuf,
    db: DevBuf,
    /// 전문가 down 출력 f32 [np][n_embd].
    dyo: DevBuf,
    /// shared down 출력 f32 [t][n_embd].
    dsh: DevBuf,
    /// shared 게이트 로짓 f32 [t][1] — 원천: sgate_all(moe.rs L43-45).
    dsg: DevBuf,
    /// 페어 라우팅 가중치 f32 [np].
    dwp: DevBuf,
    /// 토큰별 페어 범위 i32 [t+1] — (ti,e) 정렬 계약(moe.rs L200).
    dpoff: DevBuf,
    /// 스테이지 출력 f32 [t][n_embd].
    dout: DevBuf,
}

impl MoeCuda {
    /// exl3_fn_moe.fatbin 자산 해석 — LLM170_CUDA_FN_MOE_FATBIN_PATH
    /// 오버라이드 우선(자산 경로 오버라이드일 뿐 계산 경로 분기 아님).
    fn fatbin_bytes() -> Result<Vec<u8>, String> {
        const ENV: &str = "LLM170_CUDA_FN_MOE_FATBIN_PATH";
        const REL: &[&str] = &[
            "crates/backend-gpu/src/rawcuda/assets/exl3_fn_moe.fatbin",
            "src/rawcuda/assets/exl3_fn_moe.fatbin",
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
            "exl3_fn_moe.fatbin 없음 — scripts/build_cuda.bat 실행 또는 {ENV} 지정 (탐색: {REL:?})"
        ))
    }

    /// 컨텍스트 + MoE 커널 4종 로드.
    pub fn new(dims: MoeDims) -> Result<Self, String> {
        if dims.n_used == 0 || dims.n_used > MOE_USED_MAX {
            return Err(format!(
                "moe new: n_used={} — (0, {MOE_USED_MAX}] 도메인(커널 sel 슬롯)",
                dims.n_used
            ));
        }
        if dims.n_embd == 0 || dims.n_expert == 0 || dims.n_ff == 0 || dims.n_ff_sh == 0 {
            return Err(format!(
                "moe new: 형상 0 금지 — {:?}",
                (dims.n_embd, dims.n_expert, dims.n_ff, dims.n_ff_sh)
            ));
        }
        if dims.n_expert * 5 > 48 << 10 {
            return Err(format!(
                "moe new: n_expert={} — 동적 공유 48KB 초과(라우팅 prob+taken)",
                dims.n_expert
            ));
        }
        let image = Self::fatbin_bytes()?;
        let mut cc = CudaCtx::new()?;
        let _g = cc.guard()?;
        cc.load_fatbin(
            "fn_moe",
            &image,
            &[
                "fn_moe_route",
                "fn_moe_gemm_f16_ptr",
                "fn_moe_ew",
                "fn_moe_combine",
            ],
        )?;
        let zeros = vec![0; dims.n_expert];
        Ok(MoeCuda {
            cc,
            dims,
            w_route: 0,
            w_route_sh: 0,
            sh_gate: 0,
            sh_up: 0,
            sh_down: 0,
            eg: zeros.clone(),
            eu: zeros.clone(),
            ed: zeros,
            dx: DevBuf::new(),
            dlogits: DevBuf::new(),
            dids: DevBuf::new(),
            dwts: DevBuf::new(),
            dcnt: DevBuf::new(),
            dxp: DevBuf::new(),
            dwptr: DevBuf::new(),
            da: DevBuf::new(),
            db: DevBuf::new(),
            dyo: DevBuf::new(),
            dsh: DevBuf::new(),
            dsg: DevBuf::new(),
            dwp: DevBuf::new(),
            dpoff: DevBuf::new(),
            dout: DevBuf::new(),
        })
    }

    pub fn device_name(&self) -> &str {
        &self.cc.device_name
    }

    /// 대형 h2d 청크 분할 업로드(G2 원장 패턴 미러 — 4MB 청크).
    fn h2d_chunked(cc: &CudaCtx, dst: CUdeviceptr, src: &[u8]) -> Result<(), String> {
        let mut off = 0usize;
        while off < src.len() {
            let hi = (off + H2D_CHUNK).min(src.len());
            // SAFETY: src는 호출자 소유 슬라이스 — 부분 슬라이스는 호출 내 유효.
            let part = unsafe { std::slice::from_raw_parts(src.as_ptr().add(off), hi - off) };
            cc.h2d(dst + off as u64, part)?;
            off = hi;
        }
        Ok(())
    }

    /// 상주 가중치 슬롯 교체(기존 있으면 해제 후 재할당·업로드).
    fn replace_w(slot: &mut CUdeviceptr, cc: &CudaCtx, bytes: &[u8]) -> Result<(), String> {
        if *slot != 0 {
            // SAFETY: 이전 alloc 산출물 — 교체 시 1회 해제.
            cc.free(*slot)?;
        }
        let p = cc.alloc(bytes.len())?;
        Self::h2d_chunked(cc, p, bytes)?;
        *slot = p;
        Ok(())
    }

    /// 라우터 등록 — route f16 [n_expert][n_embd]·route_sh f16 [n_embd].
    /// 원천: ffn_gate_inp·ffn_gate_inp_sh(moe.rs L25-28).
    pub fn set_router_f16(&mut self, route: &[u8], route_sh: &[u8]) -> Result<(), String> {
        let d = &self.dims;
        if route.len() != d.n_expert * d.n_embd * 2 {
            return Err(format!(
                "moe router: bytes={} != n_exp·n_embd·2={}",
                route.len(),
                d.n_expert * d.n_embd * 2
            ));
        }
        if route_sh.len() != d.n_embd * 2 {
            return Err(format!(
                "moe router_sh: bytes={} != n_embd·2={}",
                route_sh.len(),
                d.n_embd * 2
            ));
        }
        let _g = self.cc.guard()?;
        Self::replace_w(&mut self.w_route, &self.cc, route)?;
        Self::replace_w(&mut self.w_route_sh, &self.cc, route_sh)?;
        Ok(())
    }

    /// shared 전문가 등록 — gate/up f16 [n_ff_sh][n_embd]·down f16
    /// [n_embd][n_ff_sh]. 원천: ffn_{gate,up,down}_shexp(moe.rs L29-32).
    pub fn set_shared_f16(&mut self, gate: &[u8], up: &[u8], down: &[u8]) -> Result<(), String> {
        let d = &self.dims;
        let gu = d.n_ff_sh * d.n_embd * 2;
        let dn = d.n_embd * d.n_ff_sh * 2;
        if gate.len() != gu || up.len() != gu {
            return Err(format!(
                "moe shared gate/up: bytes={}/{} != n_ff_sh·n_embd·2={gu}",
                gate.len(),
                up.len()
            ));
        }
        if down.len() != dn {
            return Err(format!(
                "moe shared down: bytes={} != n_embd·n_ff_sh·2={dn}",
                down.len()
            ));
        }
        let _g = self.cc.guard()?;
        Self::replace_w(&mut self.sh_gate, &self.cc, gate)?;
        Self::replace_w(&mut self.sh_up, &self.cc, up)?;
        Self::replace_w(&mut self.sh_down, &self.cc, down)?;
        Ok(())
    }

    /// 전문가 등록(전문가별 상주 — mm_paired "전문가별 Weight 리스트" 미러).
    /// gate/up f16 [n_ff][n_embd]·down f16 [n_embd][n_ff]. 원천:
    /// ffn_{gate,up,down}_exps(moe.rs expert_w 슬라이스 — L96-100·L230-241).
    pub fn add_expert_f16(
        &mut self,
        e: usize,
        gate: &[u8],
        up: &[u8],
        down: &[u8],
    ) -> Result<(), String> {
        let d = &self.dims;
        if e >= d.n_expert {
            return Err(format!("moe add_expert: e={e} >= n_expert={}", d.n_expert));
        }
        let gu = d.n_ff * d.n_embd * 2;
        let dn = d.n_embd * d.n_ff * 2;
        if gate.len() != gu || up.len() != gu || down.len() != dn {
            return Err(format!(
                "moe add_expert: gate={}/{} up={}/{} down={}/{} — 형상 계약 위반",
                gate.len(),
                gu,
                up.len(),
                gu,
                down.len(),
                dn
            ));
        }
        let _g = self.cc.guard()?;
        Self::replace_w(&mut self.eg[e], &self.cc, gate)?;
        Self::replace_w(&mut self.eu[e], &self.cc, up)?;
        Self::replace_w(&mut self.ed[e], &self.cc, down)?;
        Ok(())
    }

    /// 입력xs 업로드 → dx [t][n_embd] 확보·적재.
    fn stage_xs(&mut self, xs: &[Vec<f32>]) -> Result<CUdeviceptr, String> {
        let t = xs.len();
        if t == 0 || t > 65535 {
            return Err(format!("moe: t={t} — (0, 65535] 도메인(그리드 y 상한)"));
        }
        for (ti, x) in xs.iter().enumerate() {
            if x.len() != self.dims.n_embd {
                return Err(format!(
                    "moe: xs[{ti}].len={} != n_embd={}",
                    x.len(),
                    self.dims.n_embd
                ));
            }
        }
        let mut flat = Vec::with_capacity(t * self.dims.n_embd);
        for x in xs {
            flat.extend_from_slice(x);
        }
        let p = self.dx.ensure(&self.cc, flat.len() * 4)?;
        // SAFETY: flat은 f32 슬라이스 — 길이 일치 바이트 뷰 변환.
        let b = unsafe { std::slice::from_raw_parts(flat.as_ptr() as *const u8, flat.len() * 4) };
        self.cc.h2d(p, b)?;
        Ok(p)
    }

    /// F16 포인터 배열 GEMM(mm_batch·mm_paired 공용 계급) —
    /// out[p][o] = Σ_i x[p][i]·W_p[o][i](fn_moe_gemm_f16_ptr).
    fn gemm_f16_ptrs(
        &mut self,
        x: CUdeviceptr,
        out: CUdeviceptr,
        n_in: usize,
        n_out: usize,
        ptrs: &[u64],
    ) -> Result<(), String> {
        let np = ptrs.len();
        if np == 0 || np > 65535 {
            return Err(format!(
                "moe gemm: np={np} — (0, 65535] 도메인(그리드 y 상한)"
            ));
        }
        if n_out == 0 || n_out > i32::MAX as usize || n_in == 0 || n_in > i32::MAX as usize {
            return Err(format!(
                "moe gemm: n_in={n_in} n_out={n_out} — (0, 2^31) 도메인"
            ));
        }
        let wp = self.dwptr.ensure(&self.cc, np * 8)?;
        let mut pb = Vec::with_capacity(np * 8);
        for v in ptrs {
            pb.extend_from_slice(&v.to_le_bytes());
        }
        self.cc.h2d(wp, &pb)?;
        let f = self.cc.function("fn_moe_gemm_f16_ptr")?;
        let (mut a0, mut a1, mut a2) = (x, wp, out);
        let (mut ni, mut no) = (n_in as i32, n_out as i32);
        let mut args: [*mut std::ffi::c_void; 5] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
            (&mut ni) as *mut _ as *mut _,
            (&mut no) as *mut _ as *mut _,
        ];
        self.cc
            .launch(f, n_out.div_ceil(128) as u32, np as u32, 128, &mut args)
    }

    /// ew(silu·mul) 제자리 — y = silu(g)·u(fn_moe_ew, g==y 제자리 허용).
    fn ew_inplace(&mut self, g: CUdeviceptr, u: CUdeviceptr, n: usize) -> Result<(), String> {
        if n == 0 || n > i32::MAX as usize {
            return Err(format!("moe ew: n={n} — (0, 2^31) 도메인"));
        }
        let f = self.cc.function("fn_moe_ew")?;
        let (mut a0, mut a1, mut a2) = (g, u, g);
        let mut nn = n as i32;
        let mut args: [*mut std::ffi::c_void; 4] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
            (&mut nn) as *mut _ as *mut _,
        ];
        self.cc.launch(f, n.div_ceil(128) as u32, 1, 128, &mut args)
    }

    /// 라우팅 — 원천: moe.rs L40-45(라우터 배치) + L46-70(선택).
    /// 반환: 토큰별 (전문가 id, 가중치) 선택 순서(확률 내림·동률 id
    /// 오름차) 리스트 — w==0 원소는 스킵됨(core by_expert push 가드).
    pub fn moe_route(&mut self, xs: &[Vec<f32>]) -> Result<Vec<Vec<(u32, f32)>>, String> {
        let d = self.dims.clone();
        let t = xs.len();
        let dxp = self.stage_xs(xs)?;
        // 라우터 배치 GEMM — wptr 전 원소 동일 포인터(mm_batch 계급).
        let logits = self.dlogits.ensure(&self.cc, t * d.n_expert * 4)?;
        self.gemm_f16_ptrs(dxp, logits, d.n_embd, d.n_expert, &vec![self.w_route; t])?;
        let ids = self.dids.ensure(&self.cc, t * d.n_used * 4)?;
        let wts = self.dwts.ensure(&self.cc, t * d.n_used * 4)?;
        let cnt = self.dcnt.ensure(&self.cc, t * 4)?;
        let f = self.cc.function("fn_moe_route")?;
        // 인자 순서 계약: 커널 시그니처 (logits, n_exp, n_used, ids, wts,
        // cnt) — 포인터·정수 교차 배치(순서 혼동이 G8 디버그 원장 클래스).
        let (mut a0, mut a1, mut a2, mut a3) = (logits, ids, wts, cnt);
        let (mut ne, mut nu) = (d.n_expert as i32, d.n_used as i32);
        let mut args: [*mut std::ffi::c_void; 6] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut ne) as *mut _ as *mut _,
            (&mut nu) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
            (&mut a3) as *mut _ as *mut _,
        ];
        self.cc
            .launch_shared(f, t as u32, 1, 32, (d.n_expert * 5) as u32, &mut args)?;
        let mut ib = vec![0u8; t * d.n_used * 4];
        let mut wb = vec![0u8; t * d.n_used * 4];
        let mut cb = vec![0u8; t * 4];
        self.cc.d2h(&mut ib, ids)?;
        self.cc.d2h(&mut wb, wts)?;
        self.cc.d2h(&mut cb, cnt)?;
        self.cc.sync()?;
        // SAFETY: d2h·sync 완료 후 재해석(길이·정렬 일치 — G2 판독 패턴).
        let ids = unsafe { std::slice::from_raw_parts(ib.as_ptr() as *const i32, t * d.n_used) };
        let wts = unsafe { std::slice::from_raw_parts(wb.as_ptr() as *const f32, t * d.n_used) };
        let cnts = unsafe { std::slice::from_raw_parts(cb.as_ptr() as *const i32, t) };
        let mut out = Vec::with_capacity(t);
        for ti in 0..t {
            let c = cnts[ti];
            if !(0..=d.n_used as i32).contains(&c) {
                return Err(format!(
                    "moe route: cnt[{ti}]={c} — [0, n_used] 도메인 위반"
                ));
            }
            let mut row = Vec::with_capacity(c as usize);
            for k in 0..c as usize {
                let id = ids[ti * d.n_used + k];
                if id < 0 || id as usize >= d.n_expert {
                    return Err(format!(
                        "moe route: id[{ti}][{k}]={id} — 전문가 id 도메인 위반"
                    ));
                }
                row.push((id as u32, wts[ti * d.n_used + k]));
            }
            out.push(row);
        }
        Ok(out)
    }

    /// MoE FFN 전체 — 원천: stages/moe.rs moe_ffn L22-320(그룹 경로).
    /// 순서 계약: 라우터 2종 배치(L40-45) → 선택(L46-70) → 토큰-메이저
    /// (ti,e,w) 페어(L198-216 — 토큰 내 e 오름차) → 전문가 gate·up·ew·
    /// down(mm_paired 계급) → shared(gate·up·ew·down + sigmoid 게이트
    /// L246-318) → 결합(전문가 누산 후 shared 가산 L313-318).
    pub fn moe_ffn(&mut self, xs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
        let d = self.dims.clone();
        let t = xs.len();
        let sel = self.moe_route(xs)?;
        // 페어 구성 — moe.rs L198-201: by_expert 전개 후 (ti, e) 정렬 ==
        // 토큰-메이저·토큰 내 e 오름차(전문가별 경로의 누산 순서와 동일).
        let mut pairs: Vec<(u32, u32, f32)> = Vec::with_capacity(t * d.n_used);
        for (ti, row) in sel.iter().enumerate() {
            let mut row: Vec<(u32, f32)> = row.clone();
            row.sort_by_key(|&(e, _)| e);
            for (e, w) in row {
                pairs.push((ti as u32, e, w));
            }
        }
        let np = pairs.len();
        // 전문가 3 role — 페어 게더 후 포인터 배열 GEMM(mm_paired 미러).
        let dyo = if np > 0 {
            let mut xp = Vec::with_capacity(np * d.n_embd);
            for &(ti, _, _) in &pairs {
                xp.extend_from_slice(&xs[ti as usize]);
            }
            let dxp = self.dxp.ensure(&self.cc, np * d.n_embd * 4)?;
            // SAFETY: xp은 f32 슬라이스 — 길이 일치 바이트 뷰 변환.
            let xb = unsafe { std::slice::from_raw_parts(xp.as_ptr() as *const u8, xp.len() * 4) };
            self.cc.h2d(dxp, xb)?;
            let need_expert =
                |which: &str, p: CUdeviceptr, e: u32| -> Result<CUdeviceptr, String> {
                    if p == 0 {
                        Err(format!(
                            "moe_ffn: {which} 전문가 e={e} 미상주 — add_expert_f16 먼저"
                        ))
                    } else {
                        Ok(p)
                    }
                };
            let ga = self.da.ensure(&self.cc, np * d.n_ff * 4)?;
            let pg: Vec<CUdeviceptr> = pairs
                .iter()
                .map(|&(_, e, _)| need_expert("gate", self.eg[e as usize], e))
                .collect::<Result<_, _>>()?;
            self.gemm_f16_ptrs(dxp, ga, d.n_embd, d.n_ff, &pg)?;
            let gb = self.db.ensure(&self.cc, np * d.n_ff * 4)?;
            let pu: Vec<CUdeviceptr> = pairs
                .iter()
                .map(|&(_, e, _)| need_expert("up", self.eu[e as usize], e))
                .collect::<Result<_, _>>()?;
            self.gemm_f16_ptrs(dxp, gb, d.n_embd, d.n_ff, &pu)?;
            self.ew_inplace(ga, gb, np * d.n_ff)?;
            let dyo = self.dyo.ensure(&self.cc, np * d.n_embd * 4)?;
            let pd: Vec<CUdeviceptr> = pairs
                .iter()
                .map(|&(_, e, _)| need_expert("down", self.ed[e as usize], e))
                .collect::<Result<_, _>>()?;
            self.gemm_f16_ptrs(ga, dyo, d.n_ff, d.n_embd, &pd)?;
            dyo
        } else {
            self.dyo.ensure(&self.cc, 4)?
        };
        // shared 전문가 — 전 토큰 배치(L246-262·L303-313: gate·up 동일
        // 입력 그룹 → ew → down) + sgate 라우터(L43-45).
        let sga = self.da.ensure(&self.cc, t * d.n_ff_sh * 4)?;
        let dx_in = self.dx.p;
        self.gemm_f16_ptrs(dx_in, sga, d.n_embd, d.n_ff_sh, &vec![self.sh_gate; t])?;
        let sbu = self.db.ensure(&self.cc, t * d.n_ff_sh * 4)?;
        self.gemm_f16_ptrs(dx_in, sbu, d.n_embd, d.n_ff_sh, &vec![self.sh_up; t])?;
        self.ew_inplace(sga, sbu, t * d.n_ff_sh)?;
        let shout = self.dsh.ensure(&self.cc, t * d.n_embd * 4)?;
        self.gemm_f16_ptrs(sga, shout, d.n_ff_sh, d.n_embd, &vec![self.sh_down; t])?;
        let dsg = self.dsg.ensure(&self.cc, t * 4)?;
        self.gemm_f16_ptrs(dx_in, dsg, d.n_embd, 1, &vec![self.w_route_sh; t])?;
        // 결합 — 전문가 누산(토큰 내 페어 순서 = e 오름차) 후 shared
        // sigmoid 게이트 가산(L313-318).
        let mut wpair = Vec::with_capacity(np);
        let mut poff = Vec::with_capacity(t + 1);
        poff.push(0i32);
        for (ti, row) in sel.iter().enumerate() {
            let mut row: Vec<(u32, f32)> = row.clone();
            row.sort_by_key(|&(e, _)| e);
            poff.push(poff[ti] + row.len() as i32);
            wpair.extend(row.iter().map(|&(_, w)| w));
        }
        let dwp = self.dwp.ensure(&self.cc, np * 4)?;
        // SAFETY: wpair은 f32 슬라이스 — 길이 일치 바이트 뷰 변환.
        let wb = unsafe { std::slice::from_raw_parts(wpair.as_ptr() as *const u8, np * 4) };
        self.cc.h2d(dwp, wb)?;
        let dpo = self.dpoff.ensure(&self.cc, (t + 1) * 4)?;
        let mut pob = Vec::with_capacity((t + 1) * 4);
        for v in &poff {
            pob.extend_from_slice(&v.to_le_bytes());
        }
        self.cc.h2d(dpo, &pob)?;
        let dout = self.dout.ensure(&self.cc, t * d.n_embd * 4)?;
        let f = self.cc.function("fn_moe_combine")?;
        let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5) = (dyo, dwp, dpo, shout, dsg, dout);
        let mut ne = d.n_embd as i32;
        let mut args: [*mut std::ffi::c_void; 7] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
            (&mut a3) as *mut _ as *mut _,
            (&mut a4) as *mut _ as *mut _,
            (&mut a5) as *mut _ as *mut _,
            (&mut ne) as *mut _ as *mut _,
        ];
        self.cc.launch(f, t as u32, 1, 256, &mut args)?;
        let mut ob = vec![0u8; t * d.n_embd * 4];
        self.cc.d2h(&mut ob, dout)?;
        self.cc.sync()?;
        // SAFETY: d2h·sync 완료 후 재해석(길이·정렬 일치 — G2 판독 패턴).
        let flat = unsafe { std::slice::from_raw_parts(ob.as_ptr() as *const f32, t * d.n_embd) };
        Ok(flat.chunks(d.n_embd).map(<[f32]>::to_vec).collect())
    }
}

impl Drop for MoeCuda {
    fn drop(&mut self) {
        // SAFETY: 각 포인터는 이 ctx의 alloc 산출물이며 drop에서 1회 해제.
        let r = (|| {
            for p in self
                .eg
                .iter()
                .chain(self.eu.iter())
                .chain(self.ed.iter())
                .chain([
                    &self.w_route,
                    &self.w_route_sh,
                    &self.sh_gate,
                    &self.sh_up,
                    &self.sh_down,
                ])
            {
                if *p != 0 {
                    self.cc.free(*p)?;
                }
            }
            for b in [
                &mut self.dx,
                &mut self.dlogits,
                &mut self.dids,
                &mut self.dwts,
                &mut self.dcnt,
                &mut self.dxp,
                &mut self.dwptr,
                &mut self.da,
                &mut self.db,
                &mut self.dyo,
                &mut self.dsh,
                &mut self.dsg,
                &mut self.dwp,
                &mut self.dpoff,
                &mut self.dout,
            ] {
                if b.p != 0 {
                    self.cc.free(b.p)?;
                }
            }
            Ok::<(), String>(())
        })();
        if let Err(e) = r {
            eprintln!("moe_cuda: drop 해제 실패: {e}");
        }
    }
}
