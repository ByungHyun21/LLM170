//! DeepSeek-V4 MoE 스테이지 CUDA 모듈층 — plans/130 B3, 2026-10-05.
//! 해시 라우팅(L0-2, tid2eid 표 룩업) + noaux_tc 라우팅(L3+, sqrtsoftplus
//! + e_score_correction_bias top-k) + EXL3 trellis 전문가 SwiGLU(limit 10
//! 비대칭 클램프) + 공유 전문가 무스케일 가산의 단일 진실.
//! 원천 산술 계약: crates/core/src/deepseek4/stages/moe.rs(CPU 황금 기준)
//! — 커널 산술 계약은 assets/ds4_moe.cu 헤더.
//!
//! 3층 분리 원칙(plans/124 §5): 이 층은 가중치 상주 + 계산 API. 검증 자산
//! (오라클·픽스처·판정)은 ds4_moe_cuda_probe.rs — 모듈 파일 오염 금지(원장).
//!
//! [호출 규약 — FNE moe_cuda.rs 미러(plans/124 G001 FNA→FNE)]
//! - 게이트 GEMV: k-major f32 [dim][n_routed](loader.rs plain_kmat 전치
//!   결과) — ds4_gate 커널(스레드당 라우터 1출력·k 오름차순 순차 누산).
//! - 라우팅: ds4_route_hash/ds4_route_routed — 토큰당 단일 스레드(Σ 순서·
//!   top-k 안정 동점 계약). 출력 (eid 오름차순, w) 쌍.
//! - 전문가 3 role: mm_paired 계급(moe_cuda.rs GEMM 포인터 배열 디스패치)
//!   — 토큰-메이저 (ti,e,w) 페어(moe_forward by_expert BTreeMap 계약의
//!   누산 순서와 동일). 전문가 가중치는 EXL3 trellis 디양자화 f32
//!   (호스트 loader.rs linear L203-247 산출 — K=3 링, 공유 K=5)를
//!   k-major 그대로 상주(라우팅 식·SwiGLU 비대칭 클램프·가중치 원이
//!   Q4/F16이 아니라는 점이 FNE와의 DELTA).
//! - 공유 전문가: 전 토큰 동일 가중치 배치(w=1.0 — 무스케일) + 결합.
//!
//! [가중치 형식] 전부 f32 LE. gate.weight [dim][n_routed] k-major ·
//! gate.bias [n_routed](라우티드 층만 — 해시 층 bias 무시, loader.rs
//! L357-363) · tid2eid i64 [vocab·n_active](해시 층만) · 전문가
//! w1/w3 [dim][inter]·w2 [inter][dim] k-major.
//!
//! 단일 상주 원칙(2026-10-04 동결 사고): 이 모듈이 CudaCtx를 소유한다 —
//! 한 프로세스에 모델 1개(다른 rawcuda 모듈과 동시 상주 금지).
//!
//! [정합 — 검증 원장, sm_89 실측 2026-10-05] 실측 3혈상 전부 비트동일
//! (maxdiff 0.000e0 — 임계 ROUTE_W 1e-6·MOE_VAL 3e-4): (i) 해시 L0 실측
//! t=2 np=12(ids EXACT) · (ii) 라우티드 L3 실측 noaux_tc t=2 np=12 ·
//! (iii) 합성 존+클램프 dim=128. 음성대조 3계급 NEG-DETECTED — 세부
//! 원장은 ds4_moe_cuda_probe.rs 머리.
//!
//! [독립 컴파일 계약] scripts/cuda_probe_shim.rs가 rustc로 단독 컴파일 —
//! std 외 크레이트 금지(plans/124 G1).

use crate::rawcuda::ctx::CudaCtx;
use crate::rawcuda::ffi::CUdeviceptr;

/// DS4 MoE 형상 — config.json MoE 서브셋(hidden_size·n_routed_experts·
/// num_experts_per_tok·moe_intermediate_size·routed_scaling_factor·
/// swiglu_limit — core config.rs 대응, 값은 픽스처 config에서).
#[derive(Debug, Clone, PartialEq)]
pub struct Ds4MoeDims {
    pub dim: usize,
    pub n_routed: usize,
    pub n_active: usize,
    pub inter: usize,
    pub route_scale: f32,
    pub swiglu_limit: f32,
}

/// 라우팅 커널 로컬 스코어 배열 상한(ds4_route_routed b[1024]).
pub const DS4_ROUTE_MAX: usize = 1024;
/// 라우팅 선택 슬롯 상한(ds4_route_* sel[16] 도메인 — 실측 6).
pub const DS4_ACTIVE_MAX: usize = 16;

/// 대형 h2d 청크 상한(4MB — 페이지 미매핑 가드, G2 원장 패턴 미러).
const H2D_CHUNK: usize = 4 << 20;

/// 재할당 가능 디바이스 버퍼(용량 바이트 단위 grow-only).
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

/// DeepSeek-V4 MoE CUDA 모듈 — 가중치 상주 + route/ffn API.
pub struct Ds4MoeCuda {
    cc: CudaCtx,
    dims: Ds4MoeDims,
    /// 게이트 가중치 f32 k-major [dim][n_routed] — 원천: ffn.gate.weight
    /// F16 [n_routed][dim] 전치(loader.rs plain_kmat).
    w_gate: CUdeviceptr,
    /// 게이트 바이어스 f32 [n_routed] — 라우티드 층만(0=미설정).
    /// 원천: ffn.gate.bias F16(noaux_tc e_score_correction_bias).
    d_bias: CUdeviceptr,
    /// 해시 표 i64 [vocab·n_active] — 해시 층만(0=미설정).
    d_tid2eid: CUdeviceptr,
    /// 해시 표 행 수(vocab) — set_gate_f32 산출·tids 검증용.
    hash_rows: usize,
    /// 공유 전문가 w1 f32 [dim][inter] — 원천: ffn.shared_experts.w1
    /// (trellis K=5 디양자화, loader.rs expert_at L401-409).
    sw1: CUdeviceptr,
    /// 공유 전문가 w2 f32 [inter][dim].
    sw2: CUdeviceptr,
    /// 공유 전문가 w3 f32 [dim][inter].
    sw3: CUdeviceptr,
    /// 전문가 w1 f32 [dim][inter] — 인덱스=전문가 id(0=미상주).
    eg1: Vec<CUdeviceptr>,
    /// 전문가 w2 f32 [inter][dim].
    eg2: Vec<CUdeviceptr>,
    /// 전문가 w3 f32 [dim][inter].
    eg3: Vec<CUdeviceptr>,
    // ── 작업 버퍼(grow-only — 호출 간 유지) ──
    /// 입력 스테이징 f32 [t][dim].
    dx: DevBuf,
    /// 게이트 스코어 f32 [t][n_routed].
    dsc: DevBuf,
    /// 해시 토큰 id u32 [t].
    dtids: DevBuf,
    /// 라우팅 선택 id i32 [t][n_active].
    dids: DevBuf,
    /// 라우팅 가중치 f32 [t][n_active].
    dwts: DevBuf,
    /// 페어 게더 입력 f32 [np][dim] — fp8-sim 제자리 → xq.
    dxp: DevBuf,
    /// 페어 가중치 포인터 배열 u64 [np].
    dwptr: DevBuf,
    /// 중간(g→h) f32 — 최대(max(np,t)·inter).
    da: DevBuf,
    /// 중간(up) f32 — 최대(max(np,t)·inter).
    db: DevBuf,
    /// 전문가 down 출력 f32 [np][dim].
    dyo: DevBuf,
    /// 공유 입력 스테이징 f32 [t][dim] — fp8-sim 제자리.
    dshx: DevBuf,
    /// 공유 down 출력 f32 [t][dim].
    dsh: DevBuf,
    /// 페어 라우팅 가중치 f32 [max(np,t)](공유는 1.0 채움).
    dwp: DevBuf,
    /// 토큰별 페어 범위 i32 [t+1] — (ti,e) 정렬 계약.
    dpoff: DevBuf,
    /// 스테이지 출력 f32 [t][dim].
    dout: DevBuf,
}

impl Ds4MoeCuda {
    /// ds4_moe.fatbin 자산 해석 — LLM170_CUDA_DS4_MOE_FATBIN_PATH
    /// 오버라이드 우선(자산 경로 오버라이드일 뿐 계산 경로 분기 아님).
    fn fatbin_bytes() -> Result<Vec<u8>, String> {
        const ENV: &str = "LLM170_CUDA_DS4_MOE_FATBIN_PATH";
        const REL: &[&str] = &[
            "crates/backend-gpu/src/rawcuda/assets/ds4_moe.fatbin",
            "src/rawcuda/assets/ds4_moe.fatbin",
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
            "ds4_moe.fatbin 없음 — scripts/build_cuda.bat 실행 또는 {ENV} 지정 (탐색: {REL:?})"
        ))
    }

    /// 컨텍스트 + DS4 MoE 커널 로드.
    pub fn new(dims: Ds4MoeDims) -> Result<Self, String> {
        if dims.n_active == 0 || dims.n_active > DS4_ACTIVE_MAX {
            return Err(format!(
                "ds4_moe new: n_active={} — (0, {DS4_ACTIVE_MAX}] 도메인(커널 sel 슬롯)",
                dims.n_active
            ));
        }
        if dims.n_routed == 0 || dims.n_routed > DS4_ROUTE_MAX {
            return Err(format!(
                "ds4_moe new: n_routed={} — (0, {DS4_ROUTE_MAX}] 도메인(커널 로컬 스코어)",
                dims.n_routed
            ));
        }
        if dims.n_routed < dims.n_active {
            return Err(format!(
                "ds4_moe new: n_routed={} < n_active={}",
                dims.n_routed, dims.n_active
            ));
        }
        if dims.dim == 0 || dims.dim % 128 != 0 || dims.inter == 0 || dims.inter % 128 != 0 {
            return Err(format!(
                "ds4_moe new: dim={} inter={} — 128 배수 계약(fp8-sim 블록·trellis 정렬)",
                dims.dim, dims.inter
            ));
        }
        if !dims.route_scale.is_finite() || !dims.swiglu_limit.is_finite() {
            return Err(format!(
                "ds4_moe new: route_scale/swiglu_limit 비유한 — {:?}",
                (dims.route_scale, dims.swiglu_limit)
            ));
        }
        let image = Self::fatbin_bytes()?;
        let mut cc = CudaCtx::new()?;
        let _g = cc.guard()?;
        cc.load_fatbin(
            "ds4_moe",
            &image,
            &[
                "ds4_gate",
                "ds4_route_hash",
                "ds4_route_routed",
                "ds4_fp8_sim",
                "ds4_gemv_f32_ptr",
                "ds4_swiglu",
                "ds4_bf16_round",
                "ds4_combine",
            ],
        )?;
        let zeros = vec![0; dims.n_routed];
        Ok(Ds4MoeCuda {
            cc,
            dims,
            w_gate: 0,
            d_bias: 0,
            d_tid2eid: 0,
            hash_rows: 0,
            sw1: 0,
            sw2: 0,
            sw3: 0,
            eg1: zeros.clone(),
            eg2: zeros.clone(),
            eg3: zeros,
            dx: DevBuf::new(),
            dsc: DevBuf::new(),
            dtids: DevBuf::new(),
            dids: DevBuf::new(),
            dwts: DevBuf::new(),
            dxp: DevBuf::new(),
            dwptr: DevBuf::new(),
            da: DevBuf::new(),
            db: DevBuf::new(),
            dyo: DevBuf::new(),
            dshx: DevBuf::new(),
            dsh: DevBuf::new(),
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

    /// f32 슬라이스 → LE 바이트.
    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        let mut b = Vec::with_capacity(v.len() * 4);
        for x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
        b
    }

    /// 게이트 등록 — weight f32 k-major [dim][n_routed](F16 원본 전치값),
    /// bias f32 [n_routed](라우티드 층 — noaux_tc e_score_correction_bias),
    /// tid2eid i64 [vocab·n_active](해시 층 — 어느 쪽이든 미설정 허용,
    /// 해당 경로 호출 시 필요). 원천: loader.rs block 게이트 L352-366
    /// (해시 층 bias=None·라우티드 층 tid2eid=None — 계약 동일).
    pub fn set_gate_f32(
        &mut self,
        weight: &[f32],
        bias: Option<&[f32]>,
        tid2eid: Option<&[i64]>,
    ) -> Result<(), String> {
        let d = &self.dims;
        if weight.len() != d.dim * d.n_routed {
            return Err(format!(
                "ds4_moe gate: weight.len={} != dim·n_routed={}",
                weight.len(),
                d.dim * d.n_routed
            ));
        }
        if let Some(b) = bias {
            if b.len() != d.n_routed {
                return Err(format!(
                    "ds4_moe gate: bias.len={} != n_routed={}",
                    b.len(),
                    d.n_routed
                ));
            }
        }
        if let Some(tt) = tid2eid {
            if tt.len() % d.n_active != 0 {
                return Err(format!(
                    "ds4_moe gate: tid2eid.len={} — n_active={} 배수 아님",
                    tt.len(),
                    d.n_active
                ));
            }
        }
        let _g = self.cc.guard()?;
        Self::replace_w(&mut self.w_gate, &self.cc, &Self::f32_bytes(weight))?;
        self.hash_rows = 0;
        match bias {
            Some(b) => Self::replace_w(&mut self.d_bias, &self.cc, &Self::f32_bytes(b))?,
            None => {
                if self.d_bias != 0 {
                    // SAFETY: 이전 alloc 산출물 — 미설정 전환 시 해제.
                    self.cc.free(self.d_bias)?;
                    self.d_bias = 0;
                }
            }
        }
        match tid2eid {
            Some(tt) => {
                let mut b = Vec::with_capacity(tt.len() * 8);
                for v in tt {
                    b.extend_from_slice(&v.to_le_bytes());
                }
                Self::replace_w(&mut self.d_tid2eid, &self.cc, &b)?;
                self.hash_rows = tt.len() / d.n_active;
            }
            None => {
                if self.d_tid2eid != 0 {
                    // SAFETY: 이전 alloc 산출물 — 미설정 전환 시 해제.
                    self.cc.free(self.d_tid2eid)?;
                    self.d_tid2eid = 0;
                }
            }
        }
        Ok(())
    }

    /// 공유 전문가 등록 — w1/w3 f32 [dim][inter]·w2 f32 [inter][dim]
    /// (trellis K=5 디양자화값). 원천: loader.rs shared L399-401.
    pub fn set_shared_f32(&mut self, w1: &[f32], w2: &[f32], w3: &[f32]) -> Result<(), String> {
        let d = &self.dims;
        let wi = d.dim * d.inter;
        let wo = d.inter * d.dim;
        if w1.len() != wi || w3.len() != wi || w2.len() != wo {
            return Err(format!(
                "ds4_moe shared: w1={}/{} w3={}/{} w2={}/{} — 형상 계약 위반",
                w1.len(),
                wi,
                w3.len(),
                wi,
                w2.len(),
                wo
            ));
        }
        let _g = self.cc.guard()?;
        Self::replace_w(&mut self.sw1, &self.cc, &Self::f32_bytes(w1))?;
        Self::replace_w(&mut self.sw2, &self.cc, &Self::f32_bytes(w2))?;
        Self::replace_w(&mut self.sw3, &self.cc, &Self::f32_bytes(w3))?;
        Ok(())
    }

    /// 라우티드 전문가 등록(전문가별 상주 — mm_paired "전문가별 Weight
    /// 리스트" 미러). w1/w3 f32 [dim][inter]·w2 f32 [inter][dim]
    /// (trellis K=3 디양자화값). 원천: loader.rs expert L393-397.
    pub fn add_expert_f32(
        &mut self,
        e: usize,
        w1: &[f32],
        w2: &[f32],
        w3: &[f32],
    ) -> Result<(), String> {
        let d = &self.dims;
        if e >= d.n_routed {
            return Err(format!(
                "ds4_moe add_expert: e={e} >= n_routed={}",
                d.n_routed
            ));
        }
        let wi = d.dim * d.inter;
        let wo = d.inter * d.dim;
        if w1.len() != wi || w3.len() != wi || w2.len() != wo {
            return Err(format!(
                "ds4_moe add_expert: w1={}/{} w3={}/{} w2={}/{} — 형상 계약 위반",
                w1.len(),
                wi,
                w3.len(),
                wi,
                w2.len(),
                wo
            ));
        }
        let _g = self.cc.guard()?;
        Self::replace_w(&mut self.eg1[e], &self.cc, &Self::f32_bytes(w1))?;
        Self::replace_w(&mut self.eg2[e], &self.cc, &Self::f32_bytes(w2))?;
        Self::replace_w(&mut self.eg3[e], &self.cc, &Self::f32_bytes(w3))?;
        Ok(())
    }

    /// 입력 xs 업로드 → dx [t][dim] 확보·적재 + 형상 검증.
    fn stage_xs(&mut self, xs: &[Vec<f32>]) -> Result<CUdeviceptr, String> {
        let t = xs.len();
        if t == 0 || t > 65535 {
            return Err(format!("ds4_moe: t={t} — (0, 65535] 도메인(그리드 y 상한)"));
        }
        for (ti, x) in xs.iter().enumerate() {
            if x.len() != self.dims.dim {
                return Err(format!(
                    "ds4_moe: xs[{ti}].len={} != dim={}",
                    x.len(),
                    self.dims.dim
                ));
            }
        }
        let mut flat = Vec::with_capacity(t * self.dims.dim);
        for x in xs {
            flat.extend_from_slice(x);
        }
        let p = self.dx.ensure(&self.cc, flat.len() * 4)?;
        self.cc.h2d(p, &Self::f32_bytes(&flat))?;
        Ok(p)
    }

    /// 게이트 GEMV + sqrtsoftplus — ds4_gate. dsc [t][n_routed] 반환.
    fn run_gate(&mut self, dx: CUdeviceptr, t: usize) -> Result<CUdeviceptr, String> {
        if self.w_gate == 0 {
            return Err("ds4_moe: 게이트 가중치 미설정 — set_gate_f32 먼저".into());
        }
        let d = self.dims.clone();
        let s = self.dsc.ensure(&self.cc, t * d.n_routed * 4)?;
        let f = self.cc.function("ds4_gate")?;
        let (mut a0, mut a1, mut a2) = (dx, self.w_gate, s);
        let (mut ni, mut nn) = (d.dim as i32, d.n_routed as i32);
        let mut args: [*mut std::ffi::c_void; 5] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
            (&mut ni) as *mut _ as *mut _,
            (&mut nn) as *mut _ as *mut _,
        ];
        self.cc
            .launch(f, d.n_routed.div_ceil(128) as u32, t as u32, 128, &mut args)?;
        Ok(s)
    }

    /// 라우팅 커널 발사(해시|라우티드) → (dids, dwts) [t][n_active].
    fn run_route(
        &mut self,
        dsc: CUdeviceptr,
        t: usize,
        tids: Option<&[u32]>,
    ) -> Result<(CUdeviceptr, CUdeviceptr), String> {
        let d = self.dims.clone();
        let ids = self.dids.ensure(&self.cc, t * d.n_active * 4)?;
        let wts = self.dwts.ensure(&self.cc, t * d.n_active * 4)?;
        let mut rs = d.route_scale;
        if let Some(tt) = tids {
            if tt.len() != t {
                return Err(format!("ds4_moe route: tids.len={} != t={t}", tt.len()));
            }
            if self.d_tid2eid == 0 {
                return Err("ds4_moe route: tid2eid 미설정 — set_gate_f32(해시 층)".into());
            }
            for (ti, &tid) in tt.iter().enumerate() {
                if tid as usize >= self.hash_rows {
                    return Err(format!(
                        "ds4_moe route: tids[{ti}]={tid} — vocab={} 도메인 위반",
                        self.hash_rows
                    ));
                }
            }
            let dt = self.dtids.ensure(&self.cc, t * 4)?;
            let mut tb = Vec::with_capacity(t * 4);
            for v in tt {
                tb.extend_from_slice(&v.to_le_bytes());
            }
            self.cc.h2d(dt, &tb)?;
            let f = self.cc.function("ds4_route_hash")?;
            // 인자 순서 계약: 커널 시그니처 (s, tid2eid, tids, n_routed, k,
            // route_scale, ids, wts) — 포인터·정수 교차 배치(순서 혼동이
            // G8 디버그 원장 클래스 — 라우티드 분기와 배치 순서 상이 주의).
            let (mut a0, mut a1, mut a2) = (dsc, self.d_tid2eid, dt);
            let (mut nr, mut ka) = (d.n_routed as i32, d.n_active as i32);
            let (mut a6, mut a7) = (ids, wts);
            let mut args: [*mut std::ffi::c_void; 8] = [
                (&mut a0) as *mut _ as *mut _,
                (&mut a1) as *mut _ as *mut _,
                (&mut a2) as *mut _ as *mut _,
                (&mut nr) as *mut _ as *mut _,
                (&mut ka) as *mut _ as *mut _,
                (&mut rs) as *mut _ as *mut _,
                (&mut a6) as *mut _ as *mut _,
                (&mut a7) as *mut _ as *mut _,
            ];
            self.cc.launch(f, t as u32, 1, 32, &mut args)?;
        } else {
            if self.d_bias == 0 {
                return Err("ds4_moe route: bias 미설정 — set_gate_f32(라우티드 층)".into());
            }
            let f = self.cc.function("ds4_route_routed")?;
            let (mut a0, mut a1, mut a2, mut a3) = (dsc, self.d_bias, ids, wts);
            let (mut nr, mut ka) = (d.n_routed as i32, d.n_active as i32);
            let mut args: [*mut std::ffi::c_void; 7] = [
                (&mut a0) as *mut _ as *mut _,
                (&mut a1) as *mut _ as *mut _,
                (&mut nr) as *mut _ as *mut _,
                (&mut ka) as *mut _ as *mut _,
                (&mut rs) as *mut _ as *mut _,
                (&mut a2) as *mut _ as *mut _,
                (&mut a3) as *mut _ as *mut _,
            ];
            self.cc.launch(f, t as u32, 1, 32, &mut args)?;
        }
        Ok((ids, wts))
    }

    /// 라우팅 판독·검증 → 토큰별 (eid 오름차순 (id, w)) 리스트.
    fn readback_route(
        &mut self,
        ids: CUdeviceptr,
        wts: CUdeviceptr,
        t: usize,
    ) -> Result<Vec<Vec<(u32, f32)>>, String> {
        let d = self.dims.clone();
        let mut ib = vec![0u8; t * d.n_active * 4];
        let mut wb = vec![0u8; t * d.n_active * 4];
        self.cc.d2h(&mut ib, ids)?;
        self.cc.d2h(&mut wb, wts)?;
        self.cc.sync()?;
        // SAFETY: d2h·sync 완료 후 재해석(길이·정렬 일치 — G2 판독 패턴).
        let ids = unsafe { std::slice::from_raw_parts(ib.as_ptr() as *const i32, t * d.n_active) };
        let wts = unsafe { std::slice::from_raw_parts(wb.as_ptr() as *const f32, t * d.n_active) };
        let mut out = Vec::with_capacity(t);
        for ti in 0..t {
            let mut row = Vec::with_capacity(d.n_active);
            for k in 0..d.n_active {
                let id = ids[ti * d.n_active + k];
                if id < 0 || id as usize >= d.n_routed {
                    return Err(format!(
                        "ds4_moe route: ids[{ti}][{k}]={id} — 전문가 id 도메인 위반"
                    ));
                }
                let w = wts[ti * d.n_active + k];
                if !w.is_finite() {
                    return Err(format!(
                        "ds4_moe route: wts[{ti}][{k}]={w} — 비유한 라우팅 가중치"
                    ));
                }
                row.push((id as u32, w));
            }
            out.push(row);
        }
        Ok(out)
    }

    /// 라우팅 단독 API(해시) — 원천: moe_forward 1단계(L146-165).
    pub fn moe_route_hash(
        &mut self,
        xs: &[Vec<f32>],
        tids: &[u32],
    ) -> Result<Vec<Vec<(u32, f32)>>, String> {
        let t = xs.len();
        let dx = self.stage_xs(xs)?;
        let dsc = self.run_gate(dx, t)?;
        let (ids, wts) = self.run_route(dsc, t, Some(tids))?;
        self.readback_route(ids, wts, t)
    }

    /// 라우팅 단독 API(라우티드) — 원천: route_routed L87-96.
    pub fn moe_route_routed(&mut self, xs: &[Vec<f32>]) -> Result<Vec<Vec<(u32, f32)>>, String> {
        let t = xs.len();
        let dx = self.stage_xs(xs)?;
        let dsc = self.run_gate(dx, t)?;
        let (ids, wts) = self.run_route(dsc, t, None)?;
        self.readback_route(ids, wts, t)
    }

    /// f32 포인터 배열 GEMV(mm_paired 계급) — out[p][o] = Σ_i x[p][i]·W_p[i][o]
    /// (ds4_gemv_f32_ptr, k-major). 게이트·전문가·공유 공용.
    fn gemv_f32_ptrs(
        &mut self,
        x: CUdeviceptr,
        out: CUdeviceptr,
        n_in: usize,
        n_out: usize,
        ptrs: &[CUdeviceptr],
    ) -> Result<(), String> {
        let np = ptrs.len();
        if np == 0 || np > 65535 {
            return Err(format!(
                "ds4_moe gemv: np={np} — (0, 65535] 도메인(그리드 y 상한)"
            ));
        }
        if n_in == 0 || n_out == 0 || n_in > i32::MAX as usize || n_out > i32::MAX as usize {
            return Err(format!(
                "ds4_moe gemv: n_in={n_in} n_out={n_out} — (0, 2^31) 도메인"
            ));
        }
        let wp = self.dwptr.ensure(&self.cc, np * 8)?;
        let mut pb = Vec::with_capacity(np * 8);
        for v in ptrs {
            pb.extend_from_slice(&v.to_le_bytes());
        }
        self.cc.h2d(wp, &pb)?;
        let f = self.cc.function("ds4_gemv_f32_ptr")?;
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

    /// FP8-sim 제자리(128블록) — ds4_fp8_sim. rows×cols 그리드.
    fn fp8_sim_inplace(
        &mut self,
        buf: CUdeviceptr,
        rows: usize,
        cols: usize,
    ) -> Result<(), String> {
        if rows == 0 || cols == 0 {
            return Err(format!(
                "ds4_moe fp8_sim: rows={rows} cols={cols} — 0 도메인 위반"
            ));
        }
        let f = self.cc.function("ds4_fp8_sim")?;
        let (mut a0, mut a1) = (buf, cols as i32);
        let mut args: [*mut std::ffi::c_void; 2] =
            [(&mut a0) as *mut _ as *mut _, (&mut a1) as *mut _ as *mut _];
        self.cc
            .launch(f, (rows * cols.div_ceil(128)) as u32, 1, 128, &mut args)
    }

    /// bf16 경계 제자리 — ds4_bf16_round(전 원소).
    fn bf16_round_buf(&mut self, buf: CUdeviceptr, n: usize) -> Result<(), String> {
        if n == 0 || n > i32::MAX as usize {
            return Err(format!("ds4_moe bf16: n={n} — (0, 2^31) 도메인"));
        }
        let f = self.cc.function("ds4_bf16_round")?;
        let (mut a0, mut a1) = (buf, n as i32);
        let mut args: [*mut std::ffi::c_void; 2] =
            [(&mut a0) as *mut _ as *mut _, (&mut a1) as *mut _ as *mut _];
        self.cc.launch(f, n.div_ceil(128) as u32, 1, 128, &mut args)
    }

    /// SwiGLU+가중치+bf16 — ds4_swiglu. h는 g와 동일 버퍼(제자리 허용).
    fn swiglu_inplace(
        &mut self,
        g: CUdeviceptr,
        u: CUdeviceptr,
        wp: CUdeviceptr,
        rows: usize,
        n: usize,
    ) -> Result<(), String> {
        if rows == 0 || rows > 65535 || n == 0 || n > i32::MAX as usize {
            return Err(format!("ds4_moe swiglu: rows={rows} n={n} — 도메인 위반"));
        }
        let f = self.cc.function("ds4_swiglu")?;
        let (mut a0, mut a1, mut a2, mut a3) = (g, u, wp, g);
        let (mut nn, mut lim) = (n as i32, self.dims.swiglu_limit);
        let mut args: [*mut std::ffi::c_void; 6] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
            (&mut a3) as *mut _ as *mut _,
            (&mut nn) as *mut _ as *mut _,
            (&mut lim) as *mut _ as *mut _,
        ];
        self.cc
            .launch(f, n.div_ceil(128) as u32, rows as u32, 128, &mut args)
    }

    /// MoE FFN 공통 파이프라인 — 원천: moe_forward L133-216.
    /// 순서 계약: 게이트(L146-159) → 라우팅 → 토큰-메이저 (ti,e,w) 페어
    /// (L168-176 by_expert 계약 — 토큰 내 e 오름차) → 전문가 fp8→w1/w3→
    /// bf16→swiglu(×w)→bf16→fp8→w2→bf16(expert_ffn L104-130) → 공유
    /// (L199-212, w=1.0) → 결합(eid 오름차 누산+공유+최종 bf16).
    fn moe_ffn_impl(
        &mut self,
        xs: &[Vec<f32>],
        tids: Option<&[u32]>,
    ) -> Result<Vec<Vec<f32>>, String> {
        let d = self.dims.clone();
        let t = xs.len();
        let dx = self.stage_xs(xs)?;
        let dsc = self.run_gate(dx, t)?;
        let (dids, dwts) = self.run_route(dsc, t, tids)?;
        let sel = self.readback_route(dids, dwts, t)?;
        // 페어 구성 — 커널 출력은 이미 eid 오름차(정렬 계약) — 토큰-메이저.
        let mut pairs: Vec<(u32, u32, f32)> = Vec::with_capacity(t * d.n_active);
        for (ti, row) in sel.iter().enumerate() {
            for &(e, w) in row {
                pairs.push((ti as u32, e, w));
            }
        }
        let np = pairs.len();
        // 전문가 3 role — 페어 게더 → fp8-sim 제자리 → 포인터 배열 GEMV.
        let dyo = if np > 0 {
            let mut xp = Vec::with_capacity(np * d.dim);
            for &(ti, _, _) in &pairs {
                xp.extend_from_slice(&xs[ti as usize]);
            }
            let dxp = self.dxp.ensure(&self.cc, np * d.dim * 4)?;
            self.cc.h2d(dxp, &Self::f32_bytes(&xp))?;
            self.fp8_sim_inplace(dxp, np, d.dim)?;
            let need_expert =
                |which: &str, p: CUdeviceptr, e: u32| -> Result<CUdeviceptr, String> {
                    if p == 0 {
                        Err(format!(
                            "ds4_moe ffn: {which} 전문가 e={e} 미상주 — add_expert_f32 먼저"
                        ))
                    } else {
                        Ok(p)
                    }
                };
            let g = self.da.ensure(&self.cc, np * d.inter * 4)?;
            let p1: Vec<CUdeviceptr> = pairs
                .iter()
                .map(|&(_, e, _)| need_expert("w1", self.eg1[e as usize], e))
                .collect::<Result<_, _>>()?;
            self.gemv_f32_ptrs(dxp, g, d.dim, d.inter, &p1)?;
            self.bf16_round_buf(g, np * d.inter)?;
            let u = self.db.ensure(&self.cc, np * d.inter * 4)?;
            let p3: Vec<CUdeviceptr> = pairs
                .iter()
                .map(|&(_, e, _)| need_expert("w3", self.eg3[e as usize], e))
                .collect::<Result<_, _>>()?;
            self.gemv_f32_ptrs(dxp, u, d.dim, d.inter, &p3)?;
            self.bf16_round_buf(u, np * d.inter)?;
            // 라우팅 가중치 페어 업로드(swiglu ×w — expert_ffn L119).
            let dwp = self.dwp.ensure(&self.cc, np * 4)?;
            let wb: Vec<u8> = pairs
                .iter()
                .flat_map(|&(_, _, w)| w.to_le_bytes())
                .collect();
            self.cc.h2d(dwp, &wb)?;
            // swiglu h는 g 제자리(bf16 포함) → fp8-sim → w2 down.
            self.swiglu_inplace(g, u, dwp, np, d.inter)?;
            self.fp8_sim_inplace(g, np, d.inter)?;
            let dyo = self.dyo.ensure(&self.cc, np * d.dim * 4)?;
            let p2: Vec<CUdeviceptr> = pairs
                .iter()
                .map(|&(_, e, _)| need_expert("w2", self.eg2[e as usize], e))
                .collect::<Result<_, _>>()?;
            self.gemv_f32_ptrs(g, dyo, d.inter, d.dim, &p2)?;
            self.bf16_round_buf(dyo, np * d.dim)?;
            dyo
        } else {
            self.dyo.ensure(&self.cc, 4)?
        };
        // 공유 전문가 — 전 토큰 배치(w=1.0 무스케일 — moe_forward L199-206).
        let dshx = self.dshx.ensure(&self.cc, t * d.dim * 4)?;
        let mut flat = Vec::with_capacity(t * d.dim);
        for x in xs {
            flat.extend_from_slice(x);
        }
        self.cc.h2d(dshx, &Self::f32_bytes(&flat))?;
        self.fp8_sim_inplace(dshx, t, d.dim)?;
        let sga = self.da.ensure(&self.cc, t * d.inter * 4)?;
        self.gemv_f32_ptrs(dshx, sga, d.dim, d.inter, &vec![self.sw1; t])?;
        self.bf16_round_buf(sga, t * d.inter)?;
        let sub = self.db.ensure(&self.cc, t * d.inter * 4)?;
        self.gemv_f32_ptrs(dshx, sub, d.dim, d.inter, &vec![self.sw3; t])?;
        self.bf16_round_buf(sub, t * d.inter)?;
        let dwp = self.dwp.ensure(&self.cc, t.max(np) * 4)?;
        let ones: Vec<u8> = (0..t).flat_map(|_| 1.0f32.to_le_bytes()).collect();
        self.cc.h2d(dwp, &ones)?;
        self.swiglu_inplace(sga, sub, dwp, t, d.inter)?;
        self.fp8_sim_inplace(sga, t, d.inter)?;
        let dsh = self.dsh.ensure(&self.cc, t * d.dim * 4)?;
        self.gemv_f32_ptrs(sga, dsh, d.inter, d.dim, &vec![self.sw2; t])?;
        self.bf16_round_buf(dsh, t * d.dim)?;
        // 결합 — 페어 오프셋 [t+1](토큰-메이저·토큰 내 eid 오름차).
        let dpo = self.dpoff.ensure(&self.cc, (t + 1) * 4)?;
        let mut pob = Vec::with_capacity((t + 1) * 4);
        let mut acc = 0i32;
        pob.extend_from_slice(&acc.to_le_bytes());
        for row in &sel {
            acc += row.len() as i32;
            pob.extend_from_slice(&acc.to_le_bytes());
        }
        self.cc.h2d(dpo, &pob)?;
        let dout = self.dout.ensure(&self.cc, t * d.dim * 4)?;
        let f = self.cc.function("ds4_combine")?;
        let (mut a0, mut a1, mut a2, mut a3) = (dyo, dpo, dsh, dout);
        let mut ne = d.dim as i32;
        let mut args: [*mut std::ffi::c_void; 5] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
            (&mut a3) as *mut _ as *mut _,
            (&mut ne) as *mut _ as *mut _,
        ];
        self.cc.launch(f, t as u32, 1, 256, &mut args)?;
        let mut ob = vec![0u8; t * d.dim * 4];
        self.cc.d2h(&mut ob, dout)?;
        self.cc.sync()?;
        // SAFETY: d2h·sync 완료 후 재해석(길이·정렬 일치 — G2 판독 패턴).
        let flat = unsafe { std::slice::from_raw_parts(ob.as_ptr() as *const f32, t * d.dim) };
        Ok(flat.chunks(d.dim).map(<[f32]>::to_vec).collect())
    }

    /// 해시 층 MoE FFN(L0-2) — tid2eid 행 룩업 라우팅.
    /// 원천: moe_forward is_hash=true 경로(L152-159).
    pub fn moe_ffn_hash(&mut self, xs: &[Vec<f32>], tids: &[u32]) -> Result<Vec<Vec<f32>>, String> {
        self.moe_ffn_impl(xs, Some(tids))
    }

    /// 라우티드 층 MoE FFN(L3+) — noaux_tc top-k 라우팅.
    /// 원천: moe_forward is_hash=false 경로(L160-165).
    pub fn moe_ffn_routed(&mut self, xs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
        self.moe_ffn_impl(xs, None)
    }
}

impl Drop for Ds4MoeCuda {
    fn drop(&mut self) {
        // SAFETY: 각 포인터는 이 ctx의 alloc 산출물이며 drop에서 1회 해제.
        let r = (|| {
            for p in self
                .eg1
                .iter()
                .chain(self.eg2.iter())
                .chain(self.eg3.iter())
                .chain([
                    &self.w_gate,
                    &self.d_bias,
                    &self.d_tid2eid,
                    &self.sw1,
                    &self.sw2,
                    &self.sw3,
                ])
            {
                if *p != 0 {
                    self.cc.free(*p)?;
                }
            }
            for b in [
                &mut self.dx,
                &mut self.dsc,
                &mut self.dtids,
                &mut self.dids,
                &mut self.dwts,
                &mut self.dxp,
                &mut self.dwptr,
                &mut self.da,
                &mut self.db,
                &mut self.dyo,
                &mut self.dshx,
                &mut self.dsh,
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
            eprintln!("ds4_moe_cuda: drop 해제 실패: {e}");
        }
    }
}
