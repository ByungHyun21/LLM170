//! Flash-Next(qwen4exp) GDN 스테이지 모듈층 — plans/124 FNF(fn-gdn),
//! 2026-10-05. 원천(유일 기준): crates/core/src/qwen4exp/stages/gdn.rs
//! gdn_layer L11-170 + 그 호출원 core/src/gdn.rs·gdn_norm.rs·ops.rs.
//! 투영(qkv/z/b/a 그룹 GEMM L20-53·ssm_out L169)은 FNF 범위 밖(계약지도
//! REUSE(조건부) — krate 7 이슈로 후속 목표) — 본 모듈은 투영 **이후**
//! 스테이지 산술(conv 링 → β/e^g·l2 → scan → norm_gated sigmoid)만 담는.
//!
//! ═══════════════════════════════════════════════════════════════════════
//! [FNF 재사용-판정표 — qwen4exp gdn_layer vs G5 EXL3 GDN, 2026-10-05 종결]
//! ═══════════════════════════════════════════════════════════════════════
//! 산술 진실의 계층(plans/124 §6): 커널 → **core CPU 참조(값 maxdiff 판정의
//! 유일 기준)** → 수학. 좌변 원천 = core 파일:행, 우변 = G5 착지분.
//!
//! ┌─ 단계 ────┬─ qwen4exp 원천(core) ──────────┬─ G5 EXL3(착지) ──┬─ 판정 ─┐
//! │ conv 구조 │ stages/gdn.rs L69-85: conv_k=4 │ gdn_cuda.rs      │ REUSE  │
//! │           │ — sum = w[c·4+3]·x + Σ_{j<3}   │ set_gdn L150:    │ (실측 │
//! │           │ w[c·4+j]·s_j, silu, 링 [3][ch] │ cw [n][ch][4]·   │ 비트   │
//! │           │ 시프트(s0←s1←s2←x). 실측 GGUF  │ exl3_gdn.cu      │ 동일)  │
//! │           │ ssm_conv1d F32 ne=[4,10240] =  │ exl3_gdn_conv    │        │
//! │           │ [c][4] 채널-주요(플랫 직결).    │ L119: o=w3·xt+   │        │
//! │           │                                │ w0·h0+w1·h1+     │        │
//! │           │                                │ w2·h2, 링 [3][ch]│        │
//! │           │                                │ — **동일 DAG**.  │        │
//! │           │ ※ FNA 지도 "G5 conv 3탭 고정"은 링 깊이 3(=conv_k−1)      │
//! │           │   오독 — 커널은 w0..w3 전량 판독(4탭). 27B EXL3 conv1d도  │
//! │           │   [ch][4]로 실측(G5 set_gdn 계약 자체가 증거).            │
//! ├───────────┼────────────────────────────────┼─────────────────┼────────┤
//! │ β/e^g 사전│ stages/gdn.rs L57-58: β=       │ exl3_gdn_l2perm  │ PORT   │
//! │           │ sigmoid(b), g=softplus(a+      │ L254: a/b를 잔류 │ (신규  │
//! │           │ dt_bias)·ssm_a — b/a는 dt_rank │ xn·abuf **도트** │ fn_gdn_│
//! │           │ 폭 투영 입력, ssm_a는 GGUF 원값 │ 로 산출, ssm_a=  │ prep)  │
//! │           │ (EXL3 A_log의 −exp 환산 불필요)│ −exp(alog).      │        │
//! ├───────────┼────────────────────────────────┼─────────────────┼────────┤
//! │ q/k l2    │ stages/gdn.rs L86-97: n_group= │ exl3_gdn_l2perm  │ PORT   │
//! │           │ 16헤드 l2_norm(ops.rs L39-44,  │ L254: 가산 eps   │ (동일  │
//! │           │ **eps floor** 1/max(√Σ,eps)),  │ 1/√(Σ+eps) + lc │ 커널에  │
//! │           │ **lc 순열 없음**(자연 헤드 순서)│ 순열 scatter 동반│ 서술)  │
//! ├───────────┼────────────────────────────────┼─────────────────┼────────┤
//! │ scan 알고 │ core/gdn.rs gdn_chunk_seq L22· │ exl3_gdn_scan    │ PORT   │
//! │ 리즘      │ gdn_chunk_head L66-197 —       │ L300: 청크 대수  │ (신규  │
//! │           │ qwen35(27B)와 **공유 함수**(    │ 동일(후진 소거·  │ fn_gdn_│
//! │           │ stages/gdn.rs L147 호출). 수학  │ S0≠0 상태 경로). │ scan)  │
//! │           │ 은 G5 미러와 동일 — 그러나 **실측│ 재사용 불가 2건:  │ 재사용 │
//! │           │ 재사용 불가**: (1) gdn_expf k-비트│ (1) gdn_expf 정의│ 불가   │
//! │           │ 재구성이 x≳−709 요구(1023+k≥0) —│ 역 위반 — ssm_a  │ 실측:  │
//! │           │ Flash-Next 실측 ssm_a(−0.028..  │ −158×softplus →  │ ord=0  │
//! │           │ −158)×softplus → T=32 cumsum    │ gcs<−900 → NaN   │ o nan  │
//! │           │ gcs<−900 → 쓰레기 비트. (2) f16 │ 4헤드(o 9088·상태│ 9088·  │
//! │           │ 저장 오차 1.7e-5는 통과지만 게이트│ 65536). (2) f16 │ 상태   │
//! │           │ rms 스케일(≤1/√eps≈1000배)이 소-│ 오차가 게이트    │ nan    │
//! │           │ o 헤드에서 증폭 → 종단 1.7~2.6e-3│ rms로 증폭 →      │ 65536; │
//! │           │ > 임계 2e-4. 포트: f32 전체·CS= │ 종단 2.6e-3 >    │ (2) 종 │
//! │           │ 64·core 동일 적산순서·expf(장치)│ 2e-4.            │ 단 2.6 │
//! │           │ vs libm .exp() ~ulp가 잔차.     │                  │ e-3    │
//! ├───────────┼────────────────────────────────┼─────────────────┼────────┤
//! │ 상태·헤드 │ core: 상태 [h_v][128²](s2·d+dv) │ 동일 레이아웃·   │ (동일)  │
//! │ 매핑      │ kh=h%h_k(gdn.rs L84·L390),     │ kh=h%h_k — 자연  │        │
//! │           │ h_v=dt_rank=48·h_k=n_group=16  │ 순서 입력으로    │        │
//! │           │ = 27B 기하(h_v=48·h_k=16·d=128)│ 지정 시 EXL3의 lc│        │
//! │           │ 와 동일. EXL3의 lc 순열·역순열 │ 순열·게이트 역순 │        │
//! │           │ 은 qwen4exp에 **적용 없음**(자연│ 순열 미발동.     │        │
//! │           │ 순서 입력으로 커널 불변).       │                  │        │
//! ├───────────┼────────────────────────────────┼─────────────────┼────────┤
//! │ 게이트    │ stages/gdn.rs L156-168 →       │ exl3_gdn_gate    │ PORT   │
//! │           │ gdn_norm.rs gdn_norm_gated L26 │ L427: silu 게이트│ (신규  │
//! │           │ — **GdnGate::Sigmoid**(L10,    │ ·f32 트리 rms·   │ fn_gdn_│
//! │           │ qwen35 silu와의 유일 차이)·    │ 역lc 순열.       │ gate)  │
//! │           │ rms=ops.rs sq_sum L11 32세그   │                  │        │
//! │           │ 먼트 f64(L33)·eps=hp.eps.      │                  │        │
//! ├───────────┼────────────────────────────────┼─────────────────┼────────┤
//! │ t=1 AR    │ stages/gdn.rs L135 →           │ fn_gdn_scan n=1  │ PORT   │
//! │           │ core/gdn.rs gdn_ar_batch L375· │ 특수형(동일 대수)│ (승계)  │
//! │           │ gdn_ar_head L326 — scan과 구조 │ — G5 커널은 T=1  │        │
//! │           │ 동일 수학(scale 내부 적용).     │ 에서 KS/QS 패스  │        │
//! │           │                                │ 가 [T][k_len] 버 │        │
//! │           │                                │ 퍼 밖 q 판독(실  │        │
//! │           │                                │ 측) — 포트로 해소│        │
//! └───────────┴────────────────────────────────┴─────────────────┴────────┘
//!
//! 종합 판정: **하이브리드 — conv는 G5 exl3_gdn.fatbin을 수정 없이 재사용
//! (실측 비트동일), β/e^g·l2·게이트·scan은 신규 커널(exl3_fn_gdn.cu)로
//! 이식.** 이식분 중 prep·gate는 ops.rs 트랜센던트·누산 순서까지 미러해
//! core 대비 **비트동일**(프로브 0.000e0 판정), scan은 f32 전체·core 동일
//! 적산순서 + expf 잔차 — 전 단계·종단 값 maxdiff ≤ 2e-4 판정(임치 근거:
//! plans/124 §1 GDN 종단 2e-4).
//!
//! [링/상태 계약] 링 [n_gdn][3][conv_ch]·상태 [n_gdn][dt_rank][128²]은
//! 상주 버퍼로 커널이 r/w(순차 디코드 계약 — gdn_cuda.rs 미러). S0≠0
//! 실입력 검증 의무(plans/124 §3.3 — 합성 S0=0은 상태 버그를 가린다).
//!
//! [속도] 측정 대기 sm_80 — CMP 170HX 미도착(plans/124 §0). 재사용 conv의
//! sm_80 설계 근거는 exl3_gdn.cu 머리 승계, 신규 prep·gate·scan의 근거는
//! exl3_fn_gdn.cu 머리(그리드 (dt_rank,T)·동적 smem 98,816B 1블록/SM).
//!
//! [구조 계약] FnCuda(fn_support)에 스테이지 필드를 추가하지 않는다(공유
//! 파일 충돌 최소화 — FNB-FNE 동시 작업 계약). 본 모듈이 자체 CudaCtx를
//! 소유(단일 상주 원칙 — Exl3CudaDecoder::empty 계급)하며 FNH(프레임 체인)
//! 조립 시 FnCuda 필드 1줄로 위임한다.
//!
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 의존 금지 — scripts/
//! cuda_probe_shim.rs 단독 컴파일.

use crate::rawcuda::ctx::CudaCtx;
use crate::rawcuda::ffi::CUdeviceptr;
use crate::rawcuda::fn_support::FnDims;
/// GDN층 수(compress[il]==0 — 48층 중 36, FnDims.compress 실측 유도).
/// 스캐폴드 상수를 유지한다(FNA 계약지도 인용 대상).
pub const FN_GDN_LAYERS: usize = 36;
/// GDN conv 탭 수(ssm.conv_kernel 실측 4 — 27B EXL3와 동일, FNA 지도의
/// "상이" 기술은 링 깊이 3 오독: 판정표 참조).
pub const FN_GDN_CONV_K: usize = 4;
/// fn_gdn_scan 동적 공유메모리 바이트(assets/exl3_fn_gdn.cu 계약 —
/// qp/kp/d_out 64·128·4B×3 + bp·gcs 64·4B×2 = 98,816B. CUDA 정적
/// __shared__ 한계 48KB 초과 → opt-in 동적 할당; sm_80 164KB·sm_89 99KB
/// 상한 내 1블록/SM).
pub const FN_GDN_SCAN_SMEM: u32 = 98_816;

/// exl3_gdn.fatbin 자산 해석 — gdn_cuda.rs가 쓰는 것과 **동일 자산**
/// (LLM170_CUDA_EXL3_GDN_FATBIN_PATH 오버라이드 규약 미리 — exl3_cuda.rs
/// open_ctx L687 리졸버와 동일 ENV·경로. 자산 경로 오버라이드일 뿐 계산
/// 경로 분기 아님).
fn exl3_gdn_fatbin_bytes() -> Result<Vec<u8>, String> {
    const ENV: &str = "LLM170_CUDA_EXL3_GDN_FATBIN_PATH";
    const REL: &[&str] = &[
        "crates/backend-gpu/src/rawcuda/assets/exl3_gdn.fatbin",
        "src/rawcuda/assets/exl3_gdn.fatbin",
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
        "exl3_gdn.fatbin 없음 — scripts/build_cuda.bat 실행 또는 {ENV} 지정 (탐색: {REL:?})"
    ))
}

/// exl3_fn_gdn.fatbin 자산 해석(신규 — LLM170_CUDA_FN_GDN_FATBIN_PATH).
fn fn_gdn_fatbin_bytes() -> Result<Vec<u8>, String> {
    const ENV: &str = "LLM170_CUDA_FN_GDN_FATBIN_PATH";
    const REL: &[&str] = &[
        "crates/backend-gpu/src/rawcuda/assets/exl3_fn_gdn.fatbin",
        "src/rawcuda/assets/exl3_fn_gdn.fatbin",
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
        "exl3_fn_gdn.fatbin 없음 — scripts/build_cuda.bat 실행 또는 {ENV} 지정 (탐색: {REL:?})"
    ))
}

/// GDN 체인 중간 산출 판독(검증층 진단 — 단계별 값 판정용. 3층 분리:
/// 판독만, 계산 없음).
pub struct FnGdnMids {
    /// conv 산출 q/k/v [t][k_len]/[t][k_len]/[t][v_len].
    pub conv_q: Vec<f32>,
    pub conv_k: Vec<f32>,
    pub conv_v: Vec<f32>,
    /// l2 완료 q/k [t][k_len] · bg [t][2·dt_rank](beta|g 자연 순서).
    pub q2: Vec<f32>,
    pub k2: Vec<f32>,
    pub bg: Vec<f32>,
    /// scan 산출 o_all [t][v_len].
    pub o: Vec<f32>,
    /// norm_gated 산출 gated [t][v_len].
    pub gated: Vec<f32>,
    /// T행 처리 후 링 [3][conv_ch](층 슬라이스)·상태 [dt_rank][128²](층 슬라이스).
    pub ring_post: Vec<f32>,
    pub st_post: Vec<f32>,
}

/// Flash-Next GDN 스테이지 상주체 — 형상 스냅샷 + 가중치/링/상태/작업
/// 버퍼. 커널 재사용 계약: conv·scan = exl3_gdn.fatbin(G5 그대로),
/// prep·gate(+음성대조 conv3) = exl3_fn_gdn.fatbin.
pub struct FnGdnCuda {
    /// 디바이스 컨텍스트(모듈 소유 — 단일 상주 원칙).
    pub cc: CudaCtx,
    n_gdn: usize,
    conv_ch: usize,
    k_len: usize,
    v_len: usize,
    dt_rank: usize,
    d_state: usize,
    n_group: usize,
    conv_k: usize,
    eps: f32,
    // 상주 가중치
    dcw: CUdeviceptr,  // [n_gdn][conv_ch][4]
    ddtb: CUdeviceptr, // [n_gdn][dt_rank]
    dssa: CUdeviceptr, // [n_gdn][dt_rank]
    dnw: CUdeviceptr,  // [n_gdn][128]
    // 상주 상태
    dring: CUdeviceptr, // [n_gdn][conv_k-1][conv_ch]
    dgst: CUdeviceptr,  // [n_gdn][dt_rank][d_state²]
    // 작업 버퍼(t 상한)
    dqkv: CUdeviceptr,
    dz: CUdeviceptr,
    db: CUdeviceptr,
    da: CUdeviceptr,
    dgq: CUdeviceptr,
    dgk: CUdeviceptr,
    dgv: CUdeviceptr,
    dq2: CUdeviceptr,
    dk2: CUdeviceptr,
    dbg: CUdeviceptr,
    dgo: CUdeviceptr,
    dgate: CUdeviceptr,
    t_cap: usize,
}

impl FnGdnCuda {
    /// 개방 — 형상(FnDims 서브셋)·가중치 등록 + 링/상태 제로 상주 할당.
    /// conv_k는 4 고정(exl3_gdn_conv 링 [3] 계약 — 판정표), dt_rank·d_state·
    /// n_group은 계약값 48/128/16 검증(형상 명시 등록, 추정 금지).
    /// 인자는 전 GDN층 배열(커널 layer 인자 인덱싱): cw [n_gdn][conv_ch][4]
    /// (GGUF ssm_conv1d 플랫 [c][4] 그대로 — ne=[4,conv_ch] 행렉스)·
    /// dtb/ssa [n_gdn][dt_rank]·nw [n_gdn][128].
    pub fn open(
        dims: &FnDims,
        eps: f32,
        cw: &[f32],
        dtb: &[f32],
        ssa: &[f32],
        nw: &[f32],
    ) -> Result<Self, String> {
        // compress[il]==0 → GDN 서수 매핑(전층 등록 계약).
        let gdn_ils: Vec<usize> = (0..dims.n_layer)
            .filter(|&il| dims.compress[il] == 0)
            .collect();
        let n_gdn = gdn_ils.len();
        if n_gdn != FN_GDN_LAYERS {
            return Err(format!("GDN층 수 {n_gdn} != {FN_GDN_LAYERS}(계약)"));
        }
        if dims.conv_k != FN_GDN_CONV_K {
            return Err(format!(
                "conv_k {} != {FN_GDN_CONV_K} — exl3_gdn_conv 링[3] 계약 위반",
                dims.conv_k
            ));
        }
        if (dims.dt_rank, dims.d_state, dims.n_group) != (48, 128, 16) {
            return Err(format!(
                "GDN 형상 ({},{},{}) != (48,128,16) 계약",
                dims.dt_rank, dims.d_state, dims.n_group
            ));
        }
        let (conv_ch, k_len, v_len) = (
            dims.gdn_conv_ch(),
            dims.n_group * dims.d_state,
            dims.gdn_v_len(),
        );
        if cw.len() != n_gdn * conv_ch * 4
            || dtb.len() != n_gdn * dims.dt_rank
            || ssa.len() != n_gdn * dims.dt_rank
            || nw.len() != n_gdn * 128
        {
            return Err(format!(
                "GDN 가중치 길이 위반: cw={} dtb={} ssa={} nw={} (기대 {}/{}/{}/{})",
                cw.len(),
                dtb.len(),
                ssa.len(),
                nw.len(),
                n_gdn * conv_ch * 4,
                n_gdn * dims.dt_rank,
                n_gdn * dims.dt_rank,
                n_gdn * 128
            ));
        }
        let mut cc = CudaCtx::new()?;
        let _g = cc.guard()?;
        // 재사용fatbin(G5) — conv만 해석(l2perm/scan/gate은 G5 검증층·
        // FNF 포트분 소관 — 판정표).
        let img = exl3_gdn_fatbin_bytes()?;
        cc.load_fatbin("exl3gdn", &img, &["exl3_gdn_conv"])?;
        // 신규fatbin(FNF) — prep·scan·gate·conv3(음성대조 계기).
        let img2 = fn_gdn_fatbin_bytes()?;
        cc.load_fatbin(
            "fngdn",
            &img2,
            &["fn_gdn_prep", "fn_gdn_scan", "fn_gdn_gate", "fn_gdn_conv3"],
        )?;
        // SAFETY: f32 슬라이스 → 바이트 뷰(길이·정렬 일치 — G2 패턴).
        let b =
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        let dcw = cc.alloc(cw.len() * 4)?;
        cc.h2d(dcw, b(cw))?;
        let ddtb = cc.alloc(dtb.len() * 4)?;
        cc.h2d(ddtb, b(dtb))?;
        let dssa = cc.alloc(ssa.len() * 4)?;
        cc.h2d(dssa, b(ssa))?;
        let dnw = cc.alloc(nw.len() * 4)?;
        cc.h2d(dnw, b(nw))?;
        let dring = cc.alloc(n_gdn * 3 * conv_ch * 4)?;
        cc.h2d(dring, &vec![0u8; n_gdn * 3 * conv_ch * 4])?;
        let dgst = cc.alloc(n_gdn * dims.dt_rank * 128 * 128 * 4)?;
        cc.h2d(dgst, &vec![0u8; n_gdn * dims.dt_rank * 128 * 128 * 4])?;
        Ok(FnGdnCuda {
            cc,
            n_gdn,
            conv_ch,
            k_len,
            v_len,
            dt_rank: dims.dt_rank,
            d_state: dims.d_state,
            n_group: dims.n_group,
            conv_k: dims.conv_k,
            eps,
            dcw,
            ddtb,
            dssa,
            dnw,
            dring,
            dgst,
            dqkv: 0,
            dz: 0,
            db: 0,
            da: 0,
            dgq: 0,
            dgk: 0,
            dgv: 0,
            dq2: 0,
            dk2: 0,
            dbg: 0,
            dgo: 0,
            dgate: 0,
            t_cap: 0,
        })
    }

    /// 등록 GDN층 수(compress[il]==0 서수 공간 크기).
    pub fn n_gdn(&self) -> usize {
        self.n_gdn
    }

    /// conv 탭 수(계약 4 — 판정표: G5 링 깊이 3 = conv_k−1).
    pub fn conv_k(&self) -> usize {
        self.conv_k
    }

    /// GDN 작업 버퍼 보장(t 상한 확장 시에만 재할당 — gdn_cuda.rs 계약).
    fn ensure_bufs(&mut self, t_len: usize) -> Result<(), String> {
        if t_len <= self.t_cap {
            return Ok(());
        }
        let (cch, kl, vl, dr) = (self.conv_ch, self.k_len, self.v_len, self.dt_rank);
        for q in [
            self.dqkv, self.dz, self.db, self.da, self.dgq, self.dgk, self.dgv, self.dq2, self.dk2,
            self.dbg, self.dgo, self.dgate,
        ] {
            if q != 0 {
                self.cc.free(q)?;
            }
        }
        self.dqkv = self.cc.alloc(t_len * cch * 4)?;
        self.dz = self.cc.alloc(t_len * vl * 4)?;
        self.db = self.cc.alloc(t_len * dr * 4)?;
        self.da = self.cc.alloc(t_len * dr * 4)?;
        self.dgq = self.cc.alloc(t_len * kl * 4)?;
        self.dgk = self.cc.alloc(t_len * kl * 4)?;
        self.dgv = self.cc.alloc(t_len * vl * 4)?;
        self.dq2 = self.cc.alloc(t_len * kl * 4)?;
        self.dk2 = self.cc.alloc(t_len * kl * 4)?;
        self.dbg = self.cc.alloc(t_len * 2 * dr * 4)?;
        self.dgo = self.cc.alloc(t_len * vl * 4)?;
        self.dgate = self.cc.alloc(t_len * vl * 4)?;
        self.t_cap = t_len;
        Ok(())
    }

    /// GDN 스테이지 디바이스 4발사(conv → prep → scan → gate) — 링·상태는
    /// 상주 버퍼(r/w). conv3=true는 음성대조 계기(fn_gdn_conv3 — 3탭 오독
    /// 재현, 원장 17호. 정상 호출 금지). 그리드 계약(결함 5호): prep·gate는
    /// (dt_rank, T) — t=blockIdx.y, conv는 (conv_ch/128, 1), scan은
    /// (dt_rank, 1)·동적 공유 GDN_SCAN_SMEM(gdn_cuda.rs L33 계약 승계).
    fn gdn_stage_dev(&mut self, layer: usize, t_len: usize, conv3: bool) -> Result<(), String> {
        if layer >= self.n_gdn {
            return Err(format!("GDN layer={layer} >= n_gdn={}", self.n_gdn));
        }
        if t_len == 0 {
            return Err("GDN t_len=0".into());
        }
        self.ensure_bufs(t_len)?;
        let (mut tl, mut lay) = (t_len as i32, layer as i32);
        let (mut hk, mut hv, mut dd) = (
            self.n_group as i32,
            self.dt_rank as i32,
            self.d_state as i32,
        );
        let (mut kl, mut vl, mut cch, mut ng) = (
            self.k_len as i32,
            self.v_len as i32,
            self.conv_ch as i32,
            self.n_group as i32,
        );
        let mut eps = self.eps;

        // conv — REUSE exl3_gdn_conv(G5, 판정표: 동일 4탭 DAG) 또는 음성
        // 대조 fn_gdn_conv3. 채널축 1D 그리드(블록 128 = 완전 coalesce),
        // T행은 커널이 순차 회전(링 계약 — 한 런치).
        let f = if conv3 {
            self.cc.function("fn_gdn_conv3")?
        } else {
            self.cc.function("exl3_gdn_conv")?
        };
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
            .launch(f, (self.conv_ch / 128) as u32, 1, 128, &mut ac)?;

        // prep — NEW fn_gdn_prep: β/e^g 사전 + q/k l2(판정표 PORT).
        // 그리드 (dt_rank, T), 블록 128.
        let f = self.cc.function("fn_gdn_prep")?;
        let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7, mut p8) = (
            self.db, self.da, self.ddtb, self.dssa, self.dgq, self.dgk, self.dq2, self.dk2,
            self.dbg,
        );
        let mut ap: [*mut std::ffi::c_void; 14] = [
            (&mut p0) as *mut _ as *mut _,
            (&mut p1) as *mut _ as *mut _,
            (&mut p2) as *mut _ as *mut _,
            (&mut p3) as *mut _ as *mut _,
            (&mut p4) as *mut _ as *mut _,
            (&mut p5) as *mut _ as *mut _,
            (&mut p6) as *mut _ as *mut _,
            (&mut p7) as *mut _ as *mut _,
            (&mut p8) as *mut _ as *mut _,
            (&mut tl) as *mut _ as *mut _,
            (&mut lay) as *mut _ as *mut _,
            (&mut ng) as *mut _ as *mut _,
            (&mut hv) as *mut _ as *mut _,
            (&mut eps) as *mut _ as *mut _,
        ];
        self.cc
            .launch(f, self.dt_rank as u32, t_len as u32, 128, &mut ap)?;

        // scan — PORT fn_gdn_scan: f32 전체·CS=64·core 동일 적산순서
        // (판정표 — G5 커널은 gdn_expf 정의역·f16 오차 증폭으로 재사용
        // 불가, 실측 근거 표 참조). 그리드 (dt_rank, 1), 블록 128,
        // 동적 공유 98,816B.
        let f = self.cc.function("fn_gdn_scan")?;
        self.cc.set_dynamic_smem(f, FN_GDN_SCAN_SMEM)?;
        let (mut s0, mut s1, mut s2, mut s3, mut s4, mut s5) =
            (self.dq2, self.dk2, self.dgv, self.dbg, self.dgst, self.dgo);
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
            .launch_shared(f, self.dt_rank as u32, 1, 128, FN_GDN_SCAN_SMEM, &mut as_)?;

        // gate — NEW fn_gdn_gate: norm_gated sigmoid(판정표 PORT).
        // 그리드 (dt_rank, T), 블록 128.
        let f = self.cc.function("fn_gdn_gate")?;
        let (mut g0, mut g1, mut g2, mut g3) = (self.dgo, self.dz, self.dnw, self.dgate);
        let mut ag: [*mut std::ffi::c_void; 8] = [
            (&mut g0) as *mut _ as *mut _,
            (&mut g1) as *mut _ as *mut _,
            (&mut g2) as *mut _ as *mut _,
            (&mut g3) as *mut _ as *mut _,
            (&mut tl) as *mut _ as *mut _,
            (&mut lay) as *mut _ as *mut _,
            (&mut hv) as *mut _ as *mut _,
            (&mut eps) as *mut _ as *mut _,
        ];
        self.cc
            .launch(f, self.dt_rank as u32, t_len as u32, 128, &mut ag)?;
        Ok(())
    }

    /// GDN 스테이지 호스트 진입: qkv/z/b/a 업로드 → 4커널 → gated
    /// [t][v_len] 판독. b/a는 투영 원본(sigmoid/softplus 전 — prep 커널이
    /// 사전 계산, 원천 stages/gdn.rs L57-58). s0/ring0 = Some면 해당 층의
    /// 상태/링을 시드로 선업로드(**S0≠0 검증 의무** — None이면 상주값,
    /// 최초는 제로). 상태·링은 체인 후 갱신된 채 상주(순차 디코드 계약).
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_stage_host(
        &mut self,
        layer: usize,
        t_len: usize,
        qkv: &[f32],
        z: &[f32],
        b: &[f32],
        a: &[f32],
        s0: Option<&[f32]>,
        ring0: Option<&[f32]>,
    ) -> Result<Vec<f32>, String> {
        if qkv.len() != t_len * self.conv_ch
            || z.len() != t_len * self.v_len
            || b.len() != t_len * self.dt_rank
            || a.len() != t_len * self.dt_rank
        {
            return Err(format!(
                "GDN 입력 길이 위반: qkv={} z={} b={} a={} (t={t_len} conv_ch={} v_len={} dt_rank={})",
                qkv.len(),
                z.len(),
                b.len(),
                a.len(),
                self.conv_ch,
                self.v_len,
                self.dt_rank
            ));
        }
        self.ensure_bufs(t_len)?;
        self.seed_state(layer, s0, ring0)?;
        // SAFETY: f32 슬라이스 → 바이트 뷰(길이·정렬 일치).
        let bv =
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        self.cc.h2d(self.dqkv, bv(qkv))?;
        self.cc.h2d(self.dz, bv(z))?;
        self.cc.h2d(self.db, bv(b))?;
        self.cc.h2d(self.da, bv(a))?;
        self.gdn_stage_dev(layer, t_len, false)?;
        let mut ob = vec![0u8; t_len * self.v_len * 4];
        self.cc.d2h(&mut ob, self.dgate)?;
        self.cc.sync()?;
        // SAFETY: d2h 완료 후 재해석(길이·정렬 일치 — G2 판독 패턴).
        Ok(unsafe {
            std::slice::from_raw_parts(ob.as_ptr() as *const f32, t_len * self.v_len).to_vec()
        })
    }

    /// 음성대조 계기 진입(원장 17호): conv를 fn_gdn_conv3(3탭 오독)으로
    /// 발사한 체인. 검증층(fn_gdn_negative_check) 전용 API.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_stage_host_conv3(
        &mut self,
        layer: usize,
        t_len: usize,
        qkv: &[f32],
        z: &[f32],
        b: &[f32],
        a: &[f32],
        s0: Option<&[f32]>,
        ring0: Option<&[f32]>,
    ) -> Result<Vec<f32>, String> {
        if qkv.len() != t_len * self.conv_ch
            || z.len() != t_len * self.v_len
            || b.len() != t_len * self.dt_rank
            || a.len() != t_len * self.dt_rank
        {
            return Err("GDN 입력 길이 위반(conv3 진입)".into());
        }
        self.ensure_bufs(t_len)?;
        self.seed_state(layer, s0, ring0)?;
        // SAFETY: f32 슬라이스 → 바이트 뷰(길이·정렬 일치).
        let bv =
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        self.cc.h2d(self.dqkv, bv(qkv))?;
        self.cc.h2d(self.dz, bv(z))?;
        self.cc.h2d(self.db, bv(b))?;
        self.cc.h2d(self.da, bv(a))?;
        self.gdn_stage_dev(layer, t_len, true)?;
        let mut ob = vec![0u8; t_len * self.v_len * 4];
        self.cc.d2h(&mut ob, self.dgate)?;
        self.cc.sync()?;
        // SAFETY: d2h 완료 후 재해석.
        Ok(unsafe {
            std::slice::from_raw_parts(ob.as_ptr() as *const f32, t_len * self.v_len).to_vec()
        })
    }

    /// 층 상태/링 시드 선업로드(S0≠0 검증 경로 — gdn_cuda.rs 미러).
    fn seed_state(
        &mut self,
        layer: usize,
        s0: Option<&[f32]>,
        ring0: Option<&[f32]>,
    ) -> Result<(), String> {
        // SAFETY: f32 슬라이스 → 바이트 뷰(길이·정렬 일치).
        let bv =
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        if let Some(s) = s0 {
            if s.len() != self.dt_rank * 128 * 128 {
                return Err(format!("GDN s0 {} != {}x16384", s.len(), self.dt_rank));
            }
            // SAFETY: 층 슬라이스 오프셋 — n_gdn×dt_rank×16384 경계 내.
            let off = self.dgst + (layer * self.dt_rank * 128 * 128) as u64 * 4;
            self.cc.h2d(off, bv(s))?;
        }
        if let Some(r) = ring0 {
            if r.len() != 3 * self.conv_ch {
                return Err(format!("GDN ring0 {} != 3x{}", r.len(), self.conv_ch));
            }
            // SAFETY: 층 슬라이스 오프셋 — n_gdn×3×conv_ch 경계 내.
            let off = self.dring + (layer * 3 * self.conv_ch) as u64 * 4;
            self.cc.h2d(off, bv(r))?;
        }
        Ok(())
    }

    /// GDN 체인 중간 산출 판독(검증층 진단). 직전 gdn_stage_host 실행의
    /// 잔류 버퍼를 읽는다.
    pub fn gdn_mids_host(&mut self, layer: usize, t_len: usize) -> Result<FnGdnMids, String> {
        if self.t_cap < t_len {
            return Err(format!(
                "GDN: mids 판독 전 체인 실행 필요(t={t_len} > cap={})",
                self.t_cap
            ));
        }
        let (kl, vl, cch) = (self.k_len, self.v_len, self.conv_ch);
        let take = |cc: &CudaCtx, n: usize, src: CUdeviceptr| -> Result<Vec<f32>, String> {
            let mut buf = vec![0u8; n * 4];
            cc.d2h(&mut buf, src)?;
            // SAFETY: d2h 완료 후 재해석(길이·정렬 일치).
            Ok(unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const f32, n).to_vec() })
        };
        let conv_q = take(&self.cc, t_len * kl, self.dgq)?;
        let conv_k = take(&self.cc, t_len * kl, self.dgk)?;
        let conv_v = take(&self.cc, t_len * vl, self.dgv)?;
        let q2 = take(&self.cc, t_len * kl, self.dq2)?;
        let k2 = take(&self.cc, t_len * kl, self.dk2)?;
        let bg = take(&self.cc, t_len * 2 * self.dt_rank, self.dbg)?;
        let o = take(&self.cc, t_len * vl, self.dgo)?;
        let gated = take(&self.cc, t_len * vl, self.dgate)?;
        // SAFETY: 층 슬라이스 오프셋 — 경계 내.
        let ring_post = take(&self.cc, 3 * cch, self.dring + (layer * 3 * cch) as u64 * 4)?;
        // SAFETY: 층 슬라이스 오프셋 — 경계 내.
        let st_post = take(
            &self.cc,
            self.dt_rank * 128 * 128,
            self.dgst + (layer * self.dt_rank * 128 * 128) as u64 * 4,
        )?;
        self.cc.sync()?;
        Ok(FnGdnMids {
            conv_q,
            conv_k,
            conv_v,
            q2,
            k2,
            bg,
            o,
            gated,
            ring_post,
            st_post,
        })
    }
}
