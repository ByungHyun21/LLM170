//! W4A16 CUDA 디코더 — 호스트 스테이징 순차 체인 (W3-2).
//!
//! 구 EXL3 CUDA 호스트(exl3_cuda*.rs, B2에서 삭제 — git 586758df^)의
//! 체인 구조를 W4A16 계약으로 개조한 것. 커널 자산(attn/gdn/norm/ew)과
//! 발사 시퀀스는 동일하고, 선형 GEMV만 exl3 trellis → gptq4 split으로
//! 교체한다. 임베딩 행·최종 head는 호출자(서버)가 소유한다 — 이 모듈은
//! 디바이스 체인만 안다.
//!
//! [활성 정밀도] 체인 중간은 f32(CPU 참조와 같은 계급). GEMV 입력만
//! f16으로 캐스팅(RN-even — 커널 계약: assets/gptq4.cu는 f16 비트 입력).
//! 따라서 종단 판정은 비트가 아니라 **토큰열**(골든 대조)이다 — 모듈
//! 비트 계약은 w4a16-gemv/gemm 게이트가 따로 담당.
//!
//! [슬롯] GDN 링/스캔 상태·KV 캐시·pp가 슬롯별(포인터 오프셋). 가중치는
//! 전 슬롯 공유 — 커널 layer 인덱스를 오염시키지 않는다.

use crate::rawcuda::ctx::CudaCtx;
use crate::rawcuda::ffi::{self, CUdeviceptr};
use std::collections::HashMap;

/// fwd3s T 상한(assets/attn.cu ATTN_TMAX와 동일 값).
// 커널 ATTN_TMAX와 동기(2026-10-09: 8→32→128→512 — 프리필 청크 확대).
pub const ATTN_F3S_TMAX: usize = 512;
/// 프리필 배치 상한 — 체인 버퍼·GEMM t 계약(attn fwd3s와 동일 상한).
// 프리필 청크 상한 — 2026-10-09: 8→32→128→512(가중치 재사용 ↑ — 밀집·전문가
// 트래픽이 청크에 상각, MoE 전문가 재사용 ~4→16). mma GEMM(TC ON·t≥16)은
// t 무제한, FFMA 폴백만 32 상한(chain_device_t 가드).
pub const CHAIN_TMAX: usize = 512;
/// [B1/B3 2026-10-10] mma GEMM 타일 미러(assets/gptq4.cu MMA_M/MMA_N —
/// T1/T2, GRP_M/GRP_N — 그룹(MoE). 정적검사: gemm_mma_mirror 테스트).
pub const GEMM_MMA_M: usize = 32;
pub const GEMM_MMA_N: usize = 128; // [marlin-A2] 64→128: A 재판독 절반(L2 바운드)
/// bf16 mma GEMM(플레인) 전용 N — 커널 BMMA_N 미러(워프 매핑 N64 고정).
pub const GEMM_BMMA_N: usize = 64;
pub const GEMM_GRP_M: usize = 64;
pub const GEMM_GRP_N: usize = 32;
/// GDN scan 동적 공유메모리(assets/gdn.cu 계약 — 정적 48KB 초과).
/// gdn_scan 동적 공유메모리(커널 레이아웃 계약 — A5-4: qs 스테이징 +
/// V-타일(GDN_NSPLIT) + Stile 더블 버퍼 기준).
pub const GDN_SCAN_SMEM: u32 = 43_396;
/// [A5-4 2026-10-10 FLA 2단] gdn_scan V-타일 분할 수(커널 GDN_NSPLIT와
/// 동일 계약) — grid = h_v×NSPLIT, 블록 = GDN_NGRP×GDN_VS스레드.
pub const GDN_NSPLIT: usize = 4;
/// [A5-4] gdn_scan 워크그룹 수(커널 GDN_NGRP와 동일 계약).
pub const GDN_NGRP: usize = 16;
/// gdn_scan 청크 크기(커널 GDN_CS와 동일 계약) — prepass 그리드·스크래치 산정.
pub const GDN_CS: usize = 32;
/// [A9 2026-10-10] 배치 디코드 최대 토큰(커널 head.cu HEAD_TMAX 미러).
pub const BATCH_DEC_MAX: usize = 8;
/// [A9-fix2] t행 GEMV 블록당 행 수(커널 gptq4.cu GEMV_TR 미러) — x 재사용.
pub const GEMV_TR: usize = 8;
/// [R21] FFMA 폴백 GEMM의 t 상한(커널 gptq4.cu G4_GTMAX 미러) — 체인 가드용.
pub const GEMM_FFMA_TMAX: usize = 32;

/// [R8 2026-10-10] KV 양자화 모드 — 종전 `kvq: u8`(0/8/4) 분기 16곳의
/// 캡슐화. 바이트 규칙·커널 선택·mem_stats가 이 타입에 모인다.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KvMode {
    /// f32 KV(기본).
    F32,
    /// int8 KV(4× 절감 — `LLM170_KVQ` 비영).
    Int8,
    /// int4 KV(8× 절감, 니블 패킹 — `LLM170_KVQ=4`).
    Int4,
}

impl KvMode {
    /// env 확정 — "4"=Int4, 그 외 비영=Int8(후방 호환), 미설정/0=F32.
    pub fn from_env() -> Self {
        match llm170_diag::flag::val("LLM170_KVQ") {
            Some("4") => KvMode::Int4,
            Some(v) if v != "0" => KvMode::Int8,
            _ => KvMode::F32,
        }
    }
    /// 활성 여부(KVQ 기록/판독 경로 선택).
    pub fn is_on(self) -> bool {
        self != KvMode::F32
    }
    /// 슬롯 요소 e개의 KV 바이트 수(int4 = 니블 패킹 → 절반).
    pub fn kv_bytes(self, elems: usize) -> usize {
        match self {
            KvMode::Int4 => elems / 2,
            KvMode::Int8 => elems,
            KvMode::F32 => elems * 4,
        }
    }
    /// KV 기록 커널.
    pub fn prep_kernel(self) -> &'static str {
        match self {
            KvMode::Int4 => "attn_prep_q4",
            _ => "attn_prep_q",
        }
    }
    /// KV 판독(part) 커널.
    pub fn part_kernel(self) -> &'static str {
        match self {
            KvMode::Int4 => "attn_fwd3s_part_q4",
            _ => "attn_fwd3s_part_q",
        }
    }
}

/// [R5 2026-10-10] 단일 버퍼(ptr+cap) — ensure_*가 쌍을 함께 갱신한다.
#[derive(Clone, Copy, Default)]
pub struct Buf {
    pub ptr: CUdeviceptr,
    pub cap: usize,
}

/// 노름 3버퍼(공유 cap) — ensure_norm_bufs가 그룹 단위 교체.
#[derive(Clone, Copy, Default)]
pub struct NormBufs {
    pub dx: CUdeviceptr,
    pub dab: CUdeviceptr,
    pub dxn: CUdeviceptr,
    pub cap: usize,
}

/// ew 3버퍼(공유 cap).
#[derive(Clone, Copy, Default)]
pub struct EwBufs {
    pub dewg: CUdeviceptr,
    pub dewu: CUdeviceptr,
    pub dew: CUdeviceptr,
    pub cap: usize,
}

/// 전문가 스트리밍 4버퍼(공유 cap).
#[derive(Clone, Copy, Default)]
pub struct ExpBufs {
    pub gate: CUdeviceptr,
    pub up: CUdeviceptr,
    pub act: CUdeviceptr,
    pub dn: CUdeviceptr,
    pub cap: usize,
}

/// 스테이징 q/s(각 cap).
#[derive(Clone, Copy, Default)]
pub struct StgBufs {
    pub q: CUdeviceptr,
    pub s: CUdeviceptr,
    pub cap: (usize, usize),
}

/// [R10 2026-10-10] 체인 선형 경로 — 3체인의 차이를 주입하는 열거.
/// Gemv1: t=1 단독(1행 GEMV — 스테이징 폭 w 검사 동반),
/// Gemm: t≥2 프리필(mma/FFMA GEMM), GemvT: t≤8 배치(TR GEMV).
#[derive(Clone, Copy)]
pub enum LinPath {
    Gemv1 { w: usize },
    Gemm,
    GemvT,
}

/// GDN 체인 형상(서버가 config에서 유도해 명시 등록 — 추정 금지).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GdnDims {
    pub n_gdn: usize,
    pub hidden: usize,
    pub h_k: usize,
    pub h_v: usize,
    pub d: usize,
}

impl GdnDims {
    pub fn conv_ch(&self) -> usize {
        (2 * self.h_k + self.h_v) * self.d
    }
    pub fn k_len(&self) -> usize {
        self.h_k * self.d
    }
    pub fn v_len(&self) -> usize {
        self.h_v * self.d
    }
    pub fn bg_len(&self) -> usize {
        2 * self.h_v
    }
}

/// KV 분할 attention(P8-attn-3) — 긴 컨텍스트에서 블록 수 = q_heads×S.
/// lim ≤ 256(단문·골든 구간)은 종전 단일 블록 경로 그대로(비트 동일).
pub const ATTN_SPLITS: usize = 8; // 진단

/// 어텐션 형상(서버 등록 — d=256·rope 64차 고정 계약).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttnDims {
    pub n_attn: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub d: usize,
    pub cap: usize,
    /// [A11] full attention 간격(모델 config `full_attention_interval`) —
    /// 종전 4 하드코딩. 체인 분기 `(il+1) % interval == 0`과 `il/interval`이
    /// 이 값을 쓴다(신규 비율 모델 대응).
    pub interval: usize,
}

impl AttnDims {
    pub fn q_dim(&self) -> usize {
        self.q_heads * self.d
    }
    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.d
    }
    pub fn qg_dim(&self) -> usize {
        self.q_heads * 2 * self.d
    }
    pub fn kv_slot_elems(&self, slot: usize) -> usize {
        slot * self.n_attn * self.cap * self.kv_dim()
    }
}

/// f32 → f16 비트(RN-even, 서브노멀·inf/nan 처리) — GEMV 활성 캐스팅 계약.
/// 커널은 이 f16 비트를 dot_row_w4a16_lane과 동일 산술로 소비한다.
/// MoE top-k 선택 — 라우터 로짓 → softmax(전문가 전체) → top-k → 재정규화.
/// 시맨틱 단일 출처는 core `qwen35::stages::moe`(크레이트 의존 방향 제약으로
/// 미러 — 변경 시 양쪽 동시 갱신). 반환 = (전문가, 가중) 내림차순.
pub fn moe_topk(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let n = logits.len();
    let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p = vec![0.0f32; n];
    let mut sum = 0.0f32;
    for (i, l) in logits.iter().enumerate() {
        p[i] = (l - mx).exp();
        sum += p[i];
    }
    for v in p.iter_mut() {
        *v /= sum;
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_unstable_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
    idx.truncate(k);
    let wsum: f32 = idx.iter().map(|&e| p[e]).sum();
    idx.iter().map(|&e| (e, p[e] / wsum)).collect()
}

pub fn f32_to_f16(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32;
    let man = b & 0x7F_FFFF;
    if exp == 0xFF {
        return sign | 0x7C00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7C00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let mut sub = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        if rem > half || (rem == half && (sub & 1) == 1) {
            sub += 1;
        }
        return sign | (sub as u16);
    }
    let mut h = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | (h as u16)
}

/// ensure_* 버퍼 교체 계약(G1) — 기존 전량 해제 → 전 필드 0화 → 재할당.
/// alloc이 중간에 실패하면 필드는 0 또는 그때까지의 성공분을 유지한다:
/// 재시도는 성공분만 회수하고(이중해제 없음) 처음부터 다시 할당한다.
/// cap류는 호출부가 0으로 내린 뒤 성공 시에만 갱신할 것.
/// free/alloc을 클로저로 받는 이유: CUDA 없이 모의 주입 단위 테스트(회귀).
fn realloc_fields<const N: usize>(
    mut free: impl FnMut(CUdeviceptr) -> Result<(), String>,
    mut alloc: impl FnMut(usize) -> Result<CUdeviceptr, String>,
    mut fields: [&mut CUdeviceptr; N],
    sizes: [usize; N],
) -> Result<(), String> {
    for f in fields.iter() {
        if **f != 0 {
            free(**f)?;
        }
    }
    for f in fields.iter_mut() {
        **f = 0;
    }
    for (f, sz) in fields.into_iter().zip(sizes) {
        *f = alloc(sz)?;
    }
    Ok(())
}

/// [A8] 캡처 그래프 1개 — (slot, head, argmax)가 캐시 키.
struct GraphEntry {
    exec: ffi::CUgraphExec,
    handle: ffi::CUgraph,
    slot: usize,
    head: bool,
    argmax: bool,
}

pub struct W4a16Dec {
    pub cc: CudaCtx,
    pub hidden: usize,
    pub n_layers: usize,
    pub n_slots: usize,
    pub slot_pos: Vec<u32>,
    /// 선형 상주 — 이름 → (packed, scale, n, k).
    lins: HashMap<String, (CUdeviceptr, CUdeviceptr, usize, usize)>,
    /// 플레인 bf16 상주 — 이름 → ([k][n] 전치 ptr, n, k). MoE 모델(35B)의
    /// 비양자화 선형(GDN·어텐션·라우터·shared) — head_bf16 GEMV 재사용.
    plains: HashMap<String, (CUdeviceptr, usize, usize)>,
    /// 플레인 모드(체인 분기 — 전문가 외 전부 bf16인 모델).
    plain_weights: bool,
    /// 상주 가중치 바이트 합(모니터링 — 업로드 시 누적, 재업로드 없음 계약).
    weights_bytes: u64,
    /// 전문가 테이블 바이트 합(불변 — 1Hz 집계의 O(92k) 반복 제거).
    experts_bytes: u64,
    /// MoE 구성 — n_experts=0이면 dense FFN.
    n_experts: usize,
    top_k: usize,
    moe_ffn: usize,
    shared_ffn: usize,
    moe_group: usize,
    moe_scale_bf16: bool,
    /// 전문가 슬라이스 테이블 — (packed ptr/len, scale ptr/len), 인덱스
    /// ((il·n_experts + e)·3 + proj) — proj 0/1/2 = gate/up/down.
    /// 호스트 포인터는 서버 스토어 mmap 수명 계약(set_expert_table).
    moe_tab: Vec<(u64, u64, u64, u64)>,
    /// 전문가 상주 모드 — moe_tab이 VRAM 포인터(170HX 64GB 등).
    moe_resident: bool,
    /// 전문가 디바이스 포인터 테이블(상주 — 배치 GEMV 간접 참조, (q,s) 쌍).
    moe_dev_tab: CUdeviceptr,
    /// 선택 슬롯 인덱스·가중(호스트 → 디바이스, [top_k]).
    moe_idx: CUdeviceptr,
    moe_wt: CUdeviceptr,
    /// [B-4] 다중 세그먼트 GEMV 테이블(층×3세그먼트×4u64: w,out,n,k) +
    /// 층별 행 합(그리드). 라우터 게이트 + shared gate/up 1런치용.
    moe_multi_tab: CUdeviceptr,
    moe_multi_rows: Vec<u32>,
    /// [P11] 전문가-우선 슬롯 정렬(프리필 그룹 GEMV) — gslot[n_exp×gmax]+cnt.
    moe_gslot: CUdeviceptr,
    moe_gcnt: CUdeviceptr,
    moe_goff: CUdeviceptr,
    moe_gmax: usize,
    /// 배치 전문가 출력([top_k][n_ff] · [top_k][hidden]) — act는 ew 전용
    /// 별도 버퍼(ew 커널 __restrict__ 계약 — 제자리 호출 금지).
    exp: ExpBufs,
    /// MoE 스테이징 — 전문가 packed/scale(1쌍 재사용, 스트림 순서 안전).
    stg: StgBufs,
    /// [C2 2026-10-09] 호스트 RAM 스테이징 — 스트리밍 전문가의 mmap 슬라이스
    /// 1회 복사본(set_expert_table). 스트리밍 업로드가 SSD/페이지캐시 경로에
    /// 의존하지 않게 한다. Vec 버퍼 주소는 불변이므로 moe_tab이 이 안을
    /// 가리켜도 안전(용량 추가 변경 금지 — 구축 후 불변).
    host_stage: Vec<u8>,
    /// [P7-fix 2026-10-10] host_stage 선두에서 cuMemHostRegister에 성공한
    /// 바이트 수(0 = 미등록). 등록 구간 소스는 h2d DMA가 직행한다.
    host_stage_reg: usize,
    /// 라우터 로짓/ shared 게이트 스크래치(n_experts ≥ 1).
    drt: Buf,
    /// MoE 출력(hidden) — 잔차 ab로 소비된다.
    dmo: Buf,
    moe_bufs_ok: bool,
    /// GEMV 스테이징 — x f16 [t][k], y f32 [t][n].
    xh: Buf,
    dy: Buf,
    // ── norm ──
    dnw: CUdeviceptr,
    norm_w_rows: usize,
    norm: NormBufs,
    // ── ew ──
    ew: EwBufs,
    // ── GDN ──
    gdn: Option<GdnDims>,
    dcw: CUdeviceptr,
    dab_c: CUdeviceptr,
    dalog: CUdeviceptr,
    ddtb: CUdeviceptr,
    dnwg: CUdeviceptr,
    dring: CUdeviceptr,
    dgst: CUdeviceptr,
    dqkv: CUdeviceptr,
    dzv: CUdeviceptr,
    dgxn: CUdeviceptr,
    dgq: CUdeviceptr,
    dgk: CUdeviceptr,
    dgv: CUdeviceptr,
    dq2: CUdeviceptr,
    dk2: CUdeviceptr,
    dv2: CUdeviceptr,
    dbg: CUdeviceptr,
    dgo: CUdeviceptr,
    /// [P9] t=1 GDN 분할 부분합([h_v][4][2][128] f32)과 dc([h_v][128]).
    dgpart: CUdeviceptr,
    dgdc: CUdeviceptr,
    /// [A5-4] gdn_scan A/KQ prepass 스크래치 — [2][h_v][maxchunks][CS][CS] half.
    dakq: CUdeviceptr,
    dgate: CUdeviceptr,
    gdn_t_cap: usize,
    // ── attn ──
    attn: Option<AttnDims>,
    dqnw_a: CUdeviceptr,
    dknw_a: CUdeviceptr,
    dkc: CUdeviceptr,
    dvc: CUdeviceptr,
    /// [P13] int8 KV 스케일(K/V 각각 [n_attn*cap][kv_heads] f32) — KVQ 전용.
    dksc: CUdeviceptr,
    dvsc: CUdeviceptr,
    /// [P13/C3/R8] KV 양자화 모드 — set_attn에서 env로 확정(KvMode::from_env).
    kvq: KvMode,
    dpp: CUdeviceptr,
    dqg_a: CUdeviceptr,
    dkin_a: CUdeviceptr,
    dvin_a: CUdeviceptr,
    dqh_a: CUdeviceptr,
    /// P8-attn-3: KV 분할 attention 부분합 스크래치
    /// ([T][q_heads][ATTN_SPLITS][258] f32 — m·l·acc256).
    dattn_part: CUdeviceptr,
    doutv_a: CUdeviceptr,
    attn_t_cap: usize,
    // ── 디바이스 체인(S10 — 연산별 왕복 제거) ──
    dres: CUdeviceptr,
    dab_dev: CUdeviceptr,
    dchain: [CUdeviceptr; 5],
    stg_w0: usize,
    stg_w1: usize,
    stg_w2: usize,
    chain_bufs_ok: bool,
    /// t=1 GEMV 입력 x32(h2f 왕복 f32) 버퍼.
    dx32: Buf,
    /// [marlin-A 2026-10-10] dx32의 f16 미러(split 경로 A — cast/norm이 함께
    /// 기록). mma GEMM이 이 버퍼를 복사-스테이징 → A L2 대역 절반.
    dx16: Buf,
    /// t≥2 GEMM 출력 스크래치([t][max_n] f32).
    dyt: Buf,
    // ── GPU head(output.weight bf16) ──
    head_w: CUdeviceptr,
    head_n: usize,
    head_k: usize,
    head_out: CUdeviceptr,
    /// [P3] argmax 결과(u32 1개) — 그래프 4B readback 대상.
    argmax_out: CUdeviceptr,
    // ── CUDA Graph(체인 캡처 — 토큰당 1 launch) ──
    /// [A8 2026-10-09] 슬롯·모드별 캡처 exec 캐시 — 종전 단일 그래프를
    /// 슬롯/모드 전환마다 destroy+재캡처(다중 슬롯 틱마다 수 ms)했고, 이
    /// 캐시가 그 비용을 1회 캡처로 상각한다. 키 = (slot, head, argmax).
    /// 버퍼 재할당(graph_invalidate) 시 전량 폐기(옛 포인터 replay 차단).
    graph_cache: Vec<GraphEntry>,
    /// 캡처 실패 후 직접 경로 고정(매 토큰 재시도 방지).
    graph_failed: bool,
    /// 캡처 중 임베딩 복사 소스를 pinned로 고정(캡처는 pageable async 불가).
    capture_pinned_src: bool,
    pin_embed: *mut std::ffi::c_void,
    pin_pos: *mut std::ffi::c_void,
    pin_out: *mut std::ffi::c_void,
    pin_out_len: usize,
    /// [A9 2026-10-10] 배치 디코드 부속 — 로짓[t×head_n]·argmax[t]·핀드
    /// (토큰·임베딩 행·슬롯 pos) + 슬롯집합별 캡처 그래프 캐시.
    dbatch_lg: CUdeviceptr,
    dbatch_am: CUdeviceptr,
    pin_batch_tok: *mut std::ffi::c_void,
    pin_batch_in: *mut std::ffi::c_void,
    pin_batch_pos: *mut std::ffi::c_void,
    batch_graphs: Vec<(Vec<usize>, ffi::CUgraphExec, ffi::CUgraph)>,
    /// 캡처 실패 후 직접 발사 고정(재시도 방지 — t=1 graph_failed와 동형).
    batch_capture_failed: bool,
    /// 층별 잔차 합 덤프(CPU LLM170_DUMP=debug_layers와 대조용).
    pub debug_layers: bool,
}

// SAFETY: CUDA 핸들(*mut c_void)은 Send가 아니지만, 이 디코더는 서버
// 슬롯 스레드 1개가 소유·사용한다(공유 없음). 컨텍스트 current 전환은
// 진입마다 cc.guard()가 수행한다 — 구 Exl3CudaDecoder의 동일 계약
// (586758df^ exl3_cuda.rs L234 unsafe impl Send) 미러.
unsafe impl Send for W4a16Dec {}

mod attn;
mod buffers;
mod chains;
mod gdn;
mod graphs;
mod kernels;
mod moe;
mod probe;

#[cfg(test)]
mod tests;

impl W4a16Dec {
    /// 디코더 생성 — 5개 fatbin(체인 커널) 로드. 가중치는 upload_*로 공급.
    pub fn new(n_slots: usize, hidden: usize, n_layers: usize) -> Result<Self, String> {
        let mut cc = CudaCtx::new()?;
        // [R9] 자산 매니페스트 단일 출처(rawcuda/assets.rs) — 부팅 로드분.
        for a in crate::rawcuda::assets::ASSETS.iter().filter(|a| a.boot) {
            cc.load_fatbin(a.name, &crate::rawcuda::assets::asset_bytes(a)?, a.syms)?;
        }
        Ok(W4a16Dec {
            cc,
            hidden,
            n_layers,
            n_slots: n_slots.max(1),
            slot_pos: vec![0; n_slots.max(1)],
            lins: HashMap::new(),
            plains: HashMap::new(),
            plain_weights: false,
            weights_bytes: 0,
            experts_bytes: 0,
            n_experts: 0,
            top_k: 0,
            moe_ffn: 0,
            shared_ffn: 0,
            moe_group: 32,
            moe_scale_bf16: true,
            moe_tab: Vec::new(),
            moe_resident: false,
            moe_dev_tab: 0,
            moe_idx: 0,
            moe_wt: 0,
            moe_multi_tab: 0,
            moe_multi_rows: Vec::new(),
            moe_gslot: 0,
            moe_gcnt: 0,
            moe_goff: 0,
            moe_gmax: 0,
            exp: ExpBufs::default(),
            stg: StgBufs::default(),
            host_stage: Vec::new(),
            host_stage_reg: 0,
            drt: Buf::default(),
            dmo: Buf::default(),
            moe_bufs_ok: false,
            xh: Buf::default(),
            dy: Buf::default(),
            dnw: 0,
            norm_w_rows: 0,
            norm: NormBufs::default(),
            ew: EwBufs::default(),
            gdn: None,
            dcw: 0,
            dab_c: 0,
            dalog: 0,
            ddtb: 0,
            dnwg: 0,
            dring: 0,
            dgst: 0,
            dqkv: 0,
            dzv: 0,
            dgxn: 0,
            dgq: 0,
            dgk: 0,
            dgv: 0,
            dq2: 0,
            dk2: 0,
            dv2: 0,
            dbg: 0,
            dgo: 0,
            dgpart: 0,
            dgdc: 0,
            dakq: 0,
            dgate: 0,
            gdn_t_cap: 0,
            attn: None,
            dqnw_a: 0,
            dknw_a: 0,
            dkc: 0,
            dvc: 0,
            dksc: 0,
            dvsc: 0,
            kvq: KvMode::F32,
            dpp: 0,
            dqg_a: 0,
            dkin_a: 0,
            dvin_a: 0,
            dqh_a: 0,
            doutv_a: 0,
            dattn_part: 0,
            attn_t_cap: 0,
            dres: 0,
            dab_dev: 0,
            dchain: [0; 5],
            stg_w0: 0,
            stg_w1: 0,
            stg_w2: 0,
            chain_bufs_ok: false,
            dx32: Buf::default(),
            dx16: Buf::default(),
            dyt: Buf::default(),
            head_w: 0,
            head_n: 0,
            head_k: 0,
            head_out: 0,
            argmax_out: 0,
            graph_cache: Vec::new(),
            graph_failed: false,
            capture_pinned_src: false,
            pin_embed: std::ptr::null_mut(),
            pin_pos: std::ptr::null_mut(),
            pin_out: std::ptr::null_mut(),
            pin_out_len: 0,
            dbatch_lg: 0,
            dbatch_am: 0,
            pin_batch_tok: std::ptr::null_mut(),
            pin_batch_in: std::ptr::null_mut(),
            pin_batch_pos: std::ptr::null_mut(),
            batch_graphs: Vec::new(),
            batch_capture_failed: false,
            debug_layers: false,
        })
    }

    pub fn device_name(&self) -> &str {
        &self.cc.device_name
    }

    pub(crate) fn h2d_chunked(cc: &CudaCtx, dst: CUdeviceptr, src: &[u8]) -> Result<(), String> {
        const CH: usize = 8 << 20;
        // 페이지러블 비동기(드라이버 스테이징 — 호출 내 스테이징 완료 계약) +
        // 8청크(64MiB)마다 sync로 스테이징 큐 상한.
        //
        // [실측 2026-10-09, 4090·Gen4 x16] 수동 핀드 링 경로(CPU→핀드 복사 +
        // 핑퐁 DMA)는 ~5GB/s — CPU의 핀드(비캐시) 기록이 병목이었다. 드라이버
        // 스테이징은 9.3GB/s(h2d-bench), 핀드 DMA는 13.8GB/s지만 그 앞단 복사가
        // 더 느리다. 합계 20GB급 업로드가 4.4s → ~2.2s.
        const INFLIGHT: usize = 64 << 20; // 스테이징 큐 상한(바이트 기준)
        let mut pending = 0usize;
        for (i, chunk) in src.chunks(CH).enumerate() {
            cc.h2d_async(dst + (i * CH) as u64, chunk)?;
            pending += chunk.len();
            if pending >= INFLIGHT {
                cc.sync()?;
                pending = 0;
            }
            // 업로드는 수 초~수십 초 — 와치독이 로드를 스텔로 오판하지 않게 심박.
            llm170_diag::watchdog::bump();
        }
        cc.sync()
    }

    fn zero_dev(cc: &CudaCtx, ptr: CUdeviceptr, len: usize) -> Result<(), String> {
        // [2026-10-09 H] 동기 h2d 4MiB 청크 루프 → cuMemsetD8Async 1콜.
        // (종전에는 청크마다 스테이징·호출 — 게다가 캡처 경로에서 금지되는
        // 동기 복사였다.) 스트림 순서라 후속 커널·판독과 정합.
        cc.memset0_async(ptr, len)
    }

    /// 선형 상주 업로드 — packed u32 [n][k/8] · scale f16 [n][k/128].
    pub fn upload_lin(
        &mut self,
        name: &str,
        q: &[u8],
        s: &[u8],
        n: usize,
        k: usize,
    ) -> Result<(), String> {
        if q.len() != n * (k / 2) || s.len() != n * (k / 64) {
            return Err(format!(
                "upload_lin {name}: q={} s={} != n{n} k{k} 계약",
                q.len(),
                s.len()
            ));
        }
        // G2: alloc 전량 성공 → h2d 성공 시에만 상주 등록. 어느 단계든 실패하면
        // 성공분을 회수한다(부분 업로드 유실 금지 — 재시도가 처음부터).
        let dq = self.cc.alloc(q.len())?;
        let ds = match self.cc.alloc(s.len()) {
            Ok(p) => p,
            Err(e) => {
                let _ = self.cc.free(dq);
                return Err(e);
            }
        };
        let r = (|| {
            Self::h2d_chunked(&self.cc, dq, q)?;
            Self::h2d_chunked(&self.cc, ds, s)
        })();
        if let Err(e) = r {
            let _ = self.cc.free(dq);
            let _ = self.cc.free(ds);
            return Err(e);
        }
        self.weights_bytes += (q.len() + s.len()) as u64;
        if let Some((oq, os, _, _)) = self.lins.insert(name.to_string(), (dq, ds, n, k)) {
            let _ = self.cc.free(oq);
            let _ = self.cc.free(os);
        }
        Ok(())
    }

    /// 상주 선형 GEMV — x f32 → (RN-even) f16 → 커널 → y f32 [n].
    pub fn gemv_host(&mut self, name: &str, x: &[f32]) -> Result<Vec<f32>, String> {
        let _g = self.cc.guard()?;
        let Some(&(dq, ds, n, k)) = self.lins.get(name) else {
            return Err(format!("gemv: 상주 선형 없음: {name}"));
        };
        if x.len() != k {
            return Err(format!("gemv {name}: x={} != k={k}", x.len()));
        }
        if k > self.xh.cap {
            self.graph_invalidate(); // [P10]
            self.cc.sync()?;
            if self.xh.ptr != 0 {
                self.cc.free(self.xh.ptr)?;
            }
            self.xh.ptr = 0; // G1: alloc 실패 시 재시도 이중해제 방지.
            self.xh.cap = 0;
            self.xh.ptr = self.cc.alloc(k * 2)?;
            self.xh.cap = k;
        }
        if n > self.dy.cap {
            self.graph_invalidate(); // [P10]
            self.cc.sync()?; // [P10] 비행 커널의 해제 버퍼 사용 차단.
            if self.dy.ptr != 0 {
                self.cc.free(self.dy.ptr)?;
            }
            self.dy.ptr = 0; // G1
            self.dy.cap = 0;
            self.dy.ptr = self.cc.alloc(n * 4)?;
            self.dy.cap = n;
        }
        // P3-b: 신 GEMM은 f32 x 계약 — 호스트에서 h2f(f2h(v)) 동형 변환.
        if k > self.dx32.cap {
            self.graph_invalidate(); // [P10]
            self.cc.sync()?;
            if self.dx32.ptr != 0 {
                self.cc.free(self.dx32.ptr)?;
            }
            self.dx32.ptr = 0; // G1
            self.dx32.cap = 0;
            self.dx32.ptr = self.cc.alloc(k * 4)?;
            self.dx32.cap = k;
        }
        let xf: Vec<f32> = x
            .iter()
            .map(|&v| crate::rawcuda::gptq4::h2f(f32_to_f16(v)))
            .collect();
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
        let xb = unsafe { std::slice::from_raw_parts(xf.as_ptr() as *const u8, xf.len() * 4) };
        self.cc.h2d(self.dx32.ptr, xb)?;
        let f = self.cc.function("w4a16_gemm_g128")?;
        let (mut p_q, mut p_s, mut p_x, mut p_y) = (dq, ds, self.dx32.ptr, self.dy.ptr);
        let (mut p_n, mut p_k, mut p_t) = (n as i32, k as i32, 1i32);
        // 스테이징 폴백(t=1)도 신 GEMM 커널 계약(8행/블록·512스레드)으로 —
        // 구 계약(grid n/8·block 64)은 재작성 후 1/8행만 계산하는 결함이었다.
        self.cc.launch(
            f,
            n.div_ceil(8) as u32,
            1,
            512,
            &mut crate::rawcuda::args::l7(
                &mut p_q, &mut p_s, &mut p_x, &mut p_y, &mut p_n, &mut p_k, &mut p_t,
            ),
        )?;
        let mut ob = vec![0u8; n * 4];
        self.cc.d2h_async(ob.as_mut_ptr(), self.dy.ptr, n * 4)?; // 커스텀 스트림 대비
        self.cc.sync()?;
        Ok(ob
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    // ── norm ──

    /// 노름 가중 상주 등록 — nw [rows][hidden] f32(행 포인터 계약 w·hidden).
    pub fn set_norm_weights(&mut self, nw: &[f32], rows: usize) -> Result<(), String> {
        if self.hidden == 0 || !self.hidden.is_multiple_of(1024) || self.hidden > 8192 {
            return Err(format!(
                "norm: hidden={} — 1024 배수·8192 이하 계약",
                self.hidden
            ));
        }
        if rows == 0 || nw.len() != rows * self.hidden {
            return Err(format!(
                "norm: nw {} != rows {rows} × hidden {}",
                nw.len(),
                self.hidden
            ));
        }
        if self.dnw != 0 {
            self.cc.free(self.dnw)?;
        }
        self.dnw = 0; // G1: 실패 시 재시도가 0을 해제하지 않게.
        self.norm_w_rows = 0;
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
        let b = unsafe { std::slice::from_raw_parts(nw.as_ptr() as *const u8, nw.len() * 4) };
        let d = self.cc.alloc(nw.len() * 4)?;
        if let Err(e) = Self::h2d_chunked(&self.cc, d, b) {
            let _ = self.cc.free(d); // G1: 부분 업로드 실패 시 유실 방지.
            return Err(e);
        }
        self.dnw = d;
        self.norm_w_rows = rows;
        Ok(())
    }

    pub fn ew_host(&mut self, g: &[f32], u: &[f32]) -> Result<Vec<f32>, String> {
        if g.is_empty() || g.len() != u.len() {
            return Err(format!("ew: g={} u={} 계약 위반", g.len(), u.len()));
        }
        self.ensure_ew_bufs(g.len())?;
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
        let gb = unsafe { std::slice::from_raw_parts(g.as_ptr() as *const u8, g.len() * 4) };
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
        let ub = unsafe { std::slice::from_raw_parts(u.as_ptr() as *const u8, u.len() * 4) };
        self.cc.h2d(self.ew.dewg, gb)?;
        self.cc.h2d(self.ew.dewu, ub)?;
        let f = self.cc.function("ew")?;
        let mut nn = g.len() as i32;
        let (mut a0, mut a1, mut a2) = (self.ew.dewg, self.ew.dewu, self.ew.dew);
        self.cc.launch(
            f,
            g.len().div_ceil(128) as u32,
            1,
            128,
            &mut crate::rawcuda::args::l4(&mut a0, &mut a1, &mut a2, &mut nn),
        )?;
        let mut yb = vec![0u8; g.len() * 4];
        self.cc.d2h(&mut yb, self.ew.dew)?;
        self.cc.sync()?;
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
        Ok(unsafe { std::slice::from_raw_parts(yb.as_ptr() as *const f32, g.len()) }.to_vec())
    }

    // ── GDN ──

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
        if cw.len() != n * cch * 4
            || ab.len() != n * 2 * hv * hd
            || alog.len() != n * hv
            || dtb.len() != n * hv
            || nw.len() != n * 128
        {
            return Err("GDN: 상수 형상 계약 위반(cw/ab/alog/dtb/nw)".into());
        }
        for q in [
            self.dcw, self.dab_c, self.dalog, self.ddtb, self.dnwg, self.dring, self.dgst,
        ] {
            if q != 0 {
                self.cc.free(q)?;
            }
        }
        (self.dcw, self.dab_c, self.dalog, self.ddtb, self.dnwg) = (0, 0, 0, 0, 0);
        (self.dring, self.dgst) = (0, 0);
        let b =
            // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
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
        let slots = self.n_slots;
        let dring = self.cc.alloc(slots * n * 3 * cch * 4)?;
        Self::zero_dev(&self.cc, dring, slots * n * 3 * cch * 4)?;
        let dgst = self.cc.alloc(slots * n * hv * 128 * 128 * 4)?;
        Self::zero_dev(&self.cc, dgst, slots * n * hv * 128 * 128 * 4)?;
        (self.dcw, self.dab_c, self.dalog, self.ddtb, self.dnwg) = (dcw, dab, dal, ddt, dnw);
        self.dring = dring;
        self.dgst = dgst;
        self.gdn = Some(dims);
        Ok(())
    }

    pub fn set_attn(&mut self, dims: AttnDims, qnw: &[f32], knw: &[f32]) -> Result<(), String> {
        let n = dims.n_attn;
        if qnw.len() != n * dims.d || knw.len() != n * dims.d {
            return Err(format!("attn: qnw/knw != {n}x{}", dims.d));
        }
        let slots = self.n_slots;
        let kv_elems = slots * n * dims.cap * dims.kv_dim();
        // [P13] KV 양자화 — 옵트인(LLM170_KVQ). int8 4× 절감, 스케일은
        // 행×헤드 f32 1개(무시 가능). 미설정 = 종전 f32 경로 그대로.
        // [C3/R8] LLM170_KVQ: "4"=int4, 그 외 비영=int8(후방 호환).
        let kvq = KvMode::from_env();
        let kv_scales = slots * n * dims.cap * dims.kv_heads;
        for q in [
            self.dqnw_a,
            self.dknw_a,
            self.dkc,
            self.dvc,
            self.dksc,
            self.dvsc,
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
        (self.dksc, self.dvsc) = (0, 0);
        self.kvq = kvq;
        (
            self.dqg_a,
            self.dkin_a,
            self.dvin_a,
            self.dqh_a,
            self.doutv_a,
        ) = (0, 0, 0, 0, 0);
        self.attn_t_cap = 0;
        let b =
            // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        let dq = self.cc.alloc(qnw.len() * 4)?;
        Self::h2d_chunked(&self.cc, dq, b(qnw))?;
        let dk = self.cc.alloc(knw.len() * 4)?;
        Self::h2d_chunked(&self.cc, dk, b(knw))?;
        let (dkc, dvc) = if kvq.is_on() {
            // [P13/C3/R8] int8(1B)·int4(0.5B) KV + 스케일 2벌. 기록은
            // attn_prep_q(_4), 판독은 attn_fwd3s_part_q(_4)(병합은 f32 그대로).
            let kbytes = kvq.kv_bytes(kv_elems);
            let kc = self.cc.alloc(kbytes)?;
            Self::zero_dev(&self.cc, kc, kbytes)?;
            let vc = self.cc.alloc(kbytes)?;
            Self::zero_dev(&self.cc, vc, kbytes)?;
            let ks = self.cc.alloc(kv_scales * 4)?;
            Self::zero_dev(&self.cc, ks, kv_scales * 4)?;
            let vs = self.cc.alloc(kv_scales * 4)?;
            Self::zero_dev(&self.cc, vs, kv_scales * 4)?;
            self.dksc = ks;
            self.dvsc = vs;
            (kc, vc)
        } else {
            let kc = self.cc.alloc(kv_elems * 4)?;
            Self::zero_dev(&self.cc, kc, kv_elems * 4)?;
            let vc = self.cc.alloc(kv_elems * 4)?;
            Self::zero_dev(&self.cc, vc, kv_elems * 4)?;
            (kc, vc)
        };
        let dpp = self.cc.alloc(slots * 4)?;
        Self::zero_dev(&self.cc, dpp, slots * 4)?;
        self.dqnw_a = dq;
        self.dknw_a = dk;
        self.dkc = dkc;
        self.dvc = dvc;
        self.dpp = dpp;
        self.attn = Some(dims);
        Ok(())
    }

    fn lin_spec(&self, name: &str) -> Result<(CUdeviceptr, CUdeviceptr, usize, usize), String> {
        self.lins
            .get(name)
            .copied()
            .ok_or_else(|| format!("상주 선형 없음: {name}"))
    }

    /// 플레인 bf16 [n][k] 상주 업로드 — [k][n] 전치(head_bf16 GEMV 계약).
    pub fn upload_plain(
        &mut self,
        name: &str,
        data: &[u8],
        n: usize,
        k: usize,
    ) -> Result<(), String> {
        let _g = self.cc.guard()?;
        let need = n * k * 2;
        if n == 0 || k == 0 || data.len() < need {
            return Err(format!(
                "upload_plain({name}): 형상 계약 위반 n={n} k={k} bytes={} < {need}",
                data.len()
            ));
        }
        // [n][k] 원본 그대로 — w4a16_gemv_bf16(행=블록) 계약(전치 없음).
        let dw = self.cc.alloc(need)?;
        if let Err(e) = Self::h2d_chunked(&self.cc, dw, &data[..need]) {
            let _ = self.cc.free(dw);
            return Err(e);
        }
        self.weights_bytes += need as u64;
        if let Some((old, _, _)) = self.plains.insert(name.to_string(), (dw, n, k)) {
            let _ = self.cc.free(old);
        }
        Ok(())
    }

    /// 플레인 모드 전환(전문가 외 전부 bf16인 MoE 모델).
    pub fn set_plain_mode(&mut self, on: bool) {
        self.plain_weights = on;
    }

    /// 메모리 분류(모니터링) — 모델이 시스템을 어떻게 쓰는지.
    /// 반환: (VRAM 가중치, VRAM KV, CPU 오프로드 가중치, CPU PLE).
    /// - 가중치: 업로드 누적(weights_bytes — 로드 후 불변). 스트리밍 모드에서
    ///   전문가는 VRAM에 없고 호스트(mmAP 페이지 캐시)에서 토큰별로 올린다.
    /// - KV: 어텐션 캐시 2벌(K+V) — 기본 f32, [P13] KVQ 시 int8+스케일.
    /// - PLE: 미구현(W4-2) — 항상 0.
    pub fn mem_stats(&self) -> (u64, u64, u64, u64) {
        let kv = self
            .attn
            .map(|d| {
                let rows = (self.n_slots as u64) * (d.n_attn as u64) * (d.cap as u64);
                if self.kvq.is_on() {
                    rows * (self.kvq.kv_bytes(d.kv_dim()) as u64)
                        + rows * (d.kv_heads as u64) * 4 * 2
                } else {
                    2 * rows * (d.kv_dim() as u64) * 4
                }
            })
            .unwrap_or(0);
        let experts = self.experts_bytes;
        let (w_gpu, w_cpu) = if self.n_experts > 0 && !self.moe_resident {
            (self.weights_bytes, experts)
        } else {
            (self.weights_bytes, 0)
        };
        (w_gpu, kv, w_cpu, 0)
    }

    /// 토큰당 활성 가중치 바이트(모니터링 — 실효 대역폭 계산용).
    /// 상주 모드 = 상주 가중치 − 미선택 전문가(전문가 크기 균일 — 평균이 정확).
    /// 스트리밍 모드 = 상주 가중치(전문가는 별도 CPU 오프로드로 집계).
    pub fn active_weight_bytes(&self) -> u64 {
        let experts = self.experts_bytes;
        if self.n_experts > 0 && self.moe_resident {
            let unsel = experts / self.n_experts as u64 * (self.n_experts - self.top_k) as u64;
            self.weights_bytes.saturating_sub(unsel)
        } else {
            self.weights_bytes
        }
    }

    /// 복사 계측(모니터링) — [h2d, d2h, d2d] × (바이트, ns, 호출).
    pub fn copy_stats(&self) -> [(u64, u64, u64); 3] {
        self.cc.copy_stats()
    }

    /// MoE 배치 모드 문자열(모니터링) — "none" | "resident" | "streaming".
    pub fn moe_mode(&self) -> &'static str {
        if self.n_experts == 0 {
            "none"
        } else if self.moe_resident {
            "resident"
        } else {
            "streaming"
        }
    }

    /// MoE 모델 여부(프리필 t=1 강제 등).
    pub fn is_moe(&self) -> bool {
        self.n_experts > 0
    }

    /// 전문가 전량 상주 여부 — MoE 배치 프리필(t>1)의 전제.
    pub fn moe_experts_resident(&self) -> bool {
        self.moe_resident
    }

    /// MoE 구성 등록 — n_experts>0이면 체인은 플레인+MoE FFN 경로.
    pub fn set_moe(
        &mut self,
        n_experts: usize,
        top_k: usize,
        moe_ffn: usize,
        shared_ffn: usize,
        group: usize,
        scale_bf16: bool,
    ) {
        self.n_experts = n_experts;
        self.top_k = top_k;
        self.moe_ffn = moe_ffn;
        self.shared_ffn = shared_ffn;
        self.moe_group = group;
        self.moe_scale_bf16 = scale_bf16;
    }

    /// 전문가 슬라이스 테이블 등록 — (packed ptr/len, scale ptr/len) × (il,e,proj).
    /// 포인터는 서버 스토어 mmap 슬라이스 — 서버가 모델을 함께 보유하는 수명 계약.
    pub fn set_expert_table(&mut self, mut tab: Vec<(u64, u64, u64, u64)>) {
        // [P7-fix] 기존 스테이징이 등록돼 있으면 해제(버퍼 해제 전 필수).
        if self.host_stage_reg > 0 {
            // SAFETY: 등록 성공 상태(host_stage_reg>0)의 host_stage 선두 주소다.
            let _ = unsafe {
                self.cc
                    .host_unregister(self.host_stage.as_ptr() as *mut std::ffi::c_void)
            };
            self.host_stage_reg = 0;
        }
        // [C2 2026-10-09] 호스트 RAM 스테이징(사용자 결정) — 스트리밍 셋을
        // 로드 시 1회 호스트 RAM으로 복사한다(SSD mmap·페이지캐시 경로 의존
        // 제거). 탈출구 LLM170_MOE_HOST_STAGE=0, 할당 실패 시 mmap 유지.
        if llm170_diag::flag::ne0("LLM170_MOE_HOST_STAGE") {
            let total: u64 = tab.iter().map(|e| e.1 + e.3).sum();
            let mut buf: Vec<u8> = Vec::new();
            if total > 0 && usize::try_from(total).is_ok_and(|n| buf.try_reserve_exact(n).is_ok()) {
                let mut offs: Vec<(usize, usize)> = Vec::with_capacity(tab.len());
                for &(qp, ql, sp, sl) in &tab {
                    let qo = buf.len();
                    // SAFETY: tab 항목은 set_expert_table 수명 계약(mmap 슬라이스)
                    // — 호출 내 유효, len은 계약 검증 완료분.
                    buf.extend_from_slice(unsafe {
                        std::slice::from_raw_parts(qp as *const u8, ql as usize)
                    });
                    let so = buf.len();
                    // SAFETY: 상동(scale 슬라이스).
                    buf.extend_from_slice(unsafe {
                        std::slice::from_raw_parts(sp as *const u8, sl as usize)
                    });
                    offs.push((qo, so));
                }
                let base = buf.as_ptr() as u64;
                for (e, &(qo, so)) in tab.iter_mut().zip(offs.iter()) {
                    e.0 = base + qo as u64;
                    e.2 = base + so as u64;
                }
                eprintln!(
                    "[moe] 호스트 RAM 스테이징 {:.2}GiB — 업로드 mmap 의존 제거(C2)",
                    total as f64 / (1u64 << 30) as f64
                );
                self.host_stage = buf;
                // [P7-fix 2026-10-10] 핀드 DMA — 등록 가능한 선두 구간만
                // cuMemHostRegister(RLIMIT_MEMLOCK 한도). 등록 구간 소스 복사는
                // DMA 직행(실측 13.8GB/s), 나머지는 드라이버 스테이징 유지.
                // 예산 LLM170_MOE_PIN_MB(0=끔, 기본 4096).
                let budget_mb = llm170_diag::flag::val("LLM170_MOE_PIN_MB")
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(4096);
                let budget = budget_mb << 20;
                if budget > 0 && !self.host_stage.is_empty() {
                    // memlock 한도가 작은 환경 대비 1/4씩 축소 재시도(하한 4MiB).
                    let mut bytes = self.host_stage.len().min(budget);
                    loop {
                        // SAFETY: host_stage는 구축 후 불변 Vec(boxed slice)라
                        // 해제 전까지 포인터 수명이 유지된다(해제는 상단 unregister).
                        match unsafe {
                            self.cc.host_register(
                                self.host_stage.as_ptr() as *mut std::ffi::c_void,
                                bytes,
                            )
                        } {
                            Ok(()) => {
                                self.host_stage_reg = bytes;
                                eprintln!(
                                    "# moe: 스테이징 핀드 등록 {:.2}/{}GiB — 구간 DMA 직행(P7-fix)",
                                    bytes as f64 / (1u64 << 30) as f64,
                                    self.host_stage.len() >> 30
                                );
                                break;
                            }
                            Err(e) => {
                                if bytes <= (4 << 20) {
                                    eprintln!(
                                        "# moe: 핀드 등록 실패({e}) — 드라이버 스테이징 유지(P7-fix)"
                                    );
                                    break;
                                }
                                bytes = (bytes / 4).max(4 << 20);
                            }
                        }
                    }
                }
            } else {
                eprintln!("# moe: 호스트 스테이징 {total}B 할당 실패 — mmap 스트리밍 유지");
            }
        }
        self.experts_bytes = tab.iter().map(|e| e.1 + e.3).sum();
        self.moe_tab = tab;
    }

    fn plain_spec(&self, name: &str) -> Result<(CUdeviceptr, usize, usize), String> {
        self.plains
            .get(name)
            .copied()
            .ok_or_else(|| format!("상주 플레인 없음: {name}"))
    }

    /// 전문가 상주 업로드 — 호스트 테이블(mmAP 슬라이스)을 VRAM 아레나
    /// (proj별 packed/scale 6개)로 올리고 테이블을 VRAM 포인터로 교체한다.
    /// 상주가 가능한 기기(170HX 64GB)에서 토큰당 PCIe 스트리밍을 제거한다.
    pub fn upload_experts_resident(
        &mut self,
        host_tab: &[(u64, u64, u64, u64)],
    ) -> Result<(), String> {
        let _g = self.cc.guard()?;
        if self.n_experts == 0 || host_tab.len() != self.n_layers * self.n_experts * 3 {
            return Err("upload_experts_resident: 테이블 형상 위반".into());
        }
        let mut pk = [0usize; 3];
        let mut sk = [0usize; 3];
        for (i, e) in host_tab.iter().enumerate() {
            pk[i % 3] += e.1 as usize;
            sk[i % 3] += e.3 as usize;
        }
        let mut dpk = [0 as CUdeviceptr; 3];
        let mut dsk = [0 as CUdeviceptr; 3];
        let mut opk = [0usize; 3];
        let mut osk = [0usize; 3];
        let mut allocd = Vec::new();
        let setup = (|| -> Result<(), String> {
            for p in 0..3 {
                dpk[p] = self.cc.alloc(pk[p])?;
                allocd.push(dpk[p]);
                dsk[p] = self.cc.alloc(sk[p])?;
                allocd.push(dsk[p]);
            }
            Ok(())
        })();
        if let Err(e) = setup {
            for p in allocd {
                let _ = self.cc.free(p);
            }
            return Err(e);
        }
        let mut dev_tab = Vec::with_capacity(host_tab.len());
        // 1) 디바이스 테이블(아레나 오프셋) — 복사 없이 주소만 계산.
        for (i, e) in host_tab.iter().enumerate() {
            let p = i % 3;
            dev_tab.push((dpk[p] + opk[p] as u64, e.1, dsk[p] + osk[p] as u64, e.3));
            opk[p] += e.1 as usize;
            osk[p] += e.3 as usize;
        }
        let (mut opk2, mut osk2) = ([0usize; 3], [0usize; 3]);
        // 2) 전송 — h2d_chunked(페이지러블 비동기 + 드라이버 스테이징).
        // SAFETY: 항목은 서버 스토어 mmap 슬라이스(set_expert_table 수명 계약).
        let r = (|| -> Result<(), String> {
            for (i, e) in host_tab.iter().enumerate() {
                let p = i % 3;
                // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
                let qb = unsafe { std::slice::from_raw_parts(e.0 as *const u8, e.1 as usize) };
                // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
                let sb = unsafe { std::slice::from_raw_parts(e.2 as *const u8, e.3 as usize) };
                Self::h2d_chunked(&self.cc, dpk[p] + opk2[p] as u64, qb)?;
                Self::h2d_chunked(&self.cc, dsk[p] + osk2[p] as u64, sb)?;
                opk2[p] += e.1 as usize;
                osk2[p] += e.3 as usize;
            }
            Ok(())
        })();
        if let Err(err) = r {
            for p in allocd {
                let _ = self.cc.free(p);
            }
            return Err(err);
        }
        // 디바이스 포인터 테이블 — (q,s) 쌍 평탄 배열(커널 간접 참조).
        let mut flat: Vec<u64> = Vec::with_capacity(dev_tab.len() * 2);
        for e in &dev_tab {
            flat.push(e.0);
            flat.push(e.2);
        }
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
        let fb = unsafe { std::slice::from_raw_parts(flat.as_ptr() as *const u8, flat.len() * 8) };
        if self.moe_dev_tab != 0 {
            let _ = self.cc.free(self.moe_dev_tab);
            self.moe_dev_tab = 0;
        }
        let dtab = self.cc.alloc(flat.len() * 8)?;
        if let Err(e) = Self::h2d_chunked(&self.cc, dtab, fb) {
            let _ = self.cc.free(dtab);
            return Err(e);
        }
        self.moe_dev_tab = dtab;
        self.moe_tab = dev_tab;
        self.experts_bytes = host_tab.iter().map(|e| e.1 + e.3).sum();
        self.weights_bytes += self.experts_bytes;
        self.moe_resident = true;
        Ok(())
    }

    /// 마이크로벤치 표면(진단 전용) — 스크래치 alloc/h2d/free + GEMM 발사/sync.
    /// 핀드 스크래치 할당(벤치·진단) — h2d_bench 계약.
    pub fn alloc_pinned_scratch(&self, bytes: usize) -> Result<*mut std::ffi::c_void, String> {
        self.cc.pinned_alloc(bytes)
    }

    pub fn free_pinned_scratch(&self, p: *mut std::ffi::c_void) -> Result<(), String> {
        self.cc.pinned_free(p)
    }

    pub fn alloc_scratch(&self, bytes: usize) -> Result<CUdeviceptr, String> {
        self.cc.alloc(bytes)
    }

    pub fn free_scratch(&self, p: CUdeviceptr) -> Result<(), String> {
        self.cc.free(p)
    }

    pub fn h2d_scratch(&self, dst: CUdeviceptr, src: &[u8]) -> Result<(), String> {
        self.cc.h2d(dst, src)
    }

    pub fn sync_bench(&self) -> Result<(), String> {
        self.cc.sync()
    }

    /// 지정 선형의 t≥2 GEMM 1회 발사(벤치 전용 — y는 스크래치).
    /// 진단 타이머 리포트(P8) — LLM170_TIME=1일 때 범주별 ms·비중.
    pub fn prof_report(&mut self, label: &str) -> Result<String, String> {
        let _g = self.cc.guard()?;
        self.cc.prof_report(label)
    }

    /// bf16 head(output.weight) 상주 업로드 — head_bf16 GEMV.
    pub fn upload_head(&mut self, data: &[u8], n: usize, k: usize) -> Result<(), String> {
        let _g = self.cc.guard()?;
        let need = n * k * 2;
        if n == 0 || k == 0 || data.len() < need {
            return Err(format!(
                "upload_head: 형상 계약 위반 n={n} k={k} bytes={} < {need}",
                data.len()
            ));
        }
        for p in [self.head_w, self.head_out] {
            if p != 0 {
                self.cc.free(p)?;
            }
        }
        self.head_w = 0; // G1: 부분 상태(head_n=0)로 소비되지 않게 마지막에 대입.
        self.head_out = 0;
        self.head_n = 0;
        self.head_k = 0;
        // [A-4 2026-10-10] **원본 [n][k] 레이아웃 유지** — 종전 head_transpose로
        // [k][n]을 만들어 head_bf16(행당 k직렬·열 판독)으로 읽었으나 실측
        // DRAM 2.55GB(가중치 1.556GB의 1.64×)·디코드의 12~25%를 차지.
        // 행=블록 w4a16_gemv_bf16([n][k] 원본 — "토큰 수준 판정" 설계)을
        // 재사용하면 완전 순차 판독. 로드 전치(3.5ms)도 제거.
        let dw = self.cc.alloc(need)?;
        if let Err(e) = Self::h2d_chunked(&self.cc, dw, &data[..need]) {
            let _ = self.cc.free(dw);
            return Err(e);
        }
        let dout = match self.cc.alloc(n * 4).and_then(|p| {
            self.cc.alloc(4).map(|a| {
                self.argmax_out = a;
                p
            })
        }) {
            Ok(p) => p,
            Err(e) => {
                let _ = self.cc.free(dw);
                return Err(e);
            }
        };
        self.head_w = dw;
        self.head_out = dout;
        self.head_n = n;
        self.head_k = k;
        self.weights_bytes += (need + n * 4) as u64;
        Ok(())
    }
}
