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

fn asset_bytes(env: &str, rel: &[&str]) -> Result<Vec<u8>, String> {
    if let Some(p) = llm170_diag::flag::val(env) {
        return std::fs::read(p).map_err(|e| format!("{env}({p}) 읽기 실패: {e}"));
    }
    for r in rel {
        if let Ok(b) = std::fs::read(r) {
            return Ok(b);
        }
    }
    Err(format!("자산 부재 — {rel:?} 또는 {env}"))
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
    /// [P11] 전문가-우선 슬롯 정렬(프리필 그룹 GEMV) — gslot[n_exp×gmax]+cnt.
    moe_gslot: CUdeviceptr,
    moe_gcnt: CUdeviceptr,
    moe_goff: CUdeviceptr,
    moe_gmax: usize,
    /// 배치 전문가 출력([top_k][n_ff] · [top_k][hidden]) — act는 ew 전용
    /// 별도 버퍼(ew 커널 __restrict__ 계약 — 제자리 호출 금지).
    dexp_gate: CUdeviceptr,
    dexp_up: CUdeviceptr,
    dexp_act: CUdeviceptr,
    dexp_dn: CUdeviceptr,
    dexp_cap: usize,
    /// MoE 스테이징 — 전문가 packed/scale(1쌍 재사용, 스트림 순서 안전).
    dstg_q: CUdeviceptr,
    dstg_s: CUdeviceptr,
    dstg_cap: (usize, usize),
    /// [C2 2026-10-09] 호스트 RAM 스테이징 — 스트리밍 전문가의 mmap 슬라이스
    /// 1회 복사본(set_expert_table). 스트리밍 업로드가 SSD/페이지캐시 경로에
    /// 의존하지 않게 한다. Vec 버퍼 주소는 불변이므로 moe_tab이 이 안을
    /// 가리켜도 안전(용량 추가 변경 금지 — 구축 후 불변).
    host_stage: Vec<u8>,
    /// 라우터 로짓/ shared 게이트 스크래치(n_experts ≥ 1).
    drt: CUdeviceptr,
    drt_cap: usize,
    /// MoE 출력(hidden) — 잔차 ab로 소비된다.
    dmo: CUdeviceptr,
    dmo_cap: usize,
    moe_bufs_ok: bool,
    /// GEMV 스테이징 — x f16 [t][k], y f32 [t][n].
    dxh: CUdeviceptr,
    xh_cap: usize,
    dy: CUdeviceptr,
    y_cap: usize,
    // ── norm ──
    dnw: CUdeviceptr,
    norm_w_rows: usize,
    dx: CUdeviceptr,
    dab: CUdeviceptr,
    dxn: CUdeviceptr,
    norm_cap: usize,
    // ── ew ──
    dewg: CUdeviceptr,
    dewu: CUdeviceptr,
    dew: CUdeviceptr,
    ew_cap: usize,
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
    /// [P13] KV 양자화(int8) — set_attn에서 env(LLM170_KVQ)로 확정.
    kvq: bool,
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
    dx32: CUdeviceptr,
    dx32_cap: usize,
    /// t≥2 GEMM 출력 스크래치([t][max_n] f32).
    dyt: CUdeviceptr,
    dyt_cap: usize,
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

impl W4a16Dec {
    /// 디코더 생성 — 5개 fatbin(체인 커널) 로드. 가중치는 upload_*로 공급.
    pub fn new(n_slots: usize, hidden: usize, n_layers: usize) -> Result<Self, String> {
        let mut cc = CudaCtx::new()?;
        cc.load_fatbin(
            "gptq4",
            &asset_bytes(
                "LLM170_CUDA_GPTQ4_FATBIN_PATH",
                &[
                    "crates/backend-gpu/src/rawcuda/assets/gptq4.fatbin",
                    "src/rawcuda/assets/gptq4.fatbin",
                ],
            )?,
            &[
                "w4a16_gemm_g128",
                "w4a16_gemv_g128",
                "w4a16_gemv_g128_t",
                "w4a16_gemm_g32_bf16",
                "w4a16_gemv_g32_bf16",
                "w4a16_gemv_bf16_t",
                "w4a16_cast_x32",
                "w4a16_axpy",
                "w4a16_shared_add",
                "w4a16_gemv_experts_g32_bf16",
                "w4a16_gemm_g32_mma_grp",
                "w4a16_moe_align",
                "w4a16_moe_accum",
                "w4a16_moe_topk",
                "w4a16_moe_topk_t",
                "w4a16_gemv_bf16",
                "w4a16_gemm_bf16",
                "w4a16_gemm_bf16_t",
                "w4a16_gemm_bf16_mma",
                "w4a16_gemm_g128_mma",
            ],
        )?;
        cc.load_fatbin(
            "norm",
            &asset_bytes(
                "LLM170_CUDA_NORM_FATBIN_PATH",
                &[
                    "crates/backend-gpu/src/rawcuda/assets/norm.fatbin",
                    "src/rawcuda/assets/norm.fatbin",
                ],
            )?,
            &["norm_resid"],
        )?;
        cc.load_fatbin(
            "gdn",
            &asset_bytes(
                "LLM170_CUDA_GDN_FATBIN_PATH",
                &[
                    "crates/backend-gpu/src/rawcuda/assets/gdn.fatbin",
                    "src/rawcuda/assets/gdn.fatbin",
                ],
            )?,
            &[
                "gdn_conv",
                "gdn_l2perm",
                "gdn_scan",
                "gdn_scan_akq",
                "gdn1_part",
                "gdn1_comb",
                "gdn1_upd",
                "gdn_gate",
            ],
        )?;
        cc.load_fatbin(
            "attn",
            &asset_bytes(
                "LLM170_CUDA_ATTN_FATBIN_PATH",
                &[
                    "crates/backend-gpu/src/rawcuda/assets/attn.fatbin",
                    "src/rawcuda/assets/attn.fatbin",
                ],
            )?,
            &[
                "attn_prep",
                "attn_fwd3s",
                "attn_fwd3s_part",
                "attn_fwd3s_part_q",
                "attn_prep_q",
                "attn_fwd3s_merge",
                "attn_pos_bump",
            ],
        )?;
        cc.load_fatbin(
            "head",
            &asset_bytes(
                "LLM170_CUDA_HEAD_FATBIN_PATH",
                &[
                    "crates/backend-gpu/src/rawcuda/assets/head.fatbin",
                    "src/rawcuda/assets/head.fatbin",
                ],
            )?,
            &[
                "head_bf16",
                "head_bf16_t",
                "head_transpose",
                "w4a16_argmax_min",
                "w4a16_argmax_min_t",
            ],
        )?;
        cc.load_fatbin(
            "ew",
            &asset_bytes(
                "LLM170_CUDA_EW_FATBIN_PATH",
                &[
                    "crates/backend-gpu/src/rawcuda/assets/ew.fatbin",
                    "src/rawcuda/assets/ew.fatbin",
                ],
            )?,
            &["ew"],
        )?;
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
            moe_gslot: 0,
            moe_gcnt: 0,
            moe_goff: 0,
            moe_gmax: 0,
            dexp_gate: 0,
            dexp_up: 0,
            dexp_act: 0,
            dexp_dn: 0,
            dexp_cap: 0,
            dstg_q: 0,
            dstg_s: 0,
            dstg_cap: (0, 0),
            host_stage: Vec::new(),
            drt: 0,
            drt_cap: 0,
            dmo: 0,
            dmo_cap: 0,
            moe_bufs_ok: false,
            dxh: 0,
            xh_cap: 0,
            dy: 0,
            y_cap: 0,
            dnw: 0,
            norm_w_rows: 0,
            dx: 0,
            dab: 0,
            dxn: 0,
            norm_cap: 0,
            dewg: 0,
            dewu: 0,
            dew: 0,
            ew_cap: 0,
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
            kvq: false,
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
            dx32: 0,
            dx32_cap: 0,
            dyt: 0,
            dyt_cap: 0,
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
        if k > self.xh_cap {
            self.graph_invalidate(); // [P10]
            self.cc.sync()?;
            if self.dxh != 0 {
                self.cc.free(self.dxh)?;
            }
            self.dxh = 0; // G1: alloc 실패 시 재시도 이중해제 방지.
            self.xh_cap = 0;
            self.dxh = self.cc.alloc(k * 2)?;
            self.xh_cap = k;
        }
        if n > self.y_cap {
            self.graph_invalidate(); // [P10]
            self.cc.sync()?; // [P10] 비행 커널의 해제 버퍼 사용 차단.
            if self.dy != 0 {
                self.cc.free(self.dy)?;
            }
            self.dy = 0; // G1
            self.y_cap = 0;
            self.dy = self.cc.alloc(n * 4)?;
            self.y_cap = n;
        }
        // P3-b: 신 GEMM은 f32 x 계약 — 호스트에서 h2f(f2h(v)) 동형 변환.
        if k > self.dx32_cap {
            self.graph_invalidate(); // [P10]
            self.cc.sync()?;
            if self.dx32 != 0 {
                self.cc.free(self.dx32)?;
            }
            self.dx32 = 0; // G1
            self.dx32_cap = 0;
            self.dx32 = self.cc.alloc(k * 4)?;
            self.dx32_cap = k;
        }
        let xf: Vec<f32> = x
            .iter()
            .map(|&v| crate::rawcuda::gptq4::h2f(f32_to_f16(v)))
            .collect();
        let xb = unsafe { std::slice::from_raw_parts(xf.as_ptr() as *const u8, xf.len() * 4) };
        self.cc.h2d(self.dx32, xb)?;
        let f = self.cc.function("w4a16_gemm_g128")?;
        let (mut p_q, mut p_s, mut p_x, mut p_y) = (dq, ds, self.dx32, self.dy);
        let (mut p_n, mut p_k, mut p_t) = (n as i32, k as i32, 1i32);
        let mut args: [*mut std::ffi::c_void; 7] = [
            (&mut p_q) as *mut _ as *mut _,
            (&mut p_s) as *mut _ as *mut _,
            (&mut p_x) as *mut _ as *mut _,
            (&mut p_y) as *mut _ as *mut _,
            (&mut p_n) as *mut _ as *mut _,
            (&mut p_k) as *mut _ as *mut _,
            (&mut p_t) as *mut _ as *mut _,
        ];
        // 스테이징 폴백(t=1)도 신 GEMM 커널 계약(8행/블록·512스레드)으로 —
        // 구 계약(grid n/8·block 64)은 재작성 후 1/8행만 계산하는 결함이었다.
        self.cc.launch(f, n.div_ceil(8) as u32, 1, 512, &mut args)?;
        let mut ob = vec![0u8; n * 4];
        self.cc.d2h_async(ob.as_mut_ptr(), self.dy, n * 4)?; // 커스텀 스트림 대비
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

    fn ensure_norm_bufs(&mut self, t_len: usize) -> Result<(), String> {
        let need = t_len * self.hidden;
        if need > self.norm_cap {
            self.graph_invalidate(); // [P10] 재할당 — 캡처 옛 포인터 차단.
            self.cc.sync()?; // [P10] 비행 커널의 해제 버퍼 사용 차단.
            self.norm_cap = 0; // G1: 실패 시 재진입 보장(성공 뒤에만 갱신).
            realloc_fields(
                |p| self.cc.free(p),
                |n| self.cc.alloc(n),
                [&mut self.dx, &mut self.dab, &mut self.dxn],
                [need * 4, need * 4, need * 4],
            )?;
            self.norm_cap = need;
        }
        Ok(())
    }

    /// dx32(x32 버퍼) 용량 보장 — 융합 노름·cast_x32 공용.
    fn ensure_dx32(&mut self, n: usize) -> Result<CUdeviceptr, String> {
        if n > self.dx32_cap {
            self.graph_invalidate(); // [P10]
            self.cc.sync()?; // [P10] 비행 커널의 해제 버퍼 사용 차단.
            if self.dx32 != 0 {
                self.cc.free(self.dx32)?;
            }
            self.dx32 = 0; // G1
            self.dx32_cap = 0;
            self.dx32 = self.cc.alloc(n * 4)?;
            self.dx32_cap = n;
        }
        Ok(self.dx32)
    }

    fn norm_resid_at(
        &mut self,
        w: usize,
        x_dev: CUdeviceptr,
        ab_dev: CUdeviceptr,
        t_len: usize,
        xn32: CUdeviceptr,
    ) -> Result<CUdeviceptr, String> {
        if self.dnw == 0 {
            return Err("norm: 노름 가중 미등록".into());
        }
        if w >= self.norm_w_rows {
            return Err(format!("norm: w={w} >= rows={}", self.norm_w_rows));
        }
        let f = self.cc.function("norm_resid")?;
        let mut tl = t_len as i32;
        let mut wo = (w * self.hidden) as i32;
        let mut hd = self.hidden as i32;
        let (mut a0, mut a1, mut a2, mut a3, mut a4) = (x_dev, self.dnw, ab_dev, self.dxn, xn32);
        let mut args: [*mut std::ffi::c_void; 8] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
            (&mut a3) as *mut _ as *mut _,
            (&mut a4) as *mut _ as *mut _,
            (&mut tl) as *mut _ as *mut _,
            (&mut wo) as *mut _ as *mut _,
            (&mut hd) as *mut _ as *mut _,
        ];
        self.cc.launch(f, t_len as u32, 1, 1024, &mut args)?;
        Ok(self.dxn)
    }

    /// 잔차 x_dev에 ab(호스트) 가산 + 노름 xn 판독(스테이징 1행).
    fn norm_resid_staged(
        &mut self,
        w: usize,
        x_dev: CUdeviceptr,
        ab: &[f32],
    ) -> Result<Vec<f32>, String> {
        if self.hidden == 0 || ab.len() != self.hidden || x_dev == 0 {
            return Err("norm: 순차 잔차/분기 폭 계약 위반".into());
        }
        self.ensure_norm_bufs(1)?;
        let abb = unsafe { std::slice::from_raw_parts(ab.as_ptr() as *const u8, ab.len() * 4) };
        self.cc.h2d(self.dab, abb)?;
        let xn = self.norm_resid_at(w, x_dev, self.dab, 1, 0)?;
        let mut bytes = vec![0u8; self.hidden * 4];
        self.cc.d2h(&mut bytes, xn)?;
        self.cc.sync()?;
        Ok(
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, self.hidden) }
                .to_vec(),
        )
    }

    // ── ew(silu·mul) ──

    fn ensure_ew_bufs(&mut self, n: usize) -> Result<(), String> {
        if n > self.ew_cap {
            self.graph_invalidate(); // [P10]
            self.cc.sync()?; // [P10] 비행 커널의 해제 버퍼 사용 차단.
            self.ew_cap = 0; // G1
            realloc_fields(
                |p| self.cc.free(p),
                |b| self.cc.alloc(b),
                [&mut self.dewg, &mut self.dewu, &mut self.dew],
                [n * 4, n * 4, n * 4],
            )?;
            self.ew_cap = n;
        }
        Ok(())
    }

    pub fn ew_host(&mut self, g: &[f32], u: &[f32]) -> Result<Vec<f32>, String> {
        if g.is_empty() || g.len() != u.len() {
            return Err(format!("ew: g={} u={} 계약 위반", g.len(), u.len()));
        }
        self.ensure_ew_bufs(g.len())?;
        let gb = unsafe { std::slice::from_raw_parts(g.as_ptr() as *const u8, g.len() * 4) };
        let ub = unsafe { std::slice::from_raw_parts(u.as_ptr() as *const u8, u.len() * 4) };
        self.cc.h2d(self.dewg, gb)?;
        self.cc.h2d(self.dewu, ub)?;
        let f = self.cc.function("ew")?;
        let mut nn = g.len() as i32;
        let (mut a0, mut a1, mut a2) = (self.dewg, self.dewu, self.dew);
        let mut args: [*mut std::ffi::c_void; 4] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
            (&mut nn) as *mut _ as *mut _,
        ];
        self.cc
            .launch(f, g.len().div_ceil(128) as u32, 1, 128, &mut args)?;
        let mut yb = vec![0u8; g.len() * 4];
        self.cc.d2h(&mut yb, self.dew)?;
        self.cc.sync()?;
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

    fn ensure_gdn_bufs(&mut self, t_len: usize) -> Result<(), String> {
        if t_len <= self.gdn_t_cap {
            return Ok(());
        }
        let dm = self.gdn.ok_or("GDN: 형상 미등록")?;
        let (hd, cch, kl, vl) = (dm.hidden, dm.conv_ch(), dm.k_len(), dm.v_len());
        self.graph_invalidate(); // [P10]
        self.cc.sync()?; // [P10] 비행 커널의 해제 버퍼 사용 차단.
        self.gdn_t_cap = 0; // G1: 실패 시 재진입 보장.
        realloc_fields(
            |p| self.cc.free(p),
            |b| self.cc.alloc(b),
            [
                &mut self.dqkv,
                &mut self.dzv,
                &mut self.dgxn,
                &mut self.dgq,
                &mut self.dgk,
                &mut self.dgv,
                &mut self.dq2,
                &mut self.dk2,
                &mut self.dv2,
                &mut self.dbg,
                &mut self.dgo,
                &mut self.dgate,
                &mut self.dgpart,
                &mut self.dgdc,
                &mut self.dakq,
            ],
            [
                t_len * cch * 4,
                t_len * vl * 4,
                t_len * hd * 4,
                t_len * kl * 4,
                t_len * kl * 4,
                t_len * vl * 4,
                t_len * kl * 4,
                t_len * kl * 4,
                t_len * vl * 4,
                t_len * dm.bg_len() * 4,
                t_len * vl * 4,
                t_len * vl * 4,
                // [A9] 토큰별 트리오 스크래치 — 배치 디코드 = 슬롯 수 상한.
                self.n_slots * dm.h_v * 8 * 256 * 4,
                self.n_slots * dm.h_v * 128 * 4,
                // [A5-4] A/KQ prepass 스크래치(최대 청크 수 기준, t_len 무관).
                dm.h_v * (CHAIN_TMAX / GDN_CS) * 2 * GDN_CS * GDN_CS * 2,
            ],
        )?;
        self.gdn_t_cap = t_len;
        Ok(())
    }

    /// GDN 체인 디바이스 — [A3 2026-10-09] 입력(xn·qkv·z)을 스테이징 버퍼로
    /// d2d 복사하지 않고 **커널 인자로 직접 소비**한다(호출자 버퍼가 곧 입력).
    fn gdn_chain_dev(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        xn_dev: CUdeviceptr,
        qkv_dev: CUdeviceptr,
        z_dev: CUdeviceptr,
    ) -> Result<(), String> {
        let dm = self.gdn.ok_or("GDN: 형상 미등록")?;
        if layer >= dm.n_gdn || slot >= self.n_slots || t_len == 0 {
            return Err(format!(
                "GDN: 범위 위반 layer={layer} slot={slot} t={t_len}"
            ));
        }
        self.ensure_gdn_bufs(t_len)?;
        let ring_slot = slot * dm.n_gdn * 3 * dm.conv_ch();
        let st_slot = slot * dm.n_gdn * dm.h_v * 128 * 128;
        let (mut tl, mut lay) = (t_len as i32, layer as i32);
        let (mut hk, mut hv, mut dd) = (dm.h_k as i32, dm.h_v as i32, dm.d as i32);
        let (mut hd, mut kl, mut vl, mut cch) = (
            dm.hidden as i32,
            dm.k_len() as i32,
            dm.v_len() as i32,
            dm.conv_ch() as i32,
        );

        let f = self.cc.function("gdn_conv")?;
        let (mut c0, mut c1, mut c2, mut c3, mut c4, mut c5) = (
            qkv_dev,
            self.dcw,
            self.dring + (ring_slot as u64) * 4,
            self.dgq,
            self.dgk,
            self.dgv,
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

        let f = self.cc.function("gdn_l2perm")?;
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
            self.dgq, self.dgk, self.dgv, xn_dev, self.dab_c, self.dalog, self.ddtb, self.dq2,
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

        // [P9] t=1 전용 — i축 분할 3커널(grid h_v×4 = 192블록). ncu 실측
        // gdn_scan 점유 8.3%(지연 바운드) → 분할로 병렬도 확보. t>1은 종전.
        let mut skip_scan = false;
        if t_len == 1 {
            let fp = self.cc.function("gdn1_part")?;
            let (mut p0, mut p1, mut p2, mut p3) = (
                self.dq2,
                self.dk2,
                self.dgst + (st_slot as u64) * 4,
                self.dgpart,
            );
            let mut ap: [*mut std::ffi::c_void; 8] = [
                (&mut p0) as *mut _ as *mut _,
                (&mut p1) as *mut _ as *mut _,
                (&mut p2) as *mut _ as *mut _,
                (&mut p3) as *mut _ as *mut _,
                (&mut hk) as *mut _ as *mut _,
                (&mut hv) as *mut _ as *mut _,
                (&mut dd) as *mut _ as *mut _,
                (&mut lay) as *mut _ as *mut _,
            ];
            self.cc.launch(fp, dm.h_v as u32, 8, 128, &mut ap)?;
            let fc = self.cc.function("gdn1_comb")?;
            let (mut c0, mut c1, mut c2, mut c3, mut c4, mut c5, mut c6) = (
                self.dq2,
                self.dk2,
                self.dv2,
                self.dbg,
                self.dgpart,
                self.dgdc,
                self.dgo,
            );
            let mut acomb: [*mut std::ffi::c_void; 10] = [
                (&mut c0) as *mut _ as *mut _,
                (&mut c1) as *mut _ as *mut _,
                (&mut c2) as *mut _ as *mut _,
                (&mut c3) as *mut _ as *mut _,
                (&mut c4) as *mut _ as *mut _,
                (&mut c5) as *mut _ as *mut _,
                (&mut c6) as *mut _ as *mut _,
                (&mut hk) as *mut _ as *mut _,
                (&mut hv) as *mut _ as *mut _,
                (&mut dd) as *mut _ as *mut _,
            ];
            self.cc.launch(fc, dm.h_v as u32, 1, 128, &mut acomb)?;
            let fu = self.cc.function("gdn1_upd")?;
            let (mut u0, mut u1, mut u2, mut u3) = (
                self.dk2,
                self.dbg,
                self.dgdc,
                self.dgst + (st_slot as u64) * 4,
            );
            let mut aup: [*mut std::ffi::c_void; 8] = [
                (&mut u0) as *mut _ as *mut _,
                (&mut u1) as *mut _ as *mut _,
                (&mut u2) as *mut _ as *mut _,
                (&mut u3) as *mut _ as *mut _,
                (&mut hk) as *mut _ as *mut _,
                (&mut hv) as *mut _ as *mut _,
                (&mut dd) as *mut _ as *mut _,
                (&mut lay) as *mut _ as *mut _,
            ];
            self.cc.launch(fu, dm.h_v as u32, 8, 128, &mut aup)?;
            skip_scan = true;
        }
        if !skip_scan {
            // [A5-4] FLA 2단: A/KQ 청크 병렬 prepass(값 비트동일) →
            // 상태/출력 V-타일 스캔. 스크래치는 dakq(로드 시 확보).
            let nch = t_len.div_ceil(GDN_CS) as u32;
            let fp = self.cc.function("gdn_scan_akq")?;
            let (mut p0, mut p1, mut p2, mut p3) = (self.dq2, self.dk2, self.dbg, self.dakq);
            let (mut pt, mut phk, mut phv, mut pd) =
                (t_len as i32, dm.h_k as i32, dm.h_v as i32, dm.d as i32);
            let mut pa: [*mut std::ffi::c_void; 8] = [
                (&mut p0) as *mut _ as *mut _,
                (&mut p1) as *mut _ as *mut _,
                (&mut p2) as *mut _ as *mut _,
                (&mut p3) as *mut _ as *mut _,
                (&mut pt) as *mut _ as *mut _,
                (&mut phk) as *mut _ as *mut _,
                (&mut phv) as *mut _ as *mut _,
                (&mut pd) as *mut _ as *mut _,
            ];
            self.cc.launch(fp, dm.h_v as u32, nch, 512, &mut pa)?;

            let f = self.cc.function("gdn_scan")?;
            self.cc.set_dynamic_smem(f, GDN_SCAN_SMEM)?;
            let (mut s0, mut s1, mut s2, mut s3, mut s4, mut s5, mut s6) = (
                self.dq2,
                self.dk2,
                self.dv2,
                self.dbg,
                self.dakq,
                self.dgst + (st_slot as u64) * 4,
                self.dgo,
            );
            let mut as_: [*mut std::ffi::c_void; 12] = [
                (&mut s0) as *mut _ as *mut _,
                (&mut s1) as *mut _ as *mut _,
                (&mut s2) as *mut _ as *mut _,
                (&mut s3) as *mut _ as *mut _,
                (&mut s4) as *mut _ as *mut _,
                (&mut s5) as *mut _ as *mut _,
                (&mut s6) as *mut _ as *mut _,
                (&mut tl) as *mut _ as *mut _,
                (&mut hk) as *mut _ as *mut _,
                (&mut hv) as *mut _ as *mut _,
                (&mut dd) as *mut _ as *mut _,
                (&mut lay) as *mut _ as *mut _,
            ];
            // grid = h_v×NSPLIT(블록 = GDN_NGRP×GDN_VS = 512스레드).
            self.cc.launch_shared(
                f,
                (dm.h_v * GDN_NSPLIT) as u32,
                1,
                (GDN_NGRP * (128 / GDN_NSPLIT)) as u32,
                GDN_SCAN_SMEM,
                &mut as_,
            )?;
        }

        let f = self.cc.function("gdn_gate")?;
        let (mut g0, mut g1, mut g2, mut g3) = (self.dgo, z_dev, self.dnwg, self.dgate);
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

    /// GDN 체인 호스트 진입 — xn·qkv·z 업로드 → 4커널 → gated 판독.
    pub fn gdn_chain_host(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        xn: &[f32],
        qkv: &[f32],
        z: &[f32],
    ) -> Result<Vec<f32>, String> {
        let dm = self.gdn.ok_or("GDN: 형상 미등록")?;
        if xn.len() != t_len * dm.hidden
            || qkv.len() != t_len * dm.conv_ch()
            || z.len() != t_len * dm.v_len()
        {
            return Err("GDN: 입력 형상 계약 위반".into());
        }
        self.ensure_gdn_bufs(t_len)?;
        let b =
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        self.cc.h2d(self.dgxn, b(xn))?;
        self.cc.h2d(self.dqkv, b(qkv))?;
        self.cc.h2d(self.dzv, b(z))?;
        let (gxn, gqkv, gzv) = (self.dgxn, self.dqkv, self.dzv);
        self.gdn_chain_dev(slot, layer, t_len, gxn, gqkv, gzv)?;
        let mut ob = vec![0u8; t_len * dm.v_len() * 4];
        self.cc.d2h(&mut ob, self.dgate)?;
        self.cc.sync()?;
        Ok(
            unsafe { std::slice::from_raw_parts(ob.as_ptr() as *const f32, t_len * dm.v_len()) }
                .to_vec(),
        )
    }

    // ── 어텐션 ──

    pub fn set_attn(&mut self, dims: AttnDims, qnw: &[f32], knw: &[f32]) -> Result<(), String> {
        let n = dims.n_attn;
        if qnw.len() != n * dims.d || knw.len() != n * dims.d {
            return Err(format!("attn: qnw/knw != {n}x{}", dims.d));
        }
        let slots = self.n_slots;
        let kv_elems = slots * n * dims.cap * dims.kv_dim();
        // [P13] KV 양자화 — 옵트인(LLM170_KVQ). int8 4× 절감, 스케일은
        // 행×헤드 f32 1개(무시 가능). 미설정 = 종전 f32 경로 그대로.
        let kvq = llm170_diag::flag::on_nonzero("LLM170_KVQ");
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
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        let dq = self.cc.alloc(qnw.len() * 4)?;
        Self::h2d_chunked(&self.cc, dq, b(qnw))?;
        let dk = self.cc.alloc(knw.len() * 4)?;
        Self::h2d_chunked(&self.cc, dk, b(knw))?;
        let (dkc, dvc) = if kvq {
            // [P13] int8 KV + 스케일 2벌. 기록은 attn_prep_q, 판독은
            // attn_fwd3s_part_q(병합은 f32 part 그대로).
            let kc = self.cc.alloc(kv_elems)?;
            Self::zero_dev(&self.cc, kc, kv_elems)?;
            let vc = self.cc.alloc(kv_elems)?;
            Self::zero_dev(&self.cc, vc, kv_elems)?;
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

    fn ensure_attn_bufs(&mut self, t_len: usize) -> Result<(), String> {
        if t_len <= self.attn_t_cap {
            return Ok(());
        }
        let dm = self.attn.ok_or("attn: 형상 미등록")?;
        self.graph_invalidate(); // [P10]
        self.cc.sync()?; // [P10] 비행 커널의 해제 버퍼 사용 차단.
        self.attn_t_cap = 0; // G1: 실패 시 재진입 보장.
        realloc_fields(
            |p| self.cc.free(p),
            |b| self.cc.alloc(b),
            [
                &mut self.dqg_a,
                &mut self.dkin_a,
                &mut self.dvin_a,
                &mut self.dqh_a,
                &mut self.doutv_a,
                &mut self.dattn_part,
            ],
            [
                t_len * dm.qg_dim() * 4,
                t_len * dm.kv_dim() * 4,
                t_len * dm.kv_dim() * 4,
                t_len * dm.q_dim() * 4,
                t_len * dm.q_dim() * 4,
                t_len * dm.q_heads * ATTN_SPLITS * 258 * 4,
            ],
        )?;
        self.attn_t_cap = t_len;
        Ok(())
    }

    pub fn attn_set_pos(&mut self, slot: usize, pos: u32) -> Result<(), String> {
        if self.dpp == 0 || slot >= self.n_slots {
            return Err("attn: pp 미할당/슬롯 범위".into());
        }
        if self.capture_pinned_src {
            // 캡처 중: dpp 갱신은 replay 전 1회(그래프 밖·같은 스트림)로 옮긴다 —
            // 캡처 중 스택 임시를 소스로 잡으면 replay에서 무효 주소가 된다.
            return Ok(());
        }
        // 비동기 — 층마다 동기 H2D를 걸면 스트림이 매번 배수된다(층당 8ms 실측).
        self.cc
            .h2d_async(self.dpp + (slot as u64) * 4, &pos.to_le_bytes())
    }

    fn attn_pp_ptr(&self, slot: usize) -> CUdeviceptr {
        self.dpp + (slot as u64) * 4
    }

    fn attn_kv_ptr(&self, slot: usize) -> CUdeviceptr {
        let b = if self.kvq { 1 } else { 4 }; // [P13] int8=1바이트
        match self.attn {
            Some(dm) => self.dkc + (dm.kv_slot_elems(slot) as u64) * b,
            None => self.dkc,
        }
    }

    fn attn_vc_ptr(&self, slot: usize) -> CUdeviceptr {
        let b = if self.kvq { 1 } else { 4 };
        match self.attn {
            Some(dm) => self.dvc + (dm.kv_slot_elems(slot) as u64) * b,
            None => self.dvc,
        }
    }

    /// [P13] KV 스케일 포인터(슬롯 기저) — KVQ 전용(f32).
    fn attn_ksc_ptr(&self, slot: usize) -> CUdeviceptr {
        match self.attn {
            Some(dm) => self.dksc + (slot * dm.n_attn * dm.cap * dm.kv_heads) as u64 * 4,
            None => self.dksc,
        }
    }

    fn attn_vsc_ptr(&self, slot: usize) -> CUdeviceptr {
        match self.attn {
            Some(dm) => self.dvsc + (slot * dm.n_attn * dm.cap * dm.kv_heads) as u64 * 4,
            None => self.dvsc,
        }
    }

    /// 어텐션 prep — [A3 2026-10-09] qg·kin·vin은 커널 인자 직접 소비
    /// (스테이징 d2d 제거). 출력은 dqh_a(정규화 q)·KV 캐시.
    fn attn_prep_launch(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        qg_dev: CUdeviceptr,
        kin_dev: CUdeviceptr,
        vin_dev: CUdeviceptr,
    ) -> Result<(), String> {
        let dm = self.attn.ok_or("attn: 형상 미등록")?;
        let (mut tl, mut lay) = (t_len as i32, layer as i32);
        if self.kvq {
            // [P13] int8 기록 — ksc/vsc 추가 인자.
            let f = self.cc.function("attn_prep_q")?;
            let (mut qh, mut kvh, mut cp) = (dm.q_heads as i32, dm.kv_heads as i32, dm.cap as i32);
            #[allow(clippy::type_complexity)]
            let (
                mut a0,
                mut a1,
                mut a2,
                mut a3,
                mut a4,
                mut a5,
                mut a6,
                mut a7,
                mut a8,
                mut a9,
                mut aa,
            ) = (
                qg_dev,
                kin_dev,
                vin_dev,
                self.dqnw_a,
                self.dknw_a,
                self.dqh_a,
                self.attn_kv_ptr(slot),
                self.attn_vc_ptr(slot),
                self.attn_ksc_ptr(slot),
                self.attn_vsc_ptr(slot),
                self.attn_pp_ptr(slot),
            );
            let mut args: [*mut std::ffi::c_void; 16] = [
                (&mut a0) as *mut _ as *mut _,
                (&mut a1) as *mut _ as *mut _,
                (&mut a2) as *mut _ as *mut _,
                (&mut a3) as *mut _ as *mut _,
                (&mut a4) as *mut _ as *mut _,
                (&mut a5) as *mut _ as *mut _,
                (&mut a6) as *mut _ as *mut _,
                (&mut a7) as *mut _ as *mut _,
                (&mut a8) as *mut _ as *mut _,
                (&mut a9) as *mut _ as *mut _,
                (&mut aa) as *mut _ as *mut _,
                (&mut tl) as *mut _ as *mut _,
                (&mut lay) as *mut _ as *mut _,
                (&mut qh) as *mut _ as *mut _,
                (&mut kvh) as *mut _ as *mut _,
                (&mut cp) as *mut _ as *mut _,
            ];
            return self.cc.launch(
                f,
                t_len as u32,
                (dm.q_heads + dm.kv_heads) as u32,
                128,
                &mut args,
            );
        }
        let kv = self.attn_kv_ptr(slot);
        let f = self.cc.function("attn_prep")?;
        let (mut qh, mut kvh, mut cp) = (dm.q_heads as i32, dm.kv_heads as i32, dm.cap as i32);
        let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7, mut a8) = (
            qg_dev,
            kin_dev,
            vin_dev,
            self.dqnw_a,
            self.dknw_a,
            self.dqh_a,
            kv,
            self.attn_vc_ptr(slot),
            self.attn_pp_ptr(slot),
        );
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
        self.cc.launch(
            f,
            t_len as u32,
            (dm.q_heads + dm.kv_heads) as u32,
            128,
            &mut args,
        )
    }

    /// fwd3s — [A3 2026-10-09] gate(qg)는 커널 인자 직접 소비(스테이징 제거).
    fn attn_fwd3s_launch(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        qg_dev: CUdeviceptr,
    ) -> Result<(), String> {
        let dm = self.attn.ok_or("attn: 형상 미등록")?;
        if t_len == 0 || t_len > ATTN_F3S_TMAX {
            return Err(format!("attn fwd3s: T={t_len} — 소형 전용 도메인 위반"));
        }
        // [2026-10-09 P8-attn-3] KV 분할 경로 — lim = pp[0]+t+1 ≤ pos+t_len이므로
        // pos+t_len > 256이면 분할(블록 q_heads×S). ≤256은 종전 단일 경로(골든
        // 구간 비트 동일 — 단문/3토큰 프롬프트는 항상 이쪽).
        let pos = self.slot_pos[slot] as usize;
        // [2026-10-09 P8-attn-3b] 분할 수는 커널이 lim으로 결정한다(그래프 캡처
        // 무관) — 호스트는 항상 분할 경로를 쓴다. dattn_part 부재 시만 단일.
        let _ = pos;
        if self.dattn_part != 0 {
            let (mut tl, mut lay) = (t_len as i32, layer as i32);
            let (mut qh, mut kvh, mut cp) = (dm.q_heads as i32, dm.kv_heads as i32, dm.cap as i32);
            let mut sp = ATTN_SPLITS as i32;
            if self.kvq {
                // [P13] int8 KV 판독 — ksc/vsc 추가 인자.
                let fp = self.cc.function("attn_fwd3s_part_q")?;
                #[allow(clippy::type_complexity)]
                let (mut f0, mut f1, mut f2, mut f3, mut f4, mut f5) = (
                    self.dqh_a,
                    self.attn_kv_ptr(slot),
                    self.attn_vc_ptr(slot),
                    self.attn_ksc_ptr(slot),
                    self.attn_vsc_ptr(slot),
                    self.dattn_part,
                );
                let mut f6 = self.attn_pp_ptr(slot);
                let mut pa: [*mut std::ffi::c_void; 13] = [
                    (&mut f0) as *mut _ as *mut _,
                    (&mut f1) as *mut _ as *mut _,
                    (&mut f2) as *mut _ as *mut _,
                    (&mut f3) as *mut _ as *mut _,
                    (&mut f4) as *mut _ as *mut _,
                    (&mut f5) as *mut _ as *mut _,
                    (&mut f6) as *mut _ as *mut _,
                    (&mut tl) as *mut _ as *mut _,
                    (&mut lay) as *mut _ as *mut _,
                    (&mut qh) as *mut _ as *mut _,
                    (&mut kvh) as *mut _ as *mut _,
                    (&mut cp) as *mut _ as *mut _,
                    (&mut sp) as *mut _ as *mut _,
                ];
                self.cc.launch(
                    fp,
                    t_len as u32,
                    (dm.q_heads * ATTN_SPLITS) as u32,
                    256,
                    &mut pa,
                )?;
            } else {
                let fp = self.cc.function("attn_fwd3s_part")?;
                let (mut f0, mut f1, mut f2, mut f3) = (
                    self.dqh_a,
                    self.attn_kv_ptr(slot),
                    self.attn_vc_ptr(slot),
                    self.dattn_part,
                );
                let mut f4 = self.attn_pp_ptr(slot);
                let mut pa: [*mut std::ffi::c_void; 11] = [
                    (&mut f0) as *mut _ as *mut _,
                    (&mut f1) as *mut _ as *mut _,
                    (&mut f2) as *mut _ as *mut _,
                    (&mut f3) as *mut _ as *mut _,
                    (&mut f4) as *mut _ as *mut _,
                    (&mut tl) as *mut _ as *mut _,
                    (&mut lay) as *mut _ as *mut _,
                    (&mut qh) as *mut _ as *mut _,
                    (&mut kvh) as *mut _ as *mut _,
                    (&mut cp) as *mut _ as *mut _,
                    (&mut sp) as *mut _ as *mut _,
                ];
                self.cc.launch(
                    fp,
                    t_len as u32,
                    (dm.q_heads * ATTN_SPLITS) as u32,
                    256,
                    &mut pa,
                )?;
            }
            let fm = self.cc.function("attn_fwd3s_merge")?;
            let (mut mp, mut mg, mut mo) = (self.dattn_part, qg_dev, self.doutv_a);
            let (mut tl2, mut qh2) = (t_len as i32, dm.q_heads as i32);
            let mut ma: [*mut std::ffi::c_void; 6] = [
                (&mut mp) as *mut _ as *mut _,
                (&mut mg) as *mut _ as *mut _,
                (&mut mo) as *mut _ as *mut _,
                (&mut tl2) as *mut _ as *mut _,
                (&mut qh2) as *mut _ as *mut _,
                (&mut sp) as *mut _ as *mut _,
            ];
            return self
                .cc
                .launch(fm, t_len as u32, dm.q_heads as u32, 256, &mut ma);
        }
        let f = self.cc.function("attn_fwd3s")?;
        let (mut tl, mut lay) = (t_len as i32, layer as i32);
        let (mut qh, mut kvh, mut cp) = (dm.q_heads as i32, dm.kv_heads as i32, dm.cap as i32);
        let (mut f0, mut f1, mut f2, mut f3, mut f4, mut f5) = (
            self.dqh_a,
            self.attn_kv_ptr(slot),
            self.attn_vc_ptr(slot),
            qg_dev,
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

    /// 어텐션 체인 호스트 진입 — qg·kin·vin 업로드 → pp=pos0 → prep → fwd3s
    /// → outv 판독.
    pub fn attn_chain_host(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        qg: &[f32],
        kin: &[f32],
        vin: &[f32],
        pos0: u32,
    ) -> Result<Vec<f32>, String> {
        let dm = self.attn.ok_or("attn: 형상 미등록")?;
        if layer >= dm.n_attn || slot >= self.n_slots {
            return Err(format!("attn: 범위 위반 layer={layer} slot={slot}"));
        }
        if qg.len() != t_len * dm.qg_dim()
            || kin.len() != t_len * dm.kv_dim()
            || vin.len() != t_len * dm.kv_dim()
        {
            return Err("attn: 입력 형상 계약 위반".into());
        }
        if pos0 as usize + t_len > dm.cap {
            return Err(format!(
                "attn: pos0={pos0} + T={t_len} > cap={}(--ctx 상향)",
                dm.cap
            ));
        }
        self.ensure_attn_bufs(t_len)?;
        let b =
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        self.cc.h2d(self.dqg_a, b(qg))?;
        self.cc.h2d(self.dkin_a, b(kin))?;
        self.cc.h2d(self.dvin_a, b(vin))?;
        self.attn_set_pos(slot, pos0)?;
        let (aqg, akin, avin) = (self.dqg_a, self.dkin_a, self.dvin_a);
        self.attn_prep_launch(slot, layer, t_len, aqg, akin, avin)?;
        self.attn_fwd3s_launch(slot, layer, t_len, aqg)?;
        let mut buf = vec![0u8; t_len * dm.q_dim() * 4];
        self.cc.d2h(&mut buf, self.doutv_a)?;
        self.cc.sync()?;
        Ok(
            unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const f32, t_len * dm.q_dim()) }
                .to_vec(),
        )
    }

    // ── 순차 forward ──

    /// 1토큰 forward(호스트 스테이징) — 최종 노름 입력(head 입력)을 반환.
    /// KV·GDN 상태를 슬롯 위치만큼 전진시킨다.
    pub fn forward(&mut self, slot: usize, embed_row: &[f32]) -> Result<Vec<f32>, String> {
        let _g = self.cc.guard()?;
        if embed_row.len() != self.hidden || slot >= self.n_slots {
            return Err("forward: 임베딩 폭/슬롯 계약 위반".into());
        }
        let pos = self.slot_pos[slot];
        let cap = self.attn.ok_or("attn: 형상 미등록")?.cap;
        if pos as usize >= cap {
            return Err(format!(
                "context overflow: slot{slot} pos={pos} >= kvcap={cap}"
            ));
        }
        if self.norm_w_rows < 2 * self.n_layers + 1 || self.gdn.is_none() {
            return Err("forward: 디코더 상수 미등록".into());
        }
        let dres = self.cc.alloc(self.hidden * 4)?;
        let result = (|| {
            let row = unsafe {
                std::slice::from_raw_parts(embed_row.as_ptr() as *const u8, self.hidden * 4)
            };
            self.cc.h2d(dres, row)?;
            self.forward_resident(slot, pos, dres)
        })();
        let freed = self.cc.free(dres);
        match (result, freed) {
            (Err(e), _) => Err(e),
            (_, Err(e)) => Err(e),
            (Ok(v), Ok(())) => Ok(v),
        }
    }

    fn forward_resident(
        &mut self,
        slot: usize,
        pos: u32,
        dres: CUdeviceptr,
    ) -> Result<Vec<f32>, String> {
        let mut ab = vec![0.0f32; self.hidden];
        let mut gi = 0usize;
        let interval = self.attn.map(|a| a.interval).unwrap_or(4);
        for il in 0..self.n_layers {
            let xn = self
                .norm_resid_staged(2 * il, dres, &ab)
                .map_err(|e| format!("L{il} input norm: {e}"))?;
            if il == 0 && self.debug_layers {
                eprintln!("  G0 xn[0..4]={:?}", &xn[..4]);
            }
            ab = if (il + 1) % interval == 0 {
                let ai = il / interval;
                let q = self
                    .gemv_host(&format!("blk.{il}.attn_q.weight"), &xn)
                    .map_err(|e| format!("L{il} q: {e}"))?;
                let k = self
                    .gemv_host(&format!("blk.{il}.attn_k.weight"), &xn)
                    .map_err(|e| format!("L{il} k: {e}"))?;
                let v = self
                    .gemv_host(&format!("blk.{il}.attn_v.weight"), &xn)
                    .map_err(|e| format!("L{il} v: {e}"))?;
                let out = self
                    .attn_chain_host(slot, ai, 1, &q, &k, &v, pos)
                    .map_err(|e| format!("L{il} attn: {e}"))?;
                self.gemv_host(&format!("blk.{il}.attn_output.weight"), &out)
                    .map_err(|e| format!("L{il} o: {e}"))?
            } else {
                let qkv = self
                    .gemv_host(&format!("blk.{il}.attn_qkv.weight"), &xn)
                    .map_err(|e| format!("L{il} qkv: {e}"))?;
                let z = self
                    .gemv_host(&format!("blk.{il}.attn_gate.weight"), &xn)
                    .map_err(|e| format!("L{il} z: {e}"))?;
                if il == 0 && self.debug_layers {
                    eprintln!("  G0 qkv[0..4]={:?}", &qkv[..4]);
                    eprintln!("  G0 z[0..4]={:?}", &z[..4]);
                }
                let gated = self
                    .gdn_chain_host(slot, gi, 1, &xn, &qkv, &z)
                    .map_err(|e| format!("L{il} gdn: {e}"))?;
                gi += 1;
                if il == 0 && self.debug_layers {
                    eprintln!("  G0 gated[0..4]={:?}", &gated[..4]);
                }
                let o = self
                    .gemv_host(&format!("blk.{il}.ssm_out.weight"), &gated)
                    .map_err(|e| format!("L{il} ssm_out: {e}"))?;
                if il == 0 && self.debug_layers {
                    eprintln!("  G0 out[0..4]={:?}", &o[..4]);
                }
                o
            };
            let xn = self
                .norm_resid_staged(2 * il + 1, dres, &ab)
                .map_err(|e| format!("L{il} post norm: {e}"))?;
            let gate = self
                .gemv_host(&format!("blk.{il}.ffn_gate.weight"), &xn)
                .map_err(|e| format!("L{il} gate: {e}"))?;
            let up = self
                .gemv_host(&format!("blk.{il}.ffn_up.weight"), &xn)
                .map_err(|e| format!("L{il} up: {e}"))?;
            let act = self
                .ew_host(&gate, &up)
                .map_err(|e| format!("L{il} ew: {e}"))?;
            ab = self
                .gemv_host(&format!("blk.{il}.ffn_down.weight"), &act)
                .map_err(|e| format!("L{il} down: {e}"))?;
            if self.debug_layers {
                let mut db = vec![0u8; self.hidden * 4];
                self.cc.d2h(&mut db, dres)?;
                self.cc.sync()?;
                let d: f64 = db
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c) as f64)
                    .sum();
                let a: f64 = ab.iter().map(|&v| v as f64).sum();
                eprintln!(
                    "  G{il:>2} recr={} sum={:.6}",
                    (il + 1) % interval != 0,
                    d + a
                );
            }
        }
        let xn = self
            .norm_resid_staged(2 * self.n_layers, dres, &ab)
            .map_err(|e| format!("final norm: {e}"))?;
        self.slot_pos[slot] = pos + 1;
        self.attn_set_pos(slot, pos + 1)?;
        Ok(xn)
    }

    /// 슬롯 상태 0화(GDN 링/스캔 + pp + pos).
    pub fn reset_state(&mut self, slot: usize) -> Result<(), String> {
        let _g = self.cc.guard()?;
        if slot >= self.n_slots {
            return Err(format!("reset: slot={slot} >= {}", self.n_slots));
        }
        self.slot_pos[slot] = 0;
        self.attn_set_pos(slot, 0)?;
        if let Some(dims) = self.gdn {
            let ring_bytes = dims.n_gdn * 3 * dims.conv_ch() * 4;
            let st_bytes = dims.n_gdn * dims.h_v * 128 * 128 * 4;
            let ring_off = (slot * dims.n_gdn * 3 * dims.conv_ch()) as u64 * 4;
            let st_off = (slot * dims.n_gdn * dims.h_v * 128 * 128) as u64 * 4;
            Self::zero_dev(&self.cc, self.dring + ring_off, ring_bytes)?;
            Self::zero_dev(&self.cc, self.dgst + st_off, st_bytes)?;
        }
        Ok(())
    }

    // ── 디바이스 체인(S10 — 왕복 제거) ──

    fn lin_spec(&self, name: &str) -> Result<(CUdeviceptr, CUdeviceptr, usize, usize), String> {
        self.lins
            .get(name)
            .copied()
            .ok_or_else(|| format!("상주 선형 없음: {name}"))
    }

    /// 체인 작업 버퍼 보장(1회 — 폭은 선형 형상에서 산출).
    fn ensure_chain_bufs(&mut self) -> Result<(), String> {
        if self.chain_bufs_ok {
            return Ok(());
        }
        let h = self.hidden;
        let ad = self.attn.ok_or("attn: 형상 미등록")?;
        let gd = self.gdn.ok_or("GDN: 형상 미등록")?;
        let (ff_gate, ff_up) = if self.plain_weights {
            // 플레인(MoE) 모드 — dense FFN 없음. ew 스테이징 폭은 shared 폭.
            let (_, g, _) = self.plain_spec("blk.0.moe_shared_gate.weight")?;
            let (_, u, _) = self.plain_spec("blk.0.moe_shared_up.weight")?;
            (g, u)
        } else {
            let (_, _, g, _) = self.lin_spec("blk.0.ffn_gate.weight")?;
            let (_, _, u, _) = self.lin_spec("blk.0.ffn_up.weight")?;
            (g, u)
        };
        // 슬롯 0 = qg·qkv·gate/up, 슬롯 1 = kin/vin·z·up. ew가 gate·up을
        // 동시에 읽으므로 둘 다 FFN 폭 확보(구 S10 ensure_chain_bufs 계약).
        let w0 = ad.qg_dim().max(gd.conv_ch()).max(ff_gate);
        let w1 = ad.kv_dim().max(gd.v_len()).max(ff_up);
        // 배치 프리필(t≤CHAIN_TMAX)까지 수용 — 버퍼는 t배 폭으로 잡는다
        // (t=1 경로는 오프셋 0만 사용하므로 동작 불변).
        let tb = CHAIN_TMAX;
        self.graph_invalidate(); // [P10]
        self.cc.sync()?; // [P10] 비행 커널의 해제 버퍼 사용 차단.
        let [c0, c1, c2, c3, c4] = &mut self.dchain;
        realloc_fields(
            |p| self.cc.free(p),
            |b| self.cc.alloc(b),
            [&mut self.dres, &mut self.dab_dev, c0, c1, c2, c3, c4],
            [
                tb * h * 4,
                tb * h * 4,
                tb * w0 * 4,
                tb * w1 * 4,
                tb * w1 * 4,
                tb * ff_up * 4,
                tb * h * 4,
            ],
        )?;
        Self::zero_dev(&self.cc, self.dab_dev, tb * h * 4)?;
        self.stg_w0 = w0;
        self.stg_w1 = w1;
        self.stg_w2 = ff_up;
        self.chain_bufs_ok = true;
        Ok(())
    }

    /// 활성 f32 → x32(h2f 왕복) 캐스트 1회 — 같은 xn을 쓰는 GEMV들이 공유한다
    /// (q/k/v·gate/up: 종전 gemv마다 캐스트 = 런치 2배). 반환은 self.dx32.
    fn cast_x32(&mut self, x_dev: CUdeviceptr, k: usize) -> Result<CUdeviceptr, String> {
        if k > self.dx32_cap {
            // [P10] 재할당 전 무효화+동기 — 프리필 cast_x32(t×k) 성장이
            // 비행 중 norm xn32 기록을 해제 버퍼로 보낸다(새니타이저 실측).
            self.graph_invalidate();
            self.cc.sync()?;
            if self.dx32 != 0 {
                self.cc.free(self.dx32)?;
            }
            self.dx32 = 0; // G1
            self.dx32_cap = 0;
            self.dx32 = self.cc.alloc(k * 4)?;
            self.dx32_cap = k;
        }
        let f = self.cc.function("w4a16_cast_x32")?;
        let mut nn = k as i32;
        let (mut c0, mut c1) = (x_dev, self.dx32);
        let mut ca: [*mut std::ffi::c_void; 3] = [
            (&mut c0) as *mut _ as *mut _,
            (&mut c1) as *mut _ as *mut _,
            (&mut nn) as *mut _ as *mut _,
        ];
        self.cc.launch(f, k.div_ceil(256) as u32, 1, 256, &mut ca)?;
        Ok(self.dx32)
    }

    /// t≥2 GEMM 발사 — x f16 [t][k] → out [t][n] 직접 쓰기.
    fn gemm_launch(
        &mut self,
        name: &str,
        xh_dev: CUdeviceptr,
        y_out: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        let (dq, ds, n, k) = self.lin_spec(name)?;
        // [T1] split mma 경로(기본 ON — `LLM170_TC=0`으로 해제, t≥16).
        // 검증: 27B 4k·35B 600 토큰이 원본과 동일, 골든 유지, 허용오차 ~6e-4.
        // A=xh(f16 — split 경로가 이미 f2h 캐스트 제공), 계약 완화 승인 후.
        if t >= 16 && llm170_diag::flag::ne0("LLM170_TC") {
            let f = self.cc.function("w4a16_gemm_g128_mma")?;
            let (mut p_q, mut p_s, mut p_x, mut p_y) = (dq, ds, xh_dev, y_out);
            let (mut p_n, mut p_k, mut p_t) = (n as i32, k as i32, t as i32);
            let mut args: [*mut std::ffi::c_void; 7] = [
                (&mut p_q) as *mut _ as *mut _,
                (&mut p_s) as *mut _ as *mut _,
                (&mut p_x) as *mut _ as *mut _,
                (&mut p_y) as *mut _ as *mut _,
                (&mut p_n) as *mut _ as *mut _,
                (&mut p_k) as *mut _ as *mut _,
                (&mut p_t) as *mut _ as *mut _,
            ];
            return self.cc.launch(
                f,
                t.div_ceil(32) as u32,
                n.div_ceil(32) as u32,
                256,
                &mut args,
            );
        }
        let f = self.cc.function("w4a16_gemm_g128")?;
        let (mut p_q, mut p_s, mut p_x, mut p_y) = (dq, ds, xh_dev, y_out);
        let (mut p_n, mut p_k, mut p_t) = (n as i32, k as i32, t as i32);
        let mut args: [*mut std::ffi::c_void; 7] = [
            (&mut p_q) as *mut _ as *mut _,
            (&mut p_s) as *mut _ as *mut _,
            (&mut p_x) as *mut _ as *mut _,
            (&mut p_y) as *mut _ as *mut _,
            (&mut p_n) as *mut _ as *mut _,
            (&mut p_k) as *mut _ as *mut _,
            (&mut p_t) as *mut _ as *mut _,
        ];
        // 8행/블록 커널(512스레드 = 8그룹×64레인) — grid = ceil(n/8).
        self.cc.launch(f, n.div_ceil(8) as u32, 1, 512, &mut args)
    }

    /// 배치 출력 스크래치 보장([t][max_n]).
    /// 플레인 모드(MoE)는 lins가 비어 있다 — plains·전문가 폭까지 포함해야
    /// dyt가 NULL로 남지 않는다(실측: ssm_out GEMM이 NULL에 쓰러 크래시).
    fn ensure_dyt(&mut self, t: usize) -> Result<CUdeviceptr, String> {
        let (mut mn, mut mk) = (0usize, 0usize);
        for &(_, _, n, k) in self.lins.values() {
            mn = mn.max(n);
            mk = mk.max(k);
        }
        for &(_, n, k) in self.plains.values() {
            mn = mn.max(n);
            mk = mk.max(k);
        }
        mn = mn.max(self.hidden).max(self.n_experts);
        let need = t * mn;
        if need > self.dyt_cap {
            self.graph_invalidate(); // [P10]
            self.cc.sync()?; // [P10] 비행 커널의 해제 버퍼 사용 차단.
            if self.dyt != 0 {
                self.cc.free(self.dyt)?;
            }
            self.dyt = 0; // G1
            self.dyt_cap = 0;
            self.dyt = self.cc.alloc(need * 4)?;
            self.dyt_cap = need;
        }
        // 배치 캐스트 입력(xh: t×k f16)도 함께 보장.
        if t * mk > self.xh_cap {
            self.graph_invalidate(); // [P10]
            self.cc.sync()?;
            if self.dxh != 0 {
                self.cc.free(self.dxh)?;
            }
            self.dxh = 0;
            self.xh_cap = 0;
            self.dxh = self.cc.alloc(t * mk * 2)?;
            self.xh_cap = t * mk;
        }
        Ok(self.dyt)
    }

    /// GEMV 발사(공용) — x32 입력 → y_out 직접 쓰기(dy·d2d 경유 제거).
    fn gemv_launch(
        &mut self,
        name: &str,
        x32_dev: CUdeviceptr,
        y_out: CUdeviceptr,
    ) -> Result<(), String> {
        let (dq, ds, n, k) = self.lin_spec(name)?;
        let f = self.cc.function("w4a16_gemv_g128")?;
        let (mut p_q, mut p_s, mut p_x, mut p_y) = (dq, ds, x32_dev, y_out);
        let (mut p_n, mut p_k) = (n as i32, k as i32);
        let mut args: [*mut std::ffi::c_void; 6] = [
            (&mut p_q) as *mut _ as *mut _,
            (&mut p_s) as *mut _ as *mut _,
            (&mut p_x) as *mut _ as *mut _,
            (&mut p_y) as *mut _ as *mut _,
            (&mut p_n) as *mut _ as *mut _,
            (&mut p_k) as *mut _ as *mut _,
        ];
        self.cc.launch(f, n as u32, 1, 64, &mut args)
    }

    /// GEMV(x32 입력) → self.dy — 반환 포인터는 다음 gemv가 덮는다(스트림 순서).
    fn gemv_dev_x32(&mut self, name: &str, x32_dev: CUdeviceptr) -> Result<CUdeviceptr, String> {
        let (_, _, n, _) = self.lin_spec(name)?;
        let dy = self.ensure_dy(n)?;
        self.gemv_launch(name, x32_dev, dy)?;
        Ok(dy)
    }

    /// dy 버퍼 보장(n f32).
    fn ensure_dy(&mut self, n: usize) -> Result<CUdeviceptr, String> {
        if n > self.y_cap {
            self.graph_invalidate(); // [P10]
            self.cc.sync()?; // [P10] 비행 커널의 해제 버퍼 사용 차단.
            if self.dy != 0 {
                self.cc.free(self.dy)?;
            }
            self.dy = 0; // G1
            self.y_cap = 0;
            self.dy = self.cc.alloc(n * 4)?;
            self.y_cap = n;
        }
        Ok(self.dy)
    }

    /// GEMV(x32 입력) → dst 직접 쓰기 + 폭 검사(스테이징 공유 폭 계약).
    fn gemv_stage_x32(
        &mut self,
        name: &str,
        x32_dev: CUdeviceptr,
        dst: CUdeviceptr,
        w: usize,
    ) -> Result<(), String> {
        let (_, _, n, _) = self.lin_spec(name)?;
        if n > w {
            return Err(format!("gemv_stage({name}): n={n} > 스테이징 {w}"));
        }
        self.gemv_launch(name, x32_dev, dst)
    }

    // ── 플레인 bf16(MoE 모델 — 35B) + MoE FFN ──

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
                if self.kvq {
                    rows * (d.kv_dim() as u64) * 2 + rows * (d.kv_heads as u64) * 4 * 2
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

    /// bf16 GEMV — 행=블록(w4a16_gemv_bf16), x는 **원시 f32**(h2f 왕복 없음).
    /// 플레인 경로 판정은 토큰 수준(골든) — split 경로의 레인/환원 구조 미러.
    fn plain_gemv_launch(
        &self,
        name: &str,
        x_dev: CUdeviceptr,
        out_dev: CUdeviceptr,
    ) -> Result<(), String> {
        let (w, n, k) = self.plain_spec(name)?;
        let f = self.cc.function("w4a16_gemv_bf16")?;
        let (mut p_w, mut p_x, mut p_o) = (w, x_dev, out_dev);
        let (mut p_n, mut p_k) = (n as i32, k as i32);
        let mut args: [*mut std::ffi::c_void; 5] = [
            (&mut p_w) as *mut _ as *mut _,
            (&mut p_x) as *mut _ as *mut _,
            (&mut p_o) as *mut _ as *mut _,
            (&mut p_n) as *mut _ as *mut _,
            (&mut p_k) as *mut _ as *mut _,
        ];
        self.cc.launch(f, n as u32, 1, 64, &mut args)
    }

    /// 플레인 bf16 GEMM(t≤8) — x는 원시 f32 [t][k](h2f 왕복 없음),
    /// 가중치 1회 판독 × t토큰 재사용(프리필 청크의 dense 경로).
    fn plain_gemm_launch(
        &self,
        name: &str,
        x_dev: CUdeviceptr,
        y_out: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        let (w, n, k) = self.plain_spec(name)?;
        // [T2] bf16 mma GEMM(기본 ON — `LLM170_TC=0`으로 해제). t≥16에서만
        // (타일 M32 — 부분 타일은 가드로 동작하나 이득이 작음). 검증: 35B 600
        // 토큰 동일, 골든 유지, 허용오차 ~1e-3(bf16 활성 반올림).
        if t >= 16 && llm170_diag::flag::ne0("LLM170_TC") {
            {
                static ONCE: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!("[TC] mma GEMM 경로 진입: {name} n={n} k={k} t={t}");
                }
            }
            let f = self.cc.function("w4a16_gemm_bf16_mma")?;
            let (mut p_w, mut p_x, mut p_o) = (w, x_dev, y_out);
            let (mut p_n, mut p_k, mut p_t) = (n as i32, k as i32, t as i32);
            let mut args: [*mut std::ffi::c_void; 6] = [
                (&mut p_w) as *mut _ as *mut _,
                (&mut p_x) as *mut _ as *mut _,
                (&mut p_o) as *mut _ as *mut _,
                (&mut p_n) as *mut _ as *mut _,
                (&mut p_k) as *mut _ as *mut _,
                (&mut p_t) as *mut _ as *mut _,
            ];
            return self.cc.launch(
                f,
                t.div_ceil(32) as u32,
                n.div_ceil(32) as u32,
                256,
                &mut args,
            );
        }
        // [B2] v3(8행/블록 + 행별 smem + k청크) 상시 — t=1은 GEMV 경로라
        // 여기 오지 않으므로(t≥2) 실질 임계치 = t≥2. v1(t≤8 전용)은 x 재판독이
        // 있어 v3 대비 열위, 산술 순서는 동일 명시(plain_gemm_selfcheck 게이트).
        let v3 = t > 1;
        let f = self.cc.function(if v3 {
            "w4a16_gemm_bf16_t"
        } else {
            "w4a16_gemm_bf16"
        })?;
        let (mut p_w, mut p_x, mut p_o) = (w, x_dev, y_out);
        let (mut p_n, mut p_k, mut p_t) = (n as i32, k as i32, t as i32);
        let mut args: [*mut std::ffi::c_void; 6] = [
            (&mut p_w) as *mut _ as *mut _,
            (&mut p_x) as *mut _ as *mut _,
            (&mut p_o) as *mut _ as *mut _,
            (&mut p_n) as *mut _ as *mut _,
            (&mut p_k) as *mut _ as *mut _,
            (&mut p_t) as *mut _ as *mut _,
        ];
        if v3 {
            self.cc.launch(f, n.div_ceil(8) as u32, 1, 512, &mut args)
        } else {
            self.cc.launch(f, n as u32, 1, 64, &mut args)
        }
    }

    /// [A9 진단] t행 GEMV 벤치(ms/회) — 지정 선형 반복.
    pub fn bench_gemv_t(&mut self, name: &str, t: usize, reps: usize) -> Result<f64, String> {
        let (_, _, n, k) = self.lin_spec(name)?;
        let x = self.cc.alloc(t * k * 4)?;
        let y = self.cc.alloc(t * n * 4)?;
        self.cc.sync()?;
        let r = (|| -> Result<f64, String> {
            self.gemv_t_launch(name, x, y, t)?;
            self.cc.sync()?;
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                self.gemv_t_launch(name, x, y, t)?;
            }
            self.cc.sync()?;
            Ok(t0.elapsed().as_secs_f64() * 1e3 / reps as f64)
        })();
        let _ = self.cc.free(x);
        let _ = self.cc.free(y);
        r
    }

    /// [A9 2026-10-10] 배치 t행 GEMV(split g128) — 가중 판독 1회를 t행 공유,
    /// 행별 산술은 GEMV와 비트동일. x는 cast_x32 산출 f32.
    fn gemv_t_launch(
        &mut self,
        name: &str,
        x_dev: CUdeviceptr,
        y_dev: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        let (dq, ds, n, k) = self.lin_spec(name)?;
        let f = self.cc.function("w4a16_gemv_g128_t")?;
        let (mut p_q, mut p_s, mut p_x, mut p_y) = (dq, ds, x_dev, y_dev);
        let (mut p_n, mut p_k, mut p_t) = (n as i32, k as i32, t as i32);
        let mut args: [*mut std::ffi::c_void; 7] = [
            (&mut p_q) as *mut _ as *mut _,
            (&mut p_s) as *mut _ as *mut _,
            (&mut p_x) as *mut _ as *mut _,
            (&mut p_y) as *mut _ as *mut _,
            (&mut p_n) as *mut _ as *mut _,
            (&mut p_k) as *mut _ as *mut _,
            (&mut p_t) as *mut _ as *mut _,
        ];
        self.cc.launch(
            f,
            n.div_ceil(GEMV_TR) as u32,
            1,
            (64 * GEMV_TR) as u32,
            &mut args,
        )
    }

    /// [A9] 배치 t행 GEMV(플레인 bf16) — 원시 f32 x, 행별 GEMV 비트동일.
    fn plain_gemv_t_launch(
        &mut self,
        name: &str,
        x_dev: CUdeviceptr,
        y_dev: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        let (w, n, k) = self.plain_spec(name)?;
        let f = self.cc.function("w4a16_gemv_bf16_t")?;
        let (mut p_w, mut p_x, mut p_o) = (w, x_dev, y_dev);
        let (mut p_n, mut p_k, mut p_t) = (n as i32, k as i32, t as i32);
        let mut args: [*mut std::ffi::c_void; 6] = [
            (&mut p_w) as *mut _ as *mut _,
            (&mut p_x) as *mut _ as *mut _,
            (&mut p_o) as *mut _ as *mut _,
            (&mut p_n) as *mut _ as *mut _,
            (&mut p_k) as *mut _ as *mut _,
            (&mut p_t) as *mut _ as *mut _,
        ];
        self.cc.launch(
            f,
            n.div_ceil(GEMV_TR) as u32,
            1,
            (64 * GEMV_TR) as u32,
            &mut args,
        )
    }

    /// 플레인 GEMV → 스테이징 dst 직접 쓰기 + 폭 검사.
    fn plain_stage_x32(
        &self,
        name: &str,
        x_dev: CUdeviceptr,
        dst: CUdeviceptr,
        w: usize,
    ) -> Result<(), String> {
        let (_, n, _) = self.plain_spec(name)?;
        if n > w {
            return Err(format!("plain_stage({name}): n={n} > 스테이징 {w}"));
        }
        self.plain_gemv_launch(name, x_dev, dst)
    }

    /// 플레인 GEMV → self.dy.
    fn plain_gemv_dev(&mut self, name: &str, x_dev: CUdeviceptr) -> Result<CUdeviceptr, String> {
        let (_, n, _) = self.plain_spec(name)?;
        let dy = self.ensure_dy(n)?;
        self.plain_gemv_launch(name, x_dev, dy)?;
        Ok(dy)
    }

    /// 분리 GEMV 발사(이름 무경유 — MoE 전문가 스테이징).
    fn gemv_launch_raw(
        &self,
        dq: CUdeviceptr,
        ds: CUdeviceptr,
        n: usize,
        k: usize,
        x_dev: CUdeviceptr,
        y_out: CUdeviceptr,
    ) -> Result<(), String> {
        let sym = super::gptq4::kernel_sym(false, self.moe_group, self.moe_scale_bf16)?;
        let f = self.cc.function(sym)?;
        let (mut p_q, mut p_s, mut p_x, mut p_y) = (dq, ds, x_dev, y_out);
        let (mut p_n, mut p_k) = (n as i32, k as i32);
        let mut args: [*mut std::ffi::c_void; 6] = [
            (&mut p_q) as *mut _ as *mut _,
            (&mut p_s) as *mut _ as *mut _,
            (&mut p_x) as *mut _ as *mut _,
            (&mut p_y) as *mut _ as *mut _,
            (&mut p_n) as *mut _ as *mut _,
            (&mut p_k) as *mut _ as *mut _,
        ];
        self.cc.launch(f, n as u32, 1, 64, &mut args)
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
                let qb = unsafe { std::slice::from_raw_parts(e.0 as *const u8, e.1 as usize) };
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

    /// MoE 버퍼 보장(전문가 스테이징·라우터·출력).
    fn ensure_moe_bufs(&mut self) -> Result<(), String> {
        if self.moe_bufs_ok {
            return Ok(());
        }
        let h = self.hidden;
        let n_ff = self.moe_ffn;
        let n_exp = self.n_experts;
        if n_ff == 0 || n_exp == 0 || !self.moe_group.is_multiple_of(32) {
            return Err("moe: 구성 미등록".into());
        }
        // 프리필 t≤CHAIN_TMAX까지 수용 — 슬롯 = TMAX×top_k, 라우터 = TMAX×n_exp.
        let tmax = CHAIN_TMAX;
        // 전문가 최대 행렬 = h×n_ff(gate/up) = h×n_ff(down) — 동일 크기.
        // [P7 2026-10-09] 층 단위 배치 스테이징 — top_k×3 proj 슬라이스.
        // proj 방향과 무관하게 동일 크기(gate/up [n_ff,h] · down [h,n_ff]):
        // q = h·n_ff/2, s = h·n_ff/g·2. 슬라이스별 업로드 후 sync 1회.
        let np = self.top_k.max(1) * 3;
        let pk = h * n_ff / 2 * np;
        let sk = h * n_ff / self.moe_group * 2 * np;
        self.graph_invalidate(); // [P10]
        self.cc.sync()?; // [P10] 비행 커널의 해제 버퍼 사용 차단.
        realloc_fields(
            |p| self.cc.free(p),
            |b| self.cc.alloc(b),
            [
                &mut self.dstg_q,
                &mut self.dstg_s,
                &mut self.drt,
                &mut self.dmo,
            ],
            [pk, sk, tmax * n_exp * 4, tmax * h * 4],
        )?;
        self.dstg_cap = (pk, sk);
        self.drt_cap = tmax * n_exp;
        self.dmo_cap = tmax * h;
        // 배치 전문가 출력([TMAX×top_k][n_ff]·[TMAX×top_k][h]) + 슬롯 idx/가중.
        let tk = tmax * self.top_k.max(1);
        // [P11] 전문가-우선 정렬 버퍼 — gslot[n_exp][tk] + cnt[n_exp].
        if self.moe_gslot == 0 || self.moe_gmax < tk {
            self.graph_invalidate();
            self.cc.sync()?;
            for p in [self.moe_gslot, self.moe_gcnt, self.moe_goff] {
                if p != 0 {
                    self.cc.free(p)?;
                }
            }
            self.moe_gslot = 0;
            self.moe_gcnt = 0;
            self.moe_goff = 0;
            self.moe_gmax = 0;
            self.moe_gslot = self.cc.alloc(n_exp * tk * 4)?;
            self.moe_gcnt = self.cc.alloc(n_exp * 4)?;
            self.moe_goff = self.cc.alloc(n_exp * 4)?;
            self.moe_gmax = tk;
        }
        if self.dexp_cap < tk {
            for p in [self.dexp_gate, self.dexp_up, self.dexp_act, self.dexp_dn] {
                if p != 0 {
                    self.cc.free(p)?;
                }
            }
            self.dexp_gate = 0;
            self.dexp_up = 0;
            self.dexp_act = 0;
            self.dexp_dn = 0;
            self.dexp_cap = 0;
            self.dexp_gate = self.cc.alloc(tk * n_ff * 4)?;
            self.dexp_up = self.cc.alloc(tk * n_ff * 4)?;
            self.dexp_act = self.cc.alloc(tk * n_ff * 4)?;
            self.dexp_dn = self.cc.alloc(tk * h * 4)?;
            self.dexp_cap = tk;
        }
        for p in [self.moe_idx, self.moe_wt] {
            if p != 0 {
                self.cc.free(p)?;
            }
        }
        self.moe_idx = 0;
        self.moe_wt = 0;
        self.moe_idx = self.cc.alloc(tk * 4)?;
        self.moe_wt = self.cc.alloc(tk * 4)?;
        self.moe_bufs_ok = true;
        Ok(())
    }

    /// 배치 전문가 GEMV(상주) — tab 간접 참조, grid = n × nslots.
    /// xstride = 0(x 공통) 또는 k(슬롯/토큰별 활성), sp = 토큰당 슬롯 수
    /// (프리필 = top_k → x는 토큰 단위 [t][k], 디코드 = 1 → 슬롯별 [nslots][k]).
    #[allow(clippy::too_many_arguments)]
    fn gemv_experts_launch(
        &self,
        base: usize,
        nslots: usize,
        x_dev: CUdeviceptr,
        xstride: usize,
        sp: usize,
        out_dev: CUdeviceptr,
        n: usize,
        k: usize,
    ) -> Result<(), String> {
        let f = self.cc.function("w4a16_gemv_experts_g32_bf16")?;
        let (mut p_t, mut p_b, mut p_i, mut p_ns) =
            (self.moe_dev_tab, base as i32, self.moe_idx, nslots as i32);
        let (mut p_x, mut p_xs, mut p_sp, mut p_o, mut p_n, mut p_k) = (
            x_dev,
            xstride as i32,
            sp as i32,
            out_dev,
            n as i32,
            k as i32,
        );
        let mut args: [*mut std::ffi::c_void; 10] = [
            (&mut p_t) as *mut _ as *mut _,
            (&mut p_b) as *mut _ as *mut _,
            (&mut p_i) as *mut _ as *mut _,
            (&mut p_ns) as *mut _ as *mut _,
            (&mut p_x) as *mut _ as *mut _,
            (&mut p_xs) as *mut _ as *mut _,
            (&mut p_sp) as *mut _ as *mut _,
            (&mut p_o) as *mut _ as *mut _,
            (&mut p_n) as *mut _ as *mut _,
            (&mut p_k) as *mut _ as *mut _,
        ];
        self.cc.launch(f, (n * nslots) as u32, 1, 64, &mut args)
    }

    /// [P11] 전문가-우선 슬롯 순열 발사(프리필 전용) — w4a16_moe_align.
    /// n_exp ≤ 1024 계약(초과 시 호출부가 종전 경로로 폴백).
    fn moe_align_launch(&self, nslots: usize) -> Result<(), String> {
        let f = self.cc.function("w4a16_moe_align")?;
        let (mut p_i, mut p_ns) = (self.moe_idx, nslots as i32);
        let (mut p_g, mut p_c, mut p_o) = (self.moe_gslot, self.moe_gcnt, self.moe_goff);
        let mut p_ne = self.n_experts as i32;
        let mut args: [*mut std::ffi::c_void; 6] = [
            (&mut p_i) as *mut _ as *mut _,
            (&mut p_ns) as *mut _ as *mut _,
            (&mut p_g) as *mut _ as *mut _,
            (&mut p_c) as *mut _ as *mut _,
            (&mut p_o) as *mut _ as *mut _,
            (&mut p_ne) as *mut _ as *mut _,
        ];
        self.cc.launch(f, 1, 1, 256, &mut args)
    }

    /// [P11] 그룹 mma GEMM(g32) 발사 — grid (전문가 × n타일), M=전문가 슬롯 수.
    fn gemm_g32_mma_grp_launch(
        &self,
        base: usize,
        x_dev: CUdeviceptr,
        xstride: usize,
        sp: usize,
        out_dev: CUdeviceptr,
        n: usize,
        k: usize,
    ) -> Result<(), String> {
        let f = self.cc.function("w4a16_gemm_g32_mma_grp")?;
        let (mut p_t, mut p_b) = (self.moe_dev_tab, base as i32);
        let (mut p_g, mut p_c, mut p_of) = (self.moe_gslot, self.moe_gcnt, self.moe_goff);
        let (mut p_x, mut p_xs, mut p_sp, mut p_o, mut p_n, mut p_k) = (
            x_dev,
            xstride as i32,
            sp as i32,
            out_dev,
            n as i32,
            k as i32,
        );
        let mut args: [*mut std::ffi::c_void; 11] = [
            (&mut p_t) as *mut _ as *mut _,
            (&mut p_b) as *mut _ as *mut _,
            (&mut p_g) as *mut _ as *mut _,
            (&mut p_c) as *mut _ as *mut _,
            (&mut p_of) as *mut _ as *mut _,
            (&mut p_x) as *mut _ as *mut _,
            (&mut p_xs) as *mut _ as *mut _,
            (&mut p_sp) as *mut _ as *mut _,
            (&mut p_o) as *mut _ as *mut _,
            (&mut p_n) as *mut _ as *mut _,
            (&mut p_k) as *mut _ as *mut _,
        ];
        self.cc.launch(
            f,
            self.n_experts as u32,
            n.div_ceil(32) as u32,
            256,
            &mut args,
        )
    }

    /// 선택 순서 가중 누적 — y[ti] = Σ_{s∈ti} w[s]·d[s][i] (sp = 토큰당 슬롯).
    fn moe_accum_dev(
        &mut self,
        w_dev: CUdeviceptr,
        d_dev: CUdeviceptr,
        y_dev: CUdeviceptr,
        sp: usize,
        nslots: usize,
        n: usize,
    ) -> Result<(), String> {
        let f = self.cc.function("w4a16_moe_accum")?;
        let (mut p_w, mut p_d, mut p_y) = (w_dev, d_dev, y_dev);
        let (mut p_sp, mut p_ns, mut nn) = (sp as i32, nslots as i32, n as i32);
        let mut args: [*mut std::ffi::c_void; 6] = [
            (&mut p_w) as *mut _ as *mut _,
            (&mut p_d) as *mut _ as *mut _,
            (&mut p_y) as *mut _ as *mut _,
            (&mut p_sp) as *mut _ as *mut _,
            (&mut p_ns) as *mut _ as *mut _,
            (&mut nn) as *mut _ as *mut _,
        ];
        self.cc.launch(f, n.div_ceil(256) as u32, 1, 256, &mut args)
    }

    /// 가중 누적 — y += w·x.
    fn axpy_dev(&mut self, w: f32, x: CUdeviceptr, y: CUdeviceptr, n: usize) -> Result<(), String> {
        let f = self.cc.function("w4a16_axpy")?;
        let mut ww = w;
        let (mut p_x, mut p_y) = (x, y);
        let mut nn = n as i32;
        let mut args: [*mut std::ffi::c_void; 4] = [
            (&mut ww) as *mut _ as *mut _,
            (&mut p_x) as *mut _ as *mut _,
            (&mut p_y) as *mut _ as *mut _,
            (&mut nn) as *mut _ as *mut _,
        ];
        self.cc.launch(f, n.div_ceil(256) as u32, 1, 256, &mut args)
    }

    /// shared 게이트 가산 — y[t][i] += sigmoid(sg[t])·x[t][i] (grid = t).
    fn shared_add_dev(
        &mut self,
        sg: CUdeviceptr,
        x: CUdeviceptr,
        y: CUdeviceptr,
        n: usize,
        t: usize,
    ) -> Result<(), String> {
        let f = self.cc.function("w4a16_shared_add")?;
        let (mut p_sg, mut p_x, mut p_y) = (sg, x, y);
        let mut nn = n as i32;
        let mut args: [*mut std::ffi::c_void; 4] = [
            (&mut p_sg) as *mut _ as *mut _,
            (&mut p_x) as *mut _ as *mut _,
            (&mut p_y) as *mut _ as *mut _,
            (&mut nn) as *mut _ as *mut _,
        ];
        self.cc
            .launch(f, t.max(1) as u32, n.div_ceil(256) as u32, 256, &mut args)
    }

    /// MoE FFN(35B-A3B) — 라우터(bf16 GEMV→호스트 top-k) + 전문가 스트리밍
    /// GEMV + shared. 반환 = self.dmo(잔차 ab로 소비).
    /// 시맨틱은 CPU 스테이지(core qwen35::stages::moe)와 동일: 라우터 전체
    /// softmax → top-k → 재정규화, shared = sigmoid(sgate)·MLP.
    fn moe_ffn_dev(&mut self, il: usize, xn: CUdeviceptr) -> Result<CUdeviceptr, String> {
        let _g = self.cc.guard()?;
        let n_exp = self.n_experts;
        if n_exp == 0 || self.moe_tab.len() < (il + 1) * n_exp * 3 {
            return Err("moe: 구성/전문가 테이블 미등록".into());
        }
        self.ensure_moe_bufs()?;
        // [2026-10-09 P1] 상주는 라우터를 디바이스에서 마무리(top-k 디바이스) —
        // d2h+sync+호스트 moe_topk+h2d 왕복(층당 1회) 제거, 그래프 캡처 가능화.
        // 스트리밍은 전문가 파일 스테이징에 호스트 선택이 필요해 종전 경로 유지
        // (zero_dev도 axpy 누적 전용 — 상주는 moe_accum이 전 행 덮어씀, P2).
        if self.moe_resident {
            self.moe_route_dev(il, xn)?;
            self.moe_experts_batch(il, xn, self.top_k)?;
        } else {
            let sel = self.moe_route(il, xn)?;
            Self::zero_dev(&self.cc, self.dmo, self.hidden * 4)?;
            self.moe_experts_streaming(il, xn, &sel)?;
        }
        self.moe_shared(il, xn)?;
        Ok(self.dmo)
    }

    /// 라우터 디바이스 상주(P1) — gate GEMV → w4a16_moe_topk(idx/wt 디바이스).
    fn moe_route_dev(&mut self, il: usize, xn: CUdeviceptr) -> Result<(), String> {
        self.plain_gemv_launch(&format!("blk.{il}.moe_gate.weight"), xn, self.drt)?;
        let f = self.cc.function("w4a16_moe_topk")?;
        let (mut p_lg, mut p_ix, mut p_wt) = (self.drt, self.moe_idx, self.moe_wt);
        let (mut nn, mut kk) = (self.n_experts as i32, self.top_k as i32);
        let mut args: [*mut std::ffi::c_void; 5] = [
            (&mut p_lg) as *mut _ as *mut _,
            (&mut p_ix) as *mut _ as *mut _,
            (&mut p_wt) as *mut _ as *mut _,
            (&mut nn) as *mut _ as *mut _,
            (&mut kk) as *mut _ as *mut _,
        ];
        self.cc.launch(f, 1, 1, 32, &mut args)
    }

    /// 라우터 — bf16 GEMV(원시 xn) → 로짓 판독 → 호스트 top-k 선택.
    fn moe_route(&mut self, il: usize, xn: CUdeviceptr) -> Result<Vec<(usize, f32)>, String> {
        let n_exp = self.n_experts;
        self.plain_gemv_launch(&format!("blk.{il}.moe_gate.weight"), xn, self.drt)?;
        let mut lb = vec![0u8; n_exp * 4];
        self.cc.d2h_async(lb.as_mut_ptr(), self.drt, n_exp * 4)?;
        self.cc.sync()?;
        let logits: Vec<f32> = lb
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        Ok(moe_topk(&logits, self.top_k))
    }

    /// 전문가 배치(상주) — 테이블 간접 GEMV + 가중 누적.
    /// [P1] idx/wt는 호출자가 디바이스에 기록한다(moe_route_dev) — 종전
    /// h2d 2회(층당) 제거. gDN 경로(moe_ffn_dev_t)는 호스트 선택이 남아
    /// 이 함수 앞에서 moe_idx/moe_wt를 h2d로 채운 뒤 ns를 넘긴다.
    fn moe_experts_batch(&mut self, il: usize, xn: CUdeviceptr, ns: usize) -> Result<(), String> {
        let n_exp = self.n_experts;
        let h = self.hidden;
        let n_ff = self.moe_ffn;
        // gate/up 배치(x 공통) → ew → down 배치(x 슬롯별) → 가중 누적.
        let base = il * n_exp * 3;
        self.gemv_experts_launch(base, ns, xn, 0, ns, self.dexp_gate, n_ff, h)?;
        self.gemv_experts_launch(base + 1, ns, xn, 0, ns, self.dexp_up, n_ff, h)?;
        self.ew_dev(self.dexp_gate, self.dexp_up, self.dexp_act, ns * n_ff)?;
        self.gemv_experts_launch(base + 2, ns, self.dexp_act, n_ff, 1, self.dexp_dn, h, n_ff)?;
        self.moe_accum_dev(self.moe_wt, self.dexp_dn, self.dmo, ns, ns, h)
    }

    /// 전문가 스트리밍(비상주) — 스테이징 1쌍 재사용 h2d + 개별 GEMV.
    fn moe_experts_streaming(
        &mut self,
        il: usize,
        xn: CUdeviceptr,
        sel: &[(usize, f32)],
    ) -> Result<(), String> {
        let n_exp = self.n_experts;
        let h = self.hidden;
        let n_ff = self.moe_ffn;
        let [s0, s1, _s1b, s2, s3] = self.dchain;
        if sel.is_empty() {
            return Ok(());
        }
        // [P7 2026-10-09] 층 단위 배치 스테이징 — (top_k×3) q/s 슬라이스를
        // async 복사로 전부 올린 뒤 sync 1회(종전 proj마다 h2d_chunked —
        // proj당 sync 2회 × 3 × top_k = 층당 수십 sync). 소스는 호스트
        // 스테이징(C2)·mmap — 둘 다 호출 내 수명이면 충분(드라이버 스테이징).
        let qsz = h * n_ff / 2;
        let ssz = h * n_ff / self.moe_group * 2;
        debug_assert!(
            self.dstg_cap.0 >= qsz * sel.len() * 3 && self.dstg_cap.1 >= ssz * sel.len() * 3
        );
        for (j, &(e, _)) in sel.iter().enumerate() {
            let base = (il * n_exp + e) * 3;
            for p in 0..3 {
                let (qp, ql, sp, sl) = self.moe_tab[base + p];
                let qd = self.dstg_q + ((j * 3 + p) * qsz) as u64;
                let sd = self.dstg_s + ((j * 3 + p) * ssz) as u64;
                // SAFETY: tab 항목은 set_expert_table 수명 계약(mmap/호스트
                // 스테이징), ql/sl은 위 qsz/ssz 형상 계약을 따른다.
                let qb = unsafe { std::slice::from_raw_parts(qp as *const u8, ql as usize) };
                let sb = unsafe { std::slice::from_raw_parts(sp as *const u8, sl as usize) };
                self.cc.h2d_async(qd, qb)?;
                self.cc.h2d_async(sd, sb)?;
            }
        }
        self.cc.sync()?;
        for (j, &(_e, w)) in sel.iter().enumerate() {
            let q0 = self.dstg_q + ((j * 3) * qsz) as u64;
            let p0 = self.dstg_s + ((j * 3) * ssz) as u64;
            let q1 = q0 + qsz as u64;
            let p1 = p0 + ssz as u64;
            let q2 = q0 + (2 * qsz) as u64;
            let p2 = p0 + (2 * ssz) as u64;
            // proj 순서 = 테이블 조립 순서(gate, up, down) — 스테이징과 일치.
            self.gemv_launch_raw(q0, p0, n_ff, h, xn, s0)?;
            self.gemv_launch_raw(q1, p1, n_ff, h, xn, s1)?;
            self.ew_dev(s0, s1, s2, n_ff)?;
            self.gemv_launch_raw(q2, p2, h, n_ff, s2, s3)?;
            self.axpy_dev(w, s3, self.dmo, h)?;
        }
        Ok(())
    }

    /// shared 전문가 — sigmoid(sgate·xn)·down(silu(gate·xn)·up·xn).
    fn moe_shared(&mut self, il: usize, xn: CUdeviceptr) -> Result<(), String> {
        if self.shared_ffn == 0 {
            return Ok(());
        }
        let h = self.hidden;
        let sf = self.shared_ffn;
        let [s0, s1, _s1b, s2, s3] = self.dchain;
        self.plain_gemv_launch(&format!("blk.{il}.moe_shared_gate.weight"), xn, s0)?;
        self.plain_gemv_launch(&format!("blk.{il}.moe_shared_up.weight"), xn, s1)?;
        self.ew_dev(s0, s1, s2, sf)?;
        self.plain_gemv_launch(&format!("blk.{il}.moe_shared_down.weight"), s2, s3)?;
        self.plain_gemv_launch(&format!("blk.{il}.moe_shared_sgate.weight"), xn, self.drt)?;
        self.shared_add_dev(self.drt, s3, self.dmo, h, 1)
    }

    /// MoE FFN 배치(t≤8) — 라우터 플레인 GEMM → 디바이스 top-k(t×top_k 슬롯,
    /// 토큰 우선) → 전문가 배치 GEMV(gate/up x=토큰 단위 sp=top_k, down x=슬롯
    /// 단위 sp=1) → 토큰별 누적 → shared. **상주 모드 전용**(스트리밍 프리필은
    /// 미구현 — 호출부가 t=1로 떨어뜨린다).
    fn moe_ffn_dev_t(
        &mut self,
        il: usize,
        xn: CUdeviceptr,
        t: usize,
    ) -> Result<CUdeviceptr, String> {
        let _g = self.cc.guard()?;
        let n_exp = self.n_experts;
        let h = self.hidden;
        let n_ff = self.moe_ffn;
        let tk = self.top_k;
        if n_exp == 0 || self.moe_tab.len() < (il + 1) * n_exp * 3 {
            return Err("moe: 구성/전문가 테이블 미등록".into());
        }
        if !self.moe_resident {
            return Err("moe t>1: 상주 모드 전용(스트리밍 프리필 미구현)".into());
        }
        self.ensure_moe_bufs()?;
        // 1) 라우터 [t][n_exp] — 플레인 GEMM → **디바이스 top-k**(P11).
        // 종전: d2h(32KB)+sync+호스트 전체 정렬(512×t)이 층·청크마다 — 프리필의
        // ~10%. 시맨틱은 moe_topk 미러(softmax→k라운드→재정규화).
        // 실측(2026-10-09): 35B 512토큰 프리필 1294→1088ms.
        self.plain_gemm_launch(&format!("blk.{il}.moe_gate.weight"), xn, self.drt, t)?;
        {
            let f = self.cc.function("w4a16_moe_topk_t")?;
            let (mut p_lg, mut p_ix, mut p_wt) = (self.drt, self.moe_idx, self.moe_wt);
            let (mut p_t, mut p_n, mut p_k) = (t as i32, n_exp as i32, tk as i32);
            let mut a: [*mut std::ffi::c_void; 6] = [
                (&mut p_lg) as *mut _ as *mut _,
                (&mut p_ix) as *mut _ as *mut _,
                (&mut p_wt) as *mut _ as *mut _,
                (&mut p_t) as *mut _ as *mut _,
                (&mut p_n) as *mut _ as *mut _,
                (&mut p_k) as *mut _ as *mut _,
            ];
            self.cc.launch(f, t.div_ceil(8) as u32, 1, 256, &mut a)?;
        }
        let ns = t * tk;
        // 2) 전문가 배치.
        if llm170_diag::flag::on_nonzero("LLM170_MOE_DBG") {
            // xn(정규화 출력) 행별 NaN — 업스트림 vs 전문가 GEMV 판별.
            let mut vb = vec![0u8; t * h * 4];
            self.cc.d2h_async(vb.as_mut_ptr(), xn, t * h * 4)?;
            self.cc.sync()?;
            let v: Vec<f32> = vb
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            let rows: Vec<usize> = (0..t)
                .filter(|&r| v[r * h..(r + 1) * h].iter().any(|x| x.is_nan()))
                .collect();
            eprintln!("[t-dbg] xn nan-rows={rows:?} t={t}");
        }
        // (P11) idx/wt는 이미 디바이스에 있다(topk_t) — h2d 없음.
        let base = il * n_exp * 3;
        // [P11] 프리필(t>1)은 전문가-우선 정렬 + 그룹 GEMV — 같은 전문가의
        // 슬롯을 연속 처리해 가중치 행을 L2 재사용(슬롯별 산술 동일 = 비트 동일).
        let group = t > 1 && n_exp <= 1024;
        if group {
            // [P11] 그룹 mma GEMM — 전문가별 슬롯 묶음(M=슬롯 수), T1 계약 미러.
            self.moe_align_launch(ns)?;
            self.gemm_g32_mma_grp_launch(base, xn, h, tk, self.dexp_gate, n_ff, h)?;
            self.gemm_g32_mma_grp_launch(base + 1, xn, h, tk, self.dexp_up, n_ff, h)?;
        } else {
            self.gemv_experts_launch(base, ns, xn, h, tk, self.dexp_gate, n_ff, h)?;
            self.gemv_experts_launch(base + 1, ns, xn, h, tk, self.dexp_up, n_ff, h)?;
        }
        self.ew_dev(self.dexp_gate, self.dexp_up, self.dexp_act, ns * n_ff)?;
        if llm170_diag::flag::on_nonzero("LLM170_MOE_DBG") {
            let mut vb = vec![0u8; ns * n_ff * 4];
            self.cc
                .d2h_async(vb.as_mut_ptr(), self.dexp_act, ns * n_ff * 4)?;
            self.cc.sync()?;
            let v: Vec<f32> = vb
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            let bad: Vec<usize> = (0..ns)
                .filter(|&s| v[s * n_ff..(s + 1) * n_ff].iter().any(|x| x.is_nan()))
                .collect();
            eprintln!("[t-dbg] act nan-slots={bad:?} ns={ns}");
        }
        if group {
            self.gemm_g32_mma_grp_launch(base + 2, self.dexp_act, n_ff, 1, self.dexp_dn, h, n_ff)?;
        } else {
            self.gemv_experts_launch(base + 2, ns, self.dexp_act, n_ff, 1, self.dexp_dn, h, n_ff)?;
        }
        self.moe_accum_dev(self.moe_wt, self.dexp_dn, self.dmo, tk, ns, h)?;
        self.moe_shared_t(il, xn, t)?;
        Ok(self.dmo)
    }

    /// shared 전문가 배치(t≤8) — 플레인 GEMM ×3 + 토큰별 게이트 가산.
    fn moe_shared_t(&mut self, il: usize, xn: CUdeviceptr, t: usize) -> Result<(), String> {
        if self.shared_ffn == 0 {
            return Ok(());
        }
        let h = self.hidden;
        let sf = self.shared_ffn;
        let [s0, s1, _s1b, s2, s3] = self.dchain;
        self.plain_gemm_launch(&format!("blk.{il}.moe_shared_gate.weight"), xn, s0, t)?;
        self.plain_gemm_launch(&format!("blk.{il}.moe_shared_up.weight"), xn, s1, t)?;
        self.ew_dev(s0, s1, s2, t * sf)?;
        self.plain_gemm_launch(&format!("blk.{il}.moe_shared_down.weight"), s2, s3, t)?;
        self.plain_gemm_launch(
            &format!("blk.{il}.moe_shared_sgate.weight"),
            xn,
            self.drt,
            t,
        )?;
        self.shared_add_dev(self.drt, s3, self.dmo, h, t)
    }

    /// 플레인 GEMM 자가 점검 — 배치 GEMM(t=8) vs 토큰별 GEMV 비트 비교
    /// (판정 계약: 플레인 경로는 토큰 수준이나 같은 레인/환원 순서라 동일해야
    /// 한다 — 다르면 t 처리 결함).
    /// [2026-10-09] 텐서코어 도구·수치 스모크 — bf16 mma.m16n8k16 → f32 누적을
    /// CPU 참조(bf16 RN 반올림 입력 + f32 k순 합)와 대조. 차이는 누적 순서뿐
    /// (허용오차 1e-4). 1b(플레인 mma GEMM) 착륙 전 도구·프래그먼트 검증.
    pub fn mma_smoke(&mut self) -> Result<String, String> {
        let _g = self.cc.guard()?;
        self.cc.load_fatbin(
            "smoke",
            &asset_bytes(
                "LLM170_CUDA_SMOKE_FATBIN_PATH",
                &[
                    "crates/backend-gpu/src/rawcuda/assets/smoke.fatbin",
                    "src/rawcuda/assets/smoke.fatbin",
                ],
            )?,
            &["llm170_mma_smoke", "llm170_smoke_add"],
        )?;
        // 결정적 준난수 ∈ [-1, 1) — 곱·합 ≤ 16이라 f32 누적순서 오차 ~1e-6.
        let mk = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 53) as f32 / 1024.0 - 1.0
                })
                .collect()
        };
        let (a, b) = (mk(16 * 16), mk(16 * 8));
        // bf16 RN(짝수) — 커널 __floats2bfloat162_rn과 동일 규약.
        let bf = |x: f32| -> f32 {
            let u = x.to_bits();
            f32::from_bits((u.wrapping_add(0x7FFF + ((u >> 16) & 1))) & 0xFFFF_0000)
        };
        let ra: Vec<f32> = a.iter().map(|&x| bf(x)).collect();
        let rb: Vec<f32> = b.iter().map(|&x| bf(x)).collect();
        let mut cref = vec![0f32; 16 * 8];
        for m in 0..16 {
            for n2 in 0..8 {
                let mut acc = 0f32;
                for k in 0..16 {
                    acc += ra[m * 16 + k] * rb[k * 8 + n2];
                }
                cref[m * 8 + n2] = acc;
            }
        }
        let da = self.cc.alloc(a.len() * 4)?;
        let db = self.cc.alloc(b.len() * 4)?;
        let dc = self.cc.alloc(cref.len() * 4)?;
        let r = (|| -> Result<Vec<f32>, String> {
            let ab = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len() * 4) };
            let bb = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u8, b.len() * 4) };
            self.cc.h2d(da, ab)?;
            self.cc.h2d(db, bb)?;
            let f = self.cc.function("llm170_mma_smoke")?;
            let (mut pa, mut pb, mut pc) = (da, db, dc);
            let mut ca: [*mut std::ffi::c_void; 3] = [
                (&mut pa) as *mut _ as *mut _,
                (&mut pb) as *mut _ as *mut _,
                (&mut pc) as *mut _ as *mut _,
            ];
            self.cc.launch(f, 1, 1, 32, &mut ca)?;
            self.cc.sync()?;
            let mut ob = vec![0u8; cref.len() * 4];
            self.cc.d2h(&mut ob, dc)?;
            Ok(ob
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect())
        })();
        let _ = self.cc.free(da);
        let _ = self.cc.free(db);
        let _ = self.cc.free(dc);
        let out = r?;
        let mut maxd = 0f32;
        for i in 0..cref.len() {
            maxd = maxd.max((out[i] - cref[i]).abs());
        }
        if !maxd.is_finite() || maxd > 1e-4 {
            return Err(format!(
                "mma_smoke: 최대 오차 {maxd:.3e} > 1e-4 — 프래그먼트/누적 불일치"
            ));
        }
        Ok(format!(
            "mma_smoke OK — bf16 m16n8k16 f32누적(16×16 × 16×8) 최대오차 {maxd:.2e}"
        ))
    }

    pub fn plain_gemm_selfcheck(&mut self) -> Result<String, String> {
        let _g = self.cc.guard()?;
        let (name, n, k) = self
            .plains
            .iter()
            .find(|(nm, (_, n, _))| nm.contains("attn_qkv") && *n <= 8192)
            .map(|(nm, &(_, n, k))| (nm.clone(), n, k))
            .ok_or("plain_gemm_selfcheck: 플레인 qkv 가중 없음")?;
        let t = 8usize;
        let x: Vec<f32> = (0..t * k)
            .map(|i| ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as f32 / 2048.0 - 0.5)
            .collect();
        let xb = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) };
        let dx = self.cc.alloc(t * k * 4)?;
        let da = self.cc.alloc(t * n * 4)?;
        let db = self.cc.alloc(t * n * 4)?;
        let r = (|| -> Result<(Vec<f32>, Vec<f32>), String> {
            self.cc.h2d(dx, xb)?;
            // A: 배치 GEMM(v1, t=8)
            self.plain_gemm_launch(&name, dx, da, t)?;
            // B: 토큰별 GEMV(t=1) ×8 → 이어붙임
            let mut bl = Vec::with_capacity(t * n);
            for ti in 0..t {
                self.plain_gemv_launch(&name, dx + (ti * k * 4) as u64, db)?;
                self.cc.sync()?;
                let mut vb = vec![0u8; n * 4];
                self.cc.d2h(&mut vb, db)?;
                bl.extend(vb.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)));
            }
            self.cc.sync()?;
            let mut ob = vec![0u8; t * n * 4];
            self.cc.d2h(&mut ob, da)?;
            let a: Vec<f32> = ob
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            Ok((a, bl))
        })();
        let _ = self.cc.free(dx);
        let _ = self.cc.free(da);
        let _ = self.cc.free(db);
        let (a, b) = r?;
        let mism = a
            .iter()
            .zip(b.iter())
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        let nan_a = a.iter().filter(|v| v.is_nan()).count();
        let nan_b = b.iter().filter(|v| v.is_nan()).count();
        Ok(format!(
            "plain-gemm-selfcheck {name}: n={n} k={k} t={t} — 불일치 {mism}/{} nan A={nan_a} B={nan_b}",
            a.len()
        ))
    }

    /// MoE 자가 점검 — 직접 GEMV vs 배치(간접) GEMV 비트 비교(층0·전문가0·
    /// gate_proj). 상주 기기 브링업·회귀 판정용.
    pub fn moe_selfcheck(&mut self) -> Result<String, String> {
        if !self.moe_resident || self.n_experts == 0 {
            return Err("moe_selfcheck: 상주 MoE 구성 필요".into());
        }
        let _g = self.cc.guard()?;
        self.ensure_moe_bufs()?;
        let k = self.hidden;
        let n = self.moe_ffn;
        if self.moe_tab.is_empty() {
            return Err("moe_selfcheck: 테이블 부재".into());
        }
        // x — 결정적(splitmix64 계열 상수) ±0.5.
        let x: Vec<f32> = (0..k)
            .map(|i| ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as f32 / 2048.0 - 0.5)
            .collect();
        let xb = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, k * 4) };
        let dx = self.cc.alloc(k * 4)?;
        let da = self.cc.alloc(n * 4)?;
        let db = self.cc.alloc(n * 4)?;
        let r = (|| -> Result<(Vec<f32>, Vec<f32>), String> {
            self.cc.h2d(dx, xb)?;
            // 직접 — moe_tab[0] = (층0, 전문가0, gate_proj).
            let e0 = self.moe_tab[0];
            self.gemv_launch_raw(e0.0, e0.2, n, k, dx, da)?;
            // 배치 — idx=[0], base=0, nslots=1.
            let idx = [0u32];
            let ib = unsafe { std::slice::from_raw_parts(idx.as_ptr() as *const u8, 4) };
            self.cc.h2d(self.moe_idx, ib)?;
            self.gemv_experts_launch(0, 1, dx, 0, 1, db, n, k)?;
            self.cc.sync()?;
            let mut a = vec![0u8; n * 4];
            let mut b = vec![0u8; n * 4];
            self.cc.d2h(&mut a, da)?;
            self.cc.d2h(&mut b, db)?;
            let fa: Vec<f32> = a
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            let fb: Vec<f32> = b
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            Ok((fa, fb))
        })();
        let _ = self.cc.free(dx);
        let _ = self.cc.free(da);
        let _ = self.cc.free(db);
        let (fa, fb) = r?;
        let mism = fa
            .iter()
            .zip(fb.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let maxd = fa
            .iter()
            .zip(fb.iter())
            .map(|(a, b)| (a - b).abs() as f64)
            .fold(0.0f64, f64::max);
        Ok(format!(
            "moe-selfcheck: 직접 vs 배치 n={n} k={k} — 불일치 {mism}/{n} maxdiff={maxd:.3e} (head A={:?} B={:?})",
            &fa[..3],
            &fb[..3]
        ))
    }

    /// 노름 1회(디바이스 x·ab) — xn은 self.dxn(다음 노름이 덮는다).
    fn norm_resid_dev(
        &mut self,
        w: usize,
        x_dev: CUdeviceptr,
        ab_dev: CUdeviceptr,
        t_len: usize,
        xn32: CUdeviceptr,
    ) -> Result<CUdeviceptr, String> {
        self.ensure_norm_bufs(t_len)?;
        self.norm_resid_at(w, x_dev, ab_dev, t_len, xn32)
    }

    /// GDN 체인 디바이스 상주 — xn·qkv·z(디바이스) → dgate.
    fn gdn_chain_dev_run(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        xn_dev: CUdeviceptr,
        qkv_dev: CUdeviceptr,
        z_dev: CUdeviceptr,
    ) -> Result<CUdeviceptr, String> {
        // [A3] 스테이징 d2d 3회/층 제거 — 호출자 버퍼를 커널 인자로 직접 소비.
        // (gdn_chain_dev가 형상·범위 검증·버퍼 보장을 겸한다.)
        self.gdn_chain_dev(slot, layer, t_len, xn_dev, qkv_dev, z_dev)?;
        Ok(self.dgate)
    }

    /// [A9 2026-10-10] GDN 디코드 배치 — 토큰별 슬롯 상태(링·스캔)를 쓴다.
    /// 청크 스캔은 혼합 슬롯에서 의미론이 깨지므로(타 슬롯 토큰과 intra-chunk
    /// 어텐션) t=1 트리오(gdn1_*)를 토큰 수만큼 발사한다 — 행 단위 커널
    /// (l2perm·gate)만 1회. 각 토큰 산술은 단독 t=1 경로와 동일(골든 계약).
    fn gdn_chain_dev_batch(
        &mut self,
        slots: &[usize],
        layer: usize,
        t_len: usize,
        xn_dev: CUdeviceptr,
        qkv_dev: CUdeviceptr,
        z_dev: CUdeviceptr,
    ) -> Result<CUdeviceptr, String> {
        let dm = self.gdn.ok_or("GDN: 형상 미등록")?;
        if t_len == 0 || t_len != slots.len() || slots.iter().any(|&s| s >= self.n_slots) {
            return Err("GDN batch: 슬롯/토큰 계약 위반".into());
        }
        self.ensure_gdn_bufs(t_len)?;
        let (mut kl, mut vl, mut cch, mut hv, mut hd) = (
            dm.k_len() as i32,
            dm.v_len() as i32,
            dm.conv_ch() as i32,
            dm.h_v as i32,
            dm.hidden as i32,
        );
        let (mut hk, mut dd) = (dm.h_k as i32, dm.d as i32);
        let mut lay = layer as i32;
        let mut one = 1i32;
        let st_stride = (dm.n_gdn * dm.h_v * 128 * 128) as u64;
        let ring_stride = (dm.n_gdn * 3 * dm.conv_ch()) as u64;
        let bg_stride = dm.bg_len() as u64;
        // conv — 토큰별(링 = 슬롯).
        let f = self.cc.function("gdn_conv")?;
        for (k, &slot) in slots.iter().enumerate() {
            let mut c0 = qkv_dev + k as u64 * cch as u64 * 4;
            let mut c1 = self.dcw;
            let mut c2 = self.dring + (slot as u64 * ring_stride) * 4;
            let mut c3 = self.dgq + k as u64 * kl as u64 * 4;
            let mut c4 = self.dgk + k as u64 * kl as u64 * 4;
            let mut c5 = self.dgv + k as u64 * vl as u64 * 4;
            let mut ac: [*mut std::ffi::c_void; 11] = [
                (&mut c0) as *mut _ as *mut _,
                (&mut c1) as *mut _ as *mut _,
                (&mut c2) as *mut _ as *mut _,
                (&mut c3) as *mut _ as *mut _,
                (&mut c4) as *mut _ as *mut _,
                (&mut c5) as *mut _ as *mut _,
                (&mut one) as *mut _ as *mut _,
                (&mut lay) as *mut _ as *mut _,
                (&mut kl) as *mut _ as *mut _,
                (&mut vl) as *mut _ as *mut _,
                (&mut cch) as *mut _ as *mut _,
            ];
            self.cc
                .launch(f, (dm.conv_ch() / 128) as u32, 1, 128, &mut ac)?;
        }
        // l2perm — 행 단위 1회.
        {
            let f = self.cc.function("gdn_l2perm")?;
            let (mut l0, mut l1, mut l2, mut l3, mut l4, mut l5, mut l6) = (
                self.dgq, self.dgk, self.dgv, xn_dev, self.dab_c, self.dalog, self.ddtb,
            );
            let (mut l7, mut l8, mut l9, mut l10) = (self.dq2, self.dk2, self.dv2, self.dbg);
            let mut tl = t_len as i32;
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
        }
        // t=1 트리오 — 토큰별(상태 = 슬롯).
        let part_stride = (dm.h_v * 8 * 256) as u64;
        let dc_stride = (dm.h_v * 128) as u64;
        let (fp, fc, fu) = (
            self.cc.function("gdn1_part")?,
            self.cc.function("gdn1_comb")?,
            self.cc.function("gdn1_upd")?,
        );
        for (k, &slot) in slots.iter().enumerate() {
            let q2k = self.dq2 + k as u64 * kl as u64 * 4;
            let k2k = self.dk2 + k as u64 * kl as u64 * 4;
            let v2k = self.dv2 + k as u64 * vl as u64 * 4;
            let bgk = self.dbg + k as u64 * bg_stride * 4;
            let stk = self.dgst + slot as u64 * st_stride * 4;
            let partk = self.dgpart + k as u64 * part_stride * 4;
            let dck = self.dgdc + k as u64 * dc_stride * 4;
            let outk = self.dgo + k as u64 * vl as u64 * 4;
            {
                let (mut p0, mut p1, mut p2, mut p3) = (q2k, k2k, stk, partk);
                let mut ap: [*mut std::ffi::c_void; 8] = [
                    (&mut p0) as *mut _ as *mut _,
                    (&mut p1) as *mut _ as *mut _,
                    (&mut p2) as *mut _ as *mut _,
                    (&mut p3) as *mut _ as *mut _,
                    (&mut hk) as *mut _ as *mut _,
                    (&mut hv) as *mut _ as *mut _,
                    (&mut dd) as *mut _ as *mut _,
                    (&mut lay) as *mut _ as *mut _,
                ];
                self.cc.launch(fp, dm.h_v as u32, 8, 128, &mut ap)?;
            }
            {
                let (mut c0, mut c1, mut c2, mut c3, mut c4, mut c5, mut c6) =
                    (q2k, k2k, v2k, bgk, partk, dck, outk);
                let mut acomb: [*mut std::ffi::c_void; 10] = [
                    (&mut c0) as *mut _ as *mut _,
                    (&mut c1) as *mut _ as *mut _,
                    (&mut c2) as *mut _ as *mut _,
                    (&mut c3) as *mut _ as *mut _,
                    (&mut c4) as *mut _ as *mut _,
                    (&mut c5) as *mut _ as *mut _,
                    (&mut c6) as *mut _ as *mut _,
                    (&mut hk) as *mut _ as *mut _,
                    (&mut hv) as *mut _ as *mut _,
                    (&mut dd) as *mut _ as *mut _,
                ];
                self.cc.launch(fc, dm.h_v as u32, 1, 128, &mut acomb)?;
            }
            {
                let (mut u0, mut u1, mut u2, mut u3) = (k2k, bgk, dck, stk);
                let mut aup: [*mut std::ffi::c_void; 8] = [
                    (&mut u0) as *mut _ as *mut _,
                    (&mut u1) as *mut _ as *mut _,
                    (&mut u2) as *mut _ as *mut _,
                    (&mut u3) as *mut _ as *mut _,
                    (&mut hk) as *mut _ as *mut _,
                    (&mut hv) as *mut _ as *mut _,
                    (&mut dd) as *mut _ as *mut _,
                    (&mut lay) as *mut _ as *mut _,
                ];
                self.cc.launch(fu, dm.h_v as u32, 8, 128, &mut aup)?;
            }
        }
        // gate — 행 단위 1회.
        {
            let f = self.cc.function("gdn_gate")?;
            let (mut g0, mut g1, mut g2, mut g3) = (self.dgo, z_dev, self.dnwg, self.dgate);
            let mut tl = t_len as i32;
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
        }
        Ok(self.dgate)
    }

    /// 어텐션 체인 디바이스 상주 — qg·kin·vin(디바이스) → doutv.
    fn attn_chain_dev_run(
        &mut self,
        slot: usize,
        layer: usize,
        t_len: usize,
        qg_dev: CUdeviceptr,
        kin_dev: CUdeviceptr,
        vin_dev: CUdeviceptr,
    ) -> Result<CUdeviceptr, String> {
        let dm = self.attn.ok_or("attn: 형상 미등록")?;
        if t_len == 0 || t_len > ATTN_F3S_TMAX || layer >= dm.n_attn || slot >= self.n_slots {
            return Err("attn dev: 도메인/범위 위반".into());
        }
        let pos = self.slot_pos[slot];
        if pos as usize + t_len > dm.cap {
            return Err(format!("attn dev: pos{pos}+T{t_len} > cap{}", dm.cap));
        }
        self.attn_set_pos(slot, pos)?;
        self.ensure_attn_bufs(t_len)?;
        // [A3] 스테이징 d2d 3회/층 제거 — 호출자 버퍼를 커널 인자로 직접 소비.
        self.attn_prep_launch(slot, layer, t_len, qg_dev, kin_dev, vin_dev)?;
        self.attn_fwd3s_launch(slot, layer, t_len, qg_dev)?;
        Ok(self.doutv_a)
    }

    /// [A9 2026-10-10] 어텐션 디코드 배치 — 토큰별 슬롯의 KV/pos를 쓰고,
    /// prep·part·merge를 토큰 수만큼 발사(각 t=1 — 단독 경로와 동일 산술).
    /// KVQ(int8)는 미지원(직렬 폴백). 분할 경로 전용(단일 경로는 27B/3토큰
    /// 프리필 골든용 — 디코드는 항상 분할 경로).
    fn attn_chain_dev_batch(
        &mut self,
        slots: &[usize],
        layer: usize,
        t_len: usize,
        qg_dev: CUdeviceptr,
        kin_dev: CUdeviceptr,
        vin_dev: CUdeviceptr,
    ) -> Result<CUdeviceptr, String> {
        let dm = self.attn.ok_or("attn: 형상 미등록")?;
        if self.kvq {
            return Err("attn batch: KVQ 상태 — 직렬 경로로 폴백".into());
        }
        if t_len == 0
            || t_len > BATCH_DEC_MAX
            || t_len != slots.len()
            || layer >= dm.n_attn
            || self.dattn_part == 0
            || slots.iter().any(|&s| s >= self.n_slots)
        {
            return Err("attn batch: 도메인/범위 위반".into());
        }
        // dpp(슬롯 pos)는 배치 진입부가 1회 일괄 h2d(pin_batch_pos) — 캡처
        // 그래프 replay 시에도 갱신되도록 여기서는 호출하지 않는다.
        for &s in slots {
            let pos = self.slot_pos[s];
            if pos as usize + 1 > dm.cap {
                return Err(format!("attn batch: slot{s} pos{pos} > cap{}", dm.cap));
            }
        }
        self.ensure_attn_bufs(t_len)?;
        let (mut qh, mut kvh, mut cp) = (dm.q_heads as i32, dm.kv_heads as i32, dm.cap as i32);
        let (qgd, kvd, qdd) = (dm.qg_dim() as u64, dm.kv_dim() as u64, dm.q_dim() as u64);
        let pstride = (dm.q_heads * ATTN_SPLITS * 258) as u64;
        let mut sp = ATTN_SPLITS as i32;
        // prep — 토큰별.
        let f = self.cc.function("attn_prep")?;
        for (k, &slot) in slots.iter().enumerate() {
            let mut a0 = qg_dev + k as u64 * qgd * 4;
            let mut a1 = kin_dev + k as u64 * kvd * 4;
            let mut a2 = vin_dev + k as u64 * kvd * 4;
            let mut a3 = self.dqnw_a;
            let mut a4 = self.dknw_a;
            let mut a5 = self.dqh_a + k as u64 * qdd * 4;
            let mut a6 = self.attn_kv_ptr(slot);
            let mut a7 = self.attn_vc_ptr(slot);
            let mut a8 = self.attn_pp_ptr(slot);
            let (mut tl, mut lay, mut qh2, mut kvh2, mut cp2) = (1i32, layer as i32, qh, kvh, cp);
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
                (&mut qh2) as *mut _ as *mut _,
                (&mut kvh2) as *mut _ as *mut _,
                (&mut cp2) as *mut _ as *mut _,
            ];
            self.cc
                .launch(f, 1, (dm.q_heads + dm.kv_heads) as u32, 128, &mut args)?;
        }
        // fwd3s part + merge — 토큰별.
        let fp = self.cc.function("attn_fwd3s_part")?;
        let fm = self.cc.function("attn_fwd3s_merge")?;
        for (k, &slot) in slots.iter().enumerate() {
            let qhk = self.dqh_a + k as u64 * qdd * 4;
            let partk = self.dattn_part + k as u64 * pstride * 4;
            {
                let (mut f0, mut f1, mut f2, mut f3) =
                    (qhk, self.attn_kv_ptr(slot), self.attn_vc_ptr(slot), partk);
                let mut f4 = self.attn_pp_ptr(slot);
                let (mut tl, mut lay) = (1i32, layer as i32);
                let mut pa: [*mut std::ffi::c_void; 11] = [
                    (&mut f0) as *mut _ as *mut _,
                    (&mut f1) as *mut _ as *mut _,
                    (&mut f2) as *mut _ as *mut _,
                    (&mut f3) as *mut _ as *mut _,
                    (&mut f4) as *mut _ as *mut _,
                    (&mut tl) as *mut _ as *mut _,
                    (&mut lay) as *mut _ as *mut _,
                    (&mut qh) as *mut _ as *mut _,
                    (&mut kvh) as *mut _ as *mut _,
                    (&mut cp) as *mut _ as *mut _,
                    (&mut sp) as *mut _ as *mut _,
                ];
                self.cc
                    .launch(fp, 1, (dm.q_heads * ATTN_SPLITS) as u32, 256, &mut pa)?;
            }
            {
                let (mut mp, mut mg, mut mo) = (
                    partk,
                    qg_dev + k as u64 * qgd * 4,
                    self.doutv_a + k as u64 * qdd * 4,
                );
                let (mut tl2, mut qh2) = (1i32, dm.q_heads as i32);
                let mut ma: [*mut std::ffi::c_void; 6] = [
                    (&mut mp) as *mut _ as *mut _,
                    (&mut mg) as *mut _ as *mut _,
                    (&mut mo) as *mut _ as *mut _,
                    (&mut tl2) as *mut _ as *mut _,
                    (&mut qh2) as *mut _ as *mut _,
                    (&mut sp) as *mut _ as *mut _,
                ];
                self.cc.launch(fm, 1, dm.q_heads as u32, 256, &mut ma)?;
            }
        }
        Ok(self.doutv_a)
    }

    /// ew(silu·mul) 디바이스 발사 — g·u → y.
    fn ew_dev(
        &mut self,
        g_dev: CUdeviceptr,
        u_dev: CUdeviceptr,
        y_dev: CUdeviceptr,
        n: usize,
    ) -> Result<(), String> {
        if n == 0 {
            return Err("ew: n=0".into());
        }
        let f = self.cc.function("ew")?;
        let mut nn = n as i32;
        let (mut a0, mut a1, mut a2) = (g_dev, u_dev, y_dev);
        let mut args: [*mut std::ffi::c_void; 4] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
            (&mut nn) as *mut _ as *mut _,
        ];
        self.cc.launch(f, n.div_ceil(128) as u32, 1, 128, &mut args)
    }

    /// 1토큰 forward(디바이스 체인) — 왕복은 임베딩 업로드 1회 + 최종 xn
    /// 판독 1회뿐. 산술은 스테이징 경로와 같은 커널·같은 순서(층 4주기).
    fn chain_device(&mut self, slot: usize, embed_row: &[f32]) -> Result<CUdeviceptr, String> {
        let _g = self.cc.guard()?;
        if embed_row.len() != self.hidden || slot >= self.n_slots {
            return Err("forward_device: 임베딩 폭/슬롯 계약 위반".into());
        }
        let pos = self.slot_pos[slot];
        let cap = self.attn.ok_or("attn: 형상 미등록")?.cap;
        if pos as usize >= cap {
            return Err(format!(
                "context overflow: slot{slot} pos={pos} >= kvcap={cap}"
            ));
        }
        if self.norm_w_rows < 2 * self.n_layers + 1 || self.gdn.is_none() {
            return Err("forward_device: 디코더 상수 미등록".into());
        }
        self.ensure_chain_bufs()?;
        let row = if self.capture_pinned_src {
            // 캡처 중: pageable async 복사는 캡처 불가 — pinned 버퍼를 소스로
            // 기록하고 replay가 실행 직전에 내용을 채운다.
            unsafe { std::slice::from_raw_parts(self.pin_embed as *const u8, self.hidden * 4) }
        } else {
            unsafe { std::slice::from_raw_parts(embed_row.as_ptr() as *const u8, self.hidden * 4) }
        };
        self.cc.h2d_async(self.dres, row)?;
        let [s0, s1, s1b, s2, s3] = self.dchain;
        let (w0, w1, w2) = (self.stg_w0, self.stg_w1, self.stg_w2);
        let mut ab = self.dab_dev;
        let mut gi = 0usize;
        // 플레인 모드(MoE 모델) — GEMV는 bf16(head_bf16), x는 원시 f32
        // (h2f 왕복 없음 — CPU 플레인 matmul 계약과 동일). FFN은 MoE.
        let plain = self.plain_weights;
        for il in 0..self.n_layers {
            // 노름이 x32를 융합 기록(cast_x32 노드 제거) — q/k/v(또는 qkv/z) 공유.
            let x32 = if plain {
                0
            } else {
                self.ensure_dx32(self.hidden)?
            };
            let xn = self
                .norm_resid_dev(2 * il, self.dres, ab, 1, x32)
                .map_err(|e| format!("L{il} input norm: {e}"))?;
            let interval = self.attn.ok_or("attn: 형상 미등록")?.interval;
            let branch = if (il + 1) % interval == 0 {
                if plain {
                    self.plain_stage_x32(&format!("blk.{il}.attn_q.weight"), xn, s0, w0)
                        .map_err(|e| format!("L{il} q: {e}"))?;
                    self.plain_stage_x32(&format!("blk.{il}.attn_k.weight"), xn, s1, w1)
                        .map_err(|e| format!("L{il} k: {e}"))?;
                    self.plain_stage_x32(&format!("blk.{il}.attn_v.weight"), xn, s1b, w1)
                        .map_err(|e| format!("L{il} v: {e}"))?;
                } else {
                    self.gemv_stage_x32(&format!("blk.{il}.attn_q.weight"), x32, s0, w0)
                        .map_err(|e| format!("L{il} q: {e}"))?;
                    self.gemv_stage_x32(&format!("blk.{il}.attn_k.weight"), x32, s1, w1)
                        .map_err(|e| format!("L{il} k: {e}"))?;
                    self.gemv_stage_x32(&format!("blk.{il}.attn_v.weight"), x32, s1b, w1)
                        .map_err(|e| format!("L{il} v: {e}"))?;
                }
                self.attn_chain_dev_run(slot, il / interval, 1, s0, s1, s1b)
                    .map_err(|e| format!("L{il} attn: {e}"))?
            } else {
                if plain {
                    self.plain_stage_x32(&format!("blk.{il}.attn_qkv.weight"), xn, s0, w0)
                        .map_err(|e| format!("L{il} qkv: {e}"))?;
                    self.plain_stage_x32(&format!("blk.{il}.attn_gate.weight"), xn, s1, w1)
                        .map_err(|e| format!("L{il} z: {e}"))?;
                } else {
                    self.gemv_stage_x32(&format!("blk.{il}.attn_qkv.weight"), x32, s0, w0)
                        .map_err(|e| format!("L{il} qkv: {e}"))?;
                    self.gemv_stage_x32(&format!("blk.{il}.attn_gate.weight"), x32, s1, w1)
                        .map_err(|e| format!("L{il} z: {e}"))?;
                }
                let g = self
                    .gdn_chain_dev_run(slot, gi, 1, xn, s0, s1)
                    .map_err(|e| format!("L{il} gdn: {e}"))?;
                gi += 1;
                g
            };
            let lo = if (il + 1) % interval == 0 {
                format!("blk.{il}.attn_output.weight")
            } else {
                format!("blk.{il}.ssm_out.weight")
            };
            let out = if plain {
                self.plain_gemv_dev(&lo, branch)
                    .map_err(|e| format!("L{il} {lo}: {e}"))?
            } else {
                let (_, _, _, ko) = self.lin_spec(&lo)?;
                let x32b = self
                    .cast_x32(branch, ko)
                    .map_err(|e| format!("L{il} branch cast: {e}"))?;
                self.gemv_dev_x32(&lo, x32b)
                    .map_err(|e| format!("L{il} {lo}: {e}"))?
            };
            // [P10] cast_x32(ko) 재할당 대비 — 현재 dx32 재확인(위 주석 참조).
            let x32 = self.ensure_dx32(self.hidden)?;
            let xn2 = self
                .norm_resid_dev(2 * il + 1, self.dres, out, 1, x32)
                .map_err(|e| format!("L{il} post norm: {e}"))?;
            if plain {
                // MoE FFN(35B-A3B) — 잔차 ab = MoE 출력(dmo).
                ab = self
                    .moe_ffn_dev(il, xn2)
                    .map_err(|e| format!("L{il} moe: {e}"))?;
            } else {
                let x32n = x32; // 노름 융합 기록
                let _ = xn2;
                self.gemv_stage_x32(&format!("blk.{il}.ffn_gate.weight"), x32n, s0, w0)
                    .map_err(|e| format!("L{il} gate: {e}"))?;
                self.gemv_stage_x32(&format!("blk.{il}.ffn_up.weight"), x32n, s1, w1)
                    .map_err(|e| format!("L{il} up: {e}"))?;
                self.ew_dev(s0, s1, s2, w2)?;
                let dn = format!("blk.{il}.ffn_down.weight");
                let (_, _, _, kd) = self.lin_spec(&dn)?;
                let x32d = self
                    .cast_x32(s2, kd)
                    .map_err(|e| format!("L{il} down cast: {e}"))?;
                // down은 s3 직접 쓰기 — dy 경유 d2d 제거.
                self.gemv_launch(&dn, x32d, s3)
                    .map_err(|e| format!("L{il} down: {e}"))?;
                ab = s3;
            }
            if self.debug_layers {
                let mut db = vec![0u8; self.hidden * 4];
                let mut abv = vec![0u8; self.hidden * 4];
                self.cc.d2h(&mut db, self.dres)?;
                self.cc.d2h(&mut abv, ab)?;
                self.cc.sync()?;
                let d: f64 = db
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c) as f64)
                    .sum();
                let a: f64 = abv
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c) as f64)
                    .sum();
                eprintln!("  G{il:>2} recr={} sum={:.6}", (il + 1) % 4 != 0, d + a);
            }
        }
        let xn_final = self
            .norm_resid_dev(2 * self.n_layers, self.dres, ab, 1, 0)
            .map_err(|e| format!("final norm: {e}"))?;
        Ok(xn_final)
    }

    // ── CUDA Graph(체인 캡처 — P1) ──

    /// 그래프 모드 가능 여부 — **기본 ON**(LLM170_GRAPH=0으로 끔).
    /// 캡처 실패(에러 반환) 후에는 직접 경로 고정(매 토큰 재시도 방지).
    /// debug_layers는 캡처 중 d2h/sync를 하므로 그래프 불가.
    ///
    /// [실측 2026-10-08, 27B·4090] 밀집 디코드는 **중립**(그래프 39.9 vs
    /// 직접 40.8 ms/토큰, n=24) — 손해가 없어 기본 ON으로 간다. 이득 상한은
    /// 호스트 enqueue(12ms)이고, GPU가 CPU를 기다릴 때만 회수된다. 값 하는
    /// 곳은 런치 바운드: MoE 전문가 소형 커널(35B-A3B 256×40)·오프로드·
    /// 다중 슬롯 — W4-1에서 데이터 주도 디스패치와 함께 재검한다.
    fn graph_ok(&self) -> bool {
        llm170_diag::flag::ne0("LLM170_GRAPH")
            && !self.debug_layers
            && !self.graph_failed
            // [2026-10-09 P1] MoE 상주 경로는 캡처 가능해졌다: 라우터 top-k가
            // 디바이스(w4a16_moe_topk — 호스트 왕복 0), 전문가 디스패치가
            // 데이터 주도(디바이스 idx 포인터 테이블 — 포인터는 고정), h2d/sync
            // 없음(P1·P2). 스트리밍은 전문가 파일 스테이징(h2d_chunked sync)이라
            // 여전히 불가. 미검증 경로는 직접 경로 폴백(graph_failed)이 덮는다.
            && (self.n_experts == 0 || self.moe_resident)
    }

    /// 캡처 그래프 무효화 — 버퍼 재할당 시 옛 포인터 replay를 차단한다
    /// (다음 디코드가 재캡처). 재할당은 캡처 밖(프리필·업로드)에서만 일어난다.
    /// [P10 실측 2026-10-09] 워밍업 t=16 캡처 → t=128 프리필이 norm 버퍼를
    /// 재할당 → replay가 해제 주소에 기록(norm_resid OOB) → CUDA 700.
    fn graph_invalidate(&mut self) {
        // [A8] 캐시 전량 폐기 — 옛 포인터를 기록한 exec는 replay 금지.
        for e in self.graph_cache.drain(..) {
            let _ = self.cc.graph_destroy(e.exec, e.handle);
        }
        // [A9] 배치(슬롯집합) 그래프도 동일 계약.
        for (_, e, g) in self.batch_graphs.drain(..) {
            let _ = self.cc.graph_destroy(e, g);
        }
    }

    /// 캡처 전 버퍼 워밍업 — **불변식: 체인에서 지연 할당되는 모든 버퍼는
    /// 여기서 선할당한다.** 캡처 중 `cuMemAlloc`은 금지 API라 드라이버가
    /// instantiate에서 SIGSEGV로 죽는다(2026-10-08 실측 — gemv dx32/dy 누락이
    /// 원인이었다). 체인에 새 버퍼를 추가하면 반드시 이 목록에도 추가할 것.
    /// 현행 지연 할당원: chain(dres/dab_dev/dchain) · norm(dx/dab/dxn) ·
    /// gdn 12종 · attn 5종 · gemv_dev(dx32/dy, 선형 전수 최대치) ·
    /// head(head_w/head_out — upload_head 소관).
    fn warm_for_capture(&mut self) -> Result<(), String> {
        self.ensure_chain_bufs()?;
        if self.n_experts > 0 {
            // P1: MoE 상주 체인의 지연 할당원(drt/dexp_*/dmo/moe_idx/moe_wt).
            self.ensure_moe_bufs()?;
        }
        // [P10] t=1이 아니라 **최대 청크(CHAIN_TMAX)** 로 선할당 — 그래프는
        // 캡처 시점의 포인터를 기록하므로, 이후 프리필이 버퍼를 재할당하면
        // replay가 해제 주소를 쓴다(위 graph_invalidate 주석의 실측 결함).
        self.ensure_norm_bufs(CHAIN_TMAX)?;
        self.ensure_gdn_bufs(CHAIN_TMAX)?;
        self.ensure_attn_bufs(CHAIN_TMAX)?;
        self.ensure_dyt(CHAIN_TMAX)?;
        // [P11 fix] lins + plains **둘 다** — 플레인(MoE) 모델은 가중치가
        // plains에 있어 lins만 보면 dy/dx32가 0 → 캡처 중 ensure_dy 재할당 →
        // 재할당 경로의 sync가 캡처 금지 API(CUresult=900)로 캡처 실패(실측:
        // 35B serve가 직접 경로로 폴백 → tg 93.6→81.6).
        let (mut mk, mut mn) = (0usize, 0usize);
        for &(_, _, n, k) in self.lins.values() {
            mk = mk.max(k);
            mn = mn.max(n);
        }
        for &(_, n, k) in self.plains.values() {
            mk = mk.max(k);
            mn = mn.max(n);
        }
        // 프리필 norm 융합(ensure_dx32(t*h))과 FFN 캐스트(cast_x32(t×k))를
        // 모두 커버 — 캡처 후 재할당(재캡처·옛 포인터)을 봉인한다.
        mk = mk.max(CHAIN_TMAX * mk.max(self.hidden));
        mn = mn.max(self.hidden);
        if mk > self.dx32_cap {
            self.graph_invalidate(); // [P10]
            self.cc.sync()?;
            if self.dx32 != 0 {
                self.cc.free(self.dx32)?;
            }
            self.dx32 = 0; // G1 관례: 실패 시 재시도 이중해제 방지.
            self.dx32_cap = 0;
            self.dx32 = self.cc.alloc(mk * 4)?;
            self.dx32_cap = mk;
        }
        if mn > self.y_cap {
            self.graph_invalidate(); // [P10]
            self.cc.sync()?;
            if self.dy != 0 {
                self.cc.free(self.dy)?;
            }
            self.dy = 0;
            self.y_cap = 0;
            self.dy = self.cc.alloc(mn * 4)?;
            self.y_cap = mn;
        }
        Ok(())
    }

    /// 그래프 준비 — 캐시 히트면 즉시 exec 반환(A8, 슬롯/모드별 1회 캡처).
    /// 미스 시 버퍼 워밍업 → 캡처 → 인스턴스화 → 캐시 등록.
    /// 캡처 중 금지 API(동기 복사·alloc)를 배제하기 위해 ensure_*를 선행한다.
    fn ensure_graph(
        &mut self,
        slot: usize,
        head: bool,
        argmax: bool,
    ) -> Result<ffi::CUgraphExec, String> {
        // [A8] 캐시 히트 — 슬롯/모드 전환 재캡처 제거.
        if let Some(e) = self
            .graph_cache
            .iter()
            .find(|e| e.slot == slot && e.head == head && e.argmax == argmax)
        {
            return Ok(e.exec);
        }
        self.warm_for_capture()?;
        // 2) 실스트림·pinned 1회 준비.
        if self.pin_embed.is_null() {
            self.cc.create_stream()?;
            self.pin_embed = self.cc.pinned_alloc(self.hidden * 4)?;
            self.pin_pos = self.cc.pinned_alloc(self.n_slots * 4)?;
        }
        let out_len = if head {
            self.head_n * 4
        } else {
            self.hidden * 4
        };
        if self.pin_out.is_null() || self.pin_out_len < out_len {
            // [A8 잠복 수정 2026-10-10] 핀드 출력 재할당 = 기존 exec들의 d2h
            // 목적지 해제 — 캐시된 그래프 전량 폐기(모드 교대에서 해제 주소
            // 기록 UAF, 할당자 재사용으로 잠복했던 실측 결함).
            self.graph_invalidate();
            if !self.pin_out.is_null() {
                let _ = self.cc.pinned_free(self.pin_out);
            }
            self.pin_out = self.cc.pinned_alloc(out_len)?;
            self.pin_out_len = out_len;
        }
        // 3) 캡처(실행 없음 — 기록만).
        self.cc.capture_begin()?;
        self.capture_pinned_src = true;
        let cap = (|| -> Result<(), String> {
            let row = vec![0f32; self.hidden]; // 내용 무의미(캡처는 실행 아님).
            let xn = self.chain_device(slot, &row)?;
            if head {
                let f = self.cc.function("head_bf16")?;
                let (mut p_w, mut p_x, mut p_o) = (self.head_w, xn, self.head_out);
                let (mut p_n, mut p_k) = (self.head_n as i32, self.head_k as i32);
                let mut args: [*mut std::ffi::c_void; 5] = [
                    (&mut p_w) as *mut _ as *mut _,
                    (&mut p_x) as *mut _ as *mut _,
                    (&mut p_o) as *mut _ as *mut _,
                    (&mut p_n) as *mut _ as *mut _,
                    (&mut p_k) as *mut _ as *mut _,
                ];
                self.cc
                    .launch(f, self.head_n.div_ceil(4 * 256) as u32, 1, 256, &mut args)?;
                if argmax {
                    // [P3] argmax 커널 + 4B d2h — 로짓 전량 readback 제거.
                    let fa = self.cc.function("w4a16_argmax_min")?;
                    let (mut p_l, mut p_n, mut p_o) =
                        (self.head_out, self.head_n as i32, self.argmax_out);
                    let mut aa: [*mut std::ffi::c_void; 3] = [
                        (&mut p_l) as *mut _ as *mut _,
                        (&mut p_n) as *mut _ as *mut _,
                        (&mut p_o) as *mut _ as *mut _,
                    ];
                    self.cc.launch(fa, 1, 1, 1024, &mut aa)?;
                    self.cc
                        .d2h_async(self.pin_out as *mut u8, self.argmax_out, 4)?;
                } else {
                    self.cc
                        .d2h_async(self.pin_out as *mut u8, self.head_out, self.head_n * 4)?;
                }
            } else {
                self.cc
                    .d2h_async(self.pin_out as *mut u8, xn, self.hidden * 4)?;
            }
            Ok(())
        })();
        self.capture_pinned_src = false;
        if let Err(e) = cap {
            let _ = self.cc.capture_end(); // 캡처 상태 정리(그래프 폐기).
            return Err(format!("캡처 본문: {e}"));
        }
        let g = self.cc.capture_end()?;
        let e = self.cc.graph_instantiate(g)?;
        // [A8] 캐시 등록 — 상한 = 슬롯×모드 3종(방어적으로 초과 시 최古 폐기).
        let cap = self.n_slots.max(1) * 3;
        if self.graph_cache.len() >= cap {
            let old = self.graph_cache.remove(0);
            let _ = self.cc.graph_destroy(old.exec, old.handle);
        }
        self.graph_cache.push(GraphEntry {
            exec: e,
            handle: g,
            slot,
            head,
            argmax,
        });
        // 캡처 성공 1회 로그 — 경로 가시화(MoE 상주 = P1 개방분 포함).
        eprintln!(
            "[graph] captured slot={slot} head={head} argmax={argmax} moe_resident={}",
            self.moe_resident
        );
        Ok(e)
    }

    /// 그래프 replay — pinned 입력 기입 → dpp 1회 갱신 → launch 1회 → sync →
    /// pinned 출력 회수. 반환: head 모드면 로짓, 아니면 xn.
    fn graph_replay(
        &mut self,
        exec: ffi::CUgraphExec,
        slot: usize,
        embed_row: &[f32],
    ) -> Result<Vec<f32>, String> {
        let pos = self.slot_pos[slot];
        // SAFETY: pinned 버퍼는 hidden*4/n_slots*4 크기 계약(ensure_graph 할당).
        unsafe {
            std::ptr::copy_nonoverlapping(
                embed_row.as_ptr() as *const u8,
                self.pin_embed as *mut u8,
                self.hidden * 4,
            );
            std::ptr::copy_nonoverlapping(
                pos.to_le_bytes().as_ptr(),
                (self.pin_pos as *mut u8).add(slot * 4),
                4,
            );
        }
        // dpp 갱신은 그래프 밖·같은 스트림(그래프보다 먼저 실행 — 순서 보장).
        let posb =
            unsafe { std::slice::from_raw_parts(self.pin_pos as *const u8, self.n_slots * 4) };
        self.cc
            .h2d_async(self.dpp + (slot as u64) * 4, &posb[slot * 4..slot * 4 + 4])?;
        self.cc.graph_launch(exec)?;
        self.cc.sync()?;
        let out =
            unsafe { std::slice::from_raw_parts(self.pin_out as *const f32, self.pin_out_len / 4) };
        let v = out.to_vec();
        self.slot_pos[slot] = pos + 1;
        Ok(v)
    }

    /// [P3] 그래프 replay(argmax 모드) — 4B 인덱스 회수. graph_replay와 동일
    /// 계약(입력 pinned 기입 → dpp 1회 → launch 1회 → sync).
    fn graph_replay_argmax(
        &mut self,
        exec: ffi::CUgraphExec,
        slot: usize,
        embed_row: &[f32],
    ) -> Result<u32, String> {
        let pos = self.slot_pos[slot];
        // SAFETY: pinned 버퍼는 hidden*4/n_slots*4 크기 계약(ensure_graph 할당).
        unsafe {
            std::ptr::copy_nonoverlapping(
                embed_row.as_ptr() as *const u8,
                self.pin_embed as *mut u8,
                self.hidden * 4,
            );
            std::ptr::copy_nonoverlapping(
                pos.to_le_bytes().as_ptr(),
                (self.pin_pos as *mut u8).add(slot * 4),
                4,
            );
        }
        let posb =
            unsafe { std::slice::from_raw_parts(self.pin_pos as *const u8, self.n_slots * 4) };
        self.cc
            .h2d_async(self.dpp + (slot as u64) * 4, &posb[slot * 4..slot * 4 + 4])?;
        self.cc.graph_launch(exec)?;
        self.cc.sync()?;
        let ob = unsafe { std::slice::from_raw_parts(self.pin_out as *const u8, 4) };
        self.slot_pos[slot] = pos + 1;
        Ok(u32::from_le_bytes([ob[0], ob[1], ob[2], ob[3]]))
    }

    /// [P3] 디코드 argmax — head 로짓을 디바이스에서 argmax(그래프 = 4B d2h).
    /// 반환 = 토큰 인덱스(min-index-on-tie = CPU greedy_from 계약 미러).
    pub fn forward_device_argmax(&mut self, slot: usize, embed_row: &[f32]) -> Result<u32, String> {
        if self.head_w == 0 || self.argmax_out == 0 {
            return Err("forward_device_argmax: head/argmax 미등록 — upload_head 선행".into());
        }
        let _g = self.cc.guard()?;
        if self.graph_ok() {
            match self
                .ensure_graph(slot, true, true)
                .and_then(|exec| self.graph_replay_argmax(exec, slot, embed_row))
            {
                Ok(t) => return Ok(t),
                Err(e) => {
                    eprintln!("[graph] argmax 경로 실패 — 직접 경로 폴백: {e}");
                    self.graph_failed = true;
                }
            }
        }
        let pos = self.slot_pos[slot];
        let xn = self.chain_device(slot, embed_row)?;
        let f = self.cc.function("head_bf16")?;
        let (mut p_w, mut p_x, mut p_o) = (self.head_w, xn, self.head_out);
        let (mut p_n, mut p_k) = (self.head_n as i32, self.head_k as i32);
        let mut args: [*mut std::ffi::c_void; 5] = [
            (&mut p_w) as *mut _ as *mut _,
            (&mut p_x) as *mut _ as *mut _,
            (&mut p_o) as *mut _ as *mut _,
            (&mut p_n) as *mut _ as *mut _,
            (&mut p_k) as *mut _ as *mut _,
        ];
        self.cc
            .launch(f, self.head_n.div_ceil(4 * 256) as u32, 1, 256, &mut args)?;
        let fa = self.cc.function("w4a16_argmax_min")?;
        let (mut p_l, mut p_nn, mut p_a) = (self.head_out, self.head_n as i32, self.argmax_out);
        let mut aa: [*mut std::ffi::c_void; 3] = [
            (&mut p_l) as *mut _ as *mut _,
            (&mut p_nn) as *mut _ as *mut _,
            (&mut p_a) as *mut _ as *mut _,
        ];
        self.cc.launch(fa, 1, 1, 1024, &mut aa)?;
        let mut ob = [0u8; 4];
        self.cc.d2h_async(ob.as_mut_ptr(), self.argmax_out, 4)?;
        self.cc.sync()?;
        self.slot_pos[slot] = pos + 1;
        self.attn_set_pos(slot, pos + 1)?;
        Ok(u32::from_le_bytes(ob))
    }

    /// [A9 2026-10-10] 배치 greedy 디코드 — 슬롯별 임베딩 행(t=n_active)을
    /// 단일 체인으로 통과, 슬롯 순서의 다음 토큰 반환. 호출부가 전제
    /// (전 슬롯 greedy·t≤BATCH_DEC_MAX)를 보장한다.
    pub fn forward_device_argmax_batch(
        &mut self,
        slots: &[usize],
        rows: &[f32],
    ) -> Result<Vec<u32>, String> {
        let t = slots.len();
        if !(2..=BATCH_DEC_MAX).contains(&t) {
            return Err("batch: t 2..=BATCH_DEC_MAX 전용".into());
        }
        let _g = self.cc.guard()?;
        if rows.len() != t * self.hidden
            || (self.n_experts > 0 && !self.moe_resident)
            || self.head_w == 0
            || slots.iter().any(|&s| s >= self.n_slots)
        {
            return Err(format!("batch: 전제 위반 t={t}"));
        }
        let cap = self.attn.ok_or("attn: 형상 미등록")?.cap;
        for &s in slots {
            if self.slot_pos[s] as usize + 1 > cap {
                return Err(format!("batch: slot{s} 컨텍스트 초과"));
            }
        }
        self.ensure_batch_bufs()?;
        // 핀드 행/pos 기입(그래프 h2d 노드가 replay 시점에 읽는다).
        unsafe {
            std::ptr::copy_nonoverlapping(
                rows.as_ptr() as *const u8,
                self.pin_batch_in as *mut u8,
                t * self.hidden * 4,
            );
            let pp = self.pin_batch_pos as *mut u8;
            for s in 0..self.n_slots {
                let pos = self.slot_pos[s];
                std::ptr::copy_nonoverlapping(pos.to_le_bytes().as_ptr(), pp.add(s * 4), 4);
            }
        }
        let key: Vec<usize> = slots.to_vec();
        if let Some(k) = self.batch_graphs.iter().position(|(ks, _, _)| *ks == key) {
            let exec = self.batch_graphs[k].1;
            self.cc.graph_launch(exec)?;
            self.cc.sync()?;
            let out = self.batch_read_tokens(t);
            for &s in slots {
                self.slot_pos[s] += 1;
            }
            return Ok(out);
        }
        // 미스: 미캡처 실발사(정답) → 캡처(미실행)로 다음부터 replay.
        self.batch_launch(slots, t)?;
        self.cc.sync()?;
        let out = self.batch_read_tokens(t);
        if !self.batch_capture_failed {
            match self.batch_capture(&key, t) {
                Ok(()) => {}
                Err(e) => {
                    eprintln!("# batch 그래프 캡처 실패 — 직접 발사 유지: {e}");
                    self.batch_capture_failed = true;
                    self.cc.sync().ok();
                }
            }
        }
        for &s in slots {
            self.slot_pos[s] += 1;
        }
        Ok(out)
    }

    /// 배치 체인(t∈2..=8) — 프리필 청크. GEMM(t≥2) 경로 + 배치 버퍼.
    /// 반환: 마지막 행의 xn(또는 head면 로짓). t≥2 GEMM은 w4a16-gemm
    /// 게이트가 비트 판정(행별 64레인·tree64 동일) — t=1 경로와 계약 동일.
    fn chain_device_t(
        &mut self,
        slot: usize,
        rows: &[f32],
        t: usize,
        head: bool,
    ) -> Result<Vec<f32>, String> {
        let _g = self.cc.guard()?;
        if !(2..=CHAIN_TMAX).contains(&t) || rows.len() != t * self.hidden || slot >= self.n_slots {
            return Err(format!(
                "chain_device_t: t={t} rows={} 계약 위반",
                rows.len()
            ));
        }
        let pos = self.slot_pos[slot];
        let cap = self.attn.ok_or("attn: 형상 미등록")?.cap;
        if pos as usize + t > cap {
            return Err(format!("context overflow: pos{pos}+T{t} > kvcap{cap}"));
        }
        if self.norm_w_rows < 2 * self.n_layers + 1 || self.gdn.is_none() {
            return Err("chain_device_t: 디코더 상수 미등록".into());
        }
        self.ensure_chain_bufs()?;
        self.ensure_norm_bufs(t)?;
        self.ensure_gdn_bufs(t)?;
        self.ensure_attn_bufs(t)?;
        let dyt = self.ensure_dyt(t)?;
        let rb = unsafe { std::slice::from_raw_parts(rows.as_ptr() as *const u8, rows.len() * 4) };
        self.cc.h2d_async(self.dres, rb)?;
        let [s0, s1, s1b, s2, s3] = self.dchain;
        let (w2, h) = (self.stg_w2, self.hidden);
        let mut ab = self.dab_dev;
        let mut gi = 0usize;
        // 플레인(MoE) 모드 — bf16 GEMM(x 원시 f32), FFN은 MoE 배치.
        let plain = self.plain_weights;
        // [P10] GEMM 상한 — mma 경로(TC ON·t≥16)는 t 무제한, FFMA 폴백
        // 커널(G4_GTMAX/G4_TMAX2=32)만 32 상한. 폴백으로 t>32를 태우지 않는다.
        let gemm_mma = t >= 16 && llm170_diag::flag::ne0("LLM170_TC");
        if t > 32 && !gemm_mma {
            return Err(format!(
                "chain_device_t: t={t} > 32 — FFMA 폴백 상한(TC=0 진단 또는 t<16)"
            ));
        }
        for il in 0..self.n_layers {
            let xh = if plain { 0 } else { self.ensure_dx32(t * h)? };
            let xn = self
                .norm_resid_dev(2 * il, self.dres, ab, t, xh)
                .map_err(|e| format!("T{il} input norm: {e}"))?;
            let interval = self.attn.ok_or("attn: 형상 미등록")?.interval;
            let branch = if (il + 1) % interval == 0 {
                if plain {
                    self.plain_gemm_launch(&format!("blk.{il}.attn_q.weight"), xn, s0, t)?;
                    self.plain_gemm_launch(&format!("blk.{il}.attn_k.weight"), xn, s1, t)?;
                    self.plain_gemm_launch(&format!("blk.{il}.attn_v.weight"), xn, s1b, t)?;
                } else {
                    self.gemm_launch(&format!("blk.{il}.attn_q.weight"), xh, s0, t)?;
                    self.gemm_launch(&format!("blk.{il}.attn_k.weight"), xh, s1, t)?;
                    self.gemm_launch(&format!("blk.{il}.attn_v.weight"), xh, s1b, t)?;
                }
                self.attn_chain_dev_run(slot, il / interval, t, s0, s1, s1b)
                    .map_err(|e| format!("T{il} attn: {e}"))?
            } else {
                if plain {
                    self.plain_gemm_launch(&format!("blk.{il}.attn_qkv.weight"), xn, s0, t)?;
                    self.plain_gemm_launch(&format!("blk.{il}.attn_gate.weight"), xn, s1, t)?;
                } else {
                    // [2026-10-09 P6] xh(=self.dx32)는 norm_resid_dev가 이미
                    // h2f(f2h(xn)) 융합 기록(norm.cu xn32 — cast_x32와 비트 동일
                    // 계약, 실측 근거 주석 포함). 종전 cast_x32 재계산은 중복
                    // 런치였다. 값 불변(골든 검증).
                    self.gemm_launch(&format!("blk.{il}.attn_qkv.weight"), xh, s0, t)?;
                    self.gemm_launch(&format!("blk.{il}.attn_gate.weight"), xh, s1, t)?;
                }
                let g = self
                    .gdn_chain_dev_run(slot, gi, t, xn, s0, s1)
                    .map_err(|e| format!("T{il} gdn: {e}"))?;
                gi += 1;
                g
            };
            let lo = if (il + 1) % interval == 0 {
                format!("blk.{il}.attn_output.weight")
            } else {
                format!("blk.{il}.ssm_out.weight")
            };
            if plain {
                self.plain_gemm_launch(&lo, branch, dyt, t)?;
            } else {
                let (_, _, _, ko) = self.lin_spec(&lo)?;
                let xh2 = self.cast_x32(branch, t * ko)?;
                self.gemm_launch(&lo, xh2, dyt, t)?;
            }
            // [P10] cast_x32(t×ko)가 dx32를 재할당했을 수 있다 — 노름 융합
            // 기록(xn32)은 **현재** 포인터를 다시 확인한다. 옛 포인터를 계속
            // 쓰면 비행 커널이 해제 버퍼에 기록한다(새니타이저 OOB 실측).
            let xh = self.ensure_dx32(t * h)?;
            let xn2 = self
                .norm_resid_dev(2 * il + 1, self.dres, dyt, t, xh)
                .map_err(|e| format!("T{il} post norm: {e}"))?;
            if plain {
                ab = self
                    .moe_ffn_dev_t(il, xn2, t)
                    .map_err(|e| format!("T{il} moe: {e}"))?;
            } else {
                let xh3 = xh; // 노름 융합 기록
                let _ = xn2;
                self.gemm_launch(&format!("blk.{il}.ffn_gate.weight"), xh3, s0, t)?;
                self.gemm_launch(&format!("blk.{il}.ffn_up.weight"), xh3, s1, t)?;
                self.ew_dev(s0, s1, s2, t * w2)?;
                let dn = format!("blk.{il}.ffn_down.weight");
                let (_, _, _, kd) = self.lin_spec(&dn)?;
                let xh4 = self.cast_x32(s2, t * kd)?;
                self.gemm_launch(&dn, xh4, s3, t)?;
                ab = s3;
            }
        }
        // 마지막 행만 최종 노름(+head) — 중간 행 로짓은 불필요(상각).
        let last = (t - 1) as u64 * (h as u64) * 4;
        let xn_last = self
            .norm_resid_dev(2 * self.n_layers, self.dres + last, ab + last, 1, 0)
            .map_err(|e| format!("T final norm: {e}"))?;
        let mut ob = vec![0u8; if head { self.head_n * 4 } else { h * 4 }];
        if head {
            if self.head_w == 0 {
                return Err("chain_device_t: head 미등록".into());
            }
            let f = self.cc.function("head_bf16")?;
            let (mut p_w, mut p_x, mut p_o) = (self.head_w, xn_last, self.head_out);
            let (mut p_n, mut p_k) = (self.head_n as i32, self.head_k as i32);
            let mut args: [*mut std::ffi::c_void; 5] = [
                (&mut p_w) as *mut _ as *mut _,
                (&mut p_x) as *mut _ as *mut _,
                (&mut p_o) as *mut _ as *mut _,
                (&mut p_n) as *mut _ as *mut _,
                (&mut p_k) as *mut _ as *mut _,
            ];
            self.cc
                .launch(f, self.head_n.div_ceil(4 * 256) as u32, 1, 256, &mut args)?;
            // 동기 d2h 금지 — 그래프 캡처가 만든 커스텀(비차단) 스트림과
            // 경합한다(실측: serve 배치 프리필 쓰레기 토큰). 스트림 순서 복사.
            self.cc
                .d2h_async(ob.as_mut_ptr(), self.head_out, self.head_n * 4)?;
        } else {
            self.cc.d2h_async(ob.as_mut_ptr(), xn_last, h * 4)?;
        }
        self.cc.sync()?;
        let pos_after = pos + t as u32;
        self.slot_pos[slot] = pos_after;
        self.attn_set_pos(slot, pos_after)?;
        let v: Vec<f32> = ob
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        if head {
            llm170_diag::fp::fp_record("gpu.logits", &v);
        } else {
            llm170_diag::fp::fp_record("gpu.xn", &v);
        }
        Ok(v)
    }

    /// [A9 2026-10-10] 배치 디코드 단일 체인 — 슬롯별 1토큰(t=n_active)을
    /// 한 번에 통과한다. 가중 커널(GEMM·MoE·head)은 t행 배치로 상각하고,
    /// 상태 커널(attn KV/pos·GDN 링/트리오)은 토큰별 슬롯 자원을 쓴다.
    /// 반환: 슬롯 순서의 다음 토큰(argmax, greedy 전용).
    /// 전제(호출부 가드): t∈2..=BATCH_DEC_MAX · 전 슬롯 greedy · !kvq ·
    /// MoE면 상주 모드 · head/argmax 등록. 그래프 밖(직접 발사) 경로.
    /// [A9] 배치 체인 1회 발사 — 행/pos는 핀드(pin_batch_in/pos)에서 읽어
    /// h2d 노드로 기록된다(캡처 가능). sync·slot_pos 갱신은 호출부 소관.
    fn batch_launch(&mut self, slots: &[usize], t: usize) -> Result<(), String> {
        if self.norm_w_rows < 2 * self.n_layers + 1 || self.gdn.is_none() {
            return Err("batch_launch: 디코더 상수 미등록".into());
        }
        self.ensure_chain_bufs()?;
        self.ensure_norm_bufs(t)?;
        self.ensure_gdn_bufs(t)?;
        self.ensure_attn_bufs(t)?;
        self.ensure_batch_bufs()?;
        let dyt = self.ensure_dyt(t)?;
        {
            let rb = unsafe {
                std::slice::from_raw_parts(self.pin_batch_in as *const u8, t * self.hidden * 4)
            };
            self.cc.h2d_async(self.dres, rb)?;
            let pb = unsafe {
                std::slice::from_raw_parts(self.pin_batch_pos as *const u8, self.n_slots * 4)
            };
            self.cc.h2d_async(self.dpp, pb)?;
        }
        let [s0, s1, s1b, s2, s3] = self.dchain;
        let (w2, h) = (self.stg_w2, self.hidden);
        let mut ab = self.dab_dev;
        let mut gi = 0usize;
        let plain = self.plain_weights;
        if t > 32 {
            return Err(format!("chain_device_batch: t={t} > 32 FFMA 상한"));
        }
        for il in 0..self.n_layers {
            let xh = if plain { 0 } else { self.ensure_dx32(t * h)? };
            let xn = self
                .norm_resid_dev(2 * il, self.dres, ab, t, xh)
                .map_err(|e| format!("B{il} input norm: {e}"))?;
            let interval = self.attn.ok_or("attn: 형상 미등록")?.interval;
            let branch = if (il + 1) % interval == 0 {
                if plain {
                    self.plain_gemv_t_launch(&format!("blk.{il}.attn_q.weight"), xn, s0, t)?;
                    self.plain_gemv_t_launch(&format!("blk.{il}.attn_k.weight"), xn, s1, t)?;
                    self.plain_gemv_t_launch(&format!("blk.{il}.attn_v.weight"), xn, s1b, t)?;
                } else {
                    self.gemv_t_launch(&format!("blk.{il}.attn_q.weight"), xh, s0, t)?;
                    self.gemv_t_launch(&format!("blk.{il}.attn_k.weight"), xh, s1, t)?;
                    self.gemv_t_launch(&format!("blk.{il}.attn_v.weight"), xh, s1b, t)?;
                }
                self.attn_chain_dev_batch(slots, il / interval, t, s0, s1, s1b)
                    .map_err(|e| format!("B{il} attn: {e}"))?
            } else {
                if plain {
                    self.plain_gemv_t_launch(&format!("blk.{il}.attn_qkv.weight"), xn, s0, t)?;
                    self.plain_gemv_t_launch(&format!("blk.{il}.attn_gate.weight"), xn, s1, t)?;
                } else {
                    self.gemv_t_launch(&format!("blk.{il}.attn_qkv.weight"), xh, s0, t)?;
                    self.gemv_t_launch(&format!("blk.{il}.attn_gate.weight"), xh, s1, t)?;
                }
                let g = self
                    .gdn_chain_dev_batch(slots, gi, t, xn, s0, s1)
                    .map_err(|e| format!("B{il} gdn: {e}"))?;
                gi += 1;
                g
            };
            let lo = if (il + 1) % interval == 0 {
                format!("blk.{il}.attn_output.weight")
            } else {
                format!("blk.{il}.ssm_out.weight")
            };
            if plain {
                self.plain_gemv_t_launch(&lo, branch, dyt, t)?;
            } else {
                let (_, _, _, ko) = self.lin_spec(&lo)?;
                let xh2 = self.cast_x32(branch, t * ko)?;
                self.gemv_t_launch(&lo, xh2, dyt, t)?;
            }
            let xh = self.ensure_dx32(t * h)?;
            let xn2 = self
                .norm_resid_dev(2 * il + 1, self.dres, dyt, t, xh)
                .map_err(|e| format!("B{il} post norm: {e}"))?;
            if plain {
                ab = self
                    .moe_ffn_dev_t(il, xn2, t)
                    .map_err(|e| format!("B{il} moe: {e}"))?;
            } else {
                let xh3 = xh;
                let _ = xn2;
                self.gemv_t_launch(&format!("blk.{il}.ffn_gate.weight"), xh3, s0, t)?;
                self.gemv_t_launch(&format!("blk.{il}.ffn_up.weight"), xh3, s1, t)?;
                self.ew_dev(s0, s1, s2, t * w2)?;
                let dn = format!("blk.{il}.ffn_down.weight");
                let (_, _, _, kd) = self.lin_spec(&dn)?;
                let xh4 = self.cast_x32(s2, t * kd)?;
                self.gemv_t_launch(&dn, xh4, s3, t)?;
                ab = s3;
            }
        }
        // 전 행 최종 노름 → 배치 head(가중 판독 상각) → 배치 argmax → 1회 d2h.
        let xn_all = self
            .norm_resid_dev(2 * self.n_layers, self.dres, ab, t, 0)
            .map_err(|e| format!("B final norm: {e}"))?;
        {
            let f = self.cc.function("head_bf16_t")?;
            let (mut p_w, mut p_x, mut p_o) = (self.head_w, xn_all, self.dbatch_lg);
            let (mut p_n, mut p_k, mut p_t) = (self.head_n as i32, self.head_k as i32, t as i32);
            let mut args: [*mut std::ffi::c_void; 6] = [
                (&mut p_w) as *mut _ as *mut _,
                (&mut p_x) as *mut _ as *mut _,
                (&mut p_o) as *mut _ as *mut _,
                (&mut p_n) as *mut _ as *mut _,
                (&mut p_k) as *mut _ as *mut _,
                (&mut p_t) as *mut _ as *mut _,
            ];
            self.cc
                .launch(f, self.head_n.div_ceil(4 * 256) as u32, 1, 256, &mut args)?;
        }
        {
            let fa = self.cc.function("w4a16_argmax_min_t")?;
            let (mut p_l, mut p_n, mut p_t, mut p_a) =
                (self.dbatch_lg, self.head_n as i32, t as i32, self.dbatch_am);
            let mut aa: [*mut std::ffi::c_void; 4] = [
                (&mut p_l) as *mut _ as *mut _,
                (&mut p_n) as *mut _ as *mut _,
                (&mut p_t) as *mut _ as *mut _,
                (&mut p_a) as *mut _ as *mut _,
            ];
            self.cc.launch(fa, t as u32, 1, 1024, &mut aa)?;
        }
        let tb = unsafe { std::slice::from_raw_parts_mut(self.pin_batch_tok as *mut u8, t * 4) };
        self.cc.d2h_async(tb.as_mut_ptr(), self.dbatch_am, t * 4)?;
        Ok(())
    }

    /// [A9] 핀드 토큰 판독(sync 후).
    fn batch_read_tokens(&self, t: usize) -> Vec<u32> {
        let tb = unsafe { std::slice::from_raw_parts(self.pin_batch_tok as *const u8, t * 4) };
        (0..t)
            .map(|k| u32::from_le_bytes([tb[k * 4], tb[k * 4 + 1], tb[k * 4 + 2], tb[k * 4 + 3]]))
            .collect()
    }

    /// [A9] 배치 그래프 캡처 — 슬롯집합 키. 캡처는 실행하지 않는다(상태 불변).
    fn batch_capture(&mut self, key: &[usize], t: usize) -> Result<(), String> {
        if self.batch_graphs.len() >= 8 {
            let (_, e, g) = self.batch_graphs.remove(0);
            let _ = self.cc.graph_destroy(e, g);
        }
        self.cc.capture_begin()?;
        if let Err(e) = self.batch_launch(key, t) {
            let _ = self.cc.capture_end();
            return Err(e);
        }
        let g = self.cc.capture_end()?;
        let e = self.cc.graph_instantiate(g)?;
        self.batch_graphs.push((key.to_vec(), e, g));
        eprintln!("[batch-graph] captured slots={key:?}");
        Ok(())
    }

    /// [A9] 배치 디코드 부속 버퍼 — 로짓[t×head_n]·argmax[t]·핀드 입출력.
    fn ensure_batch_bufs(&mut self) -> Result<(), String> {
        if self.head_n == 0 {
            return Err("batch: head 미등록(업로드 선행)".into());
        }
        let tmax = self.n_slots.min(BATCH_DEC_MAX);
        if self.dbatch_lg == 0 {
            self.dbatch_lg = self.cc.alloc(tmax * self.head_n * 4)?;
            self.dbatch_am = self.cc.alloc(tmax * 4)?;
        }
        if self.pin_batch_tok.is_null() {
            self.pin_batch_tok = self.cc.pinned_alloc(tmax * 4)?;
            self.pin_batch_in = self.cc.pinned_alloc(tmax * self.hidden * 4)?;
            self.pin_batch_pos = self.cc.pinned_alloc(self.n_slots * 4)?;
        }
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

    /// GEMV 전 선형 1회 순회(벤치) — 실사용과 동일한 DRAM 스트림(13.9GB ≫ L2).
    /// 반환: (ms/회, 가중치 바이트 합). x는 k별 1.0 f32(수치 무의미).
    pub fn gemv_walk_bench(&mut self, reps: usize) -> Result<(f64, u64), String> {
        let _g = self.cc.guard()?;
        let mut mk = 0usize;
        let mut mn = 0usize;
        let mut wb = 0u64;
        for &(_, _, n, k) in self.lins.values() {
            mk = mk.max(k);
            mn = mn.max(n);
            wb += (n * k / 2) as u64;
        }
        let xf: Vec<f32> = vec![1.0f32; mk];
        let xb = unsafe { std::slice::from_raw_parts(xf.as_ptr() as *const u8, xf.len() * 4) };
        let dx = self.cc.alloc(xf.len() * 4)?;
        self.cc.h2d(dx, xb)?;
        let dy = self.cc.alloc(mn * 4)?;
        let names: Vec<String> = self.lins.keys().cloned().collect();
        let walk = |me: &mut Self| -> Result<(), String> {
            for name in &names {
                me.gemv_launch(name, dx, dy)?;
            }
            Ok(())
        };
        walk(self)?;
        self.cc.sync()?;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            walk(self)?;
        }
        self.cc.sync()?;
        let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
        self.cc.free(dx)?;
        self.cc.free(dy)?;
        Ok((ms, wb))
    }

    /// mma GEMM 수치 게이트(허용오차) — split 원본(w4a16_gemm_g128) vs
    /// w4a16_gemm_g128_mma 출력을 같은 x로 비교. (T1 검증 — 계약 완화 후
    /// 비트 대신 허용오차 판정.)
    pub fn mma_diff_check(&mut self, name: &str, t: usize) -> Result<String, String> {
        let _g = self.cc.guard()?;
        // 플레인(bf16)이면 T2 경로로.
        if !self.lins.contains_key(name) {
            return self.mma_diff_plain(name, t);
        }
        let (dq, ds, n, k) = self.lin_spec(name)?;
        let xf: Vec<f32> = (0..t * k)
            .map(|i| ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as f32 / 4096.0 - 0.5)
            .collect();
        let xb = unsafe { std::slice::from_raw_parts(xf.as_ptr() as *const u8, xf.len() * 4) };
        let dx = self.cc.alloc(t * k * 4)?;
        self.cc.h2d(dx, xb)?;
        let da = self.cc.alloc(t * n * 4)?;
        let db = self.cc.alloc(t * n * 4)?;
        let r = (|| -> Result<(Vec<f32>, Vec<f32>), String> {
            // 원본(8행/블록, 512스레드).
            let f = self.cc.function("w4a16_gemm_g128")?;
            let (mut p_q, mut p_s, mut p_x, mut p_y) = (dq, ds, dx, da);
            let (mut p_n, mut p_k, mut p_t) = (n as i32, k as i32, t as i32);
            let mut a1: [*mut std::ffi::c_void; 7] = [
                (&mut p_q) as *mut _ as *mut _,
                (&mut p_s) as *mut _ as *mut _,
                (&mut p_x) as *mut _ as *mut _,
                (&mut p_y) as *mut _ as *mut _,
                (&mut p_n) as *mut _ as *mut _,
                (&mut p_k) as *mut _ as *mut _,
                (&mut p_t) as *mut _ as *mut _,
            ];
            self.cc.launch(f, n.div_ceil(8) as u32, 1, 512, &mut a1)?;
            // mma(32×64, 256스레드).
            let fm = self.cc.function("w4a16_gemm_g128_mma")?;
            let (mut m_q, mut m_s, mut m_x, mut m_y) = (dq, ds, dx, db);
            let (mut m_n, mut m_k, mut m_t) = (n as i32, k as i32, t as i32);
            let mut a2: [*mut std::ffi::c_void; 7] = [
                (&mut m_q) as *mut _ as *mut _,
                (&mut m_s) as *mut _ as *mut _,
                (&mut m_x) as *mut _ as *mut _,
                (&mut m_y) as *mut _ as *mut _,
                (&mut m_n) as *mut _ as *mut _,
                (&mut m_k) as *mut _ as *mut _,
                (&mut m_t) as *mut _ as *mut _,
            ];
            self.cc.launch(
                fm,
                t.div_ceil(32) as u32,
                n.div_ceil(32) as u32,
                256,
                &mut a2,
            )?;
            self.cc.sync()?;
            let mut oa = vec![0u8; t * n * 4];
            let mut ob = vec![0u8; t * n * 4];
            self.cc.d2h(&mut oa, da)?;
            self.cc.d2h(&mut ob, db)?;
            let fa: Vec<f32> = oa
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            let fb: Vec<f32> = ob
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            Ok((fa, fb))
        })();
        let _ = self.cc.free(dx);
        let _ = self.cc.free(da);
        let _ = self.cc.free(db);
        let (fa, fb) = r?;
        let mut maxd = 0f32;
        let mut maxr = 0f32;
        let mut bad = 0usize;
        for i in 0..fa.len() {
            let d = (fa[i] - fb[i]).abs();
            if d > maxd {
                maxd = d;
            }
            let rl = d / fa[i].abs().max(1e-6);
            if rl > maxr {
                maxr = rl;
            }
            if !fb[i].is_finite() {
                bad += 1;
            }
        }
        // 불량 위치 패턴(첫 5개 + 행·열 분포) — 디버그.
        let mut firstbad: Vec<(usize, usize)> = Vec::new();
        let mut badrows = std::collections::BTreeSet::new();
        let mut badcols = std::collections::BTreeSet::new();
        for r in 0..t {
            for c in 0..n {
                let v = fb[r * n + c];
                if !v.is_finite() || (v - fa[r * n + c]).abs() > 1e-2 {
                    if firstbad.len() < 5 {
                        firstbad.push((r, c));
                    }
                    badrows.insert(r);
                    badcols.insert(c);
                }
            }
        }
        Ok(format!(
            "mma-diff {name} n={n} k={k} t={t}: maxabs={maxd:.3e} 비유한={bad}/{} 첫불량={:?} 불량행={} 불량열={} A[0..3]={:?} B[0..3]={:?}",
            fa.len(),
            firstbad,
            badrows.len(),
            badcols.len(),
            &fa[..3],
            &fb[..3]
        ))
    }

    /// [진단] ew 커널 독립 벤치 — 순수 커널 처리량(파이프라인 무관).
    pub fn bench_ew(&mut self) -> Result<String, String> {
        let _g = self.cc.guard()?;
        let n = 245_760usize;
        let dg = self.cc.alloc(n * 4)?;
        let du = self.cc.alloc(n * 4)?;
        let dy = self.cc.alloc(n * 4)?;
        let v: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.1 - 0.3).collect();
        let b = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n * 4) };
        self.cc.h2d(dg, b)?;
        self.cc.h2d(du, b)?;
        let f = self.cc.function("ew")?;
        let (mut a0, mut a1, mut a2) = (dg, du, dy);
        let mut nn = n as i32;
        let mut args: [*mut std::ffi::c_void; 4] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
            (&mut nn) as *mut _ as *mut _,
        ];
        for _ in 0..10 {
            self.cc
                .launch(f, n.div_ceil(128) as u32, 1, 128, &mut args)?;
        }
        self.cc.sync()?;
        let t0 = std::time::Instant::now();
        for _ in 0..100 {
            self.cc
                .launch(f, n.div_ceil(128) as u32, 1, 128, &mut args)?;
        }
        self.cc.sync()?;
        let el = t0.elapsed().as_secs_f64() / 100.0;
        let _ = self.cc.free(dg);
        let _ = self.cc.free(du);
        let _ = self.cc.free(dy);
        Ok(format!(
            "ew bench n={n}: {:.3}ms/launch · {:.1} GB/s (12B/elem)",
            el * 1000.0,
            (n as f64 * 12.0) / el / 1e9
        ))
    }

    /// MoE top-k 게이트 — 호스트 moe_topk vs 디바이스 w4a16_moe_topk_t.
    pub fn moe_topk_check(&mut self, t: usize, n: usize, k: usize) -> Result<String, String> {
        let _g = self.cc.guard()?;
        let lg: Vec<f32> = (0..t * n)
            .map(|i| ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as f32 / 2048.0 - 0.5)
            .collect();
        let lb = unsafe { std::slice::from_raw_parts(lg.as_ptr() as *const u8, lg.len() * 4) };
        let dl = self.cc.alloc(t * n * 4)?;
        let di = self.cc.alloc(t * k * 4)?;
        let dw = self.cc.alloc(t * k * 4)?;
        self.cc.h2d(dl, lb)?;
        let f = self.cc.function("w4a16_moe_topk_t")?;
        let (mut p_lg, mut p_ix, mut p_wt) = (dl, di, dw);
        let (mut p_t, mut p_n, mut p_k) = (t as i32, n as i32, k as i32);
        let mut a: [*mut std::ffi::c_void; 6] = [
            (&mut p_lg) as *mut _ as *mut _,
            (&mut p_ix) as *mut _ as *mut _,
            (&mut p_wt) as *mut _ as *mut _,
            (&mut p_t) as *mut _ as *mut _,
            (&mut p_n) as *mut _ as *mut _,
            (&mut p_k) as *mut _ as *mut _,
        ];
        self.cc.launch(f, t.div_ceil(8) as u32, 1, 256, &mut a)?;
        self.cc.sync()?;
        let mut ib = vec![0u8; t * k * 4];
        let mut wb = vec![0u8; t * k * 4];
        self.cc.d2h(&mut ib, di)?;
        self.cc.d2h(&mut wb, dw)?;
        let gi: Vec<u32> = ib
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        let gw: Vec<f32> = wb
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let _ = self.cc.free(dl);
        let _ = self.cc.free(di);
        let _ = self.cc.free(dw);
        let mut idx_bad = 0usize;
        let mut wmax = 0f32;
        for ti in 0..t {
            let host = moe_topk(&lg[ti * n..(ti + 1) * n], k);
            for (r, (e, w)) in host.iter().enumerate() {
                if gi[ti * k + r] as usize != *e {
                    idx_bad += 1;
                }
                let d = (gw[ti * k + r] - w).abs();
                if d > wmax {
                    wmax = d;
                }
            }
        }
        Ok(format!(
            "moe-topk-check t={t} n={n} k={k}: idx 불일치 {idx_bad}/{} · |Δw|max={wmax:.3e} · dev[0..4]={:?} host[0..4]={:?}",
            t * k,
            &gi[..4],
            &moe_topk(&lg[..n], k)
                .iter()
                .map(|&(e, _)| e as u32)
                .collect::<Vec<_>>()[..4]
        ))
    }

    /// 플레인 mma 수치 게이트 — v3(원본) vs bf16 mma.
    fn mma_diff_plain(&mut self, name: &str, t: usize) -> Result<String, String> {
        let (w, n, k) = self.plain_spec(name)?;
        let xf: Vec<f32> = (0..t * k)
            .map(|i| ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as f32 / 4096.0 - 0.5)
            .collect();
        let xb = unsafe { std::slice::from_raw_parts(xf.as_ptr() as *const u8, xf.len() * 4) };
        let dx = self.cc.alloc(t * k * 4)?;
        self.cc.h2d(dx, xb)?;
        let da = self.cc.alloc(t * n * 4)?;
        let db = self.cc.alloc(t * n * 4)?;
        let r = (|| -> Result<(Vec<f32>, Vec<f32>), String> {
            let f = self.cc.function("w4a16_gemm_bf16_t")?;
            let (mut p_w, mut p_x, mut p_o) = (w, dx, da);
            let (mut p_n, mut p_k, mut p_t) = (n as i32, k as i32, t as i32);
            let mut a1: [*mut std::ffi::c_void; 6] = [
                (&mut p_w) as *mut _ as *mut _,
                (&mut p_x) as *mut _ as *mut _,
                (&mut p_o) as *mut _ as *mut _,
                (&mut p_n) as *mut _ as *mut _,
                (&mut p_k) as *mut _ as *mut _,
                (&mut p_t) as *mut _ as *mut _,
            ];
            self.cc.launch(f, n.div_ceil(8) as u32, 1, 512, &mut a1)?;
            let fm = self.cc.function("w4a16_gemm_bf16_mma")?;
            let (mut m_w, mut m_x, mut m_o) = (w, dx, db);
            let (mut m_n, mut m_k, mut m_t) = (n as i32, k as i32, t as i32);
            let mut a2: [*mut std::ffi::c_void; 6] = [
                (&mut m_w) as *mut _ as *mut _,
                (&mut m_x) as *mut _ as *mut _,
                (&mut m_o) as *mut _ as *mut _,
                (&mut m_n) as *mut _ as *mut _,
                (&mut m_k) as *mut _ as *mut _,
                (&mut m_t) as *mut _ as *mut _,
            ];
            self.cc.launch(
                fm,
                t.div_ceil(32) as u32,
                n.div_ceil(32) as u32,
                256,
                &mut a2,
            )?;
            self.cc.sync()?;
            let mut oa = vec![0u8; t * n * 4];
            let mut ob = vec![0u8; t * n * 4];
            self.cc.d2h(&mut oa, da)?;
            self.cc.d2h(&mut ob, db)?;
            let fa: Vec<f32> = oa
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            let fb: Vec<f32> = ob
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            Ok((fa, fb))
        })();
        let _ = self.cc.free(dx);
        let _ = self.cc.free(da);
        let _ = self.cc.free(db);
        let (fa, fb) = r?;
        let mut maxd = 0f32;
        let mut bad = 0usize;
        for i in 0..fa.len() {
            let d = (fa[i] - fb[i]).abs();
            if d > maxd {
                maxd = d;
            }
            if !fb[i].is_finite() {
                bad += 1;
            }
        }
        Ok(format!(
            "mma-diff(plain) {name} n={n} k={k} t={t}: maxabs={maxd:.3e} 비유한={bad}/{} A[0..3]={:?} B[0..3]={:?}",
            fa.len(),
            &fa[..3],
            &fb[..3]
        ))
    }

    /// 플레인 GEMM 단발 발사(벤치·진단 — T2 mma 비교용).
    pub fn plain_bench_launch(
        &self,
        name: &str,
        x: CUdeviceptr,
        y: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        self.plain_gemm_launch(name, x, y, t)
    }

    /// GEMV 단발 발사(벤치·진단 — import 없이 이름·x·y만).
    pub fn gemv_bench_launch(
        &mut self,
        name: &str,
        x: CUdeviceptr,
        y: CUdeviceptr,
    ) -> Result<(), String> {
        self.gemv_launch(name, x, y)
    }

    pub fn gemm_bench_launch(
        &mut self,
        name: &str,
        xh: CUdeviceptr,
        y: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        self.gemm_launch(name, xh, y, t)
    }

    /// 프리필 청크 진입(공개) — t=1은 기존 경로(그래프 포함), t≥2는 배치 체인.
    pub fn forward_prefill(
        &mut self,
        slot: usize,
        rows: &[f32],
        t: usize,
        head: bool,
    ) -> Result<Vec<f32>, String> {
        if t == 1 {
            let row = &rows[..self.hidden];
            return if head {
                self.forward_device_head(slot, row)
            } else {
                self.forward_device(slot, row)
            };
        }
        self.chain_device_t(slot, rows, t, head)
    }

    /// 1토큰 forward(디바이스 체인) — 최종 xn까지 판독(CPU head용).
    /// guard는 래퍼 전체를 덮는다 — 체인 이후의 d2h/h2d도 같은 스레드
    /// 컨텍스트가 필요하다(CtxGuard는 드랍 시 이전 컨텍스트로 복원).
    pub fn forward_device(&mut self, slot: usize, embed_row: &[f32]) -> Result<Vec<f32>, String> {
        let _g = self.cc.guard()?;
        if self.graph_ok() {
            match self
                .ensure_graph(slot, false, false)
                .and_then(|exec| self.graph_replay(exec, slot, embed_row))
            {
                Ok(xn) => {
                    llm170_diag::fp::fp_record("gpu.xn", &xn);
                    return Ok(xn);
                }
                Err(e) => {
                    eprintln!("[graph] xn 경로 실패 — 직접 경로 폴백: {e}");
                    self.graph_failed = true;
                }
            }
        }
        let pos = self.slot_pos[slot];
        let xn_final = self.chain_device(slot, embed_row)?;
        let mut ob = vec![0u8; self.hidden * 4];
        self.cc
            .d2h_async(ob.as_mut_ptr(), xn_final, self.hidden * 4)?;
        self.cc.sync()?;
        self.slot_pos[slot] = pos + 1;
        self.attn_set_pos(slot, pos + 1)?;
        let xn =
            unsafe { std::slice::from_raw_parts(ob.as_ptr() as *const f32, self.hidden) }.to_vec();
        // S5: 지문 파이프라인 부활 — LLM170_FP_FILE 시 스테이지 해시 기록
        // (`diag diff`로 실행 2개의 최초 발산 스테이지 추적).
        llm170_diag::fp::fp_record("gpu.xn", &xn);
        Ok(xn)
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
        // 전치 업로드: 원본 n-major를 임시 버퍼로 올린 뒤 head_transpose로
        // [k][n] 상주 버퍼를 만든다(판독 응집 — 실측 근거는 assets/head.cu).
        let dtmp = self.cc.alloc(need)?;
        if let Err(e) = Self::h2d_chunked(&self.cc, dtmp, &data[..need]) {
            let _ = self.cc.free(dtmp);
            return Err(e);
        }
        let dw = self.cc.alloc(need)?;
        let tr = (|| -> Result<(), String> {
            let f = self.cc.function("head_transpose")?;
            let (mut p_in, mut p_out) = (dtmp, dw);
            let (mut p_n, mut p_k) = (n as i32, k as i32);
            let mut args: [*mut std::ffi::c_void; 4] = [
                (&mut p_in) as *mut _ as *mut _,
                (&mut p_out) as *mut _ as *mut _,
                (&mut p_n) as *mut _ as *mut _,
                (&mut p_k) as *mut _ as *mut _,
            ];
            self.cc.launch(
                f,
                k.div_ceil(32) as u32,
                n.div_ceil(32) as u32,
                1024,
                &mut args,
            )?;
            self.cc.sync()
        })();
        let _ = self.cc.free(dtmp); // 전치 완료 — 임시 해제(피크 VRAM 절감).
        if let Err(e) = tr {
            let _ = self.cc.free(dw);
            return Err(format!("head 전치: {e}"));
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

    /// 1토큰 forward + GPU head — xn 판독 없이 로짓만 회수(왕복 1회).
    pub fn forward_device_head(
        &mut self,
        slot: usize,
        embed_row: &[f32],
    ) -> Result<Vec<f32>, String> {
        if self.head_w == 0 {
            return Err("forward_device_head: head 미등록 — upload_head 선행".into());
        }
        let _g = self.cc.guard()?;
        if self.graph_ok() {
            match self
                .ensure_graph(slot, true, false)
                .and_then(|exec| self.graph_replay(exec, slot, embed_row))
            {
                Ok(lg) => {
                    llm170_diag::fp::fp_record("gpu.logits", &lg);
                    return Ok(lg);
                }
                Err(e) => {
                    eprintln!("[graph] head 경로 실패 — 직접 경로 폴백: {e}");
                    self.graph_failed = true;
                }
            }
        }
        let pos = self.slot_pos[slot];
        let xn = self.chain_device(slot, embed_row)?;
        let f = self.cc.function("head_bf16")?;
        let (mut p_w, mut p_x, mut p_o) = (self.head_w, xn, self.head_out);
        let (mut p_n, mut p_k) = (self.head_n as i32, self.head_k as i32);
        let mut args: [*mut std::ffi::c_void; 5] = [
            (&mut p_w) as *mut _ as *mut _,
            (&mut p_x) as *mut _ as *mut _,
            (&mut p_o) as *mut _ as *mut _,
            (&mut p_n) as *mut _ as *mut _,
            (&mut p_k) as *mut _ as *mut _,
        ];
        self.cc
            .launch(f, self.head_n.div_ceil(4 * 256) as u32, 1, 256, &mut args)?;
        let mut ob = vec![0u8; self.head_n * 4];
        self.cc
            .d2h_async(ob.as_mut_ptr(), self.head_out, self.head_n * 4)?;
        self.cc.sync()?;
        self.slot_pos[slot] = pos + 1;
        self.attn_set_pos(slot, pos + 1)?;
        let lg: Vec<f32> = ob
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        llm170_diag::fp::fp_record("gpu.logits", &lg);
        Ok(lg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};

    /// MoE 선택 시맨틱 — core stages/moe::select_topk 미러(변경 시 동시 갱신).
    #[test]
    fn moe_topk_mirrors_core_semantics() {
        // 재정규화(합 1)·내림차순·exp 비율 유지.
        let sel = moe_topk(&[2.0, 1.0, 0.5, -1.0], 2);
        assert_eq!(sel.len(), 2);
        assert_eq!(sel[0].0, 0);
        assert_eq!(sel[1].0, 1);
        assert!(sel[0].1 > sel[1].1);
        let sum: f32 = sel.iter().map(|&(_, w)| w).sum();
        assert!((sum - 1.0).abs() < 1e-6, "합={sum}");
        let ratio = sel[0].1 / sel[1].1;
        assert!((ratio - std::f32::consts::E).abs() < 1e-4, "비율={ratio}");
        // 동률은 낮은 인덱스(결정적).
        let tie = moe_topk(&[1.0, 1.0, 1.0], 2);
        assert_eq!(tie[0].0, 0);
        assert_eq!(tie[1].0, 1);
    }

    /// G1 회귀 — 부분 alloc 실패 후 재시도가 이중해제/유실 없이 전량 재할당.
    /// (CUDA 불필요 — free/alloc을 모의 클로저로 주입.)
    #[test]
    fn realloc_fields_partial_failure_retry_safe() {
        let mut f: [CUdeviceptr; 3] = [0x11, 0x22, 0x33];
        let live = RefCell::new(vec![0x11u64, 0x22, 0x33]);
        let frees = RefCell::new(Vec::<u64>::new());
        let next = Cell::new(0x100u64);
        let fail_next = Cell::new(1usize); // 두 번째 alloc에서 실패 주입
        let free = |p: CUdeviceptr| -> Result<(), String> {
            let mut l = live.borrow_mut();
            assert!(l.contains(&p), "이중해제 {p:#x}");
            l.retain(|&v| v != p);
            frees.borrow_mut().push(p);
            Ok(())
        };
        let alloc = |_sz: usize| -> Result<CUdeviceptr, String> {
            if fail_next.get() == 0 {
                return Err("inject".into());
            }
            fail_next.set(fail_next.get() - 1);
            let v = next.get() + 1;
            next.set(v);
            live.borrow_mut().push(v);
            Ok(v)
        };
        let [a, b, c] = &mut f;
        let r = realloc_fields(free, alloc, [a, b, c], [1, 1, 1]);
        assert!(r.is_err(), "주입 실패가 전파되어야 한다");
        assert_eq!(*frees.borrow(), vec![0x11, 0x22, 0x33]);
        assert_eq!(f, [0x101, 0, 0], "성공분 유지·실패분 0");
        // 재시도 — 성공분만 회수(이중해제 없음), 전량 재할당.
        fail_next.set(usize::MAX);
        let [a, b, c] = &mut f;
        realloc_fields(free, alloc, [a, b, c], [1, 1, 1]).expect("재시도");
        assert_eq!(*frees.borrow(), vec![0x11, 0x22, 0x33, 0x101]);
        assert_eq!(f, [0x102, 0x103, 0x104]);
        assert_eq!(live.borrow().len(), 3);
    }
}

/// H(2026-10-09): .cu가 진실 — 호스트 미러 상수를 정적으로 대조한다
/// (미러 불일치 = smem 오버런·버퍼 계약 위반이라 변경 시 사전 차단).
#[cfg(test)]
mod mirror_contract {
    use super::{ATTN_F3S_TMAX, CHAIN_TMAX, GDN_CS, GDN_NGRP, GDN_NSPLIT, GDN_SCAN_SMEM, GEMV_TR};

    /// `#define NAME 값` 파싱 — 값은 정수 리터럴만 다룬다(대상 목록 한정).
    fn define(src: &str, name: &str) -> u64 {
        for line in src.lines() {
            if let Some(rest) = line.trim().strip_prefix("#define ") {
                let mut it = rest.split_whitespace();
                if it.next() == Some(name) {
                    return it
                        .next()
                        .expect("define 값 누락")
                        .parse()
                        .expect("정수 define 아님");
                }
            }
        }
        panic!(".cu define 없음: {name}");
    }

    #[test]
    fn attn_tmax_mirror() {
        let tmax = define(include_str!("assets/attn.cu"), "ATTN_TMAX") as usize;
        assert_eq!(tmax, ATTN_F3S_TMAX, "attn.cu ATTN_TMAX ↔ ATTN_F3S_TMAX");
        assert_eq!(tmax, CHAIN_TMAX, "attn.cu ATTN_TMAX ↔ CHAIN_TMAX");
    }

    #[test]
    fn gdn_scan_layout_mirror() {
        let cu = include_str!("assets/gdn.cu");
        let cs = define(cu, "GDN_CS") as usize;
        let tile = define(cu, "GDN_TILE") as usize;
        let nsplit = define(cu, "GDN_NSPLIT") as usize;
        assert_eq!(cs, GDN_CS, "gdn.cu GDN_CS ↔ GDN_CS");
        assert_eq!(nsplit, GDN_NSPLIT, "gdn.cu GDN_NSPLIT ↔ GDN_NSPLIT");
        assert_eq!(
            define(cu, "GDN_NGRP") as usize,
            GDN_NGRP,
            "gdn.cu GDN_NGRP ↔ GDN_NGRP"
        );
        let vs = 128 / nsplit;
        // gdn_scan 레이아웃(sk/qs/sv/A/KQ/KS/QS/dc/Stile×2/bp/gcs/wsm) 바이트
        // 합 — A5-4(qs 스테이징·V-타일·더블 버퍼) 반영.
        let total = cs * 128 * 2
            + cs * 128 * 4
            + cs * vs * 2
            + cs * cs * 2
            + cs * cs * 2
            + cs * vs * 2
            + cs * vs * 2
            + cs * vs * 4
            + 2 * tile * vs * 4
            + cs * 4
            + (cs + 1) * 4
            + cs * 4;
        assert_eq!(total as u32, GDN_SCAN_SMEM, "gdn_scan 동적 smem 합");
    }

    #[test]
    fn g4_scmax_mirror() {
        let scmax = define(include_str!("assets/gptq4.cu"), "G4_SCMAX") as usize;
        assert_eq!(scmax, crate::rawcuda::gptq4::G4_SCMAX, "gptq4.cu G4_SCMAX");
    }

    #[test]
    fn gemv_tr_mirror() {
        let tr = define(include_str!("assets/gptq4.cu"), "GEMV_TR") as usize;
        assert_eq!(tr, GEMV_TR, "gptq4.cu GEMV_TR ↔ GEMV_TR");
    }
}
