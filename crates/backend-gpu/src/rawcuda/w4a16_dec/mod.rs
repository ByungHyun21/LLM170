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
use unified_cache::{CacheCategory, Tier, UnifiedCacheRegistry, base_util};

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
pub const GEMM_MMA_M: usize = 64; // [marlin-A4] 32→64: B 재판독 m타일 절반
pub const GEMM_MMA_N: usize = 128; // [marlin-A2] 64→128: A 재판독 절반(L2 바운드)
/// bf16 mma GEMM(플레인) 전용 타일 — 커널 BMMA_M/N 미러(워프 매핑 32×64 고정).
pub const GEMM_BMMA_M: usize = 32;
pub const GEMM_BMMA_N: usize = 64;
pub const GEMM_GRP_M: usize = 64;
pub const GEMM_GRP_N: usize = 32;
/// GDN scan 동적 공유메모리(assets/gdn.cu 계약 — 정적 48KB 초과).
/// gdn_scan 동적 공유메모리(커널 레이아웃 계약 — A5-4: qs 스테이징 +
/// V-타일(GDN_NSPLIT) + Stile 더블 버퍼 기준).
pub const GDN_SCAN_SMEM: u32 = 27_140; // [FLA-1] +e[CS] 128B (qs 제거 -16KB 유지)
/// [A5-4 2026-10-10 FLA 2단] gdn_scan V-타일 분할 수(커널 GDN_NSPLIT와
/// 동일 계약) — grid = h_v×NSPLIT, 블록 = GDN_NGRP×GDN_VS스레드.
pub const GDN_NSPLIT: usize = 4;
/// [A5-4] gdn_scan 워크그룹 수(커널 GDN_NGRP와 동일 계약).
pub const GDN_NGRP: usize = 16;
/// gdn_scan 청크 크기(커널 GDN_CS와 동일 계약) — prepass 그리드·스크래치 산정.
pub const GDN_CS: usize = 32;
/// [A9 2026-10-10] 배치 디코드 최대 토큰(커널 head.cu HEAD_TMAX 미러).
pub const BATCH_DEC_MAX: usize = 8;
/// [K-STEP 2026-10-12 §3.4 ①] K-스텝 디코드 로그 폭(커널 w4a16_kstep_ctrl
/// out[kmax] 계약). 서빙 k=4 — 상한 8(mtp 체인 로그와 동일 철학).
pub const KSTEP_KMAX: usize = 8;
/// [A9-fix2] t행 GEMV 블록당 행 수(커널 gptq4.cu GEMV_TR 미러) — x 재사용.
pub const GEMV_TR: usize = 8;
/// [R21] FFMA 폴백 GEMM의 t 상한(커널 gptq4.cu G4_GTMAX 미러) — 체인 가드용.
pub const GEMM_FFMA_TMAX: usize = 32;

// [KVQ 채택 2026-10-10] int8 KV 단일 경로 — KvMode 제거.
// 근거(실측): int8 PPL 델타 ±0.3% 이내(장문 27B +0.0025%) · 장문 프리필
// +54%@16K(42.7→27.7s) · 단문 동급 — f32/int4 경로는 삭제(docs/eval.md — 로컬 원장).

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
pub const ATTN_SPLITS: usize = 8; // 프리필(그리드 t×q_heads×8로 충분)
/// [§8.1 2026-10-11] 디코드(t≤8) 분할 수 — t=1 그리드 192블록(q_heads×8)이
/// 128SM에서 스케줄러당 3워프·무이슈 79.7%(ncu)라 분할 확대로 점유 회복.
pub const ATTN_SPLITS_D: usize = 32;
/// attn.cu 병합 smem(ATTN_SPLITS_MAX)과 동일해야 함(테스트 대조).
pub const ATTN_SPLITS_MAX: usize = 32;

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

/// [A8] 캡처 그래프 1개 — (slot, head, argmax)가 캐시 키.
pub(super) struct GraphEntry {
    exec: ffi::CUgraphExec,
    handle: ffi::CUgraph,
    slot: usize,
    head: bool,
    argmax: bool,
}

/// [R10 2026-10-12] MoE(전문가 라우팅·스트리밍 캐시) — W4a16Dec 필드 도메인 그룹(순수 이동).
pub(super) struct MoE {
    /// MoE 구성 — n_experts=0이면 dense FFN.
    pub(super) n_experts: usize,
    pub(super) top_k: usize,
    pub(super) moe_ffn: usize,
    pub(super) shared_ffn: usize,
    pub(super) moe_group: usize,
    pub(super) moe_scale_bf16: bool,
    /// 전문가 슬라이스 테이블 — (packed ptr/len, scale ptr/len), 인덱스
    /// ((il·n_experts + e)·3 + proj) — proj 0/1/2 = gate/up/down.
    /// 호스트 포인터는 서버 스토어 mmap 수명 계약(set_expert_table).
    pub(super) moe_tab: Vec<(u64, u64, u64, u64)>,
    /// 전문가 상주 모드 — moe_tab이 VRAM 포인터(170HX 64GB 등).
    pub(super) moe_resident: bool,
    /// 전문가 디바이스 포인터 테이블(상주 — 배치 GEMV 간접 참조, (q,s) 쌍).
    pub(super) moe_dev_tab: CUdeviceptr,
    /// 선택 슬롯 인덱스·가중(호스트 → 디바이스, [top_k]).
    pub(super) moe_idx: CUdeviceptr,
    pub(super) moe_wt: CUdeviceptr,
    /// [B-4] 다중 세그먼트 GEMV 테이블(층×3세그먼트×4u64: w,out,n,k) +
    /// 층별 행 합(그리드). 라우터 게이트 + shared gate/up 1런치용.
    pub(super) moe_multi_tab: CUdeviceptr,
    pub(super) moe_multi_rows: Vec<u32>,
    /// [S10-잔여] t행(≤8) 판 다중세그먼트 grid — 라우터+shared gate/up 3세그먼트
    /// Σceil(n/GEMV_TR)/층(sgate 제외 — 출력 레이아웃 [t][n] 충돌 회피).
    pub(super) moe_multi_rows_t3: Vec<u32>,
    /// [P11] 전문가-우선 슬롯 정렬(프리필 그룹 GEMV) — gslot[n_exp×gmax]+cnt.
    pub(super) moe_gslot: CUdeviceptr,
    pub(super) moe_gcnt: CUdeviceptr,
    pub(super) moe_goff: CUdeviceptr,
    pub(super) moe_gmax: usize,
    /// 배치 전문가 출력([top_k][n_ff] · [top_k][hidden]) — act는 ew 전용
    /// 별도 버퍼(ew 커널 __restrict__ 계약 — 제자리 호출 금지).
    pub(super) exp: ExpBufs,
    /// MoE 스테이징 — 전문가 packed/scale(1쌍 재사용, 스트림 순서 안전).
    pub(super) stg: StgBufs,
    /// [C2 2026-10-09] 호스트 RAM 스테이징 — 스트리밍 전문가의 mmap 슬라이스
    /// 1회 복사본(set_expert_table). 스트리밍 업로드가 SSD/페이지캐시 경로에
    /// 의존하지 않게 한다. Vec 버퍼 주소는 불변이므로 moe_tab이 이 안을
    /// 가리켜도 안전(용량 추가 변경 금지 — 구축 후 불변).
    pub(super) host_stage: Vec<u8>,
    /// [P7-fix 2026-10-10] host_stage 선두에서 cuMemHostRegister에 성공한
    /// 바이트 수(0 = 미등록). 등록 구간 소스는 h2d DMA가 직행한다.
    pub(super) host_stage_reg: usize,
    /// 라우터 로짓/ shared 게이트 스크래치(n_experts ≥ 1).
    pub(super) drt: Buf,
    /// MoE 출력(hidden) — 잔차 ab로 소비된다.
    pub(super) dmo: Buf,
    pub(super) moe_bufs_ok: bool,
    /// [B3 2026-10-12] shared 꼬리 포크 전용 sgate 출력 슬롯(t≤8행 f32 —
    /// 종전 drt 재사용은 topk와의 병렬 경합). 1회 할당·불변(캡처 안전).
    pub(super) dsg: CUdeviceptr,
    /// [B3] 층별 이벤트 쌍 [fork, join] — 캡처 중 재기록 회피(층당 고유).
    pub(super) ev_fork: Vec<ffi::CUevent>,
    /// [B7 2026-10-12] shared 게이트/업 전용 슬롯 — dchain 재사용은 스트리밍
    /// 전문가 스크래치와 충돌(게이트/업이 전문가보다 먼저 계산됨).
    pub(super) dsh: [CUdeviceptr; 2],
    pub(super) dsh_cap: usize,
    /// [B4-2 2026-10-12] 스트리밍 전문가 LRU 캐시(층별 K×3proj q/s).
    /// 가중치 불변 → hit 재사용은 수치 무변화(h2d 생략만). 층별 독립 LRU.
    pub(super) mcache_q: CUdeviceptr,
    pub(super) mcache_s: CUdeviceptr,
    pub(super) mcache_k: usize,
    pub(super) mcache_ids: Vec<u16>,
    pub(super) mcache_age: Vec<u32>,
    pub(super) mcache_tick: u32,
    pub(super) mcache_hits: u64,
    pub(super) mcache_miss: u64,
}

/// [R10 2026-10-12] MTP 초안기 — W4a16Dec 필드 도메인 그룹(순수 이동).
pub(super) struct Mtp {
    /// [MTP 2026-10-12] 모델 자체 MTP 초안기 — 가중치는 plain 레지스트리
    /// (이름 참조), 여기는 버퍼/상태만.
    pub(super) mtp_ok: bool,
    pub(super) mtp_nw: CUdeviceptr,
    pub(super) mtp_zero: CUdeviceptr,
    pub(super) mtp_buf: CUdeviceptr,
    pub(super) mtp_off: [usize; 17],
    pub(super) mtp_prev_off: usize,
    pub(super) mtp_layer: usize,
    pub(super) mtp_rows: usize,
    pub(super) mtp_ffn: usize,
    /// [MTP-2] 슬롯별 재귀 초안열(0=다음+1 … k-1=다음+k).
    pub(super) mtp_draft: Vec<Vec<u32>>,
    pub(super) mtp_pos: Vec<u32>,
    pub(super) mtp_prev_pos: Vec<u32>,
    pub(super) mtp_off_slot: Vec<bool>,
    /// [MTP-GRAPH 2026-10-12] MTP 콜 t(1..=8)별 캡처 — 런치 ~20회 → 1회.
    pub(super) mtp_graphs: Vec<Option<(ffi::CUgraphExec, ffi::CUgraph)>>,
    /// 캡처 실패 후 직접 발사 고정(재시도 방지).
    pub(super) mtp_graph_failed: bool,
    /// MTP 핀드 — 임베드 스테이징(MTP_TMAX×h×4) + argmax 회수(8×4).
    pub(super) mtp_pin_emb: *mut std::ffi::c_void,
    pub(super) mtp_pin_out: *mut std::ffi::c_void,
    /// [MTP-CHAIN 2026-10-12 §3.3 ②] 체인 부기 — 스텝 카운터(4B)·초안
    /// 로그(8×4B — w4a16_mtp_chain_step 기록부).
    pub(super) mtp_chain_ctr: CUdeviceptr,
    pub(super) mtp_chain_log: CUdeviceptr,
    /// [MTP-CHAIN] 체인 t=1 그래프 — 슬롯별(KV/dpp 포인터가 커널 인자로
    /// 베이크되므로 t키 mtp_graphs와 달리 슬롯 키).
    pub(super) mtp_chain_graphs: Vec<Option<(ffi::CUgraphExec, ffi::CUgraph)>>,
}

/// [R10 2026-10-12] 어텐션(KV 분할·스케일) — W4a16Dec 필드 도메인 그룹(순수 이동).
pub(super) struct Attn {
    // ── attn ──
    pub(super) attn: Option<AttnDims>,
    pub(super) dqnw_a: CUdeviceptr,
    pub(super) dknw_a: CUdeviceptr,
    pub(super) dkc: CUdeviceptr,
    pub(super) dvc: CUdeviceptr,
    /// [P13] int8 KV 스케일(K/V 각각 [n_attn*cap][kv_heads] f32) — KVQ 전용.
    pub(super) dksc: CUdeviceptr,
    pub(super) dvsc: CUdeviceptr,
    /// [K6-2 2026-10-12] KV 블록 테이블(디바이스 u64 [slot][layer][mb]×4 —
    /// k, v, ksc, vsc 기저 바이트 주소). set_attn이 구축, 내용 불변
    /// (그래프 캡처 계약 — 캡처는 주소만 베이크).
    pub(super) dblk_tab: CUdeviceptr,
    /// [K6-3 2026-10-13] KV 물리 블록 할당자(호스트) — 테이블 엔트리의 정본.
    /// 다음 단계에서 온디맨드 성장·회수(입장 보장)의 기반.
    pub(super) kv_pool: Option<blocktable::KvPool>,
    pub(super) dpp: CUdeviceptr,
    pub(super) dqg_a: CUdeviceptr,
    pub(super) dkin_a: CUdeviceptr,
    pub(super) dvin_a: CUdeviceptr,
    pub(super) dqh_a: CUdeviceptr,
    /// P8-attn-3: KV 분할 attention 부분합 스크래치
    /// ([T][q_heads][ATTN_SPLITS][258] f32 — m·l·acc256).
    pub(super) dattn_part: CUdeviceptr,
    pub(super) doutv_a: CUdeviceptr,
    pub(super) attn_t_cap: usize,
}

/// [R10 2026-10-12] GDN(디바이스 상태·검증 스냅샷) — W4a16Dec 필드 도메인 그룹(순수 이동).
pub(super) struct Gdn {
    // ── GDN ──
    pub(super) gdn: Option<GdnDims>,
    pub(super) dcw: CUdeviceptr,
    pub(super) dab_c: CUdeviceptr,
    pub(super) dalog: CUdeviceptr,
    pub(super) ddtb: CUdeviceptr,
    pub(super) dnwg: CUdeviceptr,
    pub(super) dring: CUdeviceptr,
    pub(super) dgst: CUdeviceptr,
    pub(super) dqkv: CUdeviceptr,
    pub(super) dzv: CUdeviceptr,
    pub(super) dgxn: CUdeviceptr,
    pub(super) dgq: CUdeviceptr,
    pub(super) dgk: CUdeviceptr,
    pub(super) dgv: CUdeviceptr,
    pub(super) dq2: CUdeviceptr,
    pub(super) dk2: CUdeviceptr,
    pub(super) dv2: CUdeviceptr,
    pub(super) dbg: CUdeviceptr,
    pub(super) dgo: CUdeviceptr,
    /// [P9] t=1 GDN 분할 부분합([h_v][4][2][128] f32)과 dc([h_v][128]).
    pub(super) dgpart: CUdeviceptr,
    pub(super) dgdc: CUdeviceptr,
    /// [A5-4] gdn_scan A/KQ prepass 스크래치 — [2][h_v][maxchunks][CS][CS] half.
    pub(super) dakq: CUdeviceptr,
    pub(super) dgate: CUdeviceptr,
    pub(super) gdn_t_cap: usize,
    /// [TreeWY 2.B 2026-10-12] 스펙 검증 WY 표현 — S_0([n_gdn][h_v][d*d])·
    /// 유사값 dc([n_gdn][8][h_v][d])·누적감쇠 G([n_gdn][8][h_v])·k행
    /// ([n_gdn][8][k_len], f16 반올림 후 값). 종전 dsnap(토큰별 상태
    /// 스냅샷 [t][n_layers][h_v][d*d], 27B 1.50GiB) 폐지 — 롤백은
    /// gdn_wy_reconstruct 재구성(PoC fp32 |ΔS_m|≤1.2e-7, eval 클래스).
    pub(super) dwy_s0: CUdeviceptr,
    pub(super) dwy_dc: CUdeviceptr,
    pub(super) dwy_g: CUdeviceptr,
    pub(super) dwy_k: CUdeviceptr,
    /// conv 링 스냅샷([n_gdn][8][3][conv_ch]) — 입력 이력(WY 대상 아님).
    pub(super) dsnap_ring: CUdeviceptr,
    pub(super) dsave: CUdeviceptr,
    /// [A2 2026-10-11] 접두 재사용 체크포인트(GDN 상태+conv 링, 슬롯별 1개).
    pub(super) dchk: CUdeviceptr,
    /// 체크포인트 위치(토큰 수) — 0 = 없음.
    /// [K3 2026-10-12] 체크포인트 위치 — 슬롯당 512 경계 스택 4개
    /// (512/1024/1536/2048 — 환형 저장). upto 이하 최대 경계.
    pub(super) dchk_pos: Vec<[u32; 4]>,
}

/// [K-STEP 2026-10-12 §3.4 ①] 디바이스 K-스텝 디코드 루프 상태 — W4a16Dec
/// 필드 도메인 그룹. buf 레이아웃: [ctr 4B][stop 4B][out KSTEP_KMAX×4B]
/// (w4a16_kstep_ctrl 기록부 — 리플레이 간 유지, 배치마다 호스트 8B 0 리셋).
pub(super) struct Kstep {
    /// 부기 버퍼(0 = 미할당 — 첫 캡처 시 확보).
    pub(super) buf: CUdeviceptr,
    /// 슬롯별 t=1 스텝 그래프 — 리플레이 k회 = K스텝(KV/dpp 포인터와
    /// slot·eos가 커널 인자로 베이크 → (슬롯, eos) 키. eos는 모델당 상수
    /// (서버 eng.eos())라 실전에서 재캡처는 없고, 불일치 엔트리는 폐기 후
    /// 재캡처(테스트가 서로 다른 eos로 검증).
    pub(super) graphs: Vec<Option<(u32, ffi::CUgraphExec, ffi::CUgraph)>>,
    /// 캡처 실패 후 직접 경로 고정(재시도 방지 — t=1 graph_failed와 동형).
    pub(super) failed: bool,
}

/// [R10 2026-10-12] CUDA Graph 캡처·핀드 버퍼 — W4a16Dec 필드 도메인 그룹(순수 이동).
pub(super) struct Graph {
    // ── CUDA Graph(체인 캡처 — 토큰당 1 launch) ──
    /// [A8 2026-10-09] 슬롯·모드별 캡처 exec 캐시 — 종전 단일 그래프를
    /// 슬롯/모드 전환마다 destroy+재캡처(다중 슬롯 틱마다 수 ms)했고, 이
    /// 캐시가 그 비용을 1회 캡처로 상각한다. 키 = (slot, head, argmax).
    /// 버퍼 재할당(graph_invalidate) 시 전량 폐기(옛 포인터 replay 차단).
    pub(super) graph_cache: Vec<GraphEntry>,
    /// 캡처 실패 후 직접 경로 고정(매 토큰 재시도 방지).
    pub(super) graph_failed: bool,
    /// 캡처 중 임베딩 복사 소스를 pinned로 고정(캡처는 pageable async 불가).
    pub(super) capture_pinned_src: bool,
    pub(super) pin_embed: *mut std::ffi::c_void,
    pub(super) pin_pos: *mut std::ffi::c_void,
    pub(super) pin_out: *mut std::ffi::c_void,
    pub(super) pin_out_len: usize,
    pub(super) pin_batch_tok: *mut std::ffi::c_void,
    /// [S11 2026-10-11] 샘플 배치 로짓 핀드 판독 버퍼([t][head_n] f32).
    pub(super) pin_batch_lg: *mut std::ffi::c_void,
    pub(super) pin_batch_in: *mut std::ffi::c_void,
    pub(super) pin_batch_pos: *mut std::ffi::c_void,
    pub(super) batch_graphs: Vec<(Vec<usize>, ffi::CUgraphExec, ffi::CUgraph)>,
    /// [A6-1 2026-10-11] 검증 체인 (slot, t) 그래프 캐시 — 캡처 실패 시 비활성.
    pub(super) spec_graphs: Vec<((usize, usize), ffi::CUgraphExec, ffi::CUgraph)>,
    pub(super) spec_graph_ok: bool,
    /// [A6-1] 캡처 본문에서 호스트 갱신(sync·slot_pos·체크포인트) 생략 게이트.
    pub(super) capture_no_host: bool,
    /// 캡처 실패 후 직접 발사 고정(재시도 방지 — t=1 graph_failed와 동형).
    pub(super) batch_capture_failed: bool,
}

pub struct W4a16Dec {
    pub(super) moe: MoE,
    pub(super) mtp: Mtp,
    pub(super) kstep: Kstep,
    pub(super) attn: Attn,
    pub(super) gdn: Gdn,
    pub(super) graph: Graph,
    pub cc: CudaCtx,
    /// [§3.4 ② 2026-10-12] 통합 캐시 회계(KV·체크포인트·스펙 상태·
    /// 전문가 LRU 단일 네임스페이스) — 트랜셰 2부터 예산 강제(퇴거 실해제는
    /// ucache_sweep/ucache_pressure).
    ucache: UnifiedCacheRegistry,
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
    /// GEMV 스테이징 — x f16 [t][k], y f32 [t][n].
    xh: Buf,
    dy: Buf,
    // ── norm ──
    dnw: CUdeviceptr,
    norm_w_rows: usize,
    norm: NormBufs,
    // ── ew ──
    ew: EwBufs,
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
    /// [FLA-10 2026-10-10] 플레인(bf16) 경로 A 미러([t][k] bf16) — bf16_mma가
    /// f32 대신 판독(재판독 대역 절반). cast_bf16과 커널 내 변환이 비트동일.
    dabf: Buf,
    /// [S2 2026-10-11] 다중세그먼트 테이블(디바이스 u64) — q/k/v·qkv/z·gate/up
    /// 을 1런치로(런치 꼬리 제거). 업로드·dchain 확정 후 ensure_chain_bufs에서
    /// 1회 작성 — 그래프 캡처 정적 계약(리플레이 간 내용 불변).
    dmulti: CUdeviceptr,
    /// [S2] 층별 [branch_off, branch_cnt, branch_total, gateup_off,
    /// gateup_cnt, gateup_total, branch_tgrid, gateup_tgrid] — 오프셋은 u64
    /// 인덱스, total = 행 합(grid, t=1), *_tgrid = Σceil(n/GEMV_TR)(t행 판).
    dmulti_meta: Vec<[u32; 8]>,
    spec_on: bool,
    /// [A-1 진단] spec scan 커널 on/off(기본 = spec_on) — 교차 대조용.
    spec_scan_on: bool,
    /// t≥2 GEMM 출력 스크래치([t][max_n] f32).
    dyt: Buf,
    // ── GPU head(output.weight bf16) ──
    head_w: CUdeviceptr,
    /// [I8head] int8 head 행스케일(f32[n]) — head_w는 int8 바이트로 전환.
    head_sc: CUdeviceptr,
    head_n: usize,
    head_k: usize,
    head_out: CUdeviceptr,
    /// [P3] argmax 결과(u32 1개) — 그래프 4B readback 대상.
    argmax_out: CUdeviceptr,
    /// [A9 2026-10-10] 배치 디코드 부속 — 로짓[t×head_n]·argmax[t]·핀드
    /// (토큰·임베딩 행·슬롯 pos) + 슬롯집합별 캡처 그래프 캐시.
    dbatch_lg: CUdeviceptr,
    dbatch_am: CUdeviceptr,
    /// [eval] 배치 버퍼 tmax 하한 — 청크 프리필 로짓 회수용(0=기본 공식).
    pub batch_tmax_min: usize,
    /// [S3-1 2026-10-11] 임베딩 gather — token_embd 상주(f16/bf16 원본).
    emb_dev: CUdeviceptr,
    emb_n: usize,
    emb_k: usize,
    emb_bf16: bool,
    /// 토큰 id 스테이징(디바이스, CHAIN_TMAX×4B — ensure_chain_bufs).
    dtg: CUdeviceptr,
    /// 층별 잔차 합 덤프(CPU LLM170_DUMP=debug_layers와 대조용).
    pub debug_layers: bool,
}

// SAFETY: CUDA 핸들(*mut c_void)은 Send가 아니지만, 이 디코더는 서버
// 슬롯 스레드 1개가 소유·사용한다(공유 없음). 컨텍스트 current 전환은
// 진입마다 cc.guard()가 수행한다 — 구 Exl3CudaDecoder의 동일 계약
// (586758df^ exl3_cuda.rs L234 unsafe impl Send) 미러.
unsafe impl Send for W4a16Dec {}

mod attn;
mod blocktable;
mod buffers;
mod chains;
mod gdn;
mod graphs;
mod kernels;
mod kstep;
mod mem;
mod moe;
mod mtp;
mod prefill;
mod probe;
pub mod unified_cache;
mod upload;

mod spec;

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
        // [§3.4 ②] 통합 캐시 회계 — HBM 예산 = 부팅 시점 여유 VRAM(프로브
        // 실패 시 무제한). **트랜셰 2(2026-10-12): enforce 활성** — 예산
        // 초과 시 퇴거가 실제로 일어나고 엔진(ucache_sweep)이 디바이스
        // 메모리를 해제한다. 안전 부분집합: Kv 범주는 고정(퇴거 불가 —
        // 실행 중 디코드 필수 자원), 퇴거 후보는 재계산·재적재 가능한
        // ckpt → spec 상태 → 전문가 LRU뿐(utility 순)이라 활성 워크로드는
        // 안전하다. 참고: 가중치·스크래치는 이 네임스페이스 밖이라 예산은
        // 4 캐시에만 걸린다(전량 회계는 다음 트랜셰).
        let mut ucache = UnifiedCacheRegistry::new();
        if let Some((free, _)) = crate::cuda_mem_free() {
            ucache.set_budget(Tier::Hbm, free);
        }
        ucache.set_enforce(true);
        Ok(W4a16Dec {
            moe: MoE {
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
                dsg: 0,
                ev_fork: Vec::new(),
                dsh: [0, 0],
                dsh_cap: 0,
                mcache_q: 0,
                mcache_s: 0,
                mcache_k: 0,
                mcache_ids: Vec::new(),
                mcache_age: Vec::new(),
                mcache_tick: 0,
                mcache_hits: 0,
                mcache_miss: 0,
                moe_multi_tab: 0,
                moe_multi_rows: Vec::new(),
                moe_multi_rows_t3: Vec::new(),
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
            },
            mtp: Mtp {
                mtp_ok: false,
                mtp_nw: 0,
                mtp_zero: 0,
                mtp_buf: 0,
                mtp_off: [0; 17],
                mtp_prev_off: 0,
                mtp_layer: 0,
                mtp_rows: 0,
                mtp_ffn: 0,
                mtp_draft: Vec::new(),
                mtp_pos: Vec::new(),
                mtp_prev_pos: Vec::new(),
                mtp_off_slot: Vec::new(),
                mtp_graphs: vec![None; 9],
                mtp_graph_failed: false,
                mtp_pin_emb: std::ptr::null_mut(),
                mtp_pin_out: std::ptr::null_mut(),
                mtp_chain_ctr: 0,
                mtp_chain_log: 0,
                mtp_chain_graphs: Vec::new(),
            },
            kstep: Kstep {
                buf: 0,
                graphs: Vec::new(),
                failed: false,
            },
            attn: Attn {
                attn: None,
                dqnw_a: 0,
                dknw_a: 0,
                dkc: 0,
                dvc: 0,
                dksc: 0,
                dvsc: 0,
                dblk_tab: 0,
                kv_pool: None,
                dpp: 0,
                dqg_a: 0,
                dkin_a: 0,
                dvin_a: 0,
                dqh_a: 0,
                doutv_a: 0,
                dattn_part: 0,
                attn_t_cap: 0,
            },
            gdn: Gdn {
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
                dwy_s0: 0,
                dwy_dc: 0,
                dwy_g: 0,
                dwy_k: 0,
                dsnap_ring: 0,
                dsave: 0,
                dchk: 0,
                dchk_pos: Vec::new(),
            },
            graph: Graph {
                graph_cache: Vec::new(),
                graph_failed: false,
                capture_pinned_src: false,
                pin_embed: std::ptr::null_mut(),
                pin_pos: std::ptr::null_mut(),
                pin_out: std::ptr::null_mut(),
                pin_out_len: 0,
                pin_batch_tok: std::ptr::null_mut(),
                pin_batch_lg: std::ptr::null_mut(),
                pin_batch_in: std::ptr::null_mut(),
                pin_batch_pos: std::ptr::null_mut(),
                batch_graphs: Vec::new(),
                spec_graphs: Vec::new(),
                spec_graph_ok: true,
                capture_no_host: false,
                batch_capture_failed: false,
            },
            cc,
            ucache,
            hidden,
            n_layers,
            n_slots: n_slots.max(1),
            slot_pos: vec![0; n_slots.max(1)],
            lins: HashMap::new(),
            plains: HashMap::new(),
            plain_weights: false,
            weights_bytes: 0,
            experts_bytes: 0,
            xh: Buf::default(),
            dy: Buf::default(),
            dnw: 0,
            norm_w_rows: 0,
            norm: NormBufs::default(),
            ew: EwBufs::default(),
            dres: 0,
            dab_dev: 0,
            dchain: [0; 5],
            stg_w0: 0,
            stg_w1: 0,
            stg_w2: 0,
            chain_bufs_ok: false,
            dx32: Buf::default(),
            dx16: Buf::default(),
            dabf: Buf::default(),
            dmulti: 0,
            dmulti_meta: Vec::new(),
            spec_on: false,
            spec_scan_on: true,
            dyt: Buf::default(),
            head_w: 0,
            head_sc: 0,
            head_n: 0,
            head_k: 0,
            head_out: 0,
            argmax_out: 0,
            dbatch_lg: 0,
            dbatch_am: 0,
            batch_tmax_min: 0,
            emb_dev: 0,
            emb_n: 0,
            emb_k: 0,
            emb_bf16: false,
            dtg: 0,
            debug_layers: false,
        })
    }

    pub fn device_name(&self) -> &str {
        &self.cc.device_name
    }

    /// [D2 검수 2026-10-13] **로드 실패 경로 전용** — 이 컨텍스트의 등록된
    /// 디바이스 할당 전량 해제(재시도 루프의 VRAM 회수). 성공 엔진에서는
    /// 절대 호출 금지(버퍼 포인터를 엔진이 계속 소유 — 이중 해제).
    pub fn free_all_device(&self) {
        self.cc.free_all();
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
        // [결함수정 2026-10-11] 6-param 커널에 l4(4인자) 발사 — y32/y16 파라미터가
        // 스택 잔여값을 읽는 미정의 동작이었다. 널 명시(l6).
        let (mut a0, mut a1, mut a2, mut a3, mut a4) =
            (self.ew.dewg, self.ew.dewu, self.ew.dew, 0u64, 0u64);
        self.cc.launch(
            f,
            g.len().div_ceil(128) as u32,
            1,
            128,
            &mut crate::rawcuda::args::l6(&mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut nn),
        )?;
        let mut yb = vec![0u8; g.len() * 4];
        self.cc.d2h(&mut yb, self.ew.dew)?;
        self.cc.sync()?;
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
        Ok(unsafe { std::slice::from_raw_parts(yb.as_ptr() as *const f32, g.len()) }.to_vec())
    }

    fn lin_spec(&self, name: &str) -> Result<(CUdeviceptr, CUdeviceptr, usize, usize), String> {
        self.lins
            .get(name)
            .copied()
            .ok_or_else(|| format!("상주 선형 없음: {name}"))
    }

    /// MoE 배치 모드 문자열(모니터링) — "none" | "resident" | "streaming".
    pub fn moe_mode(&self) -> &'static str {
        if self.moe.n_experts == 0 {
            "none"
        } else if self.moe.moe_resident {
            "resident"
        } else {
            "streaming"
        }
    }

    /// MoE 모델 여부(프리필 t=1 강제 등).
    pub fn is_moe(&self) -> bool {
        self.moe.n_experts > 0
    }

    /// 전문가 전량 상주 여부 — MoE 배치 프리필(t>1)의 전제.
    pub fn moe_experts_resident(&self) -> bool {
        self.moe.moe_resident
    }

    fn plain_spec(&self, name: &str) -> Result<(CUdeviceptr, usize, usize), String> {
        self.plains
            .get(name)
            .copied()
            .ok_or_else(|| format!("상주 플레인 없음: {name}"))
    }
}
