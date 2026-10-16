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
/// [2.N A2→A3 2026-10-15] gemma4 청크 상한 — 전문가·topk·attn은 t 직지원
/// (MoE 버퍼 = CHAIN_TMAX, attn.cu TMAX 512)이고 **t행 GEMV 계열만** t≤16
/// 스위치 계약이라 호스트가 행 분할 호출한다(행별 산술 독립 — 비트동일).
/// qwen 배치 상한(BATCH_DEC_MAX)과 분리 — qwen 경로 동작 불변.
/// [A3 2026-10-15 채택 — 64→512] 같은 세션 교차 A/B(Gemma-4-26B, SM
/// 2520/2790MHz 기록, reps3·의심 구간 라운드2 교차): pp256 64→128
/// **+19.2%** · 128→256 **+12.4%** · pp1024 256→512 **+6.6~7.2%** ·
/// pp4096 **+6.5%**(누적 ≈+43%) · tg·np 불변. 정합 = 청크 경계 무관
/// 토큰 스트림 비트동일(300토큰 프롬프트 64 vs 512 md5 일치 · np4 해시
/// 일치 — 행분할 비트동일 계약의 실증). 512 = 구조 상한 3종 정렬
/// (CHAIN_TMAX · attn TMAX · sched PF_CHUNK_MAX — 서빙 외부 청크와 정합,
/// 혼합 부하 캡 PF_CHUNK_MIXED=128은 그대로). 잔여 바닥 = dense 가중치
/// t/16회 재판독(~56MB/token 고정 — 512에서 전문가 판독과 공동 지배):
/// 비트계약 FFMA 타일 GEMM(dense mma 재도전 조건)의 실측 동기.
pub const G4_CHUNK_MAX: usize = 512;
/// [2.N t≤16 2026-10-15] t행 GEMV 계열 커널의 t 상한 — moe.cu G4_TMAX·
/// gptq4(GEMV_T_CASE…)·head.cu i8_t 스위치 미러(정적검사 trow_switch_coverage).
pub const G4_TROW_MAX: usize = 16;
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
/// [GQA 패킹 채택 2026-10-13] 디코드(t≤8) 분할 — 종전 32(q_heads×S 그리드로
/// 점유 회복)에서 64로. part_qg 패킹 후 그리드 = kv_heads×S(27B 4·35B 2)라
/// 장문에서 지연 바운드가 커져 분할 확대가 유효해졌다 — 실측(27B, bench):
/// tg@64K 36.05→39.75(+10%)·tg@16K 47.8→49.3(+3%)·tg@512 53.5→53.0(−0.9%)
/// / 35B 짧은 창 −0.5%. S=128은 장문 +0.4 t/s뿐이고 단문 −3%(빈 분할 블록
/// 디스패치 비용)라 64가 균형점. 곡선·게이트 수치는 원장(4090-detail).
pub const ATTN_SPLITS_D: usize = 64;
/// attn.cu 병합 smem(ATTN_SPLITS_MAX)과 동일해야 함(테스트 대조).
pub const ATTN_SPLITS_MAX: usize = 64;

/// 층별 어텐션 파라미터 — gemma4처럼 층마다 dims·종류가 다른 모델용.
/// `AttnDims.layers`(디코더 보관)가 비어 있으면 균일 스칼라 필드로 합성 —
/// qwen35 현행 경로는 완전히 동일하다.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttnLayer {
    pub kv_heads: usize,
    pub d: usize,
    /// 슬라이딩 창 어텐션(마스크) — 창 크기는 `AttnDims.sliding_window`.
    pub sliding: bool,
    /// k_eq_v — v_proj 없이 v = v_norm(k 투영 원시값).
    pub k_eq_v: bool,
    /// q·k 스케일 — 0.0 = 1/√d 유도(qwen35), gemma4 = 1.0 명시.
    pub kq_scale: f32,
}

impl AttnLayer {
    /// 유효 q·k 스케일 — 0.0이면 1/√d.
    pub fn scale(&self) -> f32 {
        if self.kq_scale > 0.0 {
            self.kq_scale
        } else {
            1.0 / (self.d as f32).sqrt()
        }
    }
}

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
    /// [gemma4 P2 2026-10-14] 슬라이딩 창(자기 포함) — sliding 층 마스크 폭. 0 = 미사용.
    pub sliding_window: usize,
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
    /// [gemma4 P2 2026-10-14] 층별 파라미터 — 현재는 균일 스칼라 합성
    /// (kq_scale 0.0 = 1/√d 유도 → qwen 0.0625 동일값·비트 불변). 층별
    /// 테이블(슬라이딩/풀 혼합) 도입 시 여기서 오버라이드 조회로 확장.
    pub fn layer(&self, _il: usize) -> AttnLayer {
        AttnLayer {
            kv_heads: self.kv_heads,
            d: self.d,
            sliding: false,
            k_eq_v: false,
            kq_scale: 0.0,
        }
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
    /// [gemma4 P2 2026-10-14] 전문가 GLU 활성 = gelu_tanh(모델별 — 기본 silu).
    pub(super) glu_gelu: bool,
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
    /// [P4 2026-10-16] topk idx d2h용 핀드 버퍼(스트리밍 프리필) — 층마다
    /// 힙 할당+페이지러블 복사 대신 상시 핀드 1회 할당 재사용. DMA 직행으로
    /// d2h 지연을 줄인다(용량 부족 시 재할당 — 스트리밍 경로에서는 ns 단조
    /// 증가 아님, 청크 tmax 상한 고정).
    pub(super) pin_moe_idx: *mut std::ffi::c_void,
    pub(super) pin_moe_idx_cap: usize,
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
    /// [gemma4 2026-10-14] 층별 어텐션 파라미터(비면 균일 = `attn`).
    pub(super) layers: Vec<AttnLayer>,
    // ── attn ──
    pub(super) attn: Option<AttnDims>,
    pub(super) dqnw_a: CUdeviceptr,
    pub(super) dknw_a: CUdeviceptr,
    /// [gemma4 P2-3 2026-10-14] g4 rope 파라미터 테이블(디바이스 f32
    /// [n_attn][7] — {d, pairs, span, denom, base, vn_on, k_eq_v}).
    /// `set_g4_rope`가 업로드 · g4 체인(prep/part_g4)이 층 인덱스로 소비.
    pub(super) g4_rp: CUdeviceptr,
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
    /// [K6-3 예산] KV 아레나 예산(바이트) — set_attn이 소비(풀 총량 = min(예산, 슬롯×ctx)).
    pub(super) kv_budget: Option<u64>,
    /// [2026-10-16] VRAM 티어 상한(바이트 — 서버 파생). 초과분은 호스트 아레나.
    /// None = 전량 VRAM(종전 동작).
    pub(super) kv_vram_cap: Option<u64>,
    /// [2026-10-16] 호스트 KV 아레나 — 핀드·디바이스 매핑(커널 직접 접근).
    /// `*_h` = 디바이스 가시 주소(표 기입용), `h*_h` = 해제용 호스트 포인터.
    pub(super) dkc_h: CUdeviceptr,
    pub(super) dvc_h: CUdeviceptr,
    pub(super) dksc_h: CUdeviceptr,
    pub(super) dvsc_h: CUdeviceptr,
    pub(super) hkc_h: *mut std::ffi::c_void,
    pub(super) hvc_h: *mut std::ffi::c_void,
    pub(super) hksc_h: *mut std::ffi::c_void,
    pub(super) hvsc_h: *mut std::ffi::c_void,
    /// VRAM 아레나가 담는 블록 수(블록 id < 이 값 = VRAM, 이후 = 호스트).
    pub(super) kv_vram_blocks: usize,
    /// 호스트 아레나 실할당 바이트(계층 회계 — /stats 오프로드율).
    pub(super) kv_host_bytes: u64,
    /// [K6-3 예산] 실할당 아레나+테이블 바이트(관측 — mem_stats/kv_pool_bytes).
    pub(super) kv_arena_bytes: u64,
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
    /// [행-스케일 2026-10-13] 상태 행(kdim)별 f32 스케일 [슬롯][n_gdn][h_v][128].
    pub(super) dsc: CUdeviceptr,
    /// [P1 2026-10-16] 스케일 초기값(1.0) 단일 슬롯분 — reset_state가 동기
    /// 페이지어블 h2d 대신 d2d(스트림 비동기)로 복사(드레인 제거).
    pub(super) dsc_ones: CUdeviceptr,
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
    /// [dsave 대체 2026-10-14] 라운드 전 conv 링 1벌([n_gdn][3][conv_ch]) —
    /// acc==0(전량 거절) 복원용. 라운드 conv가 **갱신 전** 링을 기록한다.
    pub(super) dring0: CUdeviceptr,
    /// [프로브 전용 2026-10-14] 정밀 상태 저장 스크래치 — spec_save_state가
    /// 지연 할당. serve 경로는 0(종전 enable_spec 상주분 — 27B/4슬롯 약
    /// 315MiB — 을 회수해 예약·컨텍스트로 환원).
    pub(super) dsave: CUdeviceptr,
    /// [A2 2026-10-11] 접두 재사용 체크포인트(GDN 상태+conv 링, 슬롯별 1개).
    /// [호스트 이관 2026-10-14] 접두 재사용 체크포인트 링 = **pinned 호스트
    /// RAM**(콜드 백업 — VRAM 상주 불필요). 복원 = h2d 수 ms(개발기)로 포크
    /// 도달·지연 불변, VRAM −1.26GiB(27B/4슬롯) ≈ 컨텍스트 +37K토큰.
    pub(super) dchk: *mut std::ffi::c_void,
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
    /// [배치 행 퓨전 2026-10-14] 행별 메타 핀드(blk/pp/ring/st/sc — u64×t).
    /// 호스트가 호출부에서 기입, 캡처 h2d 노드가 replay 시점에 소비.
    pub(super) pin_row_blk: *mut std::ffi::c_void,
    pub(super) pin_row_pp: *mut std::ffi::c_void,
    pub(super) pin_row_ring: *mut std::ffi::c_void,
    pub(super) pin_row_st: *mut std::ffi::c_void,
    pub(super) pin_row_sc: *mut std::ffi::c_void,
    /// [S11 2026-10-11] 샘플 배치 로짓 핀드 판독 버퍼([t][head_n] f32).
    pub(super) pin_batch_lg: *mut std::ffi::c_void,
    /// [gemma4 P3 2026-10-14] 청크 head 로짓 핀드 판독 버퍼([t][head_n] f32).
    /// 캡처 그래프의 d2h 목적지(고정 주소) — 재할당 시 그래프 전량 폐기.
    pub(super) pin_g4_out: *mut std::ffi::c_void,
    pub(super) pin_g4_out_len: usize,
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
    /// [gemma4 P3 v2 2026-10-14] t행 청크 그래프 캐시 — 키 = (t, 슬롯집합,
    /// head_last). 청크 본문(층 루프+head+d2h)이 전부 같은 스트림 발사라
    /// 1그래프로 붙는다(호스트 왕복 0 — 프롤로그/에필로그는 캡처 밖).
    pub(super) g4_graphs: Vec<G4GraphEntry>,
    /// [gemma4 P3 v2] g4 청크 캡처 실패 후 직접 발사 고정(재시도 방지).
    pub(super) g4_capture_failed: bool,
}

/// [gemma4 P3 v2 2026-10-14] 청크 그래프 1개 — 캐시 키와 exec 핸들.
pub(super) struct G4GraphEntry {
    pub(super) t: usize,
    pub(super) slots: Vec<usize>,
    pub(super) head_last: bool,
    pub(super) exec: ffi::CUgraphExec,
    pub(super) handle: ffi::CUgraph,
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
    /// [K6-3 2026-10-13] 슬롯별 KV 확보 고수위(토큰) — kv_ensure 무비용 no-op용.
    pub kv_assigned: Vec<usize>,
    /// [2.N Stage 2 2026-10-15] 슬롯별 KV 플로어(행) — 슬라이딩 층에서 이 행
    /// 미만 블록은 미보유(해제)해도 된다. 0 = 미트림(종전 동작). 풀층은 트림
    /// 대상이 아니라 플로어와 무관하다. kv_release(리셋)가 0으로 되돌린다.
    pub kv_floor: Vec<u32>,
    /// 선형 상주 — 이름 → (packed, scale, n, k).
    lins: HashMap<String, (CUdeviceptr, CUdeviceptr, usize, usize)>,
    /// [gemma4 P3 2026-10-14] t행 청크 스크래치 — A 입력([t][k] f32)·Y 출력.
    g4_a: Buf,
    g4_y: Buf,
    /// [gemma4 P3] 층별 pes(μ_e) 디바이스 상주(0 = 미등록).
    g4_pes: Vec<CUdeviceptr>,
    /// [gemma4 P3 2026-10-14] 층별 layer_scalar(디바이스 청크 — 커널 스칼라
    /// 인자로 소비). set_g4_layer_scalar 선행 계약.
    g4_ls: Vec<f32>,
    /// [gemma4 P3] 청크 디바이스 체인 스크래치 풀(슬롯별 1회 할당·재사용).
    g4_pool: Vec<Buf>,
    /// [gemma4 P3] 어텐션 행 테이블 업로드 캐시 — (슬롯, pos) 동일 시 생략.
    g4_atab: (Vec<usize>, Vec<u32>),
    /// [gemma4 P2-3 2026-10-14] 선형 그룹·스케일 dtype — 이름 → (group, scale_bf16).
    /// `upload_lin`(g128/f16) 등록분은 미기재(디스패치 기본값) · `upload_lin_g`가
    /// 기재한다. GEMV 심볼 선택(g128 vs g32_bf16)만 가른다.
    lin_grp: HashMap<String, (usize, bool)>,
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
    /// [A2b 2026-10-15] 그룹 mma A의 (hi|lo) f16 packed 미러(u32/원소, 4B) —
    /// cast_hilo 산출. 3-pass 정밀도 회수(A 반올림 ~2^-11 제거).
    dxp: Buf,
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
    /// [배치 행 퓨전 2026-10-14] 행별 메타 디바이스 배열(u64×t) — 배치 체인
    /// 퓨전 런치가 읽는다(캡처는 주소만 베이크, 내용은 h2d 노드가 기록).
    drow_blk: CUdeviceptr,
    drow_pp: CUdeviceptr,
    drow_ring: CUdeviceptr,
    drow_st: CUdeviceptr,
    drow_sc: CUdeviceptr,
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
    /// [P4 2026-10-15] 라우팅 top-k 캡처(eval 진단 전용) — g4_moe_dev가 층마다
    /// moe_idx [t][k]를 route_out으로 복사(판독만 — 수치 불변). 그래프 캡처와
    /// 양립 불가(g4_graph_ok 차단) — 켜면 직접 발사 경로로만 동작.
    pub route_capture: bool,
    /// 캡처 수집부(청크×층 순서로 t·k원소씩 append — 소비자가 재구성).
    pub route_out: Vec<u32>,
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
mod g4;
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
                glu_gelu: false,
                moe_group: 32,
                moe_scale_bf16: true,
                moe_tab: Vec::new(),
                moe_resident: false,
                moe_dev_tab: 0,
                moe_idx: 0,
                moe_wt: 0,
                pin_moe_idx: std::ptr::null_mut(),
                pin_moe_idx_cap: 0,
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
                layers: Vec::new(),
                attn: None,
                dqnw_a: 0,
                dknw_a: 0,
                g4_rp: 0,
                dkc: 0,
                dvc: 0,
                dksc: 0,
                dvsc: 0,
                dblk_tab: 0,
                kv_pool: None,
                kv_budget: None,
                kv_vram_cap: None,
                dkc_h: 0,
                dvc_h: 0,
                dksc_h: 0,
                dvsc_h: 0,
                hkc_h: std::ptr::null_mut(),
                hvc_h: std::ptr::null_mut(),
                hksc_h: std::ptr::null_mut(),
                hvsc_h: std::ptr::null_mut(),
                kv_vram_blocks: 0,
                kv_host_bytes: 0,
                kv_arena_bytes: 0,
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
                dsc: 0,
                dsc_ones: 0,
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
                dring0: 0,
                dsave: 0,
                dchk: std::ptr::null_mut(),
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
                pin_row_blk: std::ptr::null_mut(),
                pin_row_pp: std::ptr::null_mut(),
                pin_row_ring: std::ptr::null_mut(),
                pin_row_st: std::ptr::null_mut(),
                pin_row_sc: std::ptr::null_mut(),
                pin_batch_lg: std::ptr::null_mut(),
                pin_g4_out: std::ptr::null_mut(),
                pin_g4_out_len: 0,
                pin_batch_in: std::ptr::null_mut(),
                pin_batch_pos: std::ptr::null_mut(),
                batch_graphs: Vec::new(),
                spec_graphs: Vec::new(),
                spec_graph_ok: true,
                capture_no_host: false,
                batch_capture_failed: false,
                g4_graphs: Vec::new(),
                g4_capture_failed: false,
            },
            cc,
            ucache,
            hidden,
            n_layers,
            n_slots: n_slots.max(1),
            slot_pos: vec![0; n_slots.max(1)],
            kv_assigned: vec![0; n_slots.max(1)],
            kv_floor: vec![0; n_slots.max(1)],
            lins: HashMap::new(),
            g4_a: Buf { ptr: 0, cap: 0 },
            g4_y: Buf { ptr: 0, cap: 0 },
            g4_pes: Vec::new(),
            g4_ls: Vec::new(),
            g4_pool: Vec::new(),
            g4_atab: (Vec::new(), Vec::new()),
            lin_grp: HashMap::new(),
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
            dxp: Buf::default(),
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
            drow_blk: 0,
            drow_pp: 0,
            drow_ring: 0,
            drow_st: 0,
            drow_sc: 0,
            batch_tmax_min: 0,
            emb_dev: 0,
            emb_n: 0,
            emb_k: 0,
            emb_bf16: false,
            dtg: 0,
            debug_layers: false,

            route_capture: false,
            route_out: Vec::new(),
        })
    }

    pub fn device_name(&self) -> &str {
        &self.cc.device_name
    }

    /// [D2 검수 2026-10-13] **로드 실패 경로 전용** — 이 컨텍스트의 등록된
    /// 디바이스 할당 전량 해제(재시도 루프의 VRAM 회수). 성공 엔진에서는
    /// 절대 호출 금지(버퍼 포인터를 엔진이 계속 소유 — 이중 해제).
    pub fn free_all_device(&mut self) {
        // 호스트 KV 아레나(핀드) — cc 대장 밖이라 먼저 수동 해제.
        let _ = self.free_kv_host_arenas();
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
        // [gemma4 P2-3 2026-10-14] 그룹별 심볼 — g128/f16(기본) vs g32/bf16.
        // 런치 구성은 동일(게이트 `w4a16-gemm`와 같은 표) — 심볼만 교체.
        let (grp, sbf) = self.lin_grp.get(name).copied().unwrap_or((128, false));
        let sym = crate::rawcuda::gptq4::kernel_sym(true, grp, sbf)?;
        let f = self.cc.function(sym)?;
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

    /// [gemma4 P2-3 2026-10-14] 상주 선형 GEMV — **원시 f32 x**(f16 왕복 없음).
    /// gemma4 v0 검증 체인의 수치 클래스: 커널 자체는 f32 x 계약이므로 왕복은
    /// 순수 주입 오차다(체인 검증 = CPU 오라클 f32와 동일 계급). qwen 스테이징
    /// 체인의 f16 클래스(`gemv_host`)와 달리 gemma4는 f32 단일 클래스 —
    /// 서빙 체인(P3)의 클래스 선택은 도입 시 A/B·골든 재동결 소관.
    pub fn gemv_f32_host(&mut self, name: &str, x: &[f32]) -> Result<Vec<f32>, String> {
        let _g = self.cc.guard()?;
        let Some(&(dq, ds, n, k)) = self.lins.get(name) else {
            return Err(format!("gemv_f32: 상주 선형 없음: {name}"));
        };
        if x.len() != k {
            return Err(format!("gemv_f32 {name}: x={} != k={k}", x.len()));
        }
        if n > self.dy.cap {
            self.graph_invalidate(); // [P10]
            self.cc.sync()?;
            if self.dy.ptr != 0 {
                self.cc.free(self.dy.ptr)?;
            }
            self.dy.ptr = self.cc.alloc(n * 4)?;
            self.dy.cap = n;
        }
        if k > self.dx32.cap {
            self.graph_invalidate(); // [P10]
            self.cc.sync()?;
            if self.dx32.ptr != 0 {
                self.cc.free(self.dx32.ptr)?;
            }
            self.dx32.ptr = self.cc.alloc(k * 4)?;
            self.dx32.cap = k;
        }
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
        let xb = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) };
        self.cc.h2d(self.dx32.ptr, xb)?;
        // 그룹별 심볼(gemv_launch와 동일 표) — 런치 구성은 t=1 GEMV(행=블록/64레인).
        let (grp, sbf) = self.lin_grp.get(name).copied().unwrap_or((128, false));
        if k > grp * 256 {
            return Err(format!(
                "gemv_f32 {name}: k={k} > {} (t=1 smem sc 계약, g{grp})",
                grp * 256
            ));
        }
        let f = self
            .cc
            .function(crate::rawcuda::gptq4::kernel_sym(false, grp, sbf)?)?;
        let (mut p_q, mut p_s, mut p_x, mut p_y) = (dq, ds, self.dx32.ptr, self.dy.ptr);
        let (mut p_n, mut p_k) = (n as i32, k as i32);
        self.cc.launch(
            f,
            n as u32,
            1,
            64,
            &mut crate::rawcuda::args::l6(
                &mut p_q, &mut p_s, &mut p_x, &mut p_y, &mut p_n, &mut p_k,
            ),
        )?;
        let mut ob = vec![0u8; n * 4];
        self.cc.d2h_async(ob.as_mut_ptr(), self.dy.ptr, n * 4)?;
        self.cc.sync()?;
        Ok(ob
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    pub fn ew_host(&mut self, g: &[f32], u: &[f32]) -> Result<Vec<f32>, String> {
        self.ew_act_host("ew", g, u)
    }

    /// [gemma4 P2 2026-10-14] gelu_tanh 게이트·업 곱 — ew_gelu 커널(동일 버퍼).
    pub fn ew_gelu_host(&mut self, g: &[f32], u: &[f32]) -> Result<Vec<f32>, String> {
        self.ew_act_host("ew_gelu", g, u)
    }

    fn ew_act_host(&mut self, kernel: &str, g: &[f32], u: &[f32]) -> Result<Vec<f32>, String> {
        if g.is_empty() || g.len() != u.len() {
            return Err(format!("{kernel}: g={} u={} 계약 위반", g.len(), u.len()));
        }
        // [P3 2026-10-14] 워커 스레드 대비 — ensure_ew_bufs(할당)·h2d는 컨텍스트 필수.
        let _g = self.cc.guard()?;
        self.ensure_ew_bufs(g.len())?;
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
        let gb = unsafe { std::slice::from_raw_parts(g.as_ptr() as *const u8, g.len() * 4) };
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
        let ub = unsafe { std::slice::from_raw_parts(u.as_ptr() as *const u8, u.len() * 4) };
        self.cc.h2d(self.ew.dewg, gb)?;
        self.cc.h2d(self.ew.dewu, ub)?;
        let f = self.cc.function(kernel)?;
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

    /// [§2.M ① 2026-10-15] 스트리밍 전문가 LRU 캐시 계측 — (hits, misses, k).
    /// 프리필 스테이징 + t=1 경로 공용 계수기(핫스위칭 판정 원료).
    pub fn mcache_stats(&self) -> (u64, u64, usize) {
        (
            self.moe.mcache_hits,
            self.moe.mcache_miss,
            self.moe.mcache_k,
        )
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
