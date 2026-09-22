//! VkAcc — Vulkan matmul 가속기 (plans/12·13). HIP과 병립:
//! LLM170_GPU_RUNTIME=vulkan 시 주입, GDN/프레임은 CPU 폴백 (트레이트 Err).
//! 구조: 파이프라인·버퍼·가중치는 전부 지연 초기화 캐시, dispatch 헬퍼가
//! SSBO 바인딩+push+발사를 일원화 (M4b 확장 지점).

use crate::rawvk::context::{Pipes, VkBuf, VkCtx};
use ash::vk;
use ash::vk::Handle as _VkHandle;
use llm170_core::matmul::{FrameHost as _, MatmulHost, Weight};
use llm170_gguf::GgmlType;
use parking_lot::Mutex;
use std::collections::HashMap;

pub const GEMV_SPV: &[u8] = include_bytes!("../spv/gemv3.spv");
pub const QUANT_SPV: &[u8] = include_bytes!("../spv/quant_q8.spv");
pub const ARGMAX2_SPV: &[u8] = include_bytes!("../spv/argmax2.spv");
pub const RMS_SPV: &[u8] = include_bytes!("../spv/rms.spv");
pub const SILU_SPV: &[u8] = include_bytes!("../spv/silu_mul.spv");
/// plans/84 B — vk 프레임 경로 유틸 셰이더군.
const SILU_DIV_SPV: &[u8] = include_bytes!("../spv/silu_div.spv");
const SCALE_SPV: &[u8] = include_bytes!("../spv/scale.spv");
const COPY_ROWS_SPV: &[u8] = include_bytes!("../spv/copy_rows.spv");
const BCAST_ROWS_SPV: &[u8] = include_bytes!("../spv/bcast_rows.spv");
const AXPY_T_SPV: &[u8] = include_bytes!("../spv/axpy_scaled_t.spv");
/// plans/84 B — MoE 프레임 ops.
const MOE_TOP10_SPV: &[u8] = include_bytes!("../spv/moe_top10.spv");
const PERMUTE_SPV: &[u8] = include_bytes!("../spv/permute_rows.spv");
const PERMUTE_U32_SPV: &[u8] = include_bytes!("../spv/permute_rows_u32.spv");
const MOE_WSUM_SPV: &[u8] = include_bytes!("../spv/moe_wsum.spv");
const MOE_GATHER_SPV: &[u8] = include_bytes!("../spv/moe_gather.spv");
/// plans/84 B — 어텐션 반쪽 프레임 ops.
const HC_GATE_MEAN_SPV: &[u8] = include_bytes!("../spv/hc_gate_mean.spv");
const HC_COMBINE_SPV: &[u8] = include_bytes!("../spv/hc_combine.spv");
const NORM_GATED_SIG_SPV: &[u8] = include_bytes!("../spv/norm_gated_sig.spv");
const GDN_BETA_G_SPV: &[u8] = include_bytes!("../spv/gdn_beta_g.spv");
const EW_SIGMOID_SPV: &[u8] = include_bytes!("../spv/ew_sigmoid.spv");
const SPLIT3_SPV: &[u8] = include_bytes!("../spv/split3.spv");
const GDN_CONV_T2_SPV: &[u8] = include_bytes!("../spv/gdn_conv_t2.spv");
const GDN_CONV_ST_SPV: &[u8] = include_bytes!("../spv/fn_gdn_conv_state.spv");
const GDN_CONV_SEQ_SPV: &[u8] = include_bytes!("../spv/gdn_conv_seq.spv");
const L2_ROWS_SPV: &[u8] = include_bytes!("../spv/l2_rows.spv");
const L2_ROWS2_SPV: &[u8] = include_bytes!("../spv/l2_rows2_scale.spv");
/// plans/84 B — FN GDN AR: 전치 상태(gdn_ar_w_swap 동일열).
const FN_GDN_AR_SWAP_SPV: &[u8] = include_bytes!("../spv/fn_gdn_ar_swap.spv");
/// plans/84 B — FN QSA: 선택 어텐션 + 인덱서 블록키 갱신.
const FN_QSA_ATTN_SEL_SPV: &[u8] = include_bytes!("../spv/fn_qsa_attn_sel.spv");
const FN_IDX_BK_SPV: &[u8] = include_bytes!("../spv/fn_idx_bk_update.spv");
/// plans/86 §2 — QSA q/k norm+rope (qk_norm_rope 동일열).
const FN_QK_NORM_ROPE_SPV: &[u8] = include_bytes!("../spv/fn_qk_norm_rope.spv");
/// plans/85 §2 — FN QSA 디코드 선택: q norm+rope → 블록 점수 → 순위 → 전개.
const FN_IDX_Q_ROPE_SPV: &[u8] = include_bytes!("../spv/fn_idx_q_rope.spv");
const FN_IDX_SCORE_SPV: &[u8] = include_bytes!("../spv/fn_idx_score.spv");
const FN_IDX_RANK_SPV: &[u8] = include_bytes!("../spv/fn_idx_rank.spv");
const FN_IDX_EXPAND_SPV: &[u8] = include_bytes!("../spv/fn_idx_expand.spv");
/// plans/85 §2 — 프레임 로짓 행별 GPU argmax(동률 최저 인덱스).
const FN_ARGMAX_ROWS_SPV: &[u8] = include_bytes!("../spv/fn_argmax_rows.spv");
/// plans/88 P2 — MoE 그룹 프리필: 디바이스 그룹화·q4_K/q5_1 타일·융합 산란.
const FN_MOE_GROUP_SPV: &[u8] = include_bytes!("../spv/fn_moe_group.spv");
const FN_MOE_TILE_Q4K_SPV: &[u8] = include_bytes!("../spv/fn_moe_tile_q4k.spv");
const FN_MOE_TILE_Q51_SPV: &[u8] = include_bytes!("../spv/fn_moe_tile_q51.spv");
const FN_TILE_Q8_SPV: &[u8] = include_bytes!("../spv/fn_tile_q8.spv");
/// plans/88 P1 — f32/BF16 밀집 GEMV(mm_group F32·BF16 멤버 — 값폴백 소거).
const FN_MM_F32_SPV: &[u8] = include_bytes!("../spv/fn_mm_f32.spv");
/// plans/88 P1 — MoE direct-ids GEMV: gemv3의 ids 구동판(그리드 (n_out, rows),
/// 워크그룹=행, ids[r]로 전문가 베이스 산출). 행 산술은 gemv3와 비트 동일.
const FN_MOE_IDS_SPV: &[u8] = include_bytes!("../spv/fn_moe_ids.spv");
/// plans/89 P0.2 — 디코드 밀집 GEMV: decoder gemv8 패밀리(llama dmmv 포트)를
/// VkAcc(프레임)에서도 직접 발사. f32 활성 직결(quant 스킵)이라 W4A8 경로와
/// 산술 클래스가 다르다 — ckdiff·게이트 재기록 절차로 수용(원장 31 전례).
const GEMV8_Q8B_SPV: &[u8] = include_bytes!("../spv/gemv8_q8b.spv");
const GEMV8_Q4B_SPV: &[u8] = include_bytes!("../spv/gemv8_q4b.spv");
/// plans/89 P0.2 — f32/BF16 디코드 GEMV(라우터·sh-gate): fn_mm_f32(256스레드
/// f64 트리, 512WG 지연바운드 — 실측 ~0.4GB/s급)의 64스레드 서브그룹Add 판.
const MM_F32B_SPV: &[u8] = include_bytes!("../spv/mm_f32b.spv");
/// plans/89 P0.3 — MoE direct-ids 디코드: llama dmmv 기하(64스레드·2행·
/// 서브그룹Add)에 ids 간접을 얹은 판. q4_K은 q4b 파생, q5_1은 신규(FN down
/// 질량). f32 활성 직결 — MoE quant 스킵, 산술 클래스는 gemv8 전환과 동열.
const FN_MOE_IDS2_SPV: &[u8] = include_bytes!("../spv/fn_moe_ids2.spv");
const FN_MOE_IDS51_SPV: &[u8] = include_bytes!("../spv/fn_moe_ids51.spv");

/// plans/89 P1.2 — f32/BF16 밀집 프리필 타일(fn_mm_f32 가중 t-재판독 소거).
const FN_TILE_F32_SPV: &[u8] = include_bytes!("../spv/fn_tile_f32.spv");
/// plans/89 P1.1 — 밀집 프리필 coopmat 타일(decoder ms/128 패밀리 직접 재사용).
/// 스칼라 fn_tile_q8(2818ms/청크, [ts])를 f16 coopMatMulAdd 판으로 교체.
const TILE_Q8128_SPV2: &[u8] = include_bytes!("../spv/tile_q8128.spv");
const TILE_Q8MS_SPV2: &[u8] = include_bytes!("../spv/tile_q8ms.spv");
const TILE_Q4K128_SPV2: &[u8] = include_bytes!("../spv/tile_q4k128.spv");
const TILE_Q4KMS_SPV2: &[u8] = include_bytes!("../spv/tile_q4kms.spv");

/// plans/89 P1.1b — MoE 그룹 프리필 q4_K coopmat 타일(f16 스테이징).
const FN_MOE_TILE_Q4K_CM_SPV: &[u8] = include_bytes!("../spv/fn_moe_tile_q4k_cm.spv");
/// plans/89 P1.1c — MoE q8_0/q5_K 스칼라 타일(레거시 전문가 루프 대체).
const FN_MOE_TILE_Q8_SPV: &[u8] = include_bytes!("../spv/fn_moe_tile_q8.spv");
const FN_MOE_TILE_Q5K_SPV: &[u8] = include_bytes!("../spv/fn_moe_tile_q5k.spv");
/// plans/89 P1.1d — MoE q5_1 coopmat 타일(q4k_cm 동일 골격).
const FN_MOE_TILE_Q51_CM_SPV: &[u8] = include_bytes!("../spv/fn_moe_tile_q51_cm.spv");
/// plans/89 재개 — q5_1 CM 1-서브그룹 판(64스레드 — tile128v2식 서브그룹 간
/// 경쟁 가설의 정면 검증이자 스칼라 대비 생산 후보).
const FN_MOE_TILE_Q51_SG1_SPV: &[u8] = include_bytes!("../spv/fn_moe_tile_q51_sg1.spv");
/// plans/89 재개 — q4_K CM 1-서브그룹 판(엔진 결정적 — q51_sg1로 판명).
const FN_MOE_TILE_Q4K_SG1_SPV: &[u8] = include_bytes!("../spv/fn_moe_tile_q4k_sg1.spv");
/// plans/89 재개 — q4_K 8-서브블록 스테이징 판(반복/장벽 q51_sg1과 동일).
const FN_MOE_TILE_Q4K_SG8_SPV: &[u8] = include_bytes!("../spv/fn_moe_tile_q4k_sg8.spv");
/// plans/89 재개 — QSA 프리필 디바이스 선택(토큰별 점수·비토닉 top-k).
const FN_IDX_SCORE_MT_SPV: &[u8] = include_bytes!("../spv/fn_idx_score_mt.spv");
const FN_IDX_TOPK_MT_SPV: &[u8] = include_bytes!("../spv/fn_idx_topk_mt.spv");
/// plans/89 재개 — q4_K sg1 스케일-캐시 판(레지스터 압박 가설).
const FN_MOE_TILE_Q4K_SG1SC_SPV: &[u8] = include_bytes!("../spv/fn_moe_tile_q4k_sg1sc.spv");
/// plans/89 재개 — q4_K K-병렬 스칼라 타일(서브그룹 16슬라이스 — ALU 16× 절감).
const FN_MOE_TILE_Q4K_KP_SPV: &[u8] = include_bytes!("../spv/fn_moe_tile_q4k_kp.spv");
/// plans/89 P1.4 — PLE 수학 디바이스 3커널(hip q4_ple_* 포트, 비트 동일 목표).
const FN_PLE_GATE_SPV: &[u8] = include_bytes!("../spv/fn_ple_gate.spv");
const FN_PLE_CONV_SPV: &[u8] = include_bytes!("../spv/fn_ple_conv.spv");
const FN_PLE_RES_SPV: &[u8] = include_bytes!("../spv/fn_ple_res.spv");
/// plans/89 P0.4 — QSA 선택 어텐션 멀티헤드(WG=tok×kv헤드, K/V 12× 절감).
const FN_QSA_ATTN_SEL_MH_SPV: &[u8] = include_bytes!("../spv/fn_qsa_attn_sel_mh.spv");
/// 파이프라인 세트 (vk 핸들은 복사 가능).
/// 지연 파이프라인 슬롯.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Slot {
    Gemv,

    SiluDiv,
    Scale,
    CopyRows,
    BcastRows,
    AxpyT,
    MoeTop10,
    PermuteF32,
    PermuteU32,
    MoeWsum,
    MoeGatherRows,
    HcGateMean,
    HcCombine,
    NormGatedSig,
    GdnBetaG,
    EwSigmoid,
    Split3,
    GdnConvT2,
    GdnConvState,
    GdnConvSeq,
    L2Rows,
    L2Rows2Scale,
    FnGdnArSwap,
    FnQsaAttnSel,
    FnIdxBk,
    /// plans/85 §2 — QSA 디코드 선택 체인(q_rope/score/rank/expand).
    FnIdxQRope,
    /// plans/86 §2 — QSA q/k norm+rope.
    FnQkNormRope,
    FnIdxScore,
    FnIdxRank,
    FnIdxExpand,
    Quant,
    Rms,
    /// plans/85 §2 — 프레임 로짓 행별 argmax(2단계).
    FnArgmaxRows,
    Silu,
    /// plans/88 P1 — MoE direct-ids GEMV(전 타입, gemv3 파생).
    FnMoeIds,
    /// plans/88 P1 — f32/BF16 밀집 GEMV.
    FnMmf32,
    /// plans/88 P2 — MoE 디바이스 그룹화(단일 블록).
    FnMoeGroup,
    /// plans/88 P2 — q4_K/q5_1 그룹 타일 GEMM(패딩 도메인).
    FnMoeTileQ4K,
    FnMoeTileQ51,
    /// plans/89 P0.2 — 디코드 밀집 GEMV(llama dmmv 포트 재사용).
    Gemv8Q8B,
    Gemv8Q4B,
    /// plans/89 P0.2 — f32/BF16 디코드 GEMV 64스레드 판.
    MmF32b,
    /// plans/88 P2 — q8_0 밀집 프리필 타일(K-슬라이스 스테이징).
    FnTileQ8,
    /// plans/89 P0.3 — MoE direct-ids dmmv 판(q4_K/q5_1).
    FnMoeIds2,
    FnMoeIds51,
    /// plans/89 P1.2 — f32/BF16 밀집 프리필 타일.
    FnTileF32,
    /// plans/89 P1.1 — 밀집 프리필 coopmat 타일(decoder 판 재사용).
    TileQ8128Cm,
    TileQ8msCm,
    TileQ4k128Cm,
    TileQ4kmsCm,
    /// plans/89 P1.1b — MoE q4_K coopmat 타일.
    FnMoeTileQ4kCm,
    /// plans/89 P1.1c — MoE q8_0/q5_K 스칼라 타일.
    FnMoeTileQ8,
    FnMoeTileQ5k,
    /// plans/89 P1.1d — MoE q5_1 coopmat 타일.
    FnMoeTileQ51Cm,
    /// plans/89 재개 — q5_1 CM 1-서브그룹 판(경쟁 판별·후보 생산판).
    FnMoeTileQ51Sg1,
    /// plans/89 P1.4 — PLE gate/conv/residual.
    FnPleGate,
    FnMoeTileQ4kSg1,
    FnMoeTileQ4kSg8,
    FnIdxScoreMt,
    FnIdxTopkMt,
    FnMoeTileQ4kSg1sc,
    FnMoeTileQ4kKp,
    FnPleConv,
    FnPleRes,
    /// plans/89 P0.4 — QSA 어텐션 멀티헤드 판.
    FnQsaAttnSelMh,
}
/// plans/86 §6 — 모델 파트 파일 (mmap 범위 + 핸들). 대형 가중 업로드를
/// pread 스테이징으로 수행한다(hip staged_upload 미러).
struct PartSource {
    base: usize,
    len: usize,
    file: std::fs::File,
}

pub struct VkAcc {
    pub(crate) ctx: Mutex<VkCtx>,
    sources: Vec<PartSource>,
    pipes: Mutex<HashMap<Slot, Pipes>>,
    /// 가중치 캐시 (데이터 포인터 → 상주 청크들)
    wcache: Mutex<HashMap<(usize, usize), Vec<VkBuf>>>,
    tables: Mutex<Option<(VkBuf, VkBuf)>>,
    dummy: Mutex<Option<VkBuf>>,
    // 값-경로 버퍼 (필요시 성장)
    xfbuf: Mutex<Option<VkBuf>>,
    xbuf: Mutex<Option<VkBuf>>,
    /// plans/88 P2 — 프레임 quant 출력(디바이스 로컬). gemv3/타일의 xq 재판독은
    /// n_out배 증폭이라 GTT(host-visible)에서 3-4GB/s에 갇혔다 — L2 캐시가
    /// 동작하는 디바이스 메모리로 보낸다(값경로 xbuf 는 호스트 스테이징용 유지).
    xq_dev: Mutex<Option<VkBuf>>,
    obuf: Mutex<Option<VkBuf>>,
    sbufs: Mutex<Option<(VkBuf, VkBuf, VkBuf)>>,
    rbufs: Mutex<Option<(VkBuf, VkBuf, VkBuf)>>,
    // FFN 상주 체인 버퍼 (xf, xq0, fg, fu, glu, xq1, ob)
    ffnbufs: Mutex<Option<(VkBuf, VkBuf, VkBuf, VkBuf, VkBuf, VkBuf, VkBuf)>>,
    /// 그룹 배칭 가중별 출력 슬롯 (plans/19)
    gobufs: Mutex<Vec<Option<VkBuf>>>,
    /// plans/84 B — 프레임 버퍼 레지스트리 (핸들 → 상주 버퍼; host-visible).
    pub(crate) framebufs: Mutex<HashMap<u64, VkBuf>>,
    /// plans/85 §1 — 해제된 프레임 버퍼 재활용 풀. VkBuf는 파괴자가 없어
    /// 종전 frame_free는 종료까지 누출 — 디코드 스텝 스크래치(shexp 등)가
    /// 매층·매스텝 할당되므로 상한 내에서 재활용한다.
    frame_pool: Mutex<Vec<VkBuf>>,
    /// plans/85 §2 — QSA 디코드 선택 스크래치 (iqr, scr, flg, iqw, cs, sdev, ofdev).
    pub(crate) qsa_sel_bufs: Mutex<Option<(VkBuf, VkBuf, VkBuf, VkBuf, VkBuf, VkBuf, VkBuf)>>,
    /// plans/85 §2 — 프레임 argmax 스크래치 (sc, out).
    argmax_bufs: Mutex<Option<(VkBuf, VkBuf)>>,
    /// plans/84 B — MoE 스크래치 (perm, xg, inv, yg) — 필요시 성장.
    moebufs: Mutex<Option<(VkBuf, VkBuf, VkBuf, VkBuf)>>,
     /// plans/84 B — QSA 상주 풀: (층,시퀀스) → (kv_k, kv_v, idx_k, bk) + 워터마크.
     qsa_pools: Mutex<HashMap<(usize, usize), (VkBuf, VkBuf, VkBuf, VkBuf, usize)>>,
    /// plans/86 §2 — qk_norm_rope 상수(qn/kn/cs 타일) (ptr,len) 키 상주.
    qk_consts: Mutex<HashMap<(usize, usize), VkBuf>>,
    /// plans/86 §4 — QSA 업로드 판 스크래치 (ck, cv, sel_idx, sel_off) — 성장 재할당.
    qsa_up_bufs: Mutex<Option<(VkBuf, VkBuf, VkBuf, VkBuf)>>,
    /// plans/86 §4 — idx_append ikw 고정 스크래시(호출부가 매번 새 Vec).
    qsa_ikw: Mutex<Option<VkBuf>>,
    qsa_ctx: std::sync::atomic::AtomicUsize,
    frame_next: std::sync::atomic::AtomicU64,
    frame_t: std::sync::atomic::AtomicUsize,
    /// plans/88 P1 — 프레임 스텝 배치 활성(값경로 미참조 게이트).
    frame_step_batch: std::sync::atomic::AtomicBool,
    /// plans/88 P2 — MoE 라우팅 세대(MoeTop10마다 증가 — 그룹화 캐시 키).
    moe_gen: std::sync::atomic::AtomicU64,
    /// plans/88 P2 — 그룹화 캐시: 같은 세대의 3개 GEMM이 테이블을 공유.
    moe_grp: Mutex<Option<MoeGrp>>,
    /// plans/89 P1.4 — PLE 디바이스 링: seq → (버퍼, 워터마크 t).
    ple_rings: Mutex<std::collections::HashMap<usize, (VkBuf, usize)>>,
    /// plans/89 P1.4 — PLE 상수 캐시: (ptr,len) → 버퍼(모델 가중 뷰라 안정).
    ple_consts: Mutex<std::collections::HashMap<(usize, usize), VkBuf>>,
}

/// plans/88 P2 — MoE 그룹화 상주 자산(디바이스 테이블 + 스크래치).
struct MoeGrp {
    generation: u64,
    rows: usize,
    ids_h: u64,
    bound: usize,
    /// off에 할당된 전문가 수(성장 가드 — 세대 무관 재사용).
    off_n: usize,
    off: VkBuf,
    rows_pad: VkBuf,
    tilexp: VkBuf,
    perm: VkBuf,
    inv: VkBuf,
    inv_pad: VkBuf,
    rowexp: VkBuf,
    perm_pad: VkBuf,
    yg: VkBuf,
    yg_rows: usize,
}

/// 빈 VkBuf 자리표 — 풀 엔트리 지연 생성용.
fn vkbuf_null() -> VkBuf {
    VkBuf { buf: vk::Buffer::null(), ptr: std::ptr::null_mut(), bytes: 0, mem: vk::DeviceMemory::null() }
}
fn vk_ty(ty: GgmlType) -> Option<u32> {
    match ty {
        GgmlType::Q5K => Some(13),
        GgmlType::Q5_1 => Some(7),
        GgmlType::Q4K => Some(12),
        GgmlType::Q6K => Some(14),
        GgmlType::Iq4Xs => Some(23),
        GgmlType::Q8_0 => Some(8),
        GgmlType::Iq4Nl => Some(20),
        GgmlType::Q3K => Some(11),
        GgmlType::Iq3S => Some(21),
        _ => None,
    }
}

/// q8_0 양자화 활성 워드 수 — quant 커널 스트라이드(정렬 여유 포함).
/// 복제 산식 통합(plans/90 A2): rawhip q4acc::xq_words 와 동일식.
pub(crate) fn xq_words(n: usize) -> usize {
    n / 4 + n / 32 + n / 16
}

/// plans/87 §2/§3 — 슬롯 → op 태그(와치독 링·ts 라벨).
fn slot_name(slot: Slot) -> &'static str {
    match slot {
        Slot::Gemv => "gemv",

        Slot::SiluDiv => "silu_div",
        Slot::Scale => "scale",
        Slot::CopyRows => "copy_rows",
        Slot::BcastRows => "bcast_rows",
        Slot::AxpyT => "axpy_t",
        Slot::Rms => "rms",
        Slot::Silu => "silu_mul",
        Slot::EwSigmoid => "sigmoid",
        Slot::HcGateMean => "hc_gate_mean",
        Slot::HcCombine => "hc_combine",
        Slot::NormGatedSig => "norm_gated",
        Slot::GdnBetaG => "gdn_beta_g",
        Slot::Split3 => "split3",
        Slot::L2Rows => "l2_rows",
        Slot::L2Rows2Scale => "l2_rows2_scale",
        Slot::FnGdnArSwap => "gdn_ar_swap",
        Slot::FnQsaAttnSel => "qsa_attn_sel",
        Slot::PermuteU32 => "permute_u32",
        Slot::PermuteF32 => "permute_f32",
        Slot::MoeTop10 => "moe_top10",
        Slot::MoeWsum => "moe_wsum",
        Slot::MoeGatherRows => "moe_gather",
        Slot::GdnConvT2 => "gdn_conv_t2",
        Slot::GdnConvState => "gdn_conv_state",
        Slot::GdnConvSeq => "gdn_conv_seq",
        Slot::FnIdxScore => "idx_score",
        Slot::FnIdxBk => "idx_bk_update",
        Slot::FnIdxQRope => "idx_q_rope",
        Slot::FnIdxRank => "idx_rank",
        Slot::FnIdxExpand => "idx_expand",
        Slot::FnQkNormRope => "qk_norm_rope",
        Slot::Quant => "quant",
        Slot::FnArgmaxRows => "argmax_rows",
        Slot::Gemv8Q8B => "gemv8_q8b",
        Slot::Gemv8Q4B => "gemv8_q4b",
        Slot::MmF32b => "mm_f32b",
        Slot::TileQ8128Cm => "tile_q8128",
        Slot::TileQ8msCm => "tile_q8ms",
        Slot::TileQ4k128Cm => "tile_q4k128",
        Slot::TileQ4kmsCm => "tile_q4kms",
        Slot::FnMoeIds => "moe_ids",
        Slot::FnMmf32 => "mm_f32",
        Slot::FnMoeIds2 => "moe_ids2",
        Slot::FnPleGate => "ple_gate",
        Slot::FnQsaAttnSelMh => "qsa_attn_sel_mh",
        Slot::FnPleConv => "ple_conv",
        Slot::FnPleRes => "ple_res",
        Slot::FnMoeIds51 => "moe_ids51",
        Slot::FnMoeGroup => "moe_group",
        Slot::FnMoeTileQ4K => "moe_tile_q4k",
        Slot::FnTileF32 => "tile_f32",
        Slot::FnMoeTileQ8 => "moe_tile_q8",
        Slot::FnMoeTileQ5k => "moe_tile_q5k",
        Slot::FnMoeTileQ51Cm => "moe_tile_q51_cm",
        Slot::FnMoeTileQ51Sg1 => "moe_tile_q51_sg1",
        Slot::FnMoeTileQ4kSg1 => "moe_tile_q4k_sg1",
        Slot::FnMoeTileQ4kSg8 => "moe_tile_q4k_sg8",
        Slot::FnIdxScoreMt => "idx_score_mt",
        Slot::FnIdxTopkMt => "idx_topk_mt",
        Slot::FnMoeTileQ4kSg1sc => "moe_tile_q4k_sg1sc",
        Slot::FnMoeTileQ4kKp => "moe_tile_q4k_kp",
        Slot::FnMoeTileQ51 => "moe_tile_q51",
        Slot::FnMoeTileQ4kCm => "moe_tile_q4k_cm",
        Slot::FnTileQ8 => "tile_q8",
     }
 }

pub(crate) fn push_u32s(vals: &[u32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(vals.len() * 4);
    for x in vals {
        v.extend_from_slice(&x.to_le_bytes());
    }
    v
}

/// 프레임 재활용 풀 상한(바이트) — 초과 해제분은 종전대로 보류(파괴 없음).
const FRAME_POOL_CAP: usize = 64 << 20;
impl VkAcc {
    pub fn new() -> Result<Self, String> {
        Self::new_with_sources(Vec::new())
    }

    /// 파트 소스 지정판 — `Model4::part_sources()` (plans/86 §6).
    pub fn new_with_sources(parts: Vec<(usize, usize, std::path::PathBuf)>) -> Result<Self, String> {
        llm170_diag::alloc::set_on(llm170_diag::dump::opts().alloc);
        llm170_diag::alloc::set_vaddr(llm170_diag::dump::opts().vaddr);
        let sources = parts
            .into_iter()
            .filter_map(|(base, len, path)| {
                std::fs::File::open(&path).ok().map(|file| PartSource { base, len, file })
            })
            .collect();
        let ctx = VkCtx::new()?;
        if !ctx.coop_matrix {
            eprintln!("rawvk: coop matrix 미지원 (타일 경로 M3에서 필요)");
        }
        Ok(Self {
            ctx: Mutex::new(ctx),
            sources,
            pipes: Mutex::new(HashMap::new()),
            wcache: Mutex::new(HashMap::new()),
            tables: Mutex::new(None),
            dummy: Mutex::new(None),
            xfbuf: Mutex::new(None),
            xbuf: Mutex::new(None),
            xq_dev: Mutex::new(None),
            obuf: Mutex::new(None),
            sbufs: Mutex::new(None),
            rbufs: Mutex::new(None),
            ffnbufs: Mutex::new(None),
            gobufs: Mutex::new(Vec::new()),
            framebufs: Mutex::new(HashMap::new()),
            frame_pool: Mutex::new(Vec::new()),
            moebufs: Mutex::new(None),
            qsa_sel_bufs: Mutex::new(None),
            argmax_bufs: Mutex::new(None),
            qk_consts: Mutex::new(HashMap::new()),
            qsa_up_bufs: Mutex::new(None),
            qsa_ikw: Mutex::new(None),
            qsa_pools: Mutex::new(HashMap::new()),
            qsa_ctx: std::sync::atomic::AtomicUsize::new(0),
            frame_next: std::sync::atomic::AtomicU64::new(1),
            frame_t: std::sync::atomic::AtomicUsize::new(1),
            frame_step_batch: std::sync::atomic::AtomicBool::new(false),
            moe_gen: std::sync::atomic::AtomicU64::new(0),
            moe_grp: Mutex::new(None),
            ple_rings: Mutex::new(std::collections::HashMap::new()),
            ple_consts: Mutex::new(std::collections::HashMap::new()),
        })
    }

    // ─── 지연 초기화 공용 자원 ───

    pub(crate) fn pipeline(&self, ctx: &mut VkCtx, slot: Slot) -> Result<Pipes, String> {
        crate::rawvk::context::site::set_tag(slot_name(slot));
        if let Some(&p) = self.pipes.lock().get(&slot) {
            return Ok(p);
        }
        let (spv, n_buf, pb) = match slot {
            Slot::Gemv => (GEMV_SPV, 12, 24u32),

            Slot::SiluDiv => (SILU_DIV_SPV, 1, 8),    // u32 + f32
            Slot::Scale => (SCALE_SPV, 1, 8),         // u32 + f32
            Slot::CopyRows => (COPY_ROWS_SPV, 2, 12), // 3×u32
            Slot::BcastRows => (BCAST_ROWS_SPV, 2, 8),// 2×u32
            Slot::AxpyT => (AXPY_T_SPV, 3, 8),        // 2×u32
            Slot::MoeTop10 => (MOE_TOP10_SPV, 3, 8),  // 2×u32
            Slot::PermuteF32 => (PERMUTE_SPV, 3, 8),  // 2×u32
            Slot::MoeWsum => (MOE_WSUM_SPV, 3, 12),   // 3×u32
            Slot::MoeGatherRows => (MOE_GATHER_SPV, 2, 12),
            Slot::HcGateMean => (HC_GATE_MEAN_SPV, 3, 12),
            Slot::HcCombine => (HC_COMBINE_SPV, 3, 12),
            Slot::NormGatedSig => (NORM_GATED_SIG_SPV, 4, 16),  // u32,u32,f32
            Slot::GdnBetaG => (GDN_BETA_G_SPV, 5, 8),
            Slot::EwSigmoid => (EW_SIGMOID_SPV, 1, 4),
            Slot::Split3 => (SPLIT3_SPV, 4, 12),      // 기존 q35 값경로 판 재사용
            Slot::GdnConvT2 => (GDN_CONV_T2_SPV, 4, 12),
            Slot::GdnConvState => (GDN_CONV_ST_SPV, 2, 12),
            Slot::GdnConvSeq => (GDN_CONV_SEQ_SPV, 4, 12),
            Slot::L2Rows => (L2_ROWS_SPV, 1, 12),          // u32 + f32
            Slot::L2Rows2Scale => (L2_ROWS2_SPV, 2, 24),   // u32,u32,f32,f32
            Slot::FnGdnArSwap => (FN_GDN_AR_SWAP_SPV, 6, 28),
            Slot::FnQsaAttnSel => (FN_QSA_ATTN_SEL_SPV, 6, 24),
            Slot::PermuteU32 => (PERMUTE_U32_SPV, 3, 12),  // row_src,row_dst,rows
            Slot::FnIdxScore => (FN_IDX_SCORE_SPV, 3, 12), // 3×u32
            Slot::FnIdxBk => (FN_IDX_BK_SPV, 4, 20),         // f32 + 3×u32
            Slot::FnIdxQRope => (FN_IDX_Q_ROPE_SPV, 4, 8),    // f32 + u32
            Slot::FnQkNormRope => (FN_QK_NORM_ROPE_SPV, 5, 28), // 2×f32 + 5×u32
            Slot::FnIdxRank => (FN_IDX_RANK_SPV, 2, 8),      // 2×u32
            Slot::FnIdxExpand => (FN_IDX_EXPAND_SPV, 3, 16), // 4×u32
            Slot::FnArgmaxRows => (FN_ARGMAX_ROWS_SPV, 3, 12), // 3×u32
            Slot::Quant => (QUANT_SPV, 2, 12),
            Slot::Rms => (RMS_SPV, 3, 16),   // plans/84 B: w_reps 추가(기본 1 = 종전 산술)
            Slot::Silu => (SILU_SPV, 3, 4),
            Slot::Gemv8Q8B => (GEMV8_Q8B_SPV, 10, 24),  // 8W+x(f32)+out (W0만 사용)
            Slot::Gemv8Q4B => (GEMV8_Q4B_SPV, 10, 24),
            Slot::MmF32b => (MM_F32B_SPV, 10, 20),      // 8W+x(f32)+out
            Slot::FnMoeIds => (FN_MOE_IDS_SPV, 13, 28), // 8W+xq+out+ktab+grid+ids
            Slot::FnMoeIds2 => (FN_MOE_IDS2_SPV, 11, 24),   // 8W+x(f32)+out+ids
            Slot::FnMoeIds51 => (FN_MOE_IDS51_SPV, 11, 24),
            Slot::FnMmf32 => (FN_MM_F32_SPV, 10, 20),    // 8W+x(f32)+out
            Slot::FnMoeGroup => (FN_MOE_GROUP_SPV, 9, 12),        // 3×u32
            Slot::FnMoeTileQ4K => (FN_MOE_TILE_Q4K_SPV, 13, 28),  // 8W+xq+yg+rowexp+rp+perm_pad +mode+rows
            Slot::FnMoeTileQ51 => (FN_MOE_TILE_Q51_SPV, 13, 28),  // +mode+rows
            Slot::FnTileQ8 => (FN_TILE_Q8_SPV, 10, 20),  // 8W+xq+out
            Slot::FnTileF32 => (FN_TILE_F32_SPV, 10, 20),    // 8W+x(f32)+out
            Slot::FnPleGate => (FN_PLE_GATE_SPV, 8, 16),   // res,key,val,nk,nq,nc,gated,gate
            Slot::FnPleConv => (FN_PLE_CONV_SPV, 4, 20),   // gated,cw,ring,conv
            Slot::FnQsaAttnSelMh => (FN_QSA_ATTN_SEL_MH_SPV, 6, 20),
            Slot::FnPleRes => (FN_PLE_RES_SPV, 4, 12),     // res,val,gate,conv
            Slot::TileQ8128Cm => (TILE_Q8128_SPV2, 10, 24),
            Slot::TileQ8msCm => (TILE_Q8MS_SPV2, 10, 20),
            Slot::TileQ4k128Cm => (TILE_Q4K128_SPV2, 10, 24),
            Slot::TileQ4kmsCm => (TILE_Q4KMS_SPV2, 10, 20),
            Slot::FnMoeTileQ4kCm => (FN_MOE_TILE_Q4K_CM_SPV, 13, 28),
            Slot::FnMoeTileQ8 => (FN_MOE_TILE_Q8_SPV, 13, 28),
            Slot::FnMoeTileQ5k => (FN_MOE_TILE_Q5K_SPV, 13, 28),
            Slot::FnMoeTileQ51Cm => (FN_MOE_TILE_Q51_CM_SPV, 13, 28),
            Slot::FnMoeTileQ51Sg1 => (FN_MOE_TILE_Q51_SG1_SPV, 13, 28),
            Slot::FnMoeTileQ4kSg1 => (FN_MOE_TILE_Q4K_SG1_SPV, 13, 28),
            Slot::FnMoeTileQ4kSg8 => (FN_MOE_TILE_Q4K_SG8_SPV, 13, 28),
            Slot::FnIdxScoreMt => (FN_IDX_SCORE_MT_SPV, 3, 20),
            Slot::FnIdxTopkMt => (FN_IDX_TOPK_MT_SPV, 3, 20),
            Slot::FnMoeTileQ4kSg1sc => (FN_MOE_TILE_Q4K_SG1SC_SPV, 13, 28),
            Slot::FnMoeTileQ4kKp => (FN_MOE_TILE_Q4K_KP_SPV, 13, 28),
        };
        let p = ctx.pipeline_pipes(spv, n_buf, pb)?;
        self.pipes.lock().insert(slot, p);
        Ok(p)
    }

    /// ktab(iq4nl)·grid3s 테이블 + 더미 버퍼 — 최초 1회 업로드.
    pub(crate) fn ensure_shared(&self, ctx: &mut VkCtx) -> Result<(vk::Buffer, vk::Buffer, vk::Buffer), String> {
        if self.tables.lock().is_none() {
            let kv: Vec<u32> = llm170_core::ktab2_packed();
            let kb = ctx.alloc_host(1024)?;
            unsafe { std::ptr::copy_nonoverlapping(kv.as_ptr() as *const u8, kb.ptr, 1024) };
            let gb = ctx.alloc_host(2048)?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    llm170_core::IQ3S_GRID.as_ptr() as *const u8,
                    gb.ptr,
                    2048,
                );
            }
            *self.tables.lock() = Some((kb, gb));
        }
        if self.dummy.lock().is_none() {
            *self.dummy.lock() = Some(ctx.alloc_host(16)?);
        }
        let t = self.tables.lock();
        let (a, b) = t.as_ref().unwrap();
        Ok((a.buf, b.buf, self.dummy.lock().as_ref().unwrap().buf))
    }

    /// plans/89 P0.2 — 디코드(t<16) 밀집 GEMV: llama dmmv 포트(q8b/q4b)를
    /// 프레임 f32 활성 버퍼에 직결. 64스레드 2행 WG·서브그룹Add — 27B 경로
    /// 실측 272-329GB/s. 절대 인덱싱이라 단일 청크 가중만(이 장치 max_ssbo
    /// 4GiB — FN 밀집 전부 단일 청크). 미해당 타입은 false 반환(호출부 폴백).
    fn gemv8_dense(
        &self,
        ctx: &mut VkCtx,
        wbufs: &[vk::Buffer],
        n_in: usize,
        n_out: usize,
        t: usize,
        ty: u32,
        xb: vk::Buffer,
        ob: vk::Buffer,
    ) -> Result<bool, String> {
        if wbufs.len() != 1 || t >= 16 {
            return Ok(false);
        }
        let slot = match ty {
            8 => Slot::Gemv8Q8B,
            12 => Slot::Gemv8Q4B,
            _ => return Ok(false),
        };
        let (_, _, dbuf) = self.ensure_shared(ctx)?;
        let p = self.pipeline(ctx, slot)?;
        let mut binds: Vec<vk::Buffer> = wbufs.to_vec();
        while binds.len() < 8 {
            binds.push(dbuf);
        }
        binds.push(xb);
        binds.push(ob);
        let ds2 = ctx.bind_ds(&p, &binds)?;
        let push = push_u32s(&[n_in as u32, n_out as u32, t as u32, 0, 0, 2]);
        ctx.run(p.pl, ds2, p.pipe, &push, 1, n_out.div_ceil(2) as u32, t as u32)?;
        Ok(true)
    }

    /// 가중치 상주 ((ptr,len) 키 — mmap 안정) — max_ssbo 청크.
    /// plans/86 §1b: ptr 단독 키는 같은 기저의 슬라이스 뷰(전문가 1개분)과
    /// 전체 스택을 혼동한다 — 값경로 프리필이 적재한 1전문가 버퍼를 프레임
    /// MoE가 오프셋 재결합하면 GPUVM PERMISSION 폴트. (ptr,len)으로 구분.
    fn weight_bufs(&self, ctx: &mut VkCtx, w: &Weight) -> Result<Vec<vk::Buffer>, String> {
        let key = (w.data.as_ptr() as usize, w.data.len());
        crate::rawvk::context::site::scope("weight", || {
        {
            let mut wc = self.wcache.lock();
            if let std::collections::hash_map::Entry::Vacant(e) = wc.entry(key) {
                let ch = ctx.max_ssbo; // plans/29: 균일 청크 — 크기는 push(chunk_words)로 전달
                let total = w.data.len();
                let mut bufs = Vec::new();
                let mut off = 0usize;
                // plans/86 §6 — pread 스테이징: 알려진 파트 범위면 mmap 폴트
                // (4KB 랜덤, 20-180 MB/s) 대신 파일에서 8MiB 순차 pread로
                // 매핑 버퍼에 직접 채운다(실측 ~1.2 GB/s). 아니면 memcpy 폴백.
                let src = self.sources.iter().find(|s| {
                    let p = w.data.as_ptr() as usize;
                    p >= s.base && p.checked_add(total).is_some_and(|e| e <= s.base + s.len)
                });
                while off < total {
                    let n = ch.min(total - off);
                    let mut b = ctx.alloc(n)?;
                    let r = match src {
                        Some(s) => staged_fill(&s.file, b.ptr, (w.data.as_ptr() as usize - s.base) as u64 + off as u64, n),
                        None => unsafe {
                            std::ptr::copy_nonoverlapping(w.data.as_ptr().add(off), b.ptr, n);
                            Ok(())
                        },
                    };
                    if let Err(e) = r {
                        return Err(format!("가중 스테이징: {e}"));
                    }
                    ctx.unmap(&mut b)?; // WC 매핑 즉시 해제 — op당 동기 비용 방지
                    bufs.push(b);
                    off += n;
                }
                e.insert(bufs);
            }
        }
            let bufs: Vec<vk::Buffer> = {
                let wc = self.wcache.lock();
                wc.get(&key).unwrap().iter().map(|b| b.buf).collect()
            };
            if bufs.len() > 8 {
                return Err(format!("가중 청크 {}개 > 8 슬롯 (M2 한계)", bufs.len()));
            }
            Ok(bufs)
        })
    }

    /// GEMV 1회 발사: 가중 청크(8) + xq + out + ktab + grid = 12 바인딩.
    #[allow(clippy::too_many_arguments)]
    fn gemv_run(
        &self,
        ctx: &mut VkCtx,
        wbufs: &[vk::Buffer],
        n_in: usize,
        n_out: usize,
        xq_w: usize,
        ty: u32,
        t: usize,
        xq_buf: vk::Buffer,
        out_buf: vk::Buffer,
    ) -> Result<(), String> {
        if std::env::var_os("LLM170_VK_GVDBG").is_some() {
            eprintln!("[gv] ty={ty} n_in={n_in} n_out={n_out} t={t}");
        }
        let (kb, gb, dbuf) = self.ensure_shared(ctx)?;
        let p = self.pipeline(ctx, Slot::Gemv)?;
        let mut binds: Vec<vk::Buffer> = wbufs.to_vec();
        while binds.len() < 8 {
            binds.push(dbuf);
        }
        binds.push(xq_buf);
        binds.push(out_buf);
        binds.push(kb);
        binds.push(gb);
        let ds2 = ctx.bind_ds(&p, &binds)?;
        // plans/29: 균일 청크 워드 수 (weight_bufs가 max_ssbo 단위로 분할).
        let chunk_words = (ctx.max_ssbo / 4) as u32;
        let push = push_u32s(&[n_in as u32, n_out as u32, xq_w as u32, ty, t as u32, chunk_words]);
        ctx.run(p.pl, ds2, p.pipe, &push, n_out as u32, 1, 1)
    }


    /// plans/84 B — 오프셋 지원 GEMV: MoE 전문가 슬라이스(xq 행 구간, 가중
    /// 전문가 오프셋, out 행 구간). 산술은 gemv_run과 동일 커널.
    #[allow(clippy::too_many_arguments)]
    fn gemv_run_off(
        &self,
        ctx: &mut VkCtx,
        wbufs: &[vk::Buffer],
        n_in: usize,
        n_out: usize,
        xq_w: usize,
        ty: u32,
        t: usize,
        xq_buf: vk::Buffer,
        out_buf: vk::Buffer,
        xq_off: u64,
        out_off: u64,
        w_off: u64,
    ) -> Result<(), String> {
        let (kb, gb, dbuf) = self.ensure_shared(ctx)?;
        let p = self.pipeline(ctx, Slot::Gemv)?;
        let mut binds: Vec<(vk::Buffer, u64)> = wbufs
            .iter()
            .enumerate()
            .map(|(i, &b)| (b, if i == 0 { w_off } else { 0 }))
            .collect();
        while binds.len() < 8 {
            binds.push((dbuf, 0));
        }
        binds.push((xq_buf, xq_off));
        binds.push((out_buf, out_off));
        binds.push((kb, 0));
        binds.push((gb, 0));
        // 배치 모드: 녹화 중 p.ds 재기입은 불법 — 전문가별 신규 세트(배치 풀,
        // end_batch_wait에서 일괄 해제). 비배치는 그대로 전용 세트.
        let ds2 = if ctx.batching.load(std::sync::atomic::Ordering::Relaxed) {
            let ds = ctx.fresh_ds_for(&p, 11)?;
            ctx.bind_bufs_off(ds, &binds);
            ds
        } else {
            ctx.bind_bufs_off(p.ds, &binds);
            p.ds
        };
        let chunk_words = (ctx.max_ssbo / 4) as u32;
        let push = push_u32s(&[n_in as u32, n_out as u32, xq_w as u32, ty, t as u32, chunk_words]);
        ctx.run(p.pl, ds2, p.pipe, &push, n_out as u32, 1, 1)
    }

    /// xs(f32) 업로드 → quant_q8 → xq 버퍼 (값 버퍼 자동 성장).
    pub(crate) fn quant_upload(
        &self,
        ctx: &mut VkCtx,
        xs: &[Vec<f32>],
        n_in: usize,
        xq_buf: vk::Buffer,
    ) -> Result<(), String> {
        let t = xs.len();
        {
            let mut xf = self.xfbuf.lock();
            let need = t * n_in * 4;
            if !xf.as_ref().map(|b| b.bytes >= need).unwrap_or(false) {
                *xf = Some(crate::rawvk::context::site::scope("value_stage", || ctx.alloc_host(need.max(1 << 21)))?);
            }
            let b = xf.as_ref().unwrap();
            for (ti, row) in xs.iter().enumerate() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        row.as_ptr(),
                        b.ptr.add(ti * n_in * 4) as *mut f32,
                        n_in,
                    );
                }
            }
        }
        let xfbuf = self.xfbuf.lock().as_ref().unwrap().buf;
        let p = self.pipeline(ctx, Slot::Quant)?;
        let ds2 = ctx.bind_ds(&p, &[xfbuf, xq_buf])?;
        let xq_w = xq_words(n_in);
        let push = push_u32s(&[n_in as u32, t as u32, xq_w as u32]);
        ctx.run(p.pl, ds2, p.pipe, &push, ((n_in / 32) + 63) as u32 / 64, t as u32, 1)
    }

    /// 값 버퍼 확보 (필요시 성장) → 핸들 반환.
    fn value_buf(&self, ctx: &mut VkCtx, slot: &Mutex<Option<VkBuf>>, need: usize) -> Result<vk::Buffer, String> {
        let mut g = slot.lock();
        if !g.as_ref().map(|b| b.bytes >= need).unwrap_or(false) {
            *g = Some(crate::rawvk::context::site::scope("value_stage", || ctx.alloc_host(need.max(1 << 21)))?);
        }
        Ok(g.as_ref().unwrap().buf)
    }

    /// out 버퍼에서 호스트 행 복사.
    fn download_out(&self, outs: &mut [Vec<f32>], n_out: usize, t: usize) {
        let ob = self.obuf.lock();
        let host = unsafe { std::slice::from_raw_parts(ob.as_ref().unwrap().ptr as *const f32, t * n_out) };
        for ti in 0..t {
            outs[ti].copy_from_slice(&host[ti * n_out..(ti + 1) * n_out]);
        }
    }
}

unsafe impl Send for VkAcc {}
unsafe impl Sync for VkAcc {}
mod dispatch;
mod frame;
mod matmul;
mod ple;
mod qsa;

/// plans/86 §6 — 8MiB 순차 pread로 매핑 버퍼 채우기 (hip staged_upload 미러).
pub(crate) fn staged_fill(
    file: &std::fs::File,
    dst: *mut u8,
    mut off: u64,
    len: usize,
) -> Result<(), String> {
    use std::os::unix::fs::FileExt;
    const CH: usize = 8 << 20;
    let mut done = 0usize;
    while done < len {
        let n = CH.min(len - done);
        unsafe {
            file.read_exact_at(std::slice::from_raw_parts_mut(dst.add(done), n), off)
                .map_err(|e| format!("pread {off}: {e}"))?;
        }
        done += n;
        off += n as u64;
    }
    Ok(())
}

