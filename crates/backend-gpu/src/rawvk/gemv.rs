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

pub const GEMV_SPV: &[u8] = include_bytes!("spv/gemv3.spv");
const TILE128_SPV: &[u8] = include_bytes!("spv/tile128_q5k.spv");
/// plans/84 B: q5_1 타일판 — 22B 블록 바이트 조립(WGB), 산술은 gemv3 ty=7과 동일열.
const TILE128_Q51_SPV: &[u8] = include_bytes!("spv/tile128_q51.spv");
pub const QUANT_SPV: &[u8] = include_bytes!("spv/quant_q8.spv");
pub const ARGMAX2_SPV: &[u8] = include_bytes!("spv/argmax2.spv");
pub const RMS_SPV: &[u8] = include_bytes!("spv/rms.spv");
pub const SILU_SPV: &[u8] = include_bytes!("spv/silu_mul.spv");
/// plans/84 B — vk 프레임 경로 유틸 셰이더군.
const SILU_DIV_SPV: &[u8] = include_bytes!("spv/silu_div.spv");
const SCALE_SPV: &[u8] = include_bytes!("spv/scale.spv");
const COPY_ROWS_SPV: &[u8] = include_bytes!("spv/copy_rows.spv");
const BCAST_ROWS_SPV: &[u8] = include_bytes!("spv/bcast_rows.spv");
const AXPY_T_SPV: &[u8] = include_bytes!("spv/axpy_scaled_t.spv");
/// plans/84 B — MoE 프레임 ops.
const MOE_TOP10_SPV: &[u8] = include_bytes!("spv/moe_top10.spv");
const PERMUTE_SPV: &[u8] = include_bytes!("spv/permute_rows.spv");
const PERMUTE_U32_SPV: &[u8] = include_bytes!("spv/permute_rows_u32.spv");
const MOE_WSUM_SPV: &[u8] = include_bytes!("spv/moe_wsum.spv");
const MOE_GATHER_SPV: &[u8] = include_bytes!("spv/moe_gather.spv");
/// plans/84 B — 어텐션 반쪽 프레임 ops.
const HC_GATE_MEAN_SPV: &[u8] = include_bytes!("spv/hc_gate_mean.spv");
const HC_COMBINE_SPV: &[u8] = include_bytes!("spv/hc_combine.spv");
const NORM_GATED_SIG_SPV: &[u8] = include_bytes!("spv/norm_gated_sig.spv");
const GDN_BETA_G_SPV: &[u8] = include_bytes!("spv/gdn_beta_g.spv");
const EW_SIGMOID_SPV: &[u8] = include_bytes!("spv/ew_sigmoid.spv");
const SPLIT3_SPV: &[u8] = include_bytes!("spv/split3.spv");
const GDN_CONV_T2_SPV: &[u8] = include_bytes!("spv/gdn_conv_t2.spv");
const GDN_CONV_ST_SPV: &[u8] = include_bytes!("spv/fn_gdn_conv_state.spv");
const GDN_CONV_SEQ_SPV: &[u8] = include_bytes!("spv/gdn_conv_seq.spv");
const L2_ROWS_SPV: &[u8] = include_bytes!("spv/l2_rows.spv");
const L2_ROWS2_SPV: &[u8] = include_bytes!("spv/l2_rows2_scale.spv");
/// plans/84 B — FN GDN AR: 전치 상태(gdn_ar_w_swap 동일열).
const FN_GDN_AR_SWAP_SPV: &[u8] = include_bytes!("spv/fn_gdn_ar_swap.spv");
/// plans/84 B — FN QSA: 선택 어텐션 + 인덱서 블록키 갱신.
const FN_QSA_ATTN_SEL_SPV: &[u8] = include_bytes!("spv/fn_qsa_attn_sel.spv");
const FN_IDX_BK_SPV: &[u8] = include_bytes!("spv/fn_idx_bk_update.spv");
/// plans/86 §2 — QSA q/k norm+rope (qk_norm_rope 동일열).
const FN_QK_NORM_ROPE_SPV: &[u8] = include_bytes!("spv/fn_qk_norm_rope.spv");
/// plans/85 §2 — FN QSA 디코드 선택: q norm+rope → 블록 점수 → 순위 → 전개.
const FN_IDX_Q_ROPE_SPV: &[u8] = include_bytes!("spv/fn_idx_q_rope.spv");
const FN_IDX_SCORE_SPV: &[u8] = include_bytes!("spv/fn_idx_score.spv");
const FN_IDX_RANK_SPV: &[u8] = include_bytes!("spv/fn_idx_rank.spv");
const FN_IDX_EXPAND_SPV: &[u8] = include_bytes!("spv/fn_idx_expand.spv");
/// plans/85 §2 — 프레임 로짓 행별 GPU argmax(동률 최저 인덱스).
const FN_ARGMAX_ROWS_SPV: &[u8] = include_bytes!("spv/fn_argmax_rows.spv");
/// plans/88 P2 — MoE 그룹 프리필: 디바이스 그룹화·q4_K/q5_1 타일·융합 산란.
const FN_MOE_GROUP_SPV: &[u8] = include_bytes!("spv/fn_moe_group.spv");
const FN_MOE_TILE_Q4K_SPV: &[u8] = include_bytes!("spv/fn_moe_tile_q4k.spv");
const FN_MOE_TILE_Q51_SPV: &[u8] = include_bytes!("spv/fn_moe_tile_q51.spv");
const FN_TILE_Q8_SPV: &[u8] = include_bytes!("spv/fn_tile_q8.spv");
/// plans/88 P1 — f32/BF16 밀집 GEMV(mm_group F32·BF16 멤버 — 값폴백 소거).
const FN_MM_F32_SPV: &[u8] = include_bytes!("spv/fn_mm_f32.spv");
/// plans/88 P1 — MoE direct-ids GEMV: gemv3의 ids 구동판(그리드 (n_out, rows),
/// 워크그룹=행, ids[r]로 전문가 베이스 산출). 행 산술은 gemv3와 비트 동일.
const FN_MOE_IDS_SPV: &[u8] = include_bytes!("spv/fn_moe_ids.spv");
/// plans/89 P0.2 — 디코드 밀집 GEMV: decoder gemv8 패밀리(llama dmmv 포트)를
/// VkAcc(프레임)에서도 직접 발사. f32 활성 직결(quant 스킵)이라 W4A8 경로와
/// 산술 클래스가 다르다 — ckdiff·게이트 재기록 절차로 수용(원장 31 전례).
const GEMV8_Q8B_SPV: &[u8] = include_bytes!("spv/gemv8_q8b.spv");
const GEMV8_Q4B_SPV: &[u8] = include_bytes!("spv/gemv8_q4b.spv");
/// plans/89 P0.2 — f32/BF16 디코드 GEMV(라우터·sh-gate): fn_mm_f32(256스레드
/// f64 트리, 512WG 지연바운드 — 실측 ~0.4GB/s급)의 64스레드 서브그룹Add 판.
const MM_F32B_SPV: &[u8] = include_bytes!("spv/mm_f32b.spv");
/// plans/89 P0.3 — MoE direct-ids 디코드: llama dmmv 기하(64스레드·2행·
/// 서브그룹Add)에 ids 간접을 얹은 판. q4_K은 q4b 파생, q5_1은 신규(FN down
/// 질량). f32 활성 직결 — MoE quant 스킵, 산술 클래스는 gemv8 전환과 동열.
const FN_MOE_IDS2_SPV: &[u8] = include_bytes!("spv/fn_moe_ids2.spv");
const FN_MOE_IDS51_SPV: &[u8] = include_bytes!("spv/fn_moe_ids51.spv");

/// plans/89 P1.2 — f32/BF16 밀집 프리필 타일(fn_mm_f32 가중 t-재판독 소거).
const FN_TILE_F32_SPV: &[u8] = include_bytes!("spv/fn_tile_f32.spv");
/// plans/89 P1.1 — 밀집 프리필 coopmat 타일(decoder ms/128 패밀리 직접 재사용).
/// 스칼라 fn_tile_q8(2818ms/청크, [ts])를 f16 coopMatMulAdd 판으로 교체.
const TILE_Q8128_SPV2: &[u8] = include_bytes!("spv/tile_q8128.spv");
const TILE_Q8MS_SPV2: &[u8] = include_bytes!("spv/tile_q8ms.spv");
const TILE_Q4K128_SPV2: &[u8] = include_bytes!("spv/tile_q4k128.spv");
const TILE_Q4KMS_SPV2: &[u8] = include_bytes!("spv/tile_q4kms.spv");

/// plans/89 P1.1b — MoE 그룹 프리필 q4_K coopmat 타일(f16 스테이징).
const FN_MOE_TILE_Q4K_CM_SPV: &[u8] = include_bytes!("spv/fn_moe_tile_q4k_cm.spv");
/// plans/89 P1.1c — MoE q8_0/q5_K 스칼라 타일(레거시 전문가 루프 대체).
const FN_MOE_TILE_Q8_SPV: &[u8] = include_bytes!("spv/fn_moe_tile_q8.spv");
const FN_MOE_TILE_Q5K_SPV: &[u8] = include_bytes!("spv/fn_moe_tile_q5k.spv");
/// plans/89 P1.1d — MoE q5_1 coopmat 타일(q4k_cm 동일 골격).
const FN_MOE_TILE_Q51_CM_SPV: &[u8] = include_bytes!("spv/fn_moe_tile_q51_cm.spv");
/// plans/89 P1.4 — PLE 수학 디바이스 3커널(hip q4_ple_* 포트, 비트 동일 목표).
const FN_PLE_GATE_SPV: &[u8] = include_bytes!("spv/fn_ple_gate.spv");
const FN_PLE_CONV_SPV: &[u8] = include_bytes!("spv/fn_ple_conv.spv");
const FN_PLE_RES_SPV: &[u8] = include_bytes!("spv/fn_ple_res.spv");
/// 파이프라인 세트 (vk 핸들은 복사 가능).
/// 지연 파이프라인 슬롯.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Slot {
    Gemv,
    Tile128,
    /// plans/84 B: q5_1 타일판(tile128_q51) — FN 다운 질량(25.2GiB) 프리필.
    Tile128Q51,
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
    /// plans/89 P1.4 — PLE gate/conv/residual.
    FnPleGate,
    FnPleConv,
    FnPleRes,
}
/// plans/86 §6 — 모델 파트 파일 (mmap 범위 + 핸들). 대형 가중 업로드를
/// pread 스테이징으로 수행한다(hip staged_upload 미러).
struct PartSource {
    base: usize,
    len: usize,
    file: std::fs::File,
}

pub struct VkAcc {
    ctx: Mutex<VkCtx>,
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
    framebufs: Mutex<HashMap<u64, VkBuf>>,
    /// plans/85 §1 — 해제된 프레임 버퍼 재활용 풀. VkBuf는 파괴자가 없어
    /// 종전 frame_free는 종료까지 누출 — 디코드 스텝 스크래치(shexp 등)가
    /// 매층·매스텝 할당되므로 상한 내에서 재활용한다.
    frame_pool: Mutex<Vec<VkBuf>>,
    /// plans/85 §2 — QSA 디코드 선택 스크래치 (iqr, scr, flg, iqw, cs, sdev, ofdev).
    qsa_sel_bufs: Mutex<Option<(VkBuf, VkBuf, VkBuf, VkBuf, VkBuf, VkBuf, VkBuf)>>,
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

/// plans/87 §2/§3 — 슬롯 → op 태그(와치독 링·ts 라벨).
fn slot_name(slot: Slot) -> &'static str {
    match slot {
        Slot::Gemv => "gemv",
        Slot::Tile128 => "tile128_q5k",
        Slot::Tile128Q51 => "tile128_q51",
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
        Slot::FnPleConv => "ple_conv",
        Slot::FnPleRes => "ple_res",
        Slot::FnMoeIds51 => "moe_ids51",
        Slot::FnMoeGroup => "moe_group",
        Slot::FnMoeTileQ4K => "moe_tile_q4k",
        Slot::FnTileF32 => "tile_f32",
        Slot::FnMoeTileQ8 => "moe_tile_q8",
        Slot::FnMoeTileQ5k => "moe_tile_q5k",
        Slot::FnMoeTileQ51Cm => "moe_tile_q51_cm",
        Slot::FnMoeTileQ51 => "moe_tile_q51",
        Slot::FnMoeTileQ4kCm => "moe_tile_q4k_cm",
        Slot::FnTileQ8 => "tile_q8",
     }
 }

fn push_u32s(vals: &[u32]) -> Vec<u8> {
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

    fn pipeline(&self, ctx: &mut VkCtx, slot: Slot) -> Result<Pipes, String> {
        crate::rawvk::context::site::set_tag(slot_name(slot));
        if let Some(&p) = self.pipes.lock().get(&slot) {
            return Ok(p);
        }
        let (spv, n_buf, pb) = match slot {
            Slot::Gemv => (GEMV_SPV, 12, 24u32),
            Slot::Tile128 => (TILE128_SPV, 10, 16),
            Slot::Tile128Q51 => (TILE128_Q51_SPV, 10, 24),
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
            Slot::FnPleRes => (FN_PLE_RES_SPV, 4, 12),     // res,val,gate,conv
            Slot::TileQ8128Cm => (TILE_Q8128_SPV2, 10, 24),
            Slot::TileQ8msCm => (TILE_Q8MS_SPV2, 10, 20),
            Slot::TileQ4k128Cm => (TILE_Q4K128_SPV2, 10, 24),
            Slot::TileQ4kmsCm => (TILE_Q4KMS_SPV2, 10, 20),
            Slot::FnMoeTileQ4kCm => (FN_MOE_TILE_Q4K_CM_SPV, 13, 28),
            Slot::FnMoeTileQ8 => (FN_MOE_TILE_Q8_SPV, 13, 28),
            Slot::FnMoeTileQ5k => (FN_MOE_TILE_Q5K_SPV, 13, 28),
            Slot::FnMoeTileQ51Cm => (FN_MOE_TILE_Q51_CM_SPV, 13, 28),
        };
        let p = ctx.pipeline_pipes(spv, n_buf, pb)?;
        self.pipes.lock().insert(slot, p);
        Ok(p)
    }

    /// ktab(iq4nl)·grid3s 테이블 + 더미 버퍼 — 최초 1회 업로드.
    fn ensure_shared(&self, ctx: &mut VkCtx) -> Result<(vk::Buffer, vk::Buffer, vk::Buffer), String> {
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
        // plans/87 §1 — 회귀 재현 스위치: 구형 ptr 단독 키. 값경로 프리필의
        // 전문가 뷰(1전문가분)과 전체 스택이 같은 키로 충돌해 프레임 MoE가
        // 오프셋 재결합 → 실제 GPUVM 폴트(86 §1b 사건). 폴트 매처 검증용.
        let key = if std::env::var("LLM170_WCACHE_PTRKEY").as_deref() == Ok("1") {
            (w.data.as_ptr() as usize, 0)
        } else {
            (w.data.as_ptr() as usize, w.data.len())
        };
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

    /// 128행 coopmat 타일 (q5_K, t≥2) — f16 스테이징, maxrel ~4.9e-4 (HIP v4급).
    fn tile128_run(
        &self,
        ctx: &mut VkCtx,
        wbufs: &[vk::Buffer],
        n_in: usize,
        n_out: usize,
        xq_w: usize,
        t: usize,
        xq_buf: vk::Buffer,
        out_buf: vk::Buffer,
        slot: Slot,
    ) -> Result<(), String> {
        let (_, _, dbuf) = self.ensure_shared(ctx)?;
        let p = self.pipeline(ctx, slot)?;
        let mut binds: Vec<vk::Buffer> = wbufs.to_vec();
        while binds.len() < 8 {
            binds.push(dbuf);
        }
        binds.push(xq_buf);
        binds.push(out_buf);
        let ds2 = ctx.bind_ds(&p, &binds)?;
        let gx = (n_out + 127) as u32 / 128;
        for tb in (0..t).step_by(64) {
            let nt = (t - tb).min(64) as u32;
            let push = push_u32s(&[n_in as u32, n_out as u32, xq_w as u32, nt]);
            ctx.run(p.pl, ds2, p.pipe, &push, gx, 1, 1)?;
        }
        Ok(())
    }

    /// plans/84 B: q5_1 타일판 런처 — 128토큰 슬래브, push 5필드
    /// [n_in,n_out,xq_w,nt,tok_base]. (tile128_run의 4필드 push는 t>64
    /// 슬래브 오프셋을 표현 못 한다 — 여기서 바로잡는다.)
    fn tile128_q51_run(
        &self,
        ctx: &mut VkCtx,
        wbufs: &[vk::Buffer],
        n_in: usize,
        n_out: usize,
        xq_w: usize,
        t: usize,
        xq_buf: vk::Buffer,
        out_buf: vk::Buffer,
    ) -> Result<(), String> {
        let (_, _, dbuf) = self.ensure_shared(ctx)?;
        let p = self.pipeline(ctx, Slot::Tile128Q51)?;
        let mut binds: Vec<vk::Buffer> = wbufs.to_vec();
        while binds.len() < 8 {
            binds.push(dbuf);
        }
        binds.push(xq_buf);
        binds.push(out_buf);
        let ds2 = ctx.bind_ds(&p, &binds)?;
        let gx = (n_out + 127) as u32 / 128;
        // 청크 용량 = max_ssbo 바이트(워드 log2) — WG() 분할 규약.
        let cw = (ctx.max_ssbo / 4) as u32;
        let wsh = 31u32 - cw.next_power_of_two().leading_zeros();
        for tb in (0..t).step_by(128) {
            let nt = (t - tb).min(128) as u32;
            let push = push_u32s(&[n_in as u32, n_out as u32, xq_w as u32, nt, tb as u32, wsh]);
            ctx.run(p.pl, ds2, p.pipe, &push, gx, 1, 1)?;
        }
        Ok(())
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
    fn quant_upload(
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
        let xq_w = n_in / 4 + n_in / 32 + n_in / 16;
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

impl VkAcc {
    /// rms_norm 오프로드 — f32 세그먼트+f64 결합 (CPU sq_sum 미러와 동일 순서).
    pub fn rms_norm_gpu(
        &self,
        xs: &[Vec<f32>],
        w: &[f32],
        eps: f32,
        outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        let t = xs.len();
        let n = w.len();
        let mut ctx = self.ctx.lock();
        {
            let mut b = self.rbufs.lock();
            if b.is_none() {
                let xb = ctx.alloc_host((t * n * 4).max(1 << 21))?;
                let wb = ctx.alloc_host(n * 4)?;
                let ob = ctx.alloc_host((t * n * 4).max(1 << 21))?;
                *b = Some((xb, wb, ob));
            }
        }
        {
            let b = self.rbufs.lock();
            let (xv, wv, _) = b.as_ref().unwrap();
            for (ti, row) in xs.iter().enumerate() {
                unsafe { std::ptr::copy_nonoverlapping(row.as_ptr(), xv.ptr.add(ti * n * 4) as *mut f32, n) };
            }
            unsafe { std::ptr::copy_nonoverlapping(w.as_ptr(), wv.ptr as *mut f32, n) };
        }
        let (xb, wb, ob) = {
            let b = self.rbufs.lock();
            let r = b.as_ref().unwrap();
            (r.0.buf, r.1.buf, r.2.buf)
        };
        let p = self.pipeline(&mut ctx, Slot::Rms)?;
        let ds2 = ctx.bind_ds(&p, &[xb, wb, ob])?;
        let mut push = push_u32s(&[n as u32, t as u32, 1u32]);
        push.extend_from_slice(&eps.to_le_bytes());
        ctx.run(p.pl, ds2, p.pipe, &push, t as u32, 1, 1)?;
        let host = {
            let b = self.rbufs.lock();
            unsafe { std::slice::from_raw_parts(b.as_ref().unwrap().2.ptr as *const f32, t * n) }
        };
        for ti in 0..t {
            outs[ti].copy_from_slice(&host[ti * n..(ti + 1) * n]);
        }
        Ok(())
    }

    /// silu_mul 오프로드 — exp_cr f64 호너 GLSL 비트 재현.
    pub fn silu_mul_gpu(
        &self,
        gs: &[Vec<f32>],
        us: &[Vec<f32>],
        outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        let t = gs.len();
        let n = gs[0].len();
        let total = t * n;
        let mut ctx = self.ctx.lock();
        {
            let mut b = self.sbufs.lock();
            if !b.as_ref().map(|(g, _, _)| g.bytes >= total * 4).unwrap_or(false) {
                let g = ctx.alloc_host((total * 4).max(1 << 21))?;
                let u = ctx.alloc_host((total * 4).max(1 << 21))?;
                let o = ctx.alloc_host((total * 4).max(1 << 21))?;
                *b = Some((g, u, o));
            }
        }
        {
            let b = self.sbufs.lock();
            let (gv, uv, _) = b.as_ref().unwrap();
            for (ti, row) in gs.iter().enumerate() {
                unsafe { std::ptr::copy_nonoverlapping(row.as_ptr(), gv.ptr.add(ti * n * 4) as *mut f32, n) };
            }
            for (ti, row) in us.iter().enumerate() {
                unsafe { std::ptr::copy_nonoverlapping(row.as_ptr(), uv.ptr.add(ti * n * 4) as *mut f32, n) };
            }
        }
        let (gb, ub, ob) = {
            let b = self.sbufs.lock();
            let r = b.as_ref().unwrap();
            (r.0.buf, r.1.buf, r.2.buf)
        };
        let p = self.pipeline(&mut ctx, Slot::Silu)?;
        let ds2 = ctx.bind_ds(&p, &[gb, ub, ob])?;
        let total_u = total as u32;
        ctx.run(p.pl, ds2, p.pipe, &total_u.to_le_bytes(), total_u.div_ceil(256), 1, 1)?;
        let host = {
            let b = self.sbufs.lock();
            unsafe { std::slice::from_raw_parts(b.as_ref().unwrap().2.ptr as *const f32, total) }
        };
        for ti in 0..t {
            outs[ti].copy_from_slice(&host[ti * n..(ti + 1) * n]);
        }
        Ok(())
    }

    /// FFN 상주 체인 — 업로드 1회(xs)·다운로드 1회(xs), gate/up/silu/glu/down 전부 GPU 상주.
    #[allow(clippy::too_many_arguments)]
    pub fn ffn_chain_gpu(
        &self,
        xs: &[Vec<f32>],
        gate_w: &Weight,
        up_w: &Weight,
        down_w: &Weight,
        xs_out: &mut [Vec<f32>],
    ) -> Result<(), String> {
        let t = xs.len();
        let n0 = gate_w.n_in as usize; // n_embd
        let n_ff = gate_w.n_out as usize;
        let xq0_w = n0 / 4 + n0 / 32 + n0 / 16;
        let xq1_w = n_ff / 4 + n_ff / 32 + n_ff / 16;
        let mut ctx = self.ctx.lock();
        // 체인 버퍼 (고정 용량 — 모델 최대 기준)
        let (xbf, bq0, bfg, bfu, bglu, bq1, bob, xf_ptr, ob_ptr) = {
            let mut b = self.ffnbufs.lock();
            if b.is_none() {
                let xf = ctx.alloc_host(1 << 23)?;
                let xq0 = ctx.alloc_host(1 << 22)?;
                let fg = ctx.alloc_host(1 << 24)?;
                let fu = ctx.alloc_host(1 << 24)?;
                let glu = ctx.alloc_host(1 << 24)?;
                let xq1 = ctx.alloc_host(1 << 24)?;
                let ob = ctx.alloc_host(1 << 23)?;
                *b = Some((xf, xq0, fg, fu, glu, xq1, ob));
            }
            let r = b.as_ref().unwrap();
            (r.0.buf, r.1.buf, r.2.buf, r.3.buf, r.4.buf, r.5.buf, r.6.buf, r.0.ptr, r.6.ptr)
        };
        // 배치 모드 — 6연산 단일 제출 (plans/19: sync ~0.9ms×5 절감)
        if std::env::var_os("LLM170_VK_NOBATCH").is_none() {
            ctx.begin_batch()?;
        }
        // 1) xs 업로드 → quant(n0)
        for (ti, row) in xs.iter().enumerate() {
            unsafe { std::ptr::copy_nonoverlapping(row.as_ptr(), xf_ptr.add(ti * n0 * 4) as *mut f32, n0) };
        }
        {
            let p = self.pipeline(&mut ctx, Slot::Quant)?;
            let ds2 = ctx.bind_ds(&p, &[xbf, bq0])?;
            let push = push_u32s(&[n0 as u32, t as u32, xq0_w as u32]);
            ctx.run(p.pl, ds2, p.pipe, &push, ((n0 / 32) + 63) as u32 / 64, t as u32, 1)?;
        }
        // 2) gate/up GEMV (같은 xq0) — 상주 출력
        for (w, obuf) in [(gate_w, bfg), (up_w, bfu)] {
            let ty = vk_ty(w.ty).ok_or("ffn 타입 미지원")?;
            let wbufs = self.weight_bufs(&mut ctx, w)?;
            self.gemv_run(&mut ctx, &wbufs, n0, w.n_out as usize, xq0_w, ty, t, bq0, obuf)?;
        }
        // 3) silu_mul 상주 (bfg, bfu → bglu)
        {
            let p = self.pipeline(&mut ctx, Slot::Silu)?;
            let ds2 = ctx.bind_ds(&p, &[bfg, bfu, bglu])?;
            let total = (t * n_ff) as u32;
            ctx.run(p.pl, ds2, p.pipe, &total.to_le_bytes(), total.div_ceil(256), 1, 1)?;
        }
        // 4) glu quant(n_ff)
        {
            // bglu는 f32가 아니라 f32→q8 변환 입력 — quant 셰이더에 직접.
            // (bglu는 silu 출력 f32 → quant가 읽는다)
            let p = self.pipeline(&mut ctx, Slot::Quant)?;
            let ds2 = ctx.bind_ds(&p, &[bglu, bq1])?;
            let push = push_u32s(&[n_ff as u32, t as u32, xq1_w as u32]);
            ctx.run(p.pl, ds2, p.pipe, &push, ((n_ff / 32) + 63) as u32 / 64, t as u32, 1)?;
        }
        // 5) down GEMV
        {
            let ty = vk_ty(down_w.ty).ok_or("ffn down 타입 미지원")?;
            let wbufs = self.weight_bufs(&mut ctx, down_w)?;
            self.gemv_run(&mut ctx, &wbufs, n_ff, down_w.n_out as usize, xq1_w, ty, t, bq1, bob)?;
        }
        // 6) 일괄 제출·대기 → 다운로드 1회
        if std::env::var_os("LLM170_VK_NOBATCH").is_none() {
            ctx.end_batch_wait()?;
        }
        let host = unsafe { std::slice::from_raw_parts(ob_ptr as *const f32, t * n0) };
        for ti in 0..t {
            xs_out[ti].copy_from_slice(&host[ti * n0..(ti + 1) * n0]);
        }
        Ok(())
    }
}

// 미지원 capability — 모든 메서드가 기본(Err) 구현이라 빈 impl 로 충분하다.
impl llm170_core::matmul::GraphCapture for VkAcc {}
impl llm170_core::matmul::QsaOps for VkAcc {
    /// 상주 KV 풀 — 워터마크 규약은 hip과 동일(순차 적립/접두어 되감기 허용).
    /// 적립은 디바이스 간 복사(copy_rows 판) — 프레임 k/v 버퍼에서 풀로.
    fn qsa_kv_dev(
        &self,
        full_idx: usize,
        seq: usize,
        k: u64,
        v: u64,
        t: usize,
        pos0: usize,
        n_kv: usize,
        hd: usize,
    ) -> Result<(u64, u64), String> {
        let ctx_len = self.qsa_ctx.load(std::sync::atomic::Ordering::Relaxed);
        if ctx_len == 0 {
            return Err("vk qsa_kv_dev: ctx_len 미주입".into());
        }
        let bytes = ctx_len * n_kv * hd * 4;
        let need_grow;
        {
            let mut m = self.qsa_pools.lock();
            let e = m.entry((full_idx, seq)).or_insert_with(|| {
                (
                    VkBuf { buf: vk::Buffer::null(), ptr: std::ptr::null_mut(), bytes: 0, mem: vk::DeviceMemory::null() },
                    VkBuf { buf: vk::Buffer::null(), ptr: std::ptr::null_mut(), bytes: 0, mem: vk::DeviceMemory::null() },
                    VkBuf { buf: vk::Buffer::null(), ptr: std::ptr::null_mut(), bytes: 0, mem: vk::DeviceMemory::null() },
                    VkBuf { buf: vk::Buffer::null(), ptr: std::ptr::null_mut(), bytes: 0, mem: vk::DeviceMemory::null() },
                    0,
                )
            });
            if pos0 > e.4 {
                return Err(format!("vk qsa_kv_dev: 워터마크 구멍 w={} pos0={pos0}", e.4));
            }
            e.4 = pos0 + t;
            need_grow = e.0.bytes < bytes;
        }
        let mut ctx = self.ctx.lock();
        if need_grow {
            let mut m = self.qsa_pools.lock();
            let e = m.get_mut(&(full_idx, seq)).unwrap();
            // plans/84 B: 호스트 가시 — qsa_host_rebuild가 직접 판독한다.
            e.0 = ctx.alloc_host(bytes)?;
            e.1 = ctx.alloc_host(bytes)?;
        }
        let (kb, vb) = {
            let m = self.qsa_pools.lock();
            let e = m.get(&(full_idx, seq)).unwrap();
            (e.0.buf, e.1.buf)
        };
        let ksrc = self.fbuf(k)?;
        let vsrc = self.fbuf(v)?;
        let p = self.pipeline(&mut ctx, Slot::CopyRows)?;
        let n_f = t * n_kv * hd;
        let soff = (pos0 * n_kv * hd) as u32;
        let ds_k = ctx.bind_ds(&p, &[ksrc, kb])?;
        ctx.run(p.pl, ds_k, p.pipe, &push_u32s(&[n_f as u32, 0u32, soff]), (n_f as u32).div_ceil(256), 1, 1)?;
        let ds_v = ctx.bind_ds(&p, &[vsrc, vb])?;
        ctx.run(p.pl, ds_v, p.pipe, &push_u32s(&[n_f as u32, 0u32, soff]), (n_f as u32).div_ceil(256), 1, 1)?;
        Ok((kb.as_raw(), vb.as_raw()))
    }

    /// 상주 인덱서 풀 — ik 적립 + 완성 블록의 블록키(norm+rope) 갱신.
    fn qsa_idx_append_dev(
        &self,
        full_idx: usize,
        seq: usize,
        ik: u64,
        t: usize,
        pos0: usize,
        idx_dim: usize,
        r: usize,
        ikw: &[f32],
        cs_idx: &[f32],
        eps: f32,
    ) -> Result<(), String> {
        if r == 0 || idx_dim != 128 {
            return Err(format!("vk qsa_idx_append: 미지원 형상 r={r} idx_dim={idx_dim}"));
        }
        let ctx_len = self.qsa_ctx.load(std::sync::atomic::Ordering::Relaxed);
        if ctx_len == 0 {
            return Err("vk qsa_idx_append: ctx_len 미주입".into());
        }
        let nb_max = ctx_len / r + 1;
        {
            let mut m = self.qsa_pools.lock();
            // plans/85 §2: entry()로 생성+실제 갱신 — 종전 `&mut get_mut().map()
            // .unwrap_or(0)`는 임시값에 써서 워터마크가 영구 반영되지 않았고
            // (kv_dev가 대신 갱신해 온 것), 엔트리 없으면 아래 get_mut().unwrap()
            // 이 패닉했다(np 디코드: sel_dev가 kv_dev보다 먼저 append 호출).
            let e = m
                .entry((full_idx, seq))
                .or_insert_with(|| (vkbuf_null(), vkbuf_null(), vkbuf_null(), vkbuf_null(), 0));
            if pos0 > e.4 {
                return Err(format!("vk qsa_idx_append: 워터마크 구멍 w={} pos0={pos0}", e.4));
            }
            e.4 = pos0 + t;
        }
        // idx_k 풀은 kv 풀과 별도 용량 — 필요시 성장(간단 재할당).
        let mut ctx = self.ctx.lock();
        let need = (ctx_len * idx_dim * 4, nb_max * idx_dim * 4);
        {
            let mut m = self.qsa_pools.lock();
            let e = m.get_mut(&(full_idx, seq)).unwrap();
            if e.2.bytes < need.0 {
                e.2 = crate::rawvk::context::site::scope("qsa_pool", || ctx.alloc_host(need.0))?;
            }
            if e.3.bytes < need.1 {
                e.3 = crate::rawvk::context::site::scope("qsa_pool", || ctx.alloc_host(need.1))?;
            }
        }
        let (ikb, bkb) = {
            let m = self.qsa_pools.lock();
            let e = m.get(&(full_idx, seq)).unwrap();
            (e.2.buf, e.3.buf)
        };
        let iksrc = self.fbuf(ik)?;
        let p = self.pipeline(&mut ctx, Slot::CopyRows)?;
        let n_f = t * idx_dim;
        let soff = (pos0 * idx_dim) as u32;
        let ds = ctx.bind_ds(&p, &[iksrc, ikb])?;
        ctx.run(p.pl, ds, p.pipe, &push_u32s(&[n_f as u32, 0u32, soff]), (n_f as u32).div_ceil(256), 1, 1)?;
        let b0 = pos0 / r;
        let b1 = (pos0 + t) / r;
        if b1 > b0 {
            // plans/86 §4 — 상수 업로드 캐시: cs 전체 표는 (ptr,len) 키로 1회
            // (셰이더 cs 인덱싱이 절대 pos 기반이라 접두 복사 불필요), ikw 는
            // 호출부가 매번 새 Vec을 만들어 고정 스크래치에 복사(512B).
            // 종전 매호출 alloc_host 두 개는 블록 완성마다 GTT에 누출했다.
            let cs_b = self.qk_const(&mut ctx, cs_idx)?;
            let ikw_b = {
                let mut g = self.qsa_ikw.lock();
                if !g.as_ref().is_some_and(|b| b.bytes >= ikw.len() * 4) {
                    *g = Some(crate::rawvk::context::site::scope("qsa_const", || ctx.alloc_host((ikw.len() * 4).max(4096)))?);
                }
                g.as_ref().unwrap().clone()
            };
            unsafe { std::ptr::copy_nonoverlapping(ikw.as_ptr(), ikw_b.ptr as *mut f32, ikw.len()) };
            let p2 = self.pipeline(&mut ctx, Slot::FnIdxBk)?;
            let ds2 = ctx.bind_ds(&p2, &[ikb, bkb, ikw_b.buf, cs_b.buf])?;
            // plans/85 §2: 셰이더 PC는 선언순 {eps, b0, r, idx_dim} — 종전
            // [b0,r,idx_dim,eps]는 멤버가 전부 어긋나 idx_dim≠128 조기복귀로
            // 블록키가 한 번도 갱신되지 않았다(항등 선택이라 프리필은 무영향).
            let mut push = eps.to_le_bytes().to_vec();
            push.extend_from_slice(&push_u32s(&[b0 as u32, r as u32, idx_dim as u32]));
            ctx.run(p2.pl, ds2, p2.pipe, &push, (b1 - b0) as u32, 1, 1)?;
        }
        Ok(())
    }

    /// 디바이스 풀 → 호스트 캐시 재구축 — 프리필이 디코딩을 건너뛴 뒤 값
    /// 경로 호스트 선택이 필요할 때 1회. 풀 내용을 그대로 내린다.
    fn qsa_host_rebuild(
        &self,
        full_idx: usize,
        seq: usize,
        pos: usize,
        kv_row: usize,
        kv_k: &mut [f32],
        kv_v: &mut [f32],
        idx_k: &mut [f32],
        bk: &mut [f32],
        r: usize,
        idx_dim: usize,
    ) -> Result<(), String> {
        let _ = pos;
        let m = self.qsa_pools.lock();
        let Some(e) = m.get(&(full_idx, seq)) else {
            return Err("vk qsa_host_rebuild: 풀 없음".into());
        };
        if e.0.bytes < kv_k.len() * 4 || e.2.bytes < idx_k.len() * 4 {
            return Err("vk qsa_host_rebuild: 풀 용량 부족".into());
        }
        // 풀은 호스트 가시(alloc_host) — 프레임 동기 후 직접 판독.
        {
            let mut c = self.ctx.lock();
            if c.batching.load(std::sync::atomic::Ordering::Relaxed) {
                let _ = c.end_batch_wait();
            }
        }
        let kv_floats = kv_k.len();
        unsafe {
            std::ptr::copy_nonoverlapping(e.0.ptr as *const f32, kv_k.as_mut_ptr(), kv_floats);
            std::ptr::copy_nonoverlapping(e.1.ptr as *const f32, kv_v.as_mut_ptr(), kv_v.len());
            std::ptr::copy_nonoverlapping(e.2.ptr as *const f32, idx_k.as_mut_ptr(), idx_k.len());
            let nb = (pos + r - 1) / r;
            std::ptr::copy_nonoverlapping(e.3.ptr as *const f32, bk.as_mut_ptr(), (nb * idx_dim).min(bk.len()));
        }
        let _ = kv_row;
        Ok(())
    }

    /// plans/85 §2 — 디코드(t=1) 선택의 디바이스판: ik 적립+블록키 갱신(기존
    /// qsa_idx_append_dev) → iq norm+rope → 블록 점수 → 순위 → 목록 전개.
    /// 산술은 hip qsa_sel_dev와 동일열(f64 순차 rms, f64 회전, 4누산 도트,
    /// 정수 순위) — 선택 목록이 호스트 top-k와 비트 일치.
    #[allow(clippy::too_many_arguments)]
    fn qsa_sel_dev(
        &self,
        full_idx: usize,
        seq: usize,
        iq: u64,
        ik: u64,
        t: usize,
        pos0: usize,
        idx_heads: usize,
        idx_dim: usize,
        r: usize,
        idx_top_k: usize,
        iqw: &[f32],
        ikw: &[f32],
        cs_idx: &[f32],
        eps: f32,
    ) -> Result<(u64, u64, usize), String> {
        if t != 1 {
            return Err(format!("vk qsa_sel_dev: t={t} (디코드 전용)"));
        }
        if r == 0 {
            return Err("vk qsa_sel_dev: r=0".into());
        }
        if idx_dim != 128 {
            return Err(format!("vk qsa_sel_dev: idx_dim={idx_dim} (128 전용)"));
        }
        let n_past = pos0 + t;
        let n_blocks = n_past / r;
        // (1) ik 적립 + 완성 블록 키 — 기존 구현(워터마크 규약 공유).
        self.qsa_idx_append_dev(full_idx, seq, ik, t, pos0, idx_dim, r, ikw, cs_idx, eps)?;
        // n_sel 산술은 stages::qsa_select 패스 B와 동일(정수 — 무동기).
        let tail_start = n_blocks * r;
        let tail_cnt = n_past - tail_start;
        let width = n_past.min(idx_top_k + r - 1);
        let n_sel = ((width - tail_cnt) / r).min(n_blocks);
        let list_len = n_sel * r + tail_cnt;
        let iqr_bytes = t * idx_heads * idx_dim * 4;
        let scr_bytes = n_blocks.max(1) * 4;
        let sd_bytes = list_len.max(1) * 4;
        let mut ctx = self.ctx.lock();
        // 스크래치 — 필요시 성장 재할당(매 호출 alloc_host 누출 회피).
        {
            let mut g = self.qsa_sel_bufs.lock();
            let need = match g.as_ref() {
                None => true,
                Some(b) => {
                    b.0.bytes < iqr_bytes
                        || b.1.bytes < scr_bytes
                        || b.2.bytes < scr_bytes
                        || b.3.bytes < iqw.len() * 4
                        || b.4.bytes < idx_dim * 4
                        || b.5.bytes < sd_bytes
                }
            };
            if need {
                *g = Some(crate::rawvk::context::site::scope("qsa_sel", || Ok::<_, String>((
                    ctx.alloc_host(iqr_bytes.max(1 << 16))?,
                    ctx.alloc_host(scr_bytes.max(4096))?,
                    ctx.alloc_host(scr_bytes.max(4096))?,
                    ctx.alloc_host((iqw.len() * 4).max(4096))?,
                    ctx.alloc_host((idx_dim * 4).max(1 << 16))?,
                    ctx.alloc_host(sd_bytes.max(1 << 16))?,
                    ctx.alloc_host(8)?,
                )))?);
            }
        }
        let (iqr, scr, flg, iqwb, csb, sdev, ofdev) = {
            let g = self.qsa_sel_bufs.lock();
            let b = g.as_ref().unwrap();
            (b.0.clone(), b.1.clone(), b.2.clone(), b.3.clone(), b.4.clone(), b.5.clone(), b.6.clone())
        };
        // 호스트 상수(매 호출 소량) — iqw 전체, cs는 pos0행 idx_dim.
        unsafe {
            std::ptr::copy_nonoverlapping(iqw.as_ptr(), iqwb.ptr as *mut f32, iqw.len());
            std::ptr::copy_nonoverlapping(
                cs_idx[pos0 * idx_dim..].as_ptr(),
                csb.ptr as *mut f32,
                idx_dim,
            );
        }
        // (2) iq norm+rope — 워크그룹 = (헤드, 토큰), 32스레드.
        let iqb = self.fbuf(iq)?;
        {
            // PC 선언순 {eps, idx_dim} — cs 인덱싱은 업로드 상대(행 y).
            let p = self.pipeline(&mut ctx, Slot::FnIdxQRope)?;
            let ds = ctx.bind_ds(&p, &[iqb, iqr.buf, iqwb.buf, csb.buf])?;
            let mut push = eps.to_le_bytes().to_vec();
            push.extend_from_slice(&push_u32s(&[idx_dim as u32]));
            ctx.run(p.pl, ds, p.pipe, &push, idx_heads as u32, t as u32, 1)?;
        }
        let bkb = {
            let m = self.qsa_pools.lock();
            m.get(&(full_idx, seq))
                .map(|e| e.3.buf)
                .ok_or("vk qsa_sel_dev: bk 풀 없음")?
        };
        if n_blocks > 0 {
            // (3) 블록 점수 — 스레드당 블록.
            let p = self.pipeline(&mut ctx, Slot::FnIdxScore)?;
            let ds = ctx.bind_ds(&p, &[iqr.buf, bkb, scr.buf])?;
            let push = push_u32s(&[n_blocks as u32, idx_heads as u32, idx_dim as u32]);
            ctx.run(p.pl, ds, p.pipe, &push, (n_blocks as u32).div_ceil(256), 1, 1)?;
            // (4) 순위 — 결정적 top-k(점수 내림, 인덱스 오름).
            let p = self.pipeline(&mut ctx, Slot::FnIdxRank)?;
            let ds = ctx.bind_ds(&p, &[scr.buf, flg.buf])?;
            let push = push_u32s(&[n_blocks as u32, n_sel as u32]);
            ctx.run(p.pl, ds, p.pipe, &push, (n_blocks as u32).div_ceil(256), 1, 1)?;
        }
        // (5) 목록 전개 — 단일 워크그룹(무공유메모리 판).
        {
            let p = self.pipeline(&mut ctx, Slot::FnIdxExpand)?;
            let ds = ctx.bind_ds(&p, &[flg.buf, sdev.buf, ofdev.buf])?;
            let push = push_u32s(&[n_blocks as u32, n_sel as u32, r as u32, n_past as u32]);
            ctx.run(p.pl, ds, p.pipe, &push, 1, 1, 1)?;
        }
        Ok((sdev.buf.as_raw(), ofdev.buf.as_raw(), list_len))
    }

    /// plans/73 SELCHECK 진단 — qsa_sel_dev가 만든 목록을 호스트로 내린다.
    fn qsa_sel_readback(
        &self,
        sel_idx: u64,
        sel_off: u64,
        list_len: usize,
    ) -> Result<(Vec<u32>, Vec<u32>), String> {
        {
            let mut c = self.ctx.lock();
            if c.batching.load(std::sync::atomic::Ordering::Relaxed) {
                let _ = c.end_batch_wait();
            }
        }
        let g = self.qsa_sel_bufs.lock();
        let b = g.as_ref().ok_or("vk qsa_sel_readback: 스크래치 없음")?;
        if b.5.buf.as_raw() != sel_idx as u64 || b.6.buf.as_raw() != sel_off as u64 {
            return Err("vk qsa_sel_readback: 핸들 불일치(스크래치 재성장)".into());
        }
        let mut si = vec![0u32; list_len];
        unsafe { std::ptr::copy_nonoverlapping(b.5.ptr as *const u32, si.as_mut_ptr(), list_len) };
        let mut so = vec![0u32; 2];
        unsafe { std::ptr::copy_nonoverlapping(b.6.ptr as *const u32, so.as_mut_ptr(), 2) };
        Ok((si, so))
    }

    /// plans/85 §2 — sel 목록이 디바이스에 있는 상주판 어텐션(업로드 없음,
    /// fn_qsa_attn_sel 그대로 — grid (t, n_head), t=1).
    #[allow(clippy::too_many_arguments)]
    fn qsa_attention_dev_sel(
        &self,
        q: u64,
        ck: u64,
        cv: u64,
        sel_idx: u64,
        sel_off: u64,
        _list_len: usize,
        kq_scale: f32,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
        out: u64,
    ) -> Result<(), String> {
        if hd != 256 {
            return Err(format!("vk qsa_attention_dev_sel: hd={hd} 미지원"));
        }
        let mut ctx = self.ctx.lock();
        let (qb, ob) = (self.fbuf(q)?, self.fbuf(out)?);
        let cb = vk::Buffer::from_raw(ck as u64);
        let vb = vk::Buffer::from_raw(cv as u64);
        let sib = vk::Buffer::from_raw(sel_idx as u64);
        let sob = vk::Buffer::from_raw(sel_off as u64);
        let p = self.pipeline(&mut ctx, Slot::FnQsaAttnSel)?;
        let ds2 = ctx.bind_ds(&p, &[qb, cb, vb, sib, sob, ob])?;
        let mut push = kq_scale.to_le_bytes().to_vec();
        push.extend_from_slice(&push_u32s(&[n_head as u32, n_kv as u32, hd as u32, t as u32]));
        ctx.run(p.pl, ds2, p.pipe, &push, t as u32, n_head as u32, 1)
    }

    /// 선택 목록 어텐션 — fn_qsa_attn_sel 판(hd=256).
    #[allow(clippy::too_many_arguments)]
    fn qsa_attention_dev_res(
        &self,
        q: u64,
        ck: u64,
        cv: u64,
        sel_idx: &[u32],
        sel_off: &[u32],
        kq_scale: f32,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
        out: u64,
    ) -> Result<(), String> {
        if hd != 256 {
            return Err(format!("vk qsa_attention_dev_res: hd={hd} 미지원"));
        }
        let mut ctx = self.ctx.lock();
        let (qb, ob) = (self.fbuf(q)?, self.fbuf(out)?);
        let cb = vk::Buffer::from_raw(ck as u64);
        let vb = vk::Buffer::from_raw(cv as u64);
        // plans/86 §4 — sel 스크래치 캐시(종전 매호출 alloc_host 누출).
        let (si_b, so_b) = self.qsa_sel_scratch(&mut ctx, sel_idx.len(), sel_off.len())?;
        unsafe {
            std::ptr::copy_nonoverlapping(sel_idx.as_ptr(), si_b.ptr as *mut u32, sel_idx.len());
            std::ptr::copy_nonoverlapping(sel_off.as_ptr(), so_b.ptr as *mut u32, sel_off.len());
        }
        let p = self.pipeline(&mut ctx, Slot::FnQsaAttnSel)?;
        let ds2 = ctx.bind_ds(&p, &[qb, cb, vb, si_b.buf, so_b.buf, ob])?;
        let mut push = kq_scale.to_le_bytes().to_vec();
        push.extend_from_slice(&push_u32s(&[n_head as u32, n_kv as u32, hd as u32, t as u32]));
        ctx.run(p.pl, ds2, p.pipe, &push, t as u32, n_head as u32, 1)
    }

    /// 업로드 판 어텐션 — 호스트 ck/cv 를 스크래치에 올려 동일 커널(plans/86 §3:
    /// §2 이후 프레임 폴백 꼬리가 이 경로를 요구한다 — 종전 미구현으로 CPU 폴백,
    /// 그 폴백의 mask_from_list(&[]) 가 패닉이었다).
    #[allow(clippy::too_many_arguments)]
    fn qsa_attention_dev(
        &self,
        q: u64,
        ck: &[f32],
        cv: &[f32],
        sel_idx: &[u32],
        sel_off: &[u32],
        kq_scale: f32,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
        out: u64,
    ) -> Result<(), String> {
        if hd != 256 {
            return Err(format!("vk qsa_attention_dev: hd={hd} 미지원"));
        }
        let mut ctx = self.ctx.lock();
        let (qb, ob) = (self.fbuf(q)?, self.fbuf(out)?);
        let (si_b, so_b) = self.qsa_sel_scratch(&mut ctx, sel_idx.len(), sel_off.len())?;
        let (ckb, cvb) = {
            let mut g = self.qsa_up_bufs.lock();
            let ok = g.as_ref().is_some_and(|b| b.0.bytes >= ck.len() * 4 && b.1.bytes >= cv.len() * 4);
            if !ok {
                let (kb, vb) = crate::rawvk::context::site::scope("qsa_sel", || Ok::<_, String>((
                    ctx.alloc_host((ck.len() * 4).max(1 << 16))?,
                    ctx.alloc_host((cv.len() * 4).max(1 << 16))?,
                )))?;
                let (_, _, old_si, old_so) = g.take().unwrap_or((vkbuf_null(), vkbuf_null(), vkbuf_null(), vkbuf_null()));
                *g = Some((kb, vb, old_si, old_so));
            }
            let b = g.as_ref().unwrap();
            (b.0.clone(), b.1.clone())
        };
        unsafe {
            std::ptr::copy_nonoverlapping(ck.as_ptr(), ckb.ptr as *mut f32, ck.len());
            std::ptr::copy_nonoverlapping(cv.as_ptr(), cvb.ptr as *mut f32, cv.len());
            std::ptr::copy_nonoverlapping(sel_idx.as_ptr(), si_b.ptr as *mut u32, sel_idx.len());
            std::ptr::copy_nonoverlapping(sel_off.as_ptr(), so_b.ptr as *mut u32, sel_off.len());
        }
        let p = self.pipeline(&mut ctx, Slot::FnQsaAttnSel)?;
        let ds2 = ctx.bind_ds(&p, &[qb, ckb.buf, cvb.buf, si_b.buf, so_b.buf, ob])?;
        let mut push = kq_scale.to_le_bytes().to_vec();
        push.extend_from_slice(&push_u32s(&[n_head as u32, n_kv as u32, hd as u32, t as u32]));
        ctx.run(p.pl, ds2, p.pipe, &push, t as u32, n_head as u32, 1)
    }
}

impl VkAcc {
    /// plans/86 §2/§4 — (ptr,len) 키 상수 상주 업로드. 프레임이 테이블 Vec을
    /// 스텝 간 유지하므로 포인터가 곧 신원(호출부가 새 Vec을 만들면 미스).
    fn qk_const(&self, ctx: &mut VkCtx, v: &[f32]) -> Result<VkBuf, String> {
        let key = (v.as_ptr() as usize, v.len());
        if let Some(b) = self.qk_consts.lock().get(&key) {
            return Ok(b.clone());
        }
        let b = crate::rawvk::context::site::scope("qsa_const", || ctx.alloc_host(v.len().max(1) * 4))?;
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), b.ptr as *mut f32, v.len()) };
        self.qk_consts.lock().insert(key, b.clone());
        Ok(b)
    }

    /// plans/86 §4 — sel_idx/sel_off 업로드 스크래치(성장 재할당, 매호출 alloc 회피).
    fn qsa_sel_scratch(&self, ctx: &mut VkCtx, si: usize, so: usize) -> Result<(VkBuf, VkBuf), String> {
        let mut g = self.qsa_up_bufs.lock();
        let ok = g.as_ref().is_some_and(|b| b.2.bytes >= si * 4 && b.3.bytes >= so * 4);
        if !ok {
            let (old_ck, old_cv, _, _) = g.take().unwrap_or((vkbuf_null(), vkbuf_null(), vkbuf_null(), vkbuf_null()));
            let (sib, sob) = crate::rawvk::context::site::scope("qsa_sel", || Ok::<_, String>((
                ctx.alloc_host((si * 4).max(1 << 16))?,
                ctx.alloc_host((so * 4).max(1 << 16))?,
            )))?;
            *g = Some((old_ck, old_cv, sib, sob));
        }
        let b = g.as_ref().unwrap();
        Ok((b.2.clone(), b.3.clone()))
    }

    /// 프레임 핸들 → 상주 버퍼 (없으면 Err).
    fn fbuf(&self, h: u64) -> Result<vk::Buffer, String> {
        self.framebufs
            .lock()
            .get(&h)
            .map(|b| b.buf)
            .ok_or_else(|| format!("vk 프레임 핸들 없음: {h}"))
    }
}

impl llm170_core::matmul::FrameState for VkAcc {
    fn frame_begin(&self, t: usize) {
        self.frame_t.store(t.max(1), std::sync::atomic::Ordering::Relaxed);
        // plans/88 P1 — 스텝 수준 배치: 패스 전체를 세그먼트 최소 제출로 묶는다.
        // 비배치 run은 발사마다 제출+펜스 대기라 디코드 스텝(~2500발사)이
        // 호스트 간극에 지배됐다(실측 제출 2548/스텝). 스텝 도중 브리지의
        // frame_read가 플러시하면 프레임 op 진입마다 재개(frame_resume_batch).
        // 값경로는 이 게이트를 보지 않아 배치 상태가 새지 않는다.
        if std::env::var_os("LLM170_VK_NOBATCH").is_none() {
            self.frame_step_batch.store(true, std::sync::atomic::Ordering::Relaxed);
            let _ = self.ctx.lock().begin_batch();
        }
    }


    fn set_ctx_len(&self, n: usize) {
        self.qsa_ctx.store(n, std::sync::atomic::Ordering::Relaxed);
    }

    /// GDN AR (프레임) — q35 판(gdn_ar.spv) 재사용. q는 1/√d 스케일 완료
    /// 가정(scale=1.0), 순차 t 토큰 — 청크 불변(이전 입력만 의존).
    #[allow(clippy::too_many_arguments)]
    fn frame_gdn_ar(
        &self,
        q_scaled: u64,
        k: u64,
        v: u64,
        beta_ge: u64,
        states: u64,
        out: u64,
        n_seqs: usize,
        h_k: usize,
        h_v: usize,
        d: usize,
    ) -> Result<(), String> {
        if n_seqs != 1 {
            return Err("vk frame_gdn_ar: np 미지원".into());
        }
        let t = self.frame_t.load(std::sync::atomic::Ordering::Relaxed);
        let mut ctx = self.ctx.lock();
        self.frame_resume_batch(&mut ctx);
        let (sb, qb, kb, vb, bb, ob) = (
            self.fbuf(states)?, self.fbuf(q_scaled)?, self.fbuf(k)?,
            self.fbuf(v)?, self.fbuf(beta_ge)?, self.fbuf(out)?,
        );
        // plans/84 B: FN 상태는 전치 레이아웃(hip gdn_ar_w_swap과 동일 규약) —
        // grid (d, h_v), 상태 s[pair·d·d + u·d + …].
        let p = self.pipeline(&mut ctx, Slot::FnGdnArSwap)?;
        let ds2 = ctx.bind_ds(&p, &[sb, qb, kb, vb, bb, ob])?;
        let mut push = push_u32s(&[d as u32, (h_k * d) as u32, (h_v * d) as u32, h_v as u32, h_k as u32]);
        push.extend_from_slice(&1.0f32.to_le_bytes());
        push.extend_from_slice(&(t as u32).to_le_bytes());
        ctx.run(p.pl, ds2, p.pipe, &push, d as u32, h_v as u32, 1)
    }

    /// MoE 게더 — (토큰,전문가) 페어: xsel[(ti·k+s)·n] = mix[ti·n]
    /// (토큰 스트라이드 게더 — BcastRows 재사용은 1행 방송이라 틀렸다).
    fn frame_moe_gather(
        &self,
        mix: u64,
        xsel: u64,
        n: usize,
        k_sel: usize,
        t: usize,
    ) -> Result<(), String> {
        let mut ctx = self.ctx.lock();
        let (sb, db) = (self.fbuf(mix)?, self.fbuf(xsel)?);
        let p = self.pipeline(&mut ctx, Slot::MoeGatherRows)?;
        let ds2 = ctx.bind_ds(&p, &[sb, db])?;
        let push = push_u32s(&[n as u32, k_sel as u32, t as u32]);
        ctx.run(p.pl, ds2, p.pipe, &push, (n as u32).div_ceil(128), ((t * k_sel) as u32).div_ceil(4), 1)?;
        Ok(())
    }

    /// MoE 스캐터(가중합) — MoeWeightedSum 판 재사용(산술 동일).
    fn frame_moe_scatter(
        &self,
        ys: u64,
        wt: u64,
        out: u64,
        k_sel: usize,
        n: usize,
        _t: usize,
    ) -> Result<(), String> {
        use llm170_core::matmul::FrameHost;
        self.frame_op(&llm170_core::matmul::FrameOp::MoeWeightedSum { ys, wt, out, k: k_sel, n })
    }

    /// 상주 MoE GEMM — plans/84 B 슬라이스: 호스트 그룹화(hip 폴백과 동일
    /// 구조) + 디바이스 게더/전문가별 GEMV/스캐터. vk GEMV는 단일 판이라
    /// 전문가별 행수가 패밀리를 갈라놓지 않는다(청크 불변성 안전).
    fn frame_moe_gemm(
        &self,
        x: u64,
        w: &Weight,
        ids: u64,
        out: u64,
        n_expert_stack: usize,
        k_sel: usize,
    ) -> Result<(), String> {
        let n_in = w.n_in as usize;
        let ne = n_expert_stack.max(1);
        // 스택 텐서: w.n_out = 전문가당 n_out × ne — GEMM은 전문가당 폭만 쓴다.
        let n_out = w.n_out as usize / ne;
        let t = self.frame_t.load(std::sync::atomic::Ordering::Relaxed);
        let rows = t * k_sel;
        let ty = vk_ty(w.ty).ok_or("vk frame_moe_gemm: 타입 미지원")?;
        let mut ctx = self.ctx.lock();
        let xb = self.fbuf(x)?;
        let ob = self.fbuf(out)?;
        let xq_w = n_in / 4 + n_in / 32 + n_in / 16;
        // plans/89 P1.2 — ids dmmv 판이 이 호출을 가져갈 거면 xq 양자화 자체가
        // 불필요(f32 직결). 아래 조건은 ids2 분기와 동일해야 한다.
        let ids2_takes = rows > 0
            && (t == 1 || rows <= 64)
            && std::env::var("LLM170_MOE_IDS2").map(|v| v != "0").unwrap_or(true)
            && std::env::var_os("LLM170_MOE_GROUPED").is_none()
            && matches!(w.ty, GgmlType::Q4K | GgmlType::Q5_1);
        let xq = if ids2_takes {
            vk::Buffer::null()
        } else {
            let xq = self.xq_dev_buf(&mut ctx, rows * xq_w * 4)?;
            let p = self.pipeline(&mut ctx, Slot::Quant)?;
            let ds2 = ctx.bind_ds(&p, &[xb, xq])?;
            let push = push_u32s(&[n_in as u32, rows as u32, xq_w as u32]);
            ctx.run(p.pl, ds2, p.pipe, &push, ((n_in / 32) + 63) as u32 / 64, rows as u32, 1)?;
            xq
        };
        // 2b) direct-ids (plans/88 P1) — t=1·rows≤64: fn_moe_ids(gemv3 파생)
        // 그리드 (n_out, rows), 워크그룹=행 — 커널이 ids[r]을 직접 판독해 가중
        // 베이스 = ids[r]·per_expert 를 산출한다. ids d2h(동기 드레인)·호스트
        // 그룹화·perm/inv 업로드·게더·전문가 루프·스캐터 전부 소거(B1).
        // 산술은 종전 전문가별 gemv3 경로와 출력 요소당 비트 동일(레인 부담·
        // 감축 동일) — 토큰 스트림 불변 계약. 강제 스위치 LLM170_MOE_GROUPED=1.
        // 초판의 hip 16×16타일 직역은 이 vk에서 점유율 부족(160WG, 실측
        // 10GB/s vs hip 180GB/s)으로 폐기 — K-분할은 레인 분할(256)이 담당.
        if rows > 0
            && (t == 1 || rows <= 64)
            && std::env::var_os("LLM170_MOE_GROUPED").is_none()
            && matches!(
                w.ty,
                GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q5_1 | GgmlType::Q8_0
            )
        {
        if std::env::var_os("LLM170_MOE_IDS_DBG").is_some() {
            eprintln!("[moeids] ty={ty} rows={rows} t={t} n_in={n_in} n_out={n_out}");
        }
            // plans/89 P0.3 — ids dmmv 판 우선: llama dmmv 기하(64스레드·2행·
            // 서브그룹Add) + ids 간접, f32 활성 직결(MoE quant 불필요).
            // [ts] 기준선 moe_ids 30ms/step(43GB/s) — q8b급 150GB/s 기대.
            // 킬스위치 LLM170_MOE_IDS2=0(종전 fn_moe_ids).
            let wbufs = self.weight_bufs(&mut ctx, w)?;
            if std::env::var("LLM170_MOE_IDS2").map(|v| v != "0").unwrap_or(true)
                && wbufs.len() == 1
            {
                let (slot, blk) = match w.ty {
                    GgmlType::Q4K => (Slot::FnMoeIds2, 144usize),
                    GgmlType::Q5_1 => (Slot::FnMoeIds51, 24),
                    _ => (Slot::FnMoeIds, 0),
                };
                if blk != 0 {
                    let idb = self.fbuf(ids)?;
                    let per_expert = w.data.len() / ne;
                    let (_, _, dbuf) = self.ensure_shared(&mut ctx)?;
                    let p = self.pipeline(&mut ctx, slot)?;
                    let mut binds: Vec<vk::Buffer> = wbufs.clone();
                    while binds.len() < 8 {
                        binds.push(dbuf);
                    }
                    binds.push(xb);
                    binds.push(ob);
                    binds.push(idb);
                    let ds2 = ctx.bind_ds(&p, &binds)?;
                    // PC: n_in, n_out, rows, per_expert_blks, cw(0), rpf(2).
                    let push = push_u32s(&[
                        n_in as u32,
                        n_out as u32,
                        rows as u32,
                        (per_expert / blk) as u32,
                        0,
                        2,
                    ]);
                    ctx.run(p.pl, ds2, p.pipe, &push, 1, n_out.div_ceil(2) as u32, rows as u32)?;
                    return Ok(());
                }
            }
            let idb = self.fbuf(ids)?;
            let wbufs = self.weight_bufs(&mut ctx, w)?;
            let per_expert = w.data.len() / ne;
            let (kb, gb, dbuf) = self.ensure_shared(&mut ctx)?;
            let chunk_words = (ctx.max_ssbo / 4) as u32;
            let p = self.pipeline(&mut ctx, Slot::FnMoeIds)?;
            // PC 선언순: n_in, n_out, xq_w, ty, rows, per_expert, chunk_words.
            let push = push_u32s(&[
                n_in as u32,
                n_out as u32,
                xq_w as u32,
                ty,
                rows as u32,
                per_expert as u32,
                chunk_words,
            ]);
            // 바인딩순: W0..7, xq, out, ktab, grid3s, ids.
            let mut binds: Vec<vk::Buffer> = wbufs.clone();
            while binds.len() < 8 {
                binds.push(dbuf);
            }
            binds.push(xq);
            binds.push(ob);
            binds.push(kb);
            binds.push(gb);
            binds.push(idb);
            let ds2 = ctx.bind_ds(&p, &binds)?;
            ctx.run(p.pl, ds2, p.pipe, &push, n_out as u32, rows as u32, 1)?;
            return Ok(());
        }
        // 2c) 그룹 타일 (plans/88 P2) — 프리필 대량행: 디바이스 그룹화(세대
        // 캐시 — 같은 라우팅의 3개 GEMM이 테이블 공유) + 16×16 타일(패딩
        // 도메인, 타일=전문가, x는 perm_pad 간접 판독 — 게더 패스 불필요) +
        // inv_pad 산란. 호스트 ids 왕복·512 전문가 루프 전부 소거(B2).
        // 산술: hip ge/w_ids 열과 동일 표현식 — 프리필 클래스 재기록 대상.
        // 강제 스위치: LLM170_MOE_TILE=0 (구 호스트 그룹화 경로).
        let tile_ok = rows > 0
            && match w.ty {
                GgmlType::Q4K => n_in <= 4096,
                GgmlType::Q5_1 => n_in <= 2048,
                // plans/89 P1.1c — q8_0/q5_K MoE 역할(UD-Q4_K_XL 혼합)도 타일로:
                // 레거시 512-전문가 gemv3 루프([ts] gemv 1101ms/청크) 소거.
                GgmlType::Q8_0 => n_in <= 4096,
                GgmlType::Q5K => n_in <= 4096,
                _ => false,
            };
        if tile_ok {
            let idb = self.fbuf(ids)?;
            let wbufs = self.weight_bufs(&mut ctx, w)?;
            let per_expert = w.data.len() / ne;
            let bound = rows + 16 * ne;
            let (_, _, dbuf) = self.ensure_shared(&mut ctx)?;
            let chunk_words = (ctx.max_ssbo / 4) as u32;
            let generation = self.moe_gen.load(std::sync::atomic::Ordering::Relaxed);
            let hit = {
                let g = self.moe_grp.lock();
                g.as_ref().is_some_and(|g| g.generation == generation && g.rows == rows && g.ids_h == ids)
            };
            if !hit {
                // 테이블 성장(단일 상한 bound — 그룹 커널이 [rp,bound)를 0채움).
                crate::rawvk::context::site::scope("moe_grp", || -> Result<(), String> {
                    let mut g = self.moe_grp.lock();
                    let e = g.get_or_insert_with(|| MoeGrp {
                        generation: 0, rows: 0, ids_h: 0, bound: 0,
                        off: vkbuf_null(), rows_pad: vkbuf_null(), tilexp: vkbuf_null(),
                        perm: vkbuf_null(), inv: vkbuf_null(), inv_pad: vkbuf_null(),
                        rowexp: vkbuf_null(), perm_pad: vkbuf_null(),
                        yg: vkbuf_null(), yg_rows: 0,
                    });
                    if e.bound < bound {
                        e.rowexp = ctx.alloc_host(bound * 4)?;
                        e.perm_pad = ctx.alloc_host(bound * 4)?;
                        e.tilexp = ctx.alloc_host((bound / 16 + 1) * 4)?;
                    }
                    if e.rows < rows {
                        e.perm = ctx.alloc_host(rows * 4)?;
                        e.inv = ctx.alloc_host(rows * 4)?;
                        e.inv_pad = ctx.alloc_host(rows * 4)?;
                    }
                    e.off = ctx.alloc_host((ne + 1) * 4)?;
                    e.rows_pad = ctx.alloc_host(8)?;
                    Ok(())
                })?;
                let pg = self.pipeline(&mut ctx, Slot::FnMoeGroup)?;
                let (ob_, rpb, txb, pmb, ivb, ivpb, rxb, ppb) = {
                    let g = self.moe_grp.lock();
                    let g = g.as_ref().unwrap();
                    // 바인딩순 = 셰이더 선언순: off, rows_pad, tilexp, perm, inv,
                    // inv_pad, rowexp, perm_pad (초판이 inv 를 건너뛰어 전부 어긋남).
                    (g.off.buf, g.rows_pad.buf, g.tilexp.buf, g.perm.buf, g.inv.buf, g.inv_pad.buf, g.rowexp.buf, g.perm_pad.buf)
                };
                let dsg = ctx.bind_ds(&pg, &[idb, ob_, rpb, txb, pmb, ivb, ivpb, rxb, ppb])?;
                let push = push_u32s(&[ne as u32, rows as u32, bound as u32]);
                ctx.run(pg.pl, dsg, pg.pipe, &push, 1, 1, 1)?;
                {
                    let mut g = self.moe_grp.lock();
                    let gi = g.as_mut().unwrap();
                    gi.generation = generation;
                    gi.rows = rows;
                    gi.ids_h = ids;
                    gi.bound = gi.bound.max(bound);
                }
                if std::env::var_os("LLM170_MOE_GCHECK").is_some() {
                    // 진단: 그룹 테이블 불변식 검증(전문가 내 순서는 atomic이라
                    // 비결정 — 순서 무관 불변식으로 판정).
                    ctx.end_batch_wait()?;
                    let idv: Vec<u32> = {
                        let g = self.framebufs.lock();
                        let b = g.get(&ids).ok_or("ids 핸들 없음")?;
                        unsafe { std::slice::from_raw_parts(b.ptr as *const u32, rows) }.to_vec()
                    };
                    let gg = self.moe_grp.lock();
                    let gg = gg.as_ref().unwrap();
                    let rd = |b: &VkBuf, n: usize| unsafe {
                        std::slice::from_raw_parts(b.ptr as *const u32, n)
                    };
                    let dev_perm_pad = rd(&gg.perm_pad, bound);
                    let dev_inv_pad = rd(&gg.inv_pad, rows);
                    let dev_rowexp = rd(&gg.rowexp, bound);
                    let mut hoff = vec![0usize; ne + 1];
                    for &e in &idv { hoff[(e as usize).min(ne - 1) + 1] += 1; }
                    for e in 0..ne { hoff[e + 1] += hoff[e]; }
                    let mut hpoff = vec![0usize; ne + 1];
                    for e in 0..ne { hpoff[e + 1] = hpoff[e] + (hoff[e + 1] - hoff[e]).div_ceil(16) * 16; }
                    let rows_pad = hpoff[ne].max(16);
                    let rp_dev = unsafe { std::slice::from_raw_parts(gg.rows_pad.ptr as *const u32, 2) }[0] as usize;
                    let rp_dbg = unsafe { std::slice::from_raw_parts(gg.rows_pad.ptr as *const u32, 2) };
                    let mut bad = 0usize;
                    let mut seen = vec![false; rows];
                    if rp_dev != rows_pad {
                        eprintln!("[gcheck] rows_pad dev={rp_dev} host={rows_pad} dbg1={}", rp_dbg[1]);
                        bad += 1;
                    }
                    for pd in 0..rows_pad {
                        let e = dev_rowexp[pd] as usize;
                        if e >= ne || !(hpoff[e]..hpoff[e + 1]).contains(&pd) {
                            if bad < 6 { eprintln!("[gcheck] rowexp[{pd}]={e} 세그 불일치"); }
                            bad += 1;
                            continue;
                        }
                        let i = pd - hpoff[e];
                        let r = hoff[e + 1] - hoff[e];
                        let src = dev_perm_pad[pd] as usize;
                        if i < r {
                            if src >= rows || idv[src] as usize != e {
                                if bad < 6 { eprintln!("[gcheck] perm_pad[{pd}]={src} 전문가 불일치(e={e})"); }
                                bad += 1;
                            } else if seen[src] {
                                if bad < 6 { eprintln!("[gcheck] perm_pad[{pd}]={src} 중복"); }
                                bad += 1;
                            } else {
                                seen[src] = true;
                            }
                            if dev_inv_pad[src] as usize != pd {
                                if bad < 10 { eprintln!("[gcheck] inv_pad[{src}]={} != pd={pd}", dev_inv_pad[src]); }
                                bad += 1;
                            }
                        } else if src != 0 {
                            if bad < 6 { eprintln!("[gcheck] 패딩 perm_pad[{pd}]={src} != 0"); }
                            bad += 1;
                        }
                    }
                    let miss = seen.iter().filter(|s| !**s).count();
                    if miss > 0 && bad < 10 {
                        eprintln!("[gcheck] 커버 누락 {miss}행");
                    }
                    let dev_off2 = rd(&gg.off, ne + 1);
                    eprintln!("[gcheck] rows={rows} rows_pad={rows_pad} bad={bad} miss={miss} off[ne]={} off[0..3]={:?} rowexp[0..6]={:?} perm_pad[0..6]={:?}",
                        dev_off2[ne], &dev_off2[..3], &dev_rowexp[..6], &dev_perm_pad[..6]);
                }
            }
            let (rxb, rpb, ppb, ivb, ygb) = {
                let mut g = self.moe_grp.lock();
                let gi = g.as_mut().unwrap();
                let need_yg = gi.bound * n_out * 4;
                if gi.yg.bytes < need_yg {
                    gi.yg = crate::rawvk::context::site::scope("moe_grp", || ctx.alloc(need_yg))?;
                    gi.yg_rows = gi.bound;
                }
                (gi.rowexp.buf, gi.rows_pad.buf, gi.perm_pad.buf, gi.inv_pad.buf, gi.yg.buf)
            };
            // plans/89 P1.1b/d — coopmat 타일 우선(q4_K/q5_1): 스칼라 16×16 판
            // 대신 f16 coopMatMulAdd 전문가-블록 판(WG() 25비트 함정 제거판,
            // 스케일 분리 + f32 드레인). 킬스위치 LLM170_VK_MOECM=0.
            let cm_on = wbufs.len() == 1
                && std::env::var("LLM170_VK_MOECM").map(|v| v == "1").unwrap_or(false);
            let slot = match (w.ty, cm_on) {
                (GgmlType::Q4K, true) => Slot::FnMoeTileQ4kCm,
                (GgmlType::Q5_1, _q51cm) if std::env::var("LLM170_VK_Q51CM").map(|v| v == "1").unwrap_or(false) => Slot::FnMoeTileQ51Cm,
                (GgmlType::Q4K, _) => Slot::FnMoeTileQ4K,
                (GgmlType::Q5_1, _) => Slot::FnMoeTileQ51,
                (GgmlType::Q8_0, _) => Slot::FnMoeTileQ8,
                _ => Slot::FnMoeTileQ5k,
            };
            let p = self.pipeline(&mut ctx, slot)?;
            let mut binds: Vec<vk::Buffer> = wbufs.clone();
            while binds.len() < 8 {
                binds.push(dbuf);
            }
            binds.push(xq);
            binds.push(ygb);
            binds.push(rxb);
            binds.push(rpb);
            binds.push(ppb);
            let ds2 = ctx.bind_ds(&p, &binds)?;
            // PC 선언순: n_in, n_out, per_expert_bytes, chunk_words, xq_w, mode, rows.
            let push = push_u32s(&[
                n_in as u32, n_out as u32, per_expert as u32, chunk_words, xq_w as u32,
                std::env::var("LLM170_MTC_MODE").ok().and_then(|v| v.parse().ok()).unwrap_or(0u32),
                rows as u32,
            ]);
            let (gx, gy) = if cm_on {
                (n_out.div_ceil(128) as u32, bound.div_ceil(16) as u32)
            } else {
                (n_out.div_ceil(16) as u32, bound.div_ceil(16) as u32)
            };
            ctx.run(p.pl, ds2, p.pipe, &push, gx, gy, 1)?;
            // 산란: out[i] = yg[inv_pad[i]] (행 순서 복원 — SiluMul/wsum 소비).
            let ps = self.pipeline(&mut ctx, Slot::PermuteF32)?;
            let dss = ctx.bind_ds(&ps, &[ygb, ivb, ob])?;
            let push = push_u32s(&[n_out as u32, rows as u32]);
            ctx.run(ps.pl, dss, ps.pipe, &push, rows as u32, 1, 1)?;
            return Ok(());
        }
        // 1) ids 판독(호스트 그룹화) — direct-ids 가 걸러준 프리필 대량행만.
        // 가드를 내린 뒤 d2h 드레인(d2h 블록이 스스로 ctx 를 잡는다).
        drop(ctx);
        // 1) ids 판독(호스트 그룹화) — off/perm/inv 구축.
        let idv: Vec<u32> = {
            {
                let mut c = self.ctx.lock();
                if c.batching.load(std::sync::atomic::Ordering::Relaxed) {
                    let _ = c.end_batch_wait();
                }
            }
            let g = self.framebufs.lock();
            let b = g.get(&ids).ok_or("vk moe: ids 핸들 없음")?;
            unsafe { std::slice::from_raw_parts(b.ptr as *const u32, rows) }.to_vec()
        };
        let mut off = vec![0usize; ne + 1];
        for &e in &idv {
            off[(e as usize).min(ne - 1) + 1] += 1;
        }
        for e in 0..ne {
            off[e + 1] += off[e];
        }
        let mut cur = off[..ne].to_vec();
        let mut perm = vec![0u32; rows];
        for (i, &e) in idv.iter().enumerate() {
            let e = (e as usize).min(ne - 1);
            perm[cur[e]] = i as u32;
            cur[e] += 1;
        }
        let mut inv = vec![0u32; rows];
        for (p_, &orig) in perm.iter().enumerate() {
            inv[orig as usize] = p_ as u32;
        }
        let mut ctx = self.ctx.lock();
        // 3) MoE 스크래치 (perm u32, xg u32, iv u32, yg f32) — 필요시 성장.
        // xg 행 스트라이드는 16B 정렬로 패딩 — 전문가별 디스크립터 오프셋이
        // minStorageBufferOffsetAlignment를 만족해야 한다(down n_in=640의
        // 760B 행은 8 mod 16 → 미정렬 오프셋에서 오염 판독).
        let xq_w_pad = (xq_w + 3) & !3;
        let need_xg = rows * xq_w_pad * 4;
        let need_yg = rows * n_out * 4;
        {
            let mut g = self.moebufs.lock();
            // plans/86 §5 — 컴포넌트별 성장. 종전 전부-만족 검사는 gate(yg 52MB)와
            // down(yg 210MB)이 크기 계급을 달리해 매호출 4버퍼 재할당 → 48층×3gemm×
            // 청크마다 누적(pp4096 실측 33.6GiB, 카브아웃 오버플로→GTT 전이→OOM).
            let e = g.get_or_insert_with(|| (vkbuf_null(), vkbuf_null(), vkbuf_null(), vkbuf_null()));
            crate::rawvk::context::site::scope("moe_scratch", || -> Result<(), String> {
                if e.0.bytes < rows * 4 {
                    e.0 = ctx.alloc((rows * 4).max(1 << 16))?;
                }
                if e.1.bytes < need_xg {
                    e.1 = ctx.alloc(need_xg.max(1 << 16))?;
                }
                if e.2.bytes < rows * 4 {
                    e.2 = ctx.alloc((rows * 4).max(1 << 16))?;
                }
                if e.3.bytes < need_yg {
                    e.3 = ctx.alloc(need_yg.max(1 << 16))?;
                }
                Ok(())
            })?;
        }
        let (pmb, xgb, ivb, ygb) = {
            let g = self.moebufs.lock();
            let r = g.as_ref().unwrap();
            (r.0.buf, r.1.buf, r.2.buf, r.3.buf)
        };
        unsafe {
            let g = self.moebufs.lock();
            let r = g.as_ref().unwrap();
            std::ptr::copy_nonoverlapping(perm.as_ptr(), r.0.ptr as *mut u32, rows);
            std::ptr::copy_nonoverlapping(inv.as_ptr(), r.2.ptr as *mut u32, rows);
        }
        // plans/84 B: 게더→전문가별 GEMV→스캐터를 배치 세션으로 — 비배치
        // run은 매 발사마다 제출+펜스 대기라 전문가 수만큼 동기가 걸린다
        // (프레임 경로 2.9배 열세의 주원인). 1회 제출로 묶는다.
        let batching = std::env::var_os("LLM170_VK_NOBATCH").is_none();
        if batching {
            ctx.begin_batch()?;
        }
        // 4) 게더: xg[p] = xq[perm[p]] (u32 행, dst 스트라이드 = 패딩)
        {
            let p = self.pipeline(&mut ctx, Slot::PermuteU32)?;
            let ds2 = ctx.bind_ds(&p, &[xq, pmb, xgb])?;
            let push = push_u32s(&[xq_w as u32, xq_w_pad as u32, rows as u32]);
            ctx.run(p.pl, ds2, p.pipe, &push, rows as u32, 1, 1)?;
        }
        // 5) 전문가별 GEMV — xg/yg 슬라이스 + 가중 전문가 오프셋.
        let wbufs = self.weight_bufs(&mut ctx, w)?;
        let per_expert = w.data.len() / ne;
        // plans/86 §1b 진단 — 전문가 오프셋/청크 기하 (LLM170_MOE_SYNC=1).
        if std::env::var_os("LLM170_MOE_SYNC").is_some() {
            let sizes: Vec<usize> = {
                let wc = self.wcache.lock();
                wc.get(&(w.data.as_ptr() as usize, w.data.len()))
                    .map(|bs| bs.iter().map(|b| b.bytes).collect())
                    .unwrap_or_default()
            };
            eprintln!(
                "# moe-geom ty={ty} ne={ne} per_expert={per_expert} chunks={} sizes={sizes:?} max_ssbo={} ids={idv:?}",
                wbufs.len(), ctx.max_ssbo,
            );
        }
        for e in 0..ne {
            let r = off[e + 1] - off[e];
            if r == 0 {
                continue;
            }
            let xq_off = (off[e] * xq_w_pad * 4) as u64;
            let out_off = (off[e] * n_out * 4) as u64;
            let w_off = (e * per_expert) as u64;
            self.gemv_run_off(&mut ctx, &wbufs, n_in, n_out, xq_w_pad, ty, r, xgb, ygb, xq_off, out_off, w_off)?;
        }
        // 6) 스캐터: out[inv^{-1}] — inv는 원본행→순열위치: out[i] = yg[inv[i]].
        {
            let p = self.pipeline(&mut ctx, Slot::PermuteF32)?;
            let ds2 = ctx.bind_ds(&p, &[ygb, ivb, ob])?;
            let push = push_u32s(&[n_out as u32, rows as u32]);
            ctx.run(p.pl, ds2, p.pipe, &push, rows as u32, 1, 1)?;
        }
        if batching {
            ctx.end_batch_wait()?;
        }
        Ok(())
    }
}

impl VkAcc {
    /// plans/88 P2 — 프레임 quant 출력용 디바이스 로컬 버퍼(성장 재할당).
    fn xq_dev_buf(&self, ctx: &mut VkCtx, need: usize) -> Result<vk::Buffer, String> {
        let mut g = self.xq_dev.lock();
        if !g.as_ref().map(|b| b.bytes >= need).unwrap_or(false) {
            *g = Some(crate::rawvk::context::site::scope("xq_dev", || ctx.alloc(need.max(1 << 20)))?);
        }
        Ok(g.as_ref().unwrap().buf)
    }

    /// plans/88 P1 — 프레임 op 진입 시 배치 재개(플러시 후 세그먼트 재결합).
    fn frame_resume_batch(&self, ctx: &mut VkCtx) {
        if self.frame_step_batch.load(std::sync::atomic::Ordering::Relaxed)
            && !ctx.batching.load(std::sync::atomic::Ordering::Relaxed)
        {
            let _ = ctx.begin_batch();
        }
    }

    /// plans/87 §3 — 슬롯별 GPU 시간 집계 덤프(LLM170_VK_TS=1 로 풀 생성).
    /// 엔진의 ktrace 틱 지점(디코드 스텝/프리필 종료)에서 호출된다.
    pub fn ts_tick(&self) {
        let mut ctx = self.ctx.lock();
        ctx.ts_report();
    }
}

impl llm170_core::matmul::FrameHost for VkAcc {
    fn ktrace_tick(&self) {
        self.ts_tick();
    }
    /// plans/84 B: 프레임 op군이 부분 구현(엘리먼트와이스+MoE) — 완성 전에는
    /// 옵트인(LLM170_VK_FRAME=1)일 때만 엔진이 프레임 경로에 들어온다.
    /// plans/86 §8 — 프레임 경로 완성(§1 정확성·§2 QSA 디바이스화·§5 성능) 후
    /// 기본 ON. 킬스위치 LLM170_VK_FRAME=0.
    fn frame_capable(&self) -> bool {
        std::env::var("LLM170_VK_FRAME").map(|v| v != "0").unwrap_or(true)
    }
    /// plans/85 §2 — 프레임 로짓 행별 argmax: fn_argmax_rows 2단 판.
    /// 동률 최저 인덱스 — CPU greedy_from과 동일 의미. 미구현이면 greedy
    /// 디코드 전체가 값경로 재연산으로 폴백했다(np/forward/multi 공통).
    fn frame_argmax_rows(&self, logits: u64, t: usize, vocab: usize) -> Result<Vec<u32>, String> {
        let lb = self.fbuf(logits)?;
        // WG당 256스레드×8원소 = 2048. stage1은 1워크그룹(256) 축소 — n_wg ≤ 256.
        let n_wg = vocab.div_ceil(2048);
        if n_wg > 256 || vocab == 0 || t == 0 {
            return Err(format!("vk frame_argmax_rows: 형상 초과 n_wg={n_wg} vocab={vocab} t={t}"));
        }
        let sc_bytes = 2 * n_wg * t * 4;
        let out_bytes = t * 4;
        let mut ctx = self.ctx.lock();
        // plans/88 P1 — 스텝 배치 플러시: 아래 2런치는 비배치 동기 실행 후
        // 호스트가 ob.ptr 을 직접 판독한다. 녹화만 된 커맨드를 먼저 실행.
        if ctx.batching.load(std::sync::atomic::Ordering::Relaxed) {
            ctx.end_batch_wait()?;
        }
        {
            let mut g = self.argmax_bufs.lock();
            let need = match g.as_ref() {
                None => true,
                Some(b) => b.0.bytes < sc_bytes || b.1.bytes < out_bytes,
            };
            if need {
                *g = Some(crate::rawvk::context::site::scope("argmax", || Ok::<_, String>((
                    ctx.alloc_host(sc_bytes.max(1 << 16))?,
                    ctx.alloc_host(out_bytes.max(4096))?,
                )))?);
            }
        }
        let (scb, ob) = {
            let g = self.argmax_bufs.lock();
            let b = g.as_ref().unwrap();
            (b.0.clone(), b.1.clone())
        };
        let p = self.pipeline(&mut ctx, Slot::FnArgmaxRows)?;
        let ds = ctx.bind_ds(&p, &[lb, scb.buf, ob.buf])?;
        let push0 = push_u32s(&[vocab as u32, 0u32, n_wg as u32]);
        ctx.run(p.pl, ds, p.pipe, &push0, n_wg as u32, t as u32, 1)?;
        let push1 = push_u32s(&[0u32, 1u32, n_wg as u32]);
        ctx.run(p.pl, ds, p.pipe, &push1, 1, t as u32, 1)?;
        // 비배치 run은 동기 — 안전한 직접 판독.
        let mut out = vec![0u32; t];
        unsafe { std::ptr::copy_nonoverlapping(ob.ptr as *const u32, out.as_mut_ptr(), t) };
        Ok(out)
    }

    /// plans/86 §2 — QSA q/k norm+rope in-place (hip qk_norm_rope 동일열).
    /// 상수(qn/kn 헤드 타일, cs 테이블)는 (ptr,len) 키로 1회 업로드 상주 —
    /// 프레임이 타일 Vec을 스텝 간 유지하므로 포인터가 곧 신원이다(hip 교훈).
    /// kq_scale=1.0(QSA 무척도 k 규약). hd ≤ 256(공유 스테이징 폭).
    #[allow(clippy::too_many_arguments)]
    fn frame_qk_norm_rope(
        &self,
        q: u64,
        k: u64,
        q_norm: &[f32],
        k_norm: &[f32],
        cs: &[f32],
        eps: f32,
        pos0: usize,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        n_rot: usize,
        t: usize,
    ) -> Result<(), String> {
        if hd > 256 {
            return Err(format!("vk frame_qk_norm_rope: hd={hd} (≤256 전용)"));
        }
        let mut ctx = self.ctx.lock();
        let qnb = self.qk_const(&mut ctx, q_norm)?;
        let knb = self.qk_const(&mut ctx, k_norm)?;
        let csb = self.qk_const(&mut ctx, cs)?;
        let (qb, kb) = (self.fbuf(q)?, self.fbuf(k)?);
        let p = self.pipeline(&mut ctx, Slot::FnQkNormRope)?;
        let ds2 = ctx.bind_ds(&p, &[qb, kb, qnb.buf, knb.buf, csb.buf])?;
        // PC 선언순: eps, kqs, pos, n_head, n_kv, hd, n_rot.
        let mut push = eps.to_le_bytes().to_vec();
        push.extend_from_slice(&1.0f32.to_le_bytes());
        push.extend_from_slice(&push_u32s(&[
            pos0 as u32, n_head as u32, n_kv as u32, hd as u32, n_rot as u32,
        ]));
        ctx.run(p.pl, ds2, p.pipe, &push, (n_head + n_kv) as u32, t as u32, 1)
    }

    /// 프레임 버퍼 — host-visible(alloc_host)로 직접 읽기/쓰기.
    /// 값경로 버퍼와 동일 정책(plans/29).
    fn frame_alloc(&self, len: usize) -> Result<u64, String> {
        if std::env::var_os("LLM170_VK_POOL").is_some_and(|v| v == "0") {
            let mut ctx = self.ctx.lock();
            let b = crate::rawvk::context::site::scope("frame", || ctx.alloc_host(len * 4))?;
            let h = self.frame_next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.framebufs.lock().insert(h, b);
            return Ok(h);
        }
        let need = len * 4;
        // 풀에서 최소 적합 버퍼 재활용 (할당 syscall·vk 객체 회피).
        let recycled = {
            let mut pool = self.frame_pool.lock();
            pool.iter()
                .enumerate()
                .filter(|(_, b)| b.bytes >= need)
                .min_by_key(|(_, b)| b.bytes)
                .map(|(i, _)| i)
                .map(|i| pool.swap_remove(i))
        };
        let b = match recycled {
            Some(b) => b,
            None => {
                let mut ctx = self.ctx.lock();
                crate::rawvk::context::site::scope("frame", || ctx.alloc_host(need))?
            }
        };
        let h = self.frame_next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.framebufs.lock().insert(h, b);
        Ok(h)
    }
    fn frame_free(&self, h: u64) -> Result<(), String> {
        let b = self.framebufs.lock().remove(&h);
        if let Some(b) = &b {
            // plans/87 §4 — 풀 반납 기록(사이트별 재사용 재고).
            llm170_diag::alloc::recycle("frame", b.bytes);
        }
        if let Some(b) = b {
            let mut pool = self.frame_pool.lock();
            let total: usize = pool.iter().map(|b| b.bytes).sum();
            if total + b.bytes <= FRAME_POOL_CAP {
                pool.push(b);
            }
            // 상한 초과분은 종전대로 보류(파괴 없음) — 풀이 총량을 막는다.
        }
        Ok(())
    }
    fn frame_write(&self, h: u64, data: &[f32]) -> Result<(), String> {
        let (ptr, _) = {
            let g = self.framebufs.lock();
            let b = g.get(&h).ok_or("vk frame_write: 핸들 없음")?;
            (b.ptr, std::marker::PhantomData::<()>)
        };
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut f32, data.len()) };
        Ok(())
    }
    fn frame_write_u32(&self, h: u64, data: &[u32]) -> Result<(), String> {
        let ptr = self.framebufs.lock().get(&h).ok_or("vk frame_write_u32: 핸들 없음")?.ptr;
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u32, data.len()) };
        Ok(())
    }
    fn frame_sync(&self) {
        // 비배치 run()은 호출마다 제출+펜스 대기(동기) — 추가 대기 불필요.
        // 배치 모드(값경로 begin_batch)에서만 플러시한다.
        let mut ctx = self.ctx.lock();
        if ctx.batching.load(std::sync::atomic::Ordering::Relaxed) {
            let _ = ctx.end_batch_wait();
        }
    }
    fn frame_read(&self, h: u64, out: &mut [f32]) -> Result<(), String> {
        self.frame_sync();
        let ptr = self.framebufs.lock().get(&h).ok_or("vk frame_read: 핸들 없음")?.ptr;
        unsafe { std::ptr::copy_nonoverlapping(ptr as *const f32, out.as_mut_ptr(), out.len()) };
        Ok(())
    }
    /// 상주 GEMM: 프레임 f32 버퍼 → (디바이스) quant → gemv. 업/다운 없음.
    fn frame_mm(&self, x: u64, w: &Weight, out: u64, t: usize) -> Result<(), String> {
        self.frame_mm_group(x, std::slice::from_ref(w), &[out], t)
    }
    fn frame_mm_group(&self, x: u64, ws: &[Weight], outs: &[u64], t: usize) -> Result<(), String> {
        let n_in = ws[0].n_in as usize;
        let xq_w = n_in / 4 + n_in / 32 + n_in / 16;
        let mut ctx = self.ctx.lock();
        self.frame_resume_batch(&mut ctx);
        let xb = self.fbuf(x)?;
        // plans/88 P1 — F32·BF16 멤버는 fn_mm_f32 가 프레임 f32 를 직접 소비.
        // 양자 멤버가 있을 때만 quant 를 돌린다(순수 f32 그룹의 스텝 낭비 제거).
        let dense_ty = |ty: llm170_gguf::GgmlType| -> Option<u32> {
            match ty {
                llm170_gguf::GgmlType::F32 => Some(0u32),
                llm170_gguf::GgmlType::Bf16 => Some(1u32),
                _ => None,
            }
        };
        let has_quant = ws.iter().any(|w| vk_ty(w.ty).is_some());
        let xq = if has_quant {
            let xq = self.xq_dev_buf(&mut ctx, t * xq_w * 4)?;
            let p = self.pipeline(&mut ctx, Slot::Quant)?;
            let ds2 = ctx.bind_ds(&p, &[xb, xq])?;
            let push = push_u32s(&[n_in as u32, t as u32, xq_w as u32]);
            ctx.run(p.pl, ds2, p.pipe, &push, ((n_in / 32) + 63) as u32 / 64, t as u32, 1)?;
            xq
        } else {
            vk::Buffer::null()
        };
        let mut xs: Vec<Vec<f32>> = Vec::new();
        let mut need_pullback = false;
        for w in ws {
            if vk_ty(w.ty).is_none() && dense_ty(w.ty).is_none() {
                need_pullback = true;
                break;
            }
        }
        if need_pullback {
            if std::env::var_os("LLM170_VK_PULLDBG").is_some() {
                eprintln!("[pull] n_in={n_in} tys={:?}", ws.iter().map(|w| format!("{:?}", w.ty)).collect::<Vec<_>>());
            }
            // plans/84 B: 미지원 타입(f32 inject 등)은 값경로 폴백 — 프레임 f32를
            // 판독해 MatmulHost(CPU 포함)로 계산하고 out에 기록한다.
            drop(ctx);
            let mut flat = vec![0f32; t * n_in];
            self.frame_read(x, &mut flat)?;
            for ti in 0..t {
                xs.push(flat[ti * n_in..(ti + 1) * n_in].to_vec());
            }
            for (wi, w) in ws.iter().enumerate() {
                let mut outs_v = vec![vec![0f32; w.n_out as usize]; t];
                if vk_ty(w.ty).is_some() {
                    // 지원 타입도 여기선 일괄 값경로(호모지니어스 경로 유지)
                    self.matmul_batch(&xs, w, &mut outs_v)?;
                } else {
                    llm170_core::matmul::matmul_batch(&xs, w, &mut outs_v);
                }
                let mut flat_out = Vec::with_capacity(t * w.n_out as usize);
                for row in &outs_v {
                    flat_out.extend_from_slice(row);
                }
                self.frame_write(outs[wi], &flat_out)?;
            }
            return Ok(());
        }
        for (wi, w) in ws.iter().enumerate() {
            let n_out = w.n_out as usize;
            let ob = self.fbuf(outs[wi])?;
            let wbufs = self.weight_bufs(&mut ctx, w)?;
            match vk_ty(w.ty) {
                Some(ty) => {
                    if std::env::var_os("LLM170_VK_MMDBG").is_some() {
                        eprintln!("[mm] ty={ty} n_in={n_in} n_out={n_out} t={t} bytes={}", w.data.len());
                    }
                    // plans/88 P2 — 프리필(t≥2) 밀집 타일: gemv3 t-루프는
                    // 실측 ~5GB/s(gemv 6.6s/208tok). 타일(K-슬라이스 스테이징)로
                    // 대체 — 산술 클래스는 동일 표현식·스레드 직렬 누산.
                    // 스위치: LLM170_VK_DTILE=0 이면 종전 gemv.
                    // plans/89 P0.2 — 디코드(t<16) 밀집 GEMV를 llama dmmv
                    // 포트(q8b/q4b)로: f32 활성 직결(quant 불필요), 64스레드
                    // 2행 WG. [ts] 기준선 gemv 77ms/step — 272-329GB/s급으로
                    // 기대. 킬스위치 LLM170_VK_G8=0(종전 quant+gemv3).
                    if t < 16
                        && std::env::var("LLM170_VK_G8").map(|v| v != "0").unwrap_or(true)
                        && self.gemv8_dense(&mut ctx, &wbufs, n_in, n_out, t, ty, xb, ob)?
                    {
                        continue;
                    }
                    let dense_tile = t >= 2
                        && std::env::var_os("LLM170_VK_DTILE").map(|v| v != "0").unwrap_or(true)
                        && match w.ty {
                            GgmlType::Q8_0 => true,
                            GgmlType::Q4K => n_in <= 4096,
                            GgmlType::Q5_1 => n_in <= 2048,
                            _ => false,
                        };
                    if dense_tile {
                        let (_, _, dbuf) = self.ensure_shared(&mut ctx)?;
                        let chunk_words = (ctx.max_ssbo / 4) as u32;
                        let mut binds: Vec<vk::Buffer> = wbufs.clone();
                        while binds.len() < 8 {
                            binds.push(dbuf);
                        }
                        binds.push(xq);
                        binds.push(ob);
                        // plans/89 P1.1 — coopmat 타일 우선(q8_0/q4_K 밀집):
                        // decoder ms/128 패밀리(f16 coopMatMulAdd) 직접 재사용.
                        // 스칼라 K-슬라이스 타일은 ALU 바운드([ts] tile_q8
                        // 2818ms/청크). 킬스위치 LLM170_VK_CM=0.
                        if std::env::var("LLM170_VK_CM").map(|v| v != "0").unwrap_or(true)
                            && matches!(w.ty, GgmlType::Q8_0 | GgmlType::Q4K)
                            && wbufs.len() == 1
                        {
                            let big = t >= 128;
                            let slot = match (w.ty, big) {
                                (GgmlType::Q8_0, true) => Slot::TileQ8128Cm,
                                (GgmlType::Q8_0, false) => Slot::TileQ8msCm,
                                (_, true) => Slot::TileQ4k128Cm,
                                (_, false) => Slot::TileQ4kmsCm,
                            };
                            let step = if big { 128usize } else { 64 };
                            let p = self.pipeline(&mut ctx, slot)?;
                            let ds2 = ctx.bind_ds(&p, &binds)?;
                            let gx = (n_out as u32).div_ceil(64);
                            for tb in (0..t).step_by(step) {
                                let nt = (t - tb).min(step) as u32;
                                let push = if big {
                                    push_u32s(&[n_in as u32, n_out as u32, xq_w as u32, nt, 0, tb as u32])
                                } else {
                                    push_u32s(&[n_in as u32, n_out as u32, xq_w as u32, nt, tb as u32])
                                };
                                ctx.run(p.pl, ds2, p.pipe, &push, gx, 1, 1)?;
                            }
                            continue;
                        }
                        match w.ty {
                            GgmlType::Q8_0 => {
                                let p = self.pipeline(&mut ctx, Slot::FnTileQ8)?;
                                let ds2 = ctx.bind_ds(&p, &binds)?;
                                // PC: n_in, n_out, chunk_words, xq_w, t.
                                let push = push_u32s(&[
                                    n_in as u32, n_out as u32, chunk_words, xq_w as u32, t as u32,
                                ]);
                                ctx.run(p.pl, ds2, p.pipe, &push, n_out.div_ceil(16) as u32, t.div_ceil(16) as u32, 1)?;
                            }
                            _ => {
                                let slot = if w.ty == GgmlType::Q4K { Slot::FnMoeTileQ4K } else { Slot::FnMoeTileQ51 };
                                let p = self.pipeline(&mut ctx, slot)?;
                                let mut b2 = binds.clone();
                                b2.push(dbuf); // rowexp
                                b2.push(dbuf); // rows_pad
                                b2.push(dbuf); // perm_pad
                                let ds2 = ctx.bind_ds(&p, &b2)?;
                                // PC: n_in, n_out, per_expert(0), chunk_words, xq_w, mode=1, t.
                                let push = push_u32s(&[
                                    n_in as u32, n_out as u32, 0u32, chunk_words, xq_w as u32, 1u32, t as u32,
                                ]);
                                ctx.run(p.pl, ds2, p.pipe, &push, n_out.div_ceil(16) as u32, t.div_ceil(16) as u32, 1)?;
                            }
                        }
                        continue;
                    }
                    self.gemv_run(&mut ctx, &wbufs, n_in, n_out, xq_w, ty, t, xq, ob)?;
                }
                None => {
                    // plans/88 P1 — f32/BF16 밀식 GEMV(값폴백 소거).
                    let dty = dense_ty(w.ty).unwrap();
                    let (_, _, dbuf) = self.ensure_shared(&mut ctx)?;
                    // plans/89 P0.2 — 디코드(t<16)는 64스레드 판(mm_f32b):
                    // fn_mm_f32 256스레드 f64 트리는 512WG 지연바운드
                    // ([ts] 12ms/step = 0.4GB/s급). 킬스위치 LLM170_VK_MMB=0.
                    // plans/89 P1.2 — f32/BF16 프리필(t≥2) 타일: fn_mm_f32 그리드
                    // (n_out, t)의 가중 t-재판독(라우터 2.6GB/청크) 소거.
                    // 킬스위치 LLM170_VK_FT32=0.
                    if t >= 2
                        && wbufs.len() == 1
                        && std::env::var("LLM170_VK_FT32").map(|v| v != "0").unwrap_or(true)
                    {
                        let p = self.pipeline(&mut ctx, Slot::FnTileF32)?;
                        let mut binds: Vec<vk::Buffer> = wbufs.clone();
                        while binds.len() < 8 {
                            binds.push(dbuf);
                        }
                        binds.push(xb);
                        binds.push(ob);
                        let ds2 = ctx.bind_ds(&p, &binds)?;
                        let wpr = if dty == 0 { n_in } else { n_in / 2 };
                        let push = push_u32s(&[
                            n_in as u32, n_out as u32, t as u32, dty, wpr as u32,
                        ]);
                        ctx.run(p.pl, ds2, p.pipe, &push, (n_out as u32).div_ceil(16), (t as u32).div_ceil(16), 1)?;
                        continue;
                    }
                    let slot = if t < 16 && wbufs.len() == 1
                        && std::env::var("LLM170_VK_MMB").map(|v| v != "0").unwrap_or(true)
                    {
                        Slot::MmF32b
                    } else {
                        Slot::FnMmf32
                    };
                    let p = self.pipeline(&mut ctx, slot)?;
                    let mut binds: Vec<vk::Buffer> = wbufs.clone();
                    while binds.len() < 8 {
                        binds.push(dbuf);
                    }
                    binds.push(xb);
                    binds.push(ob);
                    let ds2 = ctx.bind_ds(&p, &binds)?;
                    let push = push_u32s(&[
                        n_in as u32, n_out as u32, t as u32, dty,
                        (ctx.max_ssbo / 4) as u32,
                    ]);
                    ctx.run(p.pl, ds2, p.pipe, &push, n_out as u32, t as u32, 1)?;
                }
            }
        }
        Ok(())
    }
    fn frame_op(&self, op: &llm170_core::matmul::FrameOp) -> Result<(), String> {
        use llm170_core::matmul::FrameOp as O;
        let t_cur = self.frame_t.load(std::sync::atomic::Ordering::Relaxed);
        let mut ctx = self.ctx.lock();
        self.frame_resume_batch(&mut ctx);
        match *op {
            O::RmsRows { x, w, out, eps, n, w_reps } => {
                let (xb, wb, ob) = (self.fbuf(x)?, self.fbuf(w)?, self.fbuf(out)?);
                let rows = w_reps * t_cur;
                let p = self.pipeline(&mut ctx, Slot::Rms)?;
                let ds2 = ctx.bind_ds(&p, &[xb, wb, ob])?;
                let mut push = push_u32s(&[n as u32, rows as u32, w_reps as u32]);
                push.extend_from_slice(&eps.to_le_bytes());
                ctx.run(p.pl, ds2, p.pipe, &push, rows as u32, 1, 1)?;
            }
            O::SiluDiv { t, div, n } => {
                let tb = self.fbuf(t)?;
                let p = self.pipeline(&mut ctx, Slot::SiluDiv)?;
                let ds2 = ctx.bind_ds(&p, &[tb])?;
                let mut push = push_u32s(&[n as u32]);
                push.extend_from_slice(&div.to_le_bytes());
                ctx.run(p.pl, ds2, p.pipe, &push, (n as u32).div_ceil(256), 1, 1)?;
            }
            O::SiluMul { g, u, out, n } => {
                let (gb, ub, ob) = (self.fbuf(g)?, self.fbuf(u)?, self.fbuf(out)?);
                let p = self.pipeline(&mut ctx, Slot::Silu)?;
                let ds2 = ctx.bind_ds(&p, &[gb, ub, ob])?;
                let push = push_u32s(&[n as u32]);
                ctx.run(p.pl, ds2, p.pipe, &push, (n as u32).div_ceil(256), 1, 1)?;
            }
            O::Scale { t, s, n } => {
                let tb = self.fbuf(t)?;
                let p = self.pipeline(&mut ctx, Slot::Scale)?;
                let ds2 = ctx.bind_ds(&p, &[tb])?;
                let mut push = push_u32s(&[n as u32]);
                push.extend_from_slice(&s.to_le_bytes());
                ctx.run(p.pl, ds2, p.pipe, &push, (n as u32).div_ceil(256), 1, 1)?;
            }
            O::CopyRows { src, dst, src_off, dst_off, n } => {
                let (sb, db) = (self.fbuf(src)?, self.fbuf(dst)?);
                let p = self.pipeline(&mut ctx, Slot::CopyRows)?;
                let ds2 = ctx.bind_ds(&p, &[sb, db])?;
                let push = push_u32s(&[n as u32, src_off as u32, dst_off as u32]);
                ctx.run(p.pl, ds2, p.pipe, &push, (n as u32).div_ceil(256), 1, 1)?;
            }
            O::BcastRows { src, dst, n, rows } => {
                let (sb, db) = (self.fbuf(src)?, self.fbuf(dst)?);
                let p = self.pipeline(&mut ctx, Slot::BcastRows)?;
                let ds2 = ctx.bind_ds(&p, &[sb, db])?;
                let push = push_u32s(&[n as u32, rows as u32]);
                ctx.run(p.pl, ds2, p.pipe, &push, (n as u32).div_ceil(256), 1, 1)?;
            }
            O::AxpyScaled { y, x, s, n } => {
                let (yb, xb, sb) = (self.fbuf(y)?, self.fbuf(x)?, self.fbuf(s)?);
                if t_cur <= 1 {
                    let p = self.pipeline(&mut ctx, Slot::AxpyT)?; // pp=n → s[0]와 동일
                    let ds2 = ctx.bind_ds(&p, &[yb, xb, sb])?;
                    let push = push_u32s(&[n as u32, n as u32]);
                    ctx.run(p.pl, ds2, p.pipe, &push, (n as u32).div_ceil(256), 1, 1)?;
                } else {
                    let pp = n / t_cur;
                    let p = self.pipeline(&mut ctx, Slot::AxpyT)?;
                    let ds2 = ctx.bind_ds(&p, &[yb, xb, sb])?;
                    let push = push_u32s(&[n as u32, pp as u32]);
                    ctx.run(p.pl, ds2, p.pipe, &push, (n as u32).div_ceil(256), 1, 1)?;
                }
            }
            O::HcGateMean { xn, gate, out, hc, n } => {
                let (xb, gb, ob) = (self.fbuf(xn)?, self.fbuf(gate)?, self.fbuf(out)?);
                let p = self.pipeline(&mut ctx, Slot::HcGateMean)?;
                let ds2 = ctx.bind_ds(&p, &[xb, gb, ob])?;
                let total = n * t_cur;
                let push = push_u32s(&[hc as u32, n as u32, total as u32]);
                ctx.run(p.pl, ds2, p.pipe, &push, (total as u32).div_ceil(128), 1, 1)?;
            }
            O::HcCombine { res, out, inj, hc, n, total: _ } => {
                let (rb, ob, ib) = (self.fbuf(res)?, self.fbuf(out)?, self.fbuf(inj)?);
                let p = self.pipeline(&mut ctx, Slot::HcCombine)?;
                let ds2 = ctx.bind_ds(&p, &[rb, ob, ib])?;
                let tn = n * t_cur;
                let push = push_u32s(&[hc as u32, n as u32, tn as u32]);
                ctx.run(p.pl, ds2, p.pipe, &push, (tn as u32).div_ceil(128), 1, 1)?;
            }
            O::NormGated { o, z, w, out, eps, d, n_h } => {
                let (ob, zb, wb, ub) = (self.fbuf(o)?, self.fbuf(z)?, self.fbuf(w)?, self.fbuf(out)?);
                let p = self.pipeline(&mut ctx, Slot::NormGatedSig)?;
                let ds2 = ctx.bind_ds(&p, &[ob, zb, wb, ub])?;
                let mut push = push_u32s(&[d as u32, n_h as u32]);
                push.extend_from_slice(&eps.to_le_bytes());
                ctx.run(p.pl, ds2, p.pipe, &push, (n_h * t_cur) as u32, 1, 1)?;
            }
            O::GdnBetaG { b, a, dtb, sa, bg, n_h } => {
                let (bb, ab, db, sb, gb) = (self.fbuf(b)?, self.fbuf(a)?, self.fbuf(dtb)?, self.fbuf(sa)?, self.fbuf(bg)?);
                let p = self.pipeline(&mut ctx, Slot::GdnBetaG)?;
                let ds2 = ctx.bind_ds(&p, &[bb, ab, db, sb, gb])?;
                let dr = n_h / t_cur.max(1);
                let push = push_u32s(&[n_h as u32, dr as u32]);
                // 판은 64스레드 — 128로 나누면 절반이 미기입된다(청크 크기별
                // 커버리지가 달라져 청크 불변성 위반의 원인이었다).
                ctx.run(p.pl, ds2, p.pipe, &push, (n_h as u32).div_ceil(64), 1, 1)?;
            }
            O::Sigmoid { t, n } => {
                let tb = self.fbuf(t)?;
                let p = self.pipeline(&mut ctx, Slot::EwSigmoid)?;
                let ds2 = ctx.bind_ds(&p, &[tb])?;
                let push = push_u32s(&[n as u32]);
                ctx.run(p.pl, ds2, p.pipe, &push, (n as u32).div_ceil(256), 1, 1)?;
            }
            O::Split3 { src, d0, d1, d2, n0, n1, n2 } => {
                let (sb, a0, a1, a2) = (self.fbuf(src)?, self.fbuf(d0)?, self.fbuf(d1)?, self.fbuf(d2)?);
                let p = self.pipeline(&mut ctx, Slot::Split3)?;
                let ds2 = ctx.bind_ds(&p, &[sb, a0, a1, a2])?;
                let push = push_u32s(&[n0 as u32, n1 as u32, n2 as u32]);
                let total = (n0 + n1 + n2) * t_cur;
                ctx.run(p.pl, ds2, p.pipe, &push, (total as u32).div_ceil(64), 1, 1)?;
            }
            O::L2Rows { x, eps, d, n } => {
                let xb = self.fbuf(x)?;
                let p = self.pipeline(&mut ctx, Slot::L2Rows)?;
                let ds2 = ctx.bind_ds(&p, &[xb])?;
                let mut push = push_u32s(&[d as u32]);
                push.extend_from_slice(&eps.to_le_bytes());
                let rows = (n / d).max(1);
                ctx.run(p.pl, ds2, p.pipe, &push, rows as u32, 1, 1)?;
            }
            O::L2Rows2Scale { q, k, eps, scale, d, n_group } => {
                let (qb, kb) = (self.fbuf(q)?, self.fbuf(k)?);
                let p = self.pipeline(&mut ctx, Slot::L2Rows2Scale)?;
                let ds2 = ctx.bind_ds(&p, &[qb, kb])?;
                let mut push = push_u32s(&[d as u32, n_group as u32]);
                push.extend_from_slice(&eps.to_le_bytes());
                push.extend_from_slice(&scale.to_le_bytes());
                ctx.run(p.pl, ds2, p.pipe, &push, n_group as u32, 1, 1)?;
            }
            O::GdnConv { qkv, cw, state, out, ch, k, t_len } => {
                let (qb, cb, sb, ob) = (self.fbuf(qkv)?, self.fbuf(cw)?, self.fbuf(state)?, self.fbuf(out)?);
                let binds3 = [qb, cb, sb, ob];
                if t_len >= k - 1 {
                    // 병렬 청크판 + 상태 갱신 2런치 (hip과 동일 구조)
                    let p = self.pipeline(&mut ctx, Slot::GdnConvT2)?;
                    let ds2 = ctx.bind_ds(&p, &binds3)?;
                    let push = push_u32s(&[ch as u32, k as u32, t_len as u32]);
                    ctx.run(p.pl, ds2, p.pipe, &push, (ch as u32).div_ceil(64), t_len as u32, 1)?;
                    let p2 = self.pipeline(&mut ctx, Slot::GdnConvState)?;
                    let ds3 = ctx.bind_ds(&p2, &[qb, sb])?;
                    let push2 = push_u32s(&[ch as u32, k as u32, t_len as u32]);
                    ctx.run(p2.pl, ds3, p2.pipe, &push2, (k - 1) as u32, (ch as u32).div_ceil(64), 1)?;
                } else {
                    // 짧은 꼬리: 순차판 (상태 회전 포함)
                    let p = self.pipeline(&mut ctx, Slot::GdnConvSeq)?;
                    let ds2 = ctx.bind_ds(&p, &binds3)?;
                    let push = push_u32s(&[ch as u32, k as u32, t_len as u32]);
                    ctx.run(p.pl, ds2, p.pipe, &push, (ch as u32).div_ceil(64), 1, 1)?;
                }
            }
            O::MoeTop10 { route, ids, wt, n_exp, k_sel } => {
                let (rb, ib, wb) = (self.fbuf(route)?, self.fbuf(ids)?, self.fbuf(wt)?);
                let p = self.pipeline(&mut ctx, Slot::MoeTop10)?;
                let ds2 = ctx.bind_ds(&p, &[rb, ib, wb])?;
                let push = push_u32s(&[n_exp as u32, k_sel as u32]);
                ctx.run(p.pl, ds2, p.pipe, &push, t_cur as u32, 1, 1)?;
                // plans/88 P2 — 라우팅 세대 증가: 그룹화 캐시 무효화 키.
                self.moe_gen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            O::MoeWeightedSum { ys, wt, out, k, n } => {
                let (yb, wb, ob) = (self.fbuf(ys)?, self.fbuf(wt)?, self.fbuf(out)?);
                let p = self.pipeline(&mut ctx, Slot::MoeWsum)?;
                let ds2 = ctx.bind_ds(&p, &[yb, wb, ob])?;
                let total = n * t_cur;
                let push = push_u32s(&[n as u32, k as u32, total as u32]);
                ctx.run(p.pl, ds2, p.pipe, &push, (total as u32).div_ceil(256), 1, 1)?;
            }
            ref other => return Err(format!("vk frame_op: 미지원 {other:?}")),
        }
        Ok(())
    }
}

impl llm170_core::matmul::MatmulHost for VkAcc {
    fn matmul_batch(
        &self,
        xs: &[Vec<f32>],
        w: &Weight,
        outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        let ty = match vk_ty(w.ty) {
            Some(t) => t,
            None => {
                llm170_core::matmul::matmul_batch(xs, w, outs);
                return Ok(());
            }
        };
        let n_in = w.n_in as usize;
        let n_out = w.n_out as usize;
        let t = xs.len();
        if std::env::var_os("LLM170_VK_MBDBG").is_some() {
            eprintln!("[mb] ty={:?} n_in={} n_out={} t={}", w.ty, w.n_in, w.n_out, t);
        }
        let xq_w = n_in / 4 + n_in / 32 + n_in / 16;
        let mut ctx = self.ctx.lock();
        let xq = self.value_buf(&mut ctx, &self.xbuf, t * xq_w * 4)?;
        let ob = self.value_buf(&mut ctx, &self.obuf, t * n_out * 4)?;
        self.quant_upload(&mut ctx, xs, n_in, xq)?;
        let wbufs = self.weight_bufs(&mut ctx, w)?;
        // 128행 타일 (q5_K, t≥2, env) — f16 fast 경로
        if ty == 13 && t >= 2 && std::env::var_os("LLM170_VK_TILE").is_some() {
            self.tile128_run(&mut ctx, &wbufs, n_in, n_out, xq_w, t, xq, ob, Slot::Tile128)?;
            self.download_out(outs, n_out, t);
            return Ok(());
        }
        // plans/84 B: q5_1 타일판 — FN 다운 질량 프리필(옵트인, f16 타일이라
        // GEMV와는 다른 정밀도 클래스: t<2와 t>=2 패밀리 갈림을 막으려 기본
        // 끔 — LLM170_VK_TILE_Q51=1).
        if ty == 7 && t >= 2 && std::env::var_os("LLM170_VK_TILE_Q51").is_some() {
            self.tile128_q51_run(&mut ctx, &wbufs, n_in, n_out, xq_w, t, xq, ob)?;
            self.download_out(outs, n_out, t);
            return Ok(());
        }
        self.gemv_run(&mut ctx, &wbufs, n_in, n_out, xq_w, ty, t, xq, ob)?;
        self.download_out(outs, n_out, t);
        Ok(())
    }

    /// 같은 입력 그룹: 업로드+양자화 1회 → GEMV 각각 → 개별 다운로드.
    fn matmul_group(
        &self,
        xs: &[Vec<f32>],
        ws: &[Weight],
        outs: &mut [Vec<Vec<f32>>],
    ) -> Result<(), String> {
        if ws.iter().any(|w| vk_ty(w.ty).is_none())
            || ws.iter().any(|w| w.n_in != ws[0].n_in)
        {
            for (w, out) in ws.iter().zip(outs.iter_mut()) {
                self.matmul_batch(xs, w, out)?;
            }
            return Ok(());
        }
        let n_in = ws[0].n_in as usize;
        let t = xs.len();
        let xq_w = n_in / 4 + n_in / 32 + n_in / 16;
        let mut ctx = self.ctx.lock();
        let xq = self.value_buf(&mut ctx, &self.xbuf, t * xq_w * 4)?;
        self.quant_upload(&mut ctx, xs, n_in, xq)?;
        // 배치: 모든 가중 GEMV 녹화 → 단일 제출 → 일괄 다운로드 (plans/19)
        let do_batch = std::env::var_os("LLM170_VK_NOBATCH").is_none();
        if do_batch {
            ctx.begin_batch()?;
        }
        let mut hosts: Vec<(*mut u8, usize, usize)> = Vec::with_capacity(ws.len()); // (ptr, n_out, ti스트라이드)
        for (wi, w) in ws.iter().enumerate() {
            let ty = vk_ty(w.ty).unwrap();
            let n_out = w.n_out as usize;
            let wbufs = self.weight_bufs(&mut ctx, w)?;
            // 가중별 독립 출력 버퍼 (그룹 세션 슬롯)
            let ob = {
                let mut g = self.gobufs.lock();
                while g.len() <= wi {
                    g.push(None);
                }
                if g[wi].as_ref().map(|b| b.bytes >= t * n_out * 4).unwrap_or(false) {
                    g[wi].as_ref().unwrap().buf
                } else {
                    let b = ctx.alloc_host(t * n_out * 4)?;
                    let buf = b.buf;
                    g[wi] = Some(b);
                    buf
                }
            };
            self.gemv_run(&mut ctx, &wbufs, n_in, n_out, xq_w, ty, t, xq, ob)?;
            let ptr = self.gobufs.lock()[wi].as_ref().unwrap().ptr;
            hosts.push((ptr, n_out, wi));
        }
        if do_batch {
            ctx.end_batch_wait()?;
        }
        for (ptr, n_out, wi) in hosts {
            let host = unsafe { std::slice::from_raw_parts(ptr as *const f32, t * n_out) };
            for ti in 0..t {
                outs[wi][ti].copy_from_slice(&host[ti * n_out..(ti + 1) * n_out]);
            }
        }
        Ok(())
    }

    fn matmul(&self, x: &[f32], w: &Weight, out: &mut [f32]) -> Result<(), String> {
        let xs = vec![x.to_vec()];
        let mut tmp = vec![vec![0.0f32; w.n_out as usize]];
        self.matmul_batch(&xs, w, &mut tmp)?;
        out.copy_from_slice(&tmp[0]);
        Ok(())
    }
}

impl llm170_core::matmul::EwOps for VkAcc {

    fn rms_norm(
        &self,
        xs: &[Vec<f32>],
        w: &[f32],
        eps: f32,
        outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        self.rms_norm_gpu(xs, w, eps, outs)
    }

    fn silu_mul(
        &self,
        gs: &[Vec<f32>],
        us: &[Vec<f32>],
        outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        self.silu_mul_gpu(gs, us, outs)
    }

    fn ffn_chain(
        &self,
        xs: &[Vec<f32>],
        gate_w: &Weight,
        up_w: &Weight,
        down_w: &Weight,
        xs_out: &mut [Vec<f32>],
    ) -> Result<(), String> {
        self.ffn_chain_gpu(xs, gate_w, up_w, down_w, xs_out)
    }

    /// plans/85 §1 — 디코드(t=1) shared expert gate+up: quant 1회 → gemv 2회
    /// → SiluMul. 스크래치 gh/uh는 풀 기반 frame_alloc/frame_free.
    /// frame_mm_group/frame_op가 각자 ctx를 잠그므로 여기엔 중첩 잠금이
    /// 없다(직전 시도의 self-deadlock 원인 — ctx.lock 보유 중 frame_alloc).
    fn shexp_gu(
        &self, x: u64, wg: &Weight, wu: &Weight, h: u64,
        _n_in: usize, n_hidden: usize,
    ) -> Result<(), String> {
        if std::env::var_os("LLM170_VK_SHEXP").is_some_and(|v| v == "0") {
            return Err("shexp_gu: 진단 킬스위치".into());
        }
        let gh = self.frame_alloc(n_hidden)?;
        let uh = self.frame_alloc(n_hidden)?;
        let r = self
            .frame_mm_group(x, &[*wg, *wu], &[gh, uh], 1)
            .and_then(|_| {
                self.frame_op(&llm170_core::matmul::FrameOp::SiluMul {
                    g: gh, u: uh, out: h, n: n_hidden,
                })
            });
        let _ = self.frame_free(gh);
        let _ = self.frame_free(uh);
        r
    }

    /// plans/85 §1 — 디코드(t=1) shared expert down+가산: gemv 1회 →
    /// mout += σ·dh (AxpyScaled — t=1이라 s[0] 판독과 정합).
    fn shexp_da(
        &self, h: u64, wd: &Weight, s: u64, mout: u64,
        n_in: usize, _n_hidden: usize,
    ) -> Result<(), String> {
        if self.frame_t.load(std::sync::atomic::Ordering::Relaxed) != 1 {
            return Err("shexp_da: t=1 전용 (frame_t≠1)".into());
        }
        let dh = self.frame_alloc(n_in)?;
        let r = self
            .frame_mm_group(h, &[*wd], &[dh], 1)
            .and_then(|_| {
                self.frame_op(&llm170_core::matmul::FrameOp::AxpyScaled {
                    y: mout, x: dh, s, n: n_in,
                })
            });
        let _ = self.frame_free(dh);
        r
    }

    /// plans/89 P1.4 — PLE 수학 디바이스판(디코드 t=1): hip q4_ple_* 3커널의
    /// VkAcc 발사. 링/워터마크·상수 캐시 (ptr,len) 동일 규약. 프리필(t>1)은
    /// Err → 엔진이 종전 호스트 브리지로.
    #[allow(clippy::too_many_arguments)]
    fn ple_math_dev(
        &self,
        res: u64,
        key: u64,
        value: u64,
        nk: &[f32],
        nq: &[f32],
        nc: &[f32],
        conv_w: &[f32],
        gated: u64,
        conv_out: u64,
        gate_out: u64,
        seq: usize,
        t: usize,
        eps: f32,
        n_embd: usize,
        hc: usize,
        kern: usize,
        dil: usize,
        hist: usize,
        host_ring: &[f32],
    ) -> Result<(), String> {
        if t != 1 {
            return Err("ple_math_dev: t=1 전용".into());
        }
        let hc_dim = hc * n_embd;
        let ring_bytes = hist * hc_dim * 4;
        let mut ctx = self.ctx.lock();
        self.frame_resume_batch(&mut ctx);
        // 링 + 워터마크(되감기면 호스트 링으로 리프레시).
        let rewind;
        let ringb;
        {
            let mut m = self.ple_rings.lock();
            let e = m.entry(seq).or_insert_with(|| (vkbuf_null(), 0));
            rewind = e.1 > t || e.0.ptr.is_null();
            e.1 = t;
            if e.0.ptr.is_null() {
                e.0 = ctx.alloc_host(ring_bytes)?;
            }
            if rewind {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        host_ring.as_ptr() as *const u8,
                        e.0.ptr,
                        hist * hc_dim * 4,
                    );
                }
            }
            ringb = e.0.buf;
        }
        // 상수 캐시 — 모델 가중 뷰(ptr,len 안정).
        let upload = |ctx: &mut VkCtx, s: &[f32]| -> Result<vk::Buffer, String> {
            let key = (s.as_ptr() as usize, s.len());
            let mut c = self.ple_consts.lock();
            if let Some(b) = c.get(&key) {
                return Ok(b.buf);
            }
            let b = ctx.alloc_host(s.len() * 4)?;
            unsafe { std::ptr::copy_nonoverlapping(s.as_ptr() as *const u8, b.ptr, s.len() * 4) };
            let buf = b.buf;
            c.insert(key, b);
            Ok(buf)
        };
        let nkb = upload(&mut ctx, nk)?;
        let nqb = upload(&mut ctx, nq)?;
        let ncb = upload(&mut ctx, nc)?;
        let cwb = upload(&mut ctx, conv_w)?;
        let rb = self.fbuf(res)?;
        let kb = self.fbuf(key)?;
        let vb = self.fbuf(value)?;
        let gb = self.fbuf(gated)?;
        let cob = self.fbuf(conv_out)?;
        let gob = self.fbuf(gate_out)?;
        // (1) gate+방송+그룹 norm.
        {
            let p = self.pipeline(&mut ctx, Slot::FnPleGate)?;
            let ds2 = ctx.bind_ds(&p, &[rb, kb, vb, nkb, nqb, ncb, gb, gob])?;
            let push = push_u32s(&[n_embd as u32, hc as u32, t as u32]);
            let mut p16 = eps.to_le_bytes().to_vec();
            p16.extend_from_slice(&push);
            ctx.run(p.pl, ds2, p.pipe, &p16, hc.div_ceil(8) as u32, t as u32, 1)?;
        }
        // (2) dilated conv + silu + 링 갱신.
        {
            let p = self.pipeline(&mut ctx, Slot::FnPleConv)?;
            let ds2 = ctx.bind_ds(&p, &[gb, cwb, ringb, cob])?;
            let push = push_u32s(&[
                hc_dim as u32, t as u32, kern as u32, dil as u32, hist as u32,
            ]);
            ctx.run(p.pl, ds2, p.pipe, &push, hc_dim.div_ceil(256) as u32, 1, 1)?;
        }
        // (3) 잔차.
        {
            let p = self.pipeline(&mut ctx, Slot::FnPleRes)?;
            let ds2 = ctx.bind_ds(&p, &[rb, vb, gob, cob])?;
            let push = push_u32s(&[n_embd as u32, hc as u32, t as u32]);
            ctx.run(p.pl, ds2, p.pipe, &push, n_embd.div_ceil(256) as u32, 1, 1)?;
        }
        Ok(())
    }
}


/// vk-gemv-check — VkAcc matmul vs CPU W4A8 미러 단일 텐서 검증 + 타이밍.
pub fn gemv_check(path: &str, tname: &str, t: usize) -> Result<String, String> {
    // plans/84 B: qwen4exp(Flash-Next, 멀티파트) 폴백 — arch 판별 후 단일 로드.
    enum AnyModel {
        Q35(llm170_core::qwen35::Model),
        Q4(llm170_core::qwen4exp::Model4),
    }
    let is_q4 = llm170_gguf::GgufFile::open(std::path::Path::new(path))
        .ok()
        .and_then(|g| g.arch().map(|a| a == "qwen4exp"))
        .unwrap_or(false);
    let model = if is_q4 {
        AnyModel::Q4(
            llm170_core::qwen4exp::Model4::load(std::path::Path::new(path))
                .map_err(|e| e.to_string())?,
        )
    } else {
        AnyModel::Q35(
            llm170_core::qwen35::Model::load(std::path::Path::new(path))
                .map_err(|e| e.to_string())?,
        )
    };
    let w = match &model {
        AnyModel::Q35(m) => m.w(tname).ok_or("텐서 없음")?,
        AnyModel::Q4(m) => m.w4(tname).map_err(|e| e.to_string())?,
    };
    let wref = &w;
    let n_in = w.n_in as usize;
    let acc = VkAcc::new()?;
    // ── quant 비트 검증: GPU xq vs CPU quantize_row_q8_ref ──
    {
        let mut seed2 = 0x1234abcdu64;
        let mut lcg2 = || {
            seed2 = seed2.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed2 >> 33) as f32 / 2147483648.0 - 0.5
        };
        let xrow: Vec<f32> = (0..n_in).map(|_| lcg2()).collect();
        let mut ctxg = acc.ctx.lock();
        let xq_w = n_in / 4 + n_in / 32 + n_in / 16;
        let xqb = ctxg.alloc_host(xq_w * 4)?;
        acc.quant_upload(&mut ctxg, std::slice::from_ref(&xrow), n_in, xqb.buf)?;
        let gpu: &[u32] =
            unsafe { std::slice::from_raw_parts(xqb.ptr as *const u32, xq_w) };
        let yref = llm170_core::quant::quantize_row_q8_ref(&xrow);
        // CPU 재구성: qs 워드 + d 비트 + s0/s1
        let mut qdiff = 0usize;
        let mut ddiff = 0usize;
        let mut sdiff = 0usize;
        let nwords = n_in / 4;
        let nblk = n_in / 32;
        for b in 0..nblk {
            let d_cpu = yref[b].d.to_bits();
            let d_gpu = gpu[nwords + b];
            if d_cpu != d_gpu {
                ddiff += 1;
                if ddiff <= 3 {
                    let mut amax = 0.0f32;
                    for &v in &xrow[b * 32..b * 32 + 32] {
                        amax = amax.max(v.abs());
                    }
                    let (cpu_d, gpu_d, rust_d, f64d) = (d_cpu, d_gpu, (amax / 127.0f32).to_bits(), (amax as f64 / 127.0).to_bits() as u32);
                    eprintln!("dblk{b}: amax={amax:e} cpu_d={cpu_d:08x} gpu_d={gpu_d:08x} rust_d={rust_d:08x} f64lo={f64d:08x}");
                }
            }
            let mut s0 = 0u32;
            let mut s1 = 0u32;
            for wi in 0..8 {
                let mut word = 0u32;
                for k in 0..4 {
                    let qv = yref[b].qs[wi * 4 + k] as i8 as i32 as u32;
                    word |= (qv & 0xFF) << (8 * k);
                }
                if word != gpu[b * 8 + wi] {
                    qdiff += 1;
                }
                // sd(서브바이트 차분 카운트)는 진단 전용으로 제거됨(2026-09-17).
                let bytes: i32 = (0..4)
                    .map(|k| ((gpu[b * 8 + wi] >> (8 * k)) & 0xFF) as i32)
                    .fold(0i32, |a, v| a + ((v << 24) >> 24));
                if wi < 4 {
                    s0 = s0.wrapping_add(bytes as u32);
                } else {
                    s1 = s1.wrapping_add(bytes as u32);
                }
            }
            let qsb = nwords + nblk;
            if s0 != gpu[qsb + b * 2] || s1 != gpu[qsb + b * 2 + 1] {
                sdiff += 1;
            }
        }
        eprintln!(
            "quant-bits: qs워드 {qdiff}/{nwords} d {ddiff}/{nblk} s {sdiff}/{nblk} 상이"
        );
    }
    let mut seed = 0x9e3779b9u64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as f32 / 2147483648.0 - 0.5
    };
    let xs: Vec<Vec<f32>> = (0..t).map(|_| (0..n_in).map(|_| lcg()).collect()).collect();
    let mut outs = vec![vec![0.0f32; w.n_out as usize]; t];
    acc.matmul_batch(&xs, wref, &mut outs)?;
    let t0 = std::time::Instant::now();
    for _ in 0..10 {
        acc.matmul_batch(&xs, wref, &mut outs)?;
    }
    let dt = t0.elapsed().as_secs_f64() / 10.0;
    eprintln!(
        "vk-gemv-time: {} {:.2}ms → {:.1}GB/s ({}B 가중)",
        tname,
        dt * 1e3,
        wref.data.len() as f64 / dt / 1e9,
        wref.data.len()
    );
    let mut ref_outs = vec![vec![0.0f32; w.n_out as usize]; t];
    llm170_core::matmul::matmul_batch(&xs, wref, &mut ref_outs);
    let mut mx = 0f64;
    let mut rel = 0f64;
    let mut ndiff = 0usize;
    let mut ulp_hist = std::collections::HashMap::<i64, usize>::new();
    for (a, b) in outs.iter().zip(ref_outs.iter()) {
        for (x, y) in a.iter().zip(b.iter()) {
            if x.to_bits() != y.to_bits() {
                ndiff += 1;
                let ulp = (x.to_bits() as i64 - y.to_bits() as i64).abs();
                *ulp_hist.entry(ulp).or_insert(0) += 1;
            }
            let d = (x - y).abs() as f64;
            if d > mx {
                mx = d;
            }
            let r = d / y.abs().max(1.0) as f64;
            if r > rel {
                rel = r;
            }
        }
    }
    let hist: Vec<String> = {
        let mut v: Vec<(i64, usize)> = ulp_hist.into_iter().collect();
        v.sort_by_key(|b| std::cmp::Reverse(b.1));
        v.iter().take(4).map(|(u, c)| format!("{c}x{u}ulp")).collect()
    };
    let ia = outs[0].iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i);
    let ib = ref_outs[0].iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i);
    Ok(format!(
        "vk-gemv {tname} t={t}: max|D|={mx:.3e} maxrel={rel:.2e} argmax {ia:?}=={ib:?} {} | bits {ndiff}/{} differ, top {hist:?}",
        if ia == ib { "★" } else { "MISMATCH" },
        outs.len() * w.n_out as usize
    ))
}

/// 부록87: llama matmul_q5_k_f16.spv 직접 로드 격리 측정 (t≥2 프리필 GEMM).
pub fn vk_mmq_check(path: &str, tname: &str, t: usize) -> Result<String, String> {
    let model = llm170_core::qwen35::Model::load(std::path::Path::new(path))
        .map_err(|e| e.to_string())?;
    let w = model.w(tname).ok_or("텐서 없음")?;
    let n_in = w.n_in as usize;
    let n_out = w.n_out as usize;
    if std::env::var_os("VK_DUMP_W").is_some() {
        let _ = std::fs::write("/tmp/q5k_w.bin", w.data);
        eprintln!("W 더프: {}B n_in={} n_out={}", w.data.len(), n_in, n_out);
    }
    let spv_name = std::env::var("VKMMQ_SPV").unwrap_or_else(|_| "matmul_q5_k_f16".into());
    let b_is_f32 = spv_name.ends_with("_f32") || spv_name.contains("_f32_");
    let spv = if spv_name.contains('/') {
        std::fs::read(&spv_name).map_err(|e| e.to_string())?
    } else {
        std::fs::read(format!("/home/yoon/local_llm/llama.cpp-master/build-vulkan/ggml/src/ggml-vulkan/vulkan-shaders.spv/{}.spv", spv_name))
            .map_err(|e| e.to_string())?
    };
    let acc = VkAcc::new()?;
    let mut ctxg = acc.ctx.lock();
    // 버퍼: A=가중(호스트맵→h2d는 run 전 복사), B=f16 y, D=f32 out
    let ab_vram = std::env::var("VKMMQ_VRAM").map(|v| v=="1").unwrap_or(false);
    let ab = if ab_vram {
        let mut b = ctxg.alloc(w.data.len())?;
        unsafe { std::ptr::copy_nonoverlapping(w.data.as_ptr(), b.ptr, w.data.len()); }
        ctxg.unmap(&mut b)?;
        b
    } else {
        let mut b = ctxg.alloc_host(w.data.len())?;
        unsafe { std::ptr::copy_nonoverlapping(w.data.as_ptr(), b.ptr, w.data.len()); }
        ctxg.unmap(&mut b)?;
        b
    };
    let mut ybuf: Vec<u16> = Vec::with_capacity(n_in * t);
    let mut seed = 0x1234u64;
    let mut lcg = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 33) as f32 / 2147483648.0 - 0.5 };
    // y [K][N] f16 (stride_b = K) — CPU 참조와 동일값 사용
    let mut yf: Vec<f32> = Vec::with_capacity(n_in * t);
    for _ in 0..n_in * t { let v = lcg(); yf.push(v); ybuf.push(hf(v)); }
    let b_bytes = if b_is_f32 { n_in * t * 4 } else { n_in * t * 2 };
    let mut bb = ctxg.alloc_host(b_bytes)?;
    if b_is_f32 {
        unsafe { std::ptr::copy_nonoverlapping(yf.as_ptr() as *const u8, bb.ptr, b_bytes); }
    } else {
        unsafe { std::ptr::copy_nonoverlapping(ybuf.as_ptr() as *const u8, bb.ptr, b_bytes); }
    }
    ctxg.unmap(&mut bb)?;
    eprintln!("bc: allocs ok");
    let db = ctxg.alloc_host(n_out * t * 4)?;
    // l 파이프라인 (non-cm, subgroup=64, gfx1151): ids 0..10 + ALIGNED=0
    let sp = std::env::var("VKMMQ_SPEC").unwrap_or_else(|_| "l".into());
    let is_cm1 = spv_name.contains("_cm1");
    let spec: Vec<u32> = match sp.as_str() {
        "m" => vec![128, 64, 64, 32, 64, 32, 2, 4, 2, 1, 64, 0],
        "s" => vec![64, 32, 32, 32, 32, 32, 2, 2, 2, 1, 64, 0],
        "c" => vec![128, 128, 32, 32, 64, 32, 2, 4, 4, 1, 64, 0],
        "m32" => vec![128, 64, 64, 32, 32, 32, 2, 4, 2, 1, 32, 0],
        "l32" => vec![128, 128, 128, 32, 64, 64, 2, 4, 4, 1, 32, 0],
        "cc" => vec![128, 64, 32, 32, 64, 32, 2, 4, 4, 1, 64, 0],
        "mini" => vec![32, 32, 16, 32, 32, 16, 1, 4, 4, 1, 32, 0],
        "ls" => vec![256, 128, 128, 32, 64, 64, 2, 16, 16, 16, 64, 1],  // AMD RADV l-warptile_mmq
        "ms" => vec![128, 64, 64, 32, 64, 32, 2, 16, 16, 16, 64, 1],    // m-warptile_mmq
        "ss" => vec![64, 32, 32, 32, 32, 32, 2, 16, 16, 16, 64, 1],     // s-warptile_mmq
        "def" => vec![64, 64, 64, 16, 32, 32, 2, 4, 2, 1, 32, 0],  // spv 기본값 (부록87 해독)
        "cm1" => vec![128, 128, 128, 16, 128, 64, 2, 16, 16, 16, 64, 0],
        _ => vec![128, 128, 128, 32, 128, 64, 2, 4, 4, 1, 64, 0],
    };
    eprintln!("bc: y/ab 채움");
    let (_dsl, pl, _dp, ds, pipe) = ctxg.pipeline_spec_fg(&spv, 3, 17 * 4, &spec, is_cm1)?;
    eprintln!("bc: 파이프라인 ok");
    let bufs = [ab.buf, bb.buf, db.buf];
    ctxg.bind_bufs(ds, &bufs);
    // push: M,N,K,stride_a=K,stride_b=K,stride_d=M,batch 0들 + k_split=1 등
    let mut pc: Vec<u32> = vec![
        n_out as u32, t as u32, n_in as u32,      // M, N, K
        n_in as u32, n_in as u32, n_out as u32,   // stride_a=K, stride_b=K, stride_d=M
        0, 0, 0,                                  // batch strides
        0, 1, n_in as u32,                        // base_wg_z, num_batches, k_split=K (split_k=1 규약)
        1, 1, 1, 1,                               // ne02, ne12, broadcast2, broadcast3
        t as u32,                                 // padded_n (f16 B — 비양자화 경로)
    ];
    let pcb: Vec<u8> = pc.iter().flat_map(|v| v.to_le_bytes()).collect();
    // 그리드 분모 = 스펙의 BM/BN에 정합 (부록87 그리드-스펙 매칭)
    let (dx, dy) = match sp.as_str() {
        "ls" => (128u32, 128),
        "ms" => (64, 64),
        "ss" => (32, 32),
        "m" | "m32" => (64u32, 64),
        "s" => (32, 32),
        "c" | "cc" => (64, 32),
        "mini" => (32, 16),
        "def" => (64, 64),
        _ => (128, 128),
    };
    let gx = (n_out as u32).div_ceil(dx);
    let gy = (t as u32).div_ceil(dy);
    ctxg.begin_batch()?;
    ctxg.run(pl, ds, pipe, &pcb, gx, gy, 1)?;
    ctxg.end_batch_wait()?;
    // CPU 참조 대조 (처음 8값) + 타이밍
    let out: &[f32] = unsafe { std::slice::from_raw_parts(db.ptr as *const f32, n_out * t) };
    // CPU 참조: 몇 개 (m, n) 지점 대조 — y는 f16 반올림값 사용
    let yh: Vec<f32> = ybuf.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
    let rb = w.data.len() / n_out;
    let mut dq = vec![0f32; n_in];
    let mut ok_nm = 0usize; let mut ok_mn = 0usize; let mut tot = 0usize;
    for &m in &[0usize, 100, 3000, 6143] {
        llm170_core::quant::dequant_row(w.ty, &w.data[m * rb..(m + 1) * rb], 0, n_in as u64, &mut dq);
        for n in 0..t {
            let yrow = &yh[n * n_in..(n + 1) * n_in];
            let mut acc = 0f32;
            for k in 0..n_in { acc += dq[k] * yrow[k]; }
            let g_nm = out[n * n_out + m];
            let g_mn = out[m * t + n];
            let r = |g: f32| (g - acc).abs() / acc.abs().max(1e-3);
            if r(g_nm) < 0.01 { ok_nm += 1; }
            if r(g_mn) < 0.01 { ok_mn += 1; }
            tot += 1;
        }
    }
    eprintln!("레이아웃 판별: [N][M]={}/{} · [M][N]={}/{}", ok_nm, tot, ok_mn, tot);
    {
        let m = 0usize;
        llm170_core::quant::dequant_row(w.ty, &w.data[m * rb..(m + 1) * rb], 0, n_in as u64, &mut dq);
        let mut goods = vec![];
        for n in 0..t {
            let yrow = &yh[n * n_in..(n + 1) * n_in];
            let mut acc = 0f32;
            for k in 0..n_in { acc += dq[k] * yrow[k]; }
            if ((out[n * n_out + m] - acc).abs() / acc.abs().max(1e-3)) < 0.01 { goods.push(n); }
        }
        eprintln!("m=0 정답 n ({}개): {:?}", goods.len(), &goods[..goods.len().min(20)]);
    }
    let mut maxrel = 0f32;
    for &(m, n) in &[(0, 0), (1, 0), (63, 0), (64, 0), (100, 0), (127, 0), (128, 0), (0, 1), (0, 63), (0, 64), (0, 100), (0, 127), (0, 128), (100, 7), (200, 100)] {
        if m >= n_out || n >= t { continue; }
        llm170_core::quant::dequant_row(w.ty, &w.data[m * rb..(m + 1) * rb], 0, n_in as u64, &mut dq);
        let yrow = &yh[n * n_in..(n + 1) * n_in];
        let mut acc = 0f64;
        for k in 0..n_in { acc += dq[k] as f64 * yrow[k] as f64; }
        let got = out[n * n_out + m];
        eprintln!("  ck m={m} n={n}: got={got:.5} ref={:.5}", acc);
        let rel = if acc.abs() > 1e-6 { ((got - acc as f32) / acc as f32).abs() } else { got.abs() };
        maxrel = maxrel.max(rel);
    }
    // 타이밍 20회
    let nrep: u32 = std::env::var("VKMMQ_N").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let nb: u32 = std::env::var("VKMMQ_B").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let mut per_ms = 0f64;
    for b in 0..nb {
        ctxg.begin_batch()?;
        let t0 = std::time::Instant::now();
        for _ in 0..nrep { ctxg.run(pl, ds, pipe, &pcb, gx, gy, 1)?; }
        ctxg.end_batch_wait()?;
        let el = t0.elapsed().as_secs_f64() / nrep as f64;
        eprintln!("  배치 {b}: {el:.4}ms/회");
        per_ms = el; // 마지막 배치
    }
    let dt = per_ms;
    let _ = &mut pc;
    Ok(format!(
        "vk-mmq({tname}/{spv_name} spec={sp}) t={t}: {:.4}ms/회 · maxrel={maxrel:.4} · {:.1}GB/s",
        dt * 1e3, w.data.len() as f64 / dt / 1e9
    ))
}

/// vk-ft32-check (plans/89 P1.2) — fn_tile_f32(f32/BF16 밀집 프리필 타일)의
/// 실 텐서 CPU 대조. 라우터(ffn_gate_inp, f32)형상으로 게이트 발산 원인 특정.
pub fn ft32_check(path: &str) -> Result<String, String> {
    use llm170_core::matmul::FrameState as _FS;
    use llm170_core::matmul::FrameHost as _FH;
    let is_q4 = llm170_gguf::GgufFile::open(std::path::Path::new(path))
        .ok()
        .and_then(|g| g.arch().map(|a| a == "qwen4exp"))
        .unwrap_or(false);
    if !is_q4 {
    }
    let model = llm170_core::qwen4exp::Model4::load(std::path::Path::new(path))
        .map_err(|e| e.to_string())?;
    let w = model.w4("blk.0.ffn_gate_inp.weight").map_err(|e| e.to_string())?;
    let n_in = w.n_in as usize;
    let n_out = w.n_out as usize;
    let acc = VkAcc::new()?;
    let t = 64usize;
    let mut lcg = 123456789u64;
    let mut lcgf = || {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((lcg >> 33) as f32 / 4294967296.0) - 0.5
    };
    let xs: Vec<Vec<f32>> = (0..t).map(|_| (0..n_in).map(|_| lcgf()).collect()).collect();
    let mut flat = Vec::with_capacity(t * n_in);
    for r in &xs {
        flat.extend_from_slice(r);
    }
    let xh = acc.frame_alloc(t * n_in)?;
    let oh = acc.frame_alloc(t * n_out)?;
    acc.frame_write(xh, &flat)?;
    acc.frame_begin(t);
    acc.frame_mm_group(xh, std::slice::from_ref(&w), std::slice::from_ref(&oh), t)?;

    let mut got = vec![0f32; t * n_out];
    acc.frame_read(oh, &mut got)?;
    let _ = acc.frame_free(xh);
    let _ = acc.frame_free(oh);
    // CPU 참조 — w 는 f32 그대로.
    let wf = w.data.as_ptr() as *const f32;
    let mut mx = 0f64;
    let mut bad = 0usize;
    for r in 0..t {
        for j in 0..n_out {
            let mut s = 0f64;
            for k in 0..n_in {
                s += unsafe { *wf.add(j * n_in + k) } as f64 * xs[r][k] as f64;
            }
            let d = (got[r * n_out + j] as f64 - s).abs();
            if d > 1e-3 {
                bad += 1;
            }
            mx = mx.max(d);
        }
    }
    // 혼합 그룹(q8 down + f32 inject, n_out=4 극단 shape) — 실엔진 hc 믹스.
    let wd = model.w4("blk.0.hc_attn_down.weight").map_err(|e| e.to_string())?;
    let n2 = wd.n_in as usize;
    let wi = model.w4("blk.0.hc_attn_inject.weight").map_err(|e| e.to_string())?;
    let xs2: Vec<Vec<f32>> = (0..t).map(|_| (0..n2).map(|_| lcgf()).collect()).collect();
    let mut flat2 = Vec::with_capacity(t * n2);
    for r in &xs2 {
        flat2.extend_from_slice(r);
    }
    let xh2 = acc.frame_alloc(t * n2)?;
    let od = acc.frame_alloc(t * wd.n_out as usize)?;
    let oi = acc.frame_alloc(t * wi.n_out as usize)?;
    acc.frame_write(xh2, &flat2)?;
    acc.frame_begin(t);
    acc.frame_mm_group(xh2, &[wd, wi], &[od, oi], t)?;
    let mut gi = vec![0f32; t * wi.n_out as usize];
    acc.frame_read(oi, &mut gi)?;
    let _ = (acc.frame_free(xh2), acc.frame_free(od), acc.frame_free(oi));
    let wi_f = wi.data.as_ptr() as *const f32;
    let nin_i = wi.n_in as usize;
    let mut mx2 = 0f64;
    let mut bad2 = 0usize;
    for r in 0..t {
        for j in 0..wi.n_out as usize {
            let mut s = 0f64;
            for k in 0..nin_i {
                s += unsafe { *wi_f.add(j * nin_i + k) } as f64 * xs2[r][k] as f64;
            }
            let d = (gi[r * wi.n_out as usize + j] as f64 - s).abs();
            if d > 1e-3 {
                bad2 += 1;
            }
            mx2 = mx2.max(d);
        }
    }
    Ok(format!(
        "ft32-check: router max|D|={mx:.3e} bad={bad} {} | inject(f32 {}x{}) max|D|={mx2:.3e} bad={bad2} {}",
        if bad == 0 { "★" } else { "✗" },
        wi.n_out, wi.n_in,
        if bad2 == 0 { "★" } else { "✗" }
    ))
}

/// vk-moe-tile-check <mode> (plans/89 P1.1c) — q8_0/q5_K MoE 타일의 CPU 대조.
/// 모드("q8_0"|"q5_K")에 해당하는 첫 레이어의 down/gate 스택 텐서로 검증.
pub fn moe_tile_type_check(mode: &str) -> Result<String, String> {
    use llm170_core::matmul::{FrameHost as _FH, FrameState as _FS};
    let path = "/home/yoon/models/qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf";
    let model = llm170_core::qwen4exp::Model4::load(std::path::Path::new(path))
        .map_err(|e| e.to_string())?;
    let want = match mode {
        "q8_0" => llm170_gguf::GgmlType::Q8_0,
        "q5_K" => llm170_gguf::GgmlType::Q5K,
        "q4_K" => llm170_gguf::GgmlType::Q4K,
        "q5_1" => llm170_gguf::GgmlType::Q5_1,
        _ => return Ok("moe-tile-check: 모드 q8_0|q5_K|q4_K|q5_1".into()),
    };
    // 해당 타입의 첫 스택 탐색(down 우선, q5_K는 gate/up에만 존재).
    let mut found = None;
    let names = if mode == "q5_K" {
        vec!["ffn_gate_exps", "ffn_up_exps"]
    } else {
        vec!["ffn_down_exps", "ffn_gate_exps"]
    };
    for il in 0..48 {
        for nm in &names {
            if let Ok(w) = model.w4(&format!("blk.{il}.{nm}.weight")) {
                if w.ty == want {
                    found = Some((il, w));
                    break;
                }
            }
        }
        if found.is_some() {
            break;
        }
    }
    let (il, wd) = found.ok_or("해당 타입 스택 없음")?;
    let ne = 512usize;
    let n_in_d = wd.n_in as usize;
    let n_out_d = wd.n_out as usize / ne;
    let acc = VkAcc::new()?;
    let t = std::env::var("LLM170_MTC_T").ok().and_then(|v| v.parse().ok()).unwrap_or(130usize);
    let k = 10usize;
    let mut lcg = 987654321u64;
    let mut lcgf = || {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((lcg >> 33) as f32 / 4294967296.0) - 0.5
    };
    let route0: Vec<f32> = (0..ne).map(|_| lcgf() * 4.0).collect();
    let route: Vec<f32> = (0..t).flat_map(|_| route0.iter().copied()).collect();
    let xs: Vec<Vec<f32>> = (0..t * k).map(|_| (0..n_in_d).map(|_| lcgf()).collect()).collect();
    let rh = acc.frame_alloc(t * ne)?;
    let idh = acc.frame_alloc(t * k)?;
    let wth = acc.frame_alloc(t * k)?;
    let mxh = acc.frame_alloc(t * k * n_in_d)?;
    let mgh = acc.frame_alloc(t * k * n_out_d)?;
    acc.frame_write(rh, &route)?;
    let mut flat = Vec::with_capacity(t * k * n_in_d);
    for row in &xs {
        flat.extend_from_slice(row);
    }
    acc.frame_write(mxh, &flat)?;
    acc.frame_begin(t);
    acc.frame_op(&llm170_core::matmul::FrameOp::MoeTop10 { route: rh, ids: idh, wt: wth, n_exp: ne, k_sel: k })?;
    let mut ids_g = vec![0u32; t * k];
    {
        acc.frame_sync();
        let g = acc_frame_ptr(&acc, idh);
        unsafe { std::ptr::copy_nonoverlapping(g as *const u32, ids_g.as_mut_ptr(), t * k) };
    }
    acc.frame_moe_gemm(mxh, &wd, idh, mgh, ne, k)?;
    acc.frame_begin(t);
    let mut got = vec![0f32; t * k * n_out_d];
    acc.frame_read(mgh, &mut got)?;
    for h in [rh, idh, wth, mxh, mgh] {
        let _ = acc.frame_free(h);
    }
    // CPU 참조 — ids 순행, 전expert 행 디양자화 내적.
    let m = route.iter().cloned().fold(f32::MIN, f32::max);
    let ps: Vec<f32> = route.iter().map(|&v| (v - m).exp()).collect();
    let mut idx: Vec<usize> = (0..ne).collect();
    idx.sort_by(|&a, &b| ps[b].partial_cmp(&ps[a]).unwrap().then(a.cmp(&b)));
    let sel: Vec<usize> = idx[..k].to_vec();
    let mut mx = 0f64;
    let mut bad = 0usize;
    let mut ref_row = vec![0f32; n_in_d];
    let mut checked = 0usize;
    for (r, &e) in sel.iter().enumerate() {
        if ids_g[r] as usize != e {
            mx = mx.max(1.0);
        }
        for j in 0..n_out_d.min(6) {
            llm170_core::quant::dequant_row(wd.ty, wd.data, (e * n_out_d + j) as u64, n_in_d as u64, &mut ref_row);
            let dot: f32 = ref_row.iter().zip(xs[r].iter()).map(|(a, b)| a * b).sum();
            let d = (got[r * n_out_d + j] as f64 - dot as f64).abs();
            if d > 2e-2 {
                bad += 1;
            }
            mx = mx.max(d);
            checked += 1;
        }
    }
        if std::env::var_os("LLM170_MTC_DBG").is_some() {
            for rr in 0..sel.len().min(12) {
                let ee = sel[rr];
                let mut s2 = 0f64;
                let mut rr_row = vec![0f32; n_in_d];
                llm170_core::quant::dequant_row(wd.ty, wd.data, (ee * n_out_d) as u64, n_in_d as u64, &mut rr_row);
                for (a, b) in rr_row.iter().zip(xs[rr].iter()) {
                    s2 += *a as f64 * *b as f64;
                }
                eprintln!("[mtc] row={rr} e={ee} got={:.5} ref={:.5}", got[rr * n_out_d], s2);
            }
        }
        if std::env::var_os("LLM170_MTC_DBG").is_some() {
            for rr in 0..sel.len().min(10) {
                let ee = sel[rr];
                let per = wd.data.len() / 512;
                let off = ee * per;
                let d_bits = u16::from_le_bytes([wd.data[off], wd.data[off + 1]]);
                let e10 = ((d_bits >> 10) & 0x1F) as i32;
                let m10 = (d_bits & 0x3FF) as f32;
                let dv = if e10 == 0 {
                    m10 * 2f32.powi(-24)
                } else {
                    (1024.0 + m10) * 2f32.powi(e10 - 25)
                } * if d_bits & 0x8000 != 0 { -1.0 } else { 1.0 };
                eprintln!("[mtcD] row={rr} e={ee} dBits={d_bits:#06x} d={dv:.3e}");
            }
        }
    Ok(format!(
        "moe-tile-check({mode} blk.{il} down {n_out_d}x{n_in_d}, rows={}): max|D|={mx:.3e} bad={bad}/{checked} {}",
        t * k,
        if bad == 0 { "★" } else { "✗" }
    ))
}

fn hf(v: f32) -> u16 {
    // f32→f16 변환 (반올림)
    half::f16::from_f32(v).to_bits()
}


/// vk-sdot-probe — OpSDot(정수 dot) 장치 지원 검증+타이밍. plans/33.
pub fn sdot_probe() -> Result<String, String> {
    use std::time::Instant;
    let acc = VkAcc::new()?;
    let mut ctx = acc.ctx.lock();
    let buf = ctx.alloc_host(16)?;
    unsafe {
        let p = buf.ptr as *mut u32;
        *p.add(0) = 0x0182_0304;      // a (부호 혼합 i8x4)
        *p.add(1) = 0xF0FF_7F01;      // b
        *p.add(2) = 0;
        *p.add(3) = 0;
    }
    let spv = std::fs::read("crates/backend-gpu/src/rawvk/spv/sdot_probe.spv")
        .map_err(|e| e.to_string())?;
    let (dsl, pl, pool, ds, pipe) = ctx.pipeline(&spv, 1, 4)?;
    let _ = (dsl, pool);
    ctx.bind_bufs(ds, &[buf.buf]);
    let t0 = Instant::now();
    ctx.run(pl, ds, pipe, &1_000_000u32.to_le_bytes(), 1, 1, 1)?;
    let dt = t0.elapsed().as_secs_f32();
    let r = unsafe { *(buf.ptr as *const u32).add(2) };
    // CPU 기준: acc = a; 1M회 acc = sdot(acc, b) — i32 감쇠/순환값
    let mut cacc: i32 = 0x0182_0304u32 as i32;
    let b4: i32 = 0xF0FF_7F01u32 as i32;
    let bx = |v: i32, i: u32| -> i32 {
        let byte = (v >> (i * 8)) & 0xFF;
        if byte >= 128 { byte - 256 } else { byte }
    };
    for _ in 0..1_000_000 {
        let mut s = 0i32;
        for i in 0..4 { s += bx(cacc, i) * bx(b4, i); }
        cacc = s;
    }
    let expect = cacc as u32;
    Ok(format!(
        "sdot-probe: gpu={r:#010x} cpu={expect:#010x} {} · {dt:.1}ms (1M 의존 dot)",
        if r == expect { "일치" } else { "불일치" }
    ))
}

/// vk-idot-probe (plans/89 P0.1) — OpSDot(PackedVectorFormat4x8Bit) 검증+타이밍.
/// sdot_probe(plans/33)의 어셈블리 패치는 커널 문맥에서 0을 반환했다. 이번 판의
/// 차이: (a) VkCtx가 Vulkan13Features.shader_integer_dot_product를 활성화,
/// (b) spirv-as 산출물을 val 통과 구조로 직접 인코딩(.spvasm 참조).
/// mode 0=OpSDot / 1=스칼라 에뮬레이션(gemv3 dot4 동일 산술) — 동일 커널 A/B.
pub fn idot_probe() -> Result<String, String> {
    use std::time::Instant;
    let acc = VkAcc::new()?;
    if !acc.ctx.lock().idot {
        return Ok("idot-probe: 장치가 shader_integer_dot_product 미지원".into());
    }
    let mut ctx = acc.ctx.lock();
    let buf = ctx.alloc_host(32)?;
    let spv = std::fs::read("crates/backend-gpu/src/rawvk/spv/idot_probe.spv")
        .map_err(|e| e.to_string())?;
    let (dsl, pl, pool, ds, pipe) = ctx.pipeline(&spv, 1, 8)?;
    let _ = (dsl, pool);
    ctx.bind_bufs(ds, &[buf.buf]);
    // CPU 기준 — 단일 dot(비영 검증) + 1M 의존 루프 종값.
    let bx = |v: i32, i: u32| -> i32 {
        let b = (v >> (i * 8)) & 0xFF;
        if b >= 128 { b - 256 } else { b }
    };
    let (ai, bi): (i32, i32) = (0x0182_0304u32 as i32, 0xF0FF_7F01u32 as i32);
    let single: i32 = (0..4).map(|i| bx(ai, i) * bx(bi, i)).sum();
    let mut cacc = ai;
    for _ in 0..1_000_000 {
        let mut s = 0i32;
        for i in 0..4 {
            s += bx(cacc, i) * bx(bi, i);
        }
        cacc = s;
    }
    let mut lines = String::new();
    for mode in 0..2u32 {
        unsafe {
            let p = buf.ptr as *mut u32;
            *p.add(0) = 0x0182_0304;
            *p.add(1) = 0xF0FF_7F01;
            *p.add(2) = 0;
            *p.add(3) = 0;
        }
        let t0 = Instant::now();
        ctx.run(pl, ds, pipe, &push_u32s(&[mode, 1_000_000]), 1024, 1, 1)?;
        let dt = t0.elapsed().as_secs_f32() * 1000.0;
        let (r2, r3) = unsafe {
            (
                *(buf.ptr as *const u32).add(2),
                *(buf.ptr as *const u32).add(3) as i32,
            )
        };
        let ok_loop = r2 == cacc as u32;
        let ok_single = r3 == single;
        lines.push_str(&format!(
            "  mode{mode}({}): 루프 {r2:#010x} {} · 단일 dot {r3} (cpu {single}) {} · {dt:.1}ms/1M\n",
            if mode == 0 { "OpSDot" } else { "스칼라" },
            if ok_loop { "★" } else { "✗" },
            if ok_single { "★" } else { "✗" },
        ));
    }
    unsafe {
        ctx.device.destroy_pipeline(pipe, None);
        ctx.device.destroy_pipeline_layout(pl, None);
    }
    Ok(format!("idot-probe (packed i8x4 dot, plans/89 P0.1):\n{lines}"))
}


/// vk-gemv8-check — gemv8 패밀리(llama mul_mat_vec 포트, f32 직결) 검증+타이밍.
pub fn gemv8_check(path: &str, tname: &str, t: usize) -> Result<String, String> {
    use std::time::Instant;
    let model = llm170_core::qwen35::Model::load(std::path::Path::new(path))
        .map_err(|e| e.to_string())?;
    let w = model.w(tname).ok_or("텐서 없음")?;
    let is_xs = w.ty == llm170_gguf::GgmlType::Iq4Xs;
    let is_nl = w.ty == llm170_gguf::GgmlType::Iq4Nl;
    let is_q5 = w.ty == llm170_gguf::GgmlType::Q5K;
    let is_q6 = w.ty == llm170_gguf::GgmlType::Q6K;
    let is_q4 = w.ty == llm170_gguf::GgmlType::Q4K;
    let is_q3 = w.ty == llm170_gguf::GgmlType::Q3K;
    let is_q8 = w.ty == llm170_gguf::GgmlType::Q8_0;
    if !is_xs && !is_nl && !is_q5 && !is_q6 && !is_q4 && !is_q3 && !is_q8 {
        return Err("gemv8 검증은 q3_K/q4_K/q5_K/q6_K/iq4_xs/iq4_nl만".into());
    }
    let n_in = w.n_in as usize;
    let n_out = w.n_out as usize;
    let acc = VkAcc::new()?;
    let mut ctx = acc.ctx.lock();
    let mut seed = 0x1234u64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as f32 / 2147483648.0 - 0.5
    };
    let xs: Vec<Vec<f32>> = (0..t).map(|_| (0..n_in).map(|_| lcg()).collect()).collect();
    let xa = ctx.alloc_host(t * n_in * 4)?;
    for (j, x) in xs.iter().enumerate() {
        unsafe {
            // 행 스트라이드는 바이트 — n_in f32 = n_in*4바이트 (A2: 이 오타가
            // t≥2 하니스 오염의 전부였음 — 행1이 행0의 1/4 지점을 덮어씀)
            std::ptr::copy_nonoverlapping(
                x.as_ptr(), xa.ptr.add(j * n_in * 4) as *mut f32, n_in);
        }
    }
    let ob = ctx.alloc_host(t * n_out * 4)?;  // 매핑 유지 — 판독용
    // 가중 업로드 — gemv3와 동일한 균일 청크
    let ch = ctx.max_ssbo;
    let mut wbufs = Vec::new();
    let mut off = 0usize;
    let total = w.data.len();
    // 청크 크기 2의 거듭제곱 (WG 시프트 산술) — 마지막 청크는 실제 크기만 할당:
    // o = idx & mask 는 항상 청크 내 실데이터 오프셋만 생성하므로 패딩 불필요.
    let ch = total.next_power_of_two().min(1usize << (63 - ch.leading_zeros()));
    while off < total {
        let sz = ch.min(total - off);
        let mut b = ctx.alloc(sz)?;
        unsafe { std::ptr::copy_nonoverlapping(w.data.as_ptr().add(off), b.ptr, sz) };
        ctx.unmap(&mut b)?;
        wbufs.push(b.buf);
        off += sz;
    }
    let dummy = ctx.alloc_host(16)?;
    {
        let z = [0u8; 16];
        unsafe { std::ptr::copy_nonoverlapping(z.as_ptr(), dummy.ptr, 16) };
    }
    while wbufs.len() < 8 {
        wbufs.push(dummy.buf);
    }
    let chunk_words = (ch / 4) as u32;
    let spv_path = match w.ty {
        llm170_gguf::GgmlType::Q3K if std::env::var("LLM170_Q3B").map(|v| v != "0").unwrap_or(true) =>
            "crates/backend-gpu/src/rawvk/spv/gemv8_q3b.spv",
        llm170_gguf::GgmlType::Q3K => "crates/backend-gpu/src/rawvk/spv/gemv8_q3.spv",
        llm170_gguf::GgmlType::Q4K if std::env::var("LLM170_Q4B").map(|v| v != "0").unwrap_or(true) =>
            "crates/backend-gpu/src/rawvk/spv/gemv8_q4b.spv",
        llm170_gguf::GgmlType::Q4K => "crates/backend-gpu/src/rawvk/spv/gemv8_q4.spv",
        llm170_gguf::GgmlType::Q5K if std::env::var("LLM170_Q5B").map(|v| v != "0").unwrap_or(true) =>
            "crates/backend-gpu/src/rawvk/spv/gemv8_q5b.spv",
        llm170_gguf::GgmlType::Iq4Nl =>
            "crates/backend-gpu/src/rawvk/spv/gemv8_nlb.spv",
        llm170_gguf::GgmlType::Q5K => "crates/backend-gpu/src/rawvk/spv/gemv8_q5.spv",
        llm170_gguf::GgmlType::Q6K if std::env::var("LLM170_Q6B").map(|v| v != "0").unwrap_or(true) =>
            "crates/backend-gpu/src/rawvk/spv/gemv8_q6b.spv",
        llm170_gguf::GgmlType::Q6K => "crates/backend-gpu/src/rawvk/spv/gemv8_q6.spv",
        llm170_gguf::GgmlType::Iq4Xs if std::env::var("LLM170_XSB").map(|v| v != "0").unwrap_or(true) =>
            "crates/backend-gpu/src/rawvk/spv/gemv8_xsb.spv",
        llm170_gguf::GgmlType::Iq4Xs => "crates/backend-gpu/src/rawvk/spv/gemv8_xs.spv",
        llm170_gguf::GgmlType::Q8_0 if std::env::var("LLM170_Q8B").map(|v| v != "0").unwrap_or(true) =>
            "crates/backend-gpu/src/rawvk/spv/gemv8_q8b.spv",
        llm170_gguf::GgmlType::Q8_0 => "crates/backend-gpu/src/rawvk/spv/gemv8_q8.spv",
        _ => return Err("gemv8: 미지원 타입".into()),
    };
    let spv = std::fs::read(spv_path).map_err(|e| e.to_string())?;
    let (kb, _gb, _db) = acc.ensure_shared(&mut ctx)?;
    let n_kb_h = if is_xs { 12 } else { 10 };
    let (dsl, pl, pool, ds, pipe) = ctx.pipeline(&spv, n_kb_h, 24)?;
    let _ = (dsl, pool);
    let mut binds: Vec<vk::Buffer> = wbufs.clone();
    binds.push(xa.buf);
    binds.push(ob.buf);
    if is_xs {
        binds.push(kb);
    }
    ctx.bind_bufs(ds, &binds);
    let q5b = w.ty == llm170_gguf::GgmlType::Q5K && std::env::var("LLM170_Q5B").map(|v| v != "0").unwrap_or(true);
    let q4b = w.ty == llm170_gguf::GgmlType::Q4K && std::env::var("LLM170_Q4B").map(|v| v != "0").unwrap_or(true);
    let q6b = w.ty == llm170_gguf::GgmlType::Q6K && std::env::var("LLM170_Q6B").map(|v| v != "0").unwrap_or(true);
    let q8b = w.ty == llm170_gguf::GgmlType::Q8_0 && std::env::var("LLM170_Q8B").map(|v| v != "0").unwrap_or(true);
    let q3b = w.ty == llm170_gguf::GgmlType::Q3K && std::env::var("LLM170_Q3B").map(|v| v != "0").unwrap_or(true);
    let xsb = w.ty == llm170_gguf::GgmlType::Iq4Xs && std::env::var("LLM170_XSB").map(|v| v != "0").unwrap_or(true);
    let rpf: u32 = if q5b || q4b || q6b || q8b || xsb || q3b { 2 } else if n_out < 4096 { 1 } else { 2 };   // llama NUM_ROWS=2
    let cw_log2 = 31u32 - chunk_words.leading_zeros();
    let cw_mask = (1u32 << cw_log2) - 1u32;
    // cw 단위: q5/q6(u16 typed 뷰)만 u16 단위, 나머지 u32
    let (cwpl, cwpm) = if is_q5 || is_q6 {
        (31u32 - (chunk_words * 2).leading_zeros(), (chunk_words * 2) - 1)
    } else { (cw_log2, cw_mask) };
    let push = push_u32s(&[n_in as u32, n_out as u32, t as u32, cwpl, cwpm, rpf]);
    ctx.run(pl, ds, pipe, &push, 1, n_out.div_ceil(rpf as usize) as u32, t as u32)?;
    let outs: Vec<f32> = unsafe {
        let mut v = vec![0f32; t * n_out];
        std::ptr::copy_nonoverlapping(ob.ptr as *const f32, v.as_mut_ptr(), t * n_out);
        v
    };
    // CPU 기준: 디양자화 내적
    let mut mx = 0f64;
    let mut ref_row = vec![0.0f32; n_in];
    for (j, x) in xs.iter().enumerate() {
        for r in 0..n_out.min(64) {
            llm170_core::quant::dequant_row(
                w.ty, w.data, r as u64, n_in as u64, &mut ref_row);
            let dot: f32 = ref_row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            mx = mx.max((dot - outs[j * n_out + r]).abs() as f64);
        }
    }
    let solo_t0 = Instant::now();
    for _ in 0..10 {
        ctx.run(pl, ds, pipe, &push, 1, n_out.div_ceil(rpf as usize) as u32, t as u32)?;
    }
    let solo_dt = solo_t0.elapsed().as_secs_f64() / 10.0;
    // L2 플러시 타이밍 — 반복 사이 자기 자신을 12회 연속 돌린 뒤
    // '매 반복 직전 타 텐서 1회' 교차 판독으로 캐시 몰아내기 (L2FLUSH=1).
    let flushed_dt: f64 = if std::env::var_os("LLM170_L2FLUSH").is_some() {
        // MULTI로 등록한 첫 extra 텐서를 플러시용으로 재사용: 그 weights로
        // 동일 커널 1회 (다른 ds/push 필요) — 여기선 간단히 xa를 8MB 재기록 후
        // 측정 대상 run 직전 xa 전체 재업로드 (호스트 memcpy가 L2 오염)
        let t4 = Instant::now();
        let xa2 = xs[0].clone();
        for _ in 0..10 {
            unsafe {
                std::ptr::copy_nonoverlapping(xa2.as_ptr(), xa.ptr as *mut f32, n_in);
            }
            ctx.run(pl, ds, pipe, &push, 1, n_out.div_ceil(rpf as usize) as u32, t as u32)?;
        }
        t4.elapsed().as_secs_f64() / 10.0
    } else { 0.0 };
    if flushed_dt > 0.0 {
        return Ok(format!(
            "gemv8-l2flush({tname}): {:.3}ms → {:.1}GB/s (웜 {})",
            flushed_dt * 1e3, w.data.len() as f64 / flushed_dt / 1e9,
            w.data.len() as f64 / solo_dt / 1e9
        ));
    }
    if let Ok(list) = std::env::var("LLM170_MULTI") {
        // TLB/할당수 가설: 추가 텐서들을 같은 컨텍스트에 로드(상주)시킨 뒤
        // 이 텐서의 타이밍 재측정 — 속도 붕괴 시 가설 확인.
        for extra in list.split(',').filter(|x| !x.is_empty()) {
            if extra == tname { continue; }
            let w2 = match model.w(extra) { Some(w) => w, None => continue };
            let mut off2 = 0usize;
            let tot2 = w2.data.len();
            while off2 < tot2 {
                let sz2 = ch.min(tot2 - off2);
                let mut b2 = ctx.alloc(sz2)?;
                unsafe { std::ptr::copy_nonoverlapping(w2.data.as_ptr().add(off2), b2.ptr, sz2) };
                ctx.unmap(&mut b2)?;
                off2 += sz2;
            }
        }
        let t2 = Instant::now();
        for _ in 0..10 {
            ctx.run(pl, ds, pipe, &push, 1, n_out.div_ceil(rpf as usize) as u32, t as u32)?;
        }
        let dt2 = t2.elapsed().as_secs_f64() / 10.0;
        let _ = &solo_dt;
        return Ok(format!(
            "gemv8-multi({tname}): {:.3}ms → {:.1}GB/s (단독 {:.1})",
            dt2 * 1e3, w.data.len() as f64 / dt2 / 1e9, w.data.len() as f64 / solo_dt / 1e9
        ));
    }
    Ok(format!(
        "gemv8({tname}) t={t}: {:.3}ms → {:.1}GB/s · max|D|={mx:.4}",
        solo_dt * 1e3,
        w.data.len() as f64 / solo_dt / 1e9
    ))
}

/// vk-tile-check — 타일(coopmat f16) 커널 vs CPU 디양자화 GEMM 검증 (plans/38 A2).
/// f16 스테이징 품질계약: maxrel 허용치 ~2e-2 (근접 아닌 구조 오류 검출 목적).
#[allow(clippy::if_same_then_else)] // 진단 A/B: 커널(spv)은 분기마다 다르고 gx 산식만 우연히 동일
pub fn tile_check(path: &str, tname: &str, t: usize) -> Result<String, String> {
    use std::time::Instant;
    // plans/84 B: arch 판별 후 단일 로드(vk-gemv-check와 동일 패턴) — FN 멀티파트 지원.
    enum AnyModel {
        Q35(llm170_core::qwen35::Model),
        Q4(llm170_core::qwen4exp::Model4),
    }
    let is_q4 = llm170_gguf::GgufFile::open(std::path::Path::new(path))
        .ok()
        .and_then(|g| g.arch().map(|a| a == "qwen4exp"))
        .unwrap_or(false);
    let model = if is_q4 {
        AnyModel::Q4(
            llm170_core::qwen4exp::Model4::load(std::path::Path::new(path))
                .map_err(|e| e.to_string())?,
        )
    } else {
        AnyModel::Q35(
            llm170_core::qwen35::Model::load(std::path::Path::new(path))
                .map_err(|e| e.to_string())?,
        )
    };
    let w = match &model {
        AnyModel::Q35(m) => m.w(tname).ok_or("텐서 없음")?,
        AnyModel::Q4(m) => m.w4(tname).map_err(|e| e.to_string())?,
    };
    let n_in = w.n_in as usize;
    let n_out = w.n_out as usize;
    let ms4gy = std::env::var("LLM170_TILE_MS4GY").map(|v| v=="1").unwrap_or(false) && w.ty == llm170_gguf::GgmlType::Q5K;
    let msall = std::env::var("LLM170_TILE_MSALL").map(|v| v=="1").unwrap_or(false);
    let gy2 = std::env::var("LLM170_TILE_GY2").map(|v| v=="1").unwrap_or(false) && msall && w.ty != llm170_gguf::GgmlType::Q5K;
    let bn128 = std::env::var("LLM170_TILE_BN128").map(|v| v=="1").unwrap_or(false) && msall && w.ty != llm170_gguf::GgmlType::Q5K;
    if t < 1 || (t > 128 && !ms4gy && !gy2 && !msall) {
        return Err("tile 검증 t는 1..=128 (MS4GY/GY2/MSALL는 512까지)".into());
    }
    let (spv_name, n_kb, extra) = match w.ty {
        llm170_gguf::GgmlType::Q5K if std::env::var("LLM170_TILE_MS128V2").map(|v| v=="1").unwrap_or(false) => ("tile_ms128v2.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q5K if std::env::var("LLM170_TILE_MS128").map(|v| v=="3").unwrap_or(false) => ("tile_ms128s18.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q5K if std::env::var("LLM170_TILE_MS128").map(|v| v=="1").unwrap_or(false) => ("tile_ms128.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q5K if msall => ("tile_ms4.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q4K if msall && gy2 => ("tile_q4kmgy.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q6K if msall && gy2 => ("tile_q6kmgy.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q3K if msall && gy2 => ("tile_q3kmgy.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q8_0 if msall && gy2 => ("tile_q8mgy.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Iq4Xs if msall && gy2 => ("tile_xsmgy.spv", 11u32, 1u8),
        llm170_gguf::GgmlType::Iq4Nl if msall && gy2 => ("tile_nlmgy.spv", 11u32, 1u8),
        llm170_gguf::GgmlType::Q4K if bn128 => ("tile_q4k128.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q6K if bn128 => ("tile_q6k128.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q3K if bn128 => ("tile_q3k128.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q8_0 if bn128 => ("tile_q8128.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Iq4Xs if bn128 => ("tile_xs128.spv", 11u32, 1u8),
        llm170_gguf::GgmlType::Iq4Nl if bn128 => ("tile_nl128.spv", 11u32, 1u8),
        llm170_gguf::GgmlType::Q4K if msall => ("tile_q4kms.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q6K if msall => ("tile_q6kms.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q3K if msall => ("tile_q3kms.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q8_0 if msall => ("tile_q8ms.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Iq4Xs if msall => ("tile_xsms.spv", 11u32, 1u8),
        llm170_gguf::GgmlType::Iq4Nl if msall => ("tile_nlms.spv", 11u32, 1u8),
        llm170_gguf::GgmlType::Q5K if std::env::var("LLM170_TILE_LLM").map(|v| v=="1").unwrap_or(false) => ("tile_llm.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q5K if std::env::var("LLM170_TILE_S32").map(|v| v=="1").unwrap_or(false) => ("tile_s32.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q5K if std::env::var("LLM170_TILE_S32B").map(|v| v=="1").unwrap_or(false) => ("tile_s32b.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q5K if std::env::var("LLM170_TILE_MS").map(|v| v=="1").unwrap_or(false) => ("tile_ms.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q5K if std::env::var("LLM170_TILE_MSF16B").map(|v| v=="1").unwrap_or(false) => ("tile_ms_f16b.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q5K if std::env::var("LLM170_TILE_MS2").map(|v| v=="1").unwrap_or(false) => ("tile_ms2.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q5K if std::env::var("LLM170_TILE_MS3").map(|v| v=="1").unwrap_or(false) => ("tile_ms3.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q5K if std::env::var("LLM170_TILE_MS4GY").map(|v| v=="1").unwrap_or(false) => ("tile_ms4gy.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q5K if std::env::var("LLM170_TILE_MS8").map(|v| v=="1").unwrap_or(false) => ("tile_ms8.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q5K if std::env::var("LLM170_TILE_MS7").map(|v| v=="1").unwrap_or(false) => ("tile_ms7.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q5K if std::env::var("LLM170_TILE_MS6").map(|v| v=="1").unwrap_or(false) => ("tile_ms6.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q5K if std::env::var("LLM170_TILE_MS4").map(|v| v=="1").unwrap_or(false) => ("tile_ms4.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q5K if std::env::var("LLM170_TILE_MS5").map(|v| v=="1").unwrap_or(false) => ("tile_ms5.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q5K if std::env::var("LLM170_TILE_NOOPT").map(|v| v=="1").unwrap_or(false) => ("tile128_noopt.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q5K if std::env::var("LLM170_TILE_W").map(|v| v=="1").unwrap_or(false) => ("tile128w.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q5K if std::env::var("LLM170_TILE_OCC").map(|v| v=="1").unwrap_or(false) => ("tile128o.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q5K if std::env::var("LLM170_TILE_DS").map(|v| v=="1").unwrap_or(false) => ("tile128_ds.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q5K if std::env::var_os("LLM170_TILE_V2").is_some() => ("tile128v2.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q5K => ("tile128_q5k.spv", 10u32, 0u8),
        llm170_gguf::GgmlType::Q5_1 => ("tile128_q51.spv", 10u32, 0u8), // plans/84 B: 전용 런치(t_below)
        llm170_gguf::GgmlType::Q4K => ("tile_q4k.spv", 10, 0),
        llm170_gguf::GgmlType::Q6K => ("tile_q6k.spv", 10, 0),
        llm170_gguf::GgmlType::Q3K => ("tile_q3k.spv", 10, 0),
        llm170_gguf::GgmlType::Q8_0 => ("tile_q8.spv", 10, 0),
        llm170_gguf::GgmlType::Iq4Xs => ("tile_xs.spv", 11, 1),   // ktab
        llm170_gguf::GgmlType::Iq4Nl => ("tile_nl.spv", 11, 1),    // ktab
        llm170_gguf::GgmlType::Iq3S => ("tile_iq3s.spv", 11, 2),   // grid3s
        _ => return Err("tile 검증 불가 타입".into()),
    };
    let is_128 = (w.ty == llm170_gguf::GgmlType::Q5K && std::env::var_os("LLM170_TILE_V2").is_none()) || msall
        || (std::env::var("LLM170_TILE_MS128V2").map(|v| v=="1").unwrap_or(false) && w.ty == llm170_gguf::GgmlType::Q5K)
        || std::env::var("LLM170_TILE_MS128").map(|v| v=="1").unwrap_or(false);
        let acc = VkAcc::new()?;
    let mut ctx = acc.ctx.lock();
    let mut seed = 0x5deece66u64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as f32 / 2147483648.0 - 0.5
    };
    let xs: Vec<Vec<f32>> = (0..t).map(|_| (0..n_in).map(|_| lcg()).collect()).collect();
    // xq 양자화 (GPU quant — 비트 검증 완료 경로)
    let xq_w = n_in / 4 + n_in / 32 + n_in / 16;
    let msf16b = std::env::var("LLM170_TILE_MSF16B").map(|v| v=="1").unwrap_or(false);
    let mut xqb = if std::env::var_os("LLM170_TILE_BDEV").is_some() {
        ctx.alloc((t * xq_w * 4).max(t * n_in * 2))?
    } else {
        ctx.alloc_host((t * xq_w * 4).max(t * n_in * 2))?
    };
    if msf16b {
        let mut hb: Vec<u16> = Vec::with_capacity(t * n_in);
        for row in &xs { for &v in row { hb.push(hf(v)); } }
        unsafe { std::ptr::copy_nonoverlapping(hb.as_ptr() as *const u8, xqb.ptr, t * n_in * 2) };
        ctx.unmap(&mut xqb)?;
    } else {
        acc.quant_upload(&mut ctx, &xs, n_in, xqb.buf)?;
    }
    let ob = ctx.alloc_host(t * n_out * 4)?;
    // 가중 업로드
    let total = w.data.len();
    let ch = total.next_power_of_two().min(1usize << (63 - ctx.max_ssbo.leading_zeros()));
    let mut wbufs = Vec::new();
    let mut off = 0usize;
    while off < total {
        let sz = ch.min(total - off);
        let mut b = ctx.alloc(sz)?;
        unsafe { std::ptr::copy_nonoverlapping(w.data.as_ptr().add(off), b.ptr, sz) };
        ctx.unmap(&mut b)?;
        wbufs.push(b.buf);
        off += sz;
    }
    let (ktab, grid, dummy) = acc.ensure_shared(&mut ctx)?;
    while wbufs.len() < 8 {
        wbufs.push(dummy);
    }
    let spv = std::fs::read(format!("crates/backend-gpu/src/rawvk/spv/{spv_name}"))
        .map_err(|e| e.to_string())?;
    // plans/41: ms 패밀리는 push 5필드 [n_in,n_out,xq_w,t,tok_base] (pb=20)
    let bn128spv = spv_name.ends_with("128.spv") || spv_name.starts_with("tile_ms128s");
    let is_msfam = spv_name.ends_with("ms.spv") || spv_name.ends_with("mgy.spv")
        || spv_name == "tile_ms4.spv" || bn128spv;
    let slab: usize = if bn128spv { 128 } else { 64 };
    let ms128fam_any = std::env::var("LLM170_TILE_MS128V2").map(|v| v=="1").unwrap_or(false)
        || std::env::var("LLM170_TILE_MS128").map(|v| v=="1").unwrap_or(false);
    let q51fam = w.ty == llm170_gguf::GgmlType::Q5_1;   // plans/84 B: 6필드 push(24B)
    let pb: u32 = if q51fam { 24 } else if bn128spv || ms128fam_any { 24 } else if is_msfam { 20 } else if is_128 { 16 } else { 24 };
    let mpush = |tt: u32, base: u32| push_u32s(&[n_in as u32, n_out as u32, xq_w as u32, tt, base]);
    let (dsl, pl, pool, ds, pipe) = ctx.pipeline(&spv, n_kb, pb)?;
    let _ = (dsl, pool);
    let mut binds: Vec<vk::Buffer> = wbufs.clone();
    binds.push(xqb.buf);
    binds.push(ob.buf);
    if extra == 1 {
        binds.push(ktab);
    } else if extra == 2 {
        binds.push(grid);
    }
    ctx.bind_bufs(ds, &binds);
    let cw = (ch / 4) as u32;
    let cw = cw.next_power_of_two();
    let cw_log2 = 31u32 - cw.leading_zeros();
    let cw_mask = cw - 1;
    let gx = if std::env::var("LLM170_TILE_MS4GY").map(|v| v=="1").unwrap_or(false) && w.ty == llm170_gguf::GgmlType::Q5K {
        (n_out as u32).div_ceil(64)
    } else if std::env::var("LLM170_TILE_MS8").map(|v| v=="1").unwrap_or(false) && w.ty == llm170_gguf::GgmlType::Q5K {
        (n_out as u32).div_ceil(64)
    } else if (std::env::var("LLM170_TILE_MS7").map(|v| v=="1").unwrap_or(false) || std::env::var("LLM170_TILE_MS6").map(|v| v=="1").unwrap_or(false)) && w.ty == llm170_gguf::GgmlType::Q5K {
        (n_out as u32).div_ceil(64)   // tile_ms6: WG당 64행
    } else if spv_name.starts_with("tile_ms128") && w.ty == llm170_gguf::GgmlType::Q5K {
        (n_out as u32).div_ceil(64)   // tile_ms128 계열: WG당 64행 × 128토큰
    } else if msall || ((std::env::var("LLM170_TILE_MS").map(|v| v=="1").unwrap_or(false) || msf16b || std::env::var("LLM170_TILE_MS2").map(|v| v=="1").unwrap_or(false) || std::env::var("LLM170_TILE_MS3").map(|v| v=="1").unwrap_or(false) || std::env::var("LLM170_TILE_MS4").map(|v| v=="1").unwrap_or(false) || std::env::var("LLM170_TILE_MS5").map(|v| v=="1").unwrap_or(false)) && w.ty == llm170_gguf::GgmlType::Q5K) {
        (n_out as u32).div_ceil(64)   // tile_ms: WG당 64행
    } else if std::env::var("LLM170_TILE_S32B").map(|v| v=="1").unwrap_or(false) && w.ty == llm170_gguf::GgmlType::Q5K {
        (n_out as u32).div_ceil(32)   // tile_s32b: WG당 32행
    } else if std::env::var("LLM170_TILE_S32").map(|v| v=="1").unwrap_or(false) && w.ty == llm170_gguf::GgmlType::Q5K {
        (n_out as u32).div_ceil(32)   // tile_s32: WG당 32행
    } else if std::env::var_os("LLM170_TILE_V2").is_some() && w.ty == llm170_gguf::GgmlType::Q5K {
        (n_out as u32).div_ceil(64)   // tile128v2: WG당 64행
    } else {
        (n_out as u32).div_ceil(128)
    };
    let t0 = Instant::now();
    if ms4gy {
        // gy 병렬: 단일 디스패치, gy=t/64, push t=64 (커널은 슬래브당 64토큰)
        let gy = (t as u32).div_ceil(64);
        let push = mpush(64, 0);
        ctx.run(pl, ds, pipe, &push, gy, gx, 1)?;   // plans/41 zs: 슬래브 x, 행 y
        let outs: Vec<f32> = unsafe {
            let mut v = vec![0f32; t * n_out];
            std::ptr::copy_nonoverlapping(ob.ptr as *const f32, v.as_mut_ptr(), t * n_out);
            v
        };
        let _ = &outs;
        // (검증·벤치 공용 경로로 흐르게 outs 사용은 아래와 동일 — 여기선 run만 대체)
        // 아래 기존 로직이 outs를 다시 읽으므로 여기서 반환하지 않고 흐름 유지:
        // → 실제로는 아래 outs 재판독이 이 run 결과를 본다.
        let _ = t;
    }
    if q51fam {
        // plans/84 B: q5_1 타일판 — 128토큰 슬래브 + tok_base + 청크 wsh(엔진과 동일 규약).
        for tb in (0..t).step_by(128) {
            let nt = (t - tb).min(128) as u32;
            let push = push_u32s(&[n_in as u32, n_out as u32, xq_w as u32, nt, tb as u32, cw_log2]);
            ctx.run(pl, ds, pipe, &push, gx, 1, 1)?;
        }
    } else if gy2 {
        let gy = (t as u32).div_ceil(64);
        let push = mpush(64, 0);
        ctx.run(pl, ds, pipe, &push, gy, gx, 1)?;
    } else if is_128 && !ms4gy {
        let ms128fam = std::env::var("LLM170_TILE_MS128V2").map(|v| v=="1").unwrap_or(false)
            || std::env::var("LLM170_TILE_MS128").map(|v| v=="1").unwrap_or(false);
        if is_msfam {
            // ms 패밀리: t>슬래브는 분할 (tok_base로 전 토큰 커버; ms256 = 128토큰 슬래브)
            for tb in (0..t).step_by(slab) {
                let nt = (t - tb).min(slab) as u32;
                let push = if bn128spv {
                    push_u32s(&[n_in as u32, n_out as u32, xq_w as u32, nt, 0u32, tb as u32])
                } else {
                    mpush(nt, tb as u32)
                };
                ctx.run(pl, ds, pipe, &push, gx, 1, 1)?;
            }
        } else {
            let push = if ms128fam {
                // ms128: [n_in,n_out,xq_w,t,row_off,tok_base] (pb=24)
                push_u32s(&[n_in as u32, n_out as u32, xq_w as u32, t as u32, 0u32, 0u32])
            } else {
                push_u32s(&[n_in as u32, n_out as u32, xq_w as u32, t as u32])
            };
            ctx.run(pl, ds, pipe, &push, gx, 1, 1)?;
        }
    } else {
        let push = push_u32s(&[n_in as u32, n_out as u32, xq_w as u32, t as u32, cw_log2, cw_mask]);
        ctx.run(pl, ds, pipe, &push, gx, 1, 1)?;
    }
    ctx.flush2()?;   // plans/46: 판독 전 GPU 완료 — 종전 dt는 비동기 제출만 잼(실측 허수)
    let outs: Vec<f32> = unsafe {
        let mut v = vec![0f32; t * n_out];
        std::ptr::copy_nonoverlapping(ob.ptr as *const f32, v.as_mut_ptr(), t * n_out);
        v
    };
    let dt = t0.elapsed().as_secs_f64();
    // 배치 타이밍 (신뢰): N회 녹화 → 1회 제출·대기 — 단독 submit 계측 결함 회피
    if std::env::var_os("LLM170_TILE_BENCH").is_some() {
        let n = std::env::var("LLM170_TILE_BENCH").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(100);
        let t1 = std::time::Instant::now();
        ctx.begin_batch()?;
        for _ in 0..n {
            if gy2 {
                // gy 단일 디스패치 (엔진 gy와 동일): 슬래브 x, 행 y
                let gy = (t as u32).div_ceil(64);
                let push = mpush(64, 0);
                ctx.run(pl, ds, pipe, &push, gy, gx, 1)?;
            } else if is_msfam && t > slab {
                // 순차 슬래브 (엔진 비-gy 경로와 동일 형태): tok_base=tb로 전 토큰 커버
                for tb in (0..t).step_by(slab) {
                    let nt = (t - tb).min(slab) as u32;
                    let push = mpush(nt, tb as u32);
                    ctx.run(pl, ds, pipe, &push, gx, 1, 1)?;
                }
            } else if is_msfam {
                let push = mpush(t as u32, 0);
                ctx.run(pl, ds, pipe, &push, gx, 1, 1)?;
            } else if is_128 {
                let push = push_u32s(&[n_in as u32, n_out as u32, xq_w as u32, t as u32]);
                ctx.run(pl, ds, pipe, &push, gx, 1, 1)?;
            } else {
                let push = push_u32s(&[n_in as u32, n_out as u32, xq_w as u32, t as u32, cw_log2, cw_mask]);
                ctx.run(pl, ds, pipe, &push, gx, 1, 1)?;
            }
        }
        ctx.end_batch_wait()?;
        let per = t1.elapsed().as_secs_f64() / n as f64;
        return Ok(format!(
            "tile-bench({tname}/{spv_name}) t={t}: {:.4}ms/회 × {n} → {:.1}GB/s",
            per * 1e3, w.data.len() as f64 / per / 1e9
        ));
    }
    // CPU 기준: 디양자화 · f64 내적 — 행 0..64 + WG 경계/꼬리 샘플 (plans/40:
    // 행 64+ 미검증이 ms 패밀리 매핑 버그 은폐 — 전 WG 경계 커버)
    let mut rows: Vec<usize> = (0..n_out.min(64)).collect();
    for r in [63usize, 64, 65, 127, 128, 129, 191, 192, n_out.saturating_sub(2), n_out - 1] {
        if r < n_out && !rows.contains(&r) {
            rows.push(r);
        }
    }
    let mut ref_row = vec![0.0f32; n_in];
    let mut maxrel = 0f64;
    let mut worst = (0usize, 0usize, 0f64, 0f64);
    let mut bad_rows = 0usize;
    let mut bucket_bad = std::collections::BTreeMap::<u64, usize>::new();
    for (j, x) in xs.iter().enumerate() {
        let mut row_bad = false;
        for &r in &rows {
            llm170_core::quant::dequant_row(w.ty, w.data, r as u64, n_in as u64, &mut ref_row);
            let dot: f64 = ref_row.iter().zip(x.iter()).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
            let g = outs[j * n_out + r] as f64;
            let rel = (g - dot).abs() / dot.abs().max(1.0);
            if rel > maxrel {
                maxrel = rel;
                worst = (j, r, dot, g);
            }
            if rel > 2e-2 {
                row_bad = true;
                *bucket_bad.entry((r / 64) as u64).or_default() += 1;
            }
        }
        if row_bad {
            bad_rows += 1;
        }
    }
    eprintln!("[bucket] 2%초과 행(64행 버킷): {:?}", bucket_bad);
    if std::env::var_os("LLM170_TILE_DUMP").is_some() {
        eprintln!("[dump] outs[0][0..4] = {:?}", &outs[0..4]);
        if std::env::var_os("LLM170_TILE_DUMP").is_some() && t >= 1 {
            let mut zr = None;
            for (i, v) in outs[0..n_out].iter().enumerate() { if v.abs() < 1e-30 { zr = Some(i); break; } }
            let mut last_nz = 0;
            for (i, v) in outs[0..n_out].iter().enumerate() { if v.abs() > 1e-30 { last_nz = i; } }
            eprintln!("[dump] tok0 첫0행={:?} 마지막비0행={last_nz}", zr);
        }
        if n_out >= 6144 {
            eprintln!("[dump] tok0 rows 6078..6082 = {:?}", &outs[6078..6082]);
            eprintln!("[dump] tok0 rows 6126..6130 = {:?}", &outs[6126..6130]);
        }
        if n_out > 6144 {
            eprintln!("[dump] tok0 rows 6140..6144 = {:?}", &outs[6140..6144]);
        } else {
            eprintln!("[dump] tok0 rows {}..{} = {:?}", n_out-4, n_out, &outs[n_out-4..n_out]);
        }
        eprintln!("[dump] xs[0][0..6] = {:?}", &xs[0][0..6]);
    }
    Ok(format!(
        "tile({tname}/{spv_name}) t={t}: {dt:.3}ms · maxrel={maxrel:.4} (worst j={} r={} ref={:.4} gpu={:.4}) · 2%초과 토큰 {bad_rows}/{t}",
        worst.0, worst.1, worst.2, worst.3
    ))
}



/// dbg-q3 (plans/40) — tile_q3kms 디코드를 행 0 전원소 덤프해 CPU 진실과 대조.
pub fn q3_dbg(path: &str, tname: &str) -> Result<String, String> {
    let model = llm170_core::qwen35::Model::load(std::path::Path::new(path))
        .map_err(|e| e.to_string())?;
    let w = model.w(tname).ok_or("텐서 없음")?;
    let n_in = w.n_in as usize;
    let acc = VkAcc::new()?;
    let mut ctx = acc.ctx.lock();
    let total = w.data.len();
    let ch = total.next_power_of_two().min(1usize << (63 - ctx.max_ssbo.leading_zeros()));
    let mut wbufs = Vec::new();
    let mut off = 0usize;
    while off < total {
        let sz = ch.min(total - off);
        let mut b = ctx.alloc(sz)?;
        unsafe { std::ptr::copy_nonoverlapping(w.data.as_ptr().add(off), b.ptr, sz) };
        ctx.unmap(&mut b)?;
        wbufs.push(b.buf);
        off += sz;
    }
    let ob = ctx.alloc_host(n_in * 4 + 4096)?;
    let spv = std::fs::read("crates/backend-gpu/src/rawvk/spv/dbg_q3.spv").map_err(|e| e.to_string())?;
    let (_dsl, pl, _dp, ds, pipe) = ctx.pipeline(&spv, 2, 4)?;
    ctx.bind_bufs(ds, &[wbufs[0], ob.buf]);
    let _gx = (n_in as u32) / 64 / 32 * 64;  // sb 수/64
    let n_sb = (n_in / 32) as u32;
    let gx = n_sb.div_ceil(64);
    let push = push_u32s(&[n_in as u32]);
    ctx.run(pl, ds, pipe, &push, gx, 1, 1)?;
    let outs: Vec<f32> = unsafe {
        let mut v = vec![0f32; n_in + 1024];
        std::ptr::copy_nonoverlapping(ob.ptr as *const f32, v.as_mut_ptr(), n_in + 1024);
        v
    };
    // CPU 진실
    let mut ref_row = vec![0f32; n_in];
    llm170_core::quant::dequant_row(w.ty, w.data, 0, n_in as u64, &mut ref_row);
    let mut bad = 0usize;
    let mut first = vec![];
    for k in 0..n_in {
        let rel = (outs[k] - ref_row[k]).abs() / ref_row[k].abs().max(1e-3);
        if rel > 1e-3 {
            bad += 1;
            if first.len() < 10 { first.push(format!("k={k} gpu={:.5} ref={:.5}", outs[k], ref_row[k])); }
        }
    }
    if std::env::var_os("LLM170_Q3_WIDE").is_some() {
        let isv: Vec<u32> = outs[n_in..n_in+1024].iter().map(|f| *f as u32).collect();
        let _ = &isv;
        eprintln!("[is] k352/368 (sb11 hf0/hf1의 is_i×0.001): {:.4} {:.4}", outs[352]*1000.0, outs[368]*1000.0);
        eprintln!("[is] k0/16 (sb0): {:.4} {:.4}", outs[0]*1000.0, outs[16]*1000.0);
        eprintln!("[is] sb8..15: {:?}", &isv[16..32]);
        eprintln!("[wide] k48..63 gpu: {:?}", &outs[48..64]);
eprintln!("[wide] k352..383 gpu: {:?}", &outs[352..384]);
        // 불일치 k의 (sb&3, kc>>4) 히스토그램
        let mut hh = std::collections::BTreeMap::<(usize, usize), usize>::new();
        for k in 0..n_in {
            let rel = (outs[k] - ref_row[k]).abs() / ref_row[k].abs().max(1e-3);
            if rel > 1e-3 {
                *hh.entry(((k >> 5) & 3, (k & 16) >> 4)).or_default() += 1;
            }
        }
        eprintln!("[hist] (j, hf) → 불일치 수: {:?}", hh);
    }
    Ok(format!("dbg-q3: 불일치 {bad}/{n_in} | {}", first.join(" · ")))
}



/// dbg-q3b (plans/40) — gemv8_q3b 디코드 원소 덤프 ↔ CPU 진실.
pub fn q3b_dbg(path: &str, tname: &str) -> Result<String, String> {
    let model = llm170_core::qwen35::Model::load(std::path::Path::new(path))
        .map_err(|e| e.to_string())?;
    let w = model.w(tname).ok_or("텐서 없음")?;
    let n_in = w.n_in as usize;
    let acc = VkAcc::new()?;
    let mut ctx = acc.ctx.lock();
    let mut b = ctx.alloc(w.data.len())?;
    unsafe { std::ptr::copy_nonoverlapping(w.data.as_ptr(), b.ptr, w.data.len()) };
    ctx.unmap(&mut b)?;
    let ob = ctx.alloc_host(n_in * 4)?;
    let spv = std::fs::read("crates/backend-gpu/src/rawvk/spv/dbg_q3b.spv").map_err(|e| e.to_string())?;
    let (_dsl, pl, _dp, ds, pipe) = ctx.pipeline(&spv, 2, 4)?;
    ctx.bind_bufs(ds, &[b.buf, ob.buf]);
    let gx = (n_in as u32).div_ceil(512);
    let push = push_u32s(&[n_in as u32]);
    ctx.run(pl, ds, pipe, &push, gx, 1, 1)?;
    let outs: Vec<f32> = unsafe {
        let mut v = vec![0f32; n_in];
        std::ptr::copy_nonoverlapping(ob.ptr as *const f32, v.as_mut_ptr(), n_in);
        v
    };
    let mut ref_row = vec![0f32; n_in];
    llm170_core::quant::dequant_row(w.ty, w.data, 0, n_in as u64, &mut ref_row);
    let mut bad = 0usize;
    let mut first = vec![];
    for k in 0..n_in {
        let rel = (outs[k] - ref_row[k]).abs() / ref_row[k].abs().max(1e-3);
        if rel > 1e-3 {
            bad += 1;
            if first.len() < 10 { first.push(format!("k={k} gpu={:.5} ref={:.5}", outs[k], ref_row[k])); }
        }
    }
    Ok(format!("dbg-q3b: 불일치 {bad}/{n_in} | {}", first.join(" · ")))
}

/// mmv-check (plans/40) — llama mul_mat_vec_q5_k 직접 구동 격리 측정 (t≥1 dmmv).
/// 스펙 {BLOCK 64, NUM_ROWS 2, COLS 1} + full_subgroups(강제 wave64) —
/// llama RADV 설정 직역. B=f32 [t][K], D=f32 [t][M].
pub fn mmv_check(path: &str, tname: &str, t: usize) -> Result<String, String> {
    let model = llm170_core::qwen35::Model::load(std::path::Path::new(path))
        .map_err(|e| e.to_string())?;
    let w = model.w(tname).ok_or("텐서 없음")?;
    let n_in = w.n_in as usize;
    let n_out = w.n_out as usize;
    let spv = std::fs::read("crates/backend-gpu/src/rawvk/spv/mmv_llm.spv").map_err(|e| e.to_string())?;
    let acc = VkAcc::new()?;
    let mut ctxg = acc.ctx.lock();
    let ab_vram = std::env::var("VKMMQ_VRAM").map(|v| v == "1").unwrap_or(true);
    let ab = if ab_vram {
        let mut b = ctxg.alloc(w.data.len())?;
        unsafe { std::ptr::copy_nonoverlapping(w.data.as_ptr(), b.ptr, w.data.len()) };
        ctxg.unmap(&mut b)?;
        b
    } else {
        let mut b = ctxg.alloc_host(w.data.len())?;
        unsafe { std::ptr::copy_nonoverlapping(w.data.as_ptr(), b.ptr, w.data.len()) };
        ctxg.unmap(&mut b)?;
        b
    };
    // y f32 [t][K]
    let mut seed = 0x1234u64;
    let mut lcg = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 33) as f32 / 2147483648.0 - 0.5 };
    let mut yf: Vec<f32> = Vec::with_capacity(n_in * t);
    let y0 = std::env::var("VKMMQ_Y0").map(|v| v == "1").unwrap_or(false);
    for _ in 0..n_in * t { yf.push(if y0 { 0.0 } else { lcg() }); }
    let mut bb = ctxg.alloc_host(n_in * t * 4)?;
    unsafe { std::ptr::copy_nonoverlapping(yf.as_ptr() as *const u8, bb.ptr, n_in * t * 4) };
    ctxg.unmap(&mut bb)?;
    let db = ctxg.alloc_host(n_out * t * 4)?;
    unsafe { std::ptr::write_bytes(db.ptr, 0, n_out * t * 4) };
    // F0/F1 dummy
    let mut fb = ctxg.alloc_host(64)?;
    ctxg.unmap(&mut fb)?;
    let spec = vec![64u32, 2, 1];
    let (_dsl, pl, _dp, ds, pipe) = ctxg.pipeline_spec_fg(&spv, 5, 13 * 4, &spec, true)?;
    ctxg.bind_bufs(ds, &[ab.buf, bb.buf, db.buf, fb.buf, fb.buf]);
    let mut pc: Vec<u32> = vec![
        n_in as u32, n_in as u32, n_in as u32, n_out as u32,   // ncols, stride_a, stride_b, stride_d
        0, 0, 0,            // batch strides
        0,                  // fusion_flags
        0, t as u32, 1, 1, 1,  // base_wg_y, ne02, ne12, b2, b3
    ];
    let _ = &mut pc;
    let pcb: Vec<u8> = pc.iter().flat_map(|v| v.to_le_bytes()).collect();
    let gx = (n_out as u32).div_ceil(2);
    ctxg.begin_batch()?;
    ctxg.run(pl, ds, pipe, &pcb, gx, t as u32, 1)?;
    ctxg.end_batch_wait()?;
    let out: &[f32] = unsafe { std::slice::from_raw_parts(db.ptr as *const f32, n_out * t) };
    eprintln!("mmv dbg: out[0..16]={:?} (y0={})", &out[0..16], y0);
    // 근사 검증: 첫 토큰 첫 4행
    let mut dq = vec![0f32; n_in];
    let mut maxrel = 0f64;
    for &(m, n) in &[(0usize, 0usize), (100, 0), (2000, 0), (6143, 0)] {
        if m >= n_out || n >= t { continue; }
        llm170_core::quant::dequant_row(w.ty, w.data, m as u64, n_in as u64, &mut dq);
        let yrow = &yf[n * n_in..(n + 1) * n_in];
        let acc: f64 = dq.iter().zip(yrow).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
        let got = out[n * n_out + m] as f64;
        let rel = if acc.abs() > 1e-6 { ((got - acc) / acc).abs() } else { got.abs() };
        maxrel = maxrel.max(rel);
    }
    let nrep: u32 = std::env::var("VKMMQ_N").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
    ctxg.begin_batch()?;
    let t0 = std::time::Instant::now();
    for _ in 0..nrep { ctxg.run(pl, ds, pipe, &pcb, gx, t as u32, 1)?; }
    ctxg.end_batch_wait()?;
    let dt = t0.elapsed().as_secs_f64() / nrep as f64;
    Ok(format!(
        "mmv({tname}) t={t}: {dt:.4}ms · maxrel={maxrel:.4} · {:.1}GB/s",
        w.data.len() as f64 / dt / 1e9
    ))
}

/// vk-frame-check — plans/84 B: 프레임 코어(버퍼 레지스트리+엘리먼트와이스+
/// 상주 GEMM)의 CPU 대조 검증. 각 op를 LCG 데이터로 실행해 판독 비교.
pub fn frame_check(path: &str, tname: &str) -> Result<String, String> {
    use std::time::Instant;
    enum AnyModel {
        Q35(llm170_core::qwen35::Model),
        Q4(llm170_core::qwen4exp::Model4),
    }
    let is_q4 = llm170_gguf::GgufFile::open(std::path::Path::new(path))
        .ok()
        .and_then(|g| g.arch().map(|a| a == "qwen4exp"))
        .unwrap_or(false);
    let model = if is_q4 {
        AnyModel::Q4(
            llm170_core::qwen4exp::Model4::load(std::path::Path::new(path))
                .map_err(|e| e.to_string())?,
        )
    } else {
        AnyModel::Q35(
            llm170_core::qwen35::Model::load(std::path::Path::new(path))
                .map_err(|e| e.to_string())?,
        )
    };
    let w = match &model {
        AnyModel::Q35(m) => m.w(tname).ok_or("텐서 없음")?,
        AnyModel::Q4(m) => m.w4(tname).map_err(|e| e.to_string())?,
    };
    let n_in = w.n_in as usize;
    let n_out = w.n_out as usize;
    let acc = VkAcc::new()?;
    let t = 3usize;
    let mut seed = 0x5deece66u64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as f32 / 2147483648.0 - 0.5
    };
    let xs: Vec<Vec<f32>> = (0..t).map(|_| (0..n_in).map(|_| lcg()).collect()).collect();
    let mut fails = 0usize;
    let mut report = String::new();
    let t0 = Instant::now();
    use llm170_core::matmul::FrameState;
    acc.frame_begin(t);

    // ── 1) RmsRows (w_reps=2) ──
    {
        let n = 64usize;
        let reps = 2usize;
        let xh = acc.frame_alloc(n * reps * t)?;
        let wh = acc.frame_alloc(n * reps)?;
        let oh = acc.frame_alloc(n * reps * t)?;
        // hip 규약: 입력 x도 reps*t행 (res_hc는 hc 반복 레이아웃).
        let mut x = Vec::with_capacity(n * reps * t);
        for _ in 0..reps * t {
            x.extend((0..n).map(|_| lcg()));
        }
        let wv: Vec<f32> = (0..n * reps).map(|_| lcg()).collect();
        acc.frame_write(xh, &x)?;
        acc.frame_write(wh, &wv)?;
        use llm170_core::matmul::FrameHost;
        acc.frame_op(&llm170_core::matmul::FrameOp::RmsRows {
            x: xh, w: wh, out: oh, eps: 1e-5, n, w_reps: reps,
        })?;
        let mut got = vec![0f32; n * reps * t];
        acc.frame_read(oh, &mut got)?;
        let mut mx = 0f64;
        for row in 0..reps * t {
            let s: f64 = (0..n).map(|i| (x[row * n + i] as f64).powi(2)).sum();
            let inv = 1.0 / (s / n as f64 + 1e-5).sqrt();
            for i in 0..n {
                let exp = (x[row * n + i] as f64) * inv * wv[(row % reps) * n + i] as f64;
                mx = mx.max((got[row * n + i] as f64 - exp).abs());
            }
        }
        let ok = mx < 5e-5;
        if !ok { fails += 1; }
        report.push_str(&format!("RmsRows(w_reps={reps}) max|D|={mx:.2e} {} | ", if ok { "OK" } else { "FAIL" }));
        acc.frame_free(xh)?; acc.frame_free(wh)?; acc.frame_free(oh)?;
    }
    // ── 2) SiluDiv / 3) SiluMul / 4) Scale ──
    {
        let n = 256usize;
        let a: Vec<f32> = (0..n).map(|_| lcg()).collect();
        let b: Vec<f32> = (0..n).map(|_| lcg()).collect();
        let ah = acc.frame_alloc(n)?;
        let bh = acc.frame_alloc(n)?;
        let oh = acc.frame_alloc(n)?;
        acc.frame_write(ah, &a)?;
        acc.frame_write(bh, &b)?;
        use llm170_core::matmul::FrameHost;
        acc.frame_op(&llm170_core::matmul::FrameOp::SiluDiv { t: ah, div: 320.0, n })?;
        let mut got = vec![0f32; n];
        acc.frame_read(ah, &mut got)?;
        let mut mx = 0f64;
        // CPU 참조(stages/hc.rs)와 동일: silu(x/div). (구 검증식 silu(x)/div 는
        // 셰이더와 같은 잘못을 새겨넣고 있었다 — plans/86 §1.)
        for i in 0..n {
            let x = a[i] / 320.0f32;
            let exp = (x / (1.0 + (-x).exp())) as f64;
            mx = mx.max((got[i] as f64 - exp).abs());
        }
        let ok = mx < 5e-6;
        if !ok { fails += 1; }
        report.push_str(&format!("SiluDiv max|D|={mx:.2e} {} | ", if ok { "OK" } else { "FAIL" }));

        acc.frame_write(ah, &a)?;
        acc.frame_op(&llm170_core::matmul::FrameOp::SiluMul { g: ah, u: bh, out: oh, n })?;
        acc.frame_read(oh, &mut got)?;
        mx = 0.0;
        for i in 0..n {
            let exp = (a[i] / (1.0 + (-a[i] as f32).exp())) as f64 * b[i] as f64;
            mx = mx.max((got[i] as f64 - exp).abs());
        }
        let ok = mx < 5e-6;
        if !ok { fails += 1; }
        report.push_str(&format!("SiluMul max|D|={mx:.2e} {} | ", if ok { "OK" } else { "FAIL" }));

        acc.frame_write(ah, &a)?;
        acc.frame_op(&llm170_core::matmul::FrameOp::Scale { t: ah, s: 0.5, n })?;
        acc.frame_read(ah, &mut got)?;
        mx = 0.0;
        for i in 0..n {
            mx = mx.max((got[i] as f64 - a[i] as f64 * 0.5).abs());
        }
        let ok = mx < 1e-7;
        if !ok { fails += 1; }
        report.push_str(&format!("Scale max|D|={mx:.2e} {} | ", if ok { "OK" } else { "FAIL" }));
        acc.frame_free(ah)?; acc.frame_free(bh)?; acc.frame_free(oh)?;
    }
    // ── 5) CopyRows / 6) BcastRows / 7) AxpyScaled(t) ──
    {
        let n = 100usize;
        let src: Vec<f32> = (0..n).map(|_| lcg()).collect();
        let sh = acc.frame_alloc(n)?;
        let dh = acc.frame_alloc(2 * n)?;
        acc.frame_write(sh, &src)?;
        use llm170_core::matmul::FrameHost;
        acc.frame_op(&llm170_core::matmul::FrameOp::CopyRows { src: sh, dst: dh, src_off: 7, dst_off: n + 3, n: n - 10 })?;
        let mut got = vec![0f32; 2 * n];
        acc.frame_read(dh, &mut got)?;
        let mut ok = true;
        for i in 0..(n - 10) {
            if (got[n + 3 + i] - src[7 + i]).abs() > 1e-7 { ok = false; break; }
        }
        if !ok { fails += 1; }
        report.push_str(&format!("CopyRows {} | ", if ok { "OK" } else { "FAIL" }));

        let bh2 = acc.frame_alloc(n * t)?;
        acc.frame_op(&llm170_core::matmul::FrameOp::BcastRows { src: sh, dst: bh2, n, rows: t })?;
        acc.frame_read(bh2, &mut got)?;
        let _ = &mut got;
        let mut got2 = vec![0f32; n * t];
        acc.frame_read(bh2, &mut got2)?;
        ok = true;
        for r in 0..t {
            for i in 0..n {
                if (got2[r * n + i] - src[i]).abs() > 1e-7 { ok = false; }
            }
        }
        if !ok { fails += 1; }
        report.push_str(&format!("BcastRows {} | ", if ok { "OK" } else { "FAIL" }));

        let per = 32usize;
        let y: Vec<f32> = (0..t * per).map(|_| lcg()).collect();
        let xx: Vec<f32> = (0..t * per).map(|_| lcg()).collect();
        let ss: Vec<f32> = (0..t).map(|_| lcg()).collect();
        let yh = acc.frame_alloc(t * per)?;
        let xh2 = acc.frame_alloc(t * per)?;
        let ssh = acc.frame_alloc(t)?;
        acc.frame_write(yh, &y)?;
        acc.frame_write(xh2, &xx)?;
        acc.frame_write(ssh, &ss)?;
        acc.frame_op(&llm170_core::matmul::FrameOp::AxpyScaled { y: yh, x: xh2, s: ssh, n: t * per })?;
        let mut got3 = vec![0f32; t * per];
        acc.frame_read(yh, &mut got3)?;
        let mut mx = 0f64;
        for j in 0..t * per {
            let exp = y[j] as f64 + xx[j] as f64 * ss[j / per] as f64;
            mx = mx.max((got3[j] as f64 - exp).abs());
        }
        let aok = mx < 1e-6;
        if !aok { fails += 1; }
        report.push_str(&format!("AxpyScaled(t={t}) max|D|={mx:.2e} {}", if aok { "OK" } else { "FAIL" }));
        acc.frame_free(sh)?; acc.frame_free(dh)?; acc.frame_free(bh2)?;
        acc.frame_free(yh)?; acc.frame_free(xh2)?; acc.frame_free(ssh)?;
    }
    // ── 8) frame_mm — 상주 quant+GEMM vs CPU 디양자화 내적 ──
    {
        let xh = acc.frame_alloc(n_in * t)?;
        let oh = acc.frame_alloc(n_out * t)?;
        let mut flat = Vec::with_capacity(n_in * t);
        for row in &xs { flat.extend_from_slice(row); }
        acc.frame_write(xh, &flat)?;
        use llm170_core::matmul::FrameHost;
        acc.frame_mm(xh, &w, oh, t)?;
        let mut got = vec![0f32; n_out * t];
        acc.frame_read(oh, &mut got)?;
        let mut mx = 0f64;
        let mut ref_row = vec![0f32; n_in];
        for (j, x) in xs.iter().enumerate() {
            for r in 0..n_out.min(16) {
                llm170_core::quant::dequant_row(w.ty, w.data, r as u64, n_in as u64, &mut ref_row);
                let dot: f32 = ref_row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                if std::env::var_os("LLM170_DBG_8").is_some() && j == 0 && r < 6 {
                    eprintln!("[8] r={r} got={:.6} ref={:.6}", got[j * n_out + r], dot);
                }
                mx = mx.max((dot as f64 - got[j * n_out + r] as f64).abs());
            }
        }
        let ok = mx < 5e-3;
        if !ok { fails += 1; }
        report.push_str(&format!("frame_mm max|D|={mx:.2e} {}", if ok { "OK" } else { "FAIL" }));
        acc.frame_free(xh)?; acc.frame_free(oh)?;
    }
    // ── 8b) frame_mm q4_K 밀집 (plans/88 P2): mode-1 타일 산술 분리 검증 —
    //    그룹화(perm/rowexp) 없이 타일 커널 자체의 CPU 대조. ──
    {
        use llm170_core::matmul::FrameHost;
        if let AnyModel::Q4(m) = &model {
            if let Ok(w4k) = m.w4("blk.0.ffn_gate_shexp.weight") {
                let ni = w4k.n_in as usize;
                let no = w4k.n_out as usize;
                let xs4: Vec<Vec<f32>> = (0..t).map(|_| (0..ni).map(|_| lcg()).collect()).collect();
                let xh = acc.frame_alloc(ni * t)?;
                let oh = acc.frame_alloc(no * t)?;
                let mut flat = Vec::with_capacity(ni * t);
                for row in &xs4 { flat.extend_from_slice(row); }
                acc.frame_write(xh, &flat)?;
                acc.frame_mm(xh, &w4k, oh, t)?;
                let mut got = vec![0f32; no * t];
                acc.frame_read(oh, &mut got)?;
                let mut mx = 0f64;
                let mut ref_row = vec![0f32; ni];
                for (j, x) in xs4.iter().enumerate() {
                    for r in 0..no.min(12) {
                        llm170_core::quant::dequant_row(w4k.ty, w4k.data, r as u64, ni as u64, &mut ref_row);
                        let dot: f32 = ref_row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                        if std::env::var_os("LLM170_DBG_8B").is_some() && j == 0 && r < 4 {
                            eprintln!("[8b] r={r} got={:.6} ref={:.6}", got[j * no + r], dot);
                        }
                        mx = mx.max((dot as f64 - got[j * no + r] as f64).abs());
                    }
                }
                let ok = mx < 5e-3;
                if !ok { fails += 1; }
                report.push_str(&format!("| frame_mm-q4k max|D|={mx:.2e} {}", if ok { "OK" } else { "FAIL" }));
                acc.frame_free(xh)?; acc.frame_free(oh)?;
            }
        }
    }
    // ── 9) MoE: top10 → 그룹 GEMM → 가중합 (게이트 가중, k=10) ──
    {
        use llm170_core::matmul::{FrameHost, FrameState};
        let k = 10usize;
        // FN 게이트 가중은 q4_K 스택(대부분 층) — 스택 텐서 하나로 검증.
        let wg = match &model {
            AnyModel::Q4(m) => m.w4("blk.0.ffn_gate_exps.weight").map_err(|e| e.to_string())?,
            AnyModel::Q35(m) => m.w("blk.0.ffn_gate.weight").ok_or("텐서 없음")?,
        };
        let n_in_m = wg.n_in as usize;
        let ne = match &model {
            AnyModel::Q4(_) => 512usize,
            AnyModel::Q35(_) => 1usize,
        };
        // 스택 텐서: 전문가당 폭만 출력에 쓴다(frame_moe_gemm 규약).
        let n_out_m = wg.n_out as usize / ne;
        if ne == 512 {
            let route: Vec<f32> = (0..t * ne).map(|_| lcg() * 4.0).collect();
            let mxs: Vec<Vec<f32>> = (0..t * k).map(|_| (0..n_in_m).map(|_| lcg()).collect()).collect();
            let rh = acc.frame_alloc(t * ne)?;
            let idh = acc.frame_alloc(t * k)?;
            let wth = acc.frame_alloc(t * k)?;
            let mxh = acc.frame_alloc(t * k * n_in_m)?;
            let mgh = acc.frame_alloc(t * k * n_out_m)?;
            let outh = acc.frame_alloc(t * n_out_m)?;
            acc.frame_write(rh, &route)?;
            let mut flat2 = Vec::with_capacity(t * k * n_in_m);
            for row in &mxs { flat2.extend_from_slice(row); }
            acc.frame_write(mxh, &flat2)?;
            acc.frame_op(&llm170_core::matmul::FrameOp::MoeTop10 { route: rh, ids: idh, wt: wth, n_exp: ne, k_sel: k })?;
            let mut ids_g = vec![0u32; t * k];
            {
                acc.frame_sync();
                let g = acc_frame_ptr(&acc, idh);
                unsafe { std::ptr::copy_nonoverlapping(g as *const u32, ids_g.as_mut_ptr(), t * k) };
            }
            acc.frame_moe_gemm(mxh, &wg, idh, mgh, ne, k)?;
            acc.frame_op(&llm170_core::matmul::FrameOp::MoeWeightedSum { ys: mgh, wt: wth, out: outh, k, n: n_out_m })?;
            let mut got = vec![0f32; t * n_out_m];
            acc.frame_read(outh, &mut got)?;
            // CPU 기준: softmax top-k + 디양자화 내적 + 가중합
            let mut mx = 0f64;
            for tok in 0..t {
                let r = &route[tok * ne..(tok + 1) * ne];
                let m = r.iter().cloned().fold(f32::MIN, f32::max);
                let ps: Vec<f32> = r.iter().map(|&v| (v - m).exp()).collect();
                let zs: f32 = ps.iter().sum();
                let mut idx: Vec<usize> = (0..ne).collect();
                idx.sort_by(|&a, &b| ps[b].partial_cmp(&ps[a]).unwrap().then(a.cmp(&b)));
                let sel: Vec<usize> = idx[..k].to_vec();
                let wsel: Vec<f32> = sel.iter().map(|&e| ps[e] / zs).collect();
                let wsum: f32 = wsel.iter().sum::<f32>().max(6.103515625e-5);
                for (j, &e) in sel.iter().enumerate() {
                    if ids_g[tok * k + j] as usize != e { mx = mx.max(1.0); }
                }
                let per_exp = n_out_m;
                let mut ref_row = vec![0f32; n_in_m];
                for j in 0..per_exp.min(8) {
                    let mut acc2 = 0f64;
                    for (ki, &e) in sel.iter().enumerate() {
                        llm170_core::quant::dequant_row(wg.ty, wg.data, (e * per_exp + j) as u64, n_in_m as u64, &mut ref_row);
                        let dot: f32 = ref_row.iter().zip(mxs[tok * k + ki].iter()).map(|(a, b)| a * b).sum();
                        acc2 += dot as f64 * (wsel[ki] / wsum) as f64;
                    }
                    let d = (got[tok * n_out_m + j] as f64 - acc2).abs();
                    mx = mx.max(d);
                    if std::env::var_os("LLM170_DBG_9A").is_some() && tok == 0 && j < 4 {
                        eprintln!("[9a] tok={tok} j={j} got={:.6} ref={:.6}", got[tok * n_out_m + j], acc2);
                    }
                }
            }
            let ok = mx < 3e-2;
            if !ok { fails += 1; }
            report.push_str(&format!("| MoE(k={k}) max|D|={mx:.2e} {}", if ok { "OK" } else { "FAIL" }));
            for h in [rh, idh, wth, mxh, mgh, outh] { acc.frame_free(h)?; }
        }
    }
    // ── 9b) MoE down direct-ids (plans/88 P1): q5_1 스택 t=1 — ids 직판독
    //    경로의 CPU 대조. 행마다 전문가가 다르고 x 행도 행마다 독립이다
    //    (게이트/up의 브로드캐스트 입력과 달리 행 r 을 정확히 읽어야 한다). ──
    {
        use llm170_core::matmul::{FrameHost, FrameState};
        if let AnyModel::Q4(_) = &model {
            let k = 10usize;
            let ne = 512usize;
            let wd = match &model {
                AnyModel::Q4(m) => m.w4("blk.0.ffn_down_exps.weight").map_err(|e| e.to_string())?,
                AnyModel::Q35(_) => unreachable!(),
            };
            let n_in_d = wd.n_in as usize;
            let n_out_d = wd.n_out as usize / ne;
            let route: Vec<f32> = (0..ne).map(|_| lcg() * 4.0).collect();
            let xs: Vec<Vec<f32>> = (0..k).map(|_| (0..n_in_d).map(|_| lcg()).collect()).collect();
            let rh = acc.frame_alloc(ne)?;
            let idh = acc.frame_alloc(k)?;
            let wth = acc.frame_alloc(k)?;
            let mxh = acc.frame_alloc(k * n_in_d)?;
            let mgh = acc.frame_alloc(k * n_out_d)?;
            acc.frame_write(rh, &route)?;
            let mut flat = Vec::with_capacity(k * n_in_d);
            for row in &xs {
                flat.extend_from_slice(row);
            }
            acc.frame_write(mxh, &flat)?;
            // t=1 — MoeTop10도 t토큰을 찍는다(route ne·ids k 크기 버퍼).
            // direct-ids 경로 강제(§9의 t=3 rows=30 도 지나가지만 q4_K만
            // 거친다. down 은 여기서 t=1 판을 본다).
            acc.frame_begin(1);
            acc.frame_op(&llm170_core::matmul::FrameOp::MoeTop10 {
                route: rh, ids: idh, wt: wth, n_exp: ne, k_sel: k,
            })?;
            let mut ids_g = vec![0u32; k];
            {
                acc.frame_sync();
                let g = acc_frame_ptr(&acc, idh);
                unsafe { std::ptr::copy_nonoverlapping(g as *const u32, ids_g.as_mut_ptr(), k) };
            }
            acc.frame_moe_gemm(mxh, &wd, idh, mgh, ne, k)?;
            acc.frame_begin(t);
            let mut got = vec![0f32; k * n_out_d];
            acc.frame_read(mgh, &mut got)?;
            let m = route.iter().cloned().fold(f32::MIN, f32::max);
            let ps: Vec<f32> = route.iter().map(|&v| (v - m).exp()).collect();
            let mut idx: Vec<usize> = (0..ne).collect();
            idx.sort_by(|&a, &b| ps[b].partial_cmp(&ps[a]).unwrap().then(a.cmp(&b)));
            let sel: Vec<usize> = idx[..k].to_vec();
            let mut mx = 0f64;
            let mut ref_row = vec![0f32; n_in_d];
            for (r, &e) in sel.iter().enumerate() {
                if ids_g[r] as usize != e {
                    mx = mx.max(1.0);
                }
                for j in 0..n_out_d.min(8) {
                    llm170_core::quant::dequant_row(
                        wd.ty, wd.data, (e * n_out_d + j) as u64, n_in_d as u64, &mut ref_row,
                    );
                    let dot: f32 = ref_row.iter().zip(xs[r].iter()).map(|(a, b)| a * b).sum();
                    if std::env::var_os("LLM170_DBG_9B").is_some() && r == 0 {
                        eprintln!(
                            "[9b] r={r} e={e} j={j} got={:.6} ref={:.6}",
                            got[r * n_out_d + j],
                            dot
                        );
                    }
                    let d = (got[r * n_out_d + j] as f64 - dot as f64).abs();
                    mx = mx.max(d);
                }
            }
            let ok = mx < 3e-2;
            if !ok { fails += 1; }
            report.push_str(&format!("| MoE-down-ids(k={k}) max|D|={mx:.2e} {}", if ok { "OK" } else { "FAIL" }));
            for h in [rh, idh, wth, mxh, mgh] { acc.frame_free(h)?; }
        }
    }
    // ── 9c) MoE 타일 대량행 (plans/88 P2): t=210·k=10 → rows=2100 — 디바이스
    //    그룹화+타일 경로의 CPU 대조. 소형(§9 t=3)은 direct-ids만 지나가
    //    않으므로 대량 행이 필요하다. ──
    {
        use llm170_core::matmul::{FrameHost, FrameState};
        if let AnyModel::Q4(_) = &model {
            let k = 10usize;
            let ne = 512usize;
            let t2 = std::env::var("LLM170_T2").ok().and_then(|v| v.parse().ok()).unwrap_or(210usize);
            let wg = match &model {
                AnyModel::Q4(m) => m.w4("blk.0.ffn_gate_exps.weight").map_err(|e| e.to_string())?,
                AnyModel::Q35(_) => unreachable!(),
            };
            let n_in_m = wg.n_in as usize;
            let n_out_m = wg.n_out as usize / ne;
            // t2토큰 × ne 라우트 — 균등 랜덤이면 대부분의 전문가가 비게 되어
            // rows=2100이 희소 행을 만든다(실측 결함 재현 조건).
            let route: Vec<f32> = (0..t2 * ne).map(|_| lcg() * 4.0).collect();
            let mxs: Vec<Vec<f32>> = (0..t2 * k).map(|_| (0..n_in_m).map(|_| lcg()).collect()).collect();
            let rh = acc.frame_alloc(t2 * ne)?;
            let idh = acc.frame_alloc(t2 * k)?;
            let wth = acc.frame_alloc(t2 * k)?;
            let mxh = acc.frame_alloc(t2 * k * n_in_m)?;
            let mgh = acc.frame_alloc(t2 * k * n_out_m)?;
            acc.frame_write(rh, &route)?;
            let mut flat = Vec::with_capacity(t2 * k * n_in_m);
            for row in &mxs { flat.extend_from_slice(row); }
            acc.frame_write(mxh, &flat)?;
            acc.frame_begin(t2);
            acc.frame_op(&llm170_core::matmul::FrameOp::MoeTop10 {
                route: rh, ids: idh, wt: wth, n_exp: ne, k_sel: k,
            })?;
            acc.frame_moe_gemm(mxh, &wg, idh, mgh, ne, k)?;
            acc.frame_begin(t);
            let mut got = vec![0f32; t2 * k * n_out_m];
            acc.frame_read(mgh, &mut got)?;
            let mut ids_g = vec![0u32; t2 * k];
            {
                acc.frame_sync();
                let g = acc_frame_ptr(&acc, idh);
                unsafe { std::ptr::copy_nonoverlapping(g as *const u32, ids_g.as_mut_ptr(), t2 * k) };
            }
            // CPU 참조: (토큰,슬롯) 행별 디양자 내적 — 순열과 무관하게 행 자체가 맞는지.
            let mut mx = 0f64;
            let mut ref_row = vec![0f32; n_in_m];
            for row in 0..t2 * k {
                let e = ids_g[row] as usize;
                for j in 0..n_out_m.min(4) {
                    llm170_core::quant::dequant_row(wg.ty, wg.data, (e * n_out_m + j) as u64, n_in_m as u64, &mut ref_row);
                    let dot: f32 = ref_row.iter().zip(mxs[row].iter()).map(|(a, b)| a * b).sum();
                    let d = (got[row * n_out_m + j] as f64 - dot as f64).abs();
                    if std::env::var_os("LLM170_DBG_9C3").is_some() && d > 5e-3 && row < 40 {
                        eprintln!("[9c3] row={row} e={e} j={j} got={:.6} ref={:.6} d={d:.4}", got[row * n_out_m + j], dot);
                    }
                    if std::env::var_os("LLM170_DBG_9C2").is_some() && row < 210 {
                        eprintln!("[9c] row={row} e={e} j={j} got={:.6} ref={:.6}", got[row * n_out_m + j], dot);
                    }
                    mx = mx.max(d);
                }
            }
            let ok = mx < 3e-2;
            if !ok { fails += 1; }
            report.push_str(&format!("| MoE-tile-2100 max|D|={mx:.2e} {}", if ok { "OK" } else { "FAIL" }));
            for h in [rh, idh, wth, mxh, mgh] { acc.frame_free(h)?; }
        }
    }
    // ── 10) 어텐션 반쪽: HcGateMean/HcCombine/NormGated/GdnBetaG/Sigmoid/Split3 ──
    {
        use llm170_core::matmul::FrameHost;
        let n = 48usize;
        let hc = 4usize;
        // HcGateMean
        let xn: Vec<f32> = (0..t * hc * n).map(|_| lcg()).collect();
        let gate: Vec<f32> = (0..t * hc * n).map(|_| lcg()).collect();
        let xnh = acc.frame_alloc(t * hc * n)?;
        let gth = acc.frame_alloc(t * hc * n)?;
        let mkh = acc.frame_alloc(t * n)?;
        acc.frame_write(xnh, &xn)?;
        acc.frame_write(gth, &gate)?;
        acc.frame_op(&llm170_core::matmul::FrameOp::HcGateMean { xn: xnh, gate: gth, out: mkh, hc, n })?;
        let mut got = vec![0f32; t * n];
        acc.frame_read(mkh, &mut got)?;
        let mut mx = 0f64;
        for ti in 0..t {
            for i in 0..n {
                let mut exp = 0f64;
                for s in 0..hc {
                    let k = (ti * hc + s) * n + i;
                    exp += xn[k] as f64 * (1.0 / (1.0 + (-gate[k] as f64).exp()));
                }
                mx = mx.max((got[ti * n + i] as f64 - exp / hc as f64).abs());
            }
        }
        let ok = mx < 5e-6;
        if !ok { fails += 1; }
        report.push_str(&format!("| HcGateMean {mx:.1e} {}", if ok { "OK" } else { "FAIL" }));

        // HcCombine — res 초기화 후 += 검증
        let res0: Vec<f32> = (0..t * hc * n).map(|_| lcg()).collect();
        let resh = acc.frame_alloc(t * hc * n)?;
        let inj: Vec<f32> = (0..t * hc).map(|_| lcg() * 2.0).collect();
        let ijh = acc.frame_alloc(t * hc)?;
        acc.frame_write(resh, &res0)?;
        acc.frame_write(ijh, &inj)?;
        acc.frame_op(&llm170_core::matmul::FrameOp::HcCombine { res: resh, out: mkh, inj: ijh, hc, n, total: 0 })?;
        let mut resg = vec![0f32; t * hc * n];
        acc.frame_read(resh, &mut resg)?;
        mx = 0.0;
        for ti in 0..t {
            for i in 0..n {
                for s in 0..hc {
                    let g = 2.0 / (1.0 + (-(inj[ti * hc + s] as f64) / hc as f64).exp());
                    let exp = res0[(ti * hc + s) * n + i] as f64 + got[ti * n + i] as f64 * g;
                    mx = mx.max((resg[(ti * hc + s) * n + i] as f64 - exp).abs());
                }
            }
        }
        let ok = mx < 5e-6;
        if !ok { fails += 1; }
        report.push_str(&format!("| HcCombine {mx:.1e} {}", if ok { "OK" } else { "FAIL" }));

        // NormGated(sigmoid) — d=32, n_h=3
        let d = 32usize;
        let nh = 3usize;
        let o3: Vec<f32> = (0..t * nh * d).map(|_| lcg()).collect();
        let z3: Vec<f32> = (0..t * nh * d).map(|_| lcg()).collect();
        let w3: Vec<f32> = (0..nh * d).map(|_| lcg()).collect();
        let o3h = acc.frame_alloc(t * nh * d)?;
        let z3h = acc.frame_alloc(t * nh * d)?;
        let w3h = acc.frame_alloc(nh * d)?;
        let n3h = acc.frame_alloc(t * nh * d)?;
        acc.frame_write(o3h, &o3)?;
        acc.frame_write(z3h, &z3)?;
        acc.frame_write(w3h, &w3)?;
        acc.frame_op(&llm170_core::matmul::FrameOp::NormGated { o: o3h, z: z3h, w: w3h, out: n3h, eps: 1e-5, d, n_h: nh })?;
        let mut ng = vec![0f32; t * nh * d];
        acc.frame_read(n3h, &mut ng)?;
        mx = 0.0;
        for row in 0..t * nh {
            let s: f64 = (0..d).map(|i| (o3[row * d + i] as f64).powi(2)).sum();
            let inv = 1.0 / (s / d as f64 + 1e-5).sqrt();
            for i in 0..d {
                let exp = o3[row * d + i] as f64 * inv * w3[(row % nh) * d + i] as f64 * (1.0 / (1.0 + (-z3[row * d + i] as f64).exp()));
                mx = mx.max((ng[row * d + i] as f64 - exp).abs());
            }
        }
        let ok = mx < 5e-6;
        if !ok { fails += 1; }
        report.push_str(&format!("| NormGated {mx:.1e} {}", if ok { "OK" } else { "FAIL" }));

        // GdnBetaG — dt_rank=6, n_h=6·t
        let dr = 6usize;
        let nh2 = dr * t;
        let b2: Vec<f32> = (0..nh2).map(|_| lcg()).collect();
        let a2: Vec<f32> = (0..nh2).map(|_| lcg()).collect();
        let dtb: Vec<f32> = (0..dr).map(|_| lcg()).collect();
        let sa: Vec<f32> = (0..dr).map(|_| lcg()).collect();
        let (b2h, a2h, dth, sah, bgh) = (acc.frame_alloc(nh2)?, acc.frame_alloc(nh2)?, acc.frame_alloc(dr)?, acc.frame_alloc(dr)?, acc.frame_alloc(nh2 * 2)?);
        acc.frame_write(b2h, &b2)?;
        acc.frame_write(a2h, &a2)?;
        acc.frame_write(dth, &dtb)?;
        acc.frame_write(sah, &sa)?;
        acc.frame_op(&llm170_core::matmul::FrameOp::GdnBetaG { b: b2h, a: a2h, dtb: dth, sa: sah, bg: bgh, n_h: nh2 })?;
        let mut bgv = vec![0f32; nh2 * 2];
        acc.frame_read(bgh, &mut bgv)?;
        mx = 0.0;
        for h in 0..nh2 {
            let h0 = h % dr;
            let e0 = 1.0 / (1.0 + (-b2[h] as f64).exp());
            let x = (a2[h] as f64 + dtb[h0] as f64).min(80.0);
            let sp = (1.0 + x.exp()).ln();
            let e1 = (sp * sa[h0] as f64).exp();
            mx = mx.max((bgv[h * 2] as f64 - e0).abs() + (bgv[h * 2 + 1] as f64 - e1).abs());
        }
        let ok = mx < 5e-6;
        if !ok { fails += 1; }
        report.push_str(&format!("| GdnBetaG {mx:.1e} {}", if ok { "OK" } else { "FAIL" }));

        // Sigmoid + Split3
        let v4: Vec<f32> = (0..128).map(|_| lcg() * 3.0).collect();
        let v4h = acc.frame_alloc(128)?;
        acc.frame_write(v4h, &v4)?;
        acc.frame_op(&llm170_core::matmul::FrameOp::Sigmoid { t: v4h, n: 128 })?;
        let mut sv = vec![0f32; 128];
        acc.frame_read(v4h, &mut sv)?;
        mx = 0.0;
        for j in 0..128 { mx = mx.max((sv[j] as f64 - (1.0 / (1.0 + (-v4[j] as f64).exp()))).abs()); }
        let ok = mx < 5e-7;
        if !ok { fails += 1; }
        report.push_str(&format!("| Sigmoid {mx:.1e} {}", if ok { "OK" } else { "FAIL" }));

        let (n0, n1, n2) = (10usize, 6usize, 8usize);
        let tot = n0 + n1 + n2;
        let s3: Vec<f32> = (0..t * tot).map(|_| lcg()).collect();
        let s3h = acc.frame_alloc(t * tot)?;
        let (d0h, d1h, d2h) = (acc.frame_alloc(t * n0)?, acc.frame_alloc(t * n1)?, acc.frame_alloc(t * n2)?);
        acc.frame_write(s3h, &s3)?;
        acc.frame_op(&llm170_core::matmul::FrameOp::Split3 { src: s3h, d0: d0h, d1: d1h, d2: d2h, n0, n1, n2 })?;
        let mut g0 = vec![0f32; t * n0];
        let mut g1 = vec![0f32; t * n1];
        let mut g2 = vec![0f32; t * n2];
        acc.frame_read(d0h, &mut g0)?;
        acc.frame_read(d1h, &mut g1)?;
        acc.frame_read(d2h, &mut g2)?;
        let mut ok = true;
        for ti in 0..t {
            for j in 0..n0 { if (g0[ti * n0 + j] - s3[ti * tot + j]).abs() > 1e-7 { ok = false; } }
            for j in 0..n1 { if (g1[ti * n1 + j] - s3[ti * tot + n0 + j]).abs() > 1e-7 { ok = false; } }
            for j in 0..n2 { if (g2[ti * n2 + j] - s3[ti * tot + n0 + n1 + j]).abs() > 1e-7 { ok = false; } }
        }
        if !ok { fails += 1; }
        report.push_str(&format!("| Split3 {}", if ok { "OK" } else { "FAIL" }));
        for h in [xnh, gth, mkh, resh, ijh, o3h, z3h, w3h, n3h, b2h, a2h, dth, sah, bgh, v4h, s3h, d0h, d1h, d2h] { acc.frame_free(h)?; }
    }
    // ── 11) GDN AR 청크 불변성 — 단일 t=8 대 2×t=4, 최종 상태·출력 비교 ──
    {
        use llm170_core::matmul::FrameHost as _FH;
        use llm170_core::matmul::FrameState as _FS;
        let hv = 4usize;
        let hk = 2usize;
        // 커널 레이아웃: 상태 u행 = kdim 128(32레인×4) — d≥128 필수(hip 규약).
        let d = 128usize;
        let (ks, vs) = (hk * d, hv * d);
        let full = 8usize;
        let mut s2 = 0xabcdu64;
        let mut lc2 = move || {
            s2 = s2.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s2 >> 33) as f32 / 2147483648.0 - 0.5
        };
        let q8: Vec<f32> = (0..full * ks).map(|_| lc2()).collect();
        let k8: Vec<f32> = (0..full * ks).map(|_| lc2()).collect();
        let v8: Vec<f32> = (0..full * vs).map(|_| lc2()).collect();
        let bg8: Vec<f32> = (0..full * hv * 2).map(|_| lc2()).collect();
        let st0: Vec<f32> = (0..hv * d * d).map(|_| lc2() * 0.1).collect();
        // limit: 처리할 프리픽스 토큰 수. chunk: 청크 크기.
        let run_case = |chunk: usize, limit: usize| -> Result<(Vec<f32>, Vec<f32>), String> {
            let acc2 = VkAcc::new()?;
            acc2.set_ctx_len(64);
            let qh = acc2.frame_alloc(full * ks)?;
            let kh = acc2.frame_alloc(full * ks)?;
            let vh = acc2.frame_alloc(full * vs)?;
            let bh = acc2.frame_alloc(full * hv * 2)?;
            let sh = acc2.frame_alloc(hv * d * d)?;
            acc2.frame_write(qh, &q8)?;
            acc2.frame_write(kh, &k8)?;
            acc2.frame_write(vh, &v8)?;
            acc2.frame_write(bh, &bg8)?;
            acc2.frame_write(sh, &st0)?;
            let mut got = vec![0f32; limit * vs];
            for p0 in (0..limit).step_by(chunk) {
                let tt = chunk.min(limit - p0);
                // 슬라이스 입력 버퍼 (q/k/v/bg의 [p0, p0+tt))
                let qs = acc2.frame_alloc(tt * ks)?;
                let ksb = acc2.frame_alloc(tt * ks)?;
                let vsb = acc2.frame_alloc(tt * vs)?;
                let bsb = acc2.frame_alloc(tt * hv * 2)?;
                let osb = acc2.frame_alloc(tt * vs)?;
                acc2.frame_write(qs, &q8[p0 * ks..(p0 + tt) * ks])?;
                acc2.frame_write(ksb, &k8[p0 * ks..(p0 + tt) * ks])?;
                acc2.frame_write(vsb, &v8[p0 * vs..(p0 + tt) * vs])?;
                acc2.frame_write(bsb, &bg8[p0 * hv * 2..(p0 + tt) * hv * 2])?;
                acc2.frame_begin(tt);
                acc2.frame_gdn_ar(qs, ksb, vsb, bsb, sh, osb, 1, hk, hv, d)?;
                let mut part = vec![0f32; tt * vs];
                acc2.frame_read(osb, &mut part)?;
                got[p0 * vs..(p0 + tt) * vs].copy_from_slice(&part);
                for h in [qs, ksb, vsb, bsb, osb] { acc2.frame_free(h)?; }
            }
            let mut stf = vec![0f32; hv * d * d];
            acc2.frame_read(sh, &mut stf)?;
            Ok((got, stf))
        };
        // t=1 결정론
        let (_a, s0a) = run_case(1, 1).map_err(|e| e.to_string())?;
        let (_b, s0b) = run_case(1, 1).map_err(|e| e.to_string())?;
        let mut m0 = 0f64;
        for i in 0..s0a.len() { m0 = m0.max((s0a[i] as f64 - s0b[i] as f64).abs()); }
        eprintln!("[gnar] t=1 상태 결정론={m0:.1e}");
        // t=8 결정론 + 청크 불변
        // ── 12b) MoE 청크 불변성 — 동일 64토큰, 1호출 vs 4×16 호출 ──
        {
            use llm170_core::matmul::FrameHost as _FH2;
            use llm170_core::matmul::FrameState as _FS2;
            let k10 = 10usize;
            let tt = 64usize;
            let route64: Vec<f32> = {
                let mut s3 = 0xc0deu64;
                (0..tt * 512).map(|_| { s3 = s3.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((s3 >> 33) as f32 / 2147483648.0 - 0.5) * 4.0 }).collect()
            };
            // down 경로도 같은 시험 — n_in=640(행 760B, 패딩 대상).
            let wg2 = match &model {
                AnyModel::Q4(m) => m.w4("blk.0.ffn_down_exps.weight").map_err(|e| e.to_string())?,
                AnyModel::Q35(m) => m.w("blk.0.ffn_down.weight").ok_or("텐서 없음")?,
            };
            let n_g = wg2.n_in as usize;
            let per_out = wg2.n_out as usize / 512;
            let mx_rows: Vec<f32> = {
                let mut s3 = 0x5a5au64;
                (0..tt * n_g).map(|_| { s3 = s3.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s3 >> 33) as f32 / 2147483648.0 - 0.5 }).collect()
            };
            let wd_rows: Vec<f32> = Vec::new();
            let run_moe = |chunk: usize| -> Result<Vec<f32>, String> {
                let acc5 = VkAcc::new()?;
                let mxh = acc5.frame_alloc(tt * k10 * n_g)?;
                let rh = acc5.frame_alloc(tt * 512)?;
                let idh = acc5.frame_alloc(tt * k10)?;
                let wth = acc5.frame_alloc(tt * k10)?;
                let outh = acc5.frame_alloc(tt * k10 * per_out)?;
                // mxsel: 토큰별 10슬롯 동일 입력(토큰 t의 행 = mx_rows[t])
                acc5.frame_write(rh, &route64)?;
                let mut out = vec![0f32; tt * per_out];
                for p0 in (0..tt).step_by(chunk) {
                    let c = chunk.min(tt - p0);
                    // 청크 라우트를 [0, c·512)에 적립 — 엔진이 mroute를 청크행으로
                    // 다시 쓰는 것과 동일 규약.
                    let mut rbuf = vec![0f32; c * 512];
                    rbuf.copy_from_slice(&route64[p0 * 512..(p0 + c) * 512]);
                    // rh는 프레임 버퍼 — 청크 라우트를 앞 c행에 기록(호스트 직접)
                    {
                        let g = acc5.framebufs.lock();
                        let b = g.get(&rh).unwrap();
                        unsafe { std::ptr::copy_nonoverlapping(rbuf.as_ptr(), b.ptr as *mut f32, rbuf.len()) };
                    }
                    // 청크 mx를 [0, c·k10·n_g)에 적립(엔진의 mxsel 규약).
                    {
                        let g = acc5.framebufs.lock();
                        let b = g.get(&mxh).unwrap();
                        unsafe {
                            for t2 in 0..c {
                                for s in 0..k10 {
                                    std::ptr::copy_nonoverlapping(
                                        mx_rows[(p0 + t2) * n_g..].as_ptr(),
                                        b.ptr.add(((t2 * k10 + s) * n_g * 4) as usize) as *mut f32,
                                        n_g);
                                }
                            }
                        }
                    }
                    acc5.frame_begin(c);
                    acc5.frame_op(&llm170_core::matmul::FrameOp::MoeTop10 { route: rh, ids: idh, wt: wth, n_exp: 512, k_sel: k10 })?;
                    acc5.frame_moe_gemm(mxh, &wg2, idh, outh, 512, k10)?;
                    // 게이트 출력만 비교(가중합/스캐터 생략 — gemm 자체 검증).
                    // 취하는 것: 각 토큰의 슬롯0 행(k10행 중 첫 행) — 토큰별 대표.
                    let mut part = vec![0f32; c * k10 * per_out];
                    acc5.frame_read(outh, &mut part)?;
                    for t2 in 0..c {
                        let src = t2 * k10 * per_out;
                        out[(p0 + t2) * per_out..(p0 + t2 + 1) * per_out]
                            .copy_from_slice(&part[src..src + per_out]);
                    }
                }
                let _ = wd_rows;
                Ok(out)
            };
            let m1 = run_moe(64).map_err(|e| e.to_string())?;
            let m2 = run_moe(16).map_err(|e| e.to_string())?;
            let mut mm = 0f64;
            for i in 0..m1.len() { mm = mm.max((m1[i] as f64 - m2[i] as f64).abs()); }
            eprintln!("[moech] 게이트 GEMM 청크 불변 max|D|={mm:.1e}");
            // ── 12c) f32 라우터 폴백 그룹 청크 불변성 — 실제 ffn_gate_inp ──
            {
                use llm170_core::matmul::FrameHost as _FH3;
                let wr = match &model {
                    AnyModel::Q4(m) => m.w4("blk.0.ffn_gate_inp.weight").map_err(|e| e.to_string())?,
                    AnyModel::Q35(m) => m.w("blk.0.ffn_gate.weight").ok_or("텐서 없음")?,
                };
                let nr = wr.n_in as usize;
                let nout_r = wr.n_out as usize;
                let mixr: Vec<f32> = {
                    let mut s3 = 0xfeedfaceu64;
                    (0..64 * nr).map(|_| { s3 = s3.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s3 >> 33) as f32 / 2147483648.0 - 0.5 }).collect()
                };
                let run_r = |chunk: usize| -> Result<Vec<f32>, String> {
                    let acc6 = VkAcc::new()?;
                    let xh = acc6.frame_alloc(64 * nr)?;
                    let oh = acc6.frame_alloc(64 * nout_r)?;
                    acc6.frame_write(xh, &mixr)?;
                    let mut out = vec![0f32; 64 * nout_r];
                    for p0 in (0..64).step_by(chunk) {
                        let c = chunk.min(64 - p0);
                        // 청크 입력을 [0, c·nr)에 적립
                        {
                            let g = acc6.framebufs.lock();
                            let b = g.get(&xh).unwrap();
                            unsafe { std::ptr::copy_nonoverlapping(mixr[p0 * nr..].as_ptr(), b.ptr as *mut f32, c * nr) };
                        }
                        acc6.frame_begin(c);
                        acc6.frame_mm_group(xh, std::slice::from_ref(&wr), std::slice::from_ref(&oh), c)?;
                        let mut part = vec![0f32; c * nout_r];
                        acc6.frame_read(oh, &mut part)?;
                        out[p0 * nout_r..(p0 + c) * nout_r].copy_from_slice(&part);
                    }
                    Ok(out)
                };
                let r1 = run_r(64).map_err(|e| e.to_string())?;
                let r2 = run_r(16).map_err(|e| e.to_string())?;
                let mut mr = 0f64;
                for i in 0..r1.len() { mr = mr.max((r1[i] as f64 - r2[i] as f64).abs()); }
                eprintln!("[rtech] f32 라우터 폴백 청크 불변 max|D|={mr:.1e}");
            }
        }
        let (o1, s1) = run_case(8, 8).map_err(|e| e.to_string())?;
        let (o1b, s1b) = run_case(8, 8).map_err(|e| e.to_string())?;
        let (o2, s2v) = run_case(4, 8).map_err(|e| e.to_string())?;
        let mut mo = 0f64;
        for i in 0..o1.len() { mo = mo.max((o1[i] as f64 - o2[i] as f64).abs()); }
        let mut ms = 0f64;
        for i in 0..s1.len() { ms = ms.max((s1[i] as f64 - s2v[i] as f64).abs()); }
        let mut md = 0f64;
        for i in 0..o1.len() { md = md.max((o1[i] as f64 - o1b[i] as f64).abs()); }
        let mut mds = 0f64;
        for i in 0..s1.len() { mds = mds.max((s1[i] as f64 - s1b[i] as f64).abs()); }
        eprintln!("[gnar] t=8 결정론 out={md:.1e} st={mds:.1e}");
        let mut first_diff = None;
        let mut ndiff = 0usize;
        for i in 0..s1.len() {
            if s1[i] != s1b[i] { ndiff += 1; if first_diff.is_none() { first_diff = Some(i); } }
        }
        eprintln!("[gnar] 첫 상이 idx={:?} 상이={ndiff}/{}", first_diff, s1.len());
        let ok = mo < 1e-5 && ms < 1e-5;
        if !ok { fails += 1; }
        report.push_str(&format!("| GdnARchunk(t1={m0:.0e}) out={mo:.1e} st={ms:.1e} {}", if ok { "OK" } else { "FAIL" }));
        // ── 12) GdnConv 청크 불변성 — conv 링 상태 + 출력 ──
        {
            let ch = 48usize;
            let ck = 4usize;
            let conv_total = 8usize;
            let qkv8: Vec<f32> = (0..conv_total * ch).map(|_| {
                s2 = 0u64.wrapping_add(0); // (클로저 이동으로 새 랜덤은 불가 — 상수 시드 재사용)
                0.0
            }).collect();
            let _ = qkv8;
            let conv_src: Vec<f32> = {
                let mut s3 = 0xfeedu64;
                (0..conv_total * ch).map(|_| {
                    s3 = s3.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    (s3 >> 33) as f32 / 2147483648.0 - 0.5
                }).collect()
            };
            let cw: Vec<f32> = {
                let mut s3 = 0xbeefu64;
                (0..ch * ck).map(|_| {
                    s3 = s3.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    (s3 >> 33) as f32 / 2147483648.0 - 0.5
                }).collect()
            };
            let st0c: Vec<f32> = {
                let mut s3 = 0x1234u64;
                (0..(ck - 1) * ch).map(|_| {
                    s3 = s3.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    (s3 >> 33) as f32 / 2147483648.0 - 0.5
                }).collect()
            };
            let run_conv = |chunk: usize| -> Result<(Vec<f32>, Vec<f32>), String> {
                let acc3 = VkAcc::new()?;
                let src = acc3.frame_alloc(conv_total * ch)?;
                let cwb = acc3.frame_alloc(ch * ck)?;
                let stb = acc3.frame_alloc((ck - 1) * ch)?;
                acc3.frame_write(src, &conv_src)?;
                acc3.frame_write(cwb, &cw)?;
                acc3.frame_write(stb, &st0c)?;
                let mut out = vec![0f32; conv_total * ch];
                for p0 in (0..conv_total).step_by(chunk) {
                    let tt = chunk.min(conv_total - p0);
                    let inb = acc3.frame_alloc(tt * ch)?;
                    let ob = acc3.frame_alloc(tt * ch)?;
                    acc3.frame_write(inb, &conv_src[p0 * ch..(p0 + tt) * ch])?;
                    acc3.frame_begin(tt);
                    acc3.frame_op(&llm170_core::matmul::FrameOp::GdnConv {
                        qkv: inb, cw: cwb, state: stb, out: ob, ch, k: ck, t_len: tt,
                    })?;
                    let mut part = vec![0f32; tt * ch];
                    acc3.frame_read(ob, &mut part)?;
                    out[p0 * ch..(p0 + tt) * ch].copy_from_slice(&part);
                    acc3.frame_free(inb)?;
                    acc3.frame_free(ob)?;
                }
                let mut stf = vec![0f32; (ck - 1) * ch];
                acc3.frame_read(stb, &mut stf)?;
                Ok((out, stf))
            };
            let (c1, k1) = run_conv(8).map_err(|e| e.to_string())?;
            let (c2, k2) = run_conv(4).map_err(|e| e.to_string())?;
            let (_c3, k3) = run_conv(8).map_err(|e| e.to_string())?;
            let mut mo = 0f64;
            for i in 0..c1.len() { mo = mo.max((c1[i] as f64 - c2[i] as f64).abs()); }
            let mut ms = 0f64;
            for i in 0..k1.len() { ms = ms.max((k1[i] as f64 - k2[i] as f64).abs()); }
            let mut mdet = 0f64;
            for i in 0..k1.len() { mdet = mdet.max((k1[i] as f64 - k3[i] as f64).abs()); }
            eprintln!("[gcv] conv out={mo:.1e} st={ms:.1e} 결정론 st={mdet:.1e}");
            let ok = mo < 1e-6 && ms < 1e-6;
            if !ok { fails += 1; }
            report.push_str(&format!("| GdnConvChunk out={mo:.1e} st={ms:.1e} {}", if ok { "OK" } else { "FAIL" }));
            // ── 13) GdnBetaG 청크 불변성 (실측 형상 dr=48) ──
            {
                let dr = 48usize;
                let tot = 64usize;
                let (bsrc, asrc): (Vec<f32>, Vec<f32>) = {
                    let mut s3 = 0x9999u64;
                    let mut f1 = || { s3 = s3.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s3 >> 33) as f32 / 2147483648.0 - 0.5 };
                    ((0..dr * tot).map(|_| f1()).collect(), (0..dr * tot).map(|_| f1()).collect())
                };
                let dtbv: Vec<f32> = {
                    let mut s3 = 0x8888u64;
                    (0..dr).map(|_| { s3 = s3.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s3 >> 33) as f32 / 2147483648.0 - 0.5 }).collect()
                };
                let sav: Vec<f32> = {
                    let mut s3 = 0x7777u64;
                    (0..dr).map(|_| { s3 = s3.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s3 >> 33) as f32 / 2147483648.0 - 0.5 }).collect()
                };
                let run_bg = |chunk: usize| -> Result<Vec<f32>, String> {
                    let acc4 = VkAcc::new()?;
                    let mut out = vec![0f32; dr * 2 * tot];
                    // 공유 dtb/sa 상수
                    let dth = acc4.frame_alloc(dr)?;
                    let sah = acc4.frame_alloc(dr)?;
                    acc4.frame_write(dth, &dtbv)?;
                    acc4.frame_write(sah, &sav)?;
                    for p0 in (0..tot).step_by(chunk) {
                        let tt = chunk.min(tot - p0);
                        let bh = acc4.frame_alloc(dr * tt)?;
                        let ah = acc4.frame_alloc(dr * tt)?;
                        let gh = acc4.frame_alloc(dr * tt * 2)?;
                        acc4.frame_write(bh, &bsrc[p0 * dr..(p0 + tt) * dr])?;
                        acc4.frame_write(ah, &asrc[p0 * dr..(p0 + tt) * dr])?;
                        acc4.frame_begin(tt);
                        acc4.frame_op(&llm170_core::matmul::FrameOp::GdnBetaG { b: bh, a: ah, dtb: dth, sa: sah, bg: gh, n_h: dr * tt })?;
                        let mut part = vec![0f32; dr * 2 * tt];
                        acc4.frame_read(gh, &mut part)?;
                        out[p0 * dr * 2..(p0 + tt) * dr * 2].copy_from_slice(&part);
                        acc4.frame_free(bh)?; acc4.frame_free(ah)?; acc4.frame_free(gh)?;
                    }
                    Ok(out)
                };
                let g1 = run_bg(64).map_err(|e| e.to_string())?;
                let g2 = run_bg(16).map_err(|e| e.to_string())?;
                let mut mb = 0f64;
                for i in 0..g1.len() { mb = mb.max((g1[i] as f64 - g2[i] as f64).abs()); }
                eprintln!("[gbg] chunk64 vs chunk16 max|D|={mb:.1e}");
                let ok = mb < 1e-6;
                if !ok { fails += 1; }
                report.push_str(&format!("| GdnBetaGChunk {mb:.1e} {}", if ok { "OK" } else { "FAIL" }));
            }
            // ── 14) QSA 디코드 선택(qsa_sel_dev) — 호스트 top-k 비트 일치 ──
            // 풀 사전 적립(append) + 디코드 1토큰 선택. 호스트 참조는 셰이더와
            // 동일 산술열(f64 순차 rms, f64 회전, 4누산 도트, 정수 순위).
            {
                use llm170_core::matmul::{FrameState as _, QsaOps as _};
                let (ih, dm, r, top_k) = (16usize, 128usize, 128usize, 512usize);
                let n_bulk = 1024usize;
                let n_past = n_bulk + 1;
                let eps = 1e-5f32;
                let full = 900usize;
                let acc5 = VkAcc::new()?;
                acc5.set_ctx_len(n_past);
                let mut s3 = 0x5a5au64;
                let mut lcg = || {
                    s3 = s3.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    (s3 >> 33) as f32 / 2147483648.0 - 0.5
                };
                let ik_all: Vec<f32> = (0..n_past * dm).map(|_| lcg()).collect();
                let iq1: Vec<f32> = (0..ih * dm).map(|_| lcg()).collect();
                let iqw: Vec<f32> = (0..dm).map(|_| 0.5 + lcg().abs()).collect();
                let ikw: Vec<f32> = (0..dm).map(|_| 0.5 + lcg().abs()).collect();
                let cs: Vec<f32> = (0..n_past * dm).map(|_| lcg()).collect();
                // (a) 풀 사전 적립 — 행 0..n_bulk.
                {
                    let ikh = acc5.frame_alloc(n_bulk * dm)?;
                    acc5.frame_write(ikh, &ik_all[..n_bulk * dm])?;
                    acc5.qsa_idx_append_dev(full, 0, ikh, n_bulk, 0, dm, r, &ikw, &cs, eps)
                        .map_err(|e| e.to_string())?;
                    acc5.frame_free(ikh)?;
                }
                // (b) 디코드 토큰 — qsa_sel_dev가 마지막 행 적립+블록키까지.
                let (sd, od, list_len) = {
                    let ikh = acc5.frame_alloc(dm)?;
                    acc5.frame_write(ikh, &ik_all[n_bulk * dm..])?;
                    let iqh = acc5.frame_alloc(ih * dm)?;
                    acc5.frame_write(iqh, &iq1)?;
                    let out = acc5
                        .qsa_sel_dev(full, 0, iqh, ikh, 1, n_bulk, ih, dm, r, top_k, &iqw, &ikw, &cs, eps)
                        .map_err(|e| e.to_string())?;
                    acc5.frame_free(ikh)?;
                    acc5.frame_free(iqh)?;
                    out
                };
                // (c) 호스트 참조.
                let n_blocks = n_past / r;
                let rms_scale = |parts: &[f32; 32]| -> f32 {
                    let mut sum = 0f64;
                    for uu in 0..32 {
                        sum += parts[uu] as f64;
                    }
                    1.0f32 / (sum / dm as f64 + eps as f64).sqrt() as f32
                };
                let rope = |v: &mut [f32], csrow: &[f32]| {
                    let half = dm / 2;
                    for p in 0..half {
                        let c = csrow[p * 2] as f64;
                        let sf = csrow[p * 2 + 1] as f64;
                        let (x0, x1) = (v[p] as f64, v[p + half] as f64);
                        v[p] = (x0 * c - x1 * sf) as f32;
                        v[p + half] = (x0 * sf + x1 * c) as f32;
                    }
                };
                let mut bk = vec![0f32; n_blocks * dm];
                for b in 0..n_blocks {
                    let mut pvs = [[0f32; 4]; 32];
                    let mut parts = [0f32; 32];
                    for u in 0..32 {
                        for j in 0..r {
                            let row = &ik_all[(b * r + j) * dm..][..dm];
                            for k in 0..4 {
                                pvs[u][k] += row[u * 4 + k];
                            }
                        }
                        for k in 0..4 {
                            pvs[u][k] /= r as f32;
                        }
                        parts[u] = pvs[u][0] * pvs[u][0]
                            + pvs[u][1] * pvs[u][1]
                            + pvs[u][2] * pvs[u][2]
                            + pvs[u][3] * pvs[u][3];
                    }
                    let scale = rms_scale(&parts);
                    let out = &mut bk[b * dm..][..dm];
                    for u in 0..32 {
                        for k in 0..4 {
                            out[u * 4 + k] = pvs[u][k] * scale * ikw[u * 4 + k];
                        }
                    }
                    rope(out, &cs[(b * r) * dm..]);
                }
                let mut iqr = vec![0f32; ih * dm];
                {
                    for h in 0..ih {
                        let mut parts = [0f32; 32];
                        let row = &iq1[h * dm..(h + 1) * dm];
                        for u in 0..32 {
                            let mut mp = 0f32;
                            for k in 0..4 {
                                let dv = row[u * 4 + k];
                                mp += dv * dv;
                            }
                            parts[u] = mp;
                        }
                        let scale = rms_scale(&parts);
                        for u in 0..32 {
                            for k in 0..4 {
                                iqr[h * dm + u * 4 + k] = row[u * 4 + k] * scale * iqw[u * 4 + k];
                            }
                        }
                        rope(&mut iqr[h * dm..][..dm], &cs[n_bulk * dm..]);
                    }
                }
                let mut scores = vec![0f32; n_blocks];
                for b in 0..n_blocks {
                    let mut sc = 0f32;
                    for h in 0..ih {
                        let (mut d0, mut d1, mut d2, mut d3) = (0f32, 0f32, 0f32, 0f32);
                        let qh = &iqr[h * dm..(h + 1) * dm];
                        let pk = &bk[b * dm..(b + 1) * dm];
                        let mut i2 = 0usize;
                        while i2 + 4 <= dm {
                            d0 += qh[i2] * pk[i2];
                            d1 += qh[i2 + 1] * pk[i2 + 1];
                            d2 += qh[i2 + 2] * pk[i2 + 2];
                            d3 += qh[i2 + 3] * pk[i2 + 3];
                            i2 += 4;
                        }
                        let dot = (d0 + d1) + (d2 + d3);
                        if dot > 0.0 {
                            sc += dot;
                        }
                    }
                    scores[b] = sc;
                }
                let tail_start = n_blocks * r;
                let tail_cnt = n_past - tail_start;
                let width = n_past.min(top_k + r - 1);
                let n_sel = ((width - tail_cnt) / r).min(n_blocks);
                let mut h_idx = Vec::with_capacity(n_sel * r + tail_cnt);
                let mut sel: Vec<usize> = (0..n_blocks)
                    .filter(|&b| {
                        let sb = scores[b];
                        (0..n_blocks)
                            .filter(|&b2| {
                                let s2 = scores[b2];
                                s2 > sb || (s2 == sb && b2 < b)
                            })
                            .count()
                            < n_sel
                    })
                    .collect();
                sel.sort_unstable();
                for &b in &sel {
                    for j in 0..r {
                        h_idx.push((b * r + j) as u32);
                    }
                }
                for j in 0..tail_cnt {
                    h_idx.push((tail_start + j) as u32);
                }
                let h_off = vec![0u32, (n_sel * r + tail_cnt) as u32];
                let dev_scores: Vec<f32> = {
                    let g = acc5.qsa_sel_bufs.lock();
                    let b = g.as_ref().unwrap();
                    (0..n_blocks)
                        .map(|i| unsafe { *(b.1.ptr.add(i * 4) as *const f32) })
                        .collect()
                };
                eprintln!("[qsel] host_scores={:?}", &scores[..n_blocks.min(8)]);
                eprintln!("[qsel] dev_scores={:?}", &dev_scores[..n_blocks.min(8)]);
                // (d) 대조 — 목록 전체 비트 일치.
                let (d_idx, d_off) = acc5.qsa_sel_readback(sd, od, list_len).map_err(|e| e.to_string())?;
                let ok = list_len == h_idx.len() && d_idx == h_idx && d_off == h_off;
                eprintln!(
                    "[qsel] n_blocks={n_blocks} n_sel={n_sel} list={list_len} host_sel={:?}",
                    sel
                );
                if !ok {
                    fails += 1;
                }
                report.push_str(&format!(
                    "| QsaSelDev list={list_len} {}",
                    if ok { "OK" } else { "FAIL" }
                ));
            }
            // ── 15) shexp_gu/shexp_da — 디코드 t=1 융합 vs CPU 디양자화 참조 ──
            // (qwen4exp 전용 — q35는 SKIP)
            if let AnyModel::Q4(m4) = &model {
                use llm170_core::matmul::EwOps as _;
                let il: usize = tname
                    .strip_prefix("blk.")
                    .and_then(|s| s.split('.').next())
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                let wg = m4.w4(&format!("blk.{il}.ffn_gate_shexp.weight")).map_err(|e| e.to_string())?;
                let wu = m4.w4(&format!("blk.{il}.ffn_up_shexp.weight")).map_err(|e| e.to_string())?;
                let wd = m4.w4(&format!("blk.{il}.ffn_down_shexp.weight")).map_err(|e| e.to_string())?;
                let (n1, n2) = (wg.n_in as usize, wg.n_out as usize);
                let x: Vec<f32> = (0..n1).map(|_| lcg()).collect();
                let m0: Vec<f32> = (0..n1).map(|_| lcg()).collect();
                let s_val = 0.7f32;
                let xh = acc.frame_alloc(n1)?;
                let hh = acc.frame_alloc(n2)?;
                let mh = acc.frame_alloc(n1)?;
                let sh = acc.frame_alloc(1)?;
                acc.frame_write(xh, &x)?;
                acc.frame_write(mh, &m0)?;
                acc.frame_write(sh, &[s_val])?;
                acc.frame_begin(1); // AxpyT t=1 판
                acc.shexp_gu(xh, &wg, &wu, hh, n1, n2).map_err(|e| e.to_string())?;
                acc.shexp_da(hh, &wd, sh, mh, n1, n2).map_err(|e| e.to_string())?;
                let mut hgot = vec![0f32; n2];
                acc.frame_read(hh, &mut hgot)?;
                let mut mgot = vec![0f32; n1];
                acc.frame_read(mh, &mut mgot)?;
                // CPU 참조 — 디양자화 내적 + silu + sigmoid·axpy.
                let mut grow = vec![0f32; n1];
                let mut urow = vec![0f32; n1];
                let mut href = vec![0f32; n2];
                for m in 0..n2 {
                    llm170_core::quant::dequant_row(wg.ty, wg.data, m as u64, n1 as u64, &mut grow);
                    llm170_core::quant::dequant_row(wu.ty, wu.data, m as u64, n1 as u64, &mut urow);
                    let g: f32 = grow.iter().zip(&x).map(|(a, b)| a * b).sum();
                    let u: f32 = urow.iter().zip(&x).map(|(a, b)| a * b).sum();
                    href[m] = (g / (1.0 + (-g).exp())) * u;
                }
                let mut hmx = 0f64;
                for m in 0..n2.min(256) {
                    hmx = hmx.max((href[m] as f64 - hgot[m] as f64).abs());
                }
                let mut drow = vec![0f32; n2];
                let mut mmx = 0f64;
                for i in 0..n1.min(256) {
                    llm170_core::quant::dequant_row(wd.ty, wd.data, i as u64, n2 as u64, &mut drow);
                    let dh: f32 = drow.iter().zip(&href).map(|(a, b)| a * b).sum();
                    let mref = m0[i] + s_val * dh;
                    mmx = mmx.max((mref as f64 - mgot[i] as f64).abs());
                }
                eprintln!("[shexp] h max|D|={hmx:.3e} mout max|D|={mmx:.3e}");
                let ok = hmx < 5e-2 && mmx < 1e-1;
                if !ok {
                    fails += 1;
                }
                report.push_str(&format!(
                    "| Shexp h={hmx:.1e} mout={mmx:.1e} {}",
                    if ok { "OK" } else { "FAIL" }
                ));
                acc.frame_free(xh)?;
                acc.frame_free(hh)?;
                acc.frame_free(mh)?;
                acc.frame_free(sh)?;
            }
        }
        // ── 16) GdnConv 절대 대조(t=1 순차판) — CPU 링 산술과 직접 비교 ──
        // (plans/86 §1: §12는 청크 불변성만 — t<k-1 순차 커널은 커버 밖이었다)
        {
            use llm170_core::matmul::FrameHost as _FH3;
            let (ch, ck, steps) = (48usize, 4usize, 3usize);
            let mut s4 = 0x51ceu64;
            let mut lc4 = move || {
                s4 = s4.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s4 >> 33) as f32 / 2147483648.0 - 0.5
            };
            let cw: Vec<f32> = (0..ch * ck).map(|_| lc4()).collect();
            let mut st: Vec<f32> = (0..(ck - 1) * ch).map(|_| lc4()).collect();
            let qkv: Vec<f32> = (0..steps * ch).map(|_| lc4()).collect();
            let src = acc.frame_alloc(steps * ch)?;
            let cwb = acc.frame_alloc(ch * ck)?;
            let stb = acc.frame_alloc((ck - 1) * ch)?;
            let inb = acc.frame_alloc(ch)?;
            let ob = acc.frame_alloc(ch)?;
            acc.frame_write(src, &qkv)?;
            acc.frame_write(cwb, &cw)?;
            acc.frame_write(stb, &st)?;
            acc.frame_begin(1);
            let mut mo = 0f64;
            for t in 0..steps {
                acc.frame_op(&llm170_core::matmul::FrameOp::CopyRows {
                    src, dst: inb, src_off: t * ch, dst_off: 0, n: ch,
                })?;
                acc.frame_op(&llm170_core::matmul::FrameOp::GdnConv {
                    qkv: inb, cw: cwb, state: stb, out: ob, ch, k: ck, t_len: 1,
                })?;
                let mut got = vec![0f32; ch];
                acc.frame_read(ob, &mut got)?;
                // CPU 참조 — stages/gdn.rs conv 산술 동일열(상태도 진화).
                for c in 0..ch {
                    let mut sum = cw[c * ck + (ck - 1)] * qkv[t * ch + c];
                    for j in 0..ck - 1 {
                        sum += cw[c * ck + j] * st[j * ch + c];
                    }
                    let out_c = sum / (1.0 + (-sum).exp());
                    for j in 0..ck - 2 {
                        st[j * ch + c] = st[(j + 1) * ch + c];
                    }
                    st[(ck - 2) * ch + c] = qkv[t * ch + c];
                    mo = mo.max((got[c] as f64 - out_c as f64).abs());
                }
            }
            let mut stf = vec![0f32; (ck - 1) * ch];
            acc.frame_read(stb, &mut stf)?;
            let mut ms = 0f64;
            for i in 0..st.len() {
                ms = ms.max((stf[i] as f64 - st[i] as f64).abs());
            }
            eprintln!("[gcvabs] out={mo:.1e} st={ms:.1e}");
            let ok = mo < 1e-6 && ms < 1e-6;
            if !ok { fails += 1; }
            report.push_str(&format!("| GdnConvT1 out={mo:.1e} st={ms:.1e} {}", if ok { "OK" } else { "FAIL" }));
            for h in [src, cwb, stb, inb, ob] { acc.frame_free(h)?; }
        }
        // ── 17) GDN AR 절대 대조(t=1) — 전치 상태 + CPU 미러 ──
        {
            use llm170_core::matmul::FrameState as _FS3;
            let (hk, hv, d) = (2usize, 4usize, 128usize);
            let (ks, vs) = (hk * d, hv * d);
            let mut s5 = 0x600du64;
            let mut lc5 = move || {
                s5 = s5.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s5 >> 33) as f32 / 2147483648.0 - 0.5
            };
            let scale = 1.0f32 / (d as f32).sqrt();
            let qs: Vec<f32> = (0..ks).map(|_| lc5() * scale).collect();
            let kk: Vec<f32> = (0..ks).map(|_| lc5()).collect();
            let vv: Vec<f32> = (0..vs).map(|_| lc5()).collect();
            let beta: Vec<f32> = (0..hv).map(|_| 0.4 + 0.4 * lc5()).collect();
            let g: Vec<f32> = (0..hv).map(|_| lc5() * 0.2).collect();
            let mut bg = vec![0f32; hv * 2];
            for h in 0..hv {
                bg[h * 2] = beta[h];
                bg[h * 2 + 1] = g[h].exp();
            }
            let mut st = vec![0f32; hv * d * d];
            for x in st.iter_mut() { *x = lc5() * 0.05; }
            // CPU 미러(사전 스케일 q) — gdn.rs gdn_ar_batch 산술 동일열.
            let st_in = st.clone();
            let mut st_cpu = st_in.clone();
            let mut o_cpu = vec![0f32; vs];
            for h in 0..hv {
                let kh = h % hk;
                let (qb, kb, vb) = (&qs[kh * d..kh * d + d], &kk[kh * d..kh * d + d], &vv[h * d..h * d + d]);
                let s = &mut st_cpu[h * d * d..(h + 1) * d * d];
                let mut sk = vec![0f32; d];
                for kdim in 0..d {
                    for dv in 0..d {
                        let e = &mut s[kdim * d + dv];
                        *e *= bg[h * 2 + 1];
                        sk[dv] += *e * kb[kdim];
                    }
                }
                for dv in 0..d {
                    let delta = (vb[dv] - sk[dv]) * beta[h];
                    for kdim in 0..d {
                        s[kdim * d + dv] += kb[kdim] * delta;
                    }
                }
                for dv in 0..d {
                    let mut o = 0f32;
                    for kdim in 0..d {
                        o += s[kdim * d + dv] * qb[kdim];
                    }
                    o_cpu[h * d + dv] = o;
                }
            }
            // 디바이스: 전치 상태 업로드 → AR → 판독 역전치.
            let tr = |v: &[f32]| -> Vec<f32> {
                let mut o = vec![0f32; v.len()];
                for (cb, b) in v.chunks(d * d).enumerate() {
                    let base = cb * d * d;
                    for kd in 0..d {
                        for dv in 0..d {
                            o[base + dv * d + kd] = b[kd * d + dv];
                        }
                    }
                }
                o
            };
            let qh = acc.frame_alloc(ks)?;
            let kh2 = acc.frame_alloc(ks)?;
            let vh = acc.frame_alloc(vs)?;
            let bh = acc.frame_alloc(hv * 2)?;
            let sh = acc.frame_alloc(hv * d * d)?;
            let oh = acc.frame_alloc(vs)?;
            acc.frame_write(qh, &qs)?;
            acc.frame_write(kh2, &kk)?;
            acc.frame_write(vh, &vv)?;
            acc.frame_write(bh, &bg)?;
            acc.frame_write(sh, &tr(&st_in))?;
            acc.frame_begin(1);
            acc.frame_gdn_ar(qh, kh2, vh, bh, sh, oh, 1, hk, hv, d)?;
            let mut og = vec![0f32; vs];
            acc.frame_read(oh, &mut og)?;
            let mut sg = vec![0f32; hv * d * d];
            acc.frame_read(sh, &mut sg)?;
            let sg = tr(&sg); // 역전치 — CPU 레이아웃으로
            let mut mo = 0f64;
            let mut ms = 0f64;
            for i in 0..vs { mo = mo.max((og[i] as f64 - o_cpu[i] as f64).abs()); }
            for i in 0..st_cpu.len() { ms = ms.max((sg[i] as f64 - st_cpu[i] as f64).abs()); }
            eprintln!("[gnarabs] out={mo:.1e} st={ms:.1e}");
            let ok = mo < 1e-4 && ms < 1e-4;
            if !ok { fails += 1; }
            report.push_str(&format!("| GdnART1 out={mo:.1e} st={ms:.1e} {}", if ok { "OK" } else { "FAIL" }));
            for h in [qh, kh2, vh, bh, sh, oh] { acc.frame_free(h)?; }
        }
        // ── 18) 헤드 체인 절대 대조(t=1) — 실가중 output_hc + output GEMM ──
        if let AnyModel::Q4(m4) = &model {
            use llm170_core::matmul::FrameHost as _FH4;
            let hp = &m4.hp;
            let (n, hc) = (hp.n_embd, hp.hc);
            let w_norm = m4.f32_vec4("output_hc_norm.weight").map_err(|e| e.to_string())?;
            let w_down = m4.w4("output_hc_down.weight").map_err(|e| e.to_string())?;
            let w_up = m4.w4("output_hc_up.weight").map_err(|e| e.to_string())?;
            let w_out = m4.w4("output.weight").map_err(|e| e.to_string())?;
            let mut s6 = 0x7a11u64;
            let mut lc6 = move || {
                s6 = s6.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s6 >> 33) as f32 / 2147483648.0 - 0.5
            };
            // res: 모든 스트림 동일(엔진 임베딩 방송을 미러) — norm 검증에 충분.
            let row: Vec<f32> = (0..n).map(|_| lc6()).collect();
            let mut res = vec![0f32; hc * n];
            for s in 0..hc {
                res[s * n..(s + 1) * n].copy_from_slice(&row);
            }
            let wn = acc.frame_alloc(hc * n)?;
            acc.frame_write(wn, &w_norm)?;
            let rh = acc.frame_alloc(hc * n)?;
            let xnh = acc.frame_alloc(hc * n)?;
            let loh = acc.frame_alloc(w_down.n_out as usize)?;
            let gah = acc.frame_alloc(hc * n)?;
            let hih = acc.frame_alloc(n)?;
            let lgh = acc.frame_alloc(16)?;
            acc.frame_write(rh, &res)?;
            acc.frame_begin(1);
            acc.frame_op(&llm170_core::matmul::FrameOp::RmsRows {
                x: rh, w: wn, out: xnh, eps: hp.eps, n, w_reps: hc,
            })?;
            acc.frame_mm(xnh, &w_down, loh, 1)?;
            acc.frame_op(&llm170_core::matmul::FrameOp::SiluDiv {
                t: loh, div: hc as f32, n: w_down.n_out as usize,
            })?;
            acc.frame_mm(loh, &w_up, gah, 1)?;
            acc.frame_op(&llm170_core::matmul::FrameOp::HcGateMean {
                xn: xnh, gate: gah, out: hih, hc, n,
            })?;
            acc.frame_mm(hih, &w_out, lgh, 1)?;
            let mut lg = vec![0f32; 16];
            acc.frame_read(lgh, &mut lg)?;
            // CPU 참조 — forward.rs 헤드 산술 동일열.
            let mut hxn = vec![0f32; hc * n];
            for s in 0..hc {
                let nn = llm170_core::ops::rms_norm(&row, &w_norm[s * n..(s + 1) * n], hp.eps);
                hxn[s * n..(s + 1) * n].copy_from_slice(&nn);
            }
            let mut hlo = vec![0f32; w_down.n_out as usize];
            llm170_core::matmul::matmul(&hxn, &w_down, &mut hlo);
            for v in hlo.iter_mut() { *v = llm170_core::ops::silu(*v / hc as f32); }
            let mut hgate = vec![0f32; hc * n];
            llm170_core::matmul::matmul(&hlo, &w_up, &mut hgate);
            let mut hin = vec![0f32; n];
            for i in 0..n {
                let mut m = 0f32;
                for s in 0..hc {
                    let k = s * n + i;
                    m += hxn[k] * (1.0 / (1.0 + (-hgate[k]).exp()));
                }
                hin[i] = m / hc as f32;
            }
            let mut hlg = vec![0f32; 16];
            llm170_core::matmul::matmul(&hin, &w_out, &mut hlg);
            let mut mx = 0f64;
            for i in 0..16 { mx = mx.max((lg[i] as f64 - hlg[i] as f64).abs()); }
            let scale = hlg.iter().fold(0f32, |a, &v| a.max(v.abs())) as f64;
            eprintln!("[headabs] max|D|={mx:.3e} (scale={scale:.1})");
            // 3연속 W4A8 GEMM + silu 증폭 — logit-diff.sh 의 MMA 클래스(maxrel<3e-2)와 동일 기준.
            let ok = mx / scale.max(1.0) < 3e-2;
            if !ok { fails += 1; }
            report.push_str(&format!("| HeadChain {mx:.1e} {}", if ok { "OK" } else { "FAIL" }));
            for h in [wn, rh, xnh, loh, gah, hih, lgh] { acc.frame_free(h)?; }
        }
    }
    let _ = t0;
    Ok(format!(
        "vk-frame-check({tname}, t={t}): {} — {} ({} 실패)",
        if fails == 0 { "PASS" } else { "FAIL" },
        report,
        fails
    ))
}

/// plans/87 §1 — 의도적 GPUVM 폴트 프로브: 실제 결함 패턴(디스크립터
/// 오프셋이 버퍼 끝 너머 — pipeline robustness가 주소 자체를 못 구한다)으로
/// 폴트를 유발해 RADV 주소 → va-lookup 체인을 검증한다. DEVICE_LOST가 정상.
pub fn fault_probe() -> Result<String, String> {
    let acc = VkAcc::new()?;
    llm170_diag::alloc::set_on(true);
    llm170_diag::alloc::set_vaddr(true);
    let mut ctx = acc.ctx.lock();
    let b = ctx.alloc_host(4096)?; // 원장 기록(VA 포함)
    let p = acc.pipeline(&mut ctx, Slot::Scale)?;
    let ds = ctx.fresh_ds_for(&p, 1)?;
    // 실효 패턴: 12-바인딩 gemv 파이프라인에 1개만 바인딩 — 미바인딩
    // 디스크립터(3..11)를 커널이 읽는다. 오프셋 초과는 RADV가 빈 범위로
    // 클램프해 폴트가 안 나는 것을 실측했다(정렬 무관).
    let _ = ds;
    let ds2 = ctx.bind_ds(&p, &[b.buf])?;
    let push = push_u32s(&[32u32, 32u32, 8u32, 8u32, 1u32, 1024u32]);
    let r = ctx.run(p.pl, ds2, p.pipe, &push, 1, 1, 1);
    let tsv = llm170_diag::alloc::tsv_path().unwrap_or_else(|| "(없음)".into());
    Ok(format!(
        "발사 결과: {r:?} (Err=DEVICE_LOST 정상) — tsv: {tsv} 에서 RADV 폴트 주소를 va-lookup 하라"
    ))
}

/// plans/86 §6 — 8MiB 순차 pread로 매핑 버퍼 채우기 (hip staged_upload 미러).
fn staged_fill(
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

/// 프레임 버퍼 원시 포인터(프로브 내부용).
fn acc_frame_ptr(acc: &VkAcc, h: u64) -> *mut u8 {
    acc.framebufs.lock().get(&h).map(|b| b.ptr).unwrap_or(std::ptr::null_mut())
}
