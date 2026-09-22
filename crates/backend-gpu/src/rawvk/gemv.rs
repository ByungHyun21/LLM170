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
/// plans/89 재개 — q5_1 CM 1-서브그룹 판(64스레드 — tile128v2식 서브그룹 간
/// 경쟁 가설의 정면 검증이자 스칼라 대비 생산 후보).
const FN_MOE_TILE_Q51_SG1_SPV: &[u8] = include_bytes!("spv/fn_moe_tile_q51_sg1.spv");
/// plans/89 재개 — q4_K CM 1-서브그룹 판(엔진 결정적 — q51_sg1로 판명).
const FN_MOE_TILE_Q4K_SG1_SPV: &[u8] = include_bytes!("spv/fn_moe_tile_q4k_sg1.spv");
/// plans/89 재개 — q4_K 8-서브블록 스테이징 판(반복/장벽 q51_sg1과 동일).
const FN_MOE_TILE_Q4K_SG8_SPV: &[u8] = include_bytes!("spv/fn_moe_tile_q4k_sg8.spv");
/// plans/89 재개 — QSA 프리필 디바이스 선택(토큰별 점수·비토닉 top-k).
const FN_IDX_SCORE_MT_SPV: &[u8] = include_bytes!("spv/fn_idx_score_mt.spv");
const FN_IDX_TOPK_MT_SPV: &[u8] = include_bytes!("spv/fn_idx_topk_mt.spv");
/// plans/89 재개 — q4_K sg1 스케일-캐시 판(레지스터 압박 가설).
const FN_MOE_TILE_Q4K_SG1SC_SPV: &[u8] = include_bytes!("spv/fn_moe_tile_q4k_sg1sc.spv");
/// plans/89 재개 — q4_K K-병렬 스칼라 타일(서브그룹 16슬라이스 — ALU 16× 절감).
const FN_MOE_TILE_Q4K_KP_SPV: &[u8] = include_bytes!("spv/fn_moe_tile_q4k_kp.spv");
/// plans/89 P1.4 — PLE 수학 디바이스 3커널(hip q4_ple_* 포트, 비트 동일 목표).
const FN_PLE_GATE_SPV: &[u8] = include_bytes!("spv/fn_ple_gate.spv");
const FN_PLE_CONV_SPV: &[u8] = include_bytes!("spv/fn_ple_conv.spv");
const FN_PLE_RES_SPV: &[u8] = include_bytes!("spv/fn_ple_res.spv");
/// plans/89 P0.4 — QSA 선택 어텐션 멀티헤드(WG=tok×kv헤드, K/V 12× 절감).
const FN_QSA_ATTN_SEL_MH_SPV: &[u8] = include_bytes!("spv/fn_qsa_attn_sel_mh.spv");
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
        let xq0_w = xq_words(n0);
        let xq1_w = xq_words(n_ff);
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
        // 2) gate/up GEMV (같은 xq0) — 상주 출력.
        // plans/89 — t≥2 q8_0/q4_K는 밀집 coopmat 타일로: gemv3 t-루프는
        // 512토큰 프리필에서 ~50ms/디스패치(직렬 t). 레이아웃 동일
        // (outv[tok*n_out+row]). 킬스위치 LLM170_VK_FFNCH=0.
        for (w, obuf) in [(gate_w, bfg), (up_w, bfu)] {
            self.ffn_tile_or_gemv(&mut ctx, w, n0, xq0_w, t, bq0, obuf)?;
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
            self.ffn_tile_or_gemv(&mut ctx, down_w, n_ff, xq1_w, t, bq1, bob)?;
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

    /// ffn_chain GEMV — t≥2 q8_0/q4_K는 coopmat 타일(tile_q8128/q4k128 계열)
    /// 로, 그 외는 종전 gemv3. true=타일 경로 사용.
    fn ffn_tile_or_gemv(
        &self,
        ctx: &mut VkCtx,
        w: &Weight,
        n_in: usize,
        xq_w: usize,
        t: usize,
        xq: vk::Buffer,
        ob: vk::Buffer,
    ) -> Result<bool, String> {
        let n_out = w.n_out as usize;
        let wbufs = self.weight_bufs(ctx, w)?;
        let use_tile = t >= 2
            && std::env::var_os("LLM170_VK_FFNCH").map(|v| v != "0").unwrap_or(true)
            && std::env::var("LLM170_VK_CM").map(|v| v != "0").unwrap_or(true)
            && matches!(w.ty, GgmlType::Q8_0 | GgmlType::Q4K)
            && wbufs.len() == 1
            && vk_ty(w.ty).is_some();
        if !use_tile {
            let ty = vk_ty(w.ty).ok_or("ffn 타입 미지원")?;
            self.gemv_run(ctx, &wbufs, n_in, n_out, xq_w, ty, t, xq, ob)?;
            return Ok(false);
        }
        let (_, _, dbuf) = self.ensure_shared(ctx)?;
        let mut binds: Vec<vk::Buffer> = wbufs.clone();
        while binds.len() < 8 {
            binds.push(dbuf);
        }
        binds.push(xq);
        binds.push(ob);
        let big = t >= 128;
        let slot = match (w.ty, big) {
            (GgmlType::Q8_0, true) => Slot::TileQ8128Cm,
            (GgmlType::Q8_0, false) => Slot::TileQ8msCm,
            (_, true) => Slot::TileQ4k128Cm,
            (_, false) => Slot::TileQ4kmsCm,
        };
        let step = if big { 128usize } else { 64 };
        let p = self.pipeline(ctx, slot)?;
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
        Ok(true)
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

    /// plans/89 재개 — 프리필 다중 토큰 디바이스 선택: (적립+블록키) →
    /// q_rope(t행) → 점수(t×nb) → 토큰별 비토닉 top-k → 평탄 목록+sel_off.
    /// 호스트 d2h 4회(플러시)와 CPU 점수/정렬 소거. nb ≤ 4096.
    #[allow(clippy::too_many_arguments)]
    fn qsa_sel_dev_mt(
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
        if r == 0 || idx_dim != 128 {
            return Err(format!("vk qsa_sel_dev_mt: r={r} idx_dim={idx_dim}"));
        }
        let nb_cap = (pos0 + t) / r;
        if nb_cap > 4096 {
            return Err(format!("vk qsa_sel_dev_mt: nb={nb_cap} > 4096 (호스트 폴백)"));
        }
        // 목록 총길이 — 산술(qsa_sel_list 동일식, 무동기).
        let mut list_len = 0usize;
        for tok in 0..t {
            let n_past = pos0 + tok + 1;
            let nb = n_past / r;
            let tail = n_past - nb * r;
            let width = n_past.min(idx_top_k + r - 1);
            let ns = ((width - tail) / r).min(nb);
            list_len += ns * r + tail;
        }
        self.qsa_idx_append_dev(full_idx, seq, ik, t, pos0, idx_dim, r, ikw, cs_idx, eps)?;
        let iqr_bytes = t * idx_heads * idx_dim * 4;
        let scr_bytes = t * nb_cap.max(1) * 4;
        let cs_bytes = t * (idx_dim / 2) * 2 * 4;
        let sd_bytes = list_len.max(1) * 4;
        let of_bytes = (t + 1) * 4;
        let mut ctx = self.ctx.lock();
        {
            let mut g = self.qsa_sel_bufs.lock();
            let need = match g.as_ref() {
                None => true,
                Some(b) => {
                    b.0.bytes < iqr_bytes
                        || b.1.bytes < scr_bytes
                        || b.4.bytes < cs_bytes
                        || b.5.bytes < sd_bytes
                        || b.6.bytes < of_bytes
                }
            };
            if need {
                *g = Some(crate::rawvk::context::site::scope("qsa_sel", || Ok::<_, String>((
                    ctx.alloc_host(iqr_bytes.max(1 << 16))?,
                    ctx.alloc_host(scr_bytes.max(4096))?,
                    ctx.alloc_host(4096)?,
                    ctx.alloc_host((iqw.len() * 4).max(4096))?,
                    ctx.alloc_host(cs_bytes.max(1 << 16))?,
                    ctx.alloc_host(sd_bytes.max(1 << 16))?,
                    ctx.alloc_host(of_bytes.max(4096))?,
                )))?);
            }
        }
        let (iqr, scr, _flg, iqwb, csb, sdev, ofdev) = {
            let g = self.qsa_sel_bufs.lock();
            let b = g.as_ref().unwrap();
            (b.0.clone(), b.1.clone(), b.2.clone(), b.3.clone(), b.4.clone(), b.5.clone(), b.6.clone())
        };
        unsafe {
            std::ptr::copy_nonoverlapping(iqw.as_ptr(), iqwb.ptr as *mut f32, iqw.len());
            std::ptr::copy_nonoverlapping(
                cs_idx[pos0 * idx_dim..].as_ptr(),
                csb.ptr as *mut f32,
                t * idx_dim,
            );
        }
        let iqb = self.fbuf(iq)?;
        {
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
                .ok_or("vk qsa_sel_dev_mt: bk 풀 없음")?
        };
        if nb_cap > 0 {
            let p = self.pipeline(&mut ctx, Slot::FnIdxScoreMt)?;
            let ds = ctx.bind_ds(&p, &[iqr.buf, bkb, scr.buf])?;
            let push = push_u32s(&[
                pos0 as u32, r as u32, idx_heads as u32, idx_dim as u32, nb_cap as u32,
            ]);
            ctx.run(p.pl, ds, p.pipe, &push, nb_cap.div_ceil(256) as u32, t as u32, 1)?;
        }
        {
            let p = self.pipeline(&mut ctx, Slot::FnIdxTopkMt)?;
            let ds = ctx.bind_ds(&p, &[scr.buf, sdev.buf, ofdev.buf])?;
            let push = push_u32s(&[
                pos0 as u32, t as u32, r as u32, idx_top_k as u32, nb_cap as u32,
            ]);
            ctx.run(p.pl, ds, p.pipe, &push, 1, t as u32, 1)?;
        }
        Ok((sdev.buf.as_raw(), ofdev.buf.as_raw(), list_len))
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
        self.qsa_attn_sel_run(&mut ctx, qb, cb, vb, sib, sob, ob, kq_scale, n_head, n_kv, hd, t)
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
        self.qsa_attn_sel_run(&mut ctx, qb, cb, vb, si_b.buf, so_b.buf, ob, kq_scale, n_head, n_kv, hd, t)
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
        self.qsa_attn_sel_run(&mut ctx, qb, ckb.buf, cvb.buf, si_b.buf, so_b.buf, ob, kq_scale, n_head, n_kv, hd, t)
    }
}

impl VkAcc {
    /// plans/89 P0.4 — 선택 어텐션 발사: (n_head/n_kv)%4==0 이면 멀티헤드 판
    /// (grid (t, n_kv), 256스레드=4sg×헤드 — K/V 판독 12× 절감, 헤드별 산술
    /// 판과 동일). 킬스위치 LLM170_VK_QSAMH=0.
    fn qsa_attn_sel_run(
        &self,
        ctx: &mut VkCtx,
        qb: vk::Buffer,
        cb: vk::Buffer,
        vb: vk::Buffer,
        sib: vk::Buffer,
        sob: vk::Buffer,
        ob: vk::Buffer,
        kq_scale: f32,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
    ) -> Result<(), String> {
        let mh = n_kv >= 1
            && n_head % n_kv == 0
            && (n_head / n_kv) % 4 == 0
            && hd == 256
            && std::env::var("LLM170_VK_QSAMH").map(|v| v != "0").unwrap_or(true);
        let slot = if mh { Slot::FnQsaAttnSelMh } else { Slot::FnQsaAttnSel };
        let p = self.pipeline(ctx, slot)?;
        let ds2 = ctx.bind_ds(&p, &[qb, cb, vb, sib, sob, ob])?;
        let mut push = kq_scale.to_le_bytes().to_vec();
        push.extend_from_slice(&push_u32s(&[n_head as u32, n_kv as u32, hd as u32, t as u32]));
        let gy = if mh { n_kv as u32 } else { n_head as u32 };
        ctx.run(p.pl, ds2, p.pipe, &push, t as u32, gy, 1)
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
        let xq_w = xq_words(n_in);
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
                        generation: 0, rows: 0, ids_h: 0, bound: 0, off_n: 0,
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
                    // off/rows_pad 도 bound/rows 와 동일 성장 가드 — MoeTop10가
                    // 매 스텝 moe_gen 을 올려 !hit 이 항상 참이 되므로, 무가드
                    // 재할당은 세대마다 구 버퍼를 누수시킨다(90 A2 실측 누수).
                    if e.off_n < ne + 1 {
                        e.off = ctx.alloc_host((ne + 1) * 4)?;
                        e.off_n = ne + 1;
                    }
                    if e.rows_pad.bytes == 0 {
                        e.rows_pad = ctx.alloc_host(8)?;
                    }
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
            let q4k_cm = cm_on
                && std::env::var("LLM170_VK_Q4KCM").map(|v| v != "0").unwrap_or(true);
            let slot = match (w.ty, q4k_cm) {
                (GgmlType::Q4K, _k) if wbufs.len() == 1 && std::env::var("LLM170_VK_Q4KKP").map(|v| v == "1").unwrap_or(false) => Slot::FnMoeTileQ4kKp,
                (GgmlType::Q4K, true) if std::env::var("LLM170_VK_Q4KSG1SC").map(|v| v == "1").unwrap_or(false) => Slot::FnMoeTileQ4kSg1sc,
                (GgmlType::Q4K, true) if std::env::var("LLM170_VK_Q4KSG8").map(|v| v == "1").unwrap_or(false) => Slot::FnMoeTileQ4kSg8,
                (GgmlType::Q4K, true) if std::env::var("LLM170_VK_Q4KSG1").map(|v| v == "1").unwrap_or(false) => Slot::FnMoeTileQ4kSg1,
                (GgmlType::Q4K, true) => Slot::FnMoeTileQ4kCm,
                // plans/89 재개: q51_sg1은 엔진 결정적 실측(5회 4동일+타이 1) —
                // 기본 경로로 승격(종전 스칼라는 킬스위치 LLM170_VK_Q51SG1=0).
                // 8sg판(q51_cm)과 q4k_sg1은 엔진 비결정 — 옵트인만.
                (GgmlType::Q5_1, _sg) if wbufs.len() == 1 && std::env::var("LLM170_VK_Q51SG1").map(|v| v != "0").unwrap_or(true) => Slot::FnMoeTileQ51Sg1,
                (GgmlType::Q5_1, _q51cm) if cm_on && std::env::var("LLM170_VK_Q51CM").map(|v| v != "0").unwrap_or(true) => Slot::FnMoeTileQ51Cm,
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
                0u32, // mode — 실험 파생 잔여(90 A2): 프로덕션 항상 0
                rows as u32,
            ]);
            let (gx, gy) = if matches!(slot, Slot::FnMoeTileQ4kKp) {
                (n_out.div_ceil(4) as u32, bound.div_ceil(16) as u32)
            } else if matches!(slot, Slot::FnMoeTileQ51Sg1 | Slot::FnMoeTileQ4kSg1 | Slot::FnMoeTileQ4kSg8 | Slot::FnMoeTileQ4kSg1sc) {
                (n_out.div_ceil(16) as u32, bound.div_ceil(16) as u32)
            } else if cm_on {
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
        let xq_w = xq_words(n_in);
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
        let xq_w = xq_words(n_in);
        let mut ctx = self.ctx.lock();
        let xq = self.value_buf(&mut ctx, &self.xbuf, t * xq_w * 4)?;
        let ob = self.value_buf(&mut ctx, &self.obuf, t * n_out * 4)?;
        self.quant_upload(&mut ctx, xs, n_in, xq)?;
        let wbufs = self.weight_bufs(&mut ctx, w)?;
        // plans/89 — t≥2 q8_0/q4_K는 밀집 coopmat 타일로(dense_mm와 동일
        // 판·동일 수치 클래스). gemv3 t-루프는 512토큰에서 ~50ms 직렬.
        // 킬스위치 LLM170_VK_MBTILE=0.
        if t >= 2
            && matches!(w.ty, GgmlType::Q8_0 | GgmlType::Q4K)
            && wbufs.len() == 1
            && std::env::var_os("LLM170_VK_MBTILE").map(|v| v != "0").unwrap_or(true)
            && std::env::var("LLM170_VK_CM").map(|v| v != "0").unwrap_or(true)
        {
            let (_, _, dbuf) = self.ensure_shared(&mut ctx)?;
            let mut binds: Vec<vk::Buffer> = wbufs.clone();
            while binds.len() < 8 {
                binds.push(dbuf);
            }
            binds.push(xq);
            binds.push(ob);
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
        let xq_w = xq_words(n_in);
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

