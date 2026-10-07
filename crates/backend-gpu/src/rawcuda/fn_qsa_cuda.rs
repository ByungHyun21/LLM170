//! Flash-Next QSA(인덩서 top-k 게이트드 GQA) 모듈층 스켈레톤 — plans/124
//! G001(FNA)→FNB, 2026-10-05. 최대 스테이지(core qsa.rs 635행).
//!
//! [계약 요약 — 전체 판정표는 fn_support.rs 머리 계약 지도]
//! - NEW: 선택 위치 마스크 GQA+게이트(원천: stages/qsa.rs cpu_attn_row
//!   L13-64 · qsa_sel_list L323-358) — exl3_attn_fwd3s(G6)는 T≤8 비마스크
//!   전용이라 대체 불가. q‖gate 인터리브 [n_head·2hd] 레이아웃만 동일.
//! - NEW: 블록 점수·top-k 선택(원천: L209-247, width=min(n_past, top_k+r−1))
//!   ·블록 키 풀링(L165-196)·인덩서 q norm+rope(n_rot=idx_dim=128 전체
//!   회전, L137-145 — G6 prep의 64차 부분회전과 폭 상이).
//! - REUSE 후보: exl3_attn_prep(G6) q/k norm+rope(64차·base 1e7 동일 —
//!   norm 가중은 w−1 규약 대비 원값 저장, 등록 변환 FNB 확정).
//!   k_prenormed: 이미 norm+rope된 k 재적용 금지(qsa.rs L168-171).
//! - REUSE(조건부): 5투영(q/k/v/iq/ik)+o_proj — gemv/gemm2(EXL3,
//!   **krate≤7 — G2 상한 6 초과 실측, 스테이징 확장 필요**)·q4 MMQ(GGUF).
//! - KV 캐시·pos: pp[0] 디바이스 판독(결함 4호) — FnCuda 필드는 FNB 추가.
//! - 어텐션 실패 폴백 금지 조건: core는 CPU 재계산으로 폴백(qsa.rs L489)
//!   — rawcuda 모듈 프로브는 값 maxdiff 판정만(폴백은 엔진 부착 영역).
//!
//! 독립 컴파일 계약(plans/124 G1): std 외 크레이트 금지.

use crate::rawcuda::fn_support::FnCuda;

/// QSA층 수(compress[il]==4 — 48층 중 12).
pub const FN_QSA_LAYERS: usize = 12;
/// QSA 인덩서 상위 k(GGUF qwen4exp.attention.indexer.top_k 실측 2048).
pub const FN_QSA_TOP_K: usize = 2048;
/// QSA 블록 압축비 r(compress[il] — QSA층 4).
pub const FN_QSA_R: usize = 4;

impl FnCuda {
    /// QSA층 진입 — TODO(FNB): stages/qsa.rs qsa_layer L396-515 미러.
    /// 순서 계약: mm_group 5투영(L433-452, 동일 입력 1호출) → qsa_select
    /// (L99 — KV/idx 캐시 적립·블록 키 풀링·top-k) → q norm+rope(L330-357,
    /// 병렬화 누락 회귀 원장 — diverse 게이트) → 선택 리스트 어텐션
    /// (cpu_attn_row 산술) → sigmoid 게이트 → wo 투영.
    /// k는 이미 l2/노름+rope 적용분일 수 있음(k_prenormed — 재적용 금지).
    pub fn qsa_layer(
        &mut self,
        _il: usize,
        _xs: &[Vec<f32>],
        _t_len: usize,
        _full_idx: usize,
    ) -> Result<Vec<Vec<f32>>, String> {
        Err("미구현: FNB(fn_qsa_cuda) 목표".into())
    }

    /// 선택부 — TODO(FNB): stages/qsa.rs qsa_select L99-321 미러.
    /// 패스 A(KV·idx 캐시 적립 + 인덩서 q_rope)·블록 키 캐시(증분 — O(T²)→
    /// O(T) 계약, L155-196)·패스 B(블록 점수 4-전개 dot 양수만 누적·
    /// select_nth_unstable·오름차순 정렬 L209-247).
    pub fn qsa_select(
        &mut self,
        _il: usize,
        _kk: &[Vec<f32>],
        _vv: &[Vec<f32>],
        _iq: &[Vec<f32>],
        _ik: &[Vec<f32>],
        _t_len: usize,
        _k_prenormed: bool,
    ) -> Result<(Vec<u32>, Vec<u32>, usize), String> {
        Err("미구현: FNB(fn_qsa_cuda) 목표".into())
    }

    /// 선택 목록 평탄화 — TODO(FNB): stages/qsa.rs qsa_sel_list L323-358
    /// 미러(블록 오름차순 + 테일 — 마스크 스캔과 산술 순서 동일).
    pub fn qsa_sel_list(
        &self,
        _sel_blk: &[u32],
        _sel_cnt: &[u32],
        _sel_stride: usize,
        _pos0: usize,
        _n_tok: usize,
    ) -> (Vec<u32>, Vec<u32>) {
        (Vec::new(), Vec::new())
    }
}
