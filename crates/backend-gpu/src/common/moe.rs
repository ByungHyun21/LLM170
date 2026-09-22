//! moe — 전문가 그룹화 캐시 계약(hip `MoeGroup` ↔ vk `MoeGrp`).
//!
//! 양 백엔드가 각각 유지하는 그룹화 상주 자산의 **의미 계약**. 구현체는
//! 핸들 타입이 달라(hip: raw device ptr u64, vk: VkBuf) 백엔드별로 남긴다 —
//! 산술·패딩 도메인이 커널과 비트계약으로 묶여 있기 때문(90-B3 원칙).
//!
//! # 계약
//!
//! - **세대(generation)**: 라우팅(top10)이 바뀔 때마다 증가. 캐시 히트는
//!   (generation, rows, ids) 일치일 때만 성립 — ids 는 hip은 전문가 id열
//!   비교, vk는 ids 해시(`ids_h`) 비교.
//! - **순열(perm/inv/inv_pad/perm_pad)**: ids → 전문가별 행 재배열.
//!   전문가 내부 순서는 디바이스 그룹화 atomicAdd 경쟁으로 **비결정적이어도
//!   안전** — GEMM은 행별 독립 내적이고 산란(inv)이 원행을 복원한다.
//! - **패딩 도메인([0, bound))**: bound = rows + 16·n_expert(단일 상한).
//!   [rows_pad, bound)는 그룹 커널이 전문가 0행 재판독 + 0 기여로 0채움한다.
//!   타일 그리드는 항상 bound 기준 — 성장 재할당은 bound/rows 가드.
//! - **off[0..=ne]**: 전문가별 시작 행(패딩 도메인 좌표). 그룹 커널이 매
//!   !hit마다 전량 재기록하므로 버퍼 재사용이 안전(세대 무관).
//!
//! 버퍼 성장 가드 원칙(90-A2 누수 수리): 모든 할당은 용량 가드 뒤에 둔다.
//! MoeTop10가 매 스텝 세대를 올리므로 무가드 재할당은 세대당 1회 누수.

/// 그룹화 캐시 히트 판정 — 세대·행수·ids 동일 시 재사용.
pub fn cache_hit(gen_cached: u64, gen_now: u64, rows_cached: usize, rows: usize, ids_match: bool) -> bool {
    gen_cached == gen_now && rows_cached == rows && ids_match
}
