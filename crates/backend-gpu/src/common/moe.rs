//! moe — 전문가 그룹화 캐시 계약(hip `MoeGroup` ↔ vk `MoeGrp`).
//!
//! 양 백엔드가 각각 유지하는 그룹화 상주 자산의 **의미 계약**. 구현체는
//! 핸들 타입이 달라(hip: raw device ptr u64, vk: VkBuf) 백엔드별로 남긴다 —
//! 산술·패딩 도메인이 커널과 비트계약으로 묶여 있기 때문(90-B3 원칙).
//! 단, 양 백엔드가 **바이트 동일 코드**로 유지하던 순수 계산(상한·카운팅
//! 정렬 테이블·dmmv 게이트)은 plans/110 P13 부터 이 모듈로 공용화한다.
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

use llm170_gguf::GgmlType;

/// 그룹화 캐시 히트 판정 — 세대·행수·ids 동일 시 재사용.
pub fn cache_hit(
    gen_cached: u64,
    gen_now: u64,
    rows_cached: usize,
    rows: usize,
    ids_match: bool,
) -> bool {
    gen_cached == gen_now && rows_cached == rows && ids_match
}

/// 그룹화 단일 상한 — Σ_e ceil(r_e/16)·16 ≤ rows + 16·ne (전문가당 ≤15행 패딩).
///
/// 양쪽 비교(동일식 `rows + 16 * ne`):
/// - hip `frame_moe_gemm` 디바이스 경로(rawhip/q4acc/frame.rs):
///   `let bound = rows + 16 * ne;` — 커널 인자·rowexp/perm_pad 버퍸 크기 공급.
/// - vk `moe_gemm_impl` 타일 경로(rawvk/vkacc/frame.rs):
///   `let bound = rows + 16 * ne;` — 성장 가드·타일 그리드(y=bound/16) 공급.
///
/// 종전엔 이 상한이 각 백엔드에 5곳씩 흩어져 잠재 OOB였다(2026-09-14 단일화).
#[inline]
pub fn grp_bound(rows: usize, ne: usize) -> usize {
    rows + 16 * ne
}

/// 호스트 카운팅 정렬 오프셋 — `off[e+1]-off[e]` = 전문가 e의 행 수, `off[0]=0`.
///
/// 양쪽 비교(6줄 바이트 동일):
/// - hip 호스트 그룹화 `frame_moe_gemm`(rawhip/q4acc/frame.rs)·값 경로
///   `moe_down`(rawhip/q4acc/value.rs):
///   `off[(e as usize).min(ne - 1) + 1] += 1;` + 전방 누적 `off[e+1] += off[e]`.
/// - vk `moe_gemm_impl` 폴백(호스트 그룹화)·GCHECK 진단 재계산
///   (rawvk/vkacc/frame.rs): 위와 같은 식(`hoff` 명칭).
///
/// ids 값이 ne 이상이면 마지막 전문가에 클램프(양측 동일 규약 — 라우팅
/// 이상 시에도 테이블이 역행렬 없이 닫힌다). `ne ≥ 1` 가정(호출부 max(1)).
pub fn grp_offsets(idv: &[u32], ne: usize) -> Vec<usize> {
    let mut off = vec![0usize; ne + 1];
    for &e in idv {
        off[(e as usize).min(ne - 1) + 1] += 1;
    }
    for e in 0..ne {
        off[e + 1] += off[e];
    }
    off
}

/// 카운팅 정렬 순열 — `perm[p]` = 그룹화 위치 p의 원본 행.
///
/// 양쪽 비교(본체 바이트 동일):
/// - hip `frame_moe_gemm`/`moe_down`: `let p = cur[e]; perm[p] = i as u32; cur[e] += 1;`
/// - vk `moe_gemm_impl` 폴백: `perm[cur[e]] = i as u32; cur[e] += 1;` (동일 전개)
pub fn grp_perm(idv: &[u32], ne: usize, off: &[usize]) -> Vec<u32> {
    let mut cur = off[..ne].to_vec();
    let mut perm = vec![0u32; idv.len()];
    for (i, &e) in idv.iter().enumerate() {
        let e = (e as usize).min(ne - 1);
        let p = cur[e];
        perm[p] = i as u32;
        cur[e] += 1;
    }
    perm
}

/// 역순열 — `inv[i]` = 원본 행 i의 그룹화 위치(`perm`의 역함수).
///
/// 양쪽 비교(같은 사상, 전개만 다름 — 출력 동일):
/// - hip `frame_moe_gemm`: 적립 루프 안에서 `inv[i] = p as u32;` (인라인)
/// - vk `moe_gemm_impl` 폴백: `for (p, &orig) in perm.iter().enumerate()`
///   `{ inv[orig as usize] = p as u32; }` (후행 패스)
pub fn grp_inv(perm: &[u32]) -> Vec<u32> {
    let mut inv = vec![0u32; perm.len()];
    for (p, &orig) in perm.iter().enumerate() {
        inv[orig as usize] = p as u32;
    }
    inv
}

/// 16배수 패딩 오프셋(`off_pad[e]` = 패딩 도메인에서 전문가 e 시작 행)과
/// `rows_pad = off_pad[ne].max(16)` — 타일 그리드·yg 버퍼 크기 기준.
///
/// 양쪽 비교:
/// - hip `frame_moe_gemm` 호스트 빌드(rawhip/q4acc/frame.rs):
///   `off_pad[e+1] = off_pad[e] + (off[e+1]-off[e]).div_ceil(16) * 16;`
///   `rows_pad = off_pad[ne].max(16);` (pad=16 리터럴)
/// - vk GCHECK 진단 재계산(rawvk/vkacc/frame.rs): `padmul=16` 로 같은 식
///   (`hpoff`/`rows_pad` — plans/93 sg2 32배수 실험 유산으로 padmul 파라미터).
///
/// `rows_pad` 하한 16은 양측 동일(타일 최소 1개 — 빈 라우팅에도 그리드 성립).
pub fn grp_padded(off: &[usize], ne: usize, pad: usize) -> (Vec<usize>, usize) {
    let mut off_pad = vec![0usize; ne + 1];
    for e in 0..ne {
        off_pad[e + 1] = off_pad[e] + (off[e + 1] - off[e]).div_ceil(pad) * pad;
    }
    let rows_pad = off_pad[ne].max(16);
    (off_pad, rows_pad)
}

/// HIP ids dmmv 판 게이트 — f32 활성 직결 커널이 이 호출을 가져가면
/// 활성 양자화(xq) 자체가 불필요하다. VK는 별도의 `vk_ids2_takes`를 쓴다:
/// HIP에만 Q5K f32 직독 커널이 있어 이 게이트를 공유하면 VK의 Q5K xq
/// 소비자가 null 버퍼를 읽고 0을 출력한다(plans/141, L2.moe_sc 실측).
pub fn ids2_takes(rows: usize, t: usize, ty: GgmlType) -> bool {
    // 2026-10-07 plans/141: q5k dmmv 재승격 — 커널의 qh 바이트 추출 결함을 수리했다
    // (워드 비트 → 바이트 (j&3)*8 시프트 + 2워드 처리). hip-moe-dmmv-check f64 대조:
    // 수리 전 dmmv_err=1.451(>refmax 0.830) → 수리 후 1.7e-7. 10b(활성 양자화
    // 제거, f32 직소비) — f64 오차 보고 완료, 게이트 기준선/골든 재기록 병행.
    rows > 0
        && (t == 1 || rows <= 64)
        && matches!(ty, GgmlType::Q4K | GgmlType::Q5_1 | GgmlType::Q5K)
}

/// VK에서 실제로 f32 직독 ids2 커널이 선택되는 조건. Q5K·Q8_0은
/// `fn_moe_ids`가 xq를 소비한다. 단일 SSBO가 아니면 ids2 분기가
/// 기존 ids 셰이더로 내려가므로 그때도 xq를 생략하면 안 된다.
pub fn vk_ids2_takes(
    rows: usize,
    t: usize,
    ty: GgmlType,
    weight_bytes: usize,
    max_ssbo: usize,
) -> bool {
    ids2_takes(rows, t, ty)
        && matches!(ty, GgmlType::Q4K | GgmlType::Q5_1)
        && weight_bytes <= max_ssbo
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 카운팅 정렬 테이블 불변식 — 그룹 순수성·순열 전단사·역순열 복원.
    #[test]
    fn grp_tables_invariants() {
        let ne = 5usize;
        let idv: Vec<u32> = [3, 0, 3, 4, 4, 4, 9, 0, 1]
            .into_iter()
            .map(|e| e as u32)
            .collect();
        let off = grp_offsets(&idv, ne);
        assert_eq!(off[0], 0);
        assert_eq!(off[ne], idv.len());
        // 클램프: ids의 9는 마지막 전문가(4)에 흡수 — 양측 백엔드 규약.
        assert_eq!(off[4 + 1] - off[4], 4);
        let perm = grp_perm(&idv, ne, &off);
        let inv = grp_inv(&perm);
        // perm 은 전단사.
        let mut seen = vec![false; idv.len()];
        for &i in &perm {
            assert!(!seen[i as usize]);
            seen[i as usize] = true;
        }
        // 그룹 순수성: 위치 [off[e], off[e+1]) 의 원본 행은 전부 전문가 e(클램프 포함).
        for e in 0..ne {
            for p in off[e]..off[e + 1] {
                let orig = idv[perm[p] as usize] as usize;
                assert_eq!(orig.min(ne - 1), e);
            }
        }
        // 역순열 복원: inv[perm[p]] == p (hip 인라인판 == vk 후행판 사상).
        for (p, &orig) in perm.iter().enumerate() {
            assert_eq!(inv[orig as usize], p as u32);
        }
    }

    /// 패딩 도메인 — 16배수 경계·하한 16·단일 상한 준수.
    #[test]
    fn grp_padded_invariants() {
        let ne = 3usize;
        let idv: Vec<u32> = vec![0, 0, 0, 1, 2, 2]; // r = 3,1,2
        let off = grp_offsets(&idv, ne);
        let (off_pad, rows_pad) = grp_padded(&off, ne, 16);
        assert_eq!(off_pad, vec![0, 16, 32, 48]);
        assert_eq!(rows_pad, 48);
        // 빈 라우팅(rows=0)에도 그리드 성립 — rows_pad 하한 16.
        let empty = grp_offsets(&[], ne);
        let (_, rp0) = grp_padded(&empty, ne, 16);
        assert_eq!(rp0, 16);
        // 단일 상한 계약: rows_pad ≤ rows + 16·ne.
        let rows = idv.len();
        assert!(rows_pad <= grp_bound(rows, ne));
    }

    /// ids dmmv 게이트 — rows/t/타입 3축 경계.
    #[test]
    fn ids2_gate() {
        assert!(ids2_takes(64, 8, GgmlType::Q4K)); // rows≤64 프리필 소량
        assert!(ids2_takes(1, 1, GgmlType::Q5_1)); // 디코드
        assert!(!ids2_takes(65, 8, GgmlType::Q4K)); // rows>64 且 t>1
        assert!(ids2_takes(65, 1, GgmlType::Q4K)); // t=1 이면 rows 무관
        assert!(!ids2_takes(64, 8, GgmlType::Q8_0)); // 타입 미지원
        assert!(ids2_takes(64, 8, GgmlType::Q5K)); // q5k dmmv 재승격(plans/141 커널 수리 후)
        assert!(!ids2_takes(0, 1, GgmlType::Q4K)); // 빈 라우팅
    }

    /// HIP의 Q5K 승격과 VK의 xq 소비 계약은 서로 독립이다.
    #[test]
    fn vk_ids2_gate_keeps_xq_for_fallbacks() {
        let one = 128 * 1024 * 1024;
        assert!(vk_ids2_takes(10, 1, GgmlType::Q4K, one, one));
        assert!(vk_ids2_takes(10, 1, GgmlType::Q5_1, one - 1, one));
        assert!(!vk_ids2_takes(10, 1, GgmlType::Q5K, one, one));
        assert!(!vk_ids2_takes(10, 1, GgmlType::Q8_0, one, one));
        assert!(!vk_ids2_takes(10, 1, GgmlType::Q4K, one + 1, one));
        assert!(!vk_ids2_takes(65, 8, GgmlType::Q4K, one, one));
        assert!(!vk_ids2_takes(0, 1, GgmlType::Q4K, one, one));
    }
}
