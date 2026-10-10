// ── gptq4/moe 공용 device 헬퍼·define — [R5 2026-10-10] 분할 시 추출 ──
// 양쪽 .cu가 include 한다(단일 정의 — 값 변경 시 gptq4·moe fatbin 동시
// 리빌드, preflight 스테일 가드가 헤더 mtime을 검사). 소유 항목:
//  G4_LANES — t=1 GEMV + 전 FFMA GEMM 공유(플레인 포함).
//  GEMV_TR — t행 GEMV 블록당 행(호스트 GEMV_TR 미러) — split·플레인 공용.
//  MMA_KC — 전 mma GEMM k-청크 공용.
//  shfl_down_f64 — f64 셔플(tree64 계약 병합의 공통 부품).
//  b2f — bf16→f32(호스트 deq::bf16_to_f32 미러).
//  pk2bf — bf16 쌍 패킹(mma B 프래그먼트).
//  cp_async16/4 — cp.async 스테이징(g128/bf16 mma).
// h2f/f2h는 cast_common.cuh 소유.

#define G4_LANES 64

// [A9-fix2] 블록당 GEMV_TR행 — x(t×k)를 블록 내 재사용(L1)해 행당 x
// L1/L2 재판독을 ÷GEMV_TR. 실측 t=4: 0.114 → 아래 수치.
#define GEMV_TR 8

// [B1/B3 실험] 64→32: smem 절반 → 점유 2배(지연 노출 완화). [marlin-A3
// 2026-10-10 재실측] 64 재시도 = gemm 338→451ms(점유 2블록/SM 악화) — 32 유지.
#define MMA_KC 32

// double 셔플(__shfl_down_sync는 32bit) — 트리 병렬화용.
__device__ __forceinline__ double shfl_down_f64(double v, int off) {
    unsigned lo = __double2loint(v), hi = __double2hiint(v);
    lo = __shfl_down_sync(0xffffffffu, lo, off);
    hi = __shfl_down_sync(0xffffffffu, hi, off);
    return __hiloint2double(hi, lo);
}

// bf16 → f32: 상위 16비트 좌시프트(정확) — core quant::deq::bf16_to_f32 동일.
__device__ __forceinline__ float b2f(unsigned short v) {
    return __uint_as_float(((unsigned)v) << 16);
}

__device__ __forceinline__ unsigned pk2bf(const unsigned short* p) {
    return (unsigned)p[0] | ((unsigned)p[1] << 16);
}

// [marlin-C 2026-10-10] cp.async 스테이징 — 스테이징 지연을 compute와 겹친다
// (ncu: L2 62%·No Eligible 62% = 지연 바운드, 점유 3블록).
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const unsigned sa = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sa), "l"(gmem));
}
__device__ __forceinline__ void cp_async4(void* smem, const void* gmem) {
    const unsigned sa = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(sa), "l"(gmem));
}
