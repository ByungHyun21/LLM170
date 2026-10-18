// ── marlin (vLLM 벤더링) — 27B dense t≥16 W4A16 GEMM 후보 [2026-10-18] ──
// 출처: reference/vllm/csrc/quantization/marlin/ (Apache-2.0, vLLM 프로젝트) +
// core/scalar_type.hpp(동일 라이선스). 벤더 사본 = assets/marlin/ — 변경 금지
// (업스트림 대조 정본; 수정이 필요하면 별도 TU에서 래핑한다).
// 격리 마이크로벤치(2026-10-18 CMP): t=512·n=17408·k=5120 0.84ms(108.7 TF/s) —
// 현행 g128_mma 1.85ms의 2.2×. 본 TU는 채리픽 인스턴스 13종만 방출:
//   A/C/S=f16 · B=U4B8(g128 → group_blocks 8) · stages 4 · no-zp · m_block_size_8=false
//   configs: (threads,k,n)_blocks ∈ {(256,4,16),(128,4,8),(128,8,4)} × m 1..4
//            + (256,8,8)×m1 (small-batch 1순위).
// 커널 시그니처는 MARLIN_KERNEL_PARAMS(kernel.h) — 호스트 인자 18개.
#include "marlin/kernel.h"
#include "marlin/marlin_template.h"

namespace MARLIN_NAMESPACE_NAME {

#define MARLIN_INST(m, th, tn, tk)                                            \
    template __global__ void Marlin<                                          \
        vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(),           \
        vllm::kFloat16.id(), th, m, tn, tk, false, 4, 8, false>(              \
        MARLIN_KERNEL_PARAMS);

// m=1..4 × (256, n16, k4) — large-batch 1순위
MARLIN_INST(1, 256, 16, 4)
MARLIN_INST(2, 256, 16, 4)
MARLIN_INST(3, 256, 16, 4)
MARLIN_INST(4, 256, 16, 4)
// m=1..4 × (128, n8, k4)
MARLIN_INST(1, 128, 8, 4)
MARLIN_INST(2, 128, 8, 4)
MARLIN_INST(3, 128, 8, 4)
MARLIN_INST(4, 128, 8, 4)
// m=1..4 × (128, n4, k8)
MARLIN_INST(1, 128, 4, 8)
MARLIN_INST(2, 128, 4, 8)
MARLIN_INST(3, 128, 4, 8)
MARLIN_INST(4, 128, 4, 8)
// small-batch 1순위 (256, n8, k8)
MARLIN_INST(1, 256, 8, 8)

#undef MARLIN_INST

} // namespace MARLIN_NAMESPACE_NAME
