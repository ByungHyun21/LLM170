// ── marlin_moe_wna16 (vLLM 벤더링) — 35B/gemma4 프리필 전문가 GEMM [2026-10-19] ──
// 출처: reference/vllm/csrc/moe/marlin_moe_wna16/ (Apache-2.0, vLLM 프로젝트) +
// quantization/marlin 공용 헤더 4종·core/scalar_type.hpp 사본. 벤더 사본 =
// assets/marlin_moe/ — 업스트림 대조 정본, 변경 금지(수정 필요 시 래핑).
// 채리픽 15 인스턴스 = g32(group_blocks=2)·(f16,f16,f16)·m∈{0.5,1,2,3,4} × 생성기
// 규칙(256스레드: m≤1→(128,128)·m>1→(64,256); 128스레드: (64,128)·(128,64) 전 m).
// 0.5 → thread_m_blocks=1 + m_block_size_8=true.
#include "marlin_moe/kernel.h"
#include "marlin_moe/marlin_template.h"

namespace MARLIN_NAMESPACE_NAME {

#define MARLIN_MOE_INST(m, th, tn, tk, m8)                                    \
    template __global__ void Marlin<                                          \
        vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(),           \
        vllm::kFloat16.id(), th, m, tn, tk, m8, 4, 2, false>(                 \
        MARLIN_KERNEL_PARAMS);

// 256스레드 — m≤1: (thread_k=128, thread_n=128)
MARLIN_MOE_INST(1, 256, 8, 8, true) // m_blocks 0.5
MARLIN_MOE_INST(1, 256, 8, 8, false)
// 256스레드 — m>1: (64, 256)
MARLIN_MOE_INST(2, 256, 16, 4, false)
MARLIN_MOE_INST(3, 256, 16, 4, false)
MARLIN_MOE_INST(4, 256, 16, 4, false)
// 128스레드 — (64, 128)
MARLIN_MOE_INST(1, 128, 8, 4, true)
MARLIN_MOE_INST(1, 128, 8, 4, false)
MARLIN_MOE_INST(2, 128, 8, 4, false)
MARLIN_MOE_INST(3, 128, 8, 4, false)
MARLIN_MOE_INST(4, 128, 8, 4, false)
// 128스레드 — (128, 64)
MARLIN_MOE_INST(1, 128, 4, 8, true)
MARLIN_MOE_INST(1, 128, 4, 8, false)
MARLIN_MOE_INST(2, 128, 4, 8, false)
MARLIN_MOE_INST(3, 128, 4, 8, false)
MARLIN_MOE_INST(4, 128, 4, 8, false)

#undef MARLIN_MOE_INST

} // namespace MARLIN_NAMESPACE_NAME
