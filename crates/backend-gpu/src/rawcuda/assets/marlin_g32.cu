// ── marlin g32 밀집(gemma4 dense) — vLLM 벤더 템플릿 group_blocks=2 인스턴스 ───
// [2026-10-19] gemma4-26B dense(g32·bf16 스케일) t>16 전용. 출처·계약은
// marlin.cu(g128, 2026-10-18 포트)와 동일 — 템플릿 인자 group_blocks만
// 8(g128)→2(g32). 독립 .cu인 이유: marlin.cu fatbin에 인스턴스를 추가하면
// 기존 27B g128 코드젠이 재추첨된다(실측 회귀 사례 — gemma4.cu 헤더 참조).
// 채리픽 = marlin.cu와 동일 13종(threads, m, tn, tk).
// 스케일 계약: [k/32][n] f16(marlin 64-청크 순열 — 호스트가 bf16→f32→순열→
// f16 변환; marlin_moe s13/s2와 동일 계약, upload_lin_marlin).
#include "marlin/kernel.h"
#include "marlin/marlin_template.h"

namespace MARLIN_NAMESPACE_NAME {

#define MARLIN_G32_INST(m, th, tn, tk)                                       \
    template __global__ void Marlin<                                          \
        vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(),           \
        vllm::kFloat16.id(), th, m, tn, tk, false, 4, 2, false>(              \
        MARLIN_KERNEL_PARAMS);

// m=1..4 × (256, n16, k4) — large-batch 1순위
MARLIN_G32_INST(1, 256, 16, 4)
MARLIN_G32_INST(2, 256, 16, 4)
MARLIN_G32_INST(3, 256, 16, 4)
MARLIN_G32_INST(4, 256, 16, 4)
// m=1..4 × (128, n8, k4)
MARLIN_G32_INST(1, 128, 8, 4)
MARLIN_G32_INST(2, 128, 8, 4)
MARLIN_G32_INST(3, 128, 8, 4)
MARLIN_G32_INST(4, 128, 8, 4)
// m=1..4 × (128, n4, k8)
MARLIN_G32_INST(1, 128, 4, 8)
MARLIN_G32_INST(2, 128, 4, 8)
MARLIN_G32_INST(3, 128, 4, 8)
MARLIN_G32_INST(4, 128, 4, 8)
// small-batch 1순위 (256, n8, k8)
MARLIN_G32_INST(1, 256, 8, 8)

#undef MARLIN_G32_INST

} // namespace MARLIN_NAMESPACE_NAME
