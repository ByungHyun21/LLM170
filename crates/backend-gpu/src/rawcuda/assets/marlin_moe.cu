#include "marlin_moe/kernel.h"
#include "marlin_moe/marlin_template.h"

namespace MARLIN_NAMESPACE_NAME {

#define MARLIN_MOE_INST(m, th, tn, tk, m8)                                    \
    template __global__ void Marlin<                                          \
        vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(),           \
        vllm::kFloat16.id(), th, m, tn, tk, m8, 4, 2, false>(                 \
        MARLIN_KERNEL_PARAMS);

MARLIN_MOE_INST(1, 256, 8, 8, true)
MARLIN_MOE_INST(1, 256, 8, 8, false)
MARLIN_MOE_INST(2, 256, 16, 4, false)
MARLIN_MOE_INST(3, 256, 16, 4, false)
MARLIN_MOE_INST(4, 256, 16, 4, false)
MARLIN_MOE_INST(1, 128, 8, 4, true)
MARLIN_MOE_INST(1, 128, 8, 4, false)
MARLIN_MOE_INST(2, 128, 8, 4, false)
MARLIN_MOE_INST(3, 128, 8, 4, false)
MARLIN_MOE_INST(4, 128, 8, 4, false)
MARLIN_MOE_INST(1, 128, 4, 8, true)
MARLIN_MOE_INST(1, 128, 4, 8, false)
MARLIN_MOE_INST(2, 128, 4, 8, false)
MARLIN_MOE_INST(3, 128, 4, 8, false)
MARLIN_MOE_INST(4, 128, 4, 8, false)

#undef MARLIN_MOE_INST

}
