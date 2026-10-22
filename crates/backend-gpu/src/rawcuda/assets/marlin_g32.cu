#include "marlin/kernel.h"
#include "marlin/marlin_template.h"

namespace MARLIN_NAMESPACE_NAME {

#define MARLIN_G32_INST(m, th, tn, tk)                                       \
    template __global__ void Marlin<                                          \
        vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(),           \
        vllm::kFloat16.id(), th, m, tn, tk, false, 4, 2, false>(              \
        MARLIN_KERNEL_PARAMS);

MARLIN_G32_INST(1, 256, 16, 4)
MARLIN_G32_INST(2, 256, 16, 4)
MARLIN_G32_INST(3, 256, 16, 4)
MARLIN_G32_INST(4, 256, 16, 4)
MARLIN_G32_INST(1, 128, 8, 4)
MARLIN_G32_INST(2, 128, 8, 4)
MARLIN_G32_INST(3, 128, 8, 4)
MARLIN_G32_INST(4, 128, 8, 4)
MARLIN_G32_INST(1, 128, 4, 8)
MARLIN_G32_INST(2, 128, 4, 8)
MARLIN_G32_INST(3, 128, 4, 8)
MARLIN_G32_INST(4, 128, 4, 8)
MARLIN_G32_INST(1, 256, 8, 8)

#undef MARLIN_G32_INST

}
