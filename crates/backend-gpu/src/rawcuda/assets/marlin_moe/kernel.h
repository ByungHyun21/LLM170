
#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin_moe_wna16
#endif

#include "quantization/marlin/marlin.cuh"
#include "quantization/marlin/marlin_dtypes.cuh"
#include "core/scalar_type.hpp"

#define MARLIN_KERNEL_PARAMS                                                 \
  const int4 *__restrict__ A, const int4 *__restrict__ B,                    \
      int4 *__restrict__ C, int4 *__restrict__ C_tmp,                        \
      const int4 *__restrict__ b_bias_ptr,                                   \
      const float *__restrict__ a_scales_ptr,                                \
      const int4 *__restrict__ scales_ptr,                                   \
      const float *__restrict__ global_scale_ptr,                            \
      const int4 *__restrict__ zp_ptr,                                       \
      const int32_t *__restrict__ sorted_token_ids_ptr,                      \
      const int32_t *__restrict__ expert_ids_ptr,                            \
      const int32_t *__restrict__ num_tokens_past_padded_ptr,                \
      const float *__restrict__ topk_weights_ptr, int top_k,                 \
      bool mul_topk_weights, int prob_m, int prob_n, int prob_k, int *locks, \
      bool has_bias, bool use_atomic_add, bool use_fp32_reduce

namespace MARLIN_NAMESPACE_NAME {
template <const vllm::ScalarTypeId a_type_id,
          const vllm::ScalarTypeId b_type_id,
          const vllm::ScalarTypeId c_type_id,
          const vllm::ScalarTypeId s_type_id,
          const int threads,
          const int thread_m_blocks,
          const int thread_n_blocks,
          const int thread_k_blocks,
          const bool m_block_size_8,
          const int stages,
          const int group_blocks,
          const bool is_zp_float
          >
__global__ void Marlin(MARLIN_KERNEL_PARAMS);

}
