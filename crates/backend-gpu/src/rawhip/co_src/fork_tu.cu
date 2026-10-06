#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#define CEIL_DIVIDE(x, size) (((x) + (size) - 1) / (size))
#include "util.cuh"
struct half4 { __half2 x, y, z, w; };
#include "quant/exl3_gemv_kernel.cuh"

// plans/136 P1-1 — 우리 파이프라인용 얇은 래퍼. A는 __half* (HIP half = 구조체 타입),
// C는 void* (FP32 분기가 내부 캐스팅). Hadamard 스테이지는 외부 런치, locks/suh/A_had/svh = nullptr.
template <int bits, bool FP32, int MMODE, int CFG>
__global__ __launch_bounds__(CFG == 0 ? 512 : 256)
void exl3_gemv_j128_w32(const __half* A, const uint16_t* B, void* C,
                        int size_m, int size_k, int size_n)
{
    exl3_gemv_kernel_body<bits, FP32, 2, MMODE, CFG, true, false>
    (A, B, C, size_m, size_k, size_n, nullptr, nullptr, nullptr, nullptr);
}

namespace {
template <int bits, bool FP32, int MMODE, int CFG>
void inst_j128()
{
    auto* p0 = exl3_gemv_j128_w32<bits, FP32, MMODE, CFG>;
    (void)p0;
}
}

void exl3_gemv_j128_w32_instantiate()
{
    inst_j128<4, true,  0, 1>();
    inst_j128<4, true,  1, 1>();
    inst_j128<4, true,  0, 0>();
    inst_j128<4, true,  1, 0>();
    inst_j128<4, false, 0, 1>();
    inst_j128<4, false, 1, 1>();
}
