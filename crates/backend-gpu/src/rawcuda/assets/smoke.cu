// llm170 rawcuda 스모크 커널 (2026-10-04).
// 계약 소스는 이 .cu — 빌드 자산(smoke.fatbin)은 scripts/build_cuda.bat가
// nvcc -fatbin 으로 같은 디렉터리에 생성·커밋한다(rawhip co/*.co 미러:
// 소스와 자산을 함께 커밋, 커널 산술 변경은 이 파일부터).
//
// 산출 검증: out[i] = in[i]*scale + i. 프로브(구 프로브)는
// in[i]=i*0.5, scale=4.0 을 투입해 out[i]==i*2+i==3i 의 비트동일(f32 정확표현)
// 값을 요구한다 — 스모크는 배관(모듈로드·런치·복사) 검증이므로 근사 허용 없음.
#include <cuda_bf16.h>

extern "C" __global__ void llm170_smoke_add(const float* in, float* out, int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * scale + (float)i;
}

// [2026-10-09] 텐서코어 도구·수치 스모크 — bf16 mma.sync.m16n8k16 → f32 누적.
// A 16×16(행우선) × B 16×8(k우선: b[k][n]) → C 16×8. 프래그먼트 = PTX ISA
// m16n8k16 .bf16 규약(g=lane>>2, t=lane&3): a0/a1 = k 2t,2t+1(행 g/g+8),
// a2/a3 = k 2t+8,2t+9, b0 = k 2t,2t+1(열 g), b1 = k 2t+8,2t+9,
// c0/c1 = 행 g, c2/c3 = 행 g+8(열 2t,2t+1). 호스트 참조는 bf16 RN 반올림
// 입력의 f32 k순 합 — 차이는 누적 순서뿐이므로 허용오차로 판정한다(1b
// 플레인 mma GEMM 착륙 전 도구 검증 — 170HX에서도 이 스모크로 확인 가능).
extern "C" __global__ void llm170_mma_smoke(const float* a, const float* b, float* c) {
    __shared__ float sa[16 * 16];
    __shared__ float sb[16 * 8];
    for (int i = threadIdx.x; i < 16 * 16; i += blockDim.x) {
        sa[i] = a[i];
    }
    for (int i = threadIdx.x; i < 16 * 8; i += blockDim.x) {
        sb[i] = b[i];
    }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    auto pk = [](float x, float y) {
        __nv_bfloat162 h = __floats2bfloat162_rn(x, y);
        return *reinterpret_cast<unsigned*>(&h);
    };
    const unsigned a0 = pk(sa[g * 16 + 2 * t], sa[g * 16 + 2 * t + 1]);
    const unsigned a1 = pk(sa[(g + 8) * 16 + 2 * t], sa[(g + 8) * 16 + 2 * t + 1]);
    const unsigned a2 = pk(sa[g * 16 + 2 * t + 8], sa[g * 16 + 2 * t + 9]);
    const unsigned a3 = pk(sa[(g + 8) * 16 + 2 * t + 8], sa[(g + 8) * 16 + 2 * t + 9]);
    const unsigned b0 = pk(sb[(2 * t) * 8 + g], sb[(2 * t + 1) * 8 + g]);
    const unsigned b1 = pk(sb[(2 * t + 8) * 8 + g], sb[(2 * t + 9) * 8 + g]);
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    c[g * 8 + 2 * t] = c0;
    c[g * 8 + 2 * t + 1] = c1;
    c[(g + 8) * 8 + 2 * t] = c2;
    c[(g + 8) * 8 + 2 * t + 1] = c3;
}
