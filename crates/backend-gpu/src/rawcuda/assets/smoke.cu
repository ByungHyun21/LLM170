// llm170 rawcuda 스모크 커널 (plans/124 2026-10-04).
// 계약 소스는 이 .cu — 빌드 자산(smoke.fatbin)은 scripts/build_cuda.bat가
// nvcc -fatbin 으로 같은 디렉터리에 생성·커밋한다(rawhip co/*.co 미러:
// 소스와 자산을 함께 커밋, 커널 산술 변경은 이 파일부터).
//
// 산출 검증: out[i] = in[i]*scale + i. 프로브(exl3_cuda_probe)는
// in[i]=i*0.5, scale=4.0 을 투입해 out[i]==i*2+i==3i 의 비트동일(f32 정확표현)
// 값을 요구한다 — 스모크는 배관(모듈로드·런치·복사) 검증이므로 근사 허용 없음.
extern "C" __global__ void llm170_smoke_add(const float* in, float* out, int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * scale + (float)i;
}
