// CMP 170HX (GA100/sm_80) 도착 프로브 — plan-cmp-170hx.md §1 항목의 실행 정본.
//  - 속성: SM 수·L2 3종(l2CacheSize·persistingL2CacheMaxSize·accessPolicyMaxWindowSize)·클럭·버스
//  - 스로틀: FFMA/DP4A/HFMA2/PRMT/IMAD 처리율 (ILP=8 "처리율" + dep=1 "의존사슬" 2종 —
//    커뮤니티 실측의 "FMA 레이트리미터"는 의존사슬에서 드러남. gptq4.cu 디큐트 GEMV는
//    fmaf 체인 누적이라 두 변종 사이가 실제 범위)
//  - TC: f16 wmma(m16n16k16) peak — mma 클래스 경로의 상한 기준
//  - 대역: D2D·핀드 H2D/D2H (Gen2 x16 실효) · P2P(BAR1) 가시성 + 인접쌍 대역
//  - PWRBRK# 간이 판별: FFMA 지속 런 중 reps 사이 클럭·전력 샘플(nvidia-smi) — 4×급락 서명 탐색
// 빌드: nvcc -O3 -arch=sm_80 -o target/cmp_probe scripts/cmp_probe.cu
// 기록: README 벤치 표(환경·요약) / docs/cmp170hx-detail.md(상세) — 2026-10-16 프로토콜.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#define CK(x) do { cudaError_t e_=(x); if(e_){ \
    printf("CUDA ERR %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1);} } while(0)

static int NSM = 0;

static void smi_sample(const char* tag) {
    char cmd[160];
    snprintf(cmd, sizeof cmd,
        "nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu "
        "--format=csv,noheader -i 0 | sed 's/^/  [%s] /'", tag);
    if (system(cmd) != 0) printf("  [%s] smi fail\n", tag);
}

// ── 스칼라 처리율 커널: ILP=8(독립 체인, 처리율)과 dep(의존 1체인, 레이트리미터) ──
__global__ void k_ffma_ilp8(float* o, int iters) {
    float a = threadIdx.x * 1e-9f + 1.0000001f, b = 1.0000001f, f[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) f[j] = a + j;
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) f[j] = fmaf(a, b, f[j]);
    }
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) s += f[j];
    if (s == 12345.f) o[0] = s;   // 소비 강제
}
__global__ void k_ffma_dep(float* o, int iters) {
    float f = threadIdx.x * 1e-9f + 1.0f;
    const float a = 1.0000001f, b = 1.0000001f;
    for (int i = 0; i < iters; ++i) f = fmaf(a, b, f);
    if (f == 12345.f) o[0] = f;
}
__global__ void k_dp4a_ilp8(int* o, int iters) {
    int a = threadIdx.x | 1, b = 0x01010101, c[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) c[j] = j + (threadIdx.x << 8);
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) c[j] = __dp4a(a, b, c[j]);
    }
    int s = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) s ^= c[j];
    if (s == 12345) o[0] = s;
}
__global__ void k_dp4a_dep(int* o, int iters) {
    int c = threadIdx.x, a = threadIdx.x | 1, b = 0x01010101;
    for (int i = 0; i < iters; ++i) c = __dp4a(a, b, c);
    if (c == 12345) o[0] = c;
}
__global__ void k_hfma2_ilp8(half2* o, int iters) {
    __half2 h = __floats2half2_rn(1.00001f, 1.00002f);
    __half2 z = __floats2half2_rn(0.0f, 0.0f), f[8];
    const __half2 a = __floats2half2_rn(1.0009f, 1.0011f);
#pragma unroll
    for (int j = 0; j < 8; ++j) f[j] = h;
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) f[j] = __hfma2(f[j], a, z);   // ~11k iters 내 inf 도달 — inf 처리율 동일
    }
    __half2 s = z;
#pragma unroll
    for (int j = 0; j < 8; ++j) s = __hadd2(s, f[j]);
    if (__half2float(s.x) == 12345.f) o[0] = s;
}
__global__ void k_prmt_ilp8(unsigned* o, int iters) {
    unsigned a = 0x03020100u + threadIdx.x, b = 0x07060504u, c[8];
    const unsigned s = 0x3210u;
#pragma unroll
    for (int j = 0; j < 8; ++j) c[j] = a ^ (j * 0x01010101u);
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) c[j] = __byte_perm(c[j], b, s);
    }
    unsigned r = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) r ^= c[j];
    if (r == 12345u) o[0] = r;
}
__global__ void k_imad_dep(unsigned* o, int iters) {
    unsigned c = threadIdx.x + 1;
    const unsigned a = threadIdx.x * 2 + 1, b = 0x9E3779B9u;
    for (int i = 0; i < iters; ++i) c = c * a + b;   // IMAD 의존사슬
    if (c == 12345u) o[0] = c;
}

// ── f16 텐서코어 (wmma m16n16k16, f32 누산): B=0 이라 값 포화 없음 ──
__global__ void k_mma_f16(float* o, int iters) {
    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
    wmma::fill_fragment(fa, __half(1.0f));
    wmma::fill_fragment(fb, __half(0.0f));
    wmma::fill_fragment(fc, 0.0f);
    for (int i = 0; i < iters; ++i) wmma::mma_sync(fc, fa, fb, fc);
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < fc.num_elements; ++j) s += fc.x[j];
    if (s == 12345.f) o[0] = s;
}

static float* d_f32; static int* d_i32; static half2* d_h2; static unsigned* d_u32;

static double time_kernel(void (*launch)(int), int iters, int reps) {
    cudaEvent_t s, e;
    CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));
    launch(64); CK(cudaDeviceSynchronize());          // 워밍업
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        CK(cudaEventRecord(s));
        launch(iters);
        CK(cudaEventRecord(e));
        CK(cudaEventSynchronize(e));
        float ms = 0; CK(cudaEventElapsedTime(&ms, s, e));
        if (ms < best) best = ms;
    }
    CK(cudaEventDestroy(s)); CK(cudaEventDestroy(e));
    return best;                                       // ms (best-of)
}

// 런처 래퍼(그리드 고정: SM×8블록 × 256스레드)
#define L(KER, OUT) static void l_##KER(int iters){ KER<<<(unsigned)(NSM*8),256>>>(OUT, iters); }
L(k_ffma_ilp8, d_f32) L(k_ffma_dep, d_f32)
L(k_dp4a_ilp8, d_i32) L(k_dp4a_dep, d_i32)
L(k_hfma2_ilp8, d_h2) L(k_prmt_ilp8, d_u32) L(k_imad_dep, d_u32)
L(k_mma_f16, d_f32)

static void report(const char* name, double ms, double ops, const char* unit) {
    double rate = ops / (ms * 1e-3);
    printf("%-14s %12.3f G%s/s   (%.2f ms)\n", name, rate / 1e9, unit, ms);
}

int main() {
    CK(cudaSetDevice(0));
    int dev = 0; CK(cudaGetDevice(&dev));

    // ── 속성 (plan-cmp §1: L2 3종 포함) ──
    struct { cudaDeviceAttr a; const char* n; } attrs[] = {
        {cudaDevAttrMultiProcessorCount, "SM count"},
        {cudaDevAttrL2CacheSize, "L2 size [B]"},
        {cudaDevAttrMaxPersistingL2CacheSize, "persistingL2Max [B]"},
        {cudaDevAttrMaxAccessPolicyWindowSize, "accessPolicyMaxWindow [B]"},
        {cudaDevAttrClockRate, "SM clock max [kHz]"},
        {cudaDevAttrMemoryClockRate, "mem clock [kHz]"},
        {cudaDevAttrGlobalMemoryBusWidth, "bus width [b]"},
        {cudaDevAttrMaxSharedMemoryPerBlockOptin, "smem/block optin [B]"},
        {cudaDevAttrMaxRegistersPerBlock, "regs/block"},
        {cudaDevAttrComputeCapabilityMajor, "cc major"},
        {cudaDevAttrComputeCapabilityMinor, "cc minor"},
    };
    printf("== attributes (device 0) ==\n");
    for (auto& r : attrs) { int v = 0; CK(cudaDeviceGetAttribute(&v, r.a, dev));
        printf("%-26s %d\n", r.n, v); }
    CK(cudaDeviceGetAttribute(&NSM, cudaDevAttrMultiProcessorCount, dev));
    const double threads_total = (double)NSM * 8 * 256;

    CK(cudaMalloc(&d_f32, 4)); CK(cudaMalloc(&d_i32, 4));
    CK(cudaMalloc(&d_h2, 4));  CK(cudaMalloc(&d_u32, 4));

    printf("\n== scalar/TC throughput (grid SMx8 x 256thr, best of 3; "
           "ilp8=처리율, dep=의존사슬) ==\n");
    smi_sample("pre");
    double ms;
    ms = time_kernel(l_k_ffma_ilp8, 50000, 3);
    report("ffma_ilp8", ms, threads_total * 50000 * 8 * 2, "flop(FMA)");
    ms = time_kernel(l_k_ffma_dep, 40000, 3);
    report("ffma_dep", ms, threads_total * 40000 * 2, "flop(FMA)");
    ms = time_kernel(l_k_dp4a_ilp8, 50000, 3);
    report("dp4a_ilp8", ms, threads_total * 50000 * 8, "instr");
    ms = time_kernel(l_k_dp4a_dep, 40000, 3);
    report("dp4a_dep", ms, threads_total * 40000, "instr");
    ms = time_kernel(l_k_hfma2_ilp8, 50000, 3);
    report("hfma2_ilp8", ms, threads_total * 50000 * 8, "instr");
    ms = time_kernel(l_k_prmt_ilp8, 50000, 3);
    report("prmt_ilp8", ms, threads_total * 50000 * 8, "instr");
    ms = time_kernel(l_k_imad_dep, 40000, 3);
    report("imad_dep", ms, threads_total * 40000, "instr");
    const double warps = (double)NSM * 8 * 8;          // SM×8블록×8워프
    ms = time_kernel(l_k_mma_f16, 400, 3);
    report("mma_f16_wmma", ms, warps * 400 * 8192, "flop");
    smi_sample("post-bench");

    // ── PWRBRK 간이: FFMA 지속 런(수백 ms/rep) — 클럭·전력 샘플로 4×급락 탐색 ──
    printf("\n== sustained ffma (throttle/PWRBRK signature) ==\n");
    cudaEvent_t s, e; CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));
    for (int r = 0; r < 4; ++r) {
        smi_sample("sus-start");
        CK(cudaEventRecord(s));
        l_k_ffma_ilp8(120000);
        CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
        float ms2 = 0; CK(cudaEventElapsedTime(&ms2, s, e));
        report("ffma_sus", ms2, threads_total * 120000 * 8 * 2, "flop(FMA)");
        smi_sample("sus-end");
    }

    // ── 복사 대역 ──
    printf("\n== copy bandwidth (device 0, best of 3) ==\n");
    const size_t GB = 1ull << 30, MB = 256ull << 20;
    char *a, *b; CK(cudaMalloc(&a, GB)); CK(cudaMalloc(&b, GB));
    CK(cudaMemset(a, 1, GB));
    float mmin = 1e30f;
    for (int r = 0; r < 3; ++r) { CK(cudaEventRecord(s));
        CK(cudaMemcpy(b, a, GB, cudaMemcpyDeviceToDevice)); CK(cudaEventRecord(e));
        CK(cudaEventSynchronize(e)); float m; CK(cudaEventElapsedTime(&m, s, e)); if (m < mmin) mmin = m; }
    printf("%-14s %12.1f GB/s (raw)  x2=%.1f  (%.0f ms)\n", "d2d_1G",
           GB / (mmin * 1e-3) / 1e9, 2 * GB / (mmin * 1e-3) / 1e9, mmin);
    void* hp; CK(cudaHostAlloc(&hp, MB, cudaHostAllocDefault));
    mmin = 1e30f;
    for (int r = 0; r < 3; ++r) { CK(cudaEventRecord(s));
        CK(cudaMemcpy(a, hp, MB, cudaMemcpyHostToDevice)); CK(cudaEventRecord(e));
        CK(cudaEventSynchronize(e)); float m; CK(cudaEventElapsedTime(&m, s, e)); if (m < mmin) mmin = m; }
    printf("%-14s %12.1f GB/s  (%.2f ms)\n", "h2d_pin_256M", MB / (mmin * 1e-3) / 1e9, mmin);
    mmin = 1e30f;
    for (int r = 0; r < 3; ++r) { CK(cudaEventRecord(s));
        CK(cudaMemcpy(hp, a, MB, cudaMemcpyDeviceToHost)); CK(cudaEventRecord(e));
        CK(cudaEventSynchronize(e)); float m; CK(cudaEventElapsedTime(&m, s, e)); if (m < mmin) mmin = m; }
    printf("%-14s %12.1f GB/s  (%.2f ms)\n", "d2h_pin_256M", MB / (mmin * 1e-3) / 1e9, mmin);
    cudaFreeHost(hp); cudaFree(a); cudaFree(b);

    // ── P2P(BAR1): 가시성 행렬 + 쌍 대역 ──
    int nd = 0; CK(cudaGetDeviceCount(&nd));
    printf("\n== P2P canAccessPeer matrix (row=from; 1=BAR1 P2P 가능) ==\n");
    printf("     "); for (int j = 0; j < nd; ++j) printf("%6d", j); printf("\n");
    for (int i = 0; i < nd; ++i) {
        printf("%4d ", i);
        for (int j = 0; j < nd; ++j) {
            if (i == j) { printf("%6s", "-"); continue; }
            int can = 0; CK(cudaSetDevice(i));
            cudaError_t pe = cudaDeviceCanAccessPeer(&can, i, j);
            printf("%6d", (pe == cudaSuccess && can) ? 1 : 0);
        }
        printf("\n");
    }
    int pairs[][2] = {{0,1},{1,0},{1,2},{2,1},{2,3},{3,2},{0,3},{3,0}};
    const size_t PMB = 256ull << 20;
    char *src, *dst;
    printf("\n== P2P peer copy 256M (BAR1; fail = 미지원) ==\n");
    for (auto& pr : pairs) {
        int i = pr[0], j = pr[1];
        CK(cudaSetDevice(i)); CK(cudaMalloc(&src, PMB)); CK(cudaMemset(src, 2, PMB));
        CK(cudaSetDevice(j)); CK(cudaMalloc(&dst, PMB));
        CK(cudaSetDevice(i));
        cudaError_t ee = cudaDeviceEnablePeerAccess(j, 0);
        if (ee != cudaSuccess && ee != cudaErrorPeerAccessAlreadyEnabled) {
            printf("p2p %d->%d  enable fail: %s\n", i, j, cudaGetErrorString(ee));
            cudaGetLastError(); cudaFree(src); CK(cudaSetDevice(j)); cudaFree(dst); continue;
        }
        mmin = 1e30f;
        for (int r = 0; r < 3; ++r) { CK(cudaEventRecord(s));
            ee = cudaMemcpyPeer(dst, j, src, i, PMB);
            CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
            if (ee != cudaSuccess) break;
            float m; CK(cudaEventElapsedTime(&m, s, e)); if (m < mmin) mmin = m; }
        if (ee == cudaSuccess)
            printf("p2p %d->%d  %10.1f GB/s  (%.2f ms)\n", i, j, PMB / (mmin * 1e-3) / 1e9, mmin);
        else
            printf("p2p %d->%d  copy fail: %s\n", i, j, cudaGetErrorString(ee));
        cudaGetLastError();
        cudaFree(src); CK(cudaSetDevice(j)); cudaFree(dst);
    }
    CK(cudaEventDestroy(s)); CK(cudaEventDestroy(e));
    printf("\ndone.\n");
    return 0;
}
