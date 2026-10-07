// ── EXL3 GEMV 체인 CUDA 포팅 (plans/124 G2, 2026-10-04) ──
// 산술·비트 조작식은 rawhip/kernels/src_exl3.hip의 exl3_had_in/exl3_gemv/
// exl3_had_out 1:1 직이식(원본 그대로 베낌 — 파생 금지, 2회 사고 원장).
// 차이는 헤더(roc 헤더 → cuda_fp16.h)와 이 주석뿐이다.
//
// 체인 계약(plans/124 §3.1): had_in(x·suh의 WHT + f16팩) → gemv(H도메인,
// sb는 [nseg=16][n] 분할 부분합) → had_out(nseg 합산 + WHT⁻¹·0.0884·svh).
// had_out은 정확히 1회(결함 3·15호 — 생략·이중 적용 모두 전면 붕괴).
// suh/svh는 선형별(결함 1호 — k값으로 선형 탐색 금지).
//
// [CMP 170HX(sm_80, GA100, HBM2e ~1.5TB/s) 설계 근거 — plans/124 §0]
// 이 GEMV는 메모리 본드 — 산술은 트렐리스 디코드가 ALU 병목(hip 원장
// 117GB/s, §1). 런치 기하(WG=128, 8 n-타일 레인, 8 k-타일 스테이징 창,
// nseg=16 k-분할)은 8060S에서 대역포화로 검증된 hip/vk v6 배치 그대로
// 이식했다: GA100 HBM2e 1.5TB/s에서도 동일 구조가 스트리밍 병렬성을
// 유지한다(스테이징 창 8타일 = u32 384워드 = L1 라인 협응 로드 단위).
// 개발 호스트 RTX 4070(sm_89)은 정합 검증 전용 — 이 기하로 4070 타이밍
// 튜닝을 하지 않는다(목표 기기 도착 후 실측·재조정, plans/124 §1의
// dp4a/EXL3_INT8_GEMV 후보 판정 포함).
//
// 스테이징 상한: stg[8*48] → krate ≤ 6(35B lm_head K=6이 상한 도달).
// 그리드 계약: gemv grid.x=(n/16)/8 — n은 128의 배수, k도 128의 배수
// (had_in/had_out 청크 계약과 동일 근원).
#include <cuda_fp16.h>

__device__ __forceinline__ float rtne_roundtrip(float v) {
    // f16 RTNE 왕복(had_in 규약) — half 산술은 CUDA 내장 사용.
    __half h = __float2half_rn(v);
    return __half2float(h);
}

extern "C" __global__ void exl3_had_in(
    const float* __restrict__ x,     // [T][k] f32
    const unsigned* __restrict__ suh, // [k/2] f16쌍팩
    unsigned* __restrict__ ah,       // [T][k/2] f16쌍팩
    int kchunks, int kstride)
{
    __shared__ float sm[128];
    int chunk = blockIdx.x;
    if (chunk >= kchunks) return;
    int row = blockIdx.y;
    int th = threadIdx.x;
    int ki = chunk * 128 + th;
    int ei = row * kstride + ki;

    unsigned sw = suh[ki >> 1];
    float2 sv = __half22float2(*reinterpret_cast<const __half2*>(&sw));
    float pre = rtne_roundtrip(x[ei] * (th & 1 ? sv.y : sv.x));
    sm[th] = pre;

    for (int w = 1; w < 128; w <<= 1) {
        int grp = th / w;
        int blk = (grp >> 1) * 2 * w;
        int i = blk + (th % w);
        float a = sm[i];
        float b = sm[i + w];
        __syncthreads();
        sm[i] = a + b;
        sm[i + w] = a - b;
        __syncthreads();
    }

    const float R = 0.08838834764831845f; // 1/sqrt(128)
    float o = sm[th] * R;
    __syncthreads();
    if ((th & 1) == 0) {
        __half lo = __float2half_rn(o);
        __half hi = __float2half_rn(sm[th + 1] * R);
        unsigned pack = (unsigned)*reinterpret_cast<unsigned short*>(&lo)
                      | ((unsigned)*reinterpret_cast<unsigned short*>(&hi) << 16);
        ah[ei >> 1] = pack;
    }
}

__device__ __forceinline__ float exl3_mul1_decode(unsigned w) {
    unsigned x = w * 0x83DCD12Du;
    unsigned p = (x & 0x00FF00FFu) + ((x >> 8u) & 0x00FF00FFu);
    unsigned sum = (p & 0xFFFFu) + (p >> 16u);
    return (1024.0f + (float)sum) * 0.00676727294921875f - 10.3828125f;
}

__device__ __forceinline__ unsigned exl3_extract(unsigned addr, const unsigned* stg_tile) {
    // addr = i0 | (i1<<8) | (sh<<16) — vk gemv 추출식 그대로.
    unsigned sh = addr >> 16;
    if (sh == 0) return stg_tile[(addr >> 8) & 0xFF] & 0xFFFFu;
    return ((stg_tile[(addr >> 8) & 0xFF] >> sh)
          | (stg_tile[addr & 0xFF] << (32 - sh))) & 0xFFFFu;
}

// exl3_gemv (vk v6 직이식) — WG=128, 스레드=(c=16채널 × 8 n-타일).
// f16x2 FMA 누산 + 4 k-타일 f32 폴드(참조 FOLD=4 케이던스).
// hfma2는 단일 반올림(곱+합 정확 계산 후 f16 1회 반올림) — 프로브의
// CPU 미러도 같은 단일 반올림 지점을 재현한다.
extern "C" __global__ void exl3_gemv(
    const unsigned* __restrict__ ah16,  // f16쌍팩 [k/2]
    const unsigned* __restrict__ tre,   // trellis u32
    float* __restrict__ s,              // [nseg][n]
    int ktiles, int ntiles, int K)
{
    __shared__ unsigned stg[8 * 48];
    __shared__ unsigned ahstg[8];
    int nt0 = blockIdx.x * 8;
    if (nt0 >= ntiles) return;
    int seg = blockIdx.y;
    int nseg = gridDim.y;
    int ktb = (int)((long)ktiles * seg / nseg);
    int kte = (int)((long)ktiles * (seg + 1) / nseg);
    int c = threadIdx.x & 15;
    int ntl = threadIdx.x >> 4;
    int tid = threadIdx.x;
    int words32 = 8 * K;
    int stg_words = 8 * words32;
    float acc = 0.0f;
    __half2 acc2 = __float2half2_rn(0.0f);

    unsigned addr16[16];
    #pragma unroll
    for (int r = 0; r < 16; r++) {
        int t = (4 * (c & 7) + ((r & 7) >> 1)) * 8
              + ((r >> 3) * 2 + (r & 1) + ((c >> 3) * 4));
        unsigned b0 = (unsigned)t * (unsigned)K + (unsigned)(K + 256 * K - 16);
        unsigned b1 = b0 + 16u;
        unsigned i0 = (b0 >> 5) % (unsigned)words32;
        unsigned i1 = ((b1 - 1u) >> 5) % (unsigned)words32;
        unsigned sh = (((b1 - 1u) >> 5) + 1u) * 32u - b1;
        addr16[r] = i0 | (i1 << 8) | (sh << 16);
    }

    for (int kb = ktb; kb < kte; kb += 8) {
        int nk = min(8, kte - kb);
        for (int ktl = 0; ktl < nk; ktl++) {
            long tbase = ((long)(kb + ktl) * ntiles + nt0) * words32;
            int ahb = ((kb + ktl) * 16) >> 1;
            for (int w = tid; w < stg_words; w += 128) stg[w] = tre[tbase + w];
            if (tid < 8) ahstg[tid] = ah16[ahb + tid];
            __syncthreads();
            const unsigned* tile = stg + ntl * words32;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                unsigned we = exl3_extract(addr16[2 * j], tile);
                unsigned wo = exl3_extract(addr16[2 * j + 1], tile);
                __half2 w2 = __floats2half2_rn(exl3_mul1_decode(we), exl3_mul1_decode(wo));
                __half2 a2 = *reinterpret_cast<const __half2*>(&ahstg[j]);
                acc2 = __hfma2(a2, w2, acc2);
            }
            if (((kb + ktl) & 3) == 3) {
                acc += __low2float(acc2) + __high2float(acc2);
                acc2 = __float2half2_rn(0.0f);
            }
            __syncthreads();
        }
    }
    acc += __low2float(acc2) + __high2float(acc2);
    int nt = nt0 + ntl;
    if (nt < ntiles) s[(long)seg * ntiles * 16 + (long)nt * 16 + c] = acc;
}

extern "C" __global__ void exl3_had_out(
    const float* __restrict__ s,       // [nseg][n]
    const unsigned* __restrict__ svh,  // f16쌍팩 [n/2]
    float* __restrict__ y,             // [n]
    int nchunks, int nseg, int nstride)
{
    __shared__ float sm[128];
    int chunk = blockIdx.x;
    if (chunk >= nchunks) return;
    int row = blockIdx.y;
    int th = threadIdx.x;
    int base = chunk * 128;
    float v0 = 0.0f;
    for (int g = 0; g < nseg; g++)
        v0 += s[((long)row * nseg + g) * nstride + base + th];
    sm[th] = v0;
    __syncthreads();
    for (int w = 1; w < 128; w <<= 1) {
        int grp = th / w;
        int blk = (grp >> 1) * 2 * w;
        int i = blk + (th % w);
        float a = sm[i];
        float b = sm[i + w];
        __syncthreads();
        sm[i] = a + b;
        sm[i + w] = a - b;
        __syncthreads();
    }
    int j = base + th;
    float2 sv = __half22float2(*reinterpret_cast<const __half2*>(&svh[j >> 1]));
    y[(long)row * nstride + j] = sm[th] * 0.08838834764831845f * (th & 1 ? sv.y : sv.x);
}
