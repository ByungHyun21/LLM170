// ─── EXL3 hip 포팅(plans/121 CMP 인도물) ───
// vk(rawvk/spv/exl3_had_in_tf32.comp)의 산술 그대로: WHT128 입력 스테이지.
// pre = f16(x·suh) → f32 버터플라이 → /√128 → f16 쌍팩 ah16[row][k/2].
// f16 변환은 __float2half_rn(RTNE — Rust half::from_f32 미러와 동일 규약).

__device__ __forceinline__ float rtne_roundtrip(float v) {
    // f16 RTNE 왕복(had_in 규약) — half 산술은 HIP 내장 사용.
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
                __half2 a2 = *reinterpret_cast<__half2*>(&ahstg[j]);
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

// e3_norm_resid의 hip 직이식 — xn = rms_norm(x+ab)·w (xtb), x = x+ab (제자리).
