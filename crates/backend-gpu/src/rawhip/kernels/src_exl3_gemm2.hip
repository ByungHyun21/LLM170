extern "C" __global__ void exl3_norm_resid(
    float* __restrict__ x,        // [T][5120] r/w
    const float* __restrict__ nw, // [5120]
    const float* __restrict__ ab, // [T][5120]
    float* __restrict__ xn,       // [T][5120]
    int t_len, int w_off)
{
    __shared__ float red[1024];
    int t = blockIdx.x;
    int tid = threadIdx.x;
    if (t >= t_len) return;
    int base = t * 5120;
    float v0 = x[base + tid] + ab[base + tid];
    float v1 = x[base + 1024 + tid] + ab[base + 1024 + tid];
    float v2 = x[base + 2048 + tid] + ab[base + 2048 + tid];
    float v3 = x[base + 3072 + tid] + ab[base + 3072 + tid];
    float v4 = x[base + 4096 + tid] + ab[base + 4096 + tid];
    float ss = v0*v0 + v1*v1 + v2*v2 + v3*v3 + v4*v4;
    red[tid] = ss;
    __syncthreads();
    for (int st = 512; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    // 정밀 sqrt 계약(vk 원본 주석): rsq 근사는 drift 씨앗.
    float inv = 1.0f / sqrtf(red[0] / 5120.0f + 1e-6f);
    xn[base + tid] = v0 * inv * nw[w_off + tid];
    xn[base + 1024 + tid] = v1 * inv * nw[w_off + 1024 + tid];
    xn[base + 2048 + tid] = v2 * inv * nw[w_off + 2048 + tid];
    xn[base + 3072 + tid] = v3 * inv * nw[w_off + 3072 + tid];
    xn[base + 4096 + tid] = v4 * inv * nw[w_off + 4096 + tid];
    x[base + tid] = v0;
    x[base + 1024 + tid] = v1;
    x[base + 2048 + tid] = v2;
    x[base + 3072 + tid] = v3;
    x[base + 4096 + tid] = v4;
}

__device__ __forceinline__ unsigned exl3_tile_word(
    const unsigned* tre_s, int words32, int K, int tidx, int r, int c) {
    int t_ = (4 * (c & 7) + ((r & 7) >> 1)) * 8
           + ((r >> 3) * 2 + (r & 1) + ((c >> 3) * 4));
    unsigned b0 = (unsigned)t_ * (unsigned)K + (unsigned)(K + 256 * K - 16);
    unsigned b1 = b0 + 16u;
    unsigned i0 = (b0 >> 5) % (unsigned)words32;
    unsigned i1 = ((b1 - 1u) >> 5) % (unsigned)words32;
    unsigned sh = (((b1 - 1u) >> 5) + 1u) * 32u - b1;
    unsigned w0 = tre_s[tidx * words32 + i0];
    unsigned w1 = tre_s[tidx * words32 + i1];
    if (sh == 0u) return w1 & 0xFFFFu;
    return ((w0 << (32u - sh)) | (w1 >> sh)) & 0xFFFFu;
}

// exl3_gemm2 hip 이식(모듈 4/4) — vk coopmat gemm2 골격 직이식:
// 트렐리스 스테이징·A/B 타일·드레인 인덱스 동일, MMA만 coopmat→f32 누산
// 스칼라 타일(스레드당 1 n-행×64 t-열, 레지스터 A 캐시). 그리드 (n/64, ceil(T/128)).
extern "C" __global__ void exl3_gemm2(
    const unsigned* __restrict__ ah16, // f16쌍팩 [T][k/2]
    const unsigned* __restrict__ tre,
    float* __restrict__ s,             // f32 [T][n]
    int ktiles, int ntiles, int K, int T)
{
    __shared__ unsigned tre_s[8 * 48];
    __shared__ __half2 buf_a[64 * 18];
    __shared__ __half2 buf_b[128 * 18];
    const int STRIDE = 18;
    int wg_n = blockIdx.x;
    int wg_t = blockIdx.y;
    int tid = threadIdx.x;
    int n = ntiles * 16;
    int row0 = wg_n * 64;
    int t0 = wg_t * 128;
    int nrow = min(64, n - row0);
    int nt = min(128, T - t0);
    int words32 = 8 * K;
    int nkb = ktiles >> 1;
    int arow_words = ktiles * 8;
    int n_loc = tid & 63;
    int t_half = tid >> 6;
    float acc[64];
    #pragma unroll
    for (int i = 0; i < 64; i++) acc[i] = 0.0f;

    for (int kb = 0; kb < nkb; kb++) {
        {
            int stg_words = 8 * words32;
            for (int w = tid; w < stg_words; w += 128) {
                int tidx = w / words32;
                int wq = w % words32;
                int ktl = tidx >> 2;
                int ntl = tidx & 3;
                tre_s[w] = tre[((long)(kb * 2 + ktl) * ntiles + (wg_n * 4 + ntl)) * words32 + wq];
            }
            __syncthreads();
        }
        // A: 64행×32k 쌍팩 — vk 인덱스식.
        {
            int e0 = tid * 16;
            int nl = e0 / 32;
            int k0 = e0 % 32;
            int c = nl & 15;
            int nt_hi = nl >> 4;
            #pragma unroll
            for (int q = 0; q < 8; q++) {
                int kA = k0 + 2 * q;
                int kB = kA + 1;
                float va = 0.0f, vb = 0.0f;
                if (nl < nrow) {
                    va = exl3_mul1_decode(exl3_tile_word(tre_s, words32, K, (kA >> 4) * 4 + nt_hi, kA & 15, c));
                    vb = exl3_mul1_decode(exl3_tile_word(tre_s, words32, K, (kB >> 4) * 4 + nt_hi, kB & 15, c));
                }
                buf_a[nl * STRIDE + (k0 >> 1) + q] = __floats2half2_rn(va, vb);
            }
        }
        __syncthreads();
        // B: 128t×32k — vk식 2패스×8쌍.
        {
            for (int p = 0; p < 2; p++) {
                int e0 = (tid + p * 128) * 16;
                int tl = e0 / 32;
                int k0 = e0 % 32;
                #pragma unroll
                for (int q = 0; q < 8; q++) {
                    unsigned w = 0u;
                    if (tl < nt)
                        w = ah16[(long)(t0 + tl) * arow_words + (kb * 32 + k0) / 2 + q];
                    buf_b[tl * STRIDE + (k0 >> 1) + q] = *reinterpret_cast<__half2*>(&w);
                }
            }
            __syncthreads();
        }
        {
            __half2 ar2[16];
            #pragma unroll
            for (int q = 0; q < 16; q++) ar2[q] = buf_a[n_loc * STRIDE + q];
            for (int t = 0; t < 64; t++) {
                int tt = t_half * 64 + t;
                if (tt >= nt) break;
                float sum = acc[t];
                #pragma unroll
                for (int q = 0; q < 16; q++) {
                    float2 bv = __half22float2(buf_b[tt * STRIDE + q]);
                    float2 av = __half22float2(ar2[q]);
                    sum += av.x * bv.x + av.y * bv.y;
                }
                acc[t] = sum;
            }
        }
        __syncthreads();
    }
    for (int t = 0; t < 64; t++) {
        int tt = t_half * 64 + t;
        if (tt >= nt) break;
        if (n_loc < nrow)
            s[(long)(t0 + tt) * n + row0 + n_loc] = acc[t];
    }
}
// 마커 g2h

// exl3_gdn_conv hip 직이식 — vk와 동일 산술/레이아웃(링 [n][3][10240] r/w).
