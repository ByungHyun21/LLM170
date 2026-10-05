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
    // [판정 정정 plans/132] gfx1151은 v_dot4_i32_i8가 실존(런타임 기대값
    // 4/4 실측 — __dp4a/빌트인 거부는 컴파일러 게이팅이었음, 하드웨어 부재 아님).
    // S=Σbytes(x)은 등가중 바이트합이라 dot4로 정확 치환:
    // r = dot4(x^0x80808080, 0x01010101) = Σ(x_i-128) = S-512
    // → value = (r+1536)*c1 - c2. 정수·부동 소수점 연산 순서 모두 종전과
    // 동일(비트 불변). 종전 마스크 춤(~6연산)을 mul+xor+dot4 3연산으로.
    unsigned x = w * 0x83DCD12Du;
    int r;
    int one = 0x01010101;
    asm("v_dot4_i32_i8 %0, %1, %2, %3"
        : "=v"(r)
        : "v"(x ^ 0x80808080u), "v"(one), "v"(0));
    return ((float)(r + 1536)) * 0.00676727294921875f - 10.3828125f;
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
    // plans/130: 이중 버퍼 스테이징 — 다음 k-타일 tre 로드를 현재 연산과 중첩.
    // 커널은 블록 처리량(~4.5K blocks/ms)으로 묶여 있었고(중형-n 63GB/s vs
    // 대형-n 118GB/s — 블록당 시간 동일·시간 ∝ 블록 수 실측), 벽은 k-체인의
    // load→compute 직렬화. 산술(행별 환원 순서)은 불변.
    __shared__ __align__(16) unsigned stg[2][8 * 48];
    __shared__ unsigned ahstg[2][8];
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

    {
        long tbase = ((long)ktb * ntiles + nt0) * words32;
        int ahb = (ktb * 16) >> 1;
        for (int w4 = tid * 4; w4 < stg_words; w4 += 128 * 4)
            *reinterpret_cast<uint4*>(&stg[0][w4]) =
                *reinterpret_cast<const uint4*>(tre + tbase + w4);
        if (tid < 8) ahstg[0][tid] = ah16[ahb + tid];
        __syncthreads();
    }
    for (int i = 0, n_tile = kte - ktb; i < n_tile; i++) {
        int cur = i & 1;
        int kt = ktb + i;
        if (i + 1 < n_tile) {
            long tbase = ((long)(kt + 1) * ntiles + nt0) * words32;
            int ahb = ((kt + 1) * 16) >> 1;
            for (int w4 = tid * 4; w4 < stg_words; w4 += 128 * 4)
                *reinterpret_cast<uint4*>(&stg[cur ^ 1][w4]) =
                    *reinterpret_cast<const uint4*>(tre + tbase + w4);
            if (tid < 8) ahstg[cur ^ 1][tid] = ah16[ahb + tid];
        }
        const unsigned* tile = stg[cur] + ntl * words32;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            unsigned we = exl3_extract(addr16[2 * j], tile);
            unsigned wo = exl3_extract(addr16[2 * j + 1], tile);
            __half2 w2 = __floats2half2_rn(exl3_mul1_decode(we), exl3_mul1_decode(wo));
            __half2 a2 = *reinterpret_cast<__half2*>(&ahstg[cur][j]);
            acc2 = __hfma2(a2, w2, acc2);
        }
        if ((kt & 3) == 3) {
            acc += __low2float(acc2) + __high2float(acc2);
            acc2 = __float2half2_rn(0.0f);
        }
        __syncthreads();
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

// exl3_gemv_m (plans/130) -- m-row batched GEMV (m=T<=8). Per-row arithmetic is
// bit-identical to exl3_gemv (same addr16, decode, hfma2 order, FOLD=4 cadence).
// TM is a compile-time template param: acc arrays are sized exactly and every
// row loop is fully unrolled -- a runtime-T variant kept acc[8]/acc2[8] alive
// for all T and halved occupancy (T=1 whole-model 416ms vs kseg 286ms; measured
// plans/130, reverted). Partials go to s[T][nseg][n] (dbat), summed by had_out.
template <int TM>
__device__ __forceinline__ void gemv_m_core(
    const unsigned* __restrict__ ah16,
    const unsigned* __restrict__ tre,
    float* __restrict__ s,
    int ktiles, int ntiles, int K)
{
    __shared__ __align__(16) unsigned stg[2][8 * 48];
    __shared__ unsigned ahstg[2][8 * 8];
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
    int n = ntiles * 16;
    int arow_words = ktiles * 8;
    float acc[TM];
    __half2 acc2[TM];
    #pragma unroll
    for (int t = 0; t < TM; t++) { acc2[t] = __float2half2_rn(0.0f); acc[t] = 0.0f; }

    unsigned addr16[16];
    #pragma unroll
    for (int r = 0; r < 16; r++) {
        int tt = (4 * (c & 7) + ((r & 7) >> 1)) * 8
              + ((r >> 3) * 2 + (r & 1) + ((c >> 3) * 4));
        unsigned b0 = (unsigned)tt * (unsigned)K + (unsigned)(K + 256 * K - 16);
        unsigned b1 = b0 + 16u;
        unsigned i0 = (b0 >> 5) % (unsigned)words32;
        unsigned i1 = ((b1 - 1u) >> 5) % (unsigned)words32;
        unsigned sh = (((b1 - 1u) >> 5) + 1u) * 32u - b1;
        addr16[r] = i0 | (i1 << 8) | (sh << 16);
    }

    {
        long tbase = ((long)ktb * ntiles + nt0) * words32;
        int ahb = (ktb * 16) >> 1;
        for (int w4 = tid * 4; w4 < stg_words; w4 += 128 * 4)
            *reinterpret_cast<uint4*>(&stg[0][w4]) =
                *reinterpret_cast<const uint4*>(tre + tbase + w4);
        #pragma unroll
        for (int t = 0; t < TM; t++)
            if (tid < 8)
                ahstg[0][t * 8 + tid] = ah16[(long)t * arow_words + ahb + tid];
        __syncthreads();
    }
    for (int i = 0, n_tile = kte - ktb; i < n_tile; i++) {
        int cur = i & 1;
        int kt = ktb + i;
        if (i + 1 < n_tile) {
            long tbase = ((long)(kt + 1) * ntiles + nt0) * words32;
            int ahb = ((kt + 1) * 16) >> 1;
            for (int w4 = tid * 4; w4 < stg_words; w4 += 128 * 4)
                *reinterpret_cast<uint4*>(&stg[cur ^ 1][w4]) =
                    *reinterpret_cast<const uint4*>(tre + tbase + w4);
            #pragma unroll
            for (int t = 0; t < TM; t++)
                if (tid < 8)
                    ahstg[cur ^ 1][t * 8 + tid] = ah16[(long)t * arow_words + ahb + tid];
        }
        const unsigned* tile = stg[cur] + ntl * words32;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            unsigned we = exl3_extract(addr16[2 * j], tile);
            unsigned wo = exl3_extract(addr16[2 * j + 1], tile);
            __half2 w2 = __floats2half2_rn(exl3_mul1_decode(we), exl3_mul1_decode(wo));
            #pragma unroll
            for (int t = 0; t < TM; t++) {
                __half2 a2 = *reinterpret_cast<__half2*>(&ahstg[cur][t * 8 + j]);
                acc2[t] = __hfma2(a2, w2, acc2[t]);
            }
        }
        if ((kt & 3) == 3) {
            #pragma unroll
            for (int t = 0; t < TM; t++) {
                acc[t] += __low2float(acc2[t]) + __high2float(acc2[t]);
                acc2[t] = __float2half2_rn(0.0f);
            }
        }
        __syncthreads();
    }
    int nt = nt0 + ntl;
    if (nt >= ntiles) return;
    #pragma unroll
    for (int t = 0; t < TM; t++) {
        acc[t] += __low2float(acc2[t]) + __high2float(acc2[t]);
        s[((long)t * nseg + seg) * n + (long)nt * 16 + c] = acc[t];
    }
}

#define GVM_ARGS const unsigned* __restrict__ ah16, const unsigned* __restrict__ tre, float* __restrict__ s, int ktiles, int ntiles, int K, int T
#define GVM_BODY(TT) gemv_m_core<TT>(ah16, tre, s, ktiles, ntiles, K)
#define GVM_WRAP(TT) extern "C" __global__ void exl3_gemv_m##TT(GVM_ARGS) { GVM_BODY(TT); }
GVM_WRAP(1)
GVM_WRAP(2)
GVM_WRAP(3)
GVM_WRAP(4)
GVM_WRAP(5)
GVM_WRAP(6)
GVM_WRAP(7)
GVM_WRAP(8)
