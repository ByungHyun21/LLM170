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
extern "C" __global__ void exl3_gdn_conv(
    const float* __restrict__ qkv,   // [T][10240]
    const float* __restrict__ convw, // [10240][4]×층
    float* __restrict__ ring,        // [L][3][10240]
    float* __restrict__ q_out,       // [T][2048]
    float* __restrict__ k_out,       // [T][2048]
    float* __restrict__ v_out,       // [T][6144]
    int t_len, int layer)
{
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= 10240) return;
    float w0 = convw[layer * 10240 * 4 + ch * 4 + 0];
    float w1 = convw[layer * 10240 * 4 + ch * 4 + 1];
    float w2 = convw[layer * 10240 * 4 + ch * 4 + 2];
    float w3 = convw[layer * 10240 * 4 + ch * 4 + 3];
    float h0 = ring[layer * 3 * 10240 + 0 * 10240 + ch];
    float h1 = ring[layer * 3 * 10240 + 1 * 10240 + ch];
    float h2 = ring[layer * 3 * 10240 + 2 * 10240 + ch];
    for (int t = 0; t < t_len; t++) {
        float xt = qkv[t * 10240 + ch];
        float o = xt / (1.0f + __expf(-xt)) * 0.0f + 0.0f; // 자리표시 — 아래 정정
        o = (w3 * xt + w0 * h0 + w1 * h1 + w2 * h2);
        o = o / (1.0f + __expf(-o));
        if (ch < 2048) {
            q_out[t * 2048 + ch] = o;
        } else if (ch < 4096) {
            k_out[t * 2048 + (ch - 2048)] = o;
        } else {
            v_out[t * 6144 + (ch - 4096)] = o;
        }
        h0 = h1; h1 = h2; h2 = xt;
    }
    ring[layer * 3 * 10240 + 0 * 10240 + ch] = h0;
    ring[layer * 3 * 10240 + 1 * 10240 + ch] = h1;
    ring[layer * 3 * 10240 + 2 * 10240 + ch] = h2;
}

// exl3_gdn_gate hip 직이식 — rms(o_lc)·nw·silu(z) → gated(HF), 역순열 포함.
extern "C" __global__ void exl3_gdn_gate(
    const float* __restrict__ o_lc,  // [T][6144]
    const float* __restrict__ z,     // [T][6144] (HF)
    const float* __restrict__ nw,    // [128]×층
    float* __restrict__ gated,       // [T][6144] (HF)
    int t_len, int layer)
{
    __shared__ float red[128];
    int t = blockIdx.y;
    int h = blockIdx.x;   // HF 헤드 0..47
    int tid = threadIdx.x;
    if (t >= t_len) return;
    int p_inv = (h % 3) * 16 + h / 3;
    float ov = o_lc[t * 6144 + p_inv * 128 + tid];
    red[tid] = ov * ov;
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    float inv = 1.0f / sqrtf(red[0] / 128.0f + 1e-6f);
    float zv = z[t * 6144 + h * 128 + tid];
    gated[t * 6144 + h * 128 + tid] =
        ov * inv * nw[layer * 128 + tid] * (zv / (1.0f + __expf(-zv)));
}
// 마커 cg1

// exl3_gdn_l2perm hip 직이식 — a/b 도트(xn·ab) + q/k L2 + v·beta|g lc 순열.
// 그리드 (48, T), WG=128.
extern "C" __global__ void exl3_gdn_l2perm(
    const float* __restrict__ q_in,   // [T][2048]
    const float* __restrict__ k_in,   // [T][2048]
    const float* __restrict__ v_in,   // [T][6144] HF
    const float* __restrict__ xn,     // [T][5120] (xtb)
    const float* __restrict__ abuf,   // [L][2][48][5120]
    const float* __restrict__ alog,   // [L][48]
    const float* __restrict__ dtb,    // [L][48]
    float* __restrict__ q_out,        // [T][2048]
    float* __restrict__ k_out,        // [T][2048]
    float* __restrict__ v_out,        // [T][6144] lc
    float* __restrict__ bg,           // [T][96] beta|g lc
    int t_len, int layer)
{
    __shared__ float red[128];
    int t = blockIdx.y;
    int h = blockIdx.x;
    int tid = threadIdx.x;
    if (t >= t_len) return;
    const float eps = 1e-6f;

    float pa = 0.0f;
    for (int i = tid; i < 5120; i += 128)
        pa += xn[t * 5120 + i] * abuf[(layer * 2) * 48 * 5120 + h * 5120 + i];
    red[tid] = pa;
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    float a_val = red[0];
    __syncthreads();

    float pb = 0.0f;
    for (int i = tid; i < 5120; i += 128)
        pb += xn[t * 5120 + i] * abuf[(layer * 2 + 1) * 48 * 5120 + h * 5120 + i];
    red[tid] = pb;
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    float b_val = red[0];
    __syncthreads();

    int p_inv = (h % 3) * 16 + h / 3;
    if (tid == 0) {
        float al = alog[layer * 48 + h];
        float dt = dtb[layer * 48 + h];
        float ssm_a = -__expf(al);
        float sp = (a_val + dt > 20.0f) ? (a_val + dt) : __logf(1.0f + __expf(a_val + dt));
        bg[t * 96 + p_inv] = 1.0f / (1.0f + __expf(-b_val));
        bg[t * 96 + 48 + p_inv] = sp * ssm_a;
    }

    if (h < 16) {
        int kh = h;
        float qv = q_in[t * 2048 + kh * 128 + tid];
        red[tid] = qv * qv;
        __syncthreads();
        for (int st = 64; st > 0; st >>= 1) {
            if (tid < st) red[tid] += red[tid + st];
            __syncthreads();
        }
        float qi = 1.0f / sqrtf(red[0] + eps);
        __syncthreads();
        q_out[t * 2048 + kh * 128 + tid] = qv * qi;

        float kv = k_in[t * 2048 + kh * 128 + tid];
        red[tid] = kv * kv;
        __syncthreads();
        for (int st = 64; st > 0; st >>= 1) {
            if (tid < st) red[tid] += red[tid + st];
            __syncthreads();
        }
        float ki = 1.0f / sqrtf(red[0] + eps);
        __syncthreads();
        k_out[t * 2048 + kh * 128 + tid] = kv * ki;
    }
    v_out[t * 6144 + p_inv * 128 + tid] = v_in[t * 6144 + h * 128 + tid];
}
// 마커 l2h

// exl3_gdn_scan v3 hip 직이식 — FLA 전 LDS 상주(CS=32, TILE=16, 8패스).
// 산술/구조 vk와 동일: A/KQ/ks/qs f16, dc f32 전진대입, P6 상태 갱신.
#define EXL3_CS 32
#define EXL3_TILE 16
extern "C" __global__ void exl3_gdn_scan(
    const float* __restrict__ q,     // [T][2048]
    const float* __restrict__ k,     // [T][2048] L2
    const float* __restrict__ v,     // [T][6144] lc
    const float* __restrict__ bg,    // [T][96]
    float* __restrict__ st,          // [L][48][128*128] r/w
    float* __restrict__ outv,        // [T][6144] o_lc
    int t_len, int h_k, int h_v, int d, int layer)
{
    __shared__ __half sk[EXL3_CS * 128];
    __shared__ __half sv[EXL3_CS * 128];
    __shared__ __half A[EXL3_CS * EXL3_CS];
    __shared__ __half KQ[EXL3_CS * EXL3_CS];
    __shared__ __half KS[EXL3_CS * 128];
    __shared__ __half QS[EXL3_CS * 128];
    __shared__ float dc[EXL3_CS * 128];
    __shared__ float Stile[EXL3_TILE * 128];
    __shared__ float bp[EXL3_CS];
    __shared__ float gcs[EXL3_CS + 1];
    __shared__ float wsm[EXL3_CS];
    int h = blockIdx.x;
    int kh = h % h_k;
    int tid = threadIdx.x;
    int n_chunks = (t_len + EXL3_CS - 1) / EXL3_CS;
    float qscale = rsqrtf((float)d);
    long st_h = (long)layer * h_v * d * d + (long)h * d * d;

    for (int c = 0; c < n_chunks; c++) {
        int t0 = c * EXL3_CS;
        int n = min(t_len - t0, EXL3_CS);

        for (int e = tid; e < EXL3_CS * 128; e += 128) {
            int t = e / 128, dv = e % 128;
            bool live = t < n;
            sk[e] = __float2half_rn(live ? k[(t0 + t) * (h_k * d) + kh * 128 + dv] : 0.0f);
            sv[e] = __float2half_rn(live ? v[(t0 + t) * (h_v * d) + h * 128 + dv] : 0.0f);
        }
        if (tid < EXL3_CS) {
            float acc = 0.0f;
            for (int t = 0; t < EXL3_CS; t++) {
                acc += (t < n) ? bg[(t0 + t) * (2 * h_v) + h_v + h] : 0.0f;
                gcs[t] = acc;
            }
            gcs[EXL3_CS] = acc;
            bp[tid] = (tid < n) ? bg[(t0 + tid) * (2 * h_v) + h] : 0.0f;
        }
        __syncthreads();

        for (int i = 0; i < EXL3_CS; i++) {
            if (tid < EXL3_CS && i < n) {
                int j = tid;
                float dk = 0.0f, dq = 0.0f;
                int qbase = (t0 + i) * (h_k * d) + kh * 128;
                for (int s2 = 0; s2 < 128; s2++) {
                    float kj = __half2float(sk[j * 128 + s2]);
                    dk += __half2float(sk[i * 128 + s2]) * kj;
                    dq += q[qbase + s2] * kj;
                }
                float bi = bp[i];
                A[i * EXL3_CS + j] = __float2half_rn((j < i) ? dk * bi * expf(gcs[i] - gcs[j]) : 0.0f);
                KQ[i * EXL3_CS + j] = __float2half_rn((j <= i) ? dq * qscale * expf(gcs[i] - gcs[j]) : 0.0f);
            }
        }
        __syncthreads();

        for (int e = tid; e < EXL3_CS * 128; e += 128) {
            KS[e] = __float2half_rn(0.0f);
            QS[e] = __float2half_rn(0.0f);
        }
        __syncthreads();
        for (int pass_ = 0; pass_ < 8; pass_++) {
            int s2b = pass_ * EXL3_TILE;
            for (int s2p = 0; s2p < EXL3_TILE; s2p++)
                Stile[s2p * 128 + tid] = st[st_h + (long)(s2b + s2p) * d + tid];
            __syncthreads();
            for (int i = 0; i < EXL3_CS; i++) {
                float ak = 0.0f, aq = 0.0f;
                int qbase = (t0 + i) * (h_k * d) + kh * 128;
                for (int s2p = 0; s2p < EXL3_TILE; s2p++) {
                    float s_el = Stile[s2p * 128 + tid];
                    ak += __half2float(sk[i * 128 + s2b + s2p]) * s_el;
                    aq += q[qbase + s2b + s2p] * s_el;
                }
                int ib = i * 128 + tid;
                KS[ib] = __float2half_rn(__half2float(KS[ib]) + ak);
                QS[ib] = __float2half_rn(__half2float(QS[ib]) + aq * qscale);
            }
            __syncthreads();
        }

        for (int i = 0; i < n; i++) {
            float rhs = bp[i] * (__half2float(sv[i * 128 + tid]) - expf(gcs[i]) * __half2float(KS[i * 128 + tid]));
            for (int j = 0; j < i; j++) {
                float aij = __half2float(A[i * EXL3_CS + j]);
                if (aij != 0.0f) rhs -= aij * dc[j * 128 + tid];
            }
            dc[i * 128 + tid] = rhs;
            float oi = expf(gcs[i]) * __half2float(QS[i * 128 + tid]);
            for (int p = 0; p <= i; p++) {
                float w = __half2float(KQ[i * EXL3_CS + p]);
                if (w != 0.0f) oi += w * dc[p * 128 + tid];
            }
            outv[(t0 + i) * (h_v * d) + h * 128 + tid] = oi;
        }
        __syncthreads();

        {
            float gtot = gcs[EXL3_CS];
            float gt_exp = expf(gtot);
            if (tid < EXL3_CS) wsm[tid] = (tid < n) ? expf(gtot - gcs[tid]) : 0.0f;
            __syncthreads();
            for (int s2 = 0; s2 < 128; s2++) {
                float acc = st[st_h + (long)s2 * d + tid] * gt_exp;
                for (int j = 0; j < n; j++)
                    acc += __half2float(sk[j * 128 + s2]) * wsm[j] * dc[j * 128 + tid];
                st[st_h + (long)s2 * d + tid] = acc;
            }
            __syncthreads();
        }
    }
}
// 마커 sc1
