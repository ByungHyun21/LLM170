#include <cuda_fp16.h>
#include "cast_common.cuh"

__device__ __forceinline__ double attn_exp_d(double x)
{
    if (x < -708.5) {
        return 0.0;
    }
    if (x > 709.78) {
        return __longlong_as_double(0x7ff0000000000000LL);
    }
    const double invln2 = 1.4426950408889634;
    const double ln2_hi = 6.9314718036912382e-01;
    const double ln2_lo = 1.9082149292705877e-10;
    int k = (int)floor(x * invln2 + 0.5);
    double r = x - (double)k * ln2_hi;
    r = r - (double)k * ln2_lo;
    double p = 1.0 + r * (1.0 + r * (0.5 + r * (0.16666666666666666
        + r * (0.041666666666666664 + r * (0.008333333333333333
        + r * (0.001388888888888889 + r * 0.0001984126984126984))))));
    double scale = __longlong_as_double((long long)(1023 + k) << 52);
    return p * scale;
}

__device__ __forceinline__ float attn_expf(float x)
{
    return __expf(x);
}

__device__ __forceinline__ float attn_theta(int tid)
{
    double e = -(2.0 * (double)tid) / 64.0;
    double th = attn_exp_d(16.11809565095832 * e);
    return (float)th;
}

__device__ __forceinline__ float attn_theta_p(int p, int denom, float base)
{
    double e = -(2.0 * (double)p) / (double)denom;
    double lnb = (denom == 64 && base == 1e7f) ? 16.11809565095832 : log((double)base);
    return (float)attn_exp_d(lnb * e);
}

__device__ __forceinline__ void attn_sincos_d(double a, double* co, double* si)
{
    const double invpio2 = 6.36619772367581342433e-01;
    const double pio2_1  = 1.57079632673412561417e+00;
    const double pio2_1t = 6.07710050650619224932e-11;
    double kd = floor(a * invpio2 + 0.5);
    long long k = (long long)kd;
    double r = a - (double)k * pio2_1;
    r = r - (double)k * pio2_1t;
    double z = r * r;
    double st = r * (1.0 - z * (0.16666666666666666
        - z * (0.008333333333333333
        - z * (1.9841269841269841e-04
        - z * (2.7557319223985893e-06
        - z * (2.5052108385441720e-08 - z * 1.6059043836821613e-10))))));
    double ct = 1.0 - z * (0.5
        - z * (0.041666666666666664
        - z * (0.0013888888888888889
        - z * (2.4801587301587302e-05
        - z * (2.7557319223985893e-07 - z * 2.0876756987868100e-09)))));
    int n = (int)(k & 3LL);
    if (n == 0) { *co = ct; *si = st; }
    else if (n == 1) { *co = -st; *si = ct; }
    else if (n == 2) { *co = -ct; *si = -st; }
    else { *co = st; *si = -ct; }
}

__device__ __forceinline__ float attn_cosf(float a)
{
    double c, s;
    attn_sincos_d((double)a, &c, &s);
    return (float)c;
}

__device__ __forceinline__ float attn_sinf(float a)
{
    double c, s;
    attn_sincos_d((double)a, &c, &s);
    return (float)s;
}


struct AttnKvRowPtrs {
    signed char* k;
    signed char* v;
    float* ks;
    float* vs;
};
__device__ __forceinline__ AttnKvRowPtrs attn_blk_row(
    const unsigned long long* __restrict__ tab, int layer, int cap, int row,
    long kv_dim, int kv_heads, int kh)
{
    const unsigned long long* bt =
        tab + ((long)layer * ((cap + 511) >> 9) + (row >> 9)) * 4;
    const unsigned long long ro = (unsigned long long)(row & 511);
    const long hstride = (kv_heads > 0) ? (kv_dim / kv_heads) : 256;
    AttnKvRowPtrs p;
    p.k = reinterpret_cast<signed char*>(bt[0]) +
          ro * (unsigned long long)kv_dim + (long)kh * hstride;
    p.v = reinterpret_cast<signed char*>(bt[1]) +
          ro * (unsigned long long)kv_dim + (long)kh * hstride;
    p.ks = reinterpret_cast<float*>(
        bt[2] + (ro * (unsigned long long)kv_heads + (unsigned long long)kh) * 4ULL);
    p.vs = reinterpret_cast<float*>(
        bt[3] + (ro * (unsigned long long)kv_heads + (unsigned long long)kh) * 4ULL);
    return p;
}

extern "C" __global__ void attn_prep_q(
    const float* __restrict__ qg,
    const float* __restrict__ kin,
    const float* __restrict__ vin,
    const float* __restrict__ qnw,
    const float* __restrict__ knw,
    float* __restrict__ qh,
    const unsigned long long* __restrict__ dblk_tab,
    const unsigned* __restrict__ pp,
    const unsigned long long* __restrict__ row_blk,
    const unsigned long long* __restrict__ row_pp,
    int t_len, int layer, int q_heads, int kv_heads, int cap)
{
    __shared__ float red[128];
    __shared__ float hd[256];
    int t = blockIdx.x;
    int j = blockIdx.y;
    int tid = threadIdx.x;
    const unsigned long long* blk = dblk_tab;
    int pos;
    if (row_blk != (const unsigned long long*)0) {
        blk = (const unsigned long long*)row_blk[t];
        pos = (int)((const unsigned*)row_pp[t])[0];
    } else {
        pos = (int)pp[0] + t;
    }
    long kv_dim = (long)kv_heads * 256;

    if (j < q_heads) {
        int src = t * (q_heads * 512) + j * 512;
        hd[tid] = qg[src + tid];
        hd[tid + 128] = qg[src + 128 + tid];
        __syncthreads();
        float ss = hd[tid] * hd[tid] + hd[tid + 128] * hd[tid + 128];
        red[tid] = ss;
        __syncthreads();
        for (int s = 64; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        float inv = 1.0f / sqrtf(red[0] / 256.0f + 1e-6f);
        hd[tid] = hd[tid] * inv * qnw[layer * 256 + tid];
        hd[tid + 128] = hd[tid + 128] * inv * qnw[layer * 256 + 128 + tid];
        __syncthreads();
        if (tid < 32) {
            float theta = attn_theta(tid);
            float ang = (float)pos * theta;
            float c = attn_cosf(ang), s2 = attn_sinf(ang);
            float x0 = hd[tid], x1 = hd[tid + 32];
            hd[tid] = x0 * c - x1 * s2;
            hd[tid + 32] = x0 * s2 + x1 * c;
        }
        __syncthreads();
        qh[(long)t * (q_heads * 256) + (long)j * 256 + tid] = hd[tid];
        qh[(long)t * (q_heads * 256) + (long)j * 256 + 128 + tid] = hd[tid + 128];
    } else {
        int m = j - q_heads;
        int src = t * (int)kv_dim + m * 256;
        hd[tid] = kin[src + tid];
        hd[tid + 128] = kin[src + 128 + tid];
        __syncthreads();
        float ss = hd[tid] * hd[tid] + hd[tid + 128] * hd[tid + 128];
        red[tid] = ss;
        __syncthreads();
        for (int s = 64; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        float inv = 1.0f / sqrtf(red[0] / 256.0f + 1e-6f);
        hd[tid] = hd[tid] * inv * knw[layer * 256 + tid];
        hd[tid + 128] = hd[tid + 128] * inv * knw[layer * 256 + 128 + tid];
        __syncthreads();
        if (tid < 32) {
            float theta = attn_theta(tid);
            float ang = (float)pos * theta;
            float c = attn_cosf(ang), s2 = attn_sinf(ang);
            float x0 = hd[tid], x1 = hd[tid + 32];
            hd[tid] = x0 * c - x1 * s2;
            hd[tid + 32] = x0 * s2 + x1 * c;
        }
        __syncthreads();
        AttnKvRowPtrs rp = attn_blk_row(blk, layer, cap, pos, kv_dim, kv_heads, m);
        red[tid] = fmaxf(fabsf(hd[tid]), fabsf(hd[tid + 128]));
        __syncthreads();
        for (int s = 64; s > 0; s >>= 1) {
            if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
            __syncthreads();
        }
        float ks = red[0] * (1.0f / 127.0f);
        if (ks == 0.0f) {
            ks = 1.0f;
        }
        float ik = 1.0f / ks;
        __syncthreads();
        int q0 = __float2int_rn(hd[tid] * ik);
        int q1 = __float2int_rn(hd[tid + 128] * ik);
        rp.k[tid] = (signed char)max(-127, min(127, q0));
        rp.k[128 + tid] = (signed char)max(-127, min(127, q1));
        if (tid == 0) {
            rp.ks[0] = ks;
        }
        float v0 = vin[src + tid];
        float v1 = vin[src + 128 + tid];
        red[tid] = fmaxf(fabsf(v0), fabsf(v1));
        __syncthreads();
        for (int s = 64; s > 0; s >>= 1) {
            if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
            __syncthreads();
        }
        float vs = red[0] * (1.0f / 127.0f);
        if (vs == 0.0f) {
            vs = 1.0f;
        }
        float iv = 1.0f / vs;
        int w0 = __float2int_rn(v0 * iv);
        int w1 = __float2int_rn(v1 * iv);
        rp.v[tid] = (signed char)max(-127, min(127, w0));
        rp.v[128 + tid] = (signed char)max(-127, min(127, w1));
        if (tid == 0) {
            rp.vs[0] = vs;
        }
    }
}

#define ATTN_TMAX 2048
#define ATTN_SPLIT_GRAIN 32
#define ATTN_CHUNK 256
template <typename KDot, typename VAcc>
__device__ __forceinline__ void attn_part_step(KDot kdot, VAcc vacc,
                                               float* __restrict__ sarr,
                                               float* __restrict__ reds, int nch,
                                               int tid, float& m_run, float& l_run,
                                               float& acc)
{
    float p = -1e30f;
    if (tid < nch) {
        p = kdot(tid);
    }
    sarr[tid] = p;
    __syncthreads();
    reds[tid] = (tid < nch) ? sarr[tid] : -1e30f;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (tid < st) reds[tid] = fmaxf(reds[tid], reds[tid + st]);
        __syncthreads();
    }
    float m_new = fmaxf(m_run, reds[0]);
    float corr = (m_run <= -1e29f) ? 0.0f : attn_expf(m_run - m_new);
    __syncthreads();
    float e = 0.0f;
    if (tid < nch) {
        e = attn_expf(sarr[tid] - m_new);
        sarr[tid] = e;
    }
    reds[tid] = e;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (tid < st) reds[tid] += reds[tid + st];
        __syncthreads();
    }
    acc *= corr;
#pragma unroll 8
    for (int i = 0; i < nch; i++) {
        acc += sarr[i] * vacc(i);
    }
    l_run = l_run * corr + reds[0];
    m_run = m_new;
    __syncthreads();
}

#define ATTN_SPLITS_MAX 64



extern "C" __global__ void attn_prep_g4(
    const float* __restrict__ q,
    const float* __restrict__ kin,
    const float* __restrict__ vin,
    const float* __restrict__ qnw,
    const float* __restrict__ knw,
    float* __restrict__ qh,
    const unsigned long long* __restrict__ dblk_tab,
    const unsigned* __restrict__ pp,
    const unsigned long long* __restrict__ row_blk,
    const unsigned long long* __restrict__ row_pp,
    const float* __restrict__ rp,
    int t_len, int layer, int q_heads, int kv_heads, int cap)
{
    __shared__ float hd[512];
    __shared__ float red[128];
    const int d = (int)rp[layer * 7 + 0];
    const int npairs = (int)rp[layer * 7 + 1];
    const int span = (int)rp[layer * 7 + 2];
    const int denom = (int)rp[layer * 7 + 3];
    const float rbase = rp[layer * 7 + 4];
    const int vn_on = (int)rp[layer * 7 + 5];
    const int t = blockIdx.x;
    const int j = blockIdx.y;
    const int tid = threadIdx.x;
    const unsigned long long* blk = dblk_tab;
    int pos;
    if (row_blk != (const unsigned long long*)0) {
        blk = (const unsigned long long*)row_blk[t];
        pos = (int)((const unsigned*)row_pp[t])[0];
    } else {
        pos = (int)pp[0] + t;
    }
    const long kv_dim = (long)kv_heads * d;
    if (j < q_heads) {
        const float* src = q + (long)t * q_heads * d + (long)j * d;
        float ss = 0.0f;
        for (int i = tid; i < d; i += 128) {
            hd[i] = src[i];
            ss += hd[i] * hd[i];
        }
        red[tid] = ss;
        __syncthreads();
        for (int s = 64; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        const float inv = 1.0f / sqrtf(red[0] / (float)d + 1e-6f);
        for (int i = tid; i < d; i += 128) {
            hd[i] = hd[i] * inv * qnw[i];
        }
        __syncthreads();
        if (tid < npairs) {
            const float theta = attn_theta_p(tid, denom, rbase);
            const float ang = (float)pos * theta;
            const float c = attn_cosf(ang), s2 = attn_sinf(ang);
            const float x0 = hd[tid], x1 = hd[tid + span];
            hd[tid] = x0 * c - x1 * s2;
            hd[tid + span] = x0 * s2 + x1 * c;
        }
        __syncthreads();
        float* dst = qh + (long)t * q_heads * d + (long)j * d;
        for (int i = tid; i < d; i += 128) {
            dst[i] = hd[i];
        }
        return;
    }
    const int m = j - q_heads;
    const float* ksrc = kin + (long)t * kv_dim + (long)m * d;
    float ss = 0.0f;
    for (int i = tid; i < d; i += 128) {
        hd[i] = ksrc[i];
        ss += hd[i] * hd[i];
    }
    red[tid] = ss;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float inv = 1.0f / sqrtf(red[0] / (float)d + 1e-6f);
    for (int i = tid; i < d; i += 128) {
        hd[i] = hd[i] * inv * knw[i];
    }
    __syncthreads();
    if (tid < npairs) {
        const float theta = attn_theta_p(tid, denom, rbase);
        const float ang = (float)pos * theta;
        const float c = attn_cosf(ang), s2 = attn_sinf(ang);
        const float x0 = hd[tid], x1 = hd[tid + span];
        hd[tid] = x0 * c - x1 * s2;
        hd[tid + span] = x0 * s2 + x1 * c;
    }
    AttnKvRowPtrs row = attn_blk_row(blk, layer, cap, pos, kv_dim, kv_heads, m);
    float amax = 0.0f;
    for (int i = tid; i < d; i += 128) {
        amax = fmaxf(amax, fabsf(hd[i]));
    }
    red[tid] = amax;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    float ks = red[0] * (1.0f / 127.0f);
    if (ks == 0.0f) {
        ks = 1.0f;
    }
    const float ik = 1.0f / ks;
    __syncthreads();
    for (int i = tid; i < d; i += 128) {
        const int qv = __float2int_rn(hd[i] * ik);
        row.k[i] = (signed char)max(-127, min(127, qv));
    }
    if (tid == 0) {
        row.ks[0] = ks;
    }
    const float* vsrc = vin + (long)t * kv_dim + (long)m * d;
    __syncthreads();
    if (vn_on) {
        float vss = 0.0f;
        for (int i = tid; i < d; i += 128) {
            const float v = vsrc[i];
            hd[i] = v;
            vss += v * v;
        }
        red[tid] = vss;
        __syncthreads();
        for (int s = 64; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        const float vinv = 1.0f / sqrtf(red[0] / (float)d + 1e-6f);
        for (int i = tid; i < d; i += 128) {
            hd[i] *= vinv;
        }
    } else {
        for (int i = tid; i < d; i += 128) {
            hd[i] = vsrc[i];
        }
    }
    __syncthreads();
    float vam = 0.0f;
    for (int i = tid; i < d; i += 128) {
        vam = fmaxf(vam, fabsf(hd[i]));
    }
    red[tid] = vam;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    float vs = red[0] * (1.0f / 127.0f);
    if (vs == 0.0f) {
        vs = 1.0f;
    }
    const float iv = 1.0f / vs;
    for (int i = tid; i < d; i += 128) {
        const int wv = __float2int_rn(hd[i] * iv);
        row.v[i] = (signed char)max(-127, min(127, wv));
    }
    if (tid == 0) {
        row.vs[0] = vs;
    }
}

extern "C" __global__ void attn_fwd3s_part_q(
    const float* __restrict__ qh,
    const unsigned long long* __restrict__ dblk_tab,
    float* __restrict__ part,
    const unsigned* __restrict__ pp,
    const unsigned long long* __restrict__ row_blk,
    const unsigned long long* __restrict__ row_pp,
    int t_len, int layer, int q_heads, int kv_heads, int cap, int splits,
    float kq_scale)
{
    int t = blockIdx.x;
    int hs = blockIdx.y;
    int h = hs / splits;
    int s = hs - h * splits;
    int tid = threadIdx.x;
    int gq = q_heads / kv_heads;
    int kh = h / gq;
    float scale = kq_scale;
    const unsigned long long* blk = dblk_tab;
    int lim;
    if (row_blk != (const unsigned long long*)0) {
        blk = (const unsigned long long*)row_blk[t];
        lim = (int)((const unsigned*)row_pp[t])[0] + 1;
    } else {
        lim = (int)pp[0] + t + 1;
    }
    long kv_dim = (long)kv_heads * 256;
    long qrow = (long)t * (q_heads * 256) + (long)h * 256;
    __shared__ float qs[256];
    __shared__ float sarr[ATTN_CHUNK];
    __shared__ float reds[256];
    qs[tid] = qh[qrow + tid];
    __syncthreads();
    const int eff = (lim <= 256) ? 1 : splits;
    float* out = part + (((long)t * q_heads + h) * splits + s) * 258;
    if (s >= eff) {
        if (tid == 0) {
            out[0] = -1e30f;
            out[1] = 0.0f;
        }
        out[2 + tid] = 0.0f;
        return;
    }
    int chunk_rows = (lim + eff - 1) / eff;
    chunk_rows = ((chunk_rows + ATTN_CHUNK - 1) / ATTN_CHUNK) * ATTN_CHUNK;
    int lo = s * chunk_rows;
    int hi = min(lim, lo + chunk_rows);
    float m_run = -1e30f;
    float l_run = 0.0f;
    float acc = 0.0f;
    for (int base = lo; base < hi; base += ATTN_CHUNK) {
        const int nch = min(ATTN_CHUNK, hi - base);
        attn_part_step(
            [&](int i) -> float {
                const int row = base + i;
                const AttnKvRowPtrs rp =
                    attn_blk_row(blk, layer, cap, row, kv_dim, kv_heads, kh);
                const uint4* k16 = reinterpret_cast<const uint4*>(rp.k);
                const float4* q4 = reinterpret_cast<const float4*>(qs);
                float p0 = 0.0f;
#pragma unroll 4
                for (int d16 = 0; d16 < 16; d16++) {
                    const uint4 w = k16[d16];
                    const float4 qa = q4[d16 * 4 + 0];
                    const float4 qb = q4[d16 * 4 + 1];
                    const float4 qc = q4[d16 * 4 + 2];
                    const float4 qd = q4[d16 * 4 + 3];
                    p0 += qa.x * (float)(signed char)(w.x & 0xffu);
                    p0 += qa.y * (float)(signed char)((w.x >> 8) & 0xffu);
                    p0 += qa.z * (float)(signed char)((w.x >> 16) & 0xffu);
                    p0 += qa.w * (float)(signed char)(w.x >> 24);
                    p0 += qb.x * (float)(signed char)(w.y & 0xffu);
                    p0 += qb.y * (float)(signed char)((w.y >> 8) & 0xffu);
                    p0 += qb.z * (float)(signed char)((w.y >> 16) & 0xffu);
                    p0 += qb.w * (float)(signed char)(w.y >> 24);
                    p0 += qc.x * (float)(signed char)(w.z & 0xffu);
                    p0 += qc.y * (float)(signed char)((w.z >> 8) & 0xffu);
                    p0 += qc.z * (float)(signed char)((w.z >> 16) & 0xffu);
                    p0 += qc.w * (float)(signed char)(w.z >> 24);
                    p0 += qd.x * (float)(signed char)(w.w & 0xffu);
                    p0 += qd.y * (float)(signed char)((w.w >> 8) & 0xffu);
                    p0 += qd.z * (float)(signed char)((w.w >> 16) & 0xffu);
                    p0 += qd.w * (float)(signed char)(w.w >> 24);
                }
                return p0 * (scale * rp.ks[0]);
            },
            [&](int i) -> float {
                const int row = base + i;
                const AttnKvRowPtrs rp =
                    attn_blk_row(blk, layer, cap, row, kv_dim, kv_heads, kh);
                return (float)rp.v[tid] * rp.vs[0];
            },
            sarr, reds, nch, tid, m_run, l_run, acc);
    }
    if (tid == 0) {
        out[0] = m_run;
        out[1] = l_run;
    }
    out[2 + tid] = acc;
}



extern "C" __global__ void attn_fwd3s_merge(
    const float* __restrict__ part,
    const float* __restrict__ qg,
    float* __restrict__ outv,
    float* __restrict__ out32,
    unsigned short* __restrict__ out16,
    int t_len, int q_heads, int splits)
{
    int t = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    const float* pb = part + ((long)t * q_heads + h) * splits * 258;
    __shared__ float ms[ATTN_SPLITS_MAX];
    __shared__ float ws[ATTN_SPLITS_MAX];
    __shared__ float mred[1];
    if (tid < splits) {
        ms[tid] = pb[tid * 258];
    }
    __syncthreads();
    if (tid == 0) {
        float m0 = -1e30f;
        for (int s = 0; s < splits; s++) {
            m0 = fmaxf(m0, ms[s]);
        }
        mred[0] = m0;
    }
    __syncthreads();
    const float m = mred[0];
    if (tid < splits) {
        const float* ps = pb + tid * 258;
        ws[tid] = (ps[1] <= 0.0f) ? 0.0f : attn_expf(ps[0] - m);
    }
    __syncthreads();
    float l = 0.0f;
    float acc = 0.0f;
    for (int s = 0; s < splits; s++) {
        const float* ps = pb + s * 258;
        if (ps[1] <= 0.0f) {
            continue;
        }
        l += ps[1] * ws[s];
        acc += ps[2 + tid] * ws[s];
    }
    float g = qg[(long)t * (q_heads * 512) + (long)h * 512 + 256 + tid];
    float ov = (acc / l) * (1.0f / (1.0f + attn_expf(-g)));
    const long oi = (long)t * (q_heads * 256) + (long)h * 256 + tid;
    outv[oi] = ov;
    if (out32 != (float*)0) {
        const __half h = __float2half_rn(ov);
        out32[oi] = __half2float(h);
        if (out16 != (unsigned short*)0) {
            out16[oi] = __half_as_ushort(h);
        }
    } else if (out16 != (unsigned short*)0) {
        out16[oi] = __half_as_ushort(__float2half_rn(ov));
    }
}

extern "C" __global__ void attn_fwd3s_part_g4(
    const float* __restrict__ qh,
    const unsigned long long* __restrict__ dblk_tab,
    float* __restrict__ part,
    const unsigned* __restrict__ pp,
    const unsigned long long* __restrict__ row_blk,
    const unsigned long long* __restrict__ row_pp,
    int t_len, int layer, int q_heads, int kv_heads, int cap, int splits,
    int d, int sw, float kq_scale)
{
    int t = blockIdx.x;
    int hs = blockIdx.y;
    int h = hs / splits;
    int s = hs - h * splits;
    int tid = threadIdx.x;
    int gq = q_heads / kv_heads;
    int kh = h / gq;
    float scale = kq_scale;
    const unsigned long long* blk = dblk_tab;
    int lim;
    if (row_blk != (const unsigned long long*)0) {
        blk = (const unsigned long long*)row_blk[t];
        lim = (int)((const unsigned*)row_pp[t])[0] + 1;
    } else {
        lim = (int)pp[0] + t + 1;
    }
    int start = (sw > 0 && lim > sw) ? (lim - sw) : 0;
    long kv_dim = (long)kv_heads * d;
    long qrow = (long)t * (q_heads * d) + (long)h * d;
    __shared__ float qs[512];
    __shared__ float sarr[ATTN_CHUNK];
    __shared__ float reds[256];
    if (tid < d) {
        qs[tid] = qh[qrow + tid];
    }
    __syncthreads();
    int span = lim - start;
    const int eff = (span <= 256) ? 1 : splits;
    float* out = part + (((long)t * q_heads + h) * splits + s) * (d + 2);
    if (s >= eff) {
        if (tid == 0) {
            out[0] = -1e30f;
            out[1] = 0.0f;
        }
        if (tid < d) {
            out[2 + tid] = 0.0f;
        }
        return;
    }
    int chunk_rows = (span + eff - 1) / eff;
    chunk_rows = ((chunk_rows + ATTN_SPLIT_GRAIN - 1) / ATTN_SPLIT_GRAIN) * ATTN_SPLIT_GRAIN;
    int lo = start + s * chunk_rows;
    int hi = min(lim, lo + chunk_rows);
    float m_run = -1e30f;
    float l_run = 0.0f;
    float acc = 0.0f;
    for (int base = lo; base < hi; base += ATTN_CHUNK) {
        const int nch = min(ATTN_CHUNK, hi - base);
        attn_part_step(
            [&](int i) -> float {
                const int row = base + i;
                const AttnKvRowPtrs rp =
                    attn_blk_row(blk, layer, cap, row, kv_dim, kv_heads, kh);
                const uint4* k16 = reinterpret_cast<const uint4*>(rp.k);
                const float4* q4 = reinterpret_cast<const float4*>(qs);
                float p0 = 0.0f;
                for (int d16 = 0; d16 < (d >> 4); d16++) {
                    const uint4 w = k16[d16];
                    const float4 qa = q4[d16 * 4 + 0];
                    const float4 qb = q4[d16 * 4 + 1];
                    const float4 qc = q4[d16 * 4 + 2];
                    const float4 qd = q4[d16 * 4 + 3];
                    p0 += qa.x * (float)(signed char)(w.x & 0xffu);
                    p0 += qa.y * (float)(signed char)((w.x >> 8) & 0xffu);
                    p0 += qa.z * (float)(signed char)((w.x >> 16) & 0xffu);
                    p0 += qa.w * (float)(signed char)(w.x >> 24);
                    p0 += qb.x * (float)(signed char)(w.y & 0xffu);
                    p0 += qb.y * (float)(signed char)((w.y >> 8) & 0xffu);
                    p0 += qb.z * (float)(signed char)((w.y >> 16) & 0xffu);
                    p0 += qb.w * (float)(signed char)(w.y >> 24);
                    p0 += qc.x * (float)(signed char)(w.z & 0xffu);
                    p0 += qc.y * (float)(signed char)((w.z >> 8) & 0xffu);
                    p0 += qc.z * (float)(signed char)((w.z >> 16) & 0xffu);
                    p0 += qc.w * (float)(signed char)(w.z >> 24);
                    p0 += qd.x * (float)(signed char)(w.w & 0xffu);
                    p0 += qd.y * (float)(signed char)((w.w >> 8) & 0xffu);
                    p0 += qd.z * (float)(signed char)((w.w >> 16) & 0xffu);
                    p0 += qd.w * (float)(signed char)(w.w >> 24);
                }
                return p0 * (scale * rp.ks[0]);
            },
            [&](int i) -> float {
                const int row = base + i;
                const AttnKvRowPtrs rp =
                    attn_blk_row(blk, layer, cap, row, kv_dim, kv_heads, kh);
                return (float)rp.v[tid] * rp.vs[0];
            },
            sarr, reds, nch, tid, m_run, l_run, acc);
    }
    if (tid == 0) {
        out[0] = m_run;
        out[1] = l_run;
    }
    if (tid < d) {
        out[2 + tid] = acc;
    }
}

extern "C" __global__ void attn_fwd3s_merge_g4(
    const float* __restrict__ part,
    float* __restrict__ outv,
    float* __restrict__ out32,
    unsigned short* __restrict__ out16,
    int t_len, int q_heads, int splits, int d)
{
    int t = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    const float* pb = part + ((long)t * q_heads + h) * splits * (d + 2);
    __shared__ float ms[ATTN_SPLITS_MAX];
    __shared__ float ws[ATTN_SPLITS_MAX];
    __shared__ float mred[1];
    if (tid < splits) {
        ms[tid] = pb[tid * (d + 2)];
    }
    __syncthreads();
    if (tid == 0) {
        float m0 = -1e30f;
        for (int s = 0; s < splits; s++) {
            m0 = fmaxf(m0, ms[s]);
        }
        mred[0] = m0;
    }
    __syncthreads();
    const float m = mred[0];
    if (tid < splits) {
        const float* ps = pb + tid * (d + 2);
        ws[tid] = (ps[1] <= 0.0f) ? 0.0f : attn_expf(ps[0] - m);
    }
    __syncthreads();
    float l = 0.0f;
    float acc = 0.0f;
    for (int s = 0; s < splits; s++) {
        const float* ps = pb + s * (d + 2);
        if (ps[1] <= 0.0f) {
            continue;
        }
        l += ps[1] * ws[s];
        acc += ps[2 + tid] * ws[s];
    }
    float ov = acc / l;
    const long oi = (long)t * (q_heads * d) + (long)h * d + tid;
    outv[oi] = ov;
    if (out32 != (float*)0) {
        const __half h = __float2half_rn(ov);
        out32[oi] = __half2float(h);
        if (out16 != (unsigned short*)0) {
            out16[oi] = __half_as_ushort(h);
        }
    } else if (out16 != (unsigned short*)0) {
        out16[oi] = __half_as_ushort(__float2half_rn(ov));
    }
}




#define ATTN_GQ_MAX 12
extern "C" __global__ void attn_fwd3s_part_qg(
    const float* __restrict__ qh,
    const unsigned long long* __restrict__ dblk_tab,
    float* __restrict__ part,
    const unsigned* __restrict__ pp,
    const unsigned long long* __restrict__ row_blk,
    const unsigned long long* __restrict__ row_pp,
    int t_len, int layer, int q_heads, int kv_heads, int cap, int splits,
    float kq_scale)
{
    int t = blockIdx.x;
    int hs = blockIdx.y;
    int kh = hs / splits;
    int s = hs - kh * splits;
    int tid = threadIdx.x;
    int gq = q_heads / kv_heads;
    int h0 = kh * gq;
    float scale = kq_scale;
    const unsigned long long* blk = dblk_tab;
    int lim;
    if (row_blk != (const unsigned long long*)0) {
        blk = (const unsigned long long*)row_blk[t];
        lim = (int)((const unsigned*)row_pp[t])[0] + 1;
    } else {
        lim = (int)pp[0] + t + 1;
    }
    long kv_dim = (long)kv_heads * 256;
    __shared__ float qs[ATTN_GQ_MAX][256];
    __shared__ float sarr[ATTN_GQ_MAX][ATTN_CHUNK];
    __shared__ float reds[ATTN_GQ_MAX][256];
    __shared__ float mrun[ATTN_GQ_MAX], lrun[ATTN_GQ_MAX], corr[ATTN_GQ_MAX], mnew[ATTN_GQ_MAX];
    const int eff = (lim <= 256) ? 1 : splits;
    if (s >= eff) {
        if (tid == 0) {
            for (int h = 0; h < gq; ++h) {
                float* out = part + (((long)t * q_heads + (h0 + h)) * splits + s) * 258;
                out[0] = -1e30f;
                out[1] = 0.0f;
            }
        }
        return;
    }
    for (int h = 0; h < gq; ++h) {
        const long qrow = (long)t * (q_heads * 256) + (long)(h0 + h) * 256;
        qs[h][tid] = qh[qrow + tid];
    }
    __syncthreads();
    int chunk_rows = (lim + eff - 1) / eff;
    chunk_rows = ((chunk_rows + ATTN_SPLIT_GRAIN - 1) / ATTN_SPLIT_GRAIN) * ATTN_SPLIT_GRAIN;
    int lo = s * chunk_rows;
    int hi = min(lim, lo + chunk_rows);
    if (tid < gq) {
        mrun[tid] = -1e30f;
        lrun[tid] = 0.0f;
    }
    float acc[ATTN_GQ_MAX];
#pragma unroll
    for (int h = 0; h < ATTN_GQ_MAX; ++h) {
        acc[h] = 0.0f;
    }
    for (int base = lo; base < hi; base += ATTN_CHUNK) {
        const int nch = min(ATTN_CHUNK, hi - base);
        if (tid < nch) {
            const int row = base + tid;
            const AttnKvRowPtrs rp =
                attn_blk_row(blk, layer, cap, row, kv_dim, kv_heads, kh);
            const uint4* k16 = reinterpret_cast<const uint4*>(rp.k);
            float p[ATTN_GQ_MAX];
#pragma unroll
            for (int h = 0; h < ATTN_GQ_MAX; ++h) {
                p[h] = 0.0f;
            }
#pragma unroll 4
            for (int d16 = 0; d16 < 16; d16++) {
                const uint4 w = k16[d16];
#pragma unroll
                for (int h = 0; h < ATTN_GQ_MAX; ++h) {
                    if (h < gq) {
                        const float4 qa = *reinterpret_cast<const float4*>(&qs[h][d16 * 16 + 0]);
                        const float4 qb = *reinterpret_cast<const float4*>(&qs[h][d16 * 16 + 4]);
                        const float4 qc = *reinterpret_cast<const float4*>(&qs[h][d16 * 16 + 8]);
                        const float4 qd = *reinterpret_cast<const float4*>(&qs[h][d16 * 16 + 12]);
                        p[h] += qa.x * (float)(signed char)(w.x & 0xffu);
                        p[h] += qa.y * (float)(signed char)((w.x >> 8) & 0xffu);
                        p[h] += qa.z * (float)(signed char)((w.x >> 16) & 0xffu);
                        p[h] += qa.w * (float)(signed char)(w.x >> 24);
                        p[h] += qb.x * (float)(signed char)(w.y & 0xffu);
                        p[h] += qb.y * (float)(signed char)((w.y >> 8) & 0xffu);
                        p[h] += qb.z * (float)(signed char)((w.y >> 16) & 0xffu);
                        p[h] += qb.w * (float)(signed char)(w.y >> 24);
                        p[h] += qc.x * (float)(signed char)(w.z & 0xffu);
                        p[h] += qc.y * (float)(signed char)((w.z >> 8) & 0xffu);
                        p[h] += qc.z * (float)(signed char)((w.z >> 16) & 0xffu);
                        p[h] += qc.w * (float)(signed char)(w.z >> 24);
                        p[h] += qd.x * (float)(signed char)(w.w & 0xffu);
                        p[h] += qd.y * (float)(signed char)((w.w >> 8) & 0xffu);
                        p[h] += qd.z * (float)(signed char)((w.w >> 16) & 0xffu);
                        p[h] += qd.w * (float)(signed char)(w.w >> 24);
                    }
                }
            }
            const float sc = scale * rp.ks[0];
#pragma unroll
            for (int h = 0; h < ATTN_GQ_MAX; ++h) {
                sarr[h][tid] = (h < gq) ? p[h] * sc : -1e30f;
            }
        } else {
#pragma unroll
            for (int h = 0; h < ATTN_GQ_MAX; ++h) {
                sarr[h][tid] = -1e30f;
            }
        }
        __syncthreads();
#pragma unroll
        for (int h = 0; h < ATTN_GQ_MAX; ++h) {
            reds[h][tid] = sarr[h][tid];
        }
        __syncthreads();
        for (int st = 128; st > 0; st >>= 1) {
            if (tid < st) {
#pragma unroll
                for (int h = 0; h < ATTN_GQ_MAX; ++h) {
                    reds[h][tid] = fmaxf(reds[h][tid], reds[h][tid + st]);
                }
            }
            __syncthreads();
        }
        if (tid < gq) {
            mnew[tid] = fmaxf(mrun[tid], reds[tid][0]);
            corr[tid] = (mrun[tid] <= -1e29f) ? 0.0f : attn_expf(mrun[tid] - mnew[tid]);
        }
        __syncthreads();
        float e = 0.0f;
        if (tid < nch) {
#pragma unroll
            for (int h = 0; h < ATTN_GQ_MAX; ++h) {
                if (h < gq) {
                    e = attn_expf(sarr[h][tid] - mnew[h]);
                    sarr[h][tid] = e;
                    reds[h][tid] = e;
                }
            }
        } else {
#pragma unroll
            for (int h = 0; h < ATTN_GQ_MAX; ++h) {
                reds[h][tid] = 0.0f;
            }
        }
        __syncthreads();
        for (int st = 128; st > 0; st >>= 1) {
            if (tid < st) {
#pragma unroll
                for (int h = 0; h < ATTN_GQ_MAX; ++h) {
                    reds[h][tid] += reds[h][tid + st];
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int h = 0; h < ATTN_GQ_MAX; ++h) {
            if (h < gq) {
                acc[h] *= corr[h];
            }
        }
        int i = 0;
#pragma unroll 4
        for (; i + 3 < nch; i += 4) {
            const int r0 = base + i;
            const AttnKvRowPtrs r0p =
                attn_blk_row(blk, layer, cap, r0, kv_dim, kv_heads, kh);
            const AttnKvRowPtrs r1p =
                attn_blk_row(blk, layer, cap, r0 + 1, kv_dim, kv_heads, kh);
            const AttnKvRowPtrs r2p =
                attn_blk_row(blk, layer, cap, r0 + 2, kv_dim, kv_heads, kh);
            const AttnKvRowPtrs r3p =
                attn_blk_row(blk, layer, cap, r0 + 3, kv_dim, kv_heads, kh);
            const float v0 = (float)r0p.v[tid] * r0p.vs[0];
            const float v1 = (float)r1p.v[tid] * r1p.vs[0];
            const float v2 = (float)r2p.v[tid] * r2p.vs[0];
            const float v3 = (float)r3p.v[tid] * r3p.vs[0];
#pragma unroll
            for (int h = 0; h < ATTN_GQ_MAX; ++h) {
                if (h < gq) {
                    const float4 s4 = *reinterpret_cast<const float4*>(&sarr[h][i]);
                    acc[h] += s4.x * v0;
                    acc[h] += s4.y * v1;
                    acc[h] += s4.z * v2;
                    acc[h] += s4.w * v3;
                }
            }
        }
        for (; i < nch; i++) {
            const int row = base + i;
            const AttnKvRowPtrs rp =
                attn_blk_row(blk, layer, cap, row, kv_dim, kv_heads, kh);
            const float vv = (float)rp.v[tid] * rp.vs[0];
#pragma unroll
            for (int h = 0; h < ATTN_GQ_MAX; ++h) {
                if (h < gq) {
                    acc[h] += sarr[h][i] * vv;
                }
            }
        }
        if (tid < gq) {
            lrun[tid] = lrun[tid] * corr[tid] + reds[tid][0];
            mrun[tid] = mnew[tid];
        }
        __syncthreads();
    }
    for (int h = 0; h < gq; ++h) {
        float* out = part + (((long)t * q_heads + (h0 + h)) * splits + s) * 258;
        if (tid == 0) {
            out[0] = mrun[h];
            out[1] = lrun[h];
        }
        out[2 + tid] = acc[h];
    }
}


#define FA2_BR 32
#define FA2_BC 32


__device__ __forceinline__ void fa2_mma(float* c, unsigned a0, unsigned a1,
                                        unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned fa2_k8(unsigned short v)
{
    const int a = (signed char)(v & 0xffu);
    const int b = (signed char)((v >> 8) & 0xffu);
    __half2 h = __floats2half2_rn((float)a, (float)b);
    return *reinterpret_cast<unsigned*>(&h);
}

extern "C" __global__ void __launch_bounds__(128, 2) attn_fa2_qg(
    const float* __restrict__ qh,
    const unsigned long long* __restrict__ dblk_tab,
    const float* __restrict__ qg,
    float* __restrict__ outv,
    float* __restrict__ out32,
    unsigned short* __restrict__ out16,
    const unsigned* __restrict__ pp,
    int t_len, int layer, int q_heads, int kv_heads, int cap, float kq_scale,
    int sw, int no_gate)
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g = lane >> 2;
    const int tt = lane & 3;
    const int wi = warp >> 1;
    const int wg = warp & 1;
    const int h = blockIdx.y;
    const int gq_ = q_heads / kv_heads;
    const int kh = h / gq_;
    const int rb = blockIdx.x * FA2_BR;
    const int pos0 = (int)pp[0];
    const long kv_dim = (long)kv_heads * 256;

    __shared__ __align__(16) unsigned short Qs[FA2_BR][264];
    __shared__ __align__(16) signed char K8[FA2_BC][272];
    __shared__ __align__(16) unsigned short VsT[256][34];
    __shared__ __align__(16) unsigned short Ps[FA2_BR][36];
    __shared__ float ksc_s[FA2_BC];
    __shared__ float vsc_s[FA2_BC];
    __shared__ float m_s[FA2_BR];
    __shared__ float l_s[FA2_BR];
    __shared__ float corr_s[FA2_BR];

    float qmx = 0.0f;
    for (int e = tid; e < FA2_BR * 256; e += 128) {
        const int r = e >> 8;
        const int d_ = e & 255;
        const int row = rb + r;
        if (row < t_len) {
            qmx = fmaxf(
                qmx, fabsf(qh[(size_t)row * (q_heads * 256) + (size_t)h * 256 + d_]));
        }
    }
    __shared__ float qred[128];
    qred[tid] = qmx;
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (tid < st) {
            qred[tid] = fmaxf(qred[tid], qred[tid + st]);
        }
        __syncthreads();
    }
    const float qsc = (qred[0] > 30000.0f) ? (30000.0f / qred[0]) : 1.0f;
    const float iqsc = 1.0f / qsc;
    for (int e = tid; e < FA2_BR * 256; e += 128) {
        const int r = e >> 8;
        const int d_ = e & 255;
        const int row = rb + r;
        float qv = 0.0f;
        if (row < t_len) {
            qv = qh[(size_t)row * (q_heads * 256) + (size_t)h * 256 + d_];
        }
        Qs[r][d_] = __half_as_ushort(__float2half_rn(qv * qsc));
    }
    if (tid < FA2_BR) {
        m_s[tid] = -1e30f;
        l_s[tid] = 0.0f;
        corr_s[tid] = 0.0f;
    }
    __syncthreads();

    const int row_end = min(rb + FA2_BR - 1, t_len - 1);
    const int lim_max = pos0 + row_end + 1;
    const int nchunks = (lim_max + FA2_BC - 1) / FA2_BC;
    const int pos_end = pos0 + t_len;

    float o[16][4];
#pragma unroll
    for (int f = 0; f < 16; ++f) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            o[f][i] = 0.0f;
        }
    }

    int c_start = 0;
    if (sw > 0) {
        const int lo_first = pos0 + rb + 1 - sw;
        if (lo_first > 0) {
            c_start = lo_first / FA2_BC;
        }
    }
    for (int c0 = c_start; c0 < nchunks; ++c0) {
        const int base = c0 * FA2_BC;
        for (int e = tid; e < FA2_BC * 16; e += 128) {
            const int j = e >> 4;
            const int qq = e & 15;
            const int row = base + j;
            const signed char* kp =
                attn_blk_row(dblk_tab, layer, cap, row, kv_dim, kv_heads, kh).k +
                qq * 16;
            if (row < pos_end) {
                *reinterpret_cast<uint4*>(&K8[j][qq * 16]) =
                    *reinterpret_cast<const uint4*>(kp);
            } else {
                *reinterpret_cast<uint4*>(&K8[j][qq * 16]) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        for (int e = tid; e < FA2_BC * 16; e += 128) {
            const int j = e >> 4;
            const int qq = e & 15;
            const int row = base + j;
            signed char vb[16];
            if (row < pos_end) {
                const signed char* vp =
                    attn_blk_row(dblk_tab, layer, cap, row, kv_dim, kv_heads, kh).v +
                    qq * 16;
                *reinterpret_cast<uint4*>(vb) = *reinterpret_cast<const uint4*>(vp);
            } else {
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    vb[i] = 0;
                }
            }
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                VsT[qq * 16 + i][j] = __half_as_ushort(__float2half_rn((float)vb[i]));
            }
        }
        if (tid < FA2_BC) {
            const int row = base + tid;
            const bool ok = row < pos_end;
            const AttnKvRowPtrs rp =
                attn_blk_row(dblk_tab, layer, cap, row, kv_dim, kv_heads, kh);
            ksc_s[tid] = ok ? rp.ks[0] : 0.0f;
            vsc_s[tid] = ok ? rp.vs[0] : 0.0f;
        }
        __syncthreads();

        if (wg == 0) {
            float sa[4][4];
#pragma unroll
            for (int f = 0; f < 4; ++f) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sa[f][i] = 0.0f;
                }
            }
#pragma unroll
            for (int kb = 0; kb < 256; kb += 16) {
                const unsigned a0 =
                    *reinterpret_cast<const unsigned*>(&Qs[wi * 16 + g][kb + 2 * tt]);
                const unsigned a1 =
                    *reinterpret_cast<const unsigned*>(&Qs[wi * 16 + g + 8][kb + 2 * tt]);
                const unsigned a2 =
                    *reinterpret_cast<const unsigned*>(&Qs[wi * 16 + g][kb + 2 * tt + 8]);
                const unsigned a3 = *reinterpret_cast<const unsigned*>(
                    &Qs[wi * 16 + g + 8][kb + 2 * tt + 8]);
#pragma unroll
                for (int f = 0; f < 4; ++f) {
                    const int j = f * 8 + g;
                    const unsigned b0 =
                        fa2_k8(*reinterpret_cast<const unsigned short*>(&K8[j][kb + 2 * tt]));
                    const unsigned b1 = fa2_k8(
                        *reinterpret_cast<const unsigned short*>(&K8[j][kb + 2 * tt + 8]));
                    fa2_mma(sa[f], a0, a1, a2, a3, b0, b1);
                }
            }
            const int rA = wi * 16 + g;
            const int rB = rA + 8;
            const int nA = pos0 + rb + rA + 1 - base;
            const int nB = pos0 + rb + rB + 1 - base;
            const int loA = (sw > 0) ? (pos0 + rb + rA + 1 - sw - base) : 0;
            const int loB = (sw > 0) ? (pos0 + rb + rB + 1 - sw - base) : 0;
#pragma unroll
            for (int f = 0; f < 4; ++f) {
                const int cc = f * 8 + 2 * tt;
                const float s0 = kq_scale * ksc_s[cc] * iqsc;
                const float s1 = kq_scale * ksc_s[cc + 1] * iqsc;
                sa[f][0] *= s0;
                sa[f][1] *= s1;
                sa[f][2] *= s0;
                sa[f][3] *= s1;
                if (cc >= nA) {
                    sa[f][0] = -1e30f;
                }
                if (cc + 1 >= nA) {
                    sa[f][1] = -1e30f;
                }
                if (cc >= nB) {
                    sa[f][2] = -1e30f;
                }
                if (cc + 1 >= nB) {
                    sa[f][3] = -1e30f;
                }
                if (cc < loA) {
                    sa[f][0] = -1e30f;
                }
                if (cc + 1 < loA) {
                    sa[f][1] = -1e30f;
                }
                if (cc < loB) {
                    sa[f][2] = -1e30f;
                }
                if (cc + 1 < loB) {
                    sa[f][3] = -1e30f;
                }
            }
            float mg = -1e30f;
            float mh = -1e30f;
#pragma unroll
            for (int f = 0; f < 4; ++f) {
                mg = fmaxf(mg, fmaxf(sa[f][0], sa[f][1]));
                mh = fmaxf(mh, fmaxf(sa[f][2], sa[f][3]));
            }
            mg = fmaxf(mg, __shfl_xor_sync(0xffffffffu, mg, 1));
            mg = fmaxf(mg, __shfl_xor_sync(0xffffffffu, mg, 2));
            mh = fmaxf(mh, __shfl_xor_sync(0xffffffffu, mh, 1));
            mh = fmaxf(mh, __shfl_xor_sync(0xffffffffu, mh, 2));
            const float moldA = m_s[rA];
            const float moldB = m_s[rB];
            const float mnewA = fmaxf(moldA, mg);
            const float mnewB = fmaxf(moldB, mh);
            const float corrA = (moldA <= -1e29f) ? 0.0f : attn_expf(moldA - mnewA);
            const float corrB = (moldB <= -1e29f) ? 0.0f : attn_expf(moldB - mnewB);
            float sumA = 0.0f;
            float sumB = 0.0f;
#pragma unroll
            for (int f = 0; f < 4; ++f) {
                const int cc = f * 8 + 2 * tt;
                const float vs0 = vsc_s[cc];
                const float vs1 = vsc_s[cc + 1];
                const float p0 = attn_expf(sa[f][0] - mnewA);
                const float p1 = attn_expf(sa[f][1] - mnewA);
                const float p2 = attn_expf(sa[f][2] - mnewB);
                const float p3 = attn_expf(sa[f][3] - mnewB);
                sumA += p0 + p1;
                sumB += p2 + p3;
                const __half2 h01 = __floats2half2_rn(p0 * vs0, p1 * vs1);
                const __half2 h23 = __floats2half2_rn(p2 * vs0, p3 * vs1);
                *reinterpret_cast<unsigned*>(&Ps[rA][cc]) =
                    *reinterpret_cast<const unsigned*>(&h01);
                *reinterpret_cast<unsigned*>(&Ps[rB][cc]) =
                    *reinterpret_cast<const unsigned*>(&h23);
            }
            sumA += __shfl_xor_sync(0xffffffffu, sumA, 1);
            sumA += __shfl_xor_sync(0xffffffffu, sumA, 2);
            sumB += __shfl_xor_sync(0xffffffffu, sumB, 1);
            sumB += __shfl_xor_sync(0xffffffffu, sumB, 2);
            __syncwarp();
            if (tt == 0) {
                m_s[rA] = mnewA;
                m_s[rB] = mnewB;
                l_s[rA] = l_s[rA] * corrA + sumA;
                l_s[rB] = l_s[rB] * corrB + sumB;
                corr_s[rA] = corrA;
                corr_s[rB] = corrB;
            }
        }
        __syncthreads();

        {
            const int rA = wi * 16 + g;
            const int rB = rA + 8;
            const float corrA = corr_s[rA];
            const float corrB = corr_s[rB];
#pragma unroll
            for (int f = 0; f < 16; ++f) {
                o[f][0] *= corrA;
                o[f][1] *= corrA;
                o[f][2] *= corrB;
                o[f][3] *= corrB;
            }
#pragma unroll
            for (int kb = 0; kb < FA2_BC; kb += 16) {
                const unsigned a0 =
                    *reinterpret_cast<const unsigned*>(&Ps[rA][kb + 2 * tt]);
                const unsigned a1 =
                    *reinterpret_cast<const unsigned*>(&Ps[rB][kb + 2 * tt]);
                const unsigned a2 =
                    *reinterpret_cast<const unsigned*>(&Ps[rA][kb + 2 * tt + 8]);
                const unsigned a3 =
                    *reinterpret_cast<const unsigned*>(&Ps[rB][kb + 2 * tt + 8]);
#pragma unroll
                for (int f = 0; f < 16; ++f) {
                    const int d_ = wg * 128 + f * 8 + g;
                    const unsigned b0 = *reinterpret_cast<const unsigned*>(
                        &VsT[d_][kb + 2 * tt]);
                    const unsigned b1 = *reinterpret_cast<const unsigned*>(
                        &VsT[d_][kb + 2 * tt + 8]);
                    fa2_mma(o[f], a0, a1, a2, a3, b0, b1);
                }
            }
        }
        __syncthreads();
    }

    {
        const int rA = wi * 16 + g;
        const int rB = rA + 8;
        const float invlA = 1.0f / l_s[rA];
        const float invlB = 1.0f / l_s[rB];
        const int rowA = rb + rA;
        const int rowB = rb + rB;
#pragma unroll
        for (int f = 0; f < 16; ++f) {
            const int d_ = wg * 128 + f * 8 + 2 * tt;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int r0 = (e < 2) ? rowA : rowB;
                const int d2 = d_ + (e & 1);
                if (r0 < t_len) {
                    float ov = o[f][e] * ((e < 2) ? invlA : invlB);
                    if (!no_gate) {
                        const float gv = qg[(size_t)r0 * (q_heads * 512) + (size_t)h * 512 +
                                            256 + d2];
                        ov *= 1.0f / (1.0f + attn_expf(-gv));
                    }
                    const long oi = (size_t)r0 * (q_heads * 256) + (size_t)h * 256 + d2;
                    outv[oi] = ov;
                    if (out32 != (float*)0) {
                        const __half h = __float2half_rn(ov);
                        out32[oi] = __half2float(h);
                        if (out16 != (unsigned short*)0) {
                            out16[oi] = __half_as_ushort(h);
                        }
                    } else if (out16 != (unsigned short*)0) {
                        out16[oi] = __half_as_ushort(__float2half_rn(ov));
                    }
                }
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(128, 2) attn_fa2_g4_512(
    const float* __restrict__ qh,
    const unsigned long long* __restrict__ dblk_tab,
    const float* __restrict__ qg,
    float* __restrict__ outv,
    float* __restrict__ out32,
    unsigned short* __restrict__ out16,
    const unsigned* __restrict__ pp,
    int t_len, int layer, int q_heads, int kv_heads, int cap, float kq_scale,
    int sw, int no_gate)
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g = lane >> 2;
    const int tt = lane & 3;
    const int h = blockIdx.y;
    const int gq_ = q_heads / kv_heads;
    const int kh = h / gq_;
    const int rb = blockIdx.x * 16;
    const int pos0 = (int)pp[0];
    const long kv_dim = (long)kv_heads * 512;

    __shared__ __align__(16) unsigned short Qs[16][520];
    __shared__ __align__(16) signed char K8[16][528];
    __shared__ __align__(16) unsigned short VsT[512][18];
    __shared__ __align__(16) unsigned short Ps[16][20];
    __shared__ float ksc_s[16];
    __shared__ float vsc_s[16];
    __shared__ float m_s[16];
    __shared__ float l_s[16];
    __shared__ float corr_s[16];

    float qmxr[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        qmxr[r] = 0.0f;
    }
    for (int e = tid; e < 16 * 512; e += 128) {
        const int r = e >> 9;
        const int d_ = e & 511;
        const int row = rb + r;
        if (row < t_len) {
            qmxr[r] = fmaxf(
                qmxr[r],
                fabsf(qh[(size_t)row * (q_heads * 512) + (size_t)h * 512 + d_]));
        }
    }
    __shared__ float qrm[4][16];
    __shared__ float qrow[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        float m = qmxr[r];
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 16));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 8));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 4));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
        if ((lane & 31) == 0) {
            qrm[warp][r] = m;
        }
    }
    __syncthreads();
    if (warp == 0) {
        if (lane < 16) {
            qrow[lane] = fmaxf(fmaxf(qrm[0][lane], qrm[1][lane]),
                               fmaxf(qrm[2][lane], qrm[3][lane]));
        }
    }
    __syncthreads();
    for (int e = tid; e < 16 * 512; e += 128) {
        const int r = e >> 9;
        const int d_ = e & 511;
        const int row = rb + r;
        float qv = 0.0f;
        if (row < t_len) {
            qv = qh[(size_t)row * (q_heads * 512) + (size_t)h * 512 + d_];
        }
        const float qs_r = (qrow[r] > 30000.0f) ? (30000.0f / qrow[r]) : 1.0f;
        Qs[r][d_] = __half_as_ushort(__float2half_rn(qv * qs_r));
    }
    if (tid < 16) {
        m_s[tid] = -1e30f;
        l_s[tid] = 0.0f;
        corr_s[tid] = 0.0f;
    }
    __syncthreads();

    const int row_end = min(rb + 15, t_len - 1);
    const int lim_max = pos0 + row_end + 1;
    const int nchunks = (lim_max + 15) / 16;
    const int pos_end = pos0 + t_len;

    float o[16][4];
#pragma unroll
    for (int f = 0; f < 16; ++f) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            o[f][i] = 0.0f;
        }
    }

    int c_start = 0;
    if (sw > 0) {
        const int lo_first = pos0 + rb + 1 - sw;
        if (lo_first > 0) {
            c_start = lo_first / 16;
        }
    }
    for (int c0 = c_start; c0 < nchunks; ++c0) {
        const int base = c0 * 16;
        for (int e = tid; e < 16 * 32; e += 128) {
            const int j = e >> 5;
            const int qq = e & 31;
            const int row = base + j;
            const signed char* kp =
                attn_blk_row(dblk_tab, layer, cap, row, kv_dim, kv_heads, kh).k +
                qq * 16;
            if (row < pos_end) {
                *reinterpret_cast<uint4*>(&K8[j][qq * 16]) =
                    *reinterpret_cast<const uint4*>(kp);
            } else {
                *reinterpret_cast<uint4*>(&K8[j][qq * 16]) =
                    make_uint4(0u, 0u, 0u, 0u);
            }
        }
        for (int e = tid; e < 16 * 32; e += 128) {
            const int j = e >> 5;
            const int qq = e & 31;
            const int row = base + j;
            signed char vb[16];
            if (row < pos_end) {
                const signed char* vp =
                    attn_blk_row(dblk_tab, layer, cap, row, kv_dim, kv_heads, kh).v +
                    qq * 16;
                *reinterpret_cast<uint4*>(vb) = *reinterpret_cast<const uint4*>(vp);
            } else {
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    vb[i] = 0;
                }
            }
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                VsT[qq * 16 + i][j] =
                    __half_as_ushort(__float2half_rn((float)vb[i]));
            }
        }
        if (tid < 16) {
            const int row = base + tid;
            const bool ok = row < pos_end;
            const AttnKvRowPtrs rp =
                attn_blk_row(dblk_tab, layer, cap, row, kv_dim, kv_heads, kh);
            ksc_s[tid] = ok ? rp.ks[0] : 0.0f;
            vsc_s[tid] = ok ? rp.vs[0] : 0.0f;
        }
        __syncthreads();

        if (warp == 0) {
            float sa[2][4];
#pragma unroll
            for (int f = 0; f < 2; ++f) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sa[f][i] = 0.0f;
                }
            }
#pragma unroll
            for (int kb = 0; kb < 512; kb += 16) {
                const unsigned a0 =
                    *reinterpret_cast<const unsigned*>(&Qs[g][kb + 2 * tt]);
                const unsigned a1 =
                    *reinterpret_cast<const unsigned*>(&Qs[g + 8][kb + 2 * tt]);
                const unsigned a2 =
                    *reinterpret_cast<const unsigned*>(&Qs[g][kb + 2 * tt + 8]);
                const unsigned a3 =
                    *reinterpret_cast<const unsigned*>(&Qs[g + 8][kb + 2 * tt + 8]);
#pragma unroll
                for (int f = 0; f < 2; ++f) {
                    const int j = f * 8 + g;
                    const unsigned b0 =
                        fa2_k8(*reinterpret_cast<const unsigned short*>(&K8[j][kb + 2 * tt]));
                    const unsigned b1 = fa2_k8(
                        *reinterpret_cast<const unsigned short*>(&K8[j][kb + 2 * tt + 8]));
                    fa2_mma(sa[f], a0, a1, a2, a3, b0, b1);
                }
            }
            const int rA = g;
            const int rB = g + 8;
            const int nA = pos0 + rb + rA + 1 - base;
            const int nB = pos0 + rb + rB + 1 - base;
            const int loA = (sw > 0) ? (pos0 + rb + rA + 1 - sw - base) : 0;
            const int loB = (sw > 0) ? (pos0 + rb + rB + 1 - sw - base) : 0;
            const float iqA =
                (qrow[rA] > 30000.0f) ? (qrow[rA] * (1.0f / 30000.0f)) : 1.0f;
            const float iqB =
                (qrow[rB] > 30000.0f) ? (qrow[rB] * (1.0f / 30000.0f)) : 1.0f;
#pragma unroll
            for (int f = 0; f < 2; ++f) {
                const int cc = f * 8 + 2 * tt;
                const float s0 = kq_scale * ksc_s[cc];
                const float s1 = kq_scale * ksc_s[cc + 1];
                sa[f][0] *= s0 * iqA;
                sa[f][1] *= s1 * iqA;
                sa[f][2] *= s0 * iqB;
                sa[f][3] *= s1 * iqB;
                if (cc >= nA) {
                    sa[f][0] = -1e30f;
                }
                if (cc + 1 >= nA) {
                    sa[f][1] = -1e30f;
                }
                if (cc >= nB) {
                    sa[f][2] = -1e30f;
                }
                if (cc + 1 >= nB) {
                    sa[f][3] = -1e30f;
                }
                if (cc < loA) {
                    sa[f][0] = -1e30f;
                }
                if (cc + 1 < loA) {
                    sa[f][1] = -1e30f;
                }
                if (cc < loB) {
                    sa[f][2] = -1e30f;
                }
                if (cc + 1 < loB) {
                    sa[f][3] = -1e30f;
                }
            }
            float mg = -1e30f;
            float mh = -1e30f;
#pragma unroll
            for (int f = 0; f < 2; ++f) {
                mg = fmaxf(mg, fmaxf(sa[f][0], sa[f][1]));
                mh = fmaxf(mh, fmaxf(sa[f][2], sa[f][3]));
            }
            mg = fmaxf(mg, __shfl_xor_sync(0xffffffffu, mg, 1));
            mg = fmaxf(mg, __shfl_xor_sync(0xffffffffu, mg, 2));
            mh = fmaxf(mh, __shfl_xor_sync(0xffffffffu, mh, 1));
            mh = fmaxf(mh, __shfl_xor_sync(0xffffffffu, mh, 2));
            const float moldA = m_s[rA];
            const float moldB = m_s[rB];
            const float mnewA = fmaxf(moldA, mg);
            const float mnewB = fmaxf(moldB, mh);
            const float corrA = (moldA <= -1e29f) ? 0.0f : attn_expf(moldA - mnewA);
            const float corrB = (moldB <= -1e29f) ? 0.0f : attn_expf(moldB - mnewB);
            float sumA = 0.0f;
            float sumB = 0.0f;
#pragma unroll
            for (int f = 0; f < 2; ++f) {
                const int cc = f * 8 + 2 * tt;
                const float vs0 = vsc_s[cc];
                const float vs1 = vsc_s[cc + 1];
                const float p0 = attn_expf(sa[f][0] - mnewA);
                const float p1 = attn_expf(sa[f][1] - mnewA);
                const float p2 = attn_expf(sa[f][2] - mnewB);
                const float p3 = attn_expf(sa[f][3] - mnewB);
                sumA += p0 + p1;
                sumB += p2 + p3;
                const __half2 h01 = __floats2half2_rn(p0 * vs0, p1 * vs1);
                const __half2 h23 = __floats2half2_rn(p2 * vs0, p3 * vs1);
                *reinterpret_cast<unsigned*>(&Ps[rA][cc]) =
                    *reinterpret_cast<const unsigned*>(&h01);
                *reinterpret_cast<unsigned*>(&Ps[rB][cc]) =
                    *reinterpret_cast<const unsigned*>(&h23);
            }
            sumA += __shfl_xor_sync(0xffffffffu, sumA, 1);
            sumA += __shfl_xor_sync(0xffffffffu, sumA, 2);
            sumB += __shfl_xor_sync(0xffffffffu, sumB, 1);
            sumB += __shfl_xor_sync(0xffffffffu, sumB, 2);
            __syncwarp();
            if (tt == 0) {
                m_s[rA] = mnewA;
                m_s[rB] = mnewB;
                l_s[rA] = l_s[rA] * corrA + sumA;
                l_s[rB] = l_s[rB] * corrB + sumB;
                corr_s[rA] = corrA;
                corr_s[rB] = corrB;
            }
        }
        __syncthreads();

        {
            const int rA = g;
            const int rB = g + 8;
            const float corrA = corr_s[rA];
            const float corrB = corr_s[rB];
#pragma unroll
            for (int f = 0; f < 16; ++f) {
                o[f][0] *= corrA;
                o[f][1] *= corrA;
                o[f][2] *= corrB;
                o[f][3] *= corrB;
            }
            const int kb = 0;
            const unsigned a0 =
                *reinterpret_cast<const unsigned*>(&Ps[rA][kb + 2 * tt]);
            const unsigned a1 =
                *reinterpret_cast<const unsigned*>(&Ps[rB][kb + 2 * tt]);
            const unsigned a2 =
                *reinterpret_cast<const unsigned*>(&Ps[rA][kb + 2 * tt + 8]);
            const unsigned a3 =
                *reinterpret_cast<const unsigned*>(&Ps[rB][kb + 2 * tt + 8]);
#pragma unroll
            for (int f = 0; f < 16; ++f) {
                const int d_ = warp * 128 + f * 8 + g;
                const unsigned b0 = *reinterpret_cast<const unsigned*>(
                    &VsT[d_][kb + 2 * tt]);
                const unsigned b1 = *reinterpret_cast<const unsigned*>(
                    &VsT[d_][kb + 2 * tt + 8]);
                fa2_mma(o[f], a0, a1, a2, a3, b0, b1);
            }
        }
        __syncthreads();
    }

    {
        const int rA = g;
        const int rB = g + 8;
        const float invlA = 1.0f / l_s[rA];
        const float invlB = 1.0f / l_s[rB];
        const int rowA = rb + rA;
        const int rowB = rb + rB;
#pragma unroll
        for (int f = 0; f < 16; ++f) {
            const int d_ = warp * 128 + f * 8 + 2 * tt;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int r0 = (e < 2) ? rowA : rowB;
                const int d2 = d_ + (e & 1);
                if (r0 < t_len) {
                    float ov = o[f][e] * ((e < 2) ? invlA : invlB);
                    if (!no_gate) {
                        const float gv = qg[(size_t)r0 * (q_heads * 1024) +
                                            (size_t)h * 1024 + 512 + d2];
                        ov *= 1.0f / (1.0f + attn_expf(-gv));
                    }
                    const long oi = (size_t)r0 * (q_heads * 512) + (size_t)h * 512 + d2;
                    outv[oi] = ov;
                    if (out32 != (float*)0) {
                        const __half hf = __float2half_rn(ov);
                        out32[oi] = __half2float(hf);
                        if (out16 != (unsigned short*)0) {
                            out16[oi] = __half_as_ushort(hf);
                        }
                    } else if (out16 != (unsigned short*)0) {
                        out16[oi] = __half_as_ushort(__float2half_rn(ov));
                    }
                }
            }
        }
    }
}
