#include <cuda_fp16.h>
#include "g4_common.cuh"


extern "C" __global__ void fn_hc_init(const float* __restrict__ emb,
                                      float* __restrict__ r, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int hh = 10240;
    r[i] = emb[(i / hh) * 2560 + (i % hh) % 2560];
}

extern "C" __global__ void fn_hc_norm(const float* __restrict__ r,
                                      const float* __restrict__ w,
                                      float* __restrict__ xn, int t, float eps)
{
    __shared__ float red[256];
    const int tt = blockIdx.x;
    const int tid = threadIdx.x;
    if (tt >= t) return;
    for (int s = 0; s < 4; ++s) {
        const float* x = r + (size_t)tt * 10240 + s * 2560;
        float v = 0.0f;
        #pragma unroll
        for (int j = 0; j < 10; ++j) {
            const float e = x[tid + j * 256];
            v += e * e;
        }
        red[tid] = v;
        __syncthreads();
        for (int st = 128; st > 0; st >>= 1) {
            if (tid < st) red[tid] += red[tid + st];
            __syncthreads();
        }
        const float rrms = rsqrtf(red[0] / 2560.0f + eps);
        float* o = xn + (size_t)tt * 10240 + s * 2560;
        const float* wr = w + s * 2560;
        #pragma unroll
        for (int j = 0; j < 10; ++j) {
            const int i = tid + j * 256;
            const float y = x[i] * rrms;
            o[i] = y * wr[i];
        }
        __syncthreads();
    }
}

extern "C" __global__ void fn_hc_silu(const float* __restrict__ x,
                                      float* __restrict__ y, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float u = x[i] * 0.25f;
    y[i] = u / (1.0f + __expf(-u));
}

extern "C" __global__ void fn_hc_mix(const float* __restrict__ xn,
                                     const float* __restrict__ g,
                                     float* __restrict__ x, int t)
{
    const int tt = blockIdx.x;
    if (tt >= t) return;
    const int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= 2560) return;
    float acc = 0.0f;
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        const size_t off = (size_t)tt * 10240 + s * 2560 + i;
        const float gv = g[off];
        acc += (1.0f / (1.0f + __expf(-gv))) * xn[off];
    }
    x[(size_t)tt * 2560 + i] = acc * 0.25f;
}

extern "C" __global__ void fn_hc_write(float* __restrict__ r,
                                       const float* __restrict__ y,
                                       const float* __restrict__ inj, int t)
{
    const int tt = blockIdx.x;
    if (tt >= t) return;
    const int s = blockIdx.y;
    const float u = inj[tt * 4 + s] * 0.25f;
    const float sc = 2.0f / (1.0f + __expf(-u));
    for (int i = threadIdx.x; i < 2560; i += blockDim.x) {
        r[(size_t)tt * 10240 + s * 2560 + i] += y[(size_t)tt * 2560 + i] * sc;
    }
}


extern "C" __global__ void fn_ple_gate(const float* __restrict__ key,
                                       const float* __restrict__ val,
                                       const float* __restrict__ r,
                                       const float* __restrict__ nk,
                                       const float* __restrict__ nq,
                                       const float* __restrict__ nc,
                                       float* __restrict__ gated,
                                       float* __restrict__ normed,
                                       int t, float eps)
{
    __shared__ float red[256];
    const int tt = blockIdx.x;
    const int s = blockIdx.y;
    const int tid = threadIdx.x;
    if (tt >= t) return;
    const size_t kbase = (size_t)tt * 12800 + s * 2560;
    const size_t base = (size_t)tt * 10240 + s * 2560;
    const size_t wb = s * 2560;
    const float* k = key + kbase;
    const float* q = r + base;
    float kk = 0.0f, qq = 0.0f;
    for (int j = 0; j < 10; ++j) {
        const int i = tid + j * 256;
        const float kv = k[i], qv = q[i];
        kk += kv * kv;
        qq += qv * qv;
    }
    red[tid] = kk;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    const float rk = rsqrtf(red[0] / 2560.0f + eps);
    red[tid] = qq;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    const float rq = rsqrtf(red[0] / 2560.0f + eps);
    float dot = 0.0f;
    #pragma unroll
    for (int j = 0; j < 10; ++j) {
        const int i = tid + j * 256;
        const float kn = k[i] * rk * nk[wb + i];
        const float qn = q[i] * rq * nq[wb + i];
        dot += kn * qn;
    }
    red[tid] = dot;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    const float d = red[0] / sqrtf(2560.0f);
    const float sign = d < 0.0f ? -1.0f : (d > 0.0f ? 1.0f : 0.0f);
    const float g = 1.0f / (1.0f + __expf(-sign * sqrtf(fmaxf(fabsf(d), 1e-6f))));
    float gg = 0.0f;
    const float* wv = val + (size_t)tt * 12800 + 10240;
    #pragma unroll
    for (int j = 0; j < 10; ++j) {
        const int i = tid + j * 256;
        const float gv = g * wv[i];
        gated[base + i] = gv;
        gg += gv * gv;
    }
    red[tid] = gg;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    const float rg = rsqrtf(red[0] / 2560.0f + eps);
    #pragma unroll
    for (int j = 0; j < 10; ++j) {
        const int i = tid + j * 256;
        const float gv = gated[base + i];
        normed[base + i] = gv * rg * nc[wb + i];
    }
}

extern "C" __global__ void fn_ple_conv(const float* __restrict__ normed,
                                       const float* __restrict__ gated,
                                       float* __restrict__ r,
                                       const float* __restrict__ w,
                                       const float* __restrict__ state,
                                       int t, int has_init)
{
    const int tt = blockIdx.x;
    if (tt >= t) return;
    const int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= 10240) return;
    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int jj = tt - 9 + 3 * k;
        float tap;
        if (jj >= 0) {
            tap = normed[(size_t)jj * 10240 + c];
        } else if (has_init) {
            tap = state[(size_t)(9 + jj) * 10240 + c];
        } else {
            tap = 0.0f;
        }
        acc += w[c * 4 + k] * tap;
    }
    const size_t off = (size_t)tt * 10240 + c;
    r[off] += gated[off] + acc / (1.0f + __expf(-acc));
}

extern "C" __global__ void fn_ple_conv_wb(const float* __restrict__ normed,
                                          float* __restrict__ state, int t,
                                          int has_init)
{
    const int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= 10240) return;
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        const int src = t - 9 + i;
        if (src >= 0) {
            state[(size_t)i * 10240 + c] = normed[(size_t)src * 10240 + c];
        } else if (has_init) {
            state[(size_t)i * 10240 + c] = state[(size_t)(i + t) * 10240 + c];
        } else {
            state[(size_t)i * 10240 + c] = 0.0f;
        }
    }
}

extern "C" __global__ void gdn_gate_sig(
    const float* __restrict__ o_lc,
    const float* __restrict__ z,
    const float* __restrict__ nw,
    float* __restrict__ gated,
    float* __restrict__ gated32,
    unsigned short* __restrict__ gated16,
    int t_len, int layer, int h_k, int h_v)
{
    __shared__ float red[128];
    int t = blockIdx.y;
    int h = blockIdx.x;
    int tid = threadIdx.x;
    if (t >= t_len) return;
    int G = h_v / h_k;
    int p_inv = (h % G) * h_k + h / G;
    float ov = o_lc[t * (h_v * 128) + p_inv * 128 + tid];
    red[tid] = ov * ov;
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    float inv = 1.0f / sqrtf(red[0] / 128.0f + 1e-6f);
    float zv = z[t * (h_v * 128) + h * 128 + tid];
    float gv = ov * inv * nw[layer * 128 + tid] * (1.0f / (1.0f + __expf(-zv)));
    gated[t * (h_v * 128) + h * 128 + tid] = gv;
    if (gated32 != (float*)0) {
        const __half hv = __float2half_rn(gv);
        gated32[t * (h_v * 128) + h * 128 + tid] = __half2float(hv);
        if (gated16 != (unsigned short*)0) {
            gated16[t * (h_v * 128) + h * 128 + tid] = __half_as_ushort(hv);
        }
    } else if (gated16 != (unsigned short*)0) {
        gated16[t * (h_v * 128) + h * 128 + tid] = __half_as_ushort(__float2half_rn(gv));
    }
}

#define FN_SCMAX 128


__device__ __forceinline__ void fn_gemv_row_g128(
    const unsigned* __restrict__ q,
    const unsigned short* __restrict__ s,
    const float* __restrict__ x,
    float* __restrict__ out, int o, int k)
{
    if (o < 0) return;
    const int l = threadIdx.x;
    __shared__ double red[64];
    __shared__ float sc[FN_SCMAX];
    const int k8 = k >> 3;
    const int kg = k >> 7;
    for (int g = l; g < kg; g += 64) {
        sc[g] = h2f(s[(size_t)o * kg + g]);
    }
    __syncthreads();
    const unsigned* qrow = q + (size_t)o * k8;
    const int jn = k >> 7;
    const int sh = 8 * (l & 3);
    float acc = 0.0f;
    int j = 0;
    const unsigned* qp = qrow + (l >> 2);
    const float2* xp = reinterpret_cast<const float2*>(x) + l;
    for (; j + 8 <= jn; j += 8) {
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const int jj = j + u;
            const unsigned qw = __ldcs(qp);
            const float2 xv = *xp;
            const unsigned byte = (qw >> sh) & 0xFFu;
            const float scv = sc[jj];
            const float w0 = fmaf((float)(byte & 0xFu), scv, -8.0f * scv);
            const float w1 = fmaf((float)(byte >> 4), scv, -8.0f * scv);
            acc = fmaf(w0, xv.x, acc);
            acc = fmaf(w1, xv.y, acc);
            qp += 16;
            xp += 64;
        }
    }
    for (; j < jn; ++j) {
        const unsigned qw = __ldcs(qp);
        const float2 xv = *xp;
        const unsigned byte = (qw >> sh) & 0xFFu;
        const float scv = sc[j];
        const float w0 = fmaf((float)(byte & 0xFu), scv, -8.0f * scv);
        const float w1 = fmaf((float)(byte >> 4), scv, -8.0f * scv);
        acc = fmaf(w0, xv.x, acc);
        acc = fmaf(w1, xv.y, acc);
        qp += 16;
        xp += 64;
    }
    red[l] = (double)acc;
    __syncthreads();
    if (l < 32) {
        double r = red[l] + red[l + 32];
#pragma unroll
        for (int off = 16; off >= 1; off >>= 1) {
            const double oth = shfl_down_f64(r, off);
            if (l < off) {
                r += oth;
            }
        }
        if (l == 0) {
            out[o] = (float)r;
        }
    }
}

extern "C" __global__ void w4a16_gemv_experts_g128_f16(
    const unsigned long long* __restrict__ tab, int base,
    const int* __restrict__ idx, int nslots,
    const float* __restrict__ x, int xstride, int sp,
    float* __restrict__ out, int n, int k)
{
    const int sl = blockIdx.x / n;
    if (sl >= nslots) {
        return;
    }
    const int o = blockIdx.x - sl * n;
    const unsigned long long* e = tab + (size_t)(base + idx[sl] * 3) * 2;
    const size_t xoff = (size_t)(sl / (sp > 0 ? sp : 1)) * (size_t)xstride;
    fn_gemv_row_g128((const unsigned*)e[0], (const unsigned short*)e[1],
                     x + xoff, out + (size_t)sl * n, o, k);
}

__device__ __forceinline__ void fn_gemv_row_g32_f16(
    const unsigned* __restrict__ q,
    const unsigned short* __restrict__ s,
    const float* __restrict__ x,
    float* __restrict__ out, int o, int k)
{
    if (o < 0) return;
    const int l = threadIdx.x;
    __shared__ double red[64];
    __shared__ float sc[FN_SCMAX];
    const int k8 = k >> 3;
    const int kg = k >> 5;
    for (int g = l; g < kg; g += 64) {
        sc[g] = h2f(s[(size_t)o * kg + g]);
    }
    __syncthreads();
    const unsigned* qrow = q + (size_t)o * k8;
    const int jn = k >> 7;
    const int sh = 8 * (l & 3);
    const int sgrp = l >> 4;
    float acc = 0.0f;
    int j = 0;
    const unsigned* qp = qrow + (l >> 2);
    const float2* xp = reinterpret_cast<const float2*>(x) + l;
    for (; j + 8 <= jn; j += 8) {
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const int jj = j + u;
            const unsigned qw = __ldcs(qp);
            const float2 xv = *xp;
            const unsigned byte = (qw >> sh) & 0xFFu;
            const float scv = sc[4 * jj + sgrp];
            const float w0 = fmaf((float)(byte & 0xFu), scv, -8.0f * scv);
            const float w1 = fmaf((float)(byte >> 4), scv, -8.0f * scv);
            acc = fmaf(w0, xv.x, acc);
            acc = fmaf(w1, xv.y, acc);
            qp += 16;
            xp += 64;
        }
    }
    for (; j < jn; ++j) {
        const unsigned qw = __ldcs(qp);
        const float2 xv = *xp;
        const unsigned byte = (qw >> sh) & 0xFFu;
        const float scv = sc[4 * j + sgrp];
        const float w0 = fmaf((float)(byte & 0xFu), scv, -8.0f * scv);
        const float w1 = fmaf((float)(byte >> 4), scv, -8.0f * scv);
        acc = fmaf(w0, xv.x, acc);
        acc = fmaf(w1, xv.y, acc);
        qp += 16;
        xp += 64;
    }
    red[l] = (double)acc;
    __syncthreads();
    if (l < 32) {
        double r = red[l] + red[l + 32];
#pragma unroll
        for (int off = 16; off >= 1; off >>= 1) {
            const double oth = shfl_down_f64(r, off);
            if (l < off) {
                r += oth;
            }
        }
        if (l == 0) {
            out[o] = (float)r;
        }
    }
}

extern "C" __global__ void w4a16_gemv_experts_g32_f16(
    const unsigned long long* __restrict__ tab, int base,
    const int* __restrict__ idx, int nslots,
    const float* __restrict__ x, int xstride, int sp,
    float* __restrict__ out, int n, int k)
{
    const int sl = blockIdx.x / n;
    if (sl >= nslots) {
        return;
    }
    const int o = blockIdx.x - sl * n;
    const unsigned long long* e = tab + (size_t)(base + idx[sl] * 3) * 2;
    const size_t xoff = (size_t)(sl / (sp > 0 ? sp : 1)) * (size_t)xstride;
    fn_gemv_row_g32_f16((const unsigned*)e[0], (const unsigned short*)e[1],
                        x + xoff, out + (size_t)sl * n, o, k);
}


#define ATTN_CHUNK_QSA 256
struct QsaKvRowPtrs {
    signed char* k;
    signed char* v;
    float* ks;
    float* vs;
};
__device__ __forceinline__ QsaKvRowPtrs qsa_blk_row(
    const unsigned long long* __restrict__ tab, int layer, int cap, int row,
    long kv_dim, int kv_heads, int kh)
{
    const unsigned long long* bt =
        tab + ((long)layer * ((cap + 511) >> 9) + (row >> 9)) * 4;
    const unsigned long long ro = (unsigned long long)(row & 511);
    const long hstride = (kv_heads > 0) ? (kv_dim / kv_heads) : 256;
    QsaKvRowPtrs p;
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
template <typename KDot, typename VAcc>
__device__ __forceinline__ void attn_part_step_qsa(KDot kdot, VAcc vacc,
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
    float corr = (m_run <= -1e29f) ? 0.0f : expf(m_run - m_new);
    __syncthreads();
    float e = 0.0f;
    if (tid < nch) {
        e = expf(sarr[tid] - m_new);
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
    for (int i = 0; i < nch; ++i) {
        acc += sarr[i] * vacc(i);
    }
    l_run = l_run * corr + reds[0];
    m_run = m_new;
    __syncthreads();
}

extern "C" __global__ void attn_qsa_part(
    const float* __restrict__ qh,
    const unsigned long long* __restrict__ dblk_tab,
    float* __restrict__ part,
    const unsigned* __restrict__ pp,
    const int* __restrict__ sel,
    const int* __restrict__ nsel,
    int sel_stride,
    int t_len, int layer, int q_heads, int kv_heads, int cap, float kq_scale)
{
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int gq = q_heads / kv_heads;
    const int kh = h / gq;
    const long kv_dim = (long)kv_heads * 256;
    const long qrow = (long)t * (q_heads * 256) + (long)h * 256;
    __shared__ float qs[256];
    __shared__ float sarr[ATTN_CHUNK_QSA];
    __shared__ float reds[256];
    qs[tid] = qh[qrow + tid];
    __syncthreads();
    const int lim = nsel[t];
    float* out = part + ((long)t * q_heads + h) * 258;
    float m_run = -1e30f;
    float l_run = 0.0f;
    float acc = 0.0f;
    const int* rows = sel + (long)t * sel_stride;
    for (int base = 0; base < lim; base += ATTN_CHUNK_QSA) {
        const int nch = min(ATTN_CHUNK_QSA, lim - base);
        attn_part_step_qsa(
            [&](int i) -> float {
                const int row = rows[base + i];
                if (row < 0) {
                    return -1e30f;
                }
                const QsaKvRowPtrs rp =
                    qsa_blk_row(dblk_tab, layer, cap, row, kv_dim, kv_heads, kh);
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
                return p0 * (kq_scale * rp.ks[0]);
            },
            [&](int i) -> float {
                const int row = rows[base + i];
                if (row < 0) {
                    return 0.0f;
                }
                const QsaKvRowPtrs rp =
                    qsa_blk_row(dblk_tab, layer, cap, row, kv_dim, kv_heads, kh);
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
