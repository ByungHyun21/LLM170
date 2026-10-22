


























































#include <cuda_fp16.h>
#include "cast_common.cuh"













__device__ __forceinline__ double gdn_exp_d(double x)
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










__device__ __forceinline__ float gdn_expf(float x)
{
    return __expf(x);
}

__device__ __forceinline__ double gdn_log_d(double y)
{


    const double ln2 = 6.9314718036912382e-01;
    unsigned long long bits = __double_as_longlong(y);
    int e = (int)((bits >> 52) & 0x7ff) - 1023;
    unsigned long long mb = (bits & 0x800fffffffffffffULL) | 0x3ff0000000000000ULL;
    double m = __longlong_as_double((long long)mb);
    if (m > 1.4142135623730951) { m = m * 0.5; e = e + 1; }
    double s = (m - 1.0) / (m + 1.0);
    double z = s * s;
    double q = 1.0 + z * (0.3333333333333333 + z * (0.2 + z * (0.14285714285714285
        + z * (0.1111111111111111 + z * (0.09090909090909091 + z * (0.07692307692307693
        + z * (0.06666666666666667 + z * (0.058823529411764705 + z * (0.05263157894736842
        + z * (0.047619047619047616 + z * 0.043478260869565216))))))))));
    return 2.0 * s * q + (double)e * ln2;
}


__device__ __forceinline__ float gdn_logf(float y)
{
    return __logf(y);
}








#define CONV_TC 16
extern "C" __global__ void gdn_conv(
    const float* __restrict__ qkv,
    const float* __restrict__ convw,
    float* __restrict__ ring,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    float* __restrict__ ring_snap,

    float* __restrict__ ring_pre,

    const unsigned long long* __restrict__ row_ring,

    int t_len, int layer, int k_len, int v_len, int conv_ch)
{
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= conv_ch) return;
    float w0 = convw[layer * conv_ch * 4 + ch * 4 + 0];
    float w1 = convw[layer * conv_ch * 4 + ch * 4 + 1];
    float w2 = convw[layer * conv_ch * 4 + ch * 4 + 2];
    float w3 = convw[layer * conv_ch * 4 + ch * 4 + 3];


    if (row_ring != (const unsigned long long*)0) {
        const int row = blockIdx.y;
        float* rr = (float*)row_ring[row] + (long)layer * 3 * conv_ch;
        const float xt = qkv[(long)row * conv_ch + ch];
        const float a0 = rr[ch];
        const float a1 = rr[conv_ch + ch];
        const float a2 = rr[2 * conv_ch + ch];
        float o = w3 * xt + w0 * a0 + w1 * a1 + w2 * a2;
        o = o / (1.0f + gdn_expf(-o));
        if (ch < k_len) {
            q_out[(long)row * k_len + ch] = o;
        } else if (ch < 2 * k_len) {
            k_out[(long)row * k_len + (ch - k_len)] = o;
        } else {
            v_out[(long)row * v_len + (ch - 2 * k_len)] = o;
        }
        rr[ch] = a1;
        rr[conv_ch + ch] = a2;
        rr[2 * conv_ch + ch] = xt;
        return;
    }
    const int t0 = blockIdx.y * CONV_TC;
    const int t1 = min(t_len, t0 + CONV_TC);
    float h0, h1, h2;
    if (t0 == 0) {
        h0 = ring[layer * 3 * conv_ch + 0 * conv_ch + ch];
        h1 = ring[layer * 3 * conv_ch + 1 * conv_ch + ch];
        h2 = ring[layer * 3 * conv_ch + 2 * conv_ch + ch];
        if (ring_pre != (float*)0) {


            ring_pre[0 * conv_ch + ch] = h0;
            ring_pre[conv_ch + ch] = h1;
            ring_pre[2 * conv_ch + ch] = h2;
        }
    } else {

        h0 = qkv[(t0 - 3) * conv_ch + ch];
        h1 = qkv[(t0 - 2) * conv_ch + ch];
        h2 = qkv[(t0 - 1) * conv_ch + ch];
    }
    for (int t = t0; t < t1; t++) {
        float xt = qkv[t * conv_ch + ch];
        float o = (w3 * xt + w0 * h0 + w1 * h1 + w2 * h2);
        o = o / (1.0f + gdn_expf(-o));
        if (ch < k_len) {
            q_out[t * k_len + ch] = o;
        } else if (ch < 2 * k_len) {
            k_out[t * k_len + (ch - k_len)] = o;
        } else {
            v_out[t * v_len + (ch - 2 * k_len)] = o;
        }
        h0 = h1; h1 = h2; h2 = xt;
        if (ring_snap != (float*)0) {


            float* rs = ring_snap + ((long)t * 3) * conv_ch + ch;
            rs[0] = h0;
            rs[(long)conv_ch] = h1;
            rs[(long)2 * conv_ch] = h2;
        }
    }
    if (t1 == t_len) {
        ring[layer * 3 * conv_ch + 0 * conv_ch + ch] = h0;
        ring[layer * 3 * conv_ch + 1 * conv_ch + ch] = h1;
        ring[layer * 3 * conv_ch + 2 * conv_ch + ch] = h2;
    }
}














__device__ __forceinline__ void gdn_l2perm_body(
    const float* __restrict__ q_in,
    const float* __restrict__ k_in,
    const float* __restrict__ v_in,
    const float* __restrict__ xn,
    const float* __restrict__ abuf,
    const float* __restrict__ alog,
    const float* __restrict__ dtb,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    float* __restrict__ bg,
    int t_len, int layer, int h_k, int h_v, int hidden)
{
    __shared__ float red[128];
    int t = blockIdx.y;
    int h = blockIdx.x;
    int tid = threadIdx.x;
    if (t >= t_len) return;
    const float eps = 1e-6f;



    float pa = 0.0f;
    float pb = 0.0f;
    const float* abA = abuf + (layer * 2) * h_v * hidden + h * hidden;
    const float* abB = abA + h_v * hidden;
    for (int i = tid; i < hidden; i += 128) {
        const float xv = xn[t * hidden + i];
        pa += xv * abA[i];
        pb += xv * abB[i];
    }
    red[tid] = pa;
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    float a_val = red[0];
    __syncthreads();

    red[tid] = pb;
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    float b_val = red[0];
    __syncthreads();



    int G = h_v / h_k;
    int p_inv = (h % G) * h_k + h / G;
    if (tid == 0) {
        float al = alog[layer * h_v + h];
        float dt = dtb[layer * h_v + h];
        float ssm_a = -gdn_expf(al);
        float sp = (a_val + dt > 20.0f) ? (a_val + dt) : gdn_logf(1.0f + gdn_expf(a_val + dt));
        bg[t * (2 * h_v) + p_inv] = 1.0f / (1.0f + gdn_expf(-b_val));
        bg[t * (2 * h_v) + h_v + p_inv] = sp * ssm_a;
    }

    if (h < h_k) {
        int kh = h;
        float qv = q_in[t * (h_k * 128) + kh * 128 + tid];
        red[tid] = qv * qv;
        __syncthreads();
        for (int st = 64; st > 0; st >>= 1) {
            if (tid < st) red[tid] += red[tid + st];
            __syncthreads();
        }
        float qi = 1.0f / sqrtf(red[0] + eps);
        __syncthreads();
        q_out[t * (h_k * 128) + kh * 128 + tid] = qv * qi;

        float kv = k_in[t * (h_k * 128) + kh * 128 + tid];
        red[tid] = kv * kv;
        __syncthreads();
        for (int st = 64; st > 0; st >>= 1) {
            if (tid < st) red[tid] += red[tid + st];
            __syncthreads();
        }
        float ki = 1.0f / sqrtf(red[0] + eps);
        __syncthreads();
        k_out[t * (h_k * 128) + kh * 128 + tid] = kv * ki;
    }

    v_out[t * (h_v * 128) + p_inv * 128 + tid] =
        v_in[t * (h_v * 128) + h * 128 + tid];
}


extern "C" __global__ void gdn_l2perm(
    const float* __restrict__ q_in,
    const float* __restrict__ k_in,
    const float* __restrict__ v_in,
    const float* __restrict__ xn,
    const float* __restrict__ abuf,
    const float* __restrict__ alog,
    const float* __restrict__ dtb,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    float* __restrict__ bg,
    int t_len, int layer, int h_k, int h_v, int hidden)
{
    gdn_l2perm_body(q_in, k_in, v_in, xn, abuf, alog, dtb,
                    q_out, k_out, v_out, bg, t_len, layer, h_k, h_v, hidden);
}

extern "C" __global__ void gdn_l2perm8(
    const float* __restrict__ q_in,
    const float* __restrict__ k_in,
    const float* __restrict__ v_in,
    const float* __restrict__ xn,
    const float* __restrict__ abuf,
    const float* __restrict__ alog,
    const float* __restrict__ dtb,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    float* __restrict__ bg,
    int t_len, int layer, int h_k, int h_v, int hidden)
{
    extern __shared__ float smem[];
    float* abs_ = smem;
    float* red = smem + 2 * hidden;
    const int h = blockIdx.x;
    const int lane = threadIdx.x & 127;
    const int grp = threadIdx.x >> 7;
    const int t = blockIdx.y * 8 + grp;
    const bool live = t < t_len;
    const int kh = h % h_k;
    const float eps = 1e-6f;
    const float* abA = abuf + (layer * 2) * h_v * hidden + h * hidden;
    const float* abB = abA + h_v * hidden;
    for (int i = threadIdx.x; i < 2 * hidden; i += blockDim.x) {
        abs_[i] = (i < hidden) ? abA[i] : abB[i - hidden];
    }
    __syncthreads();
    float pa = 0.0f;
    float pb = 0.0f;
    for (int i = lane; i < hidden; i += 128) {
        const float xv = live ? xn[(size_t)t * hidden + i] : 0.0f;
        pa += xv * abs_[i];
        pb += xv * abs_[hidden + i];
    }
    red[grp * 128 + lane] = pa;
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (lane < st) {
            red[grp * 128 + lane] += red[grp * 128 + lane + st];
        }
        __syncthreads();
    }
    const float a_val = red[grp * 128];
    __syncthreads();
    red[grp * 128 + lane] = pb;
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (lane < st) {
            red[grp * 128 + lane] += red[grp * 128 + lane + st];
        }
        __syncthreads();
    }
    const float b_val = red[grp * 128];
    if (lane == 0 && live) {
        const int G = h_v / h_k;
        const int p_inv = (h % G) * h_k + h / G;
        const float al = alog[layer * h_v + h];
        const float dt = dtb[layer * h_v + h];
        const float ssm_a = -gdn_expf(al);
        const float sp = (a_val + dt > 20.0f) ? (a_val + dt) : gdn_logf(1.0f + gdn_expf(a_val + dt));
        bg[(size_t)t * (2 * h_v) + p_inv] = 1.0f / (1.0f + gdn_expf(-b_val));
        bg[(size_t)t * (2 * h_v) + h_v + p_inv] = sp * ssm_a;
    }
    if (live) {
        const int G = h_v / h_k;
        const int p_inv = (h % G) * h_k + h / G;
        v_out[(size_t)t * (h_v * 128) + p_inv * 128 + lane] =
            v_in[(size_t)t * (h_v * 128) + h * 128 + lane];
    }
    if (h < h_k) {
        const int qi_off = (size_t)t * (h_k * 128) + kh * 128 + lane;
        const float qv = live ? q_in[qi_off] : 0.0f;
        red[grp * 128 + lane] = qv * qv;
        __syncthreads();
        for (int st = 64; st > 0; st >>= 1) {
            if (lane < st) {
                red[grp * 128 + lane] += red[grp * 128 + lane + st];
            }
            __syncthreads();
        }
        const float qi = 1.0f / sqrtf(red[grp * 128] + eps);
        if (live) {
            q_out[qi_off] = qv * qi;
        }
        __syncthreads();
        const float kv = live ? k_in[qi_off] : 0.0f;
        red[grp * 128 + lane] = kv * kv;
        __syncthreads();
        for (int st = 64; st > 0; st >>= 1) {
            if (lane < st) {
                red[grp * 128 + lane] += red[grp * 128 + lane + st];
            }
            __syncthreads();
        }
        const float ki = 1.0f / sqrtf(red[grp * 128] + eps);
        if (live) {
            k_out[qi_off] = kv * ki;
        }
    }
}


#define GDN_CS 32
#define GDN_TILE 16
#define GDN_NSPLIT 4
#define GDN_VS (128 / GDN_NSPLIT)
#define GDN_NGRP 8

extern "C" __global__ void gdn_scan_akq(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ bg,
    __half* __restrict__ akq,
    int t_len, int h_k, int h_v, int d)
{
    const int h = blockIdx.x;
    const int c = blockIdx.y;
    const int t0 = c * GDN_CS;
    const int n = min(t_len - t0, GDN_CS);
    const int kh = h % h_k;
    __shared__ __half sk[GDN_CS * 128];
    __shared__ __half qh[GDN_CS * 128];
    __shared__ float qs[GDN_CS * 128];
    __shared__ float gcs[GDN_CS + 1];
    __shared__ float bp[GDN_CS];
    for (int e = threadIdx.x; e < GDN_CS * 128; e += blockDim.x) {
        int t = e / 128, dv = e % 128;
        bool live = t < n;
        sk[e] = __float2half_rn(live ? k[(t0 + t) * (h_k * d) + kh * 128 + dv] : 0.0f);
        qs[e] = live ? q[(t0 + t) * (h_k * d) + kh * 128 + dv] : 0.0f;
        qh[e] = __float2half_rn(qs[e]);
    }
    if (threadIdx.x < GDN_CS) {
        float acc = 0.0f;
        for (int t = 0; t < GDN_CS; t++) {
            acc += (t < n) ? bg[(t0 + t) * (2 * h_v) + h_v + h] : 0.0f;
            gcs[t] = acc;
        }
        gcs[GDN_CS] = acc;
        bp[threadIdx.x] = (threadIdx.x < n) ? bg[(t0 + threadIdx.x) * (2 * h_v) + h] : 0.0f;
    }
    __syncthreads();
    const float qscale = 1.0f / sqrtf((float)d);
    const size_t plane = (size_t)h_v * gridDim.y * GDN_CS * GDN_CS;
    __half* Aout = akq + (size_t)h * gridDim.y * GDN_CS * GDN_CS + (size_t)c * GDN_CS * GDN_CS;
    __half* KQout = Aout + plane;



    {
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        if (warp < 8) {
            const int mt = warp >> 2, nt = warp & 3;
            float ca0 = 0.f, ca1 = 0.f, ca2 = 0.f, ca3 = 0.f;
            float cq0 = 0.f, cq1 = 0.f, cq2 = 0.f, cq3 = 0.f;
            const int a_row = (lane & 7) + 8 * ((lane >> 3) & 1);
            const int a_col = 8 * (lane >> 4);
            const int b_row = lane & 7;
            const int b_tile = (lane >> 3) & 1;
            for (int ks = 0; ks < 8; ks++) {
                unsigned a0, a1, a2, a3, q0, q1, q2, q3, b0, b1;
                const __half* kp = &sk[(mt * 16 + a_row) * 128 + (ks * 16 + a_col)];
                const __half* qp = &qh[(mt * 16 + a_row) * 128 + (ks * 16 + a_col)];
                const __half* bp2 = &sk[(nt * 8 + b_row) * 128 + (ks * 16 + 8 * b_tile)];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                             "{%0,%1,%2,%3}, [%4];\n"
                             : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                             : "l"(__cvta_generic_to_shared(kp)));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                             "{%0,%1,%2,%3}, [%4];\n"
                             : "=r"(q0), "=r"(q1), "=r"(q2), "=r"(q3)
                             : "l"(__cvta_generic_to_shared(qp)));
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
                             "{%0,%1}, [%2];\n"
                             : "=r"(b0), "=r"(b1)
                             : "l"(__cvta_generic_to_shared(bp2)));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+f"(ca0), "+f"(ca1), "+f"(ca2), "+f"(ca3)
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+f"(cq0), "+f"(cq1), "+f"(cq2), "+f"(cq3)
                             : "r"(q0), "r"(q1), "r"(q2), "r"(q3), "r"(b0), "r"(b1));
            }
            const int r0 = mt * 16 + (lane >> 2);
            const int cbase = nt * 8 + 2 * (lane & 3);
            const float cvA[4] = {ca0, ca1, ca2, ca3};
            const float cvQ[4] = {cq0, cq1, cq2, cq3};
            const int rrow[4] = {r0, r0, r0 + 8, r0 + 8};
            const int ccol[4] = {cbase, cbase + 1, cbase, cbase + 1};
#pragma unroll
            for (int q4 = 0; q4 < 4; q4++) {
                const int i = rrow[q4];
                const int j = ccol[q4];
                if (i >= n) continue;
                const float gi = gcs[i], gj = gcs[j];
                const float w = gdn_expf(gi - gj);
                Aout[i * GDN_CS + j] = __float2half_rn(
                    (j < i) ? cvA[q4] * bp[i] * w : 0.0f);
                KQout[i * GDN_CS + j] = __float2half_rn(
                    (j <= i) ? cvQ[q4] * qscale * w : 0.0f);
            }
        }
    }

}













extern "C" __global__ void gdn_wu(
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ bg,
    const __half* __restrict__ akq,
    float* __restrict__ wubuf,
    int t_len, int h_k, int h_v, int d)
{
    extern __shared__ char smem_raw[];
    __half* A = (__half*)smem_raw;
    float* R = (float*)(A + GDN_CS * GDN_CS);
    float* pe = R + GDN_CS * 2 * d;
    float* pb = pe + GDN_CS;
    const int n_chunks = (t_len + GDN_CS - 1) / GDN_CS;
    const int c = blockIdx.x, h = blockIdx.y;
    const int kh = h % h_k;
    const int col = threadIdx.x;
    const int t0 = c * GDN_CS;
    const int n = min(t_len - t0, GDN_CS);
    {
        const __half* Ag = akq + (size_t)h * n_chunks * GDN_CS * GDN_CS +
                           (size_t)c * GDN_CS * GDN_CS;
        for (int e4 = threadIdx.x; e4 < GDN_CS * GDN_CS / 4; e4 += blockDim.x) {
            reinterpret_cast<uint2*>(A)[e4] = reinterpret_cast<const uint2*>(Ag)[e4];
        }
    }
    if (threadIdx.x < GDN_CS) {
        float acc = 0.0f;
        for (int t = 0; t < GDN_CS; t++) {
            acc += (t < n) ? bg[(t0 + t) * (2 * h_v) + h_v + h] : 0.0f;
            pe[t] = acc;
        }
        for (int t = 0; t < GDN_CS; t++) pe[t] = (t < n) ? gdn_expf(pe[t]) : 0.0f;
        pb[threadIdx.x] = (threadIdx.x < n) ? bg[(t0 + threadIdx.x) * (2 * h_v) + h] : 0.0f;
    }
    __syncthreads();
    for (int j = 0; j < GDN_CS; j++) {
        float r = 0.0f;
        if (j < n) {
            if (col < d)
                r = pb[j] * pe[j] * k[(t0 + j) * (h_k * d) + kh * 128 + col];
            else
                r = pb[j] * v[(t0 + j) * (h_v * d) + h * 128 + (col - d)];
        }
        R[j * 2 * d + col] = r;
    }
    __syncthreads();
    for (int i = 0; i < n; i++) {
        float r = R[i * 2 * d + col];
#pragma unroll
        for (int j = 0; j < GDN_CS; j++) {
            if (j < i) r -= __half2float(A[i * GDN_CS + j]) * R[j * 2 * d + col];
        }
        R[i * 2 * d + col] = r;
    }
    for (int i = 0; i < n; i++) {
        wubuf[(((long)h * n_chunks + c) * GDN_CS + i) * (2 * d) + col] =
            R[i * 2 * d + col];
    }
}





extern "C" __global__ void __launch_bounds__(512, 2) gdn_h(
    const float* __restrict__ k,
    const float* __restrict__ bg,
    const float* __restrict__ wubuf,
    __half* __restrict__ st,
    const float* __restrict__ sc,
    __half* __restrict__ hbuf,
    float* __restrict__ dcbuf,
    int t_len, int h_k, int h_v, int d, int layer)
{
    extern __shared__ char smem_raw[];
    __half* sk = (__half*)smem_raw;
    __half* Wp = sk + GDN_CS * 128;
    float* Ul = (float*)(Wp + GDN_CS * 128);
    __half* Stile = (__half*)(Ul + GDN_CS * GDN_VS);
    __half* Ssc = Stile + 128 * GDN_VS;
    float* dc = (float*)(Ssc + 128 * GDN_VS);
    float* gcs = dc + GDN_CS * GDN_VS;
    float* wsm = gcs + (GDN_CS + 1);
    int h = blockIdx.x / GDN_NSPLIT;
    int vs = (blockIdx.x % GDN_NSPLIT) * GDN_VS;
    int kh = h % h_k;
    int tid = threadIdx.x % GDN_VS;
    int grp = threadIdx.x / GDN_VS;
    int n_chunks = (t_len + GDN_CS - 1) / GDN_CS;
    long st_h = (long)layer * h_v * d * d + (long)h * d * d + vs;
    const long sc_h = ((long)layer * h_v + h) * d;
    const long hb = ((long)h * n_chunks) * d * d + vs;
    const long wb = (long)h * n_chunks * GDN_CS * 2 * d;
    const long dcb = ((long)h * n_chunks) * GDN_CS * d + vs;

    for (int c = 0; c < n_chunks; c++) {
        int t0 = c * GDN_CS;
        int n = min(t_len - t0, GDN_CS);
        for (int e4 = threadIdx.x; e4 < GDN_CS * 32; e4 += blockDim.x) {
            int t = e4 / 32, dv4 = (e4 % 32) * 4;
            bool live = t < n;
            float4 kv = live ? *reinterpret_cast<const float4*>(
                                  &k[(t0 + t) * (h_k * d) + kh * 128 + dv4])
                             : make_float4(0.f, 0.f, 0.f, 0.f);
            *reinterpret_cast<__half2*>(&sk[t * 128 + dv4]) =
                __floats2half2_rn(kv.x, kv.y);
            *reinterpret_cast<__half2*>(&sk[t * 128 + dv4 + 2]) =
                __floats2half2_rn(kv.z, kv.w);
        }
        for (int q = threadIdx.x; q < GDN_CS * 128; q += blockDim.x) {
            const int j = q / 128, r = q % 128;



            Wp[q] = __float2half_rn(wubuf[wb + (long)c * GDN_CS * 2 * d + (long)j * 2 * d + r]);
        }
        for (int q = threadIdx.x; q < GDN_CS * GDN_VS; q += blockDim.x) {
            const int j = q / GDN_VS, t2 = q % GDN_VS;
            Ul[q] = wubuf[wb + (long)c * GDN_CS * 2 * d + (long)j * 2 * d + d + vs + t2];
        }
        if (threadIdx.x < GDN_CS) {
            float acc = 0.0f;
            for (int t = 0; t < GDN_CS; t++) {
                acc += (t < n) ? bg[(t0 + t) * (2 * h_v) + h_v + h] : 0.0f;
                gcs[t] = acc;
            }
            gcs[GDN_CS] = acc;
        }
        __syncthreads();
        if (c == 0) {
            for (int e = threadIdx.x; e < 128 * (GDN_VS / 8); e += blockDim.x) {
                const int r = e / (GDN_VS / 8);
                const int u = e % (GDN_VS / 8);
                *reinterpret_cast<uint4*>(&Stile[r * GDN_VS + u * 8]) =
                    *reinterpret_cast<const uint4*>(&st[st_h + (long)r * d + u * 8]);
            }
            __syncthreads();
        }
        for (int e = threadIdx.x; e < 128 * (GDN_VS / 8); e += blockDim.x) {
            const int r = e / (GDN_VS / 8);
            const int u = e % (GDN_VS / 8);
            *reinterpret_cast<uint4*>(&hbuf[hb + (long)c * d * d + (long)r * d + u * 8]) =
                *reinterpret_cast<const uint4*>(&Stile[r * GDN_VS + u * 8]);
        }


        for (int q = threadIdx.x; q < 128 * GDN_VS; q += blockDim.x) {
            const int r = q / GDN_VS;
            Ssc[q] = __float2half_rn(__half2float(Stile[q]) * sc[sc_h + r]);
        }
        __syncthreads();



        {
            constexpr int IK = (GDN_CS + GDN_NGRP - 1) / GDN_NGRP;
            const int lane = threadIdx.x & 31;
            const int warp = threadIdx.x >> 5;
            const int mt = warp >> 2;
            const int nt = warp & 3;
            float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
            const int a_row = (lane & 7) + 8 * ((lane >> 3) & 1);
            const int a_col = 8 * (lane >> 4);
            const int b_row = (lane & 7) + 8 * (lane >> 3);
            for (int ks = 0; ks < 8; ks++) {
                unsigned a0, a1, a2, a3, b0, b1;
                const __half* ap = &Wp[(mt * 16 + a_row) * 128 + (ks * 16 + a_col)];
                const __half* bp = &Ssc[(ks * 16 + b_row) * GDN_VS + nt * 8];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                             "{%0,%1,%2,%3}, [%4];\n"
                             : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                             : "l"(__cvta_generic_to_shared(ap)));
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                             "{%0,%1}, [%2];\n"
                             : "=r"(b0), "=r"(b1)
                             : "l"(__cvta_generic_to_shared(bp)));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
            const int r0 = mt * 16 + (lane >> 2);
            const int cbase = nt * 8 + 2 * (lane & 3);
            const float cv[4] = {c0, c1, c2, c3};
            const int rrow[4] = {r0, r0, r0 + 8, r0 + 8};
            const int ccol[4] = {cbase, cbase + 1, cbase, cbase + 1};
#pragma unroll
            for (int q4 = 0; q4 < 4; q4++) {
                const int i = rrow[q4];
                const int cc = ccol[q4];
                dc[i * GDN_VS + cc] = (i < n) ? (Ul[i * GDN_VS + cc] - cv[q4]) : 0.0f;
            }



            __syncthreads();
            for (int kk = 0; kk < IK; ++kk) {
                const int i = grp + kk * GDN_NGRP;
                if (i < n)
                    dcbuf[dcb + (long)c * GDN_CS * d + (long)i * d + tid] = dc[i * GDN_VS + tid];
            }
        }
        if (threadIdx.x < GDN_CS) {
            const float gtot0 = gcs[GDN_CS];
            wsm[threadIdx.x] = (threadIdx.x < n) ? gdn_expf(gtot0 - gcs[threadIdx.x]) : 0.0f;
        }
        __syncthreads();
        {
            float gtot = gcs[GDN_CS];
            float gt_exp = gdn_expf(gtot);
            float dcr[GDN_CS];
#pragma unroll
            for (int j = 0; j < GDN_CS; ++j) dcr[j] = dc[j * GDN_VS + tid];
            for (int s2 = grp; s2 < 128; s2 += GDN_NGRP) {
                const float s_i = sc[sc_h + s2];
                float acc = __half2float(Stile[s2 * GDN_VS + tid]) * s_i * gt_exp;
                for (int j = 0; j < n; j++)
                    acc += __half2float(sk[j * 128 + s2]) * wsm[j] * dcr[j];
                float y = acc / s_i;
                y = fminf(fmaxf(y, -65504.0f), 65504.0f);
                Stile[s2 * GDN_VS + tid] = __float2half_rn(y);
            }
            __syncthreads();
        }
    }
    for (int e = threadIdx.x; e < 128 * (GDN_VS / 8); e += blockDim.x) {
        const int r = e / (GDN_VS / 8);
        const int u = e % (GDN_VS / 8);
        *reinterpret_cast<uint4*>(&st[st_h + (long)r * d + u * 8]) =
            *reinterpret_cast<const uint4*>(&Stile[r * GDN_VS + u * 8]);
    }
}



#define GDN_NGRP_O 8
extern "C" __global__ void gdn_o(
    const float* __restrict__ q,
    const float* __restrict__ bg,
    const __half* __restrict__ akq,
    const float* __restrict__ dcbuf,
    const __half* __restrict__ hbuf,
    const float* __restrict__ sc,
    float* __restrict__ outv,
    int t_len, int h_k, int h_v, int d, int layer)
{
    extern __shared__ char smem_raw[];
    __half* KQ = (__half*)smem_raw;
    float* dc = (float*)(KQ + GDN_CS * GDN_CS);
    __half* Stile = (__half*)(dc + GDN_CS * d);
    __half* qt = Stile + 128 * d;
    float* gcs = (float*)(qt + GDN_CS * d);
    float* e = gcs + (GDN_CS + 1);
    const int n_chunks = (t_len + GDN_CS - 1) / GDN_CS;


    const int c = blockIdx.x / h_v;
    const int h = blockIdx.x % h_v;
    const int vs = 0;
    const int kh = h % h_k;
    const int tid = threadIdx.x % GDN_VS;
    const int grp = threadIdx.x / GDN_VS;
    const int t0 = c * GDN_CS;
    const int n = min(t_len - t0, GDN_CS);
    const float qscale = 1.0f / sqrtf((float)d);
    const long hb = ((long)h * n_chunks) * d * d + vs;
    const long dcb = ((long)h * n_chunks) * GDN_CS * d + vs;
    const long sc_h = ((long)layer * h_v + h) * d;

    {
        const size_t plane = (size_t)h_v * n_chunks * GDN_CS * GDN_CS;
        const __half* KQg = akq + plane +
            (size_t)h * n_chunks * GDN_CS * GDN_CS + (size_t)c * GDN_CS * GDN_CS;
        for (int e4 = threadIdx.x; e4 < GDN_CS * GDN_CS / 4; e4 += blockDim.x) {
            reinterpret_cast<uint2*>(KQ)[e4] = reinterpret_cast<const uint2*>(KQg)[e4];
        }
    }
    for (int q2 = threadIdx.x; q2 < GDN_CS * d; q2 += blockDim.x) {
        const int j = q2 / d, t2 = q2 % d;
        dc[q2] = dcbuf[dcb + (long)c * GDN_CS * d + (long)j * d + t2];
    }
    for (int q2 = threadIdx.x; q2 < 128 * (d / 8); q2 += blockDim.x) {
        const int r = q2 / (d / 8);
        const int u = q2 % (d / 8);
        *reinterpret_cast<uint4*>(&Stile[r * d + u * 8]) =
            *reinterpret_cast<const uint4*>(&hbuf[hb + (long)c * d * d + (long)r * d + u * 8]);
    }
    if (threadIdx.x < GDN_CS) {
        float acc = 0.0f;
        for (int t = 0; t < GDN_CS; t++) {
            acc += (t < n) ? bg[(t0 + t) * (2 * h_v) + h_v + h] : 0.0f;
            gcs[t] = acc;
        }
        gcs[GDN_CS] = acc;
    }
    __syncthreads();
    if (threadIdx.x < GDN_CS) {
        e[threadIdx.x] = (threadIdx.x < n) ? gdn_expf(gcs[threadIdx.x]) : 0.0f;
    }

    for (int q2 = threadIdx.x; q2 < GDN_CS * d; q2 += blockDim.x) {
        const int iq = q2 / d, rq = q2 % d;
        float qv = (iq < n) ? q[(t0 + iq) * (h_k * d) + kh * 128 + rq] : 0.0f;
        qt[iq * d + rq] = __float2half_rn(qv * qscale * sc[sc_h + rq]);
    }
__syncthreads();

    constexpr int IK_O = (GDN_CS + GDN_NGRP_O - 1) / GDN_NGRP_O;



    {
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int mt = warp >> 4;
        const int nt = warp & 15;
        float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
        const int a_row = (lane & 7) + 8 * ((lane >> 3) & 1);
        const int a_col = 8 * (lane >> 4);
        const int b_row = (lane & 7) + 8 * (lane >> 3);
        for (int ks = 0; ks < 8; ks++) {
            unsigned a0, a1, a2, a3, b0, b1;
            const __half* ap = &qt[(mt * 16 + a_row) * d + (ks * 16 + a_col)];
            const __half* bp = &Stile[(ks * 16 + b_row) * d + nt * 8];
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                         "{%0,%1,%2,%3}, [%4];\n"
                         : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                         : "l"(__cvta_generic_to_shared(ap)));
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                         "{%0,%1}, [%2];\n"
                         : "=r"(b0), "=r"(b1)
                         : "l"(__cvta_generic_to_shared(bp)));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                         : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
        const int r0 = mt * 16 + (lane >> 2);
        const int cbase = nt * 8 + 2 * (lane & 3);
        const float cv[4] = {c0, c1, c2, c3};
        const int rrow[4] = {r0, r0, r0 + 8, r0 + 8};
        const int ccol[4] = {cbase, cbase + 1, cbase, cbase + 1};
#pragma unroll
        for (int q4 = 0; q4 < 4; q4++) {
            const int i = rrow[q4];
            const int cc = ccol[q4];
            if (i >= n) continue;
            float oi = e[i] * cv[q4];
#pragma unroll
            for (int p = 0; p < GDN_CS; p++) {
                if (p <= i) {
                    float w = __half2float(KQ[i * GDN_CS + p]);
                    oi += w * dc[p * d + cc];
                }
            }
            outv[(t0 + i) * (h_v * d) + h * 128 + vs + cc] = oi;
        }
    }

}












#define GDN1_SPLIT 8





























extern "C" __global__ void gdn_rescale(
    __half* __restrict__ st, float* __restrict__ sc,
    int h_v, int d, int layer0, float head)
{
    const int h = blockIdx.x;
    const int layer = layer0 + (int)blockIdx.y;
    const int i = threadIdx.x;
    const long st_h = (long)layer * h_v * d * d + (long)h * d * d;
    const long sc_h = ((long)layer * h_v + h) * d;
    __half* row = &st[st_h + (long)i * d];
    const float s = sc[sc_h + i];
    float mx = 0.0f;




    const ulonglong2* rv = reinterpret_cast<const ulonglong2*>(row);
    #pragma unroll
    for (int jv = 0; jv < d / 8; ++jv) {
        const ulonglong2 w = rv[jv];
        const __half* hh = reinterpret_cast<const __half*>(&w);
        #pragma unroll
        for (int k2 = 0; k2 < 8; ++k2) {
            const float a = fabsf(__half2float(hh[k2]) * s);
            if (a > mx) mx = a;
        }
    }
    if (mx > 0.0f) {




        const float sm = mx / s;
        if (sm > head * 8.0f || sm < head / 8.0f) {
            float ns = mx / head;
            if (ns < 1e-30f) ns = 1e-30f;
            const float r = s / ns;
            for (int j = 0; j < d; ++j) {
                row[j] = __float2half_rn(__half2float(row[j]) * r);
            }
            sc[sc_h + i] = ns;
        }
    }
}

extern "C" __global__ void gdn1_part(
    const float* __restrict__ q, const float* __restrict__ k,
    const __half* __restrict__ st, const float* __restrict__ sc,
    float* __restrict__ part,
    const unsigned long long* __restrict__ row_st,
    const unsigned long long* __restrict__ row_sc,
    int h_k, int h_v, int d, int layer)
{
    const int h = blockIdx.x;
    int s = blockIdx.y;
    const int tid = threadIdx.x;
    const int kh = h % h_k;


    if (row_st != (const unsigned long long*)0) {
        const int row = blockIdx.y / GDN1_SPLIT;
        s = blockIdx.y - row * GDN1_SPLIT;
        st = (const __half*)row_st[row];
        sc = (const float*)row_sc[row];
        q = q + (long)row * h_k * 128;
        k = k + (long)row * h_k * 128;
        part = part + (long)row * h_v * GDN1_SPLIT * 256;
    }
    const int ch = 128 / GDN1_SPLIT;
    const int i0 = s * ch;
    const long st_h = (long)layer * h_v * d * d + (long)h * d * d;
    const long sc_h = ((long)layer * h_v + h) * d;
    float ks = 0.0f, qs = 0.0f;
    for (int ii = 0; ii < ch; ++ii) {
        const int i = i0 + ii;
        const float stv = __half2float(st[st_h + (long)i * d + tid]) * sc[sc_h + i];
        ks += __half2float(__float2half_rn(k[kh * 128 + i])) * stv;
        qs += q[kh * 128 + i] * stv;
    }
    float* p = part + ((long)h * GDN1_SPLIT + s) * 256;
    p[tid] = ks;
    p[128 + tid] = qs;
}

extern "C" __global__ void gdn1_comb(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ bg,
    const float* __restrict__ part, float* __restrict__ dc,
    float* __restrict__ outv, int h_k, int h_v, int d)
{
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int kh = h % h_k;


    const int row = blockIdx.y;
    q = q + (long)row * h_k * 128;
    k = k + (long)row * h_k * 128;
    v = v + (long)row * h_v * 128;
    bg = bg + (long)row * 2 * h_v;
    part = part + (long)row * h_v * GDN1_SPLIT * 256;
    dc = dc + (long)row * h_v * 128;
    outv = outv + (long)row * h_v * 128;
    float ks = 0.0f, qs = 0.0f;
    for (int s = 0; s < GDN1_SPLIT; ++s) {
        const float* p = part + ((long)h * GDN1_SPLIT + s) * 256;
        ks += p[tid];
        qs += p[128 + tid];
    }
    __shared__ float red[128];
    red[tid] = q[kh * 128 + tid] * __half2float(__float2half_rn(k[kh * 128 + tid]));
    __syncthreads();
    for (int stp = 64; stp > 0; stp >>= 1) {
        if (tid < stp) {
            red[tid] += red[tid + stp];
        }
        __syncthreads();
    }
    const float qscale = 1.0f / sqrtf((float)d);
    const float kq00 = red[0] * qscale;
    const float g = bg[h_v + h];
    const float beta = bg[h];
    const float e = gdn_expf(g);
    const float sv = __half2float(__float2half_rn(v[h * 128 + tid]));
    const float rhs = beta * (sv - e * ks);
    dc[h * 128 + tid] = rhs;
    outv[h * 128 + tid] = e * qs * qscale + kq00 * rhs;
}

extern "C" __global__ void gdn1_upd(
    const float* __restrict__ k, const float* __restrict__ bg,
    const float* __restrict__ dc, __half* __restrict__ st,
    const float* __restrict__ sc,
    const unsigned long long* __restrict__ row_st,
    const unsigned long long* __restrict__ row_sc,
    int h_k, int h_v, int d, int layer)
{
    const int h = blockIdx.x;
    int s = blockIdx.y;
    const int tid = threadIdx.x;
    const int kh = h % h_k;

    if (row_st != (const unsigned long long*)0) {
        const int row = blockIdx.y / GDN1_SPLIT;
        s = blockIdx.y - row * GDN1_SPLIT;
        st = (__half*)row_st[row];
        sc = (const float*)row_sc[row];
        k = k + (long)row * h_k * 128;
        bg = bg + (long)row * 2 * h_v;
        dc = dc + (long)row * h_v * 128;
    }
    const int ch = 128 / GDN1_SPLIT;
    const int i0 = s * ch;
    const long st_h = (long)layer * h_v * d * d + (long)h * d * d;
    const long sc_h = ((long)layer * h_v + h) * d;
    const float e = gdn_expf(bg[h_v + h]);
    const float dcv = dc[h * 128 + tid];
    for (int ii = 0; ii < ch; ++ii) {
        const int i = i0 + ii;
        const float kv = __half2float(__float2half_rn(k[kh * 128 + i]));
        const float s_i = sc[sc_h + i];
        __half* p = &st[st_h + (long)i * d + tid];
        float y = __half2float(*p) * e + (kv / s_i) * dcv;
        y = fminf(fmaxf(y, -65504.0f), 65504.0f);
        *p = __float2half_rn(y);
    }
}











extern "C" __global__ void gdn_spec_scan(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ bg,
    __half* __restrict__ st,
    const float* __restrict__ sc,
    float* __restrict__ outv,
    int t_len, int h_k, int h_v, int d, int layer)
{
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int kh = h % h_k;
    const long st_h = (long)layer * h_v * d * d + (long)h * d * d;
    const long sc_h = ((long)layer * h_v + h) * d;
    __shared__ float red[128];
    const float qscale = 1.0f / sqrtf((float)d);
    for (int t = 0; t < t_len; ++t) {
        const float* qt = q + (long)t * (h_k * 128);
        const float* kt = k + (long)t * (h_k * 128);

        float ks = 0.0f, qs = 0.0f;
        for (int s = 0; s < GDN1_SPLIT; ++s) {
            const int i0 = s * (128 / GDN1_SPLIT);
            float pks = 0.0f, pqs = 0.0f;
            for (int ii = 0; ii < 128 / GDN1_SPLIT; ++ii) {
                const int i = i0 + ii;
                const float stv = __half2float(st[st_h + (long)i * d + tid]) * sc[sc_h + i];
                pks += __half2float(__float2half_rn(kt[kh * 128 + i])) * stv;
                pqs += qt[kh * 128 + i] * stv;
            }
            ks += pks;
            qs += pqs;
        }

        red[tid] = qt[kh * 128 + tid] * __half2float(__float2half_rn(kt[kh * 128 + tid]));
        __syncthreads();
        for (int stp = 64; stp > 0; stp >>= 1) {
            if (tid < stp) {
                red[tid] += red[tid + stp];
            }
            __syncthreads();
        }
        const float kq00 = red[0] * qscale;
        const float g = bg[t * (2 * h_v) + h_v + h];
        const float beta = bg[t * (2 * h_v) + h];
        const float e = gdn_expf(g);
        const float sv = __half2float(__float2half_rn(v[t * (h_v * 128) + h * 128 + tid]));
        const float rhs = beta * (sv - e * ks);
        outv[t * (h_v * 128) + h * 128 + tid] = e * qs * qscale + kq00 * rhs;

        __half* sst = st + st_h;
        for (int i = 0; i < 128; ++i) {
            const float kv = __half2float(__float2half_rn(kt[kh * 128 + i]));
            const float s_i = sc[sc_h + i];
            __half* p = &sst[(long)i * d + tid];
            float y = __half2float(*p) * e + (kv / s_i) * rhs;
            y = fminf(fmaxf(y, -65504.0f), 65504.0f);
            *p = __float2half_rn(y);
        }
        __syncthreads();
    }
}



















extern "C" __global__ void gdn_wy_solve(
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ bg,
    const __half* __restrict__ st,
    const float* __restrict__ sc,
    float* __restrict__ wy_s0,
    float* __restrict__ wy_dc,
    float* __restrict__ wy_g,
    float* __restrict__ wy_k,
    int t_len, int h_k, int h_v, int d, int layer)
{
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int kh = h % h_k;
    const int k_len = h_k * 128;
    const long st_h = (long)layer * h_v * d * d + (long)h * d * d;
    const long sc_h = ((long)layer * h_v + h) * d;
    __shared__ float G[8];
    __shared__ float A[8][8];




    if (h < h_k) {
        for (int t = 0; t < t_len; ++t) {
            const long ix = kh * 128 + tid;
            wy_k[((long)layer * 8 + t) * k_len + ix] =
                __half2float(__float2half_rn(k[(long)t * k_len + ix]));
        }
    }

    if (tid == 0) {
        float acc = 0.0f;
        for (int t = 0; t < t_len; ++t) {
            acc += bg[(long)t * (2 * h_v) + h_v + h];
            G[t] = acc;
            wy_g[((long)layer * 8 + t) * h_v + h] = acc;
        }
    }
    __syncthreads();

    for (int e = tid; e < t_len * t_len; e += 128) {
        const int i = e / t_len, j = e % t_len;
        if (j < i) {
            float gr = 0.0f;
            for (int s2 = 0; s2 < 128; ++s2) {
                gr += __half2float(__float2half_rn(k[(long)i * k_len + kh * 128 + s2]))
                    * __half2float(__float2half_rn(k[(long)j * k_len + kh * 128 + s2]));
            }
            A[i][j] = bg[(long)i * (2 * h_v) + h] * gr * gdn_expf(G[i] - G[j]);
        } else {
            A[i][j] = 0.0f;
        }
    }
    __syncthreads();


    for (int i2 = 0; i2 < 128; ++i2) {
        wy_s0[st_h + (long)i2 * d + tid] =
            __half2float(st[st_h + (long)i2 * d + tid]) * sc[sc_h + i2];
    }

    float dcr[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (i >= t_len) break;
        const float beta = bg[(long)i * (2 * h_v) + h];
        const float di = gdn_expf(G[i]);
        float ks0 = 0.0f;
        for (int s2 = 0; s2 < 128; ++s2) {
            ks0 += __half2float(__float2half_rn(k[(long)i * k_len + kh * 128 + s2]))
                 * (__half2float(st[st_h + (long)s2 * d + tid]) * sc[sc_h + s2]);
        }
        const float sv =
            __half2float(__float2half_rn(v[(long)i * (h_v * 128) + h * 128 + tid]));
        float r = beta * (sv - di * ks0);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j < i) {
                r -= A[i][j] * dcr[j];
            }
        }
        dcr[i] = r;
        wy_dc[((long)layer * 8 + i) * (h_v * 128) + (long)h * 128 + tid] = r;
    }
}







extern "C" __global__ void gdn_wy_reconstruct(
    const float* __restrict__ wy_k,
    const float* __restrict__ wy_s0,
    const float* __restrict__ wy_dc,
    const float* __restrict__ wy_g,
    __half* __restrict__ st,
    const float* __restrict__ sc,
    int m, int h_k, int h_v, int d, int layer)
{
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int kh = h % h_k;
    const int k_len = h_k * 128;
    const long st_h = (long)layer * h_v * d * d + (long)h * d * d;
    const long sc_h = ((long)layer * h_v + h) * d;
    __shared__ float w[9];
    if (tid == 0) {
        if (m == 0) {
            w[0] = 1.0f;
        } else {
            const float gm = wy_g[((long)layer * 8 + (m - 1)) * h_v + h];
            w[0] = gdn_expf(gm);
            for (int j = 0; j < m; ++j) {
                w[1 + j] = gdn_expf(gm - wy_g[((long)layer * 8 + j) * h_v + h]);
            }
        }
    }
    __syncthreads();
    for (int i2 = 0; i2 < 128; ++i2) {
        float acc = w[0] * wy_s0[st_h + (long)i2 * d + tid];
        for (int j = 0; j < m; ++j) {
            acc += w[1 + j]
                 * wy_k[((long)layer * 8 + j) * k_len + kh * 128 + i2]
                 * wy_dc[((long)layer * 8 + j) * (h_v * 128) + (long)h * 128 + tid];
        }
        {
            const float s_i = sc[sc_h + i2];
            float y = acc / s_i;
            y = fminf(fmaxf(y, -65504.0f), 65504.0f);
            st[st_h + (long)i2 * d + tid] = __float2half_rn(y);
        }
    }
}






extern "C" __global__ void gdn_gate(
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
    float gv = ov * inv * nw[layer * 128 + tid] * (zv / (1.0f + gdn_expf(-zv)));
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
