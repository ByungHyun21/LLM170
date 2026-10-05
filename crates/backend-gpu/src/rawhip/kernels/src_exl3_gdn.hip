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

// exl3_attn_prep hip 직이식 — q 디인터리브[rms+rope], k rms+rope→KV, v 복사.
// 그리드 (T, 28): j<24=q헤드, j>=24=KV헤드(m=j-24). WG=128.
// kvcap(런타임 인자, plans/128 P0): KV 위치 상한 — 과거 1024 리터럴이
// 서빙 ctx를 1023으로 가둬 attn_prep 메모리 폴트를 냈다. dkc/dvc 할당과 동일 값.
