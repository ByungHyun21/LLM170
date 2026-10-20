// FN(Qwen3.8-Flash-Next) 전용 커널 — M1 프리필 체인(w4a16_dec/fnchain.rs).
// [빌드 계약] -fmad=false(sm_80) — gdn/attn/ew와 동일 노선. 신규 산술
// (HC·PLE)은 vLLM 0.31 참조구현(qwen4_exp common/nvidia ops)과 수식 1:1,
// 정밀도 클래스는 체인 f32(모듈 계약 — 골든 판정은 M2 토큰 대조).
// gptq4.cu는 바이트동결이라 전문가 g128 GEMV 행본체를 여기에 자체 포함
// (산술 순서 = gptq4.cu gemv_row_body<7,false> 미러).
#include <cuda_fp16.h> // gdn_gate_sig의 f16 미러 기록(gdn.cu와 동일 계약)
#include "g4_common.cuh" // h2f/f2h/b2f/shfl_down_f64 — 공용 헬퍼(단일 정의)

// ── hyper-connection(Gated Residual) ─────────────────────────────────────
// 수식 출처: vllm/models/qwen4_exp/common/hyperconnection.py + nvidia/ops/hc.py
//  · mix:    xn = GroupedGemmaRMSNorm(R)[스트림별 RMS·(1+w)] (hc_norm)
//  · gate   = σ(W_up · silu(W_down·xn / hc) / 1) — silu 인자가 /hc(4)
//  · block  = mean_s( σ(gate)_s ⊙ xn_s )
//  · inject = 2σ(W_inj·xn / hc) — 쓰기 R'_s = R_s + y·inj_s
// 최종 mixer = 노름+mix만(use_combine=False).

// R 초기화 — 임베딩 행 [t][2560]을 4스트림으로 복제 → R [t][10240].
// grid = ceil(n/256), n = t*10240.
extern "C" __global__ void fn_hc_init(const float* __restrict__ emb,
                                      float* __restrict__ r, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int hh = 10240; // hc_count(4) * hidden(2560) — FN 고정(커널 전용)
    // [fix 2026-10-20] 스트림 복제 — (i%hh)는 10240 전역이라 2560으로 접어
    // 읽는다(종전 식은 스트림≥1에서 다음 토큰 임베딩을 읽었다 — init 지문
    // 0.0938≠0.0432 발산의 원인).
    r[i] = emb[(i / hh) * 2560 + (i % hh) % 2560];
}

// 그룹 GemmaRMSNorm — R [t][10240] → xn [t][10240]. w = 해당 HC 모듈의
// hc_norm 행 포인터(호스트가 행 오프셋을 인자로 조립). 스트림별(2560)
// RMS 환원 — y = x·rrms; y += y·w(참조 커널의 FMA 형태 — fmad=false로
// 2반올림). grid = t, 블록 256(스트림 순차 4회).
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
        for (int j = 0; j < 10; ++j) { // 256레인 × 10 = 2560
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
            o[i] = y + y * wr[i];
        }
        __syncthreads();
    }
}

// silu(x/hc) — 저랭크 게이트 전활성. lora [t][320] 원소별.
// grid = ceil(n/256).
extern "C" __global__ void fn_hc_silu(const float* __restrict__ x,
                                      float* __restrict__ y, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float u = x[i] * 0.25f; // /hc_count(4) — 참조 _hc_silu_kernel
    y[i] = u / (1.0f + __expf(-u));
}

// 게이트 믹스 — x[t][2560] = Σ_s σ(g_s)·xn_s / 4 (g = W_up 출력 [t][10240]).
// grid = (t, 2560/256).
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

// HC 쓰기 — R[t][s·2560+i] += y[t][i]·2σ(inj[t·4+s]/4). inj = W_inject·xn
// 원시 출력 [t][4]. grid = (t, 4).
extern "C" __global__ void fn_hc_write(float* __restrict__ r,
                                       const float* __restrict__ y,
                                       const float* __restrict__ inj, int t)
{
    const int tt = blockIdx.x;
    if (tt >= t) return;
    const int s = blockIdx.y;
    const int i = threadIdx.x;
    if (i >= 2560) return;
    const float u = inj[tt * 4 + s] * 0.25f;
    const float sc = 2.0f / (1.0f + __expf(-u));
    r[(size_t)tt * 10240 + s * 2560 + i] += y[(size_t)tt * 2560 + i] * sc;
}

// ── PLE(position-learning enhancement) ───────────────────────────────────
// 수식 출처: vllm qwen4_exp common/ops/ple.py(_ple_gate_kernel·_ple_conv_kernel)
// · key/value = kv_proj(emb) — key [t][10240](스트림별), value [t][2560](공유)
// · per (t, s): k_n = RMS(key_s)·(1+nk_s) · q_n = RMS(R_s)·(1+nq_s)
//   d = ⟨k_n, q_n⟩/√2560 · g = σ(sign(d)·√max(|d|, 1e-6)) · gated_s = g·v
//   normed_s = RMS(gated_s)·(1+nc_s)
// · conv: depthwise k4 dil3 — out tap t−{0,3,6,9}, silu, R += gated + silu(conv)

// 게이트 — grid = (t, 4), 블록 256. 노름 가중 nk/nq/nc = [10240] 원본
// 레이아웃 그대로(스트림 오프셋 s*2560로 인덱싱).
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
    const size_t base = (size_t)tt * 10240 + s * 2560;
    const size_t wb = s * 2560; // 노름 가중 스트림 오프셋
    const float* k = key + base;
    const float* q = r + base;
    float kk = 0.0f, qq = 0.0f;
    #pragma unroll
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
    // d = ⟨k_n, q_n⟩/√H — 정규화 값의 곱을 f32 순서로 누산(참조 커널은
    // 중간 스칼라를 bf16 경계 반올림 — 우리 체인은 f32 단일 클래스,
    // 골든 판정은 M2 토큰 대조).
    float dot = 0.0f;
    #pragma unroll
    for (int j = 0; j < 10; ++j) {
        const int i = tid + j * 256;
        const float kn = k[i] * rk * (1.0f + nk[wb + i]);
        const float qn = q[i] * rq * (1.0f + nq[wb + i]);
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
    const float* wv = val + (size_t)tt * 2560;
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
        normed[base + i] = gv * rg * (1.0f + nc[wb + i]);
    }
}

// 숏컨브(depthwise k4·dil3·10240ch) + PLE 잔차 가산 — grid = (t, 10240/256).
// 상태 state[slot][9][10240] = 청크 직전 9입력(절대 위치 start−9..start−1).
// R[t][c] += gated[t][c] + silu(Σ_k w[c·4+k]·tap(t−3k)).
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
        const int jj = tt - 3 * k;
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

// conv 상태 갱신 — new_state[i] = normed[t−9+i] (src<0 = 구 state[i+t] 유지,
// 초기 청크는 0). grid = (1, 10240/256).
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
        } else if (!has_init) {
            state[(size_t)i * 10240 + c] = 0.0f;
        } // has_init && src<0 → 구값 유지(= 구 state[i+t])
    }
}

// ── GDN 출력게이트 sigmoid판(FN output_gate_type) ────────────────────────
// gdn.cu gdn_gate의 silu(z)를 sigmoid(z)로 교체한 FN 전용 복제(gdn.cu은
// 27B/35B 골든 fatbin — 바이트 불변 유지). 나머지 산술·레이아웃 동일.
extern "C" __global__ void gdn_gate_sig(
    const float* __restrict__ o_lc,  // [T][h_v*128] lc
    const float* __restrict__ z,     // [T][h_v*128] HF
    const float* __restrict__ nw,    // [L][128]
    float* __restrict__ gated,       // [T][h_v*128] HF
    float* __restrict__ gated32,     // x32(roundtrip) — nullable
    unsigned short* __restrict__ gated16, // f16 미러 — nullable
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
    // FN 계약: o = W_o[σ(z) ⊙ RMSNorm(y)] — silu 대비 분모 z 제거만.
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

// ── 전문가 g128·f16 배치 GEMV(스트리밍 mcache 간접) ──────────────────────
// gptq4.cu w4a16_gemv_experts_g32_bf16의 g128/f16 판 — 바이트동결된
// gptq4.cu 대신 fn.cu에 자체 포함. 행본체 산술 = gemv_row_body<7,false>
// 미러(64레인·f64 tree64·fmaf 3-op — k 128배수 계약).
#define FN_SCMAX 64 // k ≤ 128*64 = 8192(FN 전문가 k = 2560/640)

__device__ __forceinline__ void fn_gemv_row_g128(
    const unsigned* __restrict__ q,        // [n][k/8] u32 (lsb-first 니블)
    const unsigned short* __restrict__ s,  // [n][k/128] f16 스케일
    const float* __restrict__ x,           // [k] f32(원시 — 플레인 경로)
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
            const float scv = sc[jj]; // g128: 그룹 = 128폭 청크 그대로
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


// [QSA] attn.cu 헬퍼 사본(독립 컴파일 — decode.cu 선례). attn_part_step의
// 청크 온라인 소프트맥스와 동일 산식(값 계약 동일).
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

// ── [QSA 2026-10-20] 선택 집합 어텐션 — attn_fwd3s_part_q의 행 순회만
// sel 리스트(절대 행 번호)로 교체한 FN 인덱서 판(part 레이아웃 258 동일 —
// attn_fwd3s_merge를 splits=1로 그대로 재소비). 수치 클래스 = part_q 계열
// (q f32·K int8·온라인 소프트맥스 f32) — FN 골든은 오라클(QSA CPU) 대조.
// sel = [t][sel_stride] 패딩 -1, nsel = [t] 행별 개수. kdot의 masked 행은
// -1e30(기여 0 — attn_part_step 계약).
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
