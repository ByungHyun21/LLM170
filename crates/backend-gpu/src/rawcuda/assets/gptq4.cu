// ── W4A16(GPTQ4·g128·sym) GEMV/GEMM CUDA — 비트 계약: core dot_row_w4a16_lane ──
// 산술 계약(crates/core/src/quant/lane.rs와 1:1):
//  - 레인 l = 0..63: i = l, l+64, … f32 누산
//      w = (nib - 8) as f32 * h2f(scale[g]);  acc += w * h2f(x[i])   (mul·add 각각 반올림)
//  - 레인값 f64 변환 → tree64: a[i]+=a[i+32](0..32) 후 off=16,8,4,2,1
//  - 결과 f32 = (float)tree64  (Rust `as f32` 동일 — RN)
// zp=8 상수(sym — zero-point 미저장). h2f는 Rust half_to_f32와 비트동일
// 구현(서브노멀 정규화 루프 포함, libdevice 미사용).
//
// [빌드 계약] -fmad=false 필수 — `acc += w*x`가 FMA로 수축되면 CPU 미러와
// 비트가 어긋난다(scripts/build_cuda_kernels.sh). x·scale은 f16 비트(u16),
// packed는 u32(니블 8개/워드, lsb-first).

#include "cast_common.cuh"

// 활성 f32 [n] → f16 비트 [n] (GEMV 입력 스테이징 계약).
extern "C" __global__ void w4a16_cast_f16(const float* __restrict__ in,
                                          unsigned short* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = f2h(in[i]);
    }
}

// 활성 f32 [n] → h2f(f2h(v)) f32 [n] — t=1 GEMV 입력의 사전 변환.
// 계약: 커널 안에서 h2f(xt[i])하던 값을 밖에서 한 번 계산해 두는 것과 동일
// (f2h→h2f 왕복이 비트를 보존). 반드시 f2h/h2f와 동형 수정.
extern "C" __global__ void w4a16_cast_x32(const float* __restrict__ in,
                                          float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = h2f(f2h(in[i]));
    }
}

#define G4_LANES 64
#define G4_ROWS 8

// t=1 전용 GEMV — 한 블록 = 한 행(64레인). 구 GEMM 커널은 블록이 8행을
// 순차 처리해 메모리 지연을 못 숨기고(실측 GPU 88ms/토큰), 여기서는 격자를
// n으로 늘려 지연을 숨긴다. 스케일은 smem f32로 선변환, 활성 x는
// w4a16_cast_x32가 만든 f32(h2f 왕복). 산술 순서는 구 커널과 동일:
//   w = (nib-8) as f32 * sc[g];  acc += w * x[i];   (mul·add 분리, 레인 l은
//   i=l,l+64,… 오름차순 → f64 tree64). k ≤ 128*G4_SCMAX 계약.
#define G4_SCMAX 256

// double 셔플(__shfl_down_sync는 32bit) — 트리 병렬화용.
__device__ __forceinline__ double shfl_down_f64(double v, int off) {
    unsigned lo = __double2loint(v), hi = __double2hiint(v);
    lo = __shfl_down_sync(0xffffffffu, lo, off);
    hi = __shfl_down_sync(0xffffffffu, hi, off);
    return __hiloint2double(hi, lo);
}

// f64 tree64 — core lane.rs와 동일 순서(전 호출부 공용).
__device__ __forceinline__ float tree64(double* red) {
    for (int i = 0; i < 32; ++i) {
        red[i] += red[i + 32];
    }
    for (int off = 16; off >= 1; off >>= 1) {
        for (int i = 0; i < off; ++i) {
            red[i] += red[i + off];
        }
    }
    return (float)red[0];
}

// ── 그룹·스케일 dtype 변형(W4-1: 35B-A3B = g32·BF16 스케일) ──
// bf16 → f32: 상위 16비트 좌시프트(정확) — core quant::deq::bf16_to_f32 동일.
__device__ __forceinline__ float b2f(unsigned short v) {
    return __uint_as_float(((unsigned)v) << 16);
}

// 스케일 로드 — BF16이면 b2f, 아니면 h2f(기존 g128·f16 경로와 동일 값).
template <bool BF16>
__device__ __forceinline__ float ld_scale(const unsigned short* __restrict__ p, int i) {
    return BF16 ? b2f(p[i]) : h2f(p[i]);
}

// 스케일 그룹 인덱스 — i = l + 64j (l = 레인, j = 64원소 스텝).
// g128(SHIFT=7): (l+64j)>>7 = j>>1 (l<64라 레인 공통), g32(SHIFT=5):
// (l>>5) + 2j. 일반식 (l>>SHIFT)+(j<<(6-SHIFT))는 SHIFT=7에서 시프트가
// 음수라 분기한다.
template <int SHIFT>
__device__ __forceinline__ int sidx(int l, int j) {
    if constexpr (SHIFT == 7) {
        return j >> 1;
    } else {
        return (l >> SHIFT) + (j << (6 - SHIFT));
    }
}

// 2행/블록 — x를 두 행이 공유(L1 x 트래픽 ÷2)하고 레인당 미결 로드가 2배.
// 스케일은 행별(각 행의 srow), 산술 순서는 1행 커널과 동일(계약 불변).
// SHIFT = 그룹 로그2(7 = g128, 5 = g32) — 산술 계약은 core
// dot_row_w4a16_lane_group과 1:1.
template <int SHIFT, bool BF16>
__device__ __forceinline__ void gemv_row_body(
    const unsigned* __restrict__ q,        // [n][k/8] u32 (lsb-first 니블)
    const unsigned short* __restrict__ s,  // [n][k/2^SHIFT] 스케일(f16/bf16 비트)
    const float* __restrict__ x,           // [k] f32 = h2f(f2h(활성))
    float* __restrict__ out,               // [해당 행 1개]
    int o,
    int k)
{
    if (o < 0) {
        return;
    }
    const int l = threadIdx.x;
    __shared__ double red[G4_LANES];
    __shared__ float sc[G4_SCMAX];
    const int k8 = k >> 3;
    const int kg = k >> SHIFT;
    for (int g = l; g < kg; g += G4_LANES) {
        sc[g] = ld_scale<BF16>(s + (size_t)o * kg, g);
    }
    __syncthreads();
    const unsigned* qrow = q + (size_t)o * k8;
    // 스케일 그룹은 i>>7 = (l + 64j)>>7 = j>>1 — 전 레인 공통(유니폼 로드).
    // 16이터레이션 언롤 — 미결 q·x 로드를 16개까지 겹친다(지연 은닉, 실측 8→16
    // = 27.9→27.2ms). 2행/블록 변형은 역효과(34.8ms — 레지스터·L1 압박).
    const int jn = k >> 6;
    const int li = l & 7;
    float acc = 0.0f;
    int j = 0;
    for (; j + 16 <= jn; j += 16) {
#pragma unroll
        for (int u = 0; u < 16; ++u) {
            const int jj = j + u;
            const int i = l + (jj << 6);
            // evict-first — 한 번 읽는 가중치가 L2를 오염시키지 않게(스트리밍).
            unsigned qw = __ldcs(&qrow[i >> 3]);
            int v = (int)((qw >> (4 * li)) & 0xFu) - 8;
            float w = (float)v * sc[sidx<SHIFT>(l, jj)];
            acc += w * x[i];
        }
    }
    for (; j < jn; ++j) {
        const int i = l + (j << 6);
        unsigned qw = __ldcs(&qrow[i >> 3]);
        int v = (int)((qw >> (4 * li)) & 0xFu) - 8;
        float w = (float)v * sc[sidx<SHIFT>(l, j)];
        acc += w * x[i];
    }
    red[l] = (double)acc;
    __syncthreads();
    // 트리 병렬화(계약 순서 유지 — 1단 a[i]+=a[i+32] 후 셔플로 off 16..1).
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

template <int SHIFT, bool BF16>
__device__ __forceinline__ void gemv_body(
    const unsigned* __restrict__ q, const unsigned short* __restrict__ s,
    const float* __restrict__ x, float* __restrict__ out, int n, int k)
{
    const int o = blockIdx.x;
    if (o >= n) {
        return;
    }
    gemv_row_body<SHIFT, BF16>(q, s, x, out, o, k);
}

// 전문가 배치 GEMV — 전문가 포인터 테이블(tab, (q,s) 쌍) + 슬롯 인덱스.
// 테이블 레이아웃 = (il, e, proj) 순서 — 커널 항목 = base + idx[sl]·3
// (같은 층·proj의 전문가). xstride: 게이트/업은 0(x 공통 브로드캐스트),
// 다운은 k(슬롯별 활성 — x = [nslots][k]). grid = n × nslots(블록 = (행,슬롯)),
// out = [nslots][n]. 상주 전용(스트리밍은 호스트 h2d 경로).
extern "C" __global__ void w4a16_gemv_experts_g32_bf16(
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
    // sp = 토큰당 슬롯 수(프리필 = top_k, 디코드 = 1) — x는 토큰 단위 공유.
    const size_t xoff = (size_t)(sl / (sp > 0 ? sp : 1)) * (size_t)xstride;
    gemv_row_body<5, true>((const unsigned*)e[0], (const unsigned short*)e[1],
                           x + xoff, out + (size_t)sl * n, o, k);
}

// t=1 GEMV 래퍼 — g128·f16(27B) / g32·bf16(35B 전문가).
extern "C" __global__ void w4a16_gemv_g128(
    const unsigned* __restrict__ q, const unsigned short* __restrict__ s,
    const float* __restrict__ x, float* __restrict__ out, int n, int k) {
    gemv_body<7, false>(q, s, x, out, n, k);
}
extern "C" __global__ void w4a16_gemv_g32_bf16(
    const unsigned* __restrict__ q, const unsigned short* __restrict__ s,
    const float* __restrict__ x, float* __restrict__ out, int n, int k) {
    gemv_body<5, true>(q, s, x, out, n, k);
}

// out[t][n] = x[t][k] · W4A16(g128, sym) — t≥2(프리필 배치) 전용.
// [2026-10-08 P3-b 최종 — 채택 변형] 1라운드 진화(벤치 실측): 행=블록 43 →
// smem(R=4) 51 → smem f16→f32 변환 63 → **R=8(512스레드) 72~77(채택)** →
// 레지스터 블로킹 RL=2 61(기각) → f32 직접 52(기각) → cp.async 2단 62(기각:
// 스테이징 지연은 병목 아님). ncu는 권한(ERR_NVGPUCTRPERM)으로 카운터 불가 —
// 잔여 지연 요인은 미규명(모델: warp-jj ~30사이클 × 2.78M ≈ 380µs vs 실측 620).
// 채택본: R=8 + x f32(cast_x32 공유 — 변환·h2f 없음) + smem 스테이징 + 언롤.
// 산술 계약 불변(행·토큰별 레인 l은 i=l,l+64,… 오름차순 f32 누산 → tree64).
#define G4_GTMAX 32            // g128 GEMM 토큰 상한(2026-10-09: 8→32)
#define G4_ROWS 8
#define G4_LANES 64
#define G4_KC 128              // k-청크(64의 배수 · x smem 128×36×4=18KB)

// SHIFT = 그룹 로그2(7 = g128, 5 = g32), BF16 = 스케일 dtype.
template <int SHIFT, bool BF16>
__device__ __forceinline__ void gemm_body(
    const unsigned* __restrict__ q,        // [n][k/8] u32 (lsb-first 니블)
    const unsigned short* __restrict__ s,  // [n][k/2^SHIFT] 스케일(f16/bf16 비트)
    const float* __restrict__ x,           // [t][k] f32 = h2f(f2h(활성))
    float* __restrict__ out,               // [t][n]
    int n,
    int k,
    int t)
{
    const int o0 = blockIdx.x * G4_ROWS;
    const int g = threadIdx.x >> 6;
    const int l = threadIdx.x & (G4_LANES - 1);
    const int o = o0 + g;
    const bool live = (o < n) && (t > 0) && (t <= G4_GTMAX);
    __shared__ float sc[G4_ROWS][G4_SCMAX];
    // x 스테이징 = 전치+패딩 레이아웃([il][36]): 레인당 float4 8회 판독
    // (종전 8토큰 float4 2회 — 2026-10-09 t≤32 확대). 36 = 32토큰 + 4패딩
    // (16B 정렬·뱅크 충돌 회피: 행 stride 144B).
    __shared__ float xs[G4_KC][36];
    __shared__ double red[G4_ROWS][G4_LANES];
    float acc[G4_GTMAX];
#pragma unroll
    for (int u = 0; u < G4_GTMAX; ++u) {
        acc[u] = 0.0f;
    }
    const int k8 = k >> 3;
    const int kg = k >> SHIFT;
    if (live) {
        for (int gg = l; gg < kg; gg += G4_LANES) {
            sc[g][gg] = ld_scale<BF16>(s + (size_t)o * kg, gg);
        }
    }
    const int li = l & 7;
    for (int base = 0; base < k; base += G4_KC) {
        __syncthreads(); // 이전 청크 소비 완료 대기
        for (int idx = threadIdx.x; idx < G4_GTMAX * G4_KC; idx += blockDim.x) {
            const int ti = idx / G4_KC;
            const int il = idx - ti * G4_KC;
            const int gi = base + il;
            xs[il][ti] = (ti < t && gi < k) ? x[(size_t)ti * k + gi] : 0.0f;
        }
        __syncthreads();
        if (live) {
            const unsigned* qrow = q + (size_t)o * k8;
            const int nch = min(G4_KC, k - base);
#pragma unroll
            for (int q4 = 0; q4 < G4_KC / G4_LANES; ++q4) {
                const int il = l + (q4 << 6);
                if (il < nch) {
                    const int i = base + il;
                    const unsigned qw = __ldcs(&qrow[i >> 3]); // 가중치 1회 = t토큰 공유
                    const int v = (int)((qw >> (4 * li)) & 0xFu) - 8;
                    const float w = (float)v * sc[g][i >> SHIFT];
                    // 전치 패딩 레이아웃에서 float4 2회로 8토큰 판독(smem 트래픽 ÷4).
                    const float4 xa = *reinterpret_cast<const float4*>(&xs[il][0]);
                    const float4 xb = *reinterpret_cast<const float4*>(&xs[il][4]);
                    const float4 xc = *reinterpret_cast<const float4*>(&xs[il][8]);
                    const float4 xd = *reinterpret_cast<const float4*>(&xs[il][12]);
                    const float4 xe = *reinterpret_cast<const float4*>(&xs[il][16]);
                    const float4 xf = *reinterpret_cast<const float4*>(&xs[il][20]);
                    const float4 xg2 = *reinterpret_cast<const float4*>(&xs[il][24]);
                    const float4 xh = *reinterpret_cast<const float4*>(&xs[il][28]);
                    if (t > 0) acc[0] += w * xa.x;
                    if (t > 1) acc[1] += w * xa.y;
                    if (t > 2) acc[2] += w * xa.z;
                    if (t > 3) acc[3] += w * xa.w;
                    if (t > 4) acc[4] += w * xb.x;
                    if (t > 5) acc[5] += w * xb.y;
                    if (t > 6) acc[6] += w * xb.z;
                    if (t > 7) acc[7] += w * xb.w;
                    if (t > 8) acc[8] += w * xc.x;
                    if (t > 9) acc[9] += w * xc.y;
                    if (t > 10) acc[10] += w * xc.z;
                    if (t > 11) acc[11] += w * xc.w;
                    if (t > 12) acc[12] += w * xd.x;
                    if (t > 13) acc[13] += w * xd.y;
                    if (t > 14) acc[14] += w * xd.z;
                    if (t > 15) acc[15] += w * xd.w;
                    if (t > 16) acc[16] += w * xe.x;
                    if (t > 17) acc[17] += w * xe.y;
                    if (t > 18) acc[18] += w * xe.z;
                    if (t > 19) acc[19] += w * xe.w;
                    if (t > 20) acc[20] += w * xf.x;
                    if (t > 21) acc[21] += w * xf.y;
                    if (t > 22) acc[22] += w * xf.z;
                    if (t > 23) acc[23] += w * xf.w;
                    if (t > 24) acc[24] += w * xg2.x;
                    if (t > 25) acc[25] += w * xg2.y;
                    if (t > 26) acc[26] += w * xg2.z;
                    if (t > 27) acc[27] += w * xg2.w;
                    if (t > 28) acc[28] += w * xh.x;
                    if (t > 29) acc[29] += w * xh.y;
                    if (t > 30) acc[30] += w * xh.z;
                    if (t > 31) acc[31] += w * xh.w;
                }
            }
        }
    }
    // 토큰별 트리 — 계약 순서(1단 a[i]+=a[i+32] 후 off 16,8,4,2,1)를 병렬화:
    // 워프 셔플로 각 단을 병렬 가산(각 가산은 독립 — 비트 동일), 배리어는
    // red 기록용 1회/토큰만.
    for (int ti = 0; ti < G4_GTMAX; ++ti) {
        if (ti < t) {
            red[g][l] = (double)acc[ti];
        }
        __syncthreads();
        if (live && l < 32) {
            double r = red[g][l] + red[g][l + 32];
#pragma unroll
            for (int off = 16; off >= 1; off >>= 1) {
                const double oth = shfl_down_f64(r, off);
                if (l < off) {
                    r += oth;
                }
            }
            if (l == 0) {
                out[(size_t)ti * n + o] = (float)r;
            }
        }
        __syncthreads();
    }
}

// t≥2 GEMM 래퍼 — g128·f16(27B) / g32·bf16(35B 전문가).
extern "C" __global__ void w4a16_gemm_g128(
    const unsigned* __restrict__ q, const unsigned short* __restrict__ s,
    const float* __restrict__ x, float* __restrict__ out, int n, int k, int t) {
    gemm_body<7, false>(q, s, x, out, n, k, t);
}
extern "C" __global__ void w4a16_gemm_g32_bf16(
    const unsigned* __restrict__ q, const unsigned short* __restrict__ s,
    const float* __restrict__ x, float* __restrict__ out, int n, int k, int t) {
    gemm_body<5, true>(q, s, x, out, n, k, t);
}

// ── MoE(W4-1, 35B-A3B) 보조 커널 ──
// 플레인 bf16 GEMV — 행=블록(64레인), 레인 l = i=l,l+64,… f32 누산 → tree64.
// split 커널과 동일한 레인·환원 구조(플레인 경로 판정은 토큰 수준 — 골든).
// w는 [n][k] 행 우선(업로드 원본 그대로 — 전치 없음).
extern "C" __global__ void w4a16_gemv_bf16(
    const unsigned short* __restrict__ w,  // [n][k] bf16
    const float* __restrict__ x,           // [k] f32
    float* __restrict__ out,               // [n]
    int n, int k)
{
    const int o = blockIdx.x;
    if (o >= n) {
        return;
    }
    const int l = threadIdx.x;
    __shared__ double red[G4_LANES];
    const unsigned short* wrow = w + (size_t)o * k;
    float acc = 0.0f;
    for (int i = l; i < k; i += G4_LANES) {
        acc += b2f(wrow[i]) * x[i];
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

#define G4_TMAX 8              // 플레인 v1 GEMM 전용 토큰 상한

// 플레인 bf16 GEMM(t≤8) — 행=블록(64레인), **가중치 1회 판독 × t토큰 재사용**
// (t=1 경로와 같은 레인·환원 순서 — 판정은 토큰 수준, 골든). x는 원시 f32
// [t][k](전치 없음), out [t][n]. 프리필 청크(t≤8)의 dense 경로.
extern "C" __global__ void w4a16_gemm_bf16(
    const unsigned short* __restrict__ w,  // [n][k] bf16
    const float* __restrict__ x,           // [t][k] f32
    float* __restrict__ out,               // [t][n]
    int n, int k, int t)
{
    const int o = blockIdx.x;
    if (o >= n) {
        return;
    }
    const int l = threadIdx.x;
    __shared__ double red[G4_LANES];
    const unsigned short* wrow = w + (size_t)o * k;
    float acc[G4_TMAX];
#pragma unroll
    for (int u = 0; u < G4_TMAX; ++u) {
        acc[u] = 0.0f;
    }
    for (int i = l; i < k; i += G4_LANES) {
        const float wv = b2f(wrow[i]);
#pragma unroll
        for (int u = 0; u < G4_TMAX; ++u) {
            if (u < t) {
                acc[u] += wv * x[(size_t)u * k + i];
            }
        }
    }
    for (int u = 0; u < t; ++u) {
        red[l] = (double)acc[u];
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
                out[(size_t)u * n + o] = (float)r;
            }
        }
        __syncthreads();
    }
}

// 플레인 bf16 GEMM v3(t≤32) — **8행/블록(512스레드) + 행별 smem 가중치 + k청크**.
// [v2(1행/블록)는 x를 행마다 재판독해 L2 트래픽이 n×t×k×4(실측 8GB/층) —
// v3는 8그룹이 같은 x를 L1 공유(÷8)하고 가중치도 청크 단위 smem 재사용.]
// 산술 순서는 v1/v2와 동일(레인 l = i=l,l+64,… f32 누산 → tree64 — 토큰 독립).
#define G4_TMAX2 32
#define G4_ROWS_B 8
#define G4_KC_B 128
extern "C" __global__ void w4a16_gemm_bf16_t(
    const unsigned short* __restrict__ w,  // [n][k] bf16
    const float* __restrict__ x,           // [t][k] f32
    float* __restrict__ out,               // [t][n]
    int n, int k, int t)
{
    const int g = threadIdx.x >> 6;
    const int l = threadIdx.x & (G4_LANES - 1);
    const int o = blockIdx.x * G4_ROWS_B + g;
    const bool live = o < n;
    __shared__ double red[G4_ROWS_B][G4_LANES];
    __shared__ unsigned short ws[G4_ROWS_B][G4_KC_B];
    float acc[G4_TMAX2];
#pragma unroll
    for (int u = 0; u < G4_TMAX2; ++u) {
        acc[u] = 0.0f;
    }
    for (int base = 0; base < k; base += G4_KC_B) {
        const int nch = min(G4_KC_B, k - base);
        for (int idx = threadIdx.x; idx < G4_ROWS_B * G4_KC_B; idx += blockDim.x) {
            const int r = idx / G4_KC_B;
            const int i = idx - r * G4_KC_B;
            const int ro = blockIdx.x * G4_ROWS_B + r;
            ws[r][i] = (ro < n && i < nch) ? w[(size_t)ro * k + base + i] : (unsigned short)0;
        }
        __syncthreads();
        const unsigned short* wsrow = ws[g];
        const float* xp = x + base + l;
        for (int i = l; i < nch; i += G4_LANES, xp += G4_LANES) {
            const float wv = live ? b2f(wsrow[i]) : 0.0f;
#pragma unroll
            for (int u = 0; u < G4_TMAX2; ++u) {
                if (u < t) {
                    acc[u] += wv * xp[(size_t)u * k];
                }
            }
        }
        __syncthreads(); // 다음 청크 스테이징 전 소비 완료
    }
    for (int u = 0; u < t; ++u) {
        red[g][l] = (double)acc[u];
        __syncthreads();
        if (live && l < 32) {
            double r = red[g][l] + red[g][l + 32];
#pragma unroll
            for (int off = 16; off >= 1; off >>= 1) {
                const double oth = shfl_down_f64(r, off);
                if (l < off) {
                    r += oth;
                }
            }
            if (l == 0) {
                out[(size_t)u * n + o] = (float)r;
            }
        }
        __syncthreads();
    }
}

// 가중 누적 — y[i] += w·x[i] (mul·add 분리 — CPU MoE 스테이지와 동일 산식).
extern "C" __global__ void w4a16_axpy(float w, const float* __restrict__ x,
                                      float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] += w * x[i];
    }
}

// 전문가 가중 누적 — y[i] += Σ_s w[s]·d[s][i] (선택 순서 가산 — CPU 미러).
extern "C" __global__ void w4a16_moe_accum(const float* __restrict__ w,
                                           const float* __restrict__ d,
                                           float* __restrict__ y, int sp, int nslots,
                                           int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int spt = (sp > 0) ? sp : nslots;
        for (int ti = 0; ti < nslots; ti += spt) {
            float acc = 0.0f;
            const int end = (ti + spt < nslots) ? ti + spt : nslots;
            for (int s = ti; s < end; ++s) {
                acc += w[s] * d[(size_t)s * n + i];
            }
            y[(size_t)(ti / spt) * n + i] = acc;
        }
    }
}

// shared 게이트 가산 — y[i] += sigmoid(sg[0])·x[i] (sigmoid = 1/(1+e^-v)).
extern "C" __global__ void w4a16_shared_add(const float* __restrict__ sg,
                                            const float* __restrict__ x,
                                            float* __restrict__ y, int n) {
    const int ti = blockIdx.x;
    const int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i < n) {
        const float s = 1.0f / (1.0f + expf(-sg[ti]));
        y[(size_t)ti * n + i] += s * x[(size_t)ti * n + i];
    }
}
