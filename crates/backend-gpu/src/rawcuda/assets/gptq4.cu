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

__device__ __forceinline__ float h2f(unsigned short h) {
    unsigned sign = (unsigned)(h >> 15) << 31;
    unsigned e = (unsigned)(h >> 10) & 0x1Fu;
    unsigned m = (unsigned)h & 0x3FFu;
    unsigned bits;
    if (e == 0) {
        if (m == 0) {
            bits = sign;
        } else {
            // 서브노멀 — Rust half_to_f32와 동일 정규화 루프.
            int ee = 127 - 15 + 1;
            unsigned f = m;
            while ((f & 0x400u) == 0) {
                f <<= 1;
                ee -= 1;
            }
            f &= 0x3FFu;
            bits = sign | ((unsigned)ee << 23) | (f << 13);
        }
    } else if (e == 0x1Fu) {
        bits = sign | (0xFFu << 23) | (m << 13);
    } else {
        bits = sign | ((e + 112u) << 23) | (m << 13);
    }
    return __uint_as_float(bits);
}

// f32 → f16 비트(RN-even, 서브노멀·inf/nan) — 호스트 w4a16_dec::f32_to_f16
// 과 비트동일 계약(디바이스 체인 캐스트). 반드시 동형으로 수정할 것.
__device__ __forceinline__ unsigned short f2h(float v) {
    unsigned b = __float_as_uint(v);
    unsigned sign = (unsigned short)((b >> 16) & 0x8000u);
    int exp = (int)((b >> 23) & 0xFFu);
    unsigned man = b & 0x7FFFFFu;
    if (exp == 0xFF) {
        return (unsigned short)(sign | 0x7C00u | (man != 0 ? 0x200u : 0u));
    }
    int e = exp - 127 + 15;
    if (e >= 31) {
        return (unsigned short)(sign | 0x7C00u);
    }
    if (e <= 0) {
        if (e < -10) {
            return (unsigned short)sign;
        }
        unsigned m = man | 0x800000u;
        unsigned shift = (unsigned)(14 - e);
        unsigned half = 1u << (shift - 1);
        unsigned sub = m >> shift;
        unsigned rem = m & ((1u << shift) - 1u);
        if (rem > half || (rem == half && (sub & 1u) == 1u)) {
            sub += 1u;
        }
        return (unsigned short)(sign | sub);
    }
    unsigned h = ((unsigned)e << 10) | (man >> 13);
    unsigned rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) == 1u)) {
        h += 1u;
    }
    return (unsigned short)(sign | h);
}

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

// 2행/블록 — x를 두 행이 공유(L1 x 트래픽 ÷2)하고 레인당 미결 로드가 2배.
// 스케일은 행별(각 행의 srow), 산술 순서는 1행 커널과 동일(계약 불변).
extern "C" __global__ void w4a16_gemv_g128(
    const unsigned* __restrict__ q,        // [n][k/8] u32 (lsb-first 니블)
    const unsigned short* __restrict__ s,  // [n][k/128] f16 비트(스케일)
    const float* __restrict__ x,           // [k] f32 = h2f(f2h(활성))
    float* __restrict__ out,               // [n]
    int n,
    int k)
{
    const int o = blockIdx.x;
    if (o >= n) {
        return;
    }
    const int l = threadIdx.x;
    __shared__ double red[G4_LANES];
    __shared__ float sc[G4_SCMAX];
    const int k8 = k >> 3;
    const int kg = k >> 7;
    for (int g = l; g < kg; g += G4_LANES) {
        sc[g] = h2f(s[(size_t)o * kg + g]);
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
            float w = (float)v * sc[jj >> 1];
            acc += w * x[i];
        }
    }
    for (; j < jn; ++j) {
        const int i = l + (j << 6);
        unsigned qw = __ldcs(&qrow[i >> 3]);
        int v = (int)((qw >> (4 * li)) & 0xFu) - 8;
        float w = (float)v * sc[j >> 1];
        acc += w * x[i];
    }
    red[l] = (double)acc;
    __syncthreads();
    if (l == 0) {
        out[o] = tree64(red);
    }
}

// out[t][n] = x[t][k] · W4A16(g128, sym) — t≥2(프리필 배치) 전용.
// [2026-10-08 P3-b 최종 — 채택 변형] 1라운드 진화(벤치 실측): 행=블록 43 →
// smem(R=4) 51 → smem f16→f32 변환 63 → **R=8(512스레드) 72~77(채택)** →
// 레지스터 블로킹 RL=2 61(기각) → f32 직접 52(기각) → cp.async 2단 62(기각:
// 스테이징 지연은 병목 아님). ncu는 권한(ERR_NVGPUCTRPERM)으로 카운터 불가 —
// 잔여 지연 요인은 미규명(모델: warp-jj ~30사이클 × 2.78M ≈ 380µs vs 실측 620).
// 채택본: R=8 + x f32(cast_x32 공유 — 변환·h2f 없음) + smem 스테이징 + 언롤.
// 산술 계약 불변(행·토큰별 레인 l은 i=l,l+64,… 오름차순 f32 누산 → tree64).
#define G4_TMAX 8
#define G4_ROWS 8
#define G4_LANES 64
#define G4_KC 256              // k-청크(64의 배수)

extern "C" __global__ void w4a16_gemm_g128(
    const unsigned* __restrict__ q,        // [n][k/8] u32 (lsb-first 니블)
    const unsigned short* __restrict__ s,  // [n][k/128] f16 비트(스케일)
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
    const bool live = (o < n) && (t > 0) && (t <= G4_TMAX);
    __shared__ float sc[G4_ROWS][G4_SCMAX];
    __shared__ float xs[G4_TMAX][G4_KC];
    __shared__ double red[G4_ROWS][G4_LANES];
    float acc[G4_TMAX];
#pragma unroll
    for (int u = 0; u < G4_TMAX; ++u) {
        acc[u] = 0.0f;
    }
    const int k8 = k >> 3;
    const int kg = k >> 7;
    if (live) {
        for (int gg = l; gg < kg; gg += G4_LANES) {
            sc[g][gg] = h2f(s[(size_t)o * kg + gg]);
        }
    }
    const int li = l & 7;
    for (int base = 0; base < k; base += G4_KC) {
        __syncthreads(); // 이전 청크 소비 완료 대기
        for (int idx = threadIdx.x; idx < G4_TMAX * G4_KC; idx += blockDim.x) {
            const int ti = idx / G4_KC;
            const int il = idx - ti * G4_KC;
            const int gi = base + il;
            xs[ti][il] = (ti < t && gi < k) ? x[(size_t)ti * k + gi] : 0.0f;
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
                    const float w = (float)v * sc[g][i >> 7];
#pragma unroll
                    for (int ti = 0; ti < G4_TMAX; ++ti) {
                        if (ti < t) {
                            acc[ti] += w * xs[ti][il];
                        }
                    }
                }
            }
        }
    }
    for (int ti = 0; ti < G4_TMAX; ++ti) {
        if (ti < t) {
            red[g][l] = (double)acc[ti];
        }
        __syncthreads();
        if (live && l == 0) {
            out[(size_t)ti * n + o] = tree64(red[g]);
        }
        __syncthreads();
    }
}
