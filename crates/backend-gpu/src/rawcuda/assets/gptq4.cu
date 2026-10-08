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

#define G4_LANES 64
#define G4_ROWS 8

// out[t][n] = x[t][k] · W4A16(g128, sym) — t=1이면 GEMV.
extern "C" __global__ void w4a16_gemm_g128(
    const unsigned* __restrict__ q,        // [n][k/8] u32 (lsb-first 니블)
    const unsigned short* __restrict__ s,  // [n][k/128] f16 비트(스케일)
    const unsigned short* __restrict__ x,  // [t][k] f16 비트(활성)
    float* __restrict__ out,               // [t][n]
    int n,
    int k,
    int t)
{
    __shared__ double red[G4_LANES];
    const int l = threadIdx.x;
    const int row0 = blockIdx.x * G4_ROWS;
    const int k8 = k >> 3;
    const int kg = k >> 7;
    for (int r = 0; r < G4_ROWS; ++r) {
        const int o = row0 + r;
        if (o >= n) {
            return;
        }
        const unsigned* qrow = q + (size_t)o * k8;
        const unsigned short* srow = s + (size_t)o * kg;
        for (int ti = 0; ti < t; ++ti) {
            const unsigned short* xt = x + (size_t)ti * k;
            float acc = 0.0f;
            for (int i = l; i < k; i += G4_LANES) {
                unsigned qw = qrow[i >> 3];
                int qq = (int)((qw >> (4 * (i & 7))) & 0xFu);
                float w = (float)(qq - 8) * h2f(srow[i >> 7]);
                acc += w * h2f(xt[i]);
            }
            red[l] = (double)acc;
            __syncthreads();
            if (l == 0) {
                // tree64 — core lane.rs와 동일 순서.
                for (int i = 0; i < 32; ++i) {
                    red[i] += red[i + 32];
                }
                for (int off = 16; off >= 1; off >>= 1) {
                    for (int i = 0; i < off; ++i) {
                        red[i] += red[i + off];
                    }
                }
                out[(size_t)ti * n + o] = (float)red[0];
            }
            __syncthreads();
        }
    }
}
