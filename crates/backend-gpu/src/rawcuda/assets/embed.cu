// ── [S3-1 2026-10-11] 임베딩 행 gather(GPU) — 프리필 행 조립 대체 ──
// 호스트 embed_row(행당 mmap 디코드+Vec 할당+h2d 20KB)를 토큰 id h2d(4B/행)
// + 본 커널로 대체한다. 계약: h2f/b2f = Rust half_to_f32/bf16_to_f32와
// 비트동일(decode_f32가 쓰는 동일 함수) — 값 불변(골든 재판정).
#include "cast_common.cuh"
#include "g4_common.cuh" // b2f

extern "C" __global__ void w4a16_embed_gather(
    const unsigned short* __restrict__ emb, // [n][k] f16/bf16 원본
    const unsigned* __restrict__ toks,      // [t] u32 토큰 id(< n — 호스트 계약)
    float* __restrict__ out,                // [t][k] f32
    int t, int k, int n, int bf16)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= t * k) {
        return;
    }
    const int r = i / k;
    const int c = i - r * k;
    const unsigned tok = toks[r];
    if (tok >= (unsigned)n) {
        out[i] = 0.0f; // 방어 — 호스트가 사전 거부(계약)
        return;
    }
    const unsigned short v = emb[(size_t)tok * (size_t)k + (size_t)c];
    out[i] = bf16 ? b2f(v) : h2f(v);
}
