#include <cuda_fp16.h>
#include "cast_common.cuh"
#include "g4_common.cuh"

extern "C" __global__ void w4a16_embed_gather(
    const unsigned short* __restrict__ emb,
    const unsigned* __restrict__ toks,
    float* __restrict__ out,
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
        out[i] = 0.0f;
        return;
    }
    const unsigned short v = emb[(size_t)tok * (size_t)k + (size_t)c];
    out[i] = bf16 ? b2f(v) : __half2float(__ushort_as_half(v));
}

extern "C" __global__ void w4a16_mtp_chain_step(
    const unsigned* __restrict__ tok,
    unsigned* __restrict__ log,
    unsigned* __restrict__ ctr,
    unsigned* __restrict__ dpp,
    int slot, int kmax)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const unsigned c = *ctr;
    if (c < (unsigned)kmax) {
        log[c] = tok[0];
    }
    *ctr = c + 1u;
    atomicAdd(&dpp[slot], 1u);
}

extern "C" __global__ void w4a16_kstep_ctrl(
    const unsigned* __restrict__ tok,
    unsigned* __restrict__ out,
    unsigned* __restrict__ ctr,
    unsigned* __restrict__ stop,
    unsigned* __restrict__ dpp,
    int slot, int eos)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    if (*stop == 0u) {
        const unsigned t = tok[0];
        out[*ctr] = t;
        *ctr = *ctr + 1u;
        if (t == (unsigned)eos) {
            *stop = 1u;
        } else {
            atomicAdd(&dpp[slot], 1u);
        }
    }
}
