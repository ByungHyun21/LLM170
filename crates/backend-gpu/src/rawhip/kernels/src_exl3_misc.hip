extern "C" __global__ void exl3_pos_bump(unsigned* __restrict__ pp)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) pp[0] += 1u;
}


// embedding row gather (plans/130 A3): upload token ids only, copy dembed rows
// to dst on device - removes d2h read + h2d re-upload roundtrip and host embed residency.
extern "C" __global__ void exl3_embed_gather(const float* __restrict__ emb,
                                             const unsigned* __restrict__ ids,
                                             float* __restrict__ dst,
                                             int hidden)
{
    const int t = blockIdx.x;
    const float* src = emb + (size_t)ids[t] * hidden;
    float* d = dst + (size_t)t * hidden;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) d[i] = src[i];
}
