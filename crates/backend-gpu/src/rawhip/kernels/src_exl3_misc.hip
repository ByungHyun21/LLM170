extern "C" __global__ void exl3_pos_bump(unsigned* __restrict__ pp)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) pp[0] += 1u;
}

