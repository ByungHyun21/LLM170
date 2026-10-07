// llm170 rawcuda Flash-Next(Qwen4-Expert) scaffold smoke kernel
// (plans/124 G001 FNA, 2026-10-05).
//
// Contract source is this .cu - the built asset (exl3_fn.fatbin) is
// produced by scripts/build_cuda.bat into the same directory and committed
// (rawhip co/*.co mirror: source and asset are committed together; any
// kernel arithmetic change starts in this file).
//
// FNA scope: plumbing proof ONLY (fatbin load -> launch -> d2h -> value).
// out[i] = in[i] * hc_scale + i, probed with in[i]=i*0.5 and
// hc_scale=0.25 (hc=4 stream-average factor, stages/hc.rs /=hc): every
// product and sum is exact in f32 for i < 2^24, so the probe requires
// bit-identical output (same discipline as smoke.cu - no approximation
// in a smoke). Stage arithmetic arrives with goals FNB..FNH; this kernel
// deliberately stays trivial and must not grow stage math.
extern "C" __global__ void llm170_fn_smoke(const float* in, float* out, int n, float hc_scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * hc_scale + (float)i;
}
