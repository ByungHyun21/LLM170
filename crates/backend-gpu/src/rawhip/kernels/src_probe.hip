
// dot intrinsic 프로브 — sdot8(i4x8) 확정 + ext_vector_type sdot2/4 시도
typedef short __attribute__((ext_vector_type(2))) s2v;
extern "C" __global__ void dp4a_probe(const unsigned* x, int* out) {
    out[0] = __ockl_sdot8((int)x[0], (int)x[1], 0, false);   // i4x8: 3
    out[1] = (int)__ockl_udot8(x[2], x[3], 0u, false);       // 5
    s2v a2 = {12, 0}, b2 = {17, 0};
    out[2] = __ockl_sdot2(a2, b2, 0, false);                 // 204
    c4v a4 = {12, 0, 0, 0}, b4 = {17, 0, 0, 0};
    out[3] = __ockl_sdot4(a4, b4, 0, false);                 // 204
    // 음수 레인: {1,1,1,1}·{-1,-1,-1,-1} = -4
    c4v o4 = {1, 1, 1, 1}, m4 = {-1, -1, -1, -1};
    out[4] = __ockl_sdot4(o4, m4, 0, false);
    // {-32,2,0,0}·{17,3,5,7} = -544+6 = -538
    c4v n4 = {-32, 2, 0, 0}, q4 = {17, 3, 5, 7};
    out[5] = __ockl_sdot4(n4, q4, 0, false);
    // bitcast 경로: {0,-32,0,2}·{7,17,5,3} = 0 - 544 + 0 + 6 = -538
    out[6] = dot4(0x02E00000u, 0x03110705u, 0);
    out[7] = dot4(0x02E00000u, 0x03110705u, 1000);           // 누적 확인 462
}

// 순수 대역폭 프로브 — gemm과 동일 그리드/레인/서브블록 접근, 읽기+XOR만.
extern "C" __global__ void bw_probe(const unsigned* w, double* part, int n_in, int n_out, int bsize) {
    int o = blockIdx.y + blockIdx.z * gridDim.y;  // 토큰=x축 — L2 행 재사용
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    int n_sub = n_in >> 5;
    int cnt = (n_sub + 63 - l) >> 6;
    int blocks = n_in >> 8;
    int row_base = o * blocks * bsize;
    unsigned acc = 0;
    for (int m = 0; m < cnt; m++) {
        int sb = l + (m << 6);
        int wb = row_base + (sb >> 3) * bsize;
        int qw = wb >> 2;
        acc ^= w[qw] ^ w[qw+1] ^ w[qw+2] ^ w[qw+3];
    }
    part[o * 64 + l] = (double)acc + (double)l + (double)o;
}



// dot4 루프-오버헤드 프로브 — 순수 레지스터 체인 vs L1/L2 로드 혼합.
extern "C" __global__ void dot_roof(const unsigned* xq, const unsigned* w,
                                    double* out, int mode, int iters, int n_in) {
    int l = threadIdx.x;
    unsigned a = xq[l], b = w[l];
    int acc = 0;
    if (mode == 0) {
        // 순수 레지스터 dot 체인 (루프 카운트만)
        for (int i = 0; i < iters; i++) acc = dot4(a, b, acc);
    } else if (mode == 1) {
        // L1/글로벌 로드 동반 (같은 주소 — 캐시 히트)
        for (int i = 0; i < iters; i++) {
            unsigned yv = xq[l];
            unsigned wv = w[l];
            acc = dot4(wv, yv, acc);
        }
    } else {
        // 스트라이드 로드 (실제 커널 유사 — 캐시라인 분산)
        for (int i = 0; i < iters; i++) {
            unsigned yv = xq[(l + i * 64) % (n_in >> 2)];
            unsigned wv = w[(l + i * 64) % (n_in >> 2)];
            acc = dot4(wv, yv, acc);
        }
    }
    if (acc == 0x7fffffff) out[0] = 1.0;
    out[1] = (double)acc;
}

// mfma 발행률 프로브 — rocwmma 16x16x16f32, 레지스터 상주 vs L1 피드.
extern "C" __global__ void mfma_roof(const half* a, const half* b,
                                     double* out, int mode, int iters, int n) {
    fragment<matrix_a, 16, 16, 16, half, row_major> fa;
    fragment<matrix_b, 16, 16, 16, half, row_major> fb;
    fragment<accumulator, 16, 16, 16, float> fc;
    fill_fragment(fc, 0.0f);
    load_matrix_sync(fa, a, 16);
    load_matrix_sync(fb, b, 16);
    if (mode == 0) {
        for (int i = 0; i < iters; i++) mma_sync(fc, fa, fb, fc);
    } else {
        for (int i = 0; i < iters; i++) {
            load_matrix_sync(fa, a + ((i * 37) % n) * 256, 16);
            load_matrix_sync(fb, b + ((i * 53) % n) * 256, 16);
            mma_sync(fc, fa, fb, fc);
        }
    }
    if (fc.x[0] == 1.0e30f) out[0] = 1.0;
    out[1] = (double)fc.x[0];
    out[2] = (double)__popcll(__ballot_sync(0xffffffffffffffffull, 1));
}
// 결정적 argmax — 최저 인덱스 동률 (CPU greedy: v > bv 엄격 대소와 동일).
// 2단계: 블록별 (최대·최저인덱스) → 단일 블록 결합.
typedef struct { float v; int i; } AM;
extern "C" __global__ void argmax64(const float* x, int n, int2* out) {
    // grid n개 블록? 아니 — 단일 커널: 64레인 × 스트라이드 스캔 후 트리.
    // vocab 248320: 64레인 × 3880 요소. f32 최대 비교는 결정적 (NaN 없음 전제).
    int l = threadIdx.x;
    float bv = -3.0e38f;
    int bi = 0x7fffffff;
    for (int i = l; i < n; i += 64) {
        float v = x[i];
        if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
    }
    // shared 교환: 하위 절반과 상위 절분 최저인덱스 병합
    __shared__ float sv[32];
    __shared__ int si[32];
    if (l >= 32) { sv[l - 32] = bv; si[l - 32] = bi; }
    __syncthreads();
    if (l < 32) {
        if (sv[l] > bv || (sv[l] == bv && si[l] < bi)) { bv = sv[l]; bi = si[l]; }
        // 셔플 트리 (32레인) — 최대·최저인덱스 병합
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_down_sync(0xffffffffffffffffull, bv, off);
            int oi = __shfl_down_sync(0xffffffffffffffffull, bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (l == 0) { out->x = bv; out->y = bi; }
    }
}

