// ── exl3_gemm2_kseg — 소형-T 점유 레버(plans/121): k-타일을 grid.y 세그먼트로 분할,
// 부분합을 had_out의 nseg 레이아웃 [T][kseg][n]에 기록 — had_out(nseg=kseg)이 합산.
extern "C" __global__ void exl3_gemm2_kseg(
    const unsigned* __restrict__ ah16,
    const unsigned* __restrict__ tre,
    float* __restrict__ s,          // [T][kseg][n] 부분합
    int ktiles, int ntiles, int K, int T, int kseg)
{
    __shared__ unsigned tre_s[8 * 48];
    __shared__ __half2 buf_a[64 * 20];
    __shared__ __half2 buf_b[128 * 20];
    constexpr int STRIDE = 20;
    int wg_n = blockIdx.x;
    int seg = blockIdx.y;
    int tid = threadIdx.x;
    int n = ntiles * 16;
    int row0 = wg_n * 64;
    int t0 = 0;
    int nrow = min(64, n - row0);
    int nt = min(128, T - t0);
    int words32 = 8 * K;
    int arow_words = ktiles * 8;
    int n_loc = tid & 63;
    int t_half = tid >> 6;
    int kbp = (ktiles >> 1);
    int kb_s = (int)((long)kbp * seg / kseg);
    int kb_e = (int)((long)kbp * (seg + 1) / kseg);
    float acc[64];
    #pragma unroll
    for (int i = 0; i < 64; i++) acc[i] = 0.0f;

    for (int kb = kb_s; kb < kb_e; kb++) {
        {
            int stg_words = 8 * words32;
            for (int w = tid; w < stg_words; w += 128) {
                int tidx = w / words32;
                int wq = w % words32;
                int ktl = tidx >> 2;
                int ntl = tidx & 3;
                tre_s[w] = tre[((long)(kb * 2 + ktl) * ntiles + (wg_n * 4 + ntl)) * words32 + wq];
            }
            __syncthreads();
        }
        {
            int e0 = tid * 16;
            int nl = e0 / 32;
            int k0 = e0 % 32;
            int c = nl & 15;
            int nt_hi = nl >> 4;
            #pragma unroll
            for (int q = 0; q < 8; q++) {
                int kA = k0 + 2 * q;
                int kB = kA + 1;
                float va = 0.0f, vb = 0.0f;
                if (nl < nrow) {
                    va = exl3_mul1_decode(exl3_tile_word(tre_s, words32, K, (kA >> 4) * 4 + nt_hi, kA & 15, c));
                    vb = exl3_mul1_decode(exl3_tile_word(tre_s, words32, K, (kB >> 4) * 4 + nt_hi, kB & 15, c));
                }
                buf_a[nl * STRIDE + k0 / 2 + q] = __floats2half2_rn(va, vb);
            }
        }
        {
            int e0 = tid * 16;
            int tl = e0 / 32;
            int k0 = e0 % 32;
            #pragma unroll
            for (int q = 0; q < 8; q++) {
                unsigned w = 0u;
                if (tl < nt)
                    w = ah16[(long)(t0 + tl) * arow_words + (kb * 32 + k0) / 2 + q];
                buf_b[tl * STRIDE + (k0 >> 1) + q] = *reinterpret_cast<__half2*>(&w);
            }
            __syncthreads();
        }
        {
            __half2 ar2[16];
            #pragma unroll
            for (int q = 0; q < 16; q++) ar2[q] = buf_a[n_loc * STRIDE + q];
            for (int t = 0; t < 64; t++) {
                int tt = t_half * 64 + t;
                if (tt >= nt) break;
                float sum = acc[t];
                #pragma unroll
                for (int q = 0; q < 16; q++) {
                    float2 bv = __half22float2(buf_b[tt * STRIDE + q]);
                    float2 av = __half22float2(ar2[q]);
                    sum += av.x * bv.x + av.y * bv.y;
                }
                acc[t] = sum;
            }
        }
        __syncthreads();
    }
    for (int t = 0; t < 64; t++) {
        int tt = t_half * 64 + t;
        if (tt >= nt) break;
        if (n_loc < nrow)
            s[((long)(t0 + tt) * kseg + seg) * n + row0 + n_loc] = acc[t];
    }
}

// ── exl3_gemm2_mma (plans/127 B) — Q4-MMQ(gemm_q5k_wm) 구조의 트렐리스 이식 ──
// 스칼라 2.8TF의 구조적 한계(스레드당 f16 FMA 체인)를 mma_sync로 대체한다.
// 분할: 블록=64 가중행×64 토큰, 워프밴드(64스레드)=16 가중행 × wave32 페어
// (wpar)×32 토큰, fc[2] 파편 누산. [바이섹트 #2] 스테이징+A 디코드는 WMMA 판과
// 바이트 동일(16k 타일 루프·[ntl] 4슬롯·flat A16) — 오염 프로파일 비교용.
extern "C" __global__ void __launch_bounds__(256) exl3_gemm2_mma(
    const unsigned* __restrict__ ah16,  // f16쌍팩 [T][ktiles*8]
    const unsigned* __restrict__ tre,
    float* __restrict__ s,              // [T][n]
    int ktiles, int ntiles, int K, int T)
{
    __shared__ unsigned tre_s[8 * 48];
    __shared__ half A16[64][36];        // flat — WMMA 판과 동일 레이아웃
    __shared__ half B16[64][36];        // [토큰][32k+4]
    int bid = blockIdx.x;
    int wg_t = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 6;                 // 밴드 0..3
    int n = ntiles * 16;
    int row0 = bid * 64;
    if (row0 >= n) return;
    int t0 = wg_t * 64;
    int nrow = min(64, n - row0);
    int nt = min(64, T - t0);
    int words32 = 8 * K;                // K=krate
    int arow_words = ktiles * 8;
    int wpar = (tid >> 5) & 1;          // wave32 페어 분할(Q4 원판 동일)

    fragment<accumulator, 16, 16, 16, float> fc[2];
    #pragma unroll
    for (int i = 0; i < 2; i++) fill_fragment(fc[i], 0.0f);

    for (int kb = 0; kb < ktiles; kb++) {
        {
            int stg_words = 4 * words32;
            for (int w = tid; w < stg_words; w += 256) {
                int ntl = w / words32;
                int wq = w % words32;
                tre_s[w] = tre[((long)kb * ntiles + (bid * 4 + ntl)) * words32 + wq];
            }
            __syncthreads();
        }
        {
            for (int seg = tid; seg < 64 * 4; seg += 256) {
                int r = seg / 4;
                int c = (seg % 4) * 4;
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    int kk = c + e;
                    float v = exl3_mul1_decode(exl3_tile_word(
                        tre_s, words32, K, r >> 4, kk & 15, r & 15));
                    A16[r][kk] = __float2half(v);
                }
            }
        }
        {
            int ti = tid >> 2;
            int kc = tid & 3;
            if (ti < nt) {
                const unsigned* xt = ah16 + (long)(t0 + ti) * arow_words;
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    unsigned w = xt[kb * 8 + kc * 4 + e];
                    float2 uv = __half22float2(*reinterpret_cast<const __half2*>(&w));
                    B16[ti][kc * 8 + 2 * e] = __float2half(uv.x);
                    B16[ti][kc * 8 + 2 * e + 1] = __float2half(uv.y);
                }
            } else {
                #pragma unroll
                for (int e = 0; e < 8; e++) B16[ti][kc * 8 + e] = __float2half(0.0f);
            }
        }
        __syncthreads();
        {
            fragment<matrix_a, 16, 16, 16, half, row_major> fa;
            load_matrix_sync(fa, &A16[wid * 16][0], 36);
            #pragma unroll
            for (int tth = 0; tth < 2; tth++) {
                int tt = wpar * 2 + tth;
                fragment<matrix_b, 16, 16, 16, half, col_major> fb;
                load_matrix_sync(fb, &B16[tt * 16][0], 36);
                mma_sync(fc[tth], fa, fb, fc[tth]);
            }
        }
        __syncthreads();
    }
    // 드레인: D(i=n, j=t)를 s[T][n]에 저장 — 주소 = base + j*n + i → col_major,
    // ldm=n(row_major는 전치 저장 — 실측 인터리브 오염으로 확정).
    // 부분 t-타일(끝쪽 T가 16 배수가 아닌 경우)은 직접 저장이 [T][n] 경계를
    // 넘는다(실측 T=16 메모리 폴트) — 전체 타일만 직접, 부분 타일은 Ctmp 경유
    // 가드 드레인.
    int tail = T - t0;
    #pragma unroll
    for (int tth = 0; tth < 2; tth++) {
        int tt = wpar * 2 + tth;
        if (t0 + tt * 16 + 16 <= T) {
            store_matrix_sync(
                &s[(long)(t0 + tt * 16) * n + row0 + wid * 16],
                fc[tth], n, mem_col_major);
        }
    }
    if (tail < 64 && (tail & 15) != 0) {
        // 부분 타일 파편은 밴드(wid)마다 하나씩 존재 — Ctmp는 [64행][16열]로
        // 전 밴드 커버(종전 [16][68] 선언은 r2 16..63이 공유 밖 — T=21 行20 inf 실측).
        __shared__ float Ctmp[64][20];
        int tp = tail >> 4;              // 부분 타일 번호(유일)
        #pragma unroll
        for (int tth = 0; tth < 2; tth++) {
            if (wpar * 2 + tth == tp)
                store_matrix_sync(&Ctmp[wid * 16][0], fc[tth], 20, mem_row_major);
        }
        __syncthreads();
        int valid = tail - tp * 16;      // 타일 내 유효 토큰 수
        for (int e2 = tid; e2 < nrow * valid; e2 += 256) {
            int r2 = e2 / valid;
            int c2 = e2 % valid;
            s[(long)(t0 + tp * 16 + c2) * n + row0 + r2] = Ctmp[r2][c2];
        }
    }
}

// exl3_pos_bump — pp[0] += 1(어텐션 행 루프용, 캡처 그래프 내 pos 전진 — h2d 불가 대체).
