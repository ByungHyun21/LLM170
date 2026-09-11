// ─── 타일 배치 MMQ GEMM (mul_mat_q 구조) ───

// q5_K MMQ 포트 (ty13) — llama.cpp mul_mat_q 구조: I=64행×J=16토큰 타일,
// 256스레드(4×wave64), 스레드당 4토큰 파편. 언팩 가중·y 전부 shared,
// 가중 언팩이 16토큰에 상각. k-청크 256원소(8서브블록)/반복.
// 미러(q5k_mm)와 쌍: 스레드 q가 서브블록 {q, q+4, ...} 순서 f32 누산,
// 4-파편 tree 결합.
extern "C" __global__ void gemm_q5k_mm(const unsigned* xq, const unsigned* w,
                                       float* out, int n_in, int n_out, int xq_w, int t) {
    __shared__ unsigned wvs[8][64][8];    // [k청크 내 sb][행][워드] — 언팩 완료
    __shared__ float dsc[8][64][2];       // d·sc, dm·m
    __shared__ unsigned ytile[64][8][8];  // [토큰][sb][워드]
    __shared__ float ydt[64][8];          // [토큰][sb] yd
    __shared__ int qst[64][8];            // [토큰][sb] qsum16 합
    __shared__ float psum[4][64][32];     // [파편][행][토큰] f32 부분합

    int i = blockIdx.x * 64;              // 행 타일 기저
    int tid = threadIdx.x;                // 0..255
    int row = tid & 63;                   // 행 (스레드-행 고정)
    int wid = tid >> 6;                    // 파편 0..3 (토큰 4개 담당)
    if (i >= n_out) return;
    int nrow = min(64, n_out - i);
    int n_sub = n_in >> 5;
    int blocks = n_in >> 8;
    int qsb = (n_in >> 2) + (n_in >> 5);
    float acc[16];
    #pragma unroll
    for (int q = 0; q < 16; q++) acc[q] = 0.0f;
    for (int k0 = 0; k0 < n_sub; k0 += 8) {
        // ── 협력 스테이징: 256스레드 × (1 언팩 유닛 + 0.5 y 유닛) ──
        // 언팩: tid → (sb_local = tid>>5? ...) 256유닛 = 8sb×64행의 절반? 8*64=512 — 2패스
        for (int pass = 0; pass < 2; pass++) {
            int u = pass * 256 + tid;             // 0..511
            int sbl = u >> 6;                     // 0..7
            int r  = u & 63;                      // 0..63
            int sb = k0 + sbl;
            if (r < nrow && sb < n_sub) {
                int o = i + r;
                int js = sb & 7;
                int it = js >> 1;
                int half = js & 1;
                int wb = o * blocks * 176 + (sb >> 3) * 176;
                int wq = wb >> 2;
                unsigned w0 = w[wq];
                float d = bits_f16(w0 & 0xFFFFu);
                float dm = bits_f16(w0 >> 16);
                unsigned sc0 = w[wq+1], sc1 = w[wq+2], sc2 = w[wq+3];
                unsigned rr = (js & 3) * 8;
                unsigned b1 = js < 4 ? (sc0 >> rr) & 0xFFu : (sc1 >> rr) & 0xFFu;
                unsigned b4 = js < 4 ? (sc1 >> rr) & 0xFFu : (sc2 >> rr) & 0xFFu;
                unsigned bm = (sc0 >> rr) & 0xFFu;
                unsigned sc_v, m_v;
                if (js < 4) { sc_v = b1 & 63u; m_v = b4 & 63u; }
                else { sc_v = (b4 & 0xFu) | ((bm >> 6) << 4); m_v = (b4 >> 4) | ((b1 >> 6) << 4); }
                dsc[sbl][r][0] = d * (float)sc_v;
                dsc[sbl][r][1] = dm * (float)m_v;
                int qhw = wq + 4;
                unsigned nsh = half << 2;
                int sh = 2 * it + half;
                unsigned hb = (1u << sh) * 0x01010101u;
                int qlb = wq + 12 + it * 8;
                #pragma unroll
                for (int kk = 0; kk < 8; kk++) {
                    unsigned nib = (w[qlb + kk] >> nsh) & 0x0F0F0F0Fu;
                    unsigned bit = ((w[qhw + kk] & hb) >> sh) << 4;
                    wvs[sbl][r][kk] = nib | bit;
                }
            }
        }
        // y 스테이징: 32토큰×8sb = 256유닛 — tid 1패스 정확 커버
        {
            for (int pass = 0; pass < 2; pass++) {
            int u = pass * 256 + tid;                          // 0..255 = 32토큰×8sb
            {
                int ti = u >> 3;                  // 0..31
                int sbl = u & 7;                  // sb
                int sb = k0 + sbl;
                if (ti < t && sb < n_sub) {
                    const unsigned* xt = xq + ti * xq_w;
                    int xw = (sb << 5) >> 2;
                    #pragma unroll
                    for (int kk = 0; kk < 8; kk++) ytile[ti][sbl][kk] = xt[xw + kk];
                    ydt[ti][sbl] = __uint_as_float(xt[(n_in >> 2) + sb]);
                    qst[ti][sbl] = (int)xt[qsb + (sb << 1)] + (int)xt[qsb + (sb << 1) + 1];
                }
            }
            }
        }
        __syncthreads();
        // ── 계산: 소유형 (행 lane, 토큰 wid+4q), 순차 sb, wvs 레지스터 ──
        if (row < nrow) {
            #pragma unroll
            for (int sbl = 0; sbl < 8; sbl++) {
                int sb = k0 + sbl;
                if (sb >= n_sub) break;
                unsigned w0r = wvs[sbl][row][0];
                unsigned w1r = wvs[sbl][row][1];
                unsigned w2r = wvs[sbl][row][2];
                unsigned w3r = wvs[sbl][row][3];
                unsigned w4r = wvs[sbl][row][4];
                unsigned w5r = wvs[sbl][row][5];
                unsigned w6r = wvs[sbl][row][6];
                unsigned w7r = wvs[sbl][row][7];
                float d0 = dsc[sbl][row][0];
                float d1 = dsc[sbl][row][1];
                #pragma unroll
                for (int q = 0; q < 16; q++) {
                    int ti = wid + 4 * q;
                    if (ti >= t) break;
                    int isum = 0;
                    isum = dot4(w0r, ytile[ti][sbl][0], isum);
                    isum = dot4(w1r, ytile[ti][sbl][1], isum);
                    isum = dot4(w2r, ytile[ti][sbl][2], isum);
                    isum = dot4(w3r, ytile[ti][sbl][3], isum);
                    isum = dot4(w4r, ytile[ti][sbl][4], isum);
                    isum = dot4(w5r, ytile[ti][sbl][5], isum);
                    isum = dot4(w6r, ytile[ti][sbl][6], isum);
                    isum = dot4(w7r, ytile[ti][sbl][7], isum);
                    acc[q] += ydt[ti][sbl] * d0 * (float)isum;
                    acc[q] -= ydt[ti][sbl] * d1 * (float)qst[ti][sbl];
                }
            }
        }
        __syncthreads();
    }
    if (row < nrow) {
        #pragma unroll
        for (int q = 0; q < 16; q++) {
            int ti = wid + 4 * q;
            if (ti >= t) break;
            out[(size_t)ti * n_out + i + row] = acc[q];
        }
    }
}


// ─── MMQ 포트 (llama.cpp 타일: 64행×16토큰×256스레드, 파편 sb%4) ───
// 미러(q*_mm)와 쌍: 파편 f32 누산 + p0+p1+p2+p3.

// q5_K WMMA (ty13, fast path) — 듀얼모드: 비트계약 아님(스트림 검증).
// 구조: 64행×64토큰, 워프당 16행×4 B타일, sb 2배치, ldm=36 (≡4 mod 8).
// 스케일 접힘: A=(w_int·ds0−ds1) f16, B=y_int·yd f16 — min항 대수동치.
extern "C" __global__ void __launch_bounds__(256) gemm_q5k_wm(
        const unsigned* xq, const unsigned* w, float* out,
        int n_in, int n_out, int xq_w, int t) {
    // 수동 공유 레이아웃: A16 9216B + B16 9216B, CtmpF는 전체에 별칭(종결부)
    __shared__ char smem[18432];
    half (*A16)[4][16][36] = (half (*)[4][16][36])smem;
    half (*B16)[64][36] = (half (*)[64][36])(smem + 9216);
        int bid = blockIdx.x;
    int tid = threadIdx.x;
    int wid = tid >> 6;
    int lane = tid & 63;
    int i0 = bid * 64;
    if (i0 >= n_out) return;
    int nrow = min(64, n_out - i0);
    int row0 = wid * 16;
    int n_sub = n_in >> 5;
    int blocks = n_in >> 8;
    int qsb = (n_in >> 2) + (n_in >> 5);

    fragment<accumulator, 16, 16, 16, float> fc[2];
    #pragma unroll
    for (int i = 0; i < 2; i++) fill_fragment(fc[i], 0.0f);
    int wpar = (tid >> 5) & 1;   // wave32 페어 분할 (wid 그룹 내 2웨이브 중복 제거)

    for (int sbp = 0; sbp < n_sub; sbp += 2) {
        #pragma unroll
        for (int sbo = 0; sbo < 2; sbo++) {
            int sb = sbp + sbo;
            if (sb >= n_sub) break;
            // A16: 레인당 1행 × 8원소 — 스케일은 행당 1회
            {
                int rr = lane & 15;
                int kc = lane >> 4;          // k청크 0..3
                int row = row0 + rr;
                if (row < nrow && kc < 4) {
                    int o = i0 + row;
                    int js = sb & 7;
                    int it = js >> 1;
                    int half = js & 1;
                    int wq = ((o * blocks + (sb >> 3)) * 176) >> 2;
                    unsigned w0 = w[wq];
                    float d = bits_f16(w0 & 0xFFFFu);
                    float dm = bits_f16(w0 >> 16);
                    unsigned sc0 = w[wq+1], sc1 = w[wq+2], sc2 = w[wq+3];
                    unsigned rr2 = (js & 3) * 8;
                    unsigned b1 = js < 4 ? (sc0 >> rr2) & 0xFFu : (sc1 >> rr2) & 0xFFu;
                    unsigned b4 = js < 4 ? (sc1 >> rr2) & 0xFFu : (sc2 >> rr2) & 0xFFu;
                    unsigned bm = (sc0 >> rr2) & 0xFFu;
                    unsigned sc_v, m_v;
                    if (js < 4) { sc_v = b1 & 63u; m_v = b4 & 63u; }
                    else { sc_v = (b4 & 0xFu) | ((bm >> 6) << 4); m_v = (b4 >> 4) | ((b1 >> 6) << 4); }
                    float ds0 = d * (float)sc_v;
                    float ds1 = dm * (float)m_v;
                    unsigned nsh = half << 2;
                    int sh = 2 * it + half;
                    int qbase = wq + 12 + it * 8;
                    int qhbase = wq + 4;
                    #pragma unroll
                    for (int e = 0; e < 8; e++) {
                        int k = kc * 8 + e;
                        unsigned byte_v = (w[qbase + (k >> 2)] >> ((k & 3) * 8)) & 0xFFu;
                        unsigned nib = (byte_v >> nsh) & 0xFu;
                        unsigned hi = (w[qhbase + (k >> 2)] >> ((k & 3) * 8 + sh)) & 1u;
                        float wv = (float)nib + (hi ? 16.0f : 0.0f);
                        A16[sbo][wid][rr][k] = __float2half(wv * ds0 - ds1);
                    }
                }
            }
            (void)qsb;
            // B16: 스레드당 1토큰 8원소 — yw/yd 로드 1회
            {
                int ti = tid >> 2;
                int kc = tid & 3;
                if (ti < t) {
                    const unsigned* xt = xq + ti * xq_w;
                    int xw = (sb << 5) >> 2;
                    unsigned ywa = xt[xw + 2 * kc];
                    unsigned ywb = xt[xw + 2 * kc + 1];
                    float yd = __uint_as_float(xt[(n_in >> 2) + sb]);
                    #pragma unroll
                    for (int e = 0; e < 8; e++) {
                        int k = kc * 8 + e;
                        unsigned yw = ((e >> 2) == 0) ? ywa : ywb;
                        int b = sext8((yw >> ((k & 3) * 8)) & 0xFFu);
                        B16[sbo][ti][k] = __float2half((float)b * yd);
                    }
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int sbo = 0; sbo < 2; sbo++) {
            #pragma unroll
            for (int ks = 0; ks < 2; ks++) {
                fragment<matrix_a, 16, 16, 16, half, row_major> fa;
                load_matrix_sync(fa, &A16[sbo][wid][0][ks * 16], 36);
                #pragma unroll
                for (int tth = 0; tth < 2; tth++) {
                    int tt = wpar * 2 + tth;
                    fragment<matrix_b, 16, 16, 16, half, col_major> fb;
                    load_matrix_sync(fb, &B16[sbo][tt * 16][ks * 16], 36);
                    mma_sync(fc[tth], fa, fb, fc[tth]);
                }
            }
        }
        __syncthreads();
    }
    // Ctmp는 A16+B16(18.4KB)에 별칭 — 스테이징 데이터는 루프 후 사멸
    // 직접 드레인 (fc.x → out, Ctmp 제거): lane은 wave32 기준
    #pragma unroll
    for (int tth = 0; tth < 2; tth++) {
        int tt = wpar * 2 + tth;
        #pragma unroll
        for (int sl = 0; sl < 8; sl++) {
            int pp2 = (tid & 31) + (sl << 5);
            int dr = pp2 >> 4, dc = pp2 & 15;
            int gr = row0 + dr, gc = tt * 16 + dc;
            if (gr < nrow && gc < t)
                out[(size_t)gc * n_out + i0 + gr] = fc[tth].x[sl];
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) gemm_q4k_wm(
        const unsigned* xq, const unsigned* w, float* out,
        int n_in, int n_out, int xq_w, int t) {
    __shared__ char smem[18432];
    half (*A16)[4][16][36] = (half (*)[4][16][36])smem;
    half (*B16)[64][36] = (half (*)[64][36])(smem + 9216);
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int wid = tid >> 6;
    int lane = tid & 63;
    int i0 = bid * 64;
    if (i0 >= n_out) return;
    int nrow = min(64, n_out - i0);
    int row0 = wid * 16;
    int n_sub = n_in >> 5;
    int blocks = n_in >> 8;
    int qsb = (n_in >> 2) + (n_in >> 5);

    fragment<accumulator, 16, 16, 16, float> fc[2];
    #pragma unroll
    for (int i = 0; i < 2; i++) fill_fragment(fc[i], 0.0f);
    int wpar = (tid >> 5) & 1;   // wave32 페어 분할 (wid 그룹 내 2웨이브 중복 제거)

    for (int sbp = 0; sbp < n_sub; sbp += 2) {
        #pragma unroll
        for (int sbo = 0; sbo < 2; sbo++) {
            int sb = sbp + sbo;
            if (sb >= n_sub) break;
            // A16: 레인당 1행 × 8원소 — 스케일은 행당 1회
            {
                int rr = lane & 15;
                int kc = lane >> 4;          // k청크 0..3
                int row = row0 + rr;
                if (row < nrow && kc < 4) {
                    int o = i0 + row;
                    int js = sb & 7;
                    int it = js >> 1;
                    int half = js & 1;
                    int wq = ((o * blocks + (sb >> 3)) * 144) >> 2;
                    unsigned w0 = w[wq];
                    float d = bits_f16(w0 & 0xFFFFu);
                    float dm = bits_f16(w0 >> 16);
                    unsigned sc0 = w[wq+1], sc1 = w[wq+2], sc2 = w[wq+3];
                    unsigned rr2 = (js & 3) * 8;
                    unsigned b1 = js < 4 ? (sc0 >> rr2) & 0xFFu : (sc1 >> rr2) & 0xFFu;
                    unsigned b4 = js < 4 ? (sc1 >> rr2) & 0xFFu : (sc2 >> rr2) & 0xFFu;
                    unsigned bm = (sc0 >> rr2) & 0xFFu;
                    unsigned sc_v, m_v;
                    if (js < 4) { sc_v = b1 & 63u; m_v = b4 & 63u; }
                    else { sc_v = (b4 & 0xFu) | ((bm >> 6) << 4); m_v = (b4 >> 4) | ((b1 >> 6) << 4); }
                    float ds0 = d * (float)sc_v;
                    float ds1 = dm * (float)m_v;
                    unsigned nsh = half << 2;
                    int qbase = wq + 4 + it * 8;
                    #pragma unroll
                    for (int e = 0; e < 8; e++) {
                        int k = kc * 8 + e;
                        unsigned byte_v = (w[qbase + (k >> 2)] >> ((k & 3) * 8)) & 0xFFu;
                        unsigned nib = (byte_v >> nsh) & 0xFu;
                        A16[sbo][wid][rr][k] = __float2half((float)nib * ds0 - ds1);
                    }
                }
            }
            (void)qsb;
            // B16: 스레드당 1토큰 8원소 — yw/yd 로드 1회
            {
                int ti = tid >> 2;
                int kc = tid & 3;
                if (ti < t) {
                    const unsigned* xt = xq + ti * xq_w;
                    int xw = (sb << 5) >> 2;
                    unsigned ywa = xt[xw + 2 * kc];
                    unsigned ywb = xt[xw + 2 * kc + 1];
                    float yd = __uint_as_float(xt[(n_in >> 2) + sb]);
                    #pragma unroll
                    for (int e = 0; e < 8; e++) {
                        int k = kc * 8 + e;
                        unsigned yw = ((e >> 2) == 0) ? ywa : ywb;
                        int b = sext8((yw >> ((k & 3) * 8)) & 0xFFu);
                        B16[sbo][ti][k] = __float2half((float)b * yd);
                    }
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int sbo = 0; sbo < 2; sbo++) {
            #pragma unroll
            for (int ks = 0; ks < 2; ks++) {
                fragment<matrix_a, 16, 16, 16, half, row_major> fa;
                load_matrix_sync(fa, &A16[sbo][wid][0][ks * 16], 36);
                #pragma unroll
                for (int tth = 0; tth < 2; tth++) {
                    int tt = wpar * 2 + tth;
                    fragment<matrix_b, 16, 16, 16, half, col_major> fb;
                    load_matrix_sync(fb, &B16[sbo][tt * 16][ks * 16], 36);
                    mma_sync(fc[tth], fa, fb, fc[tth]);
                }
            }
        }
        __syncthreads();
    }
    // 직접 드레인 (fc.x → out, Ctmp 제거): lane은 wave32 기준
    #pragma unroll
    for (int tth = 0; tth < 2; tth++) {
        int tt = wpar * 2 + tth;
        #pragma unroll
        for (int sl = 0; sl < 8; sl++) {
            int pp2 = (tid & 31) + (sl << 5);
            int dr = pp2 >> 4, dc = pp2 & 15;
            int gr = row0 + dr, gc = tt * 16 + dc;
            if (gr < nrow && gc < t)
                out[(size_t)gc * n_out + i0 + gr] = fc[tth].x[sl];
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) gemm_q6k_wm(
        const unsigned* xq, const unsigned* w, float* out,
        int n_in, int n_out, int xq_w, int t) {
    __shared__ char smem[18432];
    half (*A16)[4][16][36] = (half (*)[4][16][36])smem;
    half (*B16)[64][36] = (half (*)[64][36])(smem + 9216);
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int wid = tid >> 6;
    int lane = tid & 63;
    int i0 = bid * 64;
    if (i0 >= n_out) return;
    int nrow = min(64, n_out - i0);
    int row0 = wid * 16;
    int n_sub = n_in >> 5;
    int blocks = n_in >> 8;
    int qsb = (n_in >> 2) + (n_in >> 5);

    fragment<accumulator, 16, 16, 16, float> fc[2];
    #pragma unroll
    for (int i = 0; i < 2; i++) fill_fragment(fc[i], 0.0f);
    int wpar = (tid >> 5) & 1;   // wave32 페어 분할 (wid 그룹 내 2웨이브 중복 제거)

    for (int sbp = 0; sbp < n_sub; sbp += 2) {
        #pragma unroll
        for (int sbo = 0; sbo < 2; sbo++) {
            int sb = sbp + sbo;
            if (sb >= n_sub) break;
            // A16: 레인당 1행 × 8원소 — 스케일은 행당 1회
            {
                int rr = lane & 15;
                int kc = lane >> 4;          // k청크 0..3
                int row = row0 + rr;
                if (row < nrow && kc < 4) {
                    int o = i0 + row;
                    #pragma unroll
                    for (int e = 0; e < 8; e++) {
                        int k = kc * 8 + e;
                        int g = sb * 2 + (k >> 4);      // 32열 sb = 그룹 2개
                        int kloc = k & 15;
                        int gr = g & 15;
                        int blk = g >> 4;
                        int h = gr >> 3;
                        int src = (gr & 7) >> 1;
                        int p2 = gr & 1;
                        int wbase = o * blocks * 210 + blk * 210;
                        float d = f16w(w, wbase + 208);
                        int sc = sext8(byte(w, wbase + 192 + gr));
                        float ds0 = d * (float)sc;
                        int ql_rel = h * 64 + p2 * 16 + ((src & 1) << 5);
                        int qh_rel = 128 + h * 32 + p2 * 16;
                        int nsh = (src < 2) ? 0 : 4;
                        int hsh = 2 * src;
                        int qlo = ql_rel + ((kloc >> 2) << 2);
                        int qho = qh_rel + ((kloc >> 2) << 2);
                        bool al = (wbase & 3) == 0;
                        int kk2 = kloc & 3;
                        unsigned qv, hv;
                        if (al) {
                            qv = w[(wbase + qlo) >> 2];
                            hv = w[(wbase + qho) >> 2];
                        } else {
                            int qw0 = (wbase + qlo) >> 2;
                            int hw0 = (wbase + qho) >> 2;
                            qv = (w[qw0] >> 16) | (w[qw0 + 1] << 16);
                            hv = (w[hw0] >> 16) | (w[hw0 + 1] << 16);
                        }
                        unsigned nib = (qv >> (kk2 * 8 + nsh)) & 0xFu;
                        unsigned hi = (hv >> (kk2 * 8 + hsh)) & 0x3u;
                        float v = (float)(nib + (hi << 4)) - 32.0f;
                        A16[sbo][wid][rr][k] = __float2half(v * ds0);
                    }
                }
            }
            (void)qsb;
            // B16: 스레드당 1토큰 8원소 — yw/yd 로드 1회
            {
                int ti = tid >> 2;
                int kc = tid & 3;
                if (ti < t) {
                    const unsigned* xt = xq + ti * xq_w;
                    int xw = (sb << 5) >> 2;
                    unsigned ywa = xt[xw + 2 * kc];
                    unsigned ywb = xt[xw + 2 * kc + 1];
                    float yd = __uint_as_float(xt[(n_in >> 2) + sb]);
                    #pragma unroll
                    for (int e = 0; e < 8; e++) {
                        int k = kc * 8 + e;
                        unsigned yw = ((e >> 2) == 0) ? ywa : ywb;
                        int b = sext8((yw >> ((k & 3) * 8)) & 0xFFu);
                        B16[sbo][ti][k] = __float2half((float)b * yd);
                    }
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int sbo = 0; sbo < 2; sbo++) {
            #pragma unroll
            for (int ks = 0; ks < 2; ks++) {
                fragment<matrix_a, 16, 16, 16, half, row_major> fa;
                load_matrix_sync(fa, &A16[sbo][wid][0][ks * 16], 36);
                #pragma unroll
                for (int tth = 0; tth < 2; tth++) {
                    int tt = wpar * 2 + tth;
                    fragment<matrix_b, 16, 16, 16, half, col_major> fb;
                    load_matrix_sync(fb, &B16[sbo][tt * 16][ks * 16], 36);
                    mma_sync(fc[tth], fa, fb, fc[tth]);
                }
            }
        }
        __syncthreads();
    }
    // 직접 드레인 (fc.x → out, Ctmp 제거): lane은 wave32 기준
    #pragma unroll
    for (int tth = 0; tth < 2; tth++) {
        int tt = wpar * 2 + tth;
        #pragma unroll
        for (int sl = 0; sl < 8; sl++) {
            int pp2 = (tid & 31) + (sl << 5);
            int dr = pp2 >> 4, dc = pp2 & 15;
            int gr = row0 + dr, gc = tt * 16 + dc;
            if (gr < nrow && gc < t)
                out[(size_t)gc * n_out + i0 + gr] = fc[tth].x[sl];
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) gemm_xs_wm(
        const unsigned* xq, const unsigned* w, float* out, const unsigned* ktab2,
        int n_in, int n_out, int xq_w, int t) {
    __shared__ char smem[19456];
    half (*A16)[4][16][36] = (half (*)[4][16][36])smem;
    unsigned* kt_s = (unsigned*)(smem + 18432);
    for (int i2 = threadIdx.x; i2 < 256; i2 += 256) kt_s[i2] = ktab2[i2];
    __syncthreads();   // LUT 가시성 — A 스테이징 룩업 전 필수 (경합 수정)
    half (*B16)[64][36] = (half (*)[64][36])(smem + 9216);
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int wid = tid >> 6;
    int lane = tid & 63;
    int i0 = bid * 64;
    if (i0 >= n_out) return;
    int nrow = min(64, n_out - i0);
    int row0 = wid * 16;
    int n_sub = n_in >> 5;
    int blocks = n_in >> 8;
    int qsb = (n_in >> 2) + (n_in >> 5);

    fragment<accumulator, 16, 16, 16, float> fc[2];
    #pragma unroll
    for (int i = 0; i < 2; i++) fill_fragment(fc[i], 0.0f);
    int wpar = (tid >> 5) & 1;   // wave32 페어 분할 (wid 그룹 내 2웨이브 중복 제거)

    for (int sbp = 0; sbp < n_sub; sbp += 2) {
        #pragma unroll
        for (int sbo = 0; sbo < 2; sbo++) {
            int sb = sbp + sbo;
            if (sb >= n_sub) break;
            // A16: 레인당 1행 × 8원소 — 스케일은 행당 1회
            {
                int rr = lane & 15;
                int kc = lane >> 4;          // k청크 0..3
                int row = row0 + rr;
                if (row < nrow && kc < 4) {
                    int o = i0 + row;
                    int ib = sb & 7;
                    int wb = o * blocks * 136 + (sb >> 3) * 136;
                    int wq = wb >> 2;
                    unsigned w0 = w[wq];
                    float d = bits_f16(w0 & 0xFFFFu);
                    int ls = (int)((w[wq + 1] >> (((ib >> 1) * 8 + (ib & 1) * 4))) & 0xFu)
                          | (int)((((w0 >> 16) >> (2 * ib)) & 3u) << 4);
                    float ds0 = d * (float)(ls - 32);
                    int qw = (wb + 8 + ib * 16) >> 2;
                    #pragma unroll
                    for (int e = 0; e < 8; e++) {
                        int k = kc * 8 + e;
                        unsigned qv = w[qw + ((k & 15) >> 2)];
                        unsigned byte_v = (qv >> ((k & 3) * 8)) & 0xFFu;
                        unsigned tt2 = kt_s[byte_v];
                        int val = (k < 16) ? sext8(tt2 & 0xFFu) : sext8(tt2 >> 8);
                        A16[sbo][wid][rr][k] = __float2half((float)val * ds0);
                    }
                }
            }
            (void)qsb;
            // B16: 스레드당 1토큰 8원소 — yw/yd 로드 1회
            {
                int ti = tid >> 2;
                int kc = tid & 3;
                if (ti < t) {
                    const unsigned* xt = xq + ti * xq_w;
                    int xw = (sb << 5) >> 2;
                    unsigned ywa = xt[xw + 2 * kc];
                    unsigned ywb = xt[xw + 2 * kc + 1];
                    float yd = __uint_as_float(xt[(n_in >> 2) + sb]);
                    #pragma unroll
                    for (int e = 0; e < 8; e++) {
                        int k = kc * 8 + e;
                        unsigned yw = ((e >> 2) == 0) ? ywa : ywb;
                        int b = sext8((yw >> ((k & 3) * 8)) & 0xFFu);
                        B16[sbo][ti][k] = __float2half((float)b * yd);
                    }
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int sbo = 0; sbo < 2; sbo++) {
            #pragma unroll
            for (int ks = 0; ks < 2; ks++) {
                fragment<matrix_a, 16, 16, 16, half, row_major> fa;
                load_matrix_sync(fa, &A16[sbo][wid][0][ks * 16], 36);
                #pragma unroll
                for (int tth = 0; tth < 2; tth++) {
                    int tt = wpar * 2 + tth;
                    fragment<matrix_b, 16, 16, 16, half, col_major> fb;
                    load_matrix_sync(fb, &B16[sbo][tt * 16][ks * 16], 36);
                    mma_sync(fc[tth], fa, fb, fc[tth]);
                }
            }
        }
        __syncthreads();
    }
    // 직접 드레인 (fc.x → out, Ctmp 제거): lane은 wave32 기준
    #pragma unroll
    for (int tth = 0; tth < 2; tth++) {
        int tt = wpar * 2 + tth;
        #pragma unroll
        for (int sl = 0; sl < 8; sl++) {
            int pp2 = (tid & 31) + (sl << 5);
            int dr = pp2 >> 4, dc = pp2 & 15;
            int gr = row0 + dr, gc = tt * 16 + dc;
            if (gr < nrow && gc < t)
                out[(size_t)gc * n_out + i0 + gr] = fc[tth].x[sl];
        }
    }
}

// q4_K MMQ (ty12) — 니블만, qs wq+4+it*8
extern "C" __global__ void gemm_q4k_mm(const unsigned* xq, const unsigned* w,
                                       float* out, int n_in, int n_out, int xq_w, int t) {
    __shared__ unsigned wvs[8][64][8];
    __shared__ float dsc[8][64][2];
    __shared__ unsigned ytile[64][8][8];
    __shared__ float ydt[64][8];
    __shared__ int qst[64][8];
        int i = blockIdx.x * 64;
    int tid = threadIdx.x;
    int row = tid & 63;
    int wid = tid >> 6;
    if (i >= n_out) return;
    int nrow = min(64, n_out - i);
    int n_sub = n_in >> 5;
    int blocks = n_in >> 8;
    int qsb = (n_in >> 2) + (n_in >> 5);
    float acc[16];
    #pragma unroll
    for (int q = 0; q < 16; q++) acc[q] = 0.0f;
    for (int k0 = 0; k0 < n_sub; k0 += 8) {
        for (int pass = 0; pass < 2; pass++) {
            int u = pass * 256 + tid;
            int sbl = u >> 6;
            int r = u & 63;
            int sb = k0 + sbl;
            if (r < nrow && sb < n_sub) {
                int o = i + r;
                int js = sb & 7;
                int it = js >> 1;
                int half = js & 1;
                int wb = o * blocks * 144 + (sb >> 3) * 144;
                int wq = wb >> 2;
                unsigned w0 = w[wq];
                float d = bits_f16(w0 & 0xFFFFu);
                float dm = bits_f16(w0 >> 16);
                unsigned sc0 = w[wq+1], sc1 = w[wq+2], sc2 = w[wq+3];
                unsigned rr = (js & 3) * 8;
                unsigned b1 = js < 4 ? (sc0 >> rr) & 0xFFu : (sc1 >> rr) & 0xFFu;
                unsigned b4 = js < 4 ? (sc1 >> rr) & 0xFFu : (sc2 >> rr) & 0xFFu;
                unsigned bm = (sc0 >> rr) & 0xFFu;
                unsigned sc_v, m_v;
                if (js < 4) { sc_v = b1 & 63u; m_v = b4 & 63u; }
                else { sc_v = (b4 & 0xFu) | ((bm >> 6) << 4); m_v = (b4 >> 4) | ((b1 >> 6) << 4); }
                dsc[sbl][r][0] = d * (float)sc_v;
                dsc[sbl][r][1] = dm * (float)m_v;
                unsigned nsh = half << 2;
                int qlb = wq + 4 + it * 8;
                #pragma unroll
                for (int kk = 0; kk < 8; kk++)
                    wvs[sbl][r][kk] = (w[qlb + kk] >> nsh) & 0x0F0F0F0Fu;
            }
        }
        {
            for (int pass = 0; pass < 2; pass++) {
            int u = pass * 256 + tid;
            {
                int ti = u >> 3;   // 0..31
                int sbl = u & 7;
                int sb = k0 + sbl;
                if (ti < t && sb < n_sub) {
                    const unsigned* xt = xq + ti * xq_w;
                    int xw = (sb << 5) >> 2;
                    #pragma unroll
                    for (int kk = 0; kk < 8; kk++) ytile[ti][sbl][kk] = xt[xw + kk];
                    ydt[ti][sbl] = __uint_as_float(xt[(n_in >> 2) + sb]);
                    qst[ti][sbl] = (int)xt[qsb + (sb << 1)] + (int)xt[qsb + (sb << 1) + 1];
                }
            }
            }
        }
        __syncthreads();
        if (row < nrow) {
            #pragma unroll
            for (int sbl = 0; sbl < 8; sbl++) {
                int sb = k0 + sbl;
                if (sb >= n_sub) break;
                unsigned w0r = wvs[sbl][row][0];
                unsigned w1r = wvs[sbl][row][1];
                unsigned w2r = wvs[sbl][row][2];
                unsigned w3r = wvs[sbl][row][3];
                unsigned w4r = wvs[sbl][row][4];
                unsigned w5r = wvs[sbl][row][5];
                unsigned w6r = wvs[sbl][row][6];
                unsigned w7r = wvs[sbl][row][7];
                float d0 = dsc[sbl][row][0];
                float d1 = dsc[sbl][row][1];
                #pragma unroll
                for (int q = 0; q < 16; q++) {
                    int ti = wid + 4 * q;
                    if (ti >= t) break;
                    int isum = 0;
                    isum = dot4(w0r, ytile[ti][sbl][0], isum);
                    isum = dot4(w1r, ytile[ti][sbl][1], isum);
                    isum = dot4(w2r, ytile[ti][sbl][2], isum);
                    isum = dot4(w3r, ytile[ti][sbl][3], isum);
                    isum = dot4(w4r, ytile[ti][sbl][4], isum);
                    isum = dot4(w5r, ytile[ti][sbl][5], isum);
                    isum = dot4(w6r, ytile[ti][sbl][6], isum);
                    isum = dot4(w7r, ytile[ti][sbl][7], isum);
                    acc[q] += ydt[ti][sbl] * d0 * (float)isum;
                    acc[q] -= ydt[ti][sbl] * d1 * (float)qst[ti][sbl];
                }
            }
        }
        __syncthreads();
    }
    if (row < nrow) {
        #pragma unroll
        for (int q = 0; q < 16; q++) {
            int ti = wid + 4 * q;
            if (ti >= t) break;
            out[(size_t)ti * n_out + i + row] = acc[q];
        }
    }
}


// q6_K MMQ (ty14) — 16원소 그룹: w = nib + 16·hi2 − 32 (qsum 테이블 분해).
// 그룹 2개/서브블록? 아님 — q6_K 그룹=16원소, 서브블록 32원소=2그룹.
// 여기선 k청크를 16원소 그룹 16개(256원소)로: wvs[16그룹][64행][4워드].
extern "C" __global__ void gemm_q6k_mm(const unsigned* xq, const unsigned* w,
                                       float* out, int n_in, int n_out, int xq_w, int t) {
    __shared__ unsigned wvs[16][64][4];
    __shared__ float dsc[16][64];          // d·sc 사전곱
    __shared__ unsigned ytile[64][16][4];  // [토큰][그룹][워드]
    __shared__ float ydt[64][16];
    __shared__ int qst[64][16];
        int i = blockIdx.x * 64;
    int tid = threadIdx.x;
    int row = tid & 63;
    int wid = tid >> 6;
    if (i >= n_out) return;
    int nrow = min(64, n_out - i);
    int n_g = n_in >> 4;
    int blocks = n_in >> 8;
    int qsb = (n_in >> 2) + (n_in >> 5);
    float acc[16];
    #pragma unroll
    for (int q = 0; q < 16; q++) acc[q] = 0.0f;
    for (int k0 = 0; k0 < n_g; k0 += 16) {
        for (int pass = 0; pass < 2; pass++) {
            int u = pass * 256 + tid;
            int gl = u >> 4;               // 그룹 0..15
            int r = (u >> 0) & 15;         // 하위 4비트 — 잘못… u&15
            r = u & 63;                    // 행 재정의: u = gl*64 + r → gl = u>>6 (0..7?) 
            // 재설계: 16그룹×64행 = 1024유닛 — 4패스
            (void)r; (void)gl;
        }
        // 간단화: 1024 유닛 = 4패스 × 256
        for (int pass = 0; pass < 4; pass++) {
            int u = pass * 256 + tid;      // 0..1023
            int gl = u >> 6;               // 0..15
            int r = u & 63;
            int g = k0 + gl;
            if (r < nrow && g < n_g) {
                int o = i + r;
                int blk = g >> 4;
                int kloc = g & 15;
                int h = kloc >> 3;
                int src = (kloc & 7) >> 1;
                int p2 = kloc & 1;
                int wb = o * blocks * 210 + blk * 210;
                float d = f16w(w, wb + 208);
                int sc = sext8(byte(w, wb + 192 + kloc));
                dsc[gl][r] = d * (float)sc;
                int ql_rel = h * 64 + p2 * 16 + ((src & 1) << 5);
                int qh_rel = 128 + h * 32 + p2 * 16;
                bool al = (wb & 3) == 0;
                int qlw = (wb + ql_rel) >> 2;
                int qhw = (wb + qh_rel) >> 2;
                unsigned qa[4], ha[4];
                if (al) {
                    qa[0]=w[qlw]; qa[1]=w[qlw+1]; qa[2]=w[qlw+2]; qa[3]=w[qlw+3];
                    ha[0]=w[qhw]; ha[1]=w[qhw+1]; ha[2]=w[qhw+2]; ha[3]=w[qhw+3];
                } else {
                    qa[0]=(w[qlw]>>16)|(w[qlw+1]<<16); qa[1]=(w[qlw+1]>>16)|(w[qlw+2]<<16);
                    qa[2]=(w[qlw+2]>>16)|(w[qlw+3]<<16); qa[3]=(w[qlw+3]>>16)|(w[qlw+4]<<16);
                    ha[0]=(w[qhw]>>16)|(w[qhw+1]<<16); ha[1]=(w[qhw+1]>>16)|(w[qhw+2]<<16);
                    ha[2]=(w[qhw+2]>>16)|(w[qhw+3]<<16); ha[3]=(w[qhw+3]>>16)|(w[qhw+4]<<16);
                }
                unsigned nibm = ((src == 0 || src == 1) ? 0x0F0F0F0Fu : 0xF0F0F0F0u);
                int nsh = (src == 0 || src == 1) ? 0 : 4;
                int hsh = 2 * src;
                #pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    unsigned nib = (qa[kk] >> nsh) & 0x0F0F0F0Fu;
                    unsigned hi2 = (ha[kk] >> hsh) & 0x03030303u;
                    wvs[gl][r][kk] = nib + (hi2 << 4);
                }
                (void)nibm;
            }
        }
        {
            // y: 64토큰×16그룹 = 1024유닛 — 4패스×256
            for (int pass = 0; pass < 4; pass++) {
                int u = pass * 256 + tid;
                int ti = u >> 4;
                int gl = u & 15;
                int g = k0 + gl;
                if (ti < t && g < n_g) {
                    const unsigned* xt = xq + ti * xq_w;
                    int xw = (g << 4) >> 2;
                    #pragma unroll
                    for (int kk = 0; kk < 4; kk++) ytile[ti][gl][kk] = xt[xw + kk];
                    ydt[ti][gl] = __uint_as_float(xt[(n_in >> 2) + (g >> 1)]);
                    qst[ti][gl] = (int)xt[qsb + g];  // q16[g] — 16그룹 인덱스
                }
            }
        }
        __syncthreads();
        if (row < nrow) {
            // 소유형: 순차 gl, wvs 4워드+dsc 레지스터, 토큰 wid+4q
            #pragma unroll
            for (int gl = 0; gl < 16; gl++) {
                int g = k0 + gl;
                if (g >= n_g) break;
                unsigned w0r = wvs[gl][row][0];
                unsigned w1r = wvs[gl][row][1];
                unsigned w2r = wvs[gl][row][2];
                unsigned w3r = wvs[gl][row][3];
                float d0 = dsc[gl][row];
                #pragma unroll
                for (int q = 0; q < 16; q++) {
                    int ti = wid + 4 * q;
                    if (ti >= t) break;
                    int isum = 0;
                    isum = dot4(w0r, ytile[ti][gl][0], isum);
                    isum = dot4(w1r, ytile[ti][gl][1], isum);
                    isum = dot4(w2r, ytile[ti][gl][2], isum);
                    isum = dot4(w3r, ytile[ti][gl][3], isum);
                    isum -= 32 * qst[ti][gl];
                    acc[q] += ydt[ti][gl] * d0 * (float)isum;
                }
            }
        }
        __syncthreads();
    }
    if (row < nrow) {
        #pragma unroll
        for (int q = 0; q < 16; q++) {
            int ti = wid + 4 * q;
            if (ti >= t) break;
            out[(size_t)ti * n_out + i + row] = acc[q];
        }
    }
}


// iq4_xs MMQ (ty23) — ktab2 룩업 언팩, 공유 LUT.
extern "C" __global__ void gemm_xs_mm(const unsigned* xq, const unsigned* w,
                                      float* out, const unsigned* ktab2, int n_in, int n_out,
                                      int xq_w, int t) {
    __shared__ unsigned wvs[8][64][8];
    __shared__ float dsc[8][64];
    __shared__ unsigned ytile[64][8][8];
    __shared__ float ydt[64][8];
        __shared__ unsigned kt_s[256];
    for (int a = threadIdx.x; a < 256; a += 256) kt_s[a] = ktab2[a];
    int i = blockIdx.x * 64;
    int tid = threadIdx.x;
    int row = tid & 63;
    int wid = tid >> 6;
    if (i >= n_out) return;
    int nrow = min(64, n_out - i);
    int n_sub = n_in >> 5;
    int blocks = n_in >> 8;
    float acc[16];
    #pragma unroll
    for (int q = 0; q < 16; q++) acc[q] = 0.0f;
    for (int k0 = 0; k0 < n_sub; k0 += 8) {
        for (int pass = 0; pass < 2; pass++) {
            int u = pass * 256 + tid;
            int sbl = u >> 6;
            int r = u & 63;
            int sb = k0 + sbl;
            if (r < nrow && sb < n_sub) {
                int o = i + r;
                int ib = sb & 7;
                int wb = o * blocks * 136 + (sb >> 3) * 136;
                int wq = wb >> 2;
                unsigned w0 = w[wq];
                float d = bits_f16(w0 & 0xFFFFu);
                int ls = (int)((w[wq + 1] >> (((ib >> 1) * 8 + (ib & 1) * 4))) & 0xFu)
                      | (int)((((w0 >> 16) >> (2 * ib)) & 3u) << 4);
                dsc[sbl][r] = d * (float)(ls - 32);
                int qw = (wb + 8 + ib * 16) >> 2;
                #pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    unsigned qv = w[qw + kk];
                    unsigned lov = 0, hiv = 0;
                    #pragma unroll
                    for (int b2 = 3; b2 >= 0; b2--) {
                        unsigned tt2 = kt_s[(qv >> (8 * b2)) & 0xFFu];
                        lov = (lov << 8) | (tt2 & 0xFFu);
                        hiv = (hiv << 8) | (tt2 >> 8);
                    }
                    wvs[sbl][r][kk * 2] = lov;
                    wvs[sbl][r][kk * 2 + 1] = hiv;
                }
            }
        }
        {
            for (int pass = 0; pass < 2; pass++) {
            int u = pass * 256 + tid;
            {
                int ti = u >> 3;   // 0..31
                int sbl = u & 7;
                int sb = k0 + sbl;
                if (ti < t && sb < n_sub) {
                    const unsigned* xt = xq + ti * xq_w;
                    int xw = (sb << 5) >> 2;
                    #pragma unroll
                    for (int kk = 0; kk < 8; kk++) ytile[ti][sbl][kk] = xt[xw + kk];
                    ydt[ti][sbl] = __uint_as_float(xt[(n_in >> 2) + sb]);
                }
            }
            }
        }
        __syncthreads();
        if (row < nrow) {
            // 소유형: 순차 sb, wvs 8워드+dsc 레지스터, hi는 y워드 k+4와 페어
            #pragma unroll
            for (int sbl = 0; sbl < 8; sbl++) {
                int sb = k0 + sbl;
                if (sb >= n_sub) break;
                unsigned w0r = wvs[sbl][row][0];
                unsigned w1r = wvs[sbl][row][1];
                unsigned w2r = wvs[sbl][row][2];
                unsigned w3r = wvs[sbl][row][3];
                unsigned w4r = wvs[sbl][row][4];
                unsigned w5r = wvs[sbl][row][5];
                unsigned w6r = wvs[sbl][row][6];
                unsigned w7r = wvs[sbl][row][7];
                float d0 = dsc[sbl][row];
                #pragma unroll
                for (int q = 0; q < 16; q++) {
                    int ti = wid + 4 * q;
                    if (ti >= t) break;
                    int isum = 0;
                    isum = dot4(w0r, ytile[ti][sbl][0], isum);
                    isum = dot4(w2r, ytile[ti][sbl][1], isum);
                    isum = dot4(w4r, ytile[ti][sbl][2], isum);
                    isum = dot4(w6r, ytile[ti][sbl][3], isum);
                    isum = dot4(w1r, ytile[ti][sbl][4], isum);
                    isum = dot4(w3r, ytile[ti][sbl][5], isum);
                    isum = dot4(w5r, ytile[ti][sbl][6], isum);
                    isum = dot4(w7r, ytile[ti][sbl][7], isum);
                    acc[q] += ydt[ti][sbl] * d0 * (float)isum;
                }
            }
        }
        __syncthreads();
    }
    if (row < nrow) {
        #pragma unroll
        for (int q = 0; q < 16; q++) {
            int ti = wid + 4 * q;
            if (ti >= t) break;
            out[(size_t)ti * n_out + i + row] = acc[q];
        }
    }
}


// q4_K (ty12) — 분할 형태, qh 없음 (qs 16..143)
extern "C" __global__ void gemm_q4k(const unsigned* xq, const unsigned* w,
                                    double* part, float* out, int n_in, int n_out, int xq_w) {
    int o = blockIdx.y + blockIdx.z * gridDim.y;  // 토큰=x축 — L2 행 재사용
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    xq += (int)blockIdx.x * xq_w;
    out += (int)blockIdx.x * n_out;
    int n_sub = n_in >> 5;
    int cnt = (n_sub + 63 - l) >> 6;
    int blocks = n_in >> 8;
    int row_base = o * blocks * 144;
    float acc = 0.0f;
    for (int m = 0; m < cnt; m++) {
        int sb = l + (m << 6);
        int js = sb & 7;
        int it = js >> 1;
        int half = js & 1;
        int wb = row_base + (sb >> 3) * 144;
        int wq = wb >> 2;
        unsigned w0 = w[wq];
        float d = bits_f16(w0 & 0xFFFFu);
        float dm = bits_f16(w0 >> 16);
        unsigned sc0 = w[wq+1], sc1 = w[wq+2], sc2 = w[wq+3];
        unsigned r = (js & 3) * 8;
        unsigned b_j   = js < 4 ? (sc0 >> r) & 0xFFu : (sc1 >> r) & 0xFFu;
        unsigned b_j4  = js < 4 ? (sc1 >> r) & 0xFFu : (sc2 >> r) & 0xFFu;
        unsigned b_jm4 = (sc0 >> r) & 0xFFu;
        unsigned sc_v, m_v;
        if (js < 4) { sc_v = b_j & 63u; m_v = b_j4 & 63u; }
        else {
            sc_v = (b_j4 & 0xFu) | ((b_jm4 >> 6) << 4);
            m_v  = (b_j4 >> 4) | ((b_j >> 6) << 4);
        }
        int qlb = wq + 4 + it * 8;
        unsigned q0 = w[qlb], q1 = w[qlb+1], q2 = w[qlb+2], q3 = w[qlb+3];
        unsigned q4 = w[qlb+4], q5 = w[qlb+5], q6 = w[qlb+6], q7 = w[qlb+7];
        int xw = (sb << 5) >> 2;
        unsigned y0 = xq[xw], y1 = xq[xw+1], y2 = xq[xw+2], y3 = xq[xw+3];
        unsigned y4 = xq[xw+4], y5 = xq[xw+5], y6 = xq[xw+6], y7 = xq[xw+7];
        int nsh = half << 2;
        int isum = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            unsigned qv = k < 4 ? (k==0?q0:k==1?q1:k==2?q2:q3) : (k==4?q4:k==5?q5:k==6?q6:q7);
            unsigned yv = k < 4 ? (k==0?y0:k==1?y1:k==2?y2:y3) : (k==4?y4:k==5?y5:k==6?y6:y7);
            unsigned nibw = (qv >> nsh) & 0x0F0F0F0Fu;
            isum = dot4(nibw, yv, isum);
        }
        int qsb = (n_in >> 2) + (n_in >> 5);
        int qsum = (int)xq[qsb + (sb << 1)] + (int)xq[qsb + (sb << 1) + 1];
        float yd = __uint_as_float(xq[(n_in >> 2) + sb]);
        acc += yd * (d * (float)sc_v) * (float)isum;
        acc -= yd * (dm * (float)m_v) * (float)qsum;
    }
    // 트리 환원 (미러 tree64와 동일 순서) — part 왕복 제거.
    // 셔플 폭 32 제한(RCA 2026-09-03): 상/하 절반 공유메모리 교환 후 워프 트리.
    __shared__ double sh32[32];
    double accd = (double)acc;
    if (l >= 32) sh32[l - 32] = accd;
    __syncthreads();
    if (l < 32) {
        accd += sh32[l];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            accd += __shfl_down_sync(0xffffffffffffffffull, accd, off);
        if (l == 0) out[o] = (float)accd;
    }
}

// q6_K (ty14) — 16원소 그룹, 가상워드 ql/qh (비정렬 5워드 슬라이딩)
extern "C" __global__ void gemm_q6k(const unsigned* xq, const unsigned* w,
                                    double* part, float* out, int n_in, int n_out, int xq_w) {
    int o = blockIdx.y + blockIdx.z * gridDim.y;  // 토큰=x축 — L2 행 재사용
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    xq += (int)blockIdx.x * xq_w;
    out += (int)blockIdx.x * n_out;
    int n_g = n_in >> 4;
    int cnt = (n_g + 63 - l) >> 6;
    int blocks = n_in >> 8;
    int row_base = o * blocks * 210;
    float acc = 0.0f;
    for (int m = 0; m < cnt; m++) {
        int g = l + (m << 6);
        int blk = g >> 4;
        int kloc = g - (blk << 4);
        int wb = row_base + blk * 210;
        int h = kloc >> 3;
        int src = (kloc - (h << 3)) >> 1;
        int p = kloc & 1;
        float d = f16w(w, wb + 208);
        int sc = sext8(byte(w, wb + 192 + kloc));
        int ql_rel = h * 64 + p * 16 + ((src & 1) << 5);
        int qh_rel = 128 + h * 32 + p * 16;
        bool al = (wb & 3) == 0;
        int qlw = (wb + ql_rel) >> 2;
        int qhw = (wb + qh_rel) >> 2;
        unsigned qa0, qa1, qa2, qa3, ha0, ha1, ha2, ha3;
        if (al) {
            qa0 = w[qlw]; qa1 = w[qlw+1]; qa2 = w[qlw+2]; qa3 = w[qlw+3];
            ha0 = w[qhw]; ha1 = w[qhw+1]; ha2 = w[qhw+2]; ha3 = w[qhw+3];
        } else {
            // 2-mod-4 정렬: 8바이트 경계 uint2 3개로 16B 추출 (워드당 2로드 → 3로드)
            const uint2* pq = (const uint2*)((const char*)w + (((wb + ql_rel) & ~7u)));
            unsigned qw[5];
            qw[0] = pq[0].x; qw[1] = pq[0].y; qw[2] = pq[1].x; qw[3] = pq[1].y; qw[4] = pq[2].x;
            qa0 = (qw[0] >> 16) | (qw[1] << 16);
            qa1 = (qw[1] >> 16) | (qw[2] << 16);
            qa2 = (qw[2] >> 16) | (qw[3] << 16);
            qa3 = (qw[3] >> 16) | (qw[4] << 16);
            const uint2* ph = (const uint2*)((const char*)w + (((wb + qh_rel) & ~7u)));
            unsigned hw[5];
            hw[0] = ph[0].x; hw[1] = ph[0].y; hw[2] = ph[1].x; hw[3] = ph[1].y; hw[4] = ph[2].x;
            ha0 = (hw[0] >> 16) | (hw[1] << 16);
            ha1 = (hw[1] >> 16) | (hw[2] << 16);
            ha2 = (hw[2] >> 16) | (hw[3] << 16);
            ha3 = (hw[3] >> 16) | (hw[4] << 16);
        }
        int xw = (g << 4) >> 2;
        unsigned y0 = xq[xw], y1 = xq[xw+1], y2 = xq[xw+2], y3 = xq[xw+3];
        // dot4 — 레인 간 빌림 방지 분해: w = nib + 16·hi2 − 32
        //   Σw·y = Σ(nib + 16·hi2)·y − 32·Σy  (레인 내 덧셈 ≤ 63 — 캐리 없음)
        int nsh = (src == 0 || src == 1) ? 0 : 4;
        int hsh = 2 * src;
        int isum = 0;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            unsigned qv = k==0?qa0:k==1?qa1:k==2?qa2:qa3;
            unsigned hv = k==0?ha0:k==1?ha1:k==2?ha2:ha3;
            unsigned yv = k==0?y0:k==1?y1:k==2?y2:y3;
            unsigned nibw = (qv >> nsh) & 0x0F0F0F0Fu;
            unsigned hi16w = (hv >> hsh) & 0x03030303u;
            isum = dot4(nibw + (hi16w << 4), yv, isum);
        }
        isum -= 32 * (int)xq[(n_in >> 2) + (n_in >> 5) + g]; // q16[g]=Σy, 체인은 32×Σy
        float yd = __uint_as_float(xq[(n_in >> 2) + (g >> 1)]);
        acc += yd * d * (float)sc * (float)isum;
    }
    // 트리 환원 (미러 tree64와 동일 순서) — part 왕복 제거.
    // 셔플 폭 32 제한(RCA 2026-09-03): 상/하 절반 공유메모리 교환 후 워프 트리.
    __shared__ double sh32[32];
    double accd = (double)acc;
    if (l >= 32) sh32[l - 32] = accd;
    __syncthreads();
    if (l < 32) {
        accd += sh32[l];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            accd += __shfl_down_sync(0xffffffffffffffffull, accd, off);
        if (l == 0) out[o] = (float)accd;
    }
}

// iq4_nl (ty20) — ktab2 룩업, 블록 32원소
extern "C" __global__ void gemm_nl(const unsigned* xq, const unsigned* w,
                                   double* part, const unsigned* ktab2, float* out, int n_in, int n_out, int xq_w) {
    __shared__ unsigned kt_s[256];
    for (int i = threadIdx.x; i < 256; i += 64) kt_s[i] = ktab2[i];
    __syncthreads();

    int o = blockIdx.y + blockIdx.z * gridDim.y;  // 토큰=x축 — L2 행 재사용
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    xq += (int)blockIdx.x * xq_w;
    out += (int)blockIdx.x * n_out;
    int n_sub = n_in >> 5;
    int cnt = (n_sub + 63 - l) >> 6;
    int row_base = o * n_sub * 18;
    float acc = 0.0f;
    for (int m = 0; m < cnt; m++) {
        int sb = l + (m << 6);
        int wb = row_base + sb * 18;
        float d = f16w(w, wb);
        int xw = (sb << 5) >> 2;
        unsigned y0 = xq[xw], y1 = xq[xw+1], y2 = xq[xw+2], y3 = xq[xw+3];
        unsigned y4 = xq[xw+4], y5 = xq[xw+5], y6 = xq[xw+6], y7 = xq[xw+7];
        // dot4 — kvalues 룩업 후 i8×4 패킹 (하프 워드 lo=elements 0..15, hi=16..31)
        bool al = (wb & 3) == 2;
        int wq2 = wb >> 2;
        int isum = 0;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            unsigned qvw; // 4 weight byte
            if (al) qvw = w[wq2 + 1 + k];
            else qvw = (w[wq2 + k] >> 16) | (w[wq2 + k + 1] << 16);
            unsigned lo = 0, hi = 0;
            #pragma unroll
            for (int b = 3; b >= 0; b--) {
                unsigned t = kt_s[(qvw >> (8 * b)) & 0xFFu];
                lo = (lo << 8) | (t & 0xFFu);
                hi = (hi << 8) | (t >> 8);
            }
            unsigned ylv = k==0?y0:k==1?y1:k==2?y2:y3;
            unsigned yhv = k==0?y4:k==1?y5:k==2?y6:y7;
            isum = dot4(lo, ylv, isum);
            isum = dot4(hi, yhv, isum);
        }
        float yd = __uint_as_float(xq[(n_in >> 2) + sb]);
        acc += yd * d * (float)isum;
    }
    // 트리 환원 (미러 tree64와 동일 순서) — part 왕복 제거.
    // 셔플 폭 32 제한(RCA 2026-09-03): 상/하 절반 공유메모리 교환 후 워프 트리.
    __shared__ double sh32[32];
    double accd = (double)acc;
    if (l >= 32) sh32[l - 32] = accd;
    __syncthreads();
    if (l < 32) {
        accd += sh32[l];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            accd += __shfl_down_sync(0xffffffffffffffffull, accd, off);
        if (l == 0) out[o] = (float)accd;
    }
}

__device__ __forceinline__ float f16w_ldg(const unsigned* w, int byte_off) {
    unsigned v = __ldg((const unsigned*)((const char*)w + byte_off));
    return __half2float(__ushort_as_half((unsigned short)(v & 0xFFFFu)));
}

// q3_K (ty11) — 16원소 하프블록, ql/hm byte() 로드(비정렬)
extern "C" __global__ void gemm_q3k(const unsigned* xq, const unsigned* w,
                                    double* part, float* out, int n_in, int n_out, int xq_w) {
    int o = blockIdx.y + blockIdx.z * gridDim.y;  // 토큰=x축 — L2 행 재사용
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    xq += (int)blockIdx.x * xq_w;
    out += (int)blockIdx.x * n_out;
    int n_h = n_in >> 4;
    int cnt = (n_h + 63 - l) >> 6;
    int blocks = n_in >> 8;
    int row_base = o * blocks * 110;
    float acc = 0.0f;
    // 스레드 불변량 호이스트 — h = l + 64m이므로 h%16 == l%16 (매 m 동일).
    // 구버전은 이 전체를 매 반복 재산출해 스케일 선택사슬이 병목(107GB/s).
    const int kloc = l & 15;
    const int hh = kloc >> 3;
    const int src = (kloc - (hh << 3)) >> 1;
    const int p = kloc & 1;
    const int ai = hh * 8 + src * 2 + p;
    const int aish = (ai & 3) * 8;
    const int qsh = src * 2;
    // (wb&3)은 blk%4 고정(110≡2 mod4, blk stride 4) — 워드정렬 여부 불변
    const int rb2 = (row_base + 2 * ((l >> 4) & 1)) & 3;   // row_base&3 + blk&1贡献
    const bool alq = (rb2 & 3) == 0;   // qlb2 = wb+32+hh*32+p*16 ≡ wb (mod 4)
    for (int m = 0; m < cnt; m++) {
        int h = l + (m << 6);
        // ILP-2: 다음 m의 w 로드를 현재 계산과 병렬로 발사 (지연 은닉)
        unsigned pf_qs[4]; unsigned pf_hs[4]; unsigned pf_a0, pf_a1, pf_tmp;
        int pf_ok = (m + 1 < cnt);
        int h2 = l + ((m + 1) << 6);
        int blk2 = h2 >> 4;
        int wb2 = row_base + blk2 * 110;
        int sw2 = (wb2 + 96) >> 2;
        int qlb2b = wb2 + 32 + hh * 32 + p * 16;
        int hbase2b = wb2 + p * 16;
        int qlw2b = qlb2b >> 2;
        int qhw2b = hbase2b >> 2;
        if (pf_ok) {
            pf_a0 = __ldg(&w[sw2]); pf_a1 = __ldg(&w[sw2+1]); pf_tmp = __ldg(&w[sw2+2]);
            #pragma unroll
            for (int k = 0; k < 4; k++) { pf_qs[k] = __ldg(&w[qlw2b + k]); pf_hs[k] = __ldg(&w[qhw2b + k]); }
        }
        int blk = h >> 4;
        int wb = row_base + blk * 110;
        // 스케일 12바이트(wb+96..107) — 필요한 aux 하나만 산출 (ai 고정)
        int sw = (wb + 96) >> 2;
        int off = rb2;   // (wb+96)&3 == wb&3 (96≡0 mod 4) — 스레드 불변
        unsigned a0, a1, tmp;
        if (off == 0) { a0 = __ldg(&w[sw]); a1 = __ldg(&w[sw+1]); tmp = __ldg(&w[sw+2]); }
        else { a0 = (__ldg(&w[sw]) >> 16) | (__ldg(&w[sw+1]) << 16); a1 = (__ldg(&w[sw+1]) >> 16) | (__ldg(&w[sw+2]) << 16); tmp = (__ldg(&w[sw+2]) >> 16) | (__ldg(&w[sw+3]) << 16); }
        unsigned k1 = 0x03030303u, k2 = 0x0f0f0f0fu;
        unsigned aux;
        if (ai < 4) aux = (a0 & k2) | ((tmp & k1) << 4);
        else if (ai < 8) aux = (a1 & k2) | (((tmp >> 2) & k1) << 4);
        else if (ai < 12) aux = ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4);
        else aux = ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4);
        unsigned scb = (aux >> aish) & 0xFFu;
        float dl = f16w_ldg(w, wb + 108) * (float)(sext8(scb) - 32);
        int qlb2 = wb + 32 + hh * 32 + p * 16;
        int hbase2 = wb + p * 16;
        int xw = (h << 4) >> 2;
        unsigned y0 = xq[xw], y1 = xq[xw+1], y2 = xq[xw+2], y3 = xq[xw+3];
        // dot4 — Σ(qv − 4 + 4bit)·y = Σ qv·y − 4Σy + 4Σ bit·y
        // dot4 — Σ(qv − 4 + 4bit)·y = Σ qv·y − 4Σy + 4Σ bit·y
        // 스팬 프리로드: k-루프가 w[qlw2+k]/w[qlw2+k+1]을 중복로드(9회) —
        // 스팬 5워드 1회 로드 후 인접 페어funnel로 4워드 구성 (미정렬시).
        int qlw2 = qlb2 >> 2;
        int qhw2 = hbase2 >> 2;
        // dot4 word-SIMD (gemm_q5k 패턴 이식, 2026-09-05): 니블(0..3)과
        // 하이비트(+4) 레인을 한 워드에 융합 → dot4 1회/워드. −4·Σy 보정은
        // xq q16 테이블(Σ y바이트 하/상 16). 기존 3-dot4 대비 ALU 1/3.
        unsigned qs[4], hs[4];
        if (alq) {
            #pragma unroll
            for (int k = 0; k < 4; k++) { qs[k] = __ldg(&w[qlw2 + k]); hs[k] = __ldg(&w[qhw2 + k]); }
        } else {
            unsigned s0 = w[qlw2], s1 = __ldg(&w[qlw2+1]), s2 = __ldg(&w[qlw2+2]), s3 = __ldg(&w[qlw2+3]), s4 = __ldg(&w[qlw2+4]);
            qs[0] = (s0 >> 16) | (s1 << 16); qs[1] = (s1 >> 16) | (s2 << 16);
            qs[2] = (s2 >> 16) | (s3 << 16); qs[3] = (s3 >> 16) | (s4 << 16);
            unsigned t0 = w[qhw2], t1 = __ldg(&w[qhw2+1]), t2 = __ldg(&w[qhw2+2]), t3 = __ldg(&w[qhw2+3]), t4 = __ldg(&w[qhw2+4]);
            hs[0] = (t0 >> 16) | (t1 << 16); hs[1] = (t1 >> 16) | (t2 << 16);
            hs[2] = (t2 >> 16) | (t3 << 16); hs[3] = (t3 >> 16) | (t4 << 16);
        }
        unsigned hbit = 1u << src;
        int isum = 0;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            unsigned yv = k==0?y0:k==1?y1:k==2?y2:y3;
            unsigned nibw = (qs[k] >> qsh) & 0x03030303u;
            unsigned bitw = ((hs[k] & (hbit * 0x01010101u)) >> src) << 2;  // +4 레인
            isum = dot4(nibw | bitw, yv, isum);
        }
        int qsb = (n_in >> 2) + (n_in >> 5);
        int qsum = (int)xq[qsb + h];  // h=16값 idx → 32블록 h>>1의 테이블 [2·(h>>1)] = h
        float yd = __uint_as_float(xq[(n_in >> 2) + (h >> 1)]);
        acc += yd * dl * ((float)isum - 4.0f * (float)qsum);
    }
    // 트리 환원 (미러 tree64와 동일 순서) — part 왕복 제거.
    // 셔플 폭 32 제한(RCA 2026-09-03): 상/하 절반 공유메모리 교환 후 워프 트리.
    __shared__ double sh32[32];
    double accd = (double)acc;
    if (l >= 32) sh32[l - 32] = accd;
    __syncthreads();
    if (l < 32) {
        accd += sh32[l];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            accd += __shfl_down_sync(0xffffffffffffffffull, accd, off);
        if (l == 0) out[o] = (float)accd;
    }
}


// iq3_s (ty21) — 32원소 서브블록, grid4×부호 정수 가중, f64 레인 누산
__device__ const unsigned IQ3S_GRID[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};

extern "C" __global__ void gemm_iq3s(const unsigned* xq, const unsigned* w,
                                     double* part, float* out, int n_in, int n_out, int xq_w) {
    int o = blockIdx.y + blockIdx.z * gridDim.y;  // 토큰=x축 — L2 행 재사용
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    xq += (int)blockIdx.x * xq_w;
    out += (int)blockIdx.x * n_out;
    int n_sub = n_in >> 5;
    int cnt = (n_sub + 63 - l) >> 6;
    int blocks = n_in >> 8;
    int row_base = o * blocks * 110;
    float acc = 0.0f;
    for (int m = 0; m < cnt; m++) {
        int sub = l + (m << 6);
        int blk = sub >> 3;
        int h = sub & 7;
        int wb = row_base + blk * 110;
        float d = f16w(w, wb);
        unsigned scb = byte(w, wb + 106 + (h >> 1));
        int nib = (h & 1) ? (int)(scb >> 4) : (int)(scb & 0xFu);
        float db = d * (float)(1 + 2 * nib);
        unsigned qhb = byte(w, wb + 66 + h);
        int qs_base = wb + 2 + h * 8;
        int sg_base = wb + 74 + h * 4;
        long isum = 0;
        #pragma unroll
        for (int ll = 0; ll < 4; ll++) {
            unsigned idx1 = byte(w, qs_base + 2*ll) | ((qhb << (8 - 2*ll)) & 256u);
            unsigned idx2 = byte(w, qs_base + 2*ll + 1) | ((qhb << (7 - 2*ll)) & 256u);
            unsigned g1 = IQ3S_GRID[idx1];
            unsigned g2 = IQ3S_GRID[idx2];
            unsigned sgb = byte(w, sg_base + ll);
            int e0 = 8 * ll;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int w1 = sext8((g1 >> (8*j)) & 0xFFu) * ((sgb & (1u << j)) ? -1 : 1);
                int w2 = sext8((g2 >> (8*j)) & 0xFFu) * ((sgb & (1u << (4+j))) ? -1 : 1);
                int e1 = e0 + j, e2 = e0 + 4 + j;
                int y1 = sext8(byte(xq, ((sub << 5) + e1)));
                int y2 = sext8(byte(xq, ((sub << 5) + e2)));
                isum += (long)w1 * y1 + (long)w2 * y2;
            }
        }
        float yd = __uint_as_float(xq[(n_in >> 2) + sub]);
        acc += yd * db * (float)isum;
    }
    // 트리 환원 (미러 tree64와 동일 순서) — part 왕복 제거.
    // 셔플 폭 32 제한(RCA 2026-09-03): 상/하 절반 공유메모리 교환 후 워프 트리.
    __shared__ double sh32[32];
    double accd = (double)acc;
    if (l >= 32) sh32[l - 32] = accd;
    __syncthreads();
    if (l < 32) {
        accd += sh32[l];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            accd += __shfl_down_sync(0xffffffffffffffffull, accd, off);
        if (l == 0) out[o] = (float)accd;
    }
}



