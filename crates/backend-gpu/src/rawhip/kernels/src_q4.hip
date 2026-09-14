// ─── qwen4exp 값 경로 커널 (q4acc) — 2026-09-12 ───
// 계약: core 미러(quant.rs dot_row_w4a8_* / ops.rs)와 동일 산술 계열.
// 정수 내적은 정확, 그룹 스케일 적용·f64 부분합 순서만 CPU와 다르다
// (문서화된 ≤2.4e-7 계열). 활성 xq = [n/4 워드][n/32 f32 스케일].

// q5_1 (블록 32 = d(2) m(2) qh(4) qs(16), 24B). value = d·q + m (q 무부호 5bit).
// q = lo4 + 16·hi1 로 분해 → dot4 2회 + Σxq 1회. 그룹 스케일 d/m 은 f16.
extern "C" __global__ void q4_gemm_q5_1(const unsigned* xq, const unsigned* w,
                                        double* part, float* out, int n_in, int n_out, int xq_w) {
    int o = blockIdx.y + blockIdx.z * gridDim.y;
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    xq += (int)blockIdx.x * xq_w;
    out += (int)blockIdx.x * n_out;
    int n_sub = n_in >> 5;
    int cnt = (n_sub + 63 - l) >> 6;
    int row_base = o * n_sub * 24;
    const unsigned ones = 0x01010101u;
    float acc = 0.0f;
    for (int m = 0; m < cnt; m++) {
        int sb = l + (m << 6);
        int wb = row_base + sb * 24;
        float d = f16w(w, wb);
        float mn = f16w(w, wb + 2);
        unsigned qh = w[(wb + 4) >> 2];
        unsigned qsw0 = w[(wb + 8) >> 2];
        unsigned qsw1 = w[(wb + 12) >> 2];
        unsigned qsw2 = w[(wb + 16) >> 2];
        unsigned qsw3 = w[(wb + 20) >> 2];
        int xw = (sb << 5) >> 2;
        int s_lo = 0, s_hi = 0, s1 = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            // 요소 4k..4k+3 — xq 워드 k, qs 워드 k&3 (k<4 저니블, k>=4 고니블)
            unsigned qs4 = (k & 3) == 0 ? qsw0 : (k & 3) == 1 ? qsw1 : (k & 3) == 2 ? qsw2 : qsw3;
            unsigned nib = (k < 4) ? (qs4 & 0x0F0F0F0Fu) : ((qs4 >> 4) & 0x0F0F0F0Fu);
            unsigned hb = (qh >> (4 * k)) & 0xFu;
            // 요소 i 의 고비트 → 자기 바이트의 bit4 (값 16 — 별도 <<4 불필요)
            unsigned hb_p = (hb & 1u) << 4 | (hb & 2u) << 11 | (hb & 4u) << 18 | (hb & 8u) << 25;
            unsigned y = xq[xw + k];
            s_lo += dot4(nib, y, 0);
            s_hi += dot4(hb_p, y, 0);   // hb_p 바이트 = 16 → 여기서 이미 ×16
            s1 += dot4(ones, y, 0);
        }
        float yd = __uint_as_float(xq[(n_in >> 2) + sb]);
        int isum = s_lo + s_hi;
        acc += yd * (d * (float)isum + mn * (float)s1);
    }
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

// q4_K 행-배치 타일 — 스레드 = (출력 1개 × 행 R개). 가중치 디퀀트를 **행 루프
// 밖으로** 빼서 한 번만 하고 R개 행이 재사용한다. 기존 판은 (출력,행) 쌍마다
// 디퀀트를 반복해 비용이 행 수에 비례했다(q4k-bench 실측 ~5 Tops로 ALU 상한).
// x는 같은 행을 256스레드가 읽어 브로드캐스트되고, 누산 순서는 _m과 동일.
extern "C" __global__ void q4_gemm_q4k_y(const unsigned* xq, const unsigned* w, double* part,
                                         float* out, int n_in, int n_out, int xq_w, int t, int rows_per_thread) {
    const int o = (int)blockIdx.x * blockDim.x + threadIdx.x;
    const int r0 = (int)blockIdx.y * rows_per_thread;
    if (o >= n_out) return;
    const int r1 = min(t, r0 + rows_per_thread);
    const int n_super = n_in >> 8;
    const unsigned ones = 0x01010101u;
    const unsigned char* wr = (const unsigned char*)w + (size_t)o * (n_super * 144);
    const unsigned* xrow_base = xq;
    const unsigned xw_base = (unsigned)(n_in >> 2);
    for (int sIdx = 0; sIdx < n_super; sIdx++) {
        const unsigned char* blk = wr + sIdx * 144;
        float d = bits_f16((unsigned)blk[0] | ((unsigned)blk[1] << 8));
        float dmin = bits_f16((unsigned)blk[2] | ((unsigned)blk[3] << 8));
        const unsigned char* sc = blk + 4;
        const unsigned char* qs = blk + 16;
        #pragma unroll
        for (int it = 0; it < 4; it++) {
            unsigned q[8];
            // 16B 벡터 로드 — 4개 워드를 한 트랜잭션으로. 블록(144B)·it*32는
            // 16의 배수라 정렬이 보장되고, 리틀엔디언 워드 4개는 기존 바이트
            // 조립과 **값이 동일**하다(수치 불변). 스칼라 32회 → 벡터 2회.
            const uint4* p4 = (const uint4*)(qs + it * 32);
            #pragma unroll
            for (int m4 = 0; m4 < 2; m4++) {
                uint4 t4 = p4[m4];
                q[m4 * 4 + 0] = t4.x;
                q[m4 * 4 + 1] = t4.y;
                q[m4 * 4 + 2] = t4.z;
                q[m4 * 4 + 3] = t4.w;
            }
            const int j1 = 2 * it;
            const int j2 = 2 * it + 1;
            unsigned s1, m1, s2, m2;
            if (j1 < 4) { s1 = sc[j1] & 63u; m1 = sc[j1 + 4] & 63u; }
            else { s1 = (sc[j1 + 4] & 0xFu) | ((unsigned)(sc[j1 - 4] >> 6) << 4); m1 = (sc[j1 + 4] >> 4) | ((unsigned)(sc[j1] >> 6) << 4); }
            if (j2 < 4) { s2 = sc[j2] & 63u; m2 = sc[j2 + 4] & 63u; }
            else { s2 = (sc[j2 + 4] & 0xFu) | ((unsigned)(sc[j2 - 4] >> 6) << 4); m2 = (sc[j2 + 4] >> 4) | ((unsigned)(sc[j2] >> 6) << 4); }
            const float f1 = d * (float)s1;
            const float f2 = d * (float)s2;
            const float g1 = dmin * (float)m1;
            const float g2 = dmin * (float)m2;
            for (int r = r0; r < r1; r++) {
                const unsigned* xr = xrow_base + (size_t)r * xq_w;
                const unsigned* x1 = xr + sIdx * 64 + it * 16;
                const unsigned* x2 = x1 + 8;
                int il1 = 0, il2 = 0, qs1 = 0, qs2 = 0;
                #pragma unroll
                for (int m = 0; m < 8; m++) {
                    unsigned y1 = x1[m];
                    il1 += dot4(q[m] & 0x0F0F0F0Fu, y1, 0);
                    qs1 += dot4(ones, y1, 0);
                    unsigned y2 = x2[m];
                    il2 += dot4((q[m] >> 4) & 0x0F0F0F0Fu, y2, 0);
                    qs2 += dot4(ones, y2, 0);
                }
                float yd1 = __uint_as_float(xr[xw_base + sIdx * 8 + 2 * it]);
                float yd2 = __uint_as_float(xr[xw_base + sIdx * 8 + 2 * it + 1]);
                out[(size_t)r * n_out + o] += yd1 * (f1 * (float)il1 - g1 * (float)qs1)
                                            + yd2 * (f2 * (float)il2 - g2 * (float)qs2);
            }
        }
    }
}

// q4_K x-스테이징 타일 — 블록 = 16출력 × 16행(256스레드). 슈퍼블록마다 x의
// 16행(64워드 + 8스케일 = 72워드)을 shared로 1회 적재하고, 누산은 검증된
// q4_gemm_q4k_m과 **동일 로직·순서**로 수행한다(결과 동일). 기존 판은 출력마다
// x 행 전체를 다시 읽어 비용이 행 수에 비례했다(q4k-bench: t=512가 128의 3배,
// 실효 4GB/s).
extern "C" __global__ void q4_gemm_q4k_x(const unsigned* xq, const unsigned* w, double* part,
                                         float* out, int n_in, int n_out, int xq_w, int t) {
    __shared__ unsigned xsh[16 * 72];
    const int ox = threadIdx.x & 15;
    const int rx = threadIdx.x >> 4;
    const int o0 = (int)blockIdx.x * 16;
    const int r0 = (int)blockIdx.y * 16;
    const int o = o0 + ox;
    const int r = r0 + rx;
    const int n_super = n_in >> 8;
    const unsigned ones = 0x01010101u;
    float acc = 0.0f;
    for (int sIdx = 0; sIdx < n_super; sIdx++) {
        // ── x 16행 스테이징 (행당 64 워드 + 8 스케일)
        for (int i = threadIdx.x; i < 16 * 72; i += 256) {
            int rr = i / 72, ii = i - rr * 72;
            int row = r0 + rr;
            unsigned v = 0u;
            if (row < t) {
                const unsigned* xr = xq + (size_t)row * xq_w;
                if (ii < 64) {
                    v = xr[sIdx * 64 + ii];
                } else {
                    v = xr[(n_in >> 2) + sIdx * 8 + (ii - 64)];
                }
            }
            xsh[i] = v;
        }
        __syncthreads();
        if (o < n_out && r < t) {
            const unsigned char* wr = (const unsigned char*)w + (size_t)o * (n_super * 144);
            const unsigned char* blk = wr + sIdx * 144;
            float d = bits_f16((unsigned)blk[0] | ((unsigned)blk[1] << 8));
            float dmin = bits_f16((unsigned)blk[2] | ((unsigned)blk[3] << 8));
            const unsigned char* sc = blk + 4;
            const unsigned char* qs = blk + 16;
            const unsigned* xrow = xsh + rx * 72;
            #pragma unroll
            for (int it = 0; it < 4; it++) {
                unsigned q[8];
                #pragma unroll
                for (int m = 0; m < 8; m++) {
                    const unsigned char* p = qs + it * 32 + m * 4;
                    q[m] = (unsigned)p[0] | ((unsigned)p[1] << 8) | ((unsigned)p[2] << 16) | ((unsigned)p[3] << 24);
                }
                const int j1 = 2 * it;
                const int j2 = 2 * it + 1;
                unsigned s1, m1, s2, m2;
                if (j1 < 4) { s1 = sc[j1] & 63u; m1 = sc[j1 + 4] & 63u; }
                else { s1 = (sc[j1 + 4] & 0xFu) | ((unsigned)(sc[j1 - 4] >> 6) << 4); m1 = (sc[j1 + 4] >> 4) | ((unsigned)(sc[j1] >> 6) << 4); }
                if (j2 < 4) { s2 = sc[j2] & 63u; m2 = sc[j2 + 4] & 63u; }
                else { s2 = (sc[j2 + 4] & 0xFu) | ((unsigned)(sc[j2 - 4] >> 6) << 4); m2 = (sc[j2 + 4] >> 4) | ((unsigned)(sc[j2] >> 6) << 4); }
                const unsigned* x1 = xrow + it * 16;
                const unsigned* x2 = x1 + 8;
                int il1 = 0, il2 = 0, qs1 = 0, qs2 = 0;
                #pragma unroll
                for (int m = 0; m < 8; m++) {
                    unsigned y1 = x1[m];
                    il1 += dot4(q[m] & 0x0F0F0F0Fu, y1, 0);
                    qs1 += dot4(ones, y1, 0);
                    unsigned y2 = x2[m];
                    il2 += dot4((q[m] >> 4) & 0x0F0F0F0Fu, y2, 0);
                    qs2 += dot4(ones, y2, 0);
                }
                float yd1 = __uint_as_float(xrow[64 + 2 * it]);
                float yd2 = __uint_as_float(xrow[64 + 2 * it + 1]);
                acc += yd1 * (d * (float)s1 * (float)il1 - dmin * (float)m1 * (float)qs1);
                acc += yd2 * (d * (float)s2 * (float)il2 - dmin * (float)m2 * (float)qs2);
            }
        }
        __syncthreads();
    }
    if (o < n_out && r < t) out[(size_t)r * n_out + o] = acc;
}

// q4_K 가변 타일 — outs/rows를 런타임 인자로 받아 형상 적응 스윕을 가능케 한다.
// 스레드 = (출력, 행) 한 쌍(블록 스레드 = outs*rows). plans/65 형상 스윕용.
extern "C" __global__ void q4_gemm_q4k_g(const unsigned* xq, const unsigned* w, double* part,
                                         float* out, int n_in, int n_out, int xq_w, int t,
                                         int outs, int rows) {
    int ox = threadIdx.x % outs;
    int rx = threadIdx.x / outs;
    int o = (int)blockIdx.x * outs + ox;
    int r = (int)blockIdx.y * rows + rx;
    if (o >= n_out || r >= t) return;
    int n_super = n_in >> 8;
    const unsigned char* wr = (const unsigned char*)w + (size_t)o * (n_super * 144);
    const unsigned* xr = xq + (size_t)r * xq_w;
    const unsigned xw_base = (unsigned)(n_in >> 2);
    const unsigned ones = 0x01010101u;
    float acc = 0.0f;
    for (int sIdx = 0; sIdx < n_super; sIdx++) {
        const unsigned char* blk = wr + sIdx * 144;
        float d = bits_f16((unsigned)blk[0] | ((unsigned)blk[1] << 8));
        float dmin = bits_f16((unsigned)blk[2] | ((unsigned)blk[3] << 8));
        const unsigned char* sc = blk + 4;
        const unsigned char* qs = blk + 16;
        #pragma unroll
        for (int it = 0; it < 4; it++) {
            unsigned q[8];
            // 16B 벡터 로드 — 4개 워드를 한 트랜잭션으로. 블록(144B)·it*32는
            // 16의 배수라 정렬이 보장되고, 리틀엔디언 워드 4개는 기존 바이트
            // 조립과 **값이 동일**하다(수치 불변). 스칼라 32회 → 벡터 2회.
            const uint4* p4 = (const uint4*)(qs + it * 32);
            #pragma unroll
            for (int m4 = 0; m4 < 2; m4++) {
                uint4 t4 = p4[m4];
                q[m4 * 4 + 0] = t4.x;
                q[m4 * 4 + 1] = t4.y;
                q[m4 * 4 + 2] = t4.z;
                q[m4 * 4 + 3] = t4.w;
            }
            const int j1 = 2 * it;
            const int j2 = 2 * it + 1;
            unsigned s1, m1, s2, m2;
            if (j1 < 4) { s1 = sc[j1] & 63u; m1 = sc[j1 + 4] & 63u; }
            else { s1 = (sc[j1 + 4] & 0xFu) | ((unsigned)(sc[j1 - 4] >> 6) << 4); m1 = (sc[j1 + 4] >> 4) | ((unsigned)(sc[j1] >> 6) << 4); }
            if (j2 < 4) { s2 = sc[j2] & 63u; m2 = sc[j2 + 4] & 63u; }
            else { s2 = (sc[j2 + 4] & 0xFu) | ((unsigned)(sc[j2 - 4] >> 6) << 4); m2 = (sc[j2 + 4] >> 4) | ((unsigned)(sc[j2] >> 6) << 4); }
            const unsigned* x1 = xr + sIdx * 64 + it * 16;
            const unsigned* x2 = x1 + 8;
            int il1 = 0, il2 = 0, qs1 = 0, qs2 = 0;
            #pragma unroll
            for (int m = 0; m < 8; m++) {
                unsigned y1 = x1[m];
                il1 += dot4(q[m] & 0x0F0F0F0Fu, y1, 0);
                qs1 += dot4(ones, y1, 0);
                unsigned y2 = x2[m];
                il2 += dot4((q[m] >> 4) & 0x0F0F0F0Fu, y2, 0);
                qs2 += dot4(ones, y2, 0);
            }
            float yd1 = __uint_as_float(xr[xw_base + sIdx * 8 + 2 * it]);
            float yd2 = __uint_as_float(xr[xw_base + sIdx * 8 + 2 * it + 1]);
            acc += yd1 * (d * (float)s1 * (float)il1 - dmin * (float)m1 * (float)qs1);
            acc += yd2 * (d * (float)s2 * (float)il2 - dmin * (float)m2 * (float)qs2);
        }
    }
    out[(size_t)r * n_out + o] = acc;
}

// q5_1 16행×4출력 타일 — 기존 q4_gemm_q5_1은 (행, 출력) 블록이라 같은 가중치 행을
// 행 수만큼 다시 읽었다(MoE expert-down = 25.2GiB q5_1: t=20 그룹에서 20배
// 증폭, 실측 2715ms/청크 = 전체의 54%). 이 판은 출력 1개당 16행을 묶어 가중치
// 1회 독서로 상각한다. 행별 누산 순서는 원판과 동일(비트 동일 — q4-acc-check).
extern "C" __global__ void q4_gemm_q5_1_t(const unsigned* xq, const unsigned* w,
                                          double* part, float* out, int n_in, int n_out,
                                          int xq_w, int t) {
    // 256 스레드 = 4출력 × 64레인 — 블록당 연산을 4배로(블록 수·환원 오버헤드 1/4).
    int sub = threadIdx.x >> 6;
    int l = threadIdx.x & 63;
    int o = ((int)blockIdx.y + (int)blockIdx.z * gridDim.y) * 4 + sub;
    if (o >= n_out) return;
    int r0 = (int)blockIdx.x * 16;
    int rc = min(16, t - r0);
    if (rc <= 0) return;
    int n_sub = n_in >> 5;
    int cnt = (n_sub + 63 - l) >> 6;
    int row_base = o * n_sub * 24;
    const unsigned ones = 0x01010101u;
    float acc[16];
    #pragma unroll
    for (int r = 0; r < 16; r++) acc[r] = 0.0f;
    for (int m = 0; m < cnt; m++) {
        int sb = l + (m << 6);
        int wb = row_base + sb * 24;
        float d = f16w(w, wb);
        float mn = f16w(w, wb + 2);
        unsigned qh = w[(wb + 4) >> 2];
        unsigned qsw0 = w[(wb + 8) >> 2];
        unsigned qsw1 = w[(wb + 12) >> 2];
        unsigned qsw2 = w[(wb + 16) >> 2];
        unsigned qsw3 = w[(wb + 20) >> 2];
        int xw = (sb << 5) >> 2;
        for (int r = 0; r < rc; r++) {
            const unsigned* xr = xq + (size_t)(r0 + r) * xq_w;
            int s_lo = 0, s_hi = 0, s1 = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                unsigned qs4 = (k & 3) == 0 ? qsw0 : (k & 3) == 1 ? qsw1 : (k & 3) == 2 ? qsw2 : qsw3;
                unsigned nib = (k < 4) ? (qs4 & 0x0F0F0F0Fu) : ((qs4 >> 4) & 0x0F0F0F0Fu);
                unsigned hb = (qh >> (4 * k)) & 0xFu;
                unsigned hb_p = (hb & 1u) << 4 | (hb & 2u) << 11 | (hb & 4u) << 18 | (hb & 8u) << 25;
                unsigned y = xr[xw + k];
                s_lo += dot4(nib, y, 0);
                s_hi += dot4(hb_p, y, 0);
                s1 += dot4(ones, y, 0);
            }
            float yd = __uint_as_float(xr[(n_in >> 2) + sb]);
            acc[r] += yd * (d * (float)(s_lo + s_hi) + mn * (float)s1);
        }
    }
    __shared__ double sh32[4 * 16 * 32];
    for (int r = 0; r < rc; r++) {
        if (l >= 32) sh32[(sub * 16 + r) * 32 + (l - 32)] = (double)acc[r];
    }
    __syncthreads();
    if (l < 32) {
        for (int r = 0; r < rc; r++) {
            double a = (double)acc[r] + sh32[(sub * 16 + r) * 32 + l];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                a += __shfl_down_sync(0xffffffffffffffffull, a, off);
            if (l == 0) out[(size_t)(r0 + r) * n_out + o] = (float)a;
        }
    }
}

// q5_1 MMQ급 타일 — 16출력 × 16행/블록(256스레드). 스레드당 (출력,행) 한 쌍을
// k 직렬 누산하고 가중치 행은 shared로 1회 적재해 16행이 공유한다. 스레드당
// 읽는 바이트가 원판(~1B)의 수백 배가 되어 총 지연에서 벗어난다.
// 누산 순서가 원판(레인=서브블록 + 셔플 트리)과 다르므로 **비트 동일이 아니다**
// — llama.cpp/vLLM과 같은 허용 오차 계약(q4-acc-check max_rel)으로 검증한다.
extern "C" __global__ void q4_gemm_q5_1_m(const unsigned* xq, const unsigned* w, double* part,
                                          float* out, int n_in, int n_out, int xq_w, int t) {
    extern __shared__ unsigned wsh[];
    int n_sub = n_in >> 5;
    int wrow = n_sub * 6;                 // 워드/출력행
    int o0 = (int)blockIdx.x * 16;
    int r0 = (int)blockIdx.y * 16;
    // 가중치 cooperative 적재 (16행 × wrow 워드) — 연속 워드라 코얼레스드.
    for (int i = threadIdx.x; i < 16 * wrow; i += 256) {
        int oo = i / wrow;
        int ww = i - oo * wrow;
        if (o0 + oo < n_out) wsh[i] = w[(size_t)(o0 + oo) * wrow + ww];
    }
    __syncthreads();
    int ox = threadIdx.x & 15;
    int rx = threadIdx.x >> 4;
    int o = o0 + ox;
    int r = r0 + rx;
    if (o >= n_out || r >= t) return;
    const unsigned* wr = wsh + ox * wrow;
    const unsigned* xr = xq + (size_t)r * xq_w;
    const unsigned ones = 0x01010101u;
    float acc = 0.0f;
    for (int sb = 0; sb < n_sub; sb++) {
        int wb = sb * 6;
        float d = bits_f16(wr[wb] & 0xFFFFu);
        float mn = bits_f16((wr[wb] >> 16) & 0xFFFFu);
        unsigned qh = wr[wb + 1];
        unsigned qsw0 = wr[wb + 2];
        unsigned qsw1 = wr[wb + 3];
        unsigned qsw2 = wr[wb + 4];
        unsigned qsw3 = wr[wb + 5];
        int xw = (sb << 5) >> 2;
        int s_lo = 0, s_hi = 0, s1 = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            unsigned qs4 = (k & 3) == 0 ? qsw0 : (k & 3) == 1 ? qsw1 : (k & 3) == 2 ? qsw2 : qsw3;
            unsigned nib = (k < 4) ? (qs4 & 0x0F0F0F0Fu) : ((qs4 >> 4) & 0x0F0F0F0Fu);
            unsigned hb = (qh >> (4 * k)) & 0xFu;
            unsigned hb_p = (hb & 1u) << 4 | (hb & 2u) << 11 | (hb & 4u) << 18 | (hb & 8u) << 25;
            unsigned y = xr[xw + k];
            s_lo += dot4(nib, y, 0);
            s_hi += dot4(hb_p, y, 0);
            s1 += dot4(ones, y, 0);
        }
        float yd = __uint_as_float(xr[(n_in >> 2) + sb]);
        acc += yd * (d * (float)(s_lo + s_hi) + mn * (float)s1);
    }
    out[(size_t)r * n_out + o] = acc;
}

// q5_1 그룹판 — `_m`과 동일 산술·협조 적재 + 타일별 전문가(16배수 패딩 전제).
extern "C" __global__ void q4_gemm_q5_1_gm(const unsigned* xq, const unsigned* w, double* part,
                                          float* out, const unsigned* tile_exp,
                                          int n_in, int n_out, int xq_w, int t, int expert_words) {
    extern __shared__ unsigned wsh[];
    int n_sub = n_in >> 5;
    int wrow = n_sub * 6;                 // 워드/출력행
    int o0 = (int)blockIdx.x * 16;
    int r0 = (int)blockIdx.y * 16;
    // 가중치 cooperative 적재 (16행 × wrow 워드) — 연속 워드라 코얼레스드.
    for (int i = threadIdx.x; i < 16 * wrow; i += 256) {
        int oo = i / wrow;
        int ww = i - oo * wrow;
        if (o0 + oo < n_out)
            wsh[i] = w[(size_t)tile_exp[blockIdx.y] * (size_t)expert_words + (size_t)(o0 + oo) * wrow + ww];
    }
    __syncthreads();
    int ox = threadIdx.x & 15;
    int rx = threadIdx.x >> 4;
    int o = o0 + ox;
    int r = r0 + rx;
    if (o >= n_out || r >= t) return;
    const unsigned* wr = wsh + ox * wrow;
    const unsigned* xr = xq + (size_t)r * xq_w;
    const unsigned ones = 0x01010101u;
    float acc = 0.0f;
    for (int sb = 0; sb < n_sub; sb++) {
        int wb = sb * 6;
        float d = bits_f16(wr[wb] & 0xFFFFu);
        float mn = bits_f16((wr[wb] >> 16) & 0xFFFFu);
        unsigned qh = wr[wb + 1];
        unsigned qsw0 = wr[wb + 2];
        unsigned qsw1 = wr[wb + 3];
        unsigned qsw2 = wr[wb + 4];
        unsigned qsw3 = wr[wb + 5];
        int xw = (sb << 5) >> 2;
        int s_lo = 0, s_hi = 0, s1 = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            unsigned qs4 = (k & 3) == 0 ? qsw0 : (k & 3) == 1 ? qsw1 : (k & 3) == 2 ? qsw2 : qsw3;
            unsigned nib = (k < 4) ? (qs4 & 0x0F0F0F0Fu) : ((qs4 >> 4) & 0x0F0F0F0Fu);
            unsigned hb = (qh >> (4 * k)) & 0xFu;
            unsigned hb_p = (hb & 1u) << 4 | (hb & 2u) << 11 | (hb & 4u) << 18 | (hb & 8u) << 25;
            unsigned y = xr[xw + k];
            s_lo += dot4(nib, y, 0);
            s_hi += dot4(hb_p, y, 0);
            s1 += dot4(ones, y, 0);
        }
        float yd = __uint_as_float(xr[(n_in >> 2) + sb]);
        acc += yd * (d * (float)(s_lo + s_hi) + mn * (float)s1);
    }
    out[(size_t)r * n_out + o] = acc;
}

// q4_gemm_q5_1_gm_ids — t=1 전용 direct-ids판. down 투영이 Q5_1이라 그룹화를
// 여기서도 없애야 한다(게이트/up만 direct로는 down에서 그룹화가 1회 발생).
// 행마다 전문가가 다르므로 타일 공유 적재 대신 스레드가 자기 행의 전문가 행을
// 직접 읽는다. 내적 루프는 gm과 동일 → 비트 동일 기대.
extern "C" __global__ void q4_gemm_q5_1_gm_ids(const unsigned* xq, const unsigned* w, double* part,
                                          float* out, const unsigned* ids,
                                          int n_in, int n_out, int xq_w, int t, int expert_words) {
    int n_sub = n_in >> 5;
    int wrow = n_sub * 6;                 // 워드/출력행
    int o0 = (int)blockIdx.x * 16;
    int r0 = (int)blockIdx.y * 16;
    int ox = threadIdx.x & 15;
    int rx = threadIdx.x >> 4;
    int o = o0 + ox;
    int r = r0 + rx;
    if (o >= n_out || r >= t) return;
    const unsigned* wr = w + (size_t)ids[r] * (size_t)expert_words + (size_t)o * wrow;
    const unsigned* xr = xq + (size_t)r * xq_w;
    const unsigned ones = 0x01010101u;
    float acc = 0.0f;
    for (int sb = 0; sb < n_sub; sb++) {
        int wb = sb * 6;
        float d = bits_f16(wr[wb] & 0xFFFFu);
        float mn = bits_f16((wr[wb] >> 16) & 0xFFFFu);
        unsigned qh = wr[wb + 1];
        unsigned qsw0 = wr[wb + 2];
        unsigned qsw1 = wr[wb + 3];
        unsigned qsw2 = wr[wb + 4];
        unsigned qsw3 = wr[wb + 5];
        int xw = (sb << 5) >> 2;
        int s_lo = 0, s_hi = 0, s1 = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            unsigned qs4 = (k & 3) == 0 ? qsw0 : (k & 3) == 1 ? qsw1 : (k & 3) == 2 ? qsw2 : qsw3;
            unsigned nib = (k < 4) ? (qs4 & 0x0F0F0F0Fu) : ((qs4 >> 4) & 0x0F0F0F0Fu);
            unsigned hb = (qh >> (4 * k)) & 0xFu;
            unsigned hb_p = (hb & 1u) << 4 | (hb & 2u) << 11 | (hb & 4u) << 18 | (hb & 8u) << 25;
            unsigned y = xr[xw + k];
            s_lo += dot4(nib, y, 0);
            s_hi += dot4(hb_p, y, 0);
            s1 += dot4(ones, y, 0);
        }
        float yd = __uint_as_float(xr[(n_in >> 2) + sb]);
        acc += yd * (d * (float)(s_lo + s_hi) + mn * (float)s1);
    }
    out[(size_t)r * n_out + o] = acc;
}

// q4_K MMQ급 타일 — 16출력 × 16행, 스레드당 (출력,행) 직렬 누산(스레드당 수백
// 바이트). 산술은 core `dot_q4k_q8`과 **동일 순서**(it 0..3, 각 it에서
// 저니블→고니블 순 가산)라 비트 동일이 기대된다 — q4-acc-check로 확인.
// q4_K 블록 = 144B/256원소: d(2) dmin(2) scales[12](6비트 스케일·민 8쌍) qs[128].
extern "C" __global__ void q4_gemm_q4k_m(const unsigned* xq, const unsigned* w, double* part,
                                         float* out, int n_in, int n_out, int xq_w, int t) {
    int ox = threadIdx.x & 15;
    int rx = threadIdx.x >> 4;
    int o = (int)blockIdx.x * 16 + ox;
    int r = (int)blockIdx.y * 16 + rx;
    if (o >= n_out || r >= t) return;
    int n_super = n_in >> 8;
    const unsigned char* wr = (const unsigned char*)w + (size_t)o * (n_super * 144);
    const unsigned* xr = xq + (size_t)r * xq_w;
    const unsigned xw_base = (unsigned)(n_in >> 2);
    const unsigned ones = 0x01010101u;
    float acc = 0.0f;
    for (int sIdx = 0; sIdx < n_super; sIdx++) {
        const unsigned char* blk = wr + sIdx * 144;
        float d = bits_f16((unsigned)blk[0] | ((unsigned)blk[1] << 8));
        float dmin = bits_f16((unsigned)blk[2] | ((unsigned)blk[3] << 8));
        const unsigned char* sc = blk + 4;
        const unsigned char* qs = blk + 16;
        #pragma unroll
        for (int it = 0; it < 4; it++) {
            unsigned q[8];
            // 16B 벡터 로드 — 4개 워드를 한 트랜잭션으로. 블록(144B)·it*32는
            // 16의 배수라 정렬이 보장되고, 리틀엔디언 워드 4개는 기존 바이트
            // 조립과 **값이 동일**하다(수치 불변). 스칼라 32회 → 벡터 2회.
            const uint4* p4 = (const uint4*)(qs + it * 32);
            #pragma unroll
            for (int m4 = 0; m4 < 2; m4++) {
                uint4 t4 = p4[m4];
                q[m4 * 4 + 0] = t4.x;
                q[m4 * 4 + 1] = t4.y;
                q[m4 * 4 + 2] = t4.z;
                q[m4 * 4 + 3] = t4.w;
            }
            const int j1 = 2 * it;
            const int j2 = 2 * it + 1;
            unsigned s1, m1, s2, m2;
            if (j1 < 4) { s1 = sc[j1] & 63u; m1 = sc[j1 + 4] & 63u; }
            else { s1 = (sc[j1 + 4] & 0xFu) | ((unsigned)(sc[j1 - 4] >> 6) << 4); m1 = (sc[j1 + 4] >> 4) | ((unsigned)(sc[j1] >> 6) << 4); }
            if (j2 < 4) { s2 = sc[j2] & 63u; m2 = sc[j2 + 4] & 63u; }
            else { s2 = (sc[j2 + 4] & 0xFu) | ((unsigned)(sc[j2 - 4] >> 6) << 4); m2 = (sc[j2 + 4] >> 4) | ((unsigned)(sc[j2] >> 6) << 4); }
            const unsigned* x1 = xr + sIdx * 64 + it * 16;
            const unsigned* x2 = x1 + 8;
            int il1 = 0, il2 = 0, qs1 = 0, qs2 = 0;
            #pragma unroll
            for (int m = 0; m < 8; m++) {
                unsigned y1 = x1[m];
                il1 += dot4(q[m] & 0x0F0F0F0Fu, y1, 0);
                qs1 += dot4(ones, y1, 0);
                unsigned y2 = x2[m];
                il2 += dot4((q[m] >> 4) & 0x0F0F0F0Fu, y2, 0);
                qs2 += dot4(ones, y2, 0);
            }
            float yd1 = __uint_as_float(xr[xw_base + sIdx * 8 + 2 * it]);
            float yd2 = __uint_as_float(xr[xw_base + sIdx * 8 + 2 * it + 1]);
            acc += yd1 * (d * (float)s1 * (float)il1 - dmin * (float)m1 * (float)qs1);
            acc += yd2 * (d * (float)s2 * (float)il2 - dmin * (float)m2 * (float)qs2);
        }
    }
    out[(size_t)r * n_out + o] = acc;
}

// f32 MMQ급 타일 — 16출력 × 16행/블록(256스레드). 스레드당 (출력,행) 한 쌍을
// k 직렬 누산한다. 원판은 (행,출력) 블록 + 64레인 분할이라 스레드당 일감이
// 40 MAC뿐이었고(총 지연), 같은 가중치 행을 행 수만큼 다시 읽었다.
// 누산 순서가 달라 비트 동일은 아니다(허용 오차 계약).
extern "C" __global__ void q4_gemm_f32_m(const float* x, const float* w, float* out,
                                         int n_in, int n_out, int x_stride, int t) {
    int ox = threadIdx.x & 15;
    int rx = threadIdx.x >> 4;
    int o = (int)blockIdx.x * 16 + ox;
    int r = (int)blockIdx.y * 16 + rx;
    if (o >= n_out || r >= t) return;
    const float* xr = x + (size_t)r * x_stride;
    const float* wr = w + (size_t)o * n_in;
    float acc = 0.0f;
    for (int k = 0; k < n_in; k++) acc += xr[k] * wr[k];
    out[(size_t)r * n_out + o] = acc;
}

// f32 가중치 GEMV/GEMM (MoE 라우터 ffn_gate_inp 등) — 무양자화.
// grid (t, n_out), 64레인, x 행 스트라이드 = x_stride (float).
extern "C" __global__ void q4_gemm_f32(const float* x, const float* w, float* out,
                                       int n_in, int n_out, int x_stride) {
    int o = blockIdx.y + blockIdx.z * gridDim.y;
    int l = threadIdx.x;
    if (o >= n_out || l >= 64) return;
    x += (size_t)blockIdx.x * x_stride;
    out += (size_t)blockIdx.x * n_out;
    const float* row = w + (size_t)o * n_in;
    float acc = 0.0f;
    for (int i = l; i < n_in; i += 64) acc += x[i] * row[i];
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

// QSA 마스크드 밀집 GQA — 값 경로 브리지 (f32 KV, 게이트 인터리브 q).
// q [t][n_head·2·hd] (앞 hd = q, 뒤 hd = gate) · ck/cv [n_past][n_kv·hd] ·
// mask [t][n_past] u32 (0 = 차단) · out [t][n_head·hd].
// 온라인 소프트맥스(청크 128) — CPU 참조(max 후 exp/합)와 수학 동일.
#define Q4_NEG (-3.4028235e38f)

// q4_K MMQ 타일 — **전문가 그룹판**. 원판(_m)은 전문가마다 런치해 청크당 7.4만
// 런치(호스트 ~2.7초@pp2048, KTRACE 실측)였다. rowexp(행→전문가) + expert_bytes면
// q4_moe_group_t1 — t=1(디코드) 전문가 그룹화. 층당 호스트 왕복(동기 d2h +
// 테이블 빌드 + h2d 3회)이 스텝의 44%였다(2026-09-14 실측: 48층 × 0.9ms).
// 테이블은 ids의 순수 함수라 GPU에서 만들면 왕복이 사라진다.
// 호스트판과 **동일 순서**(안정 카운팅, 전문가 id 오름차순, 16행 정렬 패딩) →
// 테이블 동일 → GEMM 결과 비트 동일. 단일 스레드(ne≤512, rows=10).
// bw_strided — 가중치 접근 패턴 재현 프로브. q4_K 블록(144B)을 스칼라로
// 읽는 방식(스레드가 같은 오프셋, 행 간 144B 스트라이드)이 실효 대역을
// 얼마나 깎는지 측정한다(2026-09-14: 디코드·프리필의 저대역 미스터리 규명용).
// mode 0 = 순차, mode 1 = 144B 스트라이드(블록 내 스칼라), mode 2 = 스트라이드
// 이지만 워드 단위 벡터 로드.
extern "C" __global__ void bw_strided(const unsigned char* __restrict__ w, unsigned* __restrict__ acc,
                                      int nblocks, int mode) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)nblocks) return;
    const unsigned char* p = w + i * 144;
    unsigned v = 0;
    if (mode == 5) {
        // 완전 병합: 워프 레인이 연속 워드를 읽는다(4B × 32 = 128B 트랜잭션 1개).
        // mode 3(스레드가 자기 행 스트림)과의 차이가 곧 "행 단위 스트라이드"의 대가.
        const unsigned* u = (const unsigned*)p;
        int lane = threadIdx.x & 31;
        unsigned v5 = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) v5 += u[lane + i * 32];
        atomicAdd(&acc[0], v5);
        return;
    }
    if (mode == 3) {
        // GEMM 패턴: 스레드마다 자기 출력행의 연속 스트림을 읽는다(행 간 5KB 스트라이드).
        // warp 32레인이 32개 다른 행 → 라인당 소량만 사용.
        const unsigned short* h = (const unsigned short*)p;
        unsigned v2 = 0;
        for (int i = 0; i < 8; i++) v2 += h[i * 320];
        atomicAdd(&acc[0], v2);
        return;
    }
    if (mode == 4) {
        // 목표 패턴: 워프가 한 행을 협동으로(레인=연속 열) → 완전 병합.
        const unsigned short* h = (const unsigned short*)p;
        int lane = threadIdx.x & 31;
        unsigned v3 = 0;
        for (int i = 0; i < 8; i++) v3 += h[lane + i * 32];
        atomicAdd(&acc[0], v3);
        return;
    }
    if (mode == 0) {
        v = (unsigned)p[0] | ((unsigned)p[1] << 8) | ((unsigned)p[2] << 16) | ((unsigned)p[3] << 24);
    } else if (mode == 1) {
        // 스칼라 4회(가중치 커널의 실제 패턴: 블록 내 소량만 읽음)
        v = (unsigned)p[0] ^ (unsigned)p[4] ^ (unsigned)p[16] ^ (unsigned)p[80];
    } else {
        const uint4* q = (const uint4*)(p + 16);
        uint4 t = *q;
        v = t.x ^ t.y ^ t.z ^ t.w;
    }
    atomicAdd(&acc[0], v);
}

extern "C" __global__ void q4_moe_group_t1(
        const unsigned* __restrict__ ids, int ne, int rows,
        int* __restrict__ out_off, unsigned* __restrict__ perm, unsigned* __restrict__ inv,
        unsigned* __restrict__ rowexp, unsigned* __restrict__ perm_pad,
        unsigned* __restrict__ inv_pad, unsigned* __restrict__ tilexp,
        int* __restrict__ rows_pad_out) {
    // 단일 스레드판은 전역 쓰기 ~700회가 캐시 미스 지연으로 직렬화되어
    // 0.5ms/층(=24ms/스텝)이었다. 스캔만 lane 0이 하고 나머지는 워프/블록
    // 병렬로 — 순서는 호스트판과 동일(안정 카운팅, 16행 정렬 패딩)이라
    // 테이블이 비트 동일하다.
    __shared__ int cnt[513];
    __shared__ int off[513];
    __shared__ int off_pad[513];
    __shared__ int cur[512];
    __shared__ int rows_pad_sh;
    const int l = (int)threadIdx.x;
    const int nt = (int)blockDim.x;
    if (ne > 512) return;
    for (int e = l; e <= ne; e += nt) { cnt[e] = 0; off[e] = 0; off_pad[e] = 0; }
    if (l < 512) cur[l] = 0;
    __syncthreads();
    for (int i = l; i < rows; i += nt) {
        int e = (int)ids[i];
        e = e < 0 ? 0 : (e >= ne ? ne - 1 : e);
        atomicAdd(&cnt[e], 1);
    }
    __syncthreads();
    if (l == 0) {
        int acc = 0;
        for (int e = 0; e < ne; e++) { off[e] = acc; acc += cnt[e]; }
        off[ne] = acc;
        int accp = 0;
        for (int e = 0; e < ne; e++) {
            int r = off[e + 1] - off[e];
            off_pad[e] = accp;
            accp += (r + 15) / 16 * 16;
        }
        off_pad[ne] = accp;
        int rp = accp < 16 ? 16 : accp;
        rows_pad_sh = rp;
        *rows_pad_out = rp;
        for (int e = 0; e < ne; e++) cur[e] = off[e];
        for (int i = 0; i < rows; i++) {
            int e = (int)ids[i];
            e = e < 0 ? 0 : (e >= ne ? ne - 1 : e);
            int p = cur[e];
            cur[e] = p + 1;
            perm[p] = (unsigned)i;
            inv[i] = (unsigned)p;
            rowexp[p] = (unsigned)e;
        }
        for (int e = 0; e < ne; e++) {
            for (int tg = off_pad[e] / 16; tg < off_pad[e + 1] / 16; tg++) tilexp[tg] = (unsigned)e;
        }
    }
    __syncthreads();
    for (int e = l; e <= ne; e += nt) out_off[e] = off[e];
    // 패딩 슬롯 병렬 채움(전역 읽기/쓰기 각 1회, 라인 단위로 흩어짐)
    const int rp = rows_pad_sh;
    for (int pd = l; pd < rp; pd += nt) {
        int e = (int)tilexp[pd >> 4];
        int i = pd - off_pad[e];
        int r = off[e + 1] - off[e];
        if (i < r) {
            unsigned src = perm[off[e] + i];
            perm_pad[pd] = src;
            inv_pad[src] = (unsigned)pd;
        } else {
            perm_pad[pd] = 0;
        }
    }
    int bound = rows * 16 + 16;
    for (int pd = (rp > (bound & ~15) ? rp : (bound & ~15)); pd < bound; pd += nt) {
        perm_pad[pd] = 0;
    }
}

// 한 런치로 전부 처리한다. 산술 순서는 _m과 동일(비트 동일).
extern "C" __global__ void q4_gemm_q4k_ge(const unsigned* xq, const unsigned char* w, double* part,
                                          float* out, const unsigned* rowexp,
                                          int n_in, int n_out, int xq_w, int t, int expert_bytes,
                                          const int* rows_pad_p) {
    // rows_pad_p가 주어지면 행 수를 디바이스에서 읽는다(GPU 그룹화 경로 —
    // 호스트 왕복 없이 정확한 타일 수를 알기 위함). NULL이면 t를 그대로 쓴다.
    if (rows_pad_p != 0) t = *rows_pad_p;
    int ox = threadIdx.x & 15;
    int rx = threadIdx.x >> 4;
    int o = (int)blockIdx.x * 16 + ox;
    int r = (int)blockIdx.y * 16 + rx;
    if (o >= n_out || r >= t) return;
    int n_super = n_in >> 8;
    int e = (int)rowexp[r];
    const unsigned char* wr = w + (size_t)e * expert_bytes + (size_t)o * (n_super * 144);
    const unsigned* xr = xq + (size_t)r * xq_w;
    const unsigned xw_base = (unsigned)(n_in >> 2);
    const unsigned ones = 0x01010101u;
    float acc = 0.0f;
    for (int sIdx = 0; sIdx < n_super; sIdx++) {
        const unsigned char* blk = wr + sIdx * 144;
        float d = bits_f16((unsigned)blk[0] | ((unsigned)blk[1] << 8));
        float dmin = bits_f16((unsigned)blk[2] | ((unsigned)blk[3] << 8));
        const unsigned char* sc = blk + 4;
        const unsigned char* qs = blk + 16;
        #pragma unroll
        for (int it = 0; it < 4; it++) {
            unsigned q[8];
            // 16B 벡터 로드 — 4개 워드를 한 트랜잭션으로. 블록(144B)·it*32는
            // 16의 배수라 정렬이 보장되고, 리틀엔디언 워드 4개는 기존 바이트
            // 조립과 **값이 동일**하다(수치 불변). 스칼라 32회 → 벡터 2회.
            const uint4* p4 = (const uint4*)(qs + it * 32);
            #pragma unroll
            for (int m4 = 0; m4 < 2; m4++) {
                uint4 t4 = p4[m4];
                q[m4 * 4 + 0] = t4.x;
                q[m4 * 4 + 1] = t4.y;
                q[m4 * 4 + 2] = t4.z;
                q[m4 * 4 + 3] = t4.w;
            }
            const int j1 = 2 * it;
            const int j2 = 2 * it + 1;
            unsigned s1, m1, s2, m2;
            if (j1 < 4) { s1 = sc[j1] & 63u; m1 = sc[j1 + 4] & 63u; }
            else { s1 = (sc[j1 + 4] & 0xFu) | ((unsigned)(sc[j1 - 4] >> 6) << 4); m1 = (sc[j1 + 4] >> 4) | ((unsigned)(sc[j1] >> 6) << 4); }
            if (j2 < 4) { s2 = sc[j2] & 63u; m2 = sc[j2 + 4] & 63u; }
            else { s2 = (sc[j2 + 4] & 0xFu) | ((unsigned)(sc[j2 - 4] >> 6) << 4); m2 = (sc[j2 + 4] >> 4) | ((unsigned)(sc[j2] >> 6) << 4); }
            const unsigned* x1 = xr + sIdx * 64 + it * 16;
            const unsigned* x2 = x1 + 8;
            int il1 = 0, il2 = 0, qs1 = 0, qs2 = 0;
            #pragma unroll
            for (int m = 0; m < 8; m++) {
                unsigned y1 = x1[m];
                il1 += dot4(q[m] & 0x0F0F0F0Fu, y1, 0);
                qs1 += dot4(ones, y1, 0);
                unsigned y2 = x2[m];
                il2 += dot4((q[m] >> 4) & 0x0F0F0F0Fu, y2, 0);
                qs2 += dot4(ones, y2, 0);
            }
            float yd1 = __uint_as_float(xr[xw_base + sIdx * 8 + 2 * it]);
            float yd2 = __uint_as_float(xr[xw_base + sIdx * 8 + 2 * it + 1]);
            acc += yd1 * (d * (float)s1 * (float)il1 - dmin * (float)m1 * (float)qs1);
            acc += yd2 * (d * (float)s2 * (float)il2 - dmin * (float)m2 * (float)qs2);
        }
    }
    out[(size_t)r * n_out + o] = acc;
}

// 한 런치로 전부 처리한다. 산술 순서는 _m과 동일(비트 동일).
// q4_gemm_q4k_ge_ids — t=1 전용 direct-ids 그룹 GEMM. 그룹화 테이블(perm_pad/
// rowexp) 없이 **커널이 ids[row]를 직접 읽어** 가중치 베이스를 정한다. 행 순서가
// 곧 ids 순서이므로 gather·scatter가 통째로 사라지고(가중합이 ids 순서를 요구),
// 호스트 왕복(ids d2h + 테이블 빌드 + h2d)도 불필요하다. t=1에서는 k_sel행이
// 모두 같은 입력 벡터이므로 xq_w에 0을 넘겨 전 행이 0번 행을 읽는다.
// 산술: 행별 내적은 _ge와 동일 순서 → 비트 동일 기대.
extern "C" __global__ void q4_gemm_q4k_ge_ids(const unsigned* xq, const unsigned char* w, double* part,
                                          float* out, const unsigned* ids,
                                          int n_in, int n_out, int xq_w, int t, int expert_bytes,
                                          const int* rows_pad_p) {
    (void)rows_pad_p; (void)xq_w;
    int ox = threadIdx.x & 15;
    int rx = threadIdx.x >> 4;
    int o = (int)blockIdx.x * 16 + ox;
    int r = (int)blockIdx.y * 16 + rx;
    if (o >= n_out || r >= t) return;
    int n_super = n_in >> 8;
    int e = (int)ids[r];
    const unsigned char* wr = w + (size_t)e * expert_bytes + (size_t)o * (n_super * 144);
    const unsigned* xr = xq;   // t=1: 전 행이 같은 입력 벡터(0번 행)
    const unsigned xw_base = (unsigned)(n_in >> 2);
    const unsigned ones = 0x01010101u;
    float acc = 0.0f;
    for (int sIdx = 0; sIdx < n_super; sIdx++) {
        const unsigned char* blk = wr + sIdx * 144;
        float d = bits_f16((unsigned)blk[0] | ((unsigned)blk[1] << 8));
        float dmin = bits_f16((unsigned)blk[2] | ((unsigned)blk[3] << 8));
        const unsigned char* sc = blk + 4;
        const unsigned char* qs = blk + 16;
        #pragma unroll
        for (int it = 0; it < 4; it++) {
            unsigned q[8];
            // 16B 벡터 로드 — 4개 워드를 한 트랜잭션으로. 블록(144B)·it*32는
            // 16의 배수라 정렬이 보장되고, 리틀엔디언 워드 4개는 기존 바이트
            // 조립과 **값이 동일**하다(수치 불변). 스칼라 32회 → 벡터 2회.
            const uint4* p4 = (const uint4*)(qs + it * 32);
            #pragma unroll
            for (int m4 = 0; m4 < 2; m4++) {
                uint4 t4 = p4[m4];
                q[m4 * 4 + 0] = t4.x;
                q[m4 * 4 + 1] = t4.y;
                q[m4 * 4 + 2] = t4.z;
                q[m4 * 4 + 3] = t4.w;
            }
            const int j1 = 2 * it;
            const int j2 = 2 * it + 1;
            unsigned s1, m1, s2, m2;
            if (j1 < 4) { s1 = sc[j1] & 63u; m1 = sc[j1 + 4] & 63u; }
            else { s1 = (sc[j1 + 4] & 0xFu) | ((unsigned)(sc[j1 - 4] >> 6) << 4); m1 = (sc[j1 + 4] >> 4) | ((unsigned)(sc[j1] >> 6) << 4); }
            if (j2 < 4) { s2 = sc[j2] & 63u; m2 = sc[j2 + 4] & 63u; }
            else { s2 = (sc[j2 + 4] & 0xFu) | ((unsigned)(sc[j2 - 4] >> 6) << 4); m2 = (sc[j2 + 4] >> 4) | ((unsigned)(sc[j2] >> 6) << 4); }
            const unsigned* x1 = xr + sIdx * 64 + it * 16;
            const unsigned* x2 = x1 + 8;
            int il1 = 0, il2 = 0, qs1 = 0, qs2 = 0;
            #pragma unroll
            for (int m = 0; m < 8; m++) {
                unsigned y1 = x1[m];
                il1 += dot4(q[m] & 0x0F0F0F0Fu, y1, 0);
                qs1 += dot4(ones, y1, 0);
                unsigned y2 = x2[m];
                il2 += dot4((q[m] >> 4) & 0x0F0F0F0Fu, y2, 0);
                qs2 += dot4(ones, y2, 0);
            }
            float yd1 = __uint_as_float(xr[xw_base + sIdx * 8 + 2 * it]);
            float yd2 = __uint_as_float(xr[xw_base + sIdx * 8 + 2 * it + 1]);
            acc += yd1 * (d * (float)s1 * (float)il1 - dmin * (float)m1 * (float)qs1);
            acc += yd2 * (d * (float)s2 * (float)il2 - dmin * (float)m2 * (float)qs2);
        }
    }
    out[(size_t)r * n_out + o] = acc;
}



// 4헤드-퍼-워프 — 선택 목록이 **토큰 단위**라(인덱서가 헤드 합산) 같은 토큰의
// 여러 Q헤드가 같은 K/V 행을 쓴다. 워프가 4헤드를 맡아 K/V 로드를 1/4로 줄인다
// (24 Q헤드 = 2 KV헤드 → 12 Q헤드가 같은 키를 읽던 것을 3워프로 접음).
// 산술은 `_sel`과 동일 순서(키당 헤드별 독립 온라인 소프트맥스) → 비트 동일.
extern "C" __global__ void q4_qsa_attn_sel4(const float* q, const float* ck, const float* cv,
                                            const unsigned* sel_idx, const unsigned* sel_off,
                                            float* out, float kq_scale,
                                            int n_head, int n_kv, int hd, int t) {
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int tok = (int)blockIdx.x * 4 + (wid >> 1);   // 블록 8워프 = 4토큰 × 2헤드묶음
    int h0 = (int)blockIdx.y * 8 + (wid & 1) * 4; // 묶음당 4헤드
    if (tok >= t || h0 + 3 >= n_head || hd != 256) return;
    int kvh = h0 / (n_head / n_kv);
    float qr[4][8], gate[4][8], acc[4][8];
    #pragma unroll
    for (int g = 0; g < 4; g++) {
        const float* qh = q + ((size_t)tok * n_head + h0 + g) * 2 * hd;
        #pragma unroll
        for (int t2 = 0; t2 < 8; t2++) {
            qr[g][t2] = qh[lane + 32 * t2];
            gate[g][t2] = qh[hd + lane + 32 * t2];
            acc[g][t2] = 0.0f;
        }
    }
    float mrun[4], lrun[4];
    #pragma unroll
    for (int g = 0; g < 4; g++) {
        mrun[g] = Q4_NEG;
        lrun[g] = 0.0f;
    }
    const unsigned long long FULL = 0xffffffffffffffffull;
    const int j0 = (int)sel_off[tok];
    const int j1 = (int)sel_off[tok + 1];
    for (int j = j0; j < j1; j++) {
        const int p = (int)sel_idx[j];
        const float* k = ck + (size_t)p * n_kv * hd + kvh * hd;
        const float* v = cv + (size_t)p * n_kv * hd + kvh * hd;
        float kr[8], vr[8];
        #pragma unroll
        for (int t2 = 0; t2 < 8; t2++) {
            kr[t2] = k[lane + 32 * t2];
            vr[t2] = v[lane + 32 * t2];
        }
        #pragma unroll
        for (int g = 0; g < 4; g++) {
            float a = 0.0f;
            #pragma unroll
            for (int t2 = 0; t2 < 8; t2++) a += qr[g][t2] * kr[t2];
            for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(FULL, a, off);
            float sc = a * kq_scale;
            float mr = mrun[g], lr = lrun[g];
            if (sc > mr) {
                float corr = (mr <= Q4_NEG) ? 0.0f : exp_cr(mr - sc);
                #pragma unroll
                for (int t2 = 0; t2 < 8; t2++) acc[g][t2] *= corr;
                lr *= corr;
                mr = sc;
            }
            float e = exp_cr(sc - mr);
            lr += e;
            #pragma unroll
            for (int t2 = 0; t2 < 8; t2++) acc[g][t2] += e * vr[t2];
            mrun[g] = mr;
            lrun[g] = lr;
        }
    }
    #pragma unroll
    for (int g = 0; g < 4; g++) {
        #pragma unroll
        for (int t2 = 0; t2 < 8; t2++) {
            float o = (lrun[g] > 0.0f) ? acc[g][t2] / lrun[g] : 0.0f;
            float gt = 1.0f / (1.0f + exp_cr(-gate[g][t2]));
            out[((size_t)tok * n_head + h0 + g) * hd + lane + 32 * t2] = o * gt;
        }
    }
}

// 워프-퍼-토큰 — 블록의 8워프가 "같은 head의 8토큰"을 각각 맡는다.
// ① K행을 8토큰이 공유(L2 트래픽 1/8, 나머지는 L1) ② 블록 배리어 없음(워프 내부
// 셔플만) — 기존 커널은 128키 청크마다 ~10회 __syncthreads를 돌렸다.
// 산술은 마스크드 온라인 소프트맥스로 동일하나 청크가 32키라 반올림 순서가 다르다.
extern "C" __global__ void q4_qsa_attn_wt(const float* q, const float* ck, const float* cv,
                                           const unsigned* mask, float* out, float kq_scale,
                                           int n_past, int n_head, int n_kv, int hd, int t) {
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int tok = (int)blockIdx.x * 16 + wid;   // 블록당 16토큰 = K/V 행 재사용 2배
    int h = blockIdx.y;
    if (tok >= t || h >= n_head || hd != 256) return;
    int kvh = h / (n_head / n_kv);
    const float* qh = q + ((size_t)tok * n_head + h) * 2 * hd;
    const float* gate = qh + hd;
    const unsigned* mrow = mask + (size_t)tok * n_past;
    float qr[8];
    #pragma unroll
    for (int t2 = 0; t2 < 8; t2++) qr[t2] = qh[lane + 32 * t2];
    float acc[8];
    #pragma unroll
    for (int t2 = 0; t2 < 8; t2++) acc[t2] = 0.0f;
    float mrun = Q4_NEG;
    float lrun = 0.0f;
    const unsigned long long FULL = 0xffffffffffffffffull;   // 이 플랫폼은 64비트 마스크 요구
    // 키를 하나씩: 레인 = 차원(hd/32 = 8개), 점수는 워프 환원으로 만든다.
    // (레인=키로 두면 셔플이 서로 다른 키의 부분합을 섞는다 — 실측 maxrel 1e2 오류.)
    for (int p = 0; p < n_past; p++) {
        bool selv = (mrow[p] != 0u);
        float a = 0.0f;
        if (selv) {
            const float* k = ck + (size_t)p * n_kv * hd + kvh * hd;
            #pragma unroll
            for (int t2 = 0; t2 < 8; t2++) a += qr[t2] * k[lane + 32 * t2];
        }
        // 셔플은 수렴 상태에서(분기 밖) — 발산 셔플은 UB/예외.
        for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(FULL, a, off);
        float s = selv ? a * kq_scale : Q4_NEG;
        if (s > mrun) {
            float corr = (mrun <= Q4_NEG) ? 0.0f : exp_cr(mrun - s);
            #pragma unroll
            for (int t2 = 0; t2 < 8; t2++) acc[t2] *= corr;
            lrun *= corr;
            mrun = s;
        }
        float e = (s > Q4_NEG) ? exp_cr(s - mrun) : 0.0f;
        lrun += e;
        if (e > 0.0f) {
            const float* vr = cv + (size_t)p * n_kv * hd + kvh * hd;
            #pragma unroll
            for (int t2 = 0; t2 < 8; t2++) acc[t2] += e * vr[lane + 32 * t2];
        }
    }
    for (int t2 = 0; t2 < 8; t2++) {
        float o = (lrun > 0.0f) ? acc[t2] / lrun : 0.0f;
        float g = 1.0f / (1.0f + exp_cr(-gate[lane + 32 * t2]));
        out[((size_t)tok * n_head + h) * hd + lane + 32 * t2] = o * g;
    }
}

// 선택 목록판 — `_wt`와 동일 산술·구조이나 키 순회가 마스크 스캔이 아니라
// 압축된 선택 인덱스 목록(오름차순)이다. 목록은 top-k 블록 + 테일을 담는다.
extern "C" __global__ void q4_qsa_attn_sel(const float* q, const float* ck, const float* cv,
                                           const unsigned* sel_idx, const unsigned* sel_off,
                                           float* out, float kq_scale,
                                           int n_head, int n_kv, int hd, int t) {
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int tok = (int)blockIdx.x * 16 + wid;   // 블록당 16토큰 = K/V 행 재사용 2배
    int h = blockIdx.y;
    if (tok >= t || h >= n_head || hd != 256) return;
    int kvh = h / (n_head / n_kv);
    const float* qh = q + ((size_t)tok * n_head + h) * 2 * hd;
    const float* gate = qh + hd;
    float qr[8];
    #pragma unroll
    for (int t2 = 0; t2 < 8; t2++) qr[t2] = qh[lane + 32 * t2];
    float acc[8];
    #pragma unroll
    for (int t2 = 0; t2 < 8; t2++) acc[t2] = 0.0f;
    float mrun = Q4_NEG;
    float lrun = 0.0f;
    const unsigned long long FULL = 0xffffffffffffffffull;   // 이 플랫폼은 64비트 마스크 요구
    // 선택 목록만 순회 — 마스크 확인·발산이 없다(전 키 스캔 대비 키 수가
    // top_k/r 배로 줄고, 내부 루프가 완전 수렴한다).
    const int j0 = (int)sel_off[tok];
    const int j1 = (int)sel_off[tok + 1];
    for (int j = j0; j < j1; j++) {
        const int p = (int)sel_idx[j];
        const float* k = ck + (size_t)p * n_kv * hd + kvh * hd;
        float a = 0.0f;
        #pragma unroll
        for (int t2 = 0; t2 < 8; t2++) a += qr[t2] * k[lane + 32 * t2];
        for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(FULL, a, off);
        float s = a * kq_scale;
        if (s > mrun) {
            float corr = (mrun <= Q4_NEG) ? 0.0f : exp_cr(mrun - s);
            #pragma unroll
            for (int t2 = 0; t2 < 8; t2++) acc[t2] *= corr;
            lrun *= corr;
            mrun = s;
        }
        float e = exp_cr(s - mrun);
        lrun += e;
        const float* vr = cv + (size_t)p * n_kv * hd + kvh * hd;
        #pragma unroll
        for (int t2 = 0; t2 < 8; t2++) acc[t2] += e * vr[lane + 32 * t2];
    }
    for (int t2 = 0; t2 < 8; t2++) {
        float o = (lrun > 0.0f) ? acc[t2] / lrun : 0.0f;
        float g = 1.0f / (1.0f + exp_cr(-gate[lane + 32 * t2]));
        out[((size_t)tok * n_head + h) * hd + lane + 32 * t2] = o * g;
    }
}

// ─── qwen4exp 프레임 op 커널 (plans/64 P1) ───
// 수치 계약: core `qwen4exp/stages/*`·`layers.rs`와 동일 산술. rms/norm_gated는
// 기존 커널(rms_part/finish·norm_gated_silu 구조)의 세그먼트 f32 + f64 결합을
// 그대로 따르고, 나머지는 원소별 순서를 유지한다.

// in-place: v ← silu(v/div)
extern "C" __global__ void q4_silu_div(float* t, float div, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float v = t[j] / div;
    t[j] = v / (1.0f + exp_cr(-v));
}

// in-place sigmoid
extern "C" __global__ void q4_sigmoid(float* t, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    t[j] = 1.0f / (1.0f + exp_cr(-t[j]));
}

// in-place scale
extern "C" __global__ void q4_scale(float* t, float s, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    t[j] *= s;
}

// 행별 L2 정규화 (in-place, d원소 행) — ops::l2_norm 미러(순차 f32 합 + floor)
extern "C" __global__ void q4_l2_rows(float* x, float eps, int d) {
    int row = blockIdx.x;
    int u = threadIdx.x;
    if (u >= 32) return;
    float part = 0.0f;
    int chunk = (d + 31) >> 5;
    int lo = u * chunk;
    if (lo < d) {
        int hi = min(lo + chunk, d);
        for (int i = lo; i < hi; i++) {
            float v = x[row * d + i];
            part += v * v;
        }
    }
    double sum = 0.0;
    #pragma unroll
    for (int j = 0; j < 32; j++) sum += (double)__shfl_sync(0xFFFFFFFFFFFFFFFFull, part, j);
    float s = 1.0f / fmaxf(sqrtf((float)sum), eps);
    for (int i = u; i < d; i += 32) x[row * d + i] *= s;
}

// hc 게이트 적용 + 스트림 평균: out[i] = Σ_s xn[s·n+i]·σ(gate[s·n+i]) / hc
// total = t·n (토큰 배치) — 행 = 토큰.
extern "C" __global__ void q4_hc_gate_mean(const float* xn, const float* gate, float* out, int hc, int n, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int ti = i / n;
    int ii = i % n;
    float acc = 0.0f;
    for (int s = 0; s < hc; s++) {
        int k = ti * hc * n + s * n + ii;
        acc += xn[k] * (1.0f / (1.0f + exp_cr(-gate[k])));
    }
    out[i] = acc / (float)hc;
}

// hc combine: res[s·n+i] += out[i]·2σ(inj[s]/hc)
// total = t·n (토큰 배치) — inj는 토큰당 hc개.
extern "C" __global__ void q4_hc_combine(float* res, const float* out, const float* inj, int hc, int n, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int ti = i / n;
    int ii = i % n;
    for (int s = 0; s < hc; s++) {
        float g = 2.0f / (1.0f + exp_cr(-inj[ti * hc + s] / (float)hc));
        res[ti * hc * n + s * n + ii] += out[i] * g;
    }
}

// GDN norm_gated(sigmoid z) — qwen4exp 변형 (qwen35는 silu).
// out = rms(o_head)·σ(z)·w_head. 행 = (y·n_h + head), w는 헤드별 [d].
extern "C" __global__ void q4_norm_gated_sig(const float* o, const float* z, const float* w,
                                             float* out, float eps, int d, int n_h) {
    int row = blockIdx.y * n_h + blockIdx.x;
    int u = threadIdx.x;
    if (blockIdx.x >= n_h) return;
    int xb = row * d;
    int wb = blockIdx.x * d;
    int chunk = (d + 31) >> 5;
    float part = 0.0f;
    {
        int lo = u * chunk;
        if (lo < d) {
            int hi = min(lo + chunk, d);
            for (int i = lo; i < hi; i++) {
                float dv = o[xb + i];
                part += dv * dv;
            }
        }
    }
    double sum = 0.0;
    #pragma unroll
    for (int j = 0; j < 32; j++) sum += (double)__shfl_sync(0xFFFFFFFFFFFFFFFFull, part, j);
    float inv = 1.0f / (float)sqrt(sum / (double)d + (double)eps);
    for (int i = u; i < d; i += 32) {
        out[xb + i] = o[xb + i] * inv * w[wb + i] * (1.0f / (1.0f + exp_cr(-z[xb + i])));
    }
}

// MoE 라우팅: softmax → top-k(확률 내림차순, 동점은 인덱스 오름차순 = CPU
// 안정 정렬 미러) → 재정규화(하한 2^-14). 단일 스레드(512 전문가) — CPU와
// 동일 순서를 보장하는 것이 성능보다 중요하다.
extern "C" __global__ void q4_moe_top10(const float* route, unsigned* ids, float* wt,
                                        int n_exp, int k_sel) {
    if (threadIdx.x != 0) return;
    int tok = blockIdx.x;
    const float* rt = route + (size_t)tok * n_exp;
    ids += (size_t)tok * k_sel;
    wt += (size_t)tok * k_sel;
    float mx = -3.4028235e38f;
    for (int e = 0; e < n_exp; e++) mx = fmaxf(mx, rt[e]);
    float zs = 0.0f;
    for (int e = 0; e < n_exp; e++) zs += exp_cr(rt[e] - mx);
    // 삽입 정렬 — 후보를 인덱스 오름차순으로 넣고 "확률 이상" 뒤에 삽입(안정)
    int sel[64];          // k_sel ≤ 64 가정 (qwen4exp = 10)
    float sp[64];
    if (k_sel > 64) return;
    int cnt = 0;
    for (int e = 0; e < n_exp; e++) {
        float p = exp_cr(rt[e] - mx) / zs;
        int pos = cnt;
        for (int j = 0; j < cnt; j++) {
            if (p > sp[j]) { pos = j; break; }
        }
        if (pos < k_sel) {
            int last = min(cnt, k_sel - 1);
            for (int j = last; j > pos; j--) { sel[j] = sel[j - 1]; sp[j] = sp[j - 1]; }
            sel[pos] = e;
            sp[pos] = p;
            if (cnt < k_sel) cnt++;
        }
    }
    float wsum = 0.0f;
    for (int j = 0; j < cnt; j++) wsum += sp[j];
    wsum = fmaxf(wsum, 6.103515625e-5f);
    for (int j = 0; j < k_sel; j++) {
        int e = (j < cnt) ? sel[j] : 0;
        ids[j] = (unsigned)e;
        wt[j] = (j < cnt) ? sp[j] / wsum : 0.0f;
    }
}

// MoE 전문가 가중 합: out[i] = Σ_e wt[e]·ys[e·n+i]
extern "C" __global__ void q4_moe_weighted_sum(const float* ys, const float* wt, float* out, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.0f;
    for (int e = 0; e < k; e++) acc += wt[e] * ys[e * n + i];
    out[i] = acc;
}

// q4_moe_top10_m — 워프(32레인)/토큰 병렬판. 원판은 **1스레드/토큰**이라
// 디코드(t=1)에서 256회 exp를 한 스레드가 돌아 0.85ms/호출이었다(디코드
// 토큰당 48콜 = 41ms = 디코드의 36%). max·exp를 워프 병렬로 계산하고
// 선택 삽입 정렬은 원판과 **동일 로직**이라 결과는 비트 동일.
extern "C" __global__ void q4_moe_top10_m(const float* route, unsigned* ids, float* wt,
                                          int n_exp, int k_sel) {
    __shared__ float sh[1024];
    // 선택 삽입 정렬용 배열 — 지역배열(동적 인덱싱)은 local memory(DRAM 왕복)로
    // 떨어져 512회 루프에서 0.9ms/호출이 됐다. shared로 올리면 알고리즘·순서가
    // 그대로여서 결과는 비트 동일하고 지연만 사라진다(2026-09-14).
    __shared__ int sh_sel[64];
    __shared__ float sh_sp[64];
    int tok = blockIdx.x;
    int l = threadIdx.x;
    if (l >= 32 || n_exp > 1024) return;
    const float* rt = route + (size_t)tok * n_exp;
    float mx = -3.4028235e38f;
    for (int e = l; e < n_exp; e += 32) mx = fmaxf(mx, rt[e]);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float pj = __shfl_xor_sync(0xffffffffffffffffull, mx, off);
        mx = fmaxf(mx, pj);
    }
    float zs = 0.0f;
    for (int e = l; e < n_exp; e += 32) {
        float p = exp_cr(rt[e] - mx);
        sh[e] = p;
        zs += p;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        zs += __shfl_xor_sync(0xffffffffffffffffull, zs, off);
    zs = __shfl_sync(0xffffffffffffffffull, zs, 0);
    __syncwarp();
    if (k_sel > 32 || n_exp > 32 * 16) return;
    int* sel = sh_sel;
    float* sp = sh_sp;
    // k_sel 라운드의 **워프 병렬 선택**: 레인당 16개를 담당하고, 매 라운드
    // (a) 레인이 자기 미사용분의 최대를 찾고 (b) 워프 셔플로 전역 최대를 뽑아
    // 승자 레인이 그 슬롯을 소진한다. 기존 lane-0 삽입정렬(512회 직렬, 호출당
    // 278µs 실측 = 디코드 최대 단일 커널)을 수 µs로 줄인다. 동률은 셔플 비교
    // (`>`) 규칙으로 결정되며 기존 안정 삽입정렬과 같은 내림차순을 준다.
    int cnt = 0;
    {
        const unsigned long long m = 0xffffffffffffffffull;
        unsigned used = 0;                 // 레인의 16슬롯 소진 마스크
        const int base = l * 16;
        ids += (size_t)tok * k_sel;
        wt += (size_t)tok * k_sel;
        for (int r = 0; r < k_sel; r++) {
            float best = -3.4028235e38f;
            int bi = -1;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int e = base + i;
                if (e < n_exp && !(used & (1u << i))) {
                    float p = sh[e];
                    if (p > best) { best = p; bi = i; }
                }
            }
            int bl = l;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                float ob = __shfl_xor_sync(m, best, off);
                int ol = __shfl_xor_sync(m, bl, off);
                if (ob > best) { best = ob; bl = ol; }
            }
            int wbi = __shfl_sync(m, bi, bl);   // 승자 레인의 슬롯
            if (l == bl) used |= (1u << wbi);
            if (l == 0 && bi >= 0 && best > -3.4028235e38f && cnt < k_sel) {
                sel[cnt] = bl * 16 + wbi;
                sp[cnt] = best / zs;
                cnt++;
            }
        }
    }
    if (l != 0) return;
    float wsum = 0.0f;
    for (int j = 0; j < cnt; j++) wsum += sp[j];
    wsum = fmaxf(wsum, 6.103515625e-5f);
    for (int j = 0; j < k_sel; j++) {
        int e = (j < cnt) ? sel[j] : 0;
        ids[j] = (unsigned)e;
        wt[j] = (j < cnt) ? sp[j] / wsum : 0.0f;
    }
}

// qwen4exp GDN AR — 상태 = core gdn_ar_batch 레이아웃 (st[kdim*d+dv])
extern "C" __global__ void q4_gdn_ar_w(float* s, const float* q, const float* k, const float* v,
                                       const float* beta_ge, float* out, int d, int k_stride,
                                       int v_stride, int h_v, int h_k, float scale, int t) {
    // qwen4exp 프레임용 — (a) 상태 레이아웃이 core `gdn_ar_batch`와 동일
    // (st[kdim*d + dv]), (b) **임의 d** 지원: 레인당 kdim = lane + 32k.
    // (qwen35용 gdn_ar_w_swap은 전치 레이아웃 + 레인당 kdim 4개 고정 = d=128 전용.)
    int u = blockIdx.x;                     // dv
    int pair = blockIdx.y;
    int lane = threadIdx.x & 31;
    if (u >= d) return;
    int base_s = pair * d * d;
    for (int ti = 0; ti < t; ti++) {
        int h = pair % h_v;
        int kh = h % h_k;
        int qk0 = ti * k_stride + kh * d;
        int v0 = ti * v_stride + h * d;
        float beta = beta_ge[ti * h_v * 2 + pair * 2];
        float g_exp = beta_ge[ti * h_v * 2 + pair * 2 + 1];
        // sk[dv] = Σ_kdim (S·e^g)[kdim][dv]·k[kdim]
        float part = 0.0f;
        for (int kd = lane; kd < d; kd += 32) {
            float sv = s[base_s + kd * d + u] * g_exp;
            s[base_s + kd * d + u] = sv;
            part += sv * k[qk0 + kd];
        }
        float sk = part;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1)
            sk += __shfl_sync(0xFFFFFFFFFFFFFFFFull, sk, (lane ^ off) & 31);
        float delta = (v[v0 + u] - sk) * beta;
        // S[kdim][dv] += k[kdim]·delta;  o[dv] = Σ_kdim S[kdim][dv]·q[kdim]
        float op = 0.0f;
        for (int kd = lane; kd < d; kd += 32) {
            float sv = s[base_s + kd * d + u] + k[qk0 + kd] * delta;
            s[base_s + kd * d + u] = sv;
            op += sv * q[qk0 + kd];
        }
        float otot = op;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1)
            otot += __shfl_sync(0xFFFFFFFFFFFFFFFFull, otot, (lane ^ off) & 31);
        if (lane == 0) out[v0 + u] = otot * scale;
    }
}


// MoE (t>1) — (토큰, 전문가) 페어 행 gather: xsel[(tok·k_sel+e)·n + i] = mix[tok·n + i]
extern "C" __global__ void q4_moe_gather(const float* mix, float* xsel, int n, int k_sel, int t) {
    long j = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)t * k_sel * n;
    if (j >= total) return;
    int ti = (int)(j / ((long)k_sel * n));
    int i = (int)(j % n);
    xsel[j] = mix[(long)ti * n + i];
}

// 행 순열: dst[g] = src[perm[g]] (row_u32 = 행당 u32 수, 행은 4바이트 정렬).
// MoE 전문가 그룹화의 gather·scatter 공용 — 산란은 역순열을 넘긴다.
extern "C" __global__ void q4_rows_permute_u32(const unsigned* src, const unsigned* perm,
                                               unsigned* dst, int row_u32, int n) {
    int g = blockIdx.x;
    if (g >= n || row_u32 <= 0) return;
    const unsigned* s = src + (size_t)perm[g] * row_u32;
    unsigned* d = dst + (size_t)g * row_u32;
    for (int i = threadIdx.x; i < row_u32; i += blockDim.x) d[i] = s[i];
}

// MoE (t>1) — 전문가 가중 합: out[tok·n + i] = Σ_e wt[tok·k_sel + e]·ys[(tok·k_sel+e)·n + i]
extern "C" __global__ void q4_moe_scatter(const float* ys, const float* wt, float* out, int k_sel, int n, int t) {
    long j = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)t * n;
    if (j >= total) return;
    int tok = (int)(j / n);
    int i = (int)(j % n);
    float acc = 0.0f;
    for (int e = 0; e < k_sel; e++)
        acc += wt[(size_t)tok * k_sel + e] * ys[((size_t)tok * k_sel + e) * n + i];
    out[j] = acc;
}



// 토큰 배치 axpy: y[t·n_i + i] += x[...]·s[t]
extern "C" __global__ void q4_axpy_scaled_t(float* y, const float* x, const float* s, int n, int per) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    y[j] = y[j] + x[j] * s[j / per];
}
