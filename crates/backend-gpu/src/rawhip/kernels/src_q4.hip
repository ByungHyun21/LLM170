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

// K-분할 부분합 리덕션 — k 오름차순(결정적).
extern "C" __global__ void q4_gemm_q4k_ids_reduce(const double* __restrict__ part,
                                                  float* __restrict__ out, int n, int ksplit) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    double s = 0.0;
    for (int k = 0; k < ksplit; k++) s += part[(size_t)i * ksplit + k];
    out[i] = (float)s;
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

// plans/73: f32 GEMV **워프-퍼-출력판**(디코드 t=1). 원판(64스레드/출력)은
// 저출력 형상에서 GPU 전체가 놀았다 — hc inject [10240→4]는 블록 4개로
// 48µs(164KB!). 레인이 연속 4원소(float4)를 소유해 완전 병합 + MLP 4배.
// 누산 순서가 원판과 달라진다(스칼라 스트라이드 → 4원소 청크) — f32 재배열
// 편차(~1e-6 상대) 계열, 게이트 스트림 동일성으로 검증한다.
extern "C" __global__ void __launch_bounds__(256) q4_gemm_f32_w(
        const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out,
        int n_in, int n_out) {
    const int o = (int)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (o >= n_out) return;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const float4* r4 = reinterpret_cast<const float4*>(w + (size_t)o * n_in);
    const int n4 = n_in >> 2;
    float4 a = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = lane; k < n4; k += 32) {
        const float4 xv = x4[k];
        const float4 rv = r4[k];
        a.x += xv.x * rv.x;
        a.y += xv.y * rv.y;
        a.z += xv.z * rv.z;
        a.w += xv.w * rv.w;
    }
    float acc = (a.x + a.y) + (a.z + a.w);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffffffffffffull, acc, off);
    if (lane == 0) out[o] = acc;
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
        int* __restrict__ rows_pad_out, int bound) {
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
        // rowexp는 GEMM이 t = rows_pad로 읽으므로 패딩 슬롯까지 채워야 한다.
        rowexp[pd] = (unsigned)e;
        if (i < r) {
            unsigned src = perm[off[e] + i];
            perm_pad[pd] = src;
            inv_pad[src] = (unsigned)pd;
        } else {
            perm_pad[pd] = 0;
        }
    }
    // [rp, bound) 전체를 0으로 — 종전엔 16정렬 꼬리만 채워 [rp, bound&~15)이
    // 낡은 값으로 남았고, gather가 그 인덱스로 xq를 읽어 OOB(폴트/오염) 냈다
    // (plans/68 12차). 0 = 원본 0번 행(유효) — 어차피 패딩 슬롯이라 안 읽힌다.
    for (int pd = rp; pd < bound; pd += nt) {
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
    // K-분할: z축이 슈퍼블록 구간을 나눠 블록 수를 ksplit배로(타일 40블록=1/CU
    // 이던 점유율을 올려 K-루프 로드 지연을 가린다). 부분합은 part에 남기고
    // reduce 커널이 k 오름차순으로 합산(결정적).
    const int ksplit = (int)gridDim.z;
    const int s0 = (int)(((long)n_super * blockIdx.z) / ksplit);
    const int s1 = (int)(((long)n_super * (blockIdx.z + 1)) / ksplit);
    float acc = 0.0f;
    for (int sIdx = s0; sIdx < s1; sIdx++) {
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
    if (ksplit > 1) {
        part[((size_t)r * n_out + o) * ksplit + blockIdx.z] = (double)acc;
    } else {
        out[(size_t)r * n_out + o] = acc;
    }
}

// ─── plans/108 P7: MoE direct-ids dmmv — vk fn_moe_ids2/fn_moe_ids51 이식 ───
// f32 활성 직소비(MoE 활성 quant·K-분할 reduce·카운팅 정렬 전부 스킵).
// 64스레드·2행/WG·grid=(rows, ceil(n_out/2), gz), 축소는 dmmv 계열 f64 트리.
// 커널이 ids[r]로 전문가 바이트 베이스를 직접 산출(간접 참조).

// q4_K (144B [d f16][m f16][sc 12B][qs 128B]) — llama mul_mat_vec_q4_k 스레드
// 매핑: itid=tid%16 → il=itid/4, ir=itid%4, v_im=il/2, v_in=il%2; ix=tid/16.
// 스케일 3워드에서 v_im 하/상반 u16 3개(s0/s4/s8)를 뽑아 워드 조합으로
// 8 서브스케일을 복호(vk fn_moe_ids2와 동일 비트열). fma 체인은
// -ffp-contract=off 곱-가산 괄호 중첩으로 동일 결합순서.
extern "C" __global__ void __launch_bounds__(64) q4_gemm_q4k_dmmv_ids(
        const float* __restrict__ x, const unsigned* __restrict__ w,
        float* __restrict__ out, const unsigned* __restrict__ ids,
        int n_in, int n_out, int rows, int expert_bytes) {
    const int tid = threadIdx.x;
    if (tid >= 64) return;
    const int r = blockIdx.x;                 // 행 = (토큰,전문가) 쌍
    if (r >= rows) return;
    const int wg = blockIdx.y + (int)blockIdx.z * gridDim.y;
    const int first_row = wg << 1;
    if (first_row >= n_out) return;
    const int num_rows = min(2, n_out - first_row);
    const int n_super = n_in >> 8;
    const size_t ebase = (size_t)ids[r] * (size_t)expert_bytes;
    const size_t rstride = (size_t)n_super * 144u;
    const int itid = tid & 15;
    const int ix = tid >> 4;                  // 슈퍼블록 스트라이드(4 iter)
    const int il = itid >> 2;
    const int ir = itid & 3;
    const int v_im = il >> 1;
    const int v_in = il & 1;
    const int l0 = 4 * (2 * ir + v_in);
    const int q_off = 32 * v_im + l0;         // qs 내 바이트 오프셋(4B 정렬)
    const int y_off = 64 * v_im + l0;
    const float* xr = x + (size_t)r * n_in;
    float t0 = 0.0f, t1 = 0.0f;
    for (int i = ix; i < n_super; i += 4) {
        const int y1 = i * 256 + y_off;
        const float by0 = xr[y1],       by1 = xr[y1+1],     by2 = xr[y1+2],     by3 = xr[y1+3];
        const float by4 = xr[y1+32],    by5 = xr[y1+33],    by6 = xr[y1+34],    by7 = xr[y1+35];
        const float by8 = xr[y1+128],   by9 = xr[y1+129],   by10 = xr[y1+130],  by11 = xr[y1+131];
        const float by12 = xr[y1+160],  by13 = xr[y1+161],  by14 = xr[y1+162],  by15 = xr[y1+163];
        #pragma unroll
        for (int n = 0; n < 2; n++) {
            if (n >= num_rows) break;
            const size_t bb = ebase + (size_t)(first_row + n) * rstride + (size_t)i * 144u;
            const unsigned wq0 = w[bb >> 2];
            const float d = bits_f16(wq0 & 0xFFFFu);
            const float dmin = bits_f16(wq0 >> 16);
            // 스케일 s0/s4/s8 — 블록 바이트 4+2v/8+2v/12+2v 의 u16 (v_im 하/상반)
            const unsigned s0 = (w[(bb + 4) >> 2]  >> (v_im << 4)) & 0xFFFFu;
            const unsigned s4 = (w[(bb + 8) >> 2]  >> (v_im << 4)) & 0xFFFFu;
            const unsigned s8 = (w[(bb + 12) >> 2] >> (v_im << 4)) & 0xFFFFu;
            const unsigned s_l = (s4 << 16) | s0;
            const unsigned s_h = (s_l & 0xC0C0C0C0u) >> 2;
            const unsigned s_84 = (((s8 << 12) | s8) & 0x0F0F0F0Fu) | s_h;
            const unsigned sl4 = s_l & 0x3F3F3F3Fu;
            const float sc0 = (float)(sl4 & 0x3Fu),        sc1 = (float)((sl4 >> 8) & 0x3Fu);
            const float sc2 = (float)((sl4 >> 16) & 0x3Fu), sc3 = (float)(sl4 >> 24);
            const float sc4 = (float)(s_84 & 0xFFu),        sc5 = (float)((s_84 >> 8) & 0xFFu);
            const float sc6 = (float)((s_84 >> 16) & 0xFFu), sc7 = (float)(s_84 >> 24);
            const unsigned qa = w[(bb + 16 + q_off) >> 2];
            const unsigned qb = w[(bb + 80 + q_off) >> 2];
            const unsigned alo = qa & 0x0F0F0F0Fu, ahi = (qa >> 4) & 0x0F0F0F0Fu;
            const unsigned blo = qb & 0x0F0F0F0Fu, bhi = (qb >> 4) & 0x0F0F0F0Fu;
            const float q0 = (float)(alo & 0xFu),        q1 = (float)((alo >> 8) & 0xFu);
            const float q2 = (float)((alo >> 16) & 0xFu), q3 = (float)(alo >> 24);
            const float q4 = (float)(ahi & 0xFu),        q5 = (float)((ahi >> 8) & 0xFu);
            const float q6 = (float)((ahi >> 16) & 0xFu), q7 = (float)(ahi >> 24);
            const float q8 = (float)(blo & 0xFu),        q9 = (float)((blo >> 8) & 0xFu);
            const float q10 = (float)((blo >> 16) & 0xFu), q11 = (float)(blo >> 24);
            const float q12 = (float)(bhi & 0xFu),       q13 = (float)((bhi >> 8) & 0xFu);
            const float q14 = (float)((bhi >> 16) & 0xFu), q15 = (float)(bhi >> 24);
            const float sx = by0*q0 + (by1*q1 + (by2*q2 + by3*q3));
            const float sy = by4*q4 + (by5*q5 + (by6*q6 + by7*q7));
            const float sz = by8*q8 + (by9*q9 + (by10*q10 + by11*q11));
            const float sw = by12*q12 + (by13*q13 + (by14*q14 + by15*q15));
            const float smin = (by0*sc2 + (by4*sc3 + (by8*sc6 + by12*sc7)))
                             + (by1*sc2 + (by5*sc3 + (by9*sc6 + by13*sc7)))
                             + (by2*sc2 + (by6*sc3 + (by10*sc6 + by14*sc7)))
                             + (by3*sc2 + (by7*sc3 + (by11*sc6 + by15*sc7)));
            const float rr = d * (sx*sc0 + (sy*sc1 + (sz*sc4 + sw*sc5))) - dmin * smin;
            if (n == 0) t0 += rr; else t1 += rr;
        }
    }
    __shared__ double sh0[32], sh1[32];
    double a0 = (double)t0, a1 = (double)t1;
    if (tid >= 32) { sh0[tid - 32] = a0; sh1[tid - 32] = a1; }
    __syncthreads();
    if (tid < 32) {
        a0 += sh0[tid];
        a1 += sh1[tid];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            a0 += __shfl_down_sync(0xffffffffffffffffull, a0, off);
            a1 += __shfl_down_sync(0xffffffffffffffffull, a1, off);
        }
        if (tid == 0) {
            float* orow = out + (size_t)r * n_out;
            orow[first_row] = (float)a0;
            if (num_rows > 1) orow[first_row + 1] = (float)a1;
        }
    }
}

// q5_1 (24B [d f16][m f16][qh 4B][qs 16B]) — gemv8 골격(레인 8원소, 스트라이드
// 512) + ids 간접. w = d·v + m (v = lo/hi니블 + qh 5번째 비트), 원소별
// f32 2-누산(aq, sx) 드레인 — vk fn_moe_ids51과 동일 배치.
extern "C" __global__ void __launch_bounds__(64) q5_1_gemm_dmmv_ids(
        const float* __restrict__ x, const unsigned* __restrict__ w,
        float* __restrict__ out, const unsigned* __restrict__ ids,
        int n_in, int n_out, int rows, int expert_bytes) {
    const int l = threadIdx.x;
    if (l >= 64) return;
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int wg = blockIdx.y + (int)blockIdx.z * gridDim.y;
    const int first_row = wg << 1;
    if (first_row >= n_out) return;
    const int num_rows = min(2, n_out - first_row);
    const int n_sub = n_in >> 5;
    const size_t ebase = (size_t)ids[r] * (size_t)expert_bytes;
    const size_t rstride = (size_t)n_sub * 24u;
    const float* xr = x + (size_t)r * n_in;
    float t0 = 0.0f, t1 = 0.0f;
    for (int col0 = l << 3; col0 < n_in; col0 += 512) {
        float xw[8];
        float sx = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            xw[j] = xr[col0 + j];
            sx += xw[j];
        }
        #pragma unroll
        for (int n = 0; n < 2; n++) {
            if (n >= num_rows) break;
            const size_t bb = ebase + (size_t)(first_row + n) * rstride
                            + (size_t)(col0 >> 5) * 24u;
            const unsigned w0 = w[bb >> 2];
            const float d = bits_f16(w0 & 0xFFFFu);
            const float m = bits_f16(w0 >> 16);
            const unsigned qh = w[(bb + 4) >> 2];
            const int rem = col0 & 31;
            const int b0 = (rem >= 16) ? (rem - 16) : rem;   // 8-정렬 → {0,8}
            const int k = b0 >> 1;                            // {0,4}
            const int qw = (int)((bb + 8 + 2 * k) >> 2);
            const unsigned qa = w[qw], qb = w[qw + 1];
            const unsigned qh8 = (qh >> rem) & 0xFFu;
            float aq = 0.0f;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const unsigned bytev = ((j < 4 ? qa : qb) >> (8 * (j & 3))) & 0xFFu;
                const unsigned vq = (rem >= 16 ? (bytev >> 4) : (bytev & 0xFu))
                                  | (((qh8 >> j) & 1u) << 4);
                aq += (float)vq * xw[j];
            }
            const float rr = d * aq + m * sx;
            if (n == 0) t0 += rr; else t1 += rr;
        }
    }
    __shared__ double sh0[32], sh1[32];
    double a0 = (double)t0, a1 = (double)t1;
    if (l >= 32) { sh0[l - 32] = a0; sh1[l - 32] = a1; }
    __syncthreads();
    if (l < 32) {
        a0 += sh0[l];
        a1 += sh1[l];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            a0 += __shfl_down_sync(0xffffffffffffffffull, a0, off);
            a1 += __shfl_down_sync(0xffffffffffffffffull, a1, off);
        }
        if (l == 0) {
            float* orow = out + (size_t)r * n_out;
            orow[first_row] = (float)a0;
            if (num_rows > 1) orow[first_row + 1] = (float)a1;
        }
    }
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

// 6헤드-퍼-워프 — `_sel4`의 확장. `_sel4`는 게이트를 레지스터에 들고 있어
// qr+gate+acc = 96 floats/lane이 한계였고, 그래서 24헤드가 6워프로 갈려
// K/V 행을 6번 읽는다(프리필 대역폭의 직접 원인: t=2048 콜당 ~34GB/236GB/s
// ~= 실측 101ms). 게이트는 마지막에 전역에서 한 번 읽으면 되므로 레지스터에서
// 빼면 6헤드(qr[6][8]+acc[6][8]=96)가 같은 예산에 들어가 재독이 4회로 준다.
// 산술 순서는 `_sel4`와 동일(헤드별 독립 온라인 소프트맥스) → 비트 동일.
// 실측(2026-09-14): pp2048 9,012-9,118 -> 8,748-8,908ms(-1.6~2.9%), diverse 스트림
// 동일. 분할판(_sel4s)도 6헤드로 올려 pp8192 tg가 1,035 -> 1,020ms가 됐다.
extern "C" __global__ void q4_qsa_attn_sel6(const float* q, const float* ck, const float* cv,
                                            const unsigned* sel_idx, const unsigned* sel_off,
                                            float* out, float kq_scale,
                                            int n_head, int n_kv, int hd, int t) {
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int tok = (int)blockIdx.x * 4 + (wid >> 1);   // 블록 8워프 = 4토큰 × 2헤드묶음
    int h0 = (int)blockIdx.y * 12 + (wid & 1) * 6; // 묶음당 6헤드
    if (tok >= t || h0 + 5 >= n_head || hd != 256) return;
    int kvh = h0 / (n_head / n_kv);
    float qr[6][8], acc[6][8];
    #pragma unroll
    for (int g = 0; g < 6; g++) {
        const float* qh = q + ((size_t)tok * n_head + h0 + g) * 2 * hd;
        #pragma unroll
        for (int t2 = 0; t2 < 8; t2++) {
            qr[g][t2] = qh[lane + 32 * t2];
            acc[g][t2] = 0.0f;
        }
    }
    float mrun[6], lrun[6];
    #pragma unroll
    for (int g = 0; g < 6; g++) { mrun[g] = Q4_NEG; lrun[g] = 0.0f; }
    const unsigned long long FULL = 0xffffffffffffffffull;
    const int j0 = (int)sel_off[tok];
    const int j1 = (int)sel_off[tok + 1];
    for (int j = j0; j < j1; j++) {
        const int p = (int)sel_idx[j];
        const float* k = ck + (size_t)p * n_kv * hd + kvh * hd;
        const float* v = cv + (size_t)p * n_kv * hd + kvh * hd;
        float kr[8], vr[8];
        #pragma unroll
        for (int t2 = 0; t2 < 8; t2++) { kr[t2] = k[lane + 32 * t2]; vr[t2] = v[lane + 32 * t2]; }
        #pragma unroll
        for (int g = 0; g < 6; g++) {
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
    for (int g = 0; g < 6; g++) {
        const float* gh = q + ((size_t)tok * n_head + h0 + g) * 2 * hd + hd;
        #pragma unroll
        for (int t2 = 0; t2 < 8; t2++) {
            float o = (lrun[g] > 0.0f) ? acc[g][t2] / lrun[g] : 0.0f;
            float gt = 1.0f / (1.0f + exp_cr(-gh[lane + 32 * t2]));
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
// 실측(2026-09-14, pp8192 t=1): 1.425 ms/호출 = 17.1 ms/스텝으로 최대 단일 커널.
// 워프 = (토큰, 헤드) 하나이고 선택 목록은 토큰별이라 **24헤드가 같은 K/V 행을
// 각자 다시 읽는다**: 8192위치 x 256 x 2 x 24헤드 = 403 MB/층 -> 236 GB/s로 1.7ms로
// 측정치와 일치(대역폭 한계). 헤드 공유(4헤드판)는 t=1에서 점유율 때문에 중립이었다.
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

// q4_qsa_attn_sel4s — t=1(디코드) 전용 위치 분할판. `_sel4`는 워프가 선택목록
// 전체를 **직렬** 순회한다: t=1에서 그리드는 24블록뿐이고 워프당 8192위치 x
// ~35사이클 ≈ 200us가 그대로 스텝 비용이 된다(실측 1.425ms/콜, pp8192).
// 여기서는 목록을 n_splits로 쪼개 (split, 헤드묶음) 그리드로 펼쳐 지연을
// n_splits분의 1로 줄이고, 부분 (m,l,acc)를 남겨 2차 커널이 병합한다.
// 산술: 키당 헤드별 독립 온라인 소프트맥스는 `_sel4`와 동일하지만 합산 순서가
// 분할 경계에서 달라진다 → 비트 동일 아님(수용 기준은 greedy 스트림 동일).
// 실측(2026-09-14, pp8192 tg): 142.5 -> 124.4 ms/스텝(-12.7%), diverse 스트림
// 완전 동일, 단문맥(pp24) 무회귀. 기본 ON이고 LLM170_QSA_SPLIT=0이 비트동일 복귀.
// 부분 버퍼는 [head][split][lane][m,l,acc0..7] — acc가 레인별이라 레인 축이
// 없으면 32레인이 같은 슬롯을 덮어쓴다(첫 구현에서 이 오류로 발산했다).
extern "C" __global__ void q4_qsa_attn_sel4s(const float* q, const float* ck, const float* cv,
                                             const unsigned* sel_idx, const unsigned* sel_off,
                                             float* part, int n_splits, float kq_scale,
                                             int n_head, int n_kv, int hd) {
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int h0 = ((int)blockIdx.y * 2 + (wid & 1)) * 6;   // 묶음당 6헤드(게이트는 병합커널이 적용)
    int sp = (int)blockIdx.x * 4 + (wid >> 1);        // 블록당 4분할
    if (h0 + 5 >= n_head || hd != 256 || sp >= n_splits) return;
    int kvh = h0 / (n_head / n_kv);
    const int j0 = (int)sel_off[0];
    const int j1 = (int)sel_off[1];
    const int len = j1 - j0;
    const int per = (len + n_splits - 1) / n_splits;
    const int a0 = j0 + sp * per;
    const int a1 = (a0 + per < j1) ? a0 + per : j1;
    float qr[6][8], acc[6][8];
    #pragma unroll
    for (int g = 0; g < 6; g++) {
        const float* qh = q + (size_t)(h0 + g) * 2 * hd;
        #pragma unroll
        for (int t2 = 0; t2 < 8; t2++) {
            qr[g][t2] = qh[lane + 32 * t2];
            acc[g][t2] = 0.0f;
        }
    }
    float mrun[6], lrun[6];
    #pragma unroll
    for (int g = 0; g < 6; g++) { mrun[g] = Q4_NEG; lrun[g] = 0.0f; }
    const unsigned long long FULL = 0xffffffffffffffffull;
    for (int j = a0; j < a1; j++) {
        const int p = (int)sel_idx[j];
        const float* k = ck + (size_t)p * n_kv * hd + kvh * hd;
        const float* v = cv + (size_t)p * n_kv * hd + kvh * hd;
        float kr[8], vr[8];
        #pragma unroll
        for (int t2 = 0; t2 < 8; t2++) { kr[t2] = k[lane + 32 * t2]; vr[t2] = v[lane + 32 * t2]; }
        #pragma unroll
        for (int g = 0; g < 6; g++) {
            float a = 0.0f;
            #pragma unroll
            for (int t2 = 0; t2 < 8; t2++) a += qr[g][t2] * kr[t2];
            for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(FULL, a, off);
            float s = a * kq_scale;
            if (s > mrun[g]) {
                float corr = (mrun[g] <= Q4_NEG) ? 0.0f : exp_cr(mrun[g] - s);
                #pragma unroll
                for (int t2 = 0; t2 < 8; t2++) acc[g][t2] *= corr;
                lrun[g] *= corr;
                mrun[g] = s;
            }
            float e = exp_cr(s - mrun[g]);
            lrun[g] += e;
            #pragma unroll
            for (int t2 = 0; t2 < 8; t2++) acc[g][t2] += e * vr[t2];
        }
    }
    // 부분 결과: [head][split][lane][ m, l, acc0..7 ] — acc는 레인별이므로
    // 레인 축이 반드시 필요하다(없으면 32레인이 같은 슬롯을 덮어쓴다).
    #pragma unroll
    for (int g = 0; g < 6; g++) {
        float* pr = part + (((size_t)(h0 + g) * n_splits + sp) * 32 + lane) * 10;
        pr[0] = mrun[g];
        pr[1] = lrun[g];
        #pragma unroll
        for (int t2 = 0; t2 < 8; t2++) pr[2 + t2] = acc[g][t2];
    }
}

// 분할 병합 — 헤드별로 n_splits개의 (m,l,acc)를 flash 규약으로 합치고 게이트를
// 적용해 출력한다. 헤드당 32레인(=hd 256을 8개씩)이 한 워프.
extern "C" __global__ void q4_qsa_attn_sel4s_merge(const float* part, const float* q,
                                                   float* out, int n_splits,
                                                   int n_head, int hd) {
    int h = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (h >= n_head || hd != 256) return;
    float m = Q4_NEG, l = 0.0f, acc[8];
    #pragma unroll
    for (int t2 = 0; t2 < 8; t2++) acc[t2] = 0.0f;
    for (int s = 0; s < n_splits; s++) {
        const float* pr = part + (((size_t)h * n_splits + s) * 32 + lane) * 10;
        float ms = pr[0];
        if (ms > m) {
            float corr = (m <= Q4_NEG) ? 0.0f : exp_cr(m - ms);
            #pragma unroll
            for (int t2 = 0; t2 < 8; t2++) acc[t2] *= corr;
            l *= corr;
            m = ms;
        }
        float e = exp_cr(ms - m);
        l += pr[1] * e;
        #pragma unroll
        for (int t2 = 0; t2 < 8; t2++) acc[t2] += pr[2 + t2] * e;
    }
    const float* gate = q + (size_t)h * 2 * hd + hd;
    #pragma unroll
    for (int t2 = 0; t2 < 8; t2++) {
        float o = (l > 0.0f) ? acc[t2] / l : 0.0f;
        float g = 1.0f / (1.0f + exp_cr(-gate[lane + 32 * t2]));
        out[(size_t)h * hd + lane + 32 * t2] = o * g;
    }
}

// ─── plans/72: Flash-Next 디코드 shared expert 융합 (8런치→2런치) ───
// q4_shexp_gu: gate=W_g×x, up=W_u×x, h=silu(gate)*up — Q8_0 가중치 f32 입력.
// q4_shexp_da: out=W_d×h, mout += s·out — W_d는 Q8_0, s는 sigmoid 스칼라.
// 디코드 t=1 전용. shared expert가 층당 8런치(19.4ms/step)의 지배 항이었다.
extern "C" __global__ void __launch_bounds__(256) q4_shexp_gu(
        const float* __restrict__ x,     // [n_in] f32 입력
        const unsigned char* __restrict__ wg, // Q8_0 [n_hidden][n_in] gate
        const unsigned char* __restrict__ wu, // Q8_0 [n_hidden][n_in] up
        float* __restrict__ h,           // [n_hidden] f32 출력 silu(gate)*up
        int n_in, int n_hidden) {
    // Q8_0 블록: 32원소당 2B f16 scale + 32B quant = 34B
    // 각 블록이 n_hidden 행 중 일부를 담당 — warp당 1행, 워프 내 lane분산 dot.
    const int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    if (row >= n_hidden) return;
    const int lane = threadIdx.x & 31;
    const int nblk = n_in / 32;

    float acc_g = 0.0f, acc_u = 0.0f;
    const unsigned char* wg_row = wg + (size_t)row * nblk * 34;
    const unsigned char* wu_row = wu + (size_t)row * nblk * 34;

    for (int blk = lane; blk < nblk; blk += 32) {
        const unsigned char* bg = wg_row + blk * 34;
        const unsigned char* bu = wu_row + blk * 34;
        // f16 scale (처음 2바이트)
        const half2 dg2 = *(const half2*)bg;
        const half2 du2 = *(const half2*)bu;
        const float dg = __low2float(dg2);
        const float du = __low2float(du2);
        // 32개 int8 quant (2바이트 뒤)
        const char* qg = (const char*)(bg + 2);
        const char* qu = (const char*)(bu + 2);
        float sg = 0.0f, su = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            const float xv = x[blk * 32 + j];
            sg += (float)qg[j] * xv;
            su += (float)qu[j] * xv;
        }
        acc_g += dg * sg;
        acc_u += du * su;
    }
    // warp reduce
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc_g += __shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, acc_g, off);
        acc_u += __shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, acc_u, off);
    }
    if (lane == 0) {
        // silu(gate) * up
        const float gate = acc_g;
        const float up = acc_u;
        h[row] = gate / (1.0f + exp_cr(-gate)) * up;
    }
}

extern "C" __global__ void __launch_bounds__(256) q4_shexp_da(
        const float* __restrict__ h,     // [n_hidden] f32 silu(gate)*up
        const unsigned char* __restrict__ wd, // Q8_0 [n_in][n_hidden] down
        const float* __restrict__ s_ptr, // [1] sigmoid 스칼라(디바이스)
        float* __restrict__ mout,        // [n_in] 누적 대상
        int n_in, int n_hidden) {
    const int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    if (row >= n_in) return;
    const int lane = threadIdx.x & 31;
    const int nblk = n_hidden / 32;
    const unsigned char* wd_row = wd + (size_t)row * nblk * 34;

    float acc = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        const unsigned char* bd = wd_row + blk * 34;
        const float dd = __low2float(*(const half2*)bd);
        const char* qd = (const char*)(bd + 2);
        float sd = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            sd += (float)qd[j] * h[blk * 32 + j];
        }
        acc += dd * sd;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, acc, off);
    }
    if (lane == 0) {
        mout[row] += (*s_ptr) * acc;
    }
}

// f32 무게 멀티토큰 GEMV(plans/74) — q4_gemm_f32_w 의 워프=행 구조에 토큰
// 누산. 무게 1회 독서로 t=2..8 라우터/PLE 투영(종전 q4_gemm_f32 t판은
// 17GB/s). 토큰별 산술 순서 = _w 판(레인 부분 float4 체인 → (x+y)+(z+w) →
// 워프 트리)과 동일.
extern "C" __global__ void __launch_bounds__(256) q4_gemm_f32_mt(
        const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out,
        int n_in, int n_out, int tt) {
    const int o = (int)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (o >= n_out) return;
    const float4* r4 = reinterpret_cast<const float4*>(w + (size_t)o * n_in);
    const int n4 = n_in >> 2;
    float4 a[8];
    #pragma unroll
    for (int ti = 0; ti < 8; ti++) { a[ti].x = 0; a[ti].y = 0; a[ti].z = 0; a[ti].w = 0; }
    for (int k = lane; k < n4; k += 32) {
        const float4 rv = r4[k];
        #pragma unroll
        for (int ti = 0; ti < 8; ti++) {
            if (ti >= tt) break;
            const float4 xv = reinterpret_cast<const float4*>(x + (size_t)ti * n_in)[k];
            a[ti].x += xv.x * rv.x;
            a[ti].y += xv.y * rv.y;
            a[ti].z += xv.z * rv.z;
            a[ti].w += xv.w * rv.w;
        }
    }
    #pragma unroll
    for (int ti = 0; ti < 8; ti++) {
        if (ti >= tt) break;
        float acc = (a[ti].x + a[ti].y) + (a[ti].z + a[ti].w);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_xor_sync(0xffffffffffffffffull, acc, off);
        if (lane == 0) out[(size_t)ti * n_out + o] = acc;
    }
}

// ─── token_embd(Q8_0) gather + hc 방송 (plans/111 W2) ───
// vk fn_emb_q8g*.comp 의 hip 미러. 블록 산술은 CPU deq_q8_0(deq.rs)과 비트 동일:
// y[j] = (int8 q) as f32 * bits_f16(d)(bits_f16 = half_to_f32 비트조립).
// f16 출력은 __float2half_rn(RTE) — CPU f32_to_f16_bits·GLSL packHalf2x16과
// 동일 반올림. 테이블은 백엔드가 mmap ptr 키로 1회 상주. 스레드=(ti, 블록 32원소).
extern "C" __global__ void q4_emb_q8g(const unsigned* ids, const unsigned char* tbl,
                                      float* out, int n, int t, int hc, int bpr) {
    const int gid = (int)(blockIdx.x * blockDim.x) + (int)threadIdx.x;
    if (gid >= t * bpr) return;
    const int ti = gid / bpr, blk = gid % bpr;
    const unsigned char* bp = tbl + ((size_t)ids[ti] * bpr + (size_t)blk) * 34;
    const float d = bits_f16((unsigned)bp[0] | ((unsigned)bp[1] << 8));
    const signed char* q = (const signed char*)(bp + 2);
    float y[32];
    #pragma unroll
    for (int j = 0; j < 32; j++) y[j] = (float)q[j] * d;
    for (int s = 0; s < hc; s++) {
        float* o = out + ((size_t)ti * hc + (size_t)s) * (size_t)n + (size_t)blk * 32;
        #pragma unroll
        for (int j = 0; j < 32; j++) o[j] = y[j];
    }
}

// f16 버스판(res_hc 쌍팩, plans/103) — blk 32정렬이라 (j,j+1) 쌍이 블록 내 머무름.
extern "C" __global__ void q4_emb_q8g_f16(const unsigned* ids, const unsigned char* tbl,
                                          unsigned* out, int n, int t, int hc, int bpr) {
    const int gid = (int)(blockIdx.x * blockDim.x) + (int)threadIdx.x;
    if (gid >= t * bpr) return;
    const int ti = gid / bpr, blk = gid % bpr;
    const unsigned char* bp = tbl + ((size_t)ids[ti] * bpr + (size_t)blk) * 34;
    const float d = bits_f16((unsigned)bp[0] | ((unsigned)bp[1] << 8));
    const signed char* q = (const signed char*)(bp + 2);
    float y[32];
    #pragma unroll
    for (int j = 0; j < 32; j++) y[j] = (float)q[j] * d;
    for (int s = 0; s < hc; s++) {
        // f32 원소 인덱스 → u32 쌍팩 워드 인덱스
        unsigned* o = out + ((((size_t)ti * hc + (size_t)s) * (size_t)n + (size_t)blk * 32) >> 1);
        #pragma unroll
        for (int j = 0; j < 32; j += 2) {
            const unsigned lo = (unsigned)__half_as_ushort(__float2half_rn(y[j]));
            const unsigned hi = (unsigned)__half_as_ushort(__float2half_rn(y[j + 1]));
            o[(unsigned)j >> 1] = lo | (hi << 16);
        }
    }
}

// ─── PLE 임베딩 테이블(IQ4_NL) gather (plans/111 W4c) ───
// vk fn_ple_gather.comp 의 hip 미러. IQ4_NL 블록: 32원소 18B = f16 d(2B) +
// qs(16B, 4bit LUT). y[j] = d·kv[qs[j]&0xF], y[16+j] = d·kv[qs[j]>>4] —
// CPU deq_iq4_nl(deq.rs)과 비트 동일(동일 스칼라 곱, kv 상수 = KVALUES_IQ4NL).
// 테이블은 백엔드가 mmap ptr 키로 1회 상주. 스레드=(row, 블록 32원소).
extern "C" __global__ void q4_ple_gather(const unsigned* rows, const unsigned char* tbl,
                                         float* emb, int hd, int nrows) {
    const int bpr = (hd + 31) >> 5;
    const int gid = (int)(blockIdx.x * blockDim.x) + (int)threadIdx.x;
    if (gid >= nrows * bpr) return;
    const int row = gid / bpr, blk = gid % bpr;
    const unsigned char* bp = tbl + ((size_t)rows[row] * bpr + (size_t)blk) * 18;
    const float d = bits_f16((unsigned)bp[0] | ((unsigned)bp[1] << 8));
    const unsigned char* qs = bp + 2;
    const float kv[16] = { -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
                           1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f };
    float* o = emb + (size_t)row * hd + (size_t)blk * 32;
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        o[j] = d * kv[qs[j] & 0xF];
        o[16 + j] = d * kv[qs[j] >> 4];
    }
}
