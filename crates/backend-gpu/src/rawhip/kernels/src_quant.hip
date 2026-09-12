// q6_K(우리 엔진 재배열 레이아웃) → f16 256값 전개 — v4 인덱싱 미러:
// d@208, sc@192+gr, ql=h*64+p2*16+((src&1)<<5)+kloc, qh=128+h*32+p2*16+kloc
// n2→(gr,kloc): sb=n2>>5, kc=(n2>>4)&1, gr=(sb*2+kc)&15, kloc=n2&15
extern "C" __global__ void dequant_q6k_f16(const unsigned char* __restrict__ w, __half* __restrict__ out,
                                           int blocks, int n_out) {
    int o = blockIdx.x, blk = blockIdx.y;
    if (o >= n_out) return;
    long bo = ((long)o * blocks + blk) * 210;
    const unsigned char* wb8 = w + bo;
    float d = __half2float(__ushort_as_half((unsigned short)(wb8[208] | (wb8[209] << 8))));
    int n2 = threadIdx.x;   // 0..255
    int sb = n2 >> 5, kc = (n2 >> 4) & 1, kloc = n2 & 15;
    int gr = (sb * 2 + kc) & 15;
    int h = gr >> 3, src = (gr & 7) >> 1, p2 = gr & 1;
    int ql_b = h * 64 + p2 * 16 + ((src & 1) << 5) + kloc;
    int qh_b = 128 + h * 32 + p2 * 16 + kloc;
    int nib = (wb8[ql_b] >> ((src < 2) ? 0 : 4)) & 0xF;
    int hi = (wb8[qh_b] >> (2 * src)) & 3;
    int sc = wb8[192 + gr]; sc = sc > 127 ? sc - 256 : sc;
    float v = d * (float)sc * ((float)(nib + (hi << 4)) - 32.0f);
    out[((long)o * blocks + blk) * 256 + n2] = __float2half(v);
}

// 우리 레이아웃 q6 → ggml 정준 재배열 (MMQ q6 해금, 부록46)
// 128스레드: l이 정준 l열 전담 — ql 2바이트+qh 1바이트를 통합 기입 (경합 없음).
// 스케일 인덱싱 동일(grp*8+quad*2+is), 위치만 192→194, d 208→0.
extern "C" __global__ void requant_q6k_canonical(const unsigned char* __restrict__ w,
                                                 unsigned char* __restrict__ out,
                                                 int blocks, int n_out) {
    int o = blockIdx.x, blk = blockIdx.y;
    if (o >= n_out) return;
    long bo = ((long)o * blocks + blk) * 210;
    const unsigned char* s = w + bo;
    unsigned char* d2 = out + bo;
    if (threadIdx.x == 0) {
        unsigned short dh = (unsigned short)(s[208] | (s[209] << 8));
        d2[0] = dh & 0xFF; d2[1] = dh >> 8;
        for (int i = 0; i < 16; i++) d2[194 + i] = s[192 + i];
        // ql/qh 나머지는 l 스레드들이 기입
    }
    int l = threadIdx.x;   // 0..127
    #pragma unroll
    for (int grp = 0; grp < 2; grp++) {
        int qb0 = 2 + grp * 64 + l;        // quads 0,2 바이트
        int qb1 = 2 + grp * 64 + 32 + l;   // quads 1,3 바이트
        int hb  = 2 + 128 + grp * 32 + l;  // qh 바이트
        unsigned b0 = 0, b1 = 0, hbb = 0;
        #pragma unroll
        for (int quad = 0; quad < 4; quad++) {
            int n2 = grp * 128 + quad * 32 + l;
            int sb = n2 >> 5, kc = (n2 >> 4) & 1, kloc = n2 & 15;
            int gr = (sb * 2 + kc) & 15;
            int h = gr >> 3, src2 = (gr & 7) >> 1, p2 = gr & 1;
            int ql_b = h * 64 + p2 * 16 + ((src2 & 1) << 5) + kloc;
            int qh_b = 128 + h * 32 + p2 * 16 + kloc;
            int nib = (s[ql_b] >> ((src2 < 2) ? 0 : 4)) & 0xF;
            int hi = (s[qh_b] >> (2 * src2)) & 3;
            int v6 = nib | (hi << 4);   // 0..63
            int lo4 = v6 & 0xF, hi2 = (v6 >> 4) & 3;
            if (quad == 0) b0 |= lo4;
            else if (quad == 1) b1 |= lo4;
            else if (quad == 2) b0 |= lo4 << 4;
            else b1 |= lo4 << 4;
            hbb |= hi2 << (2 * quad);
        }
        d2[qb0] = b0; d2[qb1] = b1; d2[hb] = hbb;
    }
}

extern "C" __global__ void quant_q8(const float* x, unsigned* xq, int n, int xq_w) {
    int nblk = n >> 5;
    int lb = blockIdx.x * blockDim.x + threadIdx.x;
    if (lb >= nblk) return; // 토큰 내 가드
    x += (size_t)blockIdx.y * n;
    xq += (size_t)blockIdx.y * xq_w;
    int nwords = n >> 2;
    int qs0 = 0, qs1 = 0;
    int base = lb << 5;
    // float4 벡터 로드 (2026-09-05: 스칼라 32로드가 지연 병목 — 79GB/s 측정).
    // 산술 동일열: amax는 fmaxf 결합律(비트불변), rounding/clamp 공식 원본 그대로.
    float4 v0 = *(const float4*)(x + base);
    float4 v1 = *(const float4*)(x + base + 4);
    float4 v2 = *(const float4*)(x + base + 8);
    float4 v3 = *(const float4*)(x + base + 12);
    float4 v4 = *(const float4*)(x + base + 16);
    float4 v5 = *(const float4*)(x + base + 20);
    float4 v6 = *(const float4*)(x + base + 24);
    float4 v7 = *(const float4*)(x + base + 28);
    const float4 vs[8] = { v0, v1, v2, v3, v4, v5, v6, v7 };
    float amax = 0.0f;
    #pragma unroll
    for (int w = 0; w < 8; w++)
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(vs[w].x), fabsf(vs[w].y)),
                                 fmaxf(fabsf(vs[w].z), fabsf(vs[w].w))));
    float d = amax / 127.0f;
    float id = d != 0.0f ? 1.0f / d : 0.0f;
    #pragma unroll
    for (int wi = 0; wi < 8; wi++) {
        float xv0 = vs[wi].x * id;
        float xv1 = vs[wi].y * id;
        float xv2 = vs[wi].z * id;
        float xv3 = vs[wi].w * id;
        float r0 = xv0 >= 0.0f ? (float)(int)(xv0 + 0.5f) : -((float)(int)(0.5f - xv0));
        float r1 = xv1 >= 0.0f ? (float)(int)(xv1 + 0.5f) : -((float)(int)(0.5f - xv1));
        float r2 = xv2 >= 0.0f ? (float)(int)(xv2 + 0.5f) : -((float)(int)(0.5f - xv2));
        float r3 = xv3 >= 0.0f ? (float)(int)(xv3 + 0.5f) : -((float)(int)(0.5f - xv3));
        float c0 = r0 > 127.0f ? 127.0f : (r0 < -127.0f ? -127.0f : r0);
        float c1 = r1 > 127.0f ? 127.0f : (r1 < -127.0f ? -127.0f : r1);
        float c2 = r2 > 127.0f ? 127.0f : (r2 < -127.0f ? -127.0f : r2);
        float c3 = r3 > 127.0f ? 127.0f : (r3 < -127.0f ? -127.0f : r3);
        unsigned word = ((((unsigned)(int)c0) & 0xFFu))
                      | ((((unsigned)(int)c1) & 0xFFu) << 8)
                      | ((((unsigned)(int)c2) & 0xFFu) << 16)
                      | ((((unsigned)(int)c3) & 0xFFu) << 24);
        xq[lb * 8 + wi] = word;
        if (wi < 4) qs0 = dot4(0x01010101u, word, qs0);
        else qs1 = dot4(0x01010101u, word, qs1);
    }
    xq[nwords + lb] = __float_as_uint(d); // d 비트 편승 (u32 저장 경로)
    xq[nwords + nblk + 2 * lb] = (unsigned)qs0;      // q16 테이블 (하위 16원소합)
    xq[nwords + nblk + 2 * lb + 1] = (unsigned)qs1;
}

// rms+quant 융합 (t=1) — rms_part/rms_finish/quant_q8 3런치 → 1.
// 산술은 셋을 그대로 이어붙인 것 (세그먼트 f32합 → f64 순서합 →
// x·inv·w 레지스터 통과 → 32블록 quant). chunk=(n+31)>>5가 32배수
// (n%1024==0)일 때만 사용 — 스레드 청크=quant 블록 경계 정렬.
extern "C" __global__ void rmsq(const float* x, const float* w, unsigned* xq,
                                float eps, int n) {
    int u = threadIdx.x;
    int chunk = (n + 31) >> 5;
    __shared__ double part[32];
    __shared__ float inv_s;
    if (u < 32) {
        int lo = u * chunk;
        int hi = min(lo + chunk, n);
        float acc = 0.0f;
        if (lo < n) {
            // 로드 4개 선행 후 가산은 원소 순서 그대로 — 합 순서 불변(계약),
            // 로드 지연만 숨긴다 (직렬 load→add→load 체인의 3cycle/원소 제거).
            int i = lo;
            for (; i + 3 < hi; i += 4) {
                float a0 = x[i], a1 = x[i + 1], a2 = x[i + 2], a3 = x[i + 3];
                acc += a0 * a0;
                acc += a1 * a1;
                acc += a2 * a2;
                acc += a3 * a3;
            }
            for (; i < hi; i++) { float dv = x[i]; acc += dv * dv; }
        }
        part[u] = (double)acc;
    }
    __syncthreads();
    if (u == 0) {
        double sum = 0.0;
        for (int u2 = 0; u2 < 32; u2++) {
            if (u2 * chunk < n) sum += part[u2];
        }
        float scale32 = (float)sqrt(sum / (double)n + (double)eps);
        inv_s = 1.0f / scale32;
    }
    __syncthreads();
    float inv = inv_s;
    int nwords = n >> 2;
    int nblk = n >> 5;
    // 스레드당 1개 32-블록 — base는 이전 (u*5+blk) 매핑과 동일 → 산술 불변.
    int b = u;
    if (b < nblk) {
        {
            int base = b << 5;
            // float4 로드 + 트리 amax — fmax는 순서 무관(결합·교환)이고 양자화는
            // 원소별이라 산술 불변. 종전 32단 의존 체인과 96개 스칼라 로드를 제거.
            float4 xv4[8];
            {
                const float4* x4 = (const float4*)(x + base);
                const float4* w4 = (const float4*)(w + base);
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    float4 a = x4[i], c = w4[i];
                    xv4[i] = make_float4(a.x * inv * c.x, a.y * inv * c.y, a.z * inv * c.z, a.w * inv * c.w);
                }
            }
            float amax = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                float4 v = xv4[i];
                amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
            }
            float d = amax / 127.0f;
            float id = d != 0.0f ? 1.0f / d : 0.0f;
            int qs0 = 0, qs1 = 0;
            #pragma unroll
            for (int wi = 0; wi < 8; wi++) {
                unsigned word = 0u;
                float4 vv4 = xv4[wi];
                float vv[4] = { vv4.x, vv4.y, vv4.z, vv4.w };
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    float v = vv[k] * id;
                    float r = v >= 0.0f ? (float)(int)(v + 0.5f)
                                        : -((float)(int)(0.5f - v));
                    float c = r > 127.0f ? 127.0f : (r < -127.0f ? -127.0f : r);
                    word |= (((unsigned)(int)c) & 0xFFu) << (k * 8);
                }
                xq[(base >> 5) * 8 + wi] = word;
                if (wi < 4) qs0 = dot4(0x01010101u, word, qs0);
                else qs1 = dot4(0x01010101u, word, qs1);
            }
            xq[nwords + (base >> 5)] = __float_as_uint(d);
            xq[nwords + nblk + 2 * (base >> 5)] = (unsigned)qs0;
            xq[nwords + nblk + 2 * (base >> 5) + 1] = (unsigned)qs1;
        }
    }
}

