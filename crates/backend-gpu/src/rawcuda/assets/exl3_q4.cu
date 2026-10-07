// ── Q4(GGUF Q4_K / Unsloth UD-Q4_K_XL) MMQ CUDA 모듈 (plans/124 G8, 2026-10-04) ──
//
// [계약 원천 — EXL3와 다름] 이 모듈의 수치 진실은 ggml 미러
// crates/core/src/quant/(deq.rs·q8.rs·lane.rs)이다 — EXL3 트렐리스 경로와
// 무관. hip 커널(src_q4.hip·src_gemm.hip·src_quant.hip)은 형상(런치 계약)과
// 비트 조작식의 원본 대조원일 뿐, 부동소수 적산·반올림은 core가 기준
// (plans/124 §1.1·§6 — hip에도 버그가 있다).
//
// [UD-Q4_K_XL 형식 판독 — core 대조로 확정] UD-Q4_K_XL(Unsloth Dynamic)은
// 양자화 시점 변형일 뿐 GGUF에는 표준 Q4_K 블록이 저장된다:
// - crates/gguf/src/types.rs: Q4K = 12, block_info = (256원소, 144B)
// - crates/core/src/quant/deq.rs dequant_row: GgmlType::Q4K → deq_q4_k
//   (144B = d(2) + dmin(2) + scales(12) + qs(128), 8개 6비트 scale/min 쌍)
// - 실파일 헤더 실측(2026-10-04, Qwen3.8-27B-UD-Q4_K_XL.gguf): ty=12 텐서가
//   144B 블록 배치로 열거됨. 따라서 디양자화는 표준 Q4_K 산술을 그대로
//   따른다(별도 XL 특수 경로 없음).
//
// [커널별 산술 계약 — 워크트리 기준 줄번호, 2026-10-04]
// 1) q4_quant_q8   : rawhip/kernels/src_quant.hip quant_q8(L13-66) 직이식 —
//                    core q8.rs quantize_row_q8_ref(L11-34) 미러(amax/127 →
//                    1/d → round-half-away → clamp ±127 → 4바이트 팩 +
//                    d 비트 편승 + q16 상/하반 바이트합 테이블).
//                    xq 행 배치(워드): [n/4 팩워드][n/32 f32 스케일]
//                    [2·n/32 q16] — q4acc/mod.rs xq_words(L284-286) 미러.
// 2) q4_dequant_q4k: core deq.rs deq_q4_k(L53-70)·scale_min_k4(L40-50)과
//                    **비트동일**(d1=d·sc, mm1=dmin·m, y=d1·q−mm1 — 각 f32
//                    곱 1회·감산 1회). f16→f32는 deq.rs half_to_f32(L13-37)과
//                    비트동일(bits_f16 — 서브노멀 frac·2^-24 경계값 동일).
// 3) q4_gemv_q4k   : rawhip/kernels/src_gemm.hip gemm_q4k(L1251-1313) 직이식 —
//                    **분할형**(acc += yd·(d·sc)·isum; acc −= yd·(dm·m)·qsum)
//                    64레인 스트라이드 + f64 트리64 환원 = core lane.rs
//                    dot_row_w4a8_q4k_lane_parts(L166-191) + tree64(L373-385)
//                    미러 → 호스트 레인 미러와 비트동일 기대(감산 결합을
//                    곱 체인 2개로 분리 — FMA 수축 면역, G5 노선).
// 4) q4_gemm_q4k_m : rawhip/kernels/src_q4.hip q4_gemm_q4k_m(16출력×16행 타일)
//                    형상 직이식 — 누산 구조만 core q8.rs dot_q4k_q8(L76-102)·
//                    lane.rs dot_row_w4a8(L11-27)의 **블록 그룹핑**으로:
//                    256원소 블록 국소합 bsum(it 0..3, 저니블→고니블)을
//                    만들고 acc += bsum(= core acc += dot_q4k_q8(blk)).
//                    원판 hip은 서브블록 항을 행 acc에 평폴드로 직접
//                    적산해 core 미러와 ~1e-7 계열 그룹핑 편차가 있다
//                    (plans/124 §6: core가 진실 — 편차는 프로브가 수치
//                    기록, G8 원장). 원판 part(미사용 더미)도 제거.
//
// [빌드 계약] 이 소스는 **-fmad=false** 로 컴파일한다(build_cuda.bat 별도
// 블록): d1·q−mm1, yd·isum 류의 곱-가산 쌍이 FMA로 수축되면 단일 반올림이
// 호스트 미러(연산별 반올림)와 어긋난다(G5-G7 원장 — 비트동일 미러 계약의
// 전제). 정수 경로(dot4/팩)는 fmad 무관.
//
// [CMP 170HX 최적화 타깃 — sm_80/GA100 70SM·HBM2e ~1.5TB/s, plans/124 §0]
// - gemv(t=1): grid (1, n_out)·블록 64(2와프) — 원본 hip 발사 계약
//   (rawhip ctx/gemm.rs gemv_q8_out: gy=n_out, 64스레드). 행당 가중 스트림
//   n_in·144/256B(27B attn_qkv 2880B), xq(행 7040B)는 전 행이 공유 →
//   L2 상주(7KB ≪ 40MB L2)라 HBM 트래픽 = 가중 순수 스트리밍 — 대역폭
//   포화 설계. 27B n_out=10240 블록·2와프 → GA100 블록 32/SM 상한으로
//   64와프/SM 풀점유(레지스터 소형 커널, cuobjdump 증거는 커밋 본문).
//   개발기(4070 sm_89)는 정합 검증 호스트일 뿐 — 타이밍 판단 근거 아님
//   (§0 계약, CMP 170HX 실측은 장비 도착 후).
// - gemm(16×16 타일, 256스레드=8와프): src_q4.hip MMQ 계약 형상 유지.
//   GA100 무공유메모리·dp4a INT 파이프 — cuobjdump(sm_80) REG:48 →
//   48×256=12,288레지/블록, 65,536/12,288 = **5블록/SM(40와프, 62.5%)**
//   로 레지스터 한계(와프 한계 8블록보다 먼저 바인딩) — k-타일당
//   레지스터 여유는 k-분할(ksplit) 확장 시 재판정.
//   MoE 2048×512·T=32 그리드는 32×2=64블록(< 70SM) — 소형 런치 레이턴시
//   도미넌트; k-분할(src_q4.hip q4_gemm_q4k_ge_ids의 ksplit 패턴)이 점유를
//   끌어올리는 후속 레버지만 G8은 모듈 단위 정합 우선(§0: 모듈별 속도·
//   정합 — 형상은 q4acc 측정 원장이 지배). 자원 사용량은
//   cuobjdump --dump-resource-usage(sm_80) 증거를 커밋 본문에 기록.
// - dequant: grid (행, 슈퍼블록)·256스레드 — 원소별 스트리밍(블록당
//   144B 판독·1KB 코얼레스드 쓰기, sc 12B·qs 128B 전부 사용 — 낭비 0).

// ── 공용 헬퍼(src_common.hip L4-22 직이식 — core half_to_f32 비트동일) ──
__device__ __forceinline__ float bits_f16(unsigned h)
{
    unsigned sign = (h & 0x8000u) << 16;
    unsigned exp = (h >> 10) & 0x1Fu;
    unsigned frac = h & 0x3FFu;
    if (exp == 0) {
        if (frac == 0) return __int_as_float(sign);
        // 서브노멀: frac·2^-24 는 f32에서 정확(core deq.rs 규격화 루프와 동일값)
        float v = (float)frac * (1.0f / 16777216.0f);
        return sign ? -v : v;
    }
    if (exp == 31) return __int_as_float(sign | 0x7F800000u | (frac << 13));
    return __int_as_float(sign | ((exp + 112) << 23) | (frac << 13));
}

// i8×4 바이트별 정수 내적 — hip __ockl_sdot4 상당(llama.cpp dp4a).
// 정수라 내부 순서 무관 — 미러와 동일 isum 보장(비트계약).
__device__ __forceinline__ int dot4(unsigned a, unsigned b, int c)
{
    return __dp4a((int)a, (int)b, c);
}

// ── 활성 q8 양자화(src_quant.hip quant_q8 직이식) ──
// x[f32 n] → xq[행 xq_w 워드]: [n/4 팩워드][n/32 d 비트][2·n/32 q16 합].
// 산술 = core q8.rs quantize_row_q8_ref 미러(amax/127 f32 나눗셈 → id=1/d →
// round-half-away → ±127 clamp). blockIdx.y = 행(토큰). 비트 조작식
// (팩·q16 적재)은 원본 그대로 베낀다(plans/124 §2 원칙 — 포맷 정의).
extern "C" __global__ void q4_quant_q8(const float* __restrict__ x,
                                       unsigned* __restrict__ xq,
                                       int n, int xq_w)
{
    int nblk = n >> 5;
    int lb = blockIdx.x * blockDim.x + threadIdx.x;
    if (lb >= nblk) return; // 토큰 내 가드
    x += (size_t)blockIdx.y * n;
    xq += (size_t)blockIdx.y * xq_w;
    int nwords = n >> 2;
    int qs0 = 0, qs1 = 0;
    int base = lb << 5;
    // float4 벡터 로드 — 산술 동일열(원본 주석: 스칼라 32로드가 지연 병목).
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
        // round half away from zero — Rust f32::round 미러(quantize_row_q8_ref)
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
    xq[nwords + nblk + 2 * lb] = (unsigned)qs0;      // q16 (하위 16원소합)
    xq[nwords + nblk + 2 * lb + 1] = (unsigned)qs1;
}

// ── Q4_K 블록 디양자화(core deq.rs deq_q4_k 비트동일) ──
// 144B 블록 → f32 256값. 요소 매핑: e=it·64+l(l<32 저니블·l>=32 고니블),
// 스케일 쌍 j = 2·it + (l>=32) — scale_min_k4(deq.rs L40-50) 원식.
// v = d1·q − mm1 (d1=d·sc, mm1=dmin·m — f32 곱 2회·감산 1회, FMA 수축
// 금지 = -fmad=false). grid (행, 슈퍼블록)·256스레드(=블록 원소수).
extern "C" __global__ void q4_dequant_q4k(const unsigned char* __restrict__ w,
                                          float* __restrict__ out,
                                          int nsuper, int n_out)
{
    int o = blockIdx.x, blk = blockIdx.y;
    if (o >= n_out) return;
    const unsigned char* wb = w + ((size_t)o * nsuper + blk) * 144;
    float d = bits_f16((unsigned)wb[0] | ((unsigned)wb[1] << 8));
    float dmin = bits_f16((unsigned)wb[2] | ((unsigned)wb[3] << 8));
    const unsigned char* sc = wb + 4;
    const unsigned char* qs = wb + 16;
    int e = threadIdx.x;              // 0..255
    int it = e >> 6;
    int l = e & 63;
    int j = (l < 32) ? (2 * it) : (2 * it + 1);
    unsigned s, mn;
    if (j < 4) { s = sc[j] & 63u; mn = sc[j + 4] & 63u; }
    else { s = (sc[j + 4] & 0xFu) | ((unsigned)(sc[j - 4] >> 6) << 4);
           mn = (sc[j + 4] >> 4) | ((unsigned)(sc[j] >> 6) << 4); }
    unsigned qbyte = qs[it * 32 + (l & 31)];
    unsigned q = (l < 32) ? (qbyte & 0xFu) : (qbyte >> 4);
    float d1 = d * (float)s;        // deq_q4_k: d1 = d * sc1 as f32
    float mm1 = dmin * (float)mn;   // deq_q4_k: mm1 = min * m1 as f32
    out[((size_t)o * nsuper + blk) * 256 + e] = d1 * (float)q - mm1;
}

// ── Q4_K MMQ GEMV(t=1) — src_gemm.hip gemm_q4k(L1251-1313) 직이식 ──
// 64레인/출력행, 레인 l: 서브블록 sb = l, l+64, … 스트라이드. 서브블록당
// **분할형** 2항(acc += yd·(d·sc)·isum / acc −= yd·(dm·m)·qsum) — core lane.rs
// dot_row_w4a8_q4k_lane_parts 미러(레인 f32 누산) → f64 트리64 환원
// (sh32 상하 가산 + 셔플 16..1 = lane.rs tree64와 동일 순서).
// 스케일 3워드 복호(s0/s4/s8 u16 조합 → 8 서브스케일)은 원본 비트열
// 그대로(scale_min_k4와 값 동일 — src_gemv4.hip gemm_q4k4 주석 계약).
// grid (토큰=1, n_out)·64스레드. 원판의 part 더미 인자는 제거.
extern "C" __global__ void q4_gemv_q4k(const unsigned* __restrict__ xq,
                                       const unsigned* __restrict__ w,
                                       float* __restrict__ out,
                                       int n_in, int n_out, int xq_w)
{
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
    // 트리 환원 (미러 tree64와 동일 순서) — 셔플 폭 32 제한(RCA 2026-09-03):
    // 상/하 절반 공유메모리 교환 후 워프 트리.
    __shared__ double sh32[32];
    double accd = (double)acc;
    if (l >= 32) sh32[l - 32] = accd;
    __syncthreads();
    if (l < 32) {
        accd += sh32[l];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            accd += __shfl_down_sync(0xffffffffu, accd, off);
        if (l == 0) out[o] = (float)accd;
    }
}

// ── Q4_K MMQ GEMM 타일 — src_q4.hip q4_gemm_q4k_m 형상 + core 그룹핑 ──
// 16출력×16행/블록(256스레드), 스레드=(출력,행) 한 쌍을 k 직렬 누산.
// 산술 = core q8.rs dot_q4k_q8(블록 내적) + lane.rs dot_row_w4a8(블록 순차
// f32) 미러 — 슈퍼블록 국소합 bsum 후 acc += bsum(core와 비트동일,
// -fmad=false). 원판 hip의 평폴드 누산과는 그룹핑이 다르다(§4 각주 —
// 편차는 프로브가 수치 기록).
// Σy(qs1/qs2)는 xq q16 테이블(비트 동일 정수 합 — plans/116-3)에서.
// grid (ceil(n_out/16), ceil(t/16))·256스레드. 원판 part 더미 인자 제거.
extern "C" __global__ void q4_gemm_q4k_m(const unsigned* __restrict__ xq,
                                         const unsigned char* __restrict__ w,
                                         float* __restrict__ out,
                                         int n_in, int n_out, int xq_w, int t)
{
    int ox = threadIdx.x & 15;
    int rx = threadIdx.x >> 4;
    int o = (int)blockIdx.x * 16 + ox;
    int r = (int)blockIdx.y * 16 + rx;
    if (o >= n_out || r >= t) return;
    int n_super = n_in >> 8;
    const unsigned char* wr = (const unsigned char*)w + (size_t)o * (n_super * 144);
    const unsigned* xr = xq + (size_t)r * xq_w;
    const unsigned xw_base = (unsigned)(n_in >> 2);
    float acc = 0.0f;
    for (int sIdx = 0; sIdx < n_super; sIdx++) {
        const unsigned char* blk = wr + sIdx * 144;
        float d = bits_f16((unsigned)blk[0] | ((unsigned)blk[1] << 8));
        float dmin = bits_f16((unsigned)blk[2] | ((unsigned)blk[3] << 8));
        const unsigned char* sc = blk + 4;
        const unsigned char* qs = blk + 16;
        // core dot_q4k_q8의 블록 국소합(sum) — acc += dot_q4k_q8(blk) 구조.
        float bsum = 0.0f;
        #pragma unroll
        for (int it = 0; it < 4; it++) {
            unsigned q[8];
            // 16B 벡터 로드 — 블록(144B)·it*32는 16의 배수라 정렬 보장,
            // 리틀엔디언 워드 4개는 바이트 조립과 값 동일(수치 불변).
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
            int il1 = 0, il2 = 0;
            #pragma unroll
            for (int m = 0; m < 8; m++) {
                il1 += dot4(q[m] & 0x0F0F0F0Fu, x1[m], 0);
                il2 += dot4((q[m] >> 4) & 0x0F0F0F0Fu, x2[m], 0);
            }
            const unsigned* q16 = xr + xw_base + (unsigned)(n_in >> 5);
            int lb1 = (sIdx << 3) + (it << 1);
            int qs1 = (int)q16[2 * lb1] + (int)q16[2 * lb1 + 1];
            int qs2 = (int)q16[2 * lb1 + 2] + (int)q16[2 * lb1 + 3];
            float yd1 = __uint_as_float(xr[xw_base + sIdx * 8 + 2 * it]);
            float yd2 = __uint_as_float(xr[xw_base + sIdx * 8 + 2 * it + 1]);
            bsum += yd1 * (d * (float)s1 * (float)il1 - dmin * (float)m1 * (float)qs1);
            bsum += yd2 * (d * (float)s2 * (float)il2 - dmin * (float)m2 * (float)qs2);
        }
        acc += bsum;
    }
    out[(size_t)r * n_out + o] = acc;
}
// 마커 q4c
