// ── DeepSeek-V4 MoE 스테이지 CUDA 커널 (plans/130 B3, 2026-10-05) ──
// 계약 소스는 이 .cu — 빌드 자산 ds4_moe.fatbin은 scripts/build_cuda.bat이
// 같은 디렉터리에 쓰고 함께 커밋된다(rawhip co/*.co 미러: 소스·자산 동반).
//
// [산술 계약 — 원천: crates/core/src/deepseek4/stages/moe.rs (CPU 황금 기준)]
// - 게이트: s_e = √softplus(Σ_i x[i]·gw[i][e]) — f32 무양자화 gemm
//   (moe.rs gate_scores L42-50 + deepseek4/ops.rs gemm_nt L58-75: k 오름차순
//   순차 누산, 곱·가산 각 1회 반올림) 후 sqrtsoftplus(ops.rs L355-358,
//   softplus = core ops.rs L138-140: x>20 → x, else ln_cr(exp_cr(x))+1 경로).
// - 해시 라우팅(L0-2): sel = tid2eid[tid·k + j] 행 룩업 — top-k/bias 없음
//   (moe.rs route_hash L81-85 + moe_forward L152-159). 가중치는 여전히
//   계산: w_j = s[sel_j]/Σ·route_scale(Σ는 행 순서 순차 — moe.rs route
//   L67-79: raw.iter().sum() 선택(행) 순서, w = s/sum×1.5 나눗셈·곱 각 1회).
// - noaux_tc 라우팅(L3+): sel = topk(s + e_score_correction_bias)(bias는
//   선택에만), w = gather(s, sel) — bias 미포함(moe.rs route_routed L87-96).
//   top-k 동점: 값 내림차순·동점 낮은 인덱스 우선(안정 정렬 — topk_stable
//   L53-65 모듈 계약). 커널은 "상향 스캔·초과 갱신(엄격 >)" 반복 선택으로
//   동치 구현(내림차 안정 정렬의 위너 = 남은 것 중 최댓값의 최저 id).
// - 전문가 FFN(moe.rs expert_ffn L98-131): FP8-sim(128블록) 입력 →
//   w1(gate)/w3(up) → 각 bf16 경계 → 비대칭 클램프 SwiGLU(swiglu_limit
//   ops.rs L361-364: gate-proj max L, up-proj [-L,L]) ×라우팅 가중치 →
//   bf16 → FP8-sim(128블록) → w2 → bf16. 가중치는 EXL3 trellis 디양자화
//   f32(호스트 loader.rs linear L203-247 — 커널은 f32 k-major 소비).
// - 결합(moe.rs moe_forward L133-216): 토큰별 전문가 id 오름차순 누산
//   (BTreeMap by_expert 계약 — (ti,e) 페어 토큰-메이저·토큰 내 e 오름차) 후
//   공유 전문가 무스케일 가산, 최종 bf16 경계.
//
// [트랜센던트 계약 — 치환이 아닌 미러]
// core exp_cr(ops.rs L52-89)·ln_cr(L91-121)은 f64 FMA 호너 DAG — fma()의
// IEEE 단일 반올림·rint(ties-even)·비트 재구성이 전부 CUDA로 이식된다
// (exl3_fn_moe.cu fn_moe_exp_cr과 동일 노선). sqrtf·f32 나눗셈은 nvcc
// 기본 IEEE 정확 반올림. e4m3 RNE 왕복·bf16 RNE·pow2_ceil은 순수 정수
// 비트 경로(이식 무결).
//
// [빌드 계약] -fmad=false 별도 블록(scripts/build_cuda.bat) — acc += x[i]*w
// 및 h = silu(g)*u*w의 곱·가산 쌍이 FMA 수축되면 호스트 오라클(core
// gemm_nt 순차 2회 반올림 미러)과 비트가 어긋난다. 정합 판정은 값 maxdiff
// (argmax 금지 — 단 라우팅 이산 선택은 exact-match 별도 판정, 과제 계약).
//
// [CMP 170HX(sm_80) 설계 근거 — plans/124 §0 계승]
// - ds4_gate/ds4_gemv_f32_ptr: 스레드당 출력 1개·내부 순차 누산(비트동일
//   계약이 우선 — FNE fn_moe_gemm_f16_ptr 동일 등급). k-major f32에서
//   인접 스레드는 n_out 스트라이드로 읽어 완전 coalesce가 아니나, 전문가
//   GEMV는 활성이 전문가당 1행(재사용 없음 — 토큰-메이저 페어 계약)이라
//   HBM2e 관점 W 스트리밍 본드가 지배적. sm_80 실측 후 mma 재판정.
// - ds4_route_*: 토큰당 단일 스레드 순차(Σ 순서·선택 순서 core 계약) —
//   256 스코어·top-6 스캔/토큰으로 ALU 미미. 블록 32(활성 1레인).
// - ds4_fp8_sim: 블록 128(128원소 청크 1블록 — amax 트리 환원, max는
//   순서 무관 정확). ds4_swiglu/ds4_bf16_round: 원소별 스트리밍.
// - ds4_combine: 그리드 (t,1)·블록 256 — FNE fn_moe_combine 동일 등급,
//   페어 루프는 레지스터 누산(순차 — eid 오름차 순서 계약).
//
// [NaN 도메인] 게이트 로짓·전문가 활성은 finite 계약(프로브 픽스처도
// finite). f32_to_e4m3의 448 초과 분은 방어적 NaN(참조와 동일).

#include <math.h>

// ── core exp_cr 직이식(core ops.rs L52-89 — 비트동일 미러) ──
// f64 FMA 호너 13차: k=rint(x·log2e) → r=x−k·ln2(hi/lo) → 테일러 →
// 2^k 비트 재구성. 연산 순서·상수 변경 금지(호스트 오라클과 리터럴까지
// 동일해야 비트동일 — exl3_fn_moe.cu fn_moe_exp_cr 동일 DAG).
__device__ __forceinline__ float ds4_exp_cr(float x)
{
    double xd = (double)x;
    if (xd > 88.72) return INFINITY;
    if (xd < -103.97) return 0.0f;
    const double LN2_HI = 6.9314718036912382e-01;
    const double LN2_LO = 1.9082149292705877e-10;
    const double INV_LN2 = 1.4426950408889634; // log2(e) — std 상수 비트동일
    double kd = rint(xd * INV_LN2);            // f64 round_ties_even 미러
    double r = fma(-kd, LN2_HI, xd);
    r = fma(-kd, LN2_LO, r);
    double p = 1.0 / 1307674368000.0;
    p = fma(p, r, 1.0 / 479001600.0);
    p = fma(p, r, 1.0 / 39916800.0);
    p = fma(p, r, 1.0 / 3628800.0);
    p = fma(p, r, 1.0 / 362880.0);
    p = fma(p, r, 1.0 / 40320.0);
    p = fma(p, r, 1.0 / 5040.0);
    p = fma(p, r, 1.0 / 720.0);
    p = fma(p, r, 1.0 / 120.0);
    p = fma(p, r, 1.0 / 24.0);
    p = fma(p, r, 1.0 / 6.0);
    p = fma(p, r, 0.5);
    p = fma(p, r, 1.0);
    p = fma(p, r, 1.0);
    long long k = (long long)kd;
    if (k > 127) return INFINITY;
    double scale = __longlong_as_double((long long)(k + 1023) << 52);
    return (float)(p * scale);
}

// ── core ln_cr 직이식(core ops.rs L91-121 — 비트동일 미러) ──
// 정규수 v ≥ 1 전용(softplus log1p 경로 — y+1 ≥ 1). m·2^k 정규화 →
// atanh 급수 t²23차 fma 호너(q: 1/25→1/23→…→1/3→1) → 2단 ln2 fma 합.
// 리터럴·연산 순서 변경 금지.
__device__ __forceinline__ double ds4_ln_cr(double v)
{
    const double LN2_HI = 6.9314718036912382e-01;
    const double LN2_LO = 1.9082149292705877e-10;
    long long bits = __double_as_longlong(v);
    long long k = (((unsigned long long)bits >> 52) & 0x7ff) - 1023;
    double m = __longlong_as_double(
        (bits & ~(0x7ffLL << 52)) | (1023LL << 52));
    double t = (m - 1.0) / (m + 1.0);
    double t2 = t * t;
    double q = 1.0 / 25.0;
    q = fma(q, t2, 1.0 / 23.0);
    q = fma(q, t2, 1.0 / 21.0);
    q = fma(q, t2, 1.0 / 19.0);
    q = fma(q, t2, 1.0 / 17.0);
    q = fma(q, t2, 1.0 / 15.0);
    q = fma(q, t2, 1.0 / 13.0);
    q = fma(q, t2, 1.0 / 11.0);
    q = fma(q, t2, 1.0 / 9.0);
    q = fma(q, t2, 1.0 / 7.0);
    q = fma(q, t2, 1.0 / 5.0);
    q = fma(q, t2, 1.0 / 3.0);
    q = fma(q, t2, 1.0);
    double lnm = 2.0 * t * q;
    double kh = (double)k * LN2_HI;
    double kl = (double)k * LN2_LO;
    double s1 = lnm + kh;
    double s2 = (lnm - s1) + kh;
    return s1 + (s2 + kl);
}

// ── softplus·silu·sqrtsoftplus(core ops.rs L128-140·L138-140,
//    deepseek4/ops.rs L355-358) ──
__device__ __forceinline__ float ds4_softplus(float x)
{
    if (x > 20.0f) return x;
    // log1p_cr(y) = ln_cr(y as f64 + 1.0) as f32 (core ops.rs L123-125).
    return (float)ds4_ln_cr((double)ds4_exp_cr(x) + 1.0);
}
__device__ __forceinline__ float ds4_silu(float x)
{
    return x / (1.0f + ds4_exp_cr(-x));
}

// ── bf16 RNE 경계(deepseek4/ops.rs bf16_round L17-23) — 순수 비트 경로 ──
__device__ __forceinline__ float ds4_bf16_round(float x)
{
    unsigned int b = __float_as_uint(x);
    unsigned int hi = ((b >> 16) & 1u);
    return __uint_as_float(((b + 0x7FFFu + hi) >> 16) << 16);
}

// ── FP8-sim 원소 트윈(deepseek4/ops.rs L77-152) — 정수 비트 경로 ──
// pow2_ceil: x>0 정규수 가정(amax 하한 1e-4 → x ≥ 2.2e-7).
__device__ __forceinline__ float ds4_pow2_ceil(float x)
{
    unsigned int b = __float_as_uint(x);
    int e = (int)((b >> 23) & 0xFF);
    int l2 = e - 127 + (int)((b & 0x7FFFFF) != 0u);
    return __uint_as_float((unsigned int)(l2 + 127) << 23);
}
// f32 → e4m3fn 비트(RNE) — |x| ≤ 448 클램프 후 가정(호출부 책임).
__device__ __forceinline__ unsigned char ds4_f32_to_e4m3(float x)
{
    unsigned int xb = __float_as_uint(x);
    unsigned char sign = (unsigned char)((xb >> 24) & 0x80u); // 부호 비트(-0.0 포함)
    float a = fabsf(x);
    if (a < 1.0f / 1024.0f) return sign;          // 2^-10 미만 → 0
    if (a < 1.0f / 64.0f) {                        // 비정규: q=a·2^9 RNE
        float q = a * 512.0f;
        float rr = rintf(q);
        if (rr >= 8.0f) return (unsigned char)(sign | (1 << 3));
        return (unsigned char)(sign | (unsigned char)rr);
    }
    unsigned int b = __float_as_uint(a);
    int e = (int)((b >> 23) & 0xFF) - 127;
    unsigned int mant = b & 0x7FFFFF;
    unsigned int rem = mant & 0xFFFFFu;
    unsigned int half = 1u << 19;
    unsigned int mm = mant >> 20;
    if (rem > half || (rem == half && (mm & 1u) != 0u)) mm += 1;
    int ee = e;
    if (mm == 8) { mm = 0; ee += 1; }
    unsigned char e4 = (unsigned char)(ee + 7);
    if (e4 > 15 || (e4 == 15 && mm > 7)) return (unsigned char)(sign | 0x7F);
    return (unsigned char)(sign | (e4 << 3) | (unsigned char)mm);
}
// e4m3fn 비트 → f32(정확).
__device__ __forceinline__ float ds4_e4m3_to_f32(unsigned char u)
{
    float sign = __uint_as_float((((unsigned int)(u & 0x80)) << 24) | 0x3F800000u);
    int e4 = (int)((u >> 3) & 0xF);
    unsigned int m = (unsigned int)(u & 7);
    if (e4 == 15 && m == 7) {
        unsigned int nb = 0x7FC00000u | (((unsigned int)(u & 0x80)) << 24);
        return __uint_as_float(nb);
    }
    if (e4 == 0) return sign * (float)m * 1.953125e-3f; // m·2^-9(정확)
    float mag = __uint_as_float((((unsigned int)(e4 - 7 + 127)) << 23) | (m << 20));
    return sign * mag;
}

// ── 게이트: 로짓 GEMV + sqrtsoftplus(moe.rs gate_scores L42-50) ──
// s[t][e] = √softplus(Σ_i x[t][i]·gw[i][e]) — gw는 k-major [dim][n_routed]
// (loader.rs plain_kmat 전치 결과). 스레드당 (t,e) 1개·i 오름차순 순차
// 누산(gemm_nt L58-75 미러 — 곱·가산 각 1회 반올림, -fmad=false).
extern "C" __global__ void ds4_gate(
    const float* __restrict__ x,
    const float* __restrict__ gw,
    float* __restrict__ s,
    int dim,
    int n_routed)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_routed) return;
    size_t t = blockIdx.y;
    const float* xr = x + t * (size_t)dim;
    float acc = 0.0f;
    for (int i = 0; i < dim; ++i)
        acc += xr[i] * gw[(size_t)i * n_routed + e];
    s[t * (size_t)n_routed + e] = sqrtf(ds4_softplus(acc));
}

// ── 공통: 페어 (eid,w) eid 오름차순 안정 삽입 정렬 + 기록 ──
// moe.rs route L77-79 sort_by_key(eid)(Rust 안정 정렬) 미러 — k ≤ 16.
__device__ __forceinline__ void ds4_sort_pairs_asc(int* eid, float* w, int k)
{
    for (int i = 1; i < k; ++i) {
        int ke = eid[i];
        float kw = w[i];
        int j = i - 1;
        while (j >= 0 && eid[j] > ke) {
            eid[j + 1] = eid[j];
            w[j + 1] = w[j];
            --j;
        }
        eid[j + 1] = ke;
        w[j + 1] = kw;
    }
}

// ── 해시 라우팅 L0-2(moe.rs route_hash L81-85 + route L67-79) ──
// sel = tid2eid[tid·k + j] 행 룩업(톱-k/bias 없음) — Σ는 행 순서 순차,
// w = s/Σ·route_scale(나눗셈·곱 각 1회). 토큰당 단일 스레드(레인 0).
extern "C" __global__ void ds4_route_hash(
    const float* __restrict__ s,
    const long long* __restrict__ tid2eid,
    const unsigned int* __restrict__ tids,
    int n_routed,
    int k,
    float route_scale,
    int* __restrict__ ids,
    float* __restrict__ wts)
{
    if (threadIdx.x != 0) return;
    size_t t = blockIdx.x;
    const float* st = s + t * (size_t)n_routed;
    int eid[16];
    float w[16];
    for (int j = 0; j < k; ++j)
        eid[j] = (int)tid2eid[(size_t)tids[t] * k + j];
    float sum = 0.0f;                               // 행 순서 순차(route L70)
    for (int j = 0; j < k; ++j)
        sum += st[eid[j]];
    for (int j = 0; j < k; ++j)
        w[j] = st[eid[j]] / sum * route_scale;       // s/sum ×1.5
    ds4_sort_pairs_asc(eid, w, k);
    for (int j = 0; j < k; ++j) {
        ids[t * k + j] = eid[j];
        wts[t * k + j] = w[j];
    }
}

// ── noaux_tc 라우팅 L3+(moe.rs route_routed L87-96 + topk_stable L53-65) ──
// biased = s + bias(f32 가산 — 선택에만), top-k 동점 낮은 인덱스 우선
// (엄격 > 상향 스캔 = 내림차 안정 정렬 동치), w = gather(s, sel)(bias
// 미포함) — Σ는 선택 순서 순차. 토큰당 단일 스레드.
extern "C" __global__ void ds4_route_routed(
    const float* __restrict__ s,
    const float* __restrict__ bias,
    int n_routed,
    int k,
    float route_scale,
    int* __restrict__ ids,
    float* __restrict__ wts)
{
    if (threadIdx.x != 0) return;
    size_t t = blockIdx.x;
    const float* st = s + t * (size_t)n_routed;
    float b[1024];                                  // n_routed ≤ 1024(모듈 검증)
    for (int e = 0; e < n_routed; ++e)
        b[e] = st[e] + bias[e];
    int eid[16];
    for (int kk = 0; kk < k; ++kk) {                // 동치 선택(엄격 >)
        int be = -1;
        float bv = -INFINITY;
        for (int e = 0; e < n_routed; ++e) {
            if (b[e] > bv) { bv = b[e]; be = e; }
        }
        eid[kk] = be;
        b[be] = -INFINITY;
    }
    float sum = 0.0f;                               // 선택 순서 순차(route L70)
    for (int kk = 0; kk < k; ++kk)
        sum += st[eid[kk]];
    float w[16];
    for (int kk = 0; kk < k; ++kk)
        w[kk] = st[eid[kk]] / sum * route_scale;    // w = s(편향 없음)/Σ×1.5
    ds4_sort_pairs_asc(eid, w, k);
    for (int j = 0; j < k; ++j) {
        ids[t * k + j] = eid[j];
        wts[t * k + j] = w[j];
    }
}

// ── FP8-sim 128블록(deepseek4/ops.rs fp8_sim L174-190) — 제자리 ──
// 블록 amax(트리 환원 — max는 순서 무관 정확) → 하한 1e-4 →
// s = pow2_ceil(amax·(1/448)) → q=(v/s) ±448 클램프 → e4m3 RNE 왕복 → ×s.
// 그리드 x = rows·ceil(cols/128), 블록 128(청크 1블록). amax 환원은
// shared 복사 후 기입 전 완료(제자리 안전). 범위 밖 스레드는 amax에
// 0만 기여(하한 1e-4에 무영향)하고 __syncthreads 전 조기 복귀 없음
// (분기 sync UB 방어) — 모듈 계약상 cols는 128 배수(도메인 n==128).
extern "C" __global__ void ds4_fp8_sim(
    float* __restrict__ x,
    int cols)
{
    int cchunks = (cols + 127) / 128;
    size_t row = blockIdx.x / (size_t)cchunks;
    int cb = blockIdx.x % cchunks;
    int base = cb * 128;
    int n = min(128, cols - base);
    bool live = (int)threadIdx.x < n;
    float* xr = x + row * (size_t)cols + base;
    float v = live ? xr[threadIdx.x] : 0.0f;
    __shared__ float smax[128];
    smax[threadIdx.x] = fabsf(v);
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if ((int)threadIdx.x < st)
            smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + st]);
        __syncthreads();
    }
    float amax = fmaxf(smax[0], 1e-4f);             // amax 하한(ops.rs L184)
    float s = ds4_pow2_ceil(amax * (1.0f / 448.0f));
    if (live) {
        float q = fminf(fmaxf(v / s, -448.0f), 448.0f);
        xr[threadIdx.x] = ds4_e4m3_to_f32(ds4_f32_to_e4m3(q)) * s;
    }
}

// ── f32 포인터 배열 GEMV(mm_paired 계급 — FNE fn_moe_gemm_f16_ptr 미러) ──
// out[p][o] = Σ_i x[p][i]·W_p[i][o] — W는 k-major f32(트렐리스 디양자화
// 값, loader.rs linear L203-247 산출). 페어 p마다 독립 가중치 포인터
// wptr[p](전문가별 디스패치 — moe_forward expert 콜백 미러). i 오름차순
// 순차 누산(gemm_nt 미러 — -fmad=false 하에서 곱·가산 2회 반올림).
extern "C" __global__ void ds4_gemv_f32_ptr(
    const float* __restrict__ x,
    const unsigned long long* __restrict__ wptr,
    float* __restrict__ out,
    int n_in,
    int n_out)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n_out) return;
    size_t p = blockIdx.y;
    const float* w = (const float*)(uintptr_t)wptr[p];
    const float* xr = x + p * (size_t)n_in;
    float acc = 0.0f;
    for (int i = 0; i < n_in; ++i)
        acc += xr[i] * w[(size_t)i * n_out + o];
    out[p * (size_t)n_out + o] = acc;
}

// ── SwiGLU limit 10(비대칭)+라우팅 가중치+bf16(moe.rs L117-123 미러) ──
// h[p][j] = bf16(silu(min(g, L)) · clamp(u, -L, L) · wp[p]) — 곱 순서:
// silu(g')·u' 1회 반올림 → ×w 1회 반올림 → bf16 경서(swiglu_limit
// ops.rs L361-364: gate-proj는 max만, up-proj는 [-L,L] 클램프 — 비대칭).
extern "C" __global__ void ds4_swiglu(
    const float* __restrict__ g,
    const float* __restrict__ u,
    const float* __restrict__ wp,
    float* __restrict__ h,
    int n,
    float limit)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    size_t p = blockIdx.y;
    float gc = fminf(g[(size_t)p * n + j], limit);
    float uc = fminf(fmaxf(u[(size_t)p * n + j], -limit), limit);
    float hv = ds4_silu(gc) * uc;
    hv = hv * wp[p];
    h[(size_t)p * n + j] = ds4_bf16_round(hv);
}

// ── bf16 경계 제자리(deepseek4/ops.rs bf16_round_slice L25-31) ──
extern "C" __global__ void ds4_bf16_round(
    float* __restrict__ y,
    int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    y[j] = ds4_bf16_round(y[j]);
}

// ── 결합: 전문가 누산(eid 오름차 페어) + 공유 무스케일 + 최종 bf16 ──
// moe.rs moe_forward L180-216: 토큰별 y += w·E(x) — 가중치는 이미 h에
// 접힘(expert_ffn weight 파라미터 — L119), yo[p]는 bf16 경계값(L129),
// 공유 가산 후 최종 bf16(L208-212). 원소별: acc = Σ_p yo[p][i](순차,
// p 순서 = 토큰 내 eid 오름차) → + sh[ti][i] → bf16.
extern "C" __global__ void ds4_combine(
    const float* __restrict__ yo,
    const int* __restrict__ poff,
    const float* __restrict__ sh,
    float* __restrict__ out,
    int dim)
{
    int ti = blockIdx.x;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float acc = 0.0f;
        for (int p = poff[ti]; p < poff[ti + 1]; ++p)
            acc += yo[(size_t)p * dim + i];
        float v = acc + sh[(size_t)ti * dim + i];
        out[(size_t)ti * dim + i] = ds4_bf16_round(v);
    }
}
