// ── 어텐션 CUDA 포팅 (G6, 2026-10-04) ──
// [KVQ 채택 2026-10-10] f32/int4 KV 커널 제거 — int8(_q) 단일 경로.
// (attn_prep·attn_fwd3s·attn_fwd3s_part·attn_prep_q4·attn_fwd3s_part_q4)
// 산술은 구 rawhip 커널의 attn_prep(L607-687)·
// attn_fwd3s(L799-853)·pos_bump(L1152-1155)을 1:1 직이식한다
// (원본 그대로 베낌 — 트렐리스 비트 조작식은 이 파일에 없다).
// 원본과의 차이는 5점(1-2 아래 · 3-4는 #include 직후 트랜센던트 블록):
// 1) hip 판이 27B 폭(q헤드 24·KV헤드 4·kv 1024·KV cap 1024)으로 경직된 상수를
//    런치 인자(q_heads/kv_heads/cap)로 일반화 — 27B 값(24/4/1024)을 넣으면
//    원소 순서까지 원본과 동일. 목적은 35B-A3B 어텐션 형상 지원
//    (q헤드 16·KV헤드 2 — 실측 형상 config.json
//    text_config num_attention_heads=16·num_key_value_heads=2 실측 2026-10-04).
// 2) hip 시그니처의 pos0_ 파라미터(사실상 미사용 — 본체는 pp[0]만 판독)를
//    CUDA에서는 아예 제거한다: 결함 4호(KV 인덱스는 pp[0] "디바이스" 판독 —
//    파라미터 아님, 그래프/루프 설계에 필수)의 계약을 시그니처 수준에서
//    강제한다. 호스트 pos 사본 경로는 제거됐다(H 정리 — 2026-10-09).
// 3) 트랜센던트/수축(아래 블록 참조): 빌드 -fmad=false(FMA 수축 제거 —
//    호스트 미러와 연산 DAG 비트동일), 초월함수는 자작 f64 미러
//    (gdn_exp_d/gdn_log_d와 동일 노선), rope theta는 exp(ln(1e7)·e)
//    f64 재구성, rsqrtf(≤2ulp 근사)는 IEEE sqrt+div로 대체. 전부
//    (hip은 참조 구현일 뿐 진실 아님)에 근거한 정밀화 — 수학 동등·정밀도
//    상향, 값 변화는 ulp 수준. 어텐션은 계약 최대tightness maxdiff ≤2e-7
//    이라 이 미러 계약이 사실상 필수다.
// 4) fwd3s 도메인 강제: fwd3s는 T≤8 소형 전용("fwd3s는 T≤8
//    소형 전용"). 커널 첫 줄 t_len>8 균등 조기복귀(기록 없음 — 부분 기록
//    오염 없는 깨끗한 거부) + 모듈층 사전 Err의 이중 계약.
// 5) 원본 fwd3s의 1e30f 무한대 대체치·환원 구조는 불변(값 계약).
// 6) [S12] 위치축 청크.online 소프트맥스 이식 — hip 판 L202-276
//    (P0 재작성본)을 그대로 옮겨 공유메모리 sarr를 [1024]→[256]으로 줄이고
//    kvcap과 무관하게 했다. exp는 attn_expf 미러를 쓰는 CUDA 규약을 유지한다.
//
// 그리드 계약(결함 5호 정신 — T축은 grid.x): prep는 (T, q_heads+kv_heads),
// fwd3s는 (T, q_heads). 블록 128(prep)·256(fwd3s) — 원본과 동일.
//
// [CMP 170HX(sm_80, GA100 70SM, HBM2e ~1.5TB/s) 설계 근거]
// - prep: 그리드 (T≤8, 28[27B]/18[35B]), 블록 128, 정적 공유 1.5KB(red 512B
//   + hd 1024B) → 레지스터·공유 모두 소형, 다중 블록/SM 상주. 지배 비용은
//   qg [T][q_heads·512] 판독(27B T=8 393KB) — 완전 coalesce 스트리밍,
//   HBM2e 대역 포화 관점 무해수. rope는 32레인만 회전(partial_rotary 0.25×
//   256=64차원=쌍32) — 원본 구조 1:1(정합 우선, 구조 재설계 금지).
// - fwd3s: 그리드 (T≤8, 24[27B]/16[35B]), 블록 256, 정적 공유 2KB(qs 1KB +
//   sarr 1KB + reds 1KB — S12 청크화로 6KB에서 축소). 블록당 kc lim행×1KB
//   순차 판독 + vc lim행×256B — 메모리 본드. T=1 디코드(24블록)는 70SM 대비
//   저점유가 설계 의도다: 정확성 우선 피벗(원본 주석)이며 처리량 경로는
//   t-블록 fwd3(후속 목표). T=8 스펙 디코드(192블록)는 70SM의 ~3블록/SM로
//   확산 — 소형 전용 도메인(T≤8)이 점유 상한과 짝한다.
//   [S12] 공유메모리가 kvcap과 무관해졌으므로 블록당 점유가 위치축으로
//   늘어나지 않는다 — 옛 1024행 상한은 sarr에서 사라졌다.
// - pos_bump: 1스레드 — 캡처 그래프 내 pos 전진(h2d 불가 대체, 결함 16호
//   정신). 원본 그대로.
#include <cuda_fp16.h>

// ── 미러 트랜센던트(G6 — G5 gdn_exp_d와 동일 노선) ──
// 계약: 장치 초월함수(libdevice expf 2ulp·rsqrtf·powf/cosf/sinf)는 호스트
// 오라클과 비트재현 불가 — 어텐션 임계 2e-7(f32 ulp의 ~16배)에서는 ulp급
// 편차가 f16 경계·소프트맥스 분모를 타고 남으므로(계약 최tight),
// 본 파일의 트랜센던트는 순수 f64 연산 DAG(IEEE mul/add/div/floor·비트
// 재구성, FMA 수축 없음 — 빌드 -fmad=false)로 자작해 양측(본 .cu ↔
// 구 프로브 오라클 트윈)의 비트동일을 계약으로 삼는다(G5 실측
// 실측: libdevice expf는 3.1M 표본 중 30%에서 참값 ±1ulp). exp는 G5
// gdn_exp_d와 동일 DAG(차수 7 테일러·비트 재구성 2^k). 도메인: exp |x|≤128,
// sincos 0≤a≤2^20(rope ang = pos·theta ≤ cap·1 — 실사용 ≤1024),
// theta |e|≤ln(1e7). 연산 순서·상수는 절대 변경 금지(비트동일 계약 —
// Rust 트윈과 리터럴까지 동일해야 한다).
__device__ __forceinline__ double attn_exp_d(double x)
{
    // [2026-10-09 P8-attn-3 결함 수정] 도메인 가드 — k 기반 2^k 재구성은
    // k ≥ -1022(즉 x ≥ -708.5)에서만 유효. KV 분할 병합이 빈 분할에 -1e30을
    // 넣어 NaN이 됐다(S=4/8 실측: 토큰 0 반복). gdn_exp_d와 동일 의미론.
    if (x < -708.5) {
        return 0.0;
    }
    if (x > 709.78) {
        return __longlong_as_double(0x7ff0000000000000LL); // +inf
    }
    // k = floor(x·invln2 + ½)(양수 음수 공용 반올림) → r = x − k·ln2(hi/lo)
    // → 테일러 차수 7 → 2^k 비트 재구성. gdn_exp_d(G5)와 동일 DAG.
    const double invln2 = 1.4426950408889634;
    const double ln2_hi = 6.9314718036912382e-01;
    const double ln2_lo = 1.9082149292705877e-10;
    int k = (int)floor(x * invln2 + 0.5);
    double r = x - (double)k * ln2_hi;
    r = r - (double)k * ln2_lo;
    double p = 1.0 + r * (1.0 + r * (0.5 + r * (0.16666666666666666
        + r * (0.041666666666666664 + r * (0.008333333333333333
        + r * (0.001388888888888889 + r * 0.0001984126984126984))))));
    double scale = __longlong_as_double((long long)(1023 + k) << 52);
    return p * scale;
}

__device__ __forceinline__ float attn_expf(float x)
{
    return (float)attn_exp_d((double)x);
}

// rope theta = powf(1e7, −2·tid/64)(원본 L630/664)의 f64 재구성:
// exp(ln(1e7)·e) — e = −2·tid/64는 f64에서 정확(2·tid 정수, /64 지수).
// ln(1e7) 리터럴은 Rust 트윈과 동일 문자열(비트동일 계약). tid=0 → 정확히 1.
__device__ __forceinline__ float attn_theta(int tid)
{
    double e = -(2.0 * (double)tid) / 64.0;
    double th = attn_exp_d(16.11809565095832 * e);
    return (float)th;
}

// sincos f64 트윈: Cody-Waite 2분할 환원(k≤2^20에서 k·pio2_1은 53비트 내
// 정확 — pio2_1은 하위 20비트가 0) + 테일러 z⁶ Horner(차 오차 ~3e-13,
// f32 ulp 6e-8의 5자리수 아래). 사분면 매핑 n=k&3. 도메인 0 ≤ a ≤ 2^20
// (rope ang ≥ 0 — pos 무부호·theta ≥ 0).
__device__ __forceinline__ void attn_sincos_d(double a, double* co, double* si)
{
    const double invpio2 = 6.36619772367581342433e-01;
    const double pio2_1  = 1.57079632673412561417e+00;
    const double pio2_1t = 6.07710050650619224932e-11;
    double kd = floor(a * invpio2 + 0.5);
    long long k = (long long)kd;
    double r = a - (double)k * pio2_1;
    r = r - (double)k * pio2_1t;
    double z = r * r;
    double st = r * (1.0 - z * (0.16666666666666666
        - z * (0.008333333333333333
        - z * (1.9841269841269841e-04
        - z * (2.7557319223985893e-06
        - z * (2.5052108385441720e-08 - z * 1.6059043836821613e-10))))));
    double ct = 1.0 - z * (0.5
        - z * (0.041666666666666664
        - z * (0.0013888888888888889
        - z * (2.4801587301587302e-05
        - z * (2.7557319223985893e-07 - z * 2.0876756987868100e-09)))));
    int n = (int)(k & 3LL);
    // [결함 21호 수정 2026-10-05] 사분면 표 n1/n3 교차 오류 → fdlibm 규약:
    // n=1 → (−st, ct) · n=3 → (st, −ct). 기존 표는 커널·오라클이 같은 오류를
    // 공유해 자기일관 게이트로 미검출(FND 교차측정으로 발견, kv 6.626e0).
    if (n == 0) { *co = ct; *si = st; }
    else if (n == 1) { *co = -st; *si = ct; }
    else if (n == 2) { *co = -ct; *si = -st; }
    else { *co = st; *si = -ct; }
}

__device__ __forceinline__ float attn_cosf(float a)
{
    double c, s;
    attn_sincos_d((double)a, &c, &s);
    return (float)c;
}

__device__ __forceinline__ float attn_sinf(float a)
{
    double c, s;
    attn_sincos_d((double)a, &c, &s);
    return (float)s;
}

// 마커 pr1c

// [P13] KV 양자화 prep — attn_prep와 산술 동일(q 브랜치 무변경), KV 기록만
// int8(행×헤드 absmax 스케일 = max|x|/127, 0이면 1). 스케일 f32 1개/행헤드.
// 값 차이는 양자화 오차뿐 — 판정은 토큰(단문 골든 + 장문). fwd3s_part_q가 소비.
extern "C" __global__ void attn_prep_q(
    const float* __restrict__ qg,    // [T][q_heads*512] q‖gate 인터리브
    const float* __restrict__ kin,   // [T][kv_heads*256]
    const float* __restrict__ vin,   // [T][kv_heads*256]
    const float* __restrict__ qnw,   // [n_attn][256]
    const float* __restrict__ knw,   // [n_attn][256]
    float* __restrict__ qh,          // [T][q_heads*256]
    signed char* __restrict__ kc,    // [n_attn*cap][kv_heads*256] int8
    signed char* __restrict__ vc,
    float* __restrict__ ksc,         // [n_attn*cap][kv_heads]
    float* __restrict__ vsc,
    const unsigned* __restrict__ pp, // [1] pos0 — 디바이스 판독(결함 4호)
    int t_len, int layer, int q_heads, int kv_heads, int cap)
{
    __shared__ float red[128];
    __shared__ float hd[256];
    int t = blockIdx.x;
    int j = blockIdx.y;
    int tid = threadIdx.x;
    int pos = (int)pp[0] + t;
    long kv_dim = (long)kv_heads * 256;

    if (j < q_heads) {
        int src = t * (q_heads * 512) + j * 512;
        hd[tid] = qg[src + tid];
        hd[tid + 128] = qg[src + 128 + tid];
        __syncthreads();
        float ss = hd[tid] * hd[tid] + hd[tid + 128] * hd[tid + 128];
        red[tid] = ss;
        __syncthreads();
        for (int s = 64; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        float inv = 1.0f / sqrtf(red[0] / 256.0f + 1e-6f);
        hd[tid] = hd[tid] * inv * qnw[layer * 256 + tid];
        hd[tid + 128] = hd[tid + 128] * inv * qnw[layer * 256 + 128 + tid];
        __syncthreads();
        if (tid < 32) {
            float theta = attn_theta(tid);
            float ang = (float)pos * theta;
            float c = attn_cosf(ang), s2 = attn_sinf(ang);
            float x0 = hd[tid], x1 = hd[tid + 32];
            hd[tid] = x0 * c - x1 * s2;
            hd[tid + 32] = x0 * s2 + x1 * c;
        }
        __syncthreads();
        qh[(long)t * (q_heads * 256) + (long)j * 256 + tid] = hd[tid];
        qh[(long)t * (q_heads * 256) + (long)j * 256 + 128 + tid] = hd[tid + 128];
    } else {
        int m = j - q_heads;
        int src = t * (int)kv_dim + m * 256;
        hd[tid] = kin[src + tid];
        hd[tid + 128] = kin[src + 128 + tid];
        __syncthreads();
        float ss = hd[tid] * hd[tid] + hd[tid + 128] * hd[tid + 128];
        red[tid] = ss;
        __syncthreads();
        for (int s = 64; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        float inv = 1.0f / sqrtf(red[0] / 256.0f + 1e-6f);
        hd[tid] = hd[tid] * inv * knw[layer * 256 + tid];
        hd[tid + 128] = hd[tid + 128] * inv * knw[layer * 256 + 128 + tid];
        __syncthreads();
        if (tid < 32) {
            float theta = attn_theta(tid);
            float ang = (float)pos * theta;
            float c = attn_cosf(ang), s2 = attn_sinf(ang);
            float x0 = hd[tid], x1 = hd[tid + 32];
            hd[tid] = x0 * c - x1 * s2;
            hd[tid + 32] = x0 * s2 + x1 * c;
        }
        __syncthreads();
        long dst = ((long)layer * cap + pos) * kv_dim + (long)m * 256;
        long sr = ((long)layer * cap + pos) * kv_heads + m;
        // K absmax → int8
        red[tid] = fmaxf(fabsf(hd[tid]), fabsf(hd[tid + 128]));
        __syncthreads();
        for (int s = 64; s > 0; s >>= 1) {
            if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
            __syncthreads();
        }
        float ks = red[0] * (1.0f / 127.0f);
        if (ks == 0.0f) {
            ks = 1.0f;
        }
        float ik = 1.0f / ks;
        __syncthreads(); // red[0](K absmax) 판독 완료 후 red 재사용
        int q0 = __float2int_rn(hd[tid] * ik);
        int q1 = __float2int_rn(hd[tid + 128] * ik);
        kc[dst + tid] = (signed char)max(-127, min(127, q0));
        kc[dst + 128 + tid] = (signed char)max(-127, min(127, q1));
        if (tid == 0) {
            ksc[sr] = ks;
        }
        // V absmax → int8
        float v0 = vin[src + tid];
        float v1 = vin[src + 128 + tid];
        red[tid] = fmaxf(fabsf(v0), fabsf(v1));
        __syncthreads();
        for (int s = 64; s > 0; s >>= 1) {
            if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
            __syncthreads();
        }
        float vs = red[0] * (1.0f / 127.0f);
        if (vs == 0.0f) {
            vs = 1.0f;
        }
        float iv = 1.0f / vs;
        int w0 = __float2int_rn(v0 * iv);
        int w1 = __float2int_rn(v1 * iv);
        vc[dst + tid] = (signed char)max(-127, min(127, w0));
        vc[dst + 128 + tid] = (signed char)max(-127, min(127, w1));
        if (tid == 0) {
            vsc[sr] = vs;
        }
    }
}

// ── fwd3s 본체(구 rawhip 커널 L799-853 직이식, 폭 인자화 + 도메인 강제) ──
// 3단 구조: (1) 스코어 q·kᵀ·scale(행당 1스레드 순차 d-누산) → sarr,
// (2) 트리 max → 지수(트윈)·트리 sum(스레드 보폭 순차 + 트리 — 환원 순서
// 미러 계약) → sarr 자리에 확률, (3) AV 순차 누산(스레드=dim 원소) →
// 게이트 sigmoid 곱. WG당 (t,h). lim = pp[0](디바이스)+t+1 — 결함 4호.
// 도메인: T≤512(ATTN_TMAX) — 위반 시 전 블록 조기복귀(기록 없음).
// [2026-10-09] 8→32: 블록=토큰 구조라 smem이 T와 무관(가드일 뿐) —
// 프리필 청크 확대(가중치 재사용 ↑)용.
#define ATTN_TMAX 512
// ── [S12 2026-10-07] 위치축 청크.online 소프트맥스 — 공유메모리 6KB→2KB ──
// 이전 구현은 sarr[1024]에 lim행 스코어를 전부 담아 두었다. 공유메모리 고정
// 크기라 lim(=pp[0]+t+1)이 1024를 넘으면 sarr[row]가 블록 밖을 넘어가고
// CUresult 700(illegal address)으로 죽는다 — cap을 아무리 키워도 pos≥1023
// 에서 서빙이 막혀, CUDA의 실질 ctx 상한이 1024로 고정돼 있었다
// (규칙적으로 확인해 봉인한 한계).
//
// 해법은 알고리즘 새로 쓰기가 아니라 **hip 판의 청크 온라인 소프트맥스를
// 그대로 이식**하는 것이었다(rawhip/kernels/구 rawhip 커널 L202-276가
// 이미 그 형태로 재작성돼 있다 — 청크 256행마다 running
// max/l/acc를 갱신하므로 smem이 kvcap과 무관해진다).
//
// [왜 lim≤256 구간은 비트동일인가 — 게이트 기준선이 깨지지 않는 근거]
// 청크 경계 안(lim≤256)에서는 청크가 하나뿐이라 이식 전후가 같은 연산을
// 같은 순서로 한다: 스코어(행당 1스레드 256차원 순차 내적·scale 곱은 동일),
// max 트리(동일 값에 대한 fmaxf 트리), exp+sum 트리(동일 per-스레드 값),
// AV 누산(acc=0에서 lim행 순차 — 청크 하나면 base=0이므로 행 순서 동일),
// 최종 (acc/l)·sigmoid 동일. corr은 첫 청크에서 0이고(acc=0·0), l_run은
// 0·0+sum이라 첫 청크 결과가 그대로다. 즉 **기존 ctx 1024 게이트의 프롬프트
// (41토큰)+16 = lim≤57은 비트 단위로 동일** — 기준선 재기록이 불필요하고,
// lim>256 구간(게이트가 처음 밟지 않던 영역)만 환원 순서가 달라진다.
// 그 구간은 이전엔 서빙이 불가능했으므로 회귀 판정 대상 자체가 없다.
//
// [환원 순서 변경 — 규칙 10a] 청크 경계를 넘는 구간은 max/sum 리덕션 순서가
// 달라져 값이 ulp 수준으로 움직인다. 수학적으로는 동등(온라인 소프트맥스
// 정의 그대로)하며 f16 GEMM 누산(1.6e-2)보다 작다. 실사용 판정은 기존과
// 동일하게 argmax 일치다.
#define ATTN_CHUNK 256
// [R13 2026-10-10] part/단일 커널 공통 청크 스텝 — 온라인 소프트맥스.
// kdot(i) = 스코어(스케일 적용 — 청크 로컬 i는 호출부가 row로 환산),
// vacc(i) = 이 스레드 dim의 V 값. 산술 순서는 종전 4벌의 인라인 루프와
// 동일(청크 내 순차·d 오름차순·exp 순서 동일) — 값 불변 계약.
template <typename KDot, typename VAcc>
__device__ __forceinline__ void attn_part_step(KDot kdot, VAcc vacc,
                                               float* __restrict__ sarr,
                                               float* __restrict__ reds, int nch,
                                               int tid, float& m_run, float& l_run,
                                               float& acc)
{
    float p = -1e30f;
    if (tid < nch) {
        p = kdot(tid);
    }
    sarr[tid] = p;
    __syncthreads();
    // 청크 max(트리) — reds[0] 확정(균질)
    reds[tid] = (tid < nch) ? sarr[tid] : -1e30f;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (tid < st) reds[tid] = fmaxf(reds[tid], reds[tid + st]);
        __syncthreads();
    }
    float m_new = fmaxf(m_run, reds[0]);
    float corr = (m_run <= -1e29f) ? 0.0f : attn_expf(m_run - m_new);
    __syncthreads(); // reds[0](청크 max) 판독 완료 후 e 기록 — 덮어쓰기 레이스 방지
    // exp+청크 sum — e를 sarr에 재기록(AV 재사용)
    float e = 0.0f;
    if (tid < nch) {
        e = attn_expf(sarr[tid] - m_new);
        sarr[tid] = e;
    }
    reds[tid] = e;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (tid < st) reds[tid] += reds[tid + st];
        __syncthreads();
    }
    // AV: 자기 dim에 청크 전 행 누산(행 순서·언롤 불변 — 계약)
    acc *= corr;
#pragma unroll 8
    for (int i = 0; i < nch; i++) {
        acc += sarr[i] * vacc(i);
    }
    l_run = l_run * corr + reds[0];
    m_run = m_new;
    __syncthreads(); // sarr 재사용 전 전체 완료(다음 청크 스코어 덮어쓰기 보호)
}

// [2026-10-09 P8-attn-3] KV 분할 attention — 4k ctx 실측: 스코어/AV가 24블록
// (q_heads)만 써 SM의 81% 유휴 + K/V 로드 지연 노출. lim>256에서 KV를
// ATTN_SPLITS(8)로 나눠 블록 = q_heads×S(27B 192블록)로 병렬화하고, 병합은
// 분할 인덱스 오름차순 고정 순서(결정적). lim≤256은 종전 단일 경로 유지 —
// 골든 구간 비트 동일 계약 보존.
#define ATTN_SPLITS_C 8



// [P13] int8 KV 판독 — attn_fwd3s_part와 구조/환원 순서 동일(온라인 소프트맥스·
// 분할 결정), K/V 로드만 int8 + 스케일 곱. 스코어 (q·k_i8)·(0.0625·ks),
// AV acc += e·(v_i8·vs). 병합은 f32 part 그대로(attn_fwd3s_merge).
extern "C" __global__ void attn_fwd3s_part_q(
    const float* __restrict__ qh,
    const signed char* __restrict__ kc,
    const signed char* __restrict__ vc,
    const float* __restrict__ ksc,
    const float* __restrict__ vsc,
    float* __restrict__ part,   // [T][q_heads][S][258]: m, l, acc[256]
    const unsigned* __restrict__ pp,
    int t_len, int layer, int q_heads, int kv_heads, int cap, int splits)
{
    int t = blockIdx.x;
    int hs = blockIdx.y;
    int h = hs / splits;
    int s = hs - h * splits;
    int tid = threadIdx.x;
    int gq = q_heads / kv_heads;
    int kh = h / gq;
    float scale = 0.0625f;
    int lim = (int)pp[0] + t + 1;
    long kv_dim = (long)kv_heads * 256;
    long qrow = (long)t * (q_heads * 256) + (long)h * 256;
    __shared__ float qs[256];
    __shared__ float sarr[ATTN_CHUNK];
    __shared__ float reds[256];
    qs[tid] = qh[qrow + tid];
    __syncthreads();
    const int eff = (lim <= 256) ? 1 : splits;
    float* out = part + (((long)t * q_heads + h) * splits + s) * 258;
    if (s >= eff) {
        if (tid == 0) {
            out[0] = -1e30f;
            out[1] = 0.0f;
        }
        out[2 + tid] = 0.0f;
        return;
    }
    int chunk_rows = (lim + eff - 1) / eff;
    chunk_rows = ((chunk_rows + ATTN_CHUNK - 1) / ATTN_CHUNK) * ATTN_CHUNK;
    int lo = s * chunk_rows;
    int hi = min(lim, lo + chunk_rows);
    float m_run = -1e30f;
    float l_run = 0.0f;
    float acc = 0.0f;
    for (int base = lo; base < hi; base += ATTN_CHUNK) {
        const int nch = min(ATTN_CHUNK, hi - base);
        attn_part_step(
            [&](int i) -> float {
                const int row = base + i;
                const unsigned* k4 = reinterpret_cast<const unsigned*>(
                    kc + ((long)layer * cap + row) * kv_dim + (long)kh * 256);
                const float4* q4 = reinterpret_cast<const float4*>(qs);
                float p0 = 0.0f;
#pragma unroll 8
                for (int d4 = 0; d4 < 64; d4++) {
                    const unsigned w = k4[d4];
                    const float4 qv = q4[d4];
                    p0 += qv.x * (float)(signed char)(w & 0xffu);
                    p0 += qv.y * (float)(signed char)((w >> 8) & 0xffu);
                    p0 += qv.z * (float)(signed char)((w >> 16) & 0xffu);
                    p0 += qv.w * (float)(signed char)(w >> 24);
                }
                return p0 * (scale * ksc[((long)layer * cap + row) * kv_heads + kh]);
            },
            [&](int i) -> float {
                const int row = base + i;
                return (float)vc[((long)layer * cap + row) * kv_dim + (long)kh * 256 + tid] *
                       vsc[((long)layer * cap + row) * kv_heads + kh];
            },
            sarr, reds, nch, tid, m_run, l_run, acc);
    }
    if (tid == 0) {
        out[0] = m_run;
        out[1] = l_run;
    }
    out[2 + tid] = acc;
}



// 병합: 분할 s=0..S-1 오름차순 고정 — 결정적. m=최대, l·acc를 exp(m_s−m)로
// 스케일해 합산(수학적으로 단일 경로와 동일, 반올림만 상이 — lim>256 완화 등급).
extern "C" __global__ void attn_fwd3s_merge(
    const float* __restrict__ part,
    const float* __restrict__ qg,
    float* __restrict__ outv,
    int t_len, int q_heads, int splits)
{
    int t = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    const float* pb = part + ((long)t * q_heads + h) * splits * 258;
    __shared__ float ms[ATTN_SPLITS_C];
    if (tid < splits) {
        ms[tid] = pb[tid * 258];
    }
    __syncthreads();
    float m = -1e30f;
    for (int s = 0; s < splits; s++) {
        m = fmaxf(m, ms[s]);
    }
    float l = 0.0f;
    float acc = 0.0f;
    for (int s = 0; s < splits; s++) {
        const float* ps = pb + s * 258;
        if (ps[1] <= 0.0f) {
            continue; // 빈 분할(0 기여 — 결정적)
        }
        float w = attn_expf(ps[0] - m);
        l += ps[1] * w;
        acc += ps[2 + tid] * w;
    }
    float g = qg[(long)t * (q_heads * 512) + (long)h * 512 + 256 + tid];
    outv[(long)t * (q_heads * 256) + (long)h * 256 + tid] =
        (acc / l) * (1.0f / (1.0f + attn_expf(-g)));
}

// 마커 f3sc

// [R3 2026-10-10] attn_pos_bump 제거 — 미사용(호스트가 attn_set_pos의
// h2d_async로 pp 갱신; 캡처 그래프에서는 replay 전 1회 갱신).
