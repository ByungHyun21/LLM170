// ── EXL3 어텐션 CUDA 포팅 (plans/124 G6, 2026-10-04) ──
// 산술은 rawhip/kernels/src_exl3.hip의 exl3_attn_prep(L607-687)·
// exl3_attn_fwd3s(L799-853)·exl3_pos_bump(L1152-1155)을 1:1 직이식한다
// (원본 그대로 베낌 — plans/124 §2. 트렐리스 비트 조작식은 이 파일에 없다).
// 원본과의 차이는 5점(1-2 아래 · 3-4는 #include 직후 트랜센던트 블록):
// 1) hip 판이 27B 폭(q헤드 24·KV헤드 4·kv 1024·KV cap 1024)으로 경직된 상수를
//    런치 인자(q_heads/kv_heads/cap)로 일반화 — 27B 값(24/4/1024)을 넣으면
//    원소 순서까지 원본과 동일. 목적은 35B-A3B 어텐션 형상 지원
//    (q헤드 16·KV헤드 2 — D:/models/Qwen3.6-35B-A3B-exl3-4.00bpw/config.json
//    text_config num_attention_heads=16·num_key_value_heads=2 실측 2026-10-04).
// 2) hip 시그니처의 pos0_ 파라미터(사실상 미사용 — 본체는 pp[0]만 판독)를
//    CUDA에서는 아예 제거한다: 결함 4호(KV 인덱스는 pp[0] "디바이스" 판독 —
//    파라미터 아님, 그래프/루프 설계에 필수)의 계약을 시그니처 수준에서
//    강제한다. 호스트 pos 사본 경로는 음성대조 전용 쌍둥이
//    exl3_attn_prep_hostpos(명시적 pos0_host 인자)로만 존재 — 프로덕션
//    경로에서 발사 금지(원장 17호 계기 원칙).
// 3) 트랜센던트/수축(아래 블록 참조): 빌드 -fmad=false(FMA 수축 제거 —
//    호스트 미러와 연산 DAG 비트동일), 초월함수는 자작 f64 미러
//    (gdn_exp_d/gdn_log_d와 동일 노선 — G5 원장), rope theta는 exp(ln(1e7)·e)
//    f64 재구성, rsqrtf(≤2ulp 근사)는 IEEE sqrt+div로 대체. 전부 plans/124 §6
//    (hip은 참조 구현일 뿐 진실 아님)에 근거한 정밀화 — 수학 동등·정밀도
//    상향, 값 변화는 ulp 수준. 어텐션은 계약 최대tightness maxdiff ≤2e-7
//    (plans/124 §1)라 이 미러 계약이 사실상 필수다.
// 4) fwd3s 도메인 강제: fwd3s는 T≤8 소형 전용(plans/124 §1 "fwd3s는 T≤8
//    소형 전용"). 커널 첫 줄 t_len>8 균등 조기복귀(기록 없음 — 부분 기록
//    오염 없는 깨끗한 거부) + 모듈층 사전 Err의 이중 계약.
// 5) 원본 fwd3s의 1e30f 무한대 대체치·환원 구조는 불변(값 계약).
//
// 그리드 계약(결함 5호 정신 — T축은 grid.x): prep는 (T, q_heads+kv_heads),
// fwd3s는 (T, q_heads). 블록 128(prep)·256(fwd3s) — 원본과 동일.
//
// [CMP 170HX(sm_80, GA100 70SM, HBM2e ~1.5TB/s) 설계 근거 — plans/124 §0]
// - prep: 그리드 (T≤8, 28[27B]/18[35B]), 블록 128, 정적 공유 1.5KB(red 512B
//   + hd 1024B) → 레지스터·공유 모두 소형, 다중 블록/SM 상주. 지배 비용은
//   qg [T][q_heads·512] 판독(27B T=8 393KB) — 완전 coalesce 스트리밍,
//   HBM2e 대역 포화 관점 무해수. rope는 32레인만 회전(partial_rotary 0.25×
//   256=64차원=쌍32) — 원본 구조 1:1(정합 우선, 구조 재설계 금지).
// - fwd3s: 그리드 (T≤8, 24[27B]/16[35B]), 블록 256, 정적 공유 6KB(qs 1KB +
//   sarr 4KB + reds 1KB). 블록당 kc lim행×1KB 순차 판독 + vc lim행×256B —
//   메모리 본드. T=1 디코드(24블록)는 70SM 대비 저점유가 설계 의도다:
//   정확성 우선 피벗(원본 주석)이며 처리량 경로는 t-블록 fwd3(후속 목표).
//   T=8 스펙 디코드(192블록)는 70SM의 ~3블록/SM로 확산 — 소형 전용
//   도메인(T≤8)이 점유 상한과 짝한다.
// - pos_bump: 1스레드 — 캡처 그래프 내 pos 전진(h2d 불가 대체, 결함 16호
//   정신). 원본 그대로.
#include <cuda_fp16.h>

// ── 미러 트랜센던트(G6 — G5 gdn_exp_d와 동일 노선, plans/124 §6) ──
// 계약: 장치 초월함수(libdevice expf 2ulp·rsqrtf·powf/cosf/sinf)는 호스트
// 오라클과 비트재현 불가 — 어텐션 임계 2e-7(f32 ulp의 ~16배)에서는 ulp급
// 편차가 f16 경계·소프트맥스 분모를 타고 남으므로(plans/124 §1 최tight),
// 본 파일의 트랜센던트는 순수 f64 연산 DAG(IEEE mul/add/div/floor·비트
// 재구성, FMA 수축 없음 — 빌드 -fmad=false)로 자작해 양측(본 .cu ↔
// exl3_cuda_probe.rs 오라클 트윈)의 비트동일을 계약으로 삼는다(G5 실측
// 원장: libdevice expf는 3.1M 표본 중 30%에서 참값 ±1ulp). exp는 G5
// gdn_exp_d와 동일 DAG(차수 7 테일러·비트 재구성 2^k). 도메인: exp |x|≤128,
// sincos 0≤a≤2^20(rope ang = pos·theta ≤ cap·1 — 실사용 ≤1024),
// theta |e|≤ln(1e7). 연산 순서·상수는 절대 변경 금지(비트동일 계약 —
// Rust 트윈과 리터럴까지 동일해야 한다).
__device__ __forceinline__ double attn_exp_d(double x)
{
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

// ── prep 본체(src_exl3.hip L607-687 직이식, 폭 인자화 + 결함 4호) ──
// q 디인터리브[rms+rope] → qh, k rms+rope → KC(디바이스 pos), v 복사 → VC.
// 그리드 (T, q_heads+kv_heads): j<q_heads=q헤드, j≥q_heads=KV헤드
// (m=j−q_heads). WG=128. KV 기록 인덱스 pos = pp[0](디바이스) + t —
// 결함 4호: pp[0] 판독이 그래프/루프 설계의 핵심(pos_bump와 짝).
extern "C" __global__ void exl3_attn_prep(
    const float* __restrict__ qg,    // [T][q_heads*512] q‖gate 인터리브
    const float* __restrict__ kin,   // [T][kv_heads*256]
    const float* __restrict__ vin,   // [T][kv_heads*256]
    const float* __restrict__ qnw,   // [n_attn][256]
    const float* __restrict__ knw,   // [n_attn][256]
    float* __restrict__ qh,          // [T][q_heads*256]
    float* __restrict__ kc,          // [n_attn*cap][kv_heads*256]
    float* __restrict__ vc,
    const unsigned* __restrict__ pp, // [1] pos0 — 디바이스 판독(결함 4호)
    int t_len, int layer, int q_heads, int kv_heads, int cap)
{
    __shared__ float red[128];
    __shared__ float hd[256];
    int t = blockIdx.x;
    int j = blockIdx.y;
    int tid = threadIdx.x;
    int pos = (int)pp[0] + t;   // 결함 4호: 호스트 파라미터가 아니라 장치 판독
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
        // G6 정밀화: rsqrtf(≤2ulp) 대신 IEEE sqrt+div — 호스트 미러와 비트동일.
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
        kc[dst + tid] = hd[tid];
        kc[dst + 128 + tid] = hd[tid + 128];
        vc[dst + tid] = vin[src + tid];
        vc[dst + 128 + tid] = vin[src + 128 + tid];
    }
}
// 마커 pr1c

// ── 음성대조 전용 쌍둥이(결함 4호 재현 — 원장 17호) ──
// pp[0] 디바이스 판독 대신 "호스트 파라미터 사본" pos0_host로 KV 인덱스를
// 계산하는 판(prep와의 유일한 차이 — pos 원천). 장치 pp[0]이 pos_bump 등으로
// 전진한 뒤 호스트 사본이 낡은 값이면 KV 기록 위치가 어긋나고, 그 이격이
// fwd3s(디바이스 판독 경로) 종단 값에서 maxdiff>2e-7로 검출됨을 증명한다.
// 프로덕션 경로에서 발사 금지 — exl3_cuda.rs 검증 전용 진입만 호출.
extern "C" __global__ void exl3_attn_prep_hostpos(
    const float* __restrict__ qg,
    const float* __restrict__ kin,
    const float* __restrict__ vin,
    const float* __restrict__ qnw,
    const float* __restrict__ knw,
    float* __restrict__ qh,
    float* __restrict__ kc,
    float* __restrict__ vc,
    const unsigned* __restrict__ pp, // 판독하지 않는다(계약 위반 재현)
    int t_len, int layer, int q_heads, int kv_heads, int cap, int pos0_host)
{
    __shared__ float red[128];
    __shared__ float hd[256];
    int t = blockIdx.x;
    int j = blockIdx.y;
    int tid = threadIdx.x;
    (void)pp;
    int pos = pos0_host + t;        // 결함 재현: 호스트 사본 — pp[0] 무시
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
        kc[dst + tid] = hd[tid];
        kc[dst + 128 + tid] = hd[tid + 128];
        vc[dst + tid] = vin[src + tid];
        vc[dst + 128 + tid] = vin[src + 128 + tid];
    }
}

// ── fwd3s 본체(src_exl3.hip L799-853 직이식, 폭 인자화 + 도메인 강제) ──
// 3단 구조: (1) 스코어 q·kᵀ·scale(행당 1스레드 순차 d-누산) → sarr,
// (2) 트리 max → 지수(트윈)·트리 sum(스레드 보폭 순차 + 트리 — 환원 순서
// 미러 계약) → sarr 자리에 확률, (3) AV 순차 누산(스레드=dim 원소) →
// 게이트 sigmoid 곱. WG당 (t,h). lim = pp[0](디바이스)+t+1 — 결함 4호.
// 도메인: T≤8(EXL3_ATTN_TMAX) — 위반 시 전 블록 조기복귀(기록 없음).
#define EXL3_ATTN_TMAX 8
#define EXL3_ATTN_SCAP 1024
// 위치축 상한(plans/cuda-port.md S9): sarr는 공유메모리이므로 lim > SCAP이면
// sarr[row]가 블록 밖을 넘어간다 — cap 1024를 넘는 KV 캐시로 서빙하면
// pos>=1023에서 illegal address(CUresult 700)로 죽는다. 이전까지는 cap을
// 늘리는 쪽에서 이 경계를 몰랐다. 여기서 조기복귀시켜 "답이 이상해지는"
// 것보다 명확하게 거부한다(조용한 오염 금지) — 호출자는 runtime Err로
//不合格을 받는다(모듈 attn_chain_dev의 cap 사전 검사와 짝).
extern "C" __global__ void exl3_attn_fwd3s(
    const float* __restrict__ qh,    // [T][q_heads*256]
    const float* __restrict__ kc,    // [n_attn*cap][kv_heads*256]
    const float* __restrict__ vc,
    const float* __restrict__ qg,    // [T][q_heads*512] gate 반 사용
    float* __restrict__ outv,        // [T][q_heads*256]
    const unsigned* __restrict__ pp, // [1] pos0 — 디바이스 판독(결함 4호)
    int t_len, int layer, int q_heads, int kv_heads, int cap)
{
    if (t_len > EXL3_ATTN_TMAX) return;   // 도메인 강제: T≤8(깨끗한 거부)
    __shared__ float qs[256];
    __shared__ float sarr[EXL3_ATTN_SCAP];
    __shared__ float reds[256];
    int t = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    int gq = q_heads / kv_heads;          // KV그룹 폭(27B 6 · 35B 8)
    int kh = h / gq;
    float scale = 0.0625f;                // 1/√256
    int lim = (int)pp[0] + t + 1;
    // 공유메모리 sarr 한계를 넘는 위치는 처리 불가 — 조기복귀(plans/cuda-port.md
    // S9). Module이 같은 경계를 runtime Err로 사전 검사하므로 여기서는
    // "조용히 틀린 값"이而非 "명확한 불일치"를 택한다.
    if (lim > EXL3_ATTN_SCAP) return;
    long kv_dim = (long)kv_heads * 256;
    qs[tid] = qh[(long)t * (q_heads * 256) + (long)h * 256 + tid];
    __syncthreads();
    for (int row = tid; row < lim; row += 256) {
        float p = 0.0f;
        for (int d = 0; d < 256; d++)
            p += qs[d] * kc[((long)layer * cap + row) * kv_dim + (long)kh * 256 + d];
        sarr[row] = p * scale;
    }
    __syncthreads();
    float lm = -1e30f;
    for (int i = tid; i < lim; i += 256) lm = fmaxf(lm, sarr[i]);
    reds[tid] = lm;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (tid < st) reds[tid] = fmaxf(reds[tid], reds[tid + st]);
        __syncthreads();
    }
    float gmax = reds[0];
    __syncthreads();
    float ls = 0.0f;
    for (int i = tid; i < lim; i += 256) {
        float e = attn_expf(sarr[i] - gmax);
        sarr[i] = e;
        ls += e;
    }
    reds[tid] = ls;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (tid < st) reds[tid] += reds[tid + st];
        __syncthreads();
    }
    float wsum = reds[0];
    __syncthreads();
    float acc = 0.0f;
    for (int row = 0; row < lim; row++)
        acc += sarr[row] * vc[((long)layer * cap + row) * kv_dim + (long)kh * 256 + tid];
    float g = qg[(long)t * (q_heads * 512) + (long)h * 512 + 256 + tid];
    outv[(long)t * (q_heads * 256) + (long)h * 256 + tid] =
        (acc / wsum) * (1.0f / (1.0f + attn_expf(-g)));
}
// 마커 f3sc

// exl3_pos_bump 직이식(src_exl3.hip L1152-1155) — pp[0] += 1. 캡처
// 그래프 내 pos 전진(h2d 불가 대체 — 결함 16호 정신). 어텐션 행 루프가
// 디바이스 체인으로 남는 결함 4호 계약의 짝.
extern "C" __global__ void exl3_attn_pos_bump(unsigned* __restrict__ pp)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) pp[0] += 1u;
}
