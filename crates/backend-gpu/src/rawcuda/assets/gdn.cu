// ── GDN 체인 CUDA 포팅 (G5, 2026-10-04) ──
// 산술은 구 rawhip 커널의 gdn_conv(L330-365)·
// gdn_gate(L368-393)·gdn_l2perm(L397-472)·gdn_scan
// (L484-608)을 1:1 직이식한다(원본 그대로 베낌).
// 원본과의 차이는 4점(1-3 아래, 4번은 #include 직후 트랜센던트 블록):
// 1) hip 판이 27B 폭(10240/2048/6144/5120/96/48헤드)으로 경직된 상수를
//    런치 인자(k_len/v_len/conv_ch/hidden/h_k/h_v)로 일반화 — 27B 값
//    (hidden=5120, h_k=16, h_v=48)을 넣으면 원소 순서까지 원본과 동일.
//    목적은 35B-A3B GDN 형상 지원(hidden=2048, h_k=16, h_v=32,
//    conv_ch=8192 — 실측 형상 config.json
//    text_config 실측 2026-10-04).
// 2) scan 공유메모리 합계 61,828B(sk/sv/KS/QS 8KB×4 · A/KQ 2KB×2 ·
//    dc 16KB · Stile 8KB · bp/gcs/wsm 388B)는 CUDA 정적 __shared__ 한계
//    48KB를 초과한다(ROCm LDS 64KB에서는 상주 가능했던 배치) → 동적
//    공유메모리(extern __shared__ + cuFuncSetAttribute opt-in)로 동일
//    배치·동일 산술로 이식. 배치 순서·f16 저장 지점은 불변.
// 3) conv 원본의 자리표시 행(o = silu(xt)·0+0 — 무효 산출, 원본 주석
//    "자리표시 — 아래 정정")은 제거. 살아있는 산술만 이식.
// 4) 트랜센던트/수축(위 블록 참조): 빌드 -fmad=false(FMA 수축 제거 —
//    호스트 미러와 연산 DAG 비트동일), 초월함수는 자작 f64 미러,
//    qscale은 IEEE sqrt+div. 전부 (hip은 참조 구현일 뿐
//    진실 아님 — CUDA가 코어/미러에 더 가까운 쪽이 옳음)에 근거한
//    정밀화. 수학 동등·정밀도 상향, 값 변화는 ulp 수준.
//
// lc 순열 계약(§3.3 — 방향 혼동으로 사고 1회): HF v헤드 h는 [h_k][G]
// k-주요 배치, lc(FLA)는 [G][h_k] 그룹-주요 — 전치 p_inv=(h%G)·h_k+h/G,
// G=h_v/h_k(27B: G=3 → (h%3)·16+h/3 — 원본식과 동일). l2perm·gate 모두
// scatter(쓰기측 인덱스에 p_inv) — CPU 미러가 gather라 방향 혼동 주의.
// gdn_l2perm_gather는 원장 17호 음성대조 전용 쌍둥이(방향 반전
// 결함 재현) — 프로덕션 경로에서 발사 금지.
//
// 그리드 계약(결함 5호): T>1 커널(l2perm/gate)은 t=blockIdx.y —
// grid (h_v, T). gy=1로 두면 행 1+가 미실행된다(과거 사고).
//
// [CMP 170HX(sm_80, GA100 70SM, HBM2e ~1.5TB/s) 설계 근거]
// - conv: 순수 스트리밍(채널축 완전 병렬, 그리드 conv_ch/128 = 27B 80·
//   35B 64블록, 128스레드가 128연속 채널 → 완전 coalesce). T행은 링
//   의존이라 순차(런치 1회로 T행 전체 순회 — §3.3 계약). w0-w3·h0-h2는
//   레지스터 상주(스레드당 f32 7개) — 소형 레지스터로 다중 블록/SM
//   상주, GA100에서 80블록 즉시 분배·스트리밍 병렬성 확보. 행 의존
//   루프라 ILP는 탭 간 덧셈 트리(w3·x+w0·h0+w1·h1+w2·h2 결합 순서는
//   산술 계약이라 불변).
// - l2perm: 그리드 (h_v, T) — 27B T=32면 1536블록으로 70SM 포화.
//   지배 비용은 abuf [L][2][h_v][hidden] 판독(27B 층당 1.97MB) — 같은
//   h의 T개 블록이 동일 abuf 슬라이스를 재판독하므로 L2(T축 재사용)
//   로 흡수된다. WG=128·red[128] 트리 환원은 산술 계약(환원 순서
//   미러 대상)이라 유지.
// - scan: 청크 순차(상태 S 의존 — 청크 c+1은 c의 상태 갱신 후에만
//   실행 가능)라 헤드(h_v블록) 이상의 병렬화는 계약상 불가. 동적 공유
//   61,828B → GA100 164KB/SM 기준 2블록/SM 상한이나 그리드 h_v(27B
//   48 < 70SM)라 실질 1블록/SM×48SM 사용 — 목표기 실측 전까지 유지
//   (D열 2분할은 A/KQ 전치점(전체 128내적) 중복 연산을 유발해 손익
//   미확정). A/KQ·KS/QS 페이즈의 스레드 활용(32/128)은 원본 구조
//   1:1 — 정합 우선, SASS 수준 최적화는 170HX 도착 후 판단.
// - gate: 그리드 (h_v, T), 블록 128 — 메모리 본드 소형, 특기 사항 없음.
#include <cuda_fp16.h>

// ── 미러 트랜센던트(G5 정밀화) ──
// 계약: 장치 초월함수(libdevice expf 2ulp·__expf/__logf fast)는 호스트
// 오라클과 비트재현 불가 — f16 저장 경계 교차(실측 2026-10-04: 27B
// 실가중 픽스처에서 스캔 단계 7.6e-6 → 게이트 rms 증폭 ×289로 종단
// 3.3e-4, 임계 2e-4 초과)를 유발한다. 본 파일의 트랜센던트는 순수 f64
// 연산 DAG(IEEE mul/add/div/floor·비트 재구성만, FMA 수축 없음 — 빌드
// -fmad=false)로 자작해 양측(본 .cu ↔ 구 프로브 오라클 트윈)의
// 비트동일을 계약으로 삼는다. 정확도는 f64 다항 근사(~1e-12 상대 —
// f32 캐스트 기준 참값과 수 ulp)로 수학 동등·정밀도 상향. 도메인:
// exp |x| ≤ 128(k 비트 재구성 상한), log y ≥ 2^-1022 정규수(softplus
// 인자 ≥ 1 — 실사용 영역). 연산 순서·상수는 절대 변경 금지(비트동일
// 계약 — Rust 트윈과 리터럴까지 동일해야 한다).
__device__ __forceinline__ double gdn_exp_d(double x)
{
    // k = floor(x·invln2 + ½)(양수 음수 공용 반올림) → r = x − k·ln2(hi/lo)
    // → 테일러 차수 7 → 2^k 비트 재구성.
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

__device__ __forceinline__ float gdn_expf(float x)
{
    return (float)gdn_exp_d((double)x);
}

__device__ __forceinline__ double gdn_log_d(double y)
{
    // m·2^e 규약(m ∈ [1,2), √2 초과 시 반감 → [√½,√2)) → atanh 급수
    // z¹¹차 → + e·ln2.
    const double ln2 = 6.9314718036912382e-01;
    unsigned long long bits = __double_as_longlong(y);
    int e = (int)((bits >> 52) & 0x7ff) - 1023;
    unsigned long long mb = (bits & 0x800fffffffffffffULL) | 0x3ff0000000000000ULL;
    double m = __longlong_as_double((long long)mb);
    if (m > 1.4142135623730951) { m = m * 0.5; e = e + 1; }
    double s = (m - 1.0) / (m + 1.0);
    double z = s * s;
    double q = 1.0 + z * (0.3333333333333333 + z * (0.2 + z * (0.14285714285714285
        + z * (0.1111111111111111 + z * (0.09090909090909091 + z * (0.07692307692307693
        + z * (0.06666666666666667 + z * (0.058823529411764705 + z * (0.05263157894736842
        + z * (0.047619047619047616 + z * 0.043478260869565216))))))))));
    return 2.0 * s * q + (double)e * ln2;
}

__device__ __forceinline__ float gdn_logf(float y)
{
    return (float)gdn_log_d((double)y);
}

// gdn_conv — 채널별 3탭 링 순차 회전(구 rawhip 커널 L330-365 직이식,
// 폭 인자화). 그리드 (conv_ch/128, 1), 블록 128. 링 [L][3][conv_ch]은
// 커널이 r/w — T행 전체를 한 런치에서 순회(§3.3).
extern "C" __global__ void gdn_conv(
    const float* __restrict__ qkv,   // [T][conv_ch]
    const float* __restrict__ convw, // [L][conv_ch][4]
    float* __restrict__ ring,        // [L][3][conv_ch]
    float* __restrict__ q_out,       // [T][k_len]
    float* __restrict__ k_out,       // [T][k_len]
    float* __restrict__ v_out,       // [T][v_len]
    int t_len, int layer, int k_len, int v_len, int conv_ch)
{
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= conv_ch) return;
    float w0 = convw[layer * conv_ch * 4 + ch * 4 + 0];
    float w1 = convw[layer * conv_ch * 4 + ch * 4 + 1];
    float w2 = convw[layer * conv_ch * 4 + ch * 4 + 2];
    float w3 = convw[layer * conv_ch * 4 + ch * 4 + 3];
    float h0 = ring[layer * 3 * conv_ch + 0 * conv_ch + ch];
    float h1 = ring[layer * 3 * conv_ch + 1 * conv_ch + ch];
    float h2 = ring[layer * 3 * conv_ch + 2 * conv_ch + ch];
    for (int t = 0; t < t_len; t++) {
        float xt = qkv[t * conv_ch + ch];
        float o = (w3 * xt + w0 * h0 + w1 * h1 + w2 * h2);
        o = o / (1.0f + gdn_expf(-o));
        if (ch < k_len) {
            q_out[t * k_len + ch] = o;
        } else if (ch < 2 * k_len) {
            k_out[t * k_len + (ch - k_len)] = o;
        } else {
            v_out[t * v_len + (ch - 2 * k_len)] = o;
        }
        h0 = h1; h1 = h2; h2 = xt;
    }
    ring[layer * 3 * conv_ch + 0 * conv_ch + ch] = h0;
    ring[layer * 3 * conv_ch + 1 * conv_ch + ch] = h1;
    ring[layer * 3 * conv_ch + 2 * conv_ch + ch] = h2;
}

// l2perm 본체 — a/b 도트(xn·abuf) + q/k L2 + v·beta|g lc 순열
// (구 rawhip 커널 L397-472 직이식, 폭 인자화). 그리드 (h_v, T), WG=128.
// GATHER=true는 음성대조 전용 방향 반전(v 판독측에 p_inv — 결함류:
// 방향). beta|g는 lc 순열로 scatter(bg[.. + p_inv]).
template <bool GATHER>
__device__ __forceinline__ void gdn_l2perm_body(
    const float* __restrict__ q_in,   // [T][k_len]
    const float* __restrict__ k_in,   // [T][k_len]
    const float* __restrict__ v_in,   // [T][h_v*128] HF
    const float* __restrict__ xn,     // [T][hidden] (xtb)
    const float* __restrict__ abuf,   // [L][2][h_v][hidden]
    const float* __restrict__ alog,   // [L][h_v]
    const float* __restrict__ dtb,    // [L][h_v]
    float* __restrict__ q_out,        // [T][k_len]
    float* __restrict__ k_out,        // [T][k_len]
    float* __restrict__ v_out,        // [T][h_v*128] lc
    float* __restrict__ bg,           // [T][2*h_v] beta|g lc
    int t_len, int layer, int h_k, int h_v, int hidden)
{
    __shared__ float red[128];
    int t = blockIdx.y;
    int h = blockIdx.x;
    int tid = threadIdx.x;
    if (t >= t_len) return;
    const float eps = 1e-6f;

    float pa = 0.0f;
    for (int i = tid; i < hidden; i += 128)
        pa += xn[t * hidden + i] * abuf[(layer * 2) * h_v * hidden + h * hidden + i];
    red[tid] = pa;
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    float a_val = red[0];
    __syncthreads();

    float pb = 0.0f;
    for (int i = tid; i < hidden; i += 128)
        pb += xn[t * hidden + i] * abuf[(layer * 2 + 1) * h_v * hidden + h * hidden + i];
    red[tid] = pb;
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    float b_val = red[0];
    __syncthreads();

    // lc 순열: HF [h_k][G] → lc [G][h_k] 전치. 27B(G=3, h_k=16)이면
    // 원본식 (h%3)*16+h/3와 동일.
    int G = h_v / h_k;
    int p_inv = (h % G) * h_k + h / G;
    if (tid == 0) {
        float al = alog[layer * h_v + h];
        float dt = dtb[layer * h_v + h];
        float ssm_a = -gdn_expf(al);
        float sp = (a_val + dt > 20.0f) ? (a_val + dt) : gdn_logf(1.0f + gdn_expf(a_val + dt));
        bg[t * (2 * h_v) + p_inv] = 1.0f / (1.0f + gdn_expf(-b_val));
        bg[t * (2 * h_v) + h_v + p_inv] = sp * ssm_a;
    }

    if (h < h_k) {
        int kh = h;
        float qv = q_in[t * (h_k * 128) + kh * 128 + tid];
        red[tid] = qv * qv;
        __syncthreads();
        for (int st = 64; st > 0; st >>= 1) {
            if (tid < st) red[tid] += red[tid + st];
            __syncthreads();
        }
        float qi = 1.0f / sqrtf(red[0] + eps);
        __syncthreads();
        q_out[t * (h_k * 128) + kh * 128 + tid] = qv * qi;

        float kv = k_in[t * (h_k * 128) + kh * 128 + tid];
        red[tid] = kv * kv;
        __syncthreads();
        for (int st = 64; st > 0; st >>= 1) {
            if (tid < st) red[tid] += red[tid + st];
            __syncthreads();
        }
        float ki = 1.0f / sqrtf(red[0] + eps);
        __syncthreads();
        k_out[t * (h_k * 128) + kh * 128 + tid] = kv * ki;
    }
    if (GATHER) {
        // 음성대조: gather 방향(원장 — CPU 미러와 같은 쪽).
        v_out[t * (h_v * 128) + h * 128 + tid] =
            v_in[t * (h_v * 128) + p_inv * 128 + tid];
    } else {
        // 프로덕션: scatter 방향(계약).
        v_out[t * (h_v * 128) + p_inv * 128 + tid] =
            v_in[t * (h_v * 128) + h * 128 + tid];
    }
}

// 프로덕션 l2perm — scatter(계약 방향).
extern "C" __global__ void gdn_l2perm(
    const float* __restrict__ q_in,
    const float* __restrict__ k_in,
    const float* __restrict__ v_in,
    const float* __restrict__ xn,
    const float* __restrict__ abuf,
    const float* __restrict__ alog,
    const float* __restrict__ dtb,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    float* __restrict__ bg,
    int t_len, int layer, int h_k, int h_v, int hidden)
{
    gdn_l2perm_body<false>(q_in, k_in, v_in, xn, abuf, alog, dtb,
                           q_out, k_out, v_out, bg, t_len, layer, h_k, h_v, hidden);
}

// 음성대조 전용 l2perm — gather(방향 결함 재현, 원장 17호 계기).
extern "C" __global__ void gdn_l2perm_gather(
    const float* __restrict__ q_in,
    const float* __restrict__ k_in,
    const float* __restrict__ v_in,
    const float* __restrict__ xn,
    const float* __restrict__ abuf,
    const float* __restrict__ alog,
    const float* __restrict__ dtb,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    float* __restrict__ bg,
    int t_len, int layer, int h_k, int h_v, int hidden)
{
    gdn_l2perm_body<true>(q_in, k_in, v_in, xn, abuf, alog, dtb,
                          q_out, k_out, v_out, bg, t_len, layer, h_k, h_v, hidden);
}

// gdn_scan — FLA 청크 알고리즘(구 rawhip 커널 L484-608 직이식,
// CS=32·TILE=16·8패스). A/KQ/KS/QS/sk/sv f16, dc f32 전진대입 소거
// (d[i] = β·(v_i − e^{g_i}·KS_i) 먼저, 이후 j<i 감산 — 순서 계약),
// o는 각 i의 소거 직후(필요한 dc[p≤i]는 전부 확정) 산출. 상태 P6
// 갱신. q 입력(q2)은 l2perm에서 이미 L2 정규화됨 — 이 커널에서 중복
// 스케일 금지(§3.3; qscale=1/√d는 어텐션 스케일이지 재정규화 아님).
// 공유메모리는 동적 61,828B(최상단 주석 2항). 그리드 (h_v, 1), WG=128.
#define GDN_CS 32
#define GDN_TILE 16
extern "C" __global__ void gdn_scan(
    const float* __restrict__ q,     // [T][k_len]
    const float* __restrict__ k,     // [T][k_len] L2
    const float* __restrict__ v,     // [T][h_v*128] lc
    const float* __restrict__ bg,    // [T][2*h_v]
    float* __restrict__ st,          // [L][h_v][128*128] r/w — S0≠0 경로 의무
    float* __restrict__ outv,        // [T][h_v*128] o_lc
    int t_len, int h_k, int h_v, int d, int layer)
{
    extern __shared__ char smem_raw[];
    __half* sk = (__half*)smem_raw;                    // [CS*128]
    __half* sv = sk + GDN_CS * 128;                   // [CS*128]
    __half* A = sv + GDN_CS * 128;                    // [CS*CS]
    __half* KQ = A + GDN_CS * GDN_CS;                // [CS*CS]
    __half* KS = KQ + GDN_CS * GDN_CS;               // [CS*128]
    __half* QS = KS + GDN_CS * 128;                   // [CS*128]
    float* dc = (float*)(QS + GDN_CS * 128);          // [CS*128] — 4B 정렬(오프셋 36864)
    float* Stile = dc + GDN_CS * 128;                 // [TILE*128]
    float* bp = Stile + GDN_TILE * 128;               // [CS]
    float* gcs = bp + GDN_CS;                         // [CS+1]
    float* wsm = gcs + (GDN_CS + 1);                  // [CS]
    int h = blockIdx.x;
    int kh = h % h_k;
    int tid = threadIdx.x;
    int n_chunks = (t_len + GDN_CS - 1) / GDN_CS;
    // G5 정밀화: rsqrtf(≤2ulp 근사) 대신 IEEE sqrt+div — 호스트 미러와
    // 비트동일(양측 sqrt.rn·div.rn).
    float qscale = 1.0f / sqrtf((float)d);
    long st_h = (long)layer * h_v * d * d + (long)h * d * d;

    for (int c = 0; c < n_chunks; c++) {
        int t0 = c * GDN_CS;
        int n = min(t_len - t0, GDN_CS);

        for (int e = tid; e < GDN_CS * 128; e += 128) {
            int t = e / 128, dv = e % 128;
            bool live = t < n;
            sk[e] = __float2half_rn(live ? k[(t0 + t) * (h_k * d) + kh * 128 + dv] : 0.0f);
            sv[e] = __float2half_rn(live ? v[(t0 + t) * (h_v * d) + h * 128 + dv] : 0.0f);
        }
        if (tid < GDN_CS) {
            float acc = 0.0f;
            for (int t = 0; t < GDN_CS; t++) {
                acc += (t < n) ? bg[(t0 + t) * (2 * h_v) + h_v + h] : 0.0f;
                gcs[t] = acc;
            }
            gcs[GDN_CS] = acc;
            bp[tid] = (tid < n) ? bg[(t0 + tid) * (2 * h_v) + h] : 0.0f;
        }
        __syncthreads();

        for (int i = 0; i < GDN_CS; i++) {
            if (tid < GDN_CS && i < n) {
                int j = tid;
                float dk = 0.0f, dq = 0.0f;
                int qbase = (t0 + i) * (h_k * d) + kh * 128;
                for (int s2 = 0; s2 < 128; s2++) {
                    float kj = __half2float(sk[j * 128 + s2]);
                    dk += __half2float(sk[i * 128 + s2]) * kj;
                    dq += q[qbase + s2] * kj;
                }
                float bi = bp[i];
                A[i * GDN_CS + j] = __float2half_rn((j < i) ? dk * bi * gdn_expf(gcs[i] - gcs[j]) : 0.0f);
                KQ[i * GDN_CS + j] = __float2half_rn((j <= i) ? dq * qscale * gdn_expf(gcs[i] - gcs[j]) : 0.0f);
            }
        }
        __syncthreads();

        for (int e = tid; e < GDN_CS * 128; e += 128) {
            KS[e] = __float2half_rn(0.0f);
            QS[e] = __float2half_rn(0.0f);
        }
        __syncthreads();
        for (int pass_ = 0; pass_ < 8; pass_++) {
            int s2b = pass_ * GDN_TILE;
            for (int s2p = 0; s2p < GDN_TILE; s2p++)
                Stile[s2p * 128 + tid] = st[st_h + (long)(s2b + s2p) * d + tid];
            __syncthreads();
            for (int i = 0; i < GDN_CS; i++) {
                float ak = 0.0f, aq = 0.0f;
                int qbase = (t0 + i) * (h_k * d) + kh * 128;
                for (int s2p = 0; s2p < GDN_TILE; s2p++) {
                    float s_el = Stile[s2p * 128 + tid];
                    ak += __half2float(sk[i * 128 + s2b + s2p]) * s_el;
                    // T=1이면 i=1..31의 q 행은
                    // 할당되지 않는다. 프로브 T=32에서 숨었던 CUresult=700;
                    // 비활성 행만 0으로 마스킹(활성 산술 순서 불변).
                    float qv = (i < n) ? q[qbase + s2b + s2p] : 0.0f;
                    aq += qv * s_el;
                }
                int ib = i * 128 + tid;
                KS[ib] = __float2half_rn(__half2float(KS[ib]) + ak);
                QS[ib] = __float2half_rn(__half2float(QS[ib]) + aq * qscale);
            }
            __syncthreads();
        }

        for (int i = 0; i < n; i++) {
            float rhs = bp[i] * (__half2float(sv[i * 128 + tid]) - gdn_expf(gcs[i]) * __half2float(KS[i * 128 + tid]));
            for (int j = 0; j < i; j++) {
                float aij = __half2float(A[i * GDN_CS + j]);
                if (aij != 0.0f) rhs -= aij * dc[j * 128 + tid];
            }
            dc[i * 128 + tid] = rhs;
            float oi = gdn_expf(gcs[i]) * __half2float(QS[i * 128 + tid]);
            for (int p = 0; p <= i; p++) {
                float w = __half2float(KQ[i * GDN_CS + p]);
                if (w != 0.0f) oi += w * dc[p * 128 + tid];
            }
            outv[(t0 + i) * (h_v * d) + h * 128 + tid] = oi;
        }
        __syncthreads();

        {
            float gtot = gcs[GDN_CS];
            float gt_exp = gdn_expf(gtot);
            if (tid < GDN_CS) wsm[tid] = (tid < n) ? gdn_expf(gtot - gcs[tid]) : 0.0f;
            __syncthreads();
            for (int s2 = 0; s2 < 128; s2++) {
                float acc = st[st_h + (long)s2 * d + tid] * gt_exp;
                for (int j = 0; j < n; j++)
                    acc += __half2float(sk[j * 128 + s2]) * wsm[j] * dc[j * 128 + tid];
                st[st_h + (long)s2 * d + tid] = acc;
            }
            __syncthreads();
        }
    }
}

// gdn_gate — rms(o_lc)·nw·silu(z) → gated(HF), 역순열 포함
// (구 rawhip 커널 L368-393 직이식, 폭 인자화). 그리드 (h_v, T), WG=128.
extern "C" __global__ void gdn_gate(
    const float* __restrict__ o_lc,  // [T][h_v*128] lc
    const float* __restrict__ z,     // [T][h_v*128] HF
    const float* __restrict__ nw,    // [L][128]
    float* __restrict__ gated,       // [T][h_v*128] HF
    int t_len, int layer, int h_k, int h_v)
{
    __shared__ float red[128];
    int t = blockIdx.y;
    int h = blockIdx.x;
    int tid = threadIdx.x;
    if (t >= t_len) return;
    int G = h_v / h_k;
    int p_inv = (h % G) * h_k + h / G;
    float ov = o_lc[t * (h_v * 128) + p_inv * 128 + tid];
    red[tid] = ov * ov;
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    float inv = 1.0f / sqrtf(red[0] / 128.0f + 1e-6f);
    float zv = z[t * (h_v * 128) + h * 128 + tid];
    gated[t * (h_v * 128) + h * 128 + tid] =
        ov * inv * nw[layer * 128 + tid] * (zv / (1.0f + gdn_expf(-zv)));
}
