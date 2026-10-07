// llm170 rawcuda DeepSeek-V4 mHC(멀티 하이퍼커넥션)+Sinkhorn 스테이지 커널
// (plans/130 B3, 2026-10-05; 보고서 .omo/evidence/ds4-architecture-report.md §4).
//
// [빌드 계약] 본 파일은 별도 -fmad=false 블록으로 빌드한다
// (scripts/build_cuda.bat — gdn/attn/ew/q4/fn/fn_hc 블록과 동일 계급).
// 이유: 본 스테이지 산술은 core 참조의 "연산별 f32 반올림"을 그대로
// 미러한다 — mix 점적산 acc += x[i]*w[i](cpu.rs ADR-0005 무-FMA mul+add),
// rms_scale 순차 f32 제곱합, Sinkhorn/적용 스칼라 곱-가산쌍. 기본 fmad는
// 이 쌍들을 FFMA로 수축해 2중 반올림을 1중으로 바꾸므로 비트가 어긋난다.
// exp 트윈 ds4_exp_cr 의 __fma_rn 은 올림-정확 FMA라 -fmad=false 와
// 무관하게 융합된다(수축 억제 대상 아님 — fn_hc 계급과 동일).
//
// [산술 원천 — 전부 값 maxdiff 판정의 기준(core) 인용, 워크트리 줄번호
// 2026-10-05]
//   - mHC 스플릿+Sinkhorn: crates/core/src/deepseek4/stages/hc.rs
//     hc_split_sinkhorn L31-112 — pre=σ(mix·scale[0]+base)+eps L52-54 ·
//     post=2σ(mix[4+j]·scale[1]+base[4+j]) L55-58 ·
//     raw[j,k]=mix[8+j·4+k]·scale[2]+base[8+..] L59-64 ·
//     softmax_rows(행최대 분산) L67-78 → +eps L81-84 → /(colsum+eps)
//     L85-95 → 19×{/(rowsum+eps); /(colsum+eps)} L96-112.
//   - hc_pre: hc.rs hc_pre L115-152 — rsqrt=rms_scale(x) 후 믹스에만 곱해
//     정규화(L149-150 *m = acc*rsqrt), y=Σ pre_j·X[j](비정규 원본 X)를
//     f32 누산 후 bf16 경계(L154-160), (y,post,comb) 반환.
//   - hc_post: hc.rs hc_post L155-181 — X'[j]=post_j·F+Σ_k comb[j,k]·X[k],
//     j·k 오름차순 f32, 출력 bf16 경계.
//   - hc_head: hc.rs hc_head L185-214(헤드 변형 fn[4·d]·base[4]·scale[1],
//     pre=σ(acc·rsqrt·scale[0]+base)+eps, y=Σ pre_j·X_j bf16 경계 —
//     frame.rs L173-184 사용).
//   - rms_scale: crates/core/src/deepseek4/ops.rs rms_scale L48-55 —
//     **순수 f32 순차 제곱합**(qwen4exp sq_sum 의 f64 세그먼트 트윈과
//     달리 세그먼트 분할 없음) → 1/sqrt(sum/n+eps). 커널도 1스레드 순차
//     루프로 미러(트리 환원 금지 — 비트 동일 조건).
//   - bf16_round: deepseek4/ops.rs bf16_round L17-22 — RNE 1회 비트 경로.
//   - exp/sigmoid: crates/core/src/ops.rs exp_cr L52-119(f64 fma 호너
//     13단)·sigmoid L133-135(1/(1+exp_cr(-x))) — hc.rs 가 crate::ops
//     경유로 사용하는 바로 그 함수들.
//   - fp32 전용 스테이지(보고서 §9.4 "압축·hc·gate fp32" — hc 파라미터는
//     체크포인트 F32).
//
// [CMP 170HX(sm_80, GA100 70SM, HBM2e ~1.5TB/s) 설계 근거 — plans/124 §0]
//   - 층당·kind당·토큰당 hc fn[24×16384] f32 ≈1.57MB 스트리밍 → HBM2e
//     하한 ~1.0µs. mix GEMV(24출력)·rms_scale(1스레드/토큰)·sinkhorn
//     (1스레드/토큰)은 소형 런치(원장 18호 계급 — 비트동일 순차 누산이
//     우선, k-분할은 sm_80 실측 후 재판정).
//   - 개발기(RTX 4070 SUPER, sm_89)는 정합 검증 호스트일 뿐 — 타이밍
//     판단 근거 아님(속도 칸은 '측정 대기 sm_80').
//
// 독립 컴파일 계약(plans/124 G1): 표준 CUDA C++만 — 외부 의존 없음.

// ── exp_cr 트윈 — crates/core/src/ops.rs exp_cr L52-119 직이식(f64 fma
// 호너 13단). 도메인 가드·k>127 오버플로·2^k 비트 재구성까지 원본과 동일.
// mul_add(호스트)≡__fma_rn(디바이스)≡올림-정확 단일 반올림 → 비트동일
// (assets/exl3_fn_hc.cu hc_exp_cr 과 동일 계급의 리터럴 1:1 직이식).
__device__ __forceinline__ float ds4_exp_cr(float x)
{
    double xd = (double)x;
    if (xd > 88.72) return __int_as_float(0x7f800000);
    if (xd < -103.97) return 0.0f;
    const double inv_ln2 = 1.4426950408889634;   // std LOG2_E 비트동일
    const double ln2_hi = 6.9314718036912382e-01;
    const double ln2_lo = 1.9082149292705877e-10;
    double kd = rint(xd * inv_ln2);
    int k = (int)kd;
    double r = __fma_rn(-kd, ln2_hi, xd);
    r = __fma_rn(-kd, ln2_lo, r);
    double p = 1.0 / 1307674368000.0;           // 1/13!
    p = __fma_rn(p, r, 1.0 / 479001600.0);      // 1/12!
    p = __fma_rn(p, r, 1.0 / 39916800.0);       // 1/11!
    p = __fma_rn(p, r, 1.0 / 3628800.0);        // 1/10!
    p = __fma_rn(p, r, 1.0 / 362880.0);         // 1/9!
    p = __fma_rn(p, r, 1.0 / 40320.0);          // 1/8!
    p = __fma_rn(p, r, 1.0 / 5040.0);           // 1/7!
    p = __fma_rn(p, r, 1.0 / 720.0);            // 1/6!
    p = __fma_rn(p, r, 1.0 / 120.0);            // 1/5!
    p = __fma_rn(p, r, 1.0 / 24.0);             // 1/4!
    p = __fma_rn(p, r, 1.0 / 6.0);              // 1/3!
    p = __fma_rn(p, r, 0.5);
    p = __fma_rn(p, r, 1.0);
    p = __fma_rn(p, r, 1.0);
    if (k > 127) return __int_as_float(0x7f800000);
    double scale = __longlong_as_double((long long)(1023 + k) << 52);
    return (float)(p * scale);
}

// ── sigmoid 트윈 — crates/core/src/ops.rs sigmoid L133-135 그대로.
__device__ __forceinline__ float ds4_sigmoid(float x)
{
    return 1.0f / (1.0f + ds4_exp_cr(-x));
}

// ── bf16 경계 반올림 트윈 — deepseek4/ops.rs bf16_round L17-22.
// RNE 1회: 하위 16비트 절반 이상이면 올림(동점은 lsb 로 짝수).
__device__ __forceinline__ float ds4_bf16_round(float x)
{
    unsigned int b = __float_as_uint(x);
    unsigned int hi = (b >> 16) & 1u;
    return __uint_as_float(((b + 0x7FFFu + hi) >> 16) << 16);
}

// ── 1) mix GEMV — hc.rs hc_pre L143-149 믹스 점적산(비-FMA f32 순차) ──
// mix_raw[t][o] = Σ_i x[t][i]·fn[o,i] — fn 행우선 [n_out][n_in]
// ((2+hc)·hc=24 행, 헤드 변형 4행 — 레이아웃 동일). rsqrt 곱은 아직
// 하지 않는다(core L149-150: 점적산 *후에* 믹스에만 rsqrt 를 곱한다 —
// sinkhorn 커널에서 sc=mix·rsqrt 로 동일 2중 반올림 재현).
// 발사: grid(ceil(n_out/128), t_len) × 128. 1스레드=1출력, k 오름차순
// 순차 누산(cpu.rs matmul 계급 — FMA 수축 없음).
extern "C" __global__ void llm170_ds4_hc_mix_gemv(
    const float* __restrict__ x,    // [t][n_in] 스트림 접합(비정규 원본)
    const float* __restrict__ fn_,  // [n_out][n_in] hc_fn (F32)
    float* __restrict__ mix,        // [t][n_out] 원시 점적산
    int n_in, int n_out)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n_out) return;
    int t = blockIdx.y;
    const float* xr = x + (size_t)t * (size_t)n_in;
    const float* wr = fn_ + (size_t)o * (size_t)n_in;
    float acc = 0.0f;
    for (int i = 0; i < n_in; ++i) acc += xr[i] * wr[i];
    mix[(size_t)t * (size_t)n_out + (size_t)o] = acc;
}

// ── 2) RMS 배율 — deepseek4/ops.rs rms_scale L48-55 미러 ──
// rsqrt[t] = 1/sqrt(Σx²/n + eps) — **순수 f32 순차 제곱합**(세그먼트
// 분할·f64 결합 없음 — qwen4exp sq_sum 트윈과 다른 deepseek4 계약).
// 비트 동일을 위해 1스레드가 전 n_in 을 오름차순으로 누산한다(트리
// 환원 금지). 발사: grid(t_len) × 1.
extern "C" __global__ void llm170_ds4_hc_rms_scale(
    const float* __restrict__ x,   // [t][n_in]
    float* __restrict__ rsqrt,     // [t]
    int n_in, float eps)
{
    int t = blockIdx.x;
    const float* xr = x + (size_t)t * (size_t)n_in;
    float sum = 0.0f;
    for (int i = 0; i < n_in; ++i) sum += xr[i] * xr[i];
    rsqrt[t] = 1.0f / sqrtf(sum / (float)n_in + eps);
}

// ── 3) mHC 스플릿 + Sinkhorn — hc.rs hc_split_sinkhorn L31-112 직이식 ──
// 토큰당 1스레드(스칼라 24입력 → pre[4]·post[4]·comb[16]). 연산열:
//   sc_j   = mix[j]·rsqrt                (hc_pre L149-150 — 정규화는
//                                         믹스에만, 원본 X 는 비정규)
//   pre[j] = σ(sc_j·scale[0]+base[j])+eps            L52-54
//   post[j]= 2σ(sc_{4+j}·scale[1]+base[4+j])         L55-58
//   comb[j·hc+k] = sc_{8+j·4+k}·scale[2]+base[8+j·4+k]  L59-64
//   Sinkhorn: softmax_rows(행최대 분산, f32::max 폴드) L67-78 →
//   +eps L81-84 → /(colsum+eps) L85-95 →
//   (iters-1)×{ /(rowsum+eps); /(colsum+eps) } L96-112 — 순서 고정.
// head_only=1 이면 헤드 변형(hc.rs hc_head L185-214): pre 만 계산
// (mix[0..hc), base[hc], scale[1] — post/comb 미기입).
// 발사: grid(t_len) × 1.
extern "C" __global__ void llm170_ds4_hc_split_sinkhorn(
    const float* __restrict__ mix,   // [t][mix_rows] 원시 점적산
    const float* __restrict__ rsqrt, // [t]
    const float* __restrict__ scale, // [3](본체) / [1](헤드)
    const float* __restrict__ base,  // [24](본체) / [4](헤드)
    float* __restrict__ pre,         // [t][hc]
    float* __restrict__ post,        // [t][hc] (head_only 면 미기입)
    float* __restrict__ comb,        // [t][hc*hc] row-major (동일)
    int hc, int iters, float eps, int head_only,
    int mix_rows)  // 믹스 행 보폭(본체 24 / 헤드 hc — gemv 기입 보폭과 일치)
{
    int t = blockIdx.x;
    const float* mr = mix + (size_t)t * (size_t)mix_rows;
    float r = rsqrt[t];
    float* pr = pre + (size_t)t * (size_t)hc;
    for (int j = 0; j < hc; ++j) {
        float sc = mr[j] * r;                       // hc_pre L149-150
        pr[j] = ds4_sigmoid(sc * scale[0] + base[j]) + eps;   // L52-54
    }
    if (head_only) return;                          // hc_head L198-199
    float* po = post + (size_t)t * (size_t)hc;
    for (int j = 0; j < hc; ++j) {
        float sc = mr[hc + j] * r;
        po[j] = 2.0f * ds4_sigmoid(sc * scale[1] + base[hc + j]);  // L55-58
    }
    float* cb = comb + (size_t)t * (size_t)hc * (size_t)hc;
    for (int j = 0; j < hc; ++j) {
        for (int k = 0; k < hc; ++k) {
            float sc = mr[(2 * hc) + j * hc + k] * r;
            cb[j * hc + k] = sc * scale[2] + base[(2 * hc) + j * hc + k];
        }
    }
    // 1) 행 softmax — 행최대 분산(NEG_INF fold → fmaxf 등가), f32 순차
    //    exp 합, 나눗셈. L67-78.
    for (int j = 0; j < hc; ++j) {
        float* row = cb + j * hc;
        float m = -__int_as_float(0x7f800000);      // f32::NEG_INFINITY
        for (int k = 0; k < hc; ++k) m = fmaxf(m, row[k]);
        float sum = 0.0f;
        for (int k = 0; k < hc; ++k) {
            row[k] = ds4_exp_cr(row[k] - m);
            sum += row[k];
        }
        for (int k = 0; k < hc; ++k) row[k] /= sum;
    }
    // 2) +eps → /(colsum+eps). L81-95.
    for (int i = 0; i < hc * hc; ++i) cb[i] += eps;
    float col[8];
    for (int k = 0; k < hc; ++k) col[k] = 0.0f;
    for (int j = 0; j < hc; ++j)
        for (int k = 0; k < hc; ++k) col[k] += cb[j * hc + k];
    for (int j = 0; j < hc; ++j)
        for (int k = 0; k < hc; ++k) cb[j * hc + k] /= col[k] + eps;
    // 3) (iters-1)× { /(rowsum+eps); /(colsum+eps) }. L96-112.
    float row[8];
    for (int it = 1; it < iters; ++it) {
        for (int j = 0; j < hc; ++j) row[j] = 0.0f;
        for (int j = 0; j < hc; ++j)
            for (int k = 0; k < hc; ++k) row[j] += cb[j * hc + k];
        for (int j = 0; j < hc; ++j)
            for (int k = 0; k < hc; ++k) cb[j * hc + k] /= row[j] + eps;
        for (int k = 0; k < hc; ++k) col[k] = 0.0f;
        for (int j = 0; j < hc; ++j)
            for (int k = 0; k < hc; ++k) col[k] += cb[j * hc + k];
        for (int j = 0; j < hc; ++j)
            for (int k = 0; k < hc; ++k) cb[j * hc + k] /= col[k] + eps;
    }
}

// ── 4) pre 적용 — hc.rs hc_pre L154-160: y = Σ_j pre_j·X[j] (원본 X,
// j 오름차순 f32 누산 0.0 시작 — 0+(-0)=+0 부호 규약까지 미러) 후
// bf16 경계. 발사: grid(ceil(n/256), t_len) × 256.
extern "C" __global__ void llm170_ds4_hc_pre_apply(
    const float* __restrict__ x,    // [t][hc*n] 비정규 원본 잔류
    const float* __restrict__ pre,  // [t][hc]
    float* __restrict__ y,          // [t][n] 층 입력 믹스(bf16 경계)
    int n, int hc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int t = blockIdx.y;
    size_t hcn = (size_t)hc * (size_t)n;
    const float* xr = x + (size_t)t * hcn;
    float acc = 0.0f;
    for (int j = 0; j < hc; ++j) acc += pre[(size_t)t * hc + j] * xr[(size_t)j * n + i];
    y[(size_t)t * (size_t)n + i] = ds4_bf16_round(acc);
}

// ── 5) post 적용 — hc.rs hc_post L155-181: X'[j] = post_j·F +
// Σ_k comb[j,k]·X[k], post 곱은 배정(누산 0 아님 — L164-166 y=pj·f 먼저),
// k 오름차순 가산, j행마다 bf16 경계. 발사: grid(ceil(hc·n/256), t_len)
// × 256(1스레드=1출력 원소, j=idx/n, i=idx%n).
extern "C" __global__ void llm170_ds4_hc_post_apply(
    const float* __restrict__ f,      // [t][n] 서브층 출력
    const float* __restrict__ res,    // [t][hc*n] 잔류(비정규 원본)
    const float* __restrict__ post,   // [t][hc]
    const float* __restrict__ comb,   // [t][hc*hc] row-major
    float* __restrict__ y,            // [t][hc*n] 갱신 잔류(bf16 경계)
    int n, int hc)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = hc * n;
    if (idx >= total) return;
    int t = blockIdx.y;
    int j = idx / n;
    int i = idx - j * n;
    size_t hcn = (size_t)hc * (size_t)n;
    const float* rt = res + (size_t)t * hcn;
    float acc = post[(size_t)t * hc + j] * f[(size_t)t * n + i];  // L164-166
    for (int k = 0; k < hc; ++k)
        acc += comb[(size_t)t * hc * hc + j * hc + k] * rt[(size_t)k * n + i];
    y[(size_t)t * hcn + idx] = ds4_bf16_round(acc);
}
