// llm170 rawcuda Flash-Next HC(hyper-connection mix) 스테이지 커널
// (plans/124 G001 FNA→FNC, 2026-10-05).
//
// [빌드 계약] 본 파일은 별도 -fmad=false 블록으로 빌드한다
// (scripts/build_cuda.bat — gdn/attn/ew/q4/fn 블록과 동일 계급). 이유:
// hc 스테이지 산술은 core 참조의 "연산별 f32 반올림"을 그대로 미러한다 —
//   - 저랭크 게이트 linear: crates/core/src/matmul/cpu.rs matmul L64-76
//     "FMA 없는 mul+add"(acc += x[i]*w[i] — ADR-0005) → 기본 fmad는
//     acc += x*w 를 FFMA 수축해 2중 반올림을 1중으로 바꾸므로 비트 어긋남.
//   - 게이트/활성화: crates/core/src/ops.rs silu·sigmoid L127-133 공식
//     (f32 add/div/mul 각 1회 반올림).
//   - exp 트랜센던트: ops.rs exp_cr L63-119 의 f64 fma 호너(13단)를
//     __fma_rn(올림-정확 FMA — -fmad 플래그와 무관하게 융합 보장)으로
//     리터럴까지 1:1 직이식. mul_add(호스트)≡__fma_rn(디바이스)≡올림-정확
//     단일 반올림 → 비트동일(G7 ew_exp_d가 fma 를 피해 설계된 것과 달리,
//     FNC 오라클은 core exp_cr 자체이므로 fma 를 그대로 쓴다 — plans/124
//     §6 "수치 진실의 계층: core 참조가 유일 기준").
//
// [산술 원천 — 전부 값 maxdiff 판정의 기준(core) 인용]
//   - grouped RMSNorm: crates/core/src/qwen4exp/stages/hc.rs grouped_rms
//     L14-21 — [hc·n] 행을 스트림별로 절단, 각각 ops.rs rms_norm.
//   - sq_sum 32세그먼트: ops.rs sq_sum L11-31 — 세그먼트 f32 순차 누산 →
//     세그먼트 f64 순차 결합. 커널은 블록(32스레드)=1세그먼트씩 소유해
//     이 순서를 정확히 재현(트리 환원 아님 — G3 norm 계급과 달리 비트
//     동일을 위해 세그먼트 구조 자체를 미러. n=2560 → chunk=80 정분할).
//   - 저랭크 게이트: stages/hc.rs hc_mix_ex L25-86 — down(+inject) →
//     silu(lo/hc) L58-61 → up → xn·sigmoid(gate) L66-71 → 스트림 평균
//     /=hc L72-79. 토큰축 배치(전 토큰 1회 체인)는 hc.rs 헤드 원장
//     (토큰당 288회 왕복 병목 → 층당 6회 고정).
//   - linear 적산: cpu.rs matmul L70-73 — 행별 f32 순차 누산(스레드
//     분할과 무관하게 행 내 순서 고정 → 배치·순차 동일 비트, h 결함
//     §4.11 회피 설계의 FNC분).
//
// [CMP 170HX(sm_80, GA100 70SM, HBM2e ~1.5TB/s) 설계 규거 — plans/124 §0]
//   - 층당·토큰당 hc 비용은 down[320×10240]+up[10240×320]+inject[4×10240]
//     f32 가중 스트리밍 ≈ 26.4MB → HBM2e 하한 ~17.6µs/(kind·token).
//     그리드: up gemv (80, T)블록×128 — T=1 에도 70SM 1웨이브 근접.
//     down gemv (3, T)·inject (1, T)은 소형 런치(원장 18호 계급 —
//     k-분할 부분합 확장은 sm_80 실측 후 재판정, 현재는 정합 우선).
//   - grouped_rms (T, 4)블록×32스레드: 행 10KB×4 스트리밍 — 대역폭
//     지배, f64 결합은 블록당 32원소로 무시 가능.
//   - 개발기(RTX 4070 SUPER, sm_89)는 정합 검증 호스트일 뿐 — 타이밍
//     판단 근거 아님(plans/124 §0 계약, 속도 칸은 '측정 대기 sm_80').
//
// 독립 컴파일 계약(plans/124 G1): 표준 CUDA C++만 — 외부 의존 없음.

// ── exp_cr 미러 — ops.rs exp_cr L63-119 직이식(f64 fma 호너 13단) ──
// 도메인 가드·k>127 오버플로·2^k 비트 재구성까지 원본과 동일.
// rint = roundTiesEven(Rust round_ties_even 미러 — 기본 RN 모드).
__device__ __forceinline__ float hc_exp_cr(float x)
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

// ── grouped RMSNorm — stages/hc.rs grouped_rms L14-21 + ops.rs L11-37 ──
// 발사: grid(t_len, hc) × 블록 32. 블록 = (토큰 t, 스트림 s) 1개 —
// 스레드 u 는 sq_sum 세그먼트 u 를 소유(chunk=ceil(n/32), f32 순차 누산,
// 요소 순서 core 와 동일). 세그먼트 f64 부분합을 공유배열에 적고
// 스레드 0 이 u=0..31 순서로 결합(core sum += part as f64 순서)한 뒤
// scale = 1/(float)sqrt(sum/n + eps) 를 전파, 각 스레드가 자기 청크에
// xn = x·scale·w 기록(두 f32 곱, FMA 수축 없음).
// w 는 스트림 s 슬라이스 w[s·n, (s+1)·n) — 원값 감마(qwen4exp 저장 규약).
extern "C" __global__ void llm170_fn_hc_grouped_rms(
    const float* __restrict__ x,   // [t][hc*n] 잔류(스트림-메이저)
    const float* __restrict__ w,   // [hc*n] 감마
    float* __restrict__ xn,        // [t][hc*n]
    int n, int hc, float eps)
{
    int t = blockIdx.x;
    int s = blockIdx.y;
    size_t hcn = (size_t)hc * (size_t)n;
    const float* xr = x + (size_t)t * hcn + (size_t)s * n;
    const float* wr = w + (size_t)s * n;
    float* orow = xn + (size_t)t * hcn + (size_t)s * n;
    __shared__ double seg[32];
    __shared__ float s_scale;
    int u = threadIdx.x;
    int chunk = (n + 31) / 32;
    int lo = u * chunk;
    float part = 0.0f;
    if (lo < n) {
        int hi = min(lo + chunk, n);
        for (int i = lo; i < hi; ++i) part += xr[i] * xr[i];
    }
    seg[u] = (double)part;
    __syncthreads();
    if (u == 0) {
        double sum = 0.0;
        for (int i = 0; i < 32; ++i) sum += seg[i];
        s_scale = 1.0f / (float)sqrt(sum / (double)n + (double)eps);
    }
    __syncthreads();
    float scale = s_scale;
    if (lo < n) {
        int hi = min(lo + chunk, n);
        for (int i = lo; i < hi; ++i) orow[i] = xr[i] * scale * wr[i];
    }
}

// ── f32 배치 GEMV — cpu.rs matmul L64-76 미러(비트동일 목표) ──
// out[t][o] = Σ_i x[t][i]·W[o,i] — W 행우선 [n_out][n_in](ggml ne0=n_in
// 규약, hc down/up/inject 패치 텐서와 동일 레이아웃). 발사:
// grid(ceil(n_out/128), t_len) × 128 — t=blockIdx.y(결함 5호 계약:
// gy 로 토큰축, T>1 행 미실행 가드). 1스레드=1출력, f32 순차 누산
// (FMA 수축 없음 — cpu.rs ADR-0005 "mul+add" 미러). 소형 n_out(down 320·
// inject 4)의 점유 한계는 원장 18호 계급 — k-분할은 sm_80 실측 후 재판정.
extern "C" __global__ void llm170_fn_hc_gemv(
    const float* __restrict__ x,   // [t][n_in]
    const float* __restrict__ w,   // [n_out][n_in]
    float* __restrict__ out,       // [t][n_out]
    int n_in, int n_out)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n_out) return;
    int t = blockIdx.y;
    const float* xr = x + (size_t)t * (size_t)n_in;
    const float* wr = w + (size_t)o * (size_t)n_in;
    float acc = 0.0f;
    for (int i = 0; i < n_in; ++i) acc += xr[i] * wr[i];
    out[(size_t)t * (size_t)n_out + (size_t)o] = acc;
}

// ── silu(lo/hc) — stages/hc.rs L58-61 + ops.rs silu L127-130 ──
// v = lo/hc(f32 나눗셈 — hc=4 는 2의 거듭제곱이라 정확), 제자리 silu.
extern "C" __global__ void llm170_fn_hc_silu_scaled(
    float* __restrict__ lo, int n, float hc_f)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = lo[i] / hc_f;
    lo[i] = v / (1.0f + hc_exp_cr(-v));
}

// ── 게이트 적용 + 스트림 평균 — stages/hc.rs L66-79 ──
// 발사: grid(ceil(n/256), t_len) × 256. 스레드 = (토큰 t, 열 i):
// s=0..hc 순서로 g_s = xn·sigmoid(gate_s)(f32 곱) → m += g_s(순차 f32
// 가산 — core m[i] += gate[s·n+i] 와 동일 순서·동일 피가산 비트) →
// mixed = m/hc. gate 버퍼에는 게이트 적산값 g_s 을 남긴다(진단 판독용).
extern "C" __global__ void llm170_fn_hc_gate_mean(
    const float* __restrict__ xn,   // [t][hc*n]
    float* __restrict__ gate,       // [t][hc*n] in/out
    float* __restrict__ mixed,      // [t][n]
    int n, int hc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int t = blockIdx.y;
    size_t hcn = (size_t)hc * (size_t)n;
    const float* xr = xn + (size_t)t * hcn;
    float* gr = gate + (size_t)t * hcn;
    float m = 0.0f;
    for (int s = 0; s < hc; ++s) {
        // core: *g = *gi * sigmoid(*g) — g 는 gate 값, gi 는 xn 값.
        float g = gr[(size_t)s * n + i];
        float sig = 1.0f / (1.0f + hc_exp_cr(-g));
        float v = xr[(size_t)s * n + i] * sig;
        gr[(size_t)s * n + i] = v;
        m += v;
    }
    mixed[(size_t)t * (size_t)n + i] = m / (float)hc;
}
