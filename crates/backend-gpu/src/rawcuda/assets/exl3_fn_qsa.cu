// ── Flash-Next(Qwen4-Expert) QSA 스테이지 CUDA 커널 (plans/124 FND, 2026-10-05) ──
// 산술 원천 = crates/core/src/qwen4exp/stages/qsa.rs(CPU 황금 계약, 635행) +
// crates/core/src/ops.rs(노름·rope·활성 스칼라 산술) — 전 커널이 이 두 파일의
// 연산 순서를 줄 단위로 미러한다(값 maxdiff 판정 — argmax 판정 금지, plans/124 §5).
// 모듈층(qsa_cuda.rs)·검증층(qsa_cuda_probe.rs)과 3층 분리(§5).
//
// [스테이지 구조 — 원천 줄번호]
// - 패스 A k행:    qsa.rs L167-181 — k_prenormed면 그대로 적립(재적용 금지,
//                  L168-171 · layers.rs:170 계약), 아니면 rms_norm+rope(부분회전
//                  n_rot=64, ops.rs rope_head L149-163).
// - 패스 A v행:    qsa.rs L182-183(그대로 적립).
// - 패스 A idx_k:  qsa.rs L184(그대로 적립).
// - 패스 A iq행:   qsa.rs L185-190 — 인덩서 q norm+rope(전체회전 n_rot=idx_dim=128).
// - 블록 키 풀링:  qsa.rs L210-228 — r행 mean-pool(j 오름차순) → /=r → rms_norm
//                  → rope(pos=b·r, 전체회전 128).
// - 패스 B 선택:   qsa.rs L249-307 — 블록 점수 4-전개 dot(L274-289, 양수만
//                  누적 L290-291) · width=min(n_past, top_k+r−1)·n_sel_blocks
//                  공식(L295-297) · select_nth_unstable_by 상위 n_sel(L300) ·
//                  오름차순 정렬(L303-305).
// - 어텐션:        qsa.rs cpu_attn_row L13-64 — dot 순차 누산(L29-34) ×kq_scale
//                  → max-sub → exp → p 오름차순 순차 합(L41-46) → w=e/sum 매
//                  위치 f32 나눗셈·w==0 skip·AV 순차 누산(L47-57) → sigmoid
//                  게이트 곱(L58-63, ops.rs sigmoid L133-136 = 1/(1+exp_cr)).
// - KV/pos 규약:   결함 4호 — 캐시 기록 위치·n_past 산출 전부 pp[0] "디바이스
//                  판독"(발사 인자 아님). pos 전진은 llm170_fn_qsa_pos_bump.
//
// [빌드 계약] 별도 -fmad=false 블록(scripts/build_cuda.bat — exl3_gdn/attn/ew/q4
// 와 동일 노선). 사유: (1) f32 누산 전부(블록 점수 d0..d3 4-전개, 어텐션 dot
// 순차 누산, 풀링 적산, AV acc)이 FFMA 수축되면 곱·합 각 1회 반올림인 코어
// 미러와 비트가 어긋난다. (2) 트랜센던트 트윈은 수축 없는 f64 mul/sub DAG.
// (3) sigmoid의 qsa_exp_cr은 "명시적" fma() intrinsic이라 -fmad=false와 무관하게
// 단일 반올림을 유지한다(코어 exp_cr의 mul_add와 동일 연산 — IEEE fma는 구현
// 무관 비트동일).
//
// [트랜센던트 트윈 계약 — G5/G6 노선 계승, 과제 지정 패턴]
// 초월함수 3종만 등장: (a) rope theta = base.powf(−2p/n_rot)(ops.rs L151, f32
// libm), (b) cos/sin(ops.rs L153, f32 libm), (c) 소프트맥스 exp(qsa.rs L44,
// f32 libm .exp()). 셋 다 장치 libdevice powf/cosf/sinf/expf와 호스트(Windows
// UCRT)가 비트재현 불가(G5 실측: libdevice expf는 3.1M 표본 중 30%가 ±1ulp)라
// 아래처럼 자작 f64 트윈으로 양측(본 .cu ↔ qsa_cuda_probe.rs 오라클 트윈)의
// 비트동일을 계약한다 — DAG·리터럴 절대 변경 금지. 단 sigmoid의 exp_cr은
// 코어가 이미 명시적 f64 fma 호너 DAG(ops.rs L52-86)라 **비트동일 직접 미러**가
// 가능하다(트윈 불필요 — fma() intrinsic이 mul_add와 동일 단일 반올림).
//
// [CMP 170HX(sm_80, GA100 70SM, HBM2e ~1.5TB/s) 설계 규거 — plans/124 §0]
// - norm_rope 계열(k/v/ik/iq/q행): grid (T, 헤드수), 블록 1스레드 — 1스레드가
//   1헤드의 노름 32세그먼트·rope 쌍 순회를 통째로 순차 미러(비트동일 우선,
//   plans/124 §6 수치 진실 계층). 청크 T=2048 기준 q행 49,152블록 = 70SM×
//   32블록/SM(1KB 공유)의 22웨이브 — 직렬 ~700사이클/블록로 ~10µs급, 스테이지
//   지배 항 아님(원장: 어텐션·mm_group). 개발기 4070(sm_89)은 정합 검증 전용.
// - pool: grid (신규 블록수), 블록 1스레드 — 블록당 2KB 순차 판독(블록 키 r행),
//   인접 블록 = 인접 주소 → L2 스트리밍. GA100 관점 대역폭 경로.
// - select: grid (T), 블록 256 — 공유 qr 2KB + sc/rk 16.4KB(QSA_BK_MAX 2052).
//   랭크 스캔은 O(n_blocks²)·결정론(정렬 버그 불가능한 선택 — 검증 모듈의
//   우선 계약). sm_80에서 n_blocks 2048·T 2048 청크 최악 ~33M 레인-반복 =
//   ALU 본드 ~0.1ms급 — 실측 후 bitonic/radix select로 최적화 여부 판단
//   (CMP 도착 전 가속 근거 없음 — plans/124 §0 개기기 타이밍 금지).
// - attn: grid (T, n_head=24), 블록 256(=hd) — 공유 qs 1KB + sl 8.2KB + sw
//   8.2KB ≈ 17.7KB → GA100 164KB/SM 기준 smem 9블록·와프 상한(64/SM·8와프)
//   8블록/SM = 2048스레드 풀점유. dot 위상 1스레드=1위치(256 MAC 순차) +
//   AV 위상 스레드=i(p 오름차순 순차) — 감산/합 순서가 코어 미러 계약이라
//   트리 환원 불가(소프트맥스 합·AV). kc/vc 접근은 완전 coalesce(행 내
//   연속), HBM2e 대역 경로.
#include <cuda_fp16.h>

// ── 미러 트랜센던트(리터럴까지 exl3_attn.cu/attn_cuda_probe.rs 트윈과 동일) ──
// exp: G5 gdn_exp_d와 동일 DAG(차수 7 테일러·2^k 비트 재구성). 도메인 |x|≤128.
__device__ __forceinline__ double qsa_exp_d(double x)
{
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

__device__ __forceinline__ float qsa_expf(float x)
{
    return (float)qsa_exp_d((double)x);
}

// rope theta = base.powf(−2·p/n_rot)(ops.rs L151)의 f64 재구성 — G6 attn_theta의
// n_rot 일반화(64=어텐션 k/q 부분회전 · 128=인덱서 전체회전, 둘 다 2의 거듭제곱
// → e=−2p/n_rot는 f64 정확). ln(1e7) 리터럴은 Rust 트윈과 동일 문자열.
// p=0 → exp(0)=1 → (float)1.0 (코어 powf(1e7,0)=1.0과 동일).
__device__ __forceinline__ float qsa_theta(int p, int n_rot)
{
    double e = -(2.0 * (double)p) / (double)n_rot;
    return (float)qsa_exp_d(16.11809565095832 * e);
}

// sincos f64 트윈 — G6 attn_sincos_d와 동일 DAG(Cody-Waite 2분할 + z⁶ Horner,
// 사분면 n=k&3). 도메인 0 ≤ a ≤ 2^20(rope ang = pos·theta ≤ cap).
__device__ __forceinline__ void qsa_sincos_d(double a, double* co, double* si)
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
    // 사분면 매핑(fdlibm __ieee754_rem_pio2 규약 — a = k·π/2 + r, |r| ≤ π/4):
    // n=1 → (cos,sin) = (−sin r, cos r) · n=3 → (sin r, −cos r).
    // [결함 21호 — FND 발견 2026-10-05] G6 exl3_attn.cu attn_sincos_d의
    // 표는 n=1/n=3이 서로 뒤바뀌어 있다((st,−ct)/(−st,ct) — 쌍둥이 오라클과
    // 같은 표를 써서 자기일치해 발견되지 않았음. 본 파일은 바른 표를 쓰고
    // G6 파일은 원 소유자·리드가 회수한다 — 원장 기록).
    int n = (int)(k & 3LL);
    if (n == 0) { *co = ct; *si = st; }
    else if (n == 1) { *co = -st; *si = ct; }
    else if (n == 2) { *co = -ct; *si = -st; }
    else { *co = st; *si = -ct; }
}

__device__ __forceinline__ float qsa_cosf(float a)
{
    double c, s;
    qsa_sincos_d((double)a, &c, &s);
    return (float)c;
}

__device__ __forceinline__ float qsa_sinf(float a)
{
    double c, s;
    qsa_sincos_d((double)a, &c, &s);
    return (float)s;
}

// exp_cr 비트동일 직접 미러 — ops.rs L52-86(코어 자체가 명시적 f64 fma 호너
// DAG라 트윈이 아니다). round_ties_even ≡ nearbyint(디바이스 기본 RTNE).
// 도메인 가드(±88.72/−103.97/k>127)까지 코어 그대로.
__device__ __forceinline__ float qsa_exp_cr(float x)
{
    double xd = (double)x;
    if (xd > 88.72) return __int_as_float(0x7f800000);
    if (xd < -103.97) return 0.0f;
    const double LN2_HI = 6.931471803691238e-1;
    const double LN2_LO = 1.9082149292705877e-10;
    const double INV_LN2 = 1.4426950408889634;
    double kd = nearbyint(xd * INV_LN2);
    long long k = (long long)kd;
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
    if (k > 127) return __int_as_float(0x7f800000);
    double scale = __longlong_as_double((k + 1023) << 52);
    return (float)(p * scale);
}

// ops.rs sigmoid L133-136 — f32 연산 순서 그대로(1.0f + e 후 나눗셈).
__device__ __forceinline__ float qsa_sigmoid(float x)
{
    return 1.0f / (1.0f + qsa_exp_cr(-x));
}

// ── 노름·rope 헤드 미러(ops.rs sq_sum L11-31 · rms_norm L33-37 · rope_head
// L149-163) — 1스레드가 통째로 순차 미러(세그먼트 경계·결합 순서 보존) ──
__device__ __forceinline__ float qsa_rms_scale(const float* x, int dim, float eps)
{
    // sq_sum: SEG=32 — 세그먼트 f32 순차 누산 → 세그먼트 합 f64 순차 결합.
    const int SEG = 32;
    int chunk = (dim + SEG - 1) / SEG;
    double sum = 0.0;
    for (int u = 0; u < SEG; u++) {
        int lo = u * chunk;
        if (lo >= dim) break;
        int hi = lo + chunk; if (hi > dim) hi = dim;
        float part = 0.0f;
        for (int i = lo; i < hi; i++) part += x[i] * x[i];
        sum += (double)part;
    }
    // rms_norm L34: scale = 1/((sum/len + eps).sqrt() as f32) — sqrt는 f64 IEEE.
    return 1.0f / (float)sqrt(sum / (double)dim + (double)eps);
}

__device__ __forceinline__ void qsa_rope_head(float* head, unsigned pos, int n_rot)
{
    // rope_head L149-163: θ=트윈(위) — 코어는 f32 powf·libm cos/sin(트윈과
    // ≤ulp급 차, 오라클 원장에서 계측 보고). (pos·θ) f32 곱·회전 f64 정확곱
// → f32 1회 반올림 — 이 부분은 코어와 동일 구조(비트동일).
    int half = n_rot / 2;
    for (int p = 0; p < half; p++) {
        float theta = qsa_theta(p, n_rot);
        float angle = (float)pos * theta;
        float c = qsa_cosf(angle);
        float s = qsa_sinf(angle);
        double x0 = (double)head[p], x1 = (double)head[p + half];
        double cf = (double)c, sf = (double)s;
        head[p] = (float)(x0 * cf - x1 * sf);
        head[p + half] = (float)(x0 * sf + x1 * cf);
    }
}

// ── 패스 A: k행(qsa.rs L167-181) — grid (t_len, n_kv), 블록 1스레드 ──
// k_prenormed: 디바이스가 이미 norm+rope 적용한 k — 그대로 적립(재적용 금지,
// L168-171 · layers.rs:170 · AGENTS.md 안티패턴). 아니면 rms_norm+rope(부분회전
// n_rot=64). KV 기록 행 = fi·cap + pp[0] + t(결함 4호: pp[0] 디바이스 판독).
extern "C" __global__ void llm170_fn_qsa_k_rows(
    const float* __restrict__ kk,     // [t][n_kv*hd]
    float* __restrict__ kv_k,         // [n_full*cap][n_kv*hd]
    const float* __restrict__ knw,    // [n_full][hd]
    const unsigned* __restrict__ pp,  // [1] pos0 — 디바이스 판독(결함 4호)
    int fi, int cap, int n_kv, int hd, int n_rot, float eps, int prenormed)
{
    __shared__ float head[256];
    int t = blockIdx.x, h = blockIdx.y;
    unsigned pos = pp[0] + (unsigned)t;
    const float* src = kk + (long)t * (n_kv * (long)hd) + (long)h * hd;
    float* dst = kv_k + (((long)fi * cap + (long)pos) * n_kv) * hd + (long)h * hd;
    if (prenormed) {
        for (int i = 0; i < hd; i++) dst[i] = src[i];
        return;
    }
    for (int i = 0; i < hd; i++) head[i] = src[i];
    float s = qsa_rms_scale(head, hd, eps);
    const float* w = knw + (long)fi * hd;
    for (int i = 0; i < hd; i++) head[i] = head[i] * s * w[i];  // v*scale*g 순서
    qsa_rope_head(head, pos, n_rot);
    for (int i = 0; i < hd; i++) dst[i] = head[i];
}

// ── 패스 A: v행 복사(qsa.rs L182-183) — grid (t_len), 블록 256(순서 무관) ──
extern "C" __global__ void llm170_fn_qsa_v_rows(
    const float* __restrict__ vv,
    float* __restrict__ kv_v,
    const unsigned* __restrict__ pp,
    int fi, int cap, int n_kv, int hd)
{
    int t = blockIdx.x, tid = threadIdx.x;
    long row = (((long)fi * cap + (long)pp[0] + t) * n_kv) * hd;
    for (int i = tid; i < n_kv * hd; i += (int)blockDim.x)
        kv_v[row + i] = vv[(long)t * (n_kv * (long)hd) + i];
}

// ── 패스 A: 인덱서 k행 복사(qsa.rs L184 — raw ik 적립) ──
extern "C" __global__ void llm170_fn_qsa_ik_rows(
    const float* __restrict__ ik,     // [t][idx_dim]
    float* __restrict__ idx_k,        // [n_full*cap][idx_dim]
    const unsigned* __restrict__ pp,
    int fi, int cap, int idx_dim)
{
    int t = blockIdx.x, tid = threadIdx.x;
    long row = ((long)fi * cap + (long)pp[0] + t) * idx_dim;
    for (int i = tid; i < idx_dim; i += (int)blockDim.x)
        idx_k[row + i] = ik[(long)t * idx_dim + i];
}

// ── 패스 A: 인덱서 q행(qsa.rs L185-190) — norm(iqw)+rope(전체회전 128) ──
// grid (t_len, idx_heads), 블록 1스레드. 산출 iqnr은 패스 B 점수의 q측.
extern "C" __global__ void llm170_fn_qsa_iq_rows(
    const float* __restrict__ iq,     // [t][idx_heads*idx_dim]
    float* __restrict__ iqnr,         // [t][idx_heads*idx_dim]
    const float* __restrict__ iqw,    // [n_full][idx_dim]
    const unsigned* __restrict__ pp,
    int fi, int idx_heads, int idx_dim, int n_rot, float eps)
{
    __shared__ float head[128];
    int t = blockIdx.x, h = blockIdx.y;
    unsigned pos = pp[0] + (unsigned)t;
    const float* src = iq + (long)t * (idx_heads * (long)idx_dim) + (long)h * idx_dim;
    for (int i = 0; i < idx_dim; i++) head[i] = src[i];
    float s = qsa_rms_scale(head, idx_dim, eps);
    const float* w = iqw + (long)fi * idx_dim;
    for (int i = 0; i < idx_dim; i++) head[i] = head[i] * s * w[i];
    qsa_rope_head(head, pos, n_rot);   // n_rot=idx_dim=128 전체회전
    float* dst = iqnr + (long)t * (idx_heads * (long)idx_dim) + (long)h * idx_dim;
    for (int i = 0; i < idx_dim; i++) dst[i] = head[i];
}

// ── 블록 키 풀링(qsa.rs L210-228) — grid (n_new), 블록 1스레드 ──
// 완전 블록 b: r개 인덱서 k mean-pool(j 오름차순 · 원소별 f32 순차) → /=r
// → rms_norm(ikw) → rope(pos=b·r, 전체회전 128). 증분: b0=기존 블록 수.
// shift는 음성대조 전용 파라미터(0=프로덕션; ≠0이면 풀링 행이 b·r+shift부터
// 시작 — 잘못된 풀링 결함 재현, 원장 17호 계기 원칙).
extern "C" __global__ void llm170_fn_qsa_pool(
    const float* __restrict__ idx_k,  // [n_full*cap][idx_dim]
    float* __restrict__ idx_bk,       // [n_full*bk_cap][idx_dim]
    const float* __restrict__ ikw,    // [n_full][idx_dim]
    int fi, int cap, int bk_cap, int r, int b0, int n_new,
    int idx_dim, int n_rot, float eps, int shift)
{
    __shared__ float pooled[128];
    int b = b0 + blockIdx.x;
    if (blockIdx.x >= n_new) return;
    const float* src = idx_k + (((long)fi * cap + (long)(b * r + shift)) * idx_dim);
    for (int i = 0; i < idx_dim; i++) {
        float acc = 0.0f;
        for (int j = 0; j < r; j++)
            acc += src[(long)j * idx_dim + i];   // L220-222: j 오름차순 원소별 적산
        pooled[i] = acc;
    }
    for (int i = 0; i < idx_dim; i++) pooled[i] /= (float)r;   // L224
    float s = qsa_rms_scale(pooled, idx_dim, eps);
    const float* w = ikw + (long)fi * idx_dim;
    for (int i = 0; i < idx_dim; i++) pooled[i] = pooled[i] * s * w[i];
    qsa_rope_head(pooled, (unsigned)(b * r), n_rot);           // L226-227
    float* dst = idx_bk + (((long)fi * bk_cap + b) * idx_dim);
    for (int i = 0; i < idx_dim; i++) dst[i] = pooled[i];
}

// ── 패스 B: 블록 점수 + top-k 선택(qsa.rs L249-307) — grid (t_len), 블록 256 ──
// 점수: 4-전개 dot(L274-289, 꼬리 처리 포함) — 양수 헤드분만 누적(L290-291,
// 헤드 순서 0..idx_heads). 선택: rank[b] = #{c | s[c]>s[b] 또는 (동점·c<b)}
// 상위 n_sel — 코어 select_nth_unstable_by(L300)와 점수 완전 유일 시 동일
// 집합, 동점 시 최저 인덱스(프루브가 경계 무동점을 게이트로 강제). 기록은
// 오름차순 인덱스 순(sort_unstable L303-305 미러). sel_delta는 음성대조
// 전용(0=프로덕션; +1이면 top-k off-by-one 결함 재현).
#define QSA_BK_MAX 2052
extern "C" __global__ void llm170_fn_qsa_select(
    const float* __restrict__ iqnr,   // [t][idx_heads*idx_dim]
    const float* __restrict__ idx_bk, // [n_full*bk_cap][idx_dim]
    const unsigned* __restrict__ pp,
    unsigned* __restrict__ sel_blk,   // [t][sel_stride]
    unsigned* __restrict__ sel_cnt,   // [t]
    int fi, int bk_cap, int idx_heads, int idx_dim, int r, int top_k,
    int sel_stride, int sel_delta)
{
    __shared__ float qr[4 * 128];
    __shared__ float sc[QSA_BK_MAX];
    __shared__ int rk[QSA_BK_MAX];
    __shared__ int n_sel_sh;
    int t = blockIdx.x, tid = threadIdx.x;
    int qn = idx_heads * idx_dim;
    for (int i = tid; i < qn; i += (int)blockDim.x)
        qr[i] = iqnr[(long)t * qn + i];
    __syncthreads();
    int pos0 = (int)pp[0];                 // 결함 4호: pos 진실은 장치 판독
    int n_past = pos0 + t + 1;             // L254
    int n_blocks = n_past / r;
    for (int b = tid; b < n_blocks; b += (int)blockDim.x) {
        const float* pk = idx_bk + (((long)fi * bk_cap + b) * idx_dim);
        float bs = 0.0f;
        for (int h = 0; h < idx_heads; h++) {
            const float* qh = qr + h * idx_dim;
            float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
            int i2 = 0;
            for (; i2 + 4 <= idx_dim; i2 += 4) {
                d0 += qh[i2] * pk[i2];
                d1 += qh[i2 + 1] * pk[i2 + 1];
                d2 += qh[i2 + 2] * pk[i2 + 2];
                d3 += qh[i2 + 3] * pk[i2 + 3];
            }
            for (; i2 < idx_dim; i2++) d0 += qh[i2] * pk[i2];
            float dot = (d0 + d1) + (d2 + d3);   // L289 결합 순서
            if (dot > 0.0f) bs += dot;           // L290-291
        }
        sc[b] = bs;
    }
    __syncthreads();
    if (tid == 0) {
        int width = n_past < (top_k + r - 1) ? n_past : (top_k + r - 1);  // L295
        int tail_cnt = n_past - n_blocks * r;
        int nb2 = (width - tail_cnt) / r;        // L296-297
        int n_sel = nb2 < n_blocks ? nb2 : n_blocks;
        n_sel += sel_delta;                      // 음성대조(0=프로덕션)
        if (n_sel < 0) n_sel = 0;
        if (n_sel > n_blocks) n_sel = n_blocks;
        n_sel_sh = n_sel;
        sel_cnt[t] = (unsigned)n_sel;
    }
    __syncthreads();
    int n_sel = n_sel_sh;
    for (int b = tid; b < n_blocks; b += (int)blockDim.x) {
        int rank = 0;
        for (int c = 0; c < n_blocks; c++) {
            if (c == b) continue;
            if (sc[c] > sc[b] || (sc[c] == sc[b] && c < b)) rank++;
        }
        rk[b] = rank;
    }
    __syncthreads();
    for (int b = tid; b < n_blocks; b += (int)blockDim.x) {
        if (rk[b] >= n_sel) continue;
        int idx_rank = 0;
        for (int c = 0; c < b; c++)
            if (rk[c] < n_sel) idx_rank++;
        sel_blk[(long)t * sel_stride + idx_rank] = (unsigned)b;
    }
}

// ── q행 norm+rope(qsa.rs L508-517 — 어텐션 직전 일괄) — grid (t_len, n_head) ──
// qg의 q반 [h·2hd, h·2hd+hd) — rms_norm(qnw)+rope(부분회전 n_rot=64) → qbuf.
// 게이트 반([h·2hd+hd])은 원값 그대로(어텐션 커널이 직독).
extern "C" __global__ void llm170_fn_qsa_q_rows(
    const float* __restrict__ qg,     // [t][n_head*2*hd] q‖gate 인터리브
    float* __restrict__ qbuf,         // [t][n_head*hd]
    const float* __restrict__ qnw,    // [n_full][hd]
    const unsigned* __restrict__ pp,
    int fi, int n_head, int hd, int n_rot, float eps)
{
    __shared__ float head[256];
    int t = blockIdx.x, h = blockIdx.y;
    unsigned pos = pp[0] + (unsigned)t;
    const float* src = qg + (((long)t * n_head + h) * 2) * hd;
    for (int i = 0; i < hd; i++) head[i] = src[i];
    float s = qsa_rms_scale(head, hd, eps);
    const float* w = qnw + (long)fi * hd;
    for (int i = 0; i < hd; i++) head[i] = head[i] * s * w[i];
    qsa_rope_head(head, pos, n_rot);
    float* dst = qbuf + ((long)t * n_head + h) * hd;
    for (int i = 0; i < hd; i++) dst[i] = head[i];
}

// ── 선택 리스트 마스크 GQA+게이트(qsa.rs cpu_attn_row L13-64) ──
// grid (t_len, n_head), 블록 256(=hd). 선택 위치 리스트(sl = qsa_sel_list
// 산출 — 블록 오름차순+테일, L323-358: "마스크 스캔과 산술 순서가 같다")를
// 순회: dot 순차 누산 ×kq_scale → max → exp(트윈) → 합·AV 전부 p 오름차순
// 순차(트리 환원 금지 — 코어 미러 계약) → w==0 skip → sigmoid 게이트 곱.
#define QSA_SEL_CAP 2052
extern "C" __global__ void llm170_fn_qsa_attn(
    const float* __restrict__ qbuf,   // [t][n_head*hd]
    const float* __restrict__ kv_k,   // [n_full*cap][n_kv*hd]
    const float* __restrict__ kv_v,
    const float* __restrict__ qg,     // [t][n_head*2*hd] 게이트 반 직독
    const unsigned* __restrict__ sel_idx,
    const unsigned* __restrict__ sel_off, // [t+1]
    float* __restrict__ out,          // [t][n_head*hd]
    const unsigned* __restrict__ pp,  // 판독 전용(pp[0]=pos0 — sl 산출과 동일 원천)
    int fi, int cap, int n_head, int n_kv, int hd, float kq_scale)
{
    __shared__ float qs[256];
    __shared__ unsigned sl[QSA_SEL_CAP];
    __shared__ float sw[QSA_SEL_CAP];
    __shared__ float red[2];
    int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    int kvh = h / (n_head / n_kv);          // GQA 매핑(L27)
    long kv_dim = (long)n_kv * hd;
    unsigned o0 = sel_off[t], o1 = sel_off[t + 1];
    int np = (int)(o1 - o0);
    qs[tid] = qbuf[((long)t * n_head + h) * hd + tid];
    for (int i = tid; i < np; i += (int)blockDim.x)
        sl[i] = sel_idx[o0 + i];
    __syncthreads();
    // (1) 스코어 — 스레드 보폭 위치 할당, 위치별 dot은 i 순차(L29-34)·×kq_scale.
    for (int pi = tid; pi < np; pi += (int)blockDim.x) {
        long p = sl[pi];
        const float* krow = kv_k + (((long)fi * cap + p) * kv_dim) + (long)kvh * hd;
        float d = 0.0f;
        for (int i = 0; i < hd; i++) d += qs[i] * krow[i];
        sw[pi] = d * kq_scale;
    }
    __syncthreads();
    // (2) max — 유한 도메인에서 순서 무관(값 동일)·스레드0 순차(L37-39 상동).
    if (tid == 0) {
        float m = -1e30f;
        for (int pi = 0; pi < np; pi++) m = fmaxf(m, sw[pi]);
        red[0] = m;
    }
    __syncthreads();
    // (3) exp(트윈) — 원소별. 합은 코어 p 오름차순 순차(L41-46 — 마스크 위치의
    // e=0 기여는 정확 항등원이라 선택 리스트 합과 비트동일).
    for (int pi = tid; pi < np; pi += (int)blockDim.x)
        sw[pi] = qsa_expf(sw[pi] - red[0]);
    __syncthreads();
    if (tid == 0) {
        float s2 = 0.0f;
        for (int pi = 0; pi < np; pi++) s2 += sw[pi];
        red[1] = s2;
    }
    __syncthreads();
    // (4) AV — 스레드=i, pi 오름차순 순차(L47-57). w=e/sum 매 위치 f32 나눗셈
    //     (코어가 위치별 w 계산 후 누산 — 일괄 분모 나눗셈 아님)·w==0 skip.
    float acc = 0.0f;
    for (int pi = 0; pi < np; pi++) {
        float w = sw[pi] / red[1];
        if (w == 0.0f) continue;
        long p = sl[pi];
        acc += w * kv_v[(((long)fi * cap + p) * kv_dim) + (long)kvh * hd + tid];
    }
    // (5) 게이트(L58-63) — sigmoid는 exp_cr 비트동일 미러(위 트윈 블록).
    float g = qg[(((long)t * n_head + h) * 2) * hd + hd + tid];
    out[((long)t * n_head + h) * hd + tid] = acc * qsa_sigmoid(g);
}

// ── pp[0] += dt(qsa.rs 캐시 적립의 pos 전진 — 결함 16호: 캡처 그래프 내
// 전진은 h2d 불가, 커널 증분. G6 exl3_attn_pos_bump의 dt 일반화) ──
extern "C" __global__ void llm170_fn_qsa_pos_bump(unsigned* __restrict__ pp, unsigned dt)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) pp[0] += dt;
}
