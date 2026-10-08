// ── EW(silu·mul) + argmax CUDA 포팅 (G7, 2026-10-04) ──
// 산술은 구 rawhip 커널의 ew(L898-908)·ew_argmax
// (L911-934)을 1:1 직이식한다(원본 그대로 베낌).
// 원본과의 차이는 1점(트랜센던트):
// 1) 원본 ew는 __expf(수 ulp 근사 내장) — G5/G6 노선(hip은 참조 구현일
//    뿐 진실 아님)에 따라 silu의 exp를 자작 f64
//    DAG 트윈(ew_exp_d — gdn_exp_d/attn_exp_d와 동일 DAG)으로
//    정밀화한다. 공식 silu(v) = v/(1+exp(−v))의 CPU 참조는
//    crates/core/src/ops.rs silu(L127-130). 빌드는 -fmad=false(FMA 수축
//    제거 — 호스트 미러와 비트동일, build_cuda.bat 별도 블록). 도메인
//    |x|≤128(gdn_exp_d 계약 계승 — sigmoid 포화 |v|≤30 실사용의 4배
//    여유). f32 add/div/mul은 양측 IEEE 정확 반올림 → 비트동일.
//    argmax는 부동소수 환원이 아니라 정수 인덱스 선택(exp 미포함,
//    fmad 무관)이라 차이 없다.
//
// 결함 8호: ew_argmax의 n은 "로짓 길이"(248320 — 어휘 폭),
// 행수가 아니다. 과잉 판독(행수>길이 창 밖)·과소 판독(길이>행수 미
// 스캔)이 고전 버그 — 검증층 음성대조(구 프로브
// cuda_argmax_negative_check)가 잘못된 n을 토큰 불일치로 잡는다.
// 동일값 최대가 여러 개일 때 위너는 "가장 낮은 tid의 잔여 클래스
// (i ≡ tid mod 1024) 내 첫 등장" — 전역 첫 등장과 다를 수 있으나
// 원본 reduction 구조 그대로(1:1 계약). 오라클도 이 규칙을 미러
// 한다(exact-match 계약, 아래 reduction 참조).
//
// [CMP 170HX(sm_80, GA100 70SM, HBM2e ~1.5TB/s) 설계 근거]
// - ew: 그리드 (ceil(n/128),1) · 블록 128 — 원본 hip 발사(구 hip 호스트
//   L786-796: launch3(grid,1,1,128))와 동일. 27B FFN n=17408(config.json
//   intermediate_size 실측 2026-10-04) → 136블록 · 스레드당 원소 1개 ·
//   g/u/y 완전 coalesce 스트리밍(블록당 512B×3 런치) — 메모리 본드
//   순수 원소별 연산. f64 트윈 exp의 ALU 비용도 HBM2e 대역폭 대비
//   무시(원소당 f64 15연산 ≈ 3스트림 f32 판독 1클록의 수 배에 불과,
//   GA100 f64 코어가 f32의 1/2배속이어도 병목은 판독). 136블록 ≪
//   70SM의 소형 런치 — T=1 레이턴시 도미넌트, 점유 확산 무의미.
//   개발기(RTX 4070 SUPER, sm_89)는 정합 호스트일 뿐 — 타이밍 판단
//   근거 아님(설계 계약).
// - argmax: 단일 블록 1024스레드(원본 계약, 도메인 n≤1M). 로짓
//   248320×4B ≈ 0.97MB를 1SM이 순차 스트리밍(스레드당 243원소) —
//   수십 µs 계급의 레이턴시 연산. 70SM 다중 블록 확산+병합 환원이
//   지연을 ~1µs급으로 줄일 수 있으나 (a) 병합 단계가 동일값 타이
//   브레이크를 재정의해야 하고(exact-match 계약 위험 — 본 모듈의
//   유일 경성 요구), (b) 전체 디코드 체인 대비 비중이 미미하므로
//   (FFN·GEMV 대비 <0.1%) 원본 1:1 구조를 유지한다. 정확성 우선
//   피벗 — 실측 장비 도착 후 병합형 재판정(그때까지 sm_80 SASS·
//   점유 문서화로 근거 대체, 커밋 본문 cuobjdump 증거).

// ── 미러 트랜센던트(G7 — G5 gdn_exp_d·G6 attn_exp_d와 동일 DAG) ──
// 계약: k = floor(x·invln2+½) → r = x−k·ln2(hi/lo 2분할) → 테일러
// 차수 7 → 2^k 비트 재구성. 연산 순서·상수는 절대 변경 금지 — 호스트
// Rust 트윈(구 프로브 gdn_exp_d, G5)과 리터럴까지 동일해야
// 비트동일(-fmad=false 빌드와 세트).
__device__ __forceinline__ double ew_exp_d(double x)
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

__device__ __forceinline__ float ew_expf(float x)
{
    return (float)ew_exp_d((double)x);
}

// ── ew 본체(구 rawhip 커널 L898-908 직이식, exp만 트윈 치환) ──
// y[j] = silu(g[j])·u[j] — T=1 디코드 FFN 게이트·업 곱. 원본식
// (v / (1.0f + __expf(-v))) * u[j]에서 __expf만 ew_expf로 바꾼다
// (f32 add/div/mul 순서 불변 — 비트동일 미러 계약).
extern "C" __global__ void ew(
    const float* __restrict__ g,
    const float* __restrict__ u,
    float* __restrict__ y,
    int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float v = g[j];
    float e = ew_expf(-v);
    y[j] = (v / (1.0f + e)) * u[j];
}

// ── argmax 본체(구 rawhip 커널 L911-934 직이식) ──
// [n] 로짓에서 최대 인덱스 1개(단일 블록 리덕션, n≤1M). n은 로짓
// 길이(248320) — 행수 아님(결함 8호). 리덕션: tid별 스트라이드 1024
// 상향 스캔(초기 −1e30, "초과" 갱신 — 클래스 내 첫 등장 유지) →
// 공유 트리(st=512..1, 동일값 낮은 tid 우선) → out[0].
extern "C" __global__ void ew_argmax(
    const float* __restrict__ lg,
    unsigned* __restrict__ out,
    int n)
{
    __shared__ float sv[1024];
    __shared__ unsigned si[1024];
    int tid = threadIdx.x;
    float best = -1e30f;
    unsigned idx = 0u;
    for (int i = tid; i < n; i += 1024) {
        float v = lg[i];
        if (v > best) {
            best = v;
            idx = (unsigned)i;
        }
    }
    sv[tid] = best;
    si[tid] = idx;
    __syncthreads();
    for (int st = 512; st > 0; st >>= 1) {
        if (tid < st) {
            if (sv[tid + st] > sv[tid]) {
                sv[tid] = sv[tid + st];
                si[tid] = si[tid + st];
            }
        }
        __syncthreads();
    }
    if (tid == 0) out[0] = si[0];
}
// 마커 ewc

