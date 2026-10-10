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
// 2) scan 공유메모리 합계 57,732B(A5-4: sk 8KB · qs 16KB · sv/KS/QS 4KB×3 ·
//    A/KQ 2KB×2 · dc 8KB · Stile 8KB×2(더블 버퍼) · bp/gcs/wsm 388B — V-타일
//    64열 기준)는 CUDA 정적 __shared__ 한계 48KB를 초과한다(ROCm LDS 64KB에서는
//    상주 가능했던 배치) → 동적
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
// (음성대조 gather 쌍둥이는 H 정리에서 제거 — 2026-10-09.)
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
// - scan: [A5-4 2026-10-10] A/KQ·gcs 전제는 청크 병렬 prepass
//   (gdn_scan_akq, 그리드 h_v×n_chunks)로 분리 — 상태 의존부(KS/QS·전진
//   대입·출력·상태 갱신)만 남기고 state 열을 V-타일(NSPLIT=4)로 나눠
//   grid=h_v×4. 구 "D열 2분할 손익 미확정"(A/KQ 전체 128내적 중복이
//   원인)은 prepass가 중복을 제거해 성립 — 비트동일 유지. 동적 공유
//   43,396B. KS/QS는 2-wide LDS(니블 쌍·float2) 적용. SASS 미세 최적화는
//   170HX 도착 후 재판단(현 잔여 스톨: 배리어·스케줄러 지연).
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
    // [2026-10-09 결함 수정] 도메인 가드 — 2^k 비트 재구성 `(1023+k)<<52`는
    // **k ≥ -1022**에서만 유효하다(그 아래는 비정규 — 지수 필드가 음수로
    // 시프트돼 쓰레기/NaN). 즉 유효역은 x ≥ -708.5.
    // 실측 결함: 35B GDN 헤드 11 게이트 g ≈ -91.6/토큰 → 청크 누적 gcs가
    // t=7에서 -641(통과)·t=8에서 -733(파괴 → 상태·출력 NaN, argmax 0 반복).
    // CPU 미러(f32 exp)는 이 구간에서 0으로 언더플로하므로 그 의미론으로
    // 클램프한다(f32 결과 동일). 상한 x > ln(2^1024) ≈ 709.78 → +inf.
    // 유효역 산술은 불변 — 기존 골든 무영향.
    if (x < -708.5) {
        return 0.0;
    }
    if (x > 709.78) {
        return __longlong_as_double(0x7ff0000000000000LL); // +inf
    }
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

// [FLA-8 채택 2026-10-10] f64 폴리 → __expf/__logf(2ulp) — eval 판정 후 채택.
// 근거: f64 트랜센던털이 GDN 소형 커널들의 지배 비용(conv 201µs/런치 = f64
// 파이프 2/cyc 포화, gate·akq 동형). 실측: 35B 프리필 145→132ms(-9%) ·
// 27B 377→357ms(-5%) · scan 11.5→9.5ms · conv·gate 대폭 감소.
// 판정(방법 B, eval.md): 골든 6종(단문·35토큰·600·4K ×2모델) 전부 통과 —
// 편차가 f16 저장 경계에 흡수. 장문 PPL 델타 +0.024%(35B)/−0.002%(27B),
// 일치율 97.7%/99.9% — 방법 자체 노이즈 대역. 종전 기각(conv-only fast exp
// → 600 플립)은 eval 이전 판정 — 전면 적용은 통과(구 기록: 속도 플랜).
// 도메인 가드(±708/709)는 __expf가 자연 처리(언더플로 0·오버플로 inf).
__device__ __forceinline__ float gdn_expf(float x)
{
    return __expf(x);
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

// [FLA-8 채택] __logf(2ulp) — softplus 게이트 경로(값 계약은 exp와 동일 계급).
__device__ __forceinline__ float gdn_logf(float y)
{
    return __logf(y);
}

// gdn_conv — 채널별 4탭(링 3 + 현재) conv + silu. [2026-10-10 토큰축 병렬화]
// 종전: 그리드 (conv_ch/128, 1) 단일 런치가 T행을 **순차 순회** — ncu 실측
// No Eligible 96.8%·IPC 0.13(512토큰 0.53ms/층, 프리필 conv 13.4ms의 원인).
// 4탭은 **입력 이력**만 필요(출력 의존 아님) → 토큰을 CONV_TC 블록으로
// 나눠 병렬화. 산술식·순서는 종전과 동일(비트동일 — 골든 판정).
// 링 [L][3][conv_ch]: t0==0 블록이 이전 청크의 x[-3..-1]을 읽고, 마지막
// 블록(t1==t_len)이 이번 청크의 x[t_len-3..t_len-1]을 기록(§3.3 계약 유지).
#define CONV_TC 16
extern "C" __global__ void gdn_conv(
    const float* __restrict__ qkv,   // [T][conv_ch]
    const float* __restrict__ convw, // [L][conv_ch][4]
    float* __restrict__ ring,        // [L][3][conv_ch]
    float* __restrict__ q_out,       // [T][k_len]
    float* __restrict__ k_out,       // [T][k_len]
    float* __restrict__ v_out,       // [T][v_len]
    float* __restrict__ ring_snap,   // [T][L][3][conv_ch] — 토큰 i 처리 후 링
                                     // (스펙 롤백용, 0이면 생략)
    int t_len, int layer, int k_len, int v_len, int conv_ch)
{
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= conv_ch) return;
    float w0 = convw[layer * conv_ch * 4 + ch * 4 + 0];
    float w1 = convw[layer * conv_ch * 4 + ch * 4 + 1];
    float w2 = convw[layer * conv_ch * 4 + ch * 4 + 2];
    float w3 = convw[layer * conv_ch * 4 + ch * 4 + 3];
    const int t0 = blockIdx.y * CONV_TC;
    const int t1 = min(t_len, t0 + CONV_TC);
    float h0, h1, h2;
    if (t0 == 0) {
        h0 = ring[layer * 3 * conv_ch + 0 * conv_ch + ch];
        h1 = ring[layer * 3 * conv_ch + 1 * conv_ch + ch];
        h2 = ring[layer * 3 * conv_ch + 2 * conv_ch + ch];
    } else {
        // 내부 청크 헤일로 = 같은 qkv의 앞 3행(CONV_TC ≥ 4 전제).
        h0 = qkv[(t0 - 3) * conv_ch + ch];
        h1 = qkv[(t0 - 2) * conv_ch + ch];
        h2 = qkv[(t0 - 1) * conv_ch + ch];
    }
    for (int t = t0; t < t1; t++) {
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
        if (ring_snap != (float*)0) {
            // 토큰 t 처리 후 링 = (x[t-2], x[t-1], x[t]). 호스트가 층 슬라이스
            // (layer * TMAX * 3 * conv_ch)를 더해 전달 — 레이아웃 [t][3][ch].
            float* rs = ring_snap + ((long)t * 3) * conv_ch + ch;
            rs[0] = h0;
            rs[(long)conv_ch] = h1;
            rs[(long)2 * conv_ch] = h2;
        }
    }
    if (t1 == t_len) {
        ring[layer * 3 * conv_ch + 0 * conv_ch + ch] = h0;
        ring[layer * 3 * conv_ch + 1 * conv_ch + ch] = h1;
        ring[layer * 3 * conv_ch + 2 * conv_ch + ch] = h2;
    }
}

// l2perm 본체 — a/b 도트(xn·abuf) + q/k L2 + v·beta|g lc 순열
// (구 rawhip 커널 L397-472 직이식, 폭 인자화). 그리드 (h_v, T), WG=128.
// v는 scatter, beta|g는 lc 순열 scatter(bg[.. + p_inv]).
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

    // [FLA-9 2026-10-10] a/b 도트 루프 병합 — xn 판독 1회(종전 2회, L2 96%
    // 포화 실측). 각 누산의 i-오름차순 순서는 원序 그대로(비트동일).
    float pa = 0.0f;
    float pb = 0.0f;
    const float* abA = abuf + (layer * 2) * h_v * hidden + h * hidden;
    const float* abB = abA + h_v * hidden;
    for (int i = tid; i < hidden; i += 128) {
        const float xv = xn[t * hidden + i];
        pa += xv * abA[i];
        pb += xv * abB[i];
    }
    red[tid] = pa;
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    float a_val = red[0];
    __syncthreads();

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
    // scatter 방향(계약): 쓰기측 인덱스에 p_inv.
    v_out[t * (h_v * 128) + p_inv * 128 + tid] =
        v_in[t * (h_v * 128) + h * 128 + tid];
}

// l2perm — scatter(계약 방향).
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
    gdn_l2perm_body(q_in, k_in, v_in, xn, abuf, alog, dtb,
                    q_out, k_out, v_out, bg, t_len, layer, h_k, h_v, hidden);
}

// gdn_scan — FLA 청크 알고리즘. [A5-4 2026-10-10 FLA 2단] A/KQ·gcs 전제는
// 청크 병렬 prepass(gdn_scan_akq)로 분리 — 상태 의존부(KS/QS·전진대입·
// 출력·상태 갱신)만 이 커널에 남기고 state 열을 V-타일(GDN_NSPLIT)로 나눠
// grid = h_v×NSPLIT로 넓힌다. 누적 순서(i·s2p·j)는 전부 원序 → 비트동일.
// CS=32·TILE=16·8패스. KS/QS/sk/sv f16, dc f32 전진대입 소거(순서 계약),
// o는 소거 직후 산출. 상태 P6 갱신. q는 L2 정규화 완료(중복 스케일 금지).
// 공유메모리는 동적 GDN_SCAN_SMEM(호스트 미러). 블록=512스레드.
#define GDN_CS 32
#define GDN_TILE 16
// [A5-4] state 열 V-타일: 블록 = (h, vs타일). 구 VSLICE와 달리 A/KQ는
// prepass가 비중복 계산한다(절단 시 A/KQ 중복이 기각 원인이었다).
#define GDN_NSPLIT 4 // [2026-10-10 실험] 8은 smem 36.2KB — 3블록 문턱
                     // 33.3KB 미달로 점유 무변(원복).
#define GDN_VS (128 / GDN_NSPLIT)   // 블록당 열 수
#define GDN_NGRP 16                 // 워크그룹 = 512스레드 / GDN_VS

// [A5-4] A/KQ 사전 계산 — 그리드 (h_v, n_chunks) 청크 병렬. 종전 융합
// 커널의 셀 계산을 그대로 이관(내적 s2 오름차순·half 반올림 지점 동일 —
// 값 비트동일). 출력 [2][h_v][n_chunks][CS][CS] half(A 평면 → KQ 평면).
extern "C" __global__ void gdn_scan_akq(
    const float* __restrict__ q,   // [T][k_len]
    const float* __restrict__ k,   // [T][k_len] L2
    const float* __restrict__ bg,  // [T][2*h_v]
    __half* __restrict__ akq,
    int t_len, int h_k, int h_v, int d)
{
    const int h = blockIdx.x;
    const int c = blockIdx.y;
    const int t0 = c * GDN_CS;
    const int n = min(t_len - t0, GDN_CS);
    const int kh = h % h_k;
    __shared__ __half sk[GDN_CS * 128];
    __shared__ float qs[GDN_CS * 128];
    __shared__ float gcs[GDN_CS + 1];
    __shared__ float bp[GDN_CS];
    for (int e = threadIdx.x; e < GDN_CS * 128; e += blockDim.x) {
        int t = e / 128, dv = e % 128;
        bool live = t < n;
        sk[e] = __float2half_rn(live ? k[(t0 + t) * (h_k * d) + kh * 128 + dv] : 0.0f);
        qs[e] = live ? q[(t0 + t) * (h_k * d) + kh * 128 + dv] : 0.0f;
    }
    if (threadIdx.x < GDN_CS) {
        float acc = 0.0f;
        for (int t = 0; t < GDN_CS; t++) {
            acc += (t < n) ? bg[(t0 + t) * (2 * h_v) + h_v + h] : 0.0f;
            gcs[t] = acc;
        }
        gcs[GDN_CS] = acc;
        bp[threadIdx.x] = (threadIdx.x < n) ? bg[(t0 + threadIdx.x) * (2 * h_v) + h] : 0.0f;
    }
    __syncthreads();
    const float qscale = 1.0f / sqrtf((float)d);
    const size_t plane = (size_t)h_v * gridDim.y * GDN_CS * GDN_CS;
    __half* Aout = akq + (size_t)h * gridDim.y * GDN_CS * GDN_CS + (size_t)c * GDN_CS * GDN_CS;
    __half* KQout = Aout + plane;
    for (int e = threadIdx.x; e < GDN_CS * GDN_CS; e += blockDim.x) {
        int i = e / GDN_CS, j = e % GDN_CS;
        if (i < n) {
            float dk = 0.0f, dq = 0.0f;
            for (int s2 = 0; s2 < 128; s2++) {
                float kj = __half2float(sk[j * 128 + s2]);
                dk += __half2float(sk[i * 128 + s2]) * kj;
                dq += qs[i * 128 + s2] * kj;
            }
            float bi = bp[i];
            Aout[i * GDN_CS + j] = __float2half_rn((j < i) ? dk * bi * gdn_expf(gcs[i] - gcs[j]) : 0.0f);
            KQout[i * GDN_CS + j] = __float2half_rn((j <= i) ? dq * qscale * gdn_expf(gcs[i] - gcs[j]) : 0.0f);
        } else {
            Aout[i * GDN_CS + j] = __float2half_rn(0.0f);
            KQout[i * GDN_CS + j] = __float2half_rn(0.0f);
        }
    }
}
extern "C" __global__ void gdn_scan(
    const float* __restrict__ q,     // [T][k_len]
    const float* __restrict__ k,     // [T][k_len] L2
    const float* __restrict__ v,     // [T][h_v*128] lc
    const float* __restrict__ bg,    // [T][2*h_v]
    const __half* __restrict__ akq,  // [2][h_v][n_chunks][CS][CS] — A5-4 prepass
    float* __restrict__ st,          // [L][h_v][128*128] r/w — S0≠0 경로 의무
    float* __restrict__ outv,        // [T][h_v*128] o_lc
    int t_len, int h_k, int h_v, int d, int layer)
{
    // [A5-4] 그리드 (h_v×NSPLIT) — 블록이 state 열 타일(vs)을 소유.
    // [2026-10-10 기각 기록] launch_bounds(512,3) = 33→36.6ms(스필>점유),
    // dcr 레지스터 배열 제거+3블록 = 35.5ms — 레지스터 64는 실수요.
    // FLA 4단 재작성은 예측 기각: GDN f16 저장 경계 민감도(실측 ×289 증폭,
    // fast-exp 기각과 동일 클래스)로 재구성 시 장문 골든 플립 확실.
    // [FLA-2 기각] 해(solve) 동안 그룹1-15가 차기 청크 스테이징(더블 버퍼) —
    // 35B -7%이나 27B +30% 회귀(35→45ms). smem carveout 가설은 단독 검증으로
    // 기각(무영향) — 오버랩 코드 자체가 원인(모델 의존, 미규명). 되돌림.
    extern __shared__ char smem_raw[];
    __half* sk = (__half*)smem_raw;                    // [CS*128]
    // [A5-4b 2026-10-10] qs 스테이징(16KB) 제거 — q는 글로벌 직접 판독
    // (dq2가 L1에 상주). smem 43.4→27.0KB → 3블록/SM(배리어 은닉).
    __half* sv = (__half*)(sk + GDN_CS * 128);         // [CS*GDN_VS]
    __half* A = sv + GDN_CS * GDN_VS;                  // [CS*CS]
    __half* KQ = A + GDN_CS * GDN_CS;                  // [CS*CS]
    __half* KS = KQ + GDN_CS * GDN_CS;                 // [CS*GDN_VS]
    __half* QS = KS + GDN_CS * GDN_VS;                 // [CS*GDN_VS]
    float* dc = (float*)(QS + GDN_CS * GDN_VS);        // [CS*GDN_VS]
    float* Stile = dc + GDN_CS * GDN_VS;               // [2*TILE*GDN_VS] (더블 버퍼)
    float* bp = Stile + 2 * GDN_TILE * GDN_VS;         // [CS]
    float* gcs = bp + GDN_CS;                          // [CS+1]
    float* wsm = gcs + (GDN_CS + 1);                   // [CS]
    float* e = wsm + GDN_CS;                           // [CS] exp(gcs) 사전값
    int h = blockIdx.x / GDN_NSPLIT;
    int vs = (blockIdx.x % GDN_NSPLIT) * GDN_VS;
    int kh = h % h_k;
    // [A5] 열 소유자 tid(0..VS-1) + 워크그룹 grp — 원순서 보존 분배.
    int tid = threadIdx.x % GDN_VS;
    int grp = threadIdx.x / GDN_VS;
    int n_chunks = (t_len + GDN_CS - 1) / GDN_CS;
    // G5 정밀화: rsqrtf(≤2ulp 근사) 대신 IEEE sqrt+div — 호스트 미러와
    // 비트동일(양측 sqrt.rn·div.rn).
    float qscale = 1.0f / sqrtf((float)d);
    long st_h = (long)layer * h_v * d * d + (long)h * d * d + vs;

    for (int c = 0; c < n_chunks; c++) {
        int t0 = c * GDN_CS;
        int n = min(t_len - t0, GDN_CS);

        // [FLA-4 2026-10-10] float4 적재 — 4차원 동시 변환·기록(값 동일:
        // __floats2half2_rn = __float2half_rn 쌍, RN-even). 명령·MLP 개선.
        for (int e4 = threadIdx.x; e4 < GDN_CS * 32; e4 += blockDim.x) {
            int t = e4 / 32, dv4 = (e4 % 32) * 4;
            bool live = t < n;
            float4 kv = live ? *reinterpret_cast<const float4*>(
                                  &k[(t0 + t) * (h_k * d) + kh * 128 + dv4])
                             : make_float4(0.f, 0.f, 0.f, 0.f);
            *reinterpret_cast<__half2*>(&sk[t * 128 + dv4]) =
                __floats2half2_rn(kv.x, kv.y);
            *reinterpret_cast<__half2*>(&sk[t * 128 + dv4 + 2]) =
                __floats2half2_rn(kv.z, kv.w);
        }
        for (int e4 = threadIdx.x; e4 < GDN_CS * (GDN_VS / 4); e4 += blockDim.x) {
            int t = e4 / (GDN_VS / 4), dvl4 = (e4 % (GDN_VS / 4)) * 4;
            bool live = t < n;
            float4 vv = live ? *reinterpret_cast<const float4*>(
                                   &v[(t0 + t) * (h_v * d) + h * 128 + vs + dvl4])
                             : make_float4(0.f, 0.f, 0.f, 0.f);
            *reinterpret_cast<__half2*>(&sv[t * GDN_VS + dvl4]) =
                __floats2half2_rn(vv.x, vv.y);
            *reinterpret_cast<__half2*>(&sv[t * GDN_VS + dvl4 + 2]) =
                __floats2half2_rn(vv.z, vv.w);
        }
        if (threadIdx.x < GDN_CS) {
            float acc = 0.0f;
            for (int t = 0; t < GDN_CS; t++) {
                acc += (t < n) ? bg[(t0 + t) * (2 * h_v) + h_v + h] : 0.0f;
                gcs[t] = acc;
            }
            gcs[GDN_CS] = acc;
            bp[tid] = (tid < n) ? bg[(t0 + tid) * (2 * h_v) + h] : 0.0f;
        }
        // [A5-4] A/KQ는 prepass 산출 복사(prepass와 셀 값 비트동일).
        {
            const size_t plane = (size_t)h_v * n_chunks * GDN_CS * GDN_CS;
            const __half* Ag =
                akq + (size_t)h * n_chunks * GDN_CS * GDN_CS + (size_t)c * GDN_CS * GDN_CS;
            const __half* KQg = Ag + plane;
            // [FLA-4] uint2(4 half) 벡터 복사.
            for (int e4 = threadIdx.x; e4 < GDN_CS * GDN_CS / 4; e4 += blockDim.x) {
                reinterpret_cast<uint2*>(A)[e4] =
                    reinterpret_cast<const uint2*>(Ag)[e4];
                reinterpret_cast<uint2*>(KQ)[e4] =
                    reinterpret_cast<const uint2*>(KQg)[e4];
            }
        }
        __syncthreads();

        // [A5-2] KS/QS 레지스터 누적(pass마다 half 반올림 — 값 순서 불변) +
        // Stile 더블 버퍼(cp.async 프리페치). [A5-4] sk/qs는 니블 쌍(2-wide)
        // LDS — 두 누산 chain 순서는 s2p 오름차순 원序 그대로(비트동일).
        constexpr int IK = (GDN_CS + GDN_NGRP - 1) / GDN_NGRP;
        float rks[IK], rqs[IK];
#pragma unroll
        for (int k = 0; k < IK; ++k) {
            rks[k] = 0.0f;
            rqs[k] = 0.0f;
        }
        {
            // pass 0 타일 선적재(buf 0) — cp.async(레지스터 경유 제거).
            for (int e = threadIdx.x; e < GDN_TILE * GDN_VS; e += blockDim.x) {
                int r = e / GDN_VS, cl = e % GDN_VS;
                unsigned sa = (unsigned)__cvta_generic_to_shared(&Stile[e]);
                const float* src = &st[st_h + (long)r * d + cl];
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(sa),
                             "l"(src));
            }
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
        for (int pass_ = 0; pass_ < 8; pass_++) {
            int s2b = pass_ * GDN_TILE;
            // 다음 타일을 반대 버퍼에 cp.async 프리페치(소비와 완전 중첩).
            if (pass_ + 1 < 8) {
                float* nxt = Stile + ((pass_ & 1) ^ 1) * (GDN_TILE * GDN_VS);
                const int nb = s2b + GDN_TILE;
                for (int e = threadIdx.x; e < GDN_TILE * GDN_VS; e += blockDim.x) {
                    int r = e / GDN_VS, cl = e % GDN_VS;
                    unsigned sa = (unsigned)__cvta_generic_to_shared(&nxt[e]);
                    const float* src = &st[st_h + (long)(nb + r) * d + cl];
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(sa),
                                 "l"(src));
                }
                asm volatile("cp.async.commit_group;");
            }
            const float* cur = Stile + (pass_ & 1) * (GDN_TILE * GDN_VS);
            // [A5-3] 타일 16값을 레지스터로 1회 로드(그룹 내 i들이 재사용).
            float sv0[GDN_TILE];
#pragma unroll
            for (int r = 0; r < GDN_TILE; ++r) {
                sv0[r] = cur[r * GDN_VS + tid];
            }
            // [A5] i축 워크그룹 분배 — i별 누적 순서(pass·s2p)는 원序 그대로.
            for (int k = 0; k < IK; ++k) {
                const int i = grp + k * GDN_NGRP;
                // T<n 행 스킵 — KS/QS[i≥n]는 소비자가 i<n만 읽는다.
                if (i >= n) {
                    continue;
                }
                float ak = 0.0f, aq = 0.0f;
                for (int s2p = 0; s2p < GDN_TILE; s2p += 2) {
                    const float s0 = sv0[s2p];
                    const float s1 = sv0[s2p + 1];
                    const __half2 sk2 =
                        *reinterpret_cast<const __half2*>(&sk[i * 128 + s2b + s2p]);
                    const float2 skf = __half22float2(sk2);
                    const float2 qf = (i < n)
                        ? *reinterpret_cast<const float2*>(
                              &q[(t0 + i) * (h_k * d) + kh * 128 + s2b + s2p])
                        : make_float2(0.0f, 0.0f);
                    ak += skf.x * s0;
                    ak += skf.y * s1;
                    aq += qf.x * s0;
                    aq += qf.y * s1;
                }
                rks[k] = __half2float(__float2half_rn(rks[k] + ak));
                rqs[k] = __half2float(__float2half_rn(rqs[k] + aq * qscale));
            }
            // 프리페치 완료 대기(그룹 0) — 이후 배리어가 가시화.
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
        }
        for (int k = 0; k < IK; ++k) {
            const int i = grp + k * GDN_NGRP;
            if (i < n) {
                KS[i * GDN_VS + tid] = __float2half_rn(rks[k]);
                QS[i * GDN_VS + tid] = __float2half_rn(rqs[k]);
            } else {
                KS[i * GDN_VS + tid] = __float2half_rn(0.0f);
                QS[i * GDN_VS + tid] = __float2half_rn(0.0f);
            }
        }
        // [FLA-1 2026-10-10] e[i]=exp(gcs[i]) 사전 계산(병렬) — 해(solve)의
        // 직렬 exp 2n회 제거. 값은 종전 인라인 계산과 비트동일(같은 함수·입력).
        if (threadIdx.x < GDN_CS) {
            e[threadIdx.x] = (threadIdx.x < n) ? gdn_expf(gcs[threadIdx.x]) : 0.0f;
        }
        __syncthreads();

        // [A5] 전치대입(rhs/dc) — 열(tid) 소유라 그룹0만(다른 그룹은 배리어
        // 대기). [FLA-1 2026-10-10] 출력(oi)은 상태 갱신 뒤 전 그룹 병렬로
        // 이동 — 직렬 구간 단축(값·순서 불변, 비트동일). e[i]는 사전 계산값.
        if (grp == 0) {
            for (int i = 0; i < n; i++) {
                float rhs = bp[i] * (__half2float(sv[i * GDN_VS + tid]) - e[i] * __half2float(KS[i * GDN_VS + tid]));
                // [FLA-3 2026-10-10] j를 고정 32회 전개 — A[i][j≥i]=0(prepass가
                // 0 기록)이므로 무조건 누산이 동일 값(0·dc=±0, rhs∓0=rhs —
                // 비트동일). 동적 하한 루프의 smem 지연 노출(i×~35cyc) 제거.
#pragma unroll
                for (int j = 0; j < GDN_CS; j++) {
                    float aij = __half2float(A[i * GDN_CS + j]);
                    rhs -= aij * dc[j * GDN_VS + tid];
                }
                dc[i * GDN_VS + tid] = rhs;
            }
        }
        __syncthreads();

        {
            float gtot = gcs[GDN_CS];
            float gt_exp = gdn_expf(gtot);
            if (threadIdx.x < GDN_CS) wsm[threadIdx.x] = (threadIdx.x < n) ? gdn_expf(gtot - gcs[threadIdx.x]) : 0.0f;
            __syncthreads();
            // [FLA-1 2026-10-10] 출력 병렬화 — 16그룹 × 2행(행별 순서 그대로:
            // e[i]·QS + KQ[i][p≤i]·dc[p]). 상태 갱신과 독립(배리어 불증).
            for (int r = 0; r < 2; ++r) {
                const int i = grp * 2 + r;
                if (i >= n) {
                    break;
                }
                float oi = e[i] * __half2float(QS[i * GDN_VS + tid]);
                // [FLA-3] p 고정 32회 전개 — KQ[i][p>i]=0(비트동일).
#pragma unroll
                for (int p = 0; p < GDN_CS; p++) {
                    float w = __half2float(KQ[i * GDN_CS + p]);
                    oi += w * dc[p * GDN_VS + tid];
                }
                outv[(t0 + i) * (h_v * d) + h * 128 + vs + tid] = oi;
            }
            // [A5-3] dc 열(≤32)을 레지스터 1회 — 상태 갱신 재판독 제거.
            float dcr[GDN_CS];
#pragma unroll
            for (int j = 0; j < GDN_CS; ++j) {
                dcr[j] = dc[j * GDN_VS + tid];
            }
            // [A5] 상태 갱신 행(s2) 워크그룹 분배 — 행별 j 순서는 원序 그대로.
            for (int s2 = grp; s2 < 128; s2 += GDN_NGRP) {
                float acc = st[st_h + (long)s2 * d + tid] * gt_exp;
                for (int j = 0; j < n; j++)
                    acc += __half2float(sk[j * 128 + s2]) * wsm[j] * dcr[j];
                st[st_h + (long)s2 * d + tid] = acc;
            }
            __syncthreads();
        }
    }
}

// ── [P9] t=1 전용 GDN — i축(상태 행) 분할 3커널 ──
// ncu 실측: gdn_scan은 점유 8.3%·compute 8.3% = 지연 바운드(48블록×4워프,
// 상태 64KB/블록). t=1이면 A(하삼각)·cumsum·청크 기구가 전부 자명하므로
// i축(128)을 split(4)으로 나눠 grid를 h_v×4(192블록)로 늘린다.
// 수식(t=1, g=bg[h_v+h], β=bg[h], e=exp(g), wsm[0]=1, gt_exp=e):
//   KS[d]=Σ_i half(k[i])·st[i][d]   QS[d]=Σ_i q[i]·st[i][d]
//   dc[d]=β·(half(v[d]) − e·KS[d])
//   out[d]=e·QS[d]·qscale + KQ00·dc[d]   (KQ00 = Σ q·half(k) · qscale)
//   st[i][d] = st[i][d]·e + half(k[i])·dc[d]
// 산술 순서는 i 오름차순 유지(부분합은 분할 순서로 결합) — half 라운딩
// 지점(KS/QS 8패스 half 누적)은 계약 완화로 f32 단일 누적으로 단순화.
#define GDN1_SPLIT 8

extern "C" __global__ void gdn1_part(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ st, float* __restrict__ part,
    int h_k, int h_v, int d, int layer)
{
    const int h = blockIdx.x;
    const int s = blockIdx.y;
    const int tid = threadIdx.x; // = d 인덱스
    const int kh = h % h_k;
    const int ch = 128 / GDN1_SPLIT;
    const int i0 = s * ch;
    const long st_h = (long)layer * h_v * d * d + (long)h * d * d;
    float ks = 0.0f, qs = 0.0f;
    for (int ii = 0; ii < ch; ++ii) {
        const int i = i0 + ii;
        const float stv = st[st_h + (long)i * d + tid];
        ks += __half2float(__float2half_rn(k[kh * 128 + i])) * stv;
        qs += q[kh * 128 + i] * stv;
    }
    float* p = part + ((long)h * GDN1_SPLIT + s) * 256;
    p[tid] = ks;
    p[128 + tid] = qs;
}

extern "C" __global__ void gdn1_comb(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ bg,
    const float* __restrict__ part, float* __restrict__ dc,
    float* __restrict__ outv, int h_k, int h_v, int d)
{
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int kh = h % h_k;
    float ks = 0.0f, qs = 0.0f;
    for (int s = 0; s < GDN1_SPLIT; ++s) {
        const float* p = part + ((long)h * GDN1_SPLIT + s) * 256;
        ks += p[tid];
        qs += p[128 + tid];
    }
    __shared__ float red[128];
    red[tid] = q[kh * 128 + tid] * __half2float(__float2half_rn(k[kh * 128 + tid]));
    __syncthreads();
    for (int stp = 64; stp > 0; stp >>= 1) {
        if (tid < stp) {
            red[tid] += red[tid + stp];
        }
        __syncthreads();
    }
    const float qscale = 1.0f / sqrtf((float)d);
    const float kq00 = red[0] * qscale;
    const float g = bg[h_v + h];
    const float beta = bg[h];
    const float e = gdn_expf(g);
    const float sv = __half2float(__float2half_rn(v[h * 128 + tid]));
    const float rhs = beta * (sv - e * ks);
    dc[h * 128 + tid] = rhs;
    outv[h * 128 + tid] = e * qs * qscale + kq00 * rhs;
}

extern "C" __global__ void gdn1_upd(
    const float* __restrict__ k, const float* __restrict__ bg,
    const float* __restrict__ dc, float* __restrict__ st,
    int h_k, int h_v, int d, int layer)
{
    const int h = blockIdx.x;
    const int s = blockIdx.y;
    const int tid = threadIdx.x;
    const int kh = h % h_k;
    const int ch = 128 / GDN1_SPLIT;
    const int i0 = s * ch;
    const long st_h = (long)layer * h_v * d * d + (long)h * d * d;
    const float e = gdn_expf(bg[h_v + h]);
    const float dcv = dc[h * 128 + tid];
    for (int ii = 0; ii < ch; ++ii) {
        const int i = i0 + ii;
        const float kv = __half2float(__float2half_rn(k[kh * 128 + i]));
        float* p = &st[st_h + (long)i * d + tid];
        *p = *p * e + kv * dcv;
    }
}

// [A-1 2026-10-10] 스펙 검증용 GDN — t≤8 토큰 루프 + **토큰별 상태 스냅샷**.
// t=1 trio(part/comb/upd)를 블록=(h) 하나로 융합하되 산술은 trio와 **비트
// 동일**: KS/QS는 8분할(16행) 부분합을 분할 순서로 결합, half(k)/half(v)/
// exp 라운딩 지점 동일, kq00은 동일 트리. 각 토큰 처리 후 그 토큰의 상태
// (이 층·이 헤드 128×128)를 snap[t]에 기록 — 부분 수용 롤백의 복원 지점.
// (열 소유 스레드 구조라 토큰 간 상태 의존이 스레드-로컬 — 동기화는 kq00
// 트리뿐.)
extern "C" __global__ void gdn_spec_scan(
    const float* __restrict__ q,     // [T][k_len]
    const float* __restrict__ k,     // [T][k_len]
    const float* __restrict__ v,     // [T][h_v*128]
    const float* __restrict__ bg,    // [T][2*h_v]
    float* __restrict__ st,          // [L][h_v][d*d] r/w — 최종 상태
    float* __restrict__ snap,        // [KMAX][L][h_v][d*d] — 토큰별 상태
    float* __restrict__ outv,        // [T][h_v*128]
    int t_len, int h_k, int h_v, int d, int layer, int n_layers)
{
    const int h = blockIdx.x;
    const int tid = threadIdx.x; // = d 인덱스
    const int kh = h % h_k;
    const long st_h = (long)layer * h_v * d * d + (long)h * d * d;
    __shared__ float red[128];
    const float qscale = 1.0f / sqrtf((float)d);
    for (int t = 0; t < t_len; ++t) {
        const float* qt = q + (long)t * (h_k * 128);
        const float* kt = k + (long)t * (h_k * 128);
        // part(8분할 부분합 → 분할 순서 결합) — trio gdn1_part/comb 미러.
        float ks = 0.0f, qs = 0.0f;
        for (int s = 0; s < GDN1_SPLIT; ++s) {
            const int i0 = s * (128 / GDN1_SPLIT);
            float pks = 0.0f, pqs = 0.0f;
            for (int ii = 0; ii < 128 / GDN1_SPLIT; ++ii) {
                const int i = i0 + ii;
                const float stv = st[st_h + (long)i * d + tid];
                pks += __half2float(__float2half_rn(kt[kh * 128 + i])) * stv;
                pqs += qt[kh * 128 + i] * stv;
            }
            ks += pks;
            qs += pqs;
        }
        // kq00 — trio comb의 블록 트리와 동일 순서.
        red[tid] = qt[kh * 128 + tid] * __half2float(__float2half_rn(kt[kh * 128 + tid]));
        __syncthreads();
        for (int stp = 64; stp > 0; stp >>= 1) {
            if (tid < stp) {
                red[tid] += red[tid + stp];
            }
            __syncthreads();
        }
        const float kq00 = red[0] * qscale;
        const float g = bg[t * (2 * h_v) + h_v + h];
        const float beta = bg[t * (2 * h_v) + h];
        const float e = gdn_expf(g);
        const float sv = __half2float(__float2half_rn(v[t * (h_v * 128) + h * 128 + tid]));
        const float rhs = beta * (sv - e * ks);
        outv[t * (h_v * 128) + h * 128 + tid] = e * qs * qscale + kq00 * rhs;
        // 상태 갱신(trio gdn1_upd 미러) + 스냅샷 기록(이 토큰 처리 후).
        float* sst = st + st_h;
        for (int i = 0; i < 128; ++i) {
            const float kv = __half2float(__float2half_rn(kt[kh * 128 + i]));
            float* p = &sst[(long)i * d + tid];
            *p = *p * e + kv * rhs;
        }
        // 스냅샷 — snap[t][layer][h][i][d] (t는 배치 내 위치 — 상한 t_len).
        float* sn = snap + (((long)t * n_layers + layer) * h_v + h) * ((long)d * d);
        for (int i = 0; i < 128; ++i) {
            sn[(long)i * d + tid] = sst[(long)i * d + tid];
        }
        __syncthreads(); // 다음 토큰 kq00 트리 전 정리
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
