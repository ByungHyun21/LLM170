// ── Flash-Next(qwen4exp) GDN 스테이지 CUDA 커널 — plans/124 FNF(fn-gdn),
// 2026-10-05. 재사용-판정 결과 이식분만 수록: 신규 4종(fn_gdn_prep·
// fn_gdn_scan·fn_gdn_gate·fn_gdn_conv3[음성대조]) + core 트랜센던트 미러
// 2종(fn_exp_cr·fn_ln_cr). conv만 G5 자산(exl3_gdn.fatbin)을 수정 없이
// 재사용한다(판정표는 fn_gdn_cuda.rs 머리 참조 — scan은 실측 결함 2건으로
// REUSE→PORT 전환, fn_gdn_scan 머리 전표).
//
// ═══ 재사용-판정 요약(FNF 실측·대조 종결 2026-10-05 — 전표는 모듈층) ═══
// · conv — REUSE exl3_gdn_conv(G5): qwen4exp conv_k=4 산술과 동일 DAG.
//   핵심 오독 정정: FNA 계약지도의 "G5 conv 3탭 고정"은 **링 깊이 3**
//   (= conv_k−1)을 가리킨다. 커널은 가중치 [ch][4]의 w0..w3 전부를
//   읽는다(o = w3·x + w0·h0 + w1·h1 + w2·h2) — 원천 stages/gdn.rs
//   L69-85(conv_w[c*conv_k+3]·x + Σ_{j<3} w[c*conv_k+j]·s_j)과 항·
//   결합순서까지 동일. 실측 Flash-Next GGUF blk.0.ssm_conv1d.weight
//   F32 ne=[4,10240](플랫 [c][4] = core f32_vec4 레이아웃 그대로).
// · scan — PORT fn_gdn_scan: core/src/gdn.rs gdn_chunk_seq L22는
//   qwen35·qwen4exp **공유 함수**(stages/gdn.rs L147 호출)라 수학이
//   G5 미러와 동일 — 그러나 실측 결함 2건으로 재사용 불가(1차 시도
//   실측: ord=0 gcs<−900에서 gdn_expf NaN[o 9088·상태 65536] · 게이트
//   rms 증폭으로 종단 2.579e-3>2e-4). 포트는 f32 전체·CS=64·core 동일
//   적산순서(비트 근젡, expf 잔차) — fn_gdn_scan 머리 참조.
// · prep — NEW: β/e^g 원천이 다르다. EXL3 l2perm은 잔류 xn과 abuf
//   도트로 a/b를 산출(assets/exl3_gdn.cu L254 계약)하나, qwen4exp는
//   dt_rank폭 투영 b/a를 입력으로 받는다(stages/gdn.rs L57-58:
//   sigmoid(b)·softplus(a+dt_bias)·ssm_a — ssm_a는 GGUF 원값 저장,
//   EXL3 A_log의 −exp 환산 불필요). q/k l2도 상이: eps가 **floor**
//   (ops.rs l2_norm L39-44: 1/max(sqrt(Σ),eps))이고 lc 순열이 없다
//   (n_group=16헤드 그룹 l2 — stages/gdn.rs L86-97).
// · gate — NEW: z-게이트가 sigmoid(gdn_norm.rs GdnGate::Sigmoid L10,
//   qwen35 silu와의 유일 차이) + rms가 f64 32세그먼트(ops.rs sq_sum
//   L11-31) + 역순열 없음. G5 gate는 silu·f32 트리·역lc순열.
//
// ═══ 비트동일 미러 계약(신규 2종에 한함) ═══
// fn_exp_cr/fn_ln_cr은 crates/core/src/ops.rs exp_cr L52-88·ln_cr
// L91-121의 **연산열 그대로**(상수·fma·round_ties_even 포함). IEEE
// fma는 양 플랫폼 모두 올림-정확 단일 연산이라 결과 비트동일(ops.rs
// 헤더 계약 "HIP 커널 exp_cr과 동일 연산열(비트 동일)"의 CUDA판).
// prep·gate는 여기에 순차 f32 누산(l2 — ops.rs L40-44)·32세그먼트
// f64 결합(sq_sum L11-31)까지 미러해 **core 대비 비트동일**을 목표로
// 한다(프로브에서 0.000e0 판정). conv·scan(재사용분)은 gdn_expf/
// f16 저장 계약을 유지하므로 ulp~1e-4급 — 값 maxdiff ≤ 2e-4 판정.
// 빌드는 -fmad=false(FMA 수축 제거 — 비트동일 조건, exl3_gdn.cu와
// 동일 규율).
//
// [CMP 170HX(sm_80, GA100 70SM, HBM2e ~1.5TB/s) 설계 근거 — plans/124 §0]
// - fn_gdn_prep: 그리드 (dt_rank=48, T), 블록 128 — dt_rank×T 블록
//   (T=32면 1536)로 70SM 포화. 본체는 bg 스칼라 2개(tid 0)+ 헤드 l2
//   스케일 2개(tid 0 순차 128 누산 ×2)로 메모리 본드 소형. 순차 누산을
//   트리 환원으로 바꾸면 점유 이득 없이 비트동일 계약이 깨진다(원천
//   ops.rs l2_norm L40-44가 순차 f32) — 원천 정렬이 계약. GA100에서
//   2KB 미만 로드/스토어/블록이라 HBM 대역 코너도 아니고 L2 상주.
// - fn_gdn_gate: 동일 그리드 규약(결함 5호: t=blockIdx.y). tid<32
//   세그먼트 부분합 → f64 32항 결합(tid 0) — core sq_sum 미러.
//   f64 연산은 헤드당 34회뿐(GA100 FP64 1/32 레이트와 무관한 규모).
// - fn_gdn_conv3: 음성대조 계기(원장 17호) — 프로덕션 발사 금지.
//   프로덕션 conv·scan의 sm_80 근거는 assets/exl3_gdn.cu 머리(레지스터
//   상주 7f32/스레드·80블록 스트리밍·동적 smem 61,828B → 2블록/SM
//   상한)가 그대로 승계된다(재사용).
#include <cuda_fp16.h>

// ── core 트랜센던트 미러(ops.rs exp_cr L52-88 직이식 — 상수·연산열 불변) ──
// round_ties_even → rint(기본 반올림 모드 = 최근접짝수), mul_add → fma.
__device__ __forceinline__ float fn_exp_cr(float x)
{
    double xd = (double)x;
    if (xd > 88.72) {
        return __int_as_float(0x7f800000); // f32::INFINITY
    }
    if (xd < -103.97) {
        return 0.0f;
    }
    const double LN2_HI = 6.931471803691238e-1;
    const double LN2_LO = 1.9082149292705877e-10;
    const double INV_LN2 = 1.4426950408889634; // std LOG2_E 비트동일
    double kd = rint(xd * INV_LN2);
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
    if (k > 127) {
        return __int_as_float(0x7f800000);
    }
    double scale = __longlong_as_double((long long)(k + 1023) << 52);
    return (float)(p * scale);
}

// ops.rs ln_cr L91-121 직이식(atanh 급수 fma 호너 — 정규수 v ≥ 1 전용).
__device__ __forceinline__ double fn_ln_cr(double v)
{
    unsigned long long bits = __double_as_longlong(v);
    long long e = (long long)((bits >> 52) & 0x7ffULL);
    long long k = e - 1023;
    unsigned long long mb =
        (bits & 0x800fffffffffffffULL) | 0x3ff0000000000000ULL;
    double m = __longlong_as_double((long long)mb);
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
    const double LN2_HI = 6.931471803691238e-1;
    const double LN2_LO = 1.9082149292705877e-10;
    double kh = (double)k * LN2_HI;
    double kl = (double)k * LN2_LO;
    double s1 = lnm + kh;
    double s2 = (lnm - s1) + kh;
    return s1 + (s2 + kl);
}

// ops.rs sigmoid L133·softplus L138(log1p_cr L123 인라인) — 결합순서 불변.
__device__ __forceinline__ float fn_sigmoid(float x)
{
    return 1.0f / (1.0f + fn_exp_cr(-x));
}

__device__ __forceinline__ float fn_softplus(float x)
{
    if (x > 20.0f) {
        return x;
    }
    // log1p_cr(exp_cr(x)) = ln_cr((double)exp_cr(x) + 1.0) as f32
    return (float)fn_ln_cr((double)fn_exp_cr(x) + 1.0);
}

// fn_gdn_prep — β/e^g 사전 + q/k 헤드 l2(원천: stages/gdn.rs L52-60·
// L86-97). 그리드 (dt_rank, T), 블록 128. h<blockIdx.x n_group 미만
// 블록이 q/k l2를 담당(전체 헤드 수 dt_rank=48, q/k 헤드 n_group=16).
// l2는 ops.rs L39-44 미러 — sum 순차 f32 누산(tid 0)·scale =
// 1/max(sqrt(sum), eps)(**eps는 floor** — EXL3 l2perm의 가산 eps와
// 상이, 판정표 참조). 순차 누산 원천 정렬 = 비트동일 계약.
extern "C" __global__ void fn_gdn_prep(
    const float* __restrict__ b_in,   // [T][dt_rank] sigmoid 전(투영 b)
    const float* __restrict__ a_in,   // [T][dt_rank] softplus 전(투영 a)
    const float* __restrict__ dtb,    // [L][dt_rank] ssm_dt.bias
    const float* __restrict__ ssa,    // [L][dt_rank] ssm_a 원값(GGUF)
    const float* __restrict__ q_in,   // [T][k_len] conv q
    const float* __restrict__ k_in,   // [T][k_len] conv k
    float* __restrict__ q_out,        // [T][k_len] l2 완료 q
    float* __restrict__ k_out,        // [T][k_len] l2 완료 k
    float* __restrict__ bg,           // [T][2*dt_rank] beta|g(자연 순서)
    int t_len, int layer, int n_group, int dt_rank, float eps)
{
    int t = blockIdx.y;
    int h = blockIdx.x;
    int tid = threadIdx.x;
    if (t >= t_len) return;
    __shared__ float sq_s, sk_s;
    if (tid == 0) {
        // stages/gdn.rs L57-58: sigmoid(b)·softplus(a+dt_bias)*ssm_a.
        float bv = b_in[t * dt_rank + h];
        float av = a_in[t * dt_rank + h];
        float dt = dtb[layer * dt_rank + h];
        float sa = ssa[layer * dt_rank + h];
        bg[t * (2 * dt_rank) + h] = fn_sigmoid(bv);
        bg[t * (2 * dt_rank) + dt_rank + h] = fn_softplus(av + dt) * sa;
    }
    if (h < n_group) {
        int base = t * (n_group * 128) + h * 128;
        if (tid == 0) {
            // ops.rs l2_norm L40-44 미러 — 순차 f32(원천 정렬).
            float s = 0.0f;
            for (int i = 0; i < 128; i++) {
                float x = q_in[base + i];
                s += x * x;
            }
            sq_s = 1.0f / fmaxf(sqrtf(s), eps);
            s = 0.0f;
            for (int i = 0; i < 128; i++) {
                float x = k_in[base + i];
                s += x * x;
            }
            sk_s = 1.0f / fmaxf(sqrtf(s), eps);
        }
        __syncthreads();
        q_out[base + tid] = q_in[base + tid] * sq_s;
        k_out[base + tid] = k_in[base + tid] * sk_s;
    }
}

// fn_gdn_gate — norm_gated(GdnGate::Sigmoid): 원천 stages/gdn.rs
// L156-168 → gdn_norm.rs gdn_norm_gated L26-50 → ops.rs rms_norm L33 +
// sq_sum L11-31·sigmoid L133. 그리드 (dt_rank, T), 블록 128. **역순열
// 없음**(qwen4exp o_all/gated는 자연 헤드 순서 — EXL3 gate의 lc 역순열
// 미적용이 판정표 차이). rms: 32세그먼트 f32 부분합(세그먼트=원소 4개
// 순차) → f64 32항 순차 결합 → scale = 1/((sum/128+eps)^0.5) — core
// sq_sum·rms_norm 미러(비트동일). 곱 순서 ((o·scale)·nw)·sigmoid(z)는
// ops.rs rms_norm map(v*scale*g) × gdn_norm_gated(n[i]*gate_act) 결합.
extern "C" __global__ void fn_gdn_gate(
    const float* __restrict__ o,   // [T][v_len] scan 산출 o_all
    const float* __restrict__ z,   // [T][v_len] z(투영 게이트 입력)
    const float* __restrict__ nw,  // [L][128] ssm_norm.weight(헤드 공유)
    float* __restrict__ gated,     // [T][v_len] gated
    int t_len, int layer, int dt_rank, float eps)
{
    int t = blockIdx.y;
    int h = blockIdx.x;
    int tid = threadIdx.x;
    if (t >= t_len) return;
    int base = t * (dt_rank * 128) + h * 128;
    float ov = o[base + tid];
    __shared__ float part[32];
    __shared__ float scale_s;
    if (tid < 32) {
        // ops.rs sq_sum L11-31 미러 — SEG=32·chunk=128/32=4, 세그먼트
        // 내 f32 순차, 세그먼트 경계 [4u, 4u+4).
        int u = tid;
        float p = 0.0f;
        for (int i = u * 4; i < u * 4 + 4; i++) {
            float x = o[base + i];
            p += x * x;
        }
        part[u] = p;
    }
    __syncthreads();
    if (tid == 0) {
        double sum = 0.0;
        for (int u = 0; u < 32; u++) {
            sum += (double)part[u];
        }
        scale_s = 1.0f / (float)sqrt(sum / 128.0 + (double)eps);
    }
    __syncthreads();
    float zv = z[base + tid];
    gated[base + tid] =
        ((ov * scale_s) * nw[layer * 128 + tid]) * fn_sigmoid(zv);
}


// fn_gdn_scan — scan PORT(판정표 2차 확정, 2026-10-05 실측 근거). 원천:
// core/gdn.rs gdn_chunk_seq L22·gdn_chunk_head L66-197 — CS=64·f32 전체·
// qp 선스케일·전진 소거(d[i] = β·v − β·e^{gcs_i}·(k·S) 후 j<i 감산)·
// o는 소거 완료 후 별도 루프·상태 P6 갱신(st·e^{g_last} + Σ_j k·e^차·d).
// 커널 구조: 그리드 (h_v, 1), 블록 128 — tid = v-열(dv). core의 열별
// 독립 스칼라 루프를 열-스레드로 1:1 재생: 각 열은 s2/j 순차 누산을
// 원천과 동일 순서로 수행(도트는 전 스레드가 동일 순서로 재계산 —
// 동일 비트). d_out 열 접근은 동일 tid만 r/w(교차 스레드 해저드 없음).
//
// [REUSE → PORT 전환 근거 — G5 exl3_gdn_scan 재사용 불가 실측 2건]
// (1) gdn_expf 정의역: k-비트 재구성이 1023+k ≥ 0(≈x ≥ −709) 요구.
//     Flash-Next 실측 ssm_a(blk.0: −0.028..−158) × softplus(0.02..0.9)
//     → 단계 g 최대 −140 → T=32 cumsum gcs < −900 → gdn_expf(gcs)가
//     쓰레기 비트(NaN) — 실측: ord=0에서 scan o nan=9088·상태 nan=65536
//     (헤드 4개 전량). 27B/35B EXL3 가중치 계급(단계 |g| 수 이내)에서는
//     미발생 — Flash-Next 형상에서 처음 노출.
// (2) f16 저장(sk/sv/A/KQ/KS/QS) 오차 1.7e-5는 통과지만 게이트 rms
//     스케일(1/√(mean(o²)+eps), eps=1e-6 → 최대 ~1000배)이 소-o 헤드에서
//     오차를 증폭 — 실측 종단 1.7e-3~2.6e-3 > 임계 2e-4.
// PORT 설계로 양쪽 모두 해소: f32 전체 저장(2번) + expf(IEEE 하류
// 언더플로우 → e^−900 = 0, 1번) + core와 동일 적산 순서(비트 근접).
// 잔차 오차 계급: expf(장치) vs 호스트 libm .exp() ~1-2 ulp뿐(원천이
// core/gdn.rs에서 std f32 .exp()를 쓰기 때문 — exp_cr 아님에 주의).
//
// [CMP 170HX(sm_80, GA100 70SM) 설계 근거 — plans/124 §0]
// 청크 순차 상태 의존(헤드 이상 병렬화 불가 — G5 동일 제약) → 그리드
// h_v=48 블록, 블록 128스레드. 동적 공유 98,820B(qp/kp/d_out 32KB×3 +
// bp 256B + gcs 260B) → opt-in 필요(정적 48KB 초과), GA100 164KB/SM
// 대비 1블록/SM 상주 — sm_89 개발기(99KB 상한)에도 적합. 상태는 전역
// 메모리 r/w(열-스트라이드 접근 — 정합 우선, SASS 수준 최적화는 170HX
// 도착 후 판단. d_out/qp/kp은 smem 브로드캐스트). reg: strided st 접근
// 때문에 스레드당 상수 수준 — 점유 제약은 smem 1블록/SM이 지배.
#define FN_GDN_CS 64
extern "C" __global__ void fn_gdn_scan(
    const float* __restrict__ q,     // [T][k_len] l2 완료(미스케일 — qp·scale 선적용)
    const float* __restrict__ k,     // [T][k_len] l2 완료
    const float* __restrict__ v,     // [T][v_len] conv 자연 순서
    const float* __restrict__ bg,    // [T][2*h_v] beta|g(자연 순서)
    float* __restrict__ st,          // [L][h_v][128*128] r/w — S0≠0 경로 의무
    float* __restrict__ outv,        // [T][v_len] o_all
    int t_len, int h_k, int h_v, int d, int layer)
{
    extern __shared__ char smem_raw[];
    float* qp = (float*)smem_raw;            // [CS*128] — 선스케일 완료
    float* kp = qp + FN_GDN_CS * d;          // [CS*128]
    float* d_out = kp + FN_GDN_CS * d;       // [CS*128]
    float* bp = d_out + FN_GDN_CS * d;       // [CS]
    float* gcs = bp + FN_GDN_CS;             // [CS]
    int h = blockIdx.x;
    int kh = h % h_k;
    int tid = threadIdx.x;
    float scale = 1.0f / sqrtf((float)d);
    long st_h = (long)layer * h_v * d * d + (long)h * d * d;
    int n_chunks = (t_len + FN_GDN_CS - 1) / FN_GDN_CS;
    for (int c = 0; c < n_chunks; c++) {
        int t0 = c * FN_GDN_CS;
        int n = min(t_len - t0, FN_GDN_CS);
        // 적재: t<n은 실값, 패드 0 — core 원천이 복사→제로 패드→일괄
        // ·scale 순서(0·scale=0이라 동일 비트).
        for (int e = tid; e < FN_GDN_CS * d; e += 128) {
            int t = e / d;
            float qv = 0.0f, kv2 = 0.0f;
            if (t < n) {
                int src = t0 + t;
                qv = q[src * (h_k * d) + kh * d + (e % d)];
                kv2 = k[src * (h_k * d) + kh * d + (e % d)];
            }
            qp[e] = qv * scale;
            kp[e] = kv2;
        }
        if (tid == 0) {
            float acc = 0.0f;
            for (int t = 0; t < FN_GDN_CS; t++) {
                acc += (t < n) ? bg[(t0 + t) * (2 * h_v) + h_v + h] : 0.0f;
                gcs[t] = acc;
            }
            for (int t = 0; t < FN_GDN_CS; t++) {
                bp[t] = (t < n) ? bg[(t0 + t) * (2 * h_v) + h] : 0.0f;
            }
        }
        __syncthreads();
        // 전진 소거 + o 산출 — core i-루프와 동일 순서. 열(tid)별 레지스터
        // rhs/oi, d_out 열은 자기 tid만 r/w.
        for (int i = 0; i < n; i++) {
            float beta_i = bp[i];
            float vp = (i < n) ? v[(t0 + i) * (h_v * d) + h * d + tid] : 0.0f;
            float rhs = beta_i * vp;
            if (beta_i != 0.0f) {
                float w0 = beta_i * expf(gcs[i]);
                for (int s2 = 0; s2 < d; s2++) {
                    float ks = kp[i * d + s2];
                    if (ks == 0.0f) continue;
                    float w = w0 * ks;
                    rhs -= w * st[st_h + (long)s2 * d + tid];
                }
            }
            float di = rhs;
            for (int j = 0; j < i; j++) {
                float dot = 0.0f;
                for (int s2 = 0; s2 < d; s2++) {
                    dot += kp[i * d + s2] * kp[j * d + s2];
                }
                float aij = dot * beta_i * expf(gcs[i] - gcs[j]);
                if (aij == 0.0f) continue;
                di -= aij * d_out[j * d + tid];
            }
            d_out[i * d + tid] = di;
            float oi = 0.0f;
            float qi_exp = expf(gcs[i]);
            for (int s2 = 0; s2 < d; s2++) {
                float qv = qp[i * d + s2];
                if (qv == 0.0f) continue;
                float w = qi_exp * qv;
                oi += w * st[st_h + (long)s2 * d + tid];
            }
            for (int j = 0; j <= i; j++) {
                float dot = 0.0f;
                for (int s2 = 0; s2 < d; s2++) {
                    dot += qp[i * d + s2] * kp[j * d + s2];
                }
                float kqij = dot * expf(gcs[i] - gcs[j]);
                if (kqij == 0.0f) continue;
                oi += kqij * d_out[j * d + tid];
            }
            outv[(t0 + i) * (h_v * d) + h * d + tid] = oi;
        }
        __syncthreads();
        // 상태 P6 갱신 — core: 전체 ·e^{g_last} 후 j 승순 가산(원소별
        // += 각각 f32 반올림 — 열 재생).
        float g_last = gcs[FN_GDN_CS - 1];
        float gl_exp = expf(g_last);
        for (int s2 = 0; s2 < d; s2++) {
            st[st_h + (long)s2 * d + tid] *= gl_exp;
        }
        for (int j = 0; j < n; j++) {
            float w = expf(g_last - gcs[j]);
            for (int s2 = 0; s2 < d; s2++) {
                float kv = kp[j * d + s2] * w;
                st[st_h + (long)s2 * d + tid] += kv * d_out[j * d + tid];
            }
        }
        __syncthreads();
    }
}

// fn_gdn_conv3 — 음성대조 계기(원장 17호: 계기 자체 검증). conv_k=4
// 계약을 3탭으로 오독한 변형(w[2]를 현행 탭으로, 링 2깊이 회전) —
// 프로덕션 경로 발사 금지(검증층 fn_gdn_negative_check 전용). silu는
// fn_exp_cr 미러(ulp 계급 — 탭 누락 오차 ~1e-2가 지배하므로 계기
// 판별에는 무영향). 링 row 2는 미기록(스테일 — 계기 전용 버퍼 계약).
extern "C" __global__ void fn_gdn_conv3(
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
    float h0 = ring[layer * 3 * conv_ch + 0 * conv_ch + ch];
    float h1 = ring[layer * 3 * conv_ch + 1 * conv_ch + ch];
    for (int t = 0; t < t_len; t++) {
        float xt = qkv[t * conv_ch + ch];
        float o = (w2 * xt + w0 * h0 + w1 * h1);
        o = o / (1.0f + fn_exp_cr(-o));
        if (ch < k_len) {
            q_out[t * k_len + ch] = o;
        } else if (ch < 2 * k_len) {
            k_out[t * k_len + (ch - k_len)] = o;
        } else {
            v_out[t * v_len + (ch - 2 * k_len)] = o;
        }
        h0 = h1;
        h1 = xt;
    }
    ring[layer * 3 * conv_ch + 0 * conv_ch + ch] = h0;
    ring[layer * 3 * conv_ch + 1 * conv_ch + ch] = h1;
}
