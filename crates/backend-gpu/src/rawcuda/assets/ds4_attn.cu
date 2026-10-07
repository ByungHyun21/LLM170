// ── DeepSeek-V4-Flash 어텐션 스테이지 CUDA 커널 (plans/130 B3, 2026-10-05) ──
// 산술 원천 = crates/core/src/deepseek4/stages/attn.rs(CPU 황금 계약, 1017행)
// + crates/core/src/deepseek4/ops.rs(QAT 시뮬·RMS·RoPE·Hadamard) — 전 커널이
// 이 두 파일의 연산 순서를 줄 단위로 미러한다(값 판정 — plans/124 §5 계승).
// 모듈층(ds4_attn_cuda.rs)·검증층(ds4_attn_cuda_probe.rs)과 3층 분리.
//
// [스테이지 구조 — 원천 줄번호(워크트리 c5da4b9a 기준)]
// - Q 프로젝션:   attn.rs project_q L84-120 — fp8_sim 활성(x, 128블록) →
//                  wq_a gemm → bf16 → q_norm 가중 RMS → fp8_sim(c, 128블록) →
//                  wq_b gemm → bf16 → 헤드별 비가중 RMS ×rsqrt → bf16.
// - Q RoPE:       attn.rs rope_q L122-135 — 헤드별 마지막 64디m rope(위치 t)
//                  + bf16(ops.rs rope_apply L335-350 인터리브 쌍 회전).
// - KV 프로젝션:  attn.rs project_kv L137-158 — fp8_sim(x) → wkv gemm → bf16 →
//                  kv_norm 가중 RMS → rope 마지막 64 + bf16 → 비로프 448디m
//                  FP8-sim(64블록 — 로프 딤 bf16 유지, 보고서 §9.2).
// - 컴프레서 풀:  attn.rs compressor_pool_prefill L183-229(CSA r=4 겹침 8행
//                  [이전 블록 첫반부 | 현재 블록 둘째반부], i=0 패드 -inf ·
//                  HCA r=128 비겹침) + pool_rows L231-254(디m별 소프트맥스 —
//                  스코어 = wgate 출력 + ape, 가중합. 합은 두 개의 별도
//                  r-오름차순 루프: s 먼저, acc 나중 — 반올림 순서 계약).
// - 컴프레서 종결: attn.rs compressor_finish L256-281 — 가중 RMS(norm) →
//                  블록 시작 위치(i·ratio) rope + bf16 → rotate면 Hadamard128
//                  +bf16+FP4-sim(인덱서 전용), 아니면 비로프 FP8-sim(64블록).
// - 인덱서 qI:    attn.rs indexer_q L394-427 — c_Q 공유 입력 fp8_sim → wq_b
//                  gemm → bf16 → 헤드 rope+bf16 → Hadamard128 → bf16 → FP4-sim.
// - 인덱서 가중치: attn.rs indexer_weights L452-463 — weights_proj gemm →
//                  bf16(bf16(v)·c) 이중 경계(스케일 c = 128^-0.5·64^-0.5).
// - 인덱서 스코어: attn.rs indexer_scores L465-492 — score[t][b] =
//                  bf16(Σ_h w·ReLU(qI·kI)) — h 오름차순 f32 누산.
// - top-k 선택:   attn.rs indexer_topk L494-529 — 인과 b < (t+1)//ratio,
//                  값 내림차순·동점 낮은 인덱스 우선(모듈 계약), +offset.
//                  랭크 산정으로 결정적 재현: rank(b) = |{b'≠b : s'>s ∨
//                  (s'==s ∧ b'<b)}| — rank<k인 후보만, 순위 위치에 기입.
// - 스파스 어텐션: attn.rs sparse_attn_one L531-580 — MQA(kv 512 공유,
//                  scale=512^-0.5), 일괄 소프트맥스 + 싱크 로짓 z':
//                  o = Σ bf16(p)·v / (Σp + exp(z'-m)) — p의 bf16 캐스트가
//                  커널·오라클 공유 경계. e-오름차순 누산(acc[dd]는 e순).
// - 출력 비회전:  attn.rs attention_forward L646-658 — o 마지막 64디m rope^-1
//                  (켤레) at 쿼리 위치 t + bf16.
// - 그룹 출력:    attn.rs attention_output L582-599 — 8그룹 wo_a(4096→1024)
//                  → latents[8192] bf16 → FP8-sim(128블록, 행=8192 전체) →
//                  wo_b(8192→4096) → bf16.
//
// [빌드 계약] 별도 -fmad=false 블록(scripts/build_cuda.bat — exl3_gdn/attn/
// ew/q4/fn 계열과 동일 노선). 사유: (1) 모든 f32 누산(gemm k-오름차순,
// RMS 제곱합, 풀 가중합, 스코어 h-합, 어텐션 e-합)이 FFMA 수축되면 곱·합
// 각 1회 반올림인 코어 미러(ops.rs gemm_nt L58-75 등)와 비트가 어긋난다.
// (2) 초월함수는 자작 f64 트윈(ds4_exp_cr — ops.rs exp_cr L52-98 과 리터럴
// 까지 동일한 fma 호너 13단; 명시적 __fma_rn 이라 -fmad=false 와 무관하게
// 단일 반올림 유지 — FNC hc_exp_cr 입증 노선). (3) RoPE cos/sin 테이블은
// 호스트에서 ops.rs RopeTable::build L280-325(f64 powf + YaRN 램프) 대로
// 구축한 [len][half][2] f32 표를 업로드해 커널이 읽는다(테이블 구동 —
// 장치 libdevice powf/cosf/sinf 전부 배제, 양측 동일 비트 보장).
// (4) sqrt/div는 IEEE sqrt.rn/div.rn — 호스트 f32 sqrt/나눗셈과 동일 비트.
//
// [부동 소수 환경] -ftz=false(기본 — 유효숫 하위 유지), 부호 있는 0 보존
// (bf16 반올림·e2m1 copysign). 이 파일 내 어떤 libdevice 초월함수·근사
// 내장(rsqrtf/__expf/powf)도 사용 금지 — 비트동일 계약 위반.
//
// [그리드 계약(결함 5호 정신)] 토큰 t·헤드 h 축은 grid.x/y로 명시.
// gemm/원소wise는 평탄화 1차원(블록 256), 행·엔트리 단위 커널은
// grid=(행수)×블록 스레드=행 원소 분담이 아닌 1스레드=1행(순차 누산
// 순서가 비트 계약이므로 스레드 내 직렬 — 정합 우선 설계, G6 원장 계승).
//
// [CMP 170HX(sm_80) 설계 근거 — plans/124 §0] 어텐션 스파스 커널은
// 1스레드=1(t,h)에 acc[512] 로컬(2KB/스레드) — E≤640 행 순차 처리.
// T=2052×64h=131k 스레드 → GA100 70SM 확산, 로컬 트래픽 ≈0.26GB/체인
// (HBM2e 무해수). gemm 은 1스레드=1출력(정합 우선 — t-블록 tiled 변형은
// sm_80 실측 후 재판정, G6 fwd3s 원장과 동일 계급). 속도 칸 '측정 대기
// sm_80'(4070 SUPER 타이밍 금지 — plans/130 §0).
//
// [음성대조 쌍둥이 — 원장 17호] 3종(프로덕션 발사 금지, 검증층 전용):
// (a) llm170_ds4_topk_vis1 — 인과 가시경계 off-by-one(visible+1).
// (b) llm170_ds4_pool_apeoff — ape 행 j→(j+1)%ratio 미스얼라인.
// (c) llm170_ds4_sparse_attn_nosink — 싱크 로짓 누락(denom 에서 z' 항 제외).

#include <cuda_runtime.h>

// ── 비트 미러 헬퍼(전부 ops.rs 직이식 — 연산 순서·리터럴 동일) ──

// ops.rs bf16_round L17-23 — RNE 1회, 부호 있는 0 보존.
__device__ __forceinline__ float ds4_bf16r(float x)
{
    unsigned b = __float_as_uint(x);
    unsigned hi = ((unsigned)(b >> 16) & 1u);
    return __uint_as_float(((b + 0x7FFFu + hi) >> 16) << 16);
}

// ops.rs pow2_ceil L77-85 — 2^ceil(log2 x) 비트 경로(x>0 정규수 가정).
__device__ __forceinline__ float ds4_pow2_ceil(float x)
{
    unsigned b = __float_as_uint(x);
    int e = (int)((b >> 23) & 0xFFu);
    int l2 = e - 127 + (int)((b & 0x7FFFFFu) != 0u);
    return __uint_as_float(((unsigned)(l2 + 127)) << 23);
}

// ops.rs f32_to_e4m3 L88-129 — S|EEEE|MMM RNE, 비정규 영역 포함.
__device__ __forceinline__ unsigned char ds4_f32_to_e4m3(float x)
{
    unsigned char sign = (__signbitf(x)) ? 0x80u : 0x00u;
    float a = fabsf(x);
    if (a < 9.765625e-4f) {          // 2^-10 미만 → 0(RNE 동점 짝수)
        return sign;
    }
    if (a < 0.015625f) {             // 2^-6 미만 비정규: ulp 2^-9
        float q = a * 512.0f;
        float r = rintf(q);          // round-ties-even
        if (r >= 8.0f) {
            return sign | 0x08u;     // 8·2^-9 → 최소 정규수 승격
        }
        return sign | (unsigned char)r;
    }
    unsigned b = __float_as_uint(a);
    int e = (int)((b >> 23) & 0xFFu) - 127;
    unsigned mant = b & 0x7FFFFFu;
    unsigned rem = mant & 0xFFFFFu;  // 버려지는 하위 20비트
    unsigned half = 0x80000u;
    unsigned mm = mant >> 20;
    int ee = e;
    if (rem > half || (rem == half && (mm & 1u) != 0u)) {
        mm += 1;
    }
    if (mm == 8) {
        mm = 0;
        ee += 1;
    }
    unsigned char e4 = (unsigned char)(ee + 7);
    if (e4 > 15 || (e4 == 15 && mm > 7)) {
        return sign | 0x7Fu;         // 448 초과 — 방어적 NaN(도달 불가 경로)
    }
    return sign | (e4 << 3) | (unsigned char)mm;
}

// ops.rs e4m3_to_f32 L131-148 — 정확 확장(부호는 ±1.0 곱).
__device__ __forceinline__ float ds4_e4m3_to_f32(unsigned char u)
{
    float sv = (u & 0x80u) ? -1.0f : 1.0f;
    int e4 = (int)((u >> 3) & 0xFu);
    unsigned m = (unsigned)(u & 7u);
    if (e4 == 15 && m == 7) {
        return __int_as_float(0x7FC00000) * sv;  // NaN(미사용 경로)
    }
    if (e4 == 0) {
        return sv * (float)m * 0.001953125f;     // m·2^-9
    }
    unsigned bits = (((unsigned)(e4 - 7 + 127)) << 23) | (m << 20);
    return sv * __uint_as_float(bits);
}

// ops.rs f32_to_e2m1 L150-160 — 그리드 {0,.5,1,1.5,2,3,4,6}, |x|≤6 가정.
__device__ __forceinline__ float ds4_f32_to_e2m1(float x)
{
    float a = fabsf(x);
    float v;
    if (a <= 0.25f) v = 0.0f;
    else if (a < 0.75f) v = 0.5f;
    else if (a <= 1.25f) v = 1.0f;
    else if (a < 1.75f) v = 1.5f;
    else if (a <= 2.5f) v = 2.0f;
    else if (a < 3.5f) v = 3.0f;
    else if (a <= 5.0f) v = 4.0f;
    else v = 6.0f;
    return copysignf(v, x);
}

// ops.rs exp_cr L52-98 직이식 — f64 fma 호너 13단 + 2^k 비트 재구성.
// 명시적 __fma_rn 은 -fmad=false 와 무관하게 단일 반올림(mul_add ≡).
// 도메인 가드: x>88.72 → +inf, x<-103.97 → 0(-inf 패드 행 포함).
__device__ __forceinline__ float ds4_exp_cr(float x)
{
    const double LN2_HI = 6.931471803691238e-1;
    const double LN2_LO = 1.9082149292705877e-10;
    const double INV_LN2 = 1.4426950408889634;   // f64::LOG2_E 비트동일
    double xd = (double)x;
    if (xd > 88.72) return __int_as_float(0x7F800000);
    if (xd < -103.97) return 0.0f;
    double kd = rint(xd * INV_LN2);              // round-ties-even(f64)
    long long k = (long long)kd;
    double r = __fma_rn(-kd, LN2_HI, xd);
    r = __fma_rn(-kd, LN2_LO, r);
    double p = 1.0 / 1307674368000.0;
    p = __fma_rn(p, r, 1.0 / 479001600.0);
    p = __fma_rn(p, r, 1.0 / 39916800.0);
    p = __fma_rn(p, r, 1.0 / 3628800.0);
    p = __fma_rn(p, r, 1.0 / 362880.0);
    p = __fma_rn(p, r, 1.0 / 40320.0);
    p = __fma_rn(p, r, 1.0 / 5040.0);
    p = __fma_rn(p, r, 1.0 / 720.0);
    p = __fma_rn(p, r, 1.0 / 120.0);
    p = __fma_rn(p, r, 1.0 / 24.0);
    p = __fma_rn(p, r, 1.0 / 6.0);
    p = __fma_rn(p, r, 0.5);
    p = __fma_rn(p, r, 1.0);
    p = __fma_rn(p, r, 1.0);
    if (k > 127) return __int_as_float(0x7F800000);
    double scale = __longlong_as_double((k + 1023) << 52);
    return (float)(p * scale);
}

// ops.rs rms_scale L48-55 — 비가중 RMS 배율(f32 스케일, 순차 제곱합).
__device__ __forceinline__ float ds4_rms_scale(const float* x, int n, float eps)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
    return 1.0f / sqrtf(sum / (float)n + eps);
}

// ops.rs rms_norm_weighted L33-46 — 가중 RMS, 출력 bf16 경계(호출부).
// out 은 x 와 별개 버퍼(제자리 아님 — 코어도 map 수집).
__device__ __forceinline__ void ds4_rms_norm_weighted(
    const float* x, const float* w, float* out, int n, float eps)
{
    float s = ds4_rms_scale(x, n, eps);
    for (int i = 0; i < n; i++) out[i] = ds4_bf16r(x[i] * s * w[i]);
}

// ops.rs rope_apply L335-350 — 인터리브 쌍 복소수 회전(inverse → 켤레).
// cs 는 [pos][half][2]=(cos,sin) 표(호스트 RopeTable::build 산출물)의
// pos 위치 슬라이스. row 길이 = 2·half.
__device__ __forceinline__ void ds4_rope_apply(
    float* row, const float* cs, int half, bool inverse)
{
    for (int p = 0; p < half; p++) {
        float c = cs[p * 2];
        float s = inverse ? -cs[p * 2 + 1] : cs[p * 2 + 1];
        float x0 = row[p * 2];
        float x1 = row[p * 2 + 1];
        row[p * 2] = x0 * c - x1 * s;
        row[p * 2 + 1] = x0 * s + x1 * c;
    }
}

// ops.rs hadamard_rotate L243-261 — 자연 순서 WHT + N^-0.5(n=128).
__device__ __forceinline__ void ds4_had128(float* v, int n)
{
    int width = 1;
    while (width < n) {
        int base = 0;
        while (base < n) {
            for (int i = 0; i < width; i++) {
                float a = v[base + i];
                float b = v[base + width + i];
                v[base + i] = a + b;
                v[base + width + i] = a - b;
            }
            base += 2 * width;
        }
        width *= 2;
    }
    float sc = 1.0f / sqrtf((float)n);
    for (int i = 0; i < n; i++) v[i] *= sc;
}

// ops.rs fp8_sim L174-192 — 1블록(스레드 1개가 블록 순차 처리).
// amax 하한 1e-4, s=pow2_ceil(amax·(1/448)), ±448 클램프 e4m3 RNE 왕복.
__device__ __forceinline__ void ds4_fp8_sim_blk(float* blk, int n)
{
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float av = fabsf(blk[i]);
        amax = fmaxf(amax, av);
    }
    amax = fmaxf(amax, 1e-4f);
    float s = ds4_pow2_ceil(amax * (1.0f / 448.0f));
    for (int i = 0; i < n; i++) {
        float q = blk[i] / s;
        q = (q < -448.0f) ? -448.0f : (q > 448.0f) ? 448.0f : q;
        blk[i] = ds4_e4m3_to_f32(ds4_f32_to_e4m3(q)) * s;
    }
}

// ops.rs fp4_sim L194-210 — 32원소 블록, amax 하한 6·2^-126(=0x01400000),
// s=pow2_ceil(amax·(1/6)), ±6 클램프 e2m1 + copysign.
__device__ __forceinline__ void ds4_fp4_sim_32(float* blk)
{
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {
        float av = fabsf(blk[i]);
        amax = fmaxf(amax, av);
    }
    amax = fmaxf(amax, __int_as_float(0x01400000));
    float s = ds4_pow2_ceil(amax * (1.0f / 6.0f));
    for (int i = 0; i < 32; i++) {
        float q = blk[i] / s;
        q = (q < -6.0f) ? -6.0f : (q > 6.0f) ? 6.0f : q;
        blk[i] = copysignf(ds4_f32_to_e2m1(q), q) * s;
    }
}

// ── 1. FP8-sim 행 블록 — ops.rs fp8_sim L174-192 (flat (row,c0)) ──
// buf: [rows][row_stride], 처리 폭 cols(c0 블록 분할 — 마지막 블록 단축).
extern "C" __global__ void llm170_ds4_fp8_rows(
    float* buf, int rows, int row_stride, int cols, int block)
{
    int nblk = (cols + block - 1) / block;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * nblk) return;
    int r = idx / nblk;
    int b = idx - r * nblk;
    int lo = b * block;
    int hi = (lo + block < cols) ? lo + block : cols;
    ds4_fp8_sim_blk(buf + (long)r * row_stride + lo, hi - lo);
}

// ── 2. gemm — ops.rs gemm_nt L58-75 (1스레드=1출력, k 오름차순 f32) ──
// y[i][j] = Σ_k x[i][j_기준 kk]·w[k][n+j] — w 는 k-major [k][n]. mul+add 각
// 1회 반올림(-fmad=false) — 코어와 동일 비트. x_stride/y_stride 는 행 폭
// (packed 젅이면 k/n — 그룹 출력 wo_a 의 o 열 슬라이스·latents 열 슬라이스
// 적재용, attn.rs attention_output L586-591 의 슬라이스 gemm 미러).
extern "C" __global__ void llm170_ds4_gemm(
    const float* x, const float* w, float* y, int t, int k, int n,
    int x_stride, int y_stride)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)t * n;
    if (idx >= total) return;
    int i = (int)(idx / n);
    int j = (int)(idx - (long)i * n);
    const float* xr = x + (long)i * x_stride;
    float acc = 0.0f;
    for (int kk = 0; kk < k; kk++) {
        acc += xr[kk] * w[(long)kk * n + j];
    }
    y[(long)i * y_stride + j] = acc;
}

// ── 3. bf16 경계 일괄 — ops.rs bf16_round_slice L25-30 ──
extern "C" __global__ void llm170_ds4_bf16_rows(float* buf, long n)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    buf[idx] = ds4_bf16r(buf[idx]);
}

// ── 4. 가중 RMS 행 — ops.rs rms_norm_weighted L33-46 (1스레드=1행) ──
extern "C" __global__ void llm170_ds4_rmsw_rows(
    const float* x, const float* w, float* y, int rows, int cols, float eps)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    ds4_rms_norm_weighted(x + (long)r * cols, w, y + (long)r * cols, cols, eps);
}

// ── 5. 헤드별 비가중 RMS — attn.rs project_q L110-119 (제자리) ──
// q[t][h·hd .. ] 헤드 전체에 ×rsqrt(mean+eps) 후 bf16.
extern "C" __global__ void llm170_ds4_head_rms(
    float* q, int t, int heads, int hd, float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= t * heads) return;
    float* head = q + (long)idx * hd;
    float s = ds4_rms_scale(head, hd, eps);
    for (int d = 0; d < hd; d++) head[d] = ds4_bf16r(head[d] * s);
}

// ── 6. RoPE 꼬리 — attn.rs rope_q L122-135 / project_kv L148-151 ──
// buf[row][h·hd .. ] 마지막 rd 디m 회전(위치=행 인덱스) + bf16.
// cs: [len][half][2] f32(호스트 RopeTable). half = rd/2.
extern "C" __global__ void llm170_ds4_rope_tail(
    float* buf, int rows, int heads, int hd, int rd,
    const float* cs, int half)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * heads) return;
    int row = idx / heads;
    float* head = buf + (long)idx * hd;
    ds4_rope_apply(head + hd - rd, cs + (long)row * half * 2, half, false);
    for (int d = hd - rd; d < hd; d++) head[d] = ds4_bf16r(head[d]);
}

// ── 7. 컴프레서 게이트 풀 — attn.rs compressor_pool_prefill L183-229 +
//    pool_rows L231-254 (1스레드=1(블록 i, 디m dd)) ──
// kv_c/score_c: [t][coff·d], ape: [ratio][coff·d], out: [nb][d].
// 겹침(coff=2): 풀 행 = [이전 블록 첫반부(스코어+ape 첫반부) |
// 현재 블록 둘째반부(ape 둘째반부)], i=0 이전 반부는 -inf 패드(가중치 0).
// 비겹침(coff=1): 현재 블록 전체. 소프트맥스 가중 w[] 로컬 보관 후
// s-합 → acc-합 순(두 루프 — 코어 반올림 순서 미러).
extern "C" __global__ void llm170_ds4_pool(
    const float* kv_c, const float* sc_c, const float* ape, float* out,
    int nb, int d, int ratio, int overlap)
{
    int coff = 1 + overlap;
    int cd = coff * d;
    int rows = coff * ratio;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nb * d) return;
    int i = idx / d;
    int dd = idx - i * d;
    float pkv[128];   // rows ≤ 256(coff=2·ratio=128은 등장하지 않음 —
    float psc[128];   //  CSA 8행·HCA 128행. 상방 가드는 모듈층 형상 검사).
    float w[128];
    for (int r = 0; r < rows; r++) {
        pkv[r] = 0.0f;
        psc[r] = -INFINITY;
    }
    int row_off = overlap ? ratio : 0;
    int src = overlap ? d : 0;
    for (int j = 0; j < ratio; j++) {
        int cur = i * ratio + j;
        pkv[row_off + j] = kv_c[(long)cur * cd + src + dd];
        psc[row_off + j] = sc_c[(long)cur * cd + src + dd] + ape[(long)j * cd + src + dd];
        if (overlap && i > 0) {
            int prev = cur - ratio;
            pkv[j] = kv_c[(long)prev * cd + dd];
            psc[j] = sc_c[(long)prev * cd + dd] + ape[(long)j * cd + dd];
        }
    }
    // pool_rows L231-254: m → w=exp_cr(−) → s → acc (r 오름차순 두 루프).
    float m = -INFINITY;
    for (int r = 0; r < rows; r++) m = fmaxf(m, psc[r]);
    float s = 0.0f;
    for (int r = 0; r < rows; r++) {
        w[r] = ds4_exp_cr(psc[r] - m);
        s += w[r];
    }
    float acc = 0.0f;
    for (int r = 0; r < rows; r++) acc += w[r] * pkv[r];
    out[(long)i * d + dd] = acc / s;
}

// ── 8. 컴프레서 종결 — attn.rs compressor_finish L256-281 (1스레드=1엔트리) ──
// pooled[i][d] 제자리: 가중 RMS → rope 꼬리(위치 i·ratio)+bf16 →
// rotate ? had128+bf16+fp4 : fp8(첫 d−rd, 64블록).
extern "C" __global__ void llm170_ds4_comp_finish(
    float* buf, const float* norm, int nb, int d, int rd, int ratio,
    int rotate, const float* cs, int half, float eps)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nb) return;
    float* row = buf + (long)i * d;
    float tmp[512];
    ds4_rms_norm_weighted(row, norm, tmp, d, eps);
    for (int c = 0; c < d; c++) row[c] = tmp[c];
    long pos = (long)i * ratio;
    ds4_rope_apply(row + d - rd, cs + pos * half * 2, half, false);
    for (int c = d - rd; c < d; c++) row[c] = ds4_bf16r(row[c]);
    if (rotate) {
        ds4_had128(row, d);
        for (int c = 0; c < d; c++) row[c] = ds4_bf16r(row[c]);
        for (int c0 = 0; c0 < d; c0 += 32) {
            ds4_fp4_sim_32(row + c0);
        }
    } else {
        int cols = d - rd;
        for (int c0 = 0; c0 < cols; c0 += 64) {
            int hi = (c0 + 64 < cols) ? c0 + 64 : cols;
            ds4_fp8_sim_blk(row + c0, hi - c0);
        }
    }
}

// ── 9. 인덱서 qI 종결 — attn.rs indexer_q L407-426 (1스레드=1(t,h)) ──
// gemm+bf16 이후 q[t][h·id..] 제자리: rope 꼬리(위치 t)+bf16 →
// had128(전 id)+bf16 → fp4(32블록). id=128 고정 계약(로컬 배열).
extern "C" __global__ void llm170_ds4_indexer_q(
    float* q, int t, int heads, int id, int rd, const float* cs, int half)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= t * heads) return;
    int row = idx / heads;
    float* head = q + (long)idx * id;
    ds4_rope_apply(head + id - rd, cs + (long)row * half * 2, half, false);
    for (int d = id - rd; d < id; d++) head[d] = ds4_bf16r(head[d]);
    ds4_had128(head, id);
    for (int d = 0; d < id; d++) head[d] = ds4_bf16r(head[d]);
    for (int d = 0; d < id; d += 32) ds4_fp4_sim_32(head + d);
}

// ── 10. 인덱서 헤드 가중치 스케일 — attn.rs indexer_weights L459-462 ──
// w = bf16(bf16(v)·c) — c = 128^-0.5·64^-0.5(호스트 산출 f32 인자).
extern "C" __global__ void llm170_ds4_iw_scale(float* w, long n, float c)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    w[idx] = ds4_bf16r(ds4_bf16r(w[idx]) * c);
}

// ── 11. 인덱서 스코어 — attn.rs indexer_scores L465-492 ──
// (1스레드=1(t,b)): score = bf16(Σ_h w[t,h]·ReLU(qI·kI)) — h·dd 오름차순.
extern "C" __global__ void llm170_ds4_indexer_scores(
    const float* iq, const float* ki, const float* w, float* out,
    int t, int nb, int ih, int id)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= t * nb) return;
    int ti = idx / nb;
    int bi = idx - ti * nb;
    const float* kb = ki + (long)bi * id;
    float sum = 0.0f;
    for (int h = 0; h < ih; h++) {
        const float* qh = iq + ((long)ti * ih + h) * id;
        float dot = 0.0f;
        for (int d = 0; d < id; d++) dot += qh[d] * kb[d];
        sum += fmaxf(dot, 0.0f) * w[(long)ti * ih + h];
    }
    out[(long)ti * nb + bi] = ds4_bf16r(sum);
}

// ── 12. i32 채움(선택 버퍼 -1 초기화 — 모듈층 편의) ──
extern "C" __global__ void llm170_ds4_fill_i32(int* dst, int val, long n)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = val;
}

// ── 13. top-k 선택 — attn.rs indexer_topk L494-529 ──
// (1스레드=1(t,b)): 랭크 산정식으로 코어의 안정 정렬(값 내림차순·동점
// 낮은 인덱스 우선)을 결정적 재현 — rank(b) = |{b'<visible : s'>s ∨
// (s'==s ∧ b'<b)}|, 선택 ⇒ rank<k, 기입 위치 = rank(정확히 정렬 순서).
// k = min(topk, nb). sel: [t][topk](사전 -1 채움 — 미선택 슬롯).
extern "C" __global__ void llm170_ds4_topk(
    const float* scores, int* sel, int t, int nb, int ratio, int topk,
    int offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= t * nb) return;
    int ti = idx / nb;
    int b = idx - ti * nb;
    int visible = (ti + 1) / ratio;
    if (b >= visible) return;
    int k = topk < nb ? topk : nb;
    float s = scores[(long)ti * nb + b];
    int rank = 0;
    for (int b2 = 0; b2 < visible; b2++) {
        float s2 = scores[(long)ti * nb + b2];
        if (s2 > s || (s2 == s && b2 < b)) rank++;
    }
    if (rank < k) sel[(long)ti * topk + rank] = b + offset;
}

// ── 14. 스파스 어텐션 — attn.rs sparse_attn_one L531-580 ──
// (1스레드=1(t,h), acc[512] 로컬): s[e]=dot·scale(e-오름차순 판독,
// idx<0 건너뜀) → m=max → denom=exp(z'−m)+Σp → acc=Σ bf16(p)·v
// (e-오름차순 — acc[dd]의 결합 순서가 비트 계약) → o=bf16(acc/denom).
// q:[t][nh·hd], kv:[rows][hd], idxs:[t][stride](-1 패드), sink:[nh].
extern "C" __global__ void llm170_ds4_sparse_attn(
    const float* q, const float* kv, const int* idxs, const float* sink,
    float* o, int t, int nh, int hd, int stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= t * nh) return;
    int ti = idx / nh;
    int h = idx - ti * nh;
    const float* qh = q + (long)ti * (long)nh * hd + (long)h * hd;
    const int* row = idxs + (long)ti * stride;
    float scale = 1.0f / sqrtf((float)hd);
    int e;
    float s[640];
    int n_idx = stride;
    for (e = 0; e < n_idx; e++) s[e] = -INFINITY;
    for (e = 0; e < n_idx; e++) {
        int ix = row[e];
        if (ix < 0) continue;
        const float* krow = kv + (long)ix * hd;
        float dot = 0.0f;
        for (int d = 0; d < hd; d++) dot += qh[d] * krow[d];
        s[e] = dot * scale;
    }
    float m = -INFINITY;
    for (e = 0; e < n_idx; e++) m = fmaxf(m, s[e]);
    float denom = ds4_exp_cr(sink[h] - m);
    float acc[512];
    for (int d = 0; d < hd; d++) acc[d] = 0.0f;
    for (e = 0; e < n_idx; e++) {
        if (s[e] == -INFINITY) continue;
        // 코어 순서(sparse_attn_one L557-561): p 산출 → denom += p(무반올림)
        // → p16 = bf16(p) — 반올림 전 p 가 분모에 들어간다.
        float p = ds4_exp_cr(s[e] - m);
        denom += p;
        float p16 = ds4_bf16r(p);
        if (row[e] < 0) continue;
        const float* krow = kv + (long)row[e] * hd;
        for (int d = 0; d < hd; d++) acc[d] += p16 * krow[d];
    }
    float* orow = o + (long)idx * hd;
    for (int d = 0; d < hd; d++) orow[d] = ds4_bf16r(acc[d] / denom);
}

// ── 15. 출력 비회전 — attn.rs attention_forward L646-658 ──
// o[t][h·hd ..] 마지막 rd 디m rope^-1(켤레, 위치 t) + bf16.
extern "C" __global__ void llm170_ds4_derot(
    float* o, int t, int nh, int hd, int rd, const float* cs, int half)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= t * nh) return;
    int row = idx / nh;
    float* head = o + (long)idx * hd;
    ds4_rope_apply(head + hd - rd, cs + (long)row * half * 2, half, true);
    for (int d = hd - rd; d < hd; d++) head[d] = ds4_bf16r(head[d]);
}

// ══ 음성대조 쌍둥이(원장 17호 — 프로덕션 발사 금지, 검증층 전용) ══

// (a) top-k 인과 경계 off-by-one — visible+1(미완 블록 1개 침투).
// 유일한 차이는 visible 산출식(+1) — 나머지는 llm170_ds4_topk 과 동일.
extern "C" __global__ void llm170_ds4_topk_vis1(
    const float* scores, int* sel, int t, int nb, int ratio, int topk,
    int offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= t * nb) return;
    int ti = idx / nb;
    int b = idx - ti * nb;
    int visible = (ti + 1) / ratio + 1;   // 결함 재현: 경계 +1
    if (b >= visible) return;
    int k = topk < nb ? topk : nb;
    float s = scores[(long)ti * nb + b];
    int rank = 0;
    for (int b2 = 0; b2 < visible && b2 < nb; b2++) {
        float s2 = scores[(long)ti * nb + b2];
        if (s2 > s || (s2 == s && b2 < b)) rank++;
    }
    if (rank < k) sel[(long)ti * topk + rank] = b + offset;
}

// (b) 컴프레서 ape 미스얼라인 — ape 행 j 대신 (j+1)%ratio(한 칸 셔프).
// 유일한 차이는 ape 행 인덱스 — 풀 가중치가 어긋나 종단에서 이격.
extern "C" __global__ void llm170_ds4_pool_apeoff(
    const float* kv_c, const float* sc_c, const float* ape, float* out,
    int nb, int d, int ratio, int overlap)
{
    int coff = 1 + overlap;
    int cd = coff * d;
    int rows = coff * ratio;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nb * d) return;
    int i = idx / d;
    int dd = idx - i * d;
    float pkv[128];
    float psc[128];
    float w[128];
    for (int r = 0; r < rows; r++) {
        pkv[r] = 0.0f;
        psc[r] = -INFINITY;
    }
    int row_off = overlap ? ratio : 0;
    int src = overlap ? d : 0;
    for (int j = 0; j < ratio; j++) {
        int jbad = (j + 1) % ratio;       // 결함 재현: ape 행 셔프
        int cur = i * ratio + j;
        pkv[row_off + j] = kv_c[(long)cur * cd + src + dd];
        psc[row_off + j] = sc_c[(long)cur * cd + src + dd] + ape[(long)jbad * cd + src + dd];
        if (overlap && i > 0) {
            int prev = cur - ratio;
            pkv[j] = kv_c[(long)prev * cd + dd];
            psc[j] = sc_c[(long)prev * cd + dd] + ape[(long)jbad * cd + dd];
        }
    }
    float m = -INFINITY;
    for (int r = 0; r < rows; r++) m = fmaxf(m, psc[r]);
    float s = 0.0f;
    for (int r = 0; r < rows; r++) {
        w[r] = ds4_exp_cr(psc[r] - m);
        s += w[r];
    }
    float acc = 0.0f;
    for (int r = 0; r < rows; r++) acc += w[r] * pkv[r];
    out[(long)i * d + dd] = acc / s;
}

// (c) 싱크 로짓 누락 — denom 에서 exp(z'−m) 항 제외(싱크 드롭).
// 유일한 차이는 denom 초기값 — 분모 축소로 종단이 이격.
extern "C" __global__ void llm170_ds4_sparse_attn_nosink(
    const float* q, const float* kv, const int* idxs, const float* sink,
    float* o, int t, int nh, int hd, int stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= t * nh) return;
    int ti = idx / nh;
    int h = idx - ti * nh;
    const float* qh = q + (long)ti * (long)nh * hd + (long)h * hd;
    const int* row = idxs + (long)ti * stride;
    float scale = 1.0f / sqrtf((float)hd);
    int e;
    float s[640];
    for (e = 0; e < stride; e++) s[e] = -INFINITY;
    for (e = 0; e < stride; e++) {
        int ix = row[e];
        if (ix < 0) continue;
        const float* krow = kv + (long)ix * hd;
        float dot = 0.0f;
        for (int d = 0; d < hd; d++) dot += qh[d] * krow[d];
        s[e] = dot * scale;
    }
    float m = -INFINITY;
    for (e = 0; e < stride; e++) m = fmaxf(m, s[e]);
    (void)sink;                               // 결함 재현: 싱크 미사용
    float denom = 0.0f;                       // exp(z'−m) 항 누락
    float acc[512];
    for (int d = 0; d < hd; d++) acc[d] = 0.0f;
    for (e = 0; e < stride; e++) {
        if (s[e] == -INFINITY) continue;
        float p = ds4_exp_cr(s[e] - m);
        denom += p;                       // 싱크항만 누락(p 합은 정상)
        float p16 = ds4_bf16r(p);
        if (row[e] < 0) continue;
        const float* krow = kv + (long)row[e] * hd;
        for (int d = 0; d < hd; d++) acc[d] += p16 * krow[d];
    }
    float* orow = o + (long)idx * hd;
    for (int d = 0; d < hd; d++) orow[d] = ds4_bf16r(acc[d] / denom);
}
