// ── DeepSeek-V4 DSpark MTP 드래프트 스테이지 CUDA 커널 (plans/130 B4, 2026-10-05) ──
// 산술 원천 = crates/core/src/deepseek4/frame.rs DSpark 코어 함수들(CPU 황금
// 계약) + blocks 내부는 착지 모듈(ds4_attn/ds4_hc/ds4_moe) 모듈 API 재사용
// (모듈층 ds4_mtp_cuda.rs · 검증층 ds4_mtp_cuda_probe.rs 와 3층 분리).
//
// [스테이지 구조 — 원천 줄번호(워크트리 3f1ee12d 기준)]
// - main_x:          frame.rs dspark_main_x L208-230 — fp8_sim 활성
//                    (main_hidden 행 12288, 128블록) → main_proj gemm →
//                    bf16 → main_norm 가중 RMS. (커널 1·2·3·4)
// - 드래프트 id:     frame.rs dspark_draft_ids L233-239 — [token, noise×4]
//                    (block_size 5, noise 128799). 호스트 산출(커널 없음).
// - 윈도우 워밍:      frame.rs dspark_window_warm L242-263 — main_x 전 토큰
//                    project_kv(attn.rs L137-158 — ds4_attn.fatbin 재사용)
//                    → 링 [win×512] 호스트 조립.
// - 디코드 어텐션:   frame.rs dspark_decode_attn L267-345 — 메인 토큰 kv
//                    project_kv(rope.at(0) — 1토큰 project_kv 계약) → 링
//                    pos%win 슬롯 기입 · 드래프트 q = project_q +
//                    rope(pos+1..pos+b)(시프트 로프표로 ds4_attn 재사용) ·
//                    드래프트 kv = project_kv_unroped L349-371 산술 =
//                    project_kv(시프트 표)와 동일 비트(로프 딤/비로프 fp8
//                    분리 불변) · [링|드래프트 kv] 스파스 어텐션+싱크(커널 5,
//                    idxs=[0..min(win,pos+1)]++[win..win+b] — 전 드래프트
//                    토큰 동일 행) → 출력 비회전 rope^-1(pos+1.., 커널 6) →
//                    그룹 출력(ds4_attn stage_output 재사용).
// - 종결 헤드:       frame.rs head_gemv_stripwise L41-77 — 공유 head(EXL3
//                    trellis, 호스트 128열 스트립 f64 디양자화 — trellis.rs
//                    dequant_view L128-167) 스트립 gemv. k블록(128)별 부분합
//                    acc 를 y[j] 에 순차 가산, 반올림 경계 없음(f32 순수).
//                    (커널 7)
// - 마르코프 바이어스: frame.rs markov_logits_bias L373-383 —
//                    bias[j] = Σ_k e[k]·w2[k·n+j] (k 오름차순 f32, w1 행은
//                    out_ids 게더 — 임베딩식). (커널 8)
// - 신뢰도:          frame.rs confidence_score L386-397 —
//                    dot([hidden(4096); markov_embed(256)], proj) f32 순차
//                    (x 먼저, markov 나중 — chain 순서 계약). (커널 9)
//
// [빌드 계약] 별도 -fmad=false 블록(scripts/build_cuda.bat — ds4_attn과 동일
// 노선). 사유: 모든 f32 누산(main_proj gemm k-오름차순, 가중 RMS 제곱합,
// 스파스 어텐션 e-오름차순 AV, 헤드 스트립 k블록 부분합, 마르코프 k-합,
// 신뢰도 chain 합)이 FFMA 수축되면 곱·합 각 1회 반올림인 코어 미러
// (ops.rs gemm_nt L58-75, frame.rs L41-77/L373-397)와 비트가 어긋난다.
// 초월함수는 ds4_exp_cr 트윈(ops.rs exp_cr L52-98 과 리터럴 동일 f64 fma
// 호너 — 명시적 __fma_rn 이라 -fmad=false 와 무관). RoPE cos/sin 표는
// 호스트에서 ops.rs RopeTable::build 대로 구축해 업로드(장치 초월함수 배제).
// sqrt/div는 IEEE sqrt.rn/div.rn.
//
// [부동 소수 환경] -ftz=false(기본), 부호 있는 0 보존. libdevice 초월함수·
// 근사 내장 사용 금지 — 비트동일 계약 위반.
//
// [커널 재사용 계약] fp8/gemm/bf16/rmsw/sparse/derot 6종은 ds4_attn.cu 의
// llm170_ds4_* 커널과 산술 리터럴까지 동일(모듈 소유 컨텍스트 분리 — 각
// 모듈이 자체 fatbin 을 적재하는 G10 파일 분할 계약). ds4m_ 접두사만 상이.
//
// [CMP 170HX(sm_80) 설계 근거 — plans/124 §0] 헤드 스트립 gemv 는 1스레드=
// 1(토큰, 스트립 내 j열): 32 k블록 × 128 곱 — 트레일리스 f32 스트리밍
// (1010 스트립 × 2MB — HBM2e 본령). 마르코프는 1스레드=1(토큰, vocab 열)
// 256-항 순차. 정합 우선 설계(4070 SUPER 타이밍 금지 — plans/130 §0).
//
// [음성대조] 본 파일의 쌍둥이 없음 — DSpark 음성대조(트렁크 타깃 순열 ·
// 마르코프 누락 · noise id 오류)는 오라클 변형으로 검증층이 판정한다
// (probe 헤더 [음성대조] 항).

#include <cuda_runtime.h>

// ── 비트 미러 헬퍼(전부 core deepseek4/ops.rs 직이식 — ds4_attn.cu 와 리터럴 동일) ──

// ops.rs bf16_round L17-23 — RNE 1회, 부호 있는 0 보존.
__device__ __forceinline__ float ds4m_bf16r(float x)
{
    unsigned b = __float_as_uint(x);
    unsigned hi = ((unsigned)(b >> 16) & 1u);
    return __uint_as_float(((b + 0x7FFFu + hi) >> 16) << 16);
}

// ops.rs pow2_ceil L77-85 — 2^ceil(log2 x) 비트 경로(x>0 정규수 가정).
__device__ __forceinline__ float ds4m_pow2_ceil(float x)
{
    unsigned b = __float_as_uint(x);
    int e = (int)((b >> 23) & 0xFFu);
    int l2 = e - 127 + (int)((b & 0x7FFFFFu) != 0u);
    return __uint_as_float(((unsigned)(l2 + 127)) << 23);
}

// ops.rs f32_to_e4m3 L88-129 — S|EEEE|MMM RNE, 비정규 영역 포함.
__device__ __forceinline__ unsigned char ds4m_f32_to_e4m3(float x)
{
    unsigned char sign = (__signbitf(x)) ? 0x80u : 0x00u;
    float a = fabsf(x);
    if (a < 9.765625e-4f) {
        return sign;
    }
    if (a < 0.015625f) {
        float q = a * 512.0f;
        float r = rintf(q);
        if (r >= 8.0f) {
            return sign | 0x08u;
        }
        return sign | (unsigned char)r;
    }
    unsigned b = __float_as_uint(a);
    int e = (int)((b >> 23) & 0xFFu) - 127;
    unsigned mant = b & 0x7FFFFFu;
    unsigned rem = mant & 0xFFFFFu;
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
        return sign | 0x7Fu;
    }
    return sign | (e4 << 3) | (unsigned char)mm;
}

// ops.rs e4m3_to_f32 L131-148 — 정확 확장(부호는 ±1.0 곱).
__device__ __forceinline__ float ds4m_e4m3_to_f32(unsigned char u)
{
    float sv = (u & 0x80u) ? -1.0f : 1.0f;
    int e4 = (int)((u >> 3) & 0xFu);
    unsigned m = (unsigned)(u & 7u);
    if (e4 == 15 && m == 7) {
        return __int_as_float(0x7FC00000) * sv;
    }
    if (e4 == 0) {
        return sv * (float)m * 0.001953125f;
    }
    unsigned bits = (((unsigned)(e4 - 7 + 127)) << 23) | (m << 20);
    return sv * __uint_as_float(bits);
}

// ops.rs exp_cr L52-98 직이식 — f64 fma 호너 13단 + 2^k 비트 재구성.
__device__ __forceinline__ float ds4m_exp_cr(float x)
{
    const double LN2_HI = 6.931471803691238e-1;
    const double LN2_LO = 1.9082149292705877e-10;
    const double INV_LN2 = 1.4426950408889634;
    double xd = (double)x;
    if (xd > 88.72) return __int_as_float(0x7F800000);
    if (xd < -103.97) return 0.0f;
    double kd = rint(xd * INV_LN2);
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

// ops.rs rms_scale L48-55 — 비가중 RMS 배율(f32 순차 제곱합).
__device__ __forceinline__ float ds4m_rms_scale(const float* x, int n, float eps)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
    return 1.0f / sqrtf(sum / (float)n + eps);
}

// ops.rs rms_norm_weighted L33-46 — 가중 RMS, 출력 bf16 경계(호출부).
__device__ __forceinline__ void ds4m_rms_norm_weighted(
    const float* x, const float* w, float* out, int n, float eps)
{
    float s = ds4m_rms_scale(x, n, eps);
    for (int i = 0; i < n; i++) out[i] = ds4m_bf16r(x[i] * s * w[i]);
}

// ops.rs rope_apply L335-350 — 인터리브 쌍 복소수 회전(inverse → 켤레).
__device__ __forceinline__ void ds4m_rope_apply(
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

// ops.rs fp8_sim L174-192 — 1블록(스레드 1개가 블록 순차 처리).
__device__ __forceinline__ void ds4m_fp8_sim_blk(float* blk, int n)
{
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float av = fabsf(blk[i]);
        amax = fmaxf(amax, av);
    }
    amax = fmaxf(amax, 1e-4f);
    float s = ds4m_pow2_ceil(amax * (1.0f / 448.0f));
    for (int i = 0; i < n; i++) {
        float q = blk[i] / s;
        q = (q < -448.0f) ? -448.0f : (q > 448.0f) ? 448.0f : q;
        blk[i] = ds4m_e4m3_to_f32(ds4m_f32_to_e4m3(q)) * s;
    }
}

// ── 1. FP8-sim 행 블록 — ops.rs fp8_sim L174-192 (flat (row,c0)) ──
// buf: [rows][row_stride], 처리 폭 cols(c0 블록 분할 — 마지막 블록 단축).
// main_proj 입력 활성(main_hidden 행 12288, 128블록) 전용.
extern "C" __global__ void llm170_ds4m_fp8_rows(
    float* buf, int rows, int row_stride, int cols, int block)
{
    int nblk = (cols + block - 1) / block;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * nblk) return;
    int r = idx / nblk;
    int b = idx - r * nblk;
    int lo = b * block;
    int hi = (lo + block < cols) ? lo + block : cols;
    ds4m_fp8_sim_blk(buf + (long)r * row_stride + lo, hi - lo);
}

// ── 2. gemm — ops.rs gemm_nt L58-75 (1스레드=1출력, k 오름차순 f32) ──
// y[i][j] = Σ_k x[i][kk]·w[k][j] — w 는 k-major [k][n]. mul+add 각 1회
// 반올림(-fmad=false). main_proj [12288→4096] gemv/gemm 공용.
extern "C" __global__ void llm170_ds4m_gemm(
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
extern "C" __global__ void llm170_ds4m_bf16_rows(float* buf, long n)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    buf[idx] = ds4m_bf16r(buf[idx]);
}

// ── 4. 가중 RMS 행 — ops.rs rms_norm_weighted L33-46 (1스레드=1행) ──
// main_norm(frame.rs dspark_main_x L226-229)·attn_norm/ffn_norm/final norm
// (layers.rs block_forward L52-58·L84-90, frame.rs 종결 norm) 공용.
extern "C" __global__ void llm170_ds4m_rmsw_rows(
    const float* x, const float* w, float* y, int rows, int cols, float eps)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    ds4m_rms_norm_weighted(x + (long)r * cols, w, y + (long)r * cols, cols, eps);
}

// ── 5. 스파스 어텐션 — attn.rs sparse_attn_one L531-580 (DSpark 형) ──
// frame.rs dspark_decode_attn L313-322 — idxs 행은 호출부가 조립한 전체
// 리스트([0..min(win,pos+1)]++[win..win+b], -1 패드 없이 정확 길이 stride).
// (1스레드=1(t,h), acc[512] 로컬): s[e]=dot·scale → m=max →
// denom=exp(z'−m)+Σp → acc=Σ bf16(p)·v(e-오름차순) → o=bf16(acc/denom).
extern "C" __global__ void llm170_ds4m_sparse_attn(
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
    float denom = ds4m_exp_cr(sink[h] - m);
    float acc[512];
    for (int d = 0; d < hd; d++) acc[d] = 0.0f;
    for (e = 0; e < stride; e++) {
        if (s[e] == -INFINITY) continue;
        float p = ds4m_exp_cr(s[e] - m);
        denom += p;
        float p16 = ds4m_bf16r(p);
        if (row[e] < 0) continue;
        const float* krow = kv + (long)row[e] * hd;
        for (int d = 0; d < hd; d++) acc[d] += p16 * krow[d];
    }
    float* orow = o + (long)idx * hd;
    for (int d = 0; d < hd; d++) orow[d] = ds4m_bf16r(acc[d] / denom);
}

// ── 6. 출력 비회전 — attn.rs attention_forward L646-658 (DSpark 형) ──
// frame.rs dspark_decode_attn L323-331 — o[b][h·hd..] 마지막 rd 디m
// rope^-1(켤레, 쿼리 위치 pos+1+i) + bf16. cs 는 시프트 표
// ([b][half][2] — 행 i = 절대 위치 pos+1+i).
extern "C" __global__ void llm170_ds4m_derot(
    float* o, int t, int nh, int hd, int rd, const float* cs, int half)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= t * nh) return;
    int row = idx / nh;
    float* head = o + (long)idx * hd;
    ds4m_rope_apply(head + hd - rd, cs + (long)row * half * 2, half, true);
    for (int d = hd - rd; d < hd; d++) head[d] = ds4m_bf16r(head[d]);
}

// ── 7. 헤드 스트립 gemv — frame.rs head_gemv_stripwise L41-77 (신규) ──
// (1스레드=1(토큰, 스트립 내 j열)): k블록(128)별 부분합 acc =
// Σ_i x[t][k0+i]·w[(k0+i)·128+j] (i 오름차순) 를 y[t][j] 에 k0 오름차순으로
// 가산 — 코어 스트립 gemv 계약(블록 부분합 → ycol[j] += acc). 반올림 경계
// 없음(f32 순수 — 참조 head 와 동일, frame.rs L41-44 원장). w 는 호스트가
// trellis 스트립 디양자화한 [k][128] f32(k-major, 스트립 재팩).
extern "C" __global__ void llm170_ds4m_head_strip(
    const float* x, const float* w, float* y, int t, int k, int strip_cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= t * strip_cols) return;
    int ti = idx / strip_cols;
    int j = idx - ti * strip_cols;
    const float* xr = x + (long)ti * k;
    float yv = 0.0f;
    for (int k0 = 0; k0 < k; k0 += 128) {
        float acc = 0.0f;
        for (int i = 0; i < 128; i++) {
            acc += xr[k0 + i] * w[(long)(k0 + i) * strip_cols + j];
        }
        yv += acc;
    }
    y[(long)ti * strip_cols + j] = yv;
}

// ── 8. 마르코프 바이어스 — frame.rs markov_logits_bias L373-383 (신규) ──
// (1스레드=1(토큰 t, vocab 열 j)): bias[t][j] = Σ_k w1[ids[t]][k]·
// w2[k·n+j] — k 오름차순 f32 순차 누산(코어 gemv 와 동일 결합 순서).
// w1 [vocab][rank] f32(호스트 적재), w2 k-major [rank][vocab].
extern "C" __global__ void llm170_ds4m_markov(
    const unsigned* ids, const float* w1, const float* w2, float* y,
    int t, int rank, int n)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)t * n;
    if (idx >= total) return;
    int ti = (int)(idx / n);
    int j = (int)(idx - (long)ti * n);
    const float* e = w1 + (long)ids[ti] * rank;
    float acc = 0.0f;
    for (int k = 0; k < rank; k++) {
        acc += e[k] * w2[(long)k * n + j];
    }
    y[(long)ti * n + j] = acc;
}

// ── 9. 신뢰도 스코어 — frame.rs confidence_score L386-397 (신규) ──
// (1스레드=1토큰): conf[t] = Σ_i h[t][i]·proj[i] (i=0..dim-1) +
// Σ_k m[t][k]·proj[dim+k] (k=0..rank-1) — x.chain(markov) 순서 f32 순차.
extern "C" __global__ void llm170_ds4m_conf(
    const float* h, const float* m, const float* proj, float* y,
    int t, int dim, int rank)
{
    int ti = blockIdx.x * blockDim.x + threadIdx.x;
    if (ti >= t) return;
    const float* hr = h + (long)ti * dim;
    const float* mr = m + (long)ti * rank;
    float acc = 0.0f;
    for (int i = 0; i < dim; i++) acc += hr[i] * proj[i];
    for (int k = 0; k < rank; k++) acc += mr[k] * proj[dim + k];
    y[ti] = acc;
}
