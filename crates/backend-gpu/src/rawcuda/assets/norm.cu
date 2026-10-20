#include <cuda_fp16.h>
#include "cast_common.cuh"

// ── norm_resid CUDA 포팅 (G3, 2026-10-04) ──
// 산술은 구 rawhip 커널의 norm_resid 1:1 직이식(원본
// 그대로 베낌 — 부동소수 적산 순서·정밀 sqrt 계약 포함). 차이는 hip 판이
// 5120으로 경직된 행 폭을 hidden 인자로 일반화한 것뿐(hidden=5120이면
// 원본과 원소 순서까지 동일 — v0..v4 스트라이드 적재가 루프 좌결합으로
// 재현된다).
//
// 계약:
// - xn = rms_norm(x+ab)·nw[w행] — eps=1e-6, 평균은 hidden으로 나눈다.
// - w는 nw 배열의 행 포인터: 오프셋 w·hidden(27B는 w·5120 — 오프셋
//   누락이 결함 2호: 전 노름이 L0 행 판독).
// - 잔차 스트림은 제자리 가산 — 커널이 x = x + ab를 기록한다.
// - 인자 순서 (x, nw, ab, xn) 고정 — 교차 시 잔차에 노름가중치가
//   더해진다(결함 7호).
// - inv는 정밀 sqrtf(vk 원본 주석): rsqrt 근사는 drift 씨앗.
//
// [CMP 170HX(sm_80, GA100, HBM2e ~1.5TB/s) 설계 근거]
// 노름은 순수 메모리 본드(행당 읽기 x·ab·w + 쓰기 x·xn = 5·hidden·4B,
// 5120 기준 100KB). 런치 기하(블록 1024=32워프, 그리드 (t_len,1), 스레드당
// hidden/1024원소 스트라이드 적재 — 5120이면 ILP-5)는 hip 검증 배치 그대로:
// GA100에서 smem red[1024]=4KB·레지스터 상한(실측 cuobjdump)으로
// 2블록/SM 상주 = 2048스레드 = 64워프 풀 점유가 성립하고, ILP-5 연속
// f32 로드로 HBM2e 스트리밍 병렬성을 확보한다. float4 벡터화는 산술
// 미러(§3.2 1:1)를 깨지 않는 선에서 후속 실측 후 판단 — 개발 호스트
// 4070(sm_89) 타이밍은 판단 근거가 아니다(정합 검증 전용).
//
// hidden 계약: 1024의 배수, 8192 이하(v[8] 레지스터 적재 상한).
// 27B hidden=5120(nper=5) · 35B hidden=2048(nper=2). 256원소 q/k_norm
// 행(결함 9호 패딩 쟁점)은 본 커널 범위 밖(G4+ MTP 단계).
extern "C" __global__ void norm_resid(
    float* __restrict__ x,        // [T][hidden] r/w — 잔차 스트림(제자리 가산)
    const float* __restrict__ nw, // [rows][hidden] 노름 가중 배열
    const float* __restrict__ ab, // [T][hidden]
    float* __restrict__ xn,       // [T][hidden]
    float* __restrict__ xn32,     // [T][hidden] h2f(f2h(xn)) — 융합 캐스트.
                                  // 0이면 생략(cast_x32 커널과 비트 동일 계약:
                                  // 노드 −2/층 → 그래프 노드·런치 절감).
    unsigned short* __restrict__ xn16, // [T][hidden] f16 미러 — mma GEMM A용
                                  // (split 경로 A는 이미 f16 값 — 비트 동일).
                                  // 0이면 생략.
    int t_len, int w_off, int hidden)
{
    __shared__ float red[1024];
    int t = blockIdx.x;
    int tid = threadIdx.x;
    if (t >= t_len) return;
    int base = t * hidden;
    int nper = (hidden + 1023) >> 10;
    float v[8];
    float ss = 0.0f;
    for (int j = 0; j < nper; j++) {
        int e = base + (j << 10) + tid;
        v[j] = x[e] + ab[e];
        ss += v[j] * v[j];
    }
    red[tid] = ss;
    __syncthreads();
    for (int st = 512; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    // 정밀 sqrt 계약(vk 원본 주석): rsq 근사는 drift 씨앗.
    float inv = 1.0f / sqrtf(red[0] / (float)hidden + 1e-6f);
    for (int j = 0; j < nper; j++) {
        int e = base + (j << 10) + tid;
        float y = v[j] * inv * nw[w_off + (j << 10) + tid];
        xn[e] = y;
        // [⑯ 2026-10-18] 소프트 f2h/h2f(서브노멀 정규화 루프·inf/nan 분기 —
        // SASS 원소당 ~35명령·BSSY/BSYNC 7쌍) → HW cvt 2명령. 등가 근거:
        // cvt.rn.f16.f32는 유한·±inf 입력에서 소프트 f2h와 비트동일(서브노멀·
        // 타이 포함 — 1.38억 비트패턴 스윕, 불일치=NaN만), f16→f32는 무손실
        // (전 65536 패턴, 불일치=NaN만). NaN 페이로드만 상이 — 실도메인(유한
        // 활성) 미발생, 골든/해시 게이트가 검증.
        if (xn32 != (float*)0) {
            const __half h = __float2half_rn(y);
            xn32[e] = __half2float(h); // cast_x32 커널과 동일 산식(비트 동일)
            if (xn16 != (unsigned short*)0) {
                xn16[e] = __half_as_ushort(h); // xn32와 동일 비트(f16) — A 재판독 대역 절반.
            }
        } else if (xn16 != (unsigned short*)0) {
            xn16[e] = __half_as_ushort(__float2half_rn(y));
        }
        x[e] = v[j];
    }
}

// ── rms_rows — 잔차 없는 순수 행별 RMS (gemma4 디바이스 상주 청크, 2026-10-14) ──
// gemma4 청크의 호스트 노름(ops::rms_norm)을 디바이스로 옮기기 위한 커널.
// norm_resid 계열과 달리 ① 잔차 가산이 없고 ② **누적 순서가 CPU 계약**
// (core/src/ops.rs sq_sum)이다 — 32세그먼트 f32 순차 누산 → f64 순차 결합.
// norm_resid의 1024스레드 트리 누적은 다른 산술이라 호스트 경로 대체가 못 된다
// (청크는 호스트 산술 계약 — 토큰 골든·np 해시가 이 비트에 걸려 있다).
// 스케일: 1/sqrt(f64(sum/hidden + eps)) → f32 → f32 역수(정밀 sqrtf 계약).
// [빌드 계약] norm.cu는 기본 fmad — __fmul_rn/__fadd_rn으로 수축을 막아
// CPU의 개별 반올림(mul·add 각 1회)과 비트동일하게 만든다. nw 곱도 좌결합
// 개별 반올림(v*scale, 그 결과*nw) — Rust의 `v * scale * g`와 동일.
// smem = [hidden]x + [32]세그먼트 + [1]스케일(동적, (hidden+33)*4B).
// 블록 256(coalesce 적재·기록) · 그리드 = 행 수(t_len).
extern "C" __global__ void rms_rows(
    const float* __restrict__ x,  // [T][hidden]
    const float* __restrict__ nw, // [rows][hidden]
    float* __restrict__ xn,       // [T][hidden]
    int t_len, int w_off, int hidden, float eps)
{
    extern __shared__ float sm[];
    float* sx = sm;             // 행 스테이징(공유)
    float* sp = sm + hidden;    // [SEG] 세그먼트 부분합 + [SEG] 스케일
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;
    if (t >= t_len) return;
    const int base = t * hidden;
    for (int e = tid; e < hidden; e += nth) {
        sx[e] = x[base + e];
    }
    __syncthreads();
    // ① 32세그먼트 f32 순차 누산 (ops.rs sq_sum 미러).
    const int SEG = 32;
    const int chunk = (hidden + SEG - 1) / SEG;
    if (tid < SEG) {
        int lo = tid * chunk;
        int hi = lo + chunk < hidden ? lo + chunk : hidden;
        float part = 0.0f;
        for (int i = lo; i < hi; i++) {
            float v = sx[i];
            part = __fadd_rn(part, __fmul_rn(v, v));
        }
        sp[tid] = part;
    }
    __syncthreads();
    // ② f64 순차 결합(세그먼트 순서 그대로) → 스케일.
    if (tid == 0) {
        double dsum = 0.0;
        for (int u = 0; u < SEG; u++) {
            if (u * chunk >= hidden) break;
            dsum += (double)sp[u];
        }
        double mean = dsum / (double)hidden + (double)eps;
        sp[SEG] = 1.0f / (float)sqrt(mean);
    }
    __syncthreads();
    const float scale = sp[SEG];
    for (int e = tid; e < hidden; e += nth) {
        xn[base + e] = __fmul_rn(__fmul_rn(sx[e], scale), nw[w_off + e]);
    }
}

// ── [g4 디코드 융합 2026-10-20] rms_rows 누산 축 공용 헬퍼 ──
// 세그먼트 부분합(①)·f64 결합(②)·스케일 산출까지 rms_rows 문장 그대로 —
// 신규 융합 커널(rms_rows2·rms_add·rms_add_scale)이 같은 산술 축을 공유해
// 기존 2런치 경로와 비트동일을 보장한다. sx 스테이징 이후에 호출.
__device__ __forceinline__ void rms_rows_reduce_smem(
    const float* __restrict__ sx,  // smem [hidden] 스테이징 완료 상태
    float* __restrict__ sp,        // smem [33] (32세그먼트 + 스케일)
    int hidden, float eps)
{
    const int tid = threadIdx.x;
    // ① 32세그먼트 f32 순차 누산 (ops.rs sq_sum 미러).
    const int SEG = 32;
    const int chunk = (hidden + SEG - 1) / SEG;
    if (tid < SEG) {
        int lo = tid * chunk;
        int hi = lo + chunk < hidden ? lo + chunk : hidden;
        float part = 0.0f;
        for (int i = lo; i < hi; i++) {
            float v = sx[i];
            part = __fadd_rn(part, __fmul_rn(v, v));
        }
        sp[tid] = part;
    }
    __syncthreads();
    // ② f64 순차 결합(세그먼트 순서 그대로) → 스케일.
    if (tid == 0) {
        double dsum = 0.0;
        for (int u = 0; u < SEG; u++) {
            if (u * chunk >= hidden) break;
            dsum += (double)sp[u];
        }
        double mean = dsum / (double)hidden + (double)eps;
        sp[SEG] = 1.0f / (float)sqrt(mean);
    }
    __syncthreads();
}

// [g4 디코드 융합 2026-10-20] 같은 입력 x에 가중 2행을 곱해 출력 2개를
// 1런치로 — gemma4 층당 라우터 입력(hn)·전문가 입력(nm) 노름이 **같은 x,
// 같은 누산**을 쓰는 구조(스케일 공유)를 이용한다. 두 출력 모두 rms_rows
// 출력과 비트동일(스케일 1회 산출 · 원소별 (x·scale)·nw 개별 반올림).
extern "C" __global__ void rms_rows2(
    const float* __restrict__ x,   // [T][hidden]
    const float* __restrict__ nw,  // [rows][hidden]
    float* __restrict__ xn,        // [T][hidden] — w_off 가중
    float* __restrict__ xn2,       // [T][hidden] — w_off2 가중
    int t_len, int w_off, int w_off2, int hidden, float eps)
{
    extern __shared__ float sm[];
    float* sx = sm;
    float* sp = sm + hidden;
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;
    if (t >= t_len) return;
    const int base = t * hidden;
    for (int e = tid; e < hidden; e += nth) {
        sx[e] = x[base + e];
    }
    __syncthreads();
    rms_rows_reduce_smem(sx, sp, hidden, eps);
    const float scale = sp[32];
    for (int e = tid; e < hidden; e += nth) {
        float v = __fmul_rn(sx[e], scale);
        xn[base + e] = __fmul_rn(v, nw[w_off + e]);
        xn2[base + e] = __fmul_rn(v, nw[w_off2 + e]);
    }
}

// [g4 디코드 융합 2026-10-20] 노름 + 잔차 가산 1런치 — gemma4 샌드위치
// 노름(post_attn·post_ff)의 소비자가 axpy(w=1)뿐인 점을 융합:
// y[i] += (x[i]·scale)·nw[i]. 산식 = rms_rows 기록값 + w4a16_axpy(w=1.0,
// mul·add 분리)와 원소별로 동일(1.0·v = v 무반올림 → 가산 1회 반올림).
// 노름 출력 자체(hh/h2)는 소비자가 없어 미기록.
extern "C" __global__ void rms_add(
    const float* __restrict__ x,   // [T][hidden] 노름 대상
    const float* __restrict__ nw,  // [rows][hidden]
    float* __restrict__ y,         // [T][hidden] 잔차(r/w — 제자리 가산)
    int t_len, int w_off, int hidden, float eps)
{
    extern __shared__ float sm[];
    float* sx = sm;
    float* sp = sm + hidden;
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;
    if (t >= t_len) return;
    const int base = t * hidden;
    for (int e = tid; e < hidden; e += nth) {
        sx[e] = x[base + e];
    }
    __syncthreads();
    rms_rows_reduce_smem(sx, sp, hidden, eps);
    const float scale = sp[32];
    for (int e = tid; e < hidden; e += nth) {
        float v = __fmul_rn(sx[e], scale);
        y[base + e] = __fadd_rn(y[base + e], __fmul_rn(v, nw[w_off + e]));
    }
}

// [g4 디코드 융합 2026-10-20] 노름 + 잔차 가산 + 층 스케일 1런치 — 마지막
// 잔차 병합(hf→x)이 axpy(w=1) 후 w4a16_scale(y·ls)로 이어지는 3런치를
// 대체: y[i] = (y[i] + (x[i]·scale)·nw[i])·s. 반올림 순서 = rms_rows 값
// 산출(2회 mul) → fadd 1회 → fmul 1회 — 기존 3런치 경로와 비트동일.
extern "C" __global__ void rms_add_scale(
    const float* __restrict__ x,   // [T][hidden] 노름 대상
    const float* __restrict__ nw,  // [rows][hidden]
    float* __restrict__ y,         // [T][hidden] 잔차(r/w)
    float s,                       // gemma4 layer_scalar
    int t_len, int w_off, int hidden, float eps)
{
    extern __shared__ float sm[];
    float* sx = sm;
    float* sp = sm + hidden;
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;
    if (t >= t_len) return;
    const int base = t * hidden;
    for (int e = tid; e < hidden; e += nth) {
        sx[e] = x[base + e];
    }
    __syncthreads();
    rms_rows_reduce_smem(sx, sp, hidden, eps);
    const float scale = sp[32];
    for (int e = tid; e < hidden; e += nth) {
        float v = __fmul_rn(sx[e], scale);
        y[base + e] = __fmul_rn(__fadd_rn(y[base + e], __fmul_rn(v, nw[w_off + e])), s);
    }
}

// ── norm_resid_g — 임의 hidden(≤8192) 가드 변형 (gemma4 2816, 2026-10-14) ──
// norm_resid는 hidden이 블록폭(1024)의 배수임을 가정한다(nper 스트라이드가
// 행 경계를 넘어 다음 행을 오독). gemma4 hidden=2816(2.75×1024)에는 행
// 경계 마스크가 필요하다 — 마지막 스트라이드를 lim에서 자른다.
// 1024 배수 입력에서는 가드가 발동하지 않아 norm_resid와 동일 산술(호스트가
// 배수면 원 커널을 선택 — 기존 경로 코드젠 불변).
// 산술 순서·정밀 sqrtf 계약·융합 캐스트(xn32/xn16)는 norm_resid와 동일.
extern "C" __global__ void norm_resid_g(
    float* __restrict__ x,
    const float* __restrict__ nw,
    const float* __restrict__ ab,
    float* __restrict__ xn,
    float* __restrict__ xn32,
    unsigned short* __restrict__ xn16,
    int t_len, int w_off, int hidden)
{
    __shared__ float red[1024];
    int t = blockIdx.x;
    int tid = threadIdx.x;
    if (t >= t_len) return;
    int base = t * hidden;
    int lim = base + hidden;
    int nper = (hidden + 1023) >> 10;
    if (nper > 8) return; // 계약: hidden ≤ 8192(v[8] 레지스터 상한)
    float v[8];
    float ss = 0.0f;
    for (int j = 0; j < nper; j++) {
        int e = base + (j << 10) + tid;
        if (e < lim) {
            v[j] = x[e] + ab[e];
            ss += v[j] * v[j];
        } else {
            v[j] = 0.0f;
        }
    }
    red[tid] = ss;
    __syncthreads();
    for (int st = 512; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    float inv = 1.0f / sqrtf(red[0] / (float)hidden + 1e-6f);
    for (int j = 0; j < nper; j++) {
        int e = base + (j << 10) + tid;
        if (e >= lim) continue;
        float y = v[j] * inv * nw[w_off + (j << 10) + tid];
        xn[e] = y;
        // [⑯ 2026-10-18] 소프트 f2h/h2f → HW cvt(근거는 norm_resid 상단 주석).
        if (xn32 != (float*)0) {
            const __half h = __float2half_rn(y);
            xn32[e] = __half2float(h);
            if (xn16 != (unsigned short*)0) {
                xn16[e] = __half_as_ushort(h);
            }
        } else if (xn16 != (unsigned short*)0) {
            xn16[e] = __half_as_ushort(__float2half_rn(y));
        }
        x[e] = v[j];
    }
}
