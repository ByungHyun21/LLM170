// gemma4 전용 tall GEMV 계열 — [P9 2026-10-17] gptq4.cu에서 분리.
// 분리 사유(코드젠 고정): gptq4.cu는 27B/35B 주경로(g128 mma·GEMV·그룹)
// 커널을 함께 담는데, 신규 커널 추가시 fatbin 재컴파일이 **타 커널
// 코드젠을 재추첨**한다 — 실측 2026-10-17: tall2 추가 한 건으로 27B
// pp/tg 전 축 −2~3% 회귀(동일 소스 복원 시 회귀 소멸 확인). 따라서
// gemma4 tall 계열은 본 파일에서 독립 컴파일하고 gptq4.cu는 바이트
// 동결(0fc62786 기준). 잔여 tall 레거시 심볰(w4a16_gemv_g32_bf16_tall)은
// 디스패치에서 미사용이나 코드젠 보존을 위해 gptq4.cu에 잔존 — 제거는
// gptq4 전체 재추첨·A/B 재측정과 동반할 때만.
// 빌드 계약: -fmad=false(gptq4 계열 — FMA 수축 제거, core 미러 비트동일).
#include <cuda_fp16.h>
#include "g4_common.cuh"

// gptq4.cu와 동일 정의(분리 시각 기준) — 값 계약: k ≤ 128*G4_SCMAX.
#define G4_SCMAX 256

// 스케일 로드 — BF16이면 b2f, 아니면 h2f(gptq4.cu 원본과 동일 값).
template <bool BF16>
__device__ __forceinline__ float ld_scale(const unsigned short* __restrict__ p, int i) {
    return BF16 ? b2f(p[i]) : h2f(p[i]);
}

// ── [P9 2026-10-17] tall 2행 레지스터 블로킹(R=2) — x L1 재판독 절반 ──
// 판정 배경(CMP 170HX = GA100·70SM·FP32 64/SM·L1 128B/clk): 종전 tall은
// 워프=1출력행이라 레인당 x float4 16회/jj → L1 파형 66/워프-jj vs FFMA
// 34사이클(2 warp-inst/clk) = **L1 1.94× 병목**(FFMA 상한 13.1TF의 51.5%).
// 실측 정합: pp1024 1087t/s·tall 60.4% → 5.8TF/s = FFMA 44% = L1 상한의
// 87% — FFMA 비포화·DRAM도 1%대(가중 1회 스트림) = **L1/LSU 처리율 지배**.
// 따라서 cp.async 가중 더블버퍼(가중 = L1 트래픽 2%·지연은 16워프로 이미
// 은닉)·smem x 스테이징(LDS가 LDG와 같은 128B/clk 자원 — 재사용 감소 없음)
// 모두 무의미하고, 유일 레버 = **행/워프 배가로 x 1판독을 2배 FFMA로 상각**.
// R=2면 L1:FFMA = 33:34로 평형 — 상한 ≈ FFMA 포화(이론 12.8TF).
// 비트계약: 행별 산술(레인 l·j-시퀀스·f32 acc·tree64)은 tall과 **완전 동일**
// — 워프가 행 2개를 지님(행 독립). acc 시퀀스는 각 행 자신의 fmaf 열
// (w0·x, w1·y, w2·z, w3·w 순) 그대로, 타행 연산과의 교차는 값에 무관.
// 구조: 블록 256스레드(8워프) = 16행(종전 8행) · grid = (ceil(n/16),
// ceil(t16/128)) — **t-축 청크 분할**(y블록이 토큰 128개 구간 담당: k-루프는
// 행별 전체라 (행,토큰) 산술·tree64가 그대로 — 토큰 청크는 행 독립 = 비트
// 불변. 그리드 확대로 웨이브 테일 희석, 대가 = 가중 L2 재판독 ×청크수(≤4,
// 가중 ≤5.8MB ≪ L2 30MB — DRAM 불변).
// t16 = 16의 배수(호스트 계약 — tail은 ≤16 스위치 발사, 종전과 동일).
// 실측 채택(2026-10-17, CMP 170HX GPU0, t=512·reps20, 기준 = t≤16 스위치
// 슬라이스와 **비트 0 불일치** 전 형상): q(4096×2816) 4.14→2.26ms(1.83×) ·
// k(2048×2816) 2.08→1.32(1.58×) · o(2816×4096) 11.49→4.21(2.73×) ·
// gate 2.09→1.33(1.58×) · down 2.54→1.48(1.72×) · 글로벌 q(8192) 7.78→4.51
// (1.72×) · 글로벌 o(2816×8192) 19.33→8.19(2.36×) — 슬라이딩 층 합산 2.02×.
// 레지스터 128·스필 0·smem 20KB(ptxas). 잔여 병목: k≥4096 형상 ~3.0TF/s
// (23%) — x 행 스트라이드 k·4B가 16/32KB(2^n)라 L1 셋 충돌 의심. 해법은
// x 패딩/타일 재배치 = 레이아웃 계약 변경이라 미실시(재개 조건으로 기록).
extern "C" __global__ void __launch_bounds__(256, 2) w4a16_gemv_g32_bf16_tall2(
    const unsigned* __restrict__ q, const unsigned short* __restrict__ s,
    const float* __restrict__ x, float* __restrict__ out,
    int n, int k, int t16)
{
    const int o0 = blockIdx.x * 16;   // 블록 = 16행(워프당 2행)
    const int tb0 = blockIdx.y * 128; // t-청크 시작(토큰 128 = T타일 8개)
    const int tb1 = min(tb0 + 128, t16);
    const int l = threadIdx.x & 31;
    const int w = threadIdx.x >> 5;   // 워프 0..7
    const int oa = o0 + 2 * w;
    const int ob = oa + 1;
    const bool la = oa < n;
    const bool lb = ob < n;
    const int k8 = k >> 3;
    const int kg = k >> 5;
    __shared__ double red[16][32];
    __shared__ float sc[16][G4_SCMAX];
    // 스케일 스테이징 — 16행 × kg(플랫 분할, 죽은 행은 미기록 = 값 무의미).
    for (int idx = threadIdx.x; idx < 16 * kg; idx += 256) {
        const int row = idx / kg;
        const int gg = idx - row * kg;
        if (o0 + row < n) {
            sc[row][gg] = ld_scale<true>(s + (size_t)(o0 + row) * kg, gg);
        }
    }
    __syncthreads();
    const int jn = k >> 7;
    const int k4 = k >> 2;
    // 죽은 행 포인터 = q 베이스(항상 유효 메모리) — 핫루프 무가드 계약.
    // 판독값은 쓰레기지만 그 행 출력은 la/lb 가드로 미기록(행 독립 — 무영향).
    const unsigned short* blank =
        reinterpret_cast<const unsigned short*>(q) + l;
    const unsigned short* qa = la
        ? reinterpret_cast<const unsigned short*>(q + (size_t)oa * k8) + l
        : blank;
    const unsigned short* qb = lb
        ? reinterpret_cast<const unsigned short*>(q + (size_t)ob * k8) + l
        : blank;
    const float4* xb = reinterpret_cast<const float4*>(x) + l;
    constexpr int T = 16;
    for (int t0 = tb0; t0 < tb1; t0 += T) {
        float acca[T], accb[T];
#pragma unroll
        for (int r = 0; r < T; ++r) {
            acca[r] = 0.0f;
            accb[r] = 0.0f;
        }
#pragma unroll 32
        for (int jj = 0; jj < jn; ++jj) {
            const unsigned qwa = (unsigned)qa[jj * 32];
            const unsigned qwb = (unsigned)qb[jj * 32];
            const float sca = sc[2 * w][(l >> 3) + (jj << 2)];
            const float scb = sc[2 * w + 1][(l >> 3) + (jj << 2)];
            const float a0 = fmaf((float)(qwa & 0xFu), sca, -8.0f * sca);
            const float a1 = fmaf((float)((qwa >> 4) & 0xFu), sca, -8.0f * sca);
            const float a2 = fmaf((float)((qwa >> 8) & 0xFu), sca, -8.0f * sca);
            const float a3 = fmaf((float)(qwa >> 12), sca, -8.0f * sca);
            const float b0 = fmaf((float)(qwb & 0xFu), scb, -8.0f * scb);
            const float b1 = fmaf((float)((qwb >> 4) & 0xFu), scb, -8.0f * scb);
            const float b2 = fmaf((float)((qwb >> 8) & 0xFu), scb, -8.0f * scb);
            const float b3 = fmaf((float)(qwb >> 12), scb, -8.0f * scb);
#pragma unroll
            for (int r = 0; r < T; ++r) {
                const float4 xv = xb[(size_t)(t0 + r) * k4 + jj * 32];
                acca[r] = fmaf(a0, xv.x, acca[r]);
                acca[r] = fmaf(a1, xv.y, acca[r]);
                acca[r] = fmaf(a2, xv.z, acca[r]);
                acca[r] = fmaf(a3, xv.w, acca[r]);
                accb[r] = fmaf(b0, xv.x, accb[r]);
                accb[r] = fmaf(b1, xv.y, accb[r]);
                accb[r] = fmaf(b2, xv.z, accb[r]);
                accb[r] = fmaf(b3, xv.w, accb[r]);
            }
        }
        // tree64 — 행별 계약 순서(첫 결합 l<16, 셔플 8..1) 그대로. red 버퍼만
        // 2행분이고 va/vb 셔플 교차는 행 독립이라 값에 무관.
        for (int rr = 0; rr < T; ++rr) {
            red[2 * w][l] = (double)acca[rr];
            red[2 * w + 1][l] = (double)accb[rr];
            __syncthreads();
            double va = (l < 16) ? (red[2 * w][l] + red[2 * w][l + 16]) : 0.0;
            double vb = (l < 16) ? (red[2 * w + 1][l] + red[2 * w + 1][l + 16])
                                 : 0.0;
#pragma unroll
            for (int off = 8; off >= 1; off >>= 1) {
                const double da = shfl_down_f64(va, off);
                const double db = shfl_down_f64(vb, off);
                if (l < off) {
                    va += da;
                    vb += db;
                }
            }
            if (l == 0) {
                if (la) {
                    out[(size_t)(t0 + rr) * n + oa] = (float)va;
                }
                if (lb) {
                    out[(size_t)(t0 + rr) * n + ob] = (float)vb;
                }
            }
            __syncthreads();
        }
    }
}
