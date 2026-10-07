// ── EXL3 배치 GEMM(gemm2/kseg) CUDA 포팅 (plans/124 G4, 2026-10-04) ──
// 산술·비트 조작식은 rawhip/kernels/src_exl3.hip의 exl3_gemm2/exl3_gemm2_kseg
// 및 rawvk v6 gemv의 추출식 1:1 직이식(트렐리스 비트 공식 파생 금지 —
// 원본 그대로 베낀다, plans/124 §2·2회 사고 원장).
//
// 체인 계약(§3.1): 배치 1회 = had_in(T행, f16팩 [T][k/2] u32) → gemm2
// (H도메인 [T][n] 단일 출력 또는 kseg 부분합 [T][kseg][n]) → had_out
// 정확 1회(nseg 합산 + WHT⁻¹·R·svh — 이중 적용이 결함 15호).
//
// [타일링 선택 — mma m16n8k16 텐서코어 (plans/124 §1 "CUDA의 승부처")]
// hip은 스칼라 f32 누산(2.6-2.8TF)이었고 plans/124은 wmma/mma m16n8k16로
// 19TF+를 목표로 명시한다. CUDA 버전은 두 커널 모두 mma로 통일했다:
//   - A=활성(ah16, f16 — 메모리에 이미 f16쌍팩), B=가중치(트렐리스 디코드
//     → f16). 곱 f16×f16은 f32에서 정확(가수 11+11비트 < 24)이라 hip 스칼라
//     경로와 같은 값 계급, 누산만 f32 레지스터(mma 내부 f32 덧셈) — 산술
//     계약(커널 → core 참조 → 수학, §6)을 그대로 지킨다.
//   - 트렐리스 디코드(≈11 정수연산/워드)가 CMP 170HX(GA100 INT32 파이프
//     ≈64 lane/SM/clk)에서 진짜 병목: 27B gate_proj n×k=89.1M 워드 디코드
//     단독 ≈ 980M 연산. 스칼라 apply는 워드당 2·T FLOP를 FP32 파이프에서
//     치르므로 T≥8이면 디코드를 능가한다 — mma는 apply를 텐서코어로
//     보내 디코드 그림자 아래로 감춘다(§0 최적화 타깃 근거).
//
// [CMP 170HX(sm_80, GA100 70SM/4480코어, HBM2e ~1.5TB/s) 설계 근거]
//   - 블록 128스레드(4와프)가 64-n 스트립(4 n-타일)을 소유, 와프는 16-n
//     타일 1개를 독점 디코드한다: 트렐리스 타일 1개(256워드)는 와프가
//     정확히 1회 디코드(스레드당 8워드 — 중복 0). n축 grid.x=n/64만으로
//     27B n=17408→272블록, 35B n=8192→128블록 ≥ 70SM — 결함 18호(80블록
//     기아)는 n축에서 자연 해소된다.
//   - kseg=8 k-분할은 소형-n 선형(out_proj n=2048→32블록 기아) 전용:
//     grid (n/64, 8)=256블록으로 확보. 부분합 [T][8][n] + had_out(nseg=8)
//     합산은 hip 계약 그대로(§4.18).
//   - 스테이징 창 = 8 k-타일 × 4 n-타일 u32(K≤6 → 6144B smem): gemv의
//     8타일 스테이징 케이던스(원장: L1 라인 협응 로드 단위) 계승.
//   - ldmatrix 미사용: B 프래그먼트를 스테이징에서 레지스터로 직접 디코드
//     (공유 f16 타일 왕복 불필요 — 뱅크 패딩·추가 sync 전부 제거), A 프래그
//     먼트는 ah16에서 직독(u32 4개 — 전 n-블록이 같은 활성 행을 재사용해
//     L2 상주, 27B T=32 활성 320KB ≪ GA100 L2 40MB).
//   - 점유 목표(측정치 반영): cuobjdump sm_80 — exl3_gemm2 REG:72,
//     exl3_gemm2_kseg REG:80, SHARED:6144(스테이징 창). GA100 레지스터
//     65536/SM 기준 7/6블록/SM = 28/24와프(레지스터 제한 — smem은
//     164KB/SM의 4%로 비제한). 트렐리스 디코드의 INT 파이프 2 IPC 포화에
//     스케줄러당 준비와프 2개(전체 8)면 충분 — 24-28와프는 래턴시 흡분
//     포함 3배 여유. 누산기 16×f32가 레지스터에 상주(스필 0 — STACK:0).
//   - 개발 호스트 RTX 4070(sm_89)은 정합 검증 전용 — 타이밍 판단 금지(§0).
//     fatbin은 sm_80+sm_89 이중 타겟(계약 그대로).
//
// 그리드 계약: gemm2 grid=(n/64, ceil(T/32)) — 모듈 add_linear_bytes의
// k,n 128배수 계약으로 나눗셈 잔여 없음. kseg는 T≤32(와프 m16 2타일) 및
// ktiles≥8(k≥128) 보장 하에 seg별 최소 1 k-타일.

#include <cuda_fp16.h>

__device__ __forceinline__ float exl3_mul1_decode(unsigned w) {
    unsigned x = w * 0x83DCD12Du;
    unsigned p = (x & 0x00FF00FFu) + ((x >> 8u) & 0x00FF00FFu);
    unsigned sum = (p & 0xFFFFu) + (p >> 16u);
    return (1024.0f + (float)sum) * 0.00676727294921875f - 10.3828125f;
}

__device__ __forceinline__ unsigned exl3_extract(unsigned addr, const unsigned* stg_tile) {
    // addr = i0 | (i1<<8) | (sh<<16) — vk gemv 추출식 그대로.
    unsigned sh = addr >> 16;
    if (sh == 0) return stg_tile[(addr >> 8) & 0xFF] & 0xFFFFu;
    return ((stg_tile[(addr >> 8) & 0xFF] >> sh)
          | (stg_tile[addr & 0xFF] << (32 - sh))) & 0xFFFFu;
}

// mma.sync m16n8k16 f16·f16→f32 — D = A(활성 16t×16k) × B(가중치 16k×8n).
// 프래그먼트 사상(PTX ISA m16n8k16): g=lane>>2, q=lane&3 에 대해
//   A: a0={A[g][2q],A[g][2q+1]} a1={A[g+8][2q],..} a2={A[g][2q+8],..}
//      a3={A[g+8][2q+8],..} — ah16에서 u32 4개 직독(짝수 k 정렬).
//   B: b0={B[2q][g],B[2q+1][g]} b1={B[2q+8][g],..} — n=lane>>2(와프 내
//      8n열), k=2q(+1,+8,+9). B[k][n]=W[n][k]이므로 워드는 (r=k%16,
//      c=n%16) 트렐리스 좌표로 추출한다.
//   D: c0=C[g][2q] c1=C[g][2q+1] c2=C[g+8][2q] c3=C[g+8][2q+1].
__device__ __forceinline__ void mma_16816(
    float& c0, float& c1, float& c2, float& c3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1)
{
    float d0, d1, d2, d3;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
    c0 = d0; c1 = d1; c2 = d2; c3 = d3;
}

// 공용 본체: t0행부터 T_TILE=32행(2 m16)×블록 64-n(와프 16n×2 n8)을
// k-윈도우(8 k-타일) 순회로 누산. kseg 커널은 k 범위를 grid.y seg로
// 분할해 [T][kseg][n] 부분합에, plain 커널은 전체 k를 [T][n]에 드레인.
__device__ __forceinline__ void gemm2_body(
    const unsigned* __restrict__ ah16, // f16쌍팩 [T][k/2]
    const unsigned* __restrict__ tre,  // trellis u32 [kt][nt][8K]
    float* __restrict__ s,             // plain [T][n] · kseg [T][kseg][n]
    int ktiles, int ntiles, int K, int T, int kseg, int seg)
{
    // 스테이징: 8 k-타일 × 4 n-타일(K≤6 상한 — 모듈 krate 가드와 동일 원천).
    __shared__ unsigned tre_s[8 * 4 * 48];
    const int n = ntiles * 16;
    const int words32 = 8 * K;
    const int arow_words = (ktiles * 16) >> 1; // k/2
    const int wg_n = blockIdx.x;
    const int row0 = wg_n * 64;                // n 128배수 계약 → 잔여 없음
    const int t0 = (kseg > 1) ? 0 : (int)blockIdx.y * 32;
    const int nt = min(32, T - t0);            // 이 블록의 t행 수
    const int tlim = t0 + nt;                  // 절대 행 경계(가드 기준)
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;                   // B n열(와프 내 0..7)
    const int q = lane & 3;
    // k-범위: kseg>1이면 ktiles를 kseg 분할(hip kseg와 동일 분배식).
    const int kts = (int)((long)ktiles * seg / kseg);
    const int kte = (int)((long)ktiles * (seg + 1) / kseg);

    // 트렐리스 워드 비트주소 선계산(원본 t_ 식 그대로 — 원장: 파생 금지).
    // j=와프 내 n8 타일(0/1), hi=k상반(+8), d=k열 내 홀짝. c=j*8+g, r=hi*8+2q+d.
    unsigned addr8[2][2][2];
    #pragma unroll
    for (int j = 0; j < 2; j++)
        #pragma unroll
        for (int hi = 0; hi < 2; hi++)
            #pragma unroll
            for (int d = 0; d < 2; d++) {
                int r = hi * 8 + 2 * q + d;
                int c = j * 8 + g;
                int t_ = (4 * (c & 7) + ((r & 7) >> 1)) * 8
                       + ((r >> 3) * 2 + (r & 1) + ((c >> 3) * 4));
                unsigned b0 = (unsigned)t_ * (unsigned)K + (unsigned)(K + 256 * K - 16);
                unsigned b1 = b0 + 16u;
                unsigned i0 = (b0 >> 5) % (unsigned)words32;
                unsigned i1 = ((b1 - 1u) >> 5) % (unsigned)words32;
                unsigned sh = (((b1 - 1u) >> 5) + 1u) * 32u - b1;
                addr8[j][hi][d] = i0 | (i1 << 8) | (sh << 16);
            }

    // 누산기: 와프당 2(m16) × 2(n8) × 4(f32) = 16레지스터.
    float acc[2][2][4];
    #pragma unroll
    for (int m = 0; m < 2; m++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            #pragma unroll
            for (int e = 0; e < 4; e++) acc[m][j][e] = 0.0f;

    for (int kb = kts; kb < kte; kb += 8) {
        int nk = min(8, kte - kb);
        int stg_words = nk * 4 * words32;
        // 스테이징: 연속 u32 협응 로드(와프당 4n-타일 × nk k-타일).
        for (int w = (int)threadIdx.x; w < stg_words; w += 128) {
            int tidx = w / words32;        // 0..nk*4
            int ktl = tidx >> 2;           // k-타일(창 내)
            int ntl = tidx & 3;            // 블록 내 n-타일
            tre_s[w] = tre[((long)(kb + ktl) * ntiles + (wg_n * 4 + ntl)) * words32
                           + (w % words32)];
        }
        __syncthreads();

        for (int kl = 0; kl < nk; kl++) {
            const unsigned* tile = tre_s + (kl * 4 + warp) * words32;
            // B 프래그먼트: 디코드 8워드 → f16×2팩 2개/ n8 타일.
            unsigned bufa[2][2];
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                unsigned w00 = exl3_extract(addr8[j][0][0], tile);
                unsigned w01 = exl3_extract(addr8[j][0][1], tile);
                unsigned w10 = exl3_extract(addr8[j][1][0], tile);
                unsigned w11 = exl3_extract(addr8[j][1][1], tile);
                __half2 b0 = __floats2half2_rn(exl3_mul1_decode(w00), exl3_mul1_decode(w01));
                __half2 b1 = __floats2half2_rn(exl3_mul1_decode(w10), exl3_mul1_decode(w11));
                bufa[j][0] = *reinterpret_cast<unsigned*>(&b0);
                bufa[j][1] = *reinterpret_cast<unsigned*>(&b1);
            }
            // A 프래그먼트: ah16 직독(짝 k 정렬 u32 — T 경계 제로 패딩).
            long kcol = (long)(kb + kl) * 8 + q;
            #pragma unroll
            for (int m = 0; m < 2; m++) {
                int tg = t0 + m * 16 + g;      // c0/c1 행
                int tg8 = tg + 8;              // c2/c3 행
                unsigned a0 = (tg < tlim) ? ah16[(long)tg * arow_words + kcol] : 0u;
                unsigned a1 = (tg8 < tlim) ? ah16[(long)tg8 * arow_words + kcol] : 0u;
                unsigned a2 = (tg < tlim) ? ah16[(long)tg * arow_words + kcol + 4] : 0u;
                unsigned a3 = (tg8 < tlim) ? ah16[(long)tg8 * arow_words + kcol + 4] : 0u;
                mma_16816(acc[m][0][0], acc[m][0][1], acc[m][0][2], acc[m][0][3],
                          a0, a1, a2, a3, bufa[0][0], bufa[0][1]);
                mma_16816(acc[m][1][0], acc[m][1][1], acc[m][1][2], acc[m][1][3],
                          a0, a1, a2, a3, bufa[1][0], bufa[1][1]);
            }
        }
        __syncthreads();
    }

    // 드레인: D c0..c3 → (t, n). plain은 seg=0/nseg=1과 동일 인덱스식.
    #pragma unroll
    for (int m = 0; m < 2; m++)
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int tg = t0 + m * 16 + g;
            int tg8 = tg + 8;
            int nc = row0 + warp * 16 + j * 8 + 2 * q;
            if (tg < tlim) {
                s[((long)tg * kseg + seg) * n + nc] = acc[m][j][0];
                s[((long)tg * kseg + seg) * n + nc + 1] = acc[m][j][1];
            }
            if (tg8 < tlim) {
                s[((long)tg8 * kseg + seg) * n + nc] = acc[m][j][2];
                s[((long)tg8 * kseg + seg) * n + nc + 1] = acc[m][j][3];
            }
        }
}

// 배치 GEMM 단일 출력 — grid (n/64, ceil(T/32)), 블록 128. 대형-T·lm_head.
extern "C" __global__ void exl3_gemm2(
    const unsigned* __restrict__ ah16,
    const unsigned* __restrict__ tre,
    float* __restrict__ s,             // f32 [T][n] H도메인
    int ktiles, int ntiles, int K, int T)
{
    gemm2_body(ah16, tre, s, ktiles, ntiles, K, T, /*kseg=*/1, /*seg=*/0);
}

// 배치 GEMM k-분할 — grid (n/64, kseg), 블록 128. 소형-T 점유 레버
// (결함 18호): 부분합 [T][kseg][n]을 had_out(nseg=kseg)이 합산.
// hip 게이트(T≤8)로만 진입하므로 t0=0 단일 m-타일 쌍으로 T 전체 커버.
extern "C" __global__ void exl3_gemm2_kseg(
    const unsigned* __restrict__ ah16,
    const unsigned* __restrict__ tre,
    float* __restrict__ s,             // f32 [T][kseg][n] 부분합
    int ktiles, int ntiles, int K, int T, int kseg)
{
    gemm2_body(ah16, tre, s, ktiles, ntiles, K, T, kseg, (int)blockIdx.y);
}
