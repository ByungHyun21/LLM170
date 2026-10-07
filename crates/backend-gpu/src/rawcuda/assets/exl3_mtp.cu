// ── EXL3 MTP 드래프트 헬퍼 커널 (plans/124 G9, 2026-10-04) ──
// MTP 체인(enorm‖hnorm → mtp.fc → 게이트 어텐션(자체 KV) → o+resid →
// FFN → resid → 공유 노름 → lm_head)의 대부분은 G2-G7 커널 재사용이다
// (mtp_cuda.rs 조립부 참조 — 어텐션은 exl3_attn_prep/fwd3s(lay=0, MTP
// 자체 KV/노름 버퍼), FFN 게이트곱은 exl3_ew, 토큰은 exl3_argmax,
// 선형은 exl3_had_in/gemv/had_out). 이 파일은 MTP 전용 잔여 2종만
// 담는다:
//   1) exl3_mtp_rms  — 잔차 가산 없는 평 RMS 노름(norm_resid의 ab=0
//      등가지만 출력 포인터가 자유 — enorm/hnorm을 cat[2n] 반쪽에
//      직접 기록, attn_norm/post_norm/공유 head norm 재사용).
//   2) exl3_mtp_axpy — 잔차 스트림 제자리 가산 x += y(o_proj·FFN down
//      출력 가산 — hip axpy 미러).
//
// 산술 계약(plans/124 §3.2·§3.4):
// - rms: inv = 1/√(Σx²/n + 1e-6)(정밀 sqrtf — rsqrt 근사 금지, 노름
//   계약), out = x·inv·w. 적산 순서는 exl3_norm_resid(assets/exl3_norm.cu)
//   1:1 — 스레드별 f32 순차(nper=⌈n/1024⌉ 스트라이드) → red[1024]
//   트리(st=512..1) → red[0]. 노름 규약 §3.4: constant_bias=1.0 노름은
//   저장소 w−1 → 등록값 +1(호출자 계약 — 커널은 원값 w를 곱한다).
// - axpy: 원소별 f32 가산(순서 무관 — 원소 독립).
//
// 빌드: 기본 fmad(norm 계열 — exl3_norm.cu와 동일 등급. 트랜센던트
// 미포함이라 비트동일 트윈 계약 대상 아님; ss += v*v의 FMA 수축
// 가능성은 오라클과 ≤1ulp/연산 차이로 종단 임계 2e-4(§1)에 무해).
//
// [CMP 170HX(sm_80, GA100 70SM, HBM2e ~1.5TB/s) 설계 근거 — plans/124 §0]
// - rms: 블록 1024(32워프) × 그리드 1(T=1 드래프트 행 — MTP 체인은
//   순차 스텝만, fwd3s T≤8 도메인과 동일 정신). smem red[1024]=4KB,
//   레지스터 v[8] — exl3_norm_resid와 동일 자원 계급으로 2블록/SM
//   상주·64워프 풀 점유. 지배 비용은 x·w 판독+쓰기 3·n·4B(5120 기준
//   60KB) — 완전 coalesce 스트라이드 적재로 HBM2e 스트리밍. 드래프트
//   스텝당 5회 호출(enorm·hnorm·attn·post·shared)은 체인 내 다른
//   GEMV(n≥10240) 대비 미시적 — 점유 확산 무의미, 레이턴시 도미넌트.
// - axpy: 블록 256 × 그리드 ⌈n/256⌉(5120 → 20블록). 원소 독립 f32
//   가산 — 완전 coalesce 2판독 1기록 스트리밍, 순수 대역폭 본드.
//   개발기(RTX 4070 SUPER, sm_89) 타이밍은 판단 근거가 아니다
//   (plans/124 §0 — 정합 검증 호스트일 뿐).

// 평 RMS 노름 1행 — exl3_norm_resid(ab=0)와 동일 적산 순서.
// 도메인: n은 1024의 배수·8192 이하(v[8] 레지스터 적재 상한 —
// exl3_norm.cu hidden 계약 계승). 27B hidden=5120(nper=5).
extern "C" __global__ void exl3_mtp_rms(
    const float* __restrict__ x,  // [n] 입력 행
    const float* __restrict__ w,  // [n] 노름 가중(행 포인터 — 등록값)
    float* __restrict__ out,      // [n] xn = rms_norm(x)·w
    int n)
{
    __shared__ float red[1024];
    int tid = threadIdx.x;
    int nper = (n + 1023) >> 10;
    float v[8];
    float ss = 0.0f;
    for (int j = 0; j < nper; j++) {
        int e = (j << 10) + tid;
        v[j] = x[e];
        ss += v[j] * v[j];
    }
    red[tid] = ss;
    __syncthreads();
    for (int st = 512; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    // 정밀 sqrt 계약(§3.2): rsq 근사는 drift 씨앗.
    float inv = 1.0f / sqrtf(red[0] / (float)n + 1e-6f);
    for (int j = 0; j < nper; j++) {
        int e = (j << 10) + tid;
        out[e] = v[j] * inv * w[e];
    }
}

// 잔차 스트림 제자리 가산 x += y — 원소별 f32(순서 무관).
// 그리드 ⌈n/256⌉ × 블록 256(완전 coalesce — GA100 대역폠 포화 관점
// 무조건 충족, 스트리밍 본드).
extern "C" __global__ void exl3_mtp_axpy(
    float* __restrict__ x,        // [n] r/w — 잔차 스트림
    const float* __restrict__ y,  // [n] 가산분(o_proj·FFN down 출력)
    int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) x[j] += y[j];
}
// 마커 mtpc
