// ── 어텐션 CUDA 포팅 (G6, 2026-10-04) ──
// [KVQ 채택 2026-10-10] f32/int4 KV 커널 제거 — int8(_q) 단일 경로.
// (attn_prep·attn_fwd3s·attn_fwd3s_part·attn_prep_q4·attn_fwd3s_part_q4)
// 산술은 구 rawhip 커널의 attn_prep(L607-687)·
// attn_fwd3s(L799-853)·pos_bump(L1152-1155)을 1:1 직이식한다
// (원본 그대로 베낌 — 트렐리스 비트 조작식은 이 파일에 없다).
// 원본과의 차이는 5점(1-2 아래 · 3-4는 #include 직후 트랜센던트 블록):
// 1) hip 판이 27B 폭(q헤드 24·KV헤드 4·kv 1024·KV cap 1024)으로 경직된 상수를
//    런치 인자(q_heads/kv_heads/cap)로 일반화 — 27B 값(24/4/1024)을 넣으면
//    원소 순서까지 원본과 동일. 목적은 35B-A3B 어텐션 형상 지원
//    (q헤드 16·KV헤드 2 — 실측 형상 config.json
//    text_config num_attention_heads=16·num_key_value_heads=2 실측 2026-10-04).
// 2) hip 시그니처의 pos0_ 파라미터(사실상 미사용 — 본체는 pp[0]만 판독)를
//    CUDA에서는 아예 제거한다: 결함 4호(KV 인덱스는 pp[0] "디바이스" 판독 —
//    파라미터 아님, 그래프/루프 설계에 필수)의 계약을 시그니처 수준에서
//    강제한다. 호스트 pos 사본 경로는 제거됐다(H 정리 — 2026-10-09).
// 3) 트랜센던트/수축(아래 블록 참조): 빌드 -fmad=false(FMA 수축 제거 —
//    호스트 미러와 연산 DAG 비트동일), 초월함수는 자작 f64 미러
//    (gdn_exp_d/gdn_log_d와 동일 노선), rope theta는 exp(ln(1e7)·e)
//    f64 재구성, rsqrtf(≤2ulp 근사)는 IEEE sqrt+div로 대체. 전부
//    (hip은 참조 구현일 뿐 진실 아님)에 근거한 정밀화 — 수학 동등·정밀도
//    상향, 값 변화는 ulp 수준. 어텐션은 계약 최대tightness maxdiff ≤2e-7
//    이라 이 미러 계약이 사실상 필수다.
// 4) fwd3s 도메인 강제: fwd3s는 T≤8 소형 전용("fwd3s는 T≤8
//    소형 전용"). 커널 첫 줄 t_len>8 균등 조기복귀(기록 없음 — 부분 기록
//    오염 없는 깨끗한 거부) + 모듈층 사전 Err의 이중 계약.
// 5) 원본 fwd3s의 1e30f 무한대 대체치·환원 구조는 불변(값 계약).
// 6) [S12] 위치축 청크.online 소프트맥스 이식 — hip 판 L202-276
//    (P0 재작성본)을 그대로 옮겨 공유메모리 sarr를 [1024]→[256]으로 줄이고
//    kvcap과 무관하게 했다. exp는 attn_expf 미러를 쓰는 CUDA 규약을 유지한다.
//
// 그리드 계약(결함 5호 정신 — T축은 grid.x): prep는 (T, q_heads+kv_heads),
// fwd3s는 (T, q_heads). 블록 128(prep)·256(fwd3s) — 원본과 동일.
//
// [CMP 170HX(sm_80, GA100 70SM, HBM2e ~1.5TB/s) 설계 근거]
// - prep: 그리드 (T≤8, 28[27B]/18[35B]), 블록 128, 정적 공유 1.5KB(red 512B
//   + hd 1024B) → 레지스터·공유 모두 소형, 다중 블록/SM 상주. 지배 비용은
//   qg [T][q_heads·512] 판독(27B T=8 393KB) — 완전 coalesce 스트리밍,
//   HBM2e 대역 포화 관점 무해수. rope는 32레인만 회전(partial_rotary 0.25×
//   256=64차원=쌍32) — 원본 구조 1:1(정합 우선, 구조 재설계 금지).
// - fwd3s: 그리드 (T≤8, 24[27B]/16[35B]), 블록 256, 정적 공유 2KB(qs 1KB +
//   sarr 1KB + reds 1KB — S12 청크화로 6KB에서 축소). 블록당 kc lim행×1KB
//   순차 판독 + vc lim행×256B — 메모리 본드. T=1 디코드(24블록)는 70SM 대비
//   저점유가 설계 의도다: 정확성 우선 피벗(원본 주석)이며 처리량 경로는
//   t-블록 fwd3(후속 목표). T=8 스펙 디코드(192블록)는 70SM의 ~3블록/SM로
//   확산 — 소형 전용 도메인(T≤8)이 점유 상한과 짝한다.
//   [S12] 공유메모리가 kvcap과 무관해졌으므로 블록당 점유가 위치축으로
//   늘어나지 않는다 — 옛 1024행 상한은 sarr에서 사라졌다.
// - pos_bump: 1스레드 — 캡처 그래프 내 pos 전진(h2d 불가 대체, 결함 16호
//   정신). 원본 그대로.
#include <cuda_fp16.h>
#include "cast_common.cuh" // [4.1] merge 융합 기록(f2h/h2f — cast 커널과 동일)

// ── 미러 트랜센던트(G6 — G5 gdn_exp_d와 동일 노선) ──
// 계약: 장치 초월함수(libdevice expf 2ulp·rsqrtf·powf/cosf/sinf)는 호스트
// 오라클과 비트재현 불가 — 어텐션 임계 2e-7(f32 ulp의 ~16배)에서는 ulp급
// 편차가 f16 경계·소프트맥스 분모를 타고 남으므로(계약 최tight),
// 본 파일의 트랜센던트는 순수 f64 연산 DAG(IEEE mul/add/div/floor·비트
// 재구성, FMA 수축 없음 — 빌드 -fmad=false)로 자작해 양측(본 .cu ↔
// 구 프로브 오라클 트윈)의 비트동일을 계약으로 삼는다(G5 실측
// 실측: libdevice expf는 3.1M 표본 중 30%에서 참값 ±1ulp). exp는 G5
// gdn_exp_d와 동일 DAG(차수 7 테일러·비트 재구성 2^k). 도메인: exp |x|≤128,
// sincos 0≤a≤2^20(rope ang = pos·theta ≤ cap·1 — 실사용 ≤1024),
// theta |e|≤ln(1e7). 연산 순서·상수는 절대 변경 금지(비트동일 계약 —
// Rust 트윈과 리터럴까지 동일해야 한다).
__device__ __forceinline__ double attn_exp_d(double x)
{
    // [2026-10-09 P8-attn-3 결함 수정] 도메인 가드 — k 기반 2^k 재구성은
    // k ≥ -1022(즉 x ≥ -708.5)에서만 유효. KV 분할 병합이 빈 분할에 -1e30을
    // 넣어 NaN이 됐다(S=4/8 실측: 토큰 0 반복). gdn_exp_d와 동일 의미론.
    if (x < -708.5) {
        return 0.0;
    }
    if (x > 709.78) {
        return __longlong_as_double(0x7ff0000000000000LL); // +inf
    }
    // k = floor(x·invln2 + ½)(양수 음수 공용 반올림) → r = x − k·ln2(hi/lo)
    // → 테일러 차수 7 → 2^k 비트 재구성. gdn_exp_d(G5)와 동일 DAG.
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

__device__ __forceinline__ float attn_expf(float x)
{
    // [FLA-8 동형 채택 2026-10-11] f64 폴리 → __expf(2ulp). GDN 트랜센던털
    // 채택(FLA-8)과 동일 클래스 — eval 판정: 27B chunk4 en PPL +0.0226%·
    // agree 100.000% / ko +0.0097%·99.808% (FLA-8 밴드 내). 프리필 이득:
    // pp4096 +3.1%·pp16384 +4.5% (27B) · 35B pp4096 +1.8%·np4 +0.9%.
    // 골든: 프로브/spec n=8·장문 600/4000·서브 전부 유지(smoke 10/10) —
    // 재동결 불요(짧은 스트림 안정). rope theta/sincos의 f64(attn_exp_d)는
    // 불변.
    return __expf(x);
}

// rope theta = powf(1e7, −2·tid/64)(원본 L630/664)의 f64 재구성:
// exp(ln(1e7)·e) — e = −2·tid/64는 f64에서 정확(2·tid 정수, /64 지수).
// ln(1e7) 리터럴은 Rust 트윈과 동일 문자열(비트동일 계약). tid=0 → 정확히 1.
__device__ __forceinline__ float attn_theta(int tid)
{
    double e = -(2.0 * (double)tid) / 64.0;
    double th = attn_exp_d(16.11809565095832 * e);
    return (float)th;
}

// sincos f64 트윈: Cody-Waite 2분할 환원(k≤2^20에서 k·pio2_1은 53비트 내
// 정확 — pio2_1은 하위 20비트가 0) + 테일러 z⁶ Horner(차 오차 ~3e-13,
// f32 ulp 6e-8의 5자리수 아래). 사분면 매핑 n=k&3. 도메인 0 ≤ a ≤ 2^20
// (rope ang ≥ 0 — pos 무부호·theta ≥ 0).
__device__ __forceinline__ void attn_sincos_d(double a, double* co, double* si)
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
    int n = (int)(k & 3LL);
    // [결함 21호 수정 2026-10-05] 사분면 표 n1/n3 교차 오류 → fdlibm 규약:
    // n=1 → (−st, ct) · n=3 → (st, −ct). 기존 표는 커널·오라클이 같은 오류를
    // 공유해 자기일관 게이트로 미검출(FND 교차측정으로 발견, kv 6.626e0).
    if (n == 0) { *co = ct; *si = st; }
    else if (n == 1) { *co = -st; *si = ct; }
    else if (n == 2) { *co = -ct; *si = -st; }
    else { *co = st; *si = -ct; }
}

__device__ __forceinline__ float attn_cosf(float a)
{
    double c, s;
    attn_sincos_d((double)a, &c, &s);
    return (float)c;
}

__device__ __forceinline__ float attn_sinf(float a)
{
    double c, s;
    attn_sincos_d((double)a, &c, &s);
    return (float)s;
}

// 마커 pr1c

// ── [K6-2 2026-10-12] KV 블록 테이블 간접 어드레싱(K6 트랜셰 2) ──
// 종전 주소 kv_base + ((layer*cap + row)*kv_dim + …)은 슬롯 내 연속
// [layer][cap][kv_dim] 전제 — 유휴 슬롯도 전체 cap을 점유하고 접두 공유가
// 불가능하다. 블록 테이블은 **512토큰 × 1층** 블록을 독립 주소 단위로
// 삼는다: 디바이스 테이블 [slot][layer][block] × u64 4개 =
// (K기저, V기저, ksc기저, vsc기저 — 바이트 주소). 호스트(set_attn)가
// 구축해 슬롯 오프셋을 포인터에 베이크해 전달(종전 attn_kv_ptr(slot)과
// 동일 방식 — 그래프 캡처는 테이블 '주소'만 베이크, 내용 불변).
//   entry = tab + (layer*mb + row/512)*4,  mb = ceil(cap/512)
//   행 주소 = entry[kind] + (row%512)*stride + 헤드 오프셋
// 트랜셰 2는 테이블이 기존 연속 할당 내 주소를 가리킨다(할당 불변):
// 블록 기저 = 슬롯기저 + (layer*cap + 블록*512)*stride이고
// row = (row/512)*512 + row%512이므로 전개하면 종전 주소와 동일한 값 —
// **비트동일은 구성으로 증명**(같은 메모리·같은 접근 순서). 이후 트랜셰
// (블록 분할 할당·접두 공유)는 테이블 내용만 교체하면 된다.
struct AttnKvRowPtrs {
    signed char* k;
    signed char* v;
    float* ks;
    float* vs;
};
__device__ __forceinline__ AttnKvRowPtrs attn_blk_row(
    const unsigned long long* __restrict__ tab, int layer, int cap, int row,
    long kv_dim, int kv_heads, int kh)
{
    const unsigned long long* bt =
        tab + ((long)layer * ((cap + 511) >> 9) + (row >> 9)) * 4;
    const unsigned long long ro = (unsigned long long)(row & 511);
    AttnKvRowPtrs p;
    p.k = reinterpret_cast<signed char*>(bt[0]) +
          ro * (unsigned long long)kv_dim + (long)kh * 256;
    p.v = reinterpret_cast<signed char*>(bt[1]) +
          ro * (unsigned long long)kv_dim + (long)kh * 256;
    p.ks = reinterpret_cast<float*>(
        bt[2] + (ro * (unsigned long long)kv_heads + (unsigned long long)kh) * 4ULL);
    p.vs = reinterpret_cast<float*>(
        bt[3] + (ro * (unsigned long long)kv_heads + (unsigned long long)kh) * 4ULL);
    return p;
}

// [P13] KV 양자화 prep — attn_prep와 산술 동일(q 브랜치 무변경), KV 기록만
// int8(행×헤드 absmax 스케일 = max|x|/127, 0이면 1). 스케일 f32 1개/행헤드.
// 값 차이는 양자화 오차뿐 — 판정은 토큰(단문 골든 + 장문). fwd3s_part_q가 소비.
// [K6-2] kc/vc/ksc/vsc 4포인터 → 블록 테이블 1포인터로 간접화(위 블록 참조).
// [KV 창 상한 실측 기각 2026-10-13] "어텐션 KV 판독을 마지막 W로 한정"(위치만의
// 함수라 접두 재사용과 정합)을 빌드 상수로 전 배선(FA2: c_start=w0/FA2_BC 청크
// 스킵 + 행별 cc<nA−W 마스크 · part_q/pg: win_lo=lim−W 스팬 분할)하고 27B 게이트
// 실측 — **기각**:
//   · 밴드(비선형 절벽): W=8192 mix **+1.95%@16K·+2.16%@32K**(CI 확실 — 밴드 밖)
//     vs W=16384@32K **−0.04%[−0.13,+0.01]**(밴드 내) — 손상은 8K↔16K 사이 급변.
//   · needle(AND 축이 지배): 창 안 ✓/창 밖 ✗ — **GDN 상태는 창 밖 패스코드 회수를
//     보상하지 못한다**(32K·동일 패스코드: W=16384 d30(거리 22.9K) ✗·d70/d90 ✓;
//     W=8192 d90(3.3K)만 ✓; 16K/24K도 동일 패턴).
//   · 장문 문맥(65K): **창 크기와 무관하게 '제거 자체'가 회수 붕괴** — W=32768·
//     W=16384 모두 깊이 90%(거리 6.6K, 창 안) ✗·전량 KV ✓(반복 루프 퇴화·gen40).
//   · 함의: 고정-메모리 절감은 검색(RULER/NIAH)에서 먼저 무너진다(정본 R10 #10
//     예측을 자체 모델로 확증). 재실험 접점 = 위 3곳(빌드 상수) — 1M/CMP 장문
//     정책 논의 시에만. 수치 전문은 원장(4090-detail) 동일 세션 기록.
// [저비트 KV 계보 감시 2026-10-13] int4 이하 금지 승계: 회전·코드북 계열(k8v4·
// TurboQuant·OSCAR·TriAxialKV·SAW-INT4·OffQ) + 2026 신생 lattice-VQ(KakeyaLattice
// D4/E8·NexusQuant E8·HyperQuant·LLVQ Leech·VQKV) — "2-bit·needle 보존" 주장 포함
// 전부 이 계보. 재개는 소유자 결정(CMP/1M 시점)에만. 외부 기억 격자 μLattice
// (eelbaz/mLattice — 9셀 상수 팬아웃·Qwen3.5-4B 소프트토큰 볼트온)는 KV 양자화가
// 아니라 어텐션 메모리 대체 축 — 학습 의존·폐쇄 어휘·LM 사전학습 미검증(감시만).
// [int8 장문 품질 실측 2026-10-13] needle(합성 패스코드 검색·깊이 10/50/90%) —
// 27B 16K/24K **6/6 hit** · 35B 8K/16K/24K **8/9**(미스 1 = 24K·깊이 90%).
// 단문 PPL(밴드 ±0.3%)과 별개의 AND 축 평가(eval.md). 하네스 규율: 판정은
// 토큰 id 부분열이 아니라 **디토큰화 문자열 포함** — 생성 시점 토큰화는 문맥
// 의존이라 id 검사는 정답을 놓친다(실측: 35B가 정답 패스코드를 다른 분할로
// 생성해 0/6 오판 — `w4a16-eval needle`, --gen/--picks 진단 옵션).
extern "C" __global__ void attn_prep_q(
    const float* __restrict__ qg,    // [T][q_heads*512] q‖gate 인터리브
    const float* __restrict__ kin,   // [T][kv_heads*256]
    const float* __restrict__ vin,   // [T][kv_heads*256]
    const float* __restrict__ qnw,   // [n_attn][256]
    const float* __restrict__ knw,   // [n_attn][256]
    float* __restrict__ qh,          // [T][q_heads*256]
    const unsigned long long* __restrict__ dblk_tab, // [K6-2] [layer][mb][4]
    const unsigned* __restrict__ pp, // [1] pos0 — 디바이스 판독(결함 4호)
    const unsigned long long* __restrict__ row_blk, // [행 퓨전 2026-10-14] nullable
    const unsigned long long* __restrict__ row_pp,  //   행별(테이블, pos) 포인터
    int t_len, int layer, int q_heads, int kv_heads, int cap)
{
    __shared__ float red[128];
    __shared__ float hd[256];
    int t = blockIdx.x;
    int j = blockIdx.y;
    int tid = threadIdx.x;
    // [배치 행 퓨전 2026-10-14] 퓨전이면 행별 blk/pos(그리드 x = 행) — null이면
    // 종전 단일 슬롯 계약(pos = pp[0] + t). 주소·산술 동일(행별 값).
    const unsigned long long* blk = dblk_tab;
    int pos;
    if (row_blk != (const unsigned long long*)0) {
        blk = (const unsigned long long*)row_blk[t];
        pos = (int)((const unsigned*)row_pp[t])[0];
    } else {
        pos = (int)pp[0] + t;
    }
    long kv_dim = (long)kv_heads * 256;

    if (j < q_heads) {
        int src = t * (q_heads * 512) + j * 512;
        hd[tid] = qg[src + tid];
        hd[tid + 128] = qg[src + 128 + tid];
        __syncthreads();
        float ss = hd[tid] * hd[tid] + hd[tid + 128] * hd[tid + 128];
        red[tid] = ss;
        __syncthreads();
        for (int s = 64; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        float inv = 1.0f / sqrtf(red[0] / 256.0f + 1e-6f);
        hd[tid] = hd[tid] * inv * qnw[layer * 256 + tid];
        hd[tid + 128] = hd[tid + 128] * inv * qnw[layer * 256 + 128 + tid];
        __syncthreads();
        if (tid < 32) {
            float theta = attn_theta(tid);
            float ang = (float)pos * theta;
            float c = attn_cosf(ang), s2 = attn_sinf(ang);
            float x0 = hd[tid], x1 = hd[tid + 32];
            hd[tid] = x0 * c - x1 * s2;
            hd[tid + 32] = x0 * s2 + x1 * c;
        }
        __syncthreads();
        qh[(long)t * (q_heads * 256) + (long)j * 256 + tid] = hd[tid];
        qh[(long)t * (q_heads * 256) + (long)j * 256 + 128 + tid] = hd[tid + 128];
    } else {
        int m = j - q_heads;
        int src = t * (int)kv_dim + m * 256;
        hd[tid] = kin[src + tid];
        hd[tid + 128] = kin[src + 128 + tid];
        __syncthreads();
        float ss = hd[tid] * hd[tid] + hd[tid + 128] * hd[tid + 128];
        red[tid] = ss;
        __syncthreads();
        for (int s = 64; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        float inv = 1.0f / sqrtf(red[0] / 256.0f + 1e-6f);
        hd[tid] = hd[tid] * inv * knw[layer * 256 + tid];
        hd[tid + 128] = hd[tid + 128] * inv * knw[layer * 256 + 128 + tid];
        __syncthreads();
        if (tid < 32) {
            float theta = attn_theta(tid);
            float ang = (float)pos * theta;
            float c = attn_cosf(ang), s2 = attn_sinf(ang);
            float x0 = hd[tid], x1 = hd[tid + 32];
            hd[tid] = x0 * c - x1 * s2;
            hd[tid + 32] = x0 * s2 + x1 * c;
        }
        __syncthreads();
        // [K6-2] 블록 테이블 간접화 — 종전 dst/sr 전개와 동일 주소(구성 증명).
        AttnKvRowPtrs rp = attn_blk_row(blk, layer, cap, pos, kv_dim, kv_heads, m);
        // K absmax → int8
        red[tid] = fmaxf(fabsf(hd[tid]), fabsf(hd[tid + 128]));
        __syncthreads();
        for (int s = 64; s > 0; s >>= 1) {
            if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
            __syncthreads();
        }
        float ks = red[0] * (1.0f / 127.0f);
        if (ks == 0.0f) {
            ks = 1.0f;
        }
        float ik = 1.0f / ks;
        __syncthreads(); // red[0](K absmax) 판독 완료 후 red 재사용
        int q0 = __float2int_rn(hd[tid] * ik);
        int q1 = __float2int_rn(hd[tid + 128] * ik);
        rp.k[tid] = (signed char)max(-127, min(127, q0));
        rp.k[128 + tid] = (signed char)max(-127, min(127, q1));
        if (tid == 0) {
            rp.ks[0] = ks;
        }
        // V absmax → int8
        float v0 = vin[src + tid];
        float v1 = vin[src + 128 + tid];
        red[tid] = fmaxf(fabsf(v0), fabsf(v1));
        __syncthreads();
        for (int s = 64; s > 0; s >>= 1) {
            if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
            __syncthreads();
        }
        float vs = red[0] * (1.0f / 127.0f);
        if (vs == 0.0f) {
            vs = 1.0f;
        }
        float iv = 1.0f / vs;
        int w0 = __float2int_rn(v0 * iv);
        int w1 = __float2int_rn(v1 * iv);
        rp.v[tid] = (signed char)max(-127, min(127, w0));
        rp.v[128 + tid] = (signed char)max(-127, min(127, w1));
        if (tid == 0) {
            rp.vs[0] = vs;
        }
    }
}

// ── fwd3s 본체(구 rawhip 커널 L799-853 직이식, 폭 인자화 + 도메인 강제) ──
// 3단 구조: (1) 스코어 q·kᵀ·scale(행당 1스레드 순차 d-누산) → sarr,
// (2) 트리 max → 지수(트윈)·트리 sum(스레드 보폭 순차 + 트리 — 환원 순서
// 미러 계약) → sarr 자리에 확률, (3) AV 순차 누산(스레드=dim 원소) →
// 게이트 sigmoid 곱. WG당 (t,h). lim = pp[0](디바이스)+t+1 — 결함 4호.
// 도메인: T≤512(ATTN_TMAX) — 위반 시 전 블록 조기복귀(기록 없음).
// [2026-10-09] 8→32: 블록=토큰 구조라 smem이 T와 무관(가드일 뿐) —
// 프리필 청크 확대(가중치 재사용 ↑)용.
#define ATTN_TMAX 512
// ── [S12 2026-10-07] 위치축 청크.online 소프트맥스 — 공유메모리 6KB→2KB ──
// 이전 구현은 sarr[1024]에 lim행 스코어를 전부 담아 두었다. 공유메모리 고정
// 크기라 lim(=pp[0]+t+1)이 1024를 넘으면 sarr[row]가 블록 밖을 넘어가고
// CUresult 700(illegal address)으로 죽는다 — cap을 아무리 키워도 pos≥1023
// 에서 서빙이 막혀, CUDA의 실질 ctx 상한이 1024로 고정돼 있었다
// (규칙적으로 확인해 봉인한 한계).
//
// 해법은 알고리즘 새로 쓰기가 아니라 **hip 판의 청크 온라인 소프트맥스를
// 그대로 이식**하는 것이었다(rawhip/kernels/구 rawhip 커널 L202-276가
// 이미 그 형태로 재작성돼 있다 — 청크 256행마다 running
// max/l/acc를 갱신하므로 smem이 kvcap과 무관해진다).
//
// [왜 lim≤256 구간은 비트동일인가 — 게이트 기준선이 깨지지 않는 근거]
// 청크 경계 안(lim≤256)에서는 청크가 하나뿐이라 이식 전후가 같은 연산을
// 같은 순서로 한다: 스코어(행당 1스레드 256차원 순차 내적·scale 곱은 동일),
// max 트리(동일 값에 대한 fmaxf 트리), exp+sum 트리(동일 per-스레드 값),
// AV 누산(acc=0에서 lim행 순차 — 청크 하나면 base=0이므로 행 순서 동일),
// 최종 (acc/l)·sigmoid 동일. corr은 첫 청크에서 0이고(acc=0·0), l_run은
// 0·0+sum이라 첫 청크 결과가 그대로다. 즉 **기존 ctx 1024 게이트의 프롬프트
// (41토큰)+16 = lim≤57은 비트 단위로 동일** — 기준선 재기록이 불필요하고,
// lim>256 구간(게이트가 처음 밟지 않던 영역)만 환원 순서가 달라진다.
// 그 구간은 이전엔 서빙이 불가능했으므로 회귀 판정 대상 자체가 없다.
//
// [환원 순서 변경 — 규칙 10a] 청크 경계를 넘는 구간은 max/sum 리덕션 순서가
// 달라져 값이 ulp 수준으로 움직인다. 수학적으로는 동등(온라인 소프트맥스
// 정의 그대로)하며 f16 GEMM 누산(1.6e-2)보다 작다. 실사용 판정은 기존과
// 동일하게 argmax 일치다.
#define ATTN_CHUNK 256
// [R13 2026-10-10] part/단일 커널 공통 청크 스텝 — 온라인 소프트맥스.
// kdot(i) = 스코어(스케일 적용 — 청크 로컬 i는 호출부가 row로 환산),
// vacc(i) = 이 스레드 dim의 V 값. 산술 순서는 종전 4벌의 인라인 루프와
// 동일(청크 내 순차·d 오름차순·exp 순서 동일) — 값 불변 계약.
template <typename KDot, typename VAcc>
__device__ __forceinline__ void attn_part_step(KDot kdot, VAcc vacc,
                                               float* __restrict__ sarr,
                                               float* __restrict__ reds, int nch,
                                               int tid, float& m_run, float& l_run,
                                               float& acc)
{
    float p = -1e30f;
    if (tid < nch) {
        p = kdot(tid);
    }
    sarr[tid] = p;
    __syncthreads();
    // 청크 max(트리) — reds[0] 확정(균질)
    reds[tid] = (tid < nch) ? sarr[tid] : -1e30f;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (tid < st) reds[tid] = fmaxf(reds[tid], reds[tid + st]);
        __syncthreads();
    }
    float m_new = fmaxf(m_run, reds[0]);
    float corr = (m_run <= -1e29f) ? 0.0f : attn_expf(m_run - m_new);
    __syncthreads(); // reds[0](청크 max) 판독 완료 후 e 기록 — 덮어쓰기 레이스 방지
    // exp+청크 sum — e를 sarr에 재기록(AV 재사용)
    float e = 0.0f;
    if (tid < nch) {
        e = attn_expf(sarr[tid] - m_new);
        sarr[tid] = e;
    }
    reds[tid] = e;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (tid < st) reds[tid] += reds[tid + st];
        __syncthreads();
    }
    // AV: 자기 dim에 청크 전 행 누산(행 순서·언롤 불변 — 계약)
    acc *= corr;
#pragma unroll 8
    for (int i = 0; i < nch; i++) {
        acc += sarr[i] * vacc(i);
    }
    l_run = l_run * corr + reds[0];
    m_run = m_new;
    __syncthreads(); // sarr 재사용 전 전체 완료(다음 청크 스코어 덮어쓰기 보호)
}

// [§8.2 진단·ILP 기각 2026-10-11] 스코어 dot 4분할 ILP(단일 누산 사슬 →
// 독립 4사슬) — **기각**: 커널 15618→15493ms(-0.8%, 노이즈)·pp16384 벽
// 25562ms 불변 + 35B 600 골든 플립(271→198). 원인: 이 커널은 **FMA 사슬이
// 아니라 로드/지연 바운드** — 행당 1스레드가 자기 K 행 256B를 4B×64회
// 순차 판독(스레드 간 stride 256B — 섹터는 전량 사용되나 스레드별 산개).
// 다음 후보(플랜 §8.2): cp.async KV 프리로드·warp 협조 dot(코얼레싱 —
// 합산 순서 변경 = eval 클래스). ncu 불가(권한)라 카운터 확정은 미완.
// [S7 분석 2026-10-11 — 착수 보류] K cp.async 프리페치의 구조 장애: 스텝이
// 스레드당 1행 전용(레인=행)이라 K는 청크(256행×256B=64KB) 단위로만 소비 —
// smem 한도로 서브타일 스테이징 시 스코어 병렬도(256행 동시)가 붕괴한다
// (§8.2 ILP 기각과 동일 벽). 유효 대안(워프 협조 dot)은 합산 순서 변경 =
// eval 클래스 — 별도 캠페인(통계 판정) 필요.
//
// [2026-10-09 P8-attn-3] KV 분할 attention — 4k ctx 실측: 스코어/AV가 24블록
// (q_heads)만 써 SM의 81% 유휴 + K/V 로드 지연 노출. lim>256에서 KV를
// ATTN_SPLITS(8)로 나눠 블록 = q_heads×S(27B 192블록)로 병렬화하고, 병합은
// 분할 인덱스 오름차순 고정 순서(결정적). lim≤256은 종전 단일 경로 유지 —
// 골든 구간 비트 동일 계약 보존.
// [§8.1 2026-10-11] 병합 smem 상한 — 디코드(t≤8) 분할 32까지 수용.
// 프리필 분할(8)은 호스트 ATTN_SPLITS, 디코드는 ATTN_SPLITS_D(32) 사용.
#define ATTN_SPLITS_MAX 64



// [P13] int8 KV 판독 — attn_fwd3s_part와 구조/환원 순서 동일(온라인 소프트맥스·
// 분할 결정), K/V 로드만 int8 + 스케일 곱. 스코어 (q·k_i8)·(0.0625·ks),
// AV acc += e·(v_i8·vs). 병합은 f32 part 그대로(attn_fwd3s_merge).
// [K6-2] KV 4포인터 → 블록 테이블 간접화(파일 상단 attn_blk_row 참조).
extern "C" __global__ void attn_fwd3s_part_q(
    const float* __restrict__ qh,
    const unsigned long long* __restrict__ dblk_tab, // [K6-2] [layer][mb][4]
    float* __restrict__ part,   // [T][q_heads][S][258]: m, l, acc[256]
    const unsigned* __restrict__ pp,
    const unsigned long long* __restrict__ row_blk, // [행 퓨전] nullable
    const unsigned long long* __restrict__ row_pp,
    int t_len, int layer, int q_heads, int kv_heads, int cap, int splits,
    float kq_scale)
{
    int t = blockIdx.x;
    int hs = blockIdx.y;
    int h = hs / splits;
    int s = hs - h * splits;
    int tid = threadIdx.x;
    int gq = q_heads / kv_heads;
    int kh = h / gq;
    // [gemma4 P2 2026-10-14] kq_scale = 인자(qwen 1/√d = 0.0625 동일값 — 비트 불변).
    float scale = kq_scale;
    // [배치 행 퓨전 2026-10-14] 행별 blk/pos — null이면 단일 슬롯 계약.
    const unsigned long long* blk = dblk_tab;
    int lim;
    if (row_blk != (const unsigned long long*)0) {
        blk = (const unsigned long long*)row_blk[t];
        lim = (int)((const unsigned*)row_pp[t])[0] + 1;
    } else {
        lim = (int)pp[0] + t + 1;
    }
    long kv_dim = (long)kv_heads * 256;
    long qrow = (long)t * (q_heads * 256) + (long)h * 256;
    __shared__ float qs[256];
    __shared__ float sarr[ATTN_CHUNK];
    __shared__ float reds[256];
    qs[tid] = qh[qrow + tid];
    __syncthreads();
    const int eff = (lim <= 256) ? 1 : splits;
    float* out = part + (((long)t * q_heads + h) * splits + s) * 258;
    if (s >= eff) {
        if (tid == 0) {
            out[0] = -1e30f;
            out[1] = 0.0f;
        }
        out[2 + tid] = 0.0f;
        return;
    }
    int chunk_rows = (lim + eff - 1) / eff;
    chunk_rows = ((chunk_rows + ATTN_CHUNK - 1) / ATTN_CHUNK) * ATTN_CHUNK;
    int lo = s * chunk_rows;
    int hi = min(lim, lo + chunk_rows);
    float m_run = -1e30f;
    float l_run = 0.0f;
    float acc = 0.0f;
    for (int base = lo; base < hi; base += ATTN_CHUNK) {
        const int nch = min(ATTN_CHUNK, hi - base);
        attn_part_step(
            [&](int i) -> float {
                const int row = base + i;
                // [§8.2 2026-10-11] K 벡터화 — 종전 unsigned(4B)×64 판독은 워프당
                // 32라인을 건드려 L1 웨이브프런트 32/로드(ncu: L1/TEX 87.5% 병목).
                // uint4(16B)×16으로 웨이브프런트 ÷4. 누산 순서는 d 0..255 오름차순
                // 그대로(4B 시절과 동일) — **비트동일 계약 유지**.
                // [K6-2] 행 기저 = 블록 테이블(종전 kc+… 전개와 동일 주소).
                const AttnKvRowPtrs rp =
                    attn_blk_row(blk, layer, cap, row, kv_dim, kv_heads, kh);
                const uint4* k16 = reinterpret_cast<const uint4*>(rp.k);
                const float4* q4 = reinterpret_cast<const float4*>(qs);
                float p0 = 0.0f;
#pragma unroll 4
                for (int d16 = 0; d16 < 16; d16++) {
                    const uint4 w = k16[d16];
                    const float4 qa = q4[d16 * 4 + 0];
                    const float4 qb = q4[d16 * 4 + 1];
                    const float4 qc = q4[d16 * 4 + 2];
                    const float4 qd = q4[d16 * 4 + 3];
                    p0 += qa.x * (float)(signed char)(w.x & 0xffu);
                    p0 += qa.y * (float)(signed char)((w.x >> 8) & 0xffu);
                    p0 += qa.z * (float)(signed char)((w.x >> 16) & 0xffu);
                    p0 += qa.w * (float)(signed char)(w.x >> 24);
                    p0 += qb.x * (float)(signed char)(w.y & 0xffu);
                    p0 += qb.y * (float)(signed char)((w.y >> 8) & 0xffu);
                    p0 += qb.z * (float)(signed char)((w.y >> 16) & 0xffu);
                    p0 += qb.w * (float)(signed char)(w.y >> 24);
                    p0 += qc.x * (float)(signed char)(w.z & 0xffu);
                    p0 += qc.y * (float)(signed char)((w.z >> 8) & 0xffu);
                    p0 += qc.z * (float)(signed char)((w.z >> 16) & 0xffu);
                    p0 += qc.w * (float)(signed char)(w.z >> 24);
                    p0 += qd.x * (float)(signed char)(w.w & 0xffu);
                    p0 += qd.y * (float)(signed char)((w.w >> 8) & 0xffu);
                    p0 += qd.z * (float)(signed char)((w.w >> 16) & 0xffu);
                    p0 += qd.w * (float)(signed char)(w.w >> 24);
                }
                return p0 * (scale * rp.ks[0]);
            },
            [&](int i) -> float {
                const int row = base + i;
                const AttnKvRowPtrs rp =
                    attn_blk_row(blk, layer, cap, row, kv_dim, kv_heads, kh);
                return (float)rp.v[tid] * rp.vs[0];
            },
            sarr, reds, nch, tid, m_run, l_run, acc);
    }
    if (tid == 0) {
        out[0] = m_run;
        out[1] = l_run;
    }
    out[2 + tid] = acc;
}



// 병합: 분할 s=0..S-1 오름차순 고정 — 결정적. m=최대, l·acc를 exp(m_s−m)로
// 스케일해 합산(수학적으로 단일 경로와 동일, 반올림만 상이 — lim>256 완화 등급).
// [4.1 2026-10-11] 생산자 융합 — 선택 출력 out32/out16(nullable) 동시 기록.
// out32 = h2f(f2h(ov))·out16 = f2h(ov) — cast_x32/cast_x16 커널과 동일 산식.
extern "C" __global__ void attn_fwd3s_merge(
    const float* __restrict__ part,
    const float* __restrict__ qg,
    float* __restrict__ outv,
    float* __restrict__ out32,            // [4.1] x32(roundtrip) — nullable
    unsigned short* __restrict__ out16,   // [4.1] f16 미러 — nullable
    int t_len, int q_heads, int splits)
{
    int t = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    const float* pb = part + ((long)t * q_heads + h) * splits * 258;
    // [§8.2 2026-10-11] 분할 가중 exp를 스레드 0..S-1이 1회만 계산해 smem
    // 공유 — 종전엔 256스레드 전부가 분할마다 exp(f64 미러)를 중복 계산
    // (블록당 S×256회 → S회). 값·누산 순서(오름차순) 완전 동일 — 비트동일.
    __shared__ float ms[ATTN_SPLITS_MAX];
    __shared__ float ws[ATTN_SPLITS_MAX];
    __shared__ float mred[1];
    if (tid < splits) {
        ms[tid] = pb[tid * 258];
    }
    __syncthreads();
    if (tid == 0) {
        float m0 = -1e30f;
        for (int s = 0; s < splits; s++) {
            m0 = fmaxf(m0, ms[s]);
        }
        mred[0] = m0;
    }
    __syncthreads();
    const float m = mred[0];
    if (tid < splits) {
        const float* ps = pb + tid * 258;
        ws[tid] = (ps[1] <= 0.0f) ? 0.0f : attn_expf(ps[0] - m);
    }
    __syncthreads();
    float l = 0.0f;
    float acc = 0.0f;
    for (int s = 0; s < splits; s++) {
        const float* ps = pb + s * 258;
        if (ps[1] <= 0.0f) {
            continue; // 빈 분할(0 기여 — 결정적)
        }
        l += ps[1] * ws[s];
        acc += ps[2 + tid] * ws[s];
    }
    float g = qg[(long)t * (q_heads * 512) + (long)h * 512 + 256 + tid];
    float ov = (acc / l) * (1.0f / (1.0f + attn_expf(-g)));
    const long oi = (long)t * (q_heads * 256) + (long)h * 256 + tid;
    outv[oi] = ov;
    if (out32 != (float*)0) {
        out32[oi] = h2f(f2h(ov));
    }
    if (out16 != (unsigned short*)0) {
        out16[oi] = f2h(ov);
    }
}

// 마커 f3sc

// [R3 2026-10-10] attn_pos_bump 제거 — 미사용(호스트가 attn_set_pos의
// h2d_async로 pp 갱신; 캡처 그래프에서는 replay 전 1회 갱신).


// [§8.2 2026-10-11] GQA 패킹 프리필 전용 — kv헤드 1개를 공유하는 gq(=q_heads/
// kv_heads ≤ 8)개 q헤드를 한 블록이 처리: K 행 판독을 레지스터에서 공유
// (gq개 dot이 같은 uint4 워드를 재사용)·V/vsc도 공유(vv 곱 1회). 트리는
// GQ_MAX 병렬(배리어 수 = 종전 1헤드분). 헤드별 산술 순서·트리 페어링·
// AV 행 오름차순 전부 동일 — **비트동일 계약**. 그리드 = (t, kv_heads×S).
// [디코드 단일 경로 채택 2026-10-13] t 도메인 전체(디코드 t≤8·스펙 검증 포함)로
// 확대 — 종전 part_q의 q헤드별 K/V·스케일 중복 판독(gq = 27B 6·35B 8회)이 장문
// 디코드에서 지연 바운드였다(27B 실측: 지연 기울기 ≈ 컨텍스트 토큰당 146ns —
// 64K tg 35.3 t/s ≈ 고유 KV 바이트의 ~4× DRAM 등가). 패킹 + 디코드 분할
// 64(ATTN_SPLITS_D) 채택 — 실측 곡선(27B):
//   tg@64K 35.25(part_q,S32) → 36.05(패킹,S32) → **39.75(채택 S64)** → 40.1(S128)
//   tg@16K 47.2 → 47.8 → **49.3** → 50.4 · tg@512 53.5 → 53.5 → **53.0** → 52.7
//   (35B 짧은 창 −0.5%·24K needle 2/3→3/3)
// 판정: ≤8K는 ATTN_CHUNK 정렬로 분할 수 무관 — 골든·지문 비트동일(4종 골든+장문
// 골든+지문 2종 확인) · >8K는 분할 경계 반올림류(merge는 수학 동일) — 게이트
// needle: 27B 16K/24K 6/6·32K 3/3·64K 3/3 (기준선 불변). 빈 분할 블록은 조기
// 반환 + m/l만 기록(merge가 l≤0 스킵 — 256-float 기록 불요).
#define ATTN_GQ_MAX 8
extern "C" __global__ void attn_fwd3s_part_qg(
    const float* __restrict__ qh,
    const unsigned long long* __restrict__ dblk_tab, // [K6-2] [layer][mb][4]
    float* __restrict__ part,   // [T][q_heads][S][258]
    const unsigned* __restrict__ pp,
    const unsigned long long* __restrict__ row_blk, // [행 퓨전] nullable
    const unsigned long long* __restrict__ row_pp,
    int t_len, int layer, int q_heads, int kv_heads, int cap, int splits,
    float kq_scale)
{
    int t = blockIdx.x;
    int hs = blockIdx.y;
    int kh = hs / splits;
    int s = hs - kh * splits;
    int tid = threadIdx.x;
    int gq = q_heads / kv_heads;
    int h0 = kh * gq;
    // [gemma4 P2 2026-10-14] kq_scale = 인자(qwen 1/√d = 0.0625 동일값 — 비트 불변).
    float scale = kq_scale;
    // [배치 행 퓨전 2026-10-14] 행별 blk/pos — null이면 단일 슬롯 계약
    // (lim = pp[0]+t+1 · t=행=토큰 순번). 값·주소 동일.
    const unsigned long long* blk = dblk_tab;
    int lim;
    if (row_blk != (const unsigned long long*)0) {
        blk = (const unsigned long long*)row_blk[t];
        lim = (int)((const unsigned*)row_pp[t])[0] + 1;
    } else {
        lim = (int)pp[0] + t + 1;
    }
    long kv_dim = (long)kv_heads * 256;
    __shared__ float qs[ATTN_GQ_MAX][256];
    __shared__ float sarr[ATTN_GQ_MAX][ATTN_CHUNK];
    __shared__ float reds[ATTN_GQ_MAX][256];
    __shared__ float mrun[ATTN_GQ_MAX], lrun[ATTN_GQ_MAX], corr[ATTN_GQ_MAX], mnew[ATTN_GQ_MAX];
    // [빈 분할 최소화 2026-10-13] eff 판정을 qs 스테이징 앞으로 — S=128 디코드
    // (장문 점유 회복)에서 대부분의 분할이 빈 블록인 단문 구간의 스테이징·동기화
    // 비용 제거. 빈 분할은 m/l만 기록(merge가 l≤0으로 스킵 — acc 미판독이므로
    // 256-float 기록 불요: 종전과 결과 비트동일).
    const int eff = (lim <= 256) ? 1 : splits;
    if (s >= eff) {
        if (tid == 0) {
            for (int h = 0; h < gq; ++h) {
                float* out = part + (((long)t * q_heads + (h0 + h)) * splits + s) * 258;
                out[0] = -1e30f;
                out[1] = 0.0f;
            }
        }
        return;
    }
    for (int h = 0; h < gq; ++h) {
        const long qrow = (long)t * (q_heads * 256) + (long)(h0 + h) * 256;
        qs[h][tid] = qh[qrow + tid];
    }
    __syncthreads();
    int chunk_rows = (lim + eff - 1) / eff;
    chunk_rows = ((chunk_rows + ATTN_CHUNK - 1) / ATTN_CHUNK) * ATTN_CHUNK;
    int lo = s * chunk_rows;
    int hi = min(lim, lo + chunk_rows);
    if (tid < gq) {
        mrun[tid] = -1e30f;
        lrun[tid] = 0.0f;
    }
    float acc[ATTN_GQ_MAX];
#pragma unroll
    for (int h = 0; h < ATTN_GQ_MAX; ++h) {
        acc[h] = 0.0f;
    }
    for (int base = lo; base < hi; base += ATTN_CHUNK) {
        const int nch = min(ATTN_CHUNK, hi - base);
        // [WC 전면 재작성 실측 기각 2026-10-11 — B1 축 종결] 워프 협조
        // 스코어 전면 구현(레인 l = k 8l..8l+7 uint2 코얼레싱·헤드별 부분합+
        // 5단 셔플 환원·q 재판독 12 LDS/행). **정합 완전**(eval chunk4 NLL
        // 1.772857 = 기저와 동일). 성능: pp4096 1568(−3.6%)·pp16384 1083
        // (−9%) — 명령 수 +80%(셔플 환원 60/행)가 K 코얼레싱·smem 절감을
        // 초과. 종합: fast exp 채택 외 스코어/AV 개조 4회 실측(R2×2·WC·
        // 분석) 전부 역행 — 현 구조가 국소 최적, 잔여 없음.
        // [R2 실측 기각 2026-10-11] 스코어 q-smem 재판독 절감 설계(레인=행2개
        // 공유) 2변형: ①전 스레드 p2 계산(작업 2배 잔존) = pp16K −14.5%
        // ②상반부 전담(128스레드×2행·작업 정확) = −1.1%, ncu로 **모델 반증**:
        // short_sb 불변(1.08 vs 1.09)·barrier ×3(3.13 vs 1.07)·L1 42 vs 51%
        // — q smem 트래픽은 지배 제약 아님(유휴 스레드 배리어 불균형이 손실).
        // 스코어/AV 축은 국소 최적 — 잔여는 전면 재작성(워프 협조/mma)뿐.
        // 스코어 — 행당 1스레드, K 행(uint4)을 gq개 헤드 dot이 공유.
        if (tid < nch) {
            const int row = base + tid;
            // [K6-2] 행 기저 = 블록 테이블(종전 kc+… 전개와 동일 주소).
            const AttnKvRowPtrs rp =
                attn_blk_row(blk, layer, cap, row, kv_dim, kv_heads, kh);
            const uint4* k16 = reinterpret_cast<const uint4*>(rp.k);
            float p[ATTN_GQ_MAX];
#pragma unroll
            for (int h = 0; h < ATTN_GQ_MAX; ++h) {
                p[h] = 0.0f;
            }
#pragma unroll 4
            for (int d16 = 0; d16 < 16; d16++) {
                const uint4 w = k16[d16];
#pragma unroll
                for (int h = 0; h < ATTN_GQ_MAX; ++h) {
                    if (h < gq) {
                        const float4 qa = *reinterpret_cast<const float4*>(&qs[h][d16 * 16 + 0]);
                        const float4 qb = *reinterpret_cast<const float4*>(&qs[h][d16 * 16 + 4]);
                        const float4 qc = *reinterpret_cast<const float4*>(&qs[h][d16 * 16 + 8]);
                        const float4 qd = *reinterpret_cast<const float4*>(&qs[h][d16 * 16 + 12]);
                        p[h] += qa.x * (float)(signed char)(w.x & 0xffu);
                        p[h] += qa.y * (float)(signed char)((w.x >> 8) & 0xffu);
                        p[h] += qa.z * (float)(signed char)((w.x >> 16) & 0xffu);
                        p[h] += qa.w * (float)(signed char)(w.x >> 24);
                        p[h] += qb.x * (float)(signed char)(w.y & 0xffu);
                        p[h] += qb.y * (float)(signed char)((w.y >> 8) & 0xffu);
                        p[h] += qb.z * (float)(signed char)((w.y >> 16) & 0xffu);
                        p[h] += qb.w * (float)(signed char)(w.y >> 24);
                        p[h] += qc.x * (float)(signed char)(w.z & 0xffu);
                        p[h] += qc.y * (float)(signed char)((w.z >> 8) & 0xffu);
                        p[h] += qc.z * (float)(signed char)((w.z >> 16) & 0xffu);
                        p[h] += qc.w * (float)(signed char)(w.z >> 24);
                        p[h] += qd.x * (float)(signed char)(w.w & 0xffu);
                        p[h] += qd.y * (float)(signed char)((w.w >> 8) & 0xffu);
                        p[h] += qd.z * (float)(signed char)((w.w >> 16) & 0xffu);
                        p[h] += qd.w * (float)(signed char)(w.w >> 24);
                    }
                }
            }
            const float sc = scale * rp.ks[0];
#pragma unroll
            for (int h = 0; h < ATTN_GQ_MAX; ++h) {
                sarr[h][tid] = (h < gq) ? p[h] * sc : -1e30f;
            }
        } else {
#pragma unroll
            for (int h = 0; h < ATTN_GQ_MAX; ++h) {
                sarr[h][tid] = -1e30f;
            }
        }
        __syncthreads();
        // 청크 max 트리 — GQ_MAX 병렬(페어링 종전과 동일).
#pragma unroll
        for (int h = 0; h < ATTN_GQ_MAX; ++h) {
            reds[h][tid] = sarr[h][tid];
        }
        __syncthreads();
        for (int st = 128; st > 0; st >>= 1) {
            if (tid < st) {
#pragma unroll
                for (int h = 0; h < ATTN_GQ_MAX; ++h) {
                    reds[h][tid] = fmaxf(reds[h][tid], reds[h][tid + st]);
                }
            }
            __syncthreads();
        }
        if (tid < gq) {
            mnew[tid] = fmaxf(mrun[tid], reds[tid][0]);
            corr[tid] = (mrun[tid] <= -1e29f) ? 0.0f : attn_expf(mrun[tid] - mnew[tid]);
        }
        __syncthreads();
        // exp + 청크 sum.
        float e = 0.0f;
        if (tid < nch) {
#pragma unroll
            for (int h = 0; h < ATTN_GQ_MAX; ++h) {
                if (h < gq) {
                    e = attn_expf(sarr[h][tid] - mnew[h]);
                    sarr[h][tid] = e;
                    reds[h][tid] = e;
                }
            }
        } else {
#pragma unroll
            for (int h = 0; h < ATTN_GQ_MAX; ++h) {
                reds[h][tid] = 0.0f;
            }
        }
        __syncthreads();
        for (int st = 128; st > 0; st >>= 1) {
            if (tid < st) {
#pragma unroll
                for (int h = 0; h < ATTN_GQ_MAX; ++h) {
                    reds[h][tid] += reds[h][tid + st];
                }
            }
            __syncthreads();
        }
        // AV — V/vsc 곱(vv)을 gq개 헤드가 공유(행 오름차순 유지 — 비트동일).
#pragma unroll
        for (int h = 0; h < ATTN_GQ_MAX; ++h) {
            if (h < gq) {
                acc[h] *= corr[h];
            }
        }
        // [§8.2 2026-10-11] AV sarr LDS.128 배칭 — 4행 묶음(행 오름차순 유지,
        // vv는 행별 종전 산식 그대로). ncu: L1 75.6%·SM 75.2% 동시 병목 —
        // sarr 스칼라 LDS(gq×256/청크)를 ÷4. 값 비트동일(합산 순서 불변).
        int i = 0;
#pragma unroll 4
        for (; i + 3 < nch; i += 4) {
            const int r0 = base + i;
            // [K6-2] 4행 분할 기저 = 블록 테이블(종전 vc/vsc 전개와 동일 주소).
            const AttnKvRowPtrs r0p =
                attn_blk_row(blk, layer, cap, r0, kv_dim, kv_heads, kh);
            const AttnKvRowPtrs r1p =
                attn_blk_row(blk, layer, cap, r0 + 1, kv_dim, kv_heads, kh);
            const AttnKvRowPtrs r2p =
                attn_blk_row(blk, layer, cap, r0 + 2, kv_dim, kv_heads, kh);
            const AttnKvRowPtrs r3p =
                attn_blk_row(blk, layer, cap, r0 + 3, kv_dim, kv_heads, kh);
            const float v0 = (float)r0p.v[tid] * r0p.vs[0];
            const float v1 = (float)r1p.v[tid] * r1p.vs[0];
            const float v2 = (float)r2p.v[tid] * r2p.vs[0];
            const float v3 = (float)r3p.v[tid] * r3p.vs[0];
#pragma unroll
            for (int h = 0; h < ATTN_GQ_MAX; ++h) {
                if (h < gq) {
                    const float4 s4 = *reinterpret_cast<const float4*>(&sarr[h][i]);
                    acc[h] += s4.x * v0;
                    acc[h] += s4.y * v1;
                    acc[h] += s4.z * v2;
                    acc[h] += s4.w * v3;
                }
            }
        }
        for (; i < nch; i++) {
            const int row = base + i;
            const AttnKvRowPtrs rp =
                attn_blk_row(blk, layer, cap, row, kv_dim, kv_heads, kh);
            const float vv = (float)rp.v[tid] * rp.vs[0];
#pragma unroll
            for (int h = 0; h < ATTN_GQ_MAX; ++h) {
                if (h < gq) {
                    acc[h] += sarr[h][i] * vv;
                }
            }
        }
        if (tid < gq) {
            lrun[tid] = lrun[tid] * corr[tid] + reds[tid][0];
            mrun[tid] = mnew[tid];
        }
        __syncthreads();
    }
    for (int h = 0; h < gq; ++h) {
        float* out = part + (((long)t * q_heads + (h0 + h)) * splits + s) * 258;
        if (tid == 0) {
            out[0] = mrun[h];
            out[1] = lrun[h];
        }
        out[2 + tid] = acc[h];
    }
}


// ── [FA2 2026-10-12] 프리필 어텐션 FA2식 재작성 — 블록 = (Q행 32 × q헤드) ──
// 구조: KV 청크(Bc=32) 타일 루프 + 레지스터 온라인 소프트맥스 + mma(f16).
// 워프 4: wi=warp>>1(행16 그룹)·wg=warp&1(d반 128). S·소프트맥스 = wg0 두 워프
// (행 0..15/16..31), 스테이징·PV = 전 워프(d반 분담). K는 int8 smem 보관
// (프래그먼트 로드 시 f16 변환 — |v|≤127 정확), V는 스테이징 시 f16 전치
// (VsT[d][j]). 스코어 = (q·k8)·0.0625·ksc, P = exp(S−m)·vsc (f16), l = Σexp
// (vsc 미포함 f32), O = Σ P·V. 인과 = 행별 n_max 마스크. 미기록 KV행 = 0
// 기록(0×NaN 전파 차단). lim = pp[0] 디바이스 판독(결함 4호 유지).
// 【수치 클래스】 q f16 라운딩·mma 스코어·P f16 — eval 대역 + 골든 재동결
// 절차 대상(기존 part_q/merge 경로는 디코드 t≤8 전용으로 불변).
#define FA2_BR 32
#define FA2_BC 32

__device__ __forceinline__ unsigned fa2_pk(unsigned short a, unsigned short b)
{
    return (unsigned)a | ((unsigned)b << 16);
}

__device__ __forceinline__ void fa2_mma(float* c, unsigned a0, unsigned a1,
                                        unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// int8 2개 → f16x2 (|v|≤127 정확 표현).
__device__ __forceinline__ unsigned fa2_k8(unsigned short v)
{
    const int a = (signed char)(v & 0xffu);
    const int b = (signed char)((v >> 8) & 0xffu);
    __half2 h = __floats2half2_rn((float)a, (float)b);
    return *reinterpret_cast<unsigned*>(&h);
}

extern "C" __global__ void __launch_bounds__(128, 2) attn_fa2_qg(
    const float* __restrict__ qh,
    const unsigned long long* __restrict__ dblk_tab, // [K6-2] [layer][mb][4]
    const float* __restrict__ qg,
    float* __restrict__ outv,
    float* __restrict__ out32,
    unsigned short* __restrict__ out16,
    const unsigned* __restrict__ pp,
    int t_len, int layer, int q_heads, int kv_heads, int cap, float kq_scale)
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g = lane >> 2;
    const int tt = lane & 3;
    const int wi = warp >> 1;
    const int wg = warp & 1;
    const int h = blockIdx.y;
    const int gq_ = q_heads / kv_heads;
    const int kh = h / gq_;
    const int rb = blockIdx.x * FA2_BR;
    const int pos0 = (int)pp[0];
    const long kv_dim = (long)kv_heads * 256;

    __shared__ __align__(16) unsigned short Qs[FA2_BR][264];
    __shared__ __align__(16) signed char K8[FA2_BC][272];  // 16B 정렬(uint4 스테이징)
    __shared__ __align__(16) unsigned short VsT[256][34];  // 4B 정렬(4B 프래그먼트 로드)
    __shared__ __align__(16) unsigned short Ps[FA2_BR][36];
    __shared__ float ksc_s[FA2_BC];
    __shared__ float vsc_s[FA2_BC];
    __shared__ float m_s[FA2_BR];
    __shared__ float l_s[FA2_BR];
    __shared__ float corr_s[FA2_BR];

    // [수정 2026-10-12] Q f16 오버플로 방지 — 타일 전역 스케일(소프트맥스는
    // 양수 스케일에 불변: 점수·m·l·P 모두 같은 인자로 스케일됨). 폭주 행
    // (|q|>65504, 분산 붕괴 시)이 f16 INF → inf×0 = NaN으로 전파되던 회귀의
    // 원인. 스케일 s = min(1, 30000/qmax) 후 점수에 1/s를 곱해 복원.
    float qmx = 0.0f;
    for (int e = tid; e < FA2_BR * 256; e += 128) {
        const int r = e >> 8;
        const int d_ = e & 255;
        const int row = rb + r;
        if (row < t_len) {
            qmx = fmaxf(
                qmx, fabsf(qh[(size_t)row * (q_heads * 256) + (size_t)h * 256 + d_]));
        }
    }
    __shared__ float qred[128];
    qred[tid] = qmx;
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (tid < st) {
            qred[tid] = fmaxf(qred[tid], qred[tid + st]);
        }
        __syncthreads();
    }
    const float qsc = (qred[0] > 30000.0f) ? (30000.0f / qred[0]) : 1.0f;
    const float iqsc = 1.0f / qsc;
    // 프롤로그: Q f16 적재(범위 밖 0) + 상태 초기화.
    for (int e = tid; e < FA2_BR * 256; e += 128) {
        const int r = e >> 8;
        const int d_ = e & 255;
        const int row = rb + r;
        float qv = 0.0f;
        if (row < t_len) {
            qv = qh[(size_t)row * (q_heads * 256) + (size_t)h * 256 + d_];
        }
        Qs[r][d_] = f2h(qv * qsc);
    }
    if (tid < FA2_BR) {
        m_s[tid] = -1e30f;
        l_s[tid] = 0.0f;
        corr_s[tid] = 0.0f;
    }
    __syncthreads();

    const int row_end = min(rb + FA2_BR - 1, t_len - 1);
    const int lim_max = pos0 + row_end + 1;
    const int nchunks = (lim_max + FA2_BC - 1) / FA2_BC;
    const int pos_end = pos0 + t_len;

    float o[16][4];
#pragma unroll
    for (int f = 0; f < 16; ++f) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            o[f][i] = 0.0f;
        }
    }

    for (int c0 = 0; c0 < nchunks; ++c0) {
        const int base = c0 * FA2_BC;
        // ── 스테이징: K int8 복사(또는 0) + V f16 전치 + 스케일 ──
        // [K6-2] 행 기저 = 블록 테이블(종전 kc/vc+… 전개와 동일 주소).
        for (int e = tid; e < FA2_BC * 16; e += 128) {
            const int j = e >> 4;
            const int qq = e & 15;
            const int row = base + j;
            const signed char* kp =
                attn_blk_row(dblk_tab, layer, cap, row, kv_dim, kv_heads, kh).k +
                qq * 16;
            if (row < pos_end) {
                *reinterpret_cast<uint4*>(&K8[j][qq * 16]) =
                    *reinterpret_cast<const uint4*>(kp);
            } else {
                *reinterpret_cast<uint4*>(&K8[j][qq * 16]) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        for (int e = tid; e < FA2_BC * 16; e += 128) {
            const int j = e >> 4;
            const int qq = e & 15;
            const int row = base + j;
            signed char vb[16];
            if (row < pos_end) {
                const signed char* vp =
                    attn_blk_row(dblk_tab, layer, cap, row, kv_dim, kv_heads, kh).v +
                    qq * 16;
                *reinterpret_cast<uint4*>(vb) = *reinterpret_cast<const uint4*>(vp);
            } else {
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    vb[i] = 0;
                }
            }
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                VsT[qq * 16 + i][j] = f2h((float)vb[i]);
            }
        }
        if (tid < FA2_BC) {
            const int row = base + tid;
            const bool ok = row < pos_end;
            const AttnKvRowPtrs rp =
                attn_blk_row(dblk_tab, layer, cap, row, kv_dim, kv_heads, kh);
            ksc_s[tid] = ok ? rp.ks[0] : 0.0f;
            vsc_s[tid] = ok ? rp.vs[0] : 0.0f;
        }
        __syncthreads();

        // ── S = Q·K^T (wg0 워프: 행 그룹 wi) + 온라인 소프트맥스 + P ──
        if (wg == 0) {
            float sa[4][4];
#pragma unroll
            for (int f = 0; f < 4; ++f) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sa[f][i] = 0.0f;
                }
            }
#pragma unroll
            for (int kb = 0; kb < 256; kb += 16) {
                const unsigned a0 =
                    *reinterpret_cast<const unsigned*>(&Qs[wi * 16 + g][kb + 2 * tt]);
                const unsigned a1 =
                    *reinterpret_cast<const unsigned*>(&Qs[wi * 16 + g + 8][kb + 2 * tt]);
                const unsigned a2 =
                    *reinterpret_cast<const unsigned*>(&Qs[wi * 16 + g][kb + 2 * tt + 8]);
                const unsigned a3 = *reinterpret_cast<const unsigned*>(
                    &Qs[wi * 16 + g + 8][kb + 2 * tt + 8]);
#pragma unroll
                for (int f = 0; f < 4; ++f) {
                    const int j = f * 8 + g;
                    const unsigned b0 =
                        fa2_k8(*reinterpret_cast<const unsigned short*>(&K8[j][kb + 2 * tt]));
                    const unsigned b1 = fa2_k8(
                        *reinterpret_cast<const unsigned short*>(&K8[j][kb + 2 * tt + 8]));
                    fa2_mma(sa[f], a0, a1, a2, a3, b0, b1);
                }
            }
            // 스케일(0.0625·ksc) + 인과 마스크 → 행 max/exp/합/P.
            const int rA = wi * 16 + g;
            const int rB = rA + 8;
            const int nA = pos0 + rb + rA + 1 - base; // 행 rA의 유효 열 수
            const int nB = pos0 + rb + rB + 1 - base;
#pragma unroll
            for (int f = 0; f < 4; ++f) {
                const int cc = f * 8 + 2 * tt;
                const float s0 = kq_scale * ksc_s[cc] * iqsc;
                const float s1 = kq_scale * ksc_s[cc + 1] * iqsc;
                sa[f][0] *= s0;
                sa[f][1] *= s1;
                sa[f][2] *= s0;
                sa[f][3] *= s1;
                if (cc >= nA) {
                    sa[f][0] = -1e30f;
                }
                if (cc + 1 >= nA) {
                    sa[f][1] = -1e30f;
                }
                if (cc >= nB) {
                    sa[f][2] = -1e30f;
                }
                if (cc + 1 >= nB) {
                    sa[f][3] = -1e30f;
                }
            }
            float mg = -1e30f;
            float mh = -1e30f;
#pragma unroll
            for (int f = 0; f < 4; ++f) {
                mg = fmaxf(mg, fmaxf(sa[f][0], sa[f][1]));
                mh = fmaxf(mh, fmaxf(sa[f][2], sa[f][3]));
            }
            mg = fmaxf(mg, __shfl_xor_sync(0xffffffffu, mg, 1));
            mg = fmaxf(mg, __shfl_xor_sync(0xffffffffu, mg, 2));
            mh = fmaxf(mh, __shfl_xor_sync(0xffffffffu, mh, 1));
            mh = fmaxf(mh, __shfl_xor_sync(0xffffffffu, mh, 2));
            const float moldA = m_s[rA];
            const float moldB = m_s[rB];
            const float mnewA = fmaxf(moldA, mg);
            const float mnewB = fmaxf(moldB, mh);
            const float corrA = (moldA <= -1e29f) ? 0.0f : attn_expf(moldA - mnewA);
            const float corrB = (moldB <= -1e29f) ? 0.0f : attn_expf(moldB - mnewB);
            float sumA = 0.0f;
            float sumB = 0.0f;
#pragma unroll
            for (int f = 0; f < 4; ++f) {
                const int cc = f * 8 + 2 * tt;
                const float vs0 = vsc_s[cc];
                const float vs1 = vsc_s[cc + 1];
                const float p0 = attn_expf(sa[f][0] - mnewA);
                const float p1 = attn_expf(sa[f][1] - mnewA);
                const float p2 = attn_expf(sa[f][2] - mnewB);
                const float p3 = attn_expf(sa[f][3] - mnewB);
                sumA += p0 + p1;
                sumB += p2 + p3;
                *reinterpret_cast<unsigned*>(&Ps[rA][cc]) =
                    fa2_pk(f2h(p0 * vs0), f2h(p1 * vs1));
                *reinterpret_cast<unsigned*>(&Ps[rB][cc]) =
                    fa2_pk(f2h(p2 * vs0), f2h(p3 * vs1));
            }
            sumA += __shfl_xor_sync(0xffffffffu, sumA, 1);
            sumA += __shfl_xor_sync(0xffffffffu, sumA, 2);
            sumB += __shfl_xor_sync(0xffffffffu, sumB, 1);
            sumB += __shfl_xor_sync(0xffffffffu, sumB, 2);
            // [경쟁 수정 2026-10-12] m_s[moldA]를 전 레인이 먼저 판독했다 — tt==0
            // 스토어 전 워프 동기(Volta+ 독립 스케줄링에서 무동기 read/write UB).
            __syncwarp();
            if (tt == 0) {
                m_s[rA] = mnewA;
                m_s[rB] = mnewB;
                l_s[rA] = l_s[rA] * corrA + sumA;
                l_s[rB] = l_s[rB] * corrB + sumB;
                corr_s[rA] = corrA;
                corr_s[rB] = corrB;
            }
        }
        __syncthreads();

        // ── PV: O = O·corr + P·V (전 워프, d반 분담) ──
        {
            const int rA = wi * 16 + g;
            const int rB = rA + 8;
            const float corrA = corr_s[rA];
            const float corrB = corr_s[rB];
#pragma unroll
            for (int f = 0; f < 16; ++f) {
                o[f][0] *= corrA;
                o[f][1] *= corrA;
                o[f][2] *= corrB;
                o[f][3] *= corrB;
            }
#pragma unroll
            for (int kb = 0; kb < FA2_BC; kb += 16) {
                const unsigned a0 =
                    *reinterpret_cast<const unsigned*>(&Ps[rA][kb + 2 * tt]);
                const unsigned a1 =
                    *reinterpret_cast<const unsigned*>(&Ps[rB][kb + 2 * tt]);
                const unsigned a2 =
                    *reinterpret_cast<const unsigned*>(&Ps[rA][kb + 2 * tt + 8]);
                const unsigned a3 =
                    *reinterpret_cast<const unsigned*>(&Ps[rB][kb + 2 * tt + 8]);
#pragma unroll
                for (int f = 0; f < 16; ++f) {
                    const int d_ = wg * 128 + f * 8 + g;
                    const unsigned b0 = *reinterpret_cast<const unsigned*>(
                        &VsT[d_][kb + 2 * tt]);
                    const unsigned b1 = *reinterpret_cast<const unsigned*>(
                        &VsT[d_][kb + 2 * tt + 8]);
                    fa2_mma(o[f], a0, a1, a2, a3, b0, b1);
                }
            }
        }
        __syncthreads();
    }

    // ── 에필로그: O/l × sigmoid(gate) + out32/out16 융합 ──
    {
        const int rA = wi * 16 + g;
        const int rB = rA + 8;
        const float invlA = 1.0f / l_s[rA];
        const float invlB = 1.0f / l_s[rB];
        const int rowA = rb + rA;
        const int rowB = rb + rB;
#pragma unroll
        for (int f = 0; f < 16; ++f) {
            const int d_ = wg * 128 + f * 8 + 2 * tt;
            // C 프래그먼트 4값 매핑: c0=(행g,열2t) c1=(행g,2t+1) c2=(행g+8,2t)
            // c3=(행g+8,2t+1).
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int r0 = (e < 2) ? rowA : rowB;
                const int d2 = d_ + (e & 1);
                if (r0 < t_len) {
                    const float gv = qg[(size_t)r0 * (q_heads * 512) + (size_t)h * 512 +
                                        256 + d2];
                    const float sg = 1.0f / (1.0f + attn_expf(-gv));
                    const float ov = o[f][e] * ((e < 2) ? invlA : invlB) * sg;
                    const long oi = (size_t)r0 * (q_heads * 256) + (size_t)h * 256 + d2;
                    outv[oi] = ov;
                    if (out32 != (float*)0) {
                        out32[oi] = h2f(f2h(ov));
                    }
                    if (out16 != (unsigned short*)0) {
                        out16[oi] = f2h(ov);
                    }
                }
            }
        }
    }
}
