extern "C" __global__ void exl3_attn_prep(
    const float* __restrict__ qg,    // [T][12288] q‖gate 인터리브
    const float* __restrict__ kin,   // [T][1024]
    const float* __restrict__ vin,   // [T][1024]
    const float* __restrict__ qnw,   // [L][256]
    const float* __restrict__ knw,   // [L][256]
    float* __restrict__ qh,          // [T][6144]
    float* __restrict__ kc,          // [kvcap][1024]
    float* __restrict__ vc,          // [kvcap][1024]
    const unsigned* __restrict__ pp, // [1] pos0
    int t_len, int pos0_, int layer, int kvcap)
{
    __shared__ float red[128];
    __shared__ float hd[256];
    int t = blockIdx.x;
    int j = blockIdx.y;
    int tid = threadIdx.x;
    int pos = pp[0] + t;

    if (j < 24) {
        int src = t * 12288 + j * 512;
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
        float inv = rsqrtf(red[0] / 256.0f + 1e-6f);
        hd[tid] = hd[tid] * inv * qnw[layer * 256 + tid];
        hd[tid + 128] = hd[tid + 128] * inv * qnw[layer * 256 + 128 + tid];
        __syncthreads();
        if (tid < 32) {
            float theta = powf(1e7f, -(2.0f * (float)tid) / 64.0f);
            float ang = (float)pos * theta;
            float c = cosf(ang), s2 = sinf(ang);
            float x0 = hd[tid], x1 = hd[tid + 32];
            hd[tid] = x0 * c - x1 * s2;
            hd[tid + 32] = x0 * s2 + x1 * c;
        }
        __syncthreads();
        qh[t * 6144 + j * 256 + tid] = hd[tid];
        qh[t * 6144 + j * 256 + 128 + tid] = hd[tid + 128];
    } else {
        int m = j - 24;
        int src = t * 1024 + m * 256;
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
        float inv = rsqrtf(red[0] / 256.0f + 1e-6f);
        hd[tid] = hd[tid] * inv * knw[layer * 256 + tid];
        hd[tid + 128] = hd[tid + 128] * inv * knw[layer * 256 + 128 + tid];
        __syncthreads();
        if (tid < 32) {
            float theta = powf(1e7f, -(2.0f * (float)tid) / 64.0f);
            float ang = (float)pos * theta;
            float c = cosf(ang), s2 = sinf(ang);
            float x0 = hd[tid], x1 = hd[tid + 32];
            hd[tid] = x0 * c - x1 * s2;
            hd[tid + 32] = x0 * s2 + x1 * c;
        }
        __syncthreads();
        long dst = ((long)layer * kvcap + pos) * 1024 + m * 256;
        kc[dst + tid] = hd[tid];
        kc[dst + 128 + tid] = hd[tid + 128];
        vc[dst + tid] = vin[src + tid];
        vc[dst + 128 + tid] = vin[src + 128 + tid];
    }
}
// 마커 pr1

// exl3_attn_fwd3 hip 직이식 — warp-행 스코어 + t-블록 공유 인과 어텐션.
// WG=256(wave64 4개), __shfl_down(…,64)가 subgroupShuffleDown.
// **현재 미런치(런치 지점 없음, 2026-10-04)** — P1 t-블록 배치화(plans/128)에서
// 사용 예정. EXL3_SCAP(스코어 smem 창) 1024: pos+t>SCAP인 시퀀스엔 부정홝
// (sarr OOB 아님 — 창 초과 행 미참조가 아니라 오버런)하므로 P1 채택 시
// fwd3s와 동일한 청크 온라인 소프트맥스로 재작성 필요. kvcap 인자는
// KV 스트라이드만 파라미터화한 것(할당 상한과 동일 값 전달).
#define EXL3_TB 4
#define EXL3_SCAP 1024
extern "C" __global__ void exl3_attn_fwd3(
    const float* __restrict__ qh,    // [T][6144]
    const float* __restrict__ kc,    // [kvcap][1024]
    const float* __restrict__ vc,    // [kvcap][1024]
    const float* __restrict__ qg,    // [T][12288] gate
    float* __restrict__ outv,        // [T][6144]
    const unsigned* __restrict__ pp, // [1] pos0
    int t_len, int pos0_, int layer, int kvcap)
{
    __shared__ float qs4[EXL3_TB * 256];
    __shared__ float sarr[EXL3_TB * EXL3_SCAP];
    __shared__ float reds[256];
    __shared__ float gw[EXL3_TB][2];
    int tb = blockIdx.x * EXL3_TB;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    int sg = tid >> 6;
    int si = tid & 63;
    int kh = h / 6;
    float scale = 0.0625f;

    int lim[EXL3_TB];
    for (int tl = 0; tl < EXL3_TB; tl++) {
        int t = tb + tl;
        lim[tl] = (t < t_len) ? ((int)pp[0] + t + 1) : 0;
    }
    int lmax = 0;
    for (int tl = 0; tl < EXL3_TB; tl++) lmax = max(lmax, lim[tl]);

    for (int tl = 0; tl < EXL3_TB; tl++) {
        int t = tb + tl;
        if (t < t_len) qs4[tl * 256 + tid] = qh[t * 6144 + h * 256 + tid];
    }
    __syncthreads();

    for (int row = sg; row < lmax; row += 4) {
        float kr0 = kc[((long)layer * kvcap + row) * 1024 + kh * 256 + si * 4 + 0];
        float kr1 = kc[((long)layer * kvcap + row) * 1024 + kh * 256 + si * 4 + 1];
        float kr2 = kc[((long)layer * kvcap + row) * 1024 + kh * 256 + si * 4 + 2];
        float kr3 = kc[((long)layer * kvcap + row) * 1024 + kh * 256 + si * 4 + 3];
        for (int tl = 0; tl < EXL3_TB; tl++) {
            if (lim[tl] == 0) {
                sarr[tl * EXL3_SCAP + row] = -1e30f;
                continue;
            }
            float p = qs4[tl * 256 + si * 4 + 0] * kr0
                    + qs4[tl * 256 + si * 4 + 1] * kr1
                    + qs4[tl * 256 + si * 4 + 2] * kr2
                    + qs4[tl * 256 + si * 4 + 3] * kr3;
            // 64레인 완전 리덕션: 32-반합 후 xor 결합(폭 파라미터 의존 제거).
            for (int off = 16; off > 0; off >>= 1)
                p += __shfl_down(p, off);
            p += __shfl_xor(p, 32, 64);
            if (si == 0) {
                bool valid = row < lim[tl];
                sarr[tl * EXL3_SCAP + row] = valid ? (p * scale) : -1e30f;
            }
        }
    }
    __syncthreads();

    for (int tl = 0; tl < EXL3_TB; tl++) {
        if (lim[tl] == 0) continue;
        float lm = -1e30f;
        for (int i = tid; i < lim[tl]; i += 256)
            lm = fmaxf(lm, sarr[tl * EXL3_SCAP + i]);
        reds[tid] = lm;
        __syncthreads();
        for (int st = 128; st > 0; st >>= 1) {
            if (tid < st) reds[tid] = fmaxf(reds[tid], reds[tid + st]);
            __syncthreads();
        }
        float gmax = reds[0];
        if (tid == 0) gw[tl][0] = gmax;
        __syncthreads();
        float ls = 0.0f;
        for (int i = tid; i < lim[tl]; i += 256) {
            float e = expf(sarr[tl * EXL3_SCAP + i] - gmax);
            sarr[tl * EXL3_SCAP + i] = e;
            ls += e;
        }
        reds[tid] = ls;
        __syncthreads();
        for (int st = 128; st > 0; st >>= 1) {
            if (tid < st) reds[tid] += reds[tid + st];
            __syncthreads();
        }
        if (tid == 0) gw[tl][1] = reds[0];
        __syncthreads();
    }

    for (int tl = 0; tl < EXL3_TB; tl++) {
        int t = tb + tl;
        if (t >= t_len) continue;
        float acc = 0.0f;
        for (int row = 0; row < lim[tl]; row++)
            acc += sarr[tl * EXL3_SCAP + row]
                 * vc[((long)layer * kvcap + row) * 1024 + kh * 256 + tid];
        float g = qg[t * 12288 + h * 512 + 256 + tid];
        float sg2 = 1.0f / (1.0f + expf(-g));
        outv[t * 6144 + h * 256 + tid] = (acc / gw[tl][1]) * sg2;
    }
}
// 마커 fw1

// exl3_attn_fwd3s — 단순형(정확성 우선 피벗, 2026-10-04): WG당 (t,h).
// t-블록 공유 제거 — 스코어 스레드 분할+smem, 트리 max/sum, AV 스레드=dim.
// [plans/128 P0] 청크 온라인 소프트맥스로 재작성: 구버전 sarr[1024] smem이
// ctx를 1024로 상한(32k 서빙 폴트). 이제 256행 청크마다 running max/l/acc
// 를 갱신 — smem은 qs/sarr/reds 각 1KB로 고정, kvcap 무관. 환원 순서 변경
// (규칙 10a — 프로브 정합·토큰 게이트로 재검증).
extern "C" __global__ void exl3_attn_fwd3s(
    const float* __restrict__ qh,
    const float* __restrict__ kc,   // [kvcap][1024]
    const float* __restrict__ vc,   // [kvcap][1024]
    const float* __restrict__ qg,
    float* __restrict__ outv,
    const unsigned* __restrict__ pp,
    int t_len, int pos0_, int layer, int kvcap)
{
    __shared__ float qs[256];
    __shared__ float sarr[256]; // 청크 스코어(온라인 — 전체 보관 아님)
    __shared__ float reds[256];
    int t = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    int kh = h / 6;
    float scale = 0.0625f;
    int lim = (int)pp[0] + t + 1;
    qs[tid] = qh[t * 6144 + h * 256 + tid];
    __syncthreads();
    // [결함 판정 기록 2026-10-04] v1/v2 모두 max 트리 후 reds[0](청크 max) 판독과
    // exp 단계의 reds[tid]=e 기록 사이에 배리어가 없어, 스레드 간 속도차로
    // reds[0]을 e로 덮여쓴 값을 max로 읽는 레이스가 있었다(WG 단위 산재 오염,
    // t≥50부터 발현 — 비결정). 수정 = corr 계산 후 전체 배리어 1회.
    float m_run = -1e30f;
    float l_run = 0.0f;
    float acc = 0.0f; // 스레드(dim=tid)별 AV 누산
    for (int base = 0; base < lim; base += 256) {
        int nch = min(256, lim - base);
        // 스코어: 행=base+tid(tid<nch), 256차원 직렬 내적(구버전과 동일 산술)
        float p = -1e30f;
        if (tid < nch) {
            int row = base + tid;
            p = 0.0f;
            for (int d = 0; d < 256; d++)
                p += qs[d] * kc[((long)layer * kvcap + row) * 1024 + kh * 256 + d];
            p *= scale;
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
        float corr = (m_run <= -1e29f) ? 0.0f : expf(m_run - m_new);
        __syncthreads(); // reds[0](청크 max) 판독 완료 후 e 기록 — 덮어쓰기 레이스 방지(결함 판정 기록 참조)
        // exp+청크 sum — e를 sarr에 재기록(AV 재사용)
        float e = 0.0f;
        if (tid < nch) {
            e = expf(sarr[tid] - m_new);
            sarr[tid] = e;
        }
        reds[tid] = e;
        __syncthreads();
        for (int st = 128; st > 0; st >>= 1) {
            if (tid < st) reds[tid] += reds[tid + st];
            __syncthreads();
        }
        // AV: 자기 dim에 청크 전 행 누산
        acc *= corr;
        for (int i = 0; i < nch; i++) {
            int row = base + i;
            acc += sarr[i] * vc[((long)layer * kvcap + row) * 1024 + kh * 256 + tid];
        }
        l_run = l_run * corr + reds[0];
        m_run = m_new;
        __syncthreads(); // sarr 재사용 전 전체 완료(다음 청크 스코어 덮어쓰기 보호)
    }
    float g = qg[t * 12288 + h * 512 + 256 + tid];
    outv[t * 6144 + h * 256 + tid] = (acc / l_run) * (1.0f / (1.0f + expf(-g)));
}
// 마커 f3s

