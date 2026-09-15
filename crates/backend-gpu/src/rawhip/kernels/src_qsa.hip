// QSA attention score: dot(q_head, ck[p]) — grid (n_past, n_head, t)
extern "C" __global__ void qsa_score(const float* q, const float* ck, const unsigned* mask,
                                     float* scores, int n_past, int n_head, int n_kv,
                                     int hd, int t_len, int sstride, int pos0) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y;
    int t = blockIdx.z;
    if (p >= n_past || h >= n_head || t >= t_len) return;
    if (mask[(pos0 + t) * sstride + p] == 0u) {
        scores[(t * n_head + h) * sstride + p] = -3.0e38f;
        return;
    }
    int kvh = h / (n_head / n_kv);
    int qb = t * n_head * 2 * hd + h * 2 * hd;
    int kb = p * n_kv * hd + kvh * hd;
    float d = 0.0f;
    for (int i = 0; i < hd; i++) d += q[qb + i] * ck[kb + i];
    scores[(t * n_head + h) * sstride + p] = d;
}
// QSA attention mix: softmax-weighted V sum + gate — sequential deterministic order
extern "C" __global__ void qsa_mix2(const float* q, const float* scores, const float* cv,
                                    float* out, int n_past, int n_head, int n_kv,
                                    int hd, int t_len, int sstride, int pos0) {
    // exp 64× 중복 제거: (t,h)당 wbuf[p]=exp값 협력 적재 1회 — 가중·순차합
    // 순서는 원소별 불변 (비트무영향). max는 결합법칙으로 스레드별 전체 스캔.
    __shared__ float wbuf[2048];
    int d_i = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y;
    int t = blockIdx.z;
    if (h >= n_head || t >= t_len) return;
    int sbase = (t * n_head + h) * sstride;
    int tid = threadIdx.x;              // 0..63 (hd=64 — 블록=hd)
    // max (스레드별 전체 — 결합법칙, 정확)
    float maxv = scores[sbase];
    for (int p = 0; p < n_past; p++) {
        float sv = scores[sbase + p];
        if (sv > maxv) maxv = sv;
    }
    // exp 재계산 (max 반영) — wbuf 갱신: 1패스, 각 원소 1회
    for (int p = tid; p < n_past; p += 64) wbuf[p] = exp_cr(scores[sbase + p] - maxv);
    __syncthreads();
    if (d_i >= hd) return;
    // 순차 합 (원본 p=0.. 순서 — 비트열 동일)
    float sum = wbuf[0];
    for (int p = 1; p < n_past; p++) sum += wbuf[p];
    int kvh = h / (n_head / n_kv);
    float a = 0.0f;
    for (int p = 0; p < n_past; p++) {
        float w = wbuf[p] / sum;
        if (w != 0.0f) {
            int kb = p * n_kv * hd + kvh * hd;
            a += w * cv[kb + d_i];
        }
    }
    int gb = t * n_head * 2 * hd + h * 2 * hd + hd;
    float g = 1.0f / (1.0f + exp_cr(-q[gb + d_i]));
    out[t * n_head * hd + h * hd + d_i] = a * g;
}


// QSA fused flash: score+softmax+mix 단일 패스 (KV 타일 online-softmax).
// 블록=(t,h) 256스레드, d=tid (hd<=256). q 1회 적재 후 전 p 재사용.
// 마스크 스킵·게이트·exp_cr는 score/mix2와 동일 시맨틱 (p 오름차순 합산 순서는 동일, 반올림 오차만 상이).
extern "C" __global__ void qsa_flash(const float* q, const half* ck, const half* cv,
                                     const unsigned* mask, float* out,
                                     int n_past, int n_head, int n_kv,
                                     int hd, int t_len, int sstride, int pos0) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    if (t >= t_len || h >= n_head) return;
    int tid = threadIdx.x;              // 0..255 (wave32 x8)
    int lane = tid & 31;
    int wid = tid >> 5;
    bool active = tid < hd;
    __shared__ float qf[256];
    __shared__ float rs[8];
    int kvh = h / (n_head / n_kv);
    int qb = t * n_head * 2 * hd + h * 2 * hd;
    if (active) qf[tid] = q[qb + tid];
    __syncthreads();
    float m = -3.4028235e38f;
    float s = 0.0f;
    float acc = 0.0f;
    for (int p = 0; p < n_past; p++) {
        if (mask[(pos0 + t) * sstride + p] == 0u) continue;
        int kb = p * n_kv * hd + kvh * hd;
        float x = active ? qf[tid] * __half2float(ck[kb + tid]) : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            x += __shfl_down_sync(0xffffffffffffffffull, x, off);
        if (lane == 0) rs[wid] = x;
        __syncthreads();
        float d;
        if (wid == 0) {
            float y = lane < 8 ? rs[lane] : 0.0f;
            #pragma unroll
            for (int off = 4; off > 0; off >>= 1)
                y += __shfl_down_sync(0xffffffffffffffffull, y, off);
            if (lane == 0) rs[0] = y;
        }
        __syncthreads();
        d = rs[0];
        float m_new = fmaxf(m, d);
        // 루프 exp는 네이티브 f32 (전 스레드 병렬). f64 exp_cr 단일스레드 공유보다 30배+ 빠름.
        // m-m_new, d-m_new <= 0 이라 오버플로 불가. 최종 오차는 반올림 수준.
        float e_m = __expf(m - m_new);
        float e_d = __expf(d - m_new);
        s = s * e_m + e_d;
        float vv = active ? __half2float(__half2float(cv[kb + tid])) : 0.0f;
        acc = acc * e_m + e_d * vv;
        m = m_new;
    }
    if (!active) return;
    float g = 1.0f / (1.0f + exp_cr(-q[qb + hd + tid]));
    out[t * n_head * hd + h * hd + tid] = acc * (1.0f / s) * g;
}
// QSA split flash, q 2행 다중화 — ck/cv 1회 로드로 t쌍 공유 (K/V 트래픽 ½).
extern "C" __global__ void qsa_flash_split4q4(const float* q, const half* ck, const half* cv,
                                              const unsigned* mask, float* part,
                                              int n_past, int n_head, int n_kv,
                                              int hd, int t_len, int sstride, int pos0, int seg) {
    int t0 = blockIdx.x * 4;
    int h = blockIdx.y;
    int sg = blockIdx.z;
    if (t0 >= t_len || h >= n_head) return;
    int nq = min(4, t_len - t0);
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;
    bool active = tid < hd;
    __shared__ float qf[4][256];
    __shared__ float rs[4][4][8];
    __shared__ float ds[4][4];
    int kvh = h / (n_head / n_kv);
    int qb0 = t0 * n_head * 2 * hd + h * 2 * hd;
    if (active) {
        #pragma unroll
        for (int r = 0; r < 4; r++)
            if (r < nq) qf[r][tid] = q[qb0 + r * n_head * 2 * hd + tid];
    }
    __syncthreads();
    float m[4] = { -3.4028235e38f, -3.4028235e38f, -3.4028235e38f, -3.4028235e38f };
    float s[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    int p0 = sg * seg;
    int p1 = min(p0 + seg, n_past);
    int mrow0 = (pos0 + t0) * sstride;
    
    for (int pb = p0; pb < p1; pb += 4) {
        int np = min(4, p1 - pb);
        float x[4][4] = {};
        bool mka[4][4];
        bool any_mk = false;
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                mka[r][j] = (r < nq) && (j < np) && (mask[mrow0 + r * sstride + pb + j] != 0u);
                if (mka[r][j]) any_mk = true;
            }
        }
        if (any_mk) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                bool jany = false;
                #pragma unroll
                for (int r = 0; r < 4; r++) if (mka[r][j]) jany = true;
                if (jany) {
                    int kb = (pb + j) * n_kv * hd + kvh * hd;
                    float kv = active ? __half2float(ck[kb + tid]) : 0.0f;
                    if (active) {
                        #pragma unroll
                        for (int r = 0; r < 4; r++)
                            if (mka[r][j]) x[r][j] = qf[r][tid] * kv;
                    }
                }
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                x[r][0] += __shfl_down_sync(0xffffffffffffffffull, x[r][0], off);
                x[r][1] += __shfl_down_sync(0xffffffffffffffffull, x[r][1], off);
                x[r][2] += __shfl_down_sync(0xffffffffffffffffull, x[r][2], off);
                x[r][3] += __shfl_down_sync(0xffffffffffffffffull, x[r][3], off);
            }
        }
        if (lane == 0) {
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                rs[r][0][wid] = x[r][0]; rs[r][1][wid] = x[r][1];
                rs[r][2][wid] = x[r][2]; rs[r][3][wid] = x[r][3];
            }
        }
        __syncthreads();
        if (wid == 0) {
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    float y = lane < 8 ? rs[r][j][lane] : 0.0f;
                    #pragma unroll
                    for (int off = 4; off > 0; off >>= 1)
                        y += __shfl_down_sync(0xffffffffffffffffull, y, off);
                    if (lane == 0) ds[r][j] = y;
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            if (r < nq) {
                float gm = m[r];
                #pragma unroll
                for (int j = 0; j < 4; j++)
                    if (j < np) gm = fmaxf(gm, ds[r][j]);
                float e_m = __expf(m[r] - gm);
                s[r] *= e_m;
                acc[r] *= e_m;
                m[r] = gm;
            }
        }
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j < np) {
                int kb = (pb + j) * n_kv * hd + kvh * hd;
                float vv = active ? __half2float(cv[kb + tid]) : 0.0f;
                #pragma unroll
                for (int r = 0; r < 4; r++) {
                    if (r < nq) {
                        float e_d = __expf(ds[r][j] - m[r]);
                        s[r] += e_d;
                        acc[r] += e_d * vv;
                    }
                }
            }
        }
        __syncthreads();
    }
    if (!active) return;
    int nseg = (n_past + seg - 1) / seg;
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        if (r < nq) {
            float* pp2 = part + ((size_t)((t0 + r) * n_head + h) * nseg + sg) * (hd + 2);
            pp2[tid] = acc[r];
            if (tid == 0) { pp2[hd] = m[r]; pp2[hd + 1] = s[r]; }
        }
    }
}

// 병합: out = Σ acc_i·e^(m_i−M) / Σ s_i·e^(m_i−M), 게이트 적용.
// warp-per-query flash (llama fattn-tile 구조 차용): 블록 32쿼리(워프당 4),
// 키 루프 내 sync/shared 왕복 없음 — k/v 재로드는 L1 흡수.
// 수치: ds·softmax 갱신 순서가 split4q4와 다름(키별 running) — 게이트 판정.
extern "C" __global__ void qsa_flash_wk(const float* q, const half* ck, const half* cv,
                                        const unsigned* mask, float* part,
                                        int n_past, int n_head, int n_kv,
                                        int hd, int t_len, int sstride, int pos0, int seg) {
    // 워프퍼쿼리 flash (llama fattn-tile 구조 차용): 블록 32쿼리(워프당 4),
    // 키 루프 내 sync/shared 왕복 없음. hd=32의 배수 전제 (레인당 hd/32 dim).
    // 원본 qsa_flash와 동일 의미: 마스크 꺼진 키는 완전 스킵.
    int q0 = blockIdx.x * 32;
    int h = blockIdx.y;
    int sg = blockIdx.z;
    if (q0 >= t_len || h >= n_head) return;
    int nq = min(32, t_len - q0);
    int lane = threadIdx.x & 31;
    int w = threadIdx.x >> 5;
    int qr = w * 4;
    if (qr >= nq) return;
    int nqw = min(4, nq - qr);
    const int dpl = hd >> 5;              // 레인당 dim 수 (hd=256 → 8)
    int kvh = h / (n_head / n_kv);
    int qb0 = (q0 + qr) * n_head * 2 * hd + h * 2 * hd;
    float qv[4][8];
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int j = 0; j < 8; j++)
            qv[r][j] = (j < dpl && r < nqw) ? q[qb0 + r * n_head * 2 * hd + lane * dpl + j] : 0.0f;
    }
    float m[4] = {-3.4028235e38f, -3.4028235e38f, -3.4028235e38f, -3.4028235e38f};
    float s[4] = {0, 0, 0, 0};
    float acc[4][8] = {};
    int p0 = sg * seg;
    int p1 = min(p0 + seg, n_past);
    int kq0 = (pos0 + q0 + qr) * sstride;
    int kb0 = kvh * hd;
    for (int k = p0; k < p1; k++) {
        float kv[8];
        #pragma unroll
        for (int j = 0; j < 8; j++)
            kv[j] = (j < dpl) ? __half2float(ck[(size_t)k * n_kv * hd + kb0 + lane * dpl + j]) : 0.0f;
        float part_[4];
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            float p_ = 0.0f;
            #pragma unroll
            for (int j = 0; j < 8; j++) p_ = fmaf(qv[r][j], kv[j], p_);
            part_[r] = p_;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float a0 = __shfl_sync(0xFFFFFFFFFFFFFFFFull, part_[0], (lane ^ off) & 31);
            float a1 = __shfl_sync(0xFFFFFFFFFFFFFFFFull, part_[1], (lane ^ off) & 31);
            float a2 = __shfl_sync(0xFFFFFFFFFFFFFFFFull, part_[2], (lane ^ off) & 31);
            float a3 = __shfl_sync(0xFFFFFFFFFFFFFFFFull, part_[3], (lane ^ off) & 31);
            part_[0] += a0; part_[1] += a1; part_[2] += a2; part_[3] += a3;
        }
        float vv[8];
        #pragma unroll
        for (int j = 0; j < 8; j++)
            vv[j] = (j < dpl) ? __half2float(cv[(size_t)k * n_kv * hd + kb0 + lane * dpl + j]) : 0.0f;
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            if (r < nqw && mask[kq0 + r * sstride + k] != 0u) {
                float ds = part_[r];
                float gm = fmaxf(m[r], ds);
                float e_m = __expf(m[r] - gm);
                float e_d = __expf(ds - gm);
                s[r] = s[r] * e_m + e_d;
                m[r] = gm;
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    acc[r][j] = acc[r][j] * e_m + ((j < dpl) ? e_d * vv[j] : 0.0f);
            }
        }
    }
    int nseg = (n_past + seg - 1) / seg;
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            if (r < nqw) {
                float* pp2 = part + ((size_t)((q0 + qr + r) * n_head + h) * nseg + sg) * (hd + 2);
                pp2[hd] = m[r]; pp2[hd + 1] = s[r];
            }
        }
    }
    __syncwarp();
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        if (r < nqw) {
            float* pp2 = part + ((size_t)((q0 + qr + r) * n_head + h) * nseg + sg) * (hd + 2);
            #pragma unroll
            for (int j = 0; j < 8; j++)
                if (j < dpl) pp2[lane * dpl + j] = acc[r][j];
        }
    }
}


extern "C" __global__ void qsa_flash_merge(const float* q, const float* part, float* out,
                                           int n_past, int n_head, int hd, int t_len, int seg) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    if (t >= t_len || h >= n_head) return;
    int tid = threadIdx.x;
    bool active = tid < hd;
    int nseg = (n_past + seg - 1) / seg;
    const float* base = part + ((size_t)(t * n_head + h) * nseg) * (hd + 2);
    // 세그먼트 m_i 는 (hd+2) 플로트 간격 = 1032B 스트라이드라 매 원소가 다른 캐시라인이다.
    // 스레드마다 104개를 직렬로 읽으면 로드 지연이 그대로 노출되므로(실측 80us/런치)
    // 워프 하나가 레인별로 나눠 읽고 셔플로 합친다.
    __shared__ float msh;
    if (tid < 32) {
        float M = -3.4028235e38f;
        for (int i = tid; i < nseg; i += 32) {
            float mi = base[i * (hd + 2) + hd];
            if (mi > M) M = mi;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            M = fmaxf(M, __shfl_xor_sync(0xffffffffffffffffull, M, off));
        if (tid == 0) msh = M;
    }
    __syncthreads();
    float M = msh;
    float a = 0.0f, ssum = 0.0f;
    for (int i = 0; i < nseg; i++) {
        const float* pi = base + i * (hd + 2);
        float w = __expf(pi[hd] - M);   // m_i=-inf 세그 → 0
        float si = pi[hd + 1];
        ssum += si * w;
        if (active) a += pi[tid] * w;
    }
    if (!active) return;
    int qb = t * n_head * 2 * hd + h * 2 * hd;
    float g = 1.0f / (1.0f + exp_cr(-q[qb + hd + tid]));
    out[t * n_head * hd + h * hd + tid] = a * (1.0f / ssum) * g;
}


// GQA 공유 디코드 flash (t=1): WG가 kv-head의 세그먼트를 담당하고 그 kv-head의
// q-head 전부(gq = n_head/n_kv, 보통 6)를 한 번의 K/V 로드로 처리 — K/V 트래픽 1/gq.
// 수치: qsa_flash_split4q4의 4행 구조를 그대로 헤드 축으로 옮긴 것(4키 배치,
// 워프트리+LDS 2단 감축, softmax 갱신 순서 동일) → 헤드별 결과가 비트동일.
extern "C" __global__ void qsa_flash_gqa(const float* q, const float* ck, const float* cv,
                                         const unsigned* mask, float* part,
                                         int n_past, int n_head, int n_kv,
                                         int hd, int t_len, int sstride, int pos0, int seg) {
    int kvh = blockIdx.y;
    int sg = blockIdx.z;
    if (kvh >= n_kv || t_len != 1 || hd > 256) return;
    int gq = n_head / n_kv;
    if (gq < 1 || gq > 8) return;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;
    bool active = tid < hd;
    __shared__ float qf[8][256];
    __shared__ float rs[8][4][8];
    __shared__ float ds[8][4];
    int h0 = kvh * gq;
    if (active) {
        #pragma unroll
        for (int r = 0; r < 8; r++)
            if (r < gq) qf[r][tid] = q[(h0 + r) * 2 * hd + tid];
    }
    __syncthreads();
    float m[8], s[8], acc[8];
    #pragma unroll
    for (int r = 0; r < 8; r++) { m[r] = -3.4028235e38f; s[r] = 0.0f; acc[r] = 0.0f; }
    int p0 = sg * seg;
    int p1 = min(p0 + seg, n_past);
    int mrow0 = pos0 * sstride;
    for (int pb = p0; pb < p1; pb += 4) {
        int np = min(4, p1 - pb);
        float x[8][4] = {};
        bool mkj[4];
        bool any_mk = false;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            mkj[j] = (j < np) && (mask[mrow0 + pb + j] != 0u);  // t=1: 마스크는 키 공통
            if (mkj[j]) any_mk = true;
        }
        if (any_mk) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                if (mkj[j]) {
                    int kb = (pb + j) * n_kv * hd + kvh * hd;   // gq 헤드가 공유
                    float kv = active ? __half2float(ck[kb + tid]) : 0.0f;
                    if (active) {
                        #pragma unroll
                        for (int r = 0; r < 8; r++)
                            if (r < gq) x[r][j] = qf[r][tid] * kv;
                    }
                }
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                if (r < gq) {
                    x[r][0] += __shfl_down_sync(0xffffffffffffffffull, x[r][0], off);
                    x[r][1] += __shfl_down_sync(0xffffffffffffffffull, x[r][1], off);
                    x[r][2] += __shfl_down_sync(0xffffffffffffffffull, x[r][2], off);
                    x[r][3] += __shfl_down_sync(0xffffffffffffffffull, x[r][3], off);
                }
            }
        }
        if (lane == 0) {
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (r < gq) { rs[r][0][wid] = x[r][0]; rs[r][1][wid] = x[r][1]; rs[r][2][wid] = x[r][2]; rs[r][3][wid] = x[r][3]; }
        }
        __syncthreads();
        if (wid == 0) {
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                if (r < gq) {
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        float y = lane < 8 ? rs[r][j][lane] : 0.0f;
                        #pragma unroll
                        for (int off = 4; off > 0; off >>= 1)
                            y += __shfl_down_sync(0xffffffffffffffffull, y, off);
                        if (lane == 0) ds[r][j] = y;
                    }
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            if (r < gq) {
                float gm = m[r];
                #pragma unroll
                for (int j = 0; j < 4; j++) if (j < np) gm = fmaxf(gm, ds[r][j]);
                float e_m = __expf(m[r] - gm);
                s[r] *= e_m;
                acc[r] *= e_m;
                m[r] = gm;
            }
        }
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j < np) {
                int kb = (pb + j) * n_kv * hd + kvh * hd;
                float vv = active ? __half2float(cv[kb + tid]) : 0.0f;
                #pragma unroll
                for (int r = 0; r < 8; r++) {
                    if (r < gq) {
                        float e_d = __expf(ds[r][j] - m[r]);
                        s[r] += e_d;
                        acc[r] += e_d * vv;
                    }
                }
            }
        }
        __syncthreads();
    }
    if (!active) return;
    int nseg = (n_past + seg - 1) / seg;
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        if (r < gq) {
            // t=1이므로 행 인덱스 0, 헤드는 (h0 + r) — split4q4의 (t0+r)*n_head+h 와 동일
            float* pp2 = part + ((size_t)(h0 + r) * nseg + sg) * (hd + 2);
            pp2[tid] = acc[r];
            if (tid == 0) { pp2[hd] = m[r]; pp2[hd + 1] = s[r]; }
        }
    }
}

// 워프퍼쿼리 flash, 16레인/행 판 (prefill 전용). 기존 qsa_flash_wk 는 32레인/행이고 4행을
// 순차 처리하므로 (행,키)당 hd MAC + L·2·log2(L) 셔플 레인연산이 든다. 여기서는 16레인/행으로
// 2행을 워프 안에서 동시에 처리한다: 워프당 2행 x (hd MAC + 16·2·4 셔플) → 약 1.5x 절감.
// 그리드 (ceil(t/16), n_head, nseg), 256스레드=8워프, 워프당 2행. hd=256 전제(레인당 16 dim).
extern "C" __global__ void qsa_flash_wk16(const float* q, const half* ck, const half* cv,
                                          const unsigned* mask, float* part,
                                          int n_past, int n_head, int n_kv,
                                          int hd, int t_len, int sstride, int pos0, int seg) {
    if (hd != 256) return;
    int q0 = blockIdx.x * 16;
    int h = blockIdx.y;
    int sg = blockIdx.z;
    if (q0 >= t_len || h >= n_head) return;
    int nq = min(16, t_len - q0);
    int lane = threadIdx.x & 31;
    int w = threadIdx.x >> 5;
    int grp = lane >> 4;             // 0/1 → 워프 내 행 슬롯
    int own = lane & 15;             // 16레인 그룹 내 위치 → 담당 dim
    int r = w * 2 + grp;
    // 셔플은 워프 전원 참여가 필수 — 범위 밖 행도 조기 반환하지 않고 계산만 하고 저장을 막는다.
    bool real = (r < nq);
    int rf = real ? r : 0;
    int kvh = h / (n_head / n_kv);
    const float* qrow = q + ((size_t)(q0 + rf) * n_head + h) * 2 * hd;
    float qv[16];
    #pragma unroll
    for (int j = 0; j < 16; j++) qv[j] = qrow[own * 16 + j];
    float m = -3.4028235e38f, s = 0.0f;
    float acc[16];
    #pragma unroll
    for (int j = 0; j < 16; j++) acc[j] = 0.0f;
    int p0 = sg * seg;
    int p1 = min(p0 + seg, n_past);
    int kb0 = kvh * hd;
    int kq = (pos0 + q0 + rf) * sstride;
    for (int k = p0; k < p1; k++) {
        const half* krow = ck + (size_t)k * n_kv * hd + kb0 + own * 16;
        float kv[16];
        #pragma unroll
        for (int j = 0; j < 16; j++) kv[j] = __half2float(krow[j]);
        float pr = 0.0f;
        #pragma unroll
        for (int j = 0; j < 16; j++) pr = fmaf(qv[j], kv[j], pr);
        #pragma unroll
        for (int off = 8; off > 0; off >>= 1)
            pr += __shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, pr, off);
        if (real && mask[kq + k] != 0u) {
            float gm = fmaxf(m, pr);
            float e_m = __expf(m - gm);
            float e_d = __expf(pr - gm);
            s = s * e_m + e_d;
            m = gm;
            const half* vrow = cv + (size_t)k * n_kv * hd + kb0 + own * 16;
            #pragma unroll
            for (int j = 0; j < 16; j++) acc[j] = acc[j] * e_m + e_d * __half2float(vrow[j]);
        }
    }
    if (!real) return;
    int nseg = (n_past + seg - 1) / seg;
    float* pp2 = part + ((size_t)((q0 + r) * n_head + h) * nseg + sg) * (hd + 2);
    if (own == 0) { pp2[hd] = m; pp2[hd + 1] = s; }
    #pragma unroll
    for (int j = 0; j < 16; j++) pp2[own * 16 + j] = acc[j];
}

// rocwmma 16x16x16 프래그먼트 레이아웃 검증 프로브.
// mode 0: B col_major (QK^T 용) — C[i][j] = Σ_k a[i][k]*b[j][k]
// mode 1: B row_major (PV 용)  — C[i][n] = Σ_k a[i][k]*b[k][n]
extern "C" __global__ void wmma_probe(const float* a, const float* b, float* c, int mode) {
    __shared__ half sa[256];
    __shared__ half sb[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        sa[i] = __float2half(a[i]);
        sb[i] = __float2half(b[i]);
    }
    __syncthreads();
    fragment<accumulator, 16, 16, 16, float> fc;
    fill_fragment(fc, 0.0f);
    fragment<matrix_a, 16, 16, 16, half, row_major> fa;
    load_matrix_sync(fa, sa, 16);
    if (mode == 0) {
        fragment<matrix_b, 16, 16, 16, half, col_major> fb;
        load_matrix_sync(fb, sb, 16);
        mma_sync(fc, fa, fb, fc);
    } else {
        fragment<matrix_b, 16, 16, 16, half, row_major> fb;
        load_matrix_sync(fb, sb, 16);
        mma_sync(fc, fa, fb, fc);
    }
    int lane = threadIdx.x & 31;
    #pragma unroll
    for (int sl = 0; sl < 8; sl++) {
        int idx = lane + (sl << 5);
        c[idx] = fc.x[sl];
    }
}

// 진단 전용: wk16 의 8레인/행 판(버터플라이 3단). 모델 경로에는 연결하지 않고
// `llm170 attn-check` 가 wk16 과 동일 입력으로 비교하는 데만 쓴다(plans/47 판별 하네스).
extern "C" __global__ void qsa_flash_wk8(const float* q, const half* ck, const half* cv,
                                         const unsigned* mask, float* part,
                                         int n_past, int n_head, int n_kv,
                                         int hd, int t_len, int sstride, int pos0, int seg) {
    if (hd != 256) return;
    int q0 = blockIdx.x * 32;
    int h = blockIdx.y;
    int sg = blockIdx.z;
    if (q0 >= t_len || h >= n_head) return;
    int nq = min(32, t_len - q0);
    int lane = threadIdx.x & 31;
    int w = threadIdx.x >> 5;
    int grp = lane >> 3;
    int own = lane & 7;
    int r = w * 4 + grp;
    bool real = (r < nq);
    int rf = real ? r : 0;
    int kvh = h / (n_head / n_kv);
    const float* qrow = q + ((size_t)(q0 + rf) * n_head + h) * 2 * hd;
    float qv[32];
    #pragma unroll
    for (int j = 0; j < 32; j++) qv[j] = qrow[own * 32 + j];
    float m = -3.4028235e38f, ssum = 0.0f;
    float acc[32];
    #pragma unroll
    for (int j = 0; j < 32; j++) acc[j] = 0.0f;
    int p0 = sg * seg;
    int p1 = min(p0 + seg, n_past);
    int kb0 = kvh * hd + own * 32;
    int kq = (pos0 + q0 + rf) * sstride;
    for (int k = p0; k < p1; k++) {
        const half* kp = ck + (size_t)k * n_kv * hd + kb0;
        float kv[32];
        #pragma unroll
        for (int j = 0; j < 32; j++) kv[j] = __half2float(kp[j]);
        float pr = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; j++) pr = fmaf(qv[j], kv[j], pr);
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1)
            pr += __shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, pr, off);
        if (real && mask[kq + k] != 0u) {
            float gm = fmaxf(m, pr);
            float e_m = __expf(m - gm);
            float e_d = __expf(pr - gm);
            ssum = ssum * e_m + e_d;
            m = gm;
            const half* vp = cv + (size_t)k * n_kv * hd + kb0;
            #pragma unroll
            for (int j = 0; j < 32; j++) acc[j] = acc[j] * e_m + e_d * __half2float(vp[j]);
        }
    }
    if (!real) return;
    int nseg = (n_past + seg - 1) / seg;
    float* pp2 = part + ((size_t)((q0 + r) * n_head + h) * nseg + sg) * (hd + 2);
    if (own == 0) { pp2[hd] = m; pp2[hd + 1] = ssum; }
    #pragma unroll
    for (int j = 0; j < 32; j++) pp2[own * 32 + j] = acc[j];
}

// plans/73: wk8의 ILP 판(qsa_flash_wk8i). 원판은 키당 pr = fmaf 32연쇄(엄격 FP
// 재결합 불가)가 임계 경로라 키 직렬 처리 시 체인 레이턴시가 그대로 노출된다.
// 여기선 레인이 **키 4개의 fmaf 체인을 인터리브**해 레이턴시를 4키로 분산한다.
// 메모리 패턴·차원 배치·셔플 트리·소프트맥스 갱신 순서(키 오름차순)는 원판
// 그대로 → **비트 동일**. (공유 타일 판은 8레인×64B 스트라이드 읽기의 4-way
// 뱅크 충돌 + 32KB 공유로 점유율이 무너져 실측 역행 — pp16k 257→209 — 폐기.)
// LLM170_NO_WK8I=1 복귀.
extern "C" __global__ void __launch_bounds__(256) qsa_flash_wk8i(
        const float* q, const half* ck, const half* cv,
        const unsigned* mask, float* part,
        int n_past, int n_head, int n_kv,
        int hd, int t_len, int sstride, int pos0, int seg) {
    if (hd != 256) return;
    int q0 = blockIdx.x * 32;
    int h = blockIdx.y;
    int sg = blockIdx.z;
    if (q0 >= t_len || h >= n_head) return;
    int nq = min(32, t_len - q0);
    int lane = threadIdx.x & 31;
    int w = threadIdx.x >> 5;
    int grp = lane >> 3;
    int own = lane & 7;
    int r = w * 4 + grp;
    bool real = (r < nq);
    int rf = real ? r : 0;
    int kvh = h / (n_head / n_kv);
    const float* qrow = q + ((size_t)(q0 + rf) * n_head + h) * 2 * hd;
    float qv[32];
    #pragma unroll
    for (int j = 0; j < 32; j++) qv[j] = qrow[own * 32 + j];
    float m = -3.4028235e38f, ssum = 0.0f;
    float acc[32];
    #pragma unroll
    for (int j = 0; j < 32; j++) acc[j] = 0.0f;
    int p0 = sg * seg;
    int p1 = min(p0 + seg, n_past);
    int kb0 = kvh * hd + own * 32;
    const size_t krow_stride = (size_t)n_kv * hd;
    int kq = (pos0 + q0 + rf) * sstride;
    for (int kb = p0; kb < p1; kb += 4) {
        const int nk4 = min(4, p1 - kb);
        float pr[4];
        #pragma unroll
        for (int kk = 0; kk < 4; kk++) pr[kk] = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            // 꼬리 그룹: 유효 키 범위 밖 로드는 마지막 유효 키로 클램프(값은 버림).
            const int k1 = (1 < nk4) ? kb + 1 : kb;
            const int k2 = (2 < nk4) ? kb + 2 : kb;
            const int k3 = (3 < nk4) ? kb + 3 : kb;
            const float kj0 = __half2float(ck[(size_t)kb * krow_stride + kb0 + j]);
            const float kj1 = __half2float(ck[(size_t)k1 * krow_stride + kb0 + j]);
            const float kj2 = __half2float(ck[(size_t)k2 * krow_stride + kb0 + j]);
            const float kj3 = __half2float(ck[(size_t)k3 * krow_stride + kb0 + j]);
            pr[0] = fmaf(qv[j], kj0, pr[0]);
            pr[1] = fmaf(qv[j], kj1, pr[1]);
            pr[2] = fmaf(qv[j], kj2, pr[2]);
            pr[3] = fmaf(qv[j], kj3, pr[3]);
        }
        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            if (kk >= nk4) continue;
            float prr = pr[kk];
            #pragma unroll
            for (int off = 4; off > 0; off >>= 1)
                prr += __shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, prr, off);
            if (real && mask[kq + kb + kk] != 0u) {
                float gm = fmaxf(m, prr);
                float e_m = __expf(m - gm);
                float e_d = __expf(prr - gm);
                ssum = ssum * e_m + e_d;
                m = gm;
                const half* vp = cv + (size_t)(kb + kk) * krow_stride + kb0;
                #pragma unroll
                for (int j = 0; j < 32; j++)
                    acc[j] = acc[j] * e_m + e_d * __half2float(vp[j]);
            }
        }
    }
    if (!real) return;
    int nseg = (n_past + seg - 1) / seg;
    float* pp2 = part + ((size_t)((q0 + r) * n_head + h) * nseg + sg) * (hd + 2);
    if (own == 0) { pp2[hd] = m; pp2[hd + 1] = ssum; }
    #pragma unroll
    for (int j = 0; j < 32; j++) pp2[own * 32 + j] = acc[j];
}

// mode2 프로브: 16행 x 256dim 타일을 ldm=256 으로 적재 (어텐션 커널과 동일 패턴 검증).
extern "C" __global__ void wmma_probe_ldm(const float* a, const float* b, float* c) {
    __shared__ half sa[16 * 256];
    __shared__ half sb[16 * 256];
    for (int i = threadIdx.x; i < 16 * 256; i += blockDim.x) {
        sa[i] = __float2half(a[i]);
        sb[i] = __float2half(b[i]);
    }
    __syncthreads();
    fragment<accumulator, 16, 16, 16, float> fc;
    fill_fragment(fc, 0.0f);
    #pragma unroll
    for (int kc = 0; kc < 16; kc++) {
        fragment<matrix_a, 16, 16, 16, half, row_major> fa;
        fragment<matrix_b, 16, 16, 16, half, col_major> fb;
        load_matrix_sync(fa, &sa[kc * 16], 256);
        load_matrix_sync(fb, &sb[kc * 16], 256);
        mma_sync(fc, fa, fb, fc);
    }
    int lane = threadIdx.x & 31;
    #pragma unroll
    for (int sl = 0; sl < 8; sl++) c[lane + (sl << 5)] = fc.x[sl];
}


// plans/73: 프리필 어텐션의 v_dot2 판(qsa_flash_wk8d). 스칼라 wk8/wk8i 는
// f32 fmaf 로 어텐션 FLOP의 ~6% 효율(103 GFLOP/층·청크 @16k — pp16k 병목).
// 디코드 기본커널 gqa2d 의 구조를 t>1 로 확장한다: 워프=쿼리 1개, 레인=키,
// 256차원 내적을 레인 안 v_dot2_f32_f16 128회로 끝낸다(QK 셔플 0회). K/V
// 32키 타일은 블록 공유(8쿼리 × 24헤드 재사용). 수치는 f16-q × f16-K f32 누적
// (gqa2d/wmma 와 같은 클래스) — greedy 스트림 동일성으로 검증.
// LLM170_NO_WK8D=1 이면 wk8i 로 복귀.
extern "C" __global__ void __launch_bounds__(256) qsa_flash_wk8d(
        const float* q, const half* ck, const half* cv,
        const unsigned* mask, float* part,
        int n_past, int n_head, int n_kv,
        int hd, int t_len, int sstride, int pos0, int seg) {
    if (hd != 256) return;
    const int q0 = blockIdx.x * 8;
    const int h = blockIdx.y;
    const int sg = blockIdx.z;
    if (q0 >= t_len || h >= n_head) return;
    const int nq = min(8, t_len - q0);
    const int kvh = h / (n_head / n_kv);
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;          // 워프 = 쿼리 q0 + wid
    const bool real = wid < nq;

    constexpr int KS = 258;
    __shared__ half q16[8][256];
    __shared__ half kv16[32 * KS];
    __shared__ float pf[8][32];
    __shared__ float stnew[8];
    __shared__ float smfin[8], sfin[8];

    for (int i = tid; i < 8 * 256; i += 256) {
        int w2 = i >> 8, d = i & 255;
        q16[w2][d] = (w2 < nq)
            ? __float2half(q[((size_t)(q0 + w2) * n_head + h) * 2 * hd + d])
            : __float2half(0.0f);
    }
    __syncthreads();

    // 러닝 최대/합은 **공유 배열**(gqa2d 규약): 레지스터로 두면 lane 0 만
    // 갱신해 레인별 스케일이 갈라진다(실측: 청크 경계부터 스트림 붕괴).
    __shared__ float mrun[8], srun[8];
    if (tid < 8) { mrun[tid] = -3.4028235e38f; srun[tid] = 0.0f; }
    __syncthreads();
    float m[8], acc[8];                 // 스레드=dim, 쿼리 8개 전부 (gqa2d PV 규약)
    #pragma unroll
    for (int r2 = 0; r2 < 8; r2++) { m[r2] = -3.4028235e38f; acc[r2] = 0.0f; }

    const int p0 = sg * seg;
    const int p1 = min(p0 + seg, n_past);
    const int mrow = real ? (pos0 + q0 + wid) * sstride : pos0 * sstride;

    for (int pb = p0; pb < p1; pb += 32) {
        const int np = min(32, p1 - pb);
        // K 타일 적재(uint4 = half8, d 방향 연속 = 코얼레스)
        for (int i = tid * 8; i < 32 * 256; i += 256 * 8) {
            int row = i >> 8, d = i & 255;
            half* dst = &kv16[row * KS + d];
            if (row < np) {
                const uint4 v = *reinterpret_cast<const uint4*>(
                        ck + ((size_t)(pb + row) * n_kv + kvh) * hd + d);
                const half* hs = reinterpret_cast<const half*>(&v);
                #pragma unroll
                for (int j = 0; j < 8; j++) dst[j] = hs[j];
            } else {
                #pragma unroll
                for (int j = 0; j < 8; j++) dst[j] = __float2half(0.0f);
            }
        }
        __syncthreads();
        // QK: 워프=쿼리, 레인=키. 내적 전체가 레인 안에서 끝난다(셔플 0회).
        {
            float a = 0.0f;
            const half2* q2 = reinterpret_cast<const half2*>(&q16[wid][0]);
            const half2* k2 = reinterpret_cast<const half2*>(&kv16[lane * KS]);
            #pragma unroll 2
            for (int d2 = 0; d2 < 128; d2++) {
                half2 qv2 = q2[d2], kv2 = k2[d2];
                asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(a) : "v"(qv2), "v"(kv2));
            }
            const int kk = pb + lane;
            // 프리필 인과경계 너머의 **완전 마스크 타일**이 존재한다(디코드 t=1엔
            // 없는 경우 — gqa2d 는 이 경로를 안 건넜다). 마스크된 키의 e 는
            // 반드시 0: -MAX-MAX=0 → exp(0)=1 로 미래 키가 확률 1로 새는 것 방지.
            const bool okk = (lane < np) && (mask[mrow + kk] != 0u);
            a = okk ? a : -3.4028235e38f;
            float mx = a;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffffffffffull, mx, off));
            const float m_old = mrun[wid];
            const float mn = fmaxf(m_old, mx);
            const float e = okk ? __expf(a - mn) : 0.0f;
            pf[wid][lane] = real ? e : 0.0f;
            float es = e;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                es += __shfl_xor_sync(0xffffffffffffffffull, es, off);
            if (lane == 0) { srun[wid] = srun[wid] * __expf(m_old - mn) + es; mrun[wid] = mn; }
            stnew[wid] = mn;
        }
        __syncthreads();
        #pragma unroll
        for (int r2 = 0; r2 < 8; r2++) {
            acc[r2] *= __expf(m[r2] - stnew[r2]);
            m[r2] = stnew[r2];
        }
        __syncthreads();
        // V 타일 적재(K 위에 덮어씀)
        for (int i = tid * 8; i < 32 * 256; i += 256 * 8) {
            int row = i >> 8, d = i & 255;
            half* dst = &kv16[row * KS + d];
            if (row < np) {
                const uint4 v = *reinterpret_cast<const uint4*>(
                        cv + ((size_t)(pb + row) * n_kv + kvh) * hd + d);
                const half* hs = reinterpret_cast<const half*>(&v);
                #pragma unroll
                for (int j = 0; j < 8; j++) dst[j] = hs[j];
            } else {
                #pragma unroll
                for (int j = 0; j < 8; j++) dst[j] = __float2half(0.0f);
            }
        }
        __syncthreads();
        {
            // 각 스레드(=dim)가 블록의 8쿼리 전부에 누적 — V 행은 1회만 읽는다.
            // (초판은 자기 워프의 쿼리만 봐 쿼리 w 행의 dims 32w..32w+31 만 채웠다.)
            float ar[8];
            #pragma unroll
            for (int r2 = 0; r2 < 8; r2++) ar[r2] = acc[r2];
            for (int k2i = 0; k2i < np; k2i++) {
                const float vv = __half2float(kv16[k2i * KS + tid]);
                #pragma unroll
                for (int r2 = 0; r2 < 8; r2++) ar[r2] += pf[r2][k2i] * vv;
            }
            #pragma unroll
            for (int r2 = 0; r2 < 8; r2++) acc[r2] = ar[r2];
        }
        __syncthreads();
    }
    if (wid < nq && lane == 0) { smfin[wid] = mrun[wid]; sfin[wid] = srun[wid]; }
    __syncthreads();
    const int nseg = (n_past + seg - 1) / seg;
    // 기록도 8쿼리 전부: 스레드(=dim)가 자기 dim 을 모든 쿼리 행에 쓴다
    // (gqa2d 최종 기록과 동일 구조 — 초판은 자기 워프 행만 써 dims 32w.. 만 채웠다).
    #pragma unroll
    for (int r2 = 0; r2 < 8; r2++) {
        if (r2 < nq) {
            float* pp2 = part + ((size_t)((q0 + r2) * n_head + h) * nseg + sg) * (hd + 2);
            pp2[tid] = acc[r2];
            if (tid == 0) { pp2[hd] = smfin[r2]; pp2[hd + 1] = sfin[r2]; }
        }
    }
}

// PV 경로 프로브: A = P(16x16, ldm=16) / B = V(16키 x 256dim, **row_major**, ldm=256).
// 어텐션 커널의 PV 와 동일 패턴 — ldm 프로브(mode2)는 col_major(K)만 검증했다.
extern "C" __global__ void wmma_probe_pv(const float* p, const float* v, float* c) {

    __shared__ half sp[16 * 16];
    __shared__ half sv[16 * 256];
    for (int i = threadIdx.x; i < 16 * 16; i += blockDim.x) sp[i] = __float2half(p[i]);
    for (int i = threadIdx.x; i < 16 * 256; i += blockDim.x) sv[i] = __float2half(v[i]);
    __syncthreads();
    fragment<accumulator, 16, 16, 16, float> fc;
    fill_fragment(fc, 0.0f);
    fragment<matrix_a, 16, 16, 16, half, row_major> fa;
    load_matrix_sync(fa, sp, 16);
    fragment<matrix_b, 16, 16, 16, half, row_major> fb;
    load_matrix_sync(fb, sv, 256);
    mma_sync(fc, fa, fb, fc);
    int lane = threadIdx.x & 31;
    #pragma unroll
    for (int sl = 0; sl < 8; sl++) c[lane + (sl << 5)] = fc.x[sl];
}

// ─── plans/73: PLE 디바이스 포팅(디코드) ───
// ple_bridge는 스텝당 4.5-11ms(레이어 1의 동기 d2h + CPU mm_batch 투영 2회
// [2560→10240] + 호스트 원소연산)이었다. 해시/gather는 GPU 무의존이라 호스트가
// 스텝 초에 수행하고, 투영은 프레임 GEMM, 아래 3커널이 수학을 맡는다.
// 산술은 stages/ple.rs 의 루프 순서를 **그대로** — 스레드당 (t,s) 직렬로
// 미러해 비트 동일(rms_norm 32세그 f64, dot i-오름차순 f32 직렬, silu/sigmoid
// 는 공유 exp_cr).

// (1) 게이트+방송+그룹 norm: k_n=rms(key[s]), q_n=rms(res[s]) → gate →
//     gated=value·gate → rms(n_conv) → gated_hist. gate_out은 (3)이 재사용.
//     스레드 = (토큰, 스트림). 2560 원소 ×수 패스 — μs급.
DEV float ple_rms_scale(const float* x, int n_embd, float eps) {
    const int chunk = (n_embd + 31) >> 5;
    float parts[32];
    #pragma unroll
    for (int u = 0; u < 32; u++) {
        float acc = 0.0f;
        const int lo = u * chunk, hi = min(lo + chunk, n_embd);
        for (int i = lo; i < hi; i++) { const float dv = x[i]; acc += dv * dv; }
        parts[u] = acc;
    }
    double sum = 0.0;
    #pragma unroll
    for (int u = 0; u < 32; u++) sum += (double)parts[u];
    return 1.0f / (float)sqrt(sum / (double)n_embd + (double)eps);
}

extern "C" __global__ void __launch_bounds__(256) q4_ple_gate(
        const float* __restrict__ res, const float* __restrict__ key,
        const float* __restrict__ value,
        const float* __restrict__ nk, const float* __restrict__ nq,
        const float* __restrict__ nc,
        float* __restrict__ gated_hist, float* __restrict__ gate_out,
        float eps, int n_embd, int hc, int t) {
    const int s = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int ti = blockIdx.y;
    const int lane = threadIdx.x & 31;
    if (s >= hc || ti >= t || lane != 0) return;
    const float* ks_row = key + (size_t)ti * hc * n_embd + (size_t)s * n_embd;
    const float* rs_row = res + (size_t)ti * hc * n_embd + (size_t)s * n_embd;
    const float sk = ple_rms_scale(ks_row, n_embd, eps);
    const float sq = ple_rms_scale(rs_row, n_embd, eps);
    // dot: k_n[i]·q_n[i] 를 i 오름차순 f32 직렬 누산(호스트와 동일).
    // k_n[i] = (key[i]·sk)·nk[i] — 저장 없이 동일 반올림 사슬로 재현.
    float dot = 0.0f;
    const float* nk_s = nk + (size_t)s * n_embd;   // 노름 가중치는 스트림별 슬라이스
    const float* nq_s = nq + (size_t)s * n_embd;
    for (int i = 0; i < n_embd; i++) {
        const float a = (ks_row[i] * sk) * nk_s[i];
        const float b = (rs_row[i] * sq) * nq_s[i];
        dot += a * b;
    }
    dot /= (float)sqrt((float)n_embd);
    const float mag = fmaxf(fabsf(dot), 1e-6f);
    const float g = 1.0f / (1.0f + exp_cr_exact(dot >= 0.0f ? -sqrtf(mag) : sqrtf(mag)));
    gate_out[(size_t)ti * hc + s] = g;
    // gated = value·gate → rms(n_conv). value는 [t][n_embd] 스트림 공유.
    const float* val = value + (size_t)ti * n_embd;
    float* out = gated_hist + (size_t)ti * hc * n_embd + (size_t)s * n_embd;
    for (int i = 0; i < n_embd; i++) out[i] = val[i] * g;
    const float sg = ple_rms_scale(out, n_embd, eps);
    for (int i = 0; i < n_embd; i++) out[i] = out[i] * sg * nc[(size_t)s * n_embd + i];
}

// (2) dilated depthwise conv(kern=4, dil=3, hist=9) + silu + 링 갱신.
//     out[c] = Σ_k conv_w[c·kern+k]·padded[hist+ti-(kern-1-k)·dil][c] (k 오름차순).
//     링 st[j] = padded[t+j] (마지막 hist 열).
extern "C" __global__ void __launch_bounds__(256) q4_ple_conv(
        const float* __restrict__ gated_hist, const float* __restrict__ conv_w,
        float* __restrict__ ring, float* __restrict__ conv_out,
        int hc_dim, int t, int kern, int dil, int hist) {
    const int c = (int)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= hc_dim) return;
    for (int ti = 0; ti < t; ti++) {
        float acc = 0.0f;
        for (int k = 0; k < kern; k++) {
            const int start = hist + ti - (kern - 1 - k) * dil;
            const float* src;
            float tmp;
            if (start >= hist) src = gated_hist + (size_t)(start - hist) * hc_dim + c;
            else src = ring + (size_t)start * hc_dim + c;
            acc += conv_w[c * kern + k] * (*src);
            (void)tmp;
        }
        conv_out[(size_t)ti * hc_dim + c] = acc / (1.0f + exp_cr_exact(-acc));
    }
    // 링 갱신(스레드 c별): st[j] = padded[t+j] — t<hist 일부는 기존 링 이동.
    for (int j = 0; j < hist; j++) {
        const int src_col = t + j;   // padded 인덱스
        float v;
        if (src_col >= hist) v = gated_hist[(size_t)(src_col - hist) * hc_dim + c];
        else v = ring[(size_t)src_col * hc_dim + c];
        ring[(size_t)j * hc_dim + c] = v;
    }
}

// (3) 잔차: res[s·n+i] += value[i]·g + conv_out[s·n+i] — 호스트 식 그대로
//     ((value·g) + conv_out) 를 한 번에 가산.
extern "C" __global__ void q4_ple_residual(
        float* __restrict__ res, const float* __restrict__ value,
        const float* __restrict__ gate_out, const float* __restrict__ conv_out,
        int n_embd, int hc, int t) {
    const int i = (int)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    for (int ti = 0; ti < t; ti++) {
        const float v = value[(size_t)ti * n_embd + i];
        for (int s = 0; s < hc; s++) {
            const size_t idx = (size_t)ti * hc * n_embd + (size_t)s * n_embd + i;
            res[idx] += v * gate_out[(size_t)ti * hc + s] + conv_out[idx];
        }
    }
}

// WMMA prefill 어텐션 (hd=256, 진단용). 조기 반환 없음(모든 가드가 플래그), 그래서
// 키-타일 __syncthreads() 와 발산이 상호작용하지 않는다. 프리미티브는 세 프로브로 검증됨.
extern "C" __global__ void __launch_bounds__(256) qsa_flash_wmma(
        const float* q, const half* ck, const half* cv,
        const unsigned* mask, float* part,
        int n_past, int n_head, int n_kv, int hd, int t_len, int sstride, int pos0, int seg) {
    if (hd != 256) return;
    int q0 = blockIdx.x * 64;
    int h = blockIdx.y;
    int sg = blockIdx.z;
    if (q0 >= t_len || h >= n_head) return;      // 블록 균일 → 발산 없음
    int nq = min(64, t_len - q0);
    int tid = threadIdx.x;
    int w = tid >> 5, lane = tid & 31;
    int pair = w >> 1, hf = w & 1;
    int qr = pair * 16;
    int nqw = (qr < nq) ? min(16, nq - qr) : 0;
    int kvh = h / (n_head / n_kv);
    int p0 = sg * seg, p1 = min(p0 + seg, n_past);
    int hc = lane >> 4, cc = lane & 15;

    // llama 의 Q_in_reg 를 따른다: Q 를 레지스터 프래그먼트로 소비한 뒤 그 공유 버퍼를
    // K/V 타일로 재사용한다. 공유 예산이 (Q + KV + P) 합이 아니라 최대값이 되므로
    // 61440 -> 32768B, 점유율 1 -> 2 블록/CU 를 노린다.
    extern __shared__ float gSmem[];
    half* gQ16  = (half*)gSmem;                  // 0 .. 32768  (Q 전용, 이후 재사용)
    half* gK16  = (half*)gSmem;                  // 0 .. 8192   (K)
    half* gV16  = (half*)(gSmem + 2048);         // 8192 .. 16384 (V)
    float* gS32 = (float*)(gSmem + 4096);        // 16384 .. 24576 (스코어 교환)
    half* gP16  = (half*)(gSmem + 6144);         // 24576 .. 26624 (P)

    for (int i = tid; i < 4 * 16 * 256; i += 256) {
        int t2 = i >> 12, rr = (i >> 8) & 15, d = i & 255;
        int row = t2 * 16 + rr, src = q0 + row;
        gQ16[i] = (row < nq) ? __float2half(q[((size_t)src * n_head + h) * 2 * hd + d])
                             : __float2half(0.0f);
    }
    __syncthreads();
    // 절반 워프는 QK 의 k 를 8청크씩 나눠 맡으므로 자기 8개 프래그먼트만 로드한다
    // (프래그먼트 8개 = half2 32개 = VGPR 32개).
    fragment<matrix_a, 16, 16, 16, half, row_major> QA[8];
    #pragma unroll
    for (int kc = 0; kc < 8; kc++) {
        int kd = hf * 8 + kc;
        load_matrix_sync(QA[kc], &gQ16[pair * 16 * 256 + kd * 16], 256);
    }
    __syncthreads();   // 전 워프가 QA 로드 완료 → gSmem[0..8192) 를 K/V 로 덮어써도 된다

    float mrun[8], srun[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) { mrun[j] = -3.4028235e38f; srun[j] = 0.0f; }
    fragment<accumulator, 16, 16, 16, float> fc[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) fill_fragment(fc[j], 0.0f);

    for (int kt = p0; kt < p1; kt += 16) {
        int nk = min(16, p1 - kt);
        for (int i = tid * 4; i < 2 * 16 * 256; i += 1024) {
            int row = i >> 8, d = i & 255;
            bool isk = (row < 16);
            int r2 = isk ? row : (row - 16);
            half* dst = (isk ? &gK16[row * 256 + d] : &gV16[(row - 16) * 256 + d]);
            if (r2 < nk) {
                const half* src = isk ? ck : cv;
                const uint2 raw = *reinterpret_cast<const uint2*>(
                        src + ((size_t)(kt + r2) * n_kv + kvh) * hd + d);
                const half* hs = reinterpret_cast<const half*>(&raw);
                dst[0] = hs[0]; dst[1] = hs[1]; dst[2] = hs[2]; dst[3] = hs[3];
            } else {
                dst[0] = __float2half(0.0f); dst[1] = __float2half(0.0f);
                dst[2] = __float2half(0.0f); dst[3] = __float2half(0.0f);
            }
        }
        __syncthreads();
        // QK 를 절반 워프가 8 mma 씩 분담(전체 16)한 뒤 공유로 합산한다.
        fragment<accumulator, 16, 16, 16, float> fs;
        fill_fragment(fs, 0.0f);
        #pragma unroll
        for (int kc = 0; kc < 8; kc++) {
            int kd = hf * 8 + kc;
            fragment<matrix_b, 16, 16, 16, half, col_major> fb;
            load_matrix_sync(fb, &gK16[kd * 16], 256);
            mma_sync(fs, QA[kc], fb, fs);
        }
        #pragma unroll
        for (int sl = 0; sl < 8; sl++) {
            int idx = lane + (sl << 5);
            gS32[(pair * 2 + hf) * 256 + idx] = fs.x[sl];
        }
        __syncthreads();
        #pragma unroll
        for (int sl = 0; sl < 8; sl++) {
            int idx = lane + (sl << 5);
            fs.x[sl] += gS32[(pair * 2 + (1 - hf)) * 256 + idx];
        }
        int krow = (pos0 + q0 + qr) * sstride;
        bool ok[8];
        #pragma unroll
        for (int sl = 0; sl < 8; sl++) {
            int r = hc + 2 * sl;
            ok[sl] = (r < nqw) && (cc < nk) && (mask[krow + r * sstride + kt + cc] != 0u);
        }
        float rmax[8];
        #pragma unroll
        for (int sl = 0; sl < 8; sl++) rmax[sl] = ok[sl] ? fs.x[sl] : -3.4028235e38f;
        #pragma unroll
        for (int off = 8; off > 0; off >>= 1)
            #pragma unroll
            for (int sl = 0; sl < 8; sl++)
                rmax[sl] = fmaxf(rmax[sl], __shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, rmax[sl], off));
        // 개별 e (P 용) 와 행 합 (s 용) 을 분리한다 — 행의 16개 키는 16개 레인에 있으므로
        // s 는 반드시 16레인 버터플라이로 합쳐야 한다(최대값만 줄이고 합을 안 줄이던 버그).
        float e_m[8], erow[8];
        #pragma unroll
        for (int sl = 0; sl < 8; sl++) {
            float mn = fmaxf(mrun[sl], rmax[sl]);
            e_m[sl] = __expf(mrun[sl] - mn);
            float e = ok[sl] ? __expf(fs.x[sl] - mn) : 0.0f;
            fs.x[sl] = e;         // P 행렬용 개별 확률
            erow[sl] = e;         // 행 합용 사본
            mrun[sl] = mn;
        }
        #pragma unroll
        for (int off = 8; off > 0; off >>= 1)
            #pragma unroll
            for (int sl = 0; sl < 8; sl++)
                erow[sl] += __shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, erow[sl], off);
        #pragma unroll
        for (int sl = 0; sl < 8; sl++) srun[sl] = srun[sl] * e_m[sl] + erow[sl];
        #pragma unroll
        for (int j = 0; j < 8; j++)
            #pragma unroll
            for (int sl = 0; sl < 8; sl++) fc[j].x[sl] *= e_m[sl];
        #pragma unroll
        for (int sl = 0; sl < 8; sl++) {
            int r = hc + 2 * sl;
            gP16[(pair * 2 + hf) * 256 + r * 16 + cc] = __float2half(fs.x[sl]);
        }
        __syncthreads();
        fragment<matrix_a, 16, 16, 16, half, row_major> fp;
        load_matrix_sync(fp, &gP16[(pair * 2 + hf) * 256], 16);
        #pragma unroll
        for (int dc = 0; dc < 8; dc++) {
            fragment<matrix_b, 16, 16, 16, half, row_major> fb;
            load_matrix_sync(fb, &gV16[(hf * 8 + dc) * 16], 256);
            mma_sync(fc[dc], fp, fb, fc[dc]);
        }
        __syncthreads();
    }
    int nseg2 = (n_past + seg - 1) / seg;
    #pragma unroll
    for (int j = 0; j < 8; j++)
        #pragma unroll
        for (int sl = 0; sl < 8; sl++) {
            int r = hc + 2 * sl;
            if (r < nqw) {
                float* pp2 = part + ((size_t)((q0 + qr + r) * n_head + h) * nseg2 + sg) * (hd + 2);
                pp2[(hf * 8 + j) * 16 + cc] = fc[j].x[sl];
            }
        }
    if (cc == 0) {
        #pragma unroll
        for (int sl = 0; sl < 8; sl++) {
            int r = hc + 2 * sl;
            if (r < nqw) {
                float* pp2 = part + ((size_t)((q0 + qr + r) * n_head + h) * nseg2 + sg) * (hd + 2);
                pp2[hd] = mrun[sl];
                pp2[hd + 1] = srun[sl];
            }
        }
    }
}

// 디코드(t=1) GQA 어텐션 v2 — 감축 구조 교체판.
// 기존 qsa_flash_gqa 는 스레드=dim(256스레드=256dim)이라 (키,헤드) 내적마다
// 5단 셔플 + 8워프 공유 왕복 + 블록 동기화 2회를 키 4개마다 치렀다.
// 여기서는 워프가 키 4개를 전담하고 워프 안에서 레인이 8 dim 씩 맡아 내적을
// 완성한다: 키당 셔플 5회(워프 내부), 크로스워프 왕복 없음, 타일(32키)당
// 동기화 2회. 인자·part 규약·그리드는 qsa_flash_gqa 와 동일하다.
extern "C" __global__ void qsa_flash_gqa2(const float* q, const float* ck, const float* cv,
                                          const unsigned* mask, float* part,
                                          int n_past, int n_head, int n_kv,
                                          int hd, int t_len, int sstride, int pos0, int seg) {
    int kvh = blockIdx.y;
    int sg = blockIdx.z;
    if (kvh >= n_kv || t_len != 1 || hd != 256) return;
    int gq = n_head / n_kv;
    if (gq < 1 || gq > 8) return;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;
    int h0 = kvh * gq;

    __shared__ float qf[8][256];
    __shared__ float sc[8][32];     // 타일 점수 → exp (헤드 x 키)
    __shared__ float pmx[8];        // 타일 최대 (헤드별)
    __shared__ float pe[8];         // 타일 exp 합 (헤드별)

    for (int i = tid; i < gq * 256; i += 256)
        qf[i >> 8][i & 255] = q[(h0 + (i >> 8)) * 2 * hd + (i & 255)];
    __syncthreads();
    // 레인 = dim [8*lane, 8*lane+8) 담당 → Q 레지스터 사본
    float qreg[8][8];
    #pragma unroll
    for (int r = 0; r < 8; r++)
        #pragma unroll
        for (int d = 0; d < 8; d++)
            qreg[r][d] = (r < gq) ? qf[r][lane * 8 + d] : 0.0f;

    float m[8], s[8], acc[8];
    #pragma unroll
    for (int r = 0; r < 8; r++) { m[r] = -3.4028235e38f; s[r] = 0.0f; acc[r] = 0.0f; }
    int p0 = sg * seg, p1 = min(p0 + seg, n_past);
    int mrow0 = pos0 * sstride;

    for (int pb = p0; pb < p1; pb += 32) {
        int np = min(32, p1 - pb);
        // ── QK: 워프 wid 가 키 [pb+4*wid, +4), 레인 lane 이 dim 8개
        // 발산 금지 — 셔플은 워프 전원이 참여해야 하므로 마스크는 값으로만 반영한다.
        float x[8][4];
        #pragma unroll
        for (int r = 0; r < 8; r++)
            #pragma unroll
            for (int j = 0; j < 4; j++) x[r][j] = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int kk = pb + wid * 4 + j;
            if (kk < p1 && (mask[mrow0 + kk] != 0u)) {
                const float4* kp = reinterpret_cast<const float4*>(
                        ck + ((size_t)kk * n_kv + kvh) * hd + lane * 8);
                float4 k0 = kp[0], k1 = kp[1];
                float kv[8] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w};
                #pragma unroll
                for (int r = 0; r < 8; r++) {
                    float a = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < 8; d++) a += qreg[r][d] * kv[d];
                    x[r][j] = (r < gq) ? a : 0.0f;
                }
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            #pragma unroll
            for (int r = 0; r < 8; r++)
                #pragma unroll
                for (int j = 0; j < 4; j++)
                    x[r][j] += __shfl_xor_sync(0xffffffffffffffffull, x[r][j], off);
        if (lane < gq * 4) {
            int r = lane >> 2, j = lane & 3;
            int kk = pb + wid * 4 + j;
            if (kk < p1)
                sc[r][wid * 4 + j] = (mask[mrow0 + kk] != 0u) ? x[r][j] : -3.4028235e38f;
        }
        __syncthreads();
        // ── 워프 0 이 타일 최대·exp·합을 전담 (타일당 동기화 2회로 끝낸다)
        // 레인 r(<gq) 이 헤드 r 을 전담한다 — 공유 배열을 다른 레인이 덮어쓰는 경합이 없다.
        if (wid == 0 && lane < gq) {
            int r = lane;
            float mx = -3.4028235e38f;
            for (int jj = 0; jj < np; jj++) mx = fmaxf(mx, sc[r][jj]);
            mx = fmaxf(mx, m[r]);
            pmx[r] = mx;
            float tmp[32];
            float sm = 0.0f;
            for (int jj = 0; jj < np; jj++) { tmp[jj] = __expf(sc[r][jj] - mx); sm += tmp[jj]; }
            for (int jj = 0; jj < np; jj++) sc[r][jj] = tmp[jj];
            pe[r] = sm;
        }
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            if (r < gq) {
                float e_m = __expf(m[r] - pmx[r]);
                s[r] = s[r] * e_m + pe[r];
                acc[r] *= e_m;
                m[r] = pmx[r];
            }
        }
        // ── PV: 스레드 = dim (감축 없음)
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            if (j < np) {
                float vv = cv[((size_t)(pb + j) * n_kv + kvh) * hd + tid];
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    if (r < gq) acc[r] += sc[r][j] * vv;
            }
        }
        __syncthreads();   // 다음 타일의 sc 기록 전에 PV 완료 보장
    }
    int nseg = (n_past + seg - 1) / seg;
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        if (r < gq) {
            float* pp2 = part + ((size_t)(h0 + r) * nseg + sg) * (hd + 2);
            pp2[tid] = acc[r];
            if (tid == 0) { pp2[hd] = m[r]; pp2[hd + 1] = s[r]; }
        }
    }
}

// KV f32→f16 변환 (요소별, 벡터화). 디코드 어텐션이 f16 KV를 읽을 수 있는지 실측하기 위한 것.
extern "C" __global__ void kv_f16(const float* src, half* dst, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < n) {
        float4 v = *reinterpret_cast<const float4*>(src + i);
        half* d = dst + i;
        d[0] = __float2half(v.x); d[1] = __float2half(v.y);
        d[2] = __float2half(v.z); d[3] = __float2half(v.w);
    }
}

// gqa2 의 f16 KV 판(실험): K 로드가 16바이트 1회로 줄어든다(32바이트 2회 → 16바이트 1회).
extern "C" __global__ void qsa_flash_gqa2h(const float* q, const half* ck, const half* cv,
                                           const unsigned* mask, float* part,
                                           int n_past, int n_head, int n_kv,
                                           int hd, int t_len, int sstride, int pos0, int seg) {
    int kvh = blockIdx.y;
    int sg = blockIdx.z;
    if (kvh >= n_kv || t_len != 1 || hd != 256) return;
    int gq = n_head / n_kv;
    if (gq < 1 || gq > 8) return;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;
    int h0 = kvh * gq;

    __shared__ float qf[8][256];
    __shared__ float sc[8][32];
    __shared__ float pmx[8];
    __shared__ float pe[8];

    for (int i = tid; i < gq * 256; i += 256)
        qf[i >> 8][i & 255] = q[(h0 + (i >> 8)) * 2 * hd + (i & 255)];
    __syncthreads();
    float qreg[8][8];
    #pragma unroll
    for (int r = 0; r < 8; r++)
        #pragma unroll
        for (int d = 0; d < 8; d++)
            qreg[r][d] = (r < gq) ? qf[r][lane * 8 + d] : 0.0f;

    float m[8], s[8], acc[8];
    #pragma unroll
    for (int r = 0; r < 8; r++) { m[r] = -3.4028235e38f; s[r] = 0.0f; acc[r] = 0.0f; }
    int p0 = sg * seg, p1 = min(p0 + seg, n_past);
    int mrow0 = pos0 * sstride;

    for (int pb = p0; pb < p1; pb += 32) {
        int np = min(32, p1 - pb);
        float x[8][4];
        #pragma unroll
        for (int r = 0; r < 8; r++)
            #pragma unroll
            for (int j = 0; j < 4; j++) x[r][j] = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int kk = pb + wid * 4 + j;
            if (kk < p1 && (mask[mrow0 + kk] != 0u)) {
                const half* kp = ck + ((size_t)kk * n_kv + kvh) * hd + lane * 8;
                const uint4 raw = *reinterpret_cast<const uint4*>(kp);
                const half* hs = reinterpret_cast<const half*>(&raw);
                float kv[8];
                #pragma unroll
                for (int d = 0; d < 8; d++) kv[d] = __half2float(hs[d]);
                #pragma unroll
                for (int r = 0; r < 8; r++) {
                    float a = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < 8; d++) a += qreg[r][d] * kv[d];
                    x[r][j] = (r < gq) ? a : 0.0f;
                }
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            #pragma unroll
            for (int r = 0; r < 8; r++)
                #pragma unroll
                for (int j = 0; j < 4; j++)
                    x[r][j] += __shfl_xor_sync(0xffffffffffffffffull, x[r][j], off);
        if (lane < gq * 4) {
            int r = lane >> 2, j = lane & 3;
            int kk = pb + wid * 4 + j;
            if (kk < p1)
                sc[r][wid * 4 + j] = (mask[mrow0 + kk] != 0u) ? x[r][j] : -3.4028235e38f;
        }
        __syncthreads();
        if (wid == 0 && lane < gq) {
            int r = lane;
            float mx = -3.4028235e38f;
            for (int jj = 0; jj < np; jj++) mx = fmaxf(mx, sc[r][jj]);
            mx = fmaxf(mx, m[r]);
            pmx[r] = mx;
            float tmp[32];
            float sm = 0.0f;
            for (int jj = 0; jj < np; jj++) { tmp[jj] = __expf(sc[r][jj] - mx); sm += tmp[jj]; }
            for (int jj = 0; jj < np; jj++) sc[r][jj] = tmp[jj];
            pe[r] = sm;
        }
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            if (r < gq) {
                float e_m = __expf(m[r] - pmx[r]);
                s[r] = s[r] * e_m + pe[r];
                acc[r] *= e_m;
                m[r] = pmx[r];
            }
        }
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            if (j < np) {
                float vv = __half2float(cv[((size_t)(pb + j) * n_kv + kvh) * hd + tid]);
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    if (r < gq) acc[r] += sc[r][j] * vv;
            }
        }
        __syncthreads();
    }
    int nseg = (n_past + seg - 1) / seg;
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        if (r < gq) {
            float* pp2 = part + ((size_t)(h0 + r) * nseg + sg) * (hd + 2);
            pp2[tid] = acc[r];
            if (tid == 0) { pp2[hd] = m[r]; pp2[hd + 1] = s[r]; }
        }
    }
}

// f16 KV + v_dot2_f32_f16 디코드 어텐션(프로토타입) — llama 의 flash_attn_tile 구조.
// 레인=키 하나, 256차원 내적을 그 레인에서 128회 v_dot2_f32_f16 으로 직렬 계산한다.
// QK 에 셔플이 전혀 없다(현 gqa2 는 (키,헤드)마다 5단 셔플). PV 는 종전과 같이 스레드=dim.
// K 는 공유에 행당 258 half(2 half 패딩 = 홀수 word 스트라이드)로 적재해 LDS 뱅크 충돌을 없앤다.
extern "C" __global__ void qsa_flash_gqa2d(const float* q, const half* ck, const half* cv,
                                           const unsigned* mask, float* part,
                                           int n_past, int n_head, int n_kv,
                                           int hd, int t_len, int sstride, int pos0, int seg) {
    int kvh = blockIdx.y;
    int sg = blockIdx.z;
    if (kvh >= n_kv || t_len != 1 || hd != 256) return;
    int gq = n_head / n_kv;
    if (gq < 1 || gq > 8) return;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;
    int h0 = kvh * gq;

    constexpr int KS = 258;                 // 행 스트라이드(half). 256+2 → word 129 = 홀수
    __shared__ half q16[8][256];
    __shared__ half kv16[32 * KS];          // K 타일 → 그 위에 V 를 덮어쓴다
    __shared__ float pf[8][32];             // 확률(스레드=dim 단계로 넘김)
    __shared__ float stnew[8];              // 타일별 새 러닝 최대(전 스레드 공유)
    __shared__ float smfin[8], sfin[8];     // 최종 m/s (워프 레인0 이 기록)

    for (int i = tid; i < gq * 256; i += 256)
        q16[i >> 8][i & 255] = __float2half(q[(h0 + (i >> 8)) * 2 * hd + (i & 255)]);
    __syncthreads();

    float m[8], acc[8];
    #pragma unroll
    for (int r = 0; r < 8; r++) { m[r] = -3.4028235e38f; acc[r] = 0.0f; }
    float mrun[8], srun[8];
    #pragma unroll
    for (int r = 0; r < 8; r++) { mrun[r] = -3.4028235e38f; srun[r] = 0.0f; }

    int p0 = sg * seg, p1 = min(p0 + seg, n_past);
    int mrow0 = pos0 * sstride;

    for (int pb = p0; pb < p1; pb += 32) {
        int np = min(32, p1 - pb);
        // K 타일 적재(전역 f16 → 공유, d 방향 연속 = 코얼레스)
        // 16바이트 로드(uint4 = half 8개) — 2바이트 스칼라 로드 32회를 4회로 줄인다
        for (int i = tid * 8; i < 32 * 256; i += 256 * 8) {
            int row = i >> 8, d = i & 255;
            half* dst = &kv16[row * KS + d];
            if (row < np) {
                const uint4 v = *reinterpret_cast<const uint4*>(
                        ck + ((size_t)(pb + row) * n_kv + kvh) * hd + d);
                const half* hs = reinterpret_cast<const half*>(&v);
                #pragma unroll
                for (int j = 0; j < 8; j++) dst[j] = hs[j];
            } else {
                #pragma unroll
                for (int j = 0; j < 8; j++) dst[j] = __float2half(0.0f);
            }
        }
        __syncthreads();
        // QK: 워프 r = 헤드 r, 레인 = 키 l. 내적 전체가 레인 안에서 끝난다.
        if (wid < gq) {
            float a = 0.0f;
            const half2* q2 = reinterpret_cast<const half2*>(&q16[wid][0]);
            const half2* k2 = reinterpret_cast<const half2*>(&kv16[lane * KS]);
            #pragma unroll 8
            for (int d2 = 0; d2 < 128; d2++) {
                half2 qv = q2[d2], kvv = k2[d2];
                asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(a) : "v"(qv), "v"(kvv));
            }
            // 마스크/범위: 유효하지 않으면 -inf
            int kk = pb + lane;
            a = ((lane < np) && (mask[mrow0 + kk] != 0u)) ? a : -3.4028235e38f;
            // 워프 max (5단) → 새 러닝 최대
            // 러닝 최대/합은 워프 레지스터 사본으로만 갱신한다 — 공유를 읽고 쓰면
            // 레인 간 경합으로 e_m 이 1 로 붕괴한다.
            float mx = a;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffffffffffull, mx, off));
            float m_old = mrun[wid];
            float mn = fmaxf(m_old, mx);
            float e = __expf(a - mn);
            pf[wid][lane] = e;
            float es = e;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                es += __shfl_xor_sync(0xffffffffffffffffull, es, off);
            if (lane == 0) { srun[wid] = srun[wid] * __expf(m_old - mn) + es; mrun[wid] = mn; }
            stnew[wid] = mn;
        }
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            if (r < gq) { acc[r] *= __expf(m[r] - stnew[r]); m[r] = stnew[r]; }
        }
        __syncthreads();
        // V 타일 적재(K 위에 덮어씀) + PV(스레드 = dim)
        for (int i = tid * 8; i < 32 * 256; i += 256 * 8) {
            int row = i >> 8, d = i & 255;
            half* dst = &kv16[row * KS + d];
            if (row < np) {
                const uint4 v = *reinterpret_cast<const uint4*>(
                        cv + ((size_t)(pb + row) * n_kv + kvh) * hd + d);
                const half* hs = reinterpret_cast<const half*>(&v);
                #pragma unroll
                for (int j = 0; j < 8; j++) dst[j] = hs[j];
            } else {
                #pragma unroll
                for (int j = 0; j < 8; j++) dst[j] = __float2half(0.0f);
            }
        }
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            if (r < gq) {
                float ar = acc[r];
                for (int kk = 0; kk < np; kk++)
                    ar += pf[r][kk] * __half2float(kv16[kk * KS + tid]);
                acc[r] = ar;
            }
        }
        __syncthreads();
    }
    if (wid < gq && lane == 0) { smfin[wid] = mrun[wid]; sfin[wid] = srun[wid]; }
    __syncthreads();
    int nseg = (n_past + seg - 1) / seg;
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        if (r < gq) {
            float* pp2 = part + ((size_t)(h0 + r) * nseg + sg) * (hd + 2);
            pp2[tid] = acc[r];
            if (tid == 0) { pp2[hd] = smfin[r]; pp2[hd + 1] = sfin[r]; }
        }
    }
}

// ─── plans/73: QSA indexer selection on device (decode t=1) ───
// Mirrors stages/qsa.rs qsa_select (pass A iq norm+rope, block-key pooling,
// pass B scores + top-k) and qsa_sel_list — same arithmetic order so the
// selection lists are bit-identical to the host path (probed end to end).
// This removes the per-QSA-layer d2h/h2d round trip that idled the GPU for
// ~3.3ms per layer at 16k context (KTRACE: "after qk_norm_rope" 40ms/step).

// (1) indexer q norm+rope — one warp per (head, token). The host norm is
// rms_norm (ops.rs): 32 f32 segment partials combined as f64 in order, then
// (v*scale)*w; rope_head applies f64 rotation with the f32 cos/sin table.
// Identical structure to qk_norm_rope but weights are shared per head (not
// tiled) and n_rot == idx_dim.
extern "C" __global__ void q4_idx_q_rope(
        const float* __restrict__ iq, float* __restrict__ iqr,
        const float* __restrict__ iqw, const float* __restrict__ cs,
        float eps, int pos0, int t, int idx_heads, int idx_dim) {
    int h = blockIdx.x;
    int y = blockIdx.y;
    int u = threadIdx.x;
    if (h >= idx_heads || y >= t || u >= 32) return;
    const size_t off = (size_t)y * (idx_heads * idx_dim) + (size_t)h * idx_dim;
    const float* src = iq + off;
    float* dst = iqr + off;
    const int chunk = (idx_dim + 31) >> 5;
    const int lo = u * chunk, hi = min(lo + chunk, idx_dim);
    float mypart = 0.0f;
    for (int i = lo; i < hi; i++) { float dv = src[i]; mypart += dv * dv; }
    float parts[32];
    #pragma unroll
    for (int uu = 0; uu < 32; uu++)
        parts[uu] = __shfl_sync(0xffffffffffffffffull, mypart, uu);
    double sum = 0.0;
    #pragma unroll
    for (int uu = 0; uu < 32; uu++) sum += parts[uu];
    const float scale = 1.0f / (float)sqrt(sum / (double)idx_dim + (double)eps);
    for (int i = u; i < idx_dim; i += 32) dst[i] = src[i] * scale * iqw[i];
    __syncwarp();
    const int half = idx_dim >> 1;
    const int csbase = (pos0 + y) * half * 2;
    for (int p = u; p < half; p += 32) {
        double c = (double)cs[csbase + p * 2];
        double sf = (double)cs[csbase + p * 2 + 1];
        double x0 = (double)dst[p], x1 = (double)dst[p + half];
        dst[p] = (float)(x0 * c - x1 * sf);
        dst[p + half] = (float)(x0 * sf + x1 * c);
    }
}

// (2) block-key update — one warp per completed block. Host: mean-pool r
// indexer-k rows (f32 adds, j ascending), /= r, rms_norm(ikw), rope(pos=b*r).
// Requires idx_dim == 128 (lane u owns elements 4u..4u+3; rope pairs sit 16
// lanes apart, exchanged with shfl_xor(16)).
extern "C" __global__ void q4_idx_bk_update(
        const float* __restrict__ idxk, float* __restrict__ bk,
        const float* __restrict__ ikw, const float* __restrict__ cs,
        float eps, int b0, int r, int idx_dim) {
    const int b = b0 + (int)blockIdx.x;
    const int u = threadIdx.x;
    if (u >= 32 || idx_dim != 128) return;
    float pv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int j = 0; j < r; j++) {
        const float* row = idxk + (size_t)(b * r + j) * idx_dim;
        #pragma unroll
        for (int k = 0; k < 4; k++) pv[k] += row[u * 4 + k];
    }
    #pragma unroll
    for (int k = 0; k < 4; k++) pv[k] /= (float)r;
    float mypart = 0.0f;
    #pragma unroll
    for (int k = 0; k < 4; k++) mypart += pv[k] * pv[k];
    float parts[32];
    #pragma unroll
    for (int uu = 0; uu < 32; uu++)
        parts[uu] = __shfl_sync(0xffffffffffffffffull, mypart, uu);
    double sum = 0.0;
    #pragma unroll
    for (int uu = 0; uu < 32; uu++) sum += parts[uu];
    const float scale = 1.0f / (float)sqrt(sum / (double)idx_dim + (double)eps);
    float nm[4];
    #pragma unroll
    for (int k = 0; k < 4; k++) nm[k] = pv[k] * scale * ikw[u * 4 + k];
    __syncwarp();
    const int half = idx_dim >> 1;
    const int pos = b * r;
    const int csbase = pos * half * 2;
    float* out = bk + (size_t)b * idx_dim;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        const int i = u * 4 + k;
        const float other = __shfl_xor_sync(0xffffffffffffffffull, nm[k], 16);
        const int p = (i < half) ? i : (i - half);
        double c = (double)cs[csbase + p * 2];
        double sf = (double)cs[csbase + p * 2 + 1];
        if (i < half) {
            double x0 = (double)nm[k], x1 = (double)other;
            out[i] = (float)(x0 * c - x1 * sf);
        } else {
            double x0 = (double)other, x1 = (double)nm[k];
            out[i] = (float)(x0 * sf + x1 * c);
        }
    }
}

// (3) block scores — one thread per block. Host pattern: per head a
// 4-accumulator unrolled dot (d0+d1)+(d2+d3), relu-added into the score.
extern "C" __global__ void q4_idx_score(
        const float* __restrict__ iqr, const float* __restrict__ bk,
        float* __restrict__ scores, int n_blocks, int idx_heads, int idx_dim) {
    const int b = (int)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const float* pk = bk + (size_t)b * idx_dim;
    float score = 0.0f;
    for (int h = 0; h < idx_heads; h++) {
        const float* qh = iqr + (size_t)h * idx_dim;   // decode: t=1
        float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
        int i2 = 0;
        for (; i2 + 4 <= idx_dim; i2 += 4) {
            d0 += qh[i2] * pk[i2];
            d1 += qh[i2 + 1] * pk[i2 + 1];
            d2 += qh[i2 + 2] * pk[i2 + 2];
            d3 += qh[i2 + 3] * pk[i2 + 3];
        }
        for (; i2 < idx_dim; i2++) d0 += qh[i2] * pk[i2];
        const float dot = (d0 + d1) + (d2 + d3);
        if (dot > 0.0f) score += dot;
    }
    scores[b] = score;
}

// (4) deterministic top-k rank — one thread per block. rank(b) counts
// blocks with higher score, ties broken by lower index, so "selected" is
// exactly the top-n_sel set under (score desc, index asc). The host's
// select_nth_unstable_by resolves threshold ties arbitrarily; the probe
// (LLM170_QSA_SELCHECK) compares the resulting lists over real runs.
extern "C" __global__ void q4_idx_rank(
        const float* __restrict__ scores, unsigned* __restrict__ selflag,
        int n_blocks, int n_sel) {
    const int b = (int)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const float sb = scores[b];
    int rank = 0;
    for (int b2 = 0; b2 < n_blocks; b2++) {
        const float s2 = scores[b2];
        if (s2 > sb || (s2 == sb && b2 < b)) rank++;
    }
    selflag[b] = (rank < n_sel) ? 1u : 0u;
}

// (5) selection-list expand — single block. Prefix (serial, ~n_blocks) then
// parallel emit of block positions (ascending) and the tail, mirroring
// qsa_sel_list. sel_off = {0, n_sel*r + tail_cnt}.
extern "C" __global__ void q4_idx_expand(
        const unsigned* __restrict__ selflag, unsigned* __restrict__ sel_idx,
        unsigned* __restrict__ sel_off, int n_blocks, int n_sel,
        int r, int n_past) {
    __shared__ unsigned slot[8192];
    const int tid = threadIdx.x;
    if (tid == 0) {
        unsigned k2 = 0;
        for (int b = 0; b < n_blocks; b++) {
            slot[b] = k2;
            if (selflag[b]) k2++;
        }
    }
    __syncthreads();
    const int tail_start = (n_past / r) * r;
    const int tail_cnt = n_past - tail_start;
    for (int b = tid; b < n_blocks; b += blockDim.x) {
        if (selflag[b]) {
            const unsigned o = slot[b] * (unsigned)r;
            for (int j = 0; j < r; j++) sel_idx[o + j] = (unsigned)(b * r + j);
        }
    }
    for (int j = tid; j < tail_cnt; j += blockDim.x)
        sel_idx[(unsigned)n_sel * (unsigned)r + (unsigned)j] = (unsigned)(tail_start + j);
    if (tid == 0) {
        sel_off[0] = 0u;
        sel_off[1] = (unsigned)(n_sel * r + tail_cnt);
    }
}

// (4') plans/73 개선: top-k + 목록 전개를 **단일 블록 비토닉 정렬**로 수행.
// 종전 rank(O(B²) 스레드당 16k 비교) + expand(직렬 prefix) 콤보는 0.228ms/콜
// (16k 문맥 × 12층 = 2.7ms/step)이었다. (score,idx)를 u64로 팩해 내림차순
// 정렬하면 상위 n_sel이 정확히 "점수 내림, 인덱스 오름차순" 집합(= rank 판정과
// 동일)이고, 인덱스만 다시 오름차순 정렬해 전개한다. n_blocks ≤ 4096 전용
// (공유메모리 34KB) — 넘으면 구형 rank+expand로 폴백.
extern "C" __global__ void q4_idx_topk(const float* __restrict__ scores,
                                       unsigned* __restrict__ sel_idx,
                                       unsigned* __restrict__ sel_off,
                                       int n_blocks, int n_sel, int r, int n_past) {
    __shared__ unsigned long long sh[4096];
    __shared__ unsigned sel_ids[4096];
    const int tid = threadIdx.x;
    // f32 → 순서 보존 u32 매핑(음수 비트 반전, 양수 최상위 비트 세트).
    for (int b = tid; b < n_blocks; b += 256) {
        unsigned u = __float_as_uint(scores[b]);
        u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
        sh[b] = ((unsigned long long)u << 32) | (unsigned)b;
    }
    for (int b = n_blocks + tid; b < 4096; b += 256) sh[b] = 0ull;
    __syncthreads();
    // 비토닉 내림차순(4096 고정 — 패딩은 최솟값 0).
    for (int k = 2; k <= 4096; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < 4096; i += 256) {
                const int l = i ^ j;
                if (l > i) {
                    const unsigned long long a = sh[i], b2 = sh[l];
                    const bool sw = ((i & k) == 0) ? (a < b2) : (a > b2);
                    if (sw) { sh[i] = b2; sh[l] = a; }
                }
            }
            __syncthreads();
        }
    }
    // 상위 n_sel 인덱스 추출 → 오름차순 정렬(np = 다음 2의 거듭제곱, 패딩 최대값).
    for (int k2 = tid; k2 < n_sel; k2 += 256)
        sel_ids[k2] = (unsigned)(sh[k2] & 0xFFFFFFFFull);
    int np = 1;
    while (np < n_sel) np <<= 1;
    for (int k2 = n_sel + tid; k2 < np; k2 += 256) sel_ids[k2] = 0xFFFFFFFFu;
    __syncthreads();
    for (int k = 2; k <= np; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < np; i += 256) {
                const int l = i ^ j;
                if (l > i) {
                    const unsigned a = sel_ids[i], b2 = sel_ids[l];
                    const bool sw = ((i & k) == 0) ? (a > b2) : (a < b2);
                    if (sw) { sel_ids[i] = b2; sel_ids[l] = a; }
                }
            }
            __syncthreads();
        }
    }
    // 전개: 선택 블록(오름차순) 위치 + 테일 — qsa_sel_list와 동일 구성.
    const int tail_start = (n_past / r) * r;
    const int tail_cnt = n_past - tail_start;
    for (int k2 = tid; k2 < n_sel; k2 += 256) {
        const unsigned b3 = sel_ids[k2];
        const unsigned o = (unsigned)k2 * (unsigned)r;
        for (int j2 = 0; j2 < r; j2++) sel_idx[o + j2] = b3 * (unsigned)r + (unsigned)j2;
    }
    for (int j2 = tid; j2 < tail_cnt; j2 += 256)
        sel_idx[(unsigned)n_sel * (unsigned)r + (unsigned)j2] = (unsigned)(tail_start + j2);
    if (tid == 0) { sel_off[0] = 0u; sel_off[1] = (unsigned)(n_sel * r + tail_cnt); }
}
