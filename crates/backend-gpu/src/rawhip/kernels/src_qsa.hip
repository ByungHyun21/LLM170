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
extern "C" __global__ void qsa_flash(const float* q, const float* ck, const float* cv,
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
        float x = active ? qf[tid] * ck[kb + tid] : 0.0f;
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
        float vv = active ? cv[kb + tid] : 0.0f;
        acc = acc * e_m + e_d * vv;
        m = m_new;
    }
    if (!active) return;
    float g = 1.0f / (1.0f + exp_cr(-q[qb + hd + tid]));
    out[t * n_head * hd + h * hd + tid] = acc * (1.0f / s) * g;
}
// QSA split flash, q 2행 다중화 — ck/cv 1회 로드로 t쌍 공유 (K/V 트래픽 ½).
extern "C" __global__ void qsa_flash_split4q4(const float* q, const float* ck, const float* cv,
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
                    float kv = active ? ck[kb + tid] : 0.0f;
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
                float vv = active ? cv[kb + tid] : 0.0f;
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
extern "C" __global__ void qsa_flash_wk(const float* q, const float* ck, const float* cv,
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
            kv[j] = (j < dpl) ? ck[(size_t)k * n_kv * hd + kb0 + lane * dpl + j] : 0.0f;
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
            vv[j] = (j < dpl) ? cv[(size_t)k * n_kv * hd + kb0 + lane * dpl + j] : 0.0f;
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
    float M = -3.4028235e38f;
    for (int i = 0; i < nseg; i++) {
        float mi = base[i * (hd + 2) + hd];
        if (mi > M) M = mi;
    }
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
                    float kv = active ? ck[kb + tid] : 0.0f;
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
                float vv = active ? cv[kb + tid] : 0.0f;
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
extern "C" __global__ void qsa_flash_wk16(const float* q, const float* ck, const float* cv,
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
        const float* krow = ck + (size_t)k * n_kv * hd + kb0 + own * 16;
        float kv[16];
        #pragma unroll
        for (int j = 0; j < 16; j++) kv[j] = krow[j];
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
            const float* vrow = cv + (size_t)k * n_kv * hd + kb0 + own * 16;
            #pragma unroll
            for (int j = 0; j < 16; j++) acc[j] = acc[j] * e_m + e_d * vrow[j];
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
