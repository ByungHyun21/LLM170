// ── Flash-Next PLE(n-gram 해시 임베딩) CUDA 커널 (plans/124 FNB, 2026-10-05) ──
//
// [빌드 계약] build_cuda.bat 별도 -fmad=false 블록(gdn/attn/ew/q4/fn 계열과
// 동일): gate/conv/resid의 곱-가산 쌍이 코어 미러(core는 f32 mul+add 2중
// 반올림)와 비트동일하려면 FMA 수축이 금지된다. exp 트윈(fn_ple_expf)의
// fma() 호출은 명시적 intrinsic이라 -fmad=false와 무관하게 항상 융합(단일
// 반올림) — core ops.rs exp_cr의 mul_add와 1:1.
//
// [산술 계약 — 수치 진실은 crates/core CPU 참조(plans/124 §6)]
// - IQ4_NL 행 디양자화: quant/deq.rs deq_iq4_nl L241-247 — y[j]=d·KV[qs[j]&0xF],
//   y[16+j]=d·KV[qs[j]>>4], d=f16(blk,0)(deq.rs f16 L6-8→half_to_f32 L11-38).
//   KV 표 = tables.rs KVALUES_IQ4NL L73-75 직이식(리터럴 정리 금지 — quant
//   AGENTS 비트 계약). 곱 순서 d·kv 유지.
// - 게이트: stages/ple.rs L119-161 — grouped_rms(stages/hc.rs L14-21 → ops.rs
//   rms_norm L33-37·sq_sum L11-31: 32세그먼트 f32 부분합→f64 순차 결합,
//   scale=1/((Σ/n+eps).sqrt() as f32) — sqrt는 f64) → dot=Σ k_n·q_n 순차 f32
//   → dot/=√n_embd → mag=√max(|dot|,1e-6) → sigmoid(sgn·mag) → value 방송×
//   게이트 → grouped rms(n_conv).
// - dilated depthwise conv: stages/ple.rs L169-184 — start=hist+ti-(kern-1-k)*dil
//   탭 순서 그대로, 출력 silu. 상태 tail 갱신 L189-193(padded[t+j]).
// - 잔차: stages/ple.rs L217-235 — row += value·g + conv_out(곱→가산→가산
//   순서 그대로; 게이트 재사용 plans/90 B4 D6).
// - sigmoid/silu: ops.rs L128-137 — 1/(1+exp(−x)), x/(1+exp(−x)).
// - exp: ops.rs exp_cr L52-91의 장치 트윈(f64 FMA 호너 15차 + 포화 가드,
//   리터럴까지 동일). G5 gdn_exp_d와는 **다른 다항식** — 본 커널의 참조는
//   core 값 경로(ops.rs)이므로 exp_cr을 베낀다(트윈 재작성 금지 — 원장 17호).
//
// [호스트/디바이스 분할 결정 — FNB]
// - HOST(모듈층 ple_cuda.rs): n-gram 해시(ple.rs ple_hash_rows L276-335 — u64
//   wrap mul/xor/mod, GPU u64 나머지 경제성 없음 + 값 경로와 동일 계약) ·
//   GGUF 테이블 행 pread 스테이징(37GiB급 테이블 전량 상주 금지 — AGMENTS
//   plans/86 §6 계약, FnGguf.read_rows) · key/value 투영(REUSE — G2/G8
//   gemv/gemm 영역, 리드가 병합 시 배선).
// - DEVICE(본 파일): IQ4_NL 게더 디양자화·게이트·방송 norm·conv·잔차 —
//   PLE 고유 산술(코싱 블록 수학)만 커널화.
//
// [sm_80/GA100 근거 — 개발기(RTX 4070 SUPER) 타이밍 판단 금지, plans/124 §0]
// - gather: 스테이징이 pread 지연 지배(행당 90B — HBM2e 대역폭 미포화,
//   그리드 t·16행으로 극소). 스레드=출력 원소(쓰기 병합 우선).
// - gate: 스레드당 순차 환원(Σ 2560 f32 dot·3× 노름)은 core 비트 계약의
//   대가 — 세그먼트 병렬 환원은 결합 순서를 바꿔 비트가 어긋난다(sq_sum
//   L11-31 주석 참조). 점유 희생을 문서화하고 GA100 실측은 도착 후.
// - conv/state/resid: 원소별 스트리밍(HBM2e 대역폭 지향, 레지스터 경량).
// - 상태 갱신은 더블버퍼(st_src→st_dst) — 제자리 갱신은 t<hist에서 판독·
//   기록 경합(원본 core는 호스트 padded 복사로 회피 — L186-193).
//
// [커널→프루브 매핑(plans/129-cuda C5)] exl3_fn_ple.cu 5커널 ↔
// ple_cuda_probe.rs `ple`/`ple-neg` 프루브(ple: 전 커널 값 판정,
// ple-neg: 해시 계수·게더 인덱스 오염 탐지).

// ── 코어 exp_cr 트윈(ops.rs L52-91 직베낌 — 리터럴·연산 순서 동일) ──
__device__ __forceinline__ float fn_ple_expf(float x) {
    double xd = (double)x;
    if (xd > 88.72) {
        return __int_as_float(0x7f800000); // +inf
    }
    if (xd < -103.97) {
        return 0.0f;
    }
    const double LN2_HI = 6.931471803691238e-01;
    const double LN2_LO = 1.9082149292705877e-10;
    const double INV_LN2 = 1.4426950408889634; // std LOG2_E와 동일 비트
    double kd = rint(xd * INV_LN2); // round_ties_even 대응
    long long k = (long long)kd;
    double r = fma(-kd, LN2_HI, xd);
    r = fma(-kd, LN2_LO, r);
    double p = 1.0 / 1307674368000.0;
    p = fma(p, r, 1.0 / 479001600.0);
    p = fma(p, r, 1.0 / 39916800.0);
    p = fma(p, r, 1.0 / 3628800.0);
    p = fma(p, r, 1.0 / 362880.0);
    p = fma(p, r, 1.0 / 40320.0);
    p = fma(p, r, 1.0 / 5040.0);
    p = fma(p, r, 1.0 / 720.0);
    p = fma(p, r, 1.0 / 120.0);
    p = fma(p, r, 1.0 / 24.0);
    p = fma(p, r, 1.0 / 6.0);
    p = fma(p, r, 0.5);
    p = fma(p, r, 1.0);
    p = fma(p, r, 1.0);
    if (k > 127) {
        return __int_as_float(0x7f800000);
    }
    double scale = __longlong_as_double((unsigned long long)(k + 1023) << 52);
    return (float)(p * scale);
}

// sigmoid/silu — ops.rs L128-137 공식(나눗셈·덧셈 f32 그대로).
__device__ __forceinline__ float fn_ple_sigmoid(float x) {
    return 1.0f / (1.0f + fn_ple_expf(-x));
}
__device__ __forceinline__ float fn_ple_silu(float x) {
    return x / (1.0f + fn_ple_expf(-x));
}

// f16→f32 비트 변환 — quant/deq.rs half_to_f32 L11-38 직베낌(서브노멀
// 정규화 루프 포함; f16→f32는 IEEE 유일 변환이라 값 동일 보장).
__device__ __forceinline__ float fn_ple_f16(unsigned short h) {
    unsigned int sign = (h >> 15) & 1u;
    unsigned int e = (h >> 10) & 0x1fu;
    unsigned int f = h & 0x3ffu;
    unsigned int bits;
    if (e == 0u) {
        if (f == 0u) {
            bits = sign << 31;
        } else {
            unsigned int ee = 127u - 15u + 1u;
            unsigned int ff = f;
            while ((ff & 0x400u) == 0u) {
                ff <<= 1;
                ee -= 1u;
            }
            ff &= 0x3ffu;
            bits = (sign << 31) | (ee << 23) | (ff << 13);
        }
    } else if (e == 0x1fu) {
        bits = (sign << 31) | (0xffu << 23) | (f << 13);
    } else {
        bits = (sign << 31) | ((e + 112u) << 23) | (f << 13);
    }
    return __uint_as_float(bits);
}

// sq_sum 스케일 미러(ops.rs sq_sum L11-31 + rms_norm L33-37의 scale식):
// 32세그먼트 f32 부분합 → f64 순차 결합 → f64 sqrt → f32 캐스트 → f32 역수.
__device__ __forceinline__ float fn_ple_rms_scale(const float* x, int n, float eps) {
    int chunk = (n + 31) / 32;
    double sum = 0.0;
    for (int u = 0; u < 32; u++) {
        int lo = u * chunk;
        if (lo >= n) {
            break;
        }
        int hi = (lo + chunk < n) ? (lo + chunk) : n;
        float part = 0.0f;
        for (int i = lo; i < hi; i++) {
            part += x[i] * x[i];
        }
        sum += (double)part;
    }
    float fs = (float)sqrt(sum / (double)n + (double)eps);
    return 1.0f / fs;
}

// ── (1) IQ4_NL 테이블 행 게더 디양자화 ──
// 원본 산술: qwen4exp/mod.rs ple_gather_parts L471-505(블록 순회) →
// deq.rs deq_iq4_nl L241-247. 스레드=출력 원소(emb[idx]).
//   idx = row·160 + e, 블록 b=e/32, 블록 내 w=e%32, 바이트 j=w&15,
//   니블 = w<16 ? qs[j]&0xF : qs[j]>>4 — 원본 y[j]/y[16+j] 전개와 동일.
extern "C" __global__ void llm170_ple_iq4nl_gather(const unsigned char* raw,
                                                   float* emb, int n_elems) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elems) {
        return;
    }
    int row = idx / 160;
    int e = idx - row * 160;
    int b = e >> 5;
    int w = e & 31;
    int base = row * 90 + b * 18;
    float d = fn_ple_f16((unsigned short)(raw[base] | (raw[base + 1] << 8)));
    // KVALUES_IQ4NL — tables.rs L73-75 직이식(i8 리터럴 정리 금지).
    const float KV[16] = {-127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f,
                          -22.0f,  -10.0f,  1.0f,   13.0f,  25.0f,  38.0f,
                          53.0f,   69.0f,  89.0f,  113.0f};
    unsigned char q = raw[base + 2 + (w & 15)];
    float kv = KV[(w < 16) ? (q & 0xF) : (q >> 4)];
    emb[idx] = d * kv; // 곱 순서 d·kv — deq_iq4_nl 그대로
}

// ── (2) 게이트 + value 방송 + grouped norm ──
// 스레드=(토큰 ti, 스트림 s). core stages/ple.rs L119-161 산술 순서 그대로
// (k_n/q_n 원소 재계산은 rms_norm 원소식 (v·scale)·g와 비트동일 — 저장
// 벡터 없이 인라인). key/res_hc는 [t][hc·n], value는 [t][n].
extern "C" __global__ void llm170_ple_gate(const float* key, const float* qy,
                                           const float* nk, const float* nq,
                                           const float* value, const float* nc,
                                           float* gates, float* gated, int t,
                                           int hc, int n, float eps) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= t * hc) {
        return;
    }
    int ti = id / hc;
    int s = id - ti * hc;
    const float* krow = key + ((long long)ti * hc + s) * n;
    const float* qrow = qy + ((long long)ti * hc + s) * n;
    const float* nk_s = nk + (long long)s * n;
    const float* nq_s = nq + (long long)s * n;
    float kscale = fn_ple_rms_scale(krow, n, eps);
    float qscale = fn_ple_rms_scale(qrow, n, eps);
    // per-stream s = Σ key·query / √n_embd (ple.rs L143-147) — 순차 f32.
    float dot = 0.0f;
    for (int i = 0; i < n; i++) {
        float kn = (krow[i] * kscale) * nk_s[i];
        float qn = (qrow[i] * qscale) * nq_s[i];
        dot += kn * qn;
    }
    dot /= sqrtf((float)n);
    // sigmoid(sgn·√max(|s|,1e-6)) (ple.rs L148-150).
    float mag = sqrtf(fmaxf(fabsf(dot), 1e-6f));
    float g = fn_ple_sigmoid((dot >= 0.0f) ? mag : -mag);
    gates[id] = g;
    // value 방송×게이트(L155-159) → grouped rms(n_conv)(L161).
    const float* vrow = value + (long long)ti * n;
    float* orow = gated + ((long long)ti * hc + s) * n;
    const float* nc_s = nc + (long long)s * n;
    int chunk = (n + 31) / 32;
    double gsum = 0.0;
    for (int u = 0; u < 32; u++) {
        int lo = u * chunk;
        if (lo >= n) {
            break;
        }
        int hi = (lo + chunk < n) ? (lo + chunk) : n;
        float gpart = 0.0f;
        for (int i = lo; i < hi; i++) {
            float gv = vrow[i] * g;
            gpart += gv * gv;
        }
        gsum += (double)gpart;
    }
    float gfs = (float)sqrt(gsum / (double)n + (double)eps);
    float gscale = 1.0f / gfs;
    for (int i = 0; i < n; i++) {
        float gv = vrow[i] * g;
        orow[i] = (gv * gscale) * nc_s[i];
    }
}

// ── (3) dilated depthwise conv(kern·dil·hist) + silu ──
// 탭 순서·인덱스: ple.rs L169-184 그대로 — start=hist+ti-(kern-1-k)·dil,
// acc += cw[c·kern+k]·src, 출력 silu. start<hist 열은 상태 버퍼에서.
extern "C" __global__ void llm170_ple_conv(const float* gated, const float* st,
                                           const float* cw, float* out, int t,
                                           int hcd, int kern, int dil, int hist) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= t * hcd) {
        return;
    }
    int ti = idx / hcd;
    int c = idx - ti * hcd;
    float acc = 0.0f;
    for (int k = 0; k < kern; k++) {
        int start = hist + ti - (kern - 1 - k) * dil;
        const float* src = (start < hist)
                               ? (st + (long long)start * hcd + c)
                               : (gated + (long long)(start - hist) * hcd + c);
        acc += cw[(long long)c * kern + k] * (*src);
    }
    out[idx] = fn_ple_silu(acc);
}

// ── (4) conv 상태 tail 갱신(더블버퍼 — 제자리 경합 회피) ──
// new st[j] = padded[t+j] (ple.rs L189-193): t+j<hist면 구상태 열,
// 이후는 gated_hist 열. 판독 st_src/기록 st_dst 분리.
extern "C" __global__ void llm170_ple_conv_state(const float* st_src,
                                                 const float* gated, float* st_dst,
                                                 int t, int hcd, int hist) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hist * hcd) {
        return;
    }
    int j = idx / hcd;
    int c = idx - j * hcd;
    int p = t + j;
    float v = (p < hist) ? st_src[(long long)p * hcd + c]
                         : gated[(long long)(p - hist) * hcd + c];
    st_dst[(long long)j * hcd + c] = v;
}

// ── (5) 잔차 2경로 가산 ──
// row[s·n+i] += value[i]·g + conv_out (ple.rs L217-235) — 곱→가산→가산
// 순서 그대로(-fmad=false로 2중 반올림 유지).
extern "C" __global__ void llm170_ple_resid(float* res, const float* value,
                                            const float* gates,
                                            const float* conv_out, int total,
                                            int hcd, int n, int hc) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int ti = idx / hcd;
    int pos = idx - ti * hcd;
    int s = pos / n;
    int i = pos - s * n;
    float tmp = value[(long long)ti * n + i] * gates[(long long)ti * hc + s] +
                conv_out[idx];
    res[idx] = res[idx] + tmp;
}
