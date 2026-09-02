//! gdn_ar — GDN 단일 토큰(AR) 상태 갱신 GPU 커널.
//!
//! 유닛 = S 행렬의 dv 열 1개 (d=128). 각 유닛은 자기 열만 읽고 쓰므로
//! 크로스 유닛 통신 없음:
//!   1) sk[dv] = Σ_kdim (S[kdim,dv]·e^g)·k[kdim]   (S ← S·e^g 동시)
//!   2) delta = (v[dv] − sk[dv])·β; S[kdim,dv] += k[kdim]·delta
//!   3) o[dv] = Σ_kdim S'[kdim,dv]·(q[kdim]·scale)
//! e^g·β·(q·scale)는 호스트 사전 계산 (커널 f32 스칼라 인수·exp 미지원 회피).
//!
//! ADR-0005: mul_add 금지 — 순수 mul+add. cubecl HIP 코드젠 주의 규칙 준수
//! (순수 u32 산술 불필요 — 전 인수 f32 텐서).

use cubecl::prelude::*;

/// 그리드: (h_v, n_seqs) — 큐브당 1 (seq, v-head), 유닛당 dv 열.
#[cube(launch_unchecked)]
pub fn gdn_ar(
    s: &mut Tensor<f32>,          // [b][h_v][d*d] 상태 (in-place)
    q_scaled: &Tensor<f32>,       // [b][k_stride] (q·scale 완료)
    k: &Tensor<f32>,              // [b][k_stride]
    v: &Tensor<f32>,              // [b][v_stride]
    beta_ge: &Tensor<f32>,        // [b][h_v][2] — β, e^g 쌍
    out: &mut Tensor<f32>,        // [b][v_stride]
    d: usize,                     // 상태 차원 (128)
    k_stride: usize,
    v_stride: usize,
    h_v: usize,
    h_k: usize,
) {
    // 큐브 = (b,h) pair 1개, 유닛 = dv 열. CUBE_DIM 128 고정 — d≤128 커버,
    // 초과 유닛 terminate (CubeDim(d<32) 비정상 실측, 2026-09-01).
    let pair = CUBE_POS_X as usize;
    let u = UNIT_POS_X as usize;
    if u >= d {
        terminate!();
    }
    let b = pair / h_v;
    let h = pair % h_v;
    let kh = h % h_k;
    let base_s = pair * d * d;
    let qk0 = b * k_stride + kh * d;
    let v0 = b * v_stride + h * d;
    let beta = beta_ge[pair * 2];
    let g_exp = beta_ge[pair * 2 + 1];

    // 1) S ← S·e^g; sk = Σ S·k
    let mut sk = 0.0f32;
    for kdim in 0..d {
        let sv = s[base_s + kdim * d + u] * g_exp;
        s[base_s + kdim * d + u] = sv;
        sk += sv * k[qk0 + kdim];
    }
    // 2) delta·k 외적 가산
    let delta = (v[v0 + u] - sk) * beta;
    for kdim in 0..d {
        let addr = base_s + kdim * d + u;
        s[addr] += k[qk0 + kdim] * delta;
    }
    // 3) o = Σ S'·q_scaled
    let mut o = 0.0f32;
    for kdim in 0..d {
        o += s[base_s + kdim * d + u] * q_scaled[qk0 + kdim];
    }
    out[v0 + u] = o;
}

/// gdn_chunk_kkt — 청크별 A/KQ 행렬 사전분해 (PP 재설계 1단계).
///
/// 각 유닛 = (i,j)쌍 1개: k_i·k_j / (q_i·scale)·k_j 내적을 **1회만** 계산
/// (구 커널은 유닛마다 동일 내적 중복 — 64× FLOP 낭비, PP 격차 분석
/// source/research/2026-09-02-pp-gap-analysis.md).
/// - A[i][j]  = ((k_i·k_j)·β_i)·e^(gcs_i−gcs_j) — 치환 계수 (j<i만)
/// - KQ[i][j] = ((q_i·s)·k_j 내적)·e^(gcs_i−gcs_j) — 출력 계수 (j≤i)
/// 그리드 (CS²/64 큐브, h_v, n_chunks) — 청크·헤드 전량 병렬.
/// 누산 순서: 내적 s2 오름차순, gcs 청크 내 순차 — CPU와 동일.
#[cube(launch_unchecked)]
pub fn gdn_chunk_kkt(
    q: &Tensor<f32>,      // [t_pad][k_stride] — 무스케일, 제로패딩
    k: &Tensor<f32>,      // [t_pad][k_stride]
    beta: &Tensor<f32>,   // [t_pad][h_v]
    g: &Tensor<f32>,      // [t_pad][h_v]
    a_mat: &mut Tensor<f32>,  // [h_v*n_chunks][CS][CS] — 제로 초기화 전제
    kq_mat: &mut Tensor<f32>, // 동일
    t_len: usize,
    h_k: usize,
    h_v: usize,
    d: usize,
) {
    let h = CUBE_POS_Y as usize;
    let ch = CUBE_POS_Z as usize;
    let u = UNIT_POS_X as usize;
    if h >= h_v {
        terminate!();
    }
    let pair = CUBE_POS_X as usize * 64 + u;
    let i = pair / CS_K;
    let j = pair % CS_K;
    let scale = 1.0f32 / f32::cast_from(d as u32).sqrt();
    let kh = h % h_k;
    let k_stride = h_k * d;
    let t0 = ch * CS_K;
    let n_chunks = (t_len + CS_K - 1) / CS_K;
    let m_base = (h * n_chunks + ch) * CS_K * CS_K;

    if j <= i {
        // gcs 로컬 순차 누산 (CPU 동일 순서)
        let mut acc = 0.0f32;
        let mut gcs_i = 0.0f32;
        let mut gcs_j = 0.0f32;
        for t in 0..=i {
            acc += g[(t0 + t) * h_v + h];
            if t == i {
                gcs_i = acc;
            }
            if t == j {
                gcs_j = acc;
            }
        }
        let qi = (t0 + i) * k_stride + kh * d;
        let kj = (t0 + j) * k_stride + kh * d;
        let mut dot_kk = 0.0f32;
        let mut dot_qk = 0.0f32;
        for s2 in 0..d {
            let kv_j = k[kj + s2];
            dot_kk += k[qi + s2] * kv_j;
            dot_qk += q[qi + s2] * scale * kv_j;
        }
        let e = (gcs_i - gcs_j).exp();
        if j < i {
            let bi = beta[(t0 + i) * h_v + h];
            a_mat[m_base + i * CS_K + j] = dot_kk * bi * e;
        }
        kq_mat[m_base + i * CS_K + j] = dot_qk * e;
    }
}

/// gdn_chunk — GDN 청크 프리필 (t>1) GPU 커널 (03 §3.1).
///
/// 큐브 = v헤드, 유닛 64개가 dv 열 분담 (dv = u, u+64, …, d≤128 → ≤2열).
/// 청크 순차(CS=64) · 청크 내 i 순차(전진 대입) — 유닛은 자기 dv열의 d_j를
/// 로컬 Array로 유지, CPU gdn_chunk_head(gdn.rs:96-216)와 동일 누산 순서
/// (내적 s2 오름차순·j 오름차순·0-스킵·동일 곱셈 그룹핑).
/// 호스트가 q/k/v/β/g를 n_chunks·CS로 제로 패딩 — 패딩 기여는 0 (CPU 동일).
pub const CS_K: usize = 64;

#[cube(launch_unchecked)]
pub fn gdn_chunk(
    v: &Tensor<f32>,          // [t_pad][v_stride] — 제로패딩
    k: &Tensor<f32>,          // [t_pad][k_stride] — 상태 갱신·rhs용
    q: &Tensor<f32>,          // [t_pad][k_stride] — S-항용 (무스케일)
    beta: &Tensor<f32>,       // [t_pad][h_v]
    g: &Tensor<f32>,          // [t_pad][h_v]
    a_mat: &Tensor<f32>,      // [h_v*n_chunks][CS][CS] — kkt 사전분해
    kq_mat: &Tensor<f32>,     // 동일
    states: &mut Tensor<f32>, // [h_v][d*d] in-place
    out: &mut Tensor<f32>,    // [t_pad][v_stride]
    t_len: usize,
    h_k: usize,
    h_v: usize,
    d: usize,
) {
    let h = CUBE_POS_X as usize;
    if h >= h_v {
        terminate!();
    }
    let u = UNIT_POS_X as usize;
    let scale = 1.0f32 / f32::cast_from(d as u32).sqrt();
    let kh = h % h_k;
    let k_stride = h_k * d;
    let n_chunks = (t_len + CS_K - 1) / CS_K;
    let v_stride = h_v * d;
    let st_base = h * d * d;
    let dv0 = u;

    for c in 0..n_chunks {
        let t0 = c * CS_K;
        let n = (t0 + CS_K).min(t_len) - t0;
        // gcs 로컬 순차 누산 (상태 갱신 가중치용)
        let mut gcs = Array::<f32>::new(CS_K);
        let mut acc = 0.0f32;
        for t in 0..CS_K {
            acc += g[(t0 + t) * h_v + h];
            gcs[t] = acc;
        }
        let g_last = gcs[CS_K - 1];
        let m_base = (h * n_chunks + c) * CS_K * CS_K;

        let mut d_all = Array::<f32>::new(2 * CS_K);
        for t in 0..(2 * CS_K) {
            d_all[t] = 0.0;
        }

        for i in 0..CS_K {
            let bi = beta[(t0 + i) * h_v + h];
            let gcs_i = gcs[i];
            for col in 0..2usize {
                let dv = dv0 + col * 64;
                let dc = col * CS_K;
                if dv < d {
                    // rhs_i[dv] = β·v − β·e^{gcs_i}·(k_i·S_prev)[dv]
                    let mut oi = bi * v[(t0 + i) * v_stride + h * d + dv];
                    if bi != 0.0 {
                        let w0 = bi * gcs_i.exp();
                        for s2 in 0..d {
                            let ks = k[(t0 + i) * k_stride + kh * d + s2];
                            if ks != 0.0 {
                                oi -= w0 * ks * states[st_base + s2 * d + dv];
                            }
                        }
                    }
                    // 전진 대입 −Σ_{j<i} A[i,j]·d_j[dv] — 사전분해 행렬 판독
                    for j in 0..i {
                        let aij = a_mat[m_base + i * CS_K + j];
                        if aij != 0.0 {
                            oi -= aij * d_all[dc + j];
                        }
                    }
                    d_all[dc + i] = oi;
                    // o_i[dv] = e^{gcs_i}·(q_i·S_prev)[dv] + Σ_{j≤i} KQ[i,j]·d_j[dv]
                    if i < n {
                        let mut o = 0.0f32;
                        let qi_exp = gcs_i.exp();
                        for s2 in 0..d {
                            let qv = q[(t0 + i) * k_stride + kh * d + s2] * scale;
                            if qv != 0.0 {
                                o += qi_exp * qv * states[st_base + s2 * d + dv];
                            }
                        }
                        let i1 = i + 1;
                        for j in 0..i1 {
                            let kqij = kq_mat[m_base + i * CS_K + j];
                            if kqij != 0.0 {
                                o += kqij * d_all[dc + j];
                            }
                        }
                        out[(t0 + i) * v_stride + h * d + dv] = o;
                    }
                }
            }
        }
        // 상태 갱신: S[s2][dv] ← S·e^{g_last} + Σ_{j<n} k_j[s2]·e^{g_last−gcs_j}·d_j[dv]
        let gl_exp = g_last.exp();
        for col in 0..2usize {
            let dv = dv0 + col * 64;
            let dc = col * CS_K;
            if dv < d {
                for s2 in 0..d {
                    let sb = st_base + s2 * d + dv;
                    let mut s = states[sb] * gl_exp;
                    for j in 0..n {
                        let kj = k[(t0 + j) * k_stride + kh * d + s2];
                        if kj != 0.0 {
                            let w = (g_last - gcs[j]).exp();
                            s += kj * w * d_all[dc + j];
                        }
                    }
                    states[sb] = s;
                }
            }
        }
    }
}
