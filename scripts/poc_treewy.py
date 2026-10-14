#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# [2.B 2026-10-12] GDN 스펙 검증 TreeWY(WY 변환) PoC — numpy 수치 검증(CPU 전용)
#
# 질문: 스펙 검증 경로(gdn.cu gdn_spec_scan — 토큰 순차 스캔 + 토큰별 상태
# 스냅샷 dsnap[8][n_layers][h_v][128²], 27B 1.50GiB·35B 640MiB)를 WY 표현
# 「solve 1회 + 유사값(pseudo-value) dc」으로 바꿔도 수치가 같은가?
# (2.B 트랙 — TreeWY 2608.20961 정리 §4.3, fp32 ~1e-7 기대)
#
# GDN 상태 갱신(gdn.cu gdn_spec_scan / gdn1 trio — 헤드·층별, d=128,
# S=[key행 128][value열 128], qscale=1/sqrt(128)):
#   e_t = exp(g_t),  KS_t = k_t^T S_{t-1},  rhs_t = β_t(v_t − e_t·KS_t)
#   o_t = e_t·(S_{t-1}^T q_t)·qscale + qscale·(q_t^T k_t)·rhs_t
#       (= qscale·S_t^T q_t — 갱신 후 상태 기준 판독과 동일 산술)
#   S_t = e_t·S_{t-1} + k_t·rhs_t^T
#
# WY(청크 solve) 형태 — gdn.cu gdn_scan(+gdn_scan_akq)이 프리필 청크(CS=32)에서
# 이미 쓰는 산술을, S_0(배치 시작 상태) 기준 t≤8 한 청크로:
#   G_i = Σ_{r≤i} g_r (포함 누적 = gcs),  D_i = exp(G_i)
#   A[i][j] = β_i·(k_i·k_j)·exp(G_i−G_j)          (j<i 강하삼각)
#   rhs0_i  = β_i·(v_i − D_i·(k_i^T S_0))
#   dc = (I−A)⁻¹·rhs0     ← 순전대입 1회 = "WY solve"
#                            (논문형 W=(I−tril(βxxᵀ))⁻¹과 동일 — 실행부에서 대조)
#   o_i = D_i·(q_i^T S_0)·qscale + Σ_{p≤i}(q_i·k_p)·qscale·exp(G_i−G_p)·dc_p
#   S_m = exp(G_{m-1})·S_0 + Σ_{j≤m-1} exp(G_{m-1}−G_j)·k_j·dc_jᵀ
#         (m=1..t 임의 접두 재구성 — 스냅샷[keep-1] D2D의 대체물)
#
# 검증 항목:
#   1) 출력 o(= 검증 로짓의 GDN 성분) 일치 — fp32 ~1e-7 기준(플랜 §4.3)
#   2) 모든 접두 상태 S_m(부분 수용 롤백 지점) 일치
#   3) 순전대입 == 밀집 역행렬(W 형태, f64) — 논문 정리와 동일 산술 대조
# 커널의 half(k)/half(v) 저장 반올림은 양 경로에 동일 입력으로 주입
# (경로 차이만 분리 — CUDA 이식 시에도 두 경로의 입력은 동일).
# A/KQ는 f32 사용(프리필 gdn_scan은 half 저장이나 t≤8은 smem 8×8 —
# f32 상주 가능; PoC는 양식 자체의 동등성이 대상).

import numpy as np

D = 128  # d_k = d_v(GDN 헤드 폭 — 27B·35B 공통)


def f16r(x):
    # 커널 미러 — k/v는 f16 저장 경유(l2perm→scan 반올림 지점).
    return x.astype(np.float16).astype(x.dtype)


def qscale_of(dtype):
    dt = np.dtype(dtype).type
    return dt(1.0) / np.sqrt(dt(D))


def seq_scan(S0, Q, K, V, beta, g):
    # 순차 스캔(gdn_spec_scan 산술 미러) — 토큰별 스냅샷이 현 엔진의 세금.
    t = len(g)
    qscale = qscale_of(S0.dtype)
    S = S0.copy()
    O = np.zeros((t, D), dtype=S0.dtype)
    snaps = []
    for i in range(t):
        e = np.exp(g[i])
        ks = K[i] @ S
        qs = Q[i] @ S
        rhs = beta[i] * (V[i] - e * ks)
        kq00 = (Q[i] @ K[i]) * qscale
        O[i] = e * qs * qscale + kq00 * rhs
        S = e * S + np.outer(K[i], rhs)
        snaps.append(S.copy())  # ← dsnap(롤백 복원 지점)
    return O, snaps, S


def wy_solve(S0, Q, K, V, beta, g):
    # WY/청크 solve(gdn_scan 산술 계열) — S_0 기준 solve 1회 + 유사값 dc.
    t = len(g)
    qscale = qscale_of(S0.dtype)
    G = np.cumsum(g)          # 포함 누적(= gcs)
    Dv = np.exp(G)
    KS = K @ S0
    rhs0 = beta[:, None] * (V - Dv[:, None] * KS)
    gram = K @ K.T
    A = np.zeros((t, t), dtype=S0.dtype)
    for i in range(t):
        for j in range(i):
            A[i, j] = beta[i] * gram[i, j] * np.exp(G[i] - G[j])
    dc = np.zeros((t, D), dtype=S0.dtype)
    for i in range(t):
        r = rhs0[i].copy()
        for j in range(i):
            r -= A[i, j] * dc[j]
        dc[i] = r
    # o_i = D_i·QS_i·qscale + Σ_{p≤i} KQ[i][p]·dc_p
    QS = Q @ S0
    O = Dv[:, None] * QS * qscale
    for i in range(t):
        for p in range(i + 1):
            O[i] += (Q[i] @ K[p]) * qscale * np.exp(G[i] - G[p]) * dc[p]
    return O, dc, G, A, rhs0


def wy_state(S0, K, dc, G, m):
    # 접두 m 상태 재구성(롤백 대체) — S_m = exp(G_{m-1})S_0 + Σ_{j<m} k_j dc_jᵀ
    Gm = G[m - 1]
    S = np.exp(Gm) * S0
    for j in range(m):
        S += np.exp(Gm - G[j]) * np.outer(K[j], dc[j])
    return S


def make_case(rng, t, dtype, *, decay="realistic", corr=0.0, beta_mode="realistic"):
    dtype = np.dtype(dtype).type
    Q = rng.standard_normal((t, D)).astype(dtype)
    K = rng.standard_normal((t, D)).astype(dtype)
    if corr > 0.0:
        base = rng.standard_normal(D).astype(dtype)
        base /= np.linalg.norm(base)
        K = np.sqrt(dtype(1 - corr)) * K + np.sqrt(dtype(corr)) * base
    # k는 헤드별 L2 정규화(gdn_l2perm 계약)
    K = K / np.sqrt(np.sum(K * K, axis=1, keepdims=True) + dtype(1e-6))
    V = rng.standard_normal((t, D)).astype(dtype)
    if decay == "realistic":
        # g = −softplus(a+dt)·exp(A_log) 부류: 음수, 폭 넓게
        a = np.abs(rng.standard_normal(t)).astype(dtype) * dtype(2.0) + dtype(0.1)
        g = -np.log1p(np.exp(a)) * rng.uniform(0.5, 2.0, t).astype(dtype)
    elif decay == "strong":
        # 35B 헤드11 실측급(g≈−91.6 → e=0 언더플로)
        g = np.full(t, -91.6, dtype=dtype)
    elif decay == "nodecay":
        # 감쇠 ≈ 1(조건수 스트레스)
        g = -np.full(t, 1e-5, dtype=dtype)
    else:
        raise ValueError(decay)
    if beta_mode == "realistic":
        beta = (1.0 / (1.0 + np.exp(-(rng.standard_normal(t) * 0.5 + 0.3)))).astype(dtype)
    else:
        beta = np.ones(t, dtype=dtype)
    S0 = (rng.standard_normal((D, D)) * 0.3).astype(dtype)
    return Q, f16r(K), f16r(V), beta, g, S0


def run_case(name, t, dtype, seed=1234, **kw):
    rng = np.random.default_rng(seed)
    Q, K, V, beta, g, S0 = make_case(rng, t, dtype, **kw)
    O_seq, snaps, S_end_seq = seq_scan(S0, Q, K, V, beta, g)
    O_wy, dc, G, A, rhs0 = wy_solve(S0, Q, K, V, beta, g)

    do_abs = float(np.max(np.abs(O_seq - O_wy)))
    do_rel = do_abs / max(float(np.max(np.abs(O_seq))), 1e-30)

    ds_abs = 0.0
    ds_rel = 0.0
    for m in range(1, t + 1):
        d = np.abs(snaps[m - 1] - wy_state(S0, K, dc, G, m))
        ds_abs = max(ds_abs, float(np.max(d)))
        ds_rel = max(ds_rel, float(np.max(d)) / max(float(np.max(np.abs(snaps[m - 1]))), 1e-30))

    # [3] 밀집 대조 — 순전대입 dc_i = rhs0_i − ΣA[i,j]dc_j 는 (I+A)dc=rhs0의
    # 해(항 이동 시 부호 +). 논문형 (I−tril(βxx^T))⁻¹과는 tril 부호 규약 차이
    # (L=−A로 두면 (I−L)⁻¹). f64로 계산(LAPACK 노이즈 제거).
    Af = A.astype(np.float64)
    M = np.eye(t) + Af
    dc_dense = np.linalg.solve(M, rhs0.astype(np.float64))
    ddense = float(np.max(np.abs(dc.astype(np.float64) - dc_dense)))
    cond = float(np.linalg.cond(M))

    print(f"{name:<28s} {np.dtype(dtype).name:<8s} t={t}  "
          f"|Δo|={do_abs:9.3e} rel={do_rel:8.2e}  "
          f"|ΔS_m|(m=1..{t})={ds_abs:9.3e} rel={ds_rel:8.2e}  "
          f"cond(I+A)={cond:8.2e}  |Δdc(subst−dense)|={ddense:8.2e}")
    if ddense > 1e-5:
        i = int(np.argmax(np.max(np.abs(dc.astype(np.float64) - dc_dense), axis=1)))
        print(f"    [diag] dense 불일치 행 {i}: subst[:3]={dc[i][:3]} dense[:3]={dc_dense[i][:3]}")
    return do_abs, ds_abs


def memory_accounting():
    # dsnap = 8×n_layers×h_v×128²×4B(gdn.rs enable_spec 할당 — 스트라이드
    # n_layers는 스캔 계약). WY 대체물 = dc[t×n_gdn×h_v×128] + G[t×n_gdn×h_v]
    # + S_0 1개(dsave급 버퍼 재사용 가능). 슬롯 상태 원장: 27B 151MB·35B 63MB.
    for tag, n_layers, n_gdn, h_v in (("27B", 64, 48, 48), ("35B", 40, 30, 32)):
        dsnap = 8 * n_layers * h_v * 128 * 128 * 4
        dc = 8 * n_gdn * h_v * 128 * 4 + 8 * n_gdn * h_v * 4
        s0 = n_gdn * h_v * 128 * 128 * 4
        per_round_now = 8 * n_gdn * h_v * 128 * 128 * 4  # 토큰별 기록(n_gdn 슬라이스)
        print(f"{tag}: dsnap {dsnap/2**30:7.3f} GiB → dc+G {dc/2**20:6.2f} MiB "
              f"(×{dsnap/dc:5.0f}) + S_0 {s0/2**20:6.1f} MiB 1개 | "
              f"라운드 스냅샷 기록 {per_round_now/2**30:5.2f} GiB → "
              f"{(s0+dc)/2**20:6.1f} MiB (×{per_round_now/(s0+dc):4.1f}↓)")


def main():
    print(f"numpy {np.__version__} — GDN TreeWY PoC (d={D}, CPU)")
    worst = [0.0, 0.0]
    print("[1] 기본 케이스(현실적 게이트)")
    for dt in (np.float32, np.float64):
        a, b = run_case("realistic", 8, dt)
        if dt == np.float32:
            worst = [max(worst[0], a), max(worst[1], b)]
    print("[2] t 스윕(spec_verify 계약 도메인 2..=8, fp32)")
    for t in range(2, 9):
        a, b = run_case("realistic", t, np.float32, seed=1000 + t)
        worst = [max(worst[0], a), max(worst[1], b)]
    print("[3] 스트레스")
    run_case("strong-decay g=−91.6", 8, np.float32, decay="strong")
    run_case("nodecay+corr.9+β=1", 8, np.float32, decay="nodecay", corr=0.9,
             beta_mode="unit")
    print("[4] 메모리 산정(현 dsnap 대비)")
    memory_accounting()
    ok = worst[0] <= 1e-6 and worst[1] <= 1e-6
    print(f"[판정] fp32 realistic 전 도메인 최대 |Δo|={worst[0]:.3e} · "
          f"|ΔS_m|={worst[1]:.3e} → {'PASS (기준 ~1e-7 대역)' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
