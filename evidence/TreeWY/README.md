# GDN 스펙 검증 TreeWY PoC — 2026-10-12 (§4.3 2.B)

## 결론

WY 변환(청크 solve 1회 + 유사값 dc)은 순차 GDN 스캔(gdn_spec_scan 산술)과
**수치적으로 동등**하다 — CUDA 이식 전제 충족.

- f64: |Δo| = 1.2e-16 · |ΔS_m| = 2.2e-16 → 양식은 같은 수학(항등 확인).
- fp32(현실적 게이트, t=2..8 전 도메인): **|Δo| ≤ 8.9e-8 · |ΔS_m| ≤ 1.2e-7**
  (기준 fp32 ~1e-7 대역 → PASS).
- 순전대입 solve == 밀집 (I+A)⁻¹ (f64 4.4e-16) — 논문(TreeWY 2608.20961)
  정리의 W=(I−tril(βxxᵀ))⁻¹ 형태와 동일 산술(부호 규약: L=−A).
- 강감쇠(g=−91.6, 35B 헤드11 부류): 차이 정확히 0(자명 케이스).
- 병리 스트레스(β=1·감쇠≈0·k 상관 0.9): 4.8e-7 — cond(I+A)=1.77, 여전히
  eval 클래스.

실행: `python3 poc_treewy.py` (CPU·numpy 전용, GPU 무접촉). 출력 사본 = result.txt.

## 현 엔진 대체 지점 (2026-10-12 판 기준)

| 역할 | 현재 | WY 대체 |
|---|---|---|
| 검증 순전파 | `gdn.rs` spec 브랜치(`spec_on && t∈2..=8`) → `gdn_spec_scan` 토큰 루프 + `dsnap[t][n_layers][h_v][128²]` 토큰별 기록 | solve형 1런치(gdn_scan 계열 산술, t≤8 단일 청크) — 출력 o 전 positions + `dc[t][n_gdn][h_v][128]`·`G[t][n_gdn][h_v]` 기록. **S_0(라운드 시작 상태) 1회 저장**(dsave급) |
| 롤백(부분 수용 keep=m) | `spec.rs::spec_rollback` — `dsnap[m-1]` 전층 D2D 복사 | 재구성 커널: `S_m = exp(G_{m-1})·S_0 + Σ_{j<m} exp(G_{m-1}−G_j)·k_j·dc_jᵀ` (k행은 라운드 입력 dgk에 잔존) |
| 할당 | `gdn.rs::enable_spec` — dsnap = 8×n_layers×h_v×128²×4B | dsnap 폐지 → dc+G + S_0 버퍼 1개 |

- conv 링 스냅샷(`dsnap_ring`)은 그대로 — 링은 **입력 이력**(WY 대상 아님),
  크기도 작음(8×n_layers×3×conv_ch×4B).
- 산술 계약 클래스: 검증 경로 수치가 바뀌는 교체이므로 채택 시 통계
  eval(PPL·argmax 일치율)+골든 재판정 절차 적용(단일 경로 방침).
  프리필 gdn_scan이 이미 같은 solve 산술을 CS=32 청크로 쓰고 있어
  위험 클래스는 기존과 동일(A/KQ 저장은 t≤8에서 f32 smem 상주 가능).

## 메모리·대역폭 산정 (n_layers/n_gdn/h_v: 27B=64/48/48, 35B=40/30/32)

| 모델 | dsnap(현) | dc+G | 절감 | S_0(1개) | 라운드별 기록 트래픽 |
|---|---|---|---|---|---|
| 27B | 1.500 GiB | 9.07 MiB | ×169 | 144.0 MiB | 1.12 GiB → 153.1 MiB(×7.5↓) |
| 35B | 0.625 GiB | 3.78 MiB | ×169 | 60.0 MiB | 0.47 GiB → 63.8 MiB(×7.5↓) |

스트리밍(비상주 MoE)에서 스펙 강제 비활성(enable_spec Err)의 주 원인인
dsnap이 사라짐 → 스트리밍 스펙 활성의 구조적 길. 유사값은 토큰당 상태
128×128 대신 128 벡터 = **128×↓**(d=128), 잔여 1.3×는 n_layers 스트라이드
낭비 제거 효과.
