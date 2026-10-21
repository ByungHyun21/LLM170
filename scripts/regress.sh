#!/usr/bin/env bash
# [테스트 재설계 2026-10-22] 검증 티어 러너 — 매 변경 루틴의 단일 진입점.
#
#   scripts/regress.sh t0    T0(~1분): 프리플라이트 + 전 크레이트 유닛테스트(무GPU)
#   scripts/regress.sh t1    T1(~15s): tiny 고정재료 CPU↔GPU 같은 실행 등가(골든 불변)
#   scripts/regress.sh t2    T2(~2분): 실모델 캐나리 — 자원 가드 + serve 골든·배치 불변
#   scripts/regress.sh all   t0+t1+t2
#
# 원칙(AGENTS VERIFICATION TIERS): 변경 축과 무관한 게이트를 매 변경마다 돌리지
# 않는다. 수치 경로(커널·산술·샘플링) 변경·캠페인 머지 = swap_models.sh 전체 +
# w4a16-eval ±3% + ab-bench(같은 세션). 골든 값 = scripts/goldens.tsv 단일 원장.
# tiny 재료 결정론 = scripts/gen_tiny_w4a16.py(시드 고정 — 재생성 동일 바이트).
set -uo pipefail
cd "$(dirname "$0")/.."
TIER="${1:-all}"

run_t0() {
    echo "== T0: preflight + cargo test(무GPU) =="
    scripts/preflight.sh || return 1
    for c in llm170-core llm170-backend-gpu llm170-server llm170-diag; do
        cargo test --release -q -p "$c" 2>&1 | grep -E "test result" | grep -q "0 failed" \
            || { echo "T0 FAIL: $c"; return 1; }
    done
    echo "T0 OK"
}

run_t1() {
    echo "== T1: tiny 등가(CPU 오라클 ↔ GPU, 골든 불변) =="
    if [ ! -d target/tiny-fixtures/tiny-dense ]; then
        python3 scripts/gen_tiny_w4a16.py || return 1
    fi
    cargo test --release -q -p llm170-server -- --ignored --test-threads=1 \
        | grep -E "test result" | grep -q " 0 failed" \
        || { echo "T1 FAIL"; return 1; }
    echo "T1 OK"
}

run_t2() {
    echo "== T2: 실모델 캐나리(가드 + serve 골든·배치 불변 — 골든=goldens.tsv) =="
    scripts/swap_models.sh --fast || { echo "T2 FAIL"; return 1; }
    echo "T2 OK"
}

case "$TIER" in
    t0) run_t0 ;;
    t1) run_t1 ;;
    t2) run_t2 ;;
    all) run_t0 && run_t1 && run_t2 ;;
    *) echo "사용: scripts/regress.sh [t0|t1|t2|all]" >&2; exit 2 ;;
esac
