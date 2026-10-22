#!/usr/bin/env bash
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
