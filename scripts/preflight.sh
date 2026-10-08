#!/usr/bin/env bash
# W10 프리플라이트 — 커밋 전 로컬 검증 단일 진실 공급원
#
# 검증: (1) rustfmt, (2) clippy -D warnings, (3) cargo 경고 0,
#       (4) env 스냅샷↔라이브 동치(108 P1).
# [2026-10-08] 단일 트랙 재편 — spv·charhash
# 스텝 제거. W4A16 커널(W2) 도입 시 그 게이트는 새 스크립트로 붙인다.
# 사용: scripts/preflight.sh
set -uo pipefail
cd "$(dirname "$0")/.."
fail=0

echo "== 1/4 rustfmt =="
if cargo fmt --all -- --check >/dev/null 2>&1; then
    echo "OK"
else
    echo "FAIL — cargo fmt --all 적용 후 재시도"; cargo fmt --all -- --check | head -10; fail=1
fi

echo "== 2/4 clippy =="
if cargo clippy --workspace --all-targets -- -D warnings >/dev/null 2>&1; then
    echo "OK"
else
    echo "FAIL:"; cargo clippy --workspace --all-targets -- -D warnings 2>&1 | grep -E "^(error|warning)" | sort | uniq -c | head -10; fail=1
fi

echo "== 3/4 cargo 경고 0 (dev check — 릴리스는 빌드 게이트가 담당) =="
w=$(cargo check --workspace --all-targets 2>&1 | grep -c "^warning" || true)
if [[ "$w" -eq 0 ]]; then echo "OK"; else echo "FAIL — warning ${w}건"; fail=1; fi

echo "== 4/4 env 동치 (108 P1) =="
if cargo build --release -q -p llm170-server >/dev/null 2>&1; then
    if ! ./target/release/llm170 diag envcheck >/dev/null; then
        ./target/release/llm170 diag envcheck; fail=1
    else
        echo "OK"
    fi
else
    echo "FAIL — 빌드 오류"; fail=1
fi

if [[ "$fail" -eq 0 ]]; then echo "== 프리플라이트 PASS =="; else echo "== 프리플라이트 FAIL =="; exit 1; fi
