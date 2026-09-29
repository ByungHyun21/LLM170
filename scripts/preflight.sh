#!/usr/bin/env bash
# W10 프리플라이트 — 커밋 전 로컬 검증 단일 진실 공급원 (plans/107)
#
# 검증: (1) rustfmt, (2) clippy -D warnings, (3) cargo 경고 0,
#       (4) spv 신선도, (5) env 스냅샷↔라이브 동치(108 P1),
#       (6) 스테이지 특성화 해시(charhash.sh verify, SKIP_CHARHASH=1 우회).
# 사용: scripts/preflight.sh           # 전체
#       SKIP_CHARHASH=1 scripts/preflight.sh   # GPU 점유 시 해시만 제외
set -uo pipefail
cd "$(dirname "$0")/.."
fail=0

echo "== 1/6 rustfmt =="
if cargo fmt --all -- --check >/dev/null 2>&1; then
    echo "OK"
else
    echo "FAIL — cargo fmt --all 적용 후 재시도"; cargo fmt --all -- --check | head -10; fail=1
fi

echo "== 2/6 clippy =="
if cargo clippy --workspace --all-targets -- -D warnings >/dev/null 2>&1; then
    echo "OK"
else
    echo "FAIL:"; cargo clippy --workspace --all-targets -- -D warnings 2>&1 | grep -E "^(error|warning)" | sort | uniq -c | head -10; fail=1
fi

echo "== 3/6 cargo 경고 0 (debug+release check) =="
w=$(cargo check --workspace --all-targets 2>&1 | grep -c "^warning" || true)
if [[ "$w" -eq 0 ]]; then echo "OK"; else echo "FAIL — warning ${w}건"; fail=1; fi

echo "== 4/6 spv 신선도 =="
stale=0
for comp in crates/backend-gpu/src/rawvk/spv/*.comp; do
    spv="${comp%.comp}.spv"
    if [[ ! -f "$spv" || "$comp" -nt "$spv" ]]; then
        echo "  stale: ${comp##*/}"; stale=$((stale+1))
    fi
done
if ! python3 scripts/spv-manifest.py --check; then fail=1; fi

echo "== 5/6 env 동치 (108 P1) =="
if cargo build --release -q -p llm170-server >/dev/null 2>&1; then
    if ! ./target/release/llm170 diag envcheck >/dev/null; then
        ./target/release/llm170 diag envcheck; fail=1
    else
        echo "OK"
    fi
else
    echo "FAIL — 빌드 오류"; fail=1
fi

echo "== 6/6 특성화 해시 =="
if [[ "${SKIP_CHARHASH:-0}" == 1 ]]; then
    echo "SKIP (SKIP_CHARHASH=1)"
else
    if ! ./scripts/charhash.sh verify; then fail=1; fi
fi

if [[ "$fail" -eq 0 ]]; then echo "== 프리플라이트 PASS =="; else echo "== 프리플라이트 FAIL =="; exit 1; fi
