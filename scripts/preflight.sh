#!/usr/bin/env bash
# W10 프리플라이트 — 커밋 전 로컬 검증 단일 진실 공급원
#
# 검증: (1) rustfmt, (2) clippy -D warnings, (3) cargo 경고 0,
#       (4) env 스냅샷↔라이브 동치(108 P1), (5) 스테일 fatbin 가드
#       (.cu/cast_common.cuh/g4_common.cuh가 fatbin보다 새로우면 FAIL —
#        grep|tail 사고 클래스).
# [2026-10-08] 단일 트랙 재편 — spv·charhash
# 스텝 제거. W4A16 커널(W2) 도입 시 그 게이트는 새 스크립트로 붙인다.
# 사용: scripts/preflight.sh
set -uo pipefail
cd "$(dirname "$0")/.."
fail=0

echo "== 1/5 rustfmt =="
if cargo fmt --all -- --check >/dev/null 2>&1; then
    echo "OK"
else
    echo "FAIL — cargo fmt --all 적용 후 재시도"; cargo fmt --all -- --check | head -10; fail=1
fi

echo "== 2/5 clippy =="
if cargo clippy --workspace --all-targets -- -D warnings >/dev/null 2>&1; then
    echo "OK"
else
    echo "FAIL:"; cargo clippy --workspace --all-targets -- -D warnings 2>&1 | grep -E "^(error|warning)" | sort | uniq -c | head -10; fail=1
fi

echo "== 3/5 cargo 경고 0 (dev check — 릴리스는 빌드 게이트가 담당) =="
w=$(cargo check --workspace --all-targets 2>&1 | grep -c "^warning" || true)
if [[ "$w" -eq 0 ]]; then echo "OK"; else echo "FAIL — warning ${w}건"; fail=1; fi

echo "== 4/5 env 동치 (108 P1) =="
if cargo build --release -q -p llm170-server >/dev/null 2>&1; then
    if ! ./target/release/llm170 diag envcheck >/dev/null; then
        ./target/release/llm170 diag envcheck; fail=1
    else
        echo "OK"
    fi
else
    echo "FAIL — 빌드 오류"; fail=1
fi

echo "== 5/5 스테일 fatbin 가드 =="
A=crates/backend-gpu/src/rawcuda/assets
stale=0
for cu in "$A"/*.cu; do
    fb="${cu%.cu}.fatbin"
    if [[ ! -f "$fb" || "$cu" -nt "$fb" ]]; then
        echo "FAIL — $(basename "$cu") 가 fatbin보다 새로움(리빌드: scripts/build_cuda_kernels.sh)"; stale=1
    fi
done
# cast_common.cuh는 gptq4/norm/ew가 include(헤더 변경도 리빌드 대상).
for n in gptq4 norm ew; do
    fb="$A/$n.fatbin"
    if [[ ! -f "$fb" || "$A/cast_common.cuh" -nt "$fb" ]]; then
        echo "FAIL — cast_common.cuh 가 $n.fatbin보다 새로움(리빌드 필요)"; stale=1
    fi
done
# g4_common.cuh는 gptq4/moe가 include(R5 분할 — 헤더 변경도 리빌드 대상).
for n in gptq4 moe; do
    fb="$A/$n.fatbin"
    if [[ ! -f "$fb" || "$A/g4_common.cuh" -nt "$fb" ]]; then
        echo "FAIL — g4_common.cuh 가 $n.fatbin보다 새로움(리빌드 필요)"; stale=1
    fi
done
if [[ "$stale" -eq 0 ]]; then echo "OK"; else fail=1; fi

if [[ "$fail" -eq 0 ]]; then echo "== 프리플라이트 PASS =="; else echo "== 프리플라이트 FAIL =="; exit 1; fi
