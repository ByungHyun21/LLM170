#!/usr/bin/env bash
# mod-check-all.sh — 전체 모듈 무결성 인증 순회 (plans/129 A23⑤·사용자 계약).
#
# 세션당 1회 전체 인증: mod-check --if-stale가 소스 지문이 마지막 PASS와
# 동일한 모듈을 건너뛰므로, 재실행되는 것은 "코드가 바뀐 모듈"뿐이다.
# 종료 코드 = FAIL 모듈 수(0 = 전 certified).
#
# 사용법:
#   scripts/mod-check-all.sh              # 지문 스킵 순회(일상용)
#   scripts/mod-check-all.sh --record-all # 골든 전부 재캡처(산술 10a 변경 시)
#
# GPU 독점 계약 — 다른 추론 프로세스 상주 시 돌리지 말 것. 피크 VRAM =
# 단일 모듈(1호출 1모듈, 부분적재).
set -uo pipefail
cd "$(dirname "$0")/.."
if [ -d /opt/rocm-10.0.0/install/bin ]; then
    export PATH=/opt/rocm-10.0.0/install/bin:$PATH
fi
if [ -d /opt/rocm-10.0.0/install/lib ]; then
    export LD_LIBRARY_PATH=/opt/rocm-10.0.0/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi
BIN=${LLM170_BIN:-./target/release/llm170}
[[ -x "$BIN" ]] || { echo "바이너리 없음: $BIN (cargo build --release -p llm170-server)"; exit 2; }

REC=""
if [[ "${1:-}" == "--record-all" ]]; then REC="--record"; fi

fail=0
for m in $("$BIN" mod-check --list); do
    if ! "$BIN" mod-check "$m" --if-stale $REC; then
        fail=$((fail+1))
    fi
done
echo "── mod-check-all 종료: FAIL ${fail}건 ──"
exit "$fail"
