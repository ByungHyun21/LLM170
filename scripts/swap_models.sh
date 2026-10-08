#!/usr/bin/env bash
# W4A16 스모크 — 참조 러너(w4a16-ref) + 명시 거부 판정. 2026-10-08 단일 트랙.
#
# 판정:
#   1. A(27B W4A16 AutoRound): 참조(CPU) greedy 토큰열이 골든 접두와 일치.
#   2. B(35B-A3B INT4 g32): 로더 명시 거부(quantization_config 부재 — g32는 W4).
#   3. C(FN FP8PLE): 자원 가드 명시 거부(120GiB > 호스트) — 가드 동작 판정.
# 사용법: scripts/swap_models.sh
set -u
cd "$(dirname "$0")/.."
BIN=./target/release/llm170
GOLDEN="156037,16072,154029,209495,30"

M_27B=../models/Qwen3.8-27B-W4A16-AutoRound
M_35B=../models/Qwen3.6-35B-A3B-INT4-W4A16
M_FN=../models/Qwen3.8-Flash-Next-W4A16-FP8PLE

fail=0
note() { echo "[w4a16] $*"; }

# ── 1. A(27B W4A16) 참조 토큰 ──
note "[1/3] A(27B W4A16) 참조 실행 — w4a16-ref"
timeout 900 "$BIN" w4a16-ref "$M_27B" --prompt-tokens 148678,65233,202419 --n-predict 24 --ctx 1024 \
  > /tmp/w4a16_a.out 2> /tmp/w4a16_a.log
A=$(grep -m1 '^tokens:' /tmp/w4a16_a.out | sed 's/^tokens: //')
case "$A" in
  "$GOLDEN"*) note "토큰 OK: $(echo "$A" | head -c 60)";;
  "") echo "[w4a16] FAIL: 참조 실행 실패 — /tmp/w4a16_a.out /tmp/w4a16_a.log"; fail=1;;
  *) echo "[w4a16] FAIL: 골든 접두 불일치: $(echo "$A" | head -c 60)"; fail=1;;
esac

# ── 2. B(35B INT4 g32) 명시 거부 ──
note "[2/3] B(35B INT4 g32) 로더 명시 거부 판정"
if timeout 60 "$BIN" w4a16-load "$M_35B" > /tmp/w4a16_b.log 2>&1; then
  echo "[w4a16] FAIL: 거부되어야 할 g32 자산이 통과 — /tmp/w4a16_b.log"; fail=1
elif grep -q "w4a16" /tmp/w4a16_b.log; then
  note "명시 거부 OK: $(head -1 /tmp/w4a16_b.log)"
else
  echo "[w4a16] FAIL: 원인 불명 거부 — /tmp/w4a16_b.log"; fail=1
fi

# ── 3. C(FN FP8PLE) 자원 가드 명시 거부 ──
note "[3/3] C(FN FP8PLE) 자원 가드 명시 거부 판정"
if timeout 60 "$BIN" w4a16-load "$M_FN" > /tmp/w4a16_c.log 2>&1; then
  echo "[w4a16] FAIL: 가드가 통과시킴(120GiB) — /tmp/w4a16_c.log"; fail=1
elif grep -qE "insufficient resources|rsrc-guard" /tmp/w4a16_c.log; then
  note "가드 거부 OK: $(head -1 /tmp/w4a16_c.log)"
else
  echo "[w4a16] FAIL: 가드 외 사유 — /tmp/w4a16_c.log"; fail=1
fi

if [ $fail -eq 0 ]; then
  echo "[w4a16] smoke: ALL PASS (A 참조 토큰 / B·C 명시 거부)"
else
  echo "[w4a16] smoke: FAILURES PRESENT"
fi
exit $fail
