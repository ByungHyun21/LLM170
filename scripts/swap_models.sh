#!/usr/bin/env bash
# W4A16 서빙 스모크 — plans/w4a16-cuda.md §1(W1). 2026-10-08 단일
# 트랙 재편(구 스왑 하네스는 이력 참조).
#
# 판정:
#   1. A(27B W4A16 AutoRound): serve 기동 + 배너(format=w4a16 runtime=cpu) +
#      고정 프롬프트 24토큰.
#   2. B(35B-A3B INT4 g32): 로더 명시 거부(quantization_config 부재 — g32 계열
#      로더는 §2 W4 과제) — 정체불명 실패 금지.
#   3. C(FN FP8PLE): 자원 가드 명시 거부(120GiB > 호스트) — 가드 동작 판정.
# 사용법: scripts/swap_models.sh [백엔드=cpu]
set -u
cd "$(dirname "$0")/.."
BACKEND=${1:-cpu}
BIN=./target/release/llm170
PORT=18210
PROMPT='[148678, 65233, 202419]'
NP=24

M_27B=../models/Qwen3.8-27B-W4A16-AutoRound
M_35B=../models/Qwen3.6-35B-A3B-INT4-W4A16
M_FN=../models/Qwen3.8-Flash-Next-W4A16-FP8PLE

fail=0
note() { echo "[w4a16] $*"; }

run_serve() {
  local model=$1 out=$2 log=$3
  : > "$log"
  ( for i in $(seq 1 60); do
      sleep 5
      grep -m1 "listening" "$log" >/dev/null 2>&1 && break
    done
    curl -s -m 600 "http://127.0.0.1:$PORT/v1/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"prompt\":$PROMPT,\"max_tokens\":$NP,\"temperature\":0}" > "$out"
    pkill -9 -x llm170 ) &
  timeout 420 "$BIN" serve --backend "$BACKEND" --model "$model" --ctx 1024 --slots 1 \
    --port $PORT > "$log" 2>&1
  wait
}

tokens_of() { python3 -c "
import json
d=json.load(open('$1'))
toks=d.get('tokens') or (d['choices'][0].get('tokens') if 'choices' in d else None)
print(','.join(map(str,toks)) if toks else 'ERR')
"; }

# ── 1. A(27B W4A16) 기동 + 토큰 ──
note "[1/3] A(27B W4A16) 기동 — backend=$BACKEND"
run_serve "$M_27B" /tmp/w4a16_a.json /tmp/w4a16_a.log
B=$(grep -m1 "^# boot: " /tmp/w4a16_a.log || true)
case "$B" in
  *"format=w4a16 runtime=cpu"*) note "배너 OK: $B" ;;
  *) echo "[w4a16] FAIL: 배너 부재/불일치: $B"; fail=1 ;;
esac
A=$(tokens_of /tmp/w4a16_a.json)
case "$A" in ERR*) echo "[w4a16] FAIL: 요청 실패: $A"; fail=1;; *) note "토큰 OK: $(echo "$A" | head -c 60)";; esac

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
  echo "[w4a16] smoke: ALL PASS (A 기동·토큰 / B·C 명시 거부, backend=$BACKEND)"
else
  echo "[w4a16] smoke: FAILURES PRESENT"
fi
exit $fail
