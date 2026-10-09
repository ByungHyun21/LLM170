#!/usr/bin/env bash
# W4A16 스모크 — 참조(CPU)·GPU 체인·serve 종단 + 명시 거부 판정.
# 2026-10-08 단일 트랙(W3-3: GPU 서빙 착륙).
#
# 판정:
#   1. A(27B): 참조(CPU) greedy 토큰열 골든 접두 일치 — 오라클.
#   2. B(27B): GPU 체인(w4a16-gpu) 토큰열 골든 접두 일치.
#   3. C(27B): serve HTTP 종단 — 배너 runtime=cuda + 골든 접두.
#   4. D(35B-A3B INT4 g32): 참조(CPU) MoE 오라클 — W4-1(2026-10-08) g32/bf16
#      전문가 + 라우터 top-8·shared 게이트. 골든 접두 일치 + 로드 완전성.
#   5. D2(35B) GPU 체인(MoE) — w4a16-gpu 골든 접두(플레인 bf16 + 전문가 상주).
#   6. D3(35B) 다중 청크(35토큰 = 32+3) — 배치 프리필 t≤32의 CPU/GPU 골든.
#      (2026-10-09 교훈: 3토큰 케이스는 단일 청크만 탄다 — 다중 청크를 상시
#       검증한다. 이 케이스가 gdn_exp 도메인 결함을 잡는 그물이다.)
#   7. E(FN FP8PLE): 자원 가드 명시 거부(120GiB > 호스트).
# 사용법: scripts/swap_models.sh
set -u
cd "$(dirname "$0")/.."
BIN=./target/release/llm170
GOLDEN="156037,16072,154029,209495,30"
# W4-1 MoE(35B-A3B g32) — 참조 골든 접두(2026-10-09 확정).
GOLDEN35="156037,16072,154029,156504,30"
# 다중 청크(35토큰)골든 — CPU·GPU 동일 실측(2026-10-09).
GOLDEN35L="271,248068,271,248069,271,168951,227596,149285,65233"
P35L="148678,65233,202419,156037,16072,154029,156504,30,149285,65233,195939,149820,152091,150868,54581,155180,177992,12434,170819,149820,152091,16869,201523,229231,204817,13,153343,51643,155497,171699,25845,159667,32762,189291,13"
# A2: 동시 실행 충돌 방지 — 실행별 런 디렉터리·포트(고정 18210/고정 파일명 해소).
mkdir -p /tmp/opencode
RUN=$(mktemp -d /tmp/opencode/w4a16-smoke.XXXXXX)
PORT=$((18210 + ($$ % 400)))

M_27B=../models/Qwen3.8-27B-W4A16-AutoRound
M_35B=../models/Qwen3.6-35B-A3B-INT4-W4A16
M_FN=../models/Qwen3.8-Flash-Next-W4A16-FP8PLE

fail=0
note() { echo "[w4a16] $*"; }

# ── 1. A(27B) 참조 토큰 ──
note "[1/7] A(27B) 참조 실행 — w4a16-ref"
timeout 900 "$BIN" w4a16-ref "$M_27B" --prompt-tokens 148678,65233,202419 --n-predict 8 --ctx 1024 \
  > "$RUN/a.out" 2> "$RUN/a.log"
A=$(grep -m1 '^tokens:' "$RUN/a.out" | sed 's/^tokens: //')
case "$A" in
  "$GOLDEN"*) note "참조 토큰 OK: $(echo "$A" | head -c 60)";;
  "") echo "[w4a16] FAIL: 참조 실행 실패 — ${RUN}/a.log"; fail=1;;
  *) echo "[w4a16] FAIL: 골든 접두 불일치: $(echo "$A" | head -c 60)"; fail=1;;
esac

# ── 2. B(27B) GPU 체인 ──
note "[2/7] B(27B) GPU 체인 — w4a16-gpu"
timeout 900 "$BIN" w4a16-gpu "$M_27B" --prompt-tokens 148678,65233,202419 --n-predict 8 --ctx 1024 \
  > "$RUN/b.out" 2> "$RUN/b.log"
B=$(grep -m1 '^ tokens:' "$RUN/b.out" | sed 's/^ tokens: //')
case "$B" in
  "$GOLDEN"*) note "GPU 토큰 OK: $(echo "$B" | head -c 60)";;
  "") echo "[w4a16] FAIL: GPU 체인 실패 — ${RUN}/b.log"; fail=1;;
  *) echo "[w4a16] FAIL: GPU 골든 불일치: $(echo "$B" | head -c 60)"; fail=1;;
esac

# ── 3. C(27B) serve HTTP 종단 ──
note "[3/7] C(27B) serve HTTP — runtime=cuda + 골든 접두"
: > "$RUN/c.log"
"$BIN" serve --model "$M_27B" --ctx 1024 --slots 1 --port "$PORT" > "$RUN/c.log" 2>&1 &
SERVE_PID=$!
# A2: 기기 전체 pkill 금지 — 이 실행의 서버 PID만 종료(개발 서버 보호).
trap 'kill "$SERVE_PID" 2>/dev/null' EXIT
for i in $(seq 1 100); do
  sleep 3
  grep -m1 "listening" "$RUN/c.log" >/dev/null 2>&1 && break
done
curl -s -m 600 "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d '{"prompt":[148678,65233,202419],"max_tokens":8,"temperature":0}' > "$RUN/c.json"
kill "$SERVE_PID" 2>/dev/null
wait "$SERVE_PID" 2>/dev/null
trap - EXIT
if grep -q "runtime=cuda" "$RUN/c.log"; then note "배너 OK(runtime=cuda)"; else
  echo "[w4a16] FAIL: 배너 runtime!=cuda — ${RUN}/c.log"; fail=1
fi
C=$(python3 -c "
import json
try:
    d=json.load(open('"$RUN/c.json"'))
    t=d.get('tokens') or (d['choices'][0].get('tokens') if 'choices' in d else None)
    print(','.join(map(str,t)) if t else 'ERR')
except Exception:
    print('ERR')
")
case "$C" in
  "$GOLDEN"*) note "serve 토큰 OK: $(echo "$C" | head -c 60)";;
  *) echo "[w4a16] FAIL: serve 응답 불일치: $(echo "$C" | head -c 60)"; fail=1;;
esac

# ── 4. D(35B INT4 g32) 참조 오라클(W4-1 MoE) ──
note "[4/7] D(35B INT4 g32) 참조 실행 — w4a16-ref (MoE g32/bf16)"
timeout 2400 "$BIN" w4a16-ref "$M_35B" --prompt-tokens 148678,65233,202419 --n-predict 8 --ctx 1024 \
  > "$RUN/d.out" 2> "$RUN/d.log"
D=$(grep -m1 '^tokens:' "$RUN/d.out" | sed 's/^tokens: //')
case "$D" in
  "$GOLDEN35"*) note "35B 참조 토큰 OK: $(echo "$D" | head -c 60)";;
  "") echo "[w4a16] FAIL: 35B 참조 실패 — ${RUN}/d.log"; fail=1;;
  *) echo "[w4a16] FAIL: 35B 골든 불일치: $(echo "$D" | head -c 60)"; fail=1;;
esac

# ── 5. D2(35B INT4 g32) GPU 체인(MoE) ──
note "[5/7] D2(35B) GPU 체인 — w4a16-gpu (MoE)"
timeout 1800 "$BIN" w4a16-gpu "$M_35B" --prompt-tokens 148678,65233,202419 --n-predict 8 --ctx 1024 \
  > "$RUN/d2.out" 2> "$RUN/d2.log"
D2=$(grep -m1 '^ tokens:' "$RUN/d2.out" | sed 's/^ tokens: //')
case "$D2" in
  "$GOLDEN35"*) note "35B GPU 토큰 OK: $(echo "$D2" | head -c 60)";;
  "") echo "[w4a16] FAIL: 35B GPU 실패 — ${RUN}/d2.log"; fail=1;;
  *) echo "[w4a16] FAIL: 35B GPU 골든 불일치: $(echo "$D2" | head -c 60)"; fail=1;;
esac

# ── 6. D3(35B) 다중 청크(35토큰) — 배치 프리필 t≤32 CPU/GPU 골든 ──
note "[6/7] D3(35B) 다중 청크 — 참조(CPU) + GPU (배치 t≤32)"
timeout 2400 "$BIN" w4a16-ref "$M_35B" --prompt-tokens "$P35L" --n-predict 8 --ctx 1024 \
  > "$RUN/d3a.out" 2> "$RUN/d3a.log"
DA=$(grep -m1 '^tokens:' "$RUN/d3a.out" | sed 's/^tokens: //')
case "$DA" in
  "$GOLDEN35L"*) note "긴 프롬프트 참조 OK: $(echo "$DA" | head -c 50)";;
  "") echo "[w4a16] FAIL: 긴 참조 실패 — ${RUN}/d3a.log"; fail=1;;
  *) echo "[w4a16] FAIL: 긴 참조 골든 불일치: $(echo "$DA" | head -c 50)"; fail=1;;
esac
timeout 900 "$BIN" w4a16-gpu "$M_35B" --prompt-tokens "$P35L" --n-predict 8 --ctx 1024 \
  > "$RUN/d3b.out" 2> "$RUN/d3b.log"
DB=$(grep -m1 '^ tokens:' "$RUN/d3b.out" | sed 's/^ tokens: //')
case "$DB" in
  "$GOLDEN35L"*) note "긴 프롬프트 GPU OK: $(echo "$DB" | head -c 50)";;
  "") echo "[w4a16] FAIL: 긴 GPU 실패 — ${RUN}/d3b.log"; fail=1;;
  *) echo "[w4a16] FAIL: 긴 GPU 골든 불일치: $(echo "$DB" | head -c 50)"; fail=1;;
esac

# ── 7. E(FN FP8PLE) 자원 가드 명시 거부 ──
note "[7/7] E(FN FP8PLE) 자원 가드 명시 거부 판정"
if timeout 60 "$BIN" w4a16-load "$M_FN" > "$RUN/e.log" 2>&1; then
  echo "[w4a16] FAIL: 가드가 통과시킴(120GiB) — ${RUN}/e.log"; fail=1
elif grep -qE "insufficient resources|rsrc-guard" "$RUN/e.log"; then
  note "가드 거부 OK: $(head -1 "$RUN/e.log")"
else
  echo "[w4a16] FAIL: 가드 외 사유 — ${RUN}/e.log"; fail=1
fi

if [ $fail -eq 0 ]; then
  echo "[w4a16] smoke: ALL PASS (27B 참조·GPU·serve / 35B MoE 참조+GPU 골든 / E 가드 거부)"
else
  echo "[w4a16] smoke: FAILURES PRESENT"
fi
exit $fail
