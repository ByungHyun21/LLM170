#!/usr/bin/env bash
set -u
BIN=${BIN:-target/release/llm170}
M=${1:-../models/Qwen3.8-27B-W4A16-AutoRound}
PORT=${2:-18096}
KV=${3:-256M}
RUN=${RUN:-/tmp/opencode/kvtier-$(date +%H%M%S)}
mkdir -p "$RUN"
fail=0
note() { echo "  [tier] $*"; }
jget() { python3 -c "import json,sys; d=json.load(sys.stdin); print($1)"; }

LLM170_KV_VRAM_MB=16 "$BIN" serve --model "$M" --kv-mem "$KV" --port "$PORT" \
  > "$RUN/s.log" 2>&1 &
PID=$!
trap 'kill "$PID" 2>/dev/null' EXIT
for _ in $(seq 1 100); do
  sleep 3
  grep -qm1 "listening" "$RUN/s.log" && break
done
echo "=== kv tier harness (kv-mem $KV · VRAM부 16M 강제) ==="

grep -m1 "# kv-mem 계층:" "$RUN/s.log" >/dev/null \
  && note "기동 계층 로그 OK: $(grep -m1 '# kv-mem 계층:' "$RUN/s.log")" \
  || note "kv-mem 총량 ≤ VRAM 상한 — 서버측 분할 로그 없음(백엔드 강제 분할만: 아래 kv-tier)"
TIER=$(grep -m1 "# kv-tier:" "$RUN/s.log")
[ -n "$TIER" ] && note "$TIER" || { echo "[tier] FAIL: kv-tier 실할당 로그 없음"; fail=1; }

HOST=0
for _ in $(seq 1 20); do
  S=$(curl -s "http://127.0.0.1:$PORT/stats")
  HOST=$(echo "$S" | jget 'd["memory"]["kv_cache"]["host"]' 2>/dev/null || echo 0)
  [ "${HOST:-0}" -gt 0 ] && break
  sleep 0.5
done
RATIO=$(echo "$S" | jget 'd["serving"]["kv_offload_ratio"]')
TOT=$(echo "$S" | jget 'd["memory"]["kv_cache"]["total_blocks"]')
HB=$(echo "$S" | jget 'd["memory"]["kv_cache"]["host_blocks"]')
FV=$(echo "$S" | jget 'd["memory"]["kv_cache"]["free_vram_blocks"]')
FH=$(echo "$S" | jget 'd["memory"]["kv_cache"]["free_host_blocks"]')
[ "${FV:-x}" != "x" ] && [ "${FH:-x}" != "x" ] \
  && note "티어 자유 블록 노출 OK — 자유 VRAM $FV/호스트 $FH" \
  || { echo "[tier] FAIL: free 블록 필드 부재"; fail=1; }
ADV=$(echo "$S" | jget 'd["serving"]["mem_advice"]')
case "$ADV" in
  hold | prefer_kv | prefer_cache) note "배치 권고 필드 OK ($ADV)";;
  *) echo "[tier] FAIL: mem_advice=$ADV"; fail=1;;
esac
[ "${HOST:-0}" -gt 0 ] \
  && note "호스트 아레나 $HOST B($HB 블록) · 전체 $TOT 블록 · 오프로드율 $RATIO" \
  || { echo "[tier] FAIL: host=$HOST"; fail=1; }
python3 -c "import sys; sys.exit(0 if float('$RATIO') > 0.5 else 1)" \
  && note "오프로드율 >50% OK" \
  || { echo "[tier] FAIL: ratio=$RATIO"; fail=1; }

R=$(curl -s -m 600 "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d '{"prompt":[148678,65233,202419],"max_tokens":8,"temperature":0}')
TOKS=$(echo "$R" | jget '",".join(map(str,d.get("tokens") or []))')
case "$TOKS" in
  156037,16072,154029,209495,30*)
    note "골든 토큰 OK(호스트 KV 경유): $(echo "$TOKS" | head -c 48)";;
  *)
    echo "[tier] FAIL: 토큰 불일치: $(echo "$TOKS" | head -c 80)"; fail=1;;
esac

R2=$(curl -s -m 600 "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d '{"prompt":[148678,65233,202419],"max_tokens":1000,"temperature":0}')
echo "$R2" | grep -q '"text_completion"' \
  && note "대형 요청 완주 OK ($(echo "$R2" | jget 'd["usage"]["completion_tokens"]')토큰)" \
  || { echo "[tier] FAIL: 대형 미완주: $(echo "$R2" | head -c 120)"; fail=1; }
F=$(curl -s "http://127.0.0.1:$PORT/stats" | jget 'd["serving"]["requests_failed"]')
[ "$F" = "0" ] && note "requests_failed=0 OK" || { echo "[tier] FAIL: failed=$F"; fail=1; }

kill "$PID" 2>/dev/null
wait "$PID" 2>/dev/null
trap - EXIT
if [ "$fail" = 0 ]; then
  echo "[tier] ALL PASS — 호스트 확장·계측·완주·골든 (${RUN})"
else
  echo "[tier] FAIL — ${RUN}"
fi
exit "$fail"
