#!/usr/bin/env bash
set -u
BIN=${BIN:-target/release/llm170}
M=${1:-../models/Qwen3.8-27B-W4A16-AutoRound}
PORT=${2:-18097}
KVMEM=${3:-64M}
RUN=${RUN:-/tmp/opencode/admit-$(date +%H%M%S)}
mkdir -p "$RUN"
fail=0
note() { echo "  [admit] $*"; }
req() {
  curl -s -m 900 "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"prompt\":[148678,65233,202419],\"max_tokens\":$1,\"temperature\":0}"
}
stats() { curl -s "http://127.0.0.1:$PORT/stats"; }
jget() { python3 -c "import json,sys; d=json.load(sys.stdin); print($1)"; }

LLM170_AGING_D=0.5 "$BIN" serve --model "$M" --kv-mem "$KVMEM" --port "$PORT" \
  > "$RUN/s.log" 2>&1 &
PID=$!
trap 'kill "$PID" 2>/dev/null' EXIT
for _ in $(seq 1 100); do
  sleep 3
  grep -qm1 "listening" "$RUN/s.log" && break
done
echo "=== admission harness (kv-mem $KVMEM · D=0.5s 고정) ==="

grep -q "# kv-mem:" "$RUN/s.log" \
  && note "로그 OK: $(grep -m1 '# kv-mem:' "$RUN/s.log")" \
  || { echo "[admit] FAIL: kv-mem 기동 로그 없음 — ${RUN}/s.log"; fail=1; }
grep -q "# effective cap:" "$RUN/s.log" \
  && note "cap 로그 OK: $(grep -m1 '# effective cap:' "$RUN/s.log")" \
  || { echo "[admit] FAIL: effective cap 로그 없음"; fail=1; }
CAP=$(stats | jget 'd["serving"]["pool_cap_tokens"]')
[ "${CAP:-0}" -ge 1024 ] \
  && note "pool cap = $CAP tokens · kv_mem_bytes = $(stats | jget 'd["serving"]["kv_mem_bytes"]')" \
  || { echo "[admit] FAIL: pool cap=$CAP"; fail=1; }
D=$(stats | jget 'd["serving"]["aging"]["d_ms"]')
for _ in $(seq 1 20); do
  [ "$D" = "500" ] && break
  sleep 0.5
  D=$(stats | jget 'd["serving"]["aging"]["d_ms"]')
done
[ "$D" = "500" ] \
  && note "aging D 고정 500ms OK(적응 OFF)" \
  || { echo "[admit] FAIL: d_ms=$D (LLM170_AGING_D=0.5 기대)"; fail=1; }

R=$(req $((CAP + 100)))
case "$R" in
  *context_length_exceeded* | *exceeds\ the\ KV\ pool*) note "예약 초과 거부 OK";;
  *) echo "[admit] FAIL: 초과가 거부되지 않음: $(echo "$R" | head -c 160)"; fail=1;;
esac
R=$(req 300000)
grep -q "max_tokens 클램프: 300000 → 262133" "$RUN/s.log" \
  && note "출력 클램프 OK(300000 → 262133 = 모델 최대−프롬프트−슬랙)" \
  || { echo "[admit] FAIL: 클램프 로그 없음(모델 몫 계약)"; fail=1; }
case "$R" in
  *exceeds\ the\ KV\ pool*) note "클램프 후 풀 초과 거부 OK(풀 몫 계약)";;
  *) echo "[admit] FAIL: 클램프 후 풀 거부 아님: $(echo "$R" | head -c 160)"; fail=1;;
esac

TGB=$(((CAP * 55 / 100) - 11))
req "$TGB" > "$RUN/b1.json" &
P1=$!
sleep 1
req "$TGB" > "$RUN/b2.json" &
P2=$!
FROZEN_MAX=0
for _ in $(seq 1 30); do
  sleep 0.5
  F=$(stats | jget 'd["serving"]["aging"]["frozen"]' 2>/dev/null || echo 0)
  if [ "$F" = "1" ]; then
    FROZEN_MAX=1
    break
  fi
done
[ "$FROZEN_MAX" = "1" ] \
  && note "동결 관측 OK(선두 대형 대기)" \
  || { echo "[admit] FAIL: 동결 미발화(폴링 15s)"; fail=1; }
req 4 > "$RUN/sm.json" &
P3=$!
sleep 1.5
BLOCKED=$(stats | jget 'd["serving"]["aging"]["blocked_jobs"]' 2>/dev/null || echo 0)
[ "${BLOCKED:-0}" -ge 1 ] \
  && note "백필 차단 계측 OK(blocked=$BLOCKED건)" \
  || { echo "[admit] FAIL: blocked=$BLOCKED — 차단 미계측"; fail=1; }

wait "$P1" "$P2" "$P3"
for f in b1 b2 sm; do
  grep -q '"text_completion"' "$RUN/$f.json" \
    && note "$f 완주 ($(python3 -c "import json;d=json.load(open('$RUN/$f.json'));print(d['usage']['completion_tokens'],'tok',d['choices'][0]['finish_reason'])"))" \
    || { echo "[admit] FAIL: $f 미완주: $(head -c 120 "$RUN/$f.json")"; fail=1; }
done
stats > "$RUN/final-stats.json"
python3 - "$RUN/final-stats.json" <<'EOF' || fail=1
import json, sys
s = json.load(open(sys.argv[1]))["serving"]
a = s["aging"]
ok = True
if a["frozen"] != 0:
    print(f"  [admit] FAIL: frozen 잔존 {a}")
    ok = False
if s["requests_failed"] != 0:
    print(f"  [admit] FAIL: requests_failed={s['requests_failed']}")
    ok = False
if a["events"] < 1:
    print(f"  [admit] FAIL: 동결 이벤트 0 {a}")
    ok = False
if ok:
    print(f"  [admit] 해제·전건 완주 OK — 동결 {a['events']}회·차단 {a['blocked_jobs']}건"
          f"·차단토큰 {a['blocked_tokens']}·requests {s['requests']}·실패 {s['requests_failed']}")
sys.exit(0 if ok else 1)
EOF

kill "$PID" 2>/dev/null
wait "$PID" 2>/dev/null
trap - EXIT
if [ "$fail" = 0 ]; then
  echo "[admit] ALL PASS — 거부·동결·차단·해제·완주 (${RUN})"
else
  echo "[admit] FAIL — ${RUN}"
fi
exit "$fail"
