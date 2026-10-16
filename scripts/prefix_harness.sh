#!/usr/bin/env bash
# (2026-10-16) 접두/체크포인트 재사용 하네스 — 정책 교체(FLOP-aware 퇴거)의
# before/after 대조와 무손실(토큰 동일) 판정.
# 시나리오: 고정 시스템 프롬프트(P)를 공유하는 다턴 요청 N건 — 512 경계를
# 넘겨 체크포인트(512/1024/1536/2048)와 슬롯 전체-접두 재사용을 모두 태운다.
# 수집: /stats prefix_reuse_tokens/ratio/full/ckpt · ttft · requests_failed ·
#       요청별 토큰 해시(정책 교체가 토큰을 바꾸면 즉시 검출).
# 사용: scripts/prefix_harness.sh [모델] [포트] [--kv-mem 값]
set -u
BIN=${BIN:-target/release/llm170}
M=${1:-../models/Qwen3.8-27B-W4A16-AutoRound}
PORT=${2:-18094}
KV=${3:-512M}
RUN=${RUN:-/tmp/opencode/prefix-$(date +%H%M%S)}
mkdir -p "$RUN"
fail=0
note() { echo "  [prefix] $*"; }
jget() { python3 -c "import json,sys; d=json.load(sys.stdin); print($1)"; }

# 결정론 토큰(LCG) — 시스템 프롬프트 700토큰(512 경계 초과) + 턴 접미사.
python3 - "$RUN" <<'PYBODY'
import json, sys
run = sys.argv[1]
x = 0x1234_5678
P = []
for _ in range(700):
    x = (x * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    P.append(x % 200000)          # vocab 안전역(27B)
def req(name, toks, mt):
    json.dump({"prompt": toks, "max_tokens": mt, "temperature": 0},
              open(f"{run}/{name}.json", "w"))
req("r1", P + [11, 22, 33], 8)     # 프롬프트 703 → 512 경계 1개
req("r2", P + [44], 8)             # 접두 700 재사용 후보(체크포인트 ≤ 700)
req("r3", P, 8)                    # 정확히 700 — 전체/체크포인트
req("r4", P + [55, 66, 77, 88, 99, 111, 222, 333, 444, 555, 666, 777], 8)
req("r5", P + [11, 22, 33], 8)     # r1과 동일 — 전체 접두 재사용 기대
PYBODY

"$BIN" serve --model "$M" --kv-mem "$KV" --port "$PORT" > "$RUN/s.log" 2>&1 &
PID=$!
trap 'kill "$PID" 2>/dev/null' EXIT
for _ in $(seq 1 100); do
  sleep 3
  grep -qm1 "listening" "$RUN/s.log" && break
done
echo "=== prefix harness ($M · kv-mem $KV) ==="

for r in r1 r2 r3 r4 r5; do
  curl -s -m 600 "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
    --data-binary "@$RUN/$r.json" > "$RUN/$r.out"
  grep -q '"text_completion"' "$RUN/$r.out" \
    || { echo "[prefix] FAIL: $r 미완주: $(head -c 120 "$RUN/$r.out")"; fail=1; }
done
# r1 vs r5 = 동일 요청 — 토큰 동일(정책 교체 무손실의 자체 검증).
# (응답의 id/created는 요청별로 다르므로 tokens 배열만 비교.)
H1=$(python3 -c "import json,hashlib;d=json.load(open('$RUN/r1.out'));print(hashlib.sha256(str(d.get('tokens')).encode()).hexdigest()[:16])")
H5=$(python3 -c "import json,hashlib;d=json.load(open('$RUN/r5.out'));print(hashlib.sha256(str(d.get('tokens')).encode()).hexdigest()[:16])")
[ "$H1" = "$H5" ] && note "동일 요청 토큰 동일 OK ($H1)" \
  || { echo "[prefix] FAIL: 동일 요청 불일치 $H1 vs $H5"; fail=1; }

S=$(curl -s "http://127.0.0.1:$PORT/stats")
PT=$(echo "$S" | jget 'd["serving"]["prefix_reuse_tokens"]')
PR=$(echo "$S" | jget 'd["serving"]["prefix_reuse_ratio"]')
PF=$(echo "$S" | jget 'd["serving"]["prefix_reuse_full"]')
PC=$(echo "$S" | jget 'd["serving"]["prefix_reuse_ckpt"]')
TT=$(echo "$S" | jget 'd["serving"]["ttft_ms_avg"]')
QF=$(echo "$S" | jget 'd["serving"]["requests_failed"]')
note "PREFIX tokens=$PT ratio=$PR full=$PF ckpt=$PC ttft_ms=$TT failed=$QF"
[ "${PT:-0}" -gt 0 ] && note "재사용 발생 OK" || { echo "[prefix] FAIL: 재사용 0"; fail=1; }
[ "$QF" = "0" ] || { echo "[prefix] FAIL: failed=$QF"; fail=1; }

kill "$PID" 2>/dev/null
wait "$PID" 2>/dev/null
trap - EXIT
if [ "$fail" = 0 ]; then
  echo "[prefix] ALL PASS — 재사용 계측·동일요청 무손실 (${RUN})"
  echo "PREFIX_BASELINE tokens=$PT ratio=$PR full=$PF ckpt=$PC"
else
  echo "[prefix] FAIL — ${RUN}"
fi
exit "$fail"
