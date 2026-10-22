#!/usr/bin/env bash
set -u
BIN=${BIN:-target/release/llm170}
M=${1:-../models/Qwen3.8-27B-W4A16-AutoRound}
PORT=${2:-18095}
KV=${3:-1G}
RUN=${RUN:-/tmp/opencode/ttft-$(date +%H%M%S)}
mkdir -p "$RUN"
note() { echo "  [ttft] $*"; }

python3 - "$RUN" <<'PYBODY'
import json, sys
run = sys.argv[1]
def lcg_list(seed, n):
    x = seed
    out = []
    for _ in range(n):
        x = (x * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        out.append(x % 200000)
    return out
def w(name, toks):
    json.dump({"prompt": toks, "max_tokens": 16, "temperature": 0},
              open(f"{run}/{name}.json", "w"))
P = lcg_list(0x1234_5678, 700)
w("s1_cold",  lcg_list(0xDEAD_BEEF, 700))
w("s2_full",  P + [11, 22, 33])
w("s3_ckpt",  P + [44, 55, 66, 77])
for c in (2, 4, 8):
    for i in range(c):
        if i % 2 == 0:
            w(f"c{c}_{i}", P + [c, i])
        else:
            w(f"c{c}_{i}", lcg_list(0xC0FFEE + c * 1000 + i, 700))
PYBODY

"$BIN" serve --model "$M" --kv-mem "$KV" --port "$PORT" > "$RUN/s.log" 2>&1 &
PID=$!
snap() { curl -s "http://127.0.0.1:$PORT/stats" | python3 -c '
import json,sys
d=json.load(sys.stdin)
s=d.get("serving",{})
print(s.get("requests",0), s.get("ttft_ms_avg",0), s.get("queue_wait_ms_avg",0))'; }

for _ in $(seq 1 100); do sleep 3; grep -qm1 listening "$RUN/s.log" && break; done
grep -qm1 listening "$RUN/s.log" || { echo "serve 기동 실패"; exit 1; }

scn() {
  local A B
  A=$(snap); curl -s -X POST "http://127.0.0.1:$PORT/v1/completions" \
      -H 'Content-Type: application/json' -d @"$RUN/$1.json" > /dev/null
  B=$(snap)
  python3 -c "
a='$A'.split(); b='$B'.split()
an,at,aq=map(float,a); bn,bt,bq=map(float,b)
dn=bn-an
ttft=(bt*bn-at*an)/dn if dn>0 and bn>0 else 0
qw=(bq*bn-aq*an)/dn if dn>0 and bn>0 else 0
print(f'$2\t구간 TTFT={ttft:8.1f}ms\t큐대기={qw:6.1f}ms')"
}

echo "=== TTFT 3시나리오 (순차) ==="
scn s1_cold "S1콜드     "
scn s2_full "S2전체접두 "
scn s3_ckpt "S3체크포인트"

echo "=== 동시성 스윕 (혼합: 절반 공유 접두) ==="
for C in 2 4 8; do
  A=$(snap); T0=$(date +%s.%N)
  cpids=""
  for i in $(seq 0 $((C-1))); do
    curl -s -m 300 -X POST "http://127.0.0.1:$PORT/v1/completions" \
      -H 'Content-Type: application/json' -d @"$RUN/c${C}_${i}.json" > /dev/null &
    cpids="$cpids $!"
  done
  wait $cpids
  T1=$(date +%s.%N); B=$(snap)
  python3 -c "
a='$A'.split(); b='$B'.split()
an,at,aq=map(float,a); bn,bt,bq=map(float,b)
dn=bn-an
ttft=(bt*bn-at*an)/dn if dn>0 and bn>0 else 0
qw=(bq*bn-aq*an)/dn if dn>0 and bn>0 else 0
wall=$T1-$T0
print(f'C=$C\t평균 TTFT={ttft:8.1f}ms\t큐대기={qw:6.1f}ms\t완단벽={wall*1000:8.0f}ms')"
done
note "완료 — 원본 $RUN/s.log"
