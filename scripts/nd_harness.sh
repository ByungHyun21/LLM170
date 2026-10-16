#!/usr/bin/env bash
# [비결정성 격리 2026-10-13] serve 반복 하네스 — N서버 × M요청, 토큰 해시 수집.
# 사용: nd_harness.sh <arm-name> <server-env...>
#   예: nd_harness.sh default
#       nd_harness.sh ngram LLM170_SPEC_DRAFT_SOURCE=ngram
#       nd_harness.sh specoff LLM170_SPEC_OFF=1
set -u
exec 2>&1
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BIN=target/release/llm170
M=../models/Qwen3.8-27B-W4A16-AutoRound
ARM="$1"; shift
RUNS="${ND_SERVERS:-3}"
REQS="${ND_REQS:-8}"
OUT="/tmp/opencode/nd/$ARM"
mkdir -p "$OUT"
rm -f "$OUT"/*.json "$OUT"/*.log "$OUT/hashes.tsv"

port=8100
for s in $(seq 1 "$RUNS"); do
  port=$((port + 1))
  env "$@" LLM170_MOE_DBG=1 "$BIN" serve --model "$M" --kv-mem 512M --port "$port" \
    > "$OUT/s$s.log" 2>&1 &
  SPID=$!
  for _ in $(seq 1 60); do
    sleep 2
    grep -q "listening" "$OUT/s$s.log" 2>/dev/null && break
  done
  for r in $(seq 1 "$REQS"); do
    curl -s -m 600 "http://127.0.0.1:$port/v1/completions" -H 'Content-Type: application/json' \
      -d '{"prompt":[148678,65233,202419],"max_tokens":128,"temperature":0}' \
      > "$OUT/s${s}r${r}.json"
  done
  kill "$SPID" 2>/dev/null
  wait "$SPID" 2>/dev/null
  sleep 1
done

python3 - "$OUT" <<'EOF'
import json, glob, sys, hashlib, collections
out = sys.argv[1]
hashes = collections.Counter()
seqs = {}
for f in sorted(glob.glob(f"{out}/s*r*.json")):
    try:
        t = json.load(open(f)).get("tokens") or []
    except Exception:
        t = []
    h = hashlib.sha1(",".join(map(str, t)).encode()).hexdigest()[:10]
    hashes[h] += 1
    seqs.setdefault(h, (f, t))
print(f"[{out}] requests={sum(hashes.values())} distinct={len(hashes)}")
for h, n in hashes.most_common():
    f, _ = seqs[h]
    print(f"  {h} x{n}  (e.g. {f.rsplit('/',1)[-1]})")
if len(hashes) > 1:
    base = seqs[hashes.most_common(1)[0][0]][1]
    for h, _ in hashes.most_common():
        f, t = seqs[h]
        if h == hashes.most_common(1)[0][0]:
            continue
        d = next((i for i, (a, b) in enumerate(zip(base, t)) if a != b), None)
        print(f"  DIFF {h}: first divergent idx {d}  ({f.rsplit('/',1)[-1]})")
        # 그 idx 주변 라운드 마커(해당 서버 로그)
        srv = f.rsplit('/',1)[-1].split('r')[0]
        try:
            lines = open(f"{out}/{srv}.log", errors="replace").read().splitlines()
            dbg = [l for l in lines if l.startswith("[spec-dbg]")]
            print(f"  spec-dbg 마커({srv}, 마지막 4): {dbg[-4:]}")
        except Exception as e:
            print("  log?", e)
EOF
