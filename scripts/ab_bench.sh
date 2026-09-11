#!/usr/bin/env bash
# A/B 벤치 — 두 바이너리를 교차 실행해 열 드리프트를 상쇄한다.
# 사용: scripts/ab_bench.sh <binA> <binB> <reps> <bench args...>
# 출력: 각 바이너리의 pp/tg 중앙값과 B/A 비율.
set -euo pipefail
A="$1"; B="$2"; REPS="$3"; shift 3
MODEL="${MODEL:-$HOME/models/qwen3.8-27b/q35work.gguf}"
ARGS=("$@")
if [ ${#ARGS[@]} -eq 0 ]; then ARGS=(--pp 512 --tg 32); fi

run() {
  local bin="$1"
  timeout 900 "$bin" bench --model "$MODEL" --gpu-runtime hip --backend gpu --reps 1 "${ARGS[@]}" 2>/dev/null \
    | awk -v tag="$bin" '
      /pp[0-9]+/ { for(i=1;i<=NF;i++) if($i=="t/s"){pp=$(i-1); break} }
      /tg[0-9]+/ { for(i=1;i<=NF;i++) if($i=="t/s"){tg=$(i-1); break}; printf "%s %s %s\n", tag, pp, tg }'
}

: > /tmp/ab_out.txt
for _ in $(seq 1 "$REPS"); do
  run "$A" >> /tmp/ab_out.txt
  run "$B" >> /tmp/ab_out.txt
done
python3 - "$A" "$B" <<'PY'
import sys, statistics, collections
a, b = sys.argv[1], sys.argv[2]
pp = collections.defaultdict(list); tg = collections.defaultdict(list)
for ln in open('/tmp/ab_out.txt'):
    p = ln.split()
    if len(p) != 3: continue
    tag, x, y = p[0], float(p[1]), float(p[2])
    pp[tag].append(x); tg[tag].append(y)
def med(v): return statistics.median(v) if v else float('nan')
print(f"  pp: A={med(pp[a]):.1f} B={med(pp[b]):.1f}  ratio B/A={med(pp[b])/med(pp[a]):.4f}  (n={len(pp[a])}/{len(pp[b])})")
print(f"  tg: A={med(tg[a]):.2f} B={med(tg[b]):.2f}  ratio B/A={med(tg[b])/med(tg[a]):.4f}")
print(f"  A pp={sorted(pp[a])}")
print(f"  A tg={sorted(tg[a])}")
print(f"  B pp={sorted(pp[b])}")
print(f"  B tg={sorted(tg[b])}")
PY
