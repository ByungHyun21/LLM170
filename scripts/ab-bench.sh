#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
if [ $# -lt 3 ]; then
    echo "사용법: $0 <model_dir> <binA> <binB> [bench 인자...]" >&2
    exit 2
fi
MODEL=$1 BIN_A=$2 BIN_B=$3
shift 3
ARGS=("$@")
[ ${#ARGS[@]} -eq 0 ] && ARGS=(--pp 512 --tg 32 --ctx 2048)
REPS=${AB_REPS:-3}
ROUNDS=${AB_ROUNDS:-1}
mkdir -p /tmp/opencode
RUN=$(mktemp -d /tmp/opencode/ab-bench.XXXXXX)

for b in "$BIN_A" "$BIN_B"; do
    [ -x "$b" ] || { echo "ab-bench: 실행 불가 $b" >&2; exit 2; }
    if [ -n "$(strings -a "$b" 2>/dev/null | grep -m1 "llm170 profile")" ]; then
        echo "ab-bench: 거부 — $b 는 profile feature(계측) 포함 빌드(성능 A/B 부적합)" >&2
        exit 2
    fi
done

clk() { nvidia-smi --query-gpu=clocks.sm --format=csv,noheader 2>/dev/null | head -1 | tr -dc '0-9'; }
echo "# ab-bench: A=$BIN_A B=$BIN_B rounds=$ROUNDS reps=$REPS args=${ARGS[*]}"
rc=0
for r in $(seq 1 "$ROUNDS"); do
    echo "# 라운드 $r — 워밍업(양쪽 1회) 후 인프로세스 반복 계측"
    "$BIN_A" bench --model "$MODEL" "${ARGS[@]}" >/dev/null 2>&1
    "$BIN_B" bench --model "$MODEL" "${ARGS[@]}" >/dev/null 2>&1
    c=$(clk)
    timeout 1800 "$BIN_A" bench --model "$MODEL" "${ARGS[@]}" --reps "$REPS" >"$RUN/a$r.out" 2>/dev/null
    ra=$?; [ "$ra" -ne 0 ] && rc=1
    echo "A$r rc=$ra sm=${c}MHz (in-process reps=$REPS)"
    c=$(clk)
    timeout 1800 "$BIN_B" bench --model "$MODEL" "${ARGS[@]}" --reps "$REPS" >"$RUN/b$r.out" 2>/dev/null
    rb=$?; [ "$rb" -ne 0 ] && rc=1
    echo "B$r rc=$rb sm=${c}MHz (in-process reps=$REPS)"
done
python3 - "$RUN" <<'PY'
import glob, statistics, sys
run = sys.argv[1]
def load(pat):
    d = {}
    for f in sorted(glob.glob(f"{run}/{pat}")):
        for line in open(f):
            parts = [x.strip() for x in line.split('|')]
            if len(parts) == 4 and parts[3].endswith('t/s'):
                d.setdefault(parts[0], []).append(float(parts[3].split()[0]))
    return d
a, b = load('a*.out'), load('b*.out')
print(f"{'항목':14s} {'A 중앙값':>9s} {'B 중앙값':>9s} {'델타(B/A)':>9s}  raw")
for k in a:
    if k not in b:
        continue
    ma, mb = statistics.median(a[k]), statistics.median(b[k])
    delta = (mb / ma - 1) * 100 if ma else float('nan')
    print(f"{k:14s} {ma:9.2f} {mb:9.2f} {delta:+8.1f}%  A={a[k]} B={b[k]}")
PY
echo "# 산출: $RUN"
exit $rc
