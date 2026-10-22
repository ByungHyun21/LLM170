#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=${BIN:-./target/release/llm170}
M27=${MODEL_27B:-../models/Qwen3.8-27B-W4A16-AutoRound}
M35=${MODEL_35B:-../models/Qwen3.6-35B-A3B-INT4-W4A16}
MG4=${MODEL_G4:-../models/Gemma-4-26B-A4B-it-QAT-W4A16}
REPS=3
while [ $# -gt 0 ]; do
    case "$1" in
        --reps)
            shift
            REPS=$1
            ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
    shift
done
TMP=$(mktemp -d /tmp/opencode/benchcmp.XXXXXX)

nvidia-smi --query-gpu=clocks.sm,power.draw,clocks_throttle_reasons.active \
    --format=csv,noheader,nounits -l 5 -i 0 > "$TMP/gpu0.csv" 2>/dev/null &
SMPID=$!
cleanup() {
    kill "$SMPID" 2>/dev/null || true
    rm -rf "$TMP"
}
trap cleanup EXIT

echo "# 전력 상한(GPU0): $(nvidia-smi -q -i 0 -d POWER 2>/dev/null | grep -m1 'Power Limit' || echo '조회 불가')" >&2

gpu_busy() {
    local v=()
    for _ in 1 2 3 4 5; do
        v+=("$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0)")
        sleep 0.4
    done
    printf '%s\n' "${v[@]}" | sort -n | sed -n 3p
}

row() {
    python3 - "$1" "$2" "$3" "$4" <<'PY'
import re, sys
model, cards, path, usenp = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
pp, tg, npv = {}, {}, None
for line in open(path):
    m = re.match(r'pp(\d+) \| rep\d+ \| *[\d.]+ ms \| *([\d.]+) t/s', line)
    if m:
        pp[int(m.group(1))] = m.group(2)
    m = re.match(r'tg\d+@pp(\d+) \| rep\d+ \| *[\d.]+ ms \| *([\d.]+) t/s', line)
    if m:
        tg[int(m.group(1))] = m.group(2)
    m = re.match(r'np(\d+)\(tg\d+\) \| .* \| *([\d.]+) t/s', line)
    if m:
        npv = m.group(2)
cells = [pp.get(512, '—'), pp.get(1024, '—'), pp.get(4096, '—'),
         tg.get(4096, '—'), npv if usenp == '1' else '—']
print(f'| {model} | {cards} | ' + ' | '.join(cells) + ' |')
PY
}

run_model() {
    local label=$1 cards=$2 model=$3 np=$4
    pkill -x llm170 2>/dev/null || true
    sleep 2
    local busy
    busy=$(gpu_busy)
    echo "# $label — GPU idle util ${busy}%$([ "${busy:-0}" -ge 10 ] && echo ' — 경고: 점유 의심(수치 오염 가능)' || echo ' (청정)')" >&2
    local npflag=()
    if [ "$np" = 1 ]; then npflag=(--np 4); fi
    timeout 3600 "$BIN" bench --model "$model" --pp 512,1024,4096 --tg 128 \
        "${npflag[@]}" --ctx 8192 --reps "$REPS" > "$TMP/out.txt" 2> "$TMP/err.txt" || {
        echo "bench 실패: $label — $(tail -2 "$TMP/err.txt" | tr '\n' ' ')" >&2
        return 1
    }
    row "$label" "$cards" "$TMP/out.txt" "$np"
}

echo "# measured $(date -u +%FT%TZ) · $(git rev-parse --short HEAD)" >&2
echo "| model | cards | pp512 | pp1024 | pp4096 | tg128@4k | np4 |"
echo "|---|---|---|---|---|---|---|"
run_model "Qwen3.8-27B-W4A16" 1 "$M27" 1
run_model "Qwen3.6-35B-A3B-W4A16" 1 "$M35" 1
run_model "Gemma-4-26B-A4B-it-QAT-W4A16" 1 "$MG4" 0

kill "$SMPID" 2>/dev/null || true
sleep 1
echo "== GPU0 계측(5s 샘플 — 참고):"
python3 - "$TMP/gpu0.csv" <<'PY'
import sys
rows = [l.strip().split(', ') for l in open(sys.argv[1]) if l.strip()]
if not rows:
    print('(샘플 없음)')
    sys.exit()
clk = sorted(int(r[0]) for r in rows if r[0].isdigit())
pw = [float(r[1]) for r in rows if len(r) > 1 and r[1].replace('.', '', 1).isdigit()]
th = [r[2] for r in rows if len(r) > 2]

def med(v):
    return v[len(v) // 2] if v else 0

non0 = sum(1 for t in th if t not in ('0x0000000000000000', 'Not Active', '0x0'))
print(f"SM클럭 min/med/max = {min(clk) if clk else 0}/{med(clk)}/{max(clk) if clk else 0} MHz"
      f" · 전력 med = {med(pw):.1f} W · 스로틀 샘플 {non0}/{len(th)}")
PY
echo "commit(LLM170 rows): $(git rev-parse --short HEAD)"
