#!/usr/bin/env bash
# 같은 세션 A/B 성능 하네스 — "수정 전/후를 같은 세션에서" 규율의 도구화.
# (루트 AGENTS.md: 절대치의 세션 간 비교 금지 — 워밍업 후 교차 측정·중앙값만.)
#
# 사용: scripts/ab-bench.sh <model_dir> <binA> <binB> [bench 인자...]
#   예: scripts/ab-bench.sh ../models/Qwen3.8-27B-W4A16-AutoRound \
#         /tmp/baseline/llm170 ./target/release/llm170 --pp 512 --tg 32 --ctx 2048
# 기본 bench 인자 = --pp 512 --tg 32 --ctx 2048 · 반복 = AB_REPS(기본 3).
#
# 가드: (1) 두 바이너리 실행 가능 검사.
#       (2) profile feature 계측 빌드 **거부** — [실측 2026-10-15] 계측 포함
#           베이스라인 바이너리로 디코드 −19% 아티팩트를 "청크 효과"로 오독할
#           뻔했다. 성능 판정용 바이너리는 기본 feature 릴리스 빌드만.
# 출력: 항목별 t/s 중앙값(A/B)과 델타% — 판정(채택/기각)은 사람이 한다.
# 종료코드: 0 = 측정 성립 · 1 = 일부 런 실패 · 2 = 사용법/가드 거부.
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
mkdir -p /tmp/opencode
RUN=$(mktemp -d /tmp/opencode/ab-bench.XXXXXX)

for b in "$BIN_A" "$BIN_B"; do
    [ -x "$b" ] || { echo "ab-bench: 실행 불가 $b" >&2; exit 2; }
    # pipefail 환경에서 `strings | grep -q`는 프로듀서 SIGPIPE(141)를 파이프 실패로
    # 만들어 가드를 무력화한다(실측) — 출력 캡처 비교로 판정.
    if [ -n "$(strings -a "$b" 2>/dev/null | grep -m1 "llm170 profile")" ]; then
        echo "ab-bench: 거부 — $b 는 profile feature(계측) 포함 빌드(성능 A/B 부적합)" >&2
        exit 2
    fi
done

clk() { nvidia-smi --query-gpu=clocks.sm --format=csv,noheader 2>/dev/null | head -1 | tr -dc '0-9'; }
echo "# ab-bench: A=$BIN_A B=$BIN_B reps=$REPS args=${ARGS[*]}"
echo "# 워밍업(양쪽 1회 — 클럭 승압 후 측정)"
"$BIN_A" bench --model "$MODEL" "${ARGS[@]}" >/dev/null 2>&1
"$BIN_B" bench --model "$MODEL" "${ARGS[@]}" >/dev/null 2>&1
rc=0
for i in $(seq 1 "$REPS"); do
    c=$(clk)
    timeout 1800 "$BIN_A" bench --model "$MODEL" "${ARGS[@]}" >"$RUN/a$i.out" 2>/dev/null
    ra=$?; [ "$ra" -ne 0 ] && rc=1
    echo "A$i rc=$ra sm=${c}MHz"
    c=$(clk)
    timeout 1800 "$BIN_B" bench --model "$MODEL" "${ARGS[@]}" >"$RUN/b$i.out" 2>/dev/null
    rb=$?; [ "$rb" -ne 0 ] && rc=1
    echo "B$i rc=$rb sm=${c}MHz"
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
