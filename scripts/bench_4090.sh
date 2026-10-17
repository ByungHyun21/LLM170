#!/usr/bin/env bash
# 4090 표준 배터리 — 4090기 전용(sm_89 행 라벨 — CMP에서는 실행 대상 아님).
# 구 원장 benchmark/4090.md는 폐지(2026-10-16; 역사 = git 이력).
# 표 행을 **그대로 출력**한다(측정 1회 = 표 갱신).
# 규율(사용자 지시 2026-10-13): 스크립트와 표의 측정 세트를 일치시켜 이중 측정을 없앤다.
#
# 사용: scripts/bench_4090.sh [--long] [--reps N]
#   기본(짧은 창): pp 512,1024,4096 + tg128 각 위치 + np4 — 표:
#     | backend | pp512 | pp1024 | pp4096 | tg128@4k | np4 |
#   --long(긴 창): pp 16384,32768,65536 + tg128 각 위치(ctx 69632) — 표:
#     | backend | pp16384 | pp32768 | pp65536 | tg@16k | tg@32k | tg@64k |
#   모델 = ../models 아래 27B·35B(환경변수 MODEL_27B/MODEL_35B로 교체 가능).
# 출력 마지막 줄에 커밋 해시 제안을 인쇄한다(구 4090.md Environment 표의 commit 행).
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=${BIN:-./target/release/llm170}
M27=${MODEL_27B:-../models/Qwen3.8-27B-W4A16-AutoRound}
M35=${MODEL_35B:-../models/Qwen3.6-35B-A3B-INT4-W4A16}
REPS=2
LONG=0
while [ $# -gt 0 ]; do
    case "$1" in
        --long) LONG=1 ;;
        --reps)
            shift
            REPS=$1
            ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
    shift
done
DATE=$(date +%F)
TMP=$(mktemp -d /tmp/opencode/bench4090.XXXXXX)
trap 'rm -rf "$TMP"' EXIT

row() { # 파일 [--long]
    python3 - "$1" "$2" "$DATE" <<'PY'
import re, sys
path, mode, date = sys.argv[1], sys.argv[2], sys.argv[3]
pp, tg, np4 = {}, {}, None
for line in open(path):
    m = re.match(r'pp(\d+) \| rep\d+ \| *[\d.]+ ms \| *([\d.]+) t/s', line)
    if m:
        pp[int(m.group(1))] = m.group(2)
    m = re.match(r'tg\d+@pp(\d+) \| rep\d+ \| *[\d.]+ ms \| *([\d.]+) t/s', line)
    if m:
        tg[int(m.group(1))] = m.group(2)
    m = re.match(r'np(\d+)\(tg\d+\) \| .* \| *([\d.]+) t/s', line)
    if m:
        np4 = m.group(2)
if mode == 'long':
    cells = [pp.get(16384, '-'), pp.get(32768, '-'), pp.get(65536, '-'),
             tg.get(16384, '-'), tg.get(32768, '-'), tg.get(65536, '-')]
    print('| CUDA (sm_89, ' + date + ') | ' + ' | '.join(cells) + ' |')
else:
    cells = [pp.get(512, '-'), pp.get(1024, '-'), pp.get(4096, '-'),
             tg.get(4096, '-'), np4 or '-']
    print('| CUDA (sm_89, ' + date + ') | ' + ' | '.join(cells) + ' |')
PY
}

gpu_busy() { # idle 사용률 중앙값(%) — 데스크톱 점유(Xorg·nautilus·tracker 등) 탐지
    local v=()
    for _ in 1 2 3 4 5; do
        v+=("$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)")
        sleep 0.4
    done
    printf '%s\n' "${v[@]}" | sort -n | sed -n 3p
}

run_model() { # 이름 모델경로
    local name=$1 model=$2
    pkill -x llm170 2>/dev/null || true
    sleep 2
    # [점유 가드 2026-10-14] 데스크톱 GPU 점유 실측 사례: idle 30~37%(nautilus·
    # tracker·Xorg)에서 전 축 −8~12% 오염(A/B로 코드 무죄 확인). ≥10%면 경고 —
    # 청정 수치가 필요하면 데스크톱을 조용히 한 뒤 재실행.
    local busy
    busy=$(gpu_busy)
    echo "# GPU idle util ${busy}%$([ "${busy:-0}" -ge 10 ] && echo ' — 경고: 데스크톱 점유 의심(수치 오염 가능)' || echo ' (청정)')" >&2
    if [ "$LONG" = 1 ]; then
        echo "## $name"
        echo
        echo "| backend | pp16384 | pp32768 | pp65536 | tg@16k | tg@32k | tg@64k |"
        echo "|---|---|---|---|---|---|---|"
        timeout 3600 "$BIN" bench --model "$model" --pp 16384,32768,65536 --tg 128 \
            --ctx 69632 --reps "$REPS" > "$TMP/long.out" 2> "$TMP/long.err" ||
            { echo "bench 실패: $name --long — $TMP/long.err" >&2; tail -3 "$TMP/long.err" >&2; exit 1; }
        row "$TMP/long.out" long
    else
        echo "## $name"
        echo
        echo "| backend | pp512 | pp1024 | pp4096 | tg128@4k | np4 |"
        echo "|---|---|---|---|---|---|"
        timeout 3600 "$BIN" bench --model "$model" --pp 512,1024,4096 --tg 128 --np 4 \
            --ctx 8192 --reps "$REPS" > "$TMP/short.out" 2> "$TMP/short.err" ||
            { echo "bench 실패: $name — $TMP/short.err" >&2; tail -3 "$TMP/short.err" >&2; exit 1; }
        row "$TMP/short.out" short
    fi
    echo
}

run_model "Qwen3.8-27B-W4A16 (g128)" "$M27"
run_model "Qwen3.6-35B-A3B-W4A16 (g32 MoE)" "$M35"
echo "commit(LLM170 rows): $(git rev-parse --short HEAD) — 4090.md Environment 표에 반영"
