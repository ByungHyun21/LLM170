#!/usr/bin/env bash
# 통계 eval 러너 — 경로 편차 계측(기준선/후보). 사용:
#   scripts/eval.sh <model_dir> <corpus> <tag> [추가 플래그...]
#   scripts/eval.sh --quick <model_dir> <corpus> <tag>   # 반복용 서브셋(--limit 24)
# [테스트 재설계 2026-10-22] --quick = 반복 탐색용 프리셋일 뿐 — 채택 판정은
# 반드시 전체 코퍼스 실행으로(±3% 이중게이트 계약 변경 없음).
# 덤프는 /tmp/opencode/eval-<tag>.tsv — agree로 두 태그 비교.
set -euo pipefail
cd "$(dirname "$0")/.."
QUICK=""
if [ "${1:-}" = "--quick" ]; then QUICK="--limit 24"; shift; fi
if [ $# -lt 3 ]; then
    echo "사용: scripts/eval.sh [--quick] <model_dir> <corpus> <tag> [--limit N ...]" >&2
    exit 1
fi
model=$1; corpus=$2; tag=$3; shift 3
./target/release/llm170 w4a16-eval ppl "$model" --corpus "$corpus" --ctx 16384 --out "/tmp/opencode/eval-$tag.tsv" $QUICK "$@"
