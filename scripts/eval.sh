#!/usr/bin/env bash
# 통계 eval 러너 — 경로 편차 계측(기준선/후보). 사용:
#   scripts/eval.sh <model_dir> <corpus> <tag> [추가 플래그...]
# 덤프는 /tmp/opencode/eval-<tag>.tsv — agree로 두 태그 비교.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ $# -lt 3 ]; then
    echo "사용: scripts/eval.sh <model_dir> <corpus> <tag> [--limit N ...]" >&2
    exit 1
fi
model=$1; corpus=$2; tag=$3; shift 3
./target/release/llm170 w4a16-eval ppl "$model" --corpus "$corpus" --ctx 16384 --out "/tmp/opencode/eval-$tag.tsv" "$@"
