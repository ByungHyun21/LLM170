#!/usr/bin/env bash
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
