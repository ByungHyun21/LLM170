#!/usr/bin/env bash
# Qwen3.8-27B W4A16(AutoRound·g128·CT pack-quantized) 다운로드 — CMP 셋업 2026-10-16.
# 소스: born2bewild/Qwen3.8-27B-W4A16-AutoRound-fast — 지문 대조로 특정(g128 sym·14.7GiB
#   = 4.125bpw 계산치 14.9G; 4090 원장 §"Qwen3.8-27B-W4A16 (g128, 4.125 bpw)" 정합).
# 대상: ../models/Qwen3.8-27B-W4A16-AutoRound (swap_models.sh M_27B 경로 동일)
# 규칙: 모델은 한 번에 1개씩(사용자 지시 2026-10-16). 재실행 = 완료 파일 스킵 + 이어받기.
#   의존 없음(curl+python3 stdlib만 — PEP668 환경에서 pip 불가).
set -euo pipefail
REPO="born2bewild/Qwen3.8-27B-W4A16-AutoRound-fast"
NAME="Qwen3.8-27B-W4A16-AutoRound"
BASE="$(cd "$(dirname "$0")/.." && pwd)"
DST="$BASE/../models/$NAME"
mkdir -p "$DST"
LISTF="/tmp/dl_list_${NAME//\//_}"
curl -fsSL --retry 5 "https://huggingface.co/api/models/$REPO/tree/main?recursive=true" \
  | python3 -c 'import json,sys
for x in json.load(sys.stdin):
    if x.get("type")=="file": print(x["size"], x["path"])' > "$LISTF"
fail=0
while read -r size path; do
  dst="$DST/$path"
  if [[ -f "$dst" ]] && (( $(stat -c%s "$dst") == size )); then echo "skip  $path"; continue; fi
  mkdir -p "$(dirname "$dst")"
  echo "get   $path ($((size>>20)) MiB)"
  curl -fL -C - --retry 8 --retry-delay 3 --retry-all-errors -o "$dst" \
       "https://huggingface.co/$REPO/resolve/main/$path" || { echo "FAIL  $path"; fail=1; }
done < "$LISTF"
echo "$REPO" > "$DST/SOURCE.txt"
bad=0; n=0; total=0
while read -r size path; do
  n=$((n+1)); total=$((total+size))
  dst="$DST/$path"
  if [[ ! -f "$dst" ]] || (( $(stat -c%s "$dst") != size )); then echo "MISS  $path"; bad=1; fi
done < "$LISTF"
rm -f "$LISTF"
echo "files=$n bytes=$((total>>30))GiB repo=$REPO"
if (( fail || bad )); then echo "RESULT: FAIL — 재실행하면 이어받기"; exit 1; fi
echo "RESULT: PASS — $NAME 준비 완료"
