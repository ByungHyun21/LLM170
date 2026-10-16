#!/usr/bin/env bash
# Qwen3.8-Flash-Next W4A16+FP8PLE 다운로드 — CMP 셋업 2026-10-16. (125GiB급 — 가장 김)
# 소스: albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE — 로컬 디렉터리명과 정확 일치(4090 원장
#   "local albucino checkpoint"). 지문: g128 sym auto-round·512전문가 top-10·moe 640·120.1GiB.
# 대상: ../models/Qwen3.8-Flash-Next-W4A16-FP8PLE (swap_models.sh M_FN 경로 동일)
# 규칙: 모델은 한 번에 1개씩(사용자 지시 2026-10-16). 재실행 = 완료 파일 스킵 + 이어받기.
set -euo pipefail
REPO="albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE"
NAME="Qwen3.8-Flash-Next-W4A16-FP8PLE"
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
