#!/usr/bin/env bash
# Qwen3.6-35B-A3B W4A16(g32·CT pack-quantized) 다운로드 — CMP 셋업 2026-10-16.
# 소스: Avesed/Qwen3.6-35B-A3B-INT4-W4A16 — 지문 대조로 특정(g32·moe_ffn 512·256전문가
#   top-8·23.9GiB = 상주 회계 23.8G 정합; 로컬 디렉터리명과 정확 일치).
# 대상: ../models/Qwen3.6-35B-A3B-INT4-W4A16 (swap_models.sh M_35B 경로 동일)
# 규칙: 모델은 한 번에 1개씩(사용자 지시 2026-10-16). 재실행 = 완료 파일 스킵 + 이어받기.
set -euo pipefail
REPO="Avesed/Qwen3.6-35B-A3B-INT4-W4A16"
NAME="Qwen3.6-35B-A3B-INT4-W4A16"
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
