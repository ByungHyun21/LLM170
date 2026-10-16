#!/usr/bin/env bash
# Gemma-4-26B-A4B-it QAT W4A16(g32·BF16 스케일·per-expert) 다운로드 — CMP 셋업 2026-10-16.
# 소스: reinforce20001/gemma4-26b-a4b-it-qat-w4a16-ct — "구글 공식 QAT W4A16 CT 미러
#   rg20001"(플랜 §2.N)의 실체. 지문: dense FFN 2176·전문가 768·16.5GiB(=16.47G 기록)
#   정합. sachin6624/xbill9판은 2112/704 구판으로 판별(다운로드 금지).
# 대상: ../models/Gemma-4-26B-A4B-it-QAT-W4A16 (swap_models.sh M_G4 경로 동일)
# 규칙: 모델은 한 번에 1개씩(사용자 지시 2026-10-16). 재실행 = 완료 파일 스킵 + 이어받기.
set -euo pipefail
REPO="reinforce20001/gemma4-26b-a4b-it-qat-w4a16-ct"
NAME="Gemma-4-26B-A4B-it-QAT-W4A16"
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
