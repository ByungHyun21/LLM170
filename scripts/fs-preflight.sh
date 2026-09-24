#!/usr/bin/env bash
# FS 프리플라이트 검사 — 벤치마크/게이트 실행 전 모델 파일 접근성 확인.
#
# 배경 (2026-09-25 사고): 111GB mmap 벤치마크 + GPU fault 후 inode 손상이
# 반복 재발(stat은 성공, open은 ENOENT 플래핑). fsck로 복구했으나 재발 방지를
# 위해 모든 무거운 작업 전 이 검사를 통과해야 한다.
#
# 사용법: scripts/fs-preflight.sh <모델파일> [--strict]
#   일반: 5회 연속 open 성공 → 통과
#   strict: 10회 + 0.2s 간격 — 플래핑 탐지
# 종료코드: 0=통과, 1=불안정(작업 중단 권장), 2=완전 차단
set -u
MODEL="${1:?사용법: fs-preflight.sh <모델파일> [--strict]}"
MODE="${2:-}"

N=5; GAP=0.2
[[ "$MODE" == "--strict" ]] && { N=10; GAP=0.3; }

ok=0
for i in $(seq 1 $N); do
  if python3 -c "import os,sys; os.close(os.open(sys.argv[1], os.R_OK))" "$MODEL" 2>/dev/null; then
    ok=$((ok+1))
  fi
  sleep $GAP
done

if [[ $ok -eq $N ]]; then
  echo "fs-preflight: OK ($ok/$N)" >&2
  exit 0
elif [[ $ok -eq 0 ]]; then
  echo "fs-preflight: BLOCKED (0/$N) — 파일시스템 손상 의심." >&2
  echo "  복구: sudo touch /forcefsck && sudo reboot" >&2
  exit 2
else
  echo "fs-preflight: UNSTABLE ($ok/$N) — inode 플래핑. 작업 중단." >&2
  echo "  복구: sudo touch /forcefsck && sudo reboot" >&2
  exit 1
fi
