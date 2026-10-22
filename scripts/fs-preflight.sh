#!/usr/bin/env bash
set -u
MODEL="${1:?사용법: fs-preflight.sh <경로> [--strict]}"
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
