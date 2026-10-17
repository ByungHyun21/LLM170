#!/usr/bin/env bash
# 2차 재빌드 스윕 W5(+W6 연동) — .cu 상수 치환·빌드·원복 사이클 + ab-bench 2진 쌍 측정.
# 플랜: plans/plan-tuning-sweep.md §5 · 우선순위 근거: docs/cmp170hx-detail.md §10
# (35B 전문가 grp2+d64 = 65.5% 지배 — GRP_KC 32는 A2b가 4090 점유용 희생분, CMP 복원 1순위).
#
# 계약:
#  - 워킹트리 dirty 시 착수 거부(원복 보장 전제) — 종료 시 무조건 원복(fatbin blob·매니페스트 포함).
#  - 치환 = "파일:MACRO:값" 콤마 목록. `#define MACRO <num>` 행 정확히 1회(미발견/중복 = 중단).
#    Rust const 미러는 현재 부재(GRP_KC/DGRP_KC — 호스트 grid 무관, 2026-10-17 확인).
#    미러가 있는 상수를 다룰 때는 듀얼 치환으로 확장할 것(미러 정적 테스트가 누락을 잡음).
#  - ⚠ gptq4.cu는 비결정 코드젠(리빌드 = 드로우 리셋 ±4%) — 판정은 항상 ab-bench
#    2진 쌍(=같은 세션 교차)으로. baseline 2진도 리빌드 후 재생성해 쌍을 맞춘다.
#  - 원복 후 preflight 게이트(fmt·clippy·fatbin 매니페스트)를 돌리고 워킹트리 clean을 확인.
#
# 사용: scripts/sweep2_const.sh <tag> "<file:MACRO:val,..." [--ab <model_dir> <bench 인자...>]
#   예: scripts/sweep2_const.sh kc64 "gptq4.cu:GRP_KC:64,moe.cu:DGRP_KC:64" \
#         --ab ../models/Qwen3.6-35B-A3B-INT4-W4A16 --pp 512,1024 --tg 128 --ctx 16384
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${1:?tag} ; SUBS=${2:?"file:MACRO:val,..."} ; shift 2 || true
AB_ARGS=()
if [ "${1:-}" = "--ab" ]; then shift; AB_ARGS=("$@"); fi
ASSETS=crates/backend-gpu/src/rawcuda/assets
git diff --quiet || { echo "sweep2: 워킹트리 dirty — 착수 거부"; exit 2; }
git rev-parse --short HEAD >/dev/null

restore() {  # fatbin·매니페스트 포함 전면 원복 + baseline 2진 재빌드
  git checkout -q -- "$ASSETS"
  cargo build --release -q 2>/dev/null || true
}
trap restore EXIT

# ① baseline 2진 보존(현 release 빌드 = 정본 fatbin)
mkdir -p /tmp/sweep2
cp target/release/llm170 "/tmp/sweep2/${TAG}_base"

# ② 치환 — 정확 매칭 1회 보장
IFS=',' read -ra PAIRS <<< "$SUBS"
for p in "${PAIRS[@]}"; do
  f="$ASSETS/${p%%:*}"; rest="${p#*:}"; macro="${rest%%:*}"; val="${rest#*:}"
  python3 - "$f" "$macro" "$val" <<'EOF'
import re, sys
f, macro, val = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(f).read()
pat = re.compile(rf'^(#define {macro} )\d+$', re.M)
n = len(pat.findall(src))
assert n == 1, f"{f}:{macro} #define {n}건(1건 기대)"
open(f, 'w').write(pat.sub(rf'\g<1>{val}', src, count=1))
print(f"sweep2: sub {f.split('/')[-1]}:{macro}->{val}")
EOF
done

# ③ 커널 재빌드(매니페스트 자동 갱신) + 호스트 빌드 → 후보 2진
scripts/build_cuda_kernels.sh >"/tmp/sweep2/${TAG}_build.log" 2>&1 \
  || { echo "sweep2: 커널 빌드 실패 — /tmp/sweep2/${TAG}_build.log"; exit 1; }
cargo build --release -q
cp target/release/llm170 "/tmp/sweep2/${TAG}_cand"
echo "sweep2: $TAG 후보 빌드 OK — /tmp/sweep2/${TAG}_cand"

# ④ 원복 + preflight 게이트 + clean 확인
restore ; trap - EXIT
scripts/preflight.sh >/tmp/sweep2/${TAG}_preflight.log 2>&1 \
  || { echo "sweep2: 원복 상태 preflight 실패 — /tmp/sweep2/${TAG}_preflight.log"; exit 1; }
git diff --quiet || { echo "sweep2: 원복 실패(잔여 diff):"; git status --short; exit 1; }
echo "sweep2: 원복 완료·preflight PASS·워킹트리 clean"

# ⑤ W6: ab-bench 2진 쌍(채택/기각 판정은 사람 — 결과는 원장에 기록)
if [ ${#AB_ARGS[@]} -gt 0 ]; then
  scripts/ab-bench.sh "${AB_ARGS[0]}" "/tmp/sweep2/${TAG}_base" "/tmp/sweep2/${TAG}_cand" "${AB_ARGS[@]:1}"
fi
