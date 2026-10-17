#!/usr/bin/env bash
# 2차 재빌드 스윕 W5(+W6 연동) — .cu 상수 치환·빌드·원복 사이클 + ab-bench 2진 쌍 측정.
# 근거: sm_80 nsys 지도(2026-10-16 — 커밋 이력) · 플랜 = plans/plan.md
# (35B 전문가 grp2+d64 = 65.5% 지배 — GRP_KC 32는 A2b가 4090 점유용 희생분, CMP 복원 1순위).
#
# 계약:
#  - 워킹트리 dirty 시 착수 거부(원복 보장 전제) — 종료 시 무조건 원복(fatbin blob·매니페스트 포함).
#  - 치환 형식 3종(전부 "1회 정확 매칭" 보장):
#      assets/<f>.cu:MACRO:val     — `#define MACRO <val>`(.cu 표준)
#      <f>.rs:CONST:val            — `pub const CONST: <ty> = <val>;`(Rust 미러 듀얼 치환)
#      <f>:RAW:<old>::<new>        — 리터럴 치환(구분자 `::` — launch_bounds 등)
#    미러 정적 테스트가 치환 누락을 빌드 단계에서 잡는다(이중 치환이 가드의 핵심).
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${1:?tag} ; SUBS=${2:?"sub,..."} ; shift 2 || true
AB_ARGS=()
if [ "${1:-}" = "--ab" ]; then shift; AB_ARGS=("$@"); fi
TRACKED="crates/backend-gpu/src"   # 원복 범위(assets + Rust 미러)
git diff --quiet || { echo "sweep2: 워킹트리 dirty — 착수 거부"; exit 2; }
git rev-parse --short HEAD >/dev/null

restore() {  # fatbin·매니페스트 포함 전면 원복 + baseline 2진 재빌드
  git checkout -q -- "$TRACKED"
  cargo build --release -q 2>/dev/null || true
}
trap restore EXIT

# ① baseline 2진 보존(현 release 빌드 = 정본 fatbin)
mkdir -p /tmp/sweep2
cp target/release/llm170 "/tmp/sweep2/${TAG}_base"

# ② 치환 — 형식별 정확 매칭 1회 보장
IFS=',' read -ra PAIRS <<< "$SUBS"
for p in "${PAIRS[@]}"; do
  f="${p%%:*}"; rest="${p#*:}"
  if [ "${rest%%:*}" = RAW ]; then
    spec="${rest#RAW:}"; old="${spec%%::*}"; new="${spec##*::}"
    python3 - "$f" "$old" "$new" <<'EOF'
import sys
f, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(f).read()
n = src.count(old)
assert n == 1, f"{f}: RAW 패턴 {n}회(1회 기대): {old!r}"
open(f, 'w').write(src.replace(old, new, 1))
print(f"sweep2: sub RAW {f}: {old!r}->{new!r}")
EOF
    continue
  fi
  macro="${rest%%:*}"; val="${rest#*:}"
  python3 - "$f" "$macro" "$val" <<'EOF'
import re, sys
f, macro, val = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(f).read()
pat = (re.compile(rf'^(#define {macro} )\d+', re.M) if f.endswith('.cu')
       else re.compile(rf'^(pub const {macro}: [^=]+? = )[\d_]+', re.M))
n = len(pat.findall(src))
assert n == 1, f"{f}:{macro} 정의 {n}건(1건 기대)"
open(f, 'w').write(pat.sub(rf'\g<1>{val}', src, count=1))
kind = 'cu' if f.endswith('.cu') else 'rs'
print(f"sweep2: sub {f.split('/')[-1]}:{macro}->{val} ({kind})")
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
