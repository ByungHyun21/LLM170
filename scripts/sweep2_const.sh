#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${1:?tag} ; SUBS=${2:?"sub,..."} ; shift 2 || true
AB_ARGS=()
if [ "${1:-}" = "--ab" ]; then shift; AB_ARGS=("$@"); fi
TRACKED="crates/backend-gpu/src"
git diff --quiet || { echo "sweep2: 워킹트리 dirty — 착수 거부"; exit 2; }
git rev-parse --short HEAD >/dev/null

restore() {
  git checkout -q -- "$TRACKED"
  cargo build --release -q 2>/dev/null || true
}
trap restore EXIT

mkdir -p /tmp/sweep2
cp target/release/llm170 "/tmp/sweep2/${TAG}_base"

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

scripts/build_cuda_kernels.sh >"/tmp/sweep2/${TAG}_build.log" 2>&1 \
  || { echo "sweep2: 커널 빌드 실패 — /tmp/sweep2/${TAG}_build.log"; exit 1; }
cargo build --release -q
cp target/release/llm170 "/tmp/sweep2/${TAG}_cand"
echo "sweep2: $TAG 후보 빌드 OK — /tmp/sweep2/${TAG}_cand"

restore ; trap - EXIT
scripts/preflight.sh >/tmp/sweep2/${TAG}_preflight.log 2>&1 \
  || { echo "sweep2: 원복 상태 preflight 실패 — /tmp/sweep2/${TAG}_preflight.log"; exit 1; }
git diff --quiet || { echo "sweep2: 원복 실패(잔여 diff):"; git status --short; exit 1; }
echo "sweep2: 원복 완료·preflight PASS·워킹트리 clean"

if [ ${#AB_ARGS[@]} -gt 0 ]; then
  scripts/ab-bench.sh "${AB_ARGS[0]}" "/tmp/sweep2/${TAG}_base" "/tmp/sweep2/${TAG}_cand" "${AB_ARGS[@]:1}"
fi
