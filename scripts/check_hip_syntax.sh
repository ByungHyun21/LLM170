#!/usr/bin/env bash
# HIP 커널 소스 문법 검사 — cargo build 는 .hip 를 보지 않는다(hipRTC 런타임 컴파일).
# 커널 파일을 엔진과 같은 순서로 이어붙여 hipcc 로 파싱만 한다.
# 사용: scripts/check_hip_syntax.sh [arch]   (기본 gfx1151, 실패 시 비0 종료)
#
# 왜 필요한가: 커널 편집 후 첫 확인이 런타임(모델 로드 뒤 hipRTC)이라, 중괄호 하나가
# 어긋나도 "출력이 깨짐"으로만 드러난다. 이 스크립트는 수 초 안에 그 지점을 짚어준다.
set -uo pipefail
ARCH="${1:-gfx1151}"
DIR="$(cd "$(dirname "$0")/.." && pwd)/crates/backend-gpu/src/rawhip/kernels"
OUT="$(mktemp -t llm170_hip_XXXX).hip"
# include_str! 순서와 동일하게 이어붙인다 (kernels/mod.rs SRC).
for f in src_common src_quant src_gemv src_gemv4 src_gemm src_probe src_ew src_gdn src_qsa src_vit src_ms; do
  [ -f "$DIR/$f.hip" ] && cat "$DIR/$f.hip" >> "$OUT"
done
if hipcc -x hip --offload-arch="$ARCH" --cuda-device-only -fsyntax-only -I"$DIR" "$OUT" 2>&1; then
  echo "HIP 문법 OK ($(wc -l < "$OUT") 줄, arch=$ARCH)"
  rm -f "$OUT"; exit 0
else
  echo "HIP 문법 오류 — 위 줄 번호는 이어붙인 파일 기준: $OUT"
  exit 1
fi
