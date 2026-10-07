#!/usr/bin/env bash
# EXL3 CUDA GDN 단일 fatbin 재생성(plans/cuda-port.md S5).
# Windows scripts/build_cuda.bat의 exl3_gdn 규칙과 동일한 sm_80+sm_89,
# -fmad=false 계약. .cu가 진실이며 fatbin은 nvcc 출력으로만 갱신한다.
# T<32에서 마스킹 q 행 OOB(원장 S5)를 고친 뒤 반드시 이 스크립트로 재빌드.
set -euo pipefail
cd "$(dirname "$0")/.."
NVCC=${NVCC:-nvcc}
command -v "$NVCC" >/dev/null || { echo "nvcc 없음: $NVCC" >&2; exit 1; }
src=crates/backend-gpu/src/rawcuda/assets/exl3_gdn.cu
out=crates/backend-gpu/src/rawcuda/assets/exl3_gdn.fatbin
tmp=$(mktemp /tmp/opencode/exl3_gdn.XXXXXX.fatbin)
trap 'rm -f "$tmp"' EXIT
"$NVCC" -fatbin -O3 -fmad=false \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_89,code=sm_89 \
    -o "$tmp" "$src"
mv "$tmp" "$out"
echo "CUDA GDN sm_80+sm_89 fatbin 갱신: $out"
