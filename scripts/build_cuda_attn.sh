#!/usr/bin/env bash
# EXL3 CUDA 어텐션 fatbin 재생성(plans/cuda-port.md S9).
# build_cuda_gdn.sh와 동일 규칙: sm_80+sm_89, -fmad=false(오라클 비트동일
# 미러 계약). .cu가 진실이며 fatbin은 nvcc 출력으로만 갱신한다.
# 위치축 상한 가드(S9)를 반영하려면 반드시 이 스크립트로 재빌드.
set -euo pipefail
cd "$(dirname "$0")/.."
NVCC=${NVCC:-nvcc}
command -v "$NVCC" >/dev/null || { echo "nvcc 없음: $NVCC" >&2; exit 1; }
src=crates/backend-gpu/src/rawcuda/assets/exl3_attn.cu
out=crates/backend-gpu/src/rawcuda/assets/exl3_attn.fatbin
tmp=$(mktemp /tmp/opencode/exl3_attn.XXXXXX.fatbin)
trap 'rm -f "$tmp"' EXIT
"$NVCC" -fatbin -O3 -fmad=false \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_89,code=sm_89 \
    -o "$tmp" "$src"
mv "$tmp" "$out"
echo "CUDA attn sm_80+sm_89 fatbin 갱신: $out"