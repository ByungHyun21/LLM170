#!/usr/bin/env bash
# EXL3 GEMV fatbin 재생성(plans/cuda-port.md §1 — exl3_gemv_t 행병렬 변형).
# build_cuda_attn.sh와 동일 규칙: sm_80+sm_89, -fmad=false(오라클 비트동일
# 미러 계약). .cu가 진실이며 fatbin은 nvcc 출력으로만 갱신한다.
# exl3_gemv(산술식 변경 금지 — 파생 금지 원장)와 exl3_gemv_t(슬롯 간
# 배치, 행별 환원 순서 보존)이 같은 fatbin에 산다.
set -euo pipefail
cd "$(dirname "$0")/.."
NVCC=${NVCC:-nvcc}
command -v "$NVCC" >/dev/null || { echo "nvcc 없음: $NVCC" >&2; exit 1; }
src=crates/backend-gpu/src/rawcuda/assets/exl3_gemv.cu
out=crates/backend-gpu/src/rawcuda/assets/exl3_gemv.fatbin
tmp=$(mktemp /tmp/opencode/exl3_gemv.XXXXXX.fatbin)
trap 'rm -f "$tmp"' EXIT
"$NVCC" -fatbin -O3 -fmad=false \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_89,code=sm_89 \
    -o "$tmp" "$src"
mv "$tmp" "$out"
echo "CUDA gemv(+gemv_t) sm_80+sm_89 fatbin 갱신: $out"
