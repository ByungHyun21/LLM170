#!/usr/bin/env bash
# rawcuda 커널 자산 일괄 재생성 (2026-10-08 단일 트랙 재편 — 구 개별
# 스크립트 3종(bat+attn.sh+gdn.sh) 통합).
# 규칙: assets/<name>.cu → assets/<name>.fatbin, sm_80+sm_89.
# -fmad=false는 아텐션/GDN/EW/GPTQ4/HEAD 한정 계약(FMA 수축 제거 — core
# 미러와의 비트동일, 각 .cu 헤더 [빌드 계약]). norm/smoke는 기본 fmad.
# .cu가 진실이며 fatbin은 nvcc 출력으로만 갱신한다(손편집 금지).
set -euo pipefail
cd "$(dirname "$0")/.."
NVCC=${NVCC:-nvcc}
command -v "$NVCC" >/dev/null || { echo "nvcc 없음: $NVCC" >&2; exit 1; }
A=crates/backend-gpu/src/rawcuda/assets

build() { # name fmad_flag("" | "-fmad=false")
    local name=$1 fmad=${2:-}
    local src=$A/$name.cu out=$A/$name.fatbin
    [ -f "$src" ] || { echo "소스 없음: $src" >&2; exit 1; }
    local tmp
    tmp=$(mktemp /tmp/opencode/${name}.XXXXXX.fatbin)
    "$NVCC" -fatbin -O3 $fmad \
        -gencode arch=compute_80,code=sm_80 \
        -gencode arch=compute_89,code=sm_89 \
        -o "$tmp" "$src"
    mv "$tmp" "$out"
    echo "CUDA $name sm_80+sm_89 fatbin 갱신: $out ${fmad:+($fmad)}"
}

build smoke
build norm
build gdn   -fmad=false
build attn  -fmad=false
build ew    -fmad=false
build gptq4 -fmad=false
build head  -fmad=false
