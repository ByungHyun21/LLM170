#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p /tmp/opencode
NVCC=${NVCC:-nvcc}
command -v "$NVCC" >/dev/null || { echo "nvcc 없음: $NVCC" >&2; exit 1; }
A=crates/backend-gpu/src/rawcuda/assets
MAN=$A/FATBINS.manifest
man_src() {
    if [ "$1" = marlin ]; then
        cat "$A/marlin.cu" $A/marlin/*.h $A/marlin/*.cuh $A/marlin/core/*.hpp \
            $A/marlin/torch/headeronly/util/Exception.h | sha256sum | cut -d' ' -f1
    elif [ "$1" = marlin_moe ]; then
        cat "$A/marlin_moe.cu" $A/marlin_moe/*.h \
            $A/marlin_moe/quantization/marlin/*.h* $A/marlin_moe/core/*.hpp \
            $A/marlin_moe/torch/headeronly/util/Exception.h | sha256sum | cut -d' ' -f1
    elif [ "$1" = marlin_g32 ]; then
        cat "$A/marlin_g32.cu" $A/marlin/*.h $A/marlin/*.cuh $A/marlin/core/*.hpp \
            $A/marlin/torch/headeronly/util/Exception.h | sha256sum | cut -d' ' -f1
    else
        cat "$A/$1.cu" "$A/cast_common.cuh" "$A/g4_common.cuh" | sha256sum | cut -d' ' -f1
    fi
}
man_fb() { sha256sum "$A/$1.fatbin" | cut -d' ' -f1; }
declare -A MAN_SRC=() MAN_FB=()
if [ -f "$MAN" ]; then
    while read -r s f n; do
        [ -n "${n:-}" ] && { MAN_SRC[$n]=$s; MAN_FB[$n]=$f; }
    done < "$MAN"
fi

build() {
    local name=$1 fmad=${2:-}
    local src=$A/$name.cu out=$A/$name.fatbin
    [ -f "$src" ] || { echo "소스 없음: $src" >&2; exit 1; }
    local hdr_new=""
    if [ "$name" = marlin ] && [ -f "$out" ]; then
        hdr_new=$(find "$A/marlin" -type f -newer "$out" 2>/dev/null | head -1) || true
    elif [ "$name" = marlin_moe ] && [ -f "$out" ]; then
        hdr_new=$(find "$A/marlin_moe" -type f -newer "$out" 2>/dev/null | head -1) || true
    elif [ "$name" = marlin_g32 ] && [ -f "$out" ]; then
        hdr_new=$(find "$A/marlin" -type f -newer "$out" 2>/dev/null | head -1) || true
    fi
    if [ -f "$out" ] && [ "$out" -nt "$src" ] && [ -z "$hdr_new" ] \
        && [ "$out" -nt "$A/cast_common.cuh" ] && [ "$out" -nt "$A/g4_common.cuh" ]; then
        echo "CUDA $name 스킵(최신 — 강제: touch $src)"
        return
    fi
    local extra=""
    [ "$name" = marlin ] && extra="--expt-relaxed-constexpr -I$A/marlin"
    [ "$name" = marlin_moe ] && extra="--expt-relaxed-constexpr -I$A/marlin_moe"
    [ "$name" = marlin_g32 ] && extra="--expt-relaxed-constexpr -I$A/marlin"
    tmp=$(mktemp /tmp/opencode/${name}.XXXXXX.fatbin)
    "$NVCC" -fatbin -O3 $fmad $extra \
        -gencode arch=compute_80,code=sm_80 \
        -o "$tmp" "$src"
    mv "$tmp" "$out"
    MAN_SRC[$name]=$(man_src "$name"); MAN_FB[$name]=$(man_fb "$name")
    echo "CUDA $name sm_80 fatbin 갱신: $out ${fmad:+($fmad)}"
}

build smoke
build norm
build gdn   -fmad=false
build attn  -fmad=false
build ew    -fmad=false
build embed -fmad=false
build gptq4 -fmad=false
build gemma4 -fmad=false
build decode -fmad=false
build moe   -fmad=false
build moepp -fmad=false
build fn    -fmad=false
build head  -fmad=false
build marlin
build marlin_moe
build marlin_g32

for name in smoke norm gdn attn ew embed gptq4 gemma4 moe moepp fn decode head marlin marlin_moe marlin_g32; do
    if [ -z "${MAN_SRC[$name]:-}" ]; then
        MAN_SRC[$name]=$(man_src "$name"); MAN_FB[$name]=$(man_fb "$name")
    fi
done
{
    for name in smoke norm gdn attn ew embed gptq4 gemma4 moe moepp fn decode head marlin marlin_moe marlin_g32; do
        echo "${MAN_SRC[$name]} ${MAN_FB[$name]} $name"
    done
} > "$MAN.tmp" && mv "$MAN.tmp" "$MAN"
echo "매니페스트 갱신: $MAN"
