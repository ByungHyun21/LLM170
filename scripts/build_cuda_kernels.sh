#!/usr/bin/env bash
# rawcuda 커널 자산 일괄 재생성 (2026-10-08 단일 트랙 재편 — 구 개별
# 스크립트 3종(bat+attn.sh+gdn.sh) 통합).
# 규칙: assets/<name>.cu → assets/<name>.fatbin, sm_80.
# -fmad=false는 아텐션/GDN/EW/GPTQ4/MOE/HEAD 한정 계약(FMA 수축 제거 — core
# 미러와의 비트동일, 각 .cu 헤더 [빌드 계약]). norm/smoke는 기본 fmad.
# .cu가 진실이며 fatbin은 nvcc 출력으로만 갱신한다(손편집 금지).
# [경고 2026-10-10] 이 스크립트의 출력을 grep 등으로 가리지 마라 — nvcc
# 문법 오류(launch_bounds 배치 실측)가 실패로 전파되지 않고 **스테일
# fatbin**이 남아 측정·골든이 오염된다(set -e는 여기서만 보장). .cu 변경
# 후에는 원출력 확인 + 골든 재판정이 필수.
set -euo pipefail
cd "$(dirname "$0")/.."
# A1: 클린 환경에서 mktemp 접두 디렉터리 부재로 즉시 실패하던 결함.
mkdir -p /tmp/opencode
NVCC=${NVCC:-nvcc}
command -v "$NVCC" >/dev/null || { echo "nvcc 없음: $NVCC" >&2; exit 1; }
A=crates/backend-gpu/src/rawcuda/assets
# [2.N 2026-10-15] 내용 무결성 매니페스트 — mtime 가드는 "외부 복원이 mtime만
# 갱신" 사고를 못 잡는다(실측: 옛 fatbin + 새 mtime → 신선도 스킵이 리빌드
# 누락). 자산별 (소스해시, fatbin해시) 쌍을 기록하고 preflight가 대조한다.
MAN=$A/FATBINS.manifest
man_src() { # [2026-10-18] marlin은 벤더 헤더 전량을 해시에 포함.
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

# [경고 2026-10-11] gptq4.cu(대형·mma)는 nvcc 13.0.48에서 **비결정 컴파일**
# (동일 플래그 재컴파일 시 SASS 상이 — 코드젠 추첨). 실측 드로우 간 27B
# pp512 1680~1820(±4%). 리빌드 후 반드시 pp 재측정 + 필요 시 재뽑기 —
# 채택 드로우는 커밋된 fatbin 바이트로 고정된다. 그 외 자산(moe/attn/gdn/
# norm/ew/head)은 결정적(2회 동일 실측).
build() { # name fmad_flag("" | "-fmad=false")
    local name=$1 fmad=${2:-}
    local src=$A/$name.cu out=$A/$name.fatbin
    [ -f "$src" ] || { echo "소스 없음: $src" >&2; exit 1; }
    # [2026-10-11] 신선도 스킵(make식) — 소스·공용 헤더보다 fatbin이 새로우면
    # 재빌드 생략. gptq4는 재빌드마다 코드젠 추첨이 리셋되므로(실측 ±4%)
    # 불필요한 재컴파일이 고정 드로우를 덮어쓰는 사고를 막는다.
    # 강제 재빌드: touch <name>.cu 후 실행.
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
    # [2026-10-18] 마린: constexpr 호출 허용 + torch headeronly 심 경로.
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
build marlin            # [2026-10-18] vLLM 벤더(TU·헤더), 기본 fmad
build marlin_moe        # [2026-10-19] vLLM MoE 벤더(TU·헤더), 기본 fmad
build marlin_g32        # [2026-10-19] g32 밀집(gemma4 dense t>16) — 벤더 TU, 기본 fmad

# 매니페스트 갱신 — 리빌드분은 새 쌍, 스킵분은 기존 엔트리 유지(부재 시 현재
# 상태를 시드 — 최초 도입/매니페스트 삭제 시에만). preflight 5단계가 대조한다.
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
