#!/usr/bin/env bash
# rawcuda 커널 자산 일괄 재생성 (2026-10-08 단일 트랙 재편 — 구 개별
# 스크립트 3종(bat+attn.sh+gdn.sh) 통합).
# 규칙: assets/<name>.cu → assets/<name>.fatbin, sm_80+sm_89.
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
    if [ -f "$out" ] && [ "$out" -nt "$src" ] \
        && [ "$out" -nt "$A/cast_common.cuh" ] && [ "$out" -nt "$A/g4_common.cuh" ]; then
        echo "CUDA $name 스킵(최신 — 강제: touch $src)"
        return
    fi
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
build embed -fmad=false
build gptq4 -fmad=false
build moe   -fmad=false
build head  -fmad=false
