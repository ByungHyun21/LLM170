#!/usr/bin/env bash
# A21g(plans/129): hipconfig PATH 자립(무출력 rc=1 즉사 방지)
if [ -d /opt/rocm-10.0.0/install/bin ]; then
    export PATH=/opt/rocm-10.0.0/install/bin:$PATH
fi
# Qwen3.8-Flash-Next 고정 토큰 게이트 + 벤치 (2026-09-14)
#
# 용도: 어떤 변경 후에도 이 스크립트 하나로 (1) 수치 불변 게이트, (2) 성능 벤치를
# 동일 조건에서 재현한다. 대화형으로 타이핑하던 측정 명령의 단일 진실 공급원.
#
# 사용법:
#   scripts/gate-flash-next.sh            # 게이트만 (16토큰 비트 동일 확인)
#   scripts/gate-flash-next.sh --bench    # 게이트 + pp2048/tg32 벤치
#   RUNTIME=vulkan scripts/gate-flash-next.sh --bench   # HIP 기본, VK 대조
#   scripts/gate-flash-next.sh --record   # 현재 출력을 새 베이스라인으로 갱신(신중히)
#
# 베이스라인(2026-09-14, 커밋 2111a14, HIP, 한국어 208토큰 프롬프트):
#   diverse: 아래 스트림 동일 / pp2048 233-252 t/s / tg 76ms(단문맥), 91ms(장문맥)
#   (구 난수 프롬프트 베이스라인: 5513 248046 198 248045 74455 198 248068 198 760 1156 579 1876 7701 310 381 7132 36412)
set -euo pipefail
cd "$(dirname "$0")/.."
# ROCm 10 런타임 기본(2026-09-14, 사용자 지시) — 없으면 시스템(7.2.2)으로 폴백.
if [ -d /opt/rocm-10.0.0/install/lib ]; then
    export LD_LIBRARY_PATH=/opt/rocm-10.0.0/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi

MODEL=${LLM170_MFLASH:-/home/yoon/models/qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}
RUNTIME=${RUNTIME:-hip}
BASELINE="17374 66 16 19 386 17 69 22 4144 66 15 15 4144 66 15 15 1692"
# 한국어 실문장 208토큰 — 실제 분포의 입력으로 게이트 (2026-09-14, 사용자 지시).
# 문장: "대한민국의 수도는 서울이고, 부산은 바닷가가 있는 도시다. 2026년에 우리는
#        새로운 GPU 커널을 12개 최적화하여 추론 속도를 3.7배 높였다."
# 주의: 다른 모델 서버가 VRAM을 점유하면 hipMalloc 2(OOM)로 실패한다 — 단독 실행.
PROMPT="148678,65233,202419,220,49849,155497,220,151314,39504,149635,13,220,174675,30061,220,152055,152065,12434,220,154854,149248,80102,20673,220,214009,149789,11,220,60177,148726,22836,220,149965,176289,220,12434,160288,220,158201,149635,13"
BASE_FILE=scripts/.gate-flash-baseline$([[ "$RUNTIME" == vulkan ]] && echo -vk || echo "").txt

[[ -f "$MODEL" ]] || { echo "모델 없음: $MODEL (LLM170_MFLASH로 지정)"; exit 2; }
# plans/93: FS 프리플라이트 — inode 손상(벤치마크 I/O + GPU fault 유발) 재발 방지.
"$(dirname "$0")/fs-preflight.sh" "$MODEL" --strict || exit 2

if [[ "${1:-}" == "--record" ]]; then
    ./target/release/llm170 infer --model "$MODEL" --prompt-tokens "$PROMPT" \
        --n-predict 16 --ctx 8192 --backend "$RUNTIME" 2>/dev/null \
        | grep -aoE '"token":[0-9]+' | grep -oE '[0-9]+' | tr '\n' ' ' | sed 's/ $//' > "$BASE_FILE"
    echo "새 베이스라인 기록: $(cat "$BASE_FILE")"
    exit 0
fi

echo "== diverse 게이트 (ctx 8192, n-predict 16, $RUNTIME) =="
OUT=$(./target/release/llm170 infer --model "$MODEL" --prompt-tokens "$PROMPT" \
    --n-predict 16 --ctx 8192 --backend "$RUNTIME" 2>/dev/null \
    | grep -aoE '"token":[0-9]+' | grep -oE '[0-9]+' | tr '\n' ' ' | sed 's/ $//')
if [[ -f "$BASE_FILE" ]]; then BASELINE=$(cat "$BASE_FILE"); fi
# 마지막 토큰은 0.27nat 근접타이(24902:13.91 vs 1692:13.64, ADR-0012 ε=1.5 내)
# — 기계 상태(열/UMA)에 따라 어느 쪽이든 정당하므로 두 변이를 모두 수용한다.
ALT_BASELINE="${BASELINE% *} 1692"
if [[ "$OUT" == "$BASELINE" || "$OUT" == "$ALT_BASELINE" ]]; then
    echo "PASS: $OUT"
else
    echo "FAIL — 기대: $BASELINE (또는 타이 변이 $ALT_BASELINE)"
    echo "       실제: $OUT"
    exit 1
fi

if [[ "${1:-}" == "--bench" ]]; then
    echo "== 벤치 (pp2048 / tg32, ctx 8192, $RUNTIME) =="
    # 표준 벤치 지점: pp 512/4k/16k, tg 128
    for pt in 512 4096 16384; do
        ./target/release/llm170 bench --model "$MODEL" --pp $pt --tg 0 \
            --ctx 20480 --backend "$RUNTIME" 2>/dev/null | grep -aE "\| pp"
    done
    ./target/release/llm170 bench --model "$MODEL" --pp 512 --tg 128 \
        --ctx 20480 --backend "$RUNTIME" 2>/dev/null | grep -aE "\| tg"
fi
