#!/usr/bin/env bash
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
PROMPT="386,18,15,15,643,20,20,19586,5876,8058,4144,67,21,7307,22,20,23,24902,17,16,23,386,18,66,19,386,17,24,19,24902,16,16,19586,66,21,65,23,1692,22,22,19,4144,341,15,11,17374,67,23,15,1692,15,65,15,1692,22,19,15,17374,66,16,19,386,17,69,22,4144,66,15,15,4144,66,15,15,24902,22,23,23,386,17,24,19,17374,18,66,19,1692,17,7385,386,17,68,19,13,220,17,15,17,21,386,16,19,19,1692,20,67,15,24902,21,65,15,386,24,178442,65,17,24,19,24902,15,66,23,386,23,20,19586,66,21,65,19,21966,24902,2059,19,386,16,16,15,1692,22,19,19,220,16,17,4144,66,16,66,24902,67,20,19586,66,23,15,16,643,21,20,19,643,20,20,23,1692,20,732,24902,67,24,19,386,23,21,15,24902,16,23,1019,65,18,66,19,386,24,22,66,220,18,13,22,386,66,18,15,17374,16,24,17,1692,21,15,15,386,17,68,19,13"
BASE_FILE=scripts/.gate-flash-baseline.txt

[[ -f "$MODEL" ]] || { echo "모델 없음: $MODEL (LLM170_MFLASH로 지정)"; exit 2; }

if [[ "${1:-}" == "--record" ]]; then
    ./target/release/llm170 infer --model "$MODEL" --prompt-tokens "$PROMPT" \
        --n-predict 16 --ctx 8192 --backend gpu --gpu-runtime "$RUNTIME" 2>/dev/null \
        | grep -aoE '"token":[0-9]+' | grep -oE '[0-9]+' | tr '\n' ' ' | sed 's/ $//' > "$BASE_FILE"
    echo "새 베이스라인 기록: $(cat "$BASE_FILE")"
    exit 0
fi

echo "== diverse 게이트 (ctx 8192, n-predict 16, $RUNTIME) =="
OUT=$(./target/release/llm170 infer --model "$MODEL" --prompt-tokens "$PROMPT" \
    --n-predict 16 --ctx 8192 --backend gpu --gpu-runtime "$RUNTIME" 2>/dev/null \
    | grep -aoE '"token":[0-9]+' | grep -oE '[0-9]+' | tr '\n' ' ' | sed 's/ $//')
if [[ -f "$BASE_FILE" ]]; then BASELINE=$(cat "$BASE_FILE"); fi
if [[ "$OUT" == "$BASELINE" ]]; then
    echo "PASS: $OUT"
else
    echo "FAIL — 기대: $BASELINE"
    echo "       실제: $OUT"
    exit 1
fi

if [[ "${1:-}" == "--bench" ]]; then
    echo "== 벤치 (pp2048 / tg32, ctx 8192, $RUNTIME) =="
    ./target/release/llm170 bench --model "$MODEL" --pp 2048 --tg 0 \
        --ctx 8192 --backend gpu --gpu-runtime "$RUNTIME" 2>/dev/null | grep -aE "\| pp"
    ./target/release/llm170 bench --model "$MODEL" --pp 0 --tg 32 \
        --ctx 8192 --backend gpu --gpu-runtime "$RUNTIME" 2>/dev/null | grep -aE "\| tg"
fi
