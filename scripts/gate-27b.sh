#!/usr/bin/env bash
# A21g(plans/129): hipconfig PATH 자립(무출력 rc=1 즉사 방지)
if [ -d /opt/rocm-10.0.0/install/bin ]; then
    export PATH=/opt/rocm-10.0.0/install/bin:$PATH
fi
# Qwen3.8-27B 고정 토큰 게이트 + 벤치 (2026-09-14)
#
# 용도: 어떤 변경 후에도 이 스크립트 하나로 (1) 수치 불변 게이트, (2) 성능 벤치를
# 동일 조건에서 재현한다. flash-next 게이트(scripts/gate-flash-next.sh)와 쌍.
#
# 사용법:
#   scripts/gate-27b.sh            # 게이트만 (토큰 스트림 비교)
#   scripts/gate-27b.sh --bench    # 게이트 + pp512/tg32 벤치
#   RUNTIME=vulkan scripts/gate-27b.sh --bench   # HIP 기본, VK 대조
#   scripts/gate-27b.sh --record   # 현재 출력을 새 베이스라인으로 갱신(신중히)
#
# 베이스라인(2026-09-14, 커밋 2111a14):
#   diverse(한국어 208토큰): 아래 스트림 동일 / HIP pp512 38.1-38.5 t/s, VK pp512 321.6 t/s
#   (구 난수 프롬프트 베이스라인: 68 220 16 15 15 15 15 15 15 15 15 15 15 15 15 15 15 —
#    난수 프롬프트 특유의 퇴화 반복이라 자연어 프롬프트로 교체)
#   (주의: 27B 프리필 최적화는 rawvk에 있음 — HIP은 미이식. README 벤치 표 참조)
#   HIP tg 86-87 ms/스텝 (VK 시대 기록 12.25-12.30 t/s = 81-82 ms)
set -euo pipefail
cd "$(dirname "$0")/.."
# ROCm 10 런타임 기본(2026-09-14, 사용자 지시) — 없으면 시스템(7.2.2)으로 폴백.
if [ -d /opt/rocm-10.0.0/install/lib ]; then
    export LD_LIBRARY_PATH=/opt/rocm-10.0.0/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi

MODEL=${LLM170_M27:-/home/yoon/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf}
RUNTIME=${RUNTIME:-hip}
BASELINE="198 386 17 15 16 23 386 16 19 19 1692 20 67 15 1692 22 19"
# 한국어 실문장 208토큰 — flash 게이트와 동일 프롬프트 (2026-09-14, 사용자 지시).
# 문장: "대한민국의 수도는 서울이고, 부산은 바닷가가 있는 도시다. 2026년에 우리는
#        새로운 GPU 커널을 12개 최적화하여 추론 속도를 3.7배 높였다."
# 주의: 다른 모델 서버가 VRAM을 점유하면 hipMalloc 2(OOM)로 실패한다 — 단독 실행.
PROMPT="148678,65233,202419,220,49849,155497,220,151314,39504,149635,13,220,174675,30061,220,152055,152065,12434,220,154854,149248,80102,20673,220,214009,149789,11,220,60177,148726,22836,220,149965,176289,220,12434,160288,220,158201,149635,13"
# 베이스라인은 런타임별 (2026-09-19, plans/83 D: vk 다중쿼리 flash의 산술
# 순서 변경이 near-tie 토큰을 뒤집켜 스트림이 갈림 — plans/archive/decisions.md 원장 참조).
BASE_FILE=scripts/.gate-27b-baseline$([[ "$RUNTIME" == vulkan ]] && echo -vk || echo "").txt

[[ -f "$MODEL" ]] || { echo "모델 없음: $MODEL (LLM170_M27로 지정)"; exit 2; }

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
if [[ "$OUT" == "$BASELINE" ]]; then
    echo "PASS: $OUT"
else
    echo "FAIL — 기대: $BASELINE"
    echo "       실제: $OUT"
    exit 1
fi

if [[ "${1:-}" == "--bench" ]]; then
    echo "== 벤치 (pp512 / tg32, ctx 4096, $RUNTIME) =="
    # 표준 벤치 지점: pp 512/4k/16k, tg 128
    for pt in 512 4096 16384; do
        ./target/release/llm170 bench --model "$MODEL" --pp $pt --tg 0 \
            --ctx 20480 --backend "$RUNTIME" 2>/dev/null | grep -aE "\| pp"
    done
    ./target/release/llm170 bench --model "$MODEL" --pp 512 --tg 128 \
        --ctx 20480 --backend "$RUNTIME" 2>/dev/null | grep -aE "\| tg"
fi
