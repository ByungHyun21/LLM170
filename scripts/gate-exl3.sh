#!/usr/bin/env bash
# EXL3 27B 고정 토큰 게이트 + 벤치 (plans/129 A1, 2026-10-05)
#
# 용도: EXL3 변경(커널·스케줄·경로 전환)의 회귀 판정을 고정 기준 파일로 내린다.
# 127-E 판정(교차 백엔드 토큰 동일성 불성립 — 독립 커널·로짓차 O(1e-2))에 따라
# 기준은 백엔드별로 각각 기록한다(vk·hip 이중 기준 파일).
#
# 사용법:
#   scripts/gate-exl3.sh                # 게이트만 (16토큰 greedy 비트 동일)
#   scripts/gate-exl3.sh --bench        # 게이트 + pp512/tg128 벤치
#   RUNTIME=vulkan scripts/gate-exl3.sh # hip 기본, vk 대조
#   scripts/gate-exl3.sh --record       # 현재 출력을 새 베이스라인으로(신중히)
#
# 주의: GPU 독점 계약 — 다른 추론 프로세스 상주 시 단독 실행할 것.
# 계약상 산술 환원 순서 변경(규칙 10a) 시에만 --record 허용.
set -euo pipefail
cd "$(dirname "$0")/.."
# hipconfig(charhash 무음 즉사 방지 — plans/129 A21g) + ROCm 10 런타임.
if [ -d /opt/rocm-10.0.0/install/bin ]; then
    export PATH=/opt/rocm-10.0.0/install/bin:$PATH
fi
if [ -d /opt/rocm-10.0.0/install/lib ]; then
    export LD_LIBRARY_PATH=/opt/rocm-10.0.0/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi

MODEL=${LLM170_MEXL3:-/home/yoon/models/Qwen3.8-27B-exl3-4.00bpw}
RUNTIME=${RUNTIME:-hip}
# 포맷 자동 판별 계약(2026-10-05): --model이 디렉터리면 EXL3로 라우팅,
# --backend는 순수 런타임(hip|vulkan)만.
BACKEND=$RUNTIME
BASE_FILE=scripts/.gate-exl3-baseline$([[ "$RUNTIME" == vulkan ]] && echo -vk || echo "").txt
# 고정 프롬프트 — gate-flash-next와 동일 한국어 문장 토큰(208개)의 앞부분.
# 토큰 id는 Qwen3.8 계열 어휘 내 유효(고정성만이 게이트 요건).
PROMPT="386,18,15,15,643,20,20,19586,5876,8058,4144,67,21,7307,22,20,23,24902,17,16,23,386,18,66,19,386,17,24,19,24902,16,16,19586,66,21,65,23,1692,22,22,19,4144,341,15,11,17374,67,23,15,1692,15,65,15,1692,22,19,15,17374,66,16,19,386,17,69,22,4144,66,15,15,4144,66,15,15,24902,22,23,23,386,17,24,19,17374,18,66,19,1692,17,7385,386,17,68,19,13,220,17,15,17,21,386,16,19,19,1692,20,67,15,24902,21,65,15,386,24,178442,65,17,24,19,24902,15,66,23,386,23,20,19586,66,21,65,19,21966,24902,2059,19,386,16,16,15,1692,22,19,19,220,16,17,4144,66,16,66,24902,67,20,19586,66,23,15,16,643,21,20,19,643,20,20,23,1692,20,732,24902,67,24,19,386,23,21,15,24902,16,23,1019,65,18,66,19,386,24,22,66,220,18,13,22,386,66,18,15,17374,16,24,17,1692,21,15,15,386,17,68,19,13"

[[ -d "$MODEL" ]] || { echo "모델 디렉터리 없음: $MODEL (LLM170_MEXL3로 지정)"; exit 2; }
# fs 프리플라이트 — 디렉터리 전체가 아닌 첫 safetensors 샤드에 strict 검사.
# (set -e 무음 즉사 방지: 실패 가능 명령은 || true로 완충)
FIRST_SHARD=$(ls "$MODEL"/model-*-of-*.safetensors 2>/dev/null | head -1 || true)
if [[ -z "$FIRST_SHARD" ]]; then
    FIRST_SHARD=$(ls "$MODEL"/*.safetensors 2>/dev/null | head -1 || true)
fi
if [[ -n "$FIRST_SHARD" ]]; then
    "$(dirname "$0")/fs-preflight.sh" "$FIRST_SHARD" --strict || exit 2
else
    echo "샤드 파일 없음: $MODEL"; exit 2
fi

run_gate() {
    ./target/release/llm170 infer --model "$MODEL" --prompt-tokens "$PROMPT" \
        --n-predict 16 --ctx 8192 --backend "$BACKEND" 2>/dev/null \
        | grep -aoE '"token":[0-9]+' | grep -oE '[0-9]+' | tr '\n' ' ' | sed 's/ $//'
}

if [[ "${1:-}" == "--record" ]]; then
    OUT=$(run_gate)
    [[ -n "$OUT" ]] || { echo "기록 실패 — 출력 없음(빌드·모델 확인)"; exit 1; }
    echo "$OUT" > "$BASE_FILE"
    echo "새 베이스라인 기록($RUNTIME): $OUT"
    exit 0
fi

echo "== EXL3 게이트 (ctx 8192, n-predict 16, $BACKEND) =="
OUT=$(run_gate)
if [[ ! -f "$BASE_FILE" ]]; then
    echo "기준 파일 없음($BASE_FILE) — --record 로 먼저 기록"
    exit 2
fi
BASELINE=$(cat "$BASE_FILE")
if [[ "$OUT" == "$BASELINE" ]]; then
    echo "PASS: $OUT"
else
    echo "FAIL — 기대: $BASELINE"
    echo "       실제: $OUT"
    exit 1
fi

if [[ "${1:-}" == "--bench" ]]; then
    echo "== 벤치 (pp512 / tg128, ctx 8192, $BACKEND) =="
    ./target/release/llm170 bench --model "$MODEL" --pp 512 --tg 128 \
        --reps 3 --ctx 8192 --backend "$BACKEND" 2>/dev/null | grep -aE "\| (pp|tg)"
fi
