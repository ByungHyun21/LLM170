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
#   RUNTIME=cuda scripts/gate-exl3.sh    # cuda 단독 기준 대조(원장 S5)
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
# cuda는 원장(plans/cuda-port.md) S5 단독 백엔드 — 기준 파일을 분리한다
# (127-E: 교차 백엔드 토큰 동일성은 성립 대상이 아님).
case "$RUNTIME" in
    hip) BASE_FILE=scripts/.gate-exl3-baseline.txt ;;
    vulkan) BASE_FILE=scripts/.gate-exl3-baseline-vk.txt ;;
    cuda) BASE_FILE=scripts/.gate-exl3-baseline-cuda.txt ;;
    *) echo "RUNTIME 미지원: $RUNTIME (hip|vulkan|cuda)"; exit 2 ;;
esac
# 고정 프롬프트 — gate-flash-next와 동일 한국어 문장 토큰(208개)의 앞부분.
# 토큰 id는 Qwen3.8 계열 어휘 내 유효(고정성만이 게이트 요건).
PROMPT="148678,65233,202419,220,49849,155497,220,151314,39504,149635,13,220,174675,30061,220,152055,152065,12434,220,154854,149248,80102,20673,220,214009,149789,11,220,60177,148726,22836,220,149965,176289,220,12434,160288,220,158201,149635,13"

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
    # 프롬프트(41) + n_predict(16)이 fwd3s 위치 한계(1024) 안에 들어와야 한다
    # (plans/cuda-port.md S9: ctx가 커도 pos>=1023에서 거절 — ctx를 일단 낮춰
    # 모듈 결함이 아니라 게이트 설정 오류로 드러나게 한다).
    ./target/release/llm170 infer --model "$MODEL" --prompt-tokens "$PROMPT" \
        --n-predict 16 --ctx 1024 --backend "$BACKEND" 2>/dev/null \
        | grep -aoE '"token":[0-9]+' | grep -oE '[0-9]+' | tr '\n' ' ' | sed 's/ $//'
}

if [[ "${1:-}" == "--record" ]]; then
    OUT=$(run_gate)
    [[ -n "$OUT" ]] || { echo "기록 실패 — 출력 없음(빌드·모델 확인)"; exit 1; }
    echo "$OUT" > "$BASE_FILE"
    echo "새 베이스라인 기록($RUNTIME): $OUT"
    exit 0
fi

echo "== EXL3 게이트 (ctx 1024, n-predict 16, $BACKEND) =="
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
    echo "== 벤치 (pp512 / tg128, ctx 1024, $BACKEND) =="
    ./target/release/llm170 bench --model "$MODEL" --pp 512 --tg 128 \
        --reps 3 --ctx 1024 --backend "$BACKEND" 2>/dev/null | grep -aE "\| (pp|tg)"
fi
