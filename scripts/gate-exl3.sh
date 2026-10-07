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
#   RUNTIME=cuda scripts/gate-exl3.sh --ctx-long   # 확장 ctx 게이트(S12)
#
# [--ctx-long] — 위치 청크 경계를 넘는 토큰열을 고정한다. 기본 게이트의
# 프롬프트는 41토큰이라 lim≤57, 어텐션 청크가 하나뿐인 구간만 본다.
# S12(fwd3s 위치 청크.online 소프트맥스)로 청크 수가 늘었는데 그 구간을
# 아무도 지키지 않았다 — 옛 1024 상한을 넘긴 경로가 회귀해도 게이트가
# 조용히 통과하는 구멍이었다. 확장 프롬프트는 41토큰을 26회 반복한
# 1066토큰이라 lim이 1082까지 가고 **옛 상한 1024를 넘는다**.
# 기준 파일은 평면과 분리(기본 게이트의 기준이 바뀌어도 확장 게이트는
# 독립 판정된다).
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
    hip) BASE_FILE=scripts/.gate-exl3-baseline.txt; BASE_LONG=scripts/.gate-exl3-baseline-ctx4k.txt ;;
    vulkan) BASE_FILE=scripts/.gate-exl3-baseline-vk.txt; BASE_LONG=scripts/.gate-exl3-baseline-vk-ctx4k.txt ;;
    cuda) BASE_FILE=scripts/.gate-exl3-baseline-cuda.txt; BASE_LONG=scripts/.gate-exl3-baseline-cuda-ctx4k.txt ;;
    *) echo "RUNTIME 미지원: $RUNTIME (hip|vulkan|cuda)"; exit 2 ;;
esac
# [--ctx-long] 확장 게이트(S12) — 위치 청크 경계를 넘는 프롬프트로 토큰열
# 을 고정한다. 프롬프트 41토큰을 26회 반복 = 1066토큰, +16 생성 = lim 1082
# 로 **옛 fwd3s 상한 1024를 넘어간다.** 청크 수 5(1082/256)이 실제로 돌고
# 그 결과가 기준선에 묶여 있어야 청크.online 소프트맥스 회귀가 잡힌다.
# 플래그는 순서 무관하게 모두 읽는다($1만 보면 --ctx-long --record 조합이
# 조용히 무시돼 기록이 안 된다 — 실제로 겪은 함정).
CTX_LONG=0
RECORD=0
BENCH=0
for a in "$@"; do
    case "$a" in
        --ctx-long) CTX_LONG=1 ;;
        --record) RECORD=1 ;;
        --bench) BENCH=1 ;;
        *) echo "알 수 없는 플래그: $a (--ctx-long|--record|--bench)"; exit 2 ;;
    esac
done
GATE_CTX=1024
if [[ $CTX_LONG -eq 1 ]]; then
    BASE_FILE=$BASE_LONG
    GATE_CTX=4096
fi
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
    # [S12] 그 거절은 사라졌다(fwd3s 위치 청크.online 소프트맥스) — 옛 사유를
    # 주석으로 남긴 채 ctx를 올릴 수 있게 했다. --ctx-long은 그 이의를 넘어
    # 청크 5회 구간(lim 1082)까지 본다.
    local prompt=$PROMPT
    if [[ $CTX_LONG -eq 1 ]]; then
        # 41토큰 × LLM170_GATE_LONG_REPEAT(기본 26) = 1066토큰.
        local -a toks
        IFS=, read -ra toks <<< "$PROMPT"
        prompt=""
        local rep=${LLM170_GATE_LONG_REPEAT:-26} i t
        for ((i = 0; i < rep; i++)); do
            for t in "${toks[@]}"; do prompt+="$t,"; done
        done
        prompt=${prompt%,}
    fi
    ./target/release/llm170 infer --model "$MODEL" --prompt-tokens "$prompt" \
        --n-predict 16 --ctx "$GATE_CTX" --backend "$BACKEND" 2>/dev/null \
        | grep -aoE '"token":[0-9]+' | grep -oE '[0-9]+' | tr '\n' ' ' | sed 's/ $//'
}

if [[ $RECORD -eq 1 ]]; then
    OUT=$(run_gate)
    [[ -n "$OUT" ]] || { echo "기록 실패 — 출력 없음(빌드·모델 확인)"; exit 1; }
    echo "$OUT" > "$BASE_FILE"
    echo "새 베이스라인 기록($RUNTIME): $OUT"
    exit 0
fi

echo "== EXL3 게이트 (ctx $GATE_CTX, n-predict 16, $BACKEND$([ $CTX_LONG -eq 1 ] && echo ", 청크 경계 초과")) =="
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

if [[ $BENCH -eq 1 ]]; then
    echo "== 벤치 (pp512 / tg128, ctx 1024, $BACKEND) =="
    ./target/release/llm170 bench --model "$MODEL" --pp 512 --tg 128 \
        --reps 3 --ctx 1024 --backend "$BACKEND" 2>/dev/null | grep -aE "\| (pp|tg)"
fi
