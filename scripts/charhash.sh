#!/usr/bin/env bash
# W10 특성화 해시 — 스테이지별 golden bufhash/체크섬 캡처·검증 (plans/107)
#
# 용도: 대규모 삭제·이동 리팩터 각 단계가 '스테이지 수준'에서 비트동일임을
# 증명한다. 게이트(최종 토큰)보다 국소화가 좋다 — 불일치 시 어느 스테이지
# (L{il}.mids 등)에서 갈라졌는지 즉시 나온다.
#
# 사용법:
#   scripts/charhash.sh capture [hip|vulkan]   # 현재 빌드를 golden으로 기록
#   scripts/charhash.sh verify  [hip|vulkan]   # golden과 대조(불일치 스테이지 보고)
#   RUNTIME=vulkan scripts/charhash.sh verify
#
# 대상: Flash-Next 고정 프롬프트(gate-flash-next.sh 동일 208토큰, n16).
# 해시 라인: [npbh] bufhash(FNV·f32비트), [npck] 스테이지 체크섬.
# ckdiff 게이트는 별도(diag ckdiff) — 이 스크립트는 스테이지 원장이다.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d /opt/rocm-10.0.0/install/lib ]; then
    export LD_LIBRARY_PATH=/opt/rocm-10.0.0/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi

MODE=${1:-verify}
RUNTIME=${2:-${RUNTIME:-hip}}
MODEL=${LLM170_MFLASH:-/home/yoon/models/qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}
PROMPT="386,18,15,15,643,20,20,19586,5876,8058,4144,67,21,7307,22,20,23,24902,17,16,23,386,18,66,19,386,17,24,19,24902,16,16,19586,66,21,65,23,1692,22,22,19,4144,341,15,11,17374,67,23,15,1692,15,65,15,1692,22,19,15,17374,66,16,19,386,17,69,22,4144,66,15,15,4144,66,15,15,24902,22,23,23,386,17,24,19,17374,18,66,19,1692,17,7385,386,17,68,19,13,220,17,15,17,21,386,16,19,19,1692,20,67,15,24902,21,65,15,386,24,178442,65,17,24,19,24902,15,66,23,386,23,20,19586,66,21,65,19,21966,24902,2059,19,386,16,16,15,1692,22,19,19,220,16,17,4144,66,16,66,24902,67,20,19586,66,23,15,16,643,21,20,19,643,20,20,23,1692,20,732,24902,67,24,19,386,23,21,15,24902,16,23,1019,65,18,66,19,386,24,22,66,220,18,13,22,386,66,18,15,17374,16,24,17,1692,21,15,15,386,17,68,19,13"
SUF=$([[ "$RUNTIME" == vulkan ]] && echo -vk || echo "")
GOLDEN="scripts/.charhash-flash${SUF}.txt"

[[ -f "$MODEL" ]] || { echo "모델 없음: $MODEL"; exit 2; }
[[ -x target/release/llm170 ]] || { echo "target/release/llm170 없음 — cargo build --release -p llm170-server"; exit 2; }

run_hashes() {
    LLM170_DUMP=checksum,bufhash ./target/release/llm170 infer --model "$MODEL" \
        --prompt-tokens "$PROMPT" --n-predict 16 --ctx 8192 \
        --backend gpu --gpu-runtime "$RUNTIME" 2>&1 >/dev/null \
        | grep -aE '^\[(npbh|npck)\] ' | sed -E 's/sum=[-0-9.]+/sum=S/; s/v0=[-0-9.]+/v0=V/; s/mid0=[-0-9.]+/mid0=M/; s/last0=[-0-9.]+/last0=L/'
    # sum/v0/... 는 %.6f 반올림 — 마지막 자리 진동만으로 거짓 불일치를 막기
    # 위해 자릿수 표준화. [npbh] h= 는 f32 비트 FNV라 그대로 둔다.
}

case "$MODE" in
capture)
    run_hashes > "$GOLDEN"
    echo "golden 기록: $GOLDEN ($(wc -l < "$GOLDEN") 해시 라인, $RUNTIME)"
    ;;
verify)
    [[ -f "$GOLDEN" ]] || { echo "golden 없음 — 먼저 capture (현재 빌드가 기준)"; exit 2; }
    NOW=$(mktemp)
    run_hashes > "$NOW"
    if diff -q "$GOLDEN" "$NOW" >/dev/null; then
        echo "PASS: 스테이지 해시 전수 일치 ($(wc -l < "$NOW") 라인, $RUNTIME)"
        rm -f "$NOW"
    else
        echo "FAIL: 스테이지 해시 불일치 — 최초 분기:"
        diff "$GOLDEN" "$NOW" | head -12
        rm -f "$NOW"
        exit 1
    fi
    ;;
*)
    echo "usage: $0 capture|verify [hip|vulkan]"; exit 2 ;;
esac
