#!/usr/bin/env bash
# 90 후속: FN vk 간헐 발산 스트레스 재현 (12연속, 체크섬 덤프).
# 게이트와 동일 프롬프트/조건. 발산 로그는 /tmp/stress_N.log.
set -u
cd "$(dirname "$0")/.."
MODEL=/home/yoon/models/qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
PROMPT=$(sed -n 's/^PROMPT="\([^"]*\)".*/\1/p' scripts/gate-flash-next.sh | head -1)
EXP="17374 66 16 19 386 17 69 22 4144 66 15 15 4144 66 15 15 24902"
ALT="17374 66 16 19 386 17 69 22 4144 66 15 15 4144 66 15 15 1692"
ok=0; div=0
for i in $(seq 1 12); do
    export LLM170_DUMP=checksum
    out=$(./target/release/llm170 infer --model "$MODEL" --prompt-tokens "$PROMPT" \
        --n-predict 16 --ctx 8192 --backend gpu --gpu-runtime vulkan 2>"/tmp/stress_$i.log" \
        | grep -aoE '"token":[0-9]+' | grep -oE '[0-9]+' | tr '\n' ' ' | sed 's/ $//')
    unset LLM170_DUMP
    if [ "$out" == "$EXP" ] || [ "$out" == "$ALT" ]; then
        ok=$((ok+1)); echo "run$i OK"
    else
        div=$((div+1)); echo "run$i DIVERGED: $out"
    fi
done
echo "PASS=$ok DIV=$div"
