#!/bin/bash
# 부팅 직후 원샷 측정 — 첫 스트레스 사이클이 FS 손상시키기 전에 핵심 데이터 확보
cd /home/yoon/LLM170
M="/home/yoon/models/qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD_Q4_K_XL-00001-of-00004.gguf"
L=/home/yoon/LLM170/oneshot.log
echo "=== 원샷 측정 $(date) ===" > $L

# 0. 프리플라이트 (FAIL이면 15초 재시도 ×8 — 부팅 직후 안정화 대기)
for i in $(seq 1 8); do
  if ./scripts/fs-preflight.sh "$M" 2>>$L; then break; fi
  echo "대기 $i..." >> $L; sleep 15
done

# 1. pp512 기본 sg1 (최우선 — 현재 최고 기록 검증)
echo "== 1. pp512 sg1 ==" >> $L
timeout 240 ./target/release/llm170 bench --model "$M" --pp 512 --tg 0 --ctx 20480 --backend gpu --gpu-runtime vulkan 2>/dev/null | grep -a "| pp" >> $L

# 2. pp4096
echo "== 2. pp4096 ==" >> $L
timeout 600 ./target/release/llm170 bench --model "$M" --pp 4096 --tg 0 --ctx 20480 --backend gpu --gpu-runtime vulkan 2>/dev/null | grep -a "| pp" >> $L

# 3. 게이트
echo "== 3. 게이트 ==" >> $L
RUNTIME=vulkan timeout 120 scripts/gate-flash-next.sh 2>&1 | grep -aE "PASS|FAIL" >> $L

# 4. sg2 안전 테스트 (t=7만 — 크래시 시 FS 리스크 있으니 마지막)
echo "== 4. sg2 t=7 ==" >> $L
P7="386,18,15,15,643,20,20"
LLM170_VK_Q4KSG2=1 timeout 90 ./target/release/llm170 infer --model "$M" --prompt-tokens "$P7" --n-predict 1 --ctx 2048 --backend gpu --gpu-runtime vulkan >> $L 2>&1
echo "sg2 exit=$?" >> $L

# 5. sg2가 살아있으면 pp512
if grep -q '"pos"' $L; then
  echo "== 5. sg2 pp512 ==" >> $L
  LLM170_VK_Q4KSG2=1 timeout 240 ./target/release/llm170 bench --model "$M" --pp 512 --tg 0 --ctx 20480 --backend gpu --gpu-runtime vulkan 2>/dev/null | grep -a "| pp" >> $L
fi
echo "=== 완료 $(date) ===" >> $L
cat $L
