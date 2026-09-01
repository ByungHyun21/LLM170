#!/bin/bash
# 재부팅 후 검증 완주 스크립트 — HIP(VRAM 배치) 경로.
# 사용: 재부팅 → tmux new-session -d -s post 'bash /tmp/postrun.sh'
set -x
cd /home/yoon/LLM170
export LD_LIBRARY_PATH=/opt/rocm-7.2.2/lib
unset LLM170_GPU_RUNTIME LLM170_W_CAP_GB LLM170_Q4_CHUNK

# 0) HIP 커널 건재
./target/release/llm170 gdn-ar-check && ./target/release/llm170 moe-down-check

# 1) qwen35 서버 매트릭스 (HIP=VRAM, 11케이스 전체)
./target/release/llm170 serve --model /home/yoon/local_llm/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf --port 18080 --ctx 4096 \
  --backend gpu --gpu-runtime hip > /tmp/srv35_hip.log 2>&1 &
sleep 14
LLM170_EXTRA_ARGS="--backend gpu --gpu-runtime hip" timeout 5400 python3 scripts/verify.py 18080 > /tmp/v35_final.txt 2>&1
grep -E "PASS|FAIL" /tmp/v35_final.txt
pkill -f "llm170 serve"; sleep 2

# 2) qwen4exp 장문 단일·장문+np2 (HIP, VRAM 48GB — llama와 동일 배치)
M=/home/yoon/local_llm/models/qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
# 장문 프롬프트 재생성 (/tmp는 재부팅 소멸)
python3 - <<'PYGEN'
ids = []
para = [760, 3841, 13477, 37550, 33075, 888, 279, 15217, 5388, 13, 561, 6511, 314, 9338, 369, 13399, 821, 38042, 13]
while len(ids) < 2311: ids += para
open("/tmp/long_ids.txt","w").write(",".join(map(str,ids[:2311])))
open("/tmp/long2_ids.txt","w").write(",".join(str((i*9973+11) % 240000) for i in range(1904)))
PYGEN
L1=$(cat /tmp/long_ids.txt); L2=$(cat /tmp/long2_ids.txt)
LLM170_W_CAP_GB=48 ./target/release/llm170 infer --backend gpu --gpu-runtime hip \
  --model "$M" --prompt-tokens "$L1" --n-predict 24 --ctx 4096 2>/tmp/ls_hip.err >/tmp/ls_hip.jsonl
echo "장문단일 lines=$(wc -l < /tmp/ls_hip.jsonl)"
LLM170_W_CAP_GB=48 ./target/release/llm170 infer --backend gpu --gpu-runtime hip \
  --model "$M" --prompt-tokens "$L1" --prompt-tokens "$L2" --n-predict 24 --ctx 4096 2>/tmp/lnp_hip.err >/tmp/lnp_hip.jsonl
echo "장문np2 lines=$(wc -l < /tmp/lnp_hip.jsonl)"

# 3) 판정 (병렬↔단독 + 단독 재실행)
LLM170_W_CAP_GB=48 ./target/release/llm170 infer --backend gpu --gpu-runtime hip \
  --model "$M" --prompt-tokens "$L2" --n-predict 24 --ctx 4096 2>/dev/null >/tmp/l2_solo.jsonl
python3 - <<'PY'
import json
def toks(f, seq=None):
    out = []
    for l in open(f):
        d = json.loads(l)
        if seq is None or d.get("seq") == seq:
            out.append(d["token"])
    return out
try:
    s1 = toks("/tmp/ls_hip.jsonl"); n0 = toks("/tmp/lnp_hip.jsonl", 0); n1 = toks("/tmp/lnp_hip.jsonl", 1)
    s2 = toks("/tmp/l2_solo.jsonl")
    a = min(len(s1), len(n0)); b = min(len(s2), len(n1))
    print(f"장문np2 seq0: {sum(x==y for x,y in zip(s1[:a],n0[:a]))}/{a}")
    print(f"장문np2 seq1: {sum(x==y for x,y in zip(s2[:b],n1[:b]))}/{b}")
except Exception as e:
    print("판정:", e)
PY
echo "[포스트리부트 완주 종료]"
