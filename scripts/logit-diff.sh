#!/usr/bin/env bash
# 엔진 품질 게이트 — 기본(fast) vs LLM170_EXACT=1 로짓 발산 측정.
# 사용법: scripts/logit-diff.sh [model.gguf] [n_prompt_tokens]
# 판정: maxrel < 3e-2 && argmax 일치 → 통과 (llama.cpp MMA 클래스)
set -euo pipefail
MODEL="${1:-/home/yoon/models/qwen3.8-27b/q35work.gguf}"
NTOK="${2:-40}"
BIN="$(dirname "$0")/../target/release/llm170"
P=$(python3 -c "print(','.join(str(100+i*7) for i in range($NTOK)))")

CO_ARG="${CO_PATH:-}"
LLM170_CO_PATH="$CO_ARG" LLM170_DUMP_LOGITS=/tmp/gate_fast.f32 "$BIN" infer --model "$MODEL" \
  --prompt-tokens "$P" --n-predict 1 --backend gpu --gpu-runtime hip >/dev/null 2>&1
LLM170_CO_PATH="$CO_ARG" LLM170_EXACT=1 LLM170_DUMP_LOGITS=/tmp/gate_exact.f32 "$BIN" infer --model "$MODEL" \
  --prompt-tokens "$P" --n-predict 1 --backend gpu --gpu-runtime hip >/dev/null 2>&1

python3 - <<'EOF'
import struct, sys
def load(p):
    b = open(p, "rb").read()
    return struct.unpack(f"<{len(b)//4}f", b)
va = load("/tmp/gate_fast.f32"); vb = load("/tmp/gate_exact.f32")
assert len(va) == len(vb) and len(va) > 0
mx = max(abs(x - y) for x, y in zip(va, vb))
scale = max(abs(x) for x in vb)
mean = sum(abs(x - y) for x, y in zip(va, vb)) / len(va)
ia = max(range(len(va)), key=va.__getitem__)
ib = max(range(len(vb)), key=vb.__getitem__)
rel = mx / scale
ok = rel < 3e-2 and ia == ib
print(f"logit-diff: n={len(va)} max|D|={mx:.4f} maxrel={rel:.1e} mean|D|={mean:.5f} argmax_match={ia == ib}")
print("PASS (llama.cpp MMA class)" if ok else "FAIL — investigate")
sys.exit(0 if ok else 1)
EOF
