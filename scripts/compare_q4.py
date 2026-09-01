#!/usr/bin/env python3
"""qwen4exp 실모델 판정 (비공존 2단계) — 서버 정지 상태에서 우리 엔진 실행·대조.
판정: 완전일치 또는 근접티(TIE_EPS=1.5nat, verify.py 규약 승계)."""
import json
import os
import pickle
import subprocess
import sys

OUT = "/tmp/q4_baselines.pkl"
BIN = "target/release/llm170"
TIE_EPS = float(os.environ.get("LLM170_TIE_EPS", "1.5"))
EXTRA = os.environ.get("LLM170_EXTRA_ARGS", "").split()

with open(OUT, "rb") as f:
    store = pickle.load(f)

ok_all = True
for name, ref in store.items():
    ids = ref["ids"]
    toks = ref["tokens"]
    args = [BIN, "infer", "--model",
            os.environ.get("LLM170_MODEL", "/home/yoon/local_llm/models/qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")]
    args += EXTRA + ["--n-predict", "24", "--ctx", "4096",
                     "--prompt-tokens", ",".join(map(str, ids))]
    r = subprocess.run(args, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-3000:])
        print(f"[FAIL] {name}: rc={r.returncode}")
        ok_all = False
        continue
    ours = []
    for line in r.stdout.splitlines():
        j = json.loads(line)
        ours.append(j["token"])
    if ours[:len(toks)] == toks:
        print(f"[PASS] {name}: 완전일치 {len(toks)}/{len(toks)} (base EOS 조기종료)")
        continue
    if ours == toks:
        print(f"[PASS] {name}: 완전일치 24/24")
        continue
    k = next((i for i, (a, b) in enumerate(zip(toks, ours)) if a != b), None)
    probs = ref["probs"]
    tie = False
    if k is not None and probs and k < len(probs):
        top = probs[k].get("top_logprobs", [])
        ids_top = [e["id"] for e in top]
        if ours[k] in ids_top:
            gap = top[0]["logprob"] - top[ids_top.index(ours[k])]["logprob"]
            if gap < TIE_EPS:
                tie = True
                print(f"[PASS] {name}: 근접티 @gen[{k}] 갭 {gap:.2f}")
    if not tie:
        print(f"[FAIL] {name}: 불일치 @gen[{k}] base={toks[k] if k is not None else '?'} ours={ours[k] if k is not None else '?'}")
        ok_all = False

print("=== " + ("전체 PASS" if ok_all else "FAIL 존재") + " ===")
sys.exit(0 if ok_all else 1)
