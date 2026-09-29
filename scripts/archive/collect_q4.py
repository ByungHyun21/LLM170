#!/usr/bin/env python3
"""qwen4exp 실모델 기준 수집 (비공존 1단계) — 서버에서 스트림만 모아 피클.
2단계(compare_q4.py)는 서버 정지 후 우리 엔진으로 판정."""
import json
import os
import pickle
import sys
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 10090
BASE = f"http://127.0.0.1:{PORT}"
OUT = "/tmp/q4_baselines.pkl"
N = 24

CASES = {
    "ko": "17 곱하기 23은 얼마야? 계산하고 답해라:",
    "short": "The quick brown fox jumps over the lazy dog. The capital of France is",
    "code": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    ",
}


def post(path, payload, timeout=2400):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def tokenize(text):
    return post("/tokenize", {"content": text})["tokens"]


store = {}
for name, text in CASES.items():
    ids = tokenize(text)
    out = post("/completion", {
        "prompt": ids, "n_predict": N, "temperature": 0.0,
        "cache_prompt": False, "return_tokens": True, "logprobs": 6,
    })
    store[name] = {"ids": ids, "tokens": out.get("tokens"),
                   "probs": out.get("completion_probabilities") or []}
    print(f"[{name}] prompt={len(ids)} gen={len(out.get('tokens') or [])}")

with open(OUT, "wb") as f:
    pickle.dump(store, f)
print(f"wrote {OUT}: {list(store)}")
