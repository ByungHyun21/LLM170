#!/usr/bin/env python3
"""llama-server np 동시 스트림 aggregate 벤치 (내부 CLI 벤치와 같은 조건 비교용).

사용:
  python3 scripts/bench_llama_np.py --port 10090 --np 4 --n-predict 128 [--prompt-repeat 24]

같은 프롬프트를 np개 슬롯에 동시에 보내고, 완료까지의 wall로 aggregate t/s를 낸다.
llama-server 0.4.x /completion 응답의 timings.predicted_n 을 합산한다.
"""
import argparse
import json
import threading
import time
import urllib.request

P = argparse.ArgumentParser()
P.add_argument("--port", type=int, default=10090)
P.add_argument("--np", type=int, default=4)
P.add_argument("--n-predict", type=int, default=128)
P.add_argument("--prompt-repeat", type=int, default=24, help="5토큰 시드를 반복해 프롬프트 구성")
P.add_argument("--api", default="/completion")
P.add_argument("--sse", action="store_true", help="SSE 서버(llm170 serve)")
P.add_argument("--text", default=None, help="프롬프트를 텍스트로 (미지정 시 시드 토큰 반복)")
A = P.parse_args()

SEED = [760, 6511, 314, 9338, 369]
PROMPT = A.text if A.text else SEED * A.prompt_repeat


def one(idx, out):
    body = json.dumps({
        "prompt": PROMPT,
        "n_predict": A.n_predict,
        "temperature": 0.0,
        "cache_prompt": False,
        "stream": bool(A.sse),
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{A.port}{A.api}",
        data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    n_events = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        if A.sse:
            # SSE: 줄 단위로 읽고 [DONE] 또는 예상 토큰 수에서 끊는다
            # (서버가 keep-alive로 연결을 닫지 않을 수 있음).
            for raw_line in r:
                line = raw_line.decode("utf-8", "replace")
                if line.startswith("data:"):
                    payload = line[5:].strip()
                    if payload and payload != "[DONE]":
                        n_events += 1
                    if payload == "[DONE]" or n_events >= A.n_predict:
                        break
        else:
            j = json.loads(r.read())
    dt = time.time() - t0
    if A.sse:
        out[idx] = (n_events, dt, len(PROMPT))
        return
    tm = j.get("timings") or {}
    n_pred = tm.get("predicted_n")
    if n_pred is None:
        n_pred = len(j.get("tokens") or [])
    out[idx] = (n_pred, dt, tm.get("prompt_n", len(PROMPT)))


res = {}
th = [threading.Thread(target=one, args=(i, res)) for i in range(A.np)]
t0 = time.time()
for t in th:
    t.start()
for t in th:
    t.join()
wall = time.time() - t0
gen = sum(v[0] for v in res.values())
pp = sum(v[2] for v in res.values())
slowest = max((v[1] for v in res.values()), default=0.0)
print(f"np={A.np} n_predict={A.n_predict} prompt={len(PROMPT)} tok")
print(f"  generated {gen} tok in {wall:.2f}s wall (slowest stream {slowest:.2f}s)"
      f" -> aggregate {gen / wall:.2f} t/s")
print(f"  pp-side tokens {pp} (ttft not measured here)")
