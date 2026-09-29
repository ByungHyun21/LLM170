#!/usr/bin/env python3
"""동시 np 요청의 prefill vs decode 분해 — n_predict 스윕으로 절편/기울기 측정.

사용:
  python3 scripts/np_decompose.py --port 18099 --np 4 --prompt-len 120 [--tag X]
같은 프롬프트로 np 동시 요청을 n_predict ∈ {1, 9, 33} 로 보내 wall 을 재고,
선형회귀로 (프리필+고정) 절편과 (스텝당) 기울기를 분리한다. 워밍 후 측정.
"""
import argparse, json, threading, time, urllib.request

P = argparse.ArgumentParser()
P.add_argument("--port", type=int, default=18099)
P.add_argument("--np", type=int, default=4)
P.add_argument("--prompt-len", type=int, default=120)
P.add_argument("--tag", default="run")
A = P.parse_args()

GATE = ("386,18,15,15,643,20,20,19586,5876,8058,4144,67,21,7307,22,20,23,24902,17,16,23,386,18,66,19,386,17,24,19,24902,16,16,19586,66,21,65,23,1692,22,22,19,4144,341,15,11,17374,67,23,15,1692,15,65,15,1692,22,19,15,17374,66,16,19,386,17,69,22,4144,66,15,15,4144,66,15,15,24902,22,23,23,386,17,24,19,17374,18,66,19,1692,17,7385,386,17,68,19,13,220,17,15,17,21,386,16,19,19,1692,20,67,15,24902,21,65,15,386,24,178442,65,17,24,19,24902,15,66,23,386,23,20,19586,66,21,65,19,21966,24902,2059,19,386,16,16,15,1692,22,19,19,220,16,17,4144,66,16,66,24902,67,20,19586,66,23,15,16,643,21,20,19,643,20,20,23,1692,20,732,24902,67,24,19,386,23,21,15,24902,16,23,1019,65,18,66,19,386,24,22,66,220,18,13,22,386,66,18,15,17374,16,24,17,1692,21,15,15,386,17,68,19,13")
ids = [int(x) for x in GATE.split(",")][: A.prompt_len]
base = f"http://127.0.0.1:{A.port}"


def fire(n):
    res = {}

    def one(i):
        body = json.dumps({"prompt": ids, "n_predict": n, "temperature": 0.0,
                           "cache_prompt": False, "stream": False}).encode()
        req = urllib.request.Request(f"{base}/completion", data=body,
                                     headers={"Content-Type": "application/json"})
        t1 = time.time()
        with urllib.request.urlopen(req, timeout=3600) as r:
            json.loads(r.read())
        res[i] = time.time() - t1

    th = [threading.Thread(target=one, args=(i,)) for i in range(A.np)]
    t0 = time.time()
    for t in th: t.start()
    for t in th: t.join()
    return time.time() - t0


# 워밍 (mmap 페이지 폴트·업로드)
fire(4); fire(4)
pts = []
for n in (1, 9, 33, 65):
    w = min(fire(n) for _ in range(2))
    pts.append((n, w))
    print(f"[{A.tag}] np={A.np} prompt={len(ids)} n_predict={n:3d} -> wall {w*1000:8.1f} ms  ({A.np*n/w:.2f} t/s agg)")
# 선형회귀: wall = a + b*n
xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
b = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x in xs)
a = my - b * mx
print(f"[{A.tag}] 고정(프리필+런치) = {a*1000:.1f} ms,  스텝당 {b*1000:.1f} ms  → 엔진 {A.np/b:.1f} t/s agg, 스텝당 {A.np*b*1000:.1f} ms")
