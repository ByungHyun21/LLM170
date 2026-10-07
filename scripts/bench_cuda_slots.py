#!/usr/bin/env python3
"""CUDA 디코드 배치 효율 측정 — plans/cuda-port §5 우선순위 정정용.

왜 필요한가
------------
§5는 "lm_head 전용 경로"를 디코드 속도의 다음 축으로 지목했다. 하지만 그
전제는 측정되지 않았다. 근거로 cited된 값들(MTP 주석의 lm_head 797MB
트렐리스 읽기, "커널 launch 지배")이 서로 다른 축을 가리키는데, 어느 쪽이
실제로 지배하는지 이 기기에서 잰다.

핵심 측정
----------
  1) 단일 슬롯 steady-state 디코드 t/s (n_predict 크게)
  2) 4슬롯에 4요청 동시 → **집합** t/s

판정
----
집합 t/s가 단일의 ~4배면 다중 슬롯이 lm_head(그리고 층 가중치) 판독을 공유해
실제 배치로 동작한다 → lm_head 단축이 4배 효율을 얻으므로 §5-2가 정당하다.
1배에 가까우면 다중 슬롯은 시간분할(time-slicing)일 뿐이다 →
배열·디코드를 공유하지 못하므로 lm_head를 줄여도 배치는 빨라지지 않는다.
어느 쪽이든 근거가 있어야 다음 순서를 정한다.

사용법(워크트리 루트)
---------------------
    LLM170_MEXL3=../models/Qwen3.8-27B-exl3-5.00bpw \\
        LLM170_BENCH_NPRED=128 scripts/bench_cuda_slots.py
"""

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

MODEL = os.environ.get("LLM170_MEXL3", "../models/Qwen3.8-27B-exl3-5.00bpw")
CTX = int(os.environ.get("LLM170_BENCH_CTX", "1024"))
NPRED = int(os.environ.get("LLM170_BENCH_NPRED", "128"))
CONC = int(os.environ.get("LLM170_BENCH_CONC", "4"))
BIN = "./target/release/llm170"

BASE = [
    148678, 65233, 202419, 220, 49849, 155497, 220, 151314, 39504, 149635, 13,
    220, 174675, 30061, 220, 152055, 152065, 12434, 220, 154854, 149248, 80102,
    20673, 220, 214009, 149789, 11, 220, 60177, 148726, 22836, 220, 149965,
    176289, 220, 12434, 160288, 220, 158201, 149635, 13,
]
# 프리필 비용을 지배하지 않게 짧게 — 여기서 재는 것은 순수 디코드다.
PROMPT = BASE * 2  # 82토큰


def post(port, n_predict, timeout=3600):
    body = json.dumps({"prompt": PROMPT, "n_predict": n_predict,
                       "temperature": 0.0}).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion", data=body,
        headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read().decode("utf-8"))
    return time.time() - t, len(out.get("tokens", []))


def start(port, slots):
    log = f"/tmp/opencode/bench_slots{slots}.log"
    os.makedirs("/tmp/opencode", exist_ok=True)
    fh = open(log, "wb")
    p = subprocess.Popen(
        [BIN, "serve", "--backend", "cuda", "--model", MODEL, "--port", str(port),
         "--ctx", str(CTX), "--slots", str(slots)],
        stdout=fh, stderr=subprocess.STDOUT)
    for _ in range(180):
        if p.poll() is not None:
            print(open(log, errors="replace").read()[-1500:], file=sys.stderr)
            sys.exit(f"slots={slots} 서버 조기 종료")
        time.sleep(3)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
            return p, log
        except Exception:
            pass
    p.kill()
    sys.exit(f"slots={slots} 기동 대기 초과")


def measure(slots, port, conc):
    p, log = start(port, slots)
    try:
        # 워밍업 1회(첫 요청은 그래프/버퍼 상태가 Cold — 어느 쪽도 동일 조건)
        post(port, 8)
        lat, ntok = [], [0]
        lock = threading.Lock()

        def one():
            t, n = post(port, NPRED)
            with lock:
                lat.append(t)
                ntok[0] += n

        t0 = time.time()
        ths = [threading.Thread(target=one) for _ in range(conc)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        wall = time.time() - t0
        tps = ntok[0] / wall
        print(f"[bench] slots={slots} conc={conc}: wall={wall:.1f}s "
              f"토큰={ntok[0]} **집합 {tps:.2f} tok/s** "
              f"(개별 {' '.join(f'{x:.1f}' for x in lat)}s)")
        return tps
    finally:
        p.kill()
        p.wait(timeout=60)
        time.sleep(4)


def main():
    if not os.path.isdir(MODEL):
        sys.exit(f"모델 없음: {MODEL}")
    print(f"[bench] 모델={MODEL} ctx={CTX} n_predict={NPRED} 프롬프트={len(PROMPT)}토큰")
    one = measure(1, 8961, 1)
    many = measure(4, 8962, CONC)
    ratio = many / one if one > 0 else 0
    print(f"\n[bench] 단일 {one:.2f} tok/s → {CONC}슬롯 집합 {many:.2f} tok/s "
          f"= {ratio:.2f}배 (이상적 {CONC}배)")
    if ratio >= 0.75 * CONC:
        print("[bench] verdict: 다중 슬롯이 **실제 배치**로 동작 — lm_head·가중치 "
              "판독이 슬롯 간 공유된다. lm_head 축감의 효율 배수 근거가 선다.")
    elif ratio <= 1.5:
        print("[bench] verdict: 다중 슬롯은 **시간분할**에 가깝다 — 판독을 공유하지 "
              "못한다. lm_head 축감만으로 배치는 빨라지지 않는다(진짜 축은 launch "
              "오버헤드·그래프 캡처).")
    else:
        print("[bench] verdict: 부분적 공유 — 중간. lm_head 축감 효과는 배수로 "
              "나누어 받는다.")


if __name__ == "__main__":
    main()