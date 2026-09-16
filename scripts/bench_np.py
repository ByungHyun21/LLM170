#!/usr/bin/env python3
"""np 동시 HTTP aggregate 벤치 (llm170 / llama-server 공용, 동일 프로토콜).

사용:
  python3 scripts/bench_np.py --cmd "..." --port 18099 --np 4 --n-predict 128
프롬프트: --prompt-file 토큰 id(쉼표) — 없으면 내장 자연어 문장 시드 반복.
출력: aggregate t/s, 스트림별 t/s, 완료 토큰 수. SSE/비SSE 자동.
"""
import argparse, json, os, shutil, signal, subprocess, threading, time, urllib.request

P = argparse.ArgumentParser()
P.add_argument("--cmd", default=None, help="서버 실행 명령(셸) — 없으면 기존 서버에 붙는다")
P.add_argument("--port", type=int, default=18099)
P.add_argument("--np", type=int, default=4)
P.add_argument("--n-predict", type=int, default=128)
P.add_argument("--prompt-file", default=None)
P.add_argument("--max-prompt", type=int, default=0)
P.add_argument("--api", default="/completion")
P.add_argument("--ready", type=int, default=1800)
P.add_argument("--tag", default="run")
P.add_argument("--repeat", type=int, default=1)
A = P.parse_args()

GATE_PROMPT = "386,18,15,15,643,20,20,19586,5876,8058,4144,67,21,7307,22,20,23,24902,17,16,23,386,18,66,19,386,17,24,19,24902,16,16,19586,66,21,65,23,1692,22,22,19,4144,341,15,11,17374,67,23,15,1692,15,65,15,1692,22,19,15,17374,66,16,19,386,17,69,22,4144,66,15,15,4144,66,15,15,24902,22,23,23,386,17,24,19,17374,18,66,19,1692,17,7385,386,17,68,19,13,220,17,15,17,21,386,16,19,19,1692,20,67,15,24902,21,65,15,386,24,178442,65,17,24,19,24902,15,66,23,386,23,20,19586,66,21,65,19,21966,24902,2059,19,386,16,16,15,1692,22,19,19,220,16,17,4144,66,16,66,24902,67,20,19586,66,23,15,16,643,21,20,19,643,20,20,23,1692,20,732,24902,67,24,19,386,23,21,15,24902,16,23,1019,65,18,66,19,386,24,22,66,220,18,13,22,386,66,18,15,17374,16,24,17,1692,21,15,15,386,17,68,19,13"

src = A.prompt_file or GATE_PROMPT
ids = [int(x) for x in open(src).read().strip().split(",")] if os.path.exists(str(src)) else [int(x) for x in str(src).split(",")]
if A.max_prompt:
    ids = ids[: A.max_prompt]
print(f"[{A.tag}] prompt {len(ids)} tok  np={A.np}  n_predict={A.n_predict}")

proc = None
if A.cmd:
    _errf = open("/tmp/bench_np_err.log", "wb")
    proc = subprocess.Popen(A.cmd, shell=True, stdout=subprocess.DEVNULL, stderr=_errf,
                            preexec_fn=os.setsid)
base = f"http://127.0.0.1:{A.port}"
try:
    t0 = time.time()
    ok = False
    while time.time() - t0 < A.ready:
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
                if r.status == 200:
                    ok = True
                    break
        except Exception:
            if proc is not None and proc.poll() is not None:
                print(f"[{A.tag}] 서버 조기 종료 rc={proc.returncode}")
                raise SystemExit(1)
            time.sleep(2)
    if not ok:
        print(f"[{A.tag}] 서버 준비 실패"); raise SystemExit(1)

    def fire(n_predict, tag):
        res = {}
        def one(i):
            body = json.dumps({"prompt": ids, "n_predict": n_predict, "temperature": 0.0,
                               "cache_prompt": False, "stream": False}).encode()
            req = urllib.request.Request(f"{base}{A.api}", data=body,
                                         headers={"Content-Type": "application/json"})
            t1 = time.time()
            with urllib.request.urlopen(req, timeout=7200) as r:
                j = json.loads(r.read())
            dt = time.time() - t1
            toks = j.get("tokens") or []
            n = (j.get("timings") or {}).get("predicted_n") or len(toks)
            res[i] = (n, dt)
        th = [threading.Thread(target=one, args=(i,)) for i in range(A.np)]
        t0 = time.time()
        for t in th: t.start()
        for t in th: t.join()
        wall = time.time() - t0
        gen = sum(v[0] for v in res.values())
        per = [f"{v[0]/v[1]:.2f}" for v in res.values()]
        print(f"[{A.tag}] {tag}: aggregate {gen/wall:.2f} t/s  ({gen} tok / {wall:.3f}s wall)  per-slot {per}")
        return gen / wall

    # 워밍: 첫 패스는 mmap 페이지 폴트·최초 업로드로 수 배 느리다(문서 경고) — 계측 전 2회.
    fire(min(8, A.n_predict), "warmup1")
    fire(min(8, A.n_predict), "warmup2")
    best = 0.0
    for r in range(A.repeat):
        best = max(best, fire(A.n_predict, f"run{r+1}"))
    print(f"[{A.tag}] BEST aggregate {best:.2f} t/s")
    raise SystemExit(0)

    res = {}
    def one(i):
        body = json.dumps({"prompt": ids, "n_predict": A.n_predict, "temperature": 0.0,
                           "cache_prompt": False, "stream": False}).encode()
        req = urllib.request.Request(f"{base}{A.api}", data=body,
                                     headers={"Content-Type": "application/json"})
        t1 = time.time()
        with urllib.request.urlopen(req, timeout=7200) as r:
            j = json.loads(r.read())
        dt = time.time() - t1
        toks = j.get("tokens")
        if toks is None:
            toks = (j.get("choices") or [{}])[0].get("text", "").split()
        n = (j.get("timings") or {}).get("predicted_n") or len(toks)
        res[i] = (n, dt)
    th = [threading.Thread(target=one, args=(i,)) for i in range(A.np)]
    t0 = time.time()
    for t in th: t.start()
    for t in th: t.join()
    wall = time.time() - t0
    gen = sum(v[0] for v in res.values())
    per = [f"{v[0]/v[1]:.2f}" for v in res.values()]
    print(f"[{A.tag}] aggregate {gen/wall:.2f} t/s  ({gen} tok / {wall:.3f}s wall)  per-slot {per}")
finally:
    if proc is not None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=60)
        except Exception:
            try: os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception: pass
