#!/usr/bin/env python3
"""np4 serve 재현 — stderr 보존, 토큰 출력. 사용: repro_np4.py [--spec 4]"""
import json, os, subprocess, sys, threading, time, urllib.request, signal

BIN = "target/release/llm170"
MODEL = os.environ.get("LLM170_MODEL", "/home/yoon/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf")
STORE = os.environ.get("LLM170_STORE", "/tmp/verify_q35_base.json")
PORT = int(os.environ.get("LLM170_SERVE_PORT", "18081"))
N = int(os.environ.get("N", "24"))
SPEC = sys.argv[1] == "--spec" if len(sys.argv) > 1 else 0

with open(STORE) as f:
    pr = json.load(f)["prompts"]
longs = [pr[k] for k in ("long1", "long2", "long3", "long4")]

env = dict(os.environ)
env["LLM170_SLOTS"] = "4"
errf = open("/tmp/serve_err.log", "wb")
args = [BIN, "serve", "--model", MODEL, "--port", str(PORT), "--ctx", "4096", "--backend", "gpu"]
if SPEC:
    args += ["--spec", str(SPEC)]
proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=errf, env=env)
base = f"http://127.0.0.1:{PORT}"
try:
    for _ in range(300):
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
                if r.status == 200:
                    break
        except Exception:
            if proc.poll() is not None:
                print("serve 조기 종료"); sys.exit(1)
            time.sleep(2)
    outs = [None] * 4
    def worker(i, ids):
        try:
            req = urllib.request.Request(
                f"{base}/completion",
                data=json.dumps({"prompt": ids, "n_predict": N}).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=3600) as r:
                outs[i] = json.loads(r.read())["tokens"]
        except Exception as e:
            print(f"worker{i} 예외: {e}")
    ths = [threading.Thread(target=worker, args=(i, p)) for i, p in enumerate(longs)]
    for t in ths: t.start()
    for t in ths: t.join()
    for i, o in enumerate(outs):
        print(f"seq{i}: {o[:8] if o else o} (len {len(o) if o else 0})")
finally:
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=60)
    except Exception:
        proc.kill()
    errf.close()
    print("--- stderr tail ---")
    os.system("tail -30 /tmp/serve_err.log")
