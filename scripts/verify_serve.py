#!/usr/bin/env python3
"""llm170 serve 슬롯 엔진 검증 (plans/28 골 매트릭스 — 서버 표면).

불변식: 서버 동시 요청(슬롯 배칭) 토큰 == CLI np 배치 토큰 (완전일치 —
같은 엔진·같은 수치 계열; 배치 구성이 스트림을 바꾸지 않는다).

케이스:
  1. serve_np4_long — 장문 4종 동시 (연속 배칭 + 상태 격리)
  2. serve_np4_long_spec — serve --spec 4 (슬롯별 MTP 스펙)

사용: python3 scripts/verify_serve.py   # 우리 엔진 단독 (llama 불필요)
환경: LLM170_MODEL, LLM170_STORE(=verify.py 수집 스토어), PORT
"""
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request

BIN = "target/release/llm170"
MODEL = os.environ.get(
    "LLM170_MODEL",
    "/home/yoon/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf")
STORE = os.environ.get("LLM170_STORE", "/tmp/verify_q35_base.json")
PORT = int(os.environ.get("LLM170_SERVE_PORT", "18080"))
N = 24


def cli_np(prompts, spec, ctx):
    env = dict(os.environ)
    env["LLM170_SPEC_GPU"] = "1"
    env["LLM170_SLOTS"] = str(len(prompts))
    args = [BIN, "infer", "--model", MODEL, "--backend", "gpu",
            "--n-predict", str(N), "--ctx", str(ctx)]
    if spec:
        args += ["--spec", str(spec)]
    for p in prompts:
        args += ["--prompt-tokens", ",".join(map(str, p))]
    r = subprocess.run(args, capture_output=True, text=True, timeout=3600, env=env)
    assert r.returncode == 0, r.stderr[-400:]
    seqs = {}
    for l in r.stdout.splitlines():
        j = json.loads(l)
        seqs.setdefault(j["seq"], []).append(j["token"])
    return [seqs.get(i, [])[:N] for i in range(len(prompts))]


def serve_and_fire(prompts, spec):
    env = dict(os.environ)
    env["LLM170_SLOTS"] = str(len(prompts))
    args = [BIN, "serve", "--model", MODEL, "--port", str(PORT),
            "--ctx", "4096", "--backend", "gpu"]
    if spec:
        args += ["--spec", str(spec)]
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, env=env)
    base = f"http://127.0.0.1:{PORT}"
    try:
        deadline = time.time() + 600
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:
                if proc.poll() is not None:
                    raise RuntimeError("serve 조기 종료")
                time.sleep(2)
        else:
            raise RuntimeError("serve 헬스 타임아웃")
        outs = [None] * len(prompts)

        def worker(i, ids):
            req = urllib.request.Request(
                f"{base}/completion",
                data=json.dumps({"prompt": ids, "n_predict": N}).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=3600) as r:
                outs[i] = json.loads(r.read())["tokens"]

        ths = [threading.Thread(target=worker, args=(i, p))
               for i, p in enumerate(prompts)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        return [(o or [])[:N] for o in outs]
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except Exception:
            proc.kill()


def compare(tag, srv, cli):
    ok = all(srv[i] == cli[i] for i in range(len(cli)))
    detail = ""
    if not ok:
        for i in range(len(cli)):
            if srv[i] != cli[i]:
                k = next((j for j, (a, b) in enumerate(zip(cli[i], srv[i]))
                          if a != b), None)
                detail += f" seq{i}@{k}"
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}: serve {len(srv)}요청 vs CLI np{len(cli)}"
          + (f" — 불일치{detail}" if detail else ""))
    return ok


def main():
    with open(STORE) as f:
        pr = json.load(f)["prompts"]
    longs = [pr[k] for k in ("long1", "long2", "long3", "long4")]
    results = []
    cli = cli_np(longs, 0, 4096)
    srv = serve_and_fire(longs, 0)
    results.append(compare("serve_np4_long", srv, cli))
    cli_s = cli_np(longs, 4, 4096)
    srv_s = serve_and_fire(longs, 4)
    results.append(compare("serve_np4_long_spec", srv_s, cli_s))
    print(f"\n=== 결과: {sum(results)}/{len(results)} PASS ===")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
