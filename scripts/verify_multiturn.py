#!/usr/bin/env python3
"""멀티턴(연속 턴) 검증 (plans/28 후속) — 서버 접두 캐시 + KV 이어쓰기.

불변식: 턴2 응답 == 전체 프롬프트(턴1+생성+후속질문)를 단일 CLI prefill로
돌린 스트림 (완전일치 — 접두 캐시·슬롯 재사용이 스트림을 바꾸지 않는다).
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

BIN = "target/release/llm170"
MODEL = "/home/yoon/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf"
PORT = 18081
N = 16

st = json.load(open('/tmp/verify_q35_base.json'))
turn1 = st['prompts']['short0']
follow = st['prompts']['short1']


def cli_full(ids):
    env = dict(os.environ)
    env["LLM170_SLOTS"] = "1"
    args = [BIN, "infer", "--model", MODEL, "--backend", "gpu",
            "--n-predict", str(N), "--ctx", "4096",
            "--prompt-tokens", ",".join(map(str, ids))]
    r = subprocess.run(args, capture_output=True, text=True, timeout=3600, env=env)
    assert r.returncode == 0, r.stderr[-300:]
    return [json.loads(l)["token"] for l in r.stdout.splitlines()
            if l.startswith("{")][:N]


def main():
    # 서버 기동
    env = dict(os.environ)
    env["LLM170_SLOTS"] = "1"
    proc = subprocess.Popen([BIN, "serve", "--model", MODEL, "--port", str(PORT),
                             "--ctx", "4096", "--backend", "gpu"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            env=env)
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

        def gen(ids, n):
            req = urllib.request.Request(
                f"{base}/completion",
                data=json.dumps({"prompt": ids, "n_predict": n}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=3600) as r:
                return json.loads(r.read())["tokens"]

        # 턴1
        t1 = gen(turn1, N)
        # 턴2: 턴1 프롬프트 + 턴1 생성 + 후속 질문 (접두 캐시 경로)
        turn2_prompt = turn1 + t1 + follow
        t2 = gen(turn2_prompt, N)
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except Exception:
            proc.kill()

    # 기준: 동일 전체 프롬프트 CLI 단일 실행
    ref = cli_full(turn2_prompt)
    ok = t2 == ref
    k = next((i for i, (a, b) in enumerate(zip(ref, t2)) if a != b), None)
    print(f"[{'PASS' if ok else 'FAIL'}] multiturn_turn2: server cached-continuation "
          f"== CLI full-prefill ({len(t2)} vs {len(ref)} tok)"
          + ("" if ok else f" — 첫 불일치 @[{k}]"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
