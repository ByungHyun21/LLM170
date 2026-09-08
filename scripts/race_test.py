#!/usr/bin/env python3
"""spec==nonspec 결정론성 시험 — N회 반복 집계 (plans/35 회귀 조사)."""
import json, subprocess, os, sys

st = json.load(open('/tmp/verify_q35_base.json'))
shorts = [st["prompts"][f"short{i}"] for i in range(4)]

def run(binary, spec):
    args = [binary, "infer", "--model", "/home/yoon/models/qwen3.8-27b/q35work.gguf"]
    if spec:
        args += ["--spec", "4"]
    args += ["--n-predict", "24", "--ctx", "2048", "--gpu-runtime", "vulkan"]
    for p in shorts:
        args += ["--prompt-tokens", ",".join(map(str, p))]
    env = dict(os.environ)
    env["LLM170_SPEC_GPU"] = "1"
    r = subprocess.run(args, capture_output=True, text=True, timeout=1800, env=env)
    assert r.returncode == 0, r.stderr[-200:]
    seqs = {}
    for line in r.stdout.splitlines():
        j = json.loads(line)
        seqs.setdefault(j["seq"], []).append(j["token"])
    return [seqs.get(i, []) for i in range(4)]

binary = sys.argv[1] if len(sys.argv) > 1 else "target/release/llm170"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
plain = run(binary, False)
diffs = 0
for i in range(n):
    spec = run(binary, True)
    bad = [j for j in range(4) if plain[j] != spec[j]]
    print(f"run{i}: {'OK' if not bad else 'DIFF seq' + str(bad)}", flush=True)
    if bad:
        diffs += 1
print(f"{binary}: {diffs}/{n} runs diverged")
