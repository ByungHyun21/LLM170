#!/usr/bin/env python3
"""np 배치 디코드 == 순차 디코드 자기일관성 검증 (llama 불필요).

t≥2 경로(np 배치 / MTP verify 계열)를 바꾼 뒤 게이트(t=1 전용)가 닿지 않는
영역을 직접 검증한다. 같은 프롬프트를 (1) 단독 1시퀀스로, (2) 4시퀀스 배치로
돌려 토큰 스트림을 행별 비교한다.

사용:
  python3 scripts/verify_np_self.py [--model <gguf>] [--n 24] [--ctx 4096]
                                    [--prompt-file <토큰 csv>] [--tool llm170]
종료코드: 완전일치 0, 불일치 있으면 1 (근접티는 보고만 — 사람 판정).
"""
import argparse
import json
import os
import subprocess
import sys

P = argparse.ArgumentParser()
P.add_argument("--model", default=os.environ.get(
    "LLM170_M27", "/home/yoon/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf"))
P.add_argument("--bin", default="target/release/llm170")
P.add_argument("--n", type=int, default=24)
P.add_argument("--ctx", type=int, default=4096)
P.add_argument("--np", type=int, default=4)
P.add_argument("--prompt-file", default=None, help="콤마 구분 토큰 id (기본: 내장 4종)")
P.add_argument("--env", action="append", default=[], help="추가 환경변수 KEY=VAL")
A = P.parse_args()

GATE = ("386,18,15,15,643,20,20,19586,5876,8058,4144,67,21,7307,22,20,23,24902,17,16,23,"
        "386,18,66,19,386,17,24,19,24902,16,16,19586,66,21,65,23,1692,22,22,19,4144,341,15,11")

if A.prompt_file:
    base = [int(x) for x in open(A.prompt_file).read().strip().split(",")]
    prompts = [base]
else:
    long_ids = [int(x) for x in GATE.split(",")]
    # 서로 다른 4종: 접두를 다르게 잘라 배치 내 라우팅/상태 격리를 흔든다.
    prompts = [
        long_ids,
        long_ids[:28] + [1692, 20, 67, 15],
        long_ids[:16] + [643, 21, 20, 19, 643, 20, 20, 23, 1692, 20, 732],
        long_ids[:40] + [386, 66, 18, 15, 17374],
    ]

env = dict(os.environ)
for kv in A.env:
    k, _, v = kv.partition("=")
    env[k] = v


def run(seqs):
    cmd = [A.bin, "infer", "--model", A.model, "--n-predict", str(A.n),
           "--ctx", str(A.ctx), "--backend", "gpu"] + ["--gpu-runtime", "hip"]
    for s in seqs:
        cmd += ["--prompt-tokens", ",".join(str(t) for t in s)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print("infer 실패:", r.returncode, r.stderr[-500:])
        sys.exit(2)
    out = {}
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        j = json.loads(line)
        out.setdefault(j["seq"], []).append(j["token"])
    return [out.get(i, []) for i in range(len(seqs))]


print(f"model: {A.model}")
print(f"np={len(prompts)} n={A.n} ctx={A.ctx}")
seq = [run([p])[0] for p in prompts]
print("순차 완료")
bat = run(prompts)
print("배치 완료")

bad = 0
for i, (a, b) in enumerate(zip(seq, bat)):
    common = min(len(a), len(b))
    diff = [k for k in range(common) if a[k] != b[k]]
    status = "IDENTICAL" if not diff and len(a) == len(b) else (
        f"DIFF at {diff[:3]}" if diff else f"LEN {len(a)} vs {len(b)}")
    if diff or len(a) != len(b):
        bad += 1
    print(f"seq{i}: {status}  ({len(a)} vs {len(b)} tok)")
    if diff:
        k = diff[0]
        print(f"       ctx: 순차 {a[max(0,k-2):k+3]} / 배치 {b[max(0,k-2):k+3]}")

print(f"== {len(prompts) - bad}/{len(prompts)} identical ==")
sys.exit(1 if bad else 0)
