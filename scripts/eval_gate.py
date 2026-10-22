#!/usr/bin/env python3
import json, math, os, statistics, subprocess, sys

LLM170 = os.environ.get("LLM170_BIN", "/home/yoon/LLM170/target/release/llm170")
ROOT = "/home/yoon/LLM170"

def sh(args, timeout=3600):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, cwd=ROOT)

def load_nll(path, col):
    out = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split("\t")
        out[int(f[0])] = float(f[col])
    return out

def l2_relative(a, b):
    r = sh([LLM170, "w4a16-eval", "ci", a, b])
    txt = r.stdout + r.stderr
    ppl = None; ci = None
    for line in txt.splitlines():
        if "ΔPPL" in line and "%" in line:
            try:
                ppl = float(line.split("ΔPPL")[1].split("%")[0].split()[-1])
            except Exception:
                pass
        if "95% CI" in line and "ΔPPL" in line:
            ci = line.strip()
    print(f"[L2 상대] {ci or '(CI 미파싱)'}")
    if ppl is None:
        return None
    return ppl

def l3_peer(model, corpus, peer):
    r = sh(["python3", os.path.join(ROOT, "scripts", "peer_check.py"), "ppl", model, corpus],
           timeout=3600)
    print("[L3 피어] " + (r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:]))

def l4_truth(oracle, cand, vllm=None, name="cand"):
    o = load_nll(oracle, 1)
    def dist(p):
        c = load_nll(p, 3)
        ks = sorted(set(o) & set(c))
        return statistics.mean(abs(c[i] - o[i]) for i in ks), len(ks)
    d, n = dist(cand)
    print(f"[L4 진리] {name} |ΔNLL| 평균 {d:.6f} ({n}위치)")
    if vllm:
        dv, nv = dist(vllm)
        print(f"[L4 진리] vLLM  |ΔNLL| 평균 {dv:.6f} ({nv}위치) → "
              f"{'우리 우세' if d < dv else 'vLLM 우세'} ({(dv-d)/dv*100:+.1f}%)")

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__); sys.exit(2)
    model, corpus = args[0], args[1]
    da = args[args.index("--dump-a") + 1] if "--dump-a" in args else None
    db = args[args.index("--dump-b") + 1] if "--dump-b" in args else None
    oracle = args[args.index("--oracle") + 1] if "--oracle" in args else None
    ovllm = args[args.index("--oracle-vllm") + 1] if "--oracle-vllm" in args else None
    if da and db:
        l2_relative(da, db)
    if "--peer" in args:
        l3_peer(model, corpus, 19001)
    if oracle:
        l4_truth(oracle, db or da, ovllm)
