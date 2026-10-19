#!/usr/bin/env python3
# 피어 대조 — vLLM(참조 구현)과 우리 엔진의 절대 정합 측정. eval의 상대 게이트가
# 못 잡는 공통모드 오류(예: 2026-10-19 토크나이저 폴백 → 35B PPL 15.5 vs 9.5)를
# 잡는 그물. 두 모드:
#   ppl  — 코퍼스 PPL(vLLM prompt_logprobs) ↔ 우리 eval 덤프 대조
#   gen  — 생성 배터리: 같은 토큰열 프롬프트 → 양쪽 그리디 토큰열 → 일치율
# 사용:
#   envs/vllm/bin/python peer_check.py ppl  <model_dir> <corpus> [--ours dump.tsv] [--bos N]
#   envs/vllm/bin/python peer_check.py gen  <model_dir> [--n-predict 16] [--ours-json out.json]
# 전제: 같은 기기 vLLM(:19001) 서빙 중. 우리 생성 = llm170 w4a16-gpu.
import json, urllib.request, math, sys, subprocess, os

BASE = "http://127.0.0.1:19001"
LLM170 = os.environ.get("LLM170_BIN", "/home/yoon/LLM170/target/release/llm170")
CORPORA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval-data")

# 생성 배터리 — 코퍼스 머리 조각(모델 로드 1회로 전부 소화). (코퍼스, 시작토큰, 길이)
BATTERY = [
    ("corpus-en.txt", 0, 64),
    ("corpus-en.txt", 512, 64),
    ("corpus-ko.txt", 0, 64),
    ("corpus-mix33k.txt", 0, 64),
    ("corpus-json.txt", 0, 64),
]

def post(path, body, timeout=3600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def ids_of(model, corpus):
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    return tk.encode(open(corpus, encoding="utf-8").read())

def mode_ppl(args):
    model, corpus = args[0], args[1]
    ours_dump = args[args.index("--ours") + 1] if "--ours" in args else None
    bos = int(args[args.index("--bos") + 1]) if "--bos" in args else None
    ids = ids_of(model, corpus)
    pr = ([bos] if bos is not None else []) + ids
    r = post("/v1/completions", {"model": model, "prompt": pr, "max_tokens": 1,
                                 "temperature": 0, "prompt_logprobs": 0})
    pl = r["choices"][0]["prompt_logprobs"]
    nlls = [-list(p.values())[0]["logprob"] for p in pl if p]
    vllm_ppl = math.exp(sum(nlls) / len(nlls))
    print(f"peer: vLLM PPL {vllm_ppl:.4f} ({len(nlls)} 위치, {os.path.basename(corpus)})")
    if ours_dump:
        tot = 0.0; n = 0
        for line in open(ours_dump):
            if line and not line.startswith("#"):
                f = line.split("\t")
                tot += float(f[3]); n += 1
        ours_ppl = math.exp(tot / n)
        d = (ours_ppl / vllm_ppl - 1) * 100
        print(f"ours: PPL {ours_ppl:.4f} ({n} 위치) · Δ {d:+.3f}%")

def mode_gen(args):
    model = args[0]
    n_pred = int(args[args.index("--n-predict") + 1]) if "--n-predict" in args else 16
    out_json = args[args.index("--ours-json") + 1] if "--ours-json" in args else None
    results = []
    for corpus, start, length in BATTERY:
        ids = ids_of(model, os.path.join(CORPORA, corpus))[start:start + length]
        # vLLM 그리디
        r = post("/v1/completions", {"model": model, "prompt": ids, "max_tokens": n_pred,
                                     "temperature": 0, "return_token_ids": True})
        v_ids = r["choices"][0].get("token_ids") or []
        # 우리 엔진
        pr = ",".join(map(str, ids))
        cp = subprocess.run([LLM170, "w4a16-gpu", model, "--prompt-tokens", pr,
                             "--n-predict", str(n_pred), "--ctx", "1024"],
                            capture_output=True, text=True, timeout=1800,
                            cwd="/home/yoon/LLM170")
        ours = []
        for line in cp.stdout.splitlines():
            if line.startswith(" tokens:"):
                ours = [int(x) for x in line.split(":", 1)[1].strip().split(",") if x]
                break
        m = min(len(v_ids), len(ours))
        agree = sum(1 for i in range(m) if v_ids[i] == ours[i])
        pre = 0
        for i in range(m):
            if v_ids[i] == ours[i]:
                pre += 1
            else:
                break
        results.append({"corpus": corpus, "start": start, "vllm": v_ids, "ours": ours,
                        "agree": agree, "n": m, "prefix": pre})
        print(f"gen {corpus}[{start}:+{length}]: 일치 {agree}/{m} · 접두 {pre} · "
              f"vLLM={v_ids[:8]} ours={ours[:8]}")
    if results:
        ta = sum(r["agree"] for r in results); tn = sum(r["n"] for r in results)
        print(f"gen 합계: 토큰 일치 {ta}/{tn} ({ta/max(tn,1)*100:.1f}%)")
    if out_json:
        json.dump(results, open(out_json, "w"), indent=1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(2)
    mode, rest = sys.argv[1], sys.argv[2:]
    (mode_ppl if mode == "ppl" else mode_gen)(rest)
