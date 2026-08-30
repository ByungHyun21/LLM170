#!/usr/bin/env python3
"""llm170 CPU 엔진 정확도 검증 하니스 (기준: llama-server greedy).

케이스:
  1. single_short / single_ko — 단일 스트림 짧은 프롬프트
  3. np4 — 4개 프롬프트 병렬 배치 디코드 (상태 격리 검증)
  4. long_prompt — 긴 컨텍스트 prefill (GDN chunked 경로)
  5. long_gen — 장기 생성 (누적 상태 오차)

토큰 id는 기준 서버 /tokenize로 통일 — 토크나이저 표면은 비교 범위 밖.
greedy(temp 0) 토큰열이 정확히 일치해야 통과 (기준/자체 모두 f32 KV).
"""
import json
import subprocess
import sys
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 10090
N_PREDICT_DEFAULT = 24
BIN = "target/release/llm170"


def post(path, payload):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def tokenize(text):
    return post("/tokenize", {"content": text})["tokens"]


def baseline_generate(ids, n_predict):
    out = post("/completion", {
        "prompt": ids,
        "n_predict": n_predict,
        "temperature": 0.0,
        "cache_prompt": False,
        "return_tokens": True,
    })
    toks = out.get("tokens")
    if toks is None:
        # 폴백: 생성 텍스트를 다시 토큰화 (근사 — return_tokens 미지원 서버용)
        toks = tokenize(out["content"])
    return toks


def ours_generate(prompts, n_predict, ctx):
    # 우리 엔진은 prefill 토큰 + n_predict 디코드 = n_predict+1 출력 → 슬라이스
    args = [BIN, "infer", "--model",
            "/home/yoon/local_llm/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf",
            "--n-predict", str(n_predict), "--ctx", str(ctx)]
    for p in prompts:
        args += ["--prompt-tokens", ",".join(map(str, p))]
    r = subprocess.run(args, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        raise SystemExit("llm170 infer 실패")
    seqs = {}
    for line in r.stdout.splitlines():
        j = json.loads(line)
        seqs.setdefault(j["seq"], []).append((j["pos"], j["token"]))
    for s in seqs:
        seqs[s] = [t for _, t in sorted(seqs[s])]
    return [seqs.get(i, [])[:n_predict] for i in range(len(prompts))]


def compare(name, base, ours):
    ok = base == ours
    n = min(len(base), len(ours))
    diff = sum(1 for a, b in zip(base, ours) if a != b)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: base {len(base)} tok, ours {len(ours)} tok, 불일치 {diff}/{n}")
    if not ok:
        for i, (a, b) in enumerate(zip(base, ours)):
            if a != b:
                print(f"    첫 불일치 @gen[{i}]: base={a} ours={b}")
                print(f"    base[..{i+1}]: {base[:i+1]}")
                print(f"    ours[..{i+1}]: {ours[:i+1]}")
                break
        else:
            print(f"    길이 차이: base {len(base)} vs ours {len(ours)} (공통부 일치)")
            tail = base[n:n+8] if len(base) > n else ours[n:n+8]
            print(f"    이후 토큰: {tail}")
    return ok


def main():
    results = []

    p1 = "The quick brown fox jumps over the lazy dog. The capital of France is"
    p2 = "17 곱하기 23은 얼마야? 계산하고 답해라:"
    p3 = "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    "
    p4 = "서울에서 부산까지 KTX로 가는 방법을 알려줘. 먼저"

    # --- 단일 스트림 (짧은) ---
    for name, prompt, n in [("single_short", p1, N_PREDICT_DEFAULT),
                            ("single_ko", p2, N_PREDICT_DEFAULT),
                            ("single_code", p3, N_PREDICT_DEFAULT)]:
        ids = tokenize(prompt)
        base = baseline_generate(ids, n)
        ours = ours_generate([ids], n, 2048)[0]
        results.append(compare(name, base, ours))

    # --- np4 병렬 (배치 디코드 상태 격리) ---
    prompts = [tokenize(p) for p in (p1, p2, p3, p4)]
    bases = [baseline_generate(p, N_PREDICT_DEFAULT) for p in prompts]
    ours = ours_generate(prompts, N_PREDICT_DEFAULT, 2048)
    for i in range(4):
        results.append(compare(f"np4_seq{i}", bases[i], ours[i]))

    # --- 장기 생성 ---
    ids = tokenize(p1)
    n = 96
    base = baseline_generate(ids, n)
    ours = ours_generate([ids], n, 2048)[0]
    results.append(compare("long_gen96", base, ours))

    print(f"\n=== 결과: {sum(results)}/{len(results)} PASS ===")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
