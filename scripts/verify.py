#!/usr/bin/env python3
"""llm170 CPU 엔진 정확도 검증 하니스 (기준: llama-server greedy).

케이스 (검증 매트릭스: 짧은/장문 × 단일/np — 4종 + 장기 생성):
  1. single_short / single_ko / single_code — 단일 스트림 짧은 프롬프트
  2. np4 — 4개 프롬프트 병렬 배치 디코드 (상태 격리 검증)
  3. long_prompt — 장문(~2400토큰) 단일 prefill (GDN chunked 경로 장문)
  4. long_np2 — 장문 프롬프트 2종 병렬 (장문 prefill + np 조합)
  5. long_gen96 — 장기 생성 (누적 상태 오차)

토큰 id는 기준 서버 /tokenize로 통일 — 토크나이저 표면은 비교 범위 밖.
greedy(temp 0) 토큰열이 정확히 일치해야 통과 (기준/자체 모두 f32 KV).
LLM170_EXTRA_ARGS 환경변수로 자체 엔진 인자 추가 가능 (예: --backend gpu).

기준 서버 (재현 인자):
  llama-server -m Qwen3.8-27B-UD-Q4_K_XL.gguf --port 10090 -ngl 0 -c 8192 \
      -ctk f32 -ctv f32 -fa off --temp 0   # (-fa on/off 모두 동일 결과)

알려진 근접 마진 발산 (2026-08-31): 본 엔진은 가중치를 f32로 디양자화해 내적하는
반면 llama.cpp CPU는 활성을 q8_0으로 재양자화해 정수 내적한다. 두 경로의 양자화
노이스 차이가 로짓 갭 ~0.3 이하의 근접 토큰에서 스트림을 갈라놓는다
(single_short/single_ko/long_gen96이 이 범주 — top-k 갭 실측 0.27).
GPU(HIP/Vulkan)는 CPU와 동일 누산 순서라 발산 지점도 동일하다.
근본 해소는 ggml q8-내적 산술 재현이 필요 — 별도 과제.
"""
import json
import os
import subprocess
import sys
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 10090
N_PREDICT_DEFAULT = 24
BIN = "target/release/llm170"
MODEL_PATH = "/home/yoon/local_llm/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf"


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
            os.environ.get("LLM170_MODEL", MODEL_PATH)]
    args += os.environ.get("LLM170_EXTRA_ARGS", "").split()
    args += ["--n-predict", str(n_predict), "--ctx", str(ctx)]
    for p in prompts:
        args += ["--prompt-tokens", ",".join(map(str, p))]
    r = subprocess.run(args, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-4000:])
        raise RuntimeError(f"llm170 infer rc={r.returncode}")
    seqs = {}
    for line in r.stdout.splitlines():
        j = json.loads(line)
        seqs.setdefault(j["seq"], []).append(j["token"])
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
    # --- 장문 프롬프트 (GDN chunked prefill 장문) ---
    para = ("The quick brown fox jumps over the lazy dog. Pack my box with "
            "five dozen liquor jugs. How vexingly quick daft zebras jump! "
            "Sphinx of black quartz, judge my vow. ")
    long_ids = None
    for k in range(2, 120):
        ids = tokenize(para * k)
        if len(ids) >= 2300:
            long_ids = ids
            break
    assert long_ids, "장문 프롬프트 생성 실패"
    base = baseline_generate(long_ids, N_PREDICT_DEFAULT)
    ours = ours_generate([long_ids], N_PREDICT_DEFAULT, 4096)[0]
    results.append(compare("long_prompt", base, ours))

    # --- 장문 + np 복수 (검증 매트릭스 4종 완성) ---
    para2 = ("서울은 대한민국의 수도이며 한강이 도시를 가로지른다. 부산은 "
             "대한민국 제2의 도시로 항구 도시로 발전했다. 대전은 과학 도시로서 "
             "대덕연구단지를 품고 있다. 광주는 예술의 도시로 알려져 있다. ")
    long_ids2 = None
    for k in range(2, 160):
        ids = tokenize(para2 * k)
        if len(ids) >= 1900:
            long_ids2 = ids
            break
    assert long_ids2, "장문 프롬프트2 생성 실패"
    base1 = baseline_generate(long_ids, N_PREDICT_DEFAULT)
    base2 = baseline_generate(long_ids2, N_PREDICT_DEFAULT)
    ours = ours_generate([long_ids, long_ids2], N_PREDICT_DEFAULT, 4096)
    results.append(compare("long_np2_seq0", base1, ours[0]))
    results.append(compare("long_np2_seq1", base2, ours[1]))

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
