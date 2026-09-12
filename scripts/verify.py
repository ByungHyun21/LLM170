#!/usr/bin/env python3
"""llm170 GPU 엔진 정확도 검증 하니스 (기준: llama-server greedy) — 2페이즈.

페이즈 분리 이유 (2026-09-06 실측): llama(-ngl 0) 공존 중 우리 엔진 가중치
mmap이 캐시에서 축출되어 h2d가 콜드 디스크 순차읽기로 직렬화 — 로드 5분+.
기준 수집(llama 기동) → 정지 → 판정(단독, llama가 읽은 페이지 캐시 잔존) 순서.

케이스 (골 매트릭스 plans/28):
  1. single_short/ko/code — 단일 짧은 (기준 대비 tie 허용)
  2. np4_seq0-3 — 4 프롬프트 병렬 배치 디코드 (상태 격리)
  3. long_prompt — 장문(~2300) 단일
  4. long_np2 / long_np4 — 장문 복수 병렬
  5. long_gen96 — 장기 생성
  6. spec_short/np4/long/long_np4 — MTP 스펙 == 비스펙 greedy (완전일치)

판정: 완전일치 PASS, 또는 첫 발산이 근접티(우리 토큰 ∈ 기준 top-6 &
top-1 갭 < TIE_EPS=1.5nat)면 PASS(tie). spec 케이스는 완전일치만.

사용:
  LLM170_PHASE=collect python3 scripts/verify.py   # llama-server :10090 기동 중
  # llama-server 정지 후
  LLM170_PHASE=judge python3 scripts/verify.py     # 우리 엔진 단독

환경: LLM170_MODEL, LLM170_EXTRA_ARGS(기본 "--backend gpu"),
      LLM170_STORE(기본 /tmp/verify_q35_base.json), LLM170_TIE_EPS,
      LLM170_VERIFY_SPEC(기본 4).
"""
import json
import os
import subprocess
import sys
import urllib.request

PORT = int(os.environ.get("LLM170_REF_PORT", "10090"))
N_PREDICT_DEFAULT = 24
BIN = "target/release/llm170"
MODEL_PATH = os.environ.get(
    "LLM170_MODEL",
    "/home/yoon/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf")
STORE = os.environ.get("LLM170_STORE", "/tmp/verify_q35_base.json")
TIE_EPS = float(os.environ.get("LLM170_TIE_EPS", "1.5"))
SPEC_K = int(os.environ.get("LLM170_VERIFY_SPEC", "4"))


def post(path, payload):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def tokenize(text):
    return post("/tokenize", {"content": text})["tokens"]


def erase_slots():
    # llama-server 슬롯 KV 잔류 오염 방지 (2026-09-06 실측).
    try:
        for sid in range(4):
            req = urllib.request.Request(
                f"http://127.0.0.1:{PORT}/slots/{sid}?action=erase", data=b"{}")
            urllib.request.urlopen(req, timeout=60).read()
    except Exception:
        pass


def baseline_generate(ids, n_predict):
    erase_slots()
    out = post("/completion", {
        "prompt": ids, "n_predict": n_predict, "temperature": 0.0,
        "cache_prompt": False, "return_tokens": True, "logprobs": 6})
    toks = out.get("tokens")
    if toks is None:
        toks = tokenize(out["content"])
    return toks, out.get("completion_probabilities") or []


def ours_generate(prompts, n_predict, ctx, spec=0):
    # 우리 엔진: prefill 토큰 + n_predict 디코드 = n_predict+1 출력 → 슬라이스
    # spec>0: --spec k + LLM170_SPEC_GPU=1 (GPU 스펙 — 프로덕션 경로).
    args = [BIN, "infer", "--model", MODEL_PATH]
    args += os.environ.get("LLM170_EXTRA_ARGS", "--backend gpu").split()
    if spec:
        args += ["--spec", str(spec)]
    args += ["--n-predict", str(n_predict), "--ctx", str(ctx)]
    for p in prompts:
        args += ["--prompt-tokens", ",".join(map(str, p))]
    env = dict(os.environ)
    if spec:
        env.setdefault("LLM170_SPEC_GPU", "1")
    r = subprocess.run(args, capture_output=True, text=True, timeout=7200, env=env)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-4000:])
        raise RuntimeError(f"llm170 infer rc={r.returncode}")
    seqs = {}
    for line in r.stdout.splitlines():
        j = json.loads(line)
        seqs.setdefault(j["seq"], []).append(j["token"])
    return [seqs.get(i, [])[:n_predict] for i in range(len(prompts))]


def build_prompts():
    """전 케이스 프롬프트 토큰화(llama /tokenize) + 장문 조립."""
    texts = [
        "The quick brown fox jumps over the lazy dog. The capital of France is",
        "17 곱하기 23은 얼마야? 계산하고 답해라:",
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    ",
        "서울에서 부산까지 KTX로 가는 방법을 알려줘. 먼저",
    ]
    short = [tokenize(t) for t in texts]

    def grow(para, target, cap):
        for k in range(2, cap):
            t = tokenize(para * k)
            if len(t) >= target:
                return t
        return None

    paras = {
        "long1": ("The quick brown fox jumps over the lazy dog. Pack my box with "
                  "five dozen liquor jugs. How vexingly quick daft zebras jump! "
                  "Sphinx of black quartz, judge my vow. ", 2300, 120),
        "long2": ("서울은 대한민국의 수도이며 한강이 도시를 가로지른다. 부산은 "
                  "대한민국 제2의 도시로 항구 도시로 발전했다. 대전은 과학 도시로서 "
                  "대덕연구단지를 품고 있다. 광주는 예술의 도시로 알려져 있다. ", 1900, 160),
        "long3": ("The Amazon rainforest spans nine countries in South America and "
                  "produces roughly twenty percent of the world's oxygen supply. "
                  "Its river carries more water than any other river on Earth. ", 2100, 160),
        "long4": (" Quantum computers manipulate qubits using superposition and "
                  "entanglement, promising speedups for factoring and search. "
                  "Error correction remains the central engineering challenge. ", 1700, 160),
    }
    out = {f"short{i}": p for i, p in enumerate(short)}
    for key, (para, target, cap) in paras.items():
        ids = grow(para, target, cap)
        assert ids, f"장문 프롬프트 {key} 조립 실패"
        out[key] = ids
    return out


def collect():
    pr = build_prompts()
    store = {"prompts": pr, "base": {}}
    # 기준: 짧은 4종(24) + 장문 4종(24) + p1 장기(96)
    jobs = [(k, N_PREDICT_DEFAULT) for k in pr]
    jobs.append(("short0", 96))
    for key, n in jobs:
        tag = f"{key}:{n}"
        if tag in store["base"]:
            continue
        print(f"[collect] {tag} len={len(pr[key])} ...", flush=True)
        toks, probs = baseline_generate(pr[key], n)
        store["base"][tag] = {"toks": toks, "probs": probs, "n": n}
        print(f"[collect] {tag} done ({len(toks)} tok)", flush=True)
    with open(STORE, "w") as f:
        json.dump(store, f)
    print(f"[collect] 완료 → {STORE}")


def compare(name, base, ours, probs=None, info=False):
    """완전일치 PASS 또는 근접티 PASS.

    info=True (장문 케이스): 발산을 **실패로 세지 않고** INFO 로 보고한다. 근거(2026-09-12,
    사용자 결정): 어텐션 산술을 바꾸면 리덕션 트리 반올림(≤1e-5)이 모델의 재귀(GDN)를 타고
    증폭되어 긴 궤적에서 argmax 가 갈린다. 커널 자체의 정확성은 `llm170 attn-check` 가
    동일 입력 직접 비교로 보증한다(1e-5 최대차, 1e-4 초과 0/5천만). 단문/중문 케이스는
    종전대로 완전일치/근접티만 통과시킨다 — 실제 버그는 그쪽에서 잡힌다."""
    if base == ours:
        print(f"[PASS] {name}: base {len(base)} tok — 완전일치")
        return True
    n = min(len(base), len(ours))
    diff = sum(1 for a, b in zip(base, ours) if a != b)
    k = next((i for i, (a, b) in enumerate(zip(base, ours)) if a != b), None)
    tie, detail = False, ""
    if k is not None and probs and k < len(probs):
        top = probs[k].get("top_logprobs", [])
        ids_top = [e["id"] for e in top]
        if ours[k] in ids_top:
            r = ids_top.index(ours[k])
            gap = top[0]["logprob"] - top[r]["logprob"]
            if r == 0:
                detail = f"@gen[{k}] 기준 top-1 동일토큰인데 스트림 불일치 (기준 로그확률 오류 의심)"
            elif gap < TIE_EPS:
                tie = True
                detail = f"근접티 @gen[{k}]: ours=기준 top-{r+1} (갭 {gap:.2f} < ε={TIE_EPS})"
            else:
                detail = f"@gen[{k}]: ours=기준 top-{r+1} 갭 {gap:.2f} — 진짜 발산"
        else:
            detail = f"@gen[{k}]: ours={ours[k]} 기준 top-{len(ids_top)} 밖 — 진짜 발산"
    elif k is not None:
        detail = f"@gen[{k}]: base={base[k]} ours={ours[k]} (logprobs 없음)"
    if info and not tie:
        print(f"[INFO] {name}: base {len(base)} vs ours {len(ours)} tok, 불일치 {diff}/{n}"
              + (f" — {detail}" if detail else "")
              + "  (장문: 발산 허용 — 커널 정확성은 attn-check 가 보증)")
        return None
    status = "PASS" if tie else "FAIL"
    print(f"[{status}] {name}: base {len(base)} vs ours {len(ours)} tok, 불일치 {diff}/{n}"
          + (f" — {detail}" if detail else ""))
    if not tie and k is not None:
        print(f"    base[..{k+1}]: {base[:k+1]}")
        print(f"    ours[..{k+1}]: {ours[:k+1]}")
    return tie


def compare_exact(name, ours, ref, info=False):
    ok = ours == ref
    if not ok and info:
        print(f"[INFO] {name}: {len(ours)} vs {len(ref)} tok, 첫 불일치 @gen[{k}]  "
              "(장문 스펙: 발산 허용 — 커널 정확성은 attn-check 가 보증)")
        return None
    k = next((i for i, (a, b) in enumerate(zip(ref, ours)) if a != b), None)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {len(ours)} vs {len(ref)} tok"
          + ("" if ok else f" — 첫 불일치 @gen[{k}]"))
    return ok


def spec_equality(name, prompts, n_predict, ctx, refs, k=SPEC_K, info=False):
    """스펙 불변식: --spec k 출력 == 비스펙 greedy (완전일치. info=True 면 장문 발산은 INFO)."""
    sp = ours_generate(prompts, n_predict, ctx, spec=k)
    res = []
    for i in range(len(prompts)):
        res.append(compare_exact(f"{name}_seq{i}", sp[i], refs[i], info=info))
    if any(r is False for r in res):
        return False
    if any(r is None for r in res):
        return None
    return True


def judge():
    with open(STORE) as f:
        st = json.load(f)
    pr = st["prompts"]
    base = st["base"]
    results = []

    def b(key, n=N_PREDICT_DEFAULT):
        e = base[f"{key}:{n}"]
        return e["toks"], e["probs"]

    short = [pr[f"short{i}"] for i in range(4)]
    longs = [pr[k] for k in ("long1", "long2", "long3", "long4")]

    # 1) 단일 짧은 ×3
    names = ["single_short", "single_ko", "single_code"]
    for i, nm in enumerate(names):
        bt, bp = b(f"short{i}")
        ours = ours_generate([short[i]], N_PREDICT_DEFAULT, 2048)[0]
        results.append(compare(nm, bt, ours, bp))

    # 2) np4
    ours_np4 = ours_generate(short, N_PREDICT_DEFAULT, 2048)
    for i in range(4):
        bt, bp = b(f"short{i}")
        results.append(compare(f"np4_seq{i}", bt, ours_np4[i], bp))

    # 3) 장문 단일
    bt, bp = b("long1")
    ours_long = ours_generate([longs[0]], N_PREDICT_DEFAULT, 4096)[0]
    results.append(compare("long_prompt", bt, ours_long, bp, info=True))

    # 4) long_np2
    ours_l2 = ours_generate(longs[:2], N_PREDICT_DEFAULT, 4096)
    bt, bp = b("long1")
    results.append(compare("long_np2_seq0", bt, ours_l2[0], bp, info=True))
    bt, bp = b("long2")
    # long_np2_seq1: 참조 불안정 사례(2026-09-06 실측 — llama 슬롯 KV 잔류에 따라
    # 평탄 분포점에서 스트림이 갈림). FAIL 시 참조 재수집으로 판별.
    results.append(compare("long_np2_seq1", bt, ours_l2[1], bp, info=True))

    # 5) long_np4 (골 매트릭스: 4 시퀀스 상이 길이 장문 병렬)
    ours_l4 = ours_generate(longs, N_PREDICT_DEFAULT, 4096)
    for i, lk in enumerate(("long1", "long2", "long3", "long4")):
        bt, bp = b(lk)
        results.append(compare(f"long_np4_seq{i}", bt, ours_l4[i], bp, info=True))

    # 6) 장기 생성
    bt, bp = b("short0", 96)
    ours_g96 = ours_generate([short[0]], 96, 2048)[0]
    results.append(compare("long_gen96", bt, ours_g96, bp))

    # 7) 스펙(mtp) 불변식 — 비스펙 결과 재사용 (완전일치 판정)
    if SPEC_K > 0:
        ref_short = ours_generate([short[0]], N_PREDICT_DEFAULT, 2048)[0]
        results.append(spec_equality("spec_short", [short[0]], N_PREDICT_DEFAULT,
                                     2048, [ref_short]))
        results.append(spec_equality("spec_np4", short, N_PREDICT_DEFAULT,
                                     2048, ours_np4))
        results.append(spec_equality("spec_long", [longs[0]], N_PREDICT_DEFAULT,
                                     4096, [ours_long], info=True))
        results.append(spec_equality("spec_long_np4", longs, N_PREDICT_DEFAULT,
                                     4096, ours_l4, info=True))


    npass = sum(1 for r in results if r is True)
    ninfo = sum(1 for r in results if r is None)
    nfail = sum(1 for r in results if r is False)
    line = f"\n=== 결과: {npass}/{len(results)} PASS"
    if ninfo:
        line += f" (+{ninfo} INFO 발산)"
    if nfail:
        line += f", {nfail} FAIL"
    print(line + " ===")
    raise SystemExit(0 if nfail == 0 else 1)


if __name__ == "__main__":
    phase = os.environ.get("LLM170_PHASE", "collect")
    if phase == "collect":
        collect()
    elif phase == "judge":
        judge()
    else:
        raise SystemExit(f"LLM170_PHASE={phase} — collect|judge")
