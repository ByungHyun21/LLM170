#!/usr/bin/env python3
"""llm170 비전(mmproj) 골 매트릭스 검증 (plans/28) — 2페이즈.

불변식 (우리 엔진 내부 — exact 판정):
  1. vl GPU --spec 4   == vl GPU 비스펙       (스펙 수용 불변식)
  2. vl GPU np2 --spec 4 == vl GPU np2 비스펙  (np×spec 병합)
  3. vl np2 seq0 == vl 단일                    (np 상태 격리)
  4. vl GPU LLM170_EXACT=1 == vl CPU           (비트계약)
  5. 장문 접두(~2300토큰)+이미지: 4번·1번 동일 적용
의미 판정 (llama-server --mmproj 수집 텍스트): NYT 1면 키워드.

사용:
  LLM170_VL_PHASE=collect python3 scripts/verify_vl.py  # llama --mmproj 기동 중
  # llama 정지 후
  LLM170_VL_PHASE=judge python3 scripts/verify_vl.py    # 우리 엔진 단독
"""
import json
import os
import subprocess
import sys

BIN = "target/release/llm170"
MODEL = os.environ.get("LLM170_MODEL",
                       "/home/yoon/models/qwen3.8-27b/Qwen3.8-27B-UD_Q4_X_XL")
MODEL = os.environ.get(
    "LLM170_MODEL",
    "/home/yoon/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf")
MMPROJ = os.environ.get("MMPROJ", "/home/yoon/models/qwen3.8-27b/mmproj-F16.gguf")
IMG = os.environ.get("IMAGE",
                     "/home/yoon/LLM170/source/llama.cpp/tools/mtmd/test-1.jpeg")
PORT = int(os.environ.get("LLM170_VL_REF_PORT", "10090"))
N = int(os.environ.get("LLM170_VL_N", "24"))
STORE = os.environ.get("LLM170_VL_STORE", "/tmp/verify_vl_base.json")


def run_vl(images, extra=(), env_extra=None, n_predict=N):
    env = dict(os.environ)
    env.pop("LLM170_EXACT", None)
    env.pop("LLM170_SPEC_GPU", None)
    if env_extra:
        env.update(env_extra)
    args = [BIN, "vl", "--model", MODEL, "--mmproj", MMPROJ,
            "--n-predict", str(n_predict), *extra]
    for im in images:
        args += ["--image", im]
    r = subprocess.run(args, capture_output=True, text=True, timeout=7200, env=env)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-4000:])
        raise RuntimeError(f"vl rc={r.returncode}")
    toks, texts = {}, {}
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            j = json.loads(line)
            toks.setdefault(j["seq"], []).append(j["token"])
        elif line.startswith("seq") and ":" in line:
            s, _, t = line.partition(":")
            try:
                texts[int(s[3:])] = t.strip()
            except ValueError:
                pass
        elif line and not texts and "seq" not in line:
            texts[0] = line  # 단일 이미지 최종 텍스트
    return toks, texts


def compare_exact(name, a, b):
    ok = a == b
    k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {len(a)} vs {len(b)} tok"
          + ("" if ok else f" — 첫 불일치 @[{k}] {a[k]} vs {b[k]}"), flush=True)
    if not ok and k is not None:
        print(f"    a[..{k+4}]: {a[:k+4]}\n    b[..{k+4}]: {b[:k+4]}")
    return ok


def llama_semantic(question):
    import base64
    import urllib.request
    with open(IMG, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps({
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": question},
            ]}],
            "max_tokens": 512, "temperature": 0.0,
        }).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.loads(r.read())
    msg = out["choices"][0]["message"]
    # thinking 모델: content가 비어있고 reasoning_content에 본문이 있을 수 있음
    return (msg.get("content") or "") + "\n" + (msg.get("reasoning_content") or "")


def llama_tokenize(text):
    import urllib.request
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/tokenize",
        data=json.dumps({"content": text}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["tokens"]


def collect():
    para = ("The quick brown fox jumps over the lazy dog. Pack my box with "
            "five dozen liquor jugs. How vexingly quick daft zebras jump! "
            "Sphinx of black quartz, judge my vow. ")
    prefix = None
    for k in range(2, 120):
        ids = llama_tokenize(para * k)
        if len(ids) >= 2300:
            prefix = ids
            break
    assert prefix, "장문 접두 조립 실패"
    ref_txt = llama_semantic("Describe this image in one short sentence.")
    with open(STORE, "w") as f:
        json.dump({"prefix": prefix, "llama_text": ref_txt}, f)
    print(f"[collect] prefix {len(prefix)} tok; llama: {ref_txt[:200]}")


def judge():
    with open(STORE) as f:
        st = json.load(f)
    prefix = st["prefix"]
    pflag = ("--prefix-tokens", ",".join(map(str, prefix)))
    results = []

    # 1) 스펙 불변식 (단일)
    ref_t, _ = run_vl([IMG])
    sp_t, _ = run_vl([IMG], extra=["--spec", "4"],
                     env_extra={"LLM170_SPEC_GPU": "1"})
    results.append(compare_exact("vl_spec_short", sp_t[0], ref_t[0]))

    # 2) np2 × 스펙 불변식 + np 상태 격리
    ref2_t, _ = run_vl([IMG, IMG])
    sp2_t, _ = run_vl([IMG, IMG], extra=["--spec", "4"],
                      env_extra={"LLM170_SPEC_GPU": "1"})
    for s in range(2):
        results.append(compare_exact(f"vl_spec_np2_seq{s}", sp2_t[s], ref2_t[s]))
    results.append(compare_exact("vl_np2_isolation", ref2_t[0], ref_t[0]))

    # 3) EXACT 비트계약: GPU(EXACT=1) == CPU
    cpu_t, _ = run_vl([IMG], extra=["--backend", "cpu"])
    ex_t, _ = run_vl([IMG], env_extra={"LLM170_EXACT": "1"})
    results.append(compare_exact("vl_exact_cpu_short", ex_t[0], cpu_t[0]))

    # 4) 장문 접두: EXACT 비트계약 + 스펙 불변식
    cpuL_t, _ = run_vl([IMG], extra=["--backend", "cpu", *pflag])
    exL_t, _ = run_vl([IMG], extra=pflag, env_extra={"LLM170_EXACT": "1"})
    results.append(compare_exact("vl_exact_cpu_long", exL_t[0], cpuL_t[0]))
    refL_t, _ = run_vl([IMG], extra=pflag)
    spL_t, _ = run_vl([IMG], extra=[*pflag, "--spec", "4"],
                      env_extra={"LLM170_SPEC_GPU": "1"})
    results.append(compare_exact("vl_spec_long", spL_t[0], refL_t[0]))

    # 5) llama 의미 판정 (수집 텍스트 기준)
    _, ours_txts = run_vl([IMG])
    ours_txt = ours_txts.get(0, "")
    low_r, low_o = st["llama_text"].lower(), ours_txt.lower()
    sem = (("new york times" in low_r) or ("nyt" in low_r)) and \
          (("new york times" in low_o) or ("nyt" in low_o)) and \
          (("moon" in low_r) == ("moon" in low_o))
    print(f"[{'PASS' if sem else 'WARN'}] vl_semantic_llama (NYT/moon 키워드)")
    print(f"    llama: {st['llama_text'][:200]}")
    print(f"    ours : {ours_txt[:200]}")

    print(f"\n=== 결과: {sum(results)}/{len(results)} PASS ===")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    if os.environ.get("LLM170_VL_PHASE", "judge") == "collect":
        collect()
    else:
        judge()
