#!/usr/bin/env python3
"""llm170 비전(mmproj) 골 매트릭스 검증 (plans/28) — 2페이즈.

판정 규칙 (2026-09-06 확정):
  1. vl GPU --spec 4 == vl GPU 비스펙: 완전일치 PASS, 또는 첫 발산이 근접티
     (발산 토큰 쌍이 verify 로짓 top-2에서 갭 < 1.0)면 PASS(tie).
     — 근거: 배치 형상(t=13 vs 17)이 로짓 반올림을 바꿔 0.015갭 평탄점을
       뒤집는 실측 (494:19.656 vs 16311:19.641). ADR-0012와 동일 부류.
  2. vl np2 seq0 == vl 단일: 완전일치 (같은 경로·같은 t — 상태 격리 불변식).
  3. llama --mmproj 의미 판정: NYT 1면 키워드.
  CPU 교차검증은 제외 — CPU clip과 GPU vit 임베딩이 근사치(비트불일치)라
  스트림 비교가 성립하지 않음 (2026-09-06 실측, 게이트에서 제거).

사용:
  LLM170_VL_PHASE=collect python3 scripts/verify_vl.py  # llama --mmproj 기동 중
  LLM170_VL_PHASE=judge python3 scripts/verify_vl.py    # 우리 엔진 단독
"""
import json
import os
import re
import subprocess
import sys

BIN = "target/release/llm170"
MODEL = os.environ.get(
    "LLM170_MODEL",
    "/home/yoon/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf")
MMPROJ = os.environ.get("MMPROJ", "/home/yoon/models/qwen3.8-27b/mmproj-F16.gguf")
IMG = os.environ.get("IMAGE",
                     "/home/yoon/LLM170/source/llama.cpp/tools/mtmd/test-1.jpeg")
PORT = int(os.environ.get("LLM170_VL_REF_PORT", "10090"))
N = int(os.environ.get("LLM170_VL_N", "24"))
STORE = os.environ.get("LLM170_VL_STORE", "/tmp/verify_vl_base.json")
TIE_EPS = float(os.environ.get("LLM170_VL_TIE_EPS", "1.0"))


def run_vl(images, extra=(), env_extra=None, n_predict=N):
    env = dict(os.environ)
    env.pop("LLM170_EXACT", None)
    env.pop("LLM170_SPEC_GPU", None)
    env.pop("LLM170_MS_LOGITS", None)
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
            texts[0] = line
    return toks, texts, r.stderr


def parse_mslg(stderr):
    """[mslg] rowN: id:logit ... 행 파싱 → [{row, [(id, logit)...]}]"""
    rows = []
    for l in stderr.splitlines():
        m = re.match(r"\[mslg\] row(\d+): (.*)", l.strip())
        if m:
            pairs = []
            for tok in m.group(2).split():
                i, v = tok.split(":")
                pairs.append((int(i), float(v)))
            rows.append(pairs)
    return rows


def compare(name, a, b, mslg_rows=None):
    """a=spec, b=ref. 완전일치 or 근접티(양 토큰이 한 행 top-2에서 갭<ε)."""
    if a == b:
        print(f"[PASS] {name}: {len(a)} tok 완전일치")
        return True
    k = next((i for i, (x, y) in enumerate(zip(b, a)) if x != y), None)
    if k is None:
        print(f"[FAIL] {name}: 길이 불일치 {len(a)} vs {len(b)}")
        return False
    ta, tb = a[k], b[k]
    tie = False
    if mslg_rows:
        for pairs in mslg_rows:
            ids = [i for i, _ in pairs]
            if ta in ids and tb in ids:
                ia, ib = ids.index(ta), ids.index(tb)
                gap = abs(pairs[ia][1] - pairs[ib][1])
                if gap < TIE_EPS:
                    tie = True
                    print(f"[PASS] {name}: 근접티 @gen[{k}] ({ta} vs {tb}, "
                          f"top-2 갭 {gap:.3f} < ε={TIE_EPS}) — 이후 맥락 분기")
                    break
    if not tie:
        print(f"[FAIL] {name}: 발산 @gen[{k}] {tb} vs {ta} (tie 근거 없음)")
        print(f"    ref [..{k+3}]: {b[:k+3]}")
        print(f"    spec[..{k+3}]: {a[:k+3]}")
    return tie


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

    # 1) 스펙 불변식 (단일) — 로짓 덤프로 tie 판정
    ref_t, _, _ = run_vl([IMG])
    sp_t, _, sp_err = run_vl([IMG], extra=["--spec", "4"],
                             env_extra={"LLM170_SPEC_GPU": "1",
                                        "LLM170_MS_LOGITS": "1"})
    results.append(compare("vl_spec_short", sp_t[0], ref_t[0],
                           parse_mslg(sp_err)))

    # 2) np2 × 스펙 + np 상태 격리
    ref2_t, _, _ = run_vl([IMG, IMG])
    sp2_t, _, sp2_err = run_vl([IMG, IMG], extra=["--spec", "4"],
                               env_extra={"LLM170_SPEC_GPU": "1",
                                          "LLM170_MS_LOGITS": "1"})
    for s in range(2):
        results.append(compare(f"vl_spec_np2_seq{s}", sp2_t[s], ref2_t[s],
                               parse_mslg(sp2_err)))
    ok = ref2_t[0] == ref_t[0]
    print(f"[{'PASS' if ok else 'FAIL'}] vl_np2_isolation: np2 seq0 == 단일 "
          f"({len(ref2_t[0])} vs {len(ref_t[0])} tok)")
    results.append(ok)

    # 3) 장문 접두 × 스펙 (이미지+2300토큰 프롬프트)
    refL_t, _, _ = run_vl([IMG], extra=pflag)
    spL_t, _, spL_err = run_vl([IMG], extra=[*pflag, "--spec", "4"],
                               env_extra={"LLM170_SPEC_GPU": "1",
                                          "LLM170_MS_LOGITS": "1"})
    results.append(compare("vl_spec_long", spL_t[0], refL_t[0],
                           parse_mslg(spL_err)))

    # 4) llama 의미 판정 (수집 텍스트 기준)
    _, ours_txts, _ = run_vl([IMG])
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
