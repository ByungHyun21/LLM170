#!/usr/bin/env python3
"""llm170 CPU 엔진 정확도 검증 하니스 (기준: llama-server greedy).

케이스 (검증 매트릭스: 짧은/장문 × 단일/np — 4종 + 장기 생성):
  1. single_short / single_ko / single_code — 단일 스트림 짧은 프롬프트
  2. np4 — 4개 프롬프트 병렬 배치 디코드 (상태 격리 검증)
  3. long_prompt — 장문(~2400토큰) 단일 prefill (GDN chunked 경로 장문)
  4. long_np2 — 장문 프롬프트 2종 병렬 (장문 prefill + np 조합)
  5. long_gen96 — 장기 생성 (누적 상태 오차)

토큰 id는 기준 서버 /tokenize로 통일 — 토크나이저 표면은 비교 범위 밖.
greedy(temp 0) 판정: 완전일치, 또는 첫 발산 지점이 근접티(우리 토큰이 기준
top-6 안 & top-1 로그확률 갭 < TIE_EPS=1.5nat)면 PASS(tie) — f32 내적(우리)과
q8 활성 내적(llama)의 소음 차이가 로짓 갭 ~0.3 이하에서만 스트림을 갈라놓기 때문.
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
# 근접티 판정 임계(nat). 실측 근거: 발산 지점의 양 엔진 top-k가 동일 집합을 공유하는
# 평탄 상위 영역에서 순서만 갈림 — p1 갭 0.36(우리측 0.27), p2 갭 1.19(우리측 0.25).
# 소음 기제: llama는 활성 q8 내적 + KV f16 캐시, 우리는 활성 f32 + KV f32 — 문맥이
# 쌓일수록 편차 증가. 진짜 버그는 top-k 밖 가비지 순위로 나타나므로 ε=1.5로 분리됨.
TIE_EPS = float(os.environ.get("LLM170_TIE_EPS", "1.5"))


def post(path, payload):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=2400) as r:
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
        "logprobs": 6,
    })
    toks = out.get("tokens")
    if toks is None:
        # 폴백: 생성 텍스트를 다시 토큰화 (근사 — return_tokens 미지원 서버용)
        toks = tokenize(out["content"])
    return toks, out.get("completion_probabilities") or []


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


def ours_serve(prompts, n_predict, ctx):
    """우리 서버(llm170 serve) 기동 → 2요청 동시 발사 → 토큰 리스트 반환.
    서버는 mpsc 직렬 큐(현 구조) — 동시 발사 시 상태격리가 게이트.
    LLM170_EXTRA_ARGS는 CLI 경로와 동일하게 serve에 전달(백엔드 패리티)."""
    import signal
    import threading
    import time
    extra = os.environ.get("LLM170_EXTRA_ARGS", "").split()
    proc = subprocess.Popen(
        [BIN, "serve", "--model", MODEL_PATH, "--port", str(port),
         "--ctx", str(ctx), *extra],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 2400
    try:
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"{base_url}/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:
                if proc.poll() is not None:
                    raise RuntimeError("llm170 serve 조기 종료")
                time.sleep(2)
        else:
            raise RuntimeError("llm170 serve 헬스 대기 타임아웃")
        outs = [None] * len(prompts)

        def worker(i, ids):
            req = urllib.request.Request(
                f"{base_url}/completion",
                data=json.dumps({"prompt": ids, "n_predict": n_predict}).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=2400) as r:
                outs[i] = json.loads(r.read())["tokens"]

        ths = [threading.Thread(target=worker, args=(i, ids))
               for i, ids in enumerate(prompts)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        return [o or [] for o in outs]
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


def compare_exact(name, ours, ref):
    """서버 케이스 판정 — 완전일치만 (같은 엔진·같은 수치 계열; tie 허용 없음)."""
    ok = ours == ref
    k = next((i for i, (a, b) in enumerate(zip(ref, ours)) if a != b), None)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: server {len(ours)} tok, cli-np2 "
          f"{len(ref)} tok" + ("" if ok else f" — 첫 불일치 @gen[{k}]"))
    return ok


def compare(name, base, ours, probs=None):
    """판정: 완전일치 PASS, 또는 첫 발산 지점이 근접티(우리 토큰이 기준 top-k 안 &
    top-1과의 로그확률 갭 < TIE_EPS)면 PASS(tie). 이후 토큰은 맥락이 갈라져 판정 불가."""
    ok = base == ours
    n = min(len(base), len(ours))
    diff = sum(1 for a, b in zip(base, ours) if a != b)
    if ok:
        print(f"[PASS] {name}: base {len(base)} tok, ours {len(ours)} tok — 완전일치")
        return True
    # 첫 발산 위치
    k = next((i for i, (a, b) in enumerate(zip(base, ours)) if a != b), None)
    tie = False
    detail = ""
    if k is not None and probs and k < len(probs):
        top = probs[k].get("top_logprobs", [])
        ids_top = [e["id"] for e in top]
        if ours[k] in ids_top:
            r = ids_top.index(ours[k])
            gap = top[0]["logprob"] - top[r]["logprob"]
            if r == 0:
                detail = f"@gen[{k}] 기준 top-1과 동일 토큰인데 스트림 상 불일치 (기준 로그확률 오류 의심)"
            elif gap < TIE_EPS:
                tie = True
                detail = (f"근접티 @gen[{k}]: ours=기준 top-{r+1} (갭 {gap:.2f} < ε={TIE_EPS}) "
                          f"[top1={top[0]['id']} {top[0]['logprob']:.3f} | ours={ours[k]} {top[r]['logprob']:.3f}]")
            else:
                detail = f"@gen[{k}]: ours=기준 top-{r+1} 갭 {gap:.2f} ≥ ε={TIE_EPS} — 진짜 발산"
        else:
            detail = f"@gen[{k}]: ours={ours[k]} 기준 top-{len(ids_top)} 밖 — 진짜 발산"
    elif k is not None:
        detail = f"@gen[{k}]: base={base[k]} ours={ours[k]} (logprobs 없음)"
    status = "PASS" if tie else "FAIL"
    print(f"[{status}] {name}: base {len(base)} tok, ours {len(ours)} tok, 불일치 {diff}/{n}"
          + (f" — {detail}" if detail else ""))
    if not tie:
        if k is not None:
            print(f"    base[..{k+1}]: {base[:k+1]}")
            print(f"    ours[..{k+1}]: {ours[:k+1]}")
        else:
            tail = base[n:n+8] if len(base) > n else ours[n:n+8]
            print(f"    길이 차이: base {len(base)} vs ours {len(ours)} (공통부 일치) 이후: {tail}")
    return tie


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
        base, bprobs = baseline_generate(ids, n)
        ours = ours_generate([ids], n, 2048)[0]
        results.append(compare(name, base, ours, bprobs))

    # --- np4 병렬 (배치 디코드 상태 격리) ---
    prompts = [tokenize(p) for p in (p1, p2, p3, p4)]
    pairs = [baseline_generate(p, N_PREDICT_DEFAULT) for p in prompts]
    ours = ours_generate(prompts, N_PREDICT_DEFAULT, 2048)
    for i in range(4):
        results.append(compare(f"np4_seq{i}", pairs[i][0], ours[i], pairs[i][1]))
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
    base, bprobs = baseline_generate(long_ids, N_PREDICT_DEFAULT)
    ours = ours_generate([long_ids], N_PREDICT_DEFAULT, 4096)[0]
    results.append(compare("long_prompt", base, ours, bprobs))

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
    b1, p1_ = baseline_generate(long_ids, N_PREDICT_DEFAULT)
    b2, p2_ = baseline_generate(long_ids2, N_PREDICT_DEFAULT)
    ours = ours_generate([long_ids, long_ids2], N_PREDICT_DEFAULT, 4096)
    results.append(compare("long_np2_seq0", b1, ours[0], p1_))
    results.append(compare("long_np2_seq1", b2, ours[1], p2_))

    # --- 장기 생성 ---
    ids = tokenize(p1)
    n = 96
    base, bprobs = baseline_generate(ids, n)
    ours = ours_generate([ids], n, 2048)[0]
    results.append(compare("long_gen96", base, ours, bprobs))

    # --- 서버 배칭 케이스 (04 슬롯 엔진 게이트 — 등록 먼저, 실행은 04 완료 후) ---
    # 판정: 서버 2동시 요청 ↔ CLI np2 단독 = 완전일치만 (불변식: 배치 구성이
    # 스트림을 바꾸지 않는다). 활성: LLM170_SERVE_CHECK=1.
    if os.environ.get("LLM170_SERVE_CHECK") == "1":
        ref = ours_generate([tokenize(p1), tokenize(p2)], N_PREDICT_DEFAULT, 2048)
        srv = ours_serve([tokenize(p1), tokenize(p2)], N_PREDICT_DEFAULT, 2048)
        results.append(compare_exact("server_np_seq0", srv[0], ref[0]))
        results.append(compare_exact("server_np_seq1", srv[1], ref[1]))
        ref = ours_generate([long_ids, long_ids2], N_PREDICT_DEFAULT, 4096)
        srv = ours_serve([long_ids, long_ids2], N_PREDICT_DEFAULT, 4096)
        results.append(compare_exact("server_long_np_seq0", srv[0], ref[0]))
        results.append(compare_exact("server_long_np_seq1", srv[1], ref[1]))
    else:
        print("[skip] server_np·server_long_np — LLM170_SERVE_CHECK=1로 활성 "
              "(슬롯 엔진 04 완료 후 실행 게이트)")

    print(f"\n=== 결과: {sum(results)}/{len(results)} PASS ===")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
