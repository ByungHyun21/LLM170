#!/usr/bin/env python3
"""plans/115 P1-3 — 부분 접두 체크포인트 재사용 토큰 등가성 검증.

불변식: 체크포인트 되감기→재프리필(slots[x].cached가 새 프롬프트의 진접두,
l < cached.len) 경로의 출력 토큰 == 무재사용(신선 슬롯) 토큰 (완전일치).

케이스:
  1) partial_single — A(1200) 완료 후 B=A[:600]+다른 꼬리 (단독 프리필, ckpt 512)
  2) partial_xslot  — A·A' 0.5s 지연 도착(양 슬롯 캐시, 인터리브 단독 청크) 후
     B(slot0)·C(slot1) 순차 복원 — 슬롯 교차 복원 2회, 비트 일치
  3) partial_spec   — --spec 2에서 A→B 순차 (드래프트 초기화 경로 포함)
  4) partial_batchinfo — A×2 동시(진배치 캡처) → B: 재사용 발동·결정성만 단언.
     배치 캡처 체크포인트의 토큰이 단독 경로 참조와 비트 다를 수 있음은
     문서화된 산술 클래스다(prefill_multi.rs — 행 레이아웃이 per-seq 체인
     산술을 바꾼다). 드리프트는 정보로만 보고.

참조: 신선 서버에서 B·C 순차 (각각 빈 슬롯 — 재사용·배치 개입 없음).

사용: python3 scripts/verify_prefix_ckpt.py
환경: LLM170_MFLASH, PORT(기본 18097)
"""
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request

BIN = "target/release/llm170"
MODEL = os.environ.get(
    "LLM170_MFLASH",
    "/home/yoon/models/qwen3.8-Flash-Next/"
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")
PORT = int(os.environ.get("PORT", "18097"))
NP = 24


def lcg_ids(n, seed):
    """결정적 의사 토큰열 — 특수 id(eos 248046 등) 회피."""
    out, x = [], seed
    for _ in range(n):
        x = (x * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        out.append(1000 + x % 200000)
    return out


BASE = lcg_ids(1200, 7)
A = BASE                      # 1200토큰 — 캐시 시딩용
B = BASE[:600] + lcg_ids(300, 11)   # l=600 → ckpt 512
C = BASE[:900] + lcg_ids(300, 23)   # l=900 → ckpt 512
LOG = "/tmp/verify_p13_serve.log"


def start(spec):
    env = dict(os.environ)
    env["LLM170_SLOTS"] = "2"
    if os.path.exists("/opt/rocm-10.0.0/install/lib"):
        env["LD_LIBRARY_PATH"] = ("/opt/rocm-10.0.0/install/lib:"
                                  + env.get("LD_LIBRARY_PATH", ""))
    args = [BIN, "serve", "--model", MODEL, "--port", str(PORT),
            "--ctx", "4096", "--backend", "hip", "--slots", "2"]
    if spec:
        args += ["--spec", str(spec)]
    lf = open(LOG, "wb")
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=lf, env=env)
    base = f"http://127.0.0.1:{PORT}"
    deadline = time.time() + 600
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
                if r.status == 200:
                    return proc, base
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError(f"serve 조기 종료 — {LOG} 참조")
            time.sleep(2)
    raise RuntimeError("serve 헬스 타임아웃")


def fire(base, ids, npred, out, idx):
    req = urllib.request.Request(
        f"{base}/completion",
        data=json.dumps({"prompt": ids, "n_predict": npred}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=3600) as r:
        out[idx] = json.loads(r.read())["tokens"]


def fire_all(base, jobs):
    """jobs: [(ids, npred)] — 동시 발사, 완료 대기."""
    outs = [None] * len(jobs)
    ths = [threading.Thread(target=fire, args=(base, ids, npred, outs, i))
           for i, (ids, npred) in enumerate(jobs)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    assert all(o is not None for o in outs), "응답 누락"
    return outs


def stop(proc):
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=60)
    except Exception:
        proc.kill()
    time.sleep(2)


def loggrep(pat):
    try:
        with open(LOG, "rb") as f:
            return sum(1 for l in f if pat.encode() in l)
    except FileNotFoundError:
        return 0


def main():
    results = []
    d = os.environ.get("LLM170_DUMP", "").strip(",")
    os.environ["LLM170_DUMP"] = (d + "," if d else "") + "wall_time"

    # ── 참조: 신선 서버에서 B·C 순차(각각 빈 캐시 슬롯 — 재사용 없음)
    proc, base = start(0)
    try:
        refB = fire_all(base, [(B, NP)])[0]
        refC = fire_all(base, [(C, NP)])[0]
    finally:
        stop(proc)
    print(f"ref B={refB[:8]}…")

    # ── 1) 단독: A 시딩 → B (부분 재사용, ckpt 512)
    proc, base = start(0)
    try:
        fire_all(base, [(A, 4)])
        gotB = fire_all(base, [(B, NP)])[0]
    finally:
        n = loggrep("partial reuse")
        stop(proc)
    ok = gotB == refB and n >= 1
    results.append(("partial_single", ok, gotB, refB, n))

    # ── 2) 슬롯 교차: 슬롯0 A(400생성)의 **프리필 완료 후**(디코드 중) A' 도착
    #    → 슬롯1 (단독 청크 캡처 — 프리필 중 도착하면 prefill_multi 배치 산술로
    #    캡처돼 문서화 드리프트 클래스가 됨. LLM170_DUMP=wall_time 마커로
    #    프리필 완료를 폴링해 배치를 배제한다) → 완료 후 B(slot0)·C(slot1) 복원.
    proc, base = start(0)
    bg = [None]
    def bg_fire():
        fire(base, A, 400, bg, 0)
    th = threading.Thread(target=bg_fire)
    th.start()
    # 프리필 완료 대기 — "1200tok done"(마지막 청크)만 본다. 슬롯 인덱스나
    # 중간 청크 마커를 보면 A 자신/A' 조기 도착을 못 걸러 배치 캡처가 섞인다.
    dl = time.time() + 120
    while time.time() < dl:
        if loggrep("1200tok done") >= 1:
            break
        time.sleep(0.2)
    try:
        fire_all(base, [(A, 4)])      # 슬롯1 배정 (슬롯0 디코드 중)
        th.join(timeout=600)
        gotB2 = fire_all(base, [(B, NP)])[0]
        gotC2 = fire_all(base, [(C, NP)])[0]
    finally:
        n = loggrep("partial reuse")
        stop(proc)
        th.join(timeout=30)
    ok = gotB2 == refB and gotC2 == refC and n >= 2
    results.append(("partial_xslot", ok, (gotB2, gotC2), (refB, refC), n))

    # ── 4) 진배치 캡처(정보): A×2 동시 → B — 재사용 발동·결정성만 단언
    outs4 = []
    for _ in range(2):
        proc, base = start(0)
        try:
            fire_all(base, [(A, 4), (A, 4)])
            outs4.append(fire_all(base, [(B, NP)])[0])
        finally:
            n4 = loggrep("partial reuse")
            stop(proc)
    drift = next((i for i, (x, y) in enumerate(zip(*outs4)) if x != y), None)
    ok = n4 >= 1 and outs4[0] == outs4[1]
    results.append(("partial_batchinfo", ok, [outs4[0]], [outs4[0]], n4))
    d = 0 if outs4[0] == refB else next(
        (i for i, (x, y) in enumerate(zip(outs4[0], refB)) if x != y), -1)
    print(f"    [info] 진배치 캡처 B: 재결정성={'OK' if drift is None else drift}"
          f" 단독참조 대비 첫 분기=@{d} (문서화 산술 클래스)")

    # ── 3) 스펙: --spec 2, A → B 순차 (드래프트 초기화 포함)
    proc, base = start(2)
    try:
        fire_all(base, [(A, 4)])
        gotB3 = fire_all(base, [(B, NP)])[0]
    finally:
        n = loggrep("partial reuse")
        stop(proc)
    ok = gotB3 == refB and n >= 1
    results.append(("partial_spec", ok, gotB3, refB, n))

    print()
    npass = 0
    for tag, ok, got, ref, n in results:
        g = got if isinstance(got[0], list) else [got]
        r = ref if isinstance(ref[0], list) else [ref]
        bad = ""
        for i, (a, b) in enumerate(zip(g, r)):
            k = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), None)
            if k is not None or len(a) != len(b):
                bad += f" seq{i}@{k}(len {len(a)}vs{len(b)})"
        print(f"[{'PASS' if ok else 'FAIL'}] {tag} reuse로그={n}"
              + (f" — 불일치{bad}" if bad else ""))
        npass += ok
    print(f"\n=== 결과: {npass}/{len(results)} PASS ===")
    raise SystemExit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
