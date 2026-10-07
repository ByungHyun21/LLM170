#!/usr/bin/env python3
"""CUDA 다중 슬롯 격리 검증 — plans/cuda-port.md §5 우선순위 3번(§6-7).

왜 필요한가
------------
S8가 4슬롯 격리를 입증했지만 그 대조는 **T=1 디코드 경로(S10)** 기준이었다.
S11로 프리필이 `forward_batch_device`(배치 경로)로 전환된 뒤 그 경로의 슬롯
격리는 다시 확인되지 않았다(원장 §6-7). 코드는 슬롯 인자를 받기만 하고 실측
증거가 없다. 다중 슬롯을 실사용하면 지금 검증되지 않은 경로가 돌아간다.

판정
----
같은 프롬프트 4종에 대해
  (A) 슬롯 4 서버에 동시 4요청  vs  (B) 슬롯 1 서버에 직렬 4요청
의 **토큰열이 완전히 같아야 한다**. 스케줄러가 슬롯을 섞어도 결과는 같아야
한다. 한 토큰이라도 어긋나면 격리 실패다(127-E: 백엔드 간 동일성은 성립
대상이 아니지만, **동일 백엔드 슬롯 간** 동일성은 성립해야 한다).

왜 프롬프트 길이를 다르게 하는가
--------------------------------
전부 같은 길이면 슬롯 간 상태가 섞여도 우연히 맞을 수 있다. 41/82/123/205
토큰으로 달리면 위치가 달라 공유 상태가 드러난다. 특히 마지막 프롬프트는
배치 프리필(8토큰 단위)을 여러 번 돌리고 마지막 청크가 5토큰이라 T 경계
스윙(siblings가 아님)이 함께 검증된다.

왜 서버를 순차로 띄우는가
--------------------------
27B EXL3 가중치만 ~19.5GB이고 이 기기는 24GB 카드다. 슬롯 4/1 서버를
동시에 띄우면 39GB가 필요해 OOM한다. 판정 대상이 "동시 vs 직렬"이므로
(A)→(B) 순차 실행으로 충분하다. 4슬롯 서버만 먼저 띄워 4요청을 **실제로
동시에** 보내는 점은 보존된다.

사용법(워크트리 루트)
---------------------
    LLM170_MEXL3=../models/Qwen3.8-27B-exl3-5.00bpw \\
        scripts/verify_cuda_slots.py
    LLM170_MEXL3=... LLM170_SLOT_CTX=1024 scripts/verify_cuda_slots.py
"""

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

MODEL = os.environ.get("LLM170_MEXL3", "../models/Qwen3.8-27B-exl3-5.00bpw")
CTX = int(os.environ.get("LLM170_SLOT_CTX", "512"))
NPRED = int(os.environ.get("LLM170_SLOT_NPRED", "24"))
BIN = "./target/release/llm170"
# 동시 요청이 실제로 겹쳐 처리됐는지의 하한(Σlatency / wall).
# 연속배치는 동시 요청을 한 디코더 스텝으로 합치므로 4요청이면 수배까지
# 올라갈 수 있다. 1.5 미만이면 사실상 직렬이라 격리를 증명한 것이 아니다.
MIN_OVERLAP = float(os.environ.get("LLM170_SLOT_MIN_OVERLAP", "1.5"))

# 게이트와 동일한 한국어 문장 토큰열(41토큰). 길이만 반복해 4종을 만든다.
# **토큰 ID 배열로 넣는다** — `/completion`의 prompt가 문자열이면 텍스트로
# 재토크나이즈되어(oai.rs: jstr → greedy_encode) 토큰 수가 예측 불가능해지고,
# 게이트와 다른 입력이 되어 슬롯 대조의 의미가 흐려진다. 같은 토큰열을
# 그대로 넣으려면 JSON 배열 경로(jarr_u32)를 써야 한다.
BASE = [
    148678, 65233, 202419, 220, 49849, 155497, 220, 151314, 39504, 149635, 13,
    220, 174675, 30061, 220, 152055, 152065, 12434, 220, 154854, 149248, 80102,
    20673, 220, 214009, 149789, 11, 220, 60177, 148726, 22836, 220, 149965,
    176289, 220, 12434, 160288, 220, 158201, 149635, 13,
]

PROMPTS = [BASE * k for k in (1, 2, 3, 5)]
# 동시 요청 수(진단용). 기본 4. 2나 3으로 낮춰 "몇 행부터 깨지는가"를
# 좁히는 용도 — 슬롯 간 배치는 T행이 커질수록 공유 자원이 늘어난다.
CONC = int(os.environ.get("LLM170_SLOT_CONC", "4"))
if CONC < 1 or CONC > len(PROMPTS):
    die(f"LLM170_SLOT_CONC={CONC} — 1..{len(PROMPTS)} 범위여야 함")


def die(msg: str) -> None:
    print(f"[slot-verify] FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def free_gpu() -> None:
    """GPU 독점 계약 — 다른 추론 프로세스 상주 시 VRAM이 모자라면 판정이
    이상한 실패를 낸다(2026-09-25 inode 손상 사건과 같은 계급)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except Exception:
        return
    pids = [p for p in out.splitlines() if p.strip().isdigit()]
    if pids:
        die(f"GPU에 추론 프로세스 상주: {pids} — 정리 후 재실행")


def post(port: int, prompt: str, n_predict: int, timeout: int = 1800):
    body = json.dumps(
        {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0}
    ).encode("utf-8")  # prompt는 토큰 ID 배열(int) — 문자열이 아니다
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def start_server(port: int, slots: int):
    log = f"/tmp/opencode/slotverify_slots{slots}.log"
    os.makedirs("/tmp/opencode", exist_ok=True)
    fh = open(log, "wb")
    proc = subprocess.Popen(
        [BIN, "serve", "--backend", "cuda", "--model", MODEL,
         "--port", str(port), "--ctx", str(CTX), "--slots", str(slots)],
        stdout=fh, stderr=subprocess.STDOUT,
    )
    # 모델 로드는 19.5GB mmap — 여유 있게 기다린다.
    for _ in range(180):
        if proc.poll() is not None:
            die(f"slots={slots} 서버 조기 종료 rc={proc.returncode}\n"
                f"  로그: {open(log, errors='replace').read()[-1500:]}")
        time.sleep(3)
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=5
            ):
                return proc, log
        except Exception:
            continue
    proc.kill()
    die(f"slots={slots} 서버 기동 대기 초과 (로그 {log})")


def run_phase(slots: int, port: int, parallel: bool):
    proc, log = start_server(port, slots)
    try:
        n = CONC
        outs = [None] * n
        errs = [None] * n
        lat = [0.0] * n

        def one(i):
            t = time.time()
            try:
                outs[i] = post(port, PROMPTS[i], NPRED)
            except Exception as e:  # noqa: BLE001 — 실패 사유를 그대로 남긴다
                errs[i] = f"{type(e).__name__}: {e}"
            lat[i] = time.time() - t

        if parallel:
            # 진짜 동시 — 스케줄러가 4슬롯을 섞어야 하는 상황.
            ths = [threading.Thread(target=one, args=(i,)) for i in range(n)]
            t0 = time.time()
            for t in ths:
                t.start()
            for t in ths:
                t.join()
            wall = time.time() - t0
            lsum = sum(lat)
            # **vacuous 검사** — 토큰열 일치가 아무리 잘 나와도, 4요청이
            # 실제로 겹쳐 처리되지 않았다면 전부 슬롯 0 직렬이라 정의상
            # 일치할 뿐이다(아무것도 검증하지 않은 통과). 연속배치 스케줄러
            # 는 동시 요청을 한 디코더 스텝으로 합치므로 wall은 Σ보다
            # 훨씬 작다. 겹침이 없으면 FAIL로 막는다.
            overlap = lsum / wall if wall > 0 else 0.0
            print(f"[slot-verify] slots={slots} 동시 {len(ths)}요청 "
                  f"wall={wall:.1f}s Σlatency={lsum:.1f}s 겹침배율={overlap:.2f}x "
                  f"(개별 {' '.join(f'{x:.1f}' for x in lat)})")
            if overlap < MIN_OVERLAP:
                die(f"요청이 겹쳐 처리되지 않았다(겹침배율 {overlap:.2f} < "
                    f"{MIN_OVERLAP}) — 토큰열 일치가 자명한 통과가 된다. "
                    f"이 판정은 버린다.")
        else:
            t0 = time.time()
            for i in range(n):
                one(i)
                print(f"[slot-verify] slots={slots} 직렬 {i + 1}/{n} "
                      f"({lat[i]:.1f}s)")
            print(f"[slot-verify] slots={slots} 직렬 wall={time.time() - t0:.1f}s")

        for i, e in enumerate(errs):
            if e:
                die(f"요청 {i} 실패({slots}슬롯): {e}\n  로그: "
                    f"{open(log, errors='replace').read()[-1500:]}")
        return outs
    finally:
        proc.kill()
        proc.wait(timeout=60)
        time.sleep(4)  # VRAM 해제 대기 — 다음 기동이 OOM하지 않도록


def main() -> None:
    if not os.path.isdir(MODEL):
        die(f"모델 디렉터리 없음: {MODEL} (LLM170_MEXL3로 지정)")
    if not os.path.exists(BIN):
        die(f"바이너리 없음: {BIN} — cargo build --release 먼저")
    free_gpu()
    print(f"[slot-verify] 모델={MODEL} ctx={CTX} n_predict={NPRED}")
    print(f"[slot-verify] 프롬프트 토큰 수: "
          f"{[len(p) for p in PROMPTS]}")

    a = run_phase(CONC, 8951, parallel=True)   # 슬롯 N + 동시
    b = run_phase(CONC, 8952, parallel=False)  # 슬롯 N + 직렬

    fails = []
    for i, (ra, rb) in enumerate(zip(a, b)):
        ta, tb = ra.get("tokens", []), rb.get("tokens", [])
        if not ta:
            die(f"요청 {i}: 토큰열이 비었다(생성 실패?)")
        if ta != tb:
            first = next(
                (j for j in range(min(len(ta), len(tb))) if ta[j] != tb[j]),
                min(len(ta), len(tb)),
            )
            fails.append(
                f"  프롬프트#{i}({len(PROMPTS[i])}토큰) "
                f"최초 발산 idx={first} 동시={ta[first:first + 6]} "
                f"직렬={tb[first:first + 6]}"
            )
        print(f"[slot-verify] 프롬프트#{i} {len(PROMPTS[i])}토큰: "
              f"{len(ta)}토큰 {'일치' if ta == tb else '불일치'} "
              f"| {ra.get('text', '')[:70]!r}")

    if fails:
        print("[slot-verify] FAIL — 슬롯 격리 붕괴:", file=sys.stderr)
        for f in fails:
            print(f, file=sys.stderr)
        sys.exit(1)
    print("[slot-verify] ALL PASS — 슬롯 4 동시 == 슬롯 1 직렬 (토큰열 완전 일치)")


if __name__ == "__main__":
    main()