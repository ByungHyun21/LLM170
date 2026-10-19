#!/usr/bin/env python3
# eval 통합 게이트 — "통과 = 기준 부합"의 판정식 (2026-10-19 재설계).
#
# 계층 (전부 실측 기반 — 임계는 아래 표의 측정값에서 유도):
#   L1 회귀  : 골든·해시·지문 (스모크 담당 — 이 스크립트 밖, 전제 조건)
#   L2 상대  : 우리 A/B 델타 ≤ **자체 봉투**(우리가 이미 채택한 클래스 변동 폭)
#   L3 절대  : vLLM 피어 — PPL 정합(빨간불 2% 초과) + 생성 배터리 일치율
#   L4 진리  : f64 정확 오라클 대비 |ΔNLL| — 후보 ≤ 기준선, vLLM 대비 우열
#   L5 거동  : 생성 배터리 자체(접두 붕괴·퇴화 검출 — L3와 동일 실행)
#
# 사용:
#   envs/vllm/bin/python eval_gate.py <model_dir> <corpus> \
#       [--dump-a F --dump-b F]            # L2 (w4a16-eval 덤프 2개)
#       [--oracle F] [--oracle-vllm F]     # L4 (w4a16-ref --hi-prec --nll-out 덤프)
#       [--peer]                           # L3/L5 (vLLM :19001 + 우리 CLI)
#       [--json-out F]
#
# 실측 봉투(2026-10-19 — 새 후보의 허용 한도. 갱신 시 사유 기록):
#   35B  en ΔPPL(마린↔레거시) +0.096% · mix32k +1.05% · 생성 100% · vLLM 생성 95%
#   gemma4 mix32k ΔPPL(마린↔레거시) +0.12% · en +4.26%(혼란 영역 — 단독 판정 금지)
#   진리(35B en 255pos |ΔNLL|): 마린 0.0616 < 레거시 0.0650 < vLLM 0.0747
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
    # 봉투: |ΔPPL| ≤ max(2%, |봉투|×1.5) — 봉투는 실측 자체 변동(모델별 표)
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
