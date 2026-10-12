#!/usr/bin/env python3
# GPTQModel 교차 벤치 드라이버(27B W4A16) — llm170 bench 관례 준수:
#   LCG seed 0x12345678 합성 토큰(id<200000) · 워밍업1 + 3회 · 중앙값 ·
#   greedy + min_new_tokens(EOS 무시 상당). HF generate 공개 API만 사용
#   (엔진 코드 무수정 — 측정법만 통일).
# 측정: pp512/1024/4096(max_new_tokens=1 프리필 벽) ·
#       tg128@4k(=(t128-t1)/127 토큰당) · np4(배치4·프리필 256·128스텝 합산).
# 사용법: gptqmodel-venv/bin/python gptqmodel_bench.py <model_dir> [--backend auto]
import argparse
import statistics
import time

import torch

LCG_A = 6364136223846793005
LCG_C = 1442695040888963407
U64 = 1 << 64


def lcg(n, seed=0x12345678):
    s = seed
    out = []
    for _ in range(n):
        s = (s * LCG_A + LCG_C) % U64
        out.append((s >> 33) % 200000)
    return out


def timed_generate(model, inp, **gk):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.generate(inputs=inp, **gk)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    from gptqmodel import GPTQModel

    t0 = time.time()
    backend = None if args.backend == "auto" else args.backend
    if backend is None:
        model = GPTQModel.load(args.model, trust_remote_code=True)
    else:
        from gptqmodel import BACKEND
        model = GPTQModel.load(args.model, backend=getattr(BACKEND, backend.upper()), trust_remote_code=True)
    print(f"# load {time.time() - t0:.1f}s backend={args.backend}", flush=True)

    dev = model.model.device

    def mk(pp, batch=1, seed=0x12345678):
        ids = torch.tensor(
            [lcg(pp, seed + b * 7919) for b in range(batch)],
            dtype=torch.long,
            device=dev,
        )
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    greedy = {"do_sample": False, "pad_token_id": 0}
    one = {"max_new_tokens": 1, "min_new_tokens": 1, **greedy}
    full = {"max_new_tokens": 128, "min_new_tokens": 128, **greedy}

    for pp in (512, 1024, 4096):
        model.generate(inputs=mk(pp), **one)  # 워밍업
        ts = [
            timed_generate(model, mk(pp, seed=0x56781234 + i), **one)
            for i in range(args.reps)
        ]
        m = statistics.median(ts)
        reps_s = "/".join(f"{t * 1000:.1f}" for t in ts)
        print(f"pp{pp} | reps {reps_s} ms | median {m * 1000:.1f} ms | {pp / m:.2f} t/s", flush=True)

    model.generate(inputs=mk(4096), **one)  # 워밍업
    t1s = [
        timed_generate(model, mk(4096, seed=0xAAAA0000 + i), **one)
        for i in range(args.reps)
    ]
    t128s = [
        timed_generate(model, mk(4096, seed=0xAAAA0000 + i), **full)
        for i in range(args.reps)
    ]
    per_tok = statistics.median([(b - a) / 127 for a, b in zip(t1s, t128s)])
    print(f"tg128@4k | per-token {per_tok * 1000:.2f} ms | {1 / per_tok:.2f} t/s", flush=True)

    model.generate(inputs=mk(256, batch=4), **one)  # 워밍업
    bt1 = [
        timed_generate(model, mk(256, batch=4, seed=0xBBBB0000 + i), **one)
        for i in range(args.reps)
    ]
    bt128 = [
        timed_generate(model, mk(256, batch=4, seed=0xBBBB0000 + i), **full)
        for i in range(args.reps)
    ]
    step = statistics.median([(b - a) / 127 for a, b in zip(bt1, bt128)])
    print(f"np4(tg128) | step {step * 1000:.2f} ms | 합산 {4 / step:.2f} t/s", flush=True)


if __name__ == "__main__":
    main()
