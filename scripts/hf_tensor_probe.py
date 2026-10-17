#!/usr/bin/env python3
"""HF 체크포인트 텐서 헤더 원격 검증 — 다운로드 전 지문 대조(2026-10-16 CMP 셋업).

safetensors 헤더만 Range 요청으로 읽어 텐서 dtype/shape를 확인한다(본체 미다운로드).
용도: W4A16 미러 재배포가 많아 로더 계약(weight_packed/weight_scale/weight_shape
트리플 + 단일 스킴 group)과 실제 텐서의 일치를 사전 판정 — born2bewild -fast의
임베드 g256 혼합 사고(스모크 전실패) 재발 방지.

사용: python3 scripts/hf_tensor_probe.py <repo_id> [tensor_substring ...]
출력: 총 용량 · group(quant config) · 지정 텐서 shape · 샤드 내 W4A16 트리플 요약.
"""
import json
import sys
import urllib.request

BASE = "https://huggingface.co"


def fetch(url, rng=None):
    req = urllib.request.Request(url)
    if rng:
        req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def st_header(repo, shard):
    """샤드 헤더 JSON(길이 접두 8B + 헤더 본체) — Range 2회."""
    url = f"{BASE}/{repo}/resolve/main/{shard}"
    n = int.from_bytes(fetch(url, (0, 7)), "little")
    hdr_len = min(n, 12_000_000)  # 상식 상한(비정상 샤드 방어)
    return json.loads(fetch(url, (8, 8 + hdr_len - 1)))


def main():
    repo = sys.argv[1]
    subs = sys.argv[2:]
    tree = json.loads(
        fetch(f"{BASE}/api/models/{repo}/tree/main?recursive=true")
    )
    files = [x for x in tree if x.get("type") == "file"]
    total = sum(x["size"] for x in files)
    print(f"repo={repo} files={len(files)} total={total/2**30:.1f}GiB")
    extra = [x["path"] for x in files if "quant" in x["path"]]
    for q in ("quantization_config.json", "quant_config.json"):
        if q in (x["path"] for x in files):
            try:
                qc = json.loads(fetch(f"{BASE}/{repo}/resolve/main/{q}"))
                groups = qc.get("config_groups", {})
                for k, v in (groups or {}).items():
                    w = v.get("weights", {}) if isinstance(v, dict) else {}
                    print(f"quant[{q}][{k}] group={w.get('group_size')} num_bits={w.get('num_bits')}")
            except Exception as e:  # noqa: BLE001 — 진단 도구
                print(f"quant[{q}] parse fail: {e}")
    idx = {}
    for x in files:
        if x["path"].endswith("model.safetensors.index.json"):
            idx = json.loads(fetch(f"{BASE}/{repo}/resolve/main/{x['path']}"))
            break
    wmap = idx.get("weight_map", {})
    if not wmap:
        print("index 부재/빈 weight_map — 단일 샤드 가정")
    targets = {}
    for sub in subs:
        for name in wmap:
            if sub in name:
                targets.setdefault(wmap[name], []).append(name)
    if not wmap:
        # 샤드 목록에서 첫 샤드로 요약
        shards = [x["path"] for x in files if x["path"].endswith(".safetensors")]
        if shards:
            targets[shards[0]] = []
    for shard, names in targets.items():
        h = st_header(repo, shard)
        print(f"-- {shard} ({len(h)} entries)")
        for name in names:
            v = h.get(name)
            print(f"   {name} {v['dtype'] if v else 'MISSING'} {v['shape'] if v else ''}")
        if not names:
            # 트리플 요약: packed/scale/shape 개수 + 대표 shape + 스케일 열 분포
            pk = [k for k in h if k.endswith("weight_packed")]
            sc = {tuple(v["shape"][1:2]) for k, v in h.items()
                  if k.endswith("weight_scale")}
            dt = {v["dtype"] for k, v in h.items() if k.endswith("weight_scale")}
            print(f"   triples={len(pk)} scale_cols={sorted(sc)} scale_dtypes={dt}")


if __name__ == "__main__":
    main()
