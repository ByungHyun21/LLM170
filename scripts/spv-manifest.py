#!/usr/bin/env python3
"""spv-manifest — SPIR-V 자산 원장 (plans/107 W3).

spv/ 바이너리의 신선도·재빌드 가능성·sdot 패치 여부를 매니페스트로
lock한다 (스테일 spv 사고 4회 방지 —VkAudit).

모드:
  (무인수)      현재 상태를 TSV로 인쇄.
  --write       scripts/.spv-manifest.tsv 에 기록(커밋용).
  --check       기록된 해시와 대조 — .comp가 spv보다 새우거나 해시가
                다르면 실패(preflight 4b). 신규/삭제 spv도 보고.

컬럼: spv명 · comp유무(재빌드 가능) · patched(OpSDot 4450 탐지) · sha256.
patch_sdot 산출물(.comp 없는 prebuilt 포함)은 해시 검증으로 무결성 확보.
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPV = ROOT / "crates/backend-gpu/src/rawvk/spv"
MANIFEST = ROOT / "scripts/.spv-manifest.tsv"
SDOT_OP = 4450


def covered_by(name: str) -> str:
    """A22(plans/129) — spv 커버리지 원장(v1 분류). direct = 모듈 검증기,
    assembly = 조립 게이트. 전수 1:1 금지(ADR-0019): 원장이 빠짐을 계약으로
    막는다. 커널 추가 시 분류 갱신 — check 모드에서 빈 값/미기입은 FAIL."""
    n = name.removesuffix(".spv")
    if n.startswith("exl3_"):
        if "scan" in n: return "direct:exl3-scan-check"
        if "gdn" in n: return "direct:exl3-chain-check"
        if "attn" in n: return "direct:exl3-attn-check"
        if "gemmd" in n: return "direct:exl3-gemmd-check"
        if "gemm" in n: return "direct:exl3-gemm-check"
        if "nrh" in n or "norm" in n: return "direct:exl3-nr-check"
        if "ffn" in n: return "direct:exl3-ffn-check"
        return "assembly:gate-exl3"
    if n.startswith("gdn"): return "direct:gdn-check"
    if n.startswith("qsa"): return "direct:q4-qsa-check"
    if n.startswith("hc"): return "direct:q4-hc-check"
    if n.startswith("ple"): return "direct:q4-ple-check"
    if n.startswith("moe"): return "direct:moe-row-check"
    return "assembly:charhash+gate-27b+gate-flash-next"


def scan():
    rows = []
    for spv in sorted(SPV.glob("*.spv")):
        data = spv.read_bytes()
        has_comp = spv.with_suffix(".comp").exists()
        patched = any(
            int.from_bytes(data[i : i + 2], "little") == SDOT_OP
            for i in range(0, len(data) - 1, 4)
        )
        rows.append((spv.name, "comp" if has_comp else "prebuilt",
                     "sdot" if patched else "-", hashlib.sha256(data).hexdigest()[:16],
                     covered_by(spv.name)))
    return rows


def main():
    args = sys.argv[1:]
    rows = scan()
    if "--write" in args:
        MANIFEST.write_text(
            "\n".join("\t".join(r) for r in rows) + "\n")
        print(f"manifest: {len(rows)} spv recorded")
        return
    if "--check" in args:
        if not MANIFEST.exists():
            print("manifest 없음 — --write 로 기록 먼저")
            return 1
        old = {l.split("\t")[0]: l.rstrip("\n").split("\t") for l in MANIFEST.read_text().splitlines()}
        fail = 0
        cur = {r[0]: r for r in rows}
        for name, r in cur.items():
            o = old.get(name)
            if o is None:
                print(f"  NEW: {name}"); fail += 1
            elif o[3] != r[3]:
                print(f"  HASH-DIFF: {name}"); fail += 1
        for name in old:
            if name not in cur:
                print(f"  GONE: {name}"); fail += 1
        # .comp 신선도
        for comp in sorted(SPV.glob("*.comp")):
            spv = comp.with_suffix(".spv")
            if spv.exists() and comp.stat().st_mtime > spv.stat().st_mtime:
                print(f"  STALE: {comp.name} (comp newer than spv)"); fail += 1
            elif not spv.exists():
                # A17(plans/129): .comp 단독(미컴파일) 탐지 — 종전엔 spv 존재
                # 전제라 신규 .comp가 매니페스트·preflight 양쪽에서 무탐지였다.
                print(f"  NO-SPV: {comp.name} (comp without spv)"); fail += 1
        # A22: covered_by 원장 정합 — 컬럼 5가 비었으면 FAIL
        for line in MANIFEST.read_text().splitlines():
            cols = line.rstrip("\n").split("\t")
            if len(cols) >= 5 and not cols[4].strip():
                print(f"  UNCOVERED: {cols[0]}"); fail += 1
        print("spv-manifest PASS" if fail == 0 else f"spv-manifest FAIL ({fail})")
        return 0 if fail == 0 else 1
    for r in rows:
        print("\t".join(r))


if __name__ == "__main__":
    sys.exit(main() or 0)
