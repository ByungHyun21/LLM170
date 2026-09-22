#!/usr/bin/env python3
"""patch_sdot — sdot4p 센티널 커널을 OpSDot(SDPACKED)로 이진 패치.

Ubuntu spirv-tools 2025.1 어셈블러는 PackedVectorFormat4x8Bit 리터럴을
거부하므로(SAIL 미등록), 센티널 GLSL(sdot4p — 4분해곱 사슬)을 glslc로
컴파일한 뒤 spirv-dis 텍스트에서 사슬을 식별해 SPIR-V 워드를 직접 교체한다:

  1. 최종 IAdd(IAdd(m0,m1), IAdd(m2,m3)) — 각 mI = IMul(Bitcast(추출)) —
     에서 4곱의 피연산자 소스가 단일 a·b 워드로 수렴하는 사이트만 치환.
     추출 언랩 규칙: ShiftRightLogical 항상 통과, BitwiseAnd는 상수 255
     (바이트 추출 마스크)일 때만 통과 — 패킹 마스크(0x0F0F0F0F 등)는
     불투명하고 그 결과가 SDot 피연산자가 된다. SDot 피연산자는 죽은
     사슬 NOP 대상에서 제외(live).
  2. 죽은 중간 명령(추출/Bitcast/IMul/부분 IAdd)은 워드 수를 유지한
     OpNop 타일로 채운다.
  3. 최종 IAdd 5워드 → OpSDot(opcode 4450) 6워드 + format=0 삽입.
  4. OpCapability DotProductInput4x8BitPacked(6017)·DotProduct(6019) 추가.

산술 의미: OpSDot(PackedVectorFormat4x8Bit) = i8 sext 내적 — 기존
dot4(부호 있는 바이트)와 비트 동일. VkCtx가 켜는
Vulkan13Features::shaderIntegerDotProduct 가 실행 조건. 치환 0건이면
실패 종료(미치환 센티널은 상위바이트 무부호 오차 — 방지).

사용: patch_sdot.py <sentinel.spv> <out.spv>
빌드 순서: build_spv.py <comp> <tmp> && patch_sdot.py <tmp> <spv>
"""
import os
import re
import struct
import subprocess
import sys

NOP = 0x00010000
OP_CAPABILITY = 17
CAP_DOT_INPUT_4X8_PACKED = 6017
CAP_DOT = 6019
OP_SDOT = 4450


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    src, out = sys.argv[1], sys.argv[2]
    dis = subprocess.run(
        ["spirv-dis", "--no-color", src], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    info, order, consts = {}, [], {}
    for line in dis:
        m = re.match(r"\s*%([\w.]+) = (\w+)(.*)", line)
        if not m:
            continue
        rid, op, rest = m.group(1), m.group(2), m.group(3)
        info[rid] = (len(order), op, re.findall(r"%([\w.]+)", rest))
        order.append(rid)
        if op == "OpConstant":
            lit = re.search(r"OpConstant %[\w.]+ (\d+)", line)
            if lit:
                consts[rid] = int(lit.group(1))

    def ops(rid: str) -> list[str]:
        return info[rid][2][1:]  # 결과타입 토큰 제거

    def bc_src(rid: str) -> str | None:
        d = info.get(rid)
        if d and d[1] == "OpBitcast" and len(d[2]) >= 2:
            return d[2][1]
        return None

    def base(rid: str, depth: int = 0) -> str | None:
        if depth > 4:
            return None
        d = info.get(rid)
        if not d:
            return None
        if d[1] == "OpShiftRightLogical":
            o = d[2][1:]
            return base(o[0], depth + 1) if o else None
        if d[1] == "OpBitwiseAnd" and len(d[2]) >= 3 and consts.get(d[2][2]) == 255:
            o = d[2][1:]
            return base(o[0], depth + 1) if o else None
        return rid

    targets: list[tuple[str, str, str, set[str]]] = []
    for r in order:
        if info[r][1] != "OpIAdd":
            continue
        ar = ops(r)
        if len(ar) != 2 or ar[0] not in info or ar[1] not in info:
            continue
        d1, d2 = info[ar[0]], info[ar[1]]
        if d1[1] != "OpIAdd" or d2[1] != "OpIAdd":
            continue
        mids = ops(ar[0]) + ops(ar[1])
        if len(mids) != 4 or not all(x in info and info[x][1] == "OpIMul" for x in mids):
            continue
        pairs, ok = [], True
        for x in mids:
            mo = ops(x)
            ca, cb = bc_src(mo[0]), bc_src(mo[1])
            if ca is None or cb is None:
                ok = False
                break
            ba, bb = base(ca), base(cb)
            if ba is None or bb is None:
                ok = False
                break
            pairs.append((ba, bb))
        if not ok:
            continue
        a_ids, b_ids = {p[0] for p in pairs}, {p[1] for p in pairs}
        if len(a_ids) != 1 or len(b_ids) != 1:
            continue
        a_id, b_id = pairs[0]
        live = {a_id, b_id}
        dead = set(mids + ar)

        def chain(rid: str, depth: int = 0) -> None:
            if depth > 6 or rid in dead or rid in live:
                return
            d = info.get(rid)
            if d and d[1] in ("OpBitcast", "OpBitwiseAnd", "OpShiftRightLogical"):
                dead.add(rid)
                for y in d[2][1:]:
                    chain(y, depth + 1)

        for x in mids:
            for y in ops(x):
                chain(y)
        targets.append((r, a_id, b_id, dead))

    if not targets:
        print("patch_sdot: 치환 0건 — sdot4p 센티널 미발견", file=sys.stderr)
        sys.exit(1)

    words = list(struct.unpack("<%dI" % (os.path.getsize(src) // 4), open(src, "rb").read()))

    def pos_of(rid: str) -> int | None:
        v = int(rid)
        i = 5
        while i < len(words):
            hdr = words[i]
            wc = hdr >> 16
            if wc == 0:
                break
            if wc >= 3 and i + 2 < i + wc and words[i + 2] == v:
                return i
            i += wc
        return None

    patches = []
    for res, a, b, dead in targets:
        rp = pos_of(res)
        if rp is None:
            print(f"patch_sdot: 이진 위치 미발견 %{res}", file=sys.stderr)
            sys.exit(1)
        deads = []
        for d in dead:
            dp = pos_of(d)
            if dp is None or dp >= rp:
                print(f"patch_sdot: 죽은 id %{d} 위치 오류", file=sys.stderr)
                sys.exit(1)
            deads.append((dp, words[dp] >> 16))
        patches.append((rp, int(a), int(b), deads))

    for rp, a, b, deads in sorted(patches, key=lambda t: -t[0]):
        words[rp : rp + 5] = [(6 << 16) | OP_SDOT, words[rp + 1], words[rp + 2], a, b, 0]
        for dp, wc in sorted(deads, key=lambda t: -t[0]):
            words[dp : dp + wc] = [NOP] * wc

    i = 5
    insert_at = None
    while i < len(words):
        hdr = words[i]
        wc, op = hdr >> 16, hdr & 0xFFFF
        if op == OP_CAPABILITY:
            insert_at = i + wc
            i += wc
        else:
            break
    if insert_at is None:
        print("patch_sdot: OpCapability 블록 미발견", file=sys.stderr)
        sys.exit(1)
    words[insert_at:insert_at] = [
        (2 << 16) | OP_CAPABILITY,
        CAP_DOT_INPUT_4X8_PACKED,
        (2 << 16) | OP_CAPABILITY,
        CAP_DOT,
    ]

    open(out, "wb").write(struct.pack("<%dI" % len(words), *words))
    print(f"patch_sdot: {len(targets)}개 사이트 OpSDot 치환 완료 → {out}")


if __name__ == "__main__":
    main()
