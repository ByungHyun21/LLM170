#!/usr/bin/env python3
"""gemv5 정준 dot 패턴 → OpSDot 어셈블리 치환 (plans/33).

glslc 툴체인(shaderc 2023.8 / Ubuntu glslang 15.1)이 GL_EXT_integer_dot_product를
지원하지 않아, GLSL은 'DOT 패치 마커' 정준 패턴으로 -O0 컴파일 후
SPIR-V 어셈블리(scripts/spvtool dis)에서 해당 블록을 OpSDot로 교체한다.

사용: python3 scripts/patch_sdot.py <in.comp> <out.spv>
커널 내 마커: "// DOT 패치 마커 #LO#" / "#HI#" — sa0/sc0 변수 블록이 정준 형태여야 함.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    src, out = sys.argv[1], sys.argv[2]
    with tempfile.TemporaryDirectory() as td:
        o0 = Path(td) / "o0.spv"
        asm = Path(td) / "o0.spvasm"
        r = subprocess.run(["glslc", "--target-env=vulkan1.3", src, "-o", str(o0)])
        assert r.returncode == 0
        r = subprocess.run([str(Path(__file__).parent / "spvtool"), "dis", str(o0), str(asm)])
        assert r.returncode == 0
        lines = asm.read_text().split("\n")

        def patch(sa0: str, a_var: str, b_var: str) -> None:
            start = next(i for i, l in enumerate(lines) if f"OpLoad %int %{sa0}" in l)
            cnt = 0
            end = None
            for i in range(start, len(lines)):
                if "OpStore %qsum" in lines[i]:
                    cnt += 1
                    if cnt == 4:
                        end = i
                        break
            assert end is not None
            block = "\n".join(lines[start : end + 1])
            muls = re.findall(r"%(\d+) = OpIMul %int", block)
            adds = re.findall(r"%(\d+) = OpIAdd %int", block)
            id_dot, id_la, id_lb, id_q, id_s = muls[0], adds[0], adds[1], adds[2], adds[3]
            new = [
                f"        %{id_la} = OpLoad %uint %{a_var}",
                f"        %{id_lb} = OpLoad %uint %{b_var}",
                f"        %{id_dot} = OpSDot %int %{id_la} %{id_lb} PackedVectorFormat4x8Bit",
                f"        %{id_q} = OpLoad %int %qsum",
                f"        %{id_s} = OpIAdd %int %{id_q} %{id_dot}",
                f"               OpStore %qsum %{id_s}",
            ]
            lines[start : end + 1] = new

        patch("sa0", "aw_lo", "bw")
        patch("sc0", "aw_hi", "bw2")
        ci = 0
        for i, l in enumerate(lines):
            if l.strip().startswith("OpCapability"):
                ci = i
        lines.insert(ci + 1, 'OpExtension "SPV_KHR_integer_dot_product"')
        lines.insert(ci + 1, "OpCapability DotProduct")
        patched_asm = Path(td) / "patched.spvasm"
        patched_asm.write_text("\n".join(lines))
        r = subprocess.run(
            [str(Path(__file__).parent / "spvtool"), "asm", str(patched_asm), out])
        assert r.returncode == 0
    print(f"wrote {out} (OpSDot patched)")


if __name__ == "__main__":
    main()
