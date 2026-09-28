#!/usr/bin/env python3
"""GLSL .comp -> SPIR-V 빌드 (plans/12). 사전컴파일해 include_bytes!로 임베드.
사용: scripts/build_spv.py <src.comp> <out.spv>
"""
import subprocess
import sys


def main() -> None:
    src, out = sys.argv[1], sys.argv[2]
    r = subprocess.run(
        ["glslc", "--target-env=vulkan1.3", "-O", src, "-o", out],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        sys.exit(1)
    # plans/105: sdot4p 센티널 커널은 패처 통과 필수 — 미패치 spv는 상위바이트
    # 무부호 오차(값 파손)+스칼라 사슬(2× 지연). 원장 64 실사고.
    if "sdot4p" in open(src).read():
        rp = subprocess.run(
            [sys.executable, __file__.replace("build_spv.py", "patch_sdot.py"), out, out],
            capture_output=True, text=True)
        print(rp.stdout, end="")
        if rp.returncode != 0:
            print(rp.stderr)
            sys.exit(1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
