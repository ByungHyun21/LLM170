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
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
