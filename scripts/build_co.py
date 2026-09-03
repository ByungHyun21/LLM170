#!/usr/bin/env python3
"""ROCm offline CO builder: full.cu -> .co loadable via LLM170_CO*_PATH.
Recipe (validated 2026-09-05 against v4all.co, live-loaded mism 0):
  hipcc -c --offload-arch=gfx1151 -> host .o with .hip_fatbin bundle
  -> extract `hipv4-amdgcn-amd-amdhsa--gfx1151` entry -> use directly as .co
gfx1151 defaults to wave32, so the (ignored) amdgpu_wavesize_eb32 attr is harmless.
Usage: scripts/build_co.py <full.cu> <out.co>
"""
import struct
import subprocess
import sys

ROCM = "/opt/rocm-7.2.2"
HIPCC = f"{ROCM}/bin/hipcc"
WANT = "hipv4-amdgcn-amd-amdhsa--gfx1151"


def main() -> None:
    src, out = sys.argv[1], sys.argv[2]
    obj = out + ".tmp.o"
    r = subprocess.run(
        [HIPCC, "--offload-arch=gfx1151", "-O3", "-c", src, "-o", obj],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        sys.exit(1)
    d = open(obj, "rb").read()
    base = d.find(b"__CLANG_OFFLOAD_BUNDLE__")
    assert base > 0, "no offload bundle in .o"
    off = base + 24
    (n,) = struct.unpack_from("<Q", d, off)
    off += 8
    for _ in range(n):
        bo, bs, il = struct.unpack_from("<QQQ", d, off)
        off += 24
        bid = d[off:off + il].decode()
        off += il
        if bid == WANT:
            open(out, "wb").write(d[base + bo:base + bo + bs])
            print(f"wrote {out} ({bs} bytes) from {src}")
            return
    print(f"target {WANT} not found")
    sys.exit(1)


if __name__ == "__main__":
    main()
