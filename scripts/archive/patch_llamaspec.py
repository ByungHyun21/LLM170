#!/usr/bin/env python3
"""patch_llamaspec.py — llama 포팅 커널의 OpSpecConstant 기본값 패치.

spirv-dis → sed → spirv-as 왕복(이진 워크 오류 전례로 폐기). -O 빌드는
이름이 사라지므로 no-O 빌드에만 적용.

사용: patch_llamaspec.py in.spv out.spv NAME=VALUE [NAME=VALUE ...]
  예: patch_llamaspec.py tmp.spv out.spv MmTypeA=12 ALIGNED=1
"""
import subprocess
import sys
import tempfile


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    wants = dict(a.split("=") for a in sys.argv[3:])

    dis = subprocess.run(["spirv-dis", src, "-o", "/tmp/_pls.dis"], capture_output=True, text=True)
    if dis.returncode != 0:
        print(dis.stderr)
        sys.exit(1)
    text = open("/tmp/_pls.dis").read()
    n = 0
    for name, val in wants.items():
        old = f"    %{name} = OpSpecConstant"
        for line in text.split("\n"):
            if line.startswith(old):
                new_line = f"    %{name} = OpSpecConstant %uint {val}"
                if ";" in line:
                    new_line += ";"
                text = text.replace(line, new_line)
                n += 1
                break
        else:
            print(f"경고: %{name} 미발견")
    with open("/tmp/_pls.dis", "w") as f:
        f.write(text)
    r = subprocess.run(["spirv-as", "/tmp/_pls.dis", "-o", dst], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        sys.exit(1)
    subprocess.run(["spirv-val", dst], check=False)
    print(f"patch_llamaspec: {n}개 상수 패치 → {dst}")


if __name__ == "__main__":
    main()
