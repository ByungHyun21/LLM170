#!/usr/bin/env python3
"""Transpile llama.cpp unicode-data.cpp tables into Rust (crates/server/src/unicode_data.rs).

Source: source/llama.cpp/src/unicode-data.cpp (llama-tokenizer port, plans/83 A).
Regenerate: python3 scripts/gen_unicode_tables.py
"""

import re
from pathlib import Path

SRC = Path("source/llama.cpp/src/unicode-data.cpp")
DST = Path("crates/server/src/unicode_data.rs")


def extract(name: str) -> str:
    text = SRC.read_text()
    m = re.search(rf"{name} = \{{(.*?)\}};", text, re.S)
    assert m, name
    return m.group(1)


def pairs(body: str):
    out = []
    for a, b in re.findall(r"\{\s*(0x[0-9A-Fa-f]+)\s*,\s*(0x[0-9A-Fa-f]+)\s*\}", body):
        out.append((int(a, 16), int(b, 16)))
    return out


def singles(body: str):
    return [int(x, 16) for x in re.findall(r"0x[0-9A-Fa-f]+", body)]


ranges_flags = pairs(extract("unicode_ranges_flags"))
# sentinel: last entry is {MAX_CODEPOINTS, x} — keep as-is (start exclusive bound)
assert ranges_flags[-1][0] == 0x110000, hex(ranges_flags[-1][0])
whitespace = sorted(set(singles(extract("unicode_set_whitespace"))))
lowercase = pairs(extract("unicode_map_lowercase"))

lines = []
lines.append("//! llama.cpp unicode 카테고리 테이블 전용 (scripts/gen_unicode_tables.py 생성).")
lines.append("//! 원본: source/llama.cpp/src/unicode-data.cpp — 수동 수정 금지, 재생성만.")
lines.append("")
lines.append("/// (start, flags) — start 이상 ~ 다음 start 미만. 마지막 항목은 0x110000 상한.")
lines.append("pub static RANGES_FLAGS: [(u32, u16); %d] = [" % len(ranges_flags))
for a, b in ranges_flags:
    lines.append(f"    (0x{a:06X}, 0x{b:04X}),")
lines.append("];")
lines.append("")
lines.append("pub static SET_WHITESPACE: [u32; %d] = [" % len(whitespace))
for a in whitespace:
    lines.append(f"    0x{a:06X},")
lines.append("];")
lines.append("")
lines.append("pub static MAP_LOWERCASE: [(u32, u32); %d] = [" % len(lowercase))
for a, b in lowercase:
    lines.append(f"    (0x{a:06X}, 0x{b:06X}),")
lines.append("];")
lines.append("")

DST.write_text("\n".join(lines))
print(f"wrote {DST}: ranges_flags={len(ranges_flags)} ws={len(whitespace)} lower={len(lowercase)}")
