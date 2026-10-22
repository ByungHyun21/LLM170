#!/usr/bin/env python3
import io
import os
import sys
import tokenize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTS = {".rs", ".cu", ".cuh", ".c", ".h", ".hpp", ".cpp", ".sh", ".py"}
SKIP_DIRS = {".git", "target", "plans", "reference", ".omo", "__pycache__", "eval-data"}
SKIP_PREFIXES = ("crates/backend-gpu/src/rawcuda/assets/marlin",)


def rel(path):
    return os.path.relpath(path, ROOT)


def excluded(path):
    r = rel(path)
    if any(part in SKIP_DIRS for part in r.split(os.sep)):
        return True
    return any(r.startswith(p) for p in SKIP_PREFIXES)


def files():
    for base, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in names:
            if os.path.splitext(name)[1] not in EXTS:
                continue
            p = os.path.join(base, name)
            if not excluded(p):
                yield p


def line_col(text, pos):
    line = text.count("\n", 0, pos) + 1
    col = pos - text.rfind("\n", 0, pos)
    return line, col


def raw_prefix(text, i):
    n = len(text)
    j = i
    if j < n and text[j] in "bc" and j + 1 < n and text[j + 1] == "r":
        j += 1
    if j >= n or text[j] != "r":
        return None
    k = j + 1
    hashes = 0
    while k < n and text[k] == "#":
        hashes += 1
        k += 1
    if k < n and text[k] == '"':
        return k + 1, hashes
    return None


def scan_c(text, rust):
    out = list(text)
    hits = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            hits.append(line_col(text, i))
            while i < n and text[i] not in "\r\n":
                out[i] = " "
                i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            hits.append(line_col(text, i))
            depth = 1
            out[i] = " "
            out[i + 1] = " "
            i += 2
            while i < n and depth:
                if text[i] == "/" and i + 1 < n and text[i + 1] == "*":
                    depth += 1
                    out[i] = " "
                    out[i + 1] = " "
                    i += 2
                elif text[i] == "*" and i + 1 < n and text[i + 1] == "/":
                    depth -= 1
                    out[i] = " "
                    out[i + 1] = " "
                    i += 2
                else:
                    if text[i] not in "\r\n":
                        out[i] = " "
                    i += 1
            continue
        if rust:
            rp = raw_prefix(text, i)
            if rp is not None:
                start, hashes = rp
                close = '"' + "#" * hashes
                end = text.find(close, start)
                end = n if end < 0 else end + len(close)
                i = end
                continue
        if c == '"':
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == '"':
                    i += 1
                    break
                i += 1
            continue
        if c == "'":
            if rust:
                if i + 2 < n and text[i + 1] == "\\":
                    j = i + 2
                    while j < n and text[j] != "'":
                        j += 1
                    i = min(j + 1, n)
                    continue
                if i + 2 < n and text[i + 2] == "'":
                    i += 3
                    continue
                i += 1
                continue
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == "'":
                    i += 1
                    break
                i += 1
            continue
        i += 1
    return "".join(out), hits


def scan_hash(text):
    out = list(text)
    hits = []
    i = 0
    n = len(text)
    quote = None
    at_line_start = True
    while i < n:
        c = text[i]
        if quote is None:
            if c == "\\":
                i += 2
                at_line_start = False
                continue
            if c in "'\"":
                quote = c
                i += 1
                at_line_start = False
                continue
            if c == "#":
                line_start = text.rfind("\n", 0, i) + 1
                prefix = text[line_start:i]
                shebang = i == 0 and text.startswith("#!")
                if not shebang and (prefix.strip() == "" or i == 0 or text[i - 1] in " \t"):
                    hits.append(line_col(text, i))
                    while i < n and text[i] not in "\r\n":
                        out[i] = " "
                        i += 1
                    continue
            if c == "\n":
                at_line_start = True
            elif not c.isspace():
                at_line_start = False
            i += 1
            continue
        if c == "\\" and quote == '"':
            i += 2
            continue
        if c == quote:
            quote = None
            i += 1
            continue
        i += 1
    return "".join(out), hits


def scan_py(text, strip):
    hits = []
    if strip:
        lines = text.splitlines(keepends=True)
        edits = []
        try:
            for tok in tokenize.generate_tokens(io.StringIO(text).readline):
                if tok.type == tokenize.COMMENT:
                    if tok.start[0] == 1 and tok.string.startswith("#!"):
                        continue
                    hits.append((tok.start[0], tok.start[1] + 1))
                    edits.append((tok.start[0], tok.start[1], tok.end[1]))
        except (tokenize.TokenError, IndentationError):
            pass
        for line_no, a, b in edits:
            idx = line_no - 1
            if 0 <= idx < len(lines):
                line = lines[idx]
                lines[idx] = line[:a] + " " * (b - a) + line[b:]
        return "".join(lines), hits
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                if tok.start[0] == 1 and tok.string.startswith("#!"):
                    continue
                hits.append((tok.start[0], tok.start[1] + 1))
    except (tokenize.TokenError, IndentationError):
        pass
    return text, hits


def check(path, strip):
    ext = os.path.splitext(path)[1]
    try:
        text = open(path, encoding="utf-8").read()
    except UnicodeDecodeError:
        return [], False
    if ext == ".py":
        out, hits = scan_py(text, strip)
    elif ext in {".sh"}:
        out, hits = scan_hash(text)
    else:
        out, hits = scan_c(text, ext == ".rs")
    changed = strip and out != text
    if changed:
        open(path, "w", encoding="utf-8").write(out)
    return hits, changed


def main():
    strip = "--strip" in sys.argv[1:]
    bad = []
    changed = 0
    for path in files():
        hits, did = check(path, strip)
        if did:
            changed += 1
        for line, col in hits:
            bad.append((rel(path), line, col))
    if strip:
        print(f"stripped {changed} files, comments {len(bad)}")
        return 0
    if not bad:
        print("comments 0")
        return 0
    for path, line, col in bad[:40]:
        print(f"{path}:{line}:{col}: comment")
    print(f"comments {len(bad)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
