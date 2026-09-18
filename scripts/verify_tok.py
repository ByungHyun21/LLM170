#!/usr/bin/env python3
"""llm170 BPE 토크나이저 ↔ llama-tokenize 전수 대조 (plans/83 A).

코퍼스(저장소 문서 + 합성 엣지 케이스)를 전부 토큰화해 양쪽 id 스트림이
100% 일치하는지 검증한다. 특수 토큰 해석 on/off 양 모드 비교.

사용법:
  python3 scripts/verify_tok.py [--model /path/model.gguf] [--llama /path/llama-tokenize]
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# 합성 엣지 케이스 — 정규식 분기·병합·특수 토큰 전 영역 커버
SYNTHETIC = [
    "hello world",
    "hello world\n",
    "  leading spaces",
    "trailing spaces   ",
    "multiple   internal    spaces",
    "\n\n\n",
    " \r\n \t mix",
    "tab\tseparated\tvalues",
    "it's don't we're I've he'll she'd",
    "IT'S Don't WE'RE I'VE HE'LL SHE'D",
    "'s 't 're 've 'm 'll 'd",
    "x's y't z're",
    "numbers 1234567890 42 3.14159 1,000,000",
    "0x1F 0b1010 1e10 -42 +7",
    " Korean: 대한민국의 수도는 서울이고, 부산은 바닷가가 있는 도시다.",
    "한글과 English mixed 텍스트 with 숫자 123 and symbols!!!",
    "日本語のテキストです。混合 Chinese 中文文本。",
    "emoji: \U0001F600\U0001F680\U0001F44D plain text after",
    "combining: e\u0301 a\u0300 n\u0303",
    "arabic: مرحبا بالعالم",
    "hebrew: שלום עולם",
    "cyrillic: Привет мир",
    "greek: Γεια σου κόσμε",
    "!@#$%^&*()_+-=[]{}|;':\",./<>?`~",
    "code: fn main() { println!(\"hello\"); }",
    "code: if x >= 10 && y != 0 { z = x << 2; }",
    "url: https://example.com/path?query=1&other=2#frag",
    "html: <div class=\"foo\">bar</div>",
    "markdown: # Header\n\n- item\n- item2\n\n```rust\nlet x = 1;\n```",
    "special: <|im_start|>user\nhi<|im_end|>",
    "<|endoftext|>",
    "prefix <|im_start|>assistant middle <|im_end|> suffix",
    "<|im_start|><|im_start|>double",
    "not special: <| im_start |>",
    "empty then text",
    " ",
    "a",
    "\u00a0nbsp\u00a0",
    "\u200bzwsp\u200b zero width",
    "\u3000ideographic space\u3000",
    "line1\r\nline2\nline3\rline4",
    "unicode: ĀāĂăĄą ĠĊ control\x00bytes\x7f",
    "very long word: " + "a" * 200,
    "long korean: " + "대한민국" * 50,
    "long digits: " + "1234567890" * 20,
    "mixed ws: \t \n  \r\n \t\t mix",
    "'s at start",
    "'ll alone",
    "word' s' t're",
    "I''ll double apostrophe",
    "3.14 is pi",
    "e=mc^2",
    "《책 제목》 '인용' \"큰따옴표\"",
    "ｆｕｌｌｗｉｄｔｈ ｔｅｘｔ",
    "𝕌𝕟𝕚𝕔𝕠𝕕𝕖 mathematical",
    "㈀㈁㈂ parentheses",
    "ºª·½¼¾№",
]


def collect_corpus() -> list[tuple[str, bytes]]:
    corpus = []
    for pat in ("*.md", "*.rs", "*.py", "*.sh", "*.toml"):
        for p in sorted((REPO / "docs").rglob(pat)):
            corpus.append((str(p.relative_to(REPO)), p.read_bytes()))
    corpus.append(("README.md", (REPO / "README.md").read_bytes()))
    # 큰 파일은 64KB 청크로 분할 (양쪽 도구 모두 한 번에 처리 가능)
    out = []
    for name, data in corpus:
        if len(data) > 65536:
            for i in range(0, len(data), 65536):
                out.append((f"{name}#{i // 65536}", data[i : i + 65536]))
        else:
            out.append((name, data))
    for i, s in enumerate(SYNTHETIC):
        out.append((f"synthetic#{i}", s.encode()))
    return out
def run_llama(llama: str, model: str, data: bytes, parse_special: bool) -> list[int]:
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(data)
        path = f.name
    cmd = [llama, "-m", model, "--file", path, "--ids", "--no-escape"]
    if not parse_special:
        cmd.append("--no-parse-special")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"llama-tokenize failed: {r.stderr[-500:]}")
    out = r.stdout.strip()
    assert out.startswith("[") and out.endswith("]"), f"bad output: {out[:100]}"
    return json.loads(out)


def run_llm170(llm170: str, model: str, data: bytes, parse_special: bool) -> list[int]:
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(data)
        path = f.name
    cmd = [llm170, "tokenize", "--model", model, "--file", path]
    if not parse_special:
        cmd.append("--no-special")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"llm170 tokenize failed: {r.stderr[-500:]}")
    out = r.stdout.strip()
    assert out.startswith("[") and out.endswith("]"), f"bad output: {out[:100]}"
    return json.loads(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="/home/yoon/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf",
    )
    ap.add_argument("--llama", default="/tmp/lc-tok/bin/llama-tokenize")
    ap.add_argument("--llm170", default=str(REPO / "target/release/llm170"))
    ap.add_argument("--max-files", type=int, default=0)
    args = ap.parse_args()

    corpus = collect_corpus()
    if args.max_files:
        corpus = corpus[: args.max_files]

    total_toks = 0
    fails = 0
    for mode in (True, False):
        label = "special=on " if mode else "special=off"
        for name, data in corpus:
            try:
                a = run_llama(args.llama, args.model, data, mode)
                b = run_llm170(args.llm170, args.model, data, mode)
            except (RuntimeError, AssertionError) as e:
                print(f"FAIL [{label}] {name}: {e}")
                fails += 1
                continue
            if a != b:
                # 첫 불일치 위치 찾기
                i = next(i for i in range(min(len(a), len(b))) if a[i] != b[i]) if a != b else 0
                fails += 1
                print(
                    f"FAIL [{label}] {name}: llama={len(a)} toks llm170={len(b)} toks "
                    f"first diff at {i}: llama {a[max(0,i-3):i+5]} vs llm170 {b[max(0,i-3):i+5]}"
                )
            else:
                total_toks += len(a)
        if fails:
            print(f"[{label}] {fails} failures")
            return 1
        print(f"[{label}] all {len(corpus)} corpus files match")
    print(f"PASS: {total_toks} tokens total, 100% match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
