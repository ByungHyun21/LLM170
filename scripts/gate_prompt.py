#!/usr/bin/env python3
"""게이트 프롬프트 토큰열 → 원문 복원 검사 (plans/cuda-port.md S9).

왜 필요한가: scripts/*.sh의 PROMPT는 "한국어 문장"이라는 주석과 달리
어휘에 없는 문자를 토크나이저에 넣은 결과물이었다(원장 S9). 게이트는
토큰열 고정만 검증하므로 이 상태로도 PASS했고, 백엔드가 이상한 입력을
일관되게 처리한다는 것만 증명했다. 사람이 읽는 한국어가 실렸는지는
아무도 검사하지 않았다.

이 스크립트는 llama.cpp byte-level BPE 관례(GPT-2 bytes_to_unicode +
merges.txt 랭크)로 토큰열을 원문 바이트로 되돌리고, 반대로 텍스트를
토큰열로 인코딩한다. 게이트 스크립트의 PROMPT를 고칠 때 토큰열 생성기로
도 쓴다(하드코딩 제거 — 모델 어휘가 바뀌면 토큰열도 바뀌어야 한다).

사용:
  python3 scripts/gate_prompt.py <model_dir> --all          # 전 게이트 검사
  python3 scripts/gate_prompt.py <model_dir> --check 386,18  # 특정 토큰열
  python3 scripts/gate_prompt.py <model_dir> --encode "한글 문장"  # 토큰열 생성
"""
import json
import re
import sys
from pathlib import Path

GATES = [
    "scripts/gate-exl3.sh",
    "scripts/gate-27b.sh",
    "scripts/gate-flash-next.sh",
    "scripts/charhash.sh",
]


def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class Bpe:
    """vocab.json + merges.txt 기반 byte-level BPE (llama.cpp 관례 미러)."""

    def __init__(self, model_dir):
        self.b2u = bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}
        self.vocab = json.loads(Path(model_dir, "vocab.json").read_text())
        self.id2s = {v: k for k, v in self.vocab.items()}
        # 바이트열 -> id (인코딩 결과 조회용)
        self.byte2id = {}
        for piece, i in self.vocab.items():
            self.byte2id.setdefault(self.piece_to_bytes(piece), i)
        self.ranks = {}
        for line in Path(model_dir, "merges.txt").read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split(" ")
            if len(parts) != 2:
                continue
            self.ranks.setdefault((parts[0], parts[1]), len(self.ranks))

    def piece_to_bytes(self, piece):
        out = bytearray()
        for c in piece:
            if c in self.u2b:
                out.append(self.u2b[c])
            else:
                out.extend(c.encode("utf-8"))
        return bytes(out)

    def piece_bytes(self, tok):
        s = self.id2s.get(tok)
        return None if s is None else self.piece_to_bytes(s)

    def decode(self, toks):
        raw = b"".join(self.piece_bytes(t) or b"?" for t in toks)
        return raw, raw.decode("utf-8", "replace")

    def pre_tokenize(self, text):
        r"""Qwen2 pretokenizer 정규식 근사: \p{L}+ | \d+ | ?[^\s\w]+ | \s+"""
        return re.findall(r"[^\W\d_]+|\d+| ?[^\s\w]+|\s+", text, re.UNICODE)

    def encode(self, text):
        out = []
        for chunk in self.pre_tokenize(text):
            syms = [self.b2u[b] for b in chunk.encode("utf-8")]
            for w in self._bpe(syms):
                tid = self.byte2id.get(self.piece_to_bytes(w))
                if tid is None:
                    # 어휘 밖: 바이트 단위로 폴백(모델은 없지만 진단용)
                    for c in w:
                        b = self.u2b.get(c)
                        if b is not None and bytes([b]) in self.byte2id:
                            out.append(self.byte2id[bytes([b])])
                else:
                    out.append(tid)
        return out

    def _bpe(self, syms):
        word = list(syms)
        while len(word) > 1:
            best, bi = None, None
            for i in range(len(word) - 1):
                r = self.ranks.get((word[i], word[i + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if bi is None:
                break
            word[bi : bi + 2] = [word[bi] + word[bi + 1]]
        return word


def check(bpe, toks, tag):
    raw, text = bpe.decode(toks)
    try:
        raw.decode("utf-8")
        valid_utf8 = True
    except UnicodeDecodeError:
        valid_utf8 = False
    has_ko = any("가" <= c <= "힣" for c in text)
    oov = sum(1 for t in toks if bpe.piece_bytes(t) is None)
    print(f"[{tag}] 토큰 {len(toks)}개")
    print(f"  UTF-8 유효 : {valid_utf8}")
    print(f"  한글 포함   : {has_ko}")
    print(f"  어휘 밖 조각: {oov}")
    print(f"  원문       : {text[:140]}")
    return valid_utf8 and has_ko and oov == 0


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    bpe = Bpe(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "--all"
    if mode == "--encode":
        text = sys.argv[3]
        toks = bpe.encode(text)
        print(",".join(map(str, toks)))
        print(f"({len(toks)} 토큰)", file=sys.stderr)
        return 0
    if mode == "--check":
        toks = [int(x) for x in sys.argv[3].split(",")]
        return 0 if check(bpe, toks, "argv") else 1
    ok = True
    for g in GATES:
        p = Path(g)
        if not p.exists():
            continue
        m = re.search(r'^PROMPT="([^"]+)"', p.read_text(), re.M)
        if not m:
            continue
        toks = [int(x) for x in m.group(1).split(",")]
        ok &= check(bpe, toks, g)
        print()
    print(f"=== 전 게이트 한국어 판정: {'PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())