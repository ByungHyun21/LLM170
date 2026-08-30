#!/usr/bin/env python3
"""무작위 소형 qwen35 GGUF 생성 — llm170 vs llama.cpp 그래프 동등성 검증용.

실제 27B에서 토크나이저 kv를 통째로 복사하고, 가중치는 시드 고정 난수 f32.
차원 축소: n_embd 128, 8층(interval 4 → il∈{3,7} full-attn), GDN k2/v6·state16,
attn 4q/2kv·head32·rot8. nextn 없음.
"""
import random
import struct
import sys

SRC = "/home/yoon/local_llm/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf"
DST = sys.argv[1] if len(sys.argv) > 1 else "/tmp/qwen35-tiny.gguf"

N_EMBD = 128
N_LAYER = int(sys.argv[2]) if len(sys.argv) > 2 else 8
FULL_ATTN_INTERVAL = int(sys.argv[3]) if len(sys.argv) > 3 else 4
N_FF = 256
N_HEAD = 4
N_KV = 2
HEAD_DIM = 32
D_STATE = 16
N_GROUP = 2   # GDN K heads
DT_RANK = 6   # GDN V heads
D_INNER = DT_RANK * D_STATE  # 96
CONV_K = 4
CONV_CH = D_INNER + 2 * N_GROUP * D_STATE  # 160
VOCAB = 248320

rng = random.Random(170)
F32 = struct.Struct("<f")


def rand(n, scale):
    return b"".join(F32.pack(rng.uniform(-scale, scale)) for _ in range(n))


def near1(n):
    return b"".join(F32.pack(1.0 + rng.uniform(-0.02, 0.02)) for _ in range(n))


# ---------- 원본 토크나이저 kv 복사 ----------
def read_src_kvs():
    f = open(SRC, "rb")
    assert f.read(4) == b"GGUF"
    f.read(4)  # version
    n_t, n_kv = struct.unpack("<QQ", f.read(16))

    def rstr():
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n)

    SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
           7: "<B", 10: "<Q", 11: "<q", 12: "<d"}

    def rval(t):
        if t == 8:
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n)
        if t == 9:
            et = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            return (et, [rval(et) for _ in range(n)])
        return struct.unpack(FMT[t], f.read(SZ[t]))[0]

    kvs = []
    for _ in range(n_kv):
        k = rstr().decode()
        t = struct.unpack("<I", f.read(4))[0]
        kvs.append((k, t, rval(t)))
    return kvs


TYPE_NAMES = {0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32", 6: "f32",
              7: "bool", 8: "str", 9: "arr", 10: "u64", 11: "i64", 12: "f64"}
TYPE_IDS = {v: k for k, v in TYPE_NAMES.items()}
PACK = {"u8": "<B", "i8": "<b", "u16": "<H", "i16": "<h", "u32": "<I",
        "i32": "<i", "f32": "<f", "u64": "<Q", "i64": "<q", "f64": "<d", "bool": "<B"}


class Buf:
    def __init__(self):
        self.b = bytearray()

    def u32(self, v):
        self.b += struct.pack("<I", v)

    def u64(self, v):
        self.b += struct.pack("<Q", v)

    def f32(self, v):
        self.b += struct.pack("<f", v)

    def raw(self, d):
        self.b += d

    def s(self, text):
        e = text.encode()
        self.u64(len(e))
        self.raw(e)

    def val(self, t, v):
        n = TYPE_NAMES[t]
        if n == "str":
            self.u64(len(v))
            self.raw(v)
        elif n == "arr":
            et, items = v
            self.u32(et)
            self.u64(len(items))
            for it in items:
                self.val(et, it)
        else:
            self.b += struct.pack(PACK[n], v)


def build():
    kv = []
    for k, t, v in read_src_kvs():
        if k.startswith("tokenizer.ggml.") or k == "general.name":
            kv.append((k, t, v))

    # 실제 파일의 타입 그대로 (llama.cpp hparams는 타입 엄격)
    src_types = {k: t for k, t, _ in read_src_kvs() if k.startswith("qwen35.")}

    def add(k, v):
        t = src_types.get(k)
        if t is None:
            t = TYPE_IDS["f32" if isinstance(v, float) else "u32"]
        kv.append((k, t, v))

    add("general.architecture", 0)  # int → u64... 실제론 str이어야 — 아래에서 교정
    kv[-1] = ("general.architecture", TYPE_IDS["str"], b"qwen35")
    add("qwen35.block_count", N_LAYER)
    add("qwen35.context_length", 2048)
    add("qwen35.embedding_length", N_EMBD)
    add("qwen35.feed_forward_length", N_FF)
    add("qwen35.attention.head_count", N_HEAD)
    add("qwen35.attention.head_count_kv", N_KV)
    add("qwen35.attention.key_length", HEAD_DIM)
    kv.append(("qwen35.attention.layer_norm_rms_epsilon", TYPE_IDS["f32"], 1e-6))
    kv.append(("qwen35.rope.freq_base", TYPE_IDS["f32"], 1e7))
    add("qwen35.rope.dimension_count", 8)
    kv.append(("qwen35.rope.dimension_sections", TYPE_IDS["arr"], (TYPE_IDS["i32"], [2, 2, 2, 2])))
    add("qwen35.full_attention_interval", FULL_ATTN_INTERVAL)
    add("qwen35.ssm.conv_kernel", CONV_K)
    add("qwen35.ssm.state_size", D_STATE)
    add("qwen35.ssm.group_count", N_GROUP)
    add("qwen35.ssm.time_step_rank", DT_RANK)
    add("qwen35.ssm.inner_size", D_INNER)
    kv.append(("general.alignment", TYPE_IDS["u32"], 32))

    tensors = []  # (name, dims, bytes)

    def t(name, dims, data):
        tensors.append((name, dims, data))

    t("token_embd.weight", [N_EMBD, VOCAB], rand(N_EMBD * VOCAB, 0.02))
    t("output.weight", [N_EMBD, VOCAB], rand(N_EMBD * VOCAB, 0.02))
    t("output_norm.weight", [N_EMBD], near1(N_EMBD))

    for il in range(N_LAYER):
        p = f"blk.{il}."
        t(p + "attn_norm.weight", [N_EMBD], near1(N_EMBD))
        t(p + "post_attention_norm.weight", [N_EMBD], near1(N_EMBD))
        t(p + "ffn_gate.weight", [N_EMBD, N_FF], rand(N_EMBD * N_FF, 0.02))
        t(p + "ffn_up.weight", [N_EMBD, N_FF], rand(N_EMBD * N_FF, 0.02))
        t(p + "ffn_down.weight", [N_FF, N_EMBD], rand(N_FF * N_EMBD, 0.02))
        if il % FULL_ATTN_INTERVAL == FULL_ATTN_INTERVAL - 1:
            t(p + "attn_q.weight", [N_EMBD, N_HEAD * HEAD_DIM * 2], rand(N_EMBD * N_HEAD * HEAD_DIM * 2, 0.02))
            t(p + "attn_k.weight", [N_EMBD, N_KV * HEAD_DIM], rand(N_EMBD * N_KV * HEAD_DIM, 0.02))
            t(p + "attn_v.weight", [N_EMBD, N_KV * HEAD_DIM], rand(N_EMBD * N_KV * HEAD_DIM, 0.02))
            t(p + "attn_output.weight", [N_HEAD * HEAD_DIM, N_EMBD], rand(N_HEAD * HEAD_DIM * N_EMBD, 0.02))
            t(p + "attn_q_norm.weight", [HEAD_DIM], near1(HEAD_DIM))
            t(p + "attn_k_norm.weight", [HEAD_DIM], near1(HEAD_DIM))
        else:
            t(p + "attn_qkv.weight", [N_EMBD, CONV_CH], rand(N_EMBD * CONV_CH, 0.02))
            t(p + "attn_gate.weight", [N_EMBD, D_INNER], rand(N_EMBD * D_INNER, 0.02))
            t(p + "ssm_beta.weight", [N_EMBD, DT_RANK], rand(N_EMBD * DT_RANK, 0.02))
            t(p + "ssm_alpha.weight", [N_EMBD, DT_RANK], rand(N_EMBD * DT_RANK, 0.02))
            t(p + "ssm_a", [DT_RANK],
              b"".join(F32.pack(-abs(rng.uniform(0.1, 1.0))) for _ in range(DT_RANK)))
            t(p + "ssm_dt.bias", [DT_RANK],
              b"".join(F32.pack(rng.uniform(-0.1, 0.1)) for _ in range(DT_RANK)))
            t(p + "ssm_conv1d.weight", [CONV_K, CONV_CH], rand(CONV_K * CONV_CH, 0.05))
            t(p + "ssm_norm.weight", [D_STATE], near1(D_STATE))
            t(p + "ssm_out.weight", [D_INNER, N_EMBD], rand(D_INNER * N_EMBD, 0.02))
    return kv, tensors


def main():
    kv, tensors = build()
    out = Buf()
    out.raw(b"GGUF")
    out.u32(3)
    out.u64(len(tensors))
    out.u64(len(kv))
    for k, ty, v in kv:
        out.s(k)
        out.u32(ty)
        out.val(ty, v)

    align = 32
    offsets = []
    off = 0
    for _, _, data in tensors:
        off += (-off) % align
        offsets.append(off)
        off += len(data)

    for (name, dims, _), o in zip(tensors, offsets):
        out.s(name)
        out.u32(len(dims))
        for d in dims:
            out.u64(d)
        out.u32(0)  # f32
        out.u64(o)

    data_start = (len(out.b) + align - 1) // align * align
    for (_, _, data), o in zip(tensors, offsets):
        target = data_start + o  # o는 데이터 섹션 기준 상대 오프셋
        while len(out.b) < target:
            out.b += b"\0"
        assert len(out.b) == target, (len(out.b), target)
        out.raw(data)

    open(DST, "wb").write(bytes(out.b))
    print(f"wrote {DST}: {len(out.b)/1e6:.1f} MB, {len(tensors)} tensors")


main()
