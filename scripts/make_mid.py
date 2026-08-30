#!/usr/bin/env python3
"""실제 qwen35 차원(5120/24-4/256/GDN 16-48-128/FFN 17408) + 무작위 가중치 q8_0 소형 모델.

4층(interval 4 → il 0-2 GDN, il 3 attn). 토크나이저는 27B에서 복사.
차원 특이 버그 격리용 — llama.cpp와 llm170의 그래프 동등성 검증.
"""
import random
import struct
import sys

import numpy as np

SRC = "/home/yoon/local_llm/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf"
DST = sys.argv[1] if len(sys.argv) > 1 else "/tmp/qwen35-mid.gguf"

N_EMBD = 5120
N_LAYER = int(sys.argv[2]) if len(sys.argv) > 2 else 4
FULL_ATTN_INTERVAL = int(sys.argv[3]) if len(sys.argv) > 3 else 4
N_FF = 17408
N_HEAD = 24
N_KV = 4
HEAD_DIM = 256
N_ROT = 64
D_STATE = 128
N_GROUP = 16
DT_RANK = 48
D_INNER = 6144
CONV_K = 4
CONV_CH = D_INNER + 2 * N_GROUP * D_STATE  # 10240
VOCAB = 248320

rng = np.random.default_rng(170)
F32 = struct.Struct("<f")
F16 = struct.Struct("<e")


def randn(n, scale):
    return (rng.standard_normal(n) * scale).astype("<f4").tobytes()


def near1(n):
    return (1.0 + rng.uniform(-0.02, 0.02, n)).astype("<f4").tobytes()


def q8_0(n):
    """무작위 값을 q8_0로 직접 생성: 32개별 블록, d=max/127. 벡터화."""
    w = rng.standard_normal(n).astype(np.float32) * 0.02
    w = w.reshape(-1, 32)
    m = np.maximum(np.abs(w).max(axis=1), 1e-9)
    d = (m / 127.0).astype("<f2")
    q = np.clip(np.rint(w / d[:, None]), -127, 127).astype(np.int8)
    out = np.empty((w.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def read_src_kvs():
    f = open(SRC, "rb")
    f.read(8)
    n_t, n_kv = struct.unpack("<QQ", f.read(16))
    SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
           7: "<B", 10: "<Q", 11: "<q", 12: "<d"}

    def rstr():
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode()

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
        k = rstr()
        t = struct.unpack("<I", f.read(4))[0]
        kvs.append((k, t, rval(t)))
    return kvs


TYPE_IDS = {"u8": 0, "i8": 1, "u16": 2, "i16": 3, "u32": 4, "i32": 5, "f32": 6,
            "bool": 7, "str": 8, "arr": 9, "u64": 10, "i64": 11, "f64": 12}
TYPE_NAMES = {v: k for k, v in TYPE_IDS.items()}
PACK = {"u8": "<B", "i8": "<b", "u16": "<H", "i16": "<h", "u32": "<I",
        "i32": "<i", "f32": "<f", "u64": "<Q", "i64": "<q", "f64": "<d", "bool": "<B"}


class Buf:
    def __init__(self):
        self.b = bytearray()

    def u32(self, v):
        self.b += struct.pack("<I", v)

    def u64(self, v):
        self.b += struct.pack("<Q", v)

    def raw(self, d):
        self.b += d

    def s(self, t):
        e = t.encode()
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
    src_types = {}
    for k, t, v in read_src_kvs():
        src_types[k] = t
        if k.startswith("tokenizer.ggml.") or k == "general.name":
            kv.append((k, t, v))

    def add(k, v):
        t = src_types.get(k) or TYPE_IDS["f32" if isinstance(v, float) else "u32"]
        kv.append((k, t, v))

    kv.append(("general.architecture", TYPE_IDS["str"], b"qwen35"))
    add("qwen35.block_count", N_LAYER)
    add("qwen35.context_length", 2048)
    add("qwen35.embedding_length", N_EMBD)
    add("qwen35.feed_forward_length", N_FF)
    add("qwen35.attention.head_count", N_HEAD)
    add("qwen35.attention.head_count_kv", N_KV)
    add("qwen35.attention.key_length", HEAD_DIM)
    add("qwen35.attention.value_length", HEAD_DIM)
    add("qwen35.attention.layer_norm_rms_epsilon", 1e-6)
    add("qwen35.rope.freq_base", 1e7)
    add("qwen35.rope.dimension_count", N_ROT)
    kv.append(("qwen35.rope.dimension_sections", TYPE_IDS["arr"], (TYPE_IDS["i32"], [11, 11, 10, 0])))
    add("qwen35.full_attention_interval", FULL_ATTN_INTERVAL)
    add("qwen35.ssm.conv_kernel", CONV_K)
    add("qwen35.ssm.state_size", D_STATE)
    add("qwen35.ssm.group_count", N_GROUP)
    add("qwen35.ssm.time_step_rank", DT_RANK)
    add("qwen35.ssm.inner_size", D_INNER)
    kv.append(("general.alignment", TYPE_IDS["u32"], 32))

    tensors = []

    def t(name, dims, data, ty=0):
        tensors.append((name, dims, data, ty))

    t("token_embd.weight", [N_EMBD, VOCAB], q8_0(N_EMBD * VOCAB), 8)
    t("output.weight", [N_EMBD, VOCAB], q8_0(N_EMBD * VOCAB), 8)
    t("output_norm.weight", [N_EMBD], near1(N_EMBD))

    for il in range(N_LAYER):
        p = f"blk.{il}."
        t(p + "attn_norm.weight", [N_EMBD], near1(N_EMBD))
        t(p + "post_attention_norm.weight", [N_EMBD], near1(N_EMBD))
        t(p + "ffn_gate.weight", [N_EMBD, N_FF], q8_0(N_EMBD * N_FF), 8)
        t(p + "ffn_up.weight", [N_EMBD, N_FF], q8_0(N_EMBD * N_FF), 8)
        t(p + "ffn_down.weight", [N_FF, N_EMBD], q8_0(N_FF * N_EMBD), 8)
        if il % FULL_ATTN_INTERVAL == FULL_ATTN_INTERVAL - 1:
            t(p + "attn_q.weight", [N_EMBD, N_HEAD * HEAD_DIM * 2], q8_0(N_EMBD * N_HEAD * HEAD_DIM * 2), 8)
            t(p + "attn_k.weight", [N_EMBD, N_KV * HEAD_DIM], q8_0(N_EMBD * N_KV * HEAD_DIM), 8)
            t(p + "attn_v.weight", [N_EMBD, N_KV * HEAD_DIM], q8_0(N_EMBD * N_KV * HEAD_DIM), 8)
            t(p + "attn_output.weight", [N_HEAD * HEAD_DIM, N_EMBD], q8_0(N_HEAD * HEAD_DIM * N_EMBD), 8)
            t(p + "attn_q_norm.weight", [HEAD_DIM], near1(HEAD_DIM))
            t(p + "attn_k_norm.weight", [HEAD_DIM], near1(HEAD_DIM))
        else:
            t(p + "attn_qkv.weight", [N_EMBD, CONV_CH], q8_0(N_EMBD * CONV_CH), 8)
            t(p + "attn_gate.weight", [N_EMBD, D_INNER], q8_0(N_EMBD * D_INNER), 8)
            t(p + "ssm_beta.weight", [N_EMBD, DT_RANK], q8_0(N_EMBD * DT_RANK), 8)
            t(p + "ssm_alpha.weight", [N_EMBD, DT_RANK], q8_0(N_EMBD * DT_RANK), 8)
            t(p + "ssm_a", [DT_RANK], (-np.abs(rng.uniform(0.1, 1.0, DT_RANK))).astype("<f4").tobytes())
            t(p + "ssm_dt.bias", [DT_RANK], rng.uniform(-0.1, 0.1, DT_RANK).astype("<f4").tobytes())
            t(p + "ssm_conv1d.weight", [CONV_K, CONV_CH], randn(CONV_K * CONV_CH, 0.002))
            t(p + "ssm_norm.weight", [D_STATE], near1(D_STATE))
            t(p + "ssm_out.weight", [D_INNER, N_EMBD], q8_0(D_INNER * N_EMBD), 8)
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
    for _, _, data, _ in tensors:
        off += (-off) % align
        offsets.append(off)
        off += len(data)
    for (name, dims, _, ty), o in zip(tensors, offsets):
        out.s(name)
        out.u32(len(dims))
        for d in dims:
            out.u64(d)
        out.u32(ty)
        out.u64(o)

    data_start = (len(out.b) + align - 1) // align * align
    for (_, _, data, _), o in zip(tensors, offsets):
        target = data_start + o
        while len(out.b) < target:
            out.b += b"\0"
        out.raw(data)

    open(DST, "wb").write(bytes(out.b))
    print(f"wrote {DST}: {len(out.b)/1e9:.2f} GB, {len(tensors)} tensors")


main()
