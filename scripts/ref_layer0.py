#!/usr/bin/env python3
"""실 Qwen3.8-27B layer 0(GDN) 단층 numpy 참조 — llm170 스테이지 대조.

q5_K/q8_0/f32 역양자화 + GDN층 1토큰 포워드. 스테이지별 max 출력.
"""
import struct
import sys

import numpy as np

M = "/home/yoon/local_llm/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf"
TOKEN = int(sys.argv[1]) if len(sys.argv) > 1 else 760
IL = 0

N_EMBD, N_GROUP, DT_RANK, D_STATE = 5120, 16, 48, 128
D_INNER, CONV_K = 6144, 4
CONV_CH = D_INNER + 2 * N_GROUP * D_STATE
K_LEN = N_GROUP * D_STATE
V_LEN = DT_RANK * D_STATE
EPS = 1e-6

# gguf-dump로 확보한 텐서 오프셋 (데이터섹션 기준) + data_offset
DATA = 10996640
T = {
    "token_embd": (1042964480, "q4_K", [5120, 248320]),
    "attn_norm": (1779752960 + 0, None, None),  # 아래 실제값으로 교체
}
# 정확한 오프셋 (dump 출력 기준, 데이터섹션 기준 off)
OFF = {
    "token_embd": (1042964480, "q4_K"),
    "blk.0.attn_norm.weight": (1779752960, "f32"),
    "blk.0.attn_qkv.weight": (1779773440, "q5_K"),
    "blk.0.attn_gate.weight": (1758126080, "q5_K"),
    "blk.0.ssm_beta.weight": (1949097152, "q8_0"),
    "blk.0.ssm_alpha.weight": (1948836032, "q8_0"),
    "blk.0.ssm_a": (1948835840, "f32"),
    "blk.0.ssm_dt.bias": (1949522112, "f32"),
    "blk.0.ssm_conv1d.weight": (1949358272, "f32"),
    "blk.0.ssm_norm.weight": (1949522304, "f32"),
    "blk.0.ssm_out.weight": (1949522816, "q8_0"),
    "blk.0.post_attention_norm.weight": (1948815360, "f32"),
    "blk.0.ffn_gate.weight": (1863168000, "iq4_xs"),
    "blk.0.ffn_up.weight": (1910517760, "q3_K"),
    "blk.0.ffn_down.weight": (1815818240, "iq4_xs"),
}
DIMS = {
    "token_embd": (5120, 248320),
    "blk.0.attn_norm.weight": (5120,),
    "blk.0.attn_qkv.weight": (5120, 10240),
    "blk.0.attn_gate.weight": (5120, 6144),
    "blk.0.ssm_beta.weight": (5120, 48),
    "blk.0.ssm_alpha.weight": (5120, 48),
    "blk.0.ssm_a": (48,),
    "blk.0.ssm_dt.bias": (48,),
    "blk.0.ssm_conv1d.weight": (4, 10240),
    "blk.0.ssm_norm.weight": (128,),
    "blk.0.ssm_out.weight": (6144, 5120),
    "blk.0.post_attention_norm.weight": (5120,),
    "blk.0.ffn_gate.weight": (5120, 17408),
    "blk.0.ffn_up.weight": (5120, 17408),
    "blk.0.ffn_down.weight": (17408, 5120),
}

f = open(M, "rb")


def f16(b):
    return struct.unpack("<e", b)[0]


def get(name):
    off, ty = OFF[name]
    dims = DIMS[name]
    n = int(np.prod(dims))
    f.seek(DATA + off)
    if ty == "f32":
        a = np.frombuffer(f.read(4 * n), dtype="<f4").copy()
    elif ty == "q8_0":
        raw = np.frombuffer(f.read((n // 32) * 34), dtype=np.uint8).reshape(-1, 34).copy()
        d = raw[:, :2].view("<f2").astype(np.float32).reshape(-1)
        q = raw[:, 2:].view(np.int8).astype(np.float32)
        a = (q * d[:, None]).reshape(-1)
    elif ty == "q4_K":
        raw = np.frombuffer(f.read((n // 256) * 144), dtype=np.uint8).reshape(-1, 144).copy()
        d = raw[:, 0:2].view("<f2").astype(np.float32).reshape(-1)
        mn = raw[:, 2:4].view("<f2").astype(np.float32).reshape(-1)
        sc = raw[:, 4:16].astype(np.int32)
        qs = raw[:, 16:144].astype(np.int32)
        # get_scale_min_k4
        def sm(j, sc):
            r = np.zeros_like(sc[:, 0])
            return r
        out = np.zeros((raw.shape[0], 256), dtype=np.float32)
        for grp in range(4):
            is_ = grp * 2
            if is_ < 4:
                s1 = sc[:, is_] & 63
                m1 = sc[:, is_ + 4] & 63
                s2 = sc[:, is_ + 1] & 63
                m2 = sc[:, is_ + 5] & 63
            else:
                s1 = (sc[:, is_ + 4] & 0xF) | ((sc[:, is_ - 4] >> 6) << 4)
                m1 = (sc[:, is_ + 4] >> 4) | ((sc[:, is_] >> 6) << 4)
                s2 = (sc[:, is_ + 5] & 0xF) | ((sc[:, is_ - 3] >> 6) << 4)
                m2 = (sc[:, is_ + 5] >> 4) | ((sc[:, is_ + 1] >> 6) << 4)
            q = qs[:, grp * 32:(grp + 1) * 32]
            out[:, grp * 64:grp * 64 + 32] = d[:, None] * s1[:, None] * (q & 0xF) - mn[:, None] * m1[:, None]
            out[:, grp * 64 + 32:grp * 64 + 64] = d[:, None] * s2[:, None] * (q >> 4) - mn[:, None] * m2[:, None]
        a = out.reshape(-1)
    elif ty == "q5_K":
        raw = np.frombuffer(f.read((n // 256) * 176), dtype=np.uint8).reshape(-1, 176).copy()
        d = raw[:, 0:2].view("<f2").astype(np.float32).reshape(-1)
        mn = raw[:, 2:4].view("<f2").astype(np.float32).reshape(-1)
        sc = raw[:, 4:16].astype(np.int32)
        qh = raw[:, 16:48].astype(np.int32)
        ql = raw[:, 48:176].astype(np.int32)
        out = np.zeros((raw.shape[0], 256), dtype=np.float32)
        u1 = np.array([[1, 2, 4, 8][g % 4] for g in range(4)])  # u1: 1,4,16,64 (shift 0,2,4,6)
        u1 = np.array([1 << (2 * g) for g in range(4)])
        u2 = np.array([2 << (2 * g) for g in range(4)])
        for grp in range(4):
            is_ = grp * 2
            if is_ < 4:
                s1 = sc[:, is_] & 63
                m1 = sc[:, is_ + 4] & 63
                s2 = sc[:, is_ + 1] & 63
                m2 = sc[:, is_ + 5] & 63
            else:
                s1 = (sc[:, is_ + 4] & 0xF) | ((sc[:, is_ - 4] >> 6) << 4)
                m1 = (sc[:, is_ + 4] >> 4) | ((sc[:, is_] >> 6) << 4)
                s2 = (sc[:, is_ + 5] & 0xF) | ((sc[:, is_ - 3] >> 6) << 4)
                m2 = (sc[:, is_ + 5] >> 4) | ((sc[:, is_ + 1] >> 6) << 4)
            q = ql[:, grp * 32:(grp + 1) * 32]
            h = qh[:, :32]
            lo = (q & 0xF) + ((h & u1[grp]) != 0).astype(np.int32) * 16
            hi = (q >> 4) + ((h & u2[grp]) != 0).astype(np.int32) * 16
            out[:, grp * 64:grp * 64 + 32] = d[:, None] * s1[:, None] * lo - mn[:, None] * m1[:, None]
            out[:, grp * 64 + 32:grp * 64 + 64] = d[:, None] * s2[:, None] * hi - mn[:, None] * m2[:, None]
        a = out.reshape(-1)
    elif ty == "iq4_xs":
        raw = np.frombuffer(f.read((n // 256) * 136), dtype=np.uint8).reshape(-1, 136).copy()
        d = raw[:, 0:2].view("<f2").astype(np.float32).reshape(-1)
        sh = raw[:, 2:4].view("<u2").reshape(-1).astype(np.int32)
        sl = raw[:, 4:8].astype(np.int32)
        qs = raw[:, 8:136].astype(np.int32)
        IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32)
        out = np.zeros((raw.shape[0], 256), dtype=np.float32)
        for ib in range(8):
            ls = ((sl[:, ib // 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4)
            dl = d * (ls - 32).astype(np.float32)
            q = qs[:, ib * 16:ib * 16 + 16]
            out[:, ib * 32:ib * 32 + 16] = dl[:, None] * IQ4NL[q & 0xF]
            out[:, ib * 32 + 16:ib * 32 + 32] = dl[:, None] * IQ4NL[q >> 4]
        a = out.reshape(-1)


def rms(x, w):
    s = np.mean(x.astype(np.float64) ** 2)
    return (x / np.sqrt(s + EPS) * w).astype(np.float32)


def l2n(x):
    n = np.sqrt(np.sum(x.astype(np.float64) ** 2))
    return (x / max(n, EPS)).astype(np.float32)


sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
silu = lambda x: x / (1.0 + np.exp(-x))
softplus = lambda x: np.where(x > 20, x, np.log1p(np.exp(x)))

embd = get("token_embd")           # [vocab, 5120]
x = embd[TOKEN].astype(np.float32)
print(f"x0      max={np.abs(x).max():.5f} head={','.join(f'{v:.5f}' for v in x[:4])}")

cur = rms(x, get("blk.0.attn_norm.weight"))
print(f"cur     max={np.abs(cur).max():.5f}")

wqkv = get("blk.0.attn_qkv.weight")     # [10240, 5120]
qkv = wqkv @ cur
z = get("blk.0.attn_gate.weight") @ cur
print(f"qkv     max={np.abs(qkv).max():.5f}")
print(f"z       max={np.abs(z).max():.5f} head={','.join(f'{v:.5f}' for v in z[:4])}")

beta = sigmoid(get("blk.0.ssm_beta.weight") @ cur)
g = softplus(get("blk.0.ssm_alpha.weight") @ cur + get("blk.0.ssm_dt.bias")) * get("blk.0.ssm_a")
convw = get("blk.0.ssm_conv1d.weight")   # [10240, 4]
convout = silu(convw[:, 3] * qkv)        # 상태 0
print(f"beta[:4]={beta[:4]} g[:4]={g[:4]}")
print(f"conv    max={np.abs(convout).max():.5f}")

qh = [l2n(convout[h * D_STATE:(h + 1) * D_STATE]) for h in range(N_GROUP)]
kh = [l2n(convout[K_LEN + h * D_STATE:K_LEN + (h + 1) * D_STATE]) for h in range(N_GROUP)]
vh = [convout[2 * K_LEN + h * D_STATE:2 * K_LEN + (h + 1) * D_STATE] for h in range(DT_RANK)]

out_core = []
for h in range(DT_RANK):
    st = np.zeros((D_STATE, D_STATE), dtype=np.float32)
attn_out = get("blk.0.ssm_out.weight") @ gated
print(f"ssm_out max={np.abs(attn_out).max():.5f}")

# ---- layer 0 FFN + residual ----
x = x + attn_out
ffn_res = x.copy()
nrm = rms(x, get("blk.0.post_attention_norm.weight"))
gy = get("blk.0.ffn_gate.weight") @ nrm
uy = get("blk.0.ffn_up.weight") @ nrm
fd = get("blk.0.ffn_down.weight") @ (silu(gy) * uy)
x = ffn_res + fd
print(f"L0 out  max={np.abs(x).max():.5f} head={','.join(f'{v:.5f}' for v in x[:4])}")

# ---- layer 1 (GDN) 상태 이월: conv 상태 1토큰, L0 GDN 상태 재사용 불가(재계산) ----
# 간결화: 여기서는 L0까지만 (성장 패턴 확인용)
