#!/usr/bin/env python3
"""실 Qwen3.8-27B의 다층 순차(AR) numpy 참조 — llm170 층별 대조.

범용 GGUF 리더 + 전 타입 디양자화(q4_K/q5_K/q6_K/q8_0/q3_K/iq4_xs/iq4_nl/f32).
출력: 층별 xs max/head — `LLM170_DEBUG_LAYERS=1` 출력과 대조.
사용: ref_real.py <model.gguf> <tokens csv> [n_layers]
"""
import struct
import sys

import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else "/home/yoon/local_llm/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf"
TOKENS = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "760").split(",")]
N_LAYER_LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 8

IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32)


class G:
    def __init__(self, path):
        self.f = open(path, "rb")
        assert self.f.read(4) == b"GGUF"
        self.f.read(4)
        n_t, n_kv = struct.unpack("<QQ", self.f.read(16))
        SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
               7: "<B", 10: "<Q", 11: "<q", 12: "<d"}

        def rstr():
            n = struct.unpack("<Q", self.f.read(8))[0]
            return self.f.read(n).decode()

        def rval(t):
            if t == 8:
                n = struct.unpack("<Q", self.f.read(8))[0]
                return self.f.read(n)
            if t == 9:
                et = struct.unpack("<I", self.f.read(4))[0]
                n = struct.unpack("<Q", self.f.read(8))[0]
                return [rval(et) for _ in range(n)]
            return struct.unpack(FMT[t], self.f.read(SZ[t]))[0]

        self.kv = {}
        for _ in range(n_kv):
            k = rstr()
            t = struct.unpack("<I", self.f.read(4))[0]
            self.kv[k] = rval(t)
        self.tensors = {}
        for _ in range(n_t):
            name = rstr()
            nd = struct.unpack("<I", self.f.read(4))[0]
            dims = struct.unpack("<" + "Q" * nd, self.f.read(8 * nd))
            ty = struct.unpack("<I", self.f.read(4))[0]
            off = struct.unpack("<Q", self.f.read(8))[0]
            self.tensors[name] = (dims, ty, off)
        self.data_off = (self.f.tell() + 31) // 32 * 32
        self.cache = {}
        self.dtype_names = {0: "f32", 1: "f16", 8: "q8_0", 12: "q4_K", 14: "q6_K",
                            10: "q2_K", 11: "q3_K", 13: "q5_K", 23: "iq4_xs", 20: "iq4_nl", 30: "bf16"}

    def get(self, name):
        if name in self.cache:
            return self.cache[name]
        dims, ty, off = self.tensors[name]
        name_ty = self.dtype_names.get(ty, str(ty))
        n = int(np.prod(dims))
        self.f.seek(self.data_off + off)
        a = dequant(self.f, name_ty, n, dims)
        self.cache[name] = a
        return a


def dequant(f, ty, n, dims):
    if ty == "f32":
        a = np.frombuffer(f.read(4 * n), dtype="<f4").copy()
    elif ty == "f16":
        a = np.frombuffer(f.read(2 * n), dtype="<f2").copy().astype(np.float32)
    elif ty == "bf16":
        raw = np.frombuffer(f.read(2 * n), dtype="<u2").copy()
        a = (raw.astype(np.uint32) << 16).view("<f4").copy()
    elif ty == "q8_0":
        raw = np.frombuffer(f.read((n // 32) * 34), dtype=np.uint8).reshape(-1, 34).copy()
        d = raw[:, :2].view("<f2").astype(np.float32).reshape(-1)
        q = raw[:, 2:].view(np.int8).astype(np.float32)
        a = (q * d[:, None]).reshape(-1)
    elif ty in ("q4_K", "q5_K"):
        bsz = 144 if ty == "q4_K" else 176
        raw = np.frombuffer(f.read((n // 256) * bsz), dtype=np.uint8).reshape(-1, bsz).copy()
        d = raw[:, 0:2].view("<f2").astype(np.float32).reshape(-1)
        mn = raw[:, 2:4].view("<f2").astype(np.float32).reshape(-1)
        sc = raw[:, 4:16].astype(np.int64)
        out = np.zeros((raw.shape[0], 256), dtype=np.float32)
        if ty == "q5_K":
            qh = raw[:, 16:48].astype(np.int64)
            ql = raw[:, 48:176].astype(np.int64)
        else:
            qs = raw[:, 16:144].astype(np.int64)
        u1 = np.array([1 << (2 * g) for g in range(4)], dtype=np.int64)
        u2 = np.array([2 << (2 * g) for g in range(4)], dtype=np.int64)
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
            if ty == "q4_K":
                q = qs[:, grp * 32:(grp + 1) * 32]
                out[:, grp * 64:grp * 64 + 32] = d[:, None] * s1[:, None] * (q & 0xF) - mn[:, None] * m1[:, None]
                out[:, grp * 64 + 32:grp * 64 + 64] = d[:, None] * s2[:, None] * (q >> 4) - mn[:, None] * m2[:, None]
            else:
                q = ql[:, grp * 32:(grp + 1) * 32]
                h = qh[:, :32]
                lo = (q & 0xF) + ((h & u1[grp]) != 0) * 16
                hi = (q >> 4) + ((h & u2[grp]) != 0) * 16
                out[:, grp * 64:grp * 64 + 32] = d[:, None] * s1[:, None] * lo - mn[:, None] * m1[:, None]
                out[:, grp * 64 + 32:grp * 64 + 64] = d[:, None] * s2[:, None] * hi - mn[:, None] * m2[:, None]
        a = out.reshape(-1)
    elif ty == "q6_K":
        raw = np.frombuffer(f.read((n // 256) * 210), dtype=np.uint8).reshape(-1, 210).copy()
        d = raw[:, 208:210].view("<f2").astype(np.float32).reshape(-1)
        ql = raw[:, 0:128].astype(np.int64)
        qh = raw[:, 128:192].astype(np.int64)
        scb = raw[:, 192:208].astype(np.int64)
        sc = np.where(scb > 127, scb - 256, scb).astype(np.float32)
        out = np.zeros((raw.shape[0], 256), dtype=np.float32)
        for nb2 in range(2):
            qlb = ql[:, nb2 * 64:(nb2 + 1) * 64]   # ql advances 64 per n
            qhb = qh[:, nb2 * 32:(nb2 + 1) * 32]   # qh advances 32 per n
            scb = sc[:, nb2 * 8:(nb2 + 1) * 8]     # sc advances 8 per n
            for l in range(32):
                iis = l // 16
                q1 = ((qlb[:, l] & 0xF) | ((qhb[:, l] & 3) << 4)) - 32
                q2 = ((qlb[:, l + 32] & 0xF) | (((qhb[:, l] >> 2) & 3) << 4)) - 32
                q3 = ((qlb[:, l] >> 4) | (((qhb[:, l] >> 4) & 3) << 4)) - 32
                q4 = ((qlb[:, l + 32] >> 4) | (((qhb[:, l] >> 6) & 3) << 4)) - 32
                out[:, nb2 * 128 + l] = d * scb[:, iis] * q1
                out[:, nb2 * 128 + l + 32] = d * scb[:, iis + 2] * q2
                out[:, nb2 * 128 + l + 64] = d * scb[:, iis + 4] * q3
                out[:, nb2 * 128 + l + 96] = d * scb[:, iis + 6] * q4
        a = out.reshape(-1)
    elif ty == "iq4_xs":
        raw = np.frombuffer(f.read((n // 256) * 136), dtype=np.uint8).reshape(-1, 136).copy()
        d = raw[:, 0:2].view("<f2").astype(np.float32).reshape(-1)
        sh = raw[:, 2:4].view("<u2").reshape(-1).astype(np.int64)
        sl = raw[:, 4:8].astype(np.int64)
        qs = raw[:, 8:136].astype(np.int64)
        out = np.zeros((raw.shape[0], 256), dtype=np.float32)
        for ib in range(8):
            ls = ((sl[:, ib // 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4)
            dl = d * (ls - 32).astype(np.float32)
            q = qs[:, ib * 16:ib * 16 + 16]
            out[:, ib * 32:ib * 32 + 16] = dl[:, None] * IQ4NL[q & 0xF]
            out[:, ib * 32 + 16:ib * 32 + 32] = dl[:, None] * IQ4NL[q >> 4]
        a = out.reshape(-1)
    elif ty == "iq4_nl":
        raw = np.frombuffer(f.read((n // 32) * 18), dtype=np.uint8).reshape(-1, 18).copy()
        d = raw[:, 0:2].view("<f2").astype(np.float32).reshape(-1)
        qs = raw[:, 2:18].astype(np.int64)
        out = np.zeros((raw.shape[0], 32), dtype=np.float32)
        out[:, :16] = d[:, None] * IQ4NL[qs & 0xF]
        out[:, 16:] = d[:, None] * IQ4NL[qs >> 4]
        a = out.reshape(-1)
    elif ty == "q3_K":
        raw = np.frombuffer(f.read((n // 256) * 110), dtype=np.uint8).reshape(-1, 110).copy()
        d = raw[:, 108:110].view("<f2").astype(np.float32).reshape(-1)
        hm = raw[:, 0:32].astype(np.int64)
        q = raw[:, 32:96].astype(np.int64)
        a0 = raw[:, 96:100].copy().view("<u4").reshape(-1).astype(np.int64)
        a1 = raw[:, 100:104].copy().view("<u4").reshape(-1).astype(np.int64)
        t = raw[:, 104:108].copy().view("<u4").reshape(-1).astype(np.int64)
        km1, km2 = np.int64(0x03030303), np.int64(0x0F0F0F0F)
        x0 = (a0 & km2) | ((t & km1) << 4)
        x1 = (a1 & km2) | (((t >> 2) & km1) << 4)
        x2 = ((a0 >> 4) & km2) | (((t >> 4) & km1) << 4)
        x3 = ((a1 >> 4) & km2) | (((t >> 6) & km1) << 4)

        def v2b(v):
            return np.stack([(v) & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF], axis=1)

        scb = np.concatenate([v2b(x0), v2b(x1), v2b(x2), v2b(x3)], axis=1)
        sc = np.where(scb > 127, scb - 256, scb)
        out = np.zeros((raw.shape[0], 256), dtype=np.float32)
        for nb2 in range(2):
            m = 1
            for ji, shift in enumerate((0, 2, 4, 6)):
                for half in (0, 1):
                    idx = nb2 * 8 + ji * 2 + half
                    dl = d * (sc[:, idx].astype(np.float32) - 32.0)
                    for l in range(16):
                        qv = (q[:, nb2 * 32 + half * 16 + l] >> shift) & 3
                        sub = np.where((hm[:, half * 16 + l] & m) != 0, 0, 4)
                        out[:, nb2 * 128 + ji * 32 + half * 16 + l] = dl * (qv - sub)
                m <<= 1
        a = out.reshape(-1)
    else:
        raise SystemExit(f"unsupported type {ty}")
    return a.reshape(dims[::-1]).astype(np.float32)  # [ne1, ne0] 행=row


g = G(PATH)
u = lambda suf: g.kv[f"qwen35.{suf}"]
N_EMBD = u("embedding_length")
N_LAYER = min(u("block_count"), 64, N_LAYER_LIMIT)
N_FF = u("feed_forward_length")
N_HEAD = u("attention.head_count")
N_KV = u("attention.head_count_kv")
HD = u("attention.key_length")
N_ROT = u("rope.dimension_count")
BASE = u("rope.freq_base")
EPS = u("attention.layer_norm_rms_epsilon")
INTERVAL = u("full_attention_interval")
D_STATE = u("ssm.state_size")
N_GROUP = u("ssm.group_count")
DT_RANK = u("ssm.time_step_rank")
D_INNER = u("ssm.inner_size")
CONV_K = u("ssm.conv_kernel")
CONV_CH = D_INNER + 2 * N_GROUP * D_STATE
K_LEN = N_GROUP * D_STATE
V_LEN = DT_RANK * D_STATE
KQ = 1.0 / np.sqrt(HD)
is_recr = lambda il: il % INTERVAL != INTERVAL - 1
DBG_LAYER = int(sys.argv[4]) if len(sys.argv) > 4 else 1


def rms(x, w):
    s = np.mean(x.astype(np.float64) ** 2)
    return (x / np.sqrt(s + EPS) * w).astype(np.float32)


def l2n(x):
    n = np.sqrt(np.sum(x.astype(np.float64) ** 2))
    return (x / max(n, EPS)).astype(np.float32)


sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
silu = lambda x: x / (1.0 + np.exp(-x))
softplus = lambda x: np.where(x > 20, x, np.log1p(np.exp(x)))
half = N_ROT // 2
inv = BASE ** (-2.0 * np.arange(half) / N_ROT)


def rope(head, pos):
    ang = pos * inv
    c, s = np.cos(ang), np.sin(ang)
    out = head.copy()
    a, b = out[:half], out[half:N_ROT]
    out[:half] = a * c - b * s
    out[half:N_ROT] = a * s + b * c
    return out.astype(np.float32)


gdn_states = {}
conv_states = {}
kv_cache = {}
pos = 0

for TOKEN in TOKENS:
    x = g.get("token_embd.weight")[TOKEN].astype(np.float32)
    for il in range(N_LAYER):
        p = f"blk.{il}."
        cur = rms(x, g.get(p + "attn_norm.weight"))

        if is_recr(il):
            qkv = g.get(p + "attn_qkv.weight") @ cur
            z = g.get(p + "attn_gate.weight") @ cur
            if il == DBG_LAYER and TOKEN == TOKENS[-1]:
                print(f"  py L{il} cur max={np.abs(cur).max():.5f} qkv max={np.abs(qkv).max():.5f} z max={np.abs(z).max():.5f}")
                print(f"  py L{il} cur[:4]={[f'{v:.5f}' for v in cur[:4]]}")
            beta = sigmoid(g.get(p + "ssm_beta.weight") @ cur)
            gg = softplus(g.get(p + "ssm_alpha.weight") @ cur + g.get(p + "ssm_dt.bias")) * g.get(p + "ssm_a")
            convw = g.get(p + "ssm_conv1d.weight")
            cs = conv_states.setdefault(il, np.zeros((CONV_K - 1, CONV_CH), dtype=np.float32))
            s = np.einsum("jc,cj->c", cs, convw[:, :CONV_K - 1]) + convw[:, CONV_K - 1] * qkv
            convout = silu(s)
            cs[:-1] = cs[1:]
            cs[-1] = qkv
            qh = [l2n(convout[h * D_STATE:(h + 1) * D_STATE]) for h in range(N_GROUP)]
            kh = [l2n(convout[K_LEN + h * D_STATE:K_LEN + (h + 1) * D_STATE]) for h in range(N_GROUP)]
            vh = [convout[2 * K_LEN + h * D_STATE:2 * K_LEN + (h + 1) * D_STATE] for h in range(DT_RANK)]
            out_core = []
            for h in range(DT_RANK):
                st = gdn_states.setdefault((il, h), np.zeros((D_STATE, D_STATE), dtype=np.float32))
                kvec, qvec = kh[h % N_GROUP], qh[h % N_GROUP]
                st *= np.exp(gg[h])
                sk = st.T @ kvec
                d = (vh[h] - sk) * beta[h]
                st += np.outer(kvec, d)
                out_core.append(st.T @ qvec / np.sqrt(D_STATE))
            ssnorm = g.get(p + "ssm_norm.weight")
            gated = np.concatenate([rms(out_core[h], ssnorm) * silu(z[h * D_STATE:(h + 1) * D_STATE])
                                    for h in range(DT_RANK)])
            if il == DBG_LAYER and TOKEN == TOKENS[-1]:
                print(f"  py L{il} conv max={np.abs(convout).max():.5f} core max={max(np.abs(o).max() for o in out_core):.7f} gated max={np.abs(gated).max():.5f} h1={[f'{v:.6f}' for v in gated[16:20]]} h47={[f'{v:.6f}' for v in gated[-4:]]}")
            attn_out = g.get(p + "ssm_out.weight") @ gated
            if il == DBG_LAYER and TOKEN == TOKENS[-1]:
                mi = np.unravel_index(np.abs(attn_out).argmax(), attn_out.shape)[0]
                print(f"  py L{il} ssm_out max={np.abs(attn_out).max():.5f} @row{mi} out[:4]={[f'{v:.6f}' for v in attn_out[:4]]} out[{mi}][:3]={[f'{v:.6f}' for v in attn_out[mi:mi+3]]}")
        else:
            qg = g.get(p + "attn_q.weight") @ cur
            kk = g.get(p + "attn_k.weight") @ cur
            vv = g.get(p + "attn_v.weight") @ cur
            qn, kn = g.get(p + "attn_q_norm.weight"), g.get(p + "attn_k_norm.weight")
            cache = kv_cache.setdefault(il, [])
            k_heads = [rope(rms(kk[h * HD:(h + 1) * HD], kn), pos) for h in range(N_KV)]
            v_heads = [vv[h * HD:(h + 1) * HD] for h in range(N_KV)]
            cache.append((k_heads, v_heads))
            flat = np.zeros(N_HEAD * HD, dtype=np.float32)
            for h in range(N_HEAD):
                qh2 = rope(rms(qg[h * 2 * HD:h * 2 * HD + HD], qn), pos)
                kvh = h // (N_HEAD // N_KV)
                scores = np.array([qh2 @ kh_[kvh] for kh_, _ in cache]) * KQ
                scores = np.exp(scores - scores.max())
                scores /= scores.sum()
                acc = np.zeros(HD, dtype=np.float32)
                for pi, (_, vh_) in enumerate(cache):
                    acc += scores[pi] * vh_[kvh]
                flat[h * HD:(h + 1) * HD] = acc * sigmoid(qg[h * 2 * HD + HD:h * 2 * HD + 2 * HD])
            attn_out = g.get(p + "attn_output.weight") @ flat

        x = x + attn_out
        ffn_res = x.copy()
        nrm = rms(x, g.get(p + "post_attention_norm.weight"))
        gy = g.get(p + "ffn_gate.weight") @ nrm
        uy = g.get(p + "ffn_up.weight") @ nrm
        fd = g.get(p + "ffn_down.weight") @ (silu(gy) * uy)
        x = ffn_res + fd
        if TOKEN == TOKENS[-1]:
            print(f"layer {il:>2} recr={is_recr(il)} max|x|={np.abs(x).max():.4f} "
                  f"head={','.join(f'{v:.5f}' for v in x[:4])}")
    pos += 1
