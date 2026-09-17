#!/usr/bin/env python3
"""실차원 mid 모델(무작위 가중치)의 numpy 독립 참조 — llm170 대조용.

q8_0 역양자화 + qwen35 2층(GDN+attn) 순차 포워드. 토큰 순차(AR) 의미론.
사용: ref_mid.py <model.gguf> <tokens csv>
"""
import struct
import sys

import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/qwen35-mid2.gguf"
TOKENS = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "760").split(",")]


def load_gguf(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    f.read(4)
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
            return [rval(et) for _ in range(n)]
        return struct.unpack(FMT[t], f.read(SZ[t]))[0]

    kv = {}
    for _ in range(n_kv):
        k = rstr()
        t = struct.unpack("<I", f.read(4))[0]
        kv[k] = rval(t)
    tensors = {}
    for _ in range(n_t):
        name = rstr()
        nd = struct.unpack("<I", f.read(4))[0]
        dims = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
        ty = struct.unpack("<I", f.read(4))[0]
        off = struct.unpack("<Q", f.read(8))[0]
        tensors[name] = (dims, ty, off)
    data_off = (f.tell() + 31) // 32 * 32

    cache = {}

    def get(name):
        if name in cache:
            return cache[name]
        dims, ty, off = tensors[name]
        n = int(np.prod(dims))
        f.seek(data_off + off)
        if ty == 0:  # f32
            a = np.frombuffer(f.read(4 * n), dtype="<f4")
        elif ty == 8:  # q8_0
            raw = np.frombuffer(f.read(n * 34 // 32 * 34 if False else (n // 32) * 34), dtype=np.uint8).reshape(-1, 34)
            d = raw[:, :2].copy().view("<f2").astype(np.float32).reshape(-1)
            q = raw[:, 2:].copy().view(np.int8).astype(np.float32)
            a = (q * d[:, None]).reshape(-1)
        else:
            raise SystemExit(f"type {ty} not supported")
        a = a.reshape(dims[::-1])  # ne1-major → numpy [row, k]
        cache[name] = a
        return a

    return kv, get


kv, get = load_gguf(PATH)
u = lambda suf: kv[f"qwen35.{suf}"]
N_EMBD = u("embedding_length")
N_LAYER = u("block_count")
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

embd = get("token_embd.weight")  # [vocab, n_embd] (row-major [ne1, ne0])


def rms(x, w):
    s = np.mean(x.astype(np.float64) ** 2)
    return x / np.sqrt(s + EPS) * w


def l2n(x):
    n = np.sqrt(np.sum(x.astype(np.float64) ** 2))
    return x / max(n, EPS)


sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
silu = lambda x: x / (1.0 + np.exp(-x))
softplus = lambda x: np.where(x > 20, x, np.log1p(np.exp(x)))

# 사전 계산: rope 코사인
half = N_ROT // 2
inv = BASE ** (-2.0 * np.arange(half) / N_ROT)
pos_arr = np.array([], dtype=np.float64)


def rope(head, pos):
    ang = pos * inv
    c, s = np.cos(ang), np.sin(ang)
    out = head.copy()
    a, b = out[:half], out[half:N_ROT]
    out[:half] = a * c - b * s
    out[half:N_ROT] = a * s + b * c
    return out


gdn_states = {}
conv_states = {}
kv_cache = {}
pos = 0

for TOKEN in TOKENS:
    x = embd[TOKEN].astype(np.float32)
    for il in range(N_LAYER):
        p = f"blk.{il}."
        cur = rms(x, get(p + "attn_norm.weight"))

        if is_recr(il):
            wqkv = get(p + "attn_qkv.weight")      # [conv_ch, n_embd]
            wgate = get(p + "attn_gate.weight")    # [d_inner, n_embd]
            wbeta = get(p + "ssm_beta.weight")     # [dt_rank, n_embd]
            walpha = get(p + "ssm_alpha.weight")
            ssm_a = get(p + "ssm_a")
            dtb = get(p + "ssm_dt.bias")
            convw = get(p + "ssm_conv1d.weight")   # [conv_ch, conv_k]
            ssnorm = get(p + "ssm_norm.weight")
            wout = get(p + "ssm_out.weight")       # [n_embd, d_inner]

            qkv = wqkv @ cur
            z = wgate @ cur
            beta = sigmoid(wbeta @ cur)
            g = softplus(walpha @ cur + dtb) * ssm_a
            cs = conv_states.setdefault(il, np.zeros((CONV_K - 1, CONV_CH), dtype=np.float32))
            # conv: out[c] = Σ_j w[c,j]·cs[j,c] + w[c,K-1]·qkv[c]
            s = np.einsum("jc,cj->c", cs, convw[:, :CONV_K - 1]) + convw[:, CONV_K - 1] * qkv
            convout = silu(s)
            cs[:-1] = cs[1:]
            cs[-1] = qkv

            qh = np.stack([l2n(convout[h * D_STATE:(h + 1) * D_STATE]) for h in range(N_GROUP)])
            kh = np.stack([l2n(convout[K_LEN + h * D_STATE:K_LEN + (h + 1) * D_STATE]) for h in range(N_GROUP)])
            vh = np.stack([convout[2 * K_LEN + h * D_STATE:2 * K_LEN + (h + 1) * D_STATE] for h in range(DT_RANK)])

            out_core = []
            for h in range(DT_RANK):
                st = gdn_states.setdefault((il, h), np.zeros((D_STATE, D_STATE), dtype=np.float32))
                kvec, qvec = kh[h % N_GROUP], qh[h % N_GROUP]
                st *= np.exp(g[h])
                sk = st.T @ kvec  # [dv] = Σ_kdim S[kdim,dv]·k[kdim]
                d = (vh[h] - sk) * beta[h]
                st += np.outer(kvec, d)
                out_core.append(st.T @ qvec / np.sqrt(D_STATE))
            n = np.stack([rms(out_core[h], ssnorm) for h in range(DT_RANK)])
            gated = (n * silu(z.reshape(DT_RANK, D_STATE))).reshape(-1)
            attn_out = wout @ gated
        else:
            wq = get(p + "attn_q.weight")      # [n_head*2*hd, n_embd]
            wk = get(p + "attn_k.weight")
            wv = get(p + "attn_v.weight")
            wo = get(p + "attn_output.weight")  # [n_embd, n_head*hd]
            qn = get(p + "attn_q_norm.weight")
            kn = get(p + "attn_k_norm.weight")
            qg = wq @ cur
            kk = wk @ cur
            vv = wv @ cur
            cache = kv_cache.setdefault(il, [])
            k_heads = [rope(rms(kk[h * HD:(h + 1) * HD], kn), pos) for h in range(N_KV)]
            v_heads = [vv[h * HD:(h + 1) * HD] for h in range(N_KV)]
            cache.append((k_heads, v_heads))
            flat = np.zeros(N_HEAD * HD, dtype=np.float32)
            for h in range(N_HEAD):
                qh = rope(rms(qg[h * 2 * HD:h * 2 * HD + HD], qn), pos)
                kvh = h // (N_HEAD // N_KV)
                scores = np.array([qh @ kh_[kvh] for kh_, _ in cache]) * KQ
                scores = np.exp(scores - scores.max())
                scores /= scores.sum()
                acc = np.zeros(HD, dtype=np.float32)
                for pi, (_, vh_) in enumerate(cache):
                    acc += scores[pi] * vh_[kvh]
                flat[h * HD:(h + 1) * HD] = acc * sigmoid(qg[h * 2 * HD + HD:h * 2 * HD + 2 * HD])
            attn_out = wo @ flat

        x = x + attn_out
        ffn_res = x.copy()
        nrm = rms(x, get(p + "post_attention_norm.weight"))
        gy = get(p + "ffn_gate.weight") @ nrm
        uy = get(p + "ffn_up.weight") @ nrm
        fd = get(p + "ffn_down.weight") @ (silu(gy) * uy)
        x = ffn_res + fd
        if TOKEN == TOKENS[-1]:
            print(f"layer {il:>2} recr={is_recr(il)} max|x|={np.abs(x).max():.4f} "
                  f"head={','.join(f'{v:.5f}' for v in x[:4])}")
    pos += 1

h = rms(x, get("output_norm.weight"))
logits = get("output.weight") @ h
top = np.argsort(-logits)[:5]
print("topk:", " ".join(f"{int(i)}:{logits[i]:.4f}" for i in top))
