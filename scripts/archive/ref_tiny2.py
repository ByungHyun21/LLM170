#!/usr/bin/env python3
"""소형 qwen35 GGUF의 토큰 순차(AR) 참조 구현 — llm170 청크 경로 대조용.

토큰을 하나씩 밀면서 상태(KV/GDN/conv)를 굴린다 — 청크 수학과 독립.
마지막 토큰의 레이어별 xs[:4] + logits top-5 출력.
"""
import math
import struct
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/qwen35-gdn.gguf"
TOKENS = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "760").split(",")]


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
            assert n < (1 << 20), n
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
        self.dims = {}
        for _ in range(n_t):
            name = rstr()
            nd = struct.unpack("<I", self.f.read(4))[0]
            dims = struct.unpack("<" + "Q" * nd, self.f.read(8 * nd))
            ty = struct.unpack("<I", self.f.read(4))[0]
            off = struct.unpack("<Q", self.f.read(8))[0]
            self.tensors[name] = off
            self.dims[name] = dims
        self.data_off = (self.f.tell() + 31) // 32 * 32
        self.cache = {}

    def t(self, name):
        if name in self.cache:
            return self.cache[name]
        dims = self.dims[name]
        n = 1
        for d in dims:
            n *= d
        self.f.seek(self.data_off + self.tensors[name])
        vals = struct.unpack(f"<{n}f", self.f.read(4 * n))
        k = dims[0]
        rows = n // k
        out = [list(vals[r * k:(r + 1) * k]) for r in range(rows)]
        if len(dims) == 1:
            out = out[0]
        self.cache[name] = (out, dims)
        return out, dims


g = G(PATH)
u = lambda suf: g.kv[f"qwen35.{suf}"]
N_EMBD = u("embedding_length")
N_LAYER = min(u("block_count"), 64)
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
KQ = 1.0 / math.sqrt(HD)
K_LEN = N_GROUP * D_STATE
V_LEN = DT_RANK * D_STATE
is_recr = lambda il: il % INTERVAL != INTERVAL - 1

W = {}  # 무게 캐시 (이름 접근)


def tw(name):
    if name not in W:
        W[name] = g.t(name)[0]
    return W[name]


def rms(x, w):
    s = sum(v * v for v in x) / len(x)
    scale = 1.0 / math.sqrt(s + EPS)
    return [v * scale * w[i] for i, v in enumerate(x)]


def l2n(x):
    s = math.sqrt(sum(v * v for v in x))
    return [v / max(s, EPS) for v in x]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def silu(x):
    return x / (1.0 + math.exp(-x))


def softplus(x):
    return x if x > 20 else math.log1p(math.exp(x))


def mm(w, x):
    return [sum(a * b for a, b in zip(row, x)) for row in w]


def rope(head, pos):
    half = N_ROT // 2
    out = list(head)
    for p in range(half):
        theta = BASE ** (-(2.0 * p) / N_ROT)
        ang = pos * theta
        c, s = math.cos(ang), math.sin(ang)
        x0, x1 = out[p], out[p + half]
        out[p] = x0 * c - x1 * s
        out[p + half] = x0 * s + x1 * c
    return out


embd = tw("token_embd.weight")
gdn_states = {}   # (il, h) → [[d][d]]
conv_states = {}  # il → [conv_k-1][conv_ch]
kv_cache = {}     # (il) → list of (k_heads, v_heads)
pos = 0

for TOKEN in TOKENS:
    x = embd[TOKEN]
    for il in range(N_LAYER):
        p = f"blk.{il}."
        cur = rms(x, tw(p + "attn_norm.weight"))

        if is_recr(il):
            qkv = mm(tw(p + "attn_qkv.weight"), cur)
            z = mm(tw(p + "attn_gate.weight"), cur)
            beta = [sigmoid(v) for v in mm(tw(p + "ssm_beta.weight"), cur)]
            alpha = mm(tw(p + "ssm_alpha.weight"), cur)
            ssm_a = tw(p + "ssm_a")
            dtb = tw(p + "ssm_dt.bias")
            gg = [softplus(alpha[h] + dtb[h]) * ssm_a[h] for h in range(DT_RANK)]
            convw = tw(p + "ssm_conv1d.weight")
            cs = conv_states.setdefault(il, [[0.0] * CONV_CH for _ in range(CONV_K - 1)])
            convout = []
            for c in range(CONV_CH):
                s = sum(convw[c][j] * cs[j][c] for j in range(CONV_K - 1))
                s += convw[c][CONV_K - 1] * qkv[c]
                convout.append(silu(s))
            for j in range(CONV_K - 2):
                cs[j] = cs[j + 1]
            cs[CONV_K - 2] = list(qkv)
            qh = [l2n(convout[h * D_STATE:(h + 1) * D_STATE]) for h in range(N_GROUP)]
            kh = [l2n(convout[K_LEN + h * D_STATE:K_LEN + (h + 1) * D_STATE]) for h in range(N_GROUP)]
            vh = [convout[2 * K_LEN + h * D_STATE:2 * K_LEN + (h + 1) * D_STATE] for h in range(DT_RANK)]

            out_core = []
            for h in range(DT_RANK):
                st = gdn_states.setdefault((il, h), [[0.0] * D_STATE for _ in range(D_STATE)])
                kvec, qvec = kh[h % N_GROUP], qh[h % N_GROUP]
                gexp = math.exp(gg[h])
                sk = [0.0] * D_STATE
                for kdim in range(D_STATE):
                    for dv in range(D_STATE):
                        st[kdim][dv] *= gexp
                        sk[dv] += st[kdim][dv] * kvec[kdim]
                for dv in range(D_STATE):
                    d = (vh[h][dv] - sk[dv]) * beta[h]
                    for kdim in range(D_STATE):
                        st[kdim][dv] += kvec[kdim] * d
                scale = 1.0 / math.sqrt(D_STATE)
                out_core.append([sum(st[kdim][dv] * qvec[kdim] * scale for kdim in range(D_STATE))
                                 for dv in range(D_STATE)])
            gated = []
            ssnorm = tw(p + "ssm_norm.weight")
            for h in range(DT_RANK):
                n = rms(out_core[h], ssnorm)
                zv = z[h * D_STATE:(h + 1) * D_STATE]
                gated += [n[i] * silu(zv[i]) for i in range(D_STATE)]
            attn_out = mm(tw(p + "ssm_out.weight"), gated)
        else:
            qg = mm(tw(p + "attn_q.weight"), cur)
            kk = mm(tw(p + "attn_k.weight"), cur)
            vv = mm(tw(p + "attn_v.weight"), cur)
            qn, kn = tw(p + "attn_q_norm.weight"), tw(p + "attn_k_norm.weight")
            cache = kv_cache.setdefault(il, [])
            k_heads = [rope(rms(kk[h * HD:(h + 1) * HD], kn), pos) for h in range(N_KV)]
            v_heads = [vv[h * HD:(h + 1) * HD] for h in range(N_KV)]
            cache.append((k_heads, v_heads))
            flat = []
            for h in range(N_HEAD):
                qh = rope(rms(qg[h * 2 * HD:h * 2 * HD + HD], qn), pos)
                kvh = h // (N_HEAD // N_KV)
                scores = []
                for (kh_list, _) in cache:
                    d = sum(qh[i] * kh_list[kvh][i] for i in range(HD)) * KQ
                    scores.append(d)
                mx = max(scores)
                ex = [math.exp(s - mx) for s in scores]
                tot = sum(ex)
                acc = [0.0] * HD
                for pi, (_, vh_list) in enumerate(cache):
                    wgt = ex[pi] / tot
                    for i in range(HD):
                        acc[i] += wgt * vh_list[kvh][i]
                gate = qg[h * 2 * HD + HD:h * 2 * HD + 2 * HD]
                flat += [acc[i] * sigmoid(gate[i]) for i in range(HD)]
            attn_out = mm(tw(p + "attn_output.weight"), flat)

        x = [x[i] + attn_out[i] for i in range(N_EMBD)]
        ffn_res = list(x)
        nrm = rms(x, tw(p + "post_attention_norm.weight"))
        gy = mm(tw(p + "ffn_gate.weight"), nrm)
        uy = mm(tw(p + "ffn_up.weight"), nrm)
        sil = [silu(gy[i]) * uy[i] for i in range(N_FF)]
        fd = mm(tw(p + "ffn_down.weight"), sil)
        x = [ffn_res[i] + fd[i] for i in range(N_EMBD)]
        if TOKEN == TOKENS[-1]:
            print(f"layer {il:>2} recr={is_recr(il)} max|x|={max(abs(v) for v in x):.4f} "
                  f"head={','.join(f'{v:.5f}' for v in x[:4])}")
    pos += 1

h = rms(x, tw("output_norm.weight"))
logits = mm(tw("output.weight"), h)
top = sorted(range(len(logits)), key=lambda i: -logits[i])[:5]
print("topk:", " ".join(f"{i}:{logits[i]:.4f}" for i in top))
