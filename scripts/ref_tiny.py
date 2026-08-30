#!/usr/bin/env python3
"""소형 qwen35 GGUF의 1토큰 포워드 독립 참조 구현 (llm170 대조용).

수학 근거: ~/local_llm/llama.cpp/src/models/qwen35.cpp + delta-net-base.cpp (AR 경로).
출력: 레이어별 xs[:4] + logits top-5 — `LLM170_DEBUG_LAYERS=1` 출력과 대조.
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
            assert n < (1 << 20), f"bad len {n}"
            return self.f.read(n).decode()

        def rval(t):
            if t == 8:
                n = struct.unpack("<Q", self.f.read(8))[0]
                return self.f.read(n)
            if t == 9:
                et = struct.unpack("<I", self.f.read(4))[0]
                n = struct.unpack("<Q", self.f.read(8))[0]
                assert n < (1 << 26), f"bad arr len {n} et={et}"
                return [rval(et) for _ in range(n)]
            return struct.unpack(FMT[t], self.f.read(SZ[t]))[0]

        self.kv = {}
        for _ in range(n_kv):
            pos0 = self.f.tell()
            k = rstr()
            t = struct.unpack("<I", self.f.read(4))[0]
            self.kv[k] = rval(t)
            import sys as _s
            print(f"x", file=_s.stderr)
        self.tensors = {}
        self.dims = {}
        for _ in range(n_t):
            name = rstr()
            nd = struct.unpack("<I", self.f.read(4))[0]
            dims = struct.unpack("<" + "Q" * nd, self.f.read(8 * nd))
            ty = struct.unpack("<I", self.f.read(4))[0]
            off = struct.unpack("<Q", self.f.read(8))[0]
            self.tensors[name] = (off, ty)
            self.dims[name] = dims
        # 정렬 후 데이터 시작
        self.data_off = (self.f.tell() + 31) // 32 * 32

    def t(self, name):
        off, ty = self.tensors[name]
        dims = self.dims[name]
        n = 1
        for d in dims:
            n *= d
        self.f.seek(self.data_off + off)
        raw = self.f.read(4 * n)
        vals = struct.unpack(f"<{n}f", raw)
        k = dims[0]
        rows = n // k
        out = [list(vals[r * k:(r + 1) * k]) for r in range(rows)]
        if len(dims) == 1:
            out = out[0]  # 1D는 평탄화
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


def rms(x, w):
    s = sum(v * v for v in x) / len(x)
    scale = 1.0 / math.sqrt(s + EPS)
    return [v * scale * w[i] for i, v in enumerate(x)]


def l2n(x):
    s = math.sqrt(sum(v * v for v in x))
    scale = 1.0 / max(s, EPS)
    return [v * scale for v in x]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def silu(x):
    return x / (1.0 + math.exp(-x))


def softplus(x):
    return x if x > 20 else math.log1p(math.exp(x))


def mm(w, x):
    # w: rows 리스트 (각 행 len(x)) → y[r] = Σ w[r][i]·x[i]
    return [sum(a * b for a, b in zip(row, x)) for row in w]


embd, _ = g.t("token_embd.weight")

is_recr = lambda il: il % INTERVAL != INTERVAL - 1

# 시퀀스 상태 — 토큰 순차 AR (청크 수학과 독립)
states = {}
conv = {}
final_xs = None
layer_log = []

for ti, TOKEN in enumerate(TOKENS):
    x = embd[TOKEN]

for TOKEN in TOKENS:
    pass
# (아래에서 실제 루프)
    p = f"blk.{il}."
    norm_w, _ = g.t(p + "attn_norm.weight")
    cur = rms(x, norm_w)

    if is_recr(il):
        wqkv, _ = g.t(p + "attn_qkv.weight")
        wgate, _ = g.t(p + "attn_gate.weight")
        wbeta, _ = g.t(p + "ssm_beta.weight")
        walpha, _ = g.t(p + "ssm_alpha.weight")
        ssm_a, _ = g.t(p + "ssm_a")
        dtb, _ = g.t(p + "ssm_dt.bias")
        convw, _ = g.t(p + "ssm_conv1d.weight")
        ssnorm, _ = g.t(p + "ssm_norm.weight")
        wout, _ = g.t(p + "ssm_out.weight")

        qkv = mm(wqkv, cur)
        z = mm(wgate, cur)
        if il == 0:
            print(f"  py stage qkv max={max(abs(v) for v in qkv):.5f} z max={max(abs(v) for v in z):.5f} cur max={max(abs(v) for v in cur):.5f}")
        beta = [sigmoid(v) for v in mm(wbeta, cur)]
        alpha = mm(walpha, cur)
        gg = [softplus(alpha[h] + dtb[h]) * ssm_a[h] for h in range(DT_RANK)]

        # conv: 상태 0(첫 토큰), w[c*CONV_K + j]
        convout = []
        for c in range(CONV_CH):
            s = convw[c][CONV_K - 1] * qkv[c]  # 첫 토큰: conv 상태 0
            convout.append(silu(s))
        k_len = N_GROUP * D_STATE
        v_len = DT_RANK * D_STATE
        qh = [l2n(convout[h * D_STATE:(h + 1) * D_STATE]) for h in range(N_GROUP)]
        kh = [l2n(convout[k_len + h * D_STATE:k_len + (h + 1) * D_STATE]) for h in range(N_GROUP)]
        vh = [convout[2 * k_len + h * D_STATE:2 * k_len + (h + 1) * D_STATE] for h in range(DT_RANK)]

        # GDN AR (V 헤드 h ↔ K 헤드 h % N_GROUP)
        out_core = []
        for h in range(DT_RANK):
            st = states.setdefault((il, h), [[0.0] * D_STATE for _ in range(D_STATE)])
            kvec = kh[h % N_GROUP]
            qvec = qh[h % N_GROUP]
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

        if il == 0:
            print(f"  py stage core[:4]={[f'{v:.6f}' for v in out_core[0][:4]]} z[:4]={[f'{v:.6f}' for v in z[:4]]}")
        gated = []
        for h in range(DT_RANK):
            n = rms(out_core[h], ssnorm)
            zv = z[h * D_STATE:(h + 1) * D_STATE]
            gated += [n[i] * silu(zv[i]) for i in range(D_STATE)]
        if il == 0:
            print(f"  py gated h0={ [f'{v:.6f}' for v in gated[0:4]] }")
            print(f"  py gated h1={ [f'{v:.6f}' for v in gated[16:20]] }")
            print(f"  py gated h2={ [f'{v:.6f}' for v in gated[32:36]] }")
            print(f"  py gated h3={ [f'{v:.6f}' for v in gated[48:52]] }")
        attn_out = mm(wout, gated)
        if il == 0:
            print(f"  py stage gated max={max(abs(v) for v in gated):.5f} attn_out max={max(abs(v) for v in attn_out):.5f}")
    else:
        wq, _ = g.t(p + "attn_q.weight")
        wk, _ = g.t(p + "attn_k.weight")
        wv, _ = g.t(p + "attn_v.weight")
        wo, _ = g.t(p + "attn_output.weight")
        qn, _ = g.t(p + "attn_q_norm.weight")
        kn, _ = g.t(p + "attn_k_norm.weight")
        qg = mm(wq, cur)
        kk = mm(wk, cur)
        vv = mm(wv, cur)
        # pos 0 → rope 항등
        q_heads = [rms(qg[h * 2 * HD:h * 2 * HD + HD], qn) for h in range(N_HEAD)]
        gate_heads = [qg[h * 2 * HD + HD:h * 2 * HD + 2 * HD] for h in range(N_HEAD)]
        k_heads = [rms(kk[h * HD:(h + 1) * HD], kn) for h in range(N_KV)]
        v_heads = [vv[h * HD:(h + 1) * HD] for h in range(N_KV)]
        # 1토큰: 자기 자신만 attend → attention = v
        attn_out_flat = []
        for h in range(N_HEAD):
            kvh = h // (N_HEAD // N_KV)
            att = [v_heads[kvh][i] * sigmoid(gate_heads[h][i]) for i in range(HD)]
            attn_out_flat += att
        attn_out = mm(wo, attn_out_flat)

    x = [x[i] + attn_out[i] for i in range(N_EMBD)]
    ffn_res = list(x)
    post, _ = g.t(p + "post_attention_norm.weight")
    nrm = rms(x, post)
    wg, _ = g.t(p + "ffn_gate.weight")
    wu, _ = g.t(p + "ffn_up.weight")
    wd, _ = g.t(p + "ffn_down.weight")
    gy = mm(wg, nrm)
    uy = mm(wu, nrm)
    sil = [silu(gy[i]) * uy[i] for i in range(N_FF)]
    fd = mm(wd, sil)
    x = [ffn_res[i] + fd[i] for i in range(N_EMBD)]
    print(f"layer {il:>2} recr={is_recr(il)} max|x|={max(abs(v) for v in x):.4f} "
          f"head={','.join(f'{v:.5f}' for v in x[:4])}")

onorm, _ = g.t("output_norm.weight")
h = rms(x, onorm)
outw, _ = g.t("output.weight")
logits = mm(outw, h)
top = sorted(range(len(logits)), key=lambda i: -logits[i])[:5]
print("topk:", " ".join(f"{i}:{logits[i]:.4f}" for i in top))
