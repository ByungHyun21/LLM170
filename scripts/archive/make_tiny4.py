#!/usr/bin/env python3
"""무작위 소형 qwen4exp GGUF 생성 — 모델 볼륨 접근 불가 시 e2e 검증 경로.

토크나이저 kv는 /tmp/qwen35-tiny.gguf(존재시) 또는 원본 27B에서 복사.
차원 축소: n_embd 128, 4층(3 GDN + 1 QSA), MoE 8전문가(top-2)+shared,
PLE 없음(ple.layers 빈 배열), hc=2, 인덱서 2head·top_k 8·compress 4.
가중치는 q8_0 (GPU 배치 커널 경로 포함) — 시드 고정.
"""
import os
import random
import struct
import sys

DST = sys.argv[1] if len(sys.argv) > 1 else "/tmp/qwen4exp-tiny.gguf"

N_EMBD = 128
N_LAYER = 4                 # compress = [0,0,0,4] → 3 GDN + 1 QSA
N_FF = 64
N_EXPERT = 8
N_USED = 2
HC = 2
HC_LR = 16
N_HEAD = 4
N_KV = 2
HEAD_DIM = 32
N_ROT = 16
IDX_HEADS = 2
IDX_DIM = 16
IDX_TOP_K = 8
D_STATE = 16
N_GROUP = 2
DT_RANK = 6
D_INNER = DT_RANK * D_STATE
CONV_K = 4
VOCAB = 248320
PLE_HEAD_DIM = 512          # embedding_length_per_layer_input (iq4_nl 32 블록 정렬)
PLE_ROWS = 4096             # Σ head_vocab_sizes — 합성 축소 (실모델 320M)
PLE_HPNG = 2                # heads_per_ngram → heads = 2*(ngram-1) = 4
PLE_NGRAM = 3
PLE_CONV_K = 4
PLE_EOS = 248043

rng = random.Random(170)
F32 = struct.Struct("<f")
I8 = struct.Struct("<b")

def rand_f32(n, scale=0.05):
    return b"".join(F32.pack(rng.uniform(-scale, scale)) for _ in range(n))

def near1(n):
    return b"".join(F32.pack(1.0 + rng.uniform(-0.02, 0.02)) for _ in range(n))

def rand_iq4_nl(n_rows, n_cols):
    """iq4_nl 행렬: 블록당 f16 d + 16바이트 니블(하위닢 먼저)."""
    assert n_cols % 32 == 0
    out = []
    for _ in range(n_rows * n_cols // 32):
        dh = rng.uniform(0.004, 0.006)
        out.append(struct.pack("<e", dh))
        out.append(bytes(rng.randrange(256) for _ in range(16)))
    return b"".join(out)

def rand_q8_0(n_rows, n_cols):
    """q8_0 행렬: 블록당 f16 d + 32 i8."""
    out = []
    for _ in range(n_rows * (n_cols // 32)):
        d = rng.uniform(0.004, 0.009)
        h = struct.unpack("<e", struct.pack("<e", d))[0]
        out.append(struct.pack("<e", h))
        out.append(b"".join(I8.pack(rng.randint(-120, 120)) for _ in range(32)))
    return b"".join(out)

# ---------- GGUF writer ----------
GGUF_MAGIC = b"GGUF"
GGUF_VER = 3
T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR = range(9)
T_U64, T_I64, T_F64 = 10, 11, 12
SZ = {T_U8: 1, T_I8: 1, T_U16: 2, T_I16: 2, T_U32: 4, T_I32: 4, T_F32: 4,
      T_BOOL: 1, T_STR: 1, T_U64: 8, T_I64: 8, T_F64: 8}

def s(v):  # string bytes (len 제외)
    return v.encode()

kvs = []  # (key, type, value)
def kv(k, t, v):
    kvs.append((k, t, v))

kv("general.architecture", T_STR, "qwen4exp")
kv("general.name", T_STR, "tiny4")
kv("qwen4exp.embedding_length", T_U32, N_EMBD)
kv("qwen4exp.block_count", T_U32, N_LAYER)
kv("qwen4exp.attention.head_count", T_U32, N_HEAD)
kv("qwen4exp.attention.head_count_kv", T_U32, N_KV)
kv("qwen4exp.attention.key_length", T_U32, HEAD_DIM)
kv("qwen4exp.rope.dimension_count", T_U32, N_ROT)
kv("qwen4exp.rope.freq_base", T_F32, 1e7)
kv("qwen4exp.attention.layer_norm_rms_epsilon", T_F32, 1e-6)
kv("qwen4exp.ssm.inner_size", T_U32, D_INNER)
kv("qwen4exp.ssm.time_step_rank", T_U32, DT_RANK)
kv("qwen4exp.ssm.state_size", T_U32, D_STATE)
kv("qwen4exp.ssm.group_count", T_U32, N_GROUP)
kv("qwen4exp.ssm.conv_kernel", T_U32, CONV_K)
kv("qwen4exp.expert_count", T_U32, N_EXPERT)
kv("qwen4exp.expert_used_count", T_U32, N_USED)
kv("qwen4exp.expert_feed_forward_length", T_U32, N_FF)
kv("qwen4exp.expert_shared_feed_forward_length", T_U32, N_FF)
kv("qwen4exp.hyper_connection.count", T_U32, HC)
kv("qwen4exp.hyper_connection.low_rank", T_U32, HC_LR)
kv("qwen4exp.attention.indexer.head_count", T_U32, IDX_HEADS)
kv("qwen4exp.attention.indexer.key_length", T_U32, IDX_DIM)
kv("qwen4exp.attention.indexer.top_k", T_U32, IDX_TOP_K)
kv_arr_compress = ([0] * 3 + [4])
kv_arr_ple_layers = [1]  # blk.1 — 실모델과 동일 위치(ple.layers)
if os.environ.get("TINY4_NO_PLE") == "1":
    kv_arr_ple_layers = []  # PLE 경로 격리 디버그용 (프레임 발산 특정)
ple_mult = [2654435761, 40503, 974634551]  # ngram=3 → 인덱스 0..2
ple_off = [0, 1024, 2048, 3072]
ple_vs = [1024, 1024, 1024, 1024]

tensors = []  # (name, raw bytes, ne)
def t(name, data, ne):
    tensors.append((name, data, ne))

hc_dim = HC * N_EMBD
conv_ch = N_GROUP * D_STATE * 2 + DT_RANK * D_STATE
qkv_out = conv_ch
# 층별 텐서
for il in range(N_LAYER):
    recr = kv_arr_compress[il] == 0
    for kind in ("attn", "ffn"):
        t(f"blk.{il}.hc_{kind}_norm.weight", near1(hc_dim), [hc_dim])
        t(f"blk.{il}.hc_{kind}_down.weight", rand_f32(hc_dim * HC_LR), [hc_dim, HC_LR])
        t(f"blk.{il}.hc_{kind}_up.weight", rand_f32(hc_dim * HC_LR), [HC_LR, hc_dim])
        t(f"blk.{il}.hc_{kind}_inject.weight", rand_f32(hc_dim * HC), [hc_dim, HC])
    if recr:
        t(f"blk.{il}.attn_qkv.weight", rand_q8_0(qkv_out, N_EMBD), [N_EMBD, qkv_out])
        t(f"blk.{il}.attn_gate.weight", rand_q8_0(D_INNER, N_EMBD), [N_EMBD, D_INNER])
        t(f"blk.{il}.ssm_beta.weight", rand_f32(N_EMBD * DT_RANK), [N_EMBD, DT_RANK])
        t(f"blk.{il}.ssm_alpha.weight", rand_f32(N_EMBD * DT_RANK), [N_EMBD, DT_RANK])
        t(f"blk.{il}.ssm_a", rand_f32(DT_RANK, 0.02), [DT_RANK])
        t(f"blk.{il}.ssm_dt.bias", rand_f32(DT_RANK, 0.02), [DT_RANK])
        t(f"blk.{il}.ssm_conv1d.weight", rand_f32(conv_ch * CONV_K, 0.05), [CONV_K, conv_ch])
        t(f"blk.{il}.ssm_norm.weight", near1(D_STATE), [D_STATE])
        t(f"blk.{il}.ssm_out.weight", rand_q8_0(N_EMBD, D_INNER), [D_INNER, N_EMBD])
    else:
        q_out = N_HEAD * HEAD_DIM * 2  # q‖gate 인터리브
        t(f"blk.{il}.attn_q.weight", rand_q8_0(q_out, N_EMBD), [N_EMBD, q_out])
        t(f"blk.{il}.attn_k.weight", rand_q8_0(N_KV * HEAD_DIM, N_EMBD), [N_EMBD, N_KV * HEAD_DIM])
        t(f"blk.{il}.attn_v.weight", rand_q8_0(N_KV * HEAD_DIM, N_EMBD), [N_EMBD, N_KV * HEAD_DIM])
        t(f"blk.{il}.attn_output.weight", rand_q8_0(N_EMBD, N_HEAD * HEAD_DIM), [N_HEAD * HEAD_DIM, N_EMBD])
        t(f"blk.{il}.attn_q_norm.weight", near1(HEAD_DIM), [HEAD_DIM])
        t(f"blk.{il}.attn_k_norm.weight", near1(HEAD_DIM), [HEAD_DIM])
        t(f"blk.{il}.indexer.q_norm.weight", near1(IDX_DIM), [IDX_DIM])
        t(f"blk.{il}.indexer.k_norm.weight", near1(IDX_DIM), [IDX_DIM])
        t(f"blk.{il}.indexer.q_proj.weight", rand_q8_0(IDX_HEADS * IDX_DIM, N_EMBD), [N_EMBD, IDX_HEADS * IDX_DIM])
        t(f"blk.{il}.indexer.k_proj.weight", rand_f32(N_EMBD * IDX_DIM), [N_EMBD, IDX_DIM])
    # MoE
    t(f"blk.{il}.ffn_gate_inp.weight", rand_f32(N_EMBD * N_EXPERT), [N_EMBD, N_EXPERT])
    t(f"blk.{il}.ffn_gate_inp_shexp.weight", rand_f32(N_EMBD, 0.05), [N_EMBD])
    t(f"blk.{il}.ffn_gate_shexp.weight", rand_q8_0(N_FF, N_EMBD), [N_EMBD, N_FF])
    t(f"blk.{il}.ffn_up_shexp.weight", rand_q8_0(N_FF, N_EMBD), [N_EMBD, N_FF])
    t(f"blk.{il}.ffn_down_shexp.weight", rand_q8_0(N_EMBD, N_FF), [N_FF, N_EMBD])
    t(f"blk.{il}.ffn_gate_exps.weight", rand_q8_0(N_EXPERT * N_FF, N_EMBD), [N_EMBD, N_FF, N_EXPERT])
    t(f"blk.{il}.ffn_up_exps.weight", rand_q8_0(N_EXPERT * N_FF, N_EMBD), [N_EMBD, N_FF, N_EXPERT])
    t(f"blk.{il}.ffn_down_exps.weight", rand_q8_0(N_EXPERT * N_EMBD, N_FF), [N_FF, N_EMBD, N_EXPERT])

t("token_embd.weight", rand_q8_0(VOCAB, N_EMBD), [N_EMBD, VOCAB])
t("output.weight", rand_q8_0(VOCAB, N_EMBD), [N_EMBD, VOCAB])
t("output_hc_norm.weight", near1(hc_dim), [hc_dim])
t("output_hc_down.weight", rand_f32(hc_dim * HC_LR), [hc_dim, HC_LR])
t("output_hc_up.weight", rand_f32(hc_dim * HC_LR), [HC_LR, hc_dim])

# PLE (blk.1) — ple.rs 소비 텐서 전부 (key/value는 q8_0, norm/conv는 f32)
ple_emb_w = PLE_HPNG * 2 * PLE_HEAD_DIM  # heads×head_dim = 4×512 = 2048
t("blk.1.ple_key.weight", rand_q8_0(hc_dim, ple_emb_w), [ple_emb_w, hc_dim])
t("blk.1.ple_value.weight", rand_q8_0(hc_dim, ple_emb_w), [ple_emb_w, hc_dim])
t("blk.1.ple_norm_key.weight", near1(hc_dim), [hc_dim])
t("blk.1.ple_norm_query.weight", near1(hc_dim), [hc_dim])
t("blk.1.ple_norm_conv.weight", near1(hc_dim), [hc_dim])
t("blk.1.ple_conv1d.weight", rand_f32(hc_dim * PLE_CONV_K), [PLE_CONV_K, hc_dim])
t("per_layer_token_embd.weight", rand_iq4_nl(PLE_ROWS, PLE_HEAD_DIM), [PLE_HEAD_DIM, PLE_ROWS])

# ---------- 토크나이저 kv (기존 tiny에서 복사 시도) ----------
try:
    src = "/tmp/qwen35-tiny.gguf"
    f = open(src, "rb")
    assert f.read(4) == GGUF_MAGIC
    f.read(4)
    n_t, n_kv = struct.unpack("<QQ", f.read(16))
    def rstr():
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode()
    def rval(typ):
        if typ == T_STR:
            return rstr()
        if typ == 8:
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n)
        if typ in (0, 7):
            return struct.unpack("<B", f.read(1))[0]
        fmts = {1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 10: "<Q", 11: "<q", 12: "<d"}
        n = 1
        return struct.unpack(fmts[typ], f.read(SZ[typ]))[0]
    copied = 0
    for _ in range(n_kv):
        k = rstr()
        typ = struct.unpack("<I", f.read(4))[0]
        if k.startswith("tokenizer.ggml."):
            if typ == 9:  # array
                et = struct.unpack("<I", f.read(4))[0]
                n = struct.unpack("<Q", f.read(8))[0]
                vals = []
                for _ in range(n):
                    if et == T_STR:
                        ln = struct.unpack("<Q", f.read(8))[0]
                        vals.append(f.read(ln).decode(errors="replace"))
                    else:
                        fmts = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f"}
                        vals.append(struct.unpack(fmts[et], f.read(SZ[et]))[0])
                kvs.append((k, 9, (et, vals)))
                copied += 1
            else:
                kvs.append((k, typ, rval(typ)))
                copied += 1
    f.close()
    print(f"토크나이저 kv {copied}개 복사")
except Exception as e:
    print(f"토크나이저 복사 실패({e}) — 최소 scores만")
    kv("tokenizer.ggml.model", T_STR, "gpt2")
    kv("tokenizer.ggml.scores", 9, (T_F32, [0.0] * VOCAB))

kv("qwen4exp.attention.compress_ratios", 9, (T_I32, kv_arr_compress))
kv("qwen4exp.ple.layers", 9, (T_I32, kv_arr_ple_layers))
kv("qwen4exp.ple.layer_multipliers", 9, (T_U64, ple_mult))
kv("qwen4exp.ple.head_offsets", 9, (T_U64, ple_off))
kv("qwen4exp.ple.head_vocab_sizes", 9, (T_U64, ple_vs))
kv("qwen4exp.ple.ngram_size", T_U32, PLE_NGRAM)
kv("qwen4exp.ple.heads_per_ngram", T_U32, PLE_HPNG)
kv("qwen4exp.ple.conv_kernel", T_U32, PLE_CONV_K)
kv("qwen4exp.embedding_length_per_layer_input", T_U32, PLE_HEAD_DIM)
kv("qwen4exp.ple.eos_token_id", T_U32, PLE_EOS)
kv("qwen4exp.ple.image_token_id", T_U32, 0)

# ---------- 직렬화 ----------
Q8_0 = 8
F32_T = 0
IQ4_NL = 20

def tensor_ty(name):
    if name == "per_layer_token_embd.weight":
        return IQ4_NL
    return F32_T if is_f32_tensor(name) else Q8_0

def wstr(x):
    b = x.encode() if isinstance(x, str) else x
    return struct.pack("<Q", len(b)) + b

def write_kv_head(head, k, typ, v):
    head += wstr(k) + struct.pack("<I", typ)
    if typ == T_STR:
        head += wstr(v)
    elif typ == 9:
        et, vals = v
        head += struct.pack("<I", et) + struct.pack("<Q", len(vals))
        fmts = {T_U8: "<B", T_I8: "<b", T_U16: "<H", T_I16: "<h",
                T_U32: "<I", T_I32: "<i", T_F32: "<f", T_U64: "<Q", T_I64: "<q", T_F64: "<d"}
        for x in vals:
            if et == T_STR:
                head += wstr(x)
            else:
                head += struct.pack(fmts[et], x)
    else:
        fmts = {T_U8: "<B", T_I8: "<b", T_U16: "<H", T_I16: "<h", T_U32: "<I",
                T_I32: "<i", T_F32: "<f", T_BOOL: "<B", T_U64: "<Q", T_I64: "<q", T_F64: "<d"}
        head += struct.pack(fmts[typ], v)
    return head

head = bytearray()
head += GGUF_MAGIC + struct.pack("<I", GGUF_VER)
head += struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(kvs))
for k, typ, v in kvs:
    head = write_kv_head(head, k, typ, v)

def is_f32_tensor(name):
    # f32 대상: norm류 + 32미만 행 길이 텐서 (q8_0 블록 최소 단위)
    return (("norm" in name) or name.endswith(".ssm_a") or
            ("ssm_dt.bias" in name) or
            ("conv1d" in name) or ("gate_inp_shexp" in name) or
            ("hc_attn_down" in name) or ("hc_attn_up" in name) or
            ("hc_attn_inject" in name) or ("hc_ffn_down" in name) or
            ("hc_ffn_up" in name) or ("hc_ffn_inject" in name) or
            ("ssm_beta" in name) or ("ssm_alpha" in name) or
            ("indexer.k_proj" in name) or ("ffn_gate_inp.weight" in name) or
            ("output_hc_down" in name) or ("output_hc_up" in name))

infos = bytearray()
for name, data, ne in tensors:
    ty = tensor_ty(name)
    infos += wstr(name)
    infos += struct.pack("<I", len(ne))  # n_dims는 u32 (gguf.cpp 스펙)
    for d in ne:  # ne는 이미 ggml 순서 (ne0 최우선) — 반전 금지
        infos += struct.pack("<Q", d)
    infos += struct.pack("<I", ty)
    infos += struct.pack("<Q", 0)  # offset 자리

data_start = (len(head) + len(infos) + 31) // 32 * 32
off = 0  # GGUF 텐서 offset은 data_offset 상대 (절대 아님)
entries = []
for name, data, ne in tensors:
    ty = tensor_ty(name)
    entries.append((name, ty, ne, off, data))
    off += len(data)
    off = (off + 31) // 32 * 32

# offset 패치 (infos 내 8바이트 슬롯을 다시 씀음 — 순서 동일 재구성)
infos = bytearray()
for name, ty, ne, o, data in entries:
    infos += wstr(name)
    infos += struct.pack("<I", len(ne))  # n_dims는 u32 (gguf.cpp 스펙)
    for d in ne:  # ne는 이미 ggml 순서 (ne0 최우선) — 반전 금지
        infos += struct.pack("<Q", d)
    infos += struct.pack("<I", ty)
    infos += struct.pack("<Q", o)

out = open(DST, "wb")
out.write(head)
out.write(infos)
out.write(b"\x00" * (data_start - len(head) - len(infos)))
for name, ty, ne, o, data in entries:
    out.write(data)
    out.write(b"\x00" * (-len(data) % 32))
out.close()
import os
print(f"wrote {DST}: {os.path.getsize(DST)/1e6:.1f} MB, {len(tensors)} tensors")
