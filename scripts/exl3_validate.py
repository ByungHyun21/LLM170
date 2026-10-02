#!/usr/bin/env python3
"""EXL3 decode validator (plans/118 §3-1) — canonical.

Validates the trellis decode end-to-end against GGUF Q8_K_XL (same original
weights, corr on embeddings 0.99998) for several tensors and bitrates.

Confirmed pipeline (reference: exllamav3 CPU scalar path, moe_mul1.cpp):
  A_had = H128(x * suh) / sqrt(128)           [per-128 chunk along k]
  S     = A_had @ Wq                          [Wq from trellis decode]
  y     = H128(S) / sqrt(128) * svh           [per-128 chunk along n]
  tile[perm[t]] = mul1(word_t); word_t = ring bits [(t+1)K-16, (t+1)K)
  ring bit r = u32 word r/32, bit 31-(r%32)   [MSB-first within u32 words]
"""
import json, struct
import numpy as np
from gguf import GGUFReader

EXL3_DIR = '/home/yoon/models/Qwen3.8-27B-exl3-4.00bpw'
GGUF_PATH = '/home/yoon/models/qwen3.8-27b/Qwen3.8-27B-UD-Q8_K_XL.gguf'
SHARDS = {}


def st_hdr(shard):
    if shard not in SHARDS:
        f = open(f'{EXL3_DIR}/{shard}', 'rb')
        n = struct.unpack('<Q', f.read(8))[0]
        SHARDS[shard] = (f, json.loads(f.read(n)), 8 + n)
    return SHARDS[shard]


def find(name):
    for shard in ['model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors']:
        _, hdr, _ = st_hdr(shard)
        if name in hdr:
            return shard
    raise KeyError(name)


def read_tensor(name, dt):
    shard = find(name)
    f, hdr, base = st_hdr(shard)
    t = hdr[name]
    off, end = t['data_offsets']
    f.seek(base + off)
    return np.frombuffer(f.read(end - off), dtype=dt)


def tc_perm_inv():
    p = np.zeros(256, dtype=np.int64)
    for t in range(32):
        r0, c0 = (t % 4) * 2, t // 4
        for s, (dr, dc) in enumerate([(0, 0), (1, 0), (8, 0), (9, 0), (0, 8), (1, 8), (8, 8), (9, 8)]):
            p[t * 8 + s] = (r0 + dr) * 16 + (c0 + dc)
    inv = np.zeros(256, dtype=np.int64)
    inv[p] = np.arange(256)
    return inv


PERM_INV = tc_perm_inv()


def decode_tile(u16, K):
    """Literal port of decode_state_scalar + decode_mul1_scalar + scalar_tiles."""
    words32 = 8 * K
    u32 = [(int(u16[2 * i]) | (int(u16[2 * i + 1]) << 16)) for i in range(words32)]
    words = np.zeros(256, dtype=np.uint64)
    for t in range(256):
        b0 = t * K + K - 16 + 256 * K
        b1 = b0 + 16
        i0 = (b0 // 32) % words32
        i1 = ((b1 - 1) // 32) % words32
        s = ((b1 - 1) // 32 + 1) * 32 - b1
        merged = (u32[i0] << 32) | u32[i1]
        words[t] = (merged >> s) & 0xFFFF
    x = (words * np.uint64(0x83DCD12D)) & np.uint64(0xFFFFFFFF)
    b = (x & np.uint64(0xFF)) + ((x >> np.uint64(8)) & np.uint64(0xFF)) + \
        ((x >> np.uint64(16)) & np.uint64(0xFF)) + (x >> np.uint64(24))
    f = 1024.0 + b.astype(np.float64)
    k_inv = float(np.frombuffer(np.uint16(0x1eee).tobytes(), dtype='<f2')[0])
    k_bias = float(np.frombuffer(np.uint16(0xc931).tobytes(), dtype='<f2')[0])
    vals = (f * k_inv + k_bias).astype(np.float32)
    return vals[PERM_INV].reshape(16, 16)  # [r=k_local, c=n_local]


def had128():
    idx = np.arange(128)
    return np.where(np.bitwise_count(idx[:, None] & idx[None, :]) % 2 == 0, 1.0, -1.0)


def gguf_deq(name, rows, cols):
    t = [x for x in GGUFReader(GGUF_PATH).tensors if x.name == name][0]
    assert int(t.tensor_type) == 8, f'{name}: Q8_0 expected, got {t.tensor_type}'
    W = np.zeros((rows, cols), dtype=np.float64)
    for row in range(rows):
        rb = t.data[row]
        for blk in range(cols // 32):
            b = rb[blk * 34:(blk + 1) * 34]
            d = np.frombuffer(b[:2], dtype='<f2')[0]
            q = np.frombuffer(b[2:], dtype=np.int8).astype(np.float64)
            W[row, blk * 32:blk * 32 + 32] = d * q
    return W  # [n(out), k(in)]


def validate(exl3_key, gguf_name, kt_full, nt_full, ktiles=8, ntiles=8):
    tre = read_tensor(f'{exl3_key}.trellis', '<u2').reshape(kt_full, nt_full, -1)
    tre = tre[:ktiles, :ntiles]
    suh = read_tensor(f'{exl3_key}.suh', '<f2').astype(np.float64)
    svh = read_tensor(f'{exl3_key}.svh', '<f2').astype(np.float64)
    K = tre.shape[2] // 16
    k, n = 16 * ktiles, 16 * ntiles
    assert len(suh) >= k and len(svh) >= n

    Wq = np.zeros((k, n), dtype=np.float32)
    for kt in range(ktiles):
        for nt in range(ntiles):
            Wq[kt * 16:kt * 16 + 16, nt * 16:nt * 16 + 16] = decode_tile(tre[kt, nt], K)

    H = had128()
    rng = np.random.default_rng(7)
    x = rng.standard_normal(k)
    A = (x * suh[:k]) @ H / np.sqrt(128.0)
    y = ((A @ Wq) @ H / np.sqrt(128.0)) * svh[:n]
    y_ref = x @ gguf_deq(gguf_name, n, k).T
    c = np.corrcoef(y, y_ref)[0, 1]
    rel = np.abs(y - y_ref).max() / np.abs(y_ref).max()
    print(f'{exl3_key.split(".layers.")[-1]:22s} K={K}: corr={c:.5f} max_rel={rel:.4f}')
    return c


if __name__ == '__main__':
    P = 'model.language_model.layers'
    validate(f'{P}.0.mlp.gate_proj', 'blk.0.ffn_gate.weight', 320, 1088)            # K=3
    validate(f'{P}.0.mlp.down_proj', 'blk.0.ffn_down.weight', 1088, 320)            # K=3
    validate(f'{P}.3.self_attn.o_proj', 'blk.3.attn_output.weight', 384, 320)      # K=4
    validate(f'{P}.0.linear_attn.in_proj_qkv', 'blk.0.attn_qkv.weight', 320, 640)  # K=5
