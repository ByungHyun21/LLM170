#!/usr/bin/env python3
# [테스트 재설계 2026-10-22] 합성 tiny W4A16 모델 생성기 — CPU↔GPU 같은 실행 등가
# 테스트의 고정 재료. 시드 고정 랜덤 가중치(splitmix64 계열 LCG) — 재현 보장.
#
# 사용: scripts/gen_tiny_w4a16.py [출력루트=target/tiny-fixtures]
# 생성: <root>/tiny-dense (qwen35 dense g128, 4층: GDN 3 + attn 1)
#       <root>/tiny-moe   (qwen35 MoE g32 전문가 8·top2, 어텐션/GDN 플레인 bf16)
#
# 형상 계약(실모델 헤더 실측 + hparams.rs 공식):
#   d_state=linear_key_head_dim, dt_rank=linear_num_value_heads,
#   d_inner=nv·lkhd, conv_ch=d_inner+2·nk·lkhd, n_group=nk
#   in_proj_qkv n=conv_ch, in_proj_z n=d_inner, out_proj (hidden, d_inner)
#   linear_attn.norm.weight=[d_state], A_log/dt_bias/in_proj_a/b=[dt_rank]
#   GDN 순열 정렬: (conv_ch−2·nk·lkhd)%128==0, z n%128==0, out k%128==0
#   니블 lsb-first, zp=8, 그룹 g128(dense)/g32(MoE 전문가), 스케일 bf16
import json, struct, sys, os

def lcg(seed):
    x = seed & ((1 << 64) - 1)
    while True:
        x = (x * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        yield ((x >> 11) / float(1 << 53)) - 0.5  # [-0.5, 0.5)

def f16(v):
    import struct as s
    return s.pack('<e', v)

def bf16(v):
    b = struct.pack('<f', v)
    return b[2:] + b[3:] if False else bytes([b[2], b[3]])  # 상위 16비트 절단

class St:
    def __init__(self):
        self.t = {}
    def add(self, name, dtype, shape, data):
        self.t[name] = (dtype, shape, data)
    def bf16(self, name, shape, gen, scale=1.0):
        n = 1
        for d in shape: n *= d
        vals = [next(gen) * scale for _ in range(n)]
        self.add(name, 'BF16', shape, b''.join(bf16(v) for v in vals))
    def triple(self, name, n, k, gen, group, sigma=0.05):
        # [GPU 커널 계약] g128=F16 스케일(27B형) · g32=BF16(35B형) — 실측 헤더.
        spack = (lambda v: f16(v)) if group == 128 else bf16
        sdt = 'F16' if group == 128 else 'BF16'
        # W ~ N(0, sigma) 근사(LCG uniform 합) → 대칭 int4 그룹 양자화.
        import math
        ng = k // group
        packed = bytearray()
        scales = bytearray()
        for r in range(n):
            row = [sum(next(gen) for _ in range(4)) * sigma for _ in range(k)]
            words = [0] * (k // 8)
            srow = bytearray()
            for g in range(ng):
                seg = row[g * group:(g + 1) * group]
                amax = max(abs(v) for v in seg) or 1e-6
                s = amax / 7.0
                srow += spack(s)
                for i, v in enumerate(seg):
                    q = max(-8, min(7, round(v / s)))
                    nib = q + 8
                    w = i >> 3
                    words[w] |= (nib & 0xF) << (4 * (i & 7))
            packed += struct.pack('<%dI' % (k // 8), *words)
            scales += srow
        self.add(name + '.weight_packed', 'I32', [n, k // 8], bytes(packed))
        self.add(name + '.weight_scale', sdt, [n, ng], bytes(scales))
        self.add(name + '.weight_shape', 'I64', [2], struct.pack('<qq', n, k))
    def write(self, path):
        entries, off, blob = {}, 0, bytearray()
        for name in sorted(self.t):
            dtype, shape, data = self.t[name]
            nb = len(data)
            entries[name] = {'dtype': dtype, 'shape': shape,
                             'data_offsets': [off, off + nb]}
            blob += data
            off += nb
        hdr = json.dumps({'__metadata__': {'format': 'pt'},
                          **entries}).encode()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(struct.pack('<Q', len(hdr)))
            f.write(hdr)
            f.write(blob)
H, L, V = 256, 4, 512
FFN, MOE_FFN, N_EXP, TOPK = 512, 512, 8, 2
NK, NV, LKHD = 2, 4, 128
D_INNER, CONV_CH = NV * LKHD, NV * LKHD + 2 * NK * LKHD
HEADS, KV, HD = 4, 2, 128

def base_cfg(arch):
    qc = {"quant_method": "compressed-tensors",
          "config_groups": {"group_0": {"format": "pack-quantized",
              "weights": {"num_bits": 4, "symmetric": True, "group_size": 128}}},
          "format": "pack-quantized"}
    return {"architectures": [arch], "dtype": "bfloat16",
            "text_config": {
                "hidden_size": H, "num_hidden_layers": L,
                "num_attention_heads": HEADS, "num_key_value_heads": KV,
                "head_dim": HD, "vocab_size": V, "max_position_embeddings": 1024,
                "intermediate_size": FFN,
                "linear_key_head_dim": LKHD, "linear_num_key_heads": NK,
                "linear_num_value_heads": NV, "linear_conv_kernel_dim": 4,
                "full_attention_interval": 4, "partial_rotary_factor": 0.25,
                "rope_theta": 10000000.0, "rms_norm_eps": 1e-6,
                "quantization_config": qc}}

def gdn_layer(st, p, g, quant, group):
    # 트리플(양자화) 또는 플레인 bf16 — MoE 모델은 어텐션/GDN 투영 플레인.
    def lin(name, n, k, sigma=0.05):
        if quant: st.triple(p + name, n, k, g, group, sigma)
        else: st.bf16(p + name + '.weight', [n, k], g, sigma)
    lin('linear_attn.in_proj_qkv', CONV_CH, H)
    lin('linear_attn.in_proj_z', D_INNER, H)
    lin('linear_attn.out_proj', H, D_INNER)
    st.bf16(p + 'linear_attn.norm.weight', [LKHD], g, 0.1)
    st.bf16(p + 'linear_attn.A_log', [NV], g, -1.0)   # ssm_a=−exp(A_log) 안정(음수)
    st.bf16(p + 'linear_attn.dt_bias', [NV], g, 0.1)
    st.bf16(p + 'linear_attn.in_proj_a.weight', [NV, H], g, 0.05)
    st.bf16(p + 'linear_attn.in_proj_b.weight', [NV, H], g, 0.05)
    st.bf16(p + 'linear_attn.conv1d.weight', [CONV_CH, 1, 4], g, 0.05)
    st.bf16(p + 'input_layernorm.weight', [H], g, 0.1)      # HF 규약 w−1 → 엔진 +1
    st.bf16(p + 'post_attention_layernorm.weight', [H], g, 0.1)

def attn_layer(st, p, g, quant, group):
    def lin(name, n, k, sigma=0.05):
        if quant: st.triple(p + name, n, k, g, group, sigma)
        else: st.bf16(p + name + '.weight', [n, k], g, sigma)
    lin('self_attn.q_proj', HEADS * 2 * HD, H)  # 헤드별 q+gate 교차(attn.rs:130)
    lin('self_attn.k_proj', KV * HD, H)
    lin('self_attn.v_proj', KV * HD, H)
    lin('self_attn.o_proj', H, HEADS * HD)
    st.bf16(p + 'self_attn.q_norm.weight', [HD], g, 0.1)
    st.bf16(p + 'self_attn.k_norm.weight', [HD], g, 0.1)
    st.bf16(p + 'input_layernorm.weight', [H], g, 0.1)
    st.bf16(p + 'post_attention_layernorm.weight', [H], g, 0.1)

def build(root, name, moe):
    st, g = St(), lcg(0x7149_4901)
    arch = 'Qwen3_5MoeForConditionalGeneration' if moe else 'Qwen3_5ForConditionalGeneration'
    cfg = base_cfg(arch)
    quant_dense = not moe  # MoE 모델의 dense 투영은 플레인(실측 35B 계약)
    if moe:
        tc = cfg['text_config']
        tc.pop('intermediate_size')
        tc.update({'num_experts': N_EXP, 'num_experts_per_tok': TOPK,
                   'moe_intermediate_size': MOE_FFN,
                   'shared_expert_intermediate_size': MOE_FFN})
        tc['quantization_config']['config_groups']['group_0']['weights']['group_size'] = 32
    for il in range(L):
        p = 'model.language_model.layers.%d.' % il
        if (il + 1) % 4 == 0:  # full attention
            attn_layer(st, p, g, quant_dense, 128)
        else:
            gdn_layer(st, p, g, quant_dense, 128)
        if moe:
            st.bf16(p + 'mlp.gate.weight', [N_EXP, H], g, 0.2)
            st.bf16(p + 'mlp.shared_expert.gate_proj.weight', [MOE_FFN, H], g, 0.05)
            st.bf16(p + 'mlp.shared_expert.up_proj.weight', [MOE_FFN, H], g, 0.05)
            st.bf16(p + 'mlp.shared_expert.down_proj.weight', [H, MOE_FFN], g, 0.05)
            st.bf16(p + 'mlp.shared_expert_gate.weight', [1, H], g, 0.2)
            for e in range(N_EXP):
                ep = '%smlp.experts.%d.' % (p, e)
                st.triple(ep + 'gate_proj', MOE_FFN, H, g, 32, 0.05)
                st.triple(ep + 'up_proj', MOE_FFN, H, g, 32, 0.05)
                st.triple(ep + 'down_proj', H, MOE_FFN, g, 32, 0.05)
        else:
            st.triple(p + 'mlp.gate_proj', FFN, H, g, 128)
            st.triple(p + 'mlp.up_proj', FFN, H, g, 128)
            st.triple(p + 'mlp.down_proj', H, FFN, g, 128)
    st.bf16('model.language_model.embed_tokens.weight', [V, H], g, 0.05)
    st.bf16('model.language_model.norm.weight', [H], g, 0.1)
    st.bf16('lm_head.weight', [V, H], g, 0.05)
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    st.write(os.path.join(d, 'model-00001-of-00001.safetensors'))
    with open(os.path.join(d, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=1)
    vocab = {'model': {'vocab': {'t%d' % i: i for i in range(V)}},
             'added_tokens': []}
    with open(os.path.join(d, 'tokenizer.json'), 'w') as f:
        json.dump(vocab, f)
    print('wrote', d)

if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else 'target/tiny-fixtures'
    build(root, 'tiny-dense', moe=False)
    build(root, 'tiny-moe', moe=True)
