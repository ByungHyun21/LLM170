# qwen35 — Qwen3.8-27B Implementation Spec

Dense hybrid. Architectural detail was captured during development in local
research notes (`source/research/`, untracked) with source line references.
Ground truth: the upstream `config.json` and `llama.cpp/src/models/qwen35.cpp`
(referenced read-only during development).

## Hyperparameters

- 64 layers: `12×(3×(GDN→FFN) + 1×(GatedAttention→FFN))`, `full_attention_interval=4`
  → full-attention layers il∈{3,7,…,63} (16 layers), GDN 48 layers. GGUF
  `block_count=65` (64 body + MTP blk.64).
- n_embd 5120, FFN 17408 (SwiGLU), vocab 248320, rms_eps 1e-6, ctx 262144.
- MTP: `mtp_num_hidden_layers=1`.

## Layer Structure

**Residual**: `h = h + attn(rms(h)); h = h + ffn(rms_post(h))` — the FFN branch
uses the **post**-attention norm (`post_attention_norm.weight`).

**Gated Attention (16 layers)** — 24 Q / 4 KV heads, head_dim 256:
- `wq` is a fused **Q‖gate** [5120, 12288], interleaved per head
  (stride 2·head_dim).
- Per-head RMSNorm on Q and K [256] (`attn_q_norm` / `attn_k_norm`).
- **IMROPE**: rotary dim 64 (32 pairs), sections `[11,11,10,0]`, cyclic t/h/w
  assignment, freq_base 1e7. kq_scale = 1/√256.
- Output: `attn_out ⊙ sigmoid(gate)` before `wo`.
- KV: 4 KiB/layer/token at f16 → **64 KiB/token** (16 layers).

**Gated DeltaNet (48 layers)** — 16 K heads (grouped) / 48 V heads, head_dim 128:
- `attn_qkv` [5120,10240] (q‖k‖v — **V-heads are regrouped→tiled during
  conversion**) + `attn_gate` (z) [5120,6144].
- `ssm_conv1d` [4,10240] depthwise conv + SiLU, 3-token rolling state.
- q,k L2-norm → beta = sigmoid(`ssm_beta`); gate g =
  softplus(`ssm_alpha`+`ssm_dt.bias`)·`ssm_a`, where `ssm_a` = −exp(A_log)
  (applied during conversion).
- GDN recurrence: state S=[128,128,48] f32 = 3 MiB/layer/seq → **144 MiB/seq**
  + conv state 5.6 MiB/seq.
- out = rms_norm(core) ⊙ **SiLU**(z) (qwen35 uses SiLU; qwen4exp uses sigmoid).

**MTP/NextN (blk.64, loaded for draft-mtp only)**: input =
concat(rms_enorm(embd(t+1)), rms_hnorm(h)) → `eh_proj` [10240,5120] → one
gated-attention block → FFN → shared output_norm/LM head. Measured draft
acceptance 0.47–0.85.

## GGUF Contract

- Metadata: common block + GDN block (`ssm.conv_kernel=4, ssm.inner_size=6144,
  ssm.state_size=128, ssm.time_step_rank=48, ssm.group_count=16`) +
  `nextn_predict_layers=1`.
- Tensor names: `token_embd/output{,_norm}`, per-block `attn_norm`,
  `post_attention_norm`, full-attn `attn_{q,k,v,output,q_norm,k_norm}`, GDN
  `attn_{qkv,gate}`, `ssm_{conv1d,dt.bias,a,beta,alpha,norm,out}`. MTP:
  `blk.64.nextn.*`.
- **Conversion transforms** (already applied in the GGUF — only needed when
  loading from HF originals): in_proj_qkvz split, V-head tiled re-layout,
  `A_log→−exp(A_log)`, RMSNorm gammas stored as **w+1**.
- Tokenizer: BPE pre `qwen35` (combining-character handling), bos=eos=248044.

## Implementation Notes

- GDN fused single op (snapshot packing) + chunked path
  (solve_tri/cumsum/tri) in parallel — see the kernel list in
  [architecture.md](../architecture.md).
- **MTP chain speculative decoding implemented** (`llm170 infer --spec k`,
  k <= 8; `bench --spec k` reports effective t/s): target forward + k-draft
  chain with greedy acceptance. Ignored (with a notice) when the GGUF has
  no nextn tensors.
- KV exists for only 16 layers — the reference case for hybrid per-layer cache
  typing.
- Local operating baseline: Q4+MTP tg 15.5–24.4 t/s, pp 283–556 t/s (llama.cpp,
  np4×262K, 2026-08-18) — for relative comparison; the designated primary
  target is the ROCm 10 table in [benchmarks.md](../benchmarks.md).

## Measured (2026-08-30, `llm170 gguf-dump` — UD-Q4_K_XL)

- **866 tensors**, data slack 0 B (parser position arithmetic matches the file
  exactly).
- `rope.freq_base`=1e7 (**stored as f32** — not an integer), 
  `attention.key_length`=256, vocab 248320.
- `general.sampling.*` defaults embedded (top_k 20 / top_p 0.95 / temp 1) —
  usable as engine sampler defaults.
- UD-Q4_K_XL type mix: q5_K 45.2% · iq4_xs 17.8% · q4_K 17.4% · q6_K 16.3% ·
  q8_0 0.75% · iq4_nl 6 tensors · trace iq3_s/q3_K · f32 360 tensors
  (norms & small weights).
