# qwen4exp — Qwen3.8-Flash-Next (125B-A6B) Implementation Spec

MoE hybrid. Original detail:
[source/research/2026-08-30-qwen35-qwen4exp-arch.md](../source/research/2026-08-30-qwen35-qwen4exp-arch.md) §2.
Reference implementation (read-only during development): the upstream
`qwen4exp` runtime sources.

## Hyperparameters

- 48 layers: `12×(3×(GDN→MoE) + 1×(QSA→MoE))` — QSA layers il∈{3,7,…,47} (12
  layers), GDN 36 layers.
- n_embd 2560, vocab 248320, ctx 262144. Parameters: 125 B body (A6B) + PLE
  51 B + MTP 4 B (**dropped from the UD GGUF**).
- hyper_connection: count=4, low_rank=320.

## Differences from qwen35

1. **GDN**: identical module except the z-gate is **sigmoid**.
2. **QSA** (Qwen Sparse Attention, 12 layers): gated attention (24 Q / 2 KV,
   head_dim 256, IMROPE 64) + sparsification:
   - Per-token **indexer** (MQA 4q+1k, 128-dim) accumulates raw keys into a
     third cache that mirrors the attention cache cell-for-cell.
   - Cached keys are mean-pooled per `compress_ratio` (=4) block → RMS-norm +
     rope → dot with indexer Q → scores (ReLU, head-summed) → one score per
     block → `top_k` (**measured meta = 2048**; selection width =
     min(n_kv, top_k + ratio − 1)) → mask → **masked dense GQA attention**.
   - KV: 12 layers × 2 KV heads → f16 **24 KiB/token** (+ indexer 3 KiB/token).
3. **Hyper-connection residual** (replaces every norm): state = **4 parallel
   streams** [2560×4, T]. read = grouped RMSNorm (γ stored as w+1) + low-rank
   gate + stream mean. write = `s += out·(2·sigmoid(inject))`. Two HC mixes per
   layer + final `output_hc_*` — **no output_norm tensor**. Initial state =
   embedding ×4.
4. **MoE every layer**: 10 routed of 512 experts + 1 shared (sigmoid gate).
   Routing = softmax → top-10 → gathered weights ·normalized → per-expert GEMM.
   n_ff_exp 640 · shared 640 (measured: `expert_{,shared_}feed_forward_length`).
5. **PLE n-gram hash embedding** (single layer, blk.1):
   - `per_layer_token_embd.weight` — measured `[160, 320,001,536]` iq4_nl,
     26.82 GiB (4.5 bpw ≈ 51.2 B parameters). Row dim 160, 320 M total rows =
     Σ `ple.head_vocab_sizes` (16 heads × ~20 M). Per token: gather 16 rows ×
     160 = 2560 flattened. **Designed for offloading** (random GET_ROWS, hot
     rows in page cache).
   - Hashing is **host-side u64** (mul^xor, mod vocab+offset), bigram+trigram,
     EOS window reset.
   - key/value projections → `sigmoid(sgn(s)·√|s|)` gate → broadcast to the 4
     streams → ngram-dilated depthwise conv → two-path residual add.

## GGUF Contract (additions beyond the qwen35 common block)

- Extra metadata: `hyper_connection.{count,low_rank}`,
  `attention.indexer.{head_count=4,key_length=128,top_k=2048}`,
  `attention.compress_ratios[48]` (4 on QSA layers, else 0),
  `ple.{layers=[1], ngram_size=3, heads_per_ngram=8, conv_kernel=4,
  layer_multipliers[u64], head_offsets[16,u64], head_vocab_sizes[16,u64],
  eos_token_id, image_token_id}`, `embedding_length_per_layer_input=160`,
  `expert_count=512`, `expert_used_count=10`.
- Tensors: `hc_{attn,ffn}_{norm,down,up,inject}`, `output_hc_{norm,down,up}`,
  `indexer.{q,k}_proj`, `indexer.{q,k}_norm`, `ple_{key,value,norm_key,
  norm_query,norm_conv,conv1d}`, `per_layer_token_embd.weight`,
  `ffn_gate_inp{,_shexp}`, `ffn_{gate,up,down}_exps` (3D expert stacks),
  `ffn_{gate,up,down}_shexp`.
- 4-split GGUF (`split.no/count`, filename pattern must be preserved). Part 1
  (no=0) is metadata-only; `split.tensors.count`=1224 spread across parts 2–4.
- Expert stacks are 3D `[ff, n_embd, 512]` with per-role type mixing (measured
  example: down `q5_1`, gate/up `q4_K`, router `f32`).
- Type mix (parts 2–4, approximate): iq4_nl 26.82 GiB (the PLE table alone) ·
  q4_K ~41.3 GiB · q5_1 ~25.2 GiB · q8_0 ~9.0 GiB · q5_K 1.1 GiB · trace
  f32/bf16.

## Implementation Notes

- State: GDN S 108 MiB/seq + conv 4.2 MiB + PLE conv history. KV
  24 KiB/token (f16 fixed upstream — a PR bug prevents KV quantization there;
  this engine is free to choose).
- The PLE table lookup is the defining memory-hierarchy bottleneck: NVMe mmap +
  `MADV_RANDOM` + page-cache-friendly hot rows is the validated pattern.
- QSA indexer: pooled + normed + roped **block keys are computed once per
  block** and reused by every token in it (indexer cost O(T^2) -> O(T)).
- PLE decode prefetch: a side thread issues the gather for the predicted
  next token during the current step; the following step joins it. (A
  prefill-wide prefetch was tried and reverted — row-set mismatch.)
- Decode frame (default on, ADR-0017): activations device-resident, kernels
  chained by handle (hc/GDN/MoE/head framed; PLE host-hash bridge + QSA
  value bridge remain, ~14 syncs/step). Per-sequence handle sets support
  parallel decode; on residency failure the engine falls back to the
  per-op value path permanently.
- Value-path decode profile (hot cache, 2026-08-31): moe 1.3 ms · gdn
  0.45 ms · hc 0.22 ms · qsa 0.1 ms per token.
- Local operating baseline (HIP 7.2.2): pp 178–237 t/s (per slot), tg
  11.6–15.1 t/s; ROCm 10+master: pp 272–468, tg 11.9–19.6 (np2×262K,
  2026-08-27/29).

## Verification Status (2026-09-02)

- Reference: the upstream qwen4exp runtime (PR #27742 build) with
  `-ot per_layer_token_embd=CPU --load-mode mmap -fa on` (f16 KV), temp 0 —
  run non-coexisting (reference server collects first, then stops; both
  engines resident would contend for the same VRAM).
- Real-model matrix (Vulkan): single_ko 24/24 exact · single_short /
  long_gen48 near-tie (gap 0.01 nat) · single_code near-tie (0.12) · np2
  state isolation 25/25 ×2. Long single (2,311 tok) and long2 (1,904 tok)
  24/24 exact each (measured on the immediately preceding binary; re-run on
  the final binary queued with long+np2, pending device-memory headroom —
  Vulkan places weights in host GTT on the dev iGPU).
- Synthetic tiny4 generator (`scripts/make_tiny4.py`) — model-volume
  independent e2e covering every combination: CPU == GPU 26/26, np2
  26/26 ×2, long 2000+ 25/25, long+np2 25/25 ×2. The decode frame is
  hash-identical to the value path on this harness.
- GPU-only paths, each cross-validated against the CPU reference before
  entering the engine: GDN AR kernel (all dims 16–128), single-launch
  chunked GDN prefill kernel, batched MoE down (one launch, K experts),
  token-expert grouped prefill GEMM, element-wise set (norm kernels
  bit-exact), QSA block-key cache, decode frame.
- History note: a t=1 MoE fast-path regression (routed-expert contribution
  zero on the default decode path) was found 2026-09-01 by diffing against
  the frame path; both CPU and GPU value paths shared it, so self-consistent
  checks had passed. All numbers above are post-fix.
