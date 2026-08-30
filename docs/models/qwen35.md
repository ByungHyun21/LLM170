# qwen35 — Qwen3.8-27B 구현 스펙

Dense 하이브리드. 원본 상세(라인 근거 포함): [source/research/2026-08-30-qwen35-qwen4exp-arch.md](../source/research/2026-08-30-qwen35-qwen4exp-arch.md) §1. Ground truth: `~/local_llm-runtimes/weights/Qwen3.8-27B/config.json`, `~/local_llm/llama.cpp/src/models/qwen35.cpp`.

## 하이퍼파라미터

- 64층: `12×(3×(GDN→FFN) + 1×(GatedAttention→FFN))`, `full_attention_interval=4` → full-attn il∈{3,7,…,63}(16층), GDN 48층. GGUF `block_count=65`(64본체 + MTP blk.64).
- n_embd 5120, FFN 17408 (SwiGLU), vocab 248320, rms_eps 1e-6, ctx 262144.
- MTP: `mtp_num_hidden_layers=1`.

## 층 구조

**잔차**: `h = h + attn(rms(h)); h = h + ffn(rms_post(h))` — FFN 분기는 **post**-attn norm (`attn_post_norm`).

**Gated Attention (16층)** — Q 24 / KV 4 heads, head_dim 256:
- `wq` = fused **Q‖gate** [5120, 12288], 헤드별 인터리브(스트라이드 2·head_dim).
- Q/K 각각 per-head RMSNorm [256] (`attn_q_norm`/`attn_k_norm`).
- **IMROPE**: rotary dim 64(32쌍), sections `[11,11,10,0]`, t/h/w cyclic 할당, freq_base 1e7. kq_scale = 1/√256.
- 출력: `wo` 통과 전 `attn_out ⊙ sigmoid(gate)`.
- KV: f16 기준 **4 KiB/층/tok → 64 KiB/tok** (16층).

**Gated DeltaNet (48층)** — K 16 heads(group) / V 48 heads, head_dim 128:
- `attn_qkv` [5120,10240] (q‖k‖v, **V-head가 conversion에서 grouped→tiled 재배열됨**) + `attn_gate`(z) [5120,6144].
- `ssm_conv1d` [4,10240] depthwise conv + SiLU, 3-토큰 rolling state.
- q,k L2-norm → beta = sigmoid(`ssm_beta`); gate g = softplus(`ssm_alpha`+`ssm_dt.bias`)·`ssm_a`, `ssm_a` = −exp(A_log) (conversion에서 적용됨).
- GDN recurrence: 상태 S=[128,128,48] f32 = 3 MiB/층/seq → **144 MiB/seq** + conv state 5.6 MiB/seq.
- out = rms_norm(core) ⊙ **SiLU**(z) (qwen35는 SiLU, qwen4exp는 sigmoid — 주의).

**MTP/NextN (blk.64, draft-mtp 전용 로드)**: input = concat(rms_enorm(embd(t+1)), rms_hnorm(h)) → `eh_proj` [10240,5120] → gated-attn 블록 1개 → FFN → 공유 output_norm/LM head. 실측 draft acceptance 0.47–0.85.

## GGUF 계약

- 메타: 공통 블록 + GDN 블록(`ssm.conv_kernel=4, ssm.inner_size=6144, ssm.state_size=128, ssm.time_step_rank=48, ssm.group_count=16`) + `nextn_predict_layers=1`.
- 텐서명: `token_embd/output{,_norm}`, 블록별 `attn_norm`, `attn_post_norm`, full-attn `attn_{q,k,v,output,q_norm,k_norm}`, GDN `attn_{qkv,gate}`, `ssm_{conv1d,dt.bias,a,beta,alpha,norm,out}`. MTP: `blk.64.nextn.*`.
- **Conversion 트랜스폼** (GGUF에 이미 적용 — HF 원본 로드 시에만 재현 필요): in_proj_qkvz 분해, V-head tiled 재배열, `A_log→−exp(A_log)`, RMSNorm 감마는 **w+1** 로 저장.
- 토크나이저: BPE pre `qwen35` (`[\p{L}\p{M}]+` 결합 문자 처리), bos=eos=248044.

## 구현 메모

- GDN fused 단일 op(스냅샷 패킹) + chunked 경로(solve_tri/cumsum/tri) 병행 — [architecture.md](../architecture.md) 커널 목록.
- KV는 16층만 — 하이브리드 메모리(층별 캐시 타입) 설계의 기준 케이스.
- 로컬 운영 기준선: Q4+MTP tg 15.5–24.4 t/s, pp 283–556 t/s (llama.cpp, np4×262K, 2026-08-18) — 상대 비교용.

## 실측 (2026-08-30, `llm170 gguf-dump` — UD-Q4_K_XL)

- 텐서 **866개**, 데이터 slack 0 B (파서 위치 계산이 파일과 정확히 정합).
- `rope.freq_base`=1e7(**f32 저장** — 정수 아님 주의), `attention.key_length`=256, vocab 248320.
- `general.sampling.*` 기본값 내장(top_k 20 / top_p 0.95 / temp 1) — 엔진 샘플러 기본값 소스로 사용 가능.
- UD-Q4_K_XL 타입 믹스: q5_K 45.2% · iq4_xs 17.8% · q4_K 17.4% · q6_K 16.3% · q8_0 0.75% · iq4_nl 6텐서 · iq3_s/q3_K 극소량 · f32 360텐서(norm·소형).
