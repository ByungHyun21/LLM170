# qwen4exp — Qwen3.8-Flash-Next (125B-A6B) 구현 스펙

MoE 하이브리드. 원본 상세: [source/research/2026-08-30-qwen35-qwen4exp-arch.md](../source/research/2026-08-30-qwen35-qwen4exp-arch.md) §2. 코드: `~/local_llm-runtimes/qwen4exp/src/models/qwen4exp.cpp`.

## 하이퍼파라미터

- 48층: `12×(3×(GDN→MoE) + 1×(QSA→MoE))` — QSA il∈{3,7,…,47}(12층), GDN 36층.
- n_embd 2560, vocab 248320, ctx 262144. 파라미터: 본체 125B(A6B) + PLE 51B + MTP 4B(**UD GGUF에서 drop**).
- hyper_connection: count=4, low_rank=320.

## qwen35 대비 차이 요약

1. **GDN**: 동일 모듈, 단 z-gate가 **sigmoid**.
2. **QSA** (Qwen Sparse Attention, 12층): gated attention(24 Q / 2 KV, 256 dim, IMROPE 64) + 희소화:
   - per-token **indexer**(MQA 4q+1k, 128 dim)가 raw key를 어텐션 캐시와 cell-for-cell 미러되는 제3의 캐시에 적립.
   - 캐시 key를 `compress_ratio`(=4) 블록별 mean-pool → RMS-norm + rope → indexer Q와 스코어(ReLU, 헤드 합) → 블록당 스코어 → `top_k`(**실측 메타 = 2048**; 선택 폭 = min(n_kv, top_k + ratio − 1)) → 마스크 → **마스크 밀집 GQA 어텐션**.
   - KV: 12층 × 2 KV head → f16 **24 KiB/tok** (+indexer 3 KiB/tok).
3. **Hyper-connection 잔차** (모든 norm 대체): 상태 = **4 평행 스트림** [2560×4, T]. read = grouped RMSNorm(γ, w+1 저장) + low-rank 게이트 + 스트림 평균. write = `s += out·(2·sigmoid(inject))`. 층당 HC 2개 + 최종 `output_hc_*` — **output_norm 텐서 없음**. 초기 상태 = 임베딩×4.
4. **MoE 매층**: 512전문가 중 10 routed + 1 shared(sigmoid 게이트). 라우팅 = softmax → argsort_top_k 10 → 가중치 gather·정규화 → `mul_mat_id`. n_ff_exp 640 · shared 640 (실측 확정: `expert_{,shared_}feed_forward_length`).
5. **PLE n-gram 해시 임베딩** (blk.1 단일층):
   - `per_layer_token_embd.weight` — **실측 `[160, 320,001,536]` iq4_nl, 26.82 GiB (4.5bpw ≈ 51.2B 파라미터)**. 행 dim 160, 총 3.2억 행 = Σ `ple.head_vocab_sizes`(16 heads × ~2000만). per-token gather 16행 × 160 = 2560 flatten. (구 리서치의 `[2560, 20M]` 표기 정정) — **오프로드 전제 설계**(랜덤 GET_ROWS, hot row page cache).
   - 해시는 **호스트 u64 연산**(mul^xor, mod vocab+offset), bigram+trigram, EOS 윈도 리셋.
   - key/value 프로젝션 → `sigmoid(sgn(s)·√|s|)` 게이트 → 4스트림 방송 → ngram-dilated depthwise conv → 잔차 가산.

## GGUF 계약 (qwen35 공통 제외분)

- 추가 메타: `hyper_connection.{count,low_rank}`, `attention.indexer.{head_count=4,key_length=128,top_k=2048}`, `attention.compress_ratios[48]`(QSA층 4, else 0), `ple.{layers=[1], ngram_size=3, heads_per_ngram=8, conv_kernel=4, layer_multipliers[u64], head_offsets[16,u64], head_vocab_sizes[16,u64], eos_token_id, image_token_id}`, `embedding_length_per_layer_input=160`, `expert_count=512`, `expert_used_count=10`.
- 텐서: `hc_{attn,ffn}_{norm,down,up,inject}`, `output_hc_{norm,down,up}`, `indexer.{q_proj,k_proj,q_norm,k_norm}`, `ple_{key,value,norm_key,norm_query,norm_conv,conv1d}`, `per_layer_token_embd.weight`, `ffn_gate_inp{,_shexp}`, `ffn_{gate,up,down}_exps`(또는 merged `ffn_gate_up_exps`), `ffn_{gate,up,down}_shexp`.
- 4-split GGUF(`split.no/count`, 파일명 패턴 보존 필수). mmproj 미포함(VL 필요 시 별도).

## 미확정 해소 (2026-08-30 실측, `llm170 gguf-dump` — UD-Q4_K_XL 4-split)

- `rope.freq_base` = **1e7 확정**(f32 저장).
- shared expert FFN = **640 확정**, `ple.conv_kernel` = 4(메타 존재).
- `attention.indexer.top_k` = **2048** (research의 512 정정).
- split 구조: **part 1(no=0)은 메타 전용(텐서 0)**, `split.tensors.count`=1224는 전체 모델 텐서 수(parts 2–4에 분산).
- expert 스택 3D `[640, 2560, 512]` — 역할별 타입 혼합(실측 예: down `q5_1`, gate/up `q4_K`, router `f32`).
- 타입 믹스(parts 2–4 합산 근사): iq4_nl 26.82 GiB(PLE 단일) · q4_K ~41.3 GiB · q5_1 ~25.2 GiB · q8_0 ~9.0 GiB · q5_K 1.1 GiB · f32/bf16 소량.

## 구현 메모

- 상태: GDN S 108 MiB/seq + conv 4.2 MiB + PLE row. KV 24 KiB/tok(f16 고정 — PR 버그로 KV 양자화 불가, 우리 엔진은 자유).
- PLE 테이블 조회는 메모리 계층 설계의 핵심 병목(NVMe mmap + `MADV_RANDOM` + page cache 패턴이 검증된 원형).
- 로컬 운영 기준선(HIP 7.2.2): pp 178–237 t/s(슬롯당), tg 11.6–15.1 t/s, ROCm 10+master: pp 272–468, tg 11.9–19.6 (np2×262K, 2026-08-27/29).
