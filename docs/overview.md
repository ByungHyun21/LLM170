# LLM170 프로젝트 개요

CMP 170HX를 최종 타깃으로 하는 **순수 Rust** LLM 추론 엔진. llama.cpp/ggml 의존 없이 처음부터 자체 구현.

## 목표와 우선순위

1. **1순위 — CMP 170HX** (GA100, sm_80): 언락(40/64GB) 전제 하에 최대 성능.
2. **2순위 — 범용성**: 임의 장비(CPU, 임의 GPU)에서 동작. 혼자 개발하므로 범용 모드가 개발 속도·검증 인프라를 담당.
3. **지금 이 PC에서 하는 일**: **범용(universal) 모드 개발**. CMP 170HX는 접근 불가.

## 모드 체계

| 모드 | 대상 | 전제 | 커널 전략 |
|---|---|---|---|
| `universal` | 임의 장비 (CPU / 범용 GPU) | 제약 없음 | 이식성 우선. 이 PC가 개발 기준환경 |
| `cmp-stock` | CMP 170HX 순정 (8GB, eFUSE 스로틀) | FFMA 금지·텐서코어 금지 | half2/INT32, FMA 분해 |
| `cmp-unlocked` | CMP 170HX 언락 (40~64GB, compute unlock) | 풀레이트 가능 | 성능 최우선 (언락 검증 후 확정) |

- 모드는 **런타임 플래그 + 커널 변형 선택**으로 구현. 엔진 코어(모델·스케줄러·로더)는 모드 무공유.
- 메모리 예산은 모드별 프로파일: stock ~7 GiB / unlocked 40~64 GiB → [하드웨어/CMP 170HX](hardware/cmp170hx.md) 참조.

## 기준 모델 (GGUF, `~/local_llm/models/`)

| 모델 | 아키텍처 | 구성 | 크기 | 스펙 문서 |
|---|---|---|---|---|
| Qwen3.8-27B | `qwen35` dense hybrid | 64층 = 12×(3×GDN + 1×Gated Attn) + MTP 1층, ctx 262144, VL | Q4 16.5 / Q6 24.1 / Q8 31.4 GB | [models/qwen35.md](models/qwen35.md) |
| Qwen3.8-Flash-Next | `qwen4exp` MoE hybrid | 48층 = 12×(3×GDN + 1×QSA), hc=4 잔차, MoE 512e(A6B), PLE 51B, ctx 262144 | UD-Q4 111.3 GB (4-split) | [models/qwen4exp.md](models/qwen4exp.md) |

두 모델 모두 GDN(Gated DeltaNet) 선형어텐션 기반 하이브리드 — 구현 순서상 qwen35 먼저(참조 단순), qwen4exp가 확장.

## 개발 원칙

- **순수 Rust** — C/C++ 커널 소스·툴체인 없음 ([decisions.md](decisions.md) ADR-0001).
- **debug 빌드 = 세세한 프로파일링**: 자체 경량 프로파일러가 계측 주체 (Nsight가 CMP를 지원 안 함).
- **release 빌드 = 배포본**.
- 증거 기반 문화: 성능 수치는 조건(컨텍스트/배치/양자화/드라이버) 명시, 재현 절차 동봉.

## 문서 구조

- `docs/` — 프로젝트 문서 (추적됨)
- `source/` — 외부 참조 원본 자료: 조사 리포트, 발췌 (추적됨, 산출물 아님)
- `plans/` — 작업 계획 (gitignored)
- 운영 이슈는 각 모델/주제 문서 하단 또는 별도 ISSUES 패턴 (날짜 헤더 + 증상/원인/검증)

## 현재 단계 (2026-08-30)

하드웨어 전 사전 구축. 완료: 제약 조사·아키텍처 방향·문서 체계. 다음: 범용 모드 구현 착수 — [architecture.md](architecture.md) §단계별 계획.
