# eval ledger — 통계 품질 계측 (PPL · teacher-forced argmax 일치율)

Same table rule as README/4090.md — **values inside tables only**.
목적: 연산 경로 후보의 품질 편차 계측기(방침: 경로는 모듈당 하나·fast
기본 — 후보는 채택(골든 재동결) 또는 삭제). 기본 경로 골든과 독립.

## Environment

| item | value |
|---|---|
| GPU | NVIDIA RTX 4090 (sm_89, 24 GB) |
| probe | `llm170 w4a16-eval ppl --ctx 8192` (t=1 디코드 순회, head GPU) |
| corpus-en | benchmark/eval/corpus-en.txt (README 동결 3061B) |
| corpus-ko | benchmark/eval/corpus-ko.txt (AGENTS 동결 7785B) |
| baseline commit | 252dfc25 |

## 기준선 (2026-10-10)

| 모델 | 코퍼스 | 토큰 | NLL평균 | PPL | ms/토큰 |
|---|---|---|---|---|---|
| 27B g128 | en | 1046 | 1.772501 | 5.885553 | 23.4 |
| 27B g128 | ko | 2606 | 2.153074 | 8.611288 | 22.5 |
| 35B g32 MoE | en | 1047 | 2.748550 | 15.619966 | 11.8 |
| 35B g32 MoE | ko | 2592 | 3.028909 | 20.674659 | 10.0 |

## 결정성 (동일 빌드 2회, 27B en)

| argmax 일치 | NLL 델타 |
|---|---|
| 100.000% (1045/1045) | +0.0000% |

## 후보 편차 (채택/기각 판정 기록)

| 후보 | 모델 | 코퍼스 | PPL 델타 | argmax 일치율 | 판정 |
|---|---|---|---|---|---|
| — | — | — | — | — | — |
