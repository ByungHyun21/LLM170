# CMP 170HX — 타깃 하드웨어 명세

배포 1순위 장비. 사실 근거: [source/research/2026-08-30-cmp170hx-constraints.md](../source/research/2026-08-30-cmp170hx-constraints.md) (출처 URL 포함).

## 기본 사양

- GA100, **sm_80**, 70 SM / 4480 FP32 core / 280 tensor core. CUDA 11.0–13.x, 표준 리눅스 드라이버(535-server/570/610 검증)로 정상 동작 — 특수 드라이버 불필요.
- HBM2e 8GB (순정, 7.7 GiB 가용) / 4096-bit / 1493 이론·1355 실측 GB/s. ECC off.
- PCIe **1.1 x4** (~0.85 GB/s), BAR1 64 MiB, ReBAR 없음. NVLink 미장착.
- Headless(출력 없음), 수동냉각, **전원 = 8핀 EPS 12V × 2** (PCIe 8핀 아님), 250W TDP(실측 170–230W).
- FP64 사실상 없음 (94–195 GFLOPS). MIG 불가.

## 스로틀 매트릭스 (순정 eFUSE) — `cmp-stock` 모드의 커널 규칙

| 경로 | 실측 | 상태 |
|---|---|---|
| FP32 FFMA (fused) | 0.39 TFLOPS (1/32) | **금지** — FMA 명시 분해(mul+add) |
| FP32 mul/add 분리 | 6.2–6.3 TFLOPS | 사용 가능 |
| **FP16 half2 (CUDA core)** | **42 TFLOPS (83%)** | **최고 부동소수 경로** |
| BF16 CUDA core | ~20 TFLOPS | 사용 가능 |
| 텐서코어 HMMA/TF32 | 6.07 / 3.04 TFLOPS (12%) | **금지** (CUDA core보다 느림) |
| INT32 mul/add | 12.5 TIOPS | 무스로틀 |
| FP64 | ~0.1 TFLOPS | 금지 |

- 특성치 대체: `__expf`/`__fdividef`/`rsqrtf`. 커뮤니티 선례(eastmoe 패치): llama.cpp pp +105% / tg +83%.
- 성능 특성: **decode = 대역폭 바운드(A100급 강점), prefill = 컴퓨트 바운드(RTX 2060급)**.

## 언락 (2026-08-30 사용자 전제: 40GB/64GB 가능)

- cmpunlocker(2026, GSP/SEC2 익스플로잇, clean-room, 드라이버 610.43.03 계열): **메모리 언락 + 컴퓨트(스로틀) 언락**. 10GB 카드에서 40960 MiB 노출 사례 확인(2026-07-30). 64GB 언락 = 사용자 확인 사실.
- 설계 태도: `cmp-unlocked` 모드로 **1급 시나리오** 취용. 단 비공식이므로 `cmp-stock` 동작도 유지(모드 분리로 무상확보).
- 컴퓨트 언락 시 FFMA/텐서코어 스로틀 해제 가능성 → `cmp-unlocked` 커널은 풀레이트 경로 허용. **실측 전까지는 half2 경로를 기본값으로 두고 언락 실측 후 전환** (검증 항목 참조).

## 메모리 예산 × 기준 모델

| 구성 | 가용 | Qwen3.8-27B | Flash-Next (111.3GB + PLE 28GB) |
|---|---|---|---|
| `cmp-stock` | ~7.7 GiB | Q4(16.5GB)도 불가 → 텐서 오프로드 필수 | 대규모 오프로드 |
| `cmp-unlocked` 40GB | ~39 GiB | Q8(31.4GB)+KV 여유 | 불가 — 오프로드 |
| `cmp-unlocked` 64GB | ~63 GiB | 전 양자화 상주 | 본체 일부 상주 + PLE/일부 전문가 오프로드 |

- 오프로드 경로 제약: PCIe 1.1 x4 = 0.85 GB/s. NVMe mmap + page cache 패턴(기존 Flash-Next run.sh의 PLE 오프로드가 원형)이 합리적 기본값.
- 로드: BAR1 64MiB → whole-VRAM mmap 불가. 핀드 버퍼 청크 복사. 8GB 체 로드 ~10s+ (40/64GB는 비례 증가, 대역폭 동일하므로 60~90s [추정]).

## 검증 항목 (하드웨어 도착 시)

1. 언락 후 실측 스로틀 매트릭스 재측정 (FFMA/TC 풀레이트 여부) → `cmp-unlocked` 커널 전환 판정.
2. Vulkan compute 노출 여부 (순정/언락 각각) — 범용 모드 경로의 CMP 직결 여부.
3. 언락 메모리 안정성 soak (40/64GB 장기).
4. 냉각: 3000+ RPM 섀시 풍량 또는 A100 워터블록(Bykski N-TESLA-A100-X-V2 호환).
