# 의사결정 기록 (ADR)

역순 없음. 번경 시 갱신 + 날짜.

## ADR-0001 — 순수 Rust 단일 언어 (2026-08-30)

**결정**: 엔진 전체를 순수 Rust로 개발. C/C++ 커널 소스·RTC 문자열·CMake 툴체인 없음.
**배경**: 사용자 방침. 혼자 개발하므로 언어/툴체인 이중화 비용이 이익을 초과.
**기각**: 2026-08-30 조사 권고(Rust 코어 + CUDA 문법 C++ 커널 + NVRTC/hipRTC) — [source/research/2026-08-30-rust-gpu-bindings.md]. 사실 자료는 유효, 권고만 기각.
**파급**: GPU 커널 경로는 wgpu(WGSL) 또는 rust-gpu(.rs→SPIR-V) 중 선택(ADR-0004 시점). CUDA 네이티브는 cuda-oxide 성숙 대기.

## ADR-0002 — 모드 체계 universal / cmp-stock / cmp-unlocked (2026-08-30)

**결정**: 3 모드. 런타임 플래그 + 커널 변형 + 메모리 프로파일. 코어는 모드 무관.
**배경**: CMP 순정(eFUSE 스로틀·8GB)과 언락(풀레이트·40/64GB)이 요구하는 커널 전략이 상이. 범용 장비 검증 인프라 겸용.

## ADR-0003 — 언락 1급 시나리오 채택 (2026-08-30)

**결정**: 40/64GB 언락을 설계 전제로 승격. 단 `cmp-stock`(8GB) 동작도 유지.
**배경**: 사용자 확인(40GB/64GB 가능). cmpunlocker(2026) 사례. 비공식이므로 폴백 유지.
**메모**: 컴퓨트 언락(스로틀 해제) 실측 전까지 `cmp-unlocked` 커널도 half2 기본값.

## ADR-0004 — CPU 참조 백엔드 선행 (2026-08-30)

**결정**: GPU 백엔드보다 CPU(순수 Rust) 참조 구현을 먼저 완성. 모든 golden 테스트의 기준.
**배경**: GPU 없이 검증 가능, 수치 정확성 분리, 커널 언어 결정을 뒤로 미룰 수 있음.

## ADR-0005 — strict FP + `mul_add` 금지 (2026-08-30)

**결정**: 핫패스 `f32::mul_add` 금지, fast-math 계열 플래그 금지. 코드젠 FMA 부재 CI/프로파일러 검증.
**배경**: Rust strict FP는 암시 contraction이 없어 mul+add 분리가 자동 성립 → cmp-stock 32× 페널티 회피가 공짜.

## ADR-0006 — 문서 구조 source/ · docs/ (2026-08-30)

**결정**: `docs/`(프로젝트 문서, 추적) / `source/`(외부 참조 원본, 추적) / `plans/`(gitignored). `.gitignore`에서 docs 제거.
**배경**: 조사 원본과 프로젝트 산출물 분리, 문서의 버전 추적.

## ADR-0007 — 구현 순서 qwen35 → qwen4exp (2026-08-30)

**결정**: dense qwen35로 전 경로(파서→CPU 추론→프로파일러) 완성 후 qwen4exp 확장.
**배경**: GDN/IMROPE/MTP 하위구조 공유. qwen4exp 고유(HC/QSA/PLE/MoE)는 상위 확장.

## ADR-0008 — 범용 개발은 ROCm 대상 · 코드 이관 흐름 (2026-08-30)

**결정**: `universal` 모드의 GPU 개발·검증 대상은 **ROCm**(이 PC gfx1151). 충분히 검증되면 사용자가 코드를 170HX 장비로 **이관해 그 장비에서 이어 개발**. GPU 커널 구현 기술(cubecl-hip 등)은 CPU 참조 완료 후 결정하되 HIP/ROCm API 면 우선.
**배경**: 사용자 지시. 개발기가 ROCm 표준 환경. Backend trait이 HIP/CUDA 유사 API 양측을 수용하면 이관 비용 최소화.
**요구사항**: debug 빌드는 개발자가 **전 요소를 파악**할 수 있어야 함 — 계측(자체 프로파일러) + 구조 덤프가 기본 탑재.
**상태(2026-08-30)**: 워크스페이스 + GGUF v3 파서 + `gguf-dump` + 프로파일러 v0 완료. 테스트 6/6(합성 4 + 실측 27B/Flash split1 2). 실측으로 리서치 미확정 항목 해소 — [models/qwen4exp.md](models/qwen4exp.md) §미확정 해소.
