# docs/source — 코드 1:1 측정 기록


**Runtime default (2026-09-14)**: benchmarks and gates run on the TheRock
ROCm 10.0.0 userspace (`/opt/rocm-10.0.0/install/lib` via `LD_LIBRARY_PATH`,
soname-compatible, no relink; system 7.2.2 as fallback). Numbers quoted before
this date were measured on 7.2.2 unless noted.
`docs/benchmarks.md`가 350 KB를 넘어 커지면서, **파일 단위로 대응하는 문서**를 여기로 분리한다.
목적은 두 가지다: (1) 코드를 고칠 때 그 파일에 대한 실측 근거를 바로 찾을 수 있고,
(2) `benchmarks.md`는 **요약 지표와 모델 간 비교**만 남겨 짧게 유지한다.

## 규약

- 경로는 소스를 그대로 미러링한다: `crates/backend-gpu/src/rawhip/q4acc.rs` →
  `docs/source/backend-gpu/rawhip/q4acc.md`.
- 각 문서 상단에 원본 소스 경로를 명시한다.
- **수치와 표는 옮길 때 그대로 보존한다.** 요약·해석을 새로 쓰지 않는다.
- 항목 제목에 날짜를 유지한다(`(2026-09-14)`). 같은 주제의 후속 측정은 새 항목으로
  덧붙이고, 앞선 수치를 뒤집으면 **어느 쪽이 최신인지 명시**한다(예: "supersedes ...").
- 기각된 실험도 남긴다 — 왜 안 되는지가 다음 시도의 비용을 줄인다.
- 프로브·플래그 사용법(`LLM170_*`)은 그 플래그를 구현한 파일의 문서에 적는다.

## 현재 분리된 문서

| 문서 | 소스 | 내용 |
|---|---|---|
| `backend-gpu/rawhip/q4acc.md` | q4acc.rs | QSA 어텐션(분할·6헤드), MoE 그룹화, Q4_K MMQ 타일, f16 dequant, 프리필 GEMM |
| `backend-gpu/rawhip/mod.md` | mod.rs | KTRACE 이벤트 페어링·런치 순서 덤프, 계측기 주의 |
| `backend-gpu/rawhip/decode.md` | decode.rs | 27B 디코드/프리필 분해, GDN |
| `core/qwen4exp/stages/qsa.md` | stages/qsa.rs | 스테이지 타이머, 성분 분해 |
| `core/qwen4exp/stages/moe.md` | stages/moe.rs | 라우팅 비용 |

`docs/benchmarks.md`에는 **요약 지표**(§Primary target, 모델별 pp/tg 표)와 아직 옮기지 않은
이력 항목이 남아 있다. 파일 귀속이 분명한 항목을 만나면 해당 문서로 옮긴다.

## 계측기 (실측 시 먼저 확인)

| 플래그 | 대상 | 주의 |
|---|---|---|
| `LLM170_KTRACE=1` | 커널별 디바이스 시간 + 런치 갭 | 2026-09-14 수정: 고정 stride-2 페어링. 그 전 수치는 배치 형상에서 최대 4배 과소 |
| `LLM170_KTRACE_SEQ=N` | 런치 순서 덤프 | 갭의 주인은 **후속** op이므로 순서가 필요 |
| `LLM170_PP_PROF=1` | raw 경로 섹션 마크 | 프레임 경로(qwen4exp)에서는 무출력 |
| `LLM170_Q4ACC_TIME=1` | 가속기 호출 upload/quant/launch/d2h | d2h = 앞서 큐에 넣은 디바이스 작업 대기 |
| `LLM170_MOE_TIME=1` | MoE phase(weight/group/gemms) | weight는 최초 업로드(48x3=144회)로 콜드 |
| `LLM170_Q4_TIME=1` / `LLM170_FRAME_TIME=1` | QSA 스테이지·프레임 | **t 라벨 필수** — 없으면 프리필/디코드 평균이 오염됨 |
| `LLM170_NOLAUNCH=1` | 런치 생략 | 배치에서 신뢰 불가(27B에서 벽시계와 모순) |

## 검증 게이트 (변경 시 필수)

- diverse 스트림: `5513 248046 198 248045 74455 198 248068 198 760 1156 579 1876 7701 310 381 7132 36412`
- 27B: pp512 warm ~1,43x ms, tg8 ~87 ms/스텝 (변하지 않아야 함)
- `cargo test --release --workspace` 12/12
