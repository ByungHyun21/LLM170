# `crates/backend-gpu/src/rawhip/q4acc/value.rs` — measurement record

## 출력 스테이징 버퍼 재사용 (2026-09-14)

`run_prepared`가 호출마다 `vec![0.0f32; t * n_out]`을 새로 만들고 0으로 채운 뒤
d2h로 **전체를 덮어쓰고** 행별 Vec에 산포했다. 프리필(t=2048, n_out=6144)에서는
호출당 50MB — mm_group 하나가 5회 호출하므로 그룹마다 250MB의 할당+0-채움이
낭비였다. 이제 영속 버퍼(`ybuf`)를 재사용하고 **확장 시에만** 0을 채운다.

효과는 정직하게: pp2048 rep2 8,114ms(252.4 t/s)로 최고치와 같지만 **±5% 런간
변동 안**이라 측정으로 분리되지 않는다. 다만 제거된 작업(할당 + memset)은
확실하고 비트 동일이므로 유지한다 — 이 경로에서 "측정 불가"와 "중립"은 다르다.

### 가설 반증: 메모리 고갈이 아니다 — **잘못된 목적지 주소** (2026-09-14)

`LLM170_MEMDBG`로 h2d **직전** 여유 메모리를 찍었다(사후 조회는 sticky 오류로 0/0이므로):

```
# memdbg before h2d 8388608B dst=0x7412c7e00000: free=95732MB/98304MB
# memdbg before h2d 8388608B dst=0x7412c8600000: free=95732MB/98304MB
# memdbg before h2d 8388608B dst=0x7412c8e00000: free=95732MB/98304MB
# memdbg before h2d 2097152B dst=0x7412c9600000: free=95732MB/98304MB
# memdbg before h2d 8388608B dst=0x74128f400000: free=95254MB/98304MB   ← 실패
error: h2d 8388608B dst=0x74128f400000: 700
```

**여유 95GB** — 메모리 고갈 가설은 **기각**이다. 눈에 띄는 것은 주소다: 성공한 복사들은
`0x7412c7e…`~`0x7412c96…` 대역인데 실패한 것은 **`0x74128f40…`** 로 다른 영역이다.
즉 `dev_weight`가 **가리키는 디바이스 포인터가 유효하지 않다** — 캐시(`self.weights`,
호스트 포인터 키)가 참조하는 주소가 실제로는 해제/재할당되었거나 다른 영역의 주소다.

**다음 단계**: `dev_weight`의 캐시 수명과 `ctx.alloc`의 성격(영구인지, 풀이 재할당하는지)을
확인한다 — 특히 디바이스 그룹화 경로가 활성일 때 `xv`/프레임 버퍼 할당이 끼어들어
**가중치 대역의 주소가 바뀌는지**를 본다. 상한 5축과도, 메모리 예산과도 다른 축이다.

## VRAM 사용 방식 — llama.cpp와 다르다 (2026-09-14, 사용자 질문)

**질문**: 모델·ctx·최대 배치가 정해지면 필요한 VRAM은 고정인데, 우리도 llama.cpp처럼
한 번에 잡고 그 안에서만 재사용하는가?

**답**: 아니다. 실측(`LLM170_Q4ACC_STATS`)이 보여주는 것:

```
# q4acc: fxq   확장   337920 →   844800 B
# q4acc: yperm 확장   614400 →  2457600 B
# q4acc: xperm/gp/gi/texp/rperm/rperm2/rexp 확장 0 → N B   (첫 터치마다 지연 할당)
# q4acc: 업로드 450.0 MiB / 600.0 MiB (가중치 파이프라인 업로드)
```

세 가지가 다르다:

| | llama.cpp 방식 | 우리 (`GBuf::ensure`) |
|---|---|---|
| 시점 | 기동 시 **전부** 할당 | **지연** 할당(첫 터치) |
| 성장 | 없다(상한으로 잡음) | **재할당하고 옛 버퍼를 버린다** ✗ |
| 주소 | 고정 | **성장 시 바뀐다** ✗ |

`ensure`는 `bytes > self.bytes`이면 `ctx.alloc(bytes)`로 **새로 잡고 옛 포인터를
버린다**(`yperm`이 614,400 → 2,457,600으로 4배 커지는 것을 실측). 이는 두 가지를
낳는다: (1) 옛 버퍼 누수, (2) **그 주소를 캐시한 쪽이 해제된 메모리를 쓴다**.

**그리고 이것이 앞 절의 미스터리와 맞물린다**: 디바이스 그룹화를 켰을 때 실패한 h2d의
목적지 주소만 다른 대역(`0x74128f40…` vs 성공한 `0x7412c7e…`)이었고 메모리는 95GB
남아 있었다 — **성장으로 주소가 바뀐 버퍼를 상대편이 옛 주소로 쓰는 그림**과 일치한다.

**수정 방향**(llama.cpp 규율로): 모델·ctx·`t_max`가 정해지면 모든 풀의 상한이
결정되므로 **기동/첫 프레임에 상한으로 한 번 할당**하고, 이후 `ensure`는 **재할당하지
않는다**(초과 시 명시적 오류). 그러면 주소가 영구 고정되어 위 두 문제가 함께 사라진다.

### 보강: 그 누수는 **주소를 무효화하지 않는다** (2026-09-14)

`ensure`의 성장은 `ctx.alloc`으로 **새로 잡고 옛 포인터를 버릴 뿐 free하지 않는다**
(`hipMalloc`은 영구 — ADR-0014). 따라서 옛 주소는 **여전히 유효한 메모리**이고, 그것을
캐시한 쪽이 죽은 주소를 쓰는 일은 없다 — 앞 절에서 이 누수를 h2d 700의 원인 후보로
적었지만, **그 연결은 성립하지 않는다**(정정).

그래도 사용자 지적의 요점은 유효하다: llama.cpp는 모델·ctx·배치 상한으로 **한 번에
잡고 끝**인데 우리는 지연 할당 + 성장 재할당이라 (a) 주소가 바뀌고 (b) 옛 버퍼가
누수된다. 실측된 성장 횟수·크기는 작지만(`fxq` 2회, `yperm` 2회 등), **결정성**과
**예측 가능한 VRAM 예산**을 위해 상한 선할당으로 가는 것이 맞다. 그 작업은
`GBuf` 호출부가 `t` 대신 `t_max`(=`frame_begin`이 주는 그래프 t의 상한)를 넘기도록
바꾸는 것이 핵심이고, 그 전에 **어느 풀이 실제로 얼마나 성장하는지**를 이 로그로
확인할 수 있다(`LLM170_Q4ACC_STATS`).

### 적용: KV 풀 선할당 (2026-09-14, 사용자 지적)

지적: "모델이 정해지면 필요한 VRAM은 고정인데, vLLM처럼 한 번에 잡고 그 안에서만 쓰는 게
속도·안정성 모두 낫지 않은가?" — **맞다.** 실측 로그가 그 비용을 그대로 보여줬다:

```
# q4acc: cvv 확장 77824 → 79872 B     ← +2KB
# q4acc: ckv 확장 79872 → 81920 B     ← 매 스텝 재할당
# (세션당 성장 48회)
```

**구현**: `Accelerator` 트레이트에 `set_ctx_len`(기본 no-op)을 두고, `Engine4::with_acc`가
호스트 KV 길이에서 컨텍스트를 유도해 주입한다(`model/mod.rs:331`과 같은 식). `Q4Acc`는
`ctx_len`을 저장하고 **KV 풀(`ckv`/`cvv`)을 `ctx_len × n_kv × hd`로 선할당**한다.

**결과(실측)**: `ckv`/`cvv` 성장 **2회**(최초 할당만) — **매 스텝 재할당이 사라졌다.**
전체 성장은 48 → 32회로 줄었다. diverse 스트림 비트 동일, 테스트 12/12.

**남은 것**: 나머지 32회는 t에 비례하는 풀(`fpart`/`fxq`/`xperm`/`yperm`/`gxp`)이다 —
같은 방식으로 **t 상한을 주입**하면 사라진다(`frame_begin`이 그래프 t를 주므로 상한을
알 수 있거나, 엔진의 청크 설정에서 유도). 이것으로 "기동 시 전부 잡고 그 안에서만"에
한 걸음 더 가까워진다.

### 검증: 남은 성장은 워밍업뿐 (2026-09-14)

KV 선할당 후, 남은 32회 성장이 매 스텝인지 첫 프레임인지 확인:

| n-predict | 성장 횟수 |
|---|---|
| 8 | 32 |
| 64 | **33** |

**8배 긴 생성에 +1회** — 즉 남은 성장은 **첫 프레임(워밍업)에서 t가 커질 때 한 번씩**이고,
디코드 스텝에서는 **더 이상 재할당하지 않는다**. 사용자 지적("한 번 잡고 그 안에서만")의
핵심(매 스텝 주소가 바뀌는 것)은 **해소**되었다.

남은 개선(선택): t 비례 풀(`fpart`/`fxq`/`xperm`/`yperm`/`gxp`)을 **첫 프레임에 상한으로**
잡으면 32회도 사라진다 — `frame_begin`이 그래프 t를 주므로 그 최대값(또는 엔진의 청크
설정)을 주입하면 된다. 다만 워밍업 1회성이므로 우선순위는 낮다.

### 실패 복사의 정체 — `staged_upload`의 8 MiB 청크 (2026-09-14)

보고된 크기 8,388,608 B는 **`staged_upload`의 청크 상수와 정확히 일치**한다:

```rust
const CH: usize = 8 << 20;                       // = 8,388,608 ✓
src.file.read_exact_at(&mut stage[..n], off)?;
self.ctx.h2d(unsafe { dst.add(done) }, &stage[..n])?;   // ← 실패 지점
```

즉 실패한 복사는 **가중치 텐서를 파일에서 8 MiB씩 읽어 디바이스로 올리는 파이프라인
업로드**의 한 청크다(`dev_weight` → `staged_upload`). `done`은 `len` 안이므로 범위는
정상이고, 문제는 **`dst` 자체**(같은 함수에서 `ctx.alloc(w.data.len())`로 받은 포인터)다.

**다음 진단(제안)**: `RawCtx`가 `hipMalloc`한 범위를 **등록부**(start/end 목록)로 들고,
h2d 실패 시 “이 dst가 살아있는 할당 안인가?”를 함께 보고하면 이 미스터리가 즉시 닫힌다
(범위 밖이면 포인터가 낡았거나 다른 장치 주소이고, 범위 안이면 드라이버 문제다).
`h2d` 진단에 크기·주소·호출 사슬이 이미 있으므로 한 줄을 더하는 수준이다.

### ★ 진짜 원인: h2d는 **피해자**, 실패는 커널 런치 (2026-09-14)

두 진단이 미스터리를 닫았다.

1. **할당 등록부**(`alloc_span`)를 추가해 실패한 h2d의 목적지를 검사:
   `dst=0x71df61000000` → **`할당 내 [0x71df61000000,0x71df7d200000)`** (471MB 할당의 시작)
   — 즉 목적지는 **정상**이다.
2. **`HIP_LAUNCH_BLOCKING=1`** 로 각 런치를 동기화해 첫 실패를 드러냄:

```
error: io: rawhip: launch3: 700 kern=q4_gemm_q4k_ge gx=40 gy=519 gz=1 blk=256
```

**진짜 실패는 `q4_gemm_q4k_ge` 커널 런치**이고, 뒤따르는 가중치 h2d 700은 **오염된
스트림의 피해자**였다(sticky — `hipMemGetInfo`가 0/0을 돌려준 것과 같은 현상).

**배운 것**: (a) 스트리밍 백엔드에서 **첫 실패는 `HIP_LAUNCH_BLOCKING=1`로 찾는다** —
이후 오류는 전부 파생이다. (b) 진단은 *크기·주소·호출 사슬·할당 내 여부*까지 있어야
닫힌다(이 4개가 차례로 붙으면서 원인이 8MB h2d → 할당 내 주소 → 커널 런치로 좁혀졌다).
(c) 커널/호스트의 인자 수가 **커밋 사이에 어긋나면**(커널은 bound를 받는데 런처가 옛
시그니처) 조용히 스택 값을 읽는다 — 프로브 SIGSEGV가 그 사례였고, 이번엔 커널 런치
700으로 나타났다.

**남은 것**: `q4_gemm_q4k_ge`의 5번째 범위 가정(또는 `rows_pad_p`/`rowexp`/`x` 중 하나)
— `gy=519`(rows_pad=8,304)에서 폴트한다. 이제 `HIP_LAUNCH_BLOCKING=1`로 **정확히 그
런치에서** 재현되므로, 다음 시도는 그 커널의 인덱스 세 곳(`rows_pad_p`, `rowexp[r]`,
`xq[r*xq_w]`)을 범위 증명과 함께 점검하면 된다.

### 6차 확인: 4축 수정 상태에서도 커널 런치 폴트 — 재현 경로 확보 (2026-09-14)

현재 커밋 상태를 확인한 결과 **네 축은 이미 반영돼 있다**(커널 `rowexp[pd]=e` 채움,
`int bound` 인자, 런처 전달, `xbuf_rows` 버퍼). 그런데도 `HIP_LAUNCH_BLOCKING=1`에서

```
launch3: 700 kern=q4_gemm_q4k_ge gx=40 gy=519 gz=1 blk=256
```

가 재현된다. 즉 **다섯 번째 범위 가정이 남아 있고**, 값들은 다음과 같이 정합적이다:

| 값 | 계산 | 판정 |
|---|---|---|
| `expert_bytes` | 스택 471MB ÷ 512 = **0.92MB** (앞서 본 "8MB"는 *h2d 청크 상수*와의 우연한 일치) | 정상 |
| `gz/gx` | `n_out` 640 ÷ 16 = 40 | 정상 |
| `gy` 519 | `rows_pad` 8,304 = `rows`(240) + 패딩 | 상한 내 |
| `rowexp[r]` | 패딩 슬롯까지 채움(수정 반영) | 상한 내 |
| `xq[r*xq_w]` | 버퍼를 `rows+16*ne`로 확보(수정 반영) | 상한 내 |

**다음 시도 절차(확정)**: q4_K 분기에는 q5_1 분기의 `LLM170_GE5_DBG` 같은 값 덤프가
없다 → 먼저 `rows / rows_pad / ne / off_pad[0..4] / rowexp[0..4] / t(디바이스 값)`를
찍고, `HIP_LAUNCH_BLOCKING=1`로 **그 런치에서** 값을 본다. 재현이 3분이므로 반복이 싸다.
그 다음 `q4_gemm_q4k_ge`의 세 인덱스(`w + e*expert_bytes + o*(n_super*144)`,
`rowexp[r]`, `xq + r*xq_w`)를 범위 증명과 함께 본다 — 특히 **`rowexp`가 *가리키는
전문가 id*가 실제 스택 크기 안인지**(패딩 슬롯에 다른 타일의 e가 들어갔을 가능성).

### 7차: 범위 가설 전면 기각 — 모든 인덱스가 정상 (2026-09-14)

q4_K 그룹 런치 직전 호스트 값을 덤프해(env 게이트 `LLM170_GE4_DBG`) 커널 런치 폴트의
입력을 전부 검산했다:

```
# ge4 rows=110 rows_pad=8302 rows_pad_d=true ne=512 per_expert=921600
       n_in=2560 n_out=640 xq_w=880 xbuf=8302 gy=519
```

| 인덱스 | 최대값 | 한계 | 판정 |
|---|---|---|---|
| `w + e*per_expert` | 511 x 921,600 = 470,937,600 | 스택 471,859,200 | 내부 |
| `+ o*(n_super*144)` | 639 x 1,440 = 920,160 | (위에 포함) | 내부 |
| `+ sIdx*144` | +1,440 | 합계 471,859,200 = **할당 끝과 정확히 일치** | 내부 |
| `xq + r*xq_w` | 8,301 x 880 x 4B | x 버퍼 8,302 x 880 x 4B | 내부 |
| `rowexp[r]` | r < rp(≈260, 디바이스 계산) | bound 4바이트 확보 | 내부 |
| `t = *rows_pad_p` | rp | gyp 풀 (ne+2)*4 안 | 내부 |

**즉 범위 위반이 아니다.** 다섯 축 가설(rowexp·x·zero-fill·t·호스트 상한)은 모두
소거됐고, 남은 것은 **런치 자체가 700을 반환하는 것**이다 — 인자가 유효한데도.

**다음 가설(기록)**: (a) `q4_moe_group_t1`(직전 커널)이 **비동기로 폴트**해 스트림을
오염시켰고 우리가 보는 것은 그 파생이라는 것 — `HIP_LAUNCH_BLOCKING=1`은 *런치*를
블로킹할 뿐 *다른 스트림*(`stream2`/이벤트 대기)까지 직렬화하지 않는다. (b) 커널
이미지/함수 포인터 캐시가 그 시점에 교체됐다는 것. 둘 다 **kern=** 이름이 정직하게
찍히는지(Q4Acc가 함수 이름을 어디서 얻는지)를 보면 갈린다.

### 8차: 방아쇠 격리 — `rows_pad_p` 인자 (2026-09-14) ★

세 번의 격리 시험으로 원인을 좁혔다:

| 시험 | 결과 |
|---|---|
| `LLM170_MOE_GROUP_NOD2H=1` (비동기 d2h 예약 생략) | **여전히 폴트** → 스트림2/d2h 가설 기각 |
| `LLM170_MOE_GROUP_SYNC=1` (즉시 동기) | **여전히 폴트** → 동기화 가설도 기각 |
| **`rows_pad_p = null`** (디바이스 행 수 읽기만 차단) | **폴트 사라짐** ✓✓ (출력은 다름 = 패딩 행 미계산, 예상대로) |

**결론**: 폴트의 방아쇠는 **`q4_gemm_q4k_ge`가 `*rows_pad_p`를 읽는 것**이다. 인자 자체는
유효(7차 검산)하고 값도 정합적이지만, 그 **포인터가 가리키는 디바이스 값**이 문제다 —
가장 유력한 것은 **낡은 값**(캐시된 `MoeGroup`의 `rows_pad_d`가 가리키는 `rp`가 이전
호출의 더 큰 값) → `t`가 커져 `xq + r*xq_w`가 버퍼를 넘는다.

**수정 방향(다음 세션, 1~2줄)**: 커널이 **디바이스 값을 읽지 않게** 한다 —
호스트가 이미 아는 **상한(`rows + 16*ne`)을 스칼라 인자로** 넘기면 `t`가 결정적이고,
패딩 행의 (쓰레기) 출력은 scatter가 `inv_pad`로 버리므로 정확성도 유지된다(4축 수정이
버퍼를 상한까지 확보해 둔 상태다). 즉 `rows_pad_p`를 상한 스칼라로 대체하는 것이
가장 작고 확실한 해법이다. `HIP_LAUNCH_BLOCKING=1`로 3분 재현된다.

### 9차: 스칼라 상한으로도 폴트 — 방아쇠는 *인자의 존재* (2026-09-14)

8차의 결론("디바이스 값이 낡았다")을 직접 검증했다: 커널에 `t_rows`(호스트가 아는
상한)를 스칼라로 넘겨 **디바이스 읽기를 완전히 우회**하도록 고쳤다(하위 호환:
`t_rows > 0`이면 사용, 아니면 종전처럼 `*rows_pad_p`). **그래도 같은 폴트**였다:

```
launch3: 700 kern=q4_gemm_q4k_ge gx=40 gy=519 gz=1 blk=256
```

즉 **낡은 값 가설도 기각**이다. `rows_pad_p = null`이면 폴트가 사라지고, 값을 읽지
않게 고쳐도(`t_rows`) 폴트가 남는다 → 방아쇠는 **그 인자에 *비-null 포인터를 넘기는
것* 자체**다. 튜플 구조분해 순서는 검산해 정상이고(`rows_pad_d = rpd`, 호스트 분기는
0), `rpd`는 `gyp` 풀(`(ne+2)*4` B) 안의 유효 주소다.

**다음 후보(기록)**: `hipModuleLaunchKernel`에 넘기는 인자 개수/순서가 커널 서명과
어긋나는지(**13개**로 늘렸는데 다른 경로가 옛 개수로 부르는지 — `q4_gemm_q4k_ge`는
한 곳에서만 런치됨을 확인했으므로 배제), 또는 HIP이 그 주소의 **정렬**을 요구하는지
(`rpd = base + (ne+1)*4` = 4바이트 정렬 — `int*`로는 충분하지만 드라이버가 8바이트를
요구할 수 있다). 후자는 `rpd`를 8바이트 정렬 위치로 옮겨 3분 재현으로 즉시 검증 가능하다.

**미검증 변경은 되돌렸다**(`t_rows` 실험, 게이트 확장) — 커밋 상태가 정본이다.

**11차(2026-09-14, plans/68 — 5번째 축 발견)**: 폴트의 정체는 **out(yperm) 버퍼
상한 누락**이었다. `q4_gemm_q4k_ge`는 `r < *rows_pad`까지 `out[r·n_out+o]`에
기록하는데 yg 풀은 `rows` 크기로만 확보 — 프리필(rows=t·10)에서 패딩 행
(≤16·ne)이 버퍼 끝을 넘어 썼다. 추가로 프리필에서 **레이아웃 혼재**(GEMM 패딩
도메인 vs gather/scatter 비패딩)와 **폴백 행 수**(실제 카운트만) 결함을
발견·수정(LLM170_MOE_GCHECK / LLM170_MOE_HASH 진단 추가). 잔여: 마지막 토큰
플립 1개 + 진단 동기화 시 폴백 행 수 오염. 프리필 활성화는 OFF 유지.

**10차(정렬 가설)**: `rpd`를 8바이트 정렬로 올려(`(base + (ne+1)*4 + 7) & !7`, 풀에 +8B)
시험 → **여전히 같은 폴트**. 정렬 가설도 기각.

**소거된 가설 목록(누적)**: ①상한 부족(4축) ②메모리 고갈 ③낡은 포인터 ④스트림2/d2h
⑤값 낡음(스칼라 우회) ⑥인자 개수/순서 ⑦정렬. 남은 방향: **그 인자를 *받는 커널 쪽*
(HIP이 커널 파라미터를 처리하는 방식) 또는 **그 런치만의 조건**(예: `gy=519`일 때의
grid-limit/드라이버 버그). 후자는 `gy`를 65535 이하로 유지한 채 **행 상한을 줄여**
(상한을 실제 `rp`에 가깝게) 시험하면 갈린다 — 상한을 줄이는 것 자체가 4축 수정으로
안전해졌으므로 3분 검증이 가능하다.

## plans/73 — Flash-Next decode kernel round (2026-09-15)

Gate bit-identical throughout (SELCHECK: device selection lists == host lists).

**Device-side QSA selection (decode t=1)** — the step's largest idle was the
per-QSA-layer host round trip (4 sync frame_reads + select+list ~0.8-1.5ms +
drain/refill; KTRACE at 16k ctx: "after qk_norm_rope" 40.1ms/step over 12
layers). New path: `q4_idx_q_rope` (iq norm+rope, 32-seg f64 mirror),
`q4_idx_bk_update` (incremental block keys, mean→rms→rope, shfl_xor(16)
pairing for idx_dim=128), `q4_idx_score` (4-acc dot mirror),
`q4_idx_topk` (single-block bitonic over (score-mapped,idx) u64 — the
O(B²) `q4_idx_rank` + serial-prefix `q4_idx_expand` pair cost 0.228ms/call,
the bitonic ~0.01ms), attention via `qsa_attention_dev_sel` reading the
device list directly. Device idx_k/bk pools (watermark rewind contract as
qsa_kv) are the source of truth; host caches rebuild once at prefill entry
(`qsa_host_rebuild`), prefill appends via `qsa_idx_append_host`. Debug envs:
LLM170_QSA_HOSTSEL (force old path), LLM170_QSA_SELCHECK (shadow-compare
lists + keep host caches fresh), LLM170_QSA_TOPK=0 (rank+expand pair).

**Warp GEMV family** — `q4_gemm_f32_w` (t=1 f32: hc inject [10240→4] ran
4 blocks/48µs, router [2560→512] 106GB/s; family was ~10ms/step),
`q4_gemm_q5_1_w_ids` (Q5_1 MoE down: lane-per-row 480B stride was 8×
sector amplification at 72-89GB/s; warp-per-row coalesced, lane-0 serial
sb-order sum = bit-identical to gm_ids — an f32 warp-tree variant flipped
the gate's 16th token 1692→24902 and was replaced),
`gemm_q8_0_ids` (Q8_0-down experts: was gather+10×GEMV+scatter ~0.35ms/
layer; byte-offset addressing — q8_0's 34B rows are never word-aligned;
f64-tree reduction = bit-identical to gemm_q8_0), `gemm_q8_0_w`
(small-n_sub q8_0: hc up [320→10240] had 10 of 64 lanes active, 59GB/s).
Route guards: LLM170_F32W / Q5W / Q8IDS / Q8W (=0 disables each).
A lane-0-only `__shfl_sync` reduction crashed (divergent warp hardware
exception) — all-lane participation is mandatory.

**Const upload caching** — qk_norm_rope uploaded qn/kn/cs (3 h2d+sync)
per QSA layer per step; now FNV-hash keyed (qn/kn) and ptr-keyed (cs).

Measured: tg128@short 13.40 → 14.97 → **16.78 t/s**; tg64@16k 11.3 → 13.6 →
**16.11 t/s**. pp unchanged (273 vs 269 at pp4k).

## plans/73 session 2 — PLE device path (2026-09-15 afternoon)

`ple_math_dev` (trait method) + three kernels replace the t=1 PLE host
bridge: `q4_ple_gate` (per-stream grouped norms with the 32-segment f64
mirror via `ple_rms_scale`, serial-order dot, sigmoid gate, value
broadcast, conv-input norm), `q4_ple_conv` (dilated depthwise conv +
silu + ring update), `q4_ple_residual`. The host keeps only the n-gram
hash and the mmap gather (GPU-independent, run at step start). Norm
weights use `exp_cr_exact` — a new always-precise f64-Horner exp in
src_common (FASTEXP-independent) because the host ple_block sigmoid/silu
require bit-matching exp.

Two defects found by the new probes (`llm170 q4-ple-check` synthetic
mirror, `LLM170_PLE_CHECK` end-to-end shadow that diffs res_hc against a
host recompute from the captured pre-PLE state):
1. the sigmoid argument was missing its negation — gates came out as
   exact complements (dev+host = 1.0000 spotted in the diff);
2. norm weights needed the per-stream slice offset (nk[s*n_embd..], not
   nk[0..]).
Also: the device ring's first-use host init must check `ptr.is_null()`
BEFORE `ensure()` (ensure sets the pointer, so the check after it never
fires — uninitialized ring memory).

After fixes: `LLM170_PLE_CHECK` reports max|dev-host| = 0.000e0 across
steps and the Flash-Next gate is bit-identical. Ring rewind (bench
warmup restart) re-initializes from the host ring via a watermark.

Measured: tg128@short 16.78 → **17.54 t/s**, tg64@16k 16.11 → **16.83**,
tg128@4k 17.14. Session totals: 13.40 → 17.54 (+31%), 11.3 → 16.83 (+49%).

`gemm_q5k_v2` (the dormant llama-mmvq vdr=2 port) was wired to an opt-in
route and A/B'd for 27B decode: 10.83 vs 11.35 t/s (slower) — default
off, LLM170_Q5KV2=1 opts in.

llama-reference verify.py collection remains blocked in this environment
(four attempts): CPU mode dies on long prompts (30GB RAM), GPU mode dies
on amdgpu queue eviction at the first request. The gates' bit-identity
carries the last verified llama equivalence.
