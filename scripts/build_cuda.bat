@echo off
rem rawcuda fatbin builder (plans/124 2026-10-04; G2 extension 2026-10-04).
rem chcp 65001 MUST precede any Korean comment byte: cmd re-reads each batch
rem line with the console codepage, and UTF-8 Korean under cp949 blurs line
rem boundaries (mojibake eats the next line's 'rem' prefix — measured).
chcp 65001 >nul
rem 인라인 cmd /c 따옴표 조합이 이 기계에서 깨지므로 배치 경유가 계약이다 —
rem nvcc를 배치 밖에서 인라인 호출하지 않는다. 멱등: 출력을 덮어쓴다.
rem 소스-자산 1:1 규약(rawhip co/*.co 미러): assets\<name>.cu → assets\<name>.fatbin.
rem 새 커널 소스는 SRCS 목록에 이름만 추가하면 된다(이후 목표 재사용).
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
  echo build_cuda: vcvars64 호출 실패 1>&2
  exit /b 1
)
set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe"
set "ASSETS=%~dp0..\crates\backend-gpu\src\rawcuda\assets"
if not exist "%NVCC%" (
  echo build_cuda: nvcc 없음: %NVCC% 1>&2
  exit /b 1
)
set SRCS=smoke exl3_norm
rem exl3_gdn은 별도 블록: -fmad=false(FMA 수축 제거 — GDN 오라클과의
rem 비트동일 미러 계약, assets/exl3_gdn.cu 헤더 4항·plans/124 §6).
rem 나머지 소스(G2-G4)는 기본 fmad 유지 — f16팩 GEMV의 HFMA2 단일
rem 반올림 산술은 이미 오라클에 미러되어 있다(G2 원장).
for %%S in (%SRCS%) do (
  if not exist "%ASSETS%\%%S.cu" (
    echo build_cuda: 소스 없음: %ASSETS%\%%S.cu 1>&2
    exit /b 1
  )
  "%NVCC%" -fatbin -O3 -utf-8 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89 -o "%ASSETS%\%%S.fatbin" "%ASSETS%\%%S.cu"
  if errorlevel 1 (
    echo build_cuda: nvcc 실패: %%S.cu 1>&2
    exit /b 1
  )
  echo build_cuda: %ASSETS%\%%S.fatbin 갱신 완료
)
"%NVCC%" -fatbin -O3 -utf-8 -fmad=false -gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89 -o "%ASSETS%\exl3_gdn.fatbin" "%ASSETS%\exl3_gdn.cu"
if errorlevel 1 (
  echo build_cuda: nvcc 실패: exl3_gdn.cu -fmad=false 1>&2
  exit /b 1
)
echo build_cuda: %ASSETS%\exl3_gdn.fatbin 갱신 완료 (-fmad=false)
rem exl3_attn도 별도 -fmad=false 블록(G6 — 어텐션 오라클과의 비트동일
rem 미러 계약, assets/exl3_attn.cu 헤더 3항·plans/124 §6. 임계 2e-7이
rem 계약 최tight라 FMA 수축·libdevice 초월함수 모두 배제).
"%NVCC%" -fatbin -O3 -utf-8 -fmad=false -gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89 -o "%ASSETS%\exl3_attn.fatbin" "%ASSETS%\exl3_attn.cu"
if errorlevel 1 (
  echo build_cuda: nvcc 실패: exl3_attn.cu -fmad=false 1>&2
  exit /b 1
)
echo build_cuda: %ASSETS%\exl3_attn.fatbin 갱신 완료 (-fmad=false)
rem exl3_ew도 별도 -fmad=false 블록(G7 — ew(silu·mul)의 exp 트윈이
rem f64 DAG라 FMA 수축 제거가 호스트 미러와의 비트동일 조건,
rem assets/exl3_ew.cu 헤더 1항·plans/124 §6. argmax는 f64 미포함이나
rem 동일 fatbin).
"%NVCC%" -fatbin -O3 -utf-8 -fmad=false -gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89 -o "%ASSETS%\exl3_ew.fatbin" "%ASSETS%\exl3_ew.cu"
if errorlevel 1 (
  echo build_cuda: nvcc 실패: exl3_ew.cu -fmad=false 1>&2
  exit /b 1
)
echo build_cuda: %ASSETS%\exl3_ew.fatbin 갱신 완료 (-fmad=false)
rem exl3_q4도 별도 -fmad=false 블록(G8 — Q4_K 디양자화 v=d1·q−mm1·와
rem GEMV 분할형 acc±=yd·(d·sc)·isum 곱-가산 쌍이 FMA 수축되면 호스트
rem core 미러(deq.rs deq_q4_k·lane.rs dot_row_w4a8_q4k_lane_parts)와 비트가
rem 어긋난다. assets/exl3_q4.cu 헤더 [빌드 계약]·plans/124 §6.
"%NVCC%" -fatbin -O3 -utf-8 -fmad=false -gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89 -o "%ASSETS%\exl3_q4.fatbin" "%ASSETS%\exl3_q4.cu"
if errorlevel 1 (
  echo build_cuda: nvcc 실패: exl3_q4.cu -fmad=false 1>&2
  exit /b 1
)
echo build_cuda: %ASSETS%\exl3_q4.fatbin 갱신 완료 (-fmad=false)
rem exl3_mtp(G9 — MTP 드래프트 rms/axpy 헬퍼): 기본 fmad(norm 계열 —
rem exl3_norm.cu와 동일 등급, 트랜센던트 미포함. 어텐션·ew·argmax·
rem 선형은 기존 fatbin 재사용 — mtp_cuda.rs 조립부 계약).
"%NVCC%" -fatbin -O3 -utf-8 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89 -o "%ASSETS%\exl3_mtp.fatbin" "%ASSETS%\exl3_mtp.cu"
if errorlevel 1 (
  echo build_cuda: nvcc 실패: exl3_mtp.cu 1>&2
  exit /b 1
)
echo build_cuda: %ASSETS%\exl3_mtp.fatbin 갱신 완료
rem exl3_fn (FNA - flash-next scaffold smoke): separate -fmad=false block
rem like gdn/attn/ew/q4 - the scaffold discipline requires exact f32
rem products/sums in the plumbing probe (assets/exl3_fn.cu header).
"%NVCC%" -fatbin -O3 -utf-8 -fmad=false -gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89 -o "%ASSETS%\exl3_fn.fatbin" "%ASSETS%\exl3_fn.cu"
if errorlevel 1 (
  echo build_cuda: nvcc 실패: exl3_fn.cu -fmad=false 1>&2
  exit /b 1
)
echo build_cuda: %ASSETS%\exl3_fn.fatbin 갱신 완료 (-fmad=false)
rem exl3_fn_ple (FNB - flash-next PLE stage): separate -fmad=false block
rem like gdn/attn/ew/q4/fn - the gate/conv/resid mul+add pairs must keep
rem the double-rounding f32 mirror of the core value path (stages/ple.rs);
rem the exp twin fn_ple_expf uses explicit fma() which -fmad=false does
rem NOT disable (contraction only - assets/exl3_fn_ple.cu header).
"%NVCC%" -fatbin -O3 -utf-8 -fmad=false -gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89 -o "%ASSETS%\exl3_fn_ple.fatbin" "%ASSETS%\exl3_fn_ple.cu"
if errorlevel 1 (
  echo build_cuda: nvcc 실패: exl3_fn_ple.cu -fmad=false 1>&2
  exit /b 1
)
echo build_cuda: %ASSETS%\exl3_fn_ple.fatbin 갱신 완료 (-fmad=false)
rem exl3_fn_moe (FNE - flash-next moe ffn: route softmax/top-k + f16
rem ptr-array gemm + ew + combine): separate -fmad=false block like
rem gdn/attn/ew/q4/fn - the sequential-accumulation bit-identity mirror
rem contract (acc += x[i]*w, acc += w*y must NOT fuse into FMA - see
rem assets/exl3_fn_moe.cu header [build contract]).
"%NVCC%" -fatbin -O3 -utf-8 -fmad=false -gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89 -o "%ASSETS%\exl3_fn_moe.fatbin" "%ASSETS%\exl3_fn_moe.cu"
if errorlevel 1 (
  echo build_cuda: nvcc 실패: exl3_fn_moe.cu -fmad=false 1>&2
  exit /b 1
)
echo build_cuda: %ASSETS%\exl3_fn_moe.fatbin 갱신 완료 (-fmad=false)
rem exl3_fn_gdn (FNF - flash-next GDN stage kernels: fn_gdn_prep/gate
rem + conv3 negative-control twin + core exp_cr/ln_cr transcendent
rem mirrors). Same -fmad=false discipline as exl3_gdn.cu: the bit-identical
rem core mirror contract (assets/exl3_fn_gdn.cu header) requires no FMA
rem contraction in the f64 mirror DAGs and exact f32 products/sums.
"%NVCC%" -fatbin -O3 -utf-8 -fmad=false -gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89 -o "%ASSETS%\exl3_fn_gdn.fatbin" "%ASSETS%\exl3_fn_gdn.cu"
if errorlevel 1 (
  echo build_cuda: nvcc 실패: exl3_fn_gdn.cu -fmad=false 1>&2
  exit /b 1
)
echo build_cuda: %ASSETS%\exl3_fn_gdn.fatbin 갱신 완료 (-fmad=false)
rem exl3_fn_hc (FNA->FNC - flash-next hc stage kernels): separate
rem -fmad=false block like gdn/attn/ew/q4/fn - the hc stage mirrors core
rem per-op f32 rounding (stages/hc.rs + ops.rs exp_cr + cpu matmul
rem no-FMA dot), so FMA contraction must stay off (assets/exl3_fn_hc.cu
rem header [build contract]).
"%NVCC%" -fatbin -O3 -utf-8 -fmad=false -gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89 -o "%ASSETS%\exl3_fn_hc.fatbin" "%ASSETS%\exl3_fn_hc.cu"
if errorlevel 1 (
  echo build_cuda: nvcc 실패: exl3_fn_hc.cu -fmad=false 1>&2
  exit /b 1
)
echo build_cuda: %ASSETS%\exl3_fn_hc.fatbin 갱신 완료 (-fmad=false)
rem exl3_fn_qsa (FND - flash-next qsa stage): separate -fmad=false block
rem (qsa discipline - the f32 accumulation chains (block-score 4-unrolled
rem dot, attention dot, pooling, AV accumulation) must round like the core
rem f32 mirror (mul+add each once), and the transcendental twins are pure
rem f64 mul/sub DAGs; the sigmoid exp_cr twin uses explicit fma() which
rem survives -fmad=false. assets/exl3_fn_qsa.cu header).
"%NVCC%" -fatbin -O3 -utf-8 -fmad=false -gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89 -o "%ASSETS%\exl3_fn_qsa.fatbin" "%ASSETS%\exl3_fn_qsa.cu"
if errorlevel 1 (
  echo build_cuda: nvcc 실패: exl3_fn_qsa.cu -fmad=false 1>&2
  exit /b 1
)
echo build_cuda: %ASSETS%\exl3_fn_qsa.fatbin 갱신 완료 (-fmad=false)


rem exl3_fn_mtp_frame (FNG - flash-next mtp draft frame kernels:
rem llm170_fn_mtp_combine hc_combine + llm170_fn_mtp_argmax): separate
rem -fmad=false block like gdn/attn/ew/q4/fn/hc - the combine accumulation
rem acc += out*2sigma(inj/hc) must keep the double-rounding f32 mirror of
rem the host oracle (layers.rs hc_combine L2575-2586); the exp_cr twin uses
rem explicit __fma_rn which -fmad=false does NOT disable (contraction only
rem - assets/exl3_fn_mtp_frame.cu header [build contract]).
"%NVCC%" -fatbin -O3 -utf-8 -fmad=false -gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89 -o "%ASSETS%\exl3_fn_mtp_frame.fatbin" "%ASSETS%\exl3_fn_mtp_frame.cu"
if errorlevel 1 (
  echo build_cuda: nvcc failed: exl3_fn_mtp_frame.cu -fmad=false 1>&2
  exit /b 1
)
echo build_cuda: %ASSETS%\exl3_fn_mtp_frame.fatbin updated (-fmad=false)

rem FNH (fn-chain greedy decode chain probe, 2026-10-05): NO new fatbin —
rem the chain probe assembles the landed stage fatbins above
rem (exl3_fn_ple/exl3_fn_hc/exl3_fn_qsa/exl3_fn_moe/exl3_gdn+exl3_fn_gdn
rem + exl3_fn_mtp_frame) and adds no kernel of its own (probe header
rem kernel->probe map). This block exists so the SRCS list stays the
rem single source of truth.

rem ds4_hc (plans/130 B3 - DeepSeek-V4 mHC 4-stream hyper-connection
rem stage: mix GEMV + pure-f32 sequential rms_scale + 4x4 Sinkhorn
rem (20 iter, exact order) + pre/post apply, bf16 boundary): separate
rem -fmad=false block like fn_hc - the deepseek4 core mirror
rem (stages/hc.rs + ops.rs rms_scale/bf16_round + ops exp_cr twin)
rem requires exact f32 products/sums in every accumulation chain
rem (mix dot, rms sumsq, pre/post weighted sums) - FMA contraction would
rem collapse the double roundings the core oracle defines. The ds4_exp_cr
rem twin uses explicit __fma_rn which -fmad=false does NOT disable
rem (contraction only - assets/ds4_hc.cu header [build contract]).
"%NVCC%" -fatbin -O3 -utf-8 -fmad=false -gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89 -o "%ASSETS%\ds4_hc.fatbin" "%ASSETS%\ds4_hc.cu"
if errorlevel 1 (
  echo build_cuda: nvcc failed: ds4_hc.cu -fmad=false 1>&2
  exit /b 1
)
echo build_cuda: %ASSETS%\ds4_hc.fatbin updated (-fmad=false)
rem ds4_attn (plans/130 B3 - DeepSeek-V4-Flash attention stage: CSA/HCA/SWA
rem compressor + indexer + attention-sink + grouped output): separate
rem -fmad=false block like gdn/attn/ew/q4/fn - the ds4 oracle (core
rem deepseek4 ops.rs/attn.rs) mirrors per-op f32 rounding (gemm k-ascending
rem accumulation, RMS sums, pooling weighted sums, sparse-attn e-ascending
rem AV accumulation must NOT fuse into FMA), and the only transcendental
rem is the exp_cr f64 twin (explicit __fma_rn survives -fmad=false) plus
rem the host-built RoPE table (assets/ds4_attn.cu header [build contract]).
"%NVCC%" -fatbin -O3 -utf-8 -fmad=false -gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89 -o "%ASSETS%\ds4_attn.fatbin" "%ASSETS%\ds4_attn.cu"
if errorlevel 1 (
  echo build_cuda: nvcc failed: ds4_attn.cu -fmad=false 1>&2
  exit /b 1
)
echo build_cuda: %ASSETS%\ds4_attn.fatbin updated (-fmad=false)


rem ds4_moe (plans/130 B3 - deepseek4 MoE stage: gate sqrtsoftplus
rem gemv + hash/noaux_tc routing + fp8_sim + f32 ptr-array gemv +
rem asymmetric swiglu clamp + combine): separate -fmad=false block like
rem gdn/attn/ew/q4/fn_moe - the sequential-accumulation bit-identity
rem mirror contract (acc += x[i]*w and the swiglu mul chain must NOT
rem fuse into FMA - see assets/ds4_moe.cu header build contract).
"%NVCC%" -fatbin -O3 -utf-8 -fmad=false -gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89 -o "%ASSETS%\ds4_moe.fatbin" "%ASSETS%\ds4_moe.cu"
if errorlevel 1 (
  echo build_cuda: nvcc 실패: ds4_moe.cu -fmad=false 1>&2
  exit /b 1
)
echo build_cuda: %ASSETS%\ds4_moe.fatbin 갱신 완료 (-fmad=false)

rem ds4_mtp (plans/130 B4 - DeepSeek-V4 DSpark MTP draft stage:
rem main_proj FP8-sim GEMM + rmsw norms + DSpark sparse attention/derot
rem (ds4_attn.cu literal re-plant - shifted rope table) + shared head
rem stripwise gemv (k-block partial-sum order) + markov bias gather/gemv
rem + confidence concat dot): separate -fmad=false block like ds4_attn -
rem every f32 accumulation (main_proj gemm k-ascending, rms sumsq,
rem sparse-attn e-ascending AV, head strip k-block partials, markov k-sum,
rem confidence chain sum) must round like the core mirror (frame.rs
rem dspark fns - mul+add each once), and the only transcendental is the
rem ds4m_exp_cr f64 twin (explicit __fma_rn survives -fmad=false -
rem assets/ds4_mtp.cu header [build contract]).
"%NVCC%" -fatbin -O3 -utf-8 -fmad=false -gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89 -o "%ASSETS%\ds4_mtp.fatbin" "%ASSETS%\ds4_mtp.cu"
if errorlevel 1 (
  echo build_cuda: nvcc failed: ds4_mtp.cu -fmad=false 1>&2
  exit /b 1
)
echo build_cuda: %ASSETS%\ds4_mtp.fatbin updated (-fmad=false)
exit /b 0
