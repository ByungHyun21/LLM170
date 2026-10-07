@echo off
rem rawcuda standalone verification harness (plans/124 G2, 2026-10-04).
rem G1 ad-hoc rustc pattern, productized for reuse by later goals (G3+).
rem chcp 65001 MUST precede any Korean comment byte (cmd re-reads each line
rem with the console codepage - UTF-8 Korean under cp949 breaks parsing).
chcp 65001 >nul
rem compiles the standalone shim + rawcuda sources with rustc (the full
rem workspace cannot build on Windows - llm170-core mmap defect, G1 ledger)
rem and runs the probes from the WORKTREE ROOT so relative asset paths hold.
rem [plans/129-cuda-only C3 - golden reference record, G10 2026-10-04]
rem scripts/.verify-cuda-golden.tsv is the machine-readable ledger of every
rem probe threshold and measured value (columns: module|probe|threshold|
rem measured|note). Future probe changes are machine-judged against it:
rem a threshold or measured-value change requires an explicit golden-ledger
rem update commit. FAIL rule follows the original 129 A23: only corr/maxdiff
rem and NaN breaches FAIL - speed is WARN only (pre-CMP-arrival speed cells
rem stay 'pending sm_80' and are excluded from the golden set). This bat keeps
rem working as-is (judgement logic stays inside the probes); the golden file
rem is the reference record, not a functional rewrite.
setlocal
cd /d "%~dp0.."
where rustc >nul 2>&1
if errorlevel 1 (
  echo verify_cuda: rustc not on PATH 1>&2
  exit /b 1
)
if not exist target mkdir target
rustc --edition 2024 -O scripts\cuda_probe_shim.rs -o target\cuda_probe.exe
if errorlevel 1 (
  echo verify_cuda: rustc compile failed 1>&2
  exit /b 1
)
target\cuda_probe.exe smoke
if errorlevel 1 exit /b 1
target\cuda_probe.exe gemv
if errorlevel 1 exit /b 1
rem negative control: MUST fail (non-zero exit) with the NEG-DETECTED marker
rem (defect-ledger 17: the verifier itself must be verified to fail).
target\cuda_probe.exe gemv-neg > target\cuda_neg.out 2>&1
set NEGC=%errorlevel%
type target\cuda_neg.out
if "%NEGC%"=="0" goto negfail
findstr /C:"NEG-DETECTED" target\cuda_neg.out >nul 2>&1
if errorlevel 1 goto negfail
echo verify_cuda: negative control detected (non-zero exit, maxdiff above threshold)
goto negdone
:negfail
echo verify_cuda: negative control FAILED to fail - verifier defect 1>&2
exit /b 1
:negdone
target\cuda_probe.exe norm
if errorlevel 1 exit /b 1
rem negative control (norm): MUST fail (non-zero exit) with the NEG-DETECTED
rem marker covering BOTH defect-2 (row0 read) and eps-1e-5 controls.
target\cuda_probe.exe norm-neg > target\cuda_norm_neg.out 2>&1
set NNEGC=%errorlevel%
type target\cuda_norm_neg.out
if "%NNEGC%"=="0" goto nnegfail
findstr /C:"NEG-DETECTED" target\cuda_norm_neg.out >nul 2>&1
if errorlevel 1 goto nnegfail
echo verify_cuda: norm negative controls detected (non-zero exit, maxdiff above threshold)
goto nnegdone
:nnegfail
echo verify_cuda: norm negative control FAILED to fail - verifier defect 1>&2
exit /b 1
:nnegdone
target\cuda_probe.exe gemm2
if errorlevel 1 exit /b 1
rem negative control (gemm2): MUST fail (non-zero exit) with the NEG-DETECTED
rem marker - had_out applied twice (defect 15) must be detectable.
target\cuda_probe.exe gemm2-neg > target\cuda_gemm2_neg.out 2>&1
set GNEGC=%errorlevel%
type target\cuda_gemm2_neg.out
if "%GNEGC%"=="0" goto gnegfail
findstr /C:"NEG-DETECTED" target\cuda_gemm2_neg.out >nul 2>&1
if errorlevel 1 goto gnegfail
echo verify_cuda: gemm2 negative control detected (non-zero exit, maxdiff above threshold)
goto gnegdone
:gnegfail
echo verify_cuda: gemm2 negative control FAILED to fail - verifier defect 1>&2
exit /b 1
:gnegdone
target\cuda_probe.exe gdn
if errorlevel 1 exit /b 1
rem negative controls (gdn): MUST fail (non-zero exit) with the NEG-DETECTED
rem marker covering BOTH defect classes: (a) l2perm gather direction
rem (section 3.3 accident), (b) S0=0 state path on a diverging fixture
rem (ledger 17 - the verifier itself must be verified to fail).
target\cuda_probe.exe gdn-neg > target\cuda_gdn_neg.out 2>&1
set GNEGA=%errorlevel%
type target\cuda_gdn_neg.out
if "%GNEGA%"=="0" goto gdnegfail
findstr /C:"NEG-DETECTED" target\cuda_gdn_neg.out >nul 2>&1
if errorlevel 1 goto gdnegfail
echo verify_cuda: gdn negative controls detected (non-zero exit, maxdiff above threshold)
goto gdnegdone
:gdnegfail
echo verify_cuda: gdn negative control FAILED to fail - verifier defect 1>&2
exit /b 1
:gdnegdone
target\cuda_probe.exe attn
if errorlevel 1 exit /b 1
rem negative control (attn): MUST fail (non-zero exit) with the NEG-DETECTED
rem marker - KV index taken from a HOST pos copy instead of the device
rem pp[0] read (defect 4) must be value-detectable.
target\cuda_probe.exe attn-neg > target\cuda_attn_neg.out 2>&1
set ANEGC=%errorlevel%
type target\cuda_attn_neg.out
if "%ANEGC%"=="0" goto anegfail
findstr /C:"NEG-DETECTED" target\cuda_attn_neg.out >nul 2>&1
if errorlevel 1 goto anegfail
echo verify_cuda: attn negative control detected (non-zero exit, maxdiff above threshold)
goto anegdone
:anegfail
echo verify_cuda: attn negative control FAILED to fail - verifier defect 1>&2
exit /b 1
:anegdone
target\cuda_probe.exe ew
if errorlevel 1 exit /b 1
target\cuda_probe.exe argmax
if errorlevel 1 exit /b 1
rem negative control (argmax): MUST fail (non-zero exit) with the
rem NEG-DETECTED marker - argmax over a WRONG length (defect ledger 8:
rem n is the LOGIT LENGTH 248320, not a row count) must be token-
rem detectable (mismatch + non-zero exit).
target\cuda_probe.exe argmax-neg > target\cuda_argmax_neg.out 2>&1
set AMNEGC=%errorlevel%
type target\cuda_argmax_neg.out
if "%AMNEGC%"=="0" goto amnegfail
findstr /C:"NEG-DETECTED" target\cuda_argmax_neg.out >nul 2>&1
if errorlevel 1 goto amnegfail
echo verify_cuda: argmax negative controls detected (non-zero exit, token mismatch)
goto amnegdone
:amnegfail
echo verify_cuda: argmax negative control FAILED to fail - verifier defect 1>&2
exit /b 1
:amnegdone
rem G8 Q4 (GGUF Q4_K / UD-Q4_K_XL) module probes - real GGUF
rem fixture (D:/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf required).
rem [G10 fix 2026-10-04] the two Korean rem lines added in G8 crashed cmd
rem parsing at this point (UTF-8 rem pair under cp949 console eats the next
rem line prefix - same class as the header warning; reproduced with the G9
rem tree: verify stopped right after argmax). Comment-only rewrite to English;
rem zero functional change (per plans/129-cuda-only C3 note).
target\cuda_probe.exe q4-dequant
if errorlevel 1 exit /b 1
target\cuda_probe.exe q4-gemv
if errorlevel 1 exit /b 1
target\cuda_probe.exe q4-gemm
if errorlevel 1 exit /b 1
rem negative control (q4): MUST fail (non-zero exit) with the NEG-DETECTED
rem marker - corrupted superblock scale d (bit-flip 0x0100) must be
rem value-detectable in the dequant path (ledger 17).
target\cuda_probe.exe q4-neg > target\cuda_q4_neg.out 2>&1
set Q4NEGC=%errorlevel%
type target\cuda_q4_neg.out
if "%Q4NEGC%"=="0" goto q4negfail
findstr /C:"NEG-DETECTED" target\cuda_q4_neg.out >nul 2>&1
if errorlevel 1 goto q4negfail
echo verify_cuda: q4 negative control detected (non-zero exit, maxdiff above threshold)
goto q4negdone
:q4negfail
echo verify_cuda: q4 negative control FAILED to fail - verifier defect 1>&2
exit /b 1
:q4negdone
target\cuda_probe.exe mtp
if errorlevel 1 exit /b 1
rem negative control (mtp): MUST fail (non-zero exit) with the
rem NEG-DETECTED marker - capture point swapped to post-FFN-sum
rem (defect 10, plans/124 sec 3.4) must be value-detectable
rem (maxdiff > 2e-4), not just argmax-detectable (sec 6).
target\cuda_probe.exe mtp-neg > target\cuda_mtp_neg.out 2>&1
set MNEGC=%errorlevel%
type target\cuda_mtp_neg.out
if "%MNEGC%"=="0" goto mnegfail
findstr /C:"NEG-DETECTED" target\cuda_mtp_neg.out >nul 2>&1
if errorlevel 1 goto mnegfail
echo verify_cuda: mtp negative control detected (non-zero exit, maxdiff above threshold)
goto mnegdone
:mnegfail
echo verify_cuda: mtp negative control FAILED to fail - verifier defect 1>&2
exit /b 1
:mnegdone
rem Flash-Next (qwen4exp) scaffold arm - plans/124 G001 FNA.
rem Loads the new exl3_fn.fatbin, runs the trivial kernel, prints
rem "device: <name> | flash-next scaffold PASS". Missing fatbin or any
rem value mismatch must exit non-zero (asset resolver errors included).
target\cuda_probe.exe fn
if errorlevel 1 exit /b 1
rem Fixture inventory arm - proves BOTH fixture loaders on the real
rem files (EXL3 5.05bpw sharded safetensors + ngram header + quant
rem stream; GGUF 4-shard + mtp + mmproj), offset reads only - no full
rem loads (37GiB ngram table is never read beyond its header/metas).
target\cuda_probe.exe fn-inv
if errorlevel 1 exit /b 1
rem Flash-Next PLE stage arm (FNB) - full chain on the real Flash-Next
rem fixtures: GGUF hash params + IQ4_NL table rows + F32/Q8_0 block weights,
rem EXL3 ngram header cross-check + quantization_config.json stream gate.
rem Bit-exact judgement (bitdiff=0) per stage; any mismatch exits non-zero.
target\cuda_probe.exe ple
if errorlevel 1 exit /b 1
rem negative control (ple): MUST fail (non-zero exit) with the NEG-DETECTED
rem marker - BOTH corruption classes must be value-detected in emb:
rem (a) hash modulus vs[3]+7, (b) gather staging row shift (ledger 17).
target\cuda_probe.exe ple-neg > target\cuda_ple_neg.out 2>&1
set PLENEGC=%errorlevel%
type target\cuda_ple_neg.out
if "%PLENEGC%"=="0" goto plenegfail
findstr /C:"NEG-DETECTED" target\cuda_ple_neg.out >nul 2>&1
if errorlevel 1 goto plenegfail
echo verify_cuda: ple negative control detected (non-zero exit, maxdiff above threshold)
goto plenegdone
:plenegfail
echo verify_cuda: ple negative control FAILED to fail - verifier defect 1>&2
exit /b 1
:plenegdone
rem Flash-Next MoE FFN arm - plans/124 G001 FNA->FNE.
rem (i) 27B-width synthetic zones (routing decisions discrete EXACT)
rem + (ii-a) 35B-A3B config dims + (ii-b) Flash-Next config dims
rem vs the embedded core-mirror oracle - value maxdiff judged.
target\cuda_probe.exe moe
if errorlevel 1 exit /b 1
rem negative control (moe): MUST fail (non-zero exit) with the
rem NEG-DETECTED marker covering BOTH defect classes: (a) router
rem weight corruption (sign bit flip) and (b) swapped expert
rem assignment (routing unchanged - value divergence proves the
rem assignment class, ledger 17: the verifier itself must fail).
target\cuda_probe.exe moe-neg > target\cuda_moe_neg.out 2>&1
set MNEGA=%errorlevel%
type target\cuda_moe_neg.out
if "%MNEGA%"=="0" goto monegfail
findstr /C:"NEG-DETECTED" target\cuda_moe_neg.out >nul 2>&1
if errorlevel 1 goto monegfail
echo verify_cuda: moe negative controls detected (non-zero exit, maxdiff above threshold)
goto monegdone
:monegfail
echo verify_cuda: moe negative control FAILED to fail - verifier defect 1>&2
exit /b 1
:monegdone
rem Flash-Next GDN stage arm (FNF, plans/124 fn-gdn) - real GGUF
rem fixture (D:/models/qwen3.8-Flash-Next main shard required; override
rem via arg). Value maxdiff gates: ported prep/gate stages expected
rem bit-identical to the embedded core oracle, reused conv/scan stages
rem and end-to-end <= 2e-4 (plans/124 sec 1 GDN threshold).
target\cuda_probe.exe fn-gdn
if errorlevel 1 exit /b 1
rem negative control (fn-gdn): MUST fail (non-zero exit) with the
rem NEG-DETECTED marker covering BOTH defect classes: (a) conv misread
rem as 3-tap where the contract is conv_k=4 (the FNF reuse-verdict core
rem difference), (b) S0=0 state path on a diverging fixture (ledger 17).
target\cuda_probe.exe fn-gdn-neg > target\cuda_fn_gdn_neg.out 2>&1
set FNGNEGC=%errorlevel%
type target\cuda_fn_gdn_neg.out
if "%FNGNEGC%"=="0" goto fngnegfail
findstr /C:"NEG-DETECTED" target\cuda_fn_gdn_neg.out >nul 2>&1
if errorlevel 1 goto fngnegfail
echo verify_cuda: fn-gdn negative controls detected (non-zero exit, maxdiff above threshold)
goto fngnegdone
:fngnegfail
echo verify_cuda: fn-gdn negative control FAILED to fail - verifier defect 1>&2
exit /b 1
:fngnegdone
rem Flash-Next HC stage arm - plans/124 G001 FNA->FNC (2026-10-05).
rem hc: grouped rms + low-rank gate + stream mean (+inject) vs the
rem core-mirror oracle (stages/hc.rs), bit-exact gate; deterministic
rem synthetic mixers + the real mtp_hyper_connection_mixer_patch
rem .safetensors fixture (offset reads only, no full loads).
target\cuda_probe.exe hc
if errorlevel 1 exit /b 1
rem negative control (hc): MUST fail (non-zero exit) with the
rem NEG-DETECTED marker - wrong connection count (hc=3 mixing)
rem and wrong stream order (norm rows 0<->1 swapped at registration)
rem must be value-detectable (ledger 17 - verifier self-check).
target\cuda_probe.exe hc-neg > target\cuda_hc_neg.out 2>&1
set HCNEGC=%errorlevel%
type target\cuda_hc_neg.out
if "%HCNEGC%"=="0" goto hcnegfail
findstr /C:"NEG-DETECTED" target\cuda_hc_neg.out >nul 2>&1
if errorlevel 1 goto hcnegfail
echo verify_cuda: hc negative controls detected (non-zero exit, maxdiff above threshold)
goto hcnegdone
:hcnegfail
echo verify_cuda: hc negative control FAILED to fail - verifier defect 1>&2
exit /b 1
:hcnegdone
rem Flash-Next QSA stage arm (FND) - selection list exactness vs the
rem embedded core oracle + value maxdiff gates (twin bit-identity /
rem core-libm documented threshold). Real GGUF fixture required
rem (qwen3.8-Flash-Next UD-Q4_K_XL shard 1 - dims + real norm weights).
target\cuda_probe.exe qsa
if errorlevel 1 exit /b 1
rem negative control (qsa): MUST fail (non-zero exit) with the
rem NEG-DETECTED marker covering BOTH defect classes: (a) top-k
rem off-by-one (n_sel+1 - structural count mismatch), (b) wrong pooling
rem (block-key source rows shifted by one).
target\cuda_probe.exe qsa-neg > target\cuda_qsa_neg.out 2>&1
set QNEGC=%errorlevel%
type target\cuda_qsa_neg.out
if "%QNEGC%"=="0" goto qsanegfail
findstr /C:"NEG-DETECTED" target\cuda_qsa_neg.out >nul 2>&1
if errorlevel 1 goto qsanegfail
echo verify_cuda: qsa negative controls detected (non-zero exit)
goto qsanegdone
:qsanegfail
echo verify_cuda: qsa negative control FAILED to fail - verifier defect 1>&2
exit /b 1
:qsanegdone

rem Flash-Next MTP draft frame arm (FNG, plans/124 mtp-frame) - full chain
rem on real Flash-Next dims from config.json: eh_proj -> hc_mix(attn) ->
rem dense gated attention (own KV, CPU row core) -> hc_combine -> hc_mix(ffn)
rem -> MoE -> hc_combine -> nextn.hc_head -> output head + argmax + chain h.
rem Judged pre-MoE bit-exact, post-MoE vs the embedded core-mirror oracle
rem (frame/mtp.rs citations) within the plans/124 sec 3.4 threshold; fixture
rem replay: mixer patch safetensors + MTP Q8_0 GGUF (offset reads only).
target\cuda_probe.exe mtp-frame
if errorlevel 1 exit /b 1
rem negative control (mtp-frame): MUST fail (non-zero exit) with the
rem NEG-DETECTED marker covering BOTH defect classes: (a) wrong capture
rem point (chain exported pre-ffn-combine - defect ledger 10), (b) wrong
rem norm convention (EXL3 w-1 storage rule applied to an original-value
rem gamma - sec 3.4, ledger 17: the verifier itself must fail).
target\cuda_probe.exe mtp-frame-neg > target\cuda_mtpfn_neg.out 2>&1
set MTFNEGC=%errorlevel%
type target\cuda_mtpfn_neg.out
if "%MTFNEGC%"=="0" goto mtfnegfail
findstr /C:"NEG-DETECTED" target\cuda_mtpfn_neg.out >nul 2>&1
if errorlevel 1 goto mtfnegfail
echo verify_cuda: mtp-frame negative controls detected (non-zero exit, maxdiff above threshold)
goto mtfnegdone
:mtfnegfail
echo verify_cuda: mtp-frame negative control FAILED to fail - verifier defect 1>&2
exit /b 1
:mtfnegdone

rem Flash-Next greedy decode chain arm (FNH, plans/124 fn-chain) - the
rem landed stage modules (ple/hc/qsa/moe/gdn + mtp_frame) assembled on the
rem real core layer schedule vs the embedded core-mirror oracle chain: 3
rem greedy steps + MTP spec round (draft k=3 -> trunk verify t=2 -> greedy
rem accept/reject per core mtp_spec_step). Per-stage maxdiff table,
rem end-to-end logits chain-terminal gate 2e-3 (probe header ledger -
rem single-stage fn-gdn 2e-4 does not cover the 48-layer accumulation),
rem per-step/verified token identity (argmax allowed at chain level only).
rem Real GGUF/EXL3 fixtures, offset reads only; MoE is a reduced 32-expert
rem synthetic model (probe header ledger). Fails non-zero on any gate breach.
target\cuda_probe.exe fn-chain
if errorlevel 1 exit /b 1
rem negative control (fn-chain): MUST fail (non-zero exit) with the
rem NEG-DETECTED marker covering ALL THREE chain defect classes: (a) wrong
rem layer order (GDN slot 5<->6 swap on the module chain), (b) wrong
rem residual attach (hc_combine weight fixed to 1.0 at il=9), (c) wrong
rem accept/reject order (verify row off-by-one in the spec round).
target\cuda_probe.exe fn-chain-neg > target\cuda_fn_chain_neg.out 2>&1
set FCNEGC=%errorlevel%
type target\cuda_fn_chain_neg.out
if "%FCNEGC%"=="0" goto fcnegfail
findstr /C:"NEG-DETECTED" target\cuda_fn_chain_neg.out >nul 2>&1
if errorlevel 1 goto fcnegfail
echo verify_cuda: fn-chain negative controls detected (non-zero exit, logits maxdiff above threshold)
goto fcnegdone
:fcnegfail
echo verify_cuda: fn-chain negative control FAILED to fail - verifier defect 1>&2
exit /b 1
:fcnegdone

rem plans/130 B3 DeepSeek-V4 mHC hyper-connection arm (ds4-hc) - mix
rem GEMV + rms_scale + split/sinkhorn(20 iter) + pre/post apply vs the
rem embedded deepseek4 core oracle (stages/hc.rs), bit-exact gate, on the
rem REAL EXL3 fixture hc weights (layers.0.hc_attn/layers.2.hc_ffn/
rem hc_head - F32 offset reads only, no full loads).
target\cuda_probe.exe ds4-hc
if errorlevel 1 exit /b 1
rem negative control (ds4-hc): MUST fail (non-zero exit) with the
rem NEG-DETECTED marker covering ALL THREE defect classes: (a) sinkhorn
rem iteration order swapped (19x{col;row}), (b) comb matrix transposed
rem in hc_post, (c) hc_eps dropped everywhere (ledger 17 - the verifier
rem itself must be verified to fail).
target\cuda_probe.exe ds4-hc-neg > target\cuda_ds4_hc_neg.out 2>&1
set DHNEGC=%errorlevel%
type target\cuda_ds4_hc_neg.out
if "%DHNEGC%"=="0" goto dhnegfail
findstr /C:"NEG-DETECTED" target\cuda_ds4_hc_neg.out >nul 2>&1
if errorlevel 1 goto dhnegfail
echo verify_cuda: ds4-hc negative controls detected (non-zero exit, maxdiff above threshold)
goto dhnegdone
:dhnegfail
echo verify_cuda: ds4-hc negative control FAILED to fail - verifier defect 1>&2
exit /b 1
:dhnegdone
rem B3 ds4-attn arm (plans/130, 2026-10-05) - DeepSeek-V4-Flash attention
rem stage (SWA L0 / CSA L2 / HCA L3 + compressor + indexer + attention-sink
rem + grouped output) on the real EXL3 fixture (DeepSeek-V4-Flash-Vision-Exp
rem 3.04bpw - trellis dequant + BF16/F32 plain + embed rows, offset reads
rem only), judged per-stage bit-exact (bitdiff=0) vs the embedded core
rem oracle (crates/core/src/deepseek4 citations in the probe header).
target\cuda_probe.exe ds4-attn
if errorlevel 1 exit /b 1
rem negative control (ds4-attn): MUST fail (non-zero exit) with the
rem NEG-DETECTED marker covering ALL THREE defect classes: (a) indexer
rem top-k causal off-by-one (visible+1), (b) compressor ape misalign,
rem (c) attention-sink logit dropped (ledger 17 - verifier self-check).
target\cuda_probe.exe ds4-attn-neg > target\cuda_ds4_neg.out 2>&1
set DS4NEGC=%errorlevel%
type target\cuda_ds4_neg.out
if "%DS4NEGC%"=="0" goto ds4negfail
findstr /C:"NEG-DETECTED" target\cuda_ds4_neg.out >nul 2>&1
if errorlevel 1 goto ds4negfail
echo verify_cuda: ds4-attn negative controls detected (non-zero exit)
goto ds4negdone
:ds4negfail
echo verify_cuda: ds4-attn negative control FAILED to fail - verifier defect 1>&2
exit /b 1
:ds4negdone

rem DeepSeek-V4 MoE stage arm (plans/130 B3, 2026-10-05) - hash
rem routing (L0-2 tid2eid) + noaux_tc routed (L3+, sqrtsoftplus +
rem e_score_correction_bias top-6) + EXL3 trellis experts (K=3) +
rem shared expert (K=5) + SwiGLU limit-10 asymmetric clamp vs the
rem embedded core deepseek4 oracle (stages/moe.rs). Real EXL3 fixture
rem (default D:/models/DeepSeek-V4-Flash-Vision-Exp-exl3-3.04bpw,
rem arg or LLM170_DS4_EXL3 override), offset reads only. Routing ids
rem EXACT (never argmax-only) + route-w/out value maxdiff judged.
target\cuda_probe.exe ds4-moe
if errorlevel 1 exit /b 1
rem negative control (ds4-moe): MUST fail (non-zero exit) with the
rem NEG-DETECTED marker covering ALL THREE defect classes: (a) bias
rem included in routing weights, (b) hash table transposed
rem (module-side data corruption), (c) SwiGLU clamp missing
rem (oracle-side defect injection - ledger 17: the verifier itself
rem must be verified to fail).
target\cuda_probe.exe ds4-moe-neg > target\cuda_ds4_moe_neg.out 2>&1
set DS4NEGC=%errorlevel%
type target\cuda_ds4_moe_neg.out
if "%DS4NEGC%"=="0" goto ds4negfail
findstr /C:"NEG-DETECTED" target\cuda_ds4_moe_neg.out >nul 2>&1
if errorlevel 1 goto ds4negfail
echo verify_cuda: ds4-moe negative controls detected (non-zero exit, maxdiff above threshold)
goto ds4negdone
:ds4negfail
echo verify_cuda: ds4-moe negative control FAILED to fail - verifier defect 1>&2
exit /b 1
:ds4negdone

rem DeepSeek-V4 DSpark MTP draft stage arm (plans/130 B4, 2026-10-05) -
rem full DSpark chain on the real EXL3 fixture: main_x (mtp.0 main_proj)
rem -> window warm -> draft block (mtp.0 attn/hc/moe, window-only
rem attention) -> hc_head/norm (mtp.2) -> shared head stripwise gemv
rem -> markov bias -> confidence, vs the embedded core oracle
rem (frame.rs dspark fns citations in the probe header). Pre-MoE points
rem bit-exact, post-MoE <= 2e-3 (ds4-moe route-weight class), negatives
rem (trunk-perm/bias-omitted/noise-id) detected at logits level.
target\cuda_probe.exe ds4-mtp
if errorlevel 1 exit /b 1
rem negative control (ds4-mtp): MUST fail (non-zero exit) with the
rem NEG-DETECTED marker covering ALL THREE defect classes: (a) wrong
rem trunk target layers (plane permutation), (b) markov bias omitted
rem (|bias| scale + out-ids sensitivity), (c) wrong noise-token id
rem (128799->128798 embed row swap) - reduced chain (attn subblock +
rem bias, no MoE/head - ledger 17 verifier self-check).
target\cuda_probe.exe ds4-mtp-neg > target\cuda_ds4_mtp_neg.out 2>&1
set DMTPNEGC=%errorlevel%
type target\cuda_ds4_mtp_neg.out
if "%DMTPNEGC%"=="0" goto dmtpnegfail
findstr /C:"NEG-DETECTED" target\cuda_ds4_mtp_neg.out >nul 2>&1
if errorlevel 1 goto dmtpnegfail
echo verify_cuda: ds4-mtp negative controls detected (non-zero exit, maxdiff above threshold)
goto dmtpnegdone
:dmtpnegfail
echo verify_cuda: ds4-mtp negative control FAILED to fail - verifier defect 1>&2
exit /b 1
:dmtpnegdone
echo verify_cuda: all probes PASS
exit /b 0
