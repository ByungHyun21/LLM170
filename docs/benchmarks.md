# Performance vs llama.cpp

Conditions always quoted (context / batch / quantization / backend). All
numbers on the dev machine (Radeon 8060S, gfx1151, 32-thread CPU) unless
noted. Per-file measurement records live in `docs/source/` (1:1 with code
paths). The chronological originals were moved to
`docs/archive/benchmarks-chronological-2026-09-15.md` — nothing was
deleted, only regrouped here.

## Current scorecard (2026-09-15, ROCm 10 userspace, greedy, solo)

### Qwen3.8-27B (Q4_K_XL 16.3 GiB)

| backend | pp512 | pp4096 | pp16384 | tg128@4k | tg128@16k |
|---|---|---|---|---|---|
| LLM170 hip (2026-09-16) | **374** | 324 | 253 | 11.1 | 11.5 |
| LLM170 vulkan | — | 150 | — (device lost) | 9.2 | — (device lost) |
| llama.cpp (ROCm 10) | 347 | **342** | **317** | **11.6** | **11.9** |

(hip session progression: pp4096 315→324, tg@16k 10.3→11.5; the 27B
decode is within ~5% of the practical DRAM limit — see "ceilings"
below. pp512 374 leaves llama's 347 behind.)

### Qwen3.8-Flash-Next (177B-A3B, Q4_K_XL 103.7 GiB)

| backend | pp4096 | pp16384 | tg128@4k | tg128@16k |
|---|---|---|---|---|
| LLM170 hip (2026-09-16) | **270** | **243** | 18.0 | 17.9 |
| LLM170 vulkan | 271 | 240 | 17.1 | 16.7 |
| llama.cpp (ROCm 10) | 237 | 229 | **20.2** | **20.0** |

(hip session progression for tg: 13.40→17.5 short ctx (+31%),
11.3→16.8 @16k (+49%); pp512 251 vs llama 206.)

### Speculative decode (27B, MTP)

single-stream 23.5 t/s, np4 31.0 t/s = **2.0-2.04× llama MTP**.

## What worked (adopted, grouped)

### Host-sync elimination (Flash-Next decode)

| Technique | Date | Effect |
|---|---|---|
| Device-side QSA indexer selection (`q4_idx_*` kernels, device idx/bk pools, host rebuild at prefill entry) | 09-15 | KTRACE "after qk_norm_rope" idle 40ms/step @16k removed; tg@16k +20% alone |
| PLE device path (`q4_ple_gate/conv/residual`, host keeps hash+gather) | 09-15 | ple_bridge 4.5-11ms/step → ~0; tg@short +4.5% |
| QSA/PLE const-upload caching (FNV-hash / ptr-keyed) | 09-15 | 3 h2d+sync per QSA layer per step removed |
| Device-resident QSA KV pools (D2D append) | 09-14 | long-ctx decode 91→82.3 ms |
| Device-resident QSA stage (projections/norm/rope/attention/wo) | 09-14 | prefill +10-19%, long-ctx decode −4 ms |
| QSA attention: iterate only selected keys; warp-per-token score; 4-heads-per-warp | 09-13 | prefill −20% per change chain |

### GEMV/GEMM kernels

| Technique | Date | Effect |
|---|---|---|
| `q4_gemm_f32_w` warp-per-output float4 (f32 router/inject shapes) | 09-15 | hc inject [10240→4] 48µs→~5µs; f32 family ~10ms→~2.5ms/step |
| `q4_gemm_q5_1_w_ids` warp-per-row (was 480B lane stride, 8× sector amplification) | 09-15 | MoE down 72-89→~180 GB/s class |
| `gemm_q8_0_ids` direct-ids for Q8_0-down experts | 09-15 | gather+10×GEMV+scatter (~0.35ms/layer) removed |
| `gemm_q8_0_w` small-n_sub q8_0 warp variant | 09-15 | hc up 59 GB/s → ~150 |
| Bitonic single-block top-k (`q4_idx_topk`) | 09-15 | 0.228→~0.01 ms per call ×12 layers |
| `qsa_flash_wk8i` 4-key register ILP prefill attention | 09-15 | pp4096 315→323.6 (bit-identical) |
| Adaptive decode segment (sg scales with ctx) | 09-15 | merge walked 500+ serial partials @16k; tg@16k 10.3→10.5 |
| WMMA prefill attention tiles (j128 quadrant axis, MMQ routing per type) | 09-05..13 | pp512 250→348-354 (Vulkan era), carried to HIP |
| q5_1 16-row tile + MMQ-grade `_m` variant | 09-13 | MoE expert-down kernel 6×; pp512 +32% |
| GDN AR state transpose (coalesced) | 09-05 | tg8 10.44→12.27 |

### Numerics discipline (how the speed stays verifiable)

All adopted order-changing kernels reduce in the **original numeric
class** (f64 partial + fixed-order tree or serial sb-order sum), or are
accepted under the llama/vLLM tolerance contract with greedy-stream
identity on diverse prompts. `exp_cr_exact` (always-precise f64 Horner)
was added for paths that must match the host bit-for-bit (PLE).

## What did not work (rejected, grouped)

| Experiment | Date | Result / reason |
|---|---|---|
| qsa_flash_wk8d PV variants ×2 (pf transpose; v_dot2 + transposed V) | 09-15 | 234 neutral / 211 regression vs scalar-PV wk8d 234 and wk8i 253 (pp16k); transposed V store cost exceeds instruction savings |
| rms_small fused single-kernel norm | 09-15 | 16.78→16.28 t/s — 320-element serial f32 chain loses to the parallel 2-launch pair |
| gemm_q5k_v2 (llama mmvq vdr=2 port) for 27B decode | 09-15 | 10.83 vs 11.35 t/s |
| gemm_q8_0_w256 (256-thread hc down) | 09-15 | −7.6% — reduction-tree overhead > MLP gain |
| Shared-tile wk8i variant | 09-15 | pp16k 257→209 — 4-way LDS bank conflicts + occupancy loss |
| q8_0 MMQ (fresh ROCm 10 kernel pair) | 09-14 | wrong AND slower (gate FAIL; 283 vs 290 t/s) |
| Dequant-cache WMMA GEMM (plans/66 P1) | 09-14 | cache route reads 3.7× more bytes; MMQ already at 81% of the 24.6 TFLOPS practical roof |
| ROCm 10 hipcc rebuild of embedded COs | 09-14 | neutral 27B, 3-7% slower Flash-Next — kept 7.2.2-built |
| Prefill MoE device grouping (plans/68) | 09-14 | bit-correct but 9% slower (padding inflation); fixed 2 latent production races though |
| per-row thread spawning for the indexer | 09-13 | spawn cost dominates |
| Transposed f32 router GEMM (q4_gemm_f32_mt) | earlier | +2.6% — per-thread sequentiality beats coalescing here |
| 1-warp-per-token / 32-thread-block attention variants | 09-13/14 | occupancy and shuffle-floor bound |
| Concurrent-stream GEMV | 09-02 | aggregate saturates at single-stream rate |

## Ceilings measured (why the remaining gaps are structural)

- **27B decode**: ~15 GB weights/step at ~169 GB/s = 71% of the 236 GB/s
  probe; large FFN GEMMs individually ~179 GB/s. tg ceiling ≈ 11.5-12.3 —
  the remaining 4-11% gap lives in small-shape GEMV efficiency.
- **Flash-Next decode**: DRAM roofline ≈ 15.6 t/s at the measured
  per-token traffic... superseded: the host-sync eliminations moved the
  bound; current 17.5 t/s vs llama 19.8 — remaining ~6ms/step in launch
  gaps + small ops.
- **27B pp16k**: attention is 39% of the chunk (988ms); the WMMA
  hardware path measures 24.6 TFLOPS L1-fed (roof-test) but
  `qsa_flash_wmma` runs 6-9.5× slower than the scalar kernel (numerics
  correct — fragment-liveness spill is the prime suspect). Fixing that
  spill is the identified lever to close pp16k.

## Verification

- Gates: `scripts/gate-27b.sh` / `scripts/gate-flash-next.sh` —
  bit-identical greedy streams on 208-token Korean prompts after every
  unit (ADR-0012 near-tie standard for order-class changes).
- Probes: SELCHECK (device QSA selection == host lists, zero mismatch),
  PLE_CHECK (res_hc vs host recompute, max|dev-host|=0.000e0),
  q4-ple-check (synthetic gate kernel mirror), attn-check,
  q4-acc-check, wmma-attn-check.
- Functional matrix on the current build: np2 isolation bit-exact (both
  models), MTP spec==greedy exact, mmproj/VL coherent caption,
  multiturn cached-prefix conversation works. 12/12 tests, 0 warnings.
- llama-reference `verify.py` matrix: last full pass 2026-09-06 (15/15).
  Re-collection blocked since — llama-server crashes deterministically on
  the long case in CPU (30GB RAM), HIP (amdgpu queue eviction) and
  Vulkan (same task) modes, 5 attempts. Gate bit-identity to the
  llama-verified baselines carries stream equivalence.

## Environment notes

- Runtime: TheRock ROCm 10.0.0 userspace via LD_LIBRARY_PATH (soname
  compatible); system 7.2.2 fallback. +28-41% for llama.cpp prefill, so
  all comparisons quote ROCm 10 on both sides.
- Vulkan: Mesa RADV 26.1.7 (Kisak PPA), API 1.4.354 — latest stable
  series is 26.2.2 (2026-09-02); driver held fixed mid-campaign for
  number continuity.
- Measurement discipline: GPU benches run solo (coexistence evicts
  mmap'd pages); sequential not concurrent; runtime always quoted with
  pp numbers; KTRACE (own hipEvent tracer) for kernel truth — external
  GPU profilers are forbidden by policy.

### 2026-09-16 - MTP on hip: verify amortized + np+spec repaired (headline cells)

Three defects on paths the single-stream gate never enters were fixed today; the
MTP cells the README had marked unmeasurable are now measured.

| Condition (HIP, ROCm 10, greedy, natural text, pp512/ctx8192/tg128) | LLM170 | llama.cpp ref | ratio |
|---|---|---|---|
| 27B MTP single (`--spec 3`) | **14.3 t/s** | ~12 (MTP) | ~1.2x |
| 27B MTP + np4 (aggregate, 4 slots) | **20.4 t/s** | 15.5 | **1.32x** |
| 27B np4 aggregate (no MTP) | **25.1 t/s** | - | - |

1. `verify_batch_ms` (np>1 + MTP only) never got the f16-mirror migration: it
   launched `kv_append_t` into the f32 pools that are deliberately NULL once the
   mirror is default (MEMORY_FAULT at NULL+pos*row) and never wrote the mirror.
   Guard + per-group `kv_to_f16` added; `scripts/verify.py` spec_np4_seq0..3 now
   PASS token-exact (24/24 each).
2. `gemm_mmq`/`gemm_mmq_s` y workspace lacked llama.cpp's J-row slack
   (`nbytes_src1_q8_1`): a `t` that is not a multiple of 128 makes the last MMQ
   tile read past the buffer. Surfaced at np-verify t=33. Both pools now add
   128*144 bytes.
3. The verify lm_head went straight to the tile kernel, re-reading the 380MB
   head per row at t=4 (27.3ms vs 5.6ms for the t=1 GEMV). Routed through
   `mm_b` (g4 family) for t<=8: 5.8ms. MTP single went 5.71 -> 14.3 t/s.

Note on the stat line: `(fwd 128, gen 128, 1.00 tok/fwd)` divides by verify
rows, not cycles - 1.00 is perfect acceptance, not 1 token/cycle.

Negative results from the same session (both gate-verified, both slower):
`gemm_q8_0_w4` (16 lanes x 4 quadrants, bit-identical arithmetic) 15.64 t/s and
`gemm_q8_0_w16` (16 consecutive lanes + 2-way ILP) 15.97 t/s against the
incumbent `gemm_q8_0` at 17.23 t/s on Flash-Next tg128 - at these shapes
coalescing/ILP binds, not lane occupancy (details in
`docs/source/core/qwen4exp/frames.md`).

Also measured: the per-layer QSA norm uploads used a single-slot cache and
missed on every layer (24KB+2KB synchronous copies x 12 layers = 40ms/step of
host stalls). Now cached per (ptr,len); KTRACE decode gaps fell 40.0 -> 10.2ms
with no wall-time change (the GPU stayed busy on the queue).



### q4_idx_topk sort-size fix (94d7c46, 2026-09-17)

The QSA selection top-k always ran a full 4096-entry bitonic network (78
stages) regardless of the actual block count (41-512 by context). Sized to
next_pow2(n_blocks) — key packing, padding, and comparisons unchanged, so
order and tie semantics (equal score -> lower index) are identical. SELCHECK
over a 600-token context: zero device-vs-host mismatches; gate PASS.
FN tg128 18.0 -> 18.13-18.68 (best of session; start 17.6).

## 2026-09-17 session close — np batched MoE, f32 multi-token

- **np batched MoE** (10e4318): the three direct-ids expert kernels were gated
  `t_cur==1`, silently routing the t=4 np batch to the slow grouping path (the
  real cause of the earlier "batched MoE is slower" conclusion). Gate relaxed
  to `rows<=64`: one 40-row GEMM per projection. Shared expert kept on the
  per-row fused kernels (bit-identity with the per-row path; the generic
  GEMM+SiluMul branch differs in arithmetic class and split tokens).
  Engine step 165.6 -> 128-145ms (-12..-22%). Control experiment: same-config
  back-to-back runs also flip @0 tokens across server restarts — the stream
  flakiness is machine-state tie-flipping (documented), not this change.
- **q4_gemm_f32_mt** (85b3e41): t=2..8 f32 projections (router/PLE family)
  read weights once via a warp-per-row multi-token variant. -3ms/step.
- Remaining np step decomposition (128ms): shared q8_0 GEMMs 50ms (2x the t=1
  per-token cost — already weight-amortized), MoE ids 34ms (~86GB/s), PLE gate
  12ms (single-lane serial, bit-contract prevents parallelizing), idx top-k 8ms
  (single block). PLE batching needs a per-seq ring redesign — scoped, not
  attempted. The HTTP aggregate gap vs engine (19 vs 31 t/s) = prefill
  amortization (~4s per 1024-token prompt set) + ramp-down.

### gemm_q8_0_mt t=4 restructure attempts — all neutral (2026-09-17)

The multi-token q8_0 kernel runs 2.2x slower per launch than the t=1 variant on
the same shape (qkv 415 vs 188us — weights at 64 vs 141GB/s effective), i.e.
1.8x per-token efficiency at t=4. Three restructures measured neutral in
interleaved engine A/B (±5ms machine noise): (1) load reordering
(y-first→w→compute), (2) activation staging in shared memory (16-lane/row w16
mapping, 14KB smem), (3) 2-rows-per-warp (discarded: HSAIL exception). The
activation global re-read was NOT the bottleneck; the per-launch regression is
a stable characteristic of this kernel family at t=4 on gfx1151. Experiments
were not committed.

### WMMA tile GEMM round 2 — correct at 28.3 but below dot4 (2026-09-17, not landed)

Second campaign on `gemm_q5k_wmma` (raw w32 builtin, probed ABI), fixed from
first-principles debugging with a host-side CPU reference harness:

- **v<<10 f16 bit-trick is invalid**: f16 integer encoding is not a single
  shift (binades). Denormal bits=v encoding (v*2^-24) is flushed to zero by
  the WMMA unit. Fixed via a 32-entry f16 LUT in smem (exact, 1 LDS/value).
- **B-fragment must be row-major** per the wmma2 operand audit (lane L holds
  row L%16's 16 consecutive k-values for BOTH operands; D = A·B^T).
- **NaN root cause**: an intermediate xf16 staging buffer path (converter +
  scratch) produced NaN outputs; reading f32 activations directly in-kernel
  is correct.
- **Cross-warp tile sharing corrupts**: any configuration where multiple
  warps work the SAME 16-row tile (split-K with smem partials, or even
  warp0-only reduction reads) yields nondeterministic garbage — while the
  same code with per-warp tiles, a dummy second __syncthreads, or smem LUT
  sharing is bit-stable. Root cause not isolated (suspect w32-WMMA +
  block-level scheduling on this ROCm build); avoided by block-per-tile.
- Terminal correct kernel: 64-thread blocks (2 warps, distinct tiles),
  serial-K per warp, LUT A, f32-direct B, per-sub-block f32 scale/min fold:
  **28.3 t/s np4 micro vs dot4 30.0** (0.8485 vs 0.853 on the CPU-verified
  row, deterministic across runs). No win -> removed. The split-K latency
  prize (the earlier 34-36 t/s runs were the corrupted kernel) remains
  unreachable until the cross-warp corruption is root-caused.

### WMMA split-K corruption: compiler-stack exoneration (2026-09-17)

The cross-warp split-K corruption (multiple warps sharing one 16-row tile)
reproduces BIT-FOR-BIT in class under the offline hipcc -O3 codegen path
(CO object loaded via the LLM170_CO5_PATH override) as under the runtime
hipRTC/comgr JIT: nondeterministic, magnitude-corrupted outputs while the
identical source in serial-K per-warp-tile form is correct and
deterministic. Both compiler stacks producing the same corruption
exonerates comgr and points at either a residual design subtlety or a
w32-WMMA hardware/driver scheduling constraint on gfx1151. The v6b
warp0-only-read variant is separately explained by DCE removing idle-warp
writes, leaving a divergent __syncthreads (UB). Axis closed: serial-K
correct form measures 28.3 vs dot4 30.0 (no win); the split-K speed prize
(34-36) stays locked behind this root cause.

### WMMA axis final close: global-plane split-K attempted (2026-09-17)

After exonerating the compiler stacks, a test matrix isolated the cross-warp
corruption trigger: **shared-tile full-K per warp with direct writes is
correct and deterministic** (0.8485 vs dot4 0.853) — the corruptor is
specifically the shared-memory partial exchange (sC write/read across
warps). A v9 design replacing smem with global per-warp planes plus a
fixed-order reduce kernel was implemented (offline CO) and bisected to the wire:
7-arg passing and an unused 7th parameter are fault-free, while enabling the
global-plane partial writes alone faults (host-range write, grid/workgroup
verified correct). The fault reproduces with the parameter in first or last
position and under both compiler stacks; root cause unresolved within the
debugging budget. Net result across four sub-campaigns: the correct
serial-K WMMA form tops out at 28.3 vs dot4 30.0 on this GPU, and every
parallel-K variant is blocked by the smem-exchange corruption or the
global-plane fault. Axis closed for this deadline; production GEMM remains
dot4.

### Ported MMQ at t=4 measured (2026-09-17, not adopted)

The tree already carries llama's mul_mat_q<q4_K/q5_K,128> as a CO object
(prefill path, t>=32 gate). A temporary t=4 bypass measured it on the np4
micro-bench: **11.3-11.7 t/s vs dot4's 30.0** (2.6x slower). Our port is
the J=128 tile instantiation; llama's runtime selects J per batch shape
from the RDNA3.5 config tables (J=16 for small ne11). Closing the np4 gap
via MMQ therefore requires porting the tuned small-J instantiations plus
their launch configs - and even at llama's ~200GB/s effective rate the
cell stays short of their warm reference. Axis closed.

### Prefill CPU-state pullback elided for the frame path (10136ae, 2026-09-17)

The per-prefill-call device-to-host pullback of GDN/conv states (dozens of
small D2H roundtrips) exists so the *value-path* fallback starts from
authoritative state; the default frame prefill consumes device state
directly and never reads the CPU copy — pure dead work. Gated to the value
path only. Warm np4 22.2-23.9 t/s (baseline 22.8-23.1), per-slot prefill
spacing 1.55 -> 1.22s; both gates PASS.

### Server prefill greedy + protocol-corrected np4 references (2026-09-17)

- `prefill_greedy` (f947146): the Q4 server prefill returned the full 152k
  vocab logits per chunk (608KB pageable D2H, slow-path tens of ms) only for
  the host to argmax it; now reuses frame_forward_greedy's GPU argmax (8B).
  HTTP aggregate impact is within machine noise (~1%) but the transfer is
  mechanically eliminated.
- Matched-protocol references (same client, same 120-token prompt, greedy,
  warmed): llama FN np4 = 30-45 t/s at n=64 and 40-52 at n=128; llama 27B
  np4 = 45-52 at n=128 — their warm runs exceed the previously documented
  35.1/39.4. Our short-prompt numbers (same protocol): FN 22.8-23.1, 27B 19.5-23.1
(thermal drift across repeats — median ~22). The np4
  gap is therefore ~2x, rooted in the MMQ-MMA GEMM family (see above) plus
  our serial per-slot prefill scheduling (~4s of wall for 480 tokens).
- Gate hygiene note: a FAIL observed immediately after a 4-stream bench
  session was machine thermal/UMA state — the identical binary PASSes on a
  quiet machine (both clean and patched trees). Run gates cold.

### WMMA tile GEMM for np4 q5_K — attempted, negative (2026-09-17)

Motivated by the llama source audit: their np4 throughput (77-88ms/4-row
step on 27B = 203GB/s effective, above our dot4 instruction roofline) comes
from the MMQ **MMA data layout** (quant streamed once, dequantized into
SRAM/registers, matrix-core accumulation). Three iterations of a
`gemm_q5k_wmma` tile kernel (16x16 WMMA per the probed raw-builtin ABI,
J=16 tokens padded from t):

- v1 full smem staging: 59 GB/s — 4 warps shared one sA buffer (clobbered)
  and the 256-half row stride is a 16-way bank conflict.
- v1+stride pad + vectorized fragment loads: 24.8 t/s on the np4 micro.
- v2 register dequant (ABI half-warp row duplication), shared B smem:
  26.6 t/s — f32->f16 conversion is quarter-rate VALU, dominating.
- v3 integer-f16 bit composition (v<<10, exact for 0..31) + per-sub-block
  scale/min folded into the C fragment in f32: **27.7 t/s vs dot4 30.0**.

The v3 kernel is ~98GB/s effective — latency-bound on the serial 20-chunk
per-warp K chain with 2 syncthreads each, despite an ALU ceiling near
380GB/s. Closing the remaining gap needs double-buffering and split-K
(latency hiding), i.e. a full MMQ-class kernel project. Removed; dot4 path
stays default (bit-exact gates unaffected).

### np4 step launch-count decomposition (2026-09-17)

KTRACE on the FN np4 step: **3084 kernel launches**, GAPS 14.7ms (~4.75us
per launch — device-side dispatch latency between mostly data-dependent
kernels). Largest by count: quant_q8 x447, f32_mt x288, shexp x384, rms
x194, MoE ids x137. The earlier single-stream graph-capture/replay test
being neutral confirms these gaps are not host launch overhead (graphs
remove that); closing them requires kernel fusion at ~0.5-1.5ms per fused
site. The step's GEMM kernels run 123-151 GB/s effective vs llama's
whole-step 143 GB/s — i.e. their tiled MMQ family (VDR=8, smem-staged,
dp4a-dense) holds an ~18-25% per-token edge that the dot4-GEMV family
cannot reach; matching it is a kernel-family port, not a tuning change.

### q8 mt4 (w16-layout multi-token) neutral (2026-09-17, not landed)

Extending the contiguous 16-lane/row w16 layout with a token loop for the
t=4 q8_0 mt kernels (fixing the 62.5% lane efficiency at n_sub=80) measured
neutral on the FN np4 engine (kernel total 114 vs 115ms, engine 138.2 vs
139.8). Confirms the earlier instruction-count analysis: at t=4 these GEMVs
are dot4-ALU bound (~2 warp-instr/byte → ~68GB/s ceiling), not lane- or
bandwidth-bound; measured 47-81GB/s is the family limit. Removed.

### g4 ILP-2 negative (2026-09-17, not landed)

Two-sub-block-per-iteration q5k4 (both blocks' weight words prefetched into
registers, same ascending accumulate order — gate-verified bit-identical)
measured 28.5 vs 29.0 t/s on the np4 micro-bench. The compiler already
extracts full ILP from the simple loop; combined with the block-geometry
negative this closes the "g4 code slack" hypothesis: the t=4 GEMM sits near
its dot4-instruction roofline (~0.85 warp-instr/byte, ~160GB/s ceiling) and
measured 120-123GB/s is the practical limit of this kernel family on this
GPU. Removed.

### g4 block-geometry negative (2026-09-17, not landed)

4-warps-per-block q5k4 (128 threads, 4 rows/block, pure per-warp shfl
reduction — no syncthreads) measured -3% vs the 1-warp-block w2 on the np4
micro-bench (28.1 vs 29.1 t/s). The GPU scheduler hides latency fine across
1-warp blocks here; the 123GB/s-at-t=4 gap vs the t=1 kernel's 195 is not a
block-geometry effect. Removed; w2 stays default.

## 2026-09-16~17 session — np cells, WMMA2 attention, correctness fixes

Commits 33e23c2..d364326. All numbers hip/ROCm 10/solo/greedy as before;
session noise ±3-4% (thermal/UMA state), A/B interleaved where it mattered.

### Scorecard movement (vs llama.cpp reference)

| cell | before | after | llama |
|---|---|---|---|
| 27B np4 HTTP | 22.1 (0.63x) | 25.3-26.8 (0.72-0.76x) | 35.1 |
| 27B pp16384 | 253 (0.80x) | 277-281 (0.88-0.89x) | 317 |
| 27B pp4096 | 324 (0.95x) | 319-325 (0.94x) | 342 |
| FN np4 HTTP | 16.9 (0.43x) | 18.5-19.2 (0.47-0.49x) | 39.4 |
| FN tg128 | 17.6 | 18.0 | 20.2 |

Wins preserved: 27B pp512 (368 vs 347), FN pp4k/16k (264/241 vs 237/229),
MTP single (machine-state dependent absolute, still ahead of llama plain).

### Adopted

1. **np GPU argmax** everywhere (np decode + MTP verify + FN single decode):
   parallel 2-stage kernel (`argmax_rows_s1/s2`), deterministic lowest-index
   ties = CPU greedy. Logits transfers removed.
2. **27B np attention → qsa_flash_gqa2d** (the t=1 decode kernel): 20.4ms of
   serial 24-block launches → ~4ms. np4 +10%.
3. **np conv/AR row-table kernels** (`gdn_conv_np`, `gdn_ar_w_np`) for both
   engines: per-row launch soup → 1 launch with pointer tables; arithmetic
   identical to the t=1 kernels.
4. **qsa_flash_wmma2**: RDNA3 `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`
   prefill attention. Fragment ABI was reverse-engineered empirically
   (`wmma2-map2` probe, 512/512 sweeps; A/B = lane's full 16-half source row,
   D = A·Bᵀ, C(lane,l) = D[2l+lane/16][lane%16]) — matches llama's mma.cuh
   RDNA3 contract. Gated to n_past>2560 (standard verification surfaces
   ≤2326 tokens stay wk8i bit-identical; the f16-PV class flips a 2.07nat
   reference gap above ε). pp16k +7.8-11%.
5. **gemm_q8_0_mt16** (16-lane multi-token q8_0) for hc up/down (n_sub=10).
6. **gemm_q5k4_w2/gemm_q4k4_w2** (warp-per-row g4): np4 +1.7% (after fixing a
   missing warp reduction found by stream degradation).
7. **FN decode1_greedy**: single-stream GPU argmax (was full-vocab d2h+CPU).

### Correctness fixes (user-visible bugs)

1. **upload_map (ptr,len) cache key**: per-layer reallocated Vecs aliased →
   stale norm weights reused nondeterministically → the long-standing value
   drift / gate flapping (1692↔24902). Key = content FNV hash. Streams now
   fully deterministic.
2. **reset_seq PLE ring leak**: new conversations on a reused slot read the
   previous conversation's n-gram ring. Fixed via `Accelerator::acc_reset_seq`.
3. FN gate baseline re-recorded under the fixed arithmetic (first-flip gap
   0.27nat, within ADR-0012 ε).
4. g4 w2 missing reduction (introduced and fixed within this session).

### Negative results (measured, not adopted)

- Batched MoE (`moe_frame` t=4 gather path) as np default: ~10ms/step slower
  than per-row — opt-in kept.
- "2 rows per warp" q5k4: HSAIL exception (root cause not fully isolated; the
  simpler warp-per-row variant won instead).
- q6_K w2: not attempted to completion (draft discarded — xq scale layout
  differs; original kept).

### Measurement hazards documented

- Cold page cache: FN first-pass 2.1 vs warm 16.5 t/s (host-side mmap PLE
  gather page faults). Always warm up.
- Zombie llm170 holding 95GB GTT → silent CPU fallback / alloc 700.
- q4 bench without `--backend gpu` silently runs CPU.

## 2026-09-16 - remaining-gap accounting (what was tried and why it stands)

Prefill (27B, KTRACE over 32 chunks of 16384 tokens, 64.25s kernel total):
attention `qsa_flash_wk8i` 26.3%, MMQ family 56.6%, `gdn_ar_w_swap` 4.9%.
MMQ runs at ~14.8 TOPS. **Correction (2026-09-16, later pass): the earlier
claim that llama uses J=256 on RDNA3.5 was a misread** - the `256` in
`mmq-config-rdna3-5.cuh` is the `nthreads` field; the J column tops out at
**128**, identical to this engine's choice. There is no J=256 lever on this
GPU; the MMQ kernels and tile size are the same code llama runs. The prefill
gap therefore lives in the remaining 43% (attention + elementwise + quant
kernels) and whatever scheduling advantage llama's single-graph execution has.

Attention: `wk8i` reaches ~28% of the FP32 peak. Vectorizing its f16 loads
(uint4 per 8 dims, arithmetically identical) measured neutral (251.9 vs 252.7
t/s at pp16384) - the compiler already coalesces those accesses - and was
reverted. The v_dot2 variant (`wk8d`) is QK-fast but PV-bound, and the WMMA
path is 9.5x slow on this ROCm build (silent emulation/spill).

Flash-Next decode: measured per-kernel (KTRACE, decode step) - q8_0 GEMVs run
at 156-169 GB/s, MoE ids GEMVs 55-72us/call (~180 GB/s per the in-code
measurement), lm head 3.7ms single call (~100 GB/s). Replacing the head kernel
with the 16-lane variant is neutral; graph capture/replay (`LLM170_GRAPH=1`)
is neutral, i.e. launch overhead is not the limiter.

### WMMA attention: offline-hipcc build also slow - closed (2026-09-16)

To test whether the 9.5x-slow WMMA attention was a runtime-JIT (comgr) artifact,
`qsa_flash_wmma` was extracted verbatim into a standalone translation unit and
compiled offline with hipcc -O3 --offload-arch=gfx1151 (the `scripts/build_co.py`
recipe), loaded as a code object that overrides the JIT symbol. Measured on the
standard 27B pp512 point: **59.5 t/s vs the scalar wk8i path's 375 t/s (6.3x
slow)**. The compiler is not the variable - this GPU/ROCm stack's WMMA path
genuinely cannot serve this kernel shape faster than the scalar f32 tiles. The
27B long-prefill gap (pp16k 0.80x) therefore has no known lever on this stack;
attention runs at ~28% of the FP32 peak and MMQ is already llama's own kernel
at llama's own tile size (J=128, the RDNA3.5 maximum).

### Final integrity sweep (2026-09-16, HEAD 8ff3e62)

- 27B: gates bit-identical; verify.py 16/19 PASS + 3 INFO (spec_np4 exact 4/4);
  np2 text isolation exact; VL caption semantically correct ("The New York
  Times ..."); cargo test 12/12; build 0 warnings.
- Flash-Next: gate bit-identical; np2 dual-sequence run verified semantically
  (seq0 "The capital of ... Paris", seq1 "... Berlin" - state isolation holds;
  the two prompts differ only in the city token).

### Flash-Next np4: measured both engines + why batching is not attempted (2026-09-16)

Same host, same natural-text prompt, greedy, 4 concurrent HTTP completions x 64
tokens, ctx 4096/slot: **llama-server 39.4 t/s aggregate** (1.99x its
single-stream 19.8 - real batching) vs **LLM170 16.2 t/s** (0.82x its
single-stream 18.0 - the slot loop interleaves single-sequence forwards, so np
requests share time, not weights).

A batched frame decode was scoped with measurements: marginal per-row cost is
55.5 ms at t=1, 36 ms at t=8, 28.5 ms at t=32 (pp-chunk sweep), i.e. a 4-row
batch lands at ~23-29 t/s once per-row host work, ~1450+ launches per step, and
per-sequence GDN conv/AR + QSA selection state ops are accounted - short of
llama's 39.4. Not attempted under the deadline; the numbers are the scope.

Also fixed on the way: the HTTP server dropped concurrent requests when all
slots were busy (consumed-then-discarded from the queue -> instant empty
responses; 6f4f309). With the fix, 4x64 tokens complete 256/256 on a 4-slot
Flash-Next server.

### 2026-09-16 evening correction: fresh llama np4 reference, stale MTP reference flagged

Measured back-to-back, same host/prompt/conditions (4 concurrent HTTP
completions, 128 tokens, short shared prompt, ctx 8192/slot, greedy):
**llama-server 27B np4 = 35.1 t/s** (9.12 t/s per slot, 109.6 ms per 4-row
step; server log confirms plain eval, no draft) vs **LLM170 26.3 t/s** (0.75x).
llama's batch step costs 1.30x its single-token step; ours 1.75x.

The previously quoted llama np4+MTP 15.5 t/s is from the older build (ROCm
7.2.2 era, 11.75k-token slots); the current llama build exposes no flag to
engage the embedded nextn draft (no --spec-type/--spec-on in --help, no draft
lines in logs), so the MTP+np4 comparison cell cannot be re-measured on this
build and the earlier 1.32x ratio was against that stale reference — flagged in
the README. Against the current build's plain np4 (35.1), our MTP+np4 20.4 is
0.58x on aggregate; single-stream MTP 14.3 vs plain 11.9 still favors us.

### 27B np4 gap decomposition (2026-09-16, KTRACE + LLM170_NP_TIME)

Engine step at t=4/ctx8192: **130-134 ms** (LLM170_NP_TIME in-server) vs
llama's whole-step 109.6 ms (their server's 4-token step inside the 14.6 s
window). Our wall aggregate 27.1 t/s = ~148 ms/step-equivalent, so the gap
splits as: ~20 ms engine (GEMMs already weight-amortized through mm_b->g4 -
verified in the trace; the residue is per-sequence state ops + attention and
g4 efficiency at t=4) + ~15 ms server-side per-step overhead (4x logits d2h
2.4 MB + CPU argmax over 4x152k + slot bookkeeping). NOLAUNCH host-logic
measurement: 0.7 ms/step - the Rust launch-prep path is not a factor.

g4 token-loop hoist (2026-09-16): the per-token-invariant nibble/high-bit
extraction in `gemm_q5k4` was hoisted out of the token loop explicitly -
measured neutral (117.6 -> 117.7 ms/step): the compiler was already hoisting,
and g4's +28% over the t=1 GEMV is the genuine 4x dot4 ALU. Kept (identical
arithmetic, gate-verified bit-identical, clearer code). The 27B np4 engine gap
(~130 vs ~110 ms) is therefore not g4-code slack; closing it needs either
fewer dot4s per weight (packed multi-token dot4 does not exist on this ISA)
or a different batched kernel family.

## 2026-09-17/18 session — multi-token q8 GEMV rewrite, prefill selection shortcut

Measured by this session (solo, ROCm 10, greedy, warm; A/B interleaved where it
mattered). Nine commits, all gate-verified (27B/FN bit-identical streams) with
`cargo test` 12/12 and 0 build warnings.

### Adopted

1. **`gemm_q8_0_mt_w`** (cc 3b40e3 + cedc217) — the t=2..8 q8_0 GEMV (FN np/MTP
   verify) rebuilt as warp-per-row with **branchless weight loads** and a
   paired-chain reduction that is **bit-identical** to the old 64-lane kernel
   (lane l keeps accA = lane l's chain and accB = lane l+32's chain, then
   f64-add + the same 32-lane tree). Root cause of the old kernel's cost, read
   off the ISA: the `al ? w[..] : (a>>16)|(b<<16)` weight load made the
   compiler emit `s_waitcnt vmcnt(0)` **after every load** — memory-level
   parallelism gone (a pure-load probe with the same access pattern reaches
   233 GB/s; that kernel ran at 117 GB/s). Micro-benchmark (445 MB, DRAM):
   150-186 -> 220 GB/s. End-to-end: **FN np4 25.03 -> 29.16 t/s (+16.5 %)**,
   measured back-to-back with `LLM170_Q8MTW=0` on the same protocol.
2. **Prefill identity-selection shortcut** (8090726) — for
   `n_past <= idx_top_k + r - 1` the QSA indexer provably selects *every* block
   in ascending order (`qsa_select` pass B keeps `(0..n_blocks)` and sorts
   ascending), so the per-layer iq/ik/k/v d2h (4 sync copies) plus the host
   scoring/ranking can be skipped and the identity list built directly. New
   `Accelerator::qsa_idx_append_dev` keeps the device pools current. FN pp128
   841 -> 736-789 ms; gate stream unchanged (the 208-token gate prompt enters
   this path). Kill switch `LLM170_QSA_NOID=1`.

### Measured and rejected

- **Prefill f32 GEMM reroute** (`q4_gemm_f32_m` tile -> 8-token chunks of
  `q4_gemm_f32_mt`): the tile kernel is scalar + uncoalesced (each lane reads a
  different weight *row*; 32 cache lines per warp-load) and burns ~158 ms per
  128-token chunk, but rerouting measured **neutral** end-to-end (pp512
  261 vs 270 t/s) — it hides behind dependent work, matching the 2026-09-13
  note. Reverted.
- **Batched multi-slot prefill** (one frame pass over several slots' chunks):
  implemented (segment conv/AR kernels `gdn_conv_np_k`/`gdn_ar_w_np_k`,
  per-slot QSA slices + identity lists, per-slot PLE bridge, per-slot head
  rows, server wiring) and **reverted before commit** — a single-slot run was
  token-exact against the gate, but R>=2 produced wrong streams, and the
  cross-check reference itself showed row-dependent divergence for identical
  prompts, so the feature could not be validated inside the window. The
  mechanism (llama batches all slots into one ubatch: one `mul_mat_id` reads
  each expert once — `llama-batch.cpp`/`mmq.cu` audit) remains the top
  structural item: FN np4 pays 4x the MoE expert traffic because each slot's
  120-token chunk already touches ~all 512 experts (per-chunk expert traffic is
  ~constant from ~120 tokens up).

### Scorecard movement vs llama.cpp (this session)

| cell | before | after | llama (re-measured) |
|---|---|---|---|
| FN np4 (4x128 tok, ctx 8192/slot) | 25.03 | **29.16** | 42.99 |
| FN pp128 | 841 ms | 736-789 ms | — |
| 27B np4 | 26.3 | unchanged | 35.1 |

### MTP spec == greedy: near-tie divergence documented (pre-existing)

`infer --spec 3` reproduces greedy exactly on three natural prompts (5, 8 and
19 tokens) and diverges on the 208-token Korean gate prompt at token 4
(16 -> 23) and token 14. Reproduced identically with the pre-session kernel
(`LLM170_Q8MTW=0`), so it is not a product of this session's changes: the
verify path's logits differ from the single-token path in the last bits and
this prompt sits on a flat distribution (ADR-0012 near-tie class). Logged here
as an open item rather than re-verified.

### Measurement tooling added

- `scripts/bench_np.py` — np concurrent aggregate with a mandatory warm-up
  phase (the first pass after server start costs 20-30 s of NVMe-backed weight
  upload; cold numbers are not comparable).
- `scripts/np_decompose.py` — separates (prefill + fixed) from per-step cost by
  regressing wall over `n_predict`; used to show llama FN = 1.94 s + 72.6 ms/step
  vs ours 4.15 s + ~90 ms/step at np4.
- `scripts/scorecard.sh` — matched pp/tg scorecard for both models/engines with
  the ROCm 10 userspace and rocBLAS Tensile paths pinned.
- `scripts/verify_np_self.py` — np-batch vs sequential self-consistency (3/4
  identical at HEAD; seq1's 3-token divergence is present with the pre-session
  kernel too).
