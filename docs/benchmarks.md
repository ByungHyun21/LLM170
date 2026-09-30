# Performance vs llama.cpp

Conditions always quoted (context / batch / quantization / backend). All
numbers on the dev machine (Radeon 8060S, gfx1151, 32-thread CPU) unless
noted. Per-file measurement records live in `docs/source/` (1:1 with code
paths). The chronological originals were moved to
`docs/archive/benchmarks-chronological-2026-09-15.md` — nothing was
deleted, only regrouped here.

## Current scorecard (2026-09-18, CLI-to-CLI: `llm170 bench` vs `llama-bench`)

Protocol change (user decision 2026-09-18): README numbers are internal-bench
measurements only — `llm170 bench` for ours, `llama-bench` for llama.cpp, one
full-shape warm-up before each measurement (llama-bench parity; the old
64-token warm-up understated our first-in-session pp512 by up to 20%).
Flash-Next llama rows require the qwen4exp-capable local build
(`/home/yoon/local_llm/llama.cpp` @2cc83c6f4, `build-ab2`,
`-ot per_layer_token_embd=CPU --load-mode mmap`); master's llama-bench
rejects the `qwen4exp` architecture. llama-bench has no multi-slot mode, so
llama np4 columns remain HTTP (`scripts/bench_np.py`) and are marked (HTTP†)
in the README — ours are engine-API `bench --np 4` rows.

Session values are in the README tables (2026-09-18). Notables: 27B pp512
ours 367-370 vs llama 340; FN all cells ahead (pp4096 275-278 vs 210,
tg@16k 18.5 vs 13.9); 27B np4-tg engine batched 32.1 (supersedes the HTTP
26.67 — the prior bench tool measured sequentially); MTP+np4 engine merged
path **9.58 t/s** (2026-09-30 re-measure, `--pp 512 --tg 128 --spec 4
--np 4 --reps 3` median, with the repaired spec+np timing anchor —
plans/114 QA-15: the pre-fix 6.5-6.8 cells folded the np prefill
aggregation into the tg timing and are void; the prior 20.4 HTTP figure is
a different protocol/session).

### Tool fixes that made the CLI scorecard possible (2026-09-18, plans/79)

- `bench --np` now drives the qwen35 aggregate too (it previously read only
  `LLM170_BENCH_NP`, so `--np 4` was silently ignored for 27B).
- qwen35 np decode aggregate added (multi-seq batched `decode`, matching the
  server slot loop — sequential decode1 was 4× slower and not the protocol).
- Full-shape warm-up in both model branches (llama-bench parity).
- `scripts/scorecard.sh`: `-ngl all` → `-ngl 99` (master llama-bench rejects
  the literal).

## Vulkan compute campaign (2026-09-22, plans/89 — engine bench, solo)

### Qwen3.8-Flash-Next (Q4_K_XL) — vk vs session start

| cell | session start | now | hip | llama vk |
|---|---|---|---|---|
| pp512@20k | 60.1 | ~180 | 231-275 | 234 |
| pp4096@20k | 54.7 | 178.6 | 276 | 347 |
| pp16384@20k | — | 158.6 | 246 | 332 |
| tg128@8k | 7.51 | 17.9 | 18.4 | 23.2 |

### Qwen3.8-27B (UD-Q4_K_XL)
| cell | vk now | hip | llama vk |
|---|---|---|---|
| pp512@20k | 336.9 | 363 | 343 |
| pp4096@20k | 257.9 | 319 | 318 |
| pp8192@20k | 203.5 | 315 | 301 |
| pp16384@20k | 145.9 | 292 | 273 |
| tg128@4k | 11.61 | 11.53 | 12.05 |
| np4 agg (plain) | 11.44 | 32.1 | 15.5 *(HTTP†)* |

(MTP on the Vulkan accelerator backend is unimplemented —
`mtp_prefill_batch: 미지원(백엔드)`; the "—" cells in earlier snapshots were
unmeasured, not failures. 27B pp16384 is bounded by the f16 tile_ms128
family — [ts] profile: tile_ms128 21.6s, tile_q8128 15.6s, tile_xs128 10.7s
per pp2048@4k. Integer s8-coopmat rewrites are closed on this stack (RADV
emulates s8 coopmat, 2.3x slower — see docs/decisions.md (32f)); the next
lever is an OpSDot scalar MMQ tile family with register blocking.)

## Prior scorecard (2026-09-15, ROCm 10 userspace, greedy, solo) — superseded by the CLI scorecard above

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

## Chronological session logs

Moved to `docs/archive/benchmarks-chronological-2026-09-17-plus.md`
(per-session measurement detail; the curated summaries above carry the
verdicts).
