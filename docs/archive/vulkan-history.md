# 기록 보관 — vulkan-history

> `docs/benchmarks.md`에서 옮긴 이력 항목(제목 기준). 수치·표는 원문 그대로.
> 최신 지표는 `docs/benchmarks.md`, 파일별 실측은 `docs/source/`를 본다.

## Vulkan tile-correctness round (2026-09-09, plans/38 A2)

A new `vk-tile-check` harness (per-type tile kernel vs CPU dequant f64 dot)
found two corrupt tile kernels that the earlier gates had missed: tile_q6k
(uint zero-extension dropping the sign of negative i8 scales, plus a global
sb used as a block-local half index — garbage beyond block 0) and tile_xs
(reading 32B of a 16B sub-block, overwriting columns with the next
sub-block and OOB-writing shared rows). Both fixed; all 8 tile types now
maxrel ≤ 0.007 at t=1..64. The np4 prefill degeneracy (degenerate repeated
tokens for ≥16-token prompts) is gone and the spec path is deterministic;
the remaining spec-vs-plain near-tie divergence is a separate, deterministic
MTP numeric-order issue (vulkan-only; HIP exact). Re-assessment note: the
previous session's "vk judge 19/19×7" runs used the harness default runtime
(HIP) — the true vulkan gates start this round. Also this round: the ctx²
f32 mask was removed (qsa_flash's causal loop bound already masks — 256MB at
8k context), pp64 142 t/s / tg4 7.4 on the corrected tiles.



## Vulkan perf-parity round (2026-09-09, plans/39)

Five structural tile experiments against the llama mul_mm gap (tiles = 72%
of chunk GPU time, 38 vs llama 98 GB/s effective weight streaming): T_MAX
128 + 128-token tiles (clean, flat), an f16 pre-dequant weight cache
(63 GB/s streaming but 2x bytes = net loss, opt-in `LLM170_VK_F16W=1`),
direct coopMatStore ColumnMajor drain (unreliable on RADV — both layout
interpretations corrupt), a 2-WG/CU occupancy variant (neutral), and a
family rollout of the q5 kernel structure to all tile types (verified
maxrel<=0.007, flat). Verdict: the remaining 2.4x is inside the llama
mul_mm kernel structure (32x32 warp tiles, WMITER, BK=32, 16 KB shmem,
split_k) — a full port is the single remaining pp lever. tg levers
(addrms+quant fusion) untried. Feature gates on the rolled-out path:
np4 clean, spec divergence unchanged (known f16-prefill class), mmproj vl
reads the NYT moon-landing headline correctly. Side RCA: an init-time
17.5 GB host-heap clone in the f16w cache OOM-killed the 30 GiB host
(fixed by borrowing); heavy llm170 processes must run one at a time.

Second pass (same day): q8_0 KV cache (kv_append_q8 + dequant-on-read
qsa_flash_q8, opt-in `LLM170_VK_KV8=1`) — 3.76x KV bytes cut (2048-ctx
256→68 MB), tg32 neutral (weights-bound), early-token divergences at short
context are the quantized-KV quality class; kept opt-in for long-context
capacity. The 256-thread occupancy tile variant (tile128v2) is parked with
strong evidence of a WG-size-dependent codegen issue: byte-identical
staging code is clean in the 512-thread family and corrupts odd rows k≥16
at 256 threads (subgroup geometry verified 64x4, store semantics and dump
machinery eliminated via constant injection).



## Vulkan execution-round gate (2026-09-08, plans/36 G1-G4/P1-P4)

Full plans/36 round on q35work.gguf, 3-run medians, zero-config defaults
(batch prefill + coopmat tiles now default; kill switches VKD_BATCH=0 /
VK_NOTILE=1): **tg32 7.21 t/s** (from 7.06), **pp512 122.1 t/s** (from
65.95 opt-in, +85%). Gates: vk judge 19/19 (seven consecutive rolls incl.
VKD_BATCH+TILE and T_MAX=64 configurations), HIP judge 19/19, cargo 10/10.

What moved the numbers (per-kernel GPU times via a new VK_QUERY_POOL
timestamp profiler, LLM170_VK_TS=1 — the host-side ktime attribution was
shown to be split-boundary garbage):

- **Batch prefill + tiles promoted to default** — the 2026-09-04 batch
  divergence was the descriptor-set reuse race (fixed 2026-09-05); not
  reproducible since. Verify stays per-token by default (race isolation).
- **T_MAX 32→64**: chunked prefill re-read all weights per 32-token chunk;
  64-token chunks halve weight traffic. pp512 68→97 t/s.
- **Tile type coverage completed**: tile_nl (iq4_nl), tile_iq3s, q8 small-n.
  The straggler gemv3 calls (iq4_nl/iq3_s/q8_0 at 0.2-3 GB/s) were 36% of
  chunk GPU time. pp512 97→122 t/s.
- **gemv8_q6**: llama mul_mat_vec_q6_k direct port with a u16 block view
  (105 u16/block) — the 210-byte unaligned assembly that capped the old
  attempt at 67.6 GB/s is gone. 160 GB/s, max|D|=0 vs CPU at t=1. q6_K
  decode weights and the output head leave gemv3. tg32 6.87→7.21.
- **addrms fusion** (residual add + rms in one pass, bit-identical by
  construction), head merged into the trunk batch (single submit), batch
  auto-split 512→2048, independent-group barrier skip (gemv stages, kv
  append), lazy stage quantization (dead quant removed).
- **tile128 2-sb staging**: 64 k-elements per barrier pair (barriers halved,
  MMA chains doubled). LDS 59.5 KB; split_k was ruled out by measurement
  (46-59.5 KB LDS → 1 resident WG/CU, queuing hides nothing; double-buffer
  ping-pong was neutral).

Remaining prefill gap vs llama-vk raw loop (358): tile MFU ~8 TFLOPS vs
llama ~21.5 — a compute-efficiency frontier, not bandwidth.

i8 coopmat GEMM (plans/23) verdict: v1 per-block drain 29 t/s, v2 row-scale
(drain-free) 82 t/s — confirming the drain as the v1 bottleneck — vs tiles
126+ t/s. The i8 path stays experimental opt-in; tiles supersede it for
prefill across all 8 quant types.

Known open: the gemv8 t≥2 check-harness divergence (deterministic per
binary, in-process stable, all types; engine path exact through VKD_BATCH
spec gates) — recorded with the §8 flake; reproducer:
`llm170 vk-gemv8-check <gguf> blk.4.attn_gate.weight 2`.



## Known flake: vk spec==nonspec nondeterminism (2026-09-08)

`--gpu-runtime vulkan --spec 4` np4 runs intermittently flip one near-tie
token (spec_np4 divergence at a flat distribution point). Measured with a
fixed seed repro (`plain vs --spec 4`, 4×short2-class prompts, 24 tok):
**pre-refactor 7758e07 diverges 2/5 runs; post-refactor 4/5** — the defect
predates the plans/35 refactor, which only shifts timing (non-spec streams
are bit-identical across the refactor). `LLM170_G8=0` (verify batches fall
back to quant+gemv3) did not diverge in 2/2 runs — the trigger is the
gemv8 batched-verify configuration (t=2..5 rows), not the decode path.
Follow-up tracked in plans/38 A2 (deterministic harness reproducer). The vk judge gate therefore reports
18/19..19/19 depending on the roll; HIP is unaffected (19/19 stable).



## Vulkan — FUNCTIONAL (2026-09-07, plans/29)

The Vulkan path is now correct end-to-end after fixing the VkDecoder head
(two stacked bugs, commit 19f68bc):

1. n_vocab derivation read tuple field 4 (n_in=5120) instead of 5
   (n_out=248,320) — the head argmax'd over 5,120 vocab rows only.
2. gemv3.comp's WG() weight-chunk walker hardcoded 128MiB chunks
   (idx>>25) — the 994MB output.weight, uploaded as a single chunk, could
   only address rows <32,102; high vocab ids were unreachable.

Fix: dynamic chunk_words push constant. VkDecoder is now the default
vulkan path for infer/vl/bench/serve (VkAcc restorable via
LLM170_VK_ACC=1); serve --gpu-runtime vulkan is actually honored (was
silently ignored).

Verified on Vulkan (all 2026-09-07): np4 seq2/3 exact vs llama (seq0/1
diverge only at HIP's known near-tie points); MTP spec4 == nonspec exact
x4; long 2302tok exact; mmproj via HIP ViT + VkDecoder LLM coexisting in
one process (NYT front page read correctly); server multiturn
cached-continuation == CLI full-prefill exact (16/16). HIP regression
after all changes: verify.py 19/19 PASS.

### Vulkan performance round 1 (2026-09-07, plans/30)

gemv3.comp rewrite: 256 threads (was 64), runtime-subgroup-count tree
reduction, iq4_xs KVAL const-array → ktab SSBO LUT (const-array dynamic
index lowers to a select-chain — was the worst kernel), sub-block stride
generalized. Per-type GEMV bandwidth (vk-gemv-time, t=1): iq4_xs 19→57-73
GB/s, q8_0 60→96-107, head 86. A 4-row×64-lane variant was tested and
rejected (39KB LDS → occupancy collapse, documented in the shader header).

Engine: tg16 2.52→5.68 t/s (+125%), pp64 2.65→12.45 t/s (+369% with
LLM170_VKD_BATCH=1). tile128 (coopmat f16, q5_K t≥2) now opt-in
(LLM170_VK_TILE=1): its numeric family diverged from the t=1 gemv and
broke the spec==nonspec invariant (measured; NOTILE run restored exact
equality). VKD_BATCH root-caused: the corruption was the gemm_i8/quant_b8
route (q5_K unpacked i8 weights, t≥2) — LLM170_VK_NOI8=1 reproduced
correct ' Paris' immediately. i8 routing is now opt-in
(LLM170_VK_I8ON=1) and batch prefill is functionally correct, but
shows NO amortization gain (pp512 11.13 t/s ≈ per-token 12.45): the
gemv inner dot loop is ALU-bound per token — weights are read once per
chunk but the 32× dot work dominates. Prefill parity with llama-vk
(127+ t/s) requires per-type register-tile GEMMs (weight fragment →
all-tokens dot), the same structure as llama's mul_mat MMQ.

llama.cpp Vulkan reference on this machine (server bench, Q4_K_XL):
pp 127-431 t/s, tg 11.4-11.9. Our Vulkan standing after round 1: tg 0.48×,
pp 0.10× — NOT yet ahead; the remaining gap needs per-type tile GEMMs
(weight-amortized prefill) and further GEMV bandwidth (q3_K at 37 GB/s
is the weakest large kernel). Correctness held throughout: np4 seq2
exact + others at HIP's known tie points, spec==nonspec x4 exact (after
tile unification), long 2302tok exact, HIP regression 19/19 PASS.

### Vulkan performance round 2 — gemv4 family (2026-09-07, plans/31)

Ported llama.cpp's mul_mat_vec architecture as a new kernel family
(`gemv4_{q8,q5,xs}.comp`, opt-in via `LLM170_VK_GEMV4=1`, decode t=1 only):
32-thread wave32 workgroups, RPF rows per WG with the activation vector held
in registers across rows, f32 activations fed directly (no xq quantization
pass), vec4 activation loads (a second descriptor aliasing the same buffer,
llama's `data_b_v4` trick), and subgroupAdd reduction. Per-type bandwidth
(vk-gemv4-check, exact CPU-f32 match max|D|=0.0000): q8_0 160 GB/s (gemv3
107), iq4_xs 122 (57-73), q5_K 100-104 (55).

Engine integration through a `gemv_w` wrapper that lazily quantizes only on
the gemv3 fallback (dead-quant removal), dynamic rows-per-WG (1 for small
n_out to preserve parallelism), and per-type pipeline cache keys. Two
integration bugs worth recording: grid axes must be (1, rows/rpf, t) — the
shader reads .y as the row block; three spvs sharing one pipeline-cache name
caused descriptor mismatches (SIGSEGV). Net: decode step 154→146 ms
(tg 5.68→~5.9 t/s, +5%).

Negative results (all opt-in-preserved): q5_K gemv4 regresses in-engine
(step 178-194 vs 146-151 ms) despite winning standalone at 121 GB/s —
three suspects eliminated in sequence: register spill ([[unroll]] +
GL_EXT_control_flow_attributes, standalone 97→104 GB/s, engine unchanged),
scattered byte_at scale gathers (replaced by two direct word loads,
standalone →121 GB/s, engine unchanged), vec4-y numerics (a uvec4 nibble
extraction produced element scrambling — kept the verified scalar path).
The standalone-vs-engine gap remains unexplained; q5 stays on gemv3 by
default (`LLM170_G4_Q5=1` to test). y-staging in shared memory was
rejected after measuring llama's actual structure: it stages nothing, it
vectorizes activation loads and reuses them across rows. Batch prefill on
gemv4 re-reads the t×n_in activation per row-WG — 3× prefill regression;
decode-only routing required.

Correctness: France ' Paris' exact, spec==nonspec x4 True, long prompt
exact, np4 token streams identical to both gemv3 and commit 26f34a3
(pre-existing 2/24, 17/24, 24/24, 24/24 vs the llama store — store drift,
not a kernel effect). HIP regression 19/19 PASS (rawvk-only changes).

iq4_xs follow-up: staged the iq4nl LUT into shared memory at workgroup
start (llama's init_iq_shmem approach — 1KB, cooperative load + barrier).
Standalone 122→138 GB/s and this one DOES translate in-engine: decode
step 152.4→136.7 ms (-10%). End-to-end bench tg16 5.81 t/s (baseline
5.68, +2.3% — per-token sampling/detok overhead absorbs part of the
step win). Gates: France exact, spec==nonspec x4 True, long exact.

Per-dispatch-fence attribution (LLM170_VK_NOBATCH=1 + LLM170_G4_SYNC)
shows the gemv4 engine gap is NOT q5-specific: attention projections run
at standalone rate (attn_qkv 0.338 vs 0.296 ms) while ALL FFN sites run
+30-42% over standalone (ffn_down 0.407→0.580 ms) — L2 pollution from
the interleaved attention/conv kernels is the standing suspect.

Round 2b — chunk-walker arithmetic: replaced the per-word integer division
in WG() with shift/mask (chunks constrained to powers of two; the last
chunk is allocated at exact size — o = idx & mask always lands on real
data). Standalone: q8_0 160→247-257 GB/s (+55%), iq4_xs 138→150, q5_K
118→140, all exact. One integration bug caught by the gates: computing
the chunk log2 from the first buffer's ACTUAL size wraps single-chunk
non-pow2 tensors into a phantom chunk 1 (dummy buffer) corrupting weight
tails — spec==nonspec went False; fixed by using next_power_of_two.
Engine: decode step median 152→127 ms (-16.5%, 3-run), end-to-end
tg32 6.19-6.32 t/s (+9% over the 5.68 baseline). rpf sweep knob
(LLM170_G4_RPF) showed 2/4/8 within noise.

Round 3 — prefill coopmat tiles (plans/32). tile128 (q5_K) extended to a
prefill-only threshold (t≥16, keeping spec verify batches on the exact
gemv path) and joined by a new tile_xs (iq4_xs port of the same 512-thread
coopmat structure; A-staging dequantizes iq4_xs with the shared ktab).
Measured pp512: 11.18 (per-token gemv) → 17.45 (q5 tile) → 26.04 t/s
(both tiles, LLM170_VK_TILE=1 + VKD_BATCH=1). Cumulative +133%.

CRITICAL caveat — the tiles stage operands in f16 (RDNA coopmat has no
f32 inputs), which shifts the numeric family: with tiles on, the long
prompt diverges from the token-exact reference stream entirely (0/24 vs
the llama CPU store; llama's own Vulkan pp has the same f16 property).
The quality contract therefore keeps tiles OPT-IN
(LLM170_VK_TILE=1); the default path re-verified clean after the change:
spec==nonspec True, long 24/24. A plain LDS-y batched gemv (gemt_xs,
64 threads, 160 barriers) was also built during this round and REJECTED:
one-hot probes pass but random inputs corrupt outputs through an
unresolved shared-memory anomaly (sy rows read inconsistently across
tokens with identical inputs; markers prove staging threads alive and
stores happening) — kept as a check-harness prototype only
(vk-gemt-check), not wired into the engine. Standing: tg 0.54×,
pp 0.20× (default 0.10×). Next: full MMQ family + non-GEMM prefill
kernels, or an f16-tolerant quality contract.



## Vulkan — FIXED (2026-09-05)

Root cause of the full-model failures was never a driver leak: the sysfs GTT counters are
in **bytes** (the "18.6 GB stranded" was 18.6 MB). The real issue was memory-type
selection — VkCtx picked the first HOST_VISIBLE type (heap 0, GTT-aperture-limited), and
16.3 GB of weights cannot pin in a 15.5 GiB GTT aperture, so the first submit failed with
amdgpu vm_validate. Fix: prefer DEVICE_LOCAL|HOST_VISIBLE (RADV STRIX_HALO heap 1,
74 GiB carveout) for weights, cached GTT type for scratch, and unmap weight buffers
after upload. Full-model Vulkan now runs and the token stream matches the CPU reference
exactly (41/41 ★).

Feature coverage on Vulkan after the fix: np multi-sequence streams match the
np reference (s1 exact; s0 differs only at the known 18/19 near-tie), and MTP
speculative verify lines show live acceptance (OK/MISS) — MTP, np, and the plain
path all function on Vulkan.

Per note: after ~7 h of continuous compute on this 1.5-day-uptime host, the CPU-side
engine degraded ~180x (identical bench command: 9.33 t/s earlier in the session vs
0.05 t/s now; all 24 cores parked at 2.0 GHz, governor=powersave, 43°C — cause unclear,
no external load). Even with the host partially recovered (CPU probe 112s -> 8.4s), Vulkan holds at
1.85 t/s tg / 4 t/s pp32 — so the dominant cost is architectural, not host state:
the VkAcc design submits+waits a fence per matmul (~0.9 ms × ~325 matmuls/token
≈ 290 ms/token of sync overhead; single GEMV itself is healthy at 48.7 GB/s ★).
Command-buffer batching (plans/19) landed with **correct semantics**: per-op fresh
descriptor sets (recorded commands must not share a set — all would observe the last
binding) and write→read memory barriers between dispatches (submit+wait provided
implicit sync that a single command buffer does not). Stream matches the CPU reference
exactly (41/41 ★) and single-GEMV checks stay ★.

Performance is neutral (tg 1.88 t/s vs 1.85 unbatched): an earlier +11% reading was
taken on corrupted output (the first batch implementation lost the descriptor-set
hazard) and is retracted. On this host the barrier cost matches the fence savings;
the dominant costs remain host-side GDN/attention and per-op upload/readback, so the
path to Vulkan throughput is unchanged — GPU residency of the GDN family
(plans/19 phase 2). HIP remains the performance path at 28 t/s np4×spec4.



## Vulkan status (2026-09-05) — superseded

rawvk smoke suite passes (coopmat probe, axpy bit-exact, 25.5 GB/s) — the Vulkan compute
path is healthy. Full-model Vulkan runs are currently blocked by a kernel TTM leak:
18.6 GB of GTT stranded by crashed processes (zero holder PIDs), leaving 11 GB available
against the 16.3 GB weight pin requirement, so the first queue submit fails with
ErrorDeviceLost. The reference llama.cpp Vulkan build fails identically. Kernel log confirms:
`amdgpu_cs_ioctl: Not enough memory for command submission` — GTT total 16.6 GB
with 18.6 GB stranded (counter over total, zero holder processes; lost-device
buffer releases never ran). Full-model Vulkan ran fine earlier in this same
boot (tg 10.4, pp 128), so the leak is the sole blocker. A reboot clears the
stranded GTT; re-verification and the Vulkan MTP port follow.



## Vulkan functional audit — np/mtp/mmproj matrix (2026-09-07)

The VL (mmproj) gate had never run against the Vulkan decoder
(verify_vl.py leaves LLM170_GPU_RUNTIME unset → HIP default). Audited
explicitly (`LLM170_GPU_RUNTIME=vulkan`):

- HIP reference: 5/5 PASS (vl_spec_np2_seq0 adjudicated tie @gen[12],
  top-2 gap 0.629 < ε=1.0 — the documented 494/16311 flat point).
- Vulkan current: vl_spec_short / vl_spec_np2_seq1 / vl_spec_long PASS;
  vl_spec_np2_seq0 diverged @gen[12] (same 494-vs-16311 pair, no <ε gap
  evidence that run); vl_np2_isolation failed in the judge run but PASSED
  on direct re-run.
- Vulkan pre-session binary (26f34a3 worktree): vl_spec_np2_seq0 also
  FAILS (@gen[15]) — the flaky flat point predates this session's work;
  isolation passed there.

Conclusion: text np4/spec/long exact (established), VL+Vulkan functional
with one run-to-run flaky borderline at the known flat point — the same
class HIP adjudicates as tie. Divergence position and isolation verdict
move between runs and binaries (ViT embedding ulp sensitivity suspected).
Not a functional break; tracked as the known flat-point class.

### Prefill scaling attribution (round 3 addendum)

pp scales flat across chunk sizes (pp96 27.4, pp256 26.2, pp512 25.9 t/s
with tiles+batch): a fixed ~37 ms/token cost dominates — NOT the GEMM
tiles (which amortize: weight traffic is constant in t). The remaining
serialized per-token work lives in the non-GEMM prefill path (GDN conv
state scan, KV writes, attention history) — the next prefill lever is
attributing and batching those. An earlier in-session reading of
'3,295 t/s prefill' was a misread of a decode-step profile line at
pos=512; the bench numbers above are the authoritative prefill figures.

### Round 4 — full tile family + kernel-time attribution (2026-09-07)

Added a generic per-kernel timer (LLM170_VK_KTIME=1, best with
LLM170_VK_NOBATCH=1; per-weight keys for gemv). Attribution on a
128-token prefill (t=32 chunks): gemv 784ms dominated — the q8_0
(ssm_out) and q4_K (attention layers' FFN) tensors had no tiles.
Ported tile_q8 (34B q8_0 blocks) and tile_q4k (144B q4_K, the q5
variant minus the high-bit plane; shift/mask chunk walker). One
numbering trap recorded: q8_0 is ty=8 (a first attempt keyed ty=7
silently matched nothing).

Batch compute 1144→804ms; pp512 26.0→39.1 t/s (+50%; cumulative
+250% over the per-token 11.2 baseline). Tile-mode smoke: coherent
continuation on a 64-token prompt; default path untouched (tiles stay
opt-in behind LLM170_VK_TILE=1). HIP regression 19/19 PASS.

Standing mystery for the next round: bench wall time still runs ~4×
the summed kernel time (13.1s vs ~3.2s for pp512) — a per-token
~24ms non-kernel overhead somewhere between Engine::prefill and
step_batch. tiles: tile128 205ms, tile_q4k 75ms, tile_xs 75ms per
128 tokens; gemv residual ~300ms.

### Round 5 — full type coverage + attribution hygiene (2026-09-07)

Resolved the "4× overhead mystery": it was a division error on my side —
the ktime total is per-chunk wall time in synchronous mode and matches
the bench exactly (25.6 ≈ 25.1 ms/tok). No hidden overhead. Then closed
the type coverage: tile_q3k (110B blocks: 32B bit-packed hm, 64B 2-bit
q, 12B aux-unpacked scales, f16 d — unaligned word-span assembly), plus
tile128 ql/qh word hoisting (16→4 walker calls per thread-subblock,
200→150 ms per 32-token chunk, identical arithmetic). Verified with the
France prompt pushed through a 38-token batched prefill so ALL five
tile types engage: ' Paris' survives the f16 family.

pp512: 39.1 → 42.8 (hoisting) → 45.4 t/s (q3_K). Session cumulative
prefill: 11.2 → 45.4 (+306%), 0.36× of llama-vk. HIP 19/19. Remaining
prefill: gemv tail is now mostly q6_K (ty14) + iq3_s (ty21) stragglers;
the per-32tok chunk budget is tile128 150 / tile_xs 71 / tile_q4k 50 /
gemv ~250 / elementwise ~90 ms.

### Round 6 — q6_K tile (2026-09-07)

tile_q6k completes the type matrix (six quant types now tiled: q8_0,
q3_K, q4_K, q5_K, q6_K, iq4_xs). q6_K geometry: 210B blocks — ql nibbles
(lo/hi by quarter), qh 2-bit selectors (bits w*2..w*2+1 of the same
byte), direct i8 scales at 192, f16 d at 208 with word-span assembly.
France-through-batched-prefill verified ' Paris'; pp512 45.4 → 47.3
t/s. Session cumulative prefill: 11.2 → 47.3 (+323%, 0.37× llama-vk).
Remaining gemv tail: iq3_s (ty21) stragglers only. HIP 19/19.

### Decode round — two hypotheses rejected (2026-09-07)

Attacking the FFN-site gemv slowdown (+30-42% over standalone) with
llama/HIP-architecture analogies:

1. GTT dirty-line theory (LLM170_G4_YDEV=1): stage the activation vector
   into a device-memory buffer before each gemv4. A/B: 141.6 vs 142.1 ms
   — no effect. Rejected.
2. Tile-everything decode (LLM170_VK_TILE1=1, threshold 16→1): route the
   t=1 decode through the coopmat tiles like our HIP backend's WMMA
   path. A/B: 150.8 vs 144.0 ms — slightly worse; at t=1 the f16
   A-staging (full weight dequant per step) is not amortized, and our
   tile staging is not pipelined the way HIP's WMMA loop must be.
   Rejected for now; the HIP tg=10.4 analog remains unexplained.

Both knobs are kept as documented opt-in experiments. Decode standing:
~144 ms step (tg 6.2-6.3, 0.55× llama-vk). Prefill standing: 47.3 t/s
(0.37×).

### Round 8 — llama MMQ int-dot port attempt (2026-09-07)

Deep-dive into llama's mul_mmq revealed the actual decode architecture:
weights stay INTEGER and the dot runs on the hardware integer-dot
instruction (`dotPacked4x8EXT`, SPV_KHR_integer_dot_product — DP4A
class, one instruction per 4 elements) against q8-quantized activations.
That is the structural difference from both our paths (scalar f32 FMA
per element; f16-staged coopmat tiles).

Ported the structure as gemv5_xs (xq int activations, ktab-packed
int8x4 weight words, per-block scale epilogue). BLOCKED on toolchain:
shaderc 2023.8 (glslang 14) predates the GLSL extension; the Ubuntu
noble glslang-tools 15.1 package is missing the extension despite
upstream support (binary strings confirm); no spirv-as; no sudo; no
network fetch. Measured with an int unpack emulation instead:
149.1 vs 144.2 ms — no gain, confirming the win lives in the hardware
dot itself (emulation ≈ float-FMA op count).

Correctness: gemv5 produces ' Paris' (engine path LLM170_V5=1).
Unblock for a future round: any upstream glslang ≥13 binary (or
spirv-tools assembler), then swap the emulation block for
dotPacked4x8EXT — the surrounding kernel is done.

### Round 9 — OpSDot toolchain unblocked; int-dot measured (2026-09-07)

Unblocked the hardware integer dot without any system packages: built
scripts/spvtool (a 60-line C tool linking the installed
libSPIRV-Tools.a static archive — assembler/disassembler/validator via
ctypes-free CLI), then scripts/patch_sdot.py: gemv5's GLSL keeps a
canonical 'DOT patch marker' pattern (8 prepared sign-extended operands
+ 4 mul + 4 add), glslc -O0 emits it verbatim, and the disassembly
block is textually replaced with OpSDot + capability/extension. A
hand-assembled device probe (vk-sdot-probe) first proved the GPU
executes OpSDot correctly (1M chained signed dots, gpu==cpu bit-exact).

Measured: hw-dot gemv5 153.1 vs gemv4 147.7 ms — still no decode win.
Conclusion refined: for iq4_xs the bottleneck is the ktab LUT gather
unpacking PER ROW, not the multiply. llama's MMQ unpacks each weight
tile to int8 words in SHARED MEMORY once and reuses it across the whole
output tile — the exact structure a 'gemv6' (shared-int8 tile + int
dot, no coopmat, no f16) should copy. All groundwork for that kernel is
now in-repo and reproducible.

### gemv6 design note (2026-09-07, end of session)

Pre-implementation analysis killed the 'shared-int8 tile' idea for the
DECODE case: with t=1 the output dimension is rows, and each row's
weights are distinct — unpack-to-shared amortizes only over the token
dimension (which the coopmat tiles already exploit for prefill). So
llama's decode win is NOT shared staging; it is (a) typed packed16/32
buffer views replacing our manual word assembly, (b) their vecq LUT
path being structurally equal to ours, and (c) the q8-style types
having no LUT at all. Our per-type standalone rates bracket the
ceiling: q8_0 257 GB/s (no LUT) vs iq4_xs 150 (4 L1 gathers + byte
repack per word pair). Closing decode toward llama's 11.4 therefore
splits into: arithmetic (LUT-free) unpack for q5/q4/q3/q6 (feasible —
their dequant is shift/mask only), and the FFN-site in-engine mystery.

### Round 10 close — gemv5_q5 resolved; int-dot decode verdict (2026-09-07)

The gemv5_q5 'q-part≡0' bug resolved as a stale-binary artifact
(include_bytes! requires a cargo rebuild after .spv swaps — the engine
path was correct all along). Shipped state: gemv5 covers iq4_xs AND
q5_K with int-arithmetic unpack (LUT-free for q5), France ' Paris'
verified through both, standalone maxrel 0.09/0.26 (W4A8 quantization
class). Final A/B: gemv5 161.0 vs gemv4 142.4 ms — the int-dot decode
structure yields NO win on this stack: the hardware OpSDot executes
correctly in isolation (probe) but not inside the kernel context
(patched builds return 0 for the dot term; suspect driver codegen),
and the int emulation is op-count-neutral against float FMA. Verdict:
llama's decode advantage is not portable via the dot instruction alone
on RADV+this toolchain; the remaining decode gap (6.3 vs 11.4) is
attributed to per-type unpack costs and the FFN-site in-engine
mystery. All gemv5 paths stay opt-in (LLM170_V5=1).

### FFN-site mystery — refined (2026-09-07, session close)

Recomputing the per-dispatch-fence numbers as bandwidth: in-engine ALL
gemv sites run at a uniform ~103-107 GB/s (attn_qkv 36MB/0.338ms=107,
ffn_down 60MB/0.580ms=103, ffn_gate 47MB/0.449ms=105) while the same
kernels standalone reach 121-146. The loss is proportional to the
standalone rate, not site-specific — it is a system-level floor (DRAM
contention / dispatch environment), not an FFN-specific effect. The
engine floor (~105 GB/s over 16GB = ~150ms step) matches the measured
142-152ms steps. Next decode lever must therefore raise the engine
floor itself, not individual kernels.

### Engine-floor mystery SOLVED (2026-09-07, session close)

The synthesis: the 'standalone' kernel rates (121-268 GB/s) are
L2-ASSISTED — the timing loop re-reads the same 36-60MB tensor 10× and
a 32MB L2 retains much of it. The MULTI runs (interleaved different
tensors = L2 evicted, exactly like the engine) show the TRUE DRAM
streaming rate of our kernels: ~105 GB/s uniform. That is why slab
pooling did nothing (no allocation overhead involved) and why the
engine floor matches ~105 exactly. llama-vk's decode at an effective
182 GB/s therefore reflects genuinely better DRAM access patterns, not
magic: their mul_mat_vec maps lanes to CONSECUTIVE words inside one
row-block (K_PER_ITER 4-8 word bursts per lane), while our gemv4 maps
lanes to strided sub-blocks (176B apart for q5_K) — 4-byte
transactions scattered across 32 sectors per wavefront. The next
decode kernel work is a lane-remap to contiguous per-lane bursts
(llama's iqs layout); expected recovery toward 150-180 GB/s would take
tg from 6.3 to ~8.5-10. LLM170_MULTI and LLM170_L2FLUSH knobs are in
vk-gemv4-check for reproduction.

### Round 12 — llama iqs lane remap landed (gemv6_q5) (2026-09-07)

Built gemv6_q5 on the solved analysis: the wavefront sweeps each row's
ql words as contiguous 128B (2-pass: block headers staged to LDS, then
a linear per-lane word sweep with role dispatch). Standalone bit-checks
clean (max|D|=0.0000 vs CPU) and the L2-evicted (true-DRAM) rate
improves +10.5% (79.7 → 88.1 GB/s) — the first measured access-pattern
win on the floor. Engine A/B is neutral (142.9 vs 142.2 ms: q5 sites
are only ~30ms of the step, so +10% ≈ noise) and spec==nonspec goes
False under G6 (accumulation-order ulp differences cross ties vs the
gemv3 verify path) — the same class as the documented VL flat point.
Shipped opt-in (LLM170_G6=1) with the tie caveat; the layout now needs
to be replicated for iq4_xs/q8/q4_K/q3_K/q6_K where the FFN mass is.

### Round 13 — gemv6_xs; engine absorbs kernel gains (2026-09-07)

Replicated the llama lane layout to iq4_xs (gemv6_xs: 2-pass headers→LDS
plus a linear qs-word sweep; one LDS-overflow bug caught by ffn_down's
n_blk=68). Standalone exact, and the true-DRAM (L2-evicted) rate on
ffn_down: 93.0 → 106.6 GB/s (+14.6%). Engine A/B with gemv6 on both
q5+xs: 148.8 vs 145.1 ms — slightly worse, within noise.

Conclusion of the decode campaign: every kernel-level DRAM gain this
session (+10-15% per type, all verified at the kernel level) is
absorbed by the engine step — the step is dominated by per-dispatch/
serialization structure (~450 gemv dispatches/token across 64 layers),
not by any single kernel's bandwidth. The next decode lever is
therefore dispatch-count reduction (kernel fusion: per-layer
norm+gemv, or multi-tensor batched gemv dispatches), a different class
of work than kernel tuning.

### Correction — dispatch theory rejected by direct measurement (2026-09-07)

Added a t=1 ktime report (step-path hook): decode step wall 212.1ms vs
kernel sum 200.7ms in NOBATCH mode — kernels are ~95% of the step;
dispatch/launch overhead is NOT the bottleneck (the batched 145ms wall
further pipelines the small kernels). The Round-13 'dispatch-count'
conclusion is retracted. What the breakdown does show: gemv4_xs 22.8ms
(69 calls), rms 21.4ms (128 — inflated by per-dispatch fences in
NOBATCH), quant 11.4, gdn_ar 10.4, gemv4_q8 5.6, plus a long tail of
per-weight q5/q6/q3 gemv entries (the top-12 cut hides ~95ms). Decode
remains weight-traffic-bound: the next lever is raising the effective
per-type bandwidth of the remaining gemv tail (q5/q6/q3 via the gemv6
layout) and the rms/quant/elementwise minor kernels' real batched
costs (needs timestamp queries; per-dispatch fences distort them).

### gemv6_q6 — WIP handoff (2026-09-07, session close)

Ported the lane-remap layout to q6_K (output.weight, ~1GB/token — the
largest single decode gemv). Fixed en route: qh LDS index missing the
n*32 term, harness n_kb=10 (q6 gemv6 has no ktab/yv4), the debug-run
transposed grid (old known defect, now fixed for all routes), scale
index (mm&31)>>4. State: STILL systematically wrong from element 0
(one-hots mismatch everywhere, some NaN at higher elements) despite the
mapping re-derived twice against deq_q6_k — the defect is suspected in
the 210B-block word assembly or the staged-qh word rebuild, not the
element formulas (all verified against CPU source). Not wired into the
engine; vk-gemv4-check with LLM170_G6=1 + G4_HOT one-hots reproduces
in seconds.

- Update: found the primary gemv6_q6 defect — sh_dl was sized [8] per
  block but q6_K has SIXTEEN scales per block (two 8-scale halves);
  the n=1 paths read out of bounds. After fixing to [64][16] the
  error collapsed from ~1e37 to 1.94 — a residual mismatch remains
  (one-hots: some elements zero, some close-but-wrong), suspected in
  the staged d or the lo/hi nibble-to-element pairing corner cases.
  Next session: dump-based comparison of sh_dl[0][0] against a python
  GGUF read of output.weight bytes 192-210.

### gemv6_q6 SOLVED — family complete (2026-09-07, final)

The dump-driven debug chain closed it: Q6_DUMP ground truth (CPU
w.data read) + a skip-main-write debug variant (macro-gated
DBG_MAIN_WRITE) + a row0-scoped dump revealed GPU dl = d×(sc as
UNSIGNED byte) for exactly the negative-scale entries — the staging's
`(sc8 << 24u) >> 24u` used UINT shifts (logical, zero-extending);
sign extension requires casting to int first (int shifts are
arithmetic). One-line fix: max|D| 1.94 → 0.0000 EXACT.

gemv6 family now covers the three dominant decode types (q5_K,
iq4_xs, q6_K) — all CPU-exact, all showing the +10-15% true-DRAM
kernel-rate class, all engine-neutral (step 149.3 vs 142.0 ms; the
engine absorption pattern holds for the third time). The lane-remap
campaign is complete: the remaining decode gap to llama is NOT in
per-kernel access patterns of these types — next candidates are the
q3/q4/iq3_s tail, the elementwise minors (rms/quant/axpy ~30ms/step
combined), and timestamp-level attribution of batched execution.

### gemv6 5-type engine verdict (2026-09-07, session close)

Full-family A/B (all five types routed, two runs): g4 142.0/143.9 vs
g6 149.3/140.9 ms — statistically neutral within the session-long
140-152ms noise band. Final decode campaign verdict: the llama lane
layout delivers real, verified per-kernel true-DRAM gains (+10-15% on
every type) but the engine step remains dominated by aggregate weight
traffic at the ~105-110GB/s system floor. With q3/q4/q5/xs/q6 all
lane-remapped and exact, per-kernel access patterns are no longer the
differentiator against llama-vk's 182GB/s effective — the remaining
gap lives in per-WG row coverage (llama packs NUM_ROWS×NUM_COLS with
subgroup-quad reductions), the iq3_s/q8 stragglers, and the
elementwise minors. All gemv6 paths remain opt-in (LLM170_G6=1).

### gemv7 (y-LDS staging) — negative (2026-09-07, session close)

Tested the y-amplification theory (gemv6 reads y once per row: 160KB
y vs 28KB weights per WG — 5.7× y redundancy). gemv7 stages each
block's 256 y floats to LDS once and dots all 8 rows against it.
True-DRAM (L2-evicted) result: 70.9 vs gemv6's 91.6 GB/s — WORSE. The
costs that outweighed the y savings: per-row header re-reads (4 words
× 8 rows × 20 blocks — header traffic is now per (row, block) instead
of staged once), 40 barriers per WG (vs 16), and the serialized row
dot inside each lane. Verdict: gemv6's structure is the local optimum
for this shape; the llama gap is not y-handling. Kept as opt-in
(LLM170_G7=1) with the harness route.

### gemv6_q8 — rejected (2026-09-07, session close)

The byte-per-lane mapping (32 lanes = 32 qs bytes) is exact
(max|D|=0.0000) but collapses to 69-73 GB/s vs the existing gemv4_q8's
257: each byte read becomes a separate WG() walker call with 4 lanes
redundantly fetching the same word. q8_0's existing scalar-vectorized
path (word-per-lane over 8-word strips) is already at the ceiling —
q8 was never a straggler in the ktime data. Kept as a documented
opt-in; engine q8 routing unchanged.

### llama mmv config decoded (2026-09-07, session close)

Read llama's actual mul_mat_vec launch configuration from
ggml-vulkan.cpp: on AMD RDNA3 the K-quant decode (q3/q4/q5/q6) uses
NUM_ROWS=2 with FORCED SUBGROUP SIZE 16 (wg_size_subgroup16 pipelines,
use_subgroups16), not our rpf=8/wave32. The spec-constant row
multipliers (rm_kq=2 non-GCN AMD, 4 on GCN; rm_iq=4) and the
subgroup-16 forcing are the two launch-geometry levers we have not
tried — both are concrete, cheap experiments for the next session
(VkCtx pipeline creation with subgroupSizeFull/16 via
VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE or
requiredSubgroupSize=16, plus rpf=2). This is the remaining
untested launch-geometry gap against llama's 182GB/s effective.

### sg16 + rpf=2 (llama RDNA3 launch geometry) — rejected (2026-09-07, close)

Implemented ctx.pipeline16 (requiredSubgroupSize=16 via the subgroup
size control pnext, the exact llama wg_size_subgroup16 mechanism) and
ran gemv6_q5 with rpf=2 — llama's RDNA3 K-quant decode configuration.
Warm rate 150.2 GB/s but true-DRAM (L2-evicted): 75.0 vs our
wave32/rpf8's 102.9 — WORSE. llama's launch geometry does not
transfer to our kernel shape; our 32-lane WG with rpf=8 remains the
best measured configuration. This closes the launch-geometry lever
list. The remaining decode gap against llama's 182GB/s effective is
now attributable to their shader-level differences beyond geometry
(typed packed16/32 buffer views, vec4 B loads with K_PER_ITER=4/8
unrolling — a full llama mul_mat_vec shader port, not a config change).

### gemv8_q5 — llama mul_mat_vec FULL PORT: first engine win (2026-09-07)

Ported llama's mul_mat_vec_q5_k verbatim (typed u16 buffer views of
the same weight chunks — Vulkan allows the same buffer bound with
different GLSL block types; SIMD-in-register nibble handling via
0x0F0F0F0F masks + unpack8; fma chains; 16-threads-per-block ×
2-blocks geometry). Two port bugs found by the harness: scales must
load as SINGLE u16s (llama scales[v_im] is one element, not a pair),
and the engine's cw2 chunk constant must use pow2ceil of the first
buffer (engine first-chunk is actual size, harness uploads the pow2
cap).

Results: standalone 162.4 GB/s warm (fastest q5 ever, exact 0.0000);
ENGINE step 142→128-130ms (-10ms, first kernel win that survives the
engine); end-to-end tg32 6.60 t/s (from 6.19-6.32). Quality: France
' Paris' exact, long prompt EXACT; spec==nonspec goes False — the
fma/subgroupAdd accumulation order differs from gemv4/gemv3 by ulps,
flipping ties in the t=1-vs-t≥2 comparison (same documented class as
gemv6's caveat). Shipped opt-in (LLM170_G8=1) pending either a
matching-order variant for verify batches or accepting the tie class.

### gemv8 family extension — xs mapping note (2026-09-07, next session)

The gemv8_q5 recipe (typed views + SIMD-in-register + fma) is the
proven engine-winning template. For iq4_xs the llama generic
mul_mat_vec mapping needs care: iqs = col/2 with iq = 16*ib32 +
(iqs%16) and qshift = (iqs&16)>>2 — dequantize4 returns FOUR
consecutive same-column elements from ONE word's nibbles, i.e. the
lane covers 4 elements at col = ib32*32 + (iqs%16)*... plus the
column split at iqs 16. Derive the exact col↔iqs table against our
VERIFIED gemv4_xs mapping (byte b of sub → elements b lo, 16+b hi)
before writing gemv8_xs; the same derivation then applies to
q3/q4/q6 via their llama dequantize4 blocks in dequant_funcs.glsl
(DATA_A_Q4_K etc.). y loads via the existing yv4 alias binding.

### gemv8_q6 — exact but slower (2026-09-07, close)

Ported the gemv8 recipe to q6_K (one OOB bug caught by the harness:
scale base is 8n per half, not 16n — sh_dl[16] indexed to 23). Exact
(0.0000) but 67.6 GB/s vs gemv6_q6's ~90 — the 210-byte block
misalignment forces per-word remainder assembly in the linear sweep,
which the SIMD gains don't offset. q6 stays on gemv6 in the engine;
gemv8_q6 kept harness-only.

### gemv8 numeric-family UNIFICATION — spec invariant restored (2026-09-07)

Routed the verify batches (t<16) through gemv8 as well — gemv_w's G8
gate widened from t=1 to t<16 (prefill tiles still own t>=16) and
gemv_stage jobs dispatch via gemv8 when active. Result: the t=1 decode
and the spec verify batches now share ONE accumulation order —
spec==nonspec x4 back to True (measured), France exact, HIP 19/19.
gemv8 is now a complete-quality path: exact kernels, faster engine
(step -11.4%, tg32 6.70), invariant-safe. Still opt-in
(LLM170_G8=1) pending the long-form gate and np4 tie checks on the
unified family.



## Vulkan ms-geometry tile family + q3_K decode fixes (2026-09-10, plans/40)

Reference re-measured on this machine (llama-bench d222767c, Vulkan):
pp512 356.66 / pp64 244.08 / tg8 12.12.

Findings: the previous tile family was pinned at ~39 GB/s A-stream not by
coopmat structure but by two codegen hazards — an 8-way if-chain weight
indirection (WG()) inside the staging loop and per-MMA uniform guards.
llama's own mul_mm shader run in our harness on the same geometry reaches
67-79 GB/s; removing the indirection (single-buffer direct index; every
tensor fits one chunk since chunk size is per-tensor next-pow2) and the
guards (zero-padded B, drain clips) recovers most of it: 39 -> 64 GB/s
at the llama m-warptile geometry (BM64/BN64/BLOCK128, 2 wave64 warps).

Rolled the ms skeleton out to all quant types (q4k/q6k/q3k/q8/xs/nl via
shared ld16 unaligned-merge), 64-row workgroups, guard-free MMA.

Accuracy: while bisecting the rollout, the harness uncovered THREE latent
q3_K tile decode defects that predate this round (scale tmp assembled
from 3 bytes instead of 4 — byte 107 missing; block-half index taken
from the global 128-element half instead of the within-block half;
shift/hbit derived from (sb>>1)&3 instead of sb&3). Fixed against the
ggml reference; verified elementwise via a dump probe (0/5120) and
tile_check (maxrel 0.0033). The engine q3 path is permanently moved to
the fixed kernel. xs/nl additionally needed the shared ktab LUT bound
(omitted in the new dispatch produced garbage tokens).

verify (judge, LLM170_TILE_MSALL=1): 22 PASS / 3 FAIL — prior accepted
state was 4-8 FAILs; single_ko and np4_seq1 pass for the first time.
Remaining 3 are spec-equality borderline (first mismatch @gen 10-21).

Memory: raw_init cloned the whole model to_vec() (17.5 GB anonymous —
the two-process host-RAM OOM that repeatedly killed sessions). Weights
now stay mmap borrows through upload; after init the file pages are
madvise(DONTNEED)-returned (page-aligned ranges; tensor offsets are only
32B aligned). Steady state: anon 0.7 GB, 14.65 GB returned. Two
concurrent full-model processes complete — previously fatal.

Benchmarks (MSALL, q35work): pp64 168.8-180.2 (was 140), pp512 162.6-167.6
(was 125), tg 7.36-7.59 (decode path untouched). vs llama Vulkan: pp64
0.69-0.74x, pp512 0.46x, tg 0.61-0.63x.

Open levers: tile codegen gap to llama's own shader (64 vs 79 GB/s,
same geometry — vectorized staging loads); pp512 N-amortization (weight
re-read per 64-token slab; BN>=256 attempts regressed so far); decode
timeline (132 ms wall vs llama 89 ms; gemv8 163 vs ~196 GB/s effective).



## Vulkan decode-kernel llama ports (2026-09-10, plans/40 cont.)

Decode GEMVs ported from llama mul_mat_vec_{q5_k,q4_k,q6_k} to the
gemv8 family: 64-thread WGs (16-thread block groups), 2 rows per WG,
vec4 SIMD-in-register unpack, u16 typed views on the single weight
chunk, subgroupAdd reduce. q5 143->225, q4 152->272, q6 ~143->194 GB/s
(all max|D|=0.0000). q6 uses llama's double-buffered shared scale
cache; q4's y loads sit at {y1, y1+32, y1+128, y1+160} with
per-component smin; q5's interleave is {0,1,16,17,32,33,48,49}.

gemv8_q8 added for q8_0 (34B blocks) — moves alpha/beta/v stragglers
off the legacy gemv path. mmv-check probe runs llama's prebuilt dmmv
spvs in our harness for reference (beware: repeat-loop timing on
<32MB tensors measures the L2, not DRAM).

Benchmarks: tg8 7.36 -> 8.63 (+17%), tg32 7.59 -> 8.21. pp unchanged.
verify: 22 PASS / 3 FAIL — identical set before/after the port.

vs llama Vulkan: pp64 0.72x, pp512 0.46x, tg8 0.71x.

Follow-up: gemv8_q8b (llama generic dmmv structure, 87 -> 329 GB/s)
and gemv8_xsb (same structure with our verified iq4_xs decode,
125 -> 182 GB/s), both max|D|=0.0000.

Benchmarks after the full decode arc: tg8/tg32 7.36 -> 9.33 t/s
(+27%; llama Vulkan 12.12 — 0.77x). pp unchanged. verify: 22 PASS /
3 FAIL, identical set before/after all four ports.

Open: tile staging vectorization (64 -> 79 GB/s ceiling); pp512
N-amortization (gate+up merged dispatch); gemv8_q3 remains on the old
kernel (3 tensors).



## Speculative decode (MTP) — inert on the Vulkan raw path

`--spec k` is opt-in (bench/serve/infer only set `mtp_wanted` when a spec
budget is requested), so there is no default-path regression. On the raw
Vulkan path, however, the feature does not do useful work:

* `mtp_draft_logits` is only populated by the non-raw `verify_batch` path;
  the raw path stores `mtp_draft_tok` (the GPU MTP head's argmax) instead.
  The CPU spec loop therefore drafts greedy(empty logits) = 0 every step and
  every draft misses (bench: "fwd 8, gen 8, 1.00 tok/fwd", draft=0 in
  LLM170_SPEC_DBG).
* Enabling the MTP hook costs ~150 ms per prompt token in prefill (pp64
  with --spec 2: 9.9 s vs 0.33 s without), because the MTP layer chain runs
  per token.
* The batched GPU verification path (`LLM170_SPEC_GPU=1`) is slower still:
  1.50 t/s on tg8 (9 fwds for 8 tokens, 0.89 tok/fwd).

The single-sequence CPU spec loop is structurally unable to beat plain
decode (each candidate token costs a full target decode), so the viable
design is the batched verify path; until that is rebuilt, MTP is not a
throughput lever.



## Speculative decoding on Vulkan — cost breakdown (session 2026-09-11)

Measured with `LLM170_SPEC_TIMING=1 LLM170_SPEC_GPU=1 LLM170_VKD_BATCH=1` on tg spec=3
(q35work, pp64): step total 1280.9 ms for acc=2 tokens. Components:

| Stage | Cost | Note |
|---|---|---|
| draft chain (k=3) | 351 ms (117 ms/draft) | MTP layer GEMVs at solo rate (~30 GB/s) |
| state snapshot | 620 ms | rollback save of GDN state (~150 MB) — dominant |
| verify `step_batch` t=4 | 296 ms (74 ms/token) | **no weight amortisation at small t** |
| advance | 4 ms | |

Plain decode reference: ~62 ms/token, so the spec path was 10-20x slower than plain.

### Root finding: weight traffic is not shared across the batch dimension at small t

- gemv8 (t < 16) dispatches one workgroup per (row-pair, token): the same weight rows are
  re-read by every token slab, and the z-ordered launch gives no temporal locality for L2 to
  merge. Traffic scales ~linearly with t.
- Forcing the tile path with `LLM170_VK_TILE1=1` does **not** change this: verify t=4 stays at
  293 ms (73 ms/token), i.e. slower than 4 independent t=1 decodes.
- At t >= 16 the tile kernels *do* amortise (pp64: 624 ms / 64 = 9.8 ms/token, 6x better than
  t=1), because a workgroup keeps the weight tile in LDS while iterating over tokens.

Consequence: speculative verification with small batches (k+1 tokens) cannot win on this
backend regardless of the draft cost or snapshot cost. A batched verify only becomes
profitable if the verification batch is large (>= 16) or if the small-t GEMV path is
restructured so that one workgroup covers several tokens (i.e. move the token loop inside
the workgroup, LDS-resident weights).

### Secondary findings (kept, correctness-neutral)

- The prefill MTP KV hook used the **CPU** `mtp_step` (~150 ms/token), which was the cause of
  the "spec enabled makes prefill 30x slower" observation (pp64: 207 -> 6.5 t/s). It now uses
  `mtp_step_gpu` (argmax only, `mtp_draft_tok`); the draft fallback in `spec.rs` reads that
  field when `mtp_draft_logits` is empty. Spec output remains token-identical to the plain
  path (verified: 84 220 201 198 201 198 201 for both).
- `LLM170_VKD_BATCH=1` (batched verify) is token-identical to the per-token verify (same
  gemv8 kernels, t < 16).



## Vulkan tile WM 64->32 falsified on this backend (2026-09-12)

plans/62's V1 -- llama PR #28611, where shrinking the coopmat warp micro-tile M dimension
64->32 gave +61% pp on RDNA3 iGPUs -- was implemented for our tile family as
`tile_ms4gy_wm32`: BM 32, the A loader remapped to four threads per row (`kc = (tid&3)*8`
with the c loop cut to 0..2 so each thread still covers its own 8 k values and the row's
k coverage stays complete), `acc[2][2]`, `Ctmp[32][17]`, and the host grid at
`(n_out+31)/32`. Measured with `vk-tile-check <gguf> blk.0.attn_gate.weight <t>` plus
`LLM170_TILE_BENCH`:

| variant | t=512 | t=128 |
|---|---|---|
| `tile_ms4gy` (WM 64) | 0.2806 ms/iter (77.1 GB/s) | 0.2538 ms (85.2 GB/s) |
| `tile_ms4gy_wm32` (WM 32) | 0.4883 ms (44.3 GB/s) | 0.4486 ms (48.2 GB/s) |

**-43% at both token sizes**; the variant was withdrawn (shader, SPIR-V and host wiring
all removed). This is consistent with our own earlier characterization of this tile kernel
(every resource below 15% utilisation): the bottleneck is not the per-workgroup LDS or
register footprint, so halving the warp micro-tile only halves the work per workgroup
without buying occupancy. plans/62 pre-registered exactly this falsification condition
("if it contradicts our measurement, add it to the rejected list"), and it did.

Consequence for the Vulkan plan: V1 is closed; the remaining candidates are the int8
coopmat1 path (V2), the FA K/V shared-memory staging (V3) and the attention redesign port
(V4).



## Vulkan plan V2/V3 assessed; the FA's real fix located (2026-09-12)

**V2 (int8 coopmat1, llama PR #27952) - premise weakened.** The in-tree prototype was
measured before investing in the rewrite: `LLM170_VK_I8ON=1` at pp512 gives **32.76 t/s**
against the default's 323.06, i.e. **10x slower**, and the kernel is already documented as
superseded by the coopmat tiles ("kept for the integer-MMA contract"). The PR's approach is
a different implementation, but with the tiles already at 0.91x of llama-Vulkan and the
prototype off by an order of magnitude, a 12-quant-type shader rewrite has no measured
premise. Closed unless a concrete need for the integer path appears.

**V3 (FA K/V shared staging) is only half the story.** Our records already measured K/V
prefetch as neutral for a shuffle-bound attention, which is what the Vulkan FA is - but
reading `qsa_flash.comp` shows why staging alone cannot help and what the real fix is.
Per key the kernel currently does: one FMA, **five subgroup shuffle stages**, two
**block-wide barriers**, and a serial 8-way sum executed by a **single thread**
(`if (tid == 0)`). At np=3314 that is 6628 block barriers per (row, head) block plus a
1-thread dependency chain per key.

The HIP arc's answer ports directly and needs both halves together: **lane = key with the
whole hd dot computed in one thread** (HIP's `qsa_flash_gqa2`, which was 1.5-2.4x before
`v_dot2` was even involved) **plus the K staged in shared** - on Vulkan the lane-per-key
dot reads K with a 256-way stride straight from global, so staging is what makes the
structure viable there, whereas on RDNA3/HIP the `v_dot2_f32_f16` instruction (no portable
SPIR-V equivalent) was the second half. Estimated payoff from the FA rewrite is ~1-3% pp
and <1% tg at Vulkan's current numbers, for a ~2-3 hour shader+host rewrite - recorded so
the decision is explicit rather than implied.


