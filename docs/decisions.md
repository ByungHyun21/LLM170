# Decision Records (ADR)

Not reverse-ordered. Update on change, with dates.

## ADR-0001 — Pure-Rust single language (2026-08-30)

**Decision**: the entire engine is pure Rust. No C/C++ kernel sources, no RTC
string kernels, no CMake toolchain.
**Background**: user policy. Solo development, so dual language/toolchain costs
outweigh the benefits.
**Rejected**: the 2026-08-30 research recommendation (Rust core + CUDA-dialect
C++ kernels + NVRTC/hipRTC) —
[source/research/2026-08-30-rust-gpu-bindings.md]. The factual material stands;
only the recommendation was rejected.
**Implication**: the GPU kernel path is wgpu (WGSL) or rust-gpu (.rs→SPIR-V)
(decided at ADR-0004 time). Native CUDA waits for cuda-oxide maturity.

## ADR-0002 — Mode system universal / cmp-stock / cmp-unlocked (2026-08-30) — SUPERSEDED (2026-09-13)

**Superseded by device measurement.** The flag set `LLM170_W_CAP_GB` (which
had no reader) and `LLM170_Q4_CHUNK` (whose 1024/512 split the decode frame
already capped), so it changed no route. Removed; the runtime now measures the
device at accelerator init (name, free/total memory, host↔device bandwidth)
and gates the WMMA attention path on a probe rather than a profile.

**Decision**: three modes. Runtime flags + kernel variants + memory profiles.
The core is mode-agnostic.
**Background**: CMP stock (eFUSE throttle, 8 GB) and unlocked (full rate,
40/64 GB) demand different kernel strategies; the universal profile doubles as
the portability/verification infrastructure.

## ADR-0003 — Unlock as first-class scenario (2026-08-30)

**Decision**: the 40/64 GB unlock is a design precondition. `cmp-stock` (8 GB)
operation is still maintained.
**Background**: user confirmation (40/64 GB available). cmpunlocker (2026)
precedent. Unofficial, hence the fallback.
**Note**: until compute-unlock measurements exist, `cmp-unlocked` kernels also
default to half2.

## ADR-0004 — CPU reference backend first (2026-08-30)

**Decision**: complete the CPU (pure-Rust) reference before the GPU backend.
The golden standard for all tests.
**Background**: verifiable without a GPU, isolates numerical correctness,
defers the kernel-language decision.
**Superseded in part (2026-08-31)**: development order is now GPU-first per user
directive; the CPU reference keeps its golden-test role.

## ADR-0005 — strict FP + no `mul_add` (2026-08-30)

**Decision**: no `f32::mul_add` in hot paths, no fast-math flags. Codegen FMA
absence verified via profiler/CI.
**Background**: Rust strict FP has no implicit contraction, so mul+add
separation is automatic → the cmp-stock 32× penalty is avoided for free.

## ADR-0006 — Documentation layout source/ · docs/ (2026-08-30)

**Decision**: `docs/` (project docs, tracked) / `source/` (external reference
originals, tracked) / `plans/` (gitignored). docs removed from `.gitignore`.
**Background**: separates research originals from project output and keeps
documentation version-controlled.

## ADR-0007 — Implementation order qwen35 → qwen4exp (2026-08-30)

**Decision**: complete the full path (parser → CPU inference → profiler) with
dense qwen35 before extending to qwen4exp.
**Background**: GDN/IMROPE/MTP substructures are shared; qwen4exp-specific
parts (HC/QSA/PLE/MoE) build on top.

## ADR-0008 — Universal development targets ROCm · code-transfer flow (2026-08-30)

**Decision**: `universal`-mode GPU development and verification happen on ROCm
(this machine, gfx1151). Once verified, the user transfers the code to the
170HX machine and continues there. GPU kernel technology (cubecl-hip etc.) was
decided after CPU completion, but with a HIP/ROCm-compatible API surface first.
**Background**: user directive. The dev machine is a standard ROCm
environment. A Backend trait accommodating both HIP/CUDA-like APIs minimizes
transfer cost.
**Requirement**: debug builds must let the developer see everything —
instrumentation (built-in profiler) + structure dumps by default.
**Status (2026-08-30)**: workspace + GGUF v3 parser + `gguf-dump` + profiler v0
done. 6/6 tests (4 synthetic + 2 measured). Research unknowns resolved by
measurement — [models/qwen4exp.md](models/qwen4exp.md).

## ADR-0009 — GPU backend: cubecl + HIP (2026-08-30)

**Decision**: GPU kernels are written in Rust via cubecl macros;
hanzo-cubecl-hip (a fork repointing the sys bindings to a local ROCm build)
JIT-compiles for gfx1151. The same kernels compile to CUDA/PTX (sm_80) for the
CMP 170HX.
**Background**: user directive — GPU execution first, not CPU-verified-first.
cubecl keeps kernel sources Rust, preserving the pure-Rust policy.
**Implication**: `crates/backend-gpu`. Kernel progression: f32 GEMV (verified)
→ quantized GEMV (q4_K/q5_K/q6_K/q8_0) → GDN AR/attention → prefill.

## ADR-0010 — iq3_s block layout correction (2026-08-30)

**Decision**: the iq3_s struct is `d(2) qs(64) qh(8) signs(32) scales(4)` —
**d at the front** (per ggml-common.h).
**Background**: the previous implementation read d from the back (offset 108),
causing an NaN explosion on L14 ffn_down (the only iq3_s tensor). Found by
diffing against the authoritative gguf-py implementation.
**Lesson**: always confirm block layouts against the ggml-common.h struct
declarations — never trust research summaries.

## ADR-0011 — GPU matmul offload via `Accelerator` + dual HIP/Vulkan runtime (2026-08-30)

**Decision**: `crates/core` defines a runtime-injectable `Accelerator` trait
(`matmul`/`matmul_batch`); `crates/backend-gpu` implements it with cubecl
quantized-GEMM kernels. `llm170 infer --backend gpu [--gpu-runtime hip|vulkan]`
selects the backend at runtime (no build features). The same kernel source
compiles under hanzo-cubecl-hip (gfx1151) and cubecl-wgpu (Vulkan/WGSL), and
both runtimes reproduce the CPU greedy stream token-for-token.
**Background**: user directive — GPU-first development, and both ROCm and
Vulkan must work on the dev machine.
**Constraints found**: WGSL has no u8 tensor element, so quant bytes are
transported as u32 words and unpacked in-kernel. The HIP dialect miscompiles
`as i8` casts and value-yielding `if` expressions on the RHS of binary
operators — kernels use pure u32 arithmetic instead. Per-matmul host round
trips dominate decode latency in this first cut; pipelining/keeping
activations resident is follow-up work.
**Numerics**: GPU accumulation order mirrors the CPU reference (row-sequential
blocks, element-sequential within a block); measured max relative error vs CPU
is < 1e-3 across all eight quant types in the 27B Q4_K_XL mix.

## ADR-0012 — Verification standard: near-tie-aware token parity (2026-08-31)

**Decision**: `verify.py` passes a case when the greedy stream matches exactly,
or the first divergence is a near tie (our token within the baseline top-6 and
the top-1 logprob gap < ε = 1.5 nats). The reference server runs on GPU
(`-ngl all`, f32 KV); the CPU reference llama build (int8 activation dots) is
a different numerical identity and is not used as the reference.
**Background**: our f32 activation dots vs llama's q8-quantized activation
dots + f16 KV cache reorder flat-top distributions. Measured: both engines
share the same top-3 set at divergence points (baseline gap 1.19 / ours 0.25).
llama.cpp itself differs between its own CPU and GPU backends from token 0 —
"bitwise identical to a specific llama build" is excluded from the goals; the
golden standard is our f32 engine (which the GPU reproduces token-for-token).

## ADR-0013 — W4A8 integer-dot variant (2026-08-31)

**Decision**: a performance-path variant alongside the f32 reference:
CPU `quantize_row_q8_ref` (f32 scales — a deliberate departure from ggml's f16
storage) + per-type integer dot kernels; GPU `gemm_q6` with host-side q8
pre-quantization (qs as u32 words + per-block d) shrinking activation
transport 4×. Cross-checked against the f32 reference at rel 1–5e-3 (the
theoretical q8 activation-quantization noise).
**Implication**: the CMP dp4a-class integer-accumulation version remains a
cmp-stock follow-up once the hardware arrives.

## ADR-0014 — GPU buffer arena: allocation is permanent (2026-09-01)

**Context**: Long-prompt prefill produced NaN outputs, then
`Memory page 0 doesn't exist` faults, on both HIP and Vulkan with identical
failure points. Root cause (measured, 2026-09-01): `create_from_slice`
handles dropped at scope exit are reclaimed by the cubecl memory manager's
delayed dealloc; queued kernels still referencing that memory read garbage.
A second defect compounded it: `dev_weight` returned a shared 4-byte dummy
handle for over-budget weights, and `matmul_group`/`matmul_paired` lacked a
host-fallback check, launching GEMMs against the dummy.

**Decision**: GPU memory has a single owner, the buffer arena
(`backend-gpu/src/buffers.rs`): `WeightStore` returns a `WRef` enum where
`gpu()` errors on host fallback (the dummy-handle class of bug becomes
unrepresentable), and `ScratchPool` retains every transient upload
(x, q, ck, cv, mask) plus scratch permanently — no buffer is ever freed.
Fresh allocations drain the queue first (`client.sync()`), the client runs
`MemoryAllocationMode::Persistent`, and `main` re-execs itself so
`HIP_LAUNCH_BLOCKING=1` applies before HIP initialization (setting it after
init is a no-op — measured).

**Consequences**: VRAM is bounded by pool accounting (`POOL_TOTAL`) plus the
weight budget (`LLM170_W_CAP_GB`) instead of by frees; on a 96 GiB dev
machine the long-prompt working set measured 45.7 GiB. Verification:
2,311- and 1,904-token prefills reproduce token-exact 24/24 on Vulkan.
Known open issue: the HIP runtime still faults on multi-chunk (>=2,048-token)
prefill through the cubecl-HIP memory manager — the engine code is identical
across runtimes, so this is tracked as a runtime-layer defect (dev-machine
long verification runs on Vulkan). **Update (2026-09-01)**: root cause found
and patched locally — ADR-0016.

## ADR-0015 — Engine stage modules over a shared context (2026-09-01)

**Context**: `qwen4exp/layers.rs` had grown to 1,283 lines holding every
stage (hyper-connection mix, GDN, QSA, MoE, PLE) inside one `Engine4` impl,
with no injection point for CMP kernel variants and per-call allocation
churn (`Hparams4::clone` allocating five vectors per stage call).

**Decision**: stages live in `qwen4exp/stages/{hc,gdn,qsa,moe,ple}.rs` as
free functions taking `Ctx { model: &Model4, acc: Option<&dyn Accelerator> }`
plus, for stateful stages, `&mut SeqState4`. Dispatch helpers
(mm/mm_batch/mm_group/mm_paired) moved onto `Ctx`; the matrix-dispatch
variants (grouped same-input projections, paired per-expert rows) are part
of the context contract. `Engine4` keeps forward/prefill/decode and timing
only.

**Consequences**: `layers.rs` is 335 lines; stages are backend-independent
and independently testable. Numerics unchanged (pure code motion):
GPU↔CPU 25/25 token-exact, 1,024-token chunk smoke clean, single_ko
reference-exact 24/24, cargo suite 10/10.

## ADR-0016 — Local cubecl-runtime vendor patch: memory sweep disabled (2026-09-01)

> **SUPERSEDED by ADR-0018 (2026-09-05)** — cubecl was removed from the
> dependency graph; there is no `vendor/` tree and no `[patch]` section any
> more. Kept for the RCA record only.

**Context**: even after ADR-0014 removed every free path, intermittent
`Memory page N doesn't exist` faults and libamdhip64 access violations at
fixed instruction pointers kept firing on allocation-heavy runs. Root cause
(measured): cubecl-runtime's exclusive-pool `cleanup` drains its free pages,
deallocates them, and rewrites the surviving page indices (`update_page`).
A single dealloc shifts every later page, invalidating (a) descriptors still
held by live handles, (b) bindings of in-flight launches — which kills the
HIP runtime. The runtime's own error flush then calls `cleanup` again,
compounding the damage.

**Decision**: a vendored copy of cubecl-runtime under `vendor/` wired in via
`[patch.crates-io]` turns `MemoryManagement::cleanup` into a no-op. The
engine never frees GPU memory (ADR-0014), so the sweep can only lose.

**Consequences**: page indices stay stable for the process lifetime; the
fault class disappeared. The patch is temporary — remove it once upstream
fixes the page reindexing. No engine code depends on it.

## ADR-0017 — GPU-resident decode frame, default on for qwen4exp (2026-09-02)

**Context**: per-token decode spent ~600 host round-trips per step (per-op
readback, buffer acquire, launch marshalling). rocprof showed the GEMV
kernels already streaming at llama-level bandwidth (~141 GB/s) while
~280 ms/token was host glue.

**Decision**: `Frame4` (`core/src/qwen4exp/frame.rs`) keeps activations
device-resident for a whole decode step and chains kernels by handle —
hc, GDN AR, MoE, and the output head run framed; PLE stays a host-hash
bridge and QSA a value bridge (~14 syncs/step total). Per-sequence handle
sets support parallel decode. `main` defaults `LLM170_FRAME=1`
(`LLM170_FRAME=0` disables); with framing active the mode W_CAP preset is
skipped so the weight store can place the full expert stacks (~88 GiB) on
large carve-outs; on smaller regions the first frame error permanently
falls back to the value path (one-shot retry, notice on stderr).

**Verification**: synthetic tiny4 e2e GPU == CPU, hash-identical to the
non-frame path; real model tg16 0.43 -> 2.87 t/s (6.7x) — see
[benchmarks.md](benchmarks.md). qwen35 has no frame yet; the same pattern
extends (all its layer kernels already exist on GPU).

> **Update (2026-09-18)**: `frame.rs` is now the `frame/` module tree
> (`mod/forward/np/multi/diag`, plans/78 R9 + plans/79 A).

## ADR-0018 — cubecl removed; raw HIP + raw Vulkan backends (2026-09-05)

**Context**: cubecl's HIP runtime wedged on faults and its WGSL path could
not express the quantized kernels (no u8 buffers, no dp4a). Every production
kernel had been hand-ported twice already (HIP C++ and GLSL).

**Decision**: drop cubecl entirely. `rawhip` drives HIP via hip-sys with
HIP C++ kernel strings (hipRTC JIT + optional offline code objects);
`rawvk` drives Vulkan via ash with precompiled SPIR-V. Both mirror the CPU
W4A8 integer arithmetic (see backend-architecture.md).

**Consequence**: one kernel source per backend, no IR layer; ADR-0009's
"cubecl keeps kernel sources Rust" is superseded — kernel bodies are now
HIP C++ / GLSL embedded as strings, with Rust owning all orchestration.
ADRs 0009/0011/0016 remain as history.

## ADR-0019 — Experiment-scaffold pruning policy (2026-09-08)

**Context**: the perf campaign (gemv4→8 generations, launch-geometry sweeps,
HIP A/B batches) left ~75 `LLM170_*` gates, five copy-pasted q5_K GEMV
shader generations, and 16 zero-launch HIP kernels. An engine-level A/B
showed the promoted default already beat the surviving opt-in generation
(q6_K via gemv3+quant 7.06 t/s vs gemv6 6.71 t/s median tg32).

**Decision**: once an experiment concludes, the losing path, its env gates,
its check tools, and its shader assets are deleted in the same change — no
opt-in graveyards. Verification infrastructure is exempt (`LLM170_EXACT`
bit-exact reference kernels, per-tensor check tools, the Mesa OpSDot
reproducers). Dead-kernel deletion requires a launch-site cross-check
(kernel name × launch strings), not grep of definitions alone: probes and
env-gated variants kept several "dead-looking" kernels alive.

**Consequence** (updated 2026-09-09): rawvk GEMV routing is a type-driven
table — gemv8 (t<16, q3/q4/q5/q6/xs, kill-switch `LLM170_G8=0`), coopmat
tiles as the default t≥16 prefill path (kill-switches `LLM170_VK_NOTILE=1`,
`LLM170_VKD_BATCH=0`), quant+gemv3 (q8_0/iq4_nl/iq3_s at t<16), i8 GEMM
(experimental `LLM170_VK_I8ON`, superseded by tiles — kept for the
integer-MMA contract). Model knowledge (raw weight/const manifests,
rope table) lives in core; backend injection lives in backend-gpu; the
server is a CLI router + HTTP. The hipRTC kernel string is split into
family assets (`kernels/src_*.hip`) assembled by `include_str!` — the
concatenation is byte-identical to the monolith it replaced (hash-checked
during the split).

## ADR-0020 — HIP attention defaults after the 2026-09 attention arc (2026-09-12)

**Context**: the base cells sat at 0.89-0.98× of llama-bench ROCm. ktrace and
probe measurements localised the residual to the attention on both sides: the
prefill kernel was shuffle-bound (5 shuffle stages per (key, head)), and the
decode kernel split hd across 256 threads, so every dot product needed a
cross-warp reduction. Separately, the MTP cell measured 0.5× the base because
the batched GPU verify was reachable only behind an env gate that nothing set,
so the default ran k+1 sequential single-token decodes per speculative step.

**Decision**: four HIP defaults, each with a kill-switch:
`qsa_flash_wmma` for hd=256 prefill — an fp16 WMMA tile kernel that consumes Q
into registers and reuses its 32 KB shared buffer as the K/V tile
(`LLM170_NO_WK_WMMA=1` → `qsa_flash_wk8`); `qsa_flash_gqa2d` for t=1 decode —
one key per lane, the whole 256-dim dot in a single thread via
`v_dot2_f32_f16`, no cross-lane reduction, reading an f16 KV mirror maintained
at the KV write sites (`LLM170_NO_GQA2D=1` → the f32 `qsa_flash_gqa2`); split
segment default 1024 instead of 128 (`LLM170_QSA_SEG`); and the batched GPU
spec verify as the spec default (`LLM170_NO_SPEC_GPU=1` → the sequential
path). The attention kernels accumulate in fp16 storage but f32 accumulators,
a numerics class accepted for the prefill in the 2026-09-12 decision recorded
in `benchmarks.md`, now extended to the decode path under the same evidence
bar.

**Evidence**: `wmma-attn-check` (0 mismatches vs a CPU reference, 6e-4),
`attn-check` (max|delta| 1.1e-5, 0 outliers over 50.3M elements), `gqa-bench`
(0 mismatches at 5.05e-4 across four kernel variants and four context
lengths), token-identical greedy output on a 300-token prompt for every
change, spec==nonspec identical at 21 and 2302 tokens, and a full gate run at
17/19 PASS + 2 INFO - unchanged from the pre-change baseline.

**Consequence**: pp512 364 t/s (1.03× of llama-bench ROCm), pp3314 339 t/s
(1.01×), tg512 11.6 t/s (1.01×), tg3314 11.3 t/s (0.98×); MTP 22.1 t/s
single-stream (1.92×) and 30.6 t/s aggregate at np4 (1.97×) — those are the figures at the time of
this decision. After the subsequent KV and Vulkan work the same cells read MTP **23.5 t/s (2.04×)**
and np4 **31.0 t/s (2.00×)**, Vulkan tg32 **11.44 t/s (1.00× of llama.cpp Vulkan)**, and a 13k-context
np4+MTP acceptance passed with spec == nonspec exactly. The decode
attention is now bandwidth-bound (~221 GB/s effective), so the next lever
there is KV quantization, not kernel structure. The f16 KV mirror also halves
the KV footprint per sequence, which is the configuration the RAM/SSD
offloading work will build on.

> **Update (2026-09-18, plans/79 C)**: the attention-chain kill-switches
> (`NO_WK_WMMA`, `NO_WMMA2/V2`, `WK8D`, `NO_WK8I`, `NO_WK8`) were pruned —
> the defaults (wmma2v2 for np>2560, else wk8i) are unconditional, and
> `LLM170_NO_GQA2D` survives only as the f32-KV dump diagnostic.
> `qsa_flash_wmma` itself stays an opt-in experiment (demoted in plans/69
> after the 9.5× regression on this device).

## 2026-09-19 — Vulkan prefill GQA multi-query flash (plans/83 D)

`qsa_flash` (prefill) ran one workgroup per (row, head), scanning the whole
KV prefix serially — attention latency grew O(N^2) per chunk and dominated
long prefills (pp4096: chunk walls 1.6s→5.8s, 134-145 t/s vs llama 318).

`qsa_flash_gq` gives one workgroup 4 rows x 6 GQA sibling heads (24 queries)
sharing each staged K/V tile: attention traffic and latency drop ~24x.
Verified against a pure-Python reference and the old kernel on synthetic
data *and* on engine-dumped activations (layers 0 and 1, full gate-prompt
shape): all three agree to 5e-6. Permanent A/B probe: `llm170 vk-flash-check`.

pp4096 134 -> 232 t/s (+73%; llama-vulkan parity 318/232 = 0.73). Remaining
attention cost is warp0's serial per-query softmax chains; the obvious next
steps (persistent sp, coopMat QK) need an LDS-budget redesign.

**Near-tie baseline note**: the arithmetic-order change flips near-tie tokens
on the adversarial Korean gate prompt (the documented chunk-size residual
class — docs/chunk-invariance.md). The new stream is reference-correct;
HIP is untouched and still matches the original baseline. Gate baselines are
now per-runtime (`.gate-27b-baseline-vk.txt`); `LLM170_VK_NOGQ=1` restores
the old single-query kernel.

## 2026-09-19 — plans/83 D2 disposition: "FN vulkan decode" cell is the HIP path

`q4_gpu_wanted_str` accepts `--gpu-runtime vulkan` for qwen4exp only by
falling back to HIP with a warning (plans/64 §7: QSA kernels and capacity).
The README/plan-83 "FN vulkan" numbers were therefore HIP measurements. A
true Vulkan qwen4exp decode path does not exist; reaching the plan's
18.49 -> >=23 t/s target means either porting the frame path to rawvk
coopMat or a HIP decode kernel campaign. KTRACE on a decode step shows the
dense-GEMV family (`gemm_q8_0`, incl. gy=10240 calls) dominating the step,
with the t>=16 tile path already bypassing it — the t=1 GEMV and the MoE
grouped GEMMs are the campaign targets. Recorded here so the next plan
starts from the correct premise.

## 2026-09-19 — plans/83 D5: 27B np4 vulkan tg root cause

np4 aggregate 10.3 t/s vs hip 32.1: the VkDecoder runs the np step as
n_seq sequential single-slot passes (each ~89ms = the single-decode cost),
so the aggregate is single-stream rate with no weight-pass sharing. The
default-trait path also transfers full logits per slot; the new
`raw_step_multi_greedy` override recovers tokens via the GPU argmax
(no 608KB d2h per slot) — kept, but it is not the bottleneck. Reaching
hip-level np needs a true multi-sequence batched `step_batch` (per-row KV
append to different seq buffers + per-row attention over per-seq KV),
a dedicated kernel-path campaign.

## 2026-09-19 — plans/83 E disposition: WMMA spill target is not in the production path

`qsa_flash_wmma` (the plans/69 6-9.5x spill kernel) is invoked only from
`rawhip::probes::attn`; the production long-context (np>2560) prefill
attention is `qsa_flash_wmma2v2` (plans/74 N4, 64-thread, built from the
wmma2 probe ABI work). Current pp16384 hip = 290-296 t/s = 97-99% of
llama.cpp's 297; the plan's >=310 target exceeds llama itself. A pp16k
KTRACE window attributes the residual cost to the mmq GEMM family
(mmq_q5k/xs/q4k/q6k ~75% of kernel time), not WMMA attention. Further
pp16k gains live in the GEMM campaign, not fragment liveness.

## 2026-09-19 (2) — plans/83 D2 decode-step subtraction profile

Stage-skip subtraction on FN hip tg32 @4k ctx (32 steps, wall 1841.6ms =
57.5ms/step): GDN 682.5ms (21.3ms/step, 37%), QSA attention 310ms (9.7,
17%), MoE 192.6ms (6.0, 10%); remainder ~20ms/step (hc, PLE, norms, head,
host). The 2026-09-16 warp-variant experiment re-measured: Q8W_ALL=1 is
-4.4% end-to-end (16.64 vs 17.41 t/s) — the opt-in gate is correct.
Reaching 23 t/s needs ~14ms/step off GDN+attention+remainder jointly; no
single-kernel smoking gun. Next campaign entry points: the four GDN GEMV
shapes (qkv/gate/beta/alpha at t=1) and the ~20ms non-attributed residual.

## 2026-09-19 (3) — plans/83 D2/E closing evidence

Ground-truth re-measurement (same host, same protocol pp32/tg32, source/
llama.cpp 8b4b3558f): llama **vulkan tg32 = 23.22±0.09 t/s** (coopmat),
llama hip ≈ 17.4 (README table) — i.e. on HIP, llama matches our step time
(57ms) exactly; the 23-target is reachable only through a Vulkan qwen4exp
path with coopMat tiles, which does not exist in this tree (rawvk has zero
Engine4 references). Host-skeleton for our decode step: 2.2ms (LLM170_NOLAUNCH,
459 t/s) — the step is GPU-launch/kernel bound, not host bound. ctx512 vs
ctx4096 decode rates are identical (17.5) — KV traffic is not a factor.
Graph replay (LLM170_GRAPH=1) lands +1% (17.59).

E: the named spill kernel is not in the production path (probe-only); our
pp16384 hip 290-296 already exceeds llama hip (229) by 27%. Both items'
numeric targets are re-scoped as campaign goals in the follow-up list.

## 2026-09-19 (4) — plans/83 E closed: production WMMA A/B

Direct A/B at pp16384 hip (threshold override via temporary env, reverted):
`qsa_flash_wmma2v2` active = **290.91 t/s**; forced wk8i fallback (scalar/
dp4a family) = **242.05 t/s**. The production WMMA path is 20% faster than
its fallback — no spill regression exists in the shipping chain. The plan's
named kernel (qsa_flash_wmma v1) is probe-only and its plans/69 spill was
superseded by the wmma2/wmma2v2 line (plans/74). pp16384 290-296 stands at
127% of llama hip (229); the 310 stretch exceeds llama and belongs to the
GEMM campaign (mmq family), not attention.

## 2026-09-19 (5) — plans/83 D2: bandwidth anomaly pinned as the real target

With LLM170_GRAPH=1 + KTRACE the decode step's kernel-execution sum is
~56ms (gaps ~9ms under instrumentation) — the step is kernel-execution
bound. Necessary activated bytes/step (10/512 experts + dense trunk + QSA
+ hc + head) compute to ~5GB, i.e. ~20ms at 240GB/s — the step runs ~3x
above the weight-read roofline. llama-vk's 23.22 t/s implies ~10-13GB/step
at achievable APU bandwidth, also above the naive roofline but 25% under
us. Prime suspect: memory placement — our 58GB VRAM / 45GB GTT split makes
GTT-resident layers run at GTT read throughput, while RADV (uma:1) may
serve the whole model from one pool. Next campaign entry: per-layer step
time vs weight residency (VRAM vs GTT), then placement/tiling to close the
3x anomaly. This supersedes the "coopmat tiles" framing: the gap is
bandwidth placement, not matrix-core throughput.

## 2026-09-19 (6) — plans/83 D2: placement hypothesis falsified; occupancy starvation pinned

New probe `llm170 bw-place` (streaming 2.1GiB grid-stride read, memset-
committed, new `bw_stream` kernel — the old `bw_probe` is a sparse pattern
emulator touching ~320B/row and only measures cache): hipMalloc (VRAM
spare) 244 GB/s, hipMallocHost (GTT pinned) 239 GB/s, hipMalloc after
VRAM saturation 245 GB/s. **All memory placements stream at DDR5 peak —
there is no GTT penalty**, and ROCm silently spills hipMalloc past VRAM at
full speed. Combined with earlier findings (host skeleton 2.2ms/step,
kernel-exec sum = wall): the decode step's low effective bandwidth lives
in the kernels' shape occupancy — ~500 dense/hc/GDN GEMV launches per step
at gy=320-10240 blocks x 64 threads underoccupy the 64-CU GPU (20-40K
threads resident). Campaign direction: fuse the per-layer small GEMVs
(hc/GDN projections) into fewer wider launches. The probe stays as the
placement/BW ground-truth tool.

## 2026-09-19 (7) — plans/83 D2: stream-parallel shortcut blocked, fusion campaign scoped

The last cheap avenue — overlapping the independent latency-bound GEMVs on
side streams (bit-identical, no op changes) — is blocked by the shared MMQ
y-buffer hazard (docs/benchmarks.md known limitations: concurrent MMQ
paths clobber each other's y). The complete falsification chain for the
decode gap now reads: coopmat throughput (no vk path; hip step-time parity
with llama), memory placement (bw-place: 240 GB/s all pools), host cost
(2.2ms/step skeleton), launch gaps (kernel sum = wall; graphs +1%), KV
traffic (ctx-invariant). What remains: ~500 small-output GEMV launches per
step run occupancy-starved and latency-bound. The campaign = per-layer
projection fusion with scratch partitioning. All measurement tools are in
tree (bw-place, stage-skip subtraction, NOLAUNCH skeleton, KTRACE).

## 2026-09-19 (8) — plans/83 D2: pair-fusion tier complete

Post-fusion KTRACE census (1320 launches traced): the q8_0 dual fusion is
live (gemm_q8_0_dual gy=12800). The remaining small-kernel sea is the hc
machinery's DEPENDENT elementwise chain — rms_part x97, hc_gate_mean x97,
hc_combine x96, quant_q8 x96 — each stage feeding the next within a layer.
Pair-fusion cannot touch these (no shared-input independence); the next
tier is stage-cascade fusion (one kernel per hc norm→mean→combine chain,
replicating exact arithmetic order for bit-identity), a dedicated campaign
with the fp/gate verification ladder already in tree.

## 2026-09-19 (9) — plans/83 D2: session close-out

Graph replay re-verified working under KTRACE (launch suppression during
replay confirmed; end-to-end +1% stands). The cascade-fusion tier carries
a measured-negative precedent (plans/73 rms_small: 16.78→16.28 t/s — the
320-element serial f32 chain lost to the 2-launch parallel), so tier-2
entry must start from co-design, not naive fusion. Complete avenue ledger
for this session: 7 falsifications, 1 landed win (pair fusion +1.8%,
bit-identical), campaign spec + tools in tree. The >=23 target transfers
to the occupancy campaign with all evidence attached.

## 2026-09-19 (10) — plans/83 D2: pair tier maxed at 18.01 t/s

Corrected attribution: the earlier +1.8% was PLE-only — the GDN group
[q8 qkv, q8 gate, f32 beta, f32 alpha] is mixed-family and bypassed the
same-family branch where the dual lived. Wiring pair fusion into the
per-weight (mixed) loop landed the intended qkv+gate fusion: FN tg32
17.73 → 18.01 t/s (session total 17.41 → 18.01, +3.5%, bit-identical,
three gates PASS, pp512 unaffected at 253 t/s). The census's gy=12800
dual = QSA q+k (12288+512). Remaining pair-tier residue (~24 launches:
QSA v, indexer bf16 pairs) is worth <1%; the tier is closed. Everything
beyond this is the hc dependent-chain cascade campaign.

## 2026-09-19 (11) — plans/83 D2: SiluDivQuant slice spec (next-session ready)

The remaining cheap slice, fully designed: fuse the hc `SiluDiv+quant`
pair (96 launches/step) into one `silu_quant_q8` kernel. lo is
single-consumer (up-GEMM only — verified), q4_silu_div math is
`v=x/div; v/(1+exp_cr(-v))` (note: custom exp_cr), and quant_q8's
32-block amax/round can run on the silu'd register values — bit-identical
by construction. CLEAN wiring (avoid the stateful fxq-scratch shortcut
evaluated and rejected this session): (a) new trait method
`frame_gemm_xq(xq_handle, w, out, t)` with default Err + Q4Acc impl
calling launch_gemm; (b) a dedicated Frame4 buffer for the quantized
activation (ensured at frame_begin, t=1 sized n_in/4+n_in/32 words);
(c) FrameOp::SiluDivQuant writing that buffer; hc_mix_frame branches to
the pair at t==1 with LLM170_NO_SILUQ kill switch. Expected +0.5-1%
(18.10 → ~18.2). The identical pattern applies to HcGateMean+group-quant
(mix needs BOTH f32 and xq outputs — dual-write variant, ~60 launches).

## 2026-09-19 (12) — plans/83 D2: SiluDivQuant implemented, measured zero, reverted

Implemented the decisions.md-(11) slice end-to-end (silu_quant_q8 kernel
with exact exp_cr silu + quant_q8 arithmetic, FrameOp variant, one-shot
pointer-keyed quant memo, hc t=1 wiring): bit-identical (FN gate PASS
after fixing the memo to key on the resolved pointer). Measured: tg32
1768.2 ms — IDENTICAL to the pre-fusion wall time. The 96 quant
micro-launches were fully hidden in the pipeline; eliminating them gains
nothing. This closes the micro-launch-elimination hypothesis: the decode
step's remaining cost is GEMV execution occupancy, full stop. Reverted
per the no-weightless-complexity rule; the negative result stands as the
campaign's cleanest falsification yet.

## 2026-09-19 (13) — plans/83 D2: low-output warp GEMV, opt-in

LLM170_Q8W_SMALLN=1: warp-per-output for t=1 q8_0 GEMVs with n_out<=2048,
n_sub>32 (the occupancy-starved shapes). FN tg32 18.10 -> 18.25 (+0.8%),
FN gate PASS. Not default: the shared dispatcher has no model scoping and
the reduction-order change flips a 27B near-tie (token 5, measured).
Default adoption requires splitting the dispatch by model/frame path —
noted as a small follow-up for the campaign.

## 2026-09-19 (14) — plans/83 D2: warp-variant gain is marginal; session values

Variance measurement (3x tg32, no env): 18.18/18.12/18.10 — run noise
±0.08. The Q8W_SMALLN "+0.8%" (18.25) is ~2x noise, real but marginal;
combined with the 27B near-tie break it stays opt-in. Final session
ledger for D2: five landed bit-stable increments (17.41 → 18.10 default,
18.25 opt-in, +4-4.8%), twelve falsifications, all tools in tree. The
23 t/s target requires the mega-kernel occupancy redesign.

## 2026-09-20 (1) — qwen35 GPU reset-state and chunk-invariance fixes (plans/84 A)

Two correctness defects fixed on the qwen35 HIP path, both fenced by
`llm170 diag chunk-check`:

1. **Reset-state leak.** `Engine::reset_states` replaced only the CPU
   `SeqState`s; the GPU-resident GDN S-state and conv ring (raw decoder)
   stayed dirty, so the second conversation on a reused slot prefilled
   from the previous conversation's state (identical-prefill repeats
   diverged max|d| ~ 14, deterministic). `reset_states` now calls
   `raw_reset` for every slot and invalidates `frame_clean`, matching
   what `reset_seq` already did per-slot.
2. **Chunked-prefill non-invariance.** With the leak fixed, the checker
   showed *any* multi-call GPU prefill diverging from the single-call
   reference (max|d| 0.25-1.2 with argmax flips; the CPU path was
   invariant). Stage-level bitwise dumps localized the cause to
   kernel-family dispatch keyed on the row count t: g4 (t=2-4), tile
   _mm (<32) vs _wm (>=32) vs j128 (>64), MMQ (>=32), q8_0 GEMV (<=64),
   flash single-pass (np<=128) vs split, serial vs side-stream gate, mt
   GEMV variants (t=2-8), and t=1 prefill calls routed through the decode
   path. Each family is individually deterministic and row-invariant
   (probed bit-exact across t), but families disagree with each other by
   ulps, and the difference amplifies chaotically across layers. Fix: a
   prefill family pin — `step_batch` sets `DecodeState::pin_prefill` (+
   a module-level `PREFILL_PIN` for ctx-level gates) for the duration of
   the call, forcing the large-t family at every dispatch point; decode,
   np and spec paths keep their existing dispatch. Single-token prefill
   calls now use the batch path too.

Verification: 208-token prompt at chunk sizes 4/8/16/63/128/512 and
three identical 512 repeats all bits-identical; 9-token prompt at sizes
1-7 bits-identical; fresh-slot variant clean; gate-27b stream unchanged;
mmq-row-check / tile-row-check added as kernel-level fences.

## 2026-09-20 (2) — qsa_flash_gq warp-per-query softmax; 27B vk pp16k +24% (plans/84 D+E3)

The prefill GQA flash kernel staged QK partials in shared memory
(sp[8][33]) and ran the whole softmax chain (cross-warp sum, max, exp,
running m/s update) serially in warp 0 for each of the 24 queries per
workgroup. Redesign: warp-per-query rolling — each lane owns one key and
reduces the full head dimension serially; the eight warps process eight
different queries concurrently and complete softmax, including the
running max/sum bookkeeping, with in-warp shuffles. The sp array and the
serial chain are gone; the LDS budget is unchanged (still ~61KB).

Reduction order changes: vk-flash-check PASS (max|D| 6.1e-6 vs the f32
reference, same class as the previous kernel) and the 27B vulkan gate
stream is unchanged against the recorded baseline.

27B vulkan prefill: pp4096 233 -> 255 t/s (+9.5%), pp8192 174 -> 202
(+16%), pp16384 116 -> 143 (+24%). The fitted attention-quadratic term
drops 30% (0.357 -> 0.251 us/token^2); the linear term is unchanged.
The plans/84 E3 investigation (pp16k slowdown) is explained by the
quadratic attention cost of the serial-chain kernel: time fits
T = 2.8-2.9 ms/token linear + 0.26-0.36 us/token^2 quadratic with no
other superlinear component — not a leak or scheduling defect.

## 2026-09-20 (3) — FN t=1 decode occupancy analysis; graph replay null result (plans/84 C)

KTRACE breakdown of the Flash-Next frame decode step (53.2ms kernels +
11.3ms gaps, ~830 launches): gemm_q8_0_dual 9.3ms, MoE expert GEMMs
(ge_ids/w_ids) 8.3ms, hc mix_dual 4.8ms (108 launches), gemm_q8_0
4.2ms, output head 3.5ms, w16 up-matvecs 2.5ms (97), activation quants
2.0ms (304 launches), f32 duals 2.0ms, PLE/shexp 2.2ms.

Two findings bound the next step:
1. **hipGraph replay is numerically identical but gives zero speedup**
   (LLM170_GRAPH=1: gate stream unchanged, tg128 18.83 -> 18.84 t/s).
   The 11.3ms "gaps" are therefore host-side dispatch work between the
   capture segments (MoE routing round-trips, per-op frame dispatch),
   not launch latency — kernel-count fusion alone cannot reclaim them.
2. The small-kernel tail is real but bounded: hc mix_dual moves ~330MB
   per step (1.3ms at streaming rate) but costs 4.8ms; the w16
   up-matvecs cost 2.5ms for ~320MB and the 304 activation quants 2.0ms
   for negligible bytes — ~6ms of small-grid tail latency in total.

Design for the plans/84 C stair (layer-wise hc/GDN projection fusion):
fold the per-site sequence rms(norm) -> quant -> dual(down+inject) ->
silu -> up -> gate+mean (5 launches) into two kernels — (A) fused
rms+quant+down+inject with silu at store, (B) up+gate+stream-mean
reading the inv_rms scalars A stashed — cutting ~190 launches and the
xn/lo round-trips per step, worth an estimated 10-12% tg at the
measured tail cost. Arithmetic-order changes are expected (gate
re-baseline under the near-tie standard). The 2026-09-13 rms_small
fusion regression precedent (16.78 -> 16.28 t/s) applies: measure
per-kernel, not just end-to-end, before adopting.

## 2026-09-20 (4) — hc_mix fused kernel attempt: reverted on an unresolved device fault (plans/84 C)

The designed 2-kernel hc_mix fusion (rms+down+inject+silu | up+gate+mean,
t=1) was implemented and wired behind a capability method with automatic
fallback. The kernel reproducibly hard-faults (HSA memory fault ~2.7GB
from the frame buffers, attributed to q4_hc_a) and the implementation was
reverted; both gates pass on the reverted tree.

Bisect ladder (all configurations rebuilt via hipRTC each time):
- Fully empty kernel: clean launch, no fault.
- rms (thread-0 serial or barrier loop) + writer that stores xn WITHOUT
  the rms scale: no fault. Same config reading the rms scale into the
  store: fault.
- Full kernel (rms+writer+down+inject): fault.
- Down branch with activations hard-zeroed — i.e. only weight reads
  (f16w + q8 word loads), dot4 chain, tree64 reduce live: still faults.
  Inject branch disabled: still faults.
- Every array bound was audited repeatedly; the dot4/f16w/tree64
  sequences are textually identical to the production gemm_mix_dual q8
  side, and the 2-6-byte tail overread of the last q8 block is the same
  pattern every existing GEMV uses.

What is ruled out: launch geometry/args (empty kernel runs), barrier
divergence (uniformized), double-precision math (removed), dynamic
indexing of the scale array (constant-select tried), helper-function
indirection (inlined), use of shared vs register scale, and value-range
effects of the activations (inputs hard-zeroed and it still faults).
Next session should build a minimal standalone repro (probe kernel with
a 64-thread block doing q8 word loads + __ockl_sdot4 + tree64 after a
barrier) to decide between a code-generation fault on gfx1151 and
something in the launch path for this kernel shape. The fusion design
and its ~2-4ms/step ceiling stand (entry 2026-09-20 (3)).

## 2026-09-20 (5) — hc_mix fusion: fault root-caused (arg order); fusion loses to tuned plates (plans/84 C, final)

The mystery fault from entry (4) is solved: the kernel argument list was
pushed as (..., eps, n, hc, r) against a signature of (..., n, hc, r,
eps) — the kernel received n=0/hc=2560 and wrote invr[2559] off a
register array, wild-addressing ~2.7GB out. The `llm170 hca-repro` probe
(now a permanent asset, standalone synthetic buffers, 8 clean runs)
isolated this in seconds where engine bisects took a minute each; it
reproduced the fault bit-for-bit and validated the fix.

With the arg order fixed the 2-kernel fusion ran fault-free and its
greedy stream was IDENTICAL to the gate baseline (the arithmetic
reproduction of rms/quant/dot chains is bit-exact). But performance:
fused tg128 14.24 t/s vs 19.37 baseline (-27%) — every down block
re-quantized the whole activation row (320x duplication). A 3-kernel
restructure (quant once into a global xq buffer, down reads it) reached
17.82 t/s (-8%) with a remaining numerics bug in the xq consumption,
still below baseline.

Conclusion recorded: launch-count fusion alone cannot win here — the
production mix_dual/w16 plates are individually tuned and the ~5 saved
launches per hc site (~2ms/step ceiling) are smaller than the throughput
loss of replacement kernels. A winning C needs equal-or-better kernels
(e.g. reuse the w16 plate for `up`, fuse only rms+quant which is pure
launch savings), which is tuning work, not arithmetic work. Fused
implementation reverted; `hca-repro` stays as the diagnostic that closed
the question. Both gates pass on the reverted tree.

## 2026-09-20 (6) — Vulkan q5_1 GEMV: Flash-Next MoE-down mass enabled (plans/84 B, slice 1)

The Vulkan value-path GEMV (gemv3 uber-shader) had no q5_1 branch —
25.2GiB of Flash-Next MoE expert-down weights (docs/models/qwen4exp.md
type mix: MoE gate/up q4_K covered, down q5_1 not). Added ty=7 to gemv3
with the dot_q5_1_q8 mirror: 6-word block [d|m f16 pair][qh u32][qs x4],
low/high nibble words for elements 0-15/16-31, and the 5th-bit gather
expanding four consecutive qh bits into byte lanes (the q5_K-style
0x01010101 stride mask is for its interleaved layout and was wrong here —
first cut MISMATCHed, fixed by the expansion).

`vk-gemv-check` now auto-detects the qwen4exp architecture (arch string)
and loads via Model4, so the FN multi-part model's tensors can be probed
directly: blk.0/blk.3 ffn_down_exps (q5_1, 629MB stacked-expert tensors)
both PASS (argmax preserved, maxrel ~5e-3 — same tolerance class as the
existing types on this probe). The 27B vulkan gate is unchanged after
the shader recompile. Tile/coopmat q5_1 (prefill plates) remains open —
the GEMV path serves all t meanwhile.

## 2026-09-20 (7) — qwen4exp runs on Vulkan: value-path wiring + vk baseline (plans/84 B, slice 2)

`--gpu-runtime vulkan` now selects a real Vulkan path for Flash-Next:
`new_q4_acc_vk()` attaches VkAcc (the rawvk value-path accelerator) to
Engine4. VkAcc implements the MatmulHost composite only — FrameState/
FrameHost are empty — so the engine runs the CPU stage graph with every
GEMV staged through Vulkan (quant on device, gemv3 per weight, download).
All FN weight types are now covered (q5_1 was the gap, entry (6)).

Verification:
- 23-token smoke: vk stream identical to the HIP frame path (9 tokens).
- Chunk invariance: LLM170_Q4_CHUNK=16 == 512 exactly on vk — the value
  path chains PLE/GDN/KV state correctly across chunk boundaries (the
  HIP frame path fails this today, plans/84 E.2).
- Per-type fences: vk-gemv-check PASS on the FN stacked-expert tensors.
- 208-token gate stream diverges from HIP frame at token 4 (reduction-
  order drift of the lane-strided gemv3 accumulation, ~5e-3 rel per dot
  vs CPU — same class as every vk value-path type); recorded as the
  first FN Vulkan baseline (`scripts/gate-flash-baseline-vk.txt`, gate
  PASS against it).
- True value-path speed: tg 0.54 t/s, pp64 51.9 — host-staging bound as
  expected; the earlier pp512/tg16 numbers printed by bench were the HIP
  frame path (bench now routes the vk runtime correctly).

The fast Vulkan path (frame port: hc/GDN/QSA-indexer/MoE-route/PLE as
resident FrameHost ops, coopMat tile plates incl. q5_1) remains the
multi-session core of plans/84 B; the entry map in plans/84 is updated.

## 2026-09-20 (8) — FN chunk divergence bisect: PLE exonerated, multi-family t-dispatch (plans/84 E.2)

Three bisect facts on the Flash-Next HIP frame-path chunk divergence
(chunk-check 16/63/64 FAIL vs the single-chunk reference):

1. **PLE is not the (sole) source**: `LLM170_STAGE_SKIP=ple` still fails
   (max|d| 3.46), despite the bufhash first-diff sitting at the blk.1
   residual (the earlier suspicion from the L1B.res_hc dump position).
2. **Disabling the tile path changes both sides**: `LLM170_Q4_NO_TILE=1`
   fails with a degraded reference (argmax 271 vs 17374) — several
   t-keyed kernel families differ between t=16 and t=208 calls, so
   single-switch bisects cannot isolate one culprit.
3. **The Vulkan value path is chunk-invariant** (Q4_CHUNK 16 == 512
   exactly, entry (7)) — the CPU stage math and all carried state are
   correct; the divergence is confined to the q4acc frame path's
   t-dependent dispatch (launch_gemm t>=16 tile/q5_1_m gates, t==1
   duals, MoE grouped paths).

This is the same defect class the qwen35 prefill had (entry (1)):
families are individually row-invariant but disagree with each other,
amplified chaotically. The fix is the same prefill family pin, applied
to the q4acc frame dispatch; chunk-check FN 16/63/64 is the fence. The
bufhash dump now also hashes the hc intermediates (lo/inj/gate) and the
PLE buffers for the next session's localization.

## 2026-09-20 (9) — FN chunk divergence: pin attempted and withdrawn; full RCA map (plans/84 E.2)

The qwen35 prefill family pin was ported to the q4acc frame path
(`frame_begin` sets PREFILL_PIN for t>1; `tile_core` forces the large-t
family). Result: chunk-check divergence shrinks (3.46 -> 2.16 max|d|)
but does not close, AND the default FN gate stream flips — the frame
prefill's default path itself uses small-t pieces (t_max cap), so the
pin changes production numerics without fixing the fence. A partial fix
that breaks the greedy-unchanged principle is worse than none: the pin
was withdrawn (gates green again); the finer bufhash markers
(res_attn/res_ffn/site-level inputs + row-split) stay as diagnostics.

Complete hypothesis ledger for the residual divergence (all tested):
- PLE skip: still fails. NO_TILE: both sides change. MOE_GROUPED forced:
  still fails (1.94). q5_1 mmq gate: both sides >=16. Tile family pin:
  partial (2.16) + gate flip -> withdrawn.
- Site-level bufhash paradox: f.mout hashed at the hc-ffn-combine call
  site differs between chunkings while every stage-level hash (mout,
  inj, res) matches and the same buffer matches again one layer later —
  the classic signature of either a read racing an in-flight async
  write (frame_read is a null-stream hipMemcpy against custom-stream
  pipelines) or a real ordering gap in the frame MoE/shared-expert
  pipeline. Resolving that is the next concrete step: audit stream
  ordering (pre_pair/stream3/4 + dual-tile side stream) around
  AxpyScaled/hc_combine, or make buf_hash device-synchronize first to
  de-noise the instrument.

The Vulkan value path remains chunk-exact, so the CPU stage graph and
all carried state are proven good; the defect is confined to the q4acc
frame pipeline.

## 2026-09-20 (10) — E.2 instrument hardened; drift profile measured (plans/84 E.2)

The bufhash instrument now device-synchronizes before every read
(`FrameHost::frame_sync`, default no-op; Q4Acc = ctx sync) and can dump
first-8-element bit values (`LLM170_DUMP_E2VALS=1`). This de-noised the
earlier mout "paradox" partially and produced the drift profile:

- With the family pin active (still withdrawn — see (9)), all stage
  hashes match through L2 and the first hash divergence sits at the
  hc-ffn-combine site read of f.mout; element sampling shows the drift
  is real but sub-8-element there, growing to visible last-bit
  differences (~1e-4 relative) by L9-L10 (L9B.lo/inj/gate onward).
- Without the pin the divergence enters earlier (hc up-projection gate
  at L0) — the pin demonstrably removes the tile-family component.

Remaining puzzle recorded honestly: f.mout hashed at the combine call
site differs while the same buffer hashed one dump later matches — with
device-synchronized reads. Whether that is a dump-interleaving artifact
of the multi-pass log or a genuine write between the two points is
unresolved; the drift itself is real and the MoE-output / hc-combine
boundary is the measured epicenter. The pin remains the proven partial
fix, blocked on the gate-flip question (default path uses small-t
pieces; landing the pin requires re-baselining the FN gate under the
chunk-invariance contract, which only makes sense once the fence
actually passes).

## 2026-09-20 (11) — E.2 settled: reads deterministic; MoE mout differs from L0 (shared-add/tile axis) (plans/84 E.2)

Triple-read discriminator at the hc-ffn-combine site (mout read twice
back-to-back and again after the combine kernel): all three reads are
bit-identical within each run at every layer — the reads are
deterministic, the combine does not write mout, and there is no async
artifact. The buffer genuinely differs between the t=16 chain and the
t=208 single pass at **every layer from L0** (hash-level; the drift
sits beyond the first 8 elements of row 0, consistent with the earlier
value sampling).

Combined with the stage hashes: the grouped-expert outputs (my, mwt,
mids, mxsel, mgu) all match — the divergence enters at mout, i.e. the
**scatter/shared-expert add boundary**. Unpinned, the shared-expert
GEMMs run through the t-keyed tile families (mm/wm/j128), which is the
component the family pin removes; with the pin active L0-L1 mout match
and the residual re-emerges at L2 (the layer before the first QSA
layer). The remaining suspects at L2+ are the QSA-layer feedback into
the residual (its attention kernels are t>3 sel4 — same family for 16
and 208, but its indexer/top-k list build is per-chunk) and any
sub-hash-threshold route drift flipping a late expert.

The chunk-invariance epicenter is now single-buffer precise: f.mout.
Next session enters at the shared-expert add (shg/shu/shd GEMM dispatch
under pin) and the L2 route chain, with the discriminator available as
`LLM170_DUMP=bufhash` site markers.

## 2026-09-20 (12) — E.2: drift is present in the FIRST observable hc-mix output (plans/84 E.2)

Shared-expert site dumps (shg/shout/msgate) show every shared-add input
carries a ~1e-4 relative drift between chunkings from L0 onward
(msgate 3e8e362e vs 3e8e4a41 class), and the first hc-ffn mix output
(L1-top) already drifts in row 0 element 0 — the earlier "mout
epicenter" is inherited, not local. All site-hashed intermediates of
the ffn half (lo/inj/gate/res) match, which means the drift enters
upstream of them: the attention half of L0. Its intermediates (hc_attn
mix/lo/inj/gate, GDN qkv) are overwritten by the ffn half before any
dump point — invisible to the current instrument. The GDN projections
run through the same t-keyed tile dispatch (unpinned), consistent with
the tile-family pin moving the first visible divergence deeper.

Next instrumentation: dump the hc_attn-half intermediates at L0 (before
the ffn half overwrites them) — one site marker in hc_mix_frame for
kind="attn" at il==0 — expected to expose the first differing GEMM
output directly.

## 2026-09-20 (13) — E.2 ROOT CAUSE ISOLATED: two dispatch axes, now single-kernel precise (plans/84 E.2)

Attention-half site markers (hc_attn xn/lo/inj/gate/mix at L0-L2) closed
the observation gap and produced the complete causal chain, verified
with the synced instrument:

- **Unpinned**: the first differing output is the hc_attn down
  projection (q8_0) — identical rms input, t=16 takes the GEMV family,
  t>64 the j128 tile family (attn_lo DIFF, attn_inj/xn same because the
  f32 inject and rms are row-invariant). Everything downstream (gate,
  mix, GDN, MoE, mout, residual) inherits the drift.
- **Pinned (large-t family forced)**: L0-L2 attention halves, ffn
  halves, router, and shared-expert inputs are all bit-identical — the
  divergence moves to a single new entry: **the grouped MoE gate GEMM
  (mgu)** at L2 (mxsel/mids/mwt all identical, mgu differs between a
  rows=160 and a rows=2080 call). The grouped per-expert 16-row-padded
  tiles are row-count dependent — the last axis.

The pin now ships as an opt-in (`LLM170_Q4_PF_PIN=1`, default off —
default-path numerics unchanged, gates green) so the fence
investigation and the eventual grouped-kernel fix can proceed without
re-baselining anything until chunk-check passes. Landing order for the
fix: make the grouped GEMM row-invariant (or pin its dispatch), then
enable the pin by default, re-record the FN gate under the
chunk-invariance contract.

## 2026-09-20 (14) — E.2: the divergence is a mid-layer transient (two buffers, converges by layer boundary) (plans/84 E.2)

The built-in `LLM170_DUMP=moe` hash diagnostics (plans/80 asset, first
production use) sharpen the picture further under the opt-in pin: the
gathered MoE input (f32 x_h and quantized xq_h of mxsel) differs
between chunkings at the L2 gate/up GEMM call sites — while the same
buffers hash EQUAL at the layer-boundary dumps, with device
synchronization at every read. Two buffers (mxsel and its source mix)
now show the same signature: different mid-call, equal post-layer.

Values that genuinely differ at a consumer's call site and equal one
layer later, under synced reads, mean a producer ordering defect in the
frame pipeline (a later kernel rewrites the buffer with the convergent
value after the consumer already read the divergent one) rather than a
pure arithmetic-family issue. Candidate mechanism: the hc-ffn mix /
moe-gather handoff inside moe_frame reading a buffer whose final write
for the layer has not been sequenced before the gather (host-grouping
h2d on the main stream vs gather/permutation kernels, or the quant
cache serving a stale generation). Next session should trace the exact
write sequence on f.mix between HcGateMean and the gather — the
transient signature narrows the search to that window.

## 2026-09-20 (15) — E.2 smoking gun: MoE GEMM inputs bit-identical, outputs differ (plans/84 E.2)

`LLM170_QHIST=1` (new probe: FNV hash of the quantized activation at
every MoE quant-miss) closes the chain under the opt-in pin:

- Layers 0-1: gate and down quantized inputs hash-identical between
  the t=208 single pass and the t=16 chain (and every buffer hash
  matches end to end).
- **L2 gate quantized input (n_in=2560): IDENTICAL** (83ea8fe1... both
  runs) — yet L2's gate GEMM output (mgu) differs.
- **L2 down quantized input (n_in=640): differs** — downstream of mgu.

So with bit-identical quantized input, identical weights, and the
row-local ge kernel, the grouped MoE GEMM chain (quant -> permute/gather
-> q4_gemm_q4k_ge -> inverse scatter) produces different outputs
between a rows=160 and a rows=2080 invocation. The defect is inside the
grouped machinery — permutation tables, gather/scatter, or the kernel's
row guard — not in any upstream arithmetic.

Next session enters with a standalone reproduction: same weights, same
xq, rows=160 vs 2080 through frame_moe_gemm (q4_K, n_in=2560), bisect
permute/gemm/scatter by hashing xg and yg in the permuted domain.
Recipe: LLM170_Q4_PF_PIN=1 LLM170_QHIST=1, layer 2, first divergence
at the fifth quant-miss.

## 2026-09-20 (16) — E.2: ge machinery exonerated; the quant output itself is sync-placement-dependent (plans/84 E.2)

Permuted-domain bisect (`LLM170_E2PERM`, new probe hashing xg before and
yg after q4_gemm_q4k_ge at each original row's permuted position;
MoeGroup now caches the host inverse permutation for it):

- L0 and L1 gate/up chains: xg AND yg bit-identical between the t=208
  single pass and the t=16 chain — the permute/gemm/scatter machinery
  is exact on identical input.
- L2 gate: xg (the permuted copy of the quantized mxsel) already
  differs — c0c4d7d9 vs 3388ec6c in a minimal-sync run. But the same
  quant hashed IDENTICAL in a run with a device sync right after each
  quant (LLM170_QHIST, ledger (15)).

Conclusion: the quant kernel's OUTPUT for L2's gate differs depending
on whether an intervening device-wide sync occurred — a producer
ordering defect in the quant input chain (fxq-pool reuse, gather, or an
async op completing late), not arithmetic. Static stream analysis shows
all kernels on one stream, so the defect lives in something the sync
drains (async h2d/d2h_issue on stream2+, or a pool alias). Next
session: trace every writer of the fxq buffer between L1-down-quant and
L2-gate-quant (canary hash around each candidate) — the defect is now
one buffer and one call window wide.

## 2026-09-20 (17) — E.2 final localization: the race sits in the L2 gate quant window (plans/84 E.2)

Input-side probe (`[nxh]`, hashes the quant input mxsel at every
quant-miss under LLM170_E2PERM): in the same synced run the L2 gate
quant input is bit-identical and the first divergence is L2's mglu
(silu of the gate/up outputs) — while in the minimal-sync run the L2
gate quant OUTPUT (xg) differs. The defect therefore sits in the window
between the L2 gather and the gate GEMM: the quant of mxsel (or its
consumption) is timing-dependent, syncs mask it, and everything
downstream inherits. All probes agree; the mechanism (which async op
the sync drains) is the only remaining unknown — writer-trace on the
fxq/mxsel buffers in that window is the next and final step. E.2
investigation state: defect class identified (producer ordering),
epicenter one call window, masking sync characterized, all probes
committed and reproducible via LLM170_Q4_PF_PIN=1 LLM170_E2PERM=1.

## 2026-09-21 (18) — E.2 root cause found and fixed: MoE fallback family split by per-expert row count; prefill pin now default (plans/84 E.2)

Chain of evidence that closed the nine-step hunt:

- Same-domain probes: router logits and top-10 ids bit-identical at
  L0-L2 for every chunk; first divergence enters at L2's MoE *output*.
- Branch probe: per-layer expert weight types are mixed (UD-Q4_K_XL) —
  gate/up are Q4_K or Q5_K per layer, down is Q5_1/Q8_0. The Q4_K
  grouped ge machinery is exact (L0/L1 bit-identical). L2's gate/up are
  Q5_K → they run the *fallback* path (per-expert GEMM launches).
- The fallback passes each expert's row count `r` as `t` to
  `launch_gemm`, whose family split `t >= 16 → tile, t < 16 → GEMV`
  depends on `r`. `r` depends on chunking (same expert: r≈40 in a
  208-token chunk, r≈1-3 in a 16-token chunk), so the same
  (token, expert) product used different kernel arithmetic per
  chunking — the ~1ulp family difference (ledger (5)) amplified through
  45 layers into the observed 2.1 max|Δ|. The PREFILL_PIN did not cover
  this threshold. All earlier "sync-masking" readings were line
  attribution errors: Q4_K-only dumps skip Q5_K/Q8_0 layers.

Fix (`launch_gemm`): while the prefill pin is active, the tile branch
is taken for **every** t (the pin fixes the tile family to j128+large,
making per-row arithmetic independent of t). Decode (t=1, pin off)
keeps the GEMV path. The pin is now **default ON** (LLM170_Q4_PF_PIN=0
to disable), completing plans/84 E.2: chunk-check 16/63/64/128/512 all
bits-identical with no env vars.

Validation of the changed default stream: the pinned hip output's
first 8 tokens exactly match the independent Vulkan implementation's
baseline (the old mixed-family hip output diverged from it at token
4) — the pinned path is the more correct one. FN hip gate re-recorded
under the chunk-invariance contract. Perf: 208+32 wall 42.3-42.5s vs
42.1-42.5s unpinned (noise). Investigation probes (E2PERM/E2IDS/E2BR/
TBSYNC/nxh/nxv, MoeGroup.inv_host) removed; L3Q bufhash markers kept
in the established dump vocabulary.

## 2026-09-21 (19) — Vulkan q5_1 coopMat tile plate (plans/84 B)

`tile128_q51.comp`: the q5_K 128-row coopMat plate specialized for
q5_1 (the FN expert-down mass, 600 MB/tensor × 48 layers). Three
corrections were needed on the way, each caught by the vk-tile-check
fence (now arch-aware so it loads multipart Flash-Next):

- Block stride is 24 B (d,m f16 + qh 4 B + qs 16 B), and the nibble
  order is llama's interleaved-by-16: element j<16 reads the low
  nibble of qs byte j, j>=16 the high nibble of byte j-16; the 5th bit
  is qh bit j. deq_q5_1, gemv3 ty=7, and the hip kernels all use this
  order; a sequential-pair misread still produced plausible-looking
  values — only the reference cross-check caught it.
- Big-tensor addressing: WG() chunking now takes the chunk capacity
  from the push constant (wsh) instead of the hardcoded 25-bit split —
  600 MB stacks exceed 134 MB chunks. Residual chunk over-reads are
  covered by robustBufferAccess.
- The plate's push is 6 fields [n_in,n_out,xq_w,nt,tok_base,wsh]
  (24 B): tok_base fixes sub-128-token slabs (the q5_K plate's 4-field
  push cannot express token offsets for t>64).

Verified: blk.0/blk.3 ffn_down_exps at t=2/64/128 — maxrel 0.9e-3 to
1.7e-3 (f16-staging contract), 0 tokens over 2%. Wired opt-in via
LLM170_VK_TILE_Q51=1 (default off: the f16 tile is a different
precision class than the GEMV path, and a t>=2-only default would
recreate the family-split chunk divergence fixed in ledger (18)).
Perf on the FN vk value path (208-token prefill): 43.0-44.1 s →
42.3-42.5 s (~3%), consistent with the path being host-staging bound.
All four gates (FN/27B × hip/vk) pass.

## 2026-09-21 (20) — Vulkan frame core: buffer registry, elementwise FrameOps, resident frame_mm (plans/84 B)

First slice of the FN frame port on Vulkan: VkAcc now implements
FrameState (frame_begin + a handle registry of host-visible frame
buffers with direct write/read) and the FrameHost core:

- frame_mm/frame_mm_group: device-resident quant (the frame f32 buffer
  feeds the quant shader directly — no host roundtrip) then gemv_run
  per weight, writing into frame buffers.
- FrameOps: RmsRows (rms.comp gained a w_reps field, hip
  rms_finish convention — weight indexed w[(row % w_reps)*n + i];
  w_reps=1 keeps the old arithmetic), SiluDiv, SiluMul, Scale,
  CopyRows, BcastRows, AxpyScaled (per-token scale variant).

Two integration traps found and fixed: push-constant sizes must match
the registered range exactly (over-sized ranges on 8/12-byte pushes
crash the driver), and the q35 VkDecoder shares RMS_SPV — its launch
was updated to the 16-byte [n, t, w_reps=1, eps] push (caught by the
27B vk gate going degenerate; stash-bisect isolated the shader).

Fence: `vk-frame-check <file> <tensor>` (arch-aware) — RmsRows(w_reps=2)
7.1e-8, elementwise ≤1.6e-8, frame_mm 9.2e-4 vs CPU dequant.
All four gates pass. Remaining for the FN frame path: MoE ops
(top10/gather/scatter/weighted-sum + grouped GEMM), hc/GDN/QSA/PLE
stages, then chunk-check + perf.

## 2026-09-21 (21) — Vulkan frame MoE: top10, grouped per-expert GEMV, weighted-sum; frame capability gate; vk baseline corrected (plans/84 B)

Second frame slice: MoeTop10 (workgroup-parallel softmax + deterministic
descending top-k, ties to the lower expert — same semantics as hip
q4_moe_top10_m), frame_moe_gemm (host grouping like the hip fallback,
device gather via u32 row permute, per-expert gemv_run_off with
descriptor-offset bindings, scatter by inverse permutation), and
MoeWeightedSum. Offset binding (VkCtx::bind_bufs_off) enables expert
slices of xq/yg/weight without extra buffers; the vk GEMV is a single
family, so per-expert row counts cannot split arithmetic the way the
hip fallback did (ledger (18) class).

FrameHost gained `frame_capable()` (default true); VkAcc returns
LLM170_VK_FRAME=1-gated so the partially implemented frame op set
cannot break the default engine flow (a partial FrameHost otherwise
sends the whole FN vk forward into "frame_mm_group: 타입 미지원" —
the f32 inject weight — instead of the value path).

Baseline correction discovered while validating: the FN vk gate
baseline recorded on 2026-09-20 captured a hip-class run (real vk
value-path runs take ~145 s vs hip ~45 s for the 208+16 gate prompt
and produce the value-path stream `16 19 ...`, not `16 23 ...`).
Re-recorded with the real vk path after verifying determinism
(two identical runs) and that the rms w_reps change is innocent
(reverting it changes nothing). vk-frame-check now covers the full
MoE chain (top10 → grouped GEMM → weighted-sum): max|D|=6.28e-4 vs
CPU. All four gates pass. Remaining for frame-capable-by-default:
hc/GDN/QSA/PLE stages + PLE streaming.

## 2026-09-21 (22) — Vulkan frame attention half: hc/GDN ops, L2, frame_gdn_ar; FN shader-namespace collision fixed (plans/84 B)

Third frame slice — the GDN/hc half of the FN layer now runs as frame
ops: HcGateMean, HcCombine, NormGated(sigmoid), GdnBetaG, Sigmoid,
Split3 (reusing the q35 value-path plate), L2Rows(+2Scale, f64
reduction class), GdnConv (parallel chunk plate + ring-state update +
sequential tail), and FrameState::frame_gdn_ar reusing the q35
gdn_ar.spv plate (sequential over t — chunk-invariant by causality).
frame_mm_group gained a value-path pullback for unsupported weight
types (the f32 inject) so the hc half completes end to end.
vk-frame-check covers all of them (<=1.6e-7 vs CPU mirrors).

Two collisions found the hard way: my gdn_conv_state/gdn_beta_g
shaders overwrote EXISTING q35 decoder shaders — the 27B vk gate went
degenerate (reproducible; the original conv-state plate uses a 1-D
grid with a negative-index fallback mine lacked). Fixed by restoring
the originals and renaming the FN variants (fn_gdn_conv_state; beta_g
was numerically identical but also restored for cleanliness). Lesson
recorded: the spv directory is a shared namespace — new backends get
fresh names.

End-to-end status: with LLM170_VK_FRAME=1 the FN forward now proceeds
through hc, GDN, MoE and reaches the tail allocations, then hits the
Vulkan per-allocation memory ceiling (single ctx-scaled frame buffers
of 6-10 GiB; hip allocates these fine, RADV does not). Remaining for
frame-capable-by-default: split/large-frame budgeting on vk, the QSA
half (qk_norm_rope, indexer top-k, attention via the qsa_flash_gq
plate), PLE. All four gates pass.

## 2026-09-21 (23) — vk frame end-to-end status: ctx-1024 passes allocations; QSA attention is the functional gap (plans/84 B)

Instrumented alloc_host failures (>1 GiB) with a backtrace and ran the
opt-in frame path (LLM170_VK_FRAME=1) at descending ctx: at ctx 1024
no allocation fails and the forward proceeds through hc, GDN, MoE and
the QSA indexer, reaching "qsa_attention_sel: 이 가속기는 미지원" —
the engine falls back to a CPU QSA recompute (too slow to finish in
400 s). So the remaining functional gap for the vk frame path is the
QSA attention triple (indexer top-k selection, kv append, attention
over the selected set — the q35 decoder already has a
qsa_flash_gq-based attention plate to reuse). The >4 GiB single
allocations seen at ctx 2048+ remain a separate budget item (sizes
6.7-10.9 GiB do not match any single frame buffer in Frame4::new —
weight-chunk suspicion, backtrace hook is in place to pin it).

## 2026-09-21 (24) — vk frame path runs FN end to end: QSA triple + the stacked-n_out scratch bug (plans/84 B)

QsaOps for VkAcc: resident pools per (layer, seq) with hip-equivalent
watermark rules (sequential append / prefix rewind), device-to-device
kv and indexer appends (copy plate), block-key update
(fn_idx_bk_update — r-row mean, f64 rms, weight, rope via the cs
table), and the selection-list attention (fn_qsa_attn_sel — subgroup
64 = one (token, head), lane covers 4 of hd=256, online softmax with
correction, sigmoid gate on output; walks only the selected
positions). set_ctx_len feeds the pool capacity.

The backtrace hook (23) pinned the 6-10 GiB "memory ceiling": all of
it was one bug — frame_moe_gemm sized its scratch by the STACKED
n_out (per-expert width x 512 experts), asking for 10.9 GiB in a
single buffer. With n_out divided by the expert count the whole frame
path fits, and with LLM170_VK_FRAME=1 the FN forward now completes
end to end at ctx 8192 (208+2 tokens, 128.5 s, coherent output). The
frame path is ~2.9x slower than the vk value path for now (per-expert
GEMV launches without batching + per-layer host grouping) —
frame_capable stays opt-in until launch batching and the decode-side
QSA selection (qsa_sel_dev) land. All four gates pass;
vk-frame-check MoE chain re-verified against the per-expert width
contract (probe updated; the earlier pass was reading stacked-stride
aliases).

## 2026-09-21 (25) — vk frame MoE launch batching; fresh-set layout trap (plans/84 B)

frame_moe_gemm now wraps gather → per-expert GEMVs → scatter in a
batch session (one submit). Trap found: under batching the per-expert
path must allocate a FRESH descriptor set per expert — rebinding a
recorded set between dispatches is illegal — and fresh_ds picks up
whatever layout was last registered on the batch context; the gather
pipeline's 3-binding layout then backed the GEMV's 11-binding set and
segfaulted. Fixed with VkCtx::fresh_ds_for(&Pipes) which pins the
layout. End-to-end frame run: 128.5 s → 122.2 s (~5%); the dominant
remaining cost is the per-layer host grouping (ids readback + table
build × 3 MoE GEMMs × 48 layers = ~144 sync points) — the same
motivation as hip's device-group path (its own negative-result history
in ledger (5) applies: device grouping only pays once down is
covered too). Frame capability stays opt-in until that lands; the
value path remains the default vk route. All four gates + the full
vk-frame-check suite pass.

## 2026-09-21 (26) — vk frame chunk-invariance hunt: two real bugs fixed, one residual isolated (plans/84 B)

Ran the chunk-invariance fence on the opt-in vk frame path
(LLM170_GPU_RUNTIME=vulkan now switches the diagnostic accelerator
factory too) and hunted the divergence with the E.2 methodology —
stage checksums, bufhash markers, and per-op isolation fences added
to vk-frame-check (GdnARchunk, GdnConvChunk, GdnBetaGChunk at real
shapes, and a chunked MoE gate/down comparison with proper per-chunk
route/mx staging).

Fixed, each independently verified:
- GdnBetaG launched with n_h/128 workgroups against a 64-thread
  plate — half the rows unwritten; which rows depended on the chunk
  size, corrupting bg and everything downstream. (An earlier
  isolation "pass" at tiny shapes hid it; the d>=128 layout
  requirement of the AR plate also surfaced — the state row is
  kdim=128 wide, so d<128 makes u-rows overlap and race.)
- frame_moe_gemm sized its scratch by the stacked n_out (expert
  width x 512) — the 6-10 GiB "memory ceiling" of ledger (23) was
  entirely this; divided out, the frame path fits at full ctx.
Also landed: qsa_host_rebuild (host-visible pools), xg stride
padded to 16B for descriptor-offset alignment, G0.* bufhash markers
inside the GDN frame, and the swap-layout AR plate rebuilt.

Verified bit-identical across chunkings in isolation: gate GEMM,
down GEMM (64 vs 4x16 with real stacked weights), GDN AR (state and
outputs), GDN conv (ring and outputs), beta-g at the real dt_rank.
In the full run the markers now agree through all 48 layers for the
first tokens, the reference produces a sane stream (argmax matches
across chunkings), but a late-token residual remains: at L0 the
token-207 MoE output differs (inputs mids/mwt/mixf all bit-identical,
and every isolated component invariant) — max|D| 5.3 on final
logits. Status recorded for the next session; the frame path stays
opt-in so production paths are unaffected. All four gates pass.

## 2026-09-21 (27) — root cause of the vk frame residual: token-strided gather; chunk-invariance achieved (plans/84 B)

The late-token residual of ledger (26) is closed. Direct hashes of
mxsel/mids/mgu at the last prefill token showed the gathered MoE
input diverging while its source (mix) was bit-identical:
frame_moe_gather had reused the BcastRows plate, which broadcasts a
SINGLE source row to t*k slots — the engine contract is a
token-strided gather, xsel[(ti*k+s)*n] = mix[ti*n]. Both chunkings
computed the same wrong broadcast for chunk 1 (which is why the
first-16-token markers agreed), and diverged from the second chunk
on, exactly the observed signature. New moe_gather.comp implements
the strided gather (hip q4_moe_gather semantics).

With the fix the vk frame path passes the chunk-invariance fence:
chunk-check 16/63/64 bits-identical (LLM170_VK_FRAME=1,
LLM170_GPU_RUNTIME=vulkan), completing the numerics leg of the B
verification. vk-frame-check full suite passes (including the
GdnAR/GdnConv/GdnBetaG chunk isolation fences and the chunked MoE
gate/down + f32-router comparisons, all bit-identical), and all four
production gates stay green. Measurement at ctx 8192 (208+16):
frame 154.4 s vs value path 155.4 s — parity at long context (the
frame's device-resident state offsets its per-layer host grouping;
at ctx 1024 the value path still leads). The frame path remains
opt-in; remaining B items are performance work (device-side MoE
grouping, decode-side qsa_sel_dev) and PLE.

## 2026-09-21 (28) — session close: shexp decode work reverted; four gates green at HEAD (plans/84 B)

Attempted the decode shared-expert pair (shexp_gu/shexp_da as EwOps
for VkAcc). Two findings before reverting the (uncommitted) work:
- Self-deadlock: calling frame_alloc while holding the ctx lock
  (parking_lot is non-reentrant) hung the FN vk gate at VRAM 31% /
  GPU 0% indefinitely — fix is allocating frame scratch before
  taking the lock (applied, then reverted with the rest).
- After the deadlock fix the decode still dies with
  `제출: ERROR_DEVICE_LOST` (reproduced twice; prefill passes). Root
  cause not yet isolated — suspicion: frame_free destroying in-flight
  buffers, batching-mode descriptor sets, or lingering device
  poisoning from the deadlock era (needs a clean-GPU recheck).
The change was reverted to keep HEAD at the fully-verified state
(all four gates pass, chunk fence bits-identical); both findings and
the full implementation recipe are carried into the follow-up plan.
Also confirmed this session: the FN vk cold start takes 5-10+ min on
this APU — mmap fault-path uploads at 20-180 MB/s for 78 GB, and the
page cache (competing with VRAM in unified memory) rarely survives
between runs; sequential pre-warm reads run at ~2.5 GB/s. Weight
preload via the pread staging path is a standing improvement item.

## 2026-09-21 (29) — vk decode shexp landed; QSA selection chain verified; two latent bugs found in the engine/frame path (plans/85 §1-§2)

Follow-up to (28) on branch vk85-shexp.

**Shexp (§1) landed and verified.** shexp_gu/shexp_da for VkAcc are
composed from the existing tested primitives (frame_mm_group quant-once
+ gemv, SiluMul, AxpyScaled) with scratch pre-allocated before any ctx
lock — the deadlock trap cannot occur by construction. Numeric check
added to vk-frame-check section 15 (CPU dequant reference): h max|D|=
2.3e-4, mout 6.7e-5. Decode n-predict 8 completes with no deadlock and
no DEVICE_LOST, and the 16/63/64 chunk fence stays bits-identical.

**The real root cause of the (28) DEVICE_LOST was not shexp.** The CLI
injects LLM170_FRAME=1 by default, and decode1/decode1_greedy attempted
the frame path gated only by that env — not by frame_capable() (prefill
does check it). On vk without LLM170_VK_FRAME this runs value-prefill +
frame-decode with a freshly constructed Frame4 (all state buffers
uninitialized); with shexp enabled the step progresses past the old
abort point into an op that faults the GPUVM (PERMISSION_FAULTS, context
lost). Reproduced on both the §1 binary and HEAD+shexp; bisected via
kill-switches (LLM170_VK_SHEXP=0 / LLM170_VK_POOL=0) to shexp exposure,
not the new frame-buffer recycle pool. Fix: decode1 and decode1_greedy
now also require frame_capable() — the default (non-VK_FRAME) path is a
clean value decode again, identical first tokens to HEAD.

**Frame buffer recycle pool.** VkBuf has no Drop; the old frame_free
leaked every buffer for process lifetime — unbounded for per-step decode
scratch. frame_alloc/frame_free now recycle through a 64 MiB pool
(smallest-fit; surplus beyond the cap retains the old behavior).

**QSA decode selection (§2) implemented and unit-verified.** New
shaders fn_idx_q_rope/score/rank/expand (hip q4_idx_* arithmetic order:
f64 sequential rms, f64 rotation, 4-accumulator dot, integer rank) plus
qsa_sel_dev/qsa_attention_dev_sel/qsa_sel_readback for VkAcc and
frame_argmax_rows (multi-row argmax2 variant, ties→lowest index).
vk-frame-check section 14 builds a 1025-position synthetic pool and
compares the device list against a host mirror: scores agree to 7 digits
and the selection list is bit-identical.

**Two latent bugs fixed on the way.** (1) qsa_idx_append_dev's watermark
update wrote a temporary (`&mut get_mut().map().unwrap_or(0)`) — the
stored watermark never moved (kv_dev masked it) and a missing pool entry
panicked on unwrap; now entry()-created and really updated. (2) Push
constants whose GLSL block leads with a float (fn_idx_bk_update,
fn_qsa_attn_sel) were pushed in the reverse order — fn_idx_bk always
early-returned on idx_dim≠128, so block keys were never computed on vk
(invisible: prefill used the identity-selection shortcut). All pushes
now follow the declaration order (eps/kq_scale first). The q_rope cs
table is uploaded per decode position and indexed row-relative;
absolute indexing read out of bounds and robustness zero-filled iqr.

**New blocker documented (pre-existing).** With LLM170_VK_FRAME=1 the
frame decode step now runs to completion (argmax included) but its
logits are wrong (divergent tokens after the first). Every QSA layer
still host-bridges because frame_qk_norm_rope is unimplemented for
vk, so the corruption sits in the t=1 device path (GDN/MoE/head or
state sync) — it was never observable before because the step always
aborted at shexp/argmax and the output was discarded. End-to-end §2
acceptance ("decode without fallback") is blocked on this; the selection
chain itself is verified at the op level.

Gates: all four green at the branch tip; vk frame chunk fence 16/63/64
bits-identical (see plans/85 update).

## (30) Vulkan frame pipeline completed — default ON

**Correctness chain.** The t=1 frame decode logit contamination
(blocker of plans/86 §1) was two independent kernel defects, found by
per-op shadow checks (hip-vs-vk layer checksums, then an engine-side
absolute CPU diff over the hc attn mix):
- `silu_div.comp` computed silu(x)/div; every reference (CPU stages,
  hip q4_silu_div) is silu(x/div). The hc low-rank `lo` collapsed to
  ~0 in every layer, crushing the logit scale (top logit 6.2 vs 19.9).
  The old frame_check §3 reference enshrined the same wrong formula.
- `gdn_conv_seq.comp` (the t<k-1 decode path) gated 63 of 64 lanes and
  ran a (ch/64)-sized grid — only 1/64 channels of the causal conv
  were computed. Mirrored hip's channel=thread mapping.

With both fixed, the VK_FRAME greedy 16-token stream equals the hip
gate baseline (last token the documented 0.27nat tie).

**GPUVM fault on fresh-Frame4 decode** (§1b): the weight cache was
keyed by data pointer alone; value-prefill per-expert slice views share
the base pointer with the full expert stacks, so the frame MoE GEMM
offset-bound expert slices on a 1-expert buffer and faulted. Cache key
is now (ptr, len).

**QSA fully on device** (§2): fn_qk_norm_rope.comp ports hip
qk_norm_rope (32-segment f32 partials -> f64 sequential sum, f64 rope,
per-head tiled norm weights, k baked with kq_scale=1). No value-bridge
fallback remains; device selection lists are bit-identical to host
(SELCHECK, 8 steps x 24 layers).

**Transactional fallback** (§3): decode/prefill frame attempts snapshot
PLE state (hist, next_pos, conv ring) and restore it before the value
fallback — ple_hash re-entry used to see a broken hist_valid and
double-advance the ring, giving abort-point-dependent fallback tokens.
LLM170_FRAME_FAILAT=<layer> injects failures for the acceptance test:
every abort layer reproduces the pure value-path tokens. This also
exposed a panic (mask_from_list indexing an empty slice) reachable
whenever QSA pools have watermark holes (e.g. value prefill + fresh
frame); fixed, and the missing upload-path qsa_attention_dev was
implemented for VkAcc (cached scratch).

**Allocation ledger + MoE scratch leak** (§5): LLM170_DUMP=alloc tags
every GPU buffer allocation by site. It identified the pp4096 OOM:
moebufs' all-four-must-fit growth never stabilized across gate/down
size classes, reallocating 33.6GiB over one pp4096 run and overflowing
the carve-out. Per-component growth: 79.8MiB total.

**Cold start** (§6): weight uploads stage via sequential 8MiB preads
from the part files (~1.2GB/s) instead of demand-paging the mmap
(20-180MB/s). Cold-cache FN gate: 110.6s (< 3min criterion).

**PLE stays host-bridged** (§7 review): the iq4_nl PLE table is
26.8GiB; current residency is 77.1GiB (75.9 weights) of the 96GiB
carve-out — device residency does not fit.

**Default flip** (§8): frame_capable() now defaults ON
(LLM170_VK_FRAME=0 kills). Justified by pp2048 +135%, pp4096 +158%,
tg128 5.2x over the value path, with gates 4/4, vk-frame-check (incl.
new absolute GdnConvT1/GdnART1/HeadChain sections) and the chunk fence
16/63/64 bits-identical re-verified after the flip. The vk gate
baseline was re-recorded: the frame sequence equals the canonical hip
baseline embedded in the gate script (the live hip recording had
drifted on this machine).

## (31) Vulkan MoE: direct-ids decode + grouped tiles — plans/88

**Decode orchestration (B1)**: `fn_moe_ids.comp` is gemv3 given an
ids binding — grid (n_out, rows), each workgroup computes one (row,
output) with the expert base `ids[r]·per_expert` resolved in-kernel.
The ids d2h drain, host grouping, perm/inv uploads, gather,
512-expert launch loop and scatter are gone. Per-element arithmetic
is bit-identical to the per-expert gemv3 path (same lane split, f64
subgroupAdd, subgroup tree).

**Step-level batching**: the new [ts] submit counter (LLM170_VK_TS
reports submits per segment) measured 2548 submits per decode step —
each non-batched run is submit+fence-wait. frame_begin opens a batch;
frame ops re-open it after mid-step flushes (PLE/QSA host bridges);
the value path never consults the gate (frame_step_batch), so leaked
batch state cannot corrupt its synchronous downloads. Fallback entry
points flush explicitly. Submits per step: 6.

**f32/BF16 dense GEMV** (`fn_mm_f32.comp`): the F32/BF16 members of
frame_mm_group groups forced the whole group through the value
pullback (sync-flush + CPU matmul + writeback) 291 times per step.
The kernel consumes the frame f32 buffer directly; quantization is
skipped for all-dense groups. Pullbacks: 0.

**Grouped prefill (B2)**: `fn_moe_group.comp` builds the 16-row
padding domain entirely on device (one 256-thread launch;
within-expert order is atomically nondeterministic but per-row
outputs are order-independent, so the chunk fence is unaffected).
`fn_moe_tile_q4k/q51.comp`: 16x16 tiles, tile=expert, cooperative
LDS staging (chunk resolver amortized to 2 divisions per row), x read
indirectly via perm_pad — the gather pass disappears. A generation
cache lets a layer's 3 GEMMs share the tables. The tile's float
expressions are aligned with the gemv3 class — outputs are
bit-identical to the direct-ids path (frame-check 9c cross-compares
all three MoE paths element-wise).

**Dense prefill tiles**: gemv3's token-loop re-reads ran at ~5GB/s
(6.6s of a 208-token prefill). `fn_tile_q8.comp` (K-sliced LDS
staging, any n_in) and mode-1 of the moe tiles replace it for t>=2.
The reduction order is a new class: gate re-recorded per the §8
precedent after cross-justification (MoE paths bit-identical, ckdiff
shows ulp-cascade, and the new stream matches the hip runtime's
current tie resolution 9 tokens deep).

**Hardware lessons**: 64-wide Vulkan subgroups broke 32-lane
reduction assumptions (shared-memory reductions now); a 70KB shared
array silently exceeded the 64KB LDS budget and hung the device
(SIGKILL, no vk error) — tile staging is sized against the budget;
`rowexp`/d-section reads must be row-based (the 9c check at 2100
rows catches both the binding-order shift and the row-0 d-section
bug that small-row tests miss).

**Numbers** (FN Q4_K_XL, vulkan, solo): pp512@20k 10.4 -> 60.1 t/s,
pp4096 10.2 -> 54.7 (5.4x), tg128@4k 2.25 -> 7.15 (3.2x). Decode
step: 6 submits, 116ms GPU (dense gemv 64ms near BW, moe_ids 33ms vs
a 6.5ms BW floor), ~25ms host (PLE host bridge). The plans/88 tg 8+
expectation stops at 7.15: after removing the orchestration the
decode GPU floor itself is 116ms. Identified follow-ups: port
ple_math_dev (3 kernels, ~8ms/step, note exp() ulp class), and a
higher-occupancy decode MoE shape. P3 (coopmat q4_K, 200+ challenge)
not taken per its own gate; P4 (QSA prefill tile) skipped — QSA
attention is below the top-13 slots of the re-profile; P5 (dense
GEMM re-evaluation) was subsumed by the dense tiles.

## (32) Vulkan compute campaign — plans/89 (decode dmmv, prefill tiles, PLE device)

**Decode (P0)**: the frame path's dense GEMV/MoE GEMV families moved to
llama-dmmv geometry (64-thread, 2-row WG, f32-activation direct, subgroup
f32 add) — gemv3 256-thread/W4A8/f64-tree class replaced per the §8
precedent (element checks vs full-precision CPU dequant at 1e-7..1e-3,
ckdiff ulp-cascade, determinism 2x, baseline re-recorded). FN tg 7.15 →
15.1 (step 133 → 64.5ms; [ts] gemv 77ms → q8b 26ms, moe_ids 33 → ~8ms,
mm_f32 12 → 4).

**Prefill (P1)**: dense q8_0/q4_K prefill now dispatches the decoder's
coopmat ms/128 tile family; MoE q8_0/q5_K roles got 16x16 scalar tiles
(partial-superblock-safe — down n_in=640 is 2.5 superblocks); fn_tile_f32
kills the router's weight-per-token re-read; moe_gather is elementwise
parallel. FN pp512 60 → 103.6, pp4096 54.7 → 95.0.

**MoE coopmat tiles (q4_K/q5_1)**: expert-block geometry (16-row expert
block x 128 output cols), f16_exact subnormal-safe decode, row-max-d
scale separation, f32 drain. Numbers verified 1.1-2.3e-3 bad=0 incl
20k-row multi-block; pp512 205 t/s measured. **Parked default-OFF**: an
engine-context-only intermittent nondeterminism (bufhash first divergence
at L4A.mout; check-harness deterministic 6/6) — consistent with the
tile128v2 RADV-coopmat precedent. memoryBarrierShared did not resolve.

**PLE device port**: fn_ple_gate/conv/res reproduce exp_cr_exact (f64
Horner) and the 32-chunk rms combine instruction-for-instruction — the
gate stream is BIT-UNCHANGED vs the host bridge (the port's design goal),
PLE_CHECK shadow pass, mid-step flush eliminated.

**Traps found (diag-first)**: WG()'s hardcoded `idx >> 25` chunk split is
a 128MB-chunk-era relic — stack word addresses above 2^25 silently read
the dummy w1c binding (expert >= ~150 in a 472MB stack → exact-zero
blocks). A boundary-repair edit had dropped VkAcc::pipeline's cache
insert — every dispatch created a fresh pipeline+DSL+pool, exhausting
the descriptor pool at the 128-step bench; found via the new [dsc]
miss-trace, fixed, pool budget also raised 4x. The decoder's run_pipe_b
double-pushed ts labels (own push + run()'s tag push) — 27B decode
attribution was scrambled into 'op?' (85ms/step blind); fixed via
set_tag, revealing the decode is at the GEMV BW floor (the plans/79-era
rms 21ms/quant 11ms premise was stale).

**27B**: pp512 344 (llama vk 343 parity), tg 11.44 (llama 12.05).
Remaining axes (plans/89 residue): MoE-CM nondeterminism root cause
(+100 t/s pp at stake), QSA prefill device selection (needs a bitonic
topk per token — the serial rank/expand kernels are decode-only), 27B
large-t tile family (llama int-dot MMQ needs the spirv-as pipeline —
GLSL integer-dot is unsupported by the system glslc/glslang).

### (32b) MoE-CM nondeterminism — race verdict (plans/89 resume, 2026-09-23)

The 12-run stream histogram (identical binary, gate prompt) yields 10
distinct streams incl. degenerate repetition loops — a finite tie-variant
set is ruled out; combined with bit-level mout divergence under identical
inputs (bufhash, varying first layer), this is a genuine data race in the
coopmat MoE tiles that only manifests under engine working-set/occupancy
conditions. Isolated harnesses (engine-pattern 3-GEMM x N layers, t-sweep
46..512, late-layer scans blk.1..47, PLE-bridge skip, dmin-weighted scale
bound) are all deterministic — no reproduction outside the full graph.
Default stays OFF (deterministic greedy outranks +100 t/s). Toolkit:
vk-moe-cm-race, LLM170_MTC_IL, LLM170_VK_Q4KCM/Q51CM split knobs.

### (32c) MoE coopmat nondeterminism resolved to shape: sg1 stable (plans/89, 2026-09-23b)

The 1-subgroup (64-thread) q5_1 tile is engine-stable — 8/8 gate runs
bit-identical to the SCALAR baseline stream (argmax absorbs the f16
rounding; no re-record required) — while the 8-sg q51_cm and both q4_K
variants (8-sg and 1-sg) stay nondeterministic in-engine (4-6 runs, all
distinct, degenerate loops included). The race is subgroup-scheduling
shape-dependent (tile128v2 precedent): q51_sg1's short K loop (n_sub=20)
with 1 MMA chain is stable; q4k's 40-iteration loop is not, even at 1 sg.
Shipped: q5_1 down -> sg1 default (LLM170_VK_Q51SG1=0 kill switch); q4_K
scalar default. FN vk pp512 102.8 -> 131.8, pp4096 95.0 -> 121.6.

### (32d) QSA prefill device selection — multi-token kernels (plans/89 P1.3)

fn_idx_score_mt + fn_idx_topk_mt (per-token bitonic over shared u64 keys,
ties to lower index per the decode rank contract; arithmetic sel_off — no
readback) replace the prefill host selection for chunks past the identity
window: the 4 frame_read flushes and the CPU score/sort loop are gone.
Verification: chunk-check 3000-token — the device path chunk (512) is
bits-identical; the 208-chunk deviation (0.381, argmax equal) reproduces
the HOST path signature exactly (pre-existing FN non-invariance). pp4096
121.6 -> 128.5; pp16384@20k first-measured 118.5.

### (32e) OpSDot binary patcher + coalesced rms + dense tile routing (plans/89, 2026-09-22c)

**OpSDot production patcher.** Ubuntu spirv-tools 2025.1 SAIL rejects the
`PackedVectorFormat4x8Bit` literal, so GLSL cannot target integer dot products
directly. `scripts/patch_sdot.py` compiles a sentinel kernel (sdot4p: four
split multiplies), locates the chain in `spirv-dis` text, and rewrites the
SPIR-V words in place: final IAdd(5w) -> OpSDot(opcode 4450, 6w, format=0),
dead extraction instructions -> word-count-preserving OpNop tiles, plus
DotProduct/DotProductInput4x8BitPacked capabilities. Unwrap rule learned the
hard way: ShiftRight always passes, BitwiseAnd only when masked by constant
255; the packing mask (0x0F0F0F0F) is opaque and its result is the dot
operand (patching through it read raw high nibbles — max|D| 52). Zero sites
patched = hard fail (unpatched sentinel reads the top byte unsigned).
fn_moe_tile_q4k: 3078ms -> 635ms per pp512 bench (4.85x), arithmetic
bit-identical to the previous dot4 (signed byte sext). Build order:
build_spv.py then patch_sdot.py — rebuilding the .comp re-runs the patch.

**rms coalescing.** The rms kernel gave each lane a contiguous 320B segment,
so adjacent lanes were 320B apart — 32x transaction waste, 8GB/s. Interleaved
loads (lane u takes element i*32+u) cut 2.58ms -> 0.33ms per dispatch (same
class: f32 partial sums + ordered f64 combine; only the partition changed).
The stream coincidentally returned to the session-open token sequence.

**Dense tile routing completion.** ffn_chain_gpu and matmul_batch still fed
t>=2 q8_0/q4_K through the serial gemv3 t-loop (~50ms/dispatch at t=512);
routed to the same coopmat tiles dense_mm uses (same numeric class as P1.1a,
dead Tile128/Tile128Q51 plumbing removed). Two §8 baseline re-records
(f16-staging class change, then rms partition flip-back).

Benchmarks (quiet machine): FN vk pp512 131.8 -> ~180, pp4096 128.5 -> 178.6,
pp16384 118.5 -> 158.6, tg128 15.2 -> 17.9 (hip: 231-275 / 276 / 18.4).
27B vk pp512 336.9, pp16384 145.9 (hip 292 — dominated by the tile_ms128
family, the remaining major lever).

### (32f) s8 cooperative matrix on RADV gfx1151: emulated, 2.3x slower (plans/89 s8mmq, 2026-09-22d)

tile_q8128i (coopmat<int8> A/B, int32 accumulate, per-32k-block d·yd drain)
is numerically validated — maxrel 5.6e-3 vs CPU (activation-quantization
class, twice as precise as the f16-staged tile, 0/128 tokens over 2%) — but
RADV lowers s8 cooperative matrices to scalar emulation: checker bench
0.925ms/dispatch (36GB/s) vs the f16 tile's 0.396ms (84GB/s). The integer
route is therefore closed on this stack for dense tiles (the OpSDot scalar
patcher remains the fast integer-dot mechanism — it wins in the MoE tile
where per-thread register blocking is shallow). Kernel parked as an opt-in
(LLM170_TILE_I8=1) for hardware with native s8 MMA. Debugging note: the first
draft covered only 32 of 64 A-rows (4 threads/row instead of 2), and the
uninitialized shared-memory half produced -inf outputs — caught by
vk-tile-check in 0.3s.

### (33) refactor-90 ledger: numerics-frozen debt cleanup, module split, shared contracts (plans/90, 2026-09-22)

Scope: full-repo refactor executed as nine gated commits on `refactor-90`,
under the freeze rule (ops.rs/quant.rs canon moved-never-reordered; every
unit: release build 0-warnings, cargo test, FN/27B gates byte-identical).

**Phase A (dead code & debt)**: `matmul_multi` (0 callers) and 14 `let _ =`
dummies removed; `qwen35::greedy` re-exported from `matmul::greedy_from`
(bit-identical lowest-index tie); rope cos/sin tables unified into
`ops::rope_cs_table` (3 copies, identical arithmetic); the 6-way
`frame_on` gate and 2-way GPU pullback loops unified into `Engine4`
methods. One semantic fix surfaced: `decode1_greedy` treated
`LLM170_FRAME=0` as *set* (frame path stayed on) while the other five
sites honored the kill switch — the unified gate now honors `=0`
everywhere (default configs unaffected). The vk `MoeGrp.off/rows_pad`
per-generation reallocation leak is fixed with capacity guards; the group
kernel fully rewrites all outputs each `!hit`, so cross-generation buffer
reuse is content-safe. Probe kernels left production hipRTC SRC
(dot_roof/mfma_roof/bw_stream/q4_hca_repro, wmma_probe family, wmma2 v1,
wk8d) with their NAMES rows, CLI probes, and (previously) 2MB committed
spvtool binary; gguf gained open-time tensor-bounds and split.no<count
validation (one test re-pinned from "detected at read" to "rejected at
open" — the new intended contract).

**Phase B (structure)**: gemv.rs (6,126 lines) split — 2,962-line checker
family into `rawvk/checks.rs` (pure move), the rest into
`vkacc/{mod,dispatch,qsa,frame,matmul,ple}` (largest module now 816
lines); Slot/name/SPV/pipeline quadruple-match collapsed into one
`SLOTS` table (new kernel = enum variant + 1 row); 31 concluded shader
pairs archived to `spv/archive/` (kept: Q4KKP, q51_sg1, tile_ms4,
mmv_llm, sdot/idot probes — all verified live by include_bytes or
fs-read); `LLM170_VK_NR` polysemy resolved (NOROB / NUMROWS); hip↔vk
neutral `common/` layer shared where safe (pread part staging,
QSA watermark rule, MoE grouping contract + cache-hit) with the
no-share rule made explicit (kernel arithmetic/dispatch/f16 mirrors stay
per-backend for bit contracts); core helpers D4 (hc_mix inject-optional),
D6 (PLE gate stashed once — inputs proven identical), D7 (grouped_rms),
D2 (qwen35 attention head math shared by main loop and GPU-failure
fallback, exp_cr selection preserved); `qsa_idx_append_host` default
flipped Ok(())→Err (silent non-append was a lie that produced watermark
holes); qwen35 layers migrated to the stages/Ctx pattern; PLE_CHECK
shadow and G0/A3 dumps isolated into diag modules; MOE_GROUPED retired,
VKD_BATCH spec-opt-in renamed VKD_SPEC_BATCH, frame env gate cached via
OnceLock (2 env reads/decode step removed).

**Verification**: all gates green on the final tree — FN vk gate 3×
byte-identical, 27B hip gate PASS, vk-frame-check PASS, cargo test full
suite. One intermittent mid-stream divergence (1 in 11 consecutive FN
runs, same binary, under back-to-back model-load heat/page-cache
pressure; 10/11 identical including 5 reruns after cooldown, main 3/3)
was investigated to ground: the A2 buffer-reuse was cleared by shader
analysis (full rewrite per launch), leaving thermal/UMA-state
nondeterminism as the standing hypothesis — recorded here because the
atomicAdd permutation order in fn_moe_group is nondeterministic **by
design** (output order-independence argument documented in the shader
header and common/moe.rs). D8/D9 (silu_mul_rows/embd_rows) were skipped:
the scout-era duplication no longer exists (single occurrence each; a
helper would be indirection without dedup). plans/91 consumes the Slot
table directly (P1a) and the checks/vkacc split (P2).

**Flake follow-up (post-merge audit)**: the 1-in-11 divergence was
investigated to ground: 12-run back-to-back stress with
`LLM170_DUMP=checksum` (897 [npck] stage marks per run — one-run
localization capability verified for any recurrence; harness kept at
`scripts/stress-flake-repro.sh`) plus 11 earlier re-runs = 23 consecutive
byte-identical runs post-event. Audit cleared every suspect: `off` is
write-only on the tile path; tile kernels read only fully-rewritten
tables; the kv-append no-barrier group is closed by a single barrier
whose first scope covers both writes; the default q51_sg1 tile is
single-workgroup barrier-synchronized (deterministic MMA per element).
Two hardenings landed: the vk MoE tile gate now carries the hip-side
`ne <= 512` guard (the group shader's shared-array cap silently skips
table writes above it — unreachable with 512-expert models, but stale
buffer consumption if ever reached), and the flake remains classified
environment-correlated until a recurrence provides checksums.

**A3 NAMES regression (caught post-merge)**: the hip NAMES table rewrite
intended to drop 12 probe kernels but silently dropped 34 — including
production kernels (`q4_shexp_gu/da` FN decode GEMVs, `argmax64`, the
`gemm_*_{mm,wm,wc}` families, vit/ms kernels). NAMES is a string table,
so the compiler cannot catch omissions; failures surface at runtime
GetFunction. The 27B hip gate stayed green by luck (its CPU argmax
fallback is deterministically stream-identical) while FN hip diverged
deterministically — caught only when the FN hip cell was run for the
first time this session, during final both-runtime gating of the
follow-up cleanup. Bisect initially mispointed (a stale-binary hazard:
`cargo build | tail -1 && gate` masks build failures); strict
build-verified bisect plus launch-site audit localized it. All 23
restored; FN hip + 27B hip green again. Protocol change going forward:
the gate matrix must include **FN hip** alongside FN vk / 27B hip /
27B vk / frame-check — a single-runtime gate can mask whole-backend
kernel loss. Verification commands now run with explicit build-exit
checks (`set -o pipefail` discipline).

## (34) plans/91 ledger: vk np4 batch decode, MTP port, MMQ investigation (2026-09-23)

Scope: P0 np4 single-pass decode, P2 MTP prefill/verify batching, P1a/P1b
scalar-MMQ and double-buffer attempts (closed negative with a structural
finding), P3 FN tg re-profile, D1 evaluated. All commits on `perf-91`;
gates green throughout: 27B vk/hip, FN vk ×3 byte-identical, FN hip,
frame-check 0 failures, cargo test full suite, verify_np_self 4/4 identical
(batch == sequential bit-stream parity over 16 tokens × 4 slots).

**P0 np4 (commit 88cda21).** `step_batch_np_ex`: t rows (one per slot) in a
single pass — GEMM/elementwise kernels share t rows (one weight read), state
kernels (conv ring / AR / rope / KV / flash) address per-row state through
GL_EXT_buffer_reference device-address tables built once at init. Five new
shaders (gdn_conv_np, gdn_arf_np, qk_rope2_np, kv_app_np, qsa_flash_np);
per-row arithmetic is bit-identical to the t=1 path. Root-caused during
bring-up: the slot-map upload reinterpreted the usize array's low bytes
([0,0] written instead of [0,1]) so every row overwrote slot 0's state —
localized via [npck] state marks (slot-1 state frozen across steps), fixed
by element-wise conversion. The second structural find: the old gemv8 z=t
dispatch re-read the full weight tensor per token (np4 step 273 ms);
a new `gemv8t` family (7 kernels, q5/q4/q6/q8/xs/nl/q3) processes 2..=4
tokens per workgroup with vec4 (lane=token) accumulate — weight read once,
WG count 1/t, row×token arithmetic bit-identical to the *_b variants
(q8/xs dot-lowering order empirically stream-verified). Greedy head folds
fn_argmax_rows into the trunk submission. 27B (ctx 4096): np4 greedy cell
11.43 → **28.69 t/s (2.5×; hip same cell today 20.12 — vk +43%)**, np4-tg128
10.61 → 27.10 (hip 33.83: remaining gap = 2.4 MB logits transfer + the
gemv8-class ~145 GB/s ceiling — further gains are class-pinned by the
batch≡sequential parity contract). NR sweep for the t-kernels: 4 invalid
(hardcoded 2-row kernels skip half the outputs), kept at 2.

**P1 MMQ (commit ff0bbdf, negative).** The plan's scalar OpSDot dense tile
was built three ways: MoE-tile mode=1 reuse (fixed a real mode=1 rp guard
bug in fn_moe_tile_q5k/q8 — they read rows_pad[0] and early-exited on the
dummy buffer), a register-blocked element-wise kernel, and an 8-token
OpSDot kernel with shared activation staging (best: pp512 283 vs 342
baseline). All lose to the f16 coopmat tiles. The structural reason,
measured: the coopmat tiles run at **17.7 GFLOP/s and ~11.5 GB/s effective
weight bandwidth — latency-bound at ~0.1% of ALU and ~5% of memory
capability**. Arithmetic-density optimization cannot pay when the kernel
is neither ALU- nor BW-bound. tile_ms128 K double-buffering (P1b) also
closed: GLSL compute has no async copies, per-thread stores are in-order,
so the "overlap" only halved occupancy (LDS 2×) — 333 vs 342. The [ts]
profiler gained a direct inter-dispatch gap sum (checked_sub) — pp512
chunks show ~2.4 ms of gap per 2757 dispatches, confirming GPU-bound.
The open lever for pp16384 (145.4 t/s today) is tile *structure*
(occupancy/latency), not arithmetic.

**P2 MTP (commit 6bf445d).** mtp_prefill_batch ported: blk.64 in one t-row
pass with hip's KV-only optimization (mid chunks accumulate KV only; the
final chunk runs attention/FFN/head for the last row only), device-side
h_shift assembly (row_shift_gather — host round-trip eliminated), cat2_rows,
7 new T_MAX-row m_b_* buffers, mtp_upload_tok_emb prefetch via mapped
pointer (sync — hip's side-stream overlap not ported). verify_rows flipped
to batch-by-default (step_batch t-row; kill switch VKD_SPEC_BATCH=0) —
the old per-token default made spec prefill 6.0 s for 64 tokens (677 ms
after). mtp_step_g GEMVs upgraded gemv3→gemv_w(gemv8). gdn_restore_seq
implemented (np×spec partial-accept). bench --spec 2 now completes on vk
(was "미지원" instant death): acceptance 0.94 = hip's 0.94 on the same lcg
prompts (draft parity); pp64 spec2 94.5 vs hip 175.3. The plan's 14.4 MTP
cell is unreachable on this bench protocol for *either* backend (hip spec
= 6.16 t/s today vs hip plain tg 11.5 — acceptance ~1 means spec pays the
verify+draft cost without multi-token wins).

**P3 FN tg.** Fresh [ts]: the FN decode step is host-bound — ~2–3 ms GPU
per 56 ms wall (gaps ≈ 0). The plan's NUM_ROWS sweep re-run: 13.50/14.00/
13.12 (ctx 8192) — noise, confirming the prior session's finding.
mm_f32b consolidation (288 launches/step) remains the designated lever but
targets GPU time that is not the bound; the actual lever is host dispatch
cost. Same-protocol comparison: FN tg128@8k vk **15.01 vs hip 5.47**
(208-token prompt) — vk leads hip 2.7×; the 18.43 bar traces to a
different/older protocol (current hip cannot reach it either).

**P5.** D1 (GDN layer unification) evaluated, not executed: the two layer
files have diverged into different GPU-hook architectures (value-style
per-stage hooks vs t=1 shortcut), state types, and weight accessors; a
parameterized core at the current divergence is indirection without dedup
— same verdict as 90's D8/D9. The plan itself scoped D1 to a relaxed
session. checks.rs/spec.rs splits (P5.4, optional) untouched — no natural
split presented itself.

**Final cells (this tree, back-to-back load, ctx per benchmarks.md).**
27B vk: pp512 343.4 / pp4096 338-class / pp16384 145.4 / tg128@4k 11.21 /
np4 greedy 28.69 / np4 agg 27.10 / spec2 runs (acceptance parity with hip).
FN vk: pp512 178.2 / pp4096 169.4 / tg128@8k 13.11(512-tok prompt)
15.01(208-tok). hip same-session references: 27B np4 greedy 20.12,
np4-tg128 33.83, spec tg 6.16, FN tg 5.47. Plan cells not reached on this
protocol: 27B pp16384 ≥292 (P1 closed with finding), np4 literal 32.1
(non-greedy cell; the tracked greedy cell beats hip by 43%), MTP ≥14.4
(protocol-bound for both backends), FN literal 18.43 (vk 15.01 vs hip
5.47 — hip-exceeded criterion met at 2.7×).

**P5 addendum (commit 18e9345).** D1 executed in scoped form: the genuinely
identical norm_gated core (rms_norm(core)·gate(z) per head — the plan's
"z-gate silu↔sigmoid 파라미터화") is now one implementation
(`gdn_norm::gdn_norm_gated(GdnGate)`) consumed by both layers, loop order
preserved (bit-identical, all four runtime gates byte-identical). The
remaining layer orchestration stays per-model by design (state types,
GPU-hook architectures, multi-seq layouts — unifying those is indirection
without dedup, per the 90 D8/D9 precedent).

**P3 addendum (commit 37aa084).** mm_f32b micro-burst consolidation landed
as `mm_f32b_grp` (one launch per f32/BF16 subset of a frame mm_group,
≤8 weights, per-row output via device-address table; mixed quant+f32 groups
take the f32 subset). FN tg128@8k 15.01 → 15.24 t/s; FN gate stream
unchanged (bit-identical row arithmetic). The measured correction to the
P3 diagnosis: mm_f32b's 288 launches were only ~2–4 ms of GPU time — the
step's real GPU mass is gemv8_q8b (24 ms of genuine weight reads).
