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
