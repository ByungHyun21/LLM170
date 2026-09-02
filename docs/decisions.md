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

## ADR-0002 — Mode system universal / cmp-stock / cmp-unlocked (2026-08-30)

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
only. The runtime `--mode` flag (`core::mode`, ADR-0002 made concrete)
selects memory budgets today and is the branch key for future cmp-stock
kernel variants.

**Consequences**: `layers.rs` is 335 lines; stages are backend-independent
and independently testable. Numerics unchanged (pure code motion):
GPU↔CPU 25/25 token-exact, 1,024-token chunk smoke clean, single_ko
reference-exact 24/24, cargo suite 10/10.

## ADR-0016 — Local cubecl-runtime vendor patch: memory sweep disabled (2026-09-01)

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
