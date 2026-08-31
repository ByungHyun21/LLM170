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
