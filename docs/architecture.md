# Architecture

Design under the **pure Rust** principle. Status: policies fixed; GPU backend
implemented and verified on both HIP/ROCm and Vulkan — quantized GEMM, GDN
AR/chunked, grouped MoE, element-wise kernels, and a GPU-resident decode
frame for qwen4exp (see [decisions.md](decisions.md) ADR-0009/0011/0017).
Backend internals live in [backend-architecture.md](backend-architecture.md), the
vk/hip kernel parity table in [parity-matrix.md](parity-matrix.md);
runtime flags in [configuration.md](configuration.md) (auto-generated
catalog).

## Data Flow

```mermaid
flowchart LR
    A[GGUF loader\nsplit, metadata, quant blocks] --> B[Model graph\nqwen35/qwen4exp]
    B --> C[Scheduler\nslots, ubatch, GDN state, offload]
    C --> D[matmul dispatch\nAccelerator trait]
    D --> E1[CPU\npure Rust reference]
    D --> E2[GPU\nrawhip (HIP) / rawvk (Vulkan)]
    C --> F[Sampler + MTP draft]
    F --> G[CLI / server]
    P[diag module] -.->|dump/flags/alloc ledger| C
```

## Crate layout

| Crate | Role |
|---|---|
| `llm170-gguf` | GGUF reader (split files, metadata, quant block views) |
| `llm170-diag` | Diagnostics: `LLM170_DUMP` key space, flags snapshot, alloc ledger, watchdog, ktrace |
| `llm170-core` | Model graphs (qwen35, qwen4exp), quant reference kernels, `matmul` module (traits / dispatch / cpu / raw), sampler, MTP spec |
| `llm170-backend-gpu` | `rawhip` (HIP kernels + q4acc frame) and `rawvk` (VkDecoder + VkAcc, SPIR-V assets) |
| `llm170-server` | CLI (`infer/serve/bench/check/probes`), HTTP server, continuous-batching slot loop |

## Compute paths

- **CPU** (`--backend cpu`): pure-Rust reference. `LLM170_W4A8=1` switches
  decode matmuls to the integer mirror path (bit-identical to GPU value
  paths for the supported types).
- **HIP** (`--gpu-runtime hip`, default for GPU): `rawhip` — full pipeline.
- **Vulkan** (`--gpu-runtime vulkan`): `rawvk` VkDecoder for qwen35 and
  VkAcc value path for qwen4exp. **qwen35+Vulkan currently falls back to
  hip with a loud error** (known nondeterminism, decisions.md ledger 87/90/92).

## Scheduling & memory

- Continuous batching: N slots (`serve --slots N`), prefix-cache reuse,
  greedy/sampler per slot, MTP speculative on qwen35.
- Memory: host-mapped device buffers (HOST_COHERENT), mmap'd weights with
  paged staging; GDN state snapshotted device-side for spec rollback.
- Offload policy: RAM-first staging with SSD spill guarded by the alloc
  ledger (`LLM170_DUMP=alloc`).

## Verification model

Every behavior-affecting change is gated on:

1. **Token gates** — `scripts/gate-27b.sh`, `scripts/gate-flash-next.sh`
   (bit-identical greedy streams on fixed prompts).
2. **Stage hashes** — `scripts/charhash.sh` (15,674-line golden bufhash
   set; localizes divergence to a pipeline stage).
3. **Preflight** — `scripts/preflight.sh` (fmt, clippy `-D warnings`,
   warnings 0, spv manifest, charhash).

See [decisions.md](decisions.md) for the full ADR/ledger series (86–97 cover
the refactor-107 campaign) and [benchmarks.md](benchmarks.md) for measured
performance.
