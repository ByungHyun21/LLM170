# `crates/server/src/resource.rs` — pre-load resource guard (2026-09-16)

## Why it exists

An incident: with another inference server resident (Flash-Next, ~104 GiB
across VRAM+RAM), starting a second copy of the same model exhausted both pools
and froze the host (hard reboot needed). The guard turns that failure mode into
an immediate, descriptive error before any weight is touched.

## Policy

- required = total model bytes (all GGUF split parts, `-NNNNN-of-MMMMM.gguf`
  convention) x 1.10 slack for KV/activations/upload staging.
- capacity = free VRAM (`hipMemGetInfo` via `llm170_backend_gpu::gpu_mem_free`)
  + 0.85 x `MemAvailable` from `/proc/meminfo` (15% headroom for the process
  itself and page-cache variance).
- Any measurement that cannot be taken (non-Linux host, probe failure) is
  **skipped with a warning instead of failing the check** — the guard must
  never block a legitimate run on missing information.
- Kill switch: `LLM170_NO_RSRC_GUARD=1`.

## Wiring

`main()` scans argv for `--model <path>` before dispatching any subcommand
(serve / infer / vl / bench / check all take it), and treats the run as
GPU-bound when `--backend gpu` or `--gpu-runtime hip|vulkan` is present.

## Verified

- Unit tests: standalone Flash-Next passes, the double-resident incident
  numbers are refused, 27B alongside a resident model passes, unknown
  measurements pass.
- Live reproduction of the incident: with an llm170 Flash-Next server resident,
  a second Flash-Next load exits in <1s with
  `insufficient resources: model needs ~114.1 GiB ... but only 37.9 GiB
  available (VRAM 16.6 GiB, host 25.1 GiB)` and the host stays healthy.
