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

`main()` scans argv before dispatching any subcommand and covers every
model-loading entry:
- `--model <v>` and `--model=<v>` (serve / infer / vl / bench),
- the `check` subcommand's positional model path (its backend defaults to
  GPU, so it is treated as GPU-bound even without flags),
- `--backend gpu`, `--backend=gpu`, and any `--gpu-runtime*` form all mark
  the run GPU-bound.

Verified live with a Flash-Next server resident (VRAM 16.6 GiB free): both
`llm170 check <FN> --quick` and `llm170 infer --model <FN> --backend=gpu`
refuse in <1s; `LLM170_NO_RSRC_GUARD=1` bypasses. Note the guard prevents the
catastrophic OOM freeze, not performance degradation from contention — run
benchmarks with the GPU solo.

## Verified

- Unit tests: standalone Flash-Next passes, the double-resident incident
  numbers are refused, 27B alongside a resident model passes, unknown
  measurements pass.
- Live reproduction of the incident: with an llm170 Flash-Next server resident,
  a second Flash-Next load exits in <1s with
  `insufficient resources: model needs ~114.1 GiB ... but only 37.9 GiB
  available (VRAM 16.6 GiB, host 25.1 GiB)` and the host stays healthy.
