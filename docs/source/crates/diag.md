# crates/diag — Shared Diagnostics Layer

Pure-Rust diagnostics crate (GPU-free) shared across all engine crates.
Backend adapters (HIP/Vulkan) supply **resolved events** (timing already
computed); this crate handles representation, storage, analysis, and output.

## Dependency direction

```
diag ← core, gguf, backend-gpu, server
```

`diag` has zero GPU dependencies and zero external crate dependencies.
Everything is unit-testable without a GPU.

## Modules

### `trace` — Resolved event capture store

- `Ev { name, lane, dur_ms, gap_next_ms, seq_ms }` — completed events only.
- `capture_begin()` / `capture_end()` / `capture_on()` — AtomicBool gate.
  When off, push cost = 1 atomic load (no lock).
- `push(Ev)` — appends if capturing (200k cap, overflow counted).
- `take()` — consumes + computes sequential timestamps and gaps.
- `summarize()` / `gap_by_pred()` — aggregation for table output.

### `writer` — Text table output

- `table(evs, dropped)` — KTRACE-compatible kernel summary + gap analysis.
- `seq_dump(evs, max)` — sequential event dump (`LLM170_KTRACE_SEQ=N`).
- Format pinned by unit tests.

### `fp` — Stage fingerprint recording and comparison

The highest-value debugging tool: records a FNV-1a 64-bit hash + max-abs +
first-non-finite index for each processing stage's output buffer, then
compares two runs to find the **first divergent stage** automatically.

- `fp_record(stage, &[f32])` — hash + summary for one stage.
- `fp_diff(path_a, path_b) -> DiffReport` — first mismatch + full list.
- Gate: `LLM170_FP_FILE=<path>` (unset = zero cost, 1 atomic load).
- Subcommand: `llm170 diag diff <A> <B>` — prints report, exit 1 on divergence.

**Proven value**: during the chunk-invariance investigation (plans/80 §A),
manual localization of the PLE n-gram lookback bug took ~3 hours with ad-hoc
checksum dumps. With `fp_diff`, the same localization takes 1 minute: run
two configurations, diff the fingerprints, read the first mismatch stage.

### `flag` — Cached environment variable reads

- `env_on(name)` — cached existence check + registry auto-enrollment.
- `env_on_desc(name, desc)` — register with documentation.
- `list()` — all registered flags for `llm170 diag env`.

## Usage

```sh
# Record fingerprints for two chunk sizes:
LLM170_FP_FILE=/tmp/fp_512.txt llm170 infer --model <m> --prompt-tokens <p> \
    --n-predict 1 --backend cpu
LLM170_FP_FILE=/tmp/fp_16.txt LLM170_Q4_CHUNK=16 llm170 infer --model <m> \
    --prompt-tokens <p> --n-predict 1 --backend cpu

# Compare:
llm170 diag diff /tmp/fp_512.txt /tmp/fp_16.txt
```

## Hooks (current)

- `qwen4exp::layers::forward_timed` — `L{il}.ffn_out` per layer per step
  (48 layers × 2 steps = 96 stages for Flash-Next).

More hooks (qwen35 layer boundaries, GDN conv/AR I/O) to be added as needed.
