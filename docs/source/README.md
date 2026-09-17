# docs/source — per-code-file measurement records

Each document in this directory corresponds 1:1 to a source file and holds its
measurement evidence. Split from `docs/benchmarks.md` when that file grew past
350 KB.

## Conventions

- Paths mirror the source tree exactly:
  `crates/backend-gpu/src/rawhip/q4acc/value.rs` → `backend-gpu/rawhip/q4acc/value.md`
- Each document states the source file path at the top.
- **Numbers and tables are preserved verbatim** when moved. No new summaries
  or interpretations.
- Keep dates in section titles (e.g. `(2026-09-14)`). Later measurements on
  the same topic go in as new entries; when one supersedes an earlier result,
  **state which is current** (e.g. "supersedes ...").
- Rejected experiments are kept — knowing why something does not work reduces
  the cost of the next attempt.
- Probe and flag usage (`LLM170_*`) is documented in the file that implements
  the flag.

**Organization (2026-09-15)**: `docs/benchmarks.md` now groups results by
adopted/rejected instead of chronologically (originals preserved in
`docs/archive/benchmarks-chronological-2026-09-15.md`). These per-file
records keep their detailed prose; their section titles carry dates and
outcomes so a grep for "regression"/"reverted"/"not adopted" finds every
negative result.

**Runtime default (2026-09-14)**: benchmarks and gates run on the TheRock
ROCm 10.0.0 userspace (`/opt/rocm-10.0.0/install/lib` via `LD_LIBRARY_PATH`,
soname-compatible, no relink; system 7.2.2 as fallback). Numbers quoted
before this date were measured on 7.2.2 unless noted.

## Current documents

| Document | Source | Contents |
|---|---|---|
| `backend-gpu/rawhip/q4acc/*.md` | q4acc/ (mod/value/moe/qsa) | QSA attention (split, 6-head), MoE grouping, Q4_K MMQ tiles, f16 dequant, prefill GEMM, staging/upload RCA |
| `backend-gpu/rawhip/mod.md` | mod.rs | KTRACE event pairing, launch-order dump, instrumentation notes |
| `backend-gpu/rawhip/decode/mod.md` | decode/ | 27B decode/prefill decomposition, GDN |
| `core/qwen4exp/stages/qsa.md` | stages/qsa.rs | Stage timers, cost breakdown |

## Verification gates

| Model | Command | Pass criterion |
|---|---|---|
| Flash-Next | `scripts/gate-flash-next.sh` | 208-token Korean prompt stream identical |
| 27B | `scripts/gate-27b.sh` | 208-token Korean prompt stream identical |
| Tests | `cargo test --release --workspace` | 12/12 |
| Build | `cargo build --release` | 0 warnings |
