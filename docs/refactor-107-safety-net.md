# Refactor safety net (plans/107 W10)

Local verification assets guarding the repository-wide refactor. Single source
of truth for "did stage-level behavior change?" during mass deletion/moves.

## Stage characterization hashes — `scripts/charhash.sh`

- `capture [hip|vulkan]`: records per-stage golden hashes for the Flash-Next
  fixed 208-token prompt (same prompt as `gate-flash-next.sh`, n-predict 16).
- `verify [hip|vulkan]`: re-runs and diffs against the golden file. On mismatch
  the first diverging stage line is printed, so a regression localizes to
  `L{il}.mids`-style stage granularity instead of final tokens.
- Lines captured: `[npbh]` stage bufhash (FNV over f32 bits — exact) and
  `[npck]` stage checksums (sum/v0/mid0/last0 with the numeric field masked to
  avoid last-digit rounding wobble).
- Golden files: `scripts/.charhash-flash.txt` (hip) / `-vk` (vulkan). Re-capture
  only when an intentional arithmetic change is approved (drift-gate rule).

## Preflight — `scripts/preflight.sh`

Run before every commit series: rustfmt check, clippy `-D warnings`,
cargo warnings 0, spv freshness (any `.comp` newer than its `.spv` fails —
prevents the stale-spv incident class), and the characterization verify.
`SKIP_CHARHASH=1` skips the GPU stage when the device is busy.

## Gate relationship

- Token gates (`gate-flash-next.sh`, `gate-27b.sh`) prove end-to-end
  bit-identity; charhash proves stage-level identity; preflight is the
  commit-time umbrella. All three must pass on default paths after each
  refactor step unless the plan explicitly re-baselines.
