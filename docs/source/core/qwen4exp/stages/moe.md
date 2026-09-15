# `crates/core/src/qwen4exp/stages/moe.rs` — measurement record

> Items from `docs/benchmarks.md` that correspond to this file (by section title).
> Tables and numbers are verbatim. Summary metrics remain in benchmarks.md.

## Tiny-tensor routing — neutral (plans/45)

The pp64 profile shows the small q8 tensors (ssm_alpha/beta at 48 rows,
attn_k/v at 1024) costing ~108 ms of a ~306 ms pass - the tile kernel
launches a single workgroup for a 48-row tensor and pays the full
K-iteration latency for 0.26MB of weights. Routing everything with
n_out <= 128 through the row-parallel gemv8 kernel is token-identical but
measures neutral on all three metrics (pp64 210-214, pp512 300-301,
tg8 9.94): the gemv8 path pays for the same latency through t-fold weight
re-reads. Both paths are latency-bound for these shapes; a real fix needs
all ninety-six tiny tensors in one dispatch (multi-tensor batching).

