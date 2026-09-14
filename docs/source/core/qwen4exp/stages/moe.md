# `crates/core/src/qwen4exp/stages/moe.rs` — 측정 기록

> `docs/benchmarks.md`에서 **이 파일에 해당하는 항목만** 옮긴 것(제목 기준).
> 표·수치는 원문 그대로. 요약 지표는 benchmarks.md에 남는다.

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

