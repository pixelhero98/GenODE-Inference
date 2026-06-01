# Context-Conditional OPD

This path trains schedule selection as a per-context support policy instead of a
global solver/NFE schedule. The teacher sees `(solver, nfe, support interval,
series id, frozen context embedding)` and learns uniform-anchored utility. The
student predicts a categorical support choice over measured fixed/SER schedules
from `(solver, nfe, series id, frozen context embedding)`.

The reward target is paired within the same context:

```text
u_comp_uniform = -0.5 * (log(crps / uniform_crps) + log(mase / uniform_mase))
```

Rows must not cross solver, NFE, seed, series, target time, or context identity
when constructing rewards. BO/candidate schedules are intentionally rejected in
this path; fixed and SER schedules are the measured support.

Teacher checkpoints are selected by support-choice diagnostics on both
context-disjoint and series-disjoint calibration holdouts, not by scalar loss
alone. The selected checkpoint must pass pairwise and Spearman sanity checks,
then maximize the mean of context/series top-1 support accuracy and top-2
support recall.

The student is trained with teacher-guided top-1/top-2 categorical CE over the
measured fixed/SER support. Observed uniform-anchored utility on teacher-fit rows
rejects clear teacher top-1 mistakes; close top-1/top-2 cases use a 0.6/0.4
target. Series-unknown augmentation is sampled dynamically during student
optimization.

Final deployment is guarded per `(solver, NFE)` using calibration holdout only.
If the context student does not beat the best static fixed/SER support schedule
by the configured margin, the frozen policy falls back to that static support
for the cell. Locked-test reporting only applies the frozen guard table and must
not construct or alter guard decisions from locked-test metrics.

## Calibration Pool Size

Do not expand the full train+validation pool into all per-context schedule rows.
Use context-stratified sampling and reuse the same sampled contexts for every
schedule and solver/NFE cell.

Default for solar-style runs:

```text
120 contexts total per calibration seed
96 teacher-train contexts
24 context-holdout contexts
```

With 12 solver/NFE cells, 7 fixed/SER schedules, and 3 seeds, this is about
30k context-cell evaluations. If runtime is tight, use 72 total contexts with at
least 18 holdout contexts. If diagnostics are noisy, scale the sampled context
pool explicitly, for example `--context_sample_count 288`, before changing the
optimization method.

The calibration guard also records fixed-support oracle headroom on the same
holdout rows:

```text
oracle_context = best fixed/SER support per sample/context/cell
best_static = best static fixed/SER support per solver/NFE cell
headroom = mean(oracle_context utility - best_static utility)
```

Small headroom means the measured support itself offers little per-context gain.
Large headroom with a guarded fallback means the selector is leaving useful
support signal unrecovered.

## Entry Point

First generate per-example fixed/SER rows and the context embedding sidecar from
the forecast schedule runner:

```text
genode-run-schedules \
  --split_phase train_tuning \
  --write_forecast_context_rows \
  --baseline_scheduler_names uniform,late_power_3,flowts_power_sampling,ays,gits,ots
```

The runner writes `forecast_context_rows.csv` and
`forecast_context_embeddings.npz` under the output root.

For SER or another precomputed fixed-support schedule summary, use the schedule
summary evaluator with context rows enabled:

```text
genode-evaluate-schedule-summary \
  --schedule_summary <ser-schedule-summary.json> \
  --split_phase train_tuning \
  --write_context_rows
```

```text
genode-train-context-conditional-opd \
  --rows_csv <forecast_context_rows.csv> \
  --context_embeddings_npz <context-embeddings.npz> \
  --schedule_summary_json <ser-schedule-summary.json> \
  --out_dir <output-dir>
```

Context embeddings must come from the frozen backbone's historical context
encoder under eval/no-grad mode. Store embeddings in the NPZ sidecar rather than
duplicating vectors in metric CSV rows.
