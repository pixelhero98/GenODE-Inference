# GIPO

This path trains a continuous density policy instead of choosing a measured
schedule key as the deployed action. The active protocol is
`gipo_density_v1`.

Measured fixed/SER schedules are supervision candidates only. Every schedule grid
is converted into canonical `density_mass_v1` on normalized model time:

```text
reference edges: linspace(0, 1, 129)
p[j] = integral over reference bin j of the equal-step schedule density
sum_j p[j] = 1
```

The teacher sees `(solver, nfe, log_density, series id, frozen context
embedding)`, where `log_density = log(p / bin_width + 1e-8)` and the
log-density vector is normalized from teacher-fit rows only. The teacher target
is uniform anchored:

```text
u_comp_uniform = -0.5 * (log(crps / uniform_crps) + log(mase / uniform_mase))
```

Rewards are paired inside exact `(dataset, solver, NFE, context_id, seed)` cells.
Rows must not cross solver, NFE, seed, series, target time, or context identity.
BO/candidate schedules are rejected in this path; fixed schedules and SER are
the measured supervision references.

The teacher is trained with pairwise rank loss plus auxiliary Huber regression.
Teacher checkpoints are selected with context-disjoint and series-disjoint
calibration diagnostics. Locked-test data is not used for checkpoint selection.

The student predicts a `density_mass` vector from `(solver, nfe, series id,
frozen context embedding)`. Student targets are teacher-weighted mixtures of the
measured fixed/SER candidate densities in each context group:

```text
w_i = softmax(teacher_utility_i / temperature)
target_density = sum_i w_i * density_mass_i
loss = KL(target_density || student_density)
```

At deployment, the density is converted to a solver grid by inverse CDF at
quantiles `0/K, 1/K, ..., K/K`, where `K` is the solver macro-step count for the
requested solver/NFE cell.

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

With 12 solver/NFE cells, 7 fixed/SER schedules, and 3 seeds, this is about 30k
context-cell evaluations. If runtime is tight, use 72 total contexts with at
least 18 holdout contexts. If diagnostics are noisy, scale the sampled context
pool explicitly, for example `--context_sample_count 288`, before changing the
optimization method.

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

For SER or another precomputed fixed/SER schedule summary, use the schedule
summary evaluator with context rows enabled:

```text
genode-evaluate-schedule-summary \
  --schedule_summary <ser-schedule-summary.json> \
  --split_phase train_tuning \
  --write_context_rows
```

Then train the density policy:

```text
genode-train-gipo \
  --rows_csv <forecast_context_rows.csv> \
  --context_embeddings_npz <context-embeddings.npz> \
  --schedule_summary_json <ser-schedule-summary.json> \
  --out_dir <output-dir>
```

Locked-test reporting is reporting-only. It applies the frozen student to each
locked-test context, evaluates the generated context-specific grid, and writes
aggregate rows for comparison:

```text
genode-report-gipo-locked-test \
  --gipo_student_checkpoint <output-dir>/gipo_student.pt \
  --training_summary <output-dir>/gipo_training_summary.json \
  --context_rows <locked-fixed-context.csv>,<locked-ser-context.csv> \
  --context_embeddings_npz <locked-context-embeddings.npz> \
  --out_dir <locked-report-dir>
```

Context embeddings must come from the frozen backbone's historical context
encoder under eval/no-grad mode. Store embeddings in the NPZ sidecar rather than
duplicating vectors in metric CSV rows.
