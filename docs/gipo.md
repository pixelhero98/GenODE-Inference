# GIPO

This path trains a continuous density policy instead of choosing a measured
schedule key as the deployed action. The active protocol is
`gipo_density_v1`.

Measured fixed/SER schedules are supervision candidates only. Every schedule grid
is converted into canonical 64-bin `density_mass_v1` on normalized model time:

```text
reference edges: linspace(0, 1, 65)
p[j] = integral over reference bin j of the equal-step schedule density
sum_j p[j] = 1
```

GIPO uses strict context-only conditioning for teacher and student. The
conditioning inputs are solver/NFE features plus the frozen context embedding:

```text
z_inputs = concat(setting_features(solver, NFE), frozen_context_embedding)
z = additive_mlp_v1(z_inputs)
```

The canonical teacher is `density_form_transformer_v1`. It turns each measured
candidate density into ordered bin tokens:

```text
[log(rho_j + eps), normalized_log_density_j, t_j, delta_t_j]
```

The condition embedding is added to every density-bin token before each
RoPE-enabled self-attention block through the strict `additive_mlp_v1` path. The
teacher returns a metric utility vector. The default forecast target columns are
`u_crps_uniform,u_mase_uniform`, but the trainer accepts arbitrary utility
columns through `--teacher_metric_target_keys`. The only convention is that
larger utility means better downstream performance.

Rewards are paired inside exact `(dataset, solver, NFE, context_id, seed)` cells.
Rows must not cross solver, NFE, seed, series, target time, or context identity.
BO/candidate schedules are rejected in this path; measured fixed schedules and
SER are the default forecast supervision references.

The teacher is trained with pairwise rank loss over the scalarized utility plus
auxiliary Huber regression on the metric vector. The canonical public selector is
`weighted_normalized_regret_v1`: it normalizes soft regret across checkpoints
for context, density-family, and unseen-NFE calibration diagnostics, scores
`J_CDN = 0.5 * J_CD + 0.5 * unseen_nfe`, and tie-breaks by minimax normalized
regret, raw mean regret, then earlier checkpoint step. Locked-test data is not
used for checkpoint selection.
Series IDs are still used for row grouping, split diagnostics, series-disjoint
diagnostics, and reporting, but no series identity feature is fed to the GIPO
teacher or student.

The canonical student is `density_query_transformer_v1`. It builds one query
token per density bin from `(t_j, delta_t_j)`, applies the same additive
context-only conditioning, performs RoPE-enabled bin self-attention, emits one
logit per bin, and normalizes with softmax. Student targets are
teacher-weighted mixtures of the measured candidate densities in each context
group:

```text
w_i = softmax(teacher_utility_i / temperature)
target_density = sum_i w_i * density_mass_i
loss = KL(target_density || student_density)
```

The canonical student selector is `validation_ce_v1`: a selector pass uses a
context-disjoint calibration validation split, then the final student is
retrained from scratch on all eligible seen calibration rows for the selected
step.

At deployment, the density is converted to a solver grid by inverse CDF at
quantiles `0/K, 1/K, ..., K/K`, where `K` is the solver macro-step count for the
requested solver/NFE cell.

## Calibration Pool Size

Do not expand the full train+validation pool into all per-context schedule rows.
Use context-stratified sampling and reuse the same sampled contexts for every
schedule and solver/NFE cell.

Default for the current solar-style canonical runs:

```text
256 sampled calibration contexts for the trainer
64 density bins
student selector budget: 1000 steps
```

Locked seen/unseen panels are generated separately for reporting and use fixed
locked windows per solver/seed/NFE. They are complete over the generated locked
panel, not an exhaustive pass over every possible raw dataset window.

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
  --gipo_conditioning_style additive_mlp_v1 \
  --density_bin_count 64 \
  --out_dir <output-dir>
```

For non-forecast tasks, provide precomputed utility columns and weights:

```text
genode-train-gipo \
  --rows_csv <rows-with-utility-columns.csv> \
  --context_embeddings_npz <context-embeddings.npz> \
  --out_dir <output-dir> \
  --gipo_conditioning_style additive_mlp_v1 \
  --density_bin_count 64 \
  --teacher_metric_target_keys u_accuracy_gain,u_latency_gain \
  --teacher_utility_weights u_accuracy_gain=0.8,u_latency_gain=0.2
```

Canonical verification scripts train a single 64-bin additive policy and run
both seen-NFE locked and unseen-NFE locked student reporting without using
locked-test rows for selection. AdaLN remains an explicit noncanonical sidecar
conditioning style, never an implicit canonical replacement.

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

Teacher-oracle reporting evaluates teacher-weighted density targets directly and
remains available for local diagnostics, but it is not part of the final
canonical student-policy stream:

```text
genode-report-gipo-teacher-oracle \
  --gipo_teacher_checkpoint <output-dir>/gipo_teacher.pt \
  --training_summary <output-dir>/gipo_training_summary.json \
  --support_rows <locked-fixed-context.csv>,<locked-ser-context.csv> \
  --context_embeddings_npz <locked-context-embeddings.npz> \
  --out_dir <teacher-oracle-report-dir>
```

Context embeddings must come from the frozen backbone's historical context
encoder under eval/no-grad mode. Store embeddings in the NPZ sidecar rather than
duplicating vectors in metric CSV rows.
