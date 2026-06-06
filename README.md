# genODE

genODE contains source code for GIPO schedule optimization on OTFlow forecasting backbones. The active workflow trains a frozen-backbone metric-vector teacher and a continuous-density student using per-context measured schedule rows and frozen context embeddings.

## Source Layout

- `src/genode/data/`: dataset definitions, split builders, experiment plans, and project paths.
- `src/genode/models/`: OTFlow configuration, conditioning, backbone modules, training, and model utilities.
- `src/genode/schedule_transfer/`: fixed schedule grids, registries, table helpers, signal traces, and diagnostics.
- `src/genode/evaluation/`: checkpoint loading, schedule evaluation, solver mappings, and sampling helpers.
- `src/genode/gipo/`: GIPO teacher/student models, SER-PTG references, fixed/SER schedule evaluation, and locked-test reporting.
- `scripts/`: thin command-line wrappers for packaged entry points.

## Installation

Install in editable mode from a Python 3.11+ environment:

```bash
python -m pip install -e .
```

Or install dependencies directly first:

```bash
python -m pip install -r requirements.txt
```

Large local assets are intentionally not committed. Runs may use `data/`, `paper_datasets/`, `.venv/`, and `outputs/`; those paths are ignored by git.

Generated outputs default to:

```text
outputs/
```

The default backbone manifest path is:

```text
outputs/backbone_matrix/backbone_manifest.json
```

## GIPO

The main public entry point is:

```bash
genode-train-gipo
```

This path implements:

- per-example fixed/SER context rows with `series_id`, `target_t`, and stable `context_id`
- frozen context embeddings from the frozen forecast backbone
- configurable teacher utility columns paired inside exact context/seed/solver/NFE groups
- canonical `density_mass_v1` schedules over normalized model time `[0, 1]`
- a `density_form_transformer_v1` teacher that attends over density-bin tokens
- a `density_query_transformer_v1` student that emits one density logit per bin query
- AdaLN-Zero conditioning from solver/NFE features, series hash-Fourier features, and frozen context embeddings
- RoPE-style positional rotation in density-bin self-attention
- rank+Huber teacher training over measured density candidates
- continuous student density training by teacher-weighted MLE/KL targets, with optional pseudo-NFE student targets
- context-disjoint and series-disjoint diagnostics without locked-test selection

The calibration row/embedding artifacts are intentionally reusable. Once fixed/SER
context rows and context embeddings exist, future teacher/student changes can be
verified by rerunning only the trainer and reporter:

```bash
genode-train-gipo \
  --rows_csv outputs/context_calibration_rows.csv \
  --context_embeddings_npz outputs/context_embeddings.npz \
  --schedule_summary_json outputs/ser_ptg_schedule_summary.json \
  --out_dir outputs/gipo_policy \
  --support_schedule_keys uniform,late_power_3,flowts_power_sampling,ays,gits,ots,ser_ptg_local_defect_eta005
```

Forecast examples can use the default uniform-anchored utility columns
`u_crps_uniform,u_mase_uniform`. Other tasks can provide their own utility
columns with `--teacher_metric_target_keys` and `--teacher_utility_weights`;
GIPO only requires that larger utility means better downstream performance.

Locked-test reporting is reporting-only. It applies the frozen density student to
each locked-test context, derives a time grid by inverse CDF, evaluates that
grid, and never changes teacher checkpoints, student weights, or density
metadata:

```bash
genode-report-gipo-locked-test \
  --gipo_student_checkpoint outputs/gipo_policy/gipo_student.pt \
  --training_summary outputs/gipo_policy/gipo_training_summary.json \
  --context_rows outputs/locked_fixed_context_rows.csv,outputs/locked_ser_context_rows.csv \
  --context_embeddings_npz outputs/locked_context_embeddings.npz \
  --out_dir outputs/gipo_locked_report
```

Teacher-oracle reporting is available for diagnosing the teacher/student gap:

```bash
genode-report-gipo-teacher-oracle \
  --gipo_teacher_checkpoint outputs/gipo_policy/gipo_teacher.pt \
  --training_summary outputs/gipo_policy/gipo_training_summary.json \
  --support_rows outputs/locked_fixed_context_rows.csv,outputs/locked_ser_context_rows.csv \
  --context_embeddings_npz outputs/locked_context_embeddings.npz \
  --out_dir outputs/gipo_teacher_oracle_report
```

To generate reusable fixed/SER context rows, run schedule evaluation with context
row writing enabled:

```bash
genode-run-schedules \
  --forecast_datasets solar_energy_10m \
  --split_phase train_tuning \
  --baseline_scheduler_names uniform,late_power_3,flowts_power_sampling,ays,gits,ots \
  --write_forecast_context_rows \
  --device auto
```

For a larger context pool, pass an explicit trainer sample count such as
`--context_sample_count 288` after generating enough per-context rows. `--device
auto` uses CUDA when PyTorch reports CUDA availability and CPU otherwise.

## Other Entry Points

```bash
genode-train-backbone --dataset san_francisco_traffic --device auto
genode-run-schedules --forecast_datasets san_francisco_traffic --conditional_generation_datasets '' --split_phase validation_tuning --device auto
genode-build-ser-ptg-reference --dataset san_francisco_traffic --device auto
genode-evaluate-schedule-summary --schedule_summary outputs/example/ser_schedule_summary.json --split_phase validation_tuning --write_context_rows --device auto
```

## Development Checks

```bash
PYTHONDONTWRITEBYTECODE=1 python -m compileall -q src tests scripts
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_*.py'
```

For CPU-only smoke checks, set `CUDA_VISIBLE_DEVICES=''` and keep `--device auto`.
