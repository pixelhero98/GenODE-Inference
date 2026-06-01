# genODE

genODE contains source code for context-conditional schedule selection on OTFlow forecasting backbones. The active OPD workflow trains a frozen-backbone teacher and categorical student over measured fixed/SER support schedules using per-context rows and frozen context embeddings.

## Source Layout

- `src/genode/data/`: dataset definitions, split builders, experiment plans, and project paths.
- `src/genode/models/`: OTFlow configuration, conditioning, backbone modules, training, and model utilities.
- `src/genode/schedule_transfer/`: fixed schedule grids, registries, table helpers, signal traces, and diagnostics.
- `src/genode/evaluation/`: checkpoint loading, schedule evaluation, solver mappings, and sampling helpers.
- `src/genode/conditional_opd/`: context-conditional OPD teacher/student models, SER-PTG references, fixed/SER schedule evaluation, and locked-test reporting.
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

## Context-Conditional OPD

The main public entry point is:

```bash
genode-train-context-conditional-opd
```

This path implements:

- per-example fixed/SER context rows with `series_id`, `target_t`, and stable `context_id`
- frozen context embeddings from the frozen forecast backbone
- uniform-anchored rewards paired inside exact context/seed/solver/NFE groups
- teacher checkpoint selection by context-disjoint and series-disjoint top-1/top-2 support diagnostics
- categorical student training with teacher-guided top-1/top-2 support CE
- calibration-holdout non-regression guard against the best static fixed/SER support per solver/NFE
- calibration-holdout fixed-support oracle headroom diagnostics

The calibration row/embedding artifacts are intentionally reusable. Once fixed/SER
context rows and context embeddings exist, future teacher/student changes can be
verified by rerunning only the trainer and reporter:

```bash
genode-train-context-conditional-opd \
  --rows_csv outputs/context_calibration_rows.csv \
  --context_embeddings_npz outputs/context_embeddings.npz \
  --schedule_summary_json outputs/ser_ptg_schedule_summary.json \
  --out_dir outputs/context_policy \
  --support_schedule_keys uniform,late_power_3,flowts_power_sampling,ays,gits,ots,ser_ptg_local_defect_eta005
```

Locked-test reporting is reporting-only and applies the frozen calibration guard
stored in the student checkpoint:

```bash
genode-report-context-locked-test \
  --context_student_checkpoint outputs/context_policy/context_student.pt \
  --training_summary outputs/context_policy/context_conditional_summary.json \
  --locked_context_rows outputs/locked_fixed_context_rows.csv,outputs/locked_ser_context_rows.csv \
  --locked_context_embeddings_npz outputs/locked_context_embeddings.npz \
  --out_dir outputs/context_locked_report
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
