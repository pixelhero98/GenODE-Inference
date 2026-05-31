# genODE

genODE contains source code for schedule optimization on OTFlow forecasting backbones. The active Train20 workflow is V4.3 pooled calibration: fixed schedules, SER-PTG, and BO candidates are all re-evaluated on one calibration distribution before training a ranking-first teacher and a SER-initialized student.

## Source Layout

- `src/genode/data/`: dataset definitions, split builders, experiment plans, and project paths.
- `src/genode/models/`: OTFlow configuration, conditioning, backbone modules, training, and model utilities.
- `src/genode/schedule_transfer/`: fixed schedule grids, registries, table helpers, signal traces, and diagnostics.
- `src/genode/evaluation/`: checkpoint loading, schedule evaluation, solver mappings, and sampling helpers.
- `src/genode/conditional_opd/`: conditional OPD teacher/student models, pooled calibration, SER-PTG references, and BO candidates.
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

## V4.3 Pooled Calibration

The main public entry point is:

```bash
genode-run-train20-v43-pooled-calibration
```

It implements:

- calibration pool `C = 20% of (train + validation)`, sampled proportionally from each split
- calibration seeds `0,1` by default
- student and locked-test seeds `0,1,2` by default
- fixed references `uniform,late_power_3,flowts_power_sampling,ays,gits,ots`
- SER-PTG teacher anchor and student initialization
- one 32-family BO candidate buffer
- teacher utility selection under a hard geometry guard

Dry run:

```bash
genode-run-train20-v43-pooled-calibration --skip_locked_test
```

Full run:

```bash
genode-run-train20-v43-pooled-calibration --allow_execute
```

Useful explicit settings:

```bash
genode-run-train20-v43-pooled-calibration \
  --dataset san_francisco_traffic \
  --solver_names euler,heun,midpoint_rk2,dpmpp2m \
  --target_nfe_values 4,8,12 \
  --device auto \
  --forecast_eval_batch_size 64 \
  --allow_execute
```

`--device auto` uses CUDA when PyTorch reports CUDA availability and CPU otherwise. Use `--strict_v43_protocol` to reject non-canonical seed sets.

## Other Entry Points

```bash
genode-train-backbone --dataset san_francisco_traffic --device auto
genode-run-schedules --forecast_datasets san_francisco_traffic --conditional_generation_datasets '' --split_phase validation_tuning --device auto
genode-build-ser-ptg-reference --dataset san_francisco_traffic --device auto
genode-evaluate-schedule-summary --schedule_summary outputs/example/selected_schedule_summary.json --split_phase locked_test --device auto
```

## Development Checks

```bash
PYTHONDONTWRITEBYTECODE=1 python -m compileall -q src tests scripts
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_*.py'
```

For CPU-only smoke checks, set `CUDA_VISIBLE_DEVICES=''` and keep `--device auto`.
