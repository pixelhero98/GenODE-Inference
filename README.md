# genODE

genODE is a Python package for GIPO, a continuous-density policy for
schedule optimization on frozen OT flow-matching backbones. The public workflow
trains a metric-vector teacher over measured fixed/SER schedule candidates, then
trains a student that emits a normalized time-density for each context.

## Source Layout

- `src/genode/data/`: dataset definitions, split builders, experiment plans, and
  project-relative paths.
- `src/genode/models/`: OTFlow configuration, conditioning, backbone modules,
  training, and model utilities.
- `src/genode/schedule_transfer/`: fixed schedule grids, registries, table
  helpers, signal traces, and diagnostics.
- `src/genode/evaluation/`: checkpoint loading, schedule evaluation, solver
  mappings, and sampling helpers.
- `src/genode/gipo/`: GIPO density representation, teacher/student models,
  training, SER-PTG references, schedule-summary evaluation, and locked-test
  reporting.
- `tests/`: unit tests for public interfaces and canonical behavior.

## Installation

Use Python 3.11 or newer.

```bash
python -m pip install -e .
```

For GPU environments, install the PyTorch build that matches your CUDA or
accelerator stack before installing genODE. Generated data, checkpoints, logs,
and reports should stay in ignored project-local directories such as `data/`,
`paper_datasets/`, `outputs/`, or `reports/`.

## GIPO Protocol

The active protocol is `gipo_density_v1`. Measured fixed/SER schedules are
supervision candidates only. Each schedule grid is converted to canonical
64-bin `density_mass_v1` on normalized model time `[0, 1]`:

```text
reference edges: linspace(0, 1, 65)
p[j] = integral over reference bin j of the equal-step schedule density
sum_j p[j] = 1
```

GIPO uses strict context-only conditioning. The teacher and student both receive
solver/NFE features plus a frozen context embedding:

```text
z_inputs = concat(setting_features(solver, NFE), frozen_context_embedding)
z = additive_mlp_v1(z_inputs)
```

The canonical teacher is `density_form_transformer_v1`. It attends over ordered
density-bin tokens and predicts a metric utility vector. It is trained with a
pairwise rank objective plus Huber regression on the selected utility targets.
Default forecast utility columns are `u_crps_uniform,u_mase_uniform`, and custom
tasks can provide their own utility columns through `--teacher_metric_target_keys`
plus `--teacher_utility_weights`. Larger utility must mean better downstream
performance.

The canonical student is `density_query_transformer_v1`. It builds one query
token per density bin, applies the same additive context-only conditioning, and
normalizes one logit per bin with softmax. Student targets are teacher-weighted
mixtures of measured candidate densities:

```text
w_i = softmax(teacher_utility_i / temperature)
target_density = sum_i w_i * density_mass_i
loss = KL(target_density || student_density)
```

Teacher checkpoint selection uses `weighted_normalized_regret_v1` over
context-disjoint, density-family, and unseen-NFE calibration diagnostics.
Student checkpoint selection uses `validation_ce_v1` on a context-disjoint
calibration validation split. Locked-test rows are reporting-only and are never
used for teacher or student selection.

## Required Inputs

Training requires:

- A CSV of per-example fixed/SER context rows. Rows must include stable
  `context_id`, `series_id`, `target_t`, `solver_key`, `target_nfe`, `seed`,
  `scheduler_key`, and metric/utility columns.
- A context embedding NPZ sidecar created from the frozen backbone under
  eval/no-grad mode.
- Optional schedule-summary JSON files for non-fixed references such as SER.
- Optional unseen-NFE train-tuning rows for teacher selection diagnostics.

Rows are paired inside exact `(dataset, solver, NFE, context_id, seed)` cells.
They must not cross solver, NFE, seed, series, target time, or context identity.

## Evaluation Datasets

The active experiment matrix has exactly nine public dataset keys:

- Forecast extrapolation: `solar_energy_10m`, `traffic_hourly`,
  `weather_daily`.
- Temporal conditional generation: `cryptos`, `lobster_synthetic`,
  `long_term_st`.
- Molecule 3D coordinate generation: `molecule_3d_set1`,
  `molecule_3d_set2`, `molecule_3d_set3`.

Retired keys are not accepted by active forecast or conditional-generation
dataset parsers.

Canonical temporal experiment lengths are locked by `PAPER_EXPERIMENT_SPECS`:

| Dataset | Task | `history_len` | `future_block_len` | Rollout |
| --- | --- | ---: | ---: | --- |
| `solar_energy_10m` | forecast | 1008 | 1008 | `non_ar` |
| `traffic_hourly` | forecast | 336 | 168 | `non_ar` |
| `weather_daily` | forecast | 120 | 30 | `non_ar` |
| `cryptos` | conditional generation | 256 | 200 | `non_ar` |
| `lobster_synthetic` | conditional generation | 256 | 200 | `non_ar` |
| `long_term_st` | conditional generation | 12000 | 3000 | `non_ar` |

Monash prepared manifests also include a `context_length` field used during
data preparation; the experiment context is the `history_len` above.

Forecast datasets are downloaded from ForecastingData/Monash into
`paper_datasets/`:

```bash
python - <<'PY'
from genode.data.otflow_monash_datasets import download_monash_paper_datasets
download_monash_paper_datasets("paper_datasets")
PY
```

`cryptos` and `lobster_synthetic` use the public lobiflow data layout. Download
the prepared crypto NPZ and synthetic profile into `data/`:

```bash
python - <<'PY'
from genode.data.otflow_datasets import download_cryptos_npz, download_lobster_synthetic_profile
download_cryptos_npz()
download_lobster_synthetic_profile()
PY
```

`long_term_st` is the canonical context-only ECG continuation dataset. Place the
three raw `long_term_st-*.zip` archives outside git, for example:

```bash
export OTFLOW_MEDICAL_STAGING_ROOT=../genode-medical-staging
mkdir -p "$OTFLOW_MEDICAL_STAGING_ROOT/raw/long_term_st"
```

Prepare the context-only 100 Hz dataset:

```bash
python - <<'PY'
from genode.data.otflow_medical_datasets import prepare_long_term_st_dataset
prepare_long_term_st_dataset()
PY
```

The preparer treats the archives as one WFDB source, extracts only `RECORDS`,
`.hea`, and header-referenced `.dat` files, validates readable declared tails,
skips suspect records, ignores sparse `.atr` annotations, omits header comments,
and writes sanitized prepared arrays plus
`data/long_term_st_100hz_context_only/manifest.json`. The locked task is
`history_len=12000` and `future_block_len=3000`, i.e. a 120-second ECG context
and 30-second continuation at 100 Hz with no external condition labels.

Molecule group datasets are built from local molecule trajectory zip files. Each
group contains whole fixed-shape strata; mixed atom counts are evaluated through
per-stratum subdatasets rather than padded into one tensor.

```bash
genode-prepare-molecule-xyz \
  --balanced_groups \
  --zip_paths trajectory.zip,triangulene_3.zip \
  --group_root data/molecule_3d
```

## Train GIPO

Generate reusable fixed/SER rows and context embeddings with the schedule
runner:

```bash
genode-run-schedules \
  --forecast_datasets <dataset-key> \
  --split_phase train_tuning \
  --baseline_scheduler_names uniform,late_power_3,flowts_power_sampling,ays,gits,ots \
  --write_forecast_context_rows \
  --device auto
```

If a SER schedule summary already exists, evaluate it with row writing enabled:

```bash
genode-evaluate-schedule-summary \
  --schedule_summary <ser-schedule-summary.json> \
  --split_phase train_tuning \
  --write_context_rows \
  --device auto
```

Train the canonical 64-bin additive density policy:

```bash
genode-train-gipo \
  --rows_csv <context-rows.csv> \
  --context_embeddings_npz <context-embeddings.npz> \
  --schedule_summary_json <ser-schedule-summary.json> \
  --teacher_unseen_selection_rows_csv <unseen-nfe-rows.csv> \
  --teacher_unseen_selection_context_embeddings_npz <unseen-context-embeddings.npz> \
  --out_dir <output-dir> \
  --support_schedule_keys uniform,late_power_3,flowts_power_sampling,ays,gits,ots,ser_ptg_local_defect_eta005
```

For non-forecast tasks, provide utility columns and weights:

```bash
genode-train-gipo \
  --rows_csv <rows-with-utility-columns.csv> \
  --context_embeddings_npz <context-embeddings.npz> \
  --out_dir <output-dir> \
  --teacher_metric_target_keys u_accuracy_gain,u_latency_gain \
  --teacher_utility_weights u_accuracy_gain=0.8,u_latency_gain=0.2
```

## Report Locked Test

Locked-test reporting applies a frozen student checkpoint to each locked-test
context, converts the predicted density to a solver grid by inverse CDF, and
evaluates that grid. It does not change teacher checkpoints, student weights, or
density metadata.

```bash
genode-report-gipo-locked-test \
  --gipo_student_checkpoint <output-dir>/gipo_student.pt \
  --training_summary <output-dir>/gipo_training_summary.json \
  --context_rows <locked-fixed-context.csv>,<locked-ser-context.csv> \
  --context_embeddings_npz <locked-context-embeddings.npz> \
  --out_dir <locked-report-dir> \
  --device auto
```

## Development Checks

```bash
PYTHONDONTWRITEBYTECODE=1 python -m compileall -q src tests
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_*.py'
```

For CPU-only smoke checks, set `CUDA_VISIBLE_DEVICES=''` and keep
`--device auto`.
