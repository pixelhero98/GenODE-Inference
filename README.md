# GIPO

GIPO is a continuous-density conditional policy for inference-time optimization on frozen optimal-transport flow-matching
backbones. 

## Installation

GIPO requires Python 3.11 or newer.

```bash
python -m pip install -e .
```

Long-Term ST preparation additionally needs WFDB:

```bash
python -m pip install -e ".[medical]"
```

Install the test extra before running the development checks:

```bash
python -m pip install -e ".[test]"
```

Install the PyTorch build appropriate for the target accelerator before
installing GenODE. Keep generated datasets, checkpoints, reports, and packages
in ignored project-local directories such as `data/`, `paper_datasets/`,
`outputs/`, `reports/`, and `dist/`.

Commands resolve paths from the current working directory. When running from
another directory, point GenODE at the workspace explicitly:

```sh
# POSIX shells
export GENODE_PROJECT_ROOT=/path/to/genode-workspace
```

```powershell
# PowerShell
$env:GENODE_PROJECT_ROOT = "C:\path\to\genode-workspace"
```

The multi-line command examples below use POSIX `\` continuations. In
PowerShell, enter each command on one line without the continuation characters.

## Paper protocol

The active policy protocol is `gipo_density`. Each measured fixed or SER
schedule is represented as a 64-bin probability mass over normalized model time
`[0, 1]`:

```text
reference_edges = linspace(0, 1, 65)
density_mass[j] = integral of the equal-step schedule density over bin j
sum(density_mass) = 1
```

Both teacher and student use strict context-only conditioning: solver/NFE
features are combined with a frozen backbone context embedding through the
canonical `additive_mlp` conditioning style. The
`density_form_transformer` teacher ranks measured candidate densities and
predicts their metric utilities with rank and Huber objectives. The
`density_query_transformer` student emits one logit per density bin and is
trained against a teacher-weighted mixture of measured densities.

Teacher checkpoint selection uses `weighted_normalized_regret` on
context-disjoint and density-family calibration diagnostics. Student checkpoint
selection uses `validation_ce` on a context-disjoint calibration split.
Locked-test rows are reporting-only: they never select teacher or student
weights. Temporal backbone artifacts are exported at the exact requested
training budget only, and only after finite validation succeeds.

The paper route is intentionally strict:

- `paper_gipo` is the default and is evaluated alongside the complete required
  fixed-schedule and SER comparison matrix.
- `--allow_incomplete_comparison` permits partial exploratory reports for
  non-paper policies; it cannot relax paper comparison coverage.
- The full locked test is the default. `--locked_test_preview` explicitly
  enables a deterministic preview capped at 512 contexts per logical seed;
  `--locked_test_preview_contexts N` changes that cap and is invalid without
  preview mode.
- Ablation students are off by default. `--include_ablations` is required to
  add them to the full pipeline.

Locked-test context rows and summaries record `checkpoint_step`,
`locked_test_mode`, `locked_test_context_limit`,
`locked_test_context_limit_scope`, `selected_examples_cap_source`,
`selection_was_capped`, and `global_selection_was_capped`, so full and preview
runs remain distinguishable after serialization.

## Evaluation schema

Public evaluation rows use `scenario_key` for the benchmark scenario and
`scheduler_key` for the evaluated schedule. The canonical policy and metric
fields are:

| Field | Meaning |
| --- | --- |
| `scenario_key` | Public evaluation scenario |
| `scheduler_key` | Fixed, SER, or learned schedule identity |
| `gipo_step_budget` | GIPO optimization/training budget |
| `checkpoint_step` | Frozen backbone training step |
| `teacher_final_retrain` | Final teacher retraining metadata |
| `method_key` | Policy or baseline identity |
| `mode` | Canonical policy conditioning mode |
| `forecast_crps` | Forecast CRPS metric |
| `forecast_mase` | Forecast MASE metric |

Rows also carry the family-appropriate context, solver, NFE, seed, and metric
metadata. Retired aliases such as `dataset`, `dataset_key`, `schedule_key`,
`student_gipo_steps`, `selected_gipo_step_budget`, `final_teacher_retrain`,
`paper_method`, `setting_feature_mode`, `crps`, and `mase` are rejected rather
than silently translated. Locked-test inputs must agree on checkpoint identity,
checkpoint step, and context metadata before prediction starts.

Training requires:

- a CSV of fixed/SER per-context rows with stable `scenario_key`, `context_id`,
  `series_id`, `target_t`, `solver_key`, `target_nfe`, `seed`,
  `scheduler_key`, `checkpoint_step`, and metric/utility values;
- a checkpoint-scoped context-embedding NPZ created from the frozen backbone in
  evaluation/no-grad mode;
- schedule-summary JSON for any non-fixed supervision schedules, including SER;
- optionally, unseen-NFE pseudo rows for an explicitly requested pseudo-student
  ablation. Pseudo rows are never used to fit or select the teacher.

Contexts are paired within the same scenario, solver, NFE, checkpoint, context,
and logical seed. The physical `context_id` remains stable across checkpoint
maturities, while the embedding identity remains checkpoint-scoped.

## Evaluation scenarios

The paper matrix contains nine scenarios across three benchmark families:

- Forecast extrapolation: `solar_energy_10m`, `traffic_hourly`,
  `weather_daily`.
- Temporal conditional generation: `cryptos`, `lobster_synthetic`,
  `long_term_st`.
- Molecule 3D coordinate generation: `molecule_3d_set1`,
  `molecule_3d_set2`, `molecule_3d_set3`.

Temporal scenarios use checkpoints at 4000, 8000, 12000, 16000, and 20000
steps: `(3 forecast + 3 conditional generation) x 5 = 30` artifacts. Molecule
artifacts are counted per trainable fixed-shape stratum, giving `30 + 5N`
artifacts when `N` molecule strata are present.

Family metrics are:

- forecast: `forecast_crps` and `forecast_mase`;
- temporal conditional generation: `temporal_cw1`, `temporal_uw1`, and
  `temporal_tstr_f1` where labels exist;
- molecule generation: Kabsch RMSD, ensemble velocity/acceleration norm W1,
  and 16-step autoregressive velocity/acceleration stability errors.

Paper temporal lengths are fixed:

| Scenario | Task | History | Future block | Rollout |
| --- | --- | ---: | ---: | --- |
| `solar_energy_10m` | forecast | 1008 | 1008 | `non_ar` |
| `traffic_hourly` | forecast | 336 | 168 | `non_ar` |
| `weather_daily` | forecast | 120 | 30 | `non_ar` |
| `cryptos` | conditional generation | 256 | 128 | `non_ar` |
| `lobster_synthetic` | conditional generation | 256 | 128 | `non_ar` |
| `long_term_st` | conditional generation | 12000 | 3000 | `non_ar` |

## External data and licenses

The repository's MIT license covers GenODE software only. It does not relicense
third-party data or user-supplied molecule trajectories. Review and comply with
each upstream dataset's terms before downloading, preparing, redistributing, or
publishing derived data.

Downloads performed by GenODE are pinned and accepted only when both the exact
published byte size and checksum match:

| Input | Pinned source | Size (bytes) | Checksum | Upstream license |
| --- | --- | ---: | --- | --- |
| Solar 10-minute archive | [Zenodo 4656144](https://zenodo.org/records/4656144) | 4,559,353 | MD5 `84c0de18383c911091a3cd274661b029` | CC BY 4.0 |
| Traffic hourly archive | [Zenodo 4656132](https://zenodo.org/records/4656132) | 22,868,806 | MD5 `1cf694f99f95700217845078b467fb24` | CC BY 4.0 |
| Weather daily archive | [Zenodo 4654822](https://zenodo.org/records/4654822) | 38,820,451 | MD5 `57155594af0883ccd5e63a5948976796` | CC BY 4.0 |
| Crypto NPZ | [LoBiFlow revision `2d33cfd6b5e27d2483e2095b22d340813389cd0c`](https://huggingface.co/datasets/mpstoryfans/lobiflow/tree/2d33cfd6b5e27d2483e2095b22d340813389cd0c) | 1,962,160,259 | SHA-256 `124fff5767387373323fcb0ec17cc8b8030fe945d037909786127de6d3942e67` | No license declared in the pinned dataset card |
| LOBSTER synthetic profile | [same LoBiFlow revision](https://huggingface.co/datasets/mpstoryfans/lobiflow/tree/2d33cfd6b5e27d2483e2095b22d340813389cd0c) | 7,220 | SHA-256 `f92d3ffa3ef3bdbb67d8d45a337328b032727580a89177f967353dccbb40d50f` | No license declared in the pinned dataset card |

The LoBiFlow sizes and digests come from its [pinned checksum
manifest](https://huggingface.co/datasets/mpstoryfans/lobiflow/blob/2d33cfd6b5e27d2483e2095b22d340813389cd0c/checksums.sha256).

Long-Term ST is obtained separately from [PhysioNet Long-Term ST Database
v1.0.0](https://physionet.org/content/ltstdb/1.0.0/) under the Open Data Commons
Attribution License v1.0; cite its version DOI
[`10.13026/C2G01T`](https://doi.org/10.13026/C2G01T). Molecule trajectory
archives are user-supplied and retain their original terms.

### Prepare public temporal data

Download and prepare the three pinned Monash/Zenodo forecast scenarios:

```bash
python -c "from genode.data.otflow_monash_datasets import download_monash_paper_datasets; download_monash_paper_datasets('paper_datasets')"
```

Download the two pinned LoBiFlow inputs:

```bash
python -c "from genode.data.otflow_datasets import download_cryptos_npz, download_lobster_synthetic_profile; download_cryptos_npz(); download_lobster_synthetic_profile()"
```

For Long-Term ST, place the three source archives outside the repository and
prepare the sanitized 100 Hz context-only data:

```sh
# POSIX shells
export OTFLOW_MEDICAL_STAGING_ROOT=/path/to/medical-staging
```

```powershell
# PowerShell
$env:OTFLOW_MEDICAL_STAGING_ROOT = "C:\path\to\medical-staging"
```

```bash
python -m pip install -e ".[medical]"
python -c "from genode.data.otflow_medical_datasets import prepare_long_term_st_dataset; prepare_long_term_st_dataset()"
```

The preparer extracts only declared WFDB inputs, validates readable signal
tails, omits header comments and sparse annotations, creates deterministic
portable channel filenames, and writes a sanitized manifest.

Prepare fixed-shape molecule groups from user-supplied trajectory archives:

```bash
genode-prepare-molecule-xyz \
  --balanced_groups \
  --zip_paths trajectory.zip,additional_trajectory.zip \
  --group_root data/molecule_3d
```

## Run the paper workflow

The restartable full pipeline trains and reports `paper_gipo` plus its required
baselines. Use the canonical scenario key:

```bash
genode-run-full-pipeline \
  --scenario_key traffic_hourly \
  --device auto
```

Add `--locked_test_preview` only for a 512-context-per-seed result preview. Add
`--include_ablations` only when the paper ablation grid is required.

Generate reusable fixed/SER rows and context embeddings with the schedule
runner. This example selects one forecast scenario and disables the other
families:

```bash
genode-run-schedules \
  --forecast_datasets traffic_hourly \
  --conditional_generation_datasets '' \
  --molecule_datasets '' \
  --split_phase train_tuning \
  --nfe_role seen \
  --checkpoint_steps 4000,8000,12000,16000,20000 \
  --baseline_scheduler_names uniform,late_power_3,flowts_power_sampling,ays,gits,ots \
  --write_context_rows \
  --allow_execute \
  --device auto
```

Build and evaluate a SER summary for the same scenario:

```bash
genode-build-ser-ptg-reference \
  --scenario_key traffic_hourly \
  --reference_split train_tuning \
  --device auto

genode-evaluate-schedule-summary \
  --scenario_key traffic_hourly \
  --schedule_summary <ser-schedule-summary.json> \
  --split_phase train_tuning \
  --write_context_rows \
  --device auto
```

Train the paper density student directly when the prepared inputs already
exist:

```bash
genode-train-gipo \
  --rows_csv <context-rows.csv> \
  --context_embeddings_npz <context-embeddings.npz> \
  --schedule_summary_json <ser-schedule-summary.json> \
  --out_dir <gipo-output-dir>
```

Pseudo-distilled and alternate target-mixture students are ablations, not the
paper default. To invoke the pseudo student explicitly, provide unseen-NFE rows
and set its policy key:

```bash
genode-train-gipo \
  --rows_csv <seen-context-rows.csv> \
  --context_embeddings_npz <seen-context-embeddings.npz> \
  --schedule_summary_json <seen-ser-summary.json> \
  --student_pseudo_rows_csv <unseen-pseudo-rows.csv> \
  --student_pseudo_context_embeddings_npz <unseen-context-embeddings.npz> \
  --student_pseudo_schedule_summary_json <unseen-ser-summary.json> \
  --student_policy_key pseudo_gipo \
  --out_dir <ablation-output-dir>
```

Report a frozen paper student on complete locked-test inputs:

```bash
genode-report-gipo-locked-test \
  --gipo_student_checkpoint <gipo-output-dir>/gipo_student.pt \
  --training_summary <gipo-output-dir>/gipo_training_summary.json \
  --scenario_key traffic_hourly \
  --checkpoint_step 20000 \
  --context_rows <locked-fixed-rows.csv>,<locked-ser-rows.csv> \
  --baseline_rows <locked-fixed-rows.csv> \
  --comparator_rows <locked-ser-rows.csv> \
  --context_embeddings_npz <locked-context-embeddings.npz> \
  --out_dir <locked-report-dir> \
  --device auto
```

## Portable backbone packages

Backbone packages contain a single family, its ready artifacts, the relevant
processed data, a POSIX-relative manifest, and a checksum inventory. Packaging
rejects links, junctions, reparse points, escaping paths, and local workstation
paths, then self-validates the staging tree before publishing it or its zip.

Create each supported family package from a complete workspace:

```bash
genode-package-backbone-family \
  --family temporal-extrapolation \
  --source_root . \
  --output_dir dist/backbone-packages

genode-package-backbone-family \
  --family temporal-generation \
  --source_root . \
  --output_dir dist/backbone-packages

genode-package-backbone-family \
  --family molecule-coord-generation \
  --source_root . \
  --output_dir dist/backbone-packages
```

Validate an extracted package before use:

```bash
genode-validate-backbone-package \
  dist/backbone-packages/genode_temporal_extrapolation_backbones_datasets \
  --expected_family temporal-extrapolation
```

For downstream-only pipeline stages, pass the validated extracted directory via
`--backbone_package_root` together with `--use_provided_backbones`.

## Command-line interfaces

The package installs 15 console entry points:

| Command | Purpose |
| --- | --- |
| `genode-run-schedules` | Evaluate fixed or supplied schedules and write context rows |
| `genode-run-full-pipeline` | Run the paper-first multi-stage GIPO workflow |
| `genode-train-backbone` | Train a temporal OT flow-matching backbone |
| `genode-prepare-molecule-xyz` | Prepare fixed-shape molecule scenario groups |
| `genode-train-molecule-backbone` | Train molecule backbones by fixed-shape stratum |
| `genode-eval-molecule-backbone` | Evaluate molecule backbone metrics and rollouts |
| `genode-train-gipo` | Train a GIPO teacher and student |
| `genode-preflight-gipo-rows` | Validate GIPO rows and schedule support before training |
| `genode-report-gipo-locked-test` | Report a frozen student against locked-test baselines |
| `genode-build-ser-ptg-reference` | Build SER-PTG reference schedules |
| `genode-evaluate-schedule-summary` | Evaluate a supplied schedule-summary artifact |
| `genode-build-hardness-figure` | Build the hardness-mismatch figure |
| `genode-build-ptg-figure` | Build the PTG observed-gain figure |
| `genode-package-backbone-family` | Build a portable, self-validated family package |
| `genode-validate-backbone-package` | Validate an extracted family package |

Run `<command> --help` for the authoritative interface.

## Source layout

- `src/genode/data/`: scenario definitions, secure preparation, split builders,
  experiment plans, and project-relative paths.
- `src/genode/models/`: OT flow-matching configuration, conditioning,
  backbones, and model utilities.
- `src/genode/schedule_transfer/`: schedule construction, registries, signal
  traces, and protocol diagnostics.
- `src/genode/evaluation/`: checkpoint loading, schedule evaluation, solver
  mappings, and sampling.
- `src/genode/gipo/`: density representation, teacher/student models,
  training, SER-PTG references, schema validation, and locked-test reporting.
- `tests/`: protocol, interface, portability, and regression tests.

## Development Checks

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m compileall -q src tests
python -m pip check
git diff --check
```

These cross-platform commands do not run real training, locked-test evaluation,
or large dataset downloads. Python bytecode and coverage outputs are ignored by
default.
