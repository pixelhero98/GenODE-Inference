# GIPO

GIPO is a research-oriented Python package for optimizing numerical ODE
schedules on frozen optimal-transport flow-matching backbones. It provides:

- GIPO, a context-conditioned continuous-density scheduling policy;
- fixed and SER schedule evaluation across several solver/NFE settings;
- restartable reference workflows for temporal and molecular scenarios;
- portable backbone packaging and integrity validation; and
- opt-in consistency distillation from GIPO-guided trajectories into a
  one-evaluation endpoint flow map.

The repository contains code, schemas, tests, and command-line interfaces. It
does not include datasets, trained checkpoints, benchmark outputs, or private
infrastructure configuration.

## Project status

The package is alpha research software. The flow-map implementation and quality
gate are available, but no benchmark campaign is included in this repository.
New flow-map checkpoints therefore default to
`quality_gate.status="not_evaluated"`; the pipeline also records
`quality_status="not_evaluated"` until paired benchmark rows are supplied.
There is currently no claim that a distilled flow map matches or exceeds GIPO
or the best fixed schedule.

"One evaluation" refers to one endpoint-map forward pass for each generated
transition or block. GIPO policy selection and context encoding are reported as
separate work, and autoregressive consumers still perform their outer rollout.

## Installation

GIPO requires Python 3.11 or newer.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

For Long-Term ST preparation, install the medical extra as well:

```bash
python -m pip install -e ".[medical,test]"
```

The runtime supports `auto`, `cpu`, `cuda`, and `mps` device selection. Commands
fail clearly when a requested accelerator is unavailable; they do not silently
change devices.

## Supported solvers and reference profile

The public solver keys are `euler`, `dpmpp2m`, `heun`, and `midpoint_rk2`.
Target NFE is validated centrally, including the even-NFE requirement for Heun
and midpoint RK2.

The bundled reference profile uses five backbone maturities (4000, 8000, 12000,
16000, and 20000 steps), seen NFEs `(4, 8, 12, 16)`, and unseen NFEs
`(6, 10, 14, 20)` across nine scenarios:

| Family | Scenario keys |
| --- | --- |
| Forecast extrapolation | `solar_energy_10m`, `traffic_hourly`, `weather_daily` |
| Temporal conditional generation | `cryptos`, `lobster_synthetic`, `long_term_st` |
| Molecule coordinate generation | `molecule_3d_set1`, `molecule_3d_set2`, `molecule_3d_set3` |

Scenario registration is strict. Custom scenarios must declare their benchmark
family explicitly instead of relying on name or metric inference.

## Reference GIPO workflow

Preview a complete scenario workflow without launching training or evaluation:

```bash
genode-run-full-pipeline \
  --scenario_key solar_energy_10m \
  --run_root outputs/example \
  --dry_run
```

The default stages prepare inputs, train or resolve backbones, build fixed/SER
rows, train `gipo`, and create locked-test reports. `--include_ablations` adds
the explicit ablation grid. `--stages` accepts a comma-separated subset when a
smaller workflow is needed; stage-selection flags cannot be combined with an
explicit stage list.

GIPO learns a context-conditioned continuous-density policy named
`gipo_density`. A metric-vector teacher uses pairwise rank plus Huber objectives;
the student fits teacher-weighted `density_mass` targets on a uniform reference
grid with context-disjoint validation. The reference candidate set includes
`uniform`, `late_power_3`, `flowts_power_sampling`, `ays`, `gits`, `ots`, and
SER schedules. Locked-test labels never participate in policy or checkpoint
selection.

Use `--backbone_package_root` with `--use_provided_backbones` for downstream
workflows that must consume validated prebuilt artifacts. Provided-backbone mode
refuses the `backbone_training` stage.

All paths are configurable. Generated datasets, outputs, reports, and
checkpoints belong in ignored project-local directories or an explicitly chosen
external directory; no workstation path is embedded in tracked files.

## Endpoint flow-map distillation

Consistency distillation is opt-in. Demonstrations are generated only from
training/tuning contexts. Locked-test trajectories are rejected by both the
collector and manifest loader.

The context input is an NPZ file loaded with `allow_pickle=False` and contains:

- `context_ids`: a one-dimensional Unicode or byte-string array;
- `histories`: finite floating-point data shaped
  `[contexts, history_steps, features]`; and
- optional `conditions`: finite floating-point data shaped
  `[contexts, condition_features]`.

The default demonstration grid is the four supported solvers crossed with NFEs
`4, 6, 8, 10, 12, 14, 16, 20`. To choose a subset, pass comma-separated
`solver_key:target_nfe` pairs.

### Collect demonstrations

```bash
genode-collect-flow-map-demonstrations \
  --backbone-checkpoint <backbone.pt> \
  --gipo-checkpoint <gipo_student.pt> \
  --contexts-npz <train_tuning_contexts.npz> \
  --output-dir <demonstration-dir> \
  --split-phase train_tuning \
  --scenario-key solar_energy_10m \
  --benchmark-family temporal_extrapolation
```

The collector stores checksummed NPZ shards plus
`flow_map_demonstrations.json`. Manifests use paths relative to their own
directory and reject traversal, object arrays, corrupt hashes, incompatible
checkpoint identities, and test-split demonstrations.

### Train the endpoint map

```bash
genode-train-flow-map \
  --demonstration-manifest <demonstration-dir>/flow_map_demonstrations.json \
  --backbone-checkpoint <backbone.pt> \
  --gipo-checkpoint <gipo_student.pt> \
  --output-checkpoint <flow-map.pt> \
  --summary-json <flow-map-training.json>
```

Training is context-disjoint and validation-selected. The frozen backbone
conditioner and GIPO policy provide conditioning; the endpoint map is trained
against teacher endpoints. Its residual parameterization enforces exact terminal
identity. Source paths are not serialized into the checkpoint—only content
hashes and portable configuration are stored.

Load the verified one-step sampler with the same source checkpoints used for
distillation:

```python
from genode.distillation.checkpoint import load_flow_map_sampler

sampler, metadata = load_flow_map_sampler(
    "flow-map.pt",
    backbone_checkpoint="backbone.pt",
    gipo_checkpoint="gipo_student.pt",
    device="cpu",
)

# The bound GIPO policy produces the 64-bin density for this context and setting.
future = sampler.sample_future(
    history,
    solver_key="heun",
    target_nfe=8,
)
```

Advanced callers may pass `density_mass=` explicitly to evaluate a fixed or
externally supplied schedule. The default path composes the verified GIPO
policy with the one-evaluation flow map automatically.

### Evaluate quality

Without benchmark rows, create an explicit code-only report:

```bash
genode-evaluate-flow-map \
  --scenario-key solar_energy_10m \
  --flow-map-checkpoint <flow-map.pt> \
  --backbone-checkpoint <backbone.pt> \
  --gipo-checkpoint <gipo_student.pt> \
  --output-json <flow-map-quality.json> \
  --not-evaluated-reason "Benchmark rows have not been generated."
```

For an evaluated gate, provide paired `validation_tuning` and `locked_test` rows
for the `flow_map`, `gipo`, and `fixed` methods:

```bash
genode-evaluate-flow-map \
  --rows-csv <paired-quality-rows.csv> \
  --scenario-key solar_energy_10m \
  --flow-map-checkpoint <flow-map.pt> \
  --backbone-checkpoint <backbone.pt> \
  --gipo-checkpoint <gipo_student.pt> \
  --output-json <flow-map-quality.json>
```

Each row requires `split_phase`, `method`, `candidate_key`, `solver_key`,
`target_nfe`, `context_id`, a prespecified `selection_utility`, the scenario's
primary metrics, and the scenario/backbone/GIPO/flow-map content hashes.
Conditional metric applicability columns are required where declared by the
scenario. Candidate settings are selected on validation only and then frozen.
Validation and locked-test panels must be disjoint, rows must be unique, and at
least 20 paired locked-test contexts are required. The gate applies a
zero-margin, one-sided paired bootstrap comparison against both GIPO and fixed
comparators with Holm familywise correction. A passing report is evidence for
the exact hash-bound rows and checkpoint only; it is not inferred from training
loss.

### Pipeline integration

The three distillation stages can be appended to the reference workflow:

```bash
genode-run-full-pipeline \
  --scenario_key solar_energy_10m \
  --run_root outputs/example \
  --include_flow_map \
  --flow_map_backbone_checkpoint <backbone.pt> \
  --flow_map_contexts_npz <train_tuning_contexts.npz> \
  --dry_run
```

The pipeline reuses the checkpoint produced by `train_gipo` unless
`--flow_map_gipo_checkpoint` is supplied. If
`--flow_map_quality_rows_csv` is omitted, `evaluate_flow_map` writes a
`not_evaluated` report and makes no performance claim. The stages are also
available explicitly as `collect_flow_map_demonstrations`, `train_flow_map`, and
`evaluate_flow_map`.

## Portable backbone packages

Build and validate a self-contained family package:

```bash
genode-package-backbone-family \
  --family temporal-extrapolation \
  --output_dir <package-output>

genode-validate-backbone-package \
  <package-output>/genode_temporal_extrapolation_backbones_datasets \
  --expected_family temporal-extrapolation
```

Package creation is staged atomically using short temporary paths for Windows
compatibility. Validation checks relative paths, checksums, file sizes, artifact
grids, checkpoint loadability, and symlink/reparse-point safety.

## Data and licensing

The MIT license covers GenODE software only. It does not relicense third-party
datasets or user-supplied molecule trajectories. Review upstream terms before
downloading, redistributing, or publishing derived data.

The temporal reference inputs are pinned by size and checksum in the data
modules. Forecast archives come from the Monash forecasting collection on
Zenodo ([solar](https://zenodo.org/records/4656144),
[traffic](https://zenodo.org/records/4656132), and
[weather](https://zenodo.org/records/4654822)). Crypto and LOBSTER inputs use a
pinned [LoBiFlow revision](https://huggingface.co/datasets/mpstoryfans/lobiflow/tree/2d33cfd6b5e27d2483e2095b22d340813389cd0c),
whose dataset card does not declare a license. Long-Term ST comes from
[PhysioNet](https://physionet.org/content/ltstdb/1.0.0/) under the Open Data
Commons Attribution License; molecule archives remain user supplied.

Data preparation is explicit and never runs during import or tests. See each
preparation command's `--help` and keep source archives outside version control.

## Command-line interfaces

| Command | Purpose |
| --- | --- |
| `genode-run-schedules` | Evaluate fixed and transferred schedules |
| `genode-run-full-pipeline` | Run the restartable multi-stage workflow |
| `genode-train-backbone` | Train a temporal OTFlow backbone |
| `genode-prepare-molecule-xyz` | Prepare molecule trajectory groups |
| `genode-train-molecule-backbone` | Train a molecule backbone |
| `genode-eval-molecule-backbone` | Evaluate a molecule backbone |
| `genode-train-gipo` | Train the GIPO teacher and student |
| `genode-preflight-gipo-rows` | Validate GIPO rows and schedule coverage |
| `genode-report-gipo-locked-test` | Report a frozen GIPO policy |
| `genode-build-ser-ptg-reference` | Build SER/PTG schedule summaries |
| `genode-evaluate-schedule-summary` | Validate schedule-summary results |
| `genode-package-backbone-family` | Create a portable backbone/data package |
| `genode-validate-backbone-package` | Validate a portable package |
| `genode-collect-flow-map-demonstrations` | Collect GIPO-guided trajectories |
| `genode-train-flow-map` | Train an endpoint flow map |
| `genode-evaluate-flow-map` | Apply the validation-frozen quality gate |

Every console command supports `--help`.

## Development checks

The local verification suite uses synthetic/unit inputs only; it does not run
training or benchmark experiments.

```bash
python -m pytest -q
python -m ruff check .
python -m compileall -q src
python -m pip check
git diff --check
python -m build
```

Continuous integration runs the unit suite on Linux and Windows, checks package
installation, and invokes `--help` for every registered console command.
