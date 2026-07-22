# GenODE Inference

GenODE Inference is a research-oriented Python package for numerical inference
with frozen optimal-transport flow-matching models. Its scheduling method,
GIPO, learns where an ODE solver should evaluate the vector field. The package
also supports consistency distillation of GIPO-guided rollouts into a one-step
endpoint flow map.

The repository contains code, tests, and command-line workflows. It does not
ship datasets, trained checkpoints, benchmark results, or private/local
infrastructure configuration. The software is alpha, and newly created flow-map
checkpoints and reports make no performance claim until evaluated on paired
benchmark data.
New flow-map checkpoints use `quality_gate.status="not_evaluated"`, while
unevaluated reports use top-level `status="not_evaluated"`.

## Method

A frozen conditional flow defines the ODE

\[
\frac{d x_t}{d t}=v_\theta(x_t,t\mid c), \qquad t\in[0,1],
\]

where \(c\) is the observed context. Accuracy and cost depend not only on the
solver and number of function evaluations (NFE), but also on the placement of
those evaluations in time.

GIPO represents a normalized time density conditioned on context, solver, and
budget,

\[
\rho_\phi(t\mid c,s,K), \qquad
F_\phi(t)=\int_0^t \rho_\phi(u\mid c,s,K)\,du.
\]

The inverse CDF gives the integration grid,

\[
t_i=F_\phi^{-1}\!\left(\frac{i}{M(s,K)}\right),
\]

with solver-specific macro-step count \(M\). A metric-aware teacher learns to
rank candidate schedules and predict their utilities. A student then learns a
teacher-weighted continuous density. Model selection is context-disjoint, and
locked-test labels are never used to fit or choose a policy.

Consistency distillation treats the frozen GIPO student, solver, and NFE as a
trajectory teacher. From intermediate teacher states it learns

\[
D_\psi(x_\tau,\tau,c,s,K,\rho)
\approx
\Phi_{\theta,\rho}^{s,K}(x_\tau,\tau\!\rightarrow\!1;c),
\]

where \(\Phi\) is the numerical rollout. The residual parameterization enforces
the terminal identity \(D_\psi(x,1,\ldots)=x\). At inference, the verified GIPO
policy supplies \(\rho\), while advanced callers may supply a fixed density.

## Scope and status

The reference workflows support Euler, Heun, midpoint RK2, and DPM++2M across
forecast extrapolation, temporal conditional generation, and molecule
coordinate generation. Scenario registration and solver/NFE validation are
strict. The endpoint flow map currently supports transformer OTFlow fields;
other backbone field types are rejected explicitly.

"One endpoint-map evaluation" means one flow-map forward pass per generated
transition or future block. The reported `model_evaluations=1` excludes GIPO
policy evaluation and context encoding. It also does not collapse an outer
autoregressive rollout into one pass: each generated transition or block still
requires its own map call. GIPO and fixed-schedule comparison rows instead
require `model_evaluations=target_nfe`. This accounting describes vector-field
or endpoint-map evaluations, not wall-clock or end-to-end compute parity.

The quality gate freezes candidate selection on validation data, then performs
paired locked-test comparisons against independently selected GIPO and fixed
schedule comparators from an exact, prespecified candidate catalog. Its current
criterion requires statistically supported strict superiority on every primary
metric; equality or an inconclusive test does not pass. The claim applies only
to a flow map evaluated once with density from the bound GIPO checkpoint and to
the validation-selected solver/NFE candidates in that declared catalog. Without
suitable benchmark measurements, the result remains `not_evaluated` and
supports no claim that the flow map equals or exceeds GIPO or the best declared
fixed-schedule comparator, let alone the best schedule in a broader search.

## Installation

GenODE requires Python 3.11 or newer.

```bash
python -m venv .venv
```

Activate the environment for your shell, then install the package:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Install `.[medical,test]` when preparing Long-Term ST data. Device behavior and
all command-specific options are documented by each command's `--help`.
Relative project paths resolve from the current working directory by default;
set `GENODE_PROJECT_ROOT` to use an explicit project root.

## GIPO workflow

Preview the restartable reference pipeline without launching training or
evaluation:

```bash
genode-run-full-pipeline \
  --scenario_key solar_energy_10m \
  --run_root outputs/example \
  --dry_run
```

The pipeline prepares inputs, resolves or trains backbones, evaluates reference
schedules, trains GIPO, and builds the locked-test report. The central commands
can also be run independently:

```text
genode-train-gipo
genode-report-gipo-locked-test
```

Use validated portable backbone packages for workflows that should not retrain
the frozen model. Package commands, data-preparation tools, and figure builders
are listed in `[project.scripts]` in `pyproject.toml`.

## Flow-map distillation

The opt-in flow-map workflow is:

```text
GIPO-guided demonstrations -> endpoint-map training -> paired quality evaluation
```

The corresponding pipeline stage names are
`collect_flow_map_demonstrations`, `train_flow_map`, and `evaluate_flow_map`.

Only training/tuning contexts may be collected; locked-test trajectories are
rejected by both collection and loading. The three commands are:

```bash
genode-collect-flow-map-demonstrations \
  --backbone-checkpoint <backbone.pt> \
  --gipo-checkpoint <gipo-student.pt> \
  --contexts-npz <train-tuning-contexts.npz> \
  --output-dir <demonstrations> \
  --split-phase train_tuning \
  --scenario-key solar_energy_10m \
  --benchmark-family temporal_extrapolation

genode-train-flow-map \
  --demonstration-manifest <demonstrations>/flow_map_demonstrations.json \
  --backbone-checkpoint <backbone.pt> \
  --gipo-checkpoint <gipo-student.pt> \
  --output-checkpoint <flow-map.pt>

genode-evaluate-flow-map \
  --rows-csv <paired-quality-rows.csv> \
  --candidate-catalog <candidate-catalog.json> \
  --quality-contexts-npz <quality-contexts.npz> \
  --quality-sample-panel-npz <quality-sample-panel.npz> \
  --measurement-protocol-json <measurement-protocol.json> \
  --quality-protocol-json <evaluation-run>/protocol.json \
  --scenario-key solar_energy_10m \
  --flow-map-checkpoint <flow-map.pt> \
  --backbone-checkpoint <backbone.pt> \
  --gipo-checkpoint <gipo-student.pt> \
  --output-json <flow-map-quality.json>
```

`--contexts-npz` is an explicit, user-prepared input. It must contain a unique
one-dimensional string array `context_ids` and a finite floating-point
`histories` array whose first dimension is the context count. An optional
finite `conditions` matrix may provide one row per context. Collection never
discovers, downloads, or prepares real data implicitly.

Quality evaluation consumes five explicit, pre-existing measurement artifacts.
It validates their identities and applies the statistical gate; it does not run
the candidates, recompute metrics, or create benchmark rows. The separately
required `--quality-protocol-json` is an execution binding generated from those
five artifacts by the evaluation-only full pipeline; it is not a sixth
measurement artifact. Prepare the measurement artifacts with an independent
benchmark runner using the declared checkpoints and executions:

- `--quality-contexts-npz` contains one-dimensional string arrays `context_ids`
  and `split_phases`, a finite floating-point `histories` array with one leading
  row per context, and optional finite floating-point `conditions`. The split
  values are exactly `validation_tuning` and `locked_test`, and both panels must
  be present. Context IDs and physical fingerprints are unique across the two
  panels.
- `--quality-sample-panel-npz` contains one-dimensional `context_ids`,
  one-dimensional nonnegative integer `logical_seeds`, and finite
  floating-point `initial_states`. Each context must have the same positive
  number of replicates, with unique logical seeds within that context.
- `--candidate-catalog` declares the exact executions considered on validation.
- `--measurement-protocol-json` binds the exact checkpoint trio, candidate
  catalog, context and sample-panel files, reference-data identity, primary
  metrics, external runner implementation and environment, statistical gate
  settings, and the GenODE evaluator source plus Python/NumPy runtime releases.
- `--rows-csv` contains the externally measured, per-context metrics.

The context and sample-panel readers recompute physical fingerprints from array
values. Every candidate must use the same bound context and initial-state panel.
Demonstration, validation, and locked-test physical fingerprints must be
mutually disjoint.

The paired quality CSV must contain these provenance columns:

```text
split_phase, method, candidate_key, candidate_execution_sha256,
solver_key, target_nfe, model_evaluations, context_id, context_fingerprint,
measurement_protocol_sha256, sample_panel_sha256, replicate_count,
scenario_key, flow_map_checkpoint_sha256, backbone_checkpoint_sha256,
gipo_checkpoint_sha256
```

It must also contain every registered primary metric and any applicability
column declared for that metric. `candidate_execution_sha256` binds the
normalized execution object in the catalog. All rows use one
`measurement_protocol_sha256`, computed from the normalized measurement
protocol; `sample_panel_sha256` and `replicate_count` bind the common
initial-state replicates for that physical context. Flow-map rows must report
exactly one model evaluation, while GIPO and fixed rows report their declared
NFE. The context fingerprint is the lowercase SHA-256 returned by
`genode.distillation.artifacts.context_fingerprint(history, condition)`.

Prepare the measurement protocol before running the external benchmark. The
public readers and builder derive the exact identities used by every row:

```python
import json
from pathlib import Path

from genode.distillation.evaluation import (
    candidate_catalog_sha256,
    metric_specs_for_scenario,
    read_candidate_catalog,
    read_quality_contexts,
    read_quality_sample_panel,
)
from genode.distillation.measurement_protocol import (
    measurement_protocol_sha256,
    quality_measurement_protocol_payload,
)
from genode.provenance import file_sha256

candidates = read_candidate_catalog("candidate-catalog.json")
execution_hashes = {
    (candidate.method, candidate.candidate_key): candidate.execution_sha256
    for candidate in candidates
}
quality_contexts = read_quality_contexts("quality-contexts.npz")
sample_panels = read_quality_sample_panel(
    "quality-sample-panel.npz",
    quality_context_binding=quality_contexts,
)

metrics = [
    {
        "name": spec.name,
        "direction": spec.direction,
        "weight": spec.weight,
        "applicable_key": spec.applicable_key,
    }
    for spec in metric_specs_for_scenario("solar_energy_10m")
]
measurement_protocol = quality_measurement_protocol_payload(
    scenario_key="solar_energy_10m",
    candidate_catalog_sha256=candidate_catalog_sha256(candidates),
    quality_contexts_sha256=quality_contexts["artifact_sha256"],
    quality_sample_panel_sha256=sample_panels["artifact_sha256"],
    reference_data_sha256=file_sha256("reference-data-manifest.json"),
    artifact_binding={
        "flow_map_checkpoint_sha256": file_sha256("flow-map.pt"),
        "backbone_checkpoint_sha256": file_sha256("backbone.pt"),
        "gipo_checkpoint_sha256": file_sha256("gipo-student.pt"),
    },
    primary_metrics=metrics,
    runner={
        "name": "project-quality-runner",
        "release": "documented-release-id",
        "implementation_sha256": file_sha256("tools/quality_runner.py"),
        "environment_sha256": file_sha256("quality-runner-lock.json"),
    },
)
Path("measurement-protocol.json").write_text(
    json.dumps(measurement_protocol, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
row_measurement_protocol_sha256 = measurement_protocol_sha256(
    measurement_protocol
)
```

When changing `--bootstrap-samples`, `--seed`, or `--familywise-alpha`, pass the
same values to the protocol builder. The runner and reference-data digests are
consistency commitments supplied by the caller, not signatures, trusted
timestamps, or proof of origin. The quality gate validates those declarations
and compares the supplied statistics but does not independently recompute them.

Candidate selection is frozen independently for `flow_map`, `gipo`, and
`fixed` on `validation_tuning`. The selected flow map is then compared with the
selected GIPO and fixed candidates on matched `locked_test` contexts. Each
paired comparison requires at least 20 contexts.

The candidate catalog is a JSON list. Each entry has exactly `method`,
`candidate_key`, `solver_key`, `target_nfe`, and a method-specific `execution`
object. The following is a schema fragment, not a complete claim-producing
catalog:

```json
[
  {
    "method": "flow_map",
    "candidate_key": "flow-map-euler-nfe4",
    "solver_key": "euler",
    "target_nfe": 4,
    "execution": {
      "kind": "endpoint_flow_map",
      "density_source": "bound_gipo_checkpoint"
    }
  },
  {
    "method": "gipo",
    "candidate_key": "gipo-euler-nfe4",
    "solver_key": "euler",
    "target_nfe": 4,
    "execution": {
      "kind": "gipo_ode_rollout",
      "policy_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
    }
  },
  {
    "method": "fixed",
    "candidate_key": "uniform-euler-nfe4",
    "solver_key": "euler",
    "target_nfe": 4,
    "execution": {
      "kind": "fixed_time_grid",
      "scheduler_key": "uniform",
      "density_source_key": "uniform",
      "time_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
      "time_grid_sha256": "fe8c328c2d86c30c7ff4a041dffe9067d0eb6025a4d395f3e76e930c37fa23e4"
    }
  }
]
```

Replace the illustrative GIPO policy digest with the SHA-256 of the bound GIPO
checkpoint. Fixed candidates must name a registered scheduler and include its
exact solver-specific time grid and semantic grid digest. A claim-producing
catalog requires at least two candidates and two distinct solver/NFE settings
for each method; fixed candidates must also span at least two registered
schedule and density-source families. Validation rows must cover the declared catalog exactly;
missing, duplicate, or extra candidates are rejected.

The external measurement protocol is prepared before measurement and candidate
selection. Separately, the evaluation-only full pipeline writes `protocol.json`
to bind the finalized rows and all supplied artifacts for execution and resume;
that later file does not prove preregistration chronology. Prefer invoking the
quality gate through that pipeline. For direct low-level
`genode-evaluate-flow-map` use, first run the evaluation-only command below with
`--dry_run`; it validates and hashes the supplied artifacts and writes
`<run-root>/protocol.json` without running an experiment. Pass that file as
`--quality-protocol-json`, keeping every path and gate setting unchanged.

The resulting claim is about strict superiority of the validation-selected
one-evaluation flow-map candidate over both selected comparators within the
declared catalog. It is not evidence about an undeclared global search,
end-to-end latency, or measurements absent from the supplied CSV.

To append these stages to a pipeline preview:

```bash
genode-run-full-pipeline \
  --scenario_key solar_energy_10m \
  --run_root outputs/example \
  --include_flow_map \
  --flow_map_backbone_checkpoint <backbone.pt> \
  --flow_map_contexts_npz <train-tuning-contexts.npz> \
  --dry_run
```

That preview omits quality measurements. A corresponding non-dry run writes an
explicit `not_evaluated` report after training the flow map. The quality gate
cannot manufacture measurements for that new checkpoint.

After producing a checkpoint, prepare the measurement protocol and external
measurements against that exact checkpoint, backbone, and GIPO policy. Then run
only the evaluation stage in a new pipeline root and provide all five quality
inputs together:

```bash
genode-run-full-pipeline \
  --scenario_key solar_energy_10m \
  --run_root outputs/flow-map-quality-evaluation \
  --stages evaluate_flow_map \
  --flow_map_backbone_checkpoint <backbone.pt> \
  --flow_map_gipo_checkpoint <gipo-student.pt> \
  --flow_map_checkpoint <flow-map.pt> \
  --flow_map_quality_rows_csv <paired-quality-rows.csv> \
  --flow_map_quality_candidate_catalog <candidate-catalog.json> \
  --flow_map_quality_contexts_npz <quality-contexts.npz> \
  --flow_map_quality_sample_panel_npz <quality-sample-panel.npz> \
  --flow_map_quality_measurement_protocol <measurement-protocol.json>
```

This evaluation-only pipeline writes the finalized file and semantic hashes to
its own `protocol.json` before invoking the quality gate. Omitting the quality
inputs always keeps `performance_claim=false`.

Load a hash-verified endpoint-map sampler with its source checkpoints:

```python
from genode.distillation.checkpoint import load_flow_map_sampler

sampler, metadata = load_flow_map_sampler(
    "flow-map.pt",
    backbone_checkpoint="backbone.pt",
    gipo_checkpoint="gipo-student.pt",
    device="cpu",
)
future = sampler.sample_future(history, solver_key="heun", target_nfe=8)
```

Artifacts record content identities and portable configuration rather than
workstation paths. Every console command supports `--help`.

## Data, license, and verification

The MIT license covers GenODE software only; it does not relicense third-party
datasets or user-supplied trajectories. Review upstream terms before download,
redistribution, or publication. Real-data preparation is explicit and never
runs on package import. Tests exercise preparation safety only with synthetic,
temporary archives; they do not download or prepare real datasets.

Long-Term ST preparation is an explicit public API:

```python
from genode.data.otflow_medical_datasets import prepare_long_term_st_dataset

manifest = prepare_long_term_st_dataset(
    "datasets/long_term_st",
    archive_paths=["downloads/long-term-st.zip"],
)
```

Install `.[medical]` first. Set `OTFLOW_MEDICAL_STAGING_ROOT` when using the
default raw/staging locations instead of explicit archive paths. Preparation
uses staged promotion and refuses symlink, junction, or reparse-point output
paths.

The local checks use synthetic or unit-scale inputs and do not run benchmark
experiments:

```bash
python -m pytest -q
python -m ruff check .
python -m compileall -q src tests
python -m pip check
git diff --check
```

For a packaging check, install `build` separately and run `python -m build`.
Continuous integration exercises the test suite on Linux and Windows, installs
the built wheel, and smoke-tests every registered console command.
