# GenODE Inference

GenODE Inference is a research-oriented Python package for numerical inference
with frozen optimal-transport flow-matching models. Its scheduling method,
GIPO, learns where an ODE solver should evaluate the vector field. The package
also supports consistency distillation of GIPO-guided rollouts into a one-step
endpoint flow map.

The repository contains code, tests, and command-line workflows. It does not
ship datasets, trained checkpoints, benchmark results, or infrastructure
configuration. The software is alpha, and newly created flow-map checkpoints
and reports make no performance claim until evaluated on paired benchmark data.
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

"One evaluation" means one endpoint-map forward pass per generated transition
or block. It excludes GIPO policy evaluation and context encoding, and it does
not collapse an outer autoregressive rollout into one pass.

The quality gate freezes candidate selection on validation data, then performs
paired locked-test comparisons against independently selected GIPO and fixed
schedule comparators. Its current criterion requires statistically supported
strict superiority on every primary metric; equality or an inconclusive test
does not pass. Without suitable benchmark rows, the result remains
`not_evaluated` and supports no claim that the flow map equals or exceeds GIPO
or the best fixed schedule.

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
the frozen model. Exact stage names, package commands, data-preparation tools,
and figure builders are listed in `[project.scripts]` in `pyproject.toml`.

## Flow-map distillation

The opt-in flow-map workflow is:

```text
GIPO-guided demonstrations -> endpoint-map training -> paired quality evaluation
```

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
  --scenario-key solar_energy_10m \
  --flow-map-checkpoint <flow-map.pt> \
  --backbone-checkpoint <backbone.pt> \
  --gipo-checkpoint <gipo-student.pt> \
  --output-json <flow-map-quality.json>
```

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
redistribution, or publication. Data preparation is explicit and never runs on
package import or during tests.

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
