# Student selection and score schedules

Training minimizes distillation minus the weighted differentiable teacher score.
Teacher selection minimizes measured held-out reference-mixture regret; its default
checkpoint interval is **20 steps**. Training objectives, temperatures, conditioning
and 2,000-step horizons are unchanged.

## Deterministic student

`GICO-det-policy` checkpoints every **10 steps** by default. For each held-out
context and solver/NFE setting, compute **one student density**. Reuse it for the
validation KL and **one frozen-teacher evaluation**, with the native teacher
conditioning even when student conditioning is global. Rank by the calibrated,
metric-weighted teacher output **before auxiliary standardization or clipping**.
No generator or terminal-scorer calls occur during deterministic selection.

Among checkpoints with positive score-chasing coefficient, let `KL_min` be the
minimum mean validation KL over the complete training horizon. Admit checkpoints
with `KL <= 1.15 * KL_min`, then select the highest mean teacher utility, preferring
the earlier step on exact ties. The configurable `deterministic_kl_allowance`
defaults to `0.15`; it is fixed before selection. If the minimum KL is zero, the
allowance is zero. KL and utility weight contexts equally within settings and
settings equally; ImageNet uses equal class weighting within each setting.

The fitter retains admissible candidate weights until the final minimum is known,
so a later, lower KL can disqualify an earlier provisional winner. This is a
surrogate selection protocol, not measured image quality. It reuses validation
solve observations already used for teacher selection; it does not impose a new
split on the underlying dataset or create fresh held-out evidence.

## Stochastic student and schedules

`GICO-sto-policy` retains measured calibrated terminal-utility selection and the
`student_checkpoint_every` default of **100 steps**. NLL remains diagnostic.
Exact utility ties prefer the earlier checkpoint. No test or comparison-panel
measurements enter either selection path, and inference does not select rewards.
`deterministic_checkpoint_every` controls only deterministic checkpoint cadence.

| `score_schedule` | First 60% | Next 20% | Final 20% |
|---|---|---|---|
| `linear_60_40` (default control) | Zero | Linear ramp to half beta | Continue ramp to full beta |
| `ramp_plateau_60_20_20` | Zero | Linear ramp to full beta | Full beta |
| `constant_60_40` | Zero | Full beta immediately | Full beta |

Checkpoints are eligible when their realized coefficient is positive. Eligibility
does not require the ramp to finish. The final update is always checkpointed.
The default schedule remains the historical control pending measured comparisons.

## Evaluator contract

Python `fit` and `fit_models` require `selection_evaluator(candidate)` for
`GICO-sto-policy` or `both`. Deterministic-only fitting needs no evaluator; a
supplied factory is validated but not imported or invoked. It receives
a detached CPU `StudentCandidate` in evaluation mode and returns raw paired
terminal-measurement rows. Its `density` and `materialize` methods use the same
inference decoder as deployed policies. Measured stochastic selection never substitutes teacher predictions.
The callback must execute the frozen task generator and terminal metrics, verify
generator/scorer identity and unchanged weights, and retain its raw measurements.

Use the reference-evidence schema, restricted to its exact validation contexts,
solver/NFE cells, generation seeds and reference identities. Each cell must have
`schedule_key: uniform` and `schedule_key: student`. Student rows must include
`selection_checkpoint_id: candidate.checkpoint_id`. Record the actually executed
`density_mass` and `time_grid`; their replay and exact NFE are checked. Terminal
metrics and the uniform anchor use the unchanged measurement protocol and binding.

Stochastic evaluation uses `selection_clock_replicates` independent complete-policy
draws (default 4), with `clock_replicate` indexed from zero. Repeat the matching
uniform row for each draw. Supply `clock_seed` and `clock_request_id`, separate from
generation noise. Ensembles require `sample_clocks`, one per member; retain each
member's clock throughout the rollout. Innovations must be distinct across members
and replicates across all distinct generated members and replicates. Pair their identities across checkpoints.
The measured-evaluation utility also supports deterministic candidates with one
replicate for explicit reporting; the fitter does not use it for deterministic selection.

Raw metrics are averaged across clock and generation replicates before nonlinear
reward transformation. Apply the frozen task calibration and weights, average
contexts equally within solver/NFE settings, then average settings equally.
ImageNet averages classes equally. Uniform remains zero. The callback cannot alter
the candidate state or conditioning, and ambient Torch, NumPy and Python RNGs are
restored after evaluation. Callbacks own any external processes and must not mutate
the live generator, teacher or training inputs.

For the common configuration CLI, add a trusted adapter factory:

```json
{
  "selection_evaluator": {
    "factory": "genode.gico.evaluators:build_evaluator",
    "config": {
      "rows": "selection.jsonl", "contexts": "contexts.npz",
      "runtime": "runtime.json", "cases": "cases.json",
      "output": "selection-measurements", "clock_seed": 412
    }
  },
  "score_schedule": "ramp_plateau_60_20_20",
  "selection_clock_replicates": 4
}
```

The factory receives its `config` dictionary and returns the callback. The built-in runtime schemas are documented in [evaluators](evaluators.md). Factory
configuration paths are adapter-owned; the built-in factory resolves them from the working directory. Use absolute paths when invoking from another directory. Custom factories remain supported. Dry-run validates the
factory specification without importing the generator runtime. Evidence preparation
may produce a draft configuration without an evaluator; fitting stochastic students
requires it, while deterministic-only fitting does not.
The image CLI accepts the same factory/config object through `--selection-evaluator`.

## Artifacts and costs

Protocol v6 distinguishes `heldout_teacher_utility_kl_gate_v1` deterministic history
from `heldout_paired_terminal_utility_v1` measured stochastic history. New artifacts
record the combined `det_teacher_kl_gate_sto_measured_utility_v1` fitting criterion.
Deterministic records contain `predicted_utility`, validation KL, a prediction hash,
validation identities and the frozen-teacher fingerprint, never a measured-KID label.
The loader recomputes the KL gate and winner from the complete configured checkpoint
history and binds the selected student to its parameter/conditioning fingerprint.
It verifies the selected teacher weights, temperature and minimum-regret history.

Measured histories retain utility, measurement hash, panel identities and replicate
allowance. The v6 codec preserves serialized role keys and fingerprint inputs while
public selectors use `GICO-det-policy` and `GICO-sto-policy`. Existing valid measured
v6 student artifacts retain their recorded settings and remain readable, including
those written before the two deterministic settings were added. New deterministic
artifacts require teacher-selection proof. Public teacher reuse also requires that
proof. All v5 loading and older teacher replay require an archived runtime.

Only stochastic selection incurs new held-out generation/scoring at eligible
checkpoints. Report those costs separately from reference collection, optimization
and deterministic teacher-score evaluation. More checkpoint searches increase
selection bias; reuse of a selection panel is exploratory evidence, not a fresh
generalization result. Report actual terminal quality separately if measured; do
not present predicted utility as measured KID. A schedule comparison keeps teachers,
seeds, beta, temperatures, checkpoint cadence and panel allowance fixed.
