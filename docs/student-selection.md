# Student selection and score schedules

Training minimizes distillation minus the weighted differentiable teacher score.
Selection maximizes the **measured calibrated terminal utility** on the complete
held-out panel. Distillation and teacher predictions are diagnostics. Exact utility
ties select the earlier checkpoint. No test or comparison-panel measurements enter
selection, and no reward selection occurs at inference.

All tasks and both student kinds share this rule. Task-specific metric weights,
calibration, temperature, maximum beta, horizons and conditioning remain unchanged.

| `score_schedule` | First 60% | Next 20% | Final 20% |
|---|---|---|---|
| `linear_60_40` (default control) | Zero | Linear ramp to half beta | Continue ramp to full beta |
| `ramp_plateau_60_20_20` | Zero | Linear ramp to full beta | Full beta |
| `constant_60_40` | Zero | Full beta immediately | Full beta |

Checkpoints are eligible when their realized coefficient is positive. Eligibility
does not require the ramp to finish. The final update is always checkpointed.
The default schedule remains the historical control pending measured comparisons.

## Evaluator contract

Python `fit` and `fit_models` require `selection_evaluator(candidate)`. It receives
a detached CPU `StudentCandidate` in evaluation mode and returns raw paired
terminal-measurement rows. Its `density` and `materialize` methods use the same
inference decoder as deployed policies. There is no teacher-prediction fallback.
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
and replicates within a comparison cell. Pair their identities across checkpoints.
Deterministic selection uses one replicate, with the prescribed task ensemble size.

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
    "factory": "my_task.selection:build_evaluator",
    "config": {"panel": "/absolute/path/to/selection-panel.json"}
  },
  "score_schedule": "ramp_plateau_60_20_20",
  "selection_clock_replicates": 4
}
```

The factory receives its `config` dictionary and returns the callback. Factory
configuration paths are adapter-owned; use absolute paths. Dry-run validates the
factory specification without importing the generator runtime. Evidence preparation
may produce a draft configuration without an evaluator; actual fitting requires it.
The image CLI accepts the same factory/config object through `--selection-evaluator`.

## Artifacts and costs

Protocol v6 records the schedule, selected utility, measurement hash, panel identities,
replicate allowance and a fingerprint binding the selected parameters and conditioning.
It verifies that the saved student is the highest-utility eligible history entry.
Teacher-only reuse of v5 is supported with its original profile validation; v5
students remain accessible only through their originating runtime and are not relabelled.

Every eligible checkpoint incurs held-out generation/scoring access. Report these
trajectories, their backbone evaluations and wall time separately from reference
collection and optimizer fitting. More checkpoint searches increase selection bias;
an already inspected selection panel provides exploratory evidence, not a fresh
generalization result. A schedule comparison keeps teachers, seeds, beta, temperature,
checkpoint cadence and panel allowance fixed and stops at the registered horizon.
