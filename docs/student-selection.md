# Teacher and student selection

All collection completes first. Teacher fitting and both student selectors consume saved measurements or frozen-teacher predictions; they never call a generator or terminal scorer. Stochastic training is disabled by default in Python, CLIs and preparation commands. Use `GICO-sto-policy` or `both` explicitly.

## Teacher

Reserve `ceil(0.2 * unique_nonuniform_density_count)` identities for density validation. Equivalent aliases remain together and uniform stays available as the anchor. The shared collection manifest fixes this split across settings. Teacher fitting excludes held-out contexts and held-out densities. Calibration, native-context normalization and stochastic ratio normalization use eligible fitting evidence only.

Evaluate teacher checkpoints every 20 steps and temperatures 0.05, 0.1 and 0.5. For each observed comparison group, use the unique measured references in that group. If the teacher predicts normalized scalar utility `s_j` and the frozen reward scale is `a`, mixture weights are:

```text
w_j(T) = softmax_j(a * s_j / T)
regret(T) = max_j U_j - sum_j w_j(T) * U_j
```

`U` is measured calibrated utility before scalar reward normalization. Selection minimizes the equally weighted mean of context-holdout regret and density-holdout regret. Density holdout uses fitting contexts' held-out candidates together with their paired uniform anchors. Context holdout uses all its measured references. Ties prefer the configured preferred temperature (default 0.05), then the earlier checkpoint.

After selection the teacher is frozen. Both students receive its predictions over the **complete unique reference-density pool**, including unmeasured context/density combinations. These are predictions, never fabricated measurement rows.

## Deterministic student

Checkpoint every ten steps. Only checkpoints with a positive realized score-chasing coefficient are eligible. After the complete training horizon, compute the minimum eligible held-out KL and admit checkpoints with:

```text
KL(target density barycenter || policy density) <= 1.15 * final minimum eligible KL
```

Among admitted checkpoints, maximize raw calibrated frozen-teacher utility, preferring the earlier checkpoint on exact ties. Compute one policy density and one teacher evaluation per context/solver/NFE; reuse that density for the KL calculation. Restore the frozen scalar reward scale before aggregating utilities across settings. Contexts/classes and settings receive equal weight.

## Stochastic student

Checkpoint every 100 steps, with its own selector and `stochastic_kl_allowance: 0.20`. Use the same positive-chasing eligibility rule, then admit:

```text
KL(smoothed teacher mixture || autoregressive policy) <= 1.20 * final minimum eligible KL
```

The teacher distribution is the full mixture of Gaussian-smoothed, standardized 63-dimensional reference log-ratio vectors. Draw 32 fixed samples from every reference component and evaluate the **joint** log probabilities over all coordinates:

```text
l = log p(z) - log q(z)
KL estimate = sum_j w_j * mean_{z from component j}(l + exp(-l) - 1)
```

The likelihood-ratio estimator is nonnegative for every sample. `log p` uses the complete weighted mixture; `log q` uses autoregressive conditional parameters. Training's coordinate-averaged NLL and deterministic barycenter KL are different quantities and are not stochastic selection criteria.

Rank admitted checkpoints by expected calibrated frozen-teacher utility from four fixed full-policy draws per context/setting. Prefer earlier ties. Target samples and policy draws use independent explicit RNG streams, repeat identically at each checkpoint, and do not consume training or generation RNG state. Restore each solver's scalar reward scale before aggregation. Auxiliary standardization and clipping are training-only.

## Shared training and artifact contract

Defaults remain 2,000 teacher/student steps, teacher batch cap 64, student cap 512, microbatch eight, and task-specific dropout and score weights. The first 60% of student training is pure distillation. The default `linear_60_40` schedule ramps score chasing over the remainder; `ramp_plateau_60_20_20` and `constant_60_40` remain explicit choices. At least one positive-chasing checkpoint must exist.

The common checkpoint-retention utility discards states that cannot pass the running minimum gate; that gate can only tighten. Final selection uses the full eligible history. Teacher and student selection proofs bind source, collection, evidence, support, calibration, conditioning, selected teacher, history and selected weights. Artifacts with missing or incompatible proof fail clearly; historical artifacts require their archived runtime.

Generator-backed measurements remain available for collection and explicit reporting in `genode.gico.evaluators` and `genode.gico.reporting`. They are absent from `fit`, `fit_models`, and training configuration. Reports do not alter the selected policy.
