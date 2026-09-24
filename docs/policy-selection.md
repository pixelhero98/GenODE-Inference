# Utility surrogate and policy selection

All collection completes first. Utility surrogate fitting and both policy selectors consume saved measurements or frozen-utility surrogate predictions; they never call a generator or terminal scorer. Stochastic training is disabled by default in Python, CLIs and preparation commands. Use `stochastic` or `both` explicitly.

## Utility surrogate

Reserve `ceil(0.2 * unique_nonuniform_density_count)` identities for density validation. Equivalent aliases remain together and uniform stays available as the anchor. The shared collection manifest fixes this split across settings. Utility surrogate fitting excludes held-out contexts and held-out densities. Calibration, native-context normalization and stochastic ratio normalization use eligible fitting evidence only.

Evaluate utility-surrogate checkpoints every 20 steps. For every held-out measured group, subtract the predicted uniform-anchor vector from each nonuniform prediction before comparing it with measured paired gain. Square each component error, apply the task's frozen metric weights, restore its frozen scalar reward scale, and average candidates within a context. The context-only and density-only holdout means receive equal weight; intersection observations are excluded. The checkpoint with the lowest balanced component MSE wins, with earlier steps breaking exact ties. Research fits require both holdout axes.

At the selected checkpoint, choose among temperatures 0.05, 0.1 and 0.5 by balanced measured reference-mixture regret:

```text
w_j(T) = softmax_j(a * s_j / T)
regret(T) = max_j U_j - sum_j w_j(T) * U_j
```

Here `s_j` is normalized predicted scalar utility, `a` is the frozen scalar reward scale, and `U_j` is measured calibrated utility before scalar reward normalization. Temperature ties prefer 0.05, then the lower candidate temperature. The density holdout uses fitting contexts' held-out candidates with their paired uniform anchors. The context holdout uses its non-held-density candidates and uniform anchor.

The opt-in `density_context_projection` profile uses 500 density-only updates with pooled fitting targets, followed by 1,500 updates to the conditioning projection alone. It uses component-wise Huber weight 1.0, no ranking stage, and one optimizer across the boundary. A selected warm-up checkpoint retains its density-only input mask. Complete aligned fitting density support is required; public collection budgets do not change.

After selection the utility surrogate is frozen. Both policies receive its predictions over the **complete unique reference-density pool**, including unmeasured context/density combinations. These are predictions, never fabricated measurement rows.

## Deterministic policy

Checkpoint every ten steps. Only checkpoints with a positive realized score-chasing coefficient are eligible. After the complete training horizon, compute the minimum eligible held-out KL and admit checkpoints with:

```text
KL(target density barycenter || policy density) <= 1.15 * final minimum eligible KL
```

Among admitted checkpoints, maximize raw calibrated frozen-utility surrogate utility, preferring the earlier checkpoint on exact ties. Compute one policy density and one utility surrogate evaluation per context/solver/NFE; reuse that density for the KL calculation. Restore the frozen scalar reward scale before aggregating utilities across settings. Contexts/classes and settings receive equal weight.

## Stochastic policy

Checkpoint every 100 steps, with its own selector and `stochastic_kl_allowance: 0.20`. Use the same positive-chasing eligibility rule, then admit:

```text
KL(smoothed utility surrogate mixture || autoregressive policy) <= 1.20 * final minimum eligible KL
```

The utility surrogate distribution is the full mixture of Gaussian-smoothed, standardized 63-dimensional reference log-ratio vectors. Draw 32 fixed samples from every reference component and evaluate the **joint** log probabilities over all coordinates:

```text
l = log p(z) - log q(z)
KL estimate = sum_j w_j * mean_{z from component j}(l + exp(-l) - 1)
```

The likelihood-ratio estimator is nonnegative for every sample. `log p` uses the complete weighted mixture; `log q` uses autoregressive conditional parameters. Training's coordinate-averaged NLL and deterministic barycenter KL are different quantities and are not stochastic selection criteria.

Components whose weights underflow to exactly zero contribute neither probability nor samples to the estimate. Every positive-weight component is retained, however small its weight; no probability threshold or extra smoothing is applied.

Rank admitted checkpoints by expected calibrated frozen-utility surrogate utility from four fixed full-policy draws per context/setting. Prefer earlier ties. Target samples and policy draws use independent explicit RNG streams, repeat identically at each checkpoint, and do not consume training or generation RNG state. Restore each solver's scalar reward scale before aggregation. Auxiliary standardization and clipping are training-only.

## Shared training and artifact contract

Defaults remain 2,000 utility surrogate/policy steps, utility surrogate batch cap 64, policy cap 512, microbatch eight, and task-specific dropout and fixed refinement weight 0.05. The first 60% of policy training is pure distillation. The default `linear_60_40` schedule ramps score chasing over the remainder; `ramp_plateau_60_20_20` and `constant_60_40` remain explicit choices. At least one positive-chasing checkpoint must exist.

The common checkpoint-retention utility discards states that cannot pass the running minimum gate; that gate can only tighten. Final selection uses the full eligible history. Utility surrogate and policy selection proofs bind source, collection, evidence, support, calibration, conditioning, selected utility surrogate, history and selected weights. Artifacts with missing or incompatible proof fail clearly; historical artifacts require their archived runtime.

Generator-backed measurements remain available for collection and explicit reporting in `genode.gico.evaluators` and `genode.gico.reporting`. They are absent from `fit`, `fit_models`, and training configuration. Reports do not alter the selected policy.
