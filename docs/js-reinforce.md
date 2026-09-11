# James--Stein REINFORCE comparator

This baseline is a documented reconstruction of Yu et al.,
[Designing Instance-Level Sampling Schedules via REINFORCE with James--Stein Shrinkage](https://arxiv.org/abs/2511.22177),
version 2. It is not author-provided code, a GICO student, or an exact reproduction
of the paper's backbone and reward experiments.

The policy follows Table 4: image-shaped initial noise, native text tokens,
optional pooled text, four convolution/cross-attention blocks, a feature pyramid,
and a two-layer Dirichlet head with softplus plus 0.001. GroupNorm uses eight
groups; cross-attention uses zero dropout. These unspecified numerical choices
are explicit implementation decisions. Native input dimensions follow the
backbone; report the actual parameter count rather than assuming 20 million.

Training uses a joint schedule action, two rollouts for each prompt/noise context,
and the empirical baseline in Eqs. (4)--(10). In Eq. (10), the context-level
cross-context quantity is implemented as the mean of that context's per-rollout
leave-one-out baselines. The zero-variance case has shrinkage zero. Rewards and
baselines are detached; log likelihood is taken over the complete Dirichlet
action. There is no PPO clipping, KL penalty, entropy term, or reward-based
selection at inference. The paper's AdamW settings are weight decay 0.0001 and
gradient norm clipping at 1; learning rate and finite trajectory budget must be
recorded by each experiment.

## Explicit schedule interpretation

The paper gives L+1 Dirichlet intervals, L interior times, and a skipped interval,
but also calls the last interval a stopping margin. Without the authors' sampler
code these descriptions do not uniquely specify execution. This implementation
uses the L interior times as model-evaluation times and appends terminal zero.
It skips the interval before the first evaluation, leaves the sampled initial
Gaussian unchanged, and executes exactly L Euler updates to zero. It does not
add an uncounted evaluation at the original initial endpoint.

For a raw interval vector q of length L+1, solver intervals are
`(1 - (L+1)*delta) * q + delta`, with `delta = 2**-20`. This declared numerical
guard preserves a strictly ordered float32 grid. Raw actions are retained and
their original Dirichlet likelihood is used for REINFORCE; training and inference
use the same deterministic realization. No draws are discarded or retried.

## Interpretation and fairness

The published empirical shrinkage coefficient uses the current batch rewards.
Although the component means exclude each scored rollout, detaching the resulting
baseline does not make its coefficient independent of that rollout. Accordingly,
this reconstruction does not claim that the empirical gradient is unbiased or
that the paper's theoretical MSE statements establish improved policy performance.
It reproduces the stated empirical equations rather than silently substituting
a different cross-fitted estimator.

Compare frozen backbone weights, guidance, resolution, actual NFE, terminal
utility, evaluation prompts and generation seeds. Count every training rollout
and reward call, plus separately disclosed calibration and validation costs.
Preserve noise conditioning and the original policy family for this comparator;
report its extra information and capacity relative to GICO. Such a comparison
tests the complete methods, not the optimizer in isolation. Report preliminary
validation results separately from locked-test benchmark results.

## SANA fitting and validation

Run `python -m genode.latent_clock.js_experiment fit --config experiment.json`,
then use `validation` with exactly the same configuration file. Both commands
use separate generator and scorer devices. Each output phase must be new;
interrupted runs retain their observation journal and must use a new output
directory. The runner does not automatically resume partially completed fitting.

Configuration requires paths for `runtime_config`, `manifest`, `anchor_evidence`
(prepared shared GICO rows), `anchor_measurements` (original scored rows),
`anchor_scoring` (their scoring metadata), `reward_calibration` (the frozen
shared `RewardCalibration` payload), `scorer_python`, and `output`. It also
records `nfe`, two `noise_seeds`, and `fit`: `trajectories`, `anchor_trajectories`,
`seed`, `batch_contexts`, `rollouts`, and `learning_rate`.

Anchor reuse requires `audited_implementations`, an ordered pair of exact source
hashes for the measured anchors and this runtime. Audit the numerical generator
path before allowing a source pair; matching weights alone is insufficient.
Prompt/reference identities, native embeddings, solver settings, exact uniform
grids, and live scorer fingerprints/versions must also match. Both source hashes
remain in the artifact. Validation requires the same configuration, input file
hashes and runtime binding as fitting.

One uniform anchor is charged for each training prompt/noise context. Remaining
trajectories must fill complete B-by-K batches. Rewards use the same frozen paired
ImageReward/VQAScore calibration as GICO, including its scalar reward scale.
For example, 128 anchors plus 48 batches of 32 contexts and two rollouts equals
3,200 trajectories. Calibration costs are additional and reported separately.
There is no running reward normalization or teacher-based JS objective.

The `genode_js_reinforce_v1` artifact stores native architecture dimensions,
solver interpretation, numerical interval guard, immutable experiment identity,
training splits/seeds, reward/scorer provenance, runtime fingerprints and training
history. Loading verifies a manifest digest and uses a weights-only reader.
Ordinary inference loads only the JS policy, supplies native inputs, draws once
with `sample_intervals`, and calls `materialize_js`; rewards are never consulted.
