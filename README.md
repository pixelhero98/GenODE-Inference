# GenODE Inference

GenODE learns inference clocks for frozen generative backbones. A shared GICO teacher supervises either a deterministic density policy or an autoregressive stochastic density policy. Each task/backbone has separately trained weights. The generative backbone stays frozen.

## Install

Python 3.11 or newer is required.

```bash
git clone https://github.com/pixelhero98/GenODE-Inference.git
cd GenODE-Inference
python -m pip install -e ".[test]"
python -m pytest -q
```

The optional `latent-clock` extra supplies Bayesian-optimization dependencies. Image generators, text-to-image scorers, and pretrained weights are external assets; supply their source revisions and checkpoints explicitly. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Tasks and rewards

| Task | Frozen native context | Terminal training reward |
| --- | --- | --- |
| `solar_energy_10m`, `traffic_hourly`, `weather_daily` | Pooled backbone summary, including native auxiliary conditioning | Equal-weight CRPS and MASE log improvements |
| `molecule_3d_set1`, `molecule_3d_set2`, `molecule_3d_set3` | Pooled backbone summary of observed history | Joint-trajectory ensemble energy-score log improvement |
| `cifar10` | Explicit zero vector | Uniform KID minus candidate KID |
| `imagenet64` | Native class embedding | Equally weighted class-conditional paired KID improvement |
| `sana`, `sd15` | Pooled native text embedding | Equal-weight ImageReward and VQAScore differences, divided by frozen pilot component scales |

Pair each candidate and uniform anchor on context, backbone, solver, NFE, generation seed, ensemble size, reference data, and measurement protocol. KID uses paired sample blocks. Average repeated terminal measurements within each comparison cell **before** computing log improvements.

Positive errors use `log((anchor + epsilon) / (candidate + epsilon))`, where each frozen numerical floor is `1e-6 * median(positive uniform calibration values)`. Reject degenerate calibration. After component scalarization, divide by a single frozen reward standard deviation for each task/backbone/solver, balancing calibration contributions across training NFEs. Do not subtract a mean or use per-context or running normalization. Uniform rewards remain exactly zero. Text-to-image component scales use pilot measurements; its scalar scale uses training measurements.

ImageNet retains paired-jackknife, class-to-feature-group-to-global shrinkage estimated only from training/calibration evidence. Report unshrunk class-conditional KID separately from measured global KID/FID. Global metrics cannot be reconstructed from class KID alone.

The molecular primary score is the fair finite-ensemble energy estimator:

```text
mean_m ||phi(X_m) - phi(y)||
  - sum_{m != n} ||phi(X_m) - phi(X_n)|| / (2 M (M - 1))
```

It requires at least two independently generated complete trajectories. The frozen feature map preserves atom indices and horizon order, includes indexed pair distances and signed volumes relative to a non-collinear reference triangle, and uses deterministic atom-index tie breaking. Lengths and volumes use the training reference RMS pair distance and its cube; feature blocks use dimension normalization. The observed future is one observation, never a fabricated reference ensemble. Kabsch RMSD, motion discrepancies, clashes, and bond violations remain diagnostics.

These are optimization objectives. Log transformations, mixtures of components, and teacher approximations do not establish proper scoring or benchmark improvement for the learned policy.

## Shared architecture and optimization

All roles use a two-layer, width-128, four-head, pre-normalized Transformer with a 256-wide feed-forward block and zero dropout. Conditioning combines native context, solver identity, and continuous NFE/macro-step features. Feature normalization is fitted on training data and frozen. Initial generation noise is never a policy input.

* The teacher consumes conditioning and a candidate density and predicts the normalized metric-improvement vector. Its objective combines within-context/settings pairwise ranking (temperature 0.5) with weighted Huber regression (weight 0.25).
* The deterministic student uses 64 density-bin queries. It minimizes KL from the teacher-weighted reference-density barycenter, plus a teacher-score term.
* The stochastic student predicts 63 Gaussian log-density ratios autoregressively. Training-reference ratios are standardized; likelihood targets receive Gaussian noise with standard deviation 0.1. Predicted standard deviations are bounded to [0.05, 2]. The likelihood averages over all 63 coordinates. A reparameterized teacher-score term also trains this student.

Both students use a uniform prior over unique realized reference densities and teacher softmax temperature 1. Stochastic target smoothing is an additional modeling choice. Teacher-score weights are **0.01, 0.05, 0.1**, default 0.01. The weight ramps linearly from zero after 60% of training; normalized teacher scores are clipped to [-5, 5]. Teacher parameters remain frozen while gradients pass through density inputs.

The shared pool has 25 reference clocks, including late-p=3 and its reversal. Every reference is materialized through the same 64-bin representation as student outputs. Identical densities are deduplicated before mixture weighting. Historical evidence is reusable only if executed grids and measurement protocols match exactly; changed grids require new measurements.

## Train and decode

The common interface accepts JSON configuration:

```json
{
  "rows": "measurements.jsonl",
  "contexts": "contexts.npz",
  "calibration_rows": "calibration.jsonl",
  "output": "policy",
  "student_kind": "both",
  "teacher_score_weight": 0.01,
  "steps": 2000,
  "batch_size": 32,
  "seed": 0,
  "device": "cuda",
  "purpose": "research"
}
```

Paths are relative to the configuration file. Each measurement row contains `task`, `backbone`, `solver`, integer `nfe`, `context_id`, explicit `split`, integer `seed`, `ensemble_size`, `reference_id`, `measurement_protocol`, `schedule_key`, `metrics`, 64 `density_mass` entries, and the executed `time_grid`. Metric keys are `crps/mase`, `energy_score`, `kid`, or `preference/alignment`. Training input contains disjoint `train` and `validation` contexts; calibration contains only `train` or `calibration`. Locked-test rows are forbidden during fitting. Store native contexts with `save_context_embedding_table`.

Research molecular rows also carry the frozen `molecule_feature_map` dictionary from `MoleculeFeatureMap.to_dict()`. It is recorded in the policy artifact and checked against runtime reference geometry.

```bash
genode-train-gico --config train.json --dry-run
genode-train-gico --config train.json --student-kind both --teacher-score-weight 0.01
```

Research evidence requires all 25 references in every cell. Explicit `purpose: functional` permits a reduced reference set for integration checks; it does not produce benchmark evidence. Checkpoints are selected using validation evidence, including a teacher density-family holdout. Output directories must be new.

```python
from genode.gico.policy import load_policy

policy = load_policy("policy", student_kind="stochastic", expected_backbone="checkpoint-sha")
grid = policy.materialize(native_context, "euler", 8, seed=412, request_id="example:member:0")
```

Both students use one density-to-clock implementation. It mixes in exactly 1e-8 uniform density before inverse-CDF conversion, validates representable solver grids, and enforces exact NFE accounting. The clock RNG is separate from generation noise. Sample once per generated image or trajectory; reuse each molecular member's clock throughout its rollout. Inference performs no teacher scoring, rejection, or reward-based selection.

Forecast and molecular Python evaluators accept `policy` and `clock_seed`. Forecast policy evaluation uses batch size 1, paired with the same uniform protocol. Molecular policy evaluation conditions on the initial observed history and keeps the sampled clock throughout every horizon.

## Image and latent-image workflows

CIFAR/ImageNet preparation validates paired feature-block evidence and native backbone/context bindings:

```bash
genode-image-gico prepare --manifest raw.json --output evidence.json
genode-image-gico train --evidence evidence.json --output policy --student-kind both --teacher-score-weight 0.01
genode-image-gico validate --help
genode-image-gico materialize --help
```

The input schema is documented by `prepare_image_rows` in `genode.gico.image_supervision`. Supply global metric measurements separately when reporting them.

SANA/SD1.5 collection records executed grids, density masses, native contexts, generation seeds, and scorer identities. Use `genode-latent-clock prepare-gico --help` to convert paired collection evidence and independent pilot rows into the common configuration, then:

```bash
genode-latent-clock fit-gico --config evidence/train_config.json --student-kind both
genode-latent-clock prepare-collection --help
```

BO, PG, and LD3 remain separate comparison methods. Completed experiments remain historical results; this architecture change does not relabel or upgrade their policy states.

## Artifacts and validation

Protocol `genode-gico-v2` stores `policy.pt` plus a checksummed `manifest.json`. Artifacts record architecture, reward calibration, context normalization, reference densities and executed grids, split identities, solver semantics, RNG configuration, and fitting history. Incompatible old artifacts are rejected; there is no legacy architecture loader.

`genode-report-gico-locked-test` applies an artifact's frozen calibration to paired test measurements without selection. `genode-evaluate-schedule-summary` performs the analogous validation report. Both require new output files and matching frozen measurement protocols, native backbone bindings, and molecular feature maps. Supply `policy_sha256` and `student_kind` for learned-policy measurements.

Evaluation rows can carry a fixed `density_mass`/`time_grid` pair or `sample_clocks`, a list containing one such pair per ensemble member. This supports independently sampled stochastic clocks while averaging repeated terminal measurements before constructing log improvements. Forecast evaluators export `sample_clocks`; molecular evaluators export the same pairs with rollout provenance in `sample_clock_records`. Training reference clocks remain fixed across repeats.

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
python -m pip check
python -m build
git diff --check
```

Tests cover paired rewards, split isolation, geometric energy scoring, causal stochastic sampling, teacher-input gradients, density decoding, solver accounting, artifact integrity, and active task routing. External-asset functional checks and their environment-specific instructions belong outside the public package. Fixture coverage alone does not validate a pretrained generator or establish quality gains.
