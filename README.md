# GenODE Inference

GenODE learns **GICO** (Generative Inference Clock Optimization) policies for
frozen flow-matching and ODE backbones. A GICO policy uses the frozen backbone's own context
and the solver budget to predict a continuous density over integration time.
Integrating and inverting that density produces a strictly increasing time grid
while the generative backbone remains frozen.

## Installation

GenODE requires Python 3.11 or newer. Install the package and its command-line
tools from a clone of this repository:

```bash
git clone https://github.com/pixelhero98/GenODE-Inference.git
cd GenODE-Inference
python -m pip install .
```

For an editable development installation with the test suite:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

The image runtime uses user-supplied external source trees and checkpoints; it
does not download or redistribute them. Review
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before using image assets.

## Supported GICO scenarios

The publication workflow covers eight scenarios: three temporal extrapolation
datasets, two image datasets, and three 3D molecular sets.

| Family | Scenario key | Conditioning | Supervision objective |
| --- | --- | --- | --- |
| Temporal | `solar_energy_10m` | Frozen backbone summary, plus its auxiliary conditioning when present | Equal-weight CRPS and MASE improvement over the uniform clock |
| Temporal | `traffic_hourly` | Frozen backbone summary, plus its auxiliary conditioning when present | Equal-weight CRPS and MASE improvement over the uniform clock |
| Temporal | `weather_daily` | Frozen backbone summary, plus its auxiliary conditioning when present | Equal-weight CRPS and MASE improvement over the uniform clock |
| Image | CIFAR-10 | Explicit singleton zero context; labels are rejected | Authenticated precomputed schedule-mixture evidence |
| Image | ImageNet-64 | Native 1,000-class RF++/1-RF label embedding | KID improvement over the uniform clock |
| 3D molecule | `molecule_3d_set1` | Frozen backbone policy context | Weighted geometric and kinematic improvement over the uniform clock |
| 3D molecule | `molecule_3d_set2` | Frozen backbone policy context | Weighted geometric and kinematic improvement over the uniform clock |
| 3D molecule | `molecule_3d_set3` | Frozen backbone policy context | Weighted geometric and kinematic improvement over the uniform clock |

The image registry supports RF++ Config G and EDM VE as 1-RF for CIFAR-10,
and RF++ Config E and EDM VE as 1-RF for ImageNet-64. Both image datasets
support the deterministic and stochastic students described below.

## Conditioning, reward, and training contracts

### Temporal and 3D scenarios

`frozen_backbone_policy_context_v1` is the projected summary consumed by the
frozen field:

```text
policy_context = cache.summary
policy_context = concat(cache.summary, cache.cond_emb)  # with auxiliary conditioning
```

The backbone is frozen and in evaluation mode. Contexts are finite, detached,
width-checked, and produced without gradients. Raw encoder summaries are not
selectable policy inputs.

Each metric is expressed as a log improvement over the uniform schedule. For
the temporal CRPS/MASE rows, repeated seeds are averaged within each
solver/NFE/schedule cell before reward construction:

```text
lower-is-better utility = log((uniform + eps) / (candidate + eps))
higher-is-better utility = log((candidate + eps) / (uniform + eps))
```

The three temporal scenarios use lower-is-better CRPS and MASE with weights
`0.5, 0.5`. Each 3D set uses lower-is-better Kabsch RMSD with weight `0.40`,
plus ensemble velocity W1, ensemble acceleration W1, rollout velocity W1, and
rollout acceleration W1 with weight `0.15` each.

For all six scenarios, the metric-vector teacher is trained with pairwise
ranking and masked Huber regression. The configured metric weights scalarize
its outputs, and its scores define a soft mixture over the fixed schedule
densities. The continuous-density student minimizes cross-entropy/KL to that
mixture, with the late-ramped clipped teacher-score term when enabled. Context
holdouts and density-family holdouts are used for model selection; locked-test
rows are never used to select or train the teacher or student.

### Image scenarios

ImageNet-64 uses the frozen model's native
`native_model.model.map_label` representation and binds the 1,000-row context
table to the backbone and checkpoint identities. CIFAR-10 is unconditional:
its supervision contains one explicit all-zero context, and supplying a class
label is an error. Contexts are never manufactured, truncated, or remapped.

Both image students consume one validated `ImageGICOSupervision` law that
binds NFEs 2, 4, and 8 to the fixed schedule support, mixture weights, density
barycenter, context table, diagnostics, and semantic source identities.

For ImageNet-64, KID is minimized and the raw advantage is
`uniform KID - schedule KID`. Jackknife standard errors feed
class-to-feature-group-to-global shrinkage, followed by reward-scale
normalization and clipping to `[-5, 5]`. Exact density aliases are aggregated
before softmax; their probability is then split consistently across duplicate
schedules. The same mixture weights define the deterministic density
barycenter. CIFAR-10 starts from authenticated precomputed mixture evidence, so
the public constructor does not invent class labels or synthesize rewards.

| Student kind | Training objective | Published default |
| --- | --- | --- |
| `deterministic_barycenter` | Target-density KL, centered residual penalty, and a late-ramped clipped teacher-score term for conditional training | Conditional model with 256 hidden dimensions |
| `stochastic_causal_ar` | Terminal-weighted complete-path NLL over strict prefix-trie support | 128 model dimensions, 256-token vocabulary, four heads, 192-dimensional FFN, one Transformer block, 16-dimensional NFE embedding (339,184 parameters) |

For CIFAR-10, the deterministic artifact binds the authenticated barycenter
directly, while the stochastic student learns the authenticated path mixture.
For ImageNet-64, the deterministic teacher/student objective and the stochastic
path objective are trained from the same conditional supervision law.

The stochastic policy uses 63 actions, 64 density bins, endpoint-aware
cube-companded tokens, and a maximum clock-node quantization drift below
`0.005`. At inference it samples one complete supported path from either
caller-supplied uniforms or the replayable SHA-256 counter RNG. The
deterministic policy materializes its barycenter directly. Inference uses
neither rewards nor the teacher; it freezes the complete time grid before Euler
integration and performs exactly the requested number of field evaluations.

## GICO workflow: temporal and 3D

The same four-stage workflow applies to each temporal or molecular scenario.
Use `--help` on any command to see the complete input schema.

1. Run a dry check of the backbone and schedule pipeline, then run it without
   `--dry_run` when its paths and configuration are correct.

   ```bash
   genode-run-full-pipeline --scenario_key traffic_hourly --dry_run
   genode-run-full-pipeline --scenario_key traffic_hourly
   ```

2. Check that the schedule rows have complete contexts, fixed support, and
   reward columns before training.

   ```bash
   genode-preflight-gico-rows \
     --rows_csv rows.csv \
     --report_json artifacts/gico/preflight.json \
     --complete_rows_csv artifacts/gico/complete-rows.csv
   ```

3. Validate the resolved training configuration, then train the teacher and
   student.

   ```bash
   genode-train-gico \
     --rows_csv artifacts/gico/complete-rows.csv \
     --context_embeddings_npz contexts.npz \
     --out_dir artifacts/gico/model \
     --dry_run

   genode-train-gico \
     --rows_csv artifacts/gico/complete-rows.csv \
     --context_embeddings_npz contexts.npz \
     --out_dir artifacts/gico/model
   ```

4. Evaluate the frozen student only after training and selection are complete.
   Locked-test reporting requires an explicit uniform baseline.

   ```bash
   genode-report-gico-locked-test \
     --gico_student_checkpoint artifacts/gico/model/gico_student.pt \
     --training_summary artifacts/gico/model/gico_training_summary.json \
     --context_rows locked-contexts.csv \
     --context_embeddings_npz locked-contexts.npz \
     --baseline_rows uniform-baseline.csv \
     --out_dir artifacts/gico/locked-report
   ```

Use one output directory per scenario and keep calibration, validation, and
locked-test inputs separate. `genode-run-schedules --help` documents the
lower-level schedule runner when the full pipeline wrapper is not appropriate.

## GICO workflow: images

`genode-image-gico` provides one portable lifecycle for both image datasets and
both student kinds:

```bash
# 1. Build the shared supervision law.
genode-image-gico build-targets \
  --manifest inputs/targets.json \
  --output artifacts/supervision

# 2. Train either or both students from that exact law.
genode-image-gico train-deterministic \
  --supervision artifacts/supervision \
  --output artifacts/deterministic
genode-image-gico train-stochastic \
  --supervision artifacts/supervision \
  --output artifacts/stochastic

# 3. Validate lineage and artifact integrity.
genode-image-gico validate \
  --supervision artifacts/supervision \
  --deterministic artifacts/deterministic \
  --stochastic artifacts/stochastic

# 4. Materialize a schedule without reward evidence or a teacher.
genode-image-gico materialize \
  --student deterministic_barycenter \
  --artifact artifacts/deterministic \
  --target-nfe 4 \
  --context-indices 0 \
  --output artifacts/schedule
```

`build-targets` accepts a portable JSON manifest. Array paths are relative to
the manifest, must stay inside its directory, and must name numeric `.npy`
files loadable with `allow_pickle=False`. A conditional ImageNet-64 manifest
has this form:

```json
{
  "kind": "conditional_kid",
  "conditional_targets": "conditional-targets.json",
  "fixed_density_mass": "fixed-density-mass.npy",
  "normalized_contexts": "normalized-contexts.npy"
}
```

For CIFAR-10, set `kind` to `unconditional_mixture` and provide
`target_nfes`, `schedule_keys`, `fixed_density_mass`, `mixture_weights`, and a
nonempty `source_identities` object. The builder creates the required singleton
zero context.

Training configuration JSON is optional through `--config`. Stochastic
materialization requires either an explicit `--uniforms` NumPy array or a
`--request-sha256` with optional comma-separated `--sample-keys`. Publication
is additive: existing destinations are rejected, identities and hashes are
recorded, and absolute input paths are not stored. Run
`genode-image-gico <command> --help` for all options.

The same contracts are exported from `genode.gico`, including supervision
builders, both student trainers, strict artifact loaders, schedule
materialization, counter-uniform derivation, and exact-NFE Euler execution.
`ImageGICOStudentKind` is the source of truth for student selection.

## Default GICO reference clocks

The default supervision pool contains exactly 23 schedules: 12 base schedules
and the reversals of the 11 nonuniform schedules. Uniform is self-reversing.

| Family | Active keys |
| --- | --- |
| Uniform | `uniform` |
| AYS SD1.5 | `ays_sd15_native`, `ays_sd15_log_sigma` |
| GITS CIFAR-10 example | `gits_cifar10_native`, `gits_cifar10_log_sigma` |
| OTS linear VP | `ots_vp_linear_native`, `ots_vp_linear_log_sigma` |
| Late-p | `late_p_1p5`, `late_p_2`, `late_p_4`, `late_p_8` |
| FlowTS | `flowts_power_0p03` |

Extra late-p supervision is opt-in. Values must be finite and inside
`[1.5, 8]`; each adds both a base and reversed clock:

```bash
genode-train-gico \
  --rows_csv rows.csv \
  --context_embeddings_npz contexts.npz \
  --out_dir artifacts/gico/model \
  --extra_late_p_values 2.25,3,6
```

AYS and GITS use pinned published source nodes transferred deterministically to
GenODE NFE grids; this does not imply rerunning their upstream optimizers. OTS
uses pinned paired linear-VP tables, and FlowTS uses its released power-clock
formula. Registry records bind the source model, solver, coordinate, revision,
file, and license:

- [AYS constants in Diffusers](https://github.com/huggingface/diffusers/blob/50e7158093710f9c1b4ea9ff100137a91c9228f3/src/diffusers/schedulers/scheduling_utils.py)
- [Diffusers scaled-linear DDIM realization](https://github.com/huggingface/diffusers/blob/50e7158093710f9c1b4ea9ff100137a91c9228f3/src/diffusers/schedulers/scheduling_ddim.py)
- [GITS CIFAR-10 example](https://github.com/zju-pi/diff-sampler/tree/68d5ce427f261962b89ce3b0ee8f6b29f0577328)
- [OTS in DM-NonUniform](https://github.com/scxue/DM-NonUniform/blob/95d4ac6b8a3d1d389ab63a197e1b05d8512b6a99/step_optim.py)
- [FlowTS/FMTS sampler](https://github.com/UNITES-Lab/FlowTS/blob/1ec35fb1d3d89d91a1607a9f949a515347d54c8c/FMTS/Models/interpretable_diffusion/FMTS.py)

## Image sources and licensing

The image runtime registers four external backbones:

- unconditional CIFAR-10 RF++ Config G and EDM VE as 1-RF;
- class-conditional ImageNet-64 RF++ Config E and EDM VE as 1-RF.

Image source trees, datasets, feature weights, and checkpoints are supplied by
the user and remain under their upstream terms. The pinned RF++ registry records
the network implementation as `CC-BY-NC-SA-4.0` and notes that no separate
checkpoint license notice was found. Review
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before obtaining or using
external image assets.

## Consistency distillation (secondary)

The optional endpoint flow-map workflow is separate from primary GICO
training:

```bash
genode-collect-flow-map-demonstrations --help
genode-train-flow-map --help
genode-evaluate-flow-map --help
```

Flow-map checkpoints remain bound to the frozen backbone and GICO checkpoint
used to create them. Evaluation requires an explicit measurement protocol.

## Deterministic archives

Build checkpoint-only, named-checkpoint, or frozen policy archives with:

```bash
genode-build-release-archive backbone-manifest --help
genode-build-release-archive named-checkpoints --help
genode-build-release-archive gico-policy --help
genode-build-release-archive validate --archive release.zip
```

Each build writes a deterministic ZIP, a canonical external
`<archive>.manifest.json`, and an `<archive>.sha256` digest. Policy archives
contain the validated teacher, student, density table, context normalizers, and
portable GICO manifest; packaging does not retrain the policy.

## Development checks

```bash
python -m ruff check .
python -m ruff format --check .
python -m compileall -q src tests
python -m pytest -q
python -m pip check
git diff --check
```

The release matrix covers Python 3.11 and 3.13 with NumPy 1.26 and 2.x, builds
and inspects the wheel and source distribution, installs both in clean
environments, and smoke-tests every public CLI.

## License

GenODE is released under the [MIT License](LICENSE). Third-party code,
reference data, external model implementations, datasets, and checkpoint
weights remain subject to their respective terms.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidance. Report
vulnerabilities privately according to [SECURITY.md](SECURITY.md).
