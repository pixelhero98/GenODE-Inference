# GenODE Inference

GenODE learns **GICO** (Generative Inference Clock Optimization) policies for
frozen flow-matching and ODE backbones. GICO predicts a continuous-density
integration clock from the solver budget and the frozen backbone's own context; the
density is inverted into a strictly increasing time grid.

The primary package covers frozen backbones, reference clocks, GICO
teacher/student training, locked-test evaluation, and deterministic release
archives. One-evaluation consistency distillation remains available as a
secondary, opt-in workflow.

## Guarantees

- GICO is the only active policy name, import path, CLI namespace, and
  checkpoint protocol. There are no aliases for the former name.
- The default supervision pool contains exactly 23 clocks: 12 base clocks and
  the reversals of the 11 nonuniform clocks. Uniform is self-reversing.
- Conditional policies use the exact context space of the frozen field.
  CIFAR-10 image backbones are explicitly unconditional and reject labels.
- Checkpoint loaders are strict about schema version, tensor names, shapes,
  dtypes, finite values, solver/NFE semantics, and locked-test exclusion.
- Release ZIPs are byte-deterministic and contain canonical manifests and
  SHA-256 sidecars. Source paths, symlinks, reparse points, traversal, and
  case/Unicode member collisions are rejected.
- Datasets, generated outputs, private paths, and third-party image weights are
  not stored in this repository.

## Installation

GenODE requires Python 3.11 or newer.

```bash
python -m pip install -e .
```

For development or raw medical-data preparation:

```bash
python -m pip install -e ".[test]"
python -m pip install -e ".[medical]"
```

## Default GICO reference clocks

The canonical base order is:

| Family | Active keys |
| --- | --- |
| Uniform | `uniform` |
| AYS SD1.5 | `ays_sd15_native`, `ays_sd15_log_sigma` |
| GITS CIFAR-10 example | `gits_cifar10_native`, `gits_cifar10_log_sigma` |
| OTS linear VP | `ots_vp_linear_native`, `ots_vp_linear_log_sigma` |
| Late-p | `late_p_1p5`, `late_p_2`, `late_p_4`, `late_p_8` |
| FlowTS | `flowts_power_0p03` |

Each nonuniform key also has a `_reversed` counterpart. Extra late-p
supervision is opt-in and must be finite and inside `[1.5, 8]`; both the base
and reversed clocks are added:

```bash
genode-train-gico \
  --rows_csv rows.csv \
  --context_embeddings_npz contexts.npz \
  --out_dir outputs/gico \
  --extra_late_p_values 2.25,3,6
```

AYS and GITS start from their pinned published source nodes. The GenODE NFE
grids are deterministic, coordinate-preserving transfers of those nodes, not
claims that the upstream optimizers were rerun for a GenODE backbone. The AYS
log-sigma terminal uses the pinned SD1.5 scaled-linear scheduler realization
(`1000` steps, `beta_start=0.00085`, `beta_end=0.012`) so the terminal is finite.
OTS uses immutable paired `t_res` and `lambda_res` tables produced by the pinned
official linear-VP implementation with its float32 initialization semantics.
The supported source step counts (used as GenODE macro steps) are exactly
`2,3,4,5,6,7,8,10,12,14,16,20`; other counts are rejected, so runtime SciPy
versions cannot change either coordinate view. The exact raw upstream endpoints
and both vector families are bound by the versioned table identity. FlowTS uses
the released power-clock formula and exponent. Every catalog record is generated
from the runtime registry and includes the exact source model, solver,
coordinate, commit, file, and license:

- [AYS constants in Diffusers](https://github.com/huggingface/diffusers/blob/50e7158093710f9c1b4ea9ff100137a91c9228f3/src/diffusers/schedulers/scheduling_utils.py)
- [Diffusers scaled-linear DDIM realization](https://github.com/huggingface/diffusers/blob/50e7158093710f9c1b4ea9ff100137a91c9228f3/src/diffusers/schedulers/scheduling_ddim.py)
- [GITS CIFAR-10 example](https://github.com/zju-pi/diff-sampler/tree/68d5ce427f261962b89ce3b0ee8f6b29f0577328)
- [OTS in DM-NonUniform](https://github.com/scxue/DM-NonUniform/blob/95d4ac6b8a3d1d389ab63a197e1b05d8512b6a99/step_optim.py)
- [FlowTS/FMTS sampler](https://github.com/UNITES-Lab/FlowTS/blob/1ec35fb1d3d89d91a1607a9f949a515347d54c8c/FMTS/Models/interpretable_diffusion/FMTS.py)

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and
license details.

## Conditioning contract

For temporal, conditional-generation, and molecular backbones,
`frozen_backbone_policy_context_v1` is the projected summary actually consumed
by the frozen field:

```text
policy_context = cache.summary
policy_context = concat(cache.summary, cache.cond_emb)  # when auxiliary conditioning exists
```

The backbone must be frozen and in evaluation mode. Exported contexts are
finite, detached, width-checked, and produced without gradients. Raw encoder
summaries are never selectable policy inputs.

For ImageNet-64, GICO uses the frozen RF++/1-RF model's native
`native_model.model.map_label` representation and binds the policy to the
backbone/checkpoint/context identities. Both registered CIFAR-10 backbones are
unconditional; supplying a label is an error.

### Complete-clock stochastic decoding

ImageNet-64 policies can add a source-bound complete-clock sidecar. For each
class context and NFE, the decoder predicts probabilities `q` over whole clocks
from the supervised pool. Its deterministic schedule is the density barycenter
`mu = sum(q_i * d_i)`. A caller may also supply one replayable uniform draw and
`alpha` in `[0, 1]` to execute
`D_alpha = (1 - alpha) * mu + alpha * d_I`. Thus `alpha=0` is exactly the
barycenter, `alpha=1` is exactly one supervised complete clock, and intermediate
values preserve the mean while scaling covariance by `alpha^2`.

The default 23-clock pool is represented on the exact union of all NFE 2, 4,
and 8 clock nodes (171 nodes, 170 density bins), so complete clocks reconstruct
without finite-bin drift. Duplicate supervision aliases are trained as grouped
categorical targets and split uniformly inside each group. Randomness is never
created inside the policy: SHA-256 counter draws are explicitly bound to the
sampling plan, policy artifact, clock library, frozen context binding, seed,
class label, and NFE; `alpha` is deliberately excluded so comparisons use the
same selected clock.

The six-file sidecar contains only its manifest, model state, clock-library
identity, and exact NFE 2/4/8 clock grids. It carries neither raw/normalized
contexts nor copied conditional targets. Loading requires the original GICO
source artifact, and execution requires rebinding that source to its verified
frozen backbone so contexts can only be derived from class labels.

## Primary workflow

Run or resume the canonical backbone and schedule pipeline:

```bash
genode-run-full-pipeline --scenario_key traffic_hourly --dry_run
genode-run-schedules --help
```

Preflight support rows, train GICO, and report a frozen student:

```bash
genode-preflight-gico-rows --help
genode-train-gico --help
genode-report-gico-locked-test \
  --gico_student_checkpoint outputs/gico/student.pt \
  --training_summary outputs/gico/training-summary.json \
  --context_rows locked-contexts.csv \
  --context_embeddings_npz contexts.npz \
  --baseline_rows uniform-baseline.csv \
  --out_dir outputs/gico/locked-report
```

Locked-test data are excluded from teacher/student selection. The reporting
command requires explicit baseline rows and emits per-context and aggregate
comparisons.

The verified image runtime registers exactly four external backbones:

- unconditional CIFAR-10 RF++ Config G and EDM VE as 1-RF;
- class-conditional ImageNet-64 RF++ Config E and EDM VE as 1-RF.

Upstream source, dataset, and checkpoint terms apply. Image checkpoints are
user-supplied and are not redistributed by this repository. In particular, the
pinned RF++ registry identifies the network implementation as
`CC-BY-NC-SA-4.0` and records that no separate checkpoint license notice was
found. Review [the third-party notices](THIRD_PARTY_NOTICES.md) before obtaining
or using external image code, datasets, checkpoints, or feature weights.

## Consistency distillation (secondary)

The optional endpoint flow-map workflow is deliberately separate from primary
GICO training:

```bash
genode-collect-flow-map-demonstrations --help
genode-train-flow-map --help
genode-evaluate-flow-map --help
```

Flow-map checkpoints remain cryptographically bound to the frozen backbone and
GICO checkpoint used to create them. Evaluation requires an explicit
measurement protocol and makes no performance claim when its quality gate is
not evaluated.

## Deterministic archives

Build checkpoint-only, named-checkpoint, or frozen policy archives with:

```bash
genode-build-release-archive backbone-manifest --help
genode-build-release-archive named-checkpoints --help
genode-build-release-archive gico-policy --help
genode-build-release-archive gico-clock-policy --help
genode-build-release-archive validate --archive release.zip
```

The policy archive validates the complete source bundle, then includes the
teacher state, student state, density table, context normalizers, and a fresh
portable GICO manifest. The wrapper records the original manifest digest and
actual training clock pool without carrying obsolete protocol-labelled JSON;
packaging does not imply retraining on the current 23-clock default.

`gico-clock-policy` accepts only the native current-v4 source policy (historical
frozen policies remain supported by `gico-policy`). It performs the strict
source validation, then uses the native source and complete-clock artifact
loaders before packaging either artifact. The distinct combined schema is
self-contained after extraction: it includes all eight source-policy files and
all six sidecar files. Its portable archive metadata binds the source and
decoder artifact identities, the decoder execution-state identity, and the
independently load-verifiable serialized-state identity.

Each build writes:

- the deterministic ZIP;
- `<archive>.manifest.json`, a canonical external manifest;
- `<archive>.sha256`, the archive digest.

## Development checks

```bash
python -m ruff check .
python -m pytest -q
python -m pip check
git diff --check
```

The CI matrix runs Python 3.11 and 3.13, includes Windows portability coverage,
builds the wheel, and smoke-tests every public CLI.

## License

GenODE is released under the [MIT License](LICENSE). Third-party code,
reference data, external model implementations, datasets, and checkpoint
weights remain subject to their respective terms.
