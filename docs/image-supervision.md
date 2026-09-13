# Image supervision: GICO KID and GICO-TF LPIPS

Standard GICO on CIFAR-10 and ImageNet-64 uses KID supervision, including RF++ and EDM VE interpreted as 1-RF. GICO over frozen BézierFlow also uses KID. Only the GICO-TF target-fidelity variant uses LPIPS.

| Method | Objective protocol | Paired terminal measurement |
|---|---|---|
| GICO on native RF++ / EDM-as-1RF | `paired-image-kid-v1` | Unbiased KID between generated and real-reference feature blocks |
| GICO + frozen BézierFlow | `paired-cifar-kid-v1` | Unbiased KID with the frozen two-grid generator binding |
| GICO-TF | `paired-lpips-v1` | VGG-LPIPS to a same-noise/class high-accuracy target |

The image objective is explicit in every measurement and artifact. `GICO-det-policy` and `GICO-sto-policy` choose the student architecture under that objective; they do not choose the supervision metric. Missing objectives and mixed or mislabelled LPIPS/KID evidence fail validation.

## Standard GICO: native KID

`genode-image-gico prepare` accepts a backbone manifest, rows, and ImageNet's native class table. Use `metrics: {"kid": value}` with `measurement_protocol: "paired-image-kid-v1"`. The `image_objective.generator` is the native backbone/context binding plus `backbone` and `clock_protocol: "density64-euler"`. The `features` object pins `weights_sha256`, `implementation_sha256`, `input_protocol: "edm-uint8"`, and `estimator: "unbiased-cubic"`. Compute the implementation identity with `feature_implementation_identity` from `genode.gico.image_evaluator`.

Each row includes a `sample_block` with complete unique `seeds` and corresponding `noise_sha256` entries, and a `reference_block` with the source `dataset_sha256` and unique absolute dataset `indices`. Both blocks contain at least two members. Set `ensemble_size` to generated block size, `seed` to its first seed and `reference_id` to `kid_objective.reference_identity(reference_block)`. ImageNet blocks also carry their native `class_id`; generated and reference blocks must agree. A `panel_id` identifies the comparison panel. CIFAR context is zero; ImageNet context is the frozen native class embedding. Research evidence and reports require complete 1,000-class ImageNet panels and equal class weight.

Keep raw unbiased KID estimates, including negative values. Average repeated block measurements within each comparison cell, then compute uniform KID minus candidate KID. Divide by one frozen training-only standard deviation, balancing NFEs and classes. Uniform utility is exactly zero. No log ratios, numerical floors, shrinkage or per-context reward normalization apply. Each candidate uses the complete unchanged generated/reference block. Reference data, generation noise and panel identities cannot cross fitting and held-out splits. No high-accuracy generated target is used.

For the same KID supervision over a frozen transformed sampler, use the separate [GICO + BézierFlow contract](frozen-bezier-kid.md) through the common fitting API.

## GICO-TF only: paired LPIPS

GICO-TF optimizes fidelity to a frozen high-accuracy target generator. The learned GICO reward teacher predicts the paired LPIPS improvement; it is distinct from that target generator. Supply `metrics: {"lpips": value}`, `ensemble_size: 1`, `measurement_protocol: "paired-lpips-v1"`, `panel_id`, and the shared solver/density/grid fields. Every row carries an `image_objective` with:

- `protocol: "paired-lpips-v1"`;
- `target_generator`: `backbone`, checkpoint SHA-256, solver, positive `rtol/atol`, native `time_range`, and `precision: "float32"`;
- `lpips`: `network: "vgg"`, complete scorer weights SHA-256, implementation SHA-256, version, and `input_protocol: "decoded-float32-no-resize-no-clamp"`.

The `target` records `seed`, `class_id` (null for CIFAR), `noise_sha256` and `image_sha256`. Compute `reference_id` with `genode.gico.image_objective.target_identity(objective, target)`. The same noise/class must use the same target across clocks/NFEs. Panels, noise and targets must not cross training, calibration, selection or test splits.

Measure VGG-LPIPS on the pinned comparator's decoded float32 tensors without resizing, clamping or uint8 conversion. Average repeated measurements within each panel/class/settings group, then take uniform LPIPS minus candidate LPIPS and divide by the frozen training-only scalar scale. For a matched-supervision GICO-TF/BézierFlow comparison, use identical target pairs and train/selection splits. Retain method-specific optimizers and disclose target preparation, fitting and selection costs separately. Check target-solver convergence on training noise before reuse. This LPIPS comparison is distinct from KID-supervised GICO over frozen BézierFlow.

## Shared fitting, selection and artifacts

Both objectives use the shared teacher, deterministic and stochastic students, teacher-weighted distillation and differentiable score chasing. Initial generation noise is never a policy input. At inference only the selected student is loaded; no target generation, teacher scoring or reward selection occurs.

Select teacher checkpoint and temperature by measured held-out reference-mixture utility regret, with equal panel and density-family weight. Temperature is in raw KID-improvement units for GICO or raw LPIPS-improvement units for GICO-TF, before scalar reward normalization. Auxiliary-score normalization and clipping remain separate. Select each eligible student by its measured held-out utility under the same frozen objective.

Fresh fits use 2,000 teacher and student steps, zero image dropout, beta 0.01 and temperature candidates 0.05/0.1/0.5. Teacher-selection ties prefer 0.05, then the earlier checkpoint. Overrides remain explicit. Calibrations record their actual training NFEs; an explicitly supplied calibration panel can span additional training NFEs.

`image_protocol_metadata()` describes native GICO KID by default; `image_protocol_metadata(method="GICO-TF")` explicitly describes LPIPS. These have distinct version-9 benchmark identities. The template budgets 200 generated samples (or GICO-TF target pairs) per conditioning group in each training/selection panel: one unconditional group for CIFAR and 1,000 native class groups for ImageNet. Metadata exposes both per-group and per-dataset generated-image workloads; these exclude reference/target preparation and scoring. Locked FID50k remains dataset-wide. These budgets are not minimum KID block sizes or substitutes for actual recorded rows. Historical metadata and existing v6 artifacts retain their recorded objectives and are not relabelled. Backbone identities and artifact wire versions are unchanged.

FID50k independently evaluates the resulting distribution. Neither lower fitting KID nor lower LPIPS establishes a FID improvement. This correction performs no beta sweep, benchmark rerun or new performance comparison.
