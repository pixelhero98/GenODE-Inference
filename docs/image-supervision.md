# Paired image supervision (GICO-TF)

CIFAR-10 and ImageNet-64 optimize fidelity to a frozen high-accuracy target generator. GICO's learned reward teacher predicts this improvement; it is distinct from the target generator. The density student still uses teacher-weighted distillation and differentiable teacher-score chasing. At inference only the student is loaded, with no target generation or reward selection.

For fixed noise and class, measure VGG-LPIPS between the decoded candidate and decoded target. Use the pinned comparator's float32 tensors without resizing, clamping or uint8 conversion. Average paired measurements over seeds within each panel/class/settings group. The raw utility is mean uniform LPIPS minus mean candidate LPIPS. Divide by one frozen training-only standard deviation, balancing NFEs and classes. Uniform is exactly zero. There are no log ratios, numerical floors, running statistics or context-dependent terminal scales. Auxiliary-score standardization remains separate.

## Evidence interface

`genode-image-gico prepare` accepts a backbone manifest, rows, and ImageNet's native class table. CIFAR has explicit zero context; ImageNet uses native class embeddings. Each row uses `metrics: {"lpips": value}`, `ensemble_size: 1`, `measurement_protocol: "paired-lpips-v1"`, `panel_id`, and the shared solver/density/grid fields. A panel identifies the set of repeated seeds, independently of the individual target's `reference_id`. Initial noise is never supplied to either GICO model.

Every row carries an `image_objective` with:

- `protocol: "paired-lpips-v1"`;
- `target_generator`: `backbone`, checkpoint SHA-256, solver, positive `rtol/atol`, native `time_range`, and `precision: "float32"`;
- `lpips`: `network: "vgg"`, complete scorer weights SHA-256, implementation SHA-256, version, and `input_protocol: "decoded-float32-no-resize-no-clamp"`.

The `target` object records `seed`, `class_id` (null for CIFAR), `noise_sha256` and `image_sha256`. Compute `reference_id` with `genode.gico.image_objective.target_identity(objective, target)`. The same noise/class must use the same target across all clocks/NFEs. Panels, noise and targets must not cross training, calibration, selection or test splits. Names alone cannot establish isolation. The common fitting API validates these contracts even without the image CLI.

Image artifacts retain the shared architecture version and add the required versioned image objective and split provenance. KID uses the separate `paired-image-kid-v1` native objective or `paired-cifar-kid-v1` frozen BézierFlow objective. Mixed or mislabelled LPIPS/KID evidence is rejected. Non-image artifact compatibility is unchanged. Frozen reward calibrations record the actual calibration NFEs; an image fit can use an explicitly supplied calibration panel spanning additional training NFEs.

## Fitting and evaluation

Select reward-teacher checkpoint and temperature by measured held-out reference-mixture utility regret. Keep separately reported panel and density-family regret, equally weighted. Reference temperature is in raw LPIPS-improvement units, before division by the frozen scalar scale. Auxiliary clipping does not affect reference logits.

Both students retain the score-chasing term and measured held-out utility selection. Fresh fits use 2,000 teacher and student steps, zero image dropout, beta 0.01 and temperature candidates 0.05/0.1/0.5 in raw utility units. Ties prefer 0.05, then the earlier checkpoint. Overrides remain explicit; this release performs no beta sweep or new benchmark comparison.

## Native distributional KID

Use `metrics: {"kid": value}` with `measurement_protocol: "paired-image-kid-v1"`. The `image_objective.generator` is the native backbone/context binding plus `backbone` and `clock_protocol: "density64-euler"`. The `features` object pins `weights_sha256`, `implementation_sha256`, `input_protocol: "edm-uint8"`, and `estimator: "unbiased-cubic"`. Compute the implementation identity with `feature_implementation_identity` from `genode.gico.image_evaluator`.

Each row includes a `sample_block` with complete unique `seeds` and corresponding `noise_sha256` entries, and a `reference_block` with the source `dataset_sha256` and unique absolute dataset `indices`. Both blocks contain at least two members. Set `ensemble_size` to generated block size, `seed` to its first seed and `reference_id` to `kid_objective.reference_identity(reference_block)`. ImageNet blocks also carry their native `class_id`; generated and reference blocks must agree. The native context is zero for CIFAR and the frozen class embedding for ImageNet. Research evidence and reports require complete 1,000-class ImageNet panels and equal class weight.

Keep raw unbiased estimates, including negative values. Utility is uniform KID minus candidate KID with a training-only scalar scale; average repeated block measurements before forming utility. Each candidate uses the complete unchanged generated/reference block. Reference data, generation noise and panel identities cannot cross fitting and held-out splits. Native KID has no high-accuracy target object; GICO-TF's paired high-accuracy target objective remains LPIPS.

For a matched-supervision comparison, use identical target pairs and train/selection splits for GICO-TF and BezierFlow. Retain method-specific optimizers and disclose target preparation, fitting and selection costs separately. Matching target data does not imply matching compute. Independently check target-solver convergence on training noise before reuse. LPIPS fidelity and FID assess different properties; an LPIPS improvement is not evidence of a FID improvement.
