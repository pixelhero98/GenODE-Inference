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

Image artifacts retain the shared architecture version and add the required versioned image objective and split provenance. Historical KID image artifacts/evidence are rejected; preserve them with their originating runtime. Non-image artifact compatibility is unchanged. Frozen reward calibrations record the actual calibration NFEs; an image fit can use an explicitly supplied calibration panel spanning additional training NFEs.

## Fitting and evaluation

Select reward-teacher checkpoint and temperature by measured held-out reference-mixture utility regret. Keep separately reported panel and density-family regret, equally weighted. Reference temperature is in raw LPIPS-improvement units, before division by the frozen scalar scale. Auxiliary clipping does not affect reference logits.

Both students retain the late-ramped teacher-score term and checkpoint selection by measured held-out LPIPS utility after chasing begins. Tune dataset/backbone-specific temperature and score coefficient on selection evidence only. A bounded example is temperatures 0.05/0.2/1 times the training utility scale and coefficients 0.01/0.05/0.1. Select the final coefficient by generated selection LPIPS, freeze it, then evaluate common-seed FID10k followed by FID50k for all methods. Never tune or decide whether to continue from evaluation performance.

For a matched-supervision comparison, use identical target pairs and train/selection splits for GICO-TF and BezierFlow. Retain method-specific optimizers and disclose target preparation, fitting and selection costs separately. Matching target data does not imply matching compute. Independently check target-solver convergence on training noise before reuse. LPIPS fidelity and FID assess different properties; an LPIPS improvement is not evidence of a FID improvement.
