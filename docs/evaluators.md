# Built-in measured selection

Stochastic student fitting requires measured selection. Deterministic fitting uses the generator-free teacher-score/KL rule described in [student selection](student-selection.md) and does not invoke this evaluator.

Use `genode.gico.evaluators:build_evaluator` in the existing factory/config interface. It supports every retained task, native image GICO KID, explicit GICO-TF LPIPS and frozen BézierFlow KID. Custom factories remain an extension point.

```json
{
  "factory": "genode.gico.evaluators:build_evaluator",
  "config": {
    "rows": "selection.jsonl",
    "contexts": "contexts.npz",
    "runtime": "runtime.json",
    "cases": "cases.json",
    "output": "selection-measurements",
    "clock_seed": 412,
    "clock_replicates": 4
  }
}
```

Paths in this factory and its runtime/case files resolve from the working directory. Absolute paths can make invocation independent of that directory. The output directory must be new. `rows` contains validation evidence with measured uniform anchors; the shared fitter checks complete held-out coverage. `contexts` uses the shared NPZ context table. Missing assets, changed uniform measurements or incomplete panels fail rather than substituting predictions.

`cases.json` maps `measurement_identity(row)` from `genode.gico.evaluators` to each measurement's asset description. This hashes context, generation seed and reference identity. Multiple seeds in one context therefore resolve different targets or reference blocks. Cases are shared across candidate checkpoints and solver/NFE settings only when their measurement identity agrees.

An optional factory `python` points to an interpreter with GenODE installed and the task's external dependencies. It runs the frozen generator in that environment; fit-time policy snapshots stay detached on CPU. Each candidate directory retains its request, generated images or sequence metrics, and actual paired measurements. Clock identities derive from context, solver/NFE, generation seed, member and replicate and are reused across candidates. Stochastic identities cannot be shared by distinct generated members or replicates.

## Forecasting and molecules

Runtime configuration contains `task`, `device`, and a `backbones` map. Each entry requires `checkpoint`, `checkpoint_sha256` and the evidence's `backbone_id`.

- Forecast entries also specify `dataset_root` and `time_feature_mode`. The checkpoint determines history and prediction horizons. Each case specifies `backbone` (the map key), `example_idx` in the validation dataset, `reference_id`, `target_sha256`, `collection_batch_size: 1`, and complete consecutive `sample_seed_values`. Use the physical member seeds recorded during collection; a panel's logical seed alone cannot replay a later example.
- Molecular entries additionally specify `processed_dir`, `stratum`, `rollout_steps` and `stride_eval`. Cases specify `backbone`, validation `example_idx`, `reference_id` and `target_sha256`. The existing molecular runtime preserves the initial native context and reuses one complete clock through each member's rollout.

`tensor_digest` from `genode.gico.task_evaluators` hashes the complete validation target. For forecasts this is the target plus any future block, in evaluation order; for molecules it is `eval_item(index)["future_coords"]`. The runtime verifies actual native context and checkpoint identity before returning CRPS/MASE or the five molecular metrics.

## Native CIFAR and ImageNet

Runtime keys are `task`, `device`, `objective` (the exact row `image_objective`), `backbone_manifest`, `checkpoint` and pinned upstream `source_root`.

- GICO-TF only: LPIPS adds `lpips_checkpoint`, a complete VGG-LPIPS state dictionary, preventing implicit downloads. Pin its file SHA-256, installed package version and `lpips_implementation_identity()` from `genode.gico.image_evaluator`. Each case's `target` points to a decoded float32 NPY tensor matching `target.image_sha256` and the generated batch shape.
- Standard image GICO: KID adds `feature_checkpoint`. Each case supplies `reference_features`, its file `reference_features_sha256`, and the exact `reference_block` from the row. The NPY feature matrix must preserve all declared indices, their order and class. Pin the detector and `feature_implementation_identity()` from the same module.

The evaluator regenerates the declared noise, validates its raw tensor digests, uses the native class embeddings, executes Euler with exact NFE, and retains decoded images. See [image supervision](image-supervision.md) for objective and split rules. Supplied reference features must have been extracted with that pinned detector and input protocol.

## Frozen BézierFlow

Use task `cifar10`, the separate `paired-cifar-kid-v1` objective and a `bezier` runtime object containing:

```json
{
  "scope": {"nfe": 4},
  "assets": "assets/cifar",
  "bezier_source": "upstream/BezierFlow",
  "reflow_source": "upstream/reflow",
  "source_revisions": {"bezier": "PINNED_COMMIT", "reflow": "PINNED_COMMIT"},
  "asset_sha256": {"reflow_1.pth": "SHA256", "inception-2015-12-05.pkl": "SHA256"},
  "frozen_bezier": "assets/frozen-bezier.pt",
  "frozen_bezier_sha256": "SHA256",
  "ema_sha256": "STATE_FINGERPRINT"
}
```

Replace identity placeholders with measured digests. Supply the same KID case files as above. The bridge checks clean upstream revisions, checkpoint hashes, EMA state and both learned grids. It keeps the grid transformation and feature detector frozen, uses the enclosing runtime's device, and counts actual backbone calls. External upstream dependencies, including OmegaConf, must be installed in that runtime. See [frozen BézierFlow KID](frozen-bezier-kid.md).

## SANA and SD1.5

Runtime keys are `task`, `generator_config` (the existing explicit runtime configuration) and `scorer`. The scorer object contains `python`, `device`, `asset_manifest`, `versions`, `weight_fingerprints` and `asset_manifest_sha256`, matching collection's scorer metadata. The scorer interpreter must have GenODE, ImageReward and T2V-Metrics installed with their explicit cached assets.

Cases carry `prompt`, `image_id`, `caption_id` and `reference_id`. The evaluator reconstructs the prompt reference hash, checks the pooled native embedding, verifies actual solver/runtime/scorer identities, executes one image and retains its PNG and raw scores. SD1.5 `ipndm_v` and fixed `ipndm` require separate evidence and calibration. A different label cannot change the solver that actually ran.
