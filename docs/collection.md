# Shared complete-solve collection

Collection finishes before utility surrogate or policy fitting. `genode-collect-gico` plans a fixed set of complete generator solves and records their measurements; fitting and checkpoint selection consume only that saved evidence and frozen-utility surrogate predictions.

| Task | Default budget per NFE | Fitting / held out |
|---|---|---|
| Forecasting, molecules, SANA, SD1.5 | 256 distinct contexts, one assigned density per context, two generation seeds, paired uniform anchors additional | 204 / 52 contexts |
| CIFAR-10 | 10,000 generated images, including uniform; 400 per density | 8,000 / 2,000 images in disjoint noise/reference panels |
| ImageNet-64 | 64 candidate images per class plus paired uniform anchors; 1,000 classes, 40 assigned to each density | 800 / 200 whole classes |

For 256 contexts, the 25 density assignments receive ten contexts each and six seeded random densities receive one extra. Sampling is stratified round-robin over shuffled source strata. The default collection seed labels are `(0, 1)`; they are generation repeats, separate from the collection sampling seed and policy RNG. Preserve native sequence ensembles: five forecast trajectories or 16 molecular trajectories per repeat by default (`forecast_ensemble_size` / `molecule_ensemble_size` are explicit overrides). A repeat/context hash determines disjoint consecutive physical member seeds. Thus 256 uniform anchors alone cost `256 * 2 * ensemble_size` trajectories; candidate costs are additional except for uniform-assigned contexts. The manifest counts physical trajectories, not just measurement rows. The selected contexts, assignments and split membership are shared across NFEs. Every context's repeats and anchors stay together. If fewer contexts are available, the resolved count is recorded rather than silently adding solves; at least 50 are required to populate both splits across the pool.

For COCO, record each source image ID as `group_id`. A seeded random caption is selected from each image **before** sampling 256 images. This improves source diversity and prevents related captions crossing the split. Record caption ID as `context_id`. The split validator also rejects cross-split source groups.

CIFAR has one native zero context; two panel identities distinguish fitting from held-out noise/reference blocks without conditioning on noise. Each repeat contains 160 fitting and 40 held-out images per density. ImageNet uses 32 candidate images per class per repeat. Its 40 uniform-assigned classes reuse those outputs as anchors, so `(1000 + 960) * 64 = 125440` complete image solves are required per NFE. The two repeats divide these budgets; they do not double them. Reference-feature preparation and explicit final reporting have separate costs.

KID observations are complete paired blocks: the candidate and uniform use identical generated seeds, reference indices and class. Disjoint splits cannot reuse generated noise or reference data. The explicit GICO-TF collection option `image_objective: lpips` expands each image allowance into single-image target pairs without changing the image budget. High-accuracy target preparation is external and separately accounted.

The manifest records configuration, selected inventory, assignments, full reference support, held-out density identities, split membership, every solve request and its seed/block, source revision, completed solve count and measurement checksum. Settings are resolved before measurements are observed. A completed manifest is mandatory for research fitting. `purpose: functional` exists only for small, explicitly labelled software fixtures; it cannot be relabelled as research evidence.

Validation independently checks split counts and each density's fitting allocation, even when manifest checksums have been recomputed. Fitting counts use the configured fraction with largest-remainder allocation and the recorded density order for ties. If source groups contain multiple selected contexts, keep each group intact and choose the nearest feasible held-out count; equal-distance ties prefer the larger holdout. CIFAR panel identities and exact image-budget divisibility are also checked.

## Plan and collect

A collection JSON config contains `task`, frozen `backbone`, `source_revision`, `settings` (for example `[["euler", 4], ["euler", 6]]`), `inventory` and `output`. Inventory entries contain `context_id` and optional `stratum`; ImageNet requires `class_id` and all 1,000 classes. Native embeddings stay in a separate NPZ table. Optional `collection` fields match `CollectionConfig` in `genode.gico.collection`.

```bash
genode-collect-gico --config collection.json --plan-only
```

Inspect the plan and use its request IDs to prepare task-specific references/cases. For execution, configure `collector.factory: "genode.gico.evaluators:build_collector"` and `collector.config` with `runtime`, `contexts`, `templates`, `cases`, and `output`. Each template and case is keyed by request ID. Templates contain reference and frozen asset identities, never prefilled metrics. Supply complete planned image blocks. The native runtimes validate context, frozen generator/scorer identity and exact NFE. See [runtime configuration](evaluators.md).

Run the same deterministic planning configuration with a fresh output destination and without `--plan-only`. The result is a JSON bundle containing `rows` and `collection_manifest`. Native runtime paths resolve from the working directory; the collection output path resolves relative to the config file. Keep machine-specific files outside the repository.

The Python equivalents are `plan_collection(...)`, `collect(manifest, measure)` and `validate_collection(manifest, rows)`. The callback performs each planned request exactly once. Failed or incomplete collections cannot be fitted; this interface does not silently retry or expand the budget.

## Fitting from the completed bundle

```json
{
  "rows": "collected.json",
  "collection_manifest": "collected.json",
  "contexts": "contexts.npz",
  "output": "policy",
  "policy_kind": "deterministic"
}
```

```bash
genode-train-gico --config train.json --dry-run
genode-train-gico --config train.json
```

Both policies reuse this evidence and its context holdout. Explicit `--policy-kind both` trains both with one selected frozen utility surrogate. No generator or terminal scorer is available through fitting interfaces.
