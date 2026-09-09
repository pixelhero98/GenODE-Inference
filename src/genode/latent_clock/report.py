from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from genode.latent_clock.artifacts import read_jsonl, sha256_file, write_csv, write_new_json
from genode.latent_clock.protocol import NOISE_SEEDS, OPTIMIZER_SEEDS


def paired_bootstrap(
    differences: np.ndarray, *, seed: int = 23119, replicates: int = 10000, strata: list[str] | None = None
) -> dict:
    """Prompt bootstrap after averaging generation seeds; retain optimizer variation."""
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] not in (1, 3) or not np.isfinite(values).all():
        raise ValueError("Paired differences must be a finite [prompts, 1 or 3 optimizer seeds] matrix.")
    rng = np.random.default_rng(seed)
    means = np.empty(replicates)
    per_prompt = values.mean(axis=1)
    groups = (
        [np.arange(len(values))]
        if strata is None
        else [np.flatnonzero(np.asarray(strata) == tag) for tag in sorted(set(strata))]
    )
    if strata is not None and len(strata) != len(values):
        raise ValueError("Bootstrap strata must match the paired prompt count.")
    for index in range(replicates):
        sampled = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
        means[index] = per_prompt[sampled].mean()
    optimizer_means = values.mean(axis=0)
    return {
        "mean_difference": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "prompt_count": len(values),
        "optimizer_means": optimizer_means.tolist(),
        "optimizer_std": float(optimizer_means.std(ddof=1)) if len(optimizer_means) > 1 else None,
        "bootstrap_unit": "prompt",
        "stratified": strata is not None,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def comparison_matrix(candidate: list[dict], anchor: list[dict], metric: str) -> np.ndarray:
    def index_rows(rows):
        index = {}
        for row in rows:
            key = (row["prompt_id"], int(row["noise_seed"]), int(row.get("optimizer_seed", -1)))
            if key in index:
                raise ValueError(f"Duplicate paired observation {key}.")
            index[key] = float(row[metric])
        return index

    left, right = index_rows(candidate), index_rows(anchor)
    prompts = sorted({key[0] for key in left})
    seeds = sorted({key[2] for key in left})
    if seeds != [-1] and seeds != sorted(OPTIMIZER_SEEDS):
        raise ValueError("Learned comparisons require all three preregistered optimizer seeds.")
    required = {(p, n, s) for p in prompts for n in NOISE_SEEDS for s in seeds}
    if set(left) != required:
        raise ValueError("Candidate comparison is missing matched generation seeds or prompts.")
    anchor_seeds = sorted({key[2] for key in right})
    expected_right = {(p, n, s) for p in prompts for n in NOISE_SEEDS for s in anchor_seeds}
    if set(right) != expected_right or (anchor_seeds != [-1] and anchor_seeds != seeds):
        raise ValueError("Anchor and candidate do not have the same paired prompt/noise panel.")
    return np.asarray(
        [
            [
                np.mean([left[(p, n, s)] - right[(p, n, -1 if anchor_seeds == [-1] else s)] for n in NOISE_SEEDS])
                for s in seeds
            ]
            for p in prompts
        ]
    )


def report_comparisons(spec_path: str, output: str) -> None:
    """Spec names immutable scored manifests for each method/NFE/budget comparison."""
    specification = json.loads(Path(spec_path).read_text())
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    summaries, sources = [], {}
    for comparison in specification["comparisons"]:
        candidate, anchor = [], []
        for role, rows in (("candidate", candidate), ("anchor", anchor)):
            for source in comparison[role]:
                sources[source["path"]] = sha256_file(source["path"])
                for row in read_jsonl(source["path"]):
                    if row["split"] != "locked_test":
                        raise ValueError("Final comparison report accepts locked-test rows only.")
                    rows.append({**row, "optimizer_seed": source.get("optimizer_seed", -1)})
        for metric in ("preference", "alignment"):
            interval = paired_bootstrap(comparison_matrix(candidate, anchor, metric))
            if interval["prompt_count"] != 512:
                raise ValueError("Final target-metric comparison requires all 512 locked test prompts.")
            summaries.append(
                {
                    "comparison": comparison["name"],
                    "group": comparison.get("group", "main"),
                    "nfe": comparison["nfe"],
                    "budget": comparison["budget"],
                    "metric": metric,
                    **interval,
                }
            )
    write_new_json(destination / "summary.json", {"comparisons": summaries, "source_sha256": sources})
    write_csv(destination / "summary.csv", summaries)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric in ("preference", "alignment"):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
        for axis, nfe in zip(axes, (4, 6, 8), strict=True):
            groups = defaultdict(list)
            for row in summaries:
                if row["metric"] == metric and row["nfe"] == nfe:
                    groups[row["comparison"]].append(row)
            for name, rows in groups.items():
                rows.sort(key=lambda row: int(row["budget"]))
                axis.errorbar(
                    [int(r["budget"]) for r in rows],
                    [r["mean_difference"] for r in rows],
                    yerr=[
                        [r["mean_difference"] - r["ci95_low"] for r in rows],
                        [r["ci95_high"] - r["mean_difference"] for r in rows],
                    ],
                    label=name,
                    marker="o",
                )
            axis.set(title=f"NFE {nfe}", xlabel="Calibration budget (%)")
            axis.axhline(0, color="black", linewidth=0.6)
        axes[0].set_ylabel(f"Paired {metric} difference")
        axes[-1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(destination / f"budget-{metric}.pdf")
        plt.close(fig)
    if specification.get("results_root"):
        from genode.latent_clock.accounting import write_cost_ledger

        write_cost_ledger(specification["results_root"], destination, specification["methods"])
        _density_plots(specification, destination)
        if specification.get("include_external_audits", False):
            _audit_summary(specification, destination)
    write_new_json(
        destination / "complete.json",
        {
            "specification_sha256": sha256_file(spec_path),
            "evaluation_scope": specification.get("evaluation_scope", {"split": "locked_test"}),
            "external_audits_included": specification.get("include_external_audits", False),
        },
    )


def _density_plots(specification: dict, destination: Path) -> None:
    import matplotlib.pyplot as plt

    from genode.gico.density_representation import grid_to_density_mass

    edges = np.linspace(0, 1, 65)
    root = Path(specification["results_root"])
    for backbone in ("sana", "sd15"):
        figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
        for axis, nfe in zip(axes, (4, 6, 8), strict=True):
            groups = defaultdict(list)
            for method in specification["methods"]:
                if (
                    method["backbone"] != backbone
                    or method["budget"] != "100"
                    or method["nfe"] != nfe
                    or method["group"] != "main"
                ):
                    continue
                for row in read_jsonl(root / "evaluation" / method["name"] / "locked_test/images.jsonl"):
                    mass = grid_to_density_mass(row["nodes"], reference_time_grid=edges, macro_steps=nfe)
                    groups[method["family"]].append(np.asarray(mass) * 64)
            for family, values in groups.items():
                axis.stairs(np.mean(values, axis=0), edges, label=family)
            axis.set(title=f"NFE {nfe}", xlabel="Normalized time", yscale="log")
        axes[0].set_ylabel("Mean clock density")
        axes[-1].legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(destination / f"density-{backbone}.pdf")
        plt.close(figure)


def _audit_summary(specification: dict, destination: Path) -> None:
    root = Path(specification["results_root"])
    summaries, indexed = [], {}
    for method in specification["methods"]:
        folder = root / "evaluation" / method["name"]
        geneval = read_jsonl(folder / "geneval-results.jsonl")
        if len(geneval) != 553 * 4:
            raise ValueError("Full official GenEval outputs are required for the final report.")
        tags = defaultdict(list)
        for row in geneval:
            tags[row["tag"]].append(float(row["correct"]))
        vision = read_jsonl(folder / "visionreward/scores.jsonl")
        indexed[method["name"]] = (geneval, vision)
        summaries.append(
            {
                "method": method["name"],
                "geneval": float(np.mean([np.mean(v) for v in tags.values()])),
                "geneval_categories": {k: float(np.mean(v)) for k, v in tags.items()},
                "vision_reward": float(np.mean([row["vision_reward"] for row in vision])),
            }
        )
    write_new_json(destination / "audit-summary.json", {"methods": summaries})
    comparisons = []
    for comparison in specification["comparisons"]:
        vision_roles, geneval_roles = {}, {}
        for role in ("candidate", "anchor"):
            vision_rows, geneval_columns = [], {}
            for source in comparison[role]:
                name = Path(source["path"]).parent.name
                geneval, vision = indexed[name]
                optimizer_seed = source["optimizer_seed"]
                vision_rows.extend({**row, "optimizer_seed": optimizer_seed} for row in vision)
                prompts = defaultdict(list)
                for row in geneval:
                    prompt_id = Path(row["filename"]).parent.parent.name
                    prompts[prompt_id].append(row)
                if len(prompts) != 553 or any(len(rows) != 4 for rows in prompts.values()):
                    raise ValueError("GenEval bootstrap requires complete four-image prompt panels.")
                geneval_columns[optimizer_seed] = {
                    key: (float(np.mean([r["correct"] for r in rows])), rows[0]["tag"]) for key, rows in prompts.items()
                }
            vision_roles[role] = vision_rows
            geneval_roles[role] = geneval_columns
        vision_interval = paired_bootstrap(
            comparison_matrix(vision_roles["candidate"], vision_roles["anchor"], "vision_reward")
        )
        candidates, anchors = geneval_roles["candidate"], geneval_roles["anchor"]
        seeds = sorted(candidates)
        prompt_ids = sorted(candidates[seeds[0]])
        tags = [candidates[seeds[0]][key][1] for key in prompt_ids]
        tag_counts = {tag: tags.count(tag) for tag in set(tags)}
        values = np.asarray(
            [
                [
                    (candidates[seed][key][0] - anchors[-1 if -1 in anchors else seed][key][0])
                    * len(prompt_ids)
                    / (len(tag_counts) * tag_counts[tag])
                    for seed in seeds
                ]
                for key, tag in zip(prompt_ids, tags, strict=True)
            ]
        )
        geneval_interval = paired_bootstrap(values, strata=tags)
        comparisons.append(
            {
                "comparison": comparison["name"],
                "budget": comparison["budget"],
                "nfe": comparison["nfe"],
                "vision_reward": vision_interval,
                "geneval_macro_category_accuracy": geneval_interval,
            }
        )
    write_new_json(destination / "audit-paired-intervals.json", {"comparisons": comparisons})


def freeze_methods(*, registry_path: str, manifest_path: str, output: str) -> None:
    registry = json.loads(Path(registry_path).read_text())
    manifest = json.loads(Path(manifest_path).read_text())
    methods = registry["methods"]
    if not methods:
        raise ValueError("Cannot freeze an empty method registry.")
    identities = ["native", "uniform"]
    for method in methods:
        if method.get("checkpoint"):
            identities.append(sha256_file(method["checkpoint"]))
        if method.get("fit_report"):
            fit = json.loads(Path(method["fit_report"]).read_text())
            if fit.get("transfer_excluded_nfe") == 6 and 6 in fit["fit_nfes"]:
                raise ValueError("Transfer checkpoint illegally used NFE 6.")
    test = [r["prompt_id"] for r in manifest["records"] if r["split"] == "locked_test"]
    # The subset is fixed by manifest order before any locked evaluation.
    write_new_json(
        output,
        {
            "protocol": "latent_t2i_method_freeze_v1",
            "methods": methods,
            "method_identities": sorted(set(identities)),
            "vision_reward_prompt_ids": test[:128],
            "prompt_manifest_sha256": sha256_file(manifest_path),
            "registry_sha256": sha256_file(registry_path),
            "geneval_images_per_prompt": 4,
        },
    )
