"""Explicit experiment matrix, immutable method registry, and evaluation inputs."""

from __future__ import annotations

from pathlib import Path

from genode.latent_clock.artifacts import write_new_json
from genode.latent_clock.protocol import OPTIMIZER_SEEDS


def method_matrix(results: str | Path) -> list[dict]:
    fits = Path(results) / "fits"
    methods = []

    def add(backbone, family, nfe, budget, seed=-1, checkpoint=None, fit_report=None, group="main"):
        methods.append(
            {
                "name": f"{backbone}-{family}-n{nfe}-b{budget}-s{seed}",
                "backbone": backbone,
                "family": family,
                "nfe": nfe,
                "budget": str(budget),
                "optimizer_seed": seed,
                "method": "gico" if family.startswith("gico") else family,
                "checkpoint": str(checkpoint) if checkpoint else None,
                "fit_report": str(fit_report) if fit_report else None,
                "group": group,
            }
        )

    for backbone in ("sana", "sd15"):
        for nfe in (4, 6, 8):
            for baseline in ("uniform", "native"):
                add(backbone, baseline, nfe, "100")
            for budget in ("25", "50", "100"):
                for seed in OPTIMIZER_SEEDS:
                    for mode in ("deterministic", "stochastic"):
                        path = fits / f"{backbone}-gico-{mode}-n{nfe}-b{budget}-s{seed}"
                        add(backbone, "gico-" + mode, nfe, budget, seed, path / "policy.pt", path / "manifest.json")
                        methods[-1]["student_kind"] = mode
                        methods[-1]["clock_seed"] = seed + 491
                    if backbone == "sana":
                        for family, filename in (("bo", "clock.json"), ("pg", "policy.pt")):
                            path = fits / f"sana-{family}-n{nfe}-b{budget}-s{seed}"
                            add(backbone, family, nfe, budget, seed, path / filename, path / "complete.json")
            if backbone == "sd15":
                for seed in OPTIMIZER_SEEDS:
                    path = fits / f"sd15-ld3-n{nfe}-s{seed}"
                    add(backbone, "ld3", nfe, "100", seed, path / "schedule.json")
    if len({method["name"] for method in methods}) != len(methods):
        raise ValueError("Experiment matrix contains duplicate method identities.")
    return methods


def prepare_registry(results: str | Path, output: str | Path) -> None:
    methods = method_matrix(results)
    missing = [m[key] for m in methods for key in ("checkpoint", "fit_report") if m[key] and not Path(m[key]).is_file()]
    if missing:
        raise ValueError(f"Methods cannot be frozen before all fitting completes: {missing[:8]}")
    write_new_json(output, {"methods": methods})


def report_specification(results: str | Path, output: str | Path) -> None:
    root = Path(results)
    methods = method_matrix(root)
    comparisons = []
    groups = {}
    for method in methods:
        key = (method["backbone"], method["family"], method["nfe"], method["budget"])
        groups.setdefault(key, []).append(method)
    for (backbone, family, nfe, budget), members in groups.items():
        if family == "uniform":
            continue
        anchor_families = ["uniform", "native"]
        if family in ("bo", "pg", "ld3"):
            anchor_families.extend(("gico-deterministic", "gico-stochastic"))
        for anchor_family in anchor_families:
            if anchor_family == family:
                continue
            anchor_budget = "100" if anchor_family in ("native", "uniform") else budget
            anchors = groups[(backbone, anchor_family, nfe, anchor_budget)]

            def sources(items):
                return [
                    {
                        "path": str(root / "evaluation" / m["name"] / "locked_test-scores.jsonl"),
                        "optimizer_seed": m["optimizer_seed"],
                    }
                    for m in items
                ]

            comparisons.append(
                {
                    "name": f"{backbone}/{family} vs {anchor_family}",
                    "group": members[0]["group"],
                    "nfe": nfe,
                    "budget": budget,
                    "candidate": sources(members),
                    "anchor": sources(anchors),
                }
            )
    write_new_json(
        output,
        {
            "comparisons": comparisons,
            "results_root": str(root),
            "methods": methods,
            "include_external_audits": False,
            "evaluation_scope": {
                "primary_metrics": ["ImageReward-v1.0", "CLIP-FlanT5-XXL VQAScore"],
                "split": "locked_test",
                "prompts": 512,
                "excluded_audits": ["GenEval", "VisionReward"],
            },
        },
    )
