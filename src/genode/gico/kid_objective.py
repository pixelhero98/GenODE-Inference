"""Explicit distributional CIFAR supervision; never reinterpret LPIPS evidence."""

from __future__ import annotations

import numpy as np

from genode.artifacts.identity import semantic_sha256

KID_OBJECTIVE = "paired-cifar-kid-v1"


def reference_identity(block):
    return semantic_sha256(block, namespace="kid-reference")


def validate_kid_objective(objective):
    from genode.gico.image_objective import _digest

    generator, scorer = objective.get("generator", {}), objective.get("features", {})
    if (
        objective.get("protocol") != KID_OBJECTIVE
        or not generator.get("backbone")
        or not all(_digest(generator.get(k)) for k in ("checkpoint_sha256", "transform_sha256"))
        or generator.get("clock_protocol") != "frozen-two-grid-index-warp-v1"
        or not all(_digest(scorer.get(k)) for k in ("weights_sha256", "implementation_sha256"))
        or scorer.get("input_protocol") != "edm-uint8"
        or scorer.get("estimator") != "unbiased-cubic"
    ):
        raise ValueError("KID requires a pinned frozen generator, clock mapping and feature estimator.")


def validate_kid_rows(rows, objective):
    from genode.gico.image_objective import _digest

    validate_kid_objective(objective)
    splits, panels, samples = {}, {}, {}
    for row in rows:
        block = row.get("sample_block", {})
        seeds, noises = block.get("seeds", []), block.get("noise_sha256", [])
        reference = row.get("reference_block", {})
        indices = reference.get("indices", [])
        if (
            row.get("task") != "cifar10"
            or "target" in row
            or row.get("class_id") is not None
            or row.get("image_objective") != objective
            or row.get("measurement_protocol") != KID_OBJECTIVE
            or row.get("backbone") != objective["generator"]["backbone"]
            or row.get("backbone_binding") != objective["generator"]
            or not np.isfinite(row.get("metrics", {}).get("kid", np.nan))
        ):
            raise ValueError("KID evidence objective, frozen backbone or metric differs.")
        if (
            len(seeds) < 2
            or len(seeds) != row.get("ensemble_size")
            or len(seeds) != len(set(seeds))
            or any(type(s) is not int for s in seeds)
            or len(noises) != len(seeds)
            or not all(_digest(n) for n in noises)
            or len(set(noises)) != len(noises)
            or row.get("seed") != seeds[0]
            or len(indices) < 2
            or len(indices) != len(set(indices))
            or any(type(i) is not int or i < 0 for i in indices)
            or not _digest(reference.get("dataset_sha256"))
            or row.get("reference_id") != reference_identity(reference)
            or not row.get("panel_id")
        ):
            raise ValueError("KID requires complete generated and real reference sample blocks.")
        for identity in [
            ("panel", row["panel_id"]),
            *[("seed", s) for s in seeds],
            *[("noise", n) for n in noises],
            *[("real", reference["dataset_sha256"], i) for i in indices],
        ]:
            if splits.setdefault(identity, row["split"]) != row["split"]:
                raise ValueError("KID training and selection sample blocks overlap.")
        group = (row["split"], row["panel_id"])
        if panels.setdefault(row["context_id"], group) != group:
            raise ValueError("KID context changes its comparison panel.")
        key = (row["context_id"], row["seed"])
        if samples.setdefault(key, (block, reference)) != (block, reference):
            raise ValueError("Candidate and anchor KID sample blocks differ.")


def cubic_kid(generated, reference):
    """Fair U-statistic, including legitimate negative finite-sample estimates."""
    x, y = np.asarray(generated, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if (
        x.ndim != 2
        or y.ndim != 2
        or x.shape[1] != y.shape[1]
        or x.shape[1] == 0
        or min(len(x), len(y)) < 2
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("KID requires two finite feature matrices with at least two samples each.")
    xx, yy, xy = (x @ x.T / x.shape[1] + 1) ** 3, (y @ y.T / y.shape[1] + 1) ** 3, (x @ y.T / x.shape[1] + 1) ** 3
    return float(
        (xx.sum() - np.trace(xx)) / (len(x) * (len(x) - 1))
        + (yy.sum() - np.trace(yy)) / (len(y) * (len(y) - 1))
        - 2 * xy.mean()
    )
