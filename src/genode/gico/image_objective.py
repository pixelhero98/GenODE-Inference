"""Versioned paired target-generator fidelity evidence, independent of fitting."""

from __future__ import annotations

import numpy as np

from genode.artifacts.identity import semantic_sha256

IMAGE_OBJECTIVE = "paired-lpips-v1"
IMAGE_TASKS = frozenset(("cifar10", "imagenet64"))


def _digest(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_image_objective(objective: dict) -> None:
    from genode.gico.kid_objective import KID_OBJECTIVE, validate_kid_objective

    if isinstance(objective, dict) and objective.get("protocol") == KID_OBJECTIVE:
        validate_kid_objective(objective)
        return
    if not isinstance(objective, dict) or objective.get("protocol") != IMAGE_OBJECTIVE:
        raise ValueError(
            "Image policies require paired-lpips-v1; historical KID evidence/artifacts need their original runtime."
        )
    target, scorer = objective.get("target_generator", {}), objective.get("lpips", {})
    if (
        not target.get("backbone")
        or not _digest(target.get("checkpoint_sha256"))
        or not target.get("solver")
        or target.get("precision") != "float32"
        or not np.isfinite([target.get("rtol", np.nan), target.get("atol", np.nan)]).all()
        or min(target.get("rtol", 0), target.get("atol", 0)) <= 0
    ):
        raise ValueError("Target generator requires its frozen backbone, checkpoint, solver, tolerances and precision.")
    times = np.asarray(target.get("time_range", []), dtype=float)
    if times.shape != (2,) or not np.isfinite(times).all() or times[0] == times[1]:
        raise ValueError("Target generator requires two distinct finite solver endpoints.")
    if (
        scorer.get("network") != "vgg"
        or scorer.get("input_protocol") != "decoded-float32-no-resize-no-clamp"
        or not scorer.get("version")
        or not _digest(scorer.get("weights_sha256"))
        or not _digest(scorer.get("implementation_sha256"))
    ):
        raise ValueError("LPIPS requires pinned VGG weights, implementation and decoded float32 preprocessing.")


def target_identity(objective: dict, target: dict) -> str:
    return semantic_sha256({"generator": objective["target_generator"], **target}, namespace="lpips-target")


def validate_image_rows(rows: list[dict]) -> None:
    """Check identities across schedules, NFEs and splits, even through common fit."""
    image_rows = [row for row in rows if row.get("task") in IMAGE_TASKS]
    if not image_rows:
        return
    objective = image_rows[0].get("image_objective")
    validate_image_objective(objective)
    from genode.gico.kid_objective import KID_OBJECTIVE, validate_kid_rows

    if objective["protocol"] == KID_OBJECTIVE:
        validate_kid_rows(image_rows, objective)
        return
    splits, targets, contexts = {}, {}, {}
    for row in image_rows:
        if row.get("image_objective") != objective or row.get("measurement_protocol") != IMAGE_OBJECTIVE:
            raise ValueError("Image target generator and LPIPS protocols must match across all evidence.")
        if row.get("backbone") != objective["target_generator"]["backbone"]:
            raise ValueError("Candidate and target generator backbones differ.")
        binding = row.get("backbone_binding")
        if (
            binding is not None
            and binding.get("checkpoint_sha256") != objective["target_generator"]["checkpoint_sha256"]
        ):
            raise ValueError("Candidate and target generator checkpoint SHA-256 identities differ.")
        if any(key in row for key in ("reward_metrics", "reward_estimator", "jackknife_kid")):
            raise ValueError("KID reward overrides are incompatible with paired LPIPS.")
        metric = row.get("metrics", {}).get("lpips", np.nan)
        if not np.isfinite(metric) or metric < 0:
            raise ValueError("Raw LPIPS must be finite and nonnegative; recollect old KID evidence.")
        if type(row.get("ensemble_size")) is not int or row["ensemble_size"] != 1:
            raise ValueError("LPIPS evidence records one paired image per row.")
        label = row.get("class_id")
        if (row["task"] == "cifar10" and label is not None) or (
            row["task"] == "imagenet64" and (type(label) is not int or not 0 <= label < 1000)
        ):
            raise ValueError("Image evidence requires the native class identity (None for unconditional CIFAR).")
        target = row.get("target", {})
        if (
            type(target.get("seed")) is not int
            or target["seed"] != row.get("seed")
            or "class_id" not in target
            or target["class_id"] != label
            or not _digest(target.get("noise_sha256"))
            or not _digest(target.get("image_sha256"))
            or row.get("reference_id") != target_identity(objective, target)
        ):
            raise ValueError("LPIPS target identity must match its generation noise, seed and class.")
        panel = row.get("panel_id")
        if not isinstance(panel, str) or not panel:
            raise ValueError("LPIPS requires a comparison panel_id separate from individual targets.")
        # A renamed panel, seed, or target must not hide reuse across splits.
        for identity in (("panel", panel), ("noise", target["noise_sha256"], label), ("seed", target["seed"], label)):
            if splits.setdefault(identity, row["split"]) != row["split"]:
                raise ValueError("Image splits require disjoint paired measurement panels and noise/target identities.")
        sample = (target["seed"], label)
        if targets.setdefault(sample, row["reference_id"]) != row["reference_id"]:
            raise ValueError("The same noise/class must use one fixed high-accuracy target across clocks and NFEs.")
        group = (row["split"], panel, label)
        if contexts.setdefault(row["context_id"], group) != group:
            raise ValueError("A comparison context cannot combine different panels or classes.")


def lpips_values(scorer, candidate, target):
    """Pinned upstream convention: decoded float32 tensors, no resize or clamp."""
    import torch

    if candidate.shape != target.shape or candidate.ndim != 4 or candidate.shape[1] != 3:
        raise ValueError("LPIPS requires identically shaped NCHW RGB candidate/target batches.")
    if not torch.isfinite(candidate).all() or not torch.isfinite(target).all():
        raise ValueError("LPIPS inputs must be finite.")
    if scorer.training or any(parameter.requires_grad for parameter in scorer.parameters()):
        raise ValueError("LPIPS scorer must be frozen in evaluation mode.")
    values = scorer(target.float(), candidate.float()).reshape(-1)
    if len(values) != len(candidate) or not torch.isfinite(values).all() or torch.any(values < 0):
        raise ValueError("LPIPS must return one finite nonnegative value per image.")
    return values


def validate_image_split_identities(provenance: dict, *, protocol=IMAGE_OBJECTIVE) -> None:
    if not isinstance(provenance, dict) or set(provenance) != {"train", "calibration", "validation"}:
        raise ValueError("Image artifact lacks target/noise split provenance.")
    seen = {key: set() for key in ("panels", "targets", "noises")}
    if protocol != IMAGE_OBJECTIVE:
        seen.update(real_samples=set(), generated_seeds=set())
    for phase, fields in provenance.items():
        if not isinstance(fields, dict) or set(fields) != set(seen):
            raise ValueError("Incomplete image split provenance.")
        for key, values in fields.items():
            if (
                not isinstance(values, list)
                or any(not isinstance(v, str) or not v for v in values)
                or len(values) != len(set(values))
                or (phase != "calibration" and not values)
            ):
                raise ValueError("Invalid image split identities.")
            if key == "noises" and not all(_digest(value) for value in values):
                raise ValueError("Image split noise identities must be SHA-256 hashes.")
            if key == "targets" and not all(
                value.startswith("lpips-target:" if protocol == IMAGE_OBJECTIVE else "kid-reference:")
                and _digest(value.split(":", 1)[-1])
                for value in values
            ):
                raise ValueError("Image split targets require versioned target identities.")
            if seen[key].intersection(values):
                raise ValueError("Image split identities overlap.")
            seen[key].update(values)


def image_split_fields(row):
    """Individual identities, so renamed or partially overlapping blocks cannot hide reuse."""
    fields = {"panels": [row["panel_id"]], "targets": [row["reference_id"]]}
    if "target" in row:
        fields["noises"] = [row["target"]["noise_sha256"]]
    else:
        fields["noises"] = row["sample_block"]["noise_sha256"]
        fields["generated_seeds"] = [str(s) for s in row["sample_block"]["seeds"]]
        ref = row["reference_block"]
        fields["real_samples"] = [f"{ref['dataset_sha256']}:{i}" for i in ref["indices"]]
    return fields
