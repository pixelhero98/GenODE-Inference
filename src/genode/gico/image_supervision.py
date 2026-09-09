"""Convert measured small-image evidence to the shared paired-row interface."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from genode.artifacts.identity import semantic_sha256
from genode.backbones.protocol import ImageBackboneManifest
from genode.backbones.registry import get_image_backbone_spec
from genode.gico.clocks import verify_measurement_clock
from genode.gico.image_conditional import paired_kid_shrinkage
from genode.gico.image_conditional_context import context_binding


def prepare_image_rows(manifest: dict) -> tuple[list[dict], dict[str, list[float]], dict]:
    """Prepare raw KID rows with their exact executed densities and native context.

    Manifest contains backbone_manifest, rows, and ImageNet native_context_table
    and feature_groups. Each row follows the shared paired measurement schema;
    ImageNet adds class_id and paired jackknife_kid block measurements.
    """
    backbone = ImageBackboneManifest.from_manifest_dict(manifest["backbone_manifest"])
    task = get_image_backbone_spec(backbone.model_key).dataset_key
    if task not in {"cifar10", "imagenet64"}:
        raise ValueError("Small-image evidence requires CIFAR-10 or ImageNet-64.")
    table = np.asarray(manifest["native_context_table"] if task == "imagenet64" else [[0.0]], dtype=np.float32)
    binding = context_binding(backbone, table)
    contexts: dict[str, list[float]] = {}
    rows = []
    panel_splits: dict[tuple, str] = {}
    for original in manifest["rows"]:
        row = dict(original)
        if any(key in row for key in ("reward_metrics", "reward_estimator")):
            raise ValueError("Image preparation requires raw measurements, not previously fitted reward overrides.")
        if row.get("task", task) != task or row.get("backbone", backbone.model_key) != backbone.model_key:
            raise ValueError("Measured image task/backbone does not match its native context binding.")
        if row.get("solver") != "euler":
            raise ValueError("The pinned small-image runtime supports Euler only.")
        label = row.get("class_id")
        if task == "imagenet64" and (type(label) is not int or not 0 <= label < 1000):
            raise ValueError("ImageNet rows require an integer class_id in [0,1000).")
        if task == "cifar10" and label is not None:
            raise ValueError("Unconditional CIFAR-10 evidence cannot carry class labels.")
        if type(row["seed"]) is not int or type(row["ensemble_size"]) is not int or row["ensemble_size"] < 2:
            raise ValueError("Paired KID requires an integer generation seed and ensemble size at least two.")
        source_context_id = f"class:{label}" if task == "imagenet64" else "unconditional"
        panel = (row["reference_id"], source_context_id)
        if panel in panel_splits and panel_splits[panel] != row["split"]:
            raise ValueError(
                "Image train/calibration/validation/test splits must use disjoint paired measurement panels."
            )
        panel_splits[panel] = row["split"]
        context_id = f"{row['split']}:{row['reference_id']}:" + (
            f"class:{label}" if task == "imagenet64" else "unconditional"
        )
        contexts[context_id] = table[label if label is not None else 0].tolist()
        if row.get("context_id", context_id) != context_id:
            raise ValueError("Measured class and context ID disagree.")
        row.update(
            task=task,
            backbone=backbone.model_key,
            context_id=context_id,
            backbone_binding=binding,
            source_context_id=source_context_id,
            context_protocol="native_context_paired_panel_holdout_v1",
        )
        if row.get("split") not in {"train", "calibration", "validation", "test"}:
            raise ValueError("Image evidence requires an explicit train/calibration/validation/test split.")
        if not np.isfinite(float(row["metrics"]["kid"])):
            raise ValueError("Raw terminal KID must be finite.")
        verify_measurement_clock(row)
        rows.append(row)
    if not rows:
        raise ValueError("Image preparation requires nonempty measured rows.")
    metadata = {
        "task": task,
        "backbone_binding": binding,
        "raw_metric": "kid",
        "context_protocol": "native_context_paired_panel_holdout_v1",
    }
    report_cells = defaultdict(list)
    for row in rows:
        report_cells[tuple(row[key] for key in ("split", "solver", "nfe", "schedule_key"))].append(row)
    raw_report = []
    for key, cell in report_cells.items():
        by_class = defaultdict(list)
        for row in cell:
            by_class[row["source_context_id"]].append(float(row["metrics"]["kid"]))
        if len(by_class) != (1000 if task == "imagenet64" else 1):
            raise ValueError("Raw image metric reports require complete equally weighted class coverage.")
        raw_report.append(
            {
                **dict(zip(("split", "solver", "nfe", "schedule_key"), key, strict=True)),
                "unshrunk_class_conditional_kid" if task == "imagenet64" else "global_kid": float(
                    np.mean([np.mean(values) for values in by_class.values()])
                ),
            }
        )
    metadata["raw_metric_report"] = raw_report
    global_metrics = manifest.get("global_metrics", [])
    for result in global_metrics:
        if not np.isfinite([float(result["kid"]), float(result["fid"])]).all():
            raise ValueError("Separately measured global KID/FID diagnostics must be finite.")
        if tuple(result[key] for key in ("split", "solver", "nfe", "schedule_key")) not in report_cells:
            raise ValueError("Global KID/FID diagnostics must identify a measured image setting.")
    metadata["global_metric_report"] = [dict(result) for result in global_metrics]
    if task == "imagenet64":
        groups = dict(manifest["feature_groups"])
        stored = groups.pop("sha256")
        if semantic_sha256(groups, namespace="image-feature-groups") != stored:
            raise ValueError("Feature-group provenance hash is inconsistent.")
        if groups.get("fit_split") not in {"train", "calibration"} or not groups.get("source_reference_id"):
            raise ValueError("ImageNet feature groups must be frozen from training/calibration reference data.")
        assignments = np.asarray(groups["assignments"])
        if assignments.shape != (1000,):
            raise ValueError("ImageNet feature groups must cover all 1000 classes equally.")
        cells = defaultdict(list)
        for row in rows:
            if row["split"] in {"train", "calibration"}:
                key = tuple(
                    row[key]
                    for key in (
                        "split",
                        "solver",
                        "nfe",
                        "seed",
                        "ensemble_size",
                        "reference_id",
                        "measurement_protocol",
                    )
                )
                cells[key].append(row)
        for key, cell in cells.items():
            schedules = sorted({row["schedule_key"] for row in cell})
            if "uniform" not in schedules:
                raise ValueError("Each ImageNet paired cell requires its uniform anchor.")
            lookup = {(row["schedule_key"], row["class_id"]): row for row in cell}
            if len(lookup) != len(cell) or len(cell) != len(schedules) * 1000:
                raise ValueError("ImageNet paired cells require exactly one measurement per schedule and class.")
            ordered = [[lookup[(schedule, label)] for label in range(1000)] for schedule in schedules]
            kids = np.asarray([[row["metrics"]["kid"] for row in schedule] for schedule in ordered], dtype=float)[None]
            jackknife = np.asarray([[row["jackknife_kid"] for row in schedule] for schedule in ordered], dtype=float)[
                None
            ]
            result = paired_kid_shrinkage(
                kids, jackknife, assignments, uniform_index=schedules.index("uniform"), fit_split=key[0]
            )
            for schedule_index, schedule in enumerate(ordered):
                for label, row in enumerate(schedule):
                    row["reward_metrics"] = {
                        "kid": float(
                            kids[0, schedules.index("uniform"), label]
                            - result["shrunk_improvements"][0, schedule_index, label]
                        )
                    }
                    row["reward_estimator"] = {
                        "context_id": row["context_id"],
                        "class_id": label,
                        "feature_group": int(assignments[label]),
                        "schedule_key": row["schedule_key"],
                        "solver": row["solver"],
                        "nfe": row["nfe"],
                        "protocol": "paired_jackknife_class_group_global_v1",
                        "feature_groups_sha256": stored,
                        "coefficients": result["coefficients"][0, schedule_index, label].tolist(),
                        "standard_error": float(result["standard_errors"][0, schedule_index, label]),
                    }
        metadata["feature_groups"] = {**groups, "sha256": stored}
    return rows, contexts, metadata
