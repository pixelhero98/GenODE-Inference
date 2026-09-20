"""Bind GICO KID or explicit GICO-TF LPIPS evidence to native conditioning."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from genode.backbones.protocol import ImageBackboneManifest
from genode.backbones.registry import get_image_backbone_spec
from genode.gico.clocks import verify_measurement_clock
from genode.gico.collection import validate_collection
from genode.gico.image_conditional_context import context_binding
from genode.gico.image_objective import IMAGE_OBJECTIVE, IMAGE_TASKS, validate_image_rows
from genode.gico.kid_objective import IMAGE_KID_OBJECTIVE


def prepare_image_rows(manifest: dict) -> tuple[list[dict], dict[str, list[float]], dict]:
    """Validate paired KID blocks or GICO-TF target pairs and native class identities.

    Rows carry panel_id independently of per-sample target/reference_id. A panel's
    seeds are averaged together, and never supplied as model conditioning.
    """
    collection = manifest.get("collection_manifest")
    if collection is not None:
        validate_collection(collection, manifest["rows"])
    backbone = ImageBackboneManifest.from_manifest_dict(manifest["backbone_manifest"])
    task = get_image_backbone_spec(backbone.model_key).dataset_key
    if task not in IMAGE_TASKS:
        raise ValueError("Small-image evidence requires CIFAR-10 or ImageNet-64.")
    table = np.asarray(manifest["native_context_table"] if task == "imagenet64" else [[0.0]], dtype=np.float32)
    binding = context_binding(backbone, table)
    contexts, rows = {}, []
    for original in manifest["rows"]:
        row = dict(original)
        protocol = row.get("image_objective", {}).get("protocol")
        if protocol not in (IMAGE_OBJECTIVE, IMAGE_KID_OBJECTIVE):
            raise ValueError("Native image preparation requires paired LPIPS or paired image KID.")
        if row.get("task", task) != task or row.get("backbone", backbone.model_key) != backbone.model_key:
            raise ValueError("Measured image task/backbone does not match its native context binding.")
        if row.get("solver") != "euler":
            raise ValueError("The pinned small-image runtime supports Euler only.")
        label = row.get("class_id")
        if task == "imagenet64" and (type(label) is not int or not 0 <= label < 1000):
            raise ValueError("ImageNet rows require an integer class_id in [0,1000).")
        if task == "cifar10" and label is not None:
            raise ValueError("Unconditional CIFAR-10 evidence cannot carry class labels.")
        source_context = f"class:{label}" if task == "imagenet64" else "unconditional"
        context_id = row["context_id"]
        if not isinstance(context_id, str) or not context_id:
            raise ValueError("Measured image evidence requires its planned context identity.")
        if context_id in contexts and contexts[context_id] != table[label if label is not None else 0].tolist():
            raise ValueError("A collected context cannot refer to different native classes.")
        contexts[context_id] = table[label if label is not None else 0].tolist()
        row.update(
            task=task,
            backbone=backbone.model_key,
            context_id=context_id,
            source_context_id=source_context,
            backbone_binding=(
                binding
                if protocol == IMAGE_OBJECTIVE
                else {**binding, "backbone": backbone.model_key, "clock_protocol": "density64-euler"}
            ),
            context_protocol="native_context_paired_target_panel_v1",
        )
        if row.get("split") not in {"train", "calibration", "validation", "test"}:
            raise ValueError("Image evidence requires an explicit train/calibration/validation/test split.")
        verify_measurement_clock(row)
        rows.append(row)
    if not rows:
        raise ValueError("Image preparation requires nonempty measured rows.")
    validate_image_rows(rows)
    protocol = rows[0]["image_objective"]["protocol"]
    metric = "lpips" if protocol == IMAGE_OBJECTIVE else "kid"
    report_cells = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = tuple(row[key] for key in ("split", "solver", "nfe", "schedule_key"))
        report_cells[key][row["source_context_id"]].append(float(row["metrics"][metric]))
    reports = []
    for key, classes in report_cells.items():
        reports.append(
            {
                **dict(zip(("split", "solver", "nfe", "schedule_key"), key, strict=True)),
                metric: float(np.mean([np.mean(values) for values in classes.values()])),
                "observed_contexts": len(classes),
            }
        )
    if collection is not None:
        # Native binding is a preparation transform; preserve the plan and bind its output.
        from genode.gico.collection import complete_collection

        collection = complete_collection(collection, rows)
    return (
        rows,
        contexts,
        {
            "task": task,
            "backbone_binding": rows[0]["backbone_binding"],
            "raw_metric": metric,
            "protocol": protocol,
            "image_objective": rows[0]["image_objective"],
            "context_protocol": "native_context_paired_target_panel_v1",
            "raw_metric_report": reports,
            "collection_manifest": collection,
        },
    )
