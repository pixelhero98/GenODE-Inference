"""Synthetic paired-image identities; no pretrained scorer or runtime assets."""

from genode.artifacts.identity import semantic_sha256
from genode.gico.image_objective import IMAGE_OBJECTIVE, target_identity


def image_fields(row):
    objective = {
        "protocol": IMAGE_OBJECTIVE,
        "target_generator": {
            "backbone": row["backbone"],
            "checkpoint_sha256": "a" * 64,
            "solver": "rk45",
            "rtol": 1e-6,
            "atol": 1e-6,
            "time_range": [0.0001, 0.9999],
            "precision": "float32",
        },
        "lpips": {
            "network": "vgg",
            "weights_sha256": "b" * 64,
            "implementation_sha256": "c" * 64,
            "input_protocol": "decoded-float32-no-resize-no-clamp",
            "version": "0.1.4",
        },
    }
    panel = row.get("panel_id", row["context_id"])
    seed = int(semantic_sha256([panel, row["seed"]], namespace="fixture-seed").split(":")[-1][:8], 16)
    label = row.get("class_id", 0 if row["task"] == "imagenet64" else None)
    target = {
        "seed": seed,
        "class_id": label,
        "noise_sha256": semantic_sha256([seed], namespace="fixture-noise").split(":")[-1],
        "image_sha256": semantic_sha256([seed, label], namespace="fixture-target").split(":")[-1],
    }
    return {
        **row,
        "panel_id": panel,
        "seed": seed,
        "class_id": label,
        "ensemble_size": 1,
        "image_objective": objective,
        "measurement_protocol": IMAGE_OBJECTIVE,
        "target": target,
        "reference_id": target_identity(objective, target),
    }
