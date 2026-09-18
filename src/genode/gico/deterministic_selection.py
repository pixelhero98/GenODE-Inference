"""Generator-free deterministic selection with a held-out distillation gate."""

from collections import defaultdict

import numpy as np
import torch

from genode.gico.evidence import content_hash
from genode.gico.selection import candidate_fingerprint

DETERMINISTIC_SELECTION_PROTOCOL = "heldout_teacher_utility_kl_gate_v1"
STUDENT_SELECTION_PROTOCOL = "det_teacher_kl_gate_sto_measured_utility_v1"


def balanced_mean(values, groups, task):
    """Weight settings equally, with equal ImageNet class weight within each."""
    settings = defaultdict(lambda: defaultdict(list))
    if len(values) != len(groups) or not values or not np.isfinite(values).all():
        raise ValueError("Selection requires complete finite context/settings values.")
    for value, group in zip(values, groups, strict=True):
        row = group[0]
        unit = row["class_id"] if task == "imagenet64" else row["context_id"]
        settings[row["solver"], row["nfe"]][unit].append(float(value))
    return float(np.mean([np.mean([np.mean(v) for v in units.values()]) for units in settings.values()]))


def select_deterministic(records, allowance):
    """Choose highest teacher score inside the final relative-KL allowance."""
    eligible = [row for row in records if row.get("coefficient", 0) > 0]
    if not eligible or any(
        not np.isfinite(row.get("validation_distillation", np.nan))
        or row["validation_distillation"] < -1e-12
        or not np.isfinite(row.get("predicted_utility", np.nan))
        for row in eligible
    ):
        raise ValueError("Deterministic selection requires finite teacher scores and nonnegative KL.")
    if isinstance(allowance, bool) or not np.isfinite(allowance) or allowance < 0:
        raise ValueError("Deterministic KL allowance must be finite and nonnegative.")
    minimum = min(max(0.0, row["validation_distillation"]) for row in eligible)
    threshold = minimum + allowance * minimum
    admitted = [row for row in eligible if row["validation_distillation"] <= threshold]
    return min(admitted, key=lambda row: (-row["predicted_utility"], row["step"]))


@torch.no_grad()
def score_deterministic(model, teacher, conditioning, targets, masses, groups, evidence, weights, step, teacher_id):
    """Reuse each validation density; evaluate its native teacher exactly once."""
    if (
        any(module.training for module in teacher.modules())
        or any(parameter.requires_grad or parameter.grad is not None for parameter in teacher.parameters())
        or model.training
        or len(masses) != len(groups)
        or len(targets) != len(groups)
    ):
        raise ValueError("Deterministic selection requires frozen teacher and complete validation densities.")
    values = []
    for target, mass in zip(targets, masses, strict=True):
        prediction = teacher(target[5], mass)
        values.append(float((prediction * prediction.new_tensor(weights)).sum(-1).item()))
    return {
        "selection_protocol": DETERMINISTIC_SELECTION_PROTOCOL,
        "predicted_utility": balanced_mean(values, groups, evidence.task),
        "selection_checkpoint_id": candidate_fingerprint(model, conditioning, "GICO-det-policy", step),
        "selection_teacher_fingerprint": teacher_id,
        "selection_contexts": sorted({group[0]["context_id"] for group in groups}),
        "selection_groups": len(groups),
        "predictions_sha256": content_hash(
            [[g[0]["context_id"], g[0]["solver"], g[0]["nfe"], value] for g, value in zip(groups, values, strict=True)]
        ),
    }
