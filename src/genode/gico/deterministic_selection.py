"""Generator-free deterministic selection with a held-out distillation gate."""

import torch

from genode.gico.evidence import content_hash
from genode.gico.selection import candidate_fingerprint
from genode.gico.student_selection import balanced_mean, select_checkpoint

DETERMINISTIC_SELECTION_PROTOCOL = "heldout_teacher_utility_density_kl"


def select_deterministic(records, allowance=0.15):
    return select_checkpoint(records, allowance)


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
    for target, mass, group in zip(targets, masses, groups, strict=True):
        prediction = teacher(target[5], mass)
        values.append(
            float((prediction * prediction.new_tensor(weights)).sum(-1).item())
            * evidence.calibrations[group[0]["solver"]].reward_scale
        )
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
