"""Generator-free deterministic selection with a held-out distillation gate."""

import torch

from genode.gico.evidence import content_hash
from genode.gico.policy_selection import balanced_mean, select_checkpoint
from genode.gico.selection import candidate_fingerprint

DETERMINISTIC_SELECTION_PROTOCOL = "heldout_utility_surrogate_utility_density_kl"


def select_deterministic(records, allowance=0.15):
    return select_checkpoint(records, allowance)


@torch.no_grad()
def score_deterministic(
    model, utility_surrogate, conditioning, targets, masses, groups, evidence, weights, step, utility_surrogate_id
):
    """Reuse each validation density; evaluate its native utility_surrogate exactly once."""
    if (
        any(module.training for module in utility_surrogate.modules())
        or any(parameter.requires_grad or parameter.grad is not None for parameter in utility_surrogate.parameters())
        or model.training
        or len(masses) != len(groups)
        or len(targets) != len(groups)
    ):
        raise ValueError("Deterministic selection requires frozen utility_surrogate and complete validation densities.")
    values = []
    for target, mass, group in zip(targets, masses, groups, strict=True):
        prediction = utility_surrogate(target[5], mass)
        values.append(
            float((prediction * prediction.new_tensor(weights)).sum(-1).item())
            * evidence.calibrations[group[0]["solver"]].reward_scale
        )
    return {
        "selection_protocol": DETERMINISTIC_SELECTION_PROTOCOL,
        "predicted_utility": balanced_mean(values, groups, evidence.task),
        "selection_checkpoint_id": candidate_fingerprint(model, conditioning, "deterministic", step),
        "selection_utility_surrogate_fingerprint": utility_surrogate_id,
        "selection_contexts": sorted({group[0]["context_id"] for group in groups}),
        "selection_groups": len(groups),
        "predictions_sha256": content_hash(
            [[g[0]["context_id"], g[0]["solver"], g[0]["nfe"], value] for g, value in zip(groups, values, strict=True)]
        ),
    }
