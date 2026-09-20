"""Full-distribution KL and fixed-draw utility for stochastic checkpoints."""

import torch

from genode.gico.clocks import clock_generator
from genode.gico.evidence import content_hash
from genode.gico.selection import candidate_fingerprint
from genode.gico.student_selection import balanced_mean, select_checkpoint

STOCHASTIC_SELECTION_PROTOCOL = "heldout_expected_teacher_utility_distribution_kl"


def select_stochastic(records, allowance=0.20):
    return select_checkpoint(records, allowance)


def distribution_kl(model, condition, references, weights, noise, smoothing):
    """Estimate KL(p||q) with E_p[log(p/q) + q/p - 1].

    p is the complete Gaussian-smoothed reference mixture. Fixed stratified
    component samples reduce comparison noise. Joint log probabilities are
    summed over all 63 coordinates before evaluating the nonnegative estimator.
    """
    centers = model.ratios(references)
    samples = (centers[:, None] + smoothing * noise[None]).reshape(-1, 63)
    values = []
    # Bound Transformer activation memory independently of reference-pool size.
    for start in range(0, len(samples), 32):
        batch = samples[start : start + 32]
        mean, std = model.conditional_parameters(condition.expand(len(batch), -1), batch)
        log_q = torch.distributions.Normal(mean.double(), std.double()).log_prob(batch.double()).sum(-1)
        log_components = (
            torch.distributions.Normal(centers.double(), smoothing).log_prob(batch[:, None].double()).sum(-1)
        )
        log_p = torch.logsumexp(log_components + weights.double().log()[None], dim=-1)
        ratio = log_p - log_q
        estimate = ratio + torch.expm1(-ratio)
        if not bool(torch.isfinite(estimate).all()):
            raise ValueError("Nonfinite stochastic distribution KL; inspect the policy likelihoods.")
        values.append(estimate.clamp_min(0))
    return (torch.cat(values).reshape(len(references), len(noise)).mean(-1) * weights).sum()


@torch.no_grad()
def score_stochastic(model, teacher, conditioning, targets, groups, evidence, weights, step, teacher_id, config):
    if (
        model.training
        or any(m.training for m in teacher.modules())
        or any(p.requires_grad or p.grad is not None for p in teacher.parameters())
    ):
        raise ValueError("Stochastic selection requires an evaluated policy and frozen teacher.")
    utilities, divergences, identities = [], [], []
    for target, group in zip(targets, groups, strict=True):
        condition, references, probability, _, _, teacher_condition = target
        row = group[0]
        identity = content_hash([row["context_id"], row["solver"], row["nfe"]])
        generator = clock_generator(config.seed, f"selection-target:{identity}", device=str(condition.device))
        noise = torch.randn(config.stochastic_likelihood_samples, 63, generator=generator, device=condition.device)
        divergences.append(
            float(distribution_kl(model, condition, references, probability, noise, config.target_smoothing))
        )
        generator = clock_generator(config.seed, f"selection-policy:{identity}", device=str(condition.device))
        mass = model.sample(condition.expand(config.selection_clock_replicates, -1), generator=generator)
        predictions = teacher(teacher_condition.expand(len(mass), -1), mass)
        utility = (predictions * predictions.new_tensor(weights)).sum(-1).double()
        utility *= evidence.calibrations[row["solver"]].reward_scale
        utilities.append(float(utility.mean()))
        identities.append([identity, mass.cpu().tolist(), float(utility.mean()), divergences[-1]])
    return {
        "selection_protocol": STOCHASTIC_SELECTION_PROTOCOL,
        "validation_distillation": balanced_mean(divergences, groups, evidence.task),
        "predicted_utility": balanced_mean(utilities, groups, evidence.task),
        "selection_checkpoint_id": candidate_fingerprint(model, conditioning, "GICO-sto-policy", step),
        "selection_teacher_fingerprint": teacher_id,
        "selection_contexts": sorted({g[0]["context_id"] for g in groups}),
        "selection_groups": len(groups),
        "clock_replicates": config.selection_clock_replicates,
        "kl_samples_per_reference": config.stochastic_likelihood_samples,
        "predictions_sha256": content_hash(identities),
    }
