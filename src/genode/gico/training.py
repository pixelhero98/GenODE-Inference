"""Shared utility surrogate and two-policy fitting, with differentiable score regularization."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from genode.gico.evidence import Evidence, content_hash, prepare_evidence
from genode.gico.networks import DeterministicPolicy, ModelConfig, StochasticPolicy, UtilitySurrogate, density_kl
from genode.gico.profiles import (
    AUXILIARY_NORMALIZATION,
    REFINEMENT_WEIGHT,
    TEMPERATURE_UNITS,
    TrainingConfig,
    resolve_profile,
)


def source_fingerprint():
    root = Path(__file__).parents[1]
    return content_hash(
        {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*.py"))}
    )


POLICY_KINDS = ("deterministic", "stochastic")


def reuse_utility_surrogate(path, evidence: Evidence, config: TrainingConfig):
    """Load a frozen utility_surrogate only for the exact evidence and utility_surrogate fitting profile."""
    from genode.gico.policy import load_utility_surrogate

    location = Path(path)
    if location.is_dir():
        kind = json.loads((location / "manifest.json").read_text(encoding="utf-8"))["artifact_kind"]
        weight_path = location / ("surrogate.pt" if kind == "utility_surrogate" else "policy.pt")
    else:
        weight_path = location
    digest = hashlib.sha256(weight_path.read_bytes()).hexdigest()
    utility_surrogate, conditioning, metadata = load_utility_surrogate(path)
    if hashlib.sha256(weight_path.read_bytes()).hexdigest() != digest:
        raise ValueError("UtilitySurrogate artifact changed while loading.")
    expected = {
        "task": evidence.task,
        "backbone": evidence.backbone,
        "purpose": evidence.purpose,
        "evidence_sha256": evidence.evidence_sha256,
        "reward_calibrations": {k: v.to_payload() for k, v in evidence.calibrations.items()},
    }
    for key, value in expected.items():
        if content_hash(metadata[key]) != content_hash(value):
            raise ValueError(f"Reused utility_surrogate {key} differs from fitting evidence.")
    expected_conditioning = replace(evidence.conditioning, context_mode=config.utility_surrogate_context_mode)
    if content_hash(conditioning.to_payload()) != content_hash(expected_conditioning.to_payload()):
        raise ValueError("Reused utility_surrogate conditioning differs from fitting evidence.")
    utility_surrogate_fields = {
        "utility_surrogate_profile",
        "utility_surrogate_context_mode",
        "utility_surrogate_steps",
        "utility_surrogate_batch_groups",
        "utility_surrogate_learning_rate",
        "utility_surrogate_checkpoint_every",
        "weight_decay",
        "dropout",
        "temperatures",
        "preferred_temperature",
        "seed",
    }
    for key in utility_surrogate_fields:
        if content_hash(metadata["fitting_profile"][key]) != content_hash(getattr(config, key)):
            raise ValueError(f"Reused utility_surrogate fitting setting {key} differs from its source profile.")
    selection = metadata["history"].get("utility_surrogate_selection")
    if not selection or selection not in metadata["history"].get("utility_surrogate", []):
        raise ValueError("Reused utility_surrogate requires recorded utility_surrogate selection history.")
    if selection["temperature"] != metadata["selected_temperature"]:
        raise ValueError("Reused utility_surrogate selected temperature disagrees with its history.")
    metadata = copy.deepcopy(metadata)
    metadata["reused_artifact_sha256"] = digest
    return utility_surrogate, metadata


def score_coefficient(step: int, steps: int, weight: float, schedule: str = "linear_60_40") -> float:
    from genode.gico.profiles import SCORE_SCHEDULES

    if weight != REFINEMENT_WEIGHT:
        raise ValueError(f"Policy refinement weight is fixed at {REFINEMENT_WEIGHT}.")
    if schedule not in SCORE_SCHEDULES:
        raise ValueError("Unsupported policy refinement schedule.")
    progress = (step + 1) / steps
    if progress <= 0.6:
        return 0.0
    if schedule == "constant_60_40":
        return weight
    ramp = 0.2 if schedule == "ramp_plateau_60_20_20" else 0.4
    return weight * min(1.0, (progress - 0.6) / ramp)


def scalarize(vector: Tensor, metric_weights=None) -> Tensor:
    weights = (
        vector.new_full((vector.shape[-1],), 1 / vector.shape[-1])
        if metric_weights is None
        else vector.new_tensor(metric_weights)
    )
    return (vector * weights).sum(-1)


def utility_surrogate_loss(predicted: Tensor, target: Tensor, metric_weights=None) -> Tensor:
    regression = scalarize(F.huber_loss(predicted, target, reduction="none"), metric_weights).mean()
    score, truth = scalarize(predicted, metric_weights), scalarize(target, metric_weights)
    left, right = torch.triu_indices(len(score), len(score), offset=1, device=score.device)
    differences = truth[left] - truth[right]
    non_ties = differences != 0
    ranking = (
        F.softplus(-(score[left] - score[right])[non_ties] * differences[non_ties].sign() / 0.5).mean()
        if bool(non_ties.any())
        else regression * 0
    )
    return ranking + 0.25 * regression


def utility_surrogate_score(
    utility_surrogate, condition, mass, *, reference_mean=0.0, reference_std=1.0, metric_weights=None
):
    return (
        ((scalarize(utility_surrogate(condition, mass), metric_weights) - reference_mean) / reference_std)
        .clamp(-5, 5)
        .mean()
    )


def reference_weights(scores, *, temperature, reward_scale):
    if not np.isfinite([temperature, reward_scale]).all() or min(temperature, reward_scale) <= 0:
        raise ValueError("Temperature and frozen reward scale must be positive and finite.")
    return (scores.double() * reward_scale / temperature).softmax(-1)


def utility_surrogate_weights(
    utility_surrogate, condition, mass, *, temperature=0.05, reward_scale=1.0, metric_weights=None
):
    return reference_weights(
        scalarize(utility_surrogate(condition, mass), metric_weights),
        temperature=temperature,
        reward_scale=reward_scale,
    )


def utility_regret(predicted, truth, *, temperature, reward_scale):
    weights = reference_weights(predicted, temperature=temperature, reward_scale=reward_scale)
    return (truth.max() - (weights * truth).sum()).clamp_min(0) * reward_scale


def selection_key(regret, temperature, step, preferred):
    return regret, temperature != preferred, step, temperature


def sample_groups(rng, count, batch_size):
    return rng.choice(count, size=min(count, batch_size), replace=False).tolist()


def accumulated_step(model, optimizer, groups, loss_fn, *, microbatch_contexts):
    """One optimizer update with equal weight for each distinct group."""
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for start in range(0, len(groups), microbatch_contexts):
        chunk = groups[start : start + microbatch_contexts]
        losses = loss_fn(chunk)
        if losses.shape != (len(chunk),) or not bool(torch.isfinite(losses).all()):
            raise ValueError("Nonfinite or malformed per-group loss.")
        loss = losses.sum() / len(groups)
        loss.backward()
        total += float(loss.detach())
    nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return total


def _tensors(evidence: Evidence, group: list[dict], device: str) -> tuple[Tensor, Tensor, Tensor, float]:
    conditions = np.array(
        [evidence.conditioning.transform(evidence.contexts[r["context_id"]], r["solver"], r["nfe"]) for r in group]
    )
    return (
        torch.tensor(conditions, dtype=torch.float32, device=device),
        torch.tensor([r["density_mass"] for r in group], dtype=torch.float64, device=device),
        torch.tensor([r["reward_vector"] for r in group], dtype=torch.float32, device=device),
        evidence.calibrations[group[0]["solver"]].reward_scale,
    )


def _predict_groups(utility_surrogate, groups):
    sizes = [len(g[0]) for g in groups]
    return utility_surrogate(torch.cat([g[0] for g in groups]), torch.cat([g[1] for g in groups])).split(sizes)


def _regret(utility_surrogate, groups, config, temperature, weights):
    values = []
    for start in range(0, len(groups), config.microbatch_contexts):
        chunk = groups[start : start + config.microbatch_contexts]
        for prediction, (_, _, truth, scale) in zip(_predict_groups(utility_surrogate, chunk), chunk, strict=True):
            values.append(
                utility_regret(
                    scalarize(prediction, weights),
                    scalarize(truth, weights),
                    temperature=temperature,
                    reward_scale=scale,
                )
            )
    return float(torch.stack(values).mean())


def policy_losses(groups, *, model, kind, utility_surrogate, config, weights, coefficient, generator, score_rng):
    utility_surrogate_conditions = torch.cat([g[5] for g in groups])
    conditions = torch.cat([g[0] for g in groups])
    device = conditions.device
    if kind == "deterministic":
        predicted = model(conditions)
        target = torch.stack([(g[2][:, None] * g[1]).sum(0) for g in groups])
        distillation = density_kl(target, predicted)
        score_condition = utility_surrogate_conditions
    else:
        count = config.stochastic_likelihood_samples
        samples = []
        for _, refs, probability, _, _, _ in groups:
            choice = torch.multinomial(probability, count, replacement=True, generator=generator)
            samples.append(
                model.ratios(refs[choice])
                + config.target_smoothing * torch.randn(count, 63, generator=generator, device=device)
            )
        distillation = (
            model.nll(conditions.repeat_interleave(count, 0), torch.cat(samples)).reshape(len(groups), count).mean(-1)
        )
        score_condition = utility_surrogate_conditions.repeat_interleave(config.stochastic_score_samples, 0)
        predicted = (
            model.sample(conditions.repeat_interleave(config.stochastic_score_samples, 0), generator=score_rng)
            if coefficient
            else None
        )
    if coefficient:
        score = scalarize(utility_surrogate(score_condition, predicted), weights).reshape(len(groups), -1)
        mean = torch.stack([g[3] for g in groups])[:, None]
        std = torch.stack([g[4] for g in groups])[:, None]
        auxiliary = ((score - mean) / std).clamp(-5, 5).mean(-1)
    else:
        auxiliary = distillation * 0
    return distillation - coefficient * auxiliary


def _pooled_density_groups(groups, row_groups):
    """Pool fitting targets by realized density within each solver/NFE setting."""
    by_setting = {}
    for tensors, rows in zip(groups, row_groups, strict=True):
        setting = (rows[0]["solver"], rows[0]["nfe"])
        identities = [row["density_sha256"] for row in rows]
        if (
            len(rows) < 2
            or len(set(identities)) != len(identities)
            or sum(row["schedule_key"] == "uniform" for row in rows) != 1
        ):
            raise ValueError("Staged utility surrogate requires one uniform and unique densities per fitting context.")
        entry = by_setting.setdefault(setting, {"support": set(identities), "targets": {key: [] for key in identities}})
        if set(identities) != entry["support"]:
            raise ValueError("Staged utility surrogate requires complete aligned fitting density coverage.")
        for index, identity in enumerate(identities):
            entry["targets"][identity].append(tensors[2][index])
    pooled = []
    for tensors, rows in zip(groups, row_groups, strict=True):
        setting = (rows[0]["solver"], rows[0]["nfe"])
        targets = by_setting[setting]["targets"]
        mean = torch.stack([torch.stack(targets[row["density_sha256"]]).mean(0) for row in rows])
        pooled.append((tensors[0], tensors[1], mean, tensors[3]))
    return pooled


def _component_mse(model, groups, row_groups, weights):
    errors = []
    for tensors, rows in zip(groups, row_groups, strict=True):
        uniform = [i for i, row in enumerate(rows) if row["schedule_key"] == "uniform"]
        candidates = [i for i, row in enumerate(rows) if row["schedule_key"] != "uniform"]
        if len(uniform) != 1 or not candidates or not bool((tensors[2][uniform[0]] == 0).all()):
            raise ValueError(
                "Utility-surrogate selection requires measured candidates and one zero-gain uniform anchor."
            )
        prediction = model(tensors[0], tensors[1])
        error = prediction[candidates] - prediction[uniform[0]] - tensors[2][candidates]
        errors.append(float(scalarize(error.square(), weights).mean()) * tensors[3] ** 2)
    if not errors or not np.isfinite(errors).all():
        raise ValueError("Utility-surrogate selection requires finite held-out component errors.")
    return float(np.mean(errors))


def _fit_utility_surrogate(
    architecture,
    train,
    train_rows,
    validation,
    validation_rows,
    density_validation,
    density_rows,
    weights,
    config,
    device,
    native_context_width,
):
    # fork_rng isolates initialization from callers' generation RNG state.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed)
        utility_surrogate = UtilitySurrogate(architecture).to(device)
    utility_surrogate.native_context_width = native_context_width
    staged = config.utility_surrogate_profile == "density_context_projection"
    pooled = _pooled_density_groups(train, train_rows) if staged else None
    rng = np.random.default_rng(config.seed)
    optimizer = torch.optim.AdamW(
        utility_surrogate.parameters(), lr=config.utility_surrogate_learning_rate, weight_decay=config.weight_decay
    )
    best, best_state, utility_surrogate_history, selected_utility_surrogate = None, None, [], None
    frozen_backbone = None

    def utility_surrogate_losses(groups):
        return torch.stack(
            [
                scalarize(F.huber_loss(p, g[2], reduction="none"), weights).mean()
                if staged
                else utility_surrogate_loss(p, g[2], weights)
                for p, g in zip(_predict_groups(utility_surrogate, groups), groups, strict=True)
            ]
        )

    for step in range(config.utility_surrogate_steps):
        stage = "density" if staged and step < 500 else "context" if staged else "ranked_regression"
        if staged:
            utility_surrogate.density_only = stage == "density"
            if step == 500:
                frozen_backbone = {
                    key: value.detach().clone()
                    for key, value in utility_surrogate.state_dict().items()
                    if not key.startswith("condition.")
                }
                for name, parameter in utility_surrogate.named_parameters():
                    parameter.requires_grad_(name.startswith("condition."))
        utility_surrogate.train()
        indices = sample_groups(rng, len(train), config.utility_surrogate_batch_groups)
        source = pooled if stage == "density" else train
        loss = accumulated_step(
            utility_surrogate,
            optimizer,
            [source[i] for i in indices],
            utility_surrogate_losses,
            microbatch_contexts=config.microbatch_contexts,
        )
        if (step + 1) % config.utility_surrogate_checkpoint_every == 0 or step + 1 == config.utility_surrogate_steps:
            utility_surrogate.eval()
            with torch.no_grad():
                context_mse = _component_mse(utility_surrogate, validation, validation_rows, weights)
                density_mse = (
                    _component_mse(utility_surrogate, density_validation, density_rows, weights)
                    if density_validation
                    else None
                )
                component_mse = context_mse if density_mse is None else (context_mse + density_mse) / 2
                checkpoint_rows = []
                for temperature in config.temperatures:
                    context = _regret(utility_surrogate, validation, config, temperature, weights)
                    family = (
                        _regret(utility_surrogate, density_validation, config, temperature, weights)
                        if density_validation
                        else None
                    )
                    regret = context if family is None else (context + family) / 2
                    row = {
                        "step": step + 1,
                        "temperature": temperature,
                        "regret": regret,
                        "context_regret": context,
                        "density_regret": family,
                        "component_mse": component_mse,
                        "context_mse": context_mse,
                        "density_mse": density_mse,
                        "stage": stage,
                        "training_loss": loss,
                        "groups": len(indices),
                    }
                    utility_surrogate_history.append(row)
                    checkpoint_rows.append(row)
                    if not np.isfinite([regret, component_mse]).all():
                        raise ValueError("Nonfinite held-out utility-surrogate selection score.")
                selected_temperature = min(
                    checkpoint_rows,
                    key=lambda row: selection_key(
                        row["regret"], row["temperature"], row["step"], config.preferred_temperature
                    ),
                )
                key = (component_mse, step + 1)
                if best is None or key < best:
                    best, best_state = key, copy.deepcopy(utility_surrogate.state_dict())
                    selected_utility_surrogate = dict(selected_temperature)
    if best_state is None:
        raise ValueError("UtilitySurrogate validation did not produce a finite checkpoint.")
    if frozen_backbone is not None and any(
        not torch.equal(value, utility_surrogate.state_dict()[key]) for key, value in frozen_backbone.items()
    ):
        raise RuntimeError("Frozen density backbone changed during context projection fitting.")
    utility_surrogate.load_state_dict(best_state)
    utility_surrogate.density_only = selected_utility_surrogate["stage"] == "density"
    return utility_surrogate, utility_surrogate_history, selected_utility_surrogate


def fit_models(
    evidence: Evidence,
    config: TrainingConfig,
    *,
    policy_kind: str = "deterministic",
    device: str = "cuda",
    checkpoint_callback=None,
    utility_surrogate_artifact=None,
):
    # manual_seed seeds every visible GPU, including when fitting on the CPU.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed)
        return _fit_models(
            evidence,
            config,
            policy_kind=policy_kind,
            device=device,
            checkpoint_callback=checkpoint_callback,
            utility_surrogate_artifact=utility_surrogate_artifact,
        )


def _fit_models(evidence, config, *, policy_kind, device, checkpoint_callback=None, utility_surrogate_artifact=None):
    evidence = replace(
        evidence, conditioning=replace(evidence.conditioning, context_mode=config.utility_surrogate_context_mode)
    )
    if policy_kind not in (*POLICY_KINDS, "both", None):
        raise ValueError("policy_kind must be deterministic, stochastic, both, or null for surrogate-only fitting.")
    train_groups, val_groups = evidence.groups("train"), evidence.groups("validation")
    utility_surrogate_groups = [[r for r in group if not evidence.is_density_holdout(r)] for group in train_groups]
    context_selection_rows = [
        eligible
        for group in val_groups
        if len(eligible := [r for r in group if not evidence.is_density_holdout(r)]) > 1
    ]
    density_holdout = []
    for group in train_groups:
        held = [r for r in group if evidence.is_density_holdout(r)]
        if held:
            density_holdout.append([r for r in group if r["schedule_key"] == "uniform"] + held)
    if evidence.purpose == "research" and not density_holdout:
        raise ValueError("Research utility_surrogate fitting requires measured density holdout pairs.")
    if not context_selection_rows:
        raise ValueError("Utility-surrogate selection requires context-only measured candidate pairs.")
    if config.utility_surrogate_profile == "density_context_projection" and evidence.conditioning.unconditional:
        raise ValueError("Density-to-context fitting requires a contextual task.")
    train = [_tensors(evidence, g, device) for g in utility_surrogate_groups if g]
    validation = [_tensors(evidence, g, device) for g in context_selection_rows]
    density_validation = [_tensors(evidence, g, device) for g in density_holdout]
    weights = next(iter(evidence.calibrations.values())).metric_weights
    architecture = ModelConfig(evidence.conditioning.width, len(weights), dropout=config.dropout)
    # Fit the ratio transform only on unique training-reference densities.
    unique = {}
    for group in utility_surrogate_groups:
        for row in group:
            unique[row["density_sha256"]] = row["density_mass"]
    reference = torch.tensor(list(unique.values()), dtype=torch.float64, device=device)
    from genode.gico.networks import guarded_mass

    logmass = guarded_mass(reference).log()
    ratios = logmass[:, :-1] - logmass[:, -1:]
    ratio_mean, ratio_scale = ratios.mean(0), ratios.std(0, unbiased=False)
    ratio_scale = torch.where(ratio_scale < 1e-6, 1, ratio_scale)
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    if utility_surrogate_artifact is None:
        utility_surrogate, utility_surrogate_history, selected_utility_surrogate = _fit_utility_surrogate(
            architecture,
            train,
            utility_surrogate_groups,
            validation,
            context_selection_rows,
            density_validation,
            density_holdout,
            weights,
            config,
            device,
            len(evidence.conditioning.context.mean),
        )
        utility_surrogate_source = None
    else:
        utility_surrogate, source = reuse_utility_surrogate(utility_surrogate_artifact, evidence, config)
        utility_surrogate = utility_surrogate.to(device)
        utility_surrogate_history = copy.deepcopy(source["history"]["utility_surrogate"])
        selected_utility_surrogate = copy.deepcopy(source["history"]["utility_surrogate_selection"])
        utility_surrogate_source = source["reused_artifact_sha256"]
    utility_surrogate.eval().requires_grad_(False)
    utility_surrogate.zero_grad(set_to_none=True)

    def support_groups(groups):
        result = []
        for group in groups:
            first = group[0]
            pool = evidence.reference_support[f"{first['solver']}:{first['nfe']}"]
            unique = {}
            for value in pool.values():
                unique.setdefault(value["density_identity"], value["density_mass"])
            condition = torch.tensor(
                evidence.conditioning.transform(evidence.contexts[first["context_id"]], first["solver"], first["nfe"]),
                dtype=torch.float32,
                device=device,
            )[None].expand(len(unique), -1)
            masses = torch.tensor(list(unique.values()), dtype=torch.float64, device=device)
            # Unmeasured support has no truth tensor: these are utility_surrogate queries only.
            result.append((condition, masses, None, evidence.calibrations[first["solver"]].reward_scale))
        return result

    training_groups = support_groups(train_groups)
    policy_validation = support_groups(val_groups)
    temperature = selected_utility_surrogate["temperature"]

    from genode.gico.selection import utility_surrogate_fingerprint

    utility_surrogate_selection_fingerprint = utility_surrogate_fingerprint(
        utility_surrogate,
        evidence.conditioning,
        selected_utility_surrogate["step"],
        selected_utility_surrogate["temperature"],
        selected_utility_surrogate["stage"],
    )
    policy_conditioning = replace(evidence.conditioning, context_mode=config.policy_context_mode)

    def make_targets(groups, rows):
        result = []
        with torch.no_grad():
            for start in range(0, len(groups), config.microbatch_contexts):
                chunk = groups[start : start + config.microbatch_contexts]
                for offset, (prediction, (c, m, _, scale)) in enumerate(
                    zip(_predict_groups(utility_surrogate, chunk), chunk, strict=True)
                ):
                    row = rows[start + offset][0]
                    policy_condition = c.new_tensor(
                        policy_conditioning.transform(evidence.contexts[row["context_id"]], row["solver"], row["nfe"])
                    )[None]
                    score = scalarize(prediction, weights)
                    std = score.std(unbiased=False)
                    std = torch.where(std < 1e-6, torch.ones_like(std), std)
                    result.append(
                        (
                            policy_condition,
                            m,
                            reference_weights(score, temperature=temperature, reward_scale=scale),
                            score.mean(),
                            std,
                            c[:1],
                        )
                    )
        return result

    targets, validation_targets = (
        make_targets(training_groups, train_groups),
        make_targets(policy_validation, val_groups),
    )
    policies, histories, selections = {}, {}, {}
    kinds = POLICY_KINDS if policy_kind == "both" else () if policy_kind is None else (policy_kind,)
    for kind in kinds:
        deterministic = kind == "deterministic"
        checkpoint_every = config.deterministic_checkpoint_every if deterministic else config.policy_checkpoint_every
        candidate_states = {}
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(config.seed)
            model = (
                DeterministicPolicy(architecture)
                if kind == "deterministic"
                else StochasticPolicy(architecture, ratio_mean, ratio_scale)
            ).to(device)
        torch.manual_seed(config.seed + 419)  # Independent dropout stream for each policy kind.
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.policy_learning_rate, weight_decay=config.weight_decay
        )
        group_rng = np.random.default_rng(config.seed + 113)
        generator = torch.Generator(device=device).manual_seed(config.seed + 173)
        score_rng = torch.Generator(device=device).manual_seed(config.seed + 271)
        history, best_state, chosen = [], None, None

        for step in range(config.policy_steps):
            model.train()
            coefficient = score_coefficient(step, config.policy_steps, config.refinement_weight, config.score_schedule)
            indices = sample_groups(group_rng, len(targets), config.policy_batch_contexts)
            loss = accumulated_step(
                model,
                optimizer,
                [targets[i] for i in indices],
                partial(
                    policy_losses,
                    model=model,
                    kind=kind,
                    utility_surrogate=utility_surrogate,
                    config=config,
                    weights=weights,
                    coefficient=coefficient,
                    generator=generator,
                    score_rng=score_rng,
                ),
                microbatch_contexts=config.microbatch_contexts,
            )
            if any(p.grad is not None for p in utility_surrogate.parameters()):
                raise RuntimeError("Frozen utility_surrogate unexpectedly accumulated parameter gradients.")
            if (step + 1) % checkpoint_every == 0 or step + 1 == config.policy_steps:
                model.eval()
                validation_masses = []
                value = None
                if deterministic:
                    from genode.gico.policy_selection import balanced_mean

                    with torch.no_grad():
                        losses = []
                        for c, m, w, _, _, _ in validation_targets:
                            mass = model(c)
                            validation_masses.append(mass)
                            losses.append(float(density_kl((w[:, None] * m).sum(0, keepdim=True), mass).mean()))
                        value = balanced_mean(losses, val_groups, evidence.task)
                row = {
                    "step": step + 1,
                    "objective": loss,
                    "coefficient": coefficient,
                    "validation_distillation": value,
                    "contexts": len(indices),
                }
                if coefficient > 0:
                    if deterministic:
                        from genode.gico.deterministic_selection import score_deterministic

                        row.update(
                            score_deterministic(
                                model,
                                utility_surrogate,
                                policy_conditioning,
                                validation_targets,
                                validation_masses,
                                val_groups,
                                evidence,
                                weights,
                                step + 1,
                                utility_surrogate_selection_fingerprint,
                            )
                        )
                    else:
                        from genode.gico.stochastic_selection import score_stochastic

                        row.update(
                            score_stochastic(
                                model,
                                utility_surrogate,
                                policy_conditioning,
                                validation_targets,
                                val_groups,
                                evidence,
                                weights,
                                step + 1,
                                utility_surrogate_selection_fingerprint,
                                config,
                            )
                        )
                history.append(row)
                if checkpoint_callback is not None:
                    checkpoint_callback(kind, step + 1, model, dict(row))
                if coefficient > 0:
                    from genode.gico.deterministic_selection import select_deterministic
                    from genode.gico.policy_selection import admissible_checkpoints
                    from genode.gico.stochastic_selection import select_stochastic

                    allowance = config.deterministic_kl_allowance if deterministic else config.stochastic_kl_allowance
                    selector = select_deterministic if deterministic else select_stochastic
                    candidate_states[row["step"]] = {
                        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                    }
                    chosen = dict(selector(history, allowance))
                    best_state = candidate_states[chosen["step"]]
                    retained = {r["step"] for r in admissible_checkpoints(history, allowance)}
                    candidate_states = {key: state for key, state in candidate_states.items() if key in retained}
        if best_state is None:
            raise ValueError("Policy validation did not produce a finite checkpoint after the score ramp.")
        model.load_state_dict(best_state)
        policies[kind] = model.eval().requires_grad_(False).cpu()
        histories[kind] = history
        selections[kind] = chosen
    return (
        utility_surrogate.cpu(),
        policies,
        {
            "training": asdict(config),
            "evidence_fingerprint": evidence.evidence_sha256,
            "calibration_fingerprint": content_hash({k: v.to_payload() for k, v in evidence.calibrations.items()}),
            "collection_fingerprint": content_hash(evidence.collection_manifest),
            "support_fingerprint": content_hash(evidence.reference_support),
            "source_fingerprint": source_fingerprint(),
            "utility_surrogate_source_artifact_sha256": utility_surrogate_source,
            "utility_surrogate": utility_surrogate_history,
            "policies": histories,
            "density_holdout": evidence.density_holdout,
            "utility_surrogate_selection": selected_utility_surrogate,
            "utility_surrogate_selection_fingerprint": utility_surrogate_selection_fingerprint,
            "policy_selection": selections,
            "temperature_units": TEMPERATURE_UNITS,
            "auxiliary_normalization": AUXILIARY_NORMALIZATION,
        },
    )


def fit(
    rows: list[dict],
    contexts: dict,
    output,
    *,
    policy_kind: str = "deterministic",
    device: str = "cuda",
    purpose: str = "research",
    calibration_rows: list[dict] | None = None,
    collection_manifest: dict | None = None,
    checkpoint_callback=None,
    utility_surrogate_artifact=None,
    **fitting_settings,
) -> dict:
    if Path(output).exists():
        raise FileExistsError(f"Artifact destination already exists: {output}")
    bindings = [r["backbone_binding"] for r in rows if "backbone_binding" in r]
    if bindings and (len(bindings) != len(rows) or any(value != bindings[0] for value in bindings)):
        raise ValueError("Native backbone/context bindings differ between measurements.")
    evidence = prepare_evidence(
        rows, contexts, calibration_rows=calibration_rows, purpose=purpose, collection_manifest=collection_manifest
    )
    config = resolve_profile(evidence.task, **fitting_settings)
    if config.backbone is not None and config.backbone != evidence.backbone:
        raise ValueError("Fitting profile backbone differs from measurement evidence.")
    config = replace(config, backbone=evidence.backbone)
    evidence = replace(
        evidence, conditioning=replace(evidence.conditioning, context_mode=config.utility_surrogate_context_mode)
    )
    started = time.perf_counter()
    utility_surrogate, policies, history = fit_models(
        evidence,
        config,
        policy_kind=policy_kind,
        device=device,
        checkpoint_callback=checkpoint_callback,
        utility_surrogate_artifact=utility_surrogate_artifact,
    )
    fitting_seconds = time.perf_counter() - started
    metadata = {
        "task": evidence.task,
        "backbone": evidence.backbone,
        "purpose": purpose,
        "solvers": list(evidence.calibrations),
        "reward_calibrations": {k: v.to_payload() for k, v in evidence.calibrations.items()},
        "evidence_sha256": evidence.evidence_sha256,
        "history": history,
        "fitting_profile": {"task": evidence.task, **asdict(config)},
        "metric_weights": list(next(iter(evidence.calibrations.values())).metric_weights),
        "temperature_units": TEMPERATURE_UNITS,
        "auxiliary_normalization": AUXILIARY_NORMALIZATION,
        "utility_surrogate_selection_criterion": "heldout_component_mse_then_reference_regret",
        "policy_selection_criterion": "heldout_calibrated_utility_surrogate_utility_with_policy_kl",
        "utility_surrogate_prediction_semantics": (
            "density_only" if history["utility_surrogate_selection"]["stage"] == "density" else "native_context"
        ),
        "collection_manifest": evidence.collection_manifest,
        "collection_sha256": content_hash(evidence.collection_manifest),
        "source_code_sha256": source_fingerprint(),
        "selected_temperature": history["utility_surrogate_selection"]["temperature"],
        "split_contexts": {
            s: sorted({r["context_id"] for r in evidence.cells if r["split"] == s}) for s in ("train", "validation")
        },
        "reference_densities": {
            f"{setting}:{name}": value["density_mass"]
            for setting, pool in evidence.reference_support.items()
            for name, value in pool.items()
        },
        "reference_grids": {
            f"{setting}:{name}": value["time_grid"]
            for setting, pool in evidence.reference_support.items()
            for name, value in pool.items()
        },
        "measurement_protocols": sorted({r["measurement_protocol"] for r in evidence.cells}),
        "locked_test_used": False,
        "torch_version": str(torch.__version__),
        "fitting_wall_seconds": fitting_seconds,
        "measurement_counts": {
            split: sum(r["split"] == split for r in rows + (calibration_rows or []))
            for split in ("train", "validation", "calibration")
        },
    }
    if bindings:
        metadata["backbone_binding"] = bindings[0]
    if evidence.task in ("cifar10", "imagenet64"):
        metadata["image_objective"] = rows[0]["image_objective"]
        from genode.gico.image_objective import image_split_fields

        metadata["image_split_identities"] = {
            phase: {
                key: sorted(
                    {
                        value
                        for r in rows + (calibration_rows or [])
                        if r["split"] == phase
                        for value in image_split_fields(r)[key]
                    }
                )
                for key in image_split_fields(rows[0])
            }
            for phase in ("train", "calibration", "validation")
        }
    maps = {
        content_hash(r["molecule_feature_map"]): r["molecule_feature_map"]
        for r in rows + (calibration_rows or [])
        if "molecule_feature_map" in r
    }
    metadata["molecular_feature_maps"] = maps
    from genode.gico.policy import save_artifact

    save_artifact(
        output,
        utility_surrogate,
        policies,
        replace(
            evidence.conditioning,
            context_mode=config.policy_context_mode if policies else config.utility_surrogate_context_mode,
        ),
        metadata,
    )
    return metadata
