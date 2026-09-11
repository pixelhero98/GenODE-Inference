"""Shared teacher and two-student fitting, with differentiable score regularization."""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from genode.gico.evidence import Evidence, prepare_evidence
from genode.gico.networks import DensityTeacher, DeterministicStudent, ModelConfig, StochasticStudent, density_kl
from genode.gico.profiles import (
    AUXILIARY_NORMALIZATION,
    SCORE_WEIGHTS,
    TEMPERATURE_UNITS,
    TrainingConfig,
    resolve_profile,
)

STUDENT_KINDS = ("deterministic", "stochastic")


def score_coefficient(step: int, steps: int, weight: float) -> float:
    if weight not in SCORE_WEIGHTS:
        raise ValueError("Unsupported teacher-score weight.")
    return weight * max(0.0, min(1.0, ((step + 1) / steps - 0.6) / 0.4))


def scalarize(vector: Tensor, metric_weights=None) -> Tensor:
    weights = (
        vector.new_full((vector.shape[-1],), 1 / vector.shape[-1])
        if metric_weights is None
        else vector.new_tensor(metric_weights)
    )
    return (vector * weights).sum(-1)


def teacher_loss(predicted: Tensor, target: Tensor, metric_weights=None) -> Tensor:
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


def teacher_score(teacher, condition, mass, *, reference_mean=0.0, reference_std=1.0, metric_weights=None):
    return ((scalarize(teacher(condition, mass), metric_weights) - reference_mean) / reference_std).clamp(-5, 5).mean()


def reference_weights(scores, *, temperature, reward_scale):
    if not np.isfinite([temperature, reward_scale]).all() or min(temperature, reward_scale) <= 0:
        raise ValueError("Temperature and frozen reward scale must be positive and finite.")
    return (scores.double() * reward_scale / temperature).softmax(-1)


def teacher_weights(teacher, condition, mass, *, temperature=0.05, reward_scale=1.0, metric_weights=None):
    return reference_weights(
        scalarize(teacher(condition, mass), metric_weights), temperature=temperature, reward_scale=reward_scale
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


def _predict_groups(teacher, groups):
    sizes = [len(g[0]) for g in groups]
    return teacher(torch.cat([g[0] for g in groups]), torch.cat([g[1] for g in groups])).split(sizes)


def _regret(teacher, groups, config, temperature, weights):
    values = []
    for start in range(0, len(groups), config.microbatch_contexts):
        chunk = groups[start : start + config.microbatch_contexts]
        for prediction, (_, _, truth, scale) in zip(_predict_groups(teacher, chunk), chunk, strict=True):
            values.append(
                utility_regret(
                    scalarize(prediction, weights),
                    scalarize(truth, weights),
                    temperature=temperature,
                    reward_scale=scale,
                )
            )
    return float(torch.stack(values).mean())


def student_losses(groups, *, model, kind, teacher, config, weights, coefficient, generator, score_rng):
    teacher_conditions = torch.cat([g[5] for g in groups])
    conditions = torch.cat([g[0] for g in groups])
    device = conditions.device
    if kind == "deterministic":
        predicted = model(conditions)
        target = torch.stack([(g[2][:, None] * g[1]).sum(0) for g in groups])
        distillation = density_kl(target, predicted)
        score_condition = teacher_conditions
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
        score_condition = teacher_conditions.repeat_interleave(config.stochastic_score_samples, 0)
        predicted = (
            model.sample(conditions.repeat_interleave(config.stochastic_score_samples, 0), generator=score_rng)
            if coefficient
            else None
        )
    if coefficient:
        score = scalarize(teacher(score_condition, predicted), weights).reshape(len(groups), -1)
        mean = torch.stack([g[3] for g in groups])[:, None]
        std = torch.stack([g[4] for g in groups])[:, None]
        auxiliary = ((score - mean) / std).clamp(-5, 5).mean(-1)
    else:
        auxiliary = distillation * 0
    return distillation - coefficient * auxiliary


def fit_models(
    evidence: Evidence,
    config: TrainingConfig,
    *,
    student_kind: str = "both",
    device: str = "cuda",
    checkpoint_callback=None,
):
    devices = [torch.device(device).index or 0] if str(device).startswith("cuda") else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed)
        return _fit_models(
            evidence, config, student_kind=student_kind, device=device, checkpoint_callback=checkpoint_callback
        )


def _fit_models(evidence, config, *, student_kind, device, checkpoint_callback=None):
    evidence = replace(evidence, conditioning=replace(evidence.conditioning, context_mode=config.teacher_context_mode))
    if student_kind not in (*STUDENT_KINDS, "both"):
        raise ValueError("student_kind must be deterministic, stochastic, or both.")
    train_groups, val_groups = evidence.groups("train"), evidence.groups("validation")
    # Leave an entire density family out of teacher fitting; its observations
    # remain available for validation, never for reference-ratio normalization.
    family_holdout = set(config.density_family_holdout)
    teacher_groups = [[r for r in g if not set(r["aliases"]) <= family_holdout] for g in train_groups]
    density_holdout = [[r for r in g if set(r["aliases"]) <= family_holdout] for g in train_groups]
    density_holdout = [g for g in density_holdout if g]
    if family_holdout and evidence.purpose == "research" and not density_holdout:
        raise ValueError("Research teacher fitting requires its density-family holdout.")
    train = [_tensors(evidence, g, device) for g in teacher_groups if g]
    validation = [_tensors(evidence, g, device) for g in val_groups]
    density_validation = [_tensors(evidence, g, device) for g in density_holdout]
    weights = next(iter(evidence.calibrations.values())).metric_weights
    architecture = ModelConfig(evidence.conditioning.width, len(weights), dropout=config.dropout)
    # Fit the ratio transform only on unique training-reference densities.
    unique = {}
    for group in teacher_groups:
        for row in group:
            unique[row["density_sha256"]] = row["density_mass"]
    reference = torch.tensor(list(unique.values()), dtype=torch.float64, device=device)
    from genode.gico.networks import guarded_mass

    logmass = guarded_mass(reference).log()
    ratios = logmass[:, :-1] - logmass[:, -1:]
    ratio_mean, ratio_scale = ratios.mean(0), ratios.std(0, unbiased=False)
    ratio_scale = torch.where(ratio_scale < 1e-6, 1, ratio_scale)
    # fork_rng isolates initialization from callers' generation RNG state.
    devices = [torch.device(device).index or 0] if str(device).startswith("cuda") else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed)
        teacher = DensityTeacher(architecture).to(device)
    rng = np.random.default_rng(config.seed)
    optimizer = torch.optim.AdamW(
        teacher.parameters(), lr=config.teacher_learning_rate, weight_decay=config.weight_decay
    )
    best, best_state, teacher_history, selected_teacher = None, None, [], None

    def teacher_losses(groups):
        return torch.stack(
            [teacher_loss(p, g[2], weights) for p, g in zip(_predict_groups(teacher, groups), groups, strict=True)]
        )

    for step in range(config.teacher_steps):
        teacher.train()
        indices = sample_groups(rng, len(train), config.teacher_batch_groups)
        loss = accumulated_step(
            teacher,
            optimizer,
            [train[i] for i in indices],
            teacher_losses,
            microbatch_contexts=config.microbatch_contexts,
        )
        if (step + 1) % config.teacher_checkpoint_every == 0 or step + 1 == config.teacher_steps:
            teacher.eval()
            with torch.no_grad():
                for temperature in config.temperatures:
                    context = _regret(teacher, validation, config, temperature, weights)
                    family = (
                        _regret(teacher, density_validation, config, temperature, weights)
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
                        "training_loss": loss,
                        "groups": len(indices),
                    }
                    teacher_history.append(row)
                    if not np.isfinite(regret):
                        raise ValueError("Nonfinite held-out teacher utility regret.")
                    key = selection_key(regret, temperature, step + 1, config.preferred_temperature)
                    if best is None or key < best:
                        best, best_state, selected_teacher = key, copy.deepcopy(teacher.state_dict()), dict(row)
    if best_state is None:
        raise ValueError("Teacher validation did not produce a finite checkpoint.")
    teacher.load_state_dict(best_state)
    teacher.eval().requires_grad_(False)
    teacher.zero_grad(set_to_none=True)
    training_groups = [_tensors(evidence, g, device) for g in train_groups]
    temperature = selected_teacher["temperature"]

    student_conditioning = replace(evidence.conditioning, context_mode=config.student_context_mode)

    def make_targets(groups, rows):
        result = []
        with torch.no_grad():
            for start in range(0, len(groups), config.microbatch_contexts):
                chunk = groups[start : start + config.microbatch_contexts]
                for offset, (prediction, (c, m, _, scale)) in enumerate(
                    zip(_predict_groups(teacher, chunk), chunk, strict=True)
                ):
                    row = rows[start + offset][0]
                    student_condition = c.new_tensor(
                        student_conditioning.transform(evidence.contexts[row["context_id"]], row["solver"], row["nfe"])
                    )[None]
                    score = scalarize(prediction, weights)
                    std = score.std(unbiased=False)
                    std = torch.where(std < 1e-6, torch.ones_like(std), std)
                    result.append(
                        (
                            student_condition,
                            m,
                            reference_weights(score, temperature=temperature, reward_scale=scale),
                            score.mean(),
                            std,
                            c[:1],
                        )
                    )
        return result

    targets, validation_targets = make_targets(training_groups, train_groups), make_targets(validation, val_groups)
    students, histories, selections = {}, {}, {}
    kinds = STUDENT_KINDS if student_kind == "both" else (student_kind,)
    for kind in kinds:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(config.seed)
            model = (
                DeterministicStudent(architecture)
                if kind == "deterministic"
                else StochasticStudent(architecture, ratio_mean, ratio_scale)
            ).to(device)
        torch.manual_seed(config.seed + 419)  # Independent dropout stream for each student kind.
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.student_learning_rate, weight_decay=config.weight_decay
        )
        group_rng = np.random.default_rng(config.seed + 113)
        generator = torch.Generator(device=device).manual_seed(config.seed + 173)
        score_rng = torch.Generator(device=device).manual_seed(config.seed + 271)
        val_rng = torch.Generator(device=device).manual_seed(config.seed + 379)
        val_noise = torch.randn(config.stochastic_likelihood_samples, 63, generator=val_rng, device=device)
        history, best, best_state, chosen = [], float("inf"), None, None

        for step in range(config.student_steps):
            model.train()
            coefficient = score_coefficient(step, config.student_steps, config.teacher_score_weight)
            indices = sample_groups(group_rng, len(targets), config.student_batch_contexts)
            loss = accumulated_step(
                model,
                optimizer,
                [targets[i] for i in indices],
                partial(
                    student_losses,
                    model=model,
                    kind=kind,
                    teacher=teacher,
                    config=config,
                    weights=weights,
                    coefficient=coefficient,
                    generator=generator,
                    score_rng=score_rng,
                ),
                microbatch_contexts=config.microbatch_contexts,
            )
            if any(p.grad is not None for p in teacher.parameters()):
                raise RuntimeError("Frozen teacher unexpectedly accumulated parameter gradients.")
            if (step + 1) % config.student_checkpoint_every == 0 or step + 1 == config.student_steps:
                model.eval()
                with torch.no_grad():
                    losses = []
                    for c, m, w, _, _, _ in validation_targets:
                        if kind == "deterministic":
                            value = density_kl((w[:, None] * m).sum(0, keepdim=True), model(c)).mean()
                        else:
                            perturbed = model.ratios(m)[:, None] + config.target_smoothing * val_noise[None]
                            condition_batch = c.expand(len(m) * config.stochastic_likelihood_samples, -1)
                            value = (
                                model.nll(condition_batch, perturbed.reshape(-1, 63)).reshape(len(m), -1).mean(1) * w
                            ).sum()
                        losses.append(value)
                    value = float(torch.stack(losses).mean())
                row = {
                    "step": step + 1,
                    "objective": loss,
                    "coefficient": coefficient,
                    "validation_distillation": value,
                    "contexts": len(indices),
                }
                history.append(row)
                if checkpoint_callback is not None:
                    checkpoint_callback(kind, step + 1, model, dict(row))
                if coefficient > 0 and value < best:
                    best, best_state, chosen = value, copy.deepcopy(model.state_dict()), dict(row)
        if best_state is None:
            raise ValueError("Student validation did not produce a finite checkpoint after the score ramp.")
        model.load_state_dict(best_state)
        students[kind] = model.eval().requires_grad_(False).cpu()
        histories[kind] = history
        selections[kind] = chosen
    return (
        teacher.cpu(),
        students,
        {
            "training": asdict(config),
            "teacher": teacher_history,
            "students": histories,
            "density_holdout": sorted(family_holdout),
            "teacher_selection": selected_teacher,
            "student_selection": selections,
            "temperature_units": TEMPERATURE_UNITS,
            "auxiliary_normalization": AUXILIARY_NORMALIZATION,
        },
    )


def fit(
    rows: list[dict],
    contexts: dict,
    output,
    *,
    student_kind: str = "both",
    device: str = "cuda",
    purpose: str = "research",
    calibration_rows: list[dict] | None = None,
    checkpoint_callback=None,
    **fitting_settings,
) -> dict:
    if Path(output).exists():
        raise FileExistsError(f"Artifact destination already exists: {output}")
    bindings = [r["backbone_binding"] for r in rows if "backbone_binding" in r]
    if bindings and (len(bindings) != len(rows) or any(value != bindings[0] for value in bindings)):
        raise ValueError("Native backbone/context bindings differ between measurements.")
    evidence = prepare_evidence(rows, contexts, calibration_rows=calibration_rows, purpose=purpose)
    config = resolve_profile(evidence.task, **fitting_settings)
    if config.backbone is not None and config.backbone != evidence.backbone:
        raise ValueError("Fitting profile backbone differs from measurement evidence.")
    config = replace(config, backbone=evidence.backbone)
    evidence = replace(evidence, conditioning=replace(evidence.conditioning, context_mode=config.teacher_context_mode))
    started = time.perf_counter()
    teacher, students, history = fit_models(
        evidence, config, student_kind=student_kind, device=device, checkpoint_callback=checkpoint_callback
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
        "teacher_selection_criterion": "heldout_reference_utility_regret",
        "student_selection_criterion": "post_ramp_validation_distillation",
        "selected_temperature": history["teacher_selection"]["temperature"],
        "split_contexts": {
            s: sorted({r["context_id"] for r in evidence.cells if r["split"] == s}) for s in ("train", "validation")
        },
        "reference_densities": {
            f"{r['solver']}:{r['nfe']}:{r['schedule_key']}": r["density_mass"] for r in evidence.cells
        },
        "reference_grids": {f"{r['solver']}:{r['nfe']}:{r['schedule_key']}": r["time_grid"] for r in evidence.cells},
        "measurement_protocols": sorted({r["measurement_protocol"] for r in evidence.cells}),
        "teacher_score_is_surrogate": True,
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
    metadata["reward_estimators"] = [
        r["reward_estimator"] for r in (rows + (calibration_rows or [])) if "reward_estimator" in r
    ]
    from genode.gico.evidence import content_hash

    maps = {
        content_hash(r["molecule_feature_map"]): r["molecule_feature_map"]
        for r in rows + (calibration_rows or [])
        if "molecule_feature_map" in r
    }
    metadata["molecular_feature_maps"] = maps
    from genode.gico.policy import save_artifact

    save_artifact(
        output, teacher, students, replace(evidence.conditioning, context_mode=config.student_context_mode), metadata
    )
    return metadata
