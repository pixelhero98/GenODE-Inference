"""Shared teacher and two-student fitting, with differentiable score regularization."""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from genode.gico.evidence import Evidence, prepare_evidence
from genode.gico.networks import DensityTeacher, DeterministicStudent, ModelConfig, StochasticStudent, density_kl

SCORE_WEIGHTS = (0.01, 0.05, 0.1)
STUDENT_KINDS = ("deterministic", "stochastic")


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 2000
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    teacher_score_weight: float = 0.01
    seed: int = 0
    target_smoothing: float = 0.1

    def __post_init__(self):
        if self.teacher_score_weight not in SCORE_WEIGHTS:
            raise ValueError(f"teacher_score_weight must be one of {SCORE_WEIGHTS}.")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (self.steps, self.batch_size)):
            raise ValueError("steps and batch_size must be positive integers.")
        if self.target_smoothing != 0.1:
            raise ValueError("Unified stochastic target smoothing is fixed at 0.1.")
        if (
            not np.isfinite([self.learning_rate, self.weight_decay]).all()
            or self.learning_rate <= 0
            or self.weight_decay < 0
        ):
            raise ValueError("Invalid optimizer parameters.")


def score_coefficient(step: int, steps: int, weight: float) -> float:
    if weight not in SCORE_WEIGHTS:
        raise ValueError("Unsupported teacher-score weight.")
    return weight * max(0.0, min(1.0, ((step + 1) / steps - 0.6) / 0.4))


def teacher_loss(predicted: Tensor, target: Tensor) -> Tensor:
    regression = F.huber_loss(predicted, target, reduction="mean")
    score, truth = predicted.mean(-1), target.mean(-1)
    left, right = torch.triu_indices(len(score), len(score), offset=1, device=score.device)
    differences = truth[left] - truth[right]
    non_ties = differences != 0
    ranking = (
        F.softplus(-(score[left] - score[right])[non_ties] * differences[non_ties].sign() / 0.5).mean()
        if bool(non_ties.any())
        else regression * 0
    )
    return ranking + 0.25 * regression


def teacher_score(teacher: DensityTeacher, condition: Tensor, mass: Tensor) -> Tensor:
    return teacher(condition, mass).mean(-1).clamp(-5, 5).mean()


def teacher_weights(teacher: DensityTeacher, condition: Tensor, mass: Tensor) -> Tensor:
    """Temperature-one weights over unique references, using bounded scores."""
    return teacher(condition, mass).mean(-1).clamp(-5, 5).softmax(0)


def _tensors(evidence: Evidence, group: list[dict], device: str) -> tuple[Tensor, Tensor, Tensor]:
    conditions = np.array(
        [evidence.conditioning.transform(evidence.contexts[r["context_id"]], r["solver"], r["nfe"]) for r in group]
    )
    return (
        torch.tensor(conditions, dtype=torch.float32, device=device),
        torch.tensor([r["density_mass"] for r in group], dtype=torch.float64, device=device),
        torch.tensor([r["reward_vector"] for r in group], dtype=torch.float32, device=device),
    )


def _finite_step(model: nn.Module, optimizer, loss: Tensor) -> None:
    if not bool(torch.isfinite(loss)):
        raise ValueError("Nonfinite training objective.")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()


def _optimizer(model, config):
    return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)


def fit_models(evidence: Evidence, config: TrainingConfig, *, student_kind: str = "both", device: str = "cuda"):
    if student_kind not in (*STUDENT_KINDS, "both"):
        raise ValueError("student_kind must be deterministic, stochastic, or both.")
    train_groups, val_groups = evidence.groups("train"), evidence.groups("validation")
    # Leave an entire density family out of teacher fitting; its observations
    # remain available for validation, never for reference-ratio normalization.
    family_holdout = {"late_p_3", "late_p_3_reversed"}
    teacher_groups = [[r for r in g if not set(r["aliases"]) <= family_holdout] for g in train_groups]
    density_holdout = [[r for r in g if set(r["aliases"]) <= family_holdout] for g in train_groups]
    density_holdout = [g for g in density_holdout if g]
    if evidence.purpose == "research" and not density_holdout:
        raise ValueError("Research teacher fitting requires its density-family holdout.")
    train = [_tensors(evidence, g, device) for g in teacher_groups if g]
    validation = [_tensors(evidence, g, device) for g in val_groups]
    density_validation = [_tensors(evidence, g, device) for g in density_holdout]
    architecture = ModelConfig(evidence.conditioning.width, len(next(iter(evidence.calibrations.values())).metric_keys))
    # fork_rng isolates initialization from callers' generation RNG state.
    devices = [torch.device(device).index or 0] if str(device).startswith("cuda") else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed)
        teacher = DensityTeacher(architecture).to(device)
    rng = np.random.default_rng(config.seed)
    optimizer = _optimizer(teacher, config)
    best, best_state, teacher_history = float("inf"), None, []
    interval = max(1, config.steps // 10)
    for step in range(config.steps):
        teacher.train()
        condition, mass, target = train[int(rng.integers(len(train)))]
        _finite_step(teacher, optimizer, teacher_loss(teacher(condition, mass), target))
        if (step + 1) % interval == 0 or step + 1 == config.steps:
            teacher.eval()
            with torch.no_grad():
                context_loss = torch.stack([F.huber_loss(teacher(c, m), y) for c, m, y in validation]).mean()
                family_loss = (
                    torch.stack([F.huber_loss(teacher(c, m), y) for c, m, y in density_validation]).mean()
                    if density_validation
                    else context_loss
                )
                value = float((context_loss + family_loss) / 2)
            teacher_history.append(
                {"step": step + 1, "context_huber": float(context_loss), "density_huber": float(family_loss)}
            )
            if value < best:
                best, best_state = value, copy.deepcopy(teacher.state_dict())
    if best_state is None:
        raise ValueError("Teacher validation did not produce a finite checkpoint.")
    teacher.load_state_dict(best_state)
    teacher.eval().requires_grad_(False)
    teacher.zero_grad(set_to_none=True)
    training_groups = [_tensors(evidence, g, device) for g in train_groups]
    with torch.no_grad():
        targets = [(c[:1], m, teacher_weights(teacher, c, m)) for c, m, _ in training_groups]
        validation_targets = [(c[:1], m, teacher_weights(teacher, c, m)) for c, m, _ in validation]
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
    students, histories = {}, {}
    kinds = STUDENT_KINDS if student_kind == "both" else (student_kind,)
    for kind in kinds:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(config.seed)
            model = (
                DeterministicStudent(architecture)
                if kind == "deterministic"
                else StochasticStudent(architecture, ratio_mean, ratio_scale)
            ).to(device)
        optimizer = _optimizer(model, config)
        generator = torch.Generator(device=device).manual_seed(config.seed + 173)
        score_rng = torch.Generator(device=device).manual_seed(config.seed + 271)
        val_rng = torch.Generator(device=device).manual_seed(config.seed + 379)
        # Fixed validation noise gives comparable checkpoints without touching
        # generation or training streams.
        val_noise = torch.randn(config.batch_size, 63, generator=val_rng, device=device)
        history, best, best_state = [], float("inf"), None
        for step in range(config.steps):
            model.train()
            condition, mass, weights = targets[int(rng.integers(len(targets)))]
            coefficient = score_coefficient(step, config.steps, config.teacher_score_weight)
            if kind == "deterministic":
                target = (weights[:, None] * mass).sum(0, keepdim=True)
                predicted = model(condition)
                distillation = density_kl(target, predicted).mean()
                score = teacher_score(teacher, condition, predicted)
            else:
                choice = torch.multinomial(weights, config.batch_size, replacement=True, generator=generator)
                repeated_condition = condition.expand(config.batch_size, -1)
                target = model.ratios(mass[choice]) + config.target_smoothing * torch.randn(
                    config.batch_size, 63, generator=generator, device=device
                )
                distillation = model.nll(repeated_condition, target).mean()
                # Bound score Monte Carlo memory independently of parallel NLL.
                score_condition = condition.expand(min(4, config.batch_size), -1)
                score = (
                    teacher_score(teacher, score_condition, model.sample(score_condition, generator=score_rng))
                    if coefficient
                    else distillation * 0
                )
            loss = distillation - coefficient * score
            _finite_step(model, optimizer, loss)
            if any(p.grad is not None for p in teacher.parameters()):
                raise RuntimeError("Frozen teacher unexpectedly accumulated parameter gradients.")
            if (step + 1) % interval == 0 or step + 1 == config.steps:
                model.eval()
                with torch.no_grad():
                    losses = []
                    for c, m, w in validation_targets:
                        if kind == "deterministic":
                            value = density_kl((w[:, None] * m).sum(0, keepdim=True), model(c)).mean()
                        else:
                            # Weighted expectation over all reference paths,
                            # with fixed perturbations for checkpoint selection.
                            perturbed = model.ratios(m)[:, None] + config.target_smoothing * val_noise[None]
                            condition_batch = c.expand(len(m) * config.batch_size, -1)
                            value = (
                                model.nll(condition_batch, perturbed.reshape(-1, 63)).reshape(len(m), -1).mean(1) * w
                            ).sum()
                        losses.append(value)
                    value = float(torch.stack(losses).mean())
                history.append(
                    {
                        "step": step + 1,
                        "distillation": float(distillation.detach()),
                        "teacher_score": float(score.detach()),
                        "coefficient": coefficient,
                        "validation_distillation": value,
                    }
                )
                # Select only after score optimization has begun.
                if coefficient > 0 and value < best:
                    best, best_state = value, copy.deepcopy(model.state_dict())
        if best_state is None:
            raise ValueError("Student validation did not produce a finite checkpoint after the score ramp.")
        model.load_state_dict(best_state)
        students[kind] = model.eval().requires_grad_(False).cpu()
        histories[kind] = history
    return (
        teacher.cpu(),
        students,
        {
            "training": asdict(config),
            "teacher": teacher_history,
            "students": histories,
            "density_holdout": sorted(family_holdout),
        },
    )


def fit(
    rows: list[dict],
    contexts: dict,
    output,
    *,
    student_kind: str = "both",
    teacher_score_weight: float = 0.01,
    steps: int = 2000,
    seed: int = 0,
    device: str = "cuda",
    purpose: str = "research",
    calibration_rows: list[dict] | None = None,
    batch_size: int = 32,
) -> dict:
    if Path(output).exists():
        raise FileExistsError(f"Artifact destination already exists: {output}")
    bindings = [r["backbone_binding"] for r in rows if "backbone_binding" in r]
    if bindings and (len(bindings) != len(rows) or any(value != bindings[0] for value in bindings)):
        raise ValueError("Native backbone/context bindings differ between measurements.")
    evidence = prepare_evidence(rows, contexts, calibration_rows=calibration_rows, purpose=purpose)
    config = TrainingConfig(steps=steps, seed=seed, teacher_score_weight=teacher_score_weight, batch_size=batch_size)
    started = time.perf_counter()
    teacher, students, history = fit_models(evidence, config, student_kind=student_kind, device=device)
    fitting_seconds = time.perf_counter() - started
    metadata = {
        "task": evidence.task,
        "backbone": evidence.backbone,
        "purpose": purpose,
        "solvers": list(evidence.calibrations),
        "reward_calibrations": {k: v.to_payload() for k, v in evidence.calibrations.items()},
        "evidence_sha256": evidence.evidence_sha256,
        "history": history,
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

    save_artifact(output, teacher, students, evidence.conditioning, metadata)
    return metadata
