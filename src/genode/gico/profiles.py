"""Resolved task-specific fitting settings for the shared GICO models."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from genode.gico.rewards import TASK_METRICS

SCORE_WEIGHTS = (0.01, 0.05, 0.1)
SCORE_SCHEDULES = ("linear_60_40", "ramp_plateau_60_20_20", "constant_60_40")
TEMPERATURE_UNITS = "paired_utility_before_scalar_reward_normalization"
AUXILIARY_NORMALIZATION = "frozen_context_solver_nfe_reference_mean_std"


@dataclass(frozen=True)
class TrainingConfig:
    teacher_context_mode: str = "native"
    student_context_mode: str = "native"
    backbone: str | None = None
    teacher_steps: int = 2000
    student_steps: int = 2000
    teacher_batch_groups: int = 64
    student_batch_contexts: int = 512
    microbatch_contexts: int = 8
    teacher_learning_rate: float = 1e-3
    student_learning_rate: float = 1e-3
    teacher_checkpoint_every: int = 20
    student_checkpoint_every: int = 100
    deterministic_checkpoint_every: int = 10
    deterministic_kl_allowance: float = 0.15
    stochastic_kl_allowance: float = 0.20
    weight_decay: float = 1e-4
    dropout: float = 0.01
    teacher_score_weight: float = 0.01
    score_schedule: str = "linear_60_40"
    selection_clock_replicates: int = 4
    temperatures: tuple[float, ...] = (0.05, 0.1, 0.5)
    preferred_temperature: float = 0.05
    stochastic_likelihood_samples: int = 32
    stochastic_score_samples: int = 4
    target_smoothing: float = 0.1
    seed: int = 0

    def __post_init__(self):
        if any(mode not in ("native", "global") for mode in (self.teacher_context_mode, self.student_context_mode)):
            raise ValueError("Teacher/student context modes must be native or global.")
        if self.backbone is not None and (not isinstance(self.backbone, str) or not self.backbone.strip()):
            raise ValueError("Profile backbone must be a nonempty identity.")
        integers = (
            "teacher_steps",
            "student_steps",
            "teacher_batch_groups",
            "student_batch_contexts",
            "microbatch_contexts",
            "teacher_checkpoint_every",
            "student_checkpoint_every",
            "deterministic_checkpoint_every",
            "stochastic_likelihood_samples",
            "stochastic_score_samples",
            "selection_clock_replicates",
        )
        if any(type(getattr(self, key)) is not int or getattr(self, key) < 1 for key in integers):
            raise ValueError("Step, batch, sampling and checkpoint counts must be positive integers.")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Seed must be a nonnegative integer.")
        for allowance in (self.deterministic_kl_allowance, self.stochastic_kl_allowance):
            if isinstance(allowance, bool) or not np.isfinite(allowance) or allowance < 0:
                raise ValueError("Student KL allowances must be finite and nonnegative.")
        if self.teacher_score_weight not in SCORE_WEIGHTS:
            raise ValueError(f"teacher_score_weight must be one of {SCORE_WEIGHTS}.")
        if self.score_schedule not in SCORE_SCHEDULES:
            raise ValueError(f"score_schedule must be one of {SCORE_SCHEDULES}.")
        if self.target_smoothing != 0.1:
            raise ValueError("Stochastic target smoothing is fixed at 0.1.")
        if not np.isfinite(
            [
                self.teacher_learning_rate,
                self.student_learning_rate,
                self.weight_decay,
                self.dropout,
                self.preferred_temperature,
                *self.temperatures,
            ]
        ).all():
            raise ValueError("Fitting settings must be finite.")
        if min(self.teacher_learning_rate, self.student_learning_rate) <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer parameters.")
        if not 0 <= self.dropout <= 0.1:
            raise ValueError("Dropout must be between zero and 0.1.")
        if (
            not self.temperatures
            or min(self.temperatures) <= 0
            or len(set(self.temperatures)) != len(self.temperatures)
        ):
            raise ValueError("Temperatures must be distinct positive values.")
        if self.preferred_temperature not in self.temperatures:
            raise ValueError("Preferred temperature must be included in the candidate temperatures.")


def resolve_profile(task: str, **overrides) -> TrainingConfig:
    if task not in TASK_METRICS:
        raise ValueError(f"Unknown task profile: {task}")
    values = {}
    if task in ("sana", "sd15"):
        values.update(teacher_score_weight=0.05)
    elif task in ("cifar10", "imagenet64"):
        values.update(dropout=0.0)
    unknown = set(overrides) - {field.name for field in fields(TrainingConfig)}
    if unknown:
        raise ValueError(f"Unknown fitting settings: {sorted(unknown)}")
    values.update(overrides)
    if "temperatures" in values:
        values["temperatures"] = tuple(values["temperatures"])
    return TrainingConfig(**values)
