"""Resolved task-specific fitting settings for the shared GICO models."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from genode.gico.clocks import REFERENCE_KEYS
from genode.gico.rewards import TASK_METRICS

SCORE_WEIGHTS = (0.01, 0.05, 0.1)
TEMPERATURE_UNITS = "paired_utility_before_scalar_reward_normalization"
AUXILIARY_NORMALIZATION = "frozen_context_solver_nfe_reference_mean_std"


@dataclass(frozen=True)
class TrainingConfig:
    context_mode: str = "native"
    width: int = 128
    teacher_density_normalization: str = "none"
    teacher_steps: int = 500
    student_steps: int = 500
    teacher_batch_groups: int = 64
    student_batch_contexts: int = 512
    microbatch_contexts: int = 8
    teacher_learning_rate: float = 1e-3
    student_learning_rate: float = 1e-3
    teacher_checkpoint_every: int = 100
    student_checkpoint_every: int = 100
    weight_decay: float = 1e-4
    dropout: float = 0.05
    teacher_score_weight: float = 0.01
    temperatures: tuple[float, ...] = (0.05,)
    preferred_temperature: float = 0.05
    density_family_holdout: tuple[str, ...] = ("late_p_3", "late_p_3_reversed")
    stochastic_likelihood_samples: int = 32
    stochastic_score_samples: int = 4
    target_smoothing: float = 0.1
    seed: int = 0

    def __post_init__(self):
        if self.context_mode not in ("native", "global"):
            raise ValueError("context_mode must be native or global.")
        if self.width not in (64, 128):
            raise ValueError("Transformer width must be 64 or 128.")
        if self.teacher_density_normalization not in ("none", "training_reference"):
            raise ValueError("Unknown teacher density normalization protocol.")
        integers = (
            "teacher_steps",
            "student_steps",
            "teacher_batch_groups",
            "student_batch_contexts",
            "microbatch_contexts",
            "teacher_checkpoint_every",
            "student_checkpoint_every",
            "stochastic_likelihood_samples",
            "stochastic_score_samples",
        )
        if any(type(getattr(self, key)) is not int or getattr(self, key) < 1 for key in integers):
            raise ValueError("Step, batch, sampling and checkpoint counts must be positive integers.")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Seed must be a nonnegative integer.")
        if self.teacher_score_weight not in SCORE_WEIGHTS:
            raise ValueError(f"teacher_score_weight must be one of {SCORE_WEIGHTS}.")
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
        if set(self.density_family_holdout) - set(REFERENCE_KEYS):
            raise ValueError("Unknown density-family holdout names.")
        if len(set(self.density_family_holdout)) != len(self.density_family_holdout):
            raise ValueError("Density-family holdout names must be distinct.")


def resolve_profile(task: str, **overrides) -> TrainingConfig:
    if task not in TASK_METRICS:
        raise ValueError(f"Unknown task profile: {task}")
    values = {}
    if task in ("sana", "sd15"):
        values.update(teacher_score_weight=0.05, density_family_holdout=())
    elif task in ("cifar10", "imagenet64"):
        values.update(
            teacher_steps=2000, student_steps=2000, dropout=0.0, temperatures=(1.0,), preferred_temperature=1.0
        )
    unknown = set(overrides) - {field.name for field in fields(TrainingConfig)}
    if unknown:
        raise ValueError(f"Unknown fitting settings: {sorted(unknown)}")
    values.update(overrides)
    for key in ("temperatures", "density_family_holdout"):
        if key in values:
            values[key] = tuple(values[key])
    return TrainingConfig(**values)
