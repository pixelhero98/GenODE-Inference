from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from genode.experiment_layout import (
    REFERENCE_UNSEEN_TARGET_WEIGHT,
    STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT,
    STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_TARGET,
)
from genode.gipo.policy import (
    DEFAULT_STUDENT_TARGET_ELITE_BLEND_ALL_WEIGHT,
    DEFAULT_STUDENT_TARGET_ELITE_FRACTION,
    DEFAULT_STUDENT_TARGET_ELITE_K,
    DEFAULT_STUDENT_TARGET_ELITE_MIN_COUNT,
    DEFAULT_STUDENT_TEACHER_SCORE_WARMUP_FRACTION,
    REFERENCE_STUDENT_TEACHER_SCORE_WEIGHT,
)

ABLATION_PRESET_MAIN = "main"
ABLATION_PRESET_ALL = "main_and_appendix"
GIPO_POLICY_KEY = "gipo"


@dataclass(frozen=True)
class GIPOStudentPolicy:
    policy_key: str
    student_training_mode: str
    student_target_mixture_mode: str
    student_teacher_score_weight: float
    comparison_group: str
    student_target_elite_blend_all_weight: float = DEFAULT_STUDENT_TARGET_ELITE_BLEND_ALL_WEIGHT
    student_teacher_score_warmup_fraction: float = DEFAULT_STUDENT_TEACHER_SCORE_WARMUP_FRACTION
    student_teacher_score_include_unseen_targets: bool = False
    student_target_elite_fraction: float = DEFAULT_STUDENT_TARGET_ELITE_FRACTION
    student_target_elite_k: int = DEFAULT_STUDENT_TARGET_ELITE_K
    student_target_elite_min_count: int = DEFAULT_STUDENT_TARGET_ELITE_MIN_COUNT
    student_unseen_target_weight: float = REFERENCE_UNSEEN_TARGET_WEIGHT

    @property
    def uses_unseen_targets(self) -> bool:
        return self.student_training_mode == STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_TARGET

    def objective_settings(self) -> Dict[str, Any]:
        return {
            "student_target_mixture_mode": self.student_target_mixture_mode,
            "student_target_elite_fraction": float(self.student_target_elite_fraction),
            "student_target_elite_k": int(self.student_target_elite_k),
            "student_target_elite_min_count": int(self.student_target_elite_min_count),
            "student_target_elite_blend_all_weight": float(self.student_target_elite_blend_all_weight),
            "student_teacher_score_weight": float(self.student_teacher_score_weight),
            "student_teacher_score_warmup_fraction": float(self.student_teacher_score_warmup_fraction),
            "student_teacher_score_include_unseen_targets": bool(
                self.student_teacher_score_include_unseen_targets
            ),
        }

    def manifest_record(self) -> Dict[str, Any]:
        return {
            "student_policy_key": self.policy_key,
            "comparison_group": self.comparison_group,
            "student_training_mode": self.student_training_mode,
            "uses_unseen_targets": bool(self.uses_unseen_targets),
            "student_unseen_target_weight": float(self.student_unseen_target_weight),
            "student_objective_settings": self.objective_settings(),
        }


_GIPO_POLICY = GIPOStudentPolicy(
    GIPO_POLICY_KEY,
    STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT,
    "full",
    REFERENCE_STUDENT_TEACHER_SCORE_WEIGHT,
    "reference",
)

_ABLATION_POLICIES: Tuple[GIPOStudentPolicy, ...] = (
    GIPOStudentPolicy(
        "full_seen_only_score_000",
        STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT,
        "full",
        0.00,
        "main",
    ),
    GIPOStudentPolicy(
        "full_seen_only_score_005",
        STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT,
        "full",
        0.05,
        "main",
    ),
    GIPOStudentPolicy(
        "full_seen_plus_unseen_target_score_005",
        STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_TARGET,
        "full",
        0.05,
        "main",
    ),
    GIPOStudentPolicy(
        "elite_seen_only_score_005",
        STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT,
        "elite",
        0.05,
        "main",
    ),
    GIPOStudentPolicy(
        "blend_020_seen_only_score_005",
        STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT,
        "elite_blend",
        0.05,
        "main",
        0.20,
    ),
    GIPOStudentPolicy(
        "blend_020_seen_plus_unseen_target_score_005",
        STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_TARGET,
        "elite_blend",
        0.05,
        "main",
        0.20,
    ),
    GIPOStudentPolicy(
        "full_seen_only_score_010",
        STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT,
        "full",
        0.10,
        "appendix",
    ),
    GIPOStudentPolicy(
        "full_seen_plus_unseen_target_score_000",
        STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_TARGET,
        "full",
        0.00,
        "appendix",
    ),
    GIPOStudentPolicy(
        "full_seen_plus_unseen_target_score_001",
        STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_TARGET,
        "full",
        0.01,
        "appendix",
    ),
    GIPOStudentPolicy(
        "full_seen_plus_unseen_target_score_010",
        STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_TARGET,
        "full",
        0.10,
        "appendix",
    ),
    GIPOStudentPolicy(
        "elite_seen_plus_unseen_target_score_005",
        STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_TARGET,
        "elite",
        0.05,
        "appendix",
    ),
    GIPOStudentPolicy(
        "blend_010_seen_only_score_005",
        STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT,
        "elite_blend",
        0.05,
        "appendix",
        0.10,
    ),
    GIPOStudentPolicy(
        "blend_040_seen_only_score_005",
        STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT,
        "elite_blend",
        0.05,
        "appendix",
        0.40,
    ),
    GIPOStudentPolicy(
        "blend_010_seen_plus_unseen_target_score_005",
        STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_TARGET,
        "elite_blend",
        0.05,
        "appendix",
        0.10,
    ),
    GIPOStudentPolicy(
        "blend_040_seen_plus_unseen_target_score_005",
        STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_TARGET,
        "elite_blend",
        0.05,
        "appendix",
        0.40,
    ),
)

_PRESETS: Mapping[str, Tuple[GIPOStudentPolicy, ...]] = {
    ABLATION_PRESET_MAIN: tuple(policy for policy in _ABLATION_POLICIES if policy.comparison_group == "main"),
    ABLATION_PRESET_ALL: _ABLATION_POLICIES,
}


def ablation_preset_keys() -> Sequence[str]:
    return tuple(_PRESETS)


def ablation_student_policies(preset: str = ABLATION_PRESET_ALL) -> Tuple[GIPOStudentPolicy, ...]:
    try:
        return _PRESETS[str(preset)]
    except KeyError as exc:
        raise ValueError(f"Unknown GIPO ablation preset {preset!r}; expected one of {tuple(_PRESETS)}.") from exc


def gipo_policy() -> GIPOStudentPolicy:
    return _GIPO_POLICY


__all__ = [
    "ABLATION_PRESET_ALL",
    "ABLATION_PRESET_MAIN",
    "GIPOStudentPolicy",
    "GIPO_POLICY_KEY",
    "ablation_preset_keys",
    "ablation_student_policies",
    "gipo_policy",
]
