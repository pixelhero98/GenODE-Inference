from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from genode.canonical_experiment_layout import (
    CANONICAL_UNSEEN_TARGET_WEIGHT,
    STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT,
    STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_TARGETS,
)
from genode.gico.policy import (
    DEFAULT_STUDENT_TARGET_ELITE_BLEND_ALL_WEIGHT,
    DEFAULT_STUDENT_TARGET_ELITE_FRACTION,
    DEFAULT_STUDENT_TARGET_ELITE_K,
    DEFAULT_STUDENT_TARGET_ELITE_MIN_COUNT,
    DEFAULT_STUDENT_TEACHER_SCORE_WARMUP_FRACTION,
)

GICO_ABLATION_PRESET_PAPER_MAIN = "paper_main"
GICO_ABLATION_PRESET_PAPER_MAIN_PLUS_APPENDIX = "paper_main_plus_appendix"
DEFAULT_GICO_ABLATION_PRESET = GICO_ABLATION_PRESET_PAPER_MAIN_PLUS_APPENDIX


@dataclass(frozen=True)
class GicoAblationArm:
    arm_id: str
    student_training_mode: str
    student_target_mixture_mode: str
    student_teacher_score_weight: float
    paper_group: str
    student_target_elite_blend_all_weight: float = DEFAULT_STUDENT_TARGET_ELITE_BLEND_ALL_WEIGHT
    student_teacher_score_warmup_fraction: float = DEFAULT_STUDENT_TEACHER_SCORE_WARMUP_FRACTION
    student_teacher_score_include_unseen_targets: bool = False
    student_target_elite_fraction: float = DEFAULT_STUDENT_TARGET_ELITE_FRACTION
    student_target_elite_k: int = DEFAULT_STUDENT_TARGET_ELITE_K
    student_target_elite_min_count: int = DEFAULT_STUDENT_TARGET_ELITE_MIN_COUNT
    student_unseen_target_weight: float = CANONICAL_UNSEEN_TARGET_WEIGHT

    @property
    def uses_unseen_targets(self) -> bool:
        return self.student_training_mode == STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_TARGETS

    def objective_settings(self) -> Dict[str, Any]:
        return {
            "student_target_mixture_mode": self.student_target_mixture_mode,
            "student_target_elite_fraction": float(self.student_target_elite_fraction),
            "student_target_elite_k": int(self.student_target_elite_k),
            "student_target_elite_min_count": int(self.student_target_elite_min_count),
            "student_target_elite_blend_all_weight": float(self.student_target_elite_blend_all_weight),
            "student_teacher_score_weight": float(self.student_teacher_score_weight),
            "student_teacher_score_warmup_fraction": float(self.student_teacher_score_warmup_fraction),
            "student_teacher_score_include_unseen_targets": bool(self.student_teacher_score_include_unseen_targets),
        }

    def manifest_record(self) -> Dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "paper_group": self.paper_group,
            "student_training_mode": self.student_training_mode,
            "uses_unseen_targets": bool(self.uses_unseen_targets),
            "student_unseen_target_weight": float(self.student_unseen_target_weight),
            "student_objective_settings": self.objective_settings(),
        }


_ALL_ARMS: Tuple[GicoAblationArm, ...] = (
    GicoAblationArm("A0_full_score000_seen_only", STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT, "full", 0.00, "main"),
    GicoAblationArm("A1_full_score005_seen_only", STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT, "full", 0.05, "main"),
    GicoAblationArm("A2_full_score005_seen_plus_unseen", STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_TARGETS, "full", 0.05, "main"),
    GicoAblationArm("A3_elite_score005_seen_only", STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT, "elite", 0.05, "main"),
    GicoAblationArm("A4_blend020_score005_seen_only", STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT, "elite_blend", 0.05, "main", 0.20),
    GicoAblationArm("A5_blend020_score005_seen_plus_unseen", STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_TARGETS, "elite_blend", 0.05, "main", 0.20),
    GicoAblationArm("S0_full_score001_seen_only", STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT, "full", 0.01, "appendix"),
    GicoAblationArm("S1_full_score010_seen_only", STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT, "full", 0.10, "appendix"),
    GicoAblationArm("S2_full_score000_seen_plus_unseen", STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_TARGETS, "full", 0.00, "appendix"),
    GicoAblationArm("S3_full_score001_seen_plus_unseen", STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_TARGETS, "full", 0.01, "appendix"),
    GicoAblationArm("S4_full_score010_seen_plus_unseen", STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_TARGETS, "full", 0.10, "appendix"),
    GicoAblationArm("T0_elite_score005_seen_plus_unseen", STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_TARGETS, "elite", 0.05, "appendix"),
    GicoAblationArm("B0_blend010_score005_seen_only", STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT, "elite_blend", 0.05, "appendix", 0.10),
    GicoAblationArm("B1_blend040_score005_seen_only", STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT, "elite_blend", 0.05, "appendix", 0.40),
    GicoAblationArm("B2_blend010_score005_seen_plus_unseen", STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_TARGETS, "elite_blend", 0.05, "appendix", 0.10),
    GicoAblationArm("B3_blend040_score005_seen_plus_unseen", STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_TARGETS, "elite_blend", 0.05, "appendix", 0.40),
)

_PRESETS: Mapping[str, Tuple[GicoAblationArm, ...]] = {
    GICO_ABLATION_PRESET_PAPER_MAIN: tuple(arm for arm in _ALL_ARMS if arm.paper_group == "main"),
    GICO_ABLATION_PRESET_PAPER_MAIN_PLUS_APPENDIX: _ALL_ARMS,
}


def gico_ablation_preset_choices() -> Sequence[str]:
    return tuple(_PRESETS)


def gico_ablation_arms(preset: str = DEFAULT_GICO_ABLATION_PRESET) -> Tuple[GicoAblationArm, ...]:
    try:
        return _PRESETS[str(preset)]
    except KeyError as exc:
        raise ValueError(f"Unknown GICO ablation preset {preset!r}; expected one of {tuple(_PRESETS)}.") from exc
