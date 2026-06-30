from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from genode.canonical_experiment_layout import (
    CANONICAL_PSEUDO_TARGET_WEIGHT,
    STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT,
    STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_PSEUDO,
)
from genode.gipo.policy import (
    DEFAULT_STUDENT_TARGET_ELITE_BLEND_ALL_WEIGHT,
    DEFAULT_STUDENT_TARGET_ELITE_FRACTION,
    DEFAULT_STUDENT_TARGET_ELITE_K,
    DEFAULT_STUDENT_TARGET_ELITE_MIN_COUNT,
    DEFAULT_STUDENT_TEACHER_SCORE_WARMUP_FRACTION,
)

GIPO_ABLATION_PRESET_PAPER_MAIN = "paper_main"
GIPO_ABLATION_PRESET_PAPER_MAIN_PLUS_APPENDIX = "paper_main_plus_appendix"
DEFAULT_GIPO_ABLATION_PRESET = GIPO_ABLATION_PRESET_PAPER_MAIN_PLUS_APPENDIX
GIPO_PAPER_STUDENT_ARM_ID = "S0_full_score001_seen_only"


@dataclass(frozen=True)
class GipoAblationArm:
    arm_id: str
    student_training_mode: str
    student_target_mixture_mode: str
    student_teacher_score_weight: float
    paper_group: str
    student_target_elite_blend_all_weight: float = DEFAULT_STUDENT_TARGET_ELITE_BLEND_ALL_WEIGHT
    student_teacher_score_warmup_fraction: float = DEFAULT_STUDENT_TEACHER_SCORE_WARMUP_FRACTION
    student_teacher_score_include_pseudo: bool = False
    student_target_elite_fraction: float = DEFAULT_STUDENT_TARGET_ELITE_FRACTION
    student_target_elite_k: int = DEFAULT_STUDENT_TARGET_ELITE_K
    student_target_elite_min_count: int = DEFAULT_STUDENT_TARGET_ELITE_MIN_COUNT
    student_pseudo_target_weight: float = CANONICAL_PSEUDO_TARGET_WEIGHT

    @property
    def uses_unseen_pseudo_targets(self) -> bool:
        return self.student_training_mode == STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_PSEUDO

    def objective_settings(self) -> Dict[str, Any]:
        return {
            "student_target_mixture_mode": self.student_target_mixture_mode,
            "student_target_elite_fraction": float(self.student_target_elite_fraction),
            "student_target_elite_k": int(self.student_target_elite_k),
            "student_target_elite_min_count": int(self.student_target_elite_min_count),
            "student_target_elite_blend_all_weight": float(self.student_target_elite_blend_all_weight),
            "student_teacher_score_weight": float(self.student_teacher_score_weight),
            "student_teacher_score_warmup_fraction": float(self.student_teacher_score_warmup_fraction),
            "student_teacher_score_include_pseudo": bool(self.student_teacher_score_include_pseudo),
        }

    def manifest_record(self) -> Dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "paper_group": self.paper_group,
            "student_training_mode": self.student_training_mode,
            "uses_unseen_pseudo_targets": bool(self.uses_unseen_pseudo_targets),
            "student_pseudo_target_weight": float(self.student_pseudo_target_weight),
            "student_objective_settings": self.objective_settings(),
        }


_ALL_ARMS: Tuple[GipoAblationArm, ...] = (
    GipoAblationArm("A0_full_score000_seen_only", STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT, "full", 0.00, "main"),
    GipoAblationArm("A1_full_score005_seen_only", STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT, "full", 0.05, "main"),
    GipoAblationArm("A2_full_score005_seen_plus_pseudo", STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_PSEUDO, "full", 0.05, "main"),
    GipoAblationArm("A3_elite_score005_seen_only", STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT, "elite", 0.05, "main"),
    GipoAblationArm("A4_blend020_score005_seen_only", STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT, "elite_blend", 0.05, "main", 0.20),
    GipoAblationArm("A5_blend020_score005_seen_plus_pseudo", STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_PSEUDO, "elite_blend", 0.05, "main", 0.20),
    GipoAblationArm("S0_full_score001_seen_only", STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT, "full", 0.01, "appendix"),
    GipoAblationArm("S1_full_score010_seen_only", STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT, "full", 0.10, "appendix"),
    GipoAblationArm("S2_full_score000_seen_plus_pseudo", STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_PSEUDO, "full", 0.00, "appendix"),
    GipoAblationArm("S3_full_score001_seen_plus_pseudo", STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_PSEUDO, "full", 0.01, "appendix"),
    GipoAblationArm("S4_full_score010_seen_plus_pseudo", STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_PSEUDO, "full", 0.10, "appendix"),
    GipoAblationArm("T0_elite_score005_seen_plus_pseudo", STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_PSEUDO, "elite", 0.05, "appendix"),
    GipoAblationArm("B0_blend010_score005_seen_only", STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT, "elite_blend", 0.05, "appendix", 0.10),
    GipoAblationArm("B1_blend040_score005_seen_only", STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT, "elite_blend", 0.05, "appendix", 0.40),
    GipoAblationArm("B2_blend010_score005_seen_plus_pseudo", STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_PSEUDO, "elite_blend", 0.05, "appendix", 0.10),
    GipoAblationArm("B3_blend040_score005_seen_plus_pseudo", STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_PSEUDO, "elite_blend", 0.05, "appendix", 0.40),
)

_PRESETS: Mapping[str, Tuple[GipoAblationArm, ...]] = {
    GIPO_ABLATION_PRESET_PAPER_MAIN: tuple(arm for arm in _ALL_ARMS if arm.paper_group == "main"),
    GIPO_ABLATION_PRESET_PAPER_MAIN_PLUS_APPENDIX: _ALL_ARMS,
}
_ARMS_BY_ID: Mapping[str, GipoAblationArm] = {arm.arm_id: arm for arm in _ALL_ARMS}


def gipo_ablation_preset_choices() -> Sequence[str]:
    return tuple(_PRESETS)


def gipo_ablation_arms(preset: str = DEFAULT_GIPO_ABLATION_PRESET) -> Tuple[GipoAblationArm, ...]:
    try:
        return _PRESETS[str(preset)]
    except KeyError as exc:
        raise ValueError(f"Unknown GIPO ablation preset {preset!r}; expected one of {tuple(_PRESETS)}.") from exc


def gipo_paper_student_arm() -> GipoAblationArm:
    return _ARMS_BY_ID[GIPO_PAPER_STUDENT_ARM_ID]
