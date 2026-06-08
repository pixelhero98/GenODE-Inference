from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


REQUIRED_TEACHER_ARCH = "density_form_transformer_v1"
REQUIRED_STUDENT_ARCH = "density_query_transformer_v1"
REQUIRED_ENCODER = "continuous_v3"
REQUIRED_CONDITIONING = "additive_mlp_v1"
REQUIRED_DENSITY_ATTENTION = "bin_self_attention_rope_v1"
REQUIRED_HEADS = 4


def _close(observed: object, expected: float) -> bool:
    try:
        value = float(observed)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and abs(value - float(expected)) <= 1e-12


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a reusable GIPO probe training summary.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--crps-weight", type=float, required=True)
    parser.add_argument("--mase-weight", type=float, required=True)
    parser.add_argument("--pseudo-weight", type=float, required=True)
    parser.add_argument("--student-target-mode", default="")
    parser.add_argument("--teacher-hard-margin", type=float, default=None)
    args = parser.parse_args()

    path = Path(args.summary)
    summary = json.loads(path.read_text(encoding="utf-8"))
    teacher_cfg = dict(summary.get("teacher_model_config") or {})
    student_cfg = dict(summary.get("student_model_config") or {})
    student_training = dict(summary.get("student_training") or {})
    pseudo = dict(summary.get("pseudo_distillation") or {})
    weights = dict(summary.get("teacher_utility_weights") or {})

    issues: list[str] = []
    if summary.get("teacher_architecture") != REQUIRED_TEACHER_ARCH:
        issues.append("teacher_architecture")
    if summary.get("student_architecture") != REQUIRED_STUDENT_ARCH:
        issues.append("student_architecture")
    if summary.get("setting_encoder_mode") != REQUIRED_ENCODER:
        issues.append("setting_encoder_mode")
    for prefix, cfg in (("teacher", teacher_cfg), ("student", student_cfg)):
        if cfg.get("attention_heads") != REQUIRED_HEADS:
            issues.append(f"{prefix}_attention_heads")
        if cfg.get("conditioning_style") != REQUIRED_CONDITIONING:
            issues.append(f"{prefix}_conditioning_style")
        if cfg.get("density_token_attention") != REQUIRED_DENSITY_ATTENTION:
            issues.append(f"{prefix}_density_token_attention")
    if summary.get("locked_test_used_for_selection") is not False:
        issues.append("locked_test_used_for_selection")
    if not _close(weights.get("crps"), float(args.crps_weight)) or not _close(weights.get("mase"), float(args.mase_weight)):
        issues.append("teacher_utility_weights")
    if not _close(student_training.get("pseudo_target_weight"), float(args.pseudo_weight)):
        issues.append("student_training_pseudo_target_weight")
    if not _close(pseudo.get("pseudo_target_weight"), float(args.pseudo_weight)):
        issues.append("pseudo_distillation_pseudo_target_weight")
    if float(args.pseudo_weight) > 0.0 and not bool(student_training.get("pseudo_distillation_used", False)):
        issues.append("pseudo_distillation_used")
    if float(args.pseudo_weight) > 0.0:
        pseudo_phases = set(pseudo.get("pseudo_split_phases") or [])
        if pseudo_phases != {"train_tuning"}:
            issues.append(f"pseudo_split_phases:{sorted(pseudo_phases)}")
    if args.student_target_mode and summary.get("student_target_mode") != args.student_target_mode:
        issues.append("student_target_mode")
    if args.teacher_hard_margin is not None and not _close(summary.get("teacher_hard_margin"), float(args.teacher_hard_margin)):
        issues.append("teacher_hard_margin")
    if issues:
        raise SystemExit(f"{args.run_id} existing summary is not reusable: {issues}")


if __name__ == "__main__":
    main()
