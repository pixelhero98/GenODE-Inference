"""Shared checkpoint eligibility and calibrated surrogate aggregation."""

from collections import defaultdict

import numpy as np

STUDENT_SELECTION_PROTOCOL = "heldout_calibrated_teacher_utility_with_policy_kl"


def balanced_mean(values, groups, task):
    settings = defaultdict(lambda: defaultdict(list))
    if len(values) != len(groups) or not values or not np.isfinite(values).all():
        raise ValueError("Selection requires complete finite context/settings values.")
    for value, group in zip(values, groups, strict=True):
        row = group[0]
        unit = row["class_id"] if task == "imagenet64" else row["context_id"]
        settings[row["solver"], row["nfe"]][unit].append(float(value))
    return float(np.mean([np.mean([np.mean(v) for v in units.values()]) for units in settings.values()]))


def admissible_checkpoints(records, allowance):
    eligible = [row for row in records if row.get("coefficient", 0) > 0]
    if not eligible or any(
        not np.isfinite(row.get("validation_distillation", np.nan))
        or row["validation_distillation"] < -1e-12
        or not np.isfinite(row.get("predicted_utility", np.nan))
        for row in eligible
    ):
        raise ValueError("Student selection requires finite teacher scores and nonnegative KL.")
    if isinstance(allowance, bool) or not np.isfinite(allowance) or allowance < 0:
        raise ValueError("KL allowance must be finite and nonnegative.")
    minimum = min(max(0.0, row["validation_distillation"]) for row in eligible)
    threshold = minimum + allowance * minimum
    return [row for row in eligible if row["validation_distillation"] <= threshold]


def select_checkpoint(records, allowance):
    return min(admissible_checkpoints(records, allowance), key=lambda row: (-row["predicted_utility"], row["step"]))
