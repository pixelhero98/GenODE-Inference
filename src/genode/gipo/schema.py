from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np


RETIRED_EVALUATION_KEYS = frozenset(
    {
        "crps",
        "dataset",
        "dataset_key",
        "experimental_fixed_schedule_keys",
        "best_fixed_crps_schedule",
        "best_fixed_mase_schedule",
        "final_teacher_retrain",
        "gipo_steps",
        "mase",
        "mse",
        "baseline_schedule_keys",
        "fixed_reference_schedule_keys",
        "fixed_schedule_keys",
        "fixed_schedule_count",
        "schedule_key",
        "schedule_keys",
        "schedule_key_is_section_15_baseline",
        "scheduled_evaluation_keys",
        "selected_gipo_step_budget",
        "selected_schedule_key",
        "selected_source_schedule_key",
        "ser_ptg_schedule_key",
        "ser_reference_schedule_keys",
        "setting_encoder_mode",
        "setting_feature_mode",
        "student_gipo_steps",
        "student_schedule_key",
        "student_schedule_key_is_baseline",
        "student_schedule_keys",
        "summary_schedule_keys",
        "uniform_schedule_key",
    }
)
SPLIT_PHASE_FIELDS = ("source_split_phase", "split_phase", "split")


def validate_declared_split_phase(
    row: Mapping[str, Any],
    *,
    source: str,
) -> str:
    """Return one explicit split declaration without allowing aliases to disagree."""

    declared = {
        field: str(row.get(field, "") or "").strip()
        for field in SPLIT_PHASE_FIELDS
        if str(row.get(field, "") or "").strip()
    }
    if not declared:
        raise ValueError(f"{source} requires an explicit split phase.")
    phases = set(declared.values())
    if len(phases) != 1:
        raise ValueError(f"{source} contains conflicting split phase declarations: {declared}.")
    return next(iter(phases))


def reject_retired_evaluation_keys(value: Any, *, source: str) -> None:
    """Reject retired schema keys recursively with their exact payload paths."""

    found: list[str] = []

    def visit(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for raw_key, child in item.items():
                key = str(raw_key)
                child_path = f"{path}.{key}" if path else key
                if key in RETIRED_EVALUATION_KEYS:
                    found.append(child_path)
                visit(child, child_path)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")

    visit(value, "")
    if found:
        raise ValueError(
            f"{source} uses retired evaluation keys at {sorted(found)}; "
            "use scenario_key, scheduler_key, gipo_step_budget, checkpoint_step, "
            "teacher_final_retrain, method_key, mode, forecast_crps, and forecast_mase."
        )


def consistent_metadata_value(
    sources: Sequence[Mapping[str, Any]],
    key: str,
    *,
    source: str,
    required: bool = False,
) -> Any:
    """Return one metadata value and reject conflicting declarations."""

    declared: list[tuple[str, Any]] = []
    for index, mapping in enumerate(sources):
        if key not in mapping:
            continue
        value = mapping.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        normalized = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        declared.append((normalized, value))
    normalized_values = {normalized for normalized, _ in declared}
    if len(normalized_values) > 1:
        raise ValueError(f"{source} contains conflicting {key} values.")
    if not declared:
        if required:
            raise ValueError(f"{source} requires {key}.")
        return None
    return declared[0][1]


def cap_context_indices(
    indices: Sequence[int],
    *,
    cap: int,
    seed: int,
    salt: str,
    selection_protocol: str,
    uncapped_candidate_examples: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a deterministic cap and return standard selection metadata."""

    candidate = [int(index) for index in indices]
    selected_cap = int(cap)
    if selected_cap <= 0:
        raise ValueError(f"selected_examples_cap must be positive, got {selected_cap!r}.")
    uncapped_count = int(len(candidate) if uncapped_candidate_examples is None else uncapped_candidate_examples)
    if uncapped_count < len(candidate):
        raise ValueError(
            "uncapped_candidate_examples cannot be smaller than the selected candidate list "
            f"({uncapped_count} < {len(candidate)})."
        )
    if len(candidate) <= selected_cap:
        selected = list(candidate)
        was_capped = uncapped_count > len(selected)
    else:
        token = f"{selection_protocol}|{salt}|{int(seed)}|{len(candidate)}|{selected_cap}"
        local_seed = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16)
        rng = np.random.default_rng(local_seed)
        positions = rng.choice(np.arange(len(candidate)), size=selected_cap, replace=False)
        selected = [candidate[position] for position in sorted(int(value) for value in positions.tolist())]
        was_capped = True
    return np.asarray(selected, dtype=np.int64), {
        "example_selection_protocol": str(selection_protocol),
        "selected_examples": int(len(selected)),
        "selected_examples_cap": selected_cap,
        "uncapped_candidate_examples": uncapped_count,
        "candidate_examples_after_initial_selection": int(len(candidate)),
        "selection_was_capped": bool(was_capped),
    }


def evaluation_row_signature(
    *,
    scenario_key: str,
    split_phase: str,
    seed: int,
    target_nfe: int,
    solver_key: str,
    scheduler_key: str,
    checkpoint_id: str,
) -> str:
    """Build the stable identity for one evaluation result row."""

    return "|".join(
        [
            str(scenario_key),
            str(split_phase),
            str(seed),
            str(target_nfe),
            str(solver_key),
            str(scheduler_key),
            str(checkpoint_id),
        ]
    )


__all__ = [
    "RETIRED_EVALUATION_KEYS",
    "cap_context_indices",
    "consistent_metadata_value",
    "evaluation_row_signature",
    "reject_retired_evaluation_keys",
    "validate_declared_split_phase",
]
