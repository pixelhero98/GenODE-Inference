from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from genode.checkpoint_validation import (
    validate_locked_test_exclusion,
    validate_strict_integer,
    validate_tensor_state_dict,
)
from genode.cli import parse_csv, parse_int_csv
from genode.data.otflow_experiment_plan import CONDITIONAL_GENERATION_FAMILY, FORECAST_FAMILY
from genode.experiment_layout import SCENARIO_FAMILY_MOLECULE
from genode.data.molecule_xyz import load_molecule_group_manifest, molecule_group_root, trainable_molecule_group_members
from genode.evaluation.fm_backbone_registry import BACKBONE_NAME_OTFLOW_MOLECULE, MOLECULE_FAMILY, find_backbone_artifact, load_backbone_manifest
from genode.evaluation.molecule_metrics import evaluate_molecule_rollout_schedule, load_molecule_checkpoint_splits
from genode.solver_protocol import normalize_solver_key, normalize_solver_keys, solver_macro_steps
from genode.gipo.objectives import (
    CONDITIONAL_METRIC_SPECS,
    FORECAST_METRIC_SPECS,
    MOLECULE_METRIC_SPECS,
    UNIFORM_SCHEDULE_KEY,
    teacher_objective_specs_for_scenario,
    uniform_anchored_objective_columns,
)
from genode.gipo.ablation_plan import GIPO_POLICY_KEY
from genode.gipo.policy import (
    ARCHITECTURE_DENSITY_QUERY_TRANSFORMER,
    GIPO_PROTOCOL,
    MODEL_PAYLOAD_VERSION,
    TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET,
    EmbeddingNormalizer,
    build_gipo_student_model,
    context_embedding_id_from_row,
    context_embedding_kind_from_rows,
    context_id_from_row,
    load_context_embedding_table,
    normalize_gipo_checkpoint_payload,
    normalize_teacher_utility_weights,
    predict_gipo_density_many,
    read_metric_rows_csv,
    validate_gipo_attention_heads,
    validate_conditioning_style,
    validate_gipo_density_token_attention,
    validate_gipo_teacher_training_metadata,
    validate_context_embedding_kind,
)
from genode.gipo.evaluate_schedule_summary import (
    SER_REFERENCE_SCHEDULE_KEYS,
    _filter_rows_to_scheduler_keys,
    build_comparison_summary,
)
from genode.gipo.density_representation import DENSITY_BIN_COUNT
from genode.gipo.models import setting_encoder_config_from_payload, setting_feature_dim, validate_setting_mode
from genode.gipo.schema import reject_retired_evaluation_keys, validate_declared_split_phase
from genode.provenance import (
    file_sha256,
    fingerprint_identity,
    path_fingerprint,
)
from genode.data.otflow_paths import (
    backbone_manifest_path,
    project_dataset_root,
    resolve_project_path,
)
from genode.evaluation.otflow_evaluation_support import (
    DEFAULT_SHARED_BACKBONE_ROOT,
    LOCKED_TEST_PHASE,
    TRAIN_TUNING_PHASE,
    VALIDATION_PHASE,
    evaluate_forecast_schedule,
    load_conditional_generation_checkpoint_splits,
    load_forecast_checkpoint_splits,
)
from genode.schedule_transfer.diffusion_flow_schedules import BASELINE_SCHEDULE_KEYS, run_fixed_schedule_variant
from genode.runtime import ProgressBar, resolve_torch_device


def _report_input_fingerprints(paths_text: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for value in parse_csv(paths_text):
        path = resolve_project_path(value)
        fingerprint = path_fingerprint(path)
        records.append(
            {
                "logical_path": str(fingerprint["logical_path"]),
                **fingerprint_identity(fingerprint),
            }
        )
        if path.suffix.lower() == ".npz":
            manifest_path = path.with_suffix(path.suffix + ".manifest.json")
            manifest_fingerprint = path_fingerprint(manifest_path)
            records.append(
                {
                    "logical_path": str(manifest_fingerprint["logical_path"]),
                    **fingerprint_identity(manifest_fingerprint),
                }
            )
    return records

SELECTION_MODE_REPORTING = "reporting"
SELECTION_MODE_CALIBRATION = "calibration"
CONTEXT_DISJOINT_PHASE = "context_disjoint"
CALIBRATION_HOLDOUT_PHASES = (CONTEXT_DISJOINT_PHASE,)


def _read_csvs(paths_text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path_text in parse_csv(paths_text):
        rows.extend(read_metric_rows_csv(resolve_project_path(path_text)))
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(str(key))
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _validate_density_bin_count(payload: Mapping[str, Any], density_meta: Mapping[str, Any], *, role: str) -> None:
    expected = DENSITY_BIN_COUNT
    density_dim = validate_strict_integer(
        payload.get("density_dim"),
        label=f"GIPO {role} density_dim",
        minimum=2,
    )
    reference_bin_count = validate_strict_integer(
        density_meta.get("reference_bin_count"),
        label=f"GIPO {role} reference_bin_count",
        minimum=2,
    )
    reference_grid = tuple(density_meta.get("reference_time_grid", ()))
    if density_dim != expected or reference_bin_count != expected or len(reference_grid) != expected + 1:
        raise ValueError(
            f"GIPO {role} checkpoints require {expected} density bins; "
            f"got density_dim={density_dim}, reference_bin_count={reference_bin_count}, "
            f"reference_grid_len={len(reference_grid)}."
        )


def _source_split_phase(row: Mapping[str, Any]) -> str:
    return validate_declared_split_phase(row, source="GIPO reporter context row")


def _selection_split(row: Mapping[str, Any]) -> str:
    return str(row.get("selection_split") or row.get("report_split") or row.get("split_phase", row.get("split", ""))).strip()


def _report_context_split_fields(
    *,
    source_split_phase: str,
    report_split: str,
) -> Dict[str, str]:
    """Keep physical source provenance separate from the report partition."""

    source = str(source_split_phase).strip()
    report = str(report_split).strip()
    if not source or not report:
        raise ValueError("GIPO report rows require source_split_phase and report_split.")
    return {
        "split_phase": source,
        "source_split_phase": source,
        "report_split": report,
    }


def _validate_context_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split_phase: str,
    selection_mode: str,
) -> None:
    if not rows:
        raise ValueError("context rows input contains no rows.")
    for row_index, row in enumerate(rows):
        reject_retired_evaluation_keys(row, source=f"context row {row_index}")
        if not str(row.get("scenario_key", "")).strip():
            raise ValueError(f"context row {row_index} requires scenario_key.")
    mode = str(selection_mode)
    requested = str(split_phase)
    if mode == SELECTION_MODE_CALIBRATION:
        locked = [row for row in rows if _source_split_phase(row) == LOCKED_TEST_PHASE]
        if locked:
            raise ValueError("Calibration GIPO selection refuses locked_test rows.")
    if requested in CALIBRATION_HOLDOUT_PHASES:
        bad_selection = sorted({_selection_split(row) for row in rows if _selection_split(row) != requested})
        if bad_selection:
            raise ValueError(f"GIPO calibration reporter expected selection_split={requested!r}; found {bad_selection}.")
        bad_source = sorted({_source_split_phase(row) for row in rows if _source_split_phase(row) not in (TRAIN_TUNING_PHASE, VALIDATION_PHASE)})
        if bad_source:
            raise ValueError(f"Calibration holdout rows require train/validation source split phases; found {bad_source}.")
        return
    bad = sorted({_source_split_phase(row) for row in rows if _source_split_phase(row) != requested})
    if bad:
        raise ValueError(f"GIPO reporter expected source split_phase={requested!r}; found {bad}.")


def _evaluation_seed_from_row(row: Mapping[str, Any]) -> int:
    explicit = row.get("evaluation_seed", "")
    if explicit not in (None, "") and str(explicit).strip():
        return int(explicit)
    parent = str(row.get("parent_row_signature", "") or "")
    parts = parent.split("|")
    if len(parts) >= 3:
        try:
            return int(parts[2])
        except (TypeError, ValueError):
            pass
    return int(row["seed"])


def _logical_seed_from_row(row: Mapping[str, Any]) -> int:
    explicit = row.get("logical_seed", "")
    if explicit not in (None, "") and str(explicit).strip():
        return int(explicit)
    parent = str(row.get("parent_row_signature", "") or "")
    parts = parent.split("|")
    if len(parts) >= 3:
        try:
            return int(parts[2])
        except (TypeError, ValueError):
            pass
    seed = row.get("seed", "")
    if seed not in (None, "") and str(seed).strip():
        return int(seed)
    raise ValueError("context rows require logical_seed or seed for logical seed identity.")


def _checkpoint_step_from_row(row: Mapping[str, Any]) -> int:
    raw = row.get("checkpoint_step", "")
    if raw in (None, ""):
        raise ValueError("context rows require checkpoint_step.")
    return int(raw)


ContextMatchKey = Tuple[str, str, str, int, str, int, int, str, str]


def _row_group_key(row: Mapping[str, Any]) -> ContextMatchKey:
    return (
        _source_split_phase(row),
        str(row.get("benchmark_family", "")),
        str(row.get("scenario_key", "")),
        _logical_seed_from_row(row),
        str(row["solver_key"]),
        int(row["target_nfe"]),
        _checkpoint_step_from_row(row),
        str(row.get("checkpoint_id", "")),
        context_id_from_row(row),
    )


def _representative_context_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[ContextMatchKey, Dict[str, Any]] = {}
    metadata_by_context: Dict[Tuple[str, str, str, int, str, int, str], Dict[str, str]] = {}
    for row in rows:
        context_identity = (
            _source_split_phase(row),
            str(row.get("benchmark_family", "")),
            str(row.get("scenario_key", "")),
            _logical_seed_from_row(row),
            str(row["solver_key"]),
            int(row["target_nfe"]),
            context_id_from_row(row),
        )
        metadata = {
            field: str(row.get(field, "") or "").strip()
            for field in (
                "evaluation_seed",
                "checkpoint_step",
                "checkpoint_id",
                "context_embedding_id",
                "example_idx",
                "series_id",
                "series_idx",
                "target_t",
                "history_start",
                "history_stop",
                "target_stop",
                "locked_test_mode",
                "locked_test_context_limit",
                "locked_test_context_limit_scope",
                "selected_examples_cap_source",
                "selection_was_capped",
                "global_selection_was_capped",
            )
        }
        previous = metadata_by_context.get(context_identity)
        if previous is not None and previous != metadata:
            conflicts = sorted(field for field in metadata if previous.get(field) != metadata.get(field))
            raise ValueError(
                f"Conflicting duplicate context metadata for {context_identity!r}: fields={conflicts}."
            )
        metadata_by_context[context_identity] = metadata
        key = _row_group_key(row)
        if key not in grouped:
            copied = dict(row)
            copied["context_id"] = context_id_from_row(row)
            copied["seed"] = _logical_seed_from_row(row)
            copied["logical_seed"] = _logical_seed_from_row(row)
            copied["evaluation_seed"] = _evaluation_seed_from_row(row)
            copied["source_split_phase"] = _source_split_phase(row)
            copied["selection_split"] = _selection_split(row)
            grouped[key] = copied
    return [grouped[key] for key in sorted(grouped)]


def _filter_rows_to_contexts(
    rows: Sequence[Mapping[str, Any]],
    representatives: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    wanted = {_row_group_key(row) for row in representatives}
    filtered: List[Dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        try:
            key = _row_group_key(row)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Comparator row {row_index} cannot be matched to a locked-test context: {exc}"
            ) from exc
        if key in wanted:
            filtered.append(dict(row))
    return filtered


def _locked_test_selection_provenance(
    rows: Sequence[Mapping[str, Any]],
    *,
    split_phase: str,
) -> Dict[str, Any]:
    if str(split_phase) != LOCKED_TEST_PHASE:
        return {}
    if not rows:
        raise ValueError("Locked-test provenance requires at least one context row.")

    def required_values(field: str) -> set[str]:
        values = {str(row.get(field, "") or "").strip() for row in rows}
        if "" in values:
            raise ValueError(f"Locked-test context rows require {field} provenance.")
        return values

    modes = required_values("locked_test_mode")
    sources = required_values("selected_examples_cap_source")
    scopes = required_values("locked_test_context_limit_scope")
    if len(modes) != 1:
        raise ValueError(f"Locked-test context rows mix locked_test_mode values: {sorted(modes)}")
    mode = next(iter(modes))

    def required_bool(field: str) -> bool:
        values = required_values(field)
        normalized: set[bool] = set()
        for value in values:
            if value.lower() in {"true", "1"}:
                normalized.add(True)
            elif value.lower() in {"false", "0"}:
                normalized.add(False)
            else:
                raise ValueError(f"Locked-test context rows have invalid {field}={value!r}.")
        if len(normalized) != 1:
            raise ValueError(f"Locked-test context rows mix {field} values: {sorted(values)}")
        return next(iter(normalized))

    selection_was_capped = required_bool("selection_was_capped")
    global_selection_was_capped = required_bool("global_selection_was_capped")
    limit_values = {
        str(row.get("locked_test_context_limit", "") or "").strip()
        for row in rows
        if str(row.get("locked_test_context_limit", "") or "").strip().lower() not in {"", "none", "null"}
    }
    if mode == "full":
        if (
            sources != {"locked_test_full"}
            or scopes != {"none"}
            or limit_values
            or selection_was_capped
            or global_selection_was_capped
        ):
            raise ValueError(
                "Full locked-test context rows require selected_examples_cap_source='locked_test_full', "
                "locked_test_context_limit_scope='none', no context limit, and uncapped selection."
            )
        context_limit: int | None = None
    elif mode == "preview":
        if sources != {"locked_test_preview_contexts"} or scopes != {"per_seed"} or len(limit_values) != 1:
            raise ValueError(
                "Preview locked-test context rows require selected_examples_cap_source="
                "'locked_test_preview_contexts', scope='per_seed', and one context limit."
            )
        context_limit = int(next(iter(limit_values)))
        if context_limit <= 0:
            raise ValueError("locked_test_context_limit must be positive for preview rows.")
    else:
        raise ValueError(f"Unsupported locked_test_mode in context rows: {mode!r}.")
    return {
        "locked_test_mode": mode,
        "locked_test_context_limit": context_limit,
        "locked_test_context_limit_scope": next(iter(scopes)),
        "selected_examples_cap_source": next(iter(sources)),
        "selection_was_capped": selection_was_capped,
        "global_selection_was_capped": global_selection_was_capped,
    }


def _validate_checkpoint_step(rows: Sequence[Mapping[str, Any]], *, requested: int) -> int:
    steps = sorted({_checkpoint_step_from_row(row) for row in rows})
    if len(steps) != 1:
        raise ValueError(f"context rows mix checkpoint_step values: {steps}.")
    checkpoint_step = int(steps[0])
    if checkpoint_step != int(requested):
        raise ValueError(
            f"context rows checkpoint_step={checkpoint_step} does not match --checkpoint_step={int(requested)}."
        )
    return checkpoint_step


def _validate_loaded_checkpoint_identity(
    rows: Sequence[Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
) -> None:
    expected_id = str(checkpoint.get("checkpoint_id", "") or "").strip()
    if not expected_id:
        raise ValueError("Loaded backbone artifact is missing checkpoint_id.")
    if checkpoint.get("checkpoint_step") in (None, ""):
        raise ValueError("Loaded backbone artifact is missing checkpoint_step.")
    expected_step = validate_strict_integer(
        checkpoint["checkpoint_step"],
        label="Loaded backbone checkpoint_step",
        minimum=1,
    )
    observed_ids = sorted({str(row.get("checkpoint_id", "") or "").strip() for row in rows})
    observed_steps = sorted({_checkpoint_step_from_row(row) for row in rows})
    if observed_ids != [expected_id] or observed_steps != [expected_step]:
        raise ValueError(
            "context rows do not match the loaded backbone artifact: "
            f"checkpoint_ids={observed_ids}, checkpoint_steps={observed_steps}, "
            f"expected_id={expected_id!r}, expected_step={expected_step}."
        )


def _validate_strict_comparison_context_coverage(
    *,
    representatives: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    comparator_rows: Sequence[Mapping[str, Any]],
) -> None:
    expected = {_row_group_key(row) for row in representatives}
    problems: List[str] = []
    for label, rows, schedule_keys in (
        ("baseline", baseline_rows, BASELINE_SCHEDULE_KEYS),
        ("SER comparator", comparator_rows, SER_REFERENCE_SCHEDULE_KEYS),
    ):
        counts: Dict[Tuple[str, ContextMatchKey], int] = defaultdict(int)
        for row in rows:
            counts[(str(row.get("scheduler_key", "")), _row_group_key(row))] += 1
        for schedule_key in schedule_keys:
            observed = {key for (schedule, key), count in counts.items() if schedule == schedule_key and count > 0}
            missing = sorted(expected - observed)
            duplicates = sorted(
                key
                for (schedule, key), count in counts.items()
                if schedule == schedule_key and count > 1
            )
            if missing:
                problems.append(f"{label} schedule={schedule_key!r} missing contexts={missing[:4]}")
            if duplicates:
                problems.append(f"{label} schedule={schedule_key!r} duplicate contexts={duplicates[:4]}")
    if problems:
        raise ValueError("Strict locked-test comparison context coverage is incomplete: " + "; ".join(problems))


def _report_artifact_name(
    *,
    split_phase: str,
    selection_mode: str,
    locked_test_mode: str = "",
) -> str:
    if str(split_phase) == LOCKED_TEST_PHASE and str(selection_mode) == SELECTION_MODE_REPORTING:
        return "gipo_locked_test_preview_report" if str(locked_test_mode) == "preview" else "gipo_locked_test_report"
    return "gipo_student_calibration_report"


def _load_student_checkpoint(
    path: str | Path,
) -> Tuple[Any, EmbeddingNormalizer, Tuple[float, ...], Dict[str, Any]]:
    checkpoint_path = resolve_project_path(str(path))
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"GIPO student checkpoint {checkpoint_path} must contain a mapping payload.")
    payload = normalize_gipo_checkpoint_payload(payload)
    if str(payload.get("protocol", "")) != GIPO_PROTOCOL:
        raise ValueError(f"Unsupported GIPO student protocol {payload.get('protocol')!r}; expected {GIPO_PROTOCOL!r}.")
    if validate_strict_integer(
        payload.get("model_payload_version"),
        label="GIPO student model_payload_version",
        minimum=1,
    ) != MODEL_PAYLOAD_VERSION:
        raise ValueError(
            f"GIPO student checkpoint model_payload_version must be {MODEL_PAYLOAD_VERSION}; "
            f"got {payload.get('model_payload_version')!r}."
        )
    retired_checkpoint_keys = sorted(set(payload) & {"series_index_map", "series_conditioning"})
    if retired_checkpoint_keys:
        raise ValueError(
            f"GIPO student checkpoint uses retired series-conditioning keys: {retired_checkpoint_keys}."
        )
    if str(payload.get("student_policy_type", "")) != "continuous_density":
        raise ValueError("GIPO locked reporter only accepts continuous_density student checkpoints.")
    validate_locked_test_exclusion(
        payload,
        label="GIPO student checkpoint",
        required_root_keys=("locked_test_used_for_selection",),
    )
    density_meta = dict(payload.get("density_representation", {}))
    if str(density_meta.get("density_protocol", "")) != "density_mass":
        raise ValueError("GIPO student checkpoint is missing density_mass metadata.")
    _validate_density_bin_count(payload, density_meta, role="student")
    teacher_training = dict(payload.get("teacher_training", {}) or {})
    teacher_training_meta = validate_gipo_teacher_training_metadata(teacher_training)
    teacher_metric_targets = teacher_training_meta["teacher_metric_targets"]
    teacher_utility_weights = dict(payload.get("teacher_utility_weights") or teacher_training.get("teacher_utility_weights") or {})
    if not teacher_utility_weights:
        raise ValueError("GIPO student checkpoint is missing teacher_utility_weights.")
    normalize_teacher_utility_weights(teacher_metric_targets, teacher_utility_weights)
    reference_time_grid = tuple(float(x) for x in density_meta["reference_time_grid"])
    if "setting_feature_mode" in payload:
        raise ValueError("GIPO student checkpoint must use 'mode'; 'setting_feature_mode' is not supported.")
    if "mode" not in payload:
        raise ValueError("GIPO student checkpoint is missing mode.")
    mode_value = validate_setting_mode(str(payload["mode"]))
    raw_setting_encoder_config = payload.get("setting_encoder_config")
    if not isinstance(raw_setting_encoder_config, Mapping):
        raise ValueError("GIPO student checkpoint is missing setting_encoder_config.")
    setting_encoder_config = setting_encoder_config_from_payload(
        raw_setting_encoder_config, require_complete=True
    )
    expected_setting_dim = setting_feature_dim(mode_value, config=setting_encoder_config)
    setting_dim = validate_strict_integer(
        payload.get("setting_dim"),
        label="GIPO student setting_dim",
        minimum=1,
    )
    if setting_dim != int(expected_setting_dim):
        raise ValueError(
            f"GIPO student checkpoint setting_dim={payload['setting_dim']} does not match "
            f"setting encoder dim {expected_setting_dim} for {setting_encoder_config.mode}."
        )
    normalizer = EmbeddingNormalizer.from_payload(payload["embedding_normalizer"])
    context_dim = validate_strict_integer(
        payload.get("context_dim"),
        label="GIPO student context_dim",
        minimum=1,
    )
    if normalizer.mean.shape != (context_dim,):
        raise ValueError(
            "GIPO student checkpoint context_dim does not match its embedding normalizer."
        )
    student_architecture = str(payload.get("student_architecture", ""))
    if student_architecture != ARCHITECTURE_DENSITY_QUERY_TRANSFORMER:
        raise ValueError(
            f"GIPO student checkpoints must use {ARCHITECTURE_DENSITY_QUERY_TRANSFORMER!r}; got {student_architecture!r}."
        )
    student_model_config = dict(payload.get("student_model_config", {}) or {})
    validate_conditioning_style(
        student_model_config,
        require_present=True,
    )
    validate_gipo_density_token_attention(student_model_config, require_present=True)
    validate_gipo_attention_heads(student_model_config.get("attention_heads"))
    density_dim = validate_strict_integer(
        payload.get("density_dim"),
        label="GIPO student density_dim",
        minimum=2,
    )
    student = build_gipo_student_model(
        architecture=student_architecture,
        setting_dim=setting_dim,
        density_dim=density_dim,
        context_dim=context_dim,
        model_config=student_model_config,
    )
    raw_student_state = payload.get("student_state")
    if not isinstance(raw_student_state, Mapping):
        raise ValueError("GIPO student checkpoint is missing student_state.")
    try:
        student.load_state_dict(
            validate_tensor_state_dict(raw_student_state, label="GIPO student state"),
            strict=True,
        )
    except RuntimeError as exc:
        raise ValueError(f"GIPO student state is incompatible: {exc}") from exc
    student.eval()
    payload["mode"] = mode_value
    payload["setting_encoder_config"] = setting_encoder_config.to_payload()
    payload["student_architecture"] = student_architecture
    payload["student_model_config"] = student.model_config()
    return student, normalizer, reference_time_grid, payload


def _teacher_final_retrain_metadata(
    checkpoint_payload: Mapping[str, Any],
    training_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    if "final_teacher_retrain" in checkpoint_payload or "final_teacher_retrain" in training_summary:
        raise ValueError("GIPO metadata must use teacher_final_retrain; final_teacher_retrain is not supported.")
    checkpoint_value = dict(checkpoint_payload.get("teacher_final_retrain") or {})
    summary_value = dict(training_summary.get("teacher_final_retrain") or {})
    if checkpoint_value and summary_value and checkpoint_value != summary_value:
        raise ValueError("GIPO checkpoint and training summary disagree on teacher_final_retrain metadata.")
    return checkpoint_value or summary_value


def _gipo_step_budget_metadata(
    checkpoint_payload: Mapping[str, Any],
    training_summary: Mapping[str, Any],
) -> int:
    checkpoint_value = checkpoint_payload.get("gipo_step_budget")
    summary_value = training_summary.get("gipo_step_budget")
    if checkpoint_value in (None, "") or summary_value in (None, ""):
        raise ValueError("GIPO checkpoint and training summary must identify gipo_step_budget.")
    checkpoint_budget = validate_strict_integer(
        checkpoint_value,
        label="GIPO checkpoint gipo_step_budget",
        minimum=1,
    )
    summary_budget = validate_strict_integer(
        summary_value,
        label="GIPO training summary gipo_step_budget",
        minimum=1,
    )
    if checkpoint_budget != summary_budget:
        raise ValueError(
            "GIPO checkpoint and training summary disagree on gipo_step_budget: "
            f"{checkpoint_budget} != {summary_budget}."
        )
    return checkpoint_budget


def _mode_metadata(
    checkpoint_payload: Mapping[str, Any],
    training_summary: Mapping[str, Any],
) -> str:
    checkpoint_mode = str(checkpoint_payload.get("mode", "") or "").strip()
    summary_mode = str(training_summary.get("mode", "") or "").strip()
    if not checkpoint_mode or not summary_mode:
        raise ValueError("GIPO checkpoint and training summary must identify mode.")
    if checkpoint_mode != summary_mode:
        raise ValueError(
            "GIPO checkpoint and training summary disagree on mode: "
            f"{checkpoint_mode!r} != {summary_mode!r}."
        )
    return checkpoint_mode


def _clean_metadata_string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _model_conditioning_style(payload: Mapping[str, Any], config_key: str) -> str:
    config = payload.get(config_key)
    if not isinstance(config, Mapping):
        return ""
    return _clean_metadata_string(config.get("conditioning_style"))


def _conditioning_metadata_for_summary(
    checkpoint_payload: Mapping[str, Any],
    training_summary: Mapping[str, Any],
) -> Dict[str, str]:
    candidates = {
        style
        for style in (
            _clean_metadata_string(checkpoint_payload.get("conditioning_style")),
            _clean_metadata_string(training_summary.get("conditioning_style")),
            _model_conditioning_style(checkpoint_payload, "student_model_config"),
            _model_conditioning_style(training_summary, "student_model_config"),
            _model_conditioning_style(checkpoint_payload, "teacher_model_config"),
            _model_conditioning_style(training_summary, "teacher_model_config"),
        )
        if style
    }
    for style in candidates:
        validate_conditioning_style({"conditioning_style": style})
    if len(candidates) > 1:
        raise ValueError(f"GIPO metadata contains inconsistent conditioning_style values: {sorted(candidates)}")
    if not candidates:
        return {}
    return {"conditioning_style": next(iter(candidates))}


def _aggregate_seed_rows(rows: Sequence[Mapping[str, Any]], *, split_phase: str) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int, str, int, int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("scenario_key", "")),
                int(row["seed"]),
                str(row["solver_key"]),
                int(row["target_nfe"]),
                _checkpoint_step_from_row(row),
                str(row["scheduler_key"]),
            )
        ].append(row)
    out: List[Dict[str, Any]] = []
    for (scenario_key, seed, solver, target_nfe, checkpoint_step, scheduler_key), group in sorted(grouped.items()):
        checkpoint_ids = sorted(
            {
                str(item.get("checkpoint_id", "") or "").strip()
                for item in group
                if str(item.get("checkpoint_id", "") or "").strip()
            }
        )
        row: Dict[str, Any] = {
            "scenario_key": scenario_key,
            "split_phase": str(split_phase),
            "seed": int(seed),
            "solver_key": solver,
            "target_nfe": int(target_nfe),
            "checkpoint_step": int(checkpoint_step),
            "checkpoint_id": checkpoint_ids[0] if len(checkpoint_ids) == 1 else "",
            "checkpoint_ids_json": json.dumps(checkpoint_ids, separators=(",", ":")),
            "checkpoint_maturity_label": str(group[0].get("checkpoint_maturity_label", "")),
            "checkpoint_maturity_index": group[0].get("checkpoint_maturity_index", ""),
            "scheduler_key": scheduler_key,
            "context_count": int(len(group)),
        }
        for field in ("method_key", "gipo_step_budget", "mode", "teacher_final_retrain"):
            values = {str(item.get(field, "")) for item in group}
            if len(values) != 1 or "" in values:
                raise ValueError(
                    f"GIPO aggregate rows require one non-empty {field} value per group; "
                    f"found {sorted(values)}."
                )
            value = next(iter(values))
            row[field] = int(value) if field == "gipo_step_budget" else value
        for field in (
            "locked_test_mode",
            "locked_test_context_limit",
            "locked_test_context_limit_scope",
            "selected_examples_cap_source",
            "selection_was_capped",
            "global_selection_was_capped",
        ):
            if field in group[0]:
                row[field] = group[0].get(field)
        for metric_key in (
            "forecast_crps",
            "forecast_mase",
            "forecast_mse",
            "score_main",
            "temporal_uw1",
            "temporal_cw1",
            "temporal_tstr_f1",
            "disc_auc",
            "disc_auc_gap",
            "u_l1",
            "c_l1",
            "spread_specific_error",
            "imbalance_specific_error",
            "ret_vol_acf_error",
            "impact_response_error",
            "molecule_kabsch_rmsd_3d",
            "molecule_ensemble_velocity_norm_w1",
            "molecule_ensemble_acceleration_norm_w1",
            "molecule_rollout_velocity_norm_w1",
            "molecule_rollout_acceleration_norm_w1",
            "molecule_coordinate_w1_mean",
            "molecule_pair_distance_w1",
            "u_comp_uniform",
        ):
            vals = []
            for item in group:
                value = item.get(metric_key, "")
                if value in (None, ""):
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(numeric):
                    vals.append(float(numeric))
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                row[metric_key] = float(np.mean(arr))
                row[f"{metric_key}_std"] = float(np.std(arr))
        if any("temporal_tstr_f1_applicable" in item for item in group):
            row["temporal_tstr_f1_applicable"] = bool(
                any(str(item.get("temporal_tstr_f1_applicable", "")).lower() == "true" for item in group)
            )
        out.append(row)
    return out


def _aggregate_molecule_member_rows(rows: Sequence[Mapping[str, Any]], *, split_phase: str) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        member_key = str(row.get("member_key") or row.get("axis_member") or "").strip()
        stratum = str(row.get("stratum") or row.get("axis_stratum") or "").strip()
        if not member_key or not stratum:
            raise ValueError("Molecule member aggregate rows require member_key/stratum on every context row.")
        grouped[(member_key, stratum)].append(row)
    out: List[Dict[str, Any]] = []
    for (member_key, stratum), group in sorted(grouped.items()):
        for row in _aggregate_seed_rows(group, split_phase=split_phase):
            first = group[0]
            row.update(
                {
                    "member_key": member_key,
                    "stratum": stratum,
                    "formula": str(first.get("formula", first.get("axis_formula", ""))),
                    "atom_count": str(first.get("atom_count", first.get("axis_atom_count", ""))),
                    "source_zip_name": str(first.get("source_zip_name", "")),
                }
            )
            out.append(row)
    return out


def _numeric_metric_means(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    excluded = {
        "seed",
        "target_nfe",
        "context_count",
        "series_idx",
        "example_idx",
    }
    values: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if str(key) in excluded or str(key).endswith("_std"):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric):
                values[str(key)].append(float(numeric))
    return {key: float(np.mean(np.asarray(items, dtype=np.float64))) for key, items in sorted(values.items()) if items}


def _forecast_dataset_for_source_phase(splits: Mapping[str, Any], source_phase: str):
    phase = str(source_phase)
    if phase == TRAIN_TUNING_PHASE:
        return splits["train"]
    if phase == VALIDATION_PHASE:
        return splits["val"]
    if phase == LOCKED_TEST_PHASE:
        return splits["test"]
    raise ValueError(f"Unsupported source split_phase {source_phase!r}.")


def _benchmark_family_from_rows(rows: Sequence[Mapping[str, Any]], requested: str = "") -> str:
    requested = str(requested or "").strip()
    values = [str(row.get("benchmark_family", "") or "").strip() for row in rows]
    missing = [index for index, value in enumerate(values) if not value]
    if missing:
        raise ValueError(
            "GIPO locked-test context rows require explicit benchmark_family values; "
            f"missing at rows {missing[:8]}."
        )
    families = set(values)
    if requested:
        if families != {requested}:
            raise ValueError(f"context rows benchmark_family mismatch: requested {requested!r}, found {sorted(families)}.")
        return requested
    if len(families) != 1:
        raise ValueError(f"GIPO reporter requires one benchmark_family per report; found {sorted(families)}.")
    return next(iter(families))


def _output_prefix(split_phase: str, selection_mode: str, report_label: str = "") -> str:
    if str(split_phase) == LOCKED_TEST_PHASE and str(selection_mode) == SELECTION_MODE_REPORTING:
        return "locked_test_gipo"
    label = str(report_label).strip() or str(split_phase).strip() or "gipo"
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in label)
    return f"{safe}_gipo"


def _molecule_member_from_row(row: Mapping[str, Any]) -> Tuple[str, str]:
    member_key = str(row.get("member_key") or row.get("axis_member") or "").strip()
    stratum = str(row.get("stratum") or row.get("axis_stratum") or "").strip()
    if not member_key or not stratum:
        raise ValueError("Molecule GIPO reporting rows require member_key/axis_member and stratum/axis_stratum.")
    return member_key, stratum


def _molecule_group_member_lookup(scenario_key: str, group_root: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    manifest = load_molecule_group_manifest(str(scenario_key), group_root)
    return {
        (str(member["member_key"]), str(member["stratum"])): dict(member)
        for member in trainable_molecule_group_members(manifest)
    }


def _molecule_processed_dir(group_root: Path, scenario_key: str, member: Mapping[str, Any]) -> Path:
    return group_root / str(scenario_key) / str(member["processed_dir"])


def _validate_molecule_checkpoint_identities(
    rows: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
    scenario_key: str,
) -> None:
    checked: Dict[Tuple[str, str, int], str] = {}
    for row in rows:
        member_key, stratum = _molecule_member_from_row(row)
        checkpoint_step = _checkpoint_step_from_row(row)
        key = (member_key, stratum, checkpoint_step)
        if key not in checked:
            artifact = find_backbone_artifact(
                manifest,
                backbone_name=BACKBONE_NAME_OTFLOW_MOLECULE,
                benchmark_family=MOLECULE_FAMILY,
                dataset_key=str(scenario_key),
                train_steps=int(checkpoint_step),
                member_key=member_key,
                stratum=stratum,
            )
            checked[key] = str(artifact["checkpoint_id"])
        observed_id = str(row.get("checkpoint_id", "") or "").strip()
        if observed_id != checked[key]:
            raise ValueError(
                "context row does not match its molecule backbone artifact: "
                f"member={member_key!r}, stratum={stratum!r}, checkpoint_step={checkpoint_step}, "
                f"checkpoint_id={observed_id!r}, expected={checked[key]!r}."
            )


def _metric_specs_for_family(benchmark_family: str, scenario_key: str = ""):
    if str(benchmark_family) == FORECAST_FAMILY:
        return FORECAST_METRIC_SPECS
    if str(benchmark_family) == CONDITIONAL_GENERATION_FAMILY:
        return teacher_objective_specs_for_scenario(str(scenario_key)) if str(scenario_key).strip() else CONDITIONAL_METRIC_SPECS
    if str(benchmark_family) == SCENARIO_FAMILY_MOLECULE:
        return MOLECULE_METRIC_SPECS
    raise ValueError(f"Unsupported benchmark_family={benchmark_family!r}.")


def _row_text_value(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field, "")
    return str(value).strip() if str(value).strip() else ""


def _single_value_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    requested: str = "",
    arg_name: str | None = None,
) -> str:
    requested_text = str(requested or "").strip()
    values = sorted({_row_text_value(row, field) for row in rows if _row_text_value(row, field)})
    if requested_text:
        if requested_text not in values:
            raise ValueError(f"Requested {arg_name or field}={requested_text!r} is not present in context_rows values {values}.")
        return requested_text
    if len(values) != 1:
        raise ValueError(f"Could not infer a single {arg_name or field} from context_rows; found {values}.")
    return values[0]


def _infer_int_values_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    requested: str = "",
    arg_name: str | None = None,
) -> Tuple[int, ...]:
    if str(requested or "").strip():
        return tuple(parse_int_csv(requested))
    values = sorted({int(row[field]) for row in rows if str(row.get(field, "")).strip()})
    if not values:
        raise ValueError(f"Could not infer {arg_name or field} from context_rows; pass --{arg_name or field}.")
    return tuple(int(value) for value in values)


def _infer_solver_values_from_rows(rows: Sequence[Mapping[str, Any]], *, requested: str = "") -> Tuple[str, ...]:
    if str(requested or "").strip():
        return tuple(normalize_solver_keys(str(requested)))
    values = sorted({normalize_solver_key(str(row["solver_key"])) for row in rows if str(row.get("solver_key", "")).strip()})
    if not values:
        raise ValueError("Could not infer solver_names from context_rows; pass --solver_names.")
    return tuple(values)


def _filter_representatives_to_matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    scenario_key: str,
    seeds: Sequence[int],
    solvers: Sequence[str],
    target_nfes: Sequence[int],
) -> List[Dict[str, Any]]:
    seed_set = {int(seed) for seed in seeds}
    solver_set = {normalize_solver_key(str(solver)) for solver in solvers}
    nfe_set = {int(value) for value in target_nfes}
    filtered = [
        dict(row)
        for row in rows
        if str(row.get("scenario_key", "")) == str(scenario_key)
        and _logical_seed_from_row(row) in seed_set
        and normalize_solver_key(str(row["solver_key"])) in solver_set
        and int(row["target_nfe"]) in nfe_set
    ]
    if not filtered:
        raise ValueError("context_rows has no representative rows after applying scenario/seed/solver/NFE filters.")
    return filtered


def _attach_uniform_rewards_to_gipo_row(
    row: Mapping[str, Any],
    *,
    uniform_row: Mapping[str, Any] | None,
    benchmark_family: str,
    scenario_key: str = "",
) -> Dict[str, Any]:
    out = dict(row)
    if uniform_row is None:
        raise ValueError(
            "Locked-test GIPO reporting requires a matched uniform row for every "
            f"context/solver/NFE/checkpoint cell; missing context_id={context_id_from_row(row)!r} "
            f"solver={row.get('solver_key')!r} target_nfe={row.get('target_nfe')!r}."
        )
    reward_columns = uniform_anchored_objective_columns(
        {**out, "scheduler_key": GIPO_POLICY_KEY},
        {**dict(uniform_row), "scheduler_key": UNIFORM_SCHEDULE_KEY},
        _metric_specs_for_family(benchmark_family, scenario_key),
        uniform_scheduler_key=UNIFORM_SCHEDULE_KEY,
    )
    out.update(reward_columns)
    out["gipo_reward_protocol"] = GIPO_PROTOCOL
    out["reward_anchor_scheduler_key"] = UNIFORM_SCHEDULE_KEY
    out["reward_utility_transform"] = "directional_log_uniform_anchor"
    out["reward_granularity"] = "context_window_metric_components"
    return out


def report_gipo_locked_test(args: argparse.Namespace) -> Dict[str, Any]:
    checkpoint_path = resolve_project_path(str(args.gipo_student_checkpoint))
    student, normalizer, reference_time_grid, checkpoint_payload = _load_student_checkpoint(
        str(checkpoint_path),
    )
    training_summary_path = resolve_project_path(str(args.training_summary))
    training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    if "model_payload_version" in training_summary:
        training_summary = normalize_gipo_checkpoint_payload(training_summary)
    if str(training_summary.get("protocol", "")) != GIPO_PROTOCOL:
        raise ValueError("training_summary protocol does not match continuous-density GIPO.")
    validate_locked_test_exclusion(
        training_summary,
        label="GIPO training_summary",
        required_root_keys=("locked_test_used_for_selection",),
    )
    expected_checkpoint_hash = str(
        training_summary.get("gipo_student_checkpoint_sha256", "")
    ).strip()
    if len(expected_checkpoint_hash) != 64 or any(
        character not in "0123456789abcdef" for character in expected_checkpoint_hash
    ):
        raise ValueError(
            "training_summary requires a valid gipo_student_checkpoint_sha256."
        )
    if file_sha256(checkpoint_path) != expected_checkpoint_hash:
        raise ValueError(
            "training_summary does not belong to the supplied GIPO student checkpoint."
        )
    if not str(checkpoint_payload.get("context_embedding_kind", "") or "").strip():
        raise ValueError("GIPO checkpoint requires an explicit context_embedding_kind.")
    if not str(training_summary.get("context_embedding_kind", "") or "").strip():
        raise ValueError("GIPO training summary requires an explicit context_embedding_kind.")
    checkpoint_embedding_kind = validate_context_embedding_kind(
        checkpoint_payload.get("context_embedding_kind")
    )
    summary_embedding_kind = validate_context_embedding_kind(
        training_summary.get("context_embedding_kind")
    )
    if checkpoint_embedding_kind != summary_embedding_kind:
        raise ValueError(
            "GIPO checkpoint and training summary disagree on context_embedding_kind."
        )
    checkpoint_policy_key = str(checkpoint_payload.get("student_policy_key", "") or "").strip()
    summary_policy_key = str(training_summary.get("student_policy_key", "") or "").strip()
    if not checkpoint_policy_key or not summary_policy_key:
        raise ValueError("GIPO checkpoint and training summary must identify student_policy_key.")
    if checkpoint_policy_key != summary_policy_key:
        raise ValueError(
            "GIPO checkpoint and training summary disagree on student_policy_key: "
            f"{checkpoint_policy_key!r} != {summary_policy_key!r}."
        )
    checkpoint_scenario = str(checkpoint_payload.get("scenario_key", "") or "").strip()
    summary_scenario = str(training_summary.get("scenario_key", "") or "").strip()
    checkpoint_family = str(
        checkpoint_payload.get("benchmark_family", "") or ""
    ).strip()
    summary_family = str(training_summary.get("benchmark_family", "") or "").strip()
    if not all((checkpoint_scenario, summary_scenario, checkpoint_family, summary_family)):
        raise ValueError(
            "GIPO checkpoint and training summary require scenario and benchmark-family metadata."
        )
    if (checkpoint_scenario, checkpoint_family) != (summary_scenario, summary_family):
        raise ValueError(
            "GIPO checkpoint and training summary disagree on scenario provenance."
        )
    student_policy_key = checkpoint_policy_key
    gipo_step_budget = _gipo_step_budget_metadata(checkpoint_payload, training_summary)
    teacher_final_retrain = _teacher_final_retrain_metadata(checkpoint_payload, training_summary)
    mode = _mode_metadata(checkpoint_payload, training_summary)
    required_checkpoint_selection_mode = str(getattr(args, "require_teacher_checkpoint_selection_mode", "") or "").strip()
    if required_checkpoint_selection_mode:
        if required_checkpoint_selection_mode != TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET:
            raise ValueError(
                "GIPO reporter only supports "
                f"{TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET!r} checkpoints."
            )
        actual_selection_mode = str(
            checkpoint_payload.get("teacher_checkpoint_selection_mode")
            or training_summary.get("teacher_checkpoint_selection_mode")
            or ""
        )
        if actual_selection_mode != required_checkpoint_selection_mode:
            raise ValueError(
                f"GIPO reporter requires teacher_checkpoint_selection_mode={required_checkpoint_selection_mode!r}; "
                f"got {actual_selection_mode!r}."
            )
        if required_checkpoint_selection_mode == TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET:
            if teacher_final_retrain.get("enabled") is not True:
                raise ValueError("GIPO reporter requires final teacher retrain metadata for weighted-normalized-regret checkpoints.")

    selection_mode = str(getattr(args, "selection_mode", SELECTION_MODE_REPORTING))
    split_phase = str(getattr(args, "split_phase", LOCKED_TEST_PHASE))
    context_rows_arg = str(getattr(args, "context_rows", ""))
    embeddings_arg = str(getattr(args, "context_embeddings_npz", ""))
    if not context_rows_arg.strip():
        raise ValueError("GIPO reporter requires --context_rows.")
    if not embeddings_arg.strip():
        raise ValueError("GIPO reporter requires --context_embeddings_npz.")
    report_inputs = {
        "gipo_student_checkpoint": _report_input_fingerprints(str(checkpoint_path)),
        "training_summary": _report_input_fingerprints(str(training_summary_path)),
        "context_rows": _report_input_fingerprints(context_rows_arg),
        "context_embeddings_npz": _report_input_fingerprints(embeddings_arg),
        "baseline_rows": _report_input_fingerprints(str(args.baseline_rows)),
        "comparator_rows": _report_input_fingerprints(str(args.comparator_rows)),
    }
    context_rows = _read_csvs(context_rows_arg)
    row_embedding_kind = context_embedding_kind_from_rows(
        context_rows,
        label="GIPO reporting context_rows",
        require_explicit=True,
    )
    if row_embedding_kind != checkpoint_embedding_kind:
        raise ValueError(
            "GIPO reporting context rows use a different context_embedding_kind than "
            f"the checkpoint: {row_embedding_kind!r} != {checkpoint_embedding_kind!r}."
        )
    _validate_context_rows(context_rows, split_phase=split_phase, selection_mode=selection_mode)
    _benchmark_family_from_rows(context_rows, str(getattr(args, "benchmark_family", "") or ""))
    representatives = _representative_context_rows(context_rows)
    benchmark_family = _benchmark_family_from_rows(representatives, str(getattr(args, "benchmark_family", "") or ""))
    scenario_key = _single_value_from_rows(
        representatives,
        field="scenario_key",
        requested=str(getattr(args, "scenario_key", "") or ""),
        arg_name="scenario_key",
    )
    if (scenario_key, benchmark_family) != (checkpoint_scenario, checkpoint_family):
        raise ValueError(
            "GIPO reporting rows do not match the checkpoint training scenario."
        )
    seeds = _infer_int_values_from_rows(
        representatives,
        field="seed",
        requested=str(getattr(args, "seeds", "") or ""),
        arg_name="seeds",
    )
    solvers = _infer_solver_values_from_rows(representatives, requested=str(getattr(args, "solver_names", "") or ""))
    target_nfes = _infer_int_values_from_rows(
        representatives,
        field="target_nfe",
        requested=str(getattr(args, "target_nfe_values", "") or ""),
        arg_name="target_nfe_values",
    )
    representatives = _filter_representatives_to_matrix(
        representatives,
        scenario_key=scenario_key,
        seeds=seeds,
        solvers=solvers,
        target_nfes=target_nfes,
    )
    locked_test_provenance = _locked_test_selection_provenance(
        representatives,
        split_phase=split_phase,
    )
    checkpoint_step = _validate_checkpoint_step(
        representatives,
        requested=int(args.checkpoint_step),
    )
    group_root = resolve_project_path(str(getattr(args, "molecule_group_root", "") or molecule_group_root()))
    molecule_members_for_coverage: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if benchmark_family == SCENARIO_FAMILY_MOLECULE:
        molecule_members_for_coverage = _molecule_group_member_lookup(scenario_key, group_root)
        expected_cells = {
            (scenario_key, member_key, stratum, seed, solver, target_nfe)
            for member_key, stratum in sorted(molecule_members_for_coverage)
            for seed in seeds
            for solver in solvers
            for target_nfe in target_nfes
        }
        observed_cells = {
            (
                str(row.get("scenario_key", "")),
                str(row.get("member_key") or row.get("axis_member") or ""),
                str(row.get("stratum") or row.get("axis_stratum") or ""),
                _logical_seed_from_row(row),
                str(row["solver_key"]),
                int(row["target_nfe"]),
            )
            for row in representatives
        }
    else:
        expected_cells = {(scenario_key, seed, solver, target_nfe) for seed in seeds for solver in solvers for target_nfe in target_nfes}
        observed_cells = {
            (str(row.get("scenario_key", "")), _logical_seed_from_row(row), str(row["solver_key"]), int(row["target_nfe"]))
            for row in representatives
        }
    missing_cells = sorted(expected_cells - observed_cells)
    if missing_cells:
        raise ValueError(f"context rows are missing seed/solver/NFE cells: {missing_cells[:8]}")

    if not str(args.baseline_rows).strip():
        raise ValueError("GIPO locked-test reporting requires --baseline_rows with matched uniform rows.")
    baseline_rows = _read_csvs(str(args.baseline_rows))
    comparator_rows = _read_csvs(str(args.comparator_rows)) if str(args.comparator_rows).strip() else []
    baseline_rows = _filter_rows_to_scheduler_keys(baseline_rows, BASELINE_SCHEDULE_KEYS)
    comparator_rows = _filter_rows_to_scheduler_keys(comparator_rows, SER_REFERENCE_SCHEDULE_KEYS)
    if baseline_rows:
        _benchmark_family_from_rows(baseline_rows, benchmark_family)
    if comparator_rows:
        _benchmark_family_from_rows(comparator_rows, benchmark_family)
    baseline_rows = _filter_rows_to_contexts(baseline_rows, representatives)
    comparator_rows = _filter_rows_to_contexts(comparator_rows, representatives)
    if not baseline_rows:
        raise ValueError("--baseline_rows contains no matched fixed-baseline rows for the requested matrix.")
    strict_locked_comparison = (
        split_phase == LOCKED_TEST_PHASE
        and selection_mode == SELECTION_MODE_REPORTING
        and (
            student_policy_key == GIPO_POLICY_KEY
            or not bool(getattr(args, "allow_incomplete_comparison", False))
        )
    )
    if strict_locked_comparison and not comparator_rows:
        raise ValueError(
            "Strict locked-test reporting requires --comparator_rows with matched SER reference rows."
        )
    if strict_locked_comparison:
        _validate_strict_comparison_context_coverage(
            representatives=representatives,
            baseline_rows=baseline_rows,
            comparator_rows=comparator_rows,
        )
        for label, rows in (("baseline", baseline_rows), ("SER comparator", comparator_rows)):
            comparison_provenance = _locked_test_selection_provenance(rows, split_phase=split_phase)
            if comparison_provenance != locked_test_provenance:
                raise ValueError(
                    f"Strict locked-test {label} rows use different selection provenance: "
                    f"{comparison_provenance!r} != {locked_test_provenance!r}."
                )
    uniform_context_rows = {
        _row_group_key(row): dict(row)
        for row in baseline_rows
        if str(row.get("scheduler_key", "")) == UNIFORM_SCHEDULE_KEY
    }

    raw_embeddings = load_context_embedding_table(
        resolve_project_path(embeddings_arg),
        expected_context_embedding_kind=checkpoint_embedding_kind,
        require_manifest=True,
        expected_context_rows=context_rows,
    )
    required_embedding_ids = {context_embedding_id_from_row(row) for row in representatives}
    missing_context_embeddings = sorted(required_embedding_ids - set(raw_embeddings))
    if missing_context_embeddings:
        raise KeyError(f"context embeddings NPZ is missing contexts: {missing_context_embeddings[:8]}")
    embeddings = normalizer.transform_table(
        {context_id: raw_embeddings[context_id] for context_id in sorted(required_embedding_ids)}
    )

    device = resolve_torch_device(str(args.device))
    student.to(device)
    if benchmark_family == FORECAST_FAMILY:
        checkpoint = load_forecast_checkpoint_splits(
            cli_args=args,
            dataset_root=resolve_project_path(str(args.dataset_root)),
            shared_backbone_root=resolve_project_path(str(args.shared_backbone_root)),
            dataset=scenario_key,
            device=device,
        )
    elif benchmark_family == CONDITIONAL_GENERATION_FAMILY:
        checkpoint = load_conditional_generation_checkpoint_splits(
            cli_args=args,
            shared_backbone_root=resolve_project_path(str(args.shared_backbone_root)),
            dataset=scenario_key,
            device=device,
        )
    elif benchmark_family == SCENARIO_FAMILY_MOLECULE:
        checkpoint = None
        molecule_manifest = load_backbone_manifest(resolve_project_path(str(args.backbone_manifest)))
        molecule_members = molecule_members_for_coverage or _molecule_group_member_lookup(scenario_key, group_root)
        molecule_cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    else:
        raise NotImplementedError(f"GIPO locked-test reporting does not support benchmark_family={benchmark_family!r}.")
    if checkpoint is not None:
        _validate_loaded_checkpoint_identity(representatives, checkpoint)
        model = checkpoint["model"]
        cfg = checkpoint["cfg"]
        splits = checkpoint["splits"]
    else:
        _validate_molecule_checkpoint_identities(
            representatives,
            manifest=molecule_manifest,
            scenario_key=scenario_key,
        )
        model = None
        cfg = None
        splits = None

    out_dir = resolve_project_path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = _output_prefix(split_phase, selection_mode, str(getattr(args, "report_label", "")))
    decision_rows: List[Dict[str, Any]] = []
    per_context_rows: List[Dict[str, Any]] = []
    predictions = predict_gipo_density_many(
        student,
        rows=representatives,
        context_embeddings=embeddings,
        reference_time_grid=reference_time_grid,
        mode=mode,
        setting_encoder_config=checkpoint_payload.get("setting_encoder_config"),
        device=device,
    )
    with ProgressBar(len(representatives), f"GIPO {split_phase}") as progress:
        for row_idx, row in enumerate(representatives):
            context_id = context_id_from_row(row)
            source_phase = _source_split_phase(row)
            prediction = predictions[row_idx]
            example_idx = int(row.get("example_idx", row.get("example_index", 0)))
            solver = str(row["solver_key"])
            target_nfe = int(row["target_nfe"])
            eval_seed = int(row["evaluation_seed"])
            logical_seed = _logical_seed_from_row(row)
            macro_steps = int(solver_macro_steps(solver, target_nfe))
            if benchmark_family == FORECAST_FAMILY:
                eval_ds = _forecast_dataset_for_source_phase(splits, source_phase)
                metrics = evaluate_forecast_schedule(
                    model,
                    eval_ds,
                    cfg,
                    solver_name=normalize_solver_key(solver),
                    macro_steps=int(macro_steps),
                    target_nfe=int(target_nfe),
                    time_grid=prediction["time_grid"],
                    num_eval_samples=int(args.num_eval_samples),
                    seed=int(eval_seed),
                    scheduler_key=GIPO_POLICY_KEY,
                    scenario_key=scenario_key,
                    split_phase=source_phase,
                    checkpoint_id=str(checkpoint["checkpoint_id"]),
                    example_indices=[example_idx],
                    batch_size=int(args.forecast_eval_batch_size),
                    progress_label="",
                    return_per_example_rows=False,
                )
            elif benchmark_family == CONDITIONAL_GENERATION_FAMILY:
                eval_ds = _forecast_dataset_for_source_phase(splits, source_phase)
                target_t = int(row["target_t"])
                eval_horizon = int(row.get("eval_horizon") or (int(row.get("target_stop", target_t + 1)) - target_t))
                metrics = run_fixed_schedule_variant(
                    model=model,
                    ds=eval_ds,
                    cfg=cfg,
                    eval_horizon=int(eval_horizon),
                    eval_windows=1,
                    grid_spec={
                        "grid_name": GIPO_POLICY_KEY,
                        "grid_kind": "gipo_density_time_grid",
                        "selection_group": GIPO_POLICY_KEY,
                        "comparison_role": "student",
                        "solver_name": normalize_solver_key(solver),
                        "target_nfe": int(target_nfe),
                        "macro_steps": int(macro_steps),
                        "time_grid": prediction["time_grid"],
                    },
                    chosen_t0s=[int(target_t)],
                    generation_seed_base=int(eval_seed),
                    metrics_seed=int(eval_seed),
                    score_main_only=False,
                )
            else:
                member_key, stratum = _molecule_member_from_row(row)
                member = molecule_members.get((member_key, stratum))
                if member is None:
                    raise ValueError(f"Molecule member {member_key}/{stratum} is not present in group manifest for {scenario_key}.")
                checkpoint_step = int(row.get("checkpoint_step", args.checkpoint_step))
                cache_key = (member_key, stratum, int(checkpoint_step))
                if cache_key not in molecule_cache:
                    artifact = find_backbone_artifact(
                        molecule_manifest,
                        backbone_name=BACKBONE_NAME_OTFLOW_MOLECULE,
                        benchmark_family=MOLECULE_FAMILY,
                        dataset_key=scenario_key,
                        train_steps=int(checkpoint_step),
                        member_key=member_key,
                        stratum=stratum,
                    )
                    molecule_cache[cache_key] = load_molecule_checkpoint_splits(
                        checkpoint_path=str(artifact["checkpoint_path"]),
                        dataset_key=scenario_key,
                        stratum=stratum,
                        processed_dir=_molecule_processed_dir(group_root, scenario_key, member),
                        rollout_steps=int(getattr(args, "rollout_steps", 16)),
                        stride_eval=int(getattr(args, "molecule_stride_eval", 1)),
                        device=device,
                    )
                    molecule_cache[cache_key]["checkpoint_id"] = str(artifact["checkpoint_id"])
                loaded = molecule_cache[cache_key]
                mol_splits = loaded["splits"]
                eval_ds = _forecast_dataset_for_source_phase(mol_splits, source_phase)
                metrics = evaluate_molecule_rollout_schedule(
                    model=loaded["model"],
                    ds=eval_ds,
                    cfg=loaded["cfg"],
                    scheduler_key=GIPO_POLICY_KEY,
                    solver_key=solver,
                    target_nfe=int(target_nfe),
                    macro_steps=int(macro_steps),
                    time_grid=prediction["time_grid"],
                    example_indices=[int(example_idx)],
                    sample_count=int(getattr(args, "molecule_sample_count", 1)),
                    rollout_steps=int(getattr(args, "rollout_steps", 16)),
                    seed=int(eval_seed),
                    split_phase=source_phase,
                    checkpoint_id=str(loaded["checkpoint_id"]),
                    dataset_key=scenario_key,
                    member_key=member_key,
                    stratum=stratum,
                    formula=str(member.get("formula", "")),
                    source_zip_name=str(member.get("source_zip_name", "")),
                    device=device,
                )
            selected_row = {
                "benchmark_family": benchmark_family,
                "scenario_key": scenario_key,
                **_report_context_split_fields(
                    source_split_phase=source_phase,
                    report_split=split_phase,
                ),
                "selection_mode": selection_mode,
                "selection_split": _selection_split(row),
                "seed": int(logical_seed),
                "logical_seed": int(logical_seed),
                "evaluation_seed": int(row["evaluation_seed"]),
                "gipo_evaluation_seed": int(eval_seed),
                "checkpoint_step": _checkpoint_step_from_row(row),
                "checkpoint_id": str(row.get("checkpoint_id", "")),
                "checkpoint_maturity_label": str(row.get("checkpoint_maturity_label", "")),
                "checkpoint_maturity_index": row.get("checkpoint_maturity_index", ""),
                "solver_key": solver,
                "target_nfe": int(target_nfe),
                "scheduler_key": GIPO_POLICY_KEY,
                "method_key": student_policy_key,
                "gipo_step_budget": int(gipo_step_budget),
                "teacher_final_retrain": json.dumps(teacher_final_retrain, separators=(",", ":"), sort_keys=True),
                "context_id": context_id,
                "example_idx": int(example_idx),
                "series_id": row.get("series_id", ""),
                "series_idx": row.get("series_idx", ""),
                "member_key": row.get("member_key", row.get("axis_member", "")),
                "stratum": row.get("stratum", row.get("axis_stratum", "")),
                "formula": row.get("formula", row.get("axis_formula", "")),
                "target_t": row.get("target_t", ""),
                "time_grid_json": json.dumps(prediction["time_grid"], separators=(",", ":")),
                "density_mass_hash": prediction["density_mass_hash"],
                "schedule_grid_hash": prediction["schedule_grid_hash"],
                "density_protocol": prediction["density_protocol"],
                "reference_grid_hash": prediction["reference_grid_hash"],
                "mode": prediction["mode"],
                **locked_test_provenance,
            }
            if benchmark_family == FORECAST_FAMILY:
                selected_row.update(
                    {
                        "forecast_crps": float(metrics["forecast_crps"]),
                        "forecast_mase": float(metrics["forecast_mase"]),
                        "forecast_mse": metrics.get("forecast_mse", ""),
                    }
                )
            elif benchmark_family == CONDITIONAL_GENERATION_FAMILY:
                selected_row.update(
                    {
                        "score_main": metrics.get("score_main"),
                        "temporal_uw1": metrics.get("temporal_uw1"),
                        "temporal_cw1": metrics.get("temporal_cw1"),
                        "temporal_tstr_f1": metrics.get("temporal_tstr_f1"),
                        "temporal_tstr_f1_applicable": metrics.get("temporal_tstr_f1_applicable"),
                        "disc_auc": metrics.get("disc_auc"),
                        "disc_auc_gap": metrics.get("disc_auc_gap"),
                        "u_l1": metrics.get("u_l1"),
                        "c_l1": metrics.get("c_l1"),
                        "spread_specific_error": metrics.get("spread_specific_error"),
                        "imbalance_specific_error": metrics.get("imbalance_specific_error"),
                        "ret_vol_acf_error": metrics.get("ret_vol_acf_error"),
                        "impact_response_error": metrics.get("impact_response_error"),
                    }
                )
            else:
                selected_row.update(
                    {
                        "molecule_kabsch_rmsd_3d": metrics.get("molecule_kabsch_rmsd_3d"),
                        "molecule_ensemble_velocity_norm_w1": metrics.get("molecule_ensemble_velocity_norm_w1"),
                        "molecule_ensemble_acceleration_norm_w1": metrics.get("molecule_ensemble_acceleration_norm_w1"),
                        "molecule_rollout_velocity_norm_w1": metrics.get("molecule_rollout_velocity_norm_w1"),
                        "molecule_rollout_acceleration_norm_w1": metrics.get("molecule_rollout_acceleration_norm_w1"),
                        "molecule_coordinate_w1_mean": metrics.get("molecule_coordinate_w1_mean"),
                        "molecule_pair_distance_w1": metrics.get("molecule_pair_distance_w1"),
                    }
                )
            selected_row = _attach_uniform_rewards_to_gipo_row(
                selected_row,
                uniform_row=uniform_context_rows.get(_row_group_key(row)),
                benchmark_family=benchmark_family,
                scenario_key=scenario_key,
            )
            per_context_rows.append(selected_row)
            decision_rows.append(
                {
                    **selected_row,
                    "density_mass_json": json.dumps(prediction["density_mass"], separators=(",", ":")),
                    "macro_steps": int(prediction["macro_steps"]),
                    "policy_source": "frozen_gipo",
                    "locked_test_used_for_selection": False,
                }
            )
            progress.update()

    aggregate_rows = _aggregate_seed_rows(per_context_rows, split_phase=split_phase)
    member_aggregate_rows = (
        _aggregate_molecule_member_rows(per_context_rows, split_phase=split_phase)
        if benchmark_family == SCENARIO_FAMILY_MOLECULE
        else []
    )
    _write_csv(out_dir / f"{output_prefix}_rows.csv", per_context_rows)
    _write_csv(out_dir / f"{output_prefix}_aggregate_rows.csv", aggregate_rows)
    if member_aggregate_rows:
        _write_csv(out_dir / f"{output_prefix}_member_aggregate_rows.csv", member_aggregate_rows)
    _write_csv(out_dir / f"{output_prefix}_decisions.csv", decision_rows)

    comparison_student_rows = per_context_rows if selection_mode == SELECTION_MODE_CALIBRATION else aggregate_rows
    comparison = build_comparison_summary(
        baseline_rows=baseline_rows,
        comparator_rows=comparator_rows,
        student_rows=comparison_student_rows,
        scenario_key=scenario_key,
        benchmark_family=benchmark_family,
        split_phase=split_phase,
        seeds=seeds,
        solver_names=solvers,
        target_nfe_values=target_nfes,
    )
    comparison["method_key"] = student_policy_key
    comparison["gipo_step_budget"] = int(gipo_step_budget)
    comparison["mode"] = mode
    comparison["checkpoint_step"] = int(checkpoint_step)
    comparison["gipo_student_checkpoint_sha256"] = expected_checkpoint_hash
    comparison["report_inputs"] = report_inputs
    comparison["teacher_final_retrain"] = teacher_final_retrain
    comparison.update(locked_test_provenance)
    if strict_locked_comparison:
        missing_baselines = list(comparison.get("missing_baseline_cells", []) or [])
        missing_comparators = list(comparison.get("missing_ser_ptg_cells", []) or [])
        if missing_baselines or missing_comparators:
            raise ValueError(
                "Locked-test comparison is incomplete; "
                f"missing_baseline_cells={missing_baselines[:8]}, "
                f"missing_ser_ptg_cells={missing_comparators[:8]}."
            )
    (out_dir / f"{output_prefix}_comparison_summary.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    crps_values = [float(row["forecast_crps"]) for row in aggregate_rows if row.get("forecast_crps") not in (None, "")]
    mase_values = [float(row["forecast_mase"]) for row in aggregate_rows if row.get("forecast_mase") not in (None, "")]
    score_values = [float(row["score_main"]) for row in aggregate_rows if row.get("score_main") not in (None, "")]
    molecule_kabsch_values = [
        float(row["molecule_kabsch_rmsd_3d"])
        for row in aggregate_rows
        if row.get("molecule_kabsch_rmsd_3d") not in (None, "")
    ]
    molecule_rollout_velocity_values = [
        float(row["molecule_rollout_velocity_norm_w1"])
        for row in aggregate_rows
        if row.get("molecule_rollout_velocity_norm_w1") not in (None, "")
    ]
    artifact_name = _report_artifact_name(
        split_phase=split_phase,
        selection_mode=selection_mode,
        locked_test_mode=str(locked_test_provenance.get("locked_test_mode", "")),
    )
    comparison_file = f"{output_prefix}_comparison_summary.json"
    conditioning_metadata = _conditioning_metadata_for_summary(checkpoint_payload, training_summary)
    summary: Dict[str, Any] = {
        "artifact": artifact_name,
        "protocol": GIPO_PROTOCOL,
        "method_key": student_policy_key,
        "gipo_step_budget": int(gipo_step_budget),
        "checkpoint_step": int(checkpoint_step),
        "gipo_student_checkpoint_sha256": expected_checkpoint_hash,
        "report_inputs": report_inputs,
        "student_policy_type": "continuous_density",
        "scheduler_key": GIPO_POLICY_KEY,
        "benchmark_family": benchmark_family,
        "scenario_key": scenario_key,
        "split_phase": split_phase,
        "selection_mode": selection_mode,
        "source_split_phases": sorted({_source_split_phase(row) for row in representatives}),
        "target_nfe_values": [int(value) for value in target_nfes],
        "context_row_count": int(len(per_context_rows)),
        "aggregate_row_count": int(len(aggregate_rows)),
        "member_aggregate_row_count": int(len(member_aggregate_rows)),
        "missing_expected_cells": missing_cells,
        "missing_cell_count": int(len(missing_cells)),
        "mean_forecast_crps": float(np.mean(np.asarray(crps_values, dtype=np.float64))) if crps_values else None,
        "mean_forecast_mase": float(np.mean(np.asarray(mase_values, dtype=np.float64))) if mase_values else None,
        "mean_score_main": float(np.mean(np.asarray(score_values, dtype=np.float64))) if score_values else None,
        "mean_molecule_kabsch_rmsd_3d": float(np.mean(np.asarray(molecule_kabsch_values, dtype=np.float64))) if molecule_kabsch_values else None,
        "mean_molecule_rollout_velocity_norm_w1": float(np.mean(np.asarray(molecule_rollout_velocity_values, dtype=np.float64))) if molecule_rollout_velocity_values else None,
        "metric_means": _numeric_metric_means(aggregate_rows),
        "density_representation": checkpoint_payload.get("density_representation", {}),
        **conditioning_metadata,
        "mode": mode,
        "setting_encoder_config": checkpoint_payload.get("setting_encoder_config", {}),
        "student_training_mode": training_summary.get("student_training_mode", checkpoint_payload.get("student_training_mode", "")),
        "student_objective_settings": training_summary.get("student_objective_settings", checkpoint_payload.get("student_objective_settings", {})),
        "student_target_summary": training_summary.get("student_target_summary", checkpoint_payload.get("student_target_summary", {})),
        "student_unseen_target_distillation": training_summary.get(
            "student_unseen_target_distillation",
            checkpoint_payload.get("student_unseen_target_distillation", {}),
        ),
        "teacher_checkpoint_selection_mode": checkpoint_payload.get("teacher_checkpoint_selection_mode", training_summary.get("teacher_checkpoint_selection_mode", "")),
        "teacher_final_retrain": teacher_final_retrain,
        "locked_test_used_for_selection": False,
        **locked_test_provenance,
        "comparison_summary_path": "" if comparison is None else comparison_file,
    }
    (out_dir / f"{output_prefix}_policy_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report locked-test performance for a frozen GIPO student.")
    parser.add_argument("--gipo_student_checkpoint", required=True)
    parser.add_argument("--training_summary", required=True)
    parser.add_argument("--context_rows", default="", help="Comma-separated context-row CSVs used to enumerate report contexts.")
    parser.add_argument("--context_embeddings_npz", default="", help="Context embedding table for --context_rows.")
    parser.add_argument("--split_phase", default=LOCKED_TEST_PHASE, help="Report split label: locked_test, validation_tuning, train_tuning, or context_disjoint.")
    parser.add_argument("--selection_mode", choices=(SELECTION_MODE_REPORTING, SELECTION_MODE_CALIBRATION), default=SELECTION_MODE_REPORTING)
    parser.add_argument("--require_teacher_checkpoint_selection_mode", default="")
    parser.add_argument("--report_label", default="")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--baseline_rows", default="", help="Required uniform baseline context-row CSVs used as reward anchors.")
    parser.add_argument("--comparator_rows", default="")
    parser.add_argument(
        "--allow_incomplete_comparison",
        action="store_true",
        help="Allow a partial baseline/SER matrix for exploratory reports. Reference locked-test reports require completeness.",
    )
    parser.add_argument("--benchmark_family", default="")
    parser.add_argument("--scenario_key", default="", help="Scenario key. Defaults to the single scenario inferred from --context_rows.")
    parser.add_argument("--dataset_root", default=str(project_dataset_root()))
    parser.add_argument("--shared_backbone_root", default=str(DEFAULT_SHARED_BACKBONE_ROOT))
    parser.add_argument("--backbone_manifest", default=str(backbone_manifest_path()))
    parser.add_argument("--molecule_group_root", default=str(molecule_group_root()))
    parser.add_argument("--molecule_sample_count", type=int, default=1)
    parser.add_argument("--molecule_stride_eval", type=int, default=1)
    parser.add_argument("--cryptos_path", default="")
    parser.add_argument("--lobster_synthetic_profile_path", default="")
    parser.add_argument("--long_term_st_path", default="")
    parser.add_argument("--checkpoint_step", type=int, default=20000)
    parser.add_argument("--eval_horizon", type=int, default=0)
    parser.add_argument("--future_block_len", type=int, default=0)
    parser.add_argument("--rollout_mode", default="non_ar")
    parser.add_argument("--rollout_steps", type=int, default=16)
    parser.add_argument("--dataset_seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", default="", help="Comma-separated logical seeds. Defaults to seeds inferred from --context_rows.")
    parser.add_argument("--solver_names", default="", help="Comma-separated solver keys. Defaults to solvers inferred from --context_rows.")
    parser.add_argument("--target_nfe_values", default="", help="Comma-separated target NFEs. Defaults to NFEs inferred from --context_rows.")
    parser.add_argument("--num_eval_samples", type=int, default=5)
    parser.add_argument("--forecast_eval_batch_size", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    summary = report_gipo_locked_test(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
