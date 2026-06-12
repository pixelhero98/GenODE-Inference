from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from genode.canonical_experiment_layout import CANONICAL_SEEN_NFES, SCENARIO_FAMILY_MOLECULE
from genode.data.molecule_xyz import default_molecule_group_root, load_molecule_group_manifest, trainable_molecule_group_members
from genode.evaluation.fm_backbone_registry import BACKBONE_NAME_OTFLOW_MOLECULE, MOLECULE_FAMILY, find_backbone_artifact, load_backbone_manifest
from genode.evaluation.molecule_metrics import evaluate_molecule_rollout_schedule, load_molecule_checkpoint_splits
from genode.solver_protocol import CANONICAL_SOLVER_KEYS, normalize_solver_keys
from genode.gipo.objectives import (
    CONDITIONAL_METRIC_SPECS,
    FORECAST_METRIC_SPECS,
    MOLECULE_METRIC_SPECS,
    UNIFORM_SCHEDULE_KEY,
    uniform_anchored_objective_columns,
)
from genode.gipo.policy import (
    ARCHITECTURE_DENSITY_QUERY_TRANSFORMER,
    GIPO_PROTOCOL,
    MODEL_PAYLOAD_VERSION,
    TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET,
    EmbeddingNormalizer,
    build_gipo_student_model,
    context_embedding_id_from_row,
    context_id_from_row,
    load_context_embedding_table,
    normalize_teacher_utility_weights,
    predict_gipo_density_many,
    read_metric_rows_csv,
    validate_gipo_attention_heads,
    validate_canonical_conditioning_style,
    validate_gipo_density_token_attention,
    validate_gipo_teacher_training_metadata,
)
from genode.gipo.evaluate_schedule_summary import build_comparison_summary
from genode.gipo.models import solver_macro_steps
from genode.gipo.models import SETTING_ENCODER_MODE_CONTINUOUS_V3, setting_encoder_config_from_payload, setting_feature_dim, validate_setting_feature_mode
from genode.data.otflow_paths import (
    default_backbone_manifest_path,
    project_paper_dataset_root,
    project_outputs_root,
    resolve_project_path,
)
from genode.evaluation.otflow_evaluation_support import (
    CONDITIONAL_GENERATION_FAMILY,
    DEFAULT_SHARED_BACKBONE_ROOT,
    FORECAST_FAMILY,
    LOCKED_TEST_PHASE,
    SOLVER_RUNTIME_NAMES,
    TRAIN_TUNING_PHASE,
    VALIDATION_PHASE,
    evaluate_forecast_schedule,
    load_conditional_generation_checkpoint_splits,
    load_forecast_checkpoint_splits,
)
from genode.schedule_transfer.diffusion_flow_schedules import run_fixed_schedule_variant
from genode.runtime import ProgressBar, resolve_torch_device

GIPO_SCHEDULE_KEY = "gipo"
REQUIRED_GIPO_DENSITY_BIN_COUNT = 64
SELECTION_MODE_REPORTING = "reporting"
SELECTION_MODE_CALIBRATION = "calibration"
CONTEXT_DISJOINT_PHASE = "context_disjoint"
CALIBRATION_HOLDOUT_PHASES = (CONTEXT_DISJOINT_PHASE,)
SOURCE_SPLIT_PHASES = (TRAIN_TUNING_PHASE, VALIDATION_PHASE, LOCKED_TEST_PHASE)


def _parse_csv(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _parse_int_csv(text: str) -> List[int]:
    return [int(part) for part in _parse_csv(text)]


def _read_csvs(paths_text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path_text in _parse_csv(paths_text):
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
    expected = REQUIRED_GIPO_DENSITY_BIN_COUNT
    density_dim = int(payload.get("density_dim", -1))
    reference_bin_count = int(density_meta.get("reference_bin_count", -1))
    reference_grid = tuple(density_meta.get("reference_time_grid", ()))
    if density_dim != expected or reference_bin_count != expected or len(reference_grid) != expected + 1:
        raise ValueError(
            f"GIPO {role} checkpoints require {expected} density bins; "
            f"got density_dim={density_dim}, reference_bin_count={reference_bin_count}, "
            f"reference_grid_len={len(reference_grid)}."
        )


def _source_split_phase(row: Mapping[str, Any]) -> str:
    return str(row.get("source_split_phase") or row.get("split_phase", row.get("split", ""))).strip()


def _selection_split(row: Mapping[str, Any]) -> str:
    return str(row.get("selection_split") or row.get("report_split") or row.get("split_phase", row.get("split", ""))).strip()


def _validate_context_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split_phase: str,
    selection_mode: str,
) -> None:
    if not rows:
        raise ValueError("context rows input contains no rows.")
    mode = str(selection_mode)
    requested = str(split_phase)
    if mode == SELECTION_MODE_CALIBRATION:
        locked = [
            row
            for row in rows
            if _source_split_phase(row) == LOCKED_TEST_PHASE or str(row.get("split_phase", row.get("split", ""))) == LOCKED_TEST_PHASE
        ]
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
    parent = str(row.get("parent_row_signature", "") or "")
    parts = parent.split("|")
    if len(parts) >= 3:
        try:
            return int(parts[2])
        except (TypeError, ValueError):
            pass
    return int(row["seed"])


def _checkpoint_step_from_row(row: Mapping[str, Any]) -> int:
    raw = row.get("checkpoint_step", row.get("train_steps", row.get("otflow_train_steps", "")))
    if raw in (None, ""):
        return 0
    return int(raw)


def _row_group_key(row: Mapping[str, Any]) -> Tuple[str, str, int, str, int, int, str]:
    return (
        _source_split_phase(row),
        str(row.get("dataset", row.get("dataset_key", ""))),
        _evaluation_seed_from_row(row),
        str(row["solver_key"]),
        int(row["target_nfe"]),
        _checkpoint_step_from_row(row),
        context_id_from_row(row),
    )


def _representative_context_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, int, str, int, str], Dict[str, Any]] = {}
    for row in rows:
        key = _row_group_key(row)
        if key not in grouped:
            copied = dict(row)
            copied["context_id"] = context_id_from_row(row)
            copied["evaluation_seed"] = _evaluation_seed_from_row(row)
            copied["source_split_phase"] = _source_split_phase(row)
            copied["selection_split"] = _selection_split(row)
            grouped[key] = copied
    return [grouped[key] for key in sorted(grouped)]


def _context_match_key(row: Mapping[str, Any]) -> Tuple[str, str, int, str, int, int, str]:
    return (
        _source_split_phase(row),
        str(row.get("dataset", row.get("dataset_key", ""))),
        _evaluation_seed_from_row(row),
        str(row["solver_key"]),
        int(row["target_nfe"]),
        _checkpoint_step_from_row(row),
        context_id_from_row(row),
    )


def _filter_rows_to_contexts(
    rows: Sequence[Mapping[str, Any]],
    representatives: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    wanted = {_context_match_key(row) for row in representatives}
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        try:
            key = _context_match_key(row)
        except (KeyError, TypeError, ValueError):
            return [dict(item) for item in rows]
        if key in wanted:
            filtered.append(dict(row))
    return filtered


def _load_student_checkpoint(
    path: str | Path,
) -> Tuple[Any, Dict[str, int], EmbeddingNormalizer, Tuple[float, ...], Dict[str, Any]]:
    payload = torch.load(resolve_project_path(str(path)), map_location="cpu")
    if str(payload.get("protocol", "")) != GIPO_PROTOCOL:
        raise ValueError(f"Unsupported GIPO student protocol {payload.get('protocol')!r}; expected {GIPO_PROTOCOL!r}.")
    if int(payload.get("model_payload_version", 0)) != MODEL_PAYLOAD_VERSION:
        raise ValueError(
            f"GIPO student checkpoint model_payload_version must be {MODEL_PAYLOAD_VERSION}; "
            f"got {payload.get('model_payload_version')!r}."
        )
    if str(payload.get("student_policy_type", "")) != "continuous_density":
        raise ValueError("GIPO locked reporter only accepts continuous_density student checkpoints.")
    if bool(payload.get("locked_test_used_for_selection", False)):
        raise ValueError("GIPO student checkpoint indicates locked_test was used for selection.")
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
    setting_feature_mode = validate_setting_feature_mode(str(payload.get("setting_feature_mode", SETTING_ENCODER_MODE_CONTINUOUS_V3)))
    setting_encoder_config = setting_encoder_config_from_payload(
        payload.get("setting_encoder_config", {"mode": setting_feature_mode})
    )
    expected_setting_dim = setting_feature_dim(setting_feature_mode, config=setting_encoder_config)
    if int(payload["setting_dim"]) != int(expected_setting_dim):
        raise ValueError(
            f"GIPO student checkpoint setting_dim={payload['setting_dim']} does not match "
            f"setting encoder dim {expected_setting_dim} for {setting_encoder_config.mode}."
        )
    series_index_map = {str(key): int(value) for key, value in dict(payload["series_index_map"]).items()}
    normalizer = EmbeddingNormalizer.from_payload(payload["embedding_normalizer"])
    student_architecture = str(payload.get("student_architecture", ""))
    if student_architecture != ARCHITECTURE_DENSITY_QUERY_TRANSFORMER:
        raise ValueError(
            f"GIPO student checkpoints must use {ARCHITECTURE_DENSITY_QUERY_TRANSFORMER!r}; got {student_architecture!r}."
        )
    student_model_config = dict(payload.get("student_model_config", {}) or {})
    validate_canonical_conditioning_style(
        student_model_config,
        require_present=True,
    )
    validate_gipo_density_token_attention(student_model_config, require_present=True)
    validate_gipo_attention_heads(int(student_model_config.get("attention_heads", -1)))
    student = build_gipo_student_model(
        architecture=student_architecture,
        setting_dim=int(payload["setting_dim"]),
        density_dim=int(payload["density_dim"]),
        context_dim=int(payload["context_dim"]),
        num_series=len(series_index_map),
        model_config=student_model_config,
    )
    student.load_state_dict(payload["student_state"])
    student.eval()
    payload["setting_feature_mode"] = setting_feature_mode
    payload["setting_encoder_mode"] = setting_encoder_config.mode
    payload["setting_encoder_config"] = setting_encoder_config.to_payload()
    payload["student_architecture"] = student_architecture
    payload["student_model_config"] = student.model_config()
    return student, series_index_map, normalizer, reference_time_grid, payload


def _teacher_final_retrain_metadata(
    checkpoint_payload: Mapping[str, Any],
    training_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    return dict(
        checkpoint_payload.get("teacher_final_retrain")
        or checkpoint_payload.get("final_teacher_retrain")
        or training_summary.get("teacher_final_retrain")
        or training_summary.get("final_teacher_retrain")
        or {}
    )


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
        validate_canonical_conditioning_style({"conditioning_style": style})
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
                str(row.get("dataset", row.get("dataset_key", ""))),
                int(row["seed"]),
                str(row["solver_key"]),
                int(row["target_nfe"]),
                _checkpoint_step_from_row(row),
                str(row["scheduler_key"]),
            )
        ].append(row)
    out: List[Dict[str, Any]] = []
    for (dataset, seed, solver, target_nfe, checkpoint_step, scheduler_key), group in sorted(grouped.items()):
        row: Dict[str, Any] = {
            "dataset": dataset,
            "split_phase": str(split_phase),
            "seed": int(seed),
            "solver_key": solver,
            "target_nfe": int(target_nfe),
            "checkpoint_step": int(checkpoint_step),
            "checkpoint_id": str(group[0].get("checkpoint_id", "")),
            "checkpoint_maturity_label": str(group[0].get("checkpoint_maturity_label", "")),
            "checkpoint_maturity_index": group[0].get("checkpoint_maturity_index", ""),
            "scheduler_key": scheduler_key,
            "context_count": int(len(group)),
        }
        for metric_key in (
            "crps",
            "mase",
            "mse",
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
    families = {str(row.get("benchmark_family", "")).strip() for row in rows if str(row.get("benchmark_family", "")).strip()}
    if requested:
        if families and families != {requested}:
            raise ValueError(f"context rows benchmark_family mismatch: requested {requested!r}, found {sorted(families)}.")
        return requested
    if not families:
        has_generic_context_schema = any(
            str(row.get("context_schema", "") or "").strip()
            or any(str(row.get(key, "") or "").strip() for key in row if str(key).startswith("axis_"))
            for row in rows
        )
        if has_generic_context_schema:
            raise ValueError("Generic GIPO context rows require benchmark_family for locked-test reporting.")
        return FORECAST_FAMILY
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


def _molecule_group_member_lookup(dataset: str, group_root: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    manifest = load_molecule_group_manifest(str(dataset), group_root)
    return {
        (str(member["member_key"]), str(member["stratum"])): dict(member)
        for member in trainable_molecule_group_members(manifest)
    }


def _molecule_processed_dir(group_root: Path, dataset: str, member: Mapping[str, Any]) -> Path:
    return group_root / str(dataset) / str(member["processed_dir"])


def _metric_specs_for_family(benchmark_family: str):
    if str(benchmark_family) == FORECAST_FAMILY:
        return FORECAST_METRIC_SPECS
    if str(benchmark_family) == CONDITIONAL_GENERATION_FAMILY:
        return CONDITIONAL_METRIC_SPECS
    if str(benchmark_family) == SCENARIO_FAMILY_MOLECULE:
        return MOLECULE_METRIC_SPECS
    raise ValueError(f"Unsupported benchmark_family={benchmark_family!r}.")


def _attach_uniform_rewards_to_gipo_row(
    row: Mapping[str, Any],
    *,
    uniform_row: Mapping[str, Any] | None,
    benchmark_family: str,
) -> Dict[str, Any]:
    out = dict(row)
    if uniform_row is None:
        raise ValueError(
            "Locked-test GIPO reporting requires a matched uniform row for every "
            f"context/solver/NFE/checkpoint cell; missing context_id={context_id_from_row(row)!r} "
            f"solver={row.get('solver_key')!r} target_nfe={row.get('target_nfe')!r}."
        )
    reward_columns = uniform_anchored_objective_columns(
        {**out, "scheduler_key": GIPO_SCHEDULE_KEY},
        {**dict(uniform_row), "scheduler_key": UNIFORM_SCHEDULE_KEY},
        _metric_specs_for_family(benchmark_family),
        uniform_schedule_key=UNIFORM_SCHEDULE_KEY,
    )
    out.update(reward_columns)
    out["gipo_reward_protocol"] = GIPO_PROTOCOL
    out["reward_anchor_schedule_key"] = UNIFORM_SCHEDULE_KEY
    out["reward_utility_transform"] = "directional_log_uniform_anchor"
    out["reward_granularity"] = "context_window_metric_components"
    return out


def report_gipo_locked_test(args: argparse.Namespace) -> Dict[str, Any]:
    student, series_index_map, normalizer, reference_time_grid, checkpoint_payload = _load_student_checkpoint(
        str(args.gipo_student_checkpoint),
    )
    training_summary = json.loads(resolve_project_path(str(args.training_summary)).read_text(encoding="utf-8"))
    if str(training_summary.get("protocol", "")) != GIPO_PROTOCOL:
        raise ValueError("training_summary protocol does not match continuous-density GIPO.")
    if bool(training_summary.get("locked_test_used_for_selection", False)):
        raise ValueError("training_summary indicates locked_test was used for selection.")
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
            final_retrain = _teacher_final_retrain_metadata(checkpoint_payload, training_summary)
            if final_retrain.get("enabled") is not True:
                raise ValueError("GIPO reporter requires final teacher retrain metadata for weighted-normalized-regret checkpoints.")

    selection_mode = str(getattr(args, "selection_mode", SELECTION_MODE_REPORTING))
    split_phase = str(getattr(args, "split_phase", LOCKED_TEST_PHASE))
    context_rows_arg = str(getattr(args, "context_rows", ""))
    embeddings_arg = str(getattr(args, "context_embeddings_npz", ""))
    if not context_rows_arg.strip():
        raise ValueError("GIPO reporter requires --context_rows.")
    if not embeddings_arg.strip():
        raise ValueError("GIPO reporter requires --context_embeddings_npz.")
    context_rows = _read_csvs(context_rows_arg)
    _validate_context_rows(context_rows, split_phase=split_phase, selection_mode=selection_mode)
    _benchmark_family_from_rows(context_rows, str(getattr(args, "benchmark_family", "") or ""))
    representatives = _representative_context_rows(context_rows)
    dataset = str(args.dataset)
    benchmark_family = _benchmark_family_from_rows(representatives, str(getattr(args, "benchmark_family", "") or ""))
    seeds = tuple(_parse_int_csv(str(args.seeds)))
    solvers = tuple(normalize_solver_keys(str(args.solver_names)))
    target_nfes = tuple(_parse_int_csv(str(args.target_nfe_values)))
    molecule_group_root = resolve_project_path(str(getattr(args, "molecule_group_root", "") or default_molecule_group_root()))
    molecule_members_for_coverage: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if benchmark_family == SCENARIO_FAMILY_MOLECULE:
        molecule_members_for_coverage = _molecule_group_member_lookup(dataset, molecule_group_root)
        expected_cells = {
            (dataset, member_key, stratum, seed, solver, target_nfe)
            for member_key, stratum in sorted(molecule_members_for_coverage)
            for seed in seeds
            for solver in solvers
            for target_nfe in target_nfes
        }
        observed_cells = {
            (
                str(row.get("dataset", row.get("dataset_key", ""))),
                str(row.get("member_key") or row.get("axis_member") or ""),
                str(row.get("stratum") or row.get("axis_stratum") or ""),
                int(row["evaluation_seed"]),
                str(row["solver_key"]),
                int(row["target_nfe"]),
            )
            for row in representatives
        }
    else:
        expected_cells = {(dataset, seed, solver, target_nfe) for seed in seeds for solver in solvers for target_nfe in target_nfes}
        observed_cells = {
            (str(row.get("dataset", row.get("dataset_key", ""))), int(row["evaluation_seed"]), str(row["solver_key"]), int(row["target_nfe"]))
            for row in representatives
        }
    missing_cells = sorted(expected_cells - observed_cells)
    if missing_cells:
        raise ValueError(f"context rows are missing seed/solver/NFE cells: {missing_cells[:8]}")

    baseline_rows = _read_csvs(str(args.baseline_rows)) if str(args.baseline_rows).strip() else []
    comparator_rows = _read_csvs(str(args.comparator_rows)) if str(args.comparator_rows).strip() else []
    if baseline_rows:
        _benchmark_family_from_rows(baseline_rows, benchmark_family)
    if comparator_rows:
        _benchmark_family_from_rows(comparator_rows, benchmark_family)
    if selection_mode == SELECTION_MODE_CALIBRATION:
        baseline_rows = _filter_rows_to_contexts(baseline_rows, representatives)
        comparator_rows = _filter_rows_to_contexts(comparator_rows, representatives)
    uniform_context_rows = {
        _context_match_key(row): dict(row)
        for row in baseline_rows
        if str(row.get("scheduler_key", "")) == UNIFORM_SCHEDULE_KEY
    }

    raw_embeddings = load_context_embedding_table(resolve_project_path(embeddings_arg))
    missing_context_embeddings = sorted({context_embedding_id_from_row(row) for row in representatives} - set(raw_embeddings))
    if missing_context_embeddings:
        raise KeyError(f"context embeddings NPZ is missing contexts: {missing_context_embeddings[:8]}")
    embeddings = normalizer.transform_table(raw_embeddings)

    device = resolve_torch_device(str(args.device))
    student.to(device)
    if benchmark_family == FORECAST_FAMILY:
        checkpoint = load_forecast_checkpoint_splits(
            cli_args=args,
            dataset_root=resolve_project_path(str(args.dataset_root)),
            shared_backbone_root=resolve_project_path(str(args.shared_backbone_root)),
            dataset=dataset,
            device=device,
        )
    elif benchmark_family == CONDITIONAL_GENERATION_FAMILY:
        if not hasattr(args, "steps") or int(getattr(args, "steps", 0) or 0) <= 0:
            args.steps = int(getattr(args, "otflow_train_steps", 0) or 0)
        checkpoint = load_conditional_generation_checkpoint_splits(
            cli_args=args,
            shared_backbone_root=resolve_project_path(str(args.shared_backbone_root)),
            dataset=dataset,
            device=device,
        )
    elif benchmark_family == SCENARIO_FAMILY_MOLECULE:
        checkpoint = None
        molecule_manifest = load_backbone_manifest(resolve_project_path(str(args.backbone_manifest)))
        molecule_members = molecule_members_for_coverage or _molecule_group_member_lookup(dataset, molecule_group_root)
        molecule_cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    else:
        raise NotImplementedError(f"GIPO locked-test reporting does not support benchmark_family={benchmark_family!r}.")
    if checkpoint is not None:
        model = checkpoint["model"]
        cfg = checkpoint["cfg"]
        splits = checkpoint["splits"]
    else:
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
        series_index_map=series_index_map,
        reference_time_grid=reference_time_grid,
        setting_feature_mode=str(checkpoint_payload.get("setting_feature_mode", SETTING_ENCODER_MODE_CONTINUOUS_V3)),
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
            eval_seed = int(row["evaluation_seed"]) + 10_000 * int(row_idx)
            runtime_nfe = int(solver_macro_steps(solver, target_nfe))
            if benchmark_family == FORECAST_FAMILY:
                eval_ds = _forecast_dataset_for_source_phase(splits, source_phase)
                metrics = evaluate_forecast_schedule(
                    model,
                    eval_ds,
                    cfg,
                    solver_name=str(SOLVER_RUNTIME_NAMES[solver]),
                    runtime_nfe=int(runtime_nfe),
                    target_nfe=int(target_nfe),
                    time_grid=prediction["time_grid"],
                    num_eval_samples=int(args.num_eval_samples),
                    seed=int(eval_seed),
                    scheduler_key=GIPO_SCHEDULE_KEY,
                    dataset_key=dataset,
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
                        "grid_name": GIPO_SCHEDULE_KEY,
                        "grid_kind": "gipo_density_time_grid",
                        "selection_group": GIPO_SCHEDULE_KEY,
                        "comparison_role": "student",
                        "solver_name": str(SOLVER_RUNTIME_NAMES[solver]),
                        "nfe": int(runtime_nfe),
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
                    raise ValueError(f"Molecule member {member_key}/{stratum} is not present in group manifest for {dataset}.")
                checkpoint_step = int(row.get("checkpoint_step", getattr(args, "steps", 0) or getattr(args, "otflow_train_steps", 0) or 0))
                cache_key = (member_key, stratum, int(checkpoint_step))
                if cache_key not in molecule_cache:
                    artifact = find_backbone_artifact(
                        molecule_manifest,
                        backbone_name=BACKBONE_NAME_OTFLOW_MOLECULE,
                        benchmark_family=MOLECULE_FAMILY,
                        dataset_key=dataset,
                        train_steps=int(checkpoint_step),
                        member_key=member_key,
                        stratum=stratum,
                    )
                    molecule_cache[cache_key] = load_molecule_checkpoint_splits(
                        checkpoint_path=str(artifact["checkpoint_path"]),
                        dataset_key=dataset,
                        stratum=stratum,
                        processed_dir=_molecule_processed_dir(molecule_group_root, dataset, member),
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
                    scheduler_key=GIPO_SCHEDULE_KEY,
                    solver_key=solver,
                    target_nfe=int(target_nfe),
                    runtime_nfe=int(runtime_nfe),
                    time_grid=prediction["time_grid"],
                    example_indices=[int(example_idx)],
                    sample_count=int(getattr(args, "molecule_sample_count", 1)),
                    rollout_steps=int(getattr(args, "rollout_steps", 16)),
                    seed=int(eval_seed),
                    split_phase=source_phase,
                    checkpoint_id=str(loaded["checkpoint_id"]),
                    dataset_key=dataset,
                    member_key=member_key,
                    stratum=stratum,
                    formula=str(member.get("formula", "")),
                    source_zip_name=str(member.get("source_zip_name", "")),
                    device=device,
                )
            selected_row = {
                "benchmark_family": benchmark_family,
                "dataset": dataset,
                "split_phase": str(split_phase),
                "source_split_phase": source_phase,
                "selection_mode": selection_mode,
                "selection_split": _selection_split(row),
                "seed": int(row["evaluation_seed"]),
                "checkpoint_step": _checkpoint_step_from_row(row),
                "checkpoint_id": str(row.get("checkpoint_id", "")),
                "checkpoint_maturity_label": str(row.get("checkpoint_maturity_label", "")),
                "checkpoint_maturity_index": row.get("checkpoint_maturity_index", ""),
                "solver_key": solver,
                "target_nfe": int(target_nfe),
                "scheduler_key": GIPO_SCHEDULE_KEY,
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
                "setting_feature_mode": prediction["setting_feature_mode"],
                "setting_encoder_mode": prediction["setting_encoder_mode"],
            }
            if benchmark_family == FORECAST_FAMILY:
                selected_row.update(
                    {
                        "crps": float(metrics["forecast_crps"]),
                        "mase": float(metrics["forecast_mase"]),
                        "mse": metrics.get("forecast_mse", ""),
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
                uniform_row=uniform_context_rows.get(_context_match_key(row)),
                benchmark_family=benchmark_family,
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

    comparison = None
    if baseline_rows:
        comparison_student_rows = per_context_rows if selection_mode == SELECTION_MODE_CALIBRATION else aggregate_rows
        comparison = build_comparison_summary(
            baseline_rows=baseline_rows,
            comparator_rows=comparator_rows,
            student_rows=comparison_student_rows,
            dataset=dataset,
            benchmark_family=benchmark_family,
            split_phase=split_phase,
            seeds=seeds,
            solver_names=solvers,
            target_nfe_values=target_nfes,
        )
        (out_dir / f"{output_prefix}_comparison_summary.json").write_text(
            json.dumps(comparison, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    crps_values = [float(row["crps"]) for row in aggregate_rows if row.get("crps") not in (None, "")]
    mase_values = [float(row["mase"]) for row in aggregate_rows if row.get("mase") not in (None, "")]
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
    artifact_name = (
        "gipo_locked_test_report"
        if split_phase == LOCKED_TEST_PHASE and selection_mode == SELECTION_MODE_REPORTING
        else "gipo_student_calibration_report"
    )
    comparison_file = f"{output_prefix}_comparison_summary.json"
    conditioning_metadata = _conditioning_metadata_for_summary(checkpoint_payload, training_summary)
    summary: Dict[str, Any] = {
        "artifact": artifact_name,
        "protocol": GIPO_PROTOCOL,
        "student_policy_type": "continuous_density",
        "scheduler_key": GIPO_SCHEDULE_KEY,
        "benchmark_family": benchmark_family,
        "dataset": dataset,
        "split_phase": split_phase,
        "selection_mode": selection_mode,
        "source_split_phases": sorted({_source_split_phase(row) for row in representatives}),
        "target_nfe_values": [int(value) for value in target_nfes],
        "context_row_count": int(len(per_context_rows)),
        "aggregate_row_count": int(len(aggregate_rows)),
        "member_aggregate_row_count": int(len(member_aggregate_rows)),
        "missing_expected_cells": missing_cells,
        "missing_cell_count": int(len(missing_cells)),
        "mean_crps": float(np.mean(np.asarray(crps_values, dtype=np.float64))) if crps_values else None,
        "mean_mase": float(np.mean(np.asarray(mase_values, dtype=np.float64))) if mase_values else None,
        "mean_score_main": float(np.mean(np.asarray(score_values, dtype=np.float64))) if score_values else None,
        "mean_molecule_kabsch_rmsd_3d": float(np.mean(np.asarray(molecule_kabsch_values, dtype=np.float64))) if molecule_kabsch_values else None,
        "mean_molecule_rollout_velocity_norm_w1": float(np.mean(np.asarray(molecule_rollout_velocity_values, dtype=np.float64))) if molecule_rollout_velocity_values else None,
        "metric_means": _numeric_metric_means(aggregate_rows),
        "density_representation": checkpoint_payload.get("density_representation", {}),
        **conditioning_metadata,
        "setting_feature_mode": checkpoint_payload.get("setting_feature_mode", SETTING_ENCODER_MODE_CONTINUOUS_V3),
        "setting_encoder_mode": checkpoint_payload.get("setting_encoder_mode", ""),
        "setting_encoder_config": checkpoint_payload.get("setting_encoder_config", {}),
        "teacher_checkpoint_selection_mode": checkpoint_payload.get("teacher_checkpoint_selection_mode", training_summary.get("teacher_checkpoint_selection_mode", "")),
        "teacher_final_retrain": _teacher_final_retrain_metadata(checkpoint_payload, training_summary),
        "locked_test_used_for_selection": False,
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
    parser.add_argument("--baseline_rows", default="")
    parser.add_argument("--comparator_rows", default="")
    parser.add_argument("--benchmark_family", default="")
    parser.add_argument("--dataset", default="solar_energy_10m")
    parser.add_argument("--dataset_root", default=str(project_paper_dataset_root()))
    parser.add_argument("--shared_backbone_root", default=str(DEFAULT_SHARED_BACKBONE_ROOT))
    parser.add_argument("--backbone_manifest", default=str(default_backbone_manifest_path()))
    parser.add_argument("--molecule_group_root", default=str(default_molecule_group_root()))
    parser.add_argument("--molecule_sample_count", type=int, default=1)
    parser.add_argument("--molecule_stride_eval", type=int, default=1)
    parser.add_argument("--output_root", default=str(project_outputs_root()))
    parser.add_argument("--cryptos_path", default="")
    parser.add_argument("--lobster_synthetic_profile_path", default="")
    parser.add_argument("--long_term_st_path", default="")
    parser.add_argument("--otflow_train_steps", type=int, default=20000)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--eval_horizon", type=int, default=0)
    parser.add_argument("--future_block_len", type=int, default=0)
    parser.add_argument("--rollout_mode", default="non_ar")
    parser.add_argument("--rollout_steps", type=int, default=16)
    parser.add_argument("--dataset_seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--hidden_dim", type=int, default=160)
    parser.add_argument("--fu_net_layers", type=int, default=3)
    parser.add_argument("--fu_net_heads", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--solver_names", default=",".join(CANONICAL_SOLVER_KEYS))
    parser.add_argument("--target_nfe_values", default=",".join(str(value) for value in CANONICAL_SEEN_NFES))
    parser.add_argument("--num_eval_samples", type=int, default=5)
    parser.add_argument("--forecast_eval_batch_size", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    summary = report_gipo_locked_test(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
